#!/usr/bin/env python3
"""Create comparison tables and plots from NC convergence experiment outputs."""

import argparse
import csv
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def configure_matplotlib_cache() -> None:
    if "MPLCONFIGDIR" in os.environ:
        return
    default_cache = Path.home() / ".matplotlib"
    if default_cache.exists() and os.access(default_cache, os.W_OK):
        return
    cache_dir = Path(tempfile.gettempdir()) / "fedgraph_matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


configure_matplotlib_cache()

import matplotlib.pyplot as plt
import numpy as np

ROUND_TIMING_RE = re.compile(
    r"Round\s+(\d+):\s+Training Time\s*=\s*([0-9.eE+-]+)s,\s+"
    r"Communication Time\s*=\s*([0-9.eE+-]+)s"
)


def parse_float_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def batch_label(batch_size: int) -> str:
    return "Full batch" if batch_size == -1 else f"Batch {batch_size}"


def sort_batch_sizes(batch_sizes: Iterable[int]) -> List[int]:
    return sorted(batch_sizes, key=lambda value: (value == -1, value))


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def write_rows(path: Path, rows: List[dict]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def optional_float(value: Optional[str]) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def numeric_values(rows: List[dict], key: str) -> List[float]:
    values = []
    for row in rows:
        value = optional_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def mean_or_none(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def std_or_zero(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def true_rate(rows: List[dict], key: str) -> Optional[float]:
    values = [
        str(row.get(key)).lower() == "true"
        for row in rows
        if row.get(key) not in (None, "")
    ]
    return float(np.mean(values)) if values else None


def load_run_configs(results_dir: Path) -> Dict[str, dict]:
    configs = {}
    for config_path in (results_dir / "runs").glob("*/config.json"):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        configs[config_path.parent.name] = config
    return configs


def load_round_times(results_dir: Path) -> Dict[str, Dict[int, float]]:
    """Read cumulative training+communication time from per-run stdout logs."""
    run_times = {}
    for log_path in (results_dir / "runs").glob("*/stdout.log"):
        cumulative = 0.0
        times = {}
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for round_id, training_time, communication_time in ROUND_TIMING_RE.findall(
            text
        ):
            cumulative += float(training_time) + float(communication_time)
            times[int(round_id)] = cumulative
        run_times[log_path.parent.name] = times
    return run_times


def prepare_curves(results_dir: Path) -> List[dict]:
    rows = read_csv(results_dir / "accuracy_curves.csv")
    configs = load_run_configs(results_dir)
    fallback_times = load_round_times(results_dir)
    round_metrics_path = results_dir / "round_metrics.csv"
    round_metrics = {}
    if round_metrics_path.exists():
        round_metrics = {
            (row["run_id"], int(row["round"])): row
            for row in read_csv(round_metrics_path)
        }

    prepared = []
    for row in rows:
        run_id = row["run_id"]
        round_id = int(row["round"])
        config = configs.get(run_id, {})
        metrics = round_metrics.get((run_id, round_id), {})
        cumulative_time = optional_float(
            metrics.get("cumulative_training_communication_time_sec")
        )
        if cumulative_time is None:
            cumulative_time = fallback_times.get(run_id, {}).get(round_id)

        local_step = int(config.get("local_step", 0))
        prepared.append(
            {
                "run_id": run_id,
                "batch_size": int(row["batch_size"]),
                "seed": int(row["seed"]),
                "round": round_id,
                "accuracy": float(row["accuracy"]),
                "cumulative_time_sec": cumulative_time,
                "local_steps_per_trainer": int(
                    metrics.get("local_steps_per_trainer") or round_id * local_step
                ),
            }
        )
    return prepared


def group_runs(rows: Iterable[dict]) -> Dict[str, List[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["run_id"]].append(row)
    for run_rows in grouped.values():
        run_rows.sort(key=lambda item: item["round"])
    return grouped


def sustained_convergence_round(curve: List[dict], tolerance: float) -> Optional[int]:
    if not curve:
        return None
    threshold = max(item["accuracy"] for item in curve) - tolerance
    suffix_minimum = float("inf")
    first_round = None
    for item in reversed(curve):
        suffix_minimum = min(suffix_minimum, item["accuracy"])
        if suffix_minimum >= threshold:
            first_round = item["round"]
        else:
            break
    return first_round


def write_convergence_summary(
    grouped_runs: Dict[str, List[dict]],
    window: int,
    tolerance: float,
    output_path: Path,
) -> None:
    rows = []
    for run_id, curve in grouped_runs.items():
        values = [item["accuracy"] for item in curve]
        best_item = max(curve, key=lambda item: item["accuracy"])
        end_values = values[-window:] if len(values) >= window else []
        end_range = max(end_values) - min(end_values) if end_values else None
        end_gain = end_values[-1] - end_values[0] if end_values else None
        first_near_best = next(
            (
                item["round"]
                for item in curve
                if item["accuracy"] >= best_item["accuracy"] - tolerance
            ),
            None,
        )
        rows.append(
            {
                "run_id": run_id,
                "batch_size": curve[0]["batch_size"],
                "seed": curve[0]["seed"],
                "rounds_recorded": len(curve),
                "final_accuracy": curve[-1]["accuracy"],
                "best_accuracy": best_item["accuracy"],
                "best_round": best_item["round"],
                "first_round_within_best_tolerance": first_near_best,
                "sustained_convergence_round": sustained_convergence_round(
                    curve, tolerance
                ),
                "end_window": window,
                "end_window_gain": end_gain,
                "end_window_range": end_range,
                "converged_at_end": (
                    end_range <= tolerance if end_range is not None else False
                ),
            }
        )

    fieldnames = list(rows[0]) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(
    grouped_runs: Dict[str, List[dict]],
    x_key: str,
    x_label: str,
    output_path: Path,
    aggregate: bool = False,
) -> bool:
    available = [
        rows
        for rows in grouped_runs.values()
        if rows and rows[0].get(x_key) is not None
    ]
    if not available:
        return False

    plt.figure(figsize=(8, 5))
    if aggregate:
        by_batch_and_x = defaultdict(lambda: defaultdict(list))
        for rows in available:
            batch_size = rows[0]["batch_size"]
            for row in rows:
                x_value = row.get(x_key)
                if x_value is not None:
                    by_batch_and_x[batch_size][x_value].append(row["accuracy"])

        for batch_size in sort_batch_sizes(by_batch_and_x):
            points = sorted(by_batch_and_x[batch_size].items())
            x_values = [point[0] for point in points]
            means = [100 * np.mean(point[1]) for point in points]
            stds = [
                100 * np.std(point[1], ddof=1) if len(point[1]) > 1 else 0
                for point in points
            ]
            plt.plot(
                x_values,
                means,
                label=batch_label(batch_size),
                linewidth=2,
                alpha=0.9,
            )
            if any(stds):
                lower = [mean - std for mean, std in zip(means, stds)]
                upper = [mean + std for mean, std in zip(means, stds)]
                plt.fill_between(x_values, lower, upper, alpha=0.12)
    else:
        seen_labels = set()
        for rows in sorted(
            available,
            key=lambda item: (item[0]["batch_size"] == -1, item[0]["batch_size"]),
        ):
            batch_size = rows[0]["batch_size"]
            label = batch_label(batch_size)
            display_label = label if label not in seen_labels else None
            seen_labels.add(label)
            plt.plot(
                [row[x_key] for row in rows],
                [100 * row["accuracy"] for row in rows],
                label=display_label,
                linewidth=2,
                alpha=0.85,
            )

    plt.xlabel(x_label)
    plt.ylabel("Global test accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return True


def plot_individual_curves(
    grouped_runs: Dict[str, List[dict]],
    x_key: str,
    x_label: str,
    output_path: Path,
) -> bool:
    available = [
        rows
        for rows in grouped_runs.values()
        if rows and rows[0].get(x_key) is not None
    ]
    if not available:
        return False

    plt.figure(figsize=(8, 5))
    seen_labels = set()
    for rows in sorted(
        available, key=lambda item: (item[0]["batch_size"] == -1, item[0]["batch_size"])
    ):
        batch_size = rows[0]["batch_size"]
        label = batch_label(batch_size)
        display_label = label if label not in seen_labels else None
        seen_labels.add(label)
        plt.plot(
            [row[x_key] for row in rows],
            [100 * row["accuracy"] for row in rows],
            label=display_label,
            linewidth=2,
            alpha=0.85,
        )

    plt.xlabel(x_label)
    plt.ylabel("Global test accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return True


def write_batch_summary(summary_rows: List[dict], output_path: Path) -> List[dict]:
    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[int(row["batch_size"])].append(row)

    rows = []
    for batch_size in sort_batch_sizes(grouped):
        batch_rows = grouped[batch_size]
        completed = [row for row in batch_rows if row.get("status") == "completed"]
        final_accuracy = numeric_values(completed, "final_accuracy")
        best_accuracy = numeric_values(completed, "best_accuracy")
        duration = numeric_values(completed, "duration_sec")
        pure_training = numeric_values(completed, "pure_training_time_sec")
        communication = numeric_values(completed, "communication_time_sec")
        training_communication = numeric_values(
            completed, "training_communication_time_sec"
        )
        sustained_round = numeric_values(completed, "sustained_convergence_round")
        seeds = sorted({row.get("seed", "") for row in batch_rows})
        rows.append(
            {
                "batch_size": batch_size,
                "batch_label": batch_label(batch_size),
                "runs": len(batch_rows),
                "completed_runs": len(completed),
                "failed_runs": len(batch_rows) - len(completed),
                "seeds": ",".join(seeds),
                "final_accuracy_mean": mean_or_none(final_accuracy),
                "final_accuracy_std": std_or_zero(final_accuracy),
                "best_accuracy_mean": mean_or_none(best_accuracy),
                "best_accuracy_std": std_or_zero(best_accuracy),
                "duration_sec_mean": mean_or_none(duration),
                "pure_training_time_sec_mean": mean_or_none(pure_training),
                "communication_time_sec_mean": mean_or_none(communication),
                "training_communication_time_sec_mean": mean_or_none(
                    training_communication
                ),
                "sustained_convergence_round_mean": mean_or_none(sustained_round),
                "converged_at_end_rate": true_rate(completed, "converged_at_end"),
            }
        )

    full_batch = next((row for row in rows if row["batch_size"] == -1), None)
    baseline = full_batch["final_accuracy_mean"] if full_batch else None
    for row in rows:
        final_mean = row["final_accuracy_mean"]
        row["final_accuracy_delta_vs_full_batch"] = (
            final_mean - baseline
            if final_mean is not None and baseline is not None
            else None
        )

    write_rows(output_path, rows)
    return rows


def write_target_metrics(
    grouped_runs: Dict[str, List[dict]], targets: List[float], output_path: Path
) -> List[dict]:
    rows = []
    for run_id, curve in grouped_runs.items():
        for target in targets:
            reached = next((item for item in curve if item["accuracy"] >= target), None)
            rows.append(
                {
                    "run_id": run_id,
                    "batch_size": curve[0]["batch_size"],
                    "seed": curve[0]["seed"],
                    "target_accuracy": target,
                    "round": reached["round"] if reached else None,
                    "local_steps_per_trainer": (
                        reached["local_steps_per_trainer"] if reached else None
                    ),
                    "training_communication_time_sec": (
                        reached["cumulative_time_sec"] if reached else None
                    ),
                }
            )

    write_rows(output_path, rows)
    return rows


def write_target_summary(rows: List[dict], output_path: Path) -> List[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["batch_size"]), float(row["target_accuracy"]))].append(row)

    summary_rows = []
    for (batch_size, target), target_rows in sorted(
        grouped.items(), key=lambda item: (item[0][0] == -1, item[0][0], item[0][1])
    ):
        reached = [
            row
            for row in target_rows
            if optional_float(row["training_communication_time_sec"]) is not None
        ]
        rounds = numeric_values(reached, "round")
        local_steps = numeric_values(reached, "local_steps_per_trainer")
        times = numeric_values(reached, "training_communication_time_sec")
        summary_rows.append(
            {
                "batch_size": batch_size,
                "batch_label": batch_label(batch_size),
                "target_accuracy": target,
                "target_accuracy_percent": 100 * target,
                "runs": len(target_rows),
                "reached_runs": len(reached),
                "reach_rate": len(reached) / len(target_rows) if target_rows else None,
                "round_mean": mean_or_none(rounds),
                "round_std": std_or_zero(rounds),
                "local_steps_per_trainer_mean": mean_or_none(local_steps),
                "training_communication_time_sec_mean": mean_or_none(times),
                "training_communication_time_sec_std": std_or_zero(times),
            }
        )

    write_rows(output_path, summary_rows)
    return summary_rows


def plot_time_to_target(rows: List[dict], output_path: Path) -> bool:
    reached = [
        row for row in rows if row["training_communication_time_sec"] not in (None, "")
    ]
    if not reached:
        return False

    grouped = defaultdict(list)
    for row in reached:
        grouped[int(row["batch_size"])].append(row)

    plt.figure(figsize=(8, 5))
    for batch_size in sort_batch_sizes(grouped):
        batch_rows = grouped[batch_size]
        by_target = defaultdict(list)
        for row in batch_rows:
            by_target[float(row["target_accuracy"])].append(
                float(row["training_communication_time_sec"])
            )
        targets = sorted(by_target)
        means = [np.mean(by_target[target]) for target in targets]
        plt.plot(
            [100 * target for target in targets],
            means,
            marker="o",
            linewidth=2,
            label=batch_label(batch_size),
        )

    plt.xlabel("Target test accuracy (%)")
    plt.ylabel("Training + communication time (seconds)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return True


def plot_final_accuracy(summary_rows: List[dict], output_path: Path) -> None:
    grouped = defaultdict(list)
    for row in summary_rows:
        if row.get("status") == "completed" and row.get("final_accuracy"):
            grouped[int(row["batch_size"])].append(float(row["final_accuracy"]))

    batch_sizes = sort_batch_sizes(grouped)
    means = [100 * np.mean(grouped[batch_size]) for batch_size in batch_sizes]
    errors = [
        100 * np.std(grouped[batch_size], ddof=1) if len(grouped[batch_size]) > 1 else 0
        for batch_size in batch_sizes
    ]

    plt.figure(figsize=(7, 5))
    positions = np.arange(len(batch_sizes))
    plt.bar(positions, means, yerr=errors, capsize=4)
    plt.xticks(positions, [batch_label(value) for value in batch_sizes])
    plt.ylabel("Final test accuracy (%)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--targets", default="0.50,0.52,0.54,0.55,0.56")
    parser.add_argument("--convergence-window", type=int, default=10)
    parser.add_argument("--convergence-tolerance", type=float, default=0.0025)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = prepare_curves(args.results_dir)
    grouped_runs = group_runs(curves)
    summary_rows = read_csv(args.results_dir / "summary.csv")
    targets = parse_float_list(args.targets)

    write_convergence_summary(
        grouped_runs,
        args.convergence_window,
        args.convergence_tolerance,
        output_dir / "convergence_summary.csv",
    )
    write_batch_summary(summary_rows, output_dir / "batch_summary.csv")
    plot_curves(
        grouped_runs,
        "round",
        "Global round",
        output_dir / "accuracy_vs_round.png",
        aggregate=True,
    )
    plot_individual_curves(
        grouped_runs,
        "cumulative_time_sec",
        "Training + communication time (seconds)",
        output_dir / "accuracy_vs_time.png",
    )
    plot_curves(
        grouped_runs,
        "local_steps_per_trainer",
        "Cumulative local steps per trainer",
        output_dir / "accuracy_vs_local_steps.png",
        aggregate=True,
    )
    target_rows = write_target_metrics(
        grouped_runs, targets, output_dir / "time_to_target.csv"
    )
    write_target_summary(target_rows, output_dir / "time_to_target_summary.csv")
    plot_time_to_target(target_rows, output_dir / "time_to_target.png")
    plot_final_accuracy(summary_rows, output_dir / "final_accuracy.png")

    print(f"Comparison outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
