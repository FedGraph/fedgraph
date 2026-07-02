#!/usr/bin/env python3
"""Create comparison tables and plots from NC batch-size benchmark outputs."""

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


def write_rows(
    path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None
) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def optional_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def optional_int(value) -> Optional[int]:
    number = optional_float(value)
    return int(number) if number is not None else None


def first_present(row: dict, keys: Iterable[str]):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def optional_float_any(row: dict, keys: Iterable[str]) -> Optional[float]:
    return optional_float(first_present(row, keys))


def numeric_values_any(rows: List[dict], keys: Iterable[str]) -> List[float]:
    values = []
    for row in rows:
        value = optional_float_any(row, keys)
        if value is not None:
            values.append(value)
    return values


def mean_or_none(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def std_or_zero(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def load_run_configs(results_dir: Path) -> Dict[str, dict]:
    configs = {}
    for config_path in (results_dir / "runs").glob("*/config.json"):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        configs[config_path.parent.name] = config
    return configs


def load_round_times(results_dir: Path) -> Dict[str, Dict[int, float]]:
    run_times = {}
    for log_path in (results_dir / "runs").glob("*/stdout.log"):
        cumulative = 0.0
        times = {}
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for round_id, train_time, comm_time in ROUND_TIMING_RE.findall(text):
            cumulative += float(train_time) + float(comm_time)
            times[int(round_id)] = cumulative
        run_times[log_path.parent.name] = times
    return run_times


def curve_csv_path(results_dir: Path) -> Path:
    new_path = results_dir / "validation_curves.csv"
    if new_path.exists():
        return new_path
    legacy_path = results_dir / "accuracy_curves.csv"
    if legacy_path.exists():
        return legacy_path
    raise FileNotFoundError(
        f"Expected validation_curves.csv or accuracy_curves.csv in {results_dir}"
    )


def load_round_metrics(results_dir: Path) -> Dict[tuple, dict]:
    path = results_dir / "round_metrics.csv"
    if not path.exists():
        return {}
    return {
        (row["run_id"], int(row["round"])): row
        for row in read_csv(path)
        if row.get("run_id") and row.get("round")
    }


def prepare_curves(results_dir: Path) -> List[dict]:
    rows = read_csv(curve_csv_path(results_dir))
    configs = load_run_configs(results_dir)
    round_metrics = load_round_metrics(results_dir)
    fallback_times = load_round_times(results_dir)

    prepared = []
    for row in rows:
        run_id = row["run_id"]
        round_id = int(row["round"])
        config = configs.get(run_id, {})
        metrics = round_metrics.get((run_id, round_id), {})

        val_acc = optional_float(
            first_present(row, ["val_acc", "validation_accuracy"])
        )
        metric_name = "val_acc"
        if val_acc is None:
            val_acc = optional_float(row.get("accuracy"))
            metric_name = "legacy_test_acc"
        if val_acc is None:
            continue

        val_loss = optional_float(
            first_present(row, ["val_loss", "validation_loss"])
        )
        if val_loss is None:
            val_loss = optional_float(
                first_present(metrics, ["val_loss", "validation_loss"])
            )

        train_comm_time = optional_float(
            first_present(
                metrics,
                [
                    "cum_train_comm_time_sec",
                    "cumulative_training_communication_time_sec",
                ],
            )
        )
        if train_comm_time is None:
            train_comm_time = fallback_times.get(run_id, {}).get(round_id)

        local_step = int(config.get("local_step", 0))
        local_steps = optional_int(
            first_present(metrics, ["local_steps", "local_steps_per_trainer"])
        )
        if local_steps is None:
            local_steps = round_id * local_step

        prepared.append(
            {
                "run_id": run_id,
                "batch_size": int(row["batch_size"]),
                "seed": int(row["seed"]),
                "round": round_id,
                "metric_name": metric_name,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "local_steps": local_steps,
                "train_comm_time_sec": train_comm_time,
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


def curve_accuracy_label(grouped_runs: Dict[str, List[dict]]) -> str:
    metric_names = {
        row.get("metric_name")
        for rows in grouped_runs.values()
        for row in rows
        if row.get("metric_name")
    }
    if metric_names == {"legacy_test_acc"}:
        return "Global test accuracy (%)"
    return "Validation accuracy (%)"


def finite_values(curve: List[dict], key: str) -> List[dict]:
    return [item for item in curve if item.get(key) is not None]


def last_value_round(curve: List[dict], key: str) -> tuple[Optional[float], Optional[int]]:
    values = finite_values(curve, key)
    if not values:
        return None, None
    item = values[-1]
    return item[key], item["round"]


def best_value_round(
    curve: List[dict], key: str, mode: str
) -> tuple[Optional[float], Optional[int]]:
    values = finite_values(curve, key)
    if not values:
        return None, None
    selector = min if mode == "min" else max
    item = selector(values, key=lambda value: value[key])
    return item[key], item["round"]


def plateau_round(
    curve: List[dict], key: str, window: int, tolerance: float
) -> Optional[int]:
    values = finite_values(curve, key)
    if window <= 1 or len(values) < window:
        return None
    for end_idx in range(window - 1, len(values)):
        recent = values[end_idx - window + 1 : end_idx + 1]
        recent_values = [item[key] for item in recent]
        if max(recent_values) - min(recent_values) <= tolerance:
            return recent[-1]["round"]
    return None


def end_window_stats(
    curve: List[dict], key: str, window: int, tolerance: float
) -> dict:
    values = finite_values(curve, key)
    if window <= 1 or len(values) < window:
        return {
            f"{key}_end_gain": None,
            f"{key}_end_range": None,
            f"{key}_flat_at_end": False,
        }
    end_values = [item[key] for item in values[-window:]]
    value_range = max(end_values) - min(end_values)
    return {
        f"{key}_end_gain": end_values[-1] - end_values[0],
        f"{key}_end_range": value_range,
        f"{key}_flat_at_end": value_range <= tolerance,
    }


def write_convergence_summary(
    grouped_runs: Dict[str, List[dict]],
    window: int,
    val_loss_tolerance: float,
    val_acc_tolerance: float,
    output_path: Path,
) -> None:
    fieldnames = [
        "run_id",
        "batch_size",
        "seed",
        "metric_name",
        "rounds_recorded",
        "val_loss_final",
        "val_loss_best",
        "val_loss_best_round",
        "val_loss_plateau_round",
        "val_acc_final",
        "val_acc_best",
        "val_acc_best_round",
        "val_acc_plateau_round",
        "val_convergence_round",
        "val_loss_end_gain",
        "val_loss_end_range",
        "val_loss_flat_at_end",
        "val_acc_end_gain",
        "val_acc_end_range",
        "val_acc_flat_at_end",
    ]
    rows = []
    for run_id, curve in grouped_runs.items():
        val_loss_final, _ = last_value_round(curve, "val_loss")
        val_loss_best, val_loss_best_round = best_value_round(
            curve, "val_loss", "min"
        )
        val_acc_final, _ = last_value_round(curve, "val_acc")
        val_acc_best, val_acc_best_round = best_value_round(curve, "val_acc", "max")
        val_loss_plateau_round = plateau_round(
            curve, "val_loss", window, val_loss_tolerance
        )
        val_acc_plateau_round = plateau_round(
            curve, "val_acc", window, val_acc_tolerance
        )
        rows.append(
            {
                "run_id": run_id,
                "batch_size": curve[0]["batch_size"],
                "seed": curve[0]["seed"],
                "metric_name": curve[0].get("metric_name"),
                "rounds_recorded": len(curve),
                "val_loss_final": val_loss_final,
                "val_loss_best": val_loss_best,
                "val_loss_best_round": val_loss_best_round,
                "val_loss_plateau_round": val_loss_plateau_round,
                "val_acc_final": val_acc_final,
                "val_acc_best": val_acc_best,
                "val_acc_best_round": val_acc_best_round,
                "val_acc_plateau_round": val_acc_plateau_round,
                "val_convergence_round": val_loss_plateau_round,
                **end_window_stats(
                    curve, "val_loss", window, val_loss_tolerance
                ),
                **end_window_stats(curve, "val_acc", window, val_acc_tolerance),
            }
        )
    write_rows(output_path, rows, fieldnames)


def plot_curves(
    grouped_runs: Dict[str, List[dict]],
    x_key: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    aggregate: bool = False,
) -> bool:
    available = [
        rows for rows in grouped_runs.values() if rows and rows[0].get(x_key) is not None
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
                    by_batch_and_x[batch_size][x_value].append(row["val_acc"])

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
                [100 * row["val_acc"] for row in rows],
                label=display_label,
                linewidth=2,
                alpha=0.85,
            )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return True


def write_batch_summary(summary_rows: List[dict], output_path: Path) -> List[dict]:
    fieldnames = [
        "batch_size",
        "batch_label",
        "runs",
        "completed_runs",
        "failed_runs",
        "seeds",
        "test_acc_final_mean",
        "test_acc_final_std",
        "test_loss_final_mean",
        "val_acc_best_mean",
        "val_acc_best_std",
        "val_loss_best_mean",
        "val_convergence_round_mean",
        "val_convergence_time_sec_mean",
        "total_pure_train_time_sec_mean",
        "total_comm_time_sec_mean",
        "total_train_comm_time_sec_mean",
        "test_acc_delta_vs_full_batch",
        "val_acc_best_delta_vs_full_batch",
    ]
    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[int(row["batch_size"])].append(row)

    rows = []
    for batch_size in sort_batch_sizes(grouped):
        batch_rows = grouped[batch_size]
        completed = [row for row in batch_rows if row.get("status") == "completed"]
        test_acc = numeric_values_any(
            completed, ["test_acc_final", "final_test_accuracy", "final_accuracy"]
        )
        test_loss = numeric_values_any(
            completed, ["test_loss_final", "final_test_loss", "final_loss"]
        )
        val_acc_best = numeric_values_any(
            completed, ["val_acc_best", "best_validation_accuracy", "best_accuracy"]
        )
        val_loss_best = numeric_values_any(completed, ["val_loss_best"])
        val_convergence_round = numeric_values_any(
            completed, ["val_convergence_round", "validation_convergence_round"]
        )
        val_convergence_time = numeric_values_any(
            completed,
            [
                "val_convergence_time_sec",
                "validation_convergence_training_communication_time_sec",
            ],
        )
        pure_train_time = numeric_values_any(
            completed, ["total_pure_train_time_sec", "pure_training_time_sec"]
        )
        comm_time = numeric_values_any(
            completed, ["total_comm_time_sec", "communication_time_sec"]
        )
        train_comm_time = numeric_values_any(
            completed,
            ["total_train_comm_time_sec", "training_communication_time_sec"],
        )
        seeds = sorted({row.get("seed", "") for row in batch_rows})
        rows.append(
            {
                "batch_size": batch_size,
                "batch_label": batch_label(batch_size),
                "runs": len(batch_rows),
                "completed_runs": len(completed),
                "failed_runs": len(batch_rows) - len(completed),
                "seeds": ",".join(seeds),
                "test_acc_final_mean": mean_or_none(test_acc),
                "test_acc_final_std": std_or_zero(test_acc),
                "test_loss_final_mean": mean_or_none(test_loss),
                "val_acc_best_mean": mean_or_none(val_acc_best),
                "val_acc_best_std": std_or_zero(val_acc_best),
                "val_loss_best_mean": mean_or_none(val_loss_best),
                "val_convergence_round_mean": mean_or_none(
                    val_convergence_round
                ),
                "val_convergence_time_sec_mean": mean_or_none(
                    val_convergence_time
                ),
                "total_pure_train_time_sec_mean": mean_or_none(pure_train_time),
                "total_comm_time_sec_mean": mean_or_none(comm_time),
                "total_train_comm_time_sec_mean": mean_or_none(train_comm_time),
                "test_acc_delta_vs_full_batch": None,
                "val_acc_best_delta_vs_full_batch": None,
            }
        )

    full_batch = next((row for row in rows if row["batch_size"] == -1), None)
    test_baseline = full_batch["test_acc_final_mean"] if full_batch else None
    val_baseline = full_batch["val_acc_best_mean"] if full_batch else None
    for row in rows:
        test_mean = row["test_acc_final_mean"]
        row["test_acc_delta_vs_full_batch"] = (
            test_mean - test_baseline
            if test_mean is not None and test_baseline is not None
            else None
        )
        val_mean = row["val_acc_best_mean"]
        row["val_acc_best_delta_vs_full_batch"] = (
            val_mean - val_baseline
            if val_mean is not None and val_baseline is not None
            else None
        )

    write_rows(output_path, rows, fieldnames)
    return rows


def write_target_metrics(
    grouped_runs: Dict[str, List[dict]], targets: List[float], output_path: Path
) -> List[dict]:
    fieldnames = [
        "run_id",
        "batch_size",
        "seed",
        "metric_name",
        "target_val_acc",
        "round",
        "local_steps",
        "train_comm_time_sec",
    ]
    rows = []
    for run_id, curve in grouped_runs.items():
        for target in targets:
            reached = next((item for item in curve if item["val_acc"] >= target), None)
            rows.append(
                {
                    "run_id": run_id,
                    "batch_size": curve[0]["batch_size"],
                    "seed": curve[0]["seed"],
                    "metric_name": curve[0].get("metric_name"),
                    "target_val_acc": target,
                    "round": reached["round"] if reached else None,
                    "local_steps": reached["local_steps"] if reached else None,
                    "train_comm_time_sec": (
                        reached["train_comm_time_sec"] if reached else None
                    ),
                }
            )

    write_rows(output_path, rows, fieldnames)
    return rows


def write_target_summary(rows: List[dict], output_path: Path) -> List[dict]:
    fieldnames = [
        "batch_size",
        "batch_label",
        "target_val_acc",
        "target_val_acc_percent",
        "metric_name",
        "runs",
        "reached_runs",
        "reach_rate",
        "round_mean",
        "round_std",
        "local_steps_mean",
        "train_comm_time_sec_mean",
        "train_comm_time_sec_std",
    ]
    grouped = defaultdict(list)
    for row in rows:
        key = (
            int(row["batch_size"]),
            row.get("metric_name"),
            float(row["target_val_acc"]),
        )
        grouped[key].append(row)

    summary_rows = []
    for (batch_size, metric_name, target), target_rows in sorted(
        grouped.items(), key=lambda item: (item[0][0] == -1, item[0][0], item[0][2])
    ):
        reached = [
            row for row in target_rows if optional_float(row["train_comm_time_sec"]) is not None
        ]
        rounds = numeric_values_any(reached, ["round"])
        local_steps = numeric_values_any(reached, ["local_steps"])
        times = numeric_values_any(reached, ["train_comm_time_sec"])
        summary_rows.append(
            {
                "batch_size": batch_size,
                "batch_label": batch_label(batch_size),
                "target_val_acc": target,
                "target_val_acc_percent": 100 * target,
                "metric_name": metric_name,
                "runs": len(target_rows),
                "reached_runs": len(reached),
                "reach_rate": len(reached) / len(target_rows) if target_rows else None,
                "round_mean": mean_or_none(rounds),
                "round_std": std_or_zero(rounds),
                "local_steps_mean": mean_or_none(local_steps),
                "train_comm_time_sec_mean": mean_or_none(times),
                "train_comm_time_sec_std": std_or_zero(times),
            }
        )

    write_rows(output_path, summary_rows, fieldnames)
    return summary_rows


def plot_time_to_target(rows: List[dict], output_path: Path) -> bool:
    reached = [row for row in rows if row["train_comm_time_sec"] not in (None, "")]
    if not reached:
        return False

    grouped = defaultdict(list)
    for row in reached:
        grouped[int(row["batch_size"])].append(row)

    plt.figure(figsize=(8, 5))
    for batch_size in sort_batch_sizes(grouped):
        by_target = defaultdict(list)
        for row in grouped[batch_size]:
            by_target[float(row["target_val_acc"])].append(
                float(row["train_comm_time_sec"])
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

    plt.xlabel("Target validation accuracy (%)")
    plt.ylabel("Training + communication time (seconds)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return True


def plot_final_test_accuracy(summary_rows: List[dict], output_path: Path) -> bool:
    grouped = defaultdict(list)
    for row in summary_rows:
        value = optional_float_any(
            row, ["test_acc_final", "final_test_accuracy", "final_accuracy"]
        )
        if row.get("status") == "completed" and value is not None:
            grouped[int(row["batch_size"])].append(value)
    if not grouped:
        return False

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
    return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--targets", default="0.50,0.52,0.54,0.55,0.56")
    parser.add_argument("--convergence-window", type=int, default=10)
    parser.add_argument("--val-loss-tolerance", type=float, default=0.01)
    parser.add_argument("--val-acc-tolerance", type=float, default=0.0025)
    parser.add_argument(
        "--convergence-tolerance",
        type=float,
        default=None,
        help="Compatibility alias for --val-acc-tolerance.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.convergence_tolerance is not None:
        args.val_acc_tolerance = args.convergence_tolerance

    output_dir = args.output_dir or args.results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = prepare_curves(args.results_dir)
    grouped_runs = group_runs(curves)
    summary_rows = read_csv(args.results_dir / "summary.csv")
    targets = parse_float_list(args.targets)
    accuracy_label = curve_accuracy_label(grouped_runs)

    write_convergence_summary(
        grouped_runs,
        args.convergence_window,
        args.val_loss_tolerance,
        args.val_acc_tolerance,
        output_dir / "convergence_summary.csv",
    )
    write_batch_summary(summary_rows, output_dir / "batch_summary.csv")
    plot_curves(
        grouped_runs,
        "round",
        "Global round",
        accuracy_label,
        output_dir / "val_acc_vs_round.png",
        aggregate=True,
    )
    plot_curves(
        grouped_runs,
        "train_comm_time_sec",
        "Training + communication time (seconds)",
        accuracy_label,
        output_dir / "val_acc_vs_time.png",
    )
    plot_curves(
        grouped_runs,
        "local_steps",
        "Cumulative local steps per trainer",
        accuracy_label,
        output_dir / "val_acc_vs_local_steps.png",
        aggregate=True,
    )
    target_rows = write_target_metrics(
        grouped_runs, targets, output_dir / "time_to_target.csv"
    )
    write_target_summary(target_rows, output_dir / "time_to_target_summary.csv")
    plot_time_to_target(target_rows, output_dir / "time_to_target.png")
    plot_final_test_accuracy(summary_rows, output_dir / "test_acc_final.png")

    print(f"Comparison outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
