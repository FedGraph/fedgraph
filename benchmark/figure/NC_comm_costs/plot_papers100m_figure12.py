#!/usr/bin/env python3
"""Recreate the Papers100M batch-size figure from structured FedGraph outputs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


def _configure_matplotlib_cache() -> None:
    if "MPLCONFIGDIR" in os.environ:
        return
    default_cache = Path.home() / ".matplotlib"
    if default_cache.exists() and os.access(default_cache, os.W_OK):
        return
    cache_dir = Path(tempfile.gettempdir()) / "fedgraph_matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


_configure_matplotlib_cache()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MIB = 1024**2
MEMORY_METRICS = {
    "process_peak_rss_bytes": "Peak Process RSS (MiB)",
    "cuda_max_memory_allocated_bytes": "Peak PyTorch CUDA Allocated (MiB)",
    "cuda_max_memory_reserved_bytes": "Peak PyTorch CUDA Reserved (MiB)",
}
EXPECTED_DATASET = "ogbn-papers100M"
EXPECTED_METHOD = "FedAvg"
EXPECTED_PARTITION_POLICY = "label_dirichlet_balanced_unlabeled"


@dataclass(frozen=True)
class FitResult:
    method: str
    x: np.ndarray
    y: np.ndarray
    r_squared: float


@dataclass(frozen=True)
class SelectedRun:
    summary_path: Path
    summary: dict[str, Any]
    config_path: Path
    config: dict[str, Any]
    resource_summary_path: Path
    resource_summary: dict[str, Any]
    experiment_manifest_path: Path | None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(value)
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table to {path}")
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _resolve_recorded_path(
    summary_path: Path,
    record: dict[str, Any],
    field: str,
    fallback: Path,
    repo_root: Path,
) -> Path:
    candidates = [fallback]
    raw_path = record.get(field)
    if raw_path:
        recorded = Path(str(raw_path)).expanduser()
        if recorded.is_absolute():
            candidates.append(recorded)
        else:
            candidates.extend(
                [
                    repo_root / recorded,
                    Path.cwd() / recorded,
                    summary_path.parent / recorded,
                ]
            )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve {field}; checked: {rendered}")


def _select_runs(
    summary_paths: Sequence[Path],
    batch_sizes: Sequence[int],
    repo_root: Path,
) -> list[SelectedRun]:
    requested = set(batch_sizes)
    matches: dict[int, list[tuple[Path, dict[str, Any]]]] = {
        batch_size: [] for batch_size in batch_sizes
    }
    for summary_path in summary_paths:
        for record in _read_jsonl(summary_path):
            batch_size = record.get("batch_size")
            if batch_size in requested and record.get("status") == "completed":
                matches[int(batch_size)].append((summary_path, record))

    selected: list[SelectedRun] = []
    for batch_size in batch_sizes:
        candidates = matches[batch_size]
        if len(candidates) != 1:
            raise ValueError(
                f"Batch {batch_size} matched {len(candidates)} completed runs; "
                "provide summary files that identify exactly one run per batch"
            )
        summary_path, record = candidates[0]
        run_id = str(record.get("run_id", ""))
        if not run_id:
            raise ValueError(f"Batch {batch_size} summary has no run_id")
        run_dir = summary_path.parent / "runs" / run_id
        config_path = _resolve_recorded_path(
            summary_path, record, "config_path", run_dir / "config.json", repo_root
        )
        resource_path = _resolve_recorded_path(
            summary_path,
            record,
            "resource_snapshot_summary_path",
            run_dir / "fedgraph_logs" / "resource_snapshot_summary.json",
            repo_root,
        )
        experiment_manifest = summary_path.parent / "manifest.json"
        selected.append(
            SelectedRun(
                summary_path=summary_path.resolve(),
                summary=record,
                config_path=config_path,
                config=_read_json(config_path),
                resource_summary_path=resource_path,
                resource_summary=_read_json(resource_path),
                experiment_manifest_path=(
                    experiment_manifest.resolve()
                    if experiment_manifest.is_file()
                    else None
                ),
            )
        )
    return selected


def _require_number(record: dict[str, Any], field: str, context: str) -> float:
    value = record.get(field)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{context} has no numeric {field}")
    return float(value)


def _validate_runs(
    runs: Sequence[SelectedRun],
    *,
    expected_trainers: int,
    expected_rounds: int,
    expected_seed: int,
    expected_iid_beta: float,
) -> None:
    shared_fields = (
        "dataset",
        "method",
        "n_trainer",
        "global_rounds",
        "local_step",
        "learning_rate",
        "iid_beta",
        "distribution_type",
        "num_layers",
        "num_hops",
        "gpu",
        "evaluation_split",
        "resource_monitor_mode",
    )
    baseline = runs[0].summary
    for run in runs:
        summary = run.summary
        context = f"run {summary.get('run_id')}"
        expected_values = {
            "dataset": EXPECTED_DATASET,
            "method": EXPECTED_METHOD,
            "n_trainer": expected_trainers,
            "global_rounds": expected_rounds,
            "rounds_recorded": expected_rounds,
            "seed": expected_seed,
            "num_hops": 0,
        }
        for field, expected in expected_values.items():
            if summary.get(field) != expected:
                raise ValueError(
                    f"{context} has {field}={summary.get(field)!r}; expected {expected!r}"
                )
        iid_beta = _require_number(summary, "iid_beta", context)
        if not np.isclose(iid_beta, expected_iid_beta):
            raise ValueError(
                f"{context} has iid_beta={iid_beta}; expected {expected_iid_beta}"
            )
        training_time = _require_number(summary, "total_training_time_sec", context)
        test_accuracy = _require_number(summary, "test_acc_final", context)
        if training_time < 0:
            raise ValueError(f"{context} has negative training time")
        if not 0 <= test_accuracy <= 1:
            raise ValueError(f"{context} has test accuracy outside [0, 1]")
        for field in shared_fields:
            if field not in summary:
                raise ValueError(f"{context} has no {field}")
            if summary[field] != baseline.get(field):
                raise ValueError(
                    f"{context} has {field}={summary[field]!r}, which differs from "
                    f"the first selected run's {baseline.get(field)!r}"
                )


def _load_shards(
    artifact_manifest: dict[str, Any], expected_trainers: int
) -> dict[int, dict[str, Any]]:
    if artifact_manifest.get("n_trainer") != expected_trainers:
        raise ValueError(
            "Artifact trainer count does not match the expected experiment trainer count"
        )
    if artifact_manifest.get("partition_policy") != EXPECTED_PARTITION_POLICY:
        raise ValueError(
            "Balanced Figure 12 requires partition_policy="
            f"{EXPECTED_PARTITION_POLICY!r}, got "
            f"{artifact_manifest.get('partition_policy')!r}"
        )
    raw_shards = artifact_manifest.get("shards")
    if not isinstance(raw_shards, list):
        raise ValueError("Artifact manifest has no shards list")
    shards: dict[int, dict[str, Any]] = {}
    for shard in raw_shards:
        if not isinstance(shard, dict) or not isinstance(shard.get("trainer_id"), int):
            raise ValueError("Artifact manifest contains malformed shard metadata")
        trainer_id = int(shard["trainer_id"])
        if trainer_id in shards:
            raise ValueError(f"Artifact contains duplicate trainer_id {trainer_id}")
        _require_number(shard, "node_count", f"artifact shard {trainer_id}")
        _require_number(shard, "internal_edge_count", f"artifact shard {trainer_id}")
        shards[trainer_id] = shard
    expected_ids = set(range(expected_trainers))
    if set(shards) != expected_ids:
        raise ValueError("Artifact manifest must contain every trainer ID exactly once")
    return shards


def prepare_figure_data(
    *,
    summary_paths: Sequence[Path],
    artifact_manifest_path: Path,
    batch_sizes: Sequence[int],
    memory_metric: str,
    expected_trainers: int,
    expected_rounds: int,
    expected_seed: int,
    expected_iid_beta: float,
    repo_root: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[SelectedRun],
    dict[str, Any],
]:
    artifact_manifest = _read_json(artifact_manifest_path)
    shards = _load_shards(artifact_manifest, expected_trainers)
    runs = _select_runs(summary_paths, batch_sizes, repo_root)
    _validate_runs(
        runs,
        expected_trainers=expected_trainers,
        expected_rounds=expected_rounds,
        expected_seed=expected_seed,
        expected_iid_beta=expected_iid_beta,
    )

    batch_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    for run in runs:
        summary = run.summary
        batch_size = int(summary["batch_size"])
        batch_rows.append(
            {
                "batch_size": batch_size,
                "training_time_sec": float(summary["total_training_time_sec"]),
                "test_accuracy": float(summary["test_acc_final"]),
                "run_id": summary["run_id"],
                "seed": int(summary["seed"]),
                "global_rounds": int(summary["global_rounds"]),
            }
        )

        trainer_sources: dict[int, dict[str, Any]] = {}
        for source in run.resource_summary.get("sources", []):
            if (
                not isinstance(source, dict)
                or source.get("source") != "trainer"
                or not isinstance(source.get("trainer_id"), int)
            ):
                continue
            trainer_id = int(source["trainer_id"])
            if trainer_id in trainer_sources:
                raise ValueError(
                    f"Run {summary['run_id']} has duplicate resource summaries for "
                    f"trainer {trainer_id}"
                )
            trainer_sources[trainer_id] = source
        if set(trainer_sources) != set(shards):
            raise ValueError(
                f"Run {summary['run_id']} resource summary does not contain exactly "
                f"{expected_trainers} trainer sources"
            )
        for trainer_id, shard in shards.items():
            source = trainer_sources[trainer_id]
            memory_bytes = _require_number(
                source, memory_metric, f"run {summary['run_id']} trainer {trainer_id}"
            )
            if memory_bytes < 0:
                raise ValueError(
                    f"Run {summary['run_id']} trainer {trainer_id} has negative memory"
                )
            memory_rows.append(
                {
                    "batch_size": batch_size,
                    "trainer_id": trainer_id,
                    "memory_metric": memory_metric,
                    "memory_bytes": int(memory_bytes),
                    "memory_mib": memory_bytes / MIB,
                    "node_count": int(shard["node_count"]),
                    "internal_edge_count": int(shard["internal_edge_count"]),
                    "snapshot_count": int(source.get("snapshot_count", 0)),
                    "run_id": summary["run_id"],
                }
            )

    batch_rows.sort(key=lambda row: int(row["batch_size"]))
    memory_rows.sort(key=lambda row: (int(row["batch_size"]), int(row["trainer_id"])))
    return batch_rows, memory_rows, runs, artifact_manifest


def residual_inlier_mask(
    x: np.ndarray, y: np.ndarray, threshold_std: float
) -> np.ndarray:
    """Return a residual-based mask, or all rows when a fit is not identifiable."""
    if len(x) < 3 or np.unique(x).size < 2 or threshold_std <= 0:
        return np.ones(len(x), dtype=bool)
    coefficients = np.polyfit(x, y, 1)
    residuals = y - np.polyval(coefficients, x)
    residual_std = float(np.std(residuals))
    if residual_std == 0:
        return np.ones(len(x), dtype=bool)
    return np.abs(residuals) <= threshold_std * residual_std


def _r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    residual_sum = float(np.sum((actual - predicted) ** 2))
    total_sum = float(np.sum((actual - np.mean(actual)) ** 2))
    return 1.0 if total_sum == 0 and residual_sum == 0 else 1 - residual_sum / total_sum


def best_fit(x: np.ndarray, y: np.ndarray, axis: str) -> FitResult | None:
    if (
        len(x) < 2
        or np.unique(x).size < 2
        or (axis == "nodes" and float(np.ptp(x)) <= 1)
    ):
        return None
    x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    candidates: list[FitResult] = []

    linear = np.polyfit(x, y, 1)
    candidates.append(
        FitResult(
            method="linear",
            x=x_line,
            y=np.polyval(linear, x_line),
            r_squared=_r_squared(y, np.polyval(linear, x)),
        )
    )
    if axis == "nodes" and np.unique(x).size >= 3 and len(x) >= 6:
        quadratic = np.polyfit(x, y, 2)
        candidates.append(
            FitResult(
                method="quadratic",
                x=x_line,
                y=np.polyval(quadratic, x_line),
                r_squared=_r_squared(y, np.polyval(quadratic, x)),
            )
        )
    if axis == "edges" and np.all(x >= 0):
        log_x = np.log1p(x)
        if np.unique(log_x).size >= 2:
            logarithmic = np.polyfit(log_x, y, 1)
            candidates.append(
                FitResult(
                    method="logarithmic",
                    x=x_line,
                    y=np.polyval(logarithmic, np.log1p(x_line)),
                    r_squared=_r_squared(y, np.polyval(logarithmic, log_x)),
                )
            )
    return max(candidates, key=lambda candidate: candidate.r_squared)


def _padded_limits(
    values: Iterable[float], minimum_padding: float
) -> tuple[float, float]:
    values_array = np.asarray(list(values), dtype=float)
    low = float(np.min(values_array))
    high = float(np.max(values_array))
    padding = max((high - low) * 0.15, minimum_padding)
    return low - padding, high + padding


def create_figure(
    *,
    batch_rows: Sequence[dict[str, Any]],
    memory_rows: list[dict[str, Any]],
    memory_metric: str,
    outlier_threshold_std: float,
    pdf_path: Path,
    png_path: Path,
) -> dict[str, Any]:
    colors = ["#D94A3A", "#F0A12B", "#2BAE66"]
    batch_sizes = [int(row["batch_size"]) for row in batch_rows]
    if len(batch_sizes) > len(colors):
        colors = list(plt.get_cmap("tab10").colors[: len(batch_sizes)])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    time_axis, node_axis, edge_axis = axes
    accuracy_axis = time_axis.twinx()

    labels = [str(batch_size) for batch_size in batch_sizes]
    training_times = [float(row["training_time_sec"]) for row in batch_rows]
    test_accuracies = [float(row["test_accuracy"]) for row in batch_rows]
    bars = time_axis.bar(labels, training_times, color="#5B9BD5", alpha=0.8, width=0.58)
    accuracy_line = accuracy_axis.plot(
        labels,
        test_accuracies,
        color="#F28E2B",
        marker="o",
        linewidth=2,
        markersize=5,
    )[0]
    time_axis.set_xlabel("Batch Size")
    time_axis.set_ylabel("Train Time (s)", color="#2F6F9F")
    accuracy_axis.set_ylabel("Test Accuracy", color="#C86E10")
    time_axis.tick_params(axis="y", labelcolor="#2F6F9F")
    accuracy_axis.tick_params(axis="y", labelcolor="#C86E10")
    time_axis.set_ylim(*_padded_limits(training_times, minimum_padding=1.0))
    acc_low, acc_high = _padded_limits(test_accuracies, minimum_padding=0.001)
    accuracy_axis.set_ylim(max(0.0, acc_low), min(1.0, acc_high))
    time_axis.legend(
        [bars[0], accuracy_line], ["Train Time", "Test Accuracy"], loc="upper left"
    )

    fit_metadata: dict[str, Any] = {}
    for axis_name, axis, x_field, x_label in (
        ("nodes", node_axis, "node_count", "Number of Local Nodes"),
        ("edges", edge_axis, "internal_edge_count", "Number of Local Edges"),
    ):
        fit_metadata[axis_name] = {}
        constant_batches: list[int] = []
        outlier_label_used = False
        for batch_size, color in zip(batch_sizes, colors):
            rows = [row for row in memory_rows if row["batch_size"] == batch_size]
            x = np.asarray([row[x_field] for row in rows], dtype=float)
            y = np.asarray([row["memory_mib"] for row in rows], dtype=float)
            fit_has_variation = not (axis_name == "nodes" and float(np.ptp(x)) <= 1)
            mask = (
                residual_inlier_mask(x, y, outlier_threshold_std)
                if fit_has_variation
                else np.ones(len(x), dtype=bool)
            )
            mask_field = f"{axis_name}_fit_inlier"
            for row, inlier in zip(rows, mask):
                row[mask_field] = bool(inlier)

            axis.scatter(
                x[mask],
                y[mask],
                color=color,
                alpha=0.7,
                s=22,
                marker="x",
                label=f"Batch {batch_size}",
            )
            if np.any(~mask):
                axis.scatter(
                    x[~mask],
                    y[~mask],
                    color="lightgray",
                    alpha=0.55,
                    s=22,
                    marker="x",
                    label="Outliers" if not outlier_label_used else None,
                )
                outlier_label_used = True

            fit = best_fit(x[mask], y[mask], axis_name)
            if fit is None:
                constant_batches.append(batch_size)
                fit_metadata[axis_name][str(batch_size)] = {
                    "method": None,
                    "r_squared": None,
                    "outliers_removed": int(np.sum(~mask)),
                    "reason": "insufficient meaningful variation in x values",
                }
                continue
            axis.plot(
                fit.x,
                fit.y,
                color=color,
                linestyle="--",
                linewidth=1.7,
                label=f"Batch {batch_size} fit (R2={fit.r_squared:.3f})",
            )
            fit_metadata[axis_name][str(batch_size)] = {
                "method": fit.method,
                "r_squared": fit.r_squared,
                "outliers_removed": int(np.sum(~mask)),
                "reason": None,
            }

        axis.set_xlabel(x_label)
        axis.set_ylabel(MEMORY_METRICS[memory_metric])
        axis.set_title(f"Memory Usage vs {x_label.removeprefix('Number of ')}")
        axis.legend(fontsize=8, loc="best")
        if constant_batches:
            unique_x = sorted({int(row[x_field]) for row in memory_rows})
            axis.text(
                0.02,
                0.02,
                "Regression omitted: balanced shards have "
                f"{len(unique_x)} distinct {axis_name} count(s).",
                transform=axis.transAxes,
                fontsize=8,
                va="bottom",
            )

    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fit_metadata


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args], cwd=repo_root, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run_git("status", "--short")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
    }


def _input_file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _output_paths(output_dir: Path, output_stem: str) -> dict[str, Path]:
    return {
        "pdf": output_dir / f"{output_stem}.pdf",
        "png": output_dir / f"{output_stem}.png",
        "batch_metrics": output_dir / f"{output_stem}_batch_metrics.csv",
        "trainer_memory": output_dir / f"{output_stem}_trainer_memory.csv",
        "selected_runs": output_dir / f"{output_stem}_selected_runs.json",
        "artifact_manifest": output_dir / f"{output_stem}_artifact_manifest.json",
        "provenance": output_dir / f"{output_stem}_provenance.json",
    }


def _prepare_output(paths: dict[str, Path], overwrite: bool) -> None:
    output_dir = next(iter(paths.values())).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    collisions = [path for path in paths.values() if path.exists()]
    if collisions and not overwrite:
        rendered = ", ".join(str(path) for path in collisions)
        raise FileExistsError(
            f"Refusing to overwrite existing Figure 12 outputs: {rendered}. "
            "Use a new --output-stem or pass --overwrite."
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-jsonl",
        type=Path,
        action="append",
        required=True,
        help="Benchmark summary.jsonl; repeat for results stored in separate directories.",
    )
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        required=True,
        help="Validated splitter manifest.json used by all selected runs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default="figure12_balanced")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument(
        "--memory-metric",
        choices=sorted(MEMORY_METRICS),
        default="process_peak_rss_bytes",
    )
    parser.add_argument("--outlier-threshold-std", type=float, default=1.5)
    parser.add_argument("--expected-trainers", type=int, default=195)
    parser.add_argument("--expected-rounds", type=int, default=800)
    parser.add_argument("--expected-seed", type=int, default=42)
    parser.add_argument("--expected-iid-beta", type=float, default=10000.0)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not args.output_stem or Path(args.output_stem).name != args.output_stem:
        parser.error(
            "--output-stem must be a filename stem without directory components"
        )
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("--batch-sizes must not contain duplicates")
    if args.outlier_threshold_std < 0:
        parser.error("--outlier-threshold-std must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary_paths = [path.expanduser().resolve() for path in args.summary_jsonl]
    artifact_manifest_path = args.artifact_manifest.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    for path in [*summary_paths, artifact_manifest_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_paths = _output_paths(
        args.output_dir.expanduser().resolve(), args.output_stem
    )
    _prepare_output(output_paths, args.overwrite)
    batch_rows, memory_rows, runs, artifact_manifest = prepare_figure_data(
        summary_paths=summary_paths,
        artifact_manifest_path=artifact_manifest_path,
        batch_sizes=args.batch_sizes,
        memory_metric=args.memory_metric,
        expected_trainers=args.expected_trainers,
        expected_rounds=args.expected_rounds,
        expected_seed=args.expected_seed,
        expected_iid_beta=args.expected_iid_beta,
        repo_root=repo_root,
    )

    fit_metadata = create_figure(
        batch_rows=batch_rows,
        memory_rows=memory_rows,
        memory_metric=args.memory_metric,
        outlier_threshold_std=args.outlier_threshold_std,
        pdf_path=output_paths["pdf"],
        png_path=output_paths["png"],
    )
    _write_csv(output_paths["batch_metrics"], batch_rows)
    _write_csv(output_paths["trainer_memory"], memory_rows)
    output_paths["selected_runs"].write_text(
        json.dumps([run.summary for run in runs], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(artifact_manifest_path, output_paths["artifact_manifest"])

    input_paths = {artifact_manifest_path, *summary_paths}
    for run in runs:
        input_paths.update({run.config_path, run.resource_summary_path})
        if run.experiment_manifest_path is not None:
            input_paths.add(run.experiment_manifest_path)
    configs = [run.config for run in runs]
    hf_repositories = sorted(
        {
            str(config["hf_local_artifact_repo"])
            for config in configs
            if config.get("hf_local_artifact_repo")
        }
    )
    hf_revisions = sorted(
        {
            str(config["hf_local_artifact_revision"])
            for config in configs
            if config.get("hf_local_artifact_revision")
        }
    )
    if len(hf_repositories) > 1:
        raise ValueError(
            "Selected runs reference different Hugging Face artifact repositories"
        )
    if len(hf_revisions) > 1:
        raise ValueError(
            "Selected runs reference different Hugging Face artifact revisions"
        )
    warnings: list[str] = []
    distinct_node_counts = {
        int(shard["node_count"]) for shard in artifact_manifest["shards"]
    }
    if len(distinct_node_counts) < 2:
        warnings.append(
            "All shards have the same node count; node-memory regression was omitted."
        )
    elif max(distinct_node_counts) - min(distinct_node_counts) <= 1:
        warnings.append(
            "Balanced shard node counts differ by at most one; node-memory regression "
            "is not scientifically informative."
        )
    if hf_repositories and not hf_revisions:
        warnings.append(
            "The runs identify a Hugging Face repository but do not pin a repository "
            "revision. The copied artifact manifest and its SHA-256 preserve content "
            "identity, but future runs should pass --hf-local-artifact-revision."
        )

    provenance = {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "figure": "FedGraph Figure 12 balanced-partition redraw",
        "dataset": EXPECTED_DATASET,
        "partition_policy": artifact_manifest["partition_policy"],
        "artifact_manifest_sha256": _sha256(artifact_manifest_path),
        "hf_artifact_repositories": hf_repositories,
        "hf_artifact_revisions": hf_revisions,
        "batch_sizes": list(args.batch_sizes),
        "expected_trainers": args.expected_trainers,
        "expected_rounds": args.expected_rounds,
        "expected_seed": args.expected_seed,
        "expected_iid_beta": args.expected_iid_beta,
        "training_time_field": "total_training_time_sec",
        "test_accuracy_field": "test_acc_final",
        "memory_metric": args.memory_metric,
        "memory_unit": "MiB",
        "outlier_threshold_std": args.outlier_threshold_std,
        "fits": fit_metadata,
        "inputs": [_input_file_record(path) for path in sorted(input_paths)],
        "plot_script": _input_file_record(Path(__file__).resolve()),
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "git": _git_metadata(repo_root),
        "outputs": {
            name: _input_file_record(path)
            for name, path in output_paths.items()
            if name != "provenance"
        },
        "warnings": warnings,
    }
    output_paths["provenance"].write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"Wrote Figure 12 redraw to {output_paths['pdf']}")
    print(f"Wrote provenance to {output_paths['provenance']}")
    for warning in warnings:
        print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
