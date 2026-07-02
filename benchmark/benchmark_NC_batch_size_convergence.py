#!/usr/bin/env python3
"""Long-running FedGraph NC batch-size convergence experiments.

This script is intentionally a benchmark runner, not an integration test. It
keeps long convergence jobs out of CI while saving per-run logs, per-round
validation curves, and a compact summary CSV for later analysis.
"""

import argparse
import contextlib
import csv
import datetime as dt
import json
import os
import platform
import random
import re
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

# FedGraph's NC runner currently reports experiment metrics through stdout.
# These patterns convert those logs into structured CSV/JSON outputs. Current
# runs report per-round validation metrics and final-only test metrics.
ROUND_VALIDATION_LOSS_RE = re.compile(
    r"Round\s+(\d+):\s+Global Val Loss\s*=\s*([0-9.eE+-]+)"
)
ROUND_VALIDATION_ACCURACY_RE = re.compile(
    r"Round\s+(\d+):\s+Global Val Accuracy\s*=\s*([0-9.eE+-]+)"
)
ROUND_TIMING_RE = re.compile(
    r"Round\s+(\d+):\s+Training Time\s*=\s*([0-9.eE+-]+)s,\s+"
    r"Communication Time\s*=\s*([0-9.eE+-]+)s"
)
FINAL_TEST_ACCURACY_RES = [
    re.compile(r"Average test accuracy,\s*([0-9.eE+-]+)"),
    re.compile(r"Final test accuracy:\s*([0-9.eE+-]+)"),
]
FINAL_TEST_LOSS_RES = [
    re.compile(r"average_final_test_loss,\s*([0-9.eE+-]+)"),
    re.compile(r"Final test loss:\s*([0-9.eE+-]+)"),
]
PURE_TRAINING_TIME_RE = re.compile(
    r"Total Pure Training Time .*:\s*([0-9.eE+-]+)\s*seconds"
)
COMMUNICATION_TIME_RE = re.compile(
    r"Total Communication Time .*:\s*([0-9.eE+-]+)\s*seconds"
)
TRAIN_COMM_TIME_RE = re.compile(
    r"Total Training \+ Communication Time:\s*([0-9.eE+-]+)\s*seconds"
)


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    method: str
    batch_size: int
    seed: int
    global_rounds: int
    local_step: int
    learning_rate: float
    n_trainer: int
    iid_beta: float
    distribution_type: str
    num_layers: int
    num_hops: int
    gpu: bool
    num_cpus_per_trainer: int
    num_gpus_per_trainer: float
    use_ogb_load_patch: bool


class Tee:
    """Write text to several streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def fileno(self):
        # Libraries such as Ray's fault handler expect stdout/stderr to expose
        # the underlying terminal file descriptor.
        return self.streams[0].fileno()

    def isatty(self):
        return self.streams[0].isatty()


@contextlib.contextmanager
def patched_torch_load_for_ogb(enabled):
    """Allow PyG/OGB processed data objects to load on newer PyTorch versions."""
    if not enabled:
        yield
        return

    import torch

    original_torch_load = torch.load

    def torch_load_with_pyg_objects(*args, **kwargs):
        # PyTorch 2.6 defaults weights_only=True; OGB/PyG processed data contains
        # graph objects, so this benchmark opts into loading the full object.
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = torch_load_with_pyg_objects
    try:
        yield
    finally:
        torch.load = original_torch_load


def parse_int_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_float_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def sanitize_experiment_name(raw: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", raw.strip()).strip("-._")
    if not name:
        raise ValueError("Experiment name must contain at least one letter or number.")
    return name


def batch_sizes_slug(raw_batch_sizes: str) -> str:
    return "-".join(batch_size_slug(value) for value in parse_int_list(raw_batch_sizes))


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def default_learning_rate(dataset: str) -> float:
    if dataset.startswith("ogbn"):
        return 0.01
    return 0.5


def method_for_num_hops(num_hops: int, requested_method: Optional[str]) -> str:
    if requested_method:
        return requested_method
    # FedGraph switches 0-hop NC training to FedAvg internally; keeping the
    # benchmark label aligned makes output files easier to compare.
    return "FedGCN" if num_hops > 0 else "FedAvg"


def batch_size_slug(batch_size: int) -> str:
    return "full" if batch_size == -1 else str(batch_size)


def make_run_id(config: ExperimentConfig) -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    beta = str(config.iid_beta).replace(".", "p")
    return (
        f"{timestamp}_{config.dataset}_{config.method}_bs{batch_size_slug(config.batch_size)}"
        f"_seed{config.seed}_beta{beta}"
    )


def set_seed(seed: int) -> None:
    # Keep partitioning/model initialization as reproducible as the underlying
    # Ray/PyTorch operations allow.
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def command_output(command: List[str]) -> Optional[str]:
    try:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def git_metadata() -> dict:
    return {
        "branch": command_output(["git", "branch", "--show-current"]),
        "commit": command_output(["git", "rev-parse", "HEAD"]),
        "status_short": command_output(["git", "status", "--short"]),
    }


def environment_metadata() -> dict:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": str(Path.cwd()),
    }


def default_experiment_name(args) -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return sanitize_experiment_name(
        f"{timestamp}_{args.dataset}_bs{batch_sizes_slug(args.batch_sizes)}"
        f"_r{args.rounds}_ls{args.local_step}_clients{args.n_trainer}"
    )


def resolve_output_dir(args) -> Path:
    if args.output_dir is not None:
        experiment_name = args.experiment_name or args.output_dir.name
        output_dir = args.output_dir
    else:
        experiment_name = args.experiment_name or default_experiment_name(args)
        experiment_name = sanitize_experiment_name(experiment_name)
        output_dir = args.output_root / experiment_name

    args.experiment_name = sanitize_experiment_name(experiment_name)
    args.resolved_output_dir = output_dir
    return output_dir


def ensure_output_dir_available(output_dir: Path, allow_existing: bool) -> None:
    if allow_existing:
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"Output directory already contains files: {output_dir}\n"
            "Use a different --experiment-name or pass --allow-existing-output-dir "
            "if you intentionally want to append to this directory."
        )


def write_run_plan(output_dir: Path, configs: List[ExperimentConfig], args) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "run_plan.csv"
    fieldnames = ["run_index"] + list(asdict(configs[0]).keys()) if configs else []
    with plan_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for idx, config in enumerate(configs, start=1):
                writer.writerow({"run_index": idx, **asdict(config)})

    manifest = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_name": args.experiment_name,
        "command": sys.argv,
        "args": json_safe(vars(args)),
        "run_count": len(configs),
        "git": git_metadata(),
        "environment": environment_metadata(),
        "outputs": {
            "run_plan": str(plan_path),
            "summary": str(output_dir / "summary.csv"),
            "summary_jsonl": str(output_dir / "summary.jsonl"),
            "validation_curves": str(output_dir / "validation_curves.csv"),
            "round_metrics": str(output_dir / "round_metrics.csv"),
            "runs_dir": str(output_dir / "runs"),
        },
        "configs": [asdict(config) for config in configs],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def to_fedgraph_args(config: ExperimentConfig, logdir: Path):
    # Import lazily so --help and --dry-run work in lightweight environments
    # where training dependencies have not been installed yet.
    import attridict

    return attridict.AttriDict(
        {
            "fedgraph_task": "NC",
            "dataset": config.dataset,
            "method": config.method,
            "iid_beta": config.iid_beta,
            "distribution_type": config.distribution_type,
            "global_rounds": config.global_rounds,
            "local_step": config.local_step,
            "learning_rate": config.learning_rate,
            "n_trainer": config.n_trainer,
            "batch_size": config.batch_size,
            "num_layers": config.num_layers,
            "num_hops": config.num_hops,
            "gpu": config.gpu,
            "num_cpus_per_trainer": config.num_cpus_per_trainer,
            "num_gpus_per_trainer": config.num_gpus_per_trainer,
            "logdir": str(logdir),
            "use_encryption": False,
            "use_huggingface": False,
            "saveto_huggingface": False,
            "use_cluster": False,
            "use_lowrank": False,
            "use_dp": False,
        }
    )


def parse_metric(regexes: Iterable[re.Pattern], text: str) -> Optional[float]:
    for regex in regexes:
        matches = regex.findall(text)
        if matches:
            return float(matches[-1])
    return None


def parse_run_log(log_path: Path):
    # Each FedGraph run gets its own stdout log; parsing here avoids changing
    # the core training API just for this benchmark.
    text = log_path.read_text(encoding="utf-8", errors="replace")
    val_acc_by_round = {
        int(round_id): float(accuracy)
        for round_id, accuracy in ROUND_VALIDATION_ACCURACY_RE.findall(text)
    }
    val_loss_by_round = {
        int(round_id): float(loss)
        for round_id, loss in ROUND_VALIDATION_LOSS_RE.findall(text)
    }
    round_timing = {
        int(round_id): {
            "train_time_sec": float(training_time),
            "comm_time_sec": float(communication_time),
        }
        for round_id, training_time, communication_time in ROUND_TIMING_RE.findall(text)
    }
    cumulative_time = 0.0
    round_metrics = []
    for round_id in sorted(
        set(round_timing) | set(val_acc_by_round) | set(val_loss_by_round)
    ):
        timing = round_timing.get(round_id, {})
        round_time = timing.get("train_time_sec", 0.0) + timing.get(
            "comm_time_sec", 0.0
        )
        cumulative_time += round_time
        round_metrics.append(
            {
                "round": round_id,
                "val_acc": val_acc_by_round.get(round_id),
                "val_loss": val_loss_by_round.get(round_id),
                "train_comm_time_sec": round_time,
                "cum_train_comm_time_sec": cumulative_time,
                **timing,
            }
        )

    return {
        "round_metrics": round_metrics,
        "test_acc_final": parse_metric(FINAL_TEST_ACCURACY_RES, text),
        "test_loss_final": parse_metric(FINAL_TEST_LOSS_RES, text),
        "total_pure_train_time_sec": parse_metric([PURE_TRAINING_TIME_RE], text),
        "total_comm_time_sec": parse_metric([COMMUNICATION_TIME_RE], text),
        "total_train_comm_time_sec": parse_metric([TRAIN_COMM_TIME_RE], text),
    }


def finite_round_values(round_metrics: List[dict], key: str) -> List[dict]:
    return [item for item in round_metrics if item.get(key) is not None]


def best_round(
    round_metrics: List[dict], key: str, mode: str
) -> tuple[Optional[float], Optional[int]]:
    values = finite_round_values(round_metrics, key)
    if not values:
        return None, None
    selector = min if mode == "min" else max
    best = selector(values, key=lambda item: item[key])
    return best[key], best["round"]


def plateau_round(
    round_metrics: List[dict], key: str, window: int, tolerance: float
) -> Optional[int]:
    values = finite_round_values(round_metrics, key)
    if window <= 1 or len(values) < window:
        return None
    for end_idx in range(window - 1, len(values)):
        recent = values[end_idx - window + 1 : end_idx + 1]
        recent_values = [item[key] for item in recent]
        if max(recent_values) - min(recent_values) <= tolerance:
            return recent[-1]["round"]
    return None


def end_window_stats(
    round_metrics: List[dict], key: str, window: int, tolerance: float
) -> dict:
    values = finite_round_values(round_metrics, key)
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


def round_metric_value(
    round_metrics: List[dict], round_id: Optional[int], key: str
) -> Optional[float]:
    if round_id is None:
        return None
    for item in round_metrics:
        if item["round"] == round_id:
            return item.get(key)
    return None


def write_csv_row(path: Path, fieldnames: List[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_validation_curve_rows(path: Path, rows: List[dict]) -> None:
    fieldnames = [
        "experiment_name",
        "run_id",
        "dataset",
        "method",
        "batch_size",
        "seed",
        "iid_beta",
        "round",
        "val_acc",
        "val_loss",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def write_round_metric_rows(path: Path, rows: List[dict]) -> None:
    fieldnames = [
        "experiment_name",
        "run_id",
        "dataset",
        "method",
        "batch_size",
        "seed",
        "iid_beta",
        "round",
        "val_acc",
        "val_loss",
        "local_steps",
        "train_time_sec",
        "comm_time_sec",
        "train_comm_time_sec",
        "cum_train_comm_time_sec",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    config: ExperimentConfig,
    experiment_name: str,
    output_dir: Path,
    convergence_window: int,
    val_loss_tolerance: float,
    val_acc_tolerance: float,
) -> dict:
    run_id = make_run_id(config)
    run_dir = output_dir / "runs" / run_id
    log_dir = run_dir / "fedgraph_logs"
    log_path = run_dir / "stdout.log"
    config_path = run_dir / "config.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path.write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    start_time = time.time()
    start_timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    status = "completed"
    error = ""

    with log_path.open("w", encoding="utf-8") as log_file:
        tee_stdout = Tee(sys.stdout, log_file)
        tee_stderr = Tee(sys.stderr, log_file)
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(
            tee_stderr
        ):
            print("=" * 80)
            print(f"Starting FedGraph NC batch-size convergence run: {run_id}")
            print(json.dumps(asdict(config), indent=2, sort_keys=True))
            print("=" * 80)

            try:
                # Import FedGraph only when a real training run starts. Keeping
                # the import inside the captured block records dependency
                # failures in the run summary instead of losing the whole sweep.
                from fedgraph.federated_methods import run_fedgraph

                set_seed(config.seed)
                with patched_torch_load_for_ogb(config.use_ogb_load_patch):
                    run_fedgraph(to_fedgraph_args(config, log_dir))
            except Exception as exc:
                status = "failed"
                error = repr(exc)
                traceback.print_exc()
            finally:
                # The script runs many configurations in one process, so each
                # run must leave Ray in a clean state for the next one.
                try:
                    import ray

                    if ray.is_initialized():
                        ray.shutdown()
                except ImportError:
                    pass

    end_timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    duration_sec = time.time() - start_time
    parsed = parse_run_log(log_path)
    round_metrics = parsed["round_metrics"]

    final_round = round_metrics[-1]["round"] if round_metrics else None
    val_acc_final = round_metric_value(round_metrics, final_round, "val_acc")
    val_loss_final = round_metric_value(round_metrics, final_round, "val_loss")
    val_acc_best, val_acc_best_round = best_round(round_metrics, "val_acc", "max")
    val_loss_best, val_loss_best_round = best_round(round_metrics, "val_loss", "min")
    val_loss_plateau_round = plateau_round(
        round_metrics, "val_loss", convergence_window, val_loss_tolerance
    )
    val_acc_plateau_round = plateau_round(
        round_metrics, "val_acc", convergence_window, val_acc_tolerance
    )
    val_convergence_round = val_loss_plateau_round
    val_loss_end_stats = end_window_stats(
        round_metrics, "val_loss", convergence_window, val_loss_tolerance
    )
    val_acc_end_stats = end_window_stats(
        round_metrics, "val_acc", convergence_window, val_acc_tolerance
    )

    summary = {
        "experiment_name": experiment_name,
        "run_id": run_id,
        "dataset": config.dataset,
        "method": config.method,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "global_rounds": config.global_rounds,
        "local_step": config.local_step,
        "learning_rate": config.learning_rate,
        "n_trainer": config.n_trainer,
        "iid_beta": config.iid_beta,
        "distribution_type": config.distribution_type,
        "num_layers": config.num_layers,
        "num_hops": config.num_hops,
        "gpu": config.gpu,
        "status": status,
        "start_time_utc": start_timestamp,
        "end_time_utc": end_timestamp,
        "duration_sec": round(duration_sec, 3),
        "rounds_recorded": len(round_metrics),
        "test_acc_final": parsed["test_acc_final"],
        "test_loss_final": parsed["test_loss_final"],
        "val_acc_final": val_acc_final,
        "val_loss_final": val_loss_final,
        "val_acc_best": val_acc_best,
        "val_acc_best_round": val_acc_best_round,
        "val_loss_best": val_loss_best,
        "val_loss_best_round": val_loss_best_round,
        "val_loss_plateau_round": val_loss_plateau_round,
        "val_acc_plateau_round": val_acc_plateau_round,
        "val_convergence_round": val_convergence_round,
        "val_convergence_local_steps": (
            val_convergence_round * config.local_step
            if val_convergence_round is not None
            else None
        ),
        "val_convergence_time_sec": round_metric_value(
            round_metrics,
            val_convergence_round,
            "cum_train_comm_time_sec",
        ),
        "convergence_window": convergence_window,
        "val_loss_tolerance": val_loss_tolerance,
        "val_acc_tolerance": val_acc_tolerance,
        **val_loss_end_stats,
        **val_acc_end_stats,
        "total_pure_train_time_sec": parsed["total_pure_train_time_sec"],
        "total_comm_time_sec": parsed["total_comm_time_sec"],
        "total_train_comm_time_sec": parsed["total_train_comm_time_sec"],
        "log_path": str(log_path),
        "config_path": str(config_path),
        "error": error,
    }

    validation_curve_rows = [
        {
            "experiment_name": experiment_name,
            "run_id": run_id,
            "dataset": config.dataset,
            "method": config.method,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "iid_beta": config.iid_beta,
            "round": item["round"],
            "val_acc": item.get("val_acc"),
            "val_loss": item.get("val_loss"),
        }
        for item in round_metrics
        if item.get("val_acc") is not None or item.get("val_loss") is not None
    ]
    round_metric_rows = [
        {
            "experiment_name": experiment_name,
            "run_id": run_id,
            "dataset": config.dataset,
            "method": config.method,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "iid_beta": config.iid_beta,
            "round": item["round"],
            "val_acc": item.get("val_acc"),
            "val_loss": item.get("val_loss"),
            "local_steps": item["round"] * config.local_step,
            "train_time_sec": item.get("train_time_sec"),
            "comm_time_sec": item.get("comm_time_sec"),
            "train_comm_time_sec": item.get("train_comm_time_sec"),
            "cum_train_comm_time_sec": item.get("cum_train_comm_time_sec"),
        }
        for item in round_metrics
    ]
    write_validation_curve_rows(
        output_dir / "validation_curves.csv", validation_curve_rows
    )
    write_round_metric_rows(output_dir / "round_metrics.csv", round_metric_rows)

    summary_fields = [
        "experiment_name",
        "run_id",
        "dataset",
        "method",
        "batch_size",
        "seed",
        "global_rounds",
        "local_step",
        "learning_rate",
        "n_trainer",
        "iid_beta",
        "distribution_type",
        "num_layers",
        "num_hops",
        "gpu",
        "status",
        "start_time_utc",
        "end_time_utc",
        "duration_sec",
        "rounds_recorded",
        "test_acc_final",
        "test_loss_final",
        "val_acc_final",
        "val_loss_final",
        "val_acc_best",
        "val_acc_best_round",
        "val_loss_best",
        "val_loss_best_round",
        "val_loss_plateau_round",
        "val_acc_plateau_round",
        "val_convergence_round",
        "val_convergence_local_steps",
        "val_convergence_time_sec",
        "convergence_window",
        "val_loss_tolerance",
        "val_acc_tolerance",
        "val_loss_end_gain",
        "val_loss_end_range",
        "val_loss_flat_at_end",
        "val_acc_end_gain",
        "val_acc_end_range",
        "val_acc_flat_at_end",
        "total_pure_train_time_sec",
        "total_comm_time_sec",
        "total_train_comm_time_sec",
        "log_path",
        "config_path",
        "error",
    ]
    write_csv_row(output_dir / "summary.csv", summary_fields, summary)
    with (output_dir / "summary.jsonl").open("a", encoding="utf-8") as jsonl_file:
        jsonl_file.write(json.dumps(summary, sort_keys=True) + "\n")

    return summary


def build_configs(args) -> List[ExperimentConfig]:
    # Build the Cartesian product of the requested experiment axes.
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else default_learning_rate(args.dataset)
    )
    method = method_for_num_hops(args.num_hops, args.method)
    use_ogb_load_patch = (
        args.dataset.startswith("ogbn") and not args.disable_ogb_load_patch
    )

    configs = []
    for iid_beta in parse_float_list(args.iid_betas):
        for seed in parse_int_list(args.seeds):
            for batch_size in parse_int_list(args.batch_sizes):
                configs.append(
                    ExperimentConfig(
                        dataset=args.dataset,
                        method=method,
                        batch_size=batch_size,
                        seed=seed,
                        global_rounds=args.rounds,
                        local_step=args.local_step,
                        learning_rate=learning_rate,
                        n_trainer=args.n_trainer,
                        iid_beta=iid_beta,
                        distribution_type=args.distribution_type,
                        num_layers=args.num_layers,
                        num_hops=args.num_hops,
                        gpu=args.gpu,
                        num_cpus_per_trainer=args.num_cpus_per_trainer,
                        num_gpus_per_trainer=args.num_gpus_per_trainer,
                        use_ogb_load_patch=use_ogb_load_patch,
                    )
                )

    if args.max_runs is not None:
        return configs[: args.max_runs]
    return configs


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run long FedGraph node-classification convergence experiments over "
            "different batch sizes."
        )
    )
    parser.add_argument("--dataset", default="ogbn-arxiv")
    parser.add_argument("--batch-sizes", default="512,1024,2048,-1")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--iid-betas", default="10000")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--local-step", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--n-trainer", type=int, default=5)
    parser.add_argument("--distribution-type", default="average")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-hops", type=int, default=0)
    parser.add_argument("--method", default=None)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--num-cpus-per-trainer", type=int, default=1)
    parser.add_argument("--num-gpus-per-trainer", type=float, default=0.0)
    parser.add_argument(
        "--output-root",
        default="benchmark/results/nc_batch_size_convergence",
        type=Path,
        help=(
            "Root directory for named experiments. The final output path is "
            "--output-root/--experiment-name unless --output-dir is provided."
        ),
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help=(
            "Human-readable experiment label used as the output subdirectory. "
            "If omitted, a timestamped name is generated."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help=(
            "Exact output directory override. Prefer --experiment-name for normal "
            "runs so different experiments do not mix."
        ),
    )
    parser.add_argument("--allow-existing-output-dir", action="store_true")
    parser.add_argument("--convergence-window", type=int, default=10)
    parser.add_argument("--val-loss-tolerance", type=float, default=0.01)
    parser.add_argument("--val-acc-tolerance", type=float, default=0.0025)
    parser.add_argument(
        "--convergence-tolerance",
        type=float,
        default=None,
        help=(
            "Compatibility alias for --val-acc-tolerance. Validation loss "
            "uses --val-loss-tolerance."
        ),
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--disable-ogb-load-patch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.convergence_tolerance is not None:
        args.val_acc_tolerance = args.convergence_tolerance
    output_dir = resolve_output_dir(args)
    configs = build_configs(args)
    ensure_output_dir_available(output_dir, args.allow_existing_output_dir)

    print(f"Experiment name: {args.experiment_name}")
    print(f"Output directory: {output_dir}")
    print(f"Prepared {len(configs)} FedGraph NC convergence run(s).")
    for idx, config in enumerate(configs, start=1):
        print(
            f"{idx:02d}. dataset={config.dataset}, method={config.method}, "
            f"batch_size={config.batch_size}, seed={config.seed}, "
            f"rounds={config.global_rounds}, lr={config.learning_rate}"
        )

    write_run_plan(output_dir, configs, args)

    if args.dry_run:
        print("Dry run only; wrote manifest/run plan and no training started.")
        return 0

    os.environ.setdefault("PYTHONHASHSEED", str(configs[0].seed if configs else 42))

    for idx, config in enumerate(configs, start=1):
        print(f"\nRunning experiment {idx}/{len(configs)}")
        summary = run_experiment(
            config=config,
            experiment_name=args.experiment_name,
            output_dir=output_dir,
            convergence_window=args.convergence_window,
            val_loss_tolerance=args.val_loss_tolerance,
            val_acc_tolerance=args.val_acc_tolerance,
        )
        print(
            f"Finished {summary['run_id']} with status={summary['status']}, "
            f"test_acc_final={summary['test_acc_final']}, "
            f"val_convergence_round={summary['val_convergence_round']}"
        )
        if summary["status"] != "completed" and not args.continue_on_error:
            return 1

    print(f"\nResults written to {output_dir}")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    print(f"Run plan CSV: {output_dir / 'run_plan.csv'}")
    print(f"Summary CSV: {output_dir / 'summary.csv'}")
    print(f"Validation curves CSV: {output_dir / 'validation_curves.csv'}")
    print(
        "Plot command: "
        f"python benchmark/plot_NC_batch_size_convergence.py {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
