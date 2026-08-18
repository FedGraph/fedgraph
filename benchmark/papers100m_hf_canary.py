#!/usr/bin/env python3
"""Run one legacy Hugging Face Papers100M shard through the FedAvg train path.

This is a capacity canary, not a federated benchmark. It deliberately creates
one Trainer_General instance so a ten-trainer artifact does not trigger ten
downloads or ten GPU allocations.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from fedgraph.trainer_class import Trainer_General


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load one legacy Papers100M Hf shard and exercise GPU training."
    )
    parser.add_argument("--trainer-id", type=int, default=0)
    parser.add_argument("--n-trainer", type=int, default=10)
    parser.add_argument("--iid-beta", type=float, default=10000.0)
    parser.add_argument("--hf-artifact-num-hops", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-steps", type=int, default=1)
    parser.add_argument("--global-rounds", type=int, default=1)
    parser.add_argument(
        "--run-test",
        action="store_true",
        help="Run mini-batch test evaluation after each simulated global round.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--global-node-num", type=int, default=111_059_956)
    parser.add_argument("--class-num", type=int, default=172)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def _gib_from_kib(value_kib: int) -> float:
    return round(value_kib / 1024**2, 3)


def _host_memory_snapshot() -> dict[str, float]:
    """Return process and host memory figures from Linux kernel accounting."""
    meminfo: dict[str, int] = {}
    with open("/proc/meminfo", encoding="utf-8") as meminfo_file:
        for line in meminfo_file:
            key, value = line.split(":", maxsplit=1)
            meminfo[key] = int(value.split()[0])

    process_rss_kib = 0
    with open("/proc/self/status", encoding="utf-8") as status_file:
        for line in status_file:
            if line.startswith("VmRSS:"):
                process_rss_kib = int(line.split()[1])
                break

    return {
        "host_process_rss_gib": _gib_from_kib(process_rss_kib),
        "host_process_max_rss_gib": _gib_from_kib(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
        "host_mem_total_gib": _gib_from_kib(meminfo["MemTotal"]),
        "host_mem_available_gib": _gib_from_kib(meminfo["MemAvailable"]),
        "host_mem_free_gib": _gib_from_kib(meminfo["MemFree"]),
        "host_cached_gib": _gib_from_kib(meminfo["Cached"]),
        "host_swap_total_gib": _gib_from_kib(meminfo["SwapTotal"]),
        "host_swap_free_gib": _gib_from_kib(meminfo["SwapFree"]),
    }


def _memory_snapshot(stage: str) -> dict[str, Any]:
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "stage": stage,
        "elapsed_sec": round(time.monotonic() - _START_TIME, 3),
        **_host_memory_snapshot(),
        "cuda_allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "cuda_reserved_gib": round(torch.cuda.memory_reserved() / 1024**3, 3),
        "cuda_peak_allocated_gib": round(
            torch.cuda.max_memory_allocated() / 1024**3, 3
        ),
        "cuda_peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "cuda_free_gib": round(free_bytes / 1024**3, 3),
        "cuda_total_gib": round(total_bytes / 1024**3, 3),
    }


def _record_snapshot(result: dict[str, Any], stage: str) -> None:
    snapshot = _memory_snapshot(stage)
    result["memory"].append(snapshot)
    print(json.dumps(snapshot, sort_keys=True), flush=True)


def _write_result(result: dict[str, Any], output_json: Path | None) -> None:
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded, flush=True)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(encoded + "\n", encoding="utf-8")


def main() -> int:
    cli = _parse_args()
    if not os.environ.get("HF_HOME"):
        raise RuntimeError("Set HF_HOME to the canary instance's local NVMe cache path")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; verify the NVIDIA driver and PyTorch wheel"
        )

    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    trainer_args = SimpleNamespace(
        dataset="ogbn-papers100M",
        n_trainer=cli.n_trainer,
        iid_beta=cli.iid_beta,
        num_hops=0,
        hf_artifact_num_hops=cli.hf_artifact_num_hops,
        use_huggingface=True,
        method="FedAvg",
        local_step=cli.local_steps,
        num_layers=cli.num_layers,
        batch_size=cli.batch_size,
        learning_rate=cli.learning_rate,
        seed=cli.seed,
    )
    result: dict[str, Any] = {
        "status": "started",
        "dataset": trainer_args.dataset,
        "trainer_id": cli.trainer_id,
        "n_trainer": cli.n_trainer,
        "hf_artifact_num_hops": cli.hf_artifact_num_hops,
        "batch_size": cli.batch_size,
        "local_steps": cli.local_steps,
        "global_rounds": cli.global_rounds,
        "run_test": cli.run_test,
        "hf_home": os.environ["HF_HOME"],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "memory": [],
    }

    try:
        _record_snapshot(result, "before_trainer_load")
        trainer = Trainer_General(
            rank=cli.trainer_id,
            args_hidden=cli.hidden,
            device=torch.device("cuda:0"),
            args=trainer_args,
        )
        result.update(
            local_node_count=int(trainer.features.size(0)),
            local_adjacency_edge_count=int(trainer.adj.size(1)),
            local_adjacency_gib=round(
                trainer.adj.numel() * trainer.adj.element_size() / 1024**3, 3
            ),
        )
        _record_snapshot(result, "after_load_transfer_and_relabel")

        trainer.init_model(
            global_node_num=cli.global_node_num,
            class_num=cli.class_num,
        )
        _record_snapshot(result, "after_model_initialization")

        result["rounds"] = []
        for global_round in range(cli.global_rounds):
            torch.cuda.reset_peak_memory_stats()
            trainer.train(current_global_round=global_round)
            _record_snapshot(result, f"after_global_round_{global_round + 1}_train")
            round_result: dict[str, Any] = {
                "global_round": global_round + 1,
                "train_loss": float(trainer.train_losses[-1]),
                "train_accuracy": float(trainer.train_accs[-1]),
            }
            if cli.run_test:
                torch.cuda.reset_peak_memory_stats()
                local_test_loss, local_test_accuracy = trainer.local_test()
                _record_snapshot(result, f"after_global_round_{global_round + 1}_test")
                round_result.update(
                    test_loss=float(local_test_loss),
                    test_accuracy=float(local_test_accuracy),
                )
            result["rounds"].append(round_result)

        result.update(
            status="completed",
            train_loss=float(trainer.train_losses[-1]),
            train_accuracy=float(trainer.train_accs[-1]),
        )
        if cli.run_test:
            result.update(
                test_loss=float(trainer.test_losses[-1]),
                test_accuracy=float(trainer.test_accs[-1]),
            )
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        try:
            _record_snapshot(result, "after_failure")
        except Exception as memory_error:  # CUDA can be unusable after an OOM.
            result["memory_snapshot_error"] = repr(memory_error)
        _write_result(result, cli.output_json)
        raise

    _write_result(result, cli.output_json)
    return 0


if __name__ == "__main__":
    _START_TIME = time.monotonic()
    sys.exit(main())
