"""Small, opt-in resource snapshots for long-running FedGraph experiments."""

from __future__ import annotations

import datetime as dt
import json
import os
import resource
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional

import torch


def tensor_nbytes(value: Any) -> int:
    """Return the logical storage size of one tensor without copying it."""
    if not isinstance(value, torch.Tensor):
        return 0
    return int(value.numel() * value.element_size())


def model_parameter_nbytes(model: Optional[torch.nn.Module]) -> int:
    if model is None:
        return 0
    return sum(tensor_nbytes(parameter) for parameter in model.parameters())


def model_gradient_nbytes(model: Optional[torch.nn.Module]) -> int:
    if model is None:
        return 0
    return sum(
        tensor_nbytes(parameter.grad)
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def _nested_tensor_nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return tensor_nbytes(value)
    if isinstance(value, Mapping):
        return sum(_nested_tensor_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_nested_tensor_nbytes(item) for item in value)
    return 0


def optimizer_state_nbytes(optimizer: Optional[torch.optim.Optimizer]) -> int:
    if optimizer is None:
        return 0
    return sum(_nested_tensor_nbytes(state) for state in optimizer.state.values())


def _meminfo_bytes() -> dict[str, Optional[int]]:
    fields = {
        "MemTotal": "host_mem_total_bytes",
        "MemAvailable": "host_mem_available_bytes",
        "MemFree": "host_mem_free_bytes",
        "Cached": "host_mem_cached_bytes",
        "SwapTotal": "host_swap_total_bytes",
        "SwapFree": "host_swap_free_bytes",
    }
    values: dict[str, Optional[int]] = {name: None for name in fields.values()}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw_value = line.split(":", maxsplit=1)
            output_name = fields.get(key)
            if output_name is None:
                continue
            values[output_name] = int(raw_value.split()[0]) * 1024
    except (FileNotFoundError, OSError, ValueError):
        pass
    return values


def _process_memory_bytes() -> dict[str, Optional[int]]:
    rss_bytes: Optional[int] = None
    try:
        rss_pages = int(Path("/proc/self/statm").read_text().split()[1])
        rss_bytes = rss_pages * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, OSError, ValueError, IndexError):
        pass

    peak_rss_bytes: Optional[int] = None
    try:
        peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            peak_rss_bytes *= 1024
    except (AttributeError, ValueError):
        pass
    return {
        "process_rss_bytes": rss_bytes,
        "process_peak_rss_bytes": peak_rss_bytes,
    }


def reset_cuda_peak_memory(device: Optional[torch.device]) -> None:
    """Reset PyTorch allocator peaks without synchronizing or freeing tensors."""
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except RuntimeError:
        # A failed CUDA context should not turn observability into a new failure.
        return


def collect_resource_snapshot(
    *,
    source: str,
    event: str,
    round_id: Optional[int] = None,
    trainer_id: Optional[int] = None,
    device: Optional[torch.device] = None,
    tensors: Optional[Mapping[str, Any]] = None,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict[str, Any]:
    """Collect allocation metadata without changing tensor placement or execution."""
    snapshot: dict[str, Any] = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": source,
        "event": event,
        "round": round_id,
        "trainer_id": trainer_id,
        "pid": os.getpid(),
        **_process_memory_bytes(),
        **_meminfo_bytes(),
        "model_parameter_bytes": model_parameter_nbytes(model),
        "model_gradient_bytes": model_gradient_nbytes(model),
        "optimizer_state_bytes": optimizer_state_nbytes(optimizer),
        "cuda_memory_allocated_bytes": None,
        "cuda_memory_reserved_bytes": None,
        "cuda_max_memory_allocated_bytes": None,
        "cuda_max_memory_reserved_bytes": None,
        "cuda_free_bytes": None,
        "cuda_total_bytes": None,
    }
    for name, tensor in (tensors or {}).items():
        snapshot[f"{name}_bytes"] = tensor_nbytes(tensor)

    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            snapshot.update(
                {
                    "cuda_memory_allocated_bytes": torch.cuda.memory_allocated(device),
                    "cuda_memory_reserved_bytes": torch.cuda.memory_reserved(device),
                    "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(
                        device
                    ),
                    "cuda_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(
                        device
                    ),
                    "cuda_free_bytes": free_bytes,
                    "cuda_total_bytes": total_bytes,
                }
            )
        except RuntimeError:
            pass
    return snapshot


def append_resource_snapshot(path: Path, snapshot: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as snapshot_file:
        snapshot_file.write(json.dumps(dict(snapshot), sort_keys=True) + "\n")


def write_resource_snapshot_summary(snapshot_path: Path) -> Optional[Path]:
    """Write a compact peak summary next to a JSONL snapshot stream."""
    if not snapshot_path.exists():
        return None

    peak_fields = {
        "process_rss_bytes",
        "process_peak_rss_bytes",
        "cuda_memory_allocated_bytes",
        "cuda_memory_reserved_bytes",
        "cuda_max_memory_allocated_bytes",
        "cuda_max_memory_reserved_bytes",
        "model_parameter_bytes",
        "model_gradient_bytes",
        "optimizer_state_bytes",
        "features_bytes",
        "adjacency_bytes",
        "feature_aggregation_bytes",
    }
    minimum_fields = {
        "host_mem_available_bytes",
        "host_mem_free_bytes",
        "cuda_free_bytes",
    }
    groups: dict[tuple[Any, Any], dict[str, Any]] = {}
    event_counts: Counter[str] = Counter()
    total_snapshots = 0

    with snapshot_path.open(encoding="utf-8") as snapshot_file:
        for line in snapshot_file:
            if not line.strip():
                continue
            snapshot = json.loads(line)
            total_snapshots += 1
            event_counts[str(snapshot.get("event"))] += 1
            key = (snapshot.get("source"), snapshot.get("trainer_id"))
            group = groups.setdefault(
                key,
                {
                    "source": snapshot.get("source"),
                    "trainer_id": snapshot.get("trainer_id"),
                    "snapshot_count": 0,
                },
            )
            group["snapshot_count"] += 1
            for field in peak_fields:
                value = snapshot.get(field)
                if isinstance(value, (int, float)):
                    group[field] = max(group.get(field, value), value)
            for field in minimum_fields:
                value = snapshot.get(field)
                if isinstance(value, (int, float)):
                    group[field] = min(group.get(field, value), value)

    summary_path = snapshot_path.with_name("resource_snapshot_summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "snapshot_count": total_snapshots,
                "event_counts": dict(event_counts),
                "sources": list(groups.values()),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return summary_path
