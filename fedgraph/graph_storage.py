"""Bounded, file-backed graph storage helpers."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_RELABEL_CACHE_VERSION = 1
_FEATURE_AGGREGATION_STORE_VERSION = 1
_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


@dataclass(frozen=True)
class RelabeledAdjacency:
    """Result of building or reopening one local-ID adjacency cache."""

    edge_index: torch.Tensor
    cache_path: Path
    cache_hit: bool
    source_edge_count: int
    edge_count: int
    dropped_edge_count: int
    elapsed_sec: float


class FeatureAggregationStore:
    """Ordered, atomic file-backed storage for one trainer's feature matrix."""

    def __init__(
        self,
        output_dir: Path,
        *,
        trainer_id: int,
        row_count: int,
        feature_dim: int,
    ) -> None:
        if trainer_id < 0:
            raise ValueError("trainer_id must be non-negative")
        if row_count < 0 or feature_dim < 1:
            raise ValueError("feature aggregation shape is invalid")

        trainer_dir = output_dir.expanduser().resolve() / f"trainer-{trainer_id:03d}"
        trainer_dir.mkdir(parents=True, exist_ok=True)
        self.row_count = row_count
        self.feature_dim = feature_dim
        self.written_rows = 0
        self.data_path = trainer_dir / "feature-aggregation.f32"
        self.metadata_path = trainer_dir / "feature-aggregation.json"
        temporary_suffix = f"tmp-{os.getpid()}-{time.time_ns()}"
        self.temporary_data_path = trainer_dir / (
            f"feature-aggregation.f32.{temporary_suffix}"
        )
        self.temporary_metadata_path = trainer_dir / (
            f"feature-aggregation.json.{temporary_suffix}"
        )
        self._storage: np.memmap | None = None
        self._finalized = False
        if row_count == 0:
            self.temporary_data_path.touch()
        else:
            self._storage = np.memmap(
                self.temporary_data_path,
                dtype=np.float32,
                mode="w+",
                shape=(row_count, feature_dim),
            )

    def write(self, start_row: int, values: torch.Tensor) -> None:
        """Append one contiguous row block at the expected output position."""
        if self._finalized:
            raise RuntimeError("feature aggregation store is already finalized")
        if start_row != self.written_rows:
            raise ValueError(
                "feature aggregation chunks must be written once in row order: "
                f"expected start {self.written_rows}, got {start_row}"
            )
        if values.device.type != "cpu":
            raise ValueError("feature aggregation chunks must be CPU tensors")
        if values.ndim != 2 or values.size(1) != self.feature_dim:
            raise ValueError("feature aggregation chunk has an invalid shape")
        stop_row = start_row + values.size(0)
        if stop_row > self.row_count:
            raise ValueError("feature aggregation chunk exceeds the output shape")
        if values.numel():
            if self._storage is None:
                raise RuntimeError("feature aggregation storage is unavailable")
            self._storage[start_row:stop_row] = (
                values.detach().float().contiguous().numpy()
            )
        self.written_rows = stop_row

    def finalize(self) -> torch.Tensor:
        """Atomically publish and reopen the complete feature matrix."""
        if self._finalized:
            raise RuntimeError("feature aggregation store is already finalized")
        if self.written_rows != self.row_count:
            raise RuntimeError(
                "feature aggregation store is incomplete: "
                f"{self.written_rows}/{self.row_count} rows"
            )
        if self._storage is not None:
            self._storage.flush()
            del self._storage
            self._storage = None

        metadata = {
            "store_version": _FEATURE_AGGREGATION_STORE_VERSION,
            "dtype": "float32",
            "layout": "row_major",
            "row_count": self.row_count,
            "feature_dim": self.feature_dim,
            "size_bytes": self.row_count
            * self.feature_dim
            * np.dtype(np.float32).itemsize,
        }
        self.temporary_metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(self.temporary_data_path, self.data_path)
        os.replace(self.temporary_metadata_path, self.metadata_path)
        self._finalized = True

        if self.row_count == 0:
            return torch.empty((0, self.feature_dim), dtype=torch.float32)
        return torch.from_file(
            str(self.data_path),
            shared=False,
            size=self.row_count * self.feature_dim,
            dtype=torch.float32,
        ).view(self.row_count, self.feature_dim)

    def abort(self) -> None:
        """Discard an unfinished temporary output without touching published data."""
        if self._storage is not None:
            self._storage.flush()
            del self._storage
            self._storage = None
        self.temporary_data_path.unlink(missing_ok=True)
        self.temporary_metadata_path.unlink(missing_ok=True)


def _hash_index_tensor(indexes: torch.Tensor, chunk_elements: int = 1_000_000) -> str:
    digest = hashlib.sha256()
    flat_indexes = indexes.detach().cpu().long().flatten()
    for start in range(0, flat_indexes.numel(), chunk_elements):
        chunk = flat_indexes[start : start + chunk_elements].contiguous()
        digest.update(chunk.numpy().tobytes())
    return digest.hexdigest()


def _cache_fingerprint(
    source_path: Path,
    source_edge_count: int,
    node_indexes: torch.Tensor,
) -> dict[str, object]:
    resolved_source = source_path.expanduser().resolve()
    source_stat = resolved_source.stat()
    return {
        "cache_version": _RELABEL_CACHE_VERSION,
        "source_path": str(resolved_source),
        "source_size_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "source_edge_count": source_edge_count,
        "node_index_count": node_indexes.numel(),
        "node_index_sha256": _hash_index_tensor(node_indexes),
    }


def _load_edge_index_cache(path: Path, edge_count: int) -> torch.Tensor:
    if edge_count == 0:
        return torch.empty((2, 0), dtype=torch.long)
    flat_edges = torch.from_file(
        str(path),
        shared=False,
        size=edge_count * 2,
        dtype=torch.int64,
    )
    return flat_edges.view(2, edge_count)


def _read_cached_result(
    data_path: Path,
    metadata_path: Path,
    fingerprint: dict[str, object],
    started_at: float,
) -> RelabeledAdjacency | None:
    if not data_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get("fingerprint") != fingerprint:
        return None

    edge_count = metadata.get("edge_count")
    dropped_edge_count = metadata.get("dropped_edge_count")
    source_edge_count = fingerprint.get("source_edge_count")
    if not isinstance(edge_count, int) or edge_count < 0:
        return None
    if not isinstance(dropped_edge_count, int) or dropped_edge_count < 0:
        return None
    if not isinstance(source_edge_count, int) or source_edge_count < 0:
        return None
    if data_path.stat().st_size != edge_count * 2 * np.dtype(np.int64).itemsize:
        return None

    return RelabeledAdjacency(
        edge_index=_load_edge_index_cache(data_path, edge_count),
        cache_path=data_path,
        cache_hit=True,
        source_edge_count=source_edge_count,
        edge_count=edge_count,
        dropped_edge_count=dropped_edge_count,
        elapsed_sec=time.perf_counter() - started_at,
    )


def relabel_adjacency_to_mmap_cache(
    adjacency: torch.Tensor,
    node_indexes: torch.Tensor,
    *,
    source_path: Path,
    cache_dir: Path,
    trainer_id: int,
    chunk_edges: int,
) -> RelabeledAdjacency:
    """Relabel a global-ID edge index using bounded memory and a disk cache.

    Edges with either endpoint outside ``node_indexes`` are omitted, matching
    the induced-subgraph behavior of the historical ``k_hop_subgraph`` call.
    The cache stores a contiguous row-major int64 ``[2, E]`` tensor and is
    reopened through ``torch.from_file`` without materializing it in RAM.
    """
    started_at = time.perf_counter()
    if adjacency.device.type != "cpu" or node_indexes.device.type != "cpu":
        raise ValueError("bounded adjacency relabeling requires CPU tensors")
    if adjacency.ndim != 2 or adjacency.size(0) != 2:
        raise ValueError("adjacency must have shape [2, num_edges]")
    if adjacency.dtype not in _INTEGER_DTYPES:
        raise ValueError("adjacency must use an integer dtype")
    if node_indexes.ndim != 1:
        raise ValueError("node_indexes must be one-dimensional")
    if chunk_edges <= 0:
        raise ValueError("chunk_edges must be positive")

    local_nodes = node_indexes.detach().cpu().long().flatten()
    if local_nodes.numel() > 1 and not bool(
        torch.all(local_nodes[1:] > local_nodes[:-1])
    ):
        raise ValueError("node_indexes must be sorted and unique")

    source_path = source_path.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"adjacency artifact is missing: {source_path}")
    source_edge_count = int(adjacency.size(1))
    fingerprint = _cache_fingerprint(
        source_path,
        source_edge_count,
        local_nodes,
    )
    cache_key = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    trainer_cache_dir = cache_dir.expanduser().resolve() / f"trainer-{trainer_id:03d}"
    trainer_cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = trainer_cache_dir / f"adj-local-{cache_key}.edge-index.i64"
    metadata_path = trainer_cache_dir / f"adj-local-{cache_key}.json"

    cached_result = _read_cached_result(
        data_path,
        metadata_path,
        fingerprint,
        started_at,
    )
    if cached_result is not None:
        return cached_result

    temporary_suffix = f"tmp-{os.getpid()}-{time.time_ns()}"
    temporary_data = data_path.with_name(f"{data_path.name}.{temporary_suffix}")
    temporary_metadata = metadata_path.with_name(
        f"{metadata_path.name}.{temporary_suffix}"
    )
    output_storage: np.memmap | None = None
    output_edge_count = 0
    try:
        if source_edge_count > 0:
            output_storage = np.memmap(
                temporary_data,
                dtype=np.int64,
                mode="w+",
                shape=(2, source_edge_count),
            )
            for start in range(0, source_edge_count, chunk_edges):
                stop = min(start + chunk_edges, source_edge_count)
                edge_chunk = adjacency[:, start:stop].detach().cpu().long().contiguous()
                if local_nodes.numel() == 0:
                    continue

                local_positions = torch.searchsorted(local_nodes, edge_chunk)
                in_range = local_positions < local_nodes.numel()
                matches = torch.zeros_like(in_range)
                matches[in_range] = (
                    local_nodes[local_positions[in_range]] == edge_chunk[in_range]
                )
                retained_edges = torch.all(matches, dim=0)
                retained_count = int(retained_edges.sum().item())
                if retained_count == 0:
                    continue

                next_output_count = output_edge_count + retained_count
                output_storage[:, output_edge_count:next_output_count] = (
                    local_positions[:, retained_edges].contiguous().numpy()
                )
                output_edge_count = next_output_count

            output_storage.flush()
            del output_storage
            output_storage = None
            if 0 < output_edge_count < source_edge_count:
                output_storage = np.memmap(
                    temporary_data,
                    dtype=np.int64,
                    mode="r+",
                    shape=(source_edge_count * 2,),
                )
                for start in range(0, output_edge_count, chunk_edges):
                    stop = min(start + chunk_edges, output_edge_count)
                    target_chunk = np.array(
                        output_storage[
                            source_edge_count + start : source_edge_count + stop
                        ],
                        copy=True,
                    )
                    output_storage[
                        output_edge_count + start : output_edge_count + stop
                    ] = target_chunk
                output_storage.flush()
                del output_storage
                output_storage = None
            os.truncate(
                temporary_data,
                output_edge_count * 2 * np.dtype(np.int64).itemsize,
            )
        else:
            temporary_data.touch()

        metadata = {
            "fingerprint": fingerprint,
            "layout": "row_major_int64_2_by_e",
            "edge_count": output_edge_count,
            "dropped_edge_count": source_edge_count - output_edge_count,
        }
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_data, data_path)
        os.replace(temporary_metadata, metadata_path)
    finally:
        if output_storage is not None:
            output_storage.flush()
            del output_storage
        temporary_data.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)

    return RelabeledAdjacency(
        edge_index=_load_edge_index_cache(data_path, output_edge_count),
        cache_path=data_path,
        cache_hit=False,
        source_edge_count=source_edge_count,
        edge_count=output_edge_count,
        dropped_edge_count=source_edge_count - output_edge_count,
        elapsed_sec=time.perf_counter() - started_at,
    )
