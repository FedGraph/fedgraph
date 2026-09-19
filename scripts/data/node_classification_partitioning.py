"""Streaming writers for 0-hop node-classification shard artifacts.

The normal FedGraph loader partitions small graphs in memory.  This module is
deliberately separate: it writes versioned, local 0-hop artifacts from the raw
OGB CSV layout without involving Ray or materializing a global PyG graph.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import zipfile
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Mapping, Optional

import numpy as np
import pandas as pd
import torch

_OWNER_UNASSIGNED = np.uint8(255)
_SPLIT_CODES = {"train": 1, "val": 2, "test": 3}
_SPLIT_FILE_NAMES = {"train": "train", "val": "valid", "test": "test"}


@dataclass(frozen=True)
class RawOGBNodeClassificationFiles:
    """Raw OGB CSV paths required by the 0-hop artifact writer."""

    dataset_root: Path
    features: Path
    labels: Path
    edges: Path
    num_nodes: Path
    num_edges: Path
    split_files: Mapping[str, Path]


@dataclass(frozen=True)
class OGBNodeClassificationSource:
    """A normalized CSV or memory-mappable OGB binary graph source."""

    dataset_root: Path
    input_format: str
    features: Path
    labels: Path
    edges: Path
    num_nodes: int
    num_edges: int
    feature_dim: int
    split_files: Mapping[str, Path]
    raw_files: Mapping[str, object]


@dataclass(frozen=True)
class PartitionPlan:
    """Persistent assignment state reused by a restarted shard materialization."""

    num_nodes: int
    feature_dim: int
    class_num: int
    n_trainer: int
    iid_beta: float
    seed: int
    shard_node_counts: list[int]
    labelled_node_count: int
    unlabelled_node_count: int


def _csv_path(directory: Path, stem: str) -> Path:
    for suffix in (".csv.gz", ".csv"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {stem}.csv[.gz] under {directory}")


def discover_raw_ogb_files(
    dataset_root: Path | str, split_name: str = "time"
) -> RawOGBNodeClassificationFiles:
    """Discover the documented raw OGB CSV layout without importing PyG."""
    root = Path(dataset_root).expanduser().resolve()
    raw = root / "raw"
    split_dir = root / "split" / split_name
    split_files = {
        split: _csv_path(split_dir, file_name)
        for split, file_name in _SPLIT_FILE_NAMES.items()
    }
    return RawOGBNodeClassificationFiles(
        dataset_root=root,
        features=_csv_path(raw, "node-feat"),
        labels=_csv_path(raw, "node-label"),
        edges=_csv_path(raw, "edge"),
        num_nodes=_csv_path(raw, "num-node-list"),
        num_edges=_csv_path(raw, "num-edge-list"),
        split_files=split_files,
    )


def _csv_source(files: RawOGBNodeClassificationFiles) -> OGBNodeClassificationSource:
    return OGBNodeClassificationSource(
        dataset_root=files.dataset_root,
        input_format="csv",
        features=files.features,
        labels=files.labels,
        edges=files.edges,
        num_nodes=_read_single_integer(files.num_nodes),
        num_edges=_read_single_integer(files.num_edges),
        feature_dim=_feature_dimension(files.features),
        split_files=files.split_files,
        raw_files={
            "features": str(files.features),
            "labels": str(files.labels),
            "edges": str(files.edges),
            "num_nodes": str(files.num_nodes),
            "num_edges": str(files.num_edges),
            "splits": {split: str(path) for split, path in files.split_files.items()},
        },
    )


def _npy_member_name(archive: zipfile.ZipFile, key: str) -> str:
    expected_name = f"{key}.npy"
    matches = [
        member.filename
        for member in archive.infolist()
        if Path(member.filename).name == expected_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {expected_name} member in {archive.filename}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _archive_fingerprint(
    archive_path: Path, member_keys: tuple[str, ...]
) -> tuple[dict[str, object], dict[str, str]]:
    with zipfile.ZipFile(archive_path) as archive:
        member_names = {key: _npy_member_name(archive, key) for key in member_keys}
        members = {
            key: {
                "name": member_names[key],
                "crc": archive.getinfo(member_names[key]).CRC,
                "size": archive.getinfo(member_names[key]).file_size,
            }
            for key in member_keys
        }
    stat = archive_path.stat()
    return (
        {
            "path": str(archive_path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "members": members,
        },
        member_names,
    )


def _extract_npy_member(
    archive_path: Path, member_name: str, destination: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial")
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member_name) as source, partial.open("wb") as output:
            shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    os.replace(partial, destination)


def _prepare_binary_ogb_cache(
    dataset_root: Path, binary_cache_dir: Path | str | None
) -> tuple[dict[str, Path], dict[str, object]]:
    raw_dir = dataset_root / "raw"
    archives = {
        "data": (
            raw_dir / "data.npz",
            ("node_feat", "edge_index", "num_nodes_list", "num_edges_list"),
        ),
        "labels": (raw_dir / "node-label.npz", ("node_label",)),
    }
    missing = [str(path) for path, _ in archives.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "OGB binary input requires the following files: " + ", ".join(missing)
        )

    cache_dir = (
        Path(binary_cache_dir).expanduser().resolve()
        if binary_cache_dir is not None
        else dataset_root / ".fedgraph-binary-cache"
    )
    array_paths = {
        "features": cache_dir / "data" / "node_feat.npy",
        "edges": cache_dir / "data" / "edge_index.npy",
        "num_nodes": cache_dir / "data" / "num_nodes_list.npy",
        "num_edges": cache_dir / "data" / "num_edges_list.npy",
        "labels": cache_dir / "labels" / "node_label.npy",
    }
    fingerprints: dict[str, object] = {}
    member_names: dict[str, dict[str, str]] = {}
    for archive_name, (archive_path, member_keys) in archives.items():
        fingerprint, names = _archive_fingerprint(archive_path, member_keys)
        fingerprints[archive_name] = fingerprint
        member_names[archive_name] = names
    cache_manifest = {
        "cache_version": 1,
        "archives": fingerprints,
    }
    cache_manifest_path = cache_dir / "manifest.json"
    if cache_manifest_path.exists():
        existing_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != cache_manifest:
            raise ValueError(
                f"Binary cache {cache_dir} belongs to different source archives. "
                "Choose a new --binary-cache-dir or remove the stale cache."
            )

    source_specs = {
        "data": {
            "node_feat": array_paths["features"],
            "edge_index": array_paths["edges"],
            "num_nodes_list": array_paths["num_nodes"],
            "num_edges_list": array_paths["num_edges"],
        },
        "labels": {"node_label": array_paths["labels"]},
    }
    for archive_name, destinations in source_specs.items():
        archive_path, _ = archives[archive_name]
        for key, destination in destinations.items():
            if not destination.is_file():
                _extract_npy_member(
                    archive_path, member_names[archive_name][key], destination
                )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_manifest_path.write_text(
        json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return array_paths, {
        "format": "ogb-binary",
        "data": str(archives["data"][0]),
        "labels": str(archives["labels"][0]),
        "cache_dir": str(cache_dir),
    }


def _read_npy_single_integer(path: Path, label: str) -> int:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.size != 1:
        raise ValueError(f"Expected one {label} value in {path}, found {values.size}")
    value = int(np.asarray(values).reshape(-1)[0])
    if value < 0:
        raise ValueError(f"Expected a non-negative {label} in {path}, found {value}")
    return value


def _binary_source(
    dataset_root: Path, split_name: str, binary_cache_dir: Path | str | None
) -> OGBNodeClassificationSource:
    array_paths, raw_files = _prepare_binary_ogb_cache(dataset_root, binary_cache_dir)
    features = np.load(array_paths["features"], mmap_mode="r", allow_pickle=False)
    if features.ndim != 2:
        raise ValueError(
            f"Binary node features {array_paths['features']} must have shape (N, F), "
            f"found {features.shape}"
        )
    edges = np.load(array_paths["edges"], mmap_mode="r", allow_pickle=False)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(
            f"Binary edge index {array_paths['edges']} must have shape (2, E), "
            f"found {edges.shape}"
        )
    if not np.issubdtype(edges.dtype, np.integer):
        raise ValueError(
            f"Binary edge index {array_paths['edges']} must use an integer dtype"
        )
    num_nodes = _read_npy_single_integer(array_paths["num_nodes"], "node count")
    num_edges = _read_npy_single_integer(array_paths["num_edges"], "edge count")
    if features.shape[0] != num_nodes or edges.shape[1] != num_edges:
        raise ValueError(
            "Binary metadata does not match node-feature or edge-index dimensions"
        )
    feature_dim = int(features.shape[1])
    del features
    del edges
    split_dir = dataset_root / "split" / split_name
    split_files = {
        split: _csv_path(split_dir, file_name)
        for split, file_name in _SPLIT_FILE_NAMES.items()
    }
    return OGBNodeClassificationSource(
        dataset_root=dataset_root,
        input_format="ogb-binary",
        features=array_paths["features"],
        labels=array_paths["labels"],
        edges=array_paths["edges"],
        num_nodes=num_nodes,
        num_edges=num_edges,
        feature_dim=feature_dim,
        split_files=split_files,
        raw_files={
            **raw_files,
            "splits": {split: str(path) for split, path in split_files.items()},
        },
    )


def discover_ogb_source(
    dataset_root: Path | str,
    split_name: str = "time",
    input_format: str = "auto",
    binary_cache_dir: Path | str | None = None,
) -> OGBNodeClassificationSource:
    """Discover a raw CSV or official OGB binary graph source."""
    if input_format not in {"auto", "csv", "ogb-binary"}:
        raise ValueError("input_format must be one of: auto, csv, ogb-binary")
    root = Path(dataset_root).expanduser().resolve()
    raw = root / "raw"
    has_csv_features = any(
        (raw / f"node-feat{suffix}").is_file() for suffix in (".csv.gz", ".csv")
    )
    has_binary_data = (raw / "data.npz").is_file()
    if input_format == "csv" or (input_format == "auto" and has_csv_features):
        return _csv_source(discover_raw_ogb_files(root, split_name))
    if input_format == "ogb-binary" or (input_format == "auto" and has_binary_data):
        return _binary_source(root, split_name, binary_cache_dir)
    raise FileNotFoundError(
        f"Could not find either raw/node-feat.csv[.gz] or raw/data.npz under {raw}"
    )


def _read_single_integer(path: Path) -> int:
    with _open_text(path) as source:
        raw_value = source.readline().strip()
    if not raw_value:
        raise ValueError(f"Expected an integer in {path}, found an empty file")
    return int(raw_value)


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _feature_dimension(path: Path) -> int:
    with _open_text(path) as source:
        first_row = source.readline().strip()
    if not first_row:
        raise ValueError(f"Feature file {path} is empty")
    return first_row.count(",") + 1


def _parse_label(raw_value: str, line_number: int, path: Path) -> int:
    value = raw_value.strip()
    if value.lower() in {"", "nan", "none"}:
        return -1
    try:
        return int(float(value))
    except ValueError as exc:
        raise ValueError(
            f"Invalid scalar label at {path}:{line_number}: {raw_value!r}"
        ) from exc


def _load_labels(path: Path, num_nodes: int, destination: Path) -> np.memmap:
    labels = np.memmap(destination, dtype=np.int32, mode="w+", shape=(num_nodes,))
    rows_read = 0
    with _open_text(path) as source:
        for line_number, raw_value in enumerate(source):
            if line_number >= num_nodes:
                raise ValueError(
                    f"Label file {path} has more than the declared {num_nodes} rows"
                )
            labels[line_number] = _parse_label(raw_value, line_number + 1, path)
            rows_read = line_number + 1
    if rows_read != num_nodes:
        raise ValueError(
            f"Label file {path} has {rows_read} rows; expected {num_nodes}"
        )
    labels.flush()
    return labels


def _load_binary_labels(path: Path, num_nodes: int, destination: Path) -> np.memmap:
    source = np.load(path, mmap_mode="r", allow_pickle=False)
    if source.ndim == 2 and source.shape[1] == 1:
        source = source[:, 0]
    if source.ndim != 1 or source.size != num_nodes:
        raise ValueError(
            f"Binary labels {path} must have shape ({num_nodes},) or ({num_nodes}, 1), "
            f"found {source.shape}"
        )
    if not (
        np.issubdtype(source.dtype, np.integer)
        or np.issubdtype(source.dtype, np.floating)
    ):
        raise ValueError(f"Binary labels {path} must use a numeric dtype")

    labels = np.memmap(destination, dtype=np.int32, mode="w+", shape=(num_nodes,))
    chunk_rows = 1_000_000
    for start in range(0, num_nodes, chunk_rows):
        stop = min(start + chunk_rows, num_nodes)
        values = np.asarray(source[start:stop])
        if np.issubdtype(values.dtype, np.floating):
            missing = np.isnan(values)
            present = values[~missing]
            if not np.all(np.isfinite(present)) or not np.all(
                present == np.floor(present)
            ):
                raise ValueError(f"Binary labels {path} contain non-integral values")
            if present.size and (
                present.min() < np.iinfo(np.int32).min
                or present.max() > np.iinfo(np.int32).max
            ):
                raise ValueError(f"Binary labels {path} exceed int32 range")
            converted = np.full(values.shape, -1, dtype=np.int32)
            converted[~missing] = present.astype(np.int32)
        else:
            if values.size and (
                values.min() < np.iinfo(np.int32).min
                or values.max() > np.iinfo(np.int32).max
            ):
                raise ValueError(f"Binary labels {path} exceed int32 range")
            converted = values.astype(np.int32, copy=False)
        labels[start:stop] = converted
    labels.flush()
    return labels


def _read_split_indexes(path: Path, num_nodes: int) -> np.ndarray:
    values = pd.read_csv(path, header=None, dtype=np.int64).iloc[:, 0].to_numpy()
    if values.ndim != 1 or np.any(values < 0) or np.any(values >= num_nodes):
        raise ValueError(
            f"Split file {path} contains node IDs outside [0, {num_nodes})"
        )
    if np.unique(values).size != values.size:
        raise ValueError(f"Split file {path} contains duplicate node IDs")
    return values


def _split_targets(num_nodes: int, n_trainer: int) -> np.ndarray:
    targets = np.full(n_trainer, num_nodes // n_trainer, dtype=np.int64)
    targets[: num_nodes % n_trainer] += 1
    return targets


def _capacity_aware_dirichlet_counts(
    total: int,
    proportions: np.ndarray,
    capacity: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw Dirichlet-shaped counts without exceeding final shard capacity."""
    if total > int(capacity.sum()):
        raise ValueError("requested label assignment exceeds remaining shard capacity")

    assigned = np.zeros_like(capacity)
    remaining = total
    while remaining:
        free_capacity = capacity - assigned
        eligible = free_capacity > 0
        weighted = np.where(eligible, proportions, 0.0)
        if not weighted.any():
            weighted = free_capacity.astype(np.float64)
        weighted /= weighted.sum()
        proposed = rng.multinomial(remaining, weighted)
        accepted = np.minimum(proposed, free_capacity)
        accepted_count = int(accepted.sum())
        if accepted_count == 0:
            # A highly concentrated draw can repeatedly target a full trainer.
            # Filling proportional to free slots guarantees forward progress.
            weighted = free_capacity / free_capacity.sum()
            accepted = rng.multinomial(remaining, weighted)
            accepted = np.minimum(accepted, free_capacity)
            accepted_count = int(accepted.sum())
        assigned += accepted
        remaining -= accepted_count
    return assigned


def build_label_balanced_owner_plan(
    labels: np.ndarray,
    *,
    n_trainer: int,
    iid_beta: float,
    seed: int,
    owner_path: Path,
    local_position_path: Path,
) -> PartitionPlan:
    """Assign every node once, using Dirichlet ownership only for labeled rows.

    Labeled nodes follow the familiar FedGraph label-Dirichlet policy.  The
    much larger unlabeled population fills the remaining per-trainer capacity
    using a seeded shuffled owner-slot array.  This preserves full-node
    coverage while keeping total shard sizes exactly balanced to within one.
    """
    if n_trainer < 1 or n_trainer > int(_OWNER_UNASSIGNED):
        raise ValueError("n_trainer must be between 1 and 255")
    if iid_beta <= 0:
        raise ValueError("iid_beta must be positive")
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")

    num_nodes = int(labels.size)
    if num_nodes == 0:
        raise ValueError("cannot partition an empty graph")

    owner = np.memmap(owner_path, dtype=np.uint8, mode="w+", shape=(num_nodes,))
    owner[:] = _OWNER_UNASSIGNED
    rng = np.random.default_rng(seed)

    labeled_node_ids = np.flatnonzero(labels >= 0)
    labeled_values = np.asarray(labels[labeled_node_ids], dtype=np.int32)
    class_num = int(labeled_values.max()) + 1 if labeled_values.size else 0
    if class_num == 0:
        raise ValueError("the source dataset has no labeled nodes")

    targets = _split_targets(num_nodes, n_trainer)
    remaining_capacity = targets.copy()
    for label in range(class_num):
        class_node_ids = labeled_node_ids[labeled_values == label].copy()
        rng.shuffle(class_node_ids)
        proportions = rng.dirichlet(np.full(n_trainer, iid_beta))
        counts = _capacity_aware_dirichlet_counts(
            class_node_ids.size, proportions, remaining_capacity, rng
        )
        boundaries = np.cumsum(counts, dtype=np.int64)[:-1]
        for trainer_id, node_ids in enumerate(np.split(class_node_ids, boundaries)):
            owner[node_ids] = trainer_id
        remaining_capacity -= counts

    unlabelled_node_count = int(num_nodes - labeled_node_ids.size)
    owner_slots = np.repeat(
        np.arange(n_trainer, dtype=np.uint8), remaining_capacity.astype(np.int64)
    )
    if owner_slots.size != unlabelled_node_count:
        raise RuntimeError("owner capacity does not equal the unlabeled-node count")
    rng.shuffle(owner_slots)

    offset = 0
    assignment_chunk_rows = 1_000_000
    for start in range(0, num_nodes, assignment_chunk_rows):
        stop = min(start + assignment_chunk_rows, num_nodes)
        local_labels = labels[start:stop]
        unlabelled_mask = local_labels < 0
        count = int(unlabelled_mask.sum())
        if count:
            owner_chunk = owner[start:stop]
            owner_chunk[unlabelled_mask] = owner_slots[offset : offset + count]
            owner[start:stop] = owner_chunk
            offset += count
    if offset != owner_slots.size or np.any(owner == _OWNER_UNASSIGNED):
        raise RuntimeError("owner assignment did not cover every node")
    owner.flush()

    local_position = np.memmap(
        local_position_path, dtype=np.uint32, mode="w+", shape=(num_nodes,)
    )
    cursors = np.zeros(n_trainer, dtype=np.int64)
    for start in range(0, num_nodes, assignment_chunk_rows):
        stop = min(start + assignment_chunk_rows, num_nodes)
        owner_chunk = np.asarray(owner[start:stop], dtype=np.uint8)
        order = np.argsort(owner_chunk, kind="stable")
        ordered_owners = owner_chunk[order]
        boundaries = np.r_[
            0,
            np.flatnonzero(ordered_owners[1:] != ordered_owners[:-1]) + 1,
            order.size,
        ]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            trainer_id = int(ordered_owners[left])
            node_offsets = order[left:right]
            positions = np.arange(
                cursors[trainer_id],
                cursors[trainer_id] + node_offsets.size,
                dtype=np.uint32,
            )
            local_position[start + node_offsets] = positions
            cursors[trainer_id] += node_offsets.size
    if not np.array_equal(cursors, targets):
        raise RuntimeError(
            "local-position assignment does not match target shard sizes"
        )
    local_position.flush()

    return PartitionPlan(
        num_nodes=num_nodes,
        feature_dim=0,
        class_num=class_num,
        n_trainer=n_trainer,
        iid_beta=float(iid_beta),
        seed=seed,
        shard_node_counts=[int(count) for count in cursors],
        labelled_node_count=int(labeled_node_ids.size),
        unlabelled_node_count=unlabelled_node_count,
    )


def _write_plan(path: Path, plan: PartitionPlan) -> None:
    path.write_text(
        json.dumps(asdict(plan), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read_plan(path: Path) -> PartitionPlan:
    return PartitionPlan(**json.loads(path.read_text(encoding="utf-8")))


def _group_owner_rows(owner_values: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
    order = np.argsort(owner_values, kind="stable")
    ordered_owners = owner_values[order]
    boundaries = np.r_[
        0,
        np.flatnonzero(ordered_owners[1:] != ordered_owners[:-1]) + 1,
        order.size,
    ]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        yield int(ordered_owners[left]), order[left:right]


def _create_shard_memmaps(
    shard_root: Path, shard_node_counts: list[int], feature_dim: int
) -> tuple[list[np.memmap], list[np.memmap]]:
    feature_maps: list[np.memmap] = []
    node_maps: list[np.memmap] = []
    for trainer_id, node_count in enumerate(shard_node_counts):
        shard_dir = shard_root / f"trainer-{trainer_id:03d}"
        shard_dir.mkdir(parents=True, exist_ok=False)
        feature_maps.append(
            np.memmap(
                shard_dir / "features.f32",
                dtype=np.float32,
                mode="w+",
                shape=(node_count, feature_dim),
            )
        )
        node_maps.append(
            np.memmap(
                shard_dir / "local_node_index.i64",
                dtype=np.int64,
                mode="w+",
                shape=(node_count,),
            )
        )
    return feature_maps, node_maps


def _write_local_node_indexes(
    node_maps: list[np.memmap], owner: np.memmap, local_position: np.memmap
) -> None:
    num_nodes = int(owner.size)
    chunk_rows = 1_000_000
    for start in range(0, num_nodes, chunk_rows):
        stop = min(start + chunk_rows, num_nodes)
        node_ids = np.arange(start, stop, dtype=np.int64)
        owner_values = np.asarray(owner[start:stop], dtype=np.uint8)
        positions = np.asarray(local_position[start:stop], dtype=np.uint32)
        for trainer_id, offsets in _group_owner_rows(owner_values):
            node_maps[trainer_id][positions[offsets]] = node_ids[offsets]
    for node_map in node_maps:
        node_map.flush()


def _write_features(
    path: Path,
    feature_maps: list[np.memmap],
    owner: np.memmap,
    local_position: np.memmap,
    feature_dim: int,
    chunk_rows: int,
) -> None:
    seen_rows = 0
    for frame in pd.read_csv(path, header=None, chunksize=chunk_rows, dtype=np.float32):
        values = frame.to_numpy(dtype=np.float32, copy=False)
        if values.ndim != 2 or values.shape[1] != feature_dim:
            raise ValueError(
                f"Feature chunk beginning at row {seen_rows} has shape {values.shape}; "
                f"expected (*, {feature_dim})"
            )
        stop = seen_rows + values.shape[0]
        owner_values = np.asarray(owner[seen_rows:stop], dtype=np.uint8)
        positions = np.asarray(local_position[seen_rows:stop], dtype=np.uint32)
        for trainer_id, offsets in _group_owner_rows(owner_values):
            feature_maps[trainer_id][positions[offsets]] = values[offsets]
        seen_rows = stop
    if seen_rows != owner.size:
        raise ValueError(
            f"Feature file {path} has {seen_rows} rows; expected {owner.size}"
        )
    for feature_map in feature_maps:
        feature_map.flush()


def _write_binary_features(
    path: Path,
    feature_maps: list[np.memmap],
    owner: np.memmap,
    local_position: np.memmap,
    feature_dim: int,
    chunk_rows: int,
) -> None:
    source = np.load(path, mmap_mode="r", allow_pickle=False)
    if source.ndim != 2 or source.shape != (owner.size, feature_dim):
        raise ValueError(
            f"Binary feature array {path} has shape {source.shape}; expected "
            f"({owner.size}, {feature_dim})"
        )
    if not np.issubdtype(source.dtype, np.number):
        raise ValueError(f"Binary feature array {path} must use a numeric dtype")
    for start in range(0, owner.size, chunk_rows):
        stop = min(start + chunk_rows, owner.size)
        values = np.asarray(source[start:stop], dtype=np.float32)
        owner_values = np.asarray(owner[start:stop], dtype=np.uint8)
        positions = np.asarray(local_position[start:stop], dtype=np.uint32)
        for trainer_id, offsets in _group_owner_rows(owner_values):
            feature_maps[trainer_id][positions[offsets]] = values[offsets]
    for feature_map in feature_maps:
        feature_map.flush()


def _write_internal_edges(
    path: Path,
    shard_root: Path,
    owner: np.memmap,
    local_position: np.memmap,
    n_trainer: int,
    chunk_rows: int,
) -> tuple[int, int]:
    total_edges = 0
    internal_edges = 0
    with ExitStack() as stack:
        handles = [
            stack.enter_context(
                (shard_root / f"trainer-{trainer_id:03d}" / "edges.u32").open("wb")
            )
            for trainer_id in range(n_trainer)
        ]
        for frame in pd.read_csv(
            path, header=None, chunksize=chunk_rows, dtype=np.int64
        ):
            edge_values = frame.to_numpy(dtype=np.int64, copy=False)
            if edge_values.ndim != 2 or edge_values.shape[1] != 2:
                raise ValueError(
                    f"Edge chunk has shape {edge_values.shape}; expected (*, 2)"
                )
            sources = edge_values[:, 0]
            targets = edge_values[:, 1]
            if (
                np.any(sources < 0)
                or np.any(targets < 0)
                or np.any(sources >= owner.size)
                or np.any(targets >= owner.size)
            ):
                raise ValueError(
                    "Edge file contains an endpoint outside the node-ID range"
                )
            source_owners = np.asarray(owner[sources], dtype=np.uint8)
            target_owners = np.asarray(owner[targets], dtype=np.uint8)
            internal_mask = source_owners == target_owners
            total_edges += int(edge_values.shape[0])
            if not np.any(internal_mask):
                continue
            internal_sources = sources[internal_mask]
            internal_targets = targets[internal_mask]
            internal_owners = source_owners[internal_mask]
            local_edges = np.column_stack(
                (
                    np.asarray(local_position[internal_sources], dtype=np.uint32),
                    np.asarray(local_position[internal_targets], dtype=np.uint32),
                )
            )
            for trainer_id, offsets in _group_owner_rows(internal_owners):
                local_edges[offsets].tofile(handles[trainer_id])
            internal_edges += int(local_edges.shape[0])
    return total_edges, internal_edges


def _write_binary_internal_edges(
    path: Path,
    shard_root: Path,
    owner: np.memmap,
    local_position: np.memmap,
    n_trainer: int,
    chunk_rows: int,
) -> tuple[int, int]:
    edges = np.load(path, mmap_mode="r", allow_pickle=False)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"Binary edge index {path} must have shape (2, E)")
    if not np.issubdtype(edges.dtype, np.integer):
        raise ValueError(f"Binary edge index {path} must use an integer dtype")
    total_edges = 0
    internal_edges = 0
    with ExitStack() as stack:
        handles = [
            stack.enter_context(
                (shard_root / f"trainer-{trainer_id:03d}" / "edges.u32").open("wb")
            )
            for trainer_id in range(n_trainer)
        ]
        for start in range(0, edges.shape[1], chunk_rows):
            stop = min(start + chunk_rows, edges.shape[1])
            sources = np.asarray(edges[0, start:stop], dtype=np.int64)
            targets = np.asarray(edges[1, start:stop], dtype=np.int64)
            if (
                np.any(sources < 0)
                or np.any(targets < 0)
                or np.any(sources >= owner.size)
                or np.any(targets >= owner.size)
            ):
                raise ValueError(
                    "Binary edge index contains an endpoint outside the node-ID range"
                )
            source_owners = np.asarray(owner[sources], dtype=np.uint8)
            target_owners = np.asarray(owner[targets], dtype=np.uint8)
            internal_mask = source_owners == target_owners
            total_edges += stop - start
            if not np.any(internal_mask):
                continue
            internal_sources = sources[internal_mask]
            internal_targets = targets[internal_mask]
            internal_owners = source_owners[internal_mask]
            local_edges = np.column_stack(
                (
                    np.asarray(local_position[internal_sources], dtype=np.uint32),
                    np.asarray(local_position[internal_targets], dtype=np.uint32),
                )
            )
            for trainer_id, offsets in _group_owner_rows(internal_owners):
                local_edges[offsets].tofile(handles[trainer_id])
            internal_edges += int(local_edges.shape[0])
    return total_edges, internal_edges


def _tensor_from_memmap(
    path: Path, dtype: np.dtype, shape: tuple[int, ...]
) -> torch.Tensor:
    values = np.memmap(path, dtype=dtype, mode="r", shape=shape)
    return torch.from_numpy(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_payloads(
    split_indexes: Mapping[str, np.ndarray],
    labels: np.memmap,
    owner: np.memmap,
    local_position: np.memmap,
    n_trainer: int,
) -> dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    payloads: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {
        trainer_id: {} for trainer_id in range(n_trainer)
    }
    for split, node_ids in split_indexes.items():
        split_labels = np.asarray(labels[node_ids], dtype=np.int64)
        if np.any(split_labels < 0):
            raise ValueError(f"Official {split} split includes unlabeled nodes")
        split_owners = np.asarray(owner[node_ids], dtype=np.uint8)
        split_positions = np.asarray(local_position[node_ids], dtype=np.uint32)
        for trainer_id in range(n_trainer):
            mask = split_owners == trainer_id
            positions = split_positions[mask]
            values = split_labels[mask]
            order = np.argsort(positions, kind="stable")
            payloads[trainer_id][split] = (
                torch.from_numpy(positions[order].astype(np.int64, copy=False)),
                torch.from_numpy(values[order].astype(np.int64, copy=False)),
            )
    return payloads


def _finalize_shards(
    stage_dir: Path,
    plan: PartitionPlan,
    labels: np.memmap,
    owner: np.memmap,
    local_position: np.memmap,
    split_indexes: Mapping[str, np.ndarray],
    checksums: bool,
) -> tuple[list[dict[str, object]], int]:
    shard_root = stage_dir / "shards"
    split_payloads = _split_payloads(
        split_indexes, labels, owner, local_position, plan.n_trainer
    )
    shard_metadata: list[dict[str, object]] = []
    total_internal_edges = 0
    for trainer_id, node_count in enumerate(plan.shard_node_counts):
        shard_dir = shard_root / f"trainer-{trainer_id:03d}"
        node_tensor = _tensor_from_memmap(
            shard_dir / "local_node_index.i64", np.int64, (node_count,)
        ).long()
        feature_tensor = _tensor_from_memmap(
            shard_dir / "features.f32", np.float32, (node_count, plan.feature_dim)
        ).float()
        edge_values = np.fromfile(shard_dir / "edges.u32", dtype=np.uint32)
        if edge_values.size % 2:
            raise RuntimeError(
                f"Shard {trainer_id} has a malformed temporary edge file"
            )
        edge_count = int(edge_values.size // 2)
        adjacency = torch.from_numpy(edge_values.reshape(-1, 2).T.copy()).long()
        if adjacency.numel() == 0:
            adjacency = torch.empty((2, 0), dtype=torch.long)
        if adjacency.numel() and int(adjacency.max()) >= node_count:
            raise RuntimeError(
                f"Shard {trainer_id} has an adjacency endpoint out of range"
            )

        torch.save(node_tensor, shard_dir / "local_node_index.pt")
        torch.save(node_tensor, shard_dir / "communicate_node_index.pt")
        torch.save(adjacency, shard_dir / "adj.pt")
        torch.save(feature_tensor, shard_dir / "features.pt")
        for split in _SPLIT_CODES:
            indexes, split_labels = split_payloads[trainer_id][split]
            torch.save(indexes, shard_dir / f"idx_{split}.pt")
            torch.save(split_labels, shard_dir / f"{split}_labels.pt")
        torch.save(torch.tensor(plan.num_nodes), shard_dir / "global_node_num.pt")
        torch.save(torch.tensor(plan.class_num), shard_dir / "class_num.pt")

        temporary_files = [
            shard_dir / "features.f32",
            shard_dir / "local_node_index.i64",
            shard_dir / "edges.u32",
        ]
        for temporary_file in temporary_files:
            temporary_file.unlink()

        files = sorted(path for path in shard_dir.glob("*.pt") if path.is_file())
        metadata: dict[str, object] = {
            "trainer_id": trainer_id,
            "node_count": node_count,
            "internal_edge_count": edge_count,
            "train_count": int(split_payloads[trainer_id]["train"][0].numel()),
            "val_count": int(split_payloads[trainer_id]["val"][0].numel()),
            "test_count": int(split_payloads[trainer_id]["test"][0].numel()),
            "files": {path.name: path.stat().st_size for path in files},
        }
        if checksums:
            metadata["sha256"] = {path.name: _sha256(path) for path in files}
        (shard_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shard_metadata.append(metadata)
        total_internal_edges += edge_count
    return shard_metadata, total_internal_edges


def _validate_final_shards(stage_dir: Path, plan: PartitionPlan) -> None:
    total_nodes = 0
    total_split_counts = {split: 0 for split in _SPLIT_CODES}
    for trainer_id, expected_nodes in enumerate(plan.shard_node_counts):
        shard_dir = stage_dir / "shards" / f"trainer-{trainer_id:03d}"
        local_nodes = torch.load(shard_dir / "local_node_index.pt", weights_only=True)
        communicate_nodes = torch.load(
            shard_dir / "communicate_node_index.pt", weights_only=True
        )
        features = torch.load(shard_dir / "features.pt", weights_only=True)
        adjacency = torch.load(shard_dir / "adj.pt", weights_only=True)
        if local_nodes.numel() != expected_nodes or not torch.equal(
            local_nodes, communicate_nodes
        ):
            raise RuntimeError(f"Shard {trainer_id} has invalid 0-hop node indexes")
        if features.shape != (expected_nodes, plan.feature_dim):
            raise RuntimeError(f"Shard {trainer_id} feature shape is invalid")
        if adjacency.shape[0] != 2 or (
            adjacency.numel() and int(adjacency.max()) >= expected_nodes
        ):
            raise RuntimeError(f"Shard {trainer_id} adjacency is invalid")
        for split in _SPLIT_CODES:
            indexes = torch.load(shard_dir / f"idx_{split}.pt", weights_only=True)
            split_labels = torch.load(
                shard_dir / f"{split}_labels.pt", weights_only=True
            )
            if indexes.numel() != split_labels.numel() or (
                indexes.numel() and int(indexes.max()) >= expected_nodes
            ):
                raise RuntimeError(f"Shard {trainer_id} has an invalid {split} split")
            total_split_counts[split] += int(indexes.numel())
        total_nodes += expected_nodes
    if total_nodes != plan.num_nodes:
        raise RuntimeError("shard node counts do not cover the source node count")


def partition_raw_ogb_0hop(
    *,
    dataset_root: Path | str,
    output_dir: Path | str,
    n_trainer: int,
    iid_beta: float,
    seed: int,
    split_name: str = "time",
    chunk_rows: int = 100_000,
    checksums: bool = True,
    resume: bool = False,
    input_format: str = "auto",
    binary_cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Write a complete, label-balanced, 0-hop OGB node-classification artifact.

    ``output_dir`` is published atomically only after all tensors and manifest
    invariants have been written.  ``resume`` reuses the deterministic owner
    plan in ``<output_dir>.incomplete`` but re-materializes shards, preventing
    duplicate edges after an interrupted write.
    """
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    source = discover_ogb_source(
        dataset_root,
        split_name,
        input_format=input_format,
        binary_cache_dir=binary_cache_dir,
    )
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output}")
    stage_dir = output.with_name(f"{output.name}.incomplete")
    work_dir = stage_dir / "_work"
    plan_path = work_dir / "plan.json"
    labels_path = work_dir / "labels.i32"
    owner_path = work_dir / "owner.u8"
    local_position_path = work_dir / "local_position.u32"

    if stage_dir.exists() and not resume:
        raise FileExistsError(
            f"Found incomplete artifact at {stage_dir}; rerun with resume=True or remove it"
        )
    if not stage_dir.exists():
        work_dir.mkdir(parents=True)

    num_nodes = source.num_nodes
    feature_dim = source.feature_dim
    if resume:
        if not all(
            path.exists()
            for path in (plan_path, labels_path, owner_path, local_position_path)
        ):
            raise FileNotFoundError(
                f"Cannot resume {stage_dir}: the persistent assignment plan is incomplete"
            )
        plan = _read_plan(plan_path)
        if (
            plan.num_nodes != num_nodes
            or plan.feature_dim != feature_dim
            or plan.n_trainer != n_trainer
            or plan.iid_beta != float(iid_beta)
            or plan.seed != seed
        ):
            raise ValueError(
                "resume arguments do not match the incomplete partition plan"
            )
        labels = np.memmap(labels_path, dtype=np.int32, mode="r", shape=(num_nodes,))
        owner = np.memmap(owner_path, dtype=np.uint8, mode="r", shape=(num_nodes,))
        local_position = np.memmap(
            local_position_path, dtype=np.uint32, mode="r", shape=(num_nodes,)
        )
    else:
        labels = (
            _load_labels(source.labels, num_nodes, labels_path)
            if source.input_format == "csv"
            else _load_binary_labels(source.labels, num_nodes, labels_path)
        )
        plan = build_label_balanced_owner_plan(
            labels,
            n_trainer=n_trainer,
            iid_beta=iid_beta,
            seed=seed,
            owner_path=owner_path,
            local_position_path=local_position_path,
        )
        plan = PartitionPlan(**{**asdict(plan), "feature_dim": feature_dim})
        _write_plan(plan_path, plan)
        owner = np.memmap(owner_path, dtype=np.uint8, mode="r", shape=(num_nodes,))
        local_position = np.memmap(
            local_position_path, dtype=np.uint32, mode="r", shape=(num_nodes,)
        )

    split_indexes = {
        split: _read_split_indexes(path, num_nodes)
        for split, path in source.split_files.items()
    }
    split_codes = np.zeros(num_nodes, dtype=np.uint8)
    for split, node_ids in split_indexes.items():
        if np.any(split_codes[node_ids]):
            raise ValueError("Official train/validation/test splits overlap")
        split_codes[node_ids] = _SPLIT_CODES[split]

    shard_root = stage_dir / "shards"
    if shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir()
    feature_maps, node_maps = _create_shard_memmaps(
        shard_root, plan.shard_node_counts, plan.feature_dim
    )
    _write_local_node_indexes(node_maps, owner, local_position)
    if source.input_format == "csv":
        _write_features(
            source.features,
            feature_maps,
            owner,
            local_position,
            plan.feature_dim,
            chunk_rows,
        )
    else:
        _write_binary_features(
            source.features,
            feature_maps,
            owner,
            local_position,
            plan.feature_dim,
            chunk_rows,
        )
    del feature_maps
    del node_maps

    if source.input_format == "csv":
        total_edges, internal_edges = _write_internal_edges(
            source.edges,
            shard_root,
            owner,
            local_position,
            plan.n_trainer,
            chunk_rows,
        )
    else:
        total_edges, internal_edges = _write_binary_internal_edges(
            source.edges,
            shard_root,
            owner,
            local_position,
            plan.n_trainer,
            chunk_rows,
        )
    if total_edges != source.num_edges:
        raise ValueError(
            f"Edge source has {total_edges} rows; expected {source.num_edges}"
        )
    shard_metadata, finalized_internal_edges = _finalize_shards(
        stage_dir,
        plan,
        labels,
        owner,
        local_position,
        split_indexes,
        checksums,
    )
    if internal_edges != finalized_internal_edges:
        raise RuntimeError("final shard edge counts do not match streamed edge counts")
    _validate_final_shards(stage_dir, plan)

    manifest: dict[str, object] = {
        "artifact_version": 1,
        "hop_semantics": 0,
        "partition_policy": "label_dirichlet_balanced_unlabeled",
        "dataset_root": str(source.dataset_root),
        "input_format": source.input_format,
        "raw_files": source.raw_files,
        "global_node_num": plan.num_nodes,
        "global_edge_num": total_edges,
        "retained_internal_edge_num": internal_edges,
        "cross_partition_edge_num": total_edges - internal_edges,
        "retained_internal_edge_fraction": (
            internal_edges / total_edges if total_edges else 0.0
        ),
        "feature_dtype": "float32",
        "feature_dim": plan.feature_dim,
        "class_num": plan.class_num,
        "n_trainer": plan.n_trainer,
        "iid_beta": plan.iid_beta,
        "seed": plan.seed,
        "labelled_node_count": plan.labelled_node_count,
        "unlabelled_node_count": plan.unlabelled_node_count,
        "official_split_counts": {
            split: int(indexes.size) for split, indexes in split_indexes.items()
        },
        "shards": shard_metadata,
    }
    (stage_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.rmtree(work_dir)
    os.replace(stage_dir, output)
    return manifest
