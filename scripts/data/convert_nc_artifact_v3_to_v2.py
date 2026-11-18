#!/usr/bin/env python3
"""Convert compact v3 2-hop NC shards into the dual-adjacency v2 contract."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

_V3_FILES = {
    "local_node_index.pt",
    "communicate_node_index.pt",
    "adj.pt",
    "train_labels.pt",
    "val_labels.pt",
    "test_labels.pt",
    "features.pt",
    "idx_train.pt",
    "idx_val.pt",
    "idx_test.pt",
    "global_node_num.pt",
    "class_num.pt",
    "source_degree.pt",
    "source_offsets.pt",
}
_UNCHANGED_FILES = _V3_FILES - {"adj.pt"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: Path, destination: Path, *, copy_files: bool) -> None:
    if copy_files:
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _save_raw_edge_index(raw_path: Path, output_path: Path, edge_count: int) -> None:
    if edge_count == 0:
        edge_index = torch.empty((2, 0), dtype=torch.int64)
    else:
        edge_index = torch.from_file(
            str(raw_path),
            shared=False,
            size=2 * edge_count,
            dtype=torch.int64,
        ).view(2, edge_count)
    torch.save(edge_index, output_path)
    del edge_index
    raw_path.unlink(missing_ok=True)


def _write_local_adjacency(
    compact_adjacency: torch.Tensor,
    output_path: Path,
    *,
    chunk_edges: int,
) -> None:
    edge_count = int(compact_adjacency.size(1))
    raw_path = output_path.with_suffix(".local-i64.tmp")
    try:
        if edge_count:
            output = np.memmap(
                raw_path,
                dtype=np.int64,
                mode="w+",
                shape=(2, edge_count),
            )
            for start in range(0, edge_count, chunk_edges):
                stop = min(start + chunk_edges, edge_count)
                output[:, start:stop] = (
                    compact_adjacency[:, start:stop].long().contiguous().numpy()
                )
            output.flush()
            del output
        else:
            raw_path.touch()
        _save_raw_edge_index(raw_path, output_path, edge_count)
    finally:
        raw_path.unlink(missing_ok=True)


def _write_global_adjacency(
    compact_adjacency: torch.Tensor,
    communicate_nodes: torch.Tensor,
    output_path: Path,
    *,
    chunk_edges: int,
) -> None:
    edge_count = int(compact_adjacency.size(1))
    raw_path = output_path.with_suffix(".global-i64.tmp")
    try:
        if edge_count:
            output = np.memmap(
                raw_path,
                dtype=np.int64,
                mode="w+",
                shape=(2, edge_count),
            )
            for start in range(0, edge_count, chunk_edges):
                stop = min(start + chunk_edges, edge_count)
                local_chunk = compact_adjacency[:, start:stop].long()
                if local_chunk.numel() and (
                    int(local_chunk.min()) < 0
                    or int(local_chunk.max()) >= communicate_nodes.numel()
                ):
                    raise ValueError("compact adjacency endpoint is out of range")
                output[:, start:stop] = (
                    communicate_nodes[local_chunk].contiguous().numpy()
                )
            output.flush()
            del output
        else:
            raw_path.touch()
        _save_raw_edge_index(raw_path, output_path, edge_count)
    finally:
        raw_path.unlink(missing_ok=True)


def _v2_adjacency_contract() -> dict[str, dict[str, object]]:
    return {
        "communicate_node_index": {
            "file": "communicate_node_index.pt",
            "coordinate_system": "global_node_id",
            "sorted": True,
            "unique": True,
            "contains_all_owned_nodes": True,
        },
        "pretrain_adjacency": {
            "file": "adj_global.pt",
            "coordinate_system": "global_node_id",
            "sorted_by": "source_then_input_order",
            "edge_semantics": "induced_on_communication_nodes",
            "contiguous": True,
        },
        "training_adjacency": {
            "file": "adj.pt",
            "coordinate_system": "communicate_node_position",
            "mapping_file": "communicate_node_index.pt",
            "sorted_by": "source_then_input_order",
            "edge_semantics": "same_edges_as_pretrain_adjacency",
            "contiguous": True,
        },
        "source_degree": {
            "file": "source_degree.pt",
            "alignment": "communicate_node_position",
            "includes_self_loop": False,
        },
        "source_offsets": {
            "file": "source_offsets.pt",
            "alignment": "communicate_node_position_plus_terminal",
            "indexes": "training_adjacency_columns",
        },
    }


def _converted_manifest(
    source_manifest: dict[str, object],
    *,
    source_manifest_sha256: str,
    trainer_ids: list[int],
    chunk_edges: int,
    complete_artifact: bool,
) -> dict[str, object]:
    manifest = copy.deepcopy(source_manifest)
    manifest["artifact_version"] = 2
    manifest.pop("adjacency_layout", None)
    manifest["adjacency_contract"] = _v2_adjacency_contract()
    manifest["conversion_provenance"] = {
        "tool": "scripts/data/convert_nc_artifact_v3_to_v2.py",
        "source_artifact_version": 3,
        "source_manifest_sha256": source_manifest_sha256,
        "converted_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "chunk_edges": chunk_edges,
        "trainer_ids": trainer_ids,
        "complete_artifact": complete_artifact,
    }
    return manifest


def _convert_shard(
    source_dir: Path,
    output_dir: Path,
    trainer_id: int,
    *,
    chunk_edges: int,
    checksums: bool,
    copy_files: bool,
) -> dict[str, object]:
    source_metadata = json.loads(
        (source_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if source_metadata.get("trainer_id") != trainer_id:
        raise ValueError(f"source shard {trainer_id} has an invalid trainer ID")
    source_files = source_metadata.get("files")
    if not isinstance(source_files, dict) or set(source_files) != _V3_FILES:
        raise ValueError(f"source shard {trainer_id} is not a compact v3 shard")

    output_dir.mkdir(parents=True)
    for file_name in sorted(_UNCHANGED_FILES):
        source_path = source_dir / file_name
        if not source_path.is_file():
            raise FileNotFoundError(f"source artifact file is missing: {source_path}")
        _link_or_copy(
            source_path,
            output_dir / file_name,
            copy_files=copy_files,
        )

    compact_adjacency = torch.load(
        source_dir / "adj.pt", map_location="cpu", weights_only=True, mmap=True
    )
    communicate_nodes = torch.load(
        source_dir / "communicate_node_index.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if (
        compact_adjacency.ndim != 2
        or compact_adjacency.size(0) != 2
        or compact_adjacency.dtype != torch.int32
    ):
        raise ValueError("compact adjacency must be an int32 [2, E] tensor")
    if communicate_nodes.ndim != 1 or communicate_nodes.dtype != torch.int64:
        raise ValueError(
            "communicate_node_index must be a one-dimensional int64 tensor"
        )

    _write_local_adjacency(
        compact_adjacency,
        output_dir / "adj.pt",
        chunk_edges=chunk_edges,
    )
    _write_global_adjacency(
        compact_adjacency,
        communicate_nodes,
        output_dir / "adj_global.pt",
        chunk_edges=chunk_edges,
    )

    files = sorted(path for path in output_dir.glob("*.pt") if path.is_file())
    metadata = copy.deepcopy(source_metadata)
    metadata.pop("adjacency_layout", None)
    metadata.pop("adjacency_dtype", None)
    metadata["files"] = {path.name: path.stat().st_size for path in files}
    if checksums:
        metadata["sha256"] = {path.name: _sha256(path) for path in files}
    else:
        metadata.pop("sha256", None)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def convert_nc_artifact_v3_to_v2(
    source_dir: Path | str,
    output_dir: Path | str,
    *,
    trainer_ids: Optional[Iterable[int]] = None,
    chunk_edges: int = 1_000_000,
    checksums: bool = True,
    copy_files: bool = False,
) -> dict[str, object]:
    """Convert all or selected compact shards without modifying the v3 source."""
    if chunk_edges < 1:
        raise ValueError("chunk_edges must be positive")
    source_root = Path(source_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    if source_root == output_root:
        raise ValueError("source_dir and output_dir must be different")
    manifest_path = source_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"source manifest is missing: {manifest_path}")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        source_manifest.get("artifact_version"),
        source_manifest.get("hop_semantics"),
    ) != (3, 2,) or source_manifest.get("adjacency_layout") != "local-int32":
        raise ValueError("source artifact must use the compact v3 2-hop contract")
    n_trainer = source_manifest.get("n_trainer")
    if not isinstance(n_trainer, int) or n_trainer < 1:
        raise ValueError("source artifact has an invalid n_trainer")

    selected_ids = (
        list(range(n_trainer))
        if trainer_ids is None
        else sorted({int(trainer_id) for trainer_id in trainer_ids})
    )
    if not selected_ids:
        raise ValueError("at least one trainer ID must be selected")
    if selected_ids[0] < 0 or selected_ids[-1] >= n_trainer:
        raise ValueError(f"trainer IDs must be in [0, {n_trainer - 1}]")
    if output_root.exists():
        raise FileExistsError(f"output artifact already exists: {output_root}")
    stage_root = output_root.with_name(f"{output_root.name}.incomplete")
    if stage_root.exists():
        raise FileExistsError(f"incomplete output already exists: {stage_root}")

    source_manifest_sha256 = _sha256(manifest_path)
    manifest = _converted_manifest(
        source_manifest,
        source_manifest_sha256=source_manifest_sha256,
        trainer_ids=selected_ids,
        chunk_edges=chunk_edges,
        complete_artifact=len(selected_ids) == n_trainer,
    )
    try:
        (stage_root / "shards").mkdir(parents=True)
        for trainer_id in selected_ids:
            source_shard = source_root / "shards" / f"trainer-{trainer_id:03d}"
            if not source_shard.is_dir():
                raise FileNotFoundError(f"source shard is missing: {source_shard}")
            output_shard = stage_root / "shards" / f"trainer-{trainer_id:03d}"
            metadata = _convert_shard(
                source_shard,
                output_shard,
                trainer_id,
                chunk_edges=chunk_edges,
                checksums=checksums,
                copy_files=copy_files,
            )
            manifest["shards"][trainer_id] = metadata
            print(
                f"Converted trainer {trainer_id}: "
                f"edges={metadata['induced_edge_count']} output={output_shard}"
            )

        (stage_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(stage_root, output_root)
    except Exception:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise
    return manifest


def _parse_trainer_ids(raw: Optional[str]) -> Optional[list[int]]:
    if raw is None:
        return None
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--trainer-ids must not be empty")
    try:
        return [int(value) for value in values]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--trainer-ids must be a comma-separated integer list"
        ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert compact v3 2-hop NC artifacts to dual-adjacency v2."
    )
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--trainer-ids",
        default=None,
        help="Optional comma-separated rank subset for worker-local conversion.",
    )
    parser.add_argument("--chunk-edges", type=int, default=1_000_000)
    parser.add_argument("--no-checksums", action="store_true")
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy unchanged files instead of using same-filesystem hard links.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write only a v2 control manifest; no shard is converted.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.manifest_only:
        source_root = args.source_dir.expanduser().resolve()
        output_root = args.output_dir.expanduser().resolve()
        manifest_path = source_root / "manifest.json"
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            source_manifest.get("artifact_version"),
            source_manifest.get("hop_semantics"),
        ) != (3, 2):
            raise ValueError("source artifact must use the compact v3 2-hop contract")
        n_trainer = int(source_manifest["n_trainer"])
        trainer_ids = _parse_trainer_ids(args.trainer_ids) or list(range(n_trainer))
        if output_root.exists():
            raise FileExistsError(f"output artifact already exists: {output_root}")
        output_root.mkdir(parents=True)
        manifest = _converted_manifest(
            source_manifest,
            source_manifest_sha256=_sha256(manifest_path),
            trainer_ids=trainer_ids,
            chunk_edges=args.chunk_edges,
            complete_artifact=False,
        )
        (output_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Published v2 control manifest: output={output_root}")
        return 0
    manifest = convert_nc_artifact_v3_to_v2(
        args.source_dir,
        args.output_dir,
        trainer_ids=_parse_trainer_ids(args.trainer_ids),
        chunk_edges=args.chunk_edges,
        checksums=not args.no_checksums,
        copy_files=args.copy_files,
    )
    print(
        "Published converted artifact: "
        f"version={manifest['artifact_version']} output={args.output_dir.expanduser()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
