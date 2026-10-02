#!/usr/bin/env python3
"""Validate and publish a complete manifest-style NC artifact to Hugging Face.

This tool is intentionally separate from FedGraph's historical
``save_trainer_data_to_hugging_face`` helper. It publishes the directory tree
written by ``partition_ogbn_nc.py`` as one dataset repository, preserving the
legacy per-trainer repository format and loader unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

_COMMON_SHARD_FILES = {
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
}
_SHARD_FILES_BY_CONTRACT = {
    (1, 0): _COMMON_SHARD_FILES,
    (2, 2): _COMMON_SHARD_FILES
    | {
        "adj_global.pt",
        "source_degree.pt",
        "source_offsets.pt",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_nc_artifact_for_upload(
    artifact_dir: Path | str, *, verify_checksums: bool = True
) -> dict[str, Any]:
    """Validate a completed v1/v2 artifact without materializing its tensors."""
    artifact_root = Path(artifact_dir).expanduser().resolve()
    manifest_path = artifact_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Artifact manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = (manifest.get("artifact_version"), manifest.get("hop_semantics"))
    if contract not in _SHARD_FILES_BY_CONTRACT:
        raise ValueError("Artifact must use the version/hop contract (1, 0) or (2, 2)")
    expected_shard_files = _SHARD_FILES_BY_CONTRACT[contract]
    n_trainer = manifest.get("n_trainer")
    shards = manifest.get("shards")
    if not isinstance(n_trainer, int) or n_trainer < 1:
        raise ValueError("Artifact manifest has an invalid n_trainer")
    if not isinstance(shards, list) or len(shards) != n_trainer:
        raise ValueError("Artifact manifest must describe every trainer shard")

    for trainer_id, manifest_metadata in enumerate(shards):
        shard_dir = artifact_root / "shards" / f"trainer-{trainer_id:03d}"
        metadata_path = shard_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Artifact metadata is missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata != manifest_metadata:
            raise ValueError(
                f"Artifact metadata does not match manifest for trainer {trainer_id}"
            )
        if metadata.get("trainer_id") != trainer_id:
            raise ValueError(
                f"Artifact metadata has an invalid trainer ID: {trainer_id}"
            )
        files = metadata.get("files")
        if not isinstance(files, dict) or set(files) != expected_shard_files:
            raise ValueError(f"Artifact shard {trainer_id} has an unexpected file list")
        checksums = metadata.get("sha256")
        if verify_checksums and (
            not isinstance(checksums, dict) or set(checksums) != expected_shard_files
        ):
            raise ValueError(
                f"Artifact shard {trainer_id} has no complete SHA-256 metadata; "
                "rerun the splitter without --no-checksums or use "
                "--skip-checksum-verification deliberately"
            )
        for file_name, expected_size in files.items():
            path = shard_dir / file_name
            if not isinstance(expected_size, int) or not path.is_file():
                raise FileNotFoundError(f"Artifact data file is missing: {path}")
            if path.stat().st_size != expected_size:
                raise ValueError(f"Artifact data file has an unexpected size: {path}")
            if verify_checksums and _sha256(path) != checksums[file_name]:
                raise ValueError(f"Artifact data file has a checksum mismatch: {path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and upload a complete local NC artifact to Hugging Face."
    )
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default=None)
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--private", action="store_true", default=True)
    visibility.add_argument("--public", action="store_false", dest="private")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--skip-checksum-verification",
        action="store_true",
        help="Allow an artifact without verifying its splitter SHA-256 metadata.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_workers < 1:
        raise SystemExit("--num-workers must be positive")
    artifact_dir = args.artifact_dir.expanduser().resolve()
    manifest = validate_nc_artifact_for_upload(
        artifact_dir, verify_checksums=not args.skip_checksum_verification
    )
    print(
        "Validated artifact: "
        f"n_trainer={manifest['n_trainer']}, "
        f"global_node_num={manifest['global_node_num']}, root={artifact_dir}"
    )

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=artifact_dir,
        revision=args.revision,
        private=args.private,
        num_workers=args.num_workers,
    )
    print(f"Published artifact to https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
