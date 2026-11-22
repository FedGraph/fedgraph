"""Promote and open finalized FedGCN feature matrices without rewriting them."""

import errno
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from fedgraph.graph_storage import _FEATURE_AGGREGATION_STORE_VERSION

CACHE_MANIFEST_FILE = "feature-cache.json"
_CACHE_VERSION = 1
_RANK_MANIFEST_FILE = "cache-shard.json"
_FEATURE_FILE = "feature-aggregation.f32"
_FEATURE_METADATA_FILE = "feature-aggregation.json"
_COMMUNICATION_INDEX_FILE = "communicate_node_index.pt"


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(text: bytes, path: Path) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return _read_object(path.read_bytes(), path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


@dataclass(frozen=True)
class FeatureAggregationCacheManifest:
    cache_version: int
    source_run_id: str
    artifact_manifest_sha256: str
    n_trainer: int
    norm_type: str
    original_feature_dim: int
    feature_dim: int
    feature_compression_mode: str
    feature_compression_basis_sha256: Optional[str]

    def __post_init__(self) -> None:
        if type(self.cache_version) is not int or self.cache_version != _CACHE_VERSION:
            raise ValueError("Unsupported feature cache manifest version")
        if not isinstance(self.source_run_id, str) or not self.source_run_id:
            raise ValueError("Feature cache must identify the source run")
        if not _is_sha256(self.artifact_manifest_sha256):
            raise ValueError("Feature cache artifact fingerprint is invalid")
        for value in (self.n_trainer, self.original_feature_dim, self.feature_dim):
            if type(value) is not int or value < 1:
                raise ValueError(
                    "Feature cache dimensions and trainer count must be positive"
                )
        if self.norm_type not in ("none", "row", "sym"):
            raise ValueError("Feature cache normalization must be none, row, or sym")
        if self.feature_compression_mode == "none":
            if (
                self.feature_dim != self.original_feature_dim
                or self.feature_compression_basis_sha256 is not None
            ):
                raise ValueError("Unprojected cache must preserve the original width")
        elif self.feature_compression_mode == "shared-pca":
            if self.feature_dim > self.original_feature_dim or not _is_sha256(
                self.feature_compression_basis_sha256
            ):
                raise ValueError(
                    "Projected cache requires a valid PCA width and basis hash"
                )
        else:
            raise ValueError("Unsupported feature cache compression mode")

    @property
    def cache_id(self) -> str:
        content = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_feature_cache_manifest(cache_dir: Path) -> FeatureAggregationCacheManifest:
    path = cache_dir / CACHE_MANIFEST_FILE
    try:
        return FeatureAggregationCacheManifest(**_read_json(path))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid feature cache manifest {path}: {error}") from error


def _read_artifact(artifact_dir: Path) -> tuple[dict[str, Any], str]:
    path = artifact_dir / "manifest.json"
    content = path.read_bytes()
    artifact = _read_object(content, path)
    if (
        artifact.get("artifact_version"),
        artifact.get("hop_semantics"),
        artifact.get("adjacency_layout"),
    ) != (3, 2, "local-int32"):
        raise ValueError("Feature cache requires a compact v3 2-hop artifact")
    return artifact, hashlib.sha256(content).hexdigest()


def validate_feature_cache_artifact(
    manifest: FeatureAggregationCacheManifest, artifact_dir: Path
) -> dict[str, Any]:
    artifact, digest = _read_artifact(artifact_dir)
    if digest != manifest.artifact_manifest_sha256:
        raise ValueError("Feature cache does not match the source artifact manifest")
    if (
        artifact.get("n_trainer") != manifest.n_trainer
        or artifact.get("feature_dim") != manifest.original_feature_dim
    ):
        raise ValueError(
            "Feature cache trainer count or original width does not match the artifact"
        )
    return artifact


def _artifact_rows(
    artifact_dir: Path, artifact: dict[str, Any], trainer_id: int
) -> tuple[int, str]:
    shards = artifact.get("shards")
    if not isinstance(shards, list) or not 0 <= trainer_id < len(shards):
        raise ValueError(f"Trainer {trainer_id} is missing from the source artifact")
    shard = shards[trainer_id]
    if not isinstance(shard, dict) or shard.get("trainer_id") != trainer_id:
        raise ValueError(f"Unexpected source artifact shard for trainer {trainer_id}")
    count = shard.get("communicate_node_count")
    if type(count) is not int or count < 0:
        raise ValueError(f"Invalid communication row count for trainer {trainer_id}")
    index_path = (
        artifact_dir
        / "shards"
        / f"trainer-{trainer_id:03d}"
        / _COMMUNICATION_INDEX_FILE
    )
    digest = _file_sha256(index_path)
    checksums = shard.get("sha256", {})
    if not isinstance(checksums, dict) or (
        _COMMUNICATION_INDEX_FILE in checksums
        and checksums[_COMMUNICATION_INDEX_FILE] != digest
    ):
        raise ValueError(
            f"Communication index checksum mismatch for trainer {trainer_id}"
        )
    return count, digest


def _read_feature_files(
    rank_dir: Path, row_count: int, original_feature_dim: int
) -> tuple[dict[str, Any], str, Optional[str]]:
    metadata = _read_json(rank_dir / _FEATURE_METADATA_FILE)
    width = metadata.get("feature_dim")
    if (
        metadata.get("store_version") != _FEATURE_AGGREGATION_STORE_VERSION
        or metadata.get("dtype") != "float32"
        or metadata.get("layout") != "row_major"
        or type(metadata.get("row_count")) is not int
        or metadata["row_count"] != row_count
        or type(width) is not int
        or width < 1
    ):
        raise ValueError(f"Invalid finalized feature metadata in {rank_dir}")
    size = row_count * width * 4
    if (
        metadata.get("size_bytes") != size
        or (rank_dir / _FEATURE_FILE).stat().st_size != size
    ):
        raise ValueError(
            f"Feature cache file length does not match its shape in {rank_dir}"
        )
    provenance = metadata.get("provenance")
    if provenance is None:
        if width != original_feature_dim:
            raise ValueError("A reduced-width cache must record its PCA provenance")
        return metadata, "none", None
    if not isinstance(provenance, dict) or (
        provenance.get("feature_compression_mode") != "shared-pca"
        or provenance.get("feature_compression_original_dim") != original_feature_dim
        or provenance.get("runtime_feature_dim") != width
        or not _is_sha256(provenance.get("feature_compression_basis_sha256"))
    ):
        raise ValueError(f"Invalid shared PCA provenance in {rank_dir}")
    return metadata, "shared-pca", provenance["feature_compression_basis_sha256"]


def promote_feature_cache_rank(
    *,
    pretrain_dir: Path,
    artifact_dir: Path,
    cache_dir: Path,
    trainer_id: int,
    norm_type: str,
    copy: bool = False,
) -> FeatureAggregationCacheManifest:
    """Link or copy one rank from a completed run using its known source artifact."""
    pretrain_dir = pretrain_dir.resolve()
    artifact_dir = artifact_dir.resolve()
    cache_dir = cache_dir.resolve()
    if any(
        cache_dir.is_relative_to(source) or source.is_relative_to(cache_dir)
        for source in (pretrain_dir, artifact_dir)
    ):
        raise ValueError(
            "Use a cache directory separate from the source run and artifact"
        )
    artifact, artifact_sha256 = _read_artifact(artifact_dir)
    n_trainer = artifact.get("n_trainer")
    original_dim = artifact.get("feature_dim")
    if type(n_trainer) is not int or not 0 <= trainer_id < n_trainer:
        raise ValueError(f"Trainer {trainer_id} is outside the source artifact")
    if type(original_dim) is not int or original_dim < 1:
        raise ValueError("Invalid original feature width in the source artifact")

    rank_name = f"trainer-{trainer_id:03d}"
    source = pretrain_dir / rank_name
    destination = cache_dir / rank_name
    if destination.exists():
        raise FileExistsError(f"Feature cache rank already exists: {destination}")
    row_count, index_sha256 = _artifact_rows(artifact_dir, artifact, trainer_id)
    metadata, mode, basis_hash = _read_feature_files(source, row_count, original_dim)
    manifest = FeatureAggregationCacheManifest(
        cache_version=_CACHE_VERSION,
        source_run_id=pretrain_dir.name,
        artifact_manifest_sha256=artifact_sha256,
        n_trainer=n_trainer,
        norm_type=norm_type,
        original_feature_dim=original_dim,
        feature_dim=metadata["feature_dim"],
        feature_compression_mode=mode,
        feature_compression_basis_sha256=basis_hash,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / CACHE_MANIFEST_FILE
    if manifest_path.exists() and load_feature_cache_manifest(cache_dir) != manifest:
        raise ValueError(
            "Cannot mix different source runs or bases in one feature cache"
        )

    # Publish the rank only after both files and their fingerprints are ready.
    with tempfile.TemporaryDirectory(dir=cache_dir, prefix=".promote-") as tmp:
        stage = Path(tmp)
        rank_stage = stage / rank_name
        rank_stage.mkdir()
        src_file, staged_file = source / _FEATURE_FILE, rank_stage / _FEATURE_FILE
        if copy:
            shutil.copyfile(src_file, staged_file)
        else:
            try:
                os.link(src_file, staged_file)
            except OSError as error:
                if error.errno == errno.EXDEV:
                    raise ValueError(
                        "Feature cache is on another filesystem; use --copy"
                    ) from error
                raise
        digest = _file_sha256(staged_file)
        if staged_file.stat().st_size != metadata["size_bytes"] or (
            copy and digest != _file_sha256(src_file)
        ):
            raise ValueError(
                f"Feature payload changed while promoting trainer {trainer_id}"
            )
        _write_json(rank_stage / _FEATURE_METADATA_FILE, metadata)
        _write_json(
            rank_stage / _RANK_MANIFEST_FILE,
            {
                "cache_id": manifest.cache_id,
                "trainer_id": trainer_id,
                "communicate_node_index_sha256": index_sha256,
                "feature_payload_sha256": digest,
            },
        )
        tmp_manifest = stage / CACHE_MANIFEST_FILE
        _write_json(tmp_manifest, asdict(manifest))
        try:
            os.link(tmp_manifest, manifest_path)
        except FileExistsError:
            if load_feature_cache_manifest(cache_dir) != manifest:
                raise ValueError(
                    "Cannot mix different source runs or bases in one feature cache"
                )
        rank_stage.rename(destination)
    return manifest


def open_feature_cache_rank(
    *,
    cache_dir: Path,
    artifact_dir: Path,
    manifest: FeatureAggregationCacheManifest,
    trainer_id: int,
    expected_rows: Optional[int] = None,
    verify_payload: bool = False,
) -> tuple[torch.Tensor, Path]:
    """Validate one rank and privately map its immutable feature payload."""
    if not 0 <= trainer_id < manifest.n_trainer:
        raise ValueError(f"Trainer {trainer_id} is outside the feature cache")
    rank_dir = cache_dir / f"trainer-{trainer_id:03d}"
    path = rank_dir / _FEATURE_FILE
    if load_feature_cache_manifest(cache_dir) != manifest:
        raise ValueError("Worker cache manifest does not match the requested cache")
    artifact = validate_feature_cache_artifact(manifest, artifact_dir)
    row_count, index_sha256 = _artifact_rows(artifact_dir, artifact, trainer_id)
    if expected_rows is not None and row_count != expected_rows:
        raise ValueError(
            "Cache row count does not match the trainer's communication nodes"
        )
    receipt = _read_json(rank_dir / _RANK_MANIFEST_FILE)
    if (
        receipt.get("cache_id") != manifest.cache_id
        or receipt.get("trainer_id") != trainer_id
        or receipt.get("communicate_node_index_sha256") != index_sha256
    ):
        raise ValueError(
            "Feature cache rank, identity or communication row order does not match"
        )
    metadata, mode, basis_hash = _read_feature_files(
        rank_dir, row_count, manifest.original_feature_dim
    )
    if (
        metadata["feature_dim"] != manifest.feature_dim
        or mode != manifest.feature_compression_mode
        or basis_hash != manifest.feature_compression_basis_sha256
    ):
        raise ValueError("Feature cache width or PCA basis does not match the manifest")
    expected_sha256 = receipt.get("feature_payload_sha256")
    if not _is_sha256(expected_sha256) or (
        verify_payload and _file_sha256(path) != expected_sha256
    ):
        raise ValueError(f"Feature payload checksum mismatch for {path}")
    if row_count == 0:
        return torch.empty((0, manifest.feature_dim), dtype=torch.float32), path
    tensor = torch.from_file(
        str(path),
        size=row_count * manifest.feature_dim,
        dtype=torch.float32,
        shared=False,
    ).view(row_count, manifest.feature_dim)
    return tensor, path
