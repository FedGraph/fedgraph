"""Bounded trainer-side projection using a shared feature basis."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .shared_pca import SharedFeatureBasis

_PROJECTED_FEATURE_STORE_VERSION = 1
_SHARED_FEATURE_BASIS_STORE_VERSION = 1


@dataclass(frozen=True)
class ProjectedFeatureArtifact:
    """Published file-backed feature projection and its identity."""

    tensor: torch.Tensor
    data_path: Path
    metadata_path: Path
    basis_sha256: str


@dataclass(frozen=True)
class SharedFeatureBasisArtifact:
    """Persisted shared basis required to transform inference-time features."""

    basis: SharedFeatureBasis
    data_path: Path
    metadata_path: Path
    basis_sha256: str


def feature_basis_sha256(basis: SharedFeatureBasis) -> str:
    """Hash the float32 basis that is applied to runtime features."""
    matrix = basis.matrix.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(matrix.numpy().tobytes()).hexdigest()


def save_shared_feature_basis(
    basis: SharedFeatureBasis,
    *,
    output_dir: Path,
) -> SharedFeatureBasisArtifact:
    """Atomically persist the common projection basis and its provenance."""
    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    basis_hash = feature_basis_sha256(basis)
    stem = f"shared-pca-r{basis.rank}-{basis_hash[:16]}"
    data_path = resolved_output_dir / f"{stem}.pt"
    metadata_path = resolved_output_dir / f"{stem}.json"
    temporary_suffix = f"tmp-{os.getpid()}-{time.time_ns()}"
    temporary_data_path = resolved_output_dir / f"{stem}.pt.{temporary_suffix}"
    temporary_metadata_path = resolved_output_dir / (f"{stem}.json.{temporary_suffix}")

    payload = {
        "matrix": basis.matrix.detach().cpu().contiguous(),
        "eigenvalues": basis.eigenvalues.detach().cpu().contiguous(),
        "original_dim": basis.original_dim,
        "rank": basis.rank,
        "retained_energy": basis.retained_energy,
        "total_energy": basis.total_energy,
    }
    metadata = {
        "store_version": _SHARED_FEATURE_BASIS_STORE_VERSION,
        "projection": "uncentered_pca",
        "basis_sha256": basis_hash,
        "matrix_dtype": str(basis.matrix.dtype).removeprefix("torch."),
        "eigenvalue_dtype": str(basis.eigenvalues.dtype).removeprefix("torch."),
        "original_dim": basis.original_dim,
        "compressed_dim": basis.rank,
        "retained_energy": basis.retained_energy,
        "total_energy": basis.total_energy,
    }
    try:
        torch.save(payload, temporary_data_path)
        temporary_metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_data_path, data_path)
        os.replace(temporary_metadata_path, metadata_path)
    except Exception:
        temporary_data_path.unlink(missing_ok=True)
        temporary_metadata_path.unlink(missing_ok=True)
        raise

    return SharedFeatureBasisArtifact(
        basis=basis,
        data_path=data_path,
        metadata_path=metadata_path,
        basis_sha256=basis_hash,
    )


def load_shared_feature_basis(data_path: Path) -> SharedFeatureBasisArtifact:
    """Load and verify a persisted shared basis for later feature projection."""
    resolved_data_path = data_path.expanduser().resolve()
    metadata_path = resolved_data_path.with_suffix(".json")
    if not resolved_data_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("shared feature basis data or metadata is missing")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("store_version") != _SHARED_FEATURE_BASIS_STORE_VERSION:
        raise ValueError("unsupported shared feature basis store version")
    payload = torch.load(resolved_data_path, map_location="cpu", weights_only=True)
    basis = SharedFeatureBasis(
        matrix=payload["matrix"],
        eigenvalues=payload["eigenvalues"],
        original_dim=int(payload["original_dim"]),
        rank=int(payload["rank"]),
        retained_energy=float(payload["retained_energy"]),
        total_energy=float(payload["total_energy"]),
    )
    basis_hash = feature_basis_sha256(basis)
    if metadata.get("basis_sha256") != basis_hash:
        raise ValueError("shared feature basis hash does not match its metadata")
    if (
        metadata.get("original_dim") != basis.original_dim
        or metadata.get("compressed_dim") != basis.rank
    ):
        raise ValueError("shared feature basis shape does not match its metadata")
    return SharedFeatureBasisArtifact(
        basis=basis,
        data_path=resolved_data_path,
        metadata_path=metadata_path,
        basis_sha256=basis_hash,
    )


def _validate_projection_inputs(
    features: torch.Tensor,
    basis: SharedFeatureBasis,
    chunk_rows: int,
) -> None:
    if features.device.type != "cpu":
        raise ValueError("bounded feature projection requires a CPU tensor")
    if features.layout != torch.strided or features.ndim != 2:
        raise ValueError("features must be a dense two-dimensional tensor")
    if not features.is_floating_point():
        raise ValueError("features must use a floating-point dtype")
    if features.size(1) != basis.original_dim:
        raise ValueError(
            "feature dimension does not match the shared basis: "
            f"{features.size(1)} != {basis.original_dim}"
        )
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")


def _iter_projected_feature_chunks(
    features: torch.Tensor,
    basis: SharedFeatureBasis,
    chunk_rows: int,
) -> Iterator[tuple[int, torch.Tensor]]:
    projection_basis = basis.matrix.detach().to(device="cpu", dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, features.size(0), chunk_rows):
            source = features[start : start + chunk_rows].to(dtype=torch.float32)
            yield start, torch.mm(source, projection_basis)


def project_features_in_memory(
    features: torch.Tensor,
    basis: SharedFeatureBasis,
    *,
    chunk_rows: int,
) -> torch.Tensor:
    """Project CPU features in chunks into one float32 CPU tensor."""
    _validate_projection_inputs(features, basis, chunk_rows)
    projected = torch.empty(
        (features.size(0), basis.rank),
        dtype=torch.float32,
        device="cpu",
    )
    for start, chunk in _iter_projected_feature_chunks(features, basis, chunk_rows):
        projected[start : start + chunk.size(0)].copy_(chunk)
    return projected


def project_features_to_file(
    features: torch.Tensor,
    basis: SharedFeatureBasis,
    *,
    chunk_rows: int,
    output_dir: Path,
    trainer_id: int,
) -> ProjectedFeatureArtifact:
    """Project CPU features into an atomically published float32 mmap file."""
    _validate_projection_inputs(features, basis, chunk_rows)
    if trainer_id < 0:
        raise ValueError("trainer_id must be non-negative")

    basis_hash = feature_basis_sha256(basis)
    trainer_dir = output_dir.expanduser().resolve() / f"trainer-{trainer_id:03d}"
    trainer_dir.mkdir(parents=True, exist_ok=True)
    stem = f"features-r{basis.rank}-{basis_hash[:16]}"
    data_path = trainer_dir / f"{stem}.f32"
    metadata_path = trainer_dir / f"{stem}.json"
    temporary_suffix = f"tmp-{os.getpid()}-{time.time_ns()}"
    temporary_data_path = trainer_dir / f"{stem}.f32.{temporary_suffix}"
    temporary_metadata_path = trainer_dir / f"{stem}.json.{temporary_suffix}"
    row_count = int(features.size(0))
    storage: np.memmap | None = None

    try:
        if row_count == 0:
            temporary_data_path.touch()
        else:
            storage = np.memmap(
                temporary_data_path,
                dtype=np.float32,
                mode="w+",
                shape=(row_count, basis.rank),
            )
            for start, chunk in _iter_projected_feature_chunks(
                features, basis, chunk_rows
            ):
                stop = start + chunk.size(0)
                storage[start:stop] = chunk.contiguous().numpy()
            storage.flush()
            del storage
            storage = None

        metadata = {
            "store_version": _PROJECTED_FEATURE_STORE_VERSION,
            "projection": "uncentered_pca",
            "dtype": "float32",
            "layout": "row_major",
            "row_count": row_count,
            "original_dim": basis.original_dim,
            "compressed_dim": basis.rank,
            "basis_sha256": basis_hash,
            "retained_energy": basis.retained_energy,
            "size_bytes": row_count * basis.rank * np.dtype(np.float32).itemsize,
        }
        temporary_metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_data_path, data_path)
        os.replace(temporary_metadata_path, metadata_path)
    except Exception:
        if storage is not None:
            storage.flush()
            del storage
        temporary_data_path.unlink(missing_ok=True)
        temporary_metadata_path.unlink(missing_ok=True)
        raise

    if row_count == 0:
        tensor = torch.empty((0, basis.rank), dtype=torch.float32)
    else:
        tensor = torch.from_file(
            str(data_path),
            shared=False,
            size=row_count * basis.rank,
            dtype=torch.float32,
        ).view(row_count, basis.rank)
    return ProjectedFeatureArtifact(
        tensor=tensor,
        data_path=data_path,
        metadata_path=metadata_path,
        basis_sha256=basis_hash,
    )
