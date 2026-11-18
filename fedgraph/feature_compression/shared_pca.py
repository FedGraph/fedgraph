"""Bounded-memory shared feature projection based on uncentered PCA.

For horizontally partitioned features, each trainer can calculate its local
second moment independently. Summing those small matrices produces the exact
global second moment without collecting feature rows on the server::

    X.T @ X = sum(X_i.T @ X_i)

The leading eigenvectors of that matrix are the right singular vectors of X.
This module deliberately contains no FedGraph, Ray, or storage-lifecycle code
so the compression method remains optional and detachable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class FeatureSecondMoment:
    """Additive statistics for one horizontal partition of feature rows."""

    matrix: torch.Tensor
    row_count: int
    feature_dim: int

    def __post_init__(self) -> None:
        if self.row_count < 0:
            raise ValueError("row_count must be non-negative")
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if self.matrix.device.type != "cpu":
            raise ValueError("second-moment matrix must be on CPU")
        if self.matrix.dtype != torch.float64:
            raise ValueError("second-moment matrix must use float64")
        if self.matrix.shape != (self.feature_dim, self.feature_dim):
            raise ValueError("second-moment matrix has an invalid shape")
        if not torch.isfinite(self.matrix).all():
            raise ValueError("second-moment matrix contains non-finite values")


@dataclass(frozen=True)
class SharedFeatureBasis:
    """Top right-singular-vector basis and its retained-energy metadata."""

    matrix: torch.Tensor
    eigenvalues: torch.Tensor
    original_dim: int
    rank: int
    retained_energy: float
    total_energy: float

    def __post_init__(self) -> None:
        if self.original_dim < 1:
            raise ValueError("original_dim must be positive")
        if not 1 <= self.rank <= self.original_dim:
            raise ValueError("rank must be between 1 and original_dim")
        if self.matrix.device.type != "cpu" or self.eigenvalues.device.type != "cpu":
            raise ValueError("shared feature basis must be on CPU")
        if self.matrix.shape != (self.original_dim, self.rank):
            raise ValueError("basis matrix has an invalid shape")
        if self.eigenvalues.shape != (self.original_dim,):
            raise ValueError("eigenvalue tensor has an invalid shape")
        if (
            not self.matrix.is_floating_point()
            or self.eigenvalues.dtype != torch.float64
        ):
            raise ValueError("shared feature basis has an invalid dtype")
        if (
            not torch.isfinite(self.matrix).all()
            or not torch.isfinite(self.eigenvalues).all()
        ):
            raise ValueError("shared feature basis contains non-finite values")
        if not 0.0 <= self.retained_energy <= 1.0:
            raise ValueError("retained_energy must be between zero and one")
        if self.total_energy < 0.0:
            raise ValueError("total_energy must be non-negative")


def compute_feature_second_moment(
    features: torch.Tensor,
    *,
    chunk_rows: int,
) -> FeatureSecondMoment:
    """Calculate ``features.T @ features`` using bounded CPU row chunks.

    ``features`` may be an ordinary CPU tensor or a file-backed CPU tensor.
    Only the current row chunk is converted to float64, avoiding a float64
    copy of the complete feature matrix.
    """
    if features.device.type != "cpu":
        raise ValueError("bounded feature statistics require a CPU tensor")
    if features.layout != torch.strided or features.ndim != 2:
        raise ValueError("features must be a dense two-dimensional tensor")
    if not features.is_floating_point():
        raise ValueError("features must use a floating-point dtype")
    if features.size(1) < 1:
        raise ValueError("features must contain at least one feature column")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    row_count, feature_dim = features.shape
    second_moment = torch.zeros((feature_dim, feature_dim), dtype=torch.float64)
    for start in range(0, row_count, chunk_rows):
        chunk = features[start : start + chunk_rows].to(dtype=torch.float64)
        second_moment.addmm_(chunk.T, chunk)

    return FeatureSecondMoment(
        matrix=second_moment,
        row_count=row_count,
        feature_dim=feature_dim,
    )


def aggregate_feature_second_moments(
    statistics: Iterable[FeatureSecondMoment],
) -> FeatureSecondMoment:
    """Sum trainer statistics into the exact global second moment."""
    iterator = iter(statistics)
    try:
        first = next(iterator)
    except StopIteration as error:
        raise ValueError("at least one feature statistic is required") from error

    feature_dim = first.feature_dim
    total = first.matrix.clone()
    row_count = first.row_count
    for statistic in iterator:
        if statistic.feature_dim != feature_dim:
            raise ValueError("all feature statistics must have the same dimension")
        total.add_(statistic.matrix)
        row_count += statistic.row_count

    return FeatureSecondMoment(
        matrix=total,
        row_count=row_count,
        feature_dim=feature_dim,
    )


def _canonicalize_basis_signs(basis: torch.Tensor) -> torch.Tensor:
    """Resolve each eigenvector's arbitrary sign deterministically."""
    result = basis.clone()
    pivot_rows = result.abs().argmax(dim=0)
    columns = torch.arange(result.size(1), device=result.device)
    signs = torch.sign(result[pivot_rows, columns])
    signs[signs == 0] = 1
    result.mul_(signs)
    return result


def solve_shared_feature_basis(
    statistic: FeatureSecondMoment,
    *,
    rank: int,
    output_dtype: torch.dtype = torch.float32,
) -> SharedFeatureBasis:
    """Solve a shared rank-``rank`` basis from a global second moment."""
    if not 1 <= rank <= statistic.feature_dim:
        raise ValueError("rank must be between 1 and the feature dimension")
    if output_dtype not in {torch.float32, torch.float64}:
        raise ValueError("output_dtype must be float32 or float64")

    # Roundoff can make an accumulated second moment very slightly asymmetric.
    symmetric_moment = (statistic.matrix + statistic.matrix.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_moment)
    eigenvalues = eigenvalues.flip(0).clamp_min_(0).contiguous()
    basis = eigenvectors.flip(1)[:, :rank].contiguous()
    basis = _canonicalize_basis_signs(basis).to(dtype=output_dtype)

    total_energy = float(eigenvalues.sum().item())
    retained_energy = (
        float(eigenvalues[:rank].sum().item()) / total_energy
        if total_energy > 0.0
        else 1.0
    )
    return SharedFeatureBasis(
        matrix=basis,
        eigenvalues=eigenvalues,
        original_dim=statistic.feature_dim,
        rank=rank,
        retained_energy=retained_energy,
        total_energy=total_energy,
    )
