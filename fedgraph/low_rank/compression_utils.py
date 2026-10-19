from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def svd_compress(
    tensor: torch.Tensor, rank: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compress a tensor using SVD decomposition.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor to compress (2D)
    rank : int
        Target rank for compression

    Returns
    -------
    U, S, V : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        SVD components with reduced rank
    """
    if tensor.dim() != 2:
        raise ValueError("SVD compression only supports 2D tensors")

    U, S, V = torch.svd(tensor)

    rank = min(rank, min(tensor.shape), len(S))
    U_compressed = U[:, :rank]
    S_compressed = S[:rank]
    V_compressed = V[:, :rank]

    return U_compressed, S_compressed, V_compressed


def svd_decompress(U: torch.Tensor, S: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct tensor from SVD components.

    Parameters
    ----------
    U, S, V : torch.Tensor
        SVD components

    Returns
    -------
    torch.Tensor
        Reconstructed tensor
    """
    return torch.mm(torch.mm(U, torch.diag(S)), V.t())


def calculate_compression_ratio(original_shape: Tuple[int, int], rank: int) -> float:
    """
    Calculate compression ratio for given rank.

    Parameters
    ----------
    original_shape : Tuple[int, int]
        Shape of original tensor
    rank : int
        Compression rank

    Returns
    -------
    float
        Compression ratio
    """
    m, n = original_shape
    original_size = m * n
    compressed_size = rank * (m + n + 1)  # U + S + V
    return original_size / compressed_size


def auto_select_rank(
    tensor: torch.Tensor, compression_ratio: float = 2.0, energy_threshold: float = 0.95
) -> int:
    """
    Automatically select rank based on compression ratio or energy preservation.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor
    compression_ratio : float
        Desired compression ratio
    energy_threshold : float
        Fraction of energy to preserve

    Returns
    -------
    int
        Selected rank
    """
    m, n = tensor.shape
    max_rank = min(m, n)

    target_size = (m * n) / compression_ratio
    rank_from_ratio = int((target_size - m - n) / (m + n + 1))
    rank_from_ratio = max(1, min(rank_from_ratio, max_rank))

    _, S, _ = torch.svd(tensor)
    total_energy = torch.sum(S**2)
    cumulative_energy = torch.cumsum(S**2, dim=0)
    energy_ratios = cumulative_energy / total_energy

    rank_from_energy = torch.sum(energy_ratios < energy_threshold).item() + 1
    rank_from_energy = min(rank_from_energy, max_rank)

    return min(rank_from_ratio, rank_from_energy)


# ---------------------------------------------------------------------------
# Shared-basis (federated PCA) variant of low-rank pretraining.
#
# Every client projects its pretraining aggregate Z_k onto ONE basis W shared by
# all clients, so the projections Z_k W sum under encryption like the standard
# mode (Proposition 1 of the paper).  W is the top-r left singular subspace of a
# randomized sketch of the pooled covariance sum_k Z_k^T Z_k: each client
# contributes Y_k = Z_k^T (Z_k Omega) for a common Gaussian test matrix Omega
# (d x ell, ell = 4r by default), the server sums the Y_k (under encryption in the HE path)
# and takes the top-r left singular vectors of the sum.  Only the aggregate
# sketch is ever decrypted.
# ---------------------------------------------------------------------------
def make_omega(d: int, ell: int, seed: int = 0) -> torch.Tensor:
    """Common Gaussian test matrix Omega (d x ell), reproducible from ``seed``."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(d, ell, generator=g, dtype=torch.float32)


def basis_sketch(Z: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Client-side sketch Y_k = Z_k^T (Z_k Omega), shape d x ell."""
    Z = Z.float()
    return Z.t() @ (Z @ omega)


def shared_basis_from_sketch(Y: torch.Tensor, rank: int) -> torch.Tensor:
    """Top-r left singular vectors of the aggregated sketch Y (d x ell) -> W (d x r)."""
    U, _, _ = torch.linalg.svd(Y.float(), full_matrices=False)
    return U[:, : min(rank, U.shape[1])].contiguous()
