import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

def svd_compress(tensor: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

def auto_select_rank(tensor: torch.Tensor, compression_ratio: float = 2.0, 
                    energy_threshold: float = 0.95) -> int:
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
    total_energy = torch.sum(S ** 2)
    cumulative_energy = torch.cumsum(S ** 2, dim=0)
    energy_ratios = cumulative_energy / total_energy
    
    rank_from_energy = torch.sum(energy_ratios < energy_threshold).item() + 1
    rank_from_energy = min(rank_from_energy, max_rank)
    

    return min(rank_from_ratio, rank_from_energy)