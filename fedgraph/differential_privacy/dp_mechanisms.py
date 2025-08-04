import torch
import numpy as np
import random
import time
from typing import Dict, List, Tuple, Optional, Any

class DPMechanism:
    """
    Differential Privacy mechanisms for federated learning.
    
    Supports multiple DP mechanisms:
    - Gaussian mechanism
    - Laplace mechanism  
    - Local DP with randomized response
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, 
                 sensitivity: float = 1.0, mechanism: str = "gaussian"):
        """
        Initialize DP mechanism.
        
        Parameters
        ----------
        epsilon : float
            Privacy budget (smaller = more private)
        delta : float  
            Failure probability for (ε,δ)-DP
        sensitivity : float
            L2 sensitivity of the function
        mechanism : str
            DP mechanism ("gaussian", "laplace", "local")
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.mechanism = mechanism
        
        # Calculate noise parameters
        if mechanism == "gaussian":
            # For (ε,δ)-DP: σ ≥ sqrt(2ln(1.25/δ)) * Δ / ε
            self.sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        elif mechanism == "laplace":
            # For ε-DP: b = Δ / ε
            self.scale = sensitivity / epsilon
        elif mechanism == "local":
            # For local DP
            self.p = np.exp(epsilon) / (np.exp(epsilon) + 1)
        
        print(f"Initialized {mechanism} DP mechanism:")
        print(f"  ε={epsilon}, δ={delta}, sensitivity={sensitivity}")
        if mechanism == "gaussian":
            print(f"  Gaussian noise σ={self.sigma:.4f}")
        elif mechanism == "laplace":
            print(f"  Laplace scale={self.scale:.4f}")

    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Add differential privacy noise to tensor.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor to add noise to
            
        Returns
        -------
        torch.Tensor
            Tensor with DP noise added
        """
        if self.mechanism == "gaussian":
            noise = torch.normal(0, self.sigma, size=tensor.shape, device=tensor.device)
            return tensor + noise
            
        elif self.mechanism == "laplace":
            # Laplace noise using exponential distribution
            uniform = torch.rand(tensor.shape, device=tensor.device)
            sign = torch.sign(uniform - 0.5)
            noise = -sign * self.scale * torch.log(1 - 2 * torch.abs(uniform - 0.5))
            return tensor + noise
            
        elif self.mechanism == "local":
            # Local DP with randomized response
            prob_matrix = torch.rand(tensor.shape, device=tensor.device)
            mask = prob_matrix < self.p
            # Flip with probability (1-p)
            noisy_tensor = tensor.clone()
            noisy_tensor[~mask] = -noisy_tensor[~mask]  # Simple bit flip for demonstration
            return noisy_tensor
            
        else:
            raise ValueError(f"Unknown mechanism: {self.mechanism}")

    def clip_gradients(self, tensor: torch.Tensor, max_norm: float) -> torch.Tensor:
        """
        Clip tensor to bound sensitivity.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor to clip
        max_norm : float
            Maximum L2 norm
            
        Returns
        -------
        torch.Tensor
            Clipped tensor
        """
        current_norm = torch.norm(tensor)
        if current_norm > max_norm:
            return tensor * (max_norm / current_norm)
        return tensor

    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get privacy budget spent."""
        return self.epsilon, self.delta


class DPAccountant:
    """
    Privacy accountant for tracking cumulative privacy loss.
    """
    
    def __init__(self):
        self.total_epsilon = 0.0
        self.total_delta = 0.0
        self.rounds = 0
        
    def add_step(self, epsilon: float, delta: float):
        """Add privacy cost of one step."""
        # Simple composition (can be improved with advanced composition)
        self.total_epsilon += epsilon
        self.total_delta += delta
        self.rounds += 1
        
    def get_total_privacy_spent(self) -> Tuple[float, float]:
        """Get total privacy spent."""
        return self.total_epsilon, self.total_delta
    
    def print_privacy_budget(self):
        """Print current privacy budget."""
        print(f"Privacy Budget Used: ε={self.total_epsilon:.4f}, δ={self.total_delta:.8f}")
        print(f"Rounds completed: {self.rounds}")

