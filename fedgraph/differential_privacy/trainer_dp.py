import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..trainer_class import Trainer_General
from ..utils_nc import get_1hop_feature_sum


class Trainer_General_DP(Trainer_General):
    """
    Enhanced trainer class with Differential Privacy support.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_dp = getattr(self.args, "use_dp", False)

        if self.use_dp:
            print(f"Trainer {self.rank} initialized with DP support")

    def get_dp_local_feature_sum(self) -> Tuple[torch.Tensor, Dict]:
        """
        Get local feature sum with optional client-side DP preprocessing.

        Returns
        -------
        Tuple[torch.Tensor, Dict]
            Local feature sum and computation statistics
        """
        computation_start = time.time()

        # Compute feature sum (same as original)
        new_feature_for_trainer = torch.zeros(
            self.global_node_num, self.features.shape[1]
        ).to(self.device)
        new_feature_for_trainer[self.local_node_index] = self.features

        one_hop_neighbor_feature_sum = get_1hop_feature_sum(
            new_feature_for_trainer, self.adj, self.device
        )

        computation_time = time.time() - computation_start

        # Compute statistics for DP
        feature_sum_norm = torch.norm(one_hop_neighbor_feature_sum).item()
        data_size = (
            one_hop_neighbor_feature_sum.element_size()
            * one_hop_neighbor_feature_sum.nelement()
        )

        stats = {
            "trainer_id": self.rank,
            "computation_time": computation_time,
            "feature_sum_norm": feature_sum_norm,
            "data_size": data_size,
            "shape": one_hop_neighbor_feature_sum.shape,
        }

        print(f"Trainer {self.rank} - DP feature sum computed:")
        print(f"  Norm: {feature_sum_norm:.4f}")
        print(f"  Shape: {one_hop_neighbor_feature_sum.shape}")
        print(f"  Computation time: {computation_time:.4f}s")

        return one_hop_neighbor_feature_sum, stats
