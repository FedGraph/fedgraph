import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..server_class import Server
from .dp_mechanisms import DPAccountant, DPMechanism


class Server_DP(Server):
    """
    Enhanced server class with Differential Privacy support for FedGCN.
    Extends the original Server class to support DP in pre-training aggregation.
    """

    def __init__(
        self,
        feature_dim: int,
        args_hidden: int,
        class_num: int,
        device: torch.device,
        trainers: list,
        args: Any,
    ):
        super().__init__(feature_dim, args_hidden, class_num, device, trainers, args)

        # DP configuration
        self.use_dp = getattr(args, "use_dp", False)

        if self.use_dp:
            self.dp_epsilon = getattr(args, "dp_epsilon", 1.0)
            self.dp_delta = getattr(args, "dp_delta", 1e-5)
            self.dp_sensitivity = getattr(args, "dp_sensitivity", 1.0)
            self.dp_mechanism = getattr(args, "dp_mechanism", "gaussian")
            self.dp_clip_norm = getattr(args, "dp_clip_norm", 1.0)

            # Initialize DP mechanism
            self.dp_mechanism_obj = DPMechanism(
                epsilon=self.dp_epsilon,
                delta=self.dp_delta,
                sensitivity=self.dp_sensitivity,
                mechanism=self.dp_mechanism,
            )

            # Privacy accountant
            self.privacy_accountant = DPAccountant()

            print(f"Server initialized with Differential Privacy:")
            print(f"  Mechanism: {self.dp_mechanism}")
            print(f"  Privacy parameters: ε={self.dp_epsilon}, δ={self.dp_delta}")
            print(f"  Sensitivity: {self.dp_sensitivity}")
            print(f"  Clipping norm: {self.dp_clip_norm}")

    def aggregate_dp_feature_sums(
        self, local_feature_sums: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Aggregate feature sums with differential privacy.

        Parameters
        ----------
        local_feature_sums : List[torch.Tensor]
            List of local feature sums from trainers

        Returns
        -------
        Tuple[torch.Tensor, Dict]
            Aggregated feature sum with DP noise and statistics
        """
        aggregation_start = time.time()

        # Step 1: Clip individual contributions
        clipped_sums = []
        clipping_stats = []

        for i, local_sum in enumerate(local_feature_sums):
            original_norm = torch.norm(local_sum).item()
            clipped_sum = self.dp_mechanism_obj.clip_gradients(
                local_sum, self.dp_clip_norm
            )
            clipped_norm = torch.norm(clipped_sum).item()

            clipped_sums.append(clipped_sum)
            clipping_stats.append(
                {
                    "trainer_id": i,
                    "original_norm": original_norm,
                    "clipped_norm": clipped_norm,
                    "was_clipped": original_norm > self.dp_clip_norm,
                }
            )

        # Step 2: Aggregate clipped sums
        aggregated_sum = torch.stack(clipped_sums).sum(dim=0)

        # Step 3: Add DP noise
        noisy_aggregated_sum = self.dp_mechanism_obj.add_noise(aggregated_sum)

        aggregation_time = time.time() - aggregation_start

        # Step 4: Update privacy accountant
        self.privacy_accountant.add_step(self.dp_epsilon, self.dp_delta)

        # Statistics
        dp_stats = {
            "aggregation_time": aggregation_time,
            "clipping_stats": clipping_stats,
            "num_clipped": sum(1 for stat in clipping_stats if stat["was_clipped"]),
            "pre_noise_norm": torch.norm(aggregated_sum).item(),
            "post_noise_norm": torch.norm(noisy_aggregated_sum).item(),
            "noise_magnitude": torch.norm(noisy_aggregated_sum - aggregated_sum).item(),
            "privacy_spent": self.privacy_accountant.get_total_privacy_spent(),
        }

        return noisy_aggregated_sum, dp_stats

    def print_dp_stats(self, dp_stats: Dict):
        """Print differential privacy statistics."""
        print("\n=== Differential Privacy Statistics ===")
        print(f"Aggregation time: {dp_stats['aggregation_time']:.4f}s")
        print(
            f"Trainers clipped: {dp_stats['num_clipped']}/{len(dp_stats['clipping_stats'])}"
        )
        print(f"Pre-noise norm: {dp_stats['pre_noise_norm']:.4f}")
        print(f"Post-noise norm: {dp_stats['post_noise_norm']:.4f}")
        print(f"Noise magnitude: {dp_stats['noise_magnitude']:.4f}")

        total_eps, total_delta = dp_stats["privacy_spent"]
        print(f"Total privacy spent: ε={total_eps:.4f}, δ={total_delta:.8f}")

        # Per-trainer clipping details
        clipped_trainers = [
            stat for stat in dp_stats["clipping_stats"] if stat["was_clipped"]
        ]
        if clipped_trainers:
            print("Clipped trainers:")
            for stat in clipped_trainers:
                print(
                    f"  Trainer {stat['trainer_id']}: {stat['original_norm']:.4f} -> {stat['clipped_norm']:.4f}"
                )
