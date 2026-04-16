import random
import time
from typing import Any, Dict, List

import ray
import torch

from ..server_class import Server
from .compression_utils import auto_select_rank, svd_compress, svd_decompress


class Server_LowRank(Server):
    """
    Enhanced server class with low-rank compression support for FedAvg.
    """

    def __init__(
        self,
        feature_dim: int,
        args_hidden: int,
        class_num: int,
        device: torch.device,
        trainers: List[Any],
        args: Any,
    ):
        super().__init__(feature_dim, args_hidden, class_num, device, trainers, args)

        self.use_lowrank = getattr(args, "use_lowrank", False)
        self.lowrank_method = getattr(
            args, "lowrank_method", "fixed"
        )  # 'fixed', 'adaptive', 'energy'
        self.compression_ratio = getattr(args, "compression_ratio", 2.0)
        self.energy_threshold = getattr(args, "energy_threshold", 0.95)
        self.fixed_rank = getattr(args, "fixed_rank", 10)

        self.compression_stats = []

        print(f"Server initialized with low-rank compression: {self.use_lowrank}")
        if self.use_lowrank:
            print(f"Low-rank method: {self.lowrank_method}")
            if self.lowrank_method == "fixed":
                print(f"Fixed rank: {self.fixed_rank}")
            elif self.lowrank_method == "adaptive":
                print(f"Target compression ratio: {self.compression_ratio}")
            elif self.lowrank_method == "energy":
                print(f"Energy threshold: {self.energy_threshold}")

    def compress_params(self, params: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Compress model parameters using low-rank decomposition.

        Parameters
        ----------
        params : Dict[str, torch.Tensor]
            Model parameters to compress

        Returns
        -------
        Dict[str, Any]
            Compressed parameters with metadata
        """
        if not self.use_lowrank:
            return {"params": params, "compressed": False}

        compressed_params = {}
        compression_info = {}

        for name, param in params.items():
            if param.dim() == 2 and min(param.shape) > 1:  # Only compress 2D tensors
                # Select rank based on method
                if self.lowrank_method == "fixed":
                    rank = min(self.fixed_rank, min(param.shape))
                elif self.lowrank_method == "adaptive":
                    rank = auto_select_rank(param, self.compression_ratio, 0.95)
                elif self.lowrank_method == "energy":
                    rank = auto_select_rank(param, 10.0, self.energy_threshold)
                else:
                    rank = min(self.fixed_rank, min(param.shape))

                # Compress using SVD
                U, S, V = svd_compress(param, rank)
                compressed_params[name] = {"U": U, "S": S, "V": V, "rank": rank}

                original_size = param.numel()
                compressed_size = U.numel() + S.numel() + V.numel()
                ratio = original_size / compressed_size

                compression_info[name] = {
                    "original_shape": param.shape,
                    "rank": rank,
                    "compression_ratio": ratio,
                    "original_size": original_size,
                    "compressed_size": compressed_size,
                }
            else:
                compressed_params[name] = param
                compression_info[name] = {
                    "original_shape": param.shape,
                    "rank": None,
                    "compression_ratio": 1.0,
                    "original_size": param.numel(),
                    "compressed_size": param.numel(),
                }

        self.compression_stats.append(compression_info)
        return {
            "params": compressed_params,
            "compressed": True,
            "info": compression_info,
        }

    def decompress_params(
        self, compressed_data: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Decompress model parameters from low-rank representation.

        Parameters
        ----------
        compressed_data : Dict[str, Any]
            Compressed parameter data

        Returns
        -------
        Dict[str, torch.Tensor]
            Decompressed parameters
        """
        if not compressed_data.get("compressed", False):
            return compressed_data["params"]

        decompressed_params = {}
        compressed_params = compressed_data["params"]

        for name, param_data in compressed_params.items():
            if isinstance(param_data, dict) and "U" in param_data:
                U, S, V = param_data["U"], param_data["S"], param_data["V"]
                decompressed_params[name] = svd_decompress(U, S, V)
            else:
                decompressed_params[name] = param_data

        return decompressed_params

    @torch.no_grad()
    def train(
        self,
        current_global_epoch: int,
        sampling_type: str = "random",
        sample_ratio: float = 1,
    ) -> None:
        """
        Enhanced training with low-rank compression support.
        """
        if self.use_encryption:
            super().train(current_global_epoch, sampling_type, sample_ratio)
            return

        # Low-rank compression path
        assert 0 < sample_ratio <= 1, "Sample ratio must be between 0 and 1"
        num_samples = int(self.num_of_trainers * sample_ratio)

        if sampling_type == "random":
            selected_trainers_indices = random.sample(
                range(self.num_of_trainers), num_samples
            )
        elif sampling_type == "uniform":
            selected_trainers_indices = [
                (i + int(self.num_of_trainers * sample_ratio) * current_global_epoch)
                % self.num_of_trainers
                for i in range(num_samples)
            ]
        else:
            raise ValueError("sampling_type must be either 'random' or 'uniform'")

        for trainer_idx in selected_trainers_indices:
            self.trainers[trainer_idx].train.remote(current_global_epoch)

        if self.use_lowrank:
            params = [
                self.trainers[trainer_idx].get_compressed_params.remote()
                for trainer_idx in selected_trainers_indices
            ]

            self.zero_params()
            self.model = self.model.to("cpu")

            # Aggregate compressed parameters
            aggregated_compressed = self.aggregate_compressed_params(
                params, num_samples
            )

            # Decompress and update server model
            decompressed_params = self.decompress_params(aggregated_compressed)

            # Update server model
            for name, param in self.model.named_parameters():
                if name in decompressed_params:
                    param.data.copy_(decompressed_params[name])

            self.model = self.model.to(self.device)

            self.broadcast_compressed_params(
                current_global_epoch, aggregated_compressed
            )
        else:
            # Standard FedAvg
            super().train(current_global_epoch, sampling_type, sample_ratio)

    def aggregate_compressed_params(
        self, params_list: List, num_samples: int
    ) -> Dict[str, Any]:
        """
        Aggregate compressed parameters from multiple trainers.
        """
        # Wait for all parameters
        compressed_params_list = []
        while params_list:
            ready, params_list = ray.wait(params_list, num_returns=1)
            compressed_params_list.append(ray.get(ready[0]))

        if not compressed_params_list[0].get("compressed", False):
            return compressed_params_list[0]

        aggregated = {"params": {}, "compressed": True, "info": {}}

        param_names = list(compressed_params_list[0]["params"].keys())

        for name in param_names:
            first_param = compressed_params_list[0]["params"][name]

            if isinstance(first_param, dict) and "U" in first_param:
                rank = first_param["rank"]

                U_sum = torch.zeros_like(first_param["U"])
                S_sum = torch.zeros_like(first_param["S"])
                V_sum = torch.zeros_like(first_param["V"])

                for compressed_data in compressed_params_list:
                    param_data = compressed_data["params"][name]
                    U_sum += param_data["U"]
                    S_sum += param_data["S"]
                    V_sum += param_data["V"]

                aggregated_params = aggregated.get("params")
                if not isinstance(aggregated_params, dict):
                    aggregated_params = {}
                    aggregated["params"] = aggregated_params
                aggregated_params[name] = {
                    "U": U_sum / float(num_samples),
                    "S": S_sum / float(num_samples),
                    "V": V_sum / float(num_samples),
                    "rank": rank,
                }
            else:
                param_sum = torch.zeros_like(first_param)
                for compressed_data in compressed_params_list:
                    param_sum += compressed_data["params"][name]
                aggregated_params = aggregated.get("params")
                if not isinstance(aggregated_params, dict):
                    aggregated_params = {}
                    aggregated["params"] = aggregated_params
                aggregated_params[name] = param_sum / float(num_samples)

        return aggregated

    def broadcast_compressed_params(
        self, current_global_epoch: int, compressed_params: Dict[str, Any]
    ) -> None:
        """
        Broadcast compressed parameters to all trainers.
        """
        for trainer in self.trainers:
            trainer.update_compressed_params.remote(
                compressed_params, current_global_epoch
            )

    def print_compression_stats(self) -> None:
        """
        Print compression statistics.
        """
        if not self.compression_stats or not self.use_lowrank:
            return

        latest_stats = self.compression_stats[-1]
        total_original = sum(info["original_size"] for info in latest_stats.values())
        total_compressed = sum(
            info["compressed_size"] for info in latest_stats.values()
        )
        overall_ratio = (
            total_original / total_compressed if total_compressed > 0 else 1.0
        )

        print(f"\n=== Low-Rank Compression Statistics ===")
        print(f"Overall compression ratio: {overall_ratio:.2f}x")
        print(f"Total parameters: {total_original:,} -> {total_compressed:,}")
        print(f"Bandwidth savings: {(1 - 1/overall_ratio)*100:.1f}%")

        for name, info in latest_stats.items():
            if info["rank"] is not None:
                print(
                    f"{name}: {info['original_shape']} -> rank {info['rank']} "
                    f"(ratio: {info['compression_ratio']:.2f}x)"
                )

    def get_model_size(self) -> float:
        """
        Return total model parameter size in bytes, accounting for compression.
        """
        if not self.use_lowrank or not self.compression_stats:
            return super().get_model_size()

        latest_stats = self.compression_stats[-1]
        total_compressed_params = sum(
            info["compressed_size"] for info in latest_stats.values()
        )
        return total_compressed_params * 4  # float32
