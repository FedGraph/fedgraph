from typing import Any, Dict

import torch

from ..trainer_class import Trainer_General
from .compression_utils import svd_compress, svd_decompress


class Trainer_General_LowRank(Trainer_General):
    """
    Enhanced trainer class with low-rank compression support.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_lowrank = getattr(self.args, "use_lowrank", False)

    def get_compressed_params(self) -> Dict[str, Any]:
        """
        Get model parameters with optional compression.
        """
        if not self.use_lowrank:
            return {"params": dict(self.model.named_parameters()), "compressed": False}

        params = {
            name: param.data.cpu().detach()
            for name, param in self.model.named_parameters()
        }

        compressed_params = {}

        for name, param in params.items():
            if param.dim() == 2 and min(param.shape) > 1:
                # Use fixed rank for simplicity
                rank = getattr(self.args, "fixed_rank", 10)
                max_possible_rank = min(param.shape)
                if rank > max_possible_rank:
                    print(
                        f"Warning: rank {rank} > max possible {max_possible_rank} for {name}, using {max_possible_rank}"
                    )
                    rank = max_possible_rank
                U, S, V = svd_compress(param, rank)
                compressed_params[name] = {"U": U, "S": S, "V": V, "rank": rank}
            else:
                compressed_params[name] = param

        return {"params": compressed_params, "compressed": True}

    def update_compressed_params(
        self, compressed_data: Dict[str, Any], current_global_epoch: int
    ) -> None:
        """
        Update model parameters from compressed representation.
        """
        if not compressed_data.get("compressed", False):
            # Standard parameter update
            params = compressed_data["params"]
            self.model.to("cpu")
            for name, param in self.model.named_parameters():
                if name in params:
                    param.data.copy_(params[name])
            self.model.to(self.device)
            return

        # Decompress and update
        self.model.to("cpu")
        compressed_params = compressed_data["params"]

        for name, param in self.model.named_parameters():
            if name in compressed_params:
                param_data = compressed_params[name]
                if isinstance(param_data, dict) and "U" in param_data:
                    # Decompress SVD
                    reconstructed = svd_decompress(
                        param_data["U"], param_data["S"], param_data["V"]
                    )
                    param.data.copy_(reconstructed)
                else:
                    # Direct copy
                    param.data.copy_(param_data)

        self.model.to(self.device)
