"""Training settings shared by standard NC trainers and the driver."""

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NCTrainingOptions:
    optimizer: str
    dropout: float
    weight_decay: float
    lr_scheduler: str
    min_learning_rate: float
    model_head: str = "gcn"

    def learning_rate_at_round(
        self, initial_rate: float, rounds: int, round_index: int
    ) -> float:
        if self.lr_scheduler == "constant":
            return float(initial_rate)
        fraction = min(round_index / rounds, 1.0)
        cosine = (1.0 + math.cos(math.pi * fraction)) / 2.0
        return self.min_learning_rate + (initial_rate - self.min_learning_rate) * cosine


def _numeric_setting(args: Any, name: str, default: float) -> float:
    value = getattr(args, name, default)
    return float(value) if isinstance(value, (int, float)) else default


def resolve_nc_training_options(args: Any) -> NCTrainingOptions:
    optimizer = getattr(args, "optimizer", "sgd")
    optimizer = optimizer.lower() if isinstance(optimizer, str) else "sgd"
    scheduler = getattr(args, "lr_scheduler", "constant")
    scheduler = scheduler.lower() if isinstance(scheduler, str) else "constant"
    model_head = getattr(args, "model_head", "gcn")
    model_head = model_head.lower() if isinstance(model_head, str) else "gcn"
    dropout = _numeric_setting(args, "dropout", 0.5)
    weight_decay = _numeric_setting(args, "weight_decay", 5e-4)
    min_rate = _numeric_setting(args, "min_learning_rate", 0.0)
    initial_rate = _numeric_setting(args, "learning_rate", 0.0)

    if optimizer not in {"sgd", "adamw"}:
        raise ValueError("optimizer must be sgd or adamw")
    if scheduler not in {"constant", "cosine"}:
        raise ValueError("lr_scheduler must be constant or cosine")
    if model_head not in {"gcn", "gcn-skip"}:
        raise ValueError("model_head must be gcn or gcn-skip")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if not math.isfinite(min_rate) or min_rate < 0:
        raise ValueError("min_learning_rate must be finite and non-negative")
    if scheduler == "constant" and min_rate != 0:
        raise ValueError("min_learning_rate requires the cosine scheduler")

    if (
        optimizer != "sgd"
        or scheduler != "constant"
        or model_head != "gcn"
        or dropout != 0.5
        or weight_decay != 5e-4
    ):
        if (
            args.dataset != "ogbn-papers100M"
            or args.num_hops != 2
            or args.method != "FedGCN"
            or getattr(args, "use_encryption", False)
            or getattr(args, "use_lowrank", False)
        ):
            raise ValueError(
                "Training recipe overrides require standard plaintext 2-hop "
                "FedGCN on ogbn-papers100M"
            )
    if model_head == "gcn-skip" and args.num_layers != 2:
        raise ValueError("gcn-skip requires a two-layer AggreGCN")
    if optimizer == "adamw" or scheduler == "cosine":
        if not math.isfinite(initial_rate) or initial_rate <= 0:
            raise ValueError(
                "AdamW and cosine scheduling require a positive finite learning_rate"
            )
    if scheduler == "cosine":
        if getattr(args, "elastic_training", False) or int(args.global_rounds) < 1:
            raise ValueError(
                "cosine scheduling requires a positive fixed global_rounds budget"
            )
        if min_rate > initial_rate:
            raise ValueError("min_learning_rate must not exceed learning_rate")

    return NCTrainingOptions(
        optimizer, dropout, weight_decay, scheduler, min_rate, model_head
    )
