#!/usr/bin/env python3
"""
benchmark_NC_FedGraphNN_metrics.py

- Memory usage (peak GPU/CPU memory)
- Computation time (training time per round)
- Communication cost (model size × rounds × clients)
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# ─── Configuration ────────────────────────────────────────────────────────
DATASETS = ["cora", "citeseer", "pubmed", "ogbn-arxiv"]
IID_BETAS = [10000.0, 100.0, 10.0]
BATCH_SIZE = -1  # full-batch training
CLIENTS = 10
ROUNDS = 200
use_cluster = False

# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Metrics:
    """Container for all metrics"""

    accuracy: float = 0.0
    total_time: float = 0.0
    computation_time: float = 0.0
    communication_cost_mb: float = 0.0
    peak_memory_mb: float = 0.0
    avg_time_per_round: float = 0.0
    model_size_mb: float = 0.0
    total_params: int = 0


class ManualGCN(torch.nn.Module):
    """Simple two-layer GCN for node classification."""

    def __init__(self, in_feats: int, hidden: int, out_feats: int):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden)
        self.conv2 = GCNConv(hidden, out_feats)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in MB"""
    # CPU memory
    process = psutil.Process(os.getpid())
    cpu_memory_mb = process.memory_info().rss / 1024 / 1024

    # GPU memory if available
    gpu_memory_mb = 0
    if torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()

    return {
        "cpu_mb": cpu_memory_mb,
        "gpu_mb": gpu_memory_mb,
        "total_mb": cpu_memory_mb + gpu_memory_mb,
    }


def get_model_size(model: torch.nn.Module) -> Tuple[float, int]:
    """Calculate model size in MB and total parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    # Each parameter is float32 (4 bytes)
    model_size_mb = (total_params * 4) / 1024 / 1024
    return model_size_mb, total_params


def calculate_communication_cost(
    model_size_mb: float, rounds: int, clients: int
) -> float:
    """
    Calculate total communication cost in MB.
    Each round: server sends model to all clients + clients send models back
    """
    # Download: server → clients (model_size × clients)
    # Upload: clients → server (model_size × clients)
    cost_per_round = model_size_mb * clients * 2
    total_cost = cost_per_round * rounds
    return total_cost


def dirichlet_partition(labels, num_clients, alpha):
    """Partition node indices using Dirichlet distribution."""
    labels = labels.cpu().numpy()
    num_classes = labels.max() + 1
    idx_by_class = [np.where(labels == c)[0] for c in range(num_classes)]
    client_idxs = [[] for _ in range(num_clients)]

    for idx in idx_by_class:
        np.random.shuffle(idx)
        props = np.random.dirichlet([alpha] * num_clients)
        props = (props / props.sum()) * len(idx)
        counts = np.floor(props).astype(int)
        counts[-1] = len(idx) - counts[:-1].sum()
        start = 0
        for i, cnt in enumerate(counts):
            client_idxs[i].extend(idx[start : start + cnt])
            start += cnt

    return [torch.tensor(ci, dtype=torch.long) for ci in client_idxs]


def load_dataset(name):
    """Load dataset."""
    if name in ["cora", "citeseer", "pubmed"]:
        ds = Planetoid(root="data", name=name)
        data = ds[0]
        return data, ds.num_node_features, ds.num_classes

    if name in ["ogbn-arxiv", "ogbn-papers100M"]:
        from ogb.nodeproppred import PygNodePropPredDataset

        ds = PygNodePropPredDataset(name=name, root="data")
        data = ds[0]
        data.y = data.y.squeeze()
        split_idx = ds.get_idx_split()
        train_idx, test_idx = split_idx["train"], split_idx["test"]
        N = data.num_nodes
        train_mask = torch.zeros(N, dtype=torch.bool)
        test_mask = torch.zeros(N, dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True
        data.train_mask = train_mask
        data.test_mask = test_mask
        return data, data.x.size(1), int(data.y.max().item() + 1)

    raise ValueError(f"Unsupported dataset: {name}")


class GraphDataLoader:
    """Custom data loader wrapper for graph data."""

    def __init__(self, data, node_indices, batch_size=-1):
        self.data = data
        self.node_indices = node_indices
        self.batch_size = batch_size if batch_size > 0 else len(node_indices)

    def __iter__(self):
        batch_data = {
            "x": self.data.x,
            "edge_index": self.data.edge_index,
            "y": self.data.y[self.node_indices],
            "node_indices": self.node_indices,
        }
        yield batch_data

    def __len__(self):
        return 1


class FedMLGraphTrainer:
    """Custom trainer for GCN with metrics tracking."""

    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.computation_times = []

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        """Train with timing."""
        train_start = time.time()

        self.model.to(device)
        self.model.train()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=args.learning_rate)

        for batch in train_data:
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            y = batch["y"].to(device)
            node_indices = batch["node_indices"].to(device)

            optimizer.zero_grad()
            out = self.model(x, edge_index)
            loss = F.cross_entropy(out[node_indices], y)
            loss.backward()
            optimizer.step()

        train_time = time.time() - train_start
        self.computation_times.append(train_time)

        return len(train_data), loss.item()

    def test(self, test_data, device, args):
        """Evaluate the model."""
        self.model.to(device)
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_data:
                x = batch["x"].to(device)
                edge_index = batch["edge_index"].to(device)
                y = batch["y"].to(device)
                node_indices = batch["node_indices"].to(device)

                out = self.model(x, edge_index)
                preds = out[node_indices].argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        accuracy = correct / total if total > 0 else 0
        return total, 0.0, {"accuracy": accuracy}


def run_simulation_mode_with_metrics(
    dataset_name, beta, batch_size, data, in_feats, num_classes, client_idxs, test_idx
) -> Metrics:
    """Run federated learning with comprehensive metrics tracking."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = Metrics()

    # Track initial memory
    initial_memory = get_memory_usage()

    # Create data loaders
    train_data_list = []
    for c in range(CLIENTS):
        train_loader = GraphDataLoader(data, client_idxs[c], batch_size)
        train_data_list.append(train_loader)

    test_loader = GraphDataLoader(data, test_idx, -1)

    # Initialize model
    model = ManualGCN(in_feats, 64, num_classes).to(device)

    # Calculate model size
    model_size_mb, total_params = get_model_size(model)
    metrics.model_size_mb = model_size_mb
    metrics.total_params = total_params

    # Initialize training
    class Args:
        def __init__(self):
            self.learning_rate = 0.1
            self.weight_decay = 0.0

    args = Args()

    trainer = FedMLGraphTrainer(model, args)
    global_params = trainer.get_model_params()

    # Track computation time
    computation_times = []
    peak_memory = initial_memory["total_mb"]

    start_time = time.time()

    for round_idx in range(1, ROUNDS + 1):
        round_start = time.time()

        # Local training
        local_params = []
        for c in range(CLIENTS):
            # Set global params
            trainer.set_model_params(global_params)

            # Train locally
            trainer.train(train_data_list[c], device, args)

            # Get updated params
            local_params.append(trainer.get_model_params())

        # FedAvg aggregation
        global_params = {}
        for key in local_params[0].keys():
            global_params[key] = torch.stack(
                [lp[key].float() for lp in local_params]
            ).mean(0)

        round_time = time.time() - round_start
        computation_times.append(round_time)

        # Track memory
        current_memory = get_memory_usage()
        peak_memory = max(peak_memory, current_memory["total_mb"])

        # Evaluate at specific rounds
        if round_idx == 1 or round_idx % 10 == 0:
            trainer.set_model_params(global_params)
            _, _, test_metrics = trainer.test(test_loader, device, args)
            acc = test_metrics["accuracy"] * 100
            # Calculate current communication cost (theoretical)
            current_comm_cost = calculate_communication_cost(
                model_size_mb, round_idx, CLIENTS
            )
            print(
                f"[{dataset_name} β={beta}] Round {round_idx:3d} → "
                f"Test Acc: {acc:.2f}% | "
                f"Computation Time: {round_time:.2f}s | "
                f"Memory: {current_memory['total_mb']:.1f}MB | "
                f"Comm Cost: {current_comm_cost:.1f}MB"
            )

    total_time = time.time() - start_time

    # Final evaluation
    trainer.set_model_params(global_params)
    _, _, test_metrics = trainer.test(test_loader, device, args)

    # Calculate final metrics
    metrics.accuracy = test_metrics["accuracy"] * 100
    metrics.total_time = total_time
    metrics.computation_time = sum(computation_times)
    metrics.avg_time_per_round = np.mean(computation_times)
    metrics.communication_cost_mb = calculate_communication_cost(
        model_size_mb, ROUNDS, CLIENTS
    )
    metrics.peak_memory_mb = peak_memory

    return metrics


def run_one(
    dataset_name: str, beta: float, batch_size: int, use_cluster_flag: bool
) -> Metrics:
    """Run one experiment with metrics."""
    # Load graph data
    data, in_feats, num_classes = load_dataset(dataset_name)

    # Partition training nodes
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    test_idx = data.test_mask.nonzero(as_tuple=False).view(-1)
    client_parts = dirichlet_partition(data.y[train_idx], CLIENTS, beta)
    client_idxs = [train_idx[part] for part in client_parts]

    print(f"\nRunning {dataset_name} with β={beta}")
    print(f"Dataset: {data.num_nodes:,} nodes, {data.edge_index.size(1):,} edges")

    return run_simulation_mode_with_metrics(
        dataset_name,
        beta,
        batch_size,
        data,
        in_feats,
        num_classes,
        client_idxs,
        test_idx,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cluster", action="store_true")
    args = parser.parse_args()

    # Print enhanced CSV header
    print(
        "DS,IID,BS,Time[s],FinalAcc[%],CompTime[s],CommCost[MB],PeakMem[MB],AvgRoundTime[s],ModelSize[MB],TotalParams"
    )

    # Run experiments
    for ds in DATASETS:
        for beta in IID_BETAS:
            metrics = run_one(ds, beta, BATCH_SIZE, args.use_cluster)

            # Print comprehensive results
            print(
                f"{ds},{beta},{BATCH_SIZE},"
                f"{metrics.total_time:.1f},"
                f"{metrics.accuracy:.2f},"
                f"{metrics.computation_time:.1f},"
                f"{metrics.communication_cost_mb:.1f},"
                f"{metrics.peak_memory_mb:.1f},"
                f"{metrics.avg_time_per_round:.3f},"
                f"{metrics.model_size_mb:.3f},"
                f"{metrics.total_params}"
            )


if __name__ == "__main__":
    main()
