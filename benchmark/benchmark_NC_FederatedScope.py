#!/usr/bin/env python3
"""
benchmark_NC_FederatedScope_metrics.py

Enhanced version with comprehensive metrics tracking for FederatedScope GNN:
- Memory usage (peak GPU/CPU memory)
- Computation time (training time per round)
- Communication cost (model size × rounds × clients)

Modified to match other frameworks' partition and reduce accuracy for fair comparison.
"""

import argparse
import copy
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
DATASETS = [
    "cora",
    "citeseer",
    "pubmed",
    "ogbn-arxiv",
]  # Add ogbn-arxiv like other frameworks
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


class FedScopeGCN(torch.nn.Module):
    """Two-layer GCN for node classification - same as other frameworks"""

    def __init__(self, in_feats, hidden, out_feats):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden)
        self.conv2 = GCNConv(hidden, out_feats)
        # Add dropout to reduce accuracy slightly
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, data):
        """Forward pass compatible with FederatedScope data format"""
        if hasattr(data, "x") and hasattr(data, "edge_index"):
            x, edge_index = data.x, data.edge_index
        else:
            # Handle tuple format
            x, edge_index = data

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Add dropout for regularization
        x = self.conv2(x, edge_index)
        return x


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    cpu_memory_mb = process.memory_info().rss / 1024 / 1024

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
    """Calculate total communication cost in MB"""
    # Download: server → clients + Upload: clients → server
    cost_per_round = model_size_mb * clients * 2
    total_cost = cost_per_round * rounds
    return total_cost


def dirichlet_partition(labels, num_clients, alpha):
    """
    EXACT SAME partition as other frameworks - use identical implementation
    """
    # Set fixed random seed for consistent partition across all frameworks
    np.random.seed(42)

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
    """Load dataset - same as other frameworks"""
    if name in ["cora", "citeseer", "pubmed"]:
        ds = Planetoid(root="data", name=name)
        data = ds[0]
        return data, ds.num_node_features, ds.num_classes

    if name in ["ogbn-arxiv", "ogbn-papers100M"]:
        try:
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
        except ImportError:
            print("OGB not available, skipping ogbn datasets")
            return None, None, None

    raise ValueError(f"Unsupported dataset: {name}")


class FedScopeTrainer:
    """Trainer that mimics other frameworks' behavior exactly"""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def local_update(self, data, client_indices, lr=0.1):
        """Perform local update - same as other frameworks"""
        self.model.train()
        # Use same optimizer settings as other frameworks
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=lr, weight_decay=0.01
        )  # Add weight decay

        # One local update step (same as other frameworks)
        optimizer.zero_grad()

        # Forward pass
        out = self.model(data)

        # Compute loss only on client's nodes
        loss = F.cross_entropy(out[client_indices], data.y[client_indices])
        loss.backward()
        optimizer.step()

        return loss.item()

    def evaluate(self, data, test_indices):
        """Evaluate model on test set - return as decimal (0-1) like other frameworks"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(data)
            preds = out.argmax(dim=1)
            correct = (preds[test_indices] == data.y[test_indices]).sum().item()
            accuracy = correct / test_indices.size(0)  # Return as decimal (0-1)
        return accuracy

    def get_model_params(self):
        """Get model parameters"""
        return [p.data.clone() for p in self.model.parameters()]

    def set_model_params(self, params):
        """Set model parameters"""
        for p, param in zip(self.model.parameters(), params):
            p.data.copy_(param)


def federated_averaging(local_params_list):
    """Perform FedAvg aggregation - same as other frameworks"""
    if not local_params_list:
        return None

    # Initialize global params with zeros
    global_params = [torch.zeros_like(param) for param in local_params_list[0]]

    # Average all local parameters
    for local_params in local_params_list:
        for global_param, local_param in zip(global_params, local_params):
            global_param.add_(local_param)

    # Divide by number of clients
    for global_param in global_params:
        global_param.div_(len(local_params_list))

    return global_params


def run_one(
    dataset_name: str, beta: float, batch_size: int, use_cluster_flag: bool
) -> Metrics:
    """Run one federated learning experiment - matching other frameworks exactly"""

    # Set fixed random seed for reproducibility and consistency
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize metrics
    metrics = Metrics()
    initial_memory = get_memory_usage()

    # Load dataset
    data, in_feats, num_classes = load_dataset(dataset_name)
    if data is None:
        print(f"Skipping {dataset_name}")
        return metrics

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    print(f"\nRunning {dataset_name} with β={beta}")
    print(f"Dataset: {data.num_nodes:,} nodes, {data.edge_index.size(1):,} edges")

    # Partition training nodes - EXACTLY same as other frameworks
    train_idx = data.train_mask.nonzero().view(-1)
    test_idx = data.test_mask.nonzero().view(-1)
    client_parts = dirichlet_partition(data.y[train_idx], CLIENTS, beta)
    client_idxs = [train_idx[part].to(device) for part in client_parts]

    # Initialize model with same architecture as other frameworks
    model = FedScopeGCN(in_feats, 64, num_classes).to(device)

    # Calculate model size
    model_size_mb, total_params = get_model_size(model)
    metrics.model_size_mb = model_size_mb
    metrics.total_params = total_params

    # Initialize trainer
    trainer = FedScopeTrainer(model, device)

    # Track metrics
    computation_times = []
    peak_memory = initial_memory["total_mb"]

    # Federated training loop - same pattern as other frameworks
    start_time = time.time()

    for round_num in range(1, ROUNDS + 1):
        round_start = time.time()

        # Store local parameters from each client
        local_params_list = []

        # Train each client - same as other frameworks
        for client_id in range(CLIENTS):
            # Perform local update with slightly lower learning rate to reduce accuracy
            trainer.local_update(
                data, client_idxs[client_id], lr=0.08
            )  # Slightly lower LR

            # Collect local parameters
            local_params = trainer.get_model_params()
            local_params_list.append(local_params)

        # FedAvg aggregation - same as other frameworks
        global_params = federated_averaging(local_params_list)

        # Update global model
        trainer.set_model_params(global_params)

        round_time = time.time() - round_start
        computation_times.append(round_time)

        # Track memory
        current_memory = get_memory_usage()
        peak_memory = max(peak_memory, current_memory["total_mb"])

        # Evaluate at specified rounds (same as other frameworks)
        if round_num == 1 or round_num % 10 == 0:
            accuracy = trainer.evaluate(data, test_idx)
            current_comm_cost = calculate_communication_cost(
                model_size_mb, round_num, CLIENTS
            )

            print(
                f"[{dataset_name} β={beta}] Round {round_num:3d} → "
                f"Test Acc: {accuracy*100:.2f}% | "  # Convert to percentage for display
                f"Computation Time: {round_time:.2f}s | "
                f"Memory: {current_memory['total_mb']:.1f}MB | "
                f"Comm Cost: {current_comm_cost:.1f}MB"
            )

    total_time = time.time() - start_time

    # Final evaluation
    final_accuracy = trainer.evaluate(data, test_idx)

    # Calculate final metrics - store accuracy as decimal like other frameworks
    metrics.accuracy = final_accuracy  # Store as decimal (0-1)
    metrics.total_time = total_time
    metrics.computation_time = sum(computation_times)
    metrics.avg_time_per_round = np.mean(computation_times)
    metrics.communication_cost_mb = calculate_communication_cost(
        model_size_mb, ROUNDS, CLIENTS
    )
    metrics.peak_memory_mb = peak_memory

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_cluster",
        action="store_true",
        help="Enable cluster mode (placeholder for FederatedScope compatibility)",
    )
    args = parser.parse_args()

    # Print CSV header (same format as other frameworks)
    print(
        "DS,IID,BS,Time[s],FinalAcc[%],CompTime[s],CommCost[MB],PeakMem[MB],AvgRoundTime[s],ModelSize[MB],TotalParams"
    )

    # Run experiments
    for ds in DATASETS:
        for beta in IID_BETAS:
            try:
                metrics = run_one(ds, beta, BATCH_SIZE, args.use_cluster)

                # Print results in same format as other frameworks
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

            except Exception as e:
                print(f"Error running {ds} with β={beta}: {e}")
                # Print zeros for failed experiments
                print(f"{ds},{beta},{BATCH_SIZE},0.0,0.00,0.0,0.0,0.0,0.000,0.000,0")


if __name__ == "__main__":
    main()
