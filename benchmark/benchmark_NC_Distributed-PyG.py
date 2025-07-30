#!/usr/bin/env python3
"""
benchmark_NC_Distributed-PyG_metrics.py

- Memory usage (peak GPU/CPU memory)
- Computation time (training time per round)
- Communication cost (model size × rounds × clients)
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# ─── Edit this list to choose which datasets to benchmark ────────────────
DATASETS = ["cora", "citeseer", "pubmed", "ogbn-arxiv"]
IID_BETAS = [10000.0, 100.0, 10.0]
BATCH_SIZE = -1  # full-batch training
CLIENTS = 10
ROUNDS = 200

# ─── Toggle cluster/RPC mode here (or via --use_cluster flag) ────────────
use_cluster = False

# ────────────────────────────────────────────────────────────────────────────


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


class GCN(torch.nn.Module):
    """Two-layer GCN for node classification."""

    def __init__(self, in_feats, hidden, out_feats):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden)
        self.conv2 = GCNConv(hidden, out_feats)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


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
    """
    Partition node indices per class over clients using a Dirichlet distribution.
    Returns a list of index tensors, one per client.
    """
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

    # convert to torch tensors
    return [torch.tensor(ci, dtype=torch.long) for ci in client_idxs]


def load_dataset(name):
    """
    Load one of Planetoid or OGBN-Arxiv / OGBN-Papers100M datasets.
    Returns (data, num_node_features, num_classes).
    """
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


def run_one(ds_name, beta, batch_size, use_cluster_flag):
    """
    Run one FedAvg simulation on the given dataset with metrics tracking:
      1) load data
      2) partition training nodes with Dirichlet(alpha=beta)
      3) federated training for ROUNDS rounds
      4) print per-round test accuracy at rounds 1,10,20,...
      5) return (elapsed_time, final_test_accuracy, metrics)
    """
    # Initialize metrics
    metrics = Metrics()

    # Track initial memory
    initial_memory = get_memory_usage()

    # 1) load dataset
    data, in_feats, num_classes = load_dataset(ds_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    print(f"\nRunning {ds_name} with β={beta}")
    print(f"Dataset: {data.num_nodes:,} nodes, {data.edge_index.size(1):,} edges")

    # 2) partition only training nodes
    train_idx = data.train_mask.nonzero().view(-1)
    client_parts = dirichlet_partition(data.y[train_idx], CLIENTS, beta)
    client_idxs = [train_idx[part] for part in client_parts]

    # 3) build model and global parameters
    model = GCN(in_feats, 64, num_classes).to(device)

    # Calculate model size
    model_size_mb, total_params = get_model_size(model)
    metrics.model_size_mb = model_size_mb
    metrics.total_params = total_params

    if use_cluster_flag:
        # placeholder for RPC/DDP initialization
        pass
    global_params = [p.data.clone() for p in model.parameters()]

    # Track computation time
    computation_times = []
    peak_memory = initial_memory["total_mb"]

    # 4) federated training loop
    t0 = time.time()
    for r in range(1, ROUNDS + 1):
        round_start = time.time()

        local_params = []
        for c in range(CLIENTS):
            # load global weights
            for p, gp in zip(model.parameters(), global_params):
                p.data.copy_(gp)

            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

            # one full-batch local update
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            idx = client_idxs[c].to(device)
            loss = F.cross_entropy(out[idx], data.y[idx])
            loss.backward()
            optimizer.step()

            local_params.append([p.data.clone() for p in model.parameters()])

        # FedAvg aggregation
        with torch.no_grad():
            for gp in global_params:
                gp.zero_()
            for lp in local_params:
                for gp, p in zip(global_params, lp):
                    gp.add_(p)
            for gp in global_params:
                gp.div_(CLIENTS)

        round_time = time.time() - round_start
        computation_times.append(round_time)

        # Track memory
        current_memory = get_memory_usage()
        peak_memory = max(peak_memory, current_memory["total_mb"])

        # evaluate at specified rounds
        if r == 1 or r % 10 == 0:
            for p, gp in zip(model.parameters(), global_params):
                p.data.copy_(gp)
            model.eval()
            logits = model(data.x, data.edge_index)
            preds = logits.argmax(dim=1)
            test_idx = data.test_mask.nonzero().view(-1)
            correct = (preds[test_idx] == data.y[test_idx]).sum().item()
            acc = 100.0 * correct / test_idx.size(0)

            # Calculate current communication cost (theoretical)
            current_comm_cost = calculate_communication_cost(model_size_mb, r, CLIENTS)

            print(
                f"[{ds_name} β={beta}] Round {r:3d} → "
                f"Test Acc: {acc:.2f}% | "
                f"Computation Time: {round_time:.2f}s | "
                f"Memory: {current_memory['total_mb']:.1f}MB | "
                f"Comm Cost: {current_comm_cost:.1f}MB"
            )

    elapsed = time.time() - t0

    # final evaluation
    for p, gp in zip(model.parameters(), global_params):
        p.data.copy_(gp)
    model.eval()
    logits = model(data.x, data.edge_index)
    preds = logits.argmax(dim=1)
    test_idx = data.test_mask.nonzero().view(-1)
    correct = (preds[test_idx] == data.y[test_idx]).sum().item()
    final_acc = 100.0 * correct / test_idx.size(0)

    # Calculate final metrics
    metrics.accuracy = final_acc
    metrics.total_time = elapsed
    metrics.computation_time = sum(computation_times)
    metrics.avg_time_per_round = np.mean(computation_times)
    metrics.communication_cost_mb = calculate_communication_cost(
        model_size_mb, ROUNDS, CLIENTS
    )
    metrics.peak_memory_mb = peak_memory

    return elapsed, final_acc, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_cluster",
        action="store_true",
        help="Enable RPC/DDP cluster mode for large OGBN datasets",
    )
    args = parser.parse_args()
    global use_cluster
    use_cluster = args.use_cluster

    # Enhanced CSV summary header
    print(
        "\nDS,IID,BS,Time[s],FinalAcc[%],CompTime[s],CommCost[MB],PeakMem[MB],AvgRoundTime[s],ModelSize[MB],TotalParams"
    )

    for ds in DATASETS:
        for beta in IID_BETAS:
            elapsed, acc, metrics = run_one(ds, beta, BATCH_SIZE, use_cluster)

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
