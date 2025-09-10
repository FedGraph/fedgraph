#!/usr/bin/env python3
"""
benchmark_NC_FederatedScope_summary.py

Outputs only:
  Running <DS> with β=<IID>
  Dataset: <#nodes> nodes, <#edges> edges
  [<DS> β=<IID>] Round <round> → Test Acc: <acc>% | Computation Time: <t>s | Memory: <mem>MB | Comm Cost: <comm>MB
"""
import logging
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import resource
import time

import torch
import torch.nn.functional as F
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.fed_runner import FedRunner
from federatedscope.register import register_model
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# Experiment settings
data_sets = ["cora", "citeseer", "pubmed"]
iid_betas = [10000.0, 100.0, 10.0]
clients = 10
total_rounds = 200
local_steps = 1
lr = 0.1
hidden_dim = 64
dropout_rate = 0.0  # match FedGraph no dropout
cpus_per_trainer = 0.6
processes = 1  # standalone CPU only

# Utility to measure peak memory


def peak_memory_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On macOS it's bytes, on Linux it's KB
    if usage > 1024**2:
        return usage / (1024**2)
    return usage / 1024


# Simple 2-layer GCN model class
class TwoLayerGCN(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_ch)
        self.dropout = dropout_rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


# Factory to create and register the model builder for each dataset


def make_model_builder(name, out_channels):
    key = f"gnn_{name}"

    def builder(cfg_model, input_shape):
        if cfg_model.type != key:
            return None
        in_ch = input_shape[0][-1]
        return TwoLayerGCN(in_ch, out_channels)

    return builder, key


# Main loop: dataset × beta
for ds in data_sets:
    # Register model builder
    out_channels = {"cora": 7, "citeseer": 6, "pubmed": 3}[ds]
    builder, model_key = make_model_builder(ds, out_channels)
    register_model(model_key, builder)

    for beta in iid_betas:
        # Print run header to log
        graph = Planetoid(root="data/", name=ds)[0]
        print(f"Running {ds} with β={beta}")
        print(f"Dataset: {graph.num_nodes:,} nodes, {graph.edge_index.size(1):,} edges")

        # Build federated configuration
        cfg = global_cfg.clone()
        cfg.defrost()
        cfg.use_gpu = False
        cfg.device = -1
        cfg.seed = 42

        cfg.federate.mode = "standalone"
        cfg.federate.client_num = clients
        cfg.federate.total_round_num = total_rounds
        cfg.federate.make_global_eval = True
        cfg.federate.process_num = processes
        cfg.federate.num_cpus_per_trainer = cpus_per_trainer

        cfg.data.root = "data/"
        cfg.data.type = ds
        # Use random split to approximate `average` FedGraph distribution
        cfg.data.splitter = "random"

        cfg.dataloader.type = "pyg"
        cfg.dataloader.batch_size = 1

        cfg.model.type = model_key
        cfg.model.hidden = hidden_dim
        cfg.model.dropout = dropout_rate
        cfg.model.layer = 2
        cfg.model.out_channels = out_channels

        cfg.criterion.type = "CrossEntropyLoss"

        cfg.trainer.type = "nodefullbatch_trainer"
        cfg.train.local_update_steps = local_steps
        cfg.train.optimizer.lr = lr
        cfg.train.optimizer.weight_decay = 0.0

        cfg.eval.freq = 1
        cfg.eval.metrics = ["acc"]
        cfg.freeze()

        # Load data and run training
        data, _ = get_data(config=cfg.clone())
        start = time.time()
        runner = FedRunner(data=data, config=cfg)
        results = runner.run()
        elapsed = time.time() - start
        mem_peak = peak_memory_mb()

        # Extract final test accuracy
        if "server_global_eval" in results:
            evals = results["server_global_eval"]
            acc = evals.get("test_acc", evals.get("acc", 0.0))
        else:
            acc = results.get("test_acc", results.get("acc", 0.0))
        acc_pct = acc * 100 if acc <= 1.0 else acc

        # Estimate communication cost
        model = runner.server.model
        total_params = sum(p.numel() for p in model.parameters())
        size_mb = total_params * 4 / (1024**2)
        comm_cost = size_mb * 2 * clients * total_rounds

        # Print summary line
        print(
            f"[{ds} β={beta}] Round {total_rounds} → "
            f"Test Acc: {acc_pct:.2f}% | "
            f"Computation Time: {elapsed:.2f}s | "
            f"Memory: {mem_peak:.1f}MB | "
            f"Comm Cost: {comm_cost:.1f}MB"
        )
        print()
