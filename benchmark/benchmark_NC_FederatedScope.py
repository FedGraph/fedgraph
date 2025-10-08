#!/usr/bin/env python3
import logging
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import argparse
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.data import DummyDataTranslator
from federatedscope.core.fed_runner import FedRunner
from federatedscope.register import register_data, register_model
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

from fedgraph.utils_nc import label_dirichlet_partition

DATASETS = ["cora"]

IID_BETAS = [100.0]
CLIENT_NUM = 5
TOTAL_ROUNDS = 100
LOCAL_STEPS = 3
LEARNING_RATE = 0.5
HIDDEN_DIM = 16
DROPOUT_RATE = 0.5
CPUS_PER_TRAINER = 1
STANDALONE_PROCESSES = 1

PLANETOID_NAMES = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}


def peak_memory_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (usage / 1024**2) if usage > 1024**2 else (usage / 1024)


def calculate_communication_cost(model_size_mb, rounds, clients):
    cost_per_round = model_size_mb * clients * 2
    return cost_per_round * rounds


class TwoLayerGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, HIDDEN_DIM)
        self.conv2 = GCNConv(HIDDEN_DIM, out_channels)
        self.dropout = DROPOUT_RATE

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


def make_data_loader(name):
    def load_data(config, client_cfgs=None):
        ds = Planetoid(root="data/", name=PLANETOID_NAMES[name])
        full = ds[0]
        num_classes = int(full.y.max().item()) + 1
        
        # 与data_process.py完全一致：在全部节点上做Dirichlet分割
        split_idxs = label_dirichlet_partition(
            full.y,  # 使用全部节点，不只是训练节点
            len(full.y),  # 全部节点数
            num_classes,
            config.federate.client_num,
            config.iid_beta,
            config.distribution_type,
        )
        
        parts = []
        for idxs in split_idxs:
            client_nodes = torch.tensor(idxs)
            
            # 为每个客户端创建mask，但保持原有的train/val/test划分逻辑
            train_mask = torch.zeros(full.num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(full.num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(full.num_nodes, dtype=torch.bool)
            
            # 在客户端节点中，保持原有数据集的train/val/test划分
            for node in client_nodes:
                if full.train_mask[node]:
                    train_mask[node] = True
                elif full.val_mask[node]:
                    val_mask[node] = True
                elif full.test_mask[node]:
                    test_mask[node] = True
            
            parts.append(
                Data(
                    x=full.x,
                    edge_index=full.edge_index,
                    y=full.y,
                    train_mask=train_mask,  # 保持原有train划分
                    val_mask=val_mask,      # 保持原有val划分
                    test_mask=test_mask,    # 保持原有test划分
                )
            )
        
        data_dict = {
            i + 1: {
                "data": parts[i],
                "train": [parts[i]],
                "val": [parts[i]],
                "test": [parts[i]],
            }
            for i in range(len(parts))
        }
        data_dict[0] = {"data": full, "train": [full], "val": [full], "test": [full]}
        return DummyDataTranslator(config)(data_dict), config

    return load_data


def make_model_builder(name, num_classes):
    key = f"gnn_{name}"

    def build(cfg_model, input_shape):
        if cfg_model.type != key:
            return None
        in_feats = input_shape[0][-1]
        return TwoLayerGCN(in_feats, num_classes)

    return build, key


register_data("cora", make_data_loader("cora"))
builder, mkey = make_model_builder("cora", 7)
register_model(mkey, builder)



def run_fedscope_experiment(ds, beta):
    cfg = global_cfg.clone()
    cfg.defrost()
    cfg.use_gpu = False
    cfg.device = -1
    cfg.seed = 42
    cfg.federate.mode = "standalone"
    cfg.federate.client_num = CLIENT_NUM
    cfg.federate.total_round_num = TOTAL_ROUNDS
    cfg.federate.make_global_eval = False
    cfg.federate.process_num = CLIENT_NUM  
    cfg.federate.num_cpus_per_trainer = CPUS_PER_TRAINER
    cfg.data.root = "data/"
    cfg.data.type = ds
    cfg.data.splitter = "dirichlet"
    cfg.iid_beta = beta
    cfg.distribution_type = "average"
    cfg.dataloader.type = "pyg"
    cfg.dataloader.batch_size = 1
    cfg.model.type = f"gnn_{ds}"
    cfg.model.hidden = HIDDEN_DIM
    cfg.model.dropout = DROPOUT_RATE
    cfg.model.layer = 2
    cfg.model.out_channels = 7
    cfg.criterion.type = "CrossEntropyLoss"
    cfg.trainer.type = "nodefullbatch_trainer"
    cfg.train.local_update_steps = LOCAL_STEPS
    cfg.train.optimizer.lr = LEARNING_RATE
    cfg.train.optimizer.weight_decay = 0.0
    cfg.train.optimizer.type = "SGD"
    cfg.eval.freq = 1
    cfg.eval.metrics = ["acc"]
    cfg.freeze()
    data_fs, _ = get_data(config=cfg.clone())
    full = data_fs[0]["data"]
    t0 = time.time()
    runner = FedRunner(data=data_fs, config=cfg)
    res = runner.run()
    dur = time.time() - t0
    mem = peak_memory_mb()
    
    # 获取FederatedScope结果
    
    # 从FederatedScope的结果中获取准确率
    # 使用加权平均以与FedGraph保持一致
    acc = res.get("client_summarized_weighted_avg", {}).get("test_acc", 0.0) if res else 0.0
    
    acc_pct = acc * 100 if acc <= 1.0 else acc
    model = runner.server.model if runner.server else None
    if model is not None:
        tot_params = sum(p.numel() for p in model.parameters())
        msz = tot_params * 4 / 1024**2
    else:
        tot_params = 0
        msz = 0.0
    comm = calculate_communication_cost(msz, TOTAL_ROUNDS, CLIENT_NUM)
    return {
        "accuracy": acc_pct,
        "total_time": dur,
        "computation_time": dur,
        "communication_cost_mb": comm,
        "peak_memory_mb": mem,
        "avg_time_per_round": dur / TOTAL_ROUNDS,
        "model_size_mb": msz,
        "total_params": tot_params,
        "nodes": full.num_nodes,
        "edges": full.edge_index.size(1),
    }




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cluster", action="store_true")
    args = parser.parse_args()

    print(
        "\nDS,IID,BS,Time[s],FinalAcc[%],CompTime[s],CommCost[MB],PeakMem[MB],AvgRoundTime[s],ModelSize[MB],TotalParams"
    )
    for ds in DATASETS:
        for beta in IID_BETAS:
            try:
                print(f"Running {ds} with β={beta}")
                if ds == "cora":
                    metrics = run_fedscope_experiment(ds, beta)
                print(
                    f"Dataset: {metrics['nodes']:,} nodes, {metrics['edges']:,} edges"
                )
                print(
                    f"{ds},{beta},-1,"
                    f"{metrics['total_time']:.1f},"
                    f"{metrics['accuracy']:.2f},"
                    f"{metrics['computation_time']:.1f},"
                    f"{metrics['communication_cost_mb']:.1f},"
                    f"{metrics['peak_memory_mb']:.1f},"
                    f"{metrics['avg_time_per_round']:.3f},"
                    f"{metrics['model_size_mb']:.3f},"
                    f"{metrics['total_params']}"
                )
            except Exception as e:
                print(f"Error running {ds} with β={beta}: {e}")
                print(f"{ds},{beta},-1,0.0,0.00,0.0,0.0,0.0,0.000,0.000,0")


if __name__ == "__main__":
    main()
