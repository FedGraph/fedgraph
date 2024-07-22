"""
FedGraph Example
================

In this tutorial, you will learn the basic workflow of
FedGraph with a runnable example. This tutorial assumes that
you have basic familiarity with PyTorch and PyTorch Geometric (PyG).

(Time estimate: 15 minutes)
"""

import argparse
import copy
import os
import random
import sys
from collections import deque
from typing import Any

import attridict
import numpy as np
import ray
import torch
import yaml

from fedgraph.data_process import FedGAT_load_data, FedGAT_load_data_100
from fedgraph.gnn_models import (
    GCN,
    GNN_LP,
    AggreGCN,
    FedGATModel,
    GCN_arxiv,
    SAGE_products,
)
from fedgraph.server_class import Server_GAT
from fedgraph.trainer_class import Trainer_GAT
from fedgraph.utils_gat import (
    CreateNodeSplit,
    get_in_comm_indexes,
    label_dirichlet_partition,
)

# Seed initialization
np.random.seed(42)
torch.manual_seed(42)

# Directory and configuration setup
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../fedgraph"))
sys.path.append(os.path.join(current_dir, "../../"))
config_file = os.path.join(current_dir, "configs/config_FedGAT.yaml")

with open(config_file, "r") as file:
    args = attridict.AttriDict(yaml.safe_load(file))

# Data Loading
(
    data,
    features,
    normalized_features,
    adj,
    labels,
    one_hot_labels,
    idx_train,
    idx_val,
    idx_test,
) = FedGAT_load_data_100(args.dataset)
row, col, edge_attr = adj.coo()
edge_index = torch.stack([row, col], dim=0)
# Split Graph for Federated Learning
# client_nodes = CreateNodeSplit(data, args.n_trainer)
# print(client_nodes)
split_node_indexes = label_dirichlet_partition(
    labels, len(labels), labels.max().item() + 1, args.n_trainer, beta=args.iid_beta
)

for i in range(args.n_trainer):
    split_node_indexes[i] = np.array(split_node_indexes[i])
    split_node_indexes[i].sort()
    split_node_indexes[i] = torch.tensor(split_node_indexes[i])


# Device setup
device = torch.device("cuda" if args.gpu else "cpu")
initial_model = FedGATModel(
    normalized_features.shape[1],
    one_hot_labels.shape[1],
    args.hidden_dim,
    args.num_heads,
    args.max_deg,
    args.attn_func_parameter,
    args.attn_func_domain,
    device,
)

tr_indexes = [[] for _ in range(args.n_trainer)]
v_indexes = [[] for _ in range(args.n_trainer)]
t_indexes = [[] for _ in range(args.n_trainer)]
for i in range(args.n_trainer):
    n = len(split_node_indexes[i])
    np.random.shuffle(split_node_indexes[i].numpy())  # 随机打乱索引
    tr_indexes[i] = split_node_indexes[i][: n - 2].tolist()  # 训练集：前 n-2 个
    v_indexes[i] = split_node_indexes[i][n - 2 : n - 1].tolist()  # 验证集：倒数第 2 个
    t_indexes[i] = split_node_indexes[i][n - 1 :].tolist()  # 测试集：最后 1 个

# (
#     communicate_node_indexes,
#     in_com_train_node_indexes,
#     in_com_test_node_indexes,
#     edge_indexes_clients,
# ) = get_in_comm_indexes(
#     edge_index,
#     split_node_indexes,
#     args.n_trainer,
#     args.num_hops,
#     idx_train,
#     idx_test,
# )


# Initialize Clients
clients = []
for client_id, node_indices in enumerate(split_node_indexes):
    print("current generating client:")
    print(f"clientId: {client_id}")
    print(f"node_indices: {node_indices}")
    # split graph and then transfer the subgraph to each client based on split index
    subgraph = data.subgraph(node_indices)
    # client initialization
    client = Trainer_GAT(
        client_id=client_id,
        subgraph=subgraph,
        node_indices=node_indices,
        train_indexes=tr_indexes[client_id],
        val_indexes=v_indexes[client_id],
        test_indexes=t_indexes[client_id],
        labels=one_hot_labels,
        model=initial_model,
        train_rounds=args.global_rounds,
        num_local_iters=args.local_step,
        dual_weight=args.dual_weight,
        aug_lagrange_rho=args.aug_lagrange_rho,
        dual_lr=args.dual_lr,
        model_lr=args.learning_rate,
        model_regularisation=args.model_regularisation,
        device=device,
    )

    clients.append(client)

# Define Server
server = Server_GAT(
    graph=data,
    feats=features,
    labels=one_hot_labels,
    sample_probab=args.sample_probab,
    train_rounds=args.train_rounds,
    num_local_iters=args.num_local_iters,
    dual_weight=args.dual_weight,
    aug_lagrange_rho=args.aug_lagrange_rho,
    dual_lr=args.dual_lr,
    model_lr=args.learning_rate,
    model_regularisation=args.model_regularisation,
    feature_dim=features.shape[1],
    class_num=one_hot_labels.shape[1],
    hidden_dim=args.hidden_dim,
    num_head=args.num_heads,
    max_deg=args.max_deg,
    attn_func=args.attn_func_parameter,
    attn_func_domain=args.attn_func_domain,
    device=device,
    trainers=clients,
)

# Pre-training communication
print("Pre-training communication initiated!")

for client_id, node_indices in enumerate(split_node_indexes):
    ret_info = server.pretrain_communication(
        client_id, node_indices, data, device=args.device
    )
    clients[client_id].node_mats = ret_info
    # print(f"client{client_id} have ret_info\n {ret_info}")
print("Pre-training communication completed!")

# Federated Training
print("Commenced training!")

for client in clients:
    client.from_server(server.GATModelParams, server.Duals)
    client.train_model()

print("Training initiated!")

for t in range(server.train_rounds):
    for client in clients:
        client.train_iterate()

    if (t + 1) % server.num_local_iters == 0:
        server.TrainingUpdate()

    for client in clients:
        client.from_server(server.GATModelParams, server.Duals)

print("Training completed!")
print("Testing now!")

for client in clients:
    client.model_test()

print("Complete!")

# Summarize Experiment Results
# The server collects the local test loss and accuracy from all clients
# then calculate the overall test loss and accuracy.
