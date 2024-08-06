"""
FedGraph Example
================

In this tutorial, you will learn the basic workflow of
FedGraph with a runnable example. This tutorial assumes that
you have basic familiarity with PyTorch and PyTorch Geometric (PyG).

(Time estimate: 15 minutes)
"""

import os
import time
import subprocess
import sys
from typing import Any

import attridict
import numpy as np
import ray
import torch
import yaml

from fedgraph.data_process import FedGAT_load_data, FedGAT_load_data_100, FedAT_load_data_test
from fedgraph.gnn_models import (
    FedGATModel,
)
from fedgraph.server_class import Server_GAT
from fedgraph.trainer_class import Trainer_GAT
from fedgraph.utils_gat import (
    get_in_comm_indexes,
    label_dirichlet_partition,
)
# check env version
# result = subprocess.run(["pip", "list"], stdout=subprocess.PIPE, text=True)
# torch_versions = [line for line in result.stdout.split(
#     "\n") if "torch" in line]
# for version in torch_versions:
#     print(version)
# print(sys.version)
# print(torch.__version__)
# print(torch.version.cuda)


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
    normalized_features,
    adj,
    labels,
    one_hot_labels,
    idx_train,
    idx_val,
    idx_test,
) = FedAT_load_data_test("cora")
print(data)

(
    data,
    normalized_features,
    adj,
    labels,
    one_hot_labels,
    idx_train,
    idx_val,
    idx_test,
) = FedAT_load_data_test(args.dataset)
print(data)
(
    data,
    normalized_features,
    adj,
    labels,
    one_hot_labels,
    idx_train,
    idx_val,
    idx_test,
) = FedAT_load_data_test("ogbn-products")
print(data)
(
    data,
    normalized_features,
    adj,
    labels,
    one_hot_labels,
    idx_train,
    idx_val,
    idx_test,
) = FedAT_load_data_test("siteseer")
print(data)

time.sleep(100)
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
device = torch.device("cpu" if True else "cpu")
print(f"device: {device}")
# print(f"normalized_features.shape[1]: {normalized_features.shape[1]}")


# tr_indexes = [[] for _ in range(args.n_trainer)]
# v_indexes = [[] for _ in range(args.n_trainer)]
# t_indexes = [[] for _ in range(args.n_trainer)]
# for i in range(args.n_trainer):
#     n = len(split_node_indexes[i])
#     np.random.shuffle(split_node_indexes[i].numpy())
#     tr_indexes[i] = split_node_indexes[i][: n - 2].tolist()
#     v_indexes[i] = split_node_indexes[i][n - 2: n - 1].tolist()
#     t_indexes[i] = split_node_indexes[i][n - 1:].tolist()

(
    communicate_node_indexes,
    in_com_train_node_indexes,
    in_com_test_node_indexes,
    in_com_val_node_indexes,
    edge_indexes_clients,
    in_com_labels,
) = get_in_comm_indexes(
    edge_index,
    split_node_indexes,
    args.n_trainer,
    # args.num_hops,
    0,
    idx_train,
    idx_test,
    idx_val,
    one_hot_labels,
)
ray.init()


@ray.remote(
    num_gpus=0,
    num_cpus=1,
    scheduling_strategy="SPREAD",
)
class Trainer(Trainer_GAT):
    def __init__(
        self,
        client_id,
        subgraph,
        node_indexes,
        train_indexes,
        val_indexes,
        test_indexes,
        labels,
        features_shape,
        args,
        device,
    ):
        super().__init__(  # type: ignore
            client_id=client_id,
            subgraph=subgraph,
            node_indexes=node_indexes,
            train_indexes=train_indexes,
            val_indexes=val_indexes,
            test_indexes=test_indexes,
            labels=labels,
            features_shape=features_shape,
            args=args,
            device=device,
        )
        # print(f"client_id: {client_id}")
        # print(f"subgraph: {subgraph}")
        # print(f"node_indexes: {node_indexes} (size: {len(node_indexes)})")
        # print(f"train_indexes: {train_indexes} (size: {len(train_indexes)})")
        # print(f"val_indexes: {val_indexes} (size: {len(val_indexes)})")
        # print(f"test_indexes: {test_indexes} (size: {len(test_indexes)})")
        # print(f"labels: {labels} (size: {len(labels)})")
        # print(f"features_shape: {features_shape}")
        # print(f"args: {args}")
        # print(f"device: {device}")
        # time.sleep(100)


clients = [
    Trainer.remote(
        # Trainer(
        client_id=client_id,
        subgraph=data.subgraph(communicate_node_indexes[client_id]),
        node_indexes=communicate_node_indexes[client_id],
        train_indexes=in_com_train_node_indexes[client_id],
        val_indexes=in_com_val_node_indexes[client_id],
        test_indexes=in_com_test_node_indexes[client_id],
        labels=in_com_labels[client_id],
        features_shape=normalized_features.shape[1],
        args=args,
        device=device,
    )
    for client_id in range(len(split_node_indexes))
]
# for client_id, node_indices in enumerate(split_node_indexes):
#     # print("current generating client:")
#     # print(f"clientId: {client_id}")
#     # print(f"node_indices: {node_indices}")
#     # split graph and then transfer the subgraph to each client based on split index
#     subgraph = data.subgraph(communicate_node_indexes[client_id])
#     # print(
#     #     f"communicate_node_indexes[client_id]: {communicate_node_indexes[client_id]}")
#     # print(
#     #     f"in_com_train_node_indexes[client_id]: {in_com_train_node_indexes[client_id]}"
#     # )
#     # print(
#     #     f"in_com_val_node_indexes[client_id]: {in_com_val_node_indexes[client_id]}")
#     # print(
#     #     f"in_com_test_node_indexes[client_id]: {in_com_test_node_indexes[client_id]}")
#     # print(
#     #     f"len(communicate_node_indexes[i]): {len(communicate_node_indexes[i])}")
#     # print(one_hot_labels.size())

#     # client initialization
#     client = Trainer_GAT(
#         client_id=client_id,
#         subgraph=subgraph,
#         node_indexes=communicate_node_indexes[client_id],
#         train_indexes=in_com_train_node_indexes[client_id],
#         val_indexes=in_com_val_node_indexes[client_id],
#         test_indexes=in_com_test_node_indexes[client_id],
#         labels=in_com_labels[i],
#         features_shape=normalized_features.shape[1],
#         args=args,
#         device=device,
#     )

#     clients.append(client)
edge_index = edge_index.to(device)
# Define Server
GATModel = FedGATModel(
    in_feat=normalized_features.shape[1],
    out_feat=one_hot_labels.shape[1],
    hidden_dim=args.hidden_dim,
    num_head=args.num_heads,
    max_deg=args.max_deg,
    attn_func=args.attn_func_parameter,
    domain=args.attn_func_domain,
).to(device=device)
server = Server_GAT(
    graph=data,
    model=GATModel,
    feats=normalized_features,
    labels=one_hot_labels,
    feature_dim=normalized_features.shape[1],
    class_num=one_hot_labels.shape[1],
    device=device,
    trainers=clients,
    args=args,
)

# Pre-training communication
print("Pre-training communication initiated!")
server.pretrain_communication(
    communicate_node_indexes, data, device=args.device)
# for client_id, communicate_node_index in enumerate(communicate_node_indexes):
#     # print(f"currentClientID:{client_id}")
#     # print(f"node_indexes size: {len(communicate_node_index)}")
#     ret_info = server.pretrain_communication(
#         client_id, communicate_node_index, data, device=args.device
#     )
#     # print("printing ret_info and subgraph size:")
#     # print(len(ret_info))
#     # print(clients[client_id].graph.size())
#     # the subgraph size is 1606 but the ret_info size is 1578
#     # IndexError: Encountered an index error. Please ensure that all indices in 'edge_index' point to valid indices in the interval [0, 1578] (got interval [0, 1606])
#     clients[client_id].setNodeMats(ret_info)
#     # print(f"client{client_id} have ret_info\n {ret_info}")
print("Pre-training communication completed!")

# # Federated Training
# print("Commenced training!")

# [client.from_server.remote(server.GATModelParams, server.Duals) for client in clients]
# [client.train_model.remote() for client in clients]

# print("Training initiated!")

# for t in range(server.train_rounds):
#     [client.train_iterate.remote() for client in clients]

#     if (t + 1) % server.num_local_iters == 0:
#         server.TrainingUpdate()

#     [
#         client.from_server.remote(server.GATModelParams, server.Duals)
#         for client in clients
#     ]

# print("Training completed!")
# print("Testing now!")

# [client.model_test.remote() for client in clients]

# print("Complete!")

# # Summarize Experiment Results
# # The server collects the local test loss and accuracy from all clients
# # then calculate the overall test loss and accuracy.

# server.ResetAll()

server.TrainCoordinate(GATModel)
ray.shutdown()
