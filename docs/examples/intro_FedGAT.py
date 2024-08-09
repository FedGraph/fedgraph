"""
FedGraph Example
================

In this tutorial, you will learn the basic workflow of
FedGraph with a runnable example. This tutorial assumes that
you have basic familiarity with PyTorch and PyTorch Geometric (PyG).

(Time estimate: 15 minutes)
"""
import os
import sys

import attridict
import numpy as np
import ray
import torch
import yaml

from fedgraph.data_process import (
    FedAT_load_data_test,
)
from fedgraph.gnn_models import FedGATModel
from fedgraph.server_class import Server_GAT
from fedgraph.trainer_class import Trainer_GAT
from fedgraph.utils_gat import (
    calculate_statistics,
    get_in_comm_indexes,
    label_dirichlet_partition,
    print_client_statistics,
    print_mask_statistics,
)



# Seed initialization
np.random.seed(42)
torch.manual_seed(42)

# Directory and configuration setup
current_dir = os.getcwd()  # Use current working directory
sys.path.append(os.path.join(current_dir, "../fedgraph"))
sys.path.append(os.path.join(current_dir, "../../"))
config_file = os.path.join(current_dir, "configs/config_FedGAT.yaml")

with open(config_file, "r") as file:
    args = attridict.AttriDict(yaml.safe_load(file))


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
calculate_statistics(data)
print_mask_statistics(data)
row, col, edge_attr = adj.coo()
edge_index = torch.stack([row, col], dim=0)
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
print_client_statistics(split_node_indexes, idx_train, idx_val, idx_test)

# #######################################################################
# # Centralized GAT Test
# #######################################################################
row, col, edge_attr = adj.coo()
edge_index = torch.stack([row, col], dim=0)
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
    # args.num_hops:
    0,
    idx_train,
    idx_test,
    idx_val,
    one_hot_labels,
)
ray.init()


@ray.remote(
    num_gpus=0,
    num_cpus=8,
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


# Define Server
gat_model = FedGATModel(
    in_feat=normalized_features.shape[1],
    out_feat=one_hot_labels.shape[1],
    hidden_dim=args.hidden_dim,
    num_head=args.num_heads,
    max_deg=args.max_deg,
    attn_func=args.attn_func_parameter,
    domain=args.attn_func_domain,
).to(device=device)
# centralizedGATModel = CentralizedGATModel(
#     in_feat=normalized_features.shape[1],
#     out_feat=one_hot_labels.shape[1],
#     hidden_dim=args.hidden_dim,
#     num_head=args.num_heads,
#     max_deg=args.max_deg,
#     attn_func=args.attn_func_parameter,
#     domain=args.attn_func_domain,
# ).to(device="cpu")
server = Server_GAT(
    graph=data,
    model=gat_model,
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
server.pretrain_communication(communicate_node_indexes, data, device=args.device, args=args)
print("Pre-training communication completed!")


# server.ResetAll(gat_model, train_params=args)
server.TrainCoordinate()
ray.shutdown()
