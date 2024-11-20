"""
FedGraph Example
================

In this tutorial, you will learn the basic workflow of
FedGraph with a runnable example. This tutorial assumes that
you have basic familiarity with PyTorch and PyTorch Geometric (PyG).

(Time estimate: 15 minutes)
"""
import os
import random
import subprocess
import sys
import time
from typing import Any

import attridict
import numpy as np
import ray
import torch
import torch.nn as nn
import torch_geometric
import yaml
from torch.optim import SGD, Adam

from fedgraph.data_process import (
    FedAT_load_data_test,
    FedGAT_load_data,
    FedGAT_load_data_100,
)
from fedgraph.gnn_models import CentralizedGATModel, FedGATModel
from fedgraph.monitor_class import Monitor
from fedgraph.server_class import Server_GAT
from fedgraph.trainer_class import Trainer_GAT
from fedgraph.utils_gat import (
    calculate_statistics,
    get_in_comm_indexes,
    label_dirichlet_partition,
    print_client_statistics,
    print_mask_statistics,
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
ray.init()

print(args)


def run_fedgraph():
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
    # calculate_statistics(data)
    # print_mask_statistics(data)
    row, col, edge_attr = adj.coo()
    edge_index = torch.stack([row, col], dim=0)
    node_mats = None
    print("printing final args")
    print(args)
    # if True:
    #     #######################################################################
    #     # Centralized GAT Test
    #     #######################################################################
    #     m = "Centralized GAT"
    #     gat = CentralizedGATModel(
    #         in_feat=normalized_features.shape[1],
    #         out_feat=one_hot_labels.shape[1],
    #         hidden_dim=args.hidden_dim,
    #         num_head=args.num_heads,
    #         max_deg=args.max_deg,
    #         attn_func=args.attn_func_parameter,
    #         domain=args.attn_func_domain,
    #         num_layers=args.num_layers,
    #     ).to(device="cpu")
    #     for p in gat.parameters():
    #         p.requires_grad = True
    #     optimizer = Adam(
    #         gat.parameters(),
    #         lr=args.model_lr,
    #         weight_decay=args.model_regularisation,
    #     )
    #     # optimizer = SGD(gat.parameters(), lr=args.model_lr,
    #     #                 weight_decay=args.model_regularisation)
    #
    #     def LossFunc(y_pred, y_true, model, args):
    #         # criterion = nn.KLDivLoss(
    #         #     reduction="batchmean", log_target=False)
    #         if args.dataset == "ogbn-arxiv":
    #             criterion = nn.KLDivLoss(reduction="batchmean", log_target=False)
    #         else:
    #             criterion = nn.CrossEntropyLoss()
    #         v = criterion(y_pred, y_true)
    #         # for p in model.parameters():
    #         #     v += 0.5 * 5e-4 * torch.sum(p ** 2)
    #
    #         return v
    #
    #     # print("Starting training!")
    #     epoch = 0
    #     num_epochs = args.train_rounds
    #
    #     train_mask = idx_train
    #     validate_mask = idx_val
    #     test_mask = idx_test
    #
    #     # for p in gat.parameters():
    #     #     print(p.requires_grad)
    #     print("Starting training!")
    #     for ep in range(num_epochs):
    #         if args.batch_size:
    #             train_mask = torch.tensor(
    #                 random.sample(list(idx_train), args.batch_size)
    #             )
    #         gat.train()
    #         optimizer.zero_grad()
    #         y_pred = gat(data)
    #         if args.dataset == "ogbn-arxiv":
    #             t_loss = LossFunc(
    #                 y_pred[train_mask].log(), one_hot_labels[train_mask], gat, args
    #             )
    #         else:
    #             t_loss = LossFunc(
    #                 y_pred[train_mask], one_hot_labels[train_mask], gat, args
    #             )
    #
    #         t_loss.backward()
    #         optimizer.step()
    #
    #         with torch.no_grad():
    #             gat.eval()
    #             if args.dataset == "ogbn-arxiv":
    #                 v_loss = LossFunc(
    #                     y_pred[validate_mask].log(),
    #                     one_hot_labels[validate_mask],
    #                     gat,
    #                     args,
    #                 )
    #             else:
    #                 v_loss = LossFunc(
    #                     y_pred[validate_mask], one_hot_labels[validate_mask], gat, args
    #                 )
    #
    #             pred_labels = torch.argmax(y_pred, dim=1)
    #             true_labels = torch.argmax(one_hot_labels, dim=1)
    #
    #             t_acc = torch.sum(
    #                 pred_labels[train_mask] == true_labels[train_mask]
    #             ).item() / len(train_mask)
    #             v_acc = torch.sum(
    #                 pred_labels[validate_mask] == true_labels[validate_mask]
    #             ).item() / len(validate_mask)
    #
    #             # print(
    #             #     f"Client 0: Epoch {epoch}: Train loss: {t_loss.item():.4f}, Train acc: {t_acc*100:.2f}%, "
    #             #     f"Val loss: {v_loss.item():.4f}, Val acc {v_acc*100:.2f}%"
    #             # )
    #             gat.eval()
    #
    #             with torch.no_grad():
    #                 test_loss = LossFunc(
    #                     y_pred[test_mask], one_hot_labels[test_mask], gat, args
    #                 )
    #
    #                 pred_labels = torch.argmax(y_pred, dim=1)
    #                 true_labels = torch.argmax(one_hot_labels, dim=1)
    #
    #                 test_acc = (
    #                     torch.sum(
    #                         pred_labels[test_mask] == true_labels[test_mask]
    #                     ).item()
    #                     / len(test_mask)
    #                     * 100
    #                 )
    #                 print(
    #                     f" Log// {m}, {args.dataset}, {1}, {ep}, {test_acc}, {0}, {args.iid_beta} //end"
    #                 )
    #
    #         epoch += 1

    def run(node_mats):
        @ray.remote(
            num_gpus=0,
            num_cpus=0.1,
            scheduling_strategy="DEFAULT",
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
                type,
                batch_size,
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
                    type=type,
                    batch_size=batch_size,
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

        # print(f"Epoch {ep} completed!")
        split_node_indexes = label_dirichlet_partition(
            labels,
            len(labels),
            labels.max().item() + 1,
            args.n_trainer,
            beta=args.iid_beta,
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
            induce_node_indexes,
            origin_train_indexes,
            origin_val_indexes,
            origin_test_indexes,
            origin_labels,
        ) = get_in_comm_indexes(
            edge_index,
            split_node_indexes,
            args.n_trainer,
            args.num_layers,
            idx_train,
            idx_test,
            idx_val,
            one_hot_labels,
        )
        if False:
            args.method = "DistributedGAT"
            print(
                f"Running experiment with: Dataset={args.dataset},"
                f" Number of Trainers={n_trainer}, Distribution Type={args.method},"
                f" IID Beta={args.iid_beta}, Number of Hops={max_deg}, Batch Size={-1}"
            )
            #######################################################################
            # Distributed GAT Test
            #######################################################################
            gat = CentralizedGATModel(
                in_feat=normalized_features.shape[1],
                out_feat=one_hot_labels.shape[1],
                hidden_dim=args.hidden_dim,
                num_head=args.num_heads,
                max_deg=args.max_deg,
                attn_func=args.attn_func_parameter,
                domain=args.attn_func_domain,
                num_layers=args.num_layers,
            ).to(device="cpu")
            clients = [
                Trainer.remote(
                    # Trainer(
                    client_id=client_id,
                    subgraph=data.subgraph(communicate_node_indexes[client_id]),
                    node_indexes=communicate_node_indexes[client_id],
                    train_indexes=origin_train_indexes[client_id],
                    val_indexes=origin_val_indexes[client_id],
                    test_indexes=origin_test_indexes[client_id],
                    labels=origin_labels[client_id],
                    features_shape=normalized_features.shape[1],
                    args=args,
                    device=device,
                    type=args.method,
                    batch_size=args.batch_size,
                )
                for client_id in range(len(split_node_indexes))
            ]
            server = Server_GAT(
                graph=data,
                model=gat,
                feats=normalized_features,
                labels=one_hot_labels,
                feature_dim=normalized_features.shape[1],
                class_num=one_hot_labels.shape[1],
                device=device,
                trainers=clients,
                type=args.method,
                args=args,
            )

            # server.ResetAll(gat_model, train_params=args)
            server.TrainCoordinate()
        if True:
            args.method = "FedGAT"
            print(
                f"Running experiment with: Dataset={args.dataset},"
                f" Number of Trainers={n_trainer}, Distribution Type={args.method},"
                f" IID Beta={args.iid_beta}, Number of Hops={max_deg}, Batch Size={-1}"
            )
            #######################################################################
            # FedGAT Test
            #######################################################################
            clients = [
                Trainer.remote(
                    # Trainer(
                    client_id=client_id,
                    subgraph=data.subgraph(induce_node_indexes[client_id]),
                    node_indexes=communicate_node_indexes[client_id],
                    train_indexes=in_com_train_node_indexes[client_id],
                    val_indexes=in_com_val_node_indexes[client_id],
                    test_indexes=in_com_test_node_indexes[client_id],
                    labels=in_com_labels[client_id],
                    features_shape=normalized_features.shape[1],
                    args=args,
                    device=device,
                    type=args.method,
                    batch_size=args.batch_size,
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
                type=args.method,
            )

            # Pre-training communication
            print("Pre-training communication initiated!")
            monitor = Monitor()
            monitor.pretrain_time_start()
            if node_mats == None:
                node_mats = server.pretrain_communication(
                    induce_node_indexes, data, device=args.device, args=args
                )
            else:
                server.distribute_mats(induce_node_indexes, node_mats)
            print("Pre-training communication completed!")
            monitor.pretrain_time_end(1)
            monitor.train_time_start()
            # server.ResetAll(gat_model, train_params=args)
            server.TrainCoordinate()
            monitor.train_time_end(1)
        return node_mats

    # experiment start here
    for n_trainer in [10]:
        args.n_trainer = n_trainer
        for iid in [10000.0,1.0]:
            args.iid_beta = iid
            for max_deg in range(4,25):
                node_mats = None
                args.max_deg = max_deg
                for _ in range(3):
                    node_mats = run(node_mats)


# for d in ["ogbn-arxiv"]:
#     args.dataset = d
#     args.hidden_dim = 256
#     args.limit_node_degree = 30
#     args.batch_size = 2048
#     args.model_lr = 0.005
#     args.num_heads = 3
#     args.num_layers = 2
#     args.train_rounds = 100
#     args.global_rounds = 100
#     args.vecgen = True
#     run_fedgraph()
for d in ["cora"]:
    args.dataset = d
    args.vecgen = True
    run_fedgraph()
for d in ["citeseer"]:
    args.dataset = d
    args.vecgen = True
    run_fedgraph()
# for d in ["pubmed"]:
#     args.dataset = d
#     args.hidden_dim = 8
#     args.train_rounds = 30
#     args.global_rounds = 30
#     args.model_lr = 0.04
#     args.model_regularisation = 3.0e-3
#     run_fedgraph()

# time.sleep(100000)

ray.shutdown()
