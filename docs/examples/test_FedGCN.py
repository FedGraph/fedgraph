"""
FedGraph Example
================

In this tutorial, you will learn the basic workflow of
FedGraph with a runnable example. This tutorial assumes that
you have basic familiarity with PyTorch and PyTorch Geometric (PyG).

(Time estimate: 15 minutes)
"""
from datetime import datetime
import argparse
import time
from typing import Any

import numpy as np
import ray
import torch
import torch_geometric
from fedgraph.monitor_class import Monitor
from fedgraph.server_class import Server
from fedgraph.trainer_class import Trainer_General
from fedgraph.utils_nc import (
    get_1hop_feature_sum,
    get_in_comm_indexes,
    label_dirichlet_partition,
)

ray.init()


np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default="cora", type=str)

parser.add_argument("-f", "--fedtype", default="fedgcn", type=str)

parser.add_argument("-c", "--global_rounds", default=100, type=int)
parser.add_argument("-i", "--local_step", default=3, type=int)
parser.add_argument("-lr", "--learning_rate", default=0.1, type=float)

parser.add_argument("-n", "--n_trainer", default=3, type=int)
parser.add_argument("-nl", "--num_layers", default=2, type=int)
parser.add_argument("-nhop", "--num_hops", default=2, type=int)
parser.add_argument("-g", "--gpu", action="store_true")  # if -g, use gpu
parser.add_argument("-iid_b", "--iid_beta", default=10000, type=float)

parser.add_argument("-l", "--logdir", default="./runs", type=str)

args = parser.parse_args()


def FedGCN_load_data(dataset_str: str) -> tuple:
    if dataset_str in [
        "ogbn-arxiv",
        "ogbn-products",
        "ogbn-mag",
        "ogbn-papers100M",
    ]:  # 'ogbn-mag' is heteregeneous
        from ogb.nodeproppred import PygNodePropPredDataset

        # Download and process data at './dataset/.'

        dataset = PygNodePropPredDataset(
            name=dataset_str, transform=torch_geometric.transforms.ToSparseTensor()
        )

        split_idx = dataset.get_idx_split()
        idx_train, idx_val, idx_test = (
            split_idx["train"],
            split_idx["valid"],
            split_idx["test"],
        )
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        data = dataset[0]

        features = data.x
        labels = data.y.reshape(-1)
        if dataset_str == "ogbn-arxiv":
            adj = data.adj_t.to_symmetric()
        else:
            adj = data.adj_t



    return features.float(), adj, labels, idx_train, idx_val, idx_test

def run():
    #######################################################################
    # Data Loading
    # ------------
    # FedGraph use ``torch_geometric.data.Data`` to handle the data. Here, we
    # use Cora, a PyG built-in dataset, as an example. To load your own
    # dataset into FedGraph, you can simply load your data
    # into "features, adj, labels, idx_train, idx_val, idx_test".
    # Or you can create dataset in PyG. Please refer to `creating your own datasets
    # tutorial <https://pytorch-geometric.readthedocs.io/en/latest/notes
    # /create_dataset.html>`__ in PyG.

    features, adj, labels, idx_train, idx_val, idx_test = FedGCN_load_data(
        args.dataset)
    class_num = labels.max().item() + 1

    if args.dataset in ["simulate", "cora", "citeseer", "pubmed", "reddit"]:
        args_hidden = 16
    else:
        args_hidden = 256

    row, col, edge_attr = adj.coo()
    edge_index = torch.stack([row, col], dim=0)

    num_cpus_per_client = 4
    # specifying a target GPU
    args.gpu = False  # Test
    print(f"gpu usage: {args.gpu}")
    if args.gpu:
        device = torch.device("cuda")
        edge_index = edge_index.to("cuda")
        num_gpus_per_client = 1
    else:
        device = torch.device("cpu")
        num_gpus_per_client = 0

    #######################################################################
    # Split Graph for Federated Learning
    # ----------------------------------
    # FedGraph currents has two partition methods: label_dirichlet_partition
    # and community_partition_non_iid to split the large graph into multiple trainers

    split_node_indexes = label_dirichlet_partition(
        labels, len(labels), class_num, args.n_trainer, beta=args.iid_beta
    )

    for i in range(args.n_trainer):
        split_node_indexes[i] = np.array(split_node_indexes[i])
        split_node_indexes[i].sort()
        split_node_indexes[i] = torch.tensor(split_node_indexes[i])

    (
        communicate_node_indexes,
        in_com_train_node_indexes,
        in_com_test_node_indexes,
        edge_indexes_clients,
    ) = get_in_comm_indexes(
        edge_index,
        split_node_indexes,
        args.n_trainer,
        args.num_hops,
        idx_train,
        idx_test,
    )

    #######################################################################
    # Define and Send Data to Trainers
    # --------------------------------
    # FedGraph first determines the resources for each trainer, then send
    # the data to each remote trainer.

    @ray.remote(
        num_gpus=num_gpus_per_client,
        num_cpus=num_cpus_per_client,
        scheduling_strategy="SPREAD",
    )
    class Trainer(Trainer_General):
        def __init__(self, *args: Any, **kwds: Any):
            super().__init__(*args, **kwds)

    trainers = [
        Trainer.remote(  # type: ignore
            rank=i,
            local_node_index=split_node_indexes[i],
            communicate_node_index=communicate_node_indexes[i],
            adj=edge_indexes_clients[i],
            train_labels=labels[communicate_node_indexes[i]
                                ][in_com_train_node_indexes[i]],
            test_labels=labels[communicate_node_indexes[i]
                               ][in_com_test_node_indexes[i]],
            features=features[split_node_indexes[i]],
            idx_train=in_com_train_node_indexes[i],
            idx_test=in_com_test_node_indexes[i],
            args_hidden=args_hidden,
            global_node_num=len(features),
            class_num=class_num,
            device=device,
            args=args,
        )
        for i in range(args.n_trainer)
    ]

    #######################################################################
    # Define Server
    # -------------
    # Server class is defined for federated aggregation (e.g., FedAvg)
    # without knowing the local trainer data

    server = Server(features.shape[1], args_hidden,
                    class_num, device, trainers, args)

    #######################################################################
    # Pre-Train Communication of FedGCN
    # ---------------------------------
    # Clients send their local feature sum to the server, and the server
    # aggregates all local feature sums and send the global feature sum
    # of specific nodes back to each client.

    # starting monitor:
    monitor = Monitor()
    monitor.pretrain_time_start()
    start_time = datetime.now()
    local_neighbor_feature_sums = [
        trainer.get_local_feature_sum.remote() for trainer in server.trainers
    ]
    global_feature_sum = torch.zeros_like(features)
    while True:
        ready, left = ray.wait(local_neighbor_feature_sums,
                               num_returns=1, timeout=None)
        if ready:
            for t in ready:
                global_feature_sum += ray.get(t)
        local_neighbor_feature_sums = left
        if not local_neighbor_feature_sums:
            break
    print("server aggregates all local neighbor feature sums")
    # test if aggregation is correct
    for i in range(args.n_trainer):
        server.trainers[i].load_feature_aggregation.remote(
            global_feature_sum[communicate_node_indexes[i]]
        )
    print("clients received feature aggregation from server")
    [trainer.relabel_adj.remote() for trainer in server.trainers]
    # ending monitor:
    end_time = datetime.now()
    time_difference = end_time - start_time
    print(f"pretraining time: {time_difference}")
    time.sleep(30)
    monitor.pretrain_time_end()

    #######################################################################
    # Federated Training
    # ------------------
    # The server start training of all clients and aggregate the parameters
    # at every global round.

    print("global_rounds", args.global_rounds)
    monitor.train_time_start()
    start_time = datetime.now()
    for i in range(args.global_rounds):
        server.train(i)
    end_time = datetime.now()
    time_difference = end_time - start_time
    print(f"training time: {time_difference}")
    time.sleep(30)
    monitor.train_time_end()

    #######################################################################
    # Summarize Experiment Results
    # ----------------------------
    # The server collects the local test loss and accuracy from all clients
    # then calculate the overall test loss and accuracy.

    train_data_weights = [len(i) for i in in_com_train_node_indexes]
    test_data_weights = [len(i) for i in in_com_test_node_indexes]

    results = [trainer.local_test.remote() for trainer in server.trainers]
    results = np.array([ray.get(result) for result in results])

    average_final_test_loss = np.average(
        [row[0] for row in results], weights=test_data_weights, axis=0
    )
    average_final_test_accuracy = np.average(
        [row[1] for row in results], weights=test_data_weights, axis=0
    )

    print(average_final_test_loss, average_final_test_accuracy)


for d in ["ogbn-mag"]:
    args.dataset = d
    for b in [10000]:
        args.iid_beta = b
        print(f"at dataset: {d}, beta: {b}")
        run()
ray.shutdown()
