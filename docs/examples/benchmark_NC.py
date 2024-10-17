"""
FedGraph Example
================

In this tutorial, you will learn the basic workflow of
FedGraph with a runnable example. This tutorial assumes that
you have basic familiarity with PyTorch and PyTorch Geometric (PyG).

(Time estimate: 15 minutes)
"""
import argparse
import os
import time
from typing import Any

import numpy as np
import ray
import torch

from fedgraph.data_process import NC_load_data
from fedgraph.monitor_class import Monitor
from fedgraph.server_class import Server
from fedgraph.trainer_class import Trainer_General
from fedgraph.utils_nc import (
    get_1hop_feature_sum,
    get_in_comm_indexes,
    label_dirichlet_partition,
    save_all_trainers_data,
)

ray.init()


def run(
    dataset,
    batch_size,
    n_trainer,
    num_hops,
    iid_beta,
    distribution_type,
    use_huggingface=False,
    save=False,
):
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default=dataset, type=str)

    parser.add_argument("-f", "--method", default="fedgcn", type=str)

    parser.add_argument("-c", "--global_rounds", default=200, type=int)
    parser.add_argument("-b", "--batch_size", default=batch_size, type=int)
    parser.add_argument("-i", "--local_step", default=1, type=int)
    parser.add_argument("-lr", "--learning_rate", default=0.1, type=float)

    parser.add_argument("-n", "--n_trainer", default=n_trainer, type=int)
    parser.add_argument("-nl", "--num_layers", default=2, type=int)
    parser.add_argument("-nhop", "--num_hops", default=num_hops, type=int)
    parser.add_argument("-g", "--gpu", action="store_true")  # if -g, use gpu
    parser.add_argument("-iid_b", "--iid_beta", default=iid_beta, type=float)
    parser.add_argument(
        "-t", "--distribution_type", default=distribution_type, type=str
    )
    parser.add_argument("-l", "--logdir", default="./runs", type=str)

    args = parser.parse_args()

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
    if not use_huggingface:
        # process on the server
        features, adj, labels, idx_train, idx_val, idx_test = NC_load_data(args.dataset)
        features, adj, labels, idx_train, idx_val, idx_test = NC_load_data(args.dataset)
        class_num = labels.max().item() + 1
        row, col, edge_attr = adj.coo()
        edge_index = torch.stack([row, col], dim=0)
        #######################################################################
        # Split Graph for Federated Learning
        # ----------------------------------
        # FedGraph currents has two partition methods: label_dirichlet_partition
        # and community_partition_non_iid to split the large graph into multiple trainers
        split_node_indexes = label_dirichlet_partition(
            labels,
            len(labels),
            class_num,
            args.n_trainer,
            beta=args.iid_beta,
            distribution_type=args.distribution_type,
        )

        for i in range(args.n_trainer):
            split_node_indexes[i] = np.array(split_node_indexes[i])
            split_node_indexes[i].sort()
            split_node_indexes[i] = torch.tensor(split_node_indexes[i])

        (
            communicate_node_global_indexes,
            in_com_train_node_local_indexes,
            in_com_test_node_local_indexes,
            global_edge_indexes_clients,
        ) = get_in_comm_indexes(
            edge_index,
            split_node_indexes,
            args.n_trainer,
            args.num_hops,
            idx_train,
            idx_test,
        )
    if args.dataset in ["simulate", "cora", "citeseer", "pubmed", "reddit"]:
        args_hidden = 16
    else:
        args_hidden = 256

    num_cpus_per_client = 55  # m5.16xlarge
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

    if save:
        save_all_trainers_data(
            split_node_indexes=split_node_indexes,
            communicate_node_global_indexes=communicate_node_global_indexes,
            global_edge_indexes_clients=global_edge_indexes_clients,
            labels=labels,
            features=features,
            in_com_train_node_local_indexes=in_com_train_node_local_indexes,
            in_com_test_node_local_indexes=in_com_test_node_local_indexes,
            n_trainer=args.n_trainer,
            args=args,
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

    if use_huggingface:
        trainers = [
            Trainer.remote(  # type: ignore
                rank=i,
                args_hidden=args_hidden,
                # global_node_num=len(features),
                # class_num=class_num,
                device=device,
                args=args,
                # local_node_index=split_node_indexes[i],
                # communicate_node_index=communicate_node_global_indexes[i],
                # adj=global_edge_indexes_clients[i],
                # train_labels=labels[communicate_node_global_indexes[i]][
                #     in_com_train_node_local_indexes[i]
                # ],
                # test_labels=labels[communicate_node_global_indexes[i]][
                #     in_com_test_node_local_indexes[i]
                # ],
                # features=features[split_node_indexes[i]],
                # idx_train=in_com_train_node_local_indexes[i],
                # idx_test=in_com_test_node_local_indexes[i],
            )
            for i in range(args.n_trainer)
        ]
    else:  # load from the server
        trainers = [
            Trainer.remote(  # type: ignore
                rank=i,
                args_hidden=args_hidden,
                # global_node_num=len(features),
                # class_num=class_num,
                device=device,
                args=args,
                local_node_index=split_node_indexes[i],
                communicate_node_index=communicate_node_global_indexes[i],
                adj=global_edge_indexes_clients[i],
                train_labels=labels[communicate_node_global_indexes[i]][
                    in_com_train_node_local_indexes[i]
                ],
                test_labels=labels[communicate_node_global_indexes[i]][
                    in_com_test_node_local_indexes[i]
                ],
                features=features[split_node_indexes[i]],
                idx_train=in_com_train_node_local_indexes[i],
                idx_test=in_com_test_node_local_indexes[i],
            )
            for i in range(args.n_trainer)
        ]

    # Retrieve data information from all trainers
    trainer_information = [
        ray.get(trainers[i].get_info.remote()) for i in range(len(trainers))
    ]

    # Extract necessary details from trainer information
    global_node_num = sum([info["features_num"] for info in trainer_information])
    class_num = max([info["label_num"] for info in trainer_information])
    feature_shape = trainer_information[0]["feature_shape"]

    train_data_weights = [
        info["len_in_com_train_node_local_indexes"] for info in trainer_information
    ]
    test_data_weights = [
        info["len_in_com_test_node_local_indexes"] for info in trainer_information
    ]
    communicate_node_global_indexes = [
        info["communicate_node_global_index"] for info in trainer_information
    ]
    ray.get(
        [
            trainers[i].init_model.remote(global_node_num, class_num)
            for i in range(len(trainers))
        ]
    )
    #######################################################################
    # Define Server
    # -------------
    # Server class is defined for federated aggregation (e.g., FedAvg)
    # without knowing the local trainer data

    server = Server(feature_shape, args_hidden, class_num, device, trainers, args)
    server.broadcast_params(-1)
    #######################################################################
    # Pre-Train Communication of FedGCN
    # ---------------------------------
    # Clients send their local feature sum to the server, and the server
    # aggregates all local feature sums and send the global feature sum
    # of specific nodes back to each client.

    # starting monitor:
    monitor = Monitor()
    monitor.pretrain_time_start()
    local_neighbor_feature_sums = [
        trainer.get_local_feature_sum.remote() for trainer in server.trainers
    ]
    global_feature_sum = global_feature_sum = torch.zeros(
        (global_node_num, feature_shape), dtype=torch.float32
    )
    while True:
        print("starting collecting local feature sum")
        ready, left = ray.wait(local_neighbor_feature_sums, num_returns=1, timeout=None)
        if ready:
            for t in ready:
                global_feature_sum += ray.get(t)
                print("get one")
                print(global_feature_sum.size())
        local_neighbor_feature_sums = left
        if not local_neighbor_feature_sums:
            break
    print("server aggregates all local neighbor feature sums")
    # test if aggregation is correct
    # if args.num_hops != 0:
    #     assert (global_feature_sum != get_1hop_feature_sum(
    #         features, edge_index)).sum() == 0
    for i in range(args.n_trainer):
        server.trainers[i].load_feature_aggregation.remote(
            global_feature_sum[communicate_node_global_indexes[i]]
        )
    print("clients received feature aggregation from server")
    [trainer.relabel_adj.remote() for trainer in server.trainers]
    # ending monitor:
    monitor.pretrain_time_end(30)

    #######################################################################
    # Federated Training
    # ------------------
    # The server start training of all clients and aggregate the parameters
    # at every global round.

    print("global_rounds", args.global_rounds)
    monitor.train_time_start()
    for i in range(args.global_rounds):
        server.train(i)
    monitor.train_time_end(30)

    #######################################################################
    # Summarize Experiment Results
    # ----------------------------
    # The server collects the local test loss and accuracy from all clients
    # then calculate the overall test loss and accuracy.

    # train_data_weights = [len(i) for i in in_com_train_node_local_indexes]
    # test_data_weights = [len(i) for i in in_com_test_node_local_indexes]

    results = [trainer.local_test.remote() for trainer in server.trainers]
    results = np.array([ray.get(result) for result in results])

    average_final_test_loss = np.average(
        [row[0] for row in results], weights=test_data_weights, axis=0
    )
    average_final_test_accuracy = np.average(
        [row[1] for row in results], weights=test_data_weights, axis=0
    )

    # print(average_final_test_loss, average_final_test_accuracy)
    print(f"// Average test accuracy: {average_final_test_accuracy}//end")


for dataset in ["ogbn-papers100M"]:
    for n_trainer in [10]:
        for distribution_type in ["average"]:
            for iid_beta in [1.0]:
                for num_hops in [1]:
                    for batch_size in [-1]:
                        for i in range(1):
                            print(
                                f"Running experiment with: Dataset={dataset},"
                                f"Number of Trainers={n_trainer}, Distribution Type={distribution_type},"
                                f"IID Beta={iid_beta}, Number of Hops={num_hops}, Batch Size={batch_size}"
                            )
                            run(
                                dataset,
                                batch_size,
                                n_trainer,
                                num_hops,
                                iid_beta,
                                distribution_type,
                                use_huggingface=False,
                                save=False,
                            )


ray.shutdown()
