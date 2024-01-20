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
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

print(os.getcwd())
print(os.listdir())
print(os.listdir(".."))

import sys

# adding additional module folders
sys.path.append(os.path.join(sys.path[0], "src", "utility"))
sys.path.append(os.path.join(sys.path[0], "src", "data"))


ray.init()

from fedgraph.data_process import load_data
from fedgraph.server_class import Server
from fedgraph.trainer_class import Trainer_General
from fedgraph.utils import (
    get_1hop_feature_sum,
    get_in_comm_indexes,
    increment_dir,
    label_dirichlet_partition,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="cora", type=str)

    parser.add_argument("-f", "--fedtype", default="fedgcn", type=str)

    parser.add_argument("-c", "--global_rounds", default=100, type=int)
    parser.add_argument("-i", "--local_step", default=3, type=int)
    parser.add_argument("-lr", "--learning_rate", default=0.5, type=float)

    parser.add_argument("-n", "--n_trainer", default=5, type=int)
    parser.add_argument("-nl", "--num_layers", default=2, type=int)
    parser.add_argument("-nhop", "--num_hops", default=2, type=int)
    parser.add_argument("-g", "--gpu", action="store_true")  # if -g, use gpu
    parser.add_argument("-iid_b", "--iid_beta", default=10000, type=float)

    parser.add_argument("-l", "--logdir", default="./runs", type=str)

    parser.add_argument("-r", "--repeat_time", default=10, type=int)
    args = parser.parse_args()
    print(args)

    # 'cora', 'citeseer', 'pubmed' #simulate #other dataset twitter,
    # 'ogbn-arxiv', reddit, "ogbn-products"
    np.random.seed(42)
    torch.manual_seed(42)

    features, adj, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
    class_num = labels.max().item() + 1

    if args.dataset in ["simulate", "cora", "citeseer", "pubmed", "reddit"]:
        args_hidden = 16
    else:
        args_hidden = 256

    row, col, edge_attr = adj.coo()
    edge_index = torch.stack([row, col], dim=0)

    # specifying a target GPU
    if args.gpu:
        device = torch.device("cuda")
        # running on a local machine with multiple gpu
        if args.dataset == "ogbn-products":
            edge_index = edge_index.to("cuda:7")
        else:
            edge_index = edge_index.to("cuda:0")
    else:
        device = torch.device("cpu")

    if device.type == "cpu":
        num_cpus = 0.1
        num_gpus = 0.0
    elif args.dataset == "ogbn-arxiv":
        num_cpus = 5.0
        num_gpus = 0.5
    else:
        num_cpus = 10
        num_gpus = 1.0

    # repeat experiments
    average_final_test_loss_repeats = []
    average_final_test_accuracy_repeats = []

    for repeat in range(args.repeat_time):
        # load data to cpu
        # beta = 0.0001 extremely Non-IID, beta = 10000, IID
        split_node_indexes = label_dirichlet_partition(
            labels, len(labels), class_num, args.n_trainer, beta=args.iid_beta
        )

        for i in range(args.n_trainer):
            split_node_indexes[i] = np.array(split_node_indexes[i])
            split_node_indexes[i].sort()
            split_node_indexes[i] = torch.tensor(split_node_indexes[i])

        # determine the resources for each trainer
        @ray.remote(
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            scheduling_strategy="SPREAD",
        )
        class Trainer(Trainer_General):
            def __init__(self, *args: Any, **kwds: Any):
                super().__init__(*args, **kwds)

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

        # send data to each trainer
        trainers = [
            Trainer.remote(
                rank=i,
                local_node_index=split_node_indexes[i],
                communicate_node_index=communicate_node_indexes[i],
                adj=edge_indexes_clients[i],
                train_labels=labels[communicate_node_indexes[i]][
                    in_com_train_node_indexes[i]
                ],
                test_labels=labels[communicate_node_indexes[i]][
                    in_com_test_node_indexes[i]
                ],
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

        args.log_dir = increment_dir(Path(args.logdir) / "exp")
        os.makedirs(args.log_dir)
        yaml_file = str(Path(args.log_dir) / "args.yaml")
        with open(yaml_file, "w") as out:
            yaml.dump(args.__dict__, out, default_flow_style=False)

        writer = SummaryWriter(args.log_dir)
        # clear cache
        torch.cuda.empty_cache()
        server = Server(
            features.shape[1], args_hidden, class_num, device, trainers, args
        )

        # pre-train communication
        local_neighbor_feature_sums = [
            trainer.get_local_feature_sum.remote() for trainer in server.trainers
        ]
        global_feature_sum = torch.zeros_like(features)
        while True:
            ready, left = ray.wait(
                local_neighbor_feature_sums, num_returns=1, timeout=None
            )
            if ready:
                for t in ready:
                    global_feature_sum += ray.get(t)
            local_neighbor_feature_sums = left
            if not local_neighbor_feature_sums:
                break
        print("server aggregates all local neighbor feature sums")
        # test if aggregation is correct
        if args.num_hops != 0:
            assert (
                global_feature_sum != get_1hop_feature_sum(features, edge_index)
            ).sum() == 0
        for i in range(args.n_trainer):
            server.trainers[i].load_feature_aggregation.remote(
                global_feature_sum[communicate_node_indexes[i]]
            )
        print("clients received feature aggregation from server")
        [trainer.relabel_adj.remote() for trainer in server.trainers]

        print("global_rounds", args.global_rounds)

        for i in range(args.global_rounds):
            server.train(i)

        results = [trainer.get_all_loss_accuray.remote() for trainer in server.trainers]
        results = np.array([ray.get(result) for result in results])

        client_id = 0
        for result in results:
            for iteration in range(len(result[0])):
                writer.add_scalar(
                    "Train Loss/Client_{}".format(client_id),
                    result[0][iteration],
                    iteration,
                )
            for iteration in range(len(result[1])):
                writer.add_scalar(
                    "Train Accuracy/Client_{}".format(client_id),
                    result[1][iteration],
                    iteration,
                )
            for iteration in range(len(result[2])):
                writer.add_scalar(
                    "Test Loss/Client_{}.format(client_id)",
                    result[2][iteration],
                    iteration,
                )
            for iteration in range(len(result[3])):
                writer.add_scalar(
                    "Test Accuracy/Client_{}".format(client_id),
                    result[3][iteration],
                    iteration,
                )
            client_id += 1

        train_data_weights = [len(i) for i in in_com_train_node_indexes]
        test_data_weights = [len(i) for i in in_com_test_node_indexes]

        average_train_loss = np.average(
            [row[0] for row in results], weights=train_data_weights, axis=0
        )
        average_train_accuracy = np.average(
            [row[1] for row in results], weights=train_data_weights, axis=0
        )
        average_test_loss = np.average(
            [row[2] for row in results], weights=test_data_weights, axis=0
        )
        average_test_accuracy = np.average(
            [row[3] for row in results], weights=test_data_weights, axis=0
        )

        for iteration in range(len(results[0][0])):
            writer.add_scalar(
                "Train Loss/Clients_Overall".format(),
                average_train_loss[iteration],
                iteration,
            )
            writer.add_scalar(
                "Train Accuracy/Clients_Overall".format(),
                average_train_accuracy[iteration],
                iteration,
            )
            writer.add_scalar(
                "Test Loss/Clients_Overall".format(),
                average_test_loss[iteration],
                iteration,
            )
            writer.add_scalar(
                "Train Accuracy/Clients_Overall".format(),
                average_test_accuracy[iteration],
                iteration,
            )

        results = [trainer.local_test.remote() for trainer in server.trainers]
        results = np.array([ray.get(result) for result in results])

        average_final_test_loss = np.average(
            [row[0] for row in results], weights=test_data_weights, axis=0
        )
        average_final_test_accuracy = np.average(
            [row[1] for row in results], weights=test_data_weights, axis=0
        )

        print(average_final_test_loss, average_final_test_accuracy)

        # sleep(5)  # wait for print message from remote workers
        filename = (
            args.dataset
            + "_"
            + args.fedtype
            + "_"
            + str(args.num_layers)
            + "_layer_"
            + str(args.num_hops)
            + "_hop_iid_beta_"
            + str(args.iid_beta)
            + "_n_trainer_"
            + str(args.n_trainer)
            + "_local_step_"
            + str(args.local_step)
            + ".txt"
        )
        with open(filename, "a+") as a:
            a.write(f"{average_final_test_loss} {average_final_test_accuracy}\n")
            average_final_test_loss_repeats.append(average_final_test_loss)
            average_final_test_accuracy_repeats.append(average_final_test_accuracy)

    # finish experiments
    with open(
        f"{args.dataset}_{args.fedtype}_{args.num_layers}_layer_{args.num_hops}_hop_iid_beta_{args.iid_beta}_n_trainer_{args.n_trainer}_local_step_{args.local_step}.txt",
        "a+",
    ) as a:
        a.write(
            f"average_testing_loss {np.average(average_final_test_loss_repeats)} std {np.std(average_final_test_loss_repeats)}\n"
        )
        a.write(
            f"average_testing_accuracy {np.average(average_final_test_accuracy_repeats)} std {np.std(average_final_test_accuracy_repeats)}\n"
        )

    print(
        f"average_testing_loss {np.average(average_final_test_loss_repeats)} std {np.std(average_final_test_loss_repeats)}"
    )
    print(
        f"average_testing_accuracy {np.average(average_final_test_accuracy_repeats)} std {np.std(average_final_test_accuracy_repeats)}"
    )

ray.shutdown()

