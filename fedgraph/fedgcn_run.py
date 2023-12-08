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

from data_process import generate_data, load_data
from gnn_models import GCN, GCN_arxiv, SAGE_products
from server_class import Server
from trainer_class import Trainer_General
from utils import (
    get_in_comm_indexes,
    get_in_comm_indexes_BDS_GCN,
    increment_dir,
    label_dirichlet_partition,
    parition_non_iid,
    setdiff1d,
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

    # load data to cpu
    if args.dataset == "simulate":
        number_of_nodes = 200
        class_num = 3
        link_inclass_prob = 10 / number_of_nodes
        link_outclass_prob = link_inclass_prob / 20
        features, adj, labels, idx_train, idx_val, idx_test = generate_data(
            number_of_nodes, class_num, link_inclass_prob, link_outclass_prob
        )
    else:
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
        split_data_indexes = label_dirichlet_partition(
            labels, len(labels), class_num, args.n_trainer, beta=args.iid_beta
        )

        for i in range(args.n_trainer):
            split_data_indexes[i] = np.array(split_data_indexes[i])
            split_data_indexes[i].sort()
            split_data_indexes[i] = torch.tensor(split_data_indexes[i])

        if args.fedtype == "bds-gcn":
            print("running bds-gcn")
            # No args.num_hops
            (
                communicate_indexes,
                in_com_train_data_indexes,
                in_com_test_data_indexes,
                edge_indexes_clients,
            ) = get_in_comm_indexes_BDS_GCN(
                edge_index, split_data_indexes, args.n_trainer, idx_train, idx_test
            )
        else:
            (
                communicate_indexes,
                in_com_train_data_indexes,
                in_com_test_data_indexes,
                edge_indexes_clients,
            ) = get_in_comm_indexes(
                edge_index,
                split_data_indexes,
                args.n_trainer,
                args.num_hops,
                idx_train,
                idx_test,
            )

        # determine the resources for each trainer
        @ray.remote(
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            scheduling_strategy="SPREAD",
        )
        class Trainer(Trainer_General):
            def __init__(self, *args: Any, **kwds: Any):
                super().__init__(*args, **kwds)

        if args.fedtype == "fedsage+":
            print("running fedsage+")
            features_in_clients = []
            # assume the linear generator learnt the optimal (the average of features of neighbor nodes)
            # gaussian noise

            for i in range(args.n_trainer):
                # original features of outside neighbors of nodes in client i
                original_feature_i = features[
                    setdiff1d(split_data_indexes[i], communicate_indexes[i])
                ].clone()

                # add gaussian noise to the communicated feature
                gaussian_feature_i = (
                    original_feature_i
                    + torch.normal(0, 0.1, original_feature_i.shape).cpu()
                )

                copy_feature = features.clone()

                copy_feature[
                    setdiff1d(split_data_indexes[i], communicate_indexes[i])
                ] = gaussian_feature_i

                features_in_clients.append(copy_feature[communicate_indexes[i]])
            trainers = [
                Trainer.remote(
                    i,
                    edge_indexes_clients[i],
                    labels[communicate_indexes[i]],
                    features_in_clients[i],
                    in_com_train_data_indexes[i],
                    in_com_test_data_indexes[i],
                    args_hidden,
                    class_num,
                    device,
                    args,
                )
                for i in range(args.n_trainer)
            ]
        else:
            trainers = [
                Trainer.remote(
                    i,
                    edge_indexes_clients[i],
                    labels[communicate_indexes[i]],
                    features[communicate_indexes[i]],
                    in_com_train_data_indexes[i],
                    in_com_test_data_indexes[i],
                    args_hidden,
                    class_num,
                    device,
                    args,
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

        train_data_weights = [len(i) for i in in_com_train_data_indexes]
        test_data_weights = [len(i) for i in in_com_test_data_indexes]

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
