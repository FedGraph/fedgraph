import argparse
import copy
import os
import random
from pathlib import Path
from typing import Any

import attridict
import numpy as np
import ray
import torch

from fedgraph.data_process_gc import load_multiple_dataset, load_single_dataset
from fedgraph.server_class import Server
from fedgraph.train_func import *
from fedgraph.trainer_class import Trainer_General
from fedgraph.utils import get_1hop_feature_sum
from fedgraph.utils_gc import *


def FedGCN_Train(args: attridict, data: tuple) -> None:
    """
    Train a FedGCN model.

    Parameters
    ----------
    args
    data
    """

    ray.init()

    (
        edge_index,
        features,
        labels,
        idx_train,
        idx_test,
        class_num,
        split_node_indexes,
        communicate_node_indexes,
        in_com_train_node_indexes,
        in_com_test_node_indexes,
        edge_indexes_clients,
    ) = data

    if args.dataset in ["simulate", "cora", "citeseer", "pubmed", "reddit"]:
        args_hidden = 16
    else:
        args_hidden = 256

    num_cpus_per_client = 1
    # specifying a target GPU
    if args.gpu:
        device = torch.device("cuda")
        num_gpus_per_client = 1
    else:
        device = torch.device("cpu")
        num_gpus_per_client = 0

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

    #######################################################################
    # Define Server
    # -------------
    # Server class is defined for federated aggregation (e.g., FedAvg)
    # without knowing the local trainer data

    server = Server(features.shape[1], args_hidden, class_num, device, trainers, args)

    #######################################################################
    # Pre-Train Communication of FedGCN
    # ---------------------------------
    # Clients send their local feature sum to the server, and the server
    # aggregates all local feature sums and send the global feature sum
    # of specific nodes back to each client.

    local_neighbor_feature_sums = [
        trainer.get_local_feature_sum.remote() for trainer in server.trainers
    ]
    global_feature_sum = torch.zeros_like(features)
    while True:
        ready, left = ray.wait(local_neighbor_feature_sums, num_returns=1, timeout=None)
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

    #######################################################################
    # Federated Training
    # ------------------
    # The server start training of all clients and aggregate the parameters
    # at every global round.

    print("global_rounds", args.global_rounds)

    for i in range(args.global_rounds):
        server.train(i)

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
    print(f"average_final_test_loss, {average_final_test_loss}")
    print(f"average_final_test_accuracy, {average_final_test_accuracy}")
    ray.shutdown()


def GC_Train(config: dict) -> None:
    """
    Entrance of the training process for graph classification.

    Parameters
    ----------
    model: str
        The model to run.
    """
    # transfer the config to argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    for key, value in config.items():
        setattr(args, key, value)

    print(args)

    #################### set seeds and devices ####################
    seed_split_data = 42  # seed for splitting data must be fixed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    #################### set output directory ####################
    # outdir_base = os.path.join(args.outbase, f'seqLen{args.seq_length}')
    outdir_base = args.outbase + "/" + f"{args.model}"
    outdir = os.path.join(outdir_base, f"oneDS-nonOverlap")
    if args.model in ["SelfTrain"]:
        outdir = os.path.join(outdir, f"{args.data_group}")
    elif args.model in ["FedAvg", "FedProx"]:
        outdir = os.path.join(outdir, f"{args.data_group}-{args.num_clients}clients")
    elif args.model in ["GCFL"]:
        outdir = os.path.join(
            outdir,
            f"{args.data_group}-{args.num_clients}clients",
            f"eps_{args.epsilon1}_{args.epsilon2}",
        )
    elif args.model in ["GCFL+", "GCFL+dWs"]:
        outdir = os.path.join(
            outdir,
            f"{args.data_group}-{args.num_clients}clients",
            f"eps_{args.epsilon1}_{args.epsilon2}",
            f"seqLen{args.seq_length}",
        )

    Path(outdir).mkdir(parents=True, exist_ok=True)
    print(f"Output Path: {outdir}")

    #################### distributed one dataset to multiple clients ####################
    """ using original features """
    print("Preparing data (original features) ...")

    splited_data, df_stats = load_single_dataset(
        args.datapath,
        args.data_group,
        num_client=args.num_clients,
        batch_size=args.batch_size,
        convert_x=args.convert_x,
        seed=seed_split_data,
        overlap=args.overlap,
    )
    print("Data prepared.")

    #################### save statistics of data on clients ####################
    outdir_stats = os.path.join(outdir, f"stats_train_data.csv")
    df_stats.to_csv(outdir_stats)
    print(f"The statistics of the data are written to {outdir_stats}")

    #################### setup devices ####################
    if args.model not in ["SelfTrain"]:
        init_clients, _ = setup_clients(splited_data, args)
        init_server = setup_server(args)
        clients = copy.deepcopy(init_clients)
        server = copy.deepcopy(init_server)

    print("\nDone setting up devices.")

    ################ choose the model to run ################
    print(f"Running {args.model} ...")
    if args.model == "SelfTrain":
        output = run_GC_selftrain(
            clients=clients, server=server, local_epoch=args.local_epoch
        )

    elif args.model == "FedAvg":
        output = run_GC_fedavg(
            clients=clients,
            server=server,
            communication_rounds=args.num_rounds,
            local_epoch=args.local_epoch,
            samp=None,
        )

    elif args.model == "FedProx":
        output = run_GC_fedprox(
            clients=clients,
            server=server,
            communication_rounds=args.num_rounds,
            local_epoch=args.local_epoch,
            mu=args.mu,
            samp=None,
        )

    elif args.model == "GCFL":
        output = run_GC_gcfl(
            clients=clients,
            server=server,
            communication_rounds=args.num_rounds,
            local_epoch=args.local_epoch,
            EPS_1=args.epsilon1,
            EPS_2=args.epsilon2,
        )

    elif args.model == "GCFL+":
        output = run_GC_gcfl_plus(
            clients=clients,
            server=server,
            communication_rounds=args.num_rounds,
            local_epoch=args.local_epoch,
            EPS_1=args.epsilon1,
            EPS_2=args.epsilon2,
            seq_length=args.seq_length,
            standardize=args.standardize,
        )

    elif args.model == "GCFL+dWs":
        output = run_GC_gcfl_plus(
            clients=clients,
            server=server,
            communication_rounds=args.num_rounds,
            local_epoch=args.local_epoch,
            EPS_1=args.epsilon1,
            EPS_2=args.epsilon2,
            seq_length=args.seq_length,
            standardize=args.standardize,
        )

    else:
        raise ValueError(f"Unknown model: {args.model}")

    #################### save the output ####################
    outdir_result = os.path.join(outdir, f"accuracy_seed{args.seed}.csv")
    pd.DataFrame(output).to_csv(outdir_result)
    print(f"The output has been written to file: {outdir_result}")
