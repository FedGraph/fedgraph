import argparse
import copy
import datetime
import os
import random
import time
from pathlib import Path
from typing import Any

import attridict
import numpy as np
import pandas as pd
import ray
import torch

from fedgraph.gnn_models import GIN
from fedgraph.monitor_class import Monitor
from fedgraph.server_class import Server, Server_LP
from fedgraph.train_func import gc_avg_accuracy
from fedgraph.trainer_class import Trainer_General, Trainer_LP
from fedgraph.utils_gc import setup_server, setup_trainers
from fedgraph.utils_lp import (
    check_data_files_existance,
    get_global_user_item_mapping,
    get_start_end_time,
    to_next_day,
)
from fedgraph.utils_nc import get_1hop_feature_sum


def run_fedgraph(args: attridict, data: Any) -> None:
    """
    Run the training process for the specified task.

    Parameters
    ----------
    args: attridict
        The arguments.
    data: Any
        The data.
    """
    if args.fedgraph_task == "FedGCN":
        run_FedGCN(args, data)
    elif args.fedgraph_task == "GC":
        run_GC(args, data)
    elif args.fedgraph_task == "LP":
        run_LP(args)


def run_FedGCN(args: attridict, data: tuple) -> None:
    """
    Train a FedGCN model.

    Parameters
    ----------
    args
    data
    """

    ray.init(address="auto")

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
        edge_indexes_trainers,
    ) = data

    if args.dataset in ["simulate", "cora", "citeseer", "pubmed", "reddit"]:
        args_hidden = 16
    else:
        args_hidden = 256

    num_cpus_per_trainer = 3
    # specifying a target GPU
    if args.gpu:
        device = torch.device("cuda")
        num_gpus_per_trainer = 1
    else:
        device = torch.device("cpu")
        num_gpus_per_trainer = 0

    #######################################################################
    # Define and Send Data to Trainers
    # --------------------------------
    # FedGraph first determines the resources for each trainer, then send
    # the data to each remote trainer.

    @ray.remote(
        num_gpus=num_gpus_per_trainer,
        num_cpus=num_cpus_per_trainer,
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
            adj=edge_indexes_trainers[i],
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
    # of specific nodes back to each trainer.

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
    print("trainers received feature aggregation from server")
    [trainer.relabel_adj.remote() for trainer in server.trainers]

    #######################################################################
    # Federated Training
    # ------------------
    # The server start training of all trainers and aggregate the parameters
    # at every global round.

    print("global_rounds", args.global_rounds)

    for i in range(args.global_rounds):
        server.train(i)

    #######################################################################
    # Summarize Experiment Results
    # ----------------------------
    # The server collects the local test loss and accuracy from all trainers
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


def run_GC(args: attridict, data: Any, base_model: Any = GIN) -> None:
    """
    Entrance of the training process for graph classification.

    Parameters
    ----------
    args: attridict
        The arguments.
    data: Any
        The splitted data.
    base_model: Any
        The base model on which the federated learning is based. It applies for both the server and the trainers.
    """
    # transfer the config to argparse

    #################### set seeds and devices ####################
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    #################### set output directory ####################
    # outdir_base = os.path.join(args.outbase, f'seqLen{args.seq_length}')
    if args.save_files:
        outdir_base = args.outbase + "/" + f"{args.model}"
        outdir = os.path.join(outdir_base, f"oneDS-nonOverlap")
        if args.model in ["SelfTrain"]:
            outdir = os.path.join(outdir, f"{args.data_group}")
        elif args.model in ["FedAvg", "FedProx"]:
            outdir = os.path.join(
                outdir, f"{args.data_group}-{args.num_trainers}trainers"
            )
        elif args.model in ["GCFL"]:
            outdir = os.path.join(
                outdir,
                f"{args.data_group}-{args.num_trainers}trainers",
                f"eps_{args.epsilon1}_{args.epsilon2}",
            )
        elif args.model in ["GCFL+", "GCFL+dWs"]:
            outdir = os.path.join(
                outdir,
                f"{args.data_group}-{args.num_trainers}trainers",
                f"eps_{args.epsilon1}_{args.epsilon2}",
                f"seqLen{args.seq_length}",
            )
        outdir = os.path.join(outdir, f"seed{args.seed}")
        Path(outdir).mkdir(parents=True, exist_ok=True)
        print(f"Output Path: {outdir}")

    #################### save statistics of data on trainers ####################
    # if args.save_files and df_stats:
    #     outdir_stats = os.path.join(outdir, f"stats_train_data.csv")
    #     df_stats.to_csv(outdir_stats)
    #     print(f"The statistics of the data are written to {outdir_stats}")

    #################### setup server and trainers ####################
    init_trainers, _ = setup_trainers(data, base_model, args)
    init_server = setup_server(base_model, args)
    trainers = copy.deepcopy(init_trainers)
    server = copy.deepcopy(init_server)

    print("\nDone setting up devices.")

    ################ choose the algorithm to run ################
    print(f"Running {args.model} ...")
    if args.model == "SelfTrain":
        output = run_GC_selftrain(
            trainers=trainers, server=server, local_epoch=args.local_epoch
        )

    elif args.model == "FedAvg":
        output = run_GC_fedavg(
            trainers=trainers,
            server=server,
            communication_rounds=args.num_rounds,
            local_epoch=args.local_epoch,
        )

    elif args.model == "FedProx":
        output = run_GC_fedprox(
            trainers=trainers,
            server=server,
            communication_rounds=args.num_rounds,
            local_epoch=args.local_epoch,
            mu=args.mu,
        )

    elif args.model == "GCFL":
        output = run_GC_gcfl(
            trainers=trainers,
            server=server,
            communication_rounds=args.num_rounds,
            local_epoch=args.local_epoch,
            EPS_1=args.epsilon1,
            EPS_2=args.epsilon2,
        )

    elif args.model == "GCFL+":
        output = run_GC_gcfl_plus(
            trainers=trainers,
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
            trainers=trainers,
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
    if args.save_files:
        outdir_result = os.path.join(outdir, f"accuracy_seed{args.seed}.csv")
        pd.DataFrame(output).to_csv(outdir_result)
        print(f"The output has been written to file: {outdir_result}")


# The following code is the implementation of different federated graph classification methods.
def run_GC_selftrain(trainers: list, server: Any, local_epoch: int) -> dict:
    """
    Run the training and testing process of self-training algorithm.
    It only trains the model locally, and does not perform weights aggregation.

    Parameters
    ----------
    trainers: list
        List of trainers, each of which is a Trainer_GC object
    server: Any
        Server_GC object
    local_epoch: int
        Number of local epochs

    Returns
    -------
    all_accs: dict
        Dictionary with training and test accuracies for each trainer
    """
    # all trainers are initialized with the same weights
    for trainer in trainers:
        trainer.update_params(server)

    all_accs = {}
    for trainer in trainers:
        trainer.local_train(local_epoch=local_epoch)

        _, acc = trainer.local_test()
        all_accs[trainer.name] = [
            trainer.train_stats["trainingAccs"][-1],
            trainer.train_stats["valAccs"][-1],
            acc,
        ]
        print("  > {} done.".format(trainer.name))

    frame = pd.DataFrame(all_accs).T.iloc[:, [2]]
    frame.columns = ["test_acc"]
    print(frame)
    print(f"Average test accuracy: {gc_avg_accuracy(frame, trainers)}")
    return frame


def run_GC_fedavg(
    trainers: list,
    server: Any,
    communication_rounds: int,
    local_epoch: int,
    sampling_frac: float = 1.0,
) -> pd.DataFrame:
    """
    Run the training and testing process of FedAvg algorithm.
    It trains the model locally, aggregates the weights to the server,
    and downloads the global model within each communication round.

    Parameters
    ----------
    trainers: list
        List of trainers, each of which is a Trainer_GC object
    server: Any
        Server_GC object
    communication_rounds: int
        Number of communication rounds
    local_epoch: int
        Number of local epochs
    frac: float
        Fraction of trainers to sample

    Returns
    -------
    frame: pd.DataFrame
        Pandas dataframe with test accuracies
    """

    for trainer in trainers:
        trainer.update_params(server)  # download the global model

    # Overall training architecture:
    # whole training => { communication rounds, communication rounds, ..., communication rounds }
    # communication rounds => { local training (#epochs) -> aggregation -> download }
    #                                |
    #                           training_stats
    for c_round in range(1, communication_rounds + 1):
        if (c_round) % 10 == 0:
            print(
                f"  > Training round {c_round} finished."
            )  # print the current round every 10 rounds

        if c_round == 1:
            selected_trainers = trainers
        else:
            selected_trainers = server.random_sample_trainers(trainers, sampling_frac)
            # if sampling_frac=1.0, then all trainers are selected

        for trainer in selected_trainers:  # only get weights of graphconv layers
            # train the local model
            trainer.local_train(local_epoch=local_epoch)

        server.aggregate_weights(
            selected_trainers
        )  # aggregate the weights of selected trainers
        for trainer in selected_trainers:
            trainer.update_params(server)  # re-download the global server

    frame = pd.DataFrame()
    for trainer in trainers:
        _, acc = trainer.local_test()  # Final evaluation
        frame.loc[trainer.name, "test_acc"] = acc

    def highlight_max(s: pd.Series) -> list:
        is_max = s == s.max()
        return ["background-color: yellow" if v else "" for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)
    print(f"Average test accuracy: {gc_avg_accuracy(frame, trainers)}")
    return frame


def run_GC_fedprox(
    trainers: list,
    server: Any,
    communication_rounds: int,
    local_epoch: int,
    mu: float,
    sampling_frac: float = 1.0,
) -> pd.DataFrame:
    """
    Run the training and testing process of FedProx algorithm.
    It trains the model locally, aggregates the weights to the server,
    and downloads the global model within each communication round.

    Parameters
    ----------
    trainers: list
        List of trainers, each of which is a Trainer_GC object
    server: Any
        Server_GC object
    communication_rounds: int
        Number of communication rounds
    local_epoch: int
        Number of local epochs
    mu: float
        Proximal term
    frac: float
        Fraction of trainers to sample

    Returns:
        Frame: pandas dataframe with test accuracies
    """
    for trainer in trainers:
        trainer.update_params(server)

    for c_round in range(1, communication_rounds + 1):
        if (c_round) % 10 == 0:
            print(f"  > Training round {c_round} finished.")

        if c_round == 1:
            selected_trainers = trainers
        else:
            selected_trainers = server.random_sample_trainers(trainers, sampling_frac)

        for trainer in selected_trainers:
            trainer.local_train(
                local_epoch=local_epoch, train_option="prox", mu=mu
            )  # Different from FedAvg

        server.aggregate_weights(selected_trainers)
        for trainer in selected_trainers:
            trainer.update_params(server)

            # cache the aggregated weights for next round
            trainer.cache_weights()

    frame = pd.DataFrame()
    for trainer in trainers:
        _, acc = trainer.local_test()
        frame.loc[trainer.name, "test_acc"] = acc

    def highlight_max(s: pd.Series) -> list:
        is_max = s == s.max()
        return ["background-color: yellow" if v else "" for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)
    print(f"Average test accuracy: {gc_avg_accuracy(frame, trainers)}")
    return frame


def run_GC_gcfl(
    trainers: list,
    server: Any,
    communication_rounds: int,
    local_epoch: int,
    EPS_1: float,
    EPS_2: float,
) -> pd.DataFrame:
    """
    Run the GCFL algorithm.
    The GCFL algorithm is a cluster-based federated learning algorithm, which aggregates the weights of the trainers
    in each cluster, and dynamically splits the clusters during the training process.

    Parameters
    ----------
    trainers: list
        List of trainers, each of which is a Trainer_GC object
    server: Any
        Server_GC object
    communication_rounds: int
        Number of communication rounds
    local_epoch: int
        Number of local epochs
    EPS_1: float
        Threshold for mean update norm
    EPS_2: float
        Threshold for max update norm

    Returns
    -------
    frame: pandas.DataFrame
        Pandas dataframe with test accuracies

    """

    cluster_indices = [
        np.arange(len(trainers)).astype("int")
    ]  # cluster_indices: [[0, 1, ...]]
    trainer_clusters = [
        [trainers[i] for i in idcs] for idcs in cluster_indices
    ]  # initially there is only one cluster

    acc_trainers: list = []
    ############### COMMUNICATION ROUNDS ###############
    for c_round in range(1, communication_rounds + 1):
        if (c_round) % 10 == 0:
            print(f"  > Training round {c_round} finished.")
        if c_round == 1:
            for trainer in trainers:
                trainer.update_params(server)

        participating_trainers = server.random_sample_trainers(trainers, frac=1.0)
        for trainer in participating_trainers:
            trainer.local_train(
                local_epoch=local_epoch, train_option="gcfl"
            )  # local training
            trainer.reset_params()  # reset the gradients (discard the final gradients)

        similarities = server.compute_pairwise_similarities(trainers)

        cluster_indices_new = []
        for idc in cluster_indices:  # cluster-wise checking
            max_norm = server.compute_max_update_norm(
                [trainers[i] for i in idc]
            )  # DELTA_MAX
            mean_norm = server.compute_mean_update_norm(
                [trainers[i] for i in idc]
            )  # DELTA_MEAN
            if (
                mean_norm < EPS_1 and max_norm > EPS_2 and len(idc) > 2 and c_round > 20
            ):  # stopping condition
                server.cache_model(idc, trainers[idc[0]].W, acc_trainers)
                c1, c2 = server.min_cut(similarities[idc][:, idc], idc)
                cluster_indices_new += [c1, c2]  # split the cluster into two
            else:
                cluster_indices_new += [idc]  # keep the same cluster

        cluster_indices = cluster_indices_new
        trainer_clusters = [
            [trainers[i] for i in idcs] for idcs in cluster_indices
        ]  # initial: [[0, 1, ...]]

        server.aggregate_clusterwise(
            trainer_clusters
        )  # aggregate the weights of the trainers in each cluster

        acc_trainers = [
            trainer.local_test()[1] for trainer in trainers
        ]  # get the test accuracy of each trainer
    ############### END OF COMMUNICATION ROUNDS ###############

    for idc in cluster_indices:
        server.cache_model(
            idc, trainers[idc[0]].W, acc_trainers
        )  # cache the first trainer's weights in each cluster
    # cluster-wise model

    results = np.zeros([len(trainers), len(server.model_cache)])
    for i, (idcs, W, accs) in enumerate(server.model_cache):
        results[idcs, i] = np.array(accs)

    frame = pd.DataFrame(
        results,
        columns=["FL Model"]
        + ["Model {}".format(i) for i in range(results.shape[1] - 1)],
        index=["{}".format(trainers[i].name) for i in range(results.shape[0])],
    )
    frame = pd.DataFrame(frame.max(axis=1))
    frame.columns = ["test_acc"]

    print(frame)
    print(f"Average test accuracy: {gc_avg_accuracy(frame, trainers)}")
    return frame


def run_GC_gcfl_plus(
    trainers: list,
    server: Any,
    communication_rounds: int,
    local_epoch: int,
    EPS_1: float,
    EPS_2: float,
    seq_length: int,
    standardize: bool,
) -> pd.DataFrame:
    """
    Run the GCFL+ algorithm.

    Parameters
    ----------
    trainers: list
        List of trainers, each of which is a Trainer_GC object
    server: Any
        Server_GC object
    communication_rounds: int
        Number of communication rounds
    local_epoch: int
        Number of local epochs
    EPS_1: float
        Threshold for mean update norm
    EPS_2: float
        Threshold for max update norm
    seq_length: int
        The length of the gradient norm sequence
    standardize: bool
        Whether to standardize the distance matrix

    Returns
    -------
    frame: pandas.DataFrame
        Pandas dataframe with test accuracies
    """
    cluster_indices = [np.arange(len(trainers)).astype("int")]
    trainer_clusters = [[trainers[i] for i in idcs] for idcs in cluster_indices]

    seqs_grads: Any = {c.id: [] for c in trainers}

    for trainer in trainers:
        trainer.update_params(server)

    acc_trainers: list = []
    for c_round in range(1, communication_rounds + 1):
        if (c_round) % 10 == 0:
            print(f"  > Training round {c_round} finished.")
        if c_round == 1:
            for trainer in trainers:
                trainer.update_params(server)

        participating_trainers = server.random_sample_trainers(trainers, frac=1.0)

        for trainer in participating_trainers:
            trainer.local_train(local_epoch=local_epoch, train_option="gcfl")
            trainer.reset_params()

            seqs_grads[trainer.id].append(trainer.conv_grads_norm)

        cluster_indices_new = []
        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([trainers[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([trainers[i] for i in idc])
            if (
                mean_norm < EPS_1
                and max_norm > EPS_2
                and len(idc) > 2
                and c_round > 20
                and all(len(value) >= seq_length for value in seqs_grads.values())
            ):
                server.cache_model(idc, trainers[idc[0]].W, acc_trainers)

                tmp = [seqs_grads[id][-seq_length:] for id in idc]
                dtw_distances = server.compute_pairwise_distances(tmp, standardize)
                c1, c2 = server.min_cut(np.max(dtw_distances) - dtw_distances, idc)
                cluster_indices_new += [c1, c2]

                seqs_grads = {c.id: [] for c in trainers}
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        trainer_clusters = [[trainers[i] for i in idcs] for idcs in cluster_indices]

        server.aggregate_clusterwise(trainer_clusters)

        acc_trainers = [trainer.local_test()[1] for trainer in trainers]

    for idc in cluster_indices:
        server.cache_model(idc, trainers[idc[0]].W, acc_trainers)

    results = np.zeros([len(trainers), len(server.model_cache)])
    for i, (idcs, W, accs) in enumerate(server.model_cache):
        results[idcs, i] = np.array(accs)

    frame = pd.DataFrame(
        results,
        columns=["FL Model"]
        + ["Model {}".format(i) for i in range(results.shape[1] - 1)],
        index=["{}".format(trainers[i].name) for i in range(results.shape[0])],
    )

    frame = pd.DataFrame(frame.max(axis=1))
    frame.columns = ["test_acc"]
    print(frame)
    print(f"Average test accuracy: {gc_avg_accuracy(frame, trainers)}")

    return frame


def run_GC_gcfl_plus_dWs(
    trainers: list,
    server: Any,
    communication_rounds: int,
    local_epoch: int,
    EPS_1: float,
    EPS_2: float,
    seq_length: int,
    standardize: bool,
) -> pd.DataFrame:
    """
    Run the GCFL+ algorithm with the gradient norms of the weights.

    Parameters
    ----------
    trainers: list
        List of trainers, each of which is a Trainer_GC object
    server: Any
        Server_GC object
    communication_rounds: int
        Number of communication rounds
    local_epoch: int
        Number of local epochs
    EPS_1: float
        Threshold for mean update norm
    EPS_2: float
        Threshold for max update norm
    seq_length: int
        The length of the gradient norm sequence
    standardize: bool
        Whether to standardize the distance matrix

    Returns
    -------
    frame: pandas.DataFrame
        Pandas dataframe with test accuracies
    """
    cluster_indices = [np.arange(len(trainers)).astype("int")]
    trainer_clusters = [[trainers[i] for i in idcs] for idcs in cluster_indices]

    seqs_grads: Any = {c.id: [] for c in trainers}
    for trainer in trainers:
        trainer.update_params(server)

    acc_trainers: list = []
    for c_round in range(1, communication_rounds + 1):
        if (c_round) % 10 == 0:
            print(f"  > Training round {c_round} finished.")
        if c_round == 1:
            for trainer in trainers:
                trainer.update_params(server)

        participating_trainers = server.random_sample_trainers(trainers, frac=1.0)

        for trainer in participating_trainers:
            trainer.local_train(local_epoch=local_epoch, train_option="gcfl")
            trainer.reset_params()

            seqs_grads[trainer.id].append(trainer.conv_dWs_norm)

        cluster_indices_new = []
        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([trainers[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([trainers[i] for i in idc])
            if (
                mean_norm < EPS_1
                and max_norm > EPS_2
                and len(idc) > 2
                and c_round > 20
                and all(len(value) >= seq_length for value in seqs_grads.values())
            ):
                server.cache_model(idc, trainers[idc[0]].W, acc_trainers)

                tmp = [seqs_grads[id][-seq_length:] for id in idc]
                dtw_distances = server.compute_pairwise_distances(tmp, standardize)
                c1, c2 = server.min_cut(np.max(dtw_distances) - dtw_distances, idc)
                cluster_indices_new += [c1, c2]

                seqs_grads = {c.id: [] for c in trainers}
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        trainer_clusters = [[trainers[i] for i in idcs] for idcs in cluster_indices]

        server.aggregate_clusterwise(trainer_clusters)

        acc_trainers = [trainer.local_test()[1] for trainer in trainers]

    for idc in cluster_indices:
        server.cache_model(idc, trainers[idc[0]].W, acc_trainers)

    results = np.zeros([len(trainers), len(server.model_cache)])
    for i, (idcs, W, accs) in enumerate(server.model_cache):
        results[idcs, i] = np.array(accs)

    frame = pd.DataFrame(
        results,
        columns=["FL Model"]
        + ["Model {}".format(i) for i in range(results.shape[1] - 1)],
        index=["{}".format(trainers[i].name) for i in range(results.shape[0])],
    )
    frame = pd.DataFrame(frame.max(axis=1))
    frame.columns = ["test_acc"]
    print(frame)
    print(f"Average test accuracy: {gc_avg_accuracy(frame, trainers)}")

    return frame


def run_LP(args: attridict) -> None:
    """
    Run the training process for link prediction.

    Parameters
    ----------
    args: attridict
        The arguments.
    """

    def setup_trainer_server(
        country_codes: list,
        user_id_mapping: Any,
        item_id_mapping: Any,
        meta_data: tuple,
        hidden_channels: int = 64,
    ) -> tuple:
        """
        Setup the trainer and server

        Parameters
        ----------
        country_codes: list
            The list of country codes
        user_id_mapping: Any
            The user id mapping
        item_id_mapping: Any
            The item id mapping
        meta_data: tuple
            The meta data
        hidden_channels: int, optional
            The number of hidden channels

        Returns
        -------
        (list, Server_LP): tuple
            [0]: The list of clients
            [1]: The server
        """
        ray.init(address="auto")
        number_of_clients = len(country_codes)
        number_of_users, number_of_items = len(user_id_mapping.keys()), len(
            item_id_mapping.keys()
        )
        num_cpus_per_client = 3
        if args.device == "gpu":
            device = torch.device("cuda")
            print("gpu detected")
            num_gpus_per_client = 1
        else:
            device = torch.device("cpu")
            num_gpus_per_client = 0
            print("gpu not detected")

        @ray.remote(
            num_gpus=num_gpus_per_client,
            num_cpus=num_cpus_per_client,
            scheduling_strategy="SPREAD",
        )
        class Trainer(Trainer_LP):
            def __init__(self, *args, **kwargs):  # type: ignore
                super().__init__(*args, **kwargs)

        clients = [
            Trainer.remote(  # type: ignore
                i,
                country_code=args.country_codes[i],
                user_id_mapping=user_id_mapping,
                item_id_mapping=item_id_mapping,
                number_of_users=number_of_users,
                number_of_items=number_of_items,
                meta_data=meta_data,
                hidden_channels=args.hidden_channels,
            )
            for i in range(number_of_clients)
        ]

        server = Server_LP(  # the concrete information of users and items is not available in the server
            number_of_users=number_of_users,
            number_of_items=number_of_items,
            meta_data=meta_data,
            trainers=clients,
        )

        return clients, server

    method = args.method
    use_buffer = args.use_buffer
    buffer_size = args.buffer_size
    online_learning = args.online_learning
    repeat_time = args.repeat_time
    global_rounds = args.global_rounds
    local_steps = args.local_steps
    hidden_channels = args.hidden_channels
    record_results = args.record_results
    country_codes = args.country_codes
    monitor = Monitor()
    dataset_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), args.dataset_path
    )
    global_file_path = os.path.join(dataset_path, "data_global.txt")
    traveled_file_path = os.path.join(dataset_path, "traveled_users.txt")

    # check the validity of the input
    assert method in ["STFL", "StaticGNN", "4D-FED-GNN+", "FedLink"], "Invalid method."
    assert all(
        code in ["US", "BR", "ID", "TR", "JP"] for code in country_codes
    ), "The country codes should be in 'US', 'BR', 'ID', 'TR', 'JP'"
    if use_buffer:
        assert buffer_size > 0, "The buffer size should be greater than 0."

    check_data_files_existance(country_codes, dataset_path)

    # get global user and item mapping
    user_id_mapping, item_id_mapping = get_global_user_item_mapping(
        global_file_path=global_file_path
    )

    # set meta_data
    meta_data = (
        ["user", "item"],
        [("user", "select", "item"), ("item", "rev_select", "user")],
    )

    # repeat the training process
    for current_training_process in range(repeat_time):
        number_of_clients = len(country_codes)  # each country is a client
        clients, server = setup_trainer_server(
            country_codes=country_codes,
            user_id_mapping=user_id_mapping,
            item_id_mapping=item_id_mapping,
            meta_data=meta_data,
            hidden_channels=hidden_channels,
        )

        """Broadcast the global model parameter to all clients"""
        monitor.pretrain_time_start()
        global_model_parameter = (
            server.get_model_parameter()
        )  # fetch the global model parameter
        for i in range(number_of_clients):
            clients[i].set_model_parameter.remote(
                global_model_parameter
            )  # broadcast the global model parameter to all clients

        """Determine the start and end time of the conditional information"""
        (
            start_time,
            end_time,
            prediction_days,
            start_time_float_format,
            end_time_float_format,
        ) = get_start_end_time(online_learning=online_learning, method=method)

        if record_results:
            file_name = f"{method}_buffer_{use_buffer}_{buffer_size}_online_{online_learning}.txt"
            result_writer = open(file_name, "a+")
            time_writer = open("train_time_" + file_name, "a+")
        else:
            result_writer = None
            time_writer = None
        monitor.pretrain_time_end()
        # from 2012-04-03 to 2012-04-13
        for day in range(prediction_days):  # make predictions for each day
            # get the train and test data for each client at the current time step
            for i in range(number_of_clients):
                clients[i].get_train_test_data_at_current_time_step.remote(
                    start_time_float_format,
                    end_time_float_format,
                    use_buffer=use_buffer,
                    buffer_size=buffer_size,
                )
                clients[i].calculate_traveled_user_edge_indices.remote(
                    file_path=traveled_file_path
                )

            if online_learning:
                print(f"start training for day {day + 1}")
            else:
                print(f"start training")

            for iteration in range(global_rounds):
                # each client train on local graph
                print(f"global rounds: {iteration}")
                monitor.train_time_start()
                current_loss = LP_train_global_round(
                    server=server,
                    local_steps=local_steps,
                    use_buffer=use_buffer,
                    method=method,
                    online_learning=online_learning,
                    prediction_day=day,
                    curr_iteration=iteration,
                    global_rounds=global_rounds,
                    record_results=record_results,
                    result_writer=result_writer,
                    time_writer=time_writer,
                )
                monitor.train_time_end()

            if current_loss >= 0.01:
                print("training is not complete")

            # go to next day
            (
                start_time,
                end_time,
                start_time_float_format,
                end_time_float_format,
            ) = to_next_day(start_time=start_time, end_time=end_time, method=method)

        if result_writer is not None and time_writer is not None:
            result_writer.close()
            time_writer.close()
        print(f"Training round {current_training_process} success")
        ray.shutdown()


def LP_train_global_round(
    server: Any,
    local_steps: int,
    use_buffer: bool,
    method: str,
    online_learning: bool,
    prediction_day: int,
    curr_iteration: int,
    global_rounds: int,
    record_results: bool = False,
    result_writer: Any = None,
    time_writer: Any = None,
) -> float:
    """
    This function trains the clients for a global round and updates the server model with the average of the client models.

    Parameters
    ----------
    clients : list
        List of client objects
    server : Any
        Server object
    local_steps : int
        Number of local steps
    use_buffer : bool
        Specifies whether to use buffer
    method : str
        Specifies the method
    online_learning : bool
        Specifies online learning
    prediction_day : int
        Prediction day
    curr_iteration : int
        Current iteration
    global_rounds : int
        Global rounds
    record_results : bool, optional
        Record model AUC and Running time
    result_writer : Any, optional
        File writer object
    time_writer : Any, optional
        File writer object

    Returns
    -------
    current_loss : float
        Loss of the model on the training data
    """
    if record_results:
        assert result_writer is not None and time_writer is not None

    # local training
    number_of_clients = len(server.clients)
    print(f"Training in LP_train_global_round, number of clients: {number_of_clients}")
    local_training_results = []
    for client_id in range(number_of_clients):
        # current_loss, train_finish_times
        local_training_result_ref = server.clients[client_id].train.remote(
            client_id=client_id, local_updates=local_steps, use_buffer=use_buffer
        )  # local training
        local_training_results.append(local_training_result_ref)
    while True:
        ready, left = ray.wait(local_training_results, num_returns=1, timeout=None)
        if ready:
            for t in ready:
                client_id, current_loss, train_finish_times = ray.get(t)
                print(
                    f"clientId: {client_id} current_loss: {current_loss} train_finish_times: {train_finish_times}"
                )
                if record_results:
                    for train_finish_time in train_finish_times:
                        time_writer.write(
                            f"client {str(client_id)} train time {str(train_finish_time)}\n"
                        )
                        print(
                            f"client {str(client_id)} train time {str(train_finish_time)}\n"
                        )
        local_training_results = left
        if not local_training_results:
            break

    # aggregate the parameters and broadcast to the clients
    gnn_only = True if method == "FedLink (OnlyAvgGNN)" else False
    if method != "StaticGNN":
        model_avg_parameter = server.fedavg(gnn_only)
        server.set_model_parameter(model_avg_parameter, gnn_only)
        for client_id in range(number_of_clients):
            server.clients[client_id].set_model_parameter.remote(
                model_avg_parameter, gnn_only
            )

    # test the model
    test_results = [
        server.clients[client_id].test.remote(server.clients[client_id], use_buffer)
        for client_id in range(number_of_clients)
    ]
    avg_auc, avg_hit_rate, avg_traveled_user_hit_rate = 0.0, 0.0, 0.0
    # for client_id in range(number_of_clients):
    #     auc_score, hit_rate, traveled_user_hit_rate = server.clients[client_id].test(
    #         use_buffer=use_buffer
    #     )  # local testing
    #     avg_auc += auc_score
    #     avg_hit_rate += hit_rate
    #     avg_traveled_user_hit_rate += traveled_user_hit_rate
    #     print(
    #         f"Day {prediction_day} client {client_id} auc score: {auc_score} hit rate: {
    #             hit_rate} traveled user hit rate: {traveled_user_hit_rate}"
    #     )
    #     # write final test_auc
    #     if curr_iteration + 1 == global_rounds and record_results:
    #         result_writer.write(
    #             f"Day {prediction_day} client {client_id} final auc score: {auc_score} hit rate: {
    #                 hit_rate} traveled user hit rate: {traveled_user_hit_rate}\n"
    #         )
    while test_results:
        ready, left = ray.wait(test_results, num_returns=1, timeout=None)
        if ready:
            for t in ready:
                client_id, auc_score, hit_rate, traveled_user_hit_rate = ray.get(t)
                avg_auc += auc_score
                avg_hit_rate += hit_rate
                avg_traveled_user_hit_rate += traveled_user_hit_rate
                print(
                    f"Day {prediction_day} client {client_id} auc score: {auc_score} hit rate: {hit_rate} traveled user hit rate: {traveled_user_hit_rate}"
                )
                # write final test_auc
                if curr_iteration + 1 == global_rounds and record_results:
                    result_writer.write(
                        f"Day {prediction_day} client {client_id} final auc score: {auc_score} hit rate: {hit_rate} traveled user hit rate: {traveled_user_hit_rate}\n"
                    )
                print(
                    f"Day {prediction_day} client {client_id} final auc score: {auc_score} hit rate: {hit_rate} traveled user hit rate: {traveled_user_hit_rate}\n"
                )

        test_results = left

    avg_auc /= number_of_clients
    avg_hit_rate /= number_of_clients

    if online_learning:
        print(
            f"Predict Day {prediction_day + 1} average auc score: {avg_auc} hit rate: {avg_hit_rate}"
        )
    else:
        print(f"Predict Day 20 average auc score: {avg_auc} hit rate: {avg_hit_rate}")

    return current_loss
