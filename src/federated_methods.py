import argparse
import copy
import os
import random
from pathlib import Path
from typing import Any

import attridict
import numpy as np
import pandas as pd
import ray
import torch

from src.data_process_gc import load_single_dataset
from src.gnn_models import GIN, GIN_server
from src.server_class import Server
from src.train_func import gc_avg_accuracy
from src.trainer_class import Trainer_General
from src.utils import get_1hop_feature_sum
from src.utils_gc import setup_clients, setup_server


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


def GC_Train(
    config: dict, data: Any, model_server: Any = GIN_server, model_trainer: Any = GIN
) -> None:
    """
    Entrance of the training process for graph classification.

    Parameters
    ----------
    config: dict
        Configuration.
    data: Any
        The splitted data.
    model_server: Any
        The model which the server is built on.
    model_trainer: Any
        The model which the trainer is built on.
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
    if args.save_files:
        outdir_base = args.outbase + "/" + f"{args.model}"
        outdir = os.path.join(outdir_base, f"oneDS-nonOverlap")
        if args.model in ["SelfTrain"]:
            outdir = os.path.join(outdir, f"{args.data_group}")
        elif args.model in ["FedAvg", "FedProx"]:
            outdir = os.path.join(
                outdir, f"{args.data_group}-{args.num_clients}clients"
            )
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

    #################### save statistics of data on clients ####################
    # if args.save_files and df_stats:
    #     outdir_stats = os.path.join(outdir, f"stats_train_data.csv")
    #     df_stats.to_csv(outdir_stats)
    #     print(f"The statistics of the data are written to {outdir_stats}")

    #################### setup server and clients ####################
    init_clients, _ = setup_clients(data, model_trainer, args)
    init_server = setup_server(model_server, args)
    clients = copy.deepcopy(init_clients)
    server = copy.deepcopy(init_server)

    print("\nDone setting up devices.")

    ################ choose the algorithm to run ################
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
    if args.save_files:
        outdir_result = os.path.join(outdir, f"accuracy_seed{args.seed}.csv")
        pd.DataFrame(output).to_csv(outdir_result)
        print(f"The output has been written to file: {outdir_result}")


# The following code is the implementation of different federated graph classification methods.
def run_GC_selftrain(clients: list, server: Any, local_epoch: int) -> dict:
    """
    Run the training and testing process of self-training algorithm.
    It only trains the model locally, and does not perform weights aggregation.

    Parameters
    ----------
    clients: list
        List of clients
    server: Server
        Server object
    local_epoch: int
        Number of local epochs

    Returns
    -------
    all_accs: dict
        Dictionary with training and test accuracies for each client
    """
    # all clients are initialized with the same weights
    for client in clients:
        client.update_params(server)

    all_accs = {}
    for client in clients:
        client.local_train(local_epoch=local_epoch)

        _, acc = client.local_test()
        all_accs[client.name] = [
            client.train_stats["trainingAccs"][-1],
            client.train_stats["valAccs"][-1],
            acc,
        ]
        print("  > {} done.".format(client.name))

    frame = pd.DataFrame(all_accs).T.iloc[:, [2]]
    frame.columns = ["test_acc"]
    print(frame)
    print(f"Average test accuracy: {gc_avg_accuracy(frame, clients)}")
    return frame


def run_GC_fedavg(
    clients: list,
    server: Any,
    communication_rounds: int,
    local_epoch: int,
    samp: object = None,
    frac: float = 1.0,
) -> pd.DataFrame:
    """
    Run the training and testing process of FedAvg algorithm.
    It trains the model locally, aggregates the weights to the server,
    and downloads the global model within each communication round.

    Parameters
    ----------
    clients: list
        List of clients
    server: object
        Server object
    communication_rounds: int
        Number of communication rounds
    local_epoch: int
        Number of local epochs
    samp: str
        Sampling method
    frac: float
        Fraction of clients to sample

    Returns
    -------
    frame: pd.DataFrame
        Pandas dataframe with test accuracies
    """

    for client in clients:
        client.update_params(server)  # download the global model

    if samp is None:
        frac = 1.0

    # Overall training architecture:
    # whole training => { communication rounds, communication rounds, ..., communication rounds }
    # communication rounds => { local training (#epochs) -> aggregation -> download }
    #                                |
    #                           training_stats
    for c_round in range(1, communication_rounds + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")  # print the current round every 50 rounds

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = server.random_sample_clients(clients, frac)
            # if samp = None, frac=1.0, then all clients are selected

        for client in selected_clients:  # only get weights of graphconv layers
            client.local_train(local_epoch=local_epoch)  # train the local model

        server.aggregate_weights(
            selected_clients
        )  # aggregate the weights of selected clients
        for client in selected_clients:
            client.update_params(server)  # re-download the global server

    frame = pd.DataFrame()
    for client in clients:
        _, acc = client.local_test()  # Final evaluation
        frame.loc[client.name, "test_acc"] = acc

    def highlight_max(s: pd.Series) -> list:
        is_max = s == s.max()
        return ["background-color: yellow" if v else "" for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)
    print(f"Average test accuracy: {gc_avg_accuracy(frame, clients)}")
    return frame


def run_GC_fedprox(
    clients: list,
    server: Any,
    communication_rounds: int,
    local_epoch: int,
    mu: float,
    samp: object = None,
    frac: float = 1.0,
) -> pd.DataFrame:
    """
    Run the training and testing process of FedProx algorithm.
    It trains the model locally, aggregates the weights to the server,
    and downloads the global model within each communication round.

    Parameters
    ----------
    clients: list
        List of clients
    server: object
        Server object
    communication_rounds: int
        Number of communication rounds
    local_epoch: int
        Number of local epochs
    mu: float
        Proximal term
    samp: str
        Sampling method
    frac: float
        Fraction of clients to sample

    Returns:
        Frame: pandas dataframe with test accuracies
    """
    for client in clients:
        client.update_params(server)

    if samp is None:
        frac = 1.0

    for c_round in range(1, communication_rounds + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = server.random_sample_clients(clients, frac)

        for client in selected_clients:
            client.local_train(
                local_epoch=local_epoch, train_option="prox", mu=mu
            )  # Different from FedAvg

        server.aggregate_weights(selected_clients)
        for client in selected_clients:
            client.update_params(server)

            # cache the aggregated weights for next round
            client.cache_weights()

    frame = pd.DataFrame()
    for client in clients:
        _, acc = client.local_test()
        frame.loc[client.name, "test_acc"] = acc

    def highlight_max(s: pd.Series) -> list:
        is_max = s == s.max()
        return ["background-color: yellow" if v else "" for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)
    print(f"Average test accuracy: {gc_avg_accuracy(frame, clients)}")
    return frame


def run_GC_gcfl(
    clients: list,
    server: Any,
    communication_rounds: int,
    local_epoch: int,
    EPS_1: float,
    EPS_2: float,
) -> pd.DataFrame:
    """
    Run the GCFL algorithm.
    The GCFL algorithm is a cluster-based federated learning algorithm, which aggregates the weights of the clients
    in each cluster, and dynamically splits the clusters during the training process.

    Parameters
    ----------
    clients: list
        List of clients
    server: object
        Server object
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
        np.arange(len(clients)).astype("int")
    ]  # cluster_indices: [[0, 1, ...]]
    client_clusters = [
        [clients[i] for i in idcs] for idcs in cluster_indices
    ]  # initially there is only one cluster

    acc_clients: list = []
    ############### COMMUNICATION ROUNDS ###############
    for c_round in range(1, communication_rounds + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        if c_round == 1:
            for client in clients:
                client.update_params(server)

        participating_clients = server.random_sample_clients(clients, frac=1.0)
        for client in participating_clients:
            client.local_train(
                local_epoch=local_epoch, train_option="gcfl"
            )  # local training
            client.reset_params()  # reset the gradients (discard the final gradients)

        similarities = server.compute_pairwise_similarities(clients)

        cluster_indices_new = []
        for idc in cluster_indices:  # cluster-wise checking
            max_norm = server.compute_max_update_norm(
                [clients[i] for i in idc]
            )  # DELTA_MAX
            mean_norm = server.compute_mean_update_norm(
                [clients[i] for i in idc]
            )  # DELTA_MEAN
            if (
                mean_norm < EPS_1 and max_norm > EPS_2 and len(idc) > 2 and c_round > 20
            ):  # stopping condition
                server.cache_model(idc, clients[idc[0]].W, acc_clients)
                c1, c2 = server.min_cut(similarities[idc][:, idc], idc)
                cluster_indices_new += [c1, c2]  # split the cluster into two
            else:
                cluster_indices_new += [idc]  # keep the same cluster

        cluster_indices = cluster_indices_new
        client_clusters = [
            [clients[i] for i in idcs] for idcs in cluster_indices
        ]  # initial: [[0, 1, ...]]

        server.aggregate_clusterwise(
            client_clusters
        )  # aggregate the weights of the clients in each cluster

        acc_clients = [
            client.local_test()[1] for client in clients
        ]  # get the test accuracy of each client
    ############### END OF COMMUNICATION ROUNDS ###############

    for idc in cluster_indices:
        server.cache_model(
            idc, clients[idc[0]].W, acc_clients
        )  # cache the first client's weights in each cluster
    # cluster-wise model

    results = np.zeros([len(clients), len(server.model_cache)])
    for i, (idcs, W, accs) in enumerate(server.model_cache):
        results[idcs, i] = np.array(accs)

    frame = pd.DataFrame(
        results,
        columns=["FL Model"]
        + ["Model {}".format(i) for i in range(results.shape[1] - 1)],
        index=["{}".format(clients[i].name) for i in range(results.shape[0])],
    )
    frame = pd.DataFrame(frame.max(axis=1))
    frame.columns = ["test_acc"]

    print(frame)
    print(f"Average test accuracy: {gc_avg_accuracy(frame, clients)}")
    return frame


def run_GC_gcfl_plus(
    clients: list,
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
    clients: list
        List of clients
    server: object
        Server object
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
    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

    seqs_grads: Any = {c.id: [] for c in clients}

    for client in clients:
        client.update_params(server)

    acc_clients: list = []
    for c_round in range(1, communication_rounds + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        if c_round == 1:
            for client in clients:
                client.update_params(server)

        participating_clients = server.random_sample_clients(clients, frac=1.0)

        for client in participating_clients:
            client.local_train(local_epoch=local_epoch, train_option="gcfl")
            client.reset_params()

            seqs_grads[client.id].append(client.conv_grads_norm)

        cluster_indices_new = []
        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])
            if (
                mean_norm < EPS_1
                and max_norm > EPS_2
                and len(idc) > 2
                and c_round > 20
                and all(len(value) >= seq_length for value in seqs_grads.values())
            ):
                server.cache_model(idc, clients[idc[0]].W, acc_clients)

                tmp = [seqs_grads[id][-seq_length:] for id in idc]
                dtw_distances = server.compute_pairwise_distances(tmp, standardize)
                c1, c2 = server.min_cut(np.max(dtw_distances) - dtw_distances, idc)
                cluster_indices_new += [c1, c2]

                seqs_grads = {c.id: [] for c in clients}
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

        server.aggregate_clusterwise(client_clusters)

        acc_clients = [client.local_test()[1] for client in clients]

    for idc in cluster_indices:
        server.cache_model(idc, clients[idc[0]].W, acc_clients)

    results = np.zeros([len(clients), len(server.model_cache)])
    for i, (idcs, W, accs) in enumerate(server.model_cache):
        results[idcs, i] = np.array(accs)

    frame = pd.DataFrame(
        results,
        columns=["FL Model"]
        + ["Model {}".format(i) for i in range(results.shape[1] - 1)],
        index=["{}".format(clients[i].name) for i in range(results.shape[0])],
    )

    frame = pd.DataFrame(frame.max(axis=1))
    frame.columns = ["test_acc"]
    print(frame)
    print(f"Average test accuracy: {gc_avg_accuracy(frame, clients)}")

    return frame


def run_GC_gcfl_plus_dWs(
    clients: list,
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
    clients: list
        List of clients
    server: object
        Server object
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
    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

    seqs_grads: Any = {c.id: [] for c in clients}
    for client in clients:
        client.update_params(server)

    acc_clients: list = []
    for c_round in range(1, communication_rounds + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        if c_round == 1:
            for client in clients:
                client.update_params(server)

        participating_clients = server.random_sample_clients(clients, frac=1.0)

        for client in participating_clients:
            client.local_train(local_epoch=local_epoch, train_option="gcfl")
            client.reset_params()

            seqs_grads[client.id].append(client.conv_dWs_norm)

        cluster_indices_new = []
        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])
            if (
                mean_norm < EPS_1
                and max_norm > EPS_2
                and len(idc) > 2
                and c_round > 20
                and all(len(value) >= seq_length for value in seqs_grads.values())
            ):
                server.cache_model(idc, clients[idc[0]].W, acc_clients)

                tmp = [seqs_grads[id][-seq_length:] for id in idc]
                dtw_distances = server.compute_pairwise_distances(tmp, standardize)
                c1, c2 = server.min_cut(np.max(dtw_distances) - dtw_distances, idc)
                cluster_indices_new += [c1, c2]

                seqs_grads = {c.id: [] for c in clients}
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

        server.aggregate_clusterwise(client_clusters)

        acc_clients = [client.local_test()[1] for client in clients]

    for idc in cluster_indices:
        server.cache_model(idc, clients[idc[0]].W, acc_clients)

    results = np.zeros([len(clients), len(server.model_cache)])
    for i, (idcs, W, accs) in enumerate(server.model_cache):
        results[idcs, i] = np.array(accs)

    frame = pd.DataFrame(
        results,
        columns=["FL Model"]
        + ["Model {}".format(i) for i in range(results.shape[1] - 1)],
        index=["{}".format(clients[i].name) for i in range(results.shape[0])],
    )
    frame = pd.DataFrame(frame.max(axis=1))
    frame.columns = ["test_acc"]
    print(frame)
    print(f"Average test accuracy: {gc_avg_accuracy(frame, clients)}")

    return frame
