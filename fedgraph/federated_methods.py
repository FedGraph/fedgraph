import argparse
import copy
import datetime
import os
import pickle
import random
import sys
import time
from importlib.resources import files
from pathlib import Path
from typing import Any, List, Optional

import attridict
import numpy as np
import pandas as pd
import ray
import tenseal as ts
import torch

from fedgraph.data_process import data_loader
from fedgraph.gnn_models import GIN
from fedgraph.monitor_class import Monitor
from fedgraph.server_class import Server, Server_GC, Server_LP
from fedgraph.train_func import gc_avg_accuracy
from fedgraph.trainer_class import Trainer_GC, Trainer_General, Trainer_LP
from fedgraph.utils_gc import setup_server, setup_trainers
from fedgraph.utils_lp import (
    check_data_files_existance,
    get_global_user_item_mapping,
    get_start_end_time,
    to_next_day,
)
from fedgraph.utils_nc import get_1hop_feature_sum, save_all_trainers_data


def run_fedgraph(args: attridict) -> None:
    """
    Run the training process for the specified task.

    This is the function for running different federated graph learning tasks,
    including Node Classification (NC), Graph Classification (GC), and Link Prediction (LP)
    in the following functions.

    Parameters
    ----------
    args : attridict
        Configuration arguments that must include 'fedgraph_task' key with value
        in ['NC', 'GC', 'LP'].
    data: Any
        Input data for the federated learning task. Format depends on the specific task and
        will be explained in more detail below inside specific functions.
    """
    if args.fedgraph_task != "NC" or not args.use_huggingface:
        data = data_loader(args)
    else:
        # use hugging_face instead of use data_loader
        print("Using hugging_face for local loading")
        data = None
    if args.fedgraph_task == "NC":
        run_NC(args, data)
    elif args.fedgraph_task == "GC":
        run_GC(args, data)
    elif args.fedgraph_task == "LP":
        run_LP(args)


def run_NC(args: attridict, data: Any = None) -> None:
    """
    Train a Federated Graph Classification model using multiple trainers.

    Implements FL for node classification tasks with support of homomorphic encryption.
    Use configuration argument "use_encryption" to indicate the boolean flag for
    homomorphic encryption or plaintext calculation of feature and/or gradient aggregation
    during pre-training and training. Current algorithm that supports encryption includes
    'FedAvg' and 'FedGCN'.

    Parameters
    ----------
    args: attridict
        Configuration arguments
    data: tuple
    """
    ray.init()
    start_time = time.time()
    if args.use_cluster:
        monitor = Monitor()
    torch.manual_seed(42)
    if args.num_hops == 0:
        print("Changing method to FedAvg")
        args.method = "FedAvg"
    if not args.use_huggingface:
        (
            edge_index,
            features,
            labels,
            idx_train,
            idx_test,
            class_num,
            split_node_indexes,
            communicate_node_global_indexes,
            in_com_train_node_local_indexes,
            in_com_test_node_local_indexes,
            global_edge_indexes_clients,
        ) = data
        if args.saveto_huggingface:
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

    if args.dataset in ["simulate", "cora", "citeseer", "pubmed", "reddit"]:
        args_hidden = 16
    else:
        args_hidden = 256

    num_cpus_per_trainer = args.num_cpus_per_trainer
    # specifying a target GPU
    if args.gpu:
        device = torch.device("cuda")
        num_gpus_per_trainer = args.num_gpus_per_trainer
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
            self.use_encryption = kwds["args"].use_encryption
            if self.use_encryption:
                file_path = str(files("fedgraph").joinpath("he_context.pkl"))
                with open(file_path, "rb") as f:
                    context_bytes = pickle.load(f)
                self.he_context = ts.context_from(context_bytes)
                print(f"Trainer {self.rank} loaded HE context")

    if args.use_huggingface:
        trainers = [
            Trainer.remote(  # type: ignore
                rank=i,
                args_hidden=args_hidden,
                device=device,
                args=args,
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

    server = Server(features.shape[1], args_hidden, class_num, device, trainers, args)
    server.broadcast_params(-1)
    pretrain_start = time.time()
    if args.use_cluster:
        monitor.pretrain_time_start()
    if args.method != "FedAvg":
        #######################################################################
        # Pre-Train Communication of FedGCN
        # ---------------------------------
        # Clients send their local feature sum to the server, and the server
        # aggregates all local feature sums and send the global feature sum
        # of specific nodes back to each trainer.
        if args.use_encryption:
            print("Starting encrypted feature aggregation...")

            encrypted_data = [
                trainer.get_encrypted_local_feature_sum.remote()
                for trainer in server.trainers
            ]

            results = ray.get(encrypted_data)
            encrypted_sums = [(r[0], r[1]) for r in results]  # (encrypted_sum, shape)
            encryption_times = [r[2] for r in results]

            enc_sizes = [len(r[0]) for r in results]  # size of encrypted data

            # aggregate at server
            (
                aggregated_result,
                aggregation_time,
            ) = server.aggregate_encrypted_feature_sums(encrypted_sums)
            agg_size = len(aggregated_result[0])

            load_feature_refs = [
                trainer.load_encrypted_feature_aggregation.remote(aggregated_result)
                for trainer in server.trainers
            ]
            decryption_times = ray.get(load_feature_refs)
            pretrain_time = time.time() - pretrain_start
            pretrain_upload = sum(enc_sizes) / (1024 * 1024)  # MB
            pretrain_download = agg_size * len(server.trainers) / (1024 * 1024)  # MB
            pretrain_comm_cost = pretrain_upload + pretrain_download

            # print performance metrics
            print("\nPre-training Phase Metrics:")
            print(f"Total Pre-training Time: {pretrain_time:.2f} seconds")
            print(f"Pre-training Upload: {pretrain_upload:.2f} MB")
            print(f"Pre-training Download: {pretrain_download:.2f} MB")
            print(f"Total Pre-training Communication Cost: {pretrain_comm_cost:.2f} MB")

        else:
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
                    global_feature_sum
                    != get_1hop_feature_sum(features, edge_index, device)
                ).sum() == 0
            for i in range(args.n_trainer):
                communicate_nodes = (
                    communicate_node_global_indexes[i].clone().detach().to(device)
                )
                trainer_aggregation = global_feature_sum[communicate_nodes]
                server.trainers[i].load_feature_aggregation.remote(trainer_aggregation)
            print("clients received feature aggregation from server")
        [trainer.relabel_adj.remote() for trainer in server.trainers]
    if args.use_cluster:
        monitor.pretrain_time_end(30)
        monitor.train_time_start()
    #######################################################################
    # Federated Training
    # ------------------
    # The server start training of all trainers and aggregate the parameters
    # at every global round.
    training_start = time.time()
    print("global_rounds", args.global_rounds)
    for i in range(args.global_rounds):
        server.train(i)
    if args.use_cluster:
        monitor.train_time_end(30)
    training_time = time.time() - training_start
    if args.use_encryption:
        if hasattr(server, "aggregation_stats") and server.aggregation_stats:
            training_upload = sum(
                [r["upload_size"] for r in server.aggregation_stats]
            ) / (
                1024 * 1024
            )  # MB
            training_download = sum(
                [r["download_size"] for r in server.aggregation_stats]
            ) / (
                1024 * 1024
            )  # MB
        else:
            training_upload = training_download = 0
        training_comm_cost = training_upload + training_download

        print("\nTraining Phase Metrics:")
        print(f"Total Training Time: {training_time:.2f} seconds")
        print(f"Training Upload: {training_upload:.2f} MB")
        print(f"Training Download: {training_download:.2f} MB")
        print(f"Total Training Communication Cost: {training_comm_cost:.2f} MB")

        # Overall totals
        total_time = time.time() - start_time
        total_upload = pretrain_upload + training_upload
        total_download = pretrain_download + training_download
        total_comm_cost = total_upload + total_download

        print("\nOverall Totals:")
        print(f"Total Execution Time: {total_time:.2f} seconds")
        print(f"Total Upload: {total_upload:.2f} MB")
        print(f"Total Download: {total_download:.2f} MB")
        print(f"Total Communication Cost: {total_comm_cost:.2f} MB")
        print(f"Pre-training Time %: {(pretrain_time/total_time)*100:.1f}%")
        print(f"Training Time %: {(training_time/total_time)*100:.1f}%")
        print(f"Pre-training Comm %: {(pretrain_comm_cost/total_comm_cost)*100:.1f}%")
        print(f"Training Comm %: {(training_comm_cost/total_comm_cost)*100:.1f}%")
    #######################################################################
    # Summarize Experiment Results
    # ----------------------------
    # The server collects the local test loss and accuracy from all trainers
    # then calculate the overall test loss and accuracy.

    # train_data_weights = [len(i) for i in in_com_train_node_indexes]
    # test_data_weights = [len(i) for i in in_com_test_node_indexes]

    results = [trainer.local_test.remote() for trainer in server.trainers]
    results = np.array([ray.get(result) for result in results])

    average_final_test_loss = np.average(
        [row[0] for row in results], weights=test_data_weights, axis=0
    )
    average_final_test_accuracy = np.average(
        [row[1] for row in results], weights=test_data_weights, axis=0
    )
    print(f"average_final_test_loss, {average_final_test_loss}")
    print(f"Average test accuracy, {average_final_test_accuracy}")
    ray.shutdown()


def run_GC(args: attridict, data: Any, base_model: Any = GIN) -> None:
    """
    Entrance of the training process for graph classification.

    Supports multiple federated learning algorithms including FedAvg, FedProx, GCFL,
    GCFL+, and GCFL+dWs. Implements client-server architecture with Ray for distributed
    computing.

    Parameters
    ----------
    args: attridict
        The configuration arguments.
    data: Any
        Dictionary mapping dataset names to their respective graph data including
        dataloaders, number of node features, number of graph labels, and train size
    base_model: Any
        The base model on which the federated learning is based. It applies for both the server and the trainers (default: GIN).
    """
    # transfer the config to argparse

    #################### set seeds and devices ####################
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(current_dir, "../fedgraph"))
    sys.path.append(os.path.join(current_dir, "../../"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    base_model = GIN
    num_cpus_per_trainer = args.num_cpus_per_trainer
    # specifying a target GPU
    if args.gpu:
        print("using GPU")
        args.device = torch.device("cuda")
        num_gpus_per_trainer = args.num_gpus_per_trainer
    else:
        print("using CPU")
        args.device = torch.device("cpu")
        num_gpus_per_trainer = 0

    #################### set output directory ####################
    # outdir_base = os.path.join(args.outbase, f'seqLen{args.seq_length}')
    if args.save_files:
        outdir_base = args.outbase + "/" + f"{args.algorithm}"
        outdir = os.path.join(outdir_base, f"oneDS-nonOverlap")
        if args.algorithm in ["SelfTrain"]:
            outdir = os.path.join(outdir, f"{args.dataset}")
        elif args.algorithm in ["FedAvg", "FedProx"]:
            outdir = os.path.join(outdir, f"{args.dataset}-{args.num_trainers}trainers")
        elif args.algorithm in ["GCFL"]:
            outdir = os.path.join(
                outdir,
                f"{args.dataset}-{args.num_trainers}trainers",
                f"eps_{args.epsilon1}_{args.epsilon2}",
            )
        elif args.algorithm in ["GCFL+", "GCFL+dWs"]:
            outdir = os.path.join(
                outdir,
                f"{args.dataset}-{args.num_trainers}trainers",
                f"eps_{args.epsilon1}_{args.epsilon2}",
                f"seqLen{args.seq_length}",
            )

        Path(outdir).mkdir(parents=True, exist_ok=True)
        print(f"Output Path: {outdir}")
    #################### save statistics of data on trainers ####################
    # if args.save_files and df_stats:
    #     outdir_stats = os.path.join(outdir, f"stats_train_data.csv")
    #     df_stats.to_csv(outdir_stats)
    #     print(f"The statistics of the data are written to {outdir_stats}")

    #################### setup server and trainers ####################
    ray.init()

    @ray.remote(
        num_gpus=num_gpus_per_trainer,
        num_cpus=num_cpus_per_trainer,
        scheduling_strategy="SPREAD",
    )
    class Trainer(Trainer_GC):
        def __init__(self, idx, splited_data, dataset_trainer_name, cmodel_gc, args):  # type: ignore
            print(f"inx: {idx}")
            print(f"dataset_trainer_name: {dataset_trainer_name}")
            """acquire data"""
            dataloaders, num_node_features, num_graph_labels, train_size = splited_data

            print(f"dataloaders: {dataloaders}")
            print(f"num_node_features: {num_node_features}")
            print(f"num_graph_labels: {num_graph_labels}")
            print(f"train_size: {train_size}")

            """build optimizer"""
            optimizer = torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, cmodel_gc.parameters()),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            super().__init__(  # type: ignore
                model=cmodel_gc,
                trainer_id=idx,
                trainer_name=dataset_trainer_name,
                train_size=train_size,
                dataloader=dataloaders,
                optimizer=optimizer,
                args=args,
            )

    trainers = [
        Trainer.remote(  # type: ignore
            idx=idx,
            splited_data=data[dataset_trainer_name],
            dataset_trainer_name=dataset_trainer_name,
            # "GIN model for GC",
            cmodel_gc=base_model(
                nfeat=data[dataset_trainer_name][1],
                nhid=args.hidden,
                nclass=data[dataset_trainer_name][2],
                nlayer=args.nlayer,
                dropout=args.dropout,
            ),
            args=args,
        )
        for idx, dataset_trainer_name in enumerate(data.keys())
    ]
    server = Server_GC(
        base_model(nlayer=args.nlayer, nhid=args.hidden), args.device, args.use_cluster
    )
    # TODO: check and modify whether deepcopy should be added.
    # trainers = copy.deepcopy(init_trainers)
    # server = copy.deepcopy(init_server)

    print("\nDone setting up devices.")

    ################ choose the algorithm to run ################
    print(f"Running {args.algorithm} ...")

    model_parameters = {
        "SelfTrain": lambda: run_GC_selftrain(
            trainers=trainers, server=server, local_epoch=args.local_epoch
        ),
        "FedAvg": lambda: run_GC_Fed_algorithm(
            trainers=trainers,
            server=server,
            communication_rounds=args.num_rounds,
            local_epoch=args.local_epoch,
            algorithm="FedAvg",
        ),
        "FedProx": lambda: run_GC_Fed_algorithm(
            trainers=trainers,
            server=server,
            communication_rounds=args.num_rounds,
            local_epoch=args.local_epoch,
            algorithm="FedProx",
            mu=args.mu,
        ),
        "GCFL": lambda: run_GCFL_algorithm(
            trainers=trainers,
            server=server,
            communication_rounds=args.num_rounds,
            local_epoch=args.local_epoch,
            EPS_1=args.epsilon1,
            EPS_2=args.epsilon2,
            algorithm_type="gcfl",
        ),
        "GCFL+": lambda: run_GCFL_algorithm(
            trainers=trainers,
            server=server,
            communication_rounds=args.num_rounds,
            local_epoch=args.local_epoch,
            EPS_1=args.epsilon1,
            EPS_2=args.epsilon2,
            algorithm_type="gcfl_plus",
            seq_length=args.seq_length,
            standardize=args.standardize,
        ),
        "GCFL+dWs": lambda: run_GCFL_algorithm(
            trainers=trainers,
            server=server,
            communication_rounds=args.num_rounds,
            local_epoch=args.local_epoch,
            EPS_1=args.epsilon1,
            EPS_2=args.epsilon2,
            algorithm_type="gcfl_plus_dWs",
            seq_length=args.seq_length,
            standardize=args.standardize,
        ),
    }

    if args.algorithm in model_parameters:
        output = model_parameters[args.algorithm]()
    else:
        raise ValueError(f"Unknown model: {args.algorithm}")

    #################### save the output ####################
    if args.save_files:
        outdir_result = os.path.join(outdir, f"accuracy_seed{args.seed}.csv")
        pd.DataFrame(output).to_csv(outdir_result)
        print(f"The output has been written to file: {outdir_result}")
    ray.shutdown()


# The following code is the implementation of different federated graph classification methods.
def run_GC_selftrain(trainers: list, server: Any, local_epoch: int) -> dict:
    """
    Run the training and testing process of self-training algorithm.
    It only trains the model locally, and does not perform weights aggregation.
    It is useful as a baseline comparison for federated methods.

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
    if server.use_cluster:
        monitor = Monitor()
        monitor.pretrain_time_start()
    global_params_id = ray.put(server.W)
    for trainer in trainers:
        trainer.update_params.remote(global_params_id)
    if server.use_cluster:
        monitor.pretrain_time_end(30)
    all_accs = {}
    acc_refs = []
    if server.use_cluster:
        monitor.train_time_start()
    for trainer in trainers:
        trainer.local_train.remote(local_epoch=local_epoch)
        acc_ref = trainer.local_test.remote()
        acc_refs.append(acc_ref)
    while True:
        ready, left = ray.wait(acc_refs, num_returns=1, timeout=None)
        if ready:
            for t in ready:
                _, acc, trainer_name, trainingaccs, valaccs = ray.get(t)
                all_accs[trainer_name] = [
                    trainingaccs,
                    valaccs,
                    acc,
                ]
                print("  > {} done.".format(trainer_name))
                print(f"trainingaccs: {trainingaccs}, valaccs: {valaccs}, acc: {acc}")
        acc_refs = left
        if not acc_refs:
            break
    if server.use_cluster:
        monitor.train_time_end(30)
    frame = pd.DataFrame(all_accs).T.iloc[:, [2]]
    frame.columns = ["test_acc"]
    print(frame)
    # TODO: delete to make speed faster
    print(f"Average test accuracy: {gc_avg_accuracy(frame, trainers)}")
    return frame


def run_GC_Fed_algorithm(
    trainers: list,
    server: Any,
    communication_rounds: int,
    local_epoch: int,
    algorithm: str,
    mu: float = 0.0,
    sampling_frac: float = 1.0,
) -> pd.DataFrame:
    """
    Run the training and testing process of FedAvg or FedProx algorithm.
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
    algorithm: str
        Algorithm to run, either 'FedAvg' or 'FedProx'
    mu: float, optional
        Proximal term for FedProx (default is 0.0)
    sampling_frac: float, optional
        Fraction of trainers to sample (default is 1.0)

    Returns
    -------
    frame: pd.DataFrame
        Pandas dataframe with test accuracies
    """
    if server.use_cluster:
        monitor = Monitor()
        monitor.pretrain_time_start()
    global_params_id = ray.put(server.W)
    for trainer in trainers:
        trainer.update_params.remote(global_params_id)
    if server.use_cluster:
        monitor.pretrain_time_end(30)
        monitor.train_time_start()
    for c_round in range(1, communication_rounds + 1):
        if (c_round) % 10 == 0:
            # print the current round every 10 rounds
            print(f"  > Training round {c_round} finished.")

        if c_round == 1:
            selected_trainers = trainers
        else:
            selected_trainers = server.random_sample_trainers(trainers, sampling_frac)

        for trainer in selected_trainers:
            if algorithm == "FedAvg":
                trainer.local_train.remote(local_epoch=local_epoch)
            elif algorithm == "FedProx":
                trainer.local_train.remote(
                    local_epoch=local_epoch, train_option="prox", mu=mu
                )
            else:
                raise ValueError(
                    "Invalid algorithm. Choose either 'FedAvg' or 'FedProx'."
                )

        server.aggregate_weights(selected_trainers)
        ray.internal.free([global_params_id])  # Free the old weight memory
        global_params_id = ray.put(server.W)
        for trainer in selected_trainers:
            trainer.update_params.remote(global_params_id)
            if algorithm == "FedProx":
                trainer.cache_weights.remote()

    frame = pd.DataFrame()
    acc_refs = []
    for trainer in trainers:
        acc_ref = trainer.local_test.remote()
        acc_refs.append(acc_ref)
    while acc_refs:
        ready, left = ray.wait(acc_refs, num_returns=1, timeout=None)
        if ready:
            for t in ready:
                _, acc, trainer_name, trainingaccs, valaccs = ray.get(t)
                frame.loc[trainer_name, "test_acc"] = acc
        acc_refs = left

    def highlight_max(s: pd.Series) -> list:
        is_max = s == s.max()
        return ["background-color: yellow" if v else "" for v in is_max]

    if server.use_cluster:
        monitor.train_time_end(30)
    fs = frame.style.apply(highlight_max).data
    print(fs)
    print(f"Average test accuracy: {gc_avg_accuracy(frame, trainers)}")
    return frame


def run_GCFL_algorithm(
    trainers: list,
    server: Any,
    communication_rounds: int,
    local_epoch: int,
    EPS_1: float,
    EPS_2: float,
    algorithm_type: str,
    seq_length: int = 0,
    standardize: bool = True,
) -> pd.DataFrame:
    """
    Run the specified GCFL algorithm.

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
    algorithm_type: str
        Type of algorithm ('gcfl', 'gcfl_plus', 'gcfl_plus_dWs')
    seq_length: int, optional
        The length of the gradient norm sequence, required for 'gcfl_plus' and 'gcfl_plus_dWs'
    standardize: bool, optional
        Whether to standardize the distance matrix, required for 'gcfl_plus' and 'gcfl_plus_dWs'

    Returns
    -------
    frame: pandas.DataFrame
        Pandas dataframe with test accuracies
    """
    if algorithm_type not in ["gcfl", "gcfl_plus", "gcfl_plus_dWs"]:
        raise ValueError(
            "Invalid algorithm_type. Must be 'gcfl', 'gcfl_plus', or 'gcfl_plus_dWs'."
        )
    if server.use_cluster:
        monitor = Monitor()
        monitor.pretrain_time_start()
    cluster_indices = [np.arange(len(trainers)).astype("int")]
    trainer_clusters = [[trainers[i] for i in idcs] for idcs in cluster_indices]

    global_params_id = ray.put(server.W)
    if algorithm_type in ["gcfl_plus", "gcfl_plus_dWs"]:
        seqs_grads: Any = {ray.get(c.get_id.remote()): [] for c in trainers}

        # Perform update_params before communication rounds for GCFL+ and GCFL+ dWs

        for trainer in trainers:
            trainer.update_params.remote(global_params_id)
    if server.use_cluster:
        monitor.pretrain_time_end(30)
    acc_trainers: List[Any] = []
    if server.use_cluster:
        monitor.train_time_start()
    for c_round in range(1, communication_rounds + 1):
        if (c_round) % 10 == 0:
            print(f"  > Training round {c_round} finished.")

        if c_round == 1:
            # Perform update_params at the beginning of the first communication round
            # ray.internal.free(
            #     [global_params_id]
            # )  # Free the old weight memory in object store
            global_params_id = ray.put(server.W)
            for trainer in trainers:
                trainer.update_params.remote(global_params_id)
        reset_params_refs = []
        participating_trainers = server.random_sample_trainers(trainers, frac=1.0)
        for trainer in participating_trainers:
            trainer.local_train.remote(local_epoch=local_epoch, train_option="gcfl")
            reset_params_ref = trainer.reset_params.remote()
            reset_params_refs.append(reset_params_ref)
        ray.get(reset_params_refs)
        for trainer in participating_trainers:
            if algorithm_type == "gcfl_plus":
                seqs_grads[ray.get(trainer.get_id.remote())].append(
                    ray.get(trainer.get_conv_grads_norm.remote())
                )
            elif algorithm_type == "gcfl_plus_dWs":
                seqs_grads[ray.get(trainer.get_id.remote())].append(
                    ray.get(trainer.get_conv_dWs_norm.remote())
                )

        cluster_indices_new = []
        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([trainers[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([trainers[i] for i in idc])

            if mean_norm < EPS_1 and max_norm > EPS_2 and len(idc) > 2 and c_round > 20:
                # marginal condition for gcfl, gcfl+, gcfl+dws
                if algorithm_type == "gcfl" or all(
                    len(value) >= seq_length for value in seqs_grads.values()
                ):
                    server.cache_model(
                        idc,
                        ray.get(trainers[idc[0]].get_total_weight.remote()),
                        acc_trainers,
                    )
                    if algorithm_type == "gcfl":
                        c1, c2 = server.min_cut(
                            server.compute_pairwise_similarities(trainers)[idc][:, idc],
                            idc,
                        )
                        cluster_indices_new += [c1, c2]

                    else:  # gcfl+, gcfl+dws
                        tmp = [seqs_grads[id][-seq_length:] for id in idc]
                        dtw_distances = server.compute_pairwise_distances(
                            tmp, standardize
                        )
                        c1, c2 = server.min_cut(
                            np.max(dtw_distances) - dtw_distances, idc
                        )
                        cluster_indices_new += [c1, c2]
                        seqs_grads = {ray.get(c.get_id.remote()): [] for c in trainers}
                else:
                    cluster_indices_new += [idc]
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new

        trainer_clusters = [[trainers[i] for i in idcs] for idcs in cluster_indices]
        server.aggregate_clusterwise(trainer_clusters)

        acc_trainers = []
        acc_trainers_refs = [trainer.local_test.remote() for trainer in trainers]

        # Collect the model parameters as they become ready
        while acc_trainers_refs:
            ready, left = ray.wait(acc_trainers_refs, num_returns=1, timeout=None)
            if ready:
                for t in ready:
                    acc_trainers.append(ray.get(t)[1])
            acc_trainers_refs = left

    for idc in cluster_indices:
        server.cache_model(
            idc, ray.get(trainers[idc[0]].get_total_weight.remote()), acc_trainers
        )
    if server.use_cluster:
        monitor.train_time_end(30)
    results = np.zeros([len(trainers), len(server.model_cache)])
    for i, (idcs, W, accs) in enumerate(server.model_cache):
        results[idcs, i] = np.array(accs)

    frame = pd.DataFrame(
        results,
        columns=["FL Model"]
        + ["Model {}".format(i) for i in range(results.shape[1] - 1)],
        index=[
            "{}".format(ray.get(trainers[i].get_name.remote()))
            for i in range(results.shape[0])
        ],
    )
    frame = pd.DataFrame(frame.max(axis=1))
    frame.columns = ["test_acc"]
    print(frame)
    print(f"Average test accuracy: {gc_avg_accuracy(frame, trainers)}")

    return frame


def run_LP(args: attridict) -> None:
    """
    Implements various federated learning methods for link prediction tasks with support
    for online learning and buffer mechanisms. Handles temporal aspects of link prediction
    and cross-region user interactions.

    Algorithm choices include ('STFL', 'StaticGNN', '4D-FED-GNN+', 'FedLink').

    Parameters
    ----------
    args: attridict
        The configuration arguments.
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

        number_of_clients = len(country_codes)
        number_of_users, number_of_items = len(user_id_mapping.keys()), len(
            item_id_mapping.keys()
        )
        num_cpus_per_client = args.num_cpus_per_trainer
        if args.gpu == True:
            device = torch.device("cuda")
            print("gpu detected")
            num_gpus_per_client = args.num_gpus_per_trainer
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
                dataset_path=args.dataset_path,
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
    global_rounds = args.global_rounds
    local_steps = args.local_steps
    hidden_channels = args.hidden_channels
    record_results = args.record_results
    country_codes = args.country_codes

    current_dir = os.path.dirname(os.path.abspath(__file__))
    ray.init()
    # Append paths relative to the current script's directory
    sys.path.append(os.path.join(current_dir, "../fedgraph"))
    sys.path.append(os.path.join(current_dir, "../../"))
    dataset_path = args.dataset_path
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
    number_of_clients = len(country_codes)  # each country is a client
    clients, server = setup_trainer_server(
        country_codes=country_codes,
        user_id_mapping=user_id_mapping,
        item_id_mapping=item_id_mapping,
        meta_data=meta_data,
        hidden_channels=hidden_channels,
    )

    """Broadcast the global model parameter to all clients"""
    if args.use_cluster:
        monitor = Monitor()
        monitor.pretrain_time_start()
    global_model_parameter = (
        server.get_model_parameter()
    )  # fetch the global model parameter
    # TODO: add memory optimization here by move ref to shared raylet
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
        file_name = (
            f"{method}_buffer_{use_buffer}_{buffer_size}_online_{online_learning}.txt"
        )
        result_writer = open(file_name, "a+")
        time_writer = open("train_time_" + file_name, "a+")
    else:
        result_writer = None
        time_writer = None
    if args.use_cluster:
        monitor.pretrain_time_end(30)
        monitor.train_time_start()
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

        if current_loss >= 0.01:
            print("training is not complete")

        # go to next day
        (
            start_time,
            end_time,
            start_time_float_format,
            end_time_float_format,
        ) = to_next_day(start_time=start_time, end_time=end_time, method=method)
    if args.use_cluster:
        monitor.train_time_end(30)
    if result_writer is not None and time_writer is not None:
        result_writer.close()
        time_writer.close()
    print("The whole process has ended")
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
    This function trains the clients for a global round, handles model aggregation,
    updates the server model with the average of the client models, and and evaluates
    performance metrics including AUC scores and hit rates.
    Supports different training methods.

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
