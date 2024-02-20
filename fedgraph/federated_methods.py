from typing import Any

import numpy as np
import ray
import torch
from attrdict import AttrDict

from fedgraph.server_class import Server
from fedgraph.trainer_class import Trainer_General
from fedgraph.utils import get_1hop_feature_sum


def FedGCN_Train(args: AttrDict, data: tuple) -> None:
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
