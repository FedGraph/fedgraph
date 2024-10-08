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
import pickle as pkl
import sys
import time
from datetime import datetime
from io import BytesIO
from typing import Any

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import ray
import scipy.sparse as sp
import torch
import torch_geometric
import torch_sparse
from huggingface_hub import HfApi, HfFolder, hf_hub_download, upload_file
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from fedgraph.gnn_models import GCN, GIN, GNN_LP, AggreGCN, GCN_arxiv, SAGE_products
from fedgraph.monitor_class import Monitor
from fedgraph.server_class import Server
from fedgraph.train_func import test, train
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

parser.add_argument("-n", "--n_trainer", default=2, type=int)
parser.add_argument("-nl", "--num_layers", default=2, type=int)
parser.add_argument("-nhop", "--num_hops", default=1, type=int)
parser.add_argument("-g", "--gpu", action="store_true")  # if -g, use gpu
parser.add_argument("-iid_b", "--iid_beta", default=1.0, type=float)

parser.add_argument("-l", "--logdir", default="./runs", type=str)

args = parser.parse_args()


def FedGCN_parse_index_file(filename: str) -> list:
    """
    Reads and parses an index file

    Parameters
    ----------
    filename : str
        Name or path of the file to parse.

    Returns
    -------
    index : list
        List of integers, each integer in the list represents int of the lines of the input file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_trainer_data_from_hugging_face(trainer_id):
    repo_name = f"FedGraph/fedgraph_{args.dataset}_{args.n_trainer}trainer_{args.num_hops}hop_iid_beta_{args.iid_beta}_trainer_id_{trainer_id}"

    def download_and_load_tensor(file_name):
        file_path = hf_hub_download(
            repo_id=repo_name, repo_type="dataset", filename=file_name
        )
        with open(file_path, "rb") as f:
            buffer = BytesIO(f.read())
            tensor = torch.load(buffer)
        print(f"Loaded {file_name}, size: {tensor.size()}")
        return tensor

    print(f"Loading client data {trainer_id}")
    local_node_index = download_and_load_tensor("local_node_index.pt")
    communicate_node_index = download_and_load_tensor("communicate_node_index.pt")
    adj = download_and_load_tensor("adj.pt")
    train_labels = download_and_load_tensor("train_labels.pt")
    test_labels = download_and_load_tensor("test_labels.pt")
    features = download_and_load_tensor("features.pt")
    idx_train = download_and_load_tensor("idx_train.pt")
    idx_test = download_and_load_tensor("idx_test.pt")

    return (
        local_node_index,
        communicate_node_index,
        adj,
        train_labels,
        test_labels,
        features,
        idx_train,
        idx_test,
    )


def save_trainer_data_to_hugging_face(
    trainer_id,
    local_node_index,
    communicate_node_index,
    adj,
    train_labels,
    test_labels,
    features,
    idx_train,
    idx_test,
):
    repo_name = f"FedGraph/fedgraph_{args.dataset}_{args.n_trainer}trainer_{args.num_hops}hop_iid_beta_{args.iid_beta}_trainer_id_{trainer_id}"
    user = HfFolder.get_token()

    api = HfApi()
    try:
        api.create_repo(
            repo_id=repo_name, token=user, repo_type="dataset", exist_ok=True
        )
    except Exception as e:
        print(f"Failed to create or access the repository: {str(e)}")
        return

    def save_tensor_to_hf(tensor, file_name):
        buffer = BytesIO()
        torch.save(tensor, buffer)
        buffer.seek(0)
        api.upload_file(
            path_or_fileobj=buffer,
            path_in_repo=file_name,
            repo_id=repo_name,
            repo_type="dataset",
            token=user,
        )

    save_tensor_to_hf(local_node_index, "local_node_index.pt")
    save_tensor_to_hf(communicate_node_index, "communicate_node_index.pt")
    save_tensor_to_hf(adj, "adj.pt")
    save_tensor_to_hf(train_labels, "train_labels.pt")
    save_tensor_to_hf(test_labels, "test_labels.pt")
    save_tensor_to_hf(features, "features.pt")
    save_tensor_to_hf(idx_train, "idx_train.pt")
    save_tensor_to_hf(idx_test, "idx_test.pt")

    print(f"Uploaded data for trainer {trainer_id}")


def save_all_trainers_data(
    split_node_indexes,
    communicate_node_indexes,
    edge_indexes_clients,
    labels,
    features,
    in_com_train_node_indexes,
    in_com_test_node_indexes,
    n_trainer,
):
    for i in range(n_trainer):
        save_trainer_data_to_hugging_face(
            trainer_id=i,
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
        )


class Trainer_General:
    """
    A general trainer class for training GCN in a federated learning setup, which includes functionalities
    required for training GCN models on a subset of a distributed dataset, handling local training and testing,
    parameter updates, and feature aggregation.

    Parameters
    ----------
    rank : int
        Unique identifier for the training instance (typically representing a trainer in federated learning).
    local_node_index : torch.Tensor
        Indices of nodes local to this trainer.
    communicate_node_index : torch.Tensor
        Indices of nodes that participate in communication during training.
    adj : torch.Tensor
        The adjacency matrix representing the graph structure.
    train_labels : torch.Tensor
        Labels of the training data.
    test_labels : torch.Tensor
        Labels of the testing data.
    features : torch.Tensor
        Node features for the entire graph.
    idx_train : torch.Tensor
        Indices of training nodes.
    idx_test : torch.Tensor
        Indices of test nodes.
    args_hidden : int
        Number of hidden units in the GCN model.
    global_node_num : int
        Total number of nodes in the global graph.
    class_num : int
        Number of classes for classification.
    device : torch.device
        The device (CPU or GPU) on which the model will be trained.
    args : Any
        Additional arguments required for model initialization and training.
    """

    def __init__(
        self,
        rank: int,
        args_hidden: int,
        global_node_num: int,
        class_num: int,
        device: torch.device,
        args: Any,
    ):
        # from gnn_models import GCN_Graph_Classification
        torch.manual_seed(rank)
        (
            local_node_index,
            communicate_node_index,
            adj,
            train_labels,
            test_labels,
            features,
            idx_train,
            idx_test,
        ) = load_trainer_data_from_hugging_face(rank)

        # seems that new trainer process will not inherit sys.path from parent, need to reimport!
        if args.num_hops >= 1 and args.method == "fedgcn":
            self.model = AggreGCN(
                nfeat=features.shape[1],
                nhid=args_hidden,
                nclass=class_num,
                dropout=0.5,
                NumLayers=args.num_layers,
            ).to(device)
        else:
            if args.dataset == "ogbn-arxiv":
                self.model = GCN_arxiv(
                    nfeat=features.shape[1],
                    nhid=args_hidden,
                    nclass=class_num,
                    dropout=0.5,
                    NumLayers=args.num_layers,
                ).to(device)
            elif args.dataset == "ogbn-products":
                self.model = SAGE_products(
                    nfeat=features.shape[1],
                    nhid=args_hidden,
                    nclass=class_num,
                    dropout=0.5,
                    NumLayers=args.num_layers,
                ).to(device)
            else:
                self.model = GCN(
                    nfeat=features.shape[1],
                    nhid=args_hidden,
                    nclass=class_num,
                    dropout=0.5,
                    NumLayers=args.num_layers,
                ).to(device)

        self.rank = rank  # rank = trainer ID

        self.device = device

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=args.learning_rate, weight_decay=5e-4
        )

        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_losses: list = []
        self.train_accs: list = []

        self.test_losses: list = []
        self.test_accs: list = []

        self.local_node_index = local_node_index.to(device)
        self.communicate_node_index = communicate_node_index.to(device)

        self.adj = adj.to(device)
        self.train_labels = train_labels.to(device)
        self.test_labels = test_labels.to(device)
        self.features = features.to(device)
        self.idx_train = idx_train.to(device)
        self.idx_test = idx_test.to(device)

        self.local_step = args.local_step
        self.global_node_num = global_node_num
        self.class_num = class_num

    @torch.no_grad()
    def update_params(self, params: tuple, current_global_epoch: int) -> None:
        """
        Updates the model parameters with global parameters received from the server.

        Parameters
        ----------
        params : tuple
            A tuple containing the global parameters from the server.
        current_global_epoch : int
            The current global epoch number.
        """
        # load global parameter from global server
        self.model.to("cpu")
        for (
            p,
            mp,
        ) in zip(params, self.model.parameters()):
            mp.data = p
        self.model.to(self.device)

    def get_local_feature_sum(self) -> torch.Tensor:
        """
        Computes the sum of features of all 1-hop neighbors for each node.

        Returns
        -------
        one_hop_neighbor_feature_sum : torch.Tensor
            The sum of features of 1-hop neighbors for each node
        """

        # create a large matrix with known local node features
        new_feature_for_trainer = torch.zeros(
            self.global_node_num, self.features.shape[1]
        ).to(
            self.device
        )  # TODO:check if all the tensors are in the same device
        new_feature_for_trainer[self.local_node_index] = self.features
        # sum of features of all 1-hop nodes for each node
        one_hop_neighbor_feature_sum = get_1hop_feature_sum(
            new_feature_for_trainer, self.adj
        )
        return one_hop_neighbor_feature_sum

    def load_feature_aggregation(self, feature_aggregation: torch.Tensor) -> None:
        """
        Loads the aggregated features into the trainer.

        Parameters
        ----------
        feature_aggregation : torch.Tensor
            The aggregated features to be loaded.
        """
        self.feature_aggregation = feature_aggregation

    def relabel_adj(self) -> None:
        """
        Relabels the adjacency matrix based on the communication node index.
        """
        _, self.adj, __, ___ = torch_geometric.utils.k_hop_subgraph(
            self.communicate_node_index, 0, self.adj, relabel_nodes=True
        )

    def train(self, current_global_round: int) -> None:
        """
        Performs local training for a specified number of iterations. This method
        updates the model using the loaded feature aggregation and the adjacency matrix.

        Parameters
        ----------
        current_global_round : int
            The current global training round.
        """
        # clean cache
        torch.cuda.empty_cache()
        self.model.to(self.device)
        self.feature_aggregation = self.feature_aggregation.to(self.device)

        g = dgl.graph((self.adj[0], self.adj[1]))
        print(g)
        g.ndata["features"] = self.features
        g.ndata["labels"] = self.train_labels
        g.ndata["train_mask"] = self.idx_train
        g.ndata["test_mask"] = self.idx_test
        print(g)
        time.sleep(100)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_hops)

        dataloader = dgl.dataloading.DataLoader(
            g,
            torch.nonzero(self.idx_train).squeeze(),
            sampler,
            batch_size=6,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )  # local no parrallel

        input_nodes, output_nodes, blocks = next(iter(dataloader))
        print(blocks)
        for iteration in range(self.local_step):
            self.model.train()
            dataloader_iterator = iter(dataloader)

            for i in range(1):
                # just test the 1st one but still doesn't work
                sampled_data = next(dataloader_iterator)

                sampled_data = sampled_data.to(self.device)

                loss_train, acc_train = train(
                    iteration,
                    self.model,
                    self.optimizer,
                    sampled_data.x,
                    sampled_data.edge_index,
                    sampled_data.y[sampled_data.train_mask],
                    sampled_data.train_mask,
                )

                self.train_losses.append(loss_train)
                self.train_accs.append(acc_train)

            loss_test, acc_test = self.local_test()
            self.test_losses.append(loss_test)
            self.test_accs.append(acc_test)

    def local_test(self) -> list:
        """
        Evaluates the model on the local test dataset.

        Returns
        -------
        (list) : list
            A list containing the test loss and accuracy [local_test_loss, local_test_acc].
        """
        local_test_loss, local_test_acc = test(
            self.model,
            self.feature_aggregation,
            self.adj,
            self.test_labels,
            self.idx_test,
        )
        return [local_test_loss, local_test_acc]

    def get_params(self) -> tuple:
        """
        Retrieves the current parameters of the model.

        Returns
        -------
        (tuple) : tuple
            A tuple containing the current parameters of the model.
        """
        self.optimizer.zero_grad(set_to_none=True)
        return tuple(self.model.parameters())

    def get_all_loss_accuray(self) -> list:
        """
        Returns all recorded training and testing losses and accuracies.

        Returns
        -------
        (list) : list
            A list containing arrays of training losses, training accuracies, testing losses, and testing accuracies.
        """
        return [
            np.array(self.train_losses),
            np.array(self.train_accs),
            np.array(self.test_losses),
            np.array(self.test_accs),
        ]

    def get_rank(self) -> int:
        """
        Returns the rank (trainer ID) of the trainer.

        Returns
        -------
        (int) : int
            The rank (trainer ID) of this trainer instance.
        """
        return self.rank


def FedGCN_load_data(dataset_str: str) -> tuple:
    if dataset_str in ["cora", "citeseer", "pubmed"]:
        # download dataset from torch_geometric
        dataset = torch_geometric.datasets.Planetoid("./data", dataset_str)
        names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
        objects = []
        for i in range(len(names)):
            with open(
                "data/{}/raw/ind.{}.{}".format(dataset_str, dataset_str, names[i]), "rb"
            ) as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding="latin1"))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = FedGCN_parse_index_file(
            "data/{}/raw/ind.{}.test.index".format(dataset_str, dataset_str)
        )
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == "citeseer":
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(
                min(test_idx_reorder), max(test_idx_reorder) + 1
            )
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = torch.LongTensor(test_idx_range.tolist())
        idx_train = torch.LongTensor(range(len(y)))
        idx_val = torch.LongTensor(range(len(y), len(y) + 500))

        # features = normalize(features)
        # adj = normalize(adj)    # no normalize adj here, normalize it in the training process

        features = torch.tensor(features.toarray()).float()
        adj = torch.tensor(adj.toarray()).float()
        adj = torch_sparse.tensor.SparseTensor.from_dense(adj)
        labels = torch.tensor(labels)
        labels = torch.argmax(labels, dim=1)
    elif dataset_str in [
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
        # idx_train = (idx_train['paper'])
        # idx_val = (idx_val['paper'])
        # idx_test = (idx_test['paper'])

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        data = dataset[0]
        # print(dataset)
        # print(data.y)
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

    features, adj, labels, idx_train, idx_val, idx_test = FedGCN_load_data(args.dataset)
    class_num = labels.max().item() + 1

    if args.dataset in ["simulate", "cora", "citeseer", "pubmed", "reddit"]:
        args_hidden = 16
    else:
        args_hidden = 256

    row, col, edge_attr = adj.coo()
    edge_index = torch.stack([row, col], dim=0)

    num_cpus_per_client = 0.1
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
    # TODO: save the tensor to file
    #     save_all_trainers_data(
    #     split_node_indexes=split_node_indexes,
    #     communicate_node_indexes=communicate_node_indexes,
    #     edge_indexes_clients=edge_indexes_clients,
    #     labels=labels,
    #     features=features,
    #     in_com_train_node_indexes=in_com_train_node_indexes,
    #     in_com_test_node_indexes=in_com_test_node_indexes,
    #     n_trainer=args.n_trainer
    # )

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

    # starting monitor:
    # monitor = Monitor()
    # monitor.pretrain_time_start()
    start_time = datetime.now()
    print("starting get_local_feature_sum")
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
    # time.sleep(30)
    # monitor.pretrain_time_end()

    #######################################################################
    # Federated Training
    # ------------------
    # The server start training of all clients and aggregate the parameters
    # at every global round.

    print("global_rounds", args.global_rounds)
    # monitor.train_time_start()
    start_time = datetime.now()
    for i in range(args.global_rounds):
        server.train(i)
    end_time = datetime.now()
    time_difference = end_time - start_time
    print(f"training time: {time_difference}")
    # time.sleep(30)
    # monitor.train_time_end()

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


for d in ["cora"]:
    args.dataset = d
    for b in [1.0]:
        args.iid_beta = b
        print(f"at dataset: {d}, beta: {b}")
        run()
ray.shutdown()
