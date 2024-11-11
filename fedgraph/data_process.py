# setting of data generation

import pickle as pkl
import random
import sys
from random import choices
from typing import Any

import attridict
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch_geometric
import torch_sparse
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import add_self_loops

from fedgraph.utils_gc import (
    get_max_degree,
    get_num_graph_labels,
    get_stats,
    split_data,
)
from fedgraph.utils_nc import get_in_comm_indexes, label_dirichlet_partition


def data_loader(args: attridict) -> Any:
    """
    Load data for federated learning tasks.

    Parameters
    ----------
    args: attridict
        The configuration of the task.

    Returns
    -------
    data: Any
        The data for the task.

    Note
    ----
    The function will call the corresponding data loader function based on the task.
    If the task is "NC", the function will call data_loader_NC.
    If the task is "GC", the function will call data_loader_GC.
    If the task is "LP", only the country code needs to be specified at this stage, and the function will return None.
    """
    if args.fedgraph_task == "LP":
        return None

    data_loader_function = {
        "NC": data_loader_NC,
        "GC": data_loader_GC,
    }
    return data_loader_function[args.fedgraph_task](args)


def data_loader_NC(args: attridict) -> tuple:
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
    print("config: ", args)
    if not args.use_huggingface:
        # process on the server
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
    return (
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
    )


def data_loader_GC(args: attridict) -> dict:
    """
    Load data for graph classification tasks.

    Parameters
    ----------
    args: attridict
        The configuration of the task.

    Returns
    -------
    data: dict
        The data for the task.
    """
    if args.is_multiple_dataset:
        return data_loader_GC_multiple(
            datapath=args.datapath,
            dataset_group=args.dataset_group,
            batch_size=args.batch_size,
            convert_x=args.convert_x,
            seed=args.seed_split_data,
        )
    else:
        return data_loader_GC_single(
            datapath=args.datapath,
            dataset=args.dataset,
            num_trainer=args.num_trainers,
            batch_size=args.batch_size,
            convert_x=args.convert_x,
            seed=args.seed_split_data,
            overlap=args.overlap,
        )


def NC_parse_index_file(filename: str) -> list:
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


def NC_load_data(dataset_str: str) -> tuple:
    """
    Loads input data from 'gcn/data' directory and processes these datasets into a format
    suitable for training GCN and similar models.

    Parameters
    ----------
    dataset_str : Name of the dataset to be loaded.

    Returns
    -------
    features : torch.Tensor
        Node feature matrix as a float tensor.
    adj : torch.Tensor or torch_sparse.tensor.SparseTensor
        Adjacency matrix of the graph.
    labels : torch.Tensor
        Labels of the nodes.
    idx_train : torch.LongTensor
        Indices of training nodes.
    idx_val : torch.LongTensor
        Indices of validation nodes.
    idx_test : torch.LongTensor
        Indices of test nodes.

    Note
    ----
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.
    """
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
        test_idx_reorder = NC_parse_index_file(
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
    ]:  #'ogbn-mag' is heteregeneous
        # Download and process data at './dataset/.'
        import builtins
        from unittest.mock import patch

        from ogb.nodeproppred import PygNodePropPredDataset

        # Mock the input to always return "y" under the cluster env
        with patch.object(builtins, "input", lambda _: "y"):
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

    elif dataset_str == "reddit":
        from dgl.data import RedditDataset

        data = RedditDataset()
        g = data[0]

        adj = torch_sparse.tensor.SparseTensor.from_edge_index(g.edges())

        features = g.ndata["feat"]
        train_mask = g.ndata["train_mask"]
        val_mask = g.ndata["val_mask"]
        test_mask = g.ndata["test_mask"]

        idx_train = (train_mask == True).nonzero().view(-1)
        idx_val = (val_mask == True).nonzero().view(-1)
        idx_test = (test_mask == True).nonzero().view(-1)

        labels = g.ndata["label"]

    return features.float(), adj, labels, idx_train, idx_val, idx_test


def GC_rand_split_chunk(
    graphs: list, num_trainer: int = 10, overlap: bool = False, seed: int = 42
) -> list:
    """
    Randomly split graphs into chunks for each trainer.

    Parameters
    ----------
    graphs: list
        The list of graphs.
    num_trainer: int
        The number of trainers.
    overlap: bool
        Whether trainers have overlapped data.
    seed: int
        Seed for randomness.

    Returns
    -------
    graphs_chunks: list
        The list of chunks for each trainer.
    """
    random.seed(seed)
    np.random.seed(seed)

    totalNum = len(graphs)
    minSize = min(50, int(totalNum / num_trainer))
    graphs_chunks = []
    if not overlap:  # non-overlapping
        for i in range(num_trainer):
            graphs_chunks.append(graphs[i * minSize : (i + 1) * minSize])
        for g in graphs[num_trainer * minSize :]:
            idx_chunk = np.random.randint(low=0, high=num_trainer, size=1)[0]
            graphs_chunks[idx_chunk].append(g)
    else:
        sizes = np.random.randint(low=50, high=150, size=num_trainer)
        for s in sizes:
            graphs_chunks.append(choices(graphs, k=s))
    return graphs_chunks


def data_loader_GC_single(
    datapath: str,
    dataset: str = "PROTEINS",
    num_trainer: int = 10,
    batch_size: int = 128,
    convert_x: bool = False,
    seed: int = 42,
    overlap: bool = False,
) -> dict:
    """
    Graph Classification: prepare data for one dataset to multiple trainers.

    Parameters
    ----------
    datapath: str
        The input path of data.
    dataset: str
        The name of dataset that should be available in the TUDataset.
    num_trainer: int
        The number of trainers.
    batch_size: int
        The batch size for graph classification.
    convert_x: bool
        Whether to convert node features to one-hot degree.
    seed: int
        Seed for randomness.
    overlap: bool
        Whether trainers have overlapped data.

    Returns
    -------
    splited_data: dict
        The data for each trainer.
    """
    # if dataset == "COLLAB":
    #     tudataset = TUDataset(
    #         f"{datapath}/TUDataset", dataset, pre_transform=OneHotDegree(491, cat=False)
    #     )
    if dataset == "IMDB-BINARY":
        tudataset = TUDataset(
            f"{datapath}/TUDataset", dataset, pre_transform=OneHotDegree(135, cat=False)
        )
    elif dataset == "IMDB-MULTI":
        tudataset = TUDataset(
            f"{datapath}/TUDataset", dataset, pre_transform=OneHotDegree(88, cat=False)
        )
    else:
        tudataset = TUDataset(f"{datapath}/TUDataset", dataset)
        if convert_x:
            max_degree = get_max_degree(tudataset)
            tudataset = TUDataset(
                f"{datapath}/TUDataset",
                dataset,
                transform=OneHotDegree(max_degree, cat=False),
            )

    graphs = [x for x in tudataset]
    print("Dataset name: ", dataset, " Total number of graphs: ", len(graphs))

    """ Split data into chunks for each trainer """
    graphs_chunks = GC_rand_split_chunk(
        graphs=graphs, num_trainer=num_trainer, overlap=overlap, seed=seed
    )

    splited_data = {}
    stats_df = pd.DataFrame()
    num_node_features = graphs[0].num_node_features

    for idx, chunks in enumerate(graphs_chunks):
        ds = f"{idx}-{dataset}"  # trainer id

        """Data split"""
        ds_whole = chunks
        ds_train, ds_val_test = split_data(
            ds_whole, train_size=0.8, test_size=0.2, shuffle=True, seed=seed
        )
        ds_val, ds_test = split_data(
            ds_val_test, train_size=0.5, test_size=0.5, shuffle=True, seed=seed
        )

        """Generate data loader"""
        dataloader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        dataloader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=True)
        dataloader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True)
        num_graph_labels = get_num_graph_labels(ds_train)

        """Combine data"""
        splited_data[ds] = (
            {"train": dataloader_train, "val": dataloader_val, "test": dataloader_test},
            num_node_features,
            num_graph_labels,
            len(ds_train),
        )

    return splited_data


def data_loader_GC_multiple(
    datapath: str,
    dataset_group: str = "small",
    batch_size: int = 32,
    convert_x: bool = False,
    seed: int = 42,
) -> dict:
    """
    Graph Classification: prepare data for a group of datasets to multiple trainers.

    Parameters
    ----------
    datapath: str
        The input path of data.
    dataset_group: str
        The name of dataset group.
    batch_size: int
        The batch size for graph classification.
    convert_x: bool
        Whether to convert node features to one-hot degree.
    seed: int
        Seed for randomness.

    Returns
    -------
    splited_data: dict
        The data for each trainer.
    """
    assert dataset_group in [
        "molecules",
        "molecules_tiny",
        "small",
        "mix",
        "mix_tiny",
        "biochem",
        "biochem_tiny",
    ]

    if dataset_group == "molecules" or dataset_group == "molecules_tiny":
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1"]
    if dataset_group == "small":
        datasets = [
            "MUTAG",
            "BZR",
            "COX2",
            "DHFR",
            "PTC_MR",  # small molecules
            "ENZYMES",
            "DD",
            "PROTEINS",
        ]  # bioinformatics
    if dataset_group == "mix" or dataset_group == "mix_tiny":
        datasets = [
            "MUTAG",
            "BZR",
            "COX2",
            "DHFR",
            "PTC_MR",
            "AIDS",
            "NCI1",  # small molecules
            "ENZYMES",
            "DD",
            "PROTEINS",  # bioinformatics
            # "COLLAB",
            "IMDB-BINARY",
            "IMDB-MULTI",
        ]  # social networks
    if dataset_group == "biochem" or dataset_group == "biochem_tiny":
        datasets = [
            "MUTAG",
            "BZR",
            "COX2",
            "DHFR",
            "PTC_MR",
            "AIDS",
            "NCI1",  # small molecules
            "ENZYMES",
            "DD",
            "PROTEINS",
        ]  # bioinformatics

    splited_data = {}
    df = pd.DataFrame()

    for dataset in datasets:
        if dataset == "IMDB-BINARY":
            tudataset = TUDataset(
                f"{datapath}/TUDataset",
                dataset,
                pre_transform=OneHotDegree(135, cat=False),
            )
        elif dataset == "IMDB-MULTI":
            tudataset = TUDataset(
                f"{datapath}/TUDataset",
                dataset,
                pre_transform=OneHotDegree(88, cat=False),
            )
        else:
            tudataset = TUDataset(f"{datapath}/TUDataset", dataset)
            if convert_x:
                max_degree = get_max_degree(tudataset)
                tudataset = TUDataset(
                    f"{datapath}/TUDataset",
                    dataset,
                    transform=OneHotDegree(max_degree, cat=False),
                )

        graphs = [x for x in tudataset]
        print("Dataset name: ", dataset, " Total number of graphs: ", len(graphs))

        """Split data"""
        if dataset_group.endswith("tiny"):
            graphs, _ = split_data(graphs, train_size=150, shuffle=True, seed=seed)
            graphs_train, graphs_val_test = split_data(
                graphs, test_size=0.2, shuffle=True, seed=seed
            )
            graphs_val, graphs_test = split_data(
                graphs_val_test, train_size=0.5, test_size=0.5, shuffle=True, seed=seed
            )
        else:
            graphs_train, graphs_val_test = split_data(
                graphs, test_size=0.2, shuffle=True, seed=seed
            )
            graphs_val, graphs_test = split_data(
                graphs_val_test, train_size=0.5, test_size=0.5, shuffle=True, seed=seed
            )

        num_node_features = graphs[0].num_node_features
        num_graph_labels = get_num_graph_labels(graphs_train)

        """Generate data loader"""
        dataloader_train = DataLoader(graphs_train, batch_size=batch_size, shuffle=True)
        dataloader_val = DataLoader(graphs_val, batch_size=batch_size, shuffle=True)
        dataloader_test = DataLoader(graphs_test, batch_size=batch_size, shuffle=True)

        """Combine data"""
        splited_data[dataset] = (
            {"train": dataloader_train, "val": dataloader_val, "test": dataloader_test},
            num_node_features,
            num_graph_labels,
            len(graphs_train),
        )

    return splited_data
