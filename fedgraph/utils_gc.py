import argparse
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.utils import degree, to_networkx

from fedgraph.server_class import Server_GC
from fedgraph.trainer_class import Trainer_GC


def setup_trainers(
    splited_data: dict, base_model: Any, args: argparse.Namespace
) -> tuple:
    """
    Setup trainers for graph classification.

    Parameters
    ----------
    splited_data: dict
        The data for each trainer.
    base_model: Any
        The base model for the trainer. The base model shown in the example is GIN.
    args: argparse.ArgumentParser
        The input arguments.

    Returns
    -------
    (trainers, idx_trainers): tuple(list, dict)
        trainers: List of trainers
        idx_trainers: Dictionary with the index of the trainer as the key and the dataset name as the value
    """
    idx_trainers = {}
    trainers = []
    for idx, dataset_trainer_name in enumerate(splited_data.keys()):
        idx_trainers[idx] = dataset_trainer_name
        """acquire data"""
        dataloaders, num_node_features, num_graph_labels, train_size = splited_data[
            dataset_trainer_name
        ]

        """build GIN model"""
        cmodel_gc = base_model(
            nfeat=num_node_features,
            nhid=args.hidden,
            nclass=num_graph_labels,
            nlayer=args.nlayer,
            dropout=args.dropout,
        )

        """build optimizer"""
        optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, cmodel_gc.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        """build trainer"""
        trainer = Trainer_GC(
            model=cmodel_gc,  # GIN model
            trainer_id=idx,  # trainer id
            trainer_name=dataset_trainer_name,  # trainer name
            train_size=train_size,  # training size
            dataloader=dataloaders,  # data loader
            optimizer=optimizer,  # optimizer
            args=args,
        )

        trainers.append(trainer)

    return trainers, idx_trainers


def setup_server(base_model: Any, args: argparse.Namespace) -> Server_GC:
    """
    Setup server.

    Parameters
    ----------
    base_model: Any
        The base model for the server. The base model shown in the example is GIN_server.
    args: argparse.ArgumentParser
        The input arguments

    Returns
    -------
    server: Server_GC
        The server object
    """

    smodel = base_model(nlayer=args.nlayer, nhid=args.hidden)
    server = Server_GC(smodel, args.device)
    return server


def get_max_degree(graphs: Any) -> int:
    """
    Get the maximum degree of the graphs in the dataset.

    Parameters
    ----------
    graphs: Any
        The object of graphs

    Returns
    -------
    max_degree: int
        The maximum degree of the graphs in the dataset
    """
    max_degree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        g_degree = max(dict(g.degree).values())
        max_degree = max(max_degree, g_degree)

    return max_degree


def convert_to_node_attributes(graphs: Any) -> list:
    """
    Use only the node attributes of the graphs. This function will treat the graphs as callable objects.

    Parameters
    ----------
    graphs: Any
        The object of of graphs

    Returns
    -------
    new_graphs: list
        List of graphs with only the node attributes
    """
    num_node_attributes = graphs.num_node_attributes
    new_graphs = []
    for _, graph in enumerate(graphs):
        new_graph = graph.clone()
        new_graph.__setitem__("x", graph.x[:, :num_node_attributes])
        new_graphs.append(new_graph)
    return new_graphs


def convert_to_node_degree_features(graphs: list) -> list:
    """
    Convert the node attributes of the graphs to node degree features.

    Parameters
    ----------
    graphs: list
        List of graphs

    Returns
    -------
    new_graphs: list
        List of graphs with node degree features
    """
    graph_infos = []
    max_degree = 0
    for _, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        g_degree = max(dict(g.degree).values())
        max_degree = max(max_degree, g_degree)
        graph_infos.append(
            (graph, g.degree, graph.num_nodes)
        )  # (graph, node_degrees, num_nodes)

    new_graphs = []
    for i, tuple in enumerate(graph_infos):
        idx, x = tuple[0].edge_index[0], tuple[0].x
        deg = degree(idx, tuple[2], dtype=torch.long)
        deg = F.one_hot(deg, num_classes=max_degree + 1).to(torch.float)

        new_graph = tuple[0].clone()
        new_graph.__setitem__("x", deg)
        new_graphs.append(new_graph)

    return new_graphs


def split_data(
    graphs: list,
    train_size: float = 0.8,
    test_size: float = 0.2,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple:
    """
    Split the dataset into training and test sets.

    Parameters
    ----------
    graphs: list
        List of graphs
    train_size: float
        The proportion (ranging from 0.0 to 1.0) of the dataset to include in the training set
    test_size: float
        The proportion (ranging from 0.0 to 1.0) of the dataset to include in the test set
    shuffle: bool
        Whether or not to shuffle the data before splitting
    seed: int
        Seed for the random number generator

    Returns
    -------
    graphs_train: list
        List of training graphs
    graphs_test: list
        List of testing graphs

    Note
    ----
    The function uses sklearn.model_selection.train_test_split to split the dataset into training and test sets.
    If the dataset needs to be split into training, validation, and test sets, the function should be called twice.
    """
    y = torch.cat([graph.y for graph in graphs])
    graphs_train, graphs_test = train_test_split(
        graphs,
        train_size=train_size,
        test_size=test_size,
        stratify=y,
        shuffle=shuffle,
        random_state=seed,
    )
    return graphs_train, graphs_test


def get_num_graph_labels(dataset: list) -> int:
    """
    Get the number of unique graph labels in the dataset.

    Parameters
    ----------
    dataset: list
        List of graphs

    Returns
    -------
    (labels.length): int
        Number of unique graph labels in the dataset
    """
    s = set()
    for g in dataset:
        s.add(g.y.item())
    return len(s)


def get_avg_nodes_edges(graphs: list) -> tuple:
    """
    Calculate the average number of nodes and edges in the dataset.

    Parameters
    ----------
    graphs: list
        List of graphs

    Returns
    -------
    avg_nodes: float
        The average number of nodes in the dataset
    avg_edges: float
        The average number of edges in the dataset
    """
    num_nodes, num_edges = 0.0, 0.0
    num_graphs = len(graphs)
    for g in graphs:
        num_nodes += g.num_nodes
        num_edges += g.num_edges / 2.0  # undirected

    avg_nodes = num_nodes / num_graphs
    avg_edges = num_edges / num_graphs
    return avg_nodes, avg_edges


def get_stats(
    df: pd.DataFrame,
    dataset: str,
    graphs_train: list = [],
    graphs_val: list = [],
    graphs_test: list = [],
) -> pd.DataFrame:
    """
    Calculate and store the statistics of the dataset, including the number of graphs, average number of nodes and edges
    for the training, validation, and testing sets.

    Parameters
    ----------
    df: pd.DataFrame
        An empty DataFrame to store the statistics of the dataset.
    dataset: str
        The name of the dataset.
    graphs_train: list
        List of training graphs.
    graphs_val: list
        List of validation graphs.
    graphs_test: list
        List of testing graphs.

    Returns
    -------
    df: pd.DataFrame
        The filled statistics of the dataset.
    """

    df.loc[dataset, "#graphs_train"] = len(graphs_train)
    avgNodes, avgEdges = get_avg_nodes_edges(graphs_train)
    df.loc[dataset, "avgNodes_train"] = avgNodes
    df.loc[dataset, "avgEdges_train"] = avgEdges

    if graphs_val:
        df.loc[dataset, "#graphs_val"] = len(graphs_val)
        avgNodes, avgEdges = get_avg_nodes_edges(graphs_val)
        df.loc[dataset, "avgNodes_val"] = avgNodes
        df.loc[dataset, "avgEdges_val"] = avgEdges

    if graphs_test:
        df.loc[dataset, "#graphs_test"] = len(graphs_test)
        avgNodes, avgEdges = get_avg_nodes_edges(graphs_test)
        df.loc[dataset, "avgNodes_test"] = avgNodes
        df.loc[dataset, "avgEdges_test"] = avgEdges

    return df
