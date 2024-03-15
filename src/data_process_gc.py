import random
from random import choices

import numpy as np
import pandas as pd
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree

from src.utils_gc import get_max_degree, get_num_graph_labels, get_stats, split_data


def rand_split_chunk(
    graphs: list, num_client: int = 10, overlap: bool = False, seed: int = 42
) -> list:
    """
    Randomly split graphs into chunks for each client.

    Parameters
    ----------
    graphs: list
        The list of graphs.
    num_client: int
        The number of clients.
    overlap: bool
        Whether clients have overlapped data.
    seed: int
        Seed for randomness.

    Returns
    -------
    graphs_chunks: list
        The list of chunks for each client.
    """
    random.seed(seed)
    np.random.seed(seed)

    totalNum = len(graphs)
    minSize = min(50, int(totalNum / num_client))
    graphs_chunks = []
    if not overlap:  # non-overlapping
        for i in range(num_client):
            graphs_chunks.append(graphs[i * minSize : (i + 1) * minSize])
        for g in graphs[num_client * minSize :]:
            idx_chunk = np.random.randint(low=0, high=num_client, size=1)[0]
            graphs_chunks[idx_chunk].append(g)
    else:
        sizes = np.random.randint(low=50, high=150, size=num_client)
        for s in sizes:
            graphs_chunks.append(choices(graphs, k=s))
    return graphs_chunks


def load_single_dataset(
    datapath: str,
    dataset: str = "PROTEINS",
    num_client: int = 10,
    batch_size: int = 128,
    convert_x: bool = False,
    seed: int = 42,
    overlap: bool = False,
) -> tuple:
    """
    Graph Classification: prepare data for one dataset to multiple clients.

    Parameters
    ----------
    datapath: str
        the input path of data.
    dataset: str
        the name of dataset.
    num_client: int
        the number of clients.
    batch_size: int
        the batch size for graph classification.
    convert_x: bool
        whether to convert node features to one-hot degree.
    seed: int
        seed for randomness.
    overlap: bool
        whether clients have overlapped data.

    Returns
    -------
    splited_data: dict
        the data for each client.
    stats_df: pd.DataFrame
        the statistics of data, including the number of graphs, the number of nodes, and the number of edges
        for the training, validation, and testing sets.
    """

    if dataset == "COLLAB":
        tudataset = TUDataset(
            f"{datapath}/TUDataset", dataset, pre_transform=OneHotDegree(491, cat=False)
        )
    elif dataset == "IMDB-BINARY":
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

    """ Split data into chunks for each client """
    graphs_chunks = rand_split_chunk(
        graphs=graphs, num_client=num_client, overlap=overlap, seed=seed
    )

    splited_data = {}
    stats_df = pd.DataFrame()
    num_node_features = graphs[0].num_node_features

    for idx, chunks in enumerate(graphs_chunks):
        ds = f"{idx}-{dataset}"  # client id

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

        """Statistics"""
        stats_df = get_stats(
            df=stats_df,
            dataset=ds,  # chunked dataset
            graphs_train=ds_train,
            graphs_val=ds_val,
            graphs_test=ds_test,
        )

    return splited_data, stats_df


def load_multiple_dataset(
    datapath: str,
    dataset_group: str = "small",
    batch_size: int = 32,
    convert_x: bool = False,
    seed: int = 42,
) -> tuple:
    """
    Graph Classification: prepare data for a group of datasets to multiple clients.

    Parameters
    ----------
    datapath: str
        the input path of data.
    dataset_group: str
        the name of dataset group.
    batch_size: int
        the batch size for graph classification.
    convert_x: bool
        whether to convert node features to one-hot degree.
    seed: int
        seed for randomness.

    Returns
    -------
    splited_data: dict
        the data for each client.
    stats_df: pd.DataFrame
        the statistics of data, including the number of graphs, the number of nodes, and the number of edges
        for the training, validation, and testing sets.
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
            "COLLAB",
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
        if dataset == "COLLAB":
            tudataset = TUDataset(
                f"{datapath}/TUDataset",
                dataset,
                pre_transform=OneHotDegree(491, cat=False),
            )
        elif dataset == "IMDB-BINARY":
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

        """Statistics"""
        stats_df = get_stats(
            df=df,
            dataset=dataset,
            graphs_train=graphs_train,
            graphs_val=graphs_val,
            graphs_test=graphs_test,
        )

    return splited_data, stats_df
