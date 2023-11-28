import glob
import re
from pathlib import Path

import numpy as np
import torch
import torch_geometric
import torch_sparse


def intersect1d(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """
    This function concatenates the two input tensors, finding common elements between these two

    Argument:
    t1: (PyTorch tensor) - The first input tensor for the operation
    t2: (PyTorch tensor) - The second input tensor for the operation

    Return:
    intersection: (PyTorch tensor) - Intersection of the two input tensors
    """
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    intersection = uniques[counts > 1]
    return intersection


def setdiff1d(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """
    This function computes the set difference between the two input tensors

    Arguments:
    t1: (PyTorch tensor) - The first input tensor for the operation
    t2: (PyTorch tensor) - The second input tensor for the operation

    Return:
    difference: (PyTorch tensor) - Difference in elements of the two input tensors

    """

    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    return difference


def label_dirichlet_partition(
    labels: np.array, N: int, K: int, n_parties: int, beta: float
) -> list:
    """
    This function partitions data based on labels by using the Dirichlet distribution, to ensure even distribution of samples

    Arguments:
    labels: (NumPy array) - An array with labels or categories for each data point
    N: (int) - Total number of data points in the dataset
    K: (int) - Total number of unique labels
    n_parties: (int) - The number of groups into which the data should be partitioned
    beta: (float) - Dirichlet distribution parameter value

    Return:
    split_data_indexes (list) - list indices of data points assigned into groups

    """
    min_size = 0
    min_require_size = 10

    split_data_indexes = []

    while min_size < min_require_size:
        idx_batch: list[list[int]] = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))

            proportions = np.array(
                [
                    p * (len(idx_j) < N / n_parties)
                    for p, idx_j in zip(proportions, idx_batch)
                ]
            )

            proportions = proportions / proportions.sum()

            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        split_data_indexes.append(idx_batch[j])
    return split_data_indexes


def parition_non_iid(
    non_iid_percent: float,
    labels: torch.Tensor,
    num_clients: int,
    nclass: int,
    args_cuda: bool,
) -> list:
    """
    This function partitions data into non-IID subsets.

    Arguments:
        non_iid_percent: (float) - The percentage of non-IID data in the partition
        labels: (torch.Tensor) - Tensor with class labels
        num_clients: (int) - Number of clients
        nclass: (int) - Total number of classes in the dataset
        args_cuda: (bool) - Flag indicating whether CUDA is enabled

    Returns:
        A list containing indexes of data points assigned to each client.
    """

    split_data_indexes = []
    iid_indexes = []  # random assign
    shuffle_labels = []  # make train data points split into different devices
    for i in range(num_clients):
        current = torch.nonzero(labels == i).reshape(-1)
        current = current[np.random.permutation(len(current))]  # shuffle
        shuffle_labels.append(current)

    average_device_of_class = num_clients // nclass
    if num_clients % nclass != 0:  # for non-iid
        average_device_of_class += 1
    for i in range(num_clients):
        label_i = i // average_device_of_class
        labels_class = shuffle_labels[label_i]

        average_num = int(
            len(labels_class) // average_device_of_class * non_iid_percent
        )
        split_data_indexes.append(
            (
                labels_class[
                    average_num
                    * (i % average_device_of_class) : average_num
                    * (i % average_device_of_class + 1)
                ]
            )
        )

    if args_cuda:
        iid_indexes = setdiff1d(
            torch.tensor(range(len(labels))).cuda(), torch.cat(split_data_indexes)
        )
    else:
        iid_indexes = setdiff1d(
            torch.tensor(range(len(labels))), torch.cat(split_data_indexes)
        )
    iid_indexes = iid_indexes[np.random.permutation(len(iid_indexes))]

    for i in range(num_clients):  # for iid
        label_i = i // average_device_of_class
        labels_class = shuffle_labels[label_i]

        average_num = int(
            len(labels_class) // average_device_of_class * (1 - non_iid_percent)
        )
        split_data_indexes[i] = list(split_data_indexes[i]) + list(
            iid_indexes[:average_num]
        )

        iid_indexes = iid_indexes[average_num:]
    return split_data_indexes


def get_in_comm_indexes(
    edge_index: torch.Tensor,
    split_data_indexes: list,
    num_clients: int,
    L_hop: int,
    idx_train: torch.Tensor,
    idx_test: torch.Tensor,
) -> tuple:
    """
    This function is used to extract and preprocess data indices and edge information

    Arguments:
    edge_index: (PyTorch tensor) - Edge information (connection between nodes) of the graph dataset
    split_data_indexes: (List) - A list of indices of data points assigned to a particular group post data partition
    num_clients: (int) - Total number of clients
    L_hop: (int) - Number of hops
    idx_train: (PyTorch tensor) - Indices of training data
    idx_test: (PyTorch tensor) - Indices of test data

    Returns:
    communicate_indexes: (list) - A list of indices assigned to a particular client
    in_com_train_data_indexes: (list) - A list of tensors where each tensor contains the indices of training data points available to each client
    edge_indexes_clients: (list) - A list of edge tensors representing the edges between nodes within each client's subgraph
    """
    communicate_indexes = []
    in_com_train_data_indexes = []
    edge_indexes_clients = []

    for i in range(num_clients):
        communicate_index = split_data_indexes[i]

        if L_hop == 0:
            (
                communicate_index,
                current_edge_index,
                _,
                __,
            ) = torch_geometric.utils.k_hop_subgraph(
                communicate_index, 0, edge_index, relabel_nodes=True
            )
            del _
            del __

        for hop in range(L_hop):
            if hop != L_hop - 1:
                communicate_index = torch_geometric.utils.k_hop_subgraph(
                    communicate_index, 1, edge_index, relabel_nodes=True
                )[0]
            else:
                (
                    communicate_index,
                    current_edge_index,
                    _,
                    __,
                ) = torch_geometric.utils.k_hop_subgraph(
                    communicate_index, 1, edge_index, relabel_nodes=True
                )
                del _
                del __

        communicate_index = communicate_index.to("cpu")
        current_edge_index = current_edge_index.to("cpu")
        communicate_indexes.append(communicate_index)

        current_edge_index = torch_sparse.SparseTensor(
            row=current_edge_index[0],
            col=current_edge_index[1],
            sparse_sizes=(len(communicate_index), len(communicate_index)),
        )

        edge_indexes_clients.append(current_edge_index)

        inter = intersect1d(
            split_data_indexes[i], idx_train
        )  ###only count the train data of nodes in current server(not communicate nodes)

        in_com_train_data_indexes.append(
            torch.searchsorted(communicate_indexes[i], inter).clone()
        )  # local id in block matrix

    in_com_test_data_indexes = []
    for i in range(num_clients):
        inter = intersect1d(split_data_indexes[i], idx_test)
        in_com_test_data_indexes.append(
            torch.searchsorted(communicate_indexes[i], inter).clone()
        )
    return (
        communicate_indexes,
        in_com_train_data_indexes,
        in_com_test_data_indexes,
        edge_indexes_clients,
    )


def get_in_comm_indexes_BDS_GCN(
    edge_index: torch.Tensor,
    split_data_indexes: list,
    num_clients: int,
    idx_train: torch.Tensor,
    idx_test: torch.Tensor,
    sample_rate: float = 0.5,
) -> tuple:
    communicate_indexes = []
    in_com_train_data_indexes = []
    edge_indexes_clients = []

    for i in range(num_clients):
        communicate_index = split_data_indexes[i]

        communicate_index = torch_geometric.utils.k_hop_subgraph(
            communicate_index, 1, edge_index
        )[0].cpu()

        diff = setdiff1d(split_data_indexes[i], communicate_index)
        sample_index = torch.cat(
            (
                split_data_indexes[i],
                diff[torch.randperm(len(diff))[: int(len(diff) * sample_rate)]],
            )
        ).clone()
        sample_index = sample_index.sort()[0]

        # get edge_index with relabel_nodes
        (
            communicate_index,
            current_edge_index,
            _,
            __,
        ) = torch_geometric.utils.k_hop_subgraph(
            sample_index, 0, edge_index, relabel_nodes=True
        )
        del _
        del __

        communicate_index = communicate_index.to("cpu")
        current_edge_index = current_edge_index.to("cpu")
        communicate_indexes.append(communicate_index)

        current_edge_index = torch_sparse.SparseTensor(
            row=current_edge_index[0],
            col=current_edge_index[1],
            sparse_sizes=(len(communicate_index), len(communicate_index)),
        )

        edge_indexes_clients.append(current_edge_index)

        inter = intersect1d(
            split_data_indexes[i], idx_train
        )  ###only count the train data of nodes in current server(not communicate nodes)

        in_com_train_data_indexes.append(
            torch.searchsorted(communicate_indexes[i], inter).clone()
        )  # local id in block matrix

    in_com_test_data_indexes = []
    for i in range(num_clients):
        inter = intersect1d(split_data_indexes[i], idx_test)
        in_com_test_data_indexes.append(
            torch.searchsorted(communicate_indexes[i], inter).clone()
        )
    return (
        communicate_indexes,
        in_com_train_data_indexes,
        in_com_test_data_indexes,
        edge_indexes_clients,
    )


def increment_dir(dir: str, comment: str = "") -> str:
    """
    This function is used to create a new directory path by incrementing a numeric suffix in the original directory path

    Arguments:
    dir: (str) - The original directory path
    comment: (str, optional) - An optional comment that can be appended to the directory name

    Returns:
    Returns a string with the path of the new directory

    """
    # Increments a directory runs/exp1 --> runs/exp2_comment
    n = 0  # number
    dir = str(Path(dir))  # os-agnostic
    dirs = sorted(glob.glob(dir + "*"))  # directories
    if dirs:
        matches = [re.search(r"exp(\d+)", d) for d in dirs]
        idxs = [int(m.groups()[0]) for m in matches if m]
        if idxs:
            n = max(idxs) + 1  # increment
    return dir + str(n) + ("_" + comment if comment else "")
