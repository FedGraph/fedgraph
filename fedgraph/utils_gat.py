import random
from typing import Any

import numpy as np
import torch
import torch_geometric


def CreateNodeSplit(graph: Any, num_clients: int) -> dict:
    nodes = [i for i in range(graph.num_nodes)]
    node_split = [random.randint(0, len(nodes)) for _ in range(num_clients - 1)]
    node_split.sort()
    node_split = [0] + node_split + [len(nodes)]
    random.shuffle(nodes)
    client_nodes = {
        i: {j: True for j in nodes[node_split[i] : node_split[i + 1]]}
        for i in range(num_clients)
    }
    for id in client_nodes:
        print("Client {ID} has {num} nodes".format(ID=id, num=len(client_nodes[id])))
    return client_nodes


def AttnFunction(x, gamma):
    return gamma * x + (1 - gamma) * 0.25 * np.log(1 + np.exp(4 * x))


def MatGen(num):
    # Function to generate the orthogonal matrices needed to encode the features

    A = np.random.uniform(low=0.0, high=3.0, size=(2 * num, 2 * num))

    A = np.matmul(A, A.T)

    alpha = np.max(np.abs(np.linalg.eig(A)[0])) + 1.0

    A += alpha * np.identity(2 * num)

    orth = np.linalg.eig(A)[1].T

    return orth


def FedGATLoss(
    LossFunc,
    y_pred,
    y_true,
    params,
    glob_params,
    dual_params,
    aug_lagrange_rho,
    dual_weight,
):
    v = LossFunc(y_pred, y_true)
    # print("current in GAT loss")

    # print("dual_params details:")
    # for key, value in dual_params.items():
    #     print(f"Key: {value}")
    #     break
    for p in params:
        #     print(f"Parameter: {p}")
        #     print(f"params[{p}]: {params[p]}")
        #     print(f"glob_params[{p}]: {glob_params[p]}")
        v += aug_lagrange_rho * torch.sum(
            (glob_params[p] - params[p]) ** 2
        ) + dual_weight * torch.sum(dual_params[p] * (glob_params[p] - params[p]))
    return v


def label_dirichlet_partition(
    labels: np.array, N: int, K: int, n_parties: int, beta: float
) -> list:
    """
    Partitions data based on labels by using the Dirichlet distribution, to ensure even distribution of samples

    Parameters
    ----------
    labels : NumPy array
        An array with labels or categories for each data point.
    N : int
        Total number of data points in the dataset.
    K : int
        Total number of unique labels.
    n_parties : int
        The number of groups into which the data should be partitioned.
    beta : float
        Dirichlet distribution parameter value.

    Returns
    -------
    split_data_indexes : list
        List indices of data points assigned into groups.

    """
    min_size = 0
    min_require_size = 10

    split_data_indexes = []
    # print(f"Input labels: {labels}")
    # print(f"Total number of data points (N): {N}")
    # print(f"Total number of unique labels (K): {K}")
    # print(f"Number of groups (n_parties): {n_parties}")
    # print(f"Dirichlet distribution parameter (beta): {beta}")

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
    print(f"Output split data indexes: {split_data_indexes}")
    return split_data_indexes


def intersect1d(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """
    Concatenates the two input tensors, finding common elements between these two

    Parameters
    ----------
    t1 : torch.Tensor
        The first input tensor for the operation.
    t2 : torch.Tensor
        The second input tensor for the operation.

    Returns
    -------
    intersection : torch.Tensor
        Intersection of the two input tensors.
    """
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    intersection = uniques[counts > 1]
    return intersection


def get_in_comm_indexes(
    edge_index: torch.Tensor,
    split_node_indexes: list,
    num_clients: int,
    L_hop: int,
    idx_train: torch.Tensor,
    idx_test: torch.Tensor,
    idx_val: torch.Tensor,
) -> tuple:
    """
    Extract and preprocess data indices and edge information. It determines the nodes that each client
    will communicate with, based on the L-hop neighborhood, and aggregates the edge information accordingly.
    It also determines the indices of training and test data points that are available to each client.

    Parameters
    ----------
    edge_index : torch.Tensor
        A tensor representing the edge information (connections between nodes) of the graph dataset.
    split_node_indexes : list
        A list of node indices. Each list element corresponds to a subset of nodes assigned to a specific client
        after data partitioning.
    num_clients : int
        The total number of clients.
    L_hop : int
        The number of hops to consider when determining the neighborhood of each node. For example, if L_hop=1,
        the 1-hop neighborhood of a node includes the node itself and all of its immediate neighbors.
    idx_train : torch.Tensor
        Tensor containing indices of training data in the graph.
    idx_test : torch.Tensor
        Tensor containing indices of test data in the graph.

    Returns
    -------
    communicate_node_indexes : list
        A list of node indices for each client, representing nodes involved in communication.
    in_com_train_node_indexes : list
        A list of tensors, where each tensor contains the indices of training data points available to each client.
    in_com_test_node_indexes : list
        A list of tensors, where each tensor contains the indices of test data points available to each client.
    edge_indexes_clients : list
        A list of tensors representing the edges between nodes within each client's subgraph.
    """
    communicate_node_indexes = []
    in_com_train_node_indexes = []
    edge_indexes_clients = []

    for i in range(num_clients):
        communicate_node_index = split_node_indexes[i]
        if L_hop == 0:
            (
                communicate_node_index,
                current_edge_index,
                _,
                __,
            ) = torch_geometric.utils.k_hop_subgraph(
                communicate_node_index, 0, edge_index, relabel_nodes=False
            )
            del _
            del __
        elif L_hop == 1 or L_hop == 2:
            (
                communicate_node_index,
                current_edge_index,
                _,
                __,
            ) = torch_geometric.utils.k_hop_subgraph(
                communicate_node_index, 1, edge_index, relabel_nodes=False
            )
            del _
            del __

        communicate_node_index = communicate_node_index.to("cpu")
        current_edge_index = current_edge_index.to("cpu")
        communicate_node_indexes.append(communicate_node_index)
        """
        current_edge_index = torch_sparse.SparseTensor(
            row=current_edge_index[0],
            col=current_edge_index[1],
            sparse_sizes=(len(communicate_node_index), len(communicate_node_index)),
        )
        """

        edge_indexes_clients.append(current_edge_index)

        inter = intersect1d(
            split_node_indexes[i], idx_train
        )  # only count the train data of nodes in current server(not communicate nodes)

        in_com_train_node_indexes.append(
            torch.searchsorted(communicate_node_indexes[i], inter).clone()
        )  # local id in block matrix

    in_com_test_node_indexes = []
    for i in range(num_clients):
        inter = intersect1d(split_node_indexes[i], idx_test)
        in_com_test_node_indexes.append(
            torch.searchsorted(communicate_node_indexes[i], inter).clone()
        )
    in_com_val_node_indexes = []
    for i in range(num_clients):
        inter = intersect1d(split_node_indexes[i], idx_val)
        in_com_val_node_indexes.append(
            torch.searchsorted(communicate_node_indexes[i], inter).clone()
        )
    return (
        communicate_node_indexes,
        in_com_train_node_indexes,
        in_com_test_node_indexes,
        in_com_val_node_indexes,
        edge_indexes_clients,
    )
