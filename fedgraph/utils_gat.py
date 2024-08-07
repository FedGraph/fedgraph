import random
from typing import Any

import numpy as np
import ray
import torch
import torch_geometric
from torch_geometric.utils import degree


def CreateNodeSplit(graph: Any, num_clients: int) -> dict:
    nodes = [i for i in range(graph.num_nodes)]
    node_split = [random.randint(0, len(nodes))
                  for _ in range(num_clients - 1)]
    node_split.sort()
    node_split = [0] + node_split + [len(nodes)]
    random.shuffle(nodes)
    client_nodes = {
        i: {j: True for j in nodes[node_split[i]: node_split[i + 1]]}
        for i in range(num_clients)
    }
    for id in client_nodes:
        print("Client {ID} has {num} nodes".format(
            ID=id, num=len(client_nodes[id])))
    return client_nodes


def AttnFunction(x, gamma):
    return gamma * x + (1 - gamma) * 0.25 * np.log(1 + np.exp(4 * x))


def MatGen(num):
    A = np.random.uniform(0.0, 5.0, (2 * num, 2 * num))

    A = np.matmul(A, A.T)

    E = np.linalg.eig(A)[1].T

    return E


def FedGATLoss(
    LossFunc,
    glob_comm,
    loss_weight,
    y_pred,
    y_true,
    Model,
    glob_params,
    dual_params,
    aug_lagrange_rho,
    dual_weight,
):
    v = LossFunc(y_pred, y_true)

    if glob_comm == "ADMM":
        for p_id, p, dual in zip(
            Model.parameters(), glob_params.parameters(), dual_params.parameters()
        ):
            v += 0.5 * aug_lagrange_rho * torch.sum(
                (p - p_id) ** 2
            ) + dual_weight * torch.sum(dual * (p - p_id))
            # v += 0. * dual_weight * torch.sum(dual * (p - p_id))

    return v


def VecGen(feats1, feats2, num, dim, deg):
    V = np.random.uniform(-2, 2, (num, dim))

    indices = {}

    while len(indices) < num:
        r = random.randint(0, dim - 1)

        if indices.get(r, None) == None:
            indices.update({r: True})

    index_list = [i for i in indices]

    random.shuffle(index_list)

    Keys = np.zeros((num, dim))

    InterVec = np.zeros((deg + 1, num, dim))

    for i in range(num):
        V[:, index_list[i]] = 0

        V[i, index_list[i]] = np.random.uniform(
            1, 3) * random.sample([-1, 1], 1)[0]

        Keys[i, index_list[i]] = 1

        for j in range(deg + 1):
            InterVec[j, :, index_list[i]] = 0.0

            InterVec[j, i, index_list[i]] = 1 / V[i, index_list[i]] ** j

    InterMat = np.zeros((deg + 1, dim, dim))

    for i in range(deg + 1):
        for j in range(num):
            InterMat[i, :, :] += np.outer(InterVec[i, j, :], Keys[j, :])

    temp1 = np.random.uniform(-5, 5, dim)

    temp2 = np.random.uniform(-5, 5, dim)

    temp3 = np.random.uniform(-5, 5, dim)

    mask1 = np.zeros(dim)

    for i in range(num):
        mask1 += Keys[i, :] * \
            np.dot(Keys[i, :], temp1) / np.dot(Keys[i, :], Keys[i, :])

    mask1 = temp1 - mask1

    for i in range(deg + 1):
        InterMat[i, :, :] += np.random.uniform(-2, 2) * np.outer(mask1, mask1)

    mask2 = np.zeros(dim)

    mask2 += np.dot(mask1, temp2) * mask1 / np.dot(mask1, mask1)

    for i in range(num):
        mask2 += Keys[i, :] * \
            np.dot(Keys[i, :], temp2) / np.dot(Keys[i, :], Keys[i, :])

    mask2 = temp2 - mask2

    K1 = np.zeros(dim)

    for i in range(num):
        K1 += Keys[i, :]

    K1 += np.random.uniform(1, 4) * mask2

    K2 = np.zeros((dim, feats1.shape[1]))

    for i in range(num):
        K2 += np.outer(Keys[i, :], feats2[i, :])

    K2 += np.random.uniform(1, 3) * np.outer(
        mask2, feats2[random.randint(0, num - 1), :]
    )

    M1 = np.zeros((feats1.shape[1], dim))
    M2 = np.zeros((feats2.shape[1], dim))

    for i in range(num):
        M1 += np.outer(feats1[i, :], V[i, :])
        M2 += np.outer(feats2[i, :], V[i, :])

    return M1, M2, K1, K2, InterMat


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

            proportions = (np.cumsum(proportions) *
                           len(idx_k)).astype(int)[:-1]

            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        split_data_indexes.append(idx_batch[j])
    # print(f"Output split data indexes: {split_data_indexes}")
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
    labels: torch.Tensor,
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
    in_com_labels = []
    for i in range(num_clients):
        selected_labels = labels[communicate_node_indexes[i]]
        in_com_labels.append(selected_labels.clone())
    return (
        communicate_node_indexes,
        in_com_train_node_indexes,
        in_com_test_node_indexes,
        in_com_val_node_indexes,
        edge_indexes_clients,
        in_com_labels,
    )


@ray.remote(
    num_cpus=1,
    scheduling_strategy="SPREAD",
)
def compute_node_matrix(index_list, graph, device, feats, sample_probab, max_deg):
    node_mats = {}
    d = feats.size()[1]
    degrees = compute_degrees(graph.edge_index, graph.num_nodes)

    max_degree = degrees.max().item()
    print("The maximum degree is:", max_degree)
    max_degree = int(sample_probab * max_degree)

    for node in index_list:
        print(node)
        neighbours = get_predecessors(graph, node)

        sampled_bool = np.array(
            [
                random.choices(
                    [0, 1], [1 - sample_probab, sample_probab], k=1)[0]
                for j in range(len(neighbours))
            ]
        )

        sampled_bool = torch.from_numpy(sampled_bool).to(device=device).bool()
        sampled_neigh = neighbours[sampled_bool]

        if len(sampled_neigh) < 2:
            sampled_neigh = neighbours
        elif device == torch.device("cuda"):
            if len(sampled_neigh) > max_degree:
                sampled_neigh = random.sample(list(sampled_neigh), max_degree)

        feats1 = np.zeros((len(sampled_neigh), d))
        feats2 = np.zeros((len(sampled_neigh), d))

        for i in range(len(sampled_neigh)):
            feats1[i, :] = feats[node, :].cpu().detach().numpy()
            feats2[i, :] = feats[sampled_neigh[i].item(),
                                 :].cpu().detach().numpy()

            if device == torch.device("cuda"):
                dim = max_degree
            else:
                dim = 2 * len(sampled_neigh)

            M1, M2, K1, K2 = VecGen(
                feats1,
                feats2,
                len(sampled_neigh),
                dim,
                max_deg,
                0.6,
            )
            print(torch.from_numpy(M1).float().size())

        node_mats[node] = [
            torch.from_numpy(M1).float().to(device=device),
            torch.from_numpy(M2).float().to(device=device),
            torch.from_numpy(K1).float().to(device=device),
            torch.from_numpy(K2).float().to(device=device),
        ]
    return node_mats


def compute_degrees(edge_index, num_nodes):
    row, col = edge_index
    deg = degree(row, num_nodes=num_nodes)
    return deg


def get_predecessors(data, node):
    edge_index = data.edge_index
    mask = edge_index[1] == node
    predecessors = edge_index[0, mask]
    return predecessors


def calculate_statistics(data):
    edge_index = data.edge_index
    degrees = degree(edge_index[0], data.num_nodes)

    E_degree = degrees.mean().item()
    sqrt_E_degree_2 = torch.sqrt((degrees ** 2).mean()).item()
    print(f"E_degree: {E_degree}")
    print(f"sqrt_E_degree_2: {sqrt_E_degree_2}")


def print_mask_statistics(data):
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    num_train = train_mask.sum().item()
    num_val = val_mask.sum().item()
    num_test = test_mask.sum().item()

    print(f"Number of training idx: {num_train}")
    print(f"Number of validation idx: {num_val}")
    print(f"Number of test idx: {num_test}")


def print_client_statistics(split_node_indexes, idx_train, idx_val, idx_test):
    for i, indexes in enumerate(split_node_indexes):
        num_train = len(np.intersect1d(indexes, idx_train))
        num_val = len(np.intersect1d(indexes, idx_val))
        num_test = len(np.intersect1d(indexes, idx_test))

        print(f"Client {i}:")
        print(f"  Number of training idx: {num_train}")
        print(f"  Number of validation idx: {num_val}")
        print(f"  Number of test idx: {num_test}")
