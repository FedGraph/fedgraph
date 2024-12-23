import glob
import logging
import re
import time
from io import BytesIO
from pathlib import Path

import attridict
import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric

# from huggingface_hub import HfApi, HfFolder, hf_hub_download, upload_file


def normalize(mx: sp.csc_matrix) -> sp.csr_matrix:
    """
    This function is to row-normalize sparse matrix for efficient computation of the graph

    Parameters
    ----------
    mx : sparse matrix
        Input sparse matrix to row-normalize.

    Returns
    -------
    mx : sparse matrix
        Row-normalized sparse matrix.

    Note
    ----
    Row-normalizing is usually done in graph algorithms to enable equal node contributions
    regardless of the node's degree and to stabilize, ease numerical computations.
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


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


def setdiff1d(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """
    Computes the set difference between the two input tensors

    Parameters
    ----------
    t1 : torch.Tensor
        The first input tensor for the operation.
    t2 : torch.Tensor
        The second input tensor for the operation.

    Returns
    -------
    difference : torch.Tensor
        Difference in elements of the two input tensors.

    """

    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    return difference


def label_dirichlet_partition(
    labels: np.array,
    N: int,
    K: int,
    n_parties: int,
    beta: float,
    distribution_type: str = "average",
) -> list:
    # logger.info(
    #     f"Starting label_dirichlet_partition with {n_parties} parties and {K} classes"
    # )
    start_time = time.time()

    min_require_size = max(
        1, min(10, N // (n_parties * K))
    )  # Adjust minimum size based on dataset

    # Generate weights
    if distribution_type == "lognormal":
        weights = np.random.lognormal(mean=0, sigma=2, size=n_parties)
    elif distribution_type == "powerlaw":
        weights = np.random.power(a=0.31653612251668856, size=n_parties)
    elif distribution_type == "exponential":
        weights = np.random.exponential(scale=1.0, size=n_parties)
    else:
        weights = np.ones(n_parties)
    weights /= weights.sum()

    # logger.info(f"Generated weights using {distribution_type} distribution")

    # Pre-compute label indices
    label_indices = [np.where(labels == k)[0] for k in range(K)]

    attempts = 0
    max_attempts = 1000  # increased // can revoke later
    while attempts < max_attempts:
        attempts += 1
        idx_batch = [[] for _ in range(n_parties)]

        for k in range(K):
            idx_k = label_indices[k]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))

            if distribution_type == "average":
                proportions = np.array(
                    [
                        p * (len(idx_j) < N / n_parties)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )

            proportions *= weights
            proportions /= proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]

        min_size = min(len(idx_j) for idx_j in idx_batch)

        if min_size >= min_require_size:
            break

        # if attempts % 10 == 0:
        # logger.warning(
        #     f"Attempt {attempts}: min_size ({min_size}) < min_require_size ({min_require_size})"
        # )

    # if attempts >= max_attempts:
    # logger.warning(
    #     f"Failed to meet min_require_size after {max_attempts} attempts. Using best attempt."
    # )

    # logger.info(f"Partitioning completed after {attempts} attempts")

    split_data_indexes = [np.random.permutation(idx_j).tolist() for idx_j in idx_batch]

    # logger.info(
    #     f"label_dirichlet_partition completed in {time.time() - start_time:.2f} seconds"
    # )

    return split_data_indexes


def community_partition_non_iid(
    non_iid_percent: float,
    labels: torch.Tensor,
    num_clients: int,
    nclass: int,
    args_cuda: bool,
) -> list:
    """
    Partitions data into non-IID subsets. The function first randomly assigns data points to clients, and then
    assigns non-IID data points to each client. The non-IID data points are randomly selected from the remaining
    data points that are not assigned to any client.

    Parameters
    ----------
        non_iid_percent : float
            The percentage of non-IID data in the partition.
        labels : torch.Tensor
            Tensor with class labels.
        num_clients : int
            Number of clients.
        nclass : int
            Total number of classes in the dataset.
        args_cuda : bool
            Flag indicating whether CUDA is enabled.

    Returns
    -------
    split_data_indexes : list
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
    split_node_indexes: list,
    num_clients: int,
    L_hop: int,
    idx_train: torch.Tensor,
    idx_test: torch.Tensor,
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
                communicate_node_index, 0, edge_index, relabel_nodes=True
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
            # Assert that the number of distinct elements are equal
            # distinct_communicate_node_index = torch.unique(communicate_node_index)
            # # Flatten the 2D current_edge_index to get the unique node indices involved in edges
            # distinct_current_edge_nodes = torch.unique(current_edge_index.flatten())
            # assert len(distinct_communicate_node_index) == len(
            #     distinct_current_edge_nodes
            # ), f"Distinct counts do not match: communicate_node_index ({len(distinct_communicate_node_index)}) != current_edge_nodes ({len(distinct_current_edge_nodes)})"
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
    return (
        communicate_node_indexes,
        in_com_train_node_indexes,
        in_com_test_node_indexes,
        edge_indexes_clients,
    )


def get_1hop_feature_sum(
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    device: str,
    include_self: bool = True,
) -> torch.Tensor:
    """
    Computes the sum of features of 1-hop neighbors for each node in a graph. The function
    can be used to iterate over each node, identifying its neighbors based on the `edge_index`.


    Parameters
    ----------
    node_features : torch.Tensor
        A 2D tensor containing the features of each node in the graph. Each row corresponds to a node,
        and each column corresponds to a feature.
    edge_index : torch.Tensor
        A 2D tensor representing the adjacency information of the graph which has the size of (2, num_edges),
        where the first row represents the source node, and the second row represents the target node.
    include_self : bool, optional (default=True)
        A flag to include the node's own features in the sum. If True, the features of the node itself
        are included in the summation. If False, only the features of the neighboring nodes are summed.

    Returns
    -------
    (tensor) : torch.Tensor
        A 2D tensor where each row represents the summed features of the 1-hop neighbors for each node.
        The tensor has the same number of rows as `node_features` and the same number of columns as the
        number of features per node.
    """
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]

    num_nodes, num_features = node_features.shape
    summed_features = torch.zeros((num_nodes, num_features)).to(device)

    # encryption
    # encrypted_node_features = [ts.ckks_vector(context, node_features[i].tolist()) for i in range(num_nodes)]
    if include_self:
        # print("using spare matrix method")
        adjacency_matrix = torch.sparse_coo_tensor(
            edge_index,
            torch.ones_like(source_nodes, dtype=torch.float32),
            (num_nodes, num_nodes),
        ).to(device)
        summed_features = torch.sparse.mm(adjacency_matrix.float(), node_features)
    else:
        for node in range(num_nodes):
            neighbor_indices = torch.where(
                (source_nodes == node) & (target_nodes != node)
            )  # exclude self-loop

            neighbor_features = node_features[target_nodes[neighbor_indices]]
            summed_features[node] = torch.sum(neighbor_features, dim=0)

    return summed_features


def increment_dir(dir: str, comment: str = "") -> str:
    """
    This function is used to create a new directory path by incrementing a numeric suffix in the original
    directory path.

    Parameters
    ----------
    dir : str
        The original directory path.
    comment : str, optional)
        An optional comment that can be appended to the directory name.

    Returns
    -------
    (str) : str
        Returns a string with the path of the new directory.

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


def save_trainer_data_to_hugging_face(
    trainer_id,
    local_node_index,
    communicate_node_global_index,
    global_edge_index_client,
    train_labels,
    test_labels,
    features,
    in_com_train_node_local_indexes,
    in_com_test_node_local_indexes,
    args,
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
    save_tensor_to_hf(communicate_node_global_index, "communicate_node_index.pt")
    save_tensor_to_hf(global_edge_index_client, "adj.pt")
    save_tensor_to_hf(train_labels, "train_labels.pt")
    save_tensor_to_hf(test_labels, "test_labels.pt")
    save_tensor_to_hf(features, "features.pt")
    save_tensor_to_hf(in_com_train_node_local_indexes, "idx_train.pt")
    save_tensor_to_hf(in_com_test_node_local_indexes, "idx_test.pt")

    print(f"Uploaded data for trainer {trainer_id}")


def save_all_trainers_data(
    split_node_indexes,
    communicate_node_global_indexes,
    global_edge_indexes_clients,
    labels,
    features,
    in_com_train_node_local_indexes,
    in_com_test_node_local_indexes,
    n_trainer,
    args,
):
    for i in range(n_trainer):
        save_trainer_data_to_hugging_face(
            trainer_id=i,
            local_node_index=split_node_indexes[i],
            communicate_node_global_index=communicate_node_global_indexes[i],
            global_edge_index_client=global_edge_indexes_clients[i],
            train_labels=labels[communicate_node_global_indexes[i]][
                in_com_train_node_local_indexes[i]
            ],
            test_labels=labels[communicate_node_global_indexes[i]][
                in_com_test_node_local_indexes[i]
            ],
            features=features[split_node_indexes[i]],
            in_com_train_node_local_indexes=in_com_train_node_local_indexes[i],
            in_com_test_node_local_indexes=in_com_test_node_local_indexes[i],
            args=args,
        )
