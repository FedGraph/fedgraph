# setting of data generation

import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric
import torch_sparse


def parse_index_file(filename: str) -> list:
    """
    Reads and parses an index file

    Parameters
    ----------
    filename : str
        Name or path of the file to parse

    Returns
    -------
    index : list
        List of integers, each integer in the list represents int of the lines of the input file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


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
        Returns the row-normalized sparse matrix.

    Note
    ----
    Row-normalizing is usually done in graph algorithms to enable equal node contributions 
    regardless of the node's degree and to stabilize, ease numerical computations
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def load_data(dataset_str: str) -> tuple:
    """
    Loads input data from 'gcn/data' directory and processes these datasets into a format 
    suitable for training GCN and similar models.

    Parameters
    ----------
    dataset_str : Name of the dataset to be loaded

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
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
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
        test_idx_reorder = parse_index_file(
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
