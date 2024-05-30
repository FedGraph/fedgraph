import random
from typing import Any

import networkx as nx
import numpy as np
import ray
import torch
from dtaidistance import dtw

from fedgraph.gnn_models import GCN, GNN_LP, AggreGCN, GCN_arxiv, SAGE_products


class Server:
    """
    This is a server class for federated learning which is responsible for aggregating model parameters
    from different trainers, updating the central model, and then broadcasting the updated model parameters
    back to the trainers.

    Parameters
    ----------
    feature_dim : int
        The dimensionality of the feature vectors in the dataset.
    args_hidden : int
        The number of hidden units.
    class_num : int
        The number of classes for classification in the dataset.
    device : torch.device
        The device initialized for the server model.
    trainers : list[Trainer_General]
        A list of `Trainer_General` instances representing the trainers.
    args : Any
        Additional arguments required for initializing the server model and other configurations.

    Attributes
    ----------
    model : [AggreGCN, GCN_arxiv, SAGE_products, GCN]
        The central GCN model that is trained in a federated manner.
    trainers : list[Trainer_General]
        The list of trainer instances.
    num_of_trainers : int
        The number of trainers.
    """

    def __init__(
        self,
        feature_dim: int,
        args_hidden: int,
        class_num: int,
        device: torch.device,
        trainers: list,
        args: Any,
    ) -> None:
        # server model on cpu
        if args.num_hops >= 1 and args.fedtype == "fedgcn":
            self.model = AggreGCN(
                nfeat=feature_dim,
                nhid=args_hidden,
                nclass=class_num,
                dropout=0.5,
                NumLayers=args.num_layers,
            )
        else:
            if args.dataset == "ogbn-arxiv":
                self.model = GCN_arxiv(
                    nfeat=feature_dim,
                    nhid=args_hidden,
                    nclass=class_num,
                    dropout=0.5,
                    NumLayers=args.num_layers,
                )
            elif args.dataset == "ogbn-products":
                self.model = SAGE_products(
                    nfeat=feature_dim,
                    nhid=args_hidden,
                    nclass=class_num,
                    dropout=0.5,
                    NumLayers=args.num_layers,
                )
            else:
                self.model = GCN(
                    nfeat=feature_dim,
                    nhid=args_hidden,
                    nclass=class_num,
                    dropout=0.5,
                    NumLayers=args.num_layers,
                )

        self.trainers = trainers
        self.num_of_trainers = len(trainers)
        self.broadcast_params(-1)

    @torch.no_grad()
    def zero_params(self) -> None:
        """
        Zeros out the parameters of the central model.
        """
        for p in self.model.parameters():
            p.zero_()

    @torch.no_grad()
    def train(self, current_global_epoch: int) -> None:
        """
        Training round which perform aggregating parameters from trainers, updating the central model,
        and then broadcasting the updated parameters back to the trainers.

        Parameters
        ----------
        current_global_epoch : int
            The current global epoch number during the federated learning process.
        """
        for trainer in self.trainers:
            trainer.train.remote(current_global_epoch)
        params = [trainer.get_params.remote() for trainer in self.trainers]
        self.zero_params()

        while True:
            ready, left = ray.wait(params, num_returns=1, timeout=None)
            if ready:
                for t in ready:
                    for p, mp in zip(ray.get(t), self.model.parameters()):
                        mp.data += p.cpu()
            params = left
            if not params:
                break

        for p in self.model.parameters():
            p /= self.num_of_trainers
        self.broadcast_params(current_global_epoch)

    def broadcast_params(self, current_global_epoch: int) -> None:
        """
        Broadcasts the current parameters of the central model to all trainers.

        Parameters
        ----------
        current_global_epoch : int
            The current global epoch number during the federated learning process.
        """
        for trainer in self.trainers:
            trainer.update_params.remote(
                tuple(self.model.parameters()), current_global_epoch
            )  # run in submit order


class Server_GC:
    """
    This is a server class for federated graph classification which is responsible for
    aggregating model parameters from different trainers, updating the central model,
    and then broadcasting the updated model parameters back to the trainers.

    Parameters
    ----------
    model: torch.nn.Module
        The base model that the federated learning is performed on.
    device: torch.device
        The device to run the model on.

    Attributes
    ----------
    model: torch.nn.Module
        The base model for the server.
    W: dict
        Dictionary containing the model parameters.
    model_cache: list
        List of tuples, where each tuple contains the model parameters and the accuracies of the trainers.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model.to(device)
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.model_cache: Any = []

    ########### Public functions ###########
    def random_sample_trainers(self, all_trainers: list, frac: float) -> list:
        """
        Randomly sample a fraction of trainers.

        Parameters
        ----------
        all_trainers: list
            list of trainer objects
        frac: float
            fraction of trainers to be sampled

        Returns
        -------
        (sampled_trainers): list
            list of trainer objects
        """
        return random.sample(all_trainers, int(len(all_trainers) * frac))

    def aggregate_weights(self, selected_trainers: list) -> None:
        """
        Perform weighted aggregation among selected trainers. The weights are the number of training samples.

        Parameters
        ----------
        selected_trainers: list
            list of trainer objects
        """
        total_size = 0
        for trainer in selected_trainers:
            total_size += trainer.train_size

        for k in self.W.keys():
            # pass train_size, and weighted aggregate
            accumulate = torch.stack(
                [
                    torch.mul(trainer.W[k].data, trainer.train_size)
                    for trainer in selected_trainers
                ]
            )
            self.W[k].data = torch.div(torch.sum(accumulate, dim=0), total_size).clone()

    def compute_pairwise_similarities(self, trainers: list) -> np.ndarray:
        """
        This function computes the pairwise cosine similarities between the gradients of the trainers.

        Parameters
        ----------
        trainers: list
            list of trainer objects

        Returns
        -------
        np.ndarray
            2D np.ndarray of shape len(trainers) * len(trainers), which contains the pairwise cosine similarities
        """
        trainer_dWs = []
        for trainer in trainers:
            dW = {}
            for k in self.W.keys():
                dW[k] = trainer.dW[k]
            trainer_dWs.append(dW)

        return self.__pairwise_angles(trainer_dWs)

    def compute_pairwise_distances(
        self, seqs: list, standardize: bool = False
    ) -> np.ndarray:
        """
        This function computes the pairwise distances between the gradient norm sequences of the trainers.

        Parameters
        ----------
        seqs: list
            list of 1D np.ndarray, where each 1D np.ndarray contains the gradient norm sequence of a trainer
        standardize: bool
            whether to standardize the distance matrix

        Returns
        -------
        distances: np.ndarray
            2D np.ndarray of shape len(seqs) * len(seqs), which contains the pairwise distances
        """
        if standardize:
            # standardize to only focus on the trends
            seqs = np.array(seqs)
            seqs = seqs / np.std(seqs, axis=1).reshape(-1, 1)
            distances = dtw.distance_matrix(seqs)
        else:
            distances = dtw.distance_matrix(seqs)
        return distances

    def min_cut(self, similarity: np.ndarray, idc: list) -> tuple:
        """
        This function computes the minimum cut of the graph defined by the pairwise cosine similarities.

        Parameters
        ----------
        similarity: np.ndarray
            2D np.ndarray of shape len(trainers) * len(trainers), which contains the pairwise cosine similarities
        idc: list
            list of trainer indices

        Returns
        -------
        (c1, c2): tuple
            tuple of two lists, where each list contains the indices of the trainers in a cluster
        """
        g = nx.Graph()
        for i in range(len(similarity)):
            for j in range(len(similarity)):
                g.add_edge(i, j, weight=similarity[i][j])
        _, partition = nx.stoer_wagner(
            g
        )  # using Stoer-Wagner algorithm to find the minimum cut
        c1 = np.array([idc[x] for x in partition[0]])
        c2 = np.array([idc[x] for x in partition[1]])
        return c1, c2

    def aggregate_clusterwise(self, trainer_clusters: list) -> None:
        """
        Perform weighted aggregation among the trainers in each cluster.
        The weights are the number of training samples.

        Parameters
        ----------
        trainer_clusters: list
            list of cluster-specified trainer groups, where each group contains the trainer objects in a cluster
        """
        for cluster in trainer_clusters:  # cluster is a list of trainer objects
            targs, sours = [], []
            total_size = 0
            for trainer in cluster:
                W = {}
                dW = {}
                for k in self.W.keys():
                    W[k] = trainer.W[k]
                    dW[k] = trainer.dW[k]
                targs.append(W)
                sours.append((dW, trainer.train_size))
                total_size += trainer.train_size
            # pass train_size, and weighted aggregate
            self.__reduce_add_average(
                targets=targs, sources=sours, total_size=total_size
            )

    def compute_max_update_norm(self, cluster: list) -> float:
        """
        Compute the maximum update norm (i.e., dW) among the trainers in the cluster.
        This function is used to determine whether the cluster is ready to be split.

        Parameters
        ----------
        cluster: list
            list of trainer objects
        """
        max_dW = -np.inf
        for trainer in cluster:
            dW = {}
            for k in self.W.keys():
                dW[k] = trainer.dW[k]
            curr_dW = torch.norm(self.__flatten(dW)).item()
            max_dW = max(max_dW, curr_dW)

        return max_dW

    def compute_mean_update_norm(self, cluster: list) -> float:
        """
        Compute the mean update norm (i.e., dW) among the trainers in the cluster.
        This function is used to determine whether the cluster is ready to be split.

        Parameters
        ----------
        cluster: list
            list of trainer objects
        """
        cluster_dWs = []
        for trainer in cluster:
            dW = {}
            for k in self.W.keys():
                # dW[k] = trainer.dW[k]
                dW[k] = (
                    trainer.dW[k]
                    * trainer.train_size
                    / sum([c.train_size for c in cluster])
                )
            cluster_dWs.append(self.__flatten(dW))

        return torch.norm(torch.mean(torch.stack(cluster_dWs), dim=0)).item()

    def cache_model(self, idcs: list, params: dict, accuracies: list) -> None:
        """
        Cache the model parameters and accuracies of the trainers.

        Parameters
        ----------
        idcs: list
            list of trainer indices
        params: dict
            dictionary containing the model parameters of the trainers
        accuracies: list
            list of accuracies of the trainers
        """
        self.model_cache += [
            (
                idcs,
                {name: params[name].data.clone() for name in params},
                [accuracies[i] for i in idcs],
            )
        ]

    ########### Private functions ###########
    def __pairwise_angles(self, sources: list) -> np.ndarray:
        """
        Compute the pairwise cosine similarities between the gradients of the trainers into a 2D matrix.

        Parameters
        ----------
        sources: list
            list of dictionaries, where each dictionary contains the gradients of a trainer

        Returns
        -------
        np.ndarray
            2D np.ndarray of shape len(sources) * len(sources), which contains the pairwise cosine similarities
        """
        angles = torch.zeros([len(sources), len(sources)])
        for i, source1 in enumerate(sources):
            for j, source2 in enumerate(sources):
                s1 = self.__flatten(source1)
                s2 = self.__flatten(source2)
                angles[i, j] = (
                    torch.true_divide(
                        torch.sum(s1 * s2), max(torch.norm(s1) * torch.norm(s2), 1e-12)
                    )
                    + 1
                )

        return angles.numpy()

    def __flatten(self, source: dict) -> torch.Tensor:
        """
        Flatten the gradients of a trainer into a 1D tensor.

        Parameters
        ----------
        source: dict
            dictionary containing the gradients of a trainer

        Returns
        -------
        (flattend_gradients): torch.Tensor
            1D tensor containing the flattened gradients
        """
        return torch.cat([value.flatten() for value in source.values()])

    def __reduce_add_average(
        self, targets: list, sources: list, total_size: int
    ) -> None:
        """
        Perform weighted aggregation from the sources to the targets. The weights are the number of training samples.

        Parameters
        ----------
        targets: list
            list of dictionaries, where each dictionary contains the model parameters of a trainer
        sources: list
            list of tuples, where each tuple contains the gradients and the number of training samples of a trainer
        total_size: int
            total number of training samples
        """
        for target in targets:
            for name in target:
                weighted_stack = torch.stack(
                    [torch.mul(source[0][name].data, source[1]) for source in sources]
                )
                tmp = torch.div(torch.sum(weighted_stack, dim=0), total_size).clone()
                target[name].data += tmp


class Server_LP:
    """
    This is a server class for federated graph link prediction which is responsible for aggregating model parameters
    from different trainers, updating the central model, and then broadcasting the updated model parameters
    back to the trainers.

    Parameters
    ----------
    number_of_users: int
        The number of users in the dataset.
    number_of_items: int
        The number of items in the dataset.
    meta_data: dict
        Dictionary containing the meta data of the dataset.
    args_cuda: bool
        Whether to run the model on GPU.
    """

    def __init__(
        self,
        number_of_users: int,
        number_of_items: int,
        meta_data: tuple,
        trainers: list,
        args_cuda: bool = False,
    ) -> None:
        self.global_model = GNN_LP(
            number_of_users, number_of_items, meta_data, hidden_channels=64
        )  # create the base model

        self.global_model = self.global_model.cuda() if args_cuda else self.global_model
        self.clients = trainers

    def fedavg(self, gnn_only: bool = False) -> dict:
        """
        This function performs federated averaging on the model parameters of the clients.

        Parameters
        ----------
        clients: list
            List of client objects
        gnn_only: bool, optional
            Whether to get only the GNN parameters

        Returns
        -------
        model_avg_parameter: dict
            The averaged model parameters
        """
        local_model_parameters = [
            trainer.get_model_parameter.remote(gnn_only) for trainer in self.clients
        ]
        # Initialize an empty list to collect the results
        model_states = []

        # Collect the model parameters as they become ready
        while local_model_parameters:
            ready, left = ray.wait(local_model_parameters, num_returns=1, timeout=None)
            if ready:
                for t in ready:
                    model_states.append(ray.get(t))
            local_model_parameters = left

        model_avg_parameter = self.__average_parameter(model_states)
        return model_avg_parameter

    def set_model_parameter(
        self, model_state_dict: dict, gnn_only: bool = False
    ) -> None:
        """
        Set the model parameters

        Parameters
        ----------
        model_state_dict: dict
            The model parameters
        gnn_only: bool, optional
            Whether to set only the GNN parameters
        """
        if gnn_only:
            self.global_model.gnn.load_state_dict(model_state_dict)
        else:
            self.global_model.load_state_dict(model_state_dict)

    def get_model_parameter(self, gnn_only: bool = False) -> dict:
        """
        Get the model parameters

        Parameters
        ----------
        gnn_only: bool
            Whether to get only the GNN parameters

        Returns
        -------
        dict
            The model parameters
        """
        return (
            self.global_model.gnn.state_dict()
            if gnn_only
            else self.global_model.state_dict()
        )

    # Private functions
    def __average_parameter(self, states: list) -> dict:
        """
        This function averages the model parameters of the clients.

        Parameters
        ----------
        states: list
            List of model parameters

        Returns
        -------
        global_state: dict
            The averaged model parameters
        """
        global_state = dict()
        # Average all parameters
        for key in states[0]:
            global_state[key] = states[0][key]  # for the first client
            for i in range(1, len(states)):
                global_state[key] += states[i][key]
            global_state[key] /= len(states)  # average
        return global_state
