import copy
import random
from typing import Any

import networkx as nx
import numpy as np
import ray
import torch
from dtaidistance import dtw

from fedgraph.gnn_models import (
    GCN,
    GNN_LP,
    AggreGCN,
    FedGATModel,
    GCN_arxiv,
    SAGE_products,
)
from fedgraph.utils_gat import MatGen


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
        size_refs = [trainer.get_train_size.remote() for trainer in selected_trainers]
        while size_refs:
            ready, left = ray.wait(size_refs, num_returns=1, timeout=None)
            if ready:
                for t in ready:
                    total_size += ray.get(t)
            size_refs = left

        for k in self.W.keys():
            # pass train_size, and weighted aggregate
            accumulate_list = []

            acc_refs = []
            for trainer in selected_trainers:
                acc_ref = trainer.calculate_weighted_weight.remote(k)
                acc_refs.append(acc_ref)
            while acc_refs:
                ready, left = ray.wait(acc_refs, num_returns=1, timeout=None)
                if ready:
                    for t in ready:
                        weighted_weight = ray.get(t)
                        accumulate_list.append(weighted_weight)
                acc_refs = left
            accumulate = torch.stack(accumulate_list)
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
        ks = self.W.keys()
        for cluster in trainer_clusters:  # cluster is a list of trainer objects
            weights_list = ray.get(
                [trainer.get_weights.remote(ks) for trainer in cluster]
            )
            # Unpack the list of dictionaries into separate lists for targs, sours, and train_sizes
            targs = [weights["W"] for weights in weights_list]
            sours = [(weights["dW"], weights["train_size"]) for weights in weights_list]
            total_size = sum([weights["train_size"] for weights in weights_list])
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
        dw_refs = []
        for trainer in cluster:
            dw_ref = trainer.compute_update_norm.remote(self.W.keys())
            dw_refs.append(dw_ref)
        results = ray.get(dw_refs)

        max_dW = max(results)

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
        dw_refs = []

        total_size = sum(ray.get([c.get_train_size.remote() for c in cluster]))
        for trainer in cluster:
            dw_ref = trainer.compute_mean_norm.remote(total_size, self.W.keys())
            dw_refs.append(dw_ref)
        cluster_dWs = ray.get(dw_refs)
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


class Server_GAT:
    """
    Server class for federated learning, responsible for aggregating model parameters
    from different trainers, updating the central model, and broadcasting updated parameters
    back to the trainers.

    Parameters
    ----------
    feature_dim : int
        Dimensionality of the feature vectors in the dataset.
    hidden_dim : int
        Number of hidden units.
    class_num : int
        Number of classes for classification in the dataset.
    device : torch.device
        Device initialized for the server model.
    trainers : list[Trainer_GAT]
        List of `Trainer_GAT` instances representing the trainers.
    """

    def __init__(
        self,
        graph,
        feats,
        labels,
        feature_dim,
        class_num,
        device,
        trainers,
        args,
    ):
        self.graph = graph
        self.feats = feats
        self.labels = labels

        self.in_feat = feats.size()[1]
        self.out_feat = labels.size()[1]

        self.device = device
        self.trainers = trainers
        self.sample_probab = args.sample_probab
        self.train_rounds = args.train_rounds
        self.num_local_iters = args.num_local_iters
        self.dual_weight = args.dual_weight
        self.aug_lagrange_rho = args.aug_lagrange_rho
        self.dual_lr = args.dual_lr
        self.model_lr = args.model_lr
        self.model_regularisation = args.model_regularisation
        self.total_communicate = 0

        self.model = FedGATModel(
            in_feat=feature_dim,
            out_feat=class_num,
            hidden_dim=args.hidden_dim,
            num_head=args.num_heads,
            max_deg=args.max_deg,
            attn_func=args.attn_func_parameter,
            domain=args.attn_func_domain,
            device=device,
        ).to(device)
        self.GATModelParams = self.model.state_dict()
        self.Duals = {
            client_id: copy.deepcopy(self.model.state_dict())
            for client_id in range(len(self.trainers))
        }
        self.LocalModelParams = {}

        self.broadcast_params(-1)

    @torch.no_grad()
    def zero_params(self):
        """Zeros out the parameters of the central model."""
        for p in self.model.parameters():
            p.zero_()

    @torch.no_grad()
    def train(self, current_global_epoch):
        """
        Perform a training round: aggregate parameters from trainers, update the central model,
        and broadcast updated parameters back to the trainers.

        Parameters
        ----------
        current_global_epoch : int
            Current global epoch number during the federated learning process.
        """
        for trainer in self.trainers:
            trainer.train.remote(current_global_epoch)

        params = [trainer.get_params.remote() for trainer in self.trainers]
        self.zero_params()

        while params:
            ready, left = ray.wait(params, num_returns=1, timeout=None)
            if ready:
                for t in ready:
                    for p, mp in zip(ray.get(t), self.model.parameters()):
                        mp.data += p.cpu()
            params = left

        for p in self.model.parameters():
            p /= self.num_of_trainers
        self.broadcast_params(current_global_epoch)

    def broadcast_params(self, current_global_epoch):
        """
        Broadcast the current parameters of the central model to all trainers.

        Parameters
        ----------
        current_global_epoch : int
            Current global epoch number during the federated learning process.
        """
        for trainer in self.trainers:
            trainer.update_params.remote(
                tuple(self.model.parameters()), current_global_epoch
            )

    def get_neighbours(self, node_id, edge_index):
        mask = edge_index[0] == node_id
        neighbours = edge_index[1][mask]
        return neighbours

    def get_predecessors(self, data, node):
        edge_index = data.edge_index
        mask = edge_index[1] == node
        predecessors = edge_index[0, mask]
        return predecessors

    def pretrain_communication(self, communicate_node_indexes, graph, device):
        for client_id, communicate_node_index in enumerate(communicate_node_indexes):
            return_info = {}

            if self.Duals.get(client_id, None) is None:
                self.num_clients += 1

            self.Duals[client_id] = copy.deepcopy(self.model.state_dict())

            for param in self.Duals[client_id]:
                self.Duals[client_id][param].requires_grad = False

            self.LocalModelParams[client_id] = None

            for node in communicate_node_index:
                neighbours = self.get_predecessors(graph, node)

                temp = [
                    random.choices(
                        [0, 1], [1 - self.sample_probab, self.sample_probab]
                    )[0]
                    for j in range(len(neighbours))
                ]

                neigh = [neighbours[j] for j in range(len(neighbours)) if temp[j] == 1]

                if len(neigh) < 2:
                    neigh = [j for j in neighbours]

                num = len(neigh)

                orth_vec = MatGen(num)

                M1 = np.zeros((self.in_feat, 2 * num, 2 * num))
                M2 = np.zeros((self.in_feat, 2 * num, 2 * num))
                Q2 = np.zeros((2 * num, self.in_feat))
                Q1 = 2 * np.sum(orth_vec[0:num, :], axis=0)

                main_node_feat = self.feats[node, :].detach().cpu().numpy()

                for j in range(len(neigh)):
                    node_id = neigh[j]
                    node_feat = self.feats[node_id, :].detach().cpu().numpy()

                    Q2 += np.outer(orth_vec[j, :], node_feat)

                    for d in range(self.in_feat):
                        M1[d, :, :] += (
                            main_node_feat[d]
                            * 0.5
                            * (
                                np.outer(orth_vec[j, :], orth_vec[j, :])
                                + np.outer(orth_vec[j + num, :], orth_vec[j + num, :])
                                + 4 * np.outer(orth_vec[j, :], orth_vec[j + num, :])
                                + 0.25 * np.outer(orth_vec[j + num, :], orth_vec[j, :])
                            )
                        )

                        M2[d, :, :] += (
                            node_feat[d]
                            * 0.5
                            * (
                                np.outer(orth_vec[j, :], orth_vec[j, :])
                                + np.outer(orth_vec[j + num, :], orth_vec[j + num, :])
                                + 4 * np.outer(orth_vec[j, :], orth_vec[j + num, :])
                                + 0.25 * np.outer(orth_vec[j + num, :], orth_vec[j, :])
                            )
                        )

                return_info[node] = [
                    torch.from_numpy(M1).float().to(device=device),
                    torch.from_numpy(M2).float().to(device=device),
                    torch.from_numpy(Q1).float().to(device=device),
                    torch.from_numpy(Q2).float().to(device=device),
                ]

            count = 0
            for node in return_info:
                for j in range(len(return_info[node])):
                    count += torch.numel(return_info[node][j])
                    return_info[node][j].to(device=device)

            self.total_communicate += count

            # return_info
            self.trainers[client_id].setNodeMats.remote(return_info)

    def _calculate_matrix(self, node_feat, orth_vec, j):
        num = orth_vec.shape[0] // 2
        matrix = np.zeros((self.in_feat, 2 * num, 2 * num))
        for d in range(self.in_feat):
            matrix[d, :, :] += (
                node_feat[d]
                * 0.5
                * (
                    np.outer(orth_vec[j], orth_vec[j])
                    + np.outer(orth_vec[j + num], orth_vec[j + num])
                    + 4 * np.outer(orth_vec[j], orth_vec[j + num])
                    + 0.25 * np.outer(orth_vec[j + num], orth_vec[j])
                )
            )
        return matrix

    def TrainingUpdate(self):
        """
        Perform a global parameter update by aggregating local models and updating dual variables.
        """
        print("Global parameter update")
        with torch.no_grad():
            futures = [
                self.trainers[client_id].get_model_state_dict.remote()
                for client_id in range(len(self.trainers))
            ]

            results = ray.get(futures)

            self.LocalModelParams = {
                client_id: results[client_id] for client_id in range(len(self.trainers))
            }

            for param in self.GATModelParams:
                self.GATModelParams[param].zero_()

                for client_id in self.LocalModelParams:
                    self.GATModelParams[param] += self.LocalModelParams[client_id][
                        param
                    ] - self.dual_weight * self.Duals[client_id][param] / (
                        2 * self.aug_lagrange_rho
                    )

                self.GATModelParams[param] /= len(self.trainers)

            for client_id in self.Duals:
                for param in self.Duals[client_id]:
                    self.Duals[client_id][param] += (
                        self.dual_weight
                        * self.dual_lr
                        * (
                            self.GATModelParams[param]
                            - self.LocalModelParams[client_id][param]
                        )
                    )

            global_variance = self._calculate_global_variance()
            print(f"Global vs local parameter variance = {global_variance}")

    def _calculate_global_variance(self):
        global_variance = 0.0
        for param in self.GATModelParams:
            for client_id in self.LocalModelParams:
                global_variance += torch.norm(
                    self.GATModelParams[param] - self.LocalModelParams[client_id][param]
                )
        return global_variance / (len(self.trainers) * len(self.GATModelParams))


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
