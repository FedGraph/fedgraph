import argparse
import random
import time
from typing import Any, List, Union
from torch_geometric.data import Data
import numpy as np
import ray
import torch
import torch.nn.functional as F
import torch_geometric
from torchmetrics.functional.retrieval import retrieval_auroc
from torchmetrics.retrieval import RetrievalHitRate
from fedgraph.gnn_models import GCN, GIN, GNN_LP, AggreGCN, GCN_arxiv, SAGE_products, LocSAGEPlus
from fedgraph.train_func import test, train, accuracy
from fedgraph.utils_lp import (
    check_data_files_existance,
    get_data,
    get_data_loaders_per_time_step,
    get_global_user_item_mapping,
)
from fedgraph.utils_nc import get_1hop_feature_sum, greedy_loss_fedsage_plus
from train_func import accuracy, accuracy_missing_node_number
from data_process import get_subgraph_pyg_data


class Trainer_General:
    """
    A general trainer class for training GCN in a federated learning setup, which includes functionalities
    required for training GCN models on a subset of a distributed dataset, handling local training and testing,
    parameter updates, and feature aggregation.

    Parameters
    ----------
    rank : int
        Unique identifier for the training instance (typically representing a trainer in federated learning).
    local_node_index : torch.Tensor
        Indices of nodes local to this trainer.
    communicate_node_index : torch.Tensor
        Indices of nodes that participate in communication during training.
    adj : torch.Tensor
        The adjacency matrix representing the graph structure.
    train_labels : torch.Tensor
        Labels of the training data.
    test_labels : torch.Tensor
        Labels of the testing data.
    features : torch.Tensor
        Node features for the entire graph.
    idx_train : torch.Tensor
        Indices of training nodes.
    idx_test : torch.Tensor
        Indices of test nodes.
    args_hidden : int
        Number of hidden units in the GCN model.
    global_node_num : int
        Total number of nodes in the global graph.
    class_num : int
        Number of classes for classification.
    device : torch.device
        The device (CPU or GPU) on which the model will be trained.
    args : Any
        Additional arguments required for model initialization and training.
    """

    def __init__(
        self,
        rank: int,
        local_node_index: torch.Tensor,
        communicate_node_index: torch.Tensor,
        adj: torch.Tensor,
        train_labels: torch.Tensor,
        test_labels: torch.Tensor,
        features: torch.Tensor,
        idx_train: torch.Tensor,
        idx_test: torch.Tensor,
        args_hidden: int,
        global_node_num: int,
        class_num: int,
        device: torch.device,
        args: Any,
    ):
        # from gnn_models import GCN_Graph_Classification
        torch.manual_seed(rank)

        # seems that new trainer process will not inherit sys.path from parent, need to reimport!
        if args.num_hops >= 1 and args.fedtype == "fedgcn":
            self.model = AggreGCN(
                nfeat=features.shape[1],
                nhid=args_hidden,
                nclass=class_num,
                dropout=0.5,
                NumLayers=args.num_layers,
            ).to(device)
        else:
            if args.dataset == "ogbn-arxiv":
                self.model = GCN_arxiv(
                    nfeat=features.shape[1],
                    nhid=args_hidden,
                    nclass=class_num,
                    dropout=0.5,
                    NumLayers=args.num_layers,
                ).to(device)
            elif args.dataset == "ogbn-products":
                self.model = SAGE_products(
                    nfeat=features.shape[1],
                    nhid=args_hidden,
                    nclass=class_num,
                    dropout=0.5,
                    NumLayers=args.num_layers,
                ).to(device)
            else:
                self.model = GCN(
                    nfeat=features.shape[1],
                    nhid=args_hidden,
                    nclass=class_num,
                    dropout=0.5,
                    NumLayers=args.num_layers,
                ).to(device)

        self.rank = rank  # rank = trainer ID

        self.device = device

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=args.learning_rate, weight_decay=5e-4
        )

        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_losses: list = []
        self.train_accs: list = []

        self.test_losses: list = []
        self.test_accs: list = []

        self.local_node_index = local_node_index.to(device)
        self.communicate_node_index = communicate_node_index.to(device)

        self.adj = adj.to(device)
        self.train_labels = train_labels.to(device)
        self.test_labels = test_labels.to(device)
        self.features = features.to(device)
        self.idx_train = idx_train.to(device)
        self.idx_test = idx_test.to(device)

        self.local_step = args.local_step
        self.global_node_num = global_node_num
        self.class_num = class_num

    @torch.no_grad()
    def update_params(self, params: tuple, current_global_epoch: int) -> None:
        """
        Updates the model parameters with global parameters received from the server.

        Parameters
        ----------
        params : tuple
            A tuple containing the global parameters from the server.
        current_global_epoch : int
            The current global epoch number.
        """
        # load global parameter from global server
        self.model.to("cpu")
        for (
            p,
            mp,
        ) in zip(params, self.model.parameters()):
            mp.data = p
        self.model.to(self.device)

    def get_local_feature_sum(self) -> torch.Tensor:
        """
        Computes the sum of features of all 1-hop neighbors for each node.

        Returns
        -------
        one_hop_neighbor_feature_sum : torch.Tensor
            The sum of features of 1-hop neighbors for each node
        """

        # create a large matrix with known local node features
        new_feature_for_trainer = torch.zeros(
            self.global_node_num, self.features.shape[1]
        ).to(
            self.device
        )  # TODO:check if all the tensors are in the same device
        new_feature_for_trainer[self.local_node_index] = self.features
        # sum of features of all 1-hop nodes for each node
        one_hop_neighbor_feature_sum = get_1hop_feature_sum(
            new_feature_for_trainer, self.adj
        )
        return one_hop_neighbor_feature_sum

    def load_feature_aggregation(self, feature_aggregation: torch.Tensor) -> None:
        """
        Loads the aggregated features into the trainer.

        Parameters
        ----------
        feature_aggregation : torch.Tensor
            The aggregated features to be loaded.
        """
        self.feature_aggregation = feature_aggregation

    def relabel_adj(self) -> None:
        """
        Relabels the adjacency matrix based on the communication node index.
        """
        _, self.adj, __, ___ = torch_geometric.utils.k_hop_subgraph(
            self.communicate_node_index, 0, self.adj, relabel_nodes=True
        )

    def train(self, current_global_round: int) -> None:
        """
        Performs local training for a specified number of iterations. This method
        updates the model using the loaded feature aggregation and the adjacency matrix.

        Parameters
        ----------
        current_global_round : int
            The current global training round.
        """
        # clean cache
        torch.cuda.empty_cache()
        self.model.to(self.device)
        self.feature_aggregation = self.feature_aggregation.to(self.device)
        for iteration in range(self.local_step):
            self.model.train()
            loss_train, acc_train = train(
                iteration,
                self.model,
                self.optimizer,
                self.feature_aggregation,
                self.adj,
                self.train_labels,
                self.idx_train,
            )
            self.train_losses.append(loss_train)
            self.train_accs.append(acc_train)

            loss_test, acc_test = self.local_test()
            self.test_losses.append(loss_test)
            self.test_accs.append(acc_test)

    def local_test(self) -> list:
        """
        Evaluates the model on the local test dataset.

        Returns
        -------
        (list) : list
            A list containing the test loss and accuracy [local_test_loss, local_test_acc].
        """
        local_test_loss, local_test_acc = test(
            self.model,
            self.feature_aggregation,
            self.adj,
            self.test_labels,
            self.idx_test,
        )
        return [local_test_loss, local_test_acc]

    def get_params(self) -> tuple:
        """
        Retrieves the current parameters of the model.

        Returns
        -------
        (tuple) : tuple
            A tuple containing the current parameters of the model.
        """
        self.optimizer.zero_grad(set_to_none=True)
        return tuple(self.model.parameters())

    def get_all_loss_accuray(self) -> list:
        """
        Returns all recorded training and testing losses and accuracies.

        Returns
        -------
        (list) : list
            A list containing arrays of training losses, training accuracies, testing losses, and testing accuracies.
        """
        return [
            np.array(self.train_losses),
            np.array(self.train_accs),
            np.array(self.test_losses),
            np.array(self.test_accs),
        ]

    def get_rank(self) -> int:
        """
        Returns the rank (trainer ID) of the trainer.

        Returns
        -------
        (int) : int
            The rank (trainer ID) of this trainer instance.
        """
        return self.rank


class Trainer_GC:
    """
    A trainer class specified for graph classification tasks, which includes functionalities required
    for training GIN models on a subset of a distributed dataset, handling local training and testing,
    parameter updates, and feature aggregation.

    Parameters
    ----------
    model: object
        The model to be trained, which is based on the GIN model.
    trainer_id: int
        The ID of the trainer.
    trainer_name: str
        The name of the trainer.
    train_size: int
        The size of the training dataset.
    dataLoader: dict
        The dataloaders for training, validation, and testing.
    optimizer: object
        The optimizer for training.
    args: Any
        The arguments for the training.

    Attributes
    ----------
    model: object
        The model to be trained, which is based on the GIN model.
    id: int
        The ID of the trainer.
    name: str
        The name of the trainer.
    train_size: int
        The size of the training dataset.
    dataloader: dict
        The dataloaders for training, validation, and testing.
    optimizer: object
        The optimizer for training.
    args: object
        The arguments for the training.
    W: dict
        The weights of the model.
    dW: dict
        The gradients of the model.
    W_old: dict
        The cached weights of the model.
    gconv_names: list
        The names of the gconv layers.
    train_stats: Any
        The training statistics of the model.
    weights_norm: float
        The norm of the weights of the model.
    grads_norm: float
        The norm of the gradients of the model.
    conv_grads_norm: float
        The norm of the gradients of the gconv layers.
    conv_weights_Norm: float
        The norm of the weights of the gconv layers.
    conv_dWs_norm: float
        The norm of the gradients of the gconv layers.
    """

    def __init__(
        self,
        model: Any,
        trainer_id: int,
        trainer_name: str,
        train_size: int,
        dataloader: dict,
        optimizer: object,
        args: Any,
    ) -> None:
        self.model = model.to(args.device)
        self.id = trainer_id
        self.name = trainer_name
        self.train_size = train_size
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.args = args

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.dW = {
            key: torch.zeros_like(value) for key, value in self.model.named_parameters()
        }
        self.W_old = {
            key: value.data.clone() for key, value in self.model.named_parameters()
        }

        self.gconv_names: Any = None

        self.train_stats: dict[str, list[Any]] = {
            "trainingAccs": [],
            "valAccs": [],
            "trainingLosses": [],
            "valLosses": [],
            "testAccs": [],
            "testLosses": [],
        }
        self.weights_norm = 0.0
        self.grads_norm = 0.0
        self.conv_grads_norm = 0.0
        self.conv_weights_norm = 0.0
        self.conv_dWs_norm = 0.0

    ########### Public functions ###########
    def update_params(self, server_params: Any) -> None:
        """
        Update the model parameters by downloading the global model weights from the server.

        Parameters
        ----------
        server: Server_GC
            The server object that contains the global model weights.
        """
        self.gconv_names = server_params.keys()  # gconv layers
        for k in server_params:
            self.W[k].data = server_params[k].data.clone()

    def reset_params(self) -> None:
        """
        Reset the weights of the model to the cached weights.
        The implementation is copying the cached weights (W_old) to the model weights (W).

        """
        self.__copy_weights(target=self.W, source=self.W_old, keys=self.gconv_names)

    def cache_weights(self) -> None:
        """
        Cache the weights of the model.
        The implementation is copying the model weights (W) to the cached weights (W_old).
        """
        for name in self.W.keys():
            self.W_old[name].data = self.W[name].data.clone()

    def compute_update_norm(self, keys: dict) -> float:
        """
        Compute the max update norm (i.e., dW) for the trainer
        """
        dW = {}
        for k in keys:
            dW[k] = self.dW[k]

        curr_dW = torch.norm(
            torch.cat([value.flatten() for value in dW.values()])
        ).item()

        return curr_dW

    def compute_mean_norm(self, total_size: int, keys: dict) -> torch.Tensor:
        """
        Compute the mean update norm (i.e., dW) for the trainer
        Returns
        -------
        curr_dW: Tensor
        """
        dW = {}
        for k in keys:
            dW[k] = self.dW[k] * self.train_size / total_size

        curr_dW = torch.cat([value.flatten() for value in dW.values()])

        return curr_dW

    def set_stats_norms(self, train_stats: Any, is_gcfl: bool = False) -> None:
        """
        Set the norms of the weights and gradients of the model, as well as the statistics of the training.

        Parameters
        ----------
        train_stats: dict
            The training statistics of the model.
        is_gcfl: bool, optional
            Whether the training is for GCFL. The default is False.
        """
        self.train_stats = train_stats

        self.weights_norm = torch.norm(self.__flatten(self.W)).item()

        if self.gconv_names is not None:
            weights_conv = {key: self.W[key] for key in self.gconv_names}
            self.conv_weights_norm = torch.norm(self.__flatten(weights_conv)).item()

            grads_conv = {key: self.W[key].grad for key in self.gconv_names}
            self.conv_grads_norm = torch.norm(self.__flatten(grads_conv)).item()

        grads = {key: value.grad for key, value in self.W.items()}
        self.grads_norm = torch.norm(self.__flatten(grads)).item()

        if is_gcfl and self.gconv_names is not None:
            dWs_conv = {key: self.dW[key] for key in self.gconv_names}
            self.conv_dWs_norm = torch.norm(self.__flatten(dWs_conv)).item()

    def local_train(
        self, local_epoch: int, train_option: str = "basic", mu: float = 1
    ) -> None:
        """
        This function is a interface of the trainer class to train the model locally.
        It will call the train function specified for the training option, based on the args provided.

        Parameters
        ----------
        local_epoch: int
            The number of local epochs
        train_option: str, optional
            The training option. The possible values are 'basic', 'prox', and 'gcfl'. The default is 'basic'.
            'basic' - self-train and FedAvg
            'prox' - FedProx that includes the proximal term
            'gcfl' - GCFL, GCFL+ and GCFL+dWs
        mu: float, optional
            The proximal term. The default is 1.
        """
        assert train_option in ["basic", "prox", "gcfl"], "Invalid training option."

        if train_option == "gcfl":
            self.__copy_weights(target=self.W_old, source=self.W, keys=self.gconv_names)

        if train_option in ["basic", "prox"]:
            train_stats = self.__train(
                model=self.model,
                dataloaders=self.dataloader,
                optimizer=self.optimizer,
                local_epoch=local_epoch,
                device=self.args.device,
            )
        elif train_option == "gcfl":
            train_stats = self.__train(
                model=self.model,
                dataloaders=self.dataloader,
                optimizer=self.optimizer,
                local_epoch=local_epoch,
                device=self.args.device,
                prox=True,
                gconv_names=self.gconv_names,
                Ws=self.W,
                Wt=self.W_old,
                mu=mu,
            )

        if train_option == "gcfl":
            self.__subtract_weights(
                target=self.dW, minuend=self.W, subtrahend=self.W_old
            )
        self.set_stats_norms(train_stats)

    def local_test(self, test_option: str = "basic", mu: float = 1) -> tuple:
        """
        Final test of the model on the test dataset based on the test option.

        Parameters
        ----------
        test_option: str, optional
            The test option. The possible values are 'basic' and 'prox'. The default is 'basic'.
            'basic' - self-train, FedAvg, GCFL, GCFL+ and GCFL+dWs
            'prox' - FedProx that includes the proximal term
        mu: float, optional
            The proximal term. The default is 1.

        Returns
        -------
        (test_loss, test_acc, trainer_name, trainingAccs, valAccs): tuple(float, float, string, float, float)
            The average loss and accuracy, trainer's name, trainer.train_stats["trainingAccs"][-1], trainer.train_stats["valAccs"][-1]
        """
        assert test_option in ["basic", "prox"], "Invalid test option."
        if test_option == "basic":
            return self.__eval(
                model=self.model,
                test_loader=self.dataloader["test"],
                device=self.args.device,
            )
        elif test_option == "prox":
            return self.__eval(
                model=self.model,
                test_loader=self.dataloader["test"],
                device=self.args.device,
                prox=True,
                gconv_names=self.gconv_names,
                mu=mu,
                Wt=self.W_old,
            )
        else:
            raise ValueError("Invalid test option.")

    def get_train_size(self) -> int:
        return self.train_size

    def get_weights(self, ks: Any) -> dict[str, Any]:
        data: dict[str, Any] = {}
        W = {}
        dW = {}
        for k in ks:
            W[k], dW[k] = self.W[k], self.dW[k]
        data["W"] = W
        data["dW"] = dW
        data["train_size"] = self.train_size
        return data

    def get_total_weight(self) -> Any:
        return self.W

    def get_name(self) -> str:
        return self.name

    ########### Private functions ###########
    def __train(
        self,
        model: Any,
        dataloaders: dict,
        optimizer: Any,
        local_epoch: int,
        device: str,
        prox: bool = False,
        gconv_names: Any = None,
        Ws: Any = None,
        Wt: Any = None,
        mu: float = 0,
    ) -> dict:
        """
        Train the model on the local dataset.

        Parameters
        ----------
        model: object
            The model to be trained
        dataloaders: dict
            The dataloaders for training, validation, and testing
        optimizer: Any
            The optimizer for training
        local_epoch: int
            The number of local epochs
        device: str
            The device to run the training
        prox: bool, optional
            Whether to add the proximal term. The default is False.
        gconv_names: Any, optional
            The names of the gconv layers. The default is None.
        Ws: Any, optional
            The weights of the model. The default is None.
        Wt: Any, optional
            The target weights. The default is None.
        mu: float, optional
            The proximal term. The default is 0.

        Returns
        -------
        (results): dict
            The training statistics

        Note
        ----
        If prox is True, the function will add the proximal term to the loss function.
        Make sure to provide the required arguments `gconv_names`, `Ws`, `Wt`, and `mu` for the proximal term.
        """
        if prox:
            assert (
                (gconv_names is not None)
                and (Ws is not None)
                and (Wt is not None)
                and (mu != 0)
            ), "Please provide the required arguments for the proximal term."

        losses_train, accs_train, losses_val, accs_val, losses_test, accs_test = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        if prox:
            convGradsNorm = []
        train_loader, val_loader, test_loader = (
            dataloaders["train"],
            dataloaders["val"],
            dataloaders["test"],
        )

        for _ in range(local_epoch):
            model.train()
            loss_train, acc_train, num_graphs = 0.0, 0.0, 0

            for _, batch in enumerate(train_loader):
                batch.to(device)
                optimizer.zero_grad()
                pred = model(batch)
                label = batch.y
                loss = model.loss(pred, label)
                loss += (
                    mu / 2.0 * self.__prox_term(model, gconv_names, Wt) if prox else 0.0
                )  # add the proximal term if required
                loss.backward()
                optimizer.step()
                loss_train += loss.item() * batch.num_graphs
                acc_train += pred.max(dim=1)[1].eq(label).sum().item()
                num_graphs += batch.num_graphs

            loss_train /= num_graphs  # get the average loss per graph
            acc_train /= num_graphs  # get the average average per graph

            loss_val, acc_val, _, _, _ = self.__eval(model, val_loader, device)
            loss_test, acc_test, _, _, _ = self.__eval(model, test_loader, device)

            losses_train.append(loss_train)
            accs_train.append(acc_train)
            losses_val.append(loss_val)
            accs_val.append(acc_val)
            losses_test.append(loss_test)
            accs_test.append(acc_test)

            if prox:
                convGradsNorm.append(self.__calc_grads_norm(gconv_names, Ws))

        # record the losses and accuracies for each epoch
        res_dict = {
            "trainingLosses": losses_train,
            "trainingAccs": accs_train,
            "valLosses": losses_val,
            "valAccs": accs_val,
            "testLosses": losses_test,
            "testAccs": accs_test,
        }
        if prox:
            res_dict["convGradsNorm"] = convGradsNorm

        return res_dict

    def __eval(
        self,
        model: GIN,
        test_loader: Any,
        device: str,
        prox: bool = False,
        gconv_names: Any = None,
        mu: float = 0,
        Wt: Any = None,
    ) -> tuple:
        """
        Validate and test the model on the local dataset.

        Parameters
        ----------
        model: GIN
            The model to be tested
        test_loader: Any
            The dataloader for testing
        device: str
            The device to run the testing
        prox: bool, optional
            Whether to add the proximal term. The default is False.
        gconv_names: Any, optional
            The names of the gconv layers. The default is None.
        mu: float, optional
            The proximal term. The default is None.
        Wt: Any, optional
            The target weights. The default is None.

        Returns
        -------
        (test_loss, test_acc, trainer_name, trainingAccs, valAccs): tuple(float, float, string, float, float)
            The average loss and accuracy, trainer's name, trainer.train_stats["trainingAccs"][-1], trainer.train_stats["valAccs"][-1]

        Note
        ----
        If prox is True, the function will add the proximal term to the loss function.
        Make sure to provide the required arguments `gconv_names`, `Ws`, `Wt`, and `mu` for the proximal term.
        """
        if prox:
            assert (
                (gconv_names is not None) and (mu is not None) and (Wt != 0)
            ), "Please provide the required arguments for the proximal term."

        model.eval()
        total_loss, total_acc, num_graphs = 0.0, 0.0, 0

        for batch in test_loader:
            batch.to(device)
            with torch.no_grad():
                pred = model(batch)
                label = batch.y
                loss = model.loss(pred, label)
                loss += (
                    mu / 2.0 * self.__prox_term(model, gconv_names, Wt) if prox else 0.0
                )

            total_loss += loss.item() * batch.num_graphs
            total_acc += pred.max(dim=1)[1].eq(label).sum().item()
            num_graphs += batch.num_graphs

        current_training_acc = -1
        current_val_acc = -1
        if self.train_stats["trainingAccs"]:
            current_training_acc = self.train_stats["trainingAccs"][-1]
        if self.train_stats["valAccs"]:
            current_val_acc = self.train_stats["valAccs"][-1]

        return (
            total_loss / num_graphs,
            total_acc / num_graphs,
            self.name,
            current_training_acc,  # if no data then return -1 for 1st train round
            current_val_acc,  # if no data then return -1 for 1st train round
        )

    def __prox_term(self, model: Any, gconv_names: Any, Wt: Any) -> torch.tensor:
        """
        Compute the proximal term.

        Parameters
        ----------
        model: Any
            The model to be trained
        gconv_names: Any
            The names of the gconv layers
        Wt: Any
            The target weights

        Returns
        -------
        prox: torch.tensor
            The proximal term
        """
        prox = torch.tensor(0.0, requires_grad=True)
        for name, param in model.named_parameters():
            # only add the prox term for sharing layers (gConv)
            if name in gconv_names:
                prox = prox + torch.norm(param - Wt[name]).pow(
                    2
                )  # force the weights to be close to the old weights
        return prox

    def __calc_grads_norm(self, gconv_names: Any, Ws: Any) -> float:
        """
        Calculate the norm of the gradients of the gconv layers.

        Parameters
        ----------
        model: Any
            The model to be trained
        gconv_names: Any
            The names of the gconv layers
        Wt: Any
            The target weights

        Returns
        -------
        convGradsNorm: float
            The norm of the gradients of the gconv layers
        """
        grads_conv = {k: Ws[k].grad for k in gconv_names}
        convGradsNorm = torch.norm(self.__flatten(grads_conv)).item()
        return convGradsNorm

    def __copy_weights(
        self, target: dict, source: dict, keys: Union[list, None]
    ) -> None:
        """
        Copy the source weights to the target weights.

        Parameters
        ----------
        target: dict
            The target weights
        source: dict
            The source weights
        keys: list, optional
            The keys to be copied. The default is None.
        """
        if keys is not None:
            for name in keys:
                target[name].data = source[name].data.clone()

    def __subtract_weights(self, target: dict, minuend: dict, subtrahend: dict) -> None:
        """
        Subtract the subtrahend from the minuend and store the result in the target.

        Parameters
        ----------
        target: dict
            The target weights
        minuend: dict
            The minuend
        subtrahend: dict
            The subtrahend
        """
        for name in target:
            target[name].data = (
                minuend[name].data.clone() - subtrahend[name].data.clone()
            )

    def __flatten(self, w: dict) -> torch.tensor:
        """
        Flatten the gradients of a trainer into a 1D tensor.

        Parameters
        ----------
        w: dict
            The gradients of a trainer
        """
        return torch.cat([v.flatten() for v in w.values()])

    def calculate_weighted_weight(self, key: Any) -> torch.tensor:
        weighted_weight = torch.mul(self.W[key].data, self.train_size)
        return weighted_weight


class Trainer_LP:
    """
    A trainer class specified for graph link prediction tasks, which includes functionalities required
    for training GNN models on a subset of a distributed dataset, handling local training and testing,
    parameter updates, and feature aggregation.

    Parameters
    ----------
    client_id : int
        The ID of the client.
    country_code : str
        The country code of the client. Each client is associated with one country code.
    user_id_mapping : dict
        The mapping of user IDs.
    item_id_mapping : dict
        The mapping of item IDs.
    number_of_users : int
        The number of users.
    number_of_items : int
        The number of items.
    meta_data : tuple
        The metadata of the dataset.
    hidden_channels : int, optional
        The number of hidden channels in the GNN model. The default is 64.
    """

    def __init__(
        self,
        client_id: int,
        country_code: str,
        user_id_mapping: dict,
        item_id_mapping: dict,
        number_of_users: int,
        number_of_items: int,
        meta_data: tuple,
        hidden_channels: int = 64,
    ):
        self.client_id = client_id
        self.country_code = country_code
        file_path = f"fedgraph/data/LPDataset"
        print("checking code and file path")
        print(country_code)
        print(file_path)
        country_codes: List[str] = [self.country_code]
        check_data_files_existance(country_codes, file_path)
        # global user_id and item_id
        self.data = get_data(self.country_code, user_id_mapping, item_id_mapping)

        self.model = GNN_LP(
            number_of_users, number_of_items, meta_data, hidden_channels
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: '{self.device}'")
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_train_test_data_at_current_time_step(
        self,
        start_time_float_format: float,
        end_time_float_format: float,
        use_buffer: bool = False,
        buffer_size: int = 10,
    ) -> None:
        """
        Get the training and testing data at the current time step.

        Parameters
        ----------
        start_time_float_format : float
            The start time in float format.
        end_time_float_format : float
            The end time in float format.
        use_buffer : bool, optional
            Whether to use the buffer. The default is False.
        buffer_size : int, optional
            The size of the buffer. The default is 10.
        """
        print("loading buffer_train_data_list") if use_buffer else print(
            "loading train_data and test_data"
        )

        load_res = get_data_loaders_per_time_step(
            self.data,
            start_time_float_format,
            end_time_float_format,
            use_buffer,
            buffer_size,
        )

        if use_buffer:
            (
                self.global_train_data,
                self.test_data,
                self.buffer_train_data_list,
            ) = load_res
        else:
            self.train_data, self.test_data = load_res

    def train(
        self, client_id: int, local_updates: int, use_buffer: bool = False
    ) -> tuple:
        """
        Perform local training for a specified number of iterations.

        Parameters
        ----------
        local_updates : int
            The number of local updates.
        use_buffer : bool, optional
            Whether to use the buffer. The default is False.

        Returns
        -------
        (loss, train_finish_times) : tuple
            [0] The loss of the model
            [1] The time taken for each local update
        """
        train_finish_times = []
        if use_buffer:
            probabilities = [1 / len(self.buffer_train_data_list)] * len(
                self.buffer_train_data_list
            )

        for i in range(local_updates):
            if use_buffer:
                train_data = random.choices(
                    self.buffer_train_data_list, weights=probabilities, k=1
                )[0].to(self.device)
            else:
                train_data = self.train_data.to(self.device)

            start_train_time = time.time()

            self.optimizer.zero_grad()
            pred = self.model(train_data)
            ground_truth = train_data["user", "select", "item"].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            self.optimizer.step()

            train_finish_time = time.time() - start_train_time
            train_finish_times.append(train_finish_time)
            print(
                f"client {self.client_id} local update {i} loss {loss:.4f} train time {train_finish_time:.4f}"
            )

        return client_id, loss, train_finish_times

    def test(self, clientId: int, use_buffer: bool = False) -> tuple:
        """
        Test the model on the test data.

        Parameters
        ----------
        use_buffer : bool, optional
            Whether to use the buffer. The default is False.

        Returns
        -------
        (auc, hit_rate_at_2, traveled_user_hit_rate_at_2) : tuple
            [0] The AUC score
            [1] The hit rate at 2
            [2] The hit rate at 2 for traveled users
        """
        preds, ground_truths = [], []
        self.test_data.to(self.device)
        with torch.no_grad():
            if not use_buffer:
                self.train_data.to(self.device)
                preds.append(self.model.pred(self.train_data, self.test_data))
            else:
                self.global_train_data.to(self.device)
                preds.append(self.model.pred(self.global_train_data, self.test_data))
            ground_truths.append(self.test_data["user", "select", "item"].edge_label)

        pred = torch.cat(preds, dim=0)
        ground_truth = torch.cat(ground_truths, dim=0)
        auc = retrieval_auroc(pred, ground_truth)
        hit_rate_evaluator = RetrievalHitRate(top_k=2)
        hit_rate_at_2 = hit_rate_evaluator(
            pred,
            ground_truth,
            indexes=self.test_data["user", "select", "item"].edge_label_index[0],
        )
        traveled_user_hit_rate_at_2 = hit_rate_evaluator(
            pred[self.traveled_user_edge_indices],
            ground_truth[self.traveled_user_edge_indices],
            indexes=self.test_data["user", "select", "item"].edge_label_index[0][
                self.traveled_user_edge_indices
            ],
        )
        print(f"Test AUC: {auc:.4f}")
        print(f"Test Hit Rate at 2: {hit_rate_at_2:.4f}")
        print(f"Test Traveled User Hit Rate at 2: {traveled_user_hit_rate_at_2:.4f}")
        return clientId, auc, hit_rate_at_2, traveled_user_hit_rate_at_2

    def calculate_traveled_user_edge_indices(self, file_path: str) -> None:
        """
        Calculate the indices of the edges of the traveled users.

        Parameters
        ----------
        file_path : str
            The path to the file containing the traveled users.
        """
        with open(file_path, "r") as a:
            traveled_users = torch.tensor(
                [int(line.split("\t")[0]) for line in a]
            )  # read the user IDs of the traveled users
        mask = torch.isin(
            self.test_data["user", "select", "item"].edge_label_index[0], traveled_users
        )  # mark the indices of the edges of the traveled users as True or False
        self.traveled_user_edge_indices = torch.where(mask)[
            0
        ]  # get the indices of the edges of the traveled users

    def set_model_parameter(
        self, model_state_dict: dict, gnn_only: bool = False
    ) -> None:
        """
        Load the model parameters from the global server.

        Parameters
        ----------
        model_state_dict : dict
            The model parameters to be loaded.
        gnn_only : bool, optional
            Whether to load only the GNN parameters. The default is False.
        """
        if gnn_only:
            self.model.gnn.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)

    def get_model_parameter(self, gnn_only: bool = False) -> dict:
        """
        Get the model parameters.

        Parameters
        ----------
        gnn_only : bool, optional
            Whether to get only the GNN parameters. The default is False.

        Returns
        -------
        dict
            The model parameters.
        """
        if gnn_only:
            return self.model.gnn.state_dict()
        else:
            return self.model.state_dict()



class Trainer_FedSagePlus:
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        
        self.task = NodeClsTask(args, client_id, data, data_dir, device)
        self.task.load_custom_model(LocSAGEPlus(input_dim=self.task.num_feats, 
                                                hid_dim=self.args.hid_dim, 
                                                latent_dim=args.latent_dim, 
                                                output_dim=self.task.num_global_classes, 
                                                max_pred=args.max_pred, 
                                                dropout=self.args.dropout))
        self.args = args
        self.client_id = client_id
        self.message_pool = message_pool
        self.device = device
        self.splitted_impaired_data, self.num_missing, self.missing_feat, self.original_neighbors, self.impaired_neighbors = self.get_impaired_subgraph()
        self.send_message() # initial message for first-round neighGen training 
        
    def get_custom_loss_fn(self):
        # if training phase, loss function is sum of greedy loss of neighbour generation and loss of classification
        if self.phase == 0:
            def custom_loss_fn(embedding, logits, label, mask):    

                pred_degree = self.task.model.output_pred_degree
                pred_neig_feat = self.task.model.output_pred_neig_feat


                num_impaired_nodes = self.splitted_impaired_data["data"].x.shape[0]
                impaired_logits = logits[: num_impaired_nodes]


                loss_train_missing = F.smooth_l1_loss(pred_degree[mask], self.num_missing[mask])
                loss_train_feat = greedy_loss_fedsage_plus(pred_neig_feat[mask], 
                                              self.missing_feat[mask], 
                                              pred_degree[mask], 
                                              self.num_missing[mask], 
                                              max_pred=self.args.max_pred)
                loss_train_label= F.cross_entropy(impaired_logits[mask], label[mask])

                loss_other = 0

                for client_id in self.message_pool["sampled_clients"]:
                    if client_id != self.client_id:
                        others_central_ids = np.random.choice(self.message_pool[f"client_{client_id}"]["num_samples"], int(self.task.train_mask.sum()))
                        global_target_feat = []
                        for node_id in others_central_ids:
                            other_neighbors = self.message_pool[f"client_{client_id}"]["original_neighbors"][node_id]
                            while len(other_neighbors) == 0:
                                node_id = np.random.choice(self.message_pool[f"client_{client_id}"]["num_samples"], 1)[0]
                                other_neighbors = self.message_pool[f"client_{client_id}"]["original_neighbors"][node_id]
                            others_neig_ids = np.random.choice(list(other_neighbors), self.args.max_pred)
                            for neig_id in others_neig_ids:
                                global_target_feat.append(self.message_pool[f"client_{client_id}"]["feat"][neig_id])
                        global_target_feat = torch.stack(global_target_feat, 0).view(-1, self.args.max_pred, self.task.num_feats)
                        loss_train_feat_other = greedy_loss_fedsage_plus(pred_neig_feat[mask],
                                                            global_target_feat,
                                                            pred_degree[mask],
                                                            self.num_missing[mask],
                                                            max_pred=self.args.max_pred)

                        loss_other += loss_train_feat_other    

                loss = (self.args.num_missing_trade_off * loss_train_missing + \
                       self.args.missing_feat_trade_off * loss_train_feat + \
                       self.args.cls_trade_off * loss_train_label + \
                       self.args.missing_feat_trade_off * loss_other) / len(self.message_pool["sampled_clients"])
                       
                acc_degree = accuracy_missing_node_number(pred_degree[mask], self.num_missing[mask])
                acc_cls = accuracy(impaired_logits[mask], label[mask])

                print(f"[client {self.client_id} neighGen phase]\tacc_degree: {acc_degree:.4f}\tacc_cls: {acc_cls:.4f}\tloss_train: {loss:.4f}\tloss_degree: {loss_train_missing:.4f}\tloss_feat: {loss_train_feat:.4f}\tloss_cls: {loss_train_label:.4f}\tloss_other: {loss_other:.4f}")

                return loss
        else:
            # When phase=0 neighbour generation is finished, loss function is loss of classification
            def custom_loss_fn(embedding, logits, label, mask):    
                return F.cross_entropy(logits[mask], label[mask])
        return custom_loss_fn



    def execute(self):
        # switch phase
        if self.message_pool["round"] < self.args.gen_rounds:
            self.phase = 0
            self.task.override_evaluate = self.get_phase_0_override_evaluate()
        elif self.message_pool["round"] == self.args.gen_rounds:
            self.phase = 1
            self.splitted_filled_data = self.get_filled_subgraph()
            self.task.model.phase = 1
            def get_evaluate_splitted_data():
                return self.splitted_filled_data
            self.task.evaluate_splitted_data = get_evaluate_splitted_data()
            self.task.override_evaluate = self.get_phase_1_override_evaluate()
            
        # execute
        if not hasattr(self, "phase"): # miss the generator training phase due to partial participation
            self.phase = 1
            self.splitted_filled_data = self.get_filled_subgraph()
            self.task.model.phase = 1
            def get_evaluate_splitted_data():
                return self.splitted_filled_data
            self.task.evaluate_splitted_data = get_evaluate_splitted_data()
            self.task.override_evaluate = self.get_phase_1_override_evaluate()
            
            
        if self.phase == 0:
            self.task.loss_fn = self.get_custom_loss_fn()
            self.task.train(self.splitted_impaired_data)
        else:
            with torch.no_grad():
                for (local_param_with_name, global_param) in zip(self.task.model.named_parameters(), self.message_pool["server"]["weight"]):
                    name = local_param_with_name[0]
                    local_param = local_param_with_name[1]
                    if "classifier" in name:
                        local_param.data.copy_(global_param)
                        
            self.task.loss_fn = self.get_custom_loss_fn()
            self.task.train(self.splitted_filled_data)

            
            
            
            


    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "weight": list(self.task.model.parameters()),
            }

        if "round" not in self.message_pool or (hasattr(self, "phase") and self.phase == 0):
            self.message_pool[f"client_{self.client_id}"]["feat"] = self.task.data.x  # for 'loss_other'
            self.message_pool[f"client_{self.client_id}"]["original_neighbors"] = self.original_neighbors  # for 'loss_other'

    def get_impaired_subgraph(self):
        hide_len = int(self.args.hidden_portion * (self.task.val_mask).sum())
        could_hide_ids = self.task.val_mask.nonzero().squeeze().tolist()
        hide_ids = np.random.choice(could_hide_ids, hide_len, replace=False)
        all_ids = list(range(self.task.num_samples))
        remained_ids = list(set(all_ids) - set(hide_ids))

        impaired_subgraph = get_subgraph_pyg_data(global_dataset=self.task.data, node_list=remained_ids)

        impaired_subgraph = impaired_subgraph.to(self.device)
        num_missing_list = []
        missing_feat_list = []


        original_neighbors = {node_id: set() for node_id in range(self.task.data.x.shape[0])}
        for edge_id in range(self.task.data.edge_index.shape[1]):
            source = self.task.data.edge_index[0, edge_id].item()
            target = self.task.data.edge_index[1, edge_id].item()
            if source != target:
                original_neighbors[source].add(target)
                original_neighbors[target].add(source)

        impaired_neighbors = {node_id: set() for node_id in range(impaired_subgraph.x.shape[0])}
        for edge_id in range(impaired_subgraph.edge_index.shape[1]):
            source = impaired_subgraph.edge_index[0, edge_id].item()
            target = impaired_subgraph.edge_index[1, edge_id].item()
            if source != target:
                impaired_neighbors[source].add(target)
                impaired_neighbors[target].add(source)


        for impaired_id in range(impaired_subgraph.x.shape[0]):
            original_id = impaired_subgraph.global_map[impaired_id]
            num_original_neighbor = len(original_neighbors[original_id])
            num_impaired_neighbor = len(impaired_neighbors[impaired_id])
            impaired_neighbor_in_original = set()
            for impaired_neighbor in impaired_neighbors[impaired_id]:
                impaired_neighbor_in_original.add(impaired_subgraph.global_map[impaired_neighbor])

            num_missing_neighbors = num_original_neighbor - num_impaired_neighbor
            num_missing_list.append(num_missing_neighbors)
            missing_neighbors = original_neighbors[original_id] - impaired_neighbor_in_original



            if num_missing_neighbors == 0:
                current_missing_feat = torch.zeros((self.args.max_pred, self.task.num_feats)).to(self.device)
            else:
                if num_missing_neighbors <= self.args.max_pred:
                    zeros = torch.zeros((max(0, self.args.max_pred- num_missing_neighbors), self.task.num_feats)).to(self.device)
                    current_missing_feat = torch.vstack((self.task.data.x[list(missing_neighbors)], zeros)).view(self.args.max_pred, self.task.num_feats)
                else:
                    current_missing_feat = self.task.data.x[list(missing_neighbors)[:self.args.max_pred]].view(self.args.max_pred, self.task.num_feats)

            missing_feat_list.append(current_missing_feat)

        num_missing = torch.tensor(num_missing_list).squeeze().float().to(self.device)
        missing_feat = torch.stack(missing_feat_list, 0)

        impaired_train_mask = torch.zeros(impaired_subgraph.x.shape[0]).bool().to(self.device)
        impaired_val_mask = torch.zeros(impaired_subgraph.x.shape[0]).bool().to(self.device)
        impaired_test_mask = torch.zeros(impaired_subgraph.x.shape[0]).bool().to(self.device)

        for impaired_id in range(impaired_subgraph.x.shape[0]):
            original_id = impaired_subgraph.global_map[impaired_id]

            if self.task.train_mask[original_id]:
                impaired_train_mask[impaired_id] = 1

            if self.task.val_mask[original_id]:
                impaired_val_mask[impaired_id] = 1

            if self.task.test_mask[original_id]:
                impaired_test_mask[impaired_id] = 1

        splitted_impaired_data = {
            "data": impaired_subgraph,
            "train_mask": impaired_train_mask,
            "val_mask": impaired_val_mask,
            "test_mask": impaired_test_mask
        }

        return splitted_impaired_data, num_missing, missing_feat, original_neighbors, impaired_neighbors
    
    
    def get_filled_subgraph(self):
        with torch.no_grad():
            embedding, logits = self.task.model.forward(self.splitted_impaired_data["data"])
            pred_degree_float = self.task.model.output_pred_degree.detach()
            pred_neig_feat = self.task.model.output_pred_neig_feat.detach()
            num_impaired_nodes = self.splitted_impaired_data["data"].x.shape[0]
            global_map = self.splitted_impaired_data["data"].global_map
            num_original_nodes = self.task.data.x.shape[0]    
            ptr = num_original_nodes
            remain_feat = []
            remain_edges = []

            pred_degree = torch._cast_Int(pred_degree_float)
            

            for impaired_node_i in range(num_impaired_nodes):
                original_node_i = global_map[impaired_node_i]
                
                for gen_neighbor_j in range(min(self.args.max_pred, pred_degree[impaired_node_i])):
                    remain_feat.append(pred_neig_feat[impaired_node_i, gen_neighbor_j])
                    remain_edges.append(torch.tensor([original_node_i, ptr]).view(2, 1).to(self.device))
                    ptr += 1
                    
            
            
            if pred_degree.sum() > 0:
                filled_x = torch.vstack((self.task.data.x, torch.vstack(remain_feat)))
                filled_edge_index = torch.hstack((self.task.data.edge_index, torch.hstack(remain_edges)))
                filled_y = torch.hstack((self.task.data.y, torch.zeros(ptr-num_original_nodes).long().to(self.device)))
                filled_train_mask = torch.hstack((self.task.train_mask, torch.zeros(ptr-num_original_nodes).bool().to(self.device)))
                filled_val_mask = torch.hstack((self.task.val_mask, torch.zeros(ptr-num_original_nodes).bool().to(self.device)))
                filled_test_mask = torch.hstack((self.task.test_mask, torch.zeros(ptr-num_original_nodes).bool().to(self.device)))
            else:
                filled_x = torch.clone(self.task.data.x)
                filled_edge_index = torch.clone(self.task.data.edge_index)
                filled_y = torch.clone(self.task.data.y)
                filled_train_mask = torch.clone(self.task.train_mask)
                filled_val_mask = torch.clone(self.task.val_mask)
                filled_test_mask = torch.clone(self.task.test_mask)
                
            filled_data = Data(x=filled_x, edge_index=filled_edge_index, y=filled_y)
                
            splitted_filled_data = {
                "data": filled_data,
                "train_mask": filled_train_mask,
                "val_mask": filled_val_mask,
                "test_mask": filled_test_mask
            }

            return splitted_filled_data
        
    def get_phase_0_override_evaluate(self):
        def override_evaluate(splitted_data=None, mute=False):
            if splitted_data is None:
                splitted_data = self.task.splitted_data
            else:
                names = ["data", "train_mask", "val_mask", "test_mask"]
                for name in names:
                    assert name in splitted_data
                    
            self.task.model.phase = 1 # temporary modification for evaluation
            with torch.no_grad():
                embedding, logits = self.task.model.forward(splitted_data["data"])
                
                loss_train = F.cross_entropy(logits[splitted_data["train_mask"]], splitted_data["data"].y[splitted_data["train_mask"]])
                loss_val = F.cross_entropy(logits[splitted_data["val_mask"]], splitted_data["data"].y[splitted_data["val_mask"]])
                loss_test = F.cross_entropy(logits[splitted_data["test_mask"]], splitted_data["data"].y[splitted_data["test_mask"]])

            eval_output = {}
            eval_output["embedding"] = embedding
            eval_output["logits"] = logits
            eval_output["loss_train"] = loss_train
            eval_output["loss_val"]   = loss_val
            eval_output["loss_test"]  = loss_test
            
            
            metric_train = accuracy(output=logits[splitted_data["train_mask"]], labels=splitted_data["data"].y[splitted_data["train_mask"]])
            metric_val = accuracy(output=logits[splitted_data["val_mask"]], labels=splitted_data["data"].y[splitted_data["val_mask"]])
            metric_test = accuracy(output=logits[splitted_data["test_mask"]], labels=splitted_data["data"].y[splitted_data["test_mask"]])
            eval_output = {**eval_output, **metric_train, **metric_val, **metric_test}
            
            info = ""
            for key, val in eval_output.items():
                try:
                    info += f"\t{key}: {val:.4f}"
                except:
                    continue

            prefix = f"[client {self.client_id}]" if self.client_id is not None else "[server]"
            if not mute:
                print(prefix+info)
            self.task.model.phase = 0 # reset
            return eval_output
        return override_evaluate
    
    def get_phase_1_override_evaluate(self):
        def override_evaluate(splitted_data=None, mute=False):
            if splitted_data is None:
                splitted_data = self.splitted_filled_data
            else:
                names = ["data", "train_mask", "val_mask", "test_mask"]
                for name in names:
                    assert name in splitted_data
            
            
            eval_output = {}
            self.task.model.eval()
            with torch.no_grad():
                embedding, logits = self.task.model.forward(splitted_data["data"])
                loss_train = self.task.loss_fn(embedding, logits, splitted_data["data"].y, splitted_data["train_mask"])
                loss_val = self.task.loss_fn(embedding, logits, splitted_data["data"].y, splitted_data["val_mask"])
                loss_test = self.task.loss_fn(embedding, logits, splitted_data["data"].y, splitted_data["test_mask"])

            
            eval_output["embedding"] = embedding
            eval_output["logits"] = logits
            eval_output["loss_train"] = loss_train
            eval_output["loss_val"]   = loss_val
            eval_output["loss_test"]  = loss_test
            
            
            metric_train = accuracy(output=logits[splitted_data["train_mask"]], labels=splitted_data["data"].y[splitted_data["train_mask"]])
            metric_val = accuracy(output=logits[splitted_data["val_mask"]], labels=splitted_data["data"].y[splitted_data["val_mask"]])
            metric_test = accuracy(output=logits[splitted_data["test_mask"]], labels=splitted_data["data"].y[splitted_data["test_mask"]])
            eval_output = {**eval_output, **metric_train, **metric_val, **metric_test}
            
            info = ""
            for key, val in eval_output.items():
                try:
                    info += f"\t{key}: {val:.4f}"
                except:
                    continue

            prefix = f"[client {self.client_id}]" if self.client_id is not None else "[server]"
            if not mute:
                print(prefix+info)
            return eval_output
        
        return override_evaluate
    





class NodeClsTask:
    """
    Task class for node classification in a federated learning setup.

    Attributes:
        client_id (int): ID of the client.
        data_dir (str): Directory containing the data.
        args (Namespace): Arguments containing model and training configurations.
        device (torch.device): Device to run the computations on.
        data (object): Data specific to the task.
        model (torch.nn.Module): Model to be trained.
        optim (torch.optim.Optimizer): Optimizer for the model.
        train_mask (torch.Tensor): Mask for the training set.
        val_mask (torch.Tensor): Mask for the validation set.
        test_mask (torch.Tensor): Mask for the test set.
        splitted_data (dict): Dictionary containing split data and DataLoaders.
    """

    def __init__(self, args, client_id, data, data_dir, device):
        """
        Initialize the NodeClsTask with provided arguments, data, and device.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): ID of the client.
            data (object): Data specific to the task.
            data_dir (str): Directory containing the data.
            device (torch.device): Device to run the computations on.
        """
        #super(NodeClsTask, self).__init__(args, client_id, data, data_dir, device)
        self.client_id = client_id
        self.data_dir = data_dir
        self.args = args
        self.device = device
        
        if data is not None:
            self.data = data.to(device)
            #self.model = self.default_model.to(device)
            # model will be loaded in Trainer
            self.model = None
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            self.load_train_val_test_split()

        self.override_evaluate = None

    def load_custom_model(self, custom_model):
        self.model = custom_model.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def train(self, splitted_data=None):
        """
        Train the model on the provided or processed data.

        Args:
            splitted_data (dict, optional): Dictionary containing split data and DataLoaders. Defaults to None.
        """
    
        names = ["data", "train_mask", "val_mask", "test_mask"]
        # make sure splitted data has all the required keys
        for name in names:
            assert name in splitted_data

        self.model.train()

        for _ in range(self.args.num_epochs):
            self.optim.zero_grad()
            embedding, logits = self.model.forward(splitted_data["data"])
            # Note, will use CE loss direclty
            loss_train = self.loss_fn(embedding, logits, splitted_data["data"].y, splitted_data["train_mask"])
            loss_train.backward()
            self.optim.step()

    def evaluate(self, splitted_data=None, mute=False):
        """
        Evaluate the model on the provided or processed data.

        Args:
            splitted_data (dict, optional): Dictionary containing split data and DataLoaders. Defaults to None.
            mute (bool, optional): If True, suppress the print statements. Defaults to False.

        Returns:
            dict: Dictionary containing evaluation metrics and results.
        """
        if self.override_evaluate is None:
            if splitted_data is None:
                splitted_data = self.splitted_data  # use splitted_data to evaluate model
            else:
                names = ["data", "train_mask", "val_mask", "test_mask"]
                for name in names:
                    assert name in splitted_data

            eval_output = {}
            self.model.eval()
            with torch.no_grad():
                embedding, logits = self.model.forward(splitted_data["data"])
                loss_train = self.loss_fn(embedding, logits, splitted_data["data"].y, splitted_data["train_mask"])
                loss_val = self.loss_fn(embedding, logits, splitted_data["data"].y, splitted_data["val_mask"])
                loss_test = self.loss_fn(embedding, logits, splitted_data["data"].y, splitted_data["test_mask"])

            eval_output["embedding"] = embedding
            eval_output["logits"] = logits
            eval_output["loss_train"] = loss_train
            eval_output["loss_val"] = loss_val
            eval_output["loss_test"] = loss_test

            metric_train = accuracy(output=logits[splitted_data["train_mask"]], labels=splitted_data["data"].y[splitted_data["train_mask"]])
            metric_val = accuracy (output=logits[splitted_data["val_mask"]], labels=splitted_data["data"].y[splitted_data["val_mask"]])
            metric_test = accuracy(output=logits[splitted_data["test_mask"]], labels=splitted_data["data"].y[splitted_data["test_mask"]])
            eval_output = {**eval_output, **metric_train, **metric_val, **metric_test}

            info = ""
            for key, val in eval_output.items():
                try:
                    info += f"\t{key}: {val:.4f}"
                except:
                    continue

            prefix = f"[client {self.client_id}]" if self.client_id is not None else "[server]"
            if not mute:
                print(prefix + info)
            return eval_output

        else:
            # for custom evaluation on fedsage+ we override the evaluate function
            return self.override_evaluate(splitted_data, mute)

    def loss_fn(self, embedding, logits, label, mask):
        """
        Calculate the loss for the model.

        Args:
            embedding (torch.Tensor): Embeddings from the model.
            logits (torch.Tensor): Logits from the model.
            label (torch.Tensor): Ground truth labels.
            mask (torch.Tensor): Mask to filter the logits and labels.

        Returns:
            torch.Tensor: Calculated loss.
        """
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(logits[mask], label[mask])

    @property
    def num_samples(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.data.x.shape[0]

    @property
    def num_feats(self):
        """
        Get the number of features in the dataset.

        Returns:
            int: Number of features.
        """
        return self.data.x.shape[1]

    @property
    def num_global_classes(self):
        """
        Get the number of global classes in the dataset.

        Returns:
            int: Number of global classes.
        """
        return self.data.num_global_classes

    @property
    def default_train_val_test_split(self):
        """
        Get the default train/validation/test split based on the dataset.

        Returns:
            tuple: Default train/validation/test split ratios.
        """
        if self.client_id is None:
            return None

        if len(self.args.dataset) > 1:
            name = self.args.dataset[self.client_id]
        else:
            name = self.args.dataset[0]

        if name in ["Cora", "CiteSeer", "PubMed", "CS", "Physics", "Photo", "Computers"]:
            return 0.2, 0.4, 0.4
        elif name in ["Chameleon", "Squirrel"]:
            return 0.48, 0.32, 0.20
        elif name in ["ogbn-arxiv"]:
            return 0.6, 0.2, 0.2
        elif name in ["ogbn-products"]:
            return 0.1, 0.05, 0.85
        elif name in ["Roman-empire", "Amazon-ratings", "Tolokers", "Actor", "Questions", "Minesweeper"]:
            return 0.5, 0.25, 0.25

    @property
    def train_val_test_path(self):
        """
        Get the path to the train/validation/test split file.

        Returns:
            str: Path to the split file.
        """
        return osp.join(self.data_dir, f"node_cls")

    def load_train_val_test_split(self):
        """
        Load the train/validation/test split from a file.
        """
        if self.client_id is None and len(self.args.dataset) == 1:  # server
            glb_train = []
            glb_val = []
            glb_test = []

            for client_id in range(self.args.num_clients):
                glb_train_path = osp.join(self.train_val_test_path, f"glb_train_{client_id}.pkl")
                glb_val_path = osp.join(self.train_val_test_path, f"glb_val_{client_id}.pkl")
                glb_test_path = osp.join(self.train_val_test_path, f"glb_test_{client_id}.pkl")

                with open(glb_train_path, 'rb') as file:
                    glb_train_data = pickle.load(file)
                    glb_train += glb_train_data

                with open(glb_val_path, 'rb') as file:
                    glb_val_data = pickle.load(file)
                    glb_val += glb_val_data

                with open(glb_test_path, 'rb') as file:
                    glb_test_data = pickle.load(file)
                    glb_test += glb_test_data

            train_mask = idx_to_mask_tensor(glb_train, self.num_samples).bool()
            val_mask = idx_to_mask_tensor(glb_val, self.num_samples).bool()
            test_mask = idx_to_mask_tensor(glb_test, self.num_samples).bool()

        else:  # client
            train_path = osp.join(self.train_val_test_path, f"train_{self.client_id}.pt")
            val_path = osp.join(self.train_val_test_path, f"val_{self.client_id}.pt")
            test_path = osp.join(self.train_val_test_path, f"test_{self.client_id}.pt")
            glb_train_path = osp.join(self.train_val_test_path, f"glb_train_{self.client_id}.pkl")
            glb_val_path = osp.join(self.train_val_test_path, f"glb_val_{self.client_id}.pkl")
            glb_test_path = osp.join(self.train_val_test_path, f"glb_test_{self.client_id}.pkl")

            if osp.exists(train_path) and osp.exists(val_path) and osp.exists(test_path) \
                    and osp.exists(glb_train_path) and osp.exists(glb_val_path) and osp.exists(glb_test_path):
                train_mask = torch.load(train_path)
                val_mask = torch.load(val_path)
                test_mask = torch.load(test_path)
            else:
                train_mask, val_mask, test_mask = self.local_subgraph_train_val_test_split(self.data, self.args.train_val_test)

                if not osp.exists(self.train_val_test_path):
                    os.makedirs(self.train_val_test_path)

                torch.save(train_mask, train_path)
                torch.save(val_mask, val_path)
                torch.save(test_mask, test_path)

                if len(self.args.dataset) == 1:
                    # map to global
                    glb_train_id = []
                    glb_val_id = []
                    glb_test_id = []
                    for id_train in train_mask.nonzero():
                        glb_train_id.append(self.data.global_map[id_train.item()])
                    for id_val in val_mask.nonzero():
                        glb_val_id.append(self.data.global_map[id_val.item()])
                    for id_test in test_mask.nonzero():
                        glb_test_id.append(self.data.global_map[id_test.item()])
                    with open(glb_train_path, 'wb') as file:
                        pickle.dump(glb_train_id, file)
                    with open(glb_val_path, 'wb') as file:
                        pickle.dump(glb_val_id, file)
                    with open(glb_test_path, 'wb') as file:
                        pickle.dump(glb_test_id, file)

        self.train_mask = train_mask.to(self.device)
        self.val_mask = val_mask.to(self.device)
        self.test_mask = test_mask.to(self.device)

        self.splitted_data = {
            "data": self.data,
            "train_mask": self.train_mask,
            "val_mask": self.val_mask,
            "test_mask": self.test_mask
        }


    def local_subgraph_train_val_test_split(self, local_subgraph, split, shuffle=True):
        """
        Split the local subgraph into train, validation, and test sets.

        Args:
            local_subgraph (object): Local subgraph to be split.
            split (str or tuple): Split ratios or default split identifier.
            shuffle (bool, optional): If True, shuffle the subgraph before splitting. Defaults to True.

        Returns:
            tuple: Masks for the train, validation, and test sets.
        """
        num_nodes = local_subgraph.x.shape[0]

        if split == "default_split":
            train_, val_, test_ = self.default_train_val_test_split
        else:
            train_, val_, test_ = extract_floats(split)

        train_mask = idx_to_mask_tensor([], num_nodes)
        val_mask = idx_to_mask_tensor([], num_nodes)
        test_mask = idx_to_mask_tensor([], num_nodes)
        for class_i in range(local_subgraph.num_global_classes):
            class_i_node_mask = local_subgraph.y == class_i
            num_class_i_nodes = class_i_node_mask.sum()

            class_i_node_list = mask_tensor_to_idx(class_i_node_mask)
            if shuffle:
                np.random.shuffle(class_i_node_list)
            train_mask += idx_to_mask_tensor(class_i_node_list[:int(train_ * num_class_i_nodes)], num_nodes)
            val_mask += idx_to_mask_tensor(class_i_node_list[int(train_ * num_class_i_nodes): int((train_ + val_) * num_class_i_nodes)], num_nodes)
            test_mask += idx_to_mask_tensor(class_i_node_list[int((train_ + val_) * num_class_i_nodes): min(num_class_i_nodes, int((train_ + val_ + test_) * num_class_i_nodes))], num_nodes)

        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()
        return train_mask, val_mask, test_mask