import argparse
from typing import Any, Union

import numpy as np
import ray
import torch
import torch_geometric

from src.gnn_models import GCN, GIN, AggreGCN, GCN_arxiv, SAGE_products
from src.train_func import test, train
from src.utils import get_1hop_feature_sum


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
        )
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

        self.train_stats = ([0], [0], [0], [0])
        self.weights_norm = 0.0
        self.grads_norm = 0.0
        self.conv_grads_norm = 0.0
        self.conv_weights_norm = 0.0
        self.conv_dWs_norm = 0.0

    ########### Public functions ###########
    def update_params(self, server: Any) -> None:
        """
        Update the model parameters by downloading the global model weights from the server.

        Parameters
        ----------
        server: Server_GC
            The server object that contains the global model weights.
        """
        self.gconv_names = server.W.keys()  # gconv layers
        for k in server.W:
            self.W[k].data = server.W[k].data.clone()

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
        (test_loss, test_acc): tuple(float, float)
            The average loss and accuracy
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

            loss_val, acc_val = self.__eval(model, val_loader, device)
            loss_test, acc_test = self.__eval(model, test_loader, device)

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
        (test_loss, test_acc): tuple(float, float)
            The average loss and accuracy

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

        return total_loss / num_graphs, total_acc / num_graphs

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
            if name in gconv_names:  # only add the prox term for sharing layers (gConv)
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
