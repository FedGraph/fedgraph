import logging
import random
import time
from io import BytesIO
from typing import Any, Dict, List, Union

logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
import numpy as np
import ray
import tenseal as ts
import torch
import torch.nn.functional as F
import torch_geometric
from huggingface_hub import hf_hub_download
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torchmetrics.functional.retrieval import retrieval_auroc
from torchmetrics.retrieval import RetrievalHitRate

from fedgraph.gnn_models import (
    GCN,
    GIN,
    GNN_LP,
    AggreGCN,
    AggreGCN_Arxiv,
    GCN_arxiv,
    SAGE_products,
)
from fedgraph.train_func import test, train
from fedgraph.utils_lp import (
    check_data_files_existance,
    get_data,
    get_data_loaders_per_time_step,
    get_global_user_item_mapping,
)
from fedgraph.utils_nc import get_1hop_feature_sum


def load_trainer_data_from_hugging_face(trainer_id, args):
    repo_name = f"FedGraph/fedgraph_{args.dataset}_{args.n_trainer}trainer_{args.num_hops}hop_iid_beta_{args.iid_beta}_trainer_id_{trainer_id}"

    def download_and_load_tensor(file_name):
        file_path = hf_hub_download(
            repo_id=repo_name, repo_type="dataset", filename=file_name
        )
        with open(file_path, "rb") as f:
            buffer = BytesIO(f.read())
            tensor = torch.load(buffer)
        print(f"Loaded {file_name}, size: {tensor.size()}")
        return tensor

    print(f"Loading client data {trainer_id}")
    local_node_index = download_and_load_tensor("local_node_index.pt")
    communicate_node_global_index = download_and_load_tensor(
        "communicate_node_index.pt"
    )
    global_edge_index_client = download_and_load_tensor("adj.pt")
    train_labels = download_and_load_tensor("train_labels.pt")
    test_labels = download_and_load_tensor("test_labels.pt")
    features = download_and_load_tensor("features.pt")
    in_com_train_node_local_indexes = download_and_load_tensor("idx_train.pt")
    in_com_test_node_local_indexes = download_and_load_tensor("idx_test.pt")
    return (
        local_node_index,
        communicate_node_global_index,
        global_edge_index_client,
        train_labels,
        test_labels,
        features,
        in_com_train_node_local_indexes,
        in_com_test_node_local_indexes,
    )


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
        # local_node_index: torch.Tensor,
        # communicate_node_index: torch.Tensor,
        # adj: torch.Tensor,
        # train_labels: torch.Tensor,
        # test_labels: torch.Tensor,
        # features: torch.Tensor,
        # idx_train: torch.Tensor,
        # idx_test: torch.Tensor,
        args_hidden: int,
        # global_node_num: int,
        # class_num: int,
        device: torch.device,
        args: Any,
        local_node_index: torch.Tensor = None,
        communicate_node_index: torch.Tensor = None,
        adj: torch.Tensor = None,
        train_labels: torch.Tensor = None,
        test_labels: torch.Tensor = None,
        features: torch.Tensor = None,
        idx_train: torch.Tensor = None,
        idx_test: torch.Tensor = None,
    ):
        # from gnn_models import GCN_Graph_Classification
        torch.manual_seed(rank)
        if (
            local_node_index is None
            or communicate_node_index is None
            or adj is None
            or train_labels is None
            or test_labels is None
            or features is None
            or idx_train is None
            or idx_test is None
        ):
            (
                local_node_index,
                communicate_node_index,
                adj,
                train_labels,
                test_labels,
                features,
                idx_train,
                idx_test,
            ) = load_trainer_data_from_hugging_face(rank, args)
        self.rank = rank  # rank = trainer ID

        self.device = device

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
        self.args_hidden = args_hidden
        # self.global_node_num = global_node_num
        # self.class_num = class_num
        self.args = args
        self.model = None
        self.feature_aggregation = None
        if self.args.method == "FedAvg":
            # print("Loading feature as the feature aggregation for fedavg method")
            self.feature_aggregation = self.features

    def get_info(self):
        # assert self.train_labels.numel() > 0, "train_labels is empty"
        # assert self.test_labels.numel() > 0, "test_labels is empty"
        return {
            "features_num": len(self.features),
            "label_num": max(
                self.train_labels.max().item(), self.test_labels.max().item()
            )
            + 1,
            "feature_shape": self.features.shape[1],
            "len_in_com_train_node_local_indexes": len(self.idx_train),
            "len_in_com_test_node_local_indexes": len(self.idx_test),
            "communicate_node_global_index": self.communicate_node_index,
        }

    def init_model(self, global_node_num, class_num):
        self.global_node_num = global_node_num
        self.class_num = class_num
        self.feature_shape = None

        self.scale_factor = 1e3
        self.param_history = []

        # seems that new trainer process will not inherit sys.path from parent, need to reimport!
        if self.args.num_hops >= 1:
            if self.args.dataset == "ogbn-arxiv":
                print("running AggreGCN_Arxiv")
                self.model = AggreGCN_Arxiv(
                    nfeat=self.features.shape[1],
                    nhid=self.args_hidden,
                    nclass=class_num,
                    dropout=0.5,
                    NumLayers=self.args.num_layers,
                ).to(self.device)
            else:
                self.model = AggreGCN(
                    nfeat=self.features.shape[1],
                    nhid=self.args_hidden,
                    nclass=class_num,
                    dropout=0.5,
                    NumLayers=self.args.num_layers,
                ).to(self.device)
        else:
            if "ogbn" in self.args.dataset:  # all ogbn large datasets
                print("Running GCN_arxiv")
                self.model = GCN_arxiv(
                    nfeat=self.features.shape[1],
                    nhid=self.args_hidden,
                    nclass=class_num,
                    dropout=0.5,
                    NumLayers=self.args.num_layers,
                ).to(self.device)
            elif self.args.dataset == "ogbn-products":  # ogbn not coming here
                self.model = SAGE_products(
                    nfeat=self.features.shape[1],
                    nhid=self.args_hidden,
                    nclass=class_num,
                    dropout=0.5,
                    NumLayers=self.args.num_layers,
                ).to(self.device)
            else:  # small datasets
                self.model = GCN(
                    nfeat=self.features.shape[1],
                    nhid=self.args_hidden,
                    nclass=class_num,
                    dropout=0.5,
                    NumLayers=self.args.num_layers,
                ).to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.args.learning_rate, weight_decay=5e-4
        )

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

    def verify_param_ranges(self, params, stage="pre-encryption"):
        """Verify parameter ranges and print statistics"""
        stats = []
        for i, p in enumerate(params):
            if isinstance(p, torch.Tensor):
                p = p.detach().cpu()
            stats.append(
                {
                    "layer": i,
                    "min": float(p.min()),
                    "max": float(p.max()),
                    "mean": float(p.mean()),
                    "std": float(p.std()),
                }
            )
            print(f"{stage} Layer {i} stats:")
            print(f"Range: [{stats[-1]['min']:.6f}, {stats[-1]['max']:.6f}]")
            print(f"Mean: {stats[-1]['mean']:.6f}")
            print(f"Std: {stats[-1]['std']:.6f}")
        return stats

    def get_local_feature_sum(self) -> torch.Tensor:
        """
        Computes the sum of features of all 1-hop neighbors for each node and normalizes the result.

        Returns
        -------
        normalized_sum : torch.Tensor
            The normalized sum of features of 1-hop neighbors for each node
        """
        # Create a large matrix with known local node features
        new_feature_for_trainer = torch.zeros(
            self.global_node_num, self.features.shape[1]
        ).to(self.device)
        new_feature_for_trainer[self.local_node_index] = self.features

        # Sum of features of all 1-hop nodes for each node
        one_hop_neighbor_feature_sum = get_1hop_feature_sum(
            new_feature_for_trainer, self.adj, self.device
        )
        if self.args.use_encryption:
            print(
                f"Trainer {self.rank} - Original feature sum (first 10 and last 10 elements): "
                f"{one_hop_neighbor_feature_sum.flatten()[:10].tolist()} ... {one_hop_neighbor_feature_sum.flatten()[-10:].tolist()}"
            )

        return one_hop_neighbor_feature_sum

    def get_local_feature_sum_og(self) -> torch.Tensor:
        """
        Computes the sum of features of all 1-hop neighbors for each node, used for plain text version.

        Returns
        -------
        one_hop_neighbor_feature_sum : torch.Tensor
            The sum of features of 1-hop neighbors for each node
        """

        computation_start = time.time()
        new_feature_for_trainer = torch.zeros(
            self.global_node_num, self.features.shape[1]
        ).to(self.device)
        new_feature_for_trainer[self.local_node_index] = self.features
        one_hop_neighbor_feature_sum = get_1hop_feature_sum(
            new_feature_for_trainer, self.adj, self.device
        )
        computation_time = time.time() - computation_start

        data_size = (
            one_hop_neighbor_feature_sum.element_size()
            * one_hop_neighbor_feature_sum.nelement()
        )

        print(f"Trainer {self.rank} - Computation time: {computation_time:.4f} seconds")
        print(f"Trainer {self.rank} - Data size: {data_size / 1024:.2f} KB")
        print(f"Trainer {self.rank} - Feature sum statistics:")
        print(f"Shape: {one_hop_neighbor_feature_sum.shape}")
        print(f"Mean: {one_hop_neighbor_feature_sum.mean().item():.6f}")
        print(f"Std: {one_hop_neighbor_feature_sum.std().item():.6f}")
        print(f"Min: {one_hop_neighbor_feature_sum.min().item():.6f}")
        print(f"Max: {one_hop_neighbor_feature_sum.max().item():.6f}")
        print(f"Non-zeros: {(one_hop_neighbor_feature_sum != 0).sum().item()}")

        return one_hop_neighbor_feature_sum, computation_time, data_size

    def load_feature_aggregation(self, feature_aggregation: torch.Tensor) -> None:
        """
        Loads the aggregated features into the trainer. Used for plain text version

        Parameters
        ----------
        feature_aggregation : torch.Tensor
            The aggregated features to be loaded.
        """
        # load_start = time.time()
        self.feature_aggregation = feature_aggregation.float()
        # load_time = time.time() - load_start
        # data_size = (
        #     self.feature_aggregation.element_size()
        #     * self.feature_aggregation.nelement()
        # )
        # print(f"Trainer {self.rank} - Load time: {load_time:.4f} seconds")
        # print(f"Trainer {self.rank} - Data size: {data_size / 1024:.2f} KB")

        # return load_time

    def encrypt_feature_sum(self, feature_sum):
        feature_sum = self.get_local_feature_sum()
        # does not scale
        flattened_sum = feature_sum.flatten()
        enc_sum = ts.ckks_vector(self.he_context, flattened_sum.tolist()).serialize()

        return enc_sum, feature_sum.shape

    def decrypt_feature_sum(self, encrypted_sum, shape):
        decrypted_rows = [
            ts.ckks_vector_from(self.he_context, enc_row).decrypt()
            for enc_row in encrypted_sum
        ]

        decrypted_array = np.array(decrypted_rows)
        return torch.from_numpy(decrypted_array).float().reshape(shape)

    def get_encrypted_local_feature_sum(self):
        # Same feature sum computation as original
        new_feature_for_trainer = torch.zeros(
            self.global_node_num, self.features.shape[1]
        ).to(self.device)
        new_feature_for_trainer[self.local_node_index] = self.features
        feature_sum = get_1hop_feature_sum(
            new_feature_for_trainer, self.adj, self.device
        )

        # Encrypt the feature sum
        encryption_start = time.time()
        flattened = feature_sum.flatten().tolist()
        encrypted = ts.ckks_vector(self.he_context, flattened).serialize()
        encryption_time = time.time() - encryption_start

        return encrypted, feature_sum.shape, encryption_time

    def load_encrypted_feature_aggregation(self, encrypted_data):
        encrypted_sum, shape = encrypted_data

        decryption_start = time.time()
        decrypted = ts.ckks_vector_from(self.he_context, encrypted_sum).decrypt()

        # reshape and store
        self.feature_aggregation = torch.tensor(decrypted).reshape(shape)[
            self.communicate_node_index
        ]

        return time.time() - decryption_start

    def get_encrypted_params(self):
        """Get encrypted parameters with proper scaling"""
        params_list = []
        metadata = []

        for param in self.model.parameters():
            param_data = param.cpu().detach()
            # scale
            max_abs_val = torch.max(torch.abs(param_data))
            scale = 1e3 if max_abs_val < 1e-3 else 1e2

            scaled_params = (param_data * scale).flatten().tolist()
            encrypted = ts.ckks_vector(self.he_context, scaled_params).serialize()

            params_list.append(encrypted)
            metadata.append({"shape": param_data.shape, "scale": scale})

        return params_list, metadata

    def load_encrypted_params(self, encrypted_data: tuple, current_global_epoch: int):
        """Load encrypted parameters with rescaling"""
        params_list, metadata = encrypted_data

        self.model.to("cpu")

        # load each layer's parameters
        for param, enc_param, meta in zip(
            self.model.parameters(), params_list, metadata
        ):
            decrypted = ts.ckks_vector_from(self.he_context, enc_param).decrypt()
            param_data = torch.tensor(decrypted).reshape(meta["shape"])
            param_data = param_data / meta["scale"]  # Reverse scaling
            param.data.copy_(param_data)

        self.model.to(self.device)
        return True

    def use_fedavg_feature(self) -> None:
        self.feature_aggregation

    def relabel_adj(self) -> None:
        """
        Relabels the adjacency matrix based on the communication node index.
        """
        # print(f"Max value in adj: {self.adj.max()}")
        # print(
        #     f"Max value in communicate_node_index: {self.communicate_node_index.max()}"
        # )
        # distinct_values = torch.unique(self.adj.flatten())
        # print(f"Number of distinct values in adj: {distinct_values.numel()}")
        # print(f"distinct local: {len(self.local_node_index)}")
        # print(f"distinct communic: {len(self.communicate_node_index)}")
        # time.sleep(30)
        _, self.adj, __, ___ = torch_geometric.utils.k_hop_subgraph(
            self.communicate_node_index, 0, self.adj, relabel_nodes=True
        )
        # print(f"Max value in adj: {self.adj.max()}")
        # print(
        #     f"Max value in communicate_node_index: {self.communicate_node_index.max()}"
        # )
        # distinct_values = torch.unique(self.adj.flatten())
        # print(f"Number of distinct values in adj: {distinct_values.numel()}")
        # print(f"distinct communic: {len(self.communicate_node_index)}")

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
        assert self.model is not None
        self.model.to(self.device)
        if self.feature_aggregation is None:
            raise ValueError(
                "feature_aggregation has not been set. Ensure pre-training communication is completed."
            )

        self.feature_aggregation = self.feature_aggregation.to(self.device)
        if hasattr(self.args, "batch_size") and self.args.batch_size > 0:
            # batch preparation
            train_mask = torch.zeros(
                self.feature_aggregation.size(0), dtype=torch.bool
            ).to(self.device)
            train_mask[self.idx_train] = True

            node_labels = torch.full(
                (self.feature_aggregation.size(0),), -1, dtype=torch.long
            ).to(self.device)

            mask_indices = train_mask.nonzero(as_tuple=True)[0].to(self.device)
            node_labels[train_mask] = self.train_labels[: len(mask_indices)]
            data = Data(
                x=self.feature_aggregation,
                edge_index=self.adj,
                train_mask=train_mask,
                y=node_labels,
            )
        for iteration in range(self.local_step):
            self.model.train()
            if hasattr(self.args, "batch_size") and self.args.batch_size > 0:
                # print(f"Training with batch size {self.args.batch_size}")
                loader = NeighborLoader(
                    data,
                    num_neighbors=[-1] * self.args.num_layers,
                    batch_size=2048,
                    input_nodes=self.idx_train,
                    shuffle=False,
                    num_workers=0,
                )
                batch_iter = iter(loader)
                batch = next(batch_iter, None)
                while batch is not None:
                    batch_feature_aggregation = batch.x
                    batch_adj_matrix = batch.edge_index

                    # print(f"Batch Feature Aggregation (Node Features): {batch_feature_aggregation.size()}")
                    # print(f"Batch Adjacency Matrix (Edge Index): {batch_adj_matrix}")
                    # print(f"Training Labels (Filtered by train_mask): {batch.y[batch.train_mask]}")
                    # print(f"Train Mask: {batch.train_mask}")
                    loss_train, acc_train = train(
                        iteration,
                        self.model,
                        self.optimizer,
                        batch_feature_aggregation,
                        batch_adj_matrix,
                        batch.y[batch.train_mask],
                        batch.train_mask,
                    )
                    # print(f"acc_train: {acc_train}")

                    self.train_losses.append(loss_train)
                    self.train_accs.append(acc_train)
                    batch = next(batch_iter, None)
            else:
                # print("Training with full batch")
                # print(f"feature_aggregation size: {self.feature_aggregation.size()}")
                # print(f"adj shape: {self.adj.size()}")
                # print(f"Max value in adj: {self.adj.max()}")
                # print(f"Max value in communicate_node_index: {self.communicate_node_index.max()}")
                # Assuming class_num is the number of classes
                train_labels = self.train_labels
                class_num = self.class_num
                assert (
                    train_labels.min() >= 0
                ), f"train_labels contains negative values: {train_labels.min()}"
                assert (
                    train_labels.max() < class_num
                ), f"train_labels contains a value out of range: {train_labels.max()} (number of classes: {class_num})"

                # time.sleep(30)
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
            # print(f"acc_train: {acc_train}")
            loss_test, acc_test = self.local_test()
            self.test_losses.append(loss_test)
            self.test_accs.append(acc_test)
            # print(f"current round: {current_global_round}, acc_test: {acc_test}")

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

    def get_dW(self) -> Any:
        return self.dW

    def get_name(self) -> str:
        return self.name

    def get_id(self) -> Any:
        return self.id

    def get_conv_grads_norm(self) -> Any:
        return self.conv_grads_norm

    def get_conv_dWs_norm(self) -> Any:
        return self.conv_dWs_norm

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
        dataset_path: str,
        hidden_channels: int = 64,
    ):
        self.client_id = client_id
        self.country_code = country_code
        print(f"checking code and file path: {country_code},{dataset_path}")
        file_path = dataset_path
        country_codes: List[str] = [self.country_code]
        check_data_files_existance(country_codes, file_path)
        # global user_id and item_id
        self.data = get_data(
            self.country_code, user_id_mapping, item_id_mapping, file_path
        )
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
                f"client {self.client_id} local steps {i} loss {loss:.4f} train time {train_finish_time:.4f}"
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
