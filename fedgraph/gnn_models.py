from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.nn import (
    GCNConv,
    GINConv,
    SAGEConv,
    global_add_pool,
    global_mean_pool,
    to_hetero,
)


class GCN(torch.nn.Module):
    """
    A Graph Convolutional Network model implementation which creates a GCN with specified
    numbers of features, hidden layers, and output classes.

    Parameters
    ----------
    nfeat : int
        The number of input features
    nhid : int
        The number of hidden features in each layer of the network
    nclass : int
        The number of output classes
    dropout : float
        The dropout probability
    NumLayers : int
        The number of layers in the GCN
    """

    def __init__(
        self, nfeat: int, nhid: int, nclass: int, dropout: float, NumLayers: int
    ):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid, normalize=True, cached=False))
        for _ in range(NumLayers - 2):
            self.convs.append(GCNConv(nhid, nhid, normalize=True, cached=False))
        self.convs.append(GCNConv(nhid, nclass, normalize=True, cached=False))

        self.dropout = dropout

    def reset_parameters(self) -> None:
        """
        Available to cater to weight initialization requirements as necessary.
        """
        for conv in self.convs:
            conv.reset_parameters()
        return None

    def forward(self, x: torch.Tensor, adj_t: torch.Tensor) -> torch.Tensor:
        """
        Represents the forward pass computation of a GCN

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor for the graph nodes.
        adj_t : torch.Tensor
            Adjacency matrix of the graph.

        Returns
        -------
        (tensor) : torch.Tensor
        """
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)


# edited#
class AggreGCN(torch.nn.Module):
    """
    This class is an Aggregated GCN model with different methods of aggregation on
    the input features for the graph nodes on the first layer with a linear layer
    and the rest of the layers with GCNConv layers.

    Parameters
    ----------
    nfeat : int
        Number of input features.
    nhid : int
        Number of hidden features in the hidden layers of the network.
    nclass : int
        Number of output classes.
    dropout : float
        Dropout probability.
    NumLayers : int
        Number of GCN layers in the network.
    """

    def __init__(
        self, nfeat: int, nhid: int, nclass: int, dropout: float, NumLayers: int
    ) -> None:
        super(AggreGCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(nfeat, nhid))
        for _ in range(NumLayers - 2):
            self.convs.append(GCNConv(nhid, nhid, normalize=True, cached=False))
        self.convs.append(GCNConv(nhid, nclass, normalize=True, cached=False))

        self.dropout = dropout

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()

    def forward(
        self, aggregated_feature: torch.Tensor, adj_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Represents the forward pass computation of a GCN with different methods of aggregation
        on the input features for the graph nodes on the first layer with a linear layer and the rest of the layers
        with GCNConv layers.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor for the graph nodes aggregated by the aggregation method.
        adj_t : torch.Tensor
            Adjacency matrix of the graph.

        Returns
        -------
        (tensor) : torch.Tensor
            The log softmax of the output of the last layer.

        """
        # x = torch.matmul(aggregated_dim, self.first_layer_weight)
        for i, conv in enumerate(self.convs[:-1]):
            if i == 0:  # check dimension of adj matrix
                x = F.relu(self.convs[0](aggregated_feature))
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = conv(x, adj_t)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)


class AggreGCN_Arxiv(torch.nn.Module):
    """
    This class is an Aggregated GCN model with different methods of aggregation on
    the input features for the graph nodes on the first layer with a linear layer
    and the rest of the layers with GCNConv layers.

    Parameters
    ----------
    nfeat : int
        Number of input features.
    nhid : int
        Number of hidden features in the hidden layers of the network.
    nclass : int
        Number of output classes.
    dropout : float
        Dropout probability.
    NumLayers : int
        Number of GCN layers in the network.
    """

    def __init__(
        self, nfeat: int, nhid: int, nclass: int, dropout: float, NumLayers: int
    ) -> None:
        super(AggreGCN_Arxiv, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(nfeat, nhid))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(nhid))
        for _ in range(NumLayers - 2):
            self.convs.append(GCNConv(nhid, nhid, normalize=True, cached=False))
            self.bns.append(torch.nn.BatchNorm1d(nhid))
        self.convs.append(GCNConv(nhid, nclass, normalize=True, cached=False))
        self.dropout = dropout

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(
        self, aggregated_feature: torch.Tensor, adj_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Represents the forward pass computation of a GCN with different methods of aggregation
        on the input features for the graph nodes on the first layer with a linear layer and the rest of the layers
        with GCNConv layers.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor for the graph nodes aggregated by the aggregation method.
        adj_t : torch.Tensor
            Adjacency matrix of the graph.

        Returns
        -------
        (tensor) : torch.Tensor
            The log softmax of the output of the last layer.

        """
        # x = torch.matmul(aggregated_dim, self.first_layer_weight)
        for i, conv in enumerate(self.convs[:-1]):
            if i == 0:  # check dimension of adj matrix
                x = self.convs[0](aggregated_feature)
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = conv(x, adj_t)
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)


class GCN_products(torch.nn.Module):
    """
    A specialized GCN model implementation designed for product graphs.

    Parameters
    ---------
    nfeat : int
        Number of input features.
    nhid : int
        Number of hidden features in the hidden layers of the network.
    nclass : int
        Number of output classes.
    dropout : float
        Dropout probability.
    NumLayers : int
        Number of GCN layers in the network.
    """

    def __init__(
        self, nfeat: int, nhid: int, nclass: int, dropout: float, NumLayers: int
    ):
        super(GCN_products, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid, normalize=False))
        for _ in range(NumLayers - 2):
            self.convs.append(GCNConv(nhid, nhid, normalize=False))
        self.convs.append(GCNConv(nhid, nclass, normalize=False))

        self.dropout = dropout

    def reset_parameters(self) -> None:
        """
        This function is available to cater to weight initialization requirements as necessary.
        """
        for conv in self.convs:
            conv.reset_parameters()
        return None

    def forward(self, x: torch.Tensor, adj_t: torch.Tensor) -> torch.Tensor:
        """
        This function represents the forward pass computation of a GCN with products as input features
        for the graph nodes on the first layer and the rest of the layers with GCNConv layers.

        x : torch.Tensor
            Input feature tensor for the graph nodes.
        adj_t : torch.Tensor
            Adjacency matrix of the graph.

        Returns
        -------
        (tensor) : torch.Tensor

        """
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)


class SAGE_products(torch.nn.Module):
    """
    A Graph SAGE model designed specifically for handling product graphs as another variant of GCN.

    Parameters
    ---------
    nfeat : int
        Number of input features.
    nhid : int
        Number of hidden features in the hidden layers of the network.
    nclass : int
        Number of output classes.
    dropout : float
        Dropout probability.
    NumLayers : int
        Number of Graph Sage layers in the network.
    """

    def __init__(
        self, nfeat: int, nhid: int, nclass: int, dropout: float, NumLayers: int
    ):
        super(SAGE_products, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(nfeat, nhid))
        for _ in range(NumLayers - 2):
            self.convs.append(SAGEConv(nhid, nhid))
        self.convs.append(SAGEConv(nhid, nclass))

        self.dropout = dropout

    def reset_parameters(self) -> None:
        """
        Available to cater to weight initialization requirements as necessary.
        """
        for conv in self.convs:
            conv.reset_parameters()
        return None

    def forward(self, x: torch.Tensor, adj_t: torch.Tensor) -> torch.Tensor:
        """
        Represents the forward pass computation of a Graph Sage model

        Parameters
        ---------
        x : torch.Tensor
            Input feature tensor for the graph nodes.
        adj_t : torch.Tensor
            Adjacency matrix of the graph.

        Returns
        -------
        (tensor) : torch.Tensor

        """
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)


# +
class GCN_arxiv(torch.nn.Module):
    """
    A variant of the GCN model tailored for the arXiv dataset.

    Parameters
    ---------
    nfeat: int
        Number of input features.
    nhid: int
        Number of hidden features in the hidden layers of the network.
    nclass: int
        Number of output classes.
    dropout: float
        Dropout probability.
    NumLayers: int
        Number of GCN layers in the network.
    """

    def __init__(
        self, nfeat: int, nhid: int, nclass: int, dropout: float, NumLayers: int
    ):
        super(GCN_arxiv, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid, cached=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(nhid))
        for _ in range(NumLayers - 2):
            self.convs.append(GCNConv(nhid, nhid, cached=False))
            self.bns.append(torch.nn.BatchNorm1d(nhid))
        self.convs.append(GCNConv(nhid, nclass, cached=False))

        self.dropout = dropout

    def reset_parameters(self) -> None:
        """
        This function is available to cater to weight initialization requirements as necessary.
        """
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        return None

    def forward(self, x: torch.Tensor, adj_t: torch.Tensor) -> torch.Tensor:
        """
        Represents the forward pass computation of a GCN

        Parameters
        ---------
        x: torch.Tensor
            Input feature tensor for the graph nodes.
        adj_t: torch.Tensor
            Adjacency matrix of the graph.

        Returns
        -------
        (tensor) : torch.Tensor

        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class GIN(torch.nn.Module):
    """
    A Graph Isomorphism Network (GIN) model implementation which creates a GIN with specified
    numbers of features, hidden units, classes, layers, and dropout.
    The GIN model is a variant of the Graph Convolutional Network (GCN) model.

    Parameters
    ----------
    nhid: int
        The number of hidden features in each layer of the GIN model.
    nlayer: int
        The number of layers.
    nfeat: int, optional
        The number of input features.
    nclass: int, optional
        The number of output classes.
    dropout: float, optional
        The dropout rate.

    Attributes
    ----------
    num_layers: int
        The number of layers in the GIN model.
    dropout: float
        The dropout rate.
    pre: torch.nn.Sequential
        The pre-neural network layer.
    graph_convs: torch.nn.ModuleList
        The list of graph convolutional layers.
    nn1: torch.nn.Sequential
        The first neural network layer.
    nnk: torch.nn.Sequential
        The k-th neural network layer.
    post: torch.nn.Sequential
        The post-neural network layer.

    Note
    ----
    This base model applies for both the server and the trainer.
    When the model is used as a server, only `nhid`, and `nlayer` should be passed as arguments.
    When the model is used as a trainer, `nfeat`, `nclass`, and `dropout` should also be passed as arguments.
    """

    def __init__(
        self,
        nhid: int,
        nlayer: int,
        nfeat: Optional[int] = None,
        nclass: Optional[int] = None,
        dropout: Optional[float] = None,
    ) -> None:
        super(GIN, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout if dropout is not None else 0.5
        self.pre = (
            torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))
            if nfeat is not None
            else None
        )
        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(
            torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid)
        )
        self.graph_convs.append(GINConv(self.nn1))

        for _ in range(nlayer - 1):
            self.nnk = torch.nn.Sequential(
                torch.nn.Linear(nhid, nhid),
                torch.nn.ReLU(),
                torch.nn.Linear(nhid, nhid),
            )
            self.graph_convs.append(GINConv(self.nnk))

        self.post = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU())
        self.readout = (
            torch.nn.Sequential(torch.nn.Linear(nhid, nclass))
            if nclass is not None
            else None
        )

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        """
        Forward pass of the GIN model, which takes in the input graph data and returns the
        model's prediction.

        Parameters
        ----------
        data: torch_geometric.data.Data
            The input graph data.

        Returns
        -------
        torch.Tensor
            The prediction of the model.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.pre(x)
        for i in range(len(self.graph_convs)):
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = global_add_pool(x, batch)
        x = self.post(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.readout(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss of the model.

        Parameters
        ----------
        pred: torch.Tensor
            The prediction of the model.
        label: torch.Tensor
            The label of the input data.

        Return
        ------
        torch.Tensor
            The nll loss of the model.
        """
        return F.nll_loss(pred, label)


class GNN_base(torch.nn.Module):
    """
    A base Graph Neural Network model implementation which creates a GNN with two convolutional layers.

    Parameters
    ----------
    hidden_channels: int
        The number of hidden features in each layer of the GNN model.

    Attributes
    ----------
    conv1: torch_geometric.nn.conv.MessagePassing
        The first convolutional layer.
    conv2: torch_geometric.nn.conv.MessagePassing
        The second convolutional layer.
    """

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Represents the forward pass computation

        Parameters
        ----------
        x: torch.Tensor
            Input feature tensor for the graph nodes.
        edge_index: torch.Tensor
            Edge index tensor of the graph.

        Returns
        -------
        (tensor) : torch.Tensor
        """
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class GNN_LP(torch.nn.Module):
    """
    A Graph Nerual Network (GNN) model implementation used for link prediction tasks, which creates a GNN with specified
    numbers of user and item nodes, hidden channels, and data metadata.

    Parameters
    ----------
    user_nums: int
        The number of user nodes.
    item_nums: int
        The number of item nodes.
    data_meta_data: tuple
        The meta data.
    hidden_channels: int
        The number of hidden features in each layer of the GNN model.

    Attributes
    ----------
    user_emb: torch.nn.Embedding
        The user embedding layer.
    item_emb: torch.nn.Embedding
        The item embedding layer.
    gnn: GNN_base
        The base GNN model.
    """

    def __init__(
        self,
        user_nums: int,
        item_nums: int,
        data_meta_data: tuple,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and items:
        self.user_emb = torch.nn.Embedding(user_nums, hidden_channels)
        self.item_emb = torch.nn.Embedding(item_nums, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN_base(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data_meta_data)

    def forward(self, data: HeteroData) -> torch.Tensor:
        """
        Represents the forward pass computation that is used in the training stage.

        Parameters
        ----------
        data: HeteroData
            The input graph data.

        Returns
        -------
        (tensor) : torch.Tensor
            The prediction output of the model.
        """
        x_dict = {
            "user": self.user_emb(data["user"].node_id),
            "item": self.item_emb(data["item"].node_id),
        }
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.__classify(
            x_dict["user"],
            x_dict["item"],
            data["user", "select", "item"].edge_label_index,
        )
        return pred

    def pred(self, train_data: HeteroData, test_data: HeteroData) -> torch.Tensor:
        """
        Represents the prediction computation that is used in the test stage.

        Parameters
        ----------
        train_data: HeteroData
            The training graph data.
        test_data: HeteroData
            The testing graph data.

        Returns
        -------
        (tensor) : torch.Tensor
            The prediction output of the model.
        """
        x_dict = {
            "user": self.user_emb(train_data["user"].node_id),
            "item": self.item_emb(train_data["item"].node_id),
        }
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, train_data.edge_index_dict)
        # if does not have negative edges
        if "edge_label_index" not in test_data["user", "select", "item"]:
            pred = self.__classify(
                x_dict["user"],
                x_dict["item"],
                test_data["user", "select", "item"].edge_index,
            )
        else:
            pred = self.__classify(
                x_dict["user"],
                x_dict["item"],
                test_data["user", "select", "item"].edge_label_index,
            )
        return pred

    # Private methods
    def __classify(
        self, x_user: torch.Tensor, x_item: torch.Tensor, edge_label_index: torch.Tensor
    ) -> torch.Tensor:
        """
        A private method to classify the user and item embeddings

        Parameters
        ----------
        x_user: torch.Tensor
            The user embeddings.
        x_item: torch.Tensor
            The item embeddings.
        edge_label_index: torch.Tensor
            The edge label index.

        Returns
        -------
        (tensor) : torch.Tensor
            The prediction output of the model.
        """
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_item = x_item[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_item).sum(dim=-1)
