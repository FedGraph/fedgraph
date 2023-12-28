import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool


class GCN(torch.nn.Module):
    def __init__(
        self, nfeat: int, nhid: int, nclass: int, dropout: float, NumLayers: int
    ):
        """
        This constructor method initializes the GCN model

        Arguments:
        nfeat: (int) - Number of input features
        nhid: (int) - Number of hidden features in the hidden layers of the network
        nclass: (int) - Number of output classes
        dropout: (float) - Dropout probability
        NumLayers: (int) - Number of GCN layers in the network.
        """
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid, normalize=True, cached=True))
        for _ in range(NumLayers - 2):
            self.convs.append(GCNConv(nhid, nhid, normalize=True, cached=True))
        self.convs.append(GCNConv(nhid, nclass, normalize=True, cached=True))

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
        This function represents the forward pass computation of a GCN

        Arguments:
        x: (torch.Tensor) - Input feature tensor for the graph nodes
        adj_t: (SparseTensor) - Adjacency matrix of the graph

        Returns:
        The output of the forward pass, a PyTorch tensor

        """
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)

#edited#
class AggreGCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, NumLayers):
        super(AggreGCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(nfeat, nhid))
        for _ in range(NumLayers - 2):
            self.convs.append(
                GCNConv(nhid, nhid, normalize=True, cached=True))
        self.convs.append(
            GCNConv(nhid, nclass, normalize=True, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, aggregated_dim):
        # x = torch.matmul(aggregated_dim, self.first_layer_weight)
        for i, conv in enumerate(self.convs[:-1]):
            if i == 0 and aggregated_dim is not None:#check dimension of adj matrix
                x = F.relu(self.convs[0](aggregated_dim))
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = conv(x, adj_t)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)

class GCN_products(torch.nn.Module):
    def __init__(
        self, nfeat: int, nhid: int, nclass: int, dropout: float, NumLayers: int
    ):
        """
        This constructor method initializes the GCN_products model

        Arguments:
        nfeat: (int) - Number of input features
        nhid: (int) - Number of hidden features in the hidden layers of the network
        nclass: (int) - Number of output classes
        dropout: (float) - Dropout probability
        NumLayers: (int) - Number of GCN layers in the network.
        """

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
        This function represents the forward pass computation of a GCN

        Arguments:
        x: (torch.Tensor) - Input feature tensor for the graph nodes
        adj_t: (SparseTensor) - Adjacency matrix of the graph

        Returns:
        The output of the forward pass, a PyTorch tensor

        """
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)


class SAGE_products(torch.nn.Module):
    def __init__(
        self, nfeat: int, nhid: int, nclass: int, dropout: float, NumLayers: int
    ):
        """
        This constructor method initializes the Graph Sage model

        Arguments:
        nfeat: (int) - Number of input features
        nhid: (int) - Number of hidden features in the hidden layers of the network
        nclass: (int) - Number of output classes
        dropout: (float) - Dropout probability
        NumLayers: (int) - Number of Graph Sage layers in the network
        """
        super(SAGE_products, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(nfeat, nhid))
        for _ in range(NumLayers - 2):
            self.convs.append(SAGEConv(nhid, nhid))
        self.convs.append(SAGEConv(nhid, nclass))

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
        This function represents the forward pass computation of a GCN

        Arguments:
        x: (torch.Tensor) - Input feature tensor for the graph nodes
        adj_t: (SparseTensor) - Adjacency matrix of the graph

        Returns:
        The output of the forward pass, a PyTorch tensor

        """
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)


# +
class GCN_arxiv(torch.nn.Module):
    def __init__(
        self, nfeat: int, nhid: int, nclass: int, dropout: float, NumLayers: int
    ):
        """
        This constructor method initializes the Graph Sage model

        Arguments:
        nfeat: (int) - Number of input features
        nhid: (int) - Number of hidden features in the hidden layers of the network
        nclass: (int) - Number of output classes
        dropout: (float) - Dropout probability
        NumLayers: (int) - Number of Graph Sage layers in the network
        """
        super(GCN_arxiv, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(nhid))
        for _ in range(NumLayers - 2):
            self.convs.append(GCNConv(nhid, nhid, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(nhid))
        self.convs.append(GCNConv(nhid, nclass, cached=True))

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
        This function represents the forward pass computation of a GCN

        Arguments:
        x: (torch.Tensor) - Input feature tensor for the graph nodes
        adj_t: (SparseTensor) - Adjacency matrix of the graph

        Returns:
        The output of the forward pass, a PyTorch tensor

        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)
