import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, nfeat: int, nhid: int, nclass: int, dropout: float, NumLayers: int):
        '''
        This constructor method initializes the GCN model

        Arguments:
        nfeat: (int) - Number of input features
        nhid: (int) - Number of hidden features in the hidden layers of the network
        nclass: (int) - Number of output classes
        dropout: (float) - Dropout probability
        NumLayers: (int) - Number of GCN layers in the network.        
        '''
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(nfeat, nhid, normalize=True, cached=True))
        for _ in range(NumLayers - 2):
            self.convs.append(
                GCNConv(nhid, nhid, normalize=True, cached=True))
        self.convs.append(
            GCNConv(nhid, nclass, normalize=True, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        '''
        This function is available to cater to weight initialization requirements as necessary.      
        '''
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: torch.Tensor, adj_t: torch.Tensor) -> torch.Tensor:
        '''
        This function represents the forward pass computation of a GCN

        Arguments:
        x: (torch.Tensor) - Input feature tensor for the graph nodes
        adj_t: (SparseTensor) - Adjacency matrix of the graph

        Returns:
        The output of the forward pass, a PyTorch tensor

        '''
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)

class GCN_products(torch.nn.Module):
    def __init__(self, nfeat: int, nhid: int, nclass: int, dropout: float, NumLayers: int):
        '''
        This constructor method initializes the GCN_products model

        Arguments:
        nfeat: (int) - Number of input features
        nhid: (int) - Number of hidden features in the hidden layers of the network
        nclass: (int) - Number of output classes
        dropout: (float) - Dropout probability
        NumLayers: (int) - Number of GCN layers in the network.        
        '''

        super(GCN_products, self).__init__()
    
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(nfeat, nhid, normalize=False))
        for _ in range(NumLayers - 2):
            self.convs.append(
                GCNConv(nhid, nhid, normalize=False))
        self.convs.append(
            GCNConv(nhid, nclass, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        '''
        This function is available to cater to weight initialization requirements as necessary.      
        '''
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: torch.Tensor, adj_t: torch.Tensor) -> torch.Tensor:
        '''
        This function represents the forward pass computation of a GCN

        Arguments:
        x: (torch.Tensor) - Input feature tensor for the graph nodes
        adj_t: (SparseTensor) - Adjacency matrix of the graph

        Returns:
        The output of the forward pass, a PyTorch tensor
        
        '''
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)
    
class SAGE_products(torch.nn.Module):
    def __init__(self, nfeat: int, nhid: int, nclass: int, dropout: float, NumLayers: int):
        '''
        This constructor method initializes the Graph Sage model

        Arguments:
        nfeat: (int) - Number of input features
        nhid: (int) - Number of hidden features in the hidden layers of the network
        nclass: (int) - Number of output classes
        dropout: (float) - Dropout probability
        NumLayers: (int) - Number of Graph Sage layers in the network     
        '''
        super(SAGE_products, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(nfeat, nhid))
        for _ in range(NumLayers - 2):
            self.convs.append(SAGEConv(nhid, nhid))
        self.convs.append(SAGEConv(nhid, nclass))

        self.dropout = dropout

    def reset_parameters(self):
        '''
        This function is available to cater to weight initialization requirements as necessary.      
        '''
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: torch.Tensor, adj_t: torch.Tensor) -> torch.Tensor:
        '''
        This function represents the forward pass computation of a GCN

        Arguments:
        x: (torch.Tensor) - Input feature tensor for the graph nodes
        adj_t: (SparseTensor) - Adjacency matrix of the graph

        Returns:
        The output of the forward pass, a PyTorch tensor
        
        '''
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)
    
# +
class GCN_arxiv(torch.nn.Module):
    def __init__(self, nfeat: int, nhid: int, nclass: int, dropout: float, NumLayers: int):
        '''
        This constructor method initializes the Graph Sage model

        Arguments:
        nfeat: (int) - Number of input features
        nhid: (int) - Number of hidden features in the hidden layers of the network
        nclass: (int) - Number of output classes
        dropout: (float) - Dropout probability
        NumLayers: (int) - Number of Graph Sage layers in the network     
        '''
        super(GCN_arxiv, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(nhid))
        for _ in range(NumLayers - 2):
            self.convs.append(
                GCNConv(nhid, nhid, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(nhid))
        self.convs.append(GCNConv(nhid, nclass, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        '''
        This function is available to cater to weight initialization requirements as necessary.      
        '''
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x: torch.Tensor, adj_t: torch.Tensor) -> torch.Tensor:
        '''
        This function represents the forward pass computation of a GCN

        Arguments:
        x: (torch.Tensor) - Input feature tensor for the graph nodes
        adj_t: (SparseTensor) - Adjacency matrix of the graph

        Returns:
        The output of the forward pass, a PyTorch tensor
        
        '''
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)
    


# -
#REDUNDANT

class GCN_Graph_Classification(torch.nn.Module):
    def __init__(self, num_node_features, nhid, num_classes):
        super(GCN_Graph_Classification, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        self.conv3 = GCNConv(nhid, nhid)
        self.lin = Linear(nhid, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, nhid]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x
