# under construction
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(
        self, nfeat: int, nhid: int, nclass: int, dropout: float, nlayers: int
    ) -> None:
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid, normalize=True, cached=True))
        for _ in range(nlayers - 2):
            self.convs.append(GCNConv(nhid, nhid, normalize=True, cached=True))
        self.convs.append(GCNConv(nhid, nclass, normalize=True, cached=True))

        self.dropout = dropout

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, adj_t: int) -> Tensor:  # TODO: what type?
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)
