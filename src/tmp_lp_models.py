import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch import Tensor

def sigmoid_range(x, low, high):
    """ Sigmoid function with range (low, high) """
    return torch.sigmoid(x) * (high-low) + low

class MFAdvanced(nn.Module):
    """ Matrix factorization + user & item bias, weight init., sigmoid_range """
    def __init__(self, num_users, num_items, emb_dim, init, bias, sigmoid, item_x_dim = None):
        super().__init__()
        self.bias = bias
        self.sigmoid = sigmoid
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        if item_x_dim is not None:
            self.item_lin = torch.nn.Linear(item_x_dim, emb_dim)
        if bias:
            self.user_bias = nn.Parameter(torch.zeros(num_users))
            self.item_bias = nn.Parameter(torch.zeros(num_items))
            self.offset = nn.Parameter(torch.zeros(1))
        if init:
            self.user_emb.weight.data.uniform_(0., 0.05)
            self.item_emb.weight.data.uniform_(0., 0.05)
    def forward(self, user, item, item_x = None):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        if item_x is not None:
            item_emb += self.item_lin(item_x)
        element_product = (user_emb*item_emb).sum(1)
        if self.bias:
            user_b = self.user_bias[user]
            item_b = self.item_bias[item]
            element_product += user_b + item_b + self.offset
        if self.sigmoid:
            return sigmoid_range(element_product, 0, 1)
        return element_product


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_user: Tensor, x_item: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_item = x_item[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_item).sum(dim=-1)


class GNN_Link_Prediction(torch.nn.Module):
    def __init__(self, user_nums, item_nums, data_meta_data, hidden_channels):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and items:
        self.user_emb = torch.nn.Embedding(user_nums, hidden_channels)
        self.item_emb = torch.nn.Embedding(item_nums, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data_meta_data)
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "user": self.user_emb(data["user"].node_id),
            "item": self.item_emb(data["item"].node_id),
        }
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["user"],
            x_dict["item"],
            data["user", "select", "item"].edge_label_index,
        )
        return pred

    def pred(self, train_data: HeteroData, test_data: HeteroData) -> Tensor:
        x_dict = {
            "user": self.user_emb(train_data["user"].node_id),
            "item": self.item_emb(train_data["item"].node_id),
        }
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, train_data.edge_index_dict)
        # if does not have negative edges
        if "edge_label_index" not in test_data["user", "select", "item"]:
            pred = self.classifier(
                x_dict["user"],
                x_dict["item"],
                test_data["user", "select", "item"].edge_index,
            )
        else:
            pred = self.classifier(
                x_dict["user"],
                x_dict["item"],
                test_data["user", "select", "item"].edge_label_index,
            )
        return pred
class GNN_Link_Prediction_with_sentence_embedding(torch.nn.Module):
    def __init__(self, data, hidden_channels):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and items:
        self.item_lin = None
        if "x" in data["item"].keys():
            self.item_lin = torch.nn.Linear(data['item'].x.shape[1], hidden_channels)
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.item_emb = torch.nn.Embedding(data["item"].num_nodes, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        if self.item_lin is None:
            x_dict = {
                "user": self.user_emb(data["user"].node_id),
                "item": self.item_emb(data["item"].node_id),
            }
        else:
            x_dict = {
                "user": self.user_emb(data["user"].node_id),
                "item": self.item_lin(data["item"].x) + self.item_emb(data["item"].node_id),
            }
            # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["user"],
            x_dict["item"],
            data["user", "select", "item"].edge_label_index,
        )
        return pred

    def pred(self, train_data: HeteroData, test_data: HeteroData) -> Tensor:
        if self.item_lin is None:
            x_dict = {
                "user": self.user_emb(train_data["user"].node_id),
                "item": self.item_emb(train_data["item"].node_id),
            }
        else:
            x_dict = {
                "user": self.user_emb(train_data["user"].node_id),
                "item": self.item_lin(train_data["item"].x) + self.item_emb(data["item"].node_id),
            }
            # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, train_data.edge_index_dict)
        # if does not have negative edges
        if "edge_label_index" not in test_data["user", "select", "item"]:
            pred = self.classifier(
                x_dict["user"],
                x_dict["item"],
                test_data["user", "select", "item"].edge_index,
            )
        else:
            pred = self.classifier(
                x_dict["user"],
                x_dict["item"],
                test_data["user", "select", "item"].edge_label_index,
            )
        return pred