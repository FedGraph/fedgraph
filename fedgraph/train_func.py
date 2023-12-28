import torch
import torch.nn.functional as F


def accuracy(output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    This function returns the accuracy of the output with respect to the ground truth given

    Arguments:
    output: (torch.Tensor) - the output labels predicted by the model

    labels: (torch.Tensor) - ground truth labels

    Returns:
    The accuracy of the model (float)
    """

    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def test(
    model: torch.nn.Module,
    features: torch.Tensor,
    adj: torch.Tensor,
    labels: torch.Tensor,
    idx_test: torch.Tensor,
) -> tuple:
    """
    This function tests the model and calculates the loss and accuracy

    Arguments:
    model: (torch.nn.Module) - Specific model passed
    features: (torch.Tensor) - Tensor representing the input features
    adj: (torch.Tensor) - Adjacency matrix
    labels: (torch.Tensor) - Contains the ground truth labels for the data.
    idx_test: (torch.Tensor) - Indices specifying the test data points

    Returns:
    The loss and accuracy of the model

    """
    model.eval()
    print("test", features.shape)
    output = model(features, adj)
    print("output", output.shape)
    pred_labels = torch.argmax(output, axis=1)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    return loss_test.item(), acc_test.item()  # , f1_test, auc_test


def train(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    adj: torch.Tensor,
    labels: torch.Tensor,
    idx_train: torch.Tensor,
) -> tuple:  # Centralized or new FL
    """
    This function trains the model and returns the loss and accuracy

    Arguments:
    model: (torch.nn.Module) - Specific model passed
    features: (torch.FloatTensor) - Tensor representing the input features
    adj: (torch_sparse.tensor.SparseTensor) - Adjacency matrix
    labels: (torch.LongTensor) - Contains the ground truth labels for the data.
    idx_train: (torch.LongTensor) - Indices specifying the test data points
    epoch: (int) - specifies the number of epoch on
    optimizer: (optimizer) - type of the optimizer used

    Returns:
    The loss and accuracy of the model

    """

    model.train()
    optimizer.zero_grad()

    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss_train.item(), acc_train.item()


def Lhop_Block_matrix_train(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    adj: torch.Tensor,
    labels: torch.Tensor,
    communicate_index: torch.Tensor,
    in_com_train_data_index: torch.Tensor,
) -> tuple:
    """
    Arguments:
    model: (model type) - Specific model passed
    features: (torch.FloatTensor) - Tensor representing the input features
    adj: (torch_sparse.tensor.SparseTensor) - Adjacency matrix
    labels: (torch.LongTensor) - Contains the ground truth labels for the data
    epoch: (int) - specifies the number of epoch on
    optimizer: (optimizer) - type of the optimizer used
    communicate_index: (PyTorch tensor) - List of indices specifying which data points are used for communication
    in_com_train_data_index (PyTorch tensor): Q: Diff bet this and communicate index?

    Returns:
    The loss and accuracy of the model

    """

    model.train()
    optimizer.zero_grad()

    output = model(
        features[communicate_index], adj[communicate_index][:, communicate_index]
    )

    loss_train = F.nll_loss(
        output[in_com_train_data_index],
        labels[communicate_index][in_com_train_data_index],
    )

    acc_train = accuracy(
        output[in_com_train_data_index],
        labels[communicate_index][in_com_train_data_index],
    )

    loss_train.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss_train.item(), acc_train.item()


def FedSage_train(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    adj: torch.Tensor,
    labels: torch.Tensor,
    communicate_index: torch.Tensor,
    in_com_train_data_index: torch.Tensor,
) -> tuple:
    """
    This function is to train the FedSage model

    Arguments:
    model: (model type) - Specific model passed
    features: (torch.FloatTensor) - Tensor representing the input features
    adj: (torch_sparse.tensor.SparseTensor) - Adjacency matrix
    labels: (torch.LongTensor) - Contains the ground truth labels for the data
    epoch: (int) - specifies the number of epoch on
    optimizer: (optimizer) - type of the optimizer used
    communicate_index: (PyTorch tensor) - List of indices specifying which data points are used for communication
    in_com_train_data_index (PyTorch tensor): Q: Diff bet this and communicate index?

    Returns:
    The loss and accuracy of the model

    """

    model.train()
    optimizer.zero_grad()
    # print(features.shape)

    output = model(features, adj[communicate_index][:, communicate_index])

    loss_train = F.nll_loss(
        output[in_com_train_data_index],
        labels[communicate_index][in_com_train_data_index],
    )

    acc_train = accuracy(
        output[in_com_train_data_index],
        labels[communicate_index][in_com_train_data_index],
    )

    loss_train.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss_train.item(), acc_train.item()
