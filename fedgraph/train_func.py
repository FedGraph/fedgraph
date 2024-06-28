from typing import Any

import numpy as np
import pandas as pd
import ray
import torch
import torch.nn.functional as F


def accuracy(output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    This function returns the accuracy of the output with respect to the ground truth given

    Parameters
    ----------
    output: torch.Tensor
        the output labels predicted by the model

    labels: torch.Tensor
        ground truth labels

    Returns
    -------
    (tensor): torch.Tensor
        Accuracy of the output with respect to the ground truth given
    """

    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def gc_avg_accuracy(frame: pd.DataFrame, trainers: list) -> float:
    """
    This function calculates the weighted average accuracy of the trainers in the frame.

    Parameters
    ----------
    frame: pd.DataFrame
        The frame containing the accuracies of the trainers
    trainers: list
        List of trainer objects

    Returns
    -------
    (float): float
        The average accuracy of the trainers in the frame
    """

    # weighted average accuracy
    accs = frame["test_acc"]
    weights = [ray.get(c.get_train_size.remote()) for c in trainers]
    return np.average(accs, weights=weights)


def test(
    model: torch.nn.Module,
    features: torch.Tensor,
    adj: torch.Tensor,
    test_labels: torch.Tensor,
    idx_test: torch.Tensor,
) -> tuple:
    """
    This function tests the model and calculates the loss and accuracy

    Parameters
    ----------
    model : torch.nn.Module
        Specific model passed
    features : torch.Tensor
        Tensor representing the input features
    adj : torch.Tensor
        Adjacency matrix
    labels : torch.Tensor
        Contains the ground truth labels for the data.
    idx_test : torch.Tensor
        Indices specifying the test data points

    Returns
    -------
    loss_test.item() : float
        Loss of the model on the test data
    acc_test.item() : float
        Accuracy of the model on the test data

    """
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], test_labels)
    acc_test = accuracy(output[idx_test], test_labels)

    return loss_test.item(), acc_test.item()  # , f1_test, auc_test


def train(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    adj: torch.Tensor,
    train_labels: torch.Tensor,
    idx_train: torch.Tensor,
) -> tuple:  # Centralized or new FL
    """
    Trains the model and calculates the loss and accuracy of the model on the training data,
    performs backpropagation, and updates the model parameters.

    Parameters
    ----------
    epoch : int
        Specifies the number of epoch on which the model is trained
    model : torch.nn.Module
        Specific model to be trained
    optimizer : optimizer
        Type of the optimizer used for updating the model parameters
    features : torch.FloatTensor
        Tensor representing the input features
    adj : torch_sparse.tensor.SparseTensor
        Adjacency matrix
    train_labels : torch.LongTensor
        Contains the ground truth labels for the data.
    idx_train : torch.LongTensor
        Indices specifying the test data points


    Returns
    -------
    loss_train.item() : float
        Loss of the model on the training data
    acc_train.item() : float
        Accuracy of the model on the training data

    """

    model.train()
    optimizer.zero_grad()

    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], train_labels)
    acc_train = accuracy(output[idx_train], train_labels)
    loss_train.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss_train.item(), acc_train.item()
