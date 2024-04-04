from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from server_class import Server_LP


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
    weights = [c.train_size for c in trainers]
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


def LP_train_global_round(
    clients: list,
    server: Server_LP,
    local_steps: int,
    use_buffer: bool,
    method: str,
    online_learning: bool,
    prediction_day: int,
    curr_iteration: int,
    global_rounds: int,
    record_results: bool = False,
    result_writer: Any = None,
    time_writer: Any = None,
) -> float:
    """
    This function trains the clients for a global round and updates the server model with the average of the client models.

    Parameters
    ----------
    clients : list
        List of client objects
    server : Server_LP
        Server object
    local_steps : int
        Number of local steps
    use_buffer : bool
        Specifies whether to use buffer
    method : str
        Specifies the method
    online_learning : bool
        Specifies online learning
    prediction_day : int
        Prediction day
    curr_iteration : int
        Current iteration
    global_rounds : int
        Global rounds
    record_results : bool, optional
        Record model AUC and Running time
    result_writer : Any, optional
        File writer object
    time_writer : Any, optional
        File writer object

    Returns
    -------
    current_loss : float
        Loss of the model on the training data
    """
    if record_results:
        assert result_writer is not None and time_writer is not None

    # local training
    number_of_clients = len(clients)
    for client_id in range(number_of_clients):
        current_loss, train_finish_times = clients[client_id].train(
            local_updates=local_steps, use_buffer=use_buffer
        )  # local training
        if record_results:
            for train_finish_time in train_finish_times:
                time_writer.write(
                    f"client {str(client_id)} train time {str(train_finish_time)}\n"
                )

    # aggregate the parameters and broadcast to the clients
    gnn_only = True if method == "FedLink (OnlyAvgGNN)" else False
    if method != "StaticGNN":
        model_avg_parameter = server.fedavg(clients, gnn_only)
        server.set_model_parameter(model_avg_parameter, gnn_only)
        for client_id in range(number_of_clients):
            clients[client_id].load_model_parameter(model_avg_parameter, gnn_only)

    # test the model
    avg_auc, avg_hit_rate, avg_traveled_user_hit_rate = 0.0, 0.0, 0.0
    for client_id in range(number_of_clients):
        auc_score, hit_rate, traveled_user_hit_rate = clients[client_id].test(
            use_buffer=use_buffer
        )  # local testing
        avg_auc += auc_score
        avg_hit_rate += hit_rate
        avg_traveled_user_hit_rate += traveled_user_hit_rate
        print(
            f"Day {prediction_day} client {client_id} auc score: {auc_score} hit rate: {hit_rate} traveled user hit rate: {traveled_user_hit_rate}"
        )
        # write final test_auc
        if curr_iteration + 1 == global_rounds and record_results:
            result_writer.write(
                f"Day {prediction_day} client {client_id} final auc score: {auc_score} hit rate: {hit_rate} traveled user hit rate: {traveled_user_hit_rate}\n"
            )

    avg_auc /= number_of_clients
    avg_hit_rate /= number_of_clients

    if online_learning:
        print(
            f"Predict Day {prediction_day + 1} average auc score: {avg_auc} hit rate: {avg_hit_rate}"
        )
    else:
        print(f"Predict Day 20 average auc score: {avg_auc} hit rate: {avg_hit_rate}")

    return current_loss
