import numpy as np
import pandas as pd
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


def run_GC_selftrain(clients: list, server: object, local_epoch: int) -> dict:
    """
    Run the training and testing process of self-training algorithm.
    It only trains the model locally, and does not perform weights aggregation.

    Parameters
    ----------
    clients: list
        List of clients
    server: object
        Server object
    local_epoch: int
        Number of local epochs

    Returns
    -------
    all_accs: dict
        Dictionary with training and test accuracies for each client
    """
    # all clients are initialized with the same weights
    for client in clients:
        client.download_from_server(server)

    all_accs = {}
    for client in clients:
        client.local_train(local_epoch=local_epoch)

        _, acc = client.local_test()
        all_accs[client.name] = [
            client.train_stats["trainingAccs"][-1],
            client.train_stats["valAccs"][-1],
            acc,
        ]
        print("  > {} done.".format(client.name))

    return all_accs


def run_GC_fedavg(
    clients: list,
    server: object,
    communication_rounds: int,
    local_epoch: int,
    samp: str = None,
    frac: float = 1.0,
) -> pd.DataFrame:
    """
    Run the training and testing process of FedAvg algorithm.
    It trains the model locally, aggregates the weights to the server,
    and downloads the global model within each communication round.

    Parameters
    ----------
    clients: list
        List of clients
    server: object
        Server object
    communication_rounds: int
        Number of communication rounds
    local_epoch: int
        Number of local epochs
    samp: str
        Sampling method
    frac: float
        Fraction of clients to sample

    Returns
    -------
    frame: pd.DataFrame
        Pandas dataframe with test accuracies
    """

    for client in clients:
        client.update_params(server)  # download the global model

    if samp is None:
        frac = 1.0

    # Overall training architecture:
    # whole training => { communication rounds, communication rounds, ..., communication rounds }
    # communication rounds => { local training (#epochs) -> aggregation -> download }
    #                                |
    #                           training_stats
    for c_round in range(1, communication_rounds + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")  # print the current round every 50 rounds

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = server.random_sample_clients(clients, frac)
            # if samp = None, frac=1.0, then all clients are selected

        for client in selected_clients:  # only get weights of graphconv layers
            client.local_train(local_epoch=local_epoch)  # train the local model

        server.aggregate_weights(
            selected_clients
        )  # aggregate the weights of selected clients
        for client in selected_clients:
            client.update_params(server)  # re-download the global server

    frame = pd.DataFrame()
    for client in clients:
        _, acc = client.local_test()  # Final evaluation
        frame.loc[client.name, "test_acc"] = acc

    def highlight_max(s):
        is_max = s == s.max()
        return ["background-color: yellow" if v else "" for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)
    return frame


def run_GC_fedprox(
    clients: list,
    server: object,
    communication_rounds: int,
    local_epoch: int,
    mu: float,
    samp: str = None,
    frac: float = 1.0,
) -> pd.DataFrame:
    """
    Run the training and testing process of FedProx algorithm.
    It trains the model locally, aggregates the weights to the server,
    and downloads the global model within each communication round.

    Parameters
    ----------
    clients: list
        List of clients
    server: object
        Server object
    communication_rounds: int
        Number of communication rounds
    local_epoch: int
        Number of local epochs
    mu: float
        Proximal term
    samp: str
        Sampling method
    frac: float
        Fraction of clients to sample

    Returns:
        Frame: pandas dataframe with test accuracies
    """
    for client in clients:
        client.download_from_update_paramsserver(server)

    if samp is None:
        frac = 1.0

    for c_round in range(1, communication_rounds + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = server.random_sample_clients(clients, frac)

        for client in selected_clients:
            client.local_train(
                local_epoch=local_epoch, train_option="prox", mu=mu
            )  # Different from FedAvg

        server.aggregate_weights(selected_clients)
        for client in selected_clients:
            client.update_params(server)

            # cache the aggregated weights for next round
            client.cache_weights()

    frame = pd.DataFrame()
    for client in clients:
        _, acc = client.local_test()
        frame.loc[client.name, "test_acc"] = acc

    def highlight_max(s):
        is_max = s == s.max()
        return ["background-color: yellow" if v else "" for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)
    return frame


def run_GC_gcfl(
    clients: list,
    server: object,
    communication_rounds: int,
    local_epoch: int,
    EPS_1: float,
    EPS_2: float,
) -> pd.DataFrame:
    """
    Run the GCFL algorithm.
    The GCFL algorithm is a cluster-based federated learning algorithm, which aggregates the weights of the clients
    in each cluster, and dynamically splits the clusters during the training process.

    Parameters
    ----------
    clients: list
        List of clients
    server: object
        Server object
    communication_rounds: int
        Number of communication rounds
    local_epoch: int
        Number of local epochs
    EPS_1: float
        Threshold for mean update norm
    EPS_2: float
        Threshold for max update norm

    Returns
    -------
    frame: pandas.DataFrame
        Pandas dataframe with test accuracies

    """

    cluster_indices = [
        np.arange(len(clients)).astype("int")
    ]  # cluster_indices: [[0, 1, ...]]
    client_clusters = [
        [clients[i] for i in idcs] for idcs in cluster_indices
    ]  # initially there is only one cluster

    ############### COMMUNICATION ROUNDS ###############
    for c_round in range(1, communication_rounds + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        if c_round == 1:
            for client in clients:
                client.update_params(server)

        participating_clients = server.random_sample_clients(clients, frac=1.0)
        for client in participating_clients:
            client.local_train(
                local_epoch=local_epoch, train_option="gcfl"
            )  # local training
            client.reset_params()  # reset the gradients (discard the final gradients)

        similarities = server.compute_pairwise_similarities(clients)

        cluster_indices_new = []
        for idc in cluster_indices:  # cluster-wise checking
            max_norm = server.compute_max_update_norm(
                [clients[i] for i in idc]
            )  # DELTA_MAX
            mean_norm = server.compute_mean_update_norm(
                [clients[i] for i in idc]
            )  # DELTA_MEAN
            if (
                mean_norm < EPS_1 and max_norm > EPS_2 and len(idc) > 2 and c_round > 20
            ):  # stopping condition
                server.cache_model(idc, clients[idc[0]].W, acc_clients)
                c1, c2 = server.min_cut(similarities[idc][:, idc], idc)
                cluster_indices_new += [c1, c2]  # split the cluster into two
            else:
                cluster_indices_new += [idc]  # keep the same cluster

        cluster_indices = cluster_indices_new
        client_clusters = [
            [clients[i] for i in idcs] for idcs in cluster_indices
        ]  # initial: [[0, 1, ...]]

        server.aggregate_clusterwise(
            client_clusters
        )  # aggregate the weights of the clients in each cluster

        acc_clients = [
            client.local_test()[1] for client in clients
        ]  # get the test accuracy of each client
    ############### END OF COMMUNICATION ROUNDS ###############

    for idc in cluster_indices:
        server.cache_model(
            idc, clients[idc[0]].W, acc_clients
        )  # cache the first client's weights in each cluster
    # cluster-wise model

    results = np.zeros([len(clients), len(server.model_cache)])
    for i, (idcs, W, accs) in enumerate(server.model_cache):
        results[idcs, i] = np.array(accs)

    frame = pd.DataFrame(
        results,
        columns=["FL Model"]
        + ["Model {}".format(i) for i in range(results.shape[1] - 1)],
        index=["{}".format(clients[i].name) for i in range(results.shape[0])],
    )
    frame = pd.DataFrame(frame.max(axis=1))
    frame.columns = ["test_acc"]

    print(frame)

    return frame


def run_GC_gcfl_plus(
    clients: list,
    server: object,
    communication_rounds: int,
    local_epoch: int,
    EPS_1: float,
    EPS_2: float,
    seq_length: int,
    standardize: bool,
) -> pd.DataFrame:
    """
    Run the GCFL+ algorithm.

    Parameters
    ----------
    clients: list
        List of clients
    server: object
        Server object
    communication_rounds: int
        Number of communication rounds
    local_epoch: int
        Number of local epochs
    EPS_1: float
        Threshold for mean update norm
    EPS_2: float
        Threshold for max update norm
    seq_length: int
        The length of the gradient norm sequence
    standardize: bool
        Whether to standardize the distance matrix

    Returns
    -------
    frame: pandas.DataFrame
        Pandas dataframe with test accuracies
    """
    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

    seqs_grads = {c.id: [] for c in clients}
    for client in clients:
        client.update_params(server)

    for c_round in range(1, communication_rounds + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        if c_round == 1:
            for client in clients:
                client.update_params(server)

        participating_clients = server.random_sample_clients(clients, frac=1.0)

        for client in participating_clients:
            client.local_train(local_epoch=local_epoch, train_option="gcfl")
            client.reset_params()

            seqs_grads[client.id].append(client.convGradsNorm)

        cluster_indices_new = []
        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])
            if (
                mean_norm < EPS_1
                and max_norm > EPS_2
                and len(idc) > 2
                and c_round > 20
                and all(len(value) >= seq_length for value in seqs_grads.values())
            ):
                server.cache_model(idc, clients[idc[0]].W, acc_clients)

                tmp = [seqs_grads[id][-seq_length:] for id in idc]
                dtw_distances = server.compute_pairwise_distances(tmp, standardize)
                c1, c2 = server.min_cut(np.max(dtw_distances) - dtw_distances, idc)
                cluster_indices_new += [c1, c2]

                seqs_grads = {c.id: [] for c in clients}
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

        server.aggregate_clusterwise(client_clusters)

        acc_clients = [client.local_test()[1] for client in clients]

    for idc in cluster_indices:
        server.cache_model(idc, clients[idc[0]].W, acc_clients)

    results = np.zeros([len(clients), len(server.model_cache)])
    for i, (idcs, W, accs) in enumerate(server.model_cache):
        results[idcs, i] = np.array(accs)

    frame = pd.DataFrame(
        results,
        columns=["FL Model"]
        + ["Model {}".format(i) for i in range(results.shape[1] - 1)],
        index=["{}".format(clients[i].name) for i in range(results.shape[0])],
    )

    frame = pd.DataFrame(frame.max(axis=1))
    frame.columns = ["test_acc"]
    print(frame)

    return frame


def run_GC_gcfl_plus_dWs(
    clients: list,
    server: object,
    communication_rounds: int,
    local_epoch: int,
    EPS_1: float,
    EPS_2: float,
    seq_length: int,
    standardize: bool,
) -> pd.DataFrame:
    """
    Run the GCFL+ algorithm with the gradient norms of the weights.

    Parameters
    ----------
    clients: list
        List of clients
    server: object
        Server object
    communication_rounds: int
        Number of communication rounds
    local_epoch: int
        Number of local epochs
    EPS_1: float
        Threshold for mean update norm
    EPS_2: float
        Threshold for max update norm
    seq_length: int
        The length of the gradient norm sequence
    standardize: bool
        Whether to standardize the distance matrix

    Returns
    -------
    frame: pandas.DataFrame
        Pandas dataframe with test accuracies
    """
    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

    seqs_grads = {c.id: [] for c in clients}
    for client in clients:
        client.update_params(server)

    for c_round in range(1, communication_rounds + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        if c_round == 1:
            for client in clients:
                client.update_params(server)

        participating_clients = server.random_sample_clients(clients, frac=1.0)

        for client in participating_clients:
            client.local_train(local_epoch=local_epoch, train_option="gcfl")
            client.reset_params()

            seqs_grads[client.id].append(client.convDWsNorm)

        cluster_indices_new = []
        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])
            if (
                mean_norm < EPS_1
                and max_norm > EPS_2
                and len(idc) > 2
                and c_round > 20
                and all(len(value) >= seq_length for value in seqs_grads.values())
            ):
                server.cache_model(idc, clients[idc[0]].W, acc_clients)

                tmp = [seqs_grads[id][-seq_length:] for id in idc]
                dtw_distances = server.compute_pairwise_distances(tmp, standardize)
                c1, c2 = server.min_cut(np.max(dtw_distances) - dtw_distances, idc)
                cluster_indices_new += [c1, c2]

                seqs_grads = {c.id: [] for c in clients}
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

        server.aggregate_clusterwise(client_clusters)

        acc_clients = [client.local_test()[1] for client in clients]

    for idc in cluster_indices:
        server.cache_model(idc, clients[idc[0]].W, acc_clients)

    results = np.zeros([len(clients), len(server.model_cache)])
    for i, (idcs, W, accs) in enumerate(server.model_cache):
        results[idcs, i] = np.array(accs)

    frame = pd.DataFrame(
        results,
        columns=["FL Model"]
        + ["Model {}".format(i) for i in range(results.shape[1] - 1)],
        index=["{}".format(clients[i].name) for i in range(results.shape[0])],
    )
    frame = pd.DataFrame(frame.max(axis=1))
    frame.columns = ["test_acc"]
    print(frame)

    return frame
