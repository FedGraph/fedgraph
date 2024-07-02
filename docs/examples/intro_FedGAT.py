"""
FedGraph Example
================

In this tutorial, you will learn the basic workflow of
FedGraph with a runnable example. This tutorial assumes that
you have basic familiarity with PyTorch and PyTorch Geometric (PyG).

(Time estimate: 15 minutes)
"""


import argparse
from typing import Any

import numpy as np
import ray
import torch

from fedgraph.data_process import FedGAT_load_data
from fedgraph.server_class import Server
from fedgraph.trainer_class import Trainer_General
from fedgraph.utils_gat import CreateNodeSplit

np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default="cora", type=str)

parser.add_argument("-f", "--fedtype", default="fedgcn", type=str)

parser.add_argument("-c", "--global_rounds", default=10000, type=int)
parser.add_argument("-i", "--local_step", default=3, type=int)
parser.add_argument("-lr", "--learning_rate", default=0.1, type=float)

parser.add_argument("-n", "--n_trainer", default=5, type=int)
parser.add_argument("-nl", "--num_layers", default=2, type=int)
parser.add_argument("-nhop", "--num_hops", default=2, type=int)
parser.add_argument("-g", "--gpu", action="store_true")  # if -g, use gpu
parser.add_argument("-iid_b", "--iid_beta", default=10000, type=float)

parser.add_argument("-l", "--logdir", default="./runs", type=str)

args = parser.parse_args()


#######################################################################
# Data Loading
# ------------
# FedGraph use ``torch_geometric.data.Data`` to handle the data. Here, we
# use Cora, a PyG built-in dataset, as an example. To load your own
# dataset into FedGraph, you can simply load your data
# into "features, adj, labels, idx_train, idx_val, idx_test".
# Or you can create dataset in PyG. Please refer to `creating your own datasets
# tutorial <https://pytorch-geometric.readthedocs.io/en/latest/notes
# /create_dataset.html>`__ in PyG.

data, features, adj, labels, idx_train, idx_val, idx_test = FedGAT_load_data(
    args.dataset
)
# TODO: X_T, Y, L


#######################################################################
# Split Graph for Federated Learning
# ----------------------------------
# FedGraph currents has two partition methods: label_dirichlet_partition
# and community_partition_non_iid to split the large graph into multiple trainers

print(data)
client_nodes = CreateNodeSplit(data, args.n_trainer)
print(client_nodes)


#######################################################################
# Define and Send Data to Trainers
# --------------------------------
# FedGraph first determines the resources for each trainer, then send
# the data to each remote trainer.


#######################################################################
# Define Server
# -------------
# Server class is defined for federated aggregation (e.g., FedAvg)
# without knowing the local trainer data


#######################################################################
# Pre-Train Communication of FedGCN
# ---------------------------------
# Clients send their local feature sum to the server, and the server
# aggregates all local feature sums and send the global feature sum
# of specific nodes back to each client.


#######################################################################
# Federated Training
# ------------------
# The server start training of all clients and aggregate the parameters
# at every global round.


#######################################################################
# Summarize Experiment Results
# ----------------------------
# The server collects the local test loss and accuracy from all clients
# then calculate the overall test loss and accuracy.
