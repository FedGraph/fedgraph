"""
Simple Federated Graph Classification Example
================

Run a simple example of Federated Graph Classification.

(Time estimate: 1 minutes)
"""

#######################################################################
# Load libraries
# ------------

import sys

import yaml

sys.path.append("../fedgraph")
from fedgraph.data_process_gc import load_single_dataset
from fedgraph.federated_methods import GC_Train
from fedgraph.gnn_models import GIN, GIN_server

#######################################################################
# Choose the model and dataset
# ------------
model = "FedAvg"  # Select: "SelfTrain", "FedAvg", "FedProx", "GCFL", "GCFL+", "GCFL+dWs
dataset = "PROTEINS"
save_files = False  # if True, save the statistics and prediction results into files

#######################################################################
# Load configuration
# ------------
config_file = f"docs/examples/configs/config_gc_{model}.yaml"
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

config["data_group"] = dataset
config["save_files"] = save_files

#######################################################################
# Load dataset
# ------------
# The user can also use their own dataset and dataloader.
# The expected format of the dataset is a dictionary with the keys as the client names.
# For each client, the value `data[client]` is a tuple with 4 elements: (dataloader, num_node_features, num_graph_labels, train_size)
# - dataloader: a dictionary with keys "train", "val", "test" and values as the corresponding dataloaders
# - num_node_features: number of node features
# - num_graph_labels: number of graph labels
# - train_size: number of training samples
# For the detailed expected format of the data, please refer to the `load_single_dataset` function in `fedgraph/data_process_gc.py`
seed_split_data = 42
data, _ = load_single_dataset(
    datapath=config["datapath"],
    dataset=config["data_group"],
    num_client=config["num_clients"],
    batch_size=config["batch_size"],
    convert_x=config["convert_x"],
    seed=seed_split_data,
    overlap=config["overlap"],
)

#######################################################################
# Designate the base model for the trainer and server
# ------------
# The base model for the trainer and server is `GIN` and `GIN_server` by default.
# They can also be specified by the user, but the user needs to make sure the customized model should be compatible with the default trainer and server.
# That is, `model_trainer` and `model_server` must have all the required methods and attributes as the default `GIN` and `GIN_server`.
# For the detailed expected format of the model, please refer to the `fedgraph/gnn_models.py`
model_trainer = GIN
model_server = GIN_server

#######################################################################
# Run the designated method
# ------------
assert model in [
    "SelfTrain",
    "FedAvg",
    "FedProx",
    "GCFL",
    "GCFL+",
    "GCFL+dWs",
], f"Unknown model: {model}"
GC_Train(
    config=config, data=data, model_server=model_server, model_trainer=model_trainer
)
