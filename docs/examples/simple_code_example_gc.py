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
from src.data_process_gc import load_single_dataset, load_multiple_datasets
from src.federated_methods import GC_Train
from src.gnn_models import GIN

#######################################################################
# Choose the algorithm and dataset
# ------------
algorithm = (
    "FedProx"  # Select: "SelfTrain", "FedAvg", "FedProx", "GCFL", "GCFL+", "GCFL+dWs
)
dataset = "IMDB-BINARY" # Any dataset supplied in https://www.chrsmrrs.com/graphkerneldatasets/ (e.g., "IMDB-BINARY", "IMDB-MULTI", "PROTEINS") is valid
dataset_group = 'biochem'  # Select: 'small', 'mix', 'mix_tiny', 'biochem', 'biochem_tiny', 'molecules', 'molecules_tiny'
save_files = False  # if True, save the statistics and prediction results into files

#######################################################################
# Load configuration
# ------------
config_file = f"docs/examples/configs/config_gc_{algorithm}.yaml"
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

multiple_datasets = True
config["data_group"] = dataset_group if multiple_datasets else dataset
config["save_files"] = save_files

#######################################################################
# Load dataset
# ------------
# The user can also use their own dataset and dataloader.
# The expected format of the dataset is a dictionary with the keys as the trainer names.
# For each trainer, the value `data[trainer]` is a tuple with 4 elements: (dataloader, num_node_features, num_graph_labels, train_size)
# - dataloader: a dictionary with keys "train", "val", "test" and values as the corresponding dataloaders
# - num_node_features: number of node features
# - num_graph_labels: number of graph labels
# - train_size: number of training samples
# For the detailed expected format of the data, please refer to the `load_single_dataset` function in `fedgraph/data_process_gc.py`
seed_split_data = 42
if multiple_datasets:
    data, _ = load_multiple_datasets(
        datapath=config["datapath"],
        dataset_group=config["data_group"],
        batch_size=config["batch_size"],
        convert_x=config["convert_x"],
        seed=seed_split_data,
    )
else:
    data, _ = load_single_dataset(
        datapath=config["datapath"],
        dataset=config["data_group"],
        num_trainer=config["num_trainers"],
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
base_model = GIN

#######################################################################
# Run the designated method
# ------------
assert algorithm in [
    "SelfTrain",
    "FedAvg",
    "FedProx",
    "GCFL",
    "GCFL+",
    "GCFL+dWs",
], f"Unknown algorithm: {algorithm}"
GC_Train(config=config, data=data, base_model=base_model)
