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
from fedgraph.federated_methods import GC_Train

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
GC_Train(config=config)
