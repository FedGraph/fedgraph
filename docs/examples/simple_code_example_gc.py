"""
Simple Federated Graph Classification Example
================

Run a simple example of Federated Graph Classification.

(Time estimate: 1 minutes)
"""

#######################################################################
# Load libraries
# ------------

import yaml
import sys

sys.append("../fedgraph")
from fedgraph.federated_methods import GC_Train

#######################################################################
# Choose the model and dataset
# ------------
model = "GCFL+"
dataset = "PROTEINS"

#######################################################################
# Load configuration
# ------------
config_file = f"src/configs/config_gc_{model}.yaml"
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

config["data_group"] = dataset

#######################################################################
# Run the designated method
# ------------
assert model in ["SelfTrain", "FedAvg", "FedProx", "GCFL", "GCFL+", "GCFL+dWs"], \
    f"Unknown model: {model}"
GC_Train(config=config)