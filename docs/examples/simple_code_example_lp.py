"""
Simple Federated Link Prediction Example
================

Run a simple example of Federated Link Prediction.

(Time estimate: 1 minutes)
"""

#######################################################################
# Load libraries
# ------------
import sys

import torch_geometric
import yaml

from fedgraph.federated_methods import run_LP

torch_geometric.seed.seed_everything(42)

#######################################################################
# Load configuration
# ------------
# Here we load the configuration file for the experiment.
# The configuration file specifies the hyperparameters for the experiment.
# The user can modify the configuration file to specify the hyperparameters for the experiment.
# In the default configuration file, `method` is used to specify the federated method.
# The user can specify the federated method by setting `method` to one of the following values:
# `FedLink`, `STFL`, `StaticGNN`, `4D-FED-GNN+`.

config_file = "docs/examples/config_lp.yaml"
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

print(config)

#######################################################################
# Run the experiment
# ------------
# The user can run the experiment with the specified configuration.
run_LP(config)
