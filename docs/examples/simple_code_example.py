"""
Simple FedGraph Example
================

Run a simple example of FedGraph.

(Time estimate: 1 minutes)
"""

#######################################################################
# Load libraries
# ------------

import sys

sys.path.append("../fedgraph")
import attridict
import yaml

from src.federated_methods import FedGCN_Train
from src.utils_nc import federated_data_loader

#######################################################################
# Load configuration and federated data
# ------------

with open("config_fedgcn.yaml", "r") as f:
    config = attridict(yaml.safe_load(f))

data = federated_data_loader(config)

#######################################################################
# Run FedGCN method
# ------------

FedGCN_Train(config, data)
