"""
Simple FedGraph Example
=======================

Run a simple example of FedGraph.

(Time estimate: 3 minutes)
"""

#######################################################################
# Load libraries
# --------------

import sys

import attridict
import yaml

sys.path.append("../fedgraph")
from fedgraph.data_process import data_loader
from fedgraph.federated_methods import run_fedgraph

#######################################################################
# Specify the task
# ----------------

fedgraph_task = "GC"
assert fedgraph_task in ["FedGCN", "GC", "LP"]

GC_algorithm = "GCFL"  # For GC task, the user must specify the GC algorithm
if fedgraph_task == "GC":
    assert GC_algorithm in ["SelfTrain", "FedAvg", "FedProx", "GCFL"]

#######################################################################
# Load configuration and federated data
# -------------------------------------
config_file_path = (
    f"configs/config_{fedgraph_task}_{GC_algorithm}.yaml"
    if fedgraph_task == "GC"
    else f"configs/config_{fedgraph_task}.yaml"
)

with open(config_file_path, "r") as f:
    config = attridict(yaml.safe_load(f))
config.fedgraph_task = fedgraph_task

print(config)

data = data_loader(config)  # Load federated data

#######################################################################
# Run FedGCN method
# -----------------

run_fedgraph(config, data)
