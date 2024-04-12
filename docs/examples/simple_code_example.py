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
sys.path.append("../../")
import attridict
import yaml

from src.federated_methods import run_FedGCN, run_GC, run_LP
from src.utils_nc import federated_data_loader
from src.data_process_gc import load_single_dataset

#######################################################################
# Specify the task
# ------------

fedgraph_task = 'GC'
assert fedgraph_task in ['FedGCN', 'GC', 'LP']

GC_algorithm = 'GCFL' # For GC task, the user must specify the GC algorithm
if fedgraph_task == 'GC':
    assert GC_algorithm in ['SelfTrain', 'FedAvg', 'FedProx', 'GCFL', 'GCFL+', 'GCFL+dWs']

#######################################################################
# Load configuration and federated data
# ------------
config_file_path = f"config_{fedgraph_task}_{GC_algorithm}.yaml" if fedgraph_task == 'GC' else f"config_{fedgraph_task}.yaml"
with open(config_file_path, "r") as f:
    config = attridict(yaml.safe_load(f))

if fedgraph_task == 'FedGCN':
    data = federated_data_loader(config)
elif fedgraph_task == 'GC':
    data = load_single_dataset(config)

#######################################################################
# Run FedGCN method
# ------------

if fedgraph_task == 'FedGCN':
    run_FedGCN(config, data)
elif fedgraph_task == 'GC':
    run_GC(config, data)
elif fedgraph_task == 'LP':
    run_LP(config)

