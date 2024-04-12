"""
Simple FedGraph Example
================

Run a simple example of FedGraph.

(Time estimate: 3 minutes)
"""

#######################################################################
# Load libraries
# ------------

import sys

sys.path.append("../fedgraph")
sys.path.append("../../")
import attridict
import yaml

from fedgraph.data_process_gc import load_single_dataset
from fedgraph.federated_methods import run_FedGCN, run_GC, run_LP
from fedgraph.utils_nc import federated_data_loader

#######################################################################
# Specify the task
# ------------

fedgraph_task = "LP"
assert fedgraph_task in ["FedGCN", "GC", "LP"]

GC_algorithm = "GCFL"  # For GC task, the user must specify the GC algorithm
if fedgraph_task == "GC":
    assert GC_algorithm in ["SelfTrain", "FedAvg", "FedProx", "GCFL"]

#######################################################################
# Load configuration and federated data
# ------------
config_file_path = (
    f"configs/config_{fedgraph_task}_{GC_algorithm}.yaml"
    if fedgraph_task == "GC"
    else f"config_{fedgraph_task}.yaml"
)
with open(config_file_path, "r") as f:
    config = attridict(yaml.safe_load(f))

print(config)

if fedgraph_task == "FedGCN":
    data = federated_data_loader(config)
elif fedgraph_task == "GC":
    seed_split_data = 42  # seed for splitting data must be fixed
    data, _ = load_single_dataset(
        config.datapath,
        dataset=config.dataset,
        num_trainer=config.num_trainers,
        batch_size=config.batch_size,
        convert_x=config.convert_x,
        seed=seed_split_data,
        overlap=config.overlap,
    )

#######################################################################
# Run FedGCN method
# ------------

if fedgraph_task == "FedGCN":
    run_FedGCN(config, data)
elif fedgraph_task == "GC":
    run_GC(config, data)
elif fedgraph_task == "LP":
    run_LP(config)
