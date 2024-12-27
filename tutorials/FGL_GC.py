"""
Federated Graph Classification Example
======================================

Federated Graph Classification with GCFL+dWs on the MUTAG dataset.

(Time estimate: 3 minutes)
"""

#######################################################################
# Load libraries
# --------------

import attridict

from fedgraph.federated_methods import run_fedgraph

#######################################################################
# Specify the Graph Classification configuration
# ----------------------------------------------
config = {
    "fedgraph_task": "GC",
    # General configuration
    # algorithm options: "SelfTrain", "FedAvg", "FedProx", "GCFL", "GCFL+", "GCFL+dWs"
    "algorithm": "GCFL+dWs",
    # Dataset configuration
    "dataset": "MUTAG",
    "is_multiple_dataset": False,
    "datapath": "./data",
    "convert_x": False,
    "overlap": False,
    # Setup configuration
    "device": "cpu",
    "seed": 10,
    "seed_split_data": 42,
    # Model parameters
    "num_trainers": 2,
    "num_rounds": 200,  # Used by "FedAvg" and "GCFL" (not used in "SelfTrain")
    "local_epoch": 1,  # Used by "FedAvg" and "GCFL"
    # Specific for "SelfTrain" (used instead of "num_rounds" and "local_epoch")
    "local_epoch_selftrain": 200,
    "lr": 0.001,
    "weight_decay": 0.0005,
    "nlayer": 3,  # Number of model layers
    "hidden": 64,  # Hidden layer dimension
    "dropout": 0.5,  # Dropout rate
    "batch_size": 128,
    "gpu": False,
    "num_cpus_per_trainer": 1,
    "num_gpus_per_trainer": 0,
    # FedProx specific parameter
    "mu": 0.01,  # Regularization parameter, only used in "FedProx"
    # GCFL specific parameters
    "standardize": False,  # Used only in "GCFL", "GCFL+", "GCFL+dWs"
    "seq_length": 5,  # Sequence length, only used in "GCFL", "GCFL+", "GCFL+dWs"
    "epsilon1": 0.05,  # Privacy epsilon1, specific to "GCFL", "GCFL+", "GCFL+dWs"
    "epsilon2": 0.1,  # Privacy epsilon2, specific to "GCFL", "GCFL+", "GCFL+dWs"
    # Output configuration
    "outbase": "./outputs",
    "save_files": False,
    # Scalability and Cluster Configuration
    "use_cluster": False,  # Use Kubernetes for scalability if True
}
#######################################################################
# Run fedgraph method
# -------------------

config = attridict(config)
run_fedgraph(config)
