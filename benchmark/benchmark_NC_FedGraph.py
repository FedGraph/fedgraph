"""
Simple FedGraph Example
=======================

Run a simple example of FedGraph.

(Time estimate: 3 minutes)
"""

#######################################################################
# Load libraries
# --------------

from typing import Any, Dict

import attridict

from fedgraph.federated_methods import run_fedgraph

#######################################################################
# Specify the Node Classification configuration
# ---------------------------------------------
config: Dict[str, Any] = {
    # Task, Method, and Dataset Settings
    "fedgraph_task": "NC",
    "dataset": "cora",
    "method": "FedAvg",  # Federated learning method, e.g., "FedGCN"
    "iid_beta": 10000,  # Dirichlet distribution parameter for label distribution among clients
    "distribution_type": "average",  # Distribution type among clients
    # Training Configuration
    "global_rounds": 100,
    "local_step": 3,
    "learning_rate": 0.5,
    "n_trainer": 5,
    "batch_size": -1,  # -1 indicates full batch training
    # Model Structure
    "num_layers": 2,
    "num_hops": 0,  # Number of n-hop neighbors for client communication
    # Resource and Hardware Settings
    "gpu": False,
    "num_cpus_per_trainer": 1,
    "num_gpus_per_trainer": 0,
    "ray_address": "auto",  # Connect to existing Ray cluster
    # Logging and Output Configuration
    "logdir": "./runs",
    # Security and Privacy
    "use_encryption": False,  # Whether to use Homomorphic Encryption for secure aggregation
    # Dataset Handling Options
    "use_huggingface": False,  # Load dataset directly from Hugging Face Hub
    "saveto_huggingface": False,  # Save partitioned dataset to Hugging Face Hub
    # Scalability and Cluster Configuration
    "use_cluster": False,  # Use Kubernetes for scalability if True
    # Low-rank compression settings
    "use_lowrank": False,
    "lowrank_method": "fixed",
    "fixed_rank": 8,
    "use_dp": False,
    "dp_epsilon": 2.0,
    "dp_delta": 1e-5,
    "dp_mechanism": "gaussian",  # "gaussian", "laplace", "local"
    "dp_sensitivity": 1.0,
    "dp_clip_norm": 1.0,
}

#######################################################################
# Run fedgraph method
# -------------------

config = attridict(config)
run_fedgraph(config)

config.iid_beta=100
run_fedgraph(config)

config.iid_beta=10
run_fedgraph(config)