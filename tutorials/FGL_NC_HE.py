"""
Federated Node Classification with Homomorphic Encryption Example
======================================

Federated Node Classification with FedGCN and Homomorphic Encryption on the Cora dataset.

(Time estimate: 3 minutes)
"""

#######################################################################
# Load libraries
# --------------

import attridict

from fedgraph.federated_methods import run_fedgraph

#######################################################################
# Specify the Node Classification with Homomorphic Encryption configuration
# ---------------------------------------------
config = {
    # Task, Method, and Dataset Settings
    "fedgraph_task": "NC",
    "dataset": "cora",
    "method": "FedGCN",  # Federated learning method, e.g., "FedGCN"
    "iid_beta": 10000,  # Dirichlet distribution parameter for label distribution among clients
    "distribution_type": "average",  # Distribution type among clients
    # Training Configuration
    "global_rounds": 100,
    "local_step": 3,
    "learning_rate": 0.5,
    "n_trainer": 2,
    "batch_size": -1,  # -1 indicates full batch training
    # Model Structure
    "num_layers": 2,
    "num_hops": 1,  # Number of n-hop neighbors for client communication
    # Resource and Hardware Settings
    "gpu": False,
    "num_cpus_per_trainer": 1,
    "num_gpus_per_trainer": 0,
    # Logging and Output Configuration
    "logdir": "./runs",
    # Security and Privacy
    "use_encryption": True,  # Whether to use Homomorphic Encryption for secure aggregation
    # Dataset Handling Options
    "use_huggingface": False,  # Load dataset directly from Hugging Face Hub
    "saveto_huggingface": False,  # Save partitioned dataset to Hugging Face Hub
    # Scalability and Cluster Configuration
    "use_cluster": False,  # Use Kubernetes for scalability if True
}

#######################################################################
# Run fedgraph method
# -------------------

config = attridict(config)
run_fedgraph(config)
