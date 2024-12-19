"""
Simple FedGraph Example
=======================

Run a simple example of FedGraph.

(Time estimate: 3 minutes)
"""

#######################################################################
# Load libraries
# --------------

import os

import attridict

from fedgraph.data_process import data_loader
from fedgraph.federated_methods import run_fedgraph

#######################################################################
# Specify the Link Prediction configuration
# ----------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "LPDataset")
config = {
    "fedgraph_task": "LP",
    # method = ["STFL", "StaticGNN", "4D-FED-GNN+", "FedLink"]
    "method": "STFL",
    # Dataset configuration
    # country_codes = ['US', 'BR', 'ID', 'TR', 'JP']
    "country_codes": ["US", "BR"],
    "dataset_path": DATASET_PATH,
    # Setup configuration
    "device": "cpu",
    "use_buffer": False,
    "buffer_size": 300000,
    "online_learning": False,
    "seed": 10,
    # Model parameters
    "global_rounds": 8,
    "local_steps": 3,
    "hidden_channels": 64,
    # Output configuration
    "record_results": False,
    # System configuration
    "gpu": False,
    "num_cpus_per_trainer": 1,
    "num_gpus_per_trainer": 0,
    # Scalability and Cluster Configuration
    "use_cluster": False,  # Use Kubernetes for scalability if True
}
#######################################################################
# Run fedgraph method
# -------------------

config = attridict(config)
run_fedgraph(config)
