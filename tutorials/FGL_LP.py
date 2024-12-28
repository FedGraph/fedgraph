"""
Federated Link Prediction Example
=================================

Federated Link Prediction with STFL on the Link Prediction dataset.

(Time estimate: 3 minutes)
"""

#######################################################################
# Load libraries
# --------------

import os

import attridict

from fedgraph.federated_methods import run_fedgraph

#######################################################################
# Specify the Link Prediction configuration
# ----------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath("."))
DATASET_PATH = os.path.join(
    BASE_DIR, "data", "LPDataset"
)  # Could be modified based on the user needs
config = {
    "fedgraph_task": "LP",
    # method = ["STFL", "StaticGNN", "4D-FED-GNN+", "FedLink"]
    "method": "STFL",
    # Dataset configuration
    # country_codes = ['US', 'BR', 'ID', 'TR', 'JP']
    "country_codes": ["JP"],
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
    "use_cluster": False,  # whether use kubernetes for scalability or not
    "distribution_type": "average",  # the node number distribution among clients
    "batch_size": -1,  # -1 is full batch
}
#######################################################################
# Run fedgraph method
# -------------------

config = attridict(config)
run_fedgraph(config)
