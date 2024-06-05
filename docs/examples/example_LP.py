"""
Federated Link Prediction Example
================

In this tutorial, you will learn the basic workflow of
Federated Link Prediction with a runnable example. This tutorial assumes that
you have basic familiarity with PyTorch and PyTorch Geometric (PyG).

(Time estimate: 20 minutes)
"""

import argparse
import copy
import os
import random
import sys
from pathlib import Path

import attridict
import numpy as np
import ray
import torch
import yaml
from ray.util.metrics import Counter, Gauge, Histogram

from fedgraph.federated_methods import LP_train_global_round, run_fedgraph
from fedgraph.server_class import Server_LP
from fedgraph.trainer_class import Trainer_LP
from fedgraph.utils_lp import *

# Determine the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append paths relative to the current script's directory
sys.path.append(os.path.join(current_dir, "../fedgraph"))
sys.path.append(os.path.join(current_dir, "../../"))
#######################################################################
# Load configuration and check arguments
# ------------
# Here we load the configuration file for the experiment.
# The configuration file contains the parameters for the experiment.
# The algorithm and dataset (represented by the country code) are specified by the user here.
# We also specify some prechecks to ensure the validity of the arguments.

config_file = os.path.join(current_dir, "configs/config_LP.yaml")
with open(config_file, "r") as file:
    args = attridict(yaml.safe_load(file))

print(args)
run_fedgraph(args, None)
