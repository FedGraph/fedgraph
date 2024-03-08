"""
Federated Graph Classification Example
================

In this tutorial, you will learn the basic workflow of
Federated Graph Classification with a runnable example. This tutorial assumes that
you have basic familiarity with PyTorch and PyTorch Geometric (PyG).

(Time estimate: 15 minutes)
"""

import argparse
import copy
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.append("../fedgraph")
from fedgraph.data_process_gc import *
from fedgraph.federated_methods import (
    run_GC_fedavg,
    run_GC_fedprox,
    run_GC_gcfl,
    run_GC_gcfl_plus,
    run_GC_gcfl_plus_dWs,
    run_GC_selftrain,
)
from fedgraph.gnn_models import GIN, GIN_server
from fedgraph.utils_gc import *

#######################################################################
# Load configuration
# ------------
# Here we load the configuration file for the experiment.
# The configuration file contains the parameters for the experiment.
# The algorithm and dataset are specified by the user here. And the configuration
# file is stored in the `fedgraph/configs` directory.
# Once specified the algorithm, the corresponding configuration file will be loaded.
# Feel free to modify the configuration file to suit your needs.

algorithm = "GCFL"
dataset = "PROTEINS"
save_files = False  # if True, save the statistics and prediction results into files

config_file = f"docs/examples/configs/config_gc_{algorithm}.yaml"
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

config["data_group"] = dataset

parser = argparse.ArgumentParser()
args = parser.parse_args()
for key, value in config.items():
    setattr(args, key, value)

print(args)

#######################################################################
# Set random seed
# ------------
# Here we set the random seed for reproducibility.
# Notice that to compare the performance of different methods, the random seed
# for splitting data must be fixed.

seed_split_data = 42  # seed for splitting data must be fixed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

args.device = "cuda" if torch.cuda.is_available() else "cpu"


#######################################################################
# Set output directory
# ------------
# Here we set the output directory for the results.
# The output consists of the statistics of the data on clients and the
# accuracy of the model on the test set.

# outdir_base = os.path.join(args.outbase, f'seqLen{args.seq_length}')

if save_files:
    outdir_base = args.outbase + "/" + f"{args.model}"
    outdir = os.path.join(outdir_base, f"oneDS-nonOverlap")
    if algorithm in ["SelfTrain"]:
        outdir = os.path.join(outdir, f"{args.data_group}")
    elif algorithm in ["FedAvg", "FedProx"]:
        outdir = os.path.join(outdir, f"{args.data_group}-{args.num_clients}clients")
    elif algorithm in ["GCFL"]:
        outdir = os.path.join(
            outdir,
            f"{args.data_group}-{args.num_clients}clients",
            f"eps_{args.epsilon1}_{args.epsilon2}",
        )
    elif algorithm in ["GCFL+", "GCFL+dWs"]:
        outdir = os.path.join(
            outdir,
            f"{args.data_group}-{args.num_clients}clients",
            f"eps_{args.epsilon1}_{args.epsilon2}",
            f"seqLen{args.seq_length}",
        )

    Path(outdir).mkdir(parents=True, exist_ok=True)
    print(f"Output Path: {outdir}")


#######################################################################
# Prepare data
# ------------
# Here we prepare the data for the experiment.
# The data is split into training and test sets, and then the training set
# is further split into training and validation sets.
# The statistics of the data on clients are also computed and saved.

""" using original features """
print("Preparing data (original features) ...")

splited_data, df_stats = load_single_dataset(
    args.datapath,
    args.data_group,
    num_client=args.num_clients,
    batch_size=args.batch_size,
    convert_x=args.convert_x,
    seed=seed_split_data,
    overlap=args.overlap,
)
print("Data prepared.")

if save_files:
    outdir_stats = os.path.join(outdir, f"stats_train_data.csv")
    df_stats.to_csv(outdir_stats)
    print(f"The statistics of the data are written to {outdir_stats}")


#######################################################################
# Setup server and clients (trainers)
# ------------
# Here we set up the server and clients (trainers) for the experiment.
# The server is responsible for federated aggregation (e.g., FedAvg) without knowing the local trainer data.
# The clients (trainers) are responsible for local training and testing.
# Before setting up those, the user has to specify the base models for the trainer and server separately, which are `GIN` and `GIN_server` by default.
# They user can also use other models, but the customized model should be compatible with the default trainer and server.
# That is, `model_trainer` and `model_server` must have all the required methods and attributes as the default `GIN` and `GIN_server`.
# For the detailed expected format of the model, please refer to the `fedgraph/gnn_models.py`
model_trainer = GIN
model_server = GIN_server
init_clients, _ = setup_clients(splited_data, model_trainer, args)
init_server = setup_server(model_server, args)
clients = copy.deepcopy(init_clients)
server = copy.deepcopy(init_server)

print("\nDone setting up devices.")


#######################################################################
# Federated Training for Graph Classification
# ------------
# Here we run the federated training for graph classification.
# The server starts training of all clients and aggregates the parameters.
# The output consists of the accuracy of the model on the test set.
print(f"Running {algorithm} ...")
if algorithm == "SelfTrain":
    output = run_GC_selftrain(
        clients=clients, server=server, local_epoch=args.local_epoch
    )

elif algorithm == "FedAvg":
    output = run_GC_fedavg(
        clients=clients,
        server=server,
        communication_rounds=args.num_rounds,
        local_epoch=args.local_epoch,
        samp=None,
    )

elif algorithm == "FedProx":
    output = run_GC_fedprox(
        clients=clients,
        server=server,
        communication_rounds=args.num_rounds,
        local_epoch=args.local_epoch,
        mu=args.mu,
        samp=None,
    )

elif algorithm == "GCFL":
    output = run_GC_gcfl(
        clients=clients,
        server=server,
        communication_rounds=args.num_rounds,
        local_epoch=args.local_epoch,
        EPS_1=args.epsilon1,
        EPS_2=args.epsilon2,
    )

elif algorithm == "GCFL+":
    output = run_GC_gcfl_plus(
        clients=clients,
        server=server,
        communication_rounds=args.num_rounds,
        local_epoch=args.local_epoch,
        EPS_1=args.epsilon1,
        EPS_2=args.epsilon2,
        seq_length=args.seq_length,
        standardize=args.standardize,
    )

elif algorithm == "GCFL+dWs":
    output = run_GC_gcfl_plus_dWs(
        clients=clients,
        server=server,
        communication_rounds=args.num_rounds,
        local_epoch=args.local_epoch,
        EPS_1=args.epsilon1,
        EPS_2=args.epsilon2,
        seq_length=args.seq_length,
        standardize=args.standardize,
    )

else:
    raise ValueError(f"Unknown algorithm: {algorithm}")

#######################################################################
# Save the output
# ------------
# Here we save the results to a file, and the output directory can be specified by the user.
# If save_files == False, the output will not be saved and will only be printed in the console.
if save_files:
    outdir_result = os.path.join(outdir, f"accuracy_seed{args.seed}.csv")
    pd.DataFrame(output).to_csv(outdir_result)
    print(f"The output has been written to file: {outdir_result}")
