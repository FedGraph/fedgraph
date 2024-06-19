"""
Federated Graph Classification Example
================

In this tutorial, you will learn the basic workflow of
Federated Graph Classification with a runnable example. This tutorial assumes that
you have basic familiarity with PyTorch and PyTorch Geometric (PyG).

(Time estimate: 15 minutes)
"""

from fedgraph.utils_gc import *
from fedgraph.gnn_models import GIN
from fedgraph.federated_methods import (
    run_GC_Fed_algorithm,
    run_GCFL_algorithm,
    run_GC_selftrain,
)
from fedgraph.data_process import data_loader_GC
import argparse
import copy
import os
import random
import sys
from pathlib import Path
import ray
import attridict
import numpy as np
import torch
import yaml

sys.path.append("../fedgraph")
sys.path.append("../../")

#######################################################################
# Load configuration
# ------------
# Here we load the configuration file for the experiment.
# The configuration file contains the parameters for the experiment.
# The algorithm and dataset are specified by the user here. And the configuration
# file is stored in the `fedgraph/configs` directory.
# Once specified the algorithm, the corresponding configuration file will be loaded.
# Feel free to modify the configuration file to suit your needs.
# For `dataset`, the user can either use single or multiple datasets from TU Datasets, which is controlled by the `is_multiple_dataset` flag.
# For single dataset, any dataset supplied in https://www.chrsmrrs.com/graphkerneldatasets/ (e.g., "IMDB-BINARY", "IMDB-MULTI", "PROTEINS") is valid
# For multiple datasets, the user can choose from the following groups: 'small', 'mix', 'mix_tiny', 'biochem', 'biochem_tiny', 'molecules', 'molecules_tiny'
# For the detailed content of each group, please refer to the `load_multiple_datasets` function in `src/data_process_gc.py`

algorithm = "SelfTrain"

config_file = f"configs/config_GC_{algorithm}.yaml"
with open(config_file, "r") as file:
    args = attridict(yaml.safe_load(file))

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
base_model = GIN
args.device = "cuda" if torch.cuda.is_available() else "cpu"
num_cpus_per_trainer = 3
# specifying a target GPU
if torch.cuda.is_available():
    print("using GPU")
    device = torch.device("cuda")
    num_gpus_per_trainer = 1
else:
    print("using CPU")
    device = torch.device("cpu")
    num_gpus_per_trainer = 0

#######################################################################
# Set output directory
# ------------
# Here we set the output directory for the results.
# The output consists of the statistics of the data on trainers and the
# accuracy of the model on the test set.

# outdir_base = os.path.join(args.outbase, f'seqLen{args.seq_length}')

if args.save_files:
    outdir_base = args.outbase + "/" + f"{args.model}"
    outdir = os.path.join(outdir_base, f"oneDS-nonOverlap")
    if algorithm in ["SelfTrain"]:
        outdir = os.path.join(outdir, f"{args.dataset}")
    elif algorithm in ["FedAvg", "FedProx"]:
        outdir = os.path.join(
            outdir, f"{args.dataset}-{args.num_trainers}trainers")
    elif algorithm in ["GCFL"]:
        outdir = os.path.join(
            outdir,
            f"{args.dataset}-{args.num_trainers}trainers",
            f"eps_{args.epsilon1}_{args.epsilon2}",
        )
    elif algorithm in ["GCFL+", "GCFL+dWs"]:
        outdir = os.path.join(
            outdir,
            f"{args.dataset}-{args.num_trainers}trainers",
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
# The user can also use their own dataset and dataloader.
# The expected format of the dataset is a dictionary with the keys as the trainer names.
# For each trainer, the value `data[trainer]` is a tuple with 4 elements: (dataloader, num_node_features, num_graph_labels, train_size)
# - dataloader: a dictionary with keys "train", "val", "test" and values as the corresponding dataloaders
# - num_node_features: number of node features
# - num_graph_labels: number of graph labels
# - train_size: number of training samples
# For the detailed expected format of the data, please refer to the `load_single_dataset` function in `fedgraph/data_process_gc.py`

""" using original features """
print("Preparing data (original features) ...")

data = data_loader_GC(args)
print("Data prepared.")

#######################################################################
# Setup server and trainers
# ------------
# Here we set up the server and trainers for the experiment.
# The server is responsible for federated aggregation (e.g., FedAvg) without knowing the local trainer data.
# The trainers are responsible for local training and testing.
# Before setting up those, the user has to specify the base model for the federated learning that applies for both server and trainers.
# The default model is `GIN` (Graph Isomorphism Network) for graph classification.
# They user can also use other models, but the customized model should be compatible.
# That is, `base_model` must have all the required methods and attributes as the default `GIN`
# For the detailed expected format of the model, please refer to the `fedgraph/gnn_models.py`

ray.init()


@ray.remote(
    num_gpus=num_gpus_per_trainer,
    num_cpus=num_cpus_per_trainer,
    scheduling_strategy="SPREAD",
)
class Trainer(Trainer_GC):
    def __init__(self, idx, splited_data, dataset_trainer_name, *args, **kwargs):  # type: ignore
        print(f"inx: {idx}")
        print(f"dataset_trainer_name: {dataset_trainer_name}")
        """acquire data"""
        dataloaders, num_node_features, num_graph_labels, train_size = splited_data[
            dataset_trainer_name
        ]

        print(f"dataloaders: {dataloaders}")
        print(f"num_node_features: {num_node_features}")
        print(f"num_graph_labels: {num_graph_labels}")
        print(f"train_size: {train_size}")

        """build GIN model"""
        cmodel_gc = base_model(
            nfeat=num_node_features,
            nhid=args.hidden,
            nclass=num_graph_labels,
            nlayer=args.nlayer,
            dropout=args.dropout,
        )

        """build optimizer"""
        optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, cmodel_gc.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        super().__init__(  # type: ignore
            model=cmodel_gc,
            trainer_id=idx,
            trainer_name=dataset_trainer_name,
            train_size=train_size,
            dataloader=dataloaders,
            optimizer=optimizer,
            *args,
            **kwargs,
        )


trainers = [
    Trainer.remote(  # type: ignore
        idx=idx,
        splited_data=data,
        dataset_trainer_name=dataset_trainer_name,
        args=args,
    )
    for idx, dataset_trainer_name in enumerate(data.keys())
]
server = Server_GC(base_model(nlayer=args.nlayer,
                   nhid=args.hidden), args.device)
# TODO: check and modify whether deepcopy should be added.
# trainers = copy.deepcopy(init_trainers)
# server = copy.deepcopy(init_server)

print("\nDone setting up devices.")

################ choose the algorithm to run ################
print(f"Running {args.model} ...")

model_parameters = {
    "SelfTrain": lambda: run_GC_selftrain(
        trainers=trainers, server=server, local_epoch=args.local_epoch
    ),
    "FedAvg": lambda: run_GC_Fed_algorithm(
        trainers=trainers,
        server=server,
        communication_rounds=args.num_rounds,
        local_epoch=args.local_epoch,
        algorithm="FedAvg",
    ),
    "FedProx": lambda: run_GC_Fed_algorithm(
        trainers=trainers,
        server=server,
        communication_rounds=args.num_rounds,
        local_epoch=args.local_epoch,
        algorithm="FedProx",
        mu=args.mu,
    ),
    "GCFL": lambda: run_GCFL_algorithm(
        trainers=trainers,
        server=server,
        communication_rounds=args.num_rounds,
        local_epoch=args.local_epoch,
        EPS_1=args.epsilon1,
        EPS_2=args.epsilon2,
        algorithm_type="gcfl",
    ),
    "GCFL+": lambda: run_GCFL_algorithm(
        trainers=trainers,
        server=server,
        communication_rounds=args.num_rounds,
        local_epoch=args.local_epoch,
        EPS_1=args.epsilon1,
        EPS_2=args.epsilon2,
        algorithm_type="gcfl_plus",
        seq_length=args.seq_length,
        standardize=args.standardize,
    ),
    "GCFL+dWs": lambda: run_GCFL_algorithm(
        trainers=trainers,
        server=server,
        communication_rounds=args.num_rounds,
        local_epoch=args.local_epoch,
        EPS_1=args.epsilon1,
        EPS_2=args.epsilon2,
        algorithm_type="gcfl_plus_dWs",
        seq_length=args.seq_length,
        standardize=args.standardize,
    ),
}

if args.model in model_parameters:
    output = model_parameters[args.model]()
else:
    raise ValueError(f"Unknown model: {args.model}")

#################### save the output ####################
if args.save_files:
    outdir_result = os.path.join(outdir, f"accuracy_seed{args.seed}.csv")
    pd.DataFrame(output).to_csv(outdir_result)
    print(f"The output has been written to file: {outdir_result}")
ray.shutdown()
