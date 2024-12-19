"""
Federated Link Prediction Example
=================================

In this tutorial, you will learn the basic workflow of
Federated Link Prediction with a runnable example. This tutorial assumes that
you have basic familiarity with PyTorch and PyTorch Geometric (PyG).

(Time estimate: 20 minutes)
"""

import argparse
import copy
import datetime
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

from fedgraph.federated_methods import LP_train_global_round
from fedgraph.server_class import Server_LP
from fedgraph.trainer_class import Trainer_LP
from fedgraph.utils_lp import *

# Determine the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append paths relative to the current script's directory
sys.path.append(os.path.join(current_dir, "../fedgraph"))
sys.path.append(os.path.join(current_dir, "../../"))
ray.init(address="auto")

#######################################################################
# Load configuration and check arguments
# --------------------------------------
# Here we load the configuration file for the experiment.
# The configuration file contains the parameters for the experiment.
# The algorithm and dataset (represented by the country code) are specified by the user here.
# We also specify some prechecks to ensure the validity of the arguments.

config_file = os.path.join(current_dir, "configs/config_LP.yaml")
with open(config_file, "r") as file:
    args = attridict(yaml.safe_load(file))

dataset_path = args.dataset_path
print(dataset_path)
path = dataset_path
dataset_path = os.path.join(
    os.path.dirname(path),
    "fedgraph",
    os.path.relpath(path, start=os.path.dirname(path)),
)
print(dataset_path)
global_file_path = os.path.join(dataset_path, "data_global.txt")
traveled_file_path = os.path.join(dataset_path, "traveled_users.txt")

assert args.method in ["STFL", "StaticGNN", "4D-FED-GNN+", "FedLink"], "Invalid method."
assert all(
    code in ["US", "BR", "ID", "TR", "JP"] for code in args.country_codes
), "The country codes should be in 'US', 'BR', 'ID', 'TR', 'JP'"
if args.use_buffer:
    assert args.buffer_size > 0, "The buffer size should be greater than 0."

#######################################################################
# Generate data
# -------------
# Here we generate the data for the experiment.
# If the data is already generated, we load the data from the file.
# Otherwise, we download the data from the website and generate the data.
# We also create the mappings and meta_data for the data.

check_data_files_existance(args.country_codes, dataset_path)

(
    user_id_mapping,
    item_id_mapping,
) = get_global_user_item_mapping(  # get global user and item mapping
    global_file_path=global_file_path
)

meta_data = (
    ["user", "item"],
    [("user", "select", "item"), ("item", "rev_select", "user")],
)  # set meta_data


#######################################################################
# Initialize server and trainers
# ------------------------------
# Starting from this block, we formally begin the training process.
# If you want to run multiple experiments, you can wrap the following code in a loop.
# In this block, we initialize the server and trainers for the experiment.

number_of_clients = len(args.country_codes)
number_of_users, number_of_items = len(user_id_mapping.keys()), len(
    item_id_mapping.keys()
)
num_cpus_per_client = 3
if args.device == "gpu":
    device = torch.device("cuda")
    print("gpu detected")
    num_gpus_per_client = 1
else:
    device = torch.device("cpu")
    num_gpus_per_client = 0
    print("gpu not detected")


@ray.remote(
    num_gpus=num_gpus_per_client,
    num_cpus=num_cpus_per_client,
    scheduling_strategy="SPREAD",
)
class Trainer(Trainer_LP):
    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)


clients = [
    Trainer.remote(  # type: ignore
        i,
        country_code=args.country_codes[i],
        user_id_mapping=user_id_mapping,
        item_id_mapping=item_id_mapping,
        number_of_users=number_of_users,
        number_of_items=number_of_items,
        meta_data=meta_data,
        hidden_channels=args.hidden_channels,
    )
    for i in range(number_of_clients)
]

server = Server_LP(  # the concrete information of users and items is not available in the server
    number_of_users=number_of_users,
    number_of_items=number_of_items,
    meta_data=meta_data,
    trainers=clients,
)
pretrain_time_costs_gauge = Gauge(
    "pretrain_time_cost", description="Latencies of pretrain_time_costs in ms."
)
train_time_costs_gauge = Gauge(
    "train_time_cost", description="Latencies of train_time_costs in ms."
)

#######################################################################
# Training preparation
# --------------------
# Here we prepare the training for the experiment.
# (1) We brodcast the initial model parameter to all clients.
# (2) We determine the start and end time of the conditional information.
# (3) We open the file to record the results if the user wants to record the results.

"""Broadcast the global model parameter to all clients"""
pretrain_start_time = datetime.datetime.now()
global_model_parameter = (
    server.get_model_parameter()
)  # fetch the global model parameter
for i in range(number_of_clients):
    # broadcast the global model parameter to all clients
    clients[i].set_model_parameter.remote(global_model_parameter)


"""Determine the start and end time of the conditional information"""
(
    start_time,
    end_time,
    prediction_days,
    start_time_float_format,
    end_time_float_format,
) = get_start_end_time(online_learning=args.online_learning, method=args.method)

if not args.record_results:
    result_writer = None
    time_writer = None
else:
    file_name = f"{args.method}_buffer_{args.use_buffer}_{args.buffer_size}_online_{args.online_learning}.txt"
    result_writer = open(file_name, "a+")
    time_writer = open("train_time_" + file_name, "a+")


pretrain_time_costs_gauge.set(
    (datetime.datetime.now() - pretrain_start_time).total_seconds() * 1000
)
#######################################################################
# Train the model
# ---------------
# Here we train the model for the experiment.
# For each prediction day, we train the model for each client.
# We also record the results if the user wants to record the results.
for day in range(prediction_days):  # make predictions for each day
    # get the train and test data for each client at the current time step
    for i in range(number_of_clients):
        clients[i].get_train_test_data_at_current_time_step.remote(
            start_time_float_format,
            end_time_float_format,
            use_buffer=args.use_buffer,
            buffer_size=args.buffer_size,
        )
        clients[i].calculate_traveled_user_edge_indices.remote(
            file_path=traveled_file_path
        )

    if args.online_learning:
        print(f"start training for day {day + 1}")
    else:
        print(f"start training")
    for iteration in range(args.global_rounds):
        # each client train on local graph
        print(iteration)
        train_start_time = datetime.datetime.now()
        current_loss = LP_train_global_round(
            server=server,
            local_steps=args.local_steps,
            use_buffer=args.use_buffer,
            method=args.method,
            online_learning=args.online_learning,
            prediction_day=day,
            curr_iteration=iteration,
            global_rounds=args.global_rounds,
            record_results=args.record_results,
            result_writer=result_writer,
            time_writer=time_writer,
        )
        train_time_costs_gauge.set(
            (datetime.datetime.now() - train_start_time).total_seconds() * 1000
        )

    if current_loss >= 0.3:
        print("training is not complete")

    # go to next day
    (
        start_time,
        end_time,
        start_time_float_format,
        end_time_float_format,
    ) = to_next_day(start_time=start_time, end_time=end_time, method=args.method)


if result_writer is not None and time_writer is not None:
    result_writer.close()
    time_writer.close()

print("The whole process has ended")
ray.shutdown()
