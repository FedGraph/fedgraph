"""
Federated Node Classification Benchmark
=======================================

Run benchmarks for various federated node classification algorithms using a simplified approach.

(Time estimate: 30 minutes)
"""

import os
import time

import attridict
import ray
import torch
import yaml

from fedgraph.federated_methods import run_fedgraph

# Datasets to benchmark
datasets = [
    "ogbn-arxiv"
]  # You can add more: ["cora", "citeseer", "ogbn-arxiv", "ogbn-products"]

# Number of trainers to test
n_trainers = [1000]

# Number of hops for neighbor aggregation
num_hops_list = [0, 1]

# Distribution types for node partitioning
distribution_list_ogbn = ["average"]
distribution_list_other = ["average"]
# You can expand these: distribution_list_ogbn = ["average", "lognormal", "exponential", "powerlaw"]

# IID Beta values to test (controls how IID the data distribution is)
iid_betas = [10000.0, 100.0, 10.0]

# Number of runs per configuration
runs_per_config = 1

# Define additional required parameters that might be missing from YAML
required_params = {
    "fedgraph_task": "NC",
    "num_cpus_per_trainer": 4,
    "num_gpus_per_trainer": 1 if torch.cuda.is_available() else 0,
    "use_cluster": True,
    "num_rounds": 200,
    "local_step": 1,
    "learning_rate": 0.1,
    "num_layers": 2,
    "logdir": "./runs",
    "use_huggingface": False,
    "saveto_huggingface": False,
}

# Main benchmark loop
for dataset in datasets:
    # Determine whether to use GPU based on dataset
    gpu = False  # Set to "ogbn" in dataset if you want to use GPU for certain datasets

    # Choose distribution list based on dataset and number of trainers
    distribution_list = (
        distribution_list_other
        if n_trainers[0] > 10 or not gpu
        else distribution_list_ogbn
    )

    # Set batch sizes based on dataset
    if dataset == "ogbn-arxiv":
        batch_sizes = [-1]
    elif dataset == "ogbn-products":
        batch_sizes = [-1]
    elif dataset == "ogbn-papers100M":
        batch_sizes = [16, 32, 64, -1]
    else:
        batch_sizes = [-1]

    for n_trainer in n_trainers:
        for num_hops in num_hops_list:
            for distribution_type in distribution_list:
                for iid_beta in iid_betas:
                    for batch_size in batch_sizes:
                        # Load the base configuration
                        config = attridict({})

                        # Set all required parameters
                        for param, value in required_params.items():
                            setattr(config, param, value)

                        # Set experiment-specific parameters
                        config.dataset = dataset
                        config.method = "fedgcn" if num_hops > 0 else "FedAvg"
                        config.batch_size = batch_size
                        config.n_trainer = n_trainer
                        config.num_hops = num_hops
                        config.iid_beta = iid_beta
                        config.distribution_type = distribution_type
                        config.gpu = gpu

                        # Run multiple times for statistical significance
                        for i in range(runs_per_config):
                            print(f"\n{'-'*80}")
                            print(f"Running experiment {i+1}/{runs_per_config}:")
                            print(
                                f"Dataset: {dataset}, Trainers: {n_trainer}, Distribution: {distribution_type}, "
                                + f"IID Beta: {iid_beta}, Hops: {num_hops}, Batch Size: {batch_size}"
                            )
                            print(f"{'-'*80}\n")

                            # Run the federated learning process with clean Ray environment
                            try:
                                # Make sure Ray is shut down from any previous runs
                                if ray.is_initialized():
                                    ray.shutdown()

                                # Run the experiment
                                run_fedgraph(config)
                            except Exception as e:
                                print(f"Error running experiment: {e}")
                                print(f"Configuration: {config}")
                            finally:
                                # Always ensure Ray is shut down before the next experiment
                                if ray.is_initialized():
                                    ray.shutdown()

                            # Add a short delay between runs
                            time.sleep(5)

print("Benchmark completed.")
