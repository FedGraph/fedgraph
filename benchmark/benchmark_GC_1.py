"""
Federated Graph Classification Benchmark
=======================================

Run benchmarks for various federated graph classification algorithms using a simplified approach.

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
    "IMDB-BINARY",
    "IMDB-MULTI",
    "MUTAG",
    "BZR",
    "COX2",
    "DHFR",
    "AIDS",
    # "PTC-MR",  # not found
    # "ENZYMES",  # error with 10 clients
    # "DD",
    # "PROTEINS",
    # "COLLAB",
    # "NCI1",
]

# Algorithms to benchmark
algorithms = ["SelfTrain", "FedAvg", "FedProx", "GCFL", "GCFL+", "GCFL+dWs"]

# Number of trainers to test
trainer_numbers = [3]

# Number of runs per configuration
runs_per_config = 3

# Define additional required parameters that might be missing from YAML
required_params = {
    "fedgraph_task": "GC",
    "num_cpus_per_trainer": 2,
    "num_gpus_per_trainer": 1 if torch.cuda.is_available() else 0,
    "use_cluster": False,  # Set to True to enable monitoring
    "gpu": torch.cuda.is_available(),
}

# Main benchmark loop
for dataset_name in datasets:
    for algorithm in algorithms:
        # Load the appropriate configuration file for the algorithm
        config_file = os.path.join(
            os.path.dirname(__file__), "configs", f"config_GC_{algorithm}.yaml"
        )
        with open(config_file, "r") as file:
            config = attridict(yaml.safe_load(file))

        # Update the configuration with specific parameters for this run
        config.dataset = dataset_name

        # Add required parameters that might be missing
        for param, value in required_params.items():
            if not hasattr(config, param):
                setattr(config, param, value)

        for trainer_num in trainer_numbers:
            # Set the number of trainers
            config.num_trainers = trainer_num

            # Run multiple times for statistical significance
            for i in range(runs_per_config):
                print(f"\n{'-'*80}")
                print(f"Running experiment {i+1}/{runs_per_config}:")
                print(
                    f"Algorithm: {algorithm}, Dataset: {dataset_name}, Trainers: {trainer_num}"
                )
                print(f"{'-'*80}\n")

                # To ensure each run uses a fresh configuration object
                run_config = attridict({})
                for key, value in config.items():
                    run_config[key] = value

                # Ensure proper parameter naming
                if hasattr(run_config, "model") and not hasattr(
                    run_config, "algorithm"
                ):
                    run_config.algorithm = run_config.model
                elif not hasattr(run_config, "model"):
                    run_config.model = algorithm
                    run_config.algorithm = algorithm

                # Run the federated learning process with clean Ray environment
                try:
                    # Make sure Ray is shut down from any previous runs
                    if ray.is_initialized():
                        ray.shutdown()

                    # Run the experiment
                    run_fedgraph(run_config)
                except Exception as e:
                    print(f"Error running experiment: {e}")
                    print(f"Configuration: {run_config}")
                finally:
                    # Always ensure Ray is shut down before the next experiment
                    if ray.is_initialized():
                        ray.shutdown()

                # Add a short delay between runs
                time.sleep(5)

print("Benchmark completed.")
