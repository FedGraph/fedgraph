"""
Federated Link Prediction Benchmark
===================================

Run benchmarks for various federated link prediction algorithms using a simplified approach.

(Time estimate: 30 minutes)
"""

import os
import time

import attridict
import ray
import torch
import yaml

from fedgraph.federated_methods import run_fedgraph

# Methods to benchmark
methods = ["4D-FED-GNN+", "STFL", "StaticGNN", "FedLink"]

# Country code combinations to test
country_codes_list = [["US"], ["US", "BR"], ["US", "BR", "ID", "TR", "JP"]]

# Number of runs per configuration
runs_per_config = 1

# Define additional required parameters that might be missing from YAML
required_params = {
    "fedgraph_task": "LP",
    "num_cpus_per_trainer": 3,
    "num_gpus_per_trainer": 1 if torch.cuda.is_available() else 0,
    "use_cluster": True,
    "gpu": torch.cuda.is_available(),
    "ray_address": "auto",
}

# Main benchmark loop
for method in methods:
    for country_codes in country_codes_list:
        # Load the base configuration file
        config_file = os.path.join(
            os.path.dirname(__file__), "configs", "config_LP.yaml"
        )
        with open(config_file, "r") as file:
            config = attridict(yaml.safe_load(file))

        # Update the configuration with specific parameters for this run
        config.method = method
        config.country_codes = country_codes

        # Add required parameters that might be missing
        for param, value in required_params.items():
            if not hasattr(config, param):
                setattr(config, param, value)

        # Set dataset path
        if not hasattr(config, "dataset_path") or not config.dataset_path:
            config.dataset_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "data", "LPDataset"
            )

        # Run multiple times for statistical significance
        for i in range(runs_per_config):
            print(f"\n{'-'*80}")
            print(f"Running experiment {i+1}/{runs_per_config}:")
            print(f"Method: {method}, Countries: {', '.join(country_codes)}")
            print(f"{'-'*80}\n")

            # To ensure each run uses a fresh configuration object
            run_config = attridict({})
            for key, value in config.items():
                run_config[key] = value

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
