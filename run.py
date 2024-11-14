import attridict

from fedgraph.data_process import data_loader
from fedgraph.federated_methods import run_fedgraph

config = {
    "fedgraph_task": "GC",
    # General configuration
    "algorithm": "FedAvg",  # Options: "FedAvg", "GCFL", "SelfTrain"
    # Dataset configuration
    "dataset": "IMDB-BINARY",
    "is_multiple_dataset": False,
    "datapath": "./data",
    "convert_x": False,
    "overlap": False,
    # Setup configuration
    "device": "cpu",
    "seed": 10,
    "seed_split_data": 42,
    # Model parameters
    "num_trainers": 2,
    "num_rounds": 200,  # Used by "FedAvg" and "GCFL" (not used in "SelfTrain")
    "local_epoch": 1,  # Used by "FedAvg" and "GCFL"
    # Specific for "SelfTrain" (used instead of "num_rounds" and "local_epoch")
    "local_epoch_selftrain": 200,
    "lr": 0.001,
    "weight_decay": 0.0005,
    "nlayer": 3,  # Number of model layers
    "hidden": 64,  # Hidden layer dimension
    "dropout": 0.5,  # Dropout rate
    "batch_size": 128,
    # Resource and Hardware Settings
    "gpu": False,
    "num_cpus_per_trainer": 1,
    "num_gpus_per_trainer": 0,
    # FedAvg specific parameter
    "mu": 0.01,  # Regularization parameter, only used in "FedAvg"
    # GCFL specific parameters
    "standardize": False,  # Used only in "GCFL"
    "seq_length": 5,  # Sequence length, only used in "GCFL"
    "epsilon1": 0.05,  # Privacy epsilon1, specific to "GCFL"
    "epsilon2": 0.1,  # Privacy epsilon2, specific to "GCFL"
    # Output configuration
    "outbase": "./outputs",
    "save_files": False,
}


config = attridict(config)
if config.fedgraph_task != "NC" or not config.use_huggingface:
    data = data_loader(config)
    run_fedgraph(config, data)
else:
    run_fedgraph(config)
