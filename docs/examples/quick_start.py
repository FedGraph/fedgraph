import attridict

from fedgraph.data_process import data_loader
from fedgraph.federated_methods import run_fedgraph

config = {
    # Task, Method, and Dataset Settings
    "fedgraph_task": "NC",
    "dataset": "cora",
    "method": "FedGCN",  # Federated learning method, e.g., "FedGCN"
    "iid_beta": 10000,  # Dirichlet distribution parameter for label distribution among clients
    "distribution_type": "average",  # Distribution type among clients
    # Training Configuration
    "global_rounds": 100,
    "local_step": 3,
    "learning_rate": 0.5,
    "n_trainer": 2,
    "batch_size": -1,  # -1 indicates full batch training
    # Model Structure
    "num_layers": 2,
    "num_hops": 1,  # Number of n-hop neighbors for client communication
    # Resource and Hardware Settings
    "gpu": False,
    "num_cpus_per_trainer": 1,
    "num_gpus_per_trainer": 0,
    # Logging and Output Configuration
    "logdir": "./runs",
    # Security and Privacy
    "use_encryption": False,  # Whether to use Homomorphic Encryption for secure aggregation
    # Dataset Handling Options
    "use_huggingface": False,  # Load dataset directly from Hugging Face Hub
    "saveto_huggingface": False,  # Save partitioned dataset to Hugging Face Hub
    # Scalability and Cluster Configuration
    "use_cluster": False,  # Use Kubernetes for scalability if True
}
config = attridict(config)
if config.fedgraph_task != "NC" or not config.use_huggingface:
    data = data_loader(config)
    run_fedgraph(config, data)
else:
    run_fedgraph(config)


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
    # Scalability and Cluster Configuration
    "use_cluster": False,  # Use Kubernetes for scalability if True
}


config = attridict(config)
if config.fedgraph_task != "NC" or not config.use_huggingface:
    data = data_loader(config)
    run_fedgraph(config, data)
else:
    run_fedgraph(config)
