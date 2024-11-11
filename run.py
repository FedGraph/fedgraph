import attridict

from fedgraph.data_process import data_loader
from fedgraph.federated_methods import run_fedgraph

config = {
    "dataset": "cora",
    "fedgraph_task": "NC",
    "method": "FedGCN",
    "global_rounds": 100,
    "local_step": 3,
    "learning_rate": 0.5,
    "n_trainer": 2,
    "num_layers": 2,  # number of model layers
    "num_hops": 0,  # number of n hop neighbours for client communication
    "gpu": False,
    "iid_beta": 10000,  # dirichlet distrbution for different labels
    "logdir": "./runs",
    "use_encryption": False,  # whether use HE encrytion or not
    "use_huggingface": False,  # load dataset directly from hugging_face
    "saveto_huggingface": False,  # save the tensor after partition to hugging_face
    "num_cpus_per_trainer": 1,
    "num_gpus_per_trainer": 0,
    "use_cluster": False,  # whether use kubernetes for scalability or not
    "distribution_type": "average",  # the node number distribution among clients
    "batch_size": -1,  # -1 is full batch
}

config = attridict(config)
if not config.use_huggingface:
    data = data_loader(config)
    run_fedgraph(config, data)
else:
    run_fedgraph(config)
