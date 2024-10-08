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
    "num_layers": 2,
    "num_hops": 2,
    "gpu": False,
    "iid_beta": 10000,
    "logdir": "./runs",
}

config = attridict(config)

data = data_loader(config)
run_fedgraph(config, data)
