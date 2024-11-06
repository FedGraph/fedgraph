from fedgraph.federated_methods import run_fedgraph
from fedgraph.data_process import data_loader
import attridict

config={'dataset': 'pubmed',
        'fedgraph_task': 'NC',
        'method': 'FedGCN',
        'global_rounds': 100,
        'local_step': 3,
        'learning_rate': 0.5,
        'n_trainer': 2,
        'num_layers': 2,
        'num_hops': 2,
        'gpu': False,
        'iid_beta': 10000,
        'logdir': './runs',
        'use_encryption': True}

config = attridict(config)

data = data_loader(config)
run_fedgraph(config, data)