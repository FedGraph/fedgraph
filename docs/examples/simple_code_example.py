import sys
sys.path.append('../fedgraph')
from fedgraph.utils import federated_data_loader
from fedgraph.FedGCN import FedGCN_Train
from attrdict import AttrDict
import yaml

with open('config_fedgcn.yaml', 'r') as f:
    config = AttrDict(yaml.safe_load(f))

data = federated_data_loader(config)
FedGCN_Train(config, data)
