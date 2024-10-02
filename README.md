# Federated Graph Learning

[pypi-url]: https://pypi.python.org/pypi/fedgraph

**[Documentation](https://docs.fedgraph.org)** | **[Paper](https://arxiv.org/abs/2201.12433)**

**FedGraph** *(Federated Graph)* is a library built on top of [PyTorch Geometric (PyG)](https://www.pyg.org/),
[Ray](https://docs.ray.io/), and [PyTorch](https://pytorch.org/) to easily train Graph Neural Networks
under federated or distributed settings.

It supports various federated training methods of graph neural networks under simulated and real federated environments and supports communication between clients and the central server for model update and information aggregation.

## Main Focus
- **Federated Node Classification with Cross-Client Edges**: Our library supports communicating information stored in other clients without affecting the privacy of users.
- **Federated Link Prediction on Dynamic Graphs**: Our library supports balancing temporal heterogeneity across clients with privacy preservation.
- **Federated Graph Classification**: Our library supports federated graph classification with non-IID graphs.




## Cross Platform Training

- We support federated training across Linux, macOS, and Windows operating systems.

## Library Highlights

Whether you are a federated learning researcher or a first-time user of federated learning toolkits, here are some reasons to try out FedGraph for federated learning on graph-structured data.

- **Easy-to-use and unified API**: All it takes is 10-20 lines of code to get started with training a federated GNN model. GNN models are PyTorch models provided by PyG and DGL. The federated training process is handled by Ray. We abstract away the complexity of federated graph training and provide a unified API for training and evaluating FedGraph models.

- **Various FedGraph methods**: Most of the state-of-the-art federated graph training methods have been implemented by library developers or authors of research papers and are ready to be applied.

- **Great flexibility**: Existing FedGraph models can easily be extended for conducting your research. Simply inherit the base class of trainers and implement your methods.

- **Large-scale real-world FedGraph Training**: We focus on the need for FedGraph applications in challenging real-world scenarios with privacy preservation, and support learning on large-scale graphs across multiple clients.

## Installation
```python
pip install fedgraph
```

## Quick Start
```python
from fedgraph.federated_methods import FedGCN_Train
from fedgraph.utils import federated_data_loader
import attridict

config={'dataset': 'cora',
        'fedtype': 'fedgcn',
        'global_rounds': 100,
        'local_step': 3,
        'learning_rate': 0.5,
        'n_trainer': 2,
        'num_layers': 2,
        'num_hops': 2,
        'gpu': False,
        'iid_beta': 10000,
        'logdir': './runs'}

config = attridict(config)

data = federated_data_loader(config)
FedGCN_Train(config, data)
```

## Cite

Please cite [our paper](https://arxiv.org/abs/2201.12433) (and the respective papers of the methods used) if you use this code in your own work:

```
@article{yao2023fedgcn,
  title={FedGCN: Convergence-Communication Tradeoffs in Federated Training of Graph Convolutional Networks},
  author={Yao, Yuhang and Jin, Weizhao and Ravi, Srivatsan and Joe-Wong, Carlee},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```

Feel free to [email us](mailto:yuhangya@andrew.cmu.edu) if you wish your work to be listed in the [external resources]().
If you notice anything unexpected, please open an [issue]() and let us know.
If you have any questions or are missing a specific feature, feel free [to discuss them with us]().
We are motivated to constantly make FedGraph even better.
