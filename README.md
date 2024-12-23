# Federated Graph Learning

[pypi-url]: https://pypi.python.org/pypi/fedgraph

**[Documentation](https://docs.fedgraph.org)** | **[Paper](https://arxiv.org/abs/2410.06340)**

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
from fedgraph.federated_methods import run_fedgraph

import attridict

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
run_fedgraph(config)
```

## Cite

Please cite [our paper](https://arxiv.org/abs/2410.06340) (and the respective papers of the methods used) if you use this code in your own work:

```
@article{yao2024fedgraph,
  title={FedGraph: A Research Library and Benchmark for Federated Graph Learning},
  author={Yao, Yuhang and Li, Yuan and Fan, Xinyi and Li, Junhao and Liu, Kay and Jin, Weizhao and Ravi, Srivatsan and Yu, Philip S and Joe-Wong, Carlee},
  journal={arXiv preprint arXiv:2410.06340},
  year={2024}
}
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
