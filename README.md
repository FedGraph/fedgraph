# Federated Graph Learning

[pypi-url]: https://pypi.python.org/pypi/fedgraph

**[Documentation]()** | **[Paper](https://arxiv.org/abs/2201.12433)** 

**FedGraph** *(Federated Graph)* is a library built upon [PyTorch](https://pytorch.org/) to easily train Graph Neural Networks (GNNs) under federated (distributed) setting.

It supports various federated training methods of graph neural networks under simulated and real federated environment and supports communication between clients and the central server for model update and information aggregation.

It supports different training tasks:
* Node classification (and link prediction) on a single large graph (Main focus). Our library recoveres information stored in other client without affecting the privacy of users.
* Graph classification on multiple graphs.

Should support different platform with pytorch
* Computers (Linux, Mac OS, Windows)
* Edge devices (Raspberry Pi, Jeston Nano)
* Mobile Phone (Andriod, iOS)


## Library Highlights

Whether you are a federated (distributed) learning researcher or first-time user of federated (distributed) learning toolkits, here are some reasons to try out FedGraph for federated learning on graph-structured data.

* **Easy-to-use and unified API**:
  All it takes is 10-20 lines of code to get started with training a federated GNN model.
  FedGraph is *PyTorch-on-the-rocks*: It utilizes a tensor-centric API and keeps design principles close to vanilla PyTorch.
  If you are already familiar with PyTorch, utilizing FedGraph is straightforward.
* **Comprehensive and well-maintained GNN models**:
  Most of the state-of-the-art federated graph training methods have been implemented by library developers or authors of research papers and are ready to be applied.
* **Great flexibility**:
  Existing FedGraph models can easily be extended for conducting your own research with GNNs.
  Making modifications to existing models or creating new architectures is simple, thanks to its easy-to-use message passing API, and a variety of operators and utility functions.
* **Large-scale real-world GNN models**:
  We focus on the need of FedGNN applications in challenging real-world scenarios, and support learning on diverse types of graphs, including but not limited to: scalable FedGNNs for graphs with millions of nodes; dynamic FedGNNs for node predictions over time; heterogeneous FedGNNs with multiple node types and edge types.


## Installation
```python
pip install fedgraph
```



## Cite

Please cite [our paper](https://arxiv.org/abs/2201.12433) (and the respective papers of the methods used) if you use this code in your own work:

```
@article{yao2022fedgcn,
  title={FedGCN: Convergence and Communication Tradeoffs in Federated Training of Graph Convolutional Networks},
  author={Yao, Yuhang and Joe-Wong, Carlee},
  journal={arXiv preprint arXiv:2201.12433},
  year={2022}
}
```

Feel free to [email us](mailto:yuhangya@andrew.cmu.edu) if you wish your work to be listed in the [external resources]().
If you notice anything unexpected, please open an [issue]() and let us know.
If you have any questions or are missing a specific feature, feel free [to discuss them with us]().
We are motivated to constantly make FedGraph even better.