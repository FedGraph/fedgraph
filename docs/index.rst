.. TDC documentation master file, created by
   sphinx-quickstart on Wed Jul  7 12:08:39 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FedGraph
========

`Documentation <https://docs.fedgraph.org>`__ \| `Paper <https://arxiv.org/abs/2201.12433>`__


**FedGraph** *(Federated Graph)* is a library built on top of `PyTorch Geometric (PyG) <https://www.pyg.org/>`_,
`Ray <https://docs.ray.io/>`_, and `PyTorch <https://pytorch.org/>`_ to easily train Graph Neural Networks
under federated or distributed settings.


It supports various federated training methods of graph neural networks under simulated and real federated environments and supports communication between clients and the central server for model update and information aggregation.


Main Focus
----------------------

- **Federated Node Classification with Cross-Client Edges**: Our library supports communicating information stored in other clients without affecting the privacy of users.

- **Federated Graph Classification**: Our library supports federated graph classification with non-IID graphs.


Cross Platform Training
-------------------------

- We support federated training across Linux, macOS, and Windows operating systems.

Library Highlights
------------------

Whether you are a federated learning researcher or a first-time user of federated learning toolkits, here are some reasons to try out FedGraph for federated learning on graph-structured data.

- **Easy-to-use and unified API**: All it takes is 10-20 lines of code to get started with training a federated GNN model. GNN models are PyTorch models provided by PyG and DGL. The federated training process is handled by Ray. We abstract away the complexity of federated graph training and provide a unified API for training and evaluating FedGraph models.

- **Various FedGraph methods**: Most of the state-of-the-art federated graph training methods have been implemented by library developers or authors of research papers and are ready to be applied.

- **Great flexibility**: Existing FedGraph models can easily be extended for conducting your research. Simply inherit the base class of trainers and implement your methods.

- **Large-scale real-world FedGraph Training**: We focus on the need for FedGraph applications in challenging real-world scenarios with privacy preservation, and support learning on large-scale graphs across multiple clients.


----


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   install
   tutorials/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API References

   fedgraph.data_process
   fedgraph.gnn_models
   fedgraph.server_class
   fedgraph.train_func
   fedgraph.trainer_class
   fedgraph.utils

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

   cite
   reference
