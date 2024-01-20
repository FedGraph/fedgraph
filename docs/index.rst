.. TDC documentation master file, created by
   sphinx-quickstart on Wed Jul  7 12:08:39 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FedGraph
========

`Documentation <https://docs.fedgraph.org>`__ \| `Paper <https://arxiv.org/abs/2201.12433>`__


**FedGraph** *(Federated Graph)* is a library built upon
`PyTorch <https://pytorch.org/>`__ to easily train Graph Neural Networks
(GNNs) under federated (distributed) setting.

It supports various federated training methods of graph neural networks
under simulated and real federated environment and supports
communication between clients and the central server for model update
and information aggregation.

Various Training Tasks
----------------------

-  **Node classification (and link prediction) on a single large graph
   (Main focus)**: Our library recoveres information stored in other
   client without affecting the privacy of users.

-  **Graph classification on multiple graphs**

Cross Platform Deployment
-------------------------

-  **Computers (Linux, Mac OS, Windows)**
-  **Edge devices (Raspberry Pi, Jeston Nano)**
-  **Mobile phones (Andriod, iOS)**

Library Highlights
------------------

Whether you are a federated (distributed) learning researcher or
first-time user of federated (distributed) learning toolkits, here are
some reasons to try out FedGraph for federated learning on
graph-structured data.

-  **Easy-to-use and unified API**: All it takes is 10-20 lines of code
   to get started with training a federated GNN model. FedGraph is
   *PyTorch-on-the-rocks*: It utilizes a tensor-centric API and keeps
   design principles close to vanilla PyTorch. If you are already
   familiar with PyTorch, utilizing FedGraph is straightforward.
-  **Comprehensive and well-maintained GNN models**: Most of the
   state-of-the-art federated graph training methods have been
   implemented by library developers or authors of research papers and
   are ready to be applied.
-  **Great flexibility**: Existing FedGraph models can easily be
   extended for conducting your own research with GNNs. Making
   modifications to existing models or creating new architectures is
   simple, thanks to its easy-to-use message passing API, and a variety
   of operators and utility functions.
-  **Large-scale real-world GNN models**: We focus on the need of FedGNN
   applications in challenging real-world scenarios, and support
   learning on diverse types of graphs, including but not limited to:
   scalable FedGNNs for graphs with millions of nodes; dynamic FedGNNs
   for node predictions over time; heterogeneous FedGNNs with multiple
   node types and edge types.

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
