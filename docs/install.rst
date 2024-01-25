Installation
============


It is recommended to use **pip** for installation.
Please make sure **the latest version** is installed, as FedGraph is updated frequently:

.. code-block:: bash

   pip install fedgraph            # normal install
   pip install --upgrade fedgraph  # or update if needed


Alternatively, you could clone and run setup.py file:

.. code-block:: bash

   git clone https://github.com/FedGraph/fedgraph.git
   cd fedgraph
   pip install .

**Required Dependencies**\ :

* python>=3.8
* ray
* tensorboard

**Note on PyG and PyTorch Installation**\ :
FedGraph depends on `torch <https://https://pytorch.org/get-started/locally/>`_ and `torch_geometric (including its optional dependencies) <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#>`_.
To streamline the installation, FedGraph does **NOT** install these libraries for you.
Please install them from the above links for running FedGraph:

* torch>=2.0.0
* torch_geometric>=2.3.0
