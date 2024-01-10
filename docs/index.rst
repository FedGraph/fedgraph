.. TDC documentation master file, created by
   sphinx-quickstart on Wed Jul  7 12:08:39 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. figure:: pygod_logo.png
    :scale: 30%
    :alt: logo

----


.. image:: https://img.shields.io/pypi/v/pygod.svg?color=brightgreen
   :target: https://pypi.org/project/pygod/
   :alt: PyPI version

.. image:: https://readthedocs.org/projects/pygod/badge/?version=latest
   :target: https://docs.pygod.org/en/latest/?badge=latest
   :alt: Documentation status

.. image:: https://img.shields.io/github/stars/pygod-team/pygod.svg
   :target: https://github.com/pygod-team/pygod/stargazers
   :alt: GitHub stars

.. image:: https://img.shields.io/github/forks/pygod-team/pygod.svg?color=blue
   :target: https://github.com/pygod-team/pygod/network
   :alt: GitHub forks

.. image:: https://static.pepy.tech/personalized-badge/fedgraph?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads
   :target: https://pepy.tech/project/fedgraph
   :alt: PyPI downloads

.. image:: https://github.com/pygod-team/pygod/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/pygod-team/pygod/actions/workflows/testing.yml
   :alt: testing

.. image:: https://coveralls.io/repos/github/pygod-team/pygod/badge.svg?branch=main
   :target: https://coveralls.io/github/pygod-team/pygod?branch=main
   :alt: Coverage Status

.. image:: https://img.shields.io/github/license/pygod-team/pygod.svg
   :target: https://github.com/pygod-team/pygod/blob/master/LICENSE
   :alt: License

----

PyGOD is a **Python library** for **graph outlier detection** (anomaly detection).
This exciting yet challenging field has many key applications, e.g., detecting
suspicious activities in social networks :cite:`dou2020enhancing`  and security systems :cite:`cai2021structural`.

PyGOD includes **10+** graph outlier detection algorithms.
For consistency and accessibility, PyGOD is developed on top of `PyTorch Geometric (PyG) <https://www.pyg.org/>`_
and `PyTorch <https://pytorch.org/>`_, and follows the API design of `PyOD <https://github.com/yzhao062/pyod>`_.
See examples below for detecting outliers with PyGOD in 5 lines!


**PyGOD is featured for**:

* **Unified APIs, detailed documentation, and interactive examples** across various graph-based algorithms.
* **Comprehensive coverage** of 10+ graph outlier detectors.
* **Full support of detections at multiple levels**, such as node-, edge-, and graph-level tasks.
* **Scalable design for processing large graphs** via mini-batch and sampling.
* **Streamline data processing with PyG**--fully compatible with PyG data objects.

**Outlier Detection Using PyGOD with 5 Lines of Code**\ :


.. code-block:: python


    # train a dominant detector
    from pygod.detector import DOMINANT

    model = DOMINANT(num_layers=4, epoch=20)  # hyperparameters can be set here
    model.fit(train_data)  # input data is a PyG data object

    # get outlier scores on the training data (transductive setting)
    score = model.decision_score_

    # predict labels and scores on the testing data (inductive setting)
    pred, score = model.predict(test_data, return_score=True)

----


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   install
   tutorials/index
   api_cc

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: API References

   pygod.detector
   pygod.generator
   pygod.metric
   pygod.nn
   pygod.nn.conv
   pygod.nn.encoder
   pygod.nn.decoder
   pygod.nn.functional
   pygod.utils

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

   cite
   team
   reference
