# Overview

This python client library provide APIs to handle `raycluster` from your python application.

## Prerequisites

It is assumed that your `k8s cluster in already setup`. Your kubectl configuration is expected to be in  `~/.kube/config` if you are running the code directly from you terminal.

It is also expected that the `kuberay operator` is installed. [Installation instructions are here.](https://github.com/ray-project/kuberay#quick-start)

## Usage

There are multiple levels of using the api with increasing levels of complexity.

### director

This is the easiest form of using the api to create rayclusters with predefined cluster sizes

```python
my_kuberay_api = kuberay_cluster_api.RayClusterApi()

my_cluster_director = kuberay_cluster_builder.Director()

cluster0 = my_cluster_director.build_small_cluster(name="new-cluster0")

if cluster0:
    my_kuberay_api.create_ray_cluster(body=cluster0)
```

the director create the custer definition, and the custer_api acts as the http client sending the create (post) request to the k8s api-server

### cluster_builder

The builder allows you to build the cluster piece by piece, you are can customize more the values of the cluster definition

```python
cluster1 = (
        my_cluster_builder.build_meta(name="new-cluster1")
        .build_head()
        .build_worker(group_name="workers", replicas=3)
        .get_cluster()
    )

if not my_cluster_builder.succeeded:
    return

my_kuberay_api.create_ray_cluster(body=cluster1)
```

### cluster_utils

the cluster_utils gives you even more options to modify your cluster definition, add/remove worker groups, change replicas in a worker group, duplicate a worker group, etc.

```python
my_Cluster_utils = kuberay_cluster_utils.ClusterUtils()

cluster_to_patch, succeeded = my_Cluster_utils.update_worker_group_replicas(
        cluster2, group_name="workers", max_replicas=4, min_replicas=1, replicas=2
    )

if succeeded:
        my_kuberay_api.patch_ray_cluster(
            name=cluster_to_patch["metadata"]["name"], ray_patch=cluster_to_patch
        )
```

### cluster_api

Finally, the cluster_api is the one you always use to implement your cluster change in k8s. You can use it with raw `JSON` if you wish. The director/cluster_builder/cluster_utils are just tools to shield the user from using raw `JSON`.

## Code Organization
```
clients/
└── python-client
    ├── LICENSE
    ├── README.md
    ├── examples
    │   ├── complete-example.py
    │   ├── use-builder.py
    │   ├── use-director.py
    │   ├── use-raw-config_map_with-api.py
    │   ├── use-raw-with-api.py
    │   └── use-utils.py
    ├── pyproject.toml
    ├── python_client
    │   ├── __init__.py
    │   ├── constants.py
    │   ├── kuberay_cluster_api.py
    │   └── utils
    │       ├── __init__.py
    │       ├── kuberay_cluster_builder.py
    │       └── kuberay_cluster_utils.py
    ├── python_client_test
    │   ├── README.md
    │   ├── test_api.py
    │   ├── test_director.py
    │   └── test_utils.py
    └── setup.cfg
```
## For developers

make sure you have installed setuptool

`pip install -U pip setuptools`

#### run the pip command

from the directory `path/to/kuberay/clients/python-client`

`pip install -e .`

#### to uninstall the module run

`pip uninstall python-client`

### For testing run

 `python -m unittest discover 'path/to/kuberay/clients/python-client/python_client_test/'`
