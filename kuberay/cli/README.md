# KubeRay CLI

[![Build Status](https://github.com/ray-project/kuberay/workflows/Go-build-and-test/badge.svg)](https://github.com/ray-project/kuberay/actions)
[![Go Report Card](https://goreportcard.com/badge/github.com/ray-project/kuberay)](https://goreportcard.com/report/github.com/ray-project/kuberay)

KubeRay CLI provides the ability to manage kuberay resources (ray clusters, compute templates etc) through command line interface.

!!! note

    The KubeRay CLI is an optional interface backed by the KubeRay API server.
    It provides a layer of simplified configuration for KubeRay resources.

    The KubeRay CLI is community-managed and is not officially endorsed by the
    Ray maintainers. At this time, the only officially supported methods for
    managing KubeRay resources are

    - Direct management of KubeRay custom resources via kubectl, kustomize, and Kubernetes language clients.
    - Helm charts.

    KubeRay CLI maintainer contacts (GitHub handles):
    @Jeffwan @scarlet25151

## Installation

Please check [release page](https://github.com/ray-project/kuberay/releases) and download the binaries.

## Prerequisites

- Kuberay operator needs to be running.
- Kuberay apiserver needs to be running and accessible.

## Development

- Kuberay CLI uses [Cobra framework](https://github.com/spf13/cobra) for the CLI application.
- Kuberay CLI depends on kuberay apiserver to manage these resources by sending grpc requests to the kuberay apiserver.

You can build kuberay binary following this way.

```
cd kuberay/cli
go build -o kuberay -a main.go
```

## Usage

### Configure kuberay apiserver endpoint

- Default kuberay apiserver endpoint: `127.0.0.1:8887`.
- If kuberay apiserver is not run locally, this must be set in order to manage ray clusters and ray compute templates.

#### Read current kuberay apiserver endpoint

`./kuberay config get endpoint`

#### Reset kuberay apiserver endpoint to default (`127.0.0.1:8887`)

`./kuberay config reset endpoint`

#### Set kuberay apiserver endpoint

`./kuberay config set endpoint <kuberay apiserver endpoint>`

### Manage Ray Clusters

#### Create a Ray Cluster

```
Usage:
kuberay cluster create [flags]

Flags:
      --environment string               environment of the cluster (valid values: DEV, TESTING, STAGING, PRODUCTION) (default "DEV")
      --head-compute-template string     compute template name for ray head
      --head-image string                ray head image
      --head-service-type string         ray head service type (ClusterIP, NodePort, LoadBalancer) (default "ClusterIP", which creates a headless ClusterIP service)
      --name string                      name of the cluster
  -n, --namespace string                 kubernetes namespace where the cluster will be
      --user string                      SSO username of ray cluster creator
      --version string                   version of the ray cluster (default "1.9.0")
      --worker-compute-template string   compute template name of worker in the first worker group
      --worker-group-name string         first worker group name
      --worker-image string              image of worker in the first worker group
      --worker-replicas uint32           pod replicas of workers in the first worker group (default 1)
```

> Known Limitation: Currently only one worker compute template is supported during creation.

#### Get a Ray Cluster

`./kuberay cluster get -n <namespace> <cluster name>`

#### List Ray Clusters

`./kuberay cluster -n <namespace> list`

#### Delete a Ray Cluster

`./kuberay cluster delete -n <namespace> <cluster name>`

### Manage Ray Compute Template

#### Create a Compute Template
```
Usage:
  kuberay template compute create [flags]

Flags:
      --cpu uint32               ray pod CPU (default 1)
      --gpu uint32               ray head GPU
      --gpu-accelerator string   GPU Accelerator type
      --memory uint32            ray pod memory in GB (default 1)
      --name string              name of the compute template
  -n, --namespace string         kubernetes namespace where the compute template will be stored

```

#### Get a Ray Compute Template
`./kuberay template compute get -n <namespace> <compute template name>`

#### List Ray Compute Templates
`./kuberay template compute list -n <namespace>`

#### Delete a Ray Compute Template
`./kuberay template compute delete -n <namespace> <compute template name>`

## End to end example

Configure the endpoints

```
kubectl port-forward svc/kuberay-apiserver-service 8887:8887 -n ray-system
./kuberay config set endpoint 127.0.0.1:8887
```

Create compute templates

```
./kuberay template compute create -n <namespace> --cpu 2 --memory 4 --name "worker-template"
./kuberay template compute create -n <namespace> --cpu 1 --memory 2 --name "head-template"
```

List compute templates created

```
./kuberay template compute list
```

Create the cluster

```
./kuberay cluster create -n <namespace> --name test-cluster --user jiaxin.shan \
--head-compute-template head-template \
--head-image rayproject/ray:1.9.2 \
--worker-group-name small-wg \
--worker-compute-template worker-template \
--worker-image rayproject/ray:1.9.2
```

List the clusters

```
./kuberay cluster list
```
