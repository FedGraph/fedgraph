# FedGraph KubeRay GPU First Run

This workflow turns the walkthrough drafts into individually reviewable stages.
Every script prints a plan by default. Only an explicit `--build`, `--push`,
`--apply`, `--submit`, `--ray`, or `--cluster` changes external state.

## Fixed Versions and Topology

| Component | Version or setting |
| --- | --- |
| Kubernetes | 1.34 on EKS |
| AWS region | `eu-central-1` |
| Ray image/runtime | 2.56.0, Python 3.12, CUDA 12.1 |
| KubeRay operator chart | 1.6.0 |
| kube-prometheus-stack chart | 86.0.0 |
| NVIDIA DCGM exporter chart | 4.8.2 |
| Cluster Autoscaler chart/runtime | 9.53.0 / 1.34.2 |
| Ray namespace | `default` |
| KubeRay operator namespace | `kuberay-system` |
| Prometheus/Grafana namespace | `prometheus-system` |
| DCGM exporter namespace | `gpu-monitoring` |

The cluster starts with one `m7i.xlarge` CPU node shared by Kubernetes system
services, monitoring, the KubeRay operator, and the CPU-only Ray head. The GPU
node group uses `g4dn.xlarge` T4 instances and starts at zero.

Ray may create up to five one-GPU worker pods. Cluster Autoscaler then creates
up to five GPU EC2 nodes for those pending pods and returns idle GPU nodes to
zero after Ray removes its workers. The CPU node and EKS control plane remain
running until the explicit cluster teardown stage.

## Prerequisites

Install AWS CLI v2, Docker with Buildx, `eksctl` 0.176 or newer, a Kubernetes
1.34-compatible `kubectl`, and Helm 3 using their official installation guides.
The scripts never write long-lived AWS access keys. AWS CLI and `eksctl` use
the normal AWS credential provider chain. On the current EC2 host, temporary
credentials and the default `eu-central-1` region come from the attached IAM
role through EC2 instance metadata. `aws sts get-caller-identity` verifies that
resolved identity; it does not ask for or create credentials.

The active AWS identity needs permission to create EKS, EC2, VPC,
CloudFormation, IAM, Auto Scaling, and EBS CSI resources. Run the preflight
before building anything:

```bash
./scripts/kuberay/00-preflight.sh
```

At the time this workflow was prepared, the current EC2 role could identify
itself but was denied `eks:DescribeClusterVersions`. Cluster creation cannot
start until that role is expanded or a different authorized AWS profile is
selected.

## 1. Build and Push the Image

Use one image reference consistently in every shell. An immutable commit tag is
recommended:

```bash
export FEDGRAPH_IMAGE_TAG="gpu-ray2.56.0-py312-cu121-$(git rev-parse --short HEAD)"
./scripts/kuberay/10-build-image.sh
./scripts/kuberay/10-build-image.sh --build
./scripts/kuberay/10-build-image.sh --push
```

The plan command changes nothing. `--build` loads the image only into local
Docker. `--push` authenticates to the configured public ECR registry and pushes
the image that Kubernetes nodes will pull. Override the repository with
`FEDGRAPH_IMAGE_REPOSITORY` when needed.

## 2. Create EKS

```bash
./scripts/kuberay/20-create-eks.sh
export FEDGRAPH_ALLOW_AWS_CREATE=yes
./scripts/kuberay/20-create-eks.sh --apply
```

The first command only prints the `eksctl --dry-run` command. The apply command
creates billable AWS resources, updates kubeconfig, and lists the nodes.

## 3. Install Kubernetes Platform Services

```bash
./scripts/kuberay/30-install-platform.sh
./scripts/kuberay/30-install-platform.sh --apply
./scripts/kuberay/35-install-gpu-monitoring.sh
./scripts/kuberay/35-install-gpu-monitoring.sh --apply
```

The first apply installs gp3 storage, Prometheus, Grafana, Cluster Autoscaler,
the KubeRay operator, and Ray metric monitors. Cluster Autoscaler reuses the
IRSA-backed service account created by `eksctl`. The second apply installs one
DCGM exporter pod per GPU node. Both scripts use `helm upgrade --install`, so
rerunning them reconciles the existing releases.

## 4. Deploy Ray

Export the same image tag used by the image-build shell, then run:

```bash
export FEDGRAPH_IMAGE_TAG="gpu-ray2.56.0-py312-cu121-$(git rev-parse --short HEAD)"
./scripts/kuberay/40-deploy-ray.sh
./scripts/kuberay/40-deploy-ray.sh --check
./scripts/kuberay/40-deploy-ray.sh --apply
./scripts/kuberay/status.sh
```

`--check` asks the Kubernetes API server to validate the rendered RayCluster.
`--apply` creates it and waits for the head pod. GPU worker pods remain at zero
until Ray has GPU work to schedule.

## 5. Addresses, Ports, and Namespaces

Cluster services are private `ClusterIP` services. `kubectl port-forward`
temporarily maps a local loopback port to one of those services:

| User-facing endpoint | Kubernetes object | Namespace | Local address |
| --- | --- | --- | --- |
| Ray dashboard and Jobs API | `service/fedgraph-gpu-head-svc:8265` | `default` | `http://127.0.0.1:8265` |
| Prometheus UI/API | `service/prometheus-kube-prometheus-prometheus:9090` | `prometheus-system` | `http://127.0.0.1:9090` |
| Grafana UI | `service/prometheus-grafana:80` | `prometheus-system` | `http://127.0.0.1:3000` |

Print all commands, then start each needed forward in its own terminal:

```bash
./scripts/kuberay/50-port-forward.sh
./scripts/kuberay/50-port-forward.sh ray
./scripts/kuberay/50-port-forward.sh prometheus
./scripts/kuberay/50-port-forward.sh grafana
```

Inside the cluster, Ray uses Kubernetes DNS names such as
`prometheus-kube-prometheus-prometheus.prometheus-system.svc:9090`. Outside the
cluster, users and the local Ray CLI use `127.0.0.1` only while the corresponding
port-forward process is running.

## 6. Submit the Cora Smoke Job

Keep the Ray port-forward running, then in another terminal:

```bash
./scripts/kuberay/60-submit-cora-smoke.sh
./scripts/kuberay/60-submit-cora-smoke.sh --submit
```

The job runs batch sizes 32 and full-batch for 20 rounds, with five trainer
actors. Each actor requests one CPU and one full GPU. Ray therefore creates
five worker pods, and the Kubernetes scheduler can place at most one worker pod
on each one-GPU node. `--server-device cpu` is a temporary override for the
CPU-only Ray head. Results are written under:

```text
/results/nc_batch_size_convergence/kuberay_cora_smoke_bs32_full_20r_seed42
```

`/results` is a 20 GiB encrypted retained gp3 volume mounted only on the Ray
head. Ray temporary logs and object-spill data use pod-local `emptyDir` storage.
Prometheus and Grafana use separate 20 GiB and 5 GiB `gp3-delete` volumes so
their storage is removed during guarded cluster teardown.

## 7. Scale Down And Teardown

Two independent autoscalers participate after the job finishes:

```text
Ray autoscaler: Ray actor demand -> Ray worker pod count
Cluster Autoscaler: pending/unused pods -> GPU EC2 node count
```

Ray waits five idle minutes before removing worker pods. Cluster Autoscaler
then waits until those GPU nodes are unneeded before reducing the GPU managed
node group to zero. This stops GPU instance charges but leaves the CPU node,
EKS control plane, and persistent results volume running.

Copy results out before deleting the cluster:

```bash
head_pod="$(kubectl get pod -n default \
  -l 'ray.io/cluster=fedgraph-gpu,ray.io/node-type=head' \
  -o jsonpath='{.items[0].metadata.name}')"
kubectl cp \
  "default/${head_pod}:/results/nc_batch_size_convergence/kuberay_cora_smoke_bs32_full_20r_seed42" \
  benchmark/results/nc_batch_size_convergence/kuberay_cora_smoke_bs32_full_20r_seed42
```

The teardown script is plan-only without a mode. `--ray` removes Ray while
leaving monitoring and results available. `--cluster` requires two explicit
confirmations, changes the retained results volume to `Delete`, removes all
workflow PVCs and Helm releases, and deletes EKS:

```bash
./scripts/kuberay/70-teardown.sh
FEDGRAPH_ALLOW_RAY_DELETE=yes ./scripts/kuberay/70-teardown.sh --ray
FEDGRAPH_ALLOW_AWS_DELETE=yes FEDGRAPH_RESULTS_SAVED=yes \
  ./scripts/kuberay/70-teardown.sh --cluster
```

The full teardown does not remove images stored in ECR Public.

## First Queries

Use Prometheus to verify each layer before interpreting a graph:

```promql
up
ray_cluster_active_nodes
DCGM_FI_DEV_GPU_UTIL
DCGM_FI_DEV_FB_USED
DCGM_FI_DEV_FB_FREE
```

The Ray metric names available can vary by runtime release; search for the
`ray_` prefix in the Prometheus expression browser. DCGM reports total GPU
memory usage by device and adds Kubernetes pod identity labels when the driver
can map a GPU process to a pod. It does not split PyTorch VRAM into parameters,
activations, gradients, and optimizer state; that still requires application
instrumentation such as PyTorch memory snapshots.
