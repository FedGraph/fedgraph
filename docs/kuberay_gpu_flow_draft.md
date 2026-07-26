# FedGraph KubeRay GPU Draft Flow

This is a draft-only walkthrough. It connects the cluster files into one flow
without launching AWS, Kubernetes, Ray, or benchmark jobs.

The runnable source of truth is now `deploy/kuberay/`, `scripts/kuberay/`, and
`docs/kuberay_gpu_first_run.md`; the draft files remain teaching references.

## Files Added For Review

- `Dockerfile.gpu.draft`: CUDA/Ray/FedGraph image sketch.
- `ray_cluster_configs/eks_gpu_cluster_config.draft.yaml`: EKS node groups.
- `ray_cluster_configs/ray_kubernetes_gpu_cluster.draft.yaml`: KubeRay RayCluster.
- `kuberay/config/prometheus/dcgmExporterServiceMonitor.draft.yaml`: GPU metrics monitor placeholder.

## Runtime Flow

### 1. Build A GPU-Capable Image

The existing `Dockerfile` installs CPU PyTorch. GPU worker pods need a CUDA
PyTorch image, matching PyG extension wheels, Ray, and FedGraph. The draft image
starts from a Ray GPU base image and installs the FedGraph package.

What this component provides:

- Python runtime for the Ray head, Ray workers, and FedGraph job driver.
- CUDA-compatible PyTorch/PyG for trainers that call `torch.device("cuda")`.
- FedGraph code and benchmark scripts inside the pod filesystem.

What consumes it:

- `ray_kubernetes_gpu_cluster.draft.yaml` references the image in the head and
  worker pod templates.

Before launching:

- Confirm the Ray image tag exists.
- Confirm `torch.cuda.is_available()` is true inside the image on a GPU node.
- Push the image to ECR and replace `public.ecr.aws/i7t1s5i1/fedgraph:gpu-draft`.

### 2. Create EKS Capacity

`eks_gpu_cluster_config.draft.yaml` illustrates the EC2 node pools; the active
definition is `deploy/kuberay/eks-cluster.yaml`:

- `system-head-cpu`: fixed CPU node for system services, monitoring, Ray, and the server.
- `ray-gpu-workers`: T4 GPU nodes that can scale from 0 to 5 workers.

What this component provides:

- Real CPU, RAM, disk, and GPU capacity.
- Kubernetes node labels such as `ray-node-type: head` and
  `ray-node-type: gpu-worker`.
- Optional GPU taint so non-GPU pods do not land on GPU nodes accidentally.

What consumes it:

- Kubernetes scheduler uses node labels/taints when placing Ray pods.
- RayCluster pod templates use `nodeSelector` and `tolerations` to request those nodes.

Before launching:

- Confirm GPU instance quota and regional availability.
- Cluster Autoscaler is installed with a dedicated IRSA role and scales the GPU
  managed node group between zero and five nodes.
- AL2023 GPU nodes and `eksctl` provide the NVIDIA device plugin integration.

### 3. Let KubeRay Create The Ray Cluster

`ray_kubernetes_gpu_cluster.draft.yaml` defines the desired RayCluster state.
The KubeRay operator watches this custom resource and creates the Ray head pod,
worker pods, and head service.

What this component provides:

- One Ray head pod on the CPU node.
- Zero or more Ray GPU worker pods.
- Ray dashboard, Ray Jobs API, Ray client, and metrics ports.
- Ray autoscaler bounds: `minReplicas: 0`, `maxReplicas: 5`.

What consumes it:

- KubeRay operator reconciles this YAML into Kubernetes pods/services.
- Ray scheduler uses each worker pod's advertised resources.
- FedGraph jobs connect to the Ray head dashboard/jobs endpoint.

Important wiring:

```text
nvidia.com/gpu: "1" on worker pod
  -> KubeRay/Ray advertises 1 logical GPU on that Ray node
    -> FedGraph Trainer actor with num_gpus_per_trainer can be placed there
```

Before launching:

- Confirm `apiVersion` matches the installed KubeRay CRD.
- Confirm `rayVersion` matches the image Ray version.
- Confirm the head pod has enough memory for driver/server/data loading.
- Confirm worker `maxReplicas` is high enough for the chosen experiment.

### 4. Submit A FedGraph Job

The job should be submitted to the Ray head after port-forwarding the dashboard
or exposing it through an ingress.

Example smoke command shape:

```bash
ray job submit \
  --address http://localhost:8265 \
  --runtime-env-json '{"working_dir": ".", "excludes": [".git", "benchmark/results", "dataset"]}' \
  -- python benchmark/benchmark_NC_batch_size_convergence.py \
      --experiment-name kuberay_gpu_cora_smoke_bs32_full_20r_seed42 \
      --dataset cora \
      --num-hops 2 \
      --batch-sizes 32,-1 \
      --rounds 20 \
      --local-step 3 \
      --n-trainer 5 \
      --iid-betas 10000 \
      --seeds 42 \
      --num-layers 2 \
      --gpu \
      --server-device cpu \
      --num-gpus-per-trainer 1 \
      --num-cpus-per-trainer 1 \
      --continue-on-error
```

What this component provides:

- The FedGraph driver process.
- The CPU-only FedGraph server object inside the Ray-head driver process.
- Ray trainer actors created by FedGraph.

What consumes it:

- Ray head receives the Ray job.
- Ray scheduler places trainer actors on GPU-capable Ray workers.
- Trainers move tensors to CUDA because `--gpu` sets `args.gpu=True`.

Important wiring:

```text
--n-trainer 5
  -> FedGraph creates 5 Ray Trainer actors

--num-gpus-per-trainer 1
  -> Ray needs 5 one-GPU worker pods and 5 GPU EC2 nodes

--server-device cpu
  -> the driver-side server stays on the CPU-only Ray head
```

For OGBN-Arxiv, avoid assuming GPU VRAM is the limiting resource. The previous
T4 failures were host RAM pressure, so head and worker memory requests matter.

### 5. Observe With Prometheus And Grafana

The active platform path deploys `kube-prometheus-stack`, Cluster Autoscaler
with a `ServiceMonitor`, then KubeRay and Ray `ServiceMonitor`/`PodMonitor`
objects.

What this component provides:

- Kubernetes pod/node CPU and memory history.
- Ray head and worker metrics.
- Ray autoscaler and dashboard metrics.
- GPU metrics if DCGM exporter is installed and the draft ServiceMonitor is adapted.

What consumes it:

- Grafana dashboards.
- Manual debugging queries during benchmarks.
- Future alerting or scaling decisions.

Metrics to inspect first:

- Pod status: Pending, Running, OOMKilled, restarts.
- Node memory and pod memory usage.
- Ray available resources and actor placement.
- Ray object store usage.
- GPU utilization and GPU memory through DCGM exporter.

Prometheus records what happened. It does not split PyTorch GPU memory into
model parameters versus activations; that still needs PyTorch instrumentation
inside FedGraph trainers.

## Settled First-Run Decisions

- Does the head pod need more memory for OGBN-Arxiv driver/server work?
- The Cora smoke forces one trainer per T4 GPU and one worker pod per GPU node.
- Cluster Autoscaler handles EC2 nodes; Ray autoscaler handles Ray worker pods.
- The server uses the temporary `--server-device cpu` override on the Ray head.
- Which GPU image tag should become the real ECR image?
- Which DCGM exporter namespace/labels exist after GPU monitoring is installed?
