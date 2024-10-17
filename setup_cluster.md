# Instructions for Setting Up and Deleting a Ray Cluster on AWS EKS

## Step-by-Step Guide to Push customized Docker ECR image

Configure AWS:

```bash
aws configure
```

Login to ECR

```bash
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
```

Build Docker with amd64 architecture on the cloud and push to ECR

```bash
# You can add the cloud builder using the CLI, with the docker buildx create command.
docker buildx create --driver cloud ryanli3/fedgraph
# Set your new cloud builder as default on your local machine.
docker buildx use cloud-ryanli3-fedgraph --global
# Build and push image to ECR
docker buildx build --platform linux/amd64 -t public.ecr.aws/i7t1s5i1/fedgraph:gcn . --push
```

## Step-by-Step Guide to Set Up the Ray Cluster

Create an EKS Cluster with eksctl:

```bash
eksctl create cluster -f eks_cluster_config.yaml
# eksctl create cluster --name test --region us-east-1 --nodegroup-name standard-workers --node-type g4dn.xlarge --nodes 1 --nodes-min 1 --nodes-max 4 --managed
```

Update kubeconfig for AWS EKS:

```bash

aws eks --region us-west-2 update-kubeconfig --name large

```

Clone the KubeRay Repository and Install Prometheus

```bash
git clone https://github.com/ray-project/kuberay.git
cd kuberay
./install/prometheus/install.sh
```

Add the KubeRay Helm Repository:

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
```

Install KubeRay Operator:

```bash
helm install kuberay-operator kuberay/kuberay-operator --version 1.1.1
```

Navigate to the Example Configurations Directory:

```bash
cd docs/examples/configs
```

Apply Ray Kubernetes Cluster and Ingress Configurations:

```bash
kubectl apply -f ray_kubernetes_cluster.yaml
kubectl apply -f ray_kubernetes_ingress.yaml
```

Forward Port for Ray Dashboard:

```bash
kubectl port-forward service/raycluster-autoscaler-head-svc 8265:8265
```

Forward Ports for Ray Dashboard, Prometheus, and Grafana

```bash
kubectl port-forward raycluster-autoscaler-head-hnclp 8080:8080
kubectl port-forward prometheus-prometheus-kube-prometheus-prometheus-0 -n prometheus-system 9090:9090
kubectl port-forward deployment/prometheus-grafana -n prometheus-system 3000:3000
```

Final Check

```bash
kubectl get pods --all-namespaces -o wide
```

Submit a Ray Job:

```bash
cd fedgraph
ray job submit --runtime-env-json '{
  "working_dir": "./"
}' --address http://localhost:8265 -- python docs/examples/benchmark_NC.py

```

Stop a Ray Job:

```bash
ray job stop raysubmit_aKFaB94jPakXuQT1 --address http://localhost:8265
```

## How to Delete the Ray Cluster

Delete the RayCluster Custom Resource:

```bash
cd docs/examples/configs
kubectl delete -f ray_kubernetes_cluster.yaml
kubectl delete -f ray_kubernetes_ingress.yaml
```

Confirm that the RayCluster Pods are Terminated:

```bash
kubectl get pods
# Ensure the output shows no Ray pods except kuberay-operator
```

Uninstall the KubeRay Operator Helm Chart:

```bash
helm uninstall kuberay-operator
```

Confirm that the KubeRay Operator Pod is Terminated:

```bash
kubectl get pods -A
```

Finally, Delete the EKS Cluster:

```bash
eksctl delete cluster --region us-west-2 --name large
```

## Step 1: Pushing Data to Hugging Face Hub CLI

Use the following command to install the Hugging Face Hub CLI tool if you haven't done so already:

```bash
pip install huggingface_hub
huggingface-cli login
```
