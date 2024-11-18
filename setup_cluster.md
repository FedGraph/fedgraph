# Instructions for Setting Up a Ray Cluster on AWS EKS

## Step-by-Step Guide to Push customized Docker ECR image

Configure AWS:

```bash
aws configure
```

Login to ECR (Only for pushing public image, FedGraph already provided public docker image that includes all of the environmental dependencies)

```bash
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
```

Build Docker with amd64 architecture on the cloud and push to ECR

```bash
# You can modify the cloud builder using the CLI, with the docker buildx create command.
docker buildx create --driver cloud ryanli3/fedgraph
# Set your new cloud builder as default on your local machine.
docker buildx use cloud-ryanli3-fedgraph --global
# Build and push image to ECR
docker buildx build --platform linux/amd64 -t public.ecr.aws/i7t1s5i1/fedgraph:gcn . --push
```

## Step-by-Step Guide to Set Up the Ray Cluster

Create an EKS Cluster with eksctl:

```bash
eksctl create cluster -f eks_cluster_config.yaml --timeout=60m
```

After waiting the cluster setup, update kubeconfig for AWS EKS to config the cluster using kubectl:

```bash
# --region and --name can config in the eks_cluster_config.yaml
# metadata:
#   name: user
#   region: us-west-2
aws eks --region us-west-2 update-kubeconfig --name user

```
Optional: Check or switch current cluster only if we have multiple clusters running at the same time:

```bash

kubectl config current-context
kubectl config use-context arn:aws:eks:us-west-2:312849146674:cluster/large


```
Clone the KubeRay Repository, Install Prometheus and Grafana Server

```bash
git clone https://github.com/ray-project/kuberay.git
cd kuberay
./install/prometheus/install.sh
```

Add the KubeRay Helm Repository, Install KubeRay Operator:

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
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
Check every pod is running correctly:
```bash
kubectl get pods
# NAME                                             READY   STATUS    RESTARTS   AGE
# kuberay-operator-7d7998bcdb-bzpkj                1/1     Running   0          35m
# raycluster-autoscaler-head-47mzs                 2/2     Running   0          35m
# raycluster-autoscaler-worker-large-group-grw8w   1/1     Running   0          35m
```

If a pod status is Pending, it means the ray_kubernetes_cluster.yaml requests too many resources than the cluster can provide, delete the ray_kubernetes_cluster, modify the config and restart the kubernetes
```bash
kubectl delete -f ray_kubernetes_cluster.yaml
kubectl apply -f ray_kubernetes_cluster.yaml
```

Forward Port for Ray Dashboard, Prometheus, and Grafana

```bash
kubectl port-forward service/raycluster-autoscaler-head-svc 8265:8265
# raycluster-autoscaler-head-xxx is the pod name
kubectl port-forward raycluster-autoscaler-head-47mzs 8080:8080
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
}' --address http://localhost:8265 -- python run.py

```

Stop a Ray Job:

```bash
# raysubmit_xxx is the job name that can be found via 
ray job stop raysubmit_m5PN9xqV6drJQ8k2 --address http://localhost:8265
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

Finally, Delete the node first and then delete EKS Cluster:

```bash
kubectl get nodes -o name | xargs kubectl delete
eksctl delete cluster --region us-west-2 --name user
```

## Step to Push Data to Hugging Face Hub CLI

Use the following command to login to the Hugging Face Hub CLI tool when you set "save: True" in node classification tasks if you haven't done so already:

```bash
huggingface-cli login
```
