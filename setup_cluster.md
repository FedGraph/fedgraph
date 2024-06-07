# Instructions for Setting Up and Deleting a Ray Cluster on AWS EKS

## Step-by-Step Guide to Set Up the Ray Cluster

Create an EKS Cluster with eksctl:

```bash
eksctl create cluster --name test --region us-east-1 --nodegroup-name standard-workers --node-type g4dn.xlarge --nodes 1 --nodes-min 1 --nodes-max 4 --managed
```

Update kubeconfig for AWS EKS:

```bash
aws eks --region us-east-1 update-kubeconfig --name test
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

Get the Head Pod Name:

```bash
export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)
echo $HEAD_POD
```

Initialize Ray in the Head Pod:

```bash
kubectl exec -it $HEAD_POD -- python -c "import ray; ray.init(); print(ray.cluster_resources())"
```

Execute Example Python Script in the Head Pod:

```bash
kubectl exec -it $HEAD_POD -- python docs/examples/intro_LP.py
```

Forward Port for Ray Dashboard:

```bash
kubectl port-forward service/raycluster-autoscaler-head-svc 8265:8265
```

Submit a Ray Job:

```bash
cd fedgraph
ray job submit --runtime-env-json '{
  "working_dir": "./"
}' --address http://localhost:8265 -- python docs/examples/intro_LP.py

```

## How to Delete the Ray Cluster

Delete the RayCluster Custom Resource:

```bash
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
kubectl get pods
```

Finally, Delete the EKS Cluster:

```bash
eksctl delete cluster --name test
```
