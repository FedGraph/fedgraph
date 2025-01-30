#!/bin/bash

# =======================================
# Script to Set Up a Ray Cluster on AWS EKS
# =======================================

# Function to check command success
check_command() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed. Exiting."
        exit 1
    fi
}

# Step 1: Configure AWS credentials
echo "Configuring AWS credentials..."
read -p "Enter AWS Access Key ID: " aws_access_key
read -p "Enter AWS Secret Access Key: " aws_secret_key
read -p "Enter AWS Default Region (e.g., us-east-1): " aws_region

aws configure set aws_access_key_id $aws_access_key
check_command "AWS Access Key configuration"
aws configure set aws_secret_access_key $aws_secret_key
check_command "AWS Secret Key configuration"
aws configure set region $aws_region
check_command "AWS Region configuration"

# Step 2: Login to AWS ECR Public
echo "Logging in to AWS ECR Public..."
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
check_command "AWS ECR login"

# Step 3: Build and push Docker image to ECR
echo "Building and pushing Docker image to ECR..."

# Define the builder name
BUILDER_NAME="fedgraph-builder"

# Check if the builder already exists
if docker buildx ls | grep -q $BUILDER_NAME; then
    echo "Builder $BUILDER_NAME already exists. Using the existing builder."
    docker buildx use $BUILDER_NAME --global
else
    echo "Creating a new builder: $BUILDER_NAME"
    docker buildx create --driver docker-container --name $BUILDER_NAME
    check_command "Docker buildx create"
    docker buildx use $BUILDER_NAME --global
    check_command "Docker buildx use"
fi

# Build and push the Docker image
docker buildx build --platform linux/amd64 -t public.ecr.aws/i7t1s5i1/fedgraph:img . --push
check_command "Docker build and push"

# Step 4: Create EKS Cluster with a dynamic name
CLUSTER_NAME="mlarge-$(date +%s)"
echo "Using dynamic cluster name: $CLUSTER_NAME"

echo "Creating EKS cluster..."
if [ ! -f "eks_cluster_config.yaml" ]; then
    echo "Error: eks_cluster_config.yaml not found in the current directory."
    exit 1
fi

# Modify the configuration file to include the dynamic cluster name
sed -i.bak "s/^  name: .*/  name: $CLUSTER_NAME/" eks_cluster_config.yaml

# Create the cluster using the modified configuration file
eksctl create cluster -f eks_cluster_config.yaml --timeout=60m
check_command "EKS cluster creation"

# Step 5: Update kubeconfig for AWS EKS
echo "Updating kubeconfig for AWS EKS..."
aws eks --region $aws_region update-kubeconfig --name $CLUSTER_NAME
check_command "Kubeconfig update"

# Step 6: Clone KubeRay Repository and Install Prometheus/Grafana
echo "Cloning KubeRay repository and installing Prometheus and Grafana..."
if [ ! -d "kuberay" ]; then
    git clone https://github.com/ray-project/kuberay.git
fi
cd kuberay
./install/prometheus/install.sh
check_command "Prometheus and Grafana installation"

# Step 7: Install KubeRay Operator via Helm
echo "Installing KubeRay Operator..."
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator --version 1.1.1
check_command "KubeRay Operator installation"

# Step 8: Deploy Ray Kubernetes Cluster and Ingress
echo "Deploying Ray Kubernetes Cluster and Ingress..."
cd docs/examples/configs
kubectl apply -f ray_kubernetes_cluster.yaml
check_command "Ray Kubernetes Cluster deployment"
kubectl apply -f ray_kubernetes_ingress.yaml
check_command "Ray Kubernetes Ingress deployment"

# Step 9: Verify Pod Status
echo "Checking pod status..."
kubectl get pods
echo "If any pod status is Pending, modify ray_kubernetes_cluster.yaml and reapply."

# Step 10: Handle Pending Pod Issues (Optional)
echo "To handle Pending pods, delete the cluster and reapply:"
echo "kubectl delete -f ray_kubernetes_cluster.yaml"
echo "kubectl apply -f ray_kubernetes_cluster.yaml"

# Step 11: Forward Ports for Ray Dashboard, Prometheus, and Grafana
echo "Forwarding ports for Ray Dashboard, Prometheus, and Grafana..."
kubectl port-forward service/raycluster-autoscaler-head-svc 8265:8265 &
kubectl port-forward raycluster-autoscaler-head-47mzs 8080:8080 &
kubectl port-forward prometheus-prometheus-kube-prometheus-prometheus-0 -n prometheus-system 9090:9090 &
kubectl port-forward deployment/prometheus-grafana -n prometheus-system 3000:3000 &
check_command "Port forwarding"

# Step 12: Final Check
echo "Final check for all pods across namespaces:"
kubectl get pods --all-namespaces -o wide

# Step 13: Submit a Ray Job (Optional)
echo "To submit a Ray job, run:"
echo "cd fedgraph"
echo "ray job submit --runtime-env-json '{
  \"working_dir\": \"./\",
  \"excludes\": [\".git\"]
}' --address http://localhost:8265 -- python3 run.py"

# Step 14: Stop a Ray Job (Optional)
echo "To stop a Ray job, use:"
echo "ray job stop <job_id> --address http://localhost:8265"

# Step 15: Clean Up Resources
echo "To clean up resources, delete the RayCluster Custom Resource and EKS cluster:"
echo "cd docs/examples/configs"
echo "kubectl delete -f ray_kubernetes_cluster.yaml"
echo "kubectl delete -f ray_kubernetes_ingress.yaml"
echo "kubectl get nodes -o name | xargs kubectl delete"
echo "eksctl delete cluster --region $aws_region --name $CLUSTER_NAME"

echo "Setup completed successfully!"
