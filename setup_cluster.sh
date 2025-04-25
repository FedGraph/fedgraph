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
# Note: You do NOT need to rebuild and push the Docker image every time.
# Only rebuild if you have added new dependencies or made changes to the Dockerfile.

# echo "Logging in to AWS ECR Public..."
# aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
# check_command "AWS ECR login"

# # Step 3: Build and push Docker image to ECR
# echo "Building and pushing Docker image to ECR..."

# # Define the builder name
# BUILDER_NAME="fedgraph-builder"

# # Check if the builder already exists
# if docker buildx ls | grep -q $BUILDER_NAME; then
#     echo "Builder $BUILDER_NAME already exists. Using the existing builder."
#     docker buildx use $BUILDER_NAME --global
# else
#     echo "Creating a new builder: $BUILDER_NAME"
#     docker buildx create --driver docker-container --name $BUILDER_NAME
#     check_command "Docker buildx create"
#     docker buildx use $BUILDER_NAME --global
#     check_command "Docker buildx use"
# fi

# # Build and push the Docker image
# docker buildx build --platform linux/amd64 -t public.ecr.aws/i7t1s5i1/fedgraph:img . --push
# check_command "Docker build and push"

# Step 4: Check if EKS Cluster exists
CLUSTER_NAME="mlarge-1739510276"  # You can keep a fixed name or change it dynamically
echo "Checking if the EKS cluster '$CLUSTER_NAME' exists..."

eksctl get cluster --name $CLUSTER_NAME --region $aws_region > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "Cluster '$CLUSTER_NAME' already exists. Skipping cluster creation."
else
    echo "Cluster '$CLUSTER_NAME' does not exist. Creating EKS cluster..."

    if [ ! -f "ray_cluster_configs/eks_cluster_config.yaml" ]; then
        echo "Error: eks_cluster_config.yaml not found in the ray_cluster_configs folder."
        exit 1
    fi

    # Modify the configuration file to include the dynamic cluster name
    sed -i.bak "s/^  name: .*/  name: $CLUSTER_NAME/" ray_cluster_configs/eks_cluster_config.yaml

    # Create the cluster using the modified configuration file
    eksctl create cluster -f ray_cluster_configs/eks_cluster_config.yaml --timeout=60m
    check_command "EKS cluster creation"
fi

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
echo "Deploying Ray Kubernetes Cluster and Ingress..."Forwarding ports for Ray Dashboard, Prometheus, and Grafana
# Ensure the script starts from the root directory of the project
cd "$(dirname "$0")/.."
# Apply the Ray Kubernetes cluster and ingress YAML files from the correct path
kubectl apply -f ray_cluster_configs/ray_kubernetes_cluster.yaml
check_command "Ray Kubernetes Cluster deployment"
kubectl apply -f ray_cluster_configs/ray_kubernetes_ingress.yaml
check_command "Ray Kubernetes Ingress deployment"

# Step 9: Verify Pod Status
echo "Checking pod status..."
kubectl get pods
echo "If any pod status is Pending, modify ray_kubernetes_cluster.yaml and reapply."

# Step 10: Handle Pending Pod Issues (Optional)
echo "To handle Pending pods, delete the cluster and reapply:"
echo "kubectl delete -f ray_cluster_configs/ray_kubernetes_cluster.yaml"
echo "kubectl apply -f ray_cluster_configs/ray_kubernetes_cluster.yaml"

# Step 11: Forward Ports for Ray Dashboard, Prometheus, and Grafana
# Note: You must open separate terminal windows for each port forwarding command below.
# Do NOT run them all in one terminal with background (&) processes, as that may cause issues.
echo "Open a new terminal and run the following commands one by one in separate terminals:"
echo "kubectl port-forward service/raycluster-autoscaler-head-svc 8265:8265"
# To get <ray-head-pod-name>, run `kubectl get pods`
echo "kubectl port-forward <ray-head-pod-name> 8080:8080"
echo "kubectl port-forward prometheus-prometheus-kube-prometheus-prometheus-0 -n prometheus-system 9090:9090"
echo "kubectl port-forward deployment/prometheus-grafana -n prometheus-system 3000:3000"

# Step 12: Final Check
echo "Final check for all pods across namespaces:"
kubectl get pods --all-namespaces -o wide

# Step 13: Submit a Ray Job
echo "To submit a Ray job, run:"
echo "cd fedgraph"
echo "ray job submit \
  --address http://localhost:8265 \
  --runtime-env-json '{
    "working_dir": ".",
    "excludes": [".git", "__pycache__", "outputs", "fedgraph/he_training_context.pkl"],
    "pip": ["fsspec==2023.6.0", "huggingface_hub", "tenseal"]
  }' \
  -- python benchmark/benchmark_GC.py"

# Step 14: Stop a Ray Job (Optional)
echo "To stop a Ray job, use:"
echo "ray job stop <job_id> --address http://localhost:8265"

# Step 15: Clean Up Resources
echo "To clean up resources, delete the RayCluster Custom Resource and EKS cluster:"
echo "kubectl delete -f ray_cluster_configs/ray_kubernetes_cluster.yaml"
echo "kubectl delete -f ray_cluster_configs/ray_kubernetes_ingress.yaml"
echo "kubectl get nodes -o name | xargs kubectl delete"
echo "eksctl delete cluster --region $aws_region --name $CLUSTER_NAME"
# eksctl delete cluster --region us-east-1 --name mlarge-1739510276

echo "Setup completed successfully!"
