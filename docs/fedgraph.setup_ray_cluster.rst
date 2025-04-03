Set Up the Ray Cluster
======================

This section provides a step-by-step guide to set up a Ray Cluster on AWS EKS.

It is recommended to use the following script to set up the cluster. The script will guide you through the setup process on AWS, including Docker image building, EKS cluster creation, and deployment of Ray on Kubernetes.


Components Overview
-------------------

The following table outlines the key components used in setting up a Ray cluster on AWS EKS:

.. list-table:: Ray Cluster Components
   :widths: 25 75
   :header-rows: 1

   * - Component
     - Purpose
   * - Ray
     - Provides distributed computing for machine learning (e.g., FedGraph tasks).
   * - Kubernetes
     - Orchestrates and manages Ray's deployment in AWS EKS.
   * - AWS EKS
     - Provides the cloud infrastructure for running Kubernetes and Ray.
   * - KubeRay
     - Automates Ray cluster setup and management in Kubernetes.
   * - Helm
     - Installs KubeRay and other Kubernetes services.
   * - Ray Dashboard, Prometheus, Grafana
     - Monitor the Ray cluster’s performance.

=======

Prerequisites
-------------
Before you begin, ensure you have the following:

* AWS CLI installed and configured.
* Docker installed and running.
* Helm installed.
* kubectl installed.
* AWS ECR credentials.
* AWS EKS access.

Steps to Set Up Ray Cluster on AWS EKS
--------------------------------------

1. **Configure AWS Credentials**

   Run the following commands to set up AWS credentials:

   .. code-block:: bash

       aws configure set aws_access_key_id <YOUR_AWS_ACCESS_KEY>
       aws configure set aws_secret_access_key <YOUR_AWS_SECRET_KEY>
       aws configure set region <YOUR_AWS_REGION>

   Make sure to replace `<YOUR_AWS_ACCESS_KEY>`, `<YOUR_AWS_SECRET_KEY>`, and `<YOUR_AWS_REGION>` with your actual credentials and region.

2. **Log in to AWS ECR Public**

   To push Docker images to AWS ECR, you need to log in to the public ECR:

   .. code-block:: bash

       aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws

3. **Build and Push Docker Image to ECR**

   To build and push the Docker image, run the following commands:

   .. code-block:: bash

       docker buildx build --platform linux/amd64 -t public.ecr.aws/i7t1s5i1/fedgraph:img . --push

4. **Create EKS Cluster**

   Create a dynamic EKS cluster with the following command:

   .. code-block:: bash

       eksctl create cluster -f eks_cluster_config.yaml --timeout=60m

5. **Update Kubeconfig**

   Update your kubeconfig file to access the newly created EKS cluster:

   .. code-block:: bash

       aws eks --region <YOUR_AWS_REGION> update-kubeconfig --name <CLUSTER_NAME>

6. **Clone KubeRay Repository and Install Prometheus/Grafana**

   Clone the KubeRay repository and install monitoring tools:

   .. code-block:: bash

       git clone https://github.com/ray-project/kuberay.git
       cd kuberay
       ./install/prometheus/install.sh

7. **Install KubeRay Operator**

   To manage Ray on Kubernetes, you need to install the KubeRay operator:

   .. code-block:: bash

       helm repo add kuberay https://ray-project.github.io/kuberay-helm/
       helm repo update
       helm install kuberay-operator kuberay/kuberay-operator --version 1.1.1

8. **Deploy Ray Kubernetes Cluster**

   Apply the Kubernetes configuration to deploy Ray on EKS:

   .. code-block:: bash

       kubectl apply -f ray_kubernetes_cluster.yaml
       kubectl apply -f ray_kubernetes_ingress.yaml

9. **Verify Pod Status**

   Check the status of the pods to ensure that they are running:

   .. code-block:: bash

       kubectl get pods

10. **Port Forwarding for Ray Dashboard, Prometheus, and Grafana**

    Forward the necessary ports for accessing the Ray dashboard and monitoring tools:

    .. code-block:: bash

        kubectl port-forward service/raycluster-autoscaler-head-svc 8265:8265 &
        kubectl port-forward raycluster-autoscaler-head-47mzs 8080:8080 &
        kubectl port-forward prometheus-prometheus-kube-prometheus-prometheus-0 -n prometheus-system 9090:9090 &
        kubectl port-forward deployment/prometheus-grafana -n prometheus-system 3000:3000 &

11. **Final Check**

    To ensure everything is set up correctly, perform a final check:

    .. code-block:: bash

        kubectl get pods --all-namespaces -o wide

12. **Submit a Ray Job (Optional)**

    If you want to submit a Ray job, use the following command:

    .. code-block:: bash

        ray job submit --runtime-env-json '{"working_dir": "./", "excludes": [".git"]}' --address http://localhost:8265 -- python3 run.py

13. **Stop a Ray Job (Optional)**

    To stop a Ray job, use:

    .. code-block:: bash

        ray job stop <job_id> --address http://localhost:8265

14. **Clean Up Resources**

    To clean up resources, delete the RayCluster and EKS cluster:

    .. code-block:: bash

        kubectl delete -f ray_kubernetes_cluster.yaml
        kubectl delete -f ray_kubernetes_ingress.yaml
        kubectl get nodes -o name | xargs kubectl delete
        eksctl delete cluster --region <YOUR_AWS_REGION> --name <CLUSTER_NAME>

Setup completed successfully!
