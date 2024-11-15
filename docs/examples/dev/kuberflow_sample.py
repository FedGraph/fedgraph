from kubernetes import client, config, utils
from kubernetes.client import ApiException
from kubernetes.stream import portforward
import time
import yaml

config.load_kube_config()

api_client = client.ApiClient()
v1 = client.CoreV1Api(api_client)
custom_api = client.CustomObjectsApi(api_client) 


def create_resource_from_yaml(file_path):
    with open(file_path) as f:
        resource_yaml = yaml.safe_load(f)

    try:
        utils.create_from_yaml(api_client, file_path)
        print(f"Resource from {file_path} created successfully.")
    except ApiException as e:
        print(f"Exception when creating resource: {e}")


def wait_for_pods(namespace, label_selector, target_phase="Running", timeout=600):
    start_time = time.time()
    while time.time() - start_time < timeout:
        pods = v1.list_namespaced_pod(
            namespace, label_selector=label_selector).items
        all_in_target_phase = all(
            pod.status.phase == target_phase for pod in pods)

        if all_in_target_phase:
            print(
                f"All pods with label '{label_selector}' are in {target_phase} state.")
            return True
        else:
            print(f"Waiting for pods to reach {target_phase} state...")
            time.sleep(10)

    print(
        f"Timeout reached: Pods did not reach {target_phase} state within {timeout} seconds.")
    return False
