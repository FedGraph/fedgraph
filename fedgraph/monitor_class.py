import datetime
import re
import threading
import time
from typing import Any, Dict, List, Optional

import requests  # type: ignore
from ray.util.metrics import Gauge


class Monitor:
    def __init__(self, use_cluster: bool = False) -> None:
        self.use_cluster = use_cluster

        self.pretrain_time_cost_gauge = Gauge(
            "pretrain_time_cost", description="Latencies of pretrain_time_cost in ms."
        )
        self.train_time_cost_gauge = Gauge(
            "train_time_cost", description="Latencies of train_time_cost in ms."
        )
        self.pretrain_node_network_gauge = Gauge(
            "pretrain_node_network",
            description="Network data sent during training per pod.",
        )
        self.train_node_network_gauge = Gauge(
            "train_node_network",
            description="Network data sent during training per pod.",
        )
        self.pretrain_memory_gauge = Gauge(
            "pretrain_memory_usage", description="Memory usage during pretraining."
        )
        self.train_memory_gauge = Gauge(
            "train_memory_usage", description="Memory usage during training."
        )

        # initialization and total communication costs
        self.init_time_cost_gauge = Gauge(
            "init_time_cost", description="Latencies of initialization in ms."
        )

        self.pretrain_theoretical_comm_gauge = Gauge(
            "pretrain_theoretical_comm_MB",
            description="Theoretical communication cost in MB during pretrain phase.",
        )
        self.train_theoretical_comm_gauge = Gauge(
            "train_theoretical_comm_MB",
            description="Theoretical communication cost in MB during train phase.",
        )
        # Timestamp tracking for all phases
        self.init_start_time: Optional[datetime.datetime] = None
        self.init_end_time: Optional[datetime.datetime] = None
        self.pretrain_start_time: Optional[datetime.datetime] = None
        self.pretrain_end_time: Optional[datetime.datetime] = None
        self.train_start_time: Optional[datetime.datetime] = None
        self.train_end_time: Optional[datetime.datetime] = None
        self.total_comm_start_time: Optional[datetime.datetime] = None
        self.total_comm_end_time: Optional[datetime.datetime] = None

        self.current_round: int = 0
        self.initial_network_data: Dict[str, float] = {}
        self.final_network_data: Dict[str, float] = {}
        self.memory_usage_list: List[Any] = []

        # Add large pod mapping
        self.large_pod_mapping: Dict[str, str] = {}
        self.pretrain_theoretical_comm_MB = 0.0
        self.train_theoretical_comm_MB = 0.0
        if self.use_cluster:
            self.memory_thread = threading.Thread(
                target=self.collect_memory, daemon=True
            )
            self.memory_thread.start()

    def add_pretrain_comm_cost(self, upload_mb: float, download_mb: float):
        self.pretrain_theoretical_comm_MB += upload_mb + download_mb
        self.pretrain_theoretical_comm_gauge.set(self.pretrain_theoretical_comm_MB)

    def add_train_comm_cost(self, upload_mb: float, download_mb: float):
        self.train_theoretical_comm_MB += upload_mb + download_mb
        self.train_theoretical_comm_gauge.set(self.train_theoretical_comm_MB)

    def collect_memory(self, interval_seconds=30):
        while True:
            if self.use_cluster:
                memory_data = self._fetch_memory_usage()
                self.memory_usage_list.append(memory_data)
            time.sleep(interval_seconds)

    def _get_network_data(self) -> Dict[str, float]:
        if not self.use_cluster:
            return {}
        response = requests.get(
            "http://prometheus-kube-prometheus-prometheus.prometheus-system.svc.cluster.local:9090/api/v1/query?query=ray_node_network_sent"
        )

        data = response.json()
        pod_data = {}
        large_pod_count = 1

        for item in data["data"]["result"]:
            pod_name = item["metric"]["pod"]

            # Assign unique names for large pods
            if re.search(r"large", pod_name):
                if pod_name not in self.large_pod_mapping:
                    self.large_pod_mapping[pod_name] = f"Large{large_pod_count}"
                    large_pod_count += 1
                pod_data[self.large_pod_mapping[pod_name]] = float(item["value"][1])
            elif re.search(r"head", pod_name):
                pod_data["Server"] = float(item["value"][1])
            else:
                pod_data[pod_name] = float(item["value"][1])

        return pod_data

    def _fetch_memory_usage(self) -> Dict[str, float]:
        if not self.use_cluster:
            return {}
        response = requests.get(
            "http://prometheus-kube-prometheus-prometheus.prometheus-system.svc.cluster.local:9090/api/v1/query?query=ray_node_mem_used"
        )
        data = response.json()
        memory_data = {}
        large_pod_count = 1
        for item in data["data"]["result"]:
            pod_name = item["metric"]["pod"]

            # Use the same large pod naming scheme
            if re.search(r"large", pod_name):
                if pod_name not in self.large_pod_mapping:
                    self.large_pod_mapping[pod_name] = f"Large{large_pod_count}"
                    large_pod_count += 1
                memory_data[self.large_pod_mapping[pod_name]] = float(item["value"][1])
            elif re.search(r"head", pod_name):
                memory_data["Server"] = float(item["value"][1])
            else:
                memory_data["Server"] = float(item["value"][1])

        return memory_data

    # initialization time tracking
    def init_time_start(self) -> None:
        self.init_start_time = datetime.datetime.now()
        if self.use_cluster:
            self.initial_network_data = self._get_network_data()
            print("Initialization start: network data collected.")
        else:
            print("Initialization start time recorded.")

    def init_time_end(self) -> None:
        self.init_end_time = datetime.datetime.now()
        if self.init_start_time is not None and self.init_end_time is not None:
            elapsed = (self.init_end_time - self.init_start_time).total_seconds() * 1000
        else:
            elapsed = 0
        self.init_time_cost_gauge.set(elapsed)
        print(f"//Log init_time: {elapsed} ms //end")
        if self.use_cluster:
            self.final_network_data = self._get_network_data()
            total_diff = sum(
                self.final_network_data.get(pod, 0)
                - self.initial_network_data.get(pod, 0)
                for pod in self.final_network_data
            )
            for pod_name in self.final_network_data:
                diff = self.final_network_data[
                    pod_name
                ] - self.initial_network_data.get(pod_name, 0)
                print(f"//Log {pod_name} init network: {diff} //end")
            print(
                f"//Log Initialization Communication Cost (MB): {total_diff / (1024 * 1024):.2f} //end"
            )

    def pretrain_time_start(self) -> None:
        self.pretrain_start_time = datetime.datetime.now()
        if self.use_cluster:
            self.initial_network_data = self._get_network_data()
        print("Pretrain start time recorded.")
        self.memory_usage_list = []

    def pretrain_time_end(self) -> None:
        if self.pretrain_start_time is not None:
            self.pretrain_end_time = datetime.datetime.now()
            pretrain_duration = (
                self.pretrain_end_time - self.pretrain_start_time
            ).total_seconds() * 1000
            self.pretrain_time_cost_gauge.set(pretrain_duration)
            print(f"//pretrain_time: {pretrain_duration} ms//end")

            if self.use_cluster:
                time.sleep(30)
                self.final_network_data = self._get_network_data()

                # Output memory values for large pods
                for pod_name in self.large_pod_mapping.values():
                    large_memory_values = [
                        memory_data.get(pod_name, 0)
                        for memory_data in self.memory_usage_list
                        if pod_name in memory_data
                    ]
                    if large_memory_values:
                        print(
                            f"//Log Max memory for {pod_name}: {max(large_memory_values)} //end"
                        )
                    else:
                        print(f"No memory values found for {pod_name}.")

                # Output memory value for Server pod
                server_memory_values = [
                    max(
                        memory_data.get("Server", 0)
                        for pod_name in memory_data
                        if re.search(r"Server", pod_name)
                    )
                    for memory_data in self.memory_usage_list
                    if any(re.search(r"Server", pod) for pod in memory_data)
                ]
                if server_memory_values:
                    print(
                        f"//Log Max memory for Server: {max(server_memory_values)} //end"
                    )
                else:
                    print("No memory values found for Server.")

                # Output network data for large pods
                for pod_name, pod_value in self.final_network_data.items():
                    if re.search(r"Large", pod_name):
                        network_diff = pod_value - self.initial_network_data.get(
                            pod_name, 0
                        )
                        self.pretrain_node_network_gauge.set(network_diff)
                        print(f"//Log {pod_name} network: {network_diff} //end")

                if "Server" in self.final_network_data:
                    network_diff = self.final_network_data[
                        "Server"
                    ] - self.initial_network_data.get("Server", 0)
                    self.pretrain_node_network_gauge.set(network_diff)
                    print(f"//Log Server network: {network_diff} //end")

                print("Pretrain end time recorded and duration set to gauge.")

    def train_time_start(self) -> None:
        self.current_round += 1
        self.train_start_time = datetime.datetime.now()
        if self.use_cluster:
            self.initial_network_data = self._get_network_data()
            print("Train start: network data collected.")
        else:
            print("Train start time recorded.")
        self.memory_usage_list = []

    def train_time_end(self) -> None:
        if self.train_start_time is not None:
            self.train_end_time = datetime.datetime.now()
            train_duration = (
                self.train_end_time - self.train_start_time
            ).total_seconds() * 1000
            self.train_time_cost_gauge.set(train_duration)
            print(f"//train_time: {train_duration} ms//end")

            if self.use_cluster:
                time.sleep(30)
                self.final_network_data = self._get_network_data()

                # Output memory values for large pods
                for pod_name in self.large_pod_mapping.values():
                    large_memory_values = [
                        memory_data.get(pod_name, 0)
                        for memory_data in self.memory_usage_list
                        if pod_name in memory_data
                    ]
                    if large_memory_values:
                        print(
                            f"//Log Max memory for {pod_name}: {max(large_memory_values)} //end"
                        )
                    else:
                        print(f"No memory values found for {pod_name}.")

                # Output memory value for Server pod
                server_memory_values = [
                    max(
                        memory_data.get("Server", 0)
                        for pod_name in memory_data
                        if re.search(r"Server", pod_name)
                    )
                    for memory_data in self.memory_usage_list
                    if any(re.search(r"Server", pod) for pod in memory_data)
                ]
                if server_memory_values:
                    print(
                        f"//Log Max memory for Server: {max(server_memory_values)} //end"
                    )
                else:
                    print("No memory values found for Server.")

                # Output network data for large pods
                for pod_name, pod_value in self.final_network_data.items():
                    if re.search(r"Large", pod_name):
                        network_diff = pod_value - self.initial_network_data.get(
                            pod_name, 0
                        )
                        self.train_node_network_gauge.set(network_diff)
                        print(f"//Log {pod_name} network: {network_diff} //end")

                if "Server" in self.final_network_data:
                    network_diff = self.final_network_data[
                        "Server"
                    ] - self.initial_network_data.get("Server", 0)
                    self.train_node_network_gauge.set(network_diff)
                    print(f"//Log Server network: {network_diff} //end")

                print("Train end time recorded and duration set to gauge.")

    def print_comm_cost(self) -> None:
        print(
            f"//Log Theoretical Pretrain Comm Cost: {self.pretrain_theoretical_comm_MB:.2f} MB //end"
        )
        print(
            f"//Log Theoretical Train Comm Cost: {self.train_theoretical_comm_MB:.2f} MB //end"
        )
