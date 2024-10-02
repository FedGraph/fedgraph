import datetime
import re
import threading
import time
from typing import Any, Dict, List, Optional

import requests  # type: ignore
from ray.util.metrics import Gauge


class Monitor:
    def __init__(self) -> None:
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
        self.pretrain_start_time: Optional[datetime.datetime] = None
        self.pretrain_end_time: Optional[datetime.datetime] = None
        self.train_start_time: Optional[datetime.datetime] = None
        self.train_end_time: Optional[datetime.datetime] = None
        self.current_round = 0
        self.initial_network_data: Dict[str, float] = {}
        self.final_network_data: Dict[str, float] = {}
        self.memory_usage_list: List[Any] = []

        self.memory_thread = threading.Thread(target=self.collect_memory, daemon=True)
        self.memory_thread.start()

    def collect_memory(self, interval_seconds=30):
        while True:
            memory_data = self._fetch_memory_usage()
            # total_memory = sum(memory_data.values())
            # print(f"in collect_memory's data:{memory_data}")
            self.memory_usage_list.append(memory_data)
            time.sleep(interval_seconds)

    def _get_network_data(self) -> Dict[str, float]:
        response = requests.get(
            "http://prometheus-kube-prometheus-prometheus.prometheus-system:9090/api/v1/query?query=ray_node_network_sent"
        )
        data = response.json()
        return {
            item["metric"]["pod"]: float(item["value"][1])
            for item in data["data"]["result"]
        }

    def _fetch_memory_usage(self) -> Dict[str, float]:
        response = requests.get(
            f"http://prometheus-kube-prometheus-prometheus.prometheus-system:9090/api/v1/query?query=ray_node_mem_used"
        )
        data = response.json()
        return {
            item["metric"]["pod"]: float(item["value"][1])
            for item in data["data"]["result"]
        }

    def pretrain_time_start(self) -> None:
        self.pretrain_start_time = datetime.datetime.now()
        self.initial_network_data = self._get_network_data()
        print("Pretrain start time recorded and initial network data collected.")
        self.memory_usage_list = []

    def pretrain_time_end(self, interval_seconds=30) -> None:
        if self.pretrain_start_time is not None:
            self.pretrain_end_time = datetime.datetime.now()
            pretrain_duration = (
                self.pretrain_end_time - self.pretrain_start_time
            ).total_seconds() * 1000
            self.pretrain_time_cost_gauge.set(pretrain_duration)
            print(f"//pretrain_time: {pretrain_duration} //end")
            time.sleep(interval_seconds)
            # print("Sleeping through intervals")
            self.final_network_data = self._get_network_data()
            # print(f"memory_list:{self.memory_usage_list}")
            large_memory_values = [
                max(memory_data[pod] for pod in memory_data if re.search(r"large", pod))
                for memory_data in self.memory_usage_list
                if any(re.search(r"large", pod) for pod in memory_data)
            ]

            head_memory_values = [
                max(memory_data[pod] for pod in memory_data if re.search(r"head", pod))
                for memory_data in self.memory_usage_list
                if any(re.search(r"head", pod) for pod in memory_data)
            ]

            if large_memory_values:
                print(
                    f"//Log Max trainer memory value: {max(large_memory_values)} //end"
                )
            else:
                print("No trainer memory values found.")
            if head_memory_values:
                print(
                    f"// Log Max server memory value: {max(head_memory_values)} //end"
                )
            else:
                print("No server memory values found.")
            for pod in self.final_network_data:
                if pod in self.initial_network_data:
                    network_diff = (
                        self.final_network_data[pod] - self.initial_network_data[pod]
                    )
                    self.pretrain_node_network_gauge.set(network_diff)
                    print(f"pretrain round {self.current_round}")
                    print(f"//Log {pod} network: {network_diff} //end")
                    break

            print("Pretrain end time recorded and duration set to gauge.")

    def train_time_start(self) -> None:
        self.current_round += 1
        self.train_start_time = datetime.datetime.now()
        self.initial_network_data = self._get_network_data()
        print(self.initial_network_data)
        self.memory_usage_list = []
        print("Train start time recorded and initial network data collected.")

    def train_time_end(self, interval_seconds=30) -> None:
        if self.train_start_time is not None:
            self.train_end_time = datetime.datetime.now()
            train_duration = (
                self.train_end_time - self.train_start_time
            ).total_seconds() * 1000
            self.train_time_cost_gauge.set(train_duration)
            print(f"//Log train_time: {train_duration} //end")
            time.sleep(interval_seconds)
            self.final_network_data = self._get_network_data()
            # print(f"memory_list:{self.memory_usage_list}")
            large_memory_values = [
                max(memory_data[pod] for pod in memory_data if re.search(r"large", pod))
                for memory_data in self.memory_usage_list
                if any(re.search(r"large", pod) for pod in memory_data)
            ]

            head_memory_values = [
                max(memory_data[pod] for pod in memory_data if re.search(r"head", pod))
                for memory_data in self.memory_usage_list
                if any(re.search(r"head", pod) for pod in memory_data)
            ]

            if large_memory_values:
                print(
                    f"//Log Max trainer memory value: {max(large_memory_values)} //end"
                )
            else:
                print("No trainer memory values found.")
            if head_memory_values:
                print(f"//Log Max server memory value: {max(head_memory_values)} //end")
            else:
                print("No server memory values found.")
            for pod in self.final_network_data:
                if pod in self.initial_network_data:
                    network_diff = (
                        self.final_network_data[pod] - self.initial_network_data[pod]
                    )
                    self.train_node_network_gauge.set(network_diff)
                    print(f"train round {self.current_round}")
                    print(f"//Log {pod} network: {network_diff} //end")
                    break
            print(
                "Train end time recorded, duration set to gauge, and network data difference calculated."
            )
