import datetime
from typing import Dict, Optional, Any

import requests  # type: ignore
from ray.util.metrics import Gauge
import torch
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
        self.item_size_gauge = Gauge(
            "item_size",
            description="matrix or variable size in bytes.",
        )
        self.pretrain_start_time: Optional[datetime.datetime] = None
        self.pretrain_end_time: Optional[datetime.datetime] = None
        self.train_start_time: Optional[datetime.datetime] = None
        self.train_end_time: Optional[datetime.datetime] = None
        self.current_round = 0
        self.initial_network_data_sent: Dict[str, float] = {}
        self.final_network_data_sent: Dict[str, float] = {}
        self.initial_network_data_received: Dict[str, float] = {}
        self.final_network_data_received: Dict[str, float] = {}
        self.item_size: float = 0

    def _get_network_data_sent(self) -> Dict[str, float]:
        response = requests.get(
            "http://prometheus-kube-prometheus-prometheus.prometheus-system:9090/api/v1/query?query=ray_node_network_sent"
        )
        data = response.json()
        return {
            item["metric"]["pod"]: float(item["value"][1])
            for item in data["data"]["result"]
        }
    def _get_network_data_received(self) -> Dict[str, float]:
        response = requests.get(
            "http://prometheus-kube-prometheus-prometheus.prometheus-system:9090/api/v1/query?query=ray_node_network_received"
        )
        data = response.json()
        return {
            item["metric"]["pod"]: float(item["value"][1])
            for item in data["data"]["result"]
        }

    def pretrain_time_start(self) -> None:
        self.pretrain_start_time = datetime.datetime.now()
        self.initial_network_data_sent = self._get_network_data_sent()
        self.initial_network_data_received = self._get_network_data_received()
        print("Pretrain start time recorded and initial network data collected.")

    def pretrain_time_end(self) -> None:
        if self.pretrain_start_time is not None:
            self.pretrain_end_time = datetime.datetime.now()
            pretrain_duration = (
                self.pretrain_end_time - self.pretrain_start_time
            ).total_seconds() * 1000
            self.pretrain_time_cost_gauge.set(pretrain_duration)
            self.final_network_data_sent = self._get_network_data_sent()
            self.final_network_data_received = self._get_network_data_received()
            for pod in self.final_network_data_sent:
                if pod in self.initial_network_data_sent:
                    network_diff = (
                        self.final_network_data_sent[pod] - self.initial_network_data_sent[pod] + self.final_network_data_received[pod] - self.initial_network_data_received[pod]
                    )
                    self.pretrain_node_network_gauge.set(network_diff)
                    print(f"pretrain round {self.current_round}")
                    print(pod, network_diff)
                    break

            print("Pretrain end time recorded and duration set to gauge.")

    def train_time_start(self) -> None:
        self.current_round += 1
        self.train_start_time = datetime.datetime.now()
        self.initial_network_data_sent = self._get_network_data_sent()
        self.initial_network_data_received = self._get_network_data_received()
        print("Train start time recorded and initial network data collected.")

    def train_time_end(self) -> None:
        if self.train_start_time is not None:
            self.train_end_time = datetime.datetime.now()
            train_duration = (
                self.train_end_time - self.train_start_time
            ).total_seconds() * 1000
            self.train_time_cost_gauge.set(train_duration)

            self.final_network_data_sent = self._get_network_data_sent()
            self.final_network_data_received = self._get_network_data_received()
            for pod in self.final_network_data_sent:
                if pod in self.initial_network_data_sent:
                    network_diff = (
                        self.final_network_data_sent[pod] - self.initial_network_data_sent[pod] + self.final_network_data_received[pod] - self.initial_network_data_received[pod]
                    )
                    self.train_node_network_gauge.set(network_diff)
                    print(f"train round {self.current_round}")
                    print(pod, network_diff)
                    break
            print(
                "Train end time recorded, duration set to gauge, and network data difference calculated."
            )
    def print_item_size(self, item: torch.Tensor ) -> None:
        item_size = item.element_size() * item.nelement()
        print(f"Item size: {item_size} bytes.")
        self.item_size += item_size
        self.item_size_gauge.set(self.item_size)
        print(f"Item size set to gauge，total {self.item_size}.")

    def init_item_size(self) -> None:
        self.item_size = 0
        self.item_size_gauge.set(self.item_size)
        print(f"Item size set to 0.")
