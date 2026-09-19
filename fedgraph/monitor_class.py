import datetime
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests  # type: ignore
from ray.util.metrics import Gauge


class Monitor:
    """Record FedGraph phase metrics and optionally query Prometheus."""

    DEFAULT_PROMETHEUS_URL = (
        "http://prometheus-kube-prometheus-prometheus."
        "prometheus-system.svc.cluster.local:9090"
    )

    def __init__(
        self,
        use_cluster: bool = False,
        *,
        prometheus_enabled: Optional[bool] = None,
        prometheus_required: bool = False,
        prometheus_url: Optional[str] = None,
        prometheus_timeout_seconds: float = 5.0,
        prometheus_query_step_seconds: int = 5,
    ) -> None:
        self.use_cluster = use_cluster
        # Preserve legacy behavior for callers that only pass use_cluster. NC passes
        # prometheus_enabled explicitly so its resource-monitor mode is authoritative.
        self.prometheus_requested = (
            use_cluster if prometheus_enabled is None else prometheus_enabled
        )
        self.prometheus_enabled = self.prometheus_requested
        self.prometheus_required = prometheus_required
        self.prometheus_url = (
            prometheus_url
            or os.environ.get("FEDGRAPH_PROMETHEUS_URL")
            or os.environ.get("RAY_PROMETHEUS_HOST")
            or self.DEFAULT_PROMETHEUS_URL
        ).rstrip("/")
        self.prometheus_timeout_seconds = prometheus_timeout_seconds
        self.prometheus_query_step_seconds = prometheus_query_step_seconds
        self.prometheus_error: Optional[str] = None
        self.network_query = os.environ.get(
            "FEDGRAPH_PROMETHEUS_NETWORK_QUERY", "ray_node_network_sent"
        )
        self.memory_query = os.environ.get(
            "FEDGRAPH_PROMETHEUS_MEMORY_QUERY", "ray_node_mem_used"
        )

        self.pretrain_time_cost_gauge = Gauge(
            "pretrain_time_cost", description="Pretraining duration in ms."
        )
        self.train_time_cost_gauge = Gauge(
            "train_time_cost", description="Training duration in ms."
        )
        self.pretrain_node_network_gauge = Gauge(
            "pretrain_node_network",
            description="Total network bytes sent during pretraining.",
        )
        self.train_node_network_gauge = Gauge(
            "train_node_network",
            description="Total network bytes sent during training.",
        )
        self.pretrain_memory_gauge = Gauge(
            "pretrain_memory_usage",
            description="Maximum observed per-pod memory in bytes during pretraining.",
        )
        self.train_memory_gauge = Gauge(
            "train_memory_usage",
            description="Maximum observed per-pod memory in bytes during training.",
        )
        self.init_time_cost_gauge = Gauge(
            "init_time_cost", description="Initialization duration in ms."
        )
        self.pretrain_theoretical_comm_gauge = Gauge(
            "pretrain_theoretical_comm_MB",
            description="Theoretical communication cost in MB during pretraining.",
        )
        self.train_theoretical_comm_gauge = Gauge(
            "train_theoretical_comm_MB",
            description="Theoretical communication cost in MB during training.",
        )

        self.init_start_time: Optional[datetime.datetime] = None
        self.init_end_time: Optional[datetime.datetime] = None
        self.pretrain_start_time: Optional[datetime.datetime] = None
        self.pretrain_end_time: Optional[datetime.datetime] = None
        self.train_start_time: Optional[datetime.datetime] = None
        self.train_end_time: Optional[datetime.datetime] = None
        self.total_comm_start_time: Optional[datetime.datetime] = None
        self.total_comm_end_time: Optional[datetime.datetime] = None

        self.current_round = 0
        self.initial_network_data: Dict[str, float] = {}
        self.final_network_data: Dict[str, float] = {}
        self.memory_usage_list: List[Any] = []
        self.phase_summaries: Dict[str, Dict[str, Any]] = {}
        self._phase_network_start: Dict[str, Dict[str, float]] = {}
        self.pretrain_theoretical_comm_MB = 0.0
        self.train_theoretical_comm_MB = 0.0

    def add_pretrain_comm_cost(self, upload_mb: float, download_mb: float) -> None:
        self.pretrain_theoretical_comm_MB += upload_mb + download_mb
        self.pretrain_theoretical_comm_gauge.set(self.pretrain_theoretical_comm_MB)

    def add_train_comm_cost(self, upload_mb: float, download_mb: float) -> None:
        self.train_theoretical_comm_MB += upload_mb + download_mb
        self.train_theoretical_comm_gauge.set(self.train_theoretical_comm_MB)

    def _handle_prometheus_error(self, error: Exception) -> None:
        message = f"Prometheus query failed for {self.prometheus_url}: {error}"
        self.prometheus_error = message
        if self.prometheus_required:
            raise RuntimeError(message) from error
        self.prometheus_enabled = False
        warnings.warn(
            f"{message}; continuing without Prometheus telemetry",
            RuntimeWarning,
            stacklevel=2,
        )

    def _prometheus_request(
        self, endpoint: str, params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        if not self.prometheus_enabled:
            return []
        try:
            response = requests.get(
                f"{self.prometheus_url}{endpoint}",
                params=params,
                timeout=self.prometheus_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") != "success":
                raise ValueError(payload.get("error") or "non-success response")
            result = payload.get("data", {}).get("result")
            if not isinstance(result, list):
                raise ValueError("response data.result is not a list")
            return result
        except (requests.RequestException, TypeError, ValueError) as error:
            self._handle_prometheus_error(error)
            return []

    def validate_prometheus(self) -> bool:
        """Verify that the Prometheus query API is reachable before training."""
        if not self.prometheus_requested:
            return False
        self._prometheus_request("/api/v1/query", {"query": "up"})
        return self.prometheus_enabled

    @staticmethod
    def _target_name(labels: Dict[str, Any]) -> str:
        for key in ("pod", "instance", "node", "NodeAddress", "ray_io_cluster"):
            value = labels.get(key)
            if value:
                return str(value)
        return str(labels.get("job") or labels.get("__name__") or "unknown")

    def _query_vector(self, query: str, *, reducer: str) -> Dict[str, float]:
        result = self._prometheus_request("/api/v1/query", {"query": query})
        values: Dict[str, float] = {}
        for item in result:
            labels = item.get("metric", {})
            value = item.get("value", [None, None])
            try:
                sample = float(value[1])
            except (IndexError, TypeError, ValueError):
                continue
            target = self._target_name(labels)
            if reducer == "sum":
                values[target] = values.get(target, 0.0) + sample
            else:
                values[target] = max(values.get(target, sample), sample)
        return values

    def _get_network_data(self) -> Dict[str, float]:
        return self._query_vector(self.network_query, reducer="sum")

    def _fetch_memory_usage(self) -> Dict[str, float]:
        return self._query_vector(self.memory_query, reducer="max")

    def _fetch_memory_peaks(
        self, start: datetime.datetime, end: datetime.datetime
    ) -> Dict[str, float]:
        result = self._prometheus_request(
            "/api/v1/query_range",
            {
                "query": self.memory_query,
                "start": start.timestamp(),
                "end": max(end.timestamp(), start.timestamp() + 0.001),
                "step": self.prometheus_query_step_seconds,
            },
        )
        peaks: Dict[str, float] = {}
        for item in result:
            target = self._target_name(item.get("metric", {}))
            for value in item.get("values", []):
                try:
                    sample = float(value[1])
                except (IndexError, TypeError, ValueError):
                    continue
                peaks[target] = max(peaks.get(target, sample), sample)
        if not peaks and self.prometheus_enabled:
            return self._fetch_memory_usage()
        return peaks

    @staticmethod
    def _network_deltas(
        initial: Dict[str, float], final: Dict[str, float]
    ) -> Dict[str, float]:
        deltas: Dict[str, float] = {}
        for target, final_value in final.items():
            initial_value = initial.get(target, 0.0)
            # If a Ray process restarted, its counter starts again from zero.
            deltas[target] = (
                final_value - initial_value
                if final_value >= initial_value
                else final_value
            )
        return deltas

    def _start_prometheus_phase(self, phase: str) -> None:
        if not self.prometheus_enabled:
            return
        self._phase_network_start[phase] = self._get_network_data()

    def _finish_prometheus_phase(
        self,
        phase: str,
        start: datetime.datetime,
        end: datetime.datetime,
    ) -> Dict[str, Any]:
        if not self.prometheus_enabled:
            return {}
        final_network = self._get_network_data()
        network_deltas = self._network_deltas(
            self._phase_network_start.get(phase, {}), final_network
        )
        memory_peaks = self._fetch_memory_peaks(start, end)
        summary = {
            "start_time_utc": start.astimezone(datetime.timezone.utc).isoformat(),
            "end_time_utc": end.astimezone(datetime.timezone.utc).isoformat(),
            "duration_ms": (end - start).total_seconds() * 1000,
            "network_sent_bytes_by_target": network_deltas,
            "network_sent_bytes_total": sum(network_deltas.values()),
            "memory_used_bytes_peak_by_target": memory_peaks,
            "memory_used_bytes_peak_max": max(memory_peaks.values(), default=0.0),
        }
        self.phase_summaries[phase] = summary
        return summary

    @staticmethod
    def _print_prometheus_phase(phase: str, summary: Dict[str, Any]) -> None:
        if not summary:
            return
        for target, value in summary["memory_used_bytes_peak_by_target"].items():
            print(f"//Log Max memory for {target}: {value} //end")
        for target, value in summary["network_sent_bytes_by_target"].items():
            print(f"//Log {target} {phase} network: {value} //end")
        total_mb = summary["network_sent_bytes_total"] / (1024 * 1024)
        print(f"//Log Total Actual {phase.title()} Comm Cost: {total_mb:.2f} MB //end")

    def init_time_start(self) -> None:
        self.init_start_time = datetime.datetime.now(datetime.timezone.utc)
        self._start_prometheus_phase("initialization")
        if self.prometheus_enabled:
            print("Initialization start: Prometheus network baseline collected.")
        else:
            print("Initialization start time recorded.")

    def init_time_end(self) -> None:
        self.init_end_time = datetime.datetime.now(datetime.timezone.utc)
        if self.init_start_time is None:
            return
        elapsed = (self.init_end_time - self.init_start_time).total_seconds() * 1000
        self.init_time_cost_gauge.set(elapsed)
        print(f"//Log init_time: {elapsed} ms //end")
        summary = self._finish_prometheus_phase(
            "initialization", self.init_start_time, self.init_end_time
        )
        self._print_prometheus_phase("initialization", summary)

    def pretrain_time_start(self) -> None:
        self.pretrain_start_time = datetime.datetime.now(datetime.timezone.utc)
        self._start_prometheus_phase("pretrain")
        print("Pretrain start time recorded.")

    def pretrain_time_end(self) -> None:
        self.pretrain_end_time = datetime.datetime.now(datetime.timezone.utc)
        if self.pretrain_start_time is None:
            return
        duration = (
            self.pretrain_end_time - self.pretrain_start_time
        ).total_seconds() * 1000
        self.pretrain_time_cost_gauge.set(duration)
        print(f"//pretrain_time: {duration} ms//end")
        summary = self._finish_prometheus_phase(
            "pretrain", self.pretrain_start_time, self.pretrain_end_time
        )
        if summary:
            self.pretrain_node_network_gauge.set(summary["network_sent_bytes_total"])
            self.pretrain_memory_gauge.set(summary["memory_used_bytes_peak_max"])
        self._print_prometheus_phase("pretrain", summary)

    def train_time_start(self) -> None:
        self.current_round += 1
        self.train_start_time = datetime.datetime.now(datetime.timezone.utc)
        self._start_prometheus_phase("train")
        print("Train start time recorded.")

    def train_time_end(self) -> None:
        self.train_end_time = datetime.datetime.now(datetime.timezone.utc)
        if self.train_start_time is None:
            return
        duration = (self.train_end_time - self.train_start_time).total_seconds() * 1000
        self.train_time_cost_gauge.set(duration)
        print(f"//train_time: {duration} ms//end")
        summary = self._finish_prometheus_phase(
            "train", self.train_start_time, self.train_end_time
        )
        if summary:
            self.train_node_network_gauge.set(summary["network_sent_bytes_total"])
            self.train_memory_gauge.set(summary["memory_used_bytes_peak_max"])
        self._print_prometheus_phase("train", summary)

    def write_prometheus_summary(
        self, path: Path, *, resource_monitor_mode: Optional[str] = None
    ) -> Optional[Path]:
        """Persist compact phase aggregates without copying full time series."""
        if not self.prometheus_requested:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "resource_monitor_mode": resource_monitor_mode,
            "prometheus_url": self.prometheus_url,
            "prometheus_available": self.prometheus_error is None,
            "prometheus_error": self.prometheus_error,
            "network_query": self.network_query,
            "memory_query": self.memory_query,
            "query_step_seconds": self.prometheus_query_step_seconds,
            "phases": self.phase_summaries,
            "theoretical_communication_mb": {
                "pretrain": self.pretrain_theoretical_comm_MB,
                "train": self.train_theoretical_comm_MB,
            },
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def print_comm_cost(self) -> None:
        print(
            f"//Log Theoretical Pretrain Comm Cost: {self.pretrain_theoretical_comm_MB:.2f} MB //end"
        )
        print(
            f"//Log Theoretical Train Comm Cost: {self.train_theoretical_comm_MB:.2f} MB //end"
        )
