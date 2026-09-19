import datetime
import json
from unittest.mock import Mock, patch

import pytest
import requests

from fedgraph.monitor_class import Monitor


def _prometheus_response(result):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "status": "success",
        "data": {"result": result},
    }
    return response


@patch("fedgraph.monitor_class.Gauge")
def test_monitor_uses_configured_prometheus_url(mock_gauge, monkeypatch):
    monkeypatch.setenv("RAY_PROMETHEUS_HOST", "http://prometheus.example:9090/")

    monitor = Monitor(prometheus_enabled=True)

    assert monitor.prometheus_url == "http://prometheus.example:9090"


@patch("fedgraph.monitor_class.Gauge")
@patch("fedgraph.monitor_class.requests.get")
def test_prometheus_queries_preserve_and_aggregate_pod_identity(mock_get, mock_gauge):
    mock_get.return_value = _prometheus_response(
        [
            {"metric": {"pod": "worker-a"}, "value": [1, "10"]},
            {"metric": {"pod": "worker-a"}, "value": [1, "2"]},
            {"metric": {"pod": "worker-b"}, "value": [1, "7"]},
        ]
    )
    monitor = Monitor(prometheus_enabled=True)

    assert monitor._get_network_data() == {"worker-a": 12.0, "worker-b": 7.0}
    mock_get.assert_called_once_with(
        f"{monitor.prometheus_url}/api/v1/query",
        params={"query": "ray_node_network_sent"},
        timeout=5.0,
    )


@patch("fedgraph.monitor_class.Gauge")
@patch(
    "fedgraph.monitor_class.requests.get",
    side_effect=requests.ConnectionError("no route"),
)
def test_required_prometheus_failure_is_fatal(mock_get, mock_gauge):
    monitor = Monitor(prometheus_enabled=True, prometheus_required=True)

    with pytest.raises(RuntimeError, match="Prometheus query failed"):
        monitor.validate_prometheus()


@patch("fedgraph.monitor_class.Gauge")
@patch(
    "fedgraph.monitor_class.requests.get",
    side_effect=requests.ConnectionError("no route"),
)
def test_optional_prometheus_failure_falls_back_and_writes_status(
    mock_get, mock_gauge, tmp_path
):
    monitor = Monitor(prometheus_enabled=True, prometheus_required=False)

    with pytest.warns(RuntimeWarning, match="continuing without Prometheus"):
        assert not monitor.validate_prometheus()

    summary_path = tmp_path / "prometheus_summary.json"
    assert (
        monitor.write_prometheus_summary(summary_path, resource_monitor_mode="hybrid")
        == summary_path
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["resource_monitor_mode"] == "hybrid"
    assert summary["prometheus_available"] is False
    assert "no route" in summary["prometheus_error"]


@patch("fedgraph.monitor_class.Gauge")
@patch("fedgraph.monitor_class.requests.get")
def test_memory_range_query_keeps_each_pod_peak(mock_get, mock_gauge):
    mock_get.return_value = _prometheus_response(
        [
            {
                "metric": {"pod": "worker-a"},
                "values": [[1, "100"], [2, "250"]],
            },
            {
                "metric": {"pod": "worker-b"},
                "values": [[1, "300"], [2, "200"]],
            },
        ]
    )
    monitor = Monitor(prometheus_enabled=True)

    start = datetime.datetime.fromtimestamp(1, tz=datetime.timezone.utc)
    end = datetime.datetime.fromtimestamp(2, tz=datetime.timezone.utc)

    assert monitor._fetch_memory_peaks(start, end) == {
        "worker-a": 250.0,
        "worker-b": 300.0,
    }
