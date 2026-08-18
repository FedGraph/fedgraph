import json

import torch

from fedgraph.resource_monitor import (
    append_resource_snapshot,
    collect_resource_snapshot,
    write_resource_snapshot_summary,
)


def test_collect_resource_snapshot_accounts_for_known_tensors_and_model():
    features = torch.ones((2, 3), dtype=torch.float32)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    snapshot = collect_resource_snapshot(
        source="trainer",
        event="unit_test",
        round_id=1,
        trainer_id=7,
        device=torch.device("cpu"),
        tensors={"features": features},
        model=model,
        optimizer=optimizer,
    )

    assert snapshot["source"] == "trainer"
    assert snapshot["event"] == "unit_test"
    assert snapshot["round"] == 1
    assert snapshot["trainer_id"] == 7
    assert snapshot["features_bytes"] == features.numel() * features.element_size()
    assert snapshot["model_parameter_bytes"] == sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    assert snapshot["optimizer_state_bytes"] == 0
    assert snapshot["process_rss_bytes"] is not None
    assert snapshot["cuda_memory_allocated_bytes"] is None


def test_resource_snapshot_summary_keeps_peaks_and_available_memory_minimum(tmp_path):
    snapshot_path = tmp_path / "resource_snapshots.jsonl"
    append_resource_snapshot(
        snapshot_path,
        {
            "source": "trainer",
            "trainer_id": 0,
            "event": "round_train_end",
            "process_rss_bytes": 100,
            "cuda_max_memory_allocated_bytes": 50,
            "host_mem_available_bytes": 900,
        },
    )
    append_resource_snapshot(
        snapshot_path,
        {
            "source": "trainer",
            "trainer_id": 0,
            "event": "round_eval_end",
            "process_rss_bytes": 120,
            "cuda_max_memory_allocated_bytes": 80,
            "host_mem_available_bytes": 700,
        },
    )

    summary_path = write_resource_snapshot_summary(snapshot_path)

    assert summary_path is not None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["snapshot_count"] == 2
    assert summary["event_counts"] == {
        "round_eval_end": 1,
        "round_train_end": 1,
    }
    assert summary["sources"] == [
        {
            "cuda_max_memory_allocated_bytes": 80,
            "host_mem_available_bytes": 700,
            "process_rss_bytes": 120,
            "snapshot_count": 2,
            "source": "trainer",
            "trainer_id": 0,
        }
    ]
