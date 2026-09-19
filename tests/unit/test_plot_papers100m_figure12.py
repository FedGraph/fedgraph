import json
from pathlib import Path

import numpy as np
import pytest

from benchmark.figure.NC_comm_costs.plot_papers100m_figure12 import (
    best_fit,
    main,
    prepare_figure_data,
)


def _write_figure_inputs(tmp_path: Path, node_counts=(10, 10, 10)):
    artifact_manifest = tmp_path / "artifact-manifest.json"
    artifact_manifest.write_text(
        json.dumps(
            {
                "n_trainer": 3,
                "partition_policy": "label_dirichlet_balanced_unlabeled",
                "shards": [
                    {
                        "trainer_id": trainer_id,
                        "node_count": node_count,
                        "internal_edge_count": 20 + trainer_id,
                    }
                    for trainer_id, node_count in enumerate(node_counts)
                ],
            }
        ),
        encoding="utf-8",
    )

    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    summaries = []
    for batch_size in (16, 32, 64):
        run_id = f"run-bs{batch_size}"
        run_dir = experiment_dir / "runs" / run_id
        log_dir = run_dir / "fedgraph_logs"
        log_dir.mkdir(parents=True)
        config = {
            "hf_local_artifact_repo": "FedGraph/test-artifact",
            "hf_local_artifact_revision": "abc123",
        }
        (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        resource_summary = {
            "sources": [
                {
                    "source": "trainer",
                    "trainer_id": trainer_id,
                    "snapshot_count": 4,
                    "process_peak_rss_bytes": (700 + trainer_id + batch_size) * 1024**2,
                    "cuda_max_memory_allocated_bytes": (300 + trainer_id) * 1024**2,
                    "cuda_max_memory_reserved_bytes": (350 + trainer_id) * 1024**2,
                }
                for trainer_id in range(3)
            ]
        }
        (log_dir / "resource_snapshot_summary.json").write_text(
            json.dumps(resource_summary), encoding="utf-8"
        )
        summaries.append(
            {
                "batch_size": batch_size,
                "status": "completed",
                "run_id": run_id,
                "dataset": "ogbn-papers100M",
                "method": "FedAvg",
                "n_trainer": 3,
                "global_rounds": 8,
                "rounds_recorded": 8,
                "seed": 42,
                "num_hops": 0,
                "iid_beta": 10000.0,
                "local_step": 3,
                "learning_rate": 0.01,
                "distribution_type": "average",
                "num_layers": 2,
                "gpu": True,
                "evaluation_split": "validation",
                "resource_monitor_mode": "manual",
                "total_training_time_sec": float(batch_size),
                "test_acc_final": 0.4 + batch_size / 10000,
            }
        )
    summary_path = experiment_dir / "summary.jsonl"
    summary_path.write_text(
        "".join(json.dumps(summary) + "\n" for summary in summaries),
        encoding="utf-8",
    )
    (experiment_dir / "manifest.json").write_text("{}", encoding="utf-8")
    return artifact_manifest, summary_path


def test_prepare_figure_data_joins_trainers_with_artifact_shards(tmp_path):
    artifact_manifest, summary_path = _write_figure_inputs(tmp_path)

    batch_rows, memory_rows, runs, _ = prepare_figure_data(
        summary_paths=[summary_path],
        artifact_manifest_path=artifact_manifest,
        batch_sizes=[16, 32, 64],
        memory_metric="process_peak_rss_bytes",
        expected_trainers=3,
        expected_rounds=8,
        expected_seed=42,
        expected_iid_beta=10000.0,
        repo_root=tmp_path,
    )

    assert [row["batch_size"] for row in batch_rows] == [16, 32, 64]
    assert len(memory_rows) == 9
    assert memory_rows[0]["node_count"] == 10
    assert memory_rows[0]["internal_edge_count"] == 20
    assert memory_rows[0]["memory_mib"] == 716
    assert len(runs) == 3


def test_best_fit_omits_constant_balanced_node_counts():
    fit = best_fit(np.array([10.0, 10.0, 10.0]), np.array([1.0, 2.0, 3.0]), "nodes")

    assert fit is None


def test_best_fit_omits_node_counts_that_only_differ_by_rounding():
    fit = best_fit(np.array([10.0, 10.0, 11.0]), np.array([1.0, 2.0, 3.0]), "nodes")

    assert fit is None


def test_main_writes_isolated_outputs_and_provenance(tmp_path):
    artifact_manifest, summary_path = _write_figure_inputs(tmp_path)
    output_dir = tmp_path / "figures"
    args = [
        "--summary-jsonl",
        str(summary_path),
        "--artifact-manifest",
        str(artifact_manifest),
        "--output-dir",
        str(output_dir),
        "--expected-trainers",
        "3",
        "--expected-rounds",
        "8",
        "--repo-root",
        str(tmp_path),
    ]

    assert main(args) == 0
    assert (output_dir / "figure12_balanced.pdf").is_file()
    assert (output_dir / "figure12_balanced.png").is_file()
    provenance = json.loads(
        (output_dir / "figure12_balanced_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["training_time_field"] == "total_training_time_sec"
    assert provenance["memory_metric"] == "process_peak_rss_bytes"
    assert provenance["hf_artifact_revisions"] == ["abc123"]
    assert any("node-memory regression" in item for item in provenance["warnings"])

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        main(args)
