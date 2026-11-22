import io
import json
import logging
import sys

from benchmark.benchmark_NC_batch_size_convergence import (
    Tee,
    _restore_logging_streams,
    build_configs,
    main,
    parse_args,
    parse_run_log,
)


def test_evaluation_batch_size_is_recorded_separately(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_NC_batch_size_convergence.py",
            "--batch-sizes",
            "16",
            "--evaluation-batch-size",
            "512",
        ],
    )

    config = build_configs(parse_args())[0]

    assert config.batch_size == 16
    assert config.evaluation_batch_size == 512


def test_dry_run_accepts_feature_cache_with_default_fedgcn_method(
    tmp_path, monkeypatch
):
    artifact_dir, cache_dir = tmp_path / "artifact", tmp_path / "cache"
    artifact_dir.mkdir()
    cache_dir.mkdir()
    (artifact_dir / "manifest.json").write_text(
        '{"artifact_version": 3, "hop_semantics": 2, "n_trainer": 5}'
    )
    (cache_dir / "feature-cache.json").write_text("{}")
    output_dir = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_NC_batch_size_convergence.py",
            "--num-hops",
            "2",
            "--local-artifact-dir",
            str(artifact_dir),
            "--pretrain-feature-upload-mode",
            "indexed",
            "--pretrain-feature-aggregation-mode",
            "chunked",
            "--pretrain-feature-cache-dir",
            str(cache_dir),
            "--graph-storage-mode",
            "mmap",
            "--graph-relabel-cache-dir",
            str(tmp_path / "relabel"),
            "--feature-compression-mode",
            "shared-pca",
            "--feature-compression-rank",
            "16",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
    )

    assert main() == 0
    args = json.loads((output_dir / "manifest.json").read_text())["args"]
    assert args["pretrain_feature_cache_dir"] == str(cache_dir)
    assert args["pretrain_feature_output_dir"] is None
    assert args["feature_compression_output_dir"] is None


def test_restore_logging_streams_detaches_run_scoped_tee(tmp_path):
    console = io.StringIO()
    logger = logging.getLogger(f"fedgraph-test-{id(tmp_path)}")
    logger.propagate = False
    logger.setLevel(logging.INFO)

    with (tmp_path / "first-run.log").open("w", encoding="utf-8") as run_log:
        tee = Tee(console, run_log)
        handler = logging.StreamHandler(tee)
        logger.addHandler(handler)
        logger.info("first run")
        _restore_logging_streams({tee: console})

    try:
        logger.info("second run")
        assert handler.stream is console
        assert "first run" in console.getvalue()
        assert "second run" in console.getvalue()
    finally:
        logger.removeHandler(handler)
        handler.close()


def test_parse_run_log_keeps_per_round_test_metrics_out_of_validation_fields(
    tmp_path,
):
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "Round 1: Global Test Loss = 1.2500",
                "Round 1: Global Test Accuracy = 0.7500",
                "Round 1: Training Time = 0.10s, Communication Time = 0.20s",
            ]
        ),
        encoding="utf-8",
    )

    metric = parse_run_log(log_path)["round_metrics"][0]

    assert metric["evaluation_split"] == "test"
    assert metric["evaluation_loss"] == 1.25
    assert metric["evaluation_acc"] == 0.75
    assert metric["val_loss"] is None
    assert metric["val_acc"] is None


def test_parse_run_log_preserves_validation_metrics(tmp_path):
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "Round 1: Global Val Loss = 1.2500",
                "Round 1: Global Val Accuracy = 0.7500",
            ]
        ),
        encoding="utf-8",
    )

    metric = parse_run_log(log_path)["round_metrics"][0]

    assert metric["evaluation_split"] == "validation"
    assert metric["evaluation_loss"] == 1.25
    assert metric["evaluation_acc"] == 0.75
    assert metric["val_loss"] == 1.25
    assert metric["val_acc"] == 0.75


def test_parse_run_log_prefers_precise_structured_round_timing(tmp_path):
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "Round 1: Training Time = 0.12s, Communication Time = 0.23s",
                (
                    'NC_ROUND_METRIC {"cumulative_train_sync_time_sec":0.35802468,'
                    '"parameter_sync_time_sec":0.234567891,"round":1,'
                    '"train_sync_time_sec":0.35802468,'
                    '"training_time_sec":0.123456789}'
                ),
            ]
        ),
        encoding="utf-8",
    )

    metric = parse_run_log(log_path)["round_metrics"][0]

    assert metric["training_time_sec"] == 0.123456789
    assert metric["parameter_sync_time_sec"] == 0.234567891
    assert metric["train_sync_time_sec"] == 0.35802468
    assert metric["cumulative_train_sync_time_sec"] == 0.35802468
    assert metric["train_time_sec"] == metric["training_time_sec"]
    assert metric["comm_time_sec"] == metric["parameter_sync_time_sec"]


def test_parse_run_log_uses_corrected_total_timing_definitions(tmp_path):
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "Total Training Time (server-observed): 1.234567 seconds",
                "Total Communication Time (parameter synchronization): 0.345679 seconds",
                "Total Training + Communication Time: 1.580246 seconds",
                "Total Federated Loop Wall Time: 2.500000 seconds",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_run_log(log_path)

    assert parsed["total_training_time_sec"] == 1.234567
    assert parsed["total_comm_time_sec"] == 0.345679
    assert parsed["total_train_comm_time_sec"] == 1.580246
    assert parsed["federated_loop_wall_time_sec"] == 2.5


def test_parse_run_log_repairs_legacy_train_comm_total(tmp_path):
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "Total Pure Training Time (forward + gradient descent): 1.25 seconds",
                "Total Communication Time (parameter aggregation): 0.75 seconds",
                "Total Training + Communication Time: 9.00 seconds",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_run_log(log_path)

    assert parsed["total_training_time_sec"] == 1.25
    assert parsed["total_comm_time_sec"] == 0.75
    assert parsed["total_train_comm_time_sec"] == 2.0


def test_parse_run_log_reads_component3_and_pretrain_metrics(tmp_path):
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "NC_FEATURE_COMPRESSION_METRIC "
                '{"basis_sha256":"basis-1","compressed_feature_dim":16,'
                '"compression_wall_time_sec":3.5,"original_feature_bytes":5120,'
                '"original_feature_dim":128,"projected_feature_bytes":640,'
                '"retained_energy":0.91}',
                "NC_PRETRAIN_METRIC "
                '{"aggregation_mode":"chunked","download_bytes":320,'
                '"artifact_version":3,'
                '"adjacency_transition_wall_time_sec":0.75,'
                '"feature_compression_basis_sha256":"basis-1",'
                '"feature_dim":16,"total_traffic_bytes":640,'
                '"upload_bytes":320,"wall_time_sec":2.25}',
                "NC_NEIGHBOR_SAMPLER_METRIC "
                '{"index_bytes":4096,"max_trainer_time_sec":1.25,'
                '"prepared_trainer_count":2,"trainer_count":2,'
                '"wall_time_sec":1.5}',
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_run_log(log_path)

    assert parsed["feature_compression_metric"] == {
        "basis_sha256": "basis-1",
        "compressed_feature_dim": 16,
        "compression_wall_time_sec": 3.5,
        "original_feature_bytes": 5120,
        "original_feature_dim": 128,
        "projected_feature_bytes": 640,
        "retained_energy": 0.91,
    }
    assert parsed["pretrain_metric"]["aggregation_mode"] == "chunked"
    assert parsed["pretrain_metric"]["artifact_version"] == 3
    assert parsed["pretrain_metric"]["adjacency_transition_wall_time_sec"] == 0.75
    assert parsed["pretrain_metric"]["feature_dim"] == 16
    assert parsed["pretrain_metric"]["total_traffic_bytes"] == 640
    assert parsed["neighbor_sampler_metric"]["index_bytes"] == 4096
    assert parsed["neighbor_sampler_metric"]["wall_time_sec"] == 1.5
