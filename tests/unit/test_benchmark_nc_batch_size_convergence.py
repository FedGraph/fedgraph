from benchmark.benchmark_NC_batch_size_convergence import parse_run_log


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
