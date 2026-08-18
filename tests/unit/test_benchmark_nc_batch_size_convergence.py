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
