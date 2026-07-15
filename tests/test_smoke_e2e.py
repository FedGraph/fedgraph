"""End-to-end smoke tests for the FedGCN pipeline.

These spin up a real (single-machine) Ray cluster and run a few global
rounds of FedGCN on Cora.  They are intentionally short -- 10 rounds, 2
trainers, no checkpointing -- so the whole file finishes in well under a
minute on a laptop CPU.

The tests cover the configurations that the gcn_v2 merge could regress:

* plaintext FedGCN with the original (``norm_type='none'``) aggregation;
* plaintext FedGCN with the new GCN-standard (``norm_type='sym'``)
  aggregation introduced for FedGCN-v2;
* TenSEAL-backed encrypted FedGCN;
* OpenFHE threshold-CKKS FedGCN (skipped when the OpenFHE wheel is
  missing);
* OpenFHE + low-rank SVD compression.

Each test only asserts that the pipeline runs and produces a reasonable
test accuracy.  We do not regression-test exact numbers because the
gcn_v2 work intentionally changes the default normalization.
"""

from __future__ import annotations

import contextlib
import importlib
import os

import attridict
import pytest
import ray

from tests.conftest import needs_openfhe, needs_tenseal

# ---------------------------------------------------------------------------
# Shared minimal config
# ---------------------------------------------------------------------------


def _base_cora_config(**overrides):
    cfg = {
        "fedgraph_task": "NC",
        "dataset": "cora",
        "method": "FedGCN",
        "iid_beta": 10000,
        "distribution_type": "average",
        # Keep the run tiny so the suite stays fast.
        "global_rounds": 10,
        "local_step": 3,
        "learning_rate": 0.5,
        "n_trainer": 2,
        "batch_size": -1,
        "num_layers": 2,
        "num_hops": 2,
        "gpu": False,
        "num_cpus_per_trainer": 1,
        "num_gpus_per_trainer": 0,
        "logdir": "./runs",
        "use_huggingface": False,
        "saveto_huggingface": False,
        "use_cluster": False,
        "use_encryption": False,
        "use_lowrank": False,
        # Default to original FedGCN aggregation for the e2e tests.
        "norm_type": "none",
        "seed": 0,
        # Force Ray to use a small in-process plasma so the test does not
        # try to grab 20 GB of /dev/shm.
        "ray_init_kwargs": {
            "num_cpus": 2,
            "object_store_memory": 128 * 1024**2,  # 128 MiB
            "include_dashboard": False,
            "configure_logging": False,
        },
    }
    cfg.update(overrides)
    return attridict(cfg)


@contextlib.contextmanager
def _ray_shutdown_after():
    """Ensure a fresh Ray cluster per test so they don't share state."""
    try:
        yield
    finally:
        if ray.is_initialized():
            ray.shutdown()


def _run(cfg) -> float:
    """Run the pipeline and return the average test accuracy parsed from
    captured stdout.  We tolerate a wide range -- the goal is to verify the
    pipeline doesn't crash and produces a non-trivial number, not to
    benchmark."""
    import io
    from contextlib import redirect_stdout

    from fedgraph.federated_methods import run_fedgraph

    buf = io.StringIO()
    with redirect_stdout(buf):
        run_fedgraph(cfg)
    out = buf.getvalue()
    last_acc = None
    for line in out.splitlines():
        if "Average test accuracy" in line:
            last_acc = float(line.split(",")[-1].strip())
    assert last_acc is not None, (
        "Pipeline finished but no 'Average test accuracy' line was emitted; "
        "captured output:\n" + out[-2000:]
    )
    return last_acc


# ---------------------------------------------------------------------------
# Plaintext path -- original aggregation
# ---------------------------------------------------------------------------


@pytest.mark.timeout(120)
def test_smoke_plaintext_default_norm_none():
    with _ray_shutdown_after():
        acc = _run(_base_cora_config())
    # Plaintext FedGCN on Cora at 10 rounds typically reaches 0.6--0.85.
    assert 0.3 <= acc <= 0.95, f"unexpected accuracy {acc:.3f}"


# ---------------------------------------------------------------------------
# Plaintext path -- new (FedGCN-v2) symmetric normalization opt-in
# ---------------------------------------------------------------------------


@pytest.mark.timeout(120)
def test_smoke_plaintext_norm_sym_opt_in():
    with _ray_shutdown_after():
        acc = _run(_base_cora_config(norm_type="sym"))
    assert 0.3 <= acc <= 0.95, f"unexpected accuracy {acc:.3f}"


# ---------------------------------------------------------------------------
# TenSEAL backend
# ---------------------------------------------------------------------------


@needs_tenseal
@pytest.mark.timeout(180)
def test_smoke_tenseal_encrypted():
    cfg = _base_cora_config(use_encryption=True, he_backend="tenseal")
    with _ray_shutdown_after():
        acc = _run(cfg)
    assert 0.3 <= acc <= 0.95, f"unexpected accuracy {acc:.3f}"


# ---------------------------------------------------------------------------
# OpenFHE threshold backend
# ---------------------------------------------------------------------------


@needs_openfhe
@pytest.mark.timeout(300)
def test_smoke_openfhe_threshold_encrypted():
    cfg = _base_cora_config(
        use_encryption=True,
        he_backend="openfhe",
    )
    with _ray_shutdown_after():
        acc = _run(cfg)
    assert 0.3 <= acc <= 0.95, f"unexpected accuracy {acc:.3f}"


# ---------------------------------------------------------------------------
# OpenFHE threshold + low-rank
# ---------------------------------------------------------------------------


@needs_openfhe
@pytest.mark.timeout(300)
def test_smoke_openfhe_threshold_lowrank():
    cfg = _base_cora_config(
        use_encryption=True,
        he_backend="openfhe",
        use_lowrank=True,
        fixed_rank=50,
    )
    with _ray_shutdown_after():
        acc = _run(cfg)
    assert 0.3 <= acc <= 0.95, f"unexpected accuracy {acc:.3f}"
