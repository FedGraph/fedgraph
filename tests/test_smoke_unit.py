"""Fast unit-style smoke tests for the new FedGCN-v2 code paths.

These exercise the building blocks that the merge could break without
spinning up a Ray cluster.  Total runtime is well under five seconds on
a laptop CPU; the end-to-end pipeline is covered in
``test_smoke_e2e.py``.
"""
from __future__ import annotations

import pytest
import torch

from fedgraph.utils_nc import get_1hop_feature_sum


# ---------------------------------------------------------------------------
# Adjacency normalization (backward-compat default + new opt-in modes)
# ---------------------------------------------------------------------------

def _toy_graph():
    # 4-node line: 0-1-2-3
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    features = torch.eye(4, dtype=torch.float32)  # one-hot identity
    return features, edge_index


def test_default_norm_type_is_none_backward_compat():
    """The function-level default must be 'none' so merging into main does
    not silently change the plaintext FedGCN aggregation."""
    features, edge_index = _toy_graph()
    out = get_1hop_feature_sum(features, edge_index, device="cpu")
    # 'none' uses binary adjacency with self-loops.  Node 0 sums its own
    # feature and node 1's feature.
    expected_row_0 = features[0] + features[1]
    assert torch.allclose(out[0], expected_row_0)


def test_norm_type_sym_matches_gcn_normalization():
    features, edge_index = _toy_graph()
    out = get_1hop_feature_sum(
        features, edge_index, device="cpu", norm_type="sym"
    )
    # Row 0 has degree 2 (self + 1 neighbour); neighbour 1 has degree 3
    # (self + 0 + 2).  Symmetric weight = 1/sqrt(deg_i * deg_j).
    expected = (
        features[0] / 2.0
        + features[1] / (2 ** 0.5 * 3 ** 0.5)
    )
    assert torch.allclose(out[0], expected, atol=1e-6)


def test_norm_type_row_is_stochastic():
    features, edge_index = _toy_graph()
    out = get_1hop_feature_sum(
        features, edge_index, device="cpu", norm_type="row"
    )
    # Each output row is a convex combination of its self-loop neighbourhood.
    row_sums = out.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_unknown_norm_type_raises():
    features, edge_index = _toy_graph()
    with pytest.raises(ValueError):
        get_1hop_feature_sum(
            features, edge_index, device="cpu", norm_type="bogus"
        )


# ---------------------------------------------------------------------------
# Low-rank compression (round-trip)
# ---------------------------------------------------------------------------

def test_svd_round_trip_is_close_for_low_rank_matrix():
    from fedgraph.low_rank.compression_utils import svd_compress, svd_decompress

    # Construct a near-rank-2 matrix.
    torch.manual_seed(0)
    U = torch.randn(20, 2)
    V = torch.randn(8, 2)
    Z = U @ V.T  # exact rank 2
    Z += 1e-6 * torch.randn_like(Z)  # small noise

    Uc, Sc, Vc = svd_compress(Z, rank=2)
    Z_hat = svd_decompress(Uc, Sc, Vc)
    assert torch.allclose(Z, Z_hat, atol=1e-4)


def test_svd_handles_rank_larger_than_min_shape():
    from fedgraph.low_rank.compression_utils import svd_compress

    Z = torch.randn(5, 3)
    U, S, V = svd_compress(Z, rank=100)
    # Truncation must clip to min(shape).
    assert U.shape[1] <= min(Z.shape)
    assert V.shape[1] == U.shape[1]


# ---------------------------------------------------------------------------
# Optional OpenFHE wrapper smoke test
# ---------------------------------------------------------------------------

def test_openfhe_wrapper_imports_or_skips():
    """The wrapper module must not raise at import time even when openfhe is
    missing on the host."""
    try:
        from fedgraph.openfhe_threshold import OpenFHEThresholdCKKS
    except ImportError:
        pytest.skip("openfhe wheel not installed")
    assert OpenFHEThresholdCKKS is not None
