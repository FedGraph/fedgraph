import json

import pytest
import torch

from fedgraph.feature_compression import (
    FeatureSecondMoment,
    aggregate_feature_second_moments,
    compute_feature_second_moment,
    feature_basis_sha256,
    load_shared_feature_basis,
    project_features_in_memory,
    project_features_to_file,
    save_shared_feature_basis,
    solve_shared_feature_basis,
)


def test_chunked_second_moment_matches_direct_calculation():
    generator = torch.Generator().manual_seed(42)
    features = torch.randn((17, 5), generator=generator, dtype=torch.float32)

    statistic = compute_feature_second_moment(features, chunk_rows=4)

    expected = features.double().T @ features.double()
    assert statistic.row_count == 17
    assert statistic.feature_dim == 5
    assert statistic.matrix.dtype == torch.float64
    torch.testing.assert_close(statistic.matrix, expected)


def test_chunked_second_moment_accepts_file_backed_features(tmp_path):
    path = tmp_path / "features.f32"
    source = torch.arange(30, dtype=torch.float32).view(10, 3)
    storage = torch.from_file(
        str(path),
        shared=True,
        size=source.numel(),
        dtype=torch.float32,
    ).view_as(source)
    storage.copy_(source)
    del storage
    features = torch.from_file(
        str(path),
        shared=False,
        size=source.numel(),
        dtype=torch.float32,
    ).view_as(source)

    statistic = compute_feature_second_moment(features, chunk_rows=4)

    torch.testing.assert_close(
        statistic.matrix,
        source.double().T @ source.double(),
    )


def test_shared_feature_basis_round_trips_with_verified_identity(tmp_path):
    features = torch.tensor([[4.0, 0.0, 1.0], [3.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    basis = solve_shared_feature_basis(
        compute_feature_second_moment(features, chunk_rows=2),
        rank=2,
    )

    saved = save_shared_feature_basis(basis, output_dir=tmp_path)
    loaded = load_shared_feature_basis(saved.data_path)

    assert loaded.basis_sha256 == feature_basis_sha256(basis)
    torch.testing.assert_close(loaded.basis.matrix, basis.matrix)
    torch.testing.assert_close(loaded.basis.eigenvalues, basis.eigenvalues)
    metadata = json.loads(saved.metadata_path.read_text(encoding="utf-8"))
    assert metadata["projection"] == "uncentered_pca"
    assert metadata["compressed_dim"] == 2


def test_partitioned_second_moments_equal_global_second_moment():
    generator = torch.Generator().manual_seed(7)
    features = torch.randn((23, 6), generator=generator)
    partitions = (features[:5], features[5:14], features[14:])

    aggregated = aggregate_feature_second_moments(
        compute_feature_second_moment(partition, chunk_rows=3)
        for partition in partitions
    )

    assert aggregated.row_count == features.size(0)
    torch.testing.assert_close(
        aggregated.matrix,
        features.double().T @ features.double(),
    )


def test_shared_basis_matches_direct_svd_subspace_and_energy():
    generator = torch.Generator().manual_seed(19)
    left, _ = torch.linalg.qr(torch.randn((12, 4), generator=generator))
    right, _ = torch.linalg.qr(torch.randn((4, 4), generator=generator))
    singular_values = torch.tensor([9.0, 5.0, 2.0, 1.0])
    features = left @ torch.diag(singular_values) @ right.T
    statistic = compute_feature_second_moment(features, chunk_rows=5)

    result = solve_shared_feature_basis(statistic, rank=2)
    _, expected_singular_values, expected_vh = torch.linalg.svd(
        features.double(), full_matrices=False
    )
    expected_basis = expected_vh.T[:, :2]

    torch.testing.assert_close(
        result.matrix.double() @ result.matrix.double().T,
        expected_basis @ expected_basis.T,
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        result.eigenvalues,
        expected_singular_values.square(),
        atol=1e-10,
        rtol=1e-10,
    )
    expected_energy = float(
        singular_values[:2].square().sum() / singular_values.square().sum()
    )
    assert result.retained_energy == pytest.approx(expected_energy)
    assert result.matrix.dtype == torch.float32


def test_rank_full_basis_reconstructs_features():
    generator = torch.Generator().manual_seed(31)
    features = torch.randn((9, 4), generator=generator)
    statistic = compute_feature_second_moment(features, chunk_rows=2)
    result = solve_shared_feature_basis(statistic, rank=features.size(1))

    compressed = features @ result.matrix
    reconstructed = compressed @ result.matrix.T

    torch.testing.assert_close(reconstructed, features, atol=1e-5, rtol=1e-5)
    assert result.retained_energy == pytest.approx(1.0)


def test_chunked_in_memory_projection_matches_direct_projection():
    generator = torch.Generator().manual_seed(54)
    features = torch.randn((19, 6), generator=generator)
    statistic = compute_feature_second_moment(features, chunk_rows=5)
    basis = solve_shared_feature_basis(statistic, rank=3)

    projected = project_features_in_memory(features, basis, chunk_rows=4)

    torch.testing.assert_close(projected, features @ basis.matrix)
    assert projected.shape == (19, 3)
    assert projected.dtype == torch.float32


def test_file_backed_projection_is_atomic_and_records_basis(tmp_path):
    generator = torch.Generator().manual_seed(87)
    features = torch.randn((11, 5), generator=generator)
    statistic = compute_feature_second_moment(features, chunk_rows=4)
    basis = solve_shared_feature_basis(statistic, rank=2)

    artifact = project_features_to_file(
        features,
        basis,
        chunk_rows=3,
        output_dir=tmp_path,
        trainer_id=7,
    )

    torch.testing.assert_close(artifact.tensor, features @ basis.matrix)
    assert artifact.data_path.stat().st_size == 11 * 2 * torch.float32.itemsize
    assert artifact.basis_sha256 == feature_basis_sha256(basis)
    metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    assert metadata == {
        "basis_sha256": artifact.basis_sha256,
        "compressed_dim": 2,
        "dtype": "float32",
        "layout": "row_major",
        "original_dim": 5,
        "projection": "uncentered_pca",
        "retained_energy": basis.retained_energy,
        "row_count": 11,
        "size_bytes": 88,
        "store_version": 1,
    }
    assert not list(artifact.data_path.parent.glob("*.tmp-*"))


def test_basis_signs_are_canonicalized():
    features = torch.diag(torch.tensor([4.0, 3.0, 2.0]))
    statistic = compute_feature_second_moment(features, chunk_rows=2)

    result = solve_shared_feature_basis(statistic, rank=3)

    pivot_rows = result.matrix.abs().argmax(dim=0)
    columns = torch.arange(result.rank)
    assert torch.all(result.matrix[pivot_rows, columns] >= 0)


def test_zero_energy_features_produce_a_valid_basis():
    statistic = compute_feature_second_moment(torch.zeros((8, 3)), chunk_rows=3)

    result = solve_shared_feature_basis(statistic, rank=2)

    assert result.total_energy == 0.0
    assert result.retained_energy == 1.0
    torch.testing.assert_close(
        result.matrix.T @ result.matrix,
        torch.eye(2),
    )


def test_feature_compression_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="two-dimensional"):
        compute_feature_second_moment(torch.ones(3), chunk_rows=1)
    with pytest.raises(ValueError, match="floating-point"):
        compute_feature_second_moment(
            torch.ones((3, 2), dtype=torch.long), chunk_rows=1
        )
    with pytest.raises(ValueError, match="chunk_rows"):
        compute_feature_second_moment(torch.ones((3, 2)), chunk_rows=0)
    with pytest.raises(ValueError, match="at least one"):
        aggregate_feature_second_moments([])

    statistic = FeatureSecondMoment(
        matrix=torch.eye(2, dtype=torch.float64),
        row_count=3,
        feature_dim=2,
    )
    with pytest.raises(ValueError, match="rank"):
        solve_shared_feature_basis(statistic, rank=3)
