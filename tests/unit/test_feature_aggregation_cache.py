import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fedgraph.feature_aggregation_cache import (
    load_feature_cache_manifest,
    open_feature_cache_rank,
    promote_feature_cache_rank,
)
from fedgraph.federated_methods import _resolve_pretrain_feature_cache
from fedgraph.graph_storage import FeatureAggregationStore


def _make_cache_inputs(tmp_path: Path, width: int):
    artifact = tmp_path / "artifact"
    pretrain = tmp_path / "pretrain" / "finished-run"
    io_args = dict(artifact_dir=artifact, cache_dir=tmp_path / "cache", trainer_id=0)
    shard = artifact / "shards" / "trainer-000"
    shard.mkdir(parents=True)
    (artifact / "manifest.json").write_text(
        json.dumps(
            dict(
                artifact_version=3,
                hop_semantics=2,
                adjacency_layout="local-int32",
                n_trainer=1,
                feature_dim=128,
                shards=[dict(trainer_id=0, communicate_node_count=2)],
            )
        )
    )
    torch.save(torch.tensor([7, 20]), shard / "communicate_node_index.pt")
    values = torch.arange(2 * width, dtype=torch.float32).reshape(2, width)
    provenance = None
    if width != 128:
        provenance = dict(
            feature_compression_mode="shared-pca",
            feature_compression_original_dim=128,
            runtime_feature_dim=width,
            feature_compression_basis_sha256="a" * 64,
        )
    store = FeatureAggregationStore(
        pretrain, trainer_id=0, row_count=2, feature_dim=width, provenance=provenance
    )
    store.write(0, values[:1])
    store.write(1, values[1:])
    store.finalize()
    return io_args, pretrain, values, store.data_path


@pytest.mark.parametrize("width", [16, 64, 128])
def test_promote_papers100m_feature_widths(tmp_path, width):
    io_args, pretrain, values, source = _make_cache_inputs(tmp_path, width)
    manifest = promote_feature_cache_rank(
        **io_args, pretrain_dir=pretrain, norm_type="none"
    )
    assert manifest == load_feature_cache_manifest(io_args["cache_dir"])
    assert manifest.feature_compression_mode == (
        "none" if width == 128 else "shared-pca"
    )
    assert manifest.feature_dim == width
    assert manifest.source_run_id == "finished-run"

    mapped, path = open_feature_cache_rank(
        **io_args, manifest=manifest, verify_payload=True
    )
    assert path.stat().st_ino == source.stat().st_ino
    torch.testing.assert_close(mapped, values, rtol=0, atol=0)
    original = source.read_bytes()
    mapped[0, 0] = -1
    assert source.read_bytes() == original
    source.unlink()
    reopened, _ = open_feature_cache_rank(**io_args, manifest=manifest)
    torch.testing.assert_close(reopened, values, rtol=0, atol=0)


@pytest.mark.parametrize(
    "damage, message",
    [
        ("rows", "row order"),
        ("truncated", "file length"),
        ("checksum", "payload checksum"),
        ("basis", "PCA basis"),
        ("artifact", "source artifact manifest"),
        ("missing", "cache-shard.json"),
    ],
)
def test_reuse_rejects_incompatible_rank(tmp_path, damage, message):
    io_args, pretrain, _, _ = _make_cache_inputs(tmp_path, 64)
    manifest = promote_feature_cache_rank(
        **io_args, pretrain_dir=pretrain, norm_type="none", copy=True
    )
    rank = io_args["cache_dir"] / "trainer-000"
    feature_file = rank / "feature-aggregation.f32"
    if damage == "rows":
        torch.save(
            torch.tensor([7, 21]),
            io_args["artifact_dir"]
            / "shards"
            / "trainer-000"
            / "communicate_node_index.pt",
        )
    elif damage == "truncated":
        feature_file.write_bytes(feature_file.read_bytes()[:-4])
    elif damage == "checksum":
        feature_file.write_bytes(b"\0" * feature_file.stat().st_size)
    elif damage == "basis":
        metadata_path = rank / "feature-aggregation.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["provenance"]["feature_compression_basis_sha256"] = "b" * 64
        metadata_path.write_text(json.dumps(metadata))
    elif damage == "artifact":
        artifact_path = io_args["artifact_dir"] / "manifest.json"
        artifact_path.write_text(artifact_path.read_text() + "\n")
    else:
        (rank / "cache-shard.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        open_feature_cache_rank(**io_args, manifest=manifest, verify_payload=True)


def test_head_validates_cache_settings_before_any_actor_load(tmp_path):
    io_args, pretrain, _, _ = _make_cache_inputs(tmp_path, 64)
    manifest = promote_feature_cache_rank(
        **io_args, pretrain_dir=pretrain, norm_type="none"
    )
    args = SimpleNamespace(
        method="FedGCN",
        num_hops=2,
        local_artifact_dir=io_args["artifact_dir"],
        pretrain_feature_cache_dir=io_args["cache_dir"],
        graph_relabel_cache_dir=tmp_path / "adjacency",
    )
    modes = dict(
        graph_storage_mode="mmap",
        aggregation_mode="chunked",
        compression_mode="shared-pca",
        compression_rank=64,
    )
    assert _resolve_pretrain_feature_cache(args, **modes) == (
        io_args["cache_dir"],
        manifest,
    )
    with pytest.raises(ValueError, match="mode or rank"):
        _resolve_pretrain_feature_cache(args, **dict(modes, compression_rank=16))
    args.norm_type = "row"
    with pytest.raises(ValueError, match="normalization"):
        _resolve_pretrain_feature_cache(args, **modes)
    args.norm_type = "none"
    args.graph_relabel_cache_dir = io_args["cache_dir"] / "adjacency"
    with pytest.raises(ValueError, match="outside"):
        _resolve_pretrain_feature_cache(args, **modes)
