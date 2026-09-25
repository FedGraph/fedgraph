import gzip
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from fedgraph.trainer_class import (
    Trainer_General,
    _load_artifact_tensor,
    load_trainer_data_from_huggingface_local_artifact,
    load_trainer_data_from_local_artifact,
)
from scripts.data.node_classification_partitioning import (
    build_label_balanced_owner_plan,
    discover_ogb_source,
    partition_raw_ogb_0hop,
)
from scripts.data.upload_nc_artifact_to_hf import validate_nc_artifact_for_upload


def _write_gzip_csv(path: Path, rows: list[str]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as output:
        output.write("\n".join(rows) + "\n")


def _make_raw_dataset(root: Path) -> None:
    raw = root / "raw"
    split = root / "split" / "time"
    raw.mkdir(parents=True)
    split.mkdir(parents=True)
    _write_gzip_csv(raw / "num-node-list.csv.gz", ["6"])
    _write_gzip_csv(raw / "num-edge-list.csv.gz", ["8"])
    _write_gzip_csv(
        raw / "node-feat.csv.gz",
        ["1,10", "2,20", "3,30", "4,40", "5,50", "6,60"],
    )
    _write_gzip_csv(raw / "node-label.csv.gz", ["0", "1", "nan", "0", "1", "nan"])
    _write_gzip_csv(
        raw / "edge.csv.gz",
        ["0,1", "1,0", "0,2", "2,0", "3,4", "4,3", "4,5", "5,4"],
    )
    _write_gzip_csv(split / "train.csv.gz", ["0", "3"])
    _write_gzip_csv(split / "valid.csv.gz", ["1"])
    _write_gzip_csv(split / "test.csv.gz", ["4"])


def _make_binary_raw_dataset(root: Path) -> None:
    raw = root / "raw"
    split = root / "split" / "time"
    raw.mkdir(parents=True)
    split.mkdir(parents=True)
    np.savez(
        raw / "data.npz",
        node_feat=np.array(
            [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [6, 60]],
            dtype=np.float32,
        ),
        edge_index=np.array(
            [[0, 1, 0, 2, 3, 4, 4, 5], [1, 0, 2, 0, 4, 3, 5, 4]],
            dtype=np.int64,
        ),
        num_nodes_list=np.array([6], dtype=np.int64),
        num_edges_list=np.array([8], dtype=np.int64),
    )
    np.savez(
        raw / "node-label.npz",
        node_label=np.array([[0], [1], [np.nan], [0], [1], [np.nan]], dtype=np.float32),
    )
    _write_gzip_csv(split / "train.csv.gz", ["0", "3"])
    _write_gzip_csv(split / "valid.csv.gz", ["1"])
    _write_gzip_csv(split / "test.csv.gz", ["4"])


def test_label_balanced_owner_plan_covers_every_node_and_balances_sizes(tmp_path):
    labels = np.array([0, 1, -1, 0, 1, -1, -1, -1, 0, -1], dtype=np.int32)
    plan = build_label_balanced_owner_plan(
        labels,
        n_trainer=3,
        iid_beta=10000.0,
        seed=42,
        owner_path=tmp_path / "owner.u8",
        local_position_path=tmp_path / "local_position.u32",
    )

    owners = np.memmap(tmp_path / "owner.u8", dtype=np.uint8, mode="r", shape=(10,))
    positions = np.memmap(
        tmp_path / "local_position.u32", dtype=np.uint32, mode="r", shape=(10,)
    )

    assert np.all(owners < 3)
    assert sorted(plan.shard_node_counts) == [3, 3, 4]
    for trainer_id, node_count in enumerate(plan.shard_node_counts):
        assert sorted(positions[owners == trainer_id].tolist()) == list(
            range(node_count)
        )


def test_partition_raw_ogb_0hop_writes_complete_local_coordinate_shards(tmp_path):
    dataset_root = tmp_path / "synthetic"
    output_dir = tmp_path / "artifact"
    _make_raw_dataset(dataset_root)

    manifest = partition_raw_ogb_0hop(
        dataset_root=dataset_root,
        output_dir=output_dir,
        n_trainer=2,
        iid_beta=10000.0,
        seed=42,
        checksums=True,
        chunk_rows=2,
    )

    assert manifest["global_node_num"] == 6
    assert manifest["global_edge_num"] == 8
    assert manifest["n_trainer"] == 2
    assert (output_dir / "manifest.json").exists()
    assert not (tmp_path / "artifact.incomplete").exists()
    assert json.loads((output_dir / "manifest.json").read_text()) == manifest

    all_local_nodes = []
    for trainer_id in range(2):
        shard_dir = output_dir / "shards" / f"trainer-{trainer_id:03d}"
        local_nodes = torch.load(shard_dir / "local_node_index.pt", weights_only=True)
        communicate_nodes = torch.load(
            shard_dir / "communicate_node_index.pt", weights_only=True
        )
        features = torch.load(shard_dir / "features.pt", weights_only=True)
        adjacency = torch.load(shard_dir / "adj.pt", weights_only=True)
        metadata = json.loads((shard_dir / "metadata.json").read_text())

        assert torch.equal(local_nodes, communicate_nodes)
        assert torch.equal(local_nodes, local_nodes.sort().values)
        assert features.shape == (local_nodes.numel(), 2)
        assert adjacency.shape[0] == 2
        assert not adjacency.numel() or int(adjacency.max()) < local_nodes.numel()
        assert "features.pt" in metadata["sha256"]
        for split in ("train", "val", "test"):
            indexes = torch.load(shard_dir / f"idx_{split}.pt", weights_only=True)
            split_labels = torch.load(
                shard_dir / f"{split}_labels.pt", weights_only=True
            )
            assert indexes.numel() == split_labels.numel()
            assert not indexes.numel() or int(indexes.max()) < local_nodes.numel()
        all_local_nodes.extend(local_nodes.tolist())

    assert sorted(all_local_nodes) == list(range(6))


def test_partition_raw_ogb_0hop_reads_ogb_binary_arrays_in_chunks(tmp_path):
    dataset_root = tmp_path / "synthetic-binary"
    output_dir = tmp_path / "binary-artifact"
    cache_dir = tmp_path / "binary-cache"
    _make_binary_raw_dataset(dataset_root)

    manifest = partition_raw_ogb_0hop(
        dataset_root=dataset_root,
        output_dir=output_dir,
        n_trainer=2,
        iid_beta=10000.0,
        seed=42,
        checksums=False,
        chunk_rows=2,
        input_format="ogb-binary",
        binary_cache_dir=cache_dir,
    )

    assert manifest["input_format"] == "ogb-binary"
    assert manifest["global_node_num"] == 6
    assert manifest["global_edge_num"] == 8
    assert (cache_dir / "data" / "node_feat.npy").exists()
    assert (cache_dir / "data" / "edge_index.npy").exists()
    assert (cache_dir / "labels" / "node_label.npy").exists()
    assert json.loads((cache_dir / "manifest.json").read_text())["cache_version"] == 1
    auto_source = discover_ogb_source(
        dataset_root,
        input_format="auto",
        binary_cache_dir=cache_dir,
    )
    assert auto_source.input_format == "ogb-binary"
    assert auto_source.features == cache_dir / "data" / "node_feat.npy"

    all_local_nodes = []
    for trainer_id in range(2):
        shard_dir = output_dir / "shards" / f"trainer-{trainer_id:03d}"
        local_nodes = torch.load(shard_dir / "local_node_index.pt", weights_only=True)
        features = torch.load(shard_dir / "features.pt", weights_only=True)
        adjacency = torch.load(shard_dir / "adj.pt", weights_only=True)
        assert features.shape == (local_nodes.numel(), 2)
        assert adjacency.shape[0] == 2
        assert torch.equal(features[:, 0], local_nodes.float() + 1)
        assert torch.equal(features[:, 1], (local_nodes.float() + 1) * 10)
        assert not adjacency.numel() or int(adjacency.max()) < local_nodes.numel()
        all_local_nodes.extend(local_nodes.tolist())
    assert sorted(all_local_nodes) == list(range(6))


def test_local_artifact_loader_reads_a_complete_0hop_shard(tmp_path):
    dataset_root = tmp_path / "synthetic"
    output_dir = tmp_path / "artifact"
    _make_raw_dataset(dataset_root)
    partition_raw_ogb_0hop(
        dataset_root=dataset_root,
        output_dir=output_dir,
        n_trainer=2,
        iid_beta=10000.0,
        seed=42,
        checksums=False,
        chunk_rows=2,
    )

    # A staged worker receives the manifest and only its assigned shard.
    shutil.rmtree(output_dir / "shards" / "trainer-001")

    loaded = load_trainer_data_from_local_artifact(
        0,
        SimpleNamespace(
            local_artifact_dir=str(output_dir),
            use_huggingface=False,
            num_hops=0,
            n_trainer=2,
        ),
    )
    (
        local_node_index,
        communicate_node_index,
        adjacency,
        train_labels,
        val_labels,
        test_labels,
        features,
        idx_train,
        idx_val,
        idx_test,
        global_node_num,
        class_num,
    ) = loaded

    assert torch.equal(local_node_index, communicate_node_index)
    assert features.shape == (local_node_index.numel(), 2)
    assert adjacency.shape[0] == 2
    assert idx_train.numel() == train_labels.numel()
    assert idx_val.numel() == val_labels.numel()
    assert idx_test.numel() == test_labels.numel()
    assert global_node_num.item() == 6
    assert class_num.item() == 2


def test_local_artifact_mmap_maps_only_large_graph_tensors(tmp_path):
    dataset_root = tmp_path / "synthetic"
    output_dir = tmp_path / "artifact"
    _make_raw_dataset(dataset_root)
    partition_raw_ogb_0hop(
        dataset_root=dataset_root,
        output_dir=output_dir,
        n_trainer=2,
        iid_beta=10000.0,
        seed=42,
        checksums=False,
        chunk_rows=2,
    )
    args = SimpleNamespace(
        local_artifact_dir=str(output_dir),
        use_huggingface=False,
        num_hops=0,
        n_trainer=2,
        graph_storage_mode="mmap",
    )

    with patch(
        "fedgraph.trainer_class._load_artifact_tensor",
        wraps=_load_artifact_tensor,
    ) as load_tensor:
        loaded = load_trainer_data_from_local_artifact(0, args)

    mmap_by_name = {
        Path(call.args[0]).name: call.kwargs["memory_map"]
        for call in load_tensor.call_args_list
    }
    assert mmap_by_name["features.pt"] is True
    assert mmap_by_name["adj.pt"] is True
    assert mmap_by_name["train_labels.pt"] is False
    assert mmap_by_name["idx_train.pt"] is False
    assert loaded[2].device.type == "cpu"
    assert loaded[6].device.type == "cpu"

    trainer_args = SimpleNamespace(
        **vars(args),
        seed=42,
        local_step=1,
        method="FedAvg",
    )
    trainer = Trainer_General(
        rank=0,
        args_hidden=8,
        device=torch.device("cpu"),
        args=trainer_args,
    )

    assert trainer.graph_storage_mode == "mmap"
    assert trainer.graph_device.type == "cpu"
    assert trainer.features_memory_mapped is True
    assert trainer.adjacency_memory_mapped is True
    assert trainer.feature_aggregation is trainer.features


@patch("fedgraph.trainer_class.snapshot_download")
def test_huggingface_local_artifact_loader_downloads_only_its_shard(
    mock_snapshot_download, tmp_path
):
    dataset_root = tmp_path / "synthetic"
    output_dir = tmp_path / "artifact"
    _make_raw_dataset(dataset_root)
    partition_raw_ogb_0hop(
        dataset_root=dataset_root,
        output_dir=output_dir,
        n_trainer=2,
        iid_beta=10000.0,
        seed=42,
        checksums=True,
        chunk_rows=2,
    )
    mock_snapshot_download.return_value = str(output_dir)

    loaded = load_trainer_data_from_huggingface_local_artifact(
        1,
        SimpleNamespace(
            hf_local_artifact_repo="FedGraph/ogbn-arxiv-2-0hop-v1",
            hf_local_artifact_revision="candidate",
            hf_local_artifact_cache_dir=str(tmp_path / "hf-cache"),
            num_hops=0,
            n_trainer=2,
            graph_storage_mode="mmap",
        ),
    )

    assert loaded[0].numel() > 0
    mock_snapshot_download.assert_called_once_with(
        repo_id="FedGraph/ogbn-arxiv-2-0hop-v1",
        repo_type="dataset",
        allow_patterns=[
            "manifest.json",
            "shards/trainer-001/metadata.json",
            "shards/trainer-001/local_node_index.pt",
            "shards/trainer-001/communicate_node_index.pt",
            "shards/trainer-001/adj.pt",
            "shards/trainer-001/train_labels.pt",
            "shards/trainer-001/val_labels.pt",
            "shards/trainer-001/test_labels.pt",
            "shards/trainer-001/features.pt",
            "shards/trainer-001/idx_train.pt",
            "shards/trainer-001/idx_val.pt",
            "shards/trainer-001/idx_test.pt",
            "shards/trainer-001/global_node_num.pt",
            "shards/trainer-001/class_num.pt",
        ],
        revision="candidate",
        cache_dir=str(tmp_path / "hf-cache"),
    )


def test_upload_validation_checks_complete_artifact_without_loading_tensors(tmp_path):
    dataset_root = tmp_path / "synthetic"
    output_dir = tmp_path / "artifact"
    _make_raw_dataset(dataset_root)
    partition_raw_ogb_0hop(
        dataset_root=dataset_root,
        output_dir=output_dir,
        n_trainer=2,
        iid_beta=10000.0,
        seed=42,
        checksums=True,
        chunk_rows=2,
    )

    manifest = validate_nc_artifact_for_upload(output_dir)

    assert manifest["n_trainer"] == 2
