import json
from types import SimpleNamespace

import pytest
import torch

from fedgraph.graph_storage import (
    FeatureAggregationStore,
    relabel_adjacency_to_mmap_cache,
)
from fedgraph.trainer_class import Trainer_General


def test_feature_aggregation_store_publishes_ordered_mmap_tensor(tmp_path):
    store = FeatureAggregationStore(
        tmp_path,
        trainer_id=2,
        row_count=3,
        feature_dim=2,
    )

    store.write(0, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    store.write(2, torch.tensor([[5.0, 6.0]]))
    result = store.finalize()

    torch.testing.assert_close(
        result,
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )
    assert store.data_path.stat().st_size == 3 * 2 * torch.float32.itemsize
    assert json.loads(store.metadata_path.read_text(encoding="utf-8")) == {
        "dtype": "float32",
        "feature_dim": 2,
        "layout": "row_major",
        "row_count": 3,
        "size_bytes": 24,
        "store_version": 1,
    }


def test_feature_aggregation_store_rejects_gaps_and_cleans_up_abort(tmp_path):
    store = FeatureAggregationStore(
        tmp_path,
        trainer_id=0,
        row_count=2,
        feature_dim=1,
    )

    with pytest.raises(ValueError, match="expected start 0"):
        store.write(1, torch.ones((1, 1)))
    store.write(0, torch.ones((1, 1)))
    with pytest.raises(RuntimeError, match="incomplete"):
        store.finalize()

    temporary_data_path = store.temporary_data_path
    store.abort()
    assert not temporary_data_path.exists()
    assert not store.data_path.exists()


def test_bounded_relabel_builds_and_reuses_mmap_cache(tmp_path):
    source_path = tmp_path / "adj.pt"
    adjacency = torch.tensor(
        [
            [10, 20, 40, 30],
            [20, 30, 10, 99],
        ]
    )
    torch.save(adjacency, source_path)
    node_indexes = torch.tensor([10, 20, 30])

    result = relabel_adjacency_to_mmap_cache(
        adjacency,
        node_indexes,
        source_path=source_path,
        cache_dir=tmp_path / "cache",
        trainer_id=7,
        chunk_edges=1,
    )

    assert not result.cache_hit
    assert result.source_edge_count == 4
    assert result.edge_count == 2
    assert result.dropped_edge_count == 2
    assert result.cache_path.stat().st_size == 2 * 2 * torch.int64.itemsize
    torch.testing.assert_close(
        result.edge_index,
        torch.tensor([[0, 1], [1, 2]]),
    )
    assert result.edge_index.is_contiguous()

    cached = relabel_adjacency_to_mmap_cache(
        adjacency,
        node_indexes,
        source_path=source_path,
        cache_dir=tmp_path / "cache",
        trainer_id=7,
        chunk_edges=4,
    )

    assert cached.cache_hit
    assert cached.cache_path == result.cache_path
    assert cached.edge_count == 2
    torch.testing.assert_close(cached.edge_index, result.edge_index)


def test_trainer_mmap_relabel_records_cache_telemetry(tmp_path):
    source_path = tmp_path / "adj.pt"
    adjacency = torch.tensor([[10, 30, 99], [30, 10, 10]])
    torch.save(adjacency, source_path)
    args = SimpleNamespace(
        graph_relabel_cache_dir=str(tmp_path / "cache"),
        graph_relabel_chunk_edges=1,
    )
    trainer = object.__new__(Trainer_General)
    trainer.graph_storage_mode = "mmap"
    trainer.args = args
    trainer.adj = torch.load(
        source_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    trainer.adjacency_artifact_path = source_path
    trainer.rank = 3
    trainer.adjacency_memory_mapped = False

    trainer.relabel_adj(torch.tensor([10, 30]))

    torch.testing.assert_close(trainer.adj, torch.tensor([[0, 1], [1, 0]]))
    assert trainer.adjacency_memory_mapped
    assert trainer.adjacency_relabel_cache_hit is False
    assert trainer.adjacency_relabel_source_edge_count == 3
    assert trainer.adjacency_relabel_output_edge_count == 2
    assert trainer.adjacency_relabel_dropped_edge_count == 1
