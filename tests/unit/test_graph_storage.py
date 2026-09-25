from types import SimpleNamespace

import torch

from fedgraph.graph_storage import relabel_adjacency_to_mmap_cache
from fedgraph.trainer_class import Trainer_General


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
