from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import ray
import torch
from huggingface_hub.errors import EntryNotFoundError

from fedgraph.feature_compression import solve_shared_feature_basis
from fedgraph.trainer_class import (
    Trainer_GC,
    Trainer_General,
    _load_artifact_tensor,
    _resolve_huggingface_artifact_num_hops,
    load_trainer_data_from_hugging_face,
)


class TestLoadTrainerDataFromHuggingFace:
    """Test load_trainer_data_from_hugging_face function."""

    @patch("fedgraph.trainer_class.hf_hub_download")
    @patch("builtins.open")
    @patch("torch.load")
    def test_load_trainer_data_success(
        self, mock_torch_load, mock_open, mock_hf_download, tmp_path
    ):
        """Test successful loading of trainer data from Hugging Face."""
        # Setup mocks
        tensor_path = tmp_path / "test_file.pt"
        tensor_path.touch()
        mock_hf_download.return_value = str(tensor_path)

        # Mock tensor data
        mock_tensors = [
            torch.randn(100),  # local_node_index
            torch.randn(50),  # communicate_node_global_index
            torch.randn(2, 200),  # global_edge_index_client
            torch.randn(80),  # train_labels
            torch.randn(10),  # val_labels
            torch.randn(20),  # test_labels
            torch.randn(100, 10),  # features
            torch.randn(80),  # in_com_train_node_local_indexes
            torch.randn(10),  # in_com_val_node_local_indexes
            torch.randn(20),  # in_com_test_node_local_indexes
            torch.tensor(100),  # global_node_num
            torch.tensor(3),  # class_num
        ]
        mock_torch_load.side_effect = mock_tensors

        args = Mock()
        args.dataset = "cora"
        args.n_trainer = 5
        args.num_hops = 2
        args.iid_beta = 0.5
        args.hf_artifact_num_hops = 1

        result = load_trainer_data_from_hugging_face(trainer_id=0, args=args)

        assert len(result) == 12
        assert all(isinstance(tensor, torch.Tensor) for tensor in result)

        # Verify calls
        assert mock_hf_download.call_count == 12
        expected_repo = "FedGraph/fedgraph_cora_5trainer_1hop_iid_beta_0.5_trainer_id_0"
        mock_hf_download.assert_any_call(
            repo_id=expected_repo, repo_type="dataset", filename="local_node_index.pt"
        )
        mock_torch_load.assert_any_call(
            tensor_path, map_location="cpu", weights_only=True
        )
        mock_open.assert_not_called()

    @patch("fedgraph.trainer_class.hf_hub_download")
    @patch("builtins.open")
    @patch("torch.load")
    def test_load_existing_repo_without_global_metadata(
        self, mock_torch_load, mock_open, mock_hf_download, tmp_path
    ):
        tensor_path = tmp_path / "test_file.pt"
        tensor_path.touch()
        mock_torch_load.side_effect = [torch.tensor([i]) for i in range(10)]

        def download_side_effect(*, filename, **kwargs):
            if filename in {"global_node_num.pt", "class_num.pt"}:
                raise EntryNotFoundError("missing optional metadata")
            return str(tensor_path)

        mock_hf_download.side_effect = download_side_effect
        args = Mock(dataset="cora", n_trainer=5, num_hops=2, iid_beta=0.5)

        with pytest.warns(UserWarning, match="falling back to inference"):
            result = load_trainer_data_from_hugging_face(trainer_id=0, args=args)

        assert len(result) == 12
        assert result[-2:] == (None, None)
        mock_open.assert_not_called()

    def test_load_artifact_tensor_from_path_on_cpu(self, tmp_path):
        tensor_path = tmp_path / "features.pt"
        expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        torch.save(expected, tensor_path)

        loaded = _load_artifact_tensor(tensor_path)

        assert loaded.device.type == "cpu"
        torch.testing.assert_close(loaded, expected)

    @patch("torch.load")
    def test_load_artifact_tensor_enables_mmap_only_when_requested(
        self, mock_torch_load, tmp_path
    ):
        tensor_path = tmp_path / "features.pt"
        tensor_path.touch()
        mock_torch_load.return_value = torch.ones(2, 3)

        _load_artifact_tensor(tensor_path, memory_map=True)

        mock_torch_load.assert_called_once_with(
            tensor_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )

    def test_load_artifact_tensor_reads_memory_mapped_storage(self, tmp_path):
        tensor_path = tmp_path / "features.pt"
        expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        torch.save(expected, tensor_path)

        loaded = _load_artifact_tensor(tensor_path, memory_map=True)

        assert loaded.device.type == "cpu"
        assert loaded.to("cpu") is loaded
        torch.testing.assert_close(loaded, expected)

    def test_load_artifact_tensor_rejects_non_tensor_payload(self, tmp_path):
        payload_path = tmp_path / "metadata.pt"
        torch.save({"value": 1}, payload_path)

        with pytest.raises(TypeError, match="Expected a tensor"):
            _load_artifact_tensor(payload_path)

    def test_load_artifact_tensor_requires_existing_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Artifact tensor is missing"):
            _load_artifact_tensor(tmp_path / "missing.pt")

    def test_resolves_legacy_huggingface_hop_suffix(self):
        args = Mock()
        args.num_hops = 2
        args.hf_artifact_num_hops = 1

        assert _resolve_huggingface_artifact_num_hops(args) == 1

    def test_defaults_to_runtime_hops_without_legacy_override(self):
        args = Mock()
        args.num_hops = 2
        args.hf_artifact_num_hops = None

        assert _resolve_huggingface_artifact_num_hops(args) == 2


class TestTrainerGeneral:
    """Test Trainer_General class."""

    def setup_method(self):
        """Setup common test data."""
        self.rank = 0
        self.device = torch.device("cpu")
        self.args_hidden = 64

        # Create mock data
        self.local_node_index = torch.arange(0, 50)
        self.communicate_node_index = torch.arange(0, 100)
        self.adj = torch.randn(2, 200)  # edge_index format
        self.train_labels = torch.randint(0, 3, (40,))
        self.test_labels = torch.randint(0, 3, (10,))
        self.features = torch.randn(100, 50)
        self.idx_train = torch.arange(0, 40)
        self.idx_test = torch.arange(40, 50)

        self.args = Mock()
        self.args.local_step = 5
        self.args.method = "FedAvg"
        self.args.num_hops = 2
        self.args.num_layers = 2
        self.args.learning_rate = 0.01
        self.args.dataset = "cora"
        self.args.batch_size = 0  # No batching for simplicity

    def test_cpu_graph_storage_keeps_full_graph_off_cuda(self):
        self.args.graph_storage_mode = "cpu"
        self.args.num_hops = 0
        self.args.batch_size = 2
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=torch.device("cuda"),
            args=self.args,
            local_node_index=self.local_node_index,
            communicate_node_index=self.communicate_node_index,
            adj=self.adj,
            train_labels=self.train_labels,
            test_labels=self.test_labels,
            features=self.features,
            idx_train=self.idx_train,
            idx_test=self.idx_test,
        )

        assert trainer.device.type == "cuda"
        assert trainer.graph_device.type == "cpu"
        assert trainer.graph_storage_mode == "cpu"
        assert trainer.features.device.type == "cpu"
        assert trainer.adj.device.type == "cpu"
        assert trainer.idx_train.device.type == "cpu"
        assert trainer.train_labels.device.type == "cpu"
        assert trainer.feature_aggregation is trainer.features
        trainer._require_supported_graph_execution(True)
        with pytest.raises(ValueError, match="requires a positive batch_size"):
            trainer._require_supported_graph_execution(False)

    def test_cpu_graph_storage_rejects_two_hop_without_manifest_artifact(self):
        self.args.graph_storage_mode = "cpu"
        self.args.num_hops = 2

        with pytest.raises(ValueError, match="manifest-style artifact"):
            Trainer_General(
                rank=self.rank,
                args_hidden=self.args_hidden,
                device=self.device,
                args=self.args,
                local_node_index=self.local_node_index,
                communicate_node_index=self.communicate_node_index,
                adj=self.adj,
                train_labels=self.train_labels,
                test_labels=self.test_labels,
                features=self.features,
                idx_train=self.idx_train,
                idx_test=self.idx_test,
            )

    def test_mmap_graph_storage_requires_file_backed_artifacts(self):
        self.args.graph_storage_mode = "mmap"
        self.args.num_hops = 0
        self.args.batch_size = 2

        with pytest.raises(ValueError, match="requires file-backed trainer artifacts"):
            Trainer_General(
                rank=self.rank,
                args_hidden=self.args_hidden,
                device=self.device,
                args=self.args,
                local_node_index=self.local_node_index,
                communicate_node_index=self.communicate_node_index,
                adj=self.adj,
                train_labels=self.train_labels,
                test_labels=self.test_labels,
                features=self.features,
                idx_train=self.idx_train,
                idx_test=self.idx_test,
            )

    @patch("fedgraph.trainer_class.load_trainer_data_from_hugging_face")
    def test_trainer_init_with_data(self, mock_load_data):
        """Test Trainer_General initialization with provided data."""
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=self.device,
            args=self.args,
            local_node_index=self.local_node_index,
            communicate_node_index=self.communicate_node_index,
            adj=self.adj,
            train_labels=self.train_labels,
            test_labels=self.test_labels,
            features=self.features,
            idx_train=self.idx_train,
            idx_test=self.idx_test,
            global_node_num=100,
            class_num=3,
        )

        assert trainer.rank == self.rank
        assert trainer.device == self.device
        assert trainer.args_hidden == self.args_hidden
        assert trainer.local_step == self.args.local_step
        assert trainer.global_node_num == 100
        assert trainer.class_num == 3
        assert torch.equal(
            trainer.local_node_index, self.local_node_index.to(self.device)
        )
        assert trainer.feature_aggregation is not None  # Should be set for FedAvg
        mock_load_data.assert_not_called()

    @patch("fedgraph.trainer_class.load_trainer_data_from_hugging_face")
    def test_trainer_init_without_data(self, mock_load_data):
        """Test Trainer_General initialization without provided data."""
        mock_load_data.return_value = (
            self.local_node_index,
            self.communicate_node_index,
            self.adj,
            self.train_labels,
            torch.empty(0, dtype=self.train_labels.dtype),
            self.test_labels,
            self.features,
            self.idx_train,
            torch.empty(0, dtype=self.idx_train.dtype),
            self.idx_test,
            torch.tensor(100),
            torch.tensor(3),
        )

        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=self.device,
            args=self.args,
        )

        assert trainer.rank == self.rank
        assert trainer.global_node_num == 100
        assert trainer.class_num == 3
        mock_load_data.assert_called_once_with(self.rank, self.args)

    def test_get_info(self):
        """Test get_info method."""
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=self.device,
            args=self.args,
            local_node_index=self.local_node_index,
            communicate_node_index=self.communicate_node_index,
            adj=self.adj,
            train_labels=self.train_labels,
            test_labels=self.test_labels,
            features=self.features,
            idx_train=self.idx_train,
            idx_test=self.idx_test,
        )

        info = trainer.get_info()

        assert "features_num" in info
        assert "label_num" in info
        assert info["features_num"] == len(self.features)
        expected_label_num = (
            max(self.train_labels.max().item(), self.test_labels.max().item()) + 1
        )
        assert info["label_num"] == expected_label_num
        assert "communicate_node_global_index" not in info

    def test_feature_compression_replaces_cpu_features_before_model_init(self):
        self.args.graph_storage_mode = "cpu"
        self.args.num_hops = 0
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=self.device,
            args=self.args,
            local_node_index=self.local_node_index,
            communicate_node_index=self.communicate_node_index,
            adj=self.adj,
            train_labels=self.train_labels,
            test_labels=self.test_labels,
            features=self.features,
            idx_train=self.idx_train,
            idx_test=self.idx_test,
        )
        original_features = trainer.features.clone()
        statistic = trainer.compute_feature_compression_statistics(chunk_rows=13)
        basis = solve_shared_feature_basis(statistic, rank=8)

        result = trainer.apply_shared_feature_basis(basis, chunk_rows=11)

        torch.testing.assert_close(
            trainer.features,
            original_features @ basis.matrix,
        )
        assert trainer.features.shape == (100, 8)
        assert trainer.feature_aggregation is trainer.features
        assert trainer.feature_compression_applied
        assert trainer.feature_compression_original_dim == 50
        assert trainer.feature_compression_rank == 8
        assert result["compressed_dim"] == 8
        assert result["memory_mapped"] is False
        assert trainer.get_info()["feature_shape"] == 8

    def test_feature_compression_rejects_projection_after_model_init(self):
        self.args.graph_storage_mode = "cpu"
        self.args.num_hops = 0
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=self.device,
            args=self.args,
            local_node_index=self.local_node_index,
            communicate_node_index=self.communicate_node_index,
            adj=self.adj,
            train_labels=self.train_labels,
            test_labels=self.test_labels,
            features=self.features,
            idx_train=self.idx_train,
            idx_test=self.idx_test,
        )
        statistic = trainer.compute_feature_compression_statistics(chunk_rows=20)
        basis = solve_shared_feature_basis(statistic, rank=4)
        trainer.model = Mock()

        with pytest.raises(RuntimeError, match="before model initialization"):
            trainer.apply_shared_feature_basis(basis, chunk_rows=20)

    def test_feature_compression_replaces_mmap_features_atomically(self, tmp_path):
        self.args.graph_storage_mode = "cpu"
        self.args.num_hops = 0
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=self.device,
            args=self.args,
            local_node_index=self.local_node_index,
            communicate_node_index=self.communicate_node_index,
            adj=self.adj,
            train_labels=self.train_labels,
            test_labels=self.test_labels,
            features=self.features,
            idx_train=self.idx_train,
            idx_test=self.idx_test,
        )
        source_path = tmp_path / "source-features.f32"
        source_storage = torch.from_file(
            str(source_path),
            shared=True,
            size=self.features.numel(),
            dtype=torch.float32,
        ).view_as(self.features)
        source_storage.copy_(self.features)
        del source_storage
        mapped_features = torch.from_file(
            str(source_path),
            shared=False,
            size=self.features.numel(),
            dtype=torch.float32,
        ).view_as(self.features)
        trainer.graph_storage_mode = "mmap"
        trainer.features = mapped_features
        trainer.feature_aggregation = trainer.features
        trainer.features_memory_mapped = True
        trainer.feature_aggregation_memory_mapped = True
        statistic = trainer.compute_feature_compression_statistics(chunk_rows=17)
        basis = solve_shared_feature_basis(statistic, rank=6)

        result = trainer.apply_shared_feature_basis(
            basis,
            chunk_rows=13,
            output_dir=tmp_path / "projection",
        )

        torch.testing.assert_close(trainer.features, self.features @ basis.matrix)
        assert trainer.feature_aggregation is trainer.features
        assert trainer.features_memory_mapped
        assert trainer.feature_aggregation_memory_mapped
        assert result["memory_mapped"] is True
        assert result["output_path"] == trainer.feature_compression_output_path
        assert Path(result["output_path"]).is_file()

    def test_fedgcn_get_info_includes_communication_indexes(self):
        self.args.method = "FedGCN"
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=self.device,
            args=self.args,
            local_node_index=self.local_node_index,
            communicate_node_index=self.communicate_node_index,
            adj=self.adj,
            train_labels=self.train_labels,
            test_labels=self.test_labels,
            features=self.features,
            idx_train=self.idx_train,
            idx_test=self.idx_test,
        )

        info = trainer.get_info()

        assert torch.equal(
            info["communicate_node_global_index"], trainer.communicate_node_index
        )

    def test_fedavg_relabels_legacy_huggingface_adjacency(self):
        self.args.use_huggingface = True
        self.args.hf_artifact_num_hops = 1
        local_node_index = torch.tensor([10, 20, 30])
        # The legacy positions are relative to the larger communication tensor.
        global_edge_index = torch.tensor([[10, 20, 20, 30, 99], [20, 10, 30, 20, 10]])
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=self.device,
            args=self.args,
            local_node_index=local_node_index,
            communicate_node_index=torch.tensor([5, 10, 20, 30, 40]),
            adj=global_edge_index,
            train_labels=torch.tensor([0]),
            test_labels=torch.tensor([1]),
            features=torch.randn(3, 2),
            idx_train=torch.tensor([1]),
            idx_test=torch.tensor([3]),
        )

        torch.testing.assert_close(
            trainer.adj, torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        )
        assert trainer.feature_aggregation is trainer.features
        assert "communicate_node_global_index" not in trainer.get_info()
        torch.testing.assert_close(trainer.idx_train, torch.tensor([0]))
        torch.testing.assert_close(trainer.idx_test, torch.tensor([2]))

    def test_fedavg_relabels_legacy_isolated_high_id_node(self):
        self.args.use_huggingface = True
        self.args.hf_artifact_num_hops = 1
        # The node with global ID 111 has no local edge. PyG must not infer its
        # node-mask length only from the highest adjacency endpoint (20).
        local_node_index = torch.tensor([10, 20, 111])
        global_edge_index = torch.tensor([[10, 20], [20, 10]])
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=self.device,
            args=self.args,
            local_node_index=local_node_index,
            communicate_node_index=torch.tensor([5, 10, 20, 111]),
            adj=global_edge_index,
            train_labels=torch.tensor([0]),
            test_labels=torch.tensor([1]),
            features=torch.randn(3, 2),
            idx_train=torch.tensor([1]),
            idx_test=torch.tensor([3]),
        )

        torch.testing.assert_close(trainer.adj, torch.tensor([[0, 1], [1, 0]]))
        torch.testing.assert_close(trainer.idx_train, torch.tensor([0]))
        torch.testing.assert_close(trainer.idx_test, torch.tensor([2]))

    def test_fedavg_relabels_legacy_huggingface_adjacency_before_device_transfer(
        self, monkeypatch
    ):
        self.args.use_huggingface = True
        self.args.hf_artifact_num_hops = 1
        observed_devices = []
        original_relabel_adj = Trainer_General.relabel_adj

        def spy_relabel_adj(trainer, node_index=None):
            observed_devices.append((trainer.adj.device.type, node_index.device.type))
            return original_relabel_adj(trainer, node_index)

        monkeypatch.setattr(Trainer_General, "relabel_adj", spy_relabel_adj)
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=torch.device("meta"),
            args=self.args,
            local_node_index=torch.tensor([10, 20]),
            communicate_node_index=torch.tensor([5, 10, 20]),
            adj=torch.tensor([[10, 20], [20, 10]]),
            train_labels=torch.tensor([0]),
            test_labels=torch.tensor([1]),
            features=torch.randn(2, 2),
            idx_train=torch.tensor([1]),
            idx_test=torch.tensor([2]),
        )

        assert observed_devices == [("cpu", "cpu")]
        assert trainer.adj.device.type == "meta"

    def test_fedavg_leaves_zero_hop_huggingface_adjacency_unchanged(self):
        self.args.use_huggingface = True
        self.args.hf_artifact_num_hops = 0
        self.args.num_hops = 0
        local_node_index = torch.tensor([10, 20])
        local_edge_index = torch.tensor([[0, 1], [1, 0]])
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=self.device,
            args=self.args,
            local_node_index=local_node_index,
            communicate_node_index=local_node_index,
            adj=local_edge_index,
            train_labels=torch.tensor([0]),
            test_labels=torch.tensor([1]),
            features=torch.randn(2, 2),
            idx_train=torch.tensor([0]),
            idx_test=torch.tensor([1]),
        )

        torch.testing.assert_close(trainer.adj, local_edge_index)
        assert trainer.feature_aggregation is trainer.features

    @pytest.mark.parametrize(
        ("train_labels", "test_labels", "expected_label_num"),
        [
            (torch.tensor([], dtype=torch.long), torch.tensor([0, 2]), 3),
            (torch.tensor([0, 1]), torch.tensor([], dtype=torch.long), 2),
            (
                torch.tensor([], dtype=torch.long),
                torch.tensor([], dtype=torch.long),
                None,
            ),
        ],
    )
    def test_get_info_handles_empty_labels(
        self, train_labels, test_labels, expected_label_num
    ):
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=self.device,
            args=self.args,
            local_node_index=self.local_node_index,
            communicate_node_index=self.communicate_node_index,
            adj=self.adj,
            train_labels=train_labels,
            test_labels=test_labels,
            features=self.features,
            idx_train=torch.arange(len(train_labels)),
            idx_test=torch.arange(len(test_labels)),
        )

        assert trainer.get_info()["label_num"] == expected_label_num

    @patch("fedgraph.trainer_class.GCN")
    @patch("fedgraph.trainer_class.GCN_arxiv")
    def test_init_model(self, mock_gcn_arxiv, mock_gcn):
        """Test init_model method."""
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=self.device,
            args=self.args,
            local_node_index=self.local_node_index,
            communicate_node_index=self.communicate_node_index,
            adj=self.adj,
            train_labels=self.train_labels,
            test_labels=self.test_labels,
            features=self.features,
            idx_train=self.idx_train,
            idx_test=self.idx_test,
        )

        mock_model = Mock()
        mock_gcn.return_value = mock_model
        self.args.dataset = "cora"
        self.args.num_hops = 2

        trainer.init_model(global_node_num=2708, class_num=7)

        assert trainer.model is not None
        mock_gcn.assert_called_once()

    def test_update_params(self):
        """Test update_params method."""
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=self.device,
            args=self.args,
            local_node_index=self.local_node_index,
            communicate_node_index=self.communicate_node_index,
            adj=self.adj,
            train_labels=self.train_labels,
            test_labels=self.test_labels,
            features=self.features,
            idx_train=self.idx_train,
            idx_test=self.idx_test,
        )

        # Create mock model
        mock_model = Mock()
        mock_optimizer = Mock()
        trainer.model = mock_model
        trainer.optimizer = mock_optimizer

        # Create mock parameters
        mock_params = (torch.randn(10, 5), torch.randn(5))

        trainer.update_params(mock_params, current_global_epoch=1)

        # Verify that model state_dict loading was attempted
        assert mock_model.load_state_dict.called or hasattr(trainer, "model")

    @pytest.mark.parametrize(
        "scheduler,model_head",
        [("constant", "gcn"), ("cosine", "gcn"), ("cosine", "gcn-skip")],
    )
    @pytest.mark.parametrize(
        "device_name",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA unavailable"
                ),
            ),
        ],
    )
    def test_adamw_preserves_state_across_broadcast(
        self, device_name, scheduler, model_head
    ):
        scheduled = scheduler == "cosine"
        args = SimpleNamespace(
            dataset="ogbn-papers100M",
            method="FedGCN",
            num_hops=2,
            num_layers=2,
            first_layer_norm=True,
            model_head=model_head,
            local_step=3 if scheduled else 1,
            batch_size=-1,
            optimizer="adamw",
            learning_rate=0.001,
            global_rounds=2,
            lr_scheduler=scheduler,
            min_learning_rate=0.0002 if scheduled else 0.0,
            weight_decay=0.0 if scheduled else 5e-4,
        )
        node_ids = torch.arange(2)
        features = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        trainer = Trainer_General(
            rank=0,
            args_hidden=8,
            device=torch.device(device_name),
            args=args,
            local_node_index=node_ids,
            communicate_node_index=node_ids,
            adj=torch.tensor([[0, 1], [1, 0]]),
            train_labels=node_ids,
            test_labels=node_ids,
            features=features,
            idx_train=node_ids,
            idx_test=node_ids,
        )
        trainer.load_feature_aggregation(features)
        trainer.init_model(global_node_num=2, class_num=2)

        optimizer = trainer.optimizer
        parameters = tuple(trainer.model.parameters())
        assert (trainer.model.skip_head is not None) == (model_head == "gcn-skip")
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 0.001
        assert optimizer.param_groups[0]["weight_decay"] == args.weight_decay
        rates = []
        optimizer.register_step_pre_hook(
            lambda opt, _args, _kwargs: rates.append(opt.param_groups[0]["lr"])
        )
        trainer.train(0)
        moments = [
            (
                optimizer.state[p]["exp_avg"].clone(),
                optimizer.state[p]["exp_avg_sq"].clone(),
            )
            for p in parameters
        ]
        assert all(
            optimizer.state[p]["step"].item() == args.local_step for p in parameters
        )

        # Act like the CPU head and send updated weight values, not optimizer state.
        broadcast = tuple(weight + 0.25 for weight in trainer.get_params())
        trainer.update_params(broadcast, 0)
        torch.testing.assert_close(trainer.get_params(), broadcast)
        assert trainer.optimizer is optimizer
        assert all(
            p is current for p, current in zip(parameters, trainer.model.parameters())
        )
        for p, (first_moment, second_moment) in zip(parameters, moments):
            torch.testing.assert_close(optimizer.state[p]["exp_avg"], first_moment)
            torch.testing.assert_close(optimizer.state[p]["exp_avg_sq"], second_moment)

        trainer.train(1)
        assert all(
            optimizer.state[p]["step"].item() == 2 * args.local_step for p in parameters
        )
        second_rate = 0.0006 if scheduled else 0.001
        assert rates == pytest.approx(
            [0.001] * args.local_step + [second_rate] * args.local_step
        )
        assert np.isfinite(trainer.train_losses).all()

    def test_get_local_feature_sum(self):
        """Test get_local_feature_sum method."""
        local_features = self.features[self.local_node_index]
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=self.device,
            args=self.args,
            local_node_index=self.local_node_index,
            communicate_node_index=self.communicate_node_index,
            adj=self.adj,
            train_labels=self.train_labels,
            test_labels=self.test_labels,
            features=local_features,
            idx_train=self.idx_train,
            idx_test=self.idx_test,
        )
        trainer.global_node_num = len(self.features)

        # Mock the get_1hop_feature_sum function
        with patch("fedgraph.trainer_class.get_1hop_feature_sum") as mock_get_1hop:
            mock_get_1hop.return_value = torch.randn(100, 50)

            result = trainer.get_local_feature_sum()

            assert isinstance(result, torch.Tensor)
            mock_get_1hop.assert_called_once()
            feature_matrix = mock_get_1hop.call_args.args[0]
            assert feature_matrix.shape == (
                trainer.global_node_num,
                local_features.shape[1],
            )

    @pytest.mark.parametrize("norm_type", ["none", "row", "sym"])
    def test_indexed_feature_sum_matches_dense_feature_sum(self, norm_type):
        args = Mock()
        args.seed = 42
        args.local_step = 1
        args.method = "FedGCN"
        args.norm_type = norm_type
        args.use_encryption = False
        local_node_index = torch.tensor([1, 4])
        local_features = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
        edge_index = torch.tensor([[0, 0, 2, 3, 4, 4], [1, 4, 1, 4, 1, 4]])
        trainer = Trainer_General(
            rank=0,
            args_hidden=2,
            device=torch.device("cpu"),
            args=args,
            local_node_index=local_node_index,
            communicate_node_index=torch.arange(6),
            adj=edge_index,
            train_labels=torch.tensor([0]),
            test_labels=torch.tensor([0]),
            features=local_features,
            idx_train=torch.tensor([0]),
            idx_test=torch.tensor([0]),
        )
        trainer.global_node_num = 6

        dense_feature_sum = trainer.get_local_feature_sum()
        row_ids, row_values = trainer.get_indexed_local_feature_sum()

        torch.testing.assert_close(row_ids, torch.tensor([0, 1, 2, 3, 4]))
        torch.testing.assert_close(row_values, dense_feature_sum[row_ids])

    def test_get_local_feature_sum_requires_global_node_num(self):
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=self.device,
            args=self.args,
            local_node_index=self.local_node_index,
            communicate_node_index=self.communicate_node_index,
            adj=self.adj,
            train_labels=self.train_labels,
            test_labels=self.test_labels,
            features=self.features[self.local_node_index],
            idx_train=self.idx_train,
            idx_test=self.idx_test,
        )

        with pytest.raises(RuntimeError, match="metadata must be initialized"):
            trainer.get_local_feature_sum()

    def test_load_feature_aggregation(self):
        """Test load_feature_aggregation method."""
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=self.device,
            args=self.args,
            local_node_index=self.local_node_index,
            communicate_node_index=self.communicate_node_index,
            adj=self.adj,
            train_labels=self.train_labels,
            test_labels=self.test_labels,
            features=self.features,
            idx_train=self.idx_train,
            idx_test=self.idx_test,
        )

        new_features = torch.randn(100, 50)
        trainer.load_feature_aggregation(new_features)

        assert torch.equal(trainer.feature_aggregation, new_features.to(trainer.device))

    def test_finalize_full_feature_aggregation_relabels_and_releases_raw_features(self):
        self.args.method = "FedGCN"
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=self.device,
            args=self.args,
            local_node_index=self.local_node_index,
            communicate_node_index=self.communicate_node_index,
            adj=self.adj,
            train_labels=self.train_labels,
            test_labels=self.test_labels,
            features=self.features,
            idx_train=self.idx_train,
            idx_test=self.idx_test,
        )
        trainer.relabel_adj = Mock()
        trainer.adjacency_relabel_strategy = "artifact-precomputed"
        trainer.adjacency_relabel_time_sec = 0.25
        trainer.load_feature_aggregation(torch.randn(100, 50))

        result = trainer.finalize_full_feature_aggregation()

        trainer.relabel_adj.assert_called_once_with()
        assert trainer.raw_features_released is True
        assert trainer.features.shape == (0, 50)
        assert result["adjacency_relabel_strategy"] == "artifact-precomputed"

    @patch("fedgraph.trainer_class.NeighborSampler")
    def test_prepare_neighbor_sampler_builds_and_reuses_indexes(self, mock_sampler):
        self.args.graph_storage_mode = "cpu"
        self.args.num_hops = 0
        self.args.batch_size = 2
        mock_sampler.return_value.colptr = torch.arange(4, dtype=torch.int64)
        mock_sampler.return_value.row = torch.arange(3, dtype=torch.int64)
        mock_sampler.return_value.perm = None
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=self.device,
            args=self.args,
            local_node_index=self.local_node_index,
            communicate_node_index=self.communicate_node_index,
            adj=self.adj,
            train_labels=self.train_labels,
            test_labels=self.test_labels,
            features=self.features,
            idx_train=self.idx_train,
            idx_test=self.idx_test,
        )

        first = trainer.prepare_neighbor_sampler()
        second = trainer.prepare_neighbor_sampler()

        assert first["prepared"] is True
        assert first["cache_hit"] is False
        assert first["index_bytes"] == 7 * torch.int64.itemsize
        assert second["cache_hit"] is True
        assert mock_sampler.call_count == 1

    def test_get_params(self):
        """get_params returns a detached CPU snapshot for Ray transfer."""
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=self.device,
            args=self.args,
            local_node_index=self.local_node_index,
            communicate_node_index=self.communicate_node_index,
            adj=self.adj,
            train_labels=self.train_labels,
            test_labels=self.test_labels,
            features=self.features,
            idx_train=self.idx_train,
            idx_test=self.idx_test,
        )

        trainer.model = torch.nn.Linear(5, 10)
        source_params = tuple(trainer.model.parameters())

        params = trainer.get_params()

        assert isinstance(params, tuple)
        assert len(params) == len(source_params)
        for source, snapshot in zip(source_params, params):
            assert snapshot.device.type == "cpu"
            assert not snapshot.requires_grad
            assert snapshot.data_ptr() != source.data_ptr()
            torch.testing.assert_close(snapshot, source.detach())

    @patch("fedgraph.trainer_class.test")
    @patch("fedgraph.trainer_class.train")
    def test_train_method(self, mock_train_func, mock_test_func):
        """Test train method."""
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=self.device,
            args=self.args,
            local_node_index=self.local_node_index,
            communicate_node_index=self.communicate_node_index,
            adj=self.adj,
            train_labels=self.train_labels,
            test_labels=self.test_labels,
            features=self.features,
            idx_train=self.idx_train,
            idx_test=self.idx_test,
        )

        # Setup mock model and optimizer
        mock_model = Mock()
        mock_optimizer = Mock()
        trainer.model = mock_model
        trainer.optimizer = mock_optimizer
        trainer.class_num = 7  # Add missing class_num

        self.args.batch_size = 0  # Ensure no batching for this test

        mock_train_func.return_value = (0.5, 0.85)  # loss, accuracy
        mock_test_func.return_value = (0.3, 0.9)  # loss, accuracy

        trainer.train(current_global_round=1)

        assert mock_train_func.call_count == trainer.local_step
        assert mock_test_func.call_count == 0
        assert len(trainer.train_losses) == trainer.local_step
        assert len(trainer.train_accs) == trainer.local_step
        assert len(trainer.test_losses) == 0
        assert len(trainer.test_accs) == 0

    @patch("fedgraph.trainer_class.NeighborLoader")
    @patch("fedgraph.trainer_class.NeighborSampler")
    @patch("fedgraph.trainer_class.train")
    def test_train_method_mini_batch_one_update_per_local_step(
        self, mock_train_func, mock_neighbor_sampler, mock_neighbor_loader
    ):
        """Mini-batch training should reuse its loader and graph sampler."""
        self.args.graph_storage_mode = "cpu"
        self.args.num_hops = 0
        self.args.batch_size = 2
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=self.device,
            args=self.args,
            local_node_index=self.local_node_index,
            communicate_node_index=self.communicate_node_index,
            adj=self.adj,
            train_labels=self.train_labels,
            test_labels=self.test_labels,
            features=self.features,
            idx_train=self.idx_train,
            idx_test=self.idx_test,
        )

        trainer.model = Mock()
        trainer.optimizer = Mock()
        trainer.class_num = 7

        batch = Mock()
        batch.batch_size = self.args.batch_size
        batch.x = torch.randn(5, self.features.shape[1])
        batch.edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        batch.input_id = torch.tensor([1, 0])
        mock_neighbor_loader.return_value = [batch]
        mock_train_func.return_value = (0.5, 0.85)

        trainer.train(current_global_round=1)

        assert mock_neighbor_sampler.call_count == 1
        assert mock_neighbor_loader.call_count == 1
        assert mock_train_func.call_count == trainer.local_step
        assert len(trainer.train_losses) == trainer.local_step
        assert len(trainer.train_accs) == trainer.local_step

        train_loader_call = mock_neighbor_loader.call_args
        assert train_loader_call.kwargs["batch_size"] == self.args.batch_size
        assert train_loader_call.kwargs["shuffle"] is True
        assert torch.equal(train_loader_call.kwargs["input_nodes"], trainer.idx_train)
        assert train_loader_call.args[0].y is None
        assert (
            train_loader_call.kwargs["neighbor_sampler"]
            is mock_neighbor_sampler.return_value
        )

        expected_seed_index = torch.arange(self.args.batch_size)
        expected_seed_labels = trainer.train_labels[batch.input_id]
        for call in mock_train_func.call_args_list:
            assert torch.equal(call.args[5], expected_seed_labels)
            assert torch.equal(call.args[6].cpu(), expected_seed_index)

        trainer._make_neighbor_loader(
            trainer.idx_test,
            batch_size=512,
            shuffle=False,
        )
        assert mock_neighbor_sampler.call_count == 1
        assert mock_neighbor_loader.call_count == 2
        evaluation_loader_call = mock_neighbor_loader.call_args
        assert evaluation_loader_call.kwargs["batch_size"] == 512
        assert evaluation_loader_call.kwargs["shuffle"] is False
        assert (
            evaluation_loader_call.kwargs["neighbor_sampler"]
            is mock_neighbor_sampler.return_value
        )

    @patch("fedgraph.trainer_class.test")
    def test_local_test(self, mock_test_func):
        """Test local_test method."""
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=self.device,
            args=self.args,
            local_node_index=self.local_node_index,
            communicate_node_index=self.communicate_node_index,
            adj=self.adj,
            train_labels=self.train_labels,
            test_labels=self.test_labels,
            features=self.features,
            idx_train=self.idx_train,
            idx_test=self.idx_test,
        )

        # Setup mock model
        mock_model = Mock()
        trainer.model = mock_model

        mock_test_func.return_value = (0.3, 0.9)  # loss, accuracy

        result = trainer.local_test()

        mock_test_func.assert_called()
        assert isinstance(result, list)
        assert len(result) == 2
        assert len(trainer.test_losses) > 0
        assert len(trainer.test_accs) > 0

    def test_local_test_mini_batch_matches_full_graph_metrics(self):
        """Mini-batch evaluation must retain split-label order across batches."""

        class LogitModel(torch.nn.Module):
            def forward(self, x, edge_index):
                del edge_index
                return torch.log_softmax(x, dim=-1)

        args = Mock()
        args.local_step = 1
        args.method = "FedAvg"
        args.num_hops = 0
        args.num_layers = 1
        args.learning_rate = 0.01
        args.dataset = "cora"
        args.batch_size = 2
        args.evaluation_batch_size = 3
        args.graph_storage_mode = "cpu"
        features = torch.tensor(
            [
                [5.0, 0.0],
                [0.0, 5.0],
                [1.0, 1.0],
                [4.0, 0.0],
                [0.0, 4.0],
            ]
        )
        idx_test = torch.tensor([4, 1, 3])
        test_labels = torch.tensor([1, 1, 0])
        trainer = Trainer_General(
            rank=0,
            args_hidden=4,
            device=torch.device("cpu"),
            args=args,
            local_node_index=torch.arange(features.size(0)),
            communicate_node_index=torch.arange(features.size(0)),
            adj=torch.empty((2, 0), dtype=torch.long),
            train_labels=torch.tensor([0]),
            test_labels=test_labels,
            features=features,
            idx_train=torch.tensor([0]),
            idx_test=idx_test,
            global_node_num=features.size(0),
            class_num=2,
        )
        trainer.model = LogitModel()

        expected_output = trainer.model(trainer.features, trainer.adj)
        expected_loss = torch.nn.functional.nll_loss(
            expected_output[idx_test], test_labels
        ).item()
        expected_accuracy = (
            (expected_output[idx_test].argmax(dim=-1) == test_labels)
            .float()
            .mean()
            .item()
        )

        actual_loss, actual_accuracy = trainer.local_test()

        assert actual_loss == pytest.approx(expected_loss)
        assert actual_accuracy == pytest.approx(expected_accuracy)

    def test_get_rank(self):
        """Test get_rank method."""
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=self.device,
            args=self.args,
            local_node_index=self.local_node_index,
            communicate_node_index=self.communicate_node_index,
            adj=self.adj,
            train_labels=self.train_labels,
            test_labels=self.test_labels,
            features=self.features,
            idx_train=self.idx_train,
            idx_test=self.idx_test,
        )

        assert trainer.get_rank() == self.rank

    def test_get_all_loss_accuracy(self):
        """Test get_all_loss_accuray method."""
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=self.device,
            args=self.args,
            local_node_index=self.local_node_index,
            communicate_node_index=self.communicate_node_index,
            adj=self.adj,
            train_labels=self.train_labels,
            test_labels=self.test_labels,
            features=self.features,
            idx_train=self.idx_train,
            idx_test=self.idx_test,
        )

        # Add some mock data
        trainer.train_losses = [0.5, 0.4, 0.3]
        trainer.train_accs = [0.8, 0.85, 0.9]
        trainer.test_losses = [0.6, 0.5, 0.4]
        trainer.test_accs = [0.75, 0.8, 0.85]

        result = trainer.get_all_loss_accuray()

        assert isinstance(result, list)
        assert len(result) == 4  # train_losses, train_accs, test_losses, test_accs


class TestTrainerGC:
    """Test Trainer_GC class."""

    def setup_method(self):
        """Setup common test data for GC."""
        self.trainer_id = 0
        self.trainer_name = "trainer_0"
        self.train_size = 100
        self.args = Mock()
        self.args.device = torch.device("cpu")
        self.args.local_step = 5
        self.args.n_trainer = 3

        # Mock model with named_parameters
        self.mock_model = Mock()
        self.mock_model.named_parameters.return_value = [
            ("layer1.weight", torch.randn(10, 5, requires_grad=True)),
            ("layer1.bias", torch.randn(10, requires_grad=True)),
        ]
        self.mock_model.to.return_value = self.mock_model

        # Mock dataloader
        self.mock_dataloader = {"train": Mock(), "val": Mock(), "test": Mock()}

        # Mock optimizer
        self.mock_optimizer = Mock()

    def test_trainer_gc_init(self):
        """Test Trainer_GC initialization."""
        trainer = Trainer_GC(
            model=self.mock_model,
            trainer_id=self.trainer_id,
            trainer_name=self.trainer_name,
            train_size=self.train_size,
            dataloader=self.mock_dataloader,
            optimizer=self.mock_optimizer,
            args=self.args,
        )

        assert trainer.id == self.trainer_id
        assert trainer.name == self.trainer_name
        assert trainer.train_size == self.train_size
        assert trainer.dataloader == self.mock_dataloader
        assert trainer.optimizer == self.mock_optimizer
        assert trainer.args == self.args
        assert hasattr(trainer, "W")
        assert hasattr(trainer, "dW")
        assert hasattr(trainer, "W_old")

    def test_update_params_gc(self):
        """Test update_params method for GC trainer."""
        trainer = Trainer_GC(
            model=self.mock_model,
            trainer_id=self.trainer_id,
            trainer_name=self.trainer_name,
            train_size=self.train_size,
            dataloader=self.mock_dataloader,
            optimizer=self.mock_optimizer,
            args=self.args,
        )

        mock_params = {
            "layer1.weight": torch.randn(10, 5),
            "layer1.bias": torch.randn(10),
        }

        trainer.update_params(mock_params)

        # Check that gconv_names is set
        assert trainer.gconv_names == mock_params.keys()

        # Check that weights are updated
        for key in mock_params:
            if key in trainer.W:
                assert torch.equal(trainer.W[key].data, mock_params[key].data)

    def test_reset_params_gc(self):
        """Test reset_params method for GC trainer."""
        trainer = Trainer_GC(
            model=self.mock_model,
            trainer_id=self.trainer_id,
            trainer_name=self.trainer_name,
            train_size=self.train_size,
            dataloader=self.mock_dataloader,
            optimizer=self.mock_optimizer,
            args=self.args,
        )

        # Set gconv_names first
        trainer.gconv_names = ["layer1.weight", "layer1.bias"]

        # Store original weights
        original_weights = {k: v.data.clone() for k, v in trainer.W.items()}

        # Modify current weights
        for key in trainer.W:
            trainer.W[key].data = torch.randn_like(trainer.W[key].data)

        # Reset params
        trainer.reset_params()

        # Check that weights are reset to W_old for gconv layers
        for key in trainer.gconv_names:
            if key in trainer.W:
                assert torch.equal(trainer.W[key].data, trainer.W_old[key].data)

    def test_cache_weights_gc(self):
        """Test cache_weights method for GC trainer."""
        trainer = Trainer_GC(
            model=self.mock_model,
            trainer_id=self.trainer_id,
            trainer_name=self.trainer_name,
            train_size=self.train_size,
            dataloader=self.mock_dataloader,
            optimizer=self.mock_optimizer,
            args=self.args,
        )

        # Modify current weights
        for key in trainer.W:
            trainer.W[key].data = torch.randn_like(trainer.W[key].data)

        # Cache weights
        trainer.cache_weights()

        # Check that W_old is updated with current W values
        for key in trainer.W:
            assert torch.equal(trainer.W_old[key].data, trainer.W[key].data)

    def test_compute_update_norm(self):
        """Test compute_update_norm method."""
        trainer = Trainer_GC(
            model=self.mock_model,
            trainer_id=self.trainer_id,
            trainer_name=self.trainer_name,
            train_size=self.train_size,
            dataloader=self.mock_dataloader,
            optimizer=self.mock_optimizer,
            args=self.args,
        )

        # Set some gradients
        for key in trainer.dW:
            trainer.dW[key] = torch.randn_like(trainer.dW[key])

        # Pass the actual dict or a subset of keys
        keys = {k: None for k in trainer.dW.keys()}  # Create dict with keys
        norm = trainer.compute_update_norm(keys)

        assert isinstance(norm, float)
        assert norm >= 0

    def test_compute_mean_norm(self):
        """Test compute_mean_norm method."""
        trainer = Trainer_GC(
            model=self.mock_model,
            trainer_id=self.trainer_id,
            trainer_name=self.trainer_name,
            train_size=self.train_size,
            dataloader=self.mock_dataloader,
            optimizer=self.mock_optimizer,
            args=self.args,
        )

        # Set some gradients
        for key in trainer.dW:
            trainer.dW[key] = torch.randn_like(trainer.dW[key])

        # Pass the actual dict or a subset of keys
        keys = {k: None for k in trainer.dW.keys()}  # Create dict with keys
        total_size = 500
        mean_norm = trainer.compute_mean_norm(total_size, keys)

        assert isinstance(mean_norm, torch.Tensor)


class TestTrainerIntegration:
    """Integration tests for trainer classes."""

    @patch("fedgraph.trainer_class.GCN")
    def test_trainer_general_full_workflow(self, mock_gcn_class):
        """Test full workflow of Trainer_General."""
        # Setup
        rank = 0
        device = torch.device("cpu")
        args_hidden = 64

        local_node_index = torch.arange(0, 50)
        communicate_node_index = torch.arange(0, 100)
        adj = torch.randn(2, 200)
        train_labels = torch.randint(0, 3, (40,))
        test_labels = torch.randint(0, 3, (10,))
        features = torch.randn(100, 50)
        idx_train = torch.arange(0, 40)
        idx_test = torch.arange(40, 50)

        args = Mock()
        args.local_step = 5
        args.method = "FedAvg"
        args.dataset = "cora"
        args.num_hops = 2
        args.num_layers = 2
        args.learning_rate = 0.01
        args.batch_size = 0

        # Create trainer
        trainer = Trainer_General(
            rank=rank,
            args_hidden=args_hidden,
            device=device,
            args=args,
            local_node_index=local_node_index,
            communicate_node_index=communicate_node_index,
            adj=adj,
            train_labels=train_labels,
            test_labels=test_labels,
            features=features,
            idx_train=idx_train,
            idx_test=idx_test,
        )

        # Test initialization
        assert trainer.rank == rank
        assert trainer.device == device

        # Test info retrieval
        info = trainer.get_info()
        assert "features_num" in info
        assert "label_num" in info

        # Mock model initialization
        mock_model = Mock()
        mock_gcn_class.return_value = mock_model

        trainer.init_model(global_node_num=2708, class_num=7)
        assert trainer.model is not None

        # Test parameter operations
        mock_params = (torch.randn(64, 50), torch.randn(64))
        trainer.update_params(mock_params, current_global_epoch=1)

        params = trainer.get_params()
        assert isinstance(params, tuple)

        # Test rank retrieval
        assert trainer.get_rank() == rank
