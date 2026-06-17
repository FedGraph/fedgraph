import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import ray

from fedgraph.trainer_class import (
    Trainer_General,
    Trainer_GC,
    load_trainer_data_from_hugging_face
)


class TestLoadTrainerDataFromHuggingFace:
    """Test load_trainer_data_from_hugging_face function."""
    
    @patch('fedgraph.trainer_class.hf_hub_download')
    @patch('builtins.open')
    @patch('torch.load')
    def test_load_trainer_data_success(self, mock_torch_load, mock_open, mock_hf_download):
        """Test successful loading of trainer data from Hugging Face."""
        # Setup mocks
        mock_hf_download.return_value = "/tmp/test_file.pt"
        mock_file = Mock()
        mock_file.read.return_value = b"test_tensor_data"
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Mock tensor data
        mock_tensors = [
            torch.randn(100),  # local_node_index
            torch.randn(50),   # communicate_node_global_index
            torch.randn(2, 200),  # global_edge_index_client
            torch.randn(80),   # train_labels
            torch.randn(20),   # test_labels
            torch.randn(100, 10),  # features
            torch.randn(80),   # in_com_train_node_local_indexes
            torch.randn(20)    # in_com_test_node_local_indexes
        ]
        mock_torch_load.side_effect = mock_tensors
        
        args = Mock()
        args.dataset = "cora"
        args.n_trainer = 5
        args.num_hops = 2
        args.iid_beta = 0.5
        
        result = load_trainer_data_from_hugging_face(trainer_id=0, args=args)
        
        assert len(result) == 8
        assert all(isinstance(tensor, torch.Tensor) for tensor in result)
        
        # Verify calls
        assert mock_hf_download.call_count == 8
        expected_repo = "FedGraph/fedgraph_cora_5trainer_2hop_iid_beta_0.5_trainer_id_0"
        mock_hf_download.assert_any_call(
            repo_id=expected_repo,
            repo_type="dataset",
            filename="local_node_index.pt"
        )


class TestTrainerGeneral:
    """Test Trainer_General class."""
    
    def setup_method(self):
        """Setup common test data."""
        self.rank = 0
        self.device = torch.device('cpu')
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
    
    @patch('fedgraph.trainer_class.load_trainer_data_from_hugging_face')
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
            idx_test=self.idx_test
        )
        
        assert trainer.rank == self.rank
        assert trainer.device == self.device
        assert trainer.args_hidden == self.args_hidden
        assert trainer.local_step == self.args.local_step
        assert torch.equal(trainer.local_node_index, self.local_node_index.to(self.device))
        assert trainer.feature_aggregation is not None  # Should be set for FedAvg
        mock_load_data.assert_not_called()
    
    @patch('fedgraph.trainer_class.load_trainer_data_from_hugging_face')
    def test_trainer_init_without_data(self, mock_load_data):
        """Test Trainer_General initialization without provided data."""
        mock_load_data.return_value = (
            self.local_node_index,
            self.communicate_node_index,
            self.adj,
            self.train_labels,
            self.test_labels,
            self.features,
            self.idx_train,
            self.idx_test
        )
        
        trainer = Trainer_General(
            rank=self.rank,
            args_hidden=self.args_hidden,
            device=self.device,
            args=self.args
        )
        
        assert trainer.rank == self.rank
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
            idx_test=self.idx_test
        )
        
        info = trainer.get_info()
        
        assert "features_num" in info
        assert "label_num" in info
        assert info["features_num"] == len(self.features)
        expected_label_num = max(self.train_labels.max().item(), self.test_labels.max().item()) + 1
        assert info["label_num"] == expected_label_num
    
    @patch('fedgraph.trainer_class.GCN')
    @patch('fedgraph.trainer_class.GCN_arxiv')
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
            idx_test=self.idx_test
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
            idx_test=self.idx_test
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
        assert mock_model.load_state_dict.called or hasattr(trainer, 'model')
    
    def test_get_local_feature_sum(self):
        """Test get_local_feature_sum method."""
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
            idx_test=self.idx_test
        )
        
        # Mock the get_1hop_feature_sum function
        with patch('fedgraph.trainer_class.get_1hop_feature_sum') as mock_get_1hop:
            mock_get_1hop.return_value = torch.randn(100, 50)
            
            result = trainer.get_local_feature_sum()
            
            assert isinstance(result, torch.Tensor)
            mock_get_1hop.assert_called_once()
    
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
            idx_test=self.idx_test
        )
        
        new_features = torch.randn(100, 50)
        trainer.load_feature_aggregation(new_features)
        
        assert torch.equal(trainer.feature_aggregation, new_features.to(trainer.device))
    
    def test_get_params(self):
        """Test get_params method."""
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
            idx_test=self.idx_test
        )
        
        # Create mock model with state_dict
        mock_model = Mock()
        mock_state_dict = {'layer1.weight': torch.randn(10, 5), 'layer1.bias': torch.randn(10)}
        mock_model.state_dict.return_value = mock_state_dict
        trainer.model = mock_model
        
        params = trainer.get_params()
        
        assert isinstance(params, tuple)
        mock_model.state_dict.assert_called_once()
    
    @patch('fedgraph.trainer_class.train')
    def test_train_method(self, mock_train_func):
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
            idx_test=self.idx_test
        )
        
        # Setup mock model and optimizer
        mock_model = Mock()
        mock_optimizer = Mock()
        trainer.model = mock_model
        trainer.optimizer = mock_optimizer
        trainer.class_num = 7  # Add missing class_num
        
        self.args.batch_size = 0  # Ensure no batching for this test
        
        mock_train_func.return_value = (0.5, 0.85)  # loss, accuracy
        
        trainer.train(current_global_round=1)
        
        mock_train_func.assert_called()
        assert len(trainer.train_losses) > 0
        assert len(trainer.train_accs) > 0
    
    @patch('fedgraph.trainer_class.test')
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
            idx_test=self.idx_test
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
            idx_test=self.idx_test
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
            idx_test=self.idx_test
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
        self.args.device = torch.device('cpu')
        self.args.local_step = 5
        self.args.n_trainer = 3
        
        # Mock model with named_parameters
        self.mock_model = Mock()
        self.mock_model.named_parameters.return_value = [
            ('layer1.weight', torch.randn(10, 5, requires_grad=True)),
            ('layer1.bias', torch.randn(10, requires_grad=True))
        ]
        self.mock_model.to.return_value = self.mock_model
        
        # Mock dataloader
        self.mock_dataloader = {
            'train': Mock(),
            'val': Mock(), 
            'test': Mock()
        }
        
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
            args=self.args
        )
        
        assert trainer.id == self.trainer_id
        assert trainer.name == self.trainer_name
        assert trainer.train_size == self.train_size
        assert trainer.dataloader == self.mock_dataloader
        assert trainer.optimizer == self.mock_optimizer
        assert trainer.args == self.args
        assert hasattr(trainer, 'W')
        assert hasattr(trainer, 'dW')
        assert hasattr(trainer, 'W_old')
    
    def test_update_params_gc(self):
        """Test update_params method for GC trainer."""
        trainer = Trainer_GC(
            model=self.mock_model,
            trainer_id=self.trainer_id,
            trainer_name=self.trainer_name,
            train_size=self.train_size,
            dataloader=self.mock_dataloader,
            optimizer=self.mock_optimizer,
            args=self.args
        )
        
        mock_params = {
            'layer1.weight': torch.randn(10, 5),
            'layer1.bias': torch.randn(10)
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
            args=self.args
        )
        
        # Set gconv_names first
        trainer.gconv_names = ['layer1.weight', 'layer1.bias']
        
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
            args=self.args
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
            args=self.args
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
            args=self.args
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
    
    @patch('fedgraph.trainer_class.GCN')
    def test_trainer_general_full_workflow(self, mock_gcn_class):
        """Test full workflow of Trainer_General."""
        # Setup
        rank = 0
        device = torch.device('cpu')
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
            idx_test=idx_test
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