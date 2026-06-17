import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import attridict

from fedgraph.federated_methods import (
    run_fedgraph,
    run_fedgraph_enhanced,
    run_NC,
    run_GC,
    run_LP
)


class TestRunFedgraph:
    """Test run_fedgraph main orchestration function."""
    
    def setup_method(self):
        """Setup test data for NC tests."""
        self.args = attridict.AttriDict()
        self.args.fedgraph_task = "NC"
        self.args.use_lowrank = False
        self.args.method = "FedAvg"
        self.args.use_encryption = False
        self.args.use_huggingface = False
    
    @patch('fedgraph.federated_methods.data_loader')
    @patch('fedgraph.federated_methods.run_NC')
    def test_run_fedgraph_nc_task(self, mock_run_nc, mock_data_loader):
        """Test run_fedgraph with NC task."""
        mock_data = Mock()
        mock_data_loader.return_value = mock_data
        
        run_fedgraph(self.args)
        
        mock_data_loader.assert_called_once_with(self.args)
        mock_run_nc.assert_called_once_with(self.args, mock_data)
    
    @patch('fedgraph.federated_methods.data_loader')
    @patch('fedgraph.federated_methods.run_GC')
    def test_run_fedgraph_gc_task(self, mock_run_gc, mock_data_loader):
        """Test run_fedgraph with GC task."""
        self.args.fedgraph_task = "GC"
        mock_data = Mock()
        mock_data_loader.return_value = mock_data
        
        run_fedgraph(self.args)
        
        mock_data_loader.assert_called_once_with(self.args)
        mock_run_gc.assert_called_once_with(self.args, mock_data)
    
    @patch('fedgraph.federated_methods.run_LP')
    def test_run_fedgraph_lp_task(self, mock_run_lp):
        """Test run_fedgraph with LP task."""
        self.args.fedgraph_task = "LP"
        
        run_fedgraph(self.args)
        
        mock_run_lp.assert_called_once_with(self.args)
    
    @patch('fedgraph.federated_methods.run_NC_lowrank')
    def test_run_fedgraph_nc_lowrank(self, mock_run_nc_lowrank):
        """Test run_fedgraph with NC task and low-rank compression."""
        self.args.use_lowrank = True
        
        with patch('fedgraph.federated_methods.data_loader') as mock_data_loader:
            mock_data = Mock()
            mock_data_loader.return_value = mock_data
            
            run_fedgraph(self.args)
            
            mock_run_nc_lowrank.assert_called_once_with(self.args, mock_data)
    
    def test_run_fedgraph_lowrank_validation_nc_only(self):
        """Test that low-rank compression only works with NC tasks."""
        self.args.use_lowrank = True
        self.args.fedgraph_task = "GC"
        
        with pytest.raises(ValueError, match="Low-rank compression currently only supported for NC tasks"):
            run_fedgraph(self.args)
    
    def test_run_fedgraph_lowrank_validation_fedavg_only(self):
        """Test that low-rank compression only works with FedAvg method."""
        self.args.use_lowrank = True
        self.args.method = "FedProx"
        
        with pytest.raises(ValueError, match="Low-rank compression currently only supported for FedAvg method"):
            run_fedgraph(self.args)
    
    def test_run_fedgraph_lowrank_encryption_conflict(self):
        """Test that low-rank and encryption cannot be used together."""
        self.args.use_lowrank = True
        self.args.use_encryption = True
        
        with pytest.raises(ValueError, match="Cannot use both encryption and low-rank compression simultaneously"):
            run_fedgraph(self.args)
    
    @patch('fedgraph.federated_methods.data_loader')
    @patch('fedgraph.federated_methods.run_NC')
    def test_run_fedgraph_huggingface_nc(self, mock_run_nc, mock_data_loader):
        """Test run_fedgraph with NC task using Hugging Face."""
        self.args.use_huggingface = True
        
        run_fedgraph(self.args)
        
        mock_data_loader.assert_not_called()  # Should not load data when using HF
        mock_run_nc.assert_called_once_with(self.args, None)


class TestRunFedgraphEnhanced:
    """Test run_fedgraph_enhanced function."""
    
    @patch('fedgraph.federated_methods.run_fedgraph')
    def test_run_fedgraph_enhanced_calls_main(self, mock_run_fedgraph):
        """Test that run_fedgraph_enhanced calls the main function."""
        args = attridict.AttriDict()
        args.fedgraph_task = "NC"
        
        run_fedgraph_enhanced(args)
        
        mock_run_fedgraph.assert_called_once_with(args)


class TestRunNC:
    """Test run_NC function for node classification."""
    
    def setup_method(self):
        """Setup test data for each test method."""
        self.args = attridict.AttriDict()
        self.args.n_trainer = 3
        self.args.device = "cpu"
        self.args.local_step = 5
        self.args.global_epochs = 10
        self.args.method = "FedAvg"
        self.args.monitor = True
        self.args.dataset = "cora"
        self.args.seed = 42
        self.args.use_ray = True
        self.args.he = False
        self.args.dp = False
        
        # Mock data
        self.mock_data = (
            torch.randn(2, 100),  # edge_index
            torch.randn(100, 50), # features
            torch.randint(0, 7, (100,)), # labels
            torch.arange(0, 70),  # idx_train
            torch.arange(85, 100), # idx_test
            7,  # class_num
            [torch.arange(i*10, (i+1)*10) for i in range(3)], # split_node_indexes
            {i: torch.arange(i*20, (i+1)*20) for i in range(3)}, # communicate_node_global_indexes
            {i: torch.arange(0, 5) for i in range(3)}, # in_com_train_node_local_indexes
            {i: torch.arange(5, 8) for i in range(3)}, # in_com_test_node_local_indexes
            {i: torch.randn(2, 20) for i in range(3)}  # global_edge_indexes_clients
        )
    
    @patch('fedgraph.federated_methods.ray')
    @patch('fedgraph.federated_methods.Monitor')
    def test_run_nc_basic_setup(self, mock_monitor, mock_ray):
        """Test basic setup of run_NC function."""
        mock_monitor_instance = Mock()
        mock_monitor.return_value = mock_monitor_instance
        
        # Mock Ray
        mock_ray.init = Mock()
        mock_ray.get = Mock()
        mock_ray.remote = Mock()
        
        with patch('fedgraph.federated_methods.Server') as mock_server_class, \
             patch('fedgraph.federated_methods.torch.manual_seed'):
            
            mock_server = Mock()
            mock_server_class.return_value = mock_server
            
            # Mock remote trainer creation
            mock_trainer_class = Mock()
            mock_ray.remote.return_value = mock_trainer_class
            
            try:
                run_NC(self.args, self.mock_data)
            except Exception:
                # Expected to fail due to complex initialization, 
                # but we verify the setup calls
                pass
            
            # Verify initialization calls
            mock_ray.init.assert_called()
            if self.args.monitor:
                mock_monitor.assert_called_once()
    
    @patch('fedgraph.federated_methods.ray')
    def test_run_nc_without_ray(self, mock_ray):
        """Test run_NC without Ray distributed computing."""
        self.args.use_ray = False
        mock_ray.init = Mock()
        
        with patch('fedgraph.federated_methods.Server') as mock_server_class, \
             patch('fedgraph.federated_methods.Trainer_General') as mock_trainer_class, \
             patch('fedgraph.federated_methods.torch.manual_seed'):
            
            mock_server = Mock()
            mock_server_class.return_value = mock_server
            mock_server.trainers = []
            
            mock_trainer = Mock()
            mock_trainer_class.return_value = mock_trainer
            
            try:
                run_NC(self.args, self.mock_data)
            except Exception:
                # Expected to fail due to complex flow, but verify no Ray init
                pass
            
            mock_ray.init.assert_not_called()


class TestRunGC:
    """Test run_GC function for graph classification."""
    
    def setup_method(self):
        """Setup test data for GC."""
        self.args = attridict.AttriDict()
        self.args.dataset = "PROTEINS"
        self.args.method = "FedAvg"
        self.args.federated_method = "FedAvg"
        self.args.num_rounds = 10
        self.args.local_step = 5
        self.args.device = "cpu"
        self.args.monitor = True
        self.args.seed = 42
        
        # Mock data for GC
        self.mock_data = {
            "0-PROTEINS": (
                {"train": Mock(), "val": Mock(), "test": Mock()},
                10,  # num_node_features
                2,   # num_graph_labels
                100  # train_size
            ),
            "1-PROTEINS": (
                {"train": Mock(), "val": Mock(), "test": Mock()},
                10,  # num_node_features
                2,   # num_graph_labels
                100  # train_size
            )
        }
    
    @patch('fedgraph.federated_methods.setup_server')
    @patch('fedgraph.federated_methods.setup_trainers')
    @patch('fedgraph.federated_methods.Monitor')
    def test_run_gc_basic_setup(self, mock_monitor, mock_setup_trainers, mock_setup_server):
        """Test basic setup of run_GC function."""
        mock_monitor_instance = Mock()
        mock_monitor.return_value = mock_monitor_instance
        
        mock_server = Mock()
        mock_setup_server.return_value = mock_server
        
        mock_trainers = [Mock(), Mock()]
        mock_setup_trainers.return_value = mock_trainers
        
        with patch('fedgraph.federated_methods.run_GC_Fed_algorithm') as mock_run_fed:
            run_GC(self.args, self.mock_data)
            
            mock_setup_server.assert_called_once()
            mock_setup_trainers.assert_called_once()
            mock_run_fed.assert_called_once()
    
    @patch('fedgraph.federated_methods.run_GC_selftrain')
    def test_run_gc_selftrain(self, mock_run_selftrain):
        """Test run_GC with SelfTrain method."""
        self.args.method = "SelfTrain"
        
        run_GC(self.args, self.mock_data)
        
        mock_run_selftrain.assert_called_once_with(self.args, self.mock_data)
    
    @patch('fedgraph.federated_methods.run_GCFL_algorithm')
    def test_run_gc_gcfl(self, mock_run_gcfl):
        """Test run_GC with GCFL method."""
        self.args.method = "GCFL"
        
        with patch('fedgraph.federated_methods.setup_server') as mock_setup_server, \
             patch('fedgraph.federated_methods.setup_trainers') as mock_setup_trainers:
            
            mock_setup_server.return_value = Mock()
            mock_setup_trainers.return_value = [Mock(), Mock()]
            
            run_GC(self.args, self.mock_data)
            
            mock_run_gcfl.assert_called_once()


class TestRunLP:
    """Test run_LP function for link prediction."""
    
    def setup_method(self):
        """Setup test data for LP."""
        self.args = Mock()
        self.args.dataset = "ml-1m"
        self.args.data_path = "/tmp/data"
        self.args.num_trainer = 3
        self.args.device = "cpu"
        self.args.monitor = True
        self.args.seed = 42
    
    @patch('fedgraph.federated_methods.check_data_files_existance')
    @patch('fedgraph.federated_methods.get_global_user_item_mapping')
    @patch('fedgraph.federated_methods.get_start_end_time')
    def test_run_lp_basic_setup(self, mock_get_time, mock_get_mapping, mock_check_files):
        """Test basic setup of run_LP function."""
        mock_check_files.return_value = True
        mock_get_mapping.return_value = (Mock(), Mock())
        mock_get_time.return_value = (0, 100)
        
        with patch('fedgraph.federated_methods.Server_LP') as mock_server_class, \
             patch('fedgraph.federated_methods.Monitor') as mock_monitor, \
             patch('fedgraph.federated_methods.ray'):
            
            mock_server = Mock()
            mock_server_class.return_value = mock_server
            mock_monitor_instance = Mock()
            mock_monitor.return_value = mock_monitor_instance
            
            try:
                run_LP(self.args)
            except Exception:
                # Expected to fail due to complex initialization
                pass
            
            # Verify setup calls
            mock_check_files.assert_called_once()
            mock_get_mapping.assert_called_once()
            mock_get_time.assert_called_once()
    
    @patch('fedgraph.federated_methods.check_data_files_existance')
    def test_run_lp_missing_data_files(self, mock_check_files):
        """Test run_LP when data files are missing."""
        mock_check_files.return_value = False
        
        with pytest.raises(FileNotFoundError):
            run_LP(self.args)


class TestFederatedMethodsIntegration:
    """Integration tests for federated methods."""
    
    @patch('fedgraph.federated_methods.data_loader')
    @patch('fedgraph.federated_methods.run_NC')
    @patch('fedgraph.federated_methods.run_GC')
    @patch('fedgraph.federated_methods.run_LP')
    def test_task_routing(self, mock_run_lp, mock_run_gc, mock_run_nc, mock_data_loader):
        """Test that run_fedgraph routes to correct task functions."""
        mock_data_loader.return_value = Mock()
        
        # Test NC routing
        args_nc = attridict.AttriDict()
        args_nc.fedgraph_task = "NC"
        args_nc.use_huggingface = False
        args_nc.use_lowrank = False
        run_fedgraph(args_nc)
        mock_run_nc.assert_called_once()
        
        # Test GC routing
        args_gc = attridict.AttriDict()
        args_gc.fedgraph_task = "GC"
        args_gc.use_huggingface = False
        run_fedgraph(args_gc)
        mock_run_gc.assert_called_once()
        
        # Test LP routing
        args_lp = attridict.AttriDict()
        args_lp.fedgraph_task = "LP"
        run_fedgraph(args_lp)
        mock_run_lp.assert_called_once()
    
    def test_argument_validation_comprehensive(self):
        """Test comprehensive argument validation."""
        # Test invalid task
        args = attridict.AttriDict()
        args.fedgraph_task = "INVALID"
        
        with patch('fedgraph.federated_methods.data_loader'):
            with pytest.raises((KeyError, AttributeError)):
                run_fedgraph(args)
        
        # Test low-rank with invalid method
        args = attridict.AttriDict()
        args.fedgraph_task = "NC"
        args.use_lowrank = True
        args.method = "GCFL"
        args.use_encryption = False
        
        with pytest.raises(ValueError):
            run_fedgraph(args)
    
    @patch('fedgraph.federated_methods.data_loader')
    def test_data_loading_logic(self, mock_data_loader):
        """Test data loading logic in different scenarios."""
        mock_data = Mock()
        mock_data_loader.return_value = mock_data
        
        with patch('fedgraph.federated_methods.run_NC') as mock_run_nc:
            # Test normal data loading
            args = attridict.AttriDict()
            args.fedgraph_task = "NC"
            args.use_huggingface = False
            args.use_lowrank = False
            
            run_fedgraph(args)
            
            mock_data_loader.assert_called_once_with(args)
            mock_run_nc.assert_called_once_with(args, mock_data)
        
        # Reset mocks
        mock_data_loader.reset_mock()
        
        with patch('fedgraph.federated_methods.run_NC') as mock_run_nc:
            # Test Hugging Face data loading (should skip data_loader)
            args = attridict.AttriDict()
            args.fedgraph_task = "NC"
            args.use_huggingface = True
            args.use_lowrank = False
            
            run_fedgraph(args)
            
            mock_data_loader.assert_not_called()
            mock_run_nc.assert_called_once_with(args, None)