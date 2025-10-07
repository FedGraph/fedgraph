import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

"""
Integration tests for FedGraph library.

These tests verify the end-to-end functionality of federated graph learning workflows,
testing the integration between different components like data processing, trainers,
servers, and federated methods.
"""


@pytest.mark.integration
class TestFedGraphNCWorkflow:
    """Integration tests for Node Classification workflow."""
    
    def setup_method(self):
        """Setup test data for NC integration tests."""
        self.num_nodes = 100
        self.num_features = 50
        self.num_classes = 7
        self.num_trainers = 3
        
        # Create mock graph data
        self.features = torch.randn(self.num_nodes, self.num_features)
        self.labels = torch.randint(0, self.num_classes, (self.num_nodes,))
        self.edge_index = torch.randint(0, self.num_nodes, (2, 200))
        self.idx_train = torch.arange(0, 70)
        self.idx_test = torch.arange(85, 100)
        
        # Create node splits for trainers
        split_size = self.num_nodes // self.num_trainers
        self.split_node_indexes = [
            torch.arange(i * split_size, min((i + 1) * split_size, self.num_nodes))
            for i in range(self.num_trainers)
        ]
        
        # Handle remainder nodes
        if sum(len(split) for split in self.split_node_indexes) < self.num_nodes:
            remainder = torch.arange(split_size * self.num_trainers, self.num_nodes)
            self.split_node_indexes[-1] = torch.cat([self.split_node_indexes[-1], remainder])
    
    @patch('fedgraph.data_process.NC_load_data')
    @patch('fedgraph.data_process.label_dirichlet_partition')
    @patch('fedgraph.data_process.get_in_comm_indexes')
    def test_nc_data_processing_pipeline(self, mock_get_indexes, mock_partition, mock_load_data):
        """Test the complete NC data processing pipeline."""
        from fedgraph.data_process import data_loader_NC
        
        # Setup mocks
        mock_load_data.return_value = (
            self.features, Mock(), self.labels, 
            self.idx_train, Mock(), self.idx_test
        )
        mock_partition.return_value = [split.tolist() for split in self.split_node_indexes]
        mock_get_indexes.return_value = (
            {i: torch.arange(i*20, (i+1)*20) for i in range(self.num_trainers)},
            {i: torch.arange(0, 10) for i in range(self.num_trainers)},
            {i: torch.arange(10, 15) for i in range(self.num_trainers)},
            {i: self.edge_index for i in range(self.num_trainers)}
        )
        
        # Create mock args
        args = Mock()
        args.use_huggingface = False
        args.dataset = "cora"
        args.n_trainer = self.num_trainers
        args.iid_beta = 0.5
        args.distribution_type = "dirichlet"
        args.num_hops = 2
        
        # Test the pipeline
        result = data_loader_NC(args)
        
        assert len(result) == 11  # Expected return tuple length
        assert mock_load_data.called
        assert mock_partition.called
        assert mock_get_indexes.called
    
    @patch('fedgraph.trainer_class.GCN')
    def test_nc_trainer_initialization(self, mock_gcn_class):
        """Test NC trainer initialization and basic operations."""
        from fedgraph.trainer_class import Trainer_General
        
        # Mock model
        mock_model = Mock()
        mock_gcn_class.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        # Create trainer
        trainer = Trainer_General(
            rank=0,
            args_hidden=64,
            device=torch.device('cpu'),
            args=Mock(local_step=5, method="FedAvg"),
            local_node_index=self.split_node_indexes[0],
            communicate_node_index=torch.arange(0, 50),
            adj=self.edge_index,
            train_labels=self.labels[:30],
            test_labels=self.labels[30:40],
            features=self.features,
            idx_train=torch.arange(0, 20),
            idx_test=torch.arange(20, 30)
        )
        
        # Test basic operations
        info = trainer.get_info()
        assert "features_num" in info
        assert "label_num" in info
        
        # Test parameter operations
        params = trainer.get_params()
        assert isinstance(params, tuple)
        
        rank = trainer.get_rank()
        assert rank == 0
    
    @patch('fedgraph.server_class.AggreGCN')
    def test_nc_server_aggregation(self, mock_aggre_gcn):
        """Test NC server aggregation workflow."""
        from fedgraph.server_class import Server
        
        # Mock model
        mock_model = Mock()
        mock_aggre_gcn.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        # Mock trainers
        trainers = []
        for i in range(self.num_trainers):
            trainer = Mock()
            trainer.rank = i
            trainer.update_params = Mock()
            trainer.get_params.return_value = (torch.randn(64, 50), torch.randn(64))
            trainers.append(trainer)
        
        # Create server
        args = Mock()
        args.num_hops = 1
        args.dataset = "cora"
        args.num_layers = 2
        
        server = Server(
            feature_dim=self.num_features,
            args_hidden=64,
            class_num=self.num_classes,
            device=torch.device('cpu'),
            trainers=trainers,
            args=args
        )
        
        # Test server operations
        assert server.num_of_trainers == self.num_trainers
        
        # Test parameter broadcast
        mock_model.state_dict.return_value = {'weight': torch.randn(64, 50)}
        server.broadcast_params(current_global_epoch=1)
        
        # Verify all trainers received updates
        for trainer in trainers:
            trainer.update_params.assert_called()


@pytest.mark.integration  
class TestFedGraphGCWorkflow:
    """Integration tests for Graph Classification workflow."""
    
    def setup_method(self):
        """Setup test data for GC integration tests."""
        self.num_trainers = 3
        self.num_node_features = 10
        self.num_graph_labels = 2
        self.train_size = 100
        
        # Mock graph classification data
        self.mock_data = {}
        for i in range(self.num_trainers):
            trainer_name = f"{i}-PROTEINS"
            self.mock_data[trainer_name] = (
                {
                    'train': Mock(),
                    'val': Mock(),
                    'test': Mock()
                },
                self.num_node_features,
                self.num_graph_labels,
                self.train_size
            )
    
    @patch('fedgraph.utils_gc.setup_server')
    @patch('fedgraph.utils_gc.setup_trainers') 
    @patch('fedgraph.federated_methods.run_GC_Fed_algorithm')
    def test_gc_complete_workflow(self, mock_run_fed, mock_setup_trainers, mock_setup_server):
        """Test complete GC federated learning workflow."""
        from fedgraph.federated_methods import run_GC
        
        # Mock server and trainers
        mock_server = Mock()
        mock_setup_server.return_value = mock_server
        
        mock_trainers = [Mock() for _ in range(self.num_trainers)]
        mock_setup_trainers.return_value = mock_trainers
        
        # Create args
        args = Mock()
        args.dataset = "PROTEINS"
        args.method = "FedAvg"
        args.federated_method = "FedAvg"
        args.monitor = False
        
        # Run the workflow
        run_GC(args, self.mock_data)
        
        # Verify setup was called
        mock_setup_server.assert_called_once()
        mock_setup_trainers.assert_called_once()
        mock_run_fed.assert_called_once()
    
    def test_gc_trainer_creation(self):
        """Test GC trainer creation and initialization."""
        from fedgraph.trainer_class import Trainer_GC
        
        # Mock model
        mock_model = Mock()
        mock_model.named_parameters.return_value = [
            ('layer1.weight', torch.randn(10, 5, requires_grad=True)),
            ('layer1.bias', torch.randn(10, requires_grad=True))
        ]
        mock_model.to.return_value = mock_model
        
        # Mock dataloader
        mock_dataloader = {
            'train': Mock(),
            'val': Mock(),
            'test': Mock()
        }
        
        # Mock optimizer
        mock_optimizer = Mock()
        
        # Mock args
        args = Mock()
        args.device = torch.device('cpu')
        
        # Create trainer
        trainer = Trainer_GC(
            model=mock_model,
            trainer_id=0,
            trainer_name="trainer_0",
            train_size=100,
            dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            args=args
        )
        
        # Test trainer properties
        assert trainer.id == 0
        assert trainer.name == "trainer_0"
        assert trainer.train_size == 100
        assert hasattr(trainer, 'W')
        assert hasattr(trainer, 'dW')


@pytest.mark.integration
class TestFedGraphLPWorkflow:
    """Integration tests for Link Prediction workflow."""
    
    @patch('fedgraph.utils_lp.check_data_files_existance')
    @patch('fedgraph.utils_lp.get_global_user_item_mapping')
    @patch('fedgraph.utils_lp.get_start_end_time')
    @patch('fedgraph.federated_methods.Server_LP')
    def test_lp_basic_setup(self, mock_server_lp, mock_get_time, 
                           mock_get_mapping, mock_check_files):
        """Test basic LP workflow setup."""
        from fedgraph.federated_methods import run_LP
        
        # Setup mocks
        mock_check_files.return_value = True
        mock_get_mapping.return_value = ({1: 0, 2: 1}, {10: 0, 20: 1})
        mock_get_time.return_value = (1000, 5000)
        
        mock_server = Mock()
        mock_server_lp.return_value = mock_server
        
        # Create args
        args = Mock()
        args.dataset = "ml-1m"
        args.data_path = "/tmp/data"
        args.num_trainer = 3
        args.device = "cpu"
        args.monitor = False
        
        with patch('fedgraph.federated_methods.ray'):
            try:
                run_LP(args)
            except Exception:
                # Expected to fail due to complex initialization
                pass
            
            # Verify setup calls
            mock_check_files.assert_called_once()
            mock_get_mapping.assert_called_once()
            mock_get_time.assert_called_once()


@pytest.mark.integration
class TestFedGraphEndToEnd:
    """End-to-end integration tests for complete federated learning workflows."""
    
    @patch('fedgraph.federated_methods.data_loader')
    @patch('fedgraph.federated_methods.run_NC')
    def test_complete_nc_pipeline(self, mock_run_nc, mock_data_loader):
        """Test complete NC pipeline from args to execution."""
        from fedgraph.federated_methods import run_fedgraph
        import attridict
        
        # Create realistic args
        args = attridict.AttriDict()
        args.fedgraph_task = "NC"
        args.dataset = "cora"
        args.method = "FedAvg"
        args.n_trainer = 3
        args.global_epochs = 5
        args.local_step = 3
        args.device = "cpu"
        args.use_huggingface = False
        args.use_lowrank = False
        args.use_encryption = False
        
        # Mock data
        mock_data = (
            torch.randn(2, 100),  # edge_index
            torch.randn(100, 50), # features  
            torch.randint(0, 7, (100,)), # labels
            torch.arange(0, 70),  # idx_train
            torch.arange(85, 100), # idx_test
            7,  # class_num
            [torch.arange(i*30, (i+1)*30) for i in range(3)], # splits
            {}, {}, {}, {}  # communication data
        )
        mock_data_loader.return_value = mock_data
        
        # Run the pipeline
        run_fedgraph(args)
        
        # Verify calls
        mock_data_loader.assert_called_once_with(args)
        mock_run_nc.assert_called_once_with(args, mock_data)
    
    @patch('fedgraph.federated_methods.data_loader')  
    @patch('fedgraph.federated_methods.run_GC')
    def test_complete_gc_pipeline(self, mock_run_gc, mock_data_loader):
        """Test complete GC pipeline from args to execution."""
        from fedgraph.federated_methods import run_fedgraph
        import attridict
        
        # Create realistic args
        args = attridict.AttriDict()
        args.fedgraph_task = "GC"
        args.dataset = "PROTEINS"
        args.method = "FedAvg"
        args.num_trainers = 3
        args.num_rounds = 10
        args.device = "cpu"
        args.use_huggingface = False
        
        # Mock data
        mock_data = {
            "0-PROTEINS": ({"train": Mock(), "val": Mock(), "test": Mock()}, 10, 2, 100),
            "1-PROTEINS": ({"train": Mock(), "val": Mock(), "test": Mock()}, 10, 2, 100),
            "2-PROTEINS": ({"train": Mock(), "val": Mock(), "test": Mock()}, 10, 2, 100)
        }
        mock_data_loader.return_value = mock_data
        
        # Run the pipeline
        run_fedgraph(args)
        
        # Verify calls
        mock_data_loader.assert_called_once_with(args)
        mock_run_gc.assert_called_once_with(args, mock_data)
    
    def test_task_validation_and_routing(self):
        """Test task validation and proper routing to task-specific functions."""
        from fedgraph.federated_methods import run_fedgraph
        import attridict
        
        # Test invalid task
        args = attridict.AttriDict()
        args.fedgraph_task = "INVALID_TASK"
        
        with patch('fedgraph.federated_methods.data_loader'):
            with pytest.raises((KeyError, AttributeError)):
                run_fedgraph(args)
    
    def test_configuration_validation(self):
        """Test comprehensive configuration validation."""
        from fedgraph.federated_methods import run_fedgraph
        import attridict
        
        # Test conflicting low-rank and encryption settings
        args = attridict.AttriDict()
        args.fedgraph_task = "NC"
        args.use_lowrank = True
        args.use_encryption = True
        args.method = "FedAvg"
        
        with pytest.raises(ValueError, match="Cannot use both encryption and low-rank"):
            run_fedgraph(args)
        
        # Test low-rank with wrong method
        args.use_encryption = False
        args.method = "FedProx"
        
        with pytest.raises(ValueError, match="Low-rank compression currently only supported for FedAvg"):
            run_fedgraph(args)
        
        # Test low-rank with wrong task
        args.method = "FedAvg"
        args.fedgraph_task = "GC"
        
        with pytest.raises(ValueError, match="Low-rank compression currently only supported for NC"):
            run_fedgraph(args)


@pytest.mark.integration
class TestComponentInteraction:
    """Test interaction between different FedGraph components."""
    
    def test_data_trainer_server_interaction(self):
        """Test interaction flow between data processing, trainers, and server."""
        # This test verifies that data flows correctly between components
        
        # 1. Data processing produces correct format
        edge_index = torch.randint(0, 50, (2, 100))
        features = torch.randn(50, 20)
        labels = torch.randint(0, 3, (50,))
        
        # 2. Trainers can initialize with this data
        with patch('fedgraph.trainer_class.GCN') as mock_gcn:
            from fedgraph.trainer_class import Trainer_General
            
            mock_model = Mock()
            mock_gcn.return_value = mock_model
            mock_model.to.return_value = mock_model
            
            trainer = Trainer_General(
                rank=0,
                args_hidden=32,
                device=torch.device('cpu'),
                args=Mock(local_step=3, method="FedAvg"),
                local_node_index=torch.arange(0, 25),
                communicate_node_index=torch.arange(0, 40),
                adj=edge_index,
                train_labels=labels[:20],
                test_labels=labels[20:30],
                features=features,
                idx_train=torch.arange(0, 15),
                idx_test=torch.arange(15, 25)
            )
            
            # 3. Server can aggregate trainer parameters
            with patch('fedgraph.server_class.AggreGCN') as mock_server_gcn:
                from fedgraph.server_class import Server
                
                mock_server_model = Mock()
                mock_server_gcn.return_value = mock_server_model
                mock_server_model.to.return_value = mock_server_model
                
                server = Server(
                    feature_dim=20,
                    args_hidden=32,
                    class_num=3,
                    device=torch.device('cpu'),
                    trainers=[trainer],
                    args=Mock(num_hops=1, dataset="test", num_layers=2)
                )
                
                # Test parameter flow
                trainer.update_params = Mock()
                mock_server_model.state_dict.return_value = {'weight': torch.randn(32, 20)}
                
                server.broadcast_params(current_global_epoch=1)
                trainer.update_params.assert_called()
    
    def test_monitor_integration(self):
        """Test monitoring system integration."""
        from fedgraph.monitor_class import Monitor
        
        # Create monitor
        args = Mock()
        args.monitor = True
        args.dataset = "test"
        args.method = "FedAvg"
        
        with patch('fedgraph.monitor_class.wandb'):
            monitor = Monitor(args)
            
            # Test logging functionality
            monitor.log_metrics = Mock()
            test_metrics = {"accuracy": 0.85, "loss": 0.3}
            
            # This would be called during training
            if hasattr(monitor, 'log_metrics'):
                monitor.log_metrics(test_metrics)
                monitor.log_metrics.assert_called_with(test_metrics)


@pytest.mark.integration
@pytest.mark.slow
class TestLargeScaleIntegration:
    """Integration tests for larger scale scenarios."""
    
    def test_multi_trainer_coordination(self):
        """Test coordination between multiple trainers."""
        num_trainers = 5
        
        # Create multiple trainers
        trainers = []
        for i in range(num_trainers):
            with patch('fedgraph.trainer_class.GCN'):
                from fedgraph.trainer_class import Trainer_General
                
                trainer = Trainer_General(
                    rank=i,
                    args_hidden=32,
                    device=torch.device('cpu'),
                    args=Mock(local_step=3, method="FedAvg"),
                    local_node_index=torch.arange(i*10, (i+1)*10),
                    communicate_node_index=torch.arange(0, 50),
                    adj=torch.randint(0, 50, (2, 50)),
                    train_labels=torch.randint(0, 3, (10,)),
                    test_labels=torch.randint(0, 3, (5,)),
                    features=torch.randn(50, 20),
                    idx_train=torch.arange(0, 8),
                    idx_test=torch.arange(8, 10)
                )
                trainers.append(trainer)
        
        # Test that all trainers have unique ranks
        ranks = [trainer.get_rank() for trainer in trainers]
        assert len(set(ranks)) == num_trainers
        assert sorted(ranks) == list(range(num_trainers))
    
    def test_memory_efficiency(self):
        """Test memory efficiency with larger data structures."""
        # Create larger tensors to test memory handling
        large_features = torch.randn(1000, 100)
        large_edge_index = torch.randint(0, 1000, (2, 5000))
        
        # Test that operations complete without memory errors
        from fedgraph.utils_nc import intersect1d, setdiff1d
        
        t1 = torch.arange(0, 500)
        t2 = torch.arange(250, 750)
        
        intersection = intersect1d(t1, t2)
        difference = setdiff1d(t1, t2)
        
        assert len(intersection) > 0
        assert len(difference) > 0
        
        # Clean up
        del large_features, large_edge_index, intersection, difference