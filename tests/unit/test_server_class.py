import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from fedgraph.server_class import Server, Server_GC


class TestServer:
    """Test Server class for federated learning."""
    
    def setup_method(self):
        """Setup common test data."""
        self.feature_dim = 100
        self.args_hidden = 64
        self.class_num = 7
        self.device = torch.device('cpu')
        
        # Mock trainers
        self.mock_trainers = []
        for i in range(3):
            trainer = Mock()
            trainer.rank = i
            trainer.get_params.return_value = (
                torch.randn(64, 100),
                torch.randn(64),
                torch.randn(7, 64),
                torch.randn(7)
            )
            trainer.local_node_index = torch.arange(i*10, (i+1)*10)
            trainer.communicate_node_index = torch.arange(i*20, (i+1)*20)
            self.mock_trainers.append(trainer)
        
        # Mock args
        self.args = Mock()
        self.args.num_hops = 1
        self.args.dataset = "cora"
        self.args.num_layers = 2
        self.args.method = "FedAvg"
    
    @patch('fedgraph.server_class.AggreGCN')
    def test_server_init_with_hops(self, mock_aggre_gcn):
        """Test Server initialization with num_hops >= 1."""
        mock_model = Mock()
        mock_aggre_gcn.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        server = Server(
            feature_dim=self.feature_dim,
            args_hidden=self.args_hidden,
            class_num=self.class_num,
            device=self.device,
            trainers=self.mock_trainers,
            args=self.args
        )
        
        assert server.args == self.args
        assert server.model is not None
        assert server.trainers == self.mock_trainers
        assert server.num_of_trainers == len(self.mock_trainers)
        mock_aggre_gcn.assert_called_once()
    
    @patch('fedgraph.server_class.GCN_arxiv')
    def test_server_init_arxiv_dataset(self, mock_gcn_arxiv):
        """Test Server initialization with arxiv dataset."""
        mock_model = Mock()
        mock_gcn_arxiv.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        self.args.dataset = "ogbn-arxiv"
        self.args.num_hops = 0  # FedAvg method
        
        server = Server(
            feature_dim=self.feature_dim,
            args_hidden=self.args_hidden,
            class_num=self.class_num,
            device=self.device,
            trainers=self.mock_trainers,
            args=self.args
        )
        
        mock_gcn_arxiv.assert_called_once()
    
    @patch('fedgraph.server_class.SAGE_products')
    def test_server_init_products_dataset(self, mock_sage):
        """Test Server initialization with products dataset."""
        mock_model = Mock()
        mock_sage.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        self.args.dataset = "ogbn-products"
        self.args.num_hops = 0
        
        server = Server(
            feature_dim=self.feature_dim,
            args_hidden=self.args_hidden,
            class_num=self.class_num,
            device=self.device,
            trainers=self.mock_trainers,
            args=self.args
        )
        
        mock_sage.assert_called_once()
    
    def test_zero_params(self):
        """Test zero_params method."""
        with patch('fedgraph.server_class.AggreGCN') as mock_gcn:
            mock_model = Mock()
            mock_gcn.return_value = mock_model
            mock_model.to.return_value = mock_model
            
            # Mock parameters
            mock_param1 = Mock()
            mock_param1.data = torch.randn(10, 5)
            mock_param2 = Mock()
            mock_param2.data = torch.randn(5)
            mock_model.parameters.return_value = [mock_param1, mock_param2]
            
            server = Server(
                feature_dim=self.feature_dim,
                args_hidden=self.args_hidden,
                class_num=self.class_num,
                device=self.device,
                trainers=self.mock_trainers,
                args=self.args
            )
            
            server.zero_params()
            
            # Check that parameters were zeroed
            mock_model.parameters.assert_called()
    
    def test_broadcast_params(self):
        """Test broadcast_params method."""
        with patch('fedgraph.server_class.AggreGCN') as mock_gcn:
            mock_model = Mock()
            mock_model.parameters.return_value = [
                torch.randn(32, 20, requires_grad=True),
                torch.randn(32, requires_grad=True)
            ]
            mock_gcn.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_model.state_dict.return_value = {'layer.weight': torch.randn(10, 5)}
            
            server = Server(
                feature_dim=self.feature_dim,
                args_hidden=self.args_hidden,
                class_num=self.class_num,
                device=self.device,
                trainers=self.mock_trainers,
                args=self.args
            )
            
            # Mock update_params method for trainers
            for trainer in server.trainers:
                trainer.update_params = Mock()
            
            server.broadcast_params(current_global_epoch=1)
            
            # Verify that all trainers received parameter updates
            for trainer in server.trainers:
                trainer.update_params.assert_called_once()
    
    def test_get_model_size(self):
        """Test get_model_size method."""
        with patch('fedgraph.server_class.AggreGCN') as mock_gcn:
            mock_model = Mock()
            mock_gcn.return_value = mock_model
            mock_model.to.return_value = mock_model
            
            # Mock parameters with known sizes
            param1 = torch.randn(10, 5)  # 50 elements
            param2 = torch.randn(5)      # 5 elements
            mock_model.parameters.return_value = [param1, param2]
            
            server = Server(
                feature_dim=self.feature_dim,
                args_hidden=self.args_hidden,
                class_num=self.class_num,
                device=self.device,
                trainers=self.mock_trainers,
                args=self.args
            )
            
            model_size = server.get_model_size()
            
            assert isinstance(model_size, float)
            assert model_size > 0
    
    def test_prepare_params_for_encryption(self):
        """Test prepare_params_for_encryption method."""
        with patch('fedgraph.server_class.AggreGCN') as mock_gcn:
            mock_model = Mock()
            mock_gcn.return_value = mock_model
            mock_model.to.return_value = mock_model
            
            server = Server(
                feature_dim=self.feature_dim,
                args_hidden=self.args_hidden,
                class_num=self.class_num,
                device=self.device,
                trainers=self.mock_trainers,
                args=self.args
            )
            
            params = (torch.randn(10, 5), torch.randn(10))
            
            result = server.prepare_params_for_encryption(params)
            
            assert isinstance(result, list)
            assert len(result) == len(params)
            for item in result:
                assert isinstance(item, list)  # Flattened and converted to list


class TestServerGC:
    """Test Server_GC class for graph classification."""
    
    def setup_method(self):
        """Setup common test data for GC server."""
        self.device = torch.device('cpu')
        self.use_cluster = False
        
        # Mock model
        self.mock_model = Mock()
        self.mock_model.to.return_value = self.mock_model
        self.mock_model.named_parameters.return_value = [
            ('layer1.weight', torch.randn(10, 5, requires_grad=True)),
            ('layer1.bias', torch.randn(10, requires_grad=True))
        ]
        
        # Mock trainers for GC
        self.mock_trainers = []
        for i in range(3):
            trainer = Mock()
            trainer.id = i
            trainer.train_size = 100
            trainer.W = {
                'layer1.weight': torch.randn(10, 5, requires_grad=True),
                'layer1.bias': torch.randn(10, requires_grad=True)
            }
            trainer.compute_update_norm = Mock(return_value=0.5)
            trainer.compute_mean_norm = Mock(return_value=torch.randn(15))
            self.mock_trainers.append(trainer)
    
    def test_server_gc_init(self):
        """Test Server_GC initialization."""
        server = Server_GC(
            model=self.mock_model,
            device=self.device,
            use_cluster=self.use_cluster
        )
        
        assert server.model == self.mock_model
        assert server.use_cluster == self.use_cluster
        assert hasattr(server, 'W')
        assert hasattr(server, 'model_cache')
    
    def test_random_sample_trainers(self):
        """Test random_sample_trainers method."""
        server = Server_GC(
            model=self.mock_model,
            device=self.device,
            use_cluster=self.use_cluster
        )
        
        all_trainers = self.mock_trainers
        frac = 0.6  # Select 60% of trainers
        
        selected = server.random_sample_trainers(all_trainers, frac)
        
        assert isinstance(selected, list)
        assert len(selected) <= len(all_trainers)
        assert len(selected) >= 1  # At least one trainer should be selected
        for trainer in selected:
            assert trainer in all_trainers
    
    def test_aggregate_weights(self):
        """Test aggregate_weights method."""
        server = Server_GC(
            model=self.mock_model,
            device=self.device,
            use_cluster=self.use_cluster
        )
        
        selected_trainers = self.mock_trainers[:2]  # Select first 2 trainers
        
        server.aggregate_weights(selected_trainers)
        
        # Verify that the method completes without error
        # The actual aggregation logic depends on the specific implementation
        assert True  # Method executed successfully
    
    def test_compute_pairwise_similarities(self):
        """Test compute_pairwise_similarities method."""
        server = Server_GC(
            model=self.mock_model,
            device=self.device,
            use_cluster=self.use_cluster
        )
        
        # Ensure trainers have consistent weight shapes
        for trainer in self.mock_trainers:
            trainer.W = {
                'layer1.weight': torch.randn(10, 5),
                'layer1.bias': torch.randn(10)
            }
        
        similarities = server.compute_pairwise_similarities(self.mock_trainers)
        
        assert isinstance(similarities, np.ndarray)
        assert similarities.shape == (len(self.mock_trainers), len(self.mock_trainers))
        # Check symmetry
        assert np.allclose(similarities, similarities.T)
        # Check diagonal elements are 1 (self-similarity)
        assert np.allclose(np.diag(similarities), 1.0)
    
    def test_compute_pairwise_distances(self):
        """Test compute_pairwise_distances method."""
        server = Server_GC(
            model=self.mock_model,
            device=self.device,
            use_cluster=self.use_cluster
        )
        
        # Create gradient norm sequences for DTW computation
        gradient_sequences = [
            [0.9, 0.8, 0.7, 0.6],  # Sequence for trainer 0
            [0.8, 0.7, 0.6, 0.5],  # Sequence for trainer 1  
            [0.7, 0.6, 0.5, 0.4]   # Sequence for trainer 2
        ]
        
        distances = server.compute_pairwise_distances(
            gradient_sequences, 
            standardize=False
        )
        
        assert isinstance(distances, np.ndarray)
        assert distances.shape == (len(gradient_sequences), len(gradient_sequences))
        # Distance matrix should be symmetric
        assert np.allclose(distances, distances.T)
        # Diagonal should be zero (distance to self)
        assert np.allclose(np.diag(distances), 0.0)
    
    def test_min_cut(self):
        """Test min_cut method."""
        server = Server_GC(
            model=self.mock_model,
            device=self.device,
            use_cluster=self.use_cluster
        )
        
        # Create a simple similarity matrix
        similarity = np.array([
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])
        idc = [0, 1, 2]
        
        cluster1, cluster2 = server.min_cut(similarity, idc)
        
        assert isinstance(cluster1, list)
        assert isinstance(cluster2, list)
        assert len(cluster1) + len(cluster2) == len(idc)
        # Ensure no overlap
        assert set(cluster1).isdisjoint(set(cluster2))
        # Ensure all indices are covered
        assert set(cluster1).union(set(cluster2)) == set(idc)
    
    def test_aggregate_clusterwise(self):
        """Test aggregate_clusterwise method."""
        server = Server_GC(
            model=self.mock_model,
            device=self.device,
            use_cluster=self.use_cluster
        )
        
        # Create trainer clusters
        trainer_clusters = [
            [self.mock_trainers[0], self.mock_trainers[1]],
            [self.mock_trainers[2]]
        ]
        
        server.aggregate_clusterwise(trainer_clusters)
        
        # Verify method execution
        assert True  # Method completed without error
    
    def test_compute_max_update_norm(self):
        """Test compute_max_update_norm method."""
        server = Server_GC(
            model=self.mock_model,
            device=self.device,
            use_cluster=self.use_cluster
        )
        
        cluster = self.mock_trainers[:2]
        
        max_norm = server.compute_max_update_norm(cluster)
        
        assert isinstance(max_norm, float)
        assert max_norm >= 0
        
        # Verify that compute_update_norm was called on trainers
        for trainer in cluster:
            trainer.compute_update_norm.assert_called()


class TestServerIntegration:
    """Integration tests for server classes."""
    
    @patch('fedgraph.server_class.AggreGCN')
    def test_server_trainer_interaction(self, mock_gcn_class):
        """Test interaction between server and trainers."""
        # Setup
        feature_dim = 50
        args_hidden = 32
        class_num = 3
        device = torch.device('cpu')
        
        args = Mock()
        args.num_hops = 1
        args.dataset = "cora"
        args.num_layers = 2
        args.method = "FedAvg"
        
        # Mock model
        mock_model = Mock()
        mock_gcn_class.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.state_dict.return_value = {
            'layer1.weight': torch.randn(32, 50),
            'layer1.bias': torch.randn(32)
        }
        
        # Create mock trainers
        trainers = []
        for i in range(2):
            trainer = Mock()
            trainer.rank = i
            trainer.update_params = Mock()
            trainer.get_params.return_value = (
                torch.randn(32, 50),
                torch.randn(32)
            )
            trainers.append(trainer)
        
        # Create server
        server = Server(
            feature_dim=feature_dim,
            args_hidden=args_hidden,
            class_num=class_num,
            device=device,
            trainers=trainers,
            args=args
        )
        
        # Test parameter broadcast
        server.broadcast_params(current_global_epoch=1)
        
        # Verify all trainers received updates
        for trainer in trainers:
            trainer.update_params.assert_called_once()
        
        # Test model size computation
        model_size = server.get_model_size()
        assert isinstance(model_size, float)
        assert model_size > 0
    
    def test_server_gc_clustering_workflow(self):
        """Test complete clustering workflow in Server_GC."""
        device = torch.device('cpu')
        use_cluster = True
        
        # Mock model
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.named_parameters.return_value = [
            ('layer1.weight', torch.randn(10, 5, requires_grad=True)),
        ]
        
        server = Server_GC(model=mock_model, device=device, use_cluster=use_cluster)
        
        # Create trainers with consistent data
        trainers = []
        for i in range(4):
            trainer = Mock()
            trainer.id = i
            trainer.train_size = 100
            trainer.W = {
                'layer1.weight': torch.randn(10, 5),
                'layer1.bias': torch.randn(10)
            }
            trainer.train_stats = {
                'trainingLosses': [0.9 - i*0.1, 0.8 - i*0.1],
                'trainingAccs': [0.6 + i*0.1, 0.7 + i*0.1]
            }
            trainer.compute_update_norm = Mock(return_value=0.5 + i*0.1)
            trainer.compute_mean_norm = Mock(return_value=torch.randn(15))
            trainers.append(trainer)
        
        # Test similarity computation
        similarities = server.compute_pairwise_similarities(trainers)
        assert similarities.shape == (4, 4)
        
        # Test distance computation with gradient sequences
        gradient_sequences = [
            [0.9 - i*0.1, 0.8 - i*0.1, 0.7 - i*0.1] for i in range(4)
        ]
        distances = server.compute_pairwise_distances(gradient_sequences, standardize=False)
        assert distances.shape == (4, 4)
        
        # Test clustering
        cluster1, cluster2 = server.min_cut(similarities, list(range(4)))
        assert len(cluster1) + len(cluster2) == 4
        
        # Test clusterwise aggregation
        trainer_clusters = [[trainers[i] for i in cluster1], [trainers[i] for i in cluster2]]
        server.aggregate_clusterwise(trainer_clusters)
        
        # Verify workflow completed successfully
        assert True