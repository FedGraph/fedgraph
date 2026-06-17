import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Import utils_gc functions 
try:
    from fedgraph.utils_gc import (
        get_max_degree,
        get_num_graph_labels,
        get_stats,
        split_data,
        setup_server,
        setup_trainers
    )
    UTILS_GC_AVAILABLE = True
except ImportError:
    UTILS_GC_AVAILABLE = False

# Import utils_lp functions
try:
    from fedgraph.utils_lp import (
        check_data_files_existance,
        get_data,
        get_data_loaders_per_time_step,
        get_global_user_item_mapping,
        get_start_end_time,
        to_next_day
    )
    UTILS_LP_AVAILABLE = True
except ImportError:
    UTILS_LP_AVAILABLE = False


@pytest.mark.skipif(not UTILS_GC_AVAILABLE, reason="utils_gc not available")
class TestUtilsGC:
    """Test utils_gc functions."""
    
    def setup_method(self):
        """Setup test data for GC utils."""
        # Create mock graph data
        self.mock_graphs = []
        for i in range(5):
            graph = Mock()
            graph.num_nodes = 10 + i
            graph.num_node_features = 5
            graph.y = torch.tensor([i % 2])  # Binary classification
            graph.edge_index = torch.randint(0, 10+i, (2, 15+i))
            self.mock_graphs.append(graph)
    
    def test_get_max_degree(self):
        """Test get_max_degree function."""
        # Create mock dataset with graphs of different max degrees
        mock_dataset = self.mock_graphs
        
        with patch('torch_geometric.utils.degree') as mock_degree:
            # Mock degree function to return different max degrees
            mock_degree.side_effect = [
                torch.tensor([1, 2, 3, 4, 5] + [0]*5),  # max degree 5
                torch.tensor([2, 3, 4, 3, 2] + [0]*6),  # max degree 4  
                torch.tensor([1, 1, 1, 1, 1] + [0]*7),  # max degree 1
                torch.tensor([3, 5, 2, 4, 6] + [0]*8),  # max degree 6
                torch.tensor([2, 2, 2, 2, 2] + [0]*9),  # max degree 2
            ]
            
            max_degree = get_max_degree(mock_dataset)
            
            assert isinstance(max_degree, int)
            assert max_degree == 6  # Should be the maximum across all graphs
    
    def test_get_num_graph_labels(self):
        """Test get_num_graph_labels function."""
        # Create graphs with different labels
        graphs = []
        for i in range(4):
            graph = Mock()
            graph.y = torch.tensor([i % 3])  # Labels 0, 1, 2
            graphs.append(graph)
        
        num_labels = get_num_graph_labels(graphs)
        
        assert isinstance(num_labels, int)
        assert num_labels == 3  # Should detect 3 unique labels
    
    def test_get_stats(self):
        """Test get_stats function."""
        graphs = self.mock_graphs
        
        stats = get_stats(graphs)
        
        assert isinstance(stats, dict)
        expected_keys = ['num_graphs', 'num_nodes_avg', 'num_edges_avg', 'num_features']
        for key in expected_keys:
            assert key in stats
        
        assert stats['num_graphs'] == len(graphs)
        assert isinstance(stats['num_nodes_avg'], (int, float))
        assert isinstance(stats['num_edges_avg'], (int, float))
        assert isinstance(stats['num_features'], int)
    
    def test_split_data_with_train_test_sizes(self):
        """Test split_data with specific train/test sizes."""
        data = self.mock_graphs
        
        train_data, test_data = split_data(
            data, train_size=0.8, test_size=0.2, shuffle=True, seed=42
        )
        
        assert len(train_data) + len(test_data) == len(data)
        assert len(train_data) == int(0.8 * len(data))
        assert len(test_data) == len(data) - len(train_data)
    
    def test_split_data_with_absolute_sizes(self):
        """Test split_data with absolute sizes."""
        data = self.mock_graphs
        
        train_data, test_data = split_data(
            data, train_size=3, test_size=2, shuffle=False, seed=42
        )
        
        assert len(train_data) == 3
        assert len(test_data) == 2
    
    @patch('fedgraph.utils_gc.Server_GC')
    @patch('fedgraph.utils_gc.GIN')
    def test_setup_server(self, mock_gin, mock_server_gc):
        """Test setup_server function."""
        mock_model = Mock()
        mock_gin.return_value = mock_model
        
        mock_server = Mock()
        mock_server_gc.return_value = mock_server
        
        args = Mock()
        args.device = "cpu"
        
        num_features = 10
        num_classes = 2
        
        server = setup_server(args, num_features, num_classes)
        
        mock_gin.assert_called_once()
        mock_server_gc.assert_called_once()
        assert server == mock_server
    
    @patch('fedgraph.utils_gc.Trainer_GC')
    def test_setup_trainers(self, mock_trainer_gc):
        """Test setup_trainers function."""
        mock_trainer = Mock()
        mock_trainer_gc.return_value = mock_trainer
        
        args = Mock()
        args.device = "cpu"
        
        mock_model = Mock()
        mock_data = {
            "trainer_0": (
                {"train": Mock(), "val": Mock(), "test": Mock()},
                10, 2, 100
            ),
            "trainer_1": (
                {"train": Mock(), "val": Mock(), "test": Mock()},
                10, 2, 100
            )
        }
        
        trainers = setup_trainers(mock_model, mock_data, args)
        
        assert isinstance(trainers, list)
        assert len(trainers) == len(mock_data)
        mock_trainer_gc.assert_called()


@pytest.mark.skipif(not UTILS_LP_AVAILABLE, reason="utils_lp not available")
class TestUtilsLP:
    """Test utils_lp functions."""
    
    def test_check_data_files_existance_missing(self):
        """Test check_data_files_existance with missing files."""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            
            result = check_data_files_existance("fake_dataset", "/fake/path")
            
            assert result is False
    
    def test_check_data_files_existance_present(self):
        """Test check_data_files_existance with present files."""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            result = check_data_files_existance("ml-1m", "/fake/path")
            
            assert result is True
    
    def test_get_start_end_time(self):
        """Test get_start_end_time function."""
        # Mock data with timestamps
        mock_data = {
            "u1.base": Mock(),
            "interactions": [(1, 2, 3, 1000), (2, 3, 4, 2000), (3, 4, 5, 500)]
        }
        
        with patch('fedgraph.utils_lp.get_data') as mock_get_data:
            mock_get_data.return_value = mock_data
            
            start_time, end_time = get_start_end_time("ml-1m", "/fake/path")
            
            assert isinstance(start_time, int)
            assert isinstance(end_time, int)
            assert start_time <= end_time
    
    def test_to_next_day(self):
        """Test to_next_day function."""
        # Test with a timestamp (assuming it's in seconds)
        current_time = 1000000  # Some timestamp
        
        next_day = to_next_day(current_time)
        
        assert isinstance(next_day, int)
        assert next_day > current_time
        # Should be exactly 24 hours later (86400 seconds)
        assert next_day - current_time == 86400
    
    @patch('fedgraph.utils_lp.get_data')
    def test_get_global_user_item_mapping(self, mock_get_data):
        """Test get_global_user_item_mapping function."""
        # Mock interaction data
        mock_data = {
            "interactions": [
                (1, 10, 5, 1000),   # user 1, item 10, rating 5, time 1000
                (2, 20, 4, 2000),   # user 2, item 20, rating 4, time 2000
                (1, 30, 3, 3000),   # user 1, item 30, rating 3, time 3000
            ]
        }
        mock_get_data.return_value = mock_data
        
        user_mapping, item_mapping = get_global_user_item_mapping("ml-1m", "/fake/path")
        
        assert isinstance(user_mapping, dict)
        assert isinstance(item_mapping, dict)
        
        # Should map original IDs to sequential indices
        assert len(user_mapping) == 2  # users 1, 2
        assert len(item_mapping) == 3  # items 10, 20, 30
    
    @patch('fedgraph.utils_lp.get_data')
    @patch('fedgraph.utils_lp.create_temporal_split')
    def test_get_data_loaders_per_time_step(self, mock_create_split, mock_get_data):
        """Test get_data_loaders_per_time_step function."""
        # Mock data
        mock_data = {"interactions": [(1, 2, 3, 1000)]}
        mock_get_data.return_value = mock_data
        
        # Mock temporal split
        mock_split_data = {
            "train": [(1, 2, 3, 1000)],
            "test": [(1, 2, 3, 2000)]
        }
        mock_create_split.return_value = mock_split_data
        
        user_mapping = {1: 0}
        item_mapping = {2: 0}
        
        result = get_data_loaders_per_time_step(
            dataset="ml-1m",
            data_path="/fake/path",
            time_step=1000,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            num_users=1,
            num_items=1,
            args=Mock()
        )
        
        assert isinstance(result, dict)
        assert "train" in result
        assert "test" in result


class TestUtilsIntegration:
    """Integration tests for utility functions."""
    
    @pytest.mark.skipif(not UTILS_GC_AVAILABLE, reason="utils_gc not available")
    def test_gc_workflow(self):
        """Test complete GC utility workflow."""
        # Create mock graph data
        graphs = []
        for i in range(10):
            graph = Mock()
            graph.num_nodes = 5 + i
            graph.num_node_features = 3
            graph.y = torch.tensor([i % 2])
            graph.edge_index = torch.randint(0, 5+i, (2, 8+i))
            graphs.append(graph)
        
        # Test stats computation
        stats = get_stats(graphs)
        assert stats['num_graphs'] == 10
        
        # Test data splitting
        train_data, test_data = split_data(graphs, train_size=0.8, test_size=0.2)
        assert len(train_data) + len(test_data) == len(graphs)
        
        # Test label counting
        num_labels = get_num_graph_labels(graphs)
        assert num_labels == 2  # Binary classification
    
    @pytest.mark.skipif(not UTILS_LP_AVAILABLE, reason="utils_lp not available")
    def test_lp_data_workflow(self):
        """Test complete LP utility workflow."""
        with patch('fedgraph.utils_lp.get_data') as mock_get_data:
            # Mock complete interaction data
            mock_data = {
                "interactions": [
                    (1, 10, 5, 1000),
                    (2, 20, 4, 2000), 
                    (1, 30, 3, 3000),
                    (3, 10, 4, 4000)
                ]
            }
            mock_get_data.return_value = mock_data
            
            # Test mapping creation
            user_mapping, item_mapping = get_global_user_item_mapping("ml-1m", "/fake/path")
            
            assert len(user_mapping) == 3  # Users 1, 2, 3
            assert len(item_mapping) == 3  # Items 10, 20, 30
            
            # Test time range computation  
            start_time, end_time = get_start_end_time("ml-1m", "/fake/path")
            
            assert start_time == 1000
            assert end_time == 4000
            
            # Test next day computation
            next_day = to_next_day(start_time)
            assert next_day == start_time + 86400