import pytest
import torch
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from torch_geometric.data import Data
import shutil

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_graph_data():
    """Create sample graph data for testing."""
    num_nodes = 10
    num_features = 5
    num_classes = 3
    
    # Create node features
    x = torch.randn(num_nodes, num_features)
    
    # Create edges (simple chain graph)
    edge_index = torch.tensor([[i, i+1] for i in range(num_nodes-1)]).t().contiguous()
    
    # Create labels
    y = torch.randint(0, num_classes, (num_nodes,))
    
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

@pytest.fixture
def mock_args():
    """Create mock arguments for testing."""
    args = Mock()
    args.num_clients = 5
    args.dataset = 'Cora'
    args.method = 'FedAvg'
    args.num_rounds = 10
    args.local_epochs = 5
    args.lr = 0.01
    args.hidden = 64
    args.dropout = 0.5
    args.device = 'cpu'
    args.seed = 42
    args.data_path = '/tmp/test_data'
    args.split_type = 'louvain'
    args.alpha = 0.5
    args.beta = 1.0
    args.num_workers = 1
    args.ray_init_address = None
    args.ray_dashboard_port = 8265
    args.he = False
    args.dp = False
    return args

@pytest.fixture
def mock_ray_cluster():
    """Mock Ray cluster for testing."""
    with patch('ray.init') as mock_init, \
         patch('ray.get') as mock_get, \
         patch('ray.put') as mock_put, \
         patch('ray.remote') as mock_remote:
        
        mock_init.return_value = None
        mock_get.side_effect = lambda x: x
        mock_put.side_effect = lambda x: x
        mock_remote.side_effect = lambda x: x
        
        yield {
            'init': mock_init,
            'get': mock_get,
            'put': mock_put,
            'remote': mock_remote
        }

@pytest.fixture
def sample_dataset_splits():
    """Create sample dataset splits for federated learning."""
    num_clients = 3
    num_nodes_per_client = 5
    
    splits = {}
    for i in range(num_clients):
        client_data = {
            'train_mask': torch.zeros(num_nodes_per_client, dtype=torch.bool),
            'val_mask': torch.zeros(num_nodes_per_client, dtype=torch.bool),
            'test_mask': torch.zeros(num_nodes_per_client, dtype=torch.bool),
            'node_list': list(range(i * num_nodes_per_client, (i + 1) * num_nodes_per_client))
        }
        # Set some nodes for training
        client_data['train_mask'][:3] = True
        client_data['val_mask'][3:4] = True
        client_data['test_mask'][4:5] = True
        
        splits[f'client_{i}'] = client_data
    
    return splits

@pytest.fixture
def mock_model():
    """Create a mock GNN model for testing."""
    model = Mock()
    model.parameters.return_value = [torch.randn(10, 5, requires_grad=True)]
    model.state_dict.return_value = {'layer.weight': torch.randn(10, 5)}
    model.load_state_dict = Mock()
    model.train = Mock()
    model.eval = Mock()
    return model

@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer for testing."""
    optimizer = Mock()
    optimizer.zero_grad = Mock()
    optimizer.step = Mock()
    optimizer.state_dict.return_value = {}
    optimizer.load_state_dict = Mock()
    return optimizer