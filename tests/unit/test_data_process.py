import pytest
import torch
import numpy as np
import tempfile
import os
import pickle as pkl
from unittest.mock import Mock, patch, MagicMock, mock_open
import attridict
import torch_sparse
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from fedgraph.data_process import (
    data_loader,
    data_loader_NC,
    data_loader_GC,
    NC_parse_index_file,
    download_file_from_github,
    NC_load_data,
    GC_rand_split_chunk,
    data_loader_GC_single,
    data_loader_GC_multiple
)


class TestDataLoader:
    """Test the main data_loader function."""
    
    def test_data_loader_LP_returns_none(self):
        """Test that data_loader returns None for LP task."""
        args = Mock()
        args.fedgraph_task = "LP"
        
        result = data_loader(args)
        assert result is None
    
    @patch('fedgraph.data_process.data_loader_NC')
    def test_data_loader_NC_calls_correct_function(self, mock_nc_loader):
        """Test that data_loader calls data_loader_NC for NC task."""
        args = Mock()
        args.fedgraph_task = "NC"
        mock_nc_loader.return_value = "mock_nc_data"
        
        result = data_loader(args)
        
        mock_nc_loader.assert_called_once_with(args)
        assert result == "mock_nc_data"
    
    @patch('fedgraph.data_process.data_loader_GC')
    def test_data_loader_GC_calls_correct_function(self, mock_gc_loader):
        """Test that data_loader calls data_loader_GC for GC task."""
        args = Mock()
        args.fedgraph_task = "GC"
        mock_gc_loader.return_value = "mock_gc_data"
        
        result = data_loader(args)
        
        mock_gc_loader.assert_called_once_with(args)
        assert result == "mock_gc_data"


class TestNCParseIndexFile:
    """Test NC_parse_index_file function."""
    
    def test_parse_index_file_success(self, temp_dir):
        """Test parsing index file successfully."""
        # Create test file
        test_file = os.path.join(temp_dir, "test_index.txt")
        with open(test_file, 'w') as f:
            f.write("1\n2\n3\n10\n")
        
        result = NC_parse_index_file(test_file)
        assert result == [1, 2, 3, 10]
    
    def test_parse_empty_file(self, temp_dir):
        """Test parsing empty index file."""
        test_file = os.path.join(temp_dir, "empty.txt")
        with open(test_file, 'w') as f:
            pass
        
        result = NC_parse_index_file(test_file)
        assert result == []


class TestDownloadFileFromGithub:
    """Test download_file_from_github function."""
    
    @patch('fedgraph.data_process.requests.get')
    def test_download_file_success(self, mock_get, temp_dir):
        """Test successful file download."""
        save_path = os.path.join(temp_dir, "test_file.txt")
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b"test", b"data"]
        mock_get.return_value = mock_response
        
        download_file_from_github("http://test.com/file.txt", save_path)
        
        assert os.path.exists(save_path)
        with open(save_path, 'rb') as f:
            assert f.read() == b"testdata"
    
    @patch('fedgraph.data_process.requests.get')
    def test_download_file_failure(self, mock_get, temp_dir):
        """Test failed file download."""
        save_path = os.path.join(temp_dir, "test_file.txt")
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        with pytest.raises(Exception, match="Failed to download"):
            download_file_from_github("http://test.com/nonexistent.txt", save_path)
    
    def test_file_already_exists(self, temp_dir):
        """Test that existing files are not re-downloaded."""
        save_path = os.path.join(temp_dir, "existing_file.txt")
        with open(save_path, 'w') as f:
            f.write("existing content")
        
        with patch('fedgraph.data_process.requests.get') as mock_get:
            download_file_from_github("http://test.com/file.txt", save_path)
            mock_get.assert_not_called()


class TestNCLoadData:
    """Test NC_load_data function."""
    
    @patch('fedgraph.data_process.download_file_from_github')
    @patch('builtins.open', new_callable=mock_open)
    @patch('fedgraph.data_process.pkl.load')
    @patch('fedgraph.data_process.NC_parse_index_file')
    @patch('os.makedirs')
    def test_load_cora_dataset(self, mock_makedirs, mock_parse, mock_pkl_load, 
                              mock_open_file, mock_download):
        """Test loading Cora dataset."""
        # Setup mock data
        mock_x = np.random.random((100, 50))
        mock_y = np.random.randint(0, 7, (100, 7))
        mock_tx = np.random.random((1000, 50))
        mock_ty = np.random.randint(0, 7, (1000, 7))
        mock_allx = np.random.random((1708, 50))
        mock_ally = np.random.randint(0, 7, (1708, 7))
        mock_graph = {i: [i+1] for i in range(1707)}
        
        mock_pkl_load.side_effect = [
            mock_x, mock_y, mock_tx, mock_ty, 
            mock_allx, mock_ally, mock_graph
        ]
        mock_parse.return_value = list(range(1000, 1708))
        
        # Mock scipy sparse matrices
        with patch('scipy.sparse.vstack') as mock_vstack, \
             patch('networkx.adjacency_matrix') as mock_adj, \
             patch('networkx.from_dict_of_lists') as mock_from_dict:
            
            mock_features = Mock()
            mock_features.toarray.return_value = np.random.random((2708, 50))
            mock_vstack.return_value = mock_features
            
            mock_adj_matrix = Mock()
            mock_adj_matrix.toarray.return_value = np.random.random((2708, 2708))
            mock_adj.return_value = mock_adj_matrix
            
            result = NC_load_data("cora")
            
            assert len(result) == 6
            features, adj, labels, idx_train, idx_val, idx_test = result
            assert isinstance(features, torch.Tensor)
            assert isinstance(adj, torch_sparse.tensor.SparseTensor)
            assert isinstance(labels, torch.Tensor)
            assert isinstance(idx_train, torch.Tensor)
            assert isinstance(idx_val, torch.Tensor)
            assert isinstance(idx_test, torch.Tensor)


class TestGCRandSplitChunk:
    """Test GC_rand_split_chunk function."""
    
    def test_non_overlapping_split(self):
        """Test non-overlapping split of graphs."""
        graphs = [Mock() for _ in range(100)]
        num_trainer = 5
        
        result = GC_rand_split_chunk(graphs, num_trainer, overlap=False, seed=42)
        
        assert len(result) == num_trainer
        # Check that all chunks together contain all graphs
        total_graphs = sum(len(chunk) for chunk in result)
        assert total_graphs == len(graphs)
    
    def test_overlapping_split(self):
        """Test overlapping split of graphs."""
        graphs = [Mock() for _ in range(100)]
        num_trainer = 3
        
        with patch('numpy.random.randint') as mock_randint:
            mock_randint.return_value = np.array([75, 80, 85])
            
            result = GC_rand_split_chunk(graphs, num_trainer, overlap=True, seed=42)
            
            assert len(result) == num_trainer
            for chunk in result:
                assert len(chunk) >= 50  # minimum size from function


class TestDataLoaderGCSingle:
    """Test data_loader_GC_single function."""
    
    @patch('fedgraph.data_process.TUDataset')
    @patch('fedgraph.data_process.GC_rand_split_chunk')
    @patch('fedgraph.data_process.split_data')
    @patch('fedgraph.data_process.DataLoader')
    @patch('fedgraph.data_process.get_num_graph_labels')
    def test_data_loader_GC_single_success(self, mock_get_labels, mock_dataloader, 
                                          mock_split, mock_chunk, mock_tudataset):
        """Test successful GC single dataset loading."""
        # Setup mocks
        mock_dataset = Mock()
        mock_graph = Mock()
        mock_graph.num_node_features = 10
        mock_dataset.__iter__ = Mock(return_value=iter([mock_graph] * 100))
        mock_tudataset.return_value = mock_dataset
        
        mock_chunk.return_value = [[mock_graph] * 20 for _ in range(5)]
        mock_split.side_effect = [
            ([mock_graph] * 16, [mock_graph] * 4),  # train/val_test split
            ([mock_graph] * 2, [mock_graph] * 2)    # val/test split
        ]
        mock_get_labels.return_value = 2
        mock_dataloader.return_value = Mock()
        
        result = data_loader_GC_single(
            datapath="/tmp",
            dataset="PROTEINS",
            num_trainer=5,
            batch_size=32,
            seed=42
        )
        
        assert isinstance(result, dict)
        assert len(result) == 5  # num_trainer
        for key, value in result.items():
            assert isinstance(value, tuple)
            assert len(value) == 4  # dataloaders, num_features, num_labels, train_size


class TestDataLoaderGCMultiple:
    """Test data_loader_GC_multiple function."""
    
    @patch('fedgraph.data_process.TUDataset')
    @patch('fedgraph.data_process.split_data')
    @patch('fedgraph.data_process.DataLoader')
    @patch('fedgraph.data_process.get_num_graph_labels')
    def test_data_loader_GC_multiple_small_group(self, mock_get_labels, mock_dataloader,
                                                 mock_split, mock_tudataset):
        """Test GC multiple datasets loading for small group."""
        # Setup mocks
        mock_dataset = Mock()
        mock_graph = Mock()
        mock_graph.num_node_features = 10
        mock_dataset.__iter__ = Mock(return_value=iter([mock_graph] * 50))
        mock_tudataset.return_value = mock_dataset
        
        mock_split.side_effect = [
            ([mock_graph] * 40, [mock_graph] * 10),  # train/val_test split
            ([mock_graph] * 5, [mock_graph] * 5)     # val/test split
        ] * 8  # For 8 datasets in small group
        
        mock_get_labels.return_value = 2
        mock_dataloader.return_value = Mock()
        
        result = data_loader_GC_multiple(
            datapath="/tmp",
            dataset_group="small",
            batch_size=32,
            seed=42
        )
        
        assert isinstance(result, dict)
        expected_datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "ENZYMES", "DD", "PROTEINS"]
        assert len(result) == len(expected_datasets)
        
        for dataset_name in expected_datasets:
            assert dataset_name in result
            value = result[dataset_name]
            assert isinstance(value, tuple)
            assert len(value) == 4


class TestDataLoaderNC:
    """Test data_loader_NC function with integration."""
    
    @patch('fedgraph.data_process.NC_load_data')
    @patch('fedgraph.data_process.label_dirichlet_partition')
    @patch('fedgraph.data_process.get_in_comm_indexes')
    def test_data_loader_NC_complete_flow(self, mock_get_indexes, mock_partition, 
                                         mock_load_data):
        """Test complete NC data loading flow."""
        # Setup mock args
        args = Mock()
        args.use_huggingface = False
        args.dataset = "cora"
        args.n_trainer = 3
        args.iid_beta = 0.5
        args.distribution_type = "dirichlet"
        args.num_hops = 2
        
        # Setup mock data
        num_nodes = 100
        num_features = 50
        num_classes = 7
        
        features = torch.randn(num_nodes, num_features)
        adj = torch_sparse.tensor.SparseTensor.from_dense(
            torch.randn(num_nodes, num_nodes)
        )
        labels = torch.randint(0, num_classes, (num_nodes,))
        idx_train = torch.arange(0, 70)
        idx_val = torch.arange(70, 85)
        idx_test = torch.arange(85, 100)
        
        mock_load_data.return_value = (features, adj, labels, idx_train, idx_val, idx_test)
        
        # Mock partition results
        mock_partition.return_value = [
            list(range(0, 33)),
            list(range(33, 66)),
            list(range(66, 100))
        ]
        
        # Mock communication indexes
        mock_get_indexes.return_value = (
            {i: torch.arange(i*10, (i+1)*10) for i in range(3)},  # communicate_node_global_indexes
            {i: torch.arange(0, 5) for i in range(3)},            # in_com_train_node_local_indexes
            {i: torch.arange(5, 8) for i in range(3)},            # in_com_test_node_local_indexes
            {i: torch.stack([torch.arange(0, 10), torch.arange(1, 11)]) for i in range(3)}  # global_edge_indexes_clients
        )
        
        result = data_loader_NC(args)
        
        assert len(result) == 11
        (edge_index, returned_features, returned_labels, returned_idx_train, 
         returned_idx_test, class_num, split_node_indexes, 
         communicate_node_global_indexes, in_com_train_node_local_indexes,
         in_com_test_node_local_indexes, global_edge_indexes_clients) = result
        
        # Verify results
        assert torch.equal(returned_features, features)
        assert torch.equal(returned_labels, labels)
        assert torch.equal(returned_idx_train, idx_train)
        assert torch.equal(returned_idx_test, idx_test)
        assert class_num == num_classes
        assert len(split_node_indexes) == args.n_trainer
        
        # Verify function calls
        mock_load_data.assert_called_once_with(args.dataset)
        mock_partition.assert_called_once()
        mock_get_indexes.assert_called_once()


class TestDataLoaderGC:
    """Test data_loader_GC function."""
    
    @patch('fedgraph.data_process.data_loader_GC_multiple')
    def test_data_loader_GC_multiple_datasets(self, mock_multiple):
        """Test GC data loader for multiple datasets."""
        args = Mock()
        args.is_multiple_dataset = True
        args.datapath = "/tmp"
        args.dataset_group = "small"
        args.batch_size = 32
        args.convert_x = False
        args.seed_split_data = 42
        
        mock_multiple.return_value = {"test": "data"}
        
        result = data_loader_GC(args)
        
        mock_multiple.assert_called_once_with(
            datapath=args.datapath,
            dataset_group=args.dataset_group,
            batch_size=args.batch_size,
            convert_x=args.convert_x,
            seed=args.seed_split_data
        )
        assert result == {"test": "data"}
    
    @patch('fedgraph.data_process.data_loader_GC_single')
    def test_data_loader_GC_single_dataset(self, mock_single):
        """Test GC data loader for single dataset."""
        args = Mock()
        args.is_multiple_dataset = False
        args.datapath = "/tmp"
        args.dataset = "PROTEINS"
        args.num_trainers = 5
        args.batch_size = 32
        args.convert_x = False
        args.seed_split_data = 42
        args.overlap = False
        
        mock_single.return_value = {"test": "data"}
        
        result = data_loader_GC(args)
        
        mock_single.assert_called_once_with(
            datapath=args.datapath,
            dataset=args.dataset,
            num_trainer=args.num_trainers,
            batch_size=args.batch_size,
            convert_x=args.convert_x,
            seed=args.seed_split_data,
            overlap=args.overlap
        )
        assert result == {"test": "data"}


@pytest.mark.integration
class TestDataProcessIntegration:
    """Integration tests for data processing functions."""
    
    def test_complete_NC_workflow_mock(self):
        """Test complete NC workflow with mocked dependencies."""
        with patch('fedgraph.data_process.NC_load_data') as mock_load, \
             patch('fedgraph.data_process.label_dirichlet_partition') as mock_partition, \
             patch('fedgraph.data_process.get_in_comm_indexes') as mock_indexes:
            
            # Setup test data
            features = torch.randn(100, 50)
            adj = torch_sparse.tensor.SparseTensor.from_dense(torch.eye(100))
            labels = torch.randint(0, 7, (100,))
            idx_train = torch.arange(0, 70)
            idx_val = torch.arange(70, 85)
            idx_test = torch.arange(85, 100)
            
            mock_load.return_value = (features, adj, labels, idx_train, idx_val, idx_test)
            mock_partition.return_value = [list(range(0, 50)), list(range(50, 100))]
            mock_indexes.return_value = ({}, {}, {}, {})
            
            # Create args
            args = Mock()
            args.fedgraph_task = "NC"
            args.use_huggingface = False
            args.dataset = "cora"
            args.n_trainer = 2
            args.iid_beta = 0.5
            args.distribution_type = "dirichlet"
            args.num_hops = 2
            
            # Test the complete workflow
            result = data_loader(args)
            
            assert result is not None
            assert len(result) == 11
            mock_load.assert_called_once()
            mock_partition.assert_called_once()
            mock_indexes.assert_called_once()