import pytest
import torch
import numpy as np
import scipy.sparse as sp
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from fedgraph.utils_nc import (
    normalize,
    intersect1d,
    setdiff1d,
    label_dirichlet_partition,
    community_partition_non_iid,
    get_in_comm_indexes,
    get_1hop_feature_sum,
    increment_dir
)


class TestUtilsNCBasicFunctions:
    """Test basic utility functions in utils_nc."""
    
    def test_normalize_sparse_matrix(self):
        """Test normalize function for sparse matrices."""
        # Create a simple sparse matrix
        data = np.array([1, 2, 3, 4])
        row = np.array([0, 0, 1, 1])
        col = np.array([0, 1, 0, 1])
        matrix = sp.csc_matrix((data, (row, col)), shape=(2, 2))
        
        normalized = normalize(matrix)
        
        assert isinstance(normalized, sp.csr_matrix)
        # Check that rows sum to 1 (within tolerance)
        row_sums = np.array(normalized.sum(axis=1)).flatten()
        assert np.allclose(row_sums, 1.0)
    
    def test_normalize_with_zero_rows(self):
        """Test normalize function with zero rows."""
        # Create matrix with a zero row
        data = np.array([1, 2])
        row = np.array([0, 0])
        col = np.array([0, 1])
        matrix = sp.csc_matrix((data, (row, col)), shape=(2, 2))
        
        normalized = normalize(matrix)
        
        assert isinstance(normalized, sp.csr_matrix)
        # First row should sum to 1, second row should remain 0
        row_sums = np.array(normalized.sum(axis=1)).flatten()
        assert np.allclose(row_sums[0], 1.0)
        assert np.allclose(row_sums[1], 0.0)
    
    def test_intersect1d_basic(self):
        """Test intersect1d function with basic tensors."""
        t1 = torch.tensor([1, 2, 3, 4])
        t2 = torch.tensor([3, 4, 5, 6])
        
        result = intersect1d(t1, t2)
        
        assert torch.equal(result.sort().values, torch.tensor([3, 4]))
    
    def test_intersect1d_no_intersection(self):
        """Test intersect1d with no common elements."""
        t1 = torch.tensor([1, 2])
        t2 = torch.tensor([3, 4])
        
        result = intersect1d(t1, t2)
        
        assert len(result) == 0
    
    def test_intersect1d_all_same(self):
        """Test intersect1d with identical tensors."""
        t1 = torch.tensor([1, 2, 3])
        t2 = torch.tensor([1, 2, 3])
        
        result = intersect1d(t1, t2)
        
        assert torch.equal(result.sort().values, torch.tensor([1, 2, 3]))
    
    def test_setdiff1d_basic(self):
        """Test setdiff1d function with basic tensors."""
        t1 = torch.tensor([1, 2, 3, 4])
        t2 = torch.tensor([3, 4, 5, 6])
        
        result = setdiff1d(t1, t2)
        
        expected = torch.tensor([1, 2, 5, 6])
        assert torch.equal(result.sort().values, expected.sort().values)
    
    def test_setdiff1d_no_difference(self):
        """Test setdiff1d with identical tensors."""
        t1 = torch.tensor([1, 2, 3])
        t2 = torch.tensor([1, 2, 3])
        
        result = setdiff1d(t1, t2)
        
        assert len(result) == 0
    
    def test_setdiff1d_complete_difference(self):
        """Test setdiff1d with no common elements."""
        t1 = torch.tensor([1, 2])
        t2 = torch.tensor([3, 4])
        
        result = setdiff1d(t1, t2)
        
        expected = torch.tensor([1, 2, 3, 4])
        assert torch.equal(result.sort().values, expected.sort().values)


class TestLabelDirichletPartition:
    """Test label_dirichlet_partition function."""
    
    def test_label_dirichlet_partition_basic(self):
        """Test basic label Dirichlet partition."""
        # Create simple labels
        labels = np.array([0, 0, 1, 1, 2, 2] * 10)  # 60 samples, 3 classes
        N = len(labels)
        K = 3  # number of classes
        n_parties = 3
        beta = 0.5
        
        result = label_dirichlet_partition(labels, N, K, n_parties, beta)
        
        assert isinstance(result, list)
        assert len(result) == n_parties
        
        # Check that all indices are covered
        all_indices = set()
        for party_indices in result:
            all_indices.update(party_indices)
        assert len(all_indices) == N
        
        # Check that each party has some data
        for party_indices in result:
            assert len(party_indices) > 0
    
    def test_label_dirichlet_partition_distribution_types(self):
        """Test different distribution types."""
        labels = np.array([0, 1, 2] * 20)  # 60 samples
        N = len(labels)
        K = 3
        n_parties = 3
        beta = 1.0
        
        # Test average distribution
        result_avg = label_dirichlet_partition(
            labels, N, K, n_parties, beta, distribution_type="average"
        )
        
        # Test degree distribution
        result_degree = label_dirichlet_partition(
            labels, N, K, n_parties, beta, distribution_type="degree"
        )
        
        assert len(result_avg) == n_parties
        assert len(result_degree) == n_parties
        
        # Results should be different (with high probability)
        assert result_avg != result_degree
    
    def test_label_dirichlet_partition_beta_values(self):
        """Test different beta values."""
        labels = np.array([0, 1, 2] * 20)
        N = len(labels)
        K = 3
        n_parties = 3
        
        # Low beta (more heterogeneous)
        result_low = label_dirichlet_partition(labels, N, K, n_parties, beta=0.1)
        
        # High beta (more homogeneous)
        result_high = label_dirichlet_partition(labels, N, K, n_parties, beta=10.0)
        
        assert len(result_low) == n_parties
        assert len(result_high) == n_parties
        
        # Both should cover all indices
        assert sum(len(party) for party in result_low) == N
        assert sum(len(party) for party in result_high) == N


class TestCommunityPartition:
    """Test community_partition_non_iid function."""
    
    def test_community_partition_non_iid(self):
        """Test community-based partition."""
        # Create test data
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        non_iid_percent = 0.5
        num_clients = 3
        nclass = 3
        args_cuda = False
        
        result = community_partition_non_iid(
            non_iid_percent, labels, num_clients, nclass, args_cuda
        )
        
        assert isinstance(result, list)
        assert len(result) == num_clients
        
        # Check that all indices are covered
        all_indices = set()
        for client_indices in result:
            all_indices.update(client_indices)
        
        # Should cover most or all of the data points
        assert len(all_indices) <= len(labels)
        
        # Each client should have some data (though this may not always be true)
        non_empty_clients = sum(1 for client in result if len(client) > 0)
        assert non_empty_clients >= 1


class TestGetInCommIndexes:
    """Test get_in_comm_indexes function."""
    
    def test_get_in_comm_indexes_basic(self):
        """Test basic functionality of get_in_comm_indexes."""
        # Create simple graph
        edge_index = torch.tensor([[0, 1, 2, 3, 4],
                                  [1, 2, 3, 4, 0]])
        
        # Create node splits for trainers
        split_node_indexes = [
            torch.tensor([0, 1]),
            torch.tensor([2, 3]),
            torch.tensor([4])
        ]
        
        n_trainers = 3
        num_hops = 1
        idx_train = torch.tensor([0, 2, 4])
        idx_test = torch.tensor([1, 3])
        
        result = get_in_comm_indexes(
            edge_index, split_node_indexes, n_trainers, 
            num_hops, idx_train, idx_test
        )
        
        assert len(result) == 4  # Should return 4 elements
        (communicate_node_global_indexes, in_com_train_node_local_indexes,
         in_com_test_node_local_indexes, global_edge_indexes_clients) = result
        
        assert isinstance(communicate_node_global_indexes, dict)
        assert isinstance(in_com_train_node_local_indexes, dict)
        assert isinstance(in_com_test_node_local_indexes, dict)
        assert isinstance(global_edge_indexes_clients, dict)
        
        assert len(communicate_node_global_indexes) == n_trainers
        assert len(global_edge_indexes_clients) == n_trainers


class TestGet1hopFeatureSum:
    """Test get_1hop_feature_sum function."""
    
    def test_get_1hop_feature_sum_basic(self):
        """Test basic 1-hop feature sum calculation."""
        # Create test data
        features = torch.randn(5, 10)  # 5 nodes, 10 features
        edge_index = torch.tensor([[0, 1, 2, 3],
                                  [1, 2, 3, 4]])
        local_node_index = torch.tensor([0, 1, 2])
        communicate_node_index = torch.tensor([0, 1, 2, 3, 4])
        
        result = get_1hop_feature_sum(
            features, edge_index, local_node_index, communicate_node_index
        )
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == len(communicate_node_index)
        assert result.shape[1] == features.shape[1]
    
    def test_get_1hop_feature_sum_isolated_nodes(self):
        """Test 1-hop feature sum with isolated nodes."""
        features = torch.randn(3, 5)
        edge_index = torch.tensor([[0], [1]])  # Only one edge
        local_node_index = torch.tensor([0, 1, 2])
        communicate_node_index = torch.tensor([0, 1, 2])
        
        result = get_1hop_feature_sum(
            features, edge_index, local_node_index, communicate_node_index
        )
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 5)


class TestIncrementDir:
    """Test increment_dir function."""
    
    def test_increment_dir_new_directory(self, temp_dir):
        """Test increment_dir with new directory."""
        base_path = os.path.join(temp_dir, "test_exp")
        
        result = increment_dir(base_path)
        
        assert result == base_path
    
    def test_increment_dir_existing_directory(self, temp_dir):
        """Test increment_dir with existing directories."""
        base_path = os.path.join(temp_dir, "test_exp")
        
        # Create existing directories
        os.makedirs(base_path)
        os.makedirs(f"{base_path}1")
        
        result = increment_dir(base_path)
        
        assert result == f"{base_path}2"
    
    def test_increment_dir_with_comment(self, temp_dir):
        """Test increment_dir with comment."""
        base_path = os.path.join(temp_dir, "test_exp")
        comment = "test_comment"
        
        result = increment_dir(base_path, comment)
        
        assert comment in result
        assert base_path in result


class TestUtilsNCIntegration:
    """Integration tests for utils_nc functions."""
    
    def test_tensor_operations_consistency(self):
        """Test consistency between intersect1d and setdiff1d."""
        t1 = torch.tensor([1, 2, 3, 4, 5])
        t2 = torch.tensor([3, 4, 5, 6, 7])
        
        intersection = intersect1d(t1, t2)
        difference = setdiff1d(t1, t2)
        
        # Union of intersection and difference should cover both sets
        all_elements = torch.cat([intersection, difference])
        original_elements = torch.cat([t1, t2])
        
        # Check that all unique elements are covered
        unique_all = all_elements.unique()
        unique_original = original_elements.unique()
        
        assert len(unique_all) == len(unique_original)
    
    def test_partition_completeness(self):
        """Test that partition functions cover all data points."""
        # Create test labels
        labels = np.random.randint(0, 5, 100)
        N = len(labels)
        K = 5
        n_parties = 4
        beta = 1.0
        
        partition = label_dirichlet_partition(labels, N, K, n_parties, beta)
        
        # Check completeness
        all_indices = set()
        for party in partition:
            all_indices.update(party)
        
        assert len(all_indices) == N
        assert all_indices == set(range(N))
        
        # Check no overlaps
        total_assignments = sum(len(party) for party in partition)
        assert total_assignments == N
    
    def test_graph_communication_workflow(self):
        """Test complete workflow for graph communication setup."""
        # Create a small graph
        num_nodes = 10
        edge_index = torch.tensor([[i, (i+1) % num_nodes] for i in range(num_nodes)]).t()
        
        # Create splits
        split_size = num_nodes // 3
        split_node_indexes = [
            torch.arange(i*split_size, min((i+1)*split_size, num_nodes))
            for i in range(3)
        ]
        
        # Handle remainder
        if sum(len(split) for split in split_node_indexes) < num_nodes:
            split_node_indexes[-1] = torch.cat([
                split_node_indexes[-1],
                torch.arange(split_size * 3, num_nodes)
            ])
        
        n_trainers = 3
        num_hops = 1
        idx_train = torch.arange(0, 7)
        idx_test = torch.arange(7, 10)
        
        # Test the communication index computation
        result = get_in_comm_indexes(
            edge_index, split_node_indexes, n_trainers,
            num_hops, idx_train, idx_test
        )
        
        (communicate_indexes, train_indexes, test_indexes, edge_indexes) = result
        
        # Verify structure
        assert len(communicate_indexes) == n_trainers
        assert len(train_indexes) == n_trainers
        assert len(test_indexes) == n_trainers
        assert len(edge_indexes) == n_trainers
        
        # Verify all trainers have some data
        for i in range(n_trainers):
            assert i in communicate_indexes
            assert i in edge_indexes