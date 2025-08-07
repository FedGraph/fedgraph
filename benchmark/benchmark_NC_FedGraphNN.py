#!/usr/bin/env python3
import warnings, logging
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

import argparse, time, resource, torch, torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import numpy as np
from fedml.data.graph.data_loader import GraphDataLoader
from fedml.model.graph.gcn import GCN
from fedml.trainer.graph_trainer import GraphTrainer

DATASETS = ['cora', 'citeseer', 'pubmed']
IID_BETAS = [10000.0, 100.0, 10.0]
CLIENT_NUM = 10
TOTAL_ROUNDS = 200
LOCAL_STEPS = 1
LEARNING_RATE = 0.1
HIDDEN_DIM = 64
DROPOUT_RATE = 0.0

PLANETOID_NAMES = {
    'cora': 'Cora',
    'citeseer': 'CiteSeer',
    'pubmed': 'PubMed'
}

def peak_memory_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (usage / 1024**2) if usage > 1024**2 else (usage / 1024)

def calculate_communication_cost(model_size_mb, rounds, clients):
    cost_per_round = model_size_mb * clients * 2
    return cost_per_round * rounds

def dirichlet_partition(labels, num_clients, alpha):
    """Dirichlet partition for non-IID data distribution"""
    labels = labels.cpu().numpy()
    num_classes = labels.max() + 1
    idx_by_class = [np.where(labels == c)[0] for c in range(num_classes)]
    client_idxs = [[] for _ in range(num_clients)]

    for idx in idx_by_class:
        np.random.shuffle(idx)
        props = np.random.dirichlet([alpha] * num_clients)
        props = (props / props.sum()) * len(idx)
        counts = np.floor(props).astype(int)
        counts[-1] = len(idx) - counts[:-1].sum()
        start = 0
        for i, cnt in enumerate(counts):
            client_idxs[i].extend(idx[start : start + cnt])
            start += cnt

    return [torch.tensor(ci, dtype=torch.long) for ci in client_idxs]

class ManualGCN(torch.nn.Module):
    """Manual GCN implementation"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

class FedMLGraphDataLoader:
    """Custom data loader compatible with FedML-like interface"""
    def __init__(self, data, node_indices, batch_size=-1):
        self.data = data
        self.node_indices = node_indices
        self.batch_size = batch_size if batch_size > 0 else len(node_indices)
        
    def __iter__(self):
        # Return batch data
        batch_data = {
            'x': self.data.x,
            'edge_index': self.data.edge_index,
            'y': self.data.y[self.node_indices],
            'node_indices': self.node_indices
        }
        yield batch_data
        
    def __len__(self):
        return 1

class FedMLGraphTrainer:
    """FedML-like graph trainer"""
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.device = torch.device('cpu')
        
    def get_model_params(self):
        return self.model.cpu().state_dict()
    
    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)
    
    def train(self, train_data, device, args):
        """Train the model"""
        self.model.to(device)
        self.model.train()
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=args.learning_rate)
        
        for batch in train_data:
            x = batch['x'].to(device)
            edge_index = batch['edge_index'].to(device)
            y = batch['y'].to(device)
            node_indices = batch['node_indices'].to(device)
            
            optimizer.zero_grad()
            out = self.model(x, edge_index)
            loss = F.cross_entropy(out[node_indices], y)
            loss.backward()
            optimizer.step()
            
        return len(train_data), loss.item()
    
    def test(self, test_data, device, args):
        """Test the model"""
        self.model.to(device)
        self.model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_data:
                x = batch['x'].to(device)
                edge_index = batch['edge_index'].to(device)
                y = batch['y'].to(device)
                node_indices = batch['node_indices'].to(device)
                
                out = self.model(x, edge_index)
                preds = out[node_indices].argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        accuracy = correct / total if total > 0 else 0
        return total, 0.0, {'accuracy': accuracy}

class Args:
    def __init__(self):
        self.learning_rate = LEARNING_RATE
        self.weight_decay = 0.0

def run_fedml_experiment(ds, beta):
    """Run experiment using FedML-like framework"""
    device = torch.device('cpu')
    ds_obj = Planetoid(root='data/', name=PLANETOID_NAMES[ds])
    data = ds_obj[0].to(device)
    in_channels = data.x.size(1)
    num_classes = int(data.y.max().item()) + 1
    
    print(f"Running {ds} with β={beta}")
    print(f"Dataset: {data.num_nodes:,} nodes, {data.edge_index.size(1):,} edges")
    
    # Partition data
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    test_idx = data.test_mask.nonzero(as_tuple=False).view(-1)
    
    client_parts = dirichlet_partition(data.y[train_idx], CLIENT_NUM, beta)
    client_idxs = [train_idx[part] for part in client_parts]
    
    # Create data loaders
    train_data_list = []
    for c in range(CLIENT_NUM):
        train_loader = FedMLGraphDataLoader(data, client_idxs[c], batch_size=-1)
        train_data_list.append(train_loader)
    
    test_loader = FedMLGraphDataLoader(data, test_idx, batch_size=-1)
    
    # Initialize model and trainers
    model = GCN(in_channels, HIDDEN_DIM, num_classes, dropout=DROPOUT_RATE)
    
    args = Args()
    
    # Create trainers for each client
    trainers = []
    for client_id in range(CLIENT_NUM):
        trainer = FedMLGraphTrainer(model, args)
        trainers.append(trainer)
    
    # Get initial global parameters
    global_params = trainers[0].get_model_params()
    
    t0 = time.time()
    
    # Federated training loop
    for round_idx in range(TOTAL_ROUNDS):
        local_params = []
        
        for client_id in range(CLIENT_NUM):
            # Set global parameters
            trainers[client_id].set_model_params(global_params)
            
            # Local training
            trainers[client_id].train(train_data_list[client_id], device, args)
            
            # Get updated parameters
            local_params.append(trainers[client_id].get_model_params())
        
        # FedAvg aggregation
        global_params = {}
        for key in local_params[0].keys():
            global_params[key] = torch.stack([lp[key].float() for lp in local_params]).mean(0)
    
    dur = time.time() - t0
    
    # Final evaluation
    trainers[0].set_model_params(global_params)
    _, _, test_metrics = trainers[0].test(test_loader, device, args)
    accuracy = test_metrics['accuracy'] * 100
    
    # Calculate metrics
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / 1024**2
    comm_cost = calculate_communication_cost(model_size_mb, TOTAL_ROUNDS, CLIENT_NUM)
    mem = peak_memory_mb()
    
    return {
        'accuracy': accuracy,
        'total_time': dur,
        'computation_time': dur,
        'communication_cost_mb': comm_cost,
        'peak_memory_mb': mem,
        'avg_time_per_round': dur / TOTAL_ROUNDS,
        'model_size_mb': model_size_mb,
        'total_params': total_params,
        'nodes': data.num_nodes,
        'edges': data.edge_index.size(1)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cluster", action="store_true")
    args = parser.parse_args()

    print("\nDS,IID,BS,Time[s],FinalAcc[%],CompTime[s],CommCost[MB],PeakMem[MB],AvgRoundTime[s],ModelSize[MB],TotalParams")
    
    for ds in DATASETS:
        for beta in IID_BETAS:
            try:
                metrics = run_fedml_experiment(ds, beta)
                print(
                    f"{ds},{beta},-1,"
                    f"{metrics['total_time']:.1f},"
                    f"{metrics['accuracy']:.2f},"
                    f"{metrics['computation_time']:.1f},"
                    f"{metrics['communication_cost_mb']:.1f},"
                    f"{metrics['peak_memory_mb']:.1f},"
                    f"{metrics['avg_time_per_round']:.3f},"
                    f"{metrics['model_size_mb']:.3f},"
                    f"{metrics['total_params']}"
                )
            except Exception as e:
                print(f"Error running {ds} with β={beta}: {e}")
                import traceback
                traceback.print_exc()
                print(f"{ds},{beta},-1,0.0,0.00,0.0,0.0,0.0,0.000,0.000,0")

if __name__ == '__main__':
    main()