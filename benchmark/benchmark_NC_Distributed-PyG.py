#!/usr/bin/env python3
import warnings, logging
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

import argparse, time, resource, torch, torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import numpy as np

# Distributed PyG imports
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.models import GCN as PyGGCN
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import os

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

class DistributedGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup_distributed():
    """Cleanup distributed training"""
    destroy_process_group()

def train_client(rank, world_size, data, client_indices, model_state, device):
    """Training function for each client process"""
    # Setup distributed environment
    setup_distributed(rank, world_size)
    
    # Create model and wrap with DDP
    model = DistributedGCN(
        data.x.size(1), 
        HIDDEN_DIM, 
        int(data.y.max().item()) + 1, 
        num_layers=2, 
        dropout=DROPOUT_RATE
    ).to(device)
    
    model = DDP(model, device_ids=None if device.type == 'cpu' else [device])
    model.load_state_dict(model_state)
    
    # Create data loader for this client
    loader = NeighborLoader(
        data,
        input_nodes=client_indices,
        num_neighbors=[10, 10],
        batch_size=512 if len(client_indices) > 512 else len(client_indices),
        shuffle=True
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    
    # Local training
    for epoch in range(LOCAL_STEPS):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            
            # Use only the nodes in the current batch that are in training set
            mask = batch.train_mask[:batch.batch_size]
            if mask.sum() > 0:
                loss = F.cross_entropy(out[:batch.batch_size][mask], batch.y[:batch.batch_size][mask])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
    
    cleanup_distributed()
    return model.module.state_dict()

def run_distributed_pyg_experiment(ds, beta):
    device = torch.device('cpu')  # Use CPU for simplicity
    ds_obj = Planetoid(root='data/', name=PLANETOID_NAMES[ds])
    data = ds_obj[0].to(device)
    in_channels = data.x.size(1)
    num_classes = int(data.y.max().item()) + 1
    
    print(f"Running {ds} with β={beta}")
    print(f"Dataset: {data.num_nodes:,} nodes, {data.edge_index.size(1):,} edges")
    
    # Partition training nodes
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    test_idx = data.test_mask.nonzero(as_tuple=False).view(-1)
    
    client_parts = dirichlet_partition(data.y[train_idx], CLIENT_NUM, beta)
    client_idxs = [train_idx[part] for part in client_parts]
    
    # Initialize global model
    global_model = DistributedGCN(
        in_channels, 
        HIDDEN_DIM, 
        num_classes, 
        num_layers=2, 
        dropout=DROPOUT_RATE
    ).to(device)
    
    t0 = time.time()
    
    # Federated training loop using simulated distributed training
    for round_idx in range(TOTAL_ROUNDS):
        global_state = global_model.state_dict()
        local_states = []
        
        # Simulate distributed training for each client
        for client_id in range(CLIENT_NUM):
            # Create client model
            client_model = DistributedGCN(
                in_channels, 
                HIDDEN_DIM, 
                num_classes, 
                num_layers=2, 
                dropout=DROPOUT_RATE
            ).to(device)
            
            # Load global state
            client_model.load_state_dict(global_state)
            
            # Create client data loader using PyG's NeighborLoader
            client_loader = NeighborLoader(
                data,
                input_nodes=client_idxs[client_id],
                num_neighbors=[10, 10],
                batch_size=min(512, len(client_idxs[client_id])),
                shuffle=True
            )
            
            optimizer = torch.optim.Adam(client_model.parameters(), lr=LEARNING_RATE)
            client_model.train()
            
            # Local training
            for epoch in range(LOCAL_STEPS):
                for batch in client_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    out = client_model(batch.x, batch.edge_index)
                    
                    # Use only the nodes that are actually in training set
                    local_train_mask = torch.isin(batch.n_id[:batch.batch_size], client_idxs[client_id])
                    if local_train_mask.sum() > 0:
                        loss = F.cross_entropy(
                            out[:batch.batch_size][local_train_mask], 
                            batch.y[:batch.batch_size][local_train_mask]
                        )
                        loss.backward()
                        optimizer.step()
            
            local_states.append(client_model.state_dict())
        
        # FedAvg aggregation
        global_state = global_model.state_dict()
        for key in global_state.keys():
            global_state[key] = torch.stack([state[key].float() for state in local_states]).mean(0)
        
        global_model.load_state_dict(global_state)
    
    dur = time.time() - t0
    
    # Final evaluation using NeighborLoader for test set
    global_model.eval()
    test_loader = NeighborLoader(
        data,
        input_nodes=test_idx,
        num_neighbors=[10, 10],
        batch_size=min(1024, len(test_idx)),
        shuffle=False
    )
    
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = global_model(batch.x, batch.edge_index)
            pred = out[:batch.batch_size].argmax(dim=-1)
            correct += (pred == batch.y[:batch.batch_size]).sum().item()
            total += batch.batch_size
    
    accuracy = correct / total * 100
    
    # Calculate metrics
    total_params = sum(p.numel() for p in global_model.parameters())
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
                metrics = run_distributed_pyg_experiment(ds, beta)
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
                print(f"{ds},{beta},-1,0.0,0.00,0.0,0.0,0.0,0.000,0.000,0")

if __name__ == '__main__':
    main()