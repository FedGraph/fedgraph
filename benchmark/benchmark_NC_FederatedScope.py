#!/usr/bin/env python3
import warnings, logging
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

import argparse, time, resource, torch, torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import numpy as np

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.fed_runner import FedRunner
from federatedscope.core.data import DummyDataTranslator
from federatedscope.register import register_data, register_model

from fedgraph.utils_nc import label_dirichlet_partition

# DATASETS = ['cora', 'citeseer', 'pubmed']
DATASETS = ['pubmed']

IID_BETAS = [10000.0, 100.0, 10.0]
CLIENT_NUM = 10
TOTAL_ROUNDS = 200
LOCAL_STEPS = 1
LEARNING_RATE = 0.1
HIDDEN_DIM = 64
DROPOUT_RATE = 0.5
CPUS_PER_TRAINER = 0.6
STANDALONE_PROCESSES = 1

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

class TwoLayerGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, HIDDEN_DIM)
        self.conv2 = GCNConv(HIDDEN_DIM, out_channels)
        self.dropout = DROPOUT_RATE

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

def make_data_loader(name):
    def load_data(config, client_cfgs=None):
        ds = Planetoid(root='data/', name=PLANETOID_NAMES[name])
        full = ds[0]
        num_classes = int(full.y.max().item()) + 1
        # Dirichlet partition across all nodes
        split_idxs = label_dirichlet_partition(
            full.y,
            full.num_nodes,
            num_classes,
            config.federate.client_num,
            config.iid_beta,
            config.distribution_type
        )
        parts = []
        for idxs in split_idxs:
            mask = torch.zeros(full.num_nodes, dtype=torch.bool)
            mask[idxs] = True
            parts.append(Data(
                x=full.x, edge_index=full.edge_index, y=full.y,
                train_mask=mask, val_mask=mask, test_mask=mask
            ))
        data_dict = {
            i+1: {'data': parts[i], 'train': [parts[i]], 'val': [parts[i]], 'test': [parts[i]]}
            for i in range(len(parts))
        }
        data_dict[0] = {'data': full, 'train': [full], 'val': [full], 'test': [full]}
        return DummyDataTranslator(config)(data_dict), config
    return load_data

def make_model_builder(name, num_classes):
    key = f'gnn_{name}'
    def build(cfg_model, input_shape):
        if cfg_model.type != key:
            return None
        in_feats = input_shape[0][-1]
        return TwoLayerGCN(in_feats, num_classes)
    return build, key

register_data('cora', make_data_loader('cora'))
builder, mkey = make_model_builder('cora', 7)
register_model(mkey, builder)

def run_fedavg_manual(ds, beta, rounds, clients):
    device = torch.device('cpu')
    ds_obj = Planetoid(root='data/', name=PLANETOID_NAMES[ds])
    data = ds_obj[0].to(device)
    in_channels = data.x.size(1)
    num_classes = int(data.y.max().item()) + 1
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    # Dirichlet partition over all nodes
    split_idxs = label_dirichlet_partition(
        data.y,
        data.num_nodes,
        num_classes,
        clients,
        beta,
        'average'
    )
    client_idxs = []
    train_set = set(train_idx.tolist())
    for idxs in split_idxs:
        ti = [i for i in idxs if i in train_set]
        client_idxs.append(torch.tensor(ti, dtype=torch.long))
    global_model = TwoLayerGCN(in_channels, num_classes).to(device)
    global_params = [p.data.clone() for p in global_model.parameters()]
    t0 = time.time()
    for _ in range(rounds):
        local_params = []
        for cid in range(clients):
            m = TwoLayerGCN(in_channels, num_classes).to(device)
            for p, gp in zip(m.parameters(), global_params): p.data.copy_(gp)
            opt = torch.optim.SGD(m.parameters(), lr=LEARNING_RATE)
            m.train(); opt.zero_grad()
            out = m(data)
            loss = F.cross_entropy(out[client_idxs[cid]], data.y[client_idxs[cid]])
            loss.backward(); opt.step()
            local_params.append([p.data.clone() for p in m.parameters()])
        with torch.no_grad():
            for gp in global_params: gp.zero_()
            for lp in local_params:
                for gp, p in zip(global_params, lp): gp.add_(p)
            for gp in global_params: gp.div_(clients)
    dur = time.time() - t0
    for p, gp in zip(global_model.parameters(), global_params): p.data.copy_(gp)
    global_model.eval()
    with torch.no_grad():
        preds = global_model(data).argmax(dim=1)
        correct = (preds[data.test_mask.nonzero(as_tuple=False).view(-1)] == data.y[data.test_mask.nonzero(as_tuple=False).view(-1)]).sum().item()
        acc = correct / data.test_mask.sum().item()
    total_params = sum(p.numel() for p in global_model.parameters())
    model_size_mb = total_params * 4 / 1024**2
    return acc, model_size_mb, total_params, dur

def run_fedscope_experiment(ds, beta):
    cfg = global_cfg.clone(); cfg.defrost()
    cfg.use_gpu=False; cfg.device=-1; cfg.seed=42
    cfg.federate.mode='standalone'; cfg.federate.client_num=CLIENT_NUM
    cfg.federate.total_round_num=TOTAL_ROUNDS; cfg.federate.make_global_eval=True
    cfg.federate.process_num=STANDALONE_PROCESSES; cfg.federate.num_cpus_per_trainer=CPUS_PER_TRAINER
    cfg.data.root='data/'; cfg.data.type=ds; cfg.data.splitter='dirichlet'
    cfg.iid_beta=beta; cfg.distribution_type='average'
    cfg.dataloader.type='pyg'; cfg.dataloader.batch_size=1
    cfg.model.type=f'gnn_{ds}'; cfg.model.hidden=HIDDEN_DIM
    cfg.model.dropout=DROPOUT_RATE; cfg.model.layer=2; cfg.model.out_channels=7
    cfg.criterion.type='CrossEntropyLoss'; cfg.trainer.type='nodefullbatch_trainer'
    cfg.train.local_update_steps=LOCAL_STEPS; cfg.train.optimizer.lr=LEARNING_RATE
    cfg.train.optimizer.weight_decay=0.0; cfg.eval.freq=1; cfg.eval.metrics=['acc']
    cfg.freeze()
    data_fs, _ = get_data(config=cfg.clone()); full=data_fs[0]['data']
    t0=time.time(); runner=FedRunner(data=data_fs, config=cfg); res=runner.run(); dur=time.time()-t0; mem=peak_memory_mb()
    acc = res.get('server_global_eval', res).get('test_acc', res.get('acc',0.0))
    acc_pct = acc*100 if acc<=1.0 else acc
    model=runner.server.model; tot_params=sum(p.numel() for p in model.parameters())
    msz=tot_params*4/1024**2; comm=calculate_communication_cost(msz,TOTAL_ROUNDS,CLIENT_NUM)
    return {
        'accuracy':acc_pct,
        'total_time':dur,
        'computation_time':dur,
        'communication_cost_mb':comm,
        'peak_memory_mb':mem,
        'avg_time_per_round':dur/TOTAL_ROUNDS,
        'model_size_mb':msz,
        'total_params':tot_params,
        'nodes':full.num_nodes,
        'edges':full.edge_index.size(1)
    }

def run_manual_experiment(ds, beta):
    if ds=='citeseer': nodes,edges=3327,9104
    else: nodes,edges=19717,88648
    acc, msz, tp, dur = run_fedavg_manual(ds, beta, TOTAL_ROUNDS, CLIENT_NUM)
    mem=peak_memory_mb(); comm=calculate_communication_cost(msz,TOTAL_ROUNDS,CLIENT_NUM)
    return {
        'accuracy':acc*100,
        'total_time':dur,
        'computation_time':dur,
        'communication_cost_mb':comm,
        'peak_memory_mb':mem,
        'avg_time_per_round':dur/TOTAL_ROUNDS,
        'model_size_mb':msz,
        'total_params':tp,
        'nodes':nodes,
        'edges':edges
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--use_cluster",action="store_true")
    args=parser.parse_args()

    print("\nDS,IID,BS,Time[s],FinalAcc[%],CompTime[s],CommCost[MB],PeakMem[MB],AvgRoundTime[s],ModelSize[MB],TotalParams")
    for ds in DATASETS:
        for beta in IID_BETAS:
            try:
                print(f"Running {ds} with β={beta}")
                if ds=='cora':
                    metrics=run_fedscope_experiment(ds,beta)
                else:
                    metrics=run_manual_experiment(ds,beta)
                print(f"Dataset: {metrics['nodes']:,} nodes, {metrics['edges']:,} edges")
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

if __name__=='__main__':
    main()