# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
from time import sleep
import ray
import os
import argparse
from pathlib import Path
import yaml

try: 
    import ogb
except:
    os.system("pip install ogb")

try: 
    import tensorboard
except:
    os.system("pip install tensorboard")


import numpy as np
import torch
import torch_geometric
from torch import Tensor
from torch_geometric.loader import DataLoader

from torch.utils.tensorboard import SummaryWriter

import os

print(os.getcwd())
print(os.listdir())
print(os.listdir('..'))

import sys
#adding additional module folders
sys.path.append(os.path.join(sys.path[0],'src','utility'))
sys.path.append(os.path.join(sys.path[0],'src','data'))


remote = False #false for local simulation

if remote:
    print(os.listdir('modules'))
    # GCN model
    from gnn_models import GCN_Graph_Classification, GCN, GCN_arxiv, SAGE_products, GCN_products
    from train import test, train, Lhop_Block_matrix_train
    from utils import label_dirichlet_partition, parition_non_iid, get_in_comm_indexes, get_in_comm_indexes_BDS_GCN, increment_dir, setdiff1d
    from data_process import generate_data, load_data

else:
    from gnn_models import GCN_Graph_Classification, GCN, GCN_arxiv, SAGE_products, GCN_products
    from train import test, train, Lhop_Block_matrix_train
    from utils import label_dirichlet_partition, parition_non_iid, get_in_comm_indexes, get_in_comm_indexes_BDS_GCN, increment_dir, setdiff1d
    from data_process import generate_data, load_data


# -


class Trainer_General:
    def __init__(self, rank, communicate_index, adj, labels, features, idx_train, idx_test, local_steps, num_layers, args_hidden, class_num, learning_rate, device):
        # from gnn_models import GCN_Graph_Classification
        torch.manual_seed(rank)

        # seems that new trainer process will not inherit sys.path from parent, need to reimport!
        if args.dataset == "ogbn-arxiv":
            self.model = GCN_arxiv(nfeat=features.shape[1],
                nhid=args_hidden,
                nclass=class_num,
                dropout=0.5,
                NumLayers=args.num_layers).to(device)
        elif args.dataset == "ogbn-products":
            self.model = SAGE_products(nfeat=features.shape[1],
                nhid=args_hidden,
                nclass=class_num,
                dropout=0.5,
                NumLayers=args.num_layers).to(device)
        else:
            self.model = GCN(nfeat=in_feat,
                            nhid=args_hidden,
                            nclass=class_num,
                            dropout=0.5,
                            NumLayers=args.num_layers).to(device)
        
        self.rank = rank #rank = client ID
        
        self.device = device
        
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                              lr=learning_rate, weight_decay=5e-4)
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.train_accs = []
        
        self.test_losses = []
        self.test_accs = []
        
        self.adj = adj.to(device)
        self.labels = labels.to(device)
        self.features = features.to(device)
        self.idx_train = idx_train.to(device)
        self.idx_test = idx_test.to(device)
        
        self.local_steps = local_steps

  
    @torch.no_grad()
    def update_params(self, params, current_global_epoch):
        #load global parameter from global server
        self.model.to('cpu')
        for p, mp, in zip(params, self.model.parameters()):
            mp.data = p
        self.model.to(self.device)

    def train(self, current_global_round):
        #clean cache
        torch.cuda.empty_cache()
        for iteration in range(self.local_steps):

            self.model.train()
            
            
            loss_train, acc_train = train(iteration, self.model, self.optimizer, 
                                          self.features, self.adj, self.labels, self.idx_train)
            self.train_losses.append(loss_train)
            self.train_accs.append(acc_train)
        
        
            loss_test, acc_test = local_test_loss, local_test_acc = self.local_test()
            self.test_losses.append(loss_test)
            self.test_accs.append(acc_test)
            

    def local_test(self):
        local_test_loss, local_test_acc = test(self.model, self.features, self.adj, self.labels, self.idx_test)
        return [local_test_loss, local_test_acc]
    
    def get_params(self):
        self.optimizer.zero_grad(set_to_none=True)
        return tuple(self.model.parameters())
    def get_all_loss_accuray(self):
        return [np.array(self.train_losses), np.array(self.train_accs), np.array(self.test_losses), np.array(self.test_accs)]
    def get_rank(self):
        return self.rank



class Server:
    def __init__(self):
        #server model on cpu
        if args.dataset == "ogbn-arxiv":
            self.model = GCN_arxiv(nfeat=features.shape[1],
                nhid=args_hidden,
                nclass=class_num,
                dropout=0.5,
                NumLayers=args.num_layers)
        elif args.dataset == "ogbn-products":
            self.model = SAGE_products(nfeat=features.shape[1],
                nhid=args_hidden,
                nclass=class_num,
                dropout=0.5,
                NumLayers=args.num_layers)
        else: #CORA, CITESEER, PUBMED, REDDIT
            self.model = GCN(nfeat=in_feat,
                            nhid=args_hidden,
                            nclass=class_num,
                            dropout=0.5,
                            NumLayers=args.num_layers)
            
        
        if device.type == 'cpu':
            @ray.remote(num_cpus=0.1, scheduling_strategy='SPREAD')
            class Trainer(Trainer_General):
                def __init__(self, rank, communicate_index, adj, labels, features, idx_train, idx_test, local_step, num_layers, args_hidden, class_num, learning_rate, device):
                    super().__init__(rank, communicate_index, adj, labels, features, idx_train, idx_test, local_step, num_layers, args_hidden, class_num, learning_rate, device)
        
        elif args.dataset == "ogbn-arxiv":
            @ray.remote(num_gpus=0.5, num_cpus=5, scheduling_strategy='SPREAD')
            class Trainer(Trainer_General):
                def __init__(self, rank, communicate_index, adj, labels, features, idx_train, idx_test, local_step, num_layers, args_hidden, class_num, learning_rate, device):
                    
                    super().__init__(rank, communicate_index, adj, labels, features, idx_train, idx_test, local_step, num_layers, args_hidden, class_num, learning_rate, device)
        else:
            @ray.remote(num_gpus=1, num_cpus=10, scheduling_strategy='SPREAD')
            class Trainer(Trainer_General):
                def __init__(self, rank, communicate_index, adj, labels, features, idx_train, idx_test, local_step, num_layers, args_hidden, class_num, learning_rate, device):
                    
                    super().__init__(rank, communicate_index, adj, labels, features, idx_train, idx_test, local_step, num_layers, args_hidden, class_num, learning_rate, device)
                    
        if args.fedtype == 'fedsage+':
            print("running fedsage+")
            features_in_clients = []
            #assume the linear generator learnt the optimal (the average of features of neighbor nodes)
            #gaussian noise

            for i in range(args.n_trainer):
                #orignial features of outside neighbors of nodes in client i
                original_feature_i = features[setdiff1d(split_data_indexes[i], communicate_indexes[i])].clone()
                
                #add gaussian noise to the communicated feature
                gaussian_feature_i = original_feature_i + torch.normal(0, 0.1, original_feature_i.shape).cpu()

                copy_feature = features.clone()

                copy_feature[setdiff1d(split_data_indexes[i], communicate_indexes[i])] = gaussian_feature_i

                features_in_clients.append(copy_feature[communicate_indexes[i]])

            self.trainers = [Trainer.remote(i, communicate_indexes[i], edge_indexes_clients[i],
                                            labels[communicate_indexes[i]], features_in_clients[i],
                                            in_com_train_data_indexes[i], in_com_test_data_indexes[i], args.local_step, args.num_layers, args_hidden, class_num, args.learning_rate, device) for i in range(args.n_trainer)]   
        
        else:
            self.trainers = [Trainer.remote(i, communicate_indexes[i], edge_indexes_clients[i],
                                            labels[communicate_indexes[i]], features[communicate_indexes[i]],
                                            in_com_train_data_indexes[i], in_com_test_data_indexes[i], args.local_step, args.num_layers, args_hidden, class_num, args.learning_rate, device) for i in range(args.n_trainer)]


        self.broadcast_params(-1)
    @torch.no_grad()
    def zero_params(self):
        for p in self.model.parameters():
            p.zero_()

    @torch.no_grad()
    def train(self, current_global_epoch):

        for trainer in self.trainers:
            trainer.train.remote(i)
        params = [trainer.get_params.remote() for trainer in self.trainers]
        self.zero_params()

        while True:
            ready, left = ray.wait(params, num_returns=1, timeout=None)
            if ready:
                for t in ready:
                    for p, mp in zip(ray.get(t), self.model.parameters()):
                        mp.data += p.cpu()
            params = left
            if not params:
                break

        for p in self.model.parameters():
            p /= args.n_trainer
        self.broadcast_params(current_global_epoch)


    def broadcast_params(self, current_global_epoch):
        for trainer in self.trainers:
            trainer.update_params.remote(tuple(self.model.parameters()), current_global_epoch)  # run in submit order
# -



# +
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='cora', type=str)
    
    parser.add_argument('-f', '--fedtype', default='fedgcn', type=str)
    
    parser.add_argument('-c', '--global_rounds',default=100, type=int)
    parser.add_argument('-i', '--local_step',default=3, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.5, type=float)
    
    parser.add_argument('-n', '--n_trainer', default=5, type=int)
    parser.add_argument('-nl', '--num_layers', default=2, type=int)
    parser.add_argument('-nhop', '--num_hops', default=2, type=int)
    parser.add_argument('-g', '--gpu', action='store_true') #if -g, use gpu
    parser.add_argument('-iid_b', '--iid_beta', default=10000, type=float)
    
    parser.add_argument('-l', '--logdir', default='./runs', type=str)

    
    parser.add_argument('-r', '--repeat_time', default=10, type=int)
    args = parser.parse_args()
    print(args)
    
    #'cora', 'citeseer', 'pubmed' #simulate #other dataset twitter, 
    #'ogbn-arxiv', reddit, "ogbn-products"
    np.random.seed(42)
    torch.manual_seed(42)
    
    #load data to cpu
    if args.dataset == 'simulate':
        number_of_nodes=200
        class_num=3
        link_inclass_prob=10/number_of_nodes
        link_outclass_prob=link_inclass_prob/20
        features, adj, labels, idx_train, idx_val, idx_test = generate_data(number_of_nodes,  class_num, link_inclass_prob, link_outclass_prob)               
    else:
        features, adj, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
        class_num = labels.max().item() + 1

    in_feat = features.shape[1]
    if args.dataset in ['simulate', 'cora', 'citeseer', 'pubmed', "reddit"]:
        args_hidden = 16
    else:
        args_hidden = 256

    row, col, edge_attr = adj.coo()
    edge_index = torch.stack([row, col], dim=0)


    #specifying a target GPU
    if args.gpu:
        device = torch.device('cuda')
        #running on a local machine with multiple gpu     
        if args.dataset == 'ogbn-products':
            edge_index = edge_index.to("cuda:7")
        else:
            edge_index = edge_index.to("cuda:0")
    else:
        device = torch.device('cpu')
        
    
    
    #repeat experiments 
    average_final_test_loss_repeats = []
    average_final_test_accuracy_repeats = []
    
    for repeat in range(args.repeat_time):

        #load data to cpu

        #beta = 0.0001 extremly Non-IID, beta = 10000, IID
        split_data_indexes = label_dirichlet_partition(labels, len(labels), class_num, args.n_trainer, beta = args.iid_beta)
        
        for i in range(args.n_trainer):
            split_data_indexes[i] = np.array(split_data_indexes[i])
            split_data_indexes[i].sort()
            split_data_indexes[i] = torch.tensor(split_data_indexes[i])
        
        if args.fedtype == 'bds-gcn':
            print("running bds-gcn")
            #No args.num_hops
            communicate_indexes, in_com_train_data_indexes, in_com_test_data_indexes, edge_indexes_clients = get_in_comm_indexes_BDS_GCN(edge_index, split_data_indexes, args.n_trainer, idx_train, idx_test)
        else:
            communicate_indexes, in_com_train_data_indexes, in_com_test_data_indexes, edge_indexes_clients = get_in_comm_indexes(edge_index, split_data_indexes, args.n_trainer, args.num_hops, idx_train, idx_test)
        

        args.log_dir =  increment_dir(Path(args.logdir) / 'exp')
        os.makedirs(args.log_dir)
        yaml_file = str(Path(args.log_dir) / "args.yaml")
        with open(yaml_file, 'w') as out:
            yaml.dump(args.__dict__, out, default_flow_style=False)


        writer = SummaryWriter(args.log_dir)
        #clear cache
        torch.cuda.empty_cache()
        server = Server()
        print("global_rounds", args.global_rounds)
        for i in range(args.global_rounds):
            server.train(i)


        results = [trainer.get_all_loss_accuray.remote() for trainer in server.trainers]
        results = np.array([ray.get(result) for result in results])

        client_id = 0
        for result in results:
            for iteration in range(len(result[0])):
                writer.add_scalar('Train Loss/Client_{}'.format(client_id), result[0][iteration], iteration)
            for iteration in range(len(result[1])):
                writer.add_scalar('Train Accuracy/Client_{}'.format(client_id), result[1][iteration], iteration)
            for iteration in range(len(result[2])):
                writer.add_scalar('Test Loss/Client_{}.format(client_id)', result[2][iteration], iteration)
            for iteration in range(len(result[3])):
                writer.add_scalar('Test Accuracy/Client_{}'.format(client_id), result[3][iteration], iteration)
            client_id += 1
        #print('finished')

        train_data_weights = [len(i) for i in in_com_train_data_indexes]
        test_data_weights = [len(i) for i in in_com_test_data_indexes]
        average_train_loss = np.average(results[:,0], weights = train_data_weights, axis = 0)
        average_train_accuracy = np.average(results[:,1], weights = train_data_weights, axis = 0)
        average_test_loss = np.average(results[:,2], weights = test_data_weights, axis = 0)
        average_test_accuracy = np.average(results[:,3], weights = test_data_weights, axis = 0)

        for iteration in range(len(results[0][0])):
            writer.add_scalar('Train Loss/Clients_Overall'.format(), average_train_loss[iteration], iteration)
            writer.add_scalar('Train Accuracy/Clients_Overall'.format(), average_train_accuracy[iteration], iteration)
            writer.add_scalar('Test Loss/Clients_Overall'.format(), average_test_loss[iteration], iteration)
            writer.add_scalar('Train Accuracy/Clients_Overall'.format(), average_test_accuracy[iteration], iteration)

        

        
        results = [trainer.local_test.remote() for trainer in server.trainers]
        results = np.array([ray.get(result) for result in results])
        
        average_final_test_loss = np.average(results[:,0], weights = test_data_weights, axis = 0)
        average_final_test_accuracy = np.average(results[:,1], weights = test_data_weights, axis = 0)
        
        print(average_final_test_loss, average_final_test_accuracy)
        
        #sleep(5)  # wait for print message from remote workers
        filename = args.dataset + "_" + args.fedtype + "_" + str(args.num_layers) + "_layer_" + str(args.num_hops) + "_hop_iid_beta_" + str(args.iid_beta) + "_n_trainer_" + str(args.n_trainer) + "_local_step_" + str(args.local_step) + ".txt"
        with open(filename, 'a+') as a:
            a.write(f'{average_final_test_loss} {average_final_test_accuracy}\n')
            average_final_test_loss_repeats.append(average_final_test_loss)
            average_final_test_accuracy_repeats.append(average_final_test_accuracy)
            
    #finish experiments
    with open(f'{args.dataset}_{args.fedtype}_{args.num_layers}_layer_{args.num_hops}_hop_iid_beta_{args.iid_beta}_n_trainer_{args.n_trainer}_local_step_{args.local_step}.txt', 'a+') as a:
        a.write(f'average_testing_loss {np.average(average_final_test_loss_repeats)} std {np.std(average_final_test_loss_repeats)}\n')
        a.write(f'average_testing_accuracy {np.average(average_final_test_accuracy_repeats)} std {np.std(average_final_test_accuracy_repeats)}\n')
        
    print(f'average_testing_loss {np.average(average_final_test_loss_repeats)} std {np.std(average_final_test_loss_repeats)}')
    print(f'average_testing_accuracy {np.average(average_final_test_accuracy_repeats)} std {np.std(average_final_test_accuracy_repeats)}')

ray.shutdown()



# -

