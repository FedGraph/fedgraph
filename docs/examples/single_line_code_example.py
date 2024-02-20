import sys
sys.path.append('..')
sys.path.append('../fedgraph')
import fedgraph
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default="cora", type=str)

parser.add_argument("-f", "--fedtype", default="fedgcn", type=str)

parser.add_argument("-c", "--global_rounds", default=100, type=int)
parser.add_argument("-i", "--local_step", default=3, type=int)
parser.add_argument("-lr", "--learning_rate", default=0.5, type=float)

parser.add_argument("-n", "--n_trainer", default=2, type=int)
parser.add_argument("-nl", "--num_layers", default=2, type=int)
parser.add_argument("-nhop", "--num_hops", default=2, type=int)
parser.add_argument("-g", "--gpu", action="store_true")  # if -g, use gpu
parser.add_argument("-iid_b", "--iid_beta", default=10000, type=float)

parser.add_argument("-l", "--logdir", default="./runs", type=str)

args = parser.parse_args()

print(fedgraph.FedGCN.FedGCN_Train(args))
