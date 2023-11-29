from typing import Any

import numpy as np
import torch
from gnn_models import GCN, GCN_arxiv, SAGE_products
from train_func import test, train


class Trainer_General:
    def __init__(
        self,
        rank: int,
        adj: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor,
        idx_train: torch.Tensor,
        idx_test: torch.Tensor,
        args_hidden: int,
        class_num: int,
        device: torch.device,
        args: Any,
    ):
        # from gnn_models import GCN_Graph_Classification
        torch.manual_seed(rank)

        # seems that new trainer process will not inherit sys.path from parent, need to reimport!
        if args.dataset == "ogbn-arxiv":
            self.model = GCN_arxiv(
                nfeat=features.shape[1],
                nhid=args_hidden,
                nclass=class_num,
                dropout=0.5,
                NumLayers=args.num_layers,
            ).to(device)
        elif args.dataset == "ogbn-products":
            self.model = SAGE_products(
                nfeat=features.shape[1],
                nhid=args_hidden,
                nclass=class_num,
                dropout=0.5,
                NumLayers=args.num_layers,
            ).to(device)
        else:
            self.model = GCN(
                nfeat=features.shape[1],
                nhid=args_hidden,
                nclass=class_num,
                dropout=0.5,
                NumLayers=args.num_layers,
            ).to(device)

        self.rank = rank  # rank = client ID

        self.device = device

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=args.learning_rate, weight_decay=5e-4
        )

        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_losses: list = []
        self.train_accs: list = []

        self.test_losses: list = []
        self.test_accs: list = []

        self.adj = adj.to(device)
        self.labels = labels.to(device)
        self.features = features.to(device)
        self.idx_train = idx_train.to(device)
        self.idx_test = idx_test.to(device)

        self.local_step = args.local_step

    @torch.no_grad()
    def update_params(self, params: tuple, current_global_epoch: int) -> None:
        # load global parameter from global server
        self.model.to("cpu")
        for (
            p,
            mp,
        ) in zip(params, self.model.parameters()):
            mp.data = p
        self.model.to(self.device)

    def train(self, current_global_round: int) -> None:
        # clean cache
        torch.cuda.empty_cache()
        for iteration in range(self.local_step):
            self.model.train()

            loss_train, acc_train = train(
                iteration,
                self.model,
                self.optimizer,
                self.features,
                self.adj,
                self.labels,
                self.idx_train,
            )
            self.train_losses.append(loss_train)
            self.train_accs.append(acc_train)

            loss_test, acc_test = self.local_test()
            self.test_losses.append(loss_test)
            self.test_accs.append(acc_test)

    def local_test(self) -> list:
        local_test_loss, local_test_acc = test(
            self.model, self.features, self.adj, self.labels, self.idx_test
        )
        return [local_test_loss, local_test_acc]

    def get_params(self) -> tuple:
        self.optimizer.zero_grad(set_to_none=True)
        return tuple(self.model.parameters())

    def get_all_loss_accuray(self) -> list:
        return [
            np.array(self.train_losses),
            np.array(self.train_accs),
            np.array(self.test_losses),
            np.array(self.test_accs),
        ]

    def get_rank(self) -> int:
        return self.rank
