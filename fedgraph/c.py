class Server(object):

    def __init__(self, graph, clients, feats, labels, global_model, sample_probab, train_params):

        self.graph = graph

        self.clients = clients

        self.num_layers = train_params['num_layers']

        self.num_local_iters = train_params['num_local_iters']

        self.train_iters = train_params['train_rounds']

        self.gpu = train_params['gpu']

        self.glob_comm = train_params['glob_comm']

        self.optim_kind = train_params['optim_kind']

        self.Optim = None

        self.feats = feats.to(device=device)

        self.labels = labels.to(device=device)

        self.node_mats = {}

        self.sample_probab = sample_probab

        self.Model = global_model

        self.dampening = train_params['dampening']

        self.momentum = train_params['momentum']

        self.glob_comm = train_params['glob_comm']

        self.dual_weight = train_params['dual_weight']

        self.aug_lagrange_rho = train_params['aug_lagrange_rho']

        self.model_regularisation = train_params['model_regularisation']

        self.dual_lr = train_params['dual_lr']

        self.model_lr = train_params['model_lr']

        if train_params['glob_comm'] == 'FedAvg':

            for p in self.Model.parameters():

                p.requires_grad = True

                p.grad = torch.zeros(p.size())

        elif train_params['glob_comm'] == 'ADMM':

            for p in self.Model.parameters():

                p.requires_grad = False

        if self.glob_comm == 'FedAvg':

            if train_params['optim_kind'] == 'Adam':

                self.Optim = torch.optim.Adam(self.Model.parameters(
                ), lr=train_params['model_lr'], weight_decay=self.model_regularisation)

            elif train_params['optim_kind'] == 'SGD':

                self.Optim = torch.optim.SGD(self.Model.parameters(
                ), lr=train_params['model_lr'], momentum=train_params['momentum'], dampening=train_params['dampening'], weight_decay=self.model_regularisation)

        self.Duals = {}

        if train_params['glob_comm'] == 'ADMM':

            self.Duals = {id: copy.deepcopy(self.Model).to(
                device=device) for id in range(len(clients))}

            for id in self.Duals:

                for p in self.Duals[id].parameters():

                    p.requires_grad = False

        # self.LocalModelParams = {id: copy.deepcopy(self.Model).parameters() for id in range(len(clients))}

        self.model_loss_weights = {id: 1. for id in range(len(clients))}

        self.train_params = train_params

        self.max_deg = train_params['max_deg']

        self.optim_reset = train_params['optim_reset']

        self.loss_func = train_params['loss_func']

    def VecGen(self, feats1, feats2, num, dim, deg):

        V = np.random.uniform(-2, 2, (num, dim))

        indices = {}

        while len(indices) < num:

            r = random.randint(0, dim - 1)

            if indices.get(r, None) == None:

                indices.update({r: True})

        index_list = [i for i in indices]

        random.shuffle(index_list)

        Keys = np.zeros((num, dim))

        InterVec = np.zeros((deg + 1, num, dim))

        for i in range(num):

            V[:, index_list[i]] = 0

            V[i, index_list[i]] = np.random.uniform(
                1, 3) * random.sample([-1, 1], 1)[0]

            Keys[i, index_list[i]] = 1

            for j in range(deg + 1):

                InterVec[j, :, index_list[i]] = 0.

                InterVec[j, i, index_list[i]] = 1/V[i, index_list[i]] ** j

        InterMat = np.zeros((deg + 1, dim, dim))

        for i in range(deg + 1):

            for j in range(num):

                InterMat[i, :, :] += np.outer(InterVec[i, j, :], Keys[j, :])

        temp1 = np.random.uniform(-5, 5, dim)

        temp2 = np.random.uniform(-5, 5, dim)

        temp3 = np.random.uniform(-5, 5, dim)

        mask1 = np.zeros(dim)

        for i in range(num):

            mask1 += Keys[i, :] * \
                np.dot(Keys[i, :], temp1)/np.dot(Keys[i, :], Keys[i, :])

        mask1 = temp1 - mask1

        for i in range(deg + 1):

            InterMat[i, :,
                     :] += np.random.uniform(-2, 2) * np.outer(mask1, mask1)

        mask2 = np.zeros(dim)

        mask2 += np.dot(mask1, temp2) * mask1/np.dot(mask1, mask1)

        for i in range(num):

            mask2 += Keys[i, :] * \
                np.dot(Keys[i, :], temp2)/np.dot(Keys[i, :], Keys[i, :])

        mask2 = temp2 - mask2

        K1 = np.zeros(dim)

        for i in range(num):

            K1 += Keys[i, :]

        K1 += np.random.uniform(1, 4) * mask2

        K2 = np.zeros((dim, feats1.shape[1]))

        for i in range(num):

            K2 += np.outer(Keys[i, :], feats2[i, :])

        K2 += np.random.uniform(1, 3) * np.outer(mask2,
                                                 feats2[random.randint(0, num - 1), :])

        M1 = np.zeros((feats1.shape[1], dim))
        M2 = np.zeros((feats2.shape[1], dim))

        for i in range(num):

            M1 += np.outer(feats1[i, :], V[i, :])
            M2 += np.outer(feats2[i, :], V[i, :])

        return M1, M2, K1, K2, InterMat

    # Changed layout of the pretrain_communication algorithm

    def pretrain_communication(self):
        # Now, the function first computes matrices for all nodes, and then distributes them to each client
        # Saves computation

        print("Starting pre-train communication!\n")

        node_mats = {}
        d = self.feats.size()[1]

        # First, compute the size of the largest vector to be used

        if self.gpu:

            degree = self.graph.in_degrees()

            degree = [degree[i].item() for i in range(self.graph.num_nodes())]

            max_deg = max(degree)

            max_allowed_deg = int(self.sample_probab * max_deg)

            for node in range(self.graph.num_nodes()):

                print(str(node) + " ", end='\r')

                neigh = self.graph.predecessors(node)

                sampled_bool = np.array([random.choices([0, 1], [
                                        1 - self.sample_probab, self.sample_probab], k=1)[0] for j in range(len(neigh))])

                sampled_bool = torch.from_numpy(sampled_bool).bool()

                sampled_neigh = neigh[sampled_bool]

                if len(sampled_neigh) < 2:

                    sampled_neigh = neigh

                elif len(sampled_neigh) > max_allowed_deg:

                    sampled_neigh = random.sample(
                        list(sampled_neigh), max_allowed_deg)

                # Obtaining features and stacking them into a numpy array

                feats1 = np.zeros((len(sampled_neigh), d))
                feats2 = np.zeros((len(sampled_neigh), d))

                for i in range(len(sampled_neigh)):

                    feats1[i, :] = self.feats[node, :].cpu().detach().numpy()
                    feats2[i, :] = self.feats[sampled_neigh[i].item(),
                                              :].cpu().detach().numpy()

                    M1, M2, K1, K2, Inter = self.VecGen(feats1, feats2, len(
                        sampled_neigh), max_allowed_deg, self.max_deg)

                node_mats[node] = [torch.from_numpy(M1).float().to(device=device), torch.from_numpy(M2).float().to(
                    device=device), torch.from_numpy(K1).float().to(device=device), torch.from_numpy(K2).float().to(device=device)]

        else:

            for node in range(self.graph.num_nodes()):

                print(node)

                print(str(node) + " ", end='\r')

                neigh = self.graph.predecessors(node)

                sampled_bool = np.array([random.choices([0, 1], [
                                        1 - self.sample_probab, self.sample_probab], k=1)[0] for j in range(len(neigh))])

                sampled_bool = torch.from_numpy(sampled_bool).bool()

                sampled_neigh = neigh[sampled_bool]

                if len(sampled_neigh) < 2:

                    sampled_neigh = neigh

                # Obtaining features and stacking them into a numpy array

                feats1 = np.zeros((len(sampled_neigh), d))
                feats2 = np.zeros((len(sampled_neigh), d))

                for i in range(len(sampled_neigh)):

                    feats1[i, :] = self.feats[node, :].cpu().detach().numpy()
                    feats2[i, :] = self.feats[sampled_neigh[i].item(),
                                              :].cpu().detach().numpy()

                    M1, M2, K1, K2, Inter = self.VecGen(feats1, feats2, len(
                        sampled_neigh), int(3 * len(sampled_neigh)), self.max_deg)

                node_mats[node] = [torch.from_numpy(M1).float().to(device=device), torch.from_numpy(M2).float().to(device=device), torch.from_numpy(
                    K1).float().to(device=device), torch.from_numpy(K2).float().to(device=device), torch.from_numpy(Inter).float().to(device=device)]

        self.node_mats = node_mats

        print("Completed pre-train communication!")

    def distribute_mats(self):

        for id in range(len(self.clients)):

            for nodes in range(self.clients[id].graph.num_nodes()):

                true_node_id = self.clients[id].graph.ndata['_ID'][nodes].item(
                )

                self.clients[id].node_mats[nodes] = self.node_mats[true_node_id]

    def TrainCoordinate(self):  # This has also been changed

        # Changed function a little

        # Computing the weights for each client

        for id in range(len(self.clients)):

            self.model_loss_weights[id] = self.clients[id].graph.ndata['tr_mask'].sum(
            ).item()

        temp = sum(self.model_loss_weights.values())

        for id in self.model_loss_weights:

            self.model_loss_weights[id] /= temp

        # Assigned all the loss weights

        # Send global and dual variables to the clients, give local model weights too

        self.ResetAll(self.Model, self.train_params)

        for id in range(len(self.clients)):

            if self.glob_comm == 'FedAvg':

                self.clients[id].FromServer(copy.deepcopy(self.Model), None)

            elif self.glob_comm == 'ADMM':

                self.clients[id].FromServer(self.Model, self.Duals[id])

        # Now, we can start training!

        print("Starting training!")

        for ep in range(self.train_iters):

            for id in range(len(self.clients)):

                for i in range(self.num_local_iters):

                    self.clients[id].TrainIterate()

            self.TrainUpdate()

            for id in range(len(self.clients)):

                if self.glob_comm == 'ADMM':

                    self.clients[id].FromServer(self.Model, self.Duals[id])

                else:

                    self.clients[id].FromServer(
                        copy.deepcopy(self.Model), None)

                if self.optim_reset:

                    self.clients[id].OptimReset()

            print("Epoch {e} completed!".format(e=ep))

        print("Training completed!")

        return self.Model, self.Duals

    def ResetAll(self, Model, train_params=None):

        if train_params != None:

            self.train_params = train_params

        self.Model = Model

        if self.glob_comm == 'FedAvg':

            for p in self.Model.parameters():

                p.requires_grad = True

                p.grad = torch.zeros(p.size())

            if self.optim_kind == 'Adam':

                self.Optim = torch.optim.Adam(self.Model.parameters(
                ), lr=self.model_lr, weight_decay=self.model_regularisation)

            elif self.optim_kind == 'SGD':

                self.Optim = torch.optim.SGD(self.Model.parameters(
                ), lr=self.model_lr, momentum=self.momentum, dampening=self.dampening, weight_decay=self.model_regularisation)

        if self.glob_comm == 'ADMM':

            for p in self.Model.parameters():

                p.requires_grad = False

            self.Duals = {id: copy.deepcopy(self.Model)
                          for id in range(len(self.clients))}

            for id in range(len(self.clients)):

                for p in self.Duals[id].parameters():

                    p.requires_grad = False

        self.LoadTrainParams()

        for id in range(len(self.clients)):

            self.clients[id].Model = copy.deepcopy(self.Model)

            self.clients[id].train_rounds = self.train_params['train_rounds']

            self.clients[id].num_local_iters = self.train_params['num_local_iters']

            self.clients[id].dual_weight = self.train_params['dual_weight']

            self.clients[id].aug_lagrange_rho = self.train_params['aug_lagrange_rho']

            self.clients[id].dual_lr = self.train_params['dual_lr']

            self.clients[id].model_lr = self.train_params['model_lr']

            self.clients[id].model_regularisation = self.train_params['model_regularisation']

            self.clients[id].optim_kind = self.train_params['optim_kind']

            self.clients[id].momentum = self.train_params['momentum']

            self.clients[id].glob_comm = self.train_params['glob_comm']

            self.clients[id].optim_reset = self.train_params['optim_reset']

            self.clients[id].loss_func = self.train_params['loss_func']

            self.clients[id].OptimReset()

            self.clients[id].TrainModel()

            self.clients[id].epoch = 0

    def LoadTrainParams(self):

        self.train_iters = self.train_params['train_rounds']

        self.num_local_iters = self.train_params['num_local_iters']

        self.dual_weight = self.train_params['dual_weight']

        self.aug_lagrange_rho = self.train_params['aug_lagrange_rho']

        self.dual_lr = self.train_params['dual_lr']

        self.model_lr = self.train_params['model_lr']

        self.model_regularisation = self.train_params['model_regularisation']

        self.optim_kind = self.train_params['optim_kind']

        self.momentum = self.train_params['momentum']

        self.glob_comm = self.train_params['glob_comm']

        self.optim_reset = self.train_params['optim_reset']

        self.loss_func = self.train_params['loss_func']

    def TrainUpdate(self):  # Minr changes, but critical to algorithm working!

        # If it is FedAvg, we need to modify the gradients of the global model and train it accordingly!

        old = copy.deepcopy(self.Model)

        for p in old.parameters():

            p.requires_grad = False

        if self.glob_comm == 'FedAvg':

            self.Optim.zero_grad()

            with torch.no_grad():

                for p in self.Model.parameters():

                    p.grad = torch.zeros(p.size())

                    p.grad.data.zero_()

            with torch.no_grad():

                for id in self.clients:

                    for p, p_id in zip(self.Model.parameters(), self.clients[id].Model.parameters()):

                        p.grad += self.model_loss_weights[id] * p_id.grad

            self.Optim.step()

        elif self.glob_comm == 'ADMM':

            with torch.no_grad():

                for p in self.Model.parameters():

                    p -= p

                for id in self.clients:

                    for p, p_id, dual in zip(self.Model.parameters(), self.clients[id].Model.parameters(), self.Duals[id].parameters()):

                        p += self.model_loss_weights[id] * (
                            p_id - self.dual_weight * dual / self.aug_lagrange_rho)

                for id in self.clients:

                    for p, p_id, dual in zip(self.Model.parameters(), self.clients[id].Model.parameters(), self.Duals[id].parameters()):

                        dual += self.model_loss_weights[id] * \
                            self.dual_lr * self.dual_weight * (p - p_id)

        with torch.no_grad():

            err = 0.

            for p, p_old in zip(self.Model.parameters(), old.parameters()):

                err += torch.sum((p - p_old) ** 2)/torch.numel(p)

            print("Change in model parameters = {E}".format(E=err.item()))
