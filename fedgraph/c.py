def TrainIterate(self):
    # One iteration of training local client model

    self.Optim.zero_grad()

    y_pred = self.Model(self.graph, self.node_mats)

    t_loss = FedGATLoss(self.LossFn, self.glob_comm, self.loss_weight, y_pred[self.tr_mask], self.labels[self.tr_mask],
                        self.Model, self.global_params, self.duals, self.aug_lagrange_rho, self.dual_weight)

    t_loss.backward()

    self.Optim.step()

    with torch.no_grad():
        v_loss = FedGATLoss(self.LossFn, self.glob_comm, self.loss_weight, y_pred[self.v_mask],
                            self.labels[self.v_mask], self.Model, self.global_params, self.duals, self.aug_lagrange_rho,
                            self.dual_weight)

        pred_labels = torch.argmax(y_pred, dim=1)
        true_labels = torch.argmax(self.labels, dim=1)

        self.t_acc = torch.sum(pred_labels[self.tr_mask] == true_labels[self.tr_mask]) / torch.sum(self.tr_mask)
        self.v_acc = torch.sum(pred_labels[self.v_mask] == true_labels[self.v_mask]) / torch.sum(self.v_mask)

        print(
            "Client {ID}: Epoch {ep}: Train loss: {t_loss}, Train acc: {t_acc}%, Val loss: {v_loss}, Val acc {v_acc}%".format(
                ID=self.id, ep=self.epoch, t_loss=t_loss, t_acc=100 * self.t_acc, v_loss=v_loss,
                v_acc=100 * self.v_acc))

    self.epoch += 1