#!/usr/bin/env python
# coding: utf-8
import torch
import torch_geometric
import torch.nn.functional as F

import argparse
# from sklearn.metrics import roc_auc_score
from torchmetrics.functional.retrieval import retrieval_auroc
from torchmetrics.retrieval import RetrievalHitRate

import random
import time
import datetime
from src.utils_lp import get_global_user_item_mapping
from src.trainer_class import Trainer_LP
from src.server_class import Server_LP

torch_geometric.seed.seed_everything(42)

# Create an ArgumentParser object
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parsing arguments')

    # Add arguments
    parser.add_argument('--method', type=str, default='FedLink', help='Specify the method')
    parser.add_argument('--use_buffer', type=bool, default=False, help='Specify whether to use buffer')
    parser.add_argument('--buffer_size', type=int, default=300000, help='Specify buffer size')
    parser.add_argument('--online_learning', type=bool, default=False, help='Specify online learning')
    parser.add_argument('--repeat_time', type=int, default=10, help='Specify repeat time')
    parser.add_argument('--record_results', type=bool, default=False, help='Record model AUC and Running time')

    # Parse the arguments
    args = parser.parse_args()

    print(args)

    method = args.method
    use_buffer = args.use_buffer
    buffer_size = args.buffer_size
    online_learning = args.online_learning
    repeat_time = args.repeat_time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    user_id_mapping, item_id_mapping = get_global_user_item_mapping()
    number_of_users, number_of_items = len(user_id_mapping.keys()), len(item_id_mapping.keys())

    country_codes = ['US', 'BR', 'ID', 'TR', 'JP']  # top 5 biggest country , 'US', 'BR', 'ID', 'TR',
    meta_data = (['user', 'item'], [('user', 'select', 'item'), ('item', 'rev_select', 'user')])

    # 'STFL', 'StaticGNN', '4D-FED-GNN+', 'FedLink'

    for repeat in range(repeat_time):
        number_of_clients = len(country_codes)
        clients = []
        for i in range(number_of_clients):
            clients.append(Trainer_LP(i, country_code=country_codes[i], user_id_mapping=user_id_mapping, 
                                    item_id_mapping=item_id_mapping, number_of_users=number_of_users,
                                    number_of_items=number_of_items, hidden_channels=64, device=device,
                                    use_buffer=use_buffer, buffer_size=buffer_size))
            
        server = Server_LP(number_of_users=number_of_users, number_of_items=number_of_items, meta_data=meta_data)

        global_model_parameter = server.get_model_parameter()

        for i in range(number_of_clients):
            clients[i].load_model_parameter(global_model_parameter)

        start_time = datetime.datetime(2012, 4, 3)

        if online_learning:
            end_time = start_time + datetime.timedelta(days=1)
            prediction_days = 19
            global_rounds = 20
            local_step = 3
        else:
            end_time = start_time + datetime.timedelta(days=19)
            if method == '4D-FED-GNN+':
                start_time = start_time + datetime.timedelta(days=18)
            prediction_days = 1
            global_rounds = 20
            local_step = 3

        start_time_int_format = time.mktime(start_time.timetuple())
        end_time_int_format = time.mktime(end_time.timetuple())

        if args.record_results:
            file_name = f"{method}_buffer_{use_buffer}_{buffer_size}_online_{online_learning}.txt"
            a = open(file_name, "a+")
            time_writer = open("train_time_" + file_name, "a+")
        # from 2012-04-03 to 2012-04-13
        for day in range(prediction_days):
            for i in range(number_of_clients):
                clients[i].get_train_test_data_at_current_time_step(
                    start_time_int_format, end_time_int_format, use_buffer=use_buffer, buffer_size=buffer_size
                )
                clients[i].calculate_traveled_user_edge_indices()
            if online_learning:
                print(f"start training for day {day + 1}")
            else:
                print(f"start training")
            for iteration in range(global_rounds):
                # each client train on local graph
                print(iteration)
                for client_id in range(number_of_clients):
                    current_loss, train_finish_times = clients[client_id].train(local_updates=local_step, use_buffer=use_buffer)
                    if args.record_results:
                        for train_finish_time in train_finish_times:
                            time_writer.write(f"client {str(client_id)} train time {str(train_finish_time)}\n")
                # fedavg
                if method == 'FedLink (OnlyAvgGNN)':
                    gnn_only = True
                else:
                    gnn_only = False
                if method != 'StaticGNN':
                    model_avg_parameter = server.fedavg(clients, gnn_only)

                    server.load_model_parameter(model_avg_parameter, gnn_only)
                    for client_id in range(number_of_clients):
                        clients[client_id].load_model_parameter(model_avg_parameter, gnn_only)
                # if current_loss < 0.01:
                #    break
                avg_auc = 0
                avg_hit_rate = 0
                avg_traveled_user_hit_rate = 0
                for client_id in range(number_of_clients):
                    auc_score, hit_rate, traveled_user_hit_rate = clients[client_id].test(use_buffer=use_buffer)
                    avg_auc += auc_score
                    avg_hit_rate += hit_rate
                    avg_traveled_user_hit_rate += traveled_user_hit_rate
                    print(
                        f"Day {day} client {client_id} auc score: {auc_score} hit rate: {hit_rate} traveled user hit rate: {traveled_user_hit_rate}")
                    # write final test_auc
                    if iteration + 1 == global_rounds and args.record_results:
                        a.write(
                            f"Day {day} client {client_id} final auc score: {auc_score} hit rate: {hit_rate} traveled user hit rate: {traveled_user_hit_rate}\n")
                avg_auc /= number_of_clients
                avg_hit_rate /= number_of_clients

                if online_learning:
                    print(f"Predict Day {day + 1} average auc score: {avg_auc} hit rate: {avg_hit_rate}")
                else:
                    print(f"Predict Day 20 average auc score: {avg_auc} hit rate: {avg_hit_rate}")

            if current_loss >= 0.01:
                print("training is not complete")

            # go to next day
            if method in ['4D-FED-GNN+']:
                start_time += datetime.timedelta(days=1)
                end_time += datetime.timedelta(days=1)

                start_time_int_format = time.mktime(start_time.timetuple())
                end_time_int_format = time.mktime(end_time.timetuple())
            elif method in ['STFL', "StaticGNN", "FedLink"]:
                start_time = start_time
                end_time += datetime.timedelta(days=1)

                start_time_int_format = time.mktime(start_time.timetuple())
                end_time_int_format = time.mktime(end_time.timetuple())
            else:
                print("not implemented")
            if not use_buffer:
                del clients[client_id].train_data
            else:
                del clients[client_id].global_train_data
            del clients[client_id].test_data
        if args.record_results:
            a.close()
            time_writer.close()