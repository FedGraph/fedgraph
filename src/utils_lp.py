import datetime
import time

import pandas as pd
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import HeteroData


def convert_time(time_string: str) -> float:
    current_list = time_string.split()
    ts = datetime.datetime(int(current_list[5]), time.strptime(current_list[1], '%b').tm_mon, int(current_list[2]),
                           int(current_list[3][:2]), int(current_list[3][3:5]), int(current_list[3][6:8]))
    return time.mktime(ts.timetuple())


def load_node_csv(path, index_col, encoder=None):
    df = pd.read_csv(path, index_col=index_col, sep='\t', header=None)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoder is not None:
        unique_place_names = df[6].unique()
        # sentence_transformer_model
        encodes = encoder.encode(unique_place_names)
        unique_place_name_embeddings_dict = dict(
            [(unique_place_names[i], encodes[i]) for i in range(len(unique_place_names))])
        x = np.array([unique_place_name_embeddings_dict[locationid_info[key][3]] for key in mapping])

    return x, mapping


def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None):
    df = pd.read_csv(path, sep='\t', header=None)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = torch.tensor([convert_time(index) for index in df[2]])
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


def get_global_user_item_mapping():
    file_name = f"data_seperated_by_country/raw_Checkins_anonymized_five_countries.txt"
    user_feature, user_id_mapping = load_node_csv(file_name, index_col=0)
    item_feature, item_id_mapping = load_node_csv(file_name, index_col=1)
    return user_id_mapping, item_id_mapping


def get_data(country_code, user_id_mapping = None, item_id_mapping = None):
    file_name = f"data_seperated_by_country/raw_Checkins_anonymized_{country_code}_100day_20day_with_name_location.txt"

    if user_id_mapping is None and item_id_mapping is None:
        # use local client mapping
        user_feature, user_id_mapping = load_node_csv(file_name, index_col=0)
        item_feature, item_id_mapping = load_node_csv(file_name, index_col=1)

    data = HeteroData()

    data["user"].node_id = torch.arange(len(user_id_mapping))
    data["item"].node_id = torch.arange(len(item_id_mapping))

    edge_index, edge_attr = load_edge_csv(
        file_name,
        src_index_col=0,
        src_mapping=user_id_mapping,
        dst_index_col=1,
        dst_mapping=item_id_mapping
    )

    data['user', 'select', 'item'].edge_index = edge_index
    data['user', 'select', 'item'].edge_attr = edge_attr

    data = torch_geometric.transforms.ToUndirected()(data)
    return data


def get_data_by_time_step(data, start_time_int_format, end_time_int_format, constant_edges=-1):
    edge_mask = (start_time_int_format <= data['user', 'select', 'item'].edge_attr) & (
            data['user', 'select', 'item'].edge_attr < end_time_int_format)

    data_current_time_step = HeteroData()
    data_current_time_step["user"].node_id = data["user"].node_id
    data_current_time_step["item"].node_id = data["item"].node_id

    data_current_time_step['user', 'select', 'item'].edge_index = data['user', 'select', 'item'].edge_index[:,
                                                                  edge_mask]

    if constant_edges != -1:
        data_current_time_step['user', 'select', 'item'].edge_index = data_current_time_step[
                                                                          'user', 'select', 'item'].edge_index[:,
                                                                      -constant_edges:]
    data_current_time_step['user', 'select', 'item'].edge_attr = None

    data_current_time_step = torch_geometric.transforms.ToUndirected()(data_current_time_step)

    return data_current_time_step


def transform_negative_sample(data):
    positive_edge_index = data['user', 'select', 'item'].edge_index
    negative_sample_rate = 2
    neg_des_list = []
    src_list = []
    for i in range(int(negative_sample_rate)):
        src, pos_des, neg_des = torch_geometric.utils.structured_negative_sampling(positive_edge_index, num_nodes=data['item'].num_nodes)
        src_list.append(src)
        neg_des_list.append(neg_des)
    negative_edges = torch.stack([torch.cat(src_list, dim=-1), torch.cat(neg_des_list, dim=-1)], dim=0)
    '''
    negative_edges = torch_geometric.utils.negative_sampling(positive_edge_index,
                                                             num_nodes=(data['user'].num_nodes, data['item'].num_nodes),
                                                             num_neg_samples=int(positive_edge_index.shape[1] * negative_sample_rate))
    '''
    data['user', 'select', 'item'].edge_label_index = torch.cat([positive_edge_index, negative_edges], dim=-1)
    data['user', 'select', 'item'].edge_label = torch.cat([torch.ones(positive_edge_index.shape[1]),
                                                           torch.zeros(negative_edges.shape[1])], dim=-1)
    return data

def get_data_loaders_per_time_step(data, start_time_int_format, end_time_int_format, use_buffer=False, buffer_size=None):
    '''
    transform = torch_geometric.transforms.RandomLinkSplit(
        num_val=0,
        num_test=0,
        neg_sampling_ratio=2.0,
        edge_types=('user', 'select', 'item'),
        rev_edge_types=("item", "rev_select", "user"),
    )
    '''
    train_data = get_data_by_time_step(data, start_time_int_format, end_time_int_format)
    if use_buffer:
        # train_data = get_data_by_time_step(data, start_time_int_format, end_time_int_format, constant_edges=320000)
        buffer_train_data_list = []
        for buffer_start in range(0, len(train_data['user', 'select', 'item'].edge_index[0]), buffer_size):
            buffer_end = buffer_start + buffer_size
            Buffer_Train_Graph = HeteroData()
            Buffer_Train_Graph["user"].node_id = train_data["user"].node_id
            Buffer_Train_Graph["item"].node_id = train_data["item"].node_id

            Buffer_Train_Graph['user', 'select', 'item'].edge_index = train_data['user', 'select', 'item'].edge_index[
                                                                      :, buffer_start:buffer_end]
            Buffer_Train_Graph = torch_geometric.transforms.ToUndirected()(Buffer_Train_Graph)
            # buffer_train_data, _, __ = transform(Buffer_Train_Graph)
            buffer_train_data = transform_negative_sample(Buffer_Train_Graph)
            buffer_train_data_list.append(buffer_train_data)
    # add_negative_train_samples=False,
    # train_data, _, __ = transform(train_data)
    train_data = transform_negative_sample(train_data)

    # predict one day after the end time
    start_time_int_format = end_time_int_format

    end_time = datetime.datetime.fromtimestamp(end_time_int_format) + datetime.timedelta(days=1)
    end_time_int_format = time.mktime(end_time.timetuple())

    test_data = get_data_by_time_step(data, start_time_int_format, end_time_int_format)

    # test_data, _, __ = transform(test_data)
    test_data = transform_negative_sample(test_data)
    if use_buffer:
        return train_data, test_data, buffer_train_data_list
    else:
        return train_data, test_data