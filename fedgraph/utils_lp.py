import datetime
import os
import time
import urllib.request
from typing import Any

import gdown
import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import HeteroData


def convert_time(time_string: str) -> float:
    """
    Convert time string to float

    Parameters
    ----------
    time_string: str
        The time string to be converted, from which the year, month, day, hour, minute, and second will be extracted

    Returns
    -------
    float
        The time in float format
    """
    current_list = time_string.split()
    ts = datetime.datetime(
        int(current_list[5]),
        time.strptime(current_list[1], "%b").tm_mon,
        int(current_list[2]),
        int(current_list[3][:2]),
        int(current_list[3][3:5]),
        int(current_list[3][6:8]),
    )
    return time.mktime(ts.timetuple())


def load_node_csv(path: str, index_col: int) -> dict:
    """
    Load node csv file and create a mapping of the index to the node

    Parameters
    ----------
    path: str
        The path to the csv file
    index_col: int
        The index column

    Returns
    -------
    mapping: dict
        The mapping of the index to the node
    """
    df = pd.read_csv(path, index_col=index_col, sep="\t", header=None)
    mapping = {
        index: i for i, index in enumerate(df.index.unique())
    }  # Create a mapping of the index to the node

    return mapping


def load_edge_csv(
    path: str,
    src_index_col: int,
    src_mapping: dict,
    dst_index_col: int,
    dst_mapping: dict,
) -> tuple:
    """
    Load edge csv file

    Parameters
    ----------
    path: str
        The path to the csv file
    src_index_col: int
        The source index column
    src_mapping: dict
        The source mapping
    dst_index_col: int
        The destination index column
    dst_mapping: dict
        The destination mapping

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        [0]: The edge index, which stores the source and destination indices
        [1]: The edge attribute, which stores the time in float format
    """
    df = pd.read_csv(path, sep="\t", header=None)

    src = [
        src_mapping[index] for index in df[src_index_col]
    ]  # map the source index to the node
    dst = [
        dst_mapping[index] for index in df[dst_index_col]
    ]  # map the destination index to the node
    edge_index = torch.tensor([src, dst])  # store the source and destination indices

    edge_attr = torch.tensor(
        [convert_time(index) for index in df[2]]
    )  # store the time in float format

    return edge_index, edge_attr


def get_global_user_item_mapping(global_file_path: str) -> tuple:
    """
    Get the global user and item mapping

    Parameters
    ----------
    file_path: str
        The file path of the global data that contains all the user and item ids

    Returns
    -------
    (dict, dict)
        [0]: The user id mapping
        [1]: The item id mapping
    """
    user_id_mapping = load_node_csv(global_file_path, index_col=0)
    item_id_mapping = load_node_csv(global_file_path, index_col=1)
    return user_id_mapping, item_id_mapping


def get_data(
    country_code: str,
    user_id_mapping: Any = None,
    item_id_mapping: Any = None,
    file_path: Any = None,
) -> HeteroData:
    """
    Get the data for the specified country code

    Parameters
    ----------
    country_code: str
        The country code
    user_id_mapping: list
        The user id mapping
    item_id_mapping: list
        The item id mapping

    Returns
    -------
    data: HeteroData
        The data with nodes and edges extracted and stored

    Notes
    -----
    `HeteroData` is a data class included in `torch.geometric.data` that stores the heterogeneous graph data. It has the following attributes:
    - `node_id` (torch.Tensor): The node indices.
    - `edge_index` (torch.Tensor): The edge indices.
    - `edge_attr` (torch.Tensor): The edge attributes.
    - `edge_label_index` (torch.Tensor): The edge label indices.
    - `edge_label` (torch.Tensor): The edge labels.
    """
    print(f"printing in getdata, path: {file_path}")
    file_name = os.path.join(
        file_path,
        f"data_{country_code}.txt",
    )
    print(f"Loading data in {file_name}")

    if user_id_mapping is None and item_id_mapping is None:
        # use local client mapping
        user_id_mapping = load_node_csv(file_name, index_col=0)
        item_id_mapping = load_node_csv(file_name, index_col=1)

    data = HeteroData()

    data["user"].node_id = torch.arange(
        len(user_id_mapping)
    )  # (0, 1, ..., num_users-1)
    data["item"].node_id = torch.arange(
        len(item_id_mapping)
    )  # (0, 1, ..., num_items-1)

    edge_index, edge_attr = load_edge_csv(
        file_name,
        src_index_col=0,
        src_mapping=user_id_mapping,
        dst_index_col=1,
        dst_mapping=item_id_mapping,
    )

    data["user", "select", "item"].edge_index = edge_index
    data["user", "select", "item"].edge_attr = edge_attr
    data = torch_geometric.transforms.ToUndirected()(
        data
    )  # convert the graph to undirected

    return data


def get_data_by_time_step(
    data: HeteroData,
    start_time_float_format: float,
    end_time_float_format: float,
    constant_edges: int = -1,
) -> HeteroData:
    """
    Get the data within the time range determined by the start and end time.

    Parameters
    ----------
    data: HeteroData
        The data with nodes and edges extracted and stored
    start_time_float_format: float
        The start time in float format
    end_time_float_format: float
        The end time in float format
    constant_edges: int
        The number of constant edges. If -1, all edges within the time range will be kept,
        otherwise only the last `constant_edges` edges will be kept

    Returns
    -------
    data_current_time_step: HeteroData
        The data at the current time step
    """
    edge_attr = data["user", "select", "item"].edge_attr
    edge_selected = (start_time_float_format <= edge_attr) & (
        edge_attr < end_time_float_format
    )  # mask the edges within the time range

    data_current_time_step = HeteroData()
    data_current_time_step["user"].node_id = data["user"].node_id
    data_current_time_step["item"].node_id = data["item"].node_id

    data_current_time_step["user", "select", "item"].edge_index = data[
        "user", "select", "item"
    ].edge_index[
        :, edge_selected
    ]  # only keeps the edges within the time range

    if constant_edges != -1:  # only keep the last `constant_edges` edges
        data_current_time_step[
            "user", "select", "item"
        ].edge_index = data_current_time_step["user", "select", "item"].edge_index[
            :, -constant_edges:
        ]
    data_current_time_step[
        "user", "select", "item"
    ].edge_attr = None  # no need to store the edge attributes

    data_current_time_step = torch_geometric.transforms.ToUndirected()(
        data_current_time_step
    )

    return data_current_time_step


def transform_negative_sample(data: HeteroData) -> HeteroData:
    """
    Transform the data by adding negative samples. The negative samples reverse the edges in the graph.

    Parameters
    ----------
    data: HeteroData
        The data with nodes and edges extracted and stored

    Returns
    -------
    data: HeteroData
        The data with negative samples added
    """
    positive_edge_index = data["user", "select", "item"].edge_index
    negative_sample_rate = 2
    neg_des_list = []
    src_list = []
    for _ in range(int(negative_sample_rate)):
        src, pos_des, neg_des = torch_geometric.utils.structured_negative_sampling(
            positive_edge_index, num_nodes=data["item"].num_nodes
        )  # negative samples are not connected in the original graph
        src_list.append(src)
        neg_des_list.append(neg_des)

    negative_edges = torch.stack(
        [torch.cat(src_list, dim=-1), torch.cat(neg_des_list, dim=-1)], dim=0
    )
    data["user", "select", "item"].edge_label_index = torch.cat(
        [positive_edge_index, negative_edges], dim=-1
    )
    data["user", "select", "item"].edge_label = torch.cat(
        [
            torch.ones(positive_edge_index.shape[1]),
            torch.zeros(negative_edges.shape[1]),
        ],
        dim=-1,
    )
    return data


def get_data_loaders_per_time_step(
    data: HeteroData,
    start_time_float_format: float,
    end_time_float_format: float,
    use_buffer: bool = False,
    buffer_size: int = -1,
) -> tuple:
    """
    Get the data loaders per time step.

    Parameters
    ----------
    data: HeteroData
        The data with nodes and edges extracted and stored
    start_time_float_format: float
        The start time in float format
    end_time_float_format: float
        The end time in float format
    use_buffer: bool
        Whether to use buffer
    buffer_size: int
        The buffer size
    """
    if use_buffer:
        assert buffer_size != -1  # buffer size must be specified

    train_data = get_data_by_time_step(
        data, start_time_float_format, end_time_float_format
    )

    if use_buffer:
        buffer_train_data_list = []
        for buffer_start in range(
            0, len(train_data["user", "select", "item"].edge_index[0]), buffer_size
        ):
            buffer_end = buffer_start + buffer_size
            Buffer_Train_Graph = HeteroData()
            Buffer_Train_Graph["user"].node_id = train_data["user"].node_id
            Buffer_Train_Graph["item"].node_id = train_data["item"].node_id

            Buffer_Train_Graph["user", "select", "item"].edge_index = train_data[
                "user", "select", "item"
            ].edge_index[:, buffer_start:buffer_end]
            Buffer_Train_Graph = torch_geometric.transforms.ToUndirected()(
                Buffer_Train_Graph
            )
            # buffer_train_data, _, __ = transform(Buffer_Train_Graph)
            buffer_train_data = transform_negative_sample(Buffer_Train_Graph)
            buffer_train_data_list.append(buffer_train_data)

    train_data = transform_negative_sample(train_data)

    # predict one day after the end time
    start_time_float_format = end_time_float_format

    end_time = datetime.datetime.fromtimestamp(
        end_time_float_format
    ) + datetime.timedelta(days=1)
    end_time_float_format = time.mktime(end_time.timetuple())

    test_data = get_data_by_time_step(
        data, start_time_float_format, end_time_float_format
    )  # set the next day as the test data
    test_data = transform_negative_sample(test_data)

    if use_buffer:
        return train_data, test_data, buffer_train_data_list
    else:
        return train_data, test_data


def get_start_end_time(online_learning: bool, method: str) -> tuple:
    """
    Determine the start and end time for the conditional info, and prediction range

    Parameters
    ----------
    online_learning: bool
        Whether the learning is online or not
    method: str
        The method used

    Returns
    -------
    (datetime.datetime, datetime.datetime, int, float, float)
        [0]: The start time
        [1]: The end time
        [2]: The prediction days
        [3]: The start time in float format
        [4]: The end time in float format
    """
    start_time = datetime.datetime(2012, 4, 3)
    if (
        online_learning
    ):  # the conditional info contains only one day, and the prediction is for the next 19 days
        end_time = start_time + datetime.timedelta(days=1)
        prediction_days = 19
    else:  # the conditional info contains 19 days, and the prediction is for the next 1 day
        end_time = start_time + datetime.timedelta(days=19)
        start_time = (
            start_time + datetime.timedelta(days=18)
            if method == "4D-FED-GNN+"
            else start_time
        )
        prediction_days = 1
    start_time_float_format = time.mktime(start_time.timetuple())
    end_time_float_format = time.mktime(end_time.timetuple())
    return (
        start_time,
        end_time,
        prediction_days,
        start_time_float_format,
        end_time_float_format,
    )


def to_next_day(
    start_time: datetime.datetime, end_time: datetime.datetime, method: str
) -> tuple:
    """
    Move the start and end time to the next day

    Parameters
    ----------
    start_time: datetime.datetime
        The start time
    end_time: datetime.datetime
        The end time
    method: str
        The method used

    Returns
    -------
    (datetime.datetime, datetime.datetime, float, float)
        [0]: The start time
        [1]: The end time
        [2]: The start time in float format
        [3]: The end time in float format
    """
    if method in ["4D-FED-GNN+"]:
        start_time += datetime.timedelta(days=1)
        end_time += datetime.timedelta(days=1)

        start_time_float_format = time.mktime(start_time.timetuple())
        end_time_float_format = time.mktime(end_time.timetuple())
    elif method in ["STFL", "StaticGNN", "FedLink"]:
        start_time = start_time
        end_time += datetime.timedelta(days=1)

        start_time_float_format = time.mktime(start_time.timetuple())
        end_time_float_format = time.mktime(end_time.timetuple())
    else:
        print("not implemented")

    return start_time, end_time, start_time_float_format, end_time_float_format


def check_data_files_existance(
    country_codes: list,
    dataset_dir_path: str,
) -> None:
    """
    Check if the data files exist

    Parameters
    ----------
    country_codes: list
        The list of country codes
    dataset_dir_path: str, optional
        The directory path of the dataset
    """
    if not os.path.exists(dataset_dir_path):
        print(f"{dataset_dir_path} not exists, creating directory")
        os.makedirs(dataset_dir_path)

    all_files_list = ["traveled_users", "data_global"]
    assert len(country_codes) > 0, "No country codes specified"
    assert all(
        country_code in ["US", "BR", "ID", "TR", "JP"] for country_code in country_codes
    ), "Invalid country code"

    all_files_list += [f"data_{country_code}" for country_code in country_codes]

    for file_name in all_files_list:
        if not os.path.exists(f"{dataset_dir_path}/{file_name}.txt"):
            download_LPDataset(file_name, dataset_dir_path)


def download_LPDataset(file_name: str, dir_path: str) -> None:
    """
    Download the data files

    Parameters
    ----------
    dir_path: str
        The directory path
    file_name: str
        The file name
    """
    assert file_name in [
        "traveled_users",
        "data_US",
        "data_BR",
        "data_ID",
        "data_TR",
        "data_JP",
        "data_global",
    ], "Invalid file name"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    out_file_name = f"{dir_path}/{file_name}.txt"
    if file_name == "traveled_users":
        url = "https://drive.google.com/uc?id=1RUsyGrsz4hmY3OA3b-oqyh5yqlks02-p"
    elif file_name == "data_US":
        url = "https://drive.google.com/uc?id=1g3nH-UovAFwj4wKLTr4rBP18hlBtVJLd"
    elif file_name == "data_BR":
        url = "https://drive.google.com/uc?id=1tg69D1-NSTrKvaAGZELBeECsPh6MAnaS"
    elif file_name == "data_ID":
        url = "https://drive.google.com/uc?id=17EIuBl6rI3LNByamO8Dd-yNMUtIJw4xW"
    elif file_name == "data_TR":
        url = "https://drive.google.com/uc?id=1Tz2ckCrEHy0wn075JRYtKE9MNIyd6nTx"
    elif file_name == "data_JP":
        url = "https://drive.google.com/uc?id=1IPBW4dRYk52x8TahfBqFOh3GdxoYafJ2"
    else:
        url = "https://drive.google.com/uc?id=1CnBlVXqCbfjSswagTci5D7nAqO7laU_J"

    print(f"Downloading {file_name} from {url}...")
    gdown.download(url, out_file_name, quiet=False)
    print(f"Downloaded {file_name} to {out_file_name}")
