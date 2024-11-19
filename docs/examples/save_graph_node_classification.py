import argparse
from io import BytesIO

import numpy as np
import torch
import torch_geometric
# from huggingface_hub import HfApi, HfFolder

from fedgraph.utils_nc import get_in_comm_indexes


def label_dirichlet_partition(
    labels: np.array, N: int, K: int, n_parties: int, beta: float
) -> list:
    min_size = 0
    min_require_size = 10

    split_data_indexes = []

    # Separate the indices of nodes with label -1
    idx_minus_one = np.where(labels == -1)[0]
    np.random.shuffle(idx_minus_one)
    split_minus_one = np.array_split(idx_minus_one, n_parties)

    while min_size < min_require_size:
        idx_batch: list[list[int]] = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))

            proportions = np.array(
                [
                    p * (len(idx_j) < N / n_parties)
                    for p, idx_j in zip(proportions, idx_batch)
                ]
            )

            proportions = proportions / proportions.sum()

            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    # Append the indices of nodes with label -1 to the respective groups
    for j in range(n_parties):
        idx_batch[j] = np.concatenate((idx_batch[j], split_minus_one[j]))
        np.random.shuffle(idx_batch[j])
        split_data_indexes.append(idx_batch[j])
    return split_data_indexes


def save_trainer_data_to_hugging_face(
    trainer_id,
    local_node_index,
    communicate_node_index,
    adj,
    train_labels,
    test_labels,
    features,
    idx_train,
    idx_test,
):
    repo_name = f"FedGraph/fedgraph_{args.dataset}_{args.n_trainer}trainer_{args.num_hops}hop_iid_beta_{args.iid_beta}_trainer_id_{trainer_id}"
    user = HfFolder.get_token()

    api = HfApi()
    try:
        api.create_repo(
            repo_id=repo_name, token=user, repo_type="dataset", exist_ok=True
        )
    except Exception as e:
        print(f"Failed to create or access the repository: {str(e)}")
        return

    def save_tensor_to_hf(tensor, file_name):
        buffer = BytesIO()
        torch.save(tensor, buffer)
        buffer.seek(0)
        api.upload_file(
            path_or_fileobj=buffer,
            path_in_repo=file_name,
            repo_id=repo_name,
            repo_type="dataset",
            token=user,
        )

    save_tensor_to_hf(local_node_index, "local_node_index.pt")
    save_tensor_to_hf(communicate_node_index, "communicate_node_index.pt")
    save_tensor_to_hf(adj, "adj.pt")
    save_tensor_to_hf(train_labels, "train_labels.pt")
    save_tensor_to_hf(test_labels, "test_labels.pt")
    save_tensor_to_hf(features, "features.pt")
    save_tensor_to_hf(idx_train, "idx_train.pt")
    save_tensor_to_hf(idx_test, "idx_test.pt")

    print(f"Uploaded data for trainer {trainer_id}")


def save_all_trainers_data(
    split_node_indexes,
    communicate_node_indexes,
    edge_indexes_clients,
    labels,
    features,
    in_com_train_node_indexes,
    in_com_test_node_indexes,
    n_trainer,
):
    for i in range(n_trainer):
        save_trainer_data_to_hugging_face(
            trainer_id=i,
            local_node_index=split_node_indexes[i],
            communicate_node_index=communicate_node_indexes[i],
            adj=edge_indexes_clients[i],
            train_labels=labels[communicate_node_indexes[i]][
                in_com_train_node_indexes[i]
            ],
            test_labels=labels[communicate_node_indexes[i]][
                in_com_test_node_indexes[i]
            ],
            features=features[split_node_indexes[i]],
            idx_train=in_com_train_node_indexes[i],
            idx_test=in_com_test_node_indexes[i],
        )


def FedGCN_load_data(dataset_str: str) -> tuple:
    if dataset_str in [
        "ogbn-arxiv",
        "ogbn-products",
        "ogbn-papers100M",
    ]:  # 'ogbn-mag' is heteregeneous
        from ogb.nodeproppred import PygNodePropPredDataset

        # Download and process data at './dataset/.'

        dataset = PygNodePropPredDataset(
            name=dataset_str, transform=torch_geometric.transforms.ToSparseTensor()
        )

        split_idx = dataset.get_idx_split()
        idx_train, idx_val, idx_test = (
            split_idx["train"],
            split_idx["valid"],
            split_idx["test"],
        )

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        data = dataset[0]

        features = data.x
        print(features.shape)
        labels = data.y.reshape(-1)
        if dataset_str == "ogbn-arxiv":
            adj = data.adj_t.to_symmetric()
        else:
            adj = data.adj_t
    return features.float(), adj, labels, idx_train, idx_val, idx_test


def run():
    features, adj, labels, idx_train, idx_val, idx_test = FedGCN_load_data(args.dataset)
    class_num = int(np.nanmax(labels)) + 1
    print("class_num", class_num)
    labels[torch.isnan(labels)] = -1
    labels = labels.long()

    row, col, edge_attr = adj.coo()
    edge_index = torch.stack([row, col], dim=0)

    print(f"gpu usage: {args.gpu}")
    if args.gpu:
        edge_index = edge_index.to("cuda")

    split_node_indexes = label_dirichlet_partition(
        labels, len(labels), class_num, args.n_trainer, beta=args.iid_beta
    )

    for i in range(args.n_trainer):
        split_node_indexes[i] = np.array(split_node_indexes[i])
        print(split_node_indexes[i].shape)
        split_node_indexes[i].sort()
        split_node_indexes[i] = torch.tensor(split_node_indexes[i])

    (
        communicate_node_indexes,
        in_com_train_node_indexes,
        in_com_test_node_indexes,
        edge_indexes_clients,
    ) = get_in_comm_indexes(
        edge_index,
        split_node_indexes,
        args.n_trainer,
        args.num_hops,
        idx_train,
        idx_test,
    )
    save_all_trainers_data(
        split_node_indexes=split_node_indexes,
        communicate_node_indexes=communicate_node_indexes,
        edge_indexes_clients=edge_indexes_clients,
        labels=labels,
        features=features,
        in_com_train_node_indexes=in_com_train_node_indexes,
        in_com_test_node_indexes=in_com_test_node_indexes,
        n_trainer=args.n_trainer,
    )


np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default="ogbn-arxiv", type=str)

parser.add_argument("-n", "--n_trainer", default=5, type=int)
parser.add_argument("-g", "--gpu", action="store_true")  # if -g, use gpu
parser.add_argument("-iid_b", "--iid_beta", default=10000, type=float)
parser.add_argument("-nhop", "--num_hops", default=1, type=int)

args = parser.parse_args()


if __name__ == "__main__":
    run()
