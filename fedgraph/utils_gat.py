import random
from typing import Any


def CreateNodeSplit(graph: Any, num_clients: int) -> dict:
    nodes = [i for i in range(graph.num_nodes)]
    node_split = [random.randint(0, len(nodes))
                  for i in range(num_clients - 1)]
    node_split.sort()
    node_split = [0] + node_split + [len(nodes)]
    random.shuffle(nodes)
    client_nodes = {i: {
        j: True for j in nodes[node_split[i]:node_split[i + 1]]} for i in range(num_clients)}
    for id in client_nodes:
        print("Client {ID} has {num} nodes".format(
            ID=id, num=len(client_nodes[id])))
    return client_nodes
