# Full OGBN-Papers100M Partitioning Feasibility

## Purpose and scope

This document records the data-path audit required before provisioning a
static Ray GPU fleet for a true OGBN-Papers100M run. It is a design report,
not an implementation or a commitment to a particular EC2 topology.

The first target is a 195-trainer, plaintext, `FedAvg`, `num_hops=0`
baseline with fixed training rounds and final test evaluation. It must contain
all 111,059,956 Papers100M nodes, the official train/validation/test splits,
and an explicitly defined subset of graph edges. `num_hops=2` is out of scope
for this first run because the current FedGCN pretraining design creates a
global `N x F` feature tensor per trainer.

## Evidence from the legacy shards

The historical Hugging Face repositories named as 195-way Papers100M shards
are readable, but are not full-data shards:

| Measurement | Result |
| --- | ---: |
| Available legacy shards | 195 / 195 |
| Total feature-file size | 0.738 GiB |
| Local feature rows | 1,546,782 |
| Unique global IDs | 1,546,782 |
| Full Papers100M nodes | 111,059,956 |
| Full-node coverage | 1.3927% |

The IDs span almost the complete global ID range, and the row count closely
matches the approximately 1.5 million labeled arXiv papers in Papers100M.
The old shards therefore contain labeled nodes only. They also omit
`val_labels.pt`, `idx_val.pt`, `global_node_num.pt`, and `class_num.pt`.

The current Hugging Face loader supports these omissions only through
compatibility fallbacks. As a result, a legacy-shard run reports zero-valued
validation metrics. It can be used only for a fixed-round historical proxy,
not for validation-driven convergence or elastic training.

## Current data-path audit

### 1. The current partitioner drops all unlabeled nodes

`label_dirichlet_partition` builds one index list for each label in
`range(K)`. Papers100M nodes with label `-1` are never assigned to any
trainer. This explains the legacy 1.39% coverage.

**Required change:** the full-data partitioner must first assign every global
node to exactly one owner. Label distribution is then a separate constraint on
the labeled nodes; it cannot be the mechanism that assigns all nodes.

### 2. The current central path is not scalable

`data_loader_NC` loads the complete graph, materializes its COO edge index,
creates all partitions in the driver, and passes slices to Ray actors. This
requires the preparation process and Ray head/object store to hold full graph
objects and intermediate copies.

For Papers100M, lower bounds before temporary copies are approximately:

| Object | Lower-bound size |
| --- | ---: |
| `111,059,956 x 128` float32 features | 52.96 GiB |
| `2 x 1,615,685,872` int64 COO edge index | 24.08 GiB |
| int64 labels, if materialized for every node | 0.83 GiB |
| int16 owner array for 195 partitions | 0.21 GiB |

The PyG sparse representation, COO conversion, edge masks, `torch.cat`,
serialization, and Ray object-store copies increase peak memory substantially.
The current loader must therefore not be used as the production partitioner.

### 3. Per-client k-hop extraction repeats global edge work

`get_in_comm_indexes` calls `torch_geometric.utils.k_hop_subgraph` once per
client against the full edge index. With 195 clients, this repeatedly scans or
masks the 1.6 billion-edge graph. It is not viable as a full-data preparation
algorithm.

For the first 0-hop baseline, partition edge assignment must be a one-pass,
chunked operation: read an edge chunk, find the owners of both endpoints, and
write an edge only to the appropriate shard. Cross-partition edges must be
counted and stored as metadata, even if the baseline does not train on them.

### 4. Edge retention is an experimental decision, not an implementation detail

If all nodes are assigned uniformly at random to 195 trainers, an edge has
only about `1 / 195` probability of having both endpoints in the same shard.
The 0-hop baseline would retain roughly 0.51% of edges as local edges. It
would cover all nodes but would not preserve much graph structure.

Before generating the full artifact, choose one of these documented semantics:

1. **Hash or label-balanced ownership:** simple and balanced, but expected to
   retain few local edges. Suitable only when this limitation is explicit.
2. **Graph-aware balanced partitioning:** preserves more internal edges, but
   needs a scalable partitioner and may not realize the requested label-IID
   distribution exactly.
3. **Future multi-hop FedGCN path:** retains boundary edges and exchanges
   features. This is a separate optimization project because the current dense
   pretraining protocol is infeasible at `N = 111M`.

The first full run should use option 1 or 2, with the selected edge-retention
metric recorded in the manifest.

### 5. Existing storage is not suitable for the full artifact

The Hugging Face path downloads one set of `.pt` files per actor and has no
manifest or integrity check. It is appropriate for the small historical proxy
but not for full-data preparation.

Use object storage with per-worker EBS caching, or a shared high-throughput
filesystem. The first implementation should use a versioned manifest and
direct per-actor loading so that the Ray driver never serializes all client
data through its object store.

### 6. Current training has a separate timing risk

In normal NC training, `Server.train()` already aggregates and broadcasts
parameters, after which `run_NC` performs another plaintext aggregation and
broadcast. This duplicate transfer should be corrected and measured before
reporting full-run communication or wall-time results.

## Required partition artifact, version 1

Store a root `manifest.json` and one directory per trainer:

```text
papers100m-195-v1/
  manifest.json
  shards/
    trainer-000/
      metadata.json
      local_node_index.pt
      communicate_node_index.pt
      adj.pt
      features.pt
      train_labels.pt
      val_labels.pt
      test_labels.pt
      idx_train.pt
      idx_val.pt
      idx_test.pt
      global_node_num.pt
      class_num.pt
```

For the 0-hop baseline, the tensor contract is:

- `local_node_index.pt`: sorted global IDs owned by the trainer.
- `communicate_node_index.pt`: the same sorted global IDs.
- `features.pt`: float32 rows in the same order as `local_node_index.pt`.
- `adj.pt`: local, relabeled `2 x E_i` int64 edge index with endpoints in
  `[0, len(local_node_index))`.
- `idx_*`: positions into `features.pt`; labels contain values only for the
  corresponding split positions.
- `global_node_num.pt`: scalar `111,059,956`.
- `class_num.pt`: scalar `172`.

`manifest.json` must include source dataset/version, partition seed and
policy, trainer count, hop semantics, global node and edge totals, per-shard
counts, retained/cross-edge counts, feature dtype/dimension, checksums, and
storage URIs. `metadata.json` repeats the fields needed to validate an
individual shard before loading it.

## Proposed preparation architecture

### Preparation node

Use a dedicated high-memory CPU instance with persistent EBS. This is distinct
from the Ray head used for training. A provisional capacity discussion should
start at 256 GiB RAM and 1 TiB fast EBS, then be replaced by measurements from
the canary and raw-data loader. Do not download or partition Papers100M on the
current T4 instance.

### Chunked algorithm

1. Read the official raw feature, label, split, and edge files from persistent
   storage without converting the whole graph to one PyG object.
2. Build a compact global owner array that assigns every node once.
3. Write feature rows and global IDs to shard-local temporary files in chunks.
4. Stream edges in chunks, route internal edges to the owning shard, and count
   boundary edges.
5. Generate split-local position tensors and their label tensors.
6. Validate counts/checksums, atomically publish the manifest, and upload or
   sync the finished artifact to shared storage.

The full graph may be read once or in bounded chunks, but it must not be copied
once per trainer or sent through Ray during partitioning.

## Arxiv canary before Papers100M

Implement the future artifact writer and loader first on OGBN-Arxiv with five
or ten trainers. The canary passes only when it verifies:

1. Every global node is assigned exactly once.
2. All official train, validation, and test nodes appear in the correct shard
   and local position tensors.
3. `adj.pt` contains only valid local endpoints.
4. Manifest sums equal the source node, edge, and split totals.
5. Trainer loading produces nonempty validation data and one fixed training
   round completes.
6. Data comes from the shard storage path, not from driver-side Ray arguments.

## Implementation sequence and go/no-go criteria

1. Decide the 0-hop edge-retention semantics and expected label distribution.
2. Implement the Arxiv writer, manifest validator, and direct shard loader.
3. Run and inspect the Arxiv canary, including peak RAM/disk measurements.
4. Provision only the high-memory preparation node and run a Papers100M
   read/partition pilot on a controlled subset.
5. Use the pilot's measured peak memory, output size, and largest-shard size
   to select GPU workers and static Ray fleet capacity.
6. Generate and validate all 195 Papers100M shards before provisioning the
   full GPU fleet.

No code or benchmark change is approved by this report. Future changes stay
uncommitted until explicitly requested.
