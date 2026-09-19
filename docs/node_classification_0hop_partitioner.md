# Streaming 0-Hop Node-Classification Partitioner

## Purpose

`scripts/data/partition_ogbn_nc.py` and its helper module in `scripts/data/`
create complete, label-balanced 0-hop FedGraph node-classification shards from
raw OGB CSV or official OGB binary input. It is the preparation path for a
full-node OGBN-Papers100M baseline, not a Ray task and
not a Hugging Face upload tool.

The historical 195-way Papers100M artifacts assigned only labeled nodes. This
writer assigns every global node to exactly one trainer, keeps the official
train/validation/test splits, and records the retained local-edge fraction.

## Ownership and 0-hop semantics

1. Labeled nodes are split label by label using the existing Dirichlet idea.
2. Each trainer has an exact target capacity, differing by at most one node.
   The allocation for a label cannot exceed remaining capacity.
3. Unlabeled nodes fill the remaining slots using a seeded random permutation.
4. An edge is written only when both endpoints have the same owner. Its stored
   endpoints are the local feature-row positions, so no runtime relabeling is
   required for the 0-hop artifact.

With nearly uniform random ownership, a directed edge has approximately
`1 / n_trainer` probability of remaining local. This is an intentional
tradeoff of the label-balanced 0-hop baseline. The artifact still contains all
nodes and all official split nodes, but it does not retain cross-trainer
edges. A graph-aware ownership policy or the future 1/2-hop feature-exchange
path is needed to preserve those edges.

## Input layout

The command expects the standard OGB raw layout:

```text
<dataset-root>/
  raw/
    node-feat.csv.gz
    node-label.csv.gz
    edge.csv.gz
    num-node-list.csv.gz
    num-edge-list.csv.gz
  split/time/
    train.csv.gz
    valid.csv.gz
    test.csv.gz
```

Uncompressed `.csv` files also work. The `--split-name` argument selects a
different directory below `split/` when needed.

Papers100M is normally distributed in OGB's binary layout instead:

```text
<dataset-root>/
  raw/
    data.npz              node_feat, edge_index, num_nodes_list, num_edges_list
    node-label.npz        node_label
  split/time/
    train.csv.gz
    valid.csv.gz
    test.csv.gz
```

Use `--input-format ogb-binary` explicitly for this layout, or rely on the
default `--input-format auto`. The first binary run extracts only those five
`.npy` members to `--binary-cache-dir` (or
`<dataset-root>/.fedgraph-binary-cache` by default). Later runs memory-map the
cached arrays and reuse them. The cache manifest ties it to the archive size,
mtime, and per-member ZIP metadata; use a new cache directory if the source
archives change.

## Output contract

The writer builds `<output-dir>.incomplete/` and atomically renames it only
after all checks pass. A completed artifact contains:

```text
<output-dir>/
  manifest.json
  shards/trainer-000/
    local_node_index.pt
    communicate_node_index.pt
    adj.pt
    features.pt
    idx_train.pt          train_labels.pt
    idx_val.pt            val_labels.pt
    idx_test.pt           test_labels.pt
    global_node_num.pt    class_num.pt
    metadata.json
```

For 0-hop, `local_node_index.pt` and `communicate_node_index.pt` are identical
sorted global node IDs. `features.pt` rows and `adj.pt` use the corresponding
local coordinate system. `manifest.json` records source paths, counts, the
partition seed/policy, per-shard metadata, and optional SHA-256 checksums.

## Local Artifact Benchmarking

The batch-size benchmark accepts `--local-artifact-dir <root>` as a third data
source alongside central in-memory loading and Hugging Face. It is mutually
exclusive with `--use-huggingface` and currently supports only `--num-hops 0`.
The Ray driver does not read or serialize trainer graph tensors. Each trainer
reads and validates only `shards/trainer-<rank>` after Ray places that actor.

Without a placement map, every worker that may host a trainer must be able to
read the complete artifact root at the same absolute path. For the five-trainer
Arxiv pilot, copy the small complete artifact to the same NVMe location on all
workers, for example `/mnt/fedgraph-nvme/artifacts/ogbn-arxiv-5-0hop-pilot`.
For a single-host pilot, head and all trainer actors already share the local
path, so no artifact copy is needed. The head uses that path to validate
`manifest.json` before calling Ray.

With `--local-artifact-rank-hosts`, every worker still uses the same root path,
but needs only `manifest.json` and its assigned `shards/trainer-NNN/`
directories. The benchmark resolves each mapped private IP to a live Ray node
ID and hard-pins the corresponding trainer rank there.

This is deliberately different from the final Papers100M distribution plan:

- Instance-store NVMe is appropriate for splitting, validation, and a
  disposable per-worker cache, but is erased on stop or termination.
- It is not a shared filesystem. The current local loader therefore cannot
  use one preparation node's NVMe directly from a separate Ray worker.
- `stage-local-artifact-shards.sh` sends each worker only `manifest.json` and
  its matching `shards/trainer-NNN/` directories. It is preferred for the
  roughly 55 GiB, 195-way Papers100M artifact; copying the complete root to
  195 workers would require roughly 10.5 TiB of aggregate disk capacity and
  network transfer.

The local loader verifies manifest version, 0-hop semantics, trainer count,
tensor shapes, local adjacency bounds, and split-index bounds. It does not
re-hash every tensor at experiment startup; checksums are generated and
validated by the rank-aware staging helper while files are copied. That helper
therefore requires the default checksummed splitter output.

To run the five-trainer Arxiv artifact across five one-GPU workers, first
stage it at the same NVMe path on the head and every worker:

```bash
export ARTIFACT_DIR=/mnt/fedgraph-nvme/artifacts/ogbn-arxiv-5-0hop-pilot
mkdir -p "$(dirname "$ARTIFACT_DIR")"
rsync -a ~/fedgraph-artifacts/ogbn-arxiv-5-0hop-pilot/ "$ARTIFACT_DIR/"

./scripts/manual_ray/sync-local-artifact.sh \
  --workers-file "$WORKERS_FILE" \
  --source-dir "$ARTIFACT_DIR" \
  --remote-dir "$ARTIFACT_DIR"
```

After syncing the current source tree and starting the existing manual Ray
head/workers, submit the benchmark from the head:

```bash
export RAY_ADDRESS="${HEAD_PRIVATE_IP}:6379"
python benchmark/benchmark_NC_batch_size_convergence.py \
  --dataset ogbn-arxiv \
  --n-trainer 5 \
  --num-hops 0 \
  --local-artifact-dir "$ARTIFACT_DIR" \
  --batch-sizes 16 \
  --rounds 3 \
  --local-step 1 \
  --seeds 42 \
  --iid-betas 10000 \
  --num-layers 2 \
  --gpu \
  --server-device cpu \
  --num-gpus-per-trainer 1 \
  --num-cpus-per-trainer 1 \
  --evaluation-split validation \
  --resource-monitor-mode manual \
  --resource-snapshot-interval-rounds 1 \
  --experiment-name nc_arxiv_5trainer_0hop_local_artifact_gpu_smoke
```

This is an external trainer-owned artifact source, not a new `data_loader_NC`
central-loading path. A future storage integration can either mount the root and load one shard per
trainer directly, or upload the contents of each `trainer-XXX` directory to
its own Hugging Face dataset repository for the existing legacy loader.

### Rank-aware staging for Papers100M

For a distributed artifact, create one `<rank> <private-ipv4>` pair per
trainer. A host may appear more than once only if it has resources for those
multiple trainer actors.

```bash
cat > /tmp/fedgraph-rank-hosts.txt <<'EOF'
# trainer-rank private-worker-ip
0 10.195.25.12
1 10.195.25.14
2 10.195.25.15
3 10.195.25.16
4 10.195.25.28
5 10.195.25.33
6 10.195.25.35
7 10.195.25.38
8 10.195.25.42
9 10.195.25.61
EOF
```

After syncing the source tree and starting all mapped Ray workers, stage only
the matching shards. `ARTIFACT_DIR` must have the same absolute value on the
head and workers. The head retains the full preparation artifact; each worker
gets `manifest.json` plus only the shard directories assigned to its IP.

```bash
export ARTIFACT_DIR=/mnt/fedgraph-nvme/artifacts/papers100m-10-0hop-v1

./scripts/manual_ray/stage-local-artifact-shards.sh \
  --rank-hosts-file /tmp/fedgraph-rank-hosts.txt \
  --source-dir "$ARTIFACT_DIR" \
  --remote-dir "$ARTIFACT_DIR"
```

The helper verifies every staged `.pt` file against the source shard metadata
and SHA-256 digests. Submit a short fixed-round validation run from the head:

```bash
./scripts/manual_ray/submit-local-artifact-0hop.sh \
  --head-ip "$HEAD_PRIVATE_IP" \
  --artifact-dir "$ARTIFACT_DIR" \
  --rank-hosts-file /tmp/fedgraph-rank-hosts.txt \
  --dataset ogbn-papers100M \
  --n-trainer 10 \
  --rounds 3 \
  --batch-size 16 \
  --local-step 1 \
  --evaluation-split validation \
  --resource-snapshot-interval-rounds 1 \
  --experiment-name nc_papers100m_10trainer_0hop_local_artifact_smoke
```

The splitter itself never uploads to Hugging Face. Use the separate publisher
described below only after the checksummed artifact and staged run are verified.
EBS can retain the complete preparation artifact, but an EBS volume attached to
the head is not visible to workers; use staged shards, a shared filesystem, or
a published source for distributed training.

## Commands

First validate the full flow on OGBN-Arxiv. The command below is the dry run
already used for this implementation; it completed in about 14 seconds with a
peak process RSS of about 0.81 GiB on the current machine.

```bash
conda activate fedgraph312
python scripts/data/partition_ogbn_nc.py \
  --dataset-root dataset/ogbn_arxiv \
  --output-dir /tmp/fedgraph-arxiv-5-0hop \
  --n-trainer 5 \
  --iid-beta 10000 \
  --seed 42 \
  --chunk-rows 25000 \
  --no-checksums
```

For the full Papers100M run, use a dedicated preparation machine with enough
persistent storage for the raw dataset, temporary local shard files, and the
completed artifact. Do not run this on the current 31 GiB root disk.

```bash
python scripts/data/partition_ogbn_nc.py \
  --dataset-root /mnt/papers100m/ogbn_papers100M \
  --input-format ogb-binary \
  --binary-cache-dir /mnt/fedgraph-cache/papers100m-ogb-binary \
  --output-dir /mnt/fedgraph-artifacts/papers100m-195-0hop-v1 \
  --n-trainer 195 \
  --iid-beta 10000 \
  --seed 42 \
  --chunk-rows 100000
```

`--chunk-rows` bounds feature/edge reader and routing memory for either input
format. Lower it if the preparation host approaches RAM pressure. Checksums are enabled by default;
`--no-checksums` is useful for a short pilot but should not be used for the
published full artifact. If a run stops after the persistent owner plan has
been made, rerun the same command with `--resume`; it reuses the deterministic owner plan.

## Hugging Face Artifact Publishing and Loading

The existing `--use-huggingface` path is unchanged: it continues to load the
historical format of one flat Hugging Face dataset repository per trainer. The
new splitter artifact has a separate, explicit path:

```text
FedGraph/<artifact-repository>/
  manifest.json
  shards/trainer-000/...
  shards/trainer-001/...
```

Publish a checksummed, locally validated artifact with the standalone tool. It
uses the current Hugging Face login or `HF_TOKEN`; it does not call the legacy
FedGraph upload helper and never reconstructs a central feature matrix.

```bash
python scripts/data/upload_nc_artifact_to_hf.py \
  --artifact-dir "$ARTIFACT_DIR" \
  --repo-id FedGraph/fedgraph_ogbn-papers100M_195trainer_0hop_iid_beta_10000.0_v1 \
  --private \
  --num-workers 4
```

The publisher checks the manifest, every declared file size, and every
splitter SHA-256 digest before uploading. Keep the repository private until a
local training run has passed. `--skip-checksum-verification` exists only for
small prototype artifacts created with `--no-checksums`.

To load this repository, use `--hf-local-artifact-repo`, not
`--use-huggingface`. Each trainer downloads only `manifest.json` and its own
`shards/trainer-NNN/` files through the normal Hugging Face cache. Set the
cache root to NVMe separately on every worker:

```bash
python benchmark/benchmark_NC_batch_size_convergence.py \
  --dataset ogbn-papers100M \
  --n-trainer 195 \
  --num-hops 0 \
  --hf-local-artifact-repo FedGraph/fedgraph_ogbn-papers100M_195trainer_0hop_iid_beta_10000.0_v1 \
  --hf-local-artifact-cache-dir /mnt/fedgraph-nvme/hf-cache \
  --batch-sizes 512 \
  --rounds 100 \
  --local-step 3 \
  --iid-betas 10000 \
  --seeds 42 \
  --num-layers 2 \
  --gpu \
  --server-device cpu \
  --num-gpus-per-trainer 1 \
  --num-cpus-per-trainer 1
```

This manifest-style loader currently supports validated standard FedAvg 0-hop
artifacts only. It does not use local rank-aware staging because the trainer
retrieves its own rank's shard wherever Ray schedules it.
