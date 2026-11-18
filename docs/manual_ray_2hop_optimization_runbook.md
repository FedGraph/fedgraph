# Manual Ray 2-Hop Optimization Runbook

This runbook covers plaintext FedGraph NC comparisons for the three 2-hop
optimization components:

1. CPU or mmap graph storage with GPU mini-batch training.
2. Bounded trainer/server pretraining feature aggregation.
3. Optional shared-PCA feature compression.

Start with the [experiment plan](2hop_optimization_status.md#experiment-plan).
Part A runs Arxiv on one GPU host. Part B prepares a ten-worker Papers100M
fleet and runs the cases selected in that plan.

This document does not replace `manual_ray_papers100m_runbook.md`. That file
remains the proven 0-hop reference. Commands here use a manifest-style v2/v3
2-hop artifact and `submit-local-artifact-2hop.sh`; never use the 0-hop submit
helper for these runs.

## Campaign Layout

From the repo root, choose one campaign ID and reuse it in the Arxiv and
Papers100M shells, including inside tmux. Use a fresh ID for a new campaign.

```bash
export NC_2HOP_RESULTS="$PWD/benchmark/results/nc_2hop_optimization"
export NC_2HOP_CAMPAIGN="$(date -u +%Y%m%dT%H%M%SZ)"
printf 'Campaign ID %s\n' "$NC_2HOP_CAMPAIGN"
```

Each suite answers one comparison question. Each case receives its own
benchmark manifest, summary, per-run logs, and GPU telemetry.

```text
benchmark/results/nc_2hop_optimization/
  ogbn-arxiv/<campaign>/
    01-artifact-smoke/v2-full-cpu-none/
    02-rank-quality/v3-chunked-cpu-pca64-seed43/
  ogbn-papers100M/<campaign>/
    01-fleet-smoke/v3-chunked-mmap-pca16/
    02-storage/v3-chunked-cpu-pca16/
    03-aggregation/v2-full-cpu-pca16/
```

Pass the suite directory as `--output-root` and the case label as
`--experiment-name`. Use a suffix such as `-rep02` to preserve an earlier
attempt, including failures. Keep artifacts and runtime caches on the
designated storage paths. Historical results remain in
`benchmark/results/nc_batch_size_convergence/`.

## Comparison Contract

Keep these settings fixed while comparing optimization modes:

* Plaintext FedGCN, `--num-hops 2`, two GNN layers, and indexed feature upload.
* The same edges and owned nodes, rank-host map, seed, batch sizes, local steps,
  round budget, and evaluation split for matched cases. Derive v2 from the
  same v3 source and use v2 on both sides of the aggregation comparison.
* A CPU Ray head and GPU mini-batch training.
* Manual resource monitoring plus the experiment-scoped `nvidia-smi` logger.
* Local artifacts staged to worker NVMe. Hugging Face is used only to archive
  or restore the complete artifact on the head; trainers never download shards
  from Hugging Face during a timed comparison.

The supported combinations are:

| Label | Artifact | Graph storage | Pretraining | Compression |
|---|---|---|---|---|
| `full-device-none` | v2 | `device` | `full` | `none` |
| `full-cpu-none` | converted v2 | `cpu` | `full` | `none` |
| `full-cpu-pcaR` | converted v2 | `cpu` | `full` | `shared-pca`, rank R |
| `chunked-cpu-none` | compact v3 | `cpu` | `chunked` | `none` |
| `chunked-cpu-pcaR` | compact v3 | `cpu` | `chunked` | `shared-pca`, rank R |
| `chunked-mmap-none` | compact v3 | `mmap` | `chunked` | `none` |
| `chunked-mmap-pcaR` | compact v3 | `mmap` | `chunked` | `shared-pca`, rank R |

Do not attempt `device + chunked`, `device + shared-pca`, or `mmap + full`.
The benchmark rejects those combinations intentionally.

### Adjacency Formats

| Artifact version | Pretraining adjacency | Training adjacency |
|---|---|---|
| v2 `dual-int64` | `adj_global.pt` with global dataset IDs | `adj.pt` with positions in `communicate_node_index.pt` |
| v3 `local-int32` | Map the current edge batch through `communicate_node_index.pt` | The stored local `adj.pt`, expanded to `int64` at the training transition |

V2 stores two `int64` endpoints per edge in each of two files. V3 stores two
`int32` endpoints in one file, so the stored adjacency payload falls from
32 to 8 bytes per edge. For example, with communication IDs `[7, 20, 55]`,
the v3 local edge `(0, 2)` maps to global edge `(7, 55)`. Source degree and
offset sidecars let the chunked path find edges for the requested global
rows. The sampler still builds indexes using the local `int64` training
view.

`full + cpu` uses the existing full indexed trainer kernel. It expects
global endpoints for the whole shard to find owned feature rows and produce
global row IDs. Convert v3 to v2 offline in bounded edge batches for this
baseline. The trainer loads `adj_global.pt` for pretraining, releases it,
then loads the precomputed local `adj.pt` for training. The full aggregation
still holds complete feature payloads and needs enough RAM on both workers
and the head.

## Part A: Arxiv Single-Host Verification

### Prepare The Environment And Dataset

Run on the existing GPU host:

```bash
cd /home/ubuntu/fedgraph
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
  source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedgraph312
python -m pip install -e . --no-deps

python - <<'PY'
from ogb.nodeproppred import PygNodePropPredDataset

dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="dataset")
print(dataset)
PY
```

The expected raw root is normally `dataset/ogbn_arxiv`. Confirm it before
partitioning:

```bash
export ARXIV_ROOT=/home/ubuntu/fedgraph/dataset/ogbn_arxiv
find "$ARXIV_ROOT" -maxdepth 3 -type f | head
```

### Prepare Matched Five-Trainer Artifacts

Partition once to compact v3 and derive v2 from it. This keeps the edges,
features, and train/validation/test splits identical across the smoke cases.
Use fresh paths or reuse an existing validated pair with matching provenance.

```bash
export N_TRAINER=5
export ARXIV_ARTIFACT_ROOT="$HOME/fedgraph-artifacts/arxiv-2hop-$NC_2HOP_CAMPAIGN"
export ARXIV_V3="$ARXIV_ARTIFACT_ROOT/v3"
export ARXIV_V2="$ARXIV_ARTIFACT_ROOT/v2-derived"
mkdir -p "$ARXIV_ARTIFACT_ROOT"

python scripts/data/partition_ogbn_nc.py \
  --dataset-root "$ARXIV_ROOT" \
  --output-dir "$ARXIV_V3" \
  --n-trainer "$N_TRAINER" \
  --num-hops 2 \
  --adjacency-layout local-int32 \
  --iid-beta 10000 \
  --seed 42 \
  --chunk-rows 25000

python scripts/data/convert_nc_artifact_v3_to_v2.py \
  --source-dir "$ARXIV_V3" --output-dir "$ARXIV_V2"
sha256sum "$ARXIV_V3/manifest.json" "$ARXIV_V2/manifest.json"
```

### Run The Local Comparison Function

Stop any old external Ray cluster so `ray.init()` creates a local runtime that
can advertise this host's GPU:

```bash
unset RAY_ADDRESS
ray stop --force || true
```

Define worker-local output roots. They may use EBS for Arxiv; Papers100M must
use NVMe.

```bash
export ARXIV_RUNTIME="$HOME/fedgraph-2hop-runtime/arxiv/$NC_2HOP_CAMPAIGN"
export ARXIV_GPU_LOGS="$HOME/fedgraph-logs/arxiv-2hop/$NC_2HOP_CAMPAIGN"
export ARXIV_OUTPUT_ROOT="$NC_2HOP_RESULTS/ogbn-arxiv/$NC_2HOP_CAMPAIGN"
```

Paste this function into the same shell. It always enables application
snapshots and wraps the run in the external GPU sampler:

```bash
run_arxiv_2hop() {
  local suite="$1" name="$2" artifact="$3"
  local storage="$4" aggregation="$5" compression="$6" rank="${7:-16}"
  local output_root="$ARXIV_OUTPUT_ROOT/$suite"
  local result_dir="$output_root/$name"
  local runtime_dir="$ARXIV_RUNTIME/$suite/$name"
  local monitor_dir="$ARXIV_GPU_LOGS/$suite"
  if [[ -e "$result_dir" ]]; then
    printf 'Case directory already exists: %s\n' "$result_dir" >&2
    return 1
  fi
  local -a command=(
    python benchmark/benchmark_NC_batch_size_convergence.py
    --dataset ogbn-arxiv
    --n-trainer 5
    --num-hops 2
    --local-artifact-dir "$artifact"
    --batch-sizes "${BATCH_SIZE:-16}"
    --evaluation-batch-size 512
    --rounds "${ROUNDS:-3}"
    --local-step "${LOCAL_STEP:-1}"
    --iid-betas 10000
    --seeds "${SEED:-42}"
    --num-layers 2
    --gpu
    --server-device cpu
    --num-gpus-per-trainer 0.2
    --num-cpus-per-trainer 0
    --graph-storage-mode "$storage"
    --pretrain-feature-upload-mode indexed
    --pretrain-feature-aggregation-mode "$aggregation"
    --pretrain-feature-chunk-rows 65536
    --pretrain-feature-chunk-edges 1000000
    --pretrain-feature-max-inflight-trainers 4
    --feature-compression-mode "$compression"
    --feature-compression-rank "$rank"
    --feature-compression-chunk-rows 65536
    --feature-compression-max-inflight-trainers 4
    --evaluation-split validation
    --resource-monitor-mode manual
    --resource-snapshot-interval-rounds 1
    --experiment-name "$name"
    --output-root "$output_root"
  )

  if [[ "$aggregation" == chunked ]]; then
    command+=(
      --pretrain-feature-output-dir "$runtime_dir/pretrain"
    )
  fi
  if [[ "$compression" == shared-pca && "$storage" == mmap ]]; then
    command+=(
      --feature-compression-output-dir "$runtime_dir/compression"
    )
  fi

  ./scripts/manual_ray/gpu-monitor.sh start \
    --run-id "$name" --log-dir "$monitor_dir"
  local status=0
  "${command[@]}" || status=$?
  ./scripts/manual_ray/gpu-monitor.sh stop \
    --run-id "$name" --log-dir "$monitor_dir"
  mkdir -p "$result_dir/gpu_metrics/local"
  cp -a "$monitor_dir/gpu/$name/." "$result_dir/gpu_metrics/local/"
  return "$status"
}
```

This local comparison runs five persistent trainer actors on a four-CPU Ray
node. Reserving one Ray CPU for every actor would leave the fifth trainer
permanently pending. `--num-cpus-per-trainer 0` disables that scheduling
reservation; the actors still perform CPU work and may contend for the four
physical CPUs. Keep this value identical across the Arxiv modes. The
distributed Papers100M workflow below continues to reserve one CPU per
trainer.

For the first suite, run the cases below after reviewing the experiment plan.
The first five keep the feature width at 128 and change one factor at a time.
The two PCA cases check both execution paths at the same compressed width.

```bash
export ROUNDS=3 BATCH_SIZE=16 LOCAL_STEP=1 SEED=42

run_arxiv_2hop 01-artifact-smoke v2-full-device-none "$ARXIV_V2" device full none
run_arxiv_2hop 01-artifact-smoke v2-full-cpu-none "$ARXIV_V2" cpu full none
run_arxiv_2hop 01-artifact-smoke v2-chunked-cpu-none "$ARXIV_V2" cpu chunked none
run_arxiv_2hop 01-artifact-smoke v3-chunked-cpu-none "$ARXIV_V3" cpu chunked none
run_arxiv_2hop 01-artifact-smoke v3-chunked-mmap-none "$ARXIV_V3" mmap chunked none
run_arxiv_2hop 01-artifact-smoke v2-full-cpu-pca16 "$ARXIV_V2" cpu full shared-pca 16
run_arxiv_2hop 01-artifact-smoke v3-chunked-mmap-pca16 "$ARXIV_V3" mmap chunked shared-pca 16
```

Inspect each `summary.jsonl` and confirm the expected artifact, feature width,
sampler count, and final evaluation. Compare validation and test metrics for
the unprojected cases. For the quality suite, keep the v3 artifact and
`cpu + chunked` fixed. For example, this is one of the rank/seed pairs:

```bash
ROUNDS=100 BATCH_SIZE=512 LOCAL_STEP=1 SEED=43 \
  run_arxiv_2hop 02-rank-quality v3-chunked-cpu-pca64-seed43 \
  "$ARXIV_V3" cpu chunked shared-pca 64
```

## Part B: Papers100M Distributed Verification

### Recommended Initial Fleet

Use the existing topology for the scalable paths:

* One CPU-only head with at least 64 GiB RAM and enough local storage to retain
  the complete compact v3 artifact.
* Ten GPU workers with one trainer per GPU and one trainer per host. The
  previous chunked/mmap rank sweep ran on Tesla T4 workers.
* One private security group permitting node-to-node TCP `6379` and
  `10000-10100`.

The compact v3 artifact stores only source-sorted, communication-local `int32`
adjacency in `adj.pt`. It does not retain the duplicate global-ID adjacency
from v2. The temporary source-sorted global edge cache is removed when the
artifact is finalized. Before splitting or restoring, use `df -h` and leave
headroom for raw input, binary caches, temporary partition files, and the final
artifact. The completed manifest, `du -sh`, and the Hugging Face repository
size are the authoritative measurements.

Full CPU and full device baselines have separate fleet requirements near the
end of this document. Begin with chunked mmap plus rank-16 PCA.

### Inventory And Prepare Workers

On the head:

```bash
cd /home/ubuntu/fedgraph
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedgraph312

export HEAD_PRIVATE_IP="$(hostname -I | awk '{print $1}')"
export N_TRAINER=10
export WORKERS_FILE=$HOME/fedgraph-papers100m-workers.txt
export RANK_HOSTS_FILE=$HOME/fedgraph-papers100m-2hop-rank-hosts.txt
nano "$WORKERS_FILE"
```

The worker file contains one private IPv4 per line and does not include the
head. Confirm SSH and hardware without allowing SSH to consume later lines:

```bash
ssh-add -l || ssh-add ~/.ssh/AWS_FedGraph.pem

while IFS= read -r worker || [[ -n "$worker" ]]; do
  worker="${worker%%#*}"
  worker="${worker//[[:space:]]/}"
  [[ -z "$worker" ]] && continue
  echo "===== $worker ====="
  ssh -o BatchMode=yes ubuntu@"$worker" \
    'hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h / /mnt/fedgraph-nvme 2>/dev/null || true' \
    </dev/null
done < "$WORKERS_FILE"
```

For new Ubuntu workers, run the system stage, reboot, then create the Conda
environment:

```bash
./scripts/manual_ray/setup-workers.sh \
  --stage system --workers-file "$WORKERS_FILE" --install-nvidia-driver

# Reboot and wait for nvidia-smi on every worker.

./scripts/manual_ray/setup-workers.sh \
  --stage environment --workers-file "$WORKERS_FILE"
```

Initialize blank instance-store NVMe after creation or EC2 stop/start:

```bash
./scripts/manual_ray/setup-workers.sh \
  --stage nvme \
  --workers-file "$WORKERS_FILE" \
  --nvme-device auto \
  --format-nvme \
  --jobs 2
```

After an ordinary reboot, omit `--format-nvme`. Never format a mounted or
partitioned root/EBS device.

### Choose The Artifact Source

The remaining workflow requires one complete compact v3 artifact on head NVMe.
Use exactly one of these paths:

1. **Build path:** prepare the raw Papers100M input, run the compact
   partitioner, validate the artifact, and optionally publish it to Hugging
   Face.
2. **Restore path:** download an already published complete artifact from
   Hugging Face to head NVMe and validate it locally.

Both paths converge at **Map, Synchronize, And Stage Shards**. Do not let each
trainer download its own shard from Hugging Face. A single head download avoids
simultaneous requests and makes Hugging Face latency and rate limiting external
to the benchmark.

### Build Path: Prepare Raw Papers100M Input

Follow the persistent-ZIP and head-NVMe procedure in
`manual_ray_papers100m_runbook.md`.

First verify that the head instance-store filesystem is mounted. An ordinary
reboot preserves its contents but does not necessarily remount it. Confirm the
device name from `lsblk`; `/dev/nvme1n1` is only the expected device for the
current preparation head:

```bash
export HEAD_NVME=/mnt/fedgraph-nvme
export HEAD_NVME_DEVICE=/dev/nvme1n1

lsblk -o NAME,SIZE,TYPE,MODEL,FSTYPE,MOUNTPOINTS
sudo mkdir -p "$HEAD_NVME"
test "$(lsblk -no FSTYPE "$HEAD_NVME_DEVICE" | xargs)" = ext4

if ! mountpoint -q "$HEAD_NVME"; then
  sudo mount "$HEAD_NVME_DEVICE" "$HEAD_NVME"
fi

sudo chown ubuntu:ubuntu "$HEAD_NVME"
findmnt "$HEAD_NVME"
test -w "$HEAD_NVME"
df -h "$HEAD_NVME"
```

Do not continue unless `findmnt` shows the expected instance-store device at
`/mnt/fedgraph-nvme`. Do not run `mkfs` during a remount: it would erase the
downloaded raw data and generated artifacts. A brand-new blank instance-store
device must be formatted once, only after its identity has been verified.

Then define the dataset paths:

```bash
export OGB_PARENT=$HEAD_NVME/ogb
export BINARY_CACHE_DIR=$HEAD_NVME/cache/papers100m-ogb-binary
export ARTIFACT_DIR=$HEAD_NVME/artifacts/papers100m-${N_TRAINER}-2hop-v3-compact
export DOWNLOAD_DIR=/mnt/fedgraph-ebs/papers100m-downloads
export PAPERS100M_ZIP=$DOWNLOAD_DIR/papers100M-bin.zip
mkdir -p "$OGB_PARENT" "$BINARY_CACHE_DIR" "$HEAD_NVME/artifacts" "$DOWNLOAD_DIR"
```

Reuse the durable ZIP if it exists. Otherwise download and validate it:

```bash
test -f "$PAPERS100M_ZIP" || curl -fL --retry 5 --retry-delay 10 -C - \
  -o "$PAPERS100M_ZIP" \
  https://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip
unzip -t "$PAPERS100M_ZIP"
```

Unpack to disposable head NVMe and identify the dataset root:

```bash
unzip -q "$PAPERS100M_ZIP" -d "$OGB_PARENT"
RAW_DATA_PATH=$(find "$OGB_PARENT" -type f -path '*/raw/data.npz' -print -quit)
test -n "$RAW_DATA_PATH"
export DATASET_ROOT=$(dirname "$(dirname "$RAW_DATA_PATH")")
df -h "$HEAD_NVME"
```

### Build Path: Partition And Validate Compact V3

Run inside tmux because source sorting, induced-edge construction, tensor
publication, checksumming, and validation can all be long-running:

```bash
tmux new-session -s papers100m-2hop-split
```

Inside tmux, recreate the variables and run:

```bash
cd /home/ubuntu/fedgraph
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedgraph312
export N_TRAINER=10
export HEAD_NVME=/mnt/fedgraph-nvme
export OGB_PARENT=$HEAD_NVME/ogb
export BINARY_CACHE_DIR=$HEAD_NVME/cache/papers100m-ogb-binary
export ARTIFACT_DIR=$HEAD_NVME/artifacts/papers100m-${N_TRAINER}-2hop-v3-compact
RAW_DATA_PATH=$(find "$OGB_PARENT" -type f -path '*/raw/data.npz' -print -quit)
export DATASET_ROOT=$(dirname "$(dirname "$RAW_DATA_PATH")")

time python scripts/data/partition_ogbn_nc.py \
  --dataset-root "$DATASET_ROOT" \
  --input-format ogb-binary \
  --binary-cache-dir "$BINARY_CACHE_DIR" \
  --output-dir "$ARTIFACT_DIR" \
  --n-trainer "$N_TRAINER" \
  --num-hops 2 \
  --iid-beta 10000 \
  --seed 42 \
  --chunk-rows 500000 \
  --adjacency-layout local-int32
```

If the splitter is interrupted after creating its deterministic owner plan,
rerun the identical command with `--resume`. Do not use `--no-checksums` for
an artifact that will be staged to workers or uploaded. Compact v3 stores only
`adj.pt` in communication-local `int32` coordinates; it does not store
`adj_global.pt`. During pretraining, global source IDs are reconstructed by
indexing `communicate_node_index.pt` with the local adjacency rows. The final
shard also contains `source_degree.pt` and `source_offsets.pt` for bounded,
source-aligned aggregation.

If resuming an incomplete dual-v2 attempt, keep its original output path for
the resume. The persistent communication-membership metadata records the
absolute `_work/owner.u8` path, so renaming the incomplete directory first
would force a full edge rescan. Remove only the failed shard outputs, resume
with `--adjacency-layout local-int32`, then rename the completed artifact:

```bash
export ARTIFACT_DIR=$HEAD_NVME/artifacts/papers100m-${N_TRAINER}-2hop-v2
export COMPACT_ARTIFACT_DIR=$HEAD_NVME/artifacts/papers100m-${N_TRAINER}-2hop-v3-compact

test -f "${ARTIFACT_DIR}.incomplete/_work/plan.json"
rm -rf -- "${ARTIFACT_DIR}.incomplete/shards"

time python scripts/data/partition_ogbn_nc.py \
  --dataset-root "$DATASET_ROOT" \
  --input-format ogb-binary \
  --binary-cache-dir "$BINARY_CACHE_DIR" \
  --output-dir "$ARTIFACT_DIR" \
  --n-trainer "$N_TRAINER" \
  --num-hops 2 \
  --iid-beta 10000 \
  --seed 42 \
  --chunk-rows 500000 \
  --adjacency-layout local-int32 \
  --resume

test ! -e "$COMPACT_ARTIFACT_DIR"
mv -- "$ARTIFACT_DIR" "$COMPACT_ARTIFACT_DIR"
export ARTIFACT_DIR=$COMPACT_ARTIFACT_DIR
```

Inspect the resulting size and replication before creating the fleet:

```bash
python - "$ARTIFACT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
assert (manifest["artifact_version"], manifest["hop_semantics"]) == (3, 2)
assert manifest["adjacency_layout"] == "local-int32"
assert manifest["n_trainer"] == 10
print("nodes", manifest["global_node_num"])
print("source edges", manifest["global_edge_num"])
print("induced shard edges", manifest["total_induced_edge_num"])
print("edge replication factor", manifest["edge_replication_factor"])
for shard in manifest["shards"]:
    print(
        shard["trainer_id"], shard["owned_node_count"],
        shard["communicate_node_count"], shard["induced_edge_count"]
    )
PY

du -sh "$ARTIFACT_DIR"
df -h "$HEAD_NVME"
```

### Build Path: Publish Compact V3 To Hugging Face

Publishing is optional for the immediate run but strongly recommended before
stopping a head backed by instance-store NVMe. The uploader validates the
manifest, file sizes, and all splitter SHA-256 checksums before transferring
the complete directory as one dataset repository.

Run the upload in tmux:

```bash
tmux new-session -s papers100m-2hop-upload
```

Inside tmux, authenticate with a write-capable token and recreate the paths:

```bash
cd /home/ubuntu/fedgraph
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedgraph312

export N_TRAINER=10
export ARTIFACT_DIR=/mnt/fedgraph-nvme/artifacts/papers100m-${N_TRAINER}-2hop-v3-compact
export HF_ARTIFACT_REPO=FedGraph/fedgraph_ogbn-papers100M_${N_TRAINER}trainer_2hop_iid_beta_10000.0_v3

hf auth whoami || hf auth login
mkdir -p "$HOME/fedgraph-logs"
set -o pipefail

time python scripts/data/upload_nc_artifact_to_hf.py \
  --artifact-dir "$ARTIFACT_DIR" \
  --repo-id "$HF_ARTIFACT_REPO" \
  --private \
  --num-workers 4 \
  2>&1 | tee "$HOME/fedgraph-logs/papers100m-${N_TRAINER}trainer-2hop-v3-upload.log"
```

Do not use `--skip-checksum-verification` for the production artifact. If an
upload is interrupted, rerun the same command; Hugging Face's large-folder
uploader reuses completed work. A `403` means the active token cannot create or
write the dataset under the selected namespace. Change `HF_ARTIFACT_REPO` if
the uploaded repository uses a different name.

Record the immutable repository revision after upload:

```bash
export HF_ARTIFACT_REVISION=$(python - <<'PY'
import os
from huggingface_hub import HfApi

info = HfApi().repo_info(
    repo_id=os.environ["HF_ARTIFACT_REPO"], repo_type="dataset"
)
print(info.sha)
PY
)
printf 'HF_ARTIFACT_REPO=%s\nHF_ARTIFACT_REVISION=%s\n' \
  "$HF_ARTIFACT_REPO" "$HF_ARTIFACT_REVISION"
```

Keep these two values with the experiment metadata. Pinning the commit hash
prevents a later repository update from silently changing the artifact.

### Restore Path: Download The Complete Artifact To Head NVMe

Use this path on a new head, or whenever the local artifact was lost with
instance-store storage. It intentionally downloads the complete repository to
the head once. Workers continue to receive only their assigned shard through
the staging script.

First mount and check head NVMe as described in **Build Path: Prepare Raw
Papers100M Input**, then enter a persistent session. The restore path does not
need the raw ZIP, extracted OGB input, binary cache, or partitioner work files.

```bash
tmux new-session -s papers100m-2hop-download
```

Inside tmux:

```bash
cd /home/ubuntu/fedgraph
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedgraph312

export N_TRAINER=10
export HEAD_NVME=/mnt/fedgraph-nvme
export ARTIFACT_DIR=$HEAD_NVME/artifacts/papers100m-${N_TRAINER}-2hop-v3-compact
export HF_ARTIFACT_REPO=FedGraph/fedgraph_ogbn-papers100M_${N_TRAINER}trainer_2hop_iid_beta_10000.0_v3
export HF_ARTIFACT_REVISION="REPLACE_WITH_COMMIT_HASH_RECORDED_AFTER_UPLOAD"

hf auth whoami || hf auth login
[[ "$HF_ARTIFACT_REVISION" != REPLACE_* ]]
mkdir -p "$HEAD_NVME/artifacts" "$ARTIFACT_DIR" "$HOME/fedgraph-logs"
df -h "$HEAD_NVME"
set -o pipefail

time hf download "$HF_ARTIFACT_REPO" \
  --repo-type dataset \
  --revision "$HF_ARTIFACT_REVISION" \
  --local-dir "$ARTIFACT_DIR" \
  --max-workers 4 \
  2>&1 | tee "$HOME/fedgraph-logs/papers100m-${N_TRAINER}trainer-2hop-v3-download.log"
```

Rerun the identical `hf download` command after an interruption; already
downloaded files are reused. For a private repository, the token needs read
access. Do not use `main` for a reproducibility run when a commit hash is
available. `ARTIFACT_DIR` may contain an interrupted download of this same
repository and revision, but it must not contain a different artifact or an
incomplete local partition.

Validate the restored artifact before distributing it. This intentionally
reads and hashes every shard once:

```bash
python - "$ARTIFACT_DIR" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "scripts" / "data"))
from upload_nc_artifact_to_hf import validate_nc_artifact_for_upload

root = Path(sys.argv[1])
manifest = validate_nc_artifact_for_upload(root, verify_checksums=True)
assert (manifest["artifact_version"], manifest["hop_semantics"]) == (3, 2)
assert manifest["adjacency_layout"] == "local-int32"
print(
    "validated compact artifact:",
    f"trainers={manifest['n_trainer']}",
    f"nodes={manifest['global_node_num']}",
    f"root={root}",
)
PY

du -sh "$ARTIFACT_DIR"
df -h "$HEAD_NVME"
```

### Map, Synchronize, And Stage Shards

At this point `ARTIFACT_DIR` must refer to the complete validated artifact on
the head, whether it was just built or restored from Hugging Face. The command
below copies `manifest.json` plus only the shard assigned to each trainer rank;
it does not replicate the complete repository to every worker.

Create one trainer per worker for the first comparison round:

```bash
./scripts/manual_ray/generate-rank-hosts.sh \
  --workers-file "$WORKERS_FILE" \
  --n-trainer "$N_TRAINER" \
  --output-file "$RANK_HOSTS_FILE"
cat "$RANK_HOSTS_FILE"
```

Synchronize the current working tree, including uncommitted Component 3 code,
then stage only the assigned shard to each worker:

```bash
./scripts/manual_ray/sync-workers.sh \
  --workers-file "$WORKERS_FILE" \
  --install-editable

./scripts/manual_ray/stage-local-artifact-shards.sh \
  --rank-hosts-file "$RANK_HOSTS_FILE" \
  --source-dir "$ARTIFACT_DIR" \
  --remote-dir "$ARTIFACT_DIR"
```

Check shard and runtime capacity on every worker:

```bash
while IFS= read -r worker || [[ -n "$worker" ]]; do
  worker="${worker%%#*}"
  worker="${worker//[[:space:]]/}"
  [[ -z "$worker" ]] && continue
  ssh ubuntu@"$worker" \
    "du -sh '$ARTIFACT_DIR'; df -h /mnt/fedgraph-nvme" </dev/null
done < "$WORKERS_FILE"
```

### Derive Worker-Local V2 Shards For Full Baselines

Skip this section when running only chunked modes. The converter never modifies
the compact v3 source. It reconstructs `adj_global.pt` in bounded edge batches,
writes an int64 training `adj.pt`, and hard-links unchanged files when source
and output are on the same filesystem. Each worker converts only its mapped
rank or ranks; the head receives only a v2 control manifest.

The two int64 adjacency views are each twice the size of compact v3 `adj.pt`.
Conversion also needs temporary space for one int64 view while publishing it,
so check worker NVMe capacity first.

```bash
export V3_ARTIFACT_DIR="$ARTIFACT_DIR"
export V2_ARTIFACT_DIR=/mnt/fedgraph-nvme/artifacts/papers100m-${N_TRAINER}-2hop-v2-derived

./scripts/manual_ray/convert-local-artifact-v3-to-v2.sh \
  --rank-hosts-file "$RANK_HOSTS_FILE" \
  --source-dir "$V3_ARTIFACT_DIR" \
  --output-dir "$V2_ARTIFACT_DIR" \
  --chunk-edges 1000000 \
  --jobs 2
```

The wrapper creates `$V2_ARTIFACT_DIR/manifest.json` on the head and the
assigned converted shards under the same path on each worker. The head-side
directory is a control manifest, not a complete uploadable artifact. To create
a complete archival v2 artifact on one host instead, run:

```bash
python scripts/data/convert_nc_artifact_v3_to_v2.py \
  --source-dir "$V3_ARTIFACT_DIR" \
  --output-dir "$V2_ARTIFACT_DIR" \
  --chunk-edges 1000000
```

Do not run both conversion forms against the same output directory.

### Start The Ray Fleet

Start the CPU-only head. Put Ray spill/log data on a writable path with enough
space; the chunked path normally needs only the default 8 GiB object store.

```bash
export HEAD_PRIVATE_IP="$(hostname -I | awk '{print $1}')"
mkdir -p "$HOME/fedgraph-logs" "$HOME/fedgraph-ray"

tmux new-session -d -s ray-head \
  "cd /home/ubuntu/fedgraph && ./scripts/manual_ray/start-head.sh \
  --head-ip $HEAD_PRIVATE_IP \
  --num-cpus 8 \
  --object-store-memory-gib 8 \
  --ray-tmp-dir /home/ubuntu/fedgraph-ray \
  2>&1 | tee /home/ubuntu/fedgraph-logs/ray-head-2hop-startup.log"
sleep 3
```

Preflight, launch, and verify workers:

```bash
./scripts/manual_ray/launch-workers.sh \
  --preflight \
  --head-ip "$HEAD_PRIVATE_IP" \
  --workers-file "$WORKERS_FILE" \
  --num-cpus 8 \
  --hf-home /mnt/fedgraph-nvme/hf-cache

./scripts/manual_ray/launch-workers.sh \
  --head-ip "$HEAD_PRIVATE_IP" \
  --workers-file "$WORKERS_FILE" \
  --num-cpus 8 \
  --hf-home /mnt/fedgraph-nvme/hf-cache

./scripts/manual_ray/status.sh --head-ip "$HEAD_PRIVATE_IP"
```

Expect eleven Ray nodes and ten free GPUs before submission.

### Submit The Safe First Canary

Create a dedicated tmux session so all required exports are defined inside the
driver shell:

```bash
tmux new-session -s papers100m-2hop-submit
```

Inside tmux:

```bash
cd /home/ubuntu/fedgraph
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedgraph312

export HEAD_PRIVATE_IP="$(hostname -I | awk '{print $1}')"
export N_TRAINER=10
export WORKERS_FILE=$HOME/fedgraph-papers100m-workers.txt
export RANK_HOSTS_FILE=$HOME/fedgraph-papers100m-2hop-rank-hosts.txt
export V3_ARTIFACT_DIR=/mnt/fedgraph-nvme/artifacts/papers100m-${N_TRAINER}-2hop-v3-compact
export V2_ARTIFACT_DIR=/mnt/fedgraph-nvme/artifacts/papers100m-${N_TRAINER}-2hop-v2-derived
export ARTIFACT_DIR="$V3_ARTIFACT_DIR"
# Export the campaign ID chosen on the Arxiv host and set the root on this head.
export NC_2HOP_RESULTS="$PWD/benchmark/results/nc_2hop_optimization"
: "${NC_2HOP_CAMPAIGN:?Set the campaign ID in this shell first}"
export PAPERS_RESULT_ROOT="$NC_2HOP_RESULTS/ogbn-papers100M/$NC_2HOP_CAMPAIGN"
export PAPERS_RUNTIME_ROOT="/mnt/fedgraph-nvme/fedgraph-runtime/$NC_2HOP_CAMPAIGN"
export PAPERS_GPU_LOGS="$HOME/fedgraph-logs/local-artifact-2hop/$NC_2HOP_CAMPAIGN"

./scripts/manual_ray/status.sh --head-ip "$HEAD_PRIVATE_IP"
```

If this is a rerun after changing FedGraph source code, synchronize the updated
tree before submission. The already staged artifact does not need to be copied
again:

```bash
./scripts/manual_ray/sync-workers.sh \
  --workers-file "$WORKERS_FILE" \
  --install-editable
```

Define the case launcher in the same shell. It uses the same pretraining
limits for the first smoke and matched storage comparison. Extra arguments
at the end select the specific case's modes and round budget.

```bash
run_papers100m_2hop() {
  local suite="$1" name="$2" artifact="$3"
  shift 3
  local runtime_dir="$PAPERS_RUNTIME_ROOT/$suite/$name"
  ./scripts/manual_ray/submit-local-artifact-2hop.sh \
    --head-ip "$HEAD_PRIVATE_IP" \
    --artifact-dir "$artifact" \
    --rank-hosts-file "$RANK_HOSTS_FILE" \
    --n-trainer "$N_TRAINER" \
    --num-cpus-per-trainer 1 \
    --num-gpus-per-trainer 1 \
    --batch-size 16 \
    --evaluation-batch-size 512 \
    --local-step 1 \
    --seed "${SEED:-42}" \
    --pretrain-chunk-rows 262144 \
    --pretrain-chunk-edges 1000000 \
    --pretrain-max-inflight-trainers 10 \
    --pretrain-output-dir "$runtime_dir/pretrain" \
    --feature-compression-chunk-rows 65536 \
    --feature-compression-max-inflight-trainers 4 \
    --feature-compression-output-dir "$runtime_dir/compression" \
    --evaluation-split validation \
    --resource-monitor-mode manual \
    --resource-snapshot-interval-rounds 1 \
    --gpu-monitor-detail raw \
    --output-root "$PAPERS_RESULT_ROOT/$suite" \
    --worker-log-dir "$PAPERS_GPU_LOGS/$suite" \
    --experiment-name "$name" "$@"
}
```

After the Arxiv gate passes and the fleet is ready, submit one round with
mmap, chunking and PCA-16. To only preview the plan, append
`--dry-run --output-root "$(mktemp -d)"` to the call. The preview writes the
case manifest and run plan to scratch without contacting Ray or the workers.

```bash
run_papers100m_2hop 01-fleet-smoke v3-chunked-mmap-pca16 "$V3_ARTIFACT_DIR" \
  --rounds 1 \
  --graph-storage-mode mmap \
  --pretrain-aggregation-mode chunked \
  --feature-compression-mode shared-pca \
  --feature-compression-rank 16
```

Keep `--pretrain-chunk-edges 1000000` unchanged for this rerun. It bounds each
trainer's internal edge scan independently of the larger global row range. The
ten-way in-flight setting assumes the documented one-trainer-per-worker fleet;
reduce it if several trainer actors later share a worker.

Before continuing, require all of the following:

* Every trainer reports the same nonempty PCA basis hash.
* Runtime feature dimension is 16 for compression and pretraining.
* All pretraining output files finalize with their expected row counts.
* `NC_PRETRAIN_METRIC` reports v3 and `NC_NEIGHBOR_SAMPLER_METRIC` reports
  ten prepared samplers before the first `NC_ROUND_METRIC`.
* One training and validation round plus the final test evaluation completes.
* Final evaluation completes without a Ray host-memory kill from rebuilding the
  graph sampler.
* Ray reports ten free GPUs again after the driver exits.

### Submit Matched Comparisons

Select pairs from the [experiment plan](2hop_optimization_status.md#experiment-plan).
For `02-storage`, use the same v3 artifact, rank, and chunk limits. Start with
mmap, then run CPU after checking host RAM and local disk capacity. Both cases
below use three rounds and the same training and evaluation batch sizes.

```bash
run_papers100m_2hop 02-storage v3-chunked-mmap-pca16 "$V3_ARTIFACT_DIR" \
  --rounds 3 \
  --graph-storage-mode mmap --pretrain-aggregation-mode chunked \
  --feature-compression-mode shared-pca --feature-compression-rank 16

run_papers100m_2hop 02-storage v3-chunked-cpu-pca16 "$V3_ARTIFACT_DIR" \
  --rounds 3 \
  --graph-storage-mode cpu --pretrain-aggregation-mode chunked \
  --feature-compression-mode shared-pca --feature-compression-rank 16
```

For `03-aggregation`, size the head and workers first. Run both paths on the
same derived v2 shards at a rank that fits. The example uses rank 16, subject
to that sizing check. V3 remains the default for the other chunked suites.

```bash
run_papers100m_2hop 03-aggregation v2-chunked-cpu-pca16 "$V2_ARTIFACT_DIR" \
  --rounds 3 \
  --graph-storage-mode cpu --pretrain-aggregation-mode chunked \
  --feature-compression-mode shared-pca --feature-compression-rank 16

run_papers100m_2hop 03-aggregation v2-full-cpu-pca16 "$V2_ARTIFACT_DIR" \
  --rounds 3 \
  --graph-storage-mode cpu --pretrain-aggregation-mode full \
  --feature-compression-mode shared-pca --feature-compression-rank 16
```

Full mode receives complete trainer payloads. The chunked in-flight limit
bounds only chunked feature requests and the finalization calls. Watch head
RAM and Ray object-store pressure before promoting a full baseline.

### Performance And Accuracy Runs

Use the fixed settings and selected cases from the experiment plan. Change the
suite label when moving from a smoke test to a timing or quality comparison.
Use `SEED=43` (or `--seed 43`) and a `-seed43` case suffix for a separate seed.
Keep local step at one for the new comparison campaign. Record any later
local-step sweep as its own suite.

## Results And Interpretation

Within each case directory, inspect:

```text
benchmark/results/nc_2hop_optimization/<dataset>/<campaign>/<suite>/<case>/
  manifest.json
  run_plan.csv
  summary.csv
  summary.jsonl
  round_metrics.csv
  runs/<run-id>/config.json
  runs/<run-id>/stdout.log
  runs/<run-id>/fedgraph_logs/resource_snapshots.jsonl
  runs/<run-id>/fedgraph_logs/resource_snapshot_summary.json
  gpu_metrics/<worker-ip>/
```

Compare these metric groups:

| Question | Primary metrics |
|---|---|
| Did PCA reduce feature state? | original/projected bytes, projected fraction, runtime dimension |
| What did PCA cost? | statistics, basis, projection, and total compression wall time |
| Did traffic shrink? | pretrain upload, download, and total traffic bytes |
| Did chunking bound the head? | server peak RSS and minimum host MemAvailable |
| Did storage leave VRAM? | trainer CUDA allocated/reserved and device-wide free memory |
| Did mmap trade RAM for I/O? | trainer RSS, page-cache behavior, phase wall time |
| What did the adjacency transition cost? | adjacency transition wall/max-trainer time and strategy |
| What did sampler indexing cost? | sampler setup wall/max-trainer time and total index bytes |
| Was accuracy retained? | final/best validation accuracy and loss |

`NC_NEIGHBOR_SAMPLER_METRIC` is emitted after pretraining and before the
federated timer starts. Consequently, NeighborSampler index construction is
included in total execution time but excluded from pretraining and round-1
training time. `summary.csv` and `summary.jsonl` preserve these fields together
with artifact and source-artifact versions.

Do not compare a cold mmap run directly with a warm-cache run without labeling
that distinction. Do not treat runs on different GPU families as direct speed
comparisons.

## Full Baseline Capacity

Near the end of full indexed aggregation, the head can hold all uploaded
contributions (row indices and values), the union accumulator, and copies of
each trainer's downlink features at the same time. Applying that live set to
the recorded ten-shard Papers100M row counts gives approximate tensor-only
requirements of 41 GiB at rank 8, 78 GiB at rank 16, and 594 GiB at the original
width 128. Budget additional RAM for index workspaces, Python, and Ray's object
store. A 64 GiB head is too small for the rank-16 baseline.

On workers, account for the full global adjacency and per-edge feature gathers
during pretraining, then the local adjacency, returned features, and sampler
indexes during training. Full `device` also needs VRAM for the live graph and
the pretraining temporaries. Size worker RAM/VRAM and spare NVMe space before
staging a full baseline. A one-shard loading check covers worker capacity.
Keep `--n-trainer` equal to the manifest count in submitted benchmarks, and
use the same fleet for both sides when comparing speed.

## Runtime Storage And Shutdown

Chunked runs retain complete per-trainer aggregated feature files in each
case's worker-local `pretrain/` directory below `PAPERS_RUNTIME_ROOT`. Mmap
PCA runs also retain projected source features in `compression/`. Inspect
worker disk usage between experiments:

```bash
while IFS= read -r worker || [[ -n "$worker" ]]; do
  worker="${worker%%#*}"
  worker="${worker//[[:space:]]/}"
  [[ -z "$worker" ]] && continue
  ssh ubuntu@"$worker" \
    'du -sh /mnt/fedgraph-nvme/fedgraph-runtime/* 2>/dev/null || true; df -h /mnt/fedgraph-nvme' \
    </dev/null
done < "$WORKERS_FILE"
```

Archive benchmark results before removing runtime files or stopping instances.
When work is complete, stop the `ray-worker-*` tmux session on each worker,
then stop the `ray-head` session. Instance-store artifacts, staged shards,
runtime outputs, and Hugging Face caches are erased by EC2 stop/start.
