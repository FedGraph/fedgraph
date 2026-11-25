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

## Run Any Series

Series numbers organize results; they are not prerequisites. Set a
[campaign ID](#campaign-layout), complete the shared setup for your dataset,
then jump directly to the series you want. A smoke case on a new or changed
fleet is useful, but no later series consumes an earlier series' results.

* **Arxiv:** [prepare the GPU environment](#prepare-the-environment-and-dataset),
  [prepare v3 (and optionally v2) artifacts](#prepare-matched-five-trainer-artifacts),
  and [load the local launcher](#run-the-local-comparison-function).
* **Papers100M:** [prepare the head](#prepare-the-head),
  [prepare the workers](#inventory-and-prepare-workers),
  [restore or build compact v3](#choose-the-artifact-source),
  [stage shards](#map-synchronize-and-stage-shards),
  [start Ray](#start-the-ray-fleet), then
  [load the submission shell](#prepare-the-papers100m-submission-shell).

| Dataset / Series | Additional Setup |
|---|---|
| Arxiv [01-artifact-smoke](#run-the-arxiv-artifact-smoke) | Matched v3 and derived v2 for the full matrix |
| Arxiv [02-rank-quality](#run-the-rank-quality-sweep) | V3 only |
| Papers100M [01-fleet-smoke](#submit-the-safe-first-canary) | V3 only; optional fleet check |
| Papers100M [02-storage](#submit-matched-comparisons) | V3 only; check worker RAM before CPU mode |
| Papers100M [03-aggregation](#submit-matched-comparisons) | [Derive v2](#derive-worker-local-v2-shards-for-full-baselines) from the same v3; check [full-mode capacity](#full-baseline-capacity) |
| Papers100M [04-projection](#run-the-projection-sweep) | V3 only; check free worker NVMe between cases |
| Papers100M [05-chunk-tuning](#run-the-chunk-tuning-sweep) | V3 only; run this series' own control |
| Papers100M [06-rank-quality](#run-the-papers100m-rank-quality-sweep) | V3 only; computes features in each case |
| Papers100M [07-cache-smoke](#reuse-completed-feature-aggregation) | V3 and a [prepared feature cache](#prepare-a-reusable-feature-cache) at the rank being tested |
| Papers100M [08-model-architecture](#compare-papers100m-model-architectures), [09-training-recipe](#tune-the-papers100m-optimizer) | V3 and a [PCA-64 feature cache](#prepare-a-reusable-feature-cache) |
| Papers100M [10-hyperparameter-sweep](#run-the-hyperparameter-sweep), [11-model-heads](#compare-papers100m-model-heads) | V3 and a [feature cache](#prepare-a-reusable-feature-cache) matching the case; 11 uses no PCA |

After an EC2 stop/start, expect to reinitialize instance-store NVMe, restore
artifacts and caches (or recreate them), and stage shards again. EBS-backed
environments and files survive a stop/start. An ordinary reboot preserves
instance store, but can leave it unmounted. Check the mounts before copying
or submitting. See [AWS instance-store persistence](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-store-lifetime.html).

## Campaign Layout

From the repo root, choose one campaign ID for the runs you want to group.
Reuse it in new shells and tmux sessions or when copying results. A Papers100M
campaign can start without an Arxiv campaign. Use a fresh ID for a new campaign.

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
    04-projection/v3-chunked-mmap-pca16/
    05-chunk-tuning/v3-chunked-mmap-pca16-baseline/
    06-rank-quality/v3-chunked-mmap-pca16-seed42/
    07-cache-smoke/v3-mmap-cache-pca64/
    08-model-architecture/baseline-200r/
    09-training-recipe/normalized-wider-adamw-lr0.001-800r/
    10-hyperparameter-sweep/control-800r/
    11-model-heads/gcn-skip-800r-seed42/
```

Pass the suite directory as `--output-root` and the case label as
`--experiment-name`. Use a suffix such as `-rep02` to preserve an earlier
attempt, including failures. Keep artifacts and runtime caches on the
designated storage paths. Historical results remain in
`benchmark/results/nc_batch_size_convergence/`.
For retained evidence, see the [2-hop archive](../benchmark/archives/nc_2hop_optimization/ARCHIVE.md)
and the [earlier NC archive](../benchmark/archives/nc_batch_size_convergence/ARCHIVE.md).

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

Use a single Ubuntu GPU host with the current checkout at `/home/ubuntu/fedgraph`.
On a fresh one, use [bootstrap-worker.sh](../scripts/manual_ray/bootstrap-worker.sh)
to install or verify the `fedgraph312` GPU environment (install the driver and
reboot first if `nvidia-smi` fails). Then run:

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

Partition once to compact v3, or reuse a validated artifact. Derive v2 only
for the v2 cases in `01-artifact-smoke`. This keeps the edges, features, and
train/validation/test splits identical across the smoke cases. The rank sweep
uses only v3.

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
```

For the full smoke matrix only:

```bash
python scripts/data/convert_nc_artifact_v3_to_v2.py \
  --source-dir "$ARXIV_V3" --output-dir "$ARXIV_V2"
sha256sum "$ARXIV_V3/manifest.json" "$ARXIV_V2/manifest.json"
```

### Run The Local Comparison Function

Unset any remote address so `ray.init()` can create a local runtime that
advertises this host's GPU. Stop an old local Ray runtime only on a dedicated
host with no running jobs:

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

### Run The Arxiv Artifact Smoke

After the shared Arxiv setup, run the cases below.
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
the unprojected cases.

### Run The Rank-Quality Sweep

Use the shared Arxiv setup, v3 artifact, and `run_arxiv_2hop` function. No v2
artifact or smoke result is required. Keep CPU graph storage and chunked
pretraining fixed.
The archived runner recorded seed labels without passing them to NC, so treat
those cases as repeats. The updated runner forwards the seed. Rerun the
seed-42 control with it before comparing distinct seeds in this loop.

```bash
export ROUNDS=100 BATCH_SIZE=512 LOCAL_STEP=1

for seed in 42 43 44; do
  SEED="$seed" run_arxiv_2hop 02-rank-quality "v3-chunked-cpu-none-seed${seed}" \
    "$ARXIV_V3" cpu chunked none || break

  for rank in 128 64 32 16; do
    SEED="$seed" run_arxiv_2hop 02-rank-quality "v3-chunked-cpu-pca${rank}-seed${seed}" \
      "$ARXIV_V3" cpu chunked shared-pca "$rank" || break 2
  done
done
```

Expect 15 completed `summary.jsonl` files under
`$ARXIV_OUTPUT_ROOT/02-rank-quality/`, each with 100 recorded rounds and a final
test result. The loop stops on the first failed case. Use a `-rep02` suffix
when retrying that case to preserve its original logs.

## Part B: Papers100M Distributed Verification

### Recommended Initial Fleet

Use the existing topology for the scalable paths:

* One CPU-only head with at least 64 GiB RAM and enough local storage to retain
  the complete compact v3 artifact.
* Ten GPU workers with one trainer per GPU and one trainer per host. The
  previous chunked/mmap rank sweep ran on Tesla T4 workers.
* Head and workers on a reachable private network. Allow bidirectional Ray
  traffic only among fleet members, for example with a self-referencing TCP
  rule on a dedicated security group. TCP `6379` alone is not enough: these
  launchers leave the node manager, object manager, and other Ray ports
  unpinned. Keep SSH limited to known admin sources and head-to-worker access.
  Do not expose Ray ports publicly. See the [Ray port reference](https://docs.ray.io/en/latest/ray-core/configure.html#ports-configurations).

The compact v3 artifact stores only source-sorted, communication-local `int32`
adjacency in `adj.pt`. It does not retain the duplicate global-ID adjacency
from v2. The temporary source-sorted global edge cache is removed when the
artifact is finalized. Before splitting or restoring, use `df -h` and leave
headroom for raw input, binary caches, temporary partition files, and the final
artifact. The completed manifest, `du -sh`, and the Hugging Face repository
size are the authoritative measurements.

Full CPU and full device baselines have separate fleet requirements near the
end of this document. On an unfamiliar fleet, an optional first check is
chunked mmap plus PCA-16.

### Prepare The Head

Use an x86-64 Ubuntu head with the chosen FedGraph checkout at
`/home/ubuntu/fedgraph`. If it already has `fedgraph312`, keep it and verify
Python 3.12 and the same Ray version as the workers. For a fresh CPU-only
head, install [Miniconda](https://docs.anaconda.com/miniconda/install/) at
`$HOME/miniconda3`, then install the CPU wheels instead of the GPU worker stack:

```bash
sudo apt-get update
sudo apt-get install -y git tmux rsync curl jq unzip
cd /home/ubuntu/fedgraph
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda create -n fedgraph312 python=3.12 -y
conda activate fedgraph312
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install --only-binary=:all: \
  torch-scatter==2.1.2+pt25cpu torch-sparse==0.6.18+pt25cpu \
  torch-cluster==1.6.3+pt25cpu torch-spline-conv==1.2.2+pt25cpu \
  -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
python -m pip install -e . 'ray[default]==2.56.0' torch-geometric==2.8.0 huggingface_hub==1.22.0
python -c 'import sys, ray, torch, fedgraph; print(sys.version, ray.__version__, torch.__version__)'
```

The CPU head does not need an NVIDIA driver or OpenFHE for plaintext runs.
If Conda prompts about channel terms, review them before continuing. The
[PyTorch](https://pytorch.org/get-started/previous-versions/#v251) and
[PyG](https://data.pyg.org/whl/torch-2.5.1+cpu.html) wheel sources match the
worker bootstrap's PyTorch/PyG versions. Check the active environment on both sides
before starting Ray.

Mount head instance storage **before** building or restoring artifacts:

```bash
export HEAD_NVME=/mnt/fedgraph-nvme
lsblk -o NAME,SIZE,TYPE,MODEL,FSTYPE,MOUNTPOINTS
./scripts/manual_ray/prepare-worker-nvme.sh --device auto --mount-point "$HEAD_NVME"
findmnt --mountpoint "$HEAD_NVME"
test -w "$HEAD_NVME"
df -h "$HEAD_NVME"
```

For a confirmed blank Amazon EC2 instance-storage device, repeat the helper
with `--format` the first time. Omit it for a remount. If auto-detection is
ambiguous, pass the verified device from `lsblk` explicitly. Never format
root/EBS or an existing filesystem. Do not continue unless `findmnt` shows
the intended mount: a directory alone could be on the root filesystem.

### Inventory And Prepare Workers

On the head:

```bash
cd /home/ubuntu/fedgraph
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedgraph312

export HEAD_PRIVATE_IP="$(hostname -I | awk '{print $1}')"
export HEAD_NVME=/mnt/fedgraph-nvme
export N_TRAINER=10
export WORKERS_FILE=$HOME/fedgraph-papers100m-workers.txt
export RANK_HOSTS_FILE=$HOME/fedgraph-papers100m-2hop-rank-hosts.txt
nano "$WORKERS_FILE"
```

The worker file contains one private IPv4 per line and does not include the
head. After a restart or with a new fleet, check that it lists the current
private IPs. On a new head, verify worker SSH host keys before using batch
mode. If using a forwarded agent, load the key on your own computer and check
`ssh-add -l` on the head; do not copy the private key to the head.
For each new worker, connect once with `ssh ubuntu@WORKER_PRIVATE_IP`, compare
the displayed fingerprint with the worker's host key from a trusted console,
and accept it only if it matches. Then all automation can use noninteractive SSH.

For new Ubuntu workers, run the system stage, reboot, then create the Conda
environment. Skip these stages on restarted workers whose EBS-backed
environment and GPU driver still work:

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

Confirm hardware and mounts without allowing SSH to consume later inventory
lines:

```bash
while IFS= read -r worker || [[ -n "$worker" ]]; do
  worker="${worker%%#*}"
  worker="${worker//[[:space:]]/}"
  [[ -z "$worker" ]] && continue
  echo "===== $worker ====="
  ssh -n -o BatchMode=yes ubuntu@"$worker" \
    'hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; findmnt --mountpoint /mnt/fedgraph-nvme && df -h / /mnt/fedgraph-nvme'
done < "$WORKERS_FILE"
```

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

Confirm [head NVMe is mounted](#prepare-the-head) and the durable download
path is on EBS, not instance storage. Then define the dataset paths:

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

First [mount and check head NVMe](#prepare-the-head), then enter a persistent
session. The restore path does not need the raw ZIP, extracted OGB input,
binary cache, or partitioner work files.

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
```

Use the recorded commit hash when reproducing a previous run or restoring its
feature cache. If starting a new campaign without a recorded revision, resolve
`main` once and keep the printed commit hash with the results:

```bash
export HF_ARTIFACT_REVISION="$(python -c 'import sys; from huggingface_hub import HfApi; print(HfApi().dataset_info(sys.argv[1]).sha)' "$HF_ARTIFACT_REPO")"
printf '%s@%s\n' "$HF_ARTIFACT_REPO" "$HF_ARTIFACT_REVISION"
```

With a commit hash set and the NVMe mount verified, download to a fresh artifact
path or resume the same revision into its existing path:

```bash
(
  set -e -o pipefail
  [[ "$HF_ARTIFACT_REVISION" =~ ^[0-9a-f]{40}$ ]]
  findmnt --mountpoint "$HEAD_NVME"
  mkdir -p "$HEAD_NVME/artifacts" "$ARTIFACT_DIR" "$HOME/fedgraph-logs"
  df -h "$HEAD_NVME"

  time hf download "$HF_ARTIFACT_REPO" \
    --repo-type dataset \
    --revision "$HF_ARTIFACT_REVISION" \
    --local-dir "$ARTIFACT_DIR" \
    --max-workers 4 \
    2>&1 | tee "$HOME/fedgraph-logs/papers100m-${N_TRAINER}trainer-2hop-v3-download.log"
)
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

Return to the head shell used for [worker inventory](#inventory-and-prepare-workers),
or load those exports again. Set `ARTIFACT_DIR` to the path from your artifact
build/download session. With the defaults, create one trainer per worker:

```bash
export ARTIFACT_DIR="${ARTIFACT_DIR:-$HEAD_NVME/artifacts/papers100m-${N_TRAINER}-2hop-v3-compact}"
test -r "$ARTIFACT_DIR/manifest.json"

./scripts/manual_ray/generate-rank-hosts.sh \
  --workers-file "$WORKERS_FILE" \
  --n-trainer "$N_TRAINER" \
  --output-file "$RANK_HOSTS_FILE"
cat "$RANK_HOSTS_FILE"
```

Synchronize the current working tree, including any local changes needed for
the experiment, then stage only the assigned shard to each worker:

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

Start the CPU-only head. Use mounted NVMe for Ray spill and logs so a full
root EBS volume does not stop aggregation. The chunked path normally needs
only the default 8 GiB object store. Set a different budget only after sizing
the head for the intended comparison.

```bash
export HEAD_PRIVATE_IP="$(hostname -I | awk '{print $1}')"
export HEAD_RAY_TMP_DIR="$HEAD_NVME/ray"
export HEAD_OBJECT_STORE_GIB="${HEAD_OBJECT_STORE_GIB:-8}"
findmnt --mountpoint "$HEAD_NVME"
mkdir -p "$HOME/fedgraph-logs" "$HEAD_RAY_TMP_DIR"

tmux new-session -d -s ray-head \
  "cd /home/ubuntu/fedgraph && ./scripts/manual_ray/start-head.sh \
  --head-ip $HEAD_PRIVATE_IP \
  --num-cpus 8 \
  --object-store-memory-gib $HEAD_OBJECT_STORE_GIB \
  --ray-tmp-dir '$HEAD_RAY_TMP_DIR' \
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

After a `ray stop` restart, an old `ray-worker-*` tmux session may still have
its name even though the Ray processes have exited. Once the old fleet has
stopped and no worker launch is in progress, add `--replace-stale-sessions`
to the worker launch command (not the preflight command). The launcher
replaces the exact worker session only when no live `raylet` is running on
that host. It keeps sessions on workers that have already rejoined. This
does not stop the EC2 instances or change staged artifacts.

### Prepare The Papers100M Submission Shell

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
export HEAD_NVME=${HEAD_NVME:-/mnt/fedgraph-nvme}
export N_TRAINER=${N_TRAINER:-10}
export WORKERS_FILE="${WORKERS_FILE:-$HOME/fedgraph-papers100m-workers.txt}"
export RANK_HOSTS_FILE="${RANK_HOSTS_FILE:-$HOME/fedgraph-papers100m-2hop-rank-hosts.txt}"
# Set these explicitly if you staged your artifacts under non-default paths.
export V3_ARTIFACT_DIR="${V3_ARTIFACT_DIR:-$HEAD_NVME/artifacts/papers100m-${N_TRAINER}-2hop-v3-compact}"
export V2_ARTIFACT_DIR="${V2_ARTIFACT_DIR:-$HEAD_NVME/artifacts/papers100m-${N_TRAINER}-2hop-v2-derived}"
export ARTIFACT_DIR="$V3_ARTIFACT_DIR"
export NC_2HOP_RESULTS="${NC_2HOP_RESULTS:-$PWD/benchmark/results/nc_2hop_optimization}"
# For existing campaigns, explicitly set the recorded ID inside this shell.
export NC_2HOP_CAMPAIGN="${NC_2HOP_CAMPAIGN:-$(date -u +%Y%m%dT%H%M%SZ)}"
export PAPERS_RESULT_ROOT="$NC_2HOP_RESULTS/ogbn-papers100M/$NC_2HOP_CAMPAIGN"
export PAPERS_RUNTIME_ROOT="$HEAD_NVME/fedgraph-runtime/$NC_2HOP_CAMPAIGN"
export PAPERS_GPU_LOGS="$HOME/fedgraph-logs/local-artifact-2hop/$NC_2HOP_CAMPAIGN"
export PAPERS_FEATURE_CACHE_ROOT="${PAPERS_FEATURE_CACHE_ROOT:-$HEAD_NVME/feature-cache/$NC_2HOP_CAMPAIGN}"
export PAPERS_PCA16_CACHE="$PAPERS_FEATURE_CACHE_ROOT/pca16"
export PAPERS_PCA64_CACHE="$PAPERS_FEATURE_CACHE_ROOT/pca64"
export PAPERS_NO_PCA_CACHE="$PAPERS_FEATURE_CACHE_ROOT/none"
export PAPERS_MODEL_ADJ_CACHE="${PAPERS_MODEL_ADJ_CACHE:-$PAPERS_RUNTIME_ROOT/adjacency}"
printf 'Campaign ID %s\n' "$NC_2HOP_CAMPAIGN"

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

Define the launcher once in the same shell. Every Papers100M series uses it;
extra arguments at the end select the modes and round budget. After loading
this function, jump directly to your chosen series in the index.

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

### Submit The Safe First Canary

For an optional first check of a new fleet, submit one round with mmap,
chunking and PCA-16. To only preview the plan, append
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

To count the smoke as successful, check all of the following:

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

For `03-aggregation`, size the head and workers first, then
[derive v2 from the staged v3](#derive-worker-local-v2-shards-for-full-baselines).
Run both paths on the same v2 shards at the same rank. PCA-8 completed on
the tested fleet; full CPU PCA-16 did not fit worker RAM. V3 remains the
default for the other chunked suites.

```bash
run_papers100m_2hop 03-aggregation v2-chunked-cpu-pca8 "$V2_ARTIFACT_DIR" \
  --rounds 3 \
  --graph-storage-mode cpu --pretrain-aggregation-mode chunked \
  --feature-compression-mode shared-pca --feature-compression-rank 8

run_papers100m_2hop 03-aggregation v2-full-cpu-pca8 "$V2_ARTIFACT_DIR" \
  --rounds 3 \
  --graph-storage-mode cpu --pretrain-aggregation-mode full \
  --feature-compression-mode shared-pca --feature-compression-rank 8
```

Full mode receives complete trainer payloads. The chunked in-flight limit
bounds only chunked feature requests and the finalization calls. Watch head
RAM and Ray object-store pressure before promoting a full baseline.

### Run The Projection Sweep

Use the shared ten-worker fleet, staged compact v3 shards, rank-host map, and
[submission shell](#prepare-the-papers100m-submission-shell). This suite does
not need v2 shards or a previous experiment. Use the same code revision and
worker environments for all five cases.

Check the SSH agent, artifact manifest, and Ray resources before submission:

```bash
ssh-add -l
test -r "$V3_ARTIFACT_DIR/manifest.json"
./scripts/manual_ray/status.sh --head-ip "$HEAD_PRIVATE_IP"
df -h "$PWD" /mnt/fedgraph-nvme
```

Expect eleven live nodes and ten free GPUs. Also check the filesystem holding
the active head Ray temp/spill directory. If it is nearly full, restart Ray
with `--ray-tmp-dir /mnt/fedgraph-nvme/ray` on mounted head NVMe, then rejoin the
workers using [the fleet startup steps](#start-the-ray-fleet). Do not remove
files from an active Ray session.

Before **each** case, verify worker NVMe is mounted and use the
[worker disk check](#runtime-storage-and-shutdown). Allow the following
approximate additional GiB **per worker**, plus headroom for Ray and existing
files. The estimates include the expanded int64 adjacency and the file-backed
aggregated features. PCA cases also store projected local features.

| Run order | Projection | Runtime width | Additional NVMe / worker |
|---:|---|---:|---:|
| 1 | PCA-16 | 16 | 20 GiB |
| 2 | PCA-32 | 32 | 24 GiB |
| 3 | PCA-64 | 64 | 32 GiB |
| 4 | PCA-128 | 128 | 48 GiB |
| 5 | None | 128 | 43 GiB |

These files persist between cases. After a driver exits, its logs are archived,
and all GPUs are free, reclaim space as needed from that finished case's
worker-local `pretrain/` and `compression/` directories under
`$PAPERS_RUNTIME_ROOT/04-projection/<case>/`. Do not delete staged shards or
benchmark results.

Define the launcher in the same shell. All cases use training and evaluation
batch size 512, ten rounds, one local step, and seed 42. The helper overrides
the smoke training batch size and keeps mmap, chunked pretraining, the smoke
chunk limits (262,144 rows, 1,000,000 edges, 10 in-flight trainers), and resource
monitoring fixed.

```bash
run_papers100m_projection() {
  local projection="$1"
  shift
  local -a compression_args=(--feature-compression-mode none)
  if [[ "$projection" != none ]]; then
    compression_args=(--feature-compression-mode shared-pca
                      --feature-compression-rank "${projection#pca}")
  fi
  SEED=42 run_papers100m_2hop 04-projection \
    "v3-chunked-mmap-${projection}${PAPERS_CASE_SUFFIX:-}" "$V3_ARTIFACT_DIR" \
    --batch-size 512 --rounds 10 \
    --graph-storage-mode mmap --pretrain-aggregation-mode chunked \
    "${compression_args[@]}" "$@"
}

run_papers100m_projection pca16
```

Run the remaining commands **one at a time** in this order. Before moving to
the next case, require `status=completed`, ten recorded rounds, the expected
`pretrain_feature_dim`, ten prepared samplers, and a final test result in
`summary.jsonl`. Wait for GPU log collection, check worker `MemAvailable` and
disk space, and confirm ten GPUs are free again.

```bash
run_papers100m_projection pca32
run_papers100m_projection pca64
run_papers100m_projection pca128
run_papers100m_projection none
```

PCA-128 rotates the 128 features without reducing their width. `none` is the
unprojected full-width control. Keep both for the comparison. A failed case
should retain its logs, and a retry should use a new name, for example
`PAPERS_CASE_SUFFIX=-rep02 run_papers100m_projection pca128`. To preview a case
without Ray, append `--dry-run --output-root "$(mktemp -d)"`.

After the sweep, copy the entire `04-projection` suite to the analysis host.
From the repo root there, set `PAPERS_HEAD_SSH` to the Papers100M head's SSH
address and `NC_2HOP_CAMPAIGN` to the campaign used on the head:

```bash
p="benchmark/results/nc_2hop_optimization/ogbn-papers100M/$NC_2HOP_CAMPAIGN/04-projection"
mkdir -p "$p"
rsync -av --partial "${PAPERS_HEAD_SSH}:/home/ubuntu/fedgraph/$p/" "$p/"
```

### Run The Chunk-Tuning Sweep

Use the [case matrix](2hop_optimization_status.md#papers100m-chunk-tuning)
to compare one limit at a time after the shared fleet and
[submission shell](#prepare-the-papers100m-submission-shell) setup. No v2
conversion or projection series run is needed. This series runs its own control.

```bash
ssh-add -l
test -r "$V3_ARTIFACT_DIR/manifest.json"
./scripts/manual_ray/status.sh --head-ip "$HEAD_PRIVATE_IP"
```

Expect eleven live Ray nodes and ten free GPUs. Keep the head's Ray spill
directory on mounted NVMe and keep its object-store budget fixed. Before
each case, confirm the workers' NVMe mounts and use the
[worker disk check](#runtime-storage-and-shutdown). The suite adds about
20 GiB of runtime files per worker per case even though peak RAM varies.
Aim for at least 60 GiB free per worker before starting each case. After
collecting results and confirming the fleet is idle, reclaim space only from
finished worker-local case directories under
`$PAPERS_RUNTIME_ROOT/05-chunk-tuning/` as needed.

Define the helper in the same shell. The positional arguments are case suffix,
global rows, trainer edges, and in-flight trainer cap. The other arguments fix
PCA-16, `mmap`, indexed chunked pretraining, training/evaluation batch size 512,
three rounds, one local step, seed 42, and the existing monitoring.

```bash
run_papers100m_chunk_tuning() {
  local label="$1" rows="$2" edges="$3" inflight="$4"
  shift 4
  SEED=42 run_papers100m_2hop 05-chunk-tuning \
    "v3-chunked-mmap-pca16-${label}" "$V3_ARTIFACT_DIR" \
    --batch-size 512 --rounds 3 \
    --graph-storage-mode mmap --pretrain-aggregation-mode chunked \
    --feature-compression-mode shared-pca --feature-compression-rank 16 \
    --pretrain-chunk-rows "$rows" --pretrain-chunk-edges "$edges" \
    --pretrain-max-inflight-trainers "$inflight" "$@"
}

run_papers100m_chunk_tuning baseline 262144 1000000 10
```

Check `summary.jsonl` for `completed`, three rounds, width 16, ten prepared
samplers, and final test metrics. Check the basis hash against the baseline and
the server chunk count in `NC_PRETRAIN_CHUNK_SUMMARY`. Wait for GPU log
collection and all ten GPUs to be free again. Then run the remaining core
cases **one at a time**, comparing each to the baseline:

```bash
run_papers100m_chunk_tuning rows-131072 131072 1000000 10
run_papers100m_chunk_tuning rows-524288 524288 1000000 10
run_papers100m_chunk_tuning edges-500000 262144 500000 10
run_papers100m_chunk_tuning edges-2000000 262144 2000000 10
run_papers100m_chunk_tuning inflight-4 262144 1000000 4
```

Try `run_papers100m_chunk_tuning rows-1048576 1048576 1000000 10` only if
`rows-524288` improves pretraining time with comfortable memory and spill
headroom. To preview any command, append `--dry-run --output-root "$(mktemp -d)"`.
Use new suffixes such as `baseline-rep02` for repeats. Leave the adjacency
cache case-local for every comparison so none reuses another case's expansion.

Save `ray memory --address "$HEAD_PRIVATE_IP:6379" --stats-only` output before,
during, and after each case from a second shell. Treat spill totals as
cluster-wide counters and compare their per-case changes. Keep the benchmark's
head and worker process high-water marks and label `MemAvailable` readings as
samples, not exact peaks. Compare pretraining wall time and its adjacency
transition separately. Three rounds are a correctness check, not a rank or
accuracy sweep. Repeat the baseline and best feasible variant before claiming
a speedup.

Copy back the suite from the analysis host's repo root, with
`PAPERS_HEAD_SSH` set to the head's SSH address and the same campaign ID:

```bash
p="benchmark/results/nc_2hop_optimization/ogbn-papers100M/$NC_2HOP_CAMPAIGN/05-chunk-tuning"
mkdir -p "$p"
rsync -av --partial "${PAPERS_HEAD_SSH}:/home/ubuntu/fedgraph/$p/" "$p/"
```

### Run The Papers100M Rank-Quality Sweep

Follow the [case matrix](2hop_optimization_status.md#papers100m-rank-quality)
after the shared fleet and
[submission shell](#prepare-the-papers100m-submission-shell) setup. This series
uses compact v3, computes its own features, and needs no earlier series.

```bash
ssh-add -l
test -r "$V3_ARTIFACT_DIR/manifest.json"
findmnt --mountpoint /mnt/fedgraph-nvme
df -h "$PWD" /mnt/fedgraph-nvme
./scripts/manual_ray/status.sh --head-ip "$HEAD_PRIVATE_IP"
```

Expect eleven live Ray nodes and ten free GPUs. Keep the head's Ray temp/spill
directory on NVMe and its object-store budget fixed across cases.
Before each case, verify the worker mounts and check
[worker disk space](#runtime-storage-and-shutdown). Allow at least 60 GiB free
per worker: the historical projection sweep used roughly 20 GiB, 32 GiB, and
43 GiB per worker in new runtime files for PCA-16, PCA-64, and no PCA,
respectively. They are written before training, so their sizes do not scale
with the round budget, but they remain on disk after each case. Preserve all
ten finalized `feature-aggregation.f32` and `feature-aggregation.json` pairs
for each retained case under `$PAPERS_RUNTIME_ROOT/06-rank-quality/<case>/pretrain/`.
Keep the matching v3 artifact fingerprint and the PCA basis when applicable.
Verify an off-instance copy before deleting any retained worker cache. Only
clean finished cases when the fleet is idle; leave staged shards, logs, and
active Ray files alone.

Define the launcher in the same shell. It runs 800 fixed rounds, one local
step per round, training/evaluation batch size 512, mmap storage, and indexed
chunked pretraining. It inherits 262,144 rows, 1,000,000 edges, and 10 in-flight
trainers from the base helper. Validation still runs every round and test at
the end. Manual resource snapshots cover the first round, every 25 rounds,
and the phase boundaries. Raw `nvidia-smi` logging stays enabled.

```bash
run_papers100m_rank_quality() {
  local seed="$1" projection="$2"
  shift 2
  local -a compression_args
  case "$projection" in
    none) compression_args=(--feature-compression-mode none) ;;
    pca16|pca64)
      compression_args=(--feature-compression-mode shared-pca
                        --feature-compression-rank "${projection#pca}") ;;
    *) printf 'Expected none, pca16, or pca64; got %s\n' "$projection" >&2; return 2 ;;
  esac
  SEED="$seed" run_papers100m_2hop 06-rank-quality \
    "v3-chunked-mmap-${projection}-seed${seed}${PAPERS_CASE_SUFFIX:-}" \
    "$V3_ARTIFACT_DIR" \
    --batch-size 512 --rounds 800 \
    --graph-storage-mode mmap --pretrain-aggregation-mode chunked \
    --resource-snapshot-interval-rounds 25 "${compression_args[@]}" "$@"
}

run_papers100m_rank_quality 42 pca16 --dry-run --output-root "$(mktemp -d)"
run_papers100m_rank_quality 42 pca16
```

After the first real case completes, verify the summary. For the other cases,
use the corresponding case name and expected width (64 for `pca64`, 128 for
`none`). Also verify that `round_metrics.csv` contains round-800 validation
accuracy and loss, and inspect the resource and GPU summaries before continuing.

```bash
case_dir="$PAPERS_RESULT_ROOT/06-rank-quality/v3-chunked-mmap-pca16-seed42"
jq -e --argjson width 16 '
  .status == "completed" and .rounds_recorded == 800 and
  .evaluation_split == "validation" and .artifact_version == 3 and
  .pretrain_feature_dim == $width and
  .neighbor_sampler_prepared_trainer_count == 10 and
  (.test_acc_final | type == "number") and (.test_loss_final | type == "number")
' "$case_dir/summary.jsonl"
```

Run the other two seed-42 cases **one at a time**, checking the result, disk
space, and that all GPUs are free again after each one:

```bash
run_papers100m_rank_quality 42 pca64
run_papers100m_rank_quality 42 none
```

The archived sweep completed the seed-42 triplet only. For an independent
seed comparison, use the updated runner that forwards the seed and rerun the
seed-42 triplet on the same revision. Repeat the order at seeds 43 and 44 and
check the fleet between commands:

```bash
run_papers100m_rank_quality 43 pca16
run_papers100m_rank_quality 43 pca64
run_papers100m_rank_quality 43 none

run_papers100m_rank_quality 44 pca16
run_papers100m_rank_quality 44 pca64
run_papers100m_rank_quality 44 none
```

Preserve failed attempts and use a new label for retries, for example
`PAPERS_CASE_SUFFIX=-rep02 run_papers100m_rank_quality 42 none`. Compare quality
at rounds 100, 200, 400, and 800 using the same runs. Do not use the final test
metrics to select the rank. Copy the entire suite back after the runs, with
`PAPERS_HEAD_SSH` set to the head's SSH address and the same campaign ID:

```bash
p="benchmark/results/nc_2hop_optimization/ogbn-papers100M/$NC_2HOP_CAMPAIGN/06-rank-quality"
mkdir -p "$p"
rsync -av --partial "${PAPERS_HEAD_SSH}:/home/ubuntu/fedgraph/$p/" "$p/"
```

### Prepare A Reusable Feature Cache

Use this path for plaintext, staged compact v3 artifacts with mmap. It opens
the finalized feature files without repeating PCA or cross-trainer feature
exchange. It does not resume model weights or skip adjacency and sampler setup.
Prepare only the cache your chosen case needs. This does not require
`06-rank-quality`, `07-cache-smoke`, or another model experiment.

Choose a projection in the [submission shell](#prepare-the-papers100m-submission-shell):

```bash
CACHE_PROJECTION=none  # Or pca16 or pca64, depending on the case.
FEATURE_CACHE_DIR="$PAPERS_FEATURE_CACHE_ROOT/$CACHE_PROJECTION"
compression_options=(--feature-compression-mode none)
if [[ "$CACHE_PROJECTION" != none ]]; then
  compression_options=(--feature-compression-mode shared-pca
                       --feature-compression-rank "${CACHE_PROJECTION#pca}")
fi
```

**Use an existing cache.** If it is still on the fleet, point
`PAPERS_FEATURE_CACHE_ROOT` at its parent before loading the submission shell.
If it was backed up, restore it as shown below. The v3 manifest must match
the one used to produce the cache. If neither copy exists, build just that
cache with the following one-round job; the features are complete before
training begins.

```bash
case_name="v3-chunked-mmap-$CACHE_PROJECTION"
run_papers100m_2hop cache-setup "$case_name" "$V3_ARTIFACT_DIR" \
  --rounds 1 --local-step 1 --batch-size 16 --evaluation-batch-size 256 \
  --graph-storage-mode mmap --pretrain-aggregation-mode chunked \
  --graph-relabel-cache-dir "$PAPERS_MODEL_ADJ_CACHE" \
  "${compression_options[@]}"
```

After it succeeds, read the run ID and use it to locate the **worker-local**
features. To promote outputs from an older completed run instead, set
`SOURCE_PRETRAIN_DIR` to that run's exact `pretrain/<run-id>` directory and
use its matching v3 artifact and projection. Never mix ranks from different runs.

```bash
run_id=$(jq -er 'select(.status == "completed") | .run_id' \
  "$PAPERS_RESULT_ROOT/cache-setup/$case_name/summary.jsonl")
printf 'Source run: %s\n' "$run_id"
SOURCE_PRETRAIN_DIR="$PAPERS_RUNTIME_ROOT/cache-setup/$case_name/pretrain/$run_id"
```

If the run ID is empty or wrong, stop and check the case before promotion.
Promote each rank on its assigned worker and copy only the common header to
the head. The absolute path is the same across machines, but each worker
holds only its own ranks. Run this from the head with working SSH access:

```bash
promote_papers_feature_cache() {
  local source_dir="$1" cache_dir="$2" rank worker cmd first_worker=""
  test -s "$RANK_HOSTS_FILE" || return 1
  while read -r rank worker; do
    printf -v cmd '%q ' "$HOME/miniconda3/envs/fedgraph312/bin/python" \
      -m scripts.data.prepare_nc_feature_cache promote \
      --artifact-dir "$V3_ARTIFACT_DIR" --pretrain-dir "$source_dir" \
      --cache-dir "$cache_dir" --trainer-id "$rank" --norm-type none
    ssh -n -o BatchMode=yes "ubuntu@$worker" "cd /home/ubuntu/fedgraph && $cmd" || return 1
    first_worker="${first_worker:-$worker}"
  done < <(awk '{ sub(/#.*/, ""); if (NF) print $1, $2 }' "$RANK_HOSTS_FILE")
  [[ -n "$first_worker" ]] || return 1
  mkdir -p "$cache_dir" || return 1
  rsync -as "ubuntu@$first_worker:$cache_dir/feature-cache.json" "$cache_dir/"
}

promote_papers_feature_cache "$SOURCE_PRETRAIN_DIR" "$FEATURE_CACHE_DIR"
```

For a retry on only a missing rank, use the command below on its assigned
worker. Set `RANK` to the number from `RANK_HOSTS_FILE` and re-export the
artifact, source, and cache paths there. Promotion refuses to overwrite an
existing rank. After the missing ranks are ready, copy one worker's matching
`feature-cache.json` to `$FEATURE_CACHE_DIR` on the head if it is not there yet.

```bash
python -m scripts.data.prepare_nc_feature_cache promote \
  --artifact-dir "$V3_ARTIFACT_DIR" \
  --pretrain-dir "$SOURCE_PRETRAIN_DIR" \
  --cache-dir "$FEATURE_CACHE_DIR" \
  --trainer-id "$RANK" \
  --norm-type none
```

Promotion hard-links the finalized float32 payload by default so the same
NVMe does not need a second copy. Use `--copy` across filesystems. Do not edit
hard-linked payloads in place. Existing feature sidecars do not identify the
source graph; only promote files whose source run and matching artifact are
known. PCA mode, width, and basis hash are read from the finalized sidecar.
Keep a separate directory per projection. For the same artifact, PCA and
pretraining are independent of training seed, model head, width, and optimizer.

**Back up or restore a cache.** Retain the head's `feature-cache.json` and
each worker's complete `trainer-NNN/` directory (the `.f32` payload and both
JSON sidecars). Back up the artifact repo/revision alongside it. The full-width
cache can be hundreds of GiB across workers; budget space on EBS, S3, or
another durable store. Do not put these files in Git or assume the Hugging
Face artifact includes them. For a durable directory mounted on the head,
set `CACHE_BACKUP_DIR` to an absolute path on that mounted volume with enough
free space and run:

```bash
(
  set -e
  : "${CACHE_BACKUP_DIR:?Set the path on persistent storage first}"
  test -s "$RANK_HOSTS_FILE"
  mkdir -p "$CACHE_BACKUP_DIR"
  cp "$FEATURE_CACHE_DIR/feature-cache.json" "$CACHE_BACKUP_DIR/"
  while read -r rank worker; do
    printf -v shard 'trainer-%03d' "$rank"
    rsync -as --partial "ubuntu@$worker:$FEATURE_CACHE_DIR/$shard" "$CACHE_BACKUP_DIR/" </dev/null
  done < <(awk '{ sub(/#.*/, ""); if (NF) print $1, $2 }' "$RANK_HOSTS_FILE")
)
```

After staging the identical v3 artifact on a restarted fleet, restore to a
fresh cache path using the **current** mapping. Set `CACHE_BACKUP_DIR` to the
directory with the backup header and all rank directories. This verifies
the feature payload and communication-node hashes on the worker for each
rank, then publishes the head's header:

```bash
(
  set -e
  : "${CACHE_BACKUP_DIR:?Set the backup directory first}"
  test -s "$RANK_HOSTS_FILE"
  while read -r rank worker; do
    printf -v shard 'trainer-%03d' "$rank"
    printf -v mkdir_cmd 'mkdir -p %q' "$FEATURE_CACHE_DIR"
    ssh -n -o BatchMode=yes "ubuntu@$worker" "$mkdir_cmd"
    rsync -as --partial "$CACHE_BACKUP_DIR/feature-cache.json" \
      "$CACHE_BACKUP_DIR/$shard" "ubuntu@$worker:$FEATURE_CACHE_DIR/" </dev/null
    printf -v verify_cmd '%q ' "$HOME/miniconda3/envs/fedgraph312/bin/python" \
      -m scripts.data.prepare_nc_feature_cache verify --artifact-dir "$V3_ARTIFACT_DIR" \
      --cache-dir "$FEATURE_CACHE_DIR" --trainer-id "$rank"
    ssh -n -o BatchMode=yes "ubuntu@$worker" "cd /home/ubuntu/fedgraph && $verify_cmd"
  done < <(awk '{ sub(/#.*/, ""); if (NF) print $1, $2 }' "$RANK_HOSTS_FILE")
  mkdir -p "$FEATURE_CACHE_DIR"
  cp "$CACHE_BACKUP_DIR/feature-cache.json" "$FEATURE_CACHE_DIR/"
)
```

You can also verify an existing cache on its worker for any rank:

```bash
python -m scripts.data.prepare_nc_feature_cache verify \
  --artifact-dir "$V3_ARTIFACT_DIR" \
  --cache-dir "$FEATURE_CACHE_DIR" \
  --trainer-id "$RANK"
```

### Reuse Completed Feature Aggregation

`07-cache-smoke` checks that training reads a prepared cache and does not
exchange pretraining features. Use the cache prepared above and run one round
from the head. Other series can use that cache without running this check:

```bash
run_papers100m_2hop 07-cache-smoke "v3-mmap-cache-$CACHE_PROJECTION" "$V3_ARTIFACT_DIR" \
  --rounds 1 --batch-size 512 \
  --graph-storage-mode mmap --pretrain-aggregation-mode chunked \
  --pretrain-feature-cache-dir "$FEATURE_CACHE_DIR" \
  --graph-relabel-cache-dir "$PAPERS_RUNTIME_ROOT/07-cache-smoke/adjacency" \
  "${compression_options[@]}"
```

To repeat it at another rank, set `CACHE_PROJECTION` and its paths as above.
Use a fresh case name for each run.
The adjacency directory must be separate from the feature cache. A missing
or mismatched rank stops the run rather than falling back to pretraining.
Check `summary.jsonl` for `pretrain_feature_source=cache`, ten cache hits, the
expected feature width, and zero logical pretraining payload. The reported
cache load time covers verification and mmap setup, not every later disk read.

### Compare Papers100M Model Architectures

In the shared [submission shell](#prepare-the-papers100m-submission-shell),
[prepare the PCA-64 cache](#prepare-a-reusable-feature-cache) at
`$PAPERS_PCA64_CACHE`. Neither `07-cache-smoke` nor another series is required.
Use the same compact v3 artifact, ten-rank map, feature cache, and
training-ready adjacency cache for every case. Changing hidden width or adding
LayerNorm does not require a new feature cache.

| Case | Hidden width | LayerNorm after first linear |
|---|---:|---|
| `baseline` | 256 | Off |
| `normalized` | 256 | On |
| `wider` | 512 | Off |
| `normalized-wider` | 512 | On |

The shared shell sets the PCA-64 path and adjacency directory. The first
case can build the training-ready adjacency cache; later cases can reuse it.

```bash
run_papers100m_architecture() {
  local variant="$1" rounds="$2" width=256
  shift 2
  local -a norm=()
  case "$variant" in
    baseline) ;;
    normalized) norm=(--first-layer-norm) ;;
    wider) width=512 ;;
    normalized-wider) width=512; norm=(--first-layer-norm) ;;
    *) printf 'Unknown architecture case: %s\n' "$variant" >&2; return 2 ;;
  esac

  SEED=42 run_papers100m_2hop 08-model-architecture "${variant}-${rounds}r" \
    "$V3_ARTIFACT_DIR" \
    --rounds "$rounds" --batch-size 512 --evaluation-batch-size 512 --local-step 1 \
    --graph-storage-mode mmap --pretrain-aggregation-mode chunked \
    --pretrain-feature-cache-dir "$PAPERS_PCA64_CACHE" \
    --graph-relabel-cache-dir "$PAPERS_MODEL_ADJ_CACHE" \
    --feature-compression-mode shared-pca --feature-compression-rank 64 \
    --hidden-size "$width" "${norm[@]}" --resource-snapshot-interval-rounds 25 "$@"
}

run_papers100m_architecture normalized-wider 1 --dry-run --output-root "$(mktemp -d)"
run_papers100m_architecture normalized-wider 1
```

Check that the one-round run used ten cache hits, zero feature traffic, and
completed without memory errors. Then screen the four cases sequentially:

```bash
for variant in baseline normalized wider normalized-wider; do
  run_papers100m_architecture "$variant" 200 || break
done
```

Every case keeps two layers, SGD at learning rate 0.01, dropout 0.5, validation
each round, and final test. In `summary.jsonl`, verify `hidden_size`,
`first_layer_norm`, and the same `pretrain_feature_cache_id` across cases.
Compare validation curves, peak memory, and training/sync time. Rerun the
baseline and strongest candidates with `800` in place of `200`; these are
fresh models, not resumed weights. If running every case for 800 rounds,
read both the round-200 and round-800 metrics from those runs instead.
Tune the optimizer, learning rate, scheduler, and local-step count separately.

### Tune The Papers100M Optimizer

Use `09-training-recipe` for optimizer and learning-rate comparisons. Go
directly from the shared [submission shell](#prepare-the-papers100m-submission-shell)
and [PCA-64 cache setup](#prepare-a-reusable-feature-cache) to this series.
Synchronize any local code changes and check that all ten workers are ready:

```bash
./scripts/manual_ray/sync-workers.sh --workers-file "$WORKERS_FILE" --install-editable
./scripts/manual_ray/status.sh --head-ip "$HEAD_PRIVATE_IP"
```

Stage 1 uses the `normalized-wider` model, the existing PCA-64 cache, two
layers, training and evaluation batch size 512, and one local step. The LR
is constant. Both optimizers use weight decay `5e-4`, applied according to
the selected optimizer. AdamW retains its moments on each trainer across
rounds; the server still averages only weights.

```bash
run_papers100m_recipe() {
  local model="$1" optimizer="$2" lr="$3" rounds="$4"
  shift 4
  local -a norm=()
  case "$model" in
    normalized-wider) norm=(--first-layer-norm) ;;
    wider) ;;
    *) printf 'Expected normalized-wider or wider, got %s\n' "$model" >&2; return 2 ;;
  esac

  SEED=42 run_papers100m_2hop 09-training-recipe "${model}-${optimizer}-lr${lr}-${rounds}r" \
    "$V3_ARTIFACT_DIR" \
    --rounds "$rounds" --batch-size 512 --evaluation-batch-size 512 --local-step 1 \
    --graph-storage-mode mmap --pretrain-aggregation-mode chunked \
    --pretrain-feature-cache-dir "$PAPERS_PCA64_CACHE" \
    --graph-relabel-cache-dir "$PAPERS_MODEL_ADJ_CACHE" \
    --feature-compression-mode shared-pca --feature-compression-rank 64 \
    --hidden-size 512 "${norm[@]}" --optimizer "$optimizer" --learning-rate "$lr" \
    --resource-snapshot-interval-rounds 25 "$@"
}

run_papers100m_recipe normalized-wider adamw 0.001 2 --resource-snapshot-interval-rounds 1
```

The first round populates AdamW's state. The second trains after receiving
the averaged weights. Verify both rounds completed, with ten cache hits and
populated AdamW state on all trainers after round two:

```bash
smoke="$PAPERS_RESULT_ROOT/09-training-recipe/normalized-wider-adamw-lr0.001-2r"
jq -e '.status == "completed" and .rounds_recorded == 2 and
       .optimizer == "adamw" and .learning_rate == 0.001 and
       .pretrain_feature_cache_hits == 10 and .pretrain_total_traffic_bytes == 0' \
  "$smoke/summary.jsonl"
jq -se '[.[] | select(.source == "trainer" and .round == 2 and
                      .event == "round_train_and_sync_end" and
                      .optimizer_state_bytes > 0) | .trainer_id] | unique | length == 10' \
  "$smoke"/runs/*/fedgraph_logs/resource_snapshots.jsonl
```

Once those checks pass, start a fresh run:

```bash
run_papers100m_recipe normalized-wider adamw 0.001 800
```

Use the following matched SGD `0.01` control in this series. If you already
have an identical `08-model-architecture/normalized-wider-800r` on the same
artifact, seed, and cache, you can refer to it instead without rerunning it:

```bash
run_papers100m_recipe normalized-wider sgd 0.01 800
```

Compare validation accuracy and loss at rounds 200, 400, and 800, and mean
validation accuracy over rounds 701 through 800. Select by validation, not
by the final test result. The smoke does not resume training weights.

### Run The Hyperparameter Sweep

Use `10-hyperparameter-sweep` for configurable training cases. Use the shared
[submission shell](#prepare-the-papers100m-submission-shell) and
[prepare only the cache](#prepare-a-reusable-feature-cache) matching your case
(PCA-64 for the examples below). No results from `08` or `09` are needed.
Pass a unique case name, the matching cache, and normal submit flags. The
helper does not select a model or training recipe.

```bash
run_papers100m_tuning() {
  local name="${1:?Provide a case name}" cache_dir="${2:?Provide a feature cache directory}"
  shift 2
  run_papers100m_2hop 10-hyperparameter-sweep "$name" "$V3_ARTIFACT_DIR" \
    --graph-storage-mode mmap --pretrain-aggregation-mode chunked \
    --pretrain-feature-cache-dir "$cache_dir" \
    --graph-relabel-cache-dir "$PAPERS_MODEL_ADJ_CACHE" \
    --resource-snapshot-interval-rounds 25 "$@"
}

# Run this series' matched control.
control=(
  --first-layer-norm --hidden-size 512 --dropout 0.5
  --optimizer adamw --learning-rate 0.001 --weight-decay 5e-4
  --lr-scheduler constant --rounds 800 --local-step 1
  --batch-size 512 --evaluation-batch-size 512 --seed 42
  --feature-compression-mode shared-pca --feature-compression-rank 64
)
run_papers100m_tuning control-800r "$PAPERS_PCA64_CACHE" "${control[@]}" \
  --dry-run --output-root "$(mktemp -d)"
run_papers100m_tuning control-800r "$PAPERS_PCA64_CACHE" "${control[@]}"

# Check scheduling on two rounds before a long scheduled case.
run_papers100m_tuning cosine-smoke-2r "$PAPERS_PCA64_CACHE" "${control[@]}" \
  --rounds 2 --lr-scheduler cosine --min-learning-rate 0.0001
run_papers100m_tuning cosine-800r "$PAPERS_PCA64_CACHE" "${control[@]}" \
  --lr-scheduler cosine --min-learning-rate 0.0001
```

| Setting | Driver arguments |
|---|---|
| Architecture | `--model-head` (`gcn` or `gcn-skip`), `--first-layer-norm` or `--no-first-layer-norm`, `--hidden-size`, `--dropout` |
| Local optimizer | `--optimizer` (SGD or AdamW), `--learning-rate`, `--weight-decay` |
| LR schedule | `--lr-scheduler` (constant or cosine); cosine takes `--min-learning-rate` (default 0) |
| Budget and sampling | `--rounds`, `--local-step`, `--batch-size`, `--evaluation-batch-size` |
| Features and seed | `--feature-compression-mode`, `--feature-compression-rank`, `--seed` |

Cosine uses the initial LR on round one and approaches its floor over the
fixed round budget. Every local step in a round uses the same LR. Check the
applied rate in each `NC_ROUND_METRIC` stdout record. Recipe fields are
recorded in `config.json` and `summary.jsonl`. Two layers and the sampler
policy stay fixed.

LayerNorm and dropout are model settings and do not change the saved features.
To test rank 16, pass its cache and `--feature-compression-rank 16`; for the
unprojected cache use `--feature-compression-mode none`. Do not use the PCA-64
cache for another rank. A new rank without a saved cache requires a separate
prepare-and-promote run. The driver rejects mismatched mode/rank metadata even
in a dry run.

Start with fixed-LR AdamW candidates (for example 0.0003, 0.001, and 0.003),
then compare cosine at the best initial rate. Use that recipe to compare
LayerNorm off, width, dropout, and weight decay, one change from the same
reference at a time. Test more local steps and larger batches separately;
report validation against updates and wall time as well as rounds. Change
the rank only after picking a few recipes to compare. Confirm the best ones
on additional seeds using the updated runner. Keep the 08/09 outputs as
historical references.

### Compare Papers100M Model Heads

Use `11-model-heads` to compare the existing `gcn` classifier with the
GraphSAGE-inspired `gcn-skip` head. After the shared
[submission shell](#prepare-the-papers100m-submission-shell) setup, use
[the unprojected feature cache](#prepare-a-reusable-feature-cache) at
`$PAPERS_NO_PCA_CACHE`. You do not need results from the hyperparameter sweep.

Both heads start from the same encoder and GCN weights at a fixed seed.
The skip adds a zero-initialized linear projection of a node's own hidden
representation to its GCN logits, before one `log_softmax`. It adds no hops
and does not change pretraining, the artifact, or graph sampling. At hidden
width 1,024 and 172 classes, it adds 176,128 trainable weights, about 0.67 MiB
per model copy in float32, plus optimizer state and temporary logits.

Use the best completed seed-42 recipe from the earlier sweep as a fixed control:
LayerNorm, width 1,024, AdamW, cosine `0.005` to `0.0005`, three local steps,
training batch 768, and evaluation batch 256. The only change in the matched
pair is `--model-head`.

```bash
run_papers100m_head() {
  local head="$1" rounds="$2" seed="${SEED:-42}"
  shift 2
  run_papers100m_2hop 11-model-heads "${head}-${rounds}r-seed${seed}${PAPERS_CASE_SUFFIX:-}" \
    "$V3_ARTIFACT_DIR" \
    --rounds "$rounds" --seed "$seed" --local-step 3 \
    --batch-size 768 --evaluation-batch-size 256 \
    --graph-storage-mode mmap --pretrain-aggregation-mode chunked \
    --pretrain-feature-cache-dir "$PAPERS_NO_PCA_CACHE" \
    --graph-relabel-cache-dir "$PAPERS_MODEL_ADJ_CACHE" \
    --feature-compression-mode none --model-head "$head" \
    --first-layer-norm --hidden-size 1024 --dropout 0.5 \
    --optimizer adamw --learning-rate 0.005 --weight-decay 5e-4 \
    --lr-scheduler cosine --min-learning-rate 0.0005 \
    --resource-snapshot-interval-rounds 25 "$@"
}

run_papers100m_head gcn-skip 10 --dry-run --output-root "$(mktemp -d)"
run_papers100m_head gcn-skip 10 --resource-snapshot-interval-rounds 1
```

The smoke checks memory, weight synchronization, ten cache hits, zero
pretraining traffic, and final evaluation. Its shorter cosine schedule is
not an accuracy comparison. For the matched runs, wait until the fleet is
idle between cases:

```bash
run_papers100m_head gcn 800
run_papers100m_head gcn-skip 800
```

Confirm `model_head`, `pretrain_feature_cache_id`, and all other settings in
`summary.jsonl`. Compare validation curves, final test metrics, CUDA peaks,
and per-round training/sync time and traffic. Repeat on additional seeds
before attributing a small accuracy change to the head.

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
From the analysis host, choose the series directory and set `PAPERS_HEAD_SSH`
to your head's SSH address and `NC_2HOP_CAMPAIGN` to the recorded ID. For
example, to retrieve the model-head comparison from the default results root:

```bash
cd /home/ubuntu/fedgraph
PAPERS_SUITE=11-model-heads
p="benchmark/results/nc_2hop_optimization/ogbn-papers100M/$NC_2HOP_CAMPAIGN/$PAPERS_SUITE"
mkdir -p "$p"
rsync -av --partial "${PAPERS_HEAD_SSH}:/home/ubuntu/fedgraph/$p/" "$p/"
```

Verify the destination has the expected summaries and logs. Save any
[reusable feature caches](#prepare-a-reusable-feature-cache) separately;
copying results does not retain worker runtime data.

When work is complete, stop the `ray-worker-*` tmux session on each worker,
then stop the `ray-head` session. Instance-store artifacts, staged shards,
runtime outputs, and Hugging Face caches are erased by EC2 stop/start.
