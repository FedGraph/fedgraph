# Manual Ray Papers100M Runbook

This runbook covers manual-Ray `ogbn-papers100M` 0-hop FedAvg experiments
while KubeRay is unavailable. It uses one CPU-only Ray head, ten one-GPU
workers, and a configurable number of trainer actors.

Run commands on the head unless a step says otherwise. `sync-workers.sh` copies
intentional uncommitted source changes, so a commit is not needed to run an
experiment.

## Data Paths

The runbook keeps these dataset paths intentionally separate:

1. **Procedure A: staged local artifact** builds a complete 0-hop artifact on
   the head, then stages each shard to its assigned worker.
2. **Procedure B: manifest-style Hugging Face artifact** publishes that same
   artifact to one dataset repository; each trainer downloads its own shard.
3. **Procedure C: legacy Hugging Face datasets** preserves the historical
   one-repository-per-trainer, labeled-node-only workflow.

## Shared Fleet Setup

### Prerequisites


* Put the head and workers in the same VPC/security group. Permit private
  node-to-node TCP on `6379` and `10000-10100`; do not expose Ray ports publicly.
* The head must SSH to every worker as `ubuntu`.
* Each worker needs one GPU, the persistent `fedgraph312` environment, and at
  least 60 GiB free in its local cache. Multiple trainer actors may share one
  worker when their Ray CPU and fractional-GPU requests fit its resources.
* Store a read-only Hugging Face token locally on every worker if anonymous
  downloads might be rate limited:

```bash
install -d -m 700 ~/.config/fedgraph
printf 'export HF_TOKEN=%q\n' "$HF_TOKEN" > ~/.config/fedgraph/hf.env
chmod 600 ~/.config/fedgraph/hf.env
```

The EBS root disk persists the driver, Miniconda, environment, and checkout
across EC2 stop/start. Instance-store NVMe does not: its data is erased after a
stop/start and must be initialized again before Ray workers start.

### Inventory The Workers

```bash
cd /home/ubuntu/fedgraph
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedgraph312

export HEAD_PRIVATE_IP="$(hostname -I | awk '{print $1}')"
export N_TRAINER=195
export NUM_CPUS_PER_TRAINER=0
export NUM_GPUS_PER_TRAINER=0.05
export WORKERS_FILE=$HOME/fedgraph-papers100m-workers.txt
export RANK_HOSTS_FILE=$HOME/fedgraph-papers100m-rank-hosts.txt
nano "$WORKERS_FILE"

awk 'NF && $1 !~ /^#/' "$WORKERS_FILE" | sort -u | wc -l
```

The file has one physical worker private IP per line; the current fleet has ten.
Do not include the head. Blank lines and `#` comments are allowed. Recheck IPs
in EC2 after any stop/start. `N_TRAINER` controls the shard and actor count;
the CPU and GPU values above are the settings proven by the 195-trainer run.
For the one-trainer-per-worker 10-trainer run, set both resource values to `1`.

Load the SSH key if necessary, then confirm access, GPU visibility, and the
storage layout on all workers:

```bash
ssh-add -l || ssh-add ~/.ssh/AWS_FedGraph.pem

while IFS= read -r worker || [[ -n "$worker" ]]; do
  worker="${worker%%#*}"
  worker="${worker//[[:space:]]/}"
  [[ -z "$worker" ]] && continue
  echo "===== $worker ====="
  ssh -o BatchMode=yes ubuntu@"$worker" \
    'hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; lsblk -o NAME,SIZE,TYPE,MODEL,FSTYPE,MOUNTPOINTS' \
    </dev/null
done < "$WORKERS_FILE"
```

The `</dev/null` prevents SSH from consuming the rest of the worker list.

### Provision Or Recover Worker State

For a fresh plain Ubuntu GPU image only, install packages and the driver, then
reboot workers manually:

```bash
./scripts/manual_ray/setup-workers.sh \
  --stage system \
  --workers-file "$WORKERS_FILE" \
  --install-nvidia-driver
```

After `nvidia-smi` works, create or repair the persistent Conda environment:

```bash
./scripts/manual_ray/setup-workers.sh \
  --stage environment \
  --workers-file "$WORKERS_FILE"
```

For fresh instance storage or after EC2 stop/start, initialize every blank NVMe
instance-store volume:

```bash
./scripts/manual_ray/setup-workers.sh \
  --stage nvme \
  --workers-file "$WORKERS_FILE" \
  --nvme-device auto \
  --format-nvme \
  --jobs 2
```

`auto` independently selects exactly one unpartitioned, non-root NVMe device
whose AWS model identifies it as `Instance Storage`. It refuses no or multiple
candidates and never formats the root disk. After an ordinary reboot, rerun the
NVMe command without `--format-nvme`; do not format an existing filesystem.

## Build A Complete N-Trainer 0-Hop Artifact

### Mount Head Instance Storage

Use one blank instance-store device on the preparation head for unpacked OGB
input, the binary cache, temporary splitter files, and the final artifact. It
is erased by EC2 stop/termination. Keep the original download archive on
persistent EBS instead. Never format the root EBS device.

```bash
lsblk -o NAME,SIZE,TYPE,MODEL,FSTYPE,MOUNTPOINTS
findmnt -no SOURCE /
export HEAD_NVME_DEVICE=/dev/nvme1n1
```

Set `HEAD_NVME_DEVICE` only after confirming it is the blank AWS instance-store
disk. The device name above is an example.

```bash
sudo mkfs.ext4 -F $HEAD_NVME_DEVICE
sudo mkdir -p /mnt/fedgraph-nvme
sudo mount $HEAD_NVME_DEVICE /mnt/fedgraph-nvme
sudo chown ubuntu:ubuntu /mnt/fedgraph-nvme

export HEAD_NVME=/mnt/fedgraph-nvme
export OGB_PARENT=$HEAD_NVME/ogb
export BINARY_CACHE_DIR=$HEAD_NVME/cache/papers100m-ogb-binary
export ARTIFACT_DIR=$HEAD_NVME/artifacts/papers100m-${N_TRAINER}-0hop-v1
mkdir -p $OGB_PARENT $BINARY_CACHE_DIR $HEAD_NVME/artifacts
df -h $HEAD_NVME
```

After an ordinary reboot, remount the existing filesystem without formatting.
After stop/start, inspect it again and initialize the blank instance store.

### Download The Official Binary Dataset

Run this on the head. The new splitter uses the official binary arrays, not
the legacy Hugging Face shards.

#### Store The Download ZIP On Persistent EBS

The Papers100M outer ZIP is about 56.2 GiB. It needs at least 60 GiB free
space. Provision the head's gp3 root EBS volume with at least 80 GiB in
practice to leave filesystem and recovery margin.

The commands below keep the durable ZIP in a directory on the head's persistent
EBS root volume. Confirm that the root volume has at least 60 GiB free; no
additional device formatting or mounting is required.

```bash
export PERSISTENT_EBS=$HOME/fedgraph-ebs
export DOWNLOAD_DIR=$PERSISTENT_EBS/papers100m-downloads
mkdir -p $PERSISTENT_EBS $DOWNLOAD_DIR
df -h $PERSISTENT_EBS $HEAD_NVME
```

```bash
cd /home/ubuntu/fedgraph
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedgraph312
python -m pip install -e . --no-deps

export PAPERS100M_ZIP=$DOWNLOAD_DIR/papers100M-bin.zip
curl -fL --retry 5 --retry-delay 10 -C - \
  -o $PAPERS100M_ZIP \
  https://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip
unzip -t $PAPERS100M_ZIP
unzip -q $PAPERS100M_ZIP -d $OGB_PARENT

RAW_DATA_PATH=$(find $OGB_PARENT -type f -path '*/raw/data.npz' -print -quit)
test -n $RAW_DATA_PATH
export DATASET_ROOT=$(dirname $(dirname $RAW_DATA_PATH))
printf 'DATASET_ROOT=%s\n' $DATASET_ROOT

for file in raw/data.npz raw/node-label.npz \
  split/time/train.csv.gz split/time/valid.csv.gz split/time/test.csv.gz
do
  test -f $DATASET_ROOT/$file || { echo Missing $DATASET_ROOT/$file >&2; exit 1; }
done
```

Keep the ZIP on persistent EBS after the distributed smoke test passes. After
a head stop/start, recreate the NVMe workspace, unzip this retained source back
to $OGB_PARENT, and rerun the splitter rather than downloading again.

### Split A Complete N-Trainer Artifact

Use tmux because splitting and checksumming are long-running. Checksums must
stay enabled because both staged distribution and publishing verify them.

```bash
tmux new-session -s papers100m-split
```

Inside tmux:

```bash
cd /home/ubuntu/fedgraph
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedgraph312
export N_TRAINER=195
export HEAD_NVME=/mnt/fedgraph-nvme
export OGB_PARENT=$HEAD_NVME/ogb
export BINARY_CACHE_DIR=$HEAD_NVME/cache/papers100m-ogb-binary
export ARTIFACT_DIR=$HEAD_NVME/artifacts/papers100m-${N_TRAINER}-0hop-v1
RAW_DATA_PATH=$(find $OGB_PARENT -type f -path '*/raw/data.npz' -print -quit)
test -n $RAW_DATA_PATH
export DATASET_ROOT=$(dirname $(dirname $RAW_DATA_PATH))

time python scripts/data/partition_ogbn_nc.py \
  --dataset-root $DATASET_ROOT \
  --input-format ogb-binary \
  --binary-cache-dir $BINARY_CACHE_DIR \
  --output-dir $ARTIFACT_DIR \
  --n-trainer $N_TRAINER \
  --iid-beta 10000 \
  --seed 42 \
  --chunk-rows 100000
```

### Validate The Artifact And Map Ranks To Workers

After the splitter finishes, it has already validated every shard. Confirm the
expected manifest, then create the strict trainer-rank placement map for
**Procedure A**. **Procedure B** does not use this map: each trainer selects
its shard by trainer rank in the Hugging Face repository.

```bash
python - $ARTIFACT_DIR $N_TRAINER <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads((Path(sys.argv[1]) / 'manifest.json').read_text())
expected_n_trainer = int(sys.argv[2])
assert manifest['input_format'] == 'ogb-binary'
assert manifest['hop_semantics'] == 0
assert manifest['n_trainer'] == expected_n_trainer
assert len(manifest['shards']) == expected_n_trainer
print(manifest['global_node_num'], manifest['global_edge_num'])
PY

export WORKERS_FILE=$HOME/fedgraph-papers100m-workers.txt
export RANK_HOSTS_FILE=$HOME/fedgraph-papers100m-rank-hosts.txt
./scripts/manual_ray/generate-rank-hosts.sh \
  --workers-file $WORKERS_FILE \
  --n-trainer $N_TRAINER \
  --output-file $RANK_HOSTS_FILE
test $(wc -l < $RANK_HOSTS_FILE) -eq $N_TRAINER
cat $RANK_HOSTS_FILE
```

Each line is rank then private worker IP. The helper assigns ranks round-robin
across the worker-file order, so 195 ranks over ten workers produce five workers
with 20 ranks and five with 19. Keep this file unchanged for staging and
submission: rank 0 receives `trainer-000`, and so forth.


## Procedure A: Staged Local Artifact Distribution

### Synchronize Source And Stage Assigned Shards

Every worker must have its local NVMe mounted at `/mnt/fedgraph-nvme` from the
shared setup. The same `ARTIFACT_DIR` path is used on all machines, but each
worker receives only `manifest.json` and the shard(s) assigned to it.

```bash
cd /home/ubuntu/fedgraph
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedgraph312
ssh-add -l || ssh-add ~/.ssh/AWS_FedGraph.pem

./scripts/manual_ray/sync-workers.sh \
  --workers-file $WORKERS_FILE \
  --install-editable

./scripts/manual_ray/stage-local-artifact-shards.sh \
  --rank-hosts-file $RANK_HOSTS_FILE \
  --source-dir $ARTIFACT_DIR \
  --remote-dir $ARTIFACT_DIR
```

Use `--install-editable` only after a fresh worker environment. The staging
helper verifies the SHA-256 metadata for every copied tensor.

### Start Ray And Run The Local-Artifact Smoke Test

Start the CPU-only head and preflight workers:

```bash
export HEAD_PRIVATE_IP=$(hostname -I | awk '{print $1}')
mkdir -p $HOME/fedgraph-logs
tmux new-session -d -s ray-head \
  "cd /home/ubuntu/fedgraph && ./scripts/manual_ray/start-head.sh \
  --head-ip $HEAD_PRIVATE_IP --num-cpus 8 \
  2>&1 | tee /home/ubuntu/fedgraph-logs/ray-head-startup.log"
sleep 3

./scripts/manual_ray/launch-workers.sh \
  --preflight \
  --head-ip $HEAD_PRIVATE_IP \
  --workers-file $WORKERS_FILE \
  --num-cpus 12 \
  --hf-home /mnt/fedgraph-nvme/hf-cache

./scripts/manual_ray/launch-workers.sh \
  --head-ip $HEAD_PRIVATE_IP \
  --workers-file $WORKERS_FILE \
  --num-cpus 12 \
  --hf-home /mnt/fedgraph-nvme/hf-cache

./scripts/manual_ray/status.sh --head-ip $HEAD_PRIVATE_IP
```

Expect eleven Ray nodes and 10.0/10.0 GPU free. The local-artifact workflow
does not download Hugging Face data during training.

Create a separate tmux session for the driver, then run:

```bash
tmux new-session -s papers100m-local-artifact
cd /home/ubuntu/fedgraph
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedgraph312
export HEAD_PRIVATE_IP=$(hostname -I | awk '{print $1}')
export N_TRAINER=195
export NUM_CPUS_PER_TRAINER=0
export NUM_GPUS_PER_TRAINER=0.05
export RANK_HOSTS_FILE=$HOME/fedgraph-papers100m-rank-hosts.txt
export ARTIFACT_DIR=/mnt/fedgraph-nvme/artifacts/papers100m-${N_TRAINER}-0hop-v1

./scripts/manual_ray/submit-local-artifact-0hop.sh \
  --head-ip $HEAD_PRIVATE_IP \
  --artifact-dir $ARTIFACT_DIR \
  --rank-hosts-file $RANK_HOSTS_FILE \
  --dataset ogbn-papers100M \
  --n-trainer $N_TRAINER \
  --num-cpus-per-trainer $NUM_CPUS_PER_TRAINER \
  --num-gpus-per-trainer $NUM_GPUS_PER_TRAINER \
  --batch-size 16 \
  --rounds 3 \
  --local-step 1 \
  --evaluation-split validation \
  --resource-monitor-mode manual \
  --resource-snapshot-interval-rounds 1 \
  --gpu-monitor-detail raw \
  --gpu-sample-interval-seconds 1 \
  --experiment-name nc_papers100m_${N_TRAINER}trainer_0hop_local_artifact_smoke
```

Raw GPU monitoring derives the unique physical workers from the second column
of `RANK_HOSTS_FILE`. A placement map with many trainer ranks on one host still
starts only one `nvidia-smi` sampler on that host. The submit helper stops the
samplers on normal completion, failure, or interruption, then copies their
summaries and compressed CSV files to:

```text
<output-root>/<experiment-name>/gpu_metrics/<worker-ip>/
```

### Run The Intended Budget And Collect Results

After the smoke passes, resubmit with a new name and the intended values, for
example batch size 512, 100 rounds, and local step 3:

```bash
./scripts/manual_ray/submit-local-artifact-0hop.sh \
  --head-ip $HEAD_PRIVATE_IP \
  --artifact-dir $ARTIFACT_DIR \
  --rank-hosts-file $RANK_HOSTS_FILE \
  --dataset ogbn-papers100M \
  --n-trainer $N_TRAINER \
  --num-cpus-per-trainer $NUM_CPUS_PER_TRAINER \
  --num-gpus-per-trainer $NUM_GPUS_PER_TRAINER \
  --batch-size 512 \
  --rounds 100 \
  --local-step 3 \
  --evaluation-split validation \
  --resource-monitor-mode manual \
  --resource-snapshot-interval-rounds 10 \
  --gpu-monitor-detail raw \
  --gpu-sample-interval-seconds 1 \
  --experiment-name nc_papers100m_${N_TRAINER}trainer_0hop_local_artifact_bs512_100r
```

Results are under `benchmark/results/nc_batch_size_convergence/NAME`. Inspect
the driver output, `runs/*/fedgraph_logs` application snapshots, and Ray status
before stopping worker or head instances.


## Procedure B: Manifest-Style Hugging Face Artifact Repository

Use this procedure to test the new publisher and trainer-side selective loader.
It starts from the artifact generated in **Build A Complete N-Trainer 0-Hop
Artifact**. It is intentionally different from the legacy `--use-huggingface`
path: one repository holds `manifest.json` and every `shards/trainer-NNN/`
directory, while each trainer downloads only its own directory.

The previous preparation head was closed, so its instance-store artifact no
longer exists. Start by rerunning **Mount Head Instance Storage**, then reuse
the durable Papers100M ZIP on EBS to complete **Download The Official Binary
Dataset** and **Split A Complete N-Trainer Artifact**. This avoids another
network download but necessarily rebuilds the lost artifact.

### Publish The Validated Artifact From The Head

Publishing needs a Hugging Face token with write access to the target
organization or account. The uploader first checks the manifest, every file
size, and every SHA-256 checksum, then uses Hugging Face's resumable large-folder
upload. It creates a private repository by default; rerunning the command for
the same repository resumes or updates it rather than changing the legacy
per-trainer repositories.

```bash
cd /home/ubuntu/fedgraph
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedgraph312

export N_TRAINER=195
export ARTIFACT_DIR=/mnt/fedgraph-nvme/artifacts/papers100m-${N_TRAINER}-0hop-v1
export HF_ARTIFACT_REPO=FedGraph/fedgraph_ogbn-papers100M_${N_TRAINER}trainer_0hop_iid_beta_10000.0_v1

read -rsp 'Hugging Face write token: ' HF_TOKEN
printf '\n'
export HF_TOKEN

python scripts/data/upload_nc_artifact_to_hf.py \
  --artifact-dir "$ARTIFACT_DIR" \
  --repo-id "$HF_ARTIFACT_REPO" \
  --private \
  --num-workers 4

unset HF_TOKEN
```

Do not use `--skip-checksum-verification` for the Papers100M artifact. Use
`--public` only when the dataset is intentionally public.

### Install A Read Token On Every Worker Before Ray Starts

For a private repository, each worker needs a token that can read the artifact.
Use a read-only token when possible. `start-worker.sh` sources
`~/.config/fedgraph/hf.env` before it launches the persistent Ray worker
process, so installing this file *after* `launch-workers.sh` is too late for
trainer actors already running inside Ray.

```bash
read -rsp 'Hugging Face read token: ' HF_READ_TOKEN
printf '\n'

install -d -m 700 ~/.config/fedgraph
printf 'export HF_TOKEN=%q\n' "$HF_READ_TOKEN" > ~/.config/fedgraph/hf.env
chmod 600 ~/.config/fedgraph/hf.env

while IFS= read -r worker || [[ -n "$worker" ]]; do
  worker="${worker%%#*}"
  worker="${worker//[[:space:]]/}"
  [[ -z "$worker" ]] && continue
  ssh -o BatchMode=yes ubuntu@"$worker" 'install -d -m 700 ~/.config/fedgraph'
  rsync -az ~/.config/fedgraph/hf.env \
    ubuntu@"$worker":.config/fedgraph/hf.env
  ssh -o BatchMode=yes ubuntu@"$worker" 'chmod 600 ~/.config/fedgraph/hf.env'
done < "$WORKERS_FILE"

unset HF_READ_TOKEN
```

The cache path passed later is interpreted on each worker, not on the head. A
worker keeps only the rank-specific shard(s) it has executed in its local
Hugging Face cache.

### Synchronize Source, Start Ray, And Run A Smoke Test

This path does **not** use `stage-local-artifact-shards.sh`,
`--local-artifact-dir`, or `--local-artifact-rank-hosts`. The trainer rank
selects the matching remote shard, so no worker-to-rank placement map is
required.

```bash
cd /home/ubuntu/fedgraph
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedgraph312

./scripts/manual_ray/sync-workers.sh \
  --workers-file "$WORKERS_FILE" \
  --install-editable

export HEAD_PRIVATE_IP="$(hostname -I | awk '{print $1}')"
mkdir -p "$HOME/fedgraph-logs"
tmux new-session -d -s ray-head \
  "cd /home/ubuntu/fedgraph && ./scripts/manual_ray/start-head.sh \
  --head-ip $HEAD_PRIVATE_IP --num-cpus 8 \
  2>&1 | tee /home/ubuntu/fedgraph-logs/ray-head-startup.log"
sleep 3

./scripts/manual_ray/launch-workers.sh \
  --preflight \
  --head-ip "$HEAD_PRIVATE_IP" \
  --workers-file "$WORKERS_FILE" \
  --num-cpus 12 \
  --hf-home /mnt/fedgraph-nvme/hf-cache

./scripts/manual_ray/launch-workers.sh \
  --head-ip "$HEAD_PRIVATE_IP" \
  --workers-file "$WORKERS_FILE" \
  --num-cpus 12 \
  --hf-home /mnt/fedgraph-nvme/hf-cache

./scripts/manual_ray/status.sh --head-ip "$HEAD_PRIVATE_IP"
```

In a separate tmux session, submit the smoke run:

```bash
tmux new-session -s papers100m-hf-local-artifact
```

Inside that tmux session:

```bash
cd /home/ubuntu/fedgraph
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedgraph312

export HEAD_PRIVATE_IP="$(hostname -I | awk '{print $1}')"
export RAY_ADDRESS="$HEAD_PRIVATE_IP:6379"
export PYTHONPATH=/home/ubuntu/fedgraph
export N_TRAINER=195
export NUM_CPUS_PER_TRAINER=0
export NUM_GPUS_PER_TRAINER=0.05
export HF_ARTIFACT_REPO=FedGraph/fedgraph_ogbn-papers100M_${N_TRAINER}trainer_0hop_iid_beta_10000.0_v1

python benchmark/benchmark_NC_batch_size_convergence.py \
  --dataset ogbn-papers100M \
  --n-trainer $N_TRAINER \
  --num-hops 0 \
  --hf-local-artifact-repo "$HF_ARTIFACT_REPO" \
  --hf-local-artifact-cache-dir /mnt/fedgraph-nvme/hf-cache \
  --iid-betas 10000 \
  --seeds 42 \
  --num-layers 2 \
  --batch-sizes 16 \
  --rounds 3 \
  --local-step 1 \
  --evaluation-split validation \
  --gpu \
  --num-gpus-per-trainer $NUM_GPUS_PER_TRAINER \
  --num-cpus-per-trainer $NUM_CPUS_PER_TRAINER \
  --server-device cpu \
  --resource-monitor-mode manual \
  --resource-snapshot-interval-rounds 1 \
  --experiment-name nc_papers100m_${N_TRAINER}trainer_0hop_hf_local_artifact_smoke
```

The initial trainer logs should name the manifest-style repository and the
selected `trainer-NNN` shard. Ray's worker-list order does not determine that
rank; FedGraph's trainer ID does. A first run downloads each assigned shard;
subsequent runs reuse the worker-local cache while it persists.

### Run The Intended Manifest-Style Artifact Budget

After the smoke test succeeds, repeat the preceding command with a fresh name
and the intended fixed budget, for example:

```bash
python benchmark/benchmark_NC_batch_size_convergence.py \
  --dataset ogbn-papers100M \
  --n-trainer $N_TRAINER \
  --num-hops 0 \
  --hf-local-artifact-repo "$HF_ARTIFACT_REPO" \
  --hf-local-artifact-cache-dir /mnt/fedgraph-nvme/hf-cache \
  --iid-betas 10000 \
  --seeds 42 \
  --num-layers 2 \
  --batch-sizes 512 \
  --rounds 100 \
  --local-step 3 \
  --evaluation-split validation \
  --gpu \
  --num-gpus-per-trainer $NUM_GPUS_PER_TRAINER \
  --num-cpus-per-trainer $NUM_CPUS_PER_TRAINER \
  --server-device cpu \
  --resource-monitor-mode manual \
  --resource-snapshot-interval-rounds 10 \
  --experiment-name nc_papers100m_${N_TRAINER}trainer_0hop_hf_local_artifact_bs512_100r
```

Compare the split manifest, selected shards, learning curves, final metrics,
resource summaries, and first-download time against the staged-local run. The
first download is part of integration validation; steady-state timing should
use the warmed worker-local cache.


## Procedure C: Legacy One-Repository-Per-Trainer Hugging Face Dataset Path

### Synchronize The Current Checkout

Use `--install-editable` after a fresh worker environment. For later source-only
changes, omit that flag.

```bash
./scripts/manual_ray/sync-workers.sh \
  --workers-file "$WORKERS_FILE" \
  --install-editable
```

It syncs source and scripts, including uncommitted changes, but excludes Git
metadata, datasets, and old results.

### Start The Ray Head

```bash
export HEAD_PRIVATE_IP="$(hostname -I | awk '{print $1}')"
mkdir -p "$HOME/fedgraph-logs"

tmux new-session -d -s ray-head \
  "cd /home/ubuntu/fedgraph && \
   ./scripts/manual_ray/start-head.sh \
     --head-ip ${HEAD_PRIVATE_IP} \
     --num-cpus 8 \
     2>&1 | tee /home/ubuntu/fedgraph-logs/ray-head-startup.log"

sleep 3
tmux list-sessions
tail -n 100 ~/fedgraph-logs/ray-head-startup.log
```

The head script uses `~/fedgraph-ray`, which is writable on the head's EBS root
volume. A live `ray-head` tmux session means Ray's foreground process is alive.

### Preflight And Start All Ray Workers

Preflight first. It checks every worker's local `fedgraph312` environment,
Ray/PyTorch/PyG imports, CUDA/T4 visibility, free NVMe cache, Ray temporary
directory, and TCP access to `${HEAD_PRIVATE_IP}:6379`. It creates no Ray worker
processes.

```bash
./scripts/manual_ray/launch-workers.sh \
  --preflight \
  --head-ip "$HEAD_PRIVATE_IP" \
  --workers-file "$WORKERS_FILE" \
  --num-cpus 12 \
  --hf-home /mnt/fedgraph-nvme/hf-cache
```

Only after all ten workers pass, start the Ray worker processes and verify the
cluster:

```bash
./scripts/manual_ray/launch-workers.sh \
  --head-ip "$HEAD_PRIVATE_IP" \
  --workers-file "$WORKERS_FILE" \
  --num-cpus 12 \
  --hf-home /mnt/fedgraph-nvme/hf-cache

./scripts/manual_ray/status.sh --head-ip "$HEAD_PRIVATE_IP"
```

Expect eleven Ray nodes: one CPU-only head plus ten GPU workers, with
`10.0/10.0 GPU` free. FedGraph requests one GPU per trainer and uses `SPREAD`,
so Ray places one trainer actor on each worker. Worker-list order does not set
trainer rank.

### Submit A Monitored Benchmark

Use a dedicated tmux session so the run survives SSH disconnection. Set the
variables inside that session, rather than depending on tmux-inherited exports.

```bash
tmux new-session -s papers100m-submit
```

Inside that tmux session:

```bash
cd /home/ubuntu/fedgraph
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedgraph312

export HEAD_PRIVATE_IP="$(hostname -I | awk '{print $1}')"
export WORKERS_FILE=$HOME/fedgraph-papers100m-workers.txt
export HEAD_HF_HOME="$HOME/fedgraph-hf-cache"
export HEAD_LOG_DIR="$HOME/fedgraph-logs/papers100m-0hop"
mkdir -p "$HEAD_HF_HOME" "$HEAD_LOG_DIR"

test -f "$WORKERS_FILE"
ssh-add -l || ssh-add ~/.ssh/AWS_FedGraph.pem
./scripts/manual_ray/status.sh --head-ip "$HEAD_PRIVATE_IP"
```

Then submit one batch size and one fixed round budget. The example begins the
next sweep at batch size 256; use a fresh experiment name for every run.

```bash
./scripts/manual_ray/submit-papers100m-0hop.sh \
  --head-ip "$HEAD_PRIVATE_IP" \
  --workers-file "$WORKERS_FILE" \
  --hf-home "$HEAD_HF_HOME" \
  --log-dir "$HEAD_LOG_DIR" \
  --experiment-name nc_hf_papers100m_10trainer_0hop_bs256_10r_monitor \
  --batch-size 256 \
  --rounds 10 \
  --local-step 3 \
  --resource-monitor-mode manual \
  --resource-snapshot-interval-rounds 1 \
  --gpu-monitor-detail raw
```

The submit helper activates `fedgraph312`, exports `RAY_ADDRESS`, `HF_HOME`,
and `PYTHONPATH` for the driver, then starts one experiment-scoped `nvidia-smi`
sampler per worker. It stops, summarizes, and collects them on normal exit,
failure, or Ctrl-C. For longer runs, keep a unique name, set the intended batch
size and rounds, and normally use `--resource-snapshot-interval-rounds 10`.

### Result Locations And Shutdown

For an experiment called `NAME`:

* Driver log: `~/fedgraph-logs/papers100m-0hop/NAME_driver.log`.
* Benchmark output: `benchmark/results/nc_batch_size_convergence/NAME/`.
* Application snapshots: `.../NAME/runs/<run-id>/fedgraph_logs/`.
* Centralized GPU samples and summaries: `.../NAME/gpu_metrics/<worker-ip>/`.

Before another run, check `ray status` still reports ten free GPUs. When all
work is complete, interrupt the ten `ray-worker-*` tmux sessions first, then
`ray-head`; collect results before stopping or terminating EC2 instances.
