# Manual Ray Papers100M Runbook

This is the tested process for running the current legacy Hugging Face
`ogbn-papers100M` 0-hop FedAvg experiment while KubeRay is unavailable. It
uses one CPU-only Ray head, ten one-GPU workers, and ten trainer actors.

Run commands on the head unless a step says otherwise. `sync-workers.sh` copies
intentional uncommitted source changes, so a commit is not needed to run an
experiment.

## 1. Prerequisites

* Put the head and workers in the same VPC/security group. Permit private
  node-to-node TCP on `6379` and `10000-10100`; do not expose Ray ports publicly.
* The head must SSH to every worker as `ubuntu`.
* Each worker needs one GPU, the persistent `fedgraph312` environment, and at
  least 60 GiB free in its local cache. The current 0-hop baseline is one
  `g4dn.4xlarge` worker per trainer.
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

## 2. Inventory The Workers

```bash
cd /home/ubuntu/fedgraph
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedgraph312

export HEAD_PRIVATE_IP="$(hostname -I | awk '{print $1}')"
export WORKERS_FILE=/tmp/fedgraph-papers100m-workers.txt
nano "$WORKERS_FILE"

awk 'NF && $1 !~ /^#/' "$WORKERS_FILE" | sort -u | wc -l
```

The file has exactly ten worker private IPs, one per line. Do not include the
head. Blank lines and `#` comments are allowed. Recheck IPs in EC2 after any
stop/start; the count command must print `10`.

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

## 3. Provision Or Recover Worker State

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

## 4. Synchronize The Current Checkout

Use `--install-editable` after a fresh worker environment. For later source-only
changes, omit that flag.

```bash
./scripts/manual_ray/sync-workers.sh \
  --workers-file "$WORKERS_FILE" \
  --install-editable
```

It syncs source and scripts, including uncommitted changes, but excludes Git
metadata, datasets, and old results.

## 5. Start The Ray Head

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

## 6. Preflight And Start All Ray Workers

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

## 7. Submit A Monitored Benchmark

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
export WORKERS_FILE=/tmp/fedgraph-papers100m-workers.txt
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

## 8. Result Locations And Shutdown

For an experiment called `NAME`:

* Driver log: `~/fedgraph-logs/papers100m-0hop/NAME_driver.log`.
* Benchmark output: `benchmark/results/nc_batch_size_convergence/NAME/`.
* Application snapshots: `.../NAME/runs/<run-id>/fedgraph_logs/`.
* Centralized GPU samples and summaries: `.../NAME/gpu_metrics/<worker-ip>/`.

Before another run, check `ray status` still reports ten free GPUs. When all
work is complete, interrupt the ten `ray-worker-*` tmux sessions first, then
`ray-head`; collect results before stopping or terminating EC2 instances.
