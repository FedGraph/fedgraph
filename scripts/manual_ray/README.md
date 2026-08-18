# Manual Ray Papers100M 0-Hop Run

These scripts reproduce the manual multi-instance Ray workflow while EKS and
KubeRay are unavailable. They launch exactly one Papers100M trainer actor on
each one-GPU worker and keep the Ray head CPU-only.

They do not create EC2 instances or AWS credentials. The explicit
`setup-workers.sh` stages can prepare Ubuntu workers, but never reboot an
instance automatically and format NVMe only when the caller passes an explicit
flag. This keeps the account- and AMI-specific actions inspectable.

For the exact, tested end-to-end operational sequence, including worker recovery
after EC2 stop/start, all-worker preflight, tmux handling, and monitored
submission, see [`docs/manual_ray_papers100m_runbook.md`](../../docs/manual_ray_papers100m_runbook.md).

## Recommended fleet

For the current legacy ten-trainer, 0-hop artifact, use:

* one CPU-only Ray head with at least 32 GiB RAM;
* ten one-GPU workers, one trainer actor per worker;
* `g4dn.4xlarge` as the cost-conscious worker choice: one 16 GB T4, 64 GiB
  RAM, and 225 GB local NVMe;
* 8 Ray CPU slots requested per trainer and 12 advertised by each worker;
* 8 GiB Ray object store per node, leaving RAM for the trainer's temporary
  27-28 GiB shard load/relabel peak.

Do not use `g4dn.2xlarge` for this run. Its 32 GiB RAM leaves too little room
for the measured temporary host peak plus Ray and the operating system. A T4's
16 GB VRAM is adequate for the observed 6.34 GiB trainer peak, but validate
one `g4dn.4xlarge` with the 1x1 canary before provisioning the remaining nine.

This sizing applies only to the current 0-hop FedAvg artifact. Do not reuse it
for 1/2-hop FedGCN without a new memory canary.

## Prerequisites

1. Put the head and all workers in the same VPC and a security group that
   permits private node-to-node TCP traffic on `6379` and `10000-10100`.
   Keep those ports closed to the public internet. Allow SSH only from the
   administrator's network or the head security group.
2. Use the same Ubuntu/NVIDIA driver and `fedgraph312` environment on every
   node. The current known-good package versions are PyTorch `2.5.1+cu121`,
   PyG `2.8.0`, PyG CUDA wheels for `pt25cu121`, and Ray `2.56.0`.
3. Give every worker at least 60 GiB free in a local cache directory. The
   `g4dn.4xlarge` instance-store NVMe is a good location. Use the explicit
   `nvme` setup stage below after checking the device name; it formats only a
   caller-selected blank device.
4. Make the head able to SSH to every worker. `sync-workers.sh` and
   `launch-workers.sh` use the existing SSH agent by default. Set `SSH_OPTS`
   only if an identity file or another SSH option is needed.
5. Put a read-only Hugging Face token on each worker to avoid rate limiting:

   ```bash
   install -d -m 700 ~/.config/fedgraph
   printf 'export HF_TOKEN=%q\n' "$HF_TOKEN" > ~/.config/fedgraph/hf.env
   chmod 600 ~/.config/fedgraph/hf.env
   ```

   The worker script reads this file locally and does not send the token in an
   SSH command.

## Worker Bootstrap And Storage Lifecycle

The persistent root EBS disk keeps the NVIDIA driver, Miniconda, and
`fedgraph312` environment across reboots and stop/start cycles. The local NVMe
instance store is different: its contents persist across a reboot, but are lost
when the EC2 instance is stopped or terminated. Its mount table is also cleared
by a reboot. For that reason, the storage helper never writes `/etc/fstab`.

Run the following from the head. `setup-workers.sh` transfers a self-contained
helper with `scp`, so the worker does not need a pre-existing checkout or
`rsync`. It writes one log per worker on the head and defaults to two concurrent
workers to avoid saturating package download bandwidth.

1. On a fresh Ubuntu GPU image without an active NVIDIA driver, install the
   system packages and the recommended driver. A GPU-enabled Deep Learning AMI
   can omit `--install-nvidia-driver`.

```bash
./scripts/manual_ray/setup-workers.sh \
  --stage system \
  --workers-file "$WORKERS_FILE" \
  --install-nvidia-driver
```

   Reboot those workers manually, reconnect, and confirm `nvidia-smi` works.
   Driver installation is intentionally a separate stage because the newly
   installed kernel module cannot be used until after the reboot.

2. Install the persistent Conda environment. The helper is idempotent: it keeps
   a matching installed package stack and repairs a missing or mismatched one.
   It installs the known-good CUDA 12.1 PyTorch/PyG wheels and the exact Ray,
   TenSEAL, Hugging Face, OGB, and FedGraph runtime package versions. OpenFHE
   remains optional for this plaintext run.

```bash
./scripts/manual_ray/setup-workers.sh \
  --stage environment \
  --workers-file "$WORKERS_FILE"
```

3. Inspect the block device name before the first NVMe setup. Never assume the
   name if the instance type or AMI changes.

```bash
ssh ubuntu@WORKER_PRIVATE_IP \
  'lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS'
```

   After confirming each worker has an AWS instance-store volume, use `auto` to
   select it independently on every machine. The helper selects only one
   unpartitioned, non-root NVMe device whose model identifies it as instance
   storage; it refuses no or multiple candidates. `--format-nvme` is required
   for a blank device and is refused when a filesystem already exists.

```bash
./scripts/manual_ray/setup-workers.sh \
  --stage nvme \
  --workers-file "$WORKERS_FILE" \
  --nvme-device auto \
  --format-nvme
```

4. After an ordinary reboot, rerun only the NVMe stage without
   `--format-nvme`; it remounts the existing filesystem and recreates the cache
   directories if necessary. After a stop/start, instance-store data is gone,
   so verify `lsblk` again and rerun with `--format-nvme`.

## Runtime workflow

Run the following on the head unless a step says otherwise. The examples assume
the repository path is `/home/ubuntu/fedgraph`.

1. Create a worker inventory with one private IP per line:

   ```bash
   export HEAD_PRIVATE_IP="$(hostname -I | awk '{print $1}')"
   export WORKERS_FILE=/tmp/fedgraph-papers100m-workers.txt
   printf '%s\n' 10.0.0.21 10.0.0.22 10.0.0.23 > "$WORKERS_FILE"
   ```

   Replace the example addresses with all ten workers.

2. Synchronize the exact current source tree, including uncommitted Papers100M
   compatibility changes, but excluding generated data/results:

   ```bash
   ./scripts/manual_ray/sync-workers.sh --workers-file "$WORKERS_FILE" --install-editable
   ```

   `--install-editable` is required only for this first sync. It runs `python -m pip install -e . --no-deps` inside `fedgraph312` on every worker, registering the synced checkout without changing the already-installed CUDA/PyG packages. Later source-only syncs do not need the flag because an editable install directly uses the updated checkout.

3. Start the head in a detached tmux session:

   ```bash
   tmux new-session -d -s ray-head \
     "cd /home/ubuntu/fedgraph && ./scripts/manual_ray/start-head.sh --head-ip ${HEAD_PRIVATE_IP} --num-cpus 8"
   ```

4. Start all workers remotely. Each receives one GPU and has an independent
   Ray process. GPU sampling is intentionally not started here; it is scoped to
   each submitted experiment instead:

   ```bash
   ./scripts/manual_ray/launch-workers.sh \
     --head-ip "$HEAD_PRIVATE_IP" \
     --workers-file "$WORKERS_FILE" \
     --num-cpus 12
   ```

5. Wait for all nodes, then verify the Ray resource view reports the CPU head,
   ten worker nodes, and exactly ten GPUs:

   ```bash
   ./scripts/manual_ray/status.sh --head-ip "$HEAD_PRIVATE_IP"
   ```

6. First launch a fixed three-round fleet smoke. It uses the legacy `1hop`
   artifact as the 0-hop FedAvg input, mini-batch size 16, three local steps,
   CPU server aggregation, and test-split metrics:

   ```bash
   ./scripts/manual_ray/submit-papers100m-0hop.sh \
     --head-ip "$HEAD_PRIVATE_IP" \
     --rounds 3 \
     --experiment-name nc_hf_papers100m_10trainer_0hop_10gpu_smoke_seed42 \
     --workers-file "$WORKERS_FILE" \
     --gpu-monitor-detail raw
   ```

   The submission script starts one `nvidia-smi` sampler per worker immediately
   before the driver, stops it on normal exit, failure, or Ctrl-C, summarizes
   it remotely, and copies the compressed CSV plus JSON summary to
   `benchmark/results/.../<experiment>/gpu_metrics/`. The benchmark itself
   writes compact trainer/server application snapshots under each run's
   `fedgraph_logs/` directory. `--resource-monitor-mode manual` is the script
   default; use `--resource-snapshot-interval-rounds 1` for every global round
   or keep the default `10` for longer runs.

7. Inspect the driver log, centralized GPU metrics, resource snapshot summary,
   `ray status`, and the result manifest before using the selected full
   fixed-round budget. Run the full experiment with a new name:

   ```bash
   ./scripts/manual_ray/submit-papers100m-0hop.sh \
     --head-ip "$HEAD_PRIVATE_IP" \
     --rounds YOUR_ROUND_BUDGET \
     --experiment-name nc_hf_papers100m_10trainer_0hop_10gpu_seed42 \
     --workers-file "$WORKERS_FILE" \
     --gpu-monitor-detail raw
   ```

## Shutdown

Stop worker tmux sessions first, then the head session. On a node, use
`tmux attach -t SESSION` followed by `Ctrl-C`, or run `ray stop --force` from
the matching environment. Confirm the EC2 instances are terminated after logs
and benchmark outputs have been collected.
