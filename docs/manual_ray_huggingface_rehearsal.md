# Manual Ray and Hugging Face Rehearsal

This runbook describes the non-KubeRay path used while EKS permissions are
unavailable. It keeps scheduling separate from FedGraph: a Ray head runs the
benchmark driver and server logic, while EC2 GPU workers run Ray worker
processes and host trainer actors.

The commands are deliberately explicit. They must be run from the same
FedGraph revision and `fedgraph312` environment on every node.

## Network and storage prerequisites

* Put the head and workers in the same VPC/security group. Do not expose Ray
  ports to the public internet.
* Permit private traffic between cluster nodes for the Ray head port (`6379`)
  and the bounded port range selected below (`10000-10100`). The dashboard is
  bound to head-local `127.0.0.1` and can be reached through an SSH tunnel.
* Give every worker enough local EBS/SSD for its Hugging Face shard and set
  `HF_HOME` to that local path before starting Ray. Each worker downloads only
  the repository for the trainer actor scheduled there.

## Start the head

On the CPU/server EC2 node, replace the placeholder with its private IPv4
address. Keep the command alive in `tmux`, `screen`, `systemd`, or another
supervisor while workers and the driver run.

```bash
cd /home/ubuntu/fedgraph
conda activate fedgraph312

ray start --head --block \
  --node-ip-address="$HEAD_PRIVATE_IP" \
  --port=6379 \
  --num-cpus=8 \
  --dashboard-host=127.0.0.1 \
  --object-manager-port=10000 \
  --node-manager-port=10001 \
  --min-worker-port=10002 \
  --max-worker-port=10100
```

`--block` makes an accidental SSH-shell exit visible instead of silently
leaving an unmanaged background runtime. It is not part of FedGraph itself.
For the real run, set `--num-cpus` to the CPU capacity you want Ray to schedule,
not a deliberately inflated value.

In a separate head-node shell, confirm that the runtime is healthy:

```bash
ray status --address="$HEAD_PRIVATE_IP:6379"
```

## Start each GPU worker

On every GPU EC2 node, set the worker's private IPv4 address and a local cache
path. The code and Python environment must already be present on the worker.

```bash
cd /home/ubuntu/fedgraph
conda activate fedgraph312
export HF_HOME=/mnt/fedgraph-hf-cache

ray start --address="$HEAD_PRIVATE_IP:6379" --block \
  --node-ip-address="$WORKER_PRIVATE_IP" \
  --num-cpus=4 \
  --num-gpus=1 \
  --object-manager-port=10000 \
  --node-manager-port=10001 \
  --min-worker-port=10002 \
  --max-worker-port=10100
```

Run `ray status` again on the head. It should report one head node plus all
GPU workers and the expected aggregate GPU count. Ray uses those resource
advertisements to place `Trainer` actors; it does not impose a VRAM limit.

## Run the validated legacy-artifact smoke test

On the head, point the FedGraph driver to Ray with `RAY_ADDRESS`. Current
`run_NC` calls `ray.init()` without a hard-coded address, so Ray consumes this
environment variable and attaches rather than creating a local runtime.

```bash
cd /home/ubuntu/fedgraph
conda activate fedgraph312
export RAY_ADDRESS="$HEAD_PRIVATE_IP:6379"
export HF_HOME=/mnt/fedgraph-hf-cache

python benchmark/benchmark_NC_batch_size_convergence.py \
  --dataset ogbn-arxiv \
  --n-trainer 5 \
  --num-hops 2 \
  --use-huggingface \
  --hf-artifact-num-hops 1 \
  --pretrain-feature-upload-mode indexed \
  --evaluation-split test \
  --batch-sizes 2048 \
  --rounds 1 \
  --local-step 1 \
  --gpu \
  --num-gpus-per-trainer 1 \
  --server-device cpu \
  --experiment-name nc_hf_legacy_arxiv_5client_2hop_smoke_seed42
```

The artifact suffix is intentionally `1` while runtime `--num-hops` is `2`:
these public legacy repositories used `1hop` for the data layout that the
current two-hop FedGCN path consumes. Hugging Face pretraining must use
`indexed`; dense uploads require a centralized full feature matrix that the
sharded loader deliberately does not create.

`--evaluation-split test` records per-round test loss and accuracy in the
explicit `evaluation_*` columns of `round_metrics.csv`; it does not populate
validation curves or `val_*` fields. It is permitted only with fixed-round
training, so it cannot select a model or stop an elastic run using test data.

This five-trainer arXiv set is about 50 MiB per trainer repository. It lacks
`val_*`, `global_node_num.pt`, and `class_num.pt`; current compatibility code
infers metadata from the shards but has no validation data. Therefore it
validates Ray attachment, actor placement, Hugging Face loading, indexed
pretraining, and training, but is not a convergence-quality result.

## Local result obtained on 2026-07-28

The same command was validated locally with one T4 shared by five actors
(`--num-gpus-per-trainer 0.2`) and six deliberately oversubscribed logical CPU
slots. It completed one two-hop round:

* initialization: 14.58 s, with files already cached;
* indexed pretraining: 2.17 s;
* pure training: 0.74 s; parameter communication: 0.03 s;
* theoretical pretraining traffic: 626.18 MB;
* per-round test evaluation: loss `4.1553`, accuracy `0.0143`.

Its generated artifacts are under
`benchmark/results/nc_batch_size_convergence/nc_hf_legacy_arxiv_5client_2hop_test_eval_smoke2_seed42/`.

## Shutdown

After the driver exits, stop workers first, then the head. With `--block`, use
`Ctrl-C` in each supervised Ray shell. Verify that `ray status` no longer
reports nodes and terminate the EC2 instances when the experiment is finished.
