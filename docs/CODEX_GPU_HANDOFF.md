# Codex GPU Handoff For FedGraph Batch-Size Benchmark

Use this file to continue the FedGraph NC batch-size benchmark work from a new GPU EC2 instance and a fresh Codex session.

## Current Repository State

- Branch: `update_hc`
- Latest relevant commit: `2abebc4 Add elastic NC training checkpoints`
- Generated artifacts are intentionally untracked:
  - `benchmark/results/`
  - `dataset/`

Recent relevant commits:

```text
2abebc4 Add elastic NC training checkpoints
660e5bf Refine NC batch-size benchmark metrics
c4c1f25 Fix NC mini-batch local update semantics
07ae3ed Clean up NC validation metric handling
7a1458c Add NC validation metrics and final test evaluation
a1ff43d add batch_size benchmark scripts
```

## Implemented Benchmark Changes

The current branch includes:

- Validation metrics every round.
- Final-only test metrics for normal runs.
- Corrected NC mini-batch logic: one model update per trainer local step.
- Fixed-round NC batch-size benchmark metrics.
- Elastic NC training with checkpoint capture and post-training checkpoint test evaluation.

Key files:

- `fedgraph/federated_methods.py`
- `benchmark/benchmark_NC_batch_size_convergence.py`
- `benchmark/plot_NC_batch_size_convergence.py`
- `tests/unit/test_federated_methods.py`

## Current Conclusions From CPU Runs

Primary experiment before GPU:

```text
benchmark/results/nc_batch_size_convergence/nc_fedgcn_2hop_noenc_elastic_800to1200r_seed42
```

This was FedGCN, 2-hop, no encryption, CPU, OGBN-Arxiv, 5 trainers, `local_step=3`, `rounds=800`, `max_rounds=1200`, `val_loss_patience=30`, `val_loss_min_delta=0.02`.

| Batch size | Stop round | Patience round | Final test acc | Final test loss | Final val acc | Final val loss | Best val acc round | Best val loss round | Train-loop wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 800 | 392 | 0.5756 | 1.7528 | 0.5724 | 1.7996 | 771 | 797 | 2779.76s |
| 512 | 800 | 550 | 0.5854 | 1.7040 | 0.5817 | 1.7482 | 778 | 799 | 3273.41s |
| 1024 | 800 | 426 | 0.5948 | 1.6723 | 0.5895 | 1.7141 | 798 | 798 | 3852.30s |
| 2048 | 800 | 414 | 0.5999 | 1.6483 | 0.5935 | 1.6900 | 796 | 786 | 4500.75s |
| Full batch | 800 | 427 | 0.6015 | 1.6080 | 0.5923 | 1.6542 | 790 | 790 | 6344.77s |

Interpretation:

- 800 rounds was much better than 500 rounds.
- Full batch had the best final test accuracy, but batch size 2048 was very close and substantially faster.
- The current convergence indicators are not reliable enough for true early stopping:
  - `val_loss_patience` fired early, but accuracy/loss kept improving.
  - `val_loss_plateau` was inconsistent and absent for some batch sizes.
- Larger mini-batches currently perform better under corrected mini-batch logic.
- GPU runs are needed to explore longer budgets and larger datasets.

Patience versus plateau in the 800-round run:

| Batch size | Patience round | Patience val acc | Patience val loss | Plateau round | Plateau val acc | Plateau val loss |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 392 | 0.5285 | 2.0475 | 439 | 0.5392 | 2.0073 |
| 512 | 550 | 0.5621 | 1.8838 | 492 | 0.5499 | 1.9358 |
| 1024 | 426 | 0.5342 | 1.9686 | none | none | none |
| 2048 | 414 | 0.5517 | 1.9349 | none | none | none |
| Full batch | 427 | 0.5596 | 1.8692 | 612 | 0.5792 | 1.7508 |

## Suggested First GPU Benchmark

First, run a small GPU smoke test before starting a long sweep.

Smoke test:

```bash
/home/ubuntu/miniconda3/envs/fedgraph312/bin/python benchmark/benchmark_NC_batch_size_convergence.py \
  --experiment-name nc_fedgcn_2hop_noenc_gpu_smoke_bs2048_20r_seed42 \
  --dataset ogbn-arxiv \
  --num-hops 2 \
  --batch-sizes 2048 \
  --rounds 20 \
  --local-step 3 \
  --n-trainer 5 \
  --iid-betas 10000 \
  --seeds 42 \
  --num-layers 2 \
  --gpu \
  --num-gpus-per-trainer 1 \
  --num-cpus-per-trainer 8 \
  --continue-on-error
```

For a `g5.48xlarge` with 8 GPUs, the intended mapping is 5 Ray trainers using 1 GPU each.

After the smoke test, consider a longer GPU run:

```bash
/home/ubuntu/miniconda3/envs/fedgraph312/bin/python benchmark/benchmark_NC_batch_size_convergence.py \
  --experiment-name nc_fedgcn_2hop_noenc_gpu_elastic_800to1600r_seed42 \
  --dataset ogbn-arxiv \
  --num-hops 2 \
  --batch-sizes 1024,2048,-1 \
  --rounds 800 \
  --elastic-training \
  --max-rounds 1600 \
  --val-loss-patience 50 \
  --val-loss-min-delta 0.01 \
  --local-step 3 \
  --n-trainer 5 \
  --iid-betas 10000 \
  --seeds 42 \
  --num-layers 2 \
  --convergence-window 20 \
  --val-loss-tolerance 0.01 \
  --val-acc-tolerance 0.0025 \
  --target-val-accs 0.58,0.59,0.60,0.61,0.62 \
  --target-val-losses 1.80,1.75,1.70,1.65,1.60 \
  --target-train-comm-times-sec 600,1200,1800,2400,3000,3600,4800,6000 \
  --gpu \
  --num-gpus-per-trainer 1 \
  --num-cpus-per-trainer 8 \
  --continue-on-error
```

Start with `1024,2048,-1` on GPU to reduce cost. Expand to `256,512,1024,2048,-1` after GPU performance and memory behavior are confirmed.

## New Codex Session Prompt

If Codex on the GPU instance has no chat history, start with:

```text
We are continuing FedGraph NC batch-size benchmark work on branch update_hc. Read docs/CODEX_GPU_HANDOFF.md first. The latest relevant commit is 2abebc4 Add elastic NC training checkpoints. Do not commit or push unless I ask. Do not include benchmark/results or dataset artifacts in commits. Current goal: validate GPU setup, run GPU FedGCN 2-hop no-encryption batch-size benchmarks, and compare against CPU runs.
```

## Verification Commands

```bash
/home/ubuntu/miniconda3/envs/fedgraph312/bin/python -m py_compile \
  fedgraph/federated_methods.py \
  benchmark/benchmark_NC_batch_size_convergence.py \
  benchmark/plot_NC_batch_size_convergence.py \
  tests/unit/test_federated_methods.py

/home/ubuntu/miniconda3/envs/fedgraph312/bin/python -m pytest \
  tests/unit/test_federated_methods.py::TestNCMetricHelpers -q

git diff --check
```

## Notes For Future Code Changes

- Keep workflow changes minimal and opt-in.
- Prefer benchmark-script changes over FedGraph workflow changes when possible.
- Consider making `val_loss_plateau` checkpointing optional or post-hoc only in a future cleanup.
- Training loss/accuracy may be useful diagnostics, but validation loss should remain the primary convergence metric.
- Client drift mitigation ideas for later discussion include FedProx, SCAFFOLD, server optimizers, learning-rate schedules, and possibly FedBN-style local normalization.
