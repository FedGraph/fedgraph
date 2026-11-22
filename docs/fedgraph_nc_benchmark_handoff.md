# FedGraph NC Benchmark Handoff

This note preserves the technical context from the earlier CPU and GPU
handoffs for the `update_hc_experiments` branch. It is not a record of the
local machine or a set of credentials.

## Benchmark Semantics

- One NC mini-batch local step means one optimizer update per trainer, not
  one pass over the entire local training set. Keep local steps and training
  batch size fixed when interpreting differences in round budgets.
- Use validation to compare runs. Final test metrics are for the last model
  unless a separate checkpoint test is explicitly labeled. Legacy shards
  without validation use an explicit evaluation split; their per-round test
  metrics must not be labeled validation.
- An earlier CPU Arxiv sweep found continued improvement after the validation
  patience heuristic fired. Keep fixed budgets for matched accuracy and
  systems comparisons unless convergence behavior is the question.
- Train plus synchronization time excludes validation, pretraining, and
  separately measured sampler preparation. Logical tensor payload is not
  wire traffic, and trainer RSS on mmap runs includes resident file pages.

## Reproduction Sources

| Work | Entry point |
|---|---|
| Earlier CPU/GPU sweeps, artifact compatibility, and optimization checks | [Earlier NC archive](../benchmark/archives/nc_batch_size_convergence/ARCHIVE.md) |
| Papers100M 0-hop and balanced Figure 12 reproduction | [0-hop archive](../benchmark/results/nc_batch_size_convergence/PAPERS100M_0HOP_ARCHIVE.md) and [0-hop runbook](manual_ray_papers100m_runbook.md) |
| Papers100M 2-hop, shared PCA, and memory comparisons | [2-hop archive](../benchmark/archives/nc_2hop_optimization/ARCHIVE.md), [status](2hop_optimization_status.md), and [2-hop runbook](manual_ray_2hop_optimization_runbook.md) |

Use a clean, recorded code revision and a pinned artifact revision for new
comparisons. The runbooks describe the `fedgraph312` environment and fleet
setup without requiring a copy of a working environment. Keep worker-local
runtime output separate from raw input shards and benchmark logs.

## Next Accuracy Work

1. Preserve the finalized per-trainer PCA-64 and full-width feature caches on
   access-controlled storage independent of the compute account. Keep their
   metadata, source artifact fingerprints, and the PCA basis. The Git
   archives contain evidence and metadata, not those feature tensors.
2. Fix the benchmark-to-NC seed forwarding and verify the effective runtime
   seed. The archived Arxiv cases labeled 43 and 44 are repeats, not evidence
   of independent seeds. Papers100M rank quality is a seed-42 comparison.
3. Validate opt-in pretraining reuse on Arxiv before trying a Papers100M
   cache-only smoke. It must check each trainer's communication-row order,
   width, artifact identity, and normalization before skipping feature
   exchange. Training still requires the correct adjacency and sampler.
