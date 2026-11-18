# 2-Hop Optimization Status And Experiment Plan

Plan updated 2026-11-17 on `update_hc_experiments`. The measurement tables
describe the earlier campaign, before the separate sampler and adjacency
transition timing was added.

## Implementation Status

| Component | Implemented behavior | Evidence and remaining work |
|---|---|---|
| 1: Graph storage | `device`, `cpu`, and file-backed `mmap`; bounded adjacency relabel/cache; GPU transfer of sampled batches | Arxiv and Papers100M 0-hop storage comparisons completed. These are storage checks, not large-scale 2-hop baselines. |
| 2: Chunked pretraining | Source-sorted row requests; bounded edge batches inside each request; bounded trainer concurrency; CPU result payloads; file-backed trainer aggregation output | Matched Arxiv full/chunked comparison and Papers100M compact-v3 chunked runs completed. Matched Papers100M full/chunked comparisons remain. |
| 3: Shared feature projection | Uncentered PCA from additive, float64 `F x F` second moments; one shared basis; bounded CPU or mmap projection; compressed width retained through training | Arxiv smoke matrix, 3-seed Arxiv rank sweep, and Papers100M ranks 16/32/64/128 completed. Longer Papers100M quality comparisons remain. |
| Compact artifacts | V3 stores one local-position `int32` adjacency; degree/offset sidecars support global-row chunk requests without a full global adjacency | Papers100M v3 partitioning, upload, staging, and chunked/mmap training exercised. Runtime sampling still needs expanded `int64` adjacency and CSC indexes. |
| Full baseline preparation | Offline, bounded v3-to-v2 conversion writes separate local/global `int64` adjacency files; supports worker-local rank subsets | Synthetic conversion/equivalence tests passed. Full fleet conversion and execution are still to be measured. |
| Lifecycle and monitoring | Reused training/evaluation sampler; independent evaluation batch size; explicit full-pretraining finalization; sampler preparation timed before rounds; artifact/transition provenance | Sampler reuse and evaluation batching were exercised in Papers100M. Separate sampler-setup timing and the latest CPU adjacency transition need a new experiment. |

The feature-compression implementation is modular under
[`fedgraph/feature_compression`](../fedgraph/feature_compression/). It compresses
feature width, not model-parameter traffic. Retained feature energy measures
reconstruction information, not classification accuracy or privacy.

## Reading The Measurements

All memory values use GiB (`bytes / 2**30`). Trainer RSS and CUDA columns are
the largest per-trainer peaks in a run, not sums over trainers. Server RSS is
the Python server/driver peak, not whole-head memory including Ray's object
store. RSS includes resident mapped file pages; it is not anonymous RAM alone.
CUDA allocation is PyTorch allocation, not device-wide `nvidia-smi` memory.

For the 3-seed sweep, scalar entries are arithmetic means over seeds 42/43/44.
Memory entries are means of each run's peak as defined above. Accuracy is in
percent; loss is unitless. Failed runs are excluded from successful-run tables.

Pretraining traffic is the implementation's tensor-payload accounting, including
its row/index payloads. It is not measured TCP traffic or serialization overhead.
PCA setup traffic and time are reported separately from pretraining.

Historical train+sync times include sampler creation in round 1. The current
code emits `NC_NEIGHBOR_SAMPLER_METRIC` before the training timer starts. None
of the four Papers100M 10-round logs below contains that new metric; do not
compare their round-1 time directly with future runs.

Copied experiment directories can contain obsolete absolute paths in their
summaries. These tables resolve each run as `<summary parent>/runs/<run_id>`.

## Completed Experiment Tables

### Component 1 Storage Verification

Papers100M, legacy shards, 10 trainers, runtime 0-hop, batch size 512, one local
step, seed 42. Test evaluation was used for these historical compatibility runs.

| Storage | Rounds | Train+sync (s) | Trainer RSS | Server RSS | Trainer CUDA | Final test (%) |
|---|---:|---:|---:|---:|---:|---:|
| device | 3 | 3.793 | 23.490 | 0.689 | 6.251 | 0.945 |
| cpu | 3 | 4.069 | 22.963 | 0.696 | 0.091 | 0.945 |
| mmap, cold relabel cache | 3 | 4.265 | 14.549 | 0.696 | 0.091 | 0.945 |
| mmap, warm relabel cache | 3 | 4.083 | 9.199 | 0.696 | 0.091 | 0.945 |
| device | 20 | 20.162 | 23.066 | 0.697 | 6.252 | 6.191 |
| mmap, warm relabel cache | 20 | 27.201 | 15.007 | 0.696 | 0.091 | 6.191 |

Sources: `benchmark/results/nc_batch_size_convergence/c1_papers100m10_*`.
Six corresponding `c1_arxiv5_*` runs also completed. Warm means an existing
relabel cache was reused; it does not guarantee every operating-system page
cache was warm. The matching metrics support storage correctness and VRAM
reduction; these 0-hop graphs do not establish 2-hop resource requirements.

### Component 2 Matched Arxiv Comparison

Arxiv v2, 4 trainers, 2-hop, batch size 512, 3 rounds, one local step, seed 42,
no PCA. All four runs completed with final validation/test accuracy
1.720% / 2.031% and validation loss 4.3565.

| Storage / aggregation | Train+sync (s) | Trainer RSS | Server RSS | Trainer CUDA |
|---|---:|---:|---:|---:|
| device / full | 0.600 | 1.530 | 1.555 | 0.400 |
| cpu / full | 1.153 | 1.646 | 1.613 | 0.307 |
| cpu / chunked | 1.184 | 1.582 | 0.908 | 0.307 |
| mmap / chunked | 1.177 | 1.587 | 0.905 | 0.307 |

Source: [`c2_arxiv4_compare_20261011T022257Z`](../benchmark/results/nc_batch_size_convergence/c2_arxiv4_compare_20261011T022257Z/).
CPU chunking reduced server RSS by about 43.7%, with matching final metrics.
Small-graph trainer RSS is dominated by process/framework and sampling costs.
This comparison predates structured pretraining timing, so its train+sync
column must not be used to infer pretraining throughput.

### Component 3 Arxiv Smoke Matrix

Arxiv v2, 5 trainers sharing one GPU, batch size 16, 3 rounds, one local step,
seed 42. These are short correctness/feasibility runs.

| Storage / aggregation / projection | Pretrain (s) | Payload (GiB) | Trainer RSS | Server RSS | CUDA | Val / test (%) |
|---|---:|---:|---:|---:|---:|---:|
| device / full / none | 2.069 | 0.522 | 1.529 | 1.701 | 0.340 | 1.35 / 1.37 |
| cpu / full / none | 2.411 | 0.522 | 1.558 | 1.648 | 0.181 | 1.35 / 1.37 |
| cpu / full / PCA-128 | 2.376 | 0.522 | 1.585 | 1.657 | 0.181 | 1.25 / 1.21 |
| cpu / chunked / none | 3.946 | 0.526 | 1.528 | 1.108 | 0.181 | 1.35 / 1.37 |
| cpu / chunked / PCA-128 | 3.845 | 0.526 | 1.541 | 1.141 | 0.181 | 1.25 / 1.21 |
| cpu / chunked / PCA-64 | 1.501 | 0.267 | 1.486 | 1.031 | 0.034 | 1.17 / 1.23 |
| cpu / chunked / PCA-32 | 1.143 | 0.137 | 1.434 | 0.901 | 0.035 | 18.55 / 18.07 |
| cpu / chunked / PCA-16 | 0.821 | 0.073 | 1.437 | 0.828 | 0.028 | 2.94 / 3.08 |
| mmap / chunked / none | 3.854 | 0.526 | 1.526 | 1.127 | 0.181 | 1.35 / 1.37 |
| mmap / chunked / PCA-16 | 0.848 | 0.073 | 1.424 | 0.812 | 0.028 | 2.94 / 3.08 |

Sources: `benchmark/results/nc_batch_size_convergence/c3_arxiv_{full,chunked}_*`.
The unusually high early PCA-32 accuracy did not persist as a rank advantage
in the longer sweep below. Full-rank PCA rotates feature coordinates, and
unchanged random initialization need not follow the same optimization path.

### Component 3 Arxiv Rank Sweep

Arxiv v2, 5 trainers, CPU graph storage, chunked pretraining, batch size 512,
100 rounds, one local step. Every case completed for seeds 42/43/44.

| Projection | Energy (%) | Pretrain (s) | Payload (GiB) | Train+sync (s) | Val (%) | Test (%) | Val loss | Trainer / server RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | n/a | 3.710 | 0.526 | 20.977 | 45.263 | 43.231 | 2.500 | 1.686 / 1.120 |
| PCA-128 | 100.00 | 3.835 | 0.526 | 20.900 | 45.137 | 42.932 | 2.498 | 1.678 / 1.112 |
| PCA-64 | 95.98 | 1.578 | 0.267 | 19.075 | 42.560 | 40.572 | 2.566 | 1.552 / 1.034 |
| PCA-32 | 91.73 | 1.070 | 0.137 | 17.942 | 36.617 | 35.440 | 2.699 | 1.493 / 0.931 |
| PCA-16 | 87.91 | 0.887 | 0.073 | 17.367 | 35.310 | 33.102 | 2.750 | 1.470 / 0.830 |

Source: [`c3_arxiv_2hop_bs512_100r_rank_sweep_20261013T033300Z`](../benchmark/results/nc_batch_size_convergence/c3_arxiv_2hop_bs512_100r_rank_sweep_20261013T033300Z/).
PCA-64 reduced payload by about 49.2% and trainer/server RSS by about
7.9% / 7.6%, with a 2.66-percentage-point mean test-accuracy loss. PCA-16
reduced payload by about 86.2%, with a 10.13-point test-accuracy loss.
Retained energy therefore cannot be used alone to choose the rank.

### Papers100M 2-Hop Rank Sweep

Compact v3, 10 trainers on 10 Tesla T4 GPU workers, mmap graph storage, chunked
pretraining, batch/evaluation size 512, 10 rounds, one local step, seed 42.
All four runs used 1,048,576 pretraining rows, 1,000,000 edges per inner batch,
and 10 in-flight trainers. They completed successfully.

| Projection | Energy (%) | PCA setup (s) | Pretrain (min) | Payload (GiB) | Trainer RSS | Server RSS | Trainer CUDA |
|---|---:|---:|---:|---:|---:|---:|---:|
| PCA-16 | 82.07 | 15.97 | 13.01 | 75.52 | 40.26 | 1.60 | 2.10 |
| PCA-32 | 86.96 | 19.84 | 16.63 | 142.65 | 40.44 | 2.31 | 2.14 |
| PCA-64 | 93.00 | 26.79 | 24.90 | 276.91 | 46.18 | 3.65 | 2.09 |
| PCA-128 | 100.00 | 40.25 | 40.87 | 545.43 | 54.01 | 6.74 | 1.90 |

| Projection | Historical train+sync total (s) | Historical round 1 (s) | Mean rounds 2-10 train+sync (s) | Val / test (%) | Val / test loss |
|---|---:|---:|---:|---:|---:|
| PCA-16 | 170.80 | 169.33 | 0.163 | 6.59 / 6.07 | 5.073 / 5.057 |
| PCA-32 | 118.88 | 116.42 | 0.273 | 3.26 / 2.69 | 5.124 / 5.120 |
| PCA-64 | 187.53 | 183.08 | 0.495 | 3.29 / 2.85 | 5.062 / 5.082 |
| PCA-128 | 187.56 | 145.16 | 4.711 | 1.23 / 1.33 | 5.065 / 5.094 |

Sources: [`c3/c3_papers100m_10t_chunked_mmap_pca{16,32,64,128}_512bz_10r`](../benchmark/results/nc_batch_size_convergence/c3/).
Each source has `summary.jsonl` plus per-run `stdout.log` and resource summaries.
Round 1 includes sampling-index construction. Rounds 2-10 exclude that first
setup, but are still short, order/cache-sensitive observations. The rank-128
steady-time increase deserves reproduction with the new timing boundary.

The high mapped RSS is not evidence of imminent host OOM: minimum sampled
worker MemAvailable was approximately 41.3-42.6 GiB across these four cases.
This does not establish that CPU mode will fit: ordinary CPU allocations are
less reclaimable than mapped file pages, and the sampler needs separate RAM.
Higher rank reduces neither adjacency size nor hidden-layer width.

Earlier Papers100M PCA-16 runs included an evaluation OOM, followed by a
successful batch-16/evaluation-512 smoke and a successful 3-round batch-16 run.
The failure exposed duplicate sampling-index memory; sampler reuse and
independent evaluation batching addressed it. These are not independent
rank-sweep replicates. Duplicate copied smoke directories are counted once.

## Supported Mode Matrix

This table describes plaintext, 2-hop NC with GPU mini-batch training. `device`
means graph tensors follow the trainer device; it is GPU storage in these runs.
All manifest-style 2-hop experiments use indexed upload. Both v2 and v3 work
with chunking; v3 is the standard compact artifact for new large runs.

| Graph storage | Pretraining | Feature projection | Artifact | Status / evidence |
|---|---|---|---|---|
| device | full | none | v2 | Enabled; Arxiv measured. Papers100M 2-hop not measured. |
| cpu | full | none | v2 | Enabled; Arxiv measured. Large server memory remains a constraint. |
| cpu | full | shared-pca | v2 | Enabled; Arxiv PCA-128 measured. Papers100M baseline requires offline conversion and sizing. |
| cpu | chunked | none | v2/v3 | Enabled; Arxiv measured. Papers100M CPU run pending. |
| cpu | chunked | shared-pca | v2/v3 | Enabled; Arxiv measured. Papers100M CPU run pending. |
| mmap | chunked | none | v2/v3 | Enabled; Arxiv measured. Papers100M no-projection control pending. |
| mmap | chunked | shared-pca | v2/v3 | Enabled; Arxiv and Papers100M measured. |
| mmap | full | either | either | Intentionally disabled in the current comparison workflow; full results would remove the bounded-output benefit. |
| device | chunked | either | either | Disabled; a host-storage/GPU-chunk-compute extension is a separate future option. |
| device | full | shared-pca | v2 | Disabled; GPU statistics/projection and graph lifecycle need separate support. |
| cpu/device/mmap | full | either | compact v3 | Disabled; explicitly convert to a separate v2 artifact first. |
| cpu/mmap | chunked or shared-pca | either | either | New optimization paths are plaintext only; encryption integration is deferred. |

In CPU **and** mmap chunked modes, complete returned aggregation results are
written to worker-local files and mapped for training. CPU mode does not imply
that every runtime tensor becomes an ordinary RAM allocation. GPU training
copies sampled batches from this graph representation.

Future candidates remain GPU computation of bounded chunks while raw data
stays on the host, bounded neighbor fanout, sampling-index memory reduction,
and optional overlap of compute/transfer. Disk-backed tensors still require
RAM for touched pages and sampling indexes. None of those extensions is
needed to describe the implementations already completed.

## Experiment Plan

The fresh campaign starts at `01-artifact-smoke`. Its case directories use
`benchmark/results/nc_2hop_optimization/<dataset>/<campaign>/<suite>/<case>/`.
The earlier tables are the historical record and can guide resource estimates.
Each table row below defines the question and settings for a new suite.

| Dataset / suite | Cases and budget | Question and gate |
|---|---|---|
| Arxiv / `01-artifact-smoke` | V2 full on device and CPU, v2 chunked CPU, v3 chunked CPU and mmap, all unprojected. Add v2 full CPU and v3 chunked mmap at PCA-16. Batch 16, 3 rounds, seed 42. | Does the current version complete pretraining, sampler setup, training, and evaluation? Compare final metrics across the unprojected cases and check artifact provenance, feature width, and sampler count. |
| Arxiv / `02-rank-quality` | V3 chunked CPU, no PCA and ranks 128/64/32/16. Batch 512, 100 rounds, seeds 42/43/44. | Reproduce the rank and accuracy tradeoff. PCA-128 controls for the rotation at full width. |
| Papers100M / `01-fleet-smoke` | V3 chunked mmap PCA-16, batch 16, 1 round, seed 42. | Reproduce a known feasible path on the current code. Check final evaluation, sampler timing, and head/worker memory before expanding the suite. |
| Papers100M / `02-storage` | V3 chunked mmap and CPU at PCA-16. Batch 16, 3 rounds, seed 42. Both cases use the smoke chunk limits. | Isolate graph storage. Check worker RAM and runtime disk before CPU mode. Compare RSS, host MemAvailable, CUDA allocation, sampler setup, and time. |
| Papers100M / `03-aggregation` | Derived v2 on CPU, full vs chunked, common PCA rank after sizing. Batch 16, 3 rounds, seed 42. Optionally add matching v3 chunked CPU. | Compare chunking on the same v2 artifact and fleet. The optional v3 case measures the additional effect of the compact representation. |
| Papers100M / `04-projection` | V3 chunked mmap, no PCA and ranks 128/64/32/16. Batch 512, 10 rounds, seed 42. Run lower ranks first. | Measure PCA setup time, pretraining payload, pretraining time and memory with matched chunk settings. Use the no-PCA case as the full-width baseline. |
| Papers100M / `05-chunk-tuning` | One fitting rank on v3 with chunked pretraining. Batch 512, 3 rounds, seed 42. Start from 262,144 rows, 1,000,000 edges, and 10 in-flight trainers. Vary one of these per pair. | Trade pretraining throughput against peak host/Ray memory. Repeat promising settings on the same fleet. |
| Papers100M / `06-rank-quality` | Carry forward the feasible unprojected or full-width baseline and two useful PCA ranks. Batch 512, 100 rounds initially, seeds 42/43/44. | Use validation accuracy/loss to select ranks. Report final test accuracy and the systems metrics for the selected comparisons. |

Begin with Arxiv and the Papers100M fleet smoke. Record an unavailable full
baseline as pending if the available fleet cannot accommodate it. The mmap
projection suite can proceed independently once the fleet smoke passes.
GPU chunk compute, fanout changes, and 195-trainer 2-hop scaling are follow-up
work driven by the measured bottlenecks.

Keep plaintext FedGCN, two GNN layers, indexed upload, `norm_type=none`, one
local step, evaluation batch 512, validation each round, and final test fixed.
For matched cases, use the same code revision, graph partition, rank-host map,
seed, CPU thread settings, worker hardware, and round budget. For a v2/v3
comparison, save both manifest SHA-256 values and the conversion provenance.
Record RAM/VRAM, Ray object-store capacity, and node/driver versions once per
campaign. Create a new case label for failures and repetitions. Record the
adjacency relabel-cache condition and run order separately from filesystem
cache state. Repeat timings before making speed claims.

Collect initialization, PCA statistics/projection, pretraining aggregation,
adjacency transition, sampler construction, steady round training/sync,
validation, and final test separately. `total_train_comm_time_sec` excludes
evaluation; it is not end-to-end execution time. The updated
[`manual_ray_2hop_optimization_runbook.md`](manual_ray_2hop_optimization_runbook.md)
contains fleet and submission commands.

## Code Checkpoints

The cleanup groups related changes into three commits:

| Commit | Scope |
|---|---|
| `Add modular shared PCA feature projection` | `fedgraph/feature_compression/` and `tests/unit/test_feature_compression.py` |
| `Integrate shared PCA and compact NC artifact execution` | Trainer/server integration, metrics, compact partition/load/upload/staging, converter, sampler/evaluation lifecycle, two-hop launcher, and related regression tests |
| `Document 2-hop experiment workflow and measured status` | The two-hop runbook, historical metrics, and new campaign plan |

Raw datasets and benchmark result trees remain in their existing locations;
this report records their provenance without adding large archives.

The focused two-hop verification passed 72 tests in `fedgraph312`, including
compact conversion, chunked/full feature sums, shared PCA, and sampler reuse.
The distributed launcher's dry run wrote an isolated case plan with seed 43.

The earlier six-module unit sweep produced 161 passes and 12 failures in
trainer fixtures, enhanced routing, GC/LP setup, and argument tests. The
compression, graph storage, partitioner/converter, and benchmark-summary
suites passed. All 12 failures reproduced with the same errors on the
prior commit `0088e0a` in a separate checkout.
