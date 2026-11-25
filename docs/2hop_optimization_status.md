# 2-Hop Optimization Status And Experiment Plan

Status updated 2026-11-22 on `update_hc_experiments`. The historical tables
predate separate sampler and adjacency-transition timing. For the later
campaign, see the [benchmark archive](../benchmark/archives/nc_2hop_optimization/ARCHIVE.md).

## Implementation Status

| Component | Implemented behavior | Evidence and remaining work |
|---|---|---|
| 1: Graph storage | `device`, `cpu`, and file-backed `mmap`; bounded adjacency relabel/cache; GPU transfer of sampled batches | Arxiv and Papers100M 0-hop storage comparisons completed. These are storage checks, not large-scale 2-hop baselines. |
| 2: Chunked pretraining | Source-sorted row requests; bounded edge batches inside each request; bounded trainer concurrency; CPU result payloads; file-backed trainer aggregation output | Matched full/chunked comparisons completed on Arxiv and on Papers100M at PCA-8. Full CPU PCA-16 exceeded worker RAM on the tested fleet. |
| 3: Shared feature projection | Uncentered PCA from additive, float64 `F x F` second moments; one shared basis; bounded CPU or mmap projection; compressed width retained through training | Arxiv rank sweep and Papers100M projection sweep completed. Papers100M no PCA, PCA-16, and PCA-64 also completed 800 rounds at seed 42. Independent seed comparisons remain. |
| Compact artifacts | V3 stores one local-position `int32` adjacency; degree/offset sidecars support global-row chunk requests without a full global adjacency | Papers100M v3 partitioning, upload, staging, and chunked/mmap training exercised. Runtime sampling still needs expanded `int64` adjacency and CSC indexes. |
| Full baseline preparation | Offline, bounded v3-to-v2 conversion writes separate local/global `int64` adjacency files; supports worker-local rank subsets | The derived v2 artifact supported the matched Papers100M rank-8 comparison. Wider full-mode baselines require more headroom. |
| Lifecycle and monitoring | Reused training/evaluation sampler; independent evaluation batch size; explicit full-pretraining finalization; sampler preparation timed before rounds; artifact/transition provenance | The current campaign records separate sampler timing and exercises both the CPU and mmap adjacency transition. |

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

For rank-sweep tables, scalar entries are arithmetic means over runs labeled
42/43/44. In the later Arxiv campaign, the runner recorded the requested seed
but did not forward it to NC, so these labels do not establish independent
seeds. Memory entries are means of each run's peak as defined above. Accuracy
is in percent; loss is unitless. Failed runs are excluded from successful-run
tables.

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
| cpu | full | shared-pca | v2 | Arxiv measured. Papers100M PCA-8 completed on the derived v2 artifact; PCA-16 hit worker RAM limits. |
| cpu | chunked | none | v2/v3 | Enabled; Arxiv measured. Papers100M unprojected CPU run not measured. |
| cpu | chunked | shared-pca | v2/v3 | Enabled; Arxiv and Papers100M measured. |
| mmap | chunked | none | v2/v3 | Enabled; Arxiv and Papers100M measured, including the 800-round Papers100M control. |
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

The archived campaign starts at `01-artifact-smoke`. Its original directories use
`benchmark/results/nc_2hop_optimization/<dataset>/<campaign>/<suite>/<case>/`.
The archive preserves the same case hierarchy. Each table row below defines
the question and settings for a suite.

| Dataset / suite | Cases and budget | Question and gate |
|---|---|---|
| Arxiv / `01-artifact-smoke` | V2 full on device and CPU, v2 chunked CPU, v3 chunked CPU and mmap, all unprojected. Add v2 full CPU and v3 chunked mmap at PCA-16. Batch 16, 3 rounds, seed 42. | Does the current version complete pretraining, sampler setup, training, and evaluation? Compare final metrics across the unprojected cases and check artifact provenance, feature width, and sampler count. |
| Arxiv / `02-rank-quality` | V3 chunked CPU, no PCA and ranks 128/64/32/16. Batch 512, 100 rounds, requested seeds 42/43/44. | Compare rank and accuracy. PCA-128 controls for rotation at full width. The runner used the seed-42 fallback, so the runs are repeats, not an independent-seed sweep. |
| Papers100M / `01-fleet-smoke` | V3 chunked mmap PCA-16, batch 16, 1 round, seed 42. | Reproduce a known feasible path on the current code. Check final evaluation, sampler timing, and head/worker memory before expanding the suite. |
| Papers100M / `02-storage` | V3 chunked mmap and CPU at PCA-16. Batch 16, 3 rounds, seed 42. Both cases use the smoke chunk limits. | Isolate graph storage. Check worker RAM and runtime disk before CPU mode. Compare RSS, host MemAvailable, CUDA allocation, sampler setup, and time. |
| Papers100M / `03-aggregation` | Derived v2 on CPU, full vs chunked, PCA-8 on the tested fleet. Batch 16, 3 rounds, seed 42. Optionally add matching v3 chunked CPU. | Compare chunking on the same v2 artifact and fleet. The optional v3 case measures the additional effect of the compact representation. |
| Papers100M / `04-projection` | V3 chunked mmap, no PCA and ranks 128/64/32/16. Batch 512, 10 rounds, seed 42. Run lower ranks first. | Measure PCA setup time, pretraining payload, pretraining time and memory with matched chunk settings. Use the no-PCA case as the full-width baseline. |
| Papers100M / `05-chunk-tuning` | V3 chunked mmap PCA-16, batch/evaluation size 512, 3 rounds, seed 42. Six single-knob cases plus an optional larger row chunk. See the [case matrix](#papers100m-chunk-tuning). | Compare pretraining throughput and peak host/Ray memory. Repeat the baseline and the leading case before selecting a setting. |
| Papers100M / `06-rank-quality` | V3 chunked mmap, no PCA, PCA-16, and PCA-64. Batch/evaluation size 512, 800 fixed rounds. See the [case matrix](#papers100m-rank-quality). | The seed-42 triplet completed. Rerun the controls with the updated seed-aware runner for independent-seed comparisons. |
| Papers100M / `07-cache-smoke` | Reuse the promoted PCA-16, PCA-64, and unprojected aggregates for one round each. | All three used ten rank-local caches, zero feature traffic, and matched the fresh runs' logged round-one validation metrics. |
| Papers100M / `08-model-architecture` | Same PCA-64 cache, two layers, and one local step. Cross hidden widths 256/512 with LayerNorm off/on. Screen at 200 rounds, then run the baseline and strongest cases for 800. | Separate width and normalization effects before tuning the training recipe. See the [architecture commands](manual_ray_2hop_optimization_runbook.md#compare-papers100m-model-architectures). |
| Papers100M / `09-training-recipe` | Start with normalized-wider and AdamW at LR 0.001 for 2 then 800 rounds, using the same PCA-64 cache, layer count, and one local step. | Verify AdamW training after weight sync, then compare with SGD at 0.01 before tuning the learning rate, scheduler, or local steps. See the [optimizer commands](manual_ray_2hop_optimization_runbook.md#tune-the-papers100m-optimizer). |
| Papers100M / `10-hyperparameter-sweep` | Run an AdamW/PCA-64 control, screen fixed rates and cosine, then compare model settings, update budgets, and rank using matching caches. | Choose by validation curves and verify selected recipes across seeds. See the [sweep commands](manual_ray_2hop_optimization_runbook.md#run-the-hyperparameter-sweep). |
| Papers100M / `11-model-heads` | Same unprojected cache, LayerNorm, width 1,024, AdamW/cosine, three local steps, batch 768, evaluation batch 256. Compare `gcn` and `gcn-skip` at 800 rounds. | Isolate the learned self branch. Run a short smoke on the chosen fleet before the matched pair. See the [head comparison](manual_ray_2hop_optimization_runbook.md#compare-papers100m-model-heads). |

Use the [runbook index](manual_ray_2hop_optimization_runbook.md#run-any-series)
to run any series independently. The smoke cases are optional checks for new
environments. Record an unavailable full baseline as pending if the fleet
cannot accommodate it.
GPU chunk compute, fanout changes, and 195-trainer 2-hop scaling are follow-up
work driven by the measured bottlenecks.

Keep plaintext FedGCN, two GNN layers, indexed upload, `norm_type=none`,
validation each round, and final test fixed. Use the local-step count and
evaluation batch specified for the chosen series.
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

### Papers100M Chunk Tuning

Use the same ten v3 shards, rank-host map, CPU head, and one T4 per trainer as
`04-projection`. Fix graph storage to `mmap`, pretraining to `chunked`, and
projection to PCA-16. This path already completed on the fleet, gives room to
vary the chunk limits, and avoids changing feature width within the sweep.
Keep training and evaluation batch size 512, three global rounds, one local
step, seed 42, and the same Ray object-store and spill configuration. Start the
suite with a fresh baseline at this round budget rather than reusing the
ten-round result from `04-projection`.

| Order | Case suffix | Rows per server chunk | Edges per trainer sub-batch | In-flight trainer cap | Expected server chunks | What it tests |
|---:|---|---:|---:|---:|---:|---|
| 1 | `baseline` | 262,144 | 1,000,000 | 10 | 424 | Reference for every variant. |
| 2 | `rows-131072` | 131,072 | 1,000,000 | 10 | 848 | Smaller server buffers versus more RPCs. |
| 3 | `rows-524288` | 524,288 | 1,000,000 | 10 | 212 | Fewer RPCs versus larger buffers. |
| 4 | `edges-500000` | 262,144 | 500,000 | 10 | 424 | Smaller worker temporaries versus more inner batches. |
| 5 | `edges-2000000` | 262,144 | 2,000,000 | 10 | 424 | Fewer inner batches versus larger worker temporaries. |
| 6 | `inflight-4` | 262,144 | 1,000,000 | 4 | 424 | Lower concurrent object pressure versus less trainer overlap. |
| Optional | `rows-1048576` | 1,048,576 | 1,000,000 | 10 | 106 | Try only if `rows-524288` is faster with comfortable RAM and spill headroom. |

Row limits group global node IDs, so they change the number of server chunks
and the size of each result. Edge limits bound scans of the current row range
on each trainer. They do not split the trainer's reply for a row or change the
server chunk count. The in-flight cap bounds concurrently outstanding trainer
requests and also affects initialization and finalization RPCs. It does not
change the number of trainers or parallelize the sequential chunk downlinks.
Keep the PCA chunk limits and adjacency expansion edge limit unchanged.

Compare each variant to `baseline`, not to the preceding row of the table.
Require three completed rounds, ten prepared samplers, final test evaluation,
the same PCA basis hash, and the expected server chunk count. The unchanged
graph and feature width should produce the same logical tensor payload
(about 75.52 GiB total) and comparable validation/test metrics. That payload
is summed across the run, not the live Ray object-store footprint.

Rank candidates first by pretraining wall time and check head and worst-worker
peak process RSS, sampled host `MemAvailable`, Ray object-store usage, spill
activity, and free NVMe. Report adjacency transition separately: it is included
in pretraining wall time and may change when the in-flight cap changes. A
pretraining-time difference by itself does not establish a faster chunk loop.
Repeat the baseline and best feasible case with fresh case names and the same
adjacency-cache policy before claiming a speedup. Report the run order because
filesystem cache state can still differ. Only try a combination of winning
settings as a separate case after the individual comparisons.

Use the [launch commands](manual_ray_2hop_optimization_runbook.md#run-the-chunk-tuning-sweep)
to run and archive the cases.

### Papers100M Rank Quality

Run 800 fixed rounds with one local step per round and no elastic stopping.
The first 100 rounds are an early checkpoint on the same run, so a separate
100-round sweep is not needed. This suite measures quality at a common budget;
it does not assume that all ranks converge by round 800.

Reuse the ten v3 shards, rank-host map, CPU head, and one T4 per trainer.
Use `mmap`, indexed chunked pretraining, and the `05-chunk-tuning` baseline
limits: 262,144 rows, 1,000,000 edges, and at most 10 trainers in flight.
Keep training/evaluation batch size 512 and all other settings fixed,
including PCA statistics blocks of 65,536 rows and 4 in-flight trainers.
Use the same case-local adjacency cache policy and Ray object-store/spill
configuration. Keep validation every round, final test after round 800, and
manual resource snapshots on the first round and every 25 rounds.

| Order per seed | Case label (`<seed>` = 42, 43, or 44) | Feature width | Role | `04-projection` pretrain payload |
|---:|---|---:|---|---:|
| 1 | `v3-chunked-mmap-pca16-seed<seed>` | 16 | Largest measured savings; 82.07% retained feature energy. | 75.52 GiB |
| 2 | `v3-chunked-mmap-pca64-seed<seed>` | 64 | Higher-retention comparison; 93.00% retained feature energy. | 276.91 GiB |
| 3 | `v3-chunked-mmap-none-seed<seed>` | 128 | Unprojected reference, not the rotated PCA-128 control. | 545.43 GiB |

The payloads are measured logical upload plus download bytes, not peak RAM.
The ten-round results establish feasibility, not a quality ranking. Keep
PCA-32 as an optional follow-up if the results leave the tradeoff between 16
and 64 unclear. All three seed-42 cases completed 800 rounds, ten samplers,
and final test evaluation. Their final test accuracy was 28.85% for PCA-16,
39.04% for PCA-64, and 38.52% without projection. The unprojected case had the
highest best-validation accuracy (46.45%). Do not infer a consistent PCA-64
quality gain from one seed and one final checkpoint.

The archived runner recorded `--seeds` without forwarding the seed to NC.
The updated runner forwards it. Rerun matched seed-42 cases under the same
updated runner if comparing them with new seeds. Preserve the finalized
worker feature files before cleaning the rank-quality runtime directories.

Report validation accuracy and loss at rounds 100, 200, 400, and 800 from
`round_metrics.csv`, plus the full curves and best-validation rounds.
Compare each rank to `none` at the same seed and summarize the mean and sample
standard deviation across seeds. Use validation to judge the accuracy/loss
tradeoff, then report round-800 test accuracy and loss without using test to
choose the rank. Report PCA setup time, pretraining time and payload, steady
train+sync time per round, end-to-end time, and head/worker RSS, CUDA, and disk
separately. Sampled memory telemetry is not an exact peak over all rounds.

Use the [launch commands](manual_ray_2hop_optimization_runbook.md#run-the-papers100m-rank-quality-sweep)
to run and archive the cases.

## Code Checkpoints

The cleanup groups related changes into three commits:

| Commit | Scope |
|---|---|
| `Add modular shared PCA feature projection` | `fedgraph/feature_compression/` and `tests/unit/test_feature_compression.py` |
| `Integrate shared PCA and compact NC artifact execution` | Trainer/server integration, metrics, compact partition/load/upload/staging, converter, sampler/evaluation lifecycle, two-hop launcher, and related regression tests |
| `Document 2-hop experiment workflow and measured status` | The two-hop runbook, historical metrics, and new campaign plan |

The [archives](../benchmark/archives/nc_2hop_optimization/ARCHIVE.md) retain
reviewed experiment evidence. Raw datasets, telemetry, and worker feature
caches are not part of the Git archive.

The focused two-hop verification passed 72 tests in `fedgraph312`, including
compact conversion, chunked/full feature sums, shared PCA, and sampler reuse.
The distributed launcher's dry run wrote an isolated case plan with seed 43.

The earlier six-module unit sweep produced 161 passes and 12 failures in
trainer fixtures, enhanced routing, GC/LP setup, and argument tests. The
compression, graph storage, partitioner/converter, and benchmark-summary
suites passed. All 12 failures reproduced with the same errors on the
prior commit `0088e0a` in a separate checkout.
