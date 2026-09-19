# Papers100M 0-Hop Experiment Archive

This directory contains the compact evidence retained from the manual Ray
Papers100M 0-hop work completed before beginning the 2-hop optimization phase.
The runs use seed 42, IID beta 10000.0, and the explicit test split for final
evaluation.

## Dataset artifact

- Dataset: `ogbn-papers100M`
- Balanced artifact: 195 trainers, 0-hop ownership shards
- Hugging Face repository:
  `FedGraph/fedgraph_ogbn-papers100M_195trainer_0hop_iid_beta_10000.0_v1`
- The copied artifact manifests and plot provenance files preserve the exact
  metadata and SHA-256 hashes used by the Figure 12 redraw.

## Validation runs

The following runs establish compatibility across the legacy and new artifact
paths before the final 195-trainer sweep:

- `nc_hf_papers100m_10trainer_0hop_bs512_100r_monitor`
- `nc_papers100m_10trainer_0hop_local_artifact_smoke_3`
- `nc_papers100m_10trainer_0hop_local_artifact_bs512_100r`
- `nc_papers100m_10trainer_0hop_hf_local_artifact_bs512_100r`
- `nc_papers100m_195trainer_0hop_local_artifact_smoke`
- `nc_papers100m_195trainer_0hop_local_artifact_bs512_100r`

## Balanced 800-round sweep

The retained Figure 12 inputs cover batch sizes 16, 32, 64, 128, 256, 512,
1024, 2048, and 4096. Batch size 16 uses the successful `_2` run:

- `nc_papers100m_195trainer_0hop_balanced_local_bs16_800r_2`
- `nc_papers100m_195trainer_0hop_balanced_local_bs32_800r`
- `nc_papers100m_195trainer_0hop_balanced_local_bs64_800r`
- `nc_papers100m_195trainer_0hop_balanced_local_bs128_800r`
- `nc_papers100m_195trainer_0hop_balanced_local_bs256_800r`
- `nc_papers100m_195trainer_0hop_balanced_local_bs512_800r`
- `nc_papers100m_195trainer_0hop_balanced_local_bs1024_800r`
- `nc_papers100m_195trainer_0hop_balanced_local_bs2048_800r`
- `nc_papers100m_195trainer_0hop_balanced_local_bs4096_800r`

The 195 trainer actors were distributed over ten GPU Ray workers. The
rank-to-host placement and per-worker GPU summaries are retained with the run
outputs.

## Retention policy

Each archived run keeps its configuration, stdout, manifest, summary files,
per-round metrics, validation curves, resource snapshot summary, GPU summary,
and compressed `nvidia-smi` samples when available.

The following generated files are intentionally excluded from Git:

- `resource_snapshots.jsonl`, because the nine final runs alone total about
  338 MB and the aggregate fields used by the plots are in
  `resource_snapshot_summary.json`.
- `*.pid`, because process IDs are host-local and have no archival value.
- Failed or superseded runs, including the Hugging Face rate-limit attempt and
  the first incomplete batch-size-16 local run.
- Downloaded OGB data and generated trainer shard tensors.

The manual execution procedure is documented in
`docs/manual_ray_papers100m_runbook.md`. Figure outputs and their provenance are
under `benchmark/figure/NC_comm_costs/figure12-balanced*`.
