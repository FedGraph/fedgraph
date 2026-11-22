# NC 2-Hop Benchmark Archive

This archive preserves the experiment evidence from campaign
`20261118T191651Z`. Case paths mirror the original layout under
`benchmark/results/nc_2hop_optimization/`. Use a case directory directly with
the NC benchmark plotting tools. Raw run directories were not modified.

| Dataset | Suite | Completed / attempts | Main result |
|---|---|---:|---|
| Arxiv | `01-artifact-smoke` | 7 / 7 | Unprojected execution paths matched final quality; the two PCA-16 paths matched each other. |
| Arxiv | `02-rank-quality` | 15 / 15 | Rank 64 reduced traffic with a smaller quality cost than ranks 32 or 16 at 100 rounds. |
| Papers100M | `01-fleet-smoke` | 1 / 1 | Ten workers completed the full pipeline; sampler setup was timed separately. |
| Papers100M | `02-storage` | 2 / 2 | Mmap lowered peak worker RSS from 47.67 to 40.37 GiB and increased available RAM. |
| Papers100M | `03-aggregation` | 3 / 6 | Matched v2 CPU PCA-8 chunking lowered head process RSS from 54.03 to 0.91 GiB. |
| Papers100M | `04-projection` | 5 / 6 | The unprojected retry completed after worker disk cleanup. PCA-16 cut logical pretraining payload from 545.43 to 75.52 GiB. |
| Papers100M | `05-chunk-tuning` | 6 / 6 | The 262,144-row, 1,000,000-edge, ten-in-flight baseline remained the default. |
| Papers100M | `06-rank-quality` | 3 / 3 | At 800 rounds, final test accuracy was 28.85% (PCA-16), 39.04% (PCA-64), and 38.52% (no PCA). |

## Reproducibility

- The values labeled `seed` are the requested settings. The runner omitted the
  seed from the NC runtime arguments, so the Arxiv labels 43 and 44 are repeats
  using the fallback of 42. Papers100M `06` is a matched seed-42 comparison,
  not a multi-seed result.
- Arxiv manifests record `30e2935` with a clean tree. Papers100M manifests
  record `198a492` with local modifications. Preserve the actual fleet source
  snapshot outside this archive; that commit hash alone cannot reconstruct it.
- The Arxiv dataset directory keeps its v2/v3 manifest fingerprints. The
  runbook describes the matching compact Papers100M artifact, pinned revision,
  and derived v2 conversion. Preserve the source dataset manifest and revision
  alongside any exported feature cache.
- Payloads are logical tensor bytes, not measured network bytes. Head RSS does
  not include other Ray processes, and trainer mmap RSS includes mapped pages.

## Retention

The archive includes benchmark configs, manifests, summaries, curves,
diagnostic logs, resource and GPU summaries, and PCA metadata where available.
Machine identifiers are replaced with aliases in these copies. Paths emitted
by the original jobs remain diagnostic provenance, not portable paths. To find
a per-run file, use `runs/<run_id>/` inside the case directory.

Raw telemetry, worker session files, dataset and shard tensors, PCA basis
tensors, and finalized pretraining feature files are excluded. Cache reuse
requires the separate per-rank data and metadata, not this evidence archive.

See the [status notes](../../../docs/2hop_optimization_status.md) and
[2-hop runbook](../../../docs/manual_ray_2hop_optimization_runbook.md).
