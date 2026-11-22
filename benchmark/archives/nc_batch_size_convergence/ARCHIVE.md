# Earlier NC Benchmark Archive

This directory preserves previously untracked evidence from
`benchmark/results/nc_batch_size_convergence/`. The result hierarchy is kept so
old run labels remain searchable and the existing plotting tools can read
individual case directories. It covers the early CPU and Cora GPU sweeps,
legacy Arxiv and Papers100M artifact checks, 0-hop diagnostics, and the `c1`,
`c2`, and `c3` results before the organized 2-hop campaign.

The evidence includes configurations, manifests, summaries, per-round tables,
plots, resource and GPU summaries, and reviewed diagnostic logs, including
failed attempts. Copied machine identifiers are replaced with aliases. Raw GPU
and resource traces, process IDs, local session state, model checkpoints,
datasets, and tensor payloads are not included.

These runs span multiple code revisions and logging schemas. Some older
compatibility runs report per-round test, not validation. Use the configuration
and summary in each case to interpret the evaluation split and timing fields.
Do not combine these runs into the newer campaign's matched comparisons.

Cases already tracked in the [Papers100M 0-hop archive](../../results/nc_batch_size_convergence/PAPERS100M_0HOP_ARCHIVE.md)
are not duplicated here. The newer runs are indexed in the
[2-hop archive](../nc_2hop_optimization/ARCHIVE.md), and the earlier experiment
semantics are summarized in the [benchmark handoff](../../../docs/fedgraph_nc_benchmark_handoff.md).
