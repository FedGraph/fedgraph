#!/usr/bin/env python3
"""Create a complete label-balanced 0-hop OGB node-classification artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from node_classification_partitioning import partition_raw_ogb_0hop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream raw OGB CSV or binary files into complete, local-coordinate FedGraph "
            "0-hop shards. The input is never sent through Ray."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="OGB dataset directory containing raw/ and split/<name>/.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New artifact directory. It is published only after validation.",
    )
    parser.add_argument("--n-trainer", type=int, required=True)
    parser.add_argument("--iid-beta", type=float, default=10000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-name", default="time")
    parser.add_argument(
        "--input-format",
        choices=("auto", "csv", "ogb-binary"),
        default="auto",
        help="Source layout. auto prefers CSV when present, otherwise OGB binary.",
    )
    parser.add_argument(
        "--binary-cache-dir",
        type=Path,
        default=None,
        help=(
            "Where to extract memory-mappable .npy arrays from raw/*.npz. "
            "Defaults to <dataset-root>/.fedgraph-binary-cache."
        ),
    )

    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=100_000,
        help="Raw feature/edge rows parsed per chunk; lower it to reduce RAM.",
    )
    parser.add_argument(
        "--no-checksums",
        action="store_true",
        help="Skip SHA-256 generation after validation to shorten a prototype run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse the deterministic owner plan in <output-dir>.incomplete.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = partition_raw_ogb_0hop(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        n_trainer=args.n_trainer,
        iid_beta=args.iid_beta,
        seed=args.seed,
        split_name=args.split_name,
        chunk_rows=args.chunk_rows,
        input_format=args.input_format,
        binary_cache_dir=args.binary_cache_dir,
        checksums=not args.no_checksums,
        resume=args.resume,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
