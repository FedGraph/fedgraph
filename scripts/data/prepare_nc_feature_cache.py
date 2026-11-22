#!/usr/bin/env python3
"""Promote or verify a rank's finalized chunked FedGCN feature cache."""

import argparse
from pathlib import Path

from fedgraph.feature_aggregation_cache import (
    load_feature_cache_manifest,
    open_feature_cache_rank,
    promote_feature_cache_rank,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    promote = commands.add_parser("promote", help="Import a rank from a completed run")
    verify = commands.add_parser(
        "verify", help="Verify a staged rank including its payload hash"
    )
    for command in (promote, verify):
        command.add_argument("--artifact-dir", type=Path, required=True)
        command.add_argument("--cache-dir", type=Path, required=True)
        command.add_argument("--trainer-id", type=int, required=True)
    promote.add_argument(
        "--pretrain-dir",
        type=Path,
        required=True,
        help="Source run directory under pretrain/, containing trainer-NNN",
    )
    promote.add_argument("--norm-type", choices=("none", "row", "sym"), required=True)
    promote.add_argument(
        "--copy", action="store_true", help="Copy instead of hard-linking the payload"
    )
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.expanduser()
    cache_dir = args.cache_dir.expanduser()
    try:
        if args.command == "promote":
            manifest = promote_feature_cache_rank(
                artifact_dir=artifact_dir,
                cache_dir=cache_dir,
                trainer_id=args.trainer_id,
                pretrain_dir=args.pretrain_dir.expanduser(),
                norm_type=args.norm_type,
                copy=args.copy,
            )
        else:
            manifest = load_feature_cache_manifest(cache_dir)
            open_feature_cache_rank(
                artifact_dir=artifact_dir,
                cache_dir=cache_dir,
                trainer_id=args.trainer_id,
                manifest=manifest,
                verify_payload=True,
            )
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(
            f"Trainer {args.trainer_id} cache {args.command} failed: {error}"
        ) from error
    print(
        f"Trainer {args.trainer_id}: {args.command} OK; cache_id={manifest.cache_id}; "
        f"width={manifest.feature_dim}; source_run={manifest.source_run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
