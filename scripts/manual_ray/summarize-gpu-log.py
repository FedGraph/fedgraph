#!/usr/bin/env python3
"""Summarize one experiment-scoped nvidia-smi CSV without third-party packages."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

FIELDNAMES = (
    "timestamp",
    "gpu_index",
    "gpu_uuid",
    "gpu_name",
    "gpu_utilization_pct",
    "memory_utilization_pct",
    "memory_used_mib",
    "memory_total_mib",
    "power_draw_watts",
    "temperature_gpu_c",
)
NUMERIC_FIELDS = FIELDNAMES[4:]


def parse_number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def summarize(input_path: Path) -> dict:
    samples: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with input_path.open(encoding="utf-8", errors="replace", newline="") as csv_file:
        for row in csv.reader(csv_file):
            if len(row) != len(FIELDNAMES):
                continue
            values = dict(zip(FIELDNAMES, (item.strip() for item in row)))
            samples[(values["gpu_index"], values["gpu_uuid"])].append(values)

    gpus = []
    for (gpu_index, gpu_uuid), rows in sorted(samples.items()):
        summary = {
            "gpu_index": gpu_index,
            "gpu_uuid": gpu_uuid,
            "gpu_name": rows[0]["gpu_name"],
            "sample_count": len(rows),
            "first_timestamp": rows[0]["timestamp"],
            "last_timestamp": rows[-1]["timestamp"],
        }
        for field in NUMERIC_FIELDS:
            values = [parse_number(row[field]) for row in rows]
            values = [value for value in values if value is not None]
            summary[f"{field}_max"] = max(values) if values else None
            summary[f"{field}_mean"] = sum(values) / len(values) if values else None
        gpus.append(summary)
    return {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_csv": str(input_path),
        "gpus": gpus,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
