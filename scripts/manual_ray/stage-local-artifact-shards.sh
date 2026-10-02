#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage: stage-local-artifact-shards.sh --rank-hosts-file PATH --source-dir PATH --remote-dir PATH [options]

Stage a completed manifest-style local artifact without replicating every shard
to every worker. Supported contracts are version 1/0-hop and version 2/2-hop.
The rank-host file has one '<rank> <private-ipv4>' pair per line; it must map
every rank in the artifact manifest exactly once. Each destination receives
manifest.json and only its assigned shards/trainer-NNN directory.

Options:
  --rank-hosts-file PATH  Complete rank-to-private-IP map.
  --source-dir PATH       Completed artifact root on the preparation host.
  --remote-dir PATH       Artifact root path on the destination workers.
  --ssh-user USER         Remote SSH user. Default: ubuntu.
  --help                  Show this message.

Set SSH_OPTS for an SSH identity or jump-host option. Metadata SHA-256 digests
are verified on every destination, so this script requires a checksummed artifact.
EOF
}

rank_hosts_file=""
source_dir=""
remote_dir=""
ssh_user="ubuntu"

while (( $# > 0 )); do
    case "$1" in
        --rank-hosts-file) rank_hosts_file="$2"; shift 2 ;;
        --source-dir) source_dir="$2"; shift 2 ;;
        --remote-dir) remote_dir="$2"; shift 2 ;;
        --ssh-user) ssh_user="$2"; shift 2 ;;
        --help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

[[ -n "$rank_hosts_file" ]] || die "--rank-hosts-file is required"
[[ -n "$source_dir" ]] || die "--source-dir is required"
[[ -n "$remote_dir" ]] || die "--remote-dir is required"
require_file "$rank_hosts_file"
require_file "${source_dir}/manifest.json"
require_command python3
require_command rsync
require_command ssh

source_dir="$(cd "$source_dir" && pwd)"
n_trainer="$(python3 - "${source_dir}/manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
contract = (manifest.get("artifact_version"), manifest.get("hop_semantics"))
if contract not in {(1, 0), (2, 2)}:
    raise SystemExit(
        "unsupported artifact contract: expected version 1/0-hop or "
        f"version 2/2-hop, got version {contract[0]!r}/{contract[1]!r}-hop"
    )
n_trainer = manifest.get("n_trainer")
if not isinstance(n_trainer, int) or n_trainer < 1:
    raise SystemExit("artifact manifest has an invalid n_trainer")
print(n_trainer)
PY
)"

read -r -a ssh_opts <<< "${SSH_OPTS:-}"
declare -A seen_ranks=()
staged_count=0

verify_remote_shard() {
    local host="$1"
    local shard_name="$2"
    local remote_dir_quoted
    local shard_quoted
    printf -v remote_dir_quoted '%q' "$remote_dir"
    printf -v shard_quoted '%q' "$shard_name"
    ssh "${ssh_opts[@]}" "${ssh_user}@${host}" \
        "python3 - ${remote_dir_quoted} ${shard_quoted}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

artifact_root = Path(sys.argv[1])
shard_name = sys.argv[2]
manifest = json.loads((artifact_root / "manifest.json").read_text(encoding="utf-8"))
shard_dir = artifact_root / "shards" / shard_name
metadata = json.loads((shard_dir / "metadata.json").read_text(encoding="utf-8"))
checksums = metadata.get("sha256")
if not isinstance(checksums, dict) or not checksums:
    raise SystemExit(f"{shard_dir} has no SHA-256 metadata")
for file_name, expected_size in metadata.get("files", {}).items():
    path = shard_dir / file_name
    if not path.is_file() or path.stat().st_size != expected_size:
        raise SystemExit(f"invalid staged file: {path}")
for file_name, expected_digest in checksums.items():
    digest = hashlib.sha256()
    with (shard_dir / file_name).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_digest:
        raise SystemExit(f"checksum mismatch: {shard_dir / file_name}")
print(f"Verified {shard_name}: {len(checksums)} checksums")
PY
}

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    line="${raw_line%%#*}"
    [[ -z "${line//[[:space:]]/}" ]] && continue
    read -r raw_rank host extra <<< "$line"
    [[ -n "${raw_rank:-}" && -n "${host:-}" && -z "${extra:-}" ]] || \
        die "Invalid placement entry: '${raw_line}'"
    [[ "$raw_rank" =~ ^[0-9]+$ ]] || die "Rank must be a non-negative integer: ${raw_rank}"
    rank="$((10#$raw_rank))"
    (( rank < n_trainer )) || die "Rank ${rank} is outside [0, $((n_trainer - 1))]"
    [[ -z "${seen_ranks[$rank]+x}" ]] || die "Rank ${rank} appears more than once"
    seen_ranks[$rank]="$host"

    shard_name="trainer-$(printf '%03d' "$rank")"
    source_shard="${source_dir}/shards/${shard_name}"
    require_file "${source_shard}/metadata.json"

    printf '\nStaging rank %s (%s) to %s\n' "$rank" "$shard_name" "$host"
    ssh "${ssh_opts[@]}" "${ssh_user}@${host}" \
        "mkdir -p '${remote_dir}/shards/${shard_name}'" </dev/null
    rsync -az -e "ssh ${SSH_OPTS:-}" \
        "${source_dir}/manifest.json" \
        "${ssh_user}@${host}:${remote_dir}/manifest.json" </dev/null
    rsync -az -e "ssh ${SSH_OPTS:-}" \
        "${source_shard}/" \
        "${ssh_user}@${host}:${remote_dir}/shards/${shard_name}/" </dev/null
    verify_remote_shard "$host" "$shard_name"
    ((staged_count += 1))
done < "$rank_hosts_file"

for ((rank = 0; rank < n_trainer; rank += 1)); do
    [[ -n "${seen_ranks[$rank]+x}" ]] || die "Rank-host map is missing rank ${rank}"
done

printf '\nStaged and verified %s artifact shard(s) across %s host mapping(s).\n' \
    "$staged_count" "${#seen_ranks[@]}"
