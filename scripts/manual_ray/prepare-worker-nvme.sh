#!/usr/bin/env bash

set -euo pipefail

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: prepare-worker-nvme.sh --device PATH|auto [options]

Mount the worker's instance-store NVMe and create the Hugging Face and Ray
directories expected by start-worker.sh. Run as the normal Ubuntu user, not
through sudo. The script does not modify /etc/fstab: instance-store devices are
ephemeral across stop/start and a stale UUID must not delay future boots.

Options:
  --device PATH|auto  Required block device, for example /dev/nvme1n1. Use
                      auto to detect exactly one safe AWS instance-store NVMe.
  --mount-point PATH  Mount location. Default: /mnt/fedgraph-nvme.
  --format            Format a blank supplied device as ext4. This is required
                      on a fresh or stop/started instance-store device.
  --help              Show this message.

Without --format, a device with an existing filesystem is mounted safely. A
device with no filesystem causes an error instead of being formatted implicitly.
EOF
}

device=""
mount_point="/mnt/fedgraph-nvme"
format_device=false

while (( $# > 0 )); do
    case "$1" in
        --device) device="$2"; shift 2 ;;
        --mount-point) mount_point="$2"; shift 2 ;;
        --format) format_device=true; shift ;;
        --help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done
(( EUID != 0 )) || die "Run this as the normal Ubuntu user; the script invokes sudo only where needed"
[[ -n "$device" ]] || die "--device is required; inspect candidates with lsblk -o NAME,SIZE,TYPE,MODEL,FSTYPE,MOUNTPOINTS"
command -v sudo >/dev/null 2>&1 || die "sudo is required"
command -v findmnt >/dev/null 2>&1 || die "findmnt is required"
command -v lsblk >/dev/null 2>&1 || die "lsblk is required"
command -v mountpoint >/dev/null 2>&1 || die "mountpoint is required"

root_source="$(findmnt -n -o SOURCE --target /)"
root_source_real="$(readlink -f "$root_source")"
root_disk="$(lsblk -ndo PKNAME "$root_source_real" 2>/dev/null || true)"
if [[ -z "$root_disk" ]]; then
    root_disk="$(lsblk -ndo NAME "$root_source_real" 2>/dev/null || true)"
fi
[[ -n "$root_disk" ]] || die "Could not identify the root filesystem disk"

if [[ "$device" == "auto" ]]; then
    declare -a candidates=()
    while IFS= read -r candidate_name candidate_type; do
        [[ "$candidate_type" == "disk" && "$candidate_name" == nvme*n* ]] || continue
        [[ "$candidate_name" != "$root_disk" ]] || continue
        candidate="/dev/${candidate_name}"
        candidate_model="$(lsblk -ndo MODEL "$candidate" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
        [[ "$candidate_model" == *"Instance Storage"* ]] || continue
        candidate_partitions="$(lsblk -nrpo NAME "$candidate" | tail -n +2 | wc -l)"
        [[ "$candidate_partitions" == "0" ]] || continue
        candidates+=("$candidate")
    done < <(lsblk -dn -o NAME,TYPE)

    (( ${#candidates[@]} == 1 )) || die "--device auto found ${#candidates[@]} safe AWS instance-store NVMe candidates; inspect lsblk -o NAME,SIZE,TYPE,MODEL,FSTYPE,MOUNTPOINTS and provide an explicit --device"
    device="${candidates[0]}"
    printf 'Auto-selected instance-store device: %s\n' "$device"
fi

[[ -b "$device" ]] || die "$device is not a block device"
device_real="$(readlink -f "$device")"
device_disk="$(lsblk -ndo PKNAME "$device_real" 2>/dev/null || true)"
if [[ -z "$device_disk" ]]; then
    device_disk="$(lsblk -ndo NAME "$device_real" 2>/dev/null || true)"
fi
[[ -n "$device_disk" ]] || die "Could not identify the physical disk behind $device"
[[ "$device_disk" != "$root_disk" ]] || die "$device is on the root filesystem disk and will not be used as scratch storage"
partition_count="$(lsblk -nrpo NAME "$device_real" | tail -n +2 | wc -l)"
[[ "$partition_count" == "0" ]] || die "$device has child partitions; use an unpartitioned instance-store device instead"

mounted_target="$(findmnt -n -o TARGET --source "$device_real" 2>/dev/null || true)"
if [[ -n "$mounted_target" && "$mounted_target" != "$mount_point" ]]; then
    die "$device is already mounted at $mounted_target"
fi

if mountpoint -q "$mount_point"; then
    mounted_source="$(findmnt -n -o SOURCE --target "$mount_point")"
    [[ "$(readlink -f "$mounted_source")" == "$device_real" ]] || die "$mount_point is mounted from $mounted_source, not $device"
else
    filesystem="$(lsblk -ndo FSTYPE "$device_real" | head -n 1)"
    if [[ -z "$filesystem" ]]; then
        [[ "$format_device" == "true" ]] || die "$device has no filesystem; rerun with --format only after verifying lsblk output"
        printf 'Formatting blank device %s as ext4\n' "$device"
        sudo mkfs.ext4 -F "$device_real"
    elif [[ "$format_device" == "true" ]]; then
        die "$device already contains a $filesystem filesystem; refusing to reformat it"
    fi

    sudo mkdir -p "$mount_point"
    sudo mount "$device_real" "$mount_point"
fi

sudo chown "$(id -un):$(id -gn)" "$mount_point"
mkdir -p "$mount_point/hf-cache" "$mount_point/ray"
df -h "$mount_point"
printf 'Prepared cache: %s\n' "$mount_point/hf-cache"
printf 'Prepared Ray spill directory: %s\n' "$mount_point/ray"
