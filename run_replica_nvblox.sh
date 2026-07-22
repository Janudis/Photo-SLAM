#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LD_LIBRARY_PATH="$root_dir/third_party/nvblox/build/nvblox:${LD_LIBRARY_PATH:-}"

bin="$root_dir/bin/replica_rgbd_nvblox"
vocabulary="$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt"
orb_config="$root_dir/cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml"
nvblox_config="$root_dir/cfg/nvblox_mapper/RGB-D/Replica/replica_rgbd_nvblox.yaml"
sequence="$root_dir/scripts/data/Replica/office0"
output="${1:-$root_dir/results/replica_rgbd_nvblox/office0/online_orb}"

mkdir -p "$output"
"$bin" "$vocabulary" "$orb_config" "$nvblox_config" \
    "$sequence" "$output"
