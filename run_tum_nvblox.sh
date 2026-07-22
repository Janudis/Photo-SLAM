#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LD_LIBRARY_PATH="$root_dir/third_party/nvblox/build/nvblox:${LD_LIBRARY_PATH:-}"

bin="$root_dir/bin/tum_rgbd_nvblox"
vocabulary="$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt"
orb_config="$root_dir/cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_desk.yaml"
nvblox_config="$root_dir/cfg/nvblox_mapper/RGB-D/TUM/tum_rgbd_nvblox.yaml"
sequence="$root_dir/scripts/data/rgbd_dataset_freiburg1_desk"
association="$root_dir/cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg1_desk.txt"
output="${1:-$root_dir/results/tum_rgbd_nvblox/rgbd_dataset_freiburg1_desk/online_orb}"

mkdir -p "$output"
"$bin" "$vocabulary" "$orb_config" "$nvblox_config" \
    "$sequence" "$association" "$output"
