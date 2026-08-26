#!/usr/bin/env bash

set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BIN="$root_dir/bin/realsense_rgbd_voxel"
VOC="$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt"
ORB_CFG="$root_dir/cfg/ORB_SLAM3/RGB-D/RealCamera/realsense_d455f_rgbd_640x360.yaml"
VOX_CFG="$root_dir/cfg/voxel_mapper/RGB-D/RealCamera/realsense_rgbd_voxel.yaml"
RESULTS_ROOT="${PHOTOSLAM_RESULTS_ROOT:-$root_dir/results}"
OUT="${REALSENSE_OUTPUT:-$RESULTS_ROOT/realsense_rgbd_voxel/d455f/$(date +%Y%m%d-%H%M%S)}"

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "no_viewer" ) ]]; then
    echo "Usage: $0 [no_viewer]" >&2
    exit 1
fi
if [[ ! -x "$BIN" ]]; then
    echo "Missing executable: $BIN" >&2
    echo "Build it first with ./build.sh" >&2
    exit 1
fi

mkdir -p "$OUT"
echo "[RealSense] Output: $OUT"

viewer_args=()
if [[ $# -eq 1 ]]; then
    viewer_args+=("$1")
fi

exec "$BIN" \
    "$VOC" \
    "$ORB_CFG" \
    "$VOX_CFG" \
    "$OUT" \
    "${viewer_args[@]}"
