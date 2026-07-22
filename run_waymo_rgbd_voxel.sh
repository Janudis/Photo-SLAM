#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset_root="${WAYMO_ROOT:-/media/dimitris/T7 Shield/Dimitris/waymo_v2}"
segment="${WAYMO_SEGMENT:-10203656353524179475_7625_000_7645_000}"
source_segment="$dataset_root/validation/$segment"
prepared_segment="$dataset_root/prepared/$segment/front"
output_root="${WAYMO_OUTPUT:-$root_dir/results/waymo_rgbd_voxel/$segment}"

python_bin="${WAYMO_PYTHON:-/home/dimitris/miniconda3/envs/pyslam/bin/python}"
if [[ ! -f "$prepared_segment/rgbd.txt" ]] ||
   [[ ! -f "$prepared_segment/metadata.json" ]] ||
   [[ ! -f "$prepared_segment/depth_lidar_raw/000000.tiff" ]] ||
   ! grep -q '"lidar_projection_radius_px": 2' "$prepared_segment/metadata.json"; then
    "$python_bin" "$root_dir/scripts/prepare_waymo_v2.py" \
        --segment-root "$source_segment" \
        --output "$prepared_segment" \
        --with-lidar-depth \
        --lidar-projection-radius-px 2 \
        --overwrite
fi

export LD_LIBRARY_PATH="/home/dimitris/opt/libtorch_2.0.1_cu118/libtorch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$root_dir/third_party/simple-knn:${PYTHONPATH:-}"

mkdir -p "$output_root"
"$root_dir/bin/waymo_rgbd_voxel" \
    "$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt" \
    "$root_dir/cfg/ORB_SLAM3/RGB-D/Waymo/waymo_front_lidar.yaml" \
    "$root_dir/cfg/voxel_mapper/RGB-D/Waymo/waymo_lidar_voxel.yaml" \
    "$prepared_segment" \
    "$output_root" \
    no_viewer
