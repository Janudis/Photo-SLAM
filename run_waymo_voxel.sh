#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset_root="${WAYMO_ROOT:-/media/dimitris/T7 Shield/Dimitris/Datasets/waymo_v2}"
segment="${WAYMO_SEGMENT:-10203656353524179475_7625_000_7645_000}"
source_segment="$dataset_root/validation/$segment"
prepared_segment="$dataset_root/prepared/$segment/front"
voxel_config="$root_dir/cfg/voxel_mapper/Monocular/Waymo/waymo_mono_voxel.yaml"
output_root="${WAYMO_OUTPUT:-$root_dir/results/waymo_mono_voxel/$segment}"

if [[ ! -d "$source_segment" ]]; then
    printf '[ERROR] Missing Waymo segment on the external SSD: %s\n' \
        "$source_segment" >&2
    printf 'Set WAYMO_ROOT if the dataset is mounted elsewhere.\n' >&2
    exit 1
fi

python_bin="${WAYMO_PYTHON:-/home/dimitris/miniconda3/envs/pyslam/bin/python}"
if [[ ! -f "$prepared_segment/rgb.txt" ]]; then
    "$python_bin" "$root_dir/scripts/prepare_waymo_v2.py" \
        --segment-root "$source_segment" \
        --output "$prepared_segment" \
        --overwrite
fi

export LD_LIBRARY_PATH="/home/dimitris/opt/libtorch_2.0.1_cu118/libtorch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$root_dir/third_party/simple-knn:${PYTHONPATH:-}"

mkdir -p "$output_root"
printf '[INFO] Waymo dataset: %s\n' "$dataset_root"
printf '[INFO] Waymo segment: %s\n' "$segment"
printf '[INFO] Mapper config: %s\n' "$voxel_config"
printf '[INFO] Output: %s\n' "$output_root"
"$root_dir/bin/waymo_mono_voxel" \
    "$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt" \
    "$root_dir/cfg/ORB_SLAM3/Monocular/Waymo/waymo_front.yaml" \
    "$voxel_config" \
    "$prepared_segment" \
    "$output_root" \
    no_viewer
