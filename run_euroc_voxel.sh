#!/usr/bin/env bash

set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sequence_name="${EUROC_SEQUENCE:-V1_01_easy}"

case "$sequence_name" in
    MH_01_easy|MH_02_easy|V1_01_easy|V2_01_easy)
        ;;
    *)
        echo "[ERROR] Unsupported downloaded EuRoC sequence: $sequence_name" >&2
        exit 1
        ;;
esac

export LD_LIBRARY_PATH="/home/dimitris/opt/libtorch_2.0.1_cu118/libtorch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$root_dir/third_party/simple-knn:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

bin="$root_dir/bin/euroc_mono_voxel"
vocabulary="$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt"
orb_config="$root_dir/cfg/ORB_SLAM3/Monocular-Inertial/EuRoC/EuRoC.yaml"
voxel_config="$root_dir/cfg/voxel_mapper/Monocular/EuRoC/euroc_mono_voxel.yaml"
sequence="$root_dir/scripts/data/EuRoC/$sequence_name"
output="$root_dir/results/euroc_voxel/$sequence_name"

mkdir -p "$output"

viewer_args=(no_viewer)
if [[ "${VOXEL_VIEWER:-0}" == "1" ]]; then
    viewer_args=()
fi

"$bin" \
    "$vocabulary" \
    "$orb_config" \
    "$voxel_config" \
    "$sequence" \
    "$output" \
    "${viewer_args[@]}"
