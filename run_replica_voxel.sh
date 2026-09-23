#!/usr/bin/env bash

set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_replica_root="/media/dimitris/v4rl_rog_2t/Dimitris/Datasets_indoor/Replica"
replica_root="${REPLICA_ROOT:-$default_replica_root}"
sequence_name="${1:-${REPLICA_SEQUENCE:-office0}}"
sensor_mode="${REPLICA_SENSOR_MODE:-rgbd}"
viewer_mode="viewer"
if [[ "${VOXEL_VIEWER:-${REPLICA_VIEWER:-1}}" == "0" ]]; then
    viewer_mode="no_viewer"
fi

export LD_LIBRARY_PATH="/home/dimitris/opt/libtorch_2.0.1_cu118/libtorch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$root_dir:${PYTHONPATH:-}"

case "$sequence_name" in
    office0|office1|office2|office3|office4|room0|room1|room2)
        ;;
    *)
        printf '[ERROR] Unsupported Replica sequence: %s\n' "$sequence_name" >&2
        exit 2
        ;;
esac

sequence_dir="$replica_root/$sequence_name"
if [[ ! -d "$sequence_dir" ]]; then
    printf '[ERROR] Replica sequence not found: %s\n' "$sequence_dir" >&2
    exit 1
fi

vocabulary="$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt"
case "$sensor_mode" in
    monocular)
        bin="$root_dir/bin/replica_mono_voxel"
        orb_cfg="$root_dir/cfg/ORB_SLAM3/Monocular/Replica/$sequence_name.yaml"
        voxel_cfg="${REPLICA_VOXEL_CONFIG:-$root_dir/cfg/voxel_mapper/Monocular/Replica/replica_mono_voxel.yaml}"
        output_root="${REPLICA_VOXEL_OUTPUT_ROOT:-$root_dir/results/replica_mono_voxel}"
        run_id="${REPLICA_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
        output="${REPLICA_OUTPUT_DIRECTORY:-$output_root/$sequence_name/$run_id}"
        ;;
    rgbd)
        bin="$root_dir/bin/replica_rgbd_voxel"
        orb_cfg="$root_dir/cfg/ORB_SLAM3/RGB-D/Replica/$sequence_name.yaml"
        voxel_cfg="$root_dir/cfg/voxel_mapper/RGB-D/Replica/replica_rgbd_voxel.yaml"
        output_root="${REPLICA_RGBD_OUTPUT_ROOT:-$root_dir/results/replica_rgbd_voxel}"
        run_id="${REPLICA_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
        output="${REPLICA_OUTPUT_DIRECTORY:-$output_root/$sequence_name/$run_id}"
        ;;
    *)
        printf '[ERROR] REPLICA_SENSOR_MODE must be rgbd or monocular, got: %s\n' \
            "$sensor_mode" >&2
        exit 2
        ;;
esac

for required in "$bin" "$vocabulary" "$orb_cfg" "$voxel_cfg"; do
    if [[ ! -e "$required" ]]; then
        printf '[ERROR] Required input does not exist: %s\n' "$required" >&2
        exit 1
    fi
done

if [[ "${REPLICA_DRY_RUN:-0}" == "1" ]]; then
    printf '[DRY RUN] Sensor: %s\n' "$sensor_mode"
    printf '[DRY RUN] Sequence: %s\n' "$sequence_name"
    printf '[DRY RUN] Input dataset: %s\n' "$sequence_dir"
    printf '[DRY RUN] Mapper base: %s\n' "$voxel_cfg"
    printf '[DRY RUN] Output: %s\n' "$output"
    printf '[DRY RUN] Viewer mode: %s\n' "$viewer_mode"
    exit 0
fi

if [[ -d "$output" ]] && find "$output" -mindepth 1 -print -quit | grep -q .; then
    printf '[ERROR] Refusing to reuse non-empty output directory: %s\n' \
        "$output" >&2
    exit 1
fi

mkdir -p "$output/configs"
cp "$orb_cfg" "$output/configs/orb_slam.yaml"
orb_cfg="$output/configs/orb_slam.yaml"
cp "$voxel_cfg" "$output/configs/voxel_mapper.yaml"
voxel_cfg="$output/configs/voxel_mapper.yaml"

printf '[INFO] Replica %s: %s\n' "$sensor_mode" "$sequence_name"
printf '[INFO] Input dataset: %s\n' "$sequence_dir"
printf '[INFO] Mapper config: %s\n' "$voxel_cfg"
printf '[INFO] Output: %s\n' "$output"

"$bin" \
    "$vocabulary" \
    "$orb_cfg" \
    "$voxel_cfg" \
    "$sequence_dir" \
    "$output" \
    "$viewer_mode" \
    2>&1 | tee "$output/console.log"
