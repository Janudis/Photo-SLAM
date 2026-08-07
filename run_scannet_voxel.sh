#!/usr/bin/env bash

set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="/home/dimitris/opt/libtorch_2.0.1_cu118/libtorch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$root_dir/third_party/simple-knn:${PYTHONPATH:-}"

SCANNET_ROOT="$root_dir/scripts/data/ScanNet"
SCANNET_SCENE="scene0000_00"
SCANNET_FRAME_STRIDE=1
SCANNET_SENSOR_MODE="${SCANNET_SENSOR_MODE:-monocular}"

scan_dir="$SCANNET_ROOT/scans/$SCANNET_SCENE"
sens_path="$scan_dir/$SCANNET_SCENE.sens"
association="$scan_dir/association.txt"
generated_rgbd_orb_cfg="$scan_dir/orb_slam3_rgbd.yaml"
prepare_marker="$scan_dir/.hislam2_scannet_preprocess_v1"

if [[ ! -f "$sens_path" ]]; then
    printf '[ERROR] Missing ScanNet sensor stream: %s\n' "$sens_path" >&2
    printf 'Download it after accepting the ScanNet Terms of Use:\n' >&2
    printf '  python3 %q -o %q --id %q --type .sens\n' \
        "$root_dir/scripts/download-scannet.py" "$SCANNET_ROOT" "$SCANNET_SCENE" >&2
    printf 'For reconstruction evaluation, also download the cleaned mesh:\n' >&2
    printf '  python3 %q -o %q --id %q --type _vh_clean_2.ply\n' \
        "$root_dir/scripts/download-scannet.py" "$SCANNET_ROOT" "$SCANNET_SCENE" >&2
    exit 1
fi

if [[ ! -f "$association" || ! -f "$generated_rgbd_orb_cfg" || ! -f "$prepare_marker" ]]; then
    prepare_args=(
        --sens "$sens_path"
        --output-dir "$scan_dir"
        --frame-stride "$SCANNET_FRAME_STRIDE"
        --fps 30
        --hi-slam2-preprocess
    )
    if [[ ! -f "$prepare_marker" ]]; then
        prepare_args+=(--overwrite)
    fi
    python3 "$root_dir/scripts/prepare_scannet.py" "${prepare_args[@]}"
else
    printf '[INFO] Using prepared ScanNet frames in %s\n' "$scan_dir"
fi

vocabulary="$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt"
case "$SCANNET_SENSOR_MODE" in
    monocular)
        bin="$root_dir/bin/scannet_mono_voxel"
        orb_cfg="$root_dir/cfg/ORB_SLAM3/Monocular/ScanNet/scene0000_00.yaml"
        voxel_cfg="$root_dir/cfg/voxel_mapper/Monocular/ScanNet/scannet_mono_voxel.yaml"
        output="$root_dir/results/scannet_voxel/$SCANNET_SCENE"
        ;;
    rgbd)
        bin="$root_dir/bin/scannet_rgbd_voxel"
        orb_cfg="$generated_rgbd_orb_cfg"
        voxel_cfg="$root_dir/cfg/voxel_mapper/RGB-D/ScanNet/scannet_rgbd_voxel.yaml"
        output="$root_dir/results/scannet_rgbd_voxel/$SCANNET_SCENE"
        ;;
    *)
        printf '[ERROR] SCANNET_SENSOR_MODE must be rgbd or monocular, got: %s\n' \
            "$SCANNET_SENSOR_MODE" >&2
        exit 1
        ;;
esac
mkdir -p "$output"

viewer_args=(no_viewer)
if [[ "${VOXEL_VIEWER:-${SCANNET_VIEWER:-0}}" == "1" ]]; then
    viewer_args=()
fi

if [[ "$SCANNET_SENSOR_MODE" == "monocular" ]]; then
    "$bin" \
        "$vocabulary" \
        "$orb_cfg" \
        "$voxel_cfg" \
        "$scan_dir" \
        "$output" \
        "${viewer_args[@]}"
else
    "$bin" \
        "$vocabulary" \
        "$orb_cfg" \
        "$voxel_cfg" \
        "$scan_dir" \
        "$association" \
        "$output" \
        "${viewer_args[@]}"
fi
