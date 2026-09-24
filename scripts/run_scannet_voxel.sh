#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/.." && pwd)"
export LD_LIBRARY_PATH="/home/dimitris/opt/libtorch_2.0.1_cu118/libtorch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$root_dir:${PYTHONPATH:-}"

SCANNET_ROOT="${SCANNET_ROOT:-/media/dimitris/v4rl_rog_2t/Dimitris/Datasets_indoor/ScanNet}"
SCANNET_SCENE="${1:-${SCANNET_SCENE:-scene0000_00}}"
SCANNET_FRAME_STRIDE="${SCANNET_FRAME_STRIDE:-1}"
SCANNET_SENSOR_MODE="${SCANNET_SENSOR_MODE:-monocular}"
SCANNET_WIDTH="${SCANNET_WIDTH:-536}"
SCANNET_HEIGHT="${SCANNET_HEIGHT:-400}"
SCANNET_FPS="${SCANNET_FPS:-30}"
SCANNET_PREPARE_PYTHON="${SCANNET_PREPARE_PYTHON:-/usr/bin/python3}"
prepare_python=(
    env
    -u PYTHONPATH
    PYTHONNOUSERSITE=1
    "$SCANNET_PREPARE_PYTHON"
)

scan_dir="$SCANNET_ROOT/scans/$SCANNET_SCENE"
sens_path="$scan_dir/$SCANNET_SCENE.sens"
association="$scan_dir/association.txt"
generated_mono_orb_cfg="$scan_dir/orb_slam3_monocular.yaml"
generated_rgbd_orb_cfg="$scan_dir/orb_slam3_rgbd.yaml"
prepare_metadata="$scan_dir/scannet_metadata.json"

if [[ ! -f "$sens_path" ]]; then
    printf '[ERROR] Missing ScanNet sensor stream: %s\n' "$sens_path" >&2
    printf 'Download it after accepting the ScanNet Terms of Use:\n' >&2
    printf '  python3 %q -o %q --id %q --type .sens\n' \
        "$root_dir/scripts/download-scannet.py" "$SCANNET_ROOT" "$SCANNET_SCENE" >&2
    exit 1
fi

prepared=1
for required in \
    "$association" \
    "$generated_mono_orb_cfg" \
    "$generated_rgbd_orb_cfg" \
    "$prepare_metadata"; do
    if [[ ! -f "$required" ]]; then
        prepared=0
    fi
done
if [[ "$prepared" == "1" ]] && ! "${prepare_python[@]}" - \
    "$prepare_metadata" \
    "$SCANNET_WIDTH" \
    "$SCANNET_HEIGHT" \
    "$SCANNET_FRAME_STRIDE" \
    "$SCANNET_FPS" <<'PY'
import json
import math
import sys

metadata = json.load(open(sys.argv[1], encoding="utf-8"))
valid = (
    metadata.get("preprocessing") == "photoslam_uncropped_v1"
    and metadata.get("crop_border") == 0
    and metadata.get("width") == int(sys.argv[2])
    and metadata.get("height") == int(sys.argv[3])
    and metadata.get("frame_stride") == int(sys.argv[4])
    and math.isclose(float(metadata.get("fps", -1)), float(sys.argv[5]))
)
raise SystemExit(0 if valid else 1)
PY
then
    prepared=0
fi

if [[ "$prepared" == "0" ]]; then
    printf '[INFO] Preparing uncropped Photo-SLAM ScanNet inputs in %s\n' "$scan_dir"
    if ! "${prepare_python[@]}" -c 'import cv2, numpy' >/dev/null 2>&1; then
        printf '[ERROR] ScanNet preparation requires compatible NumPy and OpenCV modules: %s\n' \
            "$SCANNET_PREPARE_PYTHON" >&2
        exit 1
    fi
    prepare_args=(
        --sens "$sens_path"
        --output-dir "$scan_dir"
        --frame-stride "$SCANNET_FRAME_STRIDE"
        --width "$SCANNET_WIDTH"
        --height "$SCANNET_HEIGHT"
        --fps "$SCANNET_FPS"
        --overwrite
    )
    "${prepare_python[@]}" \
        "$root_dir/scripts/prepare_scannet.py" "${prepare_args[@]}"
else
    printf '[INFO] Using prepared ScanNet frames in %s\n' "$scan_dir"
fi

vocabulary="$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt"
case "$SCANNET_SENSOR_MODE" in
    monocular)
        bin="$root_dir/bin/scannet_mono_voxel"
        orb_cfg="$generated_mono_orb_cfg"
        voxel_cfg="$root_dir/cfg/voxel_mapper/Monocular/ScanNet/scannet_mono_voxel.yaml"
        default_output="$root_dir/results/scannet_voxel/$SCANNET_SCENE"
        ;;
    rgbd)
        bin="$root_dir/bin/scannet_rgbd_voxel"
        orb_cfg="$generated_rgbd_orb_cfg"
        voxel_cfg="$root_dir/cfg/voxel_mapper/RGB-D/ScanNet/scannet_rgbd_voxel.yaml"
        default_output="$root_dir/results/scannet_rgbd_voxel/$SCANNET_SCENE"
        ;;
    *)
        printf '[ERROR] SCANNET_SENSOR_MODE must be rgbd or monocular, got: %s\n' \
            "$SCANNET_SENSOR_MODE" >&2
        exit 1
        ;;
esac
output="${SCANNET_OUTPUT_DIR:-$default_output}"
if [[ ! -f "$orb_cfg" ]]; then
    printf '[ERROR] Missing ORB-SLAM configuration: %s\n' "$orb_cfg" >&2
    exit 1
fi
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
