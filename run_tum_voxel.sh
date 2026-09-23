#!/usr/bin/env bash

set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tum_data_root="${TUM_DATA_ROOT:-/media/dimitris/v4rl_rog_2t/Dimitris/Datasets_indoor/TUM}"
sequence_name="${1:-${TUM_SEQUENCE:-rgbd_dataset_freiburg1_desk}}"

unset PYTHONNOUSERSITE
export LD_LIBRARY_PATH="/home/dimitris/opt/libtorch_2.0.1_cu118/libtorch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$root_dir:${PYTHONPATH:-}"

if ! /usr/bin/python3.10 -c \
    'import rerun; import rerun.blueprint' >/dev/null 2>&1; then
    echo "[ERROR] Python Rerun SDK 0.34.1 is unavailable to /usr/bin/python3.10." >&2
    echo "Install it with: /usr/bin/python3.10 -m pip install --user rerun-sdk==0.34.1" >&2
    exit 1
fi

case "$sequence_name" in
    rgbd_dataset_freiburg1_desk)
        sequence_config="tum_freiburg1_desk"
        ;;
    rgbd_dataset_freiburg2_xyz)
        sequence_config="tum_freiburg2_xyz"
        ;;
    rgbd_dataset_freiburg3_long_office_household)
        sequence_config="tum_freiburg3_long_office_household"
        ;;
    *)
        echo "[ERROR] Unsupported TUM sequence: $sequence_name" >&2
        exit 2
        ;;
esac

VOC="$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt"
SEQ="$tum_data_root/$sequence_name"
sensor_mode="${TUM_SENSOR_MODE:-monocular}"
case "$sensor_mode" in
    monocular)
        BIN="$root_dir/bin/tum_mono_voxel"
        ORB_CFG="$root_dir/cfg/ORB_SLAM3/Monocular/TUM/$sequence_config.yaml"
        VOX_CFG="${TUM_VOXEL_CONFIG:-$root_dir/cfg/voxel_mapper/Monocular/TUM/tum_mono_voxel.yaml}"
        OUT="${TUM_OUTPUT_DIR:-$root_dir/results/tum_mono_voxel/$sequence_name}"
        ;;
    rgbd)
        BIN="$root_dir/bin/tum_rgbd_voxel"
        ORB_CFG="$root_dir/cfg/ORB_SLAM3/RGB-D/TUM/$sequence_config.yaml"
        VOX_CFG="${TUM_VOXEL_CONFIG:-$root_dir/cfg/voxel_mapper/RGB-D/TUM/tum_rgbd_voxel.yaml}"
        ASSOC="$root_dir/cfg/ORB_SLAM3/RGB-D/TUM/associations/$sequence_config.txt"
        OUT="${TUM_OUTPUT_DIR:-$root_dir/results/tum_rgbd_voxel/$sequence_name}"
        ;;
    *)
        echo "[ERROR] TUM_SENSOR_MODE must be monocular or rgbd, got: $sensor_mode" >&2
        exit 2
        ;;
esac

mkdir -p "$OUT"

if [[ ! -d "$SEQ" ]]; then
    echo "[ERROR] TUM sequence not found: $SEQ" >&2
    exit 1
fi

viewer_args=(no_viewer)
if [[ "${VOXEL_VIEWER:-0}" == "1" ]]; then
    viewer_args=()
    export VOXEL_VIEWER_PRESET="${VOXEL_VIEWER_PRESET:-tracking}"
fi

if [[ "$sensor_mode" == "rgbd" ]]; then
    "$BIN" "$VOC" "$ORB_CFG" "$VOX_CFG" "$SEQ" "$ASSOC" "$OUT" "${viewer_args[@]}"
else
    "$BIN" "$VOC" "$ORB_CFG" "$VOX_CFG" "$SEQ" "$OUT" "${viewer_args[@]}"
fi
