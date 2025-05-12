#!/usr/bin/env bash
set -e

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="/home/dimitris/opt/libtorch_2.0.1_cu118/libtorch/lib:$LD_LIBRARY_PATH"
# export LD_LIBRARY_PATH="$root_dir/../opt/libtorch_2.0.1_cu118/libtorch/lib"
# export PYTHONPATH="$root_dir/third_party:$PYTHONPATH"
export PYTHONPATH="$root_dir/third_party/simple-knn:$PYTHONPATH"

BIN=$root_dir/bin/tum_mono_voxel
VOC=$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt
ORB_CFG=$root_dir/cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg1_desk.yaml
VOX_CFG=$root_dir/cfg/voxel_mapper/Monocular/TUM/tum_mono_voxel.yaml
SEQ=$root_dir/scripts/data/rgbd_dataset_freiburg1_desk
OUT=$root_dir/results/tum_voxel/rgbd_dataset_freiburg1_desk
mkdir -p "$OUT"
"$BIN" "$VOC" "$ORB_CFG" "$VOX_CFG" "$SEQ" "$OUT" no_viewer

# # Correctly define LIBTORCH_LIB and PYTHONPATH
# LIBTORCH_LIB="$root_dir/../opt/libtorch_2.0.1_cu118/libtorch/lib"
# export LD_LIBRARY_PATH="$LIBTORCH_LIB:/lib:/usr/lib"
# export PYTHONPATH="$root_dir/third_party:$PYTHONPATH"

# BIN=$root_dir/bin/tum_mono_voxel
# VOC=$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt
# ORB_CFG=$root_dir/cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg1_desk.yaml
# VOX_CFG=$root_dir/cfg/voxel_mapper/Monocular/TUM/tum_mono_voxel.yaml
# SEQ=$root_dir/scripts/data/rgbd_dataset_freiburg1_desk
# OUT=$root_dir/results/tum_voxel/rgbd_dataset_freiburg1_desk

# mkdir -p "$OUT"

# # Run with cleaned env
# env -u LD_PRELOAD -u CUDA_VISIBLE_DEVICES \
#     LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
#     PYTHONPATH="$PYTHONPATH" \
#     "$BIN" "$VOC" "$ORB_CFG" "$VOX_CFG" "$SEQ" "$OUT" no_viewer
