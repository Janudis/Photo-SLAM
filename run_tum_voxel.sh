#!/usr/bin/env bash

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LD_LIBRARY_PATH="/home/dimitris/opt/libtorch_2.0.1_cu118/libtorch/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$root_dir/third_party/simple-knn:$PYTHONPATH"

# BIN=$root_dir/bin/tum_mono_voxel
# VOC=$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt
# ORB_CFG=$root_dir/cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg1_desk.yaml
# VOX_CFG=$root_dir/cfg/voxel_mapper/Monocular/TUM/tum_mono_voxel.yaml
# SEQ=$root_dir/scripts/data/rgbd_dataset_freiburg1_desk
# OUT=$root_dir/results/tum_voxel/rgbd_dataset_freiburg1_desk

# BIN=$root_dir/bin/tum_rgbd_voxel
# VOC=$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt
# ORB_CFG=$root_dir/cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_desk.yaml
# VOX_CFG=$root_dir/cfg/voxel_mapper/RGB-D/TUM/tum_rgbd_voxel.yaml
# SEQ=$root_dir/scripts/data/rgbd_dataset_freiburg1_desk
# ASSOC=$root_dir/cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg1_desk.txt
# OUT=$root_dir/results/tum_rgbd/rgbd_dataset_freiburg1_desk

# BIN=$root_dir/bin/replica_mono_voxel
# VOC=$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt
# ORB_CFG=$root_dir/cfg/ORB_SLAM3/Monocular/Replica/office0.yaml
# VOX_CFG=$root_dir/cfg/voxel_mapper/Monocular/Replica/replica_mono_voxel.yaml
# SEQ=$root_dir/scripts/data/Replica/office0
# OUT=$root_dir/results/replica_voxel/office0

BIN="$root_dir/bin/replica_rgbd_voxel"
VOC="$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt"
ORB_CFG="$root_dir/cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml"
VOX_CFG="$root_dir/cfg/voxel_mapper/RGB-D/Replica/replica_rgbd_voxel.yaml"
SEQ="$root_dir/scripts/data/Replica/office0"
OUT="$root_dir/results/replica_rgbd_voxel/office0"

# BIN=$root_dir/bin/statues_voxel
# VOC=$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt
# ORB_CFG=$root_dir/cfg/ORB_SLAM3/Monocular/SplatNav/statues.yaml
# VOX_CFG=$root_dir/cfg/voxel_mapper/Monocular/TUM/tum_mono_voxel.yaml
# SEQ=$root_dir/scripts/data/statues
# OUT=$root_dir/results/statues_voxel

mkdir -p "$OUT"

# Mono / Replica RGB-D runners: BIN VOC ORB_CFG VOX_CFG SEQ OUT [no_viewer]
# TUM RGB-D runners:            BIN VOC ORB_CFG VOX_CFG SEQ ASSOC OUT [no_viewer]
if [[ "$(basename "$BIN")" == "tum_rgbd_voxel" ]]; then
    "$BIN" "$VOC" "$ORB_CFG" "$VOX_CFG" "$SEQ" "$ASSOC" "$OUT" no_viewer
else
    "$BIN" "$VOC" "$ORB_CFG" "$VOX_CFG" "$SEQ" "$OUT" no_viewer
fi

# "$BIN" "$VOC" "$ORB_CFG" "$VOX_CFG" "$SEQ" "$OUT"
# echo "$BIN" "$VOC" "$ORB_CFG" "$VOX_CFG" "$SEQ" "$OUT" no_viewer #for debugging
