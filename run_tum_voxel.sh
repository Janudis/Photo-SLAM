#!/usr/bin/env bash

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LD_LIBRARY_PATH="/home/dimitris/opt/libtorch_2.0.1_cu118/libtorch/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$root_dir/third_party/simple-knn:$PYTHONPATH"

BIN=$root_dir/bin/tum_mono_voxel
VOC=$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt
ORB_CFG=$root_dir/cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg1_desk.yaml
VOX_CFG=$root_dir/cfg/voxel_mapper/Monocular/TUM/tum_mono_voxel.yaml
SEQ=$root_dir/scripts/data/rgbd_dataset_freiburg1_desk
OUT=$root_dir/results/tum_voxel/rgbd_dataset_freiburg1_desk

# BIN=$root_dir/bin/tum_rgbd_voxel
# VOC=$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt
# ORB_CFG=$root_dir/cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_desk.yaml
# VOX_CFG=$root_dir/cfg/voxel_mapper/Monocular/TUM/tum_mono_voxel.yaml
# SEQ=$root_dir/scripts/data/rgbd_dataset_freiburg1_desk
# ASSOC=$root_dir/cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg1_desk.txt
# OUT=$root_dir/results/tum_rgbd/rgbd_dataset_freiburg1_desk

# BIN=$root_dir/bin/replica_mono_voxel
# VOC=$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt
# ORB_CFG=$root_dir/cfg/ORB_SLAM3/Monocular/Replica/office0.yaml
# VOX_CFG=$root_dir/cfg/voxel_mapper/Monocular/TUM/tum_mono_voxel.yaml
# SEQ=$root_dir/scripts/data/Replica/office0
# OUT=$root_dir/results/replica_voxel/office0

# BIN=$root_dir/bin/realsense_rgbd_voxel
# VOC=$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt
# ORB_CFG=$root_dir/cfg/ORB_SLAM3/RGB-D/RealCamera/realsense_d455_rgbd.yaml
# VOX_CFG=$root_dir/cfg/voxel_mapper/Monocular/TUM/tum_mono_voxel.yaml
# OUT=$root_dir/results/realsense_d455_rgbd

# BIN=$root_dir/bin/meganerf_mono
# VOC=$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt
# ORB_CFG=$root_dir/cfg/ORB_SLAM3/Monocular/MegaNeRF/building.yaml
# VOX_CFG=$root_dir/cfg/voxel_mapper/Monocular/TUM/tum_mono_voxel.yaml
# SEQ=$root_dir/scripts/data/building-pixsfm-seq-847-940
# OUT=$root_dir/results/meganerf_mono_

mkdir -p "$OUT"

"$BIN" "$VOC" "$ORB_CFG" "$VOX_CFG" "$SEQ" "$OUT" no_viewer
# "$BIN" "$VOC" "$ORB_CFG" "$VOX_CFG" "$SEQ" "$OUT"

#tum_rgbd_voxel
# "$BIN" "$VOC" "$ORB_CFG" "$VOX_CFG" "$SEQ" "$ASSOC" "$OUT" no_viewer

#realsense_rgbd_voxel
# "$BIN" "$VOC" "$ORB_CFG" "$VOX_CFG" "$OUT"

#for debugging
# "$BIN" "$VOC" "$ORB_CFG" "$VOX_CFG" "$SEQ" "$OUT"
# echo "$BIN" "$VOC" "$ORB_CFG" "$VOX_CFG" "$SEQ" "$OUT" no_viewer 