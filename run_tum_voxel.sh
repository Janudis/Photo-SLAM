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
mkdir -p "$OUT"
"$BIN" "$VOC" "$ORB_CFG" "$VOX_CFG" "$SEQ" "$OUT" no_viewer
# "$BIN" "$VOC" "$ORB_CFG" "$VOX_CFG" "$SEQ" "$OUT"
# echo "$BIN" "$VOC" "$ORB_CFG" "$VOX_CFG" "$SEQ" "$OUT" no_viewer #for debugging