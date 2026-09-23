#!/bin/bash

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tum_data_root="${TUM_DATA_ROOT:-/media/dimitris/v4rl_rog_2t/Dimitris/Datasets_indoor/TUM}"

for i in 0 1 2 3 4
do
"$root_dir/bin/tum_rgbd" \
    "$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt" \
    "$root_dir/cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_desk.yaml" \
    "$root_dir/cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml" \
    "$tum_data_root/rgbd_dataset_freiburg1_desk" \
    "$root_dir/cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg1_desk.txt" \
    "$root_dir/results/tum_rgbd_/rgbd_dataset_freiburg1_desk" \
    no_viewer

"$root_dir/bin/tum_rgbd" \
    "$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt" \
    "$root_dir/cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg2_xyz.yaml" \
    "$root_dir/cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml" \
    "$tum_data_root/rgbd_dataset_freiburg2_xyz" \
    "$root_dir/cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg2_xyz.txt" \
    "$root_dir/results/tum_rgbd_$i/rgbd_dataset_freiburg2_xyz" \
    no_viewer

"$root_dir/bin/tum_rgbd" \
    "$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt" \
    "$root_dir/cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml" \
    "$root_dir/cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml" \
    "$tum_data_root/rgbd_dataset_freiburg3_long_office_household" \
    "$root_dir/cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt" \
    "$root_dir/results/tum_rgbd_$i/rgbd_dataset_freiburg3_long_office_household" \
    no_viewer
done
