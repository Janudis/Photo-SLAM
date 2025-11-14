#!/bin/bash

../bin/realsense_rgbd_voxel \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/RealCamera/realsense_d455_rgbd.yaml \
    ../cfg/voxel_mapper/Monocular/TUM/tum_mono_voxel.yaml \
    ../results/realsense_d455_rgbd