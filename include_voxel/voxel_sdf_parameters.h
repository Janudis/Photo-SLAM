#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <cstdint>
#include <limits>
#include <string>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

struct VoxelSdfParameters
{
    float sdf_voxel_size_m_ = 0.05f;
    float sdf_init_trunc_vox_ = 4.0f;
    float sdf_init_max_depth_m_ = 4.0f;
};

struct VoxelSdfState
{
    bool projective_sdf_init_context_valid_ = false;
    cv::Mat projective_sdf_init_depth_meters_;
    Sophus::SE3f projective_sdf_init_Tcw_;
    float projective_sdf_init_fx_ = 0.0f;
    float projective_sdf_init_fy_ = 0.0f;
    float projective_sdf_init_cx_ = 0.0f;
    float projective_sdf_init_cy_ = 0.0f;
    int projective_sdf_init_width_ = 0;
    int projective_sdf_init_height_ = 0;
    std::size_t projective_sdf_init_kfid_ = 0;
};
