#pragma once

#include <torch/torch.h>
#include <Eigen/Core>
#include <string>
#include <vector>
#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

struct VoxelKeyframe {
    // === SLAM ===
    unsigned long fid_ = 0;
    std::string img_path_;
    Sophus::SE3f Tcw;

    // === Times of use ===
    int remaining_times_of_use_ = 1;

    // === Images ===
    torch::Tensor original_image_;        // undistorted RGB
    torch::Tensor auxiliary_image_;       // depth or stereo
    torch::Tensor gaus_pyramid_image_;    // low-res for training (optional)

    // === Keypoint info (optional for feature matching or densification) ===
    std::vector<float> kps_pixel_;        // 2N entries: [u0, v0, u1, v1, ...]
    std::vector<float> kps_point_local_;  // 3N entries: [x0, y0, z0, ...]

    // === Optional flags ===
    bool done_inactive_geo_densify_ = false;

    // === Derived intrinsics tensor (optional if needed) ===
    torch::Tensor intr_;  // (3x3)
};
