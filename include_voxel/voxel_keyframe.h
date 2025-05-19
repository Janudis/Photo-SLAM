#pragma once

#include <Eigen/Core>
#include <string>

struct VoxelKeyframe {
    unsigned long fid_;
    std::string img_path_;
    Sophus::SE3f Tcw;
    int remaining_times_of_use_ = 1;  // ← required field
};
