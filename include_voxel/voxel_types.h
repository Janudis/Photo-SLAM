#pragma once

#include <cstdint>

#include <Eigen/Core>

namespace sv {

using camera_id_t = std::uint32_t;
using point3D_id_t = std::uint64_t;

// Voxel-owned sparse point type used at the ORB-to-mapper boundary.
struct Point3D {
    Eigen::Vector3d xyz_ = Eigen::Vector3d::Zero();
    Eigen::Vector3f color_ = Eigen::Vector3f::Zero();
};

} // namespace sv
