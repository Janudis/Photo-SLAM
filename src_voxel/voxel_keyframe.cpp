#include "include_voxel/voxel_keyframe.h"
#include <iostream>

// Pose‐setter: 7‐number form
void VoxelKeyframe::setPose(double qw, double qx, double qy, double qz,
                            double tx, double ty, double tz)
{
    // Delegate to the Eigen‐Quaternion overload to avoid duplicating logic
    Eigen::Quaterniond q(qw, qx, qy, qz);
    q.normalize();
    setPose(q, Eigen::Vector3d(tx, ty, tz));
}

// Pose‐setter: Quaternion + translation
void VoxelKeyframe::setPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& t)
{
    Sophus::SE3d T_d(q.normalized(), t);
    Tcw = T_d.cast<float>();
}

// Pose getter (float)
Sophus::SE3f VoxelKeyframe::getPosef() const
{
    return Tcw;
}

// Pose getter (double)
Sophus::SE3d VoxelKeyframe::getPose() const
{
    return Tcw.cast<double>();
}

// Recompute c2w and w2c tensors
void VoxelKeyframe::computeTransformTensors()
{
    // Sanity check: if Tcw was never set, warn (optional)
    if (Tcw.matrix().isZero(0))
    {
        std::cerr << "[VoxelKeyframe] Warning: pose Tcw is zero for frame " << fid_ << std::endl;
    }

    Eigen::Matrix4f M = Tcw.matrix();
    // c2w_: world→camera 4×4
    c2w_ = torch::from_blob(M.data(), {4, 4}, torch::kFloat32).clone();
    // w2c_: inverse
    w2c_ = torch::linalg_inv(c2w_);
}

// Build a MiniCam for the rasterizer
sv::MiniCam VoxelKeyframe::toMiniCam() const
{
    return sv::fromCamera(cam_, c2w_, static_cast<int>(fid_));
}

// Camera remains a simple POD, so we directly assign in the header or here
void VoxelKeyframe::setCameraParams(const sv::Camera& cam)
{
    cam_ = cam;
}

//------------------------------------------------------------------------------
// Return the stored sv::Camera by reference
const sv::Camera& VoxelKeyframe::getCamera() const
{
    return cam_;
}

