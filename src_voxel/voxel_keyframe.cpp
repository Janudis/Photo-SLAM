#include "include_voxel/voxel_keyframe.h"
#include "include_voxel/voxel_mapper_utils.h"
#include <iostream>
#include <cmath>

// Pose‐setter: 7‐number form
void VoxelKeyframe::setPose(
    const double qw,
    const double qx,
    const double qy,
    const double qz,
    const double tx,
    const double ty,
    const double tz)
{
    this->R_quaternion_.w() = qw;
    this->R_quaternion_.x() = qx;
    this->R_quaternion_.y() = qy;
    this->R_quaternion_.z() = qz;
    this->R_quaternion_.normalize();
    this->t_.x() = tx;
    this->t_.y() = ty;
    this->t_.z() = tz;

    this->Tcw_ = Sophus::SE3d(this->R_quaternion_, this->t_);

    this->set_pose_ = true;
}

// Pose‐setter: Eigen overload
void VoxelKeyframe::setPose(
    const Eigen::Quaterniond& q,
    const Eigen::Vector3d&    t)
{
    this->R_quaternion_ = q;
    this->R_quaternion_.normalize();
    this->t_ = t;

    this->Tcw_ = Sophus::SE3d(this->R_quaternion_, this->t_);

    this->set_pose_ = true;
}

Sophus::SE3d VoxelKeyframe::getPose()
{
    return this->Tcw_;
}

Sophus::SE3f VoxelKeyframe::getPosef()
{
    return this->Tcw_.cast<float>();
}

// Copy sv::Camera in
void VoxelKeyframe::setCameraParams(const sv::Camera& camera)
{
    this->cam_ = camera; 
    this->camera_id_ = camera.camera_id_;
    this->camera_model_id_ = camera.model_id_;
    this->image_height_ = camera.height_;
    this->image_width_ = camera.width_;

    this->num_gaus_pyramid_sub_levels_ = camera.num_gaus_pyramid_sub_levels_;
    this->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
    this->gaus_pyramid_width_ = camera.gaus_pyramid_width_;

    this->intr_.resize(camera.params_.size());
    for (std::size_t i = 0; i < camera.params_.size(); ++i)
        this->intr_[i] = static_cast<float>(camera.params_[i]);

    switch (this->camera_model_id_)
    {
    case 1: // Pinhole
    {
        float focal_length_x = static_cast<float>(camera.params_[0]);
        float focal_length_y = static_cast<float>(camera.params_[1]);
        this->FoVx_ = sv::focalToFov(focal_length_x, camera.width_);
        this->FoVy_ = sv::focalToFov(focal_length_y, camera.height_);
        this->set_camera_ = true;
        // std::cout << "voxel_keyframe.cpp: setCameraParams() Pinhole camera model with FoVx: "
        //           << this->FoVx_ << " and FoVy: " << this->FoVy_ << std::endl;
    }
    break;

    default:
    {
        throw std::runtime_error("Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!");
    }
    break;
    }
}

void VoxelKeyframe::computeTransformTensors()
{
    if (this->set_pose_ && this->set_camera_) {
        this->world_view_transform_ = voxel_utils::eigenMatrixToTorchTensor(
            this->getWorld2View2(this->trans_, this->scale_),
            torch::kCUDA
        ).transpose(0, 1);
        // std::cout << " voxel_keyframe znear " << this->znear_ << " zfar " << this->zfar_ << std::endl;
        if (!this->set_projection_matrix_) {
            this->projection_matrix_ = this->getProjectionMatrix(
                this->znear_,
                this->zfar_,
                this->FoVx_,
                this->FoVy_,
                torch::kCUDA
            ).transpose(0, 1);
            this->set_projection_matrix_ = true;
        }

        this->full_proj_transform_ = (this->world_view_transform_.unsqueeze(0).bmm(
            this->projection_matrix_.unsqueeze(0))).squeeze(0);

        this->camera_center_ = this->world_view_transform_.inverse().index({3, torch::indexing::Slice(0, 3)});
    }
    else if (!this->set_pose_ && this->set_camera_) {
        std::cerr << "Could not compute transform tensors for keyframe " << this->fid_ << " because POSE is not set!" << std::endl;
    }
    else if (!this->set_camera_) {
        std::cerr << "Could not compute transform tensors for keyframe " << this->fid_ << " because CAMERA is not set!" << std::endl;
    }
    else {
        std::cerr << "Could not compute transform tensors for keyframe " << this->fid_ << " because POSE and CAMERA are not set!" << std::endl;
    }
}

Eigen::Matrix4f VoxelKeyframe::getWorld2View2(
    const Eigen::Vector3f& trans,
    float                  scale) const
{    
    Eigen::Matrix4f Rt;
    Rt.setZero();
    Eigen::Matrix3f R = this->R_quaternion_.toRotationMatrix().cast<float>();
    Rt.topLeftCorner<3, 3>() = R;
    Eigen::Vector3f t = this->t_.cast<float>();
    Rt.topRightCorner<3, 1>() = t;
    Rt(3, 3) = 1.0f;

    Eigen::Matrix4f C2W = Rt.inverse();
    Eigen::Vector3f cam_center = C2W.block<3, 1>(0, 3);
    // std::cout << "VoxelKeyframe::getWorld2 trans " << trans << " scale" << scale << std::endl;
    cam_center += trans;
    cam_center *= scale;
    C2W.block<3, 1>(0, 3) = cam_center;
    Rt = C2W.inverse();
    return Rt;
}

torch::Tensor VoxelKeyframe::getProjectionMatrix(
    float znear,
    float zfar,
    float fovX,
    float fovY,
    torch::DeviceType device_type)
{
    float tanHalfFovY = std::tan(fovY / 2);
    float tanHalfFovX = std::tan(fovX / 2);

    float top = tanHalfFovY * znear;
    float bottom = -top;
    float right = tanHalfFovX * znear;
    float left = -right;

    torch::Tensor P = torch::zeros({4, 4}, torch::TensorOptions().device(device_type));

    float z_sign = 1.0f;

    P.index({0, 0}) = 2.0 * znear / (right - left);
    P.index({1, 1}) = 2.0 * znear / (top - bottom);
    P.index({0, 2}) = (right + left) / (right - left);
    P.index({1, 2}) = (top + bottom) / (top - bottom);
    P.index({3, 2}) = z_sign;
    P.index({2, 2}) = z_sign * zfar / (zfar - znear);
    P.index({2, 3}) = -(zfar * znear) / (zfar - znear);
    return P;
}

int VoxelKeyframe::getCurrentGausPyramidLevel()
{
    for (int i = 0; i < gaus_pyramid_times_of_use_.size(); ++i) {
        if (gaus_pyramid_times_of_use_[i]) {
            --gaus_pyramid_times_of_use_[i];
            return i;
        }
    }
    // If all sub levels has been used up
    return num_gaus_pyramid_sub_levels_;
}

// helper to build MiniCam
sv::MiniCam VoxelKeyframe::toMiniCam(int im_height, int im_width) const
{
    // pure camera pose:  Tcw_  is world→cam  ⇒  c2w = Tcw_.inverse()
    Eigen::Matrix4f w2c_eig = Tcw_.matrix().cast<float>();
    Eigen::Matrix4f c2w_eig = Tcw_.inverse().matrix().cast<float>();
    torch::Tensor c2w = voxel_utils::eigenMatrixToTorchTensor(
                            c2w_eig, torch::kCPU);      // keep on CPU
    torch::Tensor w2c = voxel_utils::eigenMatrixToTorchTensor(
                            w2c_eig, torch::kCPU);      // keep on CPU
    // std::cout << "voxel_keyframe.cpp: toMiniCam() FoVx_ = " << FoVx_ << std::endl;

    return sv::fromCamera(cam_, c2w, im_height, im_width, static_cast<int>(fid_));
}
