#pragma once
/* -------------------------------------------------------------------------- */
/*  Photo-SLAM-style voxel key-frame                                          */
/* -------------------------------------------------------------------------- */
#include "include_voxel/voxel_camera.h"
#include "include_voxel/mini_cam.h"

#include <torch/torch.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <vector>
#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

#include "include/types.h"
#include "include/camera.h"
#include "include/point2d.h"
#include "include/general_utils.h"
#include "include/graphics_utils.h"
#include "include/tensor_utils.h"

class VoxelKeyframe
{
public:
    VoxelKeyframe() {}
    /* ------------------------------------------------------------------ ctor */
    VoxelKeyframe(std::size_t fid = 0, int create_iter = 0)
        : fid_(fid), creation_iter_(create_iter) {}

    /* ---------------------------------------------------- pose setters */
    void setPose(
        const double qw,
        const double qx,
        const double qy,
        const double qz,
        const double tx,
        const double ty,
        const double tz);
    
    void setPose(
        const Eigen::Quaterniond& q,
        const Eigen::Vector3d& t);

    Sophus::SE3d getPose();
    Sophus::SE3f getPosef();

    /* --------------------------------------------------------- camera stuff */
    void  setCameraParams(const sv::Camera& cam);

    void setPoints2D(const std::vector<Eigen::Vector2d>& points2D);
    void setPoint3DIdxForPoint2D(
        const point2D_idx_t point2D_idx,
        const point3D_id_t point3D_id);

    /* -------------------------------------------- transform tensors */
    void computeTransformTensors();
    sv::MiniCam toMiniCam() const;

    Eigen::Matrix4f getWorld2View2(
        const Eigen::Vector3f& trans = {0.0f, 0.0f, 0.0f},
        float                  scale = 1.0f) const;

    torch::Tensor   getProjectionMatrix(
        float znear,
        float zfar,
        float fovX,
        float fovY,
        torch::DeviceType device_type) const;

public:
    std::size_t fid_;
    int creation_iter_;
    int remaining_times_of_use_ = 0;

    bool set_camera_ = false;

    camera_id_t camera_id_;
    int camera_model_id_ = 0;
    sv::Camera cam_;  

    std::string img_filename_;
    cv::Mat img_undist_, img_auxiliary_undist_;
    torch::Tensor original_image_; ///< image
    int image_width_;              ///< image
    int image_height_;             ///< image

    std::vector<float> intr_; ///< intrinsics

    float FoVx_; ///< intrinsics
    float FoVy_; ///< intrinsics

    bool set_pose_ = false;
    bool set_projection_matrix_ = false;

    Eigen::Quaterniond R_quaternion_;  ///< extrinsics
    Eigen::Vector3d t_;                ///< extrinsics
    Sophus::SE3d Tcw_;                 ///< extrinsics

    torch::Tensor R_tensor_; ///< extrinsics
    torch::Tensor t_tensor_; ///< extrinsics

    float zfar_ = 100.0f;
    float znear_ = 0.01f;

    Eigen::Vector3f trans_ = {0.0f, 0.0f, 0.0f};
    float scale_ = 1.0f;

    torch::Tensor world_view_transform_;    ///< transform tensors
    torch::Tensor projection_matrix_;       ///< transform tensors
    torch::Tensor full_proj_transform_;     ///< transform tensors
    torch::Tensor camera_center_;           ///< transform tensors

    std::vector<Point2D> points2D_;
    std::vector<float> kps_pixel_;
    std::vector<float> kps_point_local_;

    bool done_inactive_geo_densify_ = false;
};

