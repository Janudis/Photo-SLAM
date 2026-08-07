#pragma once
/* -------------------------------------------------------------------------- */
/*  Photo-SLAM-style voxel key-frame                                          */
/* -------------------------------------------------------------------------- */
#include "include_voxel/voxel_camera.h"
#include "include_voxel/mini_cam.h"
#include "include_voxel/voxel_types.h"

#include <torch/torch.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <limits>
#include <utility>
#include <vector>
#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

namespace sv {

enum class LearnedDepthSource : std::uint8_t
{
    None = 0,
    TandemMvs = 1,
    Omnidata = 2,
};

} // namespace sv

class VoxelKeyframe
{
public:
    VoxelKeyframe() {}
    /* ------------------------------------------------------------------ ctor */
    VoxelKeyframe(std::size_t fid, int creation_iter = 0)
        : fid_(fid), creation_iter_(creation_iter) {}

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
    void  setCameraParams(const sv::Camera& camera);

    void computeTransformTensors();
    sv::MiniCam toMiniCam(int im_height = 480, int im_width = 640) const;

    Eigen::Matrix4f getWorld2View2(
        const Eigen::Vector3f& trans = {0.0f, 0.0f, 0.0f},
        float                  scale = 1.0f) const;

    torch::Tensor   getProjectionMatrix(
        float znear,
        float zfar,
        float fovX,
        float fovY,
        torch::DeviceType device_type = torch::kCUDA);

    int getCurrentGausPyramidLevel();

public:
    std::size_t fid_;
    int creation_iter_;
    int remaining_times_of_use_ = 0;

    bool set_camera_ = false;

    sv::camera_id_t camera_id_;
    int camera_model_id_ = 0;
    sv::Camera cam_;  

    std::string img_filename_;
    int source_frame_id_ = -1;
    double source_timestamp_ = std::numeric_limits<double>::quiet_NaN();
    cv::Mat img_undist_, img_auxiliary_undist_;
    torch::Tensor original_image_; ///< image
    int image_width_;              ///< image
    int image_height_;             ///< image

    int num_gaus_pyramid_sub_levels_;
    std::vector<int> gaus_pyramid_times_of_use_;
    std::vector<std::size_t> gaus_pyramid_width_;            ///< gaus_pyramid image
    std::vector<std::size_t> gaus_pyramid_height_;           ///< gaus_pyramid image
    std::vector<torch::Tensor> gaus_pyramid_original_image_; ///< gaus_pyramid image

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

    std::vector<float> kps_pixel_;
    std::vector<float> kps_point_local_;

    // Snapshot of ORB-SLAM3's local covisibility graph for MVS source-frame
    // selection. The mapper refreshes this metadata under Map::mMutexMapUpdate.
    std::vector<std::pair<unsigned long, int>> covisible_keyframes_;
    bool monocular_mvs_pose_ready_ = false;
    float monocular_mvs_depth_min_ = 0.0f;
    float monocular_mvs_depth_max_ = 0.0f;
    std::size_t monocular_mvs_sparse_depth_count_ = 0;
    std::vector<Eigen::Vector2f> monocular_depth_anchor_pixels_;
    std::vector<float> monocular_depth_anchor_depths_;

    // Accepted model depth in the ORB map gauge. MVS supplies its published
    // confidence, while Omnidata supplies the alignment/consistency weight.
    cv::Mat monocular_depth_prior_;
    cv::Mat monocular_depth_confidence_;
    sv::LearnedDepthSource monocular_depth_source_ =
        sv::LearnedDepthSource::None;
    int monocular_depth_prior_iteration_ = -1;

    bool done_inactive_geo_densify_ = false;
};
