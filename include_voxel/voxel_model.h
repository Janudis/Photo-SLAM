#pragma once

#include <memory>
#include <string>
#include <filesystem>
#include <mutex>
#include <vector>
#include <fstream>
#include <algorithm>
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <Python.h>  
#include <pybind11/embed.h>                         // for Python bridge
#include <torch/extension.h>
#include <iostream>
#include <cstdlib>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"
#include "third_party/simple-knn/spatial.h"
#include "third_party/tinyply/tinyply.h"
#include "include/types.h"
#include "include/point3d.h"
#include "include/operate_points.h"
#include "include/general_utils.h"
#include "include/sh_utils.h"
#include "include/tensor_utils.h"

#include "include_voxel/voxel_parameters.h"
#include "include_voxel/voxel_scene.h"
#include "include_voxel/voxel_keyframe.h"
#include "include_voxel/mini_cam.h"
#include "include_voxel/voxel_camera.h"
#include "include_voxel/py_utils.h"
#include "include_voxel/voxel_constants.h"

#define VOXEL_MODEL_TENSORS_TO_VEC                       \
    this->Tensor_vec_center_    = { this->center_ };    \
    this->Tensor_vec_geo_       = { this->geo_ };       \
    this->Tensor_vec_sh0_       = { this->sh0_ };       \
    this->Tensor_vec_shs_       = { this->shs_ };       \
    // this->Tensor_vec_opacity_   = { this->opacity_ };

#define VOXEL_MODEL_INIT_TENSORS(device_type)                                 \
    this->center_               = torch::empty({0, 3},    torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->size_                 = torch::empty({0},       torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->geo_                  = torch::empty({0, 8},    torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->sh0_                  = torch::empty({0, 3},    torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->shs_                  = torch::empty({0, 45, 3},torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->oct_path_             = torch::empty({0},       torch::TensorOptions().dtype(torch::kLong   ).device(device_type)); \
    this->oct_level_            = torch::empty({0},       torch::TensorOptions().dtype(torch::kInt32  ).device(device_type)); \
    this->subdiv_meta_          = torch::empty({0},       torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->subdiv_p_             = torch::empty({0},       torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->subdiv_p_grad_buffer_ = torch::empty({0},       torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    VOXEL_MODEL_TENSORS_TO_VEC
// this->opacity_              = torch::empty({0},       torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \

namespace py = pybind11;
namespace sv {

class VoxelModel 
{
public:
    /// Constructor #1: supply maximum SH degree
    VoxelModel(const int sh_degree);
    /// Constructor #2: supply full parameter struct
    VoxelModel(const VoxelModelParams& model_params);

    // ───────── Accessors ─────────
    torch::Tensor getCenters()           { return this->center_; }
    torch::Tensor getSizes()             { return this->size_; }
    torch::Tensor getGeo()               { return this->geo_; }
    torch::Tensor getSh0()               { return this->sh0_; }
    torch::Tensor getShs()               { return this->shs_; }
    // torch::Tensor getOpacity()           { return this->opacity_; }
    torch::Tensor getOctreePaths()       { return this->oct_path_; }
    torch::Tensor getOctreeLevels()      { return this->oct_level_; }
    torch::Tensor getSubdivMeta()        { return this->subdiv_meta_; }
    torch::Tensor getSubdivPriority()    { return this->subdiv_p_; }

    /// SH degree controls which spherical harmonics are active
    void oneUpShDegree();
    void setShDegree(int sh);

    // ───────── Incremental insertion ─────────
    void createFromPcd(
        const std::map<point3D_id_t, Point3D>& pcd,
        const float spatial_lr_scale
    );
    /// Insert new points/colors from std::vector
    void increasePcd(std::vector<float> points, std::vector<float> colors, const int iteration);
    void increasePcd(torch::Tensor& new_point_cloud, torch::Tensor& new_colors, const int iteration);

    // ───────── Transformations ─────────
    /// Apply a uniform scale and rigid SE3 transform to all voxels
    void applyScaledTransformation(
        const float s = 1.0,
        const Sophus::SE3f T = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()));
    void scaledTransformationPostfix(
        torch::Tensor& new_xyz
        //, torch::Tensor& new_scaling
    );

    void scaledTransformVisiblePointsOfKeyframe(
        torch::Tensor&     point_not_transformed_flags,
        torch::Tensor&     diff_pose,
        torch::Tensor&     kf_world_view_transform,
        torch::Tensor&     kf_full_proj_transform,
        const int          kf_creation_iter,
        const int          stable_num_iter_existence,
        int&               num_transformed,
        const float        scale = 1.0f
    );

    // ───────── Optimizer setup ─────────
    /**
     * Once all new points have been collected for this iteration,
     * call trainingSetup() to initialize the Adam optimizer and LR schedule.
     */
    void trainingSetup(const VoxelOptimizationParams& training_args);
    float updateLearningRate(int step);
    // ───────── Individual LR setters ─────────
    void setGeoLearningRate(float geo_lr);
    void setSh0LearningRate(float sh0_lr);
    void setShsLearningRate(float shs_lr);
    // void setOpacityLearningRate(float opacity_lr);

    /// Optionally: reset opacity to very low values, mirror GaussianModel::resetOpacity()
    // void resetOpacity();
    torch::Tensor replaceTensorToOptimizer(torch::Tensor& t, int tensor_idx);

    // ───────── Pruning & Densification ─────────
    /**
     * Remove voxels whose weight (opacity) fails to satisfy mask_keep (bool mask).
     */
    void prune(const torch::Tensor& mask_keep);
    /**
     * After each densification step, append newly‐split/pruned voxels back
     * into optimizer. Overloaded to match Photo‐SLAM’s VoxelTrainer logic.
     */
    /// Subdivide every “true” index in `mask` into 8 children.
    void subdivide(const torch::Tensor& mask);

    /// Replace `subdiv_meta_` with `updated` (leaf, requires_grad).
    void setSubdivMeta(const torch::Tensor& updated);
    /// Return gradient of `subdiv_p_`; requires that backward() was already called.
    torch::Tensor getSubdivPriorityGrad() const;
    /// Scatter‐add `parent_grads` into `subdiv_p_grad_buffer_` at indices `parent_idx`.
    void accumulateSubdivGradients(
        const torch::Tensor& parent_idx,
        const torch::Tensor& parent_grads
    );
    // void addDensificationStats(
    //     torch::Tensor& viewspace_point_tensor,
    //     torch::Tensor& update_filter
    // );

    /// Load all voxel fields from a PLY file (structure matches savePly).
    void loadPly(const std::filesystem::path& ply_file);
    /// Save all voxel data to a PLY file with full attributes.
    void savePly(const std::filesystem::path& result_path);
    /// Save only sparse voxels to PLY (not implemented here).
    void saveSparsePointsPly(const std::filesystem::path& result_path);

    /// Percentage of voxels considered “dense”
    float percentDense();
    void setPercentDense(const float percent_dense);

    // ───────── Utilities ─────────
    /**
     * Return all trainable parameters for the outer training loop
     * (mirrors GaussianModel::parameters()).
     */
    std::vector<torch::Tensor> parameters() const;
    /**
     * Lookup any named tensor by string (for debugging).
     * Supported names:
     *   "center", "size", "geo", "sh0", "shs", "opacity",
     *   "octpath", "octlevel", "exist_since", "subdiv_meta", "subdiv_p"
     */
    torch::Tensor getTensor(const std::string& name) const;
    void saveTorch(const std::filesystem::path& result_path) const;

    std::unordered_map<std::string, torch::Tensor>
    render(
        const MiniCam&                   cam,
        const pybind11::array_t<uint8_t>& rgb_image,
        const std::string&               output_dir = ""
    ) const;

protected:
    float exponLrFunc(int step);

public:
    // ───────── Member Variables ─────────

    /// Device (CPU vs CUDA)
    torch::DeviceType device_type_;

    /// Current & maximum SH degree
    int active_sh_degree_;
    int max_sh_degree_;

    /// Main voxel tensors
    torch::Tensor center_;             // [N, 3]
    torch::Tensor size_;               // [N]
    torch::Tensor geo_;                // [N, 8] (covariance and pad)
    torch::Tensor sh0_;                // [N, 3]
    torch::Tensor shs_;                // [N, 45, 3]
    // torch::Tensor opacity_;            // [N]
    /// Octree / subdivision tracking
    torch::Tensor oct_path_;           // [N]
    torch::Tensor oct_level_;          // [N]
    torch::Tensor subdiv_meta_;        // [N]
    torch::Tensor subdiv_p_;           // [N]
    torch::Tensor subdiv_p_grad_buffer_;// [N]
    torch::Tensor exist_since_iter_;   // [N] (iteration when created)

    /// Vectors of tensors for replacing into optimizer
    std::vector<torch::Tensor> 
    Tensor_vec_center_,
    Tensor_vec_geo_,
    Tensor_vec_sh0_,
    Tensor_vec_shs_;
    // Tensor_vec_opacity_;                               

    /// The Adam optimizer
    std::shared_ptr<torch::optim::Adam> optimizer_;
    /// Densification percentage threshold
    float percent_dense_;
    float spatial_lr_scale_;

    torch::Tensor sparse_points_xyz_;
    torch::Tensor sparse_points_color_;

protected:
    /// Learning‐rate scheduling parameters
    float lr_init_;
    float lr_final_;
    int   lr_delay_steps_;
    float lr_delay_mult_;
    int   max_steps_;

    /// Mutex to protect runtime hyperparameter changes
    std::mutex mutex_settings_;
};

} // namespace sv