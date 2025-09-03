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
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <iostream>
#include <cstdlib>
#include <ATen/ATen.h>
#include <cmath>

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
    this->Tensor_vec_geo_       = { this->_geo_grid_pts_ };       \
    this->Tensor_vec_sh0_       = { this->sh0_ };       \
    this->Tensor_vec_shs_       = { this->shs_ };       \

// #define VOXEL_MODEL_INIT_TENSORS(device_type)                                 \
//     this->center_               = torch::empty({0, 3},    torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
//     this->size_                 = torch::empty({0},       torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
//     this->_geo_grid_pts_         = torch::empty({0, 8},    torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
//     this->sh0_                  = torch::empty({0, 3},    torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
//     this->shs_                  = torch::empty({0, 45, 3},torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
//     this->oct_path_             = torch::empty({0},       torch::TensorOptions().dtype(torch::kLong   ).device(device_type)); \
//     this->oct_level_            = torch::empty({0},       torch::TensorOptions().dtype(torch::kInt32  ).device(device_type)); \
//     this->subdiv_meta_          = torch::empty({0},       torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
//     this->subdiv_p_             = torch::empty({0},       torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
//     VOXEL_MODEL_TENSORS_TO_VEC
#define VOXEL_MODEL_INIT_TENSORS(device_type)                                              \
    this->center_       = torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->size_         = torch::empty({0},    torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->_geo_grid_pts_= torch::empty({0, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->sh0_          = torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->shs_          = torch::empty({0, 0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->oct_path_     = torch::empty({0, 1}, torch::TensorOptions().dtype(torch::kLong   ).device(device_type)); \
    this->oct_level_    = torch::empty({0, 1}, torch::TensorOptions().dtype(torch::kInt8   ).device(device_type)); \
    this->subdiv_p_     = torch::empty({0, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->max_w_         = torch::empty({0, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    VOXEL_MODEL_TENSORS_TO_VEC

// namespace py = pybind11;
namespace sv {

struct TrainingStat {
  torch::Tensor max_w;             // (N,1)
  torch::Tensor min_samp_interval; // (N,1)
  torch::Tensor view_cnt;          // (N,1)
};

class VoxelModel 
{
public:
    VoxelModel(const int sh_degree);
    VoxelModel(const VoxelModelParams& model_params);
    ~VoxelModel();  

    void oneUpShDegree();
    void setShDegree(int sh);

    torch::Tensor voxCenter() { return this->center_; }

    void createFromPcd(const std::map<point3D_id_t, Point3D>& pcd);
    void increasePcd(std::vector<float> points, std::vector<float> colors, const int iteration);
    // void increasePcd(torch::Tensor& new_point_cloud, torch::Tensor& new_colors, const int iteration);

    // ───────── Optimizer setup ─────────
    void setGeoLearningRate(float geo_lr);
    void setSh0LearningRate(float sh0_lr);
    void setShsLearningRate(float shs_lr);
    void initOptimizer(float geo_lr, float sh0_lr, float shs_lr,
                    float beta1=0.9f, float beta2=0.999f, float eps=1e-15f);
    void rebuildOptimizer(float geo_lr, float sh0_lr, float shs_lr,
                        float beta1=0.9f, float beta2=0.999f, float eps=1e-15f);
    void optimizerZeroGrad();
    void optimizerStep();
    void setLearningRates(float geo_lr, float sh0_lr, float shs_lr);
    float multiStepDecay(int iter, float base_lr,
                        const std::vector<int>& milestones,
                        float gamma);
    void createTrainer(float geo_lr, float sh0_lr, float shs_lr,
                       float beta1=0.9f, float beta2=0.999f, float eps=1e-15f,
                       const std::vector<int>& milestones = {},
                       float gamma = 0.1f);
    py::object schedulerStateDict();
    void schedulerStep();
    void schedulerLoadStateDict(const py::object& state_dict);

    // torch::Tensor errorNormalized() const;
    // void accumulateError(const torch::Tensor& vis_idx,
    //                     const torch::Tensor& err);
    // torch::Tensor validMask(float size_mul) const;
    // TrainingStat computeTrainingStat(
    //     const std::vector<MiniCam>& cameras,
    //     const py::array_t<uint8_t>& rgb_image
    // );
    // void prune(const torch::Tensor& mask_keep);
    // void subdivide(const torch::Tensor& mask);
    // void setSubdivMeta(const torch::Tensor& updated);
    // /// Return gradient of `subdiv_p_`; requires that backward() was already called.
    // torch::Tensor getSubdivPriorityGrad() const;
    // /// Scatter‐add `parent_grads` into `subdiv_p_grad_buffer_` at indices `parent_idx`.
    // void accumulateSubdivGradients(
    //     const torch::Tensor& parent_idx,
    //     const torch::Tensor& parent_grads
    // );

    // === Adaptive API (SVRaster style) ===
    struct StatPkg {
        torch::Tensor max_w;             // [N,1], float32, cuda
        torch::Tensor min_samp_interval; // [N,1], float32, cuda
        torch::Tensor view_cnt;          // [N,1], float32, cuda
    };
    // Compute per-voxel stats over a list of training cameras.
    StatPkg computeTrainingStat(const std::vector<MiniCam>& cams);
    // Topology changes (mirrors python SparseVoxelModel)
    void pruning(const torch::Tensor& prune_mask);         // mask: [N] or [N,1] bool/byte
    void subdividing(const torch::Tensor& subdivide_mask); // mask: [N] or [N,1] bool/byte
    // Read helpers (mirrors SVM tensors)
    torch::Tensor subdivisionPriority() const; // [N,1]
    torch::Tensor voxSize() const;             // [N,1]
    torch::Tensor octLevel() const;            // [N,1] int8
    int           numVoxels() const;           // N
    int           maxNumLevels() const;        // from svraster_cuda.meta.MAX_NUM_LEVELS
    // Maintenance
    void resetSubdivisionPriority();
    void freezeVoxGeo();
    void unfreezeVoxGeo();

    /// Load all voxel fields from a PLY file (structure matches savePly).
    void loadPly(const std::filesystem::path& ply_file);
    /// Save all voxel data to a PLY file with full attributes.
    void savePly(const std::filesystem::path& result_path);
    /// Save only sparse voxels to PLY (not implemented here).
    void saveSparsePointsPly(const std::filesystem::path& result_path);

    std::unordered_map<std::string, torch::Tensor>
    render(
        const MiniCam&                   cam,
        const pybind11::array_t<uint8_t>& rgb_image
    ) const;

    void applyTvOnDensityField(float lambda_tv_density);

    float paramL2(const char* name);
    float gradL2(const char* name);
    void    debugParamChain();
    void    debugOptimizer();
    torch::Tensor snapParam(const char* name);                       // <- return Tensor
    double        deltaFrom(const char* name, const torch::Tensor& prev);

public:
    torch::DeviceType device_type_;

    /// Current & maximum SH degree
    int active_sh_degree_;
    int max_sh_degree_;

    int outside_level_ = 5;    // match svraster default
    int inside_level_  = 6;    // number of “dense” levels inside the bound

    /// Main voxel tensors
    torch::Tensor center_;             // [N, 3]
    torch::Tensor size_;               // [N]
    torch::Tensor geo_;                // [N, 8] (covariance and pad)
    torch::Tensor sh0_;                // [N, 3]
    torch::Tensor shs_;                // [N, 45, 3]
    torch::Tensor oct_path_;           // [N]
    torch::Tensor oct_level_;          // [N]
    torch::Tensor subdiv_p_;           // [N]
    torch::Tensor max_w_;              // [N]
    torch::Tensor grid_pts_key_;    // [M]  int64 keys identifying each grid-point
    torch::Tensor _geo_grid_pts_;    // [M]  float32 learnable density at each grid-point
    torch::Tensor vox_key_;         // [N,8] int64 indices into grid_pts_key_
    torch::Tensor vox_size_inv_;    // [N] float32 = 1.0 / size_

    std::vector<torch::Tensor>  Tensor_vec_geo_,
                                Tensor_vec_sh0_,
                                Tensor_vec_shs_;

    /// The Adam optimizer
    // std::shared_ptr<torch::optim::Adam> optimizer_;
    // py::object svm_;
    // py::object optimizer_py_;
    struct PyState;                // forward declaration only
    std::unique_ptr<PyState> py_;  // holds all pybind objects

    torch::Tensor sparse_points_xyz_;
    torch::Tensor sparse_points_color_;

    torch::Tensor scene_center_;
    torch::Tensor scene_extent_;
    torch::Tensor scene_inside_;
    torch::Tensor bb_min_eff_, bb_max_eff_;

protected:
    // Pull all core tensors from the Python SVM after any topology change.
    void syncFromPython_();
    // Helper math
    static torch::Tensor camPosition_(const MiniCam& cam, torch::Device d);
    static torch::Tensor camForward_(const MiniCam& cam, torch::Device d);
    static float         camPixSize_(const MiniCam& cam); // max(1/fx, 1/fy)
    // Cache MAX_NUM_LEVELS
    int max_num_levels_ = 16; // safe default; overwritten at init

    std::mutex mutex_settings_;
};

} // namespace sv