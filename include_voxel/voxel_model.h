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
// #include <Python.h>  
// #include <pybind11/embed.h>                         // for Python bridge
// #include <pybind11/stl.h>
// #include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <iostream>
#include <cstdlib>
#include <ATen/ATen.h>
#include <cmath>
#include <tuple>
#include <optional>

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
#include "include_voxel/render_opts.h" 
#include <rerun.hpp>
#include "include_voxel/svraster_utils.h"       // sv::oct::*
#include "include_voxel/voxel_renderer.h"
#include "include_voxel/opt_sched.hpp"


#define VOXEL_MODEL_TENSORS_TO_VEC                       \
    this->Tensor_vec_geo_       = { this->_geo_grid_pts_ };       \
    this->Tensor_vec_sh0_       = { this->sh0_ };       \
    this->Tensor_vec_shs_       = { this->shs_ };       \

#define VOXEL_MODEL_INIT_TENSORS(device_type)                                              \
    this->center_       = torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->size_         = torch::empty({0},    torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->_geo_grid_pts_= torch::empty({0, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->sh0_          = torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->shs_          = torch::empty({0, 0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->oct_path_     = torch::empty({0, 1}, torch::TensorOptions().dtype(torch::kLong   ).device(device_type)); \
    this->oct_level_    = torch::empty({0, 1}, torch::TensorOptions().dtype(torch::kInt8   ).device(device_type)); \
    this->subdiv_p_     = torch::empty({0, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->max_w_        = torch::empty({0, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    VOXEL_MODEL_TENSORS_TO_VEC

// namespace py = pybind11;

namespace sv {
class VoxelModel 
{
public:
    VoxelModel(const int sh_degree);
    VoxelModel(const VoxelModelParams& model_params);
    ~VoxelModel();  

    void oneUpShDegree();
    void setShDegree(int sh);

    torch::Tensor voxCenter() { return this->center_; }
    const torch::Tensor& geoGridPts() const;
    const torch::Tensor& sh0() const;
    const torch::Tensor& shs() const;
    // Read helpers (mirrors SVM tensors)
    int64_t numGridPts() const;
    torch::Tensor subdivisionPriority() const; // [N,1]
    torch::Tensor voxSize() const;             // [N,1]
    torch::Tensor octLevel() const;            // [N,1] int8
    torch::Tensor octPath() const;             // [N,1] int64
    int           numVoxels() const;           // N
    int           maxNumLevels() const;        // from svraster_cuda.meta.MAX_NUM_LEVELS
    torch::Tensor SceneCenter() const;  
    torch::Tensor SceneExtent() const;
    // Helper math
    static torch::Tensor camPosition_(const MiniCam& cam, torch::Device d);
    static torch::Tensor camForward_(const MiniCam& cam, torch::Device d);
    static float         camPixSize_(const MiniCam& cam); // max(1/fx, 1/fy)

    // ---- public SVRaster-style accessors (put in public section) ----
    int64_t num_grid_pts() const;
    const at::Tensor& vox_center()   const;
    const at::Tensor& vox_size()     const;
    const at::Tensor& grid_pts_key() const;
    const at::Tensor& vox_key()      const;
    const at::Tensor& vox_size_inv() const;
    const at::Tensor& grid_pts_xyz() const; // optional, used where needed

    void createFromPcd(const std::map<point3D_id_t, Point3D>& pcd, const std::vector<sv::MiniCam>& cams);
    void increasePcd(std::vector<float> points, std::vector<float> colors, const int iteration, const std::vector<sv::MiniCam>& cams);
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
    // py::object schedulerStateDict();
    void schedulerStep();
    // void schedulerLoadStateDict(const py::object& state_dict);
    sv::optim::MultiStepLRState schedulerStateDict() const;
    void schedulerLoadStateDict(const sv::optim::MultiStepLRState& state);
    std::tuple<double,double,double> currentLearningRates() const;
    void appendGroup_(int group_idx, const torch::Tensor& add_rows, const char* svm_field_name, torch::Tensor* out_member_param);

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
    // Maintenance
    void resetSubdivisionPriority();
    // void freezeVoxGeo();
    // void unfreezeVoxGeo();

    /// Load all voxel fields from a PLY file (structure matches savePly).
    void loadPly(const std::filesystem::path& ply_file);
    /// Save all voxel data to a PLY file with full attributes.
    void savePly(const std::filesystem::path& result_path);
    /// Save only sparse voxels to PLY (not implemented here).
    void saveSparsePointsPly(const std::filesystem::path& result_path);

    std::unordered_map<std::string, torch::Tensor>
    // render(const MiniCam& cam, torch::Tensor gt_image, int image_height, int image_width, float ss = 1.0f, bool track_max_w = false) const;
    render(const sv::MiniCam& cam,
        int im_height,
        int im_width,
        const torch::Tensor& gt_image = torch::Tensor(),
        const char* color_mode = nullptr,
        bool track_max_w = false,
        std::optional<float> ss = std::nullopt,
        bool output_depth=false,
        bool output_normal=false,
        bool output_T=false,
        bool rand_bg=false,
        bool use_auto_exposure=false,
        const sv::RenderOpts& other_opt = sv::RenderOpts()) const;

    void applyTvOnDensityField(float lambda_tv_density);

    float paramL2(const char* name);
    float gradL2(const char* name);
    void    debugParamChain();
    void    debugOptimizer();
    torch::Tensor snapParam(const char* name);                       // <- return Tensor
    double        deltaFrom(const char* name, const torch::Tensor& prev);

    void rrInitOnce();
    void rrLogPointsAndAABB(int iteration,                   // keep if you want for labels
                            const at::Tensor& xyz,
                            const at::Tensor& rgb,
                            const at::Tensor& bounding_2x3,
                            const std::string& tag,
                            int64_t max_points_for_viz = 200000,
                            bool log_points = true,
                            bool log_box   = true,
                            int64_t inc = -1); 
    void logParamSignature(const char* tag);
    void rrLogVoxelBoxes(const at::Tensor& centers,
                         const at::Tensor& sizes,
                         int64_t max_boxes_for_viz = 100000000,
                         const std::string& tag = "boxes",
                         int64_t inc = -1);
    void rrLogGlobalSceneAABB(int64_t inc);

    void debugAssertTopologyConsistent(const char* where) const;

private:
    std::unique_ptr<rerun::RecordingStream> rr_;
    std::vector<rerun::Vec3D> rr_acc_mins_;
    std::vector<rerun::Vec3D> rr_acc_sizes_;
    std::vector<rerun::Rgba32> rr_acc_colors_;
    std::vector<std::string>   rr_acc_labels_;
    bool rr_initialized_ = false;
    int64_t rr_inc_step_ = 0;  // unique per increasePcd() call

public:
    torch::DeviceType device_type_;

    /// Current & maximum SH degree
    int active_sh_degree_;
    int max_sh_degree_;

    /// Main voxel tensors
    mutable torch::Tensor center_;            
    mutable torch::Tensor size_;               
    torch::Tensor geo_;               
    torch::Tensor sh0_;               
    torch::Tensor shs_;               
    torch::Tensor oct_path_;           
    torch::Tensor oct_level_;         
    torch::Tensor subdiv_p_;          
    torch::Tensor max_w_;              
    mutable torch::Tensor grid_pts_key_;    
    torch::Tensor _geo_grid_pts_;    
    mutable torch::Tensor vox_key_;         
    mutable torch::Tensor vox_size_inv_;    
    torch::Tensor frozen_vox_geo_;
    mutable torch::Tensor _grid_pts_xyz_;    
    
    float ss_ = 1.5f;
    bool  white_background_ = false;
    bool  black_background_ = false;
    int   n_samp_per_vox_ = 3;

    // ---- Signature (parity to SVRaster) ----
    struct DerivedSignature {
        int64_t num_voxels = -1;
        const c10::TensorImpl* octpath_impl = nullptr;
        const c10::TensorImpl* octlevel_impl = nullptr;
        bool operator==(const DerivedSignature& o) const {
            return num_voxels == o.num_voxels &&
                octpath_impl == o.octpath_impl &&
                octlevel_impl == o.octlevel_impl;
        }
        bool operator!=(const DerivedSignature& o) const { return !(*this == o); }
    };

    mutable DerivedSignature _check_derived_voxel_attr_signature_{};
    mutable DerivedSignature _vox_size_inv_signature_{};
    mutable DerivedSignature _grid_pts_xyz_signature_{};

    // ---- helpers (SVRaster-compatible names) ----
    DerivedSignature signature() const;
    void _check_derived_voxel_attr() const;

    std::vector<torch::Tensor>  Tensor_vec_geo_,
                                Tensor_vec_sh0_,
                                Tensor_vec_shs_;

    torch::Tensor sparse_points_xyz_;
    torch::Tensor sparse_points_color_;

    torch::Tensor scene_center_;   // [3], CUDA
    torch::Tensor scene_extent_;   // [1], CUDA
    torch::Tensor scene_min_t_;      // [3], CUDA
    torch::Tensor vox_eff_;      // [1,1], CUDA (effective voxel size for fixed level)
    int8_t  octlevel_ = 0; 
    torch::Tensor inside_extent_; // [1], CUDA

    float global_scene_extent_ = 200.0f;  // example: 200 m cube
    std::array<float,3> global_scene_center_{0.f, 0.f, 0.f};
    float fixed_vox_size_ = 0.05f;        // your chosen voxel size

    bool   fill_empty_cells_ = true; // use bigger voxels here instead of 0.05
    int64_t max_artifact_cells_ = 200000; // safety cap
    std::array<float,3> artifact_bg_rgb_{0.5f,0.5f,0.5f}; // gray (or 1,1,1 for white)

    torch::Tensor bb_min_viz, bb_max_viz, sel_artifacts_viz, ijk_box_viz;


protected:
    /// The Adam optimizer
    // std::shared_ptr<torch::optim::Adam> optimizer_;
    // py::object svm_;
    // py::object optimizer_py_;
    // struct PyState;                // forward declaration only
    // std::unique_ptr<PyState> py_;  // holds all pybind objects
    // Pull all core tensors from the Python SVM after any topology change.
    void syncFromPython_();
    // Cache MAX_NUM_LEVELS
    const int max_num_levels_ = 16; 

    std::mutex mutex_settings_;

    std::unique_ptr<sv::optim::SparseAdam>  optimizer_;
    std::unique_ptr<sv::optim::MultiStepLR> scheduler_;
    std::vector<int> sched_milestones_;
    double sched_gamma_ = 1.0;
};

} // namespace sv