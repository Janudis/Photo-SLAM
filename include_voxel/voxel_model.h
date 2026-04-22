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
    struct IncreasePcdStats {
        int64_t raw_points_in = 0;
        int64_t points_after_far_filter = 0;
        int64_t unique_voxel_candidates_before_insert_filter = 0;
        int64_t unique_voxel_candidates_after_insert_filter = 0;
        int64_t duplicate_existing_voxels = 0;
        int64_t new_voxels = 0;
        int64_t pending_promotions = 0;
        int64_t pending_support_updates = 0;
    };

    VoxelModel(const int sh_degree);
    VoxelModel(const VoxelModelParams& model_params);
    ~VoxelModel();  

    void oneUpShDegree();
    void setShDegree(int sh);

    torch::Tensor voxCenter() { return this->center_; }
    int64_t numGridPts() const;
    const torch::Tensor& geoGridPts() const;
    const torch::Tensor& sh0() const;
    const torch::Tensor& shs() const;
    torch::Tensor voxelDensityMean() const;
    torch::Tensor gridPtsKey() const { return this->grid_pts_key_; }
    torch::Tensor voxKey() const { return this->vox_key_; }
    
    torch::Tensor artificialMask() const { return this->is_artificial_voxel_; } // [N] bool, true=artificial
    torch::Tensor promotedartificialMask() const { return this->is_promoted_artificial_voxel_; } // [N] bool
    torch::Tensor existSinceIter() const { return this->exist_since_iter_; } // [N] int32, voxel creation iter
    torch::Tensor existSinceKf() const { return this->exist_since_kf_; } // [N] int32, voxel creation keyframe-count
    torch::Tensor geometricallyUnstableMask() const { return this->geometrically_unstable_voxel_; } // [N] bool
    torch::Tensor renderedDepthCandidateMask() const { return this->rendered_depth_candidate_voxel_; } // [N] bool
    torch::Tensor renderedDepthCandidateSupportCount() const { return this->rendered_depth_candidate_support_count_; } // [N] int32
    torch::Tensor renderedDepthCandidateLastSeenKf() const { return this->rendered_depth_candidate_last_seen_kf_; } // [N] int32
    torch::Tensor renderedDepthCandidateSourceKind() const { return this->rendered_depth_candidate_source_kind_; } // [N] int32
    bool hasDenseCoreBB() const { return has_dense_core_bb_; }
    torch::Tensor denseCoreBBMin() const { return dense_core_bb_min_; } // [3]
    torch::Tensor denseCoreBBMax() const { return dense_core_bb_max_; } // [3]
    bool refreshDenseCoreBBFromCurrentVoxels(bool exclude_hole_fill_real_voxels = false);
    void logDenseCoreBBoxToRerun(
        int iteration,
        const std::string& entity_path = "world/dense_core/used_for_prune") const;
    void setGeometricallyUnstableMask(const torch::Tensor& mask);
    void setEnableArtificialPromotion(const bool enable) { enable_artificial_promotion_ = enable; }
    bool enableArtificialPromotion() const { return enable_artificial_promotion_; }
    void setFilterNearVoxels(const bool enable) { filter_near_voxels_ = enable; }
    bool filterNearVoxels() const { return filter_near_voxels_; }
    void setFilterFarVoxelsOnInsert(const bool enable) { filter_far_voxels_on_insert_ = enable; }
    bool filterFarVoxelsOnInsert() const { return filter_far_voxels_on_insert_; }
    void setRenderedDepthCandidateRealAdjacency(
        const bool enable,
        const int radius_cells = 1)
    {
        rendered_depth_candidate_require_real_adjacency_ = enable;
        rendered_depth_candidate_adjacency_radius_cells_ = std::max(1, radius_cells);
    }
    bool renderedDepthCandidateRealAdjacency() const {
        return rendered_depth_candidate_require_real_adjacency_;
    }
    int renderedDepthCandidateAdjacencyRadiusCells() const {
        return rendered_depth_candidate_adjacency_radius_cells_;
    }
    void setNextRealInsertionRerunEntityPath(const std::string& entity_path) {
        pending_real_insert_rr_entity_path_ = entity_path;
    }
    void setNextRenderedDepthCandidateInsertion(const bool enable,
                                                const std::string& entity_path = "",
                                                const int source_kind = 1,
                                                const bool insert_as_real_protected = false) {
        pending_insert_rendered_depth_candidate_ = enable;
        pending_artificial_insert_rr_entity_path_ = enable ? entity_path : "";
        pending_insert_rendered_depth_candidate_source_kind_ =
            enable ? std::max(1, source_kind) : 0;
        pending_insert_rendered_depth_candidate_as_real_protected_ =
            enable && insert_as_real_protected;
    }
    int64_t totalPromotedartificialCount() const { return total_promoted_artificial_voxels_; }

    void createFromPcd(
        const std::map<point3D_id_t, Point3D>& pcd,
        const std::vector<sv::MiniCam>& cams);
    void increasePcd(
        std::vector<float> points,
        std::vector<float> colors,
        const int iteration,
        const std::vector<sv::MiniCam>& cams);
    void increasePcd(
        torch::Tensor& new_point_cloud,
        torch::Tensor& new_colors,
        const int iteration,
        const std::vector<sv::MiniCam>& cams);
    IncreasePcdStats lastIncreasePcdStats() const { return last_increase_pcd_stats_; }
    void setTopologyBirthContext(const int iteration, const int kf_count) {
        topology_birth_iter_ = static_cast<int32_t>(iteration);
        topology_birth_kf_ = static_cast<int32_t>(kf_count);
    }
    void promoteRenderedDepthCandidates(const torch::Tensor& promote_mask);
    void logFinalartificialVoxels(const int iteration);
    void logFinalPromotedartificialVoxels(const int iteration);

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
    // Read helpers (mirrors SVM tensors)
    torch::Tensor subdivisionPriority() const; // [N,1]
    torch::Tensor voxSize() const;             // [N,1]
    torch::Tensor octLevel() const;            // [N,1] int8
    torch::Tensor octPath() const;             // [N,1] int64
    int           numVoxels() const;           // N
    int           maxNumLevels() const;        // from svraster_cuda.meta.MAX_NUM_LEVELS
    torch::Tensor SceneCenter() const;  
    torch::Tensor SceneExtent() const;
    float         fixedVoxSize() const { return fixed_vox_size_; }
    int           baseOctLevel() const { return static_cast<int>(octlevel_); }
    // Maintenance
    void resetSubdivisionPriority();
    void freezeVoxGeo();
    void unfreezeVoxGeo();

    /// Load all voxel fields from a PLY file (structure matches savePly).
    void loadPly(const std::filesystem::path& ply_file);
    /// Save all voxel data to a PLY file with full attributes.
    void savePly(const std::filesystem::path& result_path);
    /// Save a triangle surface mesh extracted from voxel occupancy (PLY with faces).
    void saveSurfaceMeshPly(const std::filesystem::path& result_path);
    /// Save only sparse voxels to PLY (not implemented here).
    void saveSparsePointsPly(const std::filesystem::path& result_path);

    void savePlannerNPZ(const std::filesystem::path& npz_path, int target_max_voxels = 1000000);

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
    void debugAssertTopologyConsistent(const char* where) const;
    void logParamSignature(const char* tag);

    // Initialize once & spawn viewer
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
    void rrLogVoxelBoxes(const at::Tensor& centers,
                         const at::Tensor& sizes,
                         int64_t max_boxes_for_viz = 100000000,
                         const std::string& tag = "boxes",
                         int64_t inc = -1);
    void rrLogGlobalSceneAABB(int64_t inc);
    void rrLogTrainingCamera(const sv::MiniCam& cam, int64_t inc);
    void rrLogVoxelBoxesWithShColor(const at::Tensor& centers,
                                    const at::Tensor& sizes,
                                    const at::Tensor& sh0,
                                    int64_t max_boxes_for_viz = 100000,
                                    const std::string& tag = "boxes_sh_color",
                                    int64_t inc = -1);

    void applyTsdfTransparency(const torch::Tensor& tsdf_mask, float geo_value);
    void applySingleVoxelTsdfTransparency(int64_t voxel_idx, float geo_value);

private:
    std::unique_ptr<rerun::RecordingStream> rr_;
    std::vector<rerun::Vec3D> rr_acc_mins_;
    std::vector<rerun::Vec3D> rr_acc_sizes_;
    std::vector<rerun::Rgba32> rr_acc_colors_;
    std::vector<std::string>   rr_acc_labels_;
    bool rr_initialized_ = false;
    int64_t rr_inc_step_ = 0;  // unique per increasePcd() call
    IncreasePcdStats last_increase_pcd_stats_;
    int last_artificial_iter_ = -1;
    torch::Tensor art_key_before_iter_; // [K] int64 artificial keys at start of current iter

public:
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
    torch::Tensor oct_path_;           // [N]
    torch::Tensor oct_level_;          // [N]
    torch::Tensor subdiv_p_;           // [N]
    torch::Tensor max_w_;              // [N]
    torch::Tensor grid_pts_key_;    // [M]  int64 keys identifying each grid-point
    torch::Tensor _geo_grid_pts_;    // [M]  float32 learnable density at each grid-point
    torch::Tensor vox_key_;         // [N,8] int64 indices into grid_pts_key_
    torch::Tensor vox_size_inv_;    // [N] float32 = 1.0 / size_
    torch::Tensor frozen_vox_geo_;
    
    float ss_ = 1.5f;
    bool  white_background_ = false;
    bool  black_background_ = false;
    int   n_samp_per_vox_ = 3;

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

    float global_scene_extent_ = 200.0f;  // example: 100 m cube
    std::array<float,3> global_scene_center_{0.f, 0.f, 0.f};
    float fixed_vox_size_ = 0.1f;     

    bool   fill_empty_cells_ = false;
    int    fill_empty_cells_warmup_iters_ = 300; // delay one-shot bbox fill until this iteration
    bool   use_local_frontier_fill_ = false;  // preferred over dense bbox fill
    bool   use_dense_core_neighbor_fill_ = false; // controlled fill in dense-core region
    float  dense_core_pcd_density_rate_ = 0.001f; // same meaning as SVRaster heuristic - smaller the bigger the area
    int64_t max_dense_core_fill_cells_ = 50000; // cap dense-core support cells per call
    int64_t max_real_pcd_points_ = 800000;      // cap accumulated real PCD points
    int64_t max_artificial_cells_ = 1000000; // safety cap
    std::array<float,3> artificial_bg_rgb_{0.5f,0.5f,0.5f}; // gray (or 1,1,1 for white)

    torch::Tensor bb_min_viz, bb_max_viz, sel_artificials_viz, ijk_box_viz;
    torch::Tensor artificial_centers_accum_viz_; // [K,3] accumulated artificial centers
    torch::Tensor artificial_sizes_accum_viz_;   // [K,1] accumulated artificial sizes
    torch::Tensor inactive_geo_centers_accum_viz_; // [K,3] accumulated real voxels created by inactive geo densify
    torch::Tensor inactive_geo_sizes_accum_viz_;   // [K,1] accumulated sizes for inactive geo densify voxels
    torch::Tensor inactive_geo_rgba_accum_viz_;    // [K,4] accumulated colors for inactive geo densify voxels
    torch::Tensor depthanything_created_centers_accum_viz_; // [K,3] accumulated real voxels created by depthanything densify
    torch::Tensor depthanything_created_sizes_accum_viz_;   // [K,1]
    torch::Tensor depthanything_created_rgba_accum_viz_;    // [K,4]
    torch::Tensor depthanything_fill_holes_created_centers_accum_viz_; // [K,3] accumulated real voxels created by depthanything hole fill
    torch::Tensor depthanything_fill_holes_created_sizes_accum_viz_;   // [K,1]
    torch::Tensor depthanything_fill_holes_created_rgba_accum_viz_;    // [K,4]
    torch::Tensor rendered_depth_created_centers_accum_viz_; // [K,3] accumulated artificial voxels from rendered depth insertion
    torch::Tensor rendered_depth_created_sizes_accum_viz_;   // [K,1]
    torch::Tensor rendered_depth_created_rgba_accum_viz_;    // [K,4]
    torch::Tensor rendered_hole_fill_created_centers_accum_viz_; // [K,3] accumulated artificial voxels from rendered hole fill
    torch::Tensor rendered_hole_fill_created_sizes_accum_viz_;   // [K,1]
    torch::Tensor rendered_hole_fill_created_rgba_accum_viz_;    // [K,4]
    torch::Tensor real_pcd_points_accum_cpu_;  // [K,3] accumulated real PCD points (CPU)
    torch::Tensor is_artificial_voxel_;          // [N] bool provenance: false=real, true=artificial/support
    torch::Tensor is_promoted_artificial_voxel_; // [N] bool provenance: true if voxel was artificial and got promoted to real
    torch::Tensor exist_since_iter_;                  // [N] int32 voxel creation iteration
    torch::Tensor exist_since_kf_;                    // [N] int32 voxel creation keyframe-count
    torch::Tensor geometrically_unstable_voxel_;      // [N] bool unstable real/art flags (updated at densification)
    torch::Tensor rendered_depth_candidate_voxel_;      // [N] bool artificial candidates inserted from rendered depth
    torch::Tensor rendered_depth_candidate_support_count_; // [N] int32 support count across keyframes
    torch::Tensor rendered_depth_candidate_last_seen_kf_; // [N] int32 last keyframe count that supported the candidate
    // [N] int32 source id for rendered-depth-origin provenance.
    // This is kept after promotion so live debug views can trace current voxels
    // back to their insertion path (depth-insert vs hole-fill).
    torch::Tensor rendered_depth_candidate_source_kind_;
    int64_t max_artificial_viz_accum_ = 300000;  // cap to keep rerun responsive
    int64_t max_inactive_geo_viz_accum_ = 300000;  // cap to keep rerun responsive
    int64_t max_rendered_depth_viz_accum_ = 300000;  // cap to keep rerun responsive
    py::object svm() const;

    torch::Tensor global_pcd_min_;   // [3], CPU or CUDA
    torch::Tensor global_pcd_max_;   // [3], CPU or CUDA
    torch::Tensor dense_core_bb_min_; // [3], fixed dense-core min from createFromPcd
    torch::Tensor dense_core_bb_max_; // [3], fixed dense-core max from createFromPcd
    
    bool has_global_pcd_bb_ = false;
    bool has_dense_core_bb_ = false;
    bool fill_empty_cells_done_ = false; // run dense bbox fill once per createFromPcd()
    bool fill_empty_cells_warmup_notified_ = false;
    bool enable_artificial_promotion_ = true;
    bool filter_near_voxels_ = true;
    bool filter_far_voxels_on_insert_ = true;
    bool rendered_depth_candidate_require_real_adjacency_ = true;
    int rendered_depth_candidate_adjacency_radius_cells_ = 1;
    std::string pending_real_insert_rr_entity_path_;
    std::string pending_artificial_insert_rr_entity_path_;
    bool pending_insert_rendered_depth_candidate_ = false;
    int pending_insert_rendered_depth_candidate_source_kind_ = 0;
    bool pending_insert_rendered_depth_candidate_as_real_protected_ = false;
    int64_t total_promoted_artificial_voxels_ = 0;
    int32_t topology_birth_iter_ = -1;
    int32_t topology_birth_kf_ = -1;
    bool consumeartificialFillFlag() {
        bool v = artificial_fill_happened_;
        artificial_fill_happened_ = false;
        return v;
    }
    bool artificial_fill_happened_ = false;  // default false

protected:
    /// The Adam optimizer
    // std::shared_ptr<torch::optim::Adam> optimizer_;
    // py::object optimizer_py_;
    struct PyState;                // forward declaration only
    std::unique_ptr<PyState> py_;  // holds all pybind objects
    // Pull all core tensors from the Python SVM after any topology change.
    void syncFromPython_();
    // Helper math
    static torch::Tensor camPosition_(const MiniCam& cam, torch::Device d);
    static torch::Tensor camForward_(const MiniCam& cam, torch::Device d);
    static float         camPixSize_(const MiniCam& cam); // max(1/fx, 1/fy)
    // Cache MAX_NUM_LEVELS
    const int max_num_levels_ = 16; // safe default; overwritten at init

    std::mutex mutex_settings_;
};

} // namespace sv
