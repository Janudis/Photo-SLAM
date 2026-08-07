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
#include <torch/extension.h>
#include <iostream>
#include <cstdlib>
#include <ATen/ATen.h>
#include <cmath>
#include <tuple>
#include <optional>
#include <array>
#include <cstdint>
#include <unordered_map>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"
#include "third_party/simple-knn/spatial.h"
#include "third_party/tinyply/tinyply.h"
#include "include_voxel/voxel_parameters.h"
#include "include_voxel/voxel_scene.h"
#include "include_voxel/voxel_keyframe.h"
#include "include_voxel/mini_cam.h"
#include "include_voxel/voxel_camera.h"
#include "include_voxel/voxel_constants.h"
#include "include_voxel/voxel_types.h"
#include "include_voxel/render_opts.h" 
#include "include_voxel/voxel_svrecon_rasterizer.h"

#define VOXEL_MODEL_TENSORS_TO_VEC                       \
    this->Tensor_vec_geo_       = { this->_geo_grid_pts_ };       \
    this->Tensor_vec_sh0_       = { this->sh0_ };       \
    this->Tensor_vec_shs_       = { this->shs_ };       \

#define VOXEL_MODEL_INIT_TENSORS(device_type)                                              \
    this->_geo_grid_pts_= torch::empty({0, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->log_s_        = torch::full({1}, 0.3f, torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->sh0_          = torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->shs_          = torch::empty({0, 0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->center_       = torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->size_         = torch::empty({0},    torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->oct_path_     = torch::empty({0, 1}, torch::TensorOptions().dtype(torch::kLong   ).device(device_type)); \
    this->oct_level_    = torch::empty({0, 1}, torch::TensorOptions().dtype(torch::kInt8   ).device(device_type)); \
    this->is_leaf_      = torch::empty({0, 1}, torch::TensorOptions().dtype(torch::kBool   ).device(device_type)); \
    this->subdiv_p_     = torch::empty({0, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->subdiv_meta_  = torch::empty({0, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    this->max_w_        = torch::empty({0, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type)); \
    VOXEL_MODEL_TENSORS_TO_VEC

namespace sv {

class VoxelModel 
{
public:
    VoxelModel(const int sh_degree);
    VoxelModel(const VoxelModelParams& model_params);
    ~VoxelModel();  

    const torch::Tensor& geoGridPts() const;
    const torch::Tensor& svreconLogS() const { return log_s_; }
    void refreshSvreconLogSTargetFromVoxelSize(bool initialize_current);
    const torch::Tensor& sh0() const;
    const torch::Tensor& shs() const;

    void oneUpShDegree();
    void setShDegree(int sh);

    void createFromPcd(
        const std::map<point3D_id_t, Point3D>& pcd,
        const std::vector<sv::MiniCam>& cams);
    void increasePcd(
        std::vector<float> points,
        std::vector<float> colors,
        const int iteration,
        const std::vector<sv::MiniCam>& cams,
        bool clear_cuda_cache_before_parameter_append = false);
    void increasePcd(
        torch::Tensor& new_point_cloud,
        torch::Tensor& new_colors,
        const int iteration,
        const std::vector<sv::MiniCam>& cams,
        bool clear_cuda_cache_before_parameter_append = false);

    torch::Tensor voxCenter() { return this->center_; }
    int64_t numGridPts() const;
    torch::Tensor voxelDensityMean() const;
    torch::Tensor voxelGeoCorners() const;
    torch::Tensor voxelSdfWeightCorners() const;
    torch::Tensor gridPtsKey() const { return this->grid_pts_key_; }
    torch::Tensor voxKey() const { return this->vox_key_; }
    torch::Tensor gridPointsWorld() const;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    buildSvreconDenseExtractionGrid(int inside_level) const;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    subdivideSvreconExtractionGrid(
        const torch::Tensor& octpath,
        const torch::Tensor& octlevel) const;
    std::tuple<torch::Tensor, torch::Tensor>
    querySdfTrilinear(const torch::Tensor& points_world) const;
    // Read helpers (mirrors SVM tensors)
    torch::Tensor subdivisionPriority() const; // [N,1]
    torch::Tensor voxSize() const;             // [N,1]
    torch::Tensor octLevel() const;            // [N,1] int8
    torch::Tensor octPath() const;             // [N,1] int64
    torch::Tensor isLeaf() const { return is_leaf_; }
    int           numVoxels() const;           // N
    int           maxNumLevels() const;        // matches SVRecon CUDA MAX_NUM_LEVELS
    torch::Tensor SceneCenter() const;  
    torch::Tensor SceneExtent() const;
    torch::Tensor InsideExtent() const;
    float         fixedVoxSize() const { return fixed_vox_size_; }
    float         insertionVoxSize() const;
    int           insertionOctreeLevel() const { return static_cast<int>(octlevel_); }
    
    // Fused projective SDF helpers
    const torch::Tensor& fusedSdfGridPts() const;
    const torch::Tensor& fusedSdfWeights() const;

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
                       float gamma = 0.1f,
                       float log_s_lr = 0.0f);
    struct SchedulerState {
        bool valid = false;
        int64_t last_epoch = -1;
        float geo_lr = 0.0f;
        float sh0_lr = 0.0f;
        float shs_lr = 0.0f;
        float log_s_lr = 0.0f;
    };
    SchedulerState schedulerState() const;
    void schedulerStep();
    void schedulerLoadState(const SchedulerState& state);
    std::tuple<double,double,double> currentLearningRates() const;

    // === Adaptive octree API ===
    struct StatPkg {
        torch::Tensor max_w;             // [N,1], float32, cuda
        torch::Tensor min_samp_interval; // [N,1], float32, cuda
        torch::Tensor view_cnt;          // [N,1], float32, cuda
    };
    // Compute per-voxel stats over a list of training cameras.
    StatPkg computeTrainingStat(const std::vector<MiniCam>& cams);
    // Topology changes.
    void pruning(const torch::Tensor& prune_mask);         // mask: [N] or [N,1] bool/byte
    void subdividing(const torch::Tensor& subdivide_mask); // mask: [N] or [N,1] bool/byte
    // Maintenance
    void accumulateSubdivisionPriority();
    void resetSubdivisionPriority();
    void freezeVoxGeo();
    void unfreezeVoxGeo();

    /// Load all voxel fields from a PLY file (structure matches savePly).
    void loadPly(const std::filesystem::path& ply_file);
    /// Save all voxel data to a PLY file with full attributes.
    void savePly(const std::filesystem::path& result_path);

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
    void applySvreconGridEikonalField(float lambda_ge_density);
    void applySvreconLaplacianSmoothnessField(float lambda_ls_density);
    torch::Tensor svreconLocalEikonalLoss(float lambda_local_ge_density,
                                          int min_inside_level) const;

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
    bool hasFusedSdfField() const;
    void ensureFusedSdfField();
    void resetFusedSdfField();
    void fuseProjectiveSdfGridSamples(
        const torch::Tensor& tsdf_values,
        const torch::Tensor& weights,
        const torch::Tensor& valid_mask,
        float max_weight = -1.0f);
    void applyGeoGridRawInit(
        const torch::Tensor& raw_values,
        const torch::Tensor& valid_mask);
    std::pair<torch::Tensor, torch::Tensor> rgbdHoleSupportCellCenters(
        const torch::Tensor& surface_points_world) const;
    torch::Tensor makeGeoGridInitRows_(
        const torch::Tensor& grid_pts_key_new,
        int64_t begin,
        int64_t end,
        float default_value);
    torch::Tensor makePointPriorSdfInitRowsForKeys_(
        const torch::Tensor& grid_pts_key_rows,
        float fallback_value);
    torch::Tensor applyPendingSdfGridInitialization_(
        const torch::Tensor& grid_pts_key_rows,
        const torch::Tensor& initial_values) const;
    void rebuildGeoGridForNewGridKeys_(
        const torch::Tensor& grid_pts_key_new,
        float default_value);
    
    torch::Tensor orbVoxelMask() const { return this->is_orb_voxel_; } // [N] bool, true=originated from ORB map points
    torch::Tensor inactiveGeoVoxelMask() const { return this->is_inactive_geo_voxel_; } // [N] bool, true=originated from inactive-geo RGB-D gap fill
    torch::Tensor rgbdFillRenderHolesVoxelMask() const { return this->is_rgbd_fill_render_holes_voxel_; } // [N] bool
    torch::Tensor monocularRenderedDepthVoxelMask() const {
        return this->is_monocular_rendered_depth_voxel_;
    }
    torch::Tensor monocularMvsVoxelMask() const {
        return this->is_monocular_mvs_voxel_;
    }
    torch::Tensor monocularOmnidataVoxelMask() const {
        return this->is_monocular_omnidata_voxel_;
    }
    torch::Tensor monocularProvisionalVoxelMask() const {
        return this->is_monocular_provisional_voxel_;
    }
    torch::Tensor monocularRenderSupportHits() const {
        return this->monocular_render_support_hits_;
    }
    torch::Tensor monocularRenderOpportunities() const {
        return this->monocular_render_opportunities_;
    }
    void accumulateMonocularRenderObservation(
        const torch::Tensor& max_render_weight,
        float min_render_weight);
    void resolveMonocularProvisionalVoxels(
        const torch::Tensor& resolved_mask);
    torch::Tensor existSinceIter() const { return this->exist_since_iter_; } // [N] int32, voxel creation iter
    torch::Tensor existSinceKf() const { return this->exist_since_kf_; } // [N] int32, voxel creation keyframe-count
    torch::Tensor activeRenderableMask() const;
    
    bool hasDenseCoreBB() const { return has_dense_core_bb_; }
    torch::Tensor denseCoreBBMin() const { return dense_core_bb_min_; } // [3]
    torch::Tensor denseCoreBBMax() const { return dense_core_bb_max_; } // [3]
    bool refreshDenseCoreBBFromCurrentVoxels();
    void setFilterNearVoxels(const bool enable) { filter_near_voxels_ = enable; }
    void setOutsideLevel(const int level) {
        outside_level_ = std::clamp(level, 0, max_num_levels_);
    }
    void setFixedGlobalSceneExtent(const float extent_m) {
        configured_global_scene_extent_ = std::max(0.0f, extent_m);
        fixed_global_scene_layout_ = configured_global_scene_extent_ > 0.0f;
    }
    void setFixedVoxSize(const float voxel_size_m) {
        fixed_vox_size_ = std::max(1.0e-4f, voxel_size_m);
    }
    void setRobustSceneBounds(const bool enable) {
        robust_scene_bounds_ = enable;
    }
    bool hasFixedGlobalSceneLayout() const { return fixed_global_scene_layout_; }
    void setSdfInitializationOrbRadiusVox(const float radius_vox) {
        sdf_initialization_orb_radius_vox_ = std::max(0.0f, radius_vox);
    }
    void setTopologySdfInitializationMode(const std::string& mode);
    void setNextSdfInitializationGridSamples(
        const torch::Tensor& grid_points_world,
        const torch::Tensor& sdf_values);
    int outsideLevel() const { return outside_level_; }
    void setNextRealInsertionRerunEntityPath(const std::string& entity_path) {
        pending_real_insert_rr_entity_path_ = entity_path;
    }
    IncreasePcdStats lastIncreasePcdStats() const { return last_increase_pcd_stats_; }
    void setTopologyBirthContext(const int iteration, const int kf_count) {
        topology_birth_iter_ = static_cast<int32_t>(iteration);
        topology_birth_kf_ = static_cast<int32_t>(kf_count);
    }
private:
    IncreasePcdStats last_increase_pcd_stats_;
    void appendGroup_(int group_idx, const torch::Tensor& add_rows, torch::Tensor* out_member_param);
    void setEmptyFusedSdfField_();
    torch::Tensor voxelCornerScalarFromGrid_(const torch::Tensor& grid_scalar) const;
    void rebuildFusedSdfFieldFromVoxelCorners_(
        const torch::Tensor& voxel_sdf_values,
        const torch::Tensor& voxel_sdf_weights);
    void appendSparseSupportPoints_(const torch::Tensor& points);
    void updateExistingSupportSdfFromPoints_(
        const torch::Tensor& support_xyz,
        const torch::Tensor& support_rgb,
        const std::vector<sv::MiniCam>& cams);
    torch::Tensor visibilitySignedPointPriorSdf_(
        const torch::Tensor& grid_xyz,
        const torch::Tensor& dist,
        const torch::Tensor& points,
        float surface_band) const;
    // Helper math
    static torch::Tensor camPosition_(const MiniCam& cam, torch::Device d);
    static torch::Tensor camForward_(const MiniCam& cam, torch::Device d);
    static float         camPixSize_(const MiniCam& cam); // max(1/fx, 1/fy) 

public:
    torch::DeviceType device_type_;

    /// Current & maximum SH degree
    int active_sh_degree_;
    int max_sh_degree_;

    /// Main voxel tensors
    torch::Tensor geo_;                // [N, 8] (covariance and pad)
    torch::Tensor sh0_;                // [N, 3]
    torch::Tensor shs_;                // [N, 45, 3]
    torch::Tensor center_;             // [N, 3]
    torch::Tensor size_;               // [N]
    torch::Tensor oct_path_;           // [N]
    torch::Tensor oct_level_;          // [N]
    torch::Tensor is_leaf_;             // [N,1] bool; retained fine-level parents are non-leaf
    torch::Tensor subdiv_p_;           // [N]
    torch::Tensor subdiv_meta_;        // [N]
    torch::Tensor max_w_;              // [N]
    torch::Tensor grid_pts_key_;    // [M]  int64 keys identifying each grid-point
    torch::Tensor _geo_grid_pts_;    // [M]  float32 learnable density at each grid-point
    torch::Tensor log_s_;          // [1] SVRecon SDF sharpness parameter
    torch::Tensor vox_key_;         // [N,8] int64 indices into grid_pts_key_
    torch::Tensor vox_size_inv_;    // [N] float32 = 1.0 / size_
    torch::Tensor frozen_vox_geo_;
    
    torch::Tensor fused_sdf_grid_pts_;    // [M,1] float32 metric signed distance at each grid point
    torch::Tensor fused_sdf_weights_;     // [M,1] float32 accumulated projective TSDF weight

    std::vector<torch::Tensor>  Tensor_vec_geo_,
                                Tensor_vec_sh0_,
                                Tensor_vec_shs_;

    torch::Tensor sparse_points_xyz_;
    torch::Tensor sparse_points_color_;
    torch::Tensor sdf_init_local_support_points_;
    torch::Tensor next_sdf_init_grid_keys_;
    torch::Tensor next_sdf_init_grid_values_;
    std::vector<MiniCam> sdf_init_cams_;
    enum class SdfInitMode {
        SignedPointPrior,
        WeakSurfacePrior,
        OrbPriorOnly,
        WeakPositive,
        OrbPriorWeakCandidate
    };
    SdfInitMode pending_sdf_init_mode_ = SdfInitMode::SignedPointPrior;
    SdfInitMode topology_sdf_init_mode_ = SdfInitMode::OrbPriorOnly;
    float sdf_initialization_orb_radius_vox_ = 2.0f;
    float positive_unknown_sdf_init_vox_ = 1.5f;

    torch::Tensor scene_center_;   // [3], CUDA
    torch::Tensor scene_extent_;   // [1], CUDA
    torch::Tensor scene_min_t_;      // [3], CUDA
    torch::Tensor vox_eff_;      // [1,1], CUDA (effective voxel size for fixed level)
    int8_t  octlevel_ = 0; 
    int outside_level_ = 5;
    torch::Tensor inside_extent_; // [1], CUDA

    float global_scene_extent_ = 200.0f;  // example: 100 m cube
    std::array<float,3> global_scene_center_{0.f, 0.f, 0.f};
    float fixed_vox_size_ = 0.05f;     
    bool fixed_global_scene_layout_ = false;
    float configured_global_scene_extent_ = 0.0f;
    bool robust_scene_bounds_ = false;

    float dense_core_pcd_density_rate_ = 0.005f;
    torch::Tensor real_pcd_points_accum_cpu_;  // [K,3] accumulated real PCD points (CPU)
    torch::Tensor is_orb_voxel_;                 // [N] bool provenance: true=created from ORB map points
    torch::Tensor is_inactive_geo_voxel_;        // [N] bool provenance: true=created by inactive-geo densification
    torch::Tensor is_rgbd_fill_render_holes_voxel_; // [N] bool provenance: true=created by RGB-D render-hole fill
    torch::Tensor is_monocular_rendered_depth_voxel_; // [N] bool provenance: rendered-depth densification
    torch::Tensor is_monocular_mvs_voxel_; // [N] bool provenance: TANDEM MVS hole filling
    torch::Tensor is_monocular_omnidata_voxel_; // [N] bool provenance: aligned Omnidata hole filling
    torch::Tensor is_monocular_provisional_voxel_; // [N] bool pending co-visibility validation
    torch::Tensor monocular_render_support_hits_; // [N] int32 keyframes with max(T*alpha) above threshold
    torch::Tensor monocular_render_opportunities_; // [N] int32 keyframes tested since insertion
    torch::Tensor exist_since_iter_;                  // [N] int32 voxel creation iteration
    torch::Tensor exist_since_kf_;                    // [N] int32 voxel creation keyframe-count
    torch::Tensor global_pcd_min_;   // [3], CPU or CUDA
    torch::Tensor global_pcd_max_;   // [3], CPU or CUDA
    torch::Tensor dense_core_bb_min_; // [3]
    torch::Tensor dense_core_bb_max_; // [3]

    bool has_global_pcd_bb_ = false;
    bool has_dense_core_bb_ = false;
    bool filter_near_voxels_ = true;
    std::string pending_real_insert_rr_entity_path_;
    int32_t topology_birth_iter_ = -1;
    int32_t topology_birth_kf_ = -1;

protected:
    float ss_ = 1.5f;
    bool  white_background_ = false;
    bool  black_background_ = false;
    int   n_samp_per_vox_ = 3;
    /// The Adam optimizer
    struct AdamGroupState {
        torch::Tensor exp_avg;
        torch::Tensor exp_avg_sq;
        int64_t step = 0;
    };
    AdamGroupState adam_geo_;
    AdamGroupState adam_sh0_;
    AdamGroupState adam_shs_;
    AdamGroupState adam_log_s_;
    float optimizer_geo_lr_ = 0.0f;
    float optimizer_sh0_lr_ = 0.0f;
    float optimizer_shs_lr_ = 0.0f;
    float optimizer_log_s_lr_ = 0.0f;
    float optimizer_beta1_ = 0.9f;
    float optimizer_beta2_ = 0.999f;
    float optimizer_eps_ = 1e-15f;
    std::vector<int> scheduler_milestones_;
    float scheduler_gamma_ = 0.1f;
    int64_t scheduler_epoch_ = -1;
    bool optimizer_initialized_ = false;
    std::mutex mutex_settings_;
    // Cache MAX_NUM_LEVELS
    const int max_num_levels_ = 16; // safe default; overwritten at init  
};

} // namespace sv
