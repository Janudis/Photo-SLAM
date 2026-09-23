#pragma once

#include <torch/torch.h>
#include <jsoncpp/json/json.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <filesystem>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <mutex>
#include <random>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <thread>
#include <limits>
#include <optional>
#include <regex>
#include <iomanip>
#include <array>
#include <atomic>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "include_voxel/voxel_parameters.h"
#include "include_voxel/voxel_keyframe.h"
#include "include_voxel/voxel_trainer.h" 
#include "include_voxel/mini_cam.h"
#include "include_voxel/voxel_camera.h"
#include "include_voxel/voxel_scene.h"
#include "include_voxel/voxel_mapper_utils.h"
#include "include_voxel/voxel_model.h"
#include "include_voxel/render_opts.h"  
#include "include_voxel/voxel_rerun_parameters.h"
#include "include_voxel/voxel_sdf_parameters.h"
#include "include_voxel/voxel_constants.h"
#include "include_voxel/rgbd_tsdf_evidence.h"
// ORB_SLAM3::System
#include "ORB-SLAM3/include/System.h"
#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

#include "rerun_utils.h" 

#define CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(dir)                                       \
    if (!dir.empty() && !std::filesystem::exists(dir))                                      \
        if (!std::filesystem::create_directories(dir))                                      \
            throw std::runtime_error("Cannot create result directory at " + dir.string());

struct UndistortParams
{
    UndistortParams(
        const cv::Size& old_size,
        cv::Mat dist_coeff = (cv::Mat_<float>(1, 4) << 0.0f, 0.0f, 0.0f, 0.0f))
        : old_size_(old_size)
    {
        dist_coeff.copyTo(dist_coeff_);
    }

    cv::Size old_size_;
    cv::Mat dist_coeff_;
};
 
enum SystemSensorType {
    INVALID = 0,
    MONOCULAR = 1,
    RGBD = 2
};

namespace sv {
class TandemMvsBackend;
struct TandemMvsResult;

struct MonocularMvsPruneEvidence
{
    torch::Tensor supported;
    torch::Tensor free_space;
};
}

 struct VariableParameters
 {
     float geo_lr;
     float sh0_lr;
     float shs_lr;
     float lambda_ssim;
     int densify_interval;
     int new_kf_times_of_use;
     int stable_num_iter_existence; ///< loop closure correction
     bool keep_training;
     bool do_gaus_pyramid_training;
     bool do_inactive_geo_densify;
 };

class VoxelMapper {
public:
    VoxelMapper(
        std::shared_ptr<ORB_SLAM3::System> pSLAM,
        const std::filesystem::path& config_file_path,
        std::filesystem::path result_dir = "",
        int seed = 0,
        torch::DeviceType device_type = torch::kCUDA);

    // ~VoxelMapper();
    void readConfigFromFile(const std::filesystem::path& cfg_path);

    void run();
    void trainForOneIteration();

    bool isStopped() const;
    void signalStop(const bool going_to_stop = true);

    cv::Mat renderFromPose(
        const Sophus::SE3f& Tcw, 
        const int width, 
        const int height, 
        const bool main_vision = false);
    bool getCurrentVoxelBounds(
        Eigen::Vector3f& minimum,
        Eigen::Vector3f& maximum) const;

    int  getIteration();
    void increaseIteration(const int inc=1);
    
    float geoLearningRateInit();
    float sh0LearningRate();
    float shsLearningRate();
    float lambdaSsim();
    int densifyInterval();
    int newKeyframeTimesOfUse();
    bool isKeepingTraining();
    int stableNumIterExistence();
    bool isdoingGausPyramidTraining();
    bool isdoingInactiveGeoDensify();

    void setKeepTraining(const bool keep);

    VariableParameters getVaribleParameters();
    void setVaribleParameters(const VariableParameters& params);

    void waitForInputQueueSlot();

protected:
    bool hasMetInitialMappingConditions();
    bool hasMetIncrementalMappingConditions();
    void combineMappingOperations();       
    void adaptVoxelTopology(int iteration);
    void pruneVoxelTopology(
        int iteration,
        int prune_every,
        const std::vector<sv::MiniCam>& cameras,
        sv::VoxelModel::StatPkg& statistics);
    void subdivideVoxelTopology(
        int iteration,
        const std::vector<sv::MiniCam>& cameras,
        sv::VoxelModel::StatPkg& statistics);
    void handleNewKeyframe(std::tuple<unsigned long,
                            unsigned long,
                            Sophus::SE3f,
                            cv::Mat,
                            bool,
                            cv::Mat,
                            std::vector<float>,
                            std::vector<float>,
                            std::string> &kf);

    void generateKfidRandomShuffle();
    std::shared_ptr<VoxelKeyframe> useOneRandomSlidingWindowKeyframe();
    void increaseKeyframeTimesOfUse(const std::shared_ptr<VoxelKeyframe>& kf, int n);
    void cullKeyframes();
    void increasePcdByKeyframeInactiveGeoDensify(
        std::shared_ptr<VoxelKeyframe> pkf,
        bool include_inactive_geo = true,
        bool include_rgbd_hole_fill = true);

    // densification methods -- fill the gaps
    void flushInactiveGeoCache();
    void processRgbdClosureCache();
    bool isMatureMonocularOrbMapPoint(
        ORB_SLAM3::MapPoint* map_point,
        unsigned long current_keyframe_id) const;
    void collectNewMonocularOrbSamplingSupport(
        std::vector<float>& points,
        std::vector<float>& colors);
    void densifyMonocularFromRenderedDepth(
        const std::shared_ptr<VoxelKeyframe>& pkf);
    void resetMonocularRenderedDepthEvidenceIfLayoutChanged();
    int64_t promoteMonocularRenderedDepthEvidenceCells(
        const std::unordered_set<
            sv::RgbdTsdfGridKey,
            sv::RgbdTsdfGridKeyHash>& affected_cells);
    void logMonocularRenderedDepthEvidenceCellsToRerun(
        int iteration,
        const std::vector<sv::RgbdTsdfGridKey>& cells,
        const std::string& entity_path);
    void captureMonocularMvsKeyframeMetadata(
        const std::shared_ptr<VoxelKeyframe>& pkf,
        ORB_SLAM3::KeyFrame* orb_keyframe);
    bool isMonocularMvsPipelineEnabled() const;
    void refreshMonocularMvsKeyframeMetadata();
    std::vector<std::shared_ptr<VoxelKeyframe>>
    selectMonocularMvsSourceKeyframes(
        const std::shared_ptr<VoxelKeyframe>& reference,
        int view_num = -1) const;
    bool scheduleMonocularMvsDensification(
        const std::shared_ptr<VoxelKeyframe>& reference);
    void scheduleLatestMonocularMvsKeyframe(
        const std::vector<std::shared_ptr<VoxelKeyframe>>& candidates);
    void pollMonocularMvsDensification(bool wait_for_result = false);
    void integrateMonocularMvsDepth(const sv::TandemMvsResult& result);
    void integrateMonocularMvsTsdfEvidence(
        const sv::TandemMvsResult& result);
    void promoteMonocularMvsTsdfEvidenceCells(
        const std::unordered_set<
            sv::RgbdTsdfGridKey,
            sv::RgbdTsdfGridKeyHash>& affected_cells);
    void resetMonocularMvsTsdfEvidenceIfLayoutChanged();
    void logMonocularMvsTsdfEvidenceCellsToRerun(
        int iteration,
        const std::vector<sv::RgbdTsdfGridKey>& cells,
        const std::string& entity_path);
    void logMonocularMvsTsdfEvidenceSnapshotToRerun(int iteration);
    void cacheMonocularDepthPrior(
        const std::shared_ptr<VoxelKeyframe>& reference,
        const cv::Mat& depth,
        const cv::Mat& confidence,
        sv::LearnedDepthSource source);
    sv::MonocularMvsPruneEvidence computeMonocularMvsPruneEvidence(
        const torch::Tensor& centers_world,
        const torch::Tensor& sizes_world);
    torch::Tensor detectRgbdRenderHolePixels(
        const std::shared_ptr<VoxelKeyframe>& pkf,
        const torch::Tensor& depth,
        int pixel_stride,
        bool render_on_stride_grid,
        int64_t& valid_depth_pixels,
        int64_t& hole_pixels,
        torch::Tensor& full_hole_mask);
    void fillRgbdRenderHolesSdf(
        const std::shared_ptr<VoxelKeyframe>& pkf);
    void integrateRgbdTsdfEvidenceForRenderHoles(
        const std::shared_ptr<VoxelKeyframe>& pkf,
        std::unordered_set<
            sv::RgbdTsdfGridKey,
            sv::RgbdTsdfGridKeyHash>& affected_cells);
    void promoteRgbdTsdfEvidenceCells(
        const std::unordered_set<
            sv::RgbdTsdfGridKey,
            sv::RgbdTsdfGridKeyHash>& affected_cells);
    void resetRgbdTsdfEvidenceIfLayoutChanged();
    void logRgbdTsdfEvidenceCellsToRerun(
        int iteration,
        const std::vector<sv::RgbdTsdfGridKey>& cells,
        const std::string& entity_path);
    std::vector<sv::MiniCam> incrementalMappingCameras() const;
    std::vector<sv::MiniCam> surfaceViewPruningCameras() const;
    void markSurfaceViewPruningPending(
        const std::vector<std::shared_ptr<VoxelKeyframe>>& keyframes);
    bool surfaceViewPruningReady();
    void runPendingSurfaceViewPruning();

    void savePly(std::filesystem::path result_dir);         
    void keyframesToJson(const std::filesystem::path& dir);   

    void saveRenderedTsdfMeshPly(const std::filesystem::path& result_path);
    void saveSvreconSdfMeshPly(const std::filesystem::path& result_path);
    void saveSvreconRenderedTsdfMeshPly(const std::filesystem::path& result_path);
    torch::Tensor colorizeSvreconMeshVertices(const torch::Tensor& vertices);

    // SVRecon RGB-D SDF initialization ----------------------------------------
    float sdfMetricVoxelSize() const;
    bool prepareProjectiveSdfInitContext(const std::shared_ptr<VoxelKeyframe>& kf);
    void clearProjectiveSdfInitContext();
    torch::Tensor computeProjectiveSdfInitForGridPoints(
        const torch::Tensor& grid_points_world,
        float ray_interval_m);
    int64_t fuseProjectiveSdfInitFromKeyframe(
        const std::shared_ptr<VoxelKeyframe>& kf,
        const torch::Tensor& pixel_mask = torch::Tensor());

    struct FinalSurfacePruneStats {
        int64_t rendered_view_count = 0;
        int64_t candidate_grid_point_count = 0;
        int64_t candidate_grid_cell_count = 0;
        int64_t observed_grid_cell_count = 0;
        int64_t surface_grid_cell_count = 0;
        int64_t supported_voxel_count = 0;
        int64_t surface_prune_count = 0;
        float voxel_size = 0.0f;
        float truncation = 0.0f;
        float min_keyframe_weight = 0.0f;
        float alpha_threshold = 0.0f;
        bool rendered_tsdf_available = false;
    };

    void runFinalRefinement();
    torch::Tensor computeOnlineCovisibilityPruneMask(
        const std::vector<sv::MiniCam>& cameras,
        const torch::Tensor& view_count);
    torch::Tensor computeRenderedTsdfSurfacePruneMask(
        FinalSurfacePruneStats* stats_out = nullptr);
    torch::Tensor computeSvreconSdfPruneMask(float* sdf_threshold_out = nullptr);

    void ensureEmbeddedPythonRuntime(bool import_torch_cuda = false);

    // rerun debugging ---------------------------------------------------------
    void logKeyframeCameraToRerunRecordings(
        const std::shared_ptr<VoxelKeyframe>& pkf,
        unsigned long kf_id,
        bool log_reconstruction_mesh);
    void saveRerunRecordingsAtShutdown();
    void logLearnedDepthMapsToWholeRunRerun();
    void logCurrentOrbMapPointsToReconstructionRerun(int iteration);
    void logCurrentOrbKeyframePosesToReconstructionRerun(int iteration);
    void logWholeRunLiveVoxelsToRerun(
        int iteration,
        const torch::Tensor& centers,
        const torch::Tensor& sizes,
        const torch::Tensor& colors,
        bool log_whole_run = true,
        bool log_svrecon_debug = true,
        bool log_monocular_debug = true);
    void appendWholeRunPrunedVoxels(
        int iteration,
        const torch::Tensor& centers,
        const torch::Tensor& sizes,
        const torch::Tensor& levels,
        const torch::Tensor& colors,
        const torch::Tensor& pruned_by_sdf,
        const torch::Tensor& pruned_by_surface_views,
        const torch::Tensor& pruned_by_near_camera = torch::Tensor(),
        const torch::Tensor& pruned_by_far = torch::Tensor(),
        const torch::Tensor& pruned_by_mvs_free_space = torch::Tensor(),
        const torch::Tensor& pruned_by_final_refinement = torch::Tensor());
    void logSvreconDebugVoxelMaskToRerun(
        int iteration,
        const torch::Tensor& mask,
        const std::string& entity_path);
    void logReconstructionMeshToRerun(int iteration);

    // Rerun debugging ---------------------------------------------------------
    bool buildSparseDepthFromMapPoints(
        const sv::MiniCam& cam,
        int image_width,
        int image_height,
        torch::Tensor& sparse_uv,     // [N,2]
        torch::Tensor& sparse_depth); // [N]
    torch::Tensor computeSparseDepthLoss_Points(
        const std::shared_ptr<VoxelKeyframe>& kf,
        const sv::MiniCam& cam,
        int image_width,
        int image_height,
        const std::unordered_map<std::string, torch::Tensor>& render_pkg,
        int iteration);
    torch::Tensor computeRgbdDepthLoss(
        const std::shared_ptr<VoxelKeyframe>& kf,
        const sv::MiniCam& cam,
        const std::unordered_map<std::string, torch::Tensor>& render_pkg,
        int iteration);
    torch::Tensor computeRgbdMaskLoss(
        const std::shared_ptr<VoxelKeyframe>& kf,
        const sv::MiniCam& cam,
        const std::unordered_map<std::string, torch::Tensor>& render_pkg);
    torch::Tensor computeRgbdSdfLoss(
        const std::shared_ptr<VoxelKeyframe>& kf,
        const sv::MiniCam& cam,
        int iteration);
    torch::Tensor computeRgbdNormalLoss(
        const std::shared_ptr<VoxelKeyframe>& kf,
        const sv::MiniCam& cam,
        const std::unordered_map<std::string, torch::Tensor>& render_pkg,
        int iteration);
    torch::Tensor computeMonocularDepthLoss(
        const std::shared_ptr<VoxelKeyframe>& kf,
        const std::unordered_map<std::string, torch::Tensor>& render_pkg,
        int iteration);
    torch::Tensor computeMonocularNormalLoss(
        const std::shared_ptr<VoxelKeyframe>& kf,
        const std::unordered_map<std::string, torch::Tensor>& render_pkg,
        int iteration);

public:
    // Parameters
    std::filesystem::path config_file_path_;

    // Model
    std::shared_ptr<sv::VoxelModel> voxel_model_; 
    std::shared_ptr<sv::VoxelScene> scene_;

    // SLAM system
    std::shared_ptr<ORB_SLAM3::System>  mpSLAM;

    // Settings
    torch::DeviceType device_type_;
    torch::Device mDevice = torch::Device(torch::kCPU); 
    int num_gaus_pyramid_sub_levels_ = 0;
    std::vector<int> kf_gaus_pyramid_times_of_use_;
    std::vector<float> kf_gaus_pyramid_factors_;

    bool viewer_camera_id_set_ = false;
    std::uint32_t viewer_camera_id_ = 0;
    float rendered_image_viewer_scale_ = 1.0f;
    float rendered_image_viewer_scale_main_ = 1.0f;

    float z_near_ = 0.01f;
    float z_far_ = 100.0f;

    // Data
    bool kfid_shuffled_  = false;
    std::map<sv::camera_id_t, torch::Tensor> undistort_mask_;
    std::map<sv::camera_id_t, torch::Tensor> viewer_main_undistort_mask_;
    std::map<sv::camera_id_t, torch::Tensor> viewer_sub_undistort_mask_;

protected:
    // Parameters
    VoxelModelParams model_params_;
    VoxelOptimizationParams opt_params_;
    VoxelPipelineParams pipe_params_;    

    VoxelSdfParameters sdf_params_;
    // Feature state
    VoxelSdfState sdf_state_;
    // Rerun debugging
    VoxelRerunParameters rerun_params_;
    VoxelRerunState rerun_state_;
    std::unordered_map<unsigned long, Eigen::Matrix4f>
        rerun_reconstruction_last_orb_poses_;

    // Data
    std::vector<std::size_t> kfid_shuffle_;
    std::size_t kfid_shuffle_idx_ = 0;
    std::map<std::size_t,int> kfs_used_times_;

    // Status
    bool initial_mapped_;
    bool interrupt_training_;
    bool stopped_;
    int iteration_;
    float ema_loss_for_log_;
    bool SLAM_ended_;
    bool loop_closure_iteration_;
    bool keep_training_ = false;
    int default_sh_ = 0;

    bool disable_topology_changes_ = false;
    bool tail_refinement_active_ = false;
    bool surface_view_pruning_initialized_ = false;
    std::unordered_set<std::size_t> surface_view_pending_keyframes_;
    int input_queue_max_keyframes_ = 0;
    int incremental_mapping_window_size_ = 0;
    bool loop_closure_reinsert_points_ = true;
    std::atomic<long long> latest_consumed_keyframe_id_{-1};
    std::atomic<bool> input_backpressure_ready_{false};
    int svrecon_outside_level_ = 5;
    float global_scene_extent_m_ = 0.0f;
    bool robust_scene_bounds_ = false;
    bool sdf_initialization_rgbd_projective_ = false;
    bool allocate_orb_voxels_ = true;
    std::string sdf_initialization_mode_ = "orb_prior";

    // Settings
    SystemSensorType sensor_type_;

    float RGBD_min_depth_ = 0.0f;
    float RGBD_max_depth_ = 100.0f;

    bool inactive_geo_densify_ = true;
    bool keyframe_images_on_cpu_ = false;
    int depth_cached_ = 0;
    int max_depth_cached_ = 1;
    torch::Tensor depth_cache_points_;
    torch::Tensor depth_cache_colors_;

    // monocular densification
    std::unordered_map<
        sv::RgbdTsdfGridKey,
        sv::RgbdTsdfCornerEvidence,
        sv::RgbdTsdfGridKeyHash>
        monocular_rendered_depth_corner_evidence_;
    std::unordered_map<
        sv::RgbdTsdfGridKey,
        sv::RgbdTsdfCellEvidence,
        sv::RgbdTsdfGridKeyHash>
        monocular_rendered_depth_cell_evidence_;
    Eigen::Vector3f monocular_rendered_depth_layout_scene_min_ =
        Eigen::Vector3f::Zero();
    float monocular_rendered_depth_layout_cell_size_ = 0.0f;
    int monocular_rendered_depth_layout_grid_dim_ = 0;
    std::filesystem::path monocular_mvs_model_dir_;
    float monocular_mvs_depth_min_m_ = 0.1f;
    float monocular_mvs_depth_max_m_ = 5.0f;
    bool monocular_mvs_empty_cache_before_launch_ = false;
    // Optional full-image MVS evidence. This remains separate from active
    // SVRecon topology until a fused TSDF zero crossing is promoted.
    bool monocular_mvs_tsdf_evidence_ = false;
    std::unordered_map<
        sv::RgbdTsdfGridKey,
        sv::RgbdTsdfCornerEvidence,
        sv::RgbdTsdfGridKeyHash> monocular_mvs_tsdf_corner_evidence_;
    std::unordered_map<
        sv::RgbdTsdfGridKey,
        sv::RgbdTsdfCellEvidence,
        sv::RgbdTsdfGridKeyHash> monocular_mvs_tsdf_cell_evidence_;
    Eigen::Vector3f monocular_mvs_tsdf_layout_scene_min_ =
        Eigen::Vector3f::Zero();
    float monocular_mvs_tsdf_layout_cell_size_ = 0.0f;
    int monocular_mvs_tsdf_layout_grid_dim_ = 0;
    bool monocular_mvs_requires_inertial_ba1_ = false;
    std::shared_ptr<sv::TandemMvsBackend> monocular_mvs_backend_;
    std::shared_ptr<VoxelKeyframe> monocular_mvs_pending_reference_;
    sv::MiniCam monocular_mvs_pending_camera_;
    Eigen::Matrix4f monocular_mvs_pending_c2w_ =
        Eigen::Matrix4f::Identity();
    std::vector<unsigned long> monocular_mvs_pending_view_ids_;
    std::vector<Eigen::Matrix4f> monocular_mvs_pending_view_c2w_;
    float monocular_mvs_pending_depth_min_ = 0.0f;
    float monocular_mvs_pending_depth_max_ = 0.0f;
    cv::Mat monocular_mvs_pending_reference_rgb_;
    std::unordered_set<unsigned long> monocular_mvs_scheduled_keyframes_;
    bool rgbd_fill_render_holes_initial_backfill_ = true;
    torch::Tensor rgbd_fill_render_holes_cache_points_;
    torch::Tensor rgbd_fill_render_holes_cache_colors_;
    std::vector<std::pair<std::shared_ptr<VoxelKeyframe>, torch::Tensor>>
        rgbd_fill_render_holes_projective_cache_;
    std::vector<std::shared_ptr<VoxelKeyframe>> rgbd_hole_fill_keyframe_cache_;
    std::vector<std::shared_ptr<VoxelKeyframe>> rgbd_hole_fill_ready_keyframes_;

    // Direct RGB-D render-hole completion.
    bool rgbd_fill_render_holes_ = false;
    bool rgbd_fill_render_holes_projective_sdf_ = false;

    // Non-renderable RGB-D TSDF evidence used only to promote confirmed
    // residual-hole cells into the active SVRecon octree.
    bool rgbd_tsdf_evidence_ = false;
    bool rgbd_tsdf_evidence_initial_backfill_ = true;
    std::unordered_map<
        sv::RgbdTsdfGridKey,
        sv::RgbdTsdfCornerEvidence,
        sv::RgbdTsdfGridKeyHash> rgbd_tsdf_corner_evidence_;
    std::unordered_map<
        sv::RgbdTsdfGridKey,
        sv::RgbdTsdfCellEvidence,
        sv::RgbdTsdfGridKeyHash> rgbd_tsdf_cell_evidence_;
    Eigen::Vector3f rgbd_tsdf_layout_scene_min_ = Eigen::Vector3f::Zero();
    float rgbd_tsdf_layout_cell_size_ = 0.0f;
    int rgbd_tsdf_layout_grid_dim_ = 0;

    // Orbeez-style sparse support: only live ORB MapPoints that have survived
    // ORB's recent-point culling period seed weak SVRecon candidates.
    std::unordered_set<sv::point3D_id_t> monocular_orb_inserted_point_ids_;

    unsigned long min_num_initial_map_kfs_;
    torch::Tensor background_;
    float large_rot_th_;
    float large_trans_th_;
    torch::Tensor override_color_;

    int new_keyframe_times_of_use_;
    int local_BA_increased_times_of_use_;
    int loop_closure_increased_times_of_use_;
    bool cull_keyframes_;
    int stable_num_iter_existence_;

    bool do_gaus_pyramid_training_;

    std::filesystem::path result_dir_;
    int training_report_interval_;   
    bool record_loop_ply_;

    // Topology update schedule
    int next_subdiv_iter_   = 0;
    int next_prune_iter_    = 0;
    int next_opacity_reset_ = 0;
    int64_t last_densify_iter_ = -1;

    // Tools
    std::random_device rd_;
    
    // Mutex
    mutable std::mutex mutex_status_;
    mutable std::mutex mutex_render_;
    mutable std::mutex mutex_settings_;
};
