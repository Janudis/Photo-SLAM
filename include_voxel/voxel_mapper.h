#pragma once

#include <torch/torch.h>
#include <jsoncpp/json/json.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
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
#include "include_voxel/voxel_constants.h"
#include "include/loss_utils.h"
#include "include_voxel/voxel_scene.h"
#include "include/tensor_utils.h"
#include "include_voxel/voxel_model.h"
#include "include_voxel/render_opts.h"  
#include "include_voxel/voxel_rerun_parameters.h"
#include "include_voxel/voxel_sdf_parameters.h"
#include "include_voxel/rgbd_tsdf_evidence.h"
// #include "include/stereo_vision.h"
// #include "include/operate_points.h"

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
    STEREO = 2,
    RGBD = 3
};

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

    int  getIteration();
    void increaseIteration(const int inc=1);
    void waitForInputQueueSlot();
    
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

    void setgeoLearningRateInit(const float lr);
    void setsh0LearningRate(const float lr);
    void setshsLearningRate(const float lr);
    void setLambdaSsim(const float lambda_ssim);
    void setDensifyInterval(const int interval);
    void setNewKeyframeTimesOfUse(const int times);
    void setKeepTraining(const bool keep);
    void setStableNumIterExistence(const int niter);
    void setDoGausPyramidTraining(const bool gaus_pyramid);

    VariableParameters getVaribleParameters();
    void setVaribleParameters(const VariableParameters& params);

protected:
    bool hasMetInitialMappingConditions();
    bool hasMetIncrementalMappingConditions();
    void combineMappingOperations();       
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
    std::shared_ptr<VoxelKeyframe> useOneRandomKeyframe();

    void increaseKeyframeTimesOfUse(const std::shared_ptr<VoxelKeyframe>& kf, int n);
    void cullKeyframes();

    void increasePcdByKeyframeInactiveGeoDensify(
        std::shared_ptr<VoxelKeyframe> pkf,
        bool include_inactive_geo = true,
        bool include_rgbd_hole_fill = true);
    void flushInactiveGeoCache();
    void processRgbdClosureCache();
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
    std::map<point3D_id_t, Point3D> buildInitialRgbdPointCloud() const;

    void recordKeyframeRendered(
        torch::Tensor &rendered,
        torch::Tensor &ground_truth,
        unsigned long kfid,
        std::filesystem::path result_img_dir,
        std::filesystem::path result_gt_dir,
        std::filesystem::path result_loss_dir,
        std::string name_suffix = "");
    void renderAndRecordKeyframe(
        std::shared_ptr<VoxelKeyframe> pkf,
        float&  dssim,
        float&  psnr,
        double& render_ms,
        const std::filesystem::path& img_dir,
        const std::filesystem::path& gt_dir,
        const std::filesystem::path& loss_dir,
        const std::filesystem::path& result_depth_dir,
        const std::filesystem::path& result_normal_dir,
        const std::filesystem::path& result_svraster_normal_dir,
        const std::string&  name_suffix = "",
        std::optional<float> global_depth_scale = std::nullopt,
        bool log_maps_to_rerun = false);
    void renderAndRecordAllKeyframes(
        const std::string& name_suffix = "");

    void savePly(std::filesystem::path result_dir);         
    void keyframesToJson(const std::filesystem::path& dir);   
    void writeKeyframeUsedTimes(std::filesystem::path result_dir, std::string name_suffix = "");
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
    torch::Tensor computeSdfInitForGridPoints(
        const torch::Tensor& grid_points_world,
        float ray_interval_m);
    int64_t fuseProjectiveSdfInitFromKeyframe(
        const std::shared_ptr<VoxelKeyframe>& kf,
        const torch::Tensor& pixel_mask = torch::Tensor());
    void runFinalSpecialPrune();
    torch::Tensor computeFinalSurfaceConfidenceKeepMask(
        bool retain_connected = true);
    torch::Tensor computeSvreconSdfPruneMask(float* sdf_threshold_out = nullptr);

    void ensureEmbeddedPythonRuntime(bool import_torch_cuda = false);

    // rerun debugging ---------------------------------------------------------
    void logKeyframeCameraToRerunRecordings(
        const std::shared_ptr<VoxelKeyframe>& pkf,
        unsigned long kf_id,
        bool log_reconstruction_mesh);
    void saveRerunRecordingsAtShutdown();
    void logCurrentOrbMapPointsToReconstructionRerun(int iteration);
    void logCurrentOrbKeyframePosesToReconstructionRerun(int iteration);
    void logWholeRunLiveVoxelsToRerun(
        int iteration,
        const torch::Tensor& centers,
        const torch::Tensor& sizes,
        const torch::Tensor& colors,
        bool log_whole_run = true,
        bool log_svrecon_debug = true);
    void appendWholeRunPrunedVoxels(
        int iteration,
        const torch::Tensor& centers,
        const torch::Tensor& sizes,
        const torch::Tensor& levels,
        const torch::Tensor& colors,
        const torch::Tensor& pruned_by_sdf,
        const torch::Tensor& pruned_by_surface_views,
        const torch::Tensor& pruned_by_final_surface = torch::Tensor());
    void logSvreconDebugVoxelMaskToRerun(
        int iteration,
        const torch::Tensor& mask,
        const std::string& entity_path);
    void logReconstructionMeshToRerun(int iteration);

    // evaluation/debugging ----------------------------------------------------
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

public:
    // Parameters
    std::filesystem::path config_file_path_;

    // Model
    std::shared_ptr<sv::VoxelScene> scene_;   
    std::shared_ptr<sv::VoxelModel> voxel_model_; 

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
    std::map<camera_id_t, torch::Tensor> undistort_mask_;
    std::map<camera_id_t, torch::Tensor> viewer_main_undistort_mask_;
    std::map<camera_id_t, torch::Tensor> viewer_sub_undistort_mask_; 
    
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
    bool disable_topology_changes_ = false;
    bool tail_refinement_active_ = false;
    int input_queue_max_keyframes_ = 0;
    int incremental_mapping_window_size_ = 0;
    bool loop_closure_reinsert_points_ = true;
    std::atomic<long long> latest_consumed_keyframe_id_{-1};
    std::atomic<bool> input_backpressure_ready_{false};
    int default_sh_ = 0;
    int svrecon_outside_level_ = 5;
    float global_scene_extent_m_ = 0.0f;
    bool robust_scene_bounds_ = false;
    bool sdf_initialization_rgbd_projective_ = false;
    bool allocate_orb_voxels_ = true;
    bool rgbd_surface_point_allocation_ = false;
    float sdf_initialization_orb_radius_vox_ = 2.0f;
    std::string sdf_initialization_mode_ = "orb_prior";

    // Settings
    SystemSensorType sensor_type_;

    float inactive_geo_densify_max_pixel_dist_ = 20.0f;
    float stereo_baseline_length_ = 0.0f;
    int stereo_min_disparity_ = 0;
    int stereo_num_disparity_ = 128;
    cv::Mat stereo_Q_;
    cv::Ptr<cv::cuda::StereoSGM> stereo_cv_sgm_;
    float RGBD_min_depth_ = 0.0f;
    float RGBD_max_depth_ = 100.0f;

    bool inactive_geo_densify_ = true;
    bool rgbd_fill_render_holes_initial_backfill_ = true;
    int depth_cached_ = 0;
    int max_depth_cached_ = 1;
    torch::Tensor depth_cache_points_;
    torch::Tensor depth_cache_colors_;
    torch::Tensor rgbd_fill_render_holes_cache_points_;
    torch::Tensor rgbd_fill_render_holes_cache_colors_;
    std::vector<std::pair<std::shared_ptr<VoxelKeyframe>, torch::Tensor>>
        rgbd_fill_render_holes_projective_cache_;
    std::vector<std::shared_ptr<VoxelKeyframe>> rgbd_hole_fill_keyframe_cache_;
    std::vector<std::shared_ptr<VoxelKeyframe>> rgbd_hole_fill_ready_keyframes_;

    // RGBD fill holes
    bool rgbd_fill_render_holes_ = false;
    bool rgbd_fill_render_holes_projective_sdf_ = false;
    int rgbd_fill_render_holes_stride_ = 2;

    // Non-renderable RGB-D TSDF evidence used only to promote confirmed
    // residual-hole cells into the active SVRecon octree.
    bool rgbd_tsdf_evidence_ = false;
    bool rgbd_tsdf_evidence_initial_backfill_ = true;
    int rgbd_tsdf_evidence_pixel_stride_ = 4;
    float rgbd_tsdf_evidence_trunc_vox_ = 4.0f;
    float rgbd_tsdf_evidence_max_weight_ = 5.0f;
    int rgbd_tsdf_evidence_promote_min_views_ = 2;
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
    int keyframe_record_interval_;
    int all_keyframes_record_interval_;
    bool record_rendered_image_;
    bool record_ground_truth_image_;
    bool record_loss_image_;

    int training_report_interval_;   
    bool record_loop_ply_;

    // Tools
    std::random_device rd_;
    
    // Mutex
    mutable std::mutex mutex_status_;
    mutable std::mutex mutex_render_;
    mutable std::mutex mutex_settings_;

    // Topology update schedule
    int next_subdiv_iter_   = 0;
    int next_prune_iter_    = 0;
    int next_opacity_reset_ = 0;

    int64_t last_densify_iter_ = -1;
};
