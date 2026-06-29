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
#include <unordered_set>

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
#include "include_voxel/voxel_mono_prior_parameters.h"
#include "include_voxel/voxel_planner_parameters.h"
#include "include_voxel/voxel_rerun_parameters.h"
#include "include_voxel/voxel_sdf_parameters.h"
// #include "include/stereo_vision.h"
// #include "include/operate_points.h"

// ORB_SLAM3::System
#include "ORB-SLAM3/include/System.h"
#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

#include <nvblox/mapper/mapper.h>
#include <nvblox/map/blox.h>              // for BlockMemoryPoolParams
#include <nvblox/mapper/mapper_params.h>  // for MapperParams
#include <nvblox/sensors/camera.h>
#include <nvblox/sensors/image.h>   // DepthImage, ColorImage
#include <nvblox/map/layer.h>  // TsdfLayer, TsdfVoxel
#include <nvblox/core/types.h>      // Vector3f alias
#include <nvblox/map/voxels.h>        // TsdfVoxel
#include <nvblox/map/common_names.h>  // TsdfLayer alias
#include "rerun_utils.h" 
#include <nvblox/io/mesh_io.h>   // for io::outputColorMeshLayerToPly
#include "nvblox/integrators/tsdf_decay_integrator.h"
#include "nvblox/integrators/freespace_integrator.h"

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
     float lambda_dssim;
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
    
    float geoLearningRateInit();
    float sh0LearningRate();
    float shsLearningRate();
    float lambdaDssim();
    int densifyInterval();
    int newKeyframeTimesOfUse();
    bool isKeepingTraining();
    int stableNumIterExistence();
    bool isdoingGausPyramidTraining();
    bool isdoingInactiveGeoDensify();

    void setgeoLearningRateInit(const float lr);
    void setsh0LearningRate(const float lr);
    void setshsLearningRate(const float lr);
    void setLambdaDssim(const float lambda_dssim);
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

    void increasePcdByKeyframeInactiveGeoDensify(std::shared_ptr<VoxelKeyframe> pkf);

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
    void saveRenderedTsdfMeshPlySparseCpp(const std::filesystem::path& result_path);

    // sdf-based mapping -------------------------------------------------------
    void initializeNvbloxMapper();
    void integrateKeyframeIntoNvblox(VoxelKeyframe& kf,
                                    const cv::Mat& depth_meters);
    bool useSvrasterTsdfBackend() const;
    bool useNvbloxTsdfBackend() const;
    bool hasTsdfForSampling() const;
    float tsdfMetricVoxelSize() const;
    void integrateKeyframeIntoSvrasterSdf(VoxelKeyframe& kf,
                                          const cv::Mat& depth_meters);
    void refitSvrasterTsdfFromRegisteredKeyframes(const std::string& reason);
    bool prepareSvrasterTsdfInitContext(const std::shared_ptr<VoxelKeyframe>& kf);
    void clearSvrasterTsdfInitContext();
    torch::Tensor computeSvrasterProjectiveDensityInitForGridPoints(
        const torch::Tensor& grid_points_world,
        float ray_interval_m);
    struct TsdfSample {
        torch::Tensor tsdf;    // [N]
        torch::Tensor weight;  // [N]
        torch::Tensor success; // [N]
    };
    TsdfSample sampleTsdfAtPointsWorld(const torch::Tensor& pts_world);
    TsdfSample sampleNvbloxTsdfAtPointsWorld(const torch::Tensor& pts_world);
    torch::Tensor computeTsdfDensityInitForGridPoints(
        const torch::Tensor& grid_points_world,
        float ray_interval_m);
    struct TsdfCornerSample {
        torch::Tensor tsdf;    // [N,8] float32
        torch::Tensor weight;  // [N,8] float32
        torch::Tensor success; // [N,8] bool
        torch::Tensor points_world; // [N,8,3] float32, sampled SVRaster corners
    };
    TsdfCornerSample sampleTsdfAtVoxelCornersWorld(
        const torch::Tensor& centers_world,  // [N,3]
        const torch::Tensor& sizes_world     // [N,1] or [N]
    );
    TsdfCornerSample sampleTsdfAtSvrasterGridCornersWorld();
    void recordTsdfPruneAblation(
        const torch::Tensor& tsdf_prune_mask,
        const std::string& tag);
    void printTsdfPruneAblationSummary(const std::string& tag) const;
    void runFinalSpecialPrune();

    // mono-prior integration --------------------------------------------------
    void readMonoPriorConfigFromSettings(const cv::FileStorage& settings_file);
    void ensureEmbeddedPythonRuntime(bool import_torch_cuda = false);
    bool monoPriorUsesMetricDepth() const;
    bool ensureMonoPriorForKeyframe(
        const std::shared_ptr<VoxelKeyframe>& kf);
    bool buildAlignedMonoPriorDepthForKeyframe(
        const std::shared_ptr<VoxelKeyframe>& kf,
        const sv::MiniCam& cam,
        int image_width,
        int image_height,
        torch::Tensor& aligned_depth);
    bool depthAnythingFillHolesWarmupReady();
    void queueDepthAnythingFillHolesKeyframe(
        const std::shared_ptr<VoxelKeyframe>& pkf);
    void applyDepthAnythingFillHolesKeyframes(
        const std::vector<std::shared_ptr<VoxelKeyframe>>& keyframes,
        bool seed_global_alignment);
    void scheduleDepthAnythingFillHoles(
        const std::shared_ptr<VoxelKeyframe>& pkf);
    void processDepthAnythingFillHolesWarmup();
    torch::Tensor computeMonoPriorDepthLoss(
        const std::shared_ptr<VoxelKeyframe>& kf,
        const sv::MiniCam& cam,
        const std::unordered_map<std::string, torch::Tensor>& render_pkg,
        int iteration);
    torch::Tensor computeMonoPriorNormalLoss(
        const std::shared_ptr<VoxelKeyframe>& kf,
        const sv::MiniCam& cam,
        const std::unordered_map<std::string, torch::Tensor>& render_pkg,
        int iteration);
    void increasePcdByKeyframeMonoPriorFillHoles(std::shared_ptr<VoxelKeyframe> pkf);
    bool updateMonoPriorGlobalAlignmentFromKeyframe(const std::shared_ptr<VoxelKeyframe>& pkf);
    void accumulateMonoPriorGlobalAlignment(float scale, float shift, float weight);
    
    // monocular flood-full fill holes integration --------------------------------------------------
    void increasePcdByKeyframeRenderedDepthInsertion(std::shared_ptr<VoxelKeyframe> pkf);
    void updateRenderedDepthCandidateLifecycle();

    // rerun debugging ---------------------------------------------------------
    void logKeyframeCameraToRerunRecordings(
        const std::shared_ptr<VoxelKeyframe>& pkf,
        unsigned long kf_id,
        bool log_reconstruction_mesh);
    void saveRerunRecordingsAtShutdown();
    void appendAndLogOrbRawMapPcdToRerun(
        const std::map<point3D_id_t, Point3D>& pcd,
        int iteration);
    void appendAndLogOrbRawPointBatchToRerun(
        const std::vector<float>& points_flat,
        const std::vector<float>& colors_flat,
        int iteration);
    bool ensureRerunGtSdfGridCache(const std::string& mesh_path);
    void logTsdfUnknownVoxelsToRerun(
        int iteration,
        const torch::Tensor& voxel_colors = torch::Tensor());
    void logFloatersToRerun(int iteration);
    void logWholeRunLiveVoxelsToRerun(
        int iteration,
        const torch::Tensor& centers,
        const torch::Tensor& sizes,
        const torch::Tensor& colors);
    torch::Tensor computeNvbloxProjectiveSdfForCorners(
        const torch::Tensor& corner_points,
        const std::string& nvblox_mesh_path);
    void appendWholeRunPrunedVoxels(
        int iteration,
        const torch::Tensor& centers,
        const torch::Tensor& sizes,
        const torch::Tensor& pruned_by_tsdf,
        const torch::Tensor& pruned_by_near,
        const torch::Tensor& pruned_by_recent_unstable,
        const torch::Tensor& pruned_by_final_special = torch::Tensor());
    void logWholeRunNvbloxMeshToRerun(int iteration);
    void logReconstructionMeshToRerun(int iteration);
    void logNvbloxReconstructionMeshToRerun(int iteration);

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
    torch::Tensor computeRgbdNormalLoss(
        const std::shared_ptr<VoxelKeyframe>& kf,
        const sv::MiniCam& cam,
        const std::unordered_map<std::string, torch::Tensor>& render_pkg,
        int iteration);

    // sdf-based planner -------------------------------------------------------
    bool queryEsdfAtWorld(const Eigen::Vector3d& p_W, float& dist_out) const;
    void runTsdfPlannerAtShutdown();

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
    VoxelPlannerParameters planner_params_;
    // Feature state
    VoxelSdfState sdf_state_;
    VoxelPlannerState planner_state_;
    std::shared_ptr<nvblox::Mapper> sdf_mapper_;
    // Mono-prior state
    VoxelMonoPriorParameters mono_prior_params_;
    VoxelMonoPriorState mono_prior_state_;
    // Rerun debugging
    VoxelRerunParameters rerun_params_;
    VoxelRerunState rerun_state_;

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
    int default_sh_ = 0;

    // Settings
    SystemSensorType sensor_type_;

    float monocular_inactive_geo_densify_max_pixel_dist_ = 20.0;
    float stereo_baseline_length_ = 0.0f;
    int stereo_min_disparity_ = 0;
    int stereo_num_disparity_ = 128;
    cv::Mat stereo_Q_;
    cv::Ptr<cv::cuda::StereoSGM> stereo_cv_sgm_;
    float RGBD_min_depth_ = 0.0f;
    float RGBD_max_depth_ = 100.0f;

    bool inactive_geo_densify_ = true;
    bool rgbd_inactive_geo_initial_backfill_ = true;
    int depth_cached_ = 0;
    int max_depth_cached_ = 1;
    torch::Tensor depth_cache_points_;
    torch::Tensor depth_cache_colors_;

    // Artificial dense-core fill
    bool fill_empty_cells_ = false;

    // RGBD fill holes
    torch::Tensor rgbd_fill_render_holes_cache_points_;
    torch::Tensor rgbd_fill_render_holes_cache_colors_;
    bool rgbd_fill_render_holes_ = false;
    int rgbd_fill_render_holes_stride_ = 2;
    int rgbd_fill_render_holes_max_points_per_kf_ = 20000;

    // Monocular flood-full fill holes
    bool rendered_depth_insert_ = false;
    int rendered_depth_insert_stride_ = 4;
    int rendered_depth_insert_frontier_radius_px_ = 1;
    int rendered_depth_insert_max_points_per_kf_ = 3000;
    float rendered_depth_insert_normal_offset_vox_ = 0.5f;
    bool rendered_depth_insert_require_real_adjacency_ = true;
    int rendered_depth_insert_adjacency_radius_cells_ = 1;

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

    int64_t last_artificial_fill_iter_ = -1;
    int64_t last_densify_iter_ = -1;
};
