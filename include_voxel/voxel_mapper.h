#pragma once

#include <torch/torch.h>
#include <jsoncpp/json/json.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>     //  ← for gil_scoped_release
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

#include "include_voxel/voxel_parameters.h"
#include "include_voxel/voxel_keyframe.h"
#include "include_voxel/voxel_trainer.h" 
#include "include_voxel/mini_cam.h"
#include "include_voxel/voxel_camera.h"
#include "include_voxel/voxel_constants.h"
#include "include_voxel/py_utils.h"
#include "include/loss_utils.h"
#include "include_voxel/voxel_scene.h"
#include "include/tensor_utils.h"
#include "include_voxel/voxel_model.h"
#include "include_voxel/py_utils.h"
#include "include_voxel/render_opts.h"  
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

#include "voxel_planner.h"

pybind11::array_t<uint8_t> cvMatToNumpyRGB(const cv::Mat& img);

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
 };

class VoxelMapper {
public:
    VoxelMapper(
        std::shared_ptr<ORB_SLAM3::System> pSLAM,
        const std::filesystem::path& config_file_path,
        std::filesystem::path result_dir = "",
        int seed = 0,
        torch::DeviceType device_type = torch::kCUDA);
    
    std::unique_ptr<VoxelPlanner> planner_;
    bool planned_once_ = false;
    Eigen::Vector3f offline_goal_W_ = Eigen::Vector3f(10,5,0); // from YAML later
    float planner_clearance_m_ = 0.3f;
    bool queryEsdfAtWorld(const Eigen::Vector3d& p_W, float& dist_out) const;
    
    std::shared_ptr<nvblox::Mapper> sdf_mapper_;
    bool use_tsdf_mapping_ = false; 
    float sdf_voxel_size_m_ = 0.05f;   // example, configurable via YAML
    void initializeNvbloxMapper();
    void integrateKeyframeIntoNvblox(VoxelKeyframe& kf,
                                    const cv::Mat& depth_meters);
    struct TsdfSample {
        torch::Tensor tsdf;    // [N]
        torch::Tensor weight;  // [N]
        torch::Tensor success; // [N]
    };
    TsdfSample sampleTsdfAtPointsWorld(const torch::Tensor& pts_world);
    struct TsdfCornerSample {
        torch::Tensor tsdf;    // [N,8] float32
        torch::Tensor weight;  // [N,8] float32
        torch::Tensor success; // [N,8] bool
    };
    TsdfCornerSample sampleTsdfAtVoxelCornersWorld(
        const torch::Tensor& centers_world,  // [N,3]
        const torch::Tensor& sizes_world     // [N,1] or [N]
    );
    TsdfCornerSample sampleTsdfAtSvrasterGridCornersWorld();
    
    // ~VoxelMapper();
    void readConfigFromFile(const std::filesystem::path& cfg_path);

    void run();

    void saveDepthTensorAsPng(
        const torch::Tensor& depth_in,
        const std::filesystem::path& out_path);
    bool buildSparseDepthFromRGBD(
        const std::shared_ptr<VoxelKeyframe>& kf,
        torch::Tensor& sparse_uv,
        torch::Tensor& sparse_depth);
    torch::Tensor computeSparseDepthLoss(
        const std::shared_ptr<VoxelKeyframe>& kf,
        const std::unordered_map<std::string, torch::Tensor>& render_pkg,
        int iteration);
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
    void debugDepthStats(const cv::Mat& depth_meters, int kf_id);

    void trainForOneIteration();

    // graceful stop
    bool isStopped() const;
    // void signalStop(bool stop = true);
    void signalStop(const bool going_to_stop = true);

    void applyFinalTsdfTransparency();

    // rendering / dumping -----------------------------------------------------
    cv::Mat renderFromPose(
        const Sophus::SE3f& Tcw, 
        const int width, 
        const int height, 
        const bool main_vision = false);

    // misc helpers
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
    // SLAM‑pipeline logic -----------------------------------------------------
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
    // keyframe scheduling -----------------------------------------------------
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
        float&  depth_l1,
        float&  depth_f1,
        double& render_ms,
        const std::filesystem::path& img_dir,
        const std::filesystem::path& gt_dir,
        const std::filesystem::path& loss_dir,
        const std::filesystem::path& result_depth_dir,
        const std::string&  name_suffix = "");
    void renderAndRecordAllKeyframes(
        const std::string& name_suffix = "");

    void saveRenderedTsdfMeshPly(const std::filesystem::path& result_path);
    void saveRenderedTsdfMeshPlySvrasterPython(const std::filesystem::path& result_path);
    void saveRenderedTsdfMeshPlySparseCpp(const std::filesystem::path& result_path);
    void savePly(std::filesystem::path result_dir);         
    void keyframesToJson(const std::filesystem::path& dir);   
    void writeKeyframeUsedTimes(std::filesystem::path result_dir, std::string name_suffix = "");
    
    void saveVoxelErrorHeatmap(const sv::MiniCam& cam,
                            const at::Tensor& rendered_img,
                            const at::Tensor& gt_img,
                            int fid,
                            const std::string& base_dir);
    void savePhotometricErrorHeatmapAsPng(
        const torch::Tensor& error_tensor,
        const std::filesystem::path& out_path);

    void savePlannerNPZ(std::filesystem::path result_dir);

public:
    std::vector<float> best_loss_per_kf_;          // size == #key-frames
    std::vector<float> worst_loss_per_kf_;
    std::filesystem::path extrema_dir_;            // …/result/extrema
    
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

    std::ofstream pose_dump_stream_;
    bool poses_header_written_ = false;

    Eigen::Vector3f aabb_min_{ std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::infinity() };
    Eigen::Vector3f aabb_max_{ -std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity() };
    bool have_bounds_ = false;
    int next_batch_index_ = 0;

    std::vector<sv::MiniCam> tr_cams;

protected:
    VoxelModelParams model_params_;
    VoxelOptimizationParams opt_params_;
    VoxelPipelineParams pipe_params_;    
        
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
    int depth_cached_ = 0;
    int max_depth_cached_ = 1;
    torch::Tensor depth_cache_points_;
    torch::Tensor depth_cache_colors_;

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
    bool enable_rerun_ = true;
    bool rerun_final_only_ = false;
    int rerun_max_keyframes_ = -1; // <=0: all keyframes, >0: only keyframes in [150, 150+N) in rerun camera stream
    int rendered_mesh_backend_ = 0; // 0: current SVRaster Python exporter, 1: shared C++ sparse TSDF exporter
    bool record_depth_metrics_ = false;
    float depth_f1_threshold_m_ = 0.01f;

    int   training_report_interval_;   
    
    // Tools
    std::random_device rd_;
    
    // Mutex
    mutable std::mutex mutex_status_;
    mutable std::mutex mutex_render_;
    mutable std::mutex mutex_settings_;

    int next_subdiv_iter_   = 0;
    int next_prune_iter_    = 0;
    int next_opacity_reset_ = 0;

    // (optional) Python thread‑state holder ----------------------------------
    PyThreadState* m_savedState = nullptr;

    int64_t last_artificial_fill_iter_ = -1;
    int64_t last_densify_iter_       = -1;  // NEW: last prune/subdivide iteration
};
