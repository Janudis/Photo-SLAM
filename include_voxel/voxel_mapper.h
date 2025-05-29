#pragma once

#include <torch/torch.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>     //  ← for gil_scoped_release
#include <opencv2/opencv.hpp>

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

#include "include_voxel/voxel_config.h"
#include "include_voxel/voxel_keyframe.h"
#include "include_voxel/voxel_trainer.h" 
#include "include_voxel/mini_cam.h"
#include "include_voxel/voxel_constants.h"
#include "include_voxel/py_utils.h"
#include "include/loss_utils.h"
#include "include/gaussian_model.h"
// ORB_SLAM3::System
#include "ORB-SLAM3/include/System.h"
#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

pybind11::array_t<uint8_t> cvMatToNumpyRGB(const cv::Mat& img);

#define CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(dir)                                       \
    if (!dir.empty() && !std::filesystem::exists(dir))                                      \
        if (!std::filesystem::create_directories(dir))                                      \
            throw std::runtime_error("Cannot create result directory at " + dir.string());

enum SystemSensorType {
    INVALID = 0,
    MONOCULAR = 1,
    STEREO = 2,
    RGBD = 3
};

struct VariableParameters {
    float lambda_photo;
    float lambda_dssim;
    int   new_keyframe_times_of_use;
    int   stable_num_iter_existence;
    bool  keep_training;
    bool  do_inactive_geo_densify;
    float position_lr_init;
};

class VoxelMapper {
public:
    VoxelMapper(std::shared_ptr<ORB_SLAM3::System> pSLAM,
                const std::filesystem::path& config_file_path,
                const std::filesystem::path& seq_dir,
                const std::filesystem::path& out_dir,
                torch::DeviceType device_type = torch::kCUDA,
                int seed = 0);

    ~VoxelMapper();
    void run();
    void trainForOneIteration();
    void trainLoopWithOptimization(int num_iters);
    void finalize();

    // graceful stop
    bool isStopped() const;
    void signalStop(bool stop = true);

    // misc helpers
    int  getIteration() const { return iteration_; }
    void increaseIteration(int inc=1) { iteration_ += inc; }

    VariableParameters getVariableParameters() const;
    void setVariableParameters(const VariableParameters& p);

    // new additions for viewer/compatibility
    std::filesystem::path getConfigFilePath() const;
    bool isKeepingTraining() const;
    void setKeepTraining(bool v);
    std::shared_ptr<sv::VoxelTrainer> getTrainer() const;
    // rendering / dumping -----------------------------------------------------
    cv::Mat renderFromPose(const Sophus::SE3f& Tcw, int width, int height, bool main_vision = false);

protected:
    // configuration IO --------------------------------------------------------
    void readConfigFromFile(const std::filesystem::path& cfg_path);
    // SLAM‑pipeline logic -----------------------------------------------------
    bool hasMetInitialMappingConditions();
    bool hasMetIncrementalMappingConditions();
    void combineMappingOperations();       ///< placeholder ‑ yet to be defined in .cpp
    void cullKeyframes();                  ///< placeholder ‑ yet to be defined in .cpp
    // voxel setup -------------------------------------------------------------
    void buildInitialKeyframesAndPointCloud();
    void initializeVoxelModel();
    // keyframe scheduling -----------------------------------------------------
    void generateKfidRandomShuffle();
    std::shared_ptr<VoxelKeyframe> useOneRandomSlidingWindowKeyframe();     
    // void increaseKeyframeTimesOfUse(std::shared_ptr<VoxelKeyframe> pkf, int value);    
    void increaseKeyframeTimesOfUse(const std::shared_ptr<VoxelKeyframe>& kf, int n);
    void  trainingReport(int iter,
                        float loss,
                        float ema_loss,
                        double ms_per_iter,
                        int   kfid);   
    void writeKeyframeUsedTimes(const std::filesystem::path& dir,
                                const std::string& suffix = "");
    void renderAndRecordKeyframe(std::shared_ptr<VoxelKeyframe> pkf,
                                 float&  dssim,
                                 float&  psnr,
                                 double& render_ms,
                                 const std::filesystem::path& img_dir,
                                 const std::filesystem::path& gt_dir,
                                 const std::filesystem::path& loss_dir,
                                 const std::string&  name_suffix = "");
    void renderAndRecordAllKeyframes(const std::string& name_suffix = "");
    void savePly (const std::filesystem::path& dir);            
    void keyframesToJson(const std::filesystem::path& dir);       
    void build_adam_optimizer(float base_lr);

protected:
    // SLAM system -------------------------------------------------------------
    std::shared_ptr<ORB_SLAM3::System>  mpSLAM;
    ORB_SLAM3::GeometricCamera* mpCamera = nullptr;
    // Settings
    torch::DeviceType device_type_;
    // SVRaster interface ------------------------------------------------------
    std::shared_ptr<sv::VoxelTrainer>   mpTrainer;
    // configuration & file paths ---------------------------------------------
    std::filesystem::path               mSeqDir;
    std::filesystem::path               mOutDir;
    std::filesystem::path               config_file_path_;
    std::filesystem::path               result_dir_;
    sv::VoxelScheduleConfig             mVoxelConfig;   ///< YAML‑driven schedule
    torch::Device                       mDevice = torch::Device(torch::kCPU);
    VariableParameters                  var_params_;
    SystemSensorType                    sensor_type_ = MONOCULAR;
    // voxel state -------------------------------------------------------------
    torch::Tensor                       voxel_centers_;
    // keyframe containers -----------------------------------------------------
    std::map<std::size_t, std::shared_ptr<VoxelKeyframe>> mSceneKeyframes;
    std::vector<int>                    mKeyframeIds;
    std::vector<std::string>            mKeyframeImages;
    std::vector<Sophus::SE3f>           mKeyframePoses;
    // sliding‑window bookkeeping ---------------------------------------------
    std::vector<std::size_t>            kfid_shuffle_;
    std::size_t                         kfid_shuffle_idx_ = 0;
    std::map<std::size_t,int>           kfs_used_times_;
    bool                                kfid_shuffled_   = false;
    // optional image lists ----------------------------------------------------
    std::vector<std::string>            mImagePaths;
    std::vector<Sophus::SE3f>           mTcwList;
    std::vector<double>                 mTimestamps;
    // rendering / viewer ------------------------------------------------------
    torch::Tensor                       background_;
    torch::Tensor                       override_color_;
    // status flags ------------------------------------------------------------
    int         iteration_;
    bool        keep_training_;
    bool        stopped_;
    bool        do_inactive_geo_densify_;
    bool        initial_mapped_;
    bool        SLAM_ended_;
    bool        cull_keyframes_;
    bool        loop_closure_iteration_;
    bool        interrupt_training_;
    std::size_t min_num_initial_map_kfs_;
    float       large_rot_th_;
    float       large_trans_th_;

    int keyframe_record_interval_      = 0;
    int all_keyframes_record_interval_ = 0;
    bool record_rendered_image_        = true;
    bool record_ground_truth_image_    = false;
    bool record_loss_image_            = false;

    /* ---------- new for logging / LR schedule ---------- */
    float ema_loss_for_log_;          ///< exponential‐moving-avg loss
    int   training_report_interval_;   ///< parsed from yaml
    
    std::random_device rd_;
    std::chrono::steady_clock::time_point iter_t0_;
    // adaptive LR helper
    float   lr_init_          = 0.0f;
    int     pos_lr_max_steps_     = 100;
    float   pos_lr_delay_mult_    = 0.1f;
    // synchronisation ---------------------------------------------------------
    mutable std::mutex mutex_status_;
    mutable std::mutex mutex_render_;
    mutable std::mutex mutex_settings_;
    // (optional) Python thread‑state holder ----------------------------------
    PyThreadState* m_savedState = nullptr;

    std::unique_ptr<torch::optim::Adam> optimizer_;   ///< global Adam
    int next_subdiv_iter_   = 0;
    int next_prune_iter_    = 0;
    int next_opacity_reset_ = 0;

    std::vector<unsigned long> kfid_index_; 

};
