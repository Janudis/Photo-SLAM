#include "include_voxel/voxel_mapper.h"
#include "include_voxel/voxel_mapper_utils.h"
#include "include_voxel/voxel_mapper_evaluation.h"
#include "include/stereo_vision.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <Eigen/Eigenvalues>
#include <filesystem>
#include <numeric>
#include <opencv2/flann.hpp>
#include <queue>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "ORB-SLAM3/include/Atlas.h"
#include "ORB-SLAM3/include/MapPoint.h"

namespace py = pybind11;

VoxelMapper::VoxelMapper(std::shared_ptr<ORB_SLAM3::System> pSLAM,
                         const std::filesystem::path& config_file_path,
                        std::filesystem::path result_dir,
                        int seed,
                        torch::DeviceType device_type)
    : mpSLAM(pSLAM),
      initial_mapped_(false),
      interrupt_training_(false),
      stopped_(false),
      iteration_(0),
      ema_loss_for_log_(0.0f),
      SLAM_ended_(false),
      loop_closure_iteration_(false),
      min_num_initial_map_kfs_(15UL),
      large_rot_th_(1e-1f),
      large_trans_th_(1e-2f),
      training_report_interval_(0)
{
    std::srand(seed);
    torch::manual_seed(seed);

    if (device_type == torch::kCUDA && torch::cuda::is_available()) {
        std::cout << "[VoxelMapper] CUDA available! Training on GPU." << std::endl;
        device_type_ = torch::kCUDA;
        mDevice = torch::Device(torch::kCUDA);
        model_params_.data_device_ = "cuda";
    } else {
        std::cout << "[VoxelMapper] Training on CPU." << std::endl;
        device_type_ = torch::kCPU;
        mDevice = torch::Device(torch::kCPU);
        model_params_.data_device_ = "cpu";
    }

    // result_dir_ = mOutDir;
    result_dir_ = result_dir;
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir);
    config_file_path_ = config_file_path;
    readConfigFromFile(config_file_path);

    // Background & override color setup
    std::vector<float> bg_color = {0.0f, 0.0f, 0.0f};  // no white-background logic needed
    if (model_params_.white_background_)
         bg_color = {1.0f, 1.0f, 1.0f};
     else
         bg_color = {0.0f, 0.0f, 0.0f};
    background_ = torch::tensor(bg_color,
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    override_color_ = torch::empty(0, torch::TensorOptions().device(device_type_));

    voxel_model_ = std::make_shared<sv::VoxelModel>(model_params_);
    scene_ = std::make_shared<sv::VoxelScene>(model_params_);

    voxel_model_->fill_empty_cells_ = fill_empty_cells_;
    voxel_model_->setFilterNearVoxels(opt_params_.filter_near_voxels_);
    voxel_model_->setRenderedDepthCandidateRealAdjacency(
        rendered_depth_insert_require_real_adjacency_,
        rendered_depth_insert_adjacency_radius_cells_);
    voxel_model_->setEnableArtificialPromotion(fill_empty_cells_);
    voxel_model_->setGeoGridInitCallback(
        [this](const torch::Tensor& grid_points_world, float ray_interval_m) {
            return this->computeTsdfDensityInitForGridPoints(
                grid_points_world,
                ray_interval_m);
        });

    switch (pSLAM->getSensorType()) {
    case ORB_SLAM3::System::MONOCULAR:
    case ORB_SLAM3::System::IMU_MONOCULAR:
    {
        this->sensor_type_ = MONOCULAR;
    }
    break;
    case ORB_SLAM3::System::STEREO:
    case ORB_SLAM3::System::IMU_STEREO:
    {
        this->sensor_type_ = STEREO;
        this->stereo_baseline_length_ = pSLAM->getSettings()->b();
        this->stereo_cv_sgm_ = cv::cuda::createStereoSGM(
            this->stereo_min_disparity_,
            this->stereo_num_disparity_);
        this->stereo_Q_ = pSLAM->getSettings()->Q().clone();
        stereo_Q_.convertTo(stereo_Q_, CV_32FC3, 1.0);
    }
    break;
    case ORB_SLAM3::System::RGBD:
    case ORB_SLAM3::System::IMU_RGBD:
    {
        this->sensor_type_ = RGBD;
    }
    break;
    default:
    {
        throw std::runtime_error("[Gaussian Mapper]Unsupported sensor type!");
    }
    break;
    }

    // /* Load every ORB-SLAM3 camera, convert to Camera, pre–compute            */
    auto settings = pSLAM->getSettings();   
    cv::Size SLAM_im_size = settings->newImSize();
    UndistortParams undistort_params(
        SLAM_im_size,
        settings->camera1DistortionCoef()
    );
    auto vpCameras = pSLAM->getAtlas()->GetAllCameras();
    for (auto& SLAM_camera : vpCameras) {
        sv::Camera camera;
        camera.camera_id_ = SLAM_camera->GetId();
        if (SLAM_camera->GetType() == ORB_SLAM3::GeometricCamera::CAM_PINHOLE) {
            camera.setModelId(sv::Camera::CameraModelType::PINHOLE);
            float SLAM_fx = SLAM_camera->getParameter(0);
            float SLAM_fy = SLAM_camera->getParameter(1);
            float SLAM_cx = SLAM_camera->getParameter(2);
            float SLAM_cy = SLAM_camera->getParameter(3);

            // Old K, i.e. K in SLAM
            cv::Mat K = (
                cv::Mat_<float>(3, 3)
                    << SLAM_fx, 0.f, SLAM_cx,
                        0.f, SLAM_fy, SLAM_cy,
                        0.f, 0.f, 1.f
            );
            camera.width_ = undistort_params.old_size_.width;
            float x_ratio = static_cast<float>(camera.width_) / undistort_params.old_size_.width;
            camera.height_ = undistort_params.old_size_.height;
            float y_ratio = static_cast<float>(camera.height_) / undistort_params.old_size_.height;

            camera.num_gaus_pyramid_sub_levels_ = num_gaus_pyramid_sub_levels_;
            camera.gaus_pyramid_width_.resize(num_gaus_pyramid_sub_levels_);
            camera.gaus_pyramid_height_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                camera.gaus_pyramid_width_[l] = camera.width_ * this->kf_gaus_pyramid_factors_[l];
                camera.gaus_pyramid_height_[l] = camera.height_ * this->kf_gaus_pyramid_factors_[l];
            }

            camera.params_[0]/*new fx*/= SLAM_fx * x_ratio;
            camera.params_[1]/*new fy*/= SLAM_fy * y_ratio;
            camera.params_[2]/*new cx*/= SLAM_cx * x_ratio;
            camera.params_[3]/*new cy*/= SLAM_cy * y_ratio;

            cv::Mat K_new = (
                cv::Mat_<float>(3, 3)
                    << camera.params_[0], 0.f, camera.params_[2],
                        0.f, camera.params_[1], camera.params_[3],
                        0.f, 0.f, 1.f
            );

            // Undistortion
            if (this->sensor_type_ == MONOCULAR || this->sensor_type_ == RGBD)
                undistort_params.dist_coeff_.copyTo(camera.dist_coeff_);

            camera.initUndistortRectifyMapAndMask(K, SLAM_im_size, K_new, true);

            undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    camera.undistort_mask, device_type_);

            cv::Mat viewer_sub_undistort_mask;
            int viewer_image_height_ = camera.height_ * rendered_image_viewer_scale_;
            int viewer_image_width_ = camera.width_ * rendered_image_viewer_scale_;
            cv::resize(camera.undistort_mask, viewer_sub_undistort_mask,
                    cv::Size(viewer_image_width_, viewer_image_height_));
            viewer_sub_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_sub_undistort_mask, device_type_);

            cv::Mat viewer_main_undistort_mask;
            int viewer_image_height_main_ = camera.height_ * rendered_image_viewer_scale_main_;
            int viewer_image_width_main_ = camera.width_ * rendered_image_viewer_scale_main_;
            cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                    cv::Size(viewer_image_width_main_, viewer_image_height_main_));
            viewer_main_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_main_undistort_mask, device_type_);

            if (this->sensor_type_ == STEREO) {
                camera.stereo_bf_ = stereo_baseline_length_ * camera.params_[0];
                if (this->stereo_Q_.cols != 4) {
                    this->stereo_Q_ = cv::Mat(4, 4, CV_32FC1);
                    this->stereo_Q_.setTo(0.0f);
                    this->stereo_Q_.at<float>(0, 0) = 1.0f;
                    this->stereo_Q_.at<float>(0, 3) = -camera.params_[2];
                    this->stereo_Q_.at<float>(1, 1) = 1.0f;
                    this->stereo_Q_.at<float>(1, 3) = -camera.params_[3];
                    this->stereo_Q_.at<float>(2, 3) = camera.params_[0];
                    this->stereo_Q_.at<float>(3, 2) = 1.0f / stereo_baseline_length_;
                }
            }
        }
        else if (SLAM_camera->GetType() == ORB_SLAM3::GeometricCamera::CAM_FISHEYE) {
            camera.setModelId(sv::Camera::CameraModelType::FISHEYE);
        }
        else {
            camera.setModelId(sv::Camera::CameraModelType::INVALID);
        }

        if (!viewer_camera_id_set_) {
            viewer_camera_id_ = camera.camera_id_;
            viewer_camera_id_set_ = true;
        }
        this->scene_->addCamera(camera);
    }
}

void VoxelMapper::readConfigFromFile(const std::filesystem::path& cfg_path)
{
    cv::FileStorage settings_file(cfg_path.string(), cv::FileStorage::READ);
    if (!settings_file.isOpened()) {
        std::cerr << "[VoxelMapper] Failed to open cfg: " << cfg_path << '\n';
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[VoxelMapper] Reading parameters from " << cfg_path << '\n';
    std::unique_lock<std::mutex> lock(mutex_settings_);

    // Model parameters
     model_params_.sh_degree_ =
         settings_file["Model.sh_degree"].operator int();
     model_params_.resolution_ =
         settings_file["Model.resolution"].operator float();
     model_params_.white_background_ =
         (settings_file["Model.white_background"].operator int()) != 0;
     model_params_.eval_ =
         (settings_file["Model.eval"].operator int()) != 0;

    /* ───────── PIPELINE FLAGS ───────── */
    z_near_ =
         settings_file["Camera.z_near"].operator float();
    if (!settings_file["Monocular.inactive_geo_densify_max_pixel_dist"].empty()) {
        monocular_inactive_geo_densify_max_pixel_dist_ =
            settings_file["Monocular.inactive_geo_densify_max_pixel_dist"].operator float();
    }
    cull_keyframes_ =
        (settings_file["Mapper.cull_keyframes"].operator int()) != 0;
    min_num_initial_map_kfs_ =
        static_cast<std::size_t>(settings_file["Mapper.min_num_initial_map_kfs"].operator int());
    new_keyframe_times_of_use_ =
        settings_file["Mapper.new_keyframe_times_of_use"].operator int();
    large_rot_th_ =
        settings_file["Mapper.large_rotation_threshold"].operator float();
    large_trans_th_ =
        settings_file["Mapper.large_translation_threshold"].operator float();
    local_BA_increased_times_of_use_ = 
         settings_file["Mapper.local_BA_increased_times_of_use"].operator int();
    loop_closure_increased_times_of_use_ = 
         settings_file["Mapper.loop_closure_increased_times_of_use_"].operator int();

    RGBD_min_depth_ =
        settings_file["RGBD.min_depth"].operator float();
    RGBD_max_depth_ =
        settings_file["RGBD.max_depth"].operator float();

    inactive_geo_densify_ =
        (settings_file["Mapper.inactive_geo_densify"].operator int()) != 0;
    max_depth_cached_ =
        settings_file["Mapper.depth_cache"].operator int();
    if (!settings_file["Mapper.rgbd_inactive_geo_initial_backfill"].empty()) {
        rgbd_inactive_geo_initial_backfill_ =
            (settings_file["Mapper.rgbd_inactive_geo_initial_backfill"].operator int()) != 0;
    }
    if (!settings_file["Mapper.rgbd_fill_render_holes"].empty()) {
        rgbd_fill_render_holes_ =
            (settings_file["Mapper.rgbd_fill_render_holes"].operator int()) != 0;
    }
    if (!settings_file["Mapper.rgbd_fill_render_holes_stride"].empty()) {
        rgbd_fill_render_holes_stride_ =
            std::max(1, settings_file["Mapper.rgbd_fill_render_holes_stride"].operator int());
    }
    if (!settings_file["Mapper.rgbd_fill_render_holes_max_points_per_kf"].empty()) {
        rgbd_fill_render_holes_max_points_per_kf_ =
            std::max(0, settings_file["Mapper.rgbd_fill_render_holes_max_points_per_kf"].operator int());
    }

    sdf_params_.use_tsdf_mapping_ =
        (settings_file["Mapper.use_tsdf_mapping"].operator int()) != 0;
    sdf_params_.use_tsdf_pruning_ =
        (settings_file["Mapper.use_tsdf_pruning"].operator int()) != 0;
    sdf_params_.use_tsdf_planning_ =
        (settings_file["Mapper.use_tsdf_planning"].operator int()) != 0;
    sdf_params_.sdf_voxel_size_m_ =
        std::max(1.0e-4f, settings_file["Mapper.tsdf_voxel_size_m"].operator float());
    if (!settings_file["Mapper.tsdf_backend"].empty()) {
        sdf_params_.tsdf_backend_ =
            voxel_utils::toLowerCopy(settings_file["Mapper.tsdf_backend"].operator std::string());
        if (sdf_params_.tsdf_backend_ != "svraster" && sdf_params_.tsdf_backend_ != "nvblox") {
            std::cerr << "[TSDF] Unknown Mapper.tsdf_backend='" << sdf_params_.tsdf_backend_
                      << "', falling back to 'svraster'.\n";
            sdf_params_.tsdf_backend_ = "svraster";
        }
        sdf_params_.tsdf_prune_min_weight_ =
            std::max(0.0f, settings_file["Mapper.tsdf_prune_min_weight"].operator float());
        sdf_params_.tsdf_prune_surface_band_vox_ =
            std::max(0.0f, settings_file["Mapper.tsdf_prune_surface_band_vox"].operator float());
        sdf_params_.tsdf_prune_min_valid_corners_ =
            std::max(1, std::min(8, settings_file["Mapper.tsdf_prune_min_valid_corners"].operator int()));
        sdf_params_.tsdf_protect_surface_band_from_pruning_ =
            (settings_file["Mapper.tsdf_protect_surface_band_from_pruning"].operator int()) != 0;
        sdf_params_.tsdf_density_init_ =
            (settings_file["Mapper.tsdf_density_init"].operator int()) != 0;
        sdf_params_.tsdf_density_init_min_weight_ =
            std::max(0.0f, settings_file["Mapper.tsdf_density_init_min_weight"].operator float());
        sdf_params_.tsdf_density_init_trunc_vox_ =
            std::max(1.0e-3f, settings_file["Mapper.tsdf_density_init_trunc_vox"].operator float());
        sdf_params_.svraster_tsdf_max_integration_distance_m_ =
            std::max(0.0f, settings_file["Mapper.svraster_tsdf_max_integration_distance_m"].operator float());
        sdf_params_.svraster_tsdf_inverse_square_weighting_ =
            (settings_file["Mapper.svraster_tsdf_inverse_square_weighting"].operator int()) != 0;
        sdf_params_.svraster_tsdf_max_weight_ =
            std::max(1.0e-6f, settings_file["Mapper.svraster_tsdf_max_weight"].operator float());
        sdf_params_.svraster_tsdf_refit_on_topology_change_ =
            (settings_file["Mapper.svraster_tsdf_refit_on_topology_change"].operator int()) != 0;
        sdf_params_.tsdf_density_init_bell_a_ =
            std::max(1.0e-4f, settings_file["Mapper.tsdf_density_init_bell_a"].operator float());
        sdf_params_.tsdf_density_init_bell_b_ =
            std::clamp(settings_file["Mapper.tsdf_density_init_bell_b"].operator float(), 1.0e-4f, 0.9999f);
        sdf_params_.tsdf_density_init_alpha_min_ =
            std::clamp(settings_file["Mapper.tsdf_density_init_alpha_min"].operator float(), 1.0e-6f, 0.999f);
        sdf_params_.tsdf_density_init_alpha_max_ =
            std::clamp(settings_file["Mapper.tsdf_density_init_alpha_max"].operator float(), 1.0e-6f, 0.999f);
        if (sdf_params_.tsdf_density_init_alpha_max_ < sdf_params_.tsdf_density_init_alpha_min_) {
            std::swap(sdf_params_.tsdf_density_init_alpha_max_, sdf_params_.tsdf_density_init_alpha_min_);
        }
    }
    if (!settings_file["Mapper.depthanything_fill_holes"].empty() ||
        !settings_file["Optimization.lambda_depthanythingv2"].empty()) {
        readMonoPriorConfigFromSettings(settings_file);
    }
    if (!settings_file["Mapper.fill_empty_cells"].empty()) {
        fill_empty_cells_ =
            (settings_file["Mapper.fill_empty_cells"].operator int()) != 0;
    }
    if (!settings_file["Mapper.rendered_depth_insert"].empty()) {
        rendered_depth_insert_ =
            (settings_file["Mapper.rendered_depth_insert"].operator int()) != 0;
    }
    if (!settings_file["Mapper.rendered_depth_insert_stride"].empty()) {
        rendered_depth_insert_stride_ =
            std::max(1, settings_file["Mapper.rendered_depth_insert_stride"].operator int());
    }
    if (!settings_file["Mapper.rendered_depth_insert_frontier_radius_px"].empty()) {
        rendered_depth_insert_frontier_radius_px_ =
            std::max(1, settings_file["Mapper.rendered_depth_insert_frontier_radius_px"].operator int());
    }
    if (!settings_file["Mapper.rendered_depth_insert_max_points_per_kf"].empty()) {
        rendered_depth_insert_max_points_per_kf_ =
            std::max(1, settings_file["Mapper.rendered_depth_insert_max_points_per_kf"].operator int());
    }
    if (!settings_file["Mapper.rendered_depth_insert_normal_offset_vox"].empty()) {
        rendered_depth_insert_normal_offset_vox_ =
            std::max(0.0f, settings_file["Mapper.rendered_depth_insert_normal_offset_vox"].operator float());
    }
    if (!settings_file["Mapper.rendered_depth_insert_require_real_adjacency"].empty()) {
        rendered_depth_insert_require_real_adjacency_ =
            (settings_file["Mapper.rendered_depth_insert_require_real_adjacency"].operator int()) != 0;
    }
    if (!settings_file["Mapper.rendered_depth_insert_adjacency_radius_cells"].empty()) {
        rendered_depth_insert_adjacency_radius_cells_ =
            std::max(1, settings_file["Mapper.rendered_depth_insert_adjacency_radius_cells"].operator int());
    }

    pipe_params_.convert_SHs_ =
         (settings_file["Pipeline.convert_SHs"].operator int()) != 0;

    do_gaus_pyramid_training_ =
         (settings_file["GausPyramid.do"].operator int()) != 0;
    num_gaus_pyramid_sub_levels_ =
        settings_file["GausPyramid.num_sub_levels"].operator int();
    int sub_level_times_of_use =
        settings_file["GausPyramid.sub_level_times_of_use"].operator int();
    kf_gaus_pyramid_times_of_use_.resize(num_gaus_pyramid_sub_levels_);
    kf_gaus_pyramid_factors_.resize(num_gaus_pyramid_sub_levels_);
    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        kf_gaus_pyramid_times_of_use_[l] = sub_level_times_of_use;
        kf_gaus_pyramid_factors_[l] = std::pow(0.5f, num_gaus_pyramid_sub_levels_ - l);
    }
    
    /* ───────── OPTIMIZATION PARAMETERS ───────── */
    opt_params_.iterations_ =
        settings_file["Optimization.max_num_iterations"].operator int();
    opt_params_.geo_lr_ =
        settings_file["Optimization.geo_lr"].operator float();
    opt_params_.sh0_lr_ =
        settings_file["Optimization.sh0_lr"].operator float();
    opt_params_.shs_lr_ =
        settings_file["Optimization.shs_lr"].operator float();
    {
        cv::FileNode n = settings_file["Optimization.lr_decay_ckpt"];
        opt_params_.lr_decay_ckpt_.clear();
        if (!n.empty())
        {
            if (n.type() == cv::FileNode::SEQ) {
                // YAML: Optimization.lr_decay_ckpt: [5000, 10000, 20000]
                for (auto it = n.begin(); it != n.end(); ++it)
                    opt_params_.lr_decay_ckpt_.push_back((int)*it);
            } else if (n.isInt()) {
                // YAML: Optimization.lr_decay_ckpt: 10000
                opt_params_.lr_decay_ckpt_.push_back((int)n);
            } else if (n.isString()) {
                // YAML: Optimization.lr_decay_ckpt: "5000,10000,20000"
                std::string s = (std::string)n;
                std::stringstream ss(s);
                for (std::string tok; std::getline(ss, tok, ','); ) {
                    if (!tok.empty()) opt_params_.lr_decay_ckpt_.push_back(std::stoi(tok));
                }
            }
        }
    }
    opt_params_.optim_beta1_ =
        settings_file["Optimization.optim_beta1"].operator float();
    opt_params_.optim_beta2_ =
        settings_file["Optimization.optim_beta2"].operator float();
    opt_params_.optim_eps_ =
        settings_file["Optimization.optim_eps"].operator float();
    opt_params_.lr_decay_mult_ =
        settings_file["Optimization.lr_decay_mult"].operator float();

    opt_params_.adapt_from_ =
        settings_file["Optimization.adapt_from"].operator int();
    opt_params_.adapt_every_ =
        settings_file["Optimization.adapt_every"].operator int();
    opt_params_.prune_every_ =
        settings_file["Optimization.prune_every"].operator int();
    opt_params_.subdivide_every_ =
        settings_file["Optimization.subdivide_every"].operator int();
    opt_params_.filter_near_voxels_ =
        (settings_file["Optimization.filter_near_voxels"].operator int()) != 0;
    opt_params_.prune_far_voxels_ =
        (settings_file["Optimization.prune_far_voxels"].operator int()) != 0;
    opt_params_.prune_near_voxels_geometric_ =
        !settings_file["Optimization.prune_near_voxels_geometric"].empty() &&
        (settings_file["Optimization.prune_near_voxels_geometric"].operator int()) != 0;
    opt_params_.prune_recent_unstable_ =
        (settings_file["Optimization.prune_recent_unstable"].operator int()) != 0;
    opt_params_.prune_recent_keyframes_ =
        settings_file["Optimization.prune_recent_keyframes"].operator int();
    opt_params_.prune_recent_min_views_real_ =
        settings_file["Optimization.prune_recent_min_views_real"].operator int();
    opt_params_.prune_recent_min_views_artificial_ =
        settings_file["Optimization.prune_recent_min_views_artificial"].operator int();
    if (!settings_file["Optimization.rendered_depth_candidate_promote_min_support"].empty()) {
        opt_params_.rendered_depth_candidate_promote_min_support_ =
            std::max(1, settings_file["Optimization.rendered_depth_candidate_promote_min_support"].operator int());
    }
    if (!settings_file["Optimization.rendered_depth_candidate_prune_kf_age"].empty()) {
        opt_params_.rendered_depth_candidate_prune_kf_age_ =
            std::max(1, settings_file["Optimization.rendered_depth_candidate_prune_kf_age"].operator int());
    }
    opt_params_.final_special_prune_enable_ =
        (settings_file["Optimization.final_special_prune_enable"].operator int()) != 0;
    opt_params_.prune_until_ =
        settings_file["Optimization.prune_until"].operator int();
    opt_params_.prune_thres_init_ =
        settings_file["Optimization.prune_thres_init"].operator float();
    opt_params_.prune_thres_final_ =
        settings_file["Optimization.prune_thres_final"].operator float();
    opt_params_.prune_thres_final_at_target_ =
        settings_file["Optimization.prune_thres_final_at_target"].operator float();
    opt_params_.prune_thres_init_artificial_ =
        settings_file["Optimization.prune_thres_init_artificial"].operator float();
    opt_params_.prune_thres_final_artificial_ =
        settings_file["Optimization.prune_thres_final_artificial"].operator float();

    opt_params_.subdivide_until_ =
        settings_file["Optimization.subdivide_until"].operator int();
    opt_params_.subdivide_all_until_ =
        settings_file["Optimization.subdivide_all_until"].operator int();
    opt_params_.subdivide_samp_thres_ =
        settings_file["Optimization.subdivide_samp_thres"].operator float();
    opt_params_.subdivide_prop_ =
        settings_file["Optimization.subdivide_prop"].operator float();
    opt_params_.subdivide_max_num_ =
        settings_file["Optimization.subdivide_max_num"].operator int();

    opt_params_.lambda_dssim_ =
        settings_file["Optimization.lambda_dssim"].operator float();
    opt_params_.use_l1_ =
        (settings_file["Optimization.use_l1"].operator int()) != 0;
    opt_params_.use_huber_ =
        (settings_file["Optimization.use_huber"].operator int()) != 0;
    opt_params_.huber_thres_ =
        settings_file["Optimization.huber_thres"].operator float();
    if (opt_params_.use_l1_ && opt_params_.use_huber_) {
        std::cout << "[VoxelMapper] Both Optimization.use_l1 and Optimization.use_huber are enabled. "
                  << "Prioritizing L1 to match SVRaster." << std::endl;
    }

    opt_params_.lambda_tv_density_ =
        settings_file["Optimization.lambda_tv_density"].operator float();
    opt_params_.tv_from_ =
        settings_file["Optimization.tv_from"].operator int();
    opt_params_.tv_until_ =
        settings_file["Optimization.tv_until"].operator int();

    opt_params_.ss_aug_max_ = settings_file["Optimization.ss_aug_max"].operator float();
    opt_params_.lambda_R_concen_ = settings_file["Optimization.lambda_R_concen"].operator float();
    opt_params_.lambda_dist_ = settings_file["Optimization.lambda_dist"].operator float();
    if (!settings_file["Optimization.dist_from"].empty()) {
        opt_params_.dist_from_ = settings_file["Optimization.dist_from"].operator int();
    }
    opt_params_.lambda_T_concen_ = settings_file["Optimization.lambda_T_concen"].operator float();
    opt_params_.lambda_T_inside_ = settings_file["Optimization.lambda_T_inside"].operator float();
    opt_params_.lambda_normal_dmean_ = settings_file["Optimization.lambda_normal_dmean"].operator float();
    opt_params_.n_dmean_from_ = settings_file["Optimization.n_dmean_from"].operator int();
    opt_params_.n_dmean_end_ = settings_file["Optimization.n_dmean_end"].operator int();
    opt_params_.n_dmean_ks_ = settings_file["Optimization.n_dmean_ks"].operator int();
    opt_params_.n_dmean_tol_deg_ = settings_file["Optimization.n_dmean_tol_deg"].operator float();
    opt_params_.lambda_ssim_ = settings_file["Optimization.lambda_ssim"].operator float();

    opt_params_.lambda_sparse_depth_ = settings_file["Optimization.lambda_sparse_depth"].operator float();
    opt_params_.sparse_depth_until_ = settings_file["Optimization.sparse_depth_until"].operator int();
    if (!settings_file["Optimization.lambda_depthanythingv2"].empty()) {
        opt_params_.lambda_depthanythingv2_ = settings_file["Optimization.lambda_depthanythingv2"].operator float();
        opt_params_.depthanythingv2_from_ = settings_file["Optimization.depthanythingv2_from"].operator int();
        opt_params_.depthanythingv2_end_ = settings_file["Optimization.depthanythingv2_end"].operator int();
        opt_params_.depthanythingv2_end_mult_ = settings_file["Optimization.depthanythingv2_end_mult"].operator float();
        opt_params_.depthanythingv2_overall_ =
            (settings_file["Optimization.depthanythingv2_overall"].operator int()) != 0;
        opt_params_.depthanythingv2_alpha_adjust_ =
            (settings_file["Optimization.depthanythingv2_alpha_adjust"].operator int()) != 0;
        opt_params_.enable_da2_uncertainty_ =
            (settings_file["Optimization.enable_da2_uncertainty"].operator int()) != 0;
        opt_params_.level_uncertainty_from_ =
            settings_file["Optimization.level_uncertainty_from"].operator int();
        opt_params_.power_level_uncertainty_ =
            settings_file["Optimization.power_level_uncertainty"].operator float();
        opt_params_.lambda_ascending_ =
            settings_file["Optimization.lambda_ascending"].operator float();
        opt_params_.ascending_from_ =
            settings_file["Optimization.ascending_from"].operator int();
        opt_params_.lambda_rectify_ =
            settings_file["Optimization.lambda_rectify"].operator float();
        opt_params_.rectifiy_from_ =
            settings_file["Optimization.rectifiy_from"].operator int();
        opt_params_.lambda_scaling_penalty_ =
            settings_file["Optimization.lambda_scaling_penalty"].operator float();
        opt_params_.scaling_penalty_from_ =
            settings_file["Optimization.scaling_penalty_from"].operator int();
        opt_params_.scaling_penalty_end_ =
            settings_file["Optimization.scaling_penalty_end"].operator int();
        opt_params_.multi_view_weight_from_iter_ =
            settings_file["Optimization.multi_view_weight_from_iter"].operator int();
        opt_params_.multi_view_interval_ =
            settings_file["Optimization.multi_view_interval"].operator int();
        opt_params_.multi_view_anneal_scale_ =
            settings_file["Optimization.multi_view_anneal_scale"].operator float();
        opt_params_.multi_view_ncc_weight_ =
            settings_file["Optimization.multi_view_ncc_weight"].operator float();
        opt_params_.multi_view_geo_weight_ =
            settings_file["Optimization.multi_view_geo_weight"].operator float();
        opt_params_.multi_view_patch_size_ =
            settings_file["Optimization.multi_view_patch_size"].operator int();
        opt_params_.multi_view_sample_num_ =
            settings_file["Optimization.multi_view_sample_num"].operator int();
        opt_params_.multi_view_pixel_noise_th_ =
            settings_file["Optimization.multi_view_pixel_noise_th"].operator float();
        opt_params_.voxel_dropout_min_ =
            settings_file["Optimization.voxel_dropout_min"].operator float();
        opt_params_.lambda_depthanythingv2_normal_ =
            settings_file["Optimization.lambda_depthanythingv2_normal"].operator float();
        opt_params_.depthanythingv2_normal_from_ =
            settings_file["Optimization.depthanythingv2_normal_from"].operator int();
        opt_params_.depthanythingv2_normal_end_ =
            settings_file["Optimization.depthanythingv2_normal_end"].operator int();
        opt_params_.depthanythingv2_normal_end_mult_ =
            settings_file["Optimization.depthanythingv2_normal_end_mult"].operator float();
        opt_params_.depthanythingv2_normal_ks_ =
            settings_file["Optimization.depthanythingv2_normal_ks"].operator int();
        opt_params_.depthanythingv2_normal_tol_deg_ =
            settings_file["Optimization.depthanythingv2_normal_tol_deg"].operator float();
    }
    if (!settings_file["Optimization.lambda_rgbd_depth"].empty()) {
        opt_params_.lambda_rgbd_depth_ = settings_file["Optimization.lambda_rgbd_depth"].operator float();
        opt_params_.rgbd_depth_from_ = settings_file["Optimization.rgbd_depth_from"].operator int();
        opt_params_.rgbd_depth_end_ = settings_file["Optimization.rgbd_depth_end"].operator int();
        opt_params_.rgbd_depth_end_mult_ = settings_file["Optimization.rgbd_depth_end_mult"].operator float();
        opt_params_.lambda_rgbd_normal_ = settings_file["Optimization.lambda_rgbd_normal"].operator float();
        opt_params_.rgbd_normal_from_ = settings_file["Optimization.rgbd_normal_from"].operator int();
        opt_params_.rgbd_normal_end_ = settings_file["Optimization.rgbd_normal_end"].operator int();
        opt_params_.rgbd_normal_end_mult_ = settings_file["Optimization.rgbd_normal_end_mult"].operator float();
        opt_params_.rgbd_normal_ks_ = settings_file["Optimization.rgbd_normal_ks"].operator int();
        opt_params_.rgbd_normal_tol_deg_ = settings_file["Optimization.rgbd_normal_tol_deg"].operator float();
    }
    if (opt_params_.multi_view_weight_from_iter_ < 1000000000 &&
        (opt_params_.multi_view_ncc_weight_ > 0.0f ||
         opt_params_.multi_view_geo_weight_ > 0.0f)) {
        std::cerr << "[GeoSVR] Multi-view NCC/geometry regularization is configured, "
                     "but it is not wired into the current Photo-SLAM training loop yet. "
                     "The configs are parsed, but the term is currently inactive."
                  << std::endl;
    }

    /* ───────── LOGGING PARAMETERS ───────── */
    training_report_interval_ =
        settings_file["Record.training_report_interval"].operator int();
    keyframe_record_interval_ =
        settings_file["Record.keyframe_record_interval"].operator int();
    all_keyframes_record_interval_ =
        settings_file["Record.all_keyframes_record_interval"].operator int();
    record_rendered_image_ =
        (settings_file["Record.record_rendered_image"].operator int()) != 0;
    record_ground_truth_image_ =
        (settings_file["Record.record_ground_truth_image"].operator int()) != 0;
    record_loss_image_ =
        (settings_file["Record.record_loss_image"].operator int()) != 0;
    record_loop_ply_ =
        (settings_file["Record.record_loop_ply"].operator int()) != 0;
    rerun_params_.enable_rerun_ =
        (settings_file["Record.enable_rerun"].operator int()) != 0;
    rerun_params_.rerun_max_keyframes_ =
        settings_file["Record.rerun_max_keyframes"].operator int();
    rerun_params_.rerun_keyframe_start_ =
        std::max(0, settings_file["Record.rerun_keyframe_start"].operator int());
    rerun_params_.run_tsdf_pruned_ =
        !settings_file["Record.run_tsdf_pruned"].empty() &&
        (settings_file["Record.run_tsdf_pruned"].operator int()) != 0;
    rerun_params_.rerun_tsdf_unknown_voxels_ =
        !settings_file["Record.run_tsdf_unknown"].empty() &&
        (settings_file["Record.run_tsdf_unknown"].operator int()) != 0;
    rerun_params_.run_floaters_ =
        !settings_file["Record.run_floaters"].empty() &&
        (settings_file["Record.run_floaters"].operator int()) != 0;
    rerun_params_.run_whole_run_ =
        !settings_file["Record.run_whole_run"].empty() &&
        (settings_file["Record.run_whole_run"].operator int()) != 0;
    rerun_params_.run_sdf_pruned_nvblox_ =
        !settings_file["Record.run_sdf_pruned_nvblox"].empty() &&
        (settings_file["Record.run_sdf_pruned_nvblox"].operator int()) != 0;
    rerun_params_.run_floaters_stride_ =
        settings_file["Record.run_floaters_stride"].empty()
            ? 1
            : std::max(1, settings_file["Record.run_floaters_stride"].operator int());
    rerun_params_.rerun_tsdf_pruned_log_gt_mesh_ =
        !settings_file["Record.run_tsdf_pruned_log_gt_mesh"].empty() &&
        (settings_file["Record.run_tsdf_pruned_log_gt_mesh"].operator int()) != 0;
    rerun_params_.rerun_tsdf_pruned_align_gt_to_slam_ =
        !settings_file["Record.run_tsdf_pruned_align_gt_to_slam"].empty() &&
        (settings_file["Record.run_tsdf_pruned_align_gt_to_slam"].operator int()) != 0;
    rerun_params_.rerun_tsdf_pruned_gt_mesh_path_ =
        settings_file["Record.run_tsdf_pruned_gt_mesh_path"].empty()
            ? std::string()
            : settings_file["Record.run_tsdf_pruned_gt_mesh_path"].operator std::string();
    rerun_params_.rerun_tsdf_pruned_gt_traj_path_ =
        settings_file["Record.run_tsdf_pruned_gt_traj_path"].empty()
            ? std::string()
            : settings_file["Record.run_tsdf_pruned_gt_traj_path"].operator std::string();
    rerun_params_.rerun_tsdf_pruned_align_min_pairs_ =
        settings_file["Record.run_tsdf_pruned_align_min_pairs"].empty()
            ? 10
            : std::max(4, settings_file["Record.run_tsdf_pruned_align_min_pairs"].operator int());
    rerun_params_.save_nvblox_mesh_eval_ =
        !settings_file["Record.save_nvblox_mesh_eval"].empty() &&
        (settings_file["Record.save_nvblox_mesh_eval"].operator int()) != 0;
    rerun_params_.load_saved_nvblox_mesh_ =
        !settings_file["Record.load_saved_nvblox_mesh"].empty() &&
        (settings_file["Record.load_saved_nvblox_mesh"].operator int()) != 0;
    rerun_params_.rerun_nvblox_mesh_ =
        !settings_file["Record.rerun_nvblox_mesh"].empty() &&
        (settings_file["Record.rerun_nvblox_mesh"].operator int()) != 0;
    rerun_params_.saved_nvblox_mesh_path_ =
        settings_file["Record.saved_nvblox_mesh_path"].empty()
            ? std::string()
            : settings_file["Record.saved_nvblox_mesh_path"].operator std::string();
    rerun_params_.rerun_gt_mesh_ =
        !settings_file["Record.rerun_gt_mesh"].empty() &&
        (settings_file["Record.rerun_gt_mesh"].operator int()) != 0;
    rerun_params_.rerun_gt_mesh_path_ =
        settings_file["Record.rerun_gt_mesh_path"].empty()
            ? std::string()
            : settings_file["Record.rerun_gt_mesh_path"].operator std::string();
    rerun_params_.save_rendered_mesh_eval_ =
        settings_file["Record.save_rendered_mesh_eval"].empty()
            ? true
            : (settings_file["Record.save_rendered_mesh_eval"].operator int()) != 0;
    rerun_params_.rerun_rendered_mesh_eval_ =
        !settings_file["Record.rerun_rendered_mesh_eval"].empty() &&
        (settings_file["Record.rerun_rendered_mesh_eval"].operator int()) != 0;
    rerun_params_.rendered_mesh_backend_ =
        settings_file["Record.rendered_mesh_backend"].empty()
            ? 0
            : settings_file["Record.rendered_mesh_backend"].operator int();
    rerun_params_.rendered_mesh_backend_ = std::max(0, std::min(1, rerun_params_.rendered_mesh_backend_));
    rerun_params_.rendered_mesh_eval_voxel_size_m_ =
        settings_file["Record.rendered_mesh_eval_voxel_size_m"].empty()
            ? (5.0f / 512.0f)
            : std::max(1.0e-6f, settings_file["Record.rendered_mesh_eval_voxel_size_m"].operator float());
    rerun_params_.rendered_mesh_eval_min_weight_ =
        settings_file["Record.rendered_mesh_eval_min_weight"].empty()
            ? 1.0e-4f
            : std::max(0.0f, settings_file["Record.rendered_mesh_eval_min_weight"].operator float());
    rerun_params_.rerun_reconstruction_mesh_ =
        !settings_file["Record.run_reconstruction_mesh"].empty() &&
        (settings_file["Record.run_reconstruction_mesh"].operator int()) != 0;
    rerun_params_.rerun_reconstruction_mesh_interval_ =
        settings_file["Record.run_reconstruction_mesh_interval"].empty()
            ? 200
            : std::max(1, settings_file["Record.run_reconstruction_mesh_interval"].operator int());
    rerun_params_.rerun_reconstruction_mesh_min_weight_ =
        settings_file["Record.run_reconstruction_mesh_min_weight"].empty()
            ? 1.0e-4f
            : std::max(0.0f, settings_file["Record.run_reconstruction_mesh_min_weight"].operator float());
    rerun_params_.rerun_reconstruction_mesh_weld_vertices_ =
        settings_file["Record.run_reconstruction_mesh_weld_vertices"].empty()
            ? true
            : (settings_file["Record.run_reconstruction_mesh_weld_vertices"].operator int()) != 0;
    rerun_params_.rerun_reconstruction_mesh_max_vertices_ =
        settings_file["Record.run_reconstruction_mesh_max_vertices"].empty()
            ? static_cast<std::size_t>(250000)
            : static_cast<std::size_t>(
                  std::max(0, settings_file["Record.run_reconstruction_mesh_max_vertices"].operator int()));
    rerun_params_.rerun_reconstruction_mesh_max_faces_ =
        settings_file["Record.run_reconstruction_mesh_max_faces"].empty()
            ? static_cast<std::size_t>(500000)
            : static_cast<std::size_t>(
                  std::max(0, settings_file["Record.run_reconstruction_mesh_max_faces"].operator int()));
    rerun_params_.rerun_maps_ =
        !settings_file["Record.run_maps"].empty() &&
        (settings_file["Record.run_maps"].operator int()) != 0;
    rerun_params_.rerun_maps_stride_ =
        settings_file["Record.run_maps_stride"].empty()
            ? 1
            : std::max(1, settings_file["Record.run_maps_stride"].operator int());
    if (!rerun_params_.enable_rerun_) {
        rerun_params_.run_tsdf_pruned_ = false;
        rerun_params_.rerun_tsdf_unknown_voxels_ = false;
        rerun_params_.run_floaters_ = false;
        rerun_params_.run_whole_run_ = false;
        rerun_params_.run_sdf_pruned_nvblox_ = false;
        rerun_params_.rerun_nvblox_mesh_ = false;
        rerun_params_.rerun_gt_mesh_ = false;
        rerun_params_.rerun_rendered_mesh_eval_ = false;
        rerun_params_.rerun_reconstruction_mesh_ = false;
        rerun_params_.rerun_maps_ = false;
    }
    // Viewer Parameters
     rendered_image_viewer_scale_ =
         settings_file["VoxelViewer.image_scale"].operator float();
     rendered_image_viewer_scale_main_ =
         settings_file["VoxelViewer.image_scale_main"].operator float();

}

void VoxelMapper::run()
{
    /* expose our helper scripts to the embedded Python side */
    py::gil_scoped_acquire gil;
    py::module_::import("sys").attr("path").attr("insert")(0, "../scripts_voxel");

    sv::RerunVisualizerBridge::instance().setEnabled(rerun_params_.enable_rerun_);
    if (rerun_params_.enable_rerun_) {
        // Initialize Rerun in "headless" mode (no viewer window).
        sv::RerunVisualizerBridge::instance().init(
            "PhotoSLAM-SVRaster",
            /*spawn_viewer=*/false
        );
    }

    if (rerun_params_.enable_rerun_ && rerun_params_.rerun_gt_mesh_) {
        if (!rerun_params_.rerun_gt_mesh_path_.empty() &&
            std::filesystem::exists(rerun_params_.rerun_gt_mesh_path_)) {
            sv::RerunVisualizerBridge::instance().visualizeNvbloxPlyMesh(
                rerun_params_.rerun_gt_mesh_path_,
                0,
                "world/gt/mesh");
            std::cout << "[RERUN] requested GT mesh logging: "
                      << rerun_params_.rerun_gt_mesh_path_ << "\n";
        } else {
            std::cerr << "[RERUN] GT mesh not found: "
                      << rerun_params_.rerun_gt_mesh_path_ << "\n";
        }
    }

    if (rerun_params_.enable_rerun_ &&
        (rerun_params_.run_tsdf_pruned_ || rerun_params_.rerun_tsdf_unknown_voxels_ || rerun_params_.run_floaters_)) {
        std::string tsdf_pruned_gt_mesh = rerun_params_.rerun_tsdf_pruned_gt_mesh_path_;
        if (tsdf_pruned_gt_mesh.empty()) {
            tsdf_pruned_gt_mesh = rerun_params_.rerun_gt_mesh_path_;
        }
        if (!tsdf_pruned_gt_mesh.empty() &&
            std::filesystem::exists(tsdf_pruned_gt_mesh)) {
            if (rerun_params_.run_tsdf_pruned_ && rerun_params_.rerun_tsdf_pruned_log_gt_mesh_) {
                sv::RerunVisualizerBridge::instance().visualizeDebugGtSdfMesh(
                    "tsdf_pruned",
                    tsdf_pruned_gt_mesh,
                    rerun_params_.rerun_tsdf_pruned_align_gt_to_slam_,
                    rerun_params_.rerun_tsdf_pruned_gt_traj_path_,
                    rerun_params_.rerun_tsdf_pruned_align_min_pairs_,
                    0);
            }
            if (rerun_params_.rerun_tsdf_unknown_voxels_ && rerun_params_.rerun_tsdf_pruned_log_gt_mesh_) {
                sv::RerunVisualizerBridge::instance().visualizeDebugGtSdfMesh(
                    "tsdf_unknown",
                    tsdf_pruned_gt_mesh,
                    rerun_params_.rerun_tsdf_pruned_align_gt_to_slam_,
                    rerun_params_.rerun_tsdf_pruned_gt_traj_path_,
                    rerun_params_.rerun_tsdf_pruned_align_min_pairs_,
                    0);
            }
            if (rerun_params_.run_floaters_) {
                sv::RerunVisualizerBridge::instance().visualizeDebugGtSdfMesh(
                    "floaters",
                    tsdf_pruned_gt_mesh,
                    rerun_params_.rerun_tsdf_pruned_align_gt_to_slam_,
                    rerun_params_.rerun_tsdf_pruned_gt_traj_path_,
                    rerun_params_.rerun_tsdf_pruned_align_min_pairs_,
                    0);
            }
        }
    }

    if (rerun_params_.enable_rerun_ && rerun_params_.rerun_reconstruction_mesh_) {
        std::string reconstruction_gt_mesh = rerun_params_.rerun_tsdf_pruned_gt_mesh_path_;
        if (reconstruction_gt_mesh.empty()) {
            reconstruction_gt_mesh = rerun_params_.rerun_gt_mesh_path_;
        }
        if (!reconstruction_gt_mesh.empty() &&
            std::filesystem::exists(reconstruction_gt_mesh)) {
            sv::RerunVisualizerBridge::instance().visualizeDebugGtSdfMesh(
                "reconstruction_mesh",
                reconstruction_gt_mesh,
                rerun_params_.rerun_tsdf_pruned_align_gt_to_slam_,
                rerun_params_.rerun_tsdf_pruned_gt_traj_path_,
                rerun_params_.rerun_tsdf_pruned_align_min_pairs_,
                0);
        } else if (!reconstruction_gt_mesh.empty()) {
            std::cerr << "[RERUN/reconstruction_mesh] GT mesh not found: "
                      << reconstruction_gt_mesh << "\n";
        }
    }

    if (rerun_params_.enable_rerun_ && rerun_params_.rerun_nvblox_mesh_ && rerun_params_.load_saved_nvblox_mesh_) {
        const std::filesystem::path saved_nvblox_mesh_file =
            voxel_utils::resolveNvbloxMeshPath(rerun_params_.saved_nvblox_mesh_path_);
        if (!saved_nvblox_mesh_file.empty() &&
            std::filesystem::exists(saved_nvblox_mesh_file)) {
            sv::RerunVisualizerBridge::instance().visualizeNvbloxPlyMesh(
                saved_nvblox_mesh_file.string(),
                0,
                "world/nvblox_mesh/reference");
            std::cout << "[NVBLOX] loaded saved reference mesh into Rerun: "
                      << saved_nvblox_mesh_file << "\n";
        } else {
            std::cerr << "[NVBLOX] saved reference mesh not found: "
                      << saved_nvblox_mesh_file << "\n";
        }
    }

    const bool need_nvblox =
        (useNvbloxTsdfBackend() &&
         (sdf_params_.use_tsdf_mapping_ || sdf_params_.use_tsdf_pruning_ || sdf_params_.tsdf_density_init_)) ||
        sdf_params_.use_tsdf_planning_ ||
        rerun_params_.save_nvblox_mesh_eval_ ||
        (rerun_params_.enable_rerun_ && rerun_params_.rerun_reconstruction_mesh_) ||
        (rerun_params_.enable_rerun_ && rerun_params_.run_whole_run_) ||
        (rerun_params_.enable_rerun_ && rerun_params_.run_sdf_pruned_nvblox_) ||
        (rerun_params_.enable_rerun_ && rerun_params_.rerun_nvblox_mesh_ && !rerun_params_.load_saved_nvblox_mesh_);
    if (sensor_type_ == RGBD && need_nvblox)
    {
        initializeNvbloxMapper();
    }
    // First loop: Initial gaussian mapping
    while (!isStopped())
    {
        // Check conditions for initial mapping
        if (hasMetInitialMappingConditions())
        {
            mpSLAM->getAtlas()->clearMappingOperation();

            // Pull sparse SLAM map (get keyframes and map points)
            auto pMap = mpSLAM->getAtlas()->GetCurrentMap();
            std::vector<ORB_SLAM3::KeyFrame*> vKFs;
            std::vector<ORB_SLAM3::MapPoint*> vMPs;
            {
                std::unique_lock<std::mutex> lock_map(pMap->mMutexMapUpdate);
                vKFs = pMap->GetAllKeyFrames();
                vMPs = pMap->GetAllMapPoints();
                for (const auto& pMP : vMPs)
                {
                     Point3D point3D;
                     auto pos = pMP->GetWorldPos();
                     point3D.xyz_(0) = pos.x();
                     point3D.xyz_(1) = pos.y();
                     point3D.xyz_(2) = pos.z();
                     auto color = pMP->GetColorRGB();
                     point3D.color_(0) = color(0);
                     point3D.color_(1) = color(1);
                     point3D.color_(2) = color(2);
                     scene_->cachePoint3D(pMP->mnId, point3D);
                 }
                // B) Create VoxelKeyframes from each SLAM KeyFrame
                for (const auto& pKF : vKFs)
                {
                    std::shared_ptr<VoxelKeyframe> new_kf = std::make_shared<VoxelKeyframe>(pKF->mnId, getIteration());
                    new_kf->source_frame_id_ = voxel_utils::frameIdFromIntegerTimestamp(pKF->mTimeStamp);
                    if (new_kf->source_frame_id_ < 0) {
                        new_kf->source_frame_id_ = voxel_utils::parseFrameIdFromPath(pKF->mNameFile);
                    }
                    new_kf->znear_ = z_near_;
                    // Pose
                    auto pose = pKF->GetPose();
                    new_kf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>()
                    );
                    cv::Mat imgRGB_undistorted, imgAux_undistorted;
                    // Camera
                    sv::Camera& camera = scene_->cameras_.at(pKF->mpCamera->GetId());
                    new_kf->setCameraParams(camera);

                    // Image (left if STEREO)
                    cv::Mat imgRGB = pKF->imgLeftRGB;
                    // camera.undistortImage(imgRGB, imgRGB_undistorted);
                    if (this->sensor_type_ == STEREO)
                            imgRGB_undistorted = imgRGB;
                        else
                            camera.undistortImage(imgRGB, imgRGB_undistorted);
                    // Auxiliary Image
                    cv::Mat imgAux = pKF->imgAuxiliary;
                    if (this->sensor_type_ == RGBD)
                        camera.undistortImage(imgAux, imgAux_undistorted);
                    else
                        imgAux_undistorted = imgAux;

                    new_kf->original_image_ =
                        tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
                    new_kf->img_filename_ = pKF->mNameFile;
                    new_kf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
                    new_kf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
                    new_kf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;

                    // Compute transformations
                    // new_kf->computeTransformTensors(); //useless
                    scene_->addKeyframe(new_kf, &kfid_shuffled_);

                    increaseKeyframeTimesOfUse(new_kf, newKeyframeTimesOfUse());

                    // // Features for increasePcdByKeyframeInactiveGeoDensify
                    std::vector<float> pixels;
                    std::vector<float> pointsLocal;
                    pKF->GetKeypointInfo(pixels, pointsLocal);
                    new_kf->kps_pixel_ = std::move(pixels);
                    new_kf->kps_point_local_ = std::move(pointsLocal);
                    new_kf->img_undist_ = imgRGB_undistorted;
                    new_kf->img_auxiliary_undist_ = imgAux_undistorted;

                    logKeyframeCameraToRerunRecordings(
                        new_kf,
                        pKF->mnId,
                        /*log_reconstruction_mesh=*/false);

                    // ─── TSDF: before SVRaster topology exists, only nvblox can integrate here. ───
                    if (sensor_type_ == RGBD && need_nvblox) {
                        cv::Mat depth_meters;
                        if (voxel_utils::depthMatToMeters(imgAux_undistorted, depth_meters)) {
                            integrateKeyframeIntoNvblox(*new_kf, depth_meters);
                        }
                    }

                }
            }   // Mutex released

            // Prepare multi resolution images for training
            for (auto& kfit : scene_->keyframes()) {
                auto pkf = kfit.second;
                if (device_type_ == torch::kCUDA) {
                    cv::cuda::GpuMat img_gpu;
                    img_gpu.upload(pkf->img_undist_);
                    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                        cv::cuda::GpuMat img_resized;
                        cv::cuda::resize(img_gpu, img_resized,
                                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                        pkf->gaus_pyramid_original_image_[l] =
                            tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
                    }
                }
                else {
                    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                        cv::Mat img_resized;
                        cv::resize(pkf->img_undist_, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                        pkf->gaus_pyramid_original_image_[l] =
                            tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
                    }
                }
            }

            // Create MiniCams for all keyframes and use them for densification later
            std::vector<sv::MiniCam> tr_cams;
            tr_cams.reserve(scene_->keyframes().size());
            for (auto& kv : scene_->keyframes()) {
                // if (kv.second) tr_cams.push_back(kv.second->toMiniCam());
                auto& kf = *kv.second;
                // Use full-res here; you can choose a smaller level if you like.
                tr_cams.emplace_back(kf.toMiniCam(kf.image_height_, kf.image_width_));
            }
            //  Create voxel model & trainer setup
            if (rerun_params_.enable_rerun_) {
                appendAndLogOrbRawMapPcdToRerun(
                    scene_->cached_point_cloud_,
                    getIteration());
            }
            {
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                voxel_model_->createFromPcd(scene_->cached_point_cloud_, tr_cams);
                if (sensor_type_ == RGBD &&
                    sdf_params_.use_tsdf_mapping_ &&
                    useSvrasterTsdfBackend()) {
                    for (const auto& kv : scene_->keyframes()) {
                        if (!kv.second) {
                            continue;
                        }
                        cv::Mat depth_meters;
                        if (voxel_utils::depthMatToMeters(kv.second->img_auxiliary_undist_, depth_meters)) {
                            integrateKeyframeIntoSvrasterSdf(*kv.second, depth_meters);
                        }
                    }
                }
                if (sensor_type_ == RGBD) {
                    const int64_t roots_created = voxel_model_->numVoxels();
                    sdf_state_.tsdf_ablation_rgbd_points_created_ += roots_created;
                    sdf_state_.tsdf_ablation_rgbd_points_lineage_created_ += roots_created;
                }
                std::unique_lock<std::mutex> lock(mutex_settings_);
                voxel_model_->createTrainer(
                                            opt_params_.geo_lr_,
                                            opt_params_.sh0_lr_,
                                            opt_params_.shs_lr_,
                                            opt_params_.optim_beta1_,
                                            opt_params_.optim_beta2_,
                                            opt_params_.optim_eps_,
                                            opt_params_.lr_decay_ckpt_,
                                            opt_params_.lr_decay_mult_);
            }

            if (sensor_type_ == RGBD &&
                inactive_geo_densify_ &&
                rgbd_inactive_geo_initial_backfill_) {
                std::vector<std::shared_ptr<VoxelKeyframe>> initial_rgbd_kfs;
                initial_rgbd_kfs.reserve(scene_->keyframes().size());
                for (const auto& kv : scene_->keyframes()) {
                    if (kv.second && !kv.second->done_inactive_geo_densify_) {
                        initial_rgbd_kfs.push_back(kv.second);
                    }
                }
                for (const auto& pkf : initial_rgbd_kfs) {
                    increasePcdByKeyframeInactiveGeoDensify(pkf);
                }
            }

            if (sensor_type_ == MONOCULAR &&
                mono_prior_params_.depthanything_fill_holes_ &&
                mono_prior_params_.depthanything_fill_holes_initial_backfill_) {
                std::vector<std::shared_ptr<VoxelKeyframe>> initial_fill_kfs;
                initial_fill_kfs.reserve(scene_->keyframes().size());
                for (const auto& kv : scene_->keyframes()) {
                    if (kv.second) {
                        initial_fill_kfs.push_back(kv.second);
                    }
                }

                if (depthAnythingFillHolesWarmupReady()) {
                    applyDepthAnythingFillHolesKeyframes(
                        initial_fill_kfs,
                        /*seed_global_alignment=*/true);
                } else {
                    for (const auto& pkf : initial_fill_kfs) {
                        queueDepthAnythingFillHolesKeyframe(pkf);
                    }
                }
            }

            // One warm-up optimization step
            trainForOneIteration();

            initial_mapped_ = true;
            break;  // Exit the initial mapping loop
        }
        else if (mpSLAM->isShutDown())
        {
            break;
        }
        else
        {
            // Initial conditions not satisfied yet
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Second loop: Incremental voxel mapping
     int SLAM_stop_iter = 0;
     while (!isStopped()) {
        // Check conditions for incremental mapping
        if (hasMetIncrementalMappingConditions()) {
            combineMappingOperations();
            if (cull_keyframes_)
                cullKeyframes();
        }
 
        if (sensor_type_ == MONOCULAR &&
            mono_prior_params_.depthanything_fill_holes_ &&
            mono_prior_params_.depthanything_fill_holes_warmup_) {
            processDepthAnythingFillHolesWarmup();
        }

        // Invoke training once
        trainForOneIteration();

        if (mpSLAM->isShutDown()) {
            SLAM_stop_iter = getIteration();
            SLAM_ended_ = true;
        }

        if (SLAM_ended_ || getIteration() >= opt_params_.iterations_)
            break;
    }
    
    // Third loop: Tail optimization. Keep optimizing the existing voxel
    // parameters after SLAM stops, but freeze topology in this final stage.
    int adapt_interval = opt_params_.adapt_every_;          // cfg.procedure.adapt_every
    int n_delay_iters  = adapt_interval * 0.8f;        // same heuristic as GS code
    const bool prev_disable_topology_changes = disable_topology_changes_;
    disable_topology_changes_ = true;
    while (getIteration() - SLAM_stop_iter <= n_delay_iters
        || (getIteration() % adapt_interval) <= n_delay_iters
        || isKeepingTraining() )
    {
        trainForOneIteration();
        // Re-read in case user changed cfg at runtime
        adapt_interval = opt_params_.adapt_every_;
        n_delay_iters  = adapt_interval * 0.8f;
    }
    disable_topology_changes_ = prev_disable_topology_changes;

    runFinalSpecialPrune();
    if (sensor_type_ == RGBD && sdf_params_.use_tsdf_pruning_) {
        printTsdfPruneAblationSummary("shutdown");
    }

    runTsdfPlannerAtShutdown();
// Final voxel summary at shutdown (for quick diagnostics).
    {
        const int64_t n_total_end = static_cast<int64_t>(voxel_model_->numVoxels());
        int64_t n_above_target_end = 0;
        int64_t n_at_or_below_target_end = 0;

        auto vox_size_end = voxel_model_->voxSize();
                const float target_vox_size = voxel_model_->fixedVoxSize();
                if (vox_size_end.defined()) {
            if (vox_size_end.dim() == 2 && vox_size_end.size(1) == 1) {
                vox_size_end = vox_size_end.squeeze(1);
            } else if (vox_size_end.dim() != 1) {
                vox_size_end = vox_size_end.reshape({-1});
            }
                    if (vox_size_end.numel() == n_total_end) {
                        auto above_target_end =
                    (vox_size_end > target_vox_size).to(torch::kBool);
                        n_above_target_end = above_target_end.sum().item<int64_t>();
                        n_at_or_below_target_end = n_total_end - n_above_target_end;
                    }
        }

        int64_t n_far_end = -1;
        bool far_count_valid = false;
        if (voxel_model_->hasDenseCoreBB()) {
            auto centers_end = voxel_model_->voxCenter();
            auto bb_min_end = voxel_model_->denseCoreBBMin();
            auto bb_max_end = voxel_model_->denseCoreBBMax();
            if (centers_end.defined() && bb_min_end.defined() && bb_max_end.defined() &&
                centers_end.dim() == 2 && centers_end.size(1) == 3 &&
                centers_end.size(0) == n_total_end &&
                bb_min_end.numel() == 3 && bb_max_end.numel() == 3) {
                centers_end = centers_end.to(torch::kFloat32).contiguous();
                bb_min_end = bb_min_end.to(centers_end.device()).to(torch::kFloat32).contiguous().view({1, 3});
                bb_max_end = bb_max_end.to(centers_end.device()).to(torch::kFloat32).contiguous().view({1, 3});
                auto in_dense_core_end =
                    (centers_end >= bb_min_end).all(/*dim=*/1) &
                    (centers_end <= bb_max_end).all(/*dim=*/1);
                auto far_mask_end = (~in_dense_core_end.to(torch::kBool)).contiguous();
                n_far_end = far_mask_end.sum().item<int64_t>();
                far_count_valid = true;
            }
        }

        int64_t n_near_end = -1;
        bool near_count_valid = false;
        if (n_total_end == 0) {
            n_near_end = 0;
            near_count_valid = true;
        } else {
            std::vector<sv::MiniCam> tr_cams;
            tr_cams.reserve(scene_->keyframes().size());
            for (const auto& kv : scene_->keyframes()) {
                tr_cams.push_back(kv.second->toMiniCam(
                    kv.second->image_height_, kv.second->image_width_));
            }

            if (!tr_cams.empty()) {
                try {
                    py::gil_scoped_acquire gil;
                    static py::module svr_mod = py::module::import("svraster_cuda").attr("renderer");
                    static py::module torch_mod = py::module::import("torch");

                    auto svm = voxel_model_->svm();
                    auto octpath = svm.attr("octpath").cast<torch::Tensor>().contiguous();
                    auto vox_center = svm.attr("vox_center").cast<torch::Tensor>().contiguous();
                    auto vox_size = svm.attr("vox_size").cast<torch::Tensor>().contiguous();

                    TORCH_CHECK(octpath.size(0) == n_total_end, "octpath length mismatch at final summary");
                    TORCH_CHECK(vox_center.size(0) == n_total_end, "vox_center length mismatch at final summary");
                    TORCH_CHECK(vox_center.size(1) == 3, "vox_center must be [N,3] at final summary");
                    if (vox_size.dim() == 1) {
                        vox_size = vox_size.view({n_total_end, 1});
                    } else if (vox_size.dim() == 2) {
                        TORCH_CHECK(vox_size.size(0) == n_total_end, "vox_size length mismatch at final summary");
                    } else {
                        TORCH_CHECK(false, "vox_size must be [N] or [N,1] at final summary");
                    }

                    py::list py_cams;
                    py::object py_cuda = torch_mod.attr("device")("cuda");
                    auto move_attr_to_cuda_if_tensor =
                        [&](py::object& obj, const char* name) {
                            if (py::hasattr(obj, name)) {
                                py::object t = obj.attr(name);
                                if (py::hasattr(t, "is_cuda") && !py::bool_(t.attr("is_cuda"))) {
                                    obj.attr(name) = t.attr("to")(py_cuda);
                                }
                            }
                        };

                    for (const auto& c : tr_cams) {
                        py::object py_cam = MiniCam_to_py(c);
                        move_attr_to_cuda_if_tensor(py_cam, "w2c");
                        move_attr_to_cuda_if_tensor(py_cam, "c2w");
                        move_attr_to_cuda_if_tensor(py_cam, "position");
                        move_attr_to_cuda_if_tensor(py_cam, "lookat");
                        py_cams.append(py_cam);
                    }

                    const float near_thresh = 0.2f;
                    at::Tensor is_near = svr_mod.attr("mark_near")(
                        py_cams,
                        py::cast(octpath),
                        py::cast(vox_center),
                        py::cast(vox_size),
                        py::float_(near_thresh)
                    ).cast<at::Tensor>();
                    if (is_near.dim() == 2 && is_near.size(1) == 1) {
                        is_near = is_near.squeeze(1);
                    }
                    auto near_mask_end = is_near.to(torch::kBool).contiguous();
                    n_near_end = near_mask_end.sum().item<int64_t>();
                    near_count_valid = true;
                } catch (const std::exception& e) {
                    std::cerr << "[FINAL/vox] failed to compute near count: "
                              << e.what() << "\n";
                }
            } else {
                n_near_end = 0;
                near_count_valid = true;
            }
        }

        std::cout << "[FINAL/vox] total=" << n_total_end
                  << " above_target=" << n_above_target_end
                  << " at_or_below_target=" << n_at_or_below_target_end;
        if (far_count_valid) {
            std::cout << " far=" << n_far_end;
        } else {
            std::cout << " far=N/A";
        }
        if (near_count_valid) {
            std::cout << " near=" << n_near_end;
        } else {
            std::cout << " near=N/A";
        }
        std::cout << " target_vox_size=" << target_vox_size
                  << "\n";
    }

    // Save and clear
    renderAndRecordAllKeyframes("_shutdown");
    savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
    {
        const std::filesystem::path ply_dir =
            result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply" /
            "voxel_model" / ("iteration_" + std::to_string(getIteration()));
        const std::filesystem::path eval_mesh_path = ply_dir / "voxel_surface_mesh.ply";

        const bool want_rendered_mesh =
            rerun_params_.save_rendered_mesh_eval_ || (rerun_params_.enable_rerun_ && rerun_params_.rerun_rendered_mesh_eval_);
        if (want_rendered_mesh) {
            try {
                saveRenderedTsdfMeshPly(eval_mesh_path);
                if (rerun_params_.enable_rerun_ && rerun_params_.rerun_rendered_mesh_eval_ &&
                    std::filesystem::exists(eval_mesh_path)) {
                    sv::RerunVisualizerBridge::instance().visualizeNvbloxPlyMesh(
                        eval_mesh_path.string(),
                        getIteration(),
                        "world/voxel_model_mesh/final");
                }
            } catch (const std::exception& e) {
                std::cerr << "[saveRenderedTsdfMeshPly] shutdown export failed: "
                          << e.what() << "\n";
            }
        } else {
            std::cout << "[saveRenderedTsdfMeshPly] skipped: disabled by "
                         "Record.save_rendered_mesh_eval and Record.rerun_rendered_mesh_eval.\n";
        }
    }
    {
        const bool want_nvblox_mesh =
            sensor_type_ == RGBD &&
            !rerun_params_.load_saved_nvblox_mesh_ &&
            (rerun_params_.save_nvblox_mesh_eval_ ||
             (rerun_params_.enable_rerun_ && rerun_params_.rerun_nvblox_mesh_) ||
             (rerun_params_.enable_rerun_ && rerun_params_.run_sdf_pruned_nvblox_));
        if (want_nvblox_mesh && sdf_mapper_) {
            try {
                sdf_mapper_->updateColorMesh();
                std::filesystem::path nvblox_mesh_path =
                    voxel_utils::resolveNvbloxMeshPath(rerun_params_.saved_nvblox_mesh_path_);
                if (nvblox_mesh_path.empty()) {
                    const std::filesystem::path nvblox_dir =
                        result_dir_ / (std::to_string(getIteration()) + "_shutdown") /
                        "ply" / "nvblox";
                    nvblox_mesh_path = nvblox_dir / "nvblox_color_mesh.ply";
                }
                std::filesystem::create_directories(nvblox_mesh_path.parent_path());

                nvblox::io::outputColorMeshLayerToPly(
                    sdf_mapper_->color_mesh_layer(),
                    nvblox_mesh_path.string());

                if (rerun_params_.enable_rerun_ && rerun_params_.rerun_nvblox_mesh_) {
                    sv::RerunVisualizerBridge::instance().visualizeNvbloxPlyMesh(
                        nvblox_mesh_path.string(),
                        getIteration(),
                        "world/nvblox_mesh/final");
                }
                if (rerun_params_.enable_rerun_ && rerun_params_.run_sdf_pruned_nvblox_) {
                    sv::RerunVisualizerBridge::instance().visualizeDebugNvbloxPlyMesh(
                        "sdf_pruned_nvblox",
                        nvblox_mesh_path.string(),
                        getIteration(),
                        "world/nvblox_mesh/final");
                }

                std::cout << "[NVBLOX] saved final mesh: "
                          << nvblox_mesh_path << "\n";
            } catch (const std::exception& e) {
                std::cerr << "[NVBLOX] final mesh export failed: "
                          << e.what() << "\n";
            }
        } else if (rerun_params_.save_nvblox_mesh_eval_) {
            std::cout << "[NVBLOX] final mesh skipped: nvblox mapper was not initialized.\n";
        }
    }
    if (sdf_params_.use_tsdf_planning_) {
        voxel_model_->savePlannerNPZ(
            result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "planner.npz");
    }
    writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

    saveRerunRecordingsAtShutdown();

    signalStop();
 }

// Add training with optimization loop
void VoxelMapper::trainForOneIteration()
{
    // 1) bump global iteration counter
    increaseIteration(1);
    auto iter_start_timing = std::chrono::steady_clock::now();

    sv::RenderOpts ropts;

    // 2) pick a random keyframe from the sliding window
    std::shared_ptr<VoxelKeyframe> viewpoint_cam = useOneRandomSlidingWindowKeyframe();
    if (!viewpoint_cam) {
        increaseIteration(-1);
        return;
    }
    writeKeyframeUsedTimes(result_dir_ / "used_times");

    const int iter = getIteration();
    int training_level = num_gaus_pyramid_sub_levels_;
    int image_height, image_width;
    torch::Tensor gt_image, mask;

    if (isdoingGausPyramidTraining())
         training_level = viewpoint_cam->getCurrentGausPyramidLevel();
    if (training_level == num_gaus_pyramid_sub_levels_) {
        image_height = viewpoint_cam->image_height_;
        image_width = viewpoint_cam->image_width_;
        gt_image = viewpoint_cam->original_image_
                            .to(mDevice);          // (3,H,W)
        mask = undistort_mask_[viewpoint_cam->camera_id_]
                                    .to(mDevice)
                                    .to(torch::kFloat32); // (3,H,W)
    }
    else {
        image_height = viewpoint_cam->gaus_pyramid_height_[training_level];
        image_width = viewpoint_cam->gaus_pyramid_width_[training_level];
        gt_image = viewpoint_cam->gaus_pyramid_original_image_[training_level].to(mDevice); 
        mask = scene_->cameras_.at(viewpoint_cam->camera_id_).gaus_pyramid_undistort_mask_[training_level].to(mDevice).to(torch::kFloat32);
    }
    // 4) grow SH degree every 1000 iterations (locked during render)
    std::unique_lock<std::mutex> lock_render(mutex_render_);
    if (getIteration() % 1000 == 0 && default_sh_ < model_params_.sh_degree_)
    {
        default_sh_ += 1;
        std::cout << "[VoxelMapper] SH degree: " << default_sh_ << std::endl;
    }    
    voxel_model_->setShDegree(default_sh_);

    // // Follow SVRaster train.py:
    // // keep ss=1.0 early, then switch to augmentation or the model default.
    // ropts.ss = 1.0f;
    // if (iter > 1000) {
    //     if (opt_params_.ss_aug_max_ > 1.0f) {
    //         static thread_local std::mt19937 rng{std::random_device{}()};
    //         std::uniform_real_distribution<float> dist(1.0f, opt_params_.ss_aug_max_);
    //         ropts.ss = dist(rng);
    //     } else {
    //         ropts.ss = std::nullopt;
    //     }
    // }

    // Use default super-sampling option (enable after 1000 iters)
    if (iter > 200) {
        if (opt_params_.ss_aug_max_ > 1.0f) {
            static thread_local std::mt19937 rng{std::random_device{}()};
            std::uniform_real_distribution<float> dist(1.0f, opt_params_.ss_aug_max_);
            ropts.ss = dist(rng);                 // tr_render_opt['ss'] = U(1, ss_aug_max)
        } else {
            ropts.ss = std::nullopt;              // pop('ss') -> use model default self.ss
        }
    } else {
        ropts.ss = 1.0f;                           // disable supersampling at first
    }
    ropts.ss = 1.0f;  

    const bool need_sparse_depth = (opt_params_.lambda_sparse_depth_ > 0.0f) && (iter <= opt_params_.sparse_depth_until_);
    const bool need_depthanythingv2 =
        (opt_params_.lambda_depthanythingv2_ > 0.0f) &&
        (iter >= opt_params_.depthanythingv2_from_) &&
        (iter <= opt_params_.depthanythingv2_end_);
    const bool need_rgbd_depth =
        (sensor_type_ == RGBD) &&
        (opt_params_.lambda_rgbd_depth_ > 0.0f) &&
        (iter >= opt_params_.rgbd_depth_from_) &&
        (iter <= opt_params_.rgbd_depth_end_);
    const bool need_mono_prior_normal =
        (opt_params_.lambda_depthanythingv2_normal_ > 0.0f) &&
        (iter >= opt_params_.depthanythingv2_normal_from_) &&
        (iter <= opt_params_.depthanythingv2_normal_end_);
    const bool need_rgbd_normal =
        (sensor_type_ == RGBD) &&
        (opt_params_.lambda_rgbd_normal_ > 0.0f) &&
        (iter >= opt_params_.rgbd_normal_from_) &&
        (iter <= opt_params_.rgbd_normal_end_);
    const bool need_T_concen = (opt_params_.lambda_T_concen_ > 0.0f);
    const bool need_T_inside = (opt_params_.lambda_T_inside_ > 0.0f);
    const bool need_normal_dmean =
        (opt_params_.lambda_normal_dmean_ > 0.0f) &&
        (iter >= opt_params_.n_dmean_from_) &&
        (iter <= opt_params_.n_dmean_end_);
    ropts.output_T =
        need_T_concen || need_T_inside || need_sparse_depth || need_normal_dmean ||
        need_depthanythingv2 || need_rgbd_depth || need_mono_prior_normal ||
        need_rgbd_normal;
    ropts.output_depth = need_sparse_depth || need_normal_dmean || need_depthanythingv2 ||
                         need_rgbd_depth || need_mono_prior_normal;
    ropts.output_normal = need_normal_dmean || need_mono_prior_normal || need_rgbd_normal;

    // if (opt_params_.lambda_T_inside_ > 0.0f) {
    //     ropts.output_T = true;
    // }

    if (iter >= opt_params_.dist_from_ && opt_params_.lambda_dist_ > 0.0f) {
        ropts.lambda_dist = opt_params_.lambda_dist_;
    }

    if (iter >= opt_params_.rectifiy_from_ &&
        opt_params_.lambda_rectify_ > 0.0f) {
        ropts.lambda_ascending = -opt_params_.lambda_rectify_;
    } else if (iter >= opt_params_.ascending_from_ &&
               opt_params_.lambda_ascending_ > 0.0f) {
        ropts.lambda_ascending = opt_params_.lambda_ascending_;
    }

    if (iter > opt_params_.scaling_penalty_from_ &&
        iter <= opt_params_.scaling_penalty_end_ &&
        opt_params_.lambda_scaling_penalty_ > 0.0f) {
        auto vox_size = voxel_model_->voxSize();
        if (vox_size.defined() && vox_size.numel() > 0) {
            ropts.lambda_scaling_penalty = opt_params_.lambda_scaling_penalty_;
            ropts.min_voxel_size =
                vox_size.to(torch::kFloat32).min().item<float>();
        }
    }

    if (opt_params_.lambda_R_concen_ > 0.0f) {
        ropts.lambda_R_concen = opt_params_.lambda_R_concen_;
        ropts.gt_color = gt_image;
    }

    if (mono_prior_params_.mono_prior_loss_mode_ == "geosvr" &&
        need_depthanythingv2 &&
        opt_params_.enable_da2_uncertainty_ &&
        iter >= opt_params_.level_uncertainty_from_) {
        ropts.vox_feats = voxel_model_->octLevel().detach().to(torch::kFloat32).contiguous();
    }

    sv::MiniCam cam = viewpoint_cam->toMiniCam(image_height, image_width);

    auto render_pkg = voxel_model_->render(
        cam,
        image_height,
        image_width,
        /* gt_image   */  gt_image,            
        /* color_mode   */   nullptr,             
        /* track_max_w   */  true,
        /* ss            */  std::nullopt,
        /* output_depth  */  ropts.output_depth,
        /* output_normal */  ropts.output_normal,
        /* output_T      */  ropts.output_T,
        /* rand_bg       */  false,
        /* use_auto_exp  */  false,
        ropts               // your struct (will be used for **other_opt-safe fields)
    );
    if (render_pkg.empty() || !render_pkg.count("color") || !render_pkg.at("color").defined()) {
        return;
    }

    torch::Tensor rendered_image = render_pkg["color"].to(mDevice);
    torch::Tensor masked_image = rendered_image * mask;      // (1,3,H,W)

    // after render_pkg & rendered_image
    torch::Tensor depth_for_viz;   // declare here so it's visible later
    auto it_depth = render_pkg.find("depth");
    if (it_depth != render_pkg.end() && it_depth->second.defined()) {
        depth_for_viz = it_depth->second;  // keep on device for now
    }

    auto Ll1 = loss_utils::l1_loss(masked_image, gt_image);
    auto mse = loss_utils::l2_loss(masked_image, gt_image);

    // Match SVRaster's base photometric loss selection: L1, Huber, or MSE.
    torch::Tensor photo_loss;
    if (opt_params_.use_l1_) {
        photo_loss = Ll1;
    } else if (opt_params_.use_huber_) {
        photo_loss = loss_utils::huber_loss(masked_image, gt_image, opt_params_.huber_thres_);
    } else {
        photo_loss = mse;
    }
    auto loss = photo_loss.clone();

    // optional use loss from the original photoslam paper
    float lambda_dssim = lambdaDssim();
    auto photoslam_loss = (1.0 - lambda_dssim) * Ll1
            + lambda_dssim * (1.0 - loss_utils::ssim(masked_image, gt_image, mDevice.type()));

    // --- Sparse depth regularization (SVRaster-style) ----------------------------
    if (need_sparse_depth) {
        torch::Tensor depth_loss =
            computeSparseDepthLoss_Points(
            viewpoint_cam,   // which KF we are training on
            cam,             // MiniCam for this KF at current pyramid level
            image_width,
            image_height,
            render_pkg,
            iter);

        loss = loss + opt_params_.lambda_sparse_depth_ * depth_loss;
    }
    if (need_depthanythingv2) {
        torch::Tensor dense_depth_loss = computeMonoPriorDepthLoss(
            viewpoint_cam,
            cam,
            render_pkg,
            iter);

        loss = loss + opt_params_.lambda_depthanythingv2_ * dense_depth_loss;
    }
    if (need_rgbd_depth) {
        torch::Tensor rgbd_depth_loss =
            computeRgbdDepthLoss(viewpoint_cam, cam, render_pkg, iter);

        loss = loss + opt_params_.lambda_rgbd_depth_ * rgbd_depth_loss;
    }
    if (need_mono_prior_normal) {
        torch::Tensor dense_normal_loss = computeMonoPriorNormalLoss(
            viewpoint_cam,
            cam,
            render_pkg,
            iter);

        loss = loss + opt_params_.lambda_depthanythingv2_normal_ * dense_normal_loss;
    }
    if (need_rgbd_normal) {
        torch::Tensor rgbd_normal_loss =
            computeRgbdNormalLoss(viewpoint_cam, cam, render_pkg, iter);

        loss = loss + opt_params_.lambda_rgbd_normal_ * rgbd_normal_loss;
    }
    if (opt_params_.lambda_ssim_ > 0.0f) {
        loss += opt_params_.lambda_ssim_ * loss_utils::fast_ssim_loss(masked_image, gt_image);
    }

    if (need_T_concen || need_T_inside) {
        auto it = render_pkg.find("raw_T");
        if (it != render_pkg.end() && it->second.defined()) {
            torch::Tensor raw_T = it->second;

            // SVRaster: loss += lambda_T_concen * prob_concen_loss(raw_T)
            if (need_T_concen) {
                torch::Tensor reg_concen = loss_utils::prob_concen_loss(raw_T);
                loss = loss + opt_params_.lambda_T_concen_ * reg_concen;
            }

            // SVRaster: loss += lambda_T_inside * raw_T.square().mean()
            if (need_T_inside) {
                torch::Tensor reg_inside = raw_T.pow(2).mean();
                loss = loss + opt_params_.lambda_T_inside_ * reg_inside;
            }
        }
    }

    if (need_normal_dmean) {
        auto reg_normal_dmean = voxel_eval::normalDepthConsistencyLossSVRaster(
            cam,
            render_pkg,
            opt_params_.n_dmean_ks_,
            opt_params_.n_dmean_tol_deg_);
        loss = loss + opt_params_.lambda_normal_dmean_ * reg_normal_dmean;
    }

    voxel_model_->optimizerZeroGrad();   // move this BEFORE backward
    {
        py::gil_scoped_release no_gil;
        loss.backward();
    }

    if (opt_params_.lambda_tv_density_ > 0.f &&
        iter >= opt_params_.tv_from_ &&
        iter <= opt_params_.tv_until_) {
        voxel_model_->applyTvOnDensityField(opt_params_.lambda_tv_density_);
    }

    voxel_model_->optimizerStep();   // <-- the actual update

    // --- debug: store near voxels for rerun ---
    torch::Tensor debug_near_centers;  // [K,3]
    torch::Tensor debug_near_sizes;    // [K,1] or [K]
    bool debug_has_near = false;
    torch::Tensor debug_near_geom_centers;  // [K_geom,3]
    torch::Tensor debug_near_geom_sizes;    // [K_geom,1] or [K_geom]
    bool debug_has_near_geom = false;
    torch::Tensor debug_tsdf_free_centers;
    torch::Tensor debug_tsdf_free_sizes;
    bool debug_has_tsdf_free = false;
    torch::Tensor debug_tsdf_occupied_centers;
    torch::Tensor debug_tsdf_occupied_sizes;
    bool debug_has_tsdf_occupied = false;
    torch::Tensor debug_tsdf_surface_centers;
    torch::Tensor debug_tsdf_surface_sizes;
    bool debug_has_tsdf_surface = false;
    torch::Tensor debug_tsdf_unknown_centers;
    torch::Tensor debug_tsdf_unknown_sizes;
    bool debug_has_tsdf_unknown = false;
    torch::Tensor debug_tsdf_free_corner_points;
    torch::Tensor debug_tsdf_occupied_corner_points;
    torch::Tensor debug_tsdf_surface_corner_points;
    torch::Tensor debug_tsdf_unknown_corner_points;
    torch::Tensor debug_tsdf_unknown_corner_colors;
    bool debug_has_tsdf_free_corners = false;
    bool debug_has_tsdf_occupied_corners = false;
    bool debug_has_tsdf_surface_corners = false;
    bool debug_has_tsdf_unknown_corners = false;
    torch::Tensor debug_pruned_centers; // [K_prune,3]
    torch::Tensor debug_pruned_sizes;   // [K_prune,1] or [K_prune]
    bool debug_has_pruned = false;
    torch::Tensor debug_far_pruned_centers; // [K_far,3]
    torch::Tensor debug_far_pruned_sizes;   // [K_far,1] or [K_far]
    bool debug_has_far_pruned = false;
    const bool use_artificial_voxels = fill_empty_cells_;
    if (!disable_topology_changes_) {
        // Densification for increasePcd
        const int prune_every =
            std::max(1, (opt_params_.prune_every_ > 0) ? opt_params_.prune_every_ : opt_params_.adapt_every_);
        const int subdivide_every =
            std::max(1, (opt_params_.subdivide_every_ > 0) ? opt_params_.subdivide_every_ : opt_params_.adapt_every_);
        const bool meet_prune_period =
            (iter >= opt_params_.adapt_from_) && (iter % prune_every == 0);
        const bool meet_subdivide_period =
            (iter >= opt_params_.adapt_from_) && (iter % subdivide_every == 0);

        bool need_pruning =
            meet_prune_period && (iter <= opt_params_.prune_until_);
        bool need_subdividing =
            meet_subdivide_period &&
            (iter <= opt_params_.subdivide_until_) &&
            (voxel_model_->numVoxels() < opt_params_.subdivide_max_num_);

        if (need_pruning || need_subdividing)
        {
            // // Build list of training cameras (use all current keyframes)
            std::vector<sv::MiniCam> tr_cams; 
            tr_cams.reserve(scene_->keyframes().size());
            for (auto& kv : scene_->keyframes()) {
                if (kv.second) {
                    tr_cams.push_back(
                        kv.second->toMiniCam(kv.second->image_height_, kv.second->image_width_));
                }
            }

            auto stat = voxel_model_->computeTrainingStat(tr_cams);
            py::object sched_state = voxel_model_->schedulerStateDict();
            auto flatten_colvec = [](torch::Tensor t) {
                if (t.defined() && t.dim() == 2 && t.size(1) == 1) {
                    t = t.squeeze(1);
                }
                return t.contiguous().view({-1});
            };

            // ---------------- PRUNE ----------------
            auto run_pruning = [&]() {
                // 0) Refresh training statistics if topology changed since they were computed.
                {
                    const int N_cur = voxel_model_->numVoxels();
                    const bool stat_shape_ok =
                        stat.max_w.defined() &&
                        stat.min_samp_interval.defined() &&
                        stat.view_cnt.defined() &&
                        stat.max_w.size(0) == N_cur &&
                        stat.min_samp_interval.size(0) == N_cur &&
                        stat.view_cnt.size(0) == N_cur;
                    if (!stat_shape_ok) {
                        stat = voxel_model_->computeTrainingStat(tr_cams);
                    }
                }
                const float t1 = opt_params_.prune_thres_final_;
                const float ta1 = opt_params_.prune_thres_final_artificial_;
                const float prune_thres = t1; // fixed threshold
                // Separate threshold for at-target voxels.
                const float prune_thres_at_target = opt_params_.prune_thres_final_at_target_;
                const float prune_thres_artificial = ta1; // fixed threshold
                const int ori_n = voxel_model_->numVoxels();
                const int N     = ori_n;
                const float target_vox_size = voxel_model_->fixedVoxSize();
                torch::Tensor prune_mask_vis; // [N] bool, set when visibility filter runs
                torch::Tensor prune_mask_near; // [N] bool, near-camera prune (unprotected)
                torch::Tensor prune_mask_recent_unstable; // [N] bool, young unstable voxels
                torch::Tensor prune_mask_default;         // [N] bool, default rules only
                torch::Tensor prune_mask_real_outside_dense_core; // [N] bool, real outliers outside dense-core
                torch::Tensor prune_mask_gslam_unstable;  // [N] bool, GS-SLAM unstable (real + artificial)
                torch::Tensor prune_mask_gslam_real;      // [N] bool, GS-SLAM unstable real only
                torch::Tensor prune_mask_gslam_artificial;// [N] bool, GS-SLAM unstable artificial only

                // 1) Base Photo-SLAM / SVRaster pruning by max weight.
                auto max_w_1d = flatten_colvec(stat.max_w).to(torch::kFloat32); // [N]
                torch::Tensor art_mask_for_base;
                if (use_artificial_voxels) {
                    art_mask_for_base = voxel_model_->artificialMask();
                    if (art_mask_for_base.defined()) {
                        art_mask_for_base = flatten_colvec(
                            art_mask_for_base.to(max_w_1d.device()).to(torch::kBool));
                    }
                }
                if (!use_artificial_voxels ||
                    !art_mask_for_base.defined() ||
                    art_mask_for_base.numel() != N) {
                    art_mask_for_base = torch::zeros(
                        {N},
                        torch::TensorOptions().dtype(torch::kBool).device(max_w_1d.device()));
                }
                const bool use_rendered_depth_candidates =
                    rendered_depth_insert_ || mono_prior_params_.depthanything_fill_holes_;
                auto rendered_depth_candidate_young_protect =
                    torch::zeros(
                        {N},
                        torch::TensorOptions().dtype(torch::kBool).device(max_w_1d.device()));
                auto rendered_depth_candidate_stale_prune =
                    torch::zeros(
                        {N},
                        torch::TensorOptions().dtype(torch::kBool).device(max_w_1d.device()));
                if (use_rendered_depth_candidates) {
                    auto rendered_depth_candidate_mask = voxel_model_->renderedDepthCandidateMask();
                    if (rendered_depth_candidate_mask.defined()) {
                        rendered_depth_candidate_mask = flatten_colvec(
                            rendered_depth_candidate_mask.to(max_w_1d.device()).to(torch::kBool));
                    }
                    if (!rendered_depth_candidate_mask.defined() ||
                        rendered_depth_candidate_mask.numel() != N) {
                        rendered_depth_candidate_mask = torch::zeros(
                            {N},
                            torch::TensorOptions().dtype(torch::kBool).device(max_w_1d.device()));
                    }
                    auto rendered_depth_support_count = voxel_model_->renderedDepthCandidateSupportCount();
                    if (rendered_depth_support_count.defined()) {
                        rendered_depth_support_count = flatten_colvec(
                            rendered_depth_support_count.to(max_w_1d.device()).to(torch::kInt32));
                    }
                    if (!rendered_depth_support_count.defined() ||
                        rendered_depth_support_count.numel() != N) {
                        rendered_depth_support_count = torch::zeros(
                            {N},
                            torch::TensorOptions().dtype(torch::kInt32).device(max_w_1d.device()));
                    }
                    auto rendered_depth_last_seen_kf = voxel_model_->renderedDepthCandidateLastSeenKf();
                    if (rendered_depth_last_seen_kf.defined()) {
                        rendered_depth_last_seen_kf = flatten_colvec(
                            rendered_depth_last_seen_kf.to(max_w_1d.device()).to(torch::kInt32));
                    }
                    if (!rendered_depth_last_seen_kf.defined() ||
                        rendered_depth_last_seen_kf.numel() != N) {
                        rendered_depth_last_seen_kf = torch::full(
                            {N},
                            static_cast<int32_t>(-1),
                            torch::TensorOptions().dtype(torch::kInt32).device(max_w_1d.device()));
                    }
                    const int32_t rendered_depth_kf_now = static_cast<int32_t>(tr_cams.size());
                    auto rendered_depth_under_support =
                        (rendered_depth_support_count <
                         opt_params_.rendered_depth_candidate_promote_min_support_).to(torch::kBool);
                    auto rendered_depth_seen_valid =
                        (rendered_depth_last_seen_kf >= 0).to(torch::kBool);
                    auto rendered_depth_age_kf =
                        (torch::full(
                            {N},
                            rendered_depth_kf_now,
                            torch::TensorOptions().dtype(torch::kInt32).device(max_w_1d.device())) -
                         rendered_depth_last_seen_kf).to(torch::kInt32);
                    rendered_depth_candidate_young_protect =
                        (rendered_depth_candidate_mask &
                         rendered_depth_under_support &
                         rendered_depth_seen_valid &
                         (rendered_depth_age_kf < opt_params_.rendered_depth_candidate_prune_kf_age_))
                            .to(torch::kBool);
                    rendered_depth_candidate_stale_prune =
                        (rendered_depth_candidate_mask &
                         rendered_depth_under_support &
                         rendered_depth_seen_valid &
                         (rendered_depth_age_kf >= opt_params_.rendered_depth_candidate_prune_kf_age_))
                            .to(torch::kBool);
                }
                auto real_mask_for_base = (~art_mask_for_base).to(torch::kBool);
                auto in_target_size_mask = torch::zeros(
                    {N},
                    torch::TensorOptions().dtype(torch::kBool).device(max_w_1d.device()));
                auto vox_size_1d = voxel_model_->voxSize();
                if (vox_size_1d.defined()) {
                    if (vox_size_1d.dim() == 2 && vox_size_1d.size(1) == 1) {
                        vox_size_1d = vox_size_1d.squeeze(1);
                    } else if (vox_size_1d.dim() != 1) {
                        vox_size_1d = vox_size_1d.reshape({-1});
                    }
                    vox_size_1d = vox_size_1d.to(max_w_1d.device()).to(torch::kFloat32).contiguous();
                    if (vox_size_1d.numel() == N) {
                        in_target_size_mask =
                            (vox_size_1d <= target_vox_size).to(torch::kBool);
                    }
                }
                const int64_t n_in_target_size = in_target_size_mask.sum().item<int64_t>();
                const float prune_thres_real_in_target =
                    std::max(prune_thres, prune_thres_at_target);
                const float prune_thres_artificial_in_target =
                    std::max(prune_thres_artificial, prune_thres_at_target);
                auto max_w_real = max_w_1d.to(torch::kFloat32).contiguous();
                auto prune_thres_real_vec =
                    torch::full({N}, prune_thres,
                        torch::TensorOptions().dtype(torch::kFloat32).device(max_w_1d.device()));
                auto prune_thres_artificial_vec =
                    torch::full({N}, prune_thres_artificial,
                        torch::TensorOptions().dtype(torch::kFloat32).device(max_w_1d.device()));
                if (n_in_target_size > 0) {
                    prune_thres_real_vec = torch::where(
                        in_target_size_mask,
                        torch::full_like(prune_thres_real_vec, prune_thres_real_in_target),
                        prune_thres_real_vec);
                    prune_thres_artificial_vec = torch::where(
                        in_target_size_mask,
                        torch::full_like(prune_thres_artificial_vec, prune_thres_artificial_in_target),
                        prune_thres_artificial_vec);
                }
                auto prune_mask_base_real_raw =
                    ((max_w_1d < prune_thres_real_vec) & real_mask_for_base).to(torch::kBool);
                auto prune_mask_base_artificial_raw =
                    use_artificial_voxels
                        ? ((max_w_1d < prune_thres_artificial_vec) & art_mask_for_base).to(torch::kBool)
                        : torch::zeros(
                              {N},
                              torch::TensorOptions().dtype(torch::kBool).device(max_w_1d.device()));
                prune_mask_base_artificial_raw =
                    (prune_mask_base_artificial_raw & (~rendered_depth_candidate_young_protect))
                        .to(torch::kBool);

                auto prune_mask_base_real = prune_mask_base_real_raw.to(torch::kBool);
                auto prune_mask_base_artificial = prune_mask_base_artificial_raw.to(torch::kBool);
                auto prune_mask_base_real_at_target =
                    (prune_mask_base_real & in_target_size_mask).to(torch::kBool);
                auto prune_mask_base_artificial_at_target =
                    (prune_mask_base_artificial & in_target_size_mask).to(torch::kBool);
                auto prune_mask_base_real_at_target_extra =
                    ((max_w_real >= prune_thres) &
                     (max_w_real < prune_thres_real_in_target) &
                     real_mask_for_base &
                     in_target_size_mask).to(torch::kBool);
                auto prune_mask_base_artificial_at_target_extra =
                    ((max_w_real >= prune_thres_artificial) &
                     (max_w_real < prune_thres_artificial_in_target) &
                     art_mask_for_base &
                     in_target_size_mask).to(torch::kBool);
                torch::Tensor prune_mask_base =
                    (prune_mask_base_real |
                     prune_mask_base_artificial |
                     rendered_depth_candidate_stale_prune).to(torch::kBool); // [N]
                auto prune_mask = prune_mask_base.clone(); // [N] bool
                const int n_prune_base =
                    prune_mask_base.defined()
                        ? (int)prune_mask_base.sum().item<int64_t>()
                        : -1;

                // NEW: declare tsdf_prune_mask here, default undefined
                torch::Tensor tsdf_prune_mask;
                torch::Tensor tsdf_debug_corner_points_all;
                torch::Tensor tsdf_debug_values_all;
                torch::Tensor tsdf_debug_weights_all;
                torch::Tensor tsdf_surface_protect_mask;
                // 2) Optional TSDF/SDF pruning: prune voxels classified as strong free-space.
                if (sensor_type_ == RGBD && sdf_params_.use_tsdf_pruning_) {
                    if (N > 0 && hasTsdfForSampling()) {
                        try {
                            torch::Tensor centers_world = voxel_model_->voxCenter(); // [N,3]
                            torch::Tensor sizes_world   = voxel_model_->voxSize();   // [N,1] (your implementation)

                            if (centers_world.defined() &&
                                centers_world.dim() == 2 &&
                                centers_world.size(0) == N &&
                                centers_world.size(1) == 3 &&
                                sizes_world.defined() &&
                                sizes_world.size(0) == N)
                            {
                                // Sample TSDF at 8 corners per voxel
                                // TsdfCornerSample c = sampleTsdfAtVoxelCornersWorld(centers_world, sizes_world);
                                TsdfCornerSample c = sampleTsdfAtSvrasterGridCornersWorld();
                                torch::Tensor tsdf8   = c.tsdf;    // [N,8]
                                torch::Tensor w8      = c.weight;  // [N,8]
                                torch::Tensor ok8     = c.success; // [N,8] bool

                                // Device alignment (should already match)
                                if (tsdf8.device() != prune_mask.device()) {
                                    tsdf8 = tsdf8.to(prune_mask.device());
                                    w8    = w8.to(prune_mask.device());
                                    ok8   = ok8.to(prune_mask.device());
                                }
                                tsdf_debug_corner_points_all = c.points_world;
                                tsdf_debug_values_all = tsdf8;
                                tsdf_debug_weights_all = w8;

                                // SVRecon-style SDF pruning:
                                // - gather 8 corner SDF values per voxel
                                // - keep zero-crossing voxels
                                // - keep voxels close to the zero surface
                                // - prune only sufficiently observed voxels far from the zero surface
                                const float min_weight = sdf_params_.tsdf_prune_min_weight_;
                                const float tau_surface =
                                    std::max(0.0f, sdf_params_.tsdf_prune_surface_band_vox_) *
                                    tsdfMetricVoxelSize();
                                const float sign_eps = 1.0e-6f;

                                torch::Tensor corner_valid = (ok8 & (w8 >= min_weight)).to(torch::kBool); // [N,8]
                                torch::Tensor valid_count =
                                    corner_valid.to(torch::kInt32).sum(/*dim=*/1); // [N]
                                torch::Tensor voxel_valid =
                                    (valid_count >= sdf_params_.tsdf_prune_min_valid_corners_).to(torch::kBool); // [N]

                                torch::Tensor has_pos =
                                    ((tsdf8 > sign_eps) & corner_valid).any(/*dim=*/1); // [N]
                                torch::Tensor has_neg =
                                    ((tsdf8 < -sign_eps) & corner_valid).any(/*dim=*/1); // [N]
                                torch::Tensor has_surface =
                                    (has_pos & has_neg).to(torch::kBool); // [N], zero crossing in valid corners

                                torch::Tensor abs_tsdf = tsdf8.abs();
                                torch::Tensor inf_like =
                                    torch::full_like(abs_tsdf, std::numeric_limits<float>::infinity());
                                torch::Tensor abs_valid =
                                    torch::where(corner_valid, abs_tsdf, inf_like);
                                torch::Tensor min_abs_tsdf =
                                    std::get<0>(abs_valid.min(/*dim=*/1)); // [N]
                                torch::Tensor far_from_surface =
                                    (min_abs_tsdf > tau_surface).to(torch::kBool); // [N]

                                torch::Tensor sizes_for_tsdf = sizes_world;
                                if (sizes_for_tsdf.dim() == 2 && sizes_for_tsdf.size(1) == 1) {
                                    sizes_for_tsdf = sizes_for_tsdf.squeeze(1);
                                } else if (sizes_for_tsdf.dim() != 1) {
                                    sizes_for_tsdf = sizes_for_tsdf.reshape({N});
                                }
                                sizes_for_tsdf = sizes_for_tsdf
                                    .to(prune_mask.device())
                                    .to(torch::kFloat32)
                                    .contiguous();
                                torch::Tensor half_voxel_diag =
                                    (0.5f * std::sqrt(3.0f)) * sizes_for_tsdf; // [N]
                                torch::Tensor conservative_free_thresh =
                                    torch::full_like(half_voxel_diag, tau_surface) + half_voxel_diag;
                                torch::Tensor strong_far_from_surface =
                                    (min_abs_tsdf > conservative_free_thresh).to(torch::kBool); // [N]

                                torch::Tensor all_positive =
                                    (voxel_valid & has_pos & (~has_neg)).to(torch::kBool);
                                torch::Tensor all_negative =
                                    (voxel_valid & has_neg & (~has_pos)).to(torch::kBool);
                                torch::Tensor tsdf_unknown_mask =
                                    (~voxel_valid).to(torch::kBool);
                                torch::Tensor tsdf_free_mask =
                                    (all_positive & strong_far_from_surface).to(torch::kBool);
                                torch::Tensor tsdf_occupied_mask =
                                    (all_negative & strong_far_from_surface).to(torch::kBool);
                                torch::Tensor tsdf_surface_mask =
                                    (voxel_valid & (has_surface | (~strong_far_from_surface)))
                                        .to(torch::kBool);
                                tsdf_surface_protect_mask =
                                    tsdf_surface_mask.to(prune_mask.device()).to(torch::kBool).contiguous();

                                // Conservative external-TSDF pruning:
                                // prune only strong positive/free-space voxels. Negative-side voxels
                                // are kept for now because nvblox sign alone is not enough to prove
                                // they are useless scene geometry.
                                tsdf_prune_mask =
                                    tsdf_free_mask.to(torch::kBool); // [N]

                                // Ensure device matches prune_mask
                                if (tsdf_prune_mask.device() != prune_mask.device()) {
                                    tsdf_prune_mask = tsdf_prune_mask.to(prune_mask.device());
                                }
                                recordTsdfPruneAblation(
                                    tsdf_prune_mask,
                                    "regular_prune");

                                // Union + overlap statistics
                                auto prune_mask_union = prune_mask | tsdf_prune_mask;      // [N]
                                // Verbose TSDF union statistics disabled.

                                // 2) Cache TSDF class samples only when the Rerun debug recording needs them.
                                if (rerun_params_.enable_rerun_ &&
                                    rerun_params_.rerun_tsdf_unknown_voxels_) {
                                auto save_tsdf_class_debug =
                                    [&](const torch::Tensor& mask_in,
                                        torch::Tensor& centers_out,
                                        torch::Tensor& sizes_out,
                                        bool& has_out,
                                        const char* label)
                                {
                                    torch::Tensor mask = mask_in.to(torch::kBool);
                                    if (mask.device() != centers_world.device()) {
                                        mask = mask.to(centers_world.device());
                                    }

                                    auto idx = mask.nonzero().squeeze(1); // [K]
                                    if (idx.numel() > 0) {
                                        centers_out = centers_world.index_select(0, idx).clone(); // [K,3]
                                        sizes_out = sizes_world.index_select(0, idx).clone();     // [K,1]
                                        has_out = true;
                                    } else {
                                        centers_out = torch::Tensor();
                                        sizes_out = torch::Tensor();
                                        has_out = false;
                                    }
                                };

                                save_tsdf_class_debug(
                                    tsdf_free_mask,
                                    debug_tsdf_free_centers,
                                    debug_tsdf_free_sizes,
                                    debug_has_tsdf_free,
                                    "free-space");
                                save_tsdf_class_debug(
                                    tsdf_occupied_mask,
                                    debug_tsdf_occupied_centers,
                                    debug_tsdf_occupied_sizes,
                                    debug_has_tsdf_occupied,
                                    "occupied-side");
                                save_tsdf_class_debug(
                                    tsdf_surface_mask,
                                    debug_tsdf_surface_centers,
                                    debug_tsdf_surface_sizes,
                                    debug_has_tsdf_surface,
                                    "surface-band");
                                save_tsdf_class_debug(
                                    tsdf_unknown_mask,
                                    debug_tsdf_unknown_centers,
                                    debug_tsdf_unknown_sizes,
                                    debug_has_tsdf_unknown,
                                    "unknown");

                                auto save_tsdf_corner_debug =
                                    [&](const torch::Tensor& mask_in,
                                        torch::Tensor& points_out,
                                        torch::Tensor* colors_out,
                                        bool& has_out,
                                        float r,
                                        float g,
                                        float b,
                                        float a,
                                        const char* label,
                                        bool color_by_corner_valid)
                                {
                                    has_out = false;
                                    points_out = torch::Tensor();
                                    if (colors_out) {
                                        *colors_out = torch::Tensor();
                                    }
                                    if (!c.points_world.defined() ||
                                        c.points_world.dim() != 3 ||
                                        c.points_world.size(0) != N ||
                                        c.points_world.size(1) != 8 ||
                                        c.points_world.size(2) != 3) {
                                        return;
                                    }

                                    torch::Tensor mask = mask_in.to(torch::kBool);
                                    if (mask.device() != c.points_world.device()) {
                                        mask = mask.to(c.points_world.device());
                                    }
                                    torch::Tensor idx = mask.nonzero().squeeze(1);
                                    if (idx.numel() <= 0) {
                                        return;
                                    }

                                    constexpr int64_t kMaxTsdfSampleDebugVoxels = 50000;
                                    const int64_t original_voxels = idx.numel();
                                    if (original_voxels > kMaxTsdfSampleDebugVoxels) {
                                        idx = idx.slice(0, 0, kMaxTsdfSampleDebugVoxels);
                                    }

                                    points_out =
                                        c.points_world.index_select(0, idx)
                                            .reshape({-1, 3})
                                            .clone();
                                    has_out = points_out.defined() && points_out.numel() > 0;
                                    if (!has_out) {
                                        return;
                                    }

                                    if (colors_out) {
                                        torch::Tensor colors = torch::zeros(
                                            {points_out.size(0), 4},
                                            points_out.options());
                                        if (color_by_corner_valid) {
                                            torch::Tensor valid_corners =
                                                corner_valid
                                                    .to(c.points_world.device())
                                                    .index_select(0, idx)
                                                    .reshape({-1})
                                                    .to(torch::kBool);
                                            // Unknown voxels are colored per corner:
                                            // cyan = nvblox has a valid TSDF value there,
                                            // red = no valid TSDF sample at that corner.
                                            colors.index_put_({torch::indexing::Slice(), 0}, 1.0f);
                                            colors.index_put_({torch::indexing::Slice(), 3}, a);
                                            colors.index_put_({valid_corners, 0}, 0.0f);
                                            colors.index_put_({valid_corners, 1}, 0.9f);
                                            colors.index_put_({valid_corners, 2}, 1.0f);
                                        } else {
                                            colors.index_put_({torch::indexing::Slice(), 0}, r);
                                            colors.index_put_({torch::indexing::Slice(), 1}, g);
                                            colors.index_put_({torch::indexing::Slice(), 2}, b);
                                            colors.index_put_({torch::indexing::Slice(), 3}, a);
                                        }
                                        *colors_out = colors;
                                    }
                                };

                                save_tsdf_corner_debug(
                                    tsdf_free_mask,
                                    debug_tsdf_free_corner_points,
                                    nullptr,
                                    debug_has_tsdf_free_corners,
                                    0.0f, 0.9f, 1.0f, 0.95f,
                                    "free-space",
                                    false);
                                save_tsdf_corner_debug(
                                    tsdf_occupied_mask,
                                    debug_tsdf_occupied_corner_points,
                                    nullptr,
                                    debug_has_tsdf_occupied_corners,
                                    1.0f, 0.0f, 1.0f, 0.95f,
                                    "occupied-side",
                                    false);
                                save_tsdf_corner_debug(
                                    tsdf_surface_mask,
                                    debug_tsdf_surface_corner_points,
                                    nullptr,
                                    debug_has_tsdf_surface_corners,
                                    0.0f, 1.0f, 0.0f, 0.95f,
                                    "surface-band",
                                    false);
                                save_tsdf_corner_debug(
                                    tsdf_unknown_mask,
                                    debug_tsdf_unknown_corner_points,
                                    &debug_tsdf_unknown_corner_colors,
                                    debug_has_tsdf_unknown_corners,
                                    0.5f, 0.5f, 0.5f, 0.95f,
                                    "unknown",
                                    true);
                                }

                                // Use union as final prune mask
                                prune_mask = prune_mask_union;
                            } else {
                                // Verbose TSDF corner prune diagnostics disabled.
                            }
                        } catch (const std::exception&) {
                            // Verbose TSDF corner prune diagnostics disabled.
                        }
                    }
                }

                int64_t n_prune_near_front = 0;
                int64_t n_prune_near_geom = 0;

                // 3) SVRaster visibility and near-camera filtering.
                // This mimics octlayout_filtering(...) using mark_max_samp_rate + mark_near.
                if (!tr_cams.empty() && N > 0) {
                    try {
                        py::gil_scoped_acquire gil;
                        static py::module_ svr_mod =
                            py::module_::import("svraster_cuda").attr("renderer");
                        static py::module_ torch_mod =
                            py::module_::import("torch");

                        // Access Python SparseVoxelModel
                        py::object py_svm = voxel_model_->svm();
                        if (!py_svm.is_none()) {

                            py::object py_octpath   = py_svm.attr("octpath");
                            py::object py_octlv     = py_svm.attr("octlevel");
                            py::object py_vox_center= py_svm.attr("vox_center");
                            py::object py_vox_size  = py_svm.attr("vox_size");

                            at::Tensor octpath = py_octpath.cast<at::Tensor>().contiguous();     // [N,1] int64
                            at::Tensor L       = py_octlv.cast<at::Tensor>().contiguous();       // [N,1] int8 or int64
                            at::Tensor vox_center = py_vox_center.cast<at::Tensor>().contiguous(); // [N,3]
                            at::Tensor vox_size   = py_vox_size.cast<at::Tensor>().contiguous();   // [N,1] or [N]

                            // Basic sanity: same N
                            TORCH_CHECK(octpath.size(0) == N,
                                        "octpath.size(0) != N in pruning visibility filter");
                            TORCH_CHECK(L.size(0) == N,
                                        "octlevel.size(0) != N in pruning visibility filter");
                            TORCH_CHECK(vox_center.size(0) == N,
                                        "vox_center.size(0) != N in pruning visibility filter");
                            TORCH_CHECK(vox_center.size(1) == 3,
                                        "vox_center.size(1) must be 3");
                            if (vox_size.dim() == 1) {
                                vox_size = vox_size.view({N,1});
                            } else if (vox_size.dim() == 2) {
                                TORCH_CHECK(vox_size.size(0) == N,
                                            "vox_size.size(0) != N in pruning visibility filter");
                            } else {
                                TORCH_CHECK(false, "vox_size must be [N] or [N,1]");
                            }

                            // Build Python list of CUDA MiniCams
                            py::list py_cams;
                            py::object py_cuda = torch_mod.attr("device")("cuda");

                            auto move_attr_to_cuda_if_tensor =
                                [&](py::object& obj, const char* name){
                                    if (py::hasattr(obj, name)) {
                                        py::object t = obj.attr(name);
                                        if (py::hasattr(t, "is_cuda") &&
                                            !py::bool_(t.attr("is_cuda"))) {
                                            obj.attr(name) = t.attr("to")(py_cuda);
                                        }
                                    }
                                };

                            for (const auto& c : tr_cams) {
                                py::object py_cam = MiniCam_to_py(c);
                                move_attr_to_cuda_if_tensor(py_cam, "w2c");
                                move_attr_to_cuda_if_tensor(py_cam, "c2w");
                                move_attr_to_cuda_if_tensor(py_cam, "position");
                                move_attr_to_cuda_if_tensor(py_cam, "lookat");
                                py_cams.append(py_cam);
                            }

                            auto Nu_before = octpath.size(0);
                            TORCH_CHECK(Nu_before == N,
                                        "octpath.size(0) != N before visibility filter");

                            // 1) visibility: rate > 0
                            at::Tensor rate = svr_mod.attr("mark_max_samp_rate")(
                                py_cams,
                                py::cast(octpath),
                                py::cast(vox_center),
                                py::cast(vox_size)
                            ).cast<at::Tensor>();        // [N,1] or [N]

                            if (rate.dim() == 2 && rate.size(1) == 1)
                                rate = rate.squeeze(1);
                            rate = rate.to(torch::kFloat32);

                            at::Tensor keep_rate = (rate > 0.0f).to(torch::kBool);   // [N]
                            int64_t n_rate_pos = keep_rate.sum().item<int64_t>();

                            // 2) near filtering:
                            //    a) SVRaster mark_near (camera-facing)
                            //    b) geometric distance-to-camera test (front or behind)
                            const float near_thresh = 0.2f;
                            int64_t n_near_hit = 0;
                            int64_t n_near_geom_hit = 0;
                            at::Tensor is_near = torch::zeros(
                                {N},
                                torch::TensorOptions().dtype(torch::kBool).device(keep_rate.device()));
                            at::Tensor is_near_geom = torch::zeros(
                                {N},
                                torch::TensorOptions().dtype(torch::kBool).device(keep_rate.device()));
                            if (near_thresh > 0.0f) {
                                is_near = svr_mod.attr("mark_near")(
                                    py_cams,
                                    py::cast(octpath),
                                    py::cast(vox_center),
                                    py::cast(vox_size),
                                    py::float_(near_thresh)
                                ).cast<at::Tensor>();       // [N,1] or [N]
                                if (is_near.dim() == 2 && is_near.size(1) == 1)
                                    is_near = is_near.squeeze(1);
                                is_near = is_near.to(torch::kBool);
                                n_near_hit = is_near.sum().item<int64_t>();

                                if (opt_params_.prune_near_voxels_geometric_) {
                                    auto vox_center_f32 =
                                        vox_center.to(keep_rate.device()).to(torch::kFloat32).contiguous();
                                    auto vox_size_1d =
                                        vox_size.to(keep_rate.device()).to(torch::kFloat32).contiguous();
                                    if (vox_size_1d.dim() == 2 && vox_size_1d.size(1) == 1) {
                                        vox_size_1d = vox_size_1d.squeeze(1);
                                    } else if (vox_size_1d.dim() != 1) {
                                        vox_size_1d = vox_size_1d.reshape({-1});
                                    }
                                    TORCH_CHECK(vox_size_1d.numel() == N,
                                                "vox_size_1d.numel() != N in pruning geometric near filter");

                                    auto near_radius =
                                        (torch::full_like(vox_size_1d, near_thresh) + 0.5f * vox_size_1d)
                                            .contiguous();
                                    auto near_radius_sq = (near_radius * near_radius).contiguous();

                                    for (const auto& c : tr_cams) {
                                        auto cam_pos =
                                            c.position.to(keep_rate.device()).to(torch::kFloat32).view({1, 3});
                                        auto d2 = (vox_center_f32 - cam_pos).pow(2).sum(/*dim=*/1);
                                        is_near_geom =
                                            (is_near_geom | (d2 <= near_radius_sq)).to(torch::kBool);
                                    }
                                    n_near_geom_hit = is_near_geom.sum().item<int64_t>();
                                }
                            }

                            auto prune_near_union = (is_near | is_near_geom).to(torch::kBool);
                            n_prune_near_front = n_near_hit;
                            n_prune_near_geom = n_near_geom_hit;

                            auto near_geom_idx = is_near_geom.nonzero().squeeze(1);  // [K_geom]
                            if (near_geom_idx.numel() > 0 &&
                                rerun_params_.enable_rerun_) {
                                debug_near_geom_centers = vox_center.index({near_geom_idx}).clone(); // [K_geom,3]
                                debug_near_geom_sizes   = vox_size.index({near_geom_idx}).clone();   // [K_geom,1] or [K_geom]
                                debug_has_near_geom     = true;
                            } else {
                                debug_has_near_geom = false;
                            }

                            auto near_idx = prune_near_union.nonzero().squeeze(1);  // [K]
                            if (near_idx.numel() > 0 &&
                                rerun_params_.enable_rerun_) {
                                debug_near_centers = vox_center.index({near_idx}).clone();  // [K,3]
                                debug_near_sizes   = vox_size.index({near_idx}).clone();    // [K,1] or [K]
                                debug_has_near     = true;
                            } else {
                                debug_has_near = false;
                            }

                            keep_rate = keep_rate.view({-1}).to(torch::kBool);    // [N]
                            prune_mask_vis = (~keep_rate);     // [N], visibility only
                            prune_mask_near = prune_near_union.view({-1}).to(torch::kBool); // [N], near only

                            // Combine with existing prune_mask
                            prune_mask = prune_mask | prune_mask_vis;

                        }

                    } catch (const std::exception& e) {
                        std::cerr << "[PRUNE/visibility] exception: " << e.what() << "\n";
                    }
                }

                // 4) Extra geometric rules: dense-core outliers and recent unstable voxels.
                prune_mask_default = prune_mask.to(torch::kBool);

                int64_t n_prune_base_real = prune_mask_base_real.defined()
                    ? prune_mask_base_real.sum().item<int64_t>() : 0;
                int64_t n_prune_base_artificial = prune_mask_base_artificial.defined()
                    ? prune_mask_base_artificial.sum().item<int64_t>() : 0;
                int64_t n_prune_base_real_cand = prune_mask_base_real_raw.defined()
                    ? prune_mask_base_real_raw.sum().item<int64_t>() : 0;
                int64_t n_prune_real_outside_dense_core = 0;
                int64_t n_prune_gslam_real = 0;
                int64_t n_prune_gslam_artificial = 0;
                int64_t n_prune_base_real_at_target = 0;
                int64_t n_prune_base_artificial_at_target = 0;
                int64_t n_prune_base_real_at_target_extra = 0;
                int64_t n_prune_base_artificial_at_target_extra = 0;
                int64_t n_prune_base_real_above_target = 0;
                int64_t n_prune_base_artificial_above_target = 0;
                int64_t n_prune_base_real_pre_gates = 0;
                int64_t n_prune_base_real_at_target_pre_gates = 0;
                int64_t n_prune_base_real_above_target_pre_gates = 0;
                int64_t n_real_at_target_total = 0;
                int64_t n_real_above_target_total = 0;
                int64_t n_artificial_at_target_total = 0;
                int64_t n_artificial_above_target_total = 0;
                int64_t n_prune_near = 0;
                const int64_t n_promoted_artificial_total = voxel_model_->totalPromotedartificialCount();

                if (N > 0) {
                    bool use_far_prune_this_round = opt_params_.prune_far_voxels_;
                    if (opt_params_.prune_far_voxels_) {
                        voxel_model_->refreshDenseCoreBBFromCurrentVoxels();
                        if (!voxel_model_->hasDenseCoreBB()) {
                            use_far_prune_this_round = false;
                        }
                        if (rerun_params_.enable_rerun_ && voxel_model_->hasDenseCoreBB()) {
                            voxel_model_->logDenseCoreBBoxToRerun(
                                getIteration(),
                                "world/dense_core/used_for_prune");
                        }
                    }
                    prune_mask_real_outside_dense_core = torch::zeros(
                        {N}, torch::TensorOptions().dtype(torch::kBool).device(prune_mask.device()));
                    if (!prune_mask_near.defined() || prune_mask_near.numel() != N) {
                        prune_mask_near = torch::zeros(
                            {N}, torch::TensorOptions().dtype(torch::kBool).device(prune_mask.device()));
                    } else if (prune_mask_near.device() != prune_mask.device()) {
                        prune_mask_near = prune_mask_near.to(prune_mask.device()).to(torch::kBool).contiguous();
                    }
                    prune_mask_gslam_unstable = torch::zeros(
                        {N}, torch::TensorOptions().dtype(torch::kBool).device(prune_mask.device()));
                    prune_mask_gslam_real = torch::zeros(
                        {N}, torch::TensorOptions().dtype(torch::kBool).device(prune_mask.device()));
                    prune_mask_gslam_artificial = torch::zeros(
                        {N}, torch::TensorOptions().dtype(torch::kBool).device(prune_mask.device()));
                    auto geometrically_unstable_mask = torch::zeros(
                        {N}, torch::TensorOptions().dtype(torch::kBool).device(prune_mask.device()));

                    auto art_mask = art_mask_for_base.to(prune_mask.device()).to(torch::kBool).contiguous();
                    auto real_mask = real_mask_for_base.to(prune_mask.device()).to(torch::kBool).contiguous();
                    auto view_cnt = stat.view_cnt;
                    auto exist_since_kf = voxel_model_->existSinceKf();
                    if (view_cnt.defined() && exist_since_kf.defined()) {
                        view_cnt = flatten_colvec(view_cnt.to(prune_mask.device()).to(torch::kFloat32));
                        exist_since_kf = flatten_colvec(
                            exist_since_kf.to(prune_mask.device()).to(torch::kInt32));

                        if (view_cnt.numel() == N &&
                            exist_since_kf.numel() == N) {
                            if (use_far_prune_this_round && voxel_model_->hasDenseCoreBB()) {
                                auto centers = voxel_model_->voxCenter();
                                auto bb_min = voxel_model_->denseCoreBBMin();
                                auto bb_max = voxel_model_->denseCoreBBMax();
                                if (centers.defined() && bb_min.defined() && bb_max.defined() &&
                                    centers.dim() == 2 && centers.size(1) == 3 &&
                                    centers.size(0) == N &&
                                    bb_min.numel() == 3 && bb_max.numel() == 3) {
                                    centers = centers.to(prune_mask.device()).to(torch::kFloat32).contiguous();
                                    bb_min = bb_min.to(prune_mask.device()).to(torch::kFloat32).contiguous().view({1, 3});
                                    bb_max = bb_max.to(prune_mask.device()).to(torch::kFloat32).contiguous().view({1, 3});
                                    auto in_dense_core =
                                        (centers >= bb_min).all(/*dim=*/1) &
                                        (centers <= bb_max).all(/*dim=*/1);
                                    prune_mask_real_outside_dense_core =
                                        (real_mask & (~in_dense_core.to(torch::kBool))).to(torch::kBool);
                                }
                            }

                            if (opt_params_.prune_recent_unstable_) {
                                const int32_t current_kf_count = static_cast<int32_t>(tr_cams.size());
                                const int recent_kf_span = std::max(0, opt_params_.prune_recent_keyframes_);
                                auto born_valid_mask = (exist_since_kf >= 0).to(torch::kBool);
                                auto age_kf = (current_kf_count - exist_since_kf).to(torch::kInt32);
                                auto recent_mask =
                                    (born_valid_mask & (age_kf <= recent_kf_span)).to(torch::kBool);
                                auto other_view_cnt = torch::clamp_min(view_cnt - 1.0f, 0.0f);

                                auto prune_recent_real =
                                    recent_mask &
                                    real_mask &
                                    (other_view_cnt < static_cast<float>(opt_params_.prune_recent_min_views_real_));
                                auto prune_recent_artificial =
                                    recent_mask &
                                    art_mask &
                                    (other_view_cnt < static_cast<float>(opt_params_.prune_recent_min_views_artificial_));

                                prune_mask_gslam_real = prune_recent_real.to(torch::kBool);
                                prune_mask_gslam_artificial = prune_recent_artificial.to(torch::kBool);
                                prune_mask_gslam_unstable =
                                    (prune_mask_gslam_real | prune_mask_gslam_artificial).to(torch::kBool);
                                geometrically_unstable_mask = prune_mask_gslam_unstable.clone();
                            }
                        }
                    }

                    voxel_model_->setGeometricallyUnstableMask(geometrically_unstable_mask);
                    prune_mask_recent_unstable = prune_mask_gslam_unstable;
                    prune_mask = (prune_mask_default |
                                  prune_mask_near |
                                  prune_mask_real_outside_dense_core |
                                  prune_mask_gslam_unstable)
                        .to(torch::kBool);

                    n_prune_near = prune_mask_near.sum().item<int64_t>();
                    n_prune_real_outside_dense_core = prune_mask_real_outside_dense_core.sum().item<int64_t>();
                    n_prune_gslam_real = prune_mask_gslam_real.sum().item<int64_t>();
                    n_prune_gslam_artificial = prune_mask_gslam_artificial.sum().item<int64_t>();
                } else {
                    voxel_model_->setGeometricallyUnstableMask(torch::zeros(
                        {0},
                        torch::TensorOptions().dtype(torch::kBool).device(prune_mask.device())));
                }

                // 5) Record pre-protection threshold pruning stats.
                n_prune_base_real_pre_gates = prune_mask_base_real.defined()
                    ? prune_mask_base_real.sum().item<int64_t>() : 0;
                n_prune_base_real_at_target_pre_gates =
                    (prune_mask_base_real.to(torch::kBool) &
                     in_target_size_mask.to(torch::kBool)).sum().item<int64_t>();
                n_prune_base_real_above_target_pre_gates =
                    (prune_mask_base_real.to(torch::kBool) &
                     (~in_target_size_mask.to(torch::kBool))).sum().item<int64_t>();

                // 6) Optional TSDF surface-band protection for non-SDF prune sources.
                if (sdf_params_.tsdf_protect_surface_band_from_pruning_ &&
                    tsdf_surface_protect_mask.defined() &&
                    tsdf_surface_protect_mask.numel() == N &&
                    prune_mask.defined() &&
                    prune_mask.numel() == N) {
                    torch::Tensor surface_protect =
                        tsdf_surface_protect_mask
                            .to(prune_mask.device())
                            .to(torch::kBool)
                            .contiguous();
                    torch::Tensor sdf_prune_keep = torch::zeros(
                        {N},
                        torch::TensorOptions().dtype(torch::kBool).device(prune_mask.device()));
                    if (tsdf_prune_mask.defined() && tsdf_prune_mask.numel() == N) {
                        sdf_prune_keep =
                            tsdf_prune_mask
                                .to(prune_mask.device())
                                .to(torch::kBool)
                                .contiguous();
                    }
                    torch::Tensor non_sdf_surface_protect =
                        (surface_protect & (~sdf_prune_keep)).to(torch::kBool);
                    auto suppress_non_sdf_surface = [&](torch::Tensor& mask) {
                        if (mask.defined() && mask.numel() == N) {
                            mask = (mask.to(prune_mask.device()).to(torch::kBool) &
                                    (~non_sdf_surface_protect)).to(torch::kBool);
                        }
                    };
                    suppress_non_sdf_surface(prune_mask_base_real);
                    suppress_non_sdf_surface(prune_mask_base_artificial);
                    suppress_non_sdf_surface(prune_mask_base_real_at_target);
                    suppress_non_sdf_surface(prune_mask_base_artificial_at_target);
                    suppress_non_sdf_surface(prune_mask_base_real_at_target_extra);
                    suppress_non_sdf_surface(prune_mask_base_artificial_at_target_extra);
                    suppress_non_sdf_surface(prune_mask_default);
                    suppress_non_sdf_surface(prune_mask_vis);
                    suppress_non_sdf_surface(prune_mask_near);
                    suppress_non_sdf_surface(prune_mask_real_outside_dense_core);
                    suppress_non_sdf_surface(prune_mask_gslam_unstable);
                    suppress_non_sdf_surface(prune_mask_gslam_real);
                    suppress_non_sdf_surface(prune_mask_gslam_artificial);
                    suppress_non_sdf_surface(prune_mask_recent_unstable);
                    prune_mask =
                        ((prune_mask.to(torch::kBool) & (~non_sdf_surface_protect)) |
                         (prune_mask.to(torch::kBool) & sdf_prune_keep)).to(torch::kBool);
                }

                // 7) Report source counts, record optional debug data, then apply pruning.
                n_prune_base_real = prune_mask_base_real.defined()
                    ? prune_mask_base_real.sum().item<int64_t>() : 0;
                n_prune_base_artificial = prune_mask_base_artificial.defined()
                    ? prune_mask_base_artificial.sum().item<int64_t>() : 0;
                n_prune_base_real_at_target = prune_mask_base_real_at_target.defined()
                    ? prune_mask_base_real_at_target.sum().item<int64_t>() : 0;
                n_prune_base_artificial_at_target = prune_mask_base_artificial_at_target.defined()
                    ? prune_mask_base_artificial_at_target.sum().item<int64_t>() : 0;
                n_prune_base_real_at_target_extra = prune_mask_base_real_at_target_extra.defined()
                    ? prune_mask_base_real_at_target_extra.sum().item<int64_t>() : 0;
                n_prune_base_artificial_at_target_extra = prune_mask_base_artificial_at_target_extra.defined()
                    ? prune_mask_base_artificial_at_target_extra.sum().item<int64_t>() : 0;
                auto in_target_size_mask_final = in_target_size_mask.to(torch::kBool).contiguous();
                auto above_target_size_mask_final = (~in_target_size_mask_final).to(torch::kBool);
                n_prune_base_real_above_target =
                    (prune_mask_base_real.to(torch::kBool) & above_target_size_mask_final).sum().item<int64_t>();
                n_prune_base_artificial_above_target =
                    (prune_mask_base_artificial.to(torch::kBool) & above_target_size_mask_final).sum().item<int64_t>();
                n_real_at_target_total =
                    (real_mask_for_base.to(torch::kBool) & in_target_size_mask_final).sum().item<int64_t>();
                n_real_above_target_total =
                    (real_mask_for_base.to(torch::kBool) & above_target_size_mask_final).sum().item<int64_t>();
                n_artificial_at_target_total =
                    (art_mask_for_base.to(torch::kBool) & in_target_size_mask_final).sum().item<int64_t>();
                n_artificial_above_target_total =
                    (art_mask_for_base.to(torch::kBool) & above_target_size_mask_final).sum().item<int64_t>();
                n_prune_real_outside_dense_core = prune_mask_real_outside_dense_core.defined()
                    ? prune_mask_real_outside_dense_core.sum().item<int64_t>() : 0;
                n_prune_gslam_real = prune_mask_gslam_real.defined()
                    ? prune_mask_gslam_real.sum().item<int64_t>() : 0;
                n_prune_gslam_artificial = prune_mask_gslam_artificial.defined()
                    ? prune_mask_gslam_artificial.sum().item<int64_t>() : 0;
                const int64_t n_prune_total_selected = prune_mask.defined()
                    ? prune_mask.to(torch::kBool).sum().item<int64_t>() : 0;
                auto count_pruned_by_mask = [&](const torch::Tensor& mask) -> int64_t {
                    if (!mask.defined() || !prune_mask.defined() || mask.numel() != N) {
                        return 0;
                    }
                    return (mask.to(prune_mask.device()).to(torch::kBool) &
                            prune_mask.to(torch::kBool))
                        .sum().item<int64_t>();
                };
                const int64_t n_prune_stats_svraster =
                    count_pruned_by_mask(prune_mask_default);
                const int64_t n_prune_stats_tsdf =
                    count_pruned_by_mask(tsdf_prune_mask);
                const int64_t n_prune_stats_near =
                    count_pruned_by_mask(prune_mask_near);
                const int64_t n_prune_stats_recent =
                    count_pruned_by_mask(prune_mask_recent_unstable);
                const int64_t n_prune_stats_far =
                    count_pruned_by_mask(prune_mask_real_outside_dense_core);
                std::cout << "[PRUNE/stats] iter=" << iter
                          << " total=" << n_prune_total_selected
                          << " svraster=" << n_prune_stats_svraster
                          << " tsdf=" << n_prune_stats_tsdf
                          << " near=" << n_prune_stats_near
                          << " recent_unstable=" << n_prune_stats_recent
                          << " far=" << n_prune_stats_far
                          << "\n";

                if (rerun_params_.enable_rerun_ &&
                    rerun_params_.run_tsdf_pruned_ &&
                    sensor_type_ == RGBD &&
                    prune_mask.defined() &&
                    prune_mask.numel() == N &&
                    n_prune_total_selected > 0) {
                    std::string gt_mesh_path_for_prune = rerun_params_.rerun_tsdf_pruned_gt_mesh_path_;
                    if (gt_mesh_path_for_prune.empty()) {
                        gt_mesh_path_for_prune = rerun_params_.rerun_gt_mesh_path_;
                    }
                    if (!gt_mesh_path_for_prune.empty() &&
                        std::filesystem::exists(gt_mesh_path_for_prune) &&
                        ensureRerunGtSdfGridCache(gt_mesh_path_for_prune)) {
                        try {
                            torch::Tensor vox_key = voxel_model_->voxKey();
                            torch::Tensor vox_size = voxel_model_->voxSize();
                            if (vox_key.defined() &&
                                vox_key.dim() == 2 &&
                                vox_key.size(0) == N &&
                                vox_key.size(1) == 8 &&
                                vox_size.defined() &&
                                vox_size.size(0) == N &&
                                rerun_state_.rerun_gt_sdf_grid_pts_cpu_.defined() &&
                                rerun_state_.rerun_gt_sdf_grid_pts_cpu_.numel() > 0) {
                                torch::Tensor prune_idx =
                                    prune_mask.to(torch::kCPU).to(torch::kBool).nonzero().squeeze(1);
                                torch::Tensor key_flat_cpu =
                                    vox_key.to(torch::kCPU).to(torch::kLong).reshape({-1});
                                torch::Tensor gt_sdf_corners =
                                    rerun_state_.rerun_gt_sdf_grid_pts_cpu_
                                        .index_select(0, key_flat_cpu)
                                        .view({N, 8})
                                        .contiguous();
                                torch::Tensor gt_pruned =
                                    gt_sdf_corners.index_select(0, prune_idx).contiguous();

                                torch::Tensor sizes_cpu = vox_size.to(torch::kCPU).to(torch::kFloat32);
                                if (sizes_cpu.dim() == 2 && sizes_cpu.size(1) == 1) {
                                    sizes_cpu = sizes_cpu.squeeze(1);
                                } else if (sizes_cpu.dim() != 1) {
                                    sizes_cpu = sizes_cpu.reshape({N});
                                }
                                torch::Tensor sizes_pruned =
                                    sizes_cpu.index_select(0, prune_idx).contiguous();

                                const float tau_surface =
                                    std::max(0.0f, sdf_params_.tsdf_prune_surface_band_vox_) *
                                    tsdfMetricVoxelSize();
                                const float sign_eps = 1.0e-6f;
                                const int min_valid_corners =
                                    std::max(1, sdf_params_.tsdf_prune_min_valid_corners_);

                                torch::Tensor gt_valid = torch::isfinite(gt_pruned);
                                torch::Tensor valid_count =
                                    gt_valid.to(torch::kInt32).sum(/*dim=*/1);
                                torch::Tensor gt_known =
                                    (valid_count >= min_valid_corners).to(torch::kBool);
                                torch::Tensor gt_has_pos =
                                    ((gt_pruned > sign_eps) & gt_valid).any(/*dim=*/1);
                                torch::Tensor gt_has_neg =
                                    ((gt_pruned < -sign_eps) & gt_valid).any(/*dim=*/1);
                                torch::Tensor inf_gt =
                                    torch::full_like(gt_pruned, std::numeric_limits<float>::infinity());
                                torch::Tensor gt_min_abs =
                                    std::get<0>(torch::where(gt_valid, gt_pruned.abs(), inf_gt).min(/*dim=*/1));

                                torch::Tensor gt_surface =
                                    (gt_known &
                                     ((gt_has_pos & gt_has_neg) |
                                      (gt_min_abs <= tau_surface))).to(torch::kBool);
                                torch::Tensor gt_free =
                                    (gt_known & (~gt_surface) & gt_has_pos & (~gt_has_neg)).to(torch::kBool);
                                torch::Tensor gt_behind =
                                    (gt_known & (~gt_surface) & gt_has_neg & (~gt_has_pos)).to(torch::kBool);
                                torch::Tensor gt_unknown =
                                    (~gt_known).to(torch::kBool);

                                const int64_t pruned_total = prune_idx.size(0);
                                const int64_t gt_surface_count = gt_surface.sum().item<int64_t>();
                                torch::Tensor tsdf_pruned_final = torch::zeros(
                                    {N},
                                    torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
                                if (tsdf_prune_mask.defined() &&
                                    tsdf_prune_mask.numel() == N) {
                                    tsdf_pruned_final =
                                        (tsdf_prune_mask.to(torch::kCPU).to(torch::kBool) &
                                         prune_mask.to(torch::kCPU).to(torch::kBool)).to(torch::kBool);
                                }
                                torch::Tensor tsdf_pruned_sel =
                                    tsdf_pruned_final.index_select(0, prune_idx).to(torch::kBool);
                                const int64_t tsdf_pruned_count =
                                    tsdf_pruned_sel.sum().item<int64_t>();
                                const int64_t tsdf_surface_count =
                                    (tsdf_pruned_sel & gt_surface).sum().item<int64_t>();
                                const int64_t other_pruned_count =
                                    pruned_total - tsdf_pruned_count;
                                const int64_t other_surface_count =
                                    gt_surface_count - tsdf_surface_count;

                                std::cout << "[PRUNE GT SURFACE] iter=" << iter
                                          << " pruned=" << pruned_total
                                          << " gt_surface=" << gt_surface_count
                                          << " tsdf_pruned=" << tsdf_pruned_count
                                          << " tsdf_gt_surface=" << tsdf_surface_count
                                          << " other_pruned=" << other_pruned_count
                                          << " other_gt_surface=" << other_surface_count
                                          << std::endl;
                            }
                        } catch (const std::exception& e) {
                            std::cerr << "[PRUNE GT SURFACE] failed: "
                                      << e.what() << "\n";
                        }
                    }
                }
                // Verbose [PRUNE/real] and [PRUNE/artificial] diagnostics disabled.

                if (rerun_params_.enable_rerun_ &&
                    rerun_params_.run_tsdf_pruned_ &&
                    prune_mask.defined() &&
                    prune_mask.numel() == N) {
                    auto prune_idx = prune_mask.to(torch::kBool).nonzero().squeeze(1); // [K]
                    if (prune_idx.numel() > 0 &&
                        tsdf_debug_corner_points_all.defined() &&
                        tsdf_debug_values_all.defined() &&
                        tsdf_debug_weights_all.defined() &&
                        tsdf_debug_corner_points_all.dim() == 3 &&
                        tsdf_debug_corner_points_all.size(0) == N &&
                        tsdf_debug_corner_points_all.size(1) == 8 &&
                        tsdf_debug_corner_points_all.size(2) == 3 &&
                        tsdf_debug_values_all.dim() == 2 &&
                        tsdf_debug_values_all.size(0) == N &&
                        tsdf_debug_values_all.size(1) == 8 &&
                        tsdf_debug_weights_all.dim() == 2 &&
                        tsdf_debug_weights_all.size(0) == N &&
                        tsdf_debug_weights_all.size(1) == 8) {
                        try {
                            torch::Tensor centers_world = voxel_model_->voxCenter();
                            torch::Tensor sizes_world = voxel_model_->voxSize();
                            torch::Tensor vox_key = voxel_model_->voxKey();
                            torch::Tensor geo = voxel_model_->geoGridPts();
                            std::string mesh_path = rerun_params_.rerun_tsdf_pruned_gt_mesh_path_;
                            if (mesh_path.empty()) {
                                mesh_path = rerun_params_.rerun_gt_mesh_path_;
                            }
                            if (centers_world.defined() &&
                                centers_world.dim() == 2 &&
                                centers_world.size(0) == N &&
                                centers_world.size(1) == 3 &&
                                sizes_world.defined() &&
                                sizes_world.size(0) == N &&
                                vox_key.defined() &&
                                vox_key.dim() == 2 &&
                                vox_key.size(0) == N &&
                                vox_key.size(1) == 8 &&
                                geo.defined() &&
                                geo.numel() > 0 &&
                                !mesh_path.empty() &&
                                std::filesystem::exists(mesh_path) &&
                                ensureRerunGtSdfGridCache(mesh_path)) {

                                torch::Tensor debug_centers =
                                    centers_world.index_select(
                                        0,
                                        prune_idx.to(centers_world.device()).to(torch::kLong)).contiguous();
                                torch::Tensor debug_sizes =
                                    sizes_world.index_select(
                                        0,
                                        prune_idx.to(sizes_world.device()).to(torch::kLong)).contiguous();
                                torch::Tensor selected_corner_points =
                                    tsdf_debug_corner_points_all.index_select(
                                        0,
                                        prune_idx.to(tsdf_debug_corner_points_all.device()).to(torch::kLong)).contiguous();
                                torch::Tensor selected_tsdf =
                                    tsdf_debug_values_all.index_select(
                                        0,
                                        prune_idx.to(tsdf_debug_values_all.device()).to(torch::kLong)).contiguous();
                                torch::Tensor selected_weights =
                                    tsdf_debug_weights_all.index_select(
                                        0,
                                        prune_idx.to(tsdf_debug_weights_all.device()).to(torch::kLong)).contiguous();

                                torch::Tensor geo_dev =
                                    geo.to(vox_key.device()).to(torch::kFloat32).view({-1});
                                torch::Tensor key_flat =
                                    vox_key.to(vox_key.device()).to(torch::kLong).reshape({-1});
                                torch::Tensor density_corners =
                                    geo_dev.index_select(0, key_flat)
                                        .view({N, 8})
                                        .contiguous();
                                torch::Tensor selected_density =
                                    density_corners.index_select(
                                        0,
                                        prune_idx.to(density_corners.device()).to(torch::kLong)).contiguous();

                                torch::Tensor key_flat_cpu =
                                    vox_key.to(torch::kCPU).to(torch::kLong).reshape({-1});
                                torch::Tensor gt_sdf_corners =
                                    rerun_state_.rerun_gt_sdf_grid_pts_cpu_
                                        .index_select(0, key_flat_cpu)
                                        .view({N, 8})
                                        .contiguous();
                                torch::Tensor selected_gt_sdf =
                                    gt_sdf_corners.index_select(
                                        0,
                                        prune_idx.to(torch::kCPU).to(torch::kLong)).contiguous();

                                torch::Tensor selected_colors;
                                try {
                                    torch::Tensor sh0 = voxel_model_->sh0();
                                    if (sh0.defined() && sh0.dim() >= 2 &&
                                        sh0.size(0) == N) {
                                        torch::Tensor sh0_sel =
                                            sh0.index_select(
                                                0,
                                                prune_idx.to(sh0.device()).to(torch::kLong)).contiguous();
                                        py::gil_scoped_acquire gil2;
                                        static py::module act_mod =
                                            py::module::import("src.utils.activation_utils");
                                        py::object rgb_py =
                                            act_mod.attr("shzero2rgb")(py::cast(sh0_sel));
                                        selected_colors =
                                            rgb_py.cast<torch::Tensor>().contiguous();
                                    }
                                } catch (const std::exception& e) {
                                    std::cerr << "[RERUN/tsdf_pruned] failed to compute pruned voxel colors: "
                                              << e.what() << "\n";
                                }
                                if (!selected_colors.defined() ||
                                    selected_colors.numel() == 0 ||
                                    selected_colors.size(0) != prune_idx.size(0)) {
                                    selected_colors = torch::zeros(
                                        {prune_idx.size(0), 4},
                                        torch::TensorOptions()
                                            .dtype(torch::kFloat32)
                                            .device(debug_centers.device()));
                                    selected_colors.index_put_({torch::indexing::Slice(), 2}, 1.0f);
                                    selected_colors.index_put_({torch::indexing::Slice(), 3}, 0.7f);
                                }

                                torch::Tensor source_sdf_all = torch::zeros(
                                    {N},
                                    torch::TensorOptions().dtype(torch::kBool).device(prune_mask.device()));
                                if (tsdf_prune_mask.defined() &&
                                    tsdf_prune_mask.numel() == N) {
                                    source_sdf_all =
                                        (tsdf_prune_mask.to(prune_mask.device()).to(torch::kBool) &
                                         prune_mask.to(torch::kBool)).to(torch::kBool);
                                }
                                torch::Tensor source_svraster_all =
                                    (prune_mask.to(torch::kBool) & (~source_sdf_all)).to(torch::kBool);
                                torch::Tensor source_sdf_sel =
                                    source_sdf_all.index_select(
                                        0,
                                        prune_idx.to(source_sdf_all.device()).to(torch::kLong)).contiguous();
                                torch::Tensor source_svraster_sel =
                                    source_svraster_all.index_select(
                                        0,
                                        prune_idx.to(source_svraster_all.device()).to(torch::kLong)).contiguous();

                                sv::RerunVisualizerBridge::instance().visualizeSdfVoxelsRecording(
                                    "tsdf_pruned",
                                    debug_centers,
                                    debug_sizes,
                                    selected_corner_points,
                                    selected_tsdf,
                                    selected_weights,
                                    selected_density,
                                    selected_gt_sdf,
                                    selected_colors,
                                    prune_idx.to(torch::kCPU).to(torch::kLong),
                                    source_sdf_sel.to(torch::kCPU).to(torch::kBool),
                                    source_svraster_sel.to(torch::kCPU).to(torch::kBool),
                                    iter,
                                    mesh_path,
                                    rerun_params_.rerun_tsdf_pruned_align_gt_to_slam_,
                                    rerun_params_.rerun_tsdf_pruned_gt_traj_path_,
                                    rerun_params_.rerun_tsdf_pruned_align_min_pairs_,
                                    std::max(0.0f, sdf_params_.tsdf_prune_surface_band_vox_) *
                                        tsdfMetricVoxelSize(),
                                    sdf_params_.tsdf_prune_min_weight_,
                                    rerun_params_.rerun_tsdf_pruned_log_gt_mesh_,
                                    "world/voxels_pruned");
                            }
                        } catch (const std::exception& e) {
                            std::cerr << "[RERUN/tsdf_pruned] failed to log final pruned SDF metadata: "
                                      << e.what() << "\n";
                        }
                    }
                }

                if (rerun_params_.enable_rerun_ &&
                    rerun_params_.run_sdf_pruned_nvblox_ &&
                    prune_mask.defined() &&
                    prune_mask.numel() == N) {
                    auto prune_idx = prune_mask.to(torch::kBool).nonzero().squeeze(1);
                    if (sdf_mapper_ &&
                        sdf_mapper_->tsdf_layer().size() > 0 &&
                        prune_idx.numel() > 0 &&
                        tsdf_debug_corner_points_all.defined() &&
                        tsdf_debug_values_all.defined() &&
                        tsdf_debug_weights_all.defined() &&
                        tsdf_debug_corner_points_all.dim() == 3 &&
                        tsdf_debug_corner_points_all.size(0) == N &&
                        tsdf_debug_corner_points_all.size(1) == 8 &&
                        tsdf_debug_corner_points_all.size(2) == 3 &&
                        tsdf_debug_values_all.dim() == 2 &&
                        tsdf_debug_values_all.size(0) == N &&
                        tsdf_debug_values_all.size(1) == 8 &&
                        tsdf_debug_weights_all.dim() == 2 &&
                        tsdf_debug_weights_all.size(0) == N &&
                        tsdf_debug_weights_all.size(1) == 8) {
                        try {
                            torch::Tensor centers_world = voxel_model_->voxCenter();
                            torch::Tensor sizes_world = voxel_model_->voxSize();
                            torch::Tensor vox_key = voxel_model_->voxKey();
                            torch::Tensor geo = voxel_model_->geoGridPts();
                            if (centers_world.defined() &&
                                centers_world.dim() == 2 &&
                                centers_world.size(0) == N &&
                                centers_world.size(1) == 3 &&
                                sizes_world.defined() &&
                                sizes_world.size(0) == N &&
                                vox_key.defined() &&
                                vox_key.dim() == 2 &&
                                vox_key.size(0) == N &&
                                vox_key.size(1) == 8 &&
                                geo.defined() &&
                                geo.numel() > 0) {
                                torch::Tensor debug_centers =
                                    centers_world.index_select(
                                        0,
                                        prune_idx.to(centers_world.device()).to(torch::kLong)).contiguous();
                                torch::Tensor debug_sizes =
                                    sizes_world.index_select(
                                        0,
                                        prune_idx.to(sizes_world.device()).to(torch::kLong)).contiguous();
                                torch::Tensor selected_corner_points =
                                    tsdf_debug_corner_points_all.index_select(
                                        0,
                                        prune_idx.to(tsdf_debug_corner_points_all.device()).to(torch::kLong)).contiguous();
                                torch::Tensor selected_tsdf =
                                    tsdf_debug_values_all.index_select(
                                        0,
                                        prune_idx.to(tsdf_debug_values_all.device()).to(torch::kLong)).contiguous();
                                torch::Tensor selected_weights =
                                    tsdf_debug_weights_all.index_select(
                                        0,
                                        prune_idx.to(tsdf_debug_weights_all.device()).to(torch::kLong)).contiguous();

                                torch::Tensor geo_dev =
                                    geo.to(vox_key.device()).to(torch::kFloat32).view({-1});
                                torch::Tensor key_flat =
                                    vox_key.to(vox_key.device()).to(torch::kLong).reshape({-1});
                                torch::Tensor density_corners =
                                    geo_dev.index_select(0, key_flat)
                                        .view({N, 8})
                                        .contiguous();
                                torch::Tensor selected_density =
                                    density_corners.index_select(
                                        0,
                                        prune_idx.to(density_corners.device()).to(torch::kLong)).contiguous();

                                torch::Tensor selected_nvblox_sdf;
                                TsdfSample nvblox_sample =
                                    sampleNvbloxTsdfAtPointsWorld(
                                        selected_corner_points.view({-1, 3}).contiguous());
                                if (nvblox_sample.tsdf.defined() &&
                                    nvblox_sample.tsdf.numel() == prune_idx.size(0) * 8 &&
                                    nvblox_sample.weight.defined() &&
                                    nvblox_sample.weight.numel() == prune_idx.size(0) * 8 &&
                                    nvblox_sample.success.defined() &&
                                    nvblox_sample.success.numel() == prune_idx.size(0) * 8) {
                                    selected_nvblox_sdf =
                                        nvblox_sample.tsdf
                                            .to(torch::kCPU)
                                            .to(torch::kFloat32)
                                            .view({prune_idx.size(0), 8})
                                            .contiguous();
                                    torch::Tensor nvblox_weight =
                                        nvblox_sample.weight
                                            .to(torch::kCPU)
                                            .to(torch::kFloat32)
                                            .view({prune_idx.size(0), 8})
                                            .contiguous();
                                    torch::Tensor nvblox_success =
                                        nvblox_sample.success
                                            .to(torch::kCPU)
                                            .to(torch::kBool)
                                            .view({prune_idx.size(0), 8})
                                            .contiguous();
                                    torch::Tensor nvblox_valid =
                                        nvblox_success &
                                        torch::isfinite(selected_nvblox_sdf) &
                                        (nvblox_weight >= sdf_params_.tsdf_prune_min_weight_);
                                    selected_nvblox_sdf = torch::where(
                                        nvblox_valid,
                                        selected_nvblox_sdf,
                                        torch::full_like(
                                            selected_nvblox_sdf,
                                            std::numeric_limits<float>::quiet_NaN()));
                                } else {
                                    selected_nvblox_sdf = torch::full(
                                        {prune_idx.size(0), 8},
                                        std::numeric_limits<float>::quiet_NaN(),
                                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
                                }

                                torch::Tensor selected_colors;
                                try {
                                    torch::Tensor sh0 = voxel_model_->sh0();
                                    if (sh0.defined() && sh0.dim() >= 2 &&
                                        sh0.size(0) == N) {
                                        torch::Tensor sh0_sel =
                                            sh0.index_select(
                                                0,
                                                prune_idx.to(sh0.device()).to(torch::kLong)).contiguous();
                                        py::gil_scoped_acquire gil2;
                                        static py::module act_mod =
                                            py::module::import("src.utils.activation_utils");
                                        py::object rgb_py =
                                            act_mod.attr("shzero2rgb")(py::cast(sh0_sel));
                                        selected_colors =
                                            rgb_py.cast<torch::Tensor>().contiguous();
                                    }
                                } catch (const std::exception& e) {
                                    std::cerr << "[RERUN/sdf_pruned_nvblox] failed to compute voxel colors: "
                                              << e.what() << "\n";
                                }
                                if (!selected_colors.defined() ||
                                    selected_colors.numel() == 0 ||
                                    selected_colors.size(0) != prune_idx.size(0)) {
                                    selected_colors = torch::zeros(
                                        {prune_idx.size(0), 4},
                                        torch::TensorOptions()
                                            .dtype(torch::kFloat32)
                                            .device(debug_centers.device()));
                                    selected_colors.index_put_({torch::indexing::Slice(), 2}, 1.0f);
                                    selected_colors.index_put_({torch::indexing::Slice(), 3}, 0.7f);
                                }

                                torch::Tensor source_sdf_all = torch::zeros(
                                    {N},
                                    torch::TensorOptions().dtype(torch::kBool).device(prune_mask.device()));
                                if (tsdf_prune_mask.defined() &&
                                    tsdf_prune_mask.numel() == N) {
                                    source_sdf_all =
                                        (tsdf_prune_mask.to(prune_mask.device()).to(torch::kBool) &
                                         prune_mask.to(torch::kBool)).to(torch::kBool);
                                }
                                torch::Tensor source_svraster_all =
                                    (prune_mask.to(torch::kBool) & (~source_sdf_all)).to(torch::kBool);
                                torch::Tensor source_sdf_sel =
                                    source_sdf_all.index_select(
                                        0,
                                        prune_idx.to(source_sdf_all.device()).to(torch::kLong)).contiguous();
                                torch::Tensor source_svraster_sel =
                                    source_svraster_all.index_select(
                                        0,
                                        prune_idx.to(source_svraster_all.device()).to(torch::kLong)).contiguous();

                                sv::RerunVisualizerBridge::instance().visualizeSdfVoxelsRecording(
                                    "sdf_pruned_nvblox",
                                    debug_centers,
                                    debug_sizes,
                                    selected_corner_points,
                                    selected_tsdf,
                                    selected_weights,
                                    selected_density,
                                    selected_nvblox_sdf,
                                    selected_colors,
                                    prune_idx.to(torch::kCPU).to(torch::kLong),
                                    source_sdf_sel.to(torch::kCPU).to(torch::kBool),
                                    source_svraster_sel.to(torch::kCPU).to(torch::kBool),
                                    iter,
                                    "",
                                    false,
                                    "",
                                    4,
                                    std::max(0.0f, sdf_params_.tsdf_prune_surface_band_vox_) *
                                        tsdfMetricVoxelSize(),
                                    sdf_params_.tsdf_prune_min_weight_,
                                    false,
                                    "world/voxels_sdf_pruned_nvblox");
                            }
                        } catch (const std::exception& e) {
                            std::cerr << "[RERUN/sdf_pruned_nvblox] failed: "
                                      << e.what() << "\n";
                        }
                    }
                }

                // Save final pruned voxels (all criteria merged) for rerun visualization.
                if (prune_mask.defined() && prune_mask.numel() == N) {
                    auto prune_idx = prune_mask.to(torch::kBool).nonzero().squeeze(1); // [K]
                    if (prune_idx.numel() > 0) {
                        auto centers_world = voxel_model_->voxCenter(); // [N,3]
                        auto sizes_world   = voxel_model_->voxSize();   // [N] or [N,1]
                        if (centers_world.defined() &&
                            centers_world.dim() == 2 &&
                            centers_world.size(0) == N &&
                            centers_world.size(1) == 3 &&
                            sizes_world.defined() &&
                            sizes_world.size(0) == N)
                        {
                            debug_pruned_centers = centers_world.index({prune_idx}).clone();
                            debug_pruned_sizes   = sizes_world.index({prune_idx}).clone();
                            debug_has_pruned     = true;
                            torch::Tensor whole_run_pruned_by_tsdf;
                            if (tsdf_prune_mask.defined() &&
                                tsdf_prune_mask.numel() == N) {
                                whole_run_pruned_by_tsdf =
                                    tsdf_prune_mask
                                        .to(prune_idx.device())
                                        .to(torch::kBool)
                                        .contiguous()
                                        .index_select(0, prune_idx.to(torch::kLong))
                                        .contiguous();
                            }
                            torch::Tensor whole_run_pruned_by_near;
                            if (prune_mask_near.defined() &&
                                prune_mask_near.numel() == N) {
                                whole_run_pruned_by_near =
                                    prune_mask_near
                                        .to(prune_idx.device())
                                        .to(torch::kBool)
                                        .contiguous()
                                        .index_select(0, prune_idx.to(torch::kLong))
                                        .contiguous();
                            }
                            torch::Tensor whole_run_pruned_by_recent_unstable;
                            if (prune_mask_recent_unstable.defined() &&
                                prune_mask_recent_unstable.numel() == N) {
                                whole_run_pruned_by_recent_unstable =
                                    prune_mask_recent_unstable
                                        .to(prune_idx.device())
                                        .to(torch::kBool)
                                        .contiguous()
                                        .index_select(0, prune_idx.to(torch::kLong))
                                        .contiguous();
                            }
                            appendWholeRunPrunedVoxels(
                                iter,
                                debug_pruned_centers,
                                debug_pruned_sizes,
                                whole_run_pruned_by_tsdf,
                                whole_run_pruned_by_near,
                                whole_run_pruned_by_recent_unstable);

                            // Save far-only pruned voxels for a dedicated rerun topic.
                            debug_has_far_pruned = false;
                            if (prune_mask_real_outside_dense_core.defined() &&
                                prune_mask_real_outside_dense_core.numel() == N) {
                                auto far_pruned_mask =
                                    (prune_mask_real_outside_dense_core.to(torch::kBool) &
                                     prune_mask.to(torch::kBool)).to(torch::kBool);
                                auto far_idx = far_pruned_mask.nonzero().squeeze(1);
                                if (far_idx.numel() > 0) {
                                    debug_far_pruned_centers = centers_world.index({far_idx}).clone();
                                    debug_far_pruned_sizes   = sizes_world.index({far_idx}).clone();
                                    debug_has_far_pruned     = true;
                                }
                            }
                        } else {
                            debug_has_pruned = false;
                            debug_has_far_pruned = false;
                        }
                    } else {
                        debug_has_pruned = false;
                        debug_has_far_pruned = false;
                    }
                } else {
                    debug_has_pruned = false;
                    debug_has_far_pruned = false;
                }

                voxel_model_->pruning(prune_mask);
                if (rerun_params_.rerun_tsdf_unknown_voxels_) {
                    rerun_state_.rerun_tsdf_unknown_dirty_ = true;
                }
                if (rerun_params_.run_floaters_) {
                    rerun_state_.run_floaters_dirty_ = true;
                }
                const int new_n = voxel_model_->numVoxels();
                // Verbose [PRUNE/TOTAL] diagnostics disabled.

                // If pruning changed the voxel set (or shapes don’t match), recompute stats
                const int M = voxel_model_->numVoxels();
                const bool shape_ok =
                    stat.min_samp_interval.defined() &&
                    stat.min_samp_interval.dim() == 2 &&
                    stat.min_samp_interval.size(0) == M &&
                    stat.min_samp_interval.size(1) == 1;
                if (new_n != ori_n || !shape_ok) {
                    stat = voxel_model_->computeTrainingStat(tr_cams);
                }
            };

            // ---------------- SUBDIVIDE ----------------
            if (need_subdividing) {
                voxel_model_->setTopologyBirthContext(iter, static_cast<int>(tr_cams.size()));
                const int before = voxel_model_->numVoxels();
                if (before == 0) {
                    std::cout << "[SUBDIV:skip] M==0\n";
                } else {
                    auto vox_size_before = voxel_model_->voxSize(); // [N] or [N,1]
                    if (vox_size_before.dim() == 2 && vox_size_before.size(1) == 1) {
                        vox_size_before = vox_size_before.squeeze(1);
                    } else if (vox_size_before.dim() != 1) {
                        vox_size_before = vox_size_before.reshape({-1});
                    }
                    const double vox_size_min_before =
                        (vox_size_before.numel() > 0) ? vox_size_before.min().item<double>() : 0.0;
                    const double vox_size_max_before =
                        (vox_size_before.numel() > 0) ? vox_size_before.max().item<double>() : 0.0;

                    bool did_subdivide = false;
                    int64_t n_normal_candidates = 0;
                    int64_t n_subdiv_normal_selected = 0;
                    int64_t n_rendered_depth_blocked_normal = 0;

                    // 0) Helper masks and source accounting for subdivision.
                    auto rendered_depth_candidate_mask_for = [&](int64_t expected_n,
                                                                 const torch::Device& dev) {
                        if (!rendered_depth_insert_ &&
                            !mono_prior_params_.depthanything_fill_holes_) {
                            return torch::zeros(
                                {expected_n},
                                torch::TensorOptions().dtype(torch::kBool).device(dev));
                        }
                        auto m = voxel_model_->renderedDepthCandidateMask();
                        if (!m.defined() || m.numel() == 0) {
                            return torch::zeros(
                                {expected_n},
                                torch::TensorOptions().dtype(torch::kBool).device(dev));
                        }
                        if (m.dim() == 2 && m.size(1) == 1) {
                            m = m.squeeze(1);
                        }
                        m = m.to(dev).to(torch::kBool).contiguous().view({-1});
                        if (m.numel() != expected_n) {
                            return torch::zeros(
                                {expected_n},
                                torch::TensorOptions().dtype(torch::kBool).device(dev));
                        }
                        return m;
                    };
                    auto source_counts_for_current_voxels =
                        [&]() -> std::array<int64_t, 3>
                    {
                        const int64_t n = voxel_model_->numVoxels();
                        if (n <= 0) {
                            return {0, 0, 0};
                        }
                        const torch::Device dev = voxel_model_->voxCenter().device();
                        torch::Tensor rgbd_points_mask =
                            voxel_utils::normalizeBoolMaskOrZeros(voxel_model_->orbVoxelMask(), n, dev);
                        torch::Tensor inactive_geo_mask =
                            voxel_utils::normalizeBoolMaskOrZeros(voxel_model_->inactiveGeoVoxelMask(), n, dev);
                        torch::Tensor rgbd_fill_mask =
                            voxel_utils::normalizeBoolMaskOrZeros(voxel_model_->rgbdFillRenderHolesVoxelMask(), n, dev);
                        rgbd_points_mask =
                            (rgbd_points_mask & (~inactive_geo_mask) & (~rgbd_fill_mask))
                                .to(torch::kBool);
                        return {
                            rgbd_points_mask.sum().item<int64_t>(),
                            inactive_geo_mask.sum().item<int64_t>(),
                            rgbd_fill_mask.sum().item<int64_t>()};
                    };
                    auto selected_source_counts =
                        [&](const torch::Tensor& selected_mask) -> std::array<int64_t, 3>
                    {
                        if (!selected_mask.defined() || selected_mask.numel() <= 0) {
                            return {0, 0, 0};
                        }
                        const int64_t n = selected_mask.numel();
                        const torch::Device dev = selected_mask.device();
                        torch::Tensor selected =
                            voxel_utils::normalizeBoolMaskOrZeros(selected_mask, n, dev);
                        torch::Tensor rgbd_points_mask =
                            voxel_utils::normalizeBoolMaskOrZeros(voxel_model_->orbVoxelMask(), n, dev);
                        torch::Tensor inactive_geo_mask =
                            voxel_utils::normalizeBoolMaskOrZeros(voxel_model_->inactiveGeoVoxelMask(), n, dev);
                        torch::Tensor rgbd_fill_mask =
                            voxel_utils::normalizeBoolMaskOrZeros(voxel_model_->rgbdFillRenderHolesVoxelMask(), n, dev);
                        rgbd_points_mask =
                            (rgbd_points_mask & (~inactive_geo_mask) & (~rgbd_fill_mask))
                                .to(torch::kBool);
                        return {
                            (selected & rgbd_points_mask).sum().item<int64_t>(),
                            (selected & inactive_geo_mask).sum().item<int64_t>(),
                            (selected & rgbd_fill_mask).sum().item<int64_t>()};
                    };
                    auto account_subdivision_lineage_births =
                        [&](const std::array<int64_t, 3>& before_counts,
                            const std::array<int64_t, 3>& parent_counts)
                    {
                        const std::array<int64_t, 3> after_counts =
                            source_counts_for_current_voxels();
                        const int64_t rgbd_children =
                            std::max<int64_t>(0, after_counts[0] - before_counts[0] + parent_counts[0]);
                        const int64_t inactive_children =
                            std::max<int64_t>(0, after_counts[1] - before_counts[1] + parent_counts[1]);
                        const int64_t rgbd_fill_children =
                            std::max<int64_t>(0, after_counts[2] - before_counts[2] + parent_counts[2]);
                        sdf_state_.tsdf_ablation_rgbd_points_lineage_created_ += rgbd_children;
                        sdf_state_.tsdf_ablation_inactive_geo_lineage_created_ += inactive_children;
                        sdf_state_.tsdf_ablation_rgbd_fill_lineage_created_ += rgbd_fill_children;
                    };

                    auto compute_max_n_subdiv = [&]() -> int {
                        return std::round(
                            (opt_params_.subdivide_max_num_ - voxel_model_->numVoxels()) / 7.0);
                    };

                    // 1) One SVRaster subdivision pass for eligible voxels.
                    const int M = voxel_model_->numVoxels();
                    if (M > 0) {
                        auto min_samp_interval = stat.min_samp_interval; // [M,1]
                        if (did_subdivide ||
                            !min_samp_interval.defined() ||
                            min_samp_interval.size(0) != M) {
                            stat = voxel_model_->computeTrainingStat(tr_cams);
                            min_samp_interval = stat.min_samp_interval;
                        }
                        if (min_samp_interval.dim() == 1) {
                            min_samp_interval = min_samp_interval.view({M,1});
                        }

                        const float subdivide_samp_thres_now =
                            opt_params_.subdivide_samp_thres_;
                        const double subdivide_prop_now = std::clamp(
                            static_cast<double>(opt_params_.subdivide_prop_),
                            0.0, 1.0);

                        auto size_thres = min_samp_interval * subdivide_samp_thres_now; // [M,1]
                        auto vox_size = voxel_model_->voxSize(); // [M] or [M,1]
                        if (vox_size.dim() == 1) vox_size = vox_size.view({M,1});
                        else if (vox_size.dim() == 2 && vox_size.size(1) == 1) { /* ok */ }
                        else vox_size = vox_size.reshape({M,1});
                        auto vox_size_1d = vox_size.squeeze(1).contiguous();

                        auto large_enough = (vox_size * 0.5 > size_thres).squeeze(1); // [M] bool
                        auto octlv = voxel_model_->octLevel(); // [M] or [M,1]
                        if (octlv.defined() && octlv.dim() == 2 && octlv.size(1) == 1) {
                            octlv = octlv.squeeze(1);
                        }
                        auto non_finest =
                            (octlv.to(torch::kInt32) < voxel_model_->maxNumLevels()); // [M] bool

                        auto rendered_depth_candidate_mask =
                            rendered_depth_candidate_mask_for(M, vox_size_1d.device());
                        n_rendered_depth_blocked_normal =
                            rendered_depth_candidate_mask.sum().item<int64_t>();
                        auto valid_mask_svraster =
                            (large_enough &
                             non_finest &
                             (~rendered_depth_candidate_mask))
                                .to(torch::kBool); // [M]
                        auto normal_candidate_mask = valid_mask_svraster.clone();
                        n_normal_candidates = normal_candidate_mask.sum().item<int64_t>();

                        // Priority: may be undefined/empty right after structural changes.
                        auto priority = voxel_model_->subdivisionPriority(); // [M]
                        if (!priority.defined() || priority.numel() != M) {
                            priority = torch::zeros(
                                {M},
                                torch::TensorOptions().dtype(torch::kFloat32).device(normal_candidate_mask.device()));
                        } else if (priority.dim() == 2 && priority.size(1) == 1) {
                            priority = priority.squeeze(1);
                        } else if (priority.dim() != 1) {
                            priority = priority.reshape({M});
                        }

                        if (n_normal_candidates > 0) {
                            priority = priority * normal_candidate_mask.to(priority.scalar_type());

                            auto normal_selected_mask =
                                torch::zeros_like(normal_candidate_mask, torch::kBool);
                            if (iter <= opt_params_.subdivide_all_until_) {
                                normal_selected_mask = normal_candidate_mask.clone();
                            } else {
                                auto pos_idx = normal_candidate_mask.nonzero().squeeze(1); // [K]
                                auto pos_vals = priority.index({pos_idx});                 // [K]
                                double q = std::max(0.0, 1.0 - subdivide_prop_now);
                                auto thres = (pos_vals.numel() > 0)
                                        ? pos_vals.quantile(q)
                                        : torch::tensor(0.0, pos_vals.options());
                                if (pos_vals.numel() > 0) {
                                    auto pick = (pos_vals > thres); // [K]
                                    normal_selected_mask.index_put_({pos_idx}, pick);
                                    normal_selected_mask =
                                        (normal_selected_mask & normal_candidate_mask).to(torch::kBool);
                                }
                            }

                            int max_n_subdiv = compute_max_n_subdiv();
                            if (max_n_subdiv <= 0) {
                                std::cout << "[SUBDIV:skip] cap reached before normal stage (max_n_subdiv<=0)\n";
                                normal_selected_mask =
                                    torch::zeros_like(normal_candidate_mask, torch::kBool);
                            } else {
                                int64_t num_sel = normal_selected_mask.sum().item<int64_t>();
                                if (num_sel > max_n_subdiv) {
                                    auto pos_idx = normal_selected_mask.nonzero().squeeze(1); // [K]
                                    auto pos_vals = priority.index({pos_idx});                // [K]
                                    auto sort_pair = pos_vals.sort(/*dim=*/0, /*descending=*/true);
                                    auto order_desc = std::get<1>(sort_pair).to(torch::kLong).contiguous();
                                    auto keep_local = order_desc.index(
                                        {torch::indexing::Slice(0, max_n_subdiv)}).contiguous();
                                    auto keep_idx = pos_idx.index_select(0, keep_local).contiguous();
                                    normal_selected_mask =
                                        torch::zeros_like(normal_candidate_mask, torch::kBool);
                                    normal_selected_mask.index_put_({keep_idx}, true);
                                }
                            }

                            n_subdiv_normal_selected =
                                normal_selected_mask.sum().item<int64_t>();
                            if (n_subdiv_normal_selected > 0) {
                                const std::array<int64_t, 3> source_counts_before_subdiv =
                                    source_counts_for_current_voxels();
                                const std::array<int64_t, 3> selected_parent_counts =
                                    selected_source_counts(normal_selected_mask);
                                voxel_model_->subdividing(normal_selected_mask);
                                account_subdivision_lineage_births(
                                    source_counts_before_subdiv,
                                    selected_parent_counts);
                                did_subdivide = true;
                            }
                        }
                    }

                    auto vox_size_after = voxel_model_->voxSize(); // [N] or [N,1]
                    if (vox_size_after.dim() == 2 && vox_size_after.size(1) == 1) {
                        vox_size_after = vox_size_after.squeeze(1);
                    } else if (vox_size_after.dim() != 1) {
                        vox_size_after = vox_size_after.reshape({-1});
                    }
                    const double vox_size_min_after =
                        (vox_size_after.numel() > 0) ? vox_size_after.min().item<double>() : 0.0;
                    const double vox_size_max_after =
                        (vox_size_after.numel() > 0) ? vox_size_after.max().item<double>() : 0.0;
                    const int after = voxel_model_->numVoxels();

                    // 2) Apply post-subdivision bookkeeping.
                    if (did_subdivide) {
                        std::cout << "[SUBDIVIDING] " << std::setw(7) << before
                                  << " => "          << std::setw(7) << after
                                  << " (x" << std::fixed << std::setprecision(2)
                                  << (double)after / std::max(1, before) << ")\n";
                        refitSvrasterTsdfFromRegisteredKeyframes("subdivide");
                        if (rerun_params_.rerun_tsdf_unknown_voxels_) {
                            rerun_state_.rerun_tsdf_unknown_dirty_ = true;
                        }
                        if (rerun_params_.run_floaters_) {
                            rerun_state_.run_floaters_dirty_ = true;
                        }
                    } else {
                        std::cout << "[SUBDIV:skip] selected_total=0\n";
                    }
                    std::cout << "[SUBDIV/size] before_min=" << vox_size_min_before
                              << " before_max=" << vox_size_max_before
                              << " after_min=" << vox_size_min_after
                              << " after_max=" << vox_size_max_after
                              << "\n";
                }
            }
            if (need_pruning) {
                run_pruning();
            }
            // Keep SVRaster behavior: clear accumulated subdivision priority
            // after each adapt round that enters the subdivision branch.
            if (need_subdividing) {
                voxel_model_->resetSubdivisionPriority();
            }
            voxel_model_->createTrainer(
                opt_params_.geo_lr_,
                opt_params_.sh0_lr_,
                opt_params_.shs_lr_,
                opt_params_.optim_beta1_,
                opt_params_.optim_beta2_,
                opt_params_.optim_eps_,
                opt_params_.lr_decay_ckpt_,
                opt_params_.lr_decay_mult_
            );
            voxel_model_->schedulerLoadStateDict(sched_state);
            // Empty CUDA cache as SV does
            {
                py::gil_scoped_acquire gil;
                py::module_ torch_mod = py::module_::import("torch");
                torch_mod.attr("cuda").attr("empty_cache")();
            }
            last_densify_iter_ = iter;
        }
    }
    // Update learning rate
    voxel_model_->schedulerStep();

    if (rerun_params_.enable_rerun_) {
        // Keep the Rerun timeline alive for the full optimization, even in
        // filtered debug modes where no geometry is logged for many iterations.
        sv::RerunVisualizerBridge::instance().visualizeScalar(
            static_cast<double>(iter),
            iter,
            "world/debug/iteration");
        if (rerun_params_.run_tsdf_pruned_) {
            sv::RerunVisualizerBridge::instance().visualizeDebugScalar(
                "tsdf_pruned",
                static_cast<double>(iter),
                iter,
                "world/debug/iteration");
        }
        if (rerun_params_.rerun_tsdf_unknown_voxels_) {
            sv::RerunVisualizerBridge::instance().visualizeDebugScalar(
                "tsdf_unknown",
                static_cast<double>(iter),
                iter,
                "world/debug/iteration");
        }
        if (rerun_params_.run_whole_run_) {
            sv::RerunVisualizerBridge::instance().visualizeDebugScalar(
                "whole_run",
                static_cast<double>(iter),
                iter,
                "world/debug/iteration");
        }

        if (rerun_params_.rerun_tsdf_unknown_voxels_ && rerun_state_.rerun_tsdf_unknown_dirty_) {
            logTsdfUnknownVoxelsToRerun(iter, torch::Tensor());
            rerun_state_.rerun_tsdf_unknown_dirty_ = false;
        }
        if (rerun_params_.run_floaters_ &&
            (rerun_state_.run_floaters_dirty_ ||
             rerun_params_.run_floaters_stride_ <= 1 ||
             ((iter % rerun_params_.run_floaters_stride_) == 0))) {
            logFloatersToRerun(iter);
            rerun_state_.run_floaters_dirty_ = false;
        }
        logReconstructionMeshToRerun(iter);
        logNvbloxReconstructionMeshToRerun(iter);

        {
        // ----- 1) FULL VOXELS (unchanged) -----
        torch::Tensor centers_all = voxel_model_->voxCenter(); // [N,3]
        torch::Tensor sizes_all   = voxel_model_->voxSize();   // [N] or [N,1]
        // colors from SH0 + density as before
        torch::Tensor colors_all;
        {
            torch::Tensor sh0 = voxel_model_->sh0();
            {
                py::gil_scoped_acquire gil2;
                static py::module act_mod = py::module::import("src.utils.activation_utils");
                py::object rgb_py = act_mod.attr("shzero2rgb")(py::cast(sh0));
                colors_all = rgb_py.cast<torch::Tensor>().contiguous();
            }

            torch::Tensor density = voxel_model_->voxelDensityMean();
            if (density.defined() && density.numel() == centers_all.size(0)) {
                auto d_cpu = density.view({-1}).to(torch::kCPU);
                float d_min = d_cpu.min().item().toFloat();
                float d_max = d_cpu.max().item().toFloat();
                float eps   = 1e-6f;
                float range = d_max - d_min;

                torch::Tensor alpha_cpu;
                if (range < eps) {
                    alpha_cpu = torch::full_like(d_cpu, 0.8f);
                } else {
                    alpha_cpu = (d_cpu - d_min) / range;
                    alpha_cpu = alpha_cpu.clamp(0.05f, 1.0f);
                }
                auto col_cpu = colors_all.to(torch::kCPU);
                TORCH_CHECK(col_cpu.dim() == 2 &&
                            col_cpu.size(0) == alpha_cpu.size(0),
                            "colors and density must have same N");
                if (col_cpu.size(1) == 3) {
                    auto N = col_cpu.size(0);
                    auto col_rgba = torch::zeros({N, 4}, col_cpu.options());
                    col_rgba.index_put_(
                        {torch::indexing::Slice(), torch::indexing::Slice(0, 3)},
                        col_cpu
                    );
                    col_rgba.index_put_(
                        {torch::indexing::Slice(), 3},
                        alpha_cpu
                    );
                    colors_all = col_rgba.to(colors_all.device());
                } else if (col_cpu.size(1) == 4) {
                    col_cpu.index_put_(
                        {torch::indexing::Slice(), 3},
                        alpha_cpu
                    );
                    colors_all = col_cpu.to(colors_all.device());
                } else {
                    TORCH_CHECK(false, "colors must be [N,3] or [N,4]");
                }
            }
        }
        // Log full voxel field sparsely to keep Rerun memory bounded on long runs.
        // if ((iter % 20) == 0) {
        //     sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
        //         centers_all, sizes_all, colors_all, iter
        //     );
        // }
        sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
            centers_all, sizes_all, colors_all, iter
        );
        if (rerun_params_.run_whole_run_) {
            logWholeRunLiveVoxelsToRerun(iter, centers_all, sizes_all, colors_all);
        }

        // visualize real-only voxels each iteration:
        // real from PCD + promoted-artificial voxels
        {
            auto art_mask_all = voxel_model_->artificialMask();
            auto promoted_mask_all = voxel_model_->promotedartificialMask();
            auto real_mask_all = torch::ones(
                {centers_all.size(0)},
                torch::TensorOptions().dtype(torch::kBool).device(centers_all.device()));

            if (art_mask_all.defined()) {
                if (art_mask_all.dim() == 2 && art_mask_all.size(1) == 1) {
                    art_mask_all = art_mask_all.squeeze(1);
                }
                art_mask_all = art_mask_all.to(centers_all.device()).to(torch::kBool).contiguous().view({-1});
                if (art_mask_all.numel() == centers_all.size(0)) {
                    real_mask_all = real_mask_all & (~art_mask_all);
                }
            }
            if (promoted_mask_all.defined()) {
                if (promoted_mask_all.dim() == 2 && promoted_mask_all.size(1) == 1) {
                    promoted_mask_all = promoted_mask_all.squeeze(1);
                }
                promoted_mask_all = promoted_mask_all.to(centers_all.device()).to(torch::kBool).contiguous().view({-1});
                if (promoted_mask_all.numel() == centers_all.size(0)) {
                    real_mask_all = real_mask_all | promoted_mask_all;
                }
            }

            auto real_idx = torch::nonzero(real_mask_all).view({-1});
            if (real_idx.numel() > 0) {
                auto centers_real = centers_all.index_select(0, real_idx).contiguous();
                auto sizes_real = sizes_all.index_select(0, real_idx).contiguous();
                torch::Tensor colors_real;
                if (colors_all.defined() && colors_all.numel() > 0 &&
                    colors_all.dim() == 2 && colors_all.size(0) == centers_all.size(0)) {
                    colors_real = colors_all.index_select(0, real_idx).contiguous();
                }
                // sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
                //     centers_real, sizes_real, colors_real, iter, "world/voxels_real"
                // );
            }
        }
        // Keep artificials/all synchronized with final topology state of this iteration.
        // increasePcd() logs artificial topics at insertion time (pre-adapt); this call
        // rewrites artificials/all after prune/subdivide so it matches /voxels.
        {
            voxel_model_->logLiveOrbVoxels(iter, colors_all);
            voxel_model_->logLiveInactiveGeoVoxels(iter, colors_all);
            voxel_model_->logLiveRgbdFillRenderHolesVoxels(iter, colors_all);
            voxel_model_->logLiveDepthAnythingFillHolesVoxels(iter, colors_all);
            voxel_model_->logFinalartificialVoxels(iter);
            voxel_model_->logFinalPromotedartificialVoxels(iter);
        }

        {
        // ----- 2) NEAR VOXELS (debug overlay) -----
        if (debug_has_near &&
            debug_near_centers.defined() &&
            debug_near_centers.numel() > 0)
        {
            auto centers_near = debug_near_centers;         // [K,3] CUDA or CPU
            auto sizes_near   = debug_near_sizes;           // [K,1] or [K]

        // ensure sizes_near is [K,1] on CPU
        if (sizes_near.dim() == 1) {
            sizes_near = sizes_near.view({sizes_near.size(0), 1});
        } else if (sizes_near.dim() == 2 && sizes_near.size(1) == 1) {
            // ok
        } else {
            sizes_near = sizes_near.reshape({sizes_near.size(0), 1});
        }

        auto K = centers_near.size(0);
        torch::Tensor colors_near = torch::zeros({K, 4}, centers_near.options());
        colors_near.index_put_({torch::indexing::Slice(), 0}, 1.0f);  // R
        colors_near.index_put_({torch::indexing::Slice(), 3}, 0.7f);  // alpha

        sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
            centers_near,
            sizes_near,
            colors_near,
            iter,
            "world/voxels_near"    // <-- separate entity in blueprint
        );
        }
        // ----- 2b) GEOMETRIC-NEAR VOXELS (debug overlay) -----
        if (debug_has_near_geom &&
            debug_near_geom_centers.defined() &&
            debug_near_geom_centers.numel() > 0)
        {
            auto centers_near_geom = debug_near_geom_centers; // [K,3]
            auto sizes_near_geom   = debug_near_geom_sizes;   // [K,1] or [K]

            if (sizes_near_geom.dim() == 1) {
                sizes_near_geom = sizes_near_geom.view({sizes_near_geom.size(0), 1});
            } else if (!(sizes_near_geom.dim() == 2 && sizes_near_geom.size(1) == 1)) {
                sizes_near_geom = sizes_near_geom.reshape({sizes_near_geom.size(0), 1});
            }

            auto Kg = centers_near_geom.size(0);
            torch::Tensor colors_near_geom = torch::zeros({Kg, 4}, centers_near_geom.options());
            colors_near_geom.index_put_({torch::indexing::Slice(), 0}, 1.0f);  // R
            colors_near_geom.index_put_({torch::indexing::Slice(), 1}, 0.5f);  // G
            colors_near_geom.index_put_({torch::indexing::Slice(), 3}, 0.8f);  // alpha

            sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
                centers_near_geom,
                sizes_near_geom,
                colors_near_geom,
                iter,
                "world/voxels_near_geometric");
        }
        auto visualize_tsdf_class_boxes =
            [&](const torch::Tensor& centers_in,
                const torch::Tensor& sizes_in,
                bool has,
                float r,
                float g,
                float b,
                float a,
                const std::string& entity_path,
                const std::string& label)
        {
            if (!has || !centers_in.defined() || centers_in.numel() == 0) {
                return;
            }

            auto centers = centers_in;
            auto sizes = sizes_in;
            if (!sizes.defined() || sizes.numel() == 0) {
                return;
            }
            if (sizes.dim() == 1) {
                sizes = sizes.view({sizes.size(0), 1});
            } else if (!(sizes.dim() == 2 && sizes.size(1) == 1)) {
                sizes = sizes.reshape({sizes.size(0), 1});
            }

            auto K = centers.size(0);
            torch::Tensor colors = torch::zeros({K, 4}, centers.options());
            colors.index_put_({torch::indexing::Slice(), 0}, r);
            colors.index_put_({torch::indexing::Slice(), 1}, g);
            colors.index_put_({torch::indexing::Slice(), 2}, b);
            colors.index_put_({torch::indexing::Slice(), 3}, a);

            sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
                centers,
                sizes,
                colors,
                iter,
                entity_path);
        };

        visualize_tsdf_class_boxes(
            debug_tsdf_free_centers,
            debug_tsdf_free_sizes,
            debug_has_tsdf_free,
            0.0f, 0.9f, 1.0f, 0.45f,
            "world/tsdf/free_space",
            "TSDF free-space");
        visualize_tsdf_class_boxes(
            debug_tsdf_occupied_centers,
            debug_tsdf_occupied_sizes,
            debug_has_tsdf_occupied,
            1.0f, 0.0f, 1.0f, 0.45f,
            "world/tsdf/occupied_side",
            "TSDF occupied-side");
        visualize_tsdf_class_boxes(
            debug_tsdf_surface_centers,
            debug_tsdf_surface_sizes,
            debug_has_tsdf_surface,
            0.0f, 1.0f, 0.0f, 0.35f,
            "world/tsdf/surface_band",
            "TSDF surface-band");
        visualize_tsdf_class_boxes(
            debug_tsdf_unknown_centers,
            debug_tsdf_unknown_sizes,
            debug_has_tsdf_unknown,
            0.5f, 0.5f, 0.5f, 0.22f,
            "world/tsdf/unknown",
            "TSDF unknown");

        auto visualize_tsdf_corner_points =
            [&](const torch::Tensor& points_in,
                const torch::Tensor& colors_in,
                bool has,
                float r,
                float g,
                float b,
                float a,
                const std::string& entity_path,
                const std::string& label)
        {
            if (!has || !points_in.defined() || points_in.numel() == 0) {
                return;
            }
            auto points = points_in;
            if (points.dim() != 2 || points.size(1) != 3) {
                return;
            }

            torch::Tensor colors = colors_in;
            if (!colors.defined() || colors.numel() == 0) {
                colors = torch::zeros({points.size(0), 4}, points.options());
                colors.index_put_({torch::indexing::Slice(), 0}, r);
                colors.index_put_({torch::indexing::Slice(), 1}, g);
                colors.index_put_({torch::indexing::Slice(), 2}, b);
                colors.index_put_({torch::indexing::Slice(), 3}, a);
            }

            sv::RerunVisualizerBridge::instance().visualizePoints3D(
                points,
                colors,
                iter,
                entity_path,
                0.012f);
        };

        visualize_tsdf_corner_points(
            debug_tsdf_free_corner_points,
            torch::Tensor(),
            debug_has_tsdf_free_corners,
            0.0f, 0.9f, 1.0f, 0.95f,
            "world/tsdf_samples/free_space_corners",
            "TSDF free-space");
        visualize_tsdf_corner_points(
            debug_tsdf_occupied_corner_points,
            torch::Tensor(),
            debug_has_tsdf_occupied_corners,
            1.0f, 0.0f, 1.0f, 0.95f,
            "world/tsdf_samples/occupied_side_corners",
            "TSDF occupied-side");
        visualize_tsdf_corner_points(
            debug_tsdf_surface_corner_points,
            torch::Tensor(),
            debug_has_tsdf_surface_corners,
            0.0f, 1.0f, 0.0f, 0.95f,
            "world/tsdf_samples/surface_band_corners",
            "TSDF surface-band");
        visualize_tsdf_corner_points(
            debug_tsdf_unknown_corner_points,
            debug_tsdf_unknown_corner_colors,
            debug_has_tsdf_unknown_corners,
            0.5f, 0.5f, 0.5f, 0.95f,
            "world/tsdf_samples/unknown_corners",
            "TSDF unknown");

        // ----- 4) FINAL-PRUNED VOXELS (debug overlay) -----
        if (debug_has_pruned &&
            debug_pruned_centers.defined() &&
            debug_pruned_centers.numel() > 0)
        {
            auto centers_pruned = debug_pruned_centers; // [K_prune,3]
            auto sizes_pruned   = debug_pruned_sizes;   // [K_prune,1] or [K_prune]

        if (sizes_pruned.dim() == 1) {
            sizes_pruned = sizes_pruned.view({sizes_pruned.size(0), 1});
        } else if (sizes_pruned.dim() == 2 && sizes_pruned.size(1) == 1) {
            // ok
        } else {
            sizes_pruned = sizes_pruned.reshape({sizes_pruned.size(0), 1});
        }

        auto Kp = centers_pruned.size(0);
        torch::Tensor colors_pruned = torch::zeros({Kp, 4}, centers_pruned.options());
        colors_pruned.index_put_({torch::indexing::Slice(), 0}, 1.0f);  // R
        colors_pruned.index_put_({torch::indexing::Slice(), 1}, 1.0f);  // G (yellow)
        colors_pruned.index_put_({torch::indexing::Slice(), 3}, 0.8f);  // alpha

        // sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
        //     centers_pruned,
        //     sizes_pruned,
        //     colors_pruned,
        //     iter,
        //     "world/voxels_pruned"   // separate entity path
        // );
        }

        // ----- 6) FAR-ONLY PRUNED VOXELS (debug overlay) -----
        if (debug_has_far_pruned &&
            debug_far_pruned_centers.defined() &&
            debug_far_pruned_centers.numel() > 0)
        {
            auto centers_far_pruned = debug_far_pruned_centers; // [K_far,3]
            auto sizes_far_pruned   = debug_far_pruned_sizes;   // [K_far,1] or [K_far]

            if (sizes_far_pruned.dim() == 1) {
                sizes_far_pruned = sizes_far_pruned.view({sizes_far_pruned.size(0), 1});
            } else if (sizes_far_pruned.dim() == 2 && sizes_far_pruned.size(1) == 1) {
                // ok
            } else {
                sizes_far_pruned = sizes_far_pruned.reshape({sizes_far_pruned.size(0), 1});
            }

            auto Kf = centers_far_pruned.size(0);
            torch::Tensor colors_far_pruned = torch::zeros({Kf, 4}, centers_far_pruned.options());
            colors_far_pruned.index_put_({torch::indexing::Slice(), 0}, 1.0f);  // R
            colors_far_pruned.index_put_({torch::indexing::Slice(), 3}, 0.9f);  // alpha

            sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
                centers_far_pruned,
                sizes_far_pruned,
                colors_far_pruned,
                iter,
                "world/far_voxels");
        }

        }
        }
    }

    if (mDevice == torch::kCUDA) torch::cuda::synchronize();

    {
        torch::NoGradGuard no_grad;
        ema_loss_for_log_ = 0.4f * loss.item<float>() + 0.6f * ema_loss_for_log_;

        if (keyframe_record_interval_ &&
            getIteration() % keyframe_record_interval_ == 0)
            recordKeyframeRendered(
                masked_image,
                gt_image,
                viewpoint_cam->fid_,
                result_dir_, result_dir_, result_dir_
            );
        auto iter_end_timing = std::chrono::steady_clock::now();
        auto iter_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        iter_end_timing - iter_start_timing).count();

         // Log and save
         if (training_report_interval_ && (getIteration() % training_report_interval_ == 0))
             sv::VoxelTrainer::trainingReport(
                 getIteration(),
                 opt_params_.iterations_,
                 Ll1,
                 loss,
                 ema_loss_for_log_,
                 mse,
                 iter_time,
                 *voxel_model_,
                 *scene_,
                 pipe_params_,
                 background_
             );

        if ((all_keyframes_record_interval_ && getIteration() % all_keyframes_record_interval_ == 0)
            )
        {
            renderAndRecordAllKeyframes();
            savePly(result_dir_ / std::to_string(iteration_) / "ply");
        }
        
        if (loop_closure_iteration_)
            loop_closure_iteration_ = false;

    }
}

void VoxelMapper::combineMappingOperations()
{
    // Get Mapping Operations
    while (mpSLAM->getAtlas()->hasMappingOperation()) {
        ORB_SLAM3::MappingOperation opr =
            mpSLAM->getAtlas()->getAndPopMappingOperation();
        switch (opr.meOperationType)
        {
        case ORB_SLAM3::MappingOperation::OprType::LocalMappingBA:
        {
        bool kf_changed = false;
            // Get new keyframes
            auto& associated_kfs = opr.associatedKeyFrames();
            // Add keyframes to the scene
            for (auto& kf : associated_kfs) {
                // Keyframe Id
                auto kfid = std::get<0>(kf);
                std::shared_ptr<VoxelKeyframe> pkf = scene_->getKeyframe(kfid);
                // If the keyframe is already in the scene, only update the pose.
                // Otherwise create a new one
                if (pkf) {
                    auto& pose = std::get<2>(kf);
                    pkf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>());
                //  pkf->computeTransformTensors();
                    // Give local BA keyframes times of use
                    increaseKeyframeTimesOfUse(pkf, local_BA_increased_times_of_use_);
                    kf_changed = true;
                }
                else {
                handleNewKeyframe(kf);                   // still void
                }
            }
            // Get new points
            auto& associated_points = opr.associatedMapPoints();
            auto& points = std::get<0>(associated_points);
            auto& colors = std::get<1>(associated_points);

            // Add new points to the model
            const int iter = getIteration();
            if (initial_mapped_ && points.size() >= 30) {
                torch::NoGradGuard no_grad;
                std::unique_lock<std::mutex> lock_render(mutex_render_);
            //  voxel_model_->increasePcd(points, colors, getIteration(), kfs_for_bounding);
                // py::object sched_state = voxel_model_->schedulerStateDict();

                // Build training camera list from the keyframes we keep in the scene
                std::vector<sv::MiniCam> tr_cams;
                tr_cams.reserve(scene_->keyframes().size());
                for (auto& kv : scene_->keyframes()) {
                    // OLD:
                    // if (kv.second) tr_cams.push_back(kv.second->toMiniCam());
                    if (kv.second) {
                        tr_cams.push_back(
                            kv.second->toMiniCam(kv.second->image_height_, kv.second->image_width_));
                    }
                }
                if (points.size() >= 30) {
                    if (rerun_params_.enable_rerun_) {
                        appendAndLogOrbRawPointBatchToRerun(
                            points,
                            colors,
                            iter);
                        voxel_model_->setNextRealInsertionRerunEntityPath(
                            "world/orb/voxels_created");
                    }
                    voxel_model_->increasePcd(
                        points,
                        colors,
                        getIteration(),
                        tr_cams);
                    if (sensor_type_ == RGBD) {
                        const sv::VoxelModel::IncreasePcdStats insert_stats =
                            voxel_model_->lastIncreasePcdStats();
                        sdf_state_.tsdf_ablation_rgbd_points_created_ += insert_stats.new_voxels;
                        sdf_state_.tsdf_ablation_rgbd_points_lineage_created_ += insert_stats.new_voxels;
                        if (insert_stats.new_voxels > 0) {
                            refitSvrasterTsdfFromRegisteredKeyframes("orb_increasePcd");
                        }
                    }
                    if (rerun_params_.enable_rerun_) {
                        voxel_model_->setNextRealInsertionRerunEntityPath("");
                    }
                    if (voxel_model_ && voxel_model_->consumeartificialFillFlag()) {
                        last_artificial_fill_iter_ = static_cast<int64_t>(iter);
                        std::cout << "[VoxelMapper] artificial fill happened at iter "
                                << iter << "\n";
                    }
                }
                // voxel_model_->createTrainer(
                //                             opt_params_.geo_lr_,
                //                             opt_params_.sh0_lr_,
                //                             opt_params_.shs_lr_,
                //                             opt_params_.optim_beta1_,
                //                             opt_params_.optim_beta2_,
                //                             opt_params_.optim_eps_,
                //                             opt_params_.lr_decay_ckpt_,
                //                             opt_params_.lr_decay_mult_);
                // voxel_model_->schedulerLoadStateDict(sched_state);
            }

        }
        break;

        case ORB_SLAM3::MappingOperation::OprType::LoopClosingBA:
        {
            std::cout << "[Voxel Mapper]Loop Closure Detected."
                    << std::endl;

            bool kf_changed = false;
            // Get the loop keyframe scale modification factor
            float loop_kf_scale = opr.mfScale;

            // Get new keyframes (scaled transformation applied in ORB-SLAM3)
            auto& associated_kfs = opr.associatedKeyFrames();

            // std::vector<std::shared_ptr<VoxelKeyframe>> kfs_for_bounding;

             // Mark the transformed points to avoid transforming more than once
             torch::Tensor point_not_transformed_flags =
                 torch::full(
                     {voxel_model_->center_.size(0)},
                     true,
                     torch::TensorOptions().device(device_type_).dtype(torch::kBool));
             if (record_loop_ply_)
                 savePly(result_dir_ / (std::to_string(getIteration()) + "_0_before_loop_correction"));
             int num_transformed = 0;
             // Add keyframes to the scene
             for (auto& kf : associated_kfs) {
                 // Keyframe Id
                 auto kfid = std::get<0>(kf);
                 std::shared_ptr<VoxelKeyframe> pkf = scene_->getKeyframe(kfid);
                 // In case new points are added in handleNewKeyframe()
                 int64_t num_new_points = voxel_model_->center_.size(0) - point_not_transformed_flags.size(0);
                 if (num_new_points > 0)
                     point_not_transformed_flags = torch::cat({
                         point_not_transformed_flags,
                         torch::full({num_new_points}, true, point_not_transformed_flags.options())},
                         /*dim=*/0);
                 // If kf is already in the scene, evaluate the change in pose,
                 // if too large we perform loop correction on its visible model points.
                 // If not in the scene, create a new one.
                 if (pkf) {
                     auto& pose = std::get<2>(kf);
                     // If is loop closure kf
 // if (std::get<4>(kf)) {
 // renderAndRecordKeyframe(pkf, result_dir_, "_0_before_loop_correction");
                         Sophus::SE3f original_pose = pkf->getPosef(); // original_pose = old, inv_pose = new
                         Sophus::SE3f inv_pose = pose.inverse();
                         Sophus::SE3f diff_pose = inv_pose * original_pose;
                         bool large_rot = !diff_pose.rotationMatrix().isApprox(
                             Eigen::Matrix3f::Identity(), large_rot_th_);
                         bool large_trans = !diff_pose.translation().isMuchSmallerThan(
                             1.0, large_trans_th_);
                         if (large_rot || large_trans) {
                             std::cout << "[Voxel Mapper]Large loop correction detected, transforming visible points of kf "
                                     << kfid << std::endl;
                             diff_pose.translation() -= inv_pose.translation(); // t = (R_new * t_old + t_new) - t_new
                             diff_pose.translation() *= loop_kf_scale;          // t = s * (R_new * t_old)
                             diff_pose.translation() += inv_pose.translation(); // t = (s * R_new * t_old) + t_new
                             torch::Tensor diff_pose_tensor =
                                 tensor_utils::EigenMatrix2TorchTensor(
                                     diff_pose.matrix(), device_type_).transpose(0, 1);
                            //  {
                            //      std::unique_lock<std::mutex> lock_render(mutex_render_);
                            //      voxel_model_->scaledTransformVisiblePointsOfKeyframe(
                            //          point_not_transformed_flags,
                            //          diff_pose_tensor,
                            //          pkf->world_view_transform_,
                            //          pkf->full_proj_transform_,
                            //          pkf->creation_iter_,
                            //          stableNumIterExistence(),
                            //          num_transformed,
                            //          loop_kf_scale); // selected xyz *= s
                            //  }
                             // Give loop keyframes times of use
                             increaseKeyframeTimesOfUse(pkf, loop_closure_increased_times_of_use_);
                         }
                     pkf->setPose(
                         pose.unit_quaternion().cast<double>(),
                         pose.translation().cast<double>());
                    //  pkf->computeTransformTensors();
 // if (std::get<4>(kf)) renderAndRecordKeyframe(pkf, result_dir_, "_2_after_pose_correction");

                    kf_changed = true;
                 }
                 else {
                     handleNewKeyframe(kf);
                     pkf = scene_->getKeyframe(kfid);
                 }
             }
             if (record_loop_ply_)
                 savePly(result_dir_ / (std::to_string(getIteration()) + "_1_after_loop_correction"));
 // keyframesToJson(result_dir_ / (std::to_string(getIteration()) + "_0_before_loop_correction"));
 
             // Get new points (scaled transformation applied in ORB-SLAM3, so this step is performed at last to avoid scaling twice)
             auto& associated_points = opr.associatedMapPoints();
             auto& points = std::get<0>(associated_points);
             auto& colors = std::get<1>(associated_points);

             // Add new points to the model
             const int iter = getIteration();
             if (initial_mapped_ && points.size() >= 30) {
                std::cout << "adds new points" << std::endl;
                torch::NoGradGuard no_grad;
                std::unique_lock<std::mutex> lock_render(mutex_render_);

                // Match Photo-SLAM behavior: insert loop-closure associated points.
                std::vector<sv::MiniCam> tr_cams;
                tr_cams.reserve(scene_->keyframes().size());
                for (auto& kv : scene_->keyframes()) {
                    if (kv.second) {
                        tr_cams.push_back(
                            kv.second->toMiniCam(
                                kv.second->image_height_,
                                kv.second->image_width_));
                    }
                }
                if (points.size() >= 30) {
                    if (rerun_params_.enable_rerun_) {
                        appendAndLogOrbRawPointBatchToRerun(
                            points,
                            colors,
                            iter);
                        voxel_model_->setNextRealInsertionRerunEntityPath(
                            "world/orb/voxels_created");
                    }
                    voxel_model_->increasePcd(
                        points,
                        colors,
                        iter,
                        tr_cams);
                    if (sensor_type_ == RGBD) {
                        const sv::VoxelModel::IncreasePcdStats insert_stats =
                            voxel_model_->lastIncreasePcdStats();
                        if (insert_stats.new_voxels > 0) {
                            refitSvrasterTsdfFromRegisteredKeyframes("loop_orb_increasePcd");
                        }
                    }
                    if (rerun_params_.enable_rerun_) {
                        voxel_model_->setNextRealInsertionRerunEntityPath("");
                    }
                    if (voxel_model_ && voxel_model_->consumeartificialFillFlag()) {
                        last_artificial_fill_iter_ = static_cast<int64_t>(iter);
                        std::cout << "[VoxelMapper] artificial fill happened at iter "
                                << iter << "\n";
                    }
                }
             }

            // Mark this iteration
            loop_closure_iteration_ = true;
         }
         break;
 
         case ORB_SLAM3::MappingOperation::OprType::ScaleRefinement:
         {
             std::cout << "[Voxel Mapper]Scale refinement Detected. Transforming all kfs and points..."
                       << std::endl;
 
             float s = opr.mfScale;
             Sophus::SE3f& T = opr.mT;
             if (initial_mapped_) {
                 // Apply the scaled transformation on gaussian model points
                 {
                     std::unique_lock<std::mutex> lock_render(mutex_render_);
                    //  voxel_model_->applyScaledTransformation(s, T);
                 }
                 // Apply the scaled transformation to the scene
                //  scene_->applyScaledTransformation(s, T);
             }
             else { // TODO: the workflow should not come here, delete this branch
                 // Apply the scaled transformation to the cached points
                 for (auto& pt : scene_->cached_point_cloud_) {
                     // pt <- (s * Ryw * pt + tyw)
                     auto& pt_xyz = pt.second.xyz_;
                     pt_xyz *= s;
                     pt_xyz = T.cast<double>() * pt_xyz;
                 }
 
                 // Apply the scaled transformation on gaussian keyframes
                 for (auto& kfit : scene_->keyframes()) {
                     std::shared_ptr<VoxelKeyframe> pkf = kfit.second;
                     Sophus::SE3f Twc = pkf->getPosef().inverse();
                     Twc.translation() *= s;
                     Sophus::SE3f Tyc = T * Twc;
                     Sophus::SE3f Tcy = Tyc.inverse();
                     std::cout << "ScaleRefinement: kf " << Tcy.translation() << std::endl;
                     pkf->setPose(Tcy.unit_quaternion().cast<double>(), Tcy.translation().cast<double>());
                    //  pkf->computeTransformTensors();
                 }
             }
         }
         break;
 
         default:
         {
             throw std::runtime_error("MappingOperation type not supported!");
         }
         break;
         }
     }
 }

 bool VoxelMapper::hasMetInitialMappingConditions() {
     if (!mpSLAM->isShutDown() &&
         mpSLAM->GetNumKeyframes() >= min_num_initial_map_kfs_ &&
         mpSLAM->getAtlas()->hasMappingOperation())
         return true;
 
     bool conditions_met = false;
     return conditions_met;
}

bool VoxelMapper::hasMetIncrementalMappingConditions() {
     if (!mpSLAM->isShutDown() &&
         mpSLAM->getAtlas()->hasMappingOperation())
         return true;
 
     bool conditions_met = false;
     return conditions_met;
}

void VoxelMapper::generateKfidRandomShuffle()
{
     if (scene_->keyframes().empty())
         return;
 
     std::size_t nkfs = scene_->keyframes().size();
     kfid_shuffle_.resize(nkfs);
     std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);
     std::mt19937 g(rd_());
     std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);
 
     kfid_shuffled_ = true;
}

std::shared_ptr<VoxelKeyframe> VoxelMapper::useOneRandomSlidingWindowKeyframe()
{
    // If no keyframes, return nullptr
    if (scene_->keyframes().empty())
        return nullptr;

    // If not shuffled yet, build shuffle
    if (!kfid_shuffled_)
        generateKfidRandomShuffle();

    std::shared_ptr<VoxelKeyframe> viewpoint_cam = nullptr;
    int random_cam_idx;

    if (kfid_shuffled_) {
        int start_shuffle_idx = kfid_shuffle_idx_;
        do {
            // Next shuffled idx
            ++kfid_shuffle_idx_;
            if (kfid_shuffle_idx_ >= kfid_shuffle_.size())
                kfid_shuffle_idx_ = 0;
            // Add 1 time of use to all kfs if they are all unavalible
            if (kfid_shuffle_idx_ == start_shuffle_idx)
                for (auto& kfit : scene_->keyframes())
                    increaseKeyframeTimesOfUse(kfit.second, 1);
            // Get viewpoint kf
            random_cam_idx = kfid_shuffle_[kfid_shuffle_idx_];
            auto random_cam_it = scene_->keyframes().begin();
            for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
                ++random_cam_it;
            viewpoint_cam = (*random_cam_it).second;
        } while (viewpoint_cam->remaining_times_of_use_ <= 0);
    }

    // Count used times
    auto viewpoint_fid = viewpoint_cam->fid_;
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
        kfs_used_times_[viewpoint_fid] = 1;
    else
        ++kfs_used_times_[viewpoint_fid];
    
    // Handle times of use
    --(viewpoint_cam->remaining_times_of_use_);

    return viewpoint_cam;
}

std::shared_ptr<VoxelKeyframe> VoxelMapper::useOneRandomKeyframe()
 {
     if (scene_->keyframes().empty())
         return nullptr;
 
     // Get randomly
     int nkfs = static_cast<int>(scene_->keyframes().size());
     int random_cam_idx = std::rand() / ((RAND_MAX + 1u) / nkfs);
     auto random_cam_it = scene_->keyframes().begin();
     for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
         ++random_cam_it;
     std::shared_ptr<VoxelKeyframe> viewpoint_cam = (*random_cam_it).second;
 
     // Count used times
     auto viewpoint_fid = viewpoint_cam->fid_;
     if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
         kfs_used_times_[viewpoint_fid] = 1;
     else
         ++kfs_used_times_[viewpoint_fid];
 
     return viewpoint_cam;
 }

void VoxelMapper::cullKeyframes()
{
    // Ask ORB-SLAM3 which keyframe IDs are still “live”
    std::unordered_set<unsigned long> kfids =
        mpSLAM->getAtlas()->GetCurrentKeyFrameIds();

     std::vector<unsigned long> kfids_to_erase;
     std::size_t nkfs = scene_->keyframes().size();
     kfids_to_erase.reserve(nkfs);
     for (auto& kfit : scene_->keyframes()) {
         unsigned long kfid = kfit.first;
         if (kfids.find(kfid) == kfids.end()) {
             kfids_to_erase.emplace_back(kfid);
         }
     }
 
     for (auto& kfid : kfids_to_erase) {
         scene_->keyframes().erase(kfid);
     }
}

void VoxelMapper::handleNewKeyframe(
    std::tuple<
        unsigned long,    // 0: keyframe ID
        unsigned long,    // 1: camera ID
        Sophus::SE3f,     // 2: pose
        cv::Mat,          // 3: RGB image
        bool,             // 4: loop‐closure flag (unused here)
        cv::Mat,          // 5: auxiliary (unused here)
        std::vector<float>, // 6: keypoint pixel coords (unused here)
        std::vector<float>, // 7: keypoint local coords (unused here)
        std::string> &kf       // 8: image filename (relative or absolute)
)
{
    // ─── Create a new VoxelKeyframe, exactly like Photo-SLAM’s Gaussian case ─
    std::shared_ptr<VoxelKeyframe> pkf  = std::make_shared<VoxelKeyframe>(std::get<0>(kf), getIteration());
    pkf->znear_ = z_near_;
    // Pose
    auto& pose = std::get<2>(kf);
    pkf->setPose(
        pose.unit_quaternion().cast<double>(),
        pose.translation().cast<double>()
    );
    cv::Mat imgRGB_undistorted, imgAux_undistorted;
    // Camera
    sv::Camera& camera = scene_->cameras_.at(std::get<1>(kf));
    pkf->setCameraParams(camera);

    // Image (left if STEREO)
    cv::Mat imgRGB = std::get<3>(kf);
    if (this->sensor_type_ == STEREO)
        imgRGB_undistorted = imgRGB;
    else
        camera.undistortImage(imgRGB, imgRGB_undistorted);
    // Auxiliary Image
    cv::Mat imgAux = std::get<5>(kf);
    if (this->sensor_type_ == RGBD)
        camera.undistortImage(imgAux, imgAux_undistorted);
    else
        imgAux_undistorted = imgAux;

    pkf->original_image_ =
        tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
    pkf->img_filename_ = std::get<8>(kf);
    pkf->source_frame_id_ = voxel_utils::parseFrameIdFromPath(pkf->img_filename_);
    pkf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
    pkf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
    pkf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
     
    // Add the new keyframe to the scene
    // pkf->computeTransformTensors();
    scene_->addKeyframe(pkf, &kfid_shuffled_);

    // Give new keyframes times of use and add it to the training sliding window
    increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());

    // Get dense point cloud from the new keyframe to accelerate training
    pkf->img_undist_ = imgRGB_undistorted;
    pkf->img_auxiliary_undist_ = imgAux_undistorted;

    pkf->kps_pixel_ = std::move(std::get<6>(kf));
    pkf->kps_point_local_ = std::move(std::get<7>(kf));
    
    if (isdoingInactiveGeoDensify())
        increasePcdByKeyframeInactiveGeoDensify(pkf);
    if (mono_prior_params_.depthanything_fill_holes_)
        scheduleDepthAnythingFillHoles(pkf);
    if (rendered_depth_insert_)
        increasePcdByKeyframeRenderedDepthInsertion(pkf);

    // Prepare multi resolution images for training
    if (device_type_ == torch::kCUDA) {
        cv::cuda::GpuMat img_gpu;
        img_gpu.upload(pkf->img_undist_);
        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::cuda::GpuMat img_resized;
            cv::cuda::resize(img_gpu, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
        }
    }
    else {
        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::Mat img_resized;
            cv::resize(pkf->img_undist_, img_resized,
                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
        }
    }

    logKeyframeCameraToRerunRecordings(
        pkf,
        std::get<0>(kf),
        /*log_reconstruction_mesh=*/true);

    // // ─── TSDF: integrate this new keyframe into the selected TSDF backend. ───
    const bool need_nvblox =
        (useNvbloxTsdfBackend() &&
         (sdf_params_.use_tsdf_mapping_ || sdf_params_.use_tsdf_pruning_ || sdf_params_.tsdf_density_init_)) ||
        sdf_params_.use_tsdf_planning_ ||
        rerun_params_.save_nvblox_mesh_eval_ ||
        (rerun_params_.enable_rerun_ && rerun_params_.rerun_reconstruction_mesh_) ||
        (rerun_params_.enable_rerun_ && rerun_params_.run_whole_run_) ||
        (rerun_params_.enable_rerun_ && rerun_params_.run_sdf_pruned_nvblox_) ||
        (rerun_params_.enable_rerun_ && rerun_params_.rerun_nvblox_mesh_ && !rerun_params_.load_saved_nvblox_mesh_);
    if (sensor_type_ == RGBD && (need_nvblox ||
                                 (sdf_params_.use_tsdf_mapping_ && useSvrasterTsdfBackend()))) {
        cv::Mat depth_meters;
        if (voxel_utils::depthMatToMeters(pkf->img_auxiliary_undist_, depth_meters)) {
            if (need_nvblox) {
                integrateKeyframeIntoNvblox(*pkf, depth_meters);
                if (rerun_params_.rerun_tsdf_unknown_voxels_) {
                    rerun_state_.rerun_tsdf_unknown_dirty_ = true;
                }
                if (rerun_params_.run_floaters_) {
                    rerun_state_.run_floaters_dirty_ = true;
                }
            }
            if (sdf_params_.use_tsdf_mapping_ && useSvrasterTsdfBackend()) {
                integrateKeyframeIntoSvrasterSdf(*pkf, depth_meters);
                if (rerun_params_.rerun_tsdf_unknown_voxels_) {
                    rerun_state_.rerun_tsdf_unknown_dirty_ = true;
                }
                if (rerun_params_.run_floaters_) {
                    rerun_state_.run_floaters_dirty_ = true;
                }
            }
        }
    }
}

void VoxelMapper::increasePcdByKeyframeInactiveGeoDensify(
    std::shared_ptr<VoxelKeyframe> pkf)
{
    torch::NoGradGuard no_grad;

    const int iter = getIteration();

    // Pose of camera in world frame
    Sophus::SE3f Twc = pkf->getPosef().inverse();

    switch (this->sensor_type_)
    {
    case MONOCULAR:
    {
        assert(pkf->kps_pixel_.size() % 2 == 0);
        int N = pkf->kps_pixel_.size() / 2;

        // Keypoints and local 3D (camera frame)
        torch::Tensor kps_pixel_tensor = torch::from_blob(
            pkf->kps_pixel_.data(),
            {N, 2},
            torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);

        torch::Tensor kps_point_local_tensor = torch::from_blob(
            pkf->kps_point_local_.data(),
            {N, 3},
            torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);

        torch::Tensor kps_has3D_tensor = torch::where(
            kps_point_local_tensor.index({torch::indexing::Slice(), 2}) > 0.0f,
            true,
            false);

        // RGB image → torch
        cv::cuda::GpuMat rgb_gpu;
        rgb_gpu.upload(pkf->img_undist_);
        torch::Tensor colors = tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_gpu);
        colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();

        // Photo-SLAM’s neighborhood densification
        auto result =
            monocularPinholeInactiveGeoDensifyBySearchingNeighborhoodKeypoints(
                kps_pixel_tensor,
                kps_has3D_tensor,
                kps_point_local_tensor,
                colors,
                monocular_inactive_geo_densify_max_pixel_dist_,
                pkf->intr_,
                pkf->image_width_);

        torch::Tensor& points3D_valid = std::get<0>(result);
        torch::Tensor& colors_valid   = std::get<1>(result);

        // Transform points to world coordinates
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        voxel_utils::transformPoints(points3D_valid, Twc_tensor);

        // Add new points to the cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        } else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid},   /*dim=*/0);
        }
    }
    break;

    case STEREO:
    {
        cv::cuda::GpuMat rgb_left_gpu, rgb_right_gpu;
        cv::cuda::GpuMat gray_left_gpu, gray_right_gpu;

        rgb_left_gpu.upload(pkf->img_undist_);
        rgb_right_gpu.upload(pkf->img_auxiliary_undist_);

        // RGB → gray
        cv::cuda::cvtColor(rgb_left_gpu,  gray_left_gpu,  cv::COLOR_RGB2GRAY);
        cv::cuda::cvtColor(rgb_right_gpu, gray_right_gpu, cv::COLOR_RGB2GRAY);

        // float → uint8
        gray_left_gpu.convertTo(gray_left_gpu,   CV_8UC1, 255.0);
        gray_right_gpu.convertTo(gray_right_gpu, CV_8UC1, 255.0);

        // Compute disparity
        cv::cuda::GpuMat cv_disp;
        stereo_cv_sgm_->compute(gray_left_gpu, gray_right_gpu, cv_disp);
        cv_disp.convertTo(cv_disp, CV_32F, 1.0 / 16.0);

        // Reproject to 3D
        cv::cuda::GpuMat cv_points3D;
        cv::cuda::reprojectImageTo3D(cv_disp, cv_points3D, stereo_Q_, 3);

        // To torch
        torch::Tensor disp = tensor_utils::cvGpuMat2TorchTensor_Float32(cv_disp);
        disp = disp.flatten(0, 1).contiguous();

        torch::Tensor points3D =
            tensor_utils::cvGpuMat2TorchTensor_Float32(cv_points3D);
        points3D = points3D.permute({1, 2, 0}).flatten(0, 1).contiguous();

        torch::Tensor colors =
            tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_left_gpu);
        colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();

        // Keep only points near tracked keypoints + valid disparity range
        torch::Tensor point_valid_flags = torch::full(
            {disp.size(0)},
            false,
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));

        int nkps_twice = pkf->kps_pixel_.size();
        int width      = pkf->image_width_;
        for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
            int idx = static_cast<int>(pkf->kps_pixel_[kpidx]) +
                      static_cast<int>(pkf->kps_pixel_[kpidx + 1]) * width;
            point_valid_flags[idx] = true;
        }

        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(
                disp > static_cast<float>(stereo_cv_sgm_->getMinDisparity()),
                true,
                false));

        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(
                disp < static_cast<float>(stereo_cv_sgm_->getNumDisparities()),
                true,
                false));
        torch::Tensor points3D_valid = points3D.index({point_valid_flags});
        torch::Tensor colors_valid   = colors.index({point_valid_flags});

        // Transform to world
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        voxel_utils::transformPoints(points3D_valid, Twc_tensor);

        // Cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        } else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid},   /*dim=*/0);
        }
    }
    break;

    case RGBD:
    {
        cv::cuda::GpuMat img_rgb_gpu, img_depth_gpu;
        img_rgb_gpu.upload(pkf->img_undist_);
        img_depth_gpu.upload(pkf->img_auxiliary_undist_);

        // cv::cuda::GpuMat → torch::Tensor
        torch::Tensor rgb = tensor_utils::cvGpuMat2TorchTensor_Float32(img_rgb_gpu);
        rgb = rgb.permute({1, 2, 0}).flatten(0, 1).contiguous();

        torch::Tensor depth = tensor_utils::cvGpuMat2TorchTensor_Float32(img_depth_gpu);
        depth = depth.flatten(0, 1).contiguous();

        sv::Camera& camera = scene_->cameras_.at(pkf->camera_id_);

        // Filter depth using tracked keypoints + RGBD_min/max. This preserves
        // the original sparse RGB-D insertion path.
        torch::Tensor point_valid_flags = torch::full(
            {depth.size(0)},
            false,   // Note Photo-SLAM uses false here and then sets only around kps
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));

        int nkps_twice = pkf->kps_pixel_.size();
        int width      = pkf->image_width_;
        for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
            int idx = static_cast<int>(pkf->kps_pixel_[kpidx]) +
                      static_cast<int>(pkf->kps_pixel_[kpidx + 1]) * width;
            point_valid_flags[idx] = true;
        }

        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(depth > RGBD_min_depth_, true, false));
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(depth < RGBD_max_depth_, true, false));
        
        torch::Tensor inactive_geo_flags = point_valid_flags.clone();
        torch::Tensor rgbd_render_hole_flags = torch::zeros_like(point_valid_flags);

        int64_t rgbd_render_hole_selected = 0;
        if (rgbd_fill_render_holes_ && voxel_model_ &&
            pkf->image_height_ > 0 && pkf->image_width_ > 0) {
            std::unordered_map<std::string, torch::Tensor> render_pkg;
            {
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                render_pkg = voxel_model_->render(
                    pkf->toMiniCam(pkf->image_height_, pkf->image_width_),
                    pkf->image_height_,
                    pkf->image_width_,
                    torch::Tensor(),
                    "dontcare",
                    false,
                    std::nullopt,
                    true,
                    false,
                    true,
                    false,
                    false,
                    sv::RenderOpts{});
            }

            torch::Tensor render_depth_cpu;
            torch::Tensor render_alpha_cpu;
            torch::Tensor render_n_contrib_cpu;
            if (voxel_utils::renderPkgToDepthAlphaMaps(
                    render_pkg,
                    pkf->image_height_,
                    pkf->image_width_,
                    render_depth_cpu,
                    render_alpha_cpu,
                    render_n_contrib_cpu)) {
                torch::Tensor depth_cpu =
                    depth.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
                if (depth_cpu.dim() == 1) {
                    depth_cpu = depth_cpu.view({pkf->image_height_, pkf->image_width_});
                }

                std::vector<uint8_t> fill_mask(
                    static_cast<size_t>(pkf->image_height_) *
                    static_cast<size_t>(pkf->image_width_),
                    0);
                auto depth_acc = depth_cpu.accessor<float, 2>();
                auto render_depth_acc = render_depth_cpu.accessor<float, 2>();
                auto render_n_contrib_acc = render_n_contrib_cpu.accessor<int, 2>();

                const int active_hole_max_n_contrib = 0;
                const int stride = std::max(1, rgbd_fill_render_holes_stride_);
                std::vector<int64_t> selected_idx;
                selected_idx.reserve(
                    static_cast<size_t>((pkf->image_height_ + stride - 1) / stride) *
                    static_cast<size_t>((pkf->image_width_ + stride - 1) / stride));

                for (int y = 0; y < pkf->image_height_; ++y) {
                    for (int x = 0; x < pkf->image_width_; ++x) {
                        const float z_rgbd = depth_acc[y][x];
                        if (!std::isfinite(z_rgbd) ||
                            z_rgbd <= RGBD_min_depth_ ||
                            z_rgbd >= RGBD_max_depth_) {
                            continue;
                        }

                        const float z_render = render_depth_acc[y][x];
                        const int n_contrib = render_n_contrib_acc[y][x];
                        const bool render_hole =
                            (n_contrib <= active_hole_max_n_contrib) &&
                            (!std::isfinite(z_render) ||
                             z_render <= 1.0e-6f);
                        if (!render_hole) {
                            continue;
                        }

                        if ((x % stride) == 0 && (y % stride) == 0) {
                            selected_idx.push_back(
                                static_cast<int64_t>(y) *
                                static_cast<int64_t>(pkf->image_width_) +
                                static_cast<int64_t>(x));
                        }
                    }
                }

                if (rgbd_fill_render_holes_max_points_per_kf_ > 0 &&
                    static_cast<int>(selected_idx.size()) >
                        rgbd_fill_render_holes_max_points_per_kf_) {
                    std::vector<int64_t> keep;
                    keep.reserve(static_cast<size_t>(
                        rgbd_fill_render_holes_max_points_per_kf_));
                    if (rgbd_fill_render_holes_max_points_per_kf_ == 1) {
                        keep.push_back(selected_idx[selected_idx.size() / 2]);
                    } else {
                        const double step =
                            static_cast<double>(selected_idx.size() - 1) /
                            static_cast<double>(
                                rgbd_fill_render_holes_max_points_per_kf_ - 1);
                        for (int i = 0; i < rgbd_fill_render_holes_max_points_per_kf_; ++i) {
                            const size_t idx = static_cast<size_t>(
                                std::llround(step * static_cast<double>(i)));
                            keep.push_back(selected_idx[
                                std::min(idx, selected_idx.size() - 1)]);
                        }
                    }
                    selected_idx.swap(keep);
                }

                for (const int64_t idx : selected_idx) {
                    fill_mask[static_cast<size_t>(idx)] = 1;
                }
                rgbd_render_hole_selected = static_cast<int64_t>(selected_idx.size());

                if (rgbd_render_hole_selected > 0) {
                    torch::Tensor fill_flags =
                        torch::from_blob(
                            fill_mask.data(),
                            {depth.size(0)},
                            torch::TensorOptions()
                                .dtype(torch::kUInt8)
                                .device(torch::kCPU))
                            .clone()
                            .to(device_type_)
                            .to(torch::kBool);
                    rgbd_render_hole_flags =
                        (fill_flags & (~inactive_geo_flags)).to(torch::kBool);
                    point_valid_flags =
                        (inactive_geo_flags | rgbd_render_hole_flags).to(torch::kBool);
                    rgbd_render_hole_selected =
                        rgbd_render_hole_flags.sum().item<int64_t>();
                }
            }
        }

        // Reproject to 3D (camera coordinates)
        torch::Tensor points3D_all;

        switch (camera.model_id_)
        {
        case Camera::PINHOLE:
        {
            points3D_all = voxel_utils::reprojectDepthPinholeVoxel(
                depth,
                pkf->intr_,
                pkf->image_width_);
        }
        break;

        case Camera::FISHEYE:
        {
            // TODO: support fisheye camera?
            throw std::runtime_error("[VoxelMapper] Fisheye cameras are not supported currently!");
        }
        break;

        default:
        {
            throw std::runtime_error("[VoxelMapper] Invalid camera model!");
        }
        break;
        }

        torch::Tensor points3D_inactive_geo = points3D_all.index({inactive_geo_flags});
        torch::Tensor colors_inactive_geo = rgb.index({inactive_geo_flags});
        torch::Tensor points3D_rgbd_holes = points3D_all.index({rgbd_render_hole_flags});
        torch::Tensor colors_rgbd_holes = rgb.index({rgbd_render_hole_flags});
        // Transform to world coordinates
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        voxel_utils::transformPoints(points3D_inactive_geo, Twc_tensor);
        voxel_utils::transformPoints(points3D_rgbd_holes, Twc_tensor);

        // Cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_inactive_geo;
            depth_cache_colors_ = colors_inactive_geo;
            rgbd_fill_render_holes_cache_points_ = points3D_rgbd_holes;
            rgbd_fill_render_holes_cache_colors_ = colors_rgbd_holes;
        } else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_inactive_geo}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_inactive_geo},   /*dim=*/0);
            rgbd_fill_render_holes_cache_points_ =
                torch::cat({rgbd_fill_render_holes_cache_points_, points3D_rgbd_holes}, /*dim=*/0);
            rgbd_fill_render_holes_cache_colors_ =
                torch::cat({rgbd_fill_render_holes_cache_colors_, colors_rgbd_holes},   /*dim=*/0);
        }
    }
    break;

    default:
    {
        throw std::runtime_error("[VoxelMapper] Unsupported sensor type!");
    }
    break;
    }

    pkf->done_inactive_geo_densify_ = true;
    ++depth_cached_;

    if (depth_cached_ >= max_depth_cached_) {
        depth_cached_ = 0;

        // Add new points to the voxel model
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        std::vector<sv::MiniCam> tr_cams;
        tr_cams.reserve(scene_->keyframes().size());
        for (auto& kv : scene_->keyframes()) {
            if (kv.second) {
                tr_cams.push_back(
                    kv.second->toMiniCam(kv.second->image_height_, kv.second->image_width_));
            }
        }
        auto flush_real_cache =
            [&](torch::Tensor& cache_points,
                torch::Tensor& cache_colors,
                const std::string& entity_path)
        {
            if (!cache_points.defined() || cache_points.dim() != 2 || cache_points.size(0) <= 0) {
                return;
            }
            // This entity path is used by VoxelModel both for optional Rerun
            // insertion visualization and, more importantly, for persistent
            // source provenance flags. Set it regardless of auxiliary Rerun
            // topic settings so debug source masks remain correct.
            voxel_model_->setNextRealInsertionRerunEntityPath(entity_path);
            const bool tsdf_init_context_set =
                prepareSvrasterTsdfInitContext(pkf);
            voxel_model_->increasePcd(
                cache_points,
                cache_colors,
                getIteration(),
                tr_cams);
            if (tsdf_init_context_set) {
                clearSvrasterTsdfInitContext();
            }
            const sv::VoxelModel::IncreasePcdStats insert_stats =
                voxel_model_->lastIncreasePcdStats();
            if (entity_path == "world/voxels_inactive_geo_densify/created") {
                sdf_state_.tsdf_ablation_inactive_geo_created_ += insert_stats.new_voxels;
                sdf_state_.tsdf_ablation_inactive_geo_lineage_created_ += insert_stats.new_voxels;
            } else if (entity_path == "world/rgbd_fill_render_holes/created") {
                sdf_state_.tsdf_ablation_rgbd_fill_created_ += insert_stats.new_voxels;
                sdf_state_.tsdf_ablation_rgbd_fill_lineage_created_ += insert_stats.new_voxels;
            }
            if (insert_stats.new_voxels > 0) {
                refitSvrasterTsdfFromRegisteredKeyframes(entity_path);
            }
            voxel_model_->setNextRealInsertionRerunEntityPath("");
            cache_points = torch::Tensor();
            cache_colors = torch::Tensor();
        };

        if ((depth_cache_points_.defined() && depth_cache_points_.dim() == 2 && depth_cache_points_.size(0) > 0) ||
            (rgbd_fill_render_holes_cache_points_.defined() &&
             rgbd_fill_render_holes_cache_points_.dim() == 2 &&
             rgbd_fill_render_holes_cache_points_.size(0) > 0)) {
            flush_real_cache(
                depth_cache_points_,
                depth_cache_colors_,
                "world/voxels_inactive_geo_densify/created");
            flush_real_cache(
                rgbd_fill_render_holes_cache_points_,
                rgbd_fill_render_holes_cache_colors_,
                "world/rgbd_fill_render_holes/created");
        }
    }
}

void VoxelMapper::increasePcdByKeyframeRenderedDepthInsertion(
    std::shared_ptr<VoxelKeyframe> pkf)
{
    torch::NoGradGuard no_grad;

    if (!pkf || !voxel_model_) {
        return;
    }

    const int iter = getIteration();
    if (sensor_type_ != MONOCULAR) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    const int H = pkf->image_height_;
    const int W = pkf->image_width_;
    if (H <= 0 || W <= 0) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    sv::MiniCam cam = pkf->toMiniCam(H, W);
    std::unordered_map<std::string, torch::Tensor> render_pkg;
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        render_pkg = voxel_model_->render(
            cam,
            H,
            W,
            torch::Tensor(),
            "dontcare",
            false,
            std::nullopt,
            true,
            true,
            true,
            false,
            false,
            sv::RenderOpts{});
    }

    auto it_depth = render_pkg.find("raw_depth");
    if (it_depth == render_pkg.end()) it_depth = render_pkg.find("depth");
    auto it_normal = render_pkg.find("raw_normal");
    if (it_normal == render_pkg.end()) it_normal = render_pkg.find("normal");
    if (it_depth == render_pkg.end() || it_normal == render_pkg.end() ||
        !it_depth->second.defined() || !it_normal->second.defined()) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    auto render_depth = it_depth->second;
    if (render_depth.dim() == 4 && render_depth.size(0) == 1) render_depth = render_depth.squeeze(0);
    if (render_depth.dim() == 3 && render_depth.size(0) >= 1) render_depth = render_depth.index({0});
    if (render_depth.dim() != 2) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    auto render_normal = it_normal->second;
    if (render_normal.dim() == 4 && render_normal.size(0) == 1) render_normal = render_normal.squeeze(0);
    if (render_normal.dim() != 3 || render_normal.size(0) < 3) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }
    if (render_normal.size(0) > 3) {
        render_normal = render_normal.index({torch::indexing::Slice(0, 3)});
    }

    auto depth_cpu = render_depth.to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto normal_cpu = render_normal.to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto image_cpu = pkf->original_image_.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (image_cpu.dim() == 4 && image_cpu.size(0) == 1) image_cpu = image_cpu.squeeze(0).contiguous();

    if (depth_cpu.size(0) != H || depth_cpu.size(1) != W ||
        normal_cpu.size(1) != H || normal_cpu.size(2) != W ||
        image_cpu.dim() != 3 || image_cpu.size(0) < 3 ||
        image_cpu.size(1) != H || image_cpu.size(2) != W) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    auto depth_acc = depth_cpu.accessor<float, 2>();
    auto normal_acc = normal_cpu.accessor<float, 3>();
    auto image_acc = image_cpu.accessor<float, 3>();

    std::vector<int64_t> selected_frontier_idx;
    selected_frontier_idx.reserve(static_cast<size_t>(rendered_depth_insert_max_points_per_kf_));
    const int frontier_radius = std::max(1, rendered_depth_insert_frontier_radius_px_);
    const int stride = std::max(1, rendered_depth_insert_stride_);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const float z = depth_acc[y][x];
            if (!std::isfinite(z) || z <= 0.0f) {
                continue;
            }

            bool has_miss_neighbor = false;
            for (int dy = -frontier_radius; dy <= frontier_radius && !has_miss_neighbor; ++dy) {
                const int yy = y + dy;
                if (yy < 0 || yy >= H) continue;
                for (int dx = -frontier_radius; dx <= frontier_radius; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    const int xx = x + dx;
                    if (xx < 0 || xx >= W) continue;
                    const float z_nb = depth_acc[yy][xx];
                    if (!std::isfinite(z_nb) || z_nb <= 0.0f) {
                        has_miss_neighbor = true;
                        break;
                    }
                }
            }
            if (!has_miss_neighbor) continue;

            if ((x % stride) == 0 && (y % stride) == 0) {
                selected_frontier_idx.push_back(static_cast<int64_t>(y) * static_cast<int64_t>(W) + x);
            }
        }
    }

    if (rendered_depth_insert_max_points_per_kf_ > 0 &&
        static_cast<int>(selected_frontier_idx.size()) > rendered_depth_insert_max_points_per_kf_) {
        std::vector<int64_t> keep;
        keep.reserve(static_cast<size_t>(rendered_depth_insert_max_points_per_kf_));
        if (rendered_depth_insert_max_points_per_kf_ == 1) {
            keep.push_back(selected_frontier_idx[selected_frontier_idx.size() / 2]);
        } else {
            const double step = static_cast<double>(selected_frontier_idx.size() - 1) /
                                static_cast<double>(rendered_depth_insert_max_points_per_kf_ - 1);
            for (int i = 0; i < rendered_depth_insert_max_points_per_kf_; ++i) {
                const size_t idx = static_cast<size_t>(std::llround(step * static_cast<double>(i)));
                keep.push_back(selected_frontier_idx[std::min(idx, selected_frontier_idx.size() - 1)]);
            }
        }
        selected_frontier_idx.swap(keep);
    }

    if (selected_frontier_idx.empty()) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    const float fx = (cam.fx > 1e-6f)
        ? cam.fx
        : (0.5f * static_cast<float>(W) / std::max(cam.tanfovx, 1e-6f));
    const float fy = (cam.fy > 1e-6f)
        ? cam.fy
        : (0.5f * static_cast<float>(H) / std::max(cam.tanfovy, 1e-6f));
    if (fx <= 1e-6f || fy <= 1e-6f) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    const Sophus::SE3f Twc = pkf->getPosef().inverse();
    const Eigen::Vector3f cam_pos_world = Twc.translation();
    const float offset_m =
        std::max(0.0f, rendered_depth_insert_normal_offset_vox_) * voxel_model_->fixedVoxSize();

    std::vector<float> frontier_points_world;
    std::vector<float> candidate_points_world;
    std::vector<float> candidate_colors;
    frontier_points_world.reserve(selected_frontier_idx.size() * 3);
    candidate_points_world.reserve(selected_frontier_idx.size() * 3);
    candidate_colors.reserve(selected_frontier_idx.size() * 3);

    for (const int64_t flat_idx : selected_frontier_idx) {
        const int y = static_cast<int>(flat_idx / W);
        const int x = static_cast<int>(flat_idx % W);
        const float z = depth_acc[y][x];
        if (!std::isfinite(z) || z <= 0.0f) {
            continue;
        }

        Eigen::Vector3f normal_world(
            normal_acc[0][y][x],
            normal_acc[1][y][x],
            normal_acc[2][y][x]);
        if (!std::isfinite(normal_world.x()) ||
            !std::isfinite(normal_world.y()) ||
            !std::isfinite(normal_world.z())) {
            continue;
        }
        const float normal_norm = normal_world.norm();
        if (!(normal_norm > 1e-6f)) {
            continue;
        }
        normal_world /= normal_norm;

        const float Xc = (static_cast<float>(x) - cam.cx) / fx * z;
        const float Yc = (static_cast<float>(y) - cam.cy) / fy * z;
        const Eigen::Vector3f p_cam(Xc, Yc, z);
        Eigen::Vector3f p_world = Twc * p_cam;
        if (!std::isfinite(p_world.x()) ||
            !std::isfinite(p_world.y()) ||
            !std::isfinite(p_world.z())) {
            continue;
        }

        Eigen::Vector3f ray_dir = p_world - cam_pos_world;
        const float ray_norm = ray_dir.norm();
        if (!(ray_norm > 1e-6f)) {
            continue;
        }
        ray_dir /= ray_norm;
        if (normal_world.dot(ray_dir) < 0.0f) {
            normal_world = -normal_world;
        }

        const Eigen::Vector3f p_candidate = p_world + offset_m * normal_world;
        if (!std::isfinite(p_candidate.x()) ||
            !std::isfinite(p_candidate.y()) ||
            !std::isfinite(p_candidate.z())) {
            continue;
        }

        frontier_points_world.push_back(p_world.x());
        frontier_points_world.push_back(p_world.y());
        frontier_points_world.push_back(p_world.z());

        candidate_points_world.push_back(p_candidate.x());
        candidate_points_world.push_back(p_candidate.y());
        candidate_points_world.push_back(p_candidate.z());

        candidate_colors.push_back(std::clamp(image_acc[0][y][x], 0.0f, 1.0f));
        candidate_colors.push_back(std::clamp(image_acc[1][y][x], 0.0f, 1.0f));
        candidate_colors.push_back(std::clamp(image_acc[2][y][x], 0.0f, 1.0f));
    }

    const int64_t num_candidates = static_cast<int64_t>(candidate_points_world.size() / 3);

    if (num_candidates > 0) {
        auto points_tensor = torch::from_blob(
            candidate_points_world.data(),
            {num_candidates, 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone().to(device_type_);
        auto colors_tensor = torch::from_blob(
            candidate_colors.data(),
            {num_candidates, 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone().to(device_type_);

        std::vector<sv::MiniCam> tr_cams;
        tr_cams.reserve(scene_->keyframes().size());
        for (auto& kv : scene_->keyframes()) {
            if (kv.second) {
                tr_cams.push_back(
                    kv.second->toMiniCam(kv.second->image_height_, kv.second->image_width_));
            }
        }

        std::unique_lock<std::mutex> lock_render(mutex_render_);
        voxel_model_->setRenderedDepthCandidateRealAdjacency(
            rendered_depth_insert_require_real_adjacency_,
            rendered_depth_insert_adjacency_radius_cells_);
        voxel_model_->setNextRenderedDepthCandidateInsertion(
            true,
            "",
            voxel_utils::kRenderedCandidateSourceDepthInsert);
        const bool tsdf_init_context_set =
            prepareSvrasterTsdfInitContext(pkf);
        voxel_model_->increasePcd(points_tensor, colors_tensor, iter, tr_cams);
        if (tsdf_init_context_set) {
            clearSvrasterTsdfInitContext();
        }
        voxel_model_->setNextRenderedDepthCandidateInsertion(false);
    }

    updateRenderedDepthCandidateLifecycle();
}

void VoxelMapper::updateRenderedDepthCandidateLifecycle()
{
    if ((!rendered_depth_insert_ && !mono_prior_params_.depthanything_fill_holes_) || !voxel_model_) {
        return;
    }

    const int iter = getIteration();
    std::unique_lock<std::mutex> lock_render(mutex_render_);

    auto candidate_mask = voxel_model_->renderedDepthCandidateMask();
    if (!candidate_mask.defined() || candidate_mask.numel() == 0) {
        return;
    }

    auto flatten_mask = [](torch::Tensor t) {
        if (t.dim() == 2 && t.size(1) == 1) t = t.squeeze(1);
        return t.contiguous().view({-1});
    };

    auto centers = voxel_model_->voxCenter();
    auto sizes = voxel_model_->voxSize();
    if (!centers.defined() || !sizes.defined()) {
        return;
    }
    candidate_mask = flatten_mask(candidate_mask.to(centers.device()).to(torch::kBool));
    if (candidate_mask.numel() != centers.size(0)) {
        return;
    }

    auto support_count = voxel_model_->renderedDepthCandidateSupportCount();
    auto last_seen_kf = voxel_model_->renderedDepthCandidateLastSeenKf();
    auto source_kind = voxel_model_->renderedDepthCandidateSourceKind();
    if (!support_count.defined() || !last_seen_kf.defined()) {
        return;
    }
    support_count = flatten_mask(support_count.to(centers.device()).to(torch::kInt32));
    last_seen_kf = flatten_mask(last_seen_kf.to(centers.device()).to(torch::kInt32));
    if (!source_kind.defined()) {
        source_kind = torch::zeros(
            {candidate_mask.numel()},
            torch::TensorOptions().dtype(torch::kInt32).device(centers.device()));
    } else {
        source_kind = flatten_mask(source_kind.to(centers.device()).to(torch::kInt32));
    }
    if (support_count.numel() != candidate_mask.numel() ||
        last_seen_kf.numel() != candidate_mask.numel() ||
        source_kind.numel() != candidate_mask.numel()) {
        return;
    }

    const int32_t current_kf_count = static_cast<int32_t>(scene_->keyframes().size());
    auto promote_mask =
        (candidate_mask &
         (support_count >= opt_params_.rendered_depth_candidate_promote_min_support_)).to(torch::kBool);
    const int64_t n_promote = promote_mask.sum().item<int64_t>();
    if (n_promote > 0) {
        voxel_model_->promoteRenderedDepthCandidates(promote_mask);
    }

    candidate_mask = voxel_model_->renderedDepthCandidateMask();
    support_count = voxel_model_->renderedDepthCandidateSupportCount();
    last_seen_kf = voxel_model_->renderedDepthCandidateLastSeenKf();
    source_kind = voxel_model_->renderedDepthCandidateSourceKind();
    if (!candidate_mask.defined() || !support_count.defined() || !last_seen_kf.defined()) {
        return;
    }
    candidate_mask = flatten_mask(candidate_mask.to(centers.device()).to(torch::kBool));
    support_count = flatten_mask(support_count.to(centers.device()).to(torch::kInt32));
    last_seen_kf = flatten_mask(last_seen_kf.to(centers.device()).to(torch::kInt32));
    if (!source_kind.defined()) {
        source_kind = torch::zeros(
            {candidate_mask.numel()},
            torch::TensorOptions().dtype(torch::kInt32).device(centers.device()));
    } else {
        source_kind = flatten_mask(source_kind.to(centers.device()).to(torch::kInt32));
    }
    if (candidate_mask.numel() != centers.size(0) ||
        support_count.numel() != centers.size(0) ||
        last_seen_kf.numel() != centers.size(0) ||
        source_kind.numel() != centers.size(0)) {
        return;
    }

    auto age_kf = (torch::full(
        {candidate_mask.size(0)},
        current_kf_count,
        torch::TensorOptions().dtype(torch::kInt32).device(candidate_mask.device())) -
        last_seen_kf).to(torch::kInt32);
    auto prune_mask =
        (candidate_mask &
         (support_count < opt_params_.rendered_depth_candidate_promote_min_support_) &
         (last_seen_kf >= 0) &
         (age_kf >= opt_params_.rendered_depth_candidate_prune_kf_age_)).to(torch::kBool);
    const int64_t n_prune = prune_mask.sum().item<int64_t>();
    if (n_prune > 0) {
        // Do not prune here. Topology-removing pruning requires the normal adapt
        // path, which rebuilds the trainer/optimizer after the topology change.
    }
}

bool VoxelMapper::isStopped() const {
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    return this->stopped_;
}

void VoxelMapper::signalStop(const bool going_to_stop)
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    this->stopped_ = going_to_stop;
}

void VoxelMapper::increaseKeyframeTimesOfUse(
        const std::shared_ptr<VoxelKeyframe>& pkf,
        int times)
 {
     pkf->remaining_times_of_use_ += times;
 }

void VoxelMapper::writeKeyframeUsedTimes(std::filesystem::path result_dir, std::string name_suffix)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    std::filesystem::path result_path = result_dir / ("keyframe_used_times" + name_suffix + ".txt");
    std::ofstream out_stream;
    out_stream.open(result_path, std::ios::app);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open json at " + result_path.string());

    out_stream << "##[Voxel Mapper]Iteration " << getIteration() << " keyframe id, used times, remaining times:\n";
    for (const auto& used_times_it : kfs_used_times_)
        out_stream << used_times_it.first << " "
                   << used_times_it.second << " "
                   << scene_->keyframes().at(used_times_it.first)->remaining_times_of_use_
                   << "\n";
    out_stream << "##=========================================" <<std::endl;

    out_stream.close();
}

void VoxelMapper::recordKeyframeRendered(
    torch::Tensor&           rendered,
    torch::Tensor&           ground_truth,
    unsigned long            kfid,
    std::filesystem::path    result_img_dir,
    std::filesystem::path    result_gt_dir,
    std::filesystem::path    result_loss_dir,
    std::string              name_suffix)
{
    if (record_rendered_image_) {
         auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered);
         cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
         image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
         cv::imwrite(result_img_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + ".jpg"), image_cv);
     }
 
     if (record_ground_truth_image_) {
         auto gt_image_cv = tensor_utils::torchTensor2CvMat_Float32(ground_truth);
         cv::cvtColor(gt_image_cv, gt_image_cv, CV_RGB2BGR);
         gt_image_cv.convertTo(gt_image_cv, CV_8UC3, 255.0f);
         cv::imwrite(result_gt_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_gt.jpg"), gt_image_cv);
     }
 
     if (record_loss_image_) {
         torch::Tensor loss_tensor = torch::abs(rendered - ground_truth);
         auto loss_image_cv = tensor_utils::torchTensor2CvMat_Float32(loss_tensor);
         cv::cvtColor(loss_image_cv, loss_image_cv, CV_RGB2BGR);
         loss_image_cv.convertTo(loss_image_cv, CV_8UC3, 255.0f);
         cv::imwrite(result_loss_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_loss.jpg"), loss_image_cv);
     }
}

void VoxelMapper::renderAndRecordKeyframe(
    std::shared_ptr<VoxelKeyframe> pkf,
    float&       dssim,
    float&       psnr,
    double&      render_ms,
    const std::filesystem::path& result_img_dir,
    const std::filesystem::path& result_gt_dir,
    const std::filesystem::path& result_loss_dir,
    const std::filesystem::path& result_depth_dir,
    const std::filesystem::path& result_normal_dir,
    const std::filesystem::path& result_svraster_normal_dir,
    const std::string&           name_suffix,
    std::optional<float>         global_depth_scale,
    bool                         log_maps_to_rerun)
{
    sv::MiniCam cam = pkf->toMiniCam(pkf->image_height_, pkf->image_width_);

    // Render options: ensure depth is requested
    sv::RenderOpts ropts;
    ropts.output_T     = true;   // if you want T for other losses
    ropts.output_depth = true;   // IMPORTANT for depth saving
    ropts.output_normal = true;  // for normal debug saving

    auto start_timing = std::chrono::steady_clock::now();
    // Render
    auto render_pkg = voxel_model_->render(
        cam,
        pkf->image_height_,
        pkf->image_width_,
        /* gt_image     */ pkf->original_image_,
        /* color_mode   */ nullptr,
        /* track_max_w  */ false,
        /* ss           */ std::nullopt,
        /* output_depth */ ropts.output_depth,
        /* output_normal*/ ropts.output_normal,
        /* output_T     */ ropts.output_T,
        /* rand_bg      */ false,
        /* use_auto_exp */ false,
        ropts
    );
    // auto render_pkg = voxel_model_->render(cam, pkf->image_height_, pkf->image_width_, pkf->original_image_);
    torch::Tensor rendered_image = render_pkg.at("color").to(mDevice);          // (1,3,H,W)
    // Mask and GT on the same device
    torch::Tensor mask = undistort_mask_[pkf->camera_id_]
                            .to(mDevice)
                            .to(torch::kFloat32);                        // (3,H,W) or (1,3,H,W)
    torch::Tensor gt_image = pkf->original_image_.to(mDevice);          // (3,H,W)
    // Broadcast mask over batch if needed
    torch::Tensor masked_image = rendered_image * mask;                 // (1,3,H,W)
    masked_image = masked_image.squeeze(0);                             // (3,H,W)

    torch::cuda::synchronize();
    auto end_timing = std::chrono::steady_clock::now();
    auto render_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_timing - start_timing).count();
    render_ms = 1e-6 * render_time_ns;

    dssim = loss_utils::ssim(masked_image, gt_image, device_type_).item().toFloat();
    psnr = loss_utils::psnr(masked_image, gt_image).item().toFloat();

    const bool log_maps =
        log_maps_to_rerun &&
        rerun_params_.enable_rerun_ &&
        sv::RerunVisualizerBridge::instance().isEnabled();
    cv::Mat maps_gt_rgb;
    cv::Mat maps_rendered_rgb;
    cv::Mat maps_rgb_error;
    cv::Mat maps_gt_depth_rgb;
    cv::Mat maps_rendered_depth_rgb;
    cv::Mat maps_depth_error_rgb;
    cv::Mat maps_depth_gap_rgb;
    cv::Mat maps_gt_normal_rgb;
    cv::Mat maps_rendered_normal_rgb;
    cv::Mat maps_normal_error_rgb;
    double maps_depth_l1_m = -1.0;
    double maps_depth_gap_percent = -1.0;
    double maps_normal_mean_deg = -1.0;
    if (log_maps) {
        maps_gt_rgb = voxel_eval::chwRgbFloatTensorToU8Rgb(gt_image);
        maps_rendered_rgb = voxel_eval::chwRgbFloatTensorToU8Rgb(masked_image);
        const torch::Tensor rgb_err =
            (masked_image - gt_image).abs().mean(0).to(torch::kCPU).contiguous();
        maps_rgb_error = voxel_eval::bgrToRgbImage(voxel_eval::appendColormapLegendBar(
            voxel_eval::colorizeFiniteScalarMat(
                voxel_eval::depthTensorToCvMatFloat(rgb_err),
                0.0f,
                0.25f,
                cv::COLORMAP_VIRIDIS),
            0.0f,
            0.25f,
            " rgb",
            cv::COLORMAP_VIRIDIS,
            "",
            ""));
    }

    recordKeyframeRendered(masked_image, gt_image, pkf->fid_, result_img_dir, result_gt_dir, result_loss_dir, name_suffix);    

    std::ostringstream ss;
    ss << "kf_" << std::setw(5) << std::setfill('0') << pkf->fid_;
    const std::string stem = ss.str();
    cv::Mat gt_depth_meters_eval;
    const bool has_gt_depth_eval =
        voxel_eval::getKeyframeDepthMetersForEval(pkf, pkf->image_height_, pkf->image_width_, gt_depth_meters_eval);
    torch::Tensor eval_mask = mask.detach().to(torch::kCPU).to(torch::kFloat32);
    if (eval_mask.dim() == 4 && eval_mask.size(0) == 1) {
        eval_mask = eval_mask.squeeze(0);
    }
    if (eval_mask.dim() == 3) {
        eval_mask = eval_mask.index({0});
    }
    if (eval_mask.dim() == 2 &&
        eval_mask.size(0) == pkf->image_height_ &&
        eval_mask.size(1) == pkf->image_width_) {
        eval_mask = eval_mask > 0.5f;
    } else {
        eval_mask = torch::ones(
            {pkf->image_height_, pkf->image_width_},
            torch::TensorOptions().dtype(torch::kBool));
    }
    bool have_main_depth_viz_range = false;
    float main_viz_min = 0.0f;
    float main_viz_max = 1.0f;

    // ---- Depth saving / GT-vs-render depth debug ----
    torch::Tensor pred_depth;
    if (voxel_eval::renderPkgToMetricDepthForEval(render_pkg, pred_depth)) {
        pred_depth = pred_depth.to(torch::kCPU).contiguous();
        const int H = static_cast<int>(pred_depth.size(0));
        const int W = static_cast<int>(pred_depth.size(1));

        cv::Mat gt_depth_meters = has_gt_depth_eval ? gt_depth_meters_eval : cv::Mat();
        have_main_depth_viz_range = true;
        main_viz_min = 0.0f;
        main_viz_max = 6.0f;

        std::ostringstream final_depth_tag;
        final_depth_tag << "_iter_" << std::setw(5) << std::setfill('0') << getIteration();
        const std::filesystem::path depth_path =
            result_depth_dir / (stem + final_depth_tag.str() + ".png");
        const std::filesystem::path depth_gt_path = result_depth_dir / (stem + "_gt.png");
        const std::filesystem::path depth_pair_path =
            result_depth_dir / (stem + "_pair" + final_depth_tag.str() + ".png");
        voxel_eval::saveDepthComparisonDebugPngs(
            pred_depth,
            has_gt_depth_eval ? gt_depth_meters : cv::Mat(),
            RGBD_min_depth_,
            RGBD_max_depth_,
            depth_path,
            depth_gt_path,
            depth_pair_path,
            sensor_type_ == MONOCULAR ? global_depth_scale : std::nullopt);

        torch::Tensor pred_depth_for_eval = pred_depth;
        if (sensor_type_ == MONOCULAR &&
            global_depth_scale.has_value() &&
            std::isfinite(*global_depth_scale) &&
            *global_depth_scale > 0.0f) {
            pred_depth_for_eval = pred_depth_for_eval * (*global_depth_scale);
        }

        if (log_maps) {
            const cv::Mat pred_depth_viz_mat =
                voxel_eval::depthTensorToCvMatFloat(pred_depth_for_eval);
            if (have_main_depth_viz_range) {
                maps_rendered_depth_rgb = voxel_eval::bgrToRgbImage(voxel_eval::colorizeDepthMatJet(
                    pred_depth_viz_mat,
                    RGBD_min_depth_,
                    std::min(RGBD_max_depth_, main_viz_max),
                    main_viz_min,
                    main_viz_max));
                if (has_gt_depth_eval) {
                    maps_gt_depth_rgb = voxel_eval::bgrToRgbImage(voxel_eval::colorizeDepthMatJet(
                        gt_depth_meters,
                        RGBD_min_depth_,
                        std::min(RGBD_max_depth_, main_viz_max),
                        main_viz_min,
                        main_viz_max));
                }
            }

            if (has_gt_depth_eval) {
                torch::Tensor gt_depth = torch::from_blob(
                    gt_depth_meters.data,
                    {H, W},
                    torch::TensorOptions().dtype(torch::kFloat32)).clone();
                const torch::Tensor valid_pred =
                    torch::isfinite(pred_depth_for_eval) &
                    (pred_depth_for_eval > RGBD_min_depth_) &
                    (pred_depth_for_eval < RGBD_max_depth_) &
                    eval_mask;
                const torch::Tensor valid_gt =
                    torch::isfinite(gt_depth) &
                    (gt_depth > RGBD_min_depth_) &
                    (gt_depth < RGBD_max_depth_) &
                    eval_mask;
                const torch::Tensor valid_both = valid_pred & valid_gt;
                const int64_t both_count = valid_both.sum().item<int64_t>();
                if (both_count > 0) {
                    const torch::Tensor abs_err = (pred_depth_for_eval - gt_depth).abs();
                    maps_depth_l1_m =
                        abs_err.masked_select(valid_both).mean().item<float>();
                    maps_depth_error_rgb = voxel_eval::bgrToRgbImage(voxel_eval::appendColormapLegendBar(
                        voxel_eval::colorizeFiniteScalarMat(
                            voxel_eval::depthTensorToCvMatFloat(torch::where(
                                valid_both,
                                abs_err,
                                torch::full_like(abs_err, std::numeric_limits<float>::quiet_NaN()))),
                            0.0f,
                            0.25f,
                            cv::COLORMAP_VIRIDIS),
                        0.0f,
                        0.25f,
                        " m",
                        cv::COLORMAP_VIRIDIS,
                        "",
                        ""));
                }
                maps_depth_gap_rgb = voxel_eval::makeDepthGapMaskRgb(
                    pred_depth_viz_mat,
                    gt_depth_meters,
                    eval_mask,
                    RGBD_min_depth_,
                    RGBD_max_depth_,
                    maps_depth_gap_percent);
            }
        }

    }

    // ---- Normal saving / GT-from-depth-vs-render normal debug ----
    torch::Tensor pred_normal;
    if (voxel_eval::renderPkgToNormalForEval(render_pkg, pred_normal)) {
        const int Hn = static_cast<int>(pred_normal.size(1));
        const int Wn = static_cast<int>(pred_normal.size(2));
        torch::Tensor eval_mask_normal = eval_mask;
        if (!(eval_mask_normal.dim() == 2 &&
              eval_mask_normal.size(0) == Hn &&
              eval_mask_normal.size(1) == Wn)) {
            eval_mask_normal = torch::ones(
                {Hn, Wn},
                torch::TensorOptions().dtype(torch::kBool));
        }

        const torch::Tensor pred_normal_mag =
            pred_normal.square().sum(0).sqrt();
        const torch::Tensor pred_valid =
            torch::isfinite(pred_normal).all(0) &
            (pred_normal_mag > 1e-6f) &
            eval_mask_normal;
        torch::Tensor pred_normal_unit = torch::nn::functional::normalize(
            pred_normal,
            torch::nn::functional::NormalizeFuncOptions().dim(0).eps(1e-12));
        pred_normal_unit = torch::where(
            pred_valid.unsqueeze(0),
            pred_normal_unit,
            torch::zeros_like(pred_normal_unit));

        const torch::Tensor pred_svraster_normal_viz =
            -voxel_eval::normalWorldToCameraForViz(cam, pred_normal_unit);
        const cv::Mat pred_svraster_normal_bgr =
            voxel_eval::colorizeNormalMapBgr(pred_svraster_normal_viz);
        const std::filesystem::path svraster_normal_path =
            result_svraster_normal_dir / (stem + ".png");
        std::filesystem::create_directories(svraster_normal_path.parent_path());
        if (!pred_svraster_normal_bgr.empty()) {
            cv::imwrite(svraster_normal_path.string(), pred_svraster_normal_bgr);
        }

        torch::Tensor rendered_depth_normal_unit;
        torch::Tensor rendered_depth_normal_valid;
        cv::Mat rendered_depth_normal_bgr;
        {
            torch::Tensor rendered_depth_for_normal;
            if (voxel_eval::renderPkgToMetricDepthForEval(render_pkg, rendered_depth_for_normal)) {
                rendered_depth_for_normal =
                    rendered_depth_for_normal.to(torch::kCPU).to(torch::kFloat32).contiguous();
                if (sensor_type_ == MONOCULAR &&
                    global_depth_scale.has_value() &&
                    std::isfinite(*global_depth_scale) &&
                    *global_depth_scale > 0.0f) {
                    rendered_depth_for_normal = rendered_depth_for_normal * (*global_depth_scale);
                }
                if (rendered_depth_for_normal.size(0) != Hn ||
                    rendered_depth_for_normal.size(1) != Wn) {
                    rendered_depth_for_normal = torch::nn::functional::interpolate(
                        rendered_depth_for_normal.unsqueeze(0).unsqueeze(0),
                        torch::nn::functional::InterpolateFuncOptions()
                            .size(std::vector<int64_t>{Hn, Wn})
                            .mode(torch::kNearest)).squeeze().contiguous();
                }

                sv::MiniCam cam_cpu = cam;
                cam_cpu.c2w = cam.c2w.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
                torch::Tensor rendered_depth_normal = voxel_eval::depth2normalSVRaster(
                    cam_cpu,
                    rendered_depth_for_normal,
                    /*ks=*/3,
                    /*tol_cos=*/0.0f).to(torch::kCPU).to(torch::kFloat32).contiguous();

                const torch::Tensor rendered_depth_valid =
                    torch::isfinite(rendered_depth_for_normal) &
                    (rendered_depth_for_normal > RGBD_min_depth_) &
                    (rendered_depth_for_normal < RGBD_max_depth_) &
                    eval_mask_normal;
                const torch::Tensor rendered_depth_normal_mag =
                    rendered_depth_normal.square().sum(0).sqrt();
                rendered_depth_normal_valid =
                    torch::isfinite(rendered_depth_normal).all(0) &
                    (rendered_depth_normal_mag > 1e-6f) &
                    rendered_depth_valid;
                rendered_depth_normal_unit = torch::nn::functional::normalize(
                    rendered_depth_normal,
                    torch::nn::functional::NormalizeFuncOptions().dim(0).eps(1e-12));
                rendered_depth_normal_unit = torch::where(
                    rendered_depth_normal_valid.unsqueeze(0),
                    rendered_depth_normal_unit,
                    torch::zeros_like(rendered_depth_normal_unit));

                const torch::Tensor rendered_depth_normal_viz =
                    -voxel_eval::normalWorldToCameraForViz(cam, rendered_depth_normal_unit);
                rendered_depth_normal_bgr = voxel_eval::colorizeNormalMapBgr(rendered_depth_normal_viz);
                if (log_maps && !rendered_depth_normal_bgr.empty()) {
                    maps_rendered_normal_rgb = voxel_eval::bgrToRgbImage(rendered_depth_normal_bgr);
                }
                const std::filesystem::path normal_path = result_normal_dir / (stem + ".png");
                std::filesystem::create_directories(normal_path.parent_path());
                if (!rendered_depth_normal_bgr.empty()) {
                    cv::imwrite(normal_path.string(), rendered_depth_normal_bgr);
                }
            }
        }

        if (has_gt_depth_eval) {
            cv::Mat gt_depth_for_normal = gt_depth_meters_eval;
            if (gt_depth_for_normal.rows != Hn || gt_depth_for_normal.cols != Wn) {
                cv::resize(
                    gt_depth_for_normal,
                    gt_depth_for_normal,
                    cv::Size(Wn, Hn),
                    0.0,
                    0.0,
                    cv::INTER_NEAREST);
            }

            torch::Tensor gt_depth = torch::from_blob(
                gt_depth_for_normal.data,
                {Hn, Wn},
                torch::TensorOptions().dtype(torch::kFloat32)).clone();
            sv::MiniCam cam_cpu = cam;
            cam_cpu.c2w = cam.c2w.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
            torch::Tensor gt_normal = voxel_eval::depth2normalSVRaster(
                cam_cpu,
                gt_depth,
                /*ks=*/3,
                /*tol_cos=*/0.0f).to(torch::kCPU).to(torch::kFloat32).contiguous();

            const torch::Tensor gt_depth_valid =
                torch::isfinite(gt_depth) &
                (gt_depth > RGBD_min_depth_) &
                (gt_depth < RGBD_max_depth_) &
                eval_mask_normal;
            const torch::Tensor gt_normal_mag = gt_normal.square().sum(0).sqrt();
            const torch::Tensor gt_valid =
                torch::isfinite(gt_normal).all(0) &
                (gt_normal_mag > 1e-6f) &
                gt_depth_valid;
            torch::Tensor gt_normal_unit = torch::nn::functional::normalize(
                gt_normal,
                torch::nn::functional::NormalizeFuncOptions().dim(0).eps(1e-12));
            gt_normal_unit = torch::where(
                gt_valid.unsqueeze(0),
                gt_normal_unit,
                torch::zeros_like(gt_normal_unit));

            const torch::Tensor gt_normal_viz =
                -voxel_eval::normalWorldToCameraForViz(cam, gt_normal_unit);
            const cv::Mat gt_normal_bgr = voxel_eval::colorizeNormalMapBgr(gt_normal_viz);
            if (log_maps && !gt_normal_bgr.empty()) {
                maps_gt_normal_rgb = voxel_eval::bgrToRgbImage(gt_normal_bgr);
            }
            const std::filesystem::path normal_gt_path =
                result_normal_dir / (stem + "_gt_from_depth.png");
            if (!gt_normal_bgr.empty()) {
                cv::imwrite(normal_gt_path.string(), gt_normal_bgr);
            }

            if (!rendered_depth_normal_bgr.empty() && !gt_normal_bgr.empty()) {
                cv::Mat normal_pair_bgr;
                cv::hconcat(
                    std::vector<cv::Mat>{gt_normal_bgr, rendered_depth_normal_bgr},
                    normal_pair_bgr);
                const std::filesystem::path normal_pair_path =
                    result_normal_dir / (stem + "_pair.png");
                cv::imwrite(normal_pair_path.string(), normal_pair_bgr);
            }

            constexpr float kRadToDeg = 57.29577951308232f;
            torch::Tensor err_deg;
            torch::Tensor valid_both;
            if (rendered_depth_normal_unit.defined() &&
                rendered_depth_normal_valid.defined()) {
                valid_both = rendered_depth_normal_valid & gt_valid;
                const torch::Tensor dot =
                    (rendered_depth_normal_unit * gt_normal_unit).sum(0).clamp(-1.0f, 1.0f);
                err_deg = torch::where(
                    valid_both,
                    torch::acos(dot) * kRadToDeg,
                    torch::full_like(dot, std::numeric_limits<float>::quiet_NaN()));
            }
            if (log_maps) {
                const int64_t normal_count =
                    valid_both.defined() ? valid_both.sum().item<int64_t>() : 0;
                if (normal_count > 0 && err_deg.defined()) {
                    maps_normal_mean_deg =
                        err_deg.masked_select(valid_both).mean().item<float>();
                }
                if (err_deg.defined()) {
                    maps_normal_error_rgb = voxel_eval::bgrToRgbImage(voxel_eval::appendColormapLegendBar(
                        voxel_eval::colorizeFiniteScalarMat(
                            voxel_eval::depthTensorToCvMatFloat(err_deg),
                            0.0f,
                            45.0f,
                            cv::COLORMAP_VIRIDIS),
                        0.0f,
                        45.0f,
                        " deg",
                        cv::COLORMAP_VIRIDIS,
                        "",
                        ""));
                }
            }
            if (err_deg.defined()) {
                const cv::Mat err_bgr = voxel_eval::appendJetLegendBar(
                    voxel_eval::colorizeFiniteScalarMatJet(
                        voxel_eval::depthTensorToCvMatFloat(err_deg),
                        0.0f,
                        45.0f),
                    0.0f,
                    45.0f,
                    " deg");
                const std::filesystem::path normal_err_path =
                    result_normal_dir / (stem + "_angular_err.png");
                cv::imwrite(normal_err_path.string(), err_bgr);
            }
        }
    }

    constexpr bool save_shutdown_mono_prior_depth_debug = true;
    if (save_shutdown_mono_prior_depth_debug && sensor_type_ == MONOCULAR) {
        const int mono_prior_apply_iter = pkf->mono_prior_first_apply_iter_;
        const bool mono_prior_was_applied = mono_prior_apply_iter >= 0;
        const int mono_prior_depth_loss_iter = pkf->mono_prior_first_depth_loss_iter_;
        const bool mono_prior_was_optimized = mono_prior_depth_loss_iter >= 0;

        if ((mono_prior_was_applied || mono_prior_was_optimized) &&
            ((opt_params_.lambda_depthanythingv2_ > 0.0f) ||
             (pkf->mono_prior_.defined() && pkf->mono_prior_.numel() > 0))) {
            if (ensureMonoPriorForKeyframe(pkf)) {
                std::ostringstream mono_prior_apply_tag;
                mono_prior_apply_tag << "_iter_"
                                     << std::setw(5) << std::setfill('0')
                                     << mono_prior_apply_iter;
                std::ostringstream mono_prior_loss_tag;
                mono_prior_loss_tag << "_iter_"
                                    << std::setw(5) << std::setfill('0')
                                    << mono_prior_depth_loss_iter;
                std::ostringstream shutdown_iter_tag;
                shutdown_iter_tag << "_iter_"
                                  << std::setw(5) << std::setfill('0')
                                  << getIteration();
                const std::string mono_prior_raw_eval_stem =
                    stem + "_mono_prior_raw_densification" + mono_prior_apply_tag.str();
                const std::string mono_prior_aligned_eval_stem =
                    stem + "_mono_prior_aligned_densification" + mono_prior_apply_tag.str();
                const std::string mono_prior_target_eval_stem =
                    stem + "_mono_prior_target" + mono_prior_loss_tag.str();

                torch::Tensor mono_prior_viz =
                    pkf->mono_prior_.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
                if (mono_prior_viz.dim() == 3 && mono_prior_viz.size(0) == 1) {
                    mono_prior_viz = mono_prior_viz.squeeze(0);
                }

                torch::Tensor mono_target_viz;
                torch::Tensor aligned_mono_prior_eval_viz;
                bool have_target_debug_map = false;
                const bool apply_mono_prior_eval_scale =
                    sensor_type_ == MONOCULAR &&
                    global_depth_scale.has_value() &&
                    std::isfinite(*global_depth_scale) &&
                    *global_depth_scale > 0.0f;
                const bool have_aligned_eval_map =
                    buildAlignedMonoPriorDepthForKeyframe(
                        pkf,
                        cam,
                        pkf->image_width_,
                        pkf->image_height_,
                        aligned_mono_prior_eval_viz);
                if (have_aligned_eval_map) {
                    aligned_mono_prior_eval_viz =
                        aligned_mono_prior_eval_viz.to(torch::kCPU).to(torch::kFloat32).contiguous();
                    if (apply_mono_prior_eval_scale) {
                        aligned_mono_prior_eval_viz =
                            aligned_mono_prior_eval_viz * (*global_depth_scale);
                    }
                }
                if (mono_prior_was_optimized) {
                    if (mono_prior_params_.mono_prior_loss_mode_ == "aligned" ||
                        monoPriorUsesMetricDepth()) {
                        if (have_aligned_eval_map) {
                            mono_target_viz = aligned_mono_prior_eval_viz;
                            have_target_debug_map = true;
                        }
                    } else {
                        torch::Tensor unused_mono_prior_resized;
                        torch::Tensor unused_render_mono_prior_viz;
                        have_target_debug_map = voxel_eval::renderPkgToMonoPriorDebugMaps(
                            render_pkg,
                            pkf->mono_prior_,
                            cam.near,
                            unused_mono_prior_resized,
                            mono_target_viz,
                            unused_render_mono_prior_viz);
                    }
                }
                constexpr float kMonoPriorValidMin = 1e-6f;
                constexpr float kMonoPriorValidMax = 1e6f;
                if (mono_prior_was_applied) {
                    torch::Tensor mono_prior_raw_depth_viz = mono_prior_viz;
                    if (!monoPriorUsesMetricDepth()) {
                        const torch::Tensor raw_valid =
                            torch::isfinite(mono_prior_viz) & (mono_prior_viz > 1e-6f);
                        mono_prior_raw_depth_viz = torch::where(
                            raw_valid,
                            1.0f / mono_prior_viz.clamp_min(1e-6f),
                            torch::full_like(
                                mono_prior_viz,
                                std::numeric_limits<float>::quiet_NaN()));
                    }

                    float mono_prior_viz_min = 0.0f;
                    float mono_prior_viz_max = 1.0f;
                    if (voxel_eval::computeSharedDepthVizRange(
                            mono_prior_raw_depth_viz,
                            cv::Mat(),
                            kMonoPriorValidMin,
                            kMonoPriorValidMax,
                            mono_prior_viz_min,
                            mono_prior_viz_max)) {
                        const cv::Mat mono_prior_bgr = voxel_eval::colorizeDepthMatJet(
                            voxel_eval::depthTensorToCvMatFloat(mono_prior_raw_depth_viz),
                            kMonoPriorValidMin,
                            kMonoPriorValidMax,
                            mono_prior_viz_min,
                            mono_prior_viz_max);
                        const std::filesystem::path mono_prior_raw_eval_path =
                            result_depth_dir / (mono_prior_raw_eval_stem + ".png");
                        std::filesystem::create_directories(mono_prior_raw_eval_path.parent_path());
                        cv::imwrite(mono_prior_raw_eval_path.string(), mono_prior_bgr);
                    }

                    if (have_aligned_eval_map) {
                        float mono_aligned_viz_min = 0.0f;
                        float mono_aligned_viz_max = 1.0f;
                        if (voxel_eval::computeSharedDepthVizRange(
                                aligned_mono_prior_eval_viz,
                                cv::Mat(),
                                RGBD_min_depth_,
                                RGBD_max_depth_,
                                mono_aligned_viz_min,
                                mono_aligned_viz_max)) {
                            const cv::Mat mono_aligned_bgr = voxel_eval::colorizeDepthMatJet(
                                voxel_eval::depthTensorToCvMatFloat(aligned_mono_prior_eval_viz),
                                RGBD_min_depth_,
                                RGBD_max_depth_,
                                mono_aligned_viz_min,
                                mono_aligned_viz_max);
                            const std::filesystem::path mono_aligned_eval_path =
                                result_depth_dir / (mono_prior_aligned_eval_stem + ".png");
                            std::filesystem::create_directories(mono_aligned_eval_path.parent_path());
                            cv::imwrite(mono_aligned_eval_path.string(), mono_aligned_bgr);
                        }
                    }
                }

                if (mono_prior_was_optimized && have_target_debug_map) {
                    const cv::Mat mono_target_mat = voxel_eval::depthTensorToCvMatFloat(mono_target_viz);
                    float mono_target_viz_min = 0.0f;
                    float mono_target_viz_max = 1.0f;
                    if (voxel_eval::computeSharedDepthVizRange(
                            mono_target_viz,
                            cv::Mat(),
                            kMonoPriorValidMin,
                            kMonoPriorValidMax,
                            mono_target_viz_min,
                            mono_target_viz_max)) {
                        const cv::Mat mono_target_bgr = voxel_eval::colorizeDepthMatJet(
                            mono_target_mat,
                            kMonoPriorValidMin,
                            kMonoPriorValidMax,
                            mono_target_viz_min,
                            mono_target_viz_max);

                        const std::filesystem::path mono_target_path =
                            result_depth_dir / (mono_prior_target_eval_stem + ".png");
                        std::filesystem::create_directories(mono_target_path.parent_path());
                        cv::imwrite(mono_target_path.string(), mono_target_bgr);
                    }
                }
            }
        }

    }

    if (log_maps) {
        const int map_h = pkf->image_height_;
        const int map_w = pkf->image_width_;
        auto ensure_image = [&](cv::Mat& img) {
            if (img.empty()) {
                img = voxel_eval::blackRgbImage(map_h, map_w);
            }
        };
        ensure_image(maps_gt_rgb);
        ensure_image(maps_rendered_rgb);
        ensure_image(maps_rgb_error);
        ensure_image(maps_gt_depth_rgb);
        ensure_image(maps_rendered_depth_rgb);
        ensure_image(maps_depth_error_rgb);
        ensure_image(maps_depth_gap_rgb);
        ensure_image(maps_gt_normal_rgb);
        ensure_image(maps_rendered_normal_rgb);
        ensure_image(maps_normal_error_rgb);

        sv::RerunVisualizerBridge::instance().visualizeMapsFrameRecording(
            "maps",
            static_cast<int>(pkf->fid_),
            getIteration(),
            maps_gt_rgb,
            maps_rendered_rgb,
            maps_rgb_error,
            maps_gt_depth_rgb,
            maps_rendered_depth_rgb,
            maps_depth_error_rgb,
            maps_depth_gap_rgb,
            maps_gt_normal_rgb,
            maps_rendered_normal_rgb,
            maps_normal_error_rgb,
            static_cast<double>(psnr),
            static_cast<double>(dssim),
            maps_depth_l1_m,
            maps_depth_gap_percent,
            maps_normal_mean_deg);
    }

 }

void VoxelMapper::renderAndRecordAllKeyframes(const std::string& name_suffix)
{
    // Create result directory with current iteration number and suffix
    std::filesystem::path result_dir = result_dir_ / (std::to_string(getIteration()) + name_suffix);
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir);
    if (scene_) {
        voxel_utils::saveKeyframeFrameIdMap(scene_->keyframes(), result_dir / "kf_frame_id_map.txt");
    }

    // Create subdirectories if needed
    std::filesystem::path image_dir = result_dir / "image";
    if (record_rendered_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_dir);

    std::filesystem::path image_gt_dir = result_dir / "image_gt";
    if (record_ground_truth_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_gt_dir);

    std::filesystem::path image_loss_dir = result_dir / "image_loss";
    if (record_loss_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_loss_dir);

    // New: depth directory inside the same x_shutdown folder
    std::filesystem::path depth_dir = result_dir / "depth";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(depth_dir);
    voxel_eval::copyPngFilesToDirectory(voxel_eval::runtimeOrbDepthDebugDir(result_dir_), depth_dir);

    std::filesystem::path normal_dir = result_dir / "normal";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(normal_dir);

    std::filesystem::path svraster_normal_dir = result_dir / "normals_svraster";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(svraster_normal_dir);

    // Open logging files
    std::filesystem::path render_time_path = result_dir / "render_time.txt";
    std::ofstream out_time(render_time_path);
    out_time << "##[Voxel Mapper]Render time statistics: keyframe id, time(milliseconds)\n";

    std::filesystem::path dssim_path = result_dir / "dssim.txt";
    std::ofstream out_dssim(dssim_path);
    out_dssim << "##[Voxel Mapper]keyframe id, dssim\n";

    std::filesystem::path psnr_path = result_dir / "psnr.txt";
    std::ofstream out_psnr(psnr_path);
    out_psnr << "##[Voxel Mapper]keyframe id, psnr\n";

    const std::size_t nkfs = scene_->keyframes().size();
    std::optional<float> global_depth_scale = std::nullopt;
    if (sensor_type_ == MONOCULAR) {
        std::filesystem::path depth_scale_debug_path = result_dir / "depth_scale_debug.txt";
        std::ofstream out_depth_scale(depth_scale_debug_path);
        out_depth_scale << "##[Voxel Mapper]Monocular depth debug scale diagnostics\n";
        out_depth_scale << "# columns: kfid has_gt selected_ch selected_scale selected_overlap selected_ratio_after_trim "
                           "selected_q25 selected_q50 selected_q75 selected_pred_min selected_pred_max "
                           "ch0_scale ch0_overlap ch0_ratio_after_trim ch2_scale ch2_overlap ch2_ratio_after_trim\n";

        auto fmt_stat = [](float v) -> std::string {
            if (!std::isfinite(v)) {
                return "nan";
            }
            std::ostringstream os;
            os << std::fixed << std::setprecision(6) << v;
            return os.str();
        };

        auto render_pkg_for_depth_debug =
            [this](const std::shared_ptr<VoxelKeyframe>& pkf)
            -> std::unordered_map<std::string, torch::Tensor>
        {
            sv::MiniCam cam = pkf->toMiniCam(pkf->image_height_, pkf->image_width_);
            sv::RenderOpts ropts;
            ropts.output_T = true;
            ropts.output_depth = true;
            return voxel_model_->render(
                cam,
                pkf->image_height_,
                pkf->image_width_,
                /* gt_image     */ pkf->original_image_,
                /* color_mode   */ nullptr,
                /* track_max_w  */ false,
                /* ss           */ std::nullopt,
                /* output_depth */ ropts.output_depth,
                /* output_normal*/ false,
                /* output_T     */ ropts.output_T,
                /* rand_bg      */ false,
                /* use_auto_exp */ false,
                ropts);
        };

        std::vector<std::pair<float, double>> selected_scale_samples;
        std::vector<std::pair<float, double>> ch0_scale_samples;
        std::vector<std::pair<float, double>> ch2_scale_samples;
        std::size_t selected_valid_kfs = 0;
        std::size_t ch0_valid_kfs = 0;
        std::size_t ch2_valid_kfs = 0;

        auto kfit_debug = scene_->keyframes().begin();
        for (std::size_t i = 0; i < nkfs; ++i, ++kfit_debug) {
            const auto& pkf = (*kfit_debug).second;
            const auto render_pkg = render_pkg_for_depth_debug(pkf);

            torch::Tensor depth_tensor;
            auto it_depth = render_pkg.find("depth");
            if (it_depth == render_pkg.end() || !it_depth->second.defined()) {
                it_depth = render_pkg.find("raw_depth");
            }
            if (it_depth != render_pkg.end() && it_depth->second.defined()) {
                depth_tensor = it_depth->second;
            }

            torch::Tensor pred_selected;
            const bool has_pred_selected =
                voxel_eval::renderPkgToMetricDepthForEval(render_pkg, pred_selected);

            cv::Mat gt_depth_meters;
            bool has_gt_depth = false;
            int selected_channel = -1;
            voxel_eval::DepthScaleFitStats selected_stats;
            voxel_eval::DepthScaleFitStats ch0_stats;
            voxel_eval::DepthScaleFitStats ch2_stats;

            if (has_pred_selected) {
                pred_selected = pred_selected.to(torch::kCPU).contiguous();
                has_gt_depth = voxel_eval::getKeyframeDepthMetersForEval(
                    pkf,
                    static_cast<int>(pred_selected.size(0)),
                    static_cast<int>(pred_selected.size(1)),
                    gt_depth_meters);

                if (depth_tensor.defined()) {
                    torch::Tensor d = depth_tensor.detach();
                    if (d.dim() == 4 && d.size(0) == 1) {
                        d = d.squeeze(0);
                    }
                    if (d.dim() == 3) {
                        selected_channel = (d.size(0) > 2) ? 2 : 0;
                    } else if (d.dim() == 2) {
                        selected_channel = 0;
                    }
                }

                if (has_gt_depth) {
                    voxel_eval::computeDepthScaleFitStats(
                        pred_selected,
                        gt_depth_meters,
                        RGBD_min_depth_,
                        RGBD_max_depth_,
                        selected_stats);
                    if (selected_stats.valid) {
                        selected_scale_samples.emplace_back(
                            selected_stats.scale,
                            static_cast<double>(std::max<int64_t>(selected_stats.ratio_count_after_trim, 1)));
                        ++selected_valid_kfs;
                    }
                }
            }

            if (depth_tensor.defined() && has_gt_depth) {
                torch::Tensor pred_ch0 = voxel_eval::tensorToEvalMapExactChannel(depth_tensor, 0);
                if (pred_ch0.defined()) {
                    pred_ch0 = pred_ch0.to(torch::kCPU).contiguous();
                    voxel_eval::computeDepthScaleFitStats(
                        pred_ch0,
                        gt_depth_meters,
                        RGBD_min_depth_,
                        RGBD_max_depth_,
                        ch0_stats);
                    if (ch0_stats.valid) {
                        ch0_scale_samples.emplace_back(
                            ch0_stats.scale,
                            static_cast<double>(std::max<int64_t>(ch0_stats.ratio_count_after_trim, 1)));
                        ++ch0_valid_kfs;
                    }
                }

                torch::Tensor pred_ch2 = voxel_eval::tensorToEvalMapExactChannel(depth_tensor, 2);
                if (pred_ch2.defined()) {
                    pred_ch2 = pred_ch2.to(torch::kCPU).contiguous();
                    voxel_eval::computeDepthScaleFitStats(
                        pred_ch2,
                        gt_depth_meters,
                        RGBD_min_depth_,
                        RGBD_max_depth_,
                        ch2_stats);
                    if (ch2_stats.valid) {
                        ch2_scale_samples.emplace_back(
                            ch2_stats.scale,
                            static_cast<double>(std::max<int64_t>(ch2_stats.ratio_count_after_trim, 1)));
                        ++ch2_valid_kfs;
                    }
                }
            }

            out_depth_scale
                << (*kfit_debug).first << " "
                << (has_gt_depth ? 1 : 0) << " "
                << selected_channel << " "
                << fmt_stat(selected_stats.scale) << " "
                << selected_stats.overlap_count << " "
                << selected_stats.ratio_count_after_trim << " "
                << fmt_stat(selected_stats.ratio_q25) << " "
                << fmt_stat(selected_stats.ratio_q50) << " "
                << fmt_stat(selected_stats.ratio_q75) << " "
                << fmt_stat(selected_stats.pred_min) << " "
                << fmt_stat(selected_stats.pred_max) << " "
                << fmt_stat(ch0_stats.scale) << " "
                << ch0_stats.overlap_count << " "
                << ch0_stats.ratio_count_after_trim << " "
                << fmt_stat(ch2_stats.scale) << " "
                << ch2_stats.overlap_count << " "
                << ch2_stats.ratio_count_after_trim
                << "\n";
        }

        float selected_scale_value = 1.0f;
        float ch0_scale_value = 1.0f;
        float ch2_scale_value = 1.0f;
        const bool have_selected_scale =
            voxel_eval::computeWeightedMedianScale(selected_scale_samples, selected_scale_value);
        const bool have_ch0_scale =
            voxel_eval::computeWeightedMedianScale(ch0_scale_samples, ch0_scale_value);
        const bool have_ch2_scale =
            voxel_eval::computeWeightedMedianScale(ch2_scale_samples, ch2_scale_value);

        if (have_selected_scale) {
            global_depth_scale = selected_scale_value;
        }

        out_depth_scale << "# summary selected_valid_kfs " << selected_valid_kfs << "\n";
        out_depth_scale << "# summary selected_global_scale "
                        << (have_selected_scale ? fmt_stat(selected_scale_value) : std::string("nan")) << "\n";
        out_depth_scale << "# summary ch0_valid_kfs " << ch0_valid_kfs << "\n";
        out_depth_scale << "# summary ch0_global_scale "
                        << (have_ch0_scale ? fmt_stat(ch0_scale_value) : std::string("nan")) << "\n";
        out_depth_scale << "# summary ch2_valid_kfs " << ch2_valid_kfs << "\n";
        out_depth_scale << "# summary ch2_global_scale "
                        << (have_ch2_scale ? fmt_stat(ch2_scale_value) : std::string("nan")) << "\n";

        if (have_selected_scale) {
            std::cout << "[DepthDebug] monocular global depth scale="
                      << std::fixed << std::setprecision(6) << selected_scale_value
                      << " valid_kfs=" << selected_valid_kfs
                      << " ch0_global=" << (have_ch0_scale ? fmt_stat(ch0_scale_value) : std::string("nan"))
                      << " ch2_global=" << (have_ch2_scale ? fmt_stat(ch2_scale_value) : std::string("nan"))
                      << "\n";
        } else {
            std::cout << "[DepthDebug] could not estimate a robust monocular global depth scale.\n";
        }
    }

    // Loop through all keyframes deterministically
    auto kfit = scene_->keyframes().begin();
    float dssim, psnr;
    double render_time;
    for (std::size_t i = 0; i < nkfs; ++i) {
        const bool log_maps_for_keyframe =
            rerun_params_.enable_rerun_ &&
            rerun_params_.rerun_maps_ &&
            rerun_params_.rerun_maps_stride_ > 0 &&
            (static_cast<int>(i) % rerun_params_.rerun_maps_stride_) == 0;
        renderAndRecordKeyframe(
            (*kfit).second,
            dssim,
            psnr,
            render_time,
            image_dir,
            image_gt_dir,
            image_loss_dir,
            depth_dir,
            normal_dir,
            svraster_normal_dir,
            name_suffix,
            global_depth_scale,
            log_maps_for_keyframe);
        out_time << (*kfit).first << " " << std::fixed << std::setprecision(8) << render_time << std::endl;

        out_dssim   << (*kfit).first << " " << std::fixed << std::setprecision(10) << dssim   << std::endl;
        out_psnr    << (*kfit).first << " " << std::fixed << std::setprecision(10) << psnr    << std::endl;

        ++kfit;
    }
}

void VoxelMapper::savePly(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    // keyframesToJson(result_dir);
    // saveModelParams(result_dir);

    std::filesystem::path ply_dir = result_dir / "voxel_model";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

    ply_dir = ply_dir / ("iteration_" + std::to_string(getIteration()));
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

    // Reconstructed voxel scene
    voxel_model_->savePly(ply_dir / "voxel_model.ply");
    // Input sparse points (from ORB-SLAM map) for reference
}

void VoxelMapper::keyframesToJson(const std::filesystem::path&){ }

/* ---------------- runtime getter / setter ---------------- */
// VariableParameters VoxelMapper::getVariableParameters() const
// {
//     std::lock_guard<std::mutex> lk(mutex_status_);
//     VariableParameters p;
//     p.position_lr_init          = position_lr_init_;
//     p.new_keyframe_times_of_use_ = var_params_.new_keyframe_times_of_use_;
//     p.do_inactive_geo_densify   = do_inactive_geo_densify_;
//     p.keep_training = keep_training_;
//     return p;
// }

// void VoxelMapper::setVariableParameters(const VariableParameters& p)
// {
//     std::lock_guard<std::mutex> lk(mutex_status_);
//     /* apply only what VoxelMapper still honours */
//     position_lr_init_                               = p.position_lr_init;
//     new_keyframe_times_of_use_ = p.new_keyframe_times_of_use_;
//     do_inactive_geo_densify_               = p.do_inactive_geo_densify;
//     keep_training_                         = p.keep_training;
// }

cv::Mat VoxelMapper::renderFromPose(
    const Sophus::SE3f &Tcw,
    const int width,
    const int height,
    const bool main_vision)
{
    // Same guard as Photo-SLAM: no rendering before we have something
    if (!initial_mapped_ || getIteration() <= 0) {
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    }

    // Build a temporary keyframe for the viewer pose
    std::shared_ptr<VoxelKeyframe> pkf = std::make_shared<VoxelKeyframe>();
    // pkf->zfar_ = z_far_;   // only if you actually use z_far_ anywhere
    pkf->znear_ = z_near_;

    // Pose
    pkf->setPose(
        Tcw.unit_quaternion().cast<double>(),
        Tcw.translation().cast<double>());

    try {
        // Camera
        sv::Camera& camera = scene_->cameras_.at(viewer_camera_id_);
        pkf->setCameraParams(camera);
        // If your VoxelKeyframe has this (like GaussianKeyframe), call it:
        // pkf->computeTransformTensors();
    }
    catch (const std::out_of_range&) {
        throw std::runtime_error("[VoxelMapper::renderFromPose] KeyFrame Camera not found!");
    }

    // Build MiniCam for the viewer resolution
    sv::MiniCam cam = pkf->toMiniCam(height, width);

    // We don't want gradients in the viewer
    torch::NoGradGuard no_grad;

    // Call voxel_model_->render under the same render mutex
    std::unordered_map<std::string, torch::Tensor> pkg;
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);

        pkg = voxel_model_->render(
            cam,
            height,
            width,
            /* gt_image      */ torch::Tensor(),  // none
            /* color_mode    */ nullptr,
            /* track_max_w   */ false,
            /* ss            */ std::nullopt,
            /* output_depth  */ false,
            /* output_normal */ false,
            /* output_T      */ false,
            /* rand_bg       */ false,
            /* use_auto_exp  */ false,
            sv::RenderOpts{}   // default options
        );
    }

    // Check we actually got a color image
    auto it = pkg.find("color");
    if (it == pkg.end() || !it->second.defined()) {
        // Fallback: black image
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    }

    torch::Tensor color = it->second;  // expected shape [1,3,H,W] or [3,H,W]

    // Masking exactly like GaussianMapper
    torch::Tensor mask;
    if (main_vision) {
        mask = viewer_main_undistort_mask_[pkf->camera_id_];
    } else {
        mask = viewer_sub_undistort_mask_[pkf->camera_id_];
    }

    // Make sure mask is on the same device as color
    if (mask.device() != color.device()) {
        mask = mask.to(color.device());
    }

    // Both should be broadcastable: mask is usually [1,3,H,W] or [3,H,W]
    torch::Tensor masked_image = color * mask;

    // Reuse Photo-SLAM utility to convert to cv::Mat (float32 RGB)
    return tensor_utils::torchTensor2CvMat_Float32(masked_image);
}

// VoxelMapper::~VoxelMapper() {
//     // Explicitly reset any Python or Torch objects that may call Python at destruction
//     voxel_model_.reset();  // Deallocates all tensors and Python wrappers
//     mpSLAM.reset();
// }

int VoxelMapper::getIteration()
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    return iteration_;
}
void VoxelMapper::increaseIteration(const int inc)
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    iteration_ += inc;
}

float VoxelMapper::geoLearningRateInit()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.geo_lr_;
}

float VoxelMapper::sh0LearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.sh0_lr_;
}

float VoxelMapper::shsLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.shs_lr_;
}

float VoxelMapper::lambdaDssim()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.lambda_dssim_;
}

int VoxelMapper::densifyInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.adapt_every_;
}

int VoxelMapper::newKeyframeTimesOfUse()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return new_keyframe_times_of_use_;
}

int VoxelMapper::stableNumIterExistence()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return stable_num_iter_existence_;
}

bool VoxelMapper::isKeepingTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return keep_training_;
}
bool VoxelMapper::isdoingGausPyramidTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return do_gaus_pyramid_training_;
}

bool VoxelMapper::isdoingInactiveGeoDensify()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return inactive_geo_densify_;
}

 void VoxelMapper::setgeoLearningRateInit(const float lr)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     opt_params_.geo_lr_ = lr;
 }
 void VoxelMapper::setsh0LearningRate(const float lr)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     opt_params_.sh0_lr_ = lr;
 }
 void VoxelMapper::setshsLearningRate(const float lr)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     opt_params_.shs_lr_ = lr;
 }
 void VoxelMapper::setLambdaDssim(const float lambda_dssim)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     opt_params_.lambda_dssim_ = lambda_dssim;
 }

 void VoxelMapper::setDensifyInterval(const int interval)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     opt_params_.adapt_every_ = interval;
 }
 void VoxelMapper::setNewKeyframeTimesOfUse(const int times)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     new_keyframe_times_of_use_ = times;
 }
 void VoxelMapper::setStableNumIterExistence(const int niter)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     stable_num_iter_existence_ = niter;
 }
 void VoxelMapper::setKeepTraining(const bool keep)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     keep_training_ = keep;
 }
 void VoxelMapper::setDoGausPyramidTraining(const bool gaus_pyramid)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     do_gaus_pyramid_training_ = gaus_pyramid;
 }
 
 VariableParameters VoxelMapper::getVaribleParameters()
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     VariableParameters params;
     params.geo_lr = opt_params_.geo_lr_;
     params.sh0_lr = opt_params_.sh0_lr_;
     params.shs_lr = opt_params_.shs_lr_;
     params.lambda_dssim = opt_params_.lambda_dssim_;
     params.densify_interval = opt_params_.adapt_every_;
     params.new_kf_times_of_use = new_keyframe_times_of_use_;
     params.stable_num_iter_existence = stable_num_iter_existence_;
     params.keep_training = keep_training_;
     params.do_gaus_pyramid_training = do_gaus_pyramid_training_;
     params.do_inactive_geo_densify = inactive_geo_densify_;
     return params;
 }
 
 void VoxelMapper::setVaribleParameters(const VariableParameters &params)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     opt_params_.geo_lr_ = params.geo_lr;
     opt_params_.sh0_lr_ = params.sh0_lr;
     opt_params_.shs_lr_ = params.shs_lr;
     opt_params_.lambda_dssim_ = params.lambda_dssim;
     opt_params_.adapt_every_ = params.densify_interval;
     new_keyframe_times_of_use_ = params.new_kf_times_of_use;
     stable_num_iter_existence_ = params.stable_num_iter_existence;
     keep_training_ = params.keep_training;
     do_gaus_pyramid_training_ = params.do_gaus_pyramid_training;
     inactive_geo_densify_ = params.do_inactive_geo_densify;
 }
