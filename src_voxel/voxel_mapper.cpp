#include "src_voxel/voxel_mapper_internal.h"

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

    voxel_model_->setOutsideLevel(svrecon_outside_level_);
    voxel_model_->setFixedGlobalSceneExtent(global_scene_extent_m_);
    voxel_model_->setFixedVoxSize(sdf_params_.sdf_voxel_size_m_);
    voxel_model_->setRobustSceneBounds(robust_scene_bounds_);
    voxel_model_->setTopologySdfInitializationMode(sdf_initialization_mode_);
    voxel_model_->setFilterNearVoxels(opt_params_.filter_near_voxels_);

    switch (pSLAM->getSensorType()) {
    case ORB_SLAM3::System::MONOCULAR:
    case ORB_SLAM3::System::IMU_MONOCULAR:
    {
        this->sensor_type_ = MONOCULAR;
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
        throw std::runtime_error("[VoxelMapper] Unsupported sensor type");
    }
    break;
    }

    if (rerun_params_.rerun_monocular_debug_ && sensor_type_ != MONOCULAR) {
        throw std::runtime_error(
            "Record.rerun_monocular_debug requires a monocular sensor");
    }
    if (opt_params_.prune_mvs_consistency_enable_ &&
        (sensor_type_ != MONOCULAR || !isMonocularMvsPipelineEnabled())) {
        throw std::runtime_error(
            "Optimization.prune_mvs_consistency_enable requires a "
            "monocular TANDEM MVS densification or TSDF-evidence pipeline");
    }

    if (sensor_type_ == MONOCULAR) {
        std::cout
            << "[VoxelMapper] Monocular map flow: ORB-controlled poses and "
               "mature ORB MapPoints as weak SVRecon sampling support";
        if (sv::kMonocularRenderedDepthDensify &&
            !isMonocularMvsPipelineEnabled()) {
            std::cout
                << "; local rendered-depth hypotheses update hidden SDF "
                   "evidence before multi-view promotion";
        }
        if (monocular_mvs_tsdf_evidence_) {
            std::cout
                << "; full-image TANDEM MVS TSDF evidence promotes "
                   "confirmed SVRecon cells";
        } else if (monocular_mvs_densify_) {
            std::cout
                << "; TANDEM MVS depth closes residual render holes after "
                   "ORB monocular map initialization";
        }
        std::cout << ".\n";
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
                voxel_utils::cvMatToTorchTensorFloat32(
                    camera.undistort_mask, device_type_);

            cv::Mat viewer_sub_undistort_mask;
            int viewer_image_height_ = camera.height_ * rendered_image_viewer_scale_;
            int viewer_image_width_ = camera.width_ * rendered_image_viewer_scale_;
            cv::resize(camera.undistort_mask, viewer_sub_undistort_mask,
                    cv::Size(viewer_image_width_, viewer_image_height_));
            viewer_sub_undistort_mask_[camera.camera_id_] =
                voxel_utils::cvMatToTorchTensorFloat32(
                    viewer_sub_undistort_mask, device_type_);

            cv::Mat viewer_main_undistort_mask;
            int viewer_image_height_main_ = camera.height_ * rendered_image_viewer_scale_main_;
            int viewer_image_width_main_ = camera.width_ * rendered_image_viewer_scale_main_;
            cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                    cv::Size(viewer_image_width_main_, viewer_image_height_main_));
            viewer_main_undistort_mask_[camera.camera_id_] =
                voxel_utils::cvMatToTorchTensorFloat32(
                    viewer_main_undistort_mask, device_type_);

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

    if (isMonocularMvsPipelineEnabled()) {
        const ORB_SLAM3::System::eSensor slam_sensor = pSLAM->getSensorType();
        if (sensor_type_ != MONOCULAR ||
            (slam_sensor != ORB_SLAM3::System::MONOCULAR &&
             slam_sensor != ORB_SLAM3::System::IMU_MONOCULAR)) {
            throw std::runtime_error(
                "Monocular learned-depth densification requires ORB-SLAM3 "
                "MONOCULAR or IMU_MONOCULAR poses");
        }
        monocular_mvs_requires_inertial_ba1_ =
            slam_sensor == ORB_SLAM3::System::IMU_MONOCULAR;
        if (device_type_ != torch::kCUDA || !torch::cuda::is_available()) {
            throw std::runtime_error(
                "Monocular learned-depth inference requires CUDA");
        }
    }

    if (isMonocularMvsPipelineEnabled()) {
        std::filesystem::path model_path = resolveMapperResourcePath(
            config_file_path_, monocular_mvs_model_dir_);
        if (std::filesystem::is_directory(model_path) ||
            model_path.extension() != ".pt") {
            model_path /= "model.pt";
        }
        monocular_mvs_backend_ =
            std::make_shared<sv::TandemMvsBackend>(model_path);
        std::cout
            << "[VoxelMapper] Loaded TANDEM MVS model: "
            << model_path << "\n"
            << "[VoxelMapper] TANDEM pose gauge: "
            << (monocular_mvs_requires_inertial_ba1_
                    ? "visual-inertial metric"
                    : "pure-monocular ORB scene units")
            << ", depth range: " << sv::kMonocularMvsDepthRangeMode
            << "\n";
        if (monocular_mvs_empty_cache_before_launch_) {
            std::cout
                << "[VoxelMapper] TANDEM will release unused cached CUDA "
                   "blocks before inference.\n";
        }
        if (monocular_mvs_tsdf_evidence_) {
            std::cout
                << "[VoxelMapper] TANDEM topology mode: full-image hidden "
                   "TSDF evidence, stride="
                << sv::kMonocularMvsTsdfEvidencePixelStride
                << ", truncation="
                << sv::kMonocularMvsTsdfEvidenceTruncVox
                << " voxels, promotion views="
                << sv::kMonocularMvsTsdfEvidencePromoteMinViews
                << ".\n";
        }
        if (opt_params_.prune_mvs_consistency_enable_) {
            std::cout
                << "[VoxelMapper] TANDEM MVS pruning: protect supported "
                   "SDF/co-visibility candidates and carve multi-view "
                   "free space; support views="
                << sv::kPruneMvsMinSupportingViews
                << ", contradiction views="
                << sv::kPruneMvsMinContradictingViews
                << ", tolerance="
                << sv::kPruneMvsDepthToleranceVox
                << " voxels.\n";
        }
    }

    // Debug snapshots are queued natively and Rerun is initialized only when
    // the recording is saved, so instrumentation cannot alter online cadence.
    auto& rerun_bridge = sv::RerunVisualizerBridge::instance();
    rerun_bridge.setEnabled(rerun_params_.enable_rerun_);
    const bool requires_live_rerun =
        rerun_params_.run_whole_run_ ||
        rerun_params_.rerun_reconstruction_mesh_ ||
        rerun_params_.rerun_gt_mesh_;
    if (rerun_params_.enable_rerun_ && requires_live_rerun) {
        ensureEmbeddedPythonRuntime(/*import_torch_cuda=*/false);
        rerun_bridge.init("PhotoSLAM-SVRecon", /*spawn_viewer=*/false);
    }
}
