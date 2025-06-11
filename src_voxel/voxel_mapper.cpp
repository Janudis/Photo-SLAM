#include "include_voxel/voxel_mapper.h"

namespace py = pybind11;

inline py::array tensorToNumpyRGB_F32(const torch::Tensor& chw_f32_cpu)
{
    torch::Tensor hwc = chw_f32_cpu.permute({1,2,0}).contiguous();

    return py::array(py::buffer_info(
        hwc.data_ptr<float>(),                        // pointer
        sizeof(float),                                // itemsize
        py::format_descriptor<float>::format(),       // format
        3,                                            // ndim
        { static_cast<pybind11::ssize_t>(hwc.size(0)),
          static_cast<pybind11::ssize_t>(hwc.size(1)),
          static_cast<pybind11::ssize_t>(hwc.size(2)) },        // shape
        { static_cast<pybind11::ssize_t>(hwc.stride(0) * sizeof(float)),
          static_cast<pybind11::ssize_t>(hwc.stride(1) * sizeof(float)),
          static_cast<pybind11::ssize_t>(hwc.stride(2) * sizeof(float)) } // strides
    ));
}

inline void saveDebugImage(torch::Tensor tensor,
                           const std::string& path)
{
    // 1) Move to CPU and detach
    tensor = tensor.detach().to(torch::kCPU);

    // 2) Normalize common 4-D cases → (N,C,H,W)
    if (tensor.dim() == 4) {
        if (tensor.size(0) == 1) {
            // (1,3,H,W) → (3,H,W)
            tensor = tensor.squeeze(0);
        } else if (tensor.size(0) == 3 && tensor.size(1) == 3) {
            // accidental (3,3,H,W) → drop the first dim
            tensor = tensor.select(0, 0);
        }
    }

    // 3) Promote single-channel or 2-D masks to 3-channel
    if (tensor.dim() == 3 && tensor.size(0) == 1) {
        // (1,H,W) → (3,H,W)
        tensor = tensor.expand({3, tensor.size(1), tensor.size(2)});
    } else if (tensor.dim() == 2) {
        // (H,W) → (3,H,W)
        tensor = tensor.unsqueeze(0)
                       .expand({3, tensor.size(0), tensor.size(1)});
    }

    // 4) Final check: must be (3,H,W)
    if (tensor.dim() != 3 || tensor.size(0) != 3) {
        std::cerr << "[ERROR] saveDebugImage: unsupported tensor shape "
                  << tensor.sizes() << '\n';
        return;
    }

    // 5) Convert to 0–255 uint8
    if (tensor.dtype() != torch::kUInt8) {
        tensor = tensor.clamp(0,1).mul(255).to(torch::kUInt8);
    }

    // 6) HWC layout
    tensor = tensor.permute({1,2,0}).contiguous();  // (H,W,3)

    // 7) Wrap in OpenCV and write
    int H = tensor.size(0), W = tensor.size(1);
    cv::Mat img(H, W, CV_8UC3, tensor.data_ptr<uint8_t>());

    // OpenCV expects BGR
    cv::Mat bgr;
    cv::cvtColor(img, bgr, cv::COLOR_RGB2BGR);
    if (img.empty()) 
        std::cerr << "[ERROR] Image is empty, skipping saving..." << std::endl;
    cv::imwrite(path, bgr);
}

VoxelMapper::VoxelMapper(std::shared_ptr<ORB_SLAM3::System> pSLAM,
                         const std::filesystem::path& config_file_path,
                         const std::filesystem::path& seq_dir,
                         const std::filesystem::path& out_dir,
                         torch::DeviceType device_type,
                         int seed)
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

    result_dir_ = mOutDir;
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir_);
    config_file_path_ = config_file_path;
    readConfigFromFile(config_file_path_);
    mSeqDir = seq_dir;
    mOutDir = out_dir;    

    // Background & override color setup
    std::vector<float> bg_color = {0.0f, 0.0f, 0.0f};  // no white-background logic needed
    background_ = torch::tensor(bg_color,
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    override_color_ = torch::empty(0, torch::TensorOptions().device(device_type_));

    voxel_model_ = std::make_shared<sv::VoxelModel>(model_params_);
    scene_       = std::make_shared<sv::VoxelScene>(model_params_);
    
    switch (pSLAM->getSensorType()) {
    case ORB_SLAM3::System::MONOCULAR:
    case ORB_SLAM3::System::IMU_MONOCULAR:
        sensor_type_ = MONOCULAR;
        break;
    case ORB_SLAM3::System::STEREO:
    case ORB_SLAM3::System::IMU_STEREO:
        sensor_type_ = STEREO;
        break;
    case ORB_SLAM3::System::RGBD:
    case ORB_SLAM3::System::IMU_RGBD:
        sensor_type_ = RGBD;
        break;
    default:
        throw std::runtime_error("[Voxel Mapper]Unsupported sensor type!");
    }

    // /* Load every ORB-SLAM3 camera, convert to Camera, pre–compute            */
    auto settings = mpSLAM->getSettings();   
    cv::Size SLAM_im_size = settings->newImSize();
    UndistortParams undistort_params(
        SLAM_im_size,
        settings->camera1DistortionCoef()
    );
    auto vpCameras = mpSLAM->getAtlas()->GetAllCameras();
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

            camera.initUndistortRectifyMapAndMask(K, SLAM_im_size, K_new);

            undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    camera.undistort_mask, device_type_);
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
    std::lock_guard<std::mutex> guard(mutex_status_);

    /* ───────── PIPELINE FLAGS ───────── */
    inactive_geo_densify_ =
        (settings_file["Mapper.inactive_geo_densify"].operator int()) != 0;
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

    pipe_params_.convert_SHs_ =
         (settings_file["Pipeline.convert_SHs"].operator int()) != 0;
    pipe_params_.compute_cov3D_ =
         (settings_file["Pipeline.compute_cov3D"].operator int()) != 0;

    /* ───────── OPTIMIZATION PARAMETERS ───────── */
    opt_params_.iterations_ =
        settings_file["Optimization.max_num_iterations"].operator int();
    opt_params_.geo_lr_init_ =
        settings_file["Optimization.geo_lr_init"].operator float();
    opt_params_.geo_lr_final_ =
        settings_file["Optimization.geo_lr_final"].operator float();
    opt_params_.geo_lr_delay_mult_ =
        settings_file["Optimization.geo_lr_delay_mult"].operator float();
    opt_params_.geo_lr_max_steps_ =
        settings_file["Optimization.geo_lr_max_steps"].operator int();
    opt_params_.meta_accum_lr_ =
        settings_file["Optimization.meta_accum_lr"].operator float();
    opt_params_.sh0_lr_ =
        settings_file["Optimization.sh0_lr"].operator float();
    opt_params_.shs_lr_ =
        settings_file["Optimization.shs_lr"].operator float();

    opt_params_.subdiv_from_ =
        settings_file["Optimization.subdiv_from"].operator int();
    opt_params_.subdiv_every_ =
        settings_file["Optimization.subdiv_every"].operator int();
    opt_params_.subdiv_until_ =
        settings_file["Optimization.subdiv_until"].operator int();
    opt_params_.subdiv_quantile_ =
        settings_file["Optimization.subdiv_quantile"].operator float();
    opt_params_.subdiv_gradient_threshold_ =
        settings_file["Optimization.subdiv_gradient_threshold"].operator float();

    opt_params_.prune_from_ =
        settings_file["Optimization.prune_from"].operator int();
    opt_params_.prune_every_ =
        settings_file["Optimization.prune_every"].operator int();
    opt_params_.prune_until_ =
        settings_file["Optimization.prune_until"].operator int();
    opt_params_.prune_threshold_init_ =
        settings_file["Optimization.prune_threshold_init"].operator float();
    opt_params_.prune_threshold_final_ =
        settings_file["Optimization.prune_threshold_final"].operator float();
    opt_params_.min_voxels_ =
        settings_file["Optimization.min_voxels"].operator int();

    opt_params_.densification_interval_ =
        settings_file["Optimization.densification_interval"].operator int();
    opt_params_.lambda_dssim_ =
        settings_file["Optimization.lambda_dssim"].operator float();

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

    std::cout << "\n[CFG] Parsed Optimization Parameters:" << std::endl;
    // std::cout << "  lr:                       " << opt_params_.position_lr_final_ << std::endl;
    std::cout << "  meta_accum_lr:           " << opt_params_.meta_accum_lr_ << std::endl;
    // std::cout << "  position_lr_init:        " << opt_params_.position_lr_init_ << std::endl;
    // std::cout << "  position_lr_delay_mult:  " << opt_params_.position_lr_delay_mult_ << std::endl;
    // std::cout << "  position_lr_max_steps:   " << opt_params_.position_lr_max_steps_ << std::endl;
    std::cout << "  iterations_:      " << opt_params_.iterations_ << std::endl;
    std::cout << "  densification_interval:  " << opt_params_.densification_interval_ << std::endl;
    std::cout << "\n[CFG] Subdivision Parameters:" << std::endl;
    std::cout << "  subdiv_from:             " << opt_params_.subdiv_from_ << std::endl;
    std::cout << "  subdiv_every:            " << opt_params_.subdiv_every_ << std::endl;
    std::cout << "  subdiv_until:            " << opt_params_.subdiv_until_ << std::endl;
    std::cout << "  subdiv_quantile:         " << opt_params_.subdiv_quantile_ << std::endl;
    std::cout << "  subdiv_gradient_threshold: " << opt_params_.subdiv_gradient_threshold_ << std::endl;
    std::cout << "\n[CFG] Pruning Parameters:" << std::endl;
    std::cout << "  prune_from:              " << opt_params_.prune_from_ << std::endl;
    std::cout << "  prune_every:             " << opt_params_.prune_every_ << std::endl;
    std::cout << "  prune_until:             " << opt_params_.prune_until_ << std::endl;
    std::cout << "  prune_threshold_init:    " << opt_params_.prune_threshold_init_ << std::endl;
    std::cout << "  prune_threshold_final:   " << opt_params_.prune_threshold_final_ << std::endl;
    std::cout << "  min_voxels:              " << opt_params_.min_voxels_ << std::endl;
    std::cout << "\n[CFG] Pipeline & Mapper Flags:" << std::endl;
    std::cout << "  inactive_geo_densify:    " << inactive_geo_densify_ << std::endl;
    std::cout << "  new_keyframe_times_of_use_: " << new_keyframe_times_of_use_<< std::endl;
    std::cout << "  min_num_initial_map_kfs: " << min_num_initial_map_kfs_ << std::endl;
    std::cout << "  large_rot_th:            " << large_rot_th_ << std::endl;
    std::cout << "  large_trans_th:          " << large_trans_th_ << std::endl;
    std::cout << "  cull_keyframes:          " << cull_keyframes_ << std::endl;
    std::cout << "\n[CFG] Logging Parameters:" << std::endl;
    std::cout << "  training_report_interval: " << training_report_interval_ << std::endl;
    std::cout << "  keyframe_record_interval: " << keyframe_record_interval_ << std::endl;
    std::cout << "  all_keyframes_record_interval: " << all_keyframes_record_interval_ << std::endl;
    std::cout << "  record_rendered_image:   " << record_rendered_image_ << std::endl;
    std::cout << "  record_ground_truth_image: " << record_ground_truth_image_ << std::endl;
    std::cout << "  record_loss_image:       " << record_loss_image_ << std::endl;
}

void VoxelMapper::run()
{
    /* expose our helper scripts to the embedded Python side */
    py::gil_scoped_acquire gil;
    py::module_::import("sys").attr("path").attr("insert")(0, "../scripts_voxel");

    /* ───────────────────────────────────────────────
     *  1.  INITIAL VOXEL   M A P P I N G  LOOP
     * ─────────────────────────────────────────────── */
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

                for (auto *pMP : vMPs)
                {
                    Point3D pt;
                    const auto pos = pMP->GetWorldPos();
                    const auto color = pMP->GetColorRGB();
                    // Debug: Print RGB color and position for each MapPoint
                    // std::cout << "[DEBUG] MapPoint ID: " << pMP->mnId 
                    //           << " Position: (" << pos.x() << ", " << pos.y() << ", " << pos.z() << ")"
                    //           << " Color: (" << color(0) << ", " << color(1) << ", " << color(2) << ")\n";
                    pt.xyz_ << pos.x(), pos.y(), pos.z();
                    pt.color_ << color(0), color(1), color(2);
                    // Cache the point in the scene
                    scene_->cachePoint3D(pMP->mnId, pt);
                }
                // B) Create VoxelKeyframes from each SLAM KeyFrame
                // std::cout << "[DEBUG] Creating VoxelKeyframes..." << std::endl;
                for (auto *pKF : vKFs)
                {
                    auto new_kf = std::make_shared<VoxelKeyframe>(pKF->mnId, getIteration());
                    new_kf->zfar_  = z_far_;
                    new_kf->znear_ = z_near_;

                    // Pose
                    auto pose = pKF->GetPose();
                    new_kf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>()
                    );
                    cv::Mat imgRGB_undistorted;
                    // Debug: Print Keyframe pose
                    // std::cout << "[DEBUG] Keyframe ID: " << pKF->mnId 
                    //           << " Pose: (" << pose.translation().x() << ", " << pose.translation().y() << ", " << pose.translation().z() << ")\n";

                    // Camera
                    sv::Camera& camera = scene_->cameras_.at(pKF->mpCamera->GetId());
                    new_kf->setCameraParams(camera);
                    // Image (left if STEREO)
                    cv::Mat imgRGB = pKF->imgLeftRGB;
                    camera.undistortImage(imgRGB, imgRGB_undistorted);
                    new_kf->original_image_ =
                        tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
                    new_kf->img_filename_ = pKF->mNameFile;

                    // Compute transformations
                    new_kf->computeTransformTensors();
                    scene_->addKeyframe(new_kf, &kfid_shuffled_);

                    increaseKeyframeTimesOfUse(new_kf, newKeyframeTimesOfUse());
                }
            }   // Mutex released
                // D) Create voxel model & trainer setup
                {
                    std::unique_lock<std::mutex> lock_render(mutex_render_);
                    scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
                    voxel_model_->createFromPcd(scene_->cached_point_cloud_, scene_->cameras_extent_);
                    std::unique_lock<std::mutex> lock(mutex_settings_);
                    voxel_model_->trainingSetup(opt_params_);
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
    /* ───────────────────────────────────────────────
     *  2.  INCREMENTAL   M A P P I N G  LOOP
     * ───────────────────────────────────────────── */
    int SLAM_stop_iter = 0;
    while (!isStopped())
    {
        if (hasMetIncrementalMappingConditions())
        {
            combineMappingOperations();
            if (cull_keyframes_) 
                cullKeyframes();
        }

        trainForOneIteration();

        if (mpSLAM->isShutDown() && !SLAM_ended_)
        {
            SLAM_stop_iter = getIteration();
            SLAM_ended_    = true;
        }
        if (SLAM_ended_ || getIteration() >= opt_params_.iterations_)
            break;
    }
    /* ───────────────────────────────────────────────
     *  3.  TAIL   O P T I M I S A T I O N
     * ───────────────────────────────────────────── */
    int densify_interval = densifyInterval();
    int n_delay_iters    = int(densify_interval * 0.8f);

    while (   getIteration() - SLAM_stop_iter <= n_delay_iters
           || getIteration() % densify_interval <= n_delay_iters
           || isKeepingTraining())
    {
        trainForOneIteration();
        densify_interval = densifyInterval();
        n_delay_iters    = int(densify_interval * 0.8f);
    }
    /* ───────────────────────────────────────────────
     *  4.  SAVE & SHUTDOWN
     * ───────────────────────────────────────────── */
    renderAndRecordAllKeyframes("_shutdown");
    // savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
    writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

    signalStop();
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

void VoxelMapper::combineMappingOperations()
{
    // Continuously consume any pending MappingOperation from ORB‐SLAM3’s Atlas
    while (mpSLAM->getAtlas()->hasMappingOperation()) {
        ORB_SLAM3::MappingOperation opr =
            mpSLAM->getAtlas()->getAndPopMappingOperation();

        switch (opr.meOperationType)
        {
            // ─────────── 1) LOCAL BUNDLE‐ADJUSTMENT ───────────
            case ORB_SLAM3::MappingOperation::OprType::LocalMappingBA: {
                auto &associated_kfs = opr.associatedKeyFrames();

                for (auto &kf : associated_kfs) {
                    // Unpack the keyframe tuple
                    unsigned long kfid = std::get<0>(kf);
                    auto &pose        = std::get<2>(kf);

                    // See if this keyframe already exists in our voxel‐scene
                    std::shared_ptr<VoxelKeyframe> existing_kf =
                        scene_->getKeyframe(kfid);

                    if (existing_kf) {
                        // — Keyframe already present → update its pose
                        existing_kf->setPose(
                            pose.unit_quaternion().cast<double>(),
                            pose.translation().cast<double>()
                        );
                        existing_kf->computeTransformTensors();

                        // — Give the “Local BA” keyframe extra times‐of‐use
                        increaseKeyframeTimesOfUse(
                            existing_kf,
                            local_BA_increased_times_of_use_
                        );
                    }
                    else {
                        // — Brand‐new keyframe → hand off to handleNewKeyframe()
                        handleNewKeyframe(kf);
                    }
                }

                // ── Inject any brand‐new map‐points into the voxel model ──
                //    (Photo‐SLAM calls gaussians_->increasePcd(...). In SVRaster,
                //     you must call your own “add‐points” or “inactive‐geo densify”
                //     routine here.)
                {
                    auto &associated_points = opr.associatedMapPoints();
                    auto &pts              = std::get<0>(associated_points);
                    auto &colors           = std::get<1>(associated_points);

                    if (initial_mapped_ && pts.size() >= 30) {
                        torch::NoGradGuard no_grad;
                        std::unique_lock<std::mutex> lock_render(mutex_render_);

                        // TODO: replace the following line with your SVRaster‐equivalent:
                        //        e.g. voxel_model_->increasePointCloud(pts, colors, getIteration());
                        //        Or scene_->addNewPoints(pts, colors, getIteration());
                        //gaussians_->increasePcd(pts, colors, getIteration());
                    }
                }
                break;
            }

            // ─────────── 2) LOOP‐CLOSING BUNDLE‐ADJUSTMENT ───────────
            case ORB_SLAM3::MappingOperation::OprType::LoopClosingBA: {
                std::cout << "[VoxelMapper] Loop Closure Detected." << std::endl;

                // 2.1) Pull out scale factor (if provided by ORB-SLAM3)
                float loop_kf_scale = opr.mfScale;

                // 2.2) Build a mask for “which voxels have already been corrected”
                //      (Photo‐SLAM does: torch::full({N}, true); we simply mimic the pattern.)
                //      In SVRaster, you may not need an exact equivalent, but keep the same shape:
                //      number of voxels = e.g. voxel_centers_.size(0).
                torch::Tensor point_not_transformed_flags =
                    torch::full(
                        { static_cast<int64_t>(voxel_centers_.size(0)) },
                        true,
                        torch::TensorOptions()
                            .device(device_type_)
                            .dtype(torch::kBool)
                    );

                int num_transformed = 0;

                // 2.4) Iterate over all associated keyframes in this loop closure
                auto &associated_kfs = opr.associatedKeyFrames();
                for (auto &kf : associated_kfs) {
                    unsigned long kfid = std::get<0>(kf);
                    auto &pose        = std::get<2>(kf);

                    // See if the KF already exists in our scene
                    std::shared_ptr<VoxelKeyframe> pkf = scene_->getKeyframe(kfid);

                    // (2.4.1) If pkf exists → check if its pose changed “too much”
                    if (pkf) {
                        Sophus::SE3f original_pose = pkf->getPosef();
                        Sophus::SE3f inv_pose      = pose.inverse();
                        Sophus::SE3f diff_pose     = inv_pose * original_pose;

                        bool large_rot   = !diff_pose.rotationMatrix().isApprox(
                                               Eigen::Matrix3f::Identity(),
                                               large_rot_th_
                                           );
                        bool large_trans = !diff_pose.translation().isMuchSmallerThan(
                                               1.0f,
                                               large_trans_th_
                                           );

                        if (large_rot || large_trans) {
                            std::cout << "[VoxelMapper] Large loop correction on KF "
                                      << kfid << std::endl;

                            // Adjust translation by scale factor
                            diff_pose.translation()  -= inv_pose.translation();
                            diff_pose.translation()  *= loop_kf_scale;
                            diff_pose.translation()  += inv_pose.translation();

                            // Convert to a (4×4) torch tensor for SVRaster
                            torch::Tensor diff_pose_tensor =
                                tensor_utils::EigenMatrix2TorchTensor(
                                    diff_pose.matrix(),
                                    device_type_
                                ).transpose(0, 1);

                            {
                                // Lock while pushing transform to voxel model
                                std::unique_lock<std::mutex> lock_render(mutex_render_);

                                // TODO: call SVRaster equivalent:
                                //        e.g. voxel_model_->applyLoopCorrection(
                                //                point_not_transformed_flags,
                                //                diff_pose_tensor,
                                //                pkf->world_view_transform_,
                                //                pkf->full_proj_transform_,
                                //                pkf->creation_iter_,
                                //                stableNumIterExistence(),
                                //                num_transformed,
                                //                loop_kf_scale);
                                //
                                //gaussians_->scaledTransformVisiblePointsOfKeyframe(
                                //    point_not_transformed_flags,
                                //    diff_pose_tensor,
                                //    pkf->world_view_transform_,
                                //    pkf->full_proj_transform_,
                                //    pkf->creation_iter_,
                                //    stableNumIterExistence(),
                                //    num_transformed,
                                //    loop_kf_scale);
                            }

                            // Grant extra “times of use” for this KF, per Photo‐SLAM
                            increaseKeyframeTimesOfUse(
                                pkf,
                                loop_closure_increased_times_of_use_
                            );
                        }

                        // (2.4.2) In all cases, update KF’s pose & transforms
                        pkf->setPose(
                            pose.unit_quaternion().cast<double>(),
                            pose.translation().cast<double>()
                        );
                        pkf->computeTransformTensors();
                    }
                    else {
                        // (2.4.3) KF not yet in scene → treat as brand‐new
                        handleNewKeyframe(kf);
                    }

                    // If new points arrived in handleNewKeyframe(), extend our flag‐tensor
                    // so that newly inserted voxels are also marked “un‐corrected”:
                    int64_t new_pts = static_cast<int64_t>(voxel_centers_.size(0))
                                        - point_not_transformed_flags.size(0);
                    if (new_pts > 0) {
                        torch::Tensor extra = torch::full(
                            { new_pts },
                            true,
                            torch::TensorOptions()
                                .device(device_type_)
                                .dtype(torch::kBool)
                        );
                        point_not_transformed_flags =
                            torch::cat({ point_not_transformed_flags, extra }, 0);
                    }
                }

                // 2.6) Finally, ingest any brand‐new map‐points that came with this LoopClosingBA
                {
                    auto &assoc_pts = opr.associatedMapPoints();
                    auto &pts       = std::get<0>(assoc_pts);
                    auto &colors    = std::get<1>(assoc_pts);

                    if (initial_mapped_ && pts.size() >= 30) {
                        torch::NoGradGuard no_grad;
                        std::unique_lock<std::mutex> lock_render(mutex_render_);

                        // TODO: replace with SVRaster equivalent of “adding new points”:
                        //        e.g. voxel_model_->increasePointCloud(pts, colors, getIteration());
                        //gaussians_->increasePcd(pts, colors, getIteration());
                    }
                }

                // Mark that we’ve processed a loop‐closure iteration
                loop_closure_iteration_ = true;
                break;
            }

            // ─────────── 3) SCALE REFINEMENT ───────────
            case ORB_SLAM3::MappingOperation::OprType::ScaleRefinement: {
                std::cout << "[VoxelMapper] Scale refinement detected. Transforming all KFs & voxels..." << std::endl;

                float s       = opr.mfScale;
                Sophus::SE3f &T = opr.mT;

                if (initial_mapped_) {
                    {
                        // 3.1) First, apply to voxel model (SVRaster)
                        std::unique_lock<std::mutex> lock_render(mutex_render_);

                        // TODO: call SVRaster‐equivalent to scale “voxel geometry”:
                        //        e.g. voxel_model_->applyScaledTransformation(s, T);
                        //gaussians_->applyScaledTransformation(s, T);
                    }

                    // 3.2) Then apply to every keyframe’s pose in the scene
                    scene_->applyScaledTransformation(s, T);
                }
                else {
                    // (this branch “should not happen” once initial mapped is true,
                    //  but we mirror Photo‐SLAM’s fallback)
                    // 3.3) Transform any cached 3D points (not yet in scene_)
                    for (auto &pt_pair : scene_->cached_point_cloud_) {
                        auto &pt_xyz = pt_pair.second.xyz_;
                        pt_xyz *= s;
                        pt_xyz = T.cast<double>() * pt_xyz;
                    }

                    // 3.4) Also transform every VoxelKeyframe we’ve stored so far
                    for (auto &kv : scene_->keyframes()) {
                        std::shared_ptr<VoxelKeyframe> pkf = kv.second;

                        // Convert Twc → Tyc with scale s
                        Sophus::SE3f Twc = pkf->getPosef().inverse();
                        Twc.translation() *= s;
                        Sophus::SE3f Tyc = T * Twc;
                        Sophus::SE3f Tcy = Tyc.inverse();

                        pkf->setPose(
                            Tcy.unit_quaternion().cast<double>(),
                            Tcy.translation().cast<double>()
                        );
                        pkf->computeTransformTensors();
                    }
                }
                break;
            }

            // ─────────── 4) ANY OTHER MAPPING‐OP TYPE ───────────
            default: {
                // If Photo‐SLAM would normally handle other cases, throw an error;
                // for VoxelMapper we simply ignore everything else.
                // (You can choose to log or assert here, if you want.)
                break;
            }
        }
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
    // // ─── 1) Unpack the tuple ─────────────────────────────────────────────────
    // unsigned long kfid       = std::get<0>(kf);
    // unsigned long camid      = std::get<1>(kf);
    // const Sophus::SE3f &pose  = std::get<2>(kf);
    // const cv::Mat   &imgRGB  = std::get<3>(kf);
    // const std::string &img_file = std::get<8>(kf);
    // (We ignore fields 4, 5, 6, 7 since VoxelMapper doesn’t use them.)

    // ─── 2) Create a new VoxelKeyframe, exactly like Photo-SLAM’s Gaussian case ─
    auto pkf = std::make_shared<VoxelKeyframe>(std::get<0>(kf), getIteration());
    pkf->zfar_  = z_far_;
    pkf->znear_ = z_near_;
    auto& pose = std::get<2>(kf);
    // ─── 3) Set its pose ───────────────────────────────────────────────────────
    pkf->setPose(
        pose.unit_quaternion().cast<double>(),
        pose.translation().cast<double>()
    );
    cv::Mat imgRGB_undistorted;
    // Camera
    sv::Camera& camera = scene_->cameras_.at(std::get<1>(kf));
    pkf->setCameraParams(camera);

    // Image (left if STEREO)
    cv::Mat imgRGB = std::get<3>(kf);
    camera.undistortImage(imgRGB, imgRGB_undistorted);
    pkf->original_image_ =
        tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
    pkf->img_filename_ = std::get<8>(kf);
     
     // Add the new keyframe to the scene
     pkf->computeTransformTensors();
     scene_->addKeyframe(pkf, &kfid_shuffled_);
 
     // Give new keyframes times of use and add it to the training sliding window
     increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());
 }

bool VoxelMapper::isStopped() const {
    return stopped_;
}

void VoxelMapper::signalStop(const bool going_to_stop)
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    this->stopped_ = going_to_stop;
}

inline bool is_between(int iter, int from, int until)
{ return iter >= from && iter <= until; }

// Add training with optimization loop
void VoxelMapper::trainForOneIteration()
{
    // 1) bump global iteration counter
    increaseIteration(1);

    // 2) pick a random keyframe from the sliding window
    std::shared_ptr<VoxelKeyframe> viewpoint_cam = useOneRandomSlidingWindowKeyframe();
    if (!viewpoint_cam) {
        // if none available, roll back iteration and exit
        increaseIteration(-1);
        return;
    }
    writeKeyframeUsedTimes(result_dir_ / "used_times");

    // 3) select ground truth image + mask tensors
    //    (it always use the “original” resolution in our voxel case)
    int image_height, image_width;
    torch::Tensor gt_image, mask;
    image_height = viewpoint_cam->image_height_;
    image_width = viewpoint_cam->image_width_;
    gt_image = viewpoint_cam->original_image_
                                   .to(mDevice)          // (3,H,W)
                                   .unsqueeze(0);        // → (1,3,H,W)
    mask = undistort_mask_[viewpoint_cam->camera_id_]
                                   .to(mDevice)
                                   .to(torch::kFloat32); // (H,W)
    // if it somehow came in as 3×H×W, just take the first (they're identical)
    if (mask.dim() == 3 && mask.size(0) == 3) {
        mask = mask[0];   // now (H,W)
    }

    // now make it 1×1×H×W
    if (mask.dim() == 2) {
        mask = mask.unsqueeze(0).unsqueeze(0);
    }
    else if (mask.dim() == 3) {
        // if somebody gave you (1,H,W) already:
        mask = mask.unsqueeze(1);  // (1,1,H,W)
    }
    // std::cerr << "[DBG] final mask dim=" << mask.dim()
    //         << " sizes=" << mask.sizes() << "\n";

    // 4) grow SH degree every 1000 iterations (locked during render)
    std::unique_lock<std::mutex> lock_render(mutex_render_);
    
    if (getIteration() % 1000 == 0 && default_sh_ < model_params_.sh_degree_)
        default_sh_ += 1;
    voxel_model_->setShDegree(default_sh_);

    // 5) build a MiniCam out of this keyframe
    // const Camera& cinfo = viewpoint_cam->getCamera();
    // sv::MiniCam cam = sv::MiniCam::fromIntrinsics(
    //     cinfo.fx(), cinfo.fy(), cinfo.cx(), cinfo.cy(),
    //     img_w, img_h,
    //     static_cast<int>(viewpoint_cam->fid_)
    // );
    // // set c2w, w2c
    // {
    //     Eigen::Matrix4f pose = viewpoint_cam->pose.matrix();
    //     Eigen::Matrix4f c2w = pose.inverse();
    //     cam.c2w = torch::from_blob(c2w.data(), {4,4}, torch::kFloat32).clone().to(mDevice);
    //     cam.w2c = torch::from_blob(pose.data(),  {4,4}, torch::kFloat32).clone().to(mDevice);
    // }
    // viewpoint_cam->computeTransformTensors();            // idempotent
    sv::MiniCam cam = viewpoint_cam->toMiniCam();
    cam.c2w = cam.c2w.contiguous().to(mDevice);   // make sure contiguous + on CUDA
    cam.w2c = cam.w2c.contiguous().to(mDevice);

    // 6) update all learning rates
    if (mpSLAM) {
        int used_times = kfs_used_times_[viewpoint_cam->fid_];
        int step = (used_times <= opt_params_.geo_lr_max_steps_)
                       ? used_times
                       : opt_params_.geo_lr_max_steps_;
        float geo_lr = voxel_model_->updateLearningRate(step);
        voxel_model_->setGeoLearningRate(geo_lr);
    } else {
        voxel_model_->updateLearningRate(getIteration());
    }
    // SH-DC and higher-SH rates remain constant:
    voxel_model_->setSh0LearningRate(opt_params_.sh0_lr_);
    voxel_model_->setShsLearningRate(opt_params_.shs_lr_);

    torch::Tensor chw_u8 = viewpoint_cam->original_image_    // (3,H,W) in [0,1]
                            .mul(255.0f)
                            .clamp(0.0f, 255.0f)
                            .to(torch::kUInt8)            // <--- cast to U8
                            .cpu()
                            .contiguous();
    torch::Tensor hwc_u8 = chw_u8.permute({1,2,0}).contiguous();  // (H,W,3)
    // std::cout << "hwc_u8 sizes: " << hwc_u8.sizes() << std::endl;
    // convert CHW→HWC uint8 numpy without any CUDA involvement
    py::array rgb_numpy = sv::tensorToNumpyRGB(hwc_u8);
    // if you still want a debug JPEG, bring back as float
    saveDebugImage(chw_u8.to(torch::kFloat32).div(255.0f), (result_dir_ / "debug_gt.jpg").string());
    // std::cout << "[DEBUG] Camera c2w matrix size: " << cam.c2w.sizes() << std::endl;
    // std::cout << "[DEBUG] Camera w2c matrix size: " << cam.w2c.sizes() << std::endl;

    // std::cout << "[DEBUG] RGB numpy shape before passing to renderer: " << rgb_numpy.shape() << std::endl;
    auto render_pkg = voxel_model_->render(cam, rgb_numpy, "");
    // lock_render.unlock();
    if (render_pkg.empty() || !render_pkg.count("rgb") || !render_pkg.at("rgb").defined()) {
        std::cout << "render pkg empty" << std::endl;
        return;
    }
    // Keep the batch dimension: rendered_image is now (1,3,H,W)

    torch::Tensor rendered_image = render_pkg["rgb"].to(mDevice);

    // std::cout << "[DEBUG] gt_image size: " << gt_image.sizes() << std::endl;
    // std::cerr << "[DBG] rendered_image dim=" << rendered_image.dim()
    //     << " sizes=" << rendered_image.sizes() << "\n";
    // std::cout << "[DEBUG] mask size: " << mask.sizes() << std::endl;
    saveDebugImage(rendered_image.squeeze(0), (result_dir_ / "debug_rendered.jpg").string());
    
    torch::Tensor masked_image = rendered_image * mask;      // (1,3,H,W)
    // std::cerr << "[DBG] masked_image dim=" << masked_image.dim()
    //         << " sizes=" << masked_image.sizes() << "\n";        
    saveDebugImage(masked_image.squeeze(0), (result_dir_ / "debug_mask.jpg").string());
    
    // 8) compute masked photometric loss
    torch::Tensor Ll1, loss;
    {
        py::gil_scoped_release no_gil;
        // mask out black/invalid pixels
        // masked_image = rendered_image * mask;   // (1,3,H,W)
        // torch::Tensor masked_gt   = gt_image        * mask;   // (1,3,H,W)
        
        Ll1 = loss_utils::l1_loss(masked_image, gt_image);
        float lambda_dssim = lambdaDssim();
        loss = (1.0 - lambda_dssim) * Ll1
             + lambda_dssim * (1.0 - loss_utils::ssim(masked_image, gt_image, mDevice.type()));
        loss.backward();
        if (mDevice == torch::kCUDA) {
            torch::cuda::synchronize();
        }
        // === DEBUG GRADIENTS ===
        // auto params = voxel_model_->parameters();  // or voxel_model_->getTrainableParams()
        // for (size_t i = 0; i < params.size(); ++i) {
        //     auto& p = params[i];
        //     if (p.grad().defined()) {
        //         std::cout << "[DEBUG] Param " << i
        //                 << ": grad max = " << p.grad().abs().max().item<float>()
        //                 << ", mean = " << p.grad().abs().mean().item<float>() << "\n";
        //     } else {
        //         std::cout << "[DEBUG] Param " << i << ": grad undefined\n";
        //     }
        // }
    }
    // {
    //     // rebuild rgb_numpy (or reuse the one from above)
    //     py::array rgb_numpy2 = tensorToNumpyRGB_F32(viewpoint_cam->img_tensor.cpu());
    //     // re-lock for thread safety during render
    //     std::unique_lock<std::mutex> lock2(mutex_render_);
    //     auto render_pkg2 = voxel_model_->render(cam, rgb_numpy2, "");
    //     lock2.unlock();
    //     if (!render_pkg2.empty() && render_pkg2.count("rgb") && render_pkg2.at("rgb").defined()) {
    //         torch::Tensor post_render = render_pkg2["rgb"].to(mDevice).squeeze(0); // (3,H,W)
    //         saveDebugImage(post_render, (result_dir_ / "debug_post_update.jpg").string());
    //     }
    // }

    // 10) accumulate subdivision gradients into subdiv_meta_
    {
        torch::NoGradGuard no_grad;
        // update EMA
        ema_loss_for_log_ = 0.4f * loss.item<float>() + 0.6f * ema_loss_for_log_;

        if (keyframe_record_interval_ &&
            getIteration() % keyframe_record_interval_ == 0)
            recordKeyframeRendered(masked_image, gt_image, viewpoint_cam->fid_, result_dir_, result_dir_, result_dir_);

        // get logging gradient from subdiv_p_ → use new method name getSubdivPriorityGrad()
        torch::Tensor grad_p = voxel_model_->getSubdivPriorityGrad();
        // clamp subdiv_meta_ ← getTensor("subdiv_meta")
        torch::Tensor subdiv_meta = voxel_model_->getTensor("subdiv_meta");
        subdiv_meta = torch::clamp(subdiv_meta + grad_p * opt_params_.meta_accum_lr_, 0.0f, 1.0f);
        voxel_model_->setSubdivMeta(subdiv_meta);

        // scatter‐add full gradient array back into "grad buffer"
        torch::Tensor idx_all = torch::arange(grad_p.numel(), grad_p.options().dtype(torch::kLong));
        voxel_model_->accumulateSubdivGradients(idx_all, grad_p);

        // // every training_report_interval_ iterations, print a concise report
        // if (training_report_interval_ && iteration_ % training_report_interval_ == 0) {
        //     auto iter_end_timing = std::chrono::steady_clock::now();
        //     int64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        //         iter_end_timing - start_time_  // you can store start_time_ at top if desired
        //     ).count();
        //     VoxelTrainer::trainingReport(
        //         iteration_,
        //         opt_params_.iterations_,
        //         Ll1,
        //         loss,
        //         ema_loss_for_log,
        //         loss_utils::l1_loss,
        //         elapsed_ms,
        //         *voxel_model_,
        //         *scene_,
        //         pipe_params_,
        //         background_
        //     );
        // }

        // // densification & pruning
        // ...

        // optionally record keyframe visuals / save PLY, etc., exactly when
        // iteration_ % all_keyframes_record_interval_ == 0 (if you want)
        if (all_keyframes_record_interval_ &&
            getIteration() % all_keyframes_record_interval_ == 0)
        {
            renderAndRecordAllKeyframes();
            // savePly(result_dir_ / std::to_string(iteration_) / "ply");
        }

        // 10) optimizer_ step (only geometry group is updated by updateLearningRate)
        {
            py::gil_scoped_release no_gil;
            torch::NoGradGuard no_grad;
            voxel_model_->optimizer_->step();
            voxel_model_->optimizer_->zero_grad();
        }

        if (training_report_interval_ && iteration_ % training_report_interval_ == 0) {
            std::cout << "[TRAIN] iter " << iteration_
                    << "  L1: "    << Ll1.item<float>()
                    << "  loss: "  << loss.item<float>()
                    << "  ema: "   << ema_loss_for_log_ << '\n';
        }
        if (iteration_ % 50 == 0) {
            writeKeyframeUsedTimes(result_dir_);
        }
    }
}

//------------------------------------------------------------
// helper: copy Mat → NumPy (NumPy owns the memory)
//------------------------------------------------------------
py::array_t<uint8_t> cvMatToNumpyRGB(const cv::Mat &img)
{
    if (img.empty() || img.type() != CV_8UC3)
        throw std::runtime_error("Expected a non-empty 3-channel CV_8UC3 image");

    // allocate new NumPy array that *owns* its data
    py::array_t<uint8_t> arr({img.rows, img.cols, 3});
    std::memcpy(arr.mutable_data(), img.data,
                static_cast<size_t>(img.rows * img.cols * 3));
    return arr;   // safe – Python holds the buffer
}

// void VoxelMapper::increaseKeyframeTimesOfUse(std::shared_ptr<VoxelKeyframe> pkf, int value)
// {
//     pkf->remaining_times_of_use_ += value;
//     // kfs_used_times_[pkf->fid_] = 0;  // reset usage tracking if needed
// }
void VoxelMapper::increaseKeyframeTimesOfUse(
        const std::shared_ptr<VoxelKeyframe>& pkf,
        int times)
 {
     pkf->remaining_times_of_use_ += times;
 }

void VoxelMapper::writeKeyframeUsedTimes(
    const std::filesystem::path& dir,
    const std::string& suffix)
{
    // Ensure the output directory exists
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(dir);

    // Build the full path and open for appending
    std::filesystem::path result_path =
        dir / ("keyframe_used_times" + suffix + ".txt");
    std::ofstream out_stream(result_path, std::ios::app);
    if (!out_stream.is_open()) {
        throw std::runtime_error(
            "Cannot open keyframe‐used‐times file at " + result_path.string());
    }

    // Header line (mirrors GaussianMapper)
    out_stream << "##[Voxel Mapper]Iteration "
               << getIteration()
               << " keyframe id, used times, remaining times:\n";

    // For each entry in kfs_used_times_, look up the remaining‐use count
    // in scene_->keyframes(). If a keyframe has been culled, its remaining
    // is reported as zero.
    for (const auto& [kf_id, used_count] : kfs_used_times_) {
        int remaining = 0;
        auto it = scene_->keyframes().find(kf_id);
        if (it != scene_->keyframes().end()) {
            remaining = it->second->remaining_times_of_use_;
        }
        out_stream << kf_id << ' '
                   << used_count << ' '
                   << remaining << '\n';
    }
    out_stream << "##=========================================\n";
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
    const std::string&           name_suffix)
{
    // std::cout << "[DEBUG] Starting renderAndRecordKeyframe for frame "
    //           << pkf->fid_ << '\n';

    /* ------------------------------------------------ 1. camera  */
    sv::MiniCam cam = pkf->toMiniCam();
    cam.c2w = cam.c2w.contiguous().to(mDevice);   // make sure contiguous + on CUDA
    cam.w2c = cam.w2c.contiguous().to(mDevice);
    /* ------------------------------------------------ 2. GT → NumPy  */
    torch::Tensor chw_u8 = pkf->original_image_    // (3,H,W) in [0,1]
                            .mul(255.0f)
                            .clamp(0.0f, 255.0f)
                            .to(torch::kUInt8)            // <--- cast to U8
                            .cpu()
                            .contiguous();
    torch::Tensor hwc_u8 = chw_u8.permute({1,2,0}).contiguous();  // (H,W,3)
    // convert CHW→HWC uint8 numpy without any CUDA involvement
    py::array rgb_numpy = sv::tensorToNumpyRGB(hwc_u8);
    /* ------------------------------------------------ 3. render   */
    auto t0 = std::chrono::steady_clock::now();
    auto render_pkg = voxel_model_->render(cam, rgb_numpy, "");
    auto t1 = std::chrono::steady_clock::now();
    render_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();

    if (render_pkg.empty() || !render_pkg.count("rgb") || !render_pkg.at("rgb").defined()) {
        std::cout << "render pkg empty" << std::endl;
        return;
    }
    torch::Tensor rendered_image = render_pkg.at("rgb").to(mDevice);          // (1,3,H,W)

    torch::Tensor masked_image = rendered_image * undistort_mask_[pkf->camera_id_];
    masked_image = masked_image.squeeze(0);   
    auto gt_image = pkf->original_image_;

    dssim = loss_utils::ssim(masked_image, gt_image, device_type_).item().toFloat();
    psnr = loss_utils::psnr(masked_image, gt_image).item().toFloat();

    recordKeyframeRendered(masked_image, gt_image, pkf->fid_, result_img_dir, result_gt_dir, result_loss_dir, name_suffix);    
 }

void VoxelMapper::renderAndRecordAllKeyframes(const std::string& name_suffix)
{
    // Create result directory with current iteration number and suffix
    std::filesystem::path result_dir = result_dir_ / (std::to_string(getIteration()) + name_suffix);
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir);

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

    // Loop through all keyframes deterministically
    std::size_t nkfs = scene_->keyframes().size();
    auto kfit = scene_->keyframes().begin();
    float dssim, psnr;
    double render_time;
    for (std::size_t i = 0; i < nkfs; ++i) {
        renderAndRecordKeyframe((*kfit).second, dssim, psnr, render_time, image_dir, image_gt_dir, image_loss_dir);
        out_time << (*kfit).first << " " << std::fixed << std::setprecision(8) << render_time << std::endl;

        out_dssim   << (*kfit).first << " " << std::fixed << std::setprecision(10) << dssim   << std::endl;
        out_psnr    << (*kfit).first << " " << std::fixed << std::setprecision(10) << psnr    << std::endl;

        ++kfit;
    }
}

/* --- Optional placeholders so the call-sites compile ------------------- */
void VoxelMapper::savePly(const std::filesystem::path&){ /* TODO when trainer supports export */ }
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

// cv::Mat VoxelMapper::renderFromPose(const Sophus::SE3f& pose,
//                                     int  width,
//                                     int  height,
//                                     bool main_vision)
// {
//     // ───── 0. Early-out exactly like Photo-SLAM ─────
//     if (!initial_mapped_ || getIteration() <= 0)
//         return cv::Mat(height, width, CV_32FC3,
//                        cv::Vec3f(0.f,0.f,0.f));

//     // ───── 1. Build a temporary VoxelKeyframe ─────
//     auto pkf = std::make_shared<VoxelKeyframe>();
//     pkf->zfar_  = z_far_;
//     pkf->znear_ = z_near_;
//     pkf->setPose(pose.unit_quaternion().cast<double>(),
//                  pose.translation().cast<double>());

//     const Camera& cam = scene_->cameras_.at(viewer_camera_id_);
//     pkf->setCameraParams(cam);
//     pkf->computeTransformTensors();               // fills full_proj_transform_

//     // MiniCam helper already implemented in VoxelKeyframe
//     sv::MiniCam miniCam = pkf->toMiniCam();
//     miniCam.c2w = miniCam.c2w.to(device_type_);
//     miniCam.w2c = miniCam.w2c.to(device_type_);

//     // ───── 2. Render inside the render-mutex ─────
//     torch::Tensor rgb;
//     {
//         std::unique_lock<std::mutex> lk(mutex_render_);
//         auto out = voxel_model_->render(miniCam, py::none(), "viewer");
//         if (out.empty() || out.find("rgb") == out.end())             // safety
//             return cv::Mat(height, width, CV_32FC3,
//                            cv::Vec3f(0.f,0.f,0.f));

//         rgb = out["rgb"].cpu().squeeze(0);        // (3,H,W)
//     }

//     // ───── 3. Apply undistort mask exactly like Photo-SLAM ─────
//     torch::Tensor mask = undistort_mask_.at(pkf->camera_id_).to(torch::kFloat32);
//     if (mask.dim()==2) mask = mask.unsqueeze(0);               // (1,H,W)
//     if (main_vision)
//         rgb = rgb * mask;           // viewer_main_undistort_mask_
//     else
//         rgb = rgb * mask;           // viewer_sub_undistort_mask_

//     // ───── 4. Torch → cv::Mat (float32, H×W×3) ─────
//     rgb = rgb.permute({1,2,0});                      // (H,W,3)
//     cv::Mat img(rgb.size(0), rgb.size(1), CV_32FC3,
//                 rgb.data_ptr<float>());
//     return img.clone();   // take ownership
// }

VoxelMapper::~VoxelMapper() {
    // Explicitly reset any Python or Torch objects that may call Python at destruction
    voxel_model_.reset();  // Deallocates all tensors and Python wrappers
    mpSLAM.reset();
}

 bool VoxelMapper::isKeepingTraining()
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     return keep_training_;
 }

// void VoxelMapper::setKeepTraining(bool v) {
//     keep_training_ = v;
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

int VoxelMapper::newKeyframeTimesOfUse()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return new_keyframe_times_of_use_;
}

int VoxelMapper::densifyInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.densification_interval_;
}

float VoxelMapper::lambdaDssim()
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     return opt_params_.lambda_dssim_;
 }
