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

inline void saveDebugImage(torch::Tensor tensor, const std::string& path) {
    // Make sure it's on CPU, float32
    tensor = tensor.detach().to(torch::kCPU);
    if (tensor.dtype() != torch::kUInt8) {
        tensor = tensor.clamp(0, 1).mul(255).to(torch::kUInt8);
    }

    // Expect shape (3,H,W) or (1,3,H,W)
    if (tensor.dim() == 4) tensor = tensor.squeeze(0);
    if (tensor.dim() != 3 || tensor.size(0) != 3) {
        std::cerr << "[ERROR] Unexpected tensor shape: " << tensor.sizes() << std::endl;
        return;
    }

    // Convert to (H,W,3)
    tensor = tensor.permute({1, 2, 0}).contiguous();  // now (H,W,3)

    int H = tensor.size(0), W = tensor.size(1);
    cv::Mat img(H, W, CV_8UC3);
    std::memcpy(img.data, tensor.data_ptr<uint8_t>(), H * W * 3);

    cv::Mat img_bgr;
    cv::cvtColor(img, img_bgr, cv::COLOR_RGB2BGR);
    cv::imwrite(path, img_bgr);
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

    /* Load every ORB-SLAM3 camera, convert to sv::Camera, pre–compute            */
    /* undistort maps / masks, register in scene_, and remember one viewer id.    */
    auto settings      = mpSLAM->getSettings();
    cv::Size slam_im_size = settings->newImSize();
    UndistortParams undistort_params(
        slam_im_size,
        settings->camera1DistortionCoef()    // [k1,k2,p1,p2] from SLAM
    );

    // Loop over all SLAM cameras and register them in scene_:
    auto vpCameras = mpSLAM->getAtlas()->GetAllCameras();
    if (vpCameras.empty()) {
        std::cerr << "[VoxelMapper] Error: no cameras in SLAM atlas!\n";
        std::exit(EXIT_FAILURE);
    }

    bool first_cam = true;
    for (auto* slam_cam : vpCameras)
    {
        int   cam_id = slam_cam->GetId();
        float SLAM_fx = slam_cam->getParameter(0);
        float SLAM_fy = slam_cam->getParameter(1);
        float SLAM_cx = slam_cam->getParameter(2);
        float SLAM_cy = slam_cam->getParameter(3);

        // Build an sv::Camera with SLAM’s original intrinsics (K):
        sv::Camera cam(
            static_cast<std::size_t>(slam_im_size.width),
            static_cast<std::size_t>(slam_im_size.height),
            SLAM_fx, SLAM_fy, SLAM_cx, SLAM_cy,
            cam_id
        );
        cam.setModelId(sv::Camera::PINHOLE);

        // Copy exactly Photo-SLAM’s distortion coeffs if mono or RGB-D:
        if (sensor_type_ == MONOCULAR || sensor_type_ == RGBD) {
            undistort_params.dist_coeff_.copyTo(cam.dist_coeff_);
        }

        // Build undistort/rectify maps.  Since our sv::Camera only has the no-arg version,
        // it will read cam.params_ (SLAM’s fx/fy/cx/cy) plus cam.dist_coeff_ internally:
        cam.initUndistortRectifyMapAndMask();

        // Store this camera in the scene_ for later lookup by keyframes:
        scene_->addCamera(cam);

        // Convert single-channel undistort_mask_ → (1,H,W) Tensor on device:
        cv::Mat mask_cv;
        cam.undistort_mask_.convertTo(mask_cv, CV_32F, 1.0f / 255.0f);
        torch::Tensor mask_t = torch::from_blob(
            mask_cv.data,
            { mask_cv.rows, mask_cv.cols, 1 },
            torch::kFloat32
        )
        .clone()
        .permute({2, 0, 1})   // → (1, H, W)
        .to(device_type_);
        undistort_mask_[cam_id] = mask_t;

        // Remember the first camera as “viewer”, just like Photo-SLAM:
        if (first_cam) {
            viewer_camera_id_     = cam_id;
            viewer_camera_id_set_ = true;
            first_cam = false;
        }

        std::cout << "[VoxelMapper] Registered SLAM camera id " << cam_id
                  << "  fx=" << SLAM_fx << "  fy=" << SLAM_fy
                  << "  (" << cam.width() << "×" << cam.height() << ")\n";
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
        // ── Check conditions for initial mapping ──
        if (hasMetInitialMappingConditions())
        {
            mpSLAM->getAtlas()->clearMappingOperation();

            // ---- A. pull sparse SLAM map -----------------------------------
            auto pMap = mpSLAM->getAtlas()->GetCurrentMap();
            std::vector<ORB_SLAM3::KeyFrame*>  vKFs;
            std::vector<ORB_SLAM3::MapPoint*>  vMPs;
            {
                std::unique_lock<std::mutex> lock_map(pMap->mMutexMapUpdate);
                vKFs = pMap->GetAllKeyFrames();
                vMPs = pMap->GetAllMapPoints();
                // A-1) cache every MapPoint (RGB 0-255)
                for (auto *pMP : vMPs)
                {
                    if (!pMP || pMP->isBad()) continue;
                    Point3D pt;
                    const auto pos   = pMP->GetWorldPos();
                    const auto color = pMP->GetColorRGB();
                    pt.xyz_   << pos.x(), pos.y(), pos.z();
                    pt.color_ << color(0), color(1), color(2);
                    scene_->cachePoint3D(pMP->mnId, pt);
                }
                // A-2) create VoxelKeyframes from each SLAM KeyFrame
                for (auto *pKF : vKFs)
                {
                    if (!pKF || pKF->isBad()) continue;
                    auto vkf = std::make_shared<VoxelKeyframe>(pKF->mnId, getIteration());
                    vkf->zfar_  = z_far_;
                    vkf->znear_ = z_near_;
                    // Pose
                    auto Tcw = pKF->GetPose();
                    vkf->setPose(
                        Tcw.unit_quaternion().cast<double>(),
                        Tcw.translation().cast<double>()
                    );
                    // (2) Grab SLAM intrinsics, build an sv::Camera, copy distortion
                    int   w  = mpSLAM->getSettings()->newImSize().width;
                    int   h  = mpSLAM->getSettings()->newImSize().height;
                    float fx = pKF->mpCamera->getParameter(0);
                    float fy = pKF->mpCamera->getParameter(1);
                    float cx = pKF->mpCamera->getParameter(2);
                    float cy = pKF->mpCamera->getParameter(3);

                    sv::Camera cam(
                        static_cast<std::size_t>(w),
                        static_cast<std::size_t>(h),
                        fx, fy, cx, cy,
                        pKF->mpCamera->GetId()
                    );
                    cam.initUndistortRectifyMapAndMask();
                    vkf->setCameraParams(cam);
                    // (3) Grab “left” image (mono or left stereo); undistort it
                    cv::Mat imgRGB = pKF->imgLeftRGB;
                    cv::Mat imgRGB_undistorted;
                    if (sensor_type_ == STEREO)
                    {
                        imgRGB_undistorted = imgRGB;
                    }
                    else
                    {
                        cam.undistortImage(imgRGB, imgRGB_undistorted);
                    }

                    vkf->original_image_ =
                        tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
                    vkf->img_tensor   = vkf->original_image_.to(mDevice);
                    vkf->img_filename_ = pKF->mNameFile;

                    // (5) Build the 1-channel mask tensor exactly as Photo-SLAM does
                    vkf->mask_tensor = torch::from_blob(
                            cam.undistort_mask_.data,
                            {cam.height(), cam.width(), 1},
                            torch::kFloat32)
                        .clone()
                        .permute({2,0,1})
                        .to(mDevice);

                    vkf->computeTransformTensors();

                    scene_->addKeyframe(vkf, &kfid_shuffled_);
                    increaseKeyframeTimesOfUse(vkf, newKeyframeTimesOfUse());
                }
            }   // mutex released

            // ---- B. scene normalization extent (mirror Photo-SLAM) ------------
            {
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
            }

            // ---- C. build initial voxel centers + colors  ----------------------
            {
                constexpr float kVoxelSz = 0.05f;
                std::set<std::array<float,3>> unique_ctr;
                for (auto *pMP : vMPs)
                {
                    if (!pMP || pMP->isBad()) continue;
                    Eigen::Vector3f p = pMP->GetWorldPos().cast<float>();
                    Eigen::Vector3f c = (p / kVoxelSz).array().round() * kVoxelSz;
                    unique_ctr.insert({c.x(), c.y(), c.z()});
                }

                std::vector<torch::Tensor> ctr_tensors;
                for (auto const& c : unique_ctr)
                    ctr_tensors.emplace_back(torch::tensor({c[0],c[1],c[2]}, torch::kFloat32));
                voxel_centers_ = torch::stack(ctr_tensors).to(mDevice);
                const int64_t N = voxel_centers_.size(0);

                // average RGB per voxel
                struct CellHash {
                    std::size_t operator()(const std::array<int,3>& k) const noexcept {
                        return ((k[0]*73856093) ^ (k[1]*19349663) ^ (k[2]*83492791));
                    }
                };
                std::unordered_map<std::array<int,3>, std::pair<Eigen::Vector3f,int>, CellHash> accum;
                for (auto *pMP : vMPs)
                {
                    if (!pMP || pMP->isBad()) continue;
                    Eigen::Vector3f p = pMP->GetWorldPos().cast<float>();
                    Eigen::Vector3i c = (p / kVoxelSz).array().round().cast<int>();
                    auto key = std::array<int,3>{c.x(), c.y(), c.z()};
                    Eigen::Vector3f rgb = pMP->GetColorRGB().cast<float>() / 255.f;
                    auto &slot = accum[key];
                    slot.first += rgb;
                    slot.second += 1;
                }

                torch::Tensor colours_rgb = torch::empty({N,3}, torch::kFloat32);
                for (int64_t i = 0; i < N; ++i)
                {
                    auto key = std::array<int,3>{
                        int(voxel_centers_[i][0].item<float>() / kVoxelSz),
                        int(voxel_centers_[i][1].item<float>() / kVoxelSz),
                        int(voxel_centers_[i][2].item<float>() / kVoxelSz)};
                    Eigen::Vector3f rgb;
                    auto it = accum.find(key);
                    if (it == accum.end())
                        rgb = Eigen::Vector3f::Constant(0.5f);
                    else
                        rgb = it->second.first / float(it->second.second);
                    colours_rgb[i] = torch::tensor({rgb.x(), rgb.y(), rgb.z()});
                }
                colours_rgb = colours_rgb.to(mDevice);
                torch::Tensor sh0 = colours_rgb * 0.282095f;  // SH‐DC

                // ---- D. create voxel model & trainer setup --------------------------
                {
                    std::unique_lock<std::mutex> lock_render(mutex_render_);
                    scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
                    voxel_model_->createFromPcd(scene_->cached_point_cloud_, scene_->cameras_extent_);
                    std::unique_lock<std::mutex> lock(mutex_settings_);
                    voxel_model_->trainingSetup(opt_params_);

                    next_subdiv_iter_   = opt_params_.subdiv_from_;
                    next_prune_iter_    = opt_params_.prune_from_;
                    next_opacity_reset_ = opt_params_.densification_interval_;
                }

                // one warm‐up optimization step
                trainForOneIteration();

                initial_mapped_ = true;
                break;  // leave initial mapping loop
            }
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

{
    py::gil_scoped_acquire gil;
    auto mod = py::module_::import("python_svraster_bridge.renderer_wrapper");
    mod.attr("stop_watchdog")();
}

}

bool VoxelMapper::hasMetInitialMappingConditions() {
    return !mpSLAM->isShutDown() &&
           mpSLAM->GetNumKeyframes() >= min_num_initial_map_kfs_ &&
           mpSLAM->getAtlas()->hasMappingOperation();
}

bool VoxelMapper::hasMetIncrementalMappingConditions() {
    return !mpSLAM->isShutDown() &&
           mpSLAM->getAtlas()->hasMappingOperation();
}

void VoxelMapper::generateKfidRandomShuffle()
{
    // If no keyframes in the scene yet, nothing to do.
    if (scene_->keyframes().empty())
        return;

    // Number of keyframes
    std::size_t nkfs = scene_->keyframes().size();

    // Resize shuffle array to [0,1,…,nkfs−1]
    kfid_shuffle_.resize(nkfs);
    std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);

    // Shuffle with a local random engine (Photo-SLAM uses rd_())
    std::mt19937 g(rd_());
    std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

    // Mark as shuffled and reset the index
    kfid_shuffled_    = true;
    kfid_shuffle_idx_ = 0;
}

std::shared_ptr<VoxelKeyframe> VoxelMapper::useOneRandomSlidingWindowKeyframe()
{
    // If no keyframes, return nullptr
    if (scene_->keyframes().empty())
        return nullptr;

    // If not shuffled yet, build shuffle
    if (!kfid_shuffled_)
        generateKfidRandomShuffle();

    // Number of keyframes
    int nkfs = static_cast<int>(kfid_shuffle_.size());
    if (nkfs == 0)
        return nullptr;

    std::shared_ptr<VoxelKeyframe> viewpoint_cam = nullptr;

    // Remember where we started
    int start_shuffle_idx = kfid_shuffle_idx_;

    // Loop until we find one with remaining_times_of_use_ > 0
    do {
        // Advance the shuffled index (with wrap-around)
        kfid_shuffle_idx_ = (kfid_shuffle_idx_ + 1) % nkfs;

        // If we've wrapped back to the starting index, everyone is "out of uses" → give each +1 use
        if (kfid_shuffle_idx_ == start_shuffle_idx) {
            for (auto &kv : scene_->keyframes()) {
                auto &kf_ptr = kv.second;
                increaseKeyframeTimesOfUse(kf_ptr, 1);
            }
        }

        // Pick the “random position” from the shuffle array
        int shuffled_pos = kfid_shuffle_[kfid_shuffle_idx_];

        // Advance an iterator from scene_->keyframes().begin() by shuffled_pos
        auto it = scene_->keyframes().begin();
        std::advance(it, shuffled_pos);
        viewpoint_cam = it->second;

        // Keep looping until we find one with remaining_times_of_use_ > 0
    } while (viewpoint_cam->remaining_times_of_use_ <= 0);

    // Book-keeping: increment “used times” for this keyframe
    unsigned long fid = viewpoint_cam->fid_;
    if (kfs_used_times_.find(fid) == kfs_used_times_.end()) {
        kfs_used_times_[fid] = 1;
    } else {
        ++kfs_used_times_[fid];
    }

    // Decrement its remaining uses
    --(viewpoint_cam->remaining_times_of_use_);

    return viewpoint_cam;
}

void VoxelMapper::cullKeyframes()
{
    // Ask ORB-SLAM3 which keyframe IDs are still “live”
    std::unordered_set<unsigned long> live_kfids =
        mpSLAM->getAtlas()->GetCurrentKeyFrameIds();

    // Collect any IDs in scene_->keyframes() that are NOT in live_kfids
    std::vector<unsigned long> kfids_to_erase;
    kfids_to_erase.reserve(scene_->keyframes().size());

    for (auto const &kv : scene_->keyframes()) {
        unsigned long this_id = kv.first;
        if (live_kfids.find(this_id) == live_kfids.end()) {
            kfids_to_erase.push_back(this_id);
        }
    }

    // Erase them from the scene
    for (auto id : kfids_to_erase) {
        scene_->keyframes().erase(id);
    }

    // If we removed any, we must re-shuffle next time
    if (!kfids_to_erase.empty()) {
        kfid_shuffled_ = false;
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

                for (auto &kf_tuple : associated_kfs) {
                    // Unpack the keyframe tuple
                    unsigned long kfid = std::get<0>(kf_tuple);
                    auto &pose        = std::get<2>(kf_tuple);

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
                        handleNewKeyframe(kf_tuple);
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
                for (auto &kf_tuple : associated_kfs) {
                    unsigned long kfid = std::get<0>(kf_tuple);
                    auto &pose        = std::get<2>(kf_tuple);

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
                        handleNewKeyframe(kf_tuple);
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
        Sophus::SE3f,     // 2: Tcw
        cv::Mat,          // 3: RGB image
        bool,             // 4: loop‐closure flag (unused here)
        cv::Mat,          // 5: auxiliary (unused here)
        std::vector<float>, // 6: keypoint pixel coords (unused here)
        std::vector<float>, // 7: keypoint local coords (unused here)
        std::string       // 8: image filename (relative or absolute)
    > &kf_tuple)
{
    // ─── 1) Unpack the tuple ─────────────────────────────────────────────────
    unsigned long kfid       = std::get<0>(kf_tuple);
    unsigned long camid      = std::get<1>(kf_tuple);
    const Sophus::SE3f &Tcw  = std::get<2>(kf_tuple);
    const cv::Mat   &imgRGB  = std::get<3>(kf_tuple);
    const std::string &img_file = std::get<8>(kf_tuple);
    // (We ignore fields 4, 5, 6, 7 since VoxelMapper doesn’t use them.)

    // ─── 2) Create a new VoxelKeyframe, exactly like Photo-SLAM’s Gaussian case ─
    auto pkf = std::make_shared<VoxelKeyframe>(kfid, getIteration());
    pkf->zfar_  = z_far_;
    pkf->znear_ = z_near_;

    // ─── 3) Set its pose ───────────────────────────────────────────────────────
    pkf->setPose(
        Tcw.unit_quaternion().cast<double>(),
        Tcw.translation().cast<double>()
    );

    // ─── 4) Resolve the actual image path on disk ──────────────────────────────
    if (std::filesystem::path(img_file).is_absolute()) {
        pkf->img_path_ = img_file;
    } else {
        pkf->img_path_ = (mSeqDir / img_file).string();
    }

    // ─── 5) Fetch the corresponding sv::Camera from scene_->cameras() ───────────
    //     (If camid is not found, this will throw out_of_range.)
    sv::Camera &camera = scene_->cameras_.at(camid);
    pkf->setCameraParams(camera);

    // ─── 6) Undistort the RGB image (mono vs. stereo‐left) ───────────────────────
    cv::Mat imgRGB_undist;
    if (sensor_type_ == STEREO) {
        // For stereo, we assume “left” view is already rectified
        imgRGB_undist = imgRGB.clone();
    } else {
        camera.undistortImage(imgRGB, imgRGB_undist);
    }

    // ─── 7) Convert undistorted image → (3,H,W) float32 Tensor on device ───────
    pkf->original_image_ =
        tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undist, device_type_);
    pkf->img_tensor = pkf->original_image_.to(mDevice);

    // ─── 8) Store the filename for record‐keeping ───────────────────────────────
    pkf->img_filename_ = img_file;

    // ─── 9) Copy the 1‐channel undistort mask (already computed in ctor) ────────
    //     We precomputed undistort_mask_[camid] in the constructor.  Just reuse it.
    pkf->mask_tensor = undistort_mask_.at(camid);

    // ─── 10) Initialize “remaining_times_of_use_” to zero (default) ─────────────
    //      (We do NOT assign newKeyframeTimesOfUse() directly; instead, see step 12.)

    // ─── 11) Finalize any internal transforms (world→camera, etc.) ──────────────
    pkf->computeTransformTensors();

    // ─── 12) Insert into the scene and mark shuffle dirty ──────────────────────
    scene_->addKeyframe(pkf, &kfid_shuffled_);

    // ─── 13) Give exactly “newKeyframeTimesOfUse()” uses, just like Photo-SLAM ─
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
    // int image_height, image_width;
    // torch::Tensor gt_image, mask;

    // img_h = viewpoint_cam->image_height_;
    // img_w = viewpoint_cam->image_width_;
    // gt_image = viewpoint_cam->original_image_.cuda();
    // mask = undistort_mask_[viewpoint_cam->camera_id_];

    torch::Tensor gt_image_3d = viewpoint_cam->img_tensor;   // (3,H,W)
    torch::Tensor mask_3d     = viewpoint_cam->mask_tensor;  // (1,H,W)
    // torch::Tensor mask_3d     = undistort_mask_[viewpoint_cam->camera_id_];
    if (!gt_image_3d.defined() || !mask_3d.defined()) {
        return;
    }
    // add batch dim → (1,3,H,W) and (1,1,H,W)
    torch::Tensor gt_image = gt_image_3d.unsqueeze(0);
    // torch::Tensor mask     = mask_3d.unsqueeze(0);
    if (mask_3d.dim() == 2) mask_3d = mask_3d.unsqueeze(0); // → (1,H,W)
    mask_3d = mask_3d.unsqueeze(1); 
    torch::Tensor mask = mask_3d;

    const int img_h = gt_image.size(2);
    const int img_w = gt_image.size(3);

    // 4) grow SH degree every 1000 iterations (locked during render)
    std::unique_lock<std::mutex> lock_render(mutex_render_);
    
    if (getIteration() % 1000 == 0 && default_sh_ < model_params_.sh_degree_)
        default_sh_ += 1;
    voxel_model_->setShDegree(default_sh_);

    // 5) build a MiniCam out of this keyframe
    const sv::Camera& cinfo = viewpoint_cam->getCamera();
    sv::MiniCam cam = sv::MiniCam::fromIntrinsics(
        cinfo.fx(), cinfo.fy(), cinfo.cx(), cinfo.cy(),
        img_w, img_h,
        static_cast<int>(viewpoint_cam->fid_)
    );
    // set c2w, w2c
    {
        Eigen::Matrix4f Tcw = viewpoint_cam->Tcw.matrix();
        Eigen::Matrix4f c2w = Tcw.inverse();
        cam.c2w = torch::from_blob(c2w.data(), {4,4}, torch::kFloat32).clone().to(mDevice);
        cam.w2c = torch::from_blob(Tcw.data(),  {4,4}, torch::kFloat32).clone().to(mDevice);
    }

    // 6) update all learning rates
    if (mpSLAM) {
        int used_times = kfs_used_times_[viewpoint_cam->fid_];
        int step = (used_times <= opt_params_.geo_lr_max_steps_)
                       ? used_times
                       : opt_params_.geo_lr_max_steps_;
        float geo_lr = voxel_model_->updateLearningRate(step);
        voxel_model_->setGeoLearningRate(geo_lr);
    } else {
        voxel_model_->updateLearningRate(iteration_);
    }
    // SH-DC and higher-SH rates remain constant:
    voxel_model_->setSh0LearningRate(opt_params_.sh0_lr_);
    voxel_model_->setShsLearningRate(opt_params_.shs_lr_);

    py::array rgb_numpy = tensorToNumpyRGB_F32(viewpoint_cam->img_tensor.cpu());
    saveDebugImage(viewpoint_cam->img_tensor, (result_dir_ / "debug_gt.jpg").string());

    auto render_pkg = voxel_model_->render(cam, rgb_numpy, "");
    lock_render.unlock();
    if (render_pkg.empty() || !render_pkg.count("rgb") || !render_pkg.at("rgb").defined()) {
        return;
    }
    // Keep the batch dimension: rendered_image is now (1,3,H,W)
    torch::Tensor rendered_image = render_pkg["rgb"].to(mDevice);
    saveDebugImage(rendered_image.squeeze(0), (result_dir_ / "debug_rendered.jpg").string());

    // 8) compute masked photometric loss
    torch::Tensor Ll1, loss;
    {
        py::gil_scoped_release no_gil;
        // mask out black/invalid pixels
        torch::Tensor masked_pred = rendered_image * mask;   // (1,3,H,W)
        // torch::Tensor masked_gt   = gt_image        * mask;   // (1,3,H,W)
        std::cout << "[DEBUG] mask sum = " << mask.sum().item<float>() << std::endl;
        std::cout << "[DEBUG] Training on frame: " << viewpoint_cam->fid_ << "\n";
        std::cout << "[DEBUG] mask non-zero count = " << mask.count_nonzero().item<int>() << "\n";
        std::cout << "[DEBUG] gt min/max = " << gt_image.min().item<float>() << " / " << gt_image.max().item<float>() << "\n";
        std::cout << "[DEBUG] pred min/max = " << rendered_image.min().item<float>() << " / " << rendered_image.max().item<float>() << "\n";
        saveDebugImage(mask.squeeze(0), (result_dir_ / "debug_mask.jpg").string());
        
        Ll1 = loss_utils::l1_loss(masked_pred, gt_image);
        float lambda_dssim = lambdaDssim();
        loss = (1.0f - lambda_dssim) * Ll1
             + lambda_dssim * (1.0f - loss_utils::ssim(masked_pred, gt_image, mDevice.type()));
        loss.backward();
        if (mDevice == torch::kCUDA) {
            torch::cuda::synchronize();
        }

        // === DEBUG GRADIENTS ===
        auto params = voxel_model_->parameters();  // or voxel_model_->getTrainableParams()
        for (size_t i = 0; i < params.size(); ++i) {
            auto& p = params[i];
            if (p.grad().defined()) {
                std::cout << "[DEBUG] Param " << i
                        << ": grad max = " << p.grad().abs().max().item<float>()
                        << ", mean = " << p.grad().abs().mean().item<float>() << "\n";
            } else {
                std::cout << "[DEBUG] Param " << i << ": grad undefined\n";
            }
        }
    }

    // 9) optimizer step (only geometry group is updated by updateLearningRate)
    {
        py::gil_scoped_release no_gil;
        torch::NoGradGuard no_grad;
        voxel_model_->optimizer()->step();
        voxel_model_->optimizer()->zero_grad();
    }

    {
        // rebuild rgb_numpy (or reuse the one from above)
        py::array rgb_numpy2 = tensorToNumpyRGB_F32(viewpoint_cam->img_tensor.cpu());
        // re-lock for thread safety during render
        std::unique_lock<std::mutex> lock2(mutex_render_);
        auto render_pkg2 = voxel_model_->render(cam, rgb_numpy2, "");
        lock2.unlock();
        if (!render_pkg2.empty() && render_pkg2.count("rgb") && render_pkg2.at("rgb").defined()) {
            torch::Tensor post_render = render_pkg2["rgb"].to(mDevice).squeeze(0); // (3,H,W)
            saveDebugImage(post_render, (result_dir_ / "debug_post_update.jpg").string());
        }
    }

    // 10) accumulate subdivision gradients into subdiv_meta_
    {
        torch::NoGradGuard no_grad;
        // get logging gradient from subdiv_p_ → use new method name getSubdivPriorityGrad()
        torch::Tensor grad_p = voxel_model_->getSubdivPriorityGrad();
        // clamp subdiv_meta_ ← getTensor("subdiv_meta")
        torch::Tensor subdiv_meta = voxel_model_->getTensor("subdiv_meta");
        subdiv_meta = torch::clamp(subdiv_meta + grad_p * opt_params_.meta_accum_lr_, 0.0f, 1.0f);
        voxel_model_->setSubdivMeta(subdiv_meta);

        // scatter‐add full gradient array back into "grad buffer"
        torch::Tensor idx_all = torch::arange(grad_p.numel(), grad_p.options().dtype(torch::kLong));
        voxel_model_->accumulateSubdivGradients(idx_all, grad_p);
    }

    // 11) logging + densification/pruning + EMA + optional save
    {
        torch::NoGradGuard no_grad;

        // update EMA
        static float ema_loss_for_log = loss.item<float>();
        ema_loss_for_log = 0.4f * loss.item<float>() + 0.6f * ema_loss_for_log;

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
        // if (iteration_ < opt_params_.densify_until_iter_) {
        //     // accumulate per‐view gradient/radius stats
        //     torch::Tensor view_pts = render_pkg.at("viewspace_pts");      // if available
        //     torch::Tensor vis_mask = render_pkg.at("visibility_filter");  // if available

        //     if (view_pts.defined() && vis_mask.defined()) {
        //         voxel_model_->addDensificationStats(view_pts, vis_mask);
        //     }

        //     // split/prune at specified intervals
        //     if (iteration_ > opt_params_.densify_from_iter_ &&
        //         (iteration_ % opt_params_.densification_interval_ == 0)) {
        //         int size_threshold = (iteration_ > opt_params_.opacity_reset_interval_) ? 20 : 0;
        //         voxel_model_->densifyAndPrune(
        //             opt_params_.densify_grad_threshold_,
        //             0.005f,
        //             scene_->cameras_extent_,
        //             size_threshold
        //         );
        //     }

        //     // reset opacity periodically (or at first white_background_ iteration)
        //     if ((iteration_ % opt_params_.opacity_reset_interval_ == 0) ||
        //         (model_params_.white_background_ && iteration_ == opt_params_.densify_from_iter_))
        //     {
        //         voxel_model_->resetOpacity();
        //     }
        // }

        // optionally record keyframe visuals / save PLY, etc., exactly when
        // iteration_ % all_keyframes_record_interval_ == 0 (if you want)
        if (all_keyframes_record_interval_ &&
            iteration_ % all_keyframes_record_interval_ == 0)
        {
            renderAndRecordAllKeyframes();
            // savePly(result_dir_ / std::to_string(iteration_) / "ply");
        }
        if (training_report_interval_ && iteration_ % training_report_interval_ == 0) {
            std::cout << "[TRAIN] iter " << iteration_
                    << "  L1: "    << Ll1.item<float>()
                    << "  loss: "  << loss.item<float>()
                    << "  ema: "   << ema_loss_for_log << '\n';
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
inline void VoxelMapper::increaseKeyframeTimesOfUse(
        const std::shared_ptr<VoxelKeyframe>& kf,
        int n)
{
    if (!kf) return;
    kf->remaining_times_of_use_ += n;
    if (kf->remaining_times_of_use_ < 0)
        kf->remaining_times_of_use_ = 0;
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
    // If there’s a leading batch dimension, remove it
    torch::Tensor rend = (rendered.dim() == 4) ? rendered.squeeze(0) : rendered;       // (3,H,W)
    torch::Tensor gt   = (ground_truth.dim() == 4) ? ground_truth.squeeze(0) : ground_truth; // (3,H,W)

    if (record_rendered_image_) {
        auto img_cv = tensor_utils::torchTensor2CvMat_Float32(rend);
        cv::cvtColor(img_cv, img_cv, cv::COLOR_RGB2BGR);
        img_cv.convertTo(img_cv, CV_8UC3, 255.0f);
        std::string fname = std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + ".jpg";
        cv::imwrite((result_img_dir / fname).string(), img_cv);
    }

    if (record_ground_truth_image_) {
        auto gt_cv = tensor_utils::torchTensor2CvMat_Float32(gt);
        cv::cvtColor(gt_cv, gt_cv, cv::COLOR_RGB2BGR);
        gt_cv.convertTo(gt_cv, CV_8UC3, 255.0f);
        std::string fname = std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_gt.jpg";
        cv::imwrite((result_gt_dir / fname).string(), gt_cv);
    }

    if (record_loss_image_) {
        torch::Tensor loss_tensor = torch::abs(rend - gt); // (3,H,W)
        auto loss_cv = tensor_utils::torchTensor2CvMat_Float32(loss_tensor);
        cv::cvtColor(loss_cv, loss_cv, cv::COLOR_RGB2BGR);
        loss_cv.convertTo(loss_cv, CV_8UC3, 255.0f);
        std::string fname = std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_loss.jpg";
        cv::imwrite((result_loss_dir / fname).string(), loss_cv);
    }
}

void VoxelMapper::renderAndRecordKeyframe(
    std::shared_ptr<VoxelKeyframe> pkf,
    float&                         dssim,
    float&                         psnr,
    double&                        render_ms,
    const std::filesystem::path&   img_dir,
    const std::filesystem::path&   gt_dir,
    const std::filesystem::path&   loss_dir,
    const std::string&             suffix)
{
    std::cout << "[DEBUG] Starting renderAndRecordKeyframe for frame " << pkf->fid_ << "\n";

    // --- 1. Setup camera intrinsics
    const auto &C = pkf->getCamera();
    float fx = C.fx(), fy = C.fy(), cx = C.cx(), cy = C.cy();
    int H = pkf->original_image_.size(1);
    int W = pkf->original_image_.size(2);

    // --- 2. Setup MiniCam
    sv::MiniCam cam = sv::MiniCam::fromIntrinsics(fx, fy, cx, cy, W, H, int(pkf->fid_));
    Eigen::Matrix4f Tcw = pkf->Tcw.matrix();
    Eigen::Matrix4f c2w = Tcw.inverse();
    cam.c2w = torch::from_blob(c2w.data(), {4,4}, torch::kFloat32).clone().to(mDevice);
    cam.w2c = torch::from_blob(Tcw.data(),  {4,4}, torch::kFloat32).clone().to(mDevice);
    cam.fx = fx; cam.fy = fy; cam.cx = cx; cam.cy = cy;
    cam.width = W; cam.height = H;
    cam.near = 0.5f * W / fx;
    cam.frame_id = *reinterpret_cast<const int*>(&cam.near); // dummy trick

    // --- 3. Convert original_image_ to numpy (ground truth image)
    // torch::Tensor gt_image = pkf->original_image_; // (3,H,W)
    // py::array rgb_np = tensorToNumpyRGB_F32(
    //     gt_image.permute({1,2,0}).contiguous().to(torch::kFloat32)
    // );
    torch::Tensor gt_image = pkf->original_image_;  // (3,H,W) float in [0,1]
    torch::Tensor chw_u8 = (gt_image * 255.0f)
        .clamp(0, 255)
        .to(torch::kU8);                              // (3,H,W) uint8
    torch::Tensor hwc_u8 = chw_u8
        .permute({1, 2, 0})
        .contiguous();                                // (H,W,3) uint8
    py::array_t<uint8_t> rgb_np = sv::tensorToNumpyRGB(hwc_u8);  // safe H×W×3 layout

    // --- 4. Render from voxel model
    auto t0 = std::chrono::steady_clock::now();
    auto out = voxel_model_->render(cam, rgb_np, "");
    auto t1 = std::chrono::steady_clock::now();
    render_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!out.count("rgb") || !out.at("rgb").defined()) {
        std::cerr << "[ERROR] Render output missing 'rgb'\n";
        return;
    }

    // --- 5. Apply mask
    torch::Tensor pred = out.at("rgb").to(torch::kFloat32).to(mDevice); // (1,3,H,W)
    std::cout << "[DEBUG] pred dtype = " << pred.dtype() << std::endl;
    torch::Tensor mask = pkf->mask_tensor.unsqueeze(0).to(mDevice); // (1,1,H,W)
    torch::Tensor masked = pred * mask;
    std::cout << "[DEBUG] masked dtype = " << masked.dtype() << std::endl;

    // --- 6. DSSIM and PSNR
    torch::Tensor gt_batch = gt_image.unsqueeze(0).to(mDevice); // (1,3,H,W)
    dssim = 1.0f - loss_utils::ssim(masked, gt_batch, mDevice.type()).item<float>();
    psnr  = loss_utils::psnr(masked, gt_batch).item<float>();

    // --- 7. Save rendered image
    if (record_rendered_image_) {
        torch::Tensor u8 = masked
            .squeeze(0)
            .clamp(0,1)
            .mul(255.0f)
            .to(torch::kU8)
            .permute({1,2,0})
            .contiguous();

        torch::Tensor u8_cloned = u8.clone();
        cv::Mat cvimg(H, W, CV_8UC3);
        std::memcpy(cvimg.data,
            u8_cloned.data_ptr<uint8_t>(),
            static_cast<size_t>(H) * W * 3);
        cv::Mat cvimg_bgr;
        cv::cvtColor(cvimg, cvimg_bgr, cv::COLOR_RGB2BGR);

        std::string fn = std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + suffix + ".jpg";
        std::filesystem::path save_path = img_dir / fn;
        if (!cv::imwrite(save_path.string(), cvimg_bgr)) {
            std::cerr << "[ERROR] Failed to write image: " << save_path << "\n";
        } else {
            std::cout << "[DEBUG] Saved rendered image to: " << save_path << "\n";
        }
    }

    // --- 8. Record metrics
    recordKeyframeRendered(
        masked,
        gt_batch,
        pkf->fid_,
        img_dir, gt_dir, loss_dir,
        suffix
    );
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

    float dssim = 0.0f, psnr = 0.0f;
    double render_time = 0.0;

    for (std::size_t i = 0; i < nkfs; ++i, ++kfit) {
        auto kfid = kfit->first;
        auto pkf  = kfit->second;

        // Skip keyframes that aren't fully valid
        if (!pkf || !pkf->img_tensor.defined() || !pkf->mask_tensor.defined() || pkf->img_tensor.numel() == 0) {
            std::cerr << "[VoxelMapper] Skipping invalid keyframe #" << kfid << '\n';
            continue;
        }

        renderAndRecordKeyframe(pkf, dssim, psnr, render_time, image_dir, image_gt_dir, image_loss_dir, name_suffix);

        // Log metrics
        out_time  << kfid << " " << std::fixed << std::setprecision(8) << render_time << '\n';
        out_dssim << kfid << " " << std::fixed << std::setprecision(10) << dssim << '\n';
        out_psnr  << kfid << " " << std::fixed << std::setprecision(10) << psnr  << '\n';
    }

    out_time.close();
    out_dssim.close();
    out_psnr.close();
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

// cv::Mat VoxelMapper::renderFromPose(
//     const Sophus::SE3f &Tcw,
//     const int width,
//     const int height,
//     const bool main_vision)
// {
//     if (!initial_mapped_ || getIteration() <= 0)
//         return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));

//     // ──────── build a temporary keyframe wrapper ────────
//     VoxelKeyframe tempKf;
//     tempKf.zfar_  = z_far_;
//     tempKf.znear_ = z_near_;
//     // set pose from the passed‐in SE3
//     tempKf.setPose(
//         Tcw.unit_quaternion().cast<double>(),
//         Tcw.translation().cast<double>()
//     );

//     // ──────── fetch the “viewer” camera from scene_ ────────
//     //    (exactly like Photo-SLAM’s renderFromPose)
//     const sv::Camera& camInfo = scene_->cameras_.at(viewer_camera_id_);
//     tempKf.setCameraParams(camInfo);
//     tempKf.computeTransformTensors();

//     // ────── now convert that keyframe into a MiniCam ──────
//     const float fx = camInfo.fx();
//     const float fy = camInfo.fy();
//     const float cx = camInfo.cx();
//     const float cy = camInfo.cy();

//     const int w = (width  > 0 ? width  : 1);
//     const int h = (height > 0 ? height : 1);

//     sv::MiniCam miniCam = sv::MiniCam::fromIntrinsics(
//         fx, fy, cx, cy,
//         w, h,
//         /*frame_id=*/static_cast<int>(viewer_camera_id_)
//     );
//     // set the pose tensors
//     Eigen::Matrix4f Tcw_mat = Tcw.matrix();
//     Eigen::Matrix4f c2w_mat = Tcw.inverse().matrix();
//     miniCam.c2w = torch::from_blob(c2w_mat.data(), {4,4}, torch::kFloat32)
//                      .clone()
//                      .to(mDevice);
//     miniCam.w2c = torch::from_blob(Tcw_mat.data(), {4,4}, torch::kFloat32)
//                      .clone()
//                      .to(mDevice);

//     // ────── do the actual render ──────
//     cv::Mat rendered_image;
//     {
//         std::unique_lock<std::mutex> lock_render(mutex_render_);
//         try {
//             auto out_map = voxel_model_->render(miniCam, py::none(), "viewer");
//             torch::Tensor rendered_tensor = out_map["rgb"].cpu().squeeze(0);  // (3,H,W)
//             rendered_tensor = rendered_tensor.permute({1,2,0});               // (H,W,3)

//             // wrap into an OpenCV Mat (float32)
//             cv::Mat image(h, w, CV_32FC3, rendered_tensor.data_ptr<float>());
//             rendered_image = image.clone();  // take ownership
//         }
//         catch (const py::error_already_set& e) {
//             std::cerr << "[renderFromPose] Python exception:\n" 
//                       << e.what() << std::endl;
//             return cv::Mat(h, w, CV_32FC3, cv::Vec3f(0,0,0));
//         }
//         catch (const std::exception& e) {
//             std::cerr << "[VoxelMapper::renderFromPose] Rendering failed: " 
//                       << e.what() << std::endl;
//             return cv::Mat(h, w, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
//         }
//     }

//     return rendered_image;
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