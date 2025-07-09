#include "include_voxel/voxel_mapper.h"

namespace py = pybind11;
std::ofstream loss_log_;
std::ofstream loss_l1_log_;
std::ofstream loss_ssim_log_;
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

// helper to show a 1×3×H×W float tensor in [0,1]
inline void showTensor(const torch::Tensor &t, const std::string &winName)
{
    // squeeze batch, CHW→HWC, [0,1]→[0,255], uint8
    auto img = t.squeeze(0)                      
                 .permute({1,2,0})               
                 .mul(255.0f).clamp(0.0f,255.0f)
                 .to(torch::kUInt8)
                 .contiguous()
                 .cpu();

    int H = img.size(0), W = img.size(1);
    // wrap the data as an RGB mat
    cv::Mat mat_rgb(H, W, CV_8UC3, img.data_ptr());

    // convert to BGR for OpenCV
    cv::Mat mat_bgr;
    cv::cvtColor(mat_rgb, mat_bgr, cv::COLOR_RGB2BGR);

    cv::imshow(winName, mat_bgr);
    cv::waitKey(1);
}

inline bool tensor_defined(const py::dict& pkg, const char* key,
                           torch::Tensor& out) {
    auto it = pkg.contains(key) ? pkg[key] : py::none();
    if (!it.is_none()) {
        torch::Tensor t = it.cast<torch::Tensor>();
        if (t.defined()) {         // real tensor found
            out = t;
            return true;
        }
    }
    return false;                  // key missing or undefined
}

static inline int64_t ceil_div(int64_t x, int64_t y)
{ return (x + y - 1) / y; }

// use *existing* field names:  subdiv_until_  /  subdiv_every_
static inline int remaining_subdiv_times(
        int iter,
        const VoxelOptimizationParams& p)
{
    if (iter > p.subdiv_until_) return 0;
    return ceil_div(p.subdiv_until_ - iter, p.subdiv_every_) + 1;
}

void saveTensor(const torch::Tensor &t,
                const std::string &tag,
                const std::string &dbg_dir,
                int iter,
                int image_id)
{
    auto img = t.squeeze(0)
                 .permute({1,2,0})
               // optional gamma:
               // .clamp(0.0f, 1.0f).pow(1.0f/2.2f)
                 .mul(255.0f).clamp(0.0f,255.0f)
                 .to(torch::kUInt8)
                 .contiguous()
                 .cpu();

    int H = img.size(0), W = img.size(1);
    cv::Mat rgb(H, W, CV_8UC3, img.data_ptr());
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);

    std::ostringstream ss;
    ss << dbg_dir << "/" 
       << tag 
       << "_iter" << std::setw(6) << std::setfill('0') << iter
       << "_img"  << std::setw(3) << std::setfill('0') << image_id
       << ".png";
    cv::imwrite(ss.str(), bgr);
}

VoxelMapper::VoxelMapper(std::shared_ptr<ORB_SLAM3::System> pSLAM,
                         const std::filesystem::path& config_file_path,
                        //  const std::filesystem::path& seq_dir,
                        //  const std::filesystem::path& out_dir,
                        //  torch::DeviceType device_type,
                        //  int seed)
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
    loss_log_.open("loss.csv", std::ios::out);
    loss_ssim_log_.open("loss_ssim.csv", std::ios::out);
    loss_l1_log_.open("loss_l1.csv", std::ios::out);

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
    // mSeqDir = seq_dir;
    // mOutDir = out_dir;    

    // Background & override color setup
    std::vector<float> bg_color = {0.0f, 0.0f, 0.0f};  // no white-background logic needed
    background_ = torch::tensor(bg_color,
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    override_color_ = torch::empty(0, torch::TensorOptions().device(device_type_));

    voxel_model_ = std::make_shared<sv::VoxelModel>(model_params_);
    scene_       = std::make_shared<sv::VoxelScene>(model_params_);
    
    size_t N = scene_->keyframes().size();
    best_loss_per_kf_.assign(N,  std::numeric_limits<float>::infinity());
    worst_loss_per_kf_.assign(N, -std::numeric_limits<float>::infinity());
    extrema_dir_ = result_dir_ / "extrema";
    std::filesystem::create_directories(extrema_dir_);

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
    // inactive_geo_densify_ =
    //     (settings_file["Mapper.inactive_geo_densify"].operator int()) != 0;
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
    stable_num_iter_existence_ =
         settings_file["Mapper.stable_num_iter_existence"].operator int();

    pipe_params_.convert_SHs_ =
         (settings_file["Pipeline.convert_SHs"].operator int()) != 0;
    pipe_params_.compute_cov3D_ =
         (settings_file["Pipeline.compute_cov3D"].operator int()) != 0;

    /* ───────── OPTIMIZATION PARAMETERS ───────── */
    opt_params_.iterations_ =
        settings_file["Optimization.max_num_iterations"].operator int();
    opt_params_.geo_lr_init_ =
        settings_file["Optimization.geo_lr_init"].operator float();
    // opt_params_.geo_lr_final_ =
    //     settings_file["Optimization.geo_lr_final"].operator float();
    opt_params_.geo_lr_delay_mult_ =
        settings_file["Optimization.geo_lr_delay_mult"].operator float();
    opt_params_.geo_lr_max_steps_ =
        settings_file["Optimization.geo_lr_max_steps"].operator int();
    // opt_params_.meta_accum_lr_ =
    //     settings_file["Optimization.meta_accum_lr"].operator float();
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
    // opt_params_.subdiv_quantile_ =
    //     settings_file["Optimization.subdiv_quantile"].operator float();
    // opt_params_.subdiv_gradient_threshold_ =
    //     settings_file["Optimization.subdiv_gradient_threshold"].operator float();

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
    // opt_params_.min_voxels_ =
    //     settings_file["Optimization.min_voxels"].operator int();

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
    // Viewer Parameters
     rendered_image_viewer_scale_ =
         settings_file["VoxelViewer.image_scale"].operator float();
     rendered_image_viewer_scale_main_ =
         settings_file["VoxelViewer.image_scale_main"].operator float();

    // std::cout << "\n[CFG] Parsed Optimization Parameters:" << std::endl;
    // // std::cout << "  lr:                       " << opt_params_.position_lr_final_ << std::endl;
    // std::cout << "  meta_accum_lr:           " << opt_params_.meta_accum_lr_ << std::endl;
    // // std::cout << "  position_lr_init:        " << opt_params_.position_lr_init_ << std::endl;
    // // std::cout << "  position_lr_delay_mult:  " << opt_params_.position_lr_delay_mult_ << std::endl;
    // // std::cout << "  position_lr_max_steps:   " << opt_params_.position_lr_max_steps_ << std::endl;
    // std::cout << "  iterations_:      " << opt_params_.iterations_ << std::endl;
    // std::cout << "  densification_interval:  " << opt_params_.densification_interval_ << std::endl;
    // std::cout << "\n[CFG] Subdivision Parameters:" << std::endl;
    // std::cout << "  subdiv_from:             " << opt_params_.subdiv_from_ << std::endl;
    // std::cout << "  subdiv_every:            " << opt_params_.subdiv_every_ << std::endl;
    // std::cout << "  subdiv_until:            " << opt_params_.subdiv_until_ << std::endl;
    // std::cout << "  subdiv_quantile:         " << opt_params_.subdiv_quantile_ << std::endl;
    // std::cout << "  subdiv_gradient_threshold: " << opt_params_.subdiv_gradient_threshold_ << std::endl;
    // std::cout << "\n[CFG] Pruning Parameters:" << std::endl;
    // std::cout << "  prune_from:              " << opt_params_.prune_from_ << std::endl;
    // std::cout << "  prune_every:             " << opt_params_.prune_every_ << std::endl;
    // std::cout << "  prune_until:             " << opt_params_.prune_until_ << std::endl;
    // std::cout << "  prune_threshold_init:    " << opt_params_.prune_threshold_init_ << std::endl;
    // std::cout << "  prune_threshold_final:   " << opt_params_.prune_threshold_final_ << std::endl;
    // // std::cout << "  min_voxels:              " << opt_params_.min_voxels_ << std::endl;
    // std::cout << "\n[CFG] Pipeline & Mapper Flags:" << std::endl;
    // // std::cout << "  inactive_geo_densify:    " << inactive_geo_densify_ << std::endl;
    // std::cout << "  new_keyframe_times_of_use_: " << new_keyframe_times_of_use_<< std::endl;
    // std::cout << "  min_num_initial_map_kfs: " << min_num_initial_map_kfs_ << std::endl;
    // std::cout << "  large_rot_th:            " << large_rot_th_ << std::endl;
    // std::cout << "  large_trans_th:          " << large_trans_th_ << std::endl;
    // std::cout << "  cull_keyframes:          " << cull_keyframes_ << std::endl;
    // std::cout << "\n[CFG] Logging Parameters:" << std::endl;
    // std::cout << "  training_report_interval: " << training_report_interval_ << std::endl;
    // std::cout << "  keyframe_record_interval: " << keyframe_record_interval_ << std::endl;
    // std::cout << "  all_keyframes_record_interval: " << all_keyframes_record_interval_ << std::endl;
    // std::cout << "  record_rendered_image:   " << record_rendered_image_ << std::endl;
    // std::cout << "  record_ground_truth_image: " << record_ground_truth_image_ << std::endl;
    // std::cout << "  record_loss_image:       " << record_loss_image_ << std::endl;
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
                // std::cout << "[DEBUG] Creating VoxelKeyframes..." << std::endl;
                for (const auto& pKF : vKFs)
                {
                    std::shared_ptr<VoxelKeyframe> new_kf = std::make_shared<VoxelKeyframe>(pKF->mnId, getIteration());
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
     while (!isStopped()) {
         // Check conditions for incremental mapping
         if (hasMetIncrementalMappingConditions()) {
             combineMappingOperations();
             if (cull_keyframes_)
                 cullKeyframes();
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
    /* ───────────────────────────────────────────────
     *  3.  TAIL   O P T I M I S A T I O N
     * ───────────────────────────────────────────── */
    //  int densify_interval = densifyInterval();
    //  int n_delay_iters = densify_interval * 0.8;
    int subdiv_interval = opt_params_.subdiv_every_;
    int n_delay_iters   = int(subdiv_interval * 0.8f);
    //  while (getIteration() - SLAM_stop_iter <= n_delay_iters || getIteration() % densify_interval <= n_delay_iters || isKeepingTraining()) {
        while (getIteration() - SLAM_stop_iter <= n_delay_iters || getIteration() % subdiv_interval <= n_delay_iters || isKeepingTraining()) {
         trainForOneIteration();
        //  densify_interval = densifyInterval();
        //  n_delay_iters = densify_interval * 0.8;
        subdiv_interval = opt_params_.subdiv_every_;
        n_delay_iters   = int(subdiv_interval * 0.8f);
     }
     // Save and clear
     renderAndRecordAllKeyframes("_shutdown");
    //  savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
     writeKeyframeUsedTimes(result_dir_ / "used_times", "final");
 
     signalStop();
 }

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
        // saveDebugImage(gt_image, (result_dir_ / "debug_gt.jpg").string());

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

        // 4) grow SH degree every 1000 iterations (locked during render)
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        
        if (getIteration() % 1000 == 0 && default_sh_ < model_params_.sh_degree_)
            default_sh_ += 1;
        voxel_model_->setShDegree(default_sh_);

        // 5) build a MiniCam out of this keyframe
        // viewpoint_cam->computeTransformTensors();            // idempotent
        sv::MiniCam cam = viewpoint_cam->toMiniCam();
        cam.c2w = cam.c2w.contiguous().to(mDevice);   // make sure contiguous + on CUDA
        cam.w2c = cam.w2c.contiguous().to(mDevice);

        py::module_ np = py::module_::import("numpy");
        // {
        //     namespace fs = std::filesystem;
        //     // build per-KF directory: result_dir_/kf0000, kf0001, etc.
        //     std::ostringstream kf_ss;
        //     kf_ss << result_dir_.string() << "/kf"
        //         << std::setw(4) << std::setfill('0') << viewpoint_cam->fid_;
        //     fs::create_directories(kf_ss.str());
        //     const fs::path kf_dir = kf_ss.str();

        //     // iteration number padded
        //     std::ostringstream it_ss;
        //     it_ss << std::setw(6) << std::setfill('0') << getIteration();
        //     const std::string iter_str = it_ss.str();

        //     // 1) dump GT image
        //     {
        //         std::ostringstream fn;
        //         fn << kf_dir.string() << "/gt_kf"
        //         << std::setw(4) << std::setfill('0') << viewpoint_cam->fid_
        //         << "_iter" << iter_str << ".png";
        //         saveDebugImage(gt_image, fn.str());
        //     }

        //     // 2) dump intrinsics K
        //     {
        //         auto opts = torch::TensorOptions()
        //                         .dtype(torch::kFloat32)   // <-- 32-bit
        //                         .device(torch::kCPU);
        //         torch::Tensor K = torch::tensor(
        //             {{static_cast<float>(cam.fx), 0.f, static_cast<float>(cam.cx)},
        //             {0.f, static_cast<float>(cam.fy), static_cast<float>(cam.cy)},
        //             {0.f, 0.f, 1.f}},
        //             opts);
        //         std::ostringstream fn;
        //         fn << kf_dir.string() << "/K_kf"
        //         << std::setw(4) << std::setfill('0') << viewpoint_cam->fid_
        //         << "_iter" << iter_str << ".npy";
        //         py::module_::import("numpy").attr("save")(fn.str(), K);
        //     }

        //     // 3) dump world → camera
        //     {
        //         std::ostringstream fn;
        //         fn << kf_dir.string() << "/w2c_kf"
        //         << std::setw(4) << std::setfill('0') << viewpoint_cam->fid_
        //         << "_iter" << iter_str << ".npy";
        //         py::module_::import("numpy").attr("save")(fn.str(), cam.w2c.cpu());
        //     }

        //     // 4) dump camera → world
        //     {
        //         std::ostringstream fn;
        //         fn << kf_dir.string() << "/c2w_kf"
        //         << std::setw(4) << std::setfill('0') << viewpoint_cam->fid_
        //         << "_iter" << iter_str << ".npy";
        //         py::module_::import("numpy").attr("save")(fn.str(), cam.c2w.cpu());
        //     }
        // }

        // 6) update all learning rates
        if (mpSLAM) {
            int used_times = kfs_used_times_[viewpoint_cam->fid_];
            int step = (used_times <= opt_params_.geo_lr_max_steps_)
                        ? used_times
                        : opt_params_.geo_lr_max_steps_;
            float geo_lr = voxel_model_->updateLearningRate(step);
            // voxel_model_->setGeoLearningRate(geo_lr);
            setGeoLearningRateInit(geo_lr);
        } else {
            voxel_model_->updateLearningRate(getIteration());
        }
        // SH-DC and higher-SH rates remain constant:
        voxel_model_->setSh0LearningRate(sh0LearningRate());
        voxel_model_->setShsLearningRate(shsLearningRate());

        torch::Tensor chw_u8 = viewpoint_cam->original_image_    // (3,H,W) in [0,1]
                                .mul(255.0f)
                                .clamp(0.0f, 255.0f)
                                .to(torch::kUInt8)            // <--- cast to U8
                                .cpu()
                                .contiguous();
        torch::Tensor hwc_u8 = chw_u8.permute({1,2,0}).contiguous();  // (H,W,3)
        py::array rgb_numpy = sv::tensorToNumpyRGB(hwc_u8);
        // np.attr("save")("/home/dimitris/Photo-SLAM/gt_image.npy", rgb_numpy);
        auto render_pkg = voxel_model_->render(cam, rgb_numpy, "");
        // lock_render.unlock();
        if (render_pkg.empty() || !render_pkg.count("color") || !render_pkg.at("color").defined()) {
            std::cout << "render pkg empty" << std::endl;
            return;
        }
        torch::Tensor rendered_image = render_pkg["color"].to(mDevice);
        torch::Tensor masked_image = rendered_image * mask;      // (1,3,H,W)
        
        // 8) compute masked photometric loss
        // showTensor(gt_image,     "GT Image");
        // showTensor(masked_image, "Masked Render");
        // saveTensor(gt_image,     "gt",    "/home/dimitris/Photo-SLAM/debug", getIteration(), viewpoint_cam->fid_);
        // saveTensor(masked_image, "masked", "/home/dimitris/Photo-SLAM/debug", getIteration(), viewpoint_cam->fid_);
        auto Ll1 = loss_utils::l1_loss(masked_image, gt_image);
        float lambda_dssim = lambdaDssim();
        auto loss = (1.0 - lambda_dssim) * Ll1
                + lambda_dssim * (1.0 - loss_utils::ssim(masked_image, gt_image, mDevice.type()));

        {
            py::gil_scoped_release no_gil;
            loss.backward();

            if (mDevice == torch::kCUDA)  
                torch::cuda::synchronize();
        }

        // 10) accumulate subdivision gradients into subdiv_meta_
        {
            /* ------------------------------------------------ 0. bookkeeping */
            torch::NoGradGuard no_grad;
            // 0) bookkeeping
            ema_loss_for_log_ = 0.4f * loss.item<float>() + 0.6f * ema_loss_for_log_;
            if (keyframe_record_interval_ &&
                getIteration() % keyframe_record_interval_ == 0)
                recordKeyframeRendered(masked_image,
                                        gt_image,
                                        viewpoint_cam->fid_,
                                        result_dir_, result_dir_, result_dir_);

            // 1) grab the raw ∂L/∂subdiv_p_ from the last backward pass
            auto grad_p = voxel_model_->getSubdivPriorityGrad()  // [M,1]
                                .view(-1);                      // [M]

            // 2) restrict to just those voxels that actually contributed
            auto idx_tensor = render_pkg["idx"];
            if (!idx_tensor.defined()) {
                // nothing visible this iteration – skip optimisation step
                voxel_model_->optimizer_->zero_grad(true);
                return;
            }
            torch::Tensor vis_idx = idx_tensor.to(mDevice, /*non_blocking=*/true)
                                            .contiguous();
            if (vis_idx.numel() == 0) {
                // no voxel hit: just zero-out grads and continue
                voxel_model_->optimizer_->zero_grad(true);
                return;
            }
            voxel_model_->accumulateSubdivGradients(vis_idx,
                                                    grad_p.index({vis_idx}));
            // std::cout << "[DBG]  max|grad_buf| = "
            //         << voxel_model_->subdiv_p_grad_buffer_.abs().max().item<float>()
            //         << '\n';

            {
                // std::cout << "\n[DBG] iter " << iteration_
                //         << "  render_pkg contains:\n";
                // for (const auto& kv : render_pkg)              // std::pair<std::string,Tensor>
                // {
                //     const std::string& name = kv.first;
                //     const torch::Tensor& t  = kv.second;
                //     bool defined = t.defined();
                //     std::string dtype = defined ? std::string(t.dtype().name()) : "n/a";
                //     std::cout << "    • " << name
                //             << "  | defined=" << std::boolalpha << defined
                //             << "  | dtype="   << dtype
                //             << "  | sizes=";
                //     if (!defined || t.dim() == 0) {
                //         std::cout << "[]";
                //     } else {
                //         std::cout << '[';
                //         for (int i = 0; i < t.dim(); ++i) {
                //             std::cout << t.size(i) << (i + 1 == t.dim() ? "" : ",");
                //         }
                //         std::cout << ']';
                //     }
                //     std::cout << '\n';
                // }
                auto it = render_pkg.find("geom");
                if (it != render_pkg.end() && it->second.defined())
                {
                    /* ---------------- geometry buffer ------------------------- */
                    torch::Tensor geom = it->second.to(mDevice, /*non_blocking=*/true)
                                                    .contiguous();          // (H,W)  int64

                    /*  PER-PIXEL L1 error ..................................... */
                    torch::Tensor per_pix_err =
                        (rendered_image - gt_image).abs()   // (1,3,H,W)
                                                    .mean(1)   // → (1,H,W)
                                                    .squeeze(0);            // (H,W)

                    /*  gather only fg pixels .................................. */
                    torch::Tensor mask_fg  = geom >= 0;                      // bool(H,W)
                    torch::Tensor vox_ids  = geom.masked_select(mask_fg)
                                                    .to(torch::kLong);       // (K,)
                    torch::Tensor pix_err  = per_pix_err.masked_select(mask_fg)
                                                    .unsqueeze(1);         // (K,1)

                    voxel_model_->voxel_error_sum_ .index_add_(0, vox_ids, pix_err);
                    voxel_model_->voxel_hit_count_.index_add_(0, vox_ids,
                                                            torch::ones_like(pix_err));


                    float loss_val = loss.item<float>();
                    int fid = viewpoint_cam->fid_;                 // key-frame ID being trained
                    // --- 1) ensure our vectors are big enough ---
                    if (fid >= static_cast<int>(best_loss_per_kf_.size())) {
                        size_t newSize = fid + 1;
                        best_loss_per_kf_.resize(newSize,
                                                std::numeric_limits<float>::infinity());
                        worst_loss_per_kf_.resize(newSize,
                                                -std::numeric_limits<float>::infinity());
                    }
                    // references into the right slot
                    float &best  = best_loss_per_kf_[fid];
                    float &worst = worst_loss_per_kf_[fid];
                    // --- 2) update “best” for this KF ---
                    if (loss_val < best) {
                        best = loss_val;
                        auto kf_dir = extrema_dir_ / ("kf" + std::to_string(fid));
                        std::filesystem::create_directories(kf_dir);
                        saveTensor(gt_image,     "best_gt",     kf_dir.string(), iteration_, fid);
                        saveTensor(masked_image, "best_masked", kf_dir.string(), iteration_, fid);
                        saveVoxelErrorHeatmap(cam,
                            geom,
                            gt_image,
                            viewpoint_cam->fid_,                              // <- key-frame id
                            (result_dir_ / "heatmaps").string());
                    }
                    // --- 3) update “worst” for this KF ---
                    if (loss_val > worst) {
                        worst = loss_val;
                        auto kf_dir = extrema_dir_ / ("kf" + std::to_string(fid));
                        std::filesystem::create_directories(kf_dir);
                        saveTensor(gt_image,     "worst_gt",     kf_dir.string(), iteration_, fid);
                        saveTensor(masked_image, "worst_masked", kf_dir.string(), iteration_, fid);
                        saveVoxelErrorHeatmap(cam,
                            geom,
                            gt_image,
                            viewpoint_cam->fid_,                              // <- key-frame id
                            (result_dir_ / "heatmaps").string());
                    }
                    loss_log_ << iteration_ << "," << loss_val << "\n";
                    loss_l1_log_ << iteration_ << "," << Ll1.item<float>() << "\n";
                    loss_ssim_log_ << iteration_ << "," 
                        << (1.0 - loss_utils::ssim(masked_image, gt_image, mDevice.type()).item<float>()) << "\n";

                    /* --------------------------------------------------------- */
                    /*  DEBUG: print per-voxel mean error (top-k or full)        */
                    if (iteration_ % 50 == 0) {                         // -- every 50 iters
                        torch::Tensor mean_err =
                            voxel_model_->voxel_error_sum_
                            / voxel_model_->voxel_hit_count_.clamp_min(1);  // (N,1)
                        mean_err = mean_err.squeeze(1);                     // (N)

                        // print worst 10 voxels
                        auto top = std::get<1>(mean_err.topk(/*k=*/10));    // indices
                        // std::cout << "[DBG] iter " << iteration_
                        //         << "  worst-10 voxel errors:\n";
                        for (int i = 0; i < top.size(0); ++i) {
                            int64_t id  = top[i].item<int64_t>();
                            float    err = mean_err[id].item<float>();
                            // std::cout << "    id " << id << " : " << err << '\n';
                        }
                    }
                } // if geom defined
            }
            // 3) PRUNE
            if (getIteration() >= opt_params_.prune_from_ &&
                getIteration() % opt_params_.prune_every_ == 0)
            {
                std::cout << "PRUNE" << std::endl;
                // Try to use SVRaster’s max_w; if it wasn’t produced, fall back to grad-threshold
                auto it = render_pkg.find("max_w");
                if (it != render_pkg.end() && it->second.defined()) {
                    // 3a) compute linear threshold ramp exactly as Python does
                    torch::Tensor max_w = it->second.to(mDevice);     // (M,1)
                    float alpha = float(getIteration() - opt_params_.prune_from_) /
                                std::max(1.f,
                                        float(opt_params_.prune_until_ -
                                                opt_params_.prune_from_));
                    alpha = std::clamp(alpha, 0.f, 1.f);
                    float prune_thres = (1.f - alpha) * opt_params_.prune_threshold_init_ +
                                        alpha        * opt_params_.prune_threshold_final_;
                    // 3b) build keep mask and prune
                    torch::Tensor keep_mask = (max_w >= prune_thres).view(-1);  // bool[M]
                    voxel_model_->prune(keep_mask);
                } else {
                    // Fallback: SVAdaptive’s old gradient rule
                    torch::Tensor buf = voxel_model_->subdiv_meta_.view(-1);  // accumulated grads
                    torch::Tensor keep_mask =
                        (buf >= opt_params_.subdiv_gradient_threshold_);      // bool[M]
                    voxel_model_->prune(keep_mask);
                }
            }
            // 4) SUBDIVIDE
            if (getIteration() >= opt_params_.subdiv_from_ &&
                getIteration() %  opt_params_.subdiv_every_ == 0)
            {
                std::cout << "SUBDIVIDE" << std::endl;
                // ------------------------------------------------------------ (a) compute subdivide_prop on-the-fly
                int rem_times = remaining_subdiv_times(getIteration(), opt_params_);
                // scale factor such that repeated ×scale eventually reaches target
                float scale_each = std::pow(opt_params_.subdivide_target_scale_, 1.f / rem_times);
                float subdivide_prop = std::max(0.f, (scale_each - 1.f) / 7.f);   // identical to Python

                auto stat = voxel_model_->computeTrainingStat({cam}, rgb_numpy);

                /*  guard: nothing visible → nothing to split  */
                if (stat.min_samp_interval.numel() == 0) {
                    voxel_model_->subdiv_meta_.zero_();
                } else {
                    // torch::Tensor size_thres =
                    //     stat.min_samp_interval * opt_params_.subdivide_samp_thres_;
                    // /*  make  ‖size_thres‖ match device & shape of size_ */
                    // size_thres = size_thres.to(voxel_model_->size_.device())
                    //                     .expand_as(voxel_model_->size_);
                    auto size_thres = stat.min_samp_interval.squeeze(1) * opt_params_.subdivide_samp_thres_;
                    size_thres = size_thres.to(voxel_model_->size_.device());

                    // ------------------------------------------------------------ (b) validity gate (size + octree level)
                    torch::Tensor valid =
                        (voxel_model_->size_ * 0.5 > size_thres).view(-1) &           // per-voxel
                        (voxel_model_->oct_level_ < sv::MAX_OCT_LEVEL).view(-1);

                    // ------------------------------------------------------------ (c) priority   (= accumulated grad)
                    torch::Tensor priority =
                        voxel_model_->subdiv_meta_.view(-1) * valid;               // (M)

                    torch::Tensor rank = torch::zeros_like(priority);
                    rank.index_put_(
                        {priority.argsort(/*dim=*/0, /*descending=*/false)},           // ← modified
                        torch::arange(
                            priority.numel(),
                            torch::TensorOptions()
                                .dtype(priority.dtype())
                                .device(priority.device())));                          // ← modified
    
                    // ------------------------------------------------------------ (d) pick threshold so that
                    //                            top `subdivide_prop` fraction will split
                    float th;
                    if (getIteration() <= opt_params_.subdivide_all_until_) {
                        th = -1.f;                          // split everything valid (boot-strap phase)
                    } else {
                        th = rank.quantile(1.f - subdivide_prop)
                                .template item<float>();
                    }

                    torch::Tensor subdivide_mask = (rank > th) & valid;

                    // ------------------------------------------------------------ (e) respect global voxel cap
                    int64_t max_n_subdiv = std::max<int64_t>(
                        1,
                        (opt_params_.subdivide_max_num_ -
                        static_cast<int64_t>(voxel_model_->center_.size(0))) / 7);

                    if (subdivide_mask.sum().item<int64_t>() > max_n_subdiv)
                    {
                        // keep only the highest-rank `max_n_subdiv` parents
                        auto sel_rank = rank.index({subdivide_mask});
                        float cutoff  = std::get<0>(sel_rank.sort(/*descending=*/true))
                                            [max_n_subdiv - 1]
                                            .template item<float>();
                        subdivide_mask &= (rank >= cutoff);
                    }

                    // ------------------------------------------------------------ (f) split and reset accumulator
                    if (subdivide_mask.any().item<bool>())
                        voxel_model_->subdivide(subdivide_mask);

                    if (iteration_ % 100 == 0) {
                        int64_t n_valid  = valid.sum().item<int64_t>();
                        int64_t n_ranked = (priority > 0).sum().item<int64_t>();
                        int64_t n_split  = subdivide_mask.sum().item<int64_t>();
                        std::cout << "[DBG] it " << iteration_
                                << " valid="  << n_valid
                                << "  ranked="<< n_ranked
                                << "  split=" << n_split << '\n';
                    }
                    voxel_model_->subdiv_meta_.zero_();
                }
            }

            // every training_report_interval_ iterations, print a concise report
            // if (training_report_interval_ && iteration_ % training_report_interval_ == 0) {
            //     sv::VoxelTrainer::trainingReport(
            //         iteration_,
            //         opt_params_.iterations_,
            //         Ll1,
            //         loss,
            //         ema_loss_for_log_,
            //         loss_utils::l1_loss,
            //         // elapsed_ms,
            //         *voxel_model_,
            //         *scene_,
            //         pipe_params_,
            //         background_
            //     );
            // }

            if ((all_keyframes_record_interval_ && getIteration() % all_keyframes_record_interval_ == 0)
                )
            {
                renderAndRecordAllKeyframes();
                // savePly(result_dir_ / std::to_string(iteration_) / "ply");
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

        if (loop_closure_iteration_)
            loop_closure_iteration_ = false;

        // Optimizer step   
        if (getIteration() < opt_params_.iterations_) {
        //  py::gil_scoped_release no_gil;
            voxel_model_->optimizer_->step();
            voxel_model_->optimizer_->zero_grad(true);
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
             // std::cout << "[Gaussian Mapper]Local BA Detected."
             //           << std::endl;
 
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
                     pkf->computeTransformTensors();
 
                     // Give local BA keyframes times of use
                     increaseKeyframeTimesOfUse(pkf, local_BA_increased_times_of_use_);
                 }
                 else {
                     handleNewKeyframe(kf);
                 }
             }
 
             // Get new points
             auto& associated_points = opr.associatedMapPoints();
             auto& points = std::get<0>(associated_points);
             auto& colors = std::get<1>(associated_points);
 
            //  // Add new points to the model
            //  if (initial_mapped_ && points.size() >= 30) {
            //      torch::NoGradGuard no_grad;
            //      std::unique_lock<std::mutex> lock_render(mutex_render_);
            //      voxel_model_->increasePcd(points, colors, getIteration());
            //  }
         }
         break;
 
         case ORB_SLAM3::MappingOperation::OprType::LoopClosingBA:
         {
             std::cout << "[Voxel Mapper]Loop Closure Detected."
                       << std::endl;
 
             // Get the loop keyframe scale modification factor
             float loop_kf_scale = opr.mfScale;
 
             // Get new keyframes (scaled transformation applied in ORB-SLAM3)
             auto& associated_kfs = opr.associatedKeyFrames();
             // Mark the transformed points to avoid transforming more than once
             torch::Tensor point_not_transformed_flags =
                 torch::full(
                     {voxel_model_->center_.size(0)},
                     true,
                     torch::TensorOptions().device(device_type_).dtype(torch::kBool));
            //  if (record_loop_ply_)
            //      savePly(result_dir_ / (std::to_string(getIteration()) + "_0_before_loop_correction"));
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
 // renderAndRecordKeyframe(pkf, result_dir_, "_1_after_loop_transforming_points");
 // std::cout<<num_transformed<<std::endl;
                         }
 // }
                     pkf->setPose(
                         pose.unit_quaternion().cast<double>(),
                         pose.translation().cast<double>());
                     pkf->computeTransformTensors();
 // if (std::get<4>(kf)) renderAndRecordKeyframe(pkf, result_dir_, "_2_after_pose_correction");
                 }
                 else {
                     handleNewKeyframe(kf);
                 }
             }
            //  if (record_loop_ply_)
            //      savePly(result_dir_ / (std::to_string(getIteration()) + "_1_after_loop_correction"));
 // keyframesToJson(result_dir_ / (std::to_string(getIteration()) + "_0_before_loop_correction"));
 
             // Get new points (scaled transformation applied in ORB-SLAM3, so this step is performed at last to avoid scaling twice)
             auto& associated_points = opr.associatedMapPoints();
             auto& points = std::get<0>(associated_points);
             auto& colors = std::get<1>(associated_points);
 
             // Add new points to the model
            //  if (initial_mapped_ && points.size() >= 30) {
            //     std::cout << "adds new points" << std::endl;
            //     torch::NoGradGuard no_grad;
            //     std::unique_lock<std::mutex> lock_render(mutex_render_);
            //     voxel_model_->increasePcd(points, colors, getIteration());
            //  }
 
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
                     pkf->setPose(Tcy.unit_quaternion().cast<double>(), Tcy.translation().cast<double>());
                     pkf->computeTransformTensors();
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

    // Get dense point cloud from the new keyframe to accelerate training
     pkf->img_undist_ = imgRGB_undistorted;
    //  pkf->img_auxiliary_undist_ = imgAux_undistorted;
    //  pkf->kps_pixel_ = std::move(std::get<6>(kf));
    //  pkf->kps_point_local_ = std::move(std::get<7>(kf));
    //  if (isdoingInactiveGeoDensify())
    //      increasePcdByKeyframeInactiveGeoDensify(pkf);
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

    if (render_pkg.empty() || !render_pkg.count("color") || !render_pkg.at("color").defined()) {
        std::cout << "render pkg empty" << std::endl;
        return;
    }
    torch::Tensor rendered_image = render_pkg.at("color").to(mDevice);          // (1,3,H,W)

    torch::Tensor masked_image = rendered_image * undistort_mask_[pkf->camera_id_];
    // saveTensor(masked_image,     "masked_image",    "/home/dimitris/Photo-SLAM/debug", getIteration(), pkf->fid_);
    masked_image = masked_image.squeeze(0);   
    auto gt_image = pkf->original_image_;
    // saveTensor(gt_image,     "gt_image",    "/home/dimitris/Photo-SLAM/debug", getIteration(), pkf->fid_);

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

cv::Mat VoxelMapper::renderFromPose(
    const Sophus::SE3f &Tcw,
    const int width,
    const int height,
    const bool main_vision)
    {
    if (!initial_mapped_ || getIteration() <= 0)
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    std::shared_ptr<VoxelKeyframe> pkf = std::make_shared<VoxelKeyframe>();
    pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
    // Pose
    pkf->setPose(
        Tcw.unit_quaternion().cast<double>(),
        Tcw.translation().cast<double>());
    try {
        // Camera
        sv::Camera& camera = scene_->cameras_.at(viewer_camera_id_);
        pkf->setCameraParams(camera);
        // Transformations
        pkf->computeTransformTensors();
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[Mapper::renderFromPose]KeyFrame Camera not found!");
    }

    // // MiniCam helper already implemented in VoxelKeyframe
    // sv::MiniCam miniCam = pkf->toMiniCam();
    // miniCam.c2w = miniCam.c2w.to(device_type_);
    // miniCam.w2c = miniCam.w2c.to(device_type_);

    // auto chw_u8 = pkf->original_image_
    //                   .mul(255.0f).clamp(0.0f,255.0f)
    //                   .to(torch::kUInt8)
    //                   .cpu()
    //                   .contiguous();
    // auto hwc_u8 = chw_u8.permute({1,2,0}).contiguous();
    // py::array rgb_numpy = sv::tensorToNumpyRGB(hwc_u8);

    // 3) call into your voxel_renderer under the lock
    std::unordered_map<std::string, at::Tensor> render_pkg;
    {
        std::unique_lock<std::mutex> lock(mutex_render_);
        render_pkg = voxel_model_->render(pkf->toMiniCam(), /*rgb_numpy unused*/ py::array(), "viewer");
    }

    // 4) extract the “rgb” tensor (batch of 1×3×H×W)
    if (!render_pkg.count("color") || !render_pkg["color"].defined()) {
        // if rendering failed, return a black image
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    }
    at::Tensor rgb = render_pkg["color"];          // (1,3,H,W)
    rgb = rgb.to(torch::kCPU);
    rgb = rgb.squeeze(0);                        // → (3,H,W)

    cv::imwrite("debug_rgb.png", tensor_utils::torchTensor2CvMat_Float32(rgb));
    std::cout << "[renderFromPose] rgb min/max = "
          << rgb.min().item<float>() << "/"
          << rgb.max().item<float>() << "\n";

    // 5) apply the appropriate undistort mask
    at::Tensor mask = main_vision
        ? viewer_main_undistort_mask_.at(pkf->camera_id_)
        : viewer_sub_undistort_mask_.at(pkf->camera_id_);
    // ensure mask is on CPU and broadcastable
    mask = mask.to(torch::kFloat32).to(rgb.device());
    at::Tensor masked = rgb * mask;             // (3,H,W)

    // 6) convert back to cv::Mat
    return tensor_utils::torchTensor2CvMat_Float32(masked);
}

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

// int VoxelMapper::densifyInterval()
// {
//     std::unique_lock<std::mutex> lock(mutex_settings_);
//     return opt_params_.densification_interval_;
// }

int VoxelMapper::subdivideInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.subdiv_every_;
}

 int VoxelMapper::stableNumIterExistence()
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     return stable_num_iter_existence_;
 }

 void VoxelMapper::setGeoLearningRateInit(const float lr)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     opt_params_.geo_lr_init_ = lr;
 }

float VoxelMapper::lambdaDssim()
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     return opt_params_.lambda_dssim_;
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

//   bool VoxelMapper::isdoingInactiveGeoDensify()
//  {
//      std::unique_lock<std::mutex> lock(mutex_settings_);
//      return inactive_geo_densify_;
//  }


void VoxelMapper::saveVoxelErrorHeatmap(const sv::MiniCam&  /*cam*/,
                                        const torch::Tensor& geom,
                                        const torch::Tensor&  gt_img,
                                        int                 fid,          // NEW
                                        const std::string&  base_dir)     // ← e.g. result_dir_/heatmaps
{
    namespace fs = std::filesystem;
    const fs::path dir_kf = fs::path(base_dir) / ("kf" + std::to_string(fid));
    fs::create_directories(dir_kf);       // <-- makes .../heatmaps/kf<i>
    torch::NoGradGuard no_grad;

    /* ------------------------------------------------------------------ *
     * ❶  error per voxel  →  per-pixel array  (H,W) in [vmin,vmax]        *
     * ------------------------------------------------------------------ */
    torch::Tensor vox_err = (voxel_model_->voxel_error_sum_
                            / voxel_model_->voxel_hit_count_.clamp_min(1))
                                .squeeze(1);                                 // (N)

    const int H = geom.size(0), W = geom.size(1);

    torch::Tensor idx_flat = geom.view(-1).to(torch::kLong);
    torch::Tensor valid    = idx_flat >= 0;
    torch::Tensor safe_idx = idx_flat.clone().masked_fill(~valid, 0);

    torch::Tensor pix_err  = vox_err.index_select(0, safe_idx)
                                   .view({H, W});
    pix_err.masked_fill_(~valid.view({H, W}), 0);

    float vmin = pix_err.min().item<float>(),
          vmax = pix_err.max().item<float>(),
          range= std::max(1e-6f, vmax - vmin);

    torch::Tensor H01 = (pix_err - vmin) / range;            // → [0,1]

    /* ------------------------------------------------------------------ *
     * ❷  Jet colour-map  (same formula as before)                         *
     * ------------------------------------------------------------------ */
    auto R = (1.5f * H01 - 0.5f).clamp(0, 1);
    auto G = (1.5f - (2 * H01 - 1).abs()).clamp(0, 1);
    auto B = (0.5f - 1.5f * H01).clamp(0, 1);
    torch::Tensor rgb = torch::stack({B, G, R}, -1) * 255.0f;  // BGR for OpenCV
    rgb = rgb.to(torch::kUInt8).cpu().contiguous();            // (H,W,3)

    /* ------------------------------------------------------------------ *
     * ❸  convert to cv::Mat                                              *
     * ------------------------------------------------------------------ */
    cv::Mat img(H, W, CV_8UC3, rgb.data_ptr<uint8_t>());

    /* ------------------------------------------------------------------ *
     * ❹  legend bar  (32 px wide)                                         *
     * ------------------------------------------------------------------ */
    const int LWIDTH = 32;
    cv::Mat legend(H, LWIDTH, CV_8UC3);

    for (int y = 0; y < H; ++y)
    {
        float val = 1.f - float(y) / float(H - 1);   // top=max (red), bottom=min (blue)
        float r = std::clamp( 1.5f*val - 0.5f , 0.f, 1.f),
              g = std::clamp( 1.5f - std::abs(2*val -1) , 0.f, 1.f),
              b = std::clamp( 0.5f - 1.5f*val , 0.f, 1.f);
        cv::Vec3b col{ uint8_t(255*b), uint8_t(255*g), uint8_t(255*r) };
        legend.row(y).setTo(col);
    }

    const std::string lbl_hi = "high loss (red)";
    const std::string lbl_lo = "low loss (blue)";
    // near the top of the legend bar:
    cv::putText(legend, lbl_hi, {2, 14},
                cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(255,255,255), 1, cv::LINE_AA);
    // near the bottom of the legend bar:
    cv::putText(legend, lbl_lo, {2, H - 6},
                cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(255,255,255), 1, cv::LINE_AA);

    /* ------------------------------------------------------------------ *
     * ❺  stack data + legend & annotate tick labels                       *
     * ------------------------------------------------------------------ */
    cv::Mat out;
    cv::hconcat(img, legend, out);

    const int x0 = W + LWIDTH + 4;        // text anchor (pixels from left)
    auto put = [&](float frac, const std::string& txt)
    {
        int y = int((1.f - frac) * (H - 1));
        cv::putText(out, txt, {x0, y},
                    cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(255,255,255), 1, cv::LINE_AA);
        cv::line(out,
                 {W, y}, {W + LWIDTH - 1, y},
                 cv::Scalar(255,255,255), 1, cv::LINE_AA);
    };

    std::ostringstream ss; ss.setf(std::ios::fixed); ss.precision(3);
    ss << vmax; put(1.f, ss.str());  ss.str(""); ss.clear();
    ss << vmax - 0.25f*range; put(0.75f, ss.str()); ss.str(""); ss.clear();
    ss << vmin + 0.50f*range; put(0.50f, ss.str()); ss.str(""); ss.clear();
    ss << vmin + 0.25f*range; put(0.25f, ss.str()); ss.str(""); ss.clear();
    ss << vmin;  put(0.f, ss.str());

    /* ------------------------------------------------------------------ *
     * ❻  write png                                                       *
     * ------------------------------------------------------------------ */
    std::ostringstream fn;
    fn << "kf"   << std::setw(4) << std::setfill('0') << fid
       << "_iter"<< std::setw(6) << std::setfill('0') << iteration_
       << ".png";

    cv::imwrite((dir_kf / fn.str()).string(), out);

        // ————————————— ❼ convert & write GT image —————————————
    // assume gt_img is (1,3,H,W) float in [0,1]
    auto gt = (gt_img.squeeze(0).mul(255.0f)
                   .clamp(0.0f,255.0f)
                   .to(torch::kUInt8)
                   .permute({1,2,0})            // H,W,3 RGB
                   .cpu()
                   .contiguous());
    // convert to BGR for OpenCV:
    cv::Mat gt_mat(H, W, CV_8UC3, gt.data_ptr<uint8_t>());
    cv::cvtColor(gt_mat, gt_mat, cv::COLOR_RGB2BGR);

    std::ostringstream fn2;
    fn2<< "kf"<<std::setw(4)<<std::setfill('0')<<fid
       << "_iter"<<std::setw(6)<<std::setfill('0')<<iteration_
       << "_gt.png";
    cv::imwrite((dir_kf / fn2.str()).string(), gt_mat);
}