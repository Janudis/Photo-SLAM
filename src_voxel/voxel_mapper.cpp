#include "include_voxel/voxel_mapper.h"
#include "include_voxel/voxel_mapper_utils.h"
#include "include_voxel/voxel_mapper_supervision.h"
#include "include_voxel/mapper_depth_registry.h"
#include "include_voxel/tandem_mvs_backend.h"
#include "include_voxel/omnidata_depth_backend.h"
#include <pybind11/embed.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <Eigen/Eigenvalues>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <opencv2/flann.hpp>
#include <queue>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <c10/cuda/CUDACachingAllocator.h>

#include "ORB-SLAM3/include/Atlas.h"
#include "ORB-SLAM3/include/MapPoint.h"

namespace py = pybind11;

namespace {
struct VoxelRuntimeGpuStats
{
    float reserved_mb = 0.0f;
    float allocated_mb = 0.0f;
};

VoxelRuntimeGpuStats getVoxelRuntimeGpuStats()
{
    VoxelRuntimeGpuStats stats;
    if (!torch::cuda::is_available()) {
        return stats;
    }

    namespace c10Alloc = c10::cuda::CUDACachingAllocator;
    const c10Alloc::DeviceStats mem_stats = c10Alloc::getDeviceStats(0);
    stats.reserved_mb =
        mem_stats.reserved_bytes.front().peak /
        (1024.0f * 1024.0f);
    stats.allocated_mb =
        mem_stats.allocated_bytes.front().peak /
        (1024.0f * 1024.0f);
    return stats;
}

double voxelFileSizeMb(const std::filesystem::path& path)
{
    if (!std::filesystem::exists(path)) {
        return 0.0;
    }
    return static_cast<double>(std::filesystem::file_size(path)) /
           (1024.0 * 1024.0);
}

void saveVoxelRuntimeMetrics(
    const std::filesystem::path& path,
    int frames,
    int keyframes,
    int voxels,
    int iterations,
    double mapping_seconds,
    const std::filesystem::path& map_path,
    const VoxelRuntimeGpuStats& gpu_stats)
{
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::ofstream out(path);
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"frames\": " << frames << ",\n";
    out << "  \"keyframes\": " << keyframes << ",\n";
    out << "  \"primitive_type\": \"voxels\",\n";
    out << "  \"primitive_count\": " << voxels << ",\n";
    out << "  \"voxels\": " << voxels << ",\n";
    out << "  \"iterations\": " << iterations << ",\n";
    out << "  \"total_seconds\": " << mapping_seconds << ",\n";
    out << "  \"fps_hz\": "
        << (mapping_seconds > 0.0
                ? static_cast<double>(frames) / mapping_seconds
                : 0.0)
        << ",\n";
    out << "  \"runtime_scope\": "
           "\"mapping and tail optimization; evaluation export excluded\",\n";
    out << "  \"map_path\": \"" << map_path.string() << "\",\n";
    out << "  \"map_size_mb\": " << voxelFileSizeMb(map_path) << ",\n";
    out << "  \"gpu_memory_allocated_mb\": "
        << gpu_stats.allocated_mb << ",\n";
    out << "  \"gpu_memory_reserved_mb\": "
        << gpu_stats.reserved_mb << "\n";
    out << "}\n";
}

cv::Mat mapperDepthForKeyframe(
    const std::string& image_filename,
    const cv::Mat& tracking_depth,
    sv::Camera& camera)
{
    cv::Mat mapper_depth = sv::loadMapperDepthImage(image_filename);
    if (mapper_depth.empty()) {
        cv::Mat undistorted;
        camera.undistortImage(tracking_depth, undistorted);
        return undistorted;
    }
    if (mapper_depth.type() != CV_32FC1) {
        mapper_depth.convertTo(mapper_depth, CV_32FC1);
    }
    cv::Mat undistorted;
    cv::remap(
        mapper_depth,
        undistorted,
        camera.undistort_map1,
        camera.undistort_map2,
        cv::INTER_NEAREST);
    return undistorted;
}

std::filesystem::path resolveMapperResourcePath(
    const std::filesystem::path& config_file,
    const std::filesystem::path& configured_path)
{
    if (configured_path.empty() || configured_path.is_absolute()) {
        return configured_path;
    }
    if (std::filesystem::exists(configured_path)) {
        return std::filesystem::absolute(configured_path);
    }

    std::filesystem::path cursor =
        std::filesystem::absolute(config_file).parent_path();
    while (!cursor.empty()) {
        if (std::filesystem::exists(cursor / "CMakeLists.txt")) {
            return cursor / configured_path;
        }
        const std::filesystem::path parent = cursor.parent_path();
        if (parent == cursor) {
            break;
        }
        cursor = parent;
    }
    return configured_path;
}

void ensurePythonRuntimeInitialized(bool import_torch_cuda)
{
    static bool initialized_by_mapper = false;
    static PyThreadState* released_main_thread_state = nullptr;
    const auto configure_python_paths = []() {
        py::exec(R"PY(
import os
import site
import sys

_user_site = site.getusersitepackages()
site.addsitedir(_user_site)
for _path in (os.path.join(_user_site, "rerun_sdk"), _user_site):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)
)PY");
        py::module_::import("sys").attr("path").attr("insert")(0, "scripts_voxel");
        py::module_::import("sys").attr("path").attr("insert")(0, "../scripts_voxel");
    };

    if (!initialized_by_mapper && Py_IsInitialized() == 0) {
        py::initialize_interpreter(false);
        initialized_by_mapper = true;
        configure_python_paths();
        if (import_torch_cuda) {
            py::module_::import("torch.cuda");
        }
        released_main_thread_state = PyEval_SaveThread();
        return;
    }

    {
        py::gil_scoped_acquire gil;
        configure_python_paths();
        if (import_torch_cuda) {
            py::module_::import("torch.cuda");
        }
    }
    (void)released_main_thread_state;
}
} // namespace

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
    laptop_precheck_profiler_ =
        std::make_unique<sv::LaptopPrecheckProfiler>(
            laptop_precheck_enabled_,
            laptop_precheck_sample_interval_ms_,
            device_type_ == torch::kCUDA);
    laptop_precheck_profiler_->start();
    auto mapper_initialization_profile =
        profileLaptopModule("mapper_initialization");

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
    voxel_model_->setSdfInitializationOrbRadiusVox(
        sdf_initialization_orb_radius_vox_);
    voxel_model_->setTopologySdfInitializationMode(sdf_initialization_mode_);
    voxel_model_->setFilterNearVoxels(opt_params_.filter_near_voxels_);

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
        if (monocular_rendered_depth_densify_) {
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
        if (monocular_omnidata_densify_) {
            std::cout
                << "; scale-aligned Omnidata depth closes multi-view-"
                   "consistent residual render holes";
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

    if (isMonocularMvsPipelineEnabled() || monocular_omnidata_densify_) {
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
        auto model_load_profile =
            profileLaptopModule("mvs_model_load");
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
            << ", depth range: " << monocular_mvs_depth_range_mode_
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
                << monocular_mvs_tsdf_evidence_pixel_stride_
                << ", truncation="
                << monocular_mvs_tsdf_evidence_trunc_vox_
                << " voxels, promotion views="
                << monocular_mvs_tsdf_evidence_promote_min_views_
                << ".\n";
        }
        if (opt_params_.prune_mvs_consistency_enable_) {
            std::cout
                << "[VoxelMapper] TANDEM MVS pruning: protect supported "
                   "SDF/co-visibility candidates and carve multi-view "
                   "free space; support views="
                << opt_params_.prune_mvs_min_supporting_views_
                << ", contradiction views="
                << opt_params_.prune_mvs_min_contradicting_views_
                << ", tolerance="
                << opt_params_.prune_mvs_depth_tolerance_vox_
                << " voxels.\n";
        }
    }

    if (monocular_omnidata_densify_) {
        auto model_load_profile =
            profileLaptopModule("omnidata_model_load");
        const std::filesystem::path model_path = resolveMapperResourcePath(
            config_file_path_, monocular_omnidata_model_path_);
        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error(
                "Omnidata TorchScript model is missing: " +
                model_path.string() +
                ". Export it with scripts/export_omnidata_depth_torchscript.py");
        }
        monocular_omnidata_backend_ =
            std::make_shared<sv::OmnidataDepthBackend>(
                model_path,
                monocular_omnidata_input_size_,
                monocular_omnidata_use_amp_);
        std::cout
            << "[VoxelMapper] Loaded Omnidata depth model: "
            << model_path << "\n"
            << "[VoxelMapper] Omnidata alignment: HI-SLAM2 2x2 inverse-depth "
               "scale field in "
            << (monocular_mvs_requires_inertial_ba1_
                    ? "visual-inertial metric units"
                    : "pure-monocular ORB scene units")
            << ".\n";
    }

    // Debug snapshots are queued natively and Rerun is initialized only when
    // the recording is saved, so instrumentation cannot alter online cadence.
    auto& rerun_bridge = sv::RerunVisualizerBridge::instance();
    rerun_bridge.setEnabled(rerun_params_.enable_rerun_);
    const bool requires_live_rerun =
        rerun_params_.run_whole_run_ ||
        rerun_params_.rerun_reconstruction_mesh_ ||
        rerun_params_.rerun_maps_ ||
        rerun_params_.rerun_gt_mesh_ ||
        rerun_params_.rerun_rendered_mesh_eval_;
    if (rerun_params_.enable_rerun_ && requires_live_rerun) {
        ensureEmbeddedPythonRuntime(/*import_torch_cuda=*/false);
        rerun_bridge.init("PhotoSLAM-SVRecon", /*spawn_viewer=*/false);
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
    if (!settings_file["Camera.z_far"].empty()) {
        z_far_ = std::max(
            z_near_,
            settings_file["Camera.z_far"].operator float());
    }
    if (!settings_file["Mapper.inactive_geo_densify_max_pixel_dist"].empty()) {
        inactive_geo_densify_max_pixel_dist_ =
            settings_file["Mapper.inactive_geo_densify_max_pixel_dist"].operator float();
    }
    cull_keyframes_ =
        (settings_file["Mapper.cull_keyframes"].operator int()) != 0;
    if (!settings_file["Mapper.input_queue_max_keyframes"].empty()) {
        input_queue_max_keyframes_ = std::max(
            0,
            settings_file["Mapper.input_queue_max_keyframes"].operator int());
    }
    if (!settings_file["Mapper.incremental_mapping_window_size"].empty()) {
        incremental_mapping_window_size_ = std::max(
            0,
            settings_file["Mapper.incremental_mapping_window_size"].operator int());
    }
    if (!settings_file["Mapper.loop_closure_reinsert_points"].empty()) {
        loop_closure_reinsert_points_ =
            (settings_file["Mapper.loop_closure_reinsert_points"].operator int()) != 0;
    }
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

    // Monocular configurations use the generic camera clipping range and do
    // not need RGB-D-specific entries. RGB-D configurations retain their
    // explicit sensor limits.
    RGBD_min_depth_ = z_near_;
    RGBD_max_depth_ = z_far_;
    if (!settings_file["RGBD.min_depth"].empty()) {
        RGBD_min_depth_ =
            settings_file["RGBD.min_depth"].operator float();
    }
    if (!settings_file["RGBD.max_depth"].empty()) {
        RGBD_max_depth_ =
            settings_file["RGBD.max_depth"].operator float();
    }

    inactive_geo_densify_ =
        (settings_file["Mapper.inactive_geo_densify"].operator int()) != 0;
    if (!settings_file["Mapper.monocular_rendered_depth_densify"].empty()) {
        monocular_rendered_depth_densify_ =
            (settings_file["Mapper.monocular_rendered_depth_densify"]
                 .operator int()) != 0;
    }
    if (!settings_file["Mapper.monocular_rendered_depth_pixel_stride"].empty()) {
        monocular_rendered_depth_pixel_stride_ = std::max(
            1,
            settings_file["Mapper.monocular_rendered_depth_pixel_stride"]
                .operator int());
    }
    if (!settings_file["Mapper.monocular_rendered_depth_evidence_samples"].empty()) {
        monocular_rendered_depth_evidence_samples_ = std::max(
            2,
            settings_file["Mapper.monocular_rendered_depth_evidence_samples"]
                .operator int());
    }
    if (!settings_file["Mapper.monocular_rendered_depth_evidence_trunc_vox"].empty()) {
        monocular_rendered_depth_evidence_trunc_vox_ = std::max(
            0.25f,
            settings_file["Mapper.monocular_rendered_depth_evidence_trunc_vox"]
                .operator float());
    }
    if (!settings_file["Mapper.monocular_rendered_depth_evidence_max_weight"].empty()) {
        monocular_rendered_depth_evidence_max_weight_ = std::max(
            1.0f,
            settings_file["Mapper.monocular_rendered_depth_evidence_max_weight"]
                .operator float());
    }
    if (!settings_file["Mapper.monocular_rendered_depth_evidence_promote_min_views"].empty()) {
        monocular_rendered_depth_evidence_promote_min_views_ = std::max(
            2,
            settings_file["Mapper.monocular_rendered_depth_evidence_promote_min_views"]
                .operator int());
    }
    if (!settings_file["Mapper.monocular_rendered_depth_evidence_promote_min_weight"].empty()) {
        monocular_rendered_depth_evidence_promote_min_weight_ = std::max(
            0.0f,
            settings_file["Mapper.monocular_rendered_depth_evidence_promote_min_weight"]
                .operator float());
    }
    if (!settings_file["Mapper.monocular_rendered_depth_evidence_min_baseline_ratio"].empty()) {
        monocular_rendered_depth_evidence_min_baseline_ratio_ = std::max(
            0.0f,
            settings_file["Mapper.monocular_rendered_depth_evidence_min_baseline_ratio"]
                .operator float());
    }
    if (!settings_file["Mapper.monocular_mvs_densify"].empty()) {
        monocular_mvs_densify_ =
            (settings_file["Mapper.monocular_mvs_densify"].operator int()) != 0;
    }
    if (!settings_file["Mapper.monocular_mvs_model_dir"].empty()) {
        monocular_mvs_model_dir_ =
            settings_file["Mapper.monocular_mvs_model_dir"].operator std::string();
    }
    if (!settings_file["Mapper.monocular_mvs_width"].empty()) {
        monocular_mvs_width_ = std::max(
            1, settings_file["Mapper.monocular_mvs_width"].operator int());
    }
    if (!settings_file["Mapper.monocular_mvs_height"].empty()) {
        monocular_mvs_height_ = std::max(
            1, settings_file["Mapper.monocular_mvs_height"].operator int());
    }
    if (!settings_file["Mapper.monocular_mvs_view_num"].empty()) {
        monocular_mvs_view_num_ = std::max(
            2, settings_file["Mapper.monocular_mvs_view_num"].operator int());
    }
    if (!settings_file["Mapper.monocular_mvs_depth_range_mode"].empty()) {
        monocular_mvs_depth_range_mode_ =
            settings_file["Mapper.monocular_mvs_depth_range_mode"]
                .operator std::string();
    }
    if (!settings_file["Mapper.monocular_mvs_depth_min_m"].empty()) {
        monocular_mvs_depth_min_m_ = std::max(
            1.0e-4f,
            settings_file["Mapper.monocular_mvs_depth_min_m"].operator float());
    }
    if (!settings_file["Mapper.monocular_mvs_depth_max_m"].empty()) {
        monocular_mvs_depth_max_m_ = std::max(
            monocular_mvs_depth_min_m_ + 1.0e-4f,
            settings_file["Mapper.monocular_mvs_depth_max_m"].operator float());
    }
    if (!settings_file["Mapper.monocular_mvs_depth_min_scene"].empty()) {
        monocular_mvs_depth_min_scene_ = std::max(
            1.0e-4f,
            settings_file["Mapper.monocular_mvs_depth_min_scene"]
                .operator float());
    }
    if (!settings_file["Mapper.monocular_mvs_inverse_depth_quantile"].empty()) {
        monocular_mvs_inverse_depth_quantile_ =
            settings_file["Mapper.monocular_mvs_inverse_depth_quantile"]
                .operator float();
    }
    if (!settings_file["Mapper.monocular_mvs_depth_max_multiplier"].empty()) {
        monocular_mvs_depth_max_multiplier_ =
            settings_file["Mapper.monocular_mvs_depth_max_multiplier"]
                .operator float();
    }
    if (!settings_file["Mapper.monocular_mvs_discard_percentage"].empty()) {
        monocular_mvs_discard_percentage_ = std::clamp(
            settings_file["Mapper.monocular_mvs_discard_percentage"].operator float(),
            0.0f,
            100.0f);
    }
    if (!settings_file["Mapper.monocular_mvs_empty_cache_before_launch"].empty()) {
        monocular_mvs_empty_cache_before_launch_ =
            settings_file["Mapper.monocular_mvs_empty_cache_before_launch"]
                .operator int() != 0;
    }
    if (!settings_file["Mapper.monocular_mvs_tsdf_evidence"].empty()) {
        monocular_mvs_tsdf_evidence_ =
            settings_file["Mapper.monocular_mvs_tsdf_evidence"]
                .operator int() != 0;
    }
    if (!settings_file["Mapper.monocular_mvs_tsdf_evidence_pixel_stride"].empty()) {
        monocular_mvs_tsdf_evidence_pixel_stride_ = std::max(
            1,
            settings_file["Mapper.monocular_mvs_tsdf_evidence_pixel_stride"]
                .operator int());
    }
    if (!settings_file["Mapper.monocular_mvs_tsdf_evidence_trunc_vox"].empty()) {
        monocular_mvs_tsdf_evidence_trunc_vox_ = std::max(
            1.0f,
            settings_file["Mapper.monocular_mvs_tsdf_evidence_trunc_vox"]
                .operator float());
    }
    if (!settings_file["Mapper.monocular_mvs_tsdf_evidence_max_weight"].empty()) {
        monocular_mvs_tsdf_evidence_max_weight_ = std::max(
            1.0e-4f,
            settings_file["Mapper.monocular_mvs_tsdf_evidence_max_weight"]
                .operator float());
    }
    if (!settings_file["Mapper.monocular_mvs_tsdf_evidence_promote_min_views"].empty()) {
        monocular_mvs_tsdf_evidence_promote_min_views_ = std::max(
            1,
            settings_file["Mapper.monocular_mvs_tsdf_evidence_promote_min_views"]
                .operator int());
    }
    if (!settings_file["Mapper.monocular_mvs_tsdf_evidence_promote_min_weight"].empty()) {
        monocular_mvs_tsdf_evidence_promote_min_weight_ = std::max(
            0.0f,
            settings_file["Mapper.monocular_mvs_tsdf_evidence_promote_min_weight"]
                .operator float());
    }
    if (!settings_file["Mapper.monocular_omnidata_densify"].empty()) {
        monocular_omnidata_densify_ =
            settings_file["Mapper.monocular_omnidata_densify"].operator int() != 0;
    }
    if (!settings_file["Mapper.monocular_omnidata_model_path"].empty()) {
        monocular_omnidata_model_path_ =
            settings_file["Mapper.monocular_omnidata_model_path"]
                .operator std::string();
    }
    if (!settings_file["Mapper.monocular_omnidata_input_size"].empty()) {
        monocular_omnidata_input_size_ = std::max(
            32,
            settings_file["Mapper.monocular_omnidata_input_size"].operator int());
    }
    if (!settings_file["Mapper.monocular_omnidata_width"].empty()) {
        monocular_omnidata_width_ = std::max(
            1,
            settings_file["Mapper.monocular_omnidata_width"].operator int());
    }
    if (!settings_file["Mapper.monocular_omnidata_height"].empty()) {
        monocular_omnidata_height_ = std::max(
            1,
            settings_file["Mapper.monocular_omnidata_height"].operator int());
    }
    if (!settings_file["Mapper.monocular_omnidata_view_num"].empty()) {
        monocular_omnidata_view_num_ = std::max(
            2,
            settings_file["Mapper.monocular_omnidata_view_num"].operator int());
    }
    if (!settings_file["Mapper.monocular_omnidata_depth_multiplier"].empty()) {
        monocular_omnidata_depth_multiplier_ = std::max(
            1.0e-4f,
            settings_file["Mapper.monocular_omnidata_depth_multiplier"]
                .operator float());
    }
    if (!settings_file["Mapper.monocular_omnidata_min_alignment_anchors"].empty()) {
        monocular_omnidata_min_alignment_anchors_ = std::max(
            4,
            settings_file["Mapper.monocular_omnidata_min_alignment_anchors"]
                .operator int());
    }
    if (!settings_file["Mapper.monocular_omnidata_max_alignment_rel_error"].empty()) {
        monocular_omnidata_max_alignment_rel_error_ = std::max(
            0.0f,
            settings_file["Mapper.monocular_omnidata_max_alignment_rel_error"]
                .operator float());
    }
    if (!settings_file["Mapper.monocular_omnidata_min_source_views"].empty()) {
        monocular_omnidata_min_source_views_ = std::max(
            1,
            settings_file["Mapper.monocular_omnidata_min_source_views"]
                .operator int());
    }
    if (!settings_file["Mapper.monocular_omnidata_consistency_rel_tol"].empty()) {
        monocular_omnidata_consistency_rel_tol_ = std::max(
            0.0f,
            settings_file["Mapper.monocular_omnidata_consistency_rel_tol"]
                .operator float());
    }
    if (!settings_file["Mapper.monocular_omnidata_consistency_vox"].empty()) {
        monocular_omnidata_consistency_vox_ = std::max(
            0.0f,
            settings_file["Mapper.monocular_omnidata_consistency_vox"]
                .operator float());
    }
    if (!settings_file["Mapper.monocular_omnidata_use_amp"].empty()) {
        monocular_omnidata_use_amp_ =
            settings_file["Mapper.monocular_omnidata_use_amp"].operator int() != 0;
    }
    if (!settings_file["Mapper.monocular_omnidata_empty_cache_before_launch"].empty()) {
        monocular_omnidata_empty_cache_before_launch_ =
            settings_file["Mapper.monocular_omnidata_empty_cache_before_launch"]
                .operator int() != 0;
    }
    if (isMonocularMvsPipelineEnabled() && monocular_omnidata_densify_) {
        throw std::runtime_error(
            "TANDEM MVS and Omnidata are mutually exclusive depth ablations");
    }
    if (isMonocularMvsPipelineEnabled() || monocular_omnidata_densify_) {
        if (monocular_mvs_depth_range_mode_ != "fixed" &&
            monocular_mvs_depth_range_mode_ !=
                "tandem_sparse_quantile") {
            throw std::runtime_error(
                "Mapper.monocular_mvs_depth_range_mode must be fixed or "
                "tandem_sparse_quantile");
        }
        if (!(monocular_mvs_inverse_depth_quantile_ > 0.0f &&
              monocular_mvs_inverse_depth_quantile_ < 1.0f)) {
            throw std::runtime_error(
                "Mapper.monocular_mvs_inverse_depth_quantile must be in "
                "(0, 1)");
        }
        if (!(monocular_mvs_depth_max_multiplier_ > 0.0f)) {
            throw std::runtime_error(
                "Mapper.monocular_mvs_depth_max_multiplier must be positive");
        }
        if (isMonocularMvsPipelineEnabled() &&
            monocular_mvs_model_dir_.empty()) {
            throw std::runtime_error(
                "Mapper.monocular_mvs_model_dir is required when "
                "TANDEM MVS densification or TSDF evidence is enabled");
        }
        if (monocular_omnidata_densify_ &&
            monocular_omnidata_model_path_.empty()) {
            throw std::runtime_error(
                "Mapper.monocular_omnidata_model_path is required when "
                "Mapper.monocular_omnidata_densify=1");
        }
        if (monocular_omnidata_input_size_ % 32 != 0) {
            throw std::runtime_error(
                "Mapper.monocular_omnidata_input_size must be divisible by 32");
        }
        if (monocular_omnidata_densify_ &&
            monocular_omnidata_min_source_views_ >=
                monocular_omnidata_view_num_) {
            throw std::runtime_error(
                "Omnidata min_source_views must be smaller than view_num");
        }
        if (monocular_rendered_depth_densify_) {
            std::cout
                << "[VoxelMapper] Learned depth replaces rendered pseudo-depth "
                   "densification; disabling the latter.\n";
            monocular_rendered_depth_densify_ = false;
        }
    }
    if (!settings_file["Mapper.allocate_orb_voxels"].empty()) {
        allocate_orb_voxels_ =
            (settings_file["Mapper.allocate_orb_voxels"].operator int()) != 0;
    }
    if (!settings_file["Model.outside_level"].empty()) {
        svrecon_outside_level_ =
            std::max(0, settings_file["Model.outside_level"].operator int());
    }
    if (!settings_file["Model.global_scene_extent"].empty()) {
        global_scene_extent_m_ = std::max(
            0.0f,
            settings_file["Model.global_scene_extent"].operator float());
    }
    if (!settings_file["Model.robust_scene_bounds"].empty()) {
        robust_scene_bounds_ =
            (settings_file["Model.robust_scene_bounds"].operator int()) != 0;
    }
    if (!settings_file["Mapper.sdf_initialization_rgbd_projective"].empty()) {
        sdf_initialization_rgbd_projective_ =
            (settings_file["Mapper.sdf_initialization_rgbd_projective"].operator int()) != 0;
    }
    if (!settings_file["Mapper.sdf_initialization_orb_radius_vox"].empty()) {
        sdf_initialization_orb_radius_vox_ = std::max(
            0.0f,
            settings_file["Mapper.sdf_initialization_orb_radius_vox"].operator float());
    }
    if (!settings_file["Mapper.sdf_initialization_mode"].empty()) {
        sdf_initialization_mode_ = voxel_utils::toLowerCopy(
            settings_file["Mapper.sdf_initialization_mode"].operator std::string());
    }
    if (sdf_initialization_mode_ != "orb_prior" &&
        sdf_initialization_mode_ != "weak_positive" &&
        sdf_initialization_mode_ != "source_points" &&
        sdf_initialization_mode_ != "weak_surface_prior") {
        throw std::runtime_error(
            "[VoxelMapper] Mapper.sdf_initialization_mode must be one of: "
            "orb_prior, weak_positive, source_points, weak_surface_prior");
    }
    max_depth_cached_ =
        settings_file["Mapper.depth_cache"].operator int();
    if (!settings_file["Mapper.rgbd_fill_render_holes_initial_backfill"].empty()) {
        rgbd_fill_render_holes_initial_backfill_ =
            (settings_file["Mapper.rgbd_fill_render_holes_initial_backfill"].operator int()) != 0;
    }
    if (!settings_file["Mapper.rgbd_fill_render_holes"].empty()) {
        rgbd_fill_render_holes_ =
            (settings_file["Mapper.rgbd_fill_render_holes"].operator int()) != 0;
    }
    if (!settings_file["Mapper.rgbd_fill_render_holes_projective_sdf"].empty()) {
        rgbd_fill_render_holes_projective_sdf_ =
            (settings_file["Mapper.rgbd_fill_render_holes_projective_sdf"].operator int()) != 0;
    }
    if (!settings_file["Mapper.rgbd_fill_render_holes_stride"].empty()) {
        rgbd_fill_render_holes_stride_ =
            std::max(1, settings_file["Mapper.rgbd_fill_render_holes_stride"].operator int());
    }
    if (!settings_file["Mapper.rgbd_tsdf_evidence"].empty()) {
        rgbd_tsdf_evidence_ =
            (settings_file["Mapper.rgbd_tsdf_evidence"].operator int()) != 0;
    }
    if (!settings_file["Mapper.rgbd_tsdf_evidence_initial_backfill"].empty()) {
        rgbd_tsdf_evidence_initial_backfill_ =
            (settings_file["Mapper.rgbd_tsdf_evidence_initial_backfill"].operator int()) != 0;
    }
    if (!settings_file["Mapper.rgbd_tsdf_evidence_pixel_stride"].empty()) {
        rgbd_tsdf_evidence_pixel_stride_ = std::max(
            1,
            settings_file["Mapper.rgbd_tsdf_evidence_pixel_stride"].operator int());
    }
    if (!settings_file["Mapper.rgbd_tsdf_evidence_trunc_vox"].empty()) {
        rgbd_tsdf_evidence_trunc_vox_ = std::max(
            0.5f,
            settings_file["Mapper.rgbd_tsdf_evidence_trunc_vox"].operator float());
    }
    if (!settings_file["Mapper.rgbd_tsdf_evidence_max_weight"].empty()) {
        rgbd_tsdf_evidence_max_weight_ = std::max(
            1.0e-4f,
            settings_file["Mapper.rgbd_tsdf_evidence_max_weight"].operator float());
    }
    if (!settings_file["Mapper.rgbd_tsdf_evidence_promote_min_views"].empty()) {
        rgbd_tsdf_evidence_promote_min_views_ = std::max(
            1,
            settings_file["Mapper.rgbd_tsdf_evidence_promote_min_views"].operator int());
    }
    if (rgbd_tsdf_evidence_) {
        if (rgbd_fill_render_holes_ || sdf_initialization_rgbd_projective_) {
            std::cout
                << "[VoxelMapper] Mapper.rgbd_tsdf_evidence uses ORB, optional "
                   "inactive geometry, and evidence-only residual-hole fusion; "
                   "disabling direct RGB-D hole filling and global projective "
                   "SDF initialization.\n";
        }
        rgbd_fill_render_holes_ = false;
        sdf_initialization_rgbd_projective_ = false;
    }
    if (!settings_file["Mapper.sdf_voxel_size_m"].empty()) {
        sdf_params_.sdf_voxel_size_m_ =
            std::max(1.0e-4f, settings_file["Mapper.sdf_voxel_size_m"].operator float());
    }
    if (!settings_file["Mapper.sdf_init_trunc_vox"].empty()) {
        sdf_params_.sdf_init_trunc_vox_ =
            std::max(1.0e-3f, settings_file["Mapper.sdf_init_trunc_vox"].operator float());
    }
    if (!settings_file["Mapper.sdf_init_max_depth_m"].empty()) {
        sdf_params_.sdf_init_max_depth_m_ =
            std::max(0.0f, settings_file["Mapper.sdf_init_max_depth_m"].operator float());
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

    if (!settings_file["Optimization.adapt_from"].empty()) {
        opt_params_.adapt_from_ =
            settings_file["Optimization.adapt_from"].operator int();
    }
    if (!settings_file["Optimization.adapt_every"].empty()) {
        opt_params_.adapt_every_ =
            settings_file["Optimization.adapt_every"].operator int();
    }
    if (!settings_file["Optimization.prune_every"].empty()) {
        opt_params_.prune_every_ =
            settings_file["Optimization.prune_every"].operator int();
    }
    if (!settings_file["Optimization.subdivide_every"].empty()) {
        opt_params_.subdivide_every_ =
            settings_file["Optimization.subdivide_every"].operator int();
    }
    if (!settings_file["Optimization.filter_near_voxels"].empty()) {
        opt_params_.filter_near_voxels_ =
            (settings_file["Optimization.filter_near_voxels"].operator int()) != 0;
    }
    if (!settings_file["Optimization.prune_far_voxels"].empty()) {
        opt_params_.prune_far_voxels_ =
            (settings_file["Optimization.prune_far_voxels"].operator int()) != 0;
    }
    opt_params_.prune_near_voxels_geometric_ =
        !settings_file["Optimization.prune_near_voxels_geometric"].empty() &&
        (settings_file["Optimization.prune_near_voxels_geometric"].operator int()) != 0;
    if (!settings_file["Optimization.prune_surface_views_enable"].empty()) {
        opt_params_.prune_surface_views_enable_ =
            (settings_file["Optimization.prune_surface_views_enable"].operator int()) != 0;
    }
    if (!settings_file["Optimization.surface_min_views"].empty()) {
        opt_params_.surface_min_views_ = std::max(
            4,
            settings_file["Optimization.surface_min_views"].operator int());
    }
    if (!settings_file["Optimization.surface_view_window_size"].empty()) {
        opt_params_.surface_view_window_size_ = std::max(
            opt_params_.surface_min_views_,
            settings_file["Optimization.surface_view_window_size"]
                .operator int());
    }
    if (!settings_file["Optimization.prune_mvs_consistency_enable"].empty()) {
        opt_params_.prune_mvs_consistency_enable_ =
            settings_file["Optimization.prune_mvs_consistency_enable"]
                .operator int() != 0;
    }
    if (!settings_file["Optimization.prune_mvs_min_supporting_views"].empty()) {
        opt_params_.prune_mvs_min_supporting_views_ = std::max(
            1,
            settings_file["Optimization.prune_mvs_min_supporting_views"]
                .operator int());
    }
    if (!settings_file["Optimization.prune_mvs_min_contradicting_views"].empty()) {
        opt_params_.prune_mvs_min_contradicting_views_ = std::max(
            1,
            settings_file["Optimization.prune_mvs_min_contradicting_views"]
                .operator int());
    }
    if (!settings_file["Optimization.prune_mvs_depth_tolerance_vox"].empty()) {
        opt_params_.prune_mvs_depth_tolerance_vox_ = std::max(
            0.0f,
            settings_file["Optimization.prune_mvs_depth_tolerance_vox"]
                .operator float());
    }
    if (!settings_file["Optimization.final_refinement_enable"].empty()) {
        opt_params_.final_refinement_enable_ =
            (settings_file["Optimization.final_refinement_enable"].operator int()) != 0;
    }
    opt_params_.prune_from_ = !settings_file["Optimization.prune_from"].empty()
        ? settings_file["Optimization.prune_from"].operator int()
        : opt_params_.adapt_from_;
    opt_params_.prune_until_ =
        settings_file["Optimization.prune_until"].operator int();
    opt_params_.prune_thres_init_ =
        settings_file["Optimization.prune_thres_init"].operator float();
    opt_params_.prune_thres_final_ =
        settings_file["Optimization.prune_thres_final"].operator float();
    opt_params_.prune_thres_final_at_target_ =
        settings_file["Optimization.prune_thres_final_at_target"].operator float();
    opt_params_.subdivide_from_ = !settings_file["Optimization.subdivide_from"].empty()
        ? settings_file["Optimization.subdivide_from"].operator int()
        : opt_params_.adapt_from_;
    opt_params_.subdivide_all_until_ =
        settings_file["Optimization.subdivide_all_until"].operator int();
    opt_params_.subdivide_samp_thres_ =
        settings_file["Optimization.subdivide_samp_thres"].operator float();
    if (!settings_file["Optimization.subdivide_prop"].empty()) {
        opt_params_.subdivide_prop_ = std::clamp(
            settings_file["Optimization.subdivide_prop"].operator float(),
            0.0f,
            1.0f);
    }
    if (!settings_file["Optimization.subdivide_max_num"].empty()) {
        opt_params_.subdivide_max_num_ = std::max(
            1,
            settings_file["Optimization.subdivide_max_num"].operator int());
    }
    opt_params_.use_l1_ =
        (settings_file["Optimization.use_l1"].operator int()) != 0;
    opt_params_.use_huber_ =
        (settings_file["Optimization.use_huber"].operator int()) != 0;
    opt_params_.huber_thres_ =
        settings_file["Optimization.huber_thres"].operator float();
    if (opt_params_.use_l1_ && opt_params_.use_huber_) {
        std::cout << "[VoxelMapper] Both Optimization.use_l1 and Optimization.use_huber are enabled. "
                  << "Prioritizing L1 to match SVRecon." << std::endl;
    }

    opt_params_.lambda_tv_density_ =
        settings_file["Optimization.lambda_tv_density"].operator float();
    opt_params_.tv_from_ =
        settings_file["Optimization.tv_from"].operator int();
    opt_params_.tv_until_ =
        settings_file["Optimization.tv_until"].operator int();
    if (!settings_file["Optimization.lambda_ge_density"].empty()) {
        opt_params_.lambda_ge_density_ =
            std::max(0.0f, settings_file["Optimization.lambda_ge_density"].operator float());
    }
    if (!settings_file["Optimization.ge_from"].empty()) {
        opt_params_.ge_from_ = settings_file["Optimization.ge_from"].operator int();
    }
    if (!settings_file["Optimization.ge_until"].empty()) {
        opt_params_.ge_until_ = settings_file["Optimization.ge_until"].operator int();
    }
    if (!settings_file["Optimization.lambda_ls_density"].empty()) {
        opt_params_.lambda_ls_density_ =
            std::max(0.0f, settings_file["Optimization.lambda_ls_density"].operator float());
    }
    if (!settings_file["Optimization.ls_from"].empty()) {
        opt_params_.ls_from_ = settings_file["Optimization.ls_from"].operator int();
    }
    if (!settings_file["Optimization.ls_until"].empty()) {
        opt_params_.ls_until_ = settings_file["Optimization.ls_until"].operator int();
    }
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
    if (!settings_file["Optimization.lambda_monocular_depth"].empty()) {
        opt_params_.lambda_monocular_depth_ = std::max(
            0.0f,
            settings_file["Optimization.lambda_monocular_depth"]
                .operator float());
    }
    if (!settings_file["Optimization.monocular_depth_from"].empty()) {
        opt_params_.monocular_depth_from_ = std::max(
            0,
            settings_file["Optimization.monocular_depth_from"]
                .operator int());
    }
    if (!settings_file["Optimization.monocular_depth_end"].empty()) {
        opt_params_.monocular_depth_end_ = std::max(
            opt_params_.monocular_depth_from_,
            settings_file["Optimization.monocular_depth_end"]
                .operator int());
    }
    if (!settings_file["Optimization.monocular_depth_end_mult"].empty()) {
        opt_params_.monocular_depth_end_mult_ = std::clamp(
            settings_file["Optimization.monocular_depth_end_mult"]
                .operator float(),
            0.0f,
            1.0f);
    }
    if (!settings_file["Optimization.monocular_depth_alpha_min"].empty()) {
        opt_params_.monocular_depth_alpha_min_ = std::clamp(
            settings_file["Optimization.monocular_depth_alpha_min"]
                .operator float(),
            0.0f,
            1.0f);
    }
    if (!settings_file["Optimization.monocular_depth_confidence_min"].empty()) {
        opt_params_.monocular_depth_confidence_min_ = std::clamp(
            settings_file["Optimization.monocular_depth_confidence_min"]
                .operator float(),
            0.0f,
            1.0f);
    }
    if (!settings_file["Optimization.lambda_monocular_normal"].empty()) {
        opt_params_.lambda_monocular_normal_ = std::max(
            0.0f,
            settings_file["Optimization.lambda_monocular_normal"]
                .operator float());
    }
    if (!settings_file["Optimization.monocular_normal_from"].empty()) {
        opt_params_.monocular_normal_from_ = std::max(
            0,
            settings_file["Optimization.monocular_normal_from"]
                .operator int());
    }
    if (!settings_file["Optimization.monocular_normal_end"].empty()) {
        opt_params_.monocular_normal_end_ = std::max(
            opt_params_.monocular_normal_from_,
            settings_file["Optimization.monocular_normal_end"]
                .operator int());
    }
    if (!settings_file["Optimization.monocular_normal_end_mult"].empty()) {
        opt_params_.monocular_normal_end_mult_ = std::clamp(
            settings_file["Optimization.monocular_normal_end_mult"]
                .operator float(),
            0.0f,
            1.0f);
    }
    if (!settings_file["Optimization.monocular_normal_ks"].empty()) {
        opt_params_.monocular_normal_ks_ = std::max(
            3,
            settings_file["Optimization.monocular_normal_ks"]
                .operator int());
        if ((opt_params_.monocular_normal_ks_ % 2) == 0) {
            ++opt_params_.monocular_normal_ks_;
        }
    }
    if (!settings_file["Optimization.monocular_normal_tol_deg"].empty()) {
        opt_params_.monocular_normal_tol_deg_ = std::clamp(
            settings_file["Optimization.monocular_normal_tol_deg"]
                .operator float(),
            0.0f,
            180.0f);
    }
    if (!settings_file["Optimization.monocular_normal_max_depth_jump_rel"].empty()) {
        opt_params_.monocular_normal_max_depth_jump_rel_ = std::max(
            0.0f,
            settings_file["Optimization.monocular_normal_max_depth_jump_rel"]
                .operator float());
    }
    if (!settings_file["Optimization.lambda_rgbd_sdf"].empty()) {
        opt_params_.lambda_rgbd_sdf_ = settings_file["Optimization.lambda_rgbd_sdf"].operator float();
    }
    if (!settings_file["Optimization.rgbd_sdf_from"].empty()) {
        opt_params_.rgbd_sdf_from_ = settings_file["Optimization.rgbd_sdf_from"].operator int();
    }
    if (!settings_file["Optimization.rgbd_sdf_end"].empty()) {
        opt_params_.rgbd_sdf_end_ = settings_file["Optimization.rgbd_sdf_end"].operator int();
    }
    if (!settings_file["Optimization.rgbd_sdf_end_mult"].empty()) {
        opt_params_.rgbd_sdf_end_mult_ = settings_file["Optimization.rgbd_sdf_end_mult"].operator float();
    }
    if (!settings_file["Optimization.rgbd_sdf_trunc_vox"].empty()) {
        opt_params_.rgbd_sdf_trunc_vox_ = settings_file["Optimization.rgbd_sdf_trunc_vox"].operator float();
    }
    if (!settings_file["Optimization.rgbd_sdf_max_samples"].empty()) {
        opt_params_.rgbd_sdf_max_samples_ = settings_file["Optimization.rgbd_sdf_max_samples"].operator int();
    }
    if (!settings_file["Optimization.rgbd_sdf_ray_pixels"].empty()) {
        opt_params_.rgbd_sdf_ray_pixels_ = settings_file["Optimization.rgbd_sdf_ray_pixels"].operator int();
    }
    if (!settings_file["Optimization.rgbd_sdf_free_samples"].empty()) {
        opt_params_.rgbd_sdf_free_samples_ = settings_file["Optimization.rgbd_sdf_free_samples"].operator int();
    }
    if (!settings_file["Optimization.rgbd_sdf_surface_samples"].empty()) {
        opt_params_.rgbd_sdf_surface_samples_ = settings_file["Optimization.rgbd_sdf_surface_samples"].operator int();
    }
    if (!settings_file["Optimization.rgbd_sdf_w_fs"].empty()) {
        opt_params_.rgbd_sdf_w_fs_ = settings_file["Optimization.rgbd_sdf_w_fs"].operator float();
    }
    if (!settings_file["Optimization.rgbd_sdf_w_center"].empty()) {
        opt_params_.rgbd_sdf_w_center_ = settings_file["Optimization.rgbd_sdf_w_center"].operator float();
    }
    if (!settings_file["Optimization.rgbd_sdf_w_tail"].empty()) {
        opt_params_.rgbd_sdf_w_tail_ = settings_file["Optimization.rgbd_sdf_w_tail"].operator float();
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
    laptop_precheck_enabled_ =
        settings_file["Record.laptop_precheck"].empty()
            ? true
            : (settings_file["Record.laptop_precheck"].operator int()) != 0;
    laptop_precheck_sample_interval_ms_ =
        settings_file["Record.laptop_precheck_sample_interval_ms"].empty()
            ? 50
            : std::max(
                  10,
                  settings_file["Record.laptop_precheck_sample_interval_ms"]
                      .operator int());
    rerun_params_.enable_rerun_ =
        (settings_file["Record.enable_rerun"].operator int()) != 0;
    rerun_params_.rerun_max_keyframes_ =
        settings_file["Record.rerun_max_keyframes"].operator int();
    rerun_params_.rerun_keyframe_start_ =
        std::max(0, settings_file["Record.rerun_keyframe_start"].operator int());
    rerun_params_.run_whole_run_ =
        !settings_file["Record.run_whole_run"].empty() &&
        (settings_file["Record.run_whole_run"].operator int()) != 0;
    rerun_params_.rerun_svrecon_debug_ =
        !settings_file["Record.rerun_svrecon_debug"].empty() &&
        (settings_file["Record.rerun_svrecon_debug"].operator int()) != 0;
    rerun_params_.rerun_gt_mesh_ =
        !settings_file["Record.rerun_gt_mesh"].empty() &&
        (settings_file["Record.rerun_gt_mesh"].operator int()) != 0;
    rerun_params_.rerun_gt_mesh_path_ =
        settings_file["Record.rerun_gt_mesh_path"].empty()
            ? std::string()
            : settings_file["Record.rerun_gt_mesh_path"].operator std::string();
    rerun_params_.rerun_nvblox_mesh_ =
        !settings_file["Record.rerun_nvblox_mesh"].empty() &&
        (settings_file["Record.rerun_nvblox_mesh"].operator int()) != 0;
    rerun_params_.rerun_nvblox_mesh_path_ =
        settings_file["Record.rerun_nvblox_mesh_path"].empty()
            ? std::string()
            : settings_file["Record.rerun_nvblox_mesh_path"]
                  .operator std::string();
    if (!rerun_params_.rerun_nvblox_mesh_path_.empty()) {
        std::filesystem::path nvblox_path(
            rerun_params_.rerun_nvblox_mesh_path_);
        if (nvblox_path.is_relative()) {
            nvblox_path = cfg_path.parent_path() / nvblox_path;
        }
        rerun_params_.rerun_nvblox_mesh_path_ =
            nvblox_path.lexically_normal().string();
    }
    rerun_params_.save_rendered_mesh_eval_ =
        settings_file["Record.save_rendered_mesh_eval"].empty()
            ? true
            : (settings_file["Record.save_rendered_mesh_eval"].operator int()) != 0;
    rerun_params_.rerun_rendered_mesh_eval_ =
        !settings_file["Record.rerun_rendered_mesh_eval"].empty() &&
        (settings_file["Record.rerun_rendered_mesh_eval"].operator int()) != 0;
    rerun_params_.rendered_mesh_eval_voxel_size_m_ =
        settings_file["Record.rendered_mesh_eval_voxel_size_m"].empty()
            ? 0.05f
            : std::max(
                  1.0e-6f,
                  settings_file["Record.rendered_mesh_eval_voxel_size_m"].operator float());
    rerun_params_.rendered_mesh_eval_min_weight_ =
        settings_file["Record.rendered_mesh_eval_min_weight"].empty()
            ? 2.0f
            : std::max(
                  0.0f,
                  settings_file["Record.rendered_mesh_eval_min_weight"].operator float());
    rerun_params_.rendered_mesh_eval_trunc_vox_ =
        settings_file["Record.rendered_mesh_eval_trunc_vox"].empty()
            ? 8.0f
            : std::max(
                  1.0f,
                  settings_file["Record.rendered_mesh_eval_trunc_vox"].operator float());
    rerun_params_.rendered_mesh_eval_depth_max_m_ =
        settings_file["Record.rendered_mesh_eval_depth_max_m"].empty()
            ? 5.0f
            : std::max(
                  1.0e-6f,
                  settings_file["Record.rendered_mesh_eval_depth_max_m"].operator float());
    rerun_params_.svrecon_mesh_init_lv_ =
        settings_file["Record.svrecon_mesh_init_lv"].empty()
            ? 7
            : std::max(1, settings_file["Record.svrecon_mesh_init_lv"].operator int());
    rerun_params_.svrecon_mesh_final_lv_ =
        settings_file["Record.svrecon_mesh_final_lv"].empty()
            ? 10
            : std::max(1, settings_file["Record.svrecon_mesh_final_lv"].operator int());
    rerun_params_.svrecon_mesh_trunc_lv_ =
        settings_file["Record.svrecon_mesh_trunc_lv"].empty()
            ? 10
            : std::max(1, settings_file["Record.svrecon_mesh_trunc_lv"].operator int());
    rerun_params_.svrecon_mesh_trunc_vox_ =
        settings_file["Record.svrecon_mesh_trunc_vox"].empty()
            ? 5.0f
            : std::max(1.0e-6f, settings_file["Record.svrecon_mesh_trunc_vox"].operator float());
    rerun_params_.svrecon_mesh_pg_prune_ =
        settings_file["Record.svrecon_mesh_pg_prune"].empty()
            ? 0.6f
            : std::max(0.0f, settings_file["Record.svrecon_mesh_pg_prune"].operator float());
    rerun_params_.svrecon_mesh_crop_border_ =
        settings_file["Record.svrecon_mesh_crop_border"].empty()
            ? 0.01f
            : std::clamp(
                  settings_file["Record.svrecon_mesh_crop_border"].operator float(),
                  0.0f,
                  0.99f);
    rerun_params_.svrecon_mesh_alpha_thres_ =
        settings_file["Record.svrecon_mesh_alpha_thres"].empty()
            ? 0.5f
            : std::clamp(
                  settings_file["Record.svrecon_mesh_alpha_thres"].operator float(),
                  0.0f,
                  1.0f);
    rerun_params_.svrecon_mesh_use_mean_depth_ =
        !settings_file["Record.svrecon_mesh_use_mean_depth"].empty() &&
        (settings_file["Record.svrecon_mesh_use_mean_depth"].operator int()) != 0;
    rerun_params_.svrecon_mesh_use_vert_color_ =
        !settings_file["Record.svrecon_mesh_use_vert_color"].empty() &&
        (settings_file["Record.svrecon_mesh_use_vert_color"].operator int()) != 0;
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
    const bool any_rerun_recording_requested =
        rerun_params_.run_whole_run_ ||
        rerun_params_.rerun_svrecon_debug_ ||
        rerun_params_.rerun_gt_mesh_ ||
        rerun_params_.rerun_rendered_mesh_eval_ ||
        rerun_params_.rerun_reconstruction_mesh_ ||
        rerun_params_.rerun_maps_;
#if !PHOTOSLAM_ENABLE_RERUN
    // Deployment builds exclude Rerun regardless of values in desktop YAMLs.
    rerun_params_.enable_rerun_ = false;
#endif
    rerun_params_.enable_rerun_ =
        rerun_params_.enable_rerun_ && any_rerun_recording_requested;
    if (!rerun_params_.enable_rerun_) {
        rerun_params_.run_whole_run_ = false;
        rerun_params_.rerun_svrecon_debug_ = false;
        rerun_params_.rerun_gt_mesh_ = false;
        rerun_params_.rerun_nvblox_mesh_ = false;
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

void VoxelMapper::ensureEmbeddedPythonRuntime(bool import_torch_cuda)
{
    ensurePythonRuntimeInitialized(import_torch_cuda);
}

sv::LaptopPrecheckProfiler::Scope VoxelMapper::profileLaptopModule(
    const std::string& module,
    const std::uint64_t work_items)
{
    if (!laptop_precheck_profiler_) {
        return {};
    }
    return laptop_precheck_profiler_->profile(module, work_items);
}

void VoxelMapper::beginLaptopAsyncModule(
    const std::string& module,
    const std::uint64_t work_items)
{
    if (laptop_precheck_profiler_) {
        laptop_precheck_profiler_->beginAsync(module, work_items);
    }
}

void VoxelMapper::endLaptopAsyncModule(const std::string& module)
{
    if (laptop_precheck_profiler_) {
        laptop_precheck_profiler_->endAsync(module);
    }
}

std::string VoxelMapper::laptopPrecheckPipeline() const
{
    std::vector<std::string> modules;
    modules.emplace_back(allocate_orb_voxels_ ? "orb" : "no_orb_allocation");
    if (inactive_geo_densify_) {
        modules.emplace_back("inactive_geo");
    }
    if (rgbd_fill_render_holes_) {
        modules.emplace_back("rgbd_hole_fill");
    }
    if (rgbd_tsdf_evidence_) {
        modules.emplace_back("rgbd_tsdf_evidence");
    }
    if (monocular_rendered_depth_densify_) {
        modules.emplace_back("rendered_depth_evidence");
    }
    if (isMonocularMvsPipelineEnabled()) {
        modules.emplace_back(
            monocular_mvs_tsdf_evidence_
                ? "tandem_mvs_tsdf_evidence"
                : "tandem_mvs");
    }
    if (monocular_omnidata_densify_) {
        modules.emplace_back("omnidata");
    }

    std::ostringstream pipeline;
    for (std::size_t index = 0; index < modules.size(); ++index) {
        if (index > 0) {
            pipeline << '+';
        }
        pipeline << modules[index];
    }
    return pipeline.str();
}

void VoxelMapper::run()
{
    const auto run_start_time = std::chrono::steady_clock::now();
    sv::RerunVisualizerBridge::instance().setEnabled(rerun_params_.enable_rerun_);

    if (rerun_params_.enable_rerun_ && rerun_params_.rerun_gt_mesh_) {
        if (!rerun_params_.rerun_gt_mesh_path_.empty() &&
            std::filesystem::exists(rerun_params_.rerun_gt_mesh_path_)) {
            sv::RerunVisualizerBridge::instance().visualizePlyMesh(
                rerun_params_.rerun_gt_mesh_path_,
                0,
                "world/gt/mesh");
        } else {
            std::cerr << "[RERUN] GT mesh not found: "
                      << rerun_params_.rerun_gt_mesh_path_ << "\n";
        }
    }

    // First loop: Initial gaussian mapping
    while (!isStopped())
    {
        // Check conditions for initial mapping
        if (hasMetInitialMappingConditions())
        {
            auto initial_map_profile =
                profileLaptopModule("initial_map_build");
            mpSLAM->getAtlas()->clearMappingOperation();

            // Pull sparse SLAM map (get keyframes and map points)
            auto pMap = mpSLAM->getAtlas()->GetCurrentMap();
            std::vector<ORB_SLAM3::KeyFrame*> vKFs;
            std::vector<ORB_SLAM3::MapPoint*> vMPs;
            {
                std::unique_lock<std::mutex> lock_map(pMap->mMutexMapUpdate);
                vKFs = pMap->GetAllKeyFrames();
                vMPs = pMap->GetAllMapPoints();
                const unsigned long current_keyframe_id = pMap->GetMaxKFid();
                if (sensor_type_ == MONOCULAR) {
                    monocular_orb_inserted_point_ids_.clear();
                }
                for (const auto& pMP : vMPs)
                {
                     if (!pMP) {
                         continue;
                     }
                     if (sensor_type_ == MONOCULAR &&
                         !isMatureMonocularOrbMapPoint(
                             pMP, current_keyframe_id)) {
                         continue;
                     }
                     sv::Point3D point3D;
                     auto pos = pMP->GetWorldPos();
                     point3D.xyz_(0) = pos.x();
                     point3D.xyz_(1) = pos.y();
                     point3D.xyz_(2) = pos.z();
                     auto color = pMP->GetColorRGB();
                     point3D.color_(0) = color(0);
                     point3D.color_(1) = color(1);
                     point3D.color_(2) = color(2);
                     scene_->cachePoint3D(pMP->mnId, point3D);
                     if (sensor_type_ == MONOCULAR) {
                         monocular_orb_inserted_point_ids_.insert(pMP->mnId);
                     }
                 }
                // B) Create VoxelKeyframes from each SLAM KeyFrame
                for (const auto& pKF : vKFs)
                {
                    std::shared_ptr<VoxelKeyframe> new_kf = std::make_shared<VoxelKeyframe>(pKF->mnId, getIteration());
                    new_kf->source_timestamp_ = pKF->mTimeStamp;
                    new_kf->source_frame_id_ =
                        voxel_utils::parseFrameIdFromPath(pKF->mNameFile);
                    if (new_kf->source_frame_id_ < 0) {
                        new_kf->source_frame_id_ =
                            voxel_utils::frameIdFromIntegerTimestamp(
                                pKF->mTimeStamp);
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
                    if (this->sensor_type_ == RGBD) {
                        imgAux_undistorted = mapperDepthForKeyframe(
                            pKF->mNameFile, imgAux, camera);
                    } else {
                        imgAux_undistorted = imgAux;
                    }

                    new_kf->original_image_ =
                        voxel_utils::cvMatToTorchTensorFloat32(imgRGB_undistorted, device_type_);
                    new_kf->img_filename_ = pKF->mNameFile;
                    new_kf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
                    new_kf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
                    new_kf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;

                    // Compute transformations
                    // new_kf->computeTransformTensors(); //useless
                    scene_->addKeyframe(new_kf, &kfid_shuffled_);
                    latest_consumed_keyframe_id_.store(
                        std::max(
                            latest_consumed_keyframe_id_.load(std::memory_order_relaxed),
                            static_cast<long long>(pKF->mnId)),
                        std::memory_order_release);

                    increaseKeyframeTimesOfUse(new_kf, newKeyframeTimesOfUse());

                    // // Features for increasePcdByKeyframeInactiveGeoDensify
                    std::vector<float> pixels;
                    std::vector<float> pointsLocal;
                    pKF->GetKeypointInfo(pixels, pointsLocal);
                    new_kf->kps_pixel_ = std::move(pixels);
                    new_kf->kps_point_local_ = std::move(pointsLocal);
                    new_kf->img_undist_ = imgRGB_undistorted;
                    new_kf->img_auxiliary_undist_ = imgAux_undistorted;
                    if (isMonocularMvsPipelineEnabled() ||
                        monocular_omnidata_densify_) {
                        captureMonocularMvsKeyframeMetadata(new_kf, pKF);
                    }

                    logKeyframeCameraToRerunRecordings(
                        new_kf,
                        pKF->mnId,
                        /*log_reconstruction_mesh=*/true);

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
                            voxel_utils::cvGpuMatToTorchTensorFloat32(img_resized);
                    }
                }
                else {
                    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                        cv::Mat img_resized;
                        cv::resize(pkf->img_undist_, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                        pkf->gaus_pyramid_original_image_[l] =
                            voxel_utils::cvMatToTorchTensorFloat32(img_resized, device_type_);
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
            {
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                auto initial_topology_profile =
                    profileLaptopModule("initial_voxel_allocation");
                if (sensor_type_ == RGBD && !allocate_orb_voxels_) {
                    throw std::runtime_error(
                        "RGB-D mapping requires Mapper.allocate_orb_voxels=1 for "
                        "the initial topology.");
                }
                voxel_model_->createFromPcd(scene_->cached_point_cloud_, tr_cams);
                if (rerun_params_.run_whole_run_ ||
                    rerun_params_.rerun_svrecon_debug_) {
                    rerun_state_.whole_run_live_voxels_dirty_ = true;
                }
                initial_topology_profile.finish();
                std::unique_lock<std::mutex> lock(mutex_settings_);
                voxel_model_->createTrainer(
                                            opt_params_.geo_lr_,
                                            opt_params_.sh0_lr_,
                                            opt_params_.shs_lr_,
                                            opt_params_.optim_beta1_,
                                            opt_params_.optim_beta2_,
                                            opt_params_.optim_eps_,
                                            opt_params_.lr_decay_ckpt_,
                                            opt_params_.lr_decay_mult_,
                                            opt_params_.log_s_lr_);
            }

            logCurrentOrbMapPointsToReconstructionRerun(getIteration());
            logCurrentOrbKeyframePosesToReconstructionRerun(getIteration());

            const bool do_inactive_geo_densify =
                isdoingInactiveGeoDensify();
            const bool do_initial_rgbd_completion =
                sensor_type_ == RGBD &&
                ((rgbd_tsdf_evidence_ &&
                  rgbd_tsdf_evidence_initial_backfill_) ||
                 (rgbd_fill_render_holes_ &&
                  rgbd_fill_render_holes_initial_backfill_));
            if (do_inactive_geo_densify || do_initial_rgbd_completion) {
                std::vector<std::shared_ptr<VoxelKeyframe>> initial_rgbd_kfs;
                initial_rgbd_kfs.reserve(scene_->keyframes().size());
                for (const auto& kv : scene_->keyframes()) {
                    if (kv.second && !kv.second->done_inactive_geo_densify_) {
                        initial_rgbd_kfs.push_back(kv.second);
                    }
                }
                const int previous_depth_cache_limit = max_depth_cached_;
                max_depth_cached_ = std::max(
                    1,
                    depth_cached_ + static_cast<int>(initial_rgbd_kfs.size()));
                for (const auto& pkf : initial_rgbd_kfs) {
                    increasePcdByKeyframeInactiveGeoDensify(
                        pkf,
                        /*include_inactive_geo=*/do_inactive_geo_densify,
                        /*include_rgbd_hole_fill=*/
                            do_initial_rgbd_completion);
                }
                flushInactiveGeoCache();
                processRgbdClosureCache();
                max_depth_cached_ = previous_depth_cache_limit;
            }
            if (sensor_type_ == MONOCULAR &&
                monocular_rendered_depth_densify_) {
                for (const auto& kv : scene_->keyframes()) {
                    if (kv.second) {
                        densifyMonocularFromRenderedDepth(kv.second);
                    }
                }
            }
            if (sensor_type_ == RGBD &&
                sdf_initialization_rgbd_projective_) {
                int64_t fused_observations = 0;
                int fused_keyframes = 0;
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                for (const auto& kv : scene_->keyframes()) {
                    if (!kv.second) {
                        continue;
                    }
                    const int64_t observed =
                        fuseProjectiveSdfInitFromKeyframe(kv.second);
                    if (observed > 0) {
                        fused_observations += observed;
                        ++fused_keyframes;
                    }
                }
                std::cout
                    << "[SDF/RGBD init] keyframes=" << fused_keyframes
                    << " grid_corner_observations=" << fused_observations
                    << "\n";
            }
            if (rerun_params_.enable_rerun_ &&
                (rerun_params_.run_whole_run_ ||
                 rerun_params_.rerun_svrecon_debug_)) {
                logWholeRunLiveVoxelsToRerun(
                    getIteration(),
                    voxel_model_->voxCenter(),
                    voxel_model_->voxSize(),
                    torch::Tensor(),
                    rerun_params_.run_whole_run_,
                    rerun_params_.rerun_svrecon_debug_);
                rerun_state_.svrecon_debug_has_source_snapshot_ =
                    rerun_params_.rerun_svrecon_debug_;
                rerun_state_.whole_run_live_voxels_dirty_ = false;
            }
            if (opt_params_.prune_surface_views_enable_) {
                std::vector<std::shared_ptr<VoxelKeyframe>> initial_keyframes;
                initial_keyframes.reserve(scene_->keyframes().size());
                for (const auto& item : scene_->keyframes()) {
                    if (item.second) {
                        initial_keyframes.push_back(item.second);
                    }
                }
                markSurfaceViewPruningPending(initial_keyframes);
            }
            // One warm-up optimization step
            trainForOneIteration();

            initial_mapped_ = true;
            if (isMonocularMvsPipelineEnabled()) {
                std::vector<std::shared_ptr<VoxelKeyframe>> initial_mvs_kfs;
                initial_mvs_kfs.reserve(scene_->keyframes().size());
                for (const auto& item : scene_->keyframes()) {
                    if (item.second) {
                        initial_mvs_kfs.push_back(item.second);
                    }
                }
                scheduleLatestMonocularMvsKeyframe(initial_mvs_kfs);
            }
            if (monocular_omnidata_densify_) {
                std::vector<std::shared_ptr<VoxelKeyframe>> initial_kfs;
                initial_kfs.reserve(scene_->keyframes().size());
                for (const auto& item : scene_->keyframes()) {
                    if (item.second) {
                        initial_kfs.push_back(item.second);
                    }
                }
                scheduleLatestMonocularOmnidataKeyframe(initial_kfs);
            }
            input_backpressure_ready_.store(true, std::memory_order_release);
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
        pollMonocularMvsDensification();
        pollMonocularOmnidataDensification();
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
    
    // Third loop: Tail optimization. Match GaussianMapper: scheduled topology
    // adaptation remains active while the post-SLAM optimization finishes.
    pollMonocularMvsDensification(/*wait_for_result=*/true);
    pollMonocularOmnidataDensification(/*wait_for_result=*/true);
    if (monocular_mvs_backend_) {
        monocular_mvs_backend_.reset();
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
    if (monocular_omnidata_backend_) {
        monocular_omnidata_backend_.reset();
        c10::cuda::CUDACachingAllocator::emptyCache();
        std::cout
            << "[MONO/Omnidata] Released inference backend before tail "
               "refinement.\n";
    }
    flushInactiveGeoCache();
    processRgbdClosureCache();
    int adapt_interval = opt_params_.adapt_every_;          // cfg.procedure.adapt_every
    int n_delay_iters  = adapt_interval * 0.8f;        // same heuristic as GS code
    const bool prev_tail_refinement_active = tail_refinement_active_;
    tail_refinement_active_ = true;
    while (getIteration() - SLAM_stop_iter <= n_delay_iters
        || (getIteration() % adapt_interval) <= n_delay_iters
        || isKeepingTraining() )
    {
        trainForOneIteration();
        // Re-read in case user changed cfg at runtime
        adapt_interval = opt_params_.adapt_every_;
        n_delay_iters  = adapt_interval * 0.8f;
    }
    tail_refinement_active_ = prev_tail_refinement_active;

    runFinalRefinement();

    const double mapping_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - run_start_time).count();
    const VoxelRuntimeGpuStats runtime_gpu_stats =
        getVoxelRuntimeGpuStats();

    // Save and clear
    const std::filesystem::path shutdown_dir =
        result_dir_ / (std::to_string(getIteration()) + "_shutdown");
    auto shutdown_export_profile =
        profileLaptopModule("shutdown_export");
    if (!config_file_path_.empty() && std::filesystem::exists(config_file_path_)) {
        try {
            std::filesystem::create_directories(shutdown_dir);
            std::filesystem::path config_copy_name = config_file_path_.filename();
            if (config_copy_name.empty()) {
                config_copy_name = "voxel_mapper.yaml";
            }
            std::filesystem::copy_file(
                config_file_path_,
                shutdown_dir / config_copy_name,
                std::filesystem::copy_options::overwrite_existing);
        } catch (const std::exception& e) {
            std::cerr << "[VoxelMapper] Failed to copy config file to shutdown folder: "
                      << e.what() << "\n";
        }
    }
    renderAndRecordAllKeyframes("_shutdown");
    savePly(shutdown_dir / "ply");
    {
        const std::filesystem::path ply_dir =
            shutdown_dir / "ply" / "voxel_model" /
            ("iteration_" + std::to_string(getIteration()));
        const std::filesystem::path eval_mesh_path = ply_dir / "voxel_surface_mesh.ply";
        const std::filesystem::path sdf_mesh_path =
            ply_dir / "voxel_surface_mesh_sdf.ply";
        const std::filesystem::path rendered_tsdf_mesh_path =
            ply_dir / "voxel_surface_mesh_rendered_tsdf.ply";

        const bool want_rendered_mesh =
            rerun_params_.save_rendered_mesh_eval_ ||
            (rerun_params_.enable_rerun_ &&
             (rerun_params_.rerun_rendered_mesh_eval_ ||
              rerun_params_.rerun_reconstruction_mesh_));
        if (want_rendered_mesh) {
            try {
                std::ofstream mesh_info(ply_dir / "mesh_reconstructions.txt");
                if (!mesh_info) {
                    throw std::runtime_error("failed to open mesh_reconstructions.txt");
                }
                mesh_info
                    << "voxel_model.ply: Native SVRecon octree cells with SH/color and corner SDF; viewer/model file, not a triangle mesh.\n"
                    << "voxel_surface_mesh.ply: Fixed-resolution, keyframe-weighted TSDF fusion of alpha-valid final rendered depths; HI-SLAM2-comparable mesh.\n"
                    << "voxel_surface_mesh_rendered_tsdf.ply: Progressive SVRecon rendered-depth TSDF fusion followed by zero-level extraction.\n"
                    << "voxel_surface_mesh_sdf.ply: Direct zero-level Marching Cubes extraction from the optimized SVRecon corner SDF.\n";
            } catch (const std::exception& e) {
                std::cerr << "[mesh/info] shutdown export failed: "
                          << e.what() << "\n";
            }

            // Common SVRecon evaluation path: fixed-resolution TSDF
            // fusion of final rendered depths.
            try {
                saveRenderedTsdfMeshPly(eval_mesh_path);
                if (rerun_params_.enable_rerun_ &&
                    rerun_params_.rerun_reconstruction_mesh_ &&
                    std::filesystem::exists(eval_mesh_path)) {
                    sv::RerunVisualizerBridge::instance().visualizeDebugPlyMesh(
                        "reconstruction_mesh",
                        eval_mesh_path.string(),
                        getIteration(),
                        "world/mesh/final");
                }
            } catch (const std::exception& e) {
                std::cerr << "[mesh/rendered-TSDF-fixed] shutdown export failed: "
                          << e.what() << "\n";
            }
            c10::cuda::CUDACachingAllocator::emptyCache();

            // SVRecon extract_mesh.py --adaptive: render training views,
            // fuse median rendered depth into a TSDF, then extract iso=0.
            try {
                saveSvreconRenderedTsdfMeshPly(rendered_tsdf_mesh_path);
                if (rerun_params_.enable_rerun_ && rerun_params_.rerun_rendered_mesh_eval_ &&
                    std::filesystem::exists(rendered_tsdf_mesh_path)) {
                    sv::RerunVisualizerBridge::instance().visualizePlyMesh(
                        rendered_tsdf_mesh_path.string(),
                        getIteration(),
                        "world/voxel_model_mesh/final");
                }
            } catch (const std::exception& e) {
                std::cerr << "[SVRecon mesh/rendered-TSDF] shutdown export failed: "
                          << e.what() << "\n";
            }
            c10::cuda::CUDACachingAllocator::emptyCache();

            // SVRecon extract_mesh_sdf.py: direct zero-level extraction from
            // the optimized grid SDF.
            try {
                saveSvreconSdfMeshPly(sdf_mesh_path);
            } catch (const std::exception& e) {
                std::cerr << "[SVRecon mesh/SDF] shutdown export failed: "
                          << e.what() << "\n";
            }
        }
    }
    writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

    const int runtime_frames =
        runtime_frame_count_.load(std::memory_order_acquire);
    const std::filesystem::path final_map_path =
        shutdown_dir / "ply" / "voxel_model" /
        ("iteration_" + std::to_string(getIteration())) /
        "voxel_model.ply";
    if (runtime_frames > 0) {
        saveVoxelRuntimeMetrics(
            shutdown_dir / "runtime_metrics.json",
            runtime_frames,
            scene_ ? static_cast<int>(scene_->keyframes().size()) : 0,
            voxel_model_ ? voxel_model_->numVoxels() : 0,
            getIteration(),
            mapping_seconds,
            final_map_path,
            runtime_gpu_stats);
    }

    shutdown_export_profile.finish();
    if (laptop_precheck_profiler_ && laptop_precheck_profiler_->enabled()) {
        laptop_precheck_profiler_->stop();
        sv::LaptopPrecheckMetadata metadata;
        metadata.config_file = config_file_path_;
        metadata.map_path = final_map_path;
        metadata.sensor =
            sensor_type_ == MONOCULAR
                ? "monocular"
                : (sensor_type_ == STEREO ? "stereo" : "rgbd");
        metadata.device = device_type_ == torch::kCUDA ? "cuda" : "cpu";
        metadata.pipeline = laptopPrecheckPipeline();
        metadata.frames = runtime_frames;
        metadata.keyframes =
            scene_ ? static_cast<int>(scene_->keyframes().size()) : 0;
        metadata.voxels = voxel_model_ ? voxel_model_->numVoxels() : 0;
        metadata.iterations = getIteration();
        metadata.mapping_seconds = mapping_seconds;
        metadata.map_size_mb = voxelFileSizeMb(final_map_path);
        try {
            laptop_precheck_profiler_->writeReports(shutdown_dir, metadata);
            std::cout
                << "[Laptop Precheck] saved: "
                << (shutdown_dir / "laptop_precheck.json") << "\n";
        } catch (const std::exception& error) {
            std::cerr
                << "[Laptop Precheck] failed to save report: "
                << error.what() << "\n";
        }
    }

    alignAndLogNvbloxReferenceMesh(shutdown_dir);
    logLearnedDepthMapsToWholeRunRerun();
    saveRerunRecordingsAtShutdown();

    signalStop();
}

void VoxelMapper::setRuntimeFrameCount(int frame_count)
{
    runtime_frame_count_.store(
        std::max(0, frame_count),
        std::memory_order_release);
}

// Add training with optimization loop
void VoxelMapper::trainForOneIteration()
{
    auto iteration_profile =
        profileLaptopModule("training_iteration");
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
    }
    voxel_model_->setShDegree(default_sh_);

    // Keep the global SDF sharpness fixed at the value initialized from the
    // base voxel size. A local subdivision must not sharpen every remaining
    // coarse cell in the mixed-resolution online map.

    // Match SVRecon train.py: keep ss=1.0 early, then use augmentation
    // or remove ss so the model default is used.
    ropts.ss = 1.0f;
    if (iter > 1000) {
        if (opt_params_.ss_aug_max_ > 1.0f) {
            static thread_local std::mt19937 rng{std::random_device{}()};
            std::uniform_real_distribution<float> dist(1.0f, opt_params_.ss_aug_max_);
            ropts.ss = dist(rng);
        } else {
            ropts.ss = std::nullopt;
        }
    }

    const bool need_sparse_depth = (opt_params_.lambda_sparse_depth_ > 0.0f) && (iter <= opt_params_.sparse_depth_until_);
    const bool need_rgbd_depth =
        (sensor_type_ == RGBD) &&
        (opt_params_.lambda_rgbd_depth_ > 0.0f) &&
        (iter >= opt_params_.rgbd_depth_from_) &&
        (iter <= opt_params_.rgbd_depth_end_);
    const bool need_rgbd_sdf =
        (sensor_type_ == RGBD) &&
        (opt_params_.lambda_rgbd_sdf_ > 0.0f) &&
        (iter >= opt_params_.rgbd_sdf_from_) &&
        (iter <= opt_params_.rgbd_sdf_end_);
    const bool need_rgbd_mask = sensor_type_ == RGBD;
    const bool need_rgbd_normal =
        (sensor_type_ == RGBD) &&
        (opt_params_.lambda_rgbd_normal_ > 0.0f) &&
        (iter >= opt_params_.rgbd_normal_from_) &&
        (iter <= opt_params_.rgbd_normal_end_);
    const bool has_monocular_depth_prior =
        sensor_type_ == MONOCULAR &&
        viewpoint_cam->monocular_depth_source_ !=
            sv::LearnedDepthSource::None &&
        !viewpoint_cam->monocular_depth_prior_.empty() &&
        !viewpoint_cam->monocular_depth_confidence_.empty();
    const bool need_monocular_depth =
        has_monocular_depth_prior &&
        opt_params_.lambda_monocular_depth_ > 0.0f &&
        iter >= opt_params_.monocular_depth_from_ &&
        iter <= opt_params_.monocular_depth_end_;
    const bool need_monocular_normal =
        has_monocular_depth_prior &&
        opt_params_.lambda_monocular_normal_ > 0.0f &&
        iter >= opt_params_.monocular_normal_from_ &&
        iter <= opt_params_.monocular_normal_end_;
    const bool need_T_concen = (opt_params_.lambda_T_concen_ > 0.0f);
    const bool need_T_inside = (opt_params_.lambda_T_inside_ > 0.0f);
    const bool need_normal_dmean =
        (opt_params_.lambda_normal_dmean_ > 0.0f) &&
        (iter >= opt_params_.n_dmean_from_) &&
        (iter <= opt_params_.n_dmean_end_);
    ropts.output_T =
        need_T_concen || need_T_inside || need_sparse_depth || need_normal_dmean ||
        need_rgbd_depth || need_rgbd_normal || need_rgbd_mask ||
        need_monocular_depth || need_monocular_normal;
    ropts.output_depth =
        need_sparse_depth || need_normal_dmean || need_rgbd_depth ||
        need_monocular_depth;
    ropts.output_normal =
        need_normal_dmean || need_rgbd_normal || need_monocular_normal;

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

    sv::MiniCam cam = viewpoint_cam->toMiniCam(image_height, image_width);

    std::unordered_map<std::string, torch::Tensor> render_pkg;
    {
        auto render_profile =
            profileLaptopModule("training_render");
        render_pkg = voxel_model_->render(
            cam,
            image_height,
            image_width,
            /* gt_image   */  gt_image,
            /* color_mode   */   nullptr,
            /* track_max_w   */  false,
            /* ss            */  ropts.ss,
            /* output_depth  */  ropts.output_depth,
            /* output_normal */  ropts.output_normal,
            /* output_T      */  ropts.output_T,
            /* rand_bg       */  false,
            /* use_auto_exp  */  false,
            ropts               // your struct (will be used for **other_opt-safe fields)
        );
    }
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

    auto Ll1 = voxel_eval::l1Loss(masked_image, gt_image);
    auto mse = voxel_eval::mseLoss(masked_image, gt_image);

    // Match SVRecon's base photometric loss selection: L1, Huber, or MSE.
    torch::Tensor photo_loss;
    const char* photo_loss_name = nullptr;
    if (opt_params_.use_l1_) {
        photo_loss = Ll1;
        photo_loss_name = "L1";
    } else if (opt_params_.use_huber_) {
        photo_loss = voxel_eval::huberLoss(masked_image, gt_image, opt_params_.huber_thres_);
        photo_loss_name = "Huber";
    } else {
        photo_loss = mse;
        photo_loss_name = "MSE";
    }
    auto loss = photo_loss.clone();

    // --- Optional sparse/RGB-D depth regularization -----------------------------
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
    if (need_rgbd_depth) {
        torch::Tensor rgbd_depth_loss =
            computeRgbdDepthLoss(viewpoint_cam, cam, render_pkg, iter);

        loss = loss + opt_params_.lambda_rgbd_depth_ * rgbd_depth_loss;
    }
    if (need_rgbd_mask) {
        // SVRecon's lambda_mask supervises final transmittance using its
        // foreground mask. For RGB-D SLAM, valid measured depth is the
        // corresponding foreground observation.
        constexpr float kRgbdForegroundMaskWeight = 0.1f;
        torch::Tensor rgbd_mask_loss =
            computeRgbdMaskLoss(viewpoint_cam, cam, render_pkg);
        loss = loss + kRgbdForegroundMaskWeight * rgbd_mask_loss;
    }
    if (need_rgbd_sdf) {
        torch::Tensor rgbd_sdf_loss =
            computeRgbdSdfLoss(viewpoint_cam, cam, iter);

        loss = loss + opt_params_.lambda_rgbd_sdf_ * rgbd_sdf_loss;
    }
    if (need_rgbd_normal) {
        torch::Tensor rgbd_normal_loss =
            computeRgbdNormalLoss(viewpoint_cam, cam, render_pkg, iter);

        loss = loss + opt_params_.lambda_rgbd_normal_ * rgbd_normal_loss;
    }
    torch::Tensor monocular_depth_loss;
    if (need_monocular_depth) {
        monocular_depth_loss = computeMonocularDepthLoss(
            viewpoint_cam, render_pkg, iter);
        loss = loss +
            opt_params_.lambda_monocular_depth_ * monocular_depth_loss;
    }
    torch::Tensor monocular_normal_loss;
    if (need_monocular_normal) {
        monocular_normal_loss = computeMonocularNormalLoss(
            viewpoint_cam, render_pkg, iter);
        loss = loss +
            opt_params_.lambda_monocular_normal_ * monocular_normal_loss;
    }
    torch::Tensor ssim_loss;
    if (opt_params_.lambda_ssim_ > 0.0f) {
        ssim_loss = voxel_eval::fastSsimLoss(masked_image, gt_image);
        loss += opt_params_.lambda_ssim_ * ssim_loss;
    }

    if (need_T_concen || need_T_inside) {
        auto it = render_pkg.find("raw_T");
        if (it != render_pkg.end() && it->second.defined()) {
            torch::Tensor raw_T = it->second;

            // SVRecon: loss += lambda_T_concen * prob_concen_loss(raw_T)
            if (need_T_concen) {
                torch::Tensor reg_concen = voxel_eval::probabilityConcentrationLoss(raw_T);
                loss = loss + opt_params_.lambda_T_concen_ * reg_concen;
            }

            // SVRecon: loss += lambda_T_inside * raw_T.square().mean()
            if (need_T_inside) {
                torch::Tensor reg_inside = raw_T.pow(2).mean();
                loss = loss + opt_params_.lambda_T_inside_ * reg_inside;
            }
        }
    }

    if (need_normal_dmean) {
        auto reg_normal_dmean = voxel_eval::normalDepthConsistencyLossSvrecon(
            cam,
            render_pkg,
            opt_params_.n_dmean_ks_,
            opt_params_.n_dmean_tol_deg_);
        loss = loss + opt_params_.lambda_normal_dmean_ * reg_normal_dmean;
    }

    // SVRecon applies local Eikonal regularization after the hierarchy reaches
    // inside level 9. The paper uses 1e-11 here versus 1e-8 for the coarse
    // global Eikonal term.
    const bool eikonal_enabled =
        !tail_refinement_active_ && opt_params_.lambda_ge_density_ > 0.f;
    if (eikonal_enabled) {
        loss = loss + voxel_model_->svreconLocalEikonalLoss(
            opt_params_.lambda_ge_density_ * 1.0e-3f,
            /*min_inside_level=*/9);
    }

    {
        auto optimizer_profile =
            profileLaptopModule("training_backward_optimizer");
        voxel_model_->optimizerZeroGrad();
        loss.backward();

        if (opt_params_.lambda_tv_density_ > 0.f &&
            iter >= opt_params_.tv_from_ &&
            iter <= opt_params_.tv_until_) {
            voxel_model_->applyTvOnDensityField(opt_params_.lambda_tv_density_);
        }
        if (eikonal_enabled &&
            iter >= opt_params_.ge_from_ &&
            iter <= opt_params_.ge_until_) {
            const float ge_mult = std::pow(0.25f, std::min(iter / 2000, 2));
            voxel_model_->applySvreconGridEikonalField(
                opt_params_.lambda_ge_density_ * ge_mult);
        }
        if (opt_params_.lambda_ls_density_ > 0.f &&
            iter >= opt_params_.ls_from_ &&
            iter <= opt_params_.ls_until_) {
            const float ls_mult = std::pow(0.25f, std::min(iter / 2000, 2));
            voxel_model_->applySvreconLaplacianSmoothnessField(
                opt_params_.lambda_ls_density_ * ls_mult);
        }

        if (iter >= 500) {
            voxel_model_->accumulateSubdivisionPriority();
        }

        voxel_model_->optimizerStep();
    }

    runPendingSurfaceViewPruning();

    // This is a live diagnostic of the learned SDF, independent of the pruning
    // schedule. Recompute it after every optimizer step so its Rerun timeline
    // shows cells entering and leaving the current SVRecon prune set.
    if (rerun_params_.enable_rerun_ &&
        rerun_params_.rerun_svrecon_debug_) {
        torch::NoGradGuard no_grad;
        logSvreconDebugVoxelMaskToRerun(
            iter,
            computeSvreconSdfPruneMask(),
            "world/svrecon/sdf_prune_candidates");
    }

    if (!disable_topology_changes_) {
        // Densification for increasePcd
        const int prune_every =
            std::max(1, (opt_params_.prune_every_ > 0) ? opt_params_.prune_every_ : opt_params_.adapt_every_);
        const int subdivide_every =
            std::max(1, (opt_params_.subdivide_every_ > 0) ? opt_params_.subdivide_every_ : opt_params_.adapt_every_);
        const int prune_from = opt_params_.prune_from_;
        const int subdivide_from = opt_params_.subdivide_from_;
        const bool meet_prune_period =
            (iter >= prune_from) && (iter % prune_every == 0);
        const bool meet_subdivide_period =
            (iter >= subdivide_from) && (iter % subdivide_every == 0);

        bool need_pruning =
            meet_prune_period && (iter <= opt_params_.prune_until_);
        bool need_subdividing = meet_subdivide_period;
        need_pruning = need_pruning && (iter <= opt_params_.iterations_ - 500);

        if (need_pruning || need_subdividing)
        {
            auto topology_profile =
                profileLaptopModule("topology_adaptation");
            // HI-SLAM2 performs online map adaptation on its current window.
            // A zero window keeps the established full-history behavior.
            std::vector<sv::MiniCam> tr_cams = incrementalMappingCameras();

            auto stat = voxel_model_->computeTrainingStat(tr_cams);
            auto sched_state = voxel_model_->schedulerState();
            auto flatten_colvec = [](torch::Tensor t) {
                if (t.defined() && t.dim() == 2 && t.size(1) == 1) {
                    t = t.squeeze(1);
                }
                return t.contiguous().view({-1});
            };

            // ---------------- PRUNE ----------------
            auto run_pruning = [&]() {
                auto pruning_profile =
                    profileLaptopModule("pruning");
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
                const int prune_all_iter =
                    std::max(1, opt_params_.prune_until_ - prune_every);
                const int prune_now_iter =
                    std::max(0, iter - prune_every);
                const float prune_iter_rate =
                    std::clamp(
                        static_cast<float>(prune_now_iter) /
                            static_cast<float>(prune_all_iter),
                        0.0f,
                        1.0f);
                const float prune_thres =
                    opt_params_.prune_thres_init_ +
                    std::max(0.0f,
                             opt_params_.prune_thres_final_ -
                                 opt_params_.prune_thres_init_) *
                        prune_iter_rate;
                // Separate threshold for at-target voxels.
                const float prune_thres_at_target = opt_params_.prune_thres_final_at_target_;
                const int ori_n = voxel_model_->numVoxels();
                const int N     = ori_n;
                const float target_vox_size = voxel_model_->fixedVoxSize();
                torch::Tensor prune_mask_vis; // [N] bool, set when visibility filter runs
                torch::Tensor observed_layout_mask; // [N] bool, voxels observed by the pruning camera set
                torch::Tensor prune_mask_near; // [N] bool, near-camera prune (unprotected)
                torch::Tensor prune_mask_far; // [N] bool, outside dense-core bound prune
                torch::Tensor prune_mask_mvs_supported; // [N] bool, MVS-confirmed surface cells
                torch::Tensor prune_mask_mvs_protected; // [N] bool, soft candidates preserved by MVS
                torch::Tensor prune_mask_mvs_free_space; // [N] bool, multi-view free-space contradictions
                torch::Tensor prune_mask_mvs_free_space_extra; // [N] bool, new MVS candidates
                torch::Tensor prune_mask_default;         // [N] bool, default rules only
                int64_t mvs_prune_depth_keyframes = 0;
                int64_t mvs_prune_valid_projections = 0;

                torch::Device prune_device = mDevice;
                torch::Tensor centers_for_device = voxel_model_->voxCenter();
                if (centers_for_device.defined()) {
                    prune_device = centers_for_device.device();
                }
                auto bool_opts_prune =
                    torch::TensorOptions().dtype(torch::kBool).device(prune_device);
                auto real_mask_for_base = torch::ones({N}, bool_opts_prune);
                auto in_target_size_mask = torch::zeros({N}, bool_opts_prune);
                auto prune_mask_base_real_raw = torch::zeros({N}, bool_opts_prune);
                auto prune_mask_base_real = torch::zeros({N}, bool_opts_prune);
                auto prune_mask_base_real_at_target = torch::zeros({N}, bool_opts_prune);
                auto prune_mask_base_real_at_target_extra = torch::zeros({N}, bool_opts_prune);
                torch::Tensor prune_mask_base = torch::zeros({N}, bool_opts_prune);
                auto prune_mask = prune_mask_base.clone();
                const int n_prune_base = 0;

                torch::Tensor sdf_prune_mask;
                torch::Tensor tsdf_surface_protect_mask;
                torch::Tensor svrecon_known_voxel_mask;
                float svrecon_surface_band_m =
                    std::max(1.0e-6f, 2.0f * sdfMetricVoxelSize());
                // 2) SVRecon SDF pruning.
                if (N > 0) {
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
                                // Gather SDF at the 8 voxel grid corners. In SVRecon mode
                                // the optimized geo grid is the SDF field.
                                torch::Tensor tsdf8 =
                                    voxel_model_->voxelGeoCorners()
                                        .to(prune_mask.device())
                                        .to(torch::kFloat32)
                                        .contiguous();

                                // Device alignment (should already match)
                                if (tsdf8.device() != prune_mask.device()) {
                                    tsdf8 = tsdf8.to(prune_mask.device());
                                }
                                torch::Tensor corner_valid;
                                torch::Tensor tsdf_unknown_mask;
                                torch::Tensor tsdf_free_mask;
                                torch::Tensor tsdf_occupied_mask;
                                torch::Tensor tsdf_surface_mask;

                                // Keep only an actual learned-SDF surface: the voxel must
                                // have finite corners on both strict sides of zero.
                                corner_valid = torch::isfinite(tsdf8).to(torch::kBool);
                                torch::Tensor valid_count =
                                    corner_valid.to(torch::kInt32).sum(/*dim=*/1);
                                torch::Tensor voxel_valid = (valid_count > 0).to(torch::kBool);
                                svrecon_known_voxel_mask =
                                    voxel_valid.to(prune_mask.device()).to(torch::kBool).contiguous();
                                torch::Tensor has_pos =
                                    ((tsdf8 > 0.0f) & corner_valid).any(/*dim=*/1);
                                torch::Tensor has_neg =
                                    ((tsdf8 < 0.0f) & corner_valid).any(/*dim=*/1);
                                torch::Tensor has_surface =
                                    ((valid_count == tsdf8.size(1)) & has_pos & has_neg)
                                        .to(torch::kBool);

                                    torch::Tensor abs_sdf = tsdf8.abs();
                                    torch::Tensor inf_like =
                                        torch::full_like(abs_sdf, std::numeric_limits<float>::infinity());
                                    torch::Tensor abs_valid =
                                        torch::where(corner_valid, abs_sdf, inf_like);
                                    torch::Tensor min_abs_sdf =
                                        std::get<0>(abs_valid.min(/*dim=*/1));

                                    float global_vox_size_min =
                                        std::max(1.0e-6f, voxel_model_->fixedVoxSize());
                                    torch::Tensor vox_size_for_thresh = voxel_model_->voxSize();
                                    if (vox_size_for_thresh.defined() && vox_size_for_thresh.numel() > 0) {
                                        global_vox_size_min = std::max(
                                            1.0e-6f,
                                            vox_size_for_thresh
                                                .to(prune_mask.device())
                                                .to(torch::kFloat32)
                                                .min()
                                                .item<float>());
                                    }
                                    torch::Tensor log_s =
                                        voxel_model_->svreconLogS()
                                            .to(prune_mask.device())
                                            .to(torch::kFloat32);
                                    torch::Tensor sdf_thresh_t =
                                        torch::log(torch::tensor(
                                            199.0f,
                                            torch::TensorOptions()
                                                .dtype(torch::kFloat32)
                                                .device(prune_mask.device()))) /
                                        torch::exp(10.0f * log_s);
                                    const float sdf_thresh = std::max(
                                        2.0f * global_vox_size_min,
                                        sdf_thresh_t.reshape({-1})[0].item<float>());
                                    svrecon_surface_band_m = sdf_thresh;

                                    torch::Tensor inside_mask =
                                        torch::ones({N},
                                            torch::TensorOptions()
                                                .dtype(torch::kBool)
                                                .device(prune_mask.device()));
                                    torch::Tensor scene_center = voxel_model_->SceneCenter();
                                    torch::Tensor inside_extent = voxel_model_->InsideExtent();
                                    if (!inside_extent.defined() || inside_extent.numel() == 0) {
                                        inside_extent = voxel_model_->SceneExtent();
                                    }
                                    if (scene_center.defined() &&
                                        inside_extent.defined() &&
                                        scene_center.numel() == 3 &&
                                        inside_extent.numel() > 0) {
                                        scene_center =
                                            scene_center.to(prune_mask.device())
                                                .to(torch::kFloat32)
                                                .contiguous()
                                                .view({1, 3});
                                        const float extent =
                                            std::max(1.0e-6f,
                                                     inside_extent
                                                         .to(prune_mask.device())
                                                         .to(torch::kFloat32)
                                                         .reshape({-1})[0]
                                                         .item<float>());
                                        torch::Tensor half_extent =
                                            torch::full({1, 3}, 0.5f * extent,
                                                torch::TensorOptions()
                                                    .dtype(torch::kFloat32)
                                                    .device(prune_mask.device()));
                                        torch::Tensor centers_for_inside =
                                            centers_world.to(prune_mask.device()).to(torch::kFloat32);
                                        inside_mask =
                                            ((centers_for_inside >= (scene_center - half_extent)).all(/*dim=*/1) &
                                             (centers_for_inside <= (scene_center + half_extent)).all(/*dim=*/1))
                                                .to(torch::kBool);
                                    }

                                    torch::Tensor near_surface =
                                        (min_abs_sdf <= sdf_thresh).to(torch::kBool);
                                    torch::Tensor all_positive =
                                        (voxel_valid & has_pos & (~has_neg)).to(torch::kBool);
                                    torch::Tensor all_negative =
                                        (voxel_valid & has_neg & (~has_pos)).to(torch::kBool);
                                    tsdf_unknown_mask = (~voxel_valid).to(torch::kBool);
                                    tsdf_free_mask =
                                        (all_positive & (~near_surface) & inside_mask).to(torch::kBool);
                                    tsdf_occupied_mask =
                                        (all_negative & (~near_surface) & inside_mask).to(torch::kBool);
                                    tsdf_surface_mask = has_surface.to(torch::kBool);

                                    // SVRecon SDF pruning: preserve both actual sign-changing
                                    // cells and the near-zero band in which a surface can still
                                    // move or form. Online SLAM has no fixed offline foreground
                                    // extent, so apply the same rule to every allocated cell.
                                    sdf_prune_mask =
                                        computeSvreconSdfPruneMask()
                                            .to(prune_mask.device())
                                            .to(torch::kBool);
                                tsdf_surface_protect_mask =
                                    tsdf_surface_mask.to(prune_mask.device()).to(torch::kBool).contiguous();

                                // Ensure device matches prune_mask
                                if (sdf_prune_mask.device() != prune_mask.device()) {
                                    sdf_prune_mask = sdf_prune_mask.to(prune_mask.device());
                                }
                                prune_mask = sdf_prune_mask.to(torch::kBool);

                                // 2) Cache TSDF class samples only when the Rerun debug recording needs them.
                                auto zero_base = torch::zeros(
                                    {N},
                                    torch::TensorOptions()
                                        .dtype(torch::kBool)
                                        .device(prune_mask.device()));
                                prune_mask_base = zero_base.clone();
                                prune_mask_base_real = zero_base.clone();
                                prune_mask_base_real_at_target = zero_base.clone();
                                prune_mask_base_real_at_target_extra = zero_base.clone();
                            }
                    } catch (const std::exception&) {}
                }

                int64_t n_prune_near_front = 0;
                int64_t n_prune_near_geom = 0;

                // 3) Layout visibility / near filtering. This mirrors the
                // SVRecon octlayout_filtering utility: keep voxels with
                // mark_max_samp_rate > 0 and optionally remove mark_near hits.
                if (!tr_cams.empty() && N > 0) {
                    try {
                            at::Tensor octpath = voxel_model_->octPath().contiguous();      // [N,1] int64
                            at::Tensor L = voxel_model_->octLevel().contiguous();           // [N,1] int8
                            at::Tensor vox_center = voxel_model_->voxCenter().contiguous(); // [N,3]
                            at::Tensor vox_size = voxel_model_->voxSize().contiguous();     // [N,1]

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

                            auto Nu_before = octpath.size(0);
                            TORCH_CHECK(Nu_before == N,
                                        "octpath.size(0) != N before visibility filter");

                            // 1) visibility: rate > 0
                            at::Tensor rate =
                                sv::markSvreconMaxSampRateDirect(tr_cams, octpath, vox_center, vox_size);

                            if (rate.dim() == 2 && rate.size(1) == 1)
                                rate = rate.squeeze(1);
                            rate = rate.to(torch::kFloat32);

                            at::Tensor keep_rate = (rate > 0.0f).to(torch::kBool);   // [N]
                            int64_t n_rate_pos = keep_rate.sum().item<int64_t>();

                            // 2) near filtering:
                            //    a) SVRecon mark_near (camera-facing)
                            //    b) legacy geometric distance-to-camera test
                            //       for the old density path only.
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
                                is_near =
                                    sv::markSvreconNearDirect(tr_cams, octpath, vox_center, vox_size, near_thresh);
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

                            auto prune_near_union = torch::zeros_like(is_near);
                            if (opt_params_.filter_near_voxels_) {
                                prune_near_union =
                                    (prune_near_union | is_near).to(torch::kBool);
                            }
                            if (opt_params_.prune_near_voxels_geometric_) {
                                prune_near_union =
                                    (prune_near_union | is_near_geom).to(torch::kBool);
                            }
                            n_prune_near_front = n_near_hit;
                            n_prune_near_geom = n_near_geom_hit;

                            keep_rate = keep_rate.view({-1}).to(torch::kBool);    // [N]
                            observed_layout_mask =
                                keep_rate.to(prune_mask.device()).to(torch::kBool).contiguous();
                            prune_mask_vis =
                                (~keep_rate).to(torch::kBool);     // [N], visibility only
                            prune_mask_near =
                                prune_near_union.view({-1}).to(torch::kBool); // [N], near only

                            // Visibility remains diagnostic. Configured near
                            // filtering is applied below as a concrete prune cause.

                    } catch (const std::exception& e) {
                        std::cerr << "[PRUNE/visibility] exception: " << e.what() << "\n";
                    }
                }

                if (opt_params_.prune_far_voxels_ && N > 0) {
                    try {
                        voxel_model_->refreshDenseCoreBBFromCurrentVoxels();
                        if (voxel_model_->hasDenseCoreBB()) {
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
                                prune_mask_far = (~in_dense_core.to(torch::kBool)).to(torch::kBool);
                            }
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "[PRUNE/far] exception: " << e.what() << "\n";
	                    }
		                }

                if (prune_mask_near.defined() &&
                    prune_mask_near.numel() == N) {
                    prune_mask =
                        (prune_mask.to(torch::kBool) |
                         prune_mask_near.to(prune_mask.device())
                             .to(torch::kBool))
                            .contiguous();
                }
                if (opt_params_.prune_far_voxels_ &&
                    prune_mask_far.defined() &&
                    prune_mask_far.numel() == N) {
                    prune_mask =
                        (prune_mask.to(torch::kBool) |
                         prune_mask_far.to(prune_mask.device())
                             .to(torch::kBool))
                            .contiguous();
                }

                if (opt_params_.prune_mvs_consistency_enable_ &&
                    sensor_type_ == MONOCULAR && N > 0) {
                    auto mvs_pruning_profile =
                        profileLaptopModule("mvs_pruning_consistency");
                    try {
                        const torch::Tensor centers_world =
                            voxel_model_->voxCenter();
                        const torch::Tensor sizes_world =
                            voxel_model_->voxSize();
                        sv::MonocularMvsPruneEvidence mvs_evidence =
                            computeMonocularMvsPruneEvidence(
                                centers_world, sizes_world);
                        if (mvs_evidence.supported.defined() &&
                            mvs_evidence.free_space.defined() &&
                            mvs_evidence.supported.numel() == N &&
                            mvs_evidence.free_space.numel() == N) {
                            prune_mask_mvs_supported =
                                mvs_evidence.supported
                                    .to(prune_mask.device())
                                    .to(torch::kBool)
                                    .contiguous();
                            prune_mask_mvs_free_space =
                                mvs_evidence.free_space
                                    .to(prune_mask.device())
                                    .to(torch::kBool)
                                    .contiguous();
                            mvs_prune_depth_keyframes =
                                mvs_evidence.depth_keyframes;
                            mvs_prune_valid_projections =
                                mvs_evidence.valid_projections;

                            auto zero_mask = torch::zeros_like(
                                prune_mask.to(torch::kBool));
                            torch::Tensor soft_candidates = zero_mask.clone();
                            if (sdf_prune_mask.defined() &&
                                sdf_prune_mask.numel() == N) {
                                soft_candidates =
                                    soft_candidates |
                                    sdf_prune_mask.to(prune_mask.device())
                                        .to(torch::kBool);
                            }
                            torch::Tensor hard_candidates =
                                prune_mask_base.to(prune_mask.device())
                                    .to(torch::kBool)
                                    .clone();
                            if (prune_mask_near.defined() &&
                                prune_mask_near.numel() == N) {
                                hard_candidates =
                                    hard_candidates |
                                    prune_mask_near.to(prune_mask.device())
                                        .to(torch::kBool);
                            }
                            if (prune_mask_far.defined() &&
                                prune_mask_far.numel() == N) {
                                hard_candidates =
                                    hard_candidates |
                                    prune_mask_far.to(prune_mask.device())
                                        .to(torch::kBool);
                            }

                            const torch::Tensor pre_mvs_prune =
                                prune_mask.to(torch::kBool).clone();
                            prune_mask_mvs_protected =
                                (soft_candidates & prune_mask_mvs_supported &
                                 (~hard_candidates))
                                    .to(torch::kBool)
                                    .contiguous();
                            prune_mask_mvs_free_space_extra =
                                (prune_mask_mvs_free_space &
                                 (~pre_mvs_prune))
                                    .to(torch::kBool)
                                    .contiguous();

                            // Near/far are hard geometric filters. MVS support
                            // can protect learned-SDF candidates; independently
                            // confirmed free space contributes new candidates.
                            prune_mask =
                                (hard_candidates |
                                 (soft_candidates &
                                  (~prune_mask_mvs_supported)) |
                                 prune_mask_mvs_free_space)
                                    .to(torch::kBool)
                                    .contiguous();
                        }
                    } catch (const std::exception& e) {
                        std::cerr
                            << "[PRUNE/mvs] skipped: " << e.what() << "\n";
                    }
                }

                // Fine SVRecon parents are retained for level-9 continuity and
                // never participate in rendering or leaf-surface pruning.
                auto leaf_mask = voxel_model_->isLeaf();
                if (leaf_mask.defined() && leaf_mask.numel() == N) {
                    prune_mask = prune_mask.to(torch::kBool) &
                                 leaf_mask.to(prune_mask.device())
                                     .to(torch::kBool).reshape({-1});
                }
                // 4) Use the accumulated SVRecon pruning mask.
                prune_mask_default = prune_mask.to(torch::kBool);

                int64_t n_prune_base_real = prune_mask_base_real.defined()
                    ? prune_mask_base_real.sum().item<int64_t>() : 0;
                int64_t n_prune_base_real_cand = prune_mask_base_real_raw.defined()
                    ? prune_mask_base_real_raw.sum().item<int64_t>() : 0;
                int64_t n_prune_base_real_at_target = 0;
                int64_t n_prune_base_real_at_target_extra = 0;
                int64_t n_prune_base_real_above_target = 0;
                int64_t n_prune_base_real_pre_gates = 0;
                int64_t n_prune_base_real_at_target_pre_gates = 0;
                int64_t n_prune_base_real_above_target_pre_gates = 0;
                int64_t n_real_at_target_total = 0;
                int64_t n_real_above_target_total = 0;
                int64_t n_prune_near = 0;

                // 5) Record pre-protection threshold pruning stats.
                n_prune_base_real_pre_gates = prune_mask_base_real.defined()
                    ? prune_mask_base_real.sum().item<int64_t>() : 0;
                n_prune_base_real_at_target_pre_gates =
                    (prune_mask_base_real.to(torch::kBool) &
                     in_target_size_mask.to(torch::kBool)).sum().item<int64_t>();
                n_prune_base_real_above_target_pre_gates =
                    (prune_mask_base_real.to(torch::kBool) &
                     (~in_target_size_mask.to(torch::kBool))).sum().item<int64_t>();

                // 6) Report source counts, record optional debug data, then apply pruning.
                n_prune_base_real = prune_mask_base_real.defined()
                    ? prune_mask_base_real.sum().item<int64_t>() : 0;
                n_prune_base_real_at_target = prune_mask_base_real_at_target.defined()
                    ? prune_mask_base_real_at_target.sum().item<int64_t>() : 0;
                n_prune_base_real_at_target_extra = prune_mask_base_real_at_target_extra.defined()
                    ? prune_mask_base_real_at_target_extra.sum().item<int64_t>() : 0;
                auto in_target_size_mask_final = in_target_size_mask.to(torch::kBool).contiguous();
                auto above_target_size_mask_final = (~in_target_size_mask_final).to(torch::kBool);
                n_prune_base_real_above_target =
                    (prune_mask_base_real.to(torch::kBool) & above_target_size_mask_final).sum().item<int64_t>();
                n_real_at_target_total =
                    (real_mask_for_base.to(torch::kBool) & in_target_size_mask_final).sum().item<int64_t>();
                n_real_above_target_total =
                    (real_mask_for_base.to(torch::kBool) & above_target_size_mask_final).sum().item<int64_t>();
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
                const int64_t n_prune_stats_tsdf =
                    count_pruned_by_mask(sdf_prune_mask);
                const int64_t n_prune_stats_visibility =
                    count_pruned_by_mask(prune_mask_vis);
                const int64_t n_prune_stats_near =
                    count_pruned_by_mask(prune_mask_near);
                const int64_t n_prune_stats_far =
                    count_pruned_by_mask(prune_mask_far);
                const int64_t n_mvs_supported =
                    prune_mask_mvs_supported.defined()
                        ? prune_mask_mvs_supported.sum().item<int64_t>()
                        : 0;
                const int64_t n_mvs_protected =
                    prune_mask_mvs_protected.defined()
                        ? prune_mask_mvs_protected.sum().item<int64_t>()
                        : 0;
                const int64_t n_prune_stats_mvs_free_space =
                    count_pruned_by_mask(prune_mask_mvs_free_space);
                const int64_t n_prune_stats_mvs_free_space_extra =
                    count_pruned_by_mask(prune_mask_mvs_free_space_extra);
                std::cout << "[PRUNE/stats] iter=" << iter
                          << " total=" << n_prune_total_selected
                          << " sdf=" << n_prune_stats_tsdf
                          << " mvs_depth_keyframes=" << mvs_prune_depth_keyframes
                          << " mvs_valid_projections=" << mvs_prune_valid_projections
                          << " mvs_supported=" << n_mvs_supported
                          << " mvs_protected=" << n_mvs_protected
                          << " mvs_free_space_extra="
                          << n_prune_stats_mvs_free_space_extra
                          << " mvs_free_space_pruned="
                          << n_prune_stats_mvs_free_space
                          << " diagnostic_overlap_visibility=" << n_prune_stats_visibility
                          << " near_camera=" << n_prune_stats_near
                          << " far=" << n_prune_stats_far
                          << "\n";

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
	                            torch::Tensor pruned_centers = centers_world.index({prune_idx}).clone();
	                            torch::Tensor pruned_sizes = sizes_world.index({prune_idx}).clone();
                                torch::Tensor pruned_levels;
                                torch::Tensor oct_levels = voxel_model_->octLevel();
                                if (oct_levels.defined() &&
                                    oct_levels.numel() == N) {
                                    pruned_levels =
                                        oct_levels.reshape({N, 1})
                                            .index_select(
                                                0,
                                                prune_idx.to(oct_levels.device()).to(torch::kLong))
                                            .contiguous();
                                }
                                torch::Tensor pruned_colors;
                                torch::Tensor sh0 = voxel_model_->sh0();
                                if (sh0.defined() && sh0.dim() == 2 &&
                                    sh0.size(0) == N) {
                                    pruned_colors =
                                        (sh0.index_select(
                                             0,
                                             prune_idx.to(sh0.device()).to(torch::kLong)) *
                                             sv::kSHC0 +
                                         0.5f)
                                            .clamp(0.0f, 1.0f)
                                            .contiguous();
                                }
	                            torch::Tensor whole_run_pruned_by_tsdf;
                            if (sdf_prune_mask.defined() &&
                                sdf_prune_mask.numel() == N) {
                                whole_run_pruned_by_tsdf =
                                    sdf_prune_mask
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
                                        .index_select(
                                            0,
                                            prune_idx.to(torch::kLong))
                                        .contiguous();
                            }
                            torch::Tensor whole_run_pruned_by_far;
                            if (prune_mask_far.defined() &&
                                prune_mask_far.numel() == N) {
                                whole_run_pruned_by_far =
                                    prune_mask_far
                                        .to(prune_idx.device())
                                        .to(torch::kBool)
                                        .contiguous()
                                        .index_select(
                                            0,
                                            prune_idx.to(torch::kLong))
                                        .contiguous();
                            }
                            torch::Tensor whole_run_pruned_by_mvs_free_space;
                            if (prune_mask_mvs_free_space.defined() &&
                                prune_mask_mvs_free_space.numel() == N) {
                                whole_run_pruned_by_mvs_free_space =
                                    prune_mask_mvs_free_space
                                        .to(prune_idx.device())
                                        .to(torch::kBool)
                                        .contiguous()
                                        .index_select(
                                            0,
                                            prune_idx.to(torch::kLong))
                                        .contiguous();
                            }
                            appendWholeRunPrunedVoxels(
                                iter,
                                pruned_centers,
                                pruned_sizes,
                                pruned_levels,
                                pruned_colors,
                                whole_run_pruned_by_tsdf,
                                torch::Tensor(),
                                whole_run_pruned_by_near,
                                whole_run_pruned_by_far,
                                whole_run_pruned_by_mvs_free_space);
                        }
                    }
                }

                voxel_model_->pruning(prune_mask);
                if (rerun_params_.run_whole_run_ ||
                    rerun_params_.rerun_svrecon_debug_) {
                    rerun_state_.whole_run_live_voxels_dirty_ = true;
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

            if (need_pruning) {
                run_pruning();
            }

            // ---------------- SUBDIVIDE ----------------
            if (need_subdividing) {
                auto subdivision_profile =
                    profileLaptopModule("subdivision");
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
                    int64_t n_sdf_subdiv_candidates = 0;
                    int64_t n_subdiv_sdf_selected = 0;
                    int64_t n_sdf_crossing_candidates = 0;
                    int64_t n_sdf_base_scale_candidates = 0;
                    int64_t n_sdf_refined_candidates = 0;
                    int64_t n_sdf_priority_selected = 0;

                    // 1) One SVRecon subdivision pass for eligible voxels.
                    const int M = voxel_model_->numVoxels();
                    if (M > 0) {
                        auto vox_size = voxel_model_->voxSize(); // [M] or [M,1]
                        if (vox_size.dim() == 1) vox_size = vox_size.view({M,1});
                        else if (vox_size.dim() == 2 && vox_size.size(1) == 1) { /* ok */ }
                        else vox_size = vox_size.reshape({M,1});
                        auto vox_size_1d = vox_size.squeeze(1).contiguous();
                        auto octlv = voxel_model_->octLevel(); // [M] or [M,1]
                        if (octlv.defined() && octlv.dim() == 2 && octlv.size(1) == 1) {
                            octlv = octlv.squeeze(1);
                        }
                        auto octlv_i32 = octlv.to(torch::kInt32).contiguous();
                        const auto inside_level = octlv_i32 - svrecon_outside_level_;
                        const int level_stage_period = 2 * std::max(
                            1,
                            (opt_params_.prune_every_ > 0)
                                ? opt_params_.prune_every_
                                : opt_params_.adapt_every_);
                        const int coarse_level_limit =
                            9 - std::max(0, 2 - iter / level_stage_period);
                        // SVRecon progressively opens coarse levels 7, 8, and 9.
                        // Beyond the global association cap L=9, rendering may
                        // continue to L=10/11 but refinement is priority-limited.
                        auto non_finest =
                            (((inside_level < 9) &
                              (inside_level < coarse_level_limit)) |
                             ((inside_level >= 9) & (inside_level < 11))) &
                            (octlv_i32 < voxel_model_->maxNumLevels());
                        non_finest = non_finest.to(torch::kBool);

                        auto leaf_mask = voxel_model_->isLeaf();
                        if (!leaf_mask.defined() || leaf_mask.numel() != M) {
                            leaf_mask = torch::ones_like(non_finest, torch::kBool);
                        } else {
                            leaf_mask = leaf_mask.to(non_finest.device())
                                            .to(torch::kBool).reshape({M});
                        }
                        // Visibility is enforced during construction. Retained
                        // hierarchy parents are continuity-only and cannot split
                        // or render as leaves.
                        auto valid_mask_svrecon =
                            (non_finest.to(torch::kBool) & leaf_mask).to(torch::kBool);
                        auto normal_candidate_mask = valid_mask_svrecon.clone();
                        n_normal_candidates = normal_candidate_mask.sum().item<int64_t>();
                        torch::Tensor sdf_subdivide_candidate_mask;
                        torch::Tensor sdf_corners = voxel_model_->voxelGeoCorners();
                        if (sdf_corners.defined() &&
                            sdf_corners.numel() > 0 &&
                            sdf_corners.size(0) == M) {
                            sdf_corners =
                                sdf_corners
                                    .to(vox_size_1d.device())
                                    .to(torch::kFloat32)
                                    .contiguous();
                            torch::Tensor finite =
                                torch::isfinite(sdf_corners).to(torch::kBool);
                            torch::Tensor has_pos =
                                ((sdf_corners > 0.0f) & finite).any(/*dim=*/1);
                            torch::Tensor has_neg =
                                ((sdf_corners < 0.0f) & finite).any(/*dim=*/1);
                            torch::Tensor all_finite = finite.all(/*dim=*/1);
                            torch::Tensor has_surface =
                                (all_finite & has_pos & has_neg).to(torch::kBool);
                            // Subdivision requires an actual zero-level-set crossing.
                            // Same-sign cells in the near-zero learning band remain
                            // optimizable, but are not refined as surface geometry.
                            normal_candidate_mask =
                                (has_surface & valid_mask_svrecon)
                                    .to(torch::kBool);
                            sdf_subdivide_candidate_mask =
                                normal_candidate_mask.to(torch::kBool);
                            n_sdf_crossing_candidates =
                                (has_surface & valid_mask_svrecon).sum().item<int64_t>();
                            n_sdf_subdiv_candidates =
                                sdf_subdivide_candidate_mask.sum().item<int64_t>();
                            n_normal_candidates =
                                normal_candidate_mask.sum().item<int64_t>();
                        } else {
                            normal_candidate_mask =
                                torch::zeros_like(normal_candidate_mask, torch::kBool);
                            n_normal_candidates = 0;
                            n_sdf_subdiv_candidates = 0;
                        }

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

                        auto normal_selected_mask =
                            torch::zeros_like(normal_candidate_mask, torch::kBool);
                        if (n_normal_candidates > 0) {
                            priority = priority * normal_candidate_mask.to(priority.scalar_type());

                            // In the offline SVRecon scene, inside level 9 is a
                            // stable global transition to priority refinement.
                            // Our online scene extent changes with the initial
                            // map, so that absolute level can correspond to a
                            // much finer physical scale. Establish the configured
                            // base voxel scale densely, then use accumulated
                            // SVRecon priority for every further refinement.
                            const float base_scale_threshold =
                                0.75f * std::max(
                                    1.0e-6f,
                                    voxel_model_->fixedVoxSize());
                            auto base_scale_candidates =
                                (normal_candidate_mask &
                                 (vox_size_1d >= base_scale_threshold))
                                    .to(torch::kBool);
                            auto refined_candidates =
                                (normal_candidate_mask &
                                 (vox_size_1d < base_scale_threshold))
                                    .to(torch::kBool);
                            n_sdf_base_scale_candidates =
                                base_scale_candidates.sum().item<int64_t>();
                            n_sdf_refined_candidates =
                                refined_candidates.sum().item<int64_t>();

                            normal_selected_mask = base_scale_candidates.clone();
                            if (opt_params_.subdivide_all_until_ > 0 &&
                                iter <= opt_params_.subdivide_all_until_) {
                                normal_selected_mask = normal_candidate_mask.clone();
                            }
                            auto fine_idx = torch::nonzero(
                                                refined_candidates &
                                                (~normal_selected_mask))
                                                .reshape({-1}).to(torch::kLong);
                            if (fine_idx.numel() > 0 && opt_params_.subdivide_prop_ > 0.0f) {
                                const int64_t fine_keep = std::max<int64_t>(
                                    1,
                                    static_cast<int64_t>(std::ceil(
                                        opt_params_.subdivide_prop_ *
                                        static_cast<float>(fine_idx.numel()))));
                                auto fine_priority = priority.index_select(0, fine_idx);
                                auto order = std::get<1>(fine_priority.sort(
                                    /*dim=*/0, /*descending=*/true));
                                auto chosen = fine_idx.index_select(
                                    0,
                                    order.index({torch::indexing::Slice(0, fine_keep)})
                                        .to(torch::kLong));
                                normal_selected_mask.index_put_({chosen}, true);
                                n_sdf_priority_selected = chosen.numel();
                            }
                            // Match SVRecon's hard topology cap. Subdividing one
                            // parent replaces it with eight children, adding
                            // seven active voxels to the map.
                            const int64_t max_n_subdiv = std::max<int64_t>(
                                0,
                                (static_cast<int64_t>(opt_params_.subdivide_max_num_) -
                                 static_cast<int64_t>(M)) /
                                    7);
                            const int64_t selected_before_cap =
                                normal_selected_mask.sum().item<int64_t>();
                            if (selected_before_cap > max_n_subdiv) {
                                if (max_n_subdiv == 0) {
                                    normal_selected_mask =
                                        torch::zeros_like(
                                            normal_candidate_mask,
                                            torch::kBool);
                                } else {
                                    auto selected_idx =
                                        torch::nonzero(normal_selected_mask)
                                            .reshape({-1})
                                            .to(torch::kLong);
                                    auto selected_priority =
                                        priority.index_select(0, selected_idx);
                                    auto order = std::get<1>(
                                        selected_priority.sort(
                                            /*dim=*/0,
                                            /*descending=*/true));
                                    auto keep_idx = selected_idx.index_select(
                                        0,
                                        order.index({
                                            torch::indexing::Slice(
                                                0,
                                                max_n_subdiv)})
                                            .to(torch::kLong));
                                    normal_selected_mask =
                                        torch::zeros_like(
                                            normal_candidate_mask,
                                            torch::kBool);
                                    normal_selected_mask.index_put_(
                                        {keep_idx},
                                        true);
                                }
                                std::cout
                                    << "[SUBDIV/cap] max_voxels="
                                    << opt_params_.subdivide_max_num_
                                    << " current=" << M
                                    << " selected_before="
                                    << selected_before_cap
                                    << " selected_after="
                                    << normal_selected_mask.sum().item<int64_t>()
                                    << "\n";
                            }

                            n_sdf_priority_selected =
                                (normal_selected_mask & refined_candidates)
                                    .sum().item<int64_t>();
                            n_subdiv_normal_selected =
                                normal_selected_mask.sum().item<int64_t>();
                            if (sdf_subdivide_candidate_mask.defined() &&
                                sdf_subdivide_candidate_mask.numel() == M &&
                                n_subdiv_normal_selected > 0) {
                                n_subdiv_sdf_selected =
                                    (normal_selected_mask & sdf_subdivide_candidate_mask)
                                        .sum()
                                        .item<int64_t>();
                            }
                            if (n_subdiv_normal_selected > 0) {
		                                voxel_model_->subdividing(normal_selected_mask);
                                    if (rerun_params_.run_whole_run_ ||
                                        rerun_params_.rerun_svrecon_debug_) {
                                        rerun_state_.whole_run_live_voxels_dirty_ = true;
                                    }
                                did_subdivide = true;
                            }
                        }
                        if (rerun_params_.enable_rerun_ &&
                            rerun_params_.rerun_svrecon_debug_) {
                            logSvreconDebugVoxelMaskToRerun(
                                iter,
                                sdf_subdivide_candidate_mask,
                                "world/svrecon/subdivide_candidates");
                            logSvreconDebugVoxelMaskToRerun(
                                iter,
                                normal_selected_mask,
                                "world/svrecon/subdivide_selected");
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
                        std::cout << "[SUBDIV/sdf] zero_crossing_candidates="
                                  << n_sdf_crossing_candidates
                                  << " base_scale_candidates="
                                  << n_sdf_base_scale_candidates
                                  << " refined_candidates="
                                  << n_sdf_refined_candidates
                                  << " selected=" << n_subdiv_sdf_selected
                                  << " priority_selected="
                                  << n_sdf_priority_selected
                                  << " mode=base_scale_then_priority"
                                  << "\n";
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
            // Match SVRecon: reset accumulated subdivision metadata after a
            // subdivision round/topology rebuild.
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
                opt_params_.lr_decay_mult_,
                opt_params_.log_s_lr_
            );
            voxel_model_->schedulerLoadState(sched_state);
            // Empty CUDA cache as SV does
            c10::cuda::CUDACachingAllocator::emptyCache();
            last_densify_iter_ = iter;
        }
    }
    // Update learning rate
    voxel_model_->schedulerStep();

    if (rerun_params_.enable_rerun_) {
        logReconstructionMeshToRerun(iter);
        const int svrecon_grid_snapshot_interval = std::max(
            1,
            std::max(training_report_interval_, opt_params_.adapt_every_));
        const bool periodic_svrecon_snapshot =
            rerun_params_.rerun_svrecon_debug_ &&
            (iter % svrecon_grid_snapshot_interval) == 0;
        const bool log_whole_run_live =
            rerun_params_.run_whole_run_ &&
            rerun_state_.whole_run_live_voxels_dirty_;
        const bool log_svrecon_debug_live =
            rerun_params_.rerun_svrecon_debug_ &&
            (!rerun_state_.svrecon_debug_has_source_snapshot_ ||
             rerun_state_.whole_run_live_voxels_dirty_ ||
             periodic_svrecon_snapshot);
        if (log_whole_run_live || log_svrecon_debug_live) {
            logWholeRunLiveVoxelsToRerun(
                iter,
                voxel_model_->voxCenter(),
                voxel_model_->voxSize(),
                torch::Tensor(),
                log_whole_run_live,
                log_svrecon_debug_live);
            if (log_svrecon_debug_live) {
                rerun_state_.svrecon_debug_has_source_snapshot_ = true;
            }
            rerun_state_.whole_run_live_voxels_dirty_ = false;
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
                 photo_loss,
                 photo_loss_name,
                 ssim_loss,
                 monocular_depth_loss,
                 monocular_normal_loss,
                 ema_loss_for_log_,
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

bool VoxelMapper::isMatureMonocularOrbMapPoint(
    ORB_SLAM3::MapPoint* map_point,
    const unsigned long current_keyframe_id) const
{
    if (!map_point || map_point->isBad() || map_point->mnFirstKFid < 0) {
        return false;
    }

    // These are ORB-SLAM3's own monocular recent-point culling criteria.
    const long age =
        static_cast<long>(current_keyframe_id) - map_point->mnFirstKFid;
    if (age < 3 ||
        map_point->GetFoundRatio() < 0.25f ||
        map_point->Observations() <= 2) {
        return false;
    }

    const Eigen::Vector3f position = map_point->GetWorldPos();
    const Eigen::Vector3f color = map_point->GetColorRGB();
    return position.allFinite() && color.allFinite();
}

void VoxelMapper::collectNewMonocularOrbSamplingSupport(
    std::vector<float>& points,
    std::vector<float>& colors)
{
    points.clear();
    colors.clear();
    if (sensor_type_ != MONOCULAR || !mpSLAM || !mpSLAM->getAtlas()) {
        return;
    }

    ORB_SLAM3::Map* map = mpSLAM->getAtlas()->GetCurrentMap();
    if (!map) {
        return;
    }

    std::vector<sv::point3D_id_t> newly_transferred_ids;
    {
        std::unique_lock<std::mutex> lock_map(map->mMutexMapUpdate);
        const unsigned long current_keyframe_id = map->GetMaxKFid();
        const std::vector<ORB_SLAM3::MapPoint*> map_points =
            map->GetAllMapPoints();
        newly_transferred_ids.reserve(map_points.size());

        for (ORB_SLAM3::MapPoint* map_point : map_points) {
            if (!isMatureMonocularOrbMapPoint(
                    map_point, current_keyframe_id)) {
                continue;
            }

            const Eigen::Vector3f position = map_point->GetWorldPos();
            const Eigen::Vector3f color = map_point->GetColorRGB();

            if (monocular_orb_inserted_point_ids_.count(map_point->mnId) != 0) {
                continue;
            }
            newly_transferred_ids.push_back(map_point->mnId);
            points.insert(
                points.end(),
                {position.x(), position.y(), position.z()});
            colors.insert(
                colors.end(),
                {color.x(), color.y(), color.z()});
        }
    }

    monocular_orb_inserted_point_ids_.insert(
        newly_transferred_ids.begin(), newly_transferred_ids.end());
}

void VoxelMapper::combineMappingOperations()
{
    auto incremental_map_profile =
        profileLaptopModule("incremental_map_update");
    // Get Mapping Operations
    while (mpSLAM->getAtlas()->hasMappingOperation()) {
        ORB_SLAM3::MappingOperation opr =
            mpSLAM->getAtlas()->getAndPopMappingOperation();
        switch (opr.meOperationType)
        {
        case ORB_SLAM3::MappingOperation::OprType::LocalMappingBA:
        {
            bool kf_changed = false;
            std::vector<std::shared_ptr<VoxelKeyframe>> new_densification_kfs;
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
                    std::shared_ptr<VoxelKeyframe> new_pkf =
                        scene_->getKeyframe(kfid);
                    if (new_pkf) {
                        new_densification_kfs.push_back(new_pkf);
                    }
                    kf_changed = true;
			        }
	            }
            // Get new points
            auto& associated_points = opr.associatedMapPoints();
            auto& points = std::get<0>(associated_points);
            auto& colors = std::get<1>(associated_points);

            const int iter = getIteration();
            if (allocate_orb_voxels_ && initial_mapped_) {
                std::vector<float> allocation_points;
                std::vector<float> allocation_colors;
                if (sensor_type_ == MONOCULAR) {
                    collectNewMonocularOrbSamplingSupport(
                        allocation_points, allocation_colors);
                } else {
                    allocation_points = points;
                    allocation_colors = colors;
                }

                if (allocation_points.size() >= 3) {
                    torch::NoGradGuard no_grad;
                    std::unique_lock<std::mutex> lock_render(
                        mutex_render_);
                    std::vector<sv::MiniCam> tr_cams =
                        incrementalMappingCameras();
                    voxel_model_->setNextRealInsertionRerunEntityPath(
                        "world/orb/voxels_created");
                    {
                        auto insertion_profile =
                            profileLaptopModule("orb_voxel_insertion");
                        voxel_model_->increasePcd(
                            std::move(allocation_points),
                            std::move(allocation_colors),
                            getIteration(),
                            tr_cams);
                    }
                    voxel_model_->setNextRealInsertionRerunEntityPath("");
                    if (voxel_model_->lastIncreasePcdStats().new_voxels > 0 &&
                        (rerun_params_.run_whole_run_ ||
                         rerun_params_.rerun_svrecon_debug_)) {
                        rerun_state_.whole_run_live_voxels_dirty_ = true;
                    }
		                }
			        }

            // Preserve the established scheduling: ORB topology first,
            // then inactive geometry, then residual sensor evidence/fill.
            const bool do_inactive_geo_densify =
                isdoingInactiveGeoDensify();
            const bool do_rgbd_completion =
                sensor_type_ == RGBD &&
                (rgbd_fill_render_holes_ || rgbd_tsdf_evidence_);
            if ((do_inactive_geo_densify || do_rgbd_completion) &&
                !new_densification_kfs.empty()) {
                for (const auto& pkf : new_densification_kfs) {
                    increasePcdByKeyframeInactiveGeoDensify(
                        pkf,
                        /*include_inactive_geo=*/do_inactive_geo_densify,
                        /*include_rgbd_hole_fill=*/do_rgbd_completion);
                }
            }
            if (sensor_type_ == MONOCULAR &&
                monocular_rendered_depth_densify_) {
                for (const auto& pkf : new_densification_kfs) {
                    densifyMonocularFromRenderedDepth(pkf);
                }
            }
            if (sensor_type_ == MONOCULAR &&
                isMonocularMvsPipelineEnabled()) {
                scheduleLatestMonocularMvsKeyframe(new_densification_kfs);
            }
            if (sensor_type_ == MONOCULAR &&
                monocular_omnidata_densify_) {
                scheduleLatestMonocularOmnidataKeyframe(
                    new_densification_kfs);
            }
            processRgbdClosureCache();
            markSurfaceViewPruningPending(new_densification_kfs);
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
            std::vector<std::shared_ptr<VoxelKeyframe>> new_densification_kfs;
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
                                 voxel_utils::eigenMatrixToTorchTensor(
                                     diff_pose.matrix(), device_type_).transpose(0, 1);
	                             // Give loop keyframes times of use
	                             increaseKeyframeTimesOfUse(pkf, loop_closure_increased_times_of_use_);
	                         }
	                     pkf->setPose(
	                         pose.unit_quaternion().cast<double>(),
	                         pose.translation().cast<double>());

		                    kf_changed = true;
	                 }
			                 else if (loop_closure_reinsert_points_) {
			                     handleNewKeyframe(kf);
			                     pkf = scene_->getKeyframe(kfid);
                         if (pkf) {
                             new_densification_kfs.push_back(pkf);
                         }
		                         kf_changed = true;
			                 }
	             }
             if (record_loop_ply_)
                 savePly(result_dir_ / (std::to_string(getIteration()) + "_1_after_loop_correction"));
	             // Get new points (scaled transformation applied in ORB-SLAM3, so this step is performed at last to avoid scaling twice)
             auto& associated_points = opr.associatedMapPoints();
             auto& points = std::get<0>(associated_points);
             auto& colors = std::get<1>(associated_points);

             // Add new points to the model
             const int iter = getIteration();
	             if (loop_closure_reinsert_points_ &&
	                 allocate_orb_voxels_ &&
	                 initial_mapped_) {
                std::vector<float> allocation_points;
                std::vector<float> allocation_colors;
                if (sensor_type_ == MONOCULAR) {
                    collectNewMonocularOrbSamplingSupport(
                        allocation_points, allocation_colors);
                } else {
                    allocation_points = points;
                    allocation_colors = colors;
                }

                if (allocation_points.size() >= 3) {
                    torch::NoGradGuard no_grad;
                    std::unique_lock<std::mutex> lock_render(
                        mutex_render_);

                    // Match Photo-SLAM behavior: insert loop-closure associated points.
                    std::vector<sv::MiniCam> tr_cams =
                        incrementalMappingCameras();
                    voxel_model_->setNextRealInsertionRerunEntityPath(
                        "world/orb/voxels_created");
                    {
                        auto insertion_profile =
                            profileLaptopModule("orb_voxel_insertion");
                        voxel_model_->increasePcd(
                            std::move(allocation_points),
                            std::move(allocation_colors),
                            iter,
                            tr_cams);
                    }
                    voxel_model_->setNextRealInsertionRerunEntityPath("");
                    if (voxel_model_->lastIncreasePcdStats().new_voxels > 0 &&
                        (rerun_params_.run_whole_run_ ||
                         rerun_params_.rerun_svrecon_debug_)) {
                        rerun_state_.whole_run_live_voxels_dirty_ = true;
                    }
                }
			             }

                const bool do_inactive_geo_densify =
                    isdoingInactiveGeoDensify();
                const bool do_rgbd_completion =
                    sensor_type_ == RGBD &&
                    (rgbd_fill_render_holes_ || rgbd_tsdf_evidence_);
                if ((do_inactive_geo_densify || do_rgbd_completion) &&
                    !new_densification_kfs.empty()) {
                    for (const auto& pkf : new_densification_kfs) {
                        increasePcdByKeyframeInactiveGeoDensify(
                            pkf,
                            /*include_inactive_geo=*/do_inactive_geo_densify,
                            /*include_rgbd_hole_fill=*/do_rgbd_completion);
                    }
                }
                if (sensor_type_ == MONOCULAR &&
                    monocular_rendered_depth_densify_) {
                    for (const auto& pkf : new_densification_kfs) {
                        densifyMonocularFromRenderedDepth(pkf);
                    }
                }
                if (sensor_type_ == MONOCULAR &&
                    isMonocularMvsPipelineEnabled()) {
                    scheduleLatestMonocularMvsKeyframe(
                        new_densification_kfs);
                }
                if (sensor_type_ == MONOCULAR &&
                    monocular_omnidata_densify_) {
                    scheduleLatestMonocularOmnidataKeyframe(
                        new_densification_kfs);
                }
                processRgbdClosureCache();
                markSurfaceViewPruningPending(new_densification_kfs);
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
	                 }
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
     logCurrentOrbMapPointsToReconstructionRerun(getIteration());
     logCurrentOrbKeyframePosesToReconstructionRerun(getIteration());
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
    if (this->sensor_type_ == RGBD) {
        imgAux_undistorted = mapperDepthForKeyframe(
            std::get<8>(kf), imgAux, camera);
    } else {
        imgAux_undistorted = imgAux;
    }

    pkf->original_image_ =
        voxel_utils::cvMatToTorchTensorFloat32(imgRGB_undistorted, device_type_);
    pkf->img_filename_ = std::get<8>(kf);
    pkf->source_timestamp_ =
        voxel_utils::parseFrameTimestampFromPath(pkf->img_filename_);
    pkf->source_frame_id_ = voxel_utils::parseFrameIdFromPath(pkf->img_filename_);
    pkf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
    pkf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
    pkf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
     
    // Add the new keyframe to the scene
    // pkf->computeTransformTensors();
    scene_->addKeyframe(pkf, &kfid_shuffled_);
    latest_consumed_keyframe_id_.store(
        std::max(
            latest_consumed_keyframe_id_.load(std::memory_order_relaxed),
            static_cast<long long>(std::get<0>(kf))),
        std::memory_order_release);

    // Give new keyframes times of use and add it to the training sliding window
    increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());

    // Get dense point cloud from the new keyframe to accelerate training
    pkf->img_undist_ = imgRGB_undistorted;
    pkf->img_auxiliary_undist_ = imgAux_undistorted;

    pkf->kps_pixel_ = std::move(std::get<6>(kf));
    pkf->kps_point_local_ = std::move(std::get<7>(kf));

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
                voxel_utils::cvGpuMatToTorchTensorFloat32(img_resized);
        }
    }
    else {
        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::Mat img_resized;
            cv::resize(pkf->img_undist_, img_resized,
                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                voxel_utils::cvMatToTorchTensorFloat32(img_resized, device_type_);
        }
    }

    logKeyframeCameraToRerunRecordings(
        pkf,
        std::get<0>(kf),
        /*log_reconstruction_mesh=*/true);

    if (rerun_params_.run_whole_run_ ||
        rerun_params_.rerun_svrecon_debug_) {
        rerun_state_.whole_run_live_voxels_dirty_ = true;
    }

    if (initial_mapped_ && sensor_type_ == RGBD &&
        sdf_initialization_rgbd_projective_) {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        fuseProjectiveSdfInitFromKeyframe(pkf);
    }
}


torch::Tensor VoxelMapper::detectRgbdRenderHolePixels(
    const std::shared_ptr<VoxelKeyframe>& pkf,
    const torch::Tensor& depth,
    const int pixel_stride,
    const bool render_on_stride_grid,
    int64_t& valid_depth_pixels,
    int64_t& hole_pixels,
    torch::Tensor& full_hole_mask)
{
    valid_depth_pixels = 0;
    hole_pixels = 0;
    full_hole_mask = torch::Tensor();
    if (!pkf || !voxel_model_ || !depth.defined() ||
        pkf->image_height_ <= 0 || pkf->image_width_ <= 0) {
        return torch::Tensor();
    }

    const int stride = std::max(1, pixel_stride);
    const int render_scale = render_on_stride_grid ? stride : 1;
    const int render_height =
        (pkf->image_height_ + render_scale - 1) / render_scale;
    const int render_width =
        (pkf->image_width_ + render_scale - 1) / render_scale;
    sv::MiniCam render_camera =
        pkf->toMiniCam(pkf->image_height_, pkf->image_width_);
    if (render_scale > 1) {
        const float scale = static_cast<float>(render_scale);
        render_camera.width = render_width;
        render_camera.height = render_height;
        render_camera.fx /= scale;
        render_camera.fy /= scale;
        render_camera.cx /= scale;
        render_camera.cy /= scale;
        const float fovx = sv::focalToFov(
            render_camera.fx, render_width);
        const float fovy = sv::focalToFov(
            render_camera.fy, render_height);
        render_camera.tanfovx = std::tan(0.5f * fovx);
        render_camera.tanfovy = std::tan(0.5f * fovy);
        render_camera.pix_size =
            2.0f * render_camera.tanfovx /
            static_cast<float>(render_width);
    }

    std::unordered_map<std::string, torch::Tensor> render_pkg;
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        render_pkg = voxel_model_->render(
            render_camera,
            render_height,
            render_width,
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
    if (!voxel_utils::renderPkgToDepthAlphaMaps(
            render_pkg,
            render_height,
            render_width,
            render_depth_cpu,
            render_alpha_cpu,
            render_n_contrib_cpu)) {
        return torch::Tensor();
    }

    torch::Tensor depth_cpu =
        depth.detach().to(torch::kCPU).to(torch::kFloat32)
            .reshape({pkf->image_height_, pkf->image_width_}).contiguous();
    std::vector<uint8_t> selected_mask(
        static_cast<size_t>(pkf->image_height_) *
        static_cast<size_t>(pkf->image_width_),
        0);
    std::vector<uint8_t> all_holes_mask(selected_mask.size(), 0);
    std::vector<int64_t> selected;
    auto depth_acc = depth_cpu.accessor<float, 2>();
    auto render_depth_acc = render_depth_cpu.accessor<float, 2>();
    auto render_n_contrib_acc = render_n_contrib_cpu.accessor<int, 2>();
    for (int render_y = 0; render_y < render_height; ++render_y) {
        const int y = render_y * render_scale;
        for (int render_x = 0; render_x < render_width; ++render_x) {
            const int x = render_x * render_scale;
            const float z_rgbd = depth_acc[y][x];
            if (!std::isfinite(z_rgbd) ||
                z_rgbd <= RGBD_min_depth_ || z_rgbd >= RGBD_max_depth_) {
                continue;
            }
            ++valid_depth_pixels;
            const float z_render = render_depth_acc[render_y][render_x];
            const int n_contrib =
                render_n_contrib_acc[render_y][render_x];
            const bool no_rendered_depth =
                !std::isfinite(z_render) || z_render <= 1.0e-6f;
            const bool structural_hole =
                n_contrib <= 0 && no_rendered_depth;
            if (!structural_hole) {
                continue;
            }
            ++hole_pixels;
            all_holes_mask[
                static_cast<size_t>(y) * pkf->image_width_ + x] = 1;
            if (render_on_stride_grid ||
                ((x % stride) == 0 && (y % stride) == 0)) {
                selected.push_back(
                    static_cast<int64_t>(y) * pkf->image_width_ + x);
            }
        }
    }

    for (const int64_t idx : selected) {
        selected_mask[static_cast<size_t>(idx)] = 1;
    }
    full_hole_mask = torch::from_blob(
                         all_holes_mask.data(),
                         {static_cast<int64_t>(all_holes_mask.size())},
                         torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
                         .clone().to(device_type_).to(torch::kBool).contiguous();
    return torch::from_blob(
               selected_mask.data(),
               {static_cast<int64_t>(selected_mask.size())},
               torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
        .clone().to(device_type_).to(torch::kBool).contiguous();
}

void VoxelMapper::fillRgbdRenderHolesSdf(
    const std::shared_ptr<VoxelKeyframe>& pkf)
{
    auto fill_profile =
        profileLaptopModule("rgbd_direct_hole_sampling");
    if (!pkf || !voxel_model_ || !rgbd_fill_render_holes_ ||
        pkf->img_undist_.empty() || pkf->img_auxiliary_undist_.empty()) {
        return;
    }

    cv::cuda::GpuMat img_rgb_gpu, img_depth_gpu;
    img_rgb_gpu.upload(pkf->img_undist_);
    img_depth_gpu.upload(pkf->img_auxiliary_undist_);
    torch::Tensor rgb =
        voxel_utils::cvGpuMatToTorchTensorFloat32(img_rgb_gpu)
            .permute({1, 2, 0})
            .flatten(0, 1)
            .contiguous();
    torch::Tensor depth =
        voxel_utils::cvGpuMatToTorchTensorFloat32(img_depth_gpu)
            .flatten(0, 1)
            .contiguous();

    const sv::Camera& camera = scene_->cameras_.at(pkf->camera_id_);
    if (camera.model_id_ != sv::Camera::PINHOLE) {
        throw std::runtime_error(
            "[VoxelMapper] RGB-D render-hole filling supports pinhole cameras only.");
    }

    int64_t valid_depth_pixels = 0;
    int64_t hole_pixels = 0;
    torch::Tensor full_hole_mask;
    torch::Tensor hole_mask = detectRgbdRenderHolePixels(
        pkf,
        depth,
        rgbd_fill_render_holes_stride_,
        /*render_on_stride_grid=*/false,
        valid_depth_pixels,
        hole_pixels,
        full_hole_mask);
    if (!hole_mask.defined() || hole_mask.numel() == 0 ||
        !hole_mask.any().item<bool>()) {
        return;
    }

    torch::Tensor points3D_camera =
        voxel_utils::reprojectDepthPinholeVoxel(
            depth, pkf->intr_, pkf->image_width_);
    torch::Tensor surface_world =
        points3D_camera.index({hole_mask}).contiguous();
    torch::Tensor surface_colors = rgb.index({hole_mask}).contiguous();
    const Sophus::SE3f Twc = pkf->getPosef().inverse();
    torch::Tensor Twc_tensor =
        voxel_utils::eigenMatrixToTorchTensor(
            Twc.matrix(), device_type_).transpose(0, 1);
    voxel_utils::transformPoints(surface_world, Twc_tensor);

    // Match the established batched densification flow: collect all hole
    // samples from the cached keyframes and perform one topology update.
    if (!rgbd_fill_render_holes_cache_points_.defined() ||
        rgbd_fill_render_holes_cache_points_.dim() != 2 ||
        rgbd_fill_render_holes_cache_points_.size(0) == 0) {
        rgbd_fill_render_holes_cache_points_ = surface_world;
        rgbd_fill_render_holes_cache_colors_ = surface_colors;
    } else {
        rgbd_fill_render_holes_cache_points_ = torch::cat(
            {rgbd_fill_render_holes_cache_points_, surface_world}, 0);
        rgbd_fill_render_holes_cache_colors_ = torch::cat(
            {rgbd_fill_render_holes_cache_colors_, surface_colors}, 0);
    }
    if (rgbd_fill_render_holes_projective_sdf_) {
        rgbd_fill_render_holes_projective_cache_.emplace_back(
            pkf, hole_mask);
    }
}

void VoxelMapper::increasePcdByKeyframeInactiveGeoDensify(
    std::shared_ptr<VoxelKeyframe> pkf,
    const bool include_inactive_geo,
    const bool include_rgbd_hole_fill)
{
    auto prepare_profile =
        profileLaptopModule("inactive_geometry_depth_prepare");
    torch::NoGradGuard no_grad;

    // Pose of camera in world frame
    Sophus::SE3f Twc = pkf->getPosef().inverse();

    switch (this->sensor_type_)
    {
    case MONOCULAR:
    {
        if (include_inactive_geo) {
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
            torch::Tensor colors = voxel_utils::cvGpuMatToTorchTensorFloat32(rgb_gpu);
            colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();

            // Photo-SLAM’s neighborhood densification
            auto result =
                voxel_utils::monocularPinholeInactiveGeoDensifyBySearchingNeighborhoodKeypoints(
                    kps_pixel_tensor,
                    kps_has3D_tensor,
                    kps_point_local_tensor,
                    colors,
                    inactive_geo_densify_max_pixel_dist_,
                    pkf->intr_,
                    pkf->image_width_);

            torch::Tensor& points3D_valid = std::get<0>(result);
            torch::Tensor& colors_valid   = std::get<1>(result);

            // Transform points to world coordinates
            torch::Tensor Twc_tensor =
                voxel_utils::eigenMatrixToTorchTensor(
                    Twc.matrix(), device_type_).transpose(0, 1);
            voxel_utils::transformPoints(points3D_valid, Twc_tensor);

            // Add new points to the cache
            if (!depth_cache_points_.defined() ||
                depth_cache_points_.numel() == 0) {
                depth_cache_points_ = points3D_valid;
                depth_cache_colors_ = colors_valid;
            } else {
                depth_cache_points_ = torch::cat(
                    {depth_cache_points_, points3D_valid}, /*dim=*/0);
                depth_cache_colors_ = torch::cat(
                    {depth_cache_colors_, colors_valid}, /*dim=*/0);
            }
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
        torch::Tensor disp = voxel_utils::cvGpuMatToTorchTensorFloat32(cv_disp);
        disp = disp.flatten(0, 1).contiguous();

        torch::Tensor points3D =
            voxel_utils::cvGpuMatToTorchTensorFloat32(cv_points3D);
        points3D = points3D.permute({1, 2, 0}).flatten(0, 1).contiguous();

        torch::Tensor colors =
            voxel_utils::cvGpuMatToTorchTensorFloat32(rgb_left_gpu);
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
            voxel_utils::eigenMatrixToTorchTensor(
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
        torch::Tensor rgb = voxel_utils::cvGpuMatToTorchTensorFloat32(img_rgb_gpu);
        rgb = rgb.permute({1, 2, 0}).flatten(0, 1).contiguous();

        torch::Tensor depth = voxel_utils::cvGpuMatToTorchTensorFloat32(img_depth_gpu);
        depth = depth.flatten(0, 1).contiguous();

        sv::Camera& camera = scene_->cameras_.at(pkf->camera_id_);

        // Match Photo-SLAM inactive geometry: only tracked keypoints are
        // candidates for direct depth reprojection.
        torch::Tensor point_valid_flags = torch::full(
            {depth.size(0)},
            false,   // Note Photo-SLAM uses false here and then sets only around kps
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));

        int nkps_twice = pkf->kps_pixel_.size();
        int width      = pkf->image_width_;
        std::vector<int64_t> keypoint_indices;
        keypoint_indices.reserve(static_cast<size_t>(nkps_twice / 2));
        for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
            const int x = static_cast<int>(pkf->kps_pixel_[kpidx]);
            const int y = static_cast<int>(pkf->kps_pixel_[kpidx + 1]);
            if (x >= 0 && x < width && y >= 0 && y < pkf->image_height_) {
                keypoint_indices.push_back(
                    static_cast<int64_t>(y) * width + x);
            }
        }
        if (!keypoint_indices.empty()) {
            torch::Tensor keypoint_idx = torch::from_blob(
                keypoint_indices.data(),
                {static_cast<int64_t>(keypoint_indices.size())},
                torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
                .clone()
                .to(device_type_);
            point_valid_flags.index_fill_(0, keypoint_idx, true);
        }

        torch::Tensor valid_depth =
            torch::isfinite(depth) &
            (depth > RGBD_min_depth_) &
            (depth < RGBD_max_depth_);
        point_valid_flags = point_valid_flags & valid_depth;
        
        torch::Tensor inactive_geo_flags =
            include_inactive_geo
            ? point_valid_flags.clone()
            : torch::zeros_like(point_valid_flags);

        // Reproject to 3D (camera coordinates)
        torch::Tensor points3D_all;

        switch (camera.model_id_)
        {
        case sv::Camera::PINHOLE:
        {
            points3D_all = voxel_utils::reprojectDepthPinholeVoxel(
                depth,
                pkf->intr_,
                pkf->image_width_);
        }
        break;

        case sv::Camera::FISHEYE:
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
        // Transform to world coordinates
        torch::Tensor Twc_tensor =
            voxel_utils::eigenMatrixToTorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        voxel_utils::transformPoints(points3D_inactive_geo, Twc_tensor);

        if (points3D_inactive_geo.defined() &&
            points3D_inactive_geo.dim() == 2 &&
            points3D_inactive_geo.size(0) > 0) {
            if (!depth_cache_points_.defined() ||
                depth_cache_points_.dim() != 2 ||
                depth_cache_points_.size(0) == 0) {
                depth_cache_points_ = points3D_inactive_geo;
                depth_cache_colors_ = colors_inactive_geo;
            } else {
                depth_cache_points_ = torch::cat(
                    {depth_cache_points_, points3D_inactive_geo}, 0);
                depth_cache_colors_ = torch::cat(
                    {depth_cache_colors_, colors_inactive_geo}, 0);
            }
        }

        if (include_rgbd_hole_fill &&
            (rgbd_fill_render_holes_ || rgbd_tsdf_evidence_)) {
            rgbd_hole_fill_keyframe_cache_.push_back(pkf);
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
        flushInactiveGeoCache();
    }
}

void VoxelMapper::flushInactiveGeoCache()
{
    auto insertion_profile =
        profileLaptopModule("inactive_geometry_insertion");
    const bool have_inactive_points =
        depth_cache_points_.defined() &&
        depth_cache_points_.dim() == 2 &&
        depth_cache_points_.size(0) > 0;
    if (depth_cached_ <= 0 && !have_inactive_points &&
        rgbd_hole_fill_keyframe_cache_.empty()) {
        return;
    }
    depth_cached_ = 0;

    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        std::vector<sv::MiniCam> tr_cams = incrementalMappingCameras();

        auto flush_points =
            [&](torch::Tensor& points,
                torch::Tensor& colors,
                const std::string& entity_path)
        {
            if (!points.defined() || points.dim() != 2 || points.size(0) <= 0) {
                points = torch::Tensor();
                colors = torch::Tensor();
                return;
            }
            voxel_model_->setNextRealInsertionRerunEntityPath(entity_path);
            voxel_model_->increasePcd(
                points,
                colors,
                getIteration(),
                tr_cams);
            voxel_model_->setNextRealInsertionRerunEntityPath("");
            if (voxel_model_->lastIncreasePcdStats().new_voxels > 0 &&
                (rerun_params_.run_whole_run_ ||
                 rerun_params_.rerun_svrecon_debug_)) {
                rerun_state_.whole_run_live_voxels_dirty_ = true;
            }
            points = torch::Tensor();
            colors = torch::Tensor();
        };

        // Commit inactive geometry before residual-hole detection.
        flush_points(
            depth_cache_points_,
            depth_cache_colors_,
            "world/voxels_inactive_geo_densify/created");
    }

    // Detect residual holes only after ORB and inactive topology are present.
    rgbd_hole_fill_ready_keyframes_.insert(
        rgbd_hole_fill_ready_keyframes_.end(),
        rgbd_hole_fill_keyframe_cache_.begin(),
        rgbd_hole_fill_keyframe_cache_.end());
    rgbd_hole_fill_keyframe_cache_.clear();
}

void VoxelMapper::processRgbdClosureCache()
{
    auto closure_profile =
        profileLaptopModule("rgbd_gap_filling");
    std::vector<std::shared_ptr<VoxelKeyframe>> rgbd_closure_kfs;
    rgbd_closure_kfs.swap(rgbd_hole_fill_ready_keyframes_);
    std::unordered_set<
        sv::RgbdTsdfGridKey,
        sv::RgbdTsdfGridKeyHash> affected_evidence_cells;
    for (const auto& closure_kf : rgbd_closure_kfs) {
        if (rgbd_fill_render_holes_) {
            fillRgbdRenderHolesSdf(closure_kf);
        }
        if (rgbd_tsdf_evidence_) {
            integrateRgbdTsdfEvidenceForRenderHoles(
                closure_kf,
                affected_evidence_cells);
        }
    }

    const bool have_direct_fill =
        rgbd_fill_render_holes_cache_points_.defined() &&
        rgbd_fill_render_holes_cache_points_.dim() == 2 &&
        rgbd_fill_render_holes_cache_points_.size(0) > 0;
    if (have_direct_fill) {
        sv::VoxelModel::IncreasePcdStats stats;
        {
            std::unique_lock<std::mutex> lock_render(mutex_render_);
            voxel_model_->setNextRealInsertionRerunEntityPath(
                "world/rgbd_fill_render_holes/created");
            voxel_model_->increasePcd(
                rgbd_fill_render_holes_cache_points_,
                rgbd_fill_render_holes_cache_colors_,
                getIteration(),
                incrementalMappingCameras());
            voxel_model_->setNextRealInsertionRerunEntityPath("");
            stats = voxel_model_->lastIncreasePcdStats();

            for (const auto& item :
                 rgbd_fill_render_holes_projective_cache_) {
                fuseProjectiveSdfInitFromKeyframe(item.first, item.second);
            }
        }
        if (stats.new_voxels > 0 &&
            (rerun_params_.run_whole_run_ ||
             rerun_params_.rerun_svrecon_debug_)) {
            rerun_state_.whole_run_live_voxels_dirty_ = true;
        }
    }
    rgbd_fill_render_holes_cache_points_ = torch::Tensor();
    rgbd_fill_render_holes_cache_colors_ = torch::Tensor();
    rgbd_fill_render_holes_projective_cache_.clear();

    if (rgbd_tsdf_evidence_ && !affected_evidence_cells.empty()) {
        promoteRgbdTsdfEvidenceCells(affected_evidence_cells);
    }

    // Re-fuse after deferred evidence promotion or direct batched insertion so
    // newly allocated cells receive projective D-z corner values.
    if (initial_mapped_ && sdf_initialization_rgbd_projective_) {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        for (const auto& closure_kf : rgbd_closure_kfs) {
            fuseProjectiveSdfInitFromKeyframe(closure_kf);
        }
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
    out_stream.open(result_path, std::ios::out | std::ios::trunc);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open json at " + result_path.string());

    out_stream << "##[Voxel Mapper]Iteration " << getIteration() << " keyframe id, used times, remaining times:\n";
    for (const auto& used_times_it : kfs_used_times_) {
        const auto scene_kf_it = scene_->keyframes().find(used_times_it.first);
        const int remaining_times =
            scene_kf_it != scene_->keyframes().end() && scene_kf_it->second
                ? scene_kf_it->second->remaining_times_of_use_
                : -1;
        out_stream << used_times_it.first << " "
                   << used_times_it.second << " "
                   << remaining_times
                   << "\n";
    }
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
         auto image_cv = voxel_utils::torchTensorToCvMatFloat32(rendered);
         cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
         image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
         cv::imwrite(result_img_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + ".jpg"), image_cv);
     }
 
     if (record_ground_truth_image_) {
         auto gt_image_cv = voxel_utils::torchTensorToCvMatFloat32(ground_truth);
         cv::cvtColor(gt_image_cv, gt_image_cv, CV_RGB2BGR);
         gt_image_cv.convertTo(gt_image_cv, CV_8UC3, 255.0f);
         cv::imwrite(result_gt_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_gt.jpg"), gt_image_cv);
     }
 
     if (record_loss_image_) {
         torch::Tensor loss_tensor = torch::abs(rendered - ground_truth);
         auto loss_image_cv = voxel_utils::torchTensorToCvMatFloat32(loss_tensor);
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
    const std::filesystem::path& result_svrecon_normal_dir,
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

    dssim = voxel_eval::ssim(masked_image, gt_image, device_type_).item().toFloat();
    psnr = voxel_eval::psnr(masked_image, gt_image).item().toFloat();

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

        voxel_eval::saveMetricDepthPngMillimeters(
            pred_depth_for_eval,
            result_depth_dir.parent_path() / "depth_metric" / (stem + ".png"),
            RGBD_min_depth_,
            RGBD_max_depth_);

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

        const torch::Tensor pred_svrecon_normal_viz =
            -voxel_eval::normalWorldToCameraForViz(cam, pred_normal_unit);
        const cv::Mat pred_svrecon_normal_bgr =
            voxel_eval::colorizeNormalMapBgr(pred_svrecon_normal_viz);
        const std::filesystem::path svrecon_normal_path =
            result_svrecon_normal_dir / (stem + ".png");
        std::filesystem::create_directories(svrecon_normal_path.parent_path());
        if (!pred_svrecon_normal_bgr.empty()) {
            cv::imwrite(svrecon_normal_path.string(), pred_svrecon_normal_bgr);
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
                torch::Tensor rendered_depth_normal = voxel_eval::depthToNormal(
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
            torch::Tensor gt_normal = voxel_eval::depthToNormal(
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

    std::filesystem::path svrecon_normal_dir = result_dir / "normals_svrecon";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(svrecon_normal_dir);

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
            svrecon_normal_dir,
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
    return voxel_utils::torchTensorToCvMatFloat32(masked_image);
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

std::vector<sv::MiniCam> VoxelMapper::incrementalMappingCameras() const
{
    const auto& keyframes = scene_->keyframes();
    const std::size_t limit = incremental_mapping_window_size_ > 0
        ? std::min<std::size_t>(
              static_cast<std::size_t>(incremental_mapping_window_size_),
              keyframes.size())
        : keyframes.size();

    std::vector<sv::MiniCam> cameras;
    cameras.reserve(limit);
    for (auto it = keyframes.rbegin();
         it != keyframes.rend() && cameras.size() < limit;
         ++it) {
        if (!it->second) {
            continue;
        }
        cameras.push_back(it->second->toMiniCam(
            it->second->image_height_, it->second->image_width_));
    }
    return cameras;
}

std::vector<sv::MiniCam> VoxelMapper::surfaceViewPruningCameras() const
{
    const auto& keyframes = scene_->keyframes();
    const std::size_t limit = std::min<std::size_t>(
        static_cast<std::size_t>(
            std::max(1, opt_params_.surface_view_window_size_)),
        keyframes.size());

    std::vector<sv::MiniCam> cameras;
    cameras.reserve(limit);
    for (auto it = keyframes.rbegin();
         it != keyframes.rend() && cameras.size() < limit;
         ++it) {
        if (!it->second) {
            continue;
        }
        cameras.push_back(it->second->toMiniCam(
            it->second->image_height_, it->second->image_width_));
    }
    return cameras;
}

void VoxelMapper::markSurfaceViewPruningPending(
    const std::vector<std::shared_ptr<VoxelKeyframe>>& keyframes)
{
    if (!opt_params_.prune_surface_views_enable_) {
        return;
    }
    for (const auto& keyframe : keyframes) {
        if (keyframe) {
            surface_view_pending_keyframes_.insert(keyframe->fid_);
        }
    }
}

bool VoxelMapper::surfaceViewPruningReady()
{
    if (!opt_params_.prune_surface_views_enable_ ||
        disable_topology_changes_ ||
        !scene_ ||
        !voxel_model_ ||
        surface_view_pending_keyframes_.empty()) {
        return false;
    }

    for (auto it = surface_view_pending_keyframes_.begin();
         it != surface_view_pending_keyframes_.end();) {
        const auto keyframe_it = scene_->keyframes().find(*it);
        if (keyframe_it == scene_->keyframes().end() || !keyframe_it->second) {
            it = surface_view_pending_keyframes_.erase(it);
            continue;
        }
        if (keyframe_it->second->remaining_times_of_use_ > 0) {
            return false;
        }
        ++it;
    }

    if (surface_view_pending_keyframes_.empty()) {
        return false;
    }
    return scene_->keyframes().size() >=
        static_cast<std::size_t>(
            std::max(1, opt_params_.surface_view_window_size_));
}

void VoxelMapper::waitForInputQueueSlot()
{
    if (input_queue_max_keyframes_ <= 0) {
        return;
    }

    while (!isStopped() && !mpSLAM->isShutDown()) {
        const auto keyframe_ids = mpSLAM->getAtlas()->GetCurrentKeyFrameIds();
        if (keyframe_ids.empty()) {
            return;
        }

        if (!input_backpressure_ready_.load(std::memory_order_acquire)) {
            if (keyframe_ids.size() <= min_num_initial_map_kfs_) {
                return;
            }
        } else {
            const auto latest_orb_keyframe = static_cast<long long>(
                *std::max_element(keyframe_ids.begin(), keyframe_ids.end()));
            const auto latest_mapper_keyframe =
                latest_consumed_keyframe_id_.load(std::memory_order_acquire);
            if (latest_mapper_keyframe < 0 ||
                latest_orb_keyframe - latest_mapper_keyframe <=
                    input_queue_max_keyframes_) {
                return;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
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

float VoxelMapper::lambdaSsim()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.lambda_ssim_;
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
 void VoxelMapper::setLambdaSsim(const float lambda_ssim)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     opt_params_.lambda_ssim_ = lambda_ssim;
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
     params.lambda_ssim = opt_params_.lambda_ssim_;
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
     opt_params_.lambda_ssim_ = params.lambda_ssim;
     opt_params_.adapt_every_ = params.densify_interval;
     new_keyframe_times_of_use_ = params.new_kf_times_of_use;
     stable_num_iter_existence_ = params.stable_num_iter_existence;
     keep_training_ = params.keep_training;
     do_gaus_pyramid_training_ = params.do_gaus_pyramid_training;
     inactive_geo_densify_ = params.do_inactive_geo_densify;
 }
