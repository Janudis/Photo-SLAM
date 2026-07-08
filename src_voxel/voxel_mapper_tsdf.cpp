#include "include_voxel/voxel_mapper.h"
#include "include_voxel/voxel_mapper_utils.h"
#include "include/stereo_vision.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <opencv2/flann.hpp>
#include <random>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "ORB-SLAM3/include/Atlas.h"
#include "ORB-SLAM3/include/MapPoint.h"

namespace {
// sv::Camera  →  nvblox::Camera
inline nvblox::Camera toNvbloxCamera(const VoxelKeyframe& kf)
{
    const sv::Camera& cam = kf.cam_;
    // These should be the full undistorted image resolution.
    const int width  = kf.image_width_;
    const int height = kf.image_height_;

    return nvblox::Camera(
        cam.fx(), cam.fy(),   // fu, fv
        cam.cx(), cam.cy(),   // cu, cv
        width, height         // width, height
    );
}

// VoxelKeyframe (Tcw_) → nvblox::Transform T_L_C  (camera → layer/world)
inline nvblox::Transform toNvbloxTransform(VoxelKeyframe& kf)
{
    // Tcw_ is world→cam  ⇒  Twc = Tcw_.inverse() is cam→world
    Sophus::SE3f Tcw_f = kf.getPosef();       // SE3f world→cam
    Sophus::SE3f Twc_f = Tcw_f.inverse();     // SE3f cam→world

    nvblox::Transform T_L_C;
    T_L_C.linear()      = Twc_f.rotationMatrix();
    T_L_C.translation() = Twc_f.translation();
    return T_L_C;  // layer frame L == world frame W
}

// cv::Mat (CV_32FC1, meters) → nvblox::DepthImage (device)
inline void cvDepthToNvbloxDepth(const cv::Mat& depth_meters,
                                 nvblox::DepthImage* depth_img)
{
    using namespace nvblox;
    CHECK(depth_meters.type() == CV_32FC1);

    const int rows = depth_meters.rows;
    const int cols = depth_meters.cols;

    // Allocate / resize on device
    *depth_img = DepthImage(rows, cols, MemoryType::kDevice);

    DepthImage host_img(rows, cols, MemoryType::kHost);
    for (int r = 0; r < rows; ++r) {
        const float* src_row = depth_meters.ptr<float>(r);
        for (int c = 0; c < cols; ++c) {
            host_img(r, c) = src_row[c];
        }
    }

    // Host → device (synchronous)
    depth_img->copyFrom(host_img);
}

// Build an NVBlox camera whose resolution matches the depth image,
// and whose intrinsics are a scaled version of sv::Camera intrinsics.
inline nvblox::Camera makeNvbloxCameraFromDepthAndSvCam(
    const cv::Mat& depth_meters,
    const sv::Camera& cam)
{
    const int width  = depth_meters.cols;
    const int height = depth_meters.rows;

    CHECK(width  > 0);
    CHECK(height > 0);

    // Start from the Photo-SLAM camera intrinsics.
    float fx = cam.fx();
    float fy = cam.fy();
    float cx = cam.cx();
    float cy = cam.cy();

    const float cam_w = static_cast<float>(cam.width());
    const float cam_h = static_cast<float>(cam.height());

    // If Photo-SLAM uses a different internal resolution (e.g. 318x255),
    // rescale intrinsics to match the actual depth image size.
    if (cam_w > 0.0f && cam_h > 0.0f &&
        (static_cast<int>(cam_w) != width ||
         static_cast<int>(cam_h) != height))
    {
        const float scale_x = static_cast<float>(width)  / cam_w;
        const float scale_y = static_cast<float>(height) / cam_h;

        fx *= scale_x;
        fy *= scale_y;
        cx *= scale_x;
        cy *= scale_y;

        // std::cout << "[NVBLOX] Rescaling intrinsics for depth image: "
        //           << "cam(" << cam.width() << "x" << cam.height()
        //           << ") -> img(" << width << "x" << height << ") "
        //           << "scale_x=" << scale_x << " scale_y=" << scale_y
        //           << std::endl;
    }

    return nvblox::Camera(fx, fy, cx, cy, width, height);
}

void cvRgbToNvbloxColor(const cv::Mat& rgb_image,
                        nvblox::ColorImage* color_img,
                        nvblox::CudaStream* /*stream*/)
{
    CHECK(color_img != nullptr);
    CHECK(!rgb_image.empty());

    cv::Mat rgb_u8;
    if (rgb_image.type() == CV_8UC3) {
        rgb_u8 = rgb_image;
    } else {
        // Photo-SLAM often stores float images; convert to [0,255]
        rgb_image.convertTo(rgb_u8, CV_8UC3, 255.0);
    }

    const int rows = rgb_u8.rows;
    const int cols = rgb_u8.cols;

    // Host buffer
    nvblox::ColorImage color_host(rows, cols, nvblox::MemoryType::kHost);
    for (int v = 0; v < rows; ++v) {
        const cv::Vec3b* row_ptr = rgb_u8.ptr<cv::Vec3b>(v);
        for (int u = 0; u < cols; ++u) {
            const cv::Vec3b& c = row_ptr[u];
            nvblox::Color col;

            // The mapper stores keyframe images in RGB order before they reach
            // nvblox, so copy channels directly into nvblox::Color.
            col.r() = c[0];
            col.g() = c[1];
            col.b() = c[2];

            color_host(v, u) = col;
        }
    }

    // Allocate device image and copy
    *color_img = nvblox::ColorImage(rows, cols, nvblox::MemoryType::kDevice);
    // copyFrom(ImageBase) – no stream parameter in this overload
    color_img->copyFrom(color_host);
}


static bool depthMatToMeters(const cv::Mat& depth_in, cv::Mat& depth_meters)
{
    if (depth_in.empty()) {
        return false;
    }

    cv::Mat d = depth_in;
    if (d.channels() > 1) {
        cv::extractChannel(d, d, 0);
    }

    if (d.type() == CV_32FC1) {
        depth_meters = d;
        return true;
    }
    if (d.type() == CV_16UC1) {
        // Handle both common 16-bit conventions:
        // - mm depth (TUM-style): depth_m = raw / 1000
        // - Replica-style uint16 encoding: depth_m = raw / 6553.5
        double max_val = 0.0;
        cv::minMaxLoc(d, nullptr, &max_val);
        const double scale = (max_val > 20000.0) ? (1.0 / 6553.5) : (1.0 / 1000.0);
        d.convertTo(depth_meters, CV_32FC1, scale);
        return true;
    }

    d.convertTo(depth_meters, CV_32FC1);
    return true;
}

static bool loadReplicaDepthFromRgbPath(const std::string& rgb_filename, cv::Mat& depth_meters)
{
    if (rgb_filename.empty()) {
        return false;
    }

    const std::filesystem::path rgb_path(rgb_filename);
    if (!std::filesystem::exists(rgb_path)) {
        return false;
    }

    const std::string name = rgb_path.filename().string();
    if (name.rfind("frame", 0) != 0) {
        return false;
    }

    const std::filesystem::path parent = rgb_path.parent_path();
    const std::string stem = rgb_path.stem().string();   // e.g., frame000123
    const std::string suffix_stem = (stem.size() > 5 ? stem.substr(5) : std::string());
    const std::string suffix_name = name.substr(5);      // e.g., 000123.jpg

    std::vector<std::filesystem::path> candidates;
    candidates.push_back(parent / ("depth" + suffix_name));
    if (!suffix_stem.empty()) {
        candidates.push_back(parent / ("depth" + suffix_stem + ".png"));
        candidates.push_back(parent / ("depth" + suffix_stem + ".exr"));
        candidates.push_back(parent / ("depth" + suffix_stem + ".tiff"));
        candidates.push_back(parent / ("depth" + suffix_stem + ".tif"));
    }

    for (const auto& p : candidates) {
        if (!std::filesystem::exists(p)) {
            continue;
        }
        const cv::Mat depth_raw = cv::imread(p.string(), cv::IMREAD_UNCHANGED);
        if (depth_raw.empty()) {
            continue;
        }
        return depthMatToMeters(depth_raw, depth_meters);
    }

    return false;
}

static bool loadTumDepthFromRgbPath(const std::string& rgb_filename, cv::Mat& depth_meters)
{
    if (rgb_filename.empty()) {
        return false;
    }

    const std::filesystem::path rgb_path(rgb_filename);
    if (!std::filesystem::exists(rgb_path)) {
        return false;
    }

    const std::filesystem::path rgb_dir = rgb_path.parent_path();
    if (rgb_dir.filename() != "rgb") {
        return false;
    }

    const std::filesystem::path dataset_root = rgb_dir.parent_path();
    const std::filesystem::path depth_dir = dataset_root / "depth";
    const std::filesystem::path depth_txt = dataset_root / "depth.txt";
    if (!std::filesystem::exists(depth_dir)) {
        return false;
    }

    const std::string stem = rgb_path.stem().string();
    const std::filesystem::path exact_depth_path = depth_dir / (stem + rgb_path.extension().string());
    if (std::filesystem::exists(exact_depth_path)) {
        const cv::Mat depth_raw = cv::imread(exact_depth_path.string(), cv::IMREAD_UNCHANGED);
        if (!depth_raw.empty()) {
            return depthMatToMeters(depth_raw, depth_meters);
        }
    }

    double rgb_ts = 0.0;
    try {
        rgb_ts = std::stod(stem);
    } catch (...) {
        return false;
    }

    struct TumDepthIndexEntry {
        double timestamp = 0.0;
        std::filesystem::path path;
    };

    static std::mutex s_tum_depth_cache_mutex;
    static std::unordered_map<std::string, std::vector<TumDepthIndexEntry>> s_tum_depth_cache;

    std::vector<TumDepthIndexEntry> depth_index;
    {
        std::lock_guard<std::mutex> lock(s_tum_depth_cache_mutex);
        auto it = s_tum_depth_cache.find(dataset_root.string());
        if (it == s_tum_depth_cache.end()) {
            std::vector<TumDepthIndexEntry> parsed;
            if (std::filesystem::exists(depth_txt)) {
                std::ifstream in(depth_txt);
                std::string line;
                while (std::getline(in, line)) {
                    if (line.empty() || line[0] == '#') {
                        continue;
                    }
                    std::istringstream iss(line);
                    double ts = 0.0;
                    std::string rel_path;
                    if (!(iss >> ts >> rel_path)) {
                        continue;
                    }
                    std::filesystem::path p = dataset_root / rel_path;
                    if (!std::filesystem::exists(p)) {
                        continue;
                    }
                    parsed.push_back({ts, p});
                }
            }
            std::sort(
                parsed.begin(),
                parsed.end(),
                [](const TumDepthIndexEntry& a, const TumDepthIndexEntry& b) {
                    return a.timestamp < b.timestamp;
                });
            it = s_tum_depth_cache.emplace(dataset_root.string(), std::move(parsed)).first;
        }
        depth_index = it->second;
    }

    if (depth_index.empty()) {
        return false;
    }

    auto lb = std::lower_bound(
        depth_index.begin(),
        depth_index.end(),
        rgb_ts,
        [](const TumDepthIndexEntry& e, double t) {
            return e.timestamp < t;
        });

    auto best_it = depth_index.end();
    double best_dt = std::numeric_limits<double>::infinity();
    if (lb != depth_index.end()) {
        best_it = lb;
        best_dt = std::abs(lb->timestamp - rgb_ts);
    }
    if (lb != depth_index.begin()) {
        auto prev = std::prev(lb);
        const double prev_dt = std::abs(prev->timestamp - rgb_ts);
        if (prev_dt < best_dt) {
            best_it = prev;
            best_dt = prev_dt;
        }
    }

    constexpr double kTumMaxDepthAssocDeltaSec = 0.05;
    if (best_it == depth_index.end() || !(best_dt <= kTumMaxDepthAssocDeltaSec)) {
        return false;
    }

    const cv::Mat depth_raw = cv::imread(best_it->path.string(), cv::IMREAD_UNCHANGED);
    if (depth_raw.empty()) {
        return false;
    }
    return depthMatToMeters(depth_raw, depth_meters);
}

static torch::Tensor normalizeBoolMaskOrZeros(
    torch::Tensor mask,
    const int64_t N,
    const torch::Device& device)
{
    if (!mask.defined() || mask.numel() != N) {
        return torch::zeros(
            {N},
            torch::TensorOptions().dtype(torch::kBool).device(device));
    }
    if (mask.dim() == 2 && mask.size(1) == 1) {
        mask = mask.squeeze(1);
    } else if (mask.dim() != 1) {
        mask = mask.reshape({N});
    }
    return mask.to(device).to(torch::kBool).contiguous();
}

} // namespace

void VoxelMapper::initializeNvbloxMapper()
{
    using namespace nvblox;

    // 1) Configure where TSDF lives (device is standard)
    BlockMemoryPoolParams pool_params;
    pool_params.memory_type = MemoryType::kDevice;

    // 2) Create mapper that integrates TSDF (no freespace / occupancy)
    auto cuda_stream = std::make_shared<CudaStreamOwning>();
    sdf_mapper_ = std::make_shared<Mapper>(
        sdf_params_.sdf_voxel_size_m_,
        pool_params,
        ProjectiveLayerType::kTsdf,   // TSDF only
        cuda_stream
    );

    // 3) Set mapper params (defaults + small tweaks)
    MapperParams mapper_params;  // default-constructed

    mapper_params.esdf_integrator_params.esdf_integrator_max_distance_m = 5.0f;
    mapper_params.projective_integrator_params
        .projective_integrator_max_integration_distance_m = 4.0f;

    sdf_mapper_->setMapperParams(mapper_params);

    // Use every depth pixel when deciding which TSDF blocks to allocate.
    // The nvblox default subsamples this raycast, which can miss isolated
    // RGB-D points that we still insert into SVRaster.
    auto& tsdf_int = sdf_mapper_->tsdf_integrator();
    tsdf_int.view_calculator().raycast_subsampling_factor(1);

    // Now override appearance integrator settings explicitly.
    auto& color_int = sdf_mapper_->color_integrator();
    color_int.sphere_tracing_ray_subsampling_factor(1);
    color_int.view_calculator().raycast_subsampling_factor(1);

    // (Optional) if you ever use feature integration:
    // auto& feat_int = sdf_mapper_->feature_integrator();
    // feat_int.sphere_tracing_ray_subsampling_factor(1);
    // feat_int.view_calculator().raycast_subsampling_factor(1);
    // std::cout << "[NVBLOX] tsdf_integrator view_calculator.raycast_subsampling_factor = "
    //           << tsdf_int.view_calculator().raycast_subsampling_factor() << "\n";
    // std::cout << "[NVBLOX] color_integrator sphere_tracing_ray_subsampling_factor = "
    //           << color_int.sphere_tracing_ray_subsampling_factor() << "\n";
    // std::cout << "[NVBLOX] color_integrator view_calculator.raycast_subsampling_factor = "
    //           << color_int.view_calculator().raycast_subsampling_factor() << "\n";

    // --- DEBUG: print TSDF decay free distance and approximate truncation ---
    {
        nvblox::TsdfDecayIntegrator tsdf_decay;
        nvblox::FreespaceIntegrator freespace;
        std::cout << "[TEST] TsdfDecayIntegrator.free_distance_vox() = "
                << tsdf_decay.free_distance_vox() << " vox\n";
        std::cout << "[TEST] FreespaceIntegrator.max_tsdf_distance_for_occupancy_m() = "
                << freespace.max_tsdf_distance_for_occupancy_m() << " m\n";
    // // After sdf_mapper_ is constructed and setMapperParams() has been called:
    //     auto& decay = sdf_mapper_->tsdf_decay_integrator();
    //     std::cout << "[NVBLOX] TsdfDecayIntegrator.free_distance_vox() = "
    //               << decay.free_distance_vox() << " vox\n";

    //     auto& freespace = sdf_mapper_->freespace_integrator();
    //     std::cout << "[NVBLOX] FreespaceIntegrator.max_tsdf_distance_for_occupancy_m() = "
    //               << freespace.max_tsdf_distance_for_occupancy_m() << " m\n";
    }
}

void VoxelMapper::integrateKeyframeIntoNvblox(
    VoxelKeyframe& kf,
    const cv::Mat& depth_meters)
{
    if (!sdf_mapper_) {
        return;
    }

    const sv::Camera& cam = kf.cam_;

    // Depth resolution drives NVBlox camera resolution
    const int depth_w = depth_meters.cols;
    const int depth_h = depth_meters.rows;

    if (depth_w <= 0 || depth_h <= 0) {
        std::cout << "[NVBLOX] Warning: depth_meters has invalid size ("
                  << depth_w << "x" << depth_h << "), skipping integration.\n";
        return;
    }

    // Just for debugging, look at both RGB and depth sizes
    const int img_w = kf.img_undist_.cols;
    const int img_h = kf.img_undist_.rows;

    // std::cout << "[NVBLOX] integrate KF: "
    //           << "depth " << depth_w << "x" << depth_h
    //           << "  rgb "   << img_w   << "x" << img_h
    //           << "  cam.width()="  << cam.width()
    //           << " cam.height()=" << cam.height()
    //           << std::endl;

    // Build NVBlox camera whose resolution matches the depth image
    nvblox::Camera nvb_cam = makeNvbloxCameraFromDepthAndSvCam(depth_meters, cam);

    // 3) Pose (camera → world == layer)
    nvblox::Transform T_L_C = toNvbloxTransform(kf);

    // 4) Depth image → NVBlox
    static nvblox::DepthImage depth_img(nvblox::MemoryType::kDevice);
    cvDepthToNvbloxDepth(depth_meters, &depth_img);

    // Integrate TSDF
    sdf_mapper_->integrateDepth(depth_img, T_L_C, nvb_cam);

    // --- Color integration (optional) ---
    const cv::Mat& rgb_undistorted = kf.img_undist_;  // parallel color image

    if (!rgb_undistorted.empty()) {
        // It is nice (but not strictly required) that RGB matches depth size
        if (rgb_undistorted.cols != depth_w || rgb_undistorted.rows != depth_h) {
            std::cout << "[NVBLOX] Warning: RGB size ("
                      << rgb_undistorted.cols << "x" << rgb_undistorted.rows
                      << ") != depth size (" << depth_w << "x" << depth_h
                      << "). Color integration may be inconsistent." << std::endl;
        }

        static nvblox::ColorImage color_img(nvblox::MemoryType::kDevice);
        cvRgbToNvbloxColor(rgb_undistorted, &color_img, /*stream=*/nullptr);

        sdf_mapper_->integrateColor(color_img, T_L_C, nvb_cam);
    } else {
        std::cout << "[NVBLOX] kf.img_undist_ is empty, skipping color integration.\n";
    }
}

bool VoxelMapper::useSvrasterTsdfBackend() const
{
    return sdf_params_.tsdf_backend_ == "svraster";
}

bool VoxelMapper::useNvbloxTsdfBackend() const
{
    return sdf_params_.tsdf_backend_ == "nvblox";
}

float VoxelMapper::tsdfMetricVoxelSize() const
{
    if (useNvbloxTsdfBackend() && sdf_mapper_ && sdf_mapper_->tsdf_layer().size() > 0) {
        return static_cast<float>(sdf_mapper_->tsdf_layer().voxel_size());
    }
    if (useSvrasterTsdfBackend() && voxel_model_) {
        try {
            torch::Tensor sizes = voxel_model_->voxSize();
            if (sizes.defined() && sizes.numel() > 0) {
                const float finest_current_vox =
                    sizes.view({-1}).min().item<float>();
                if (std::isfinite(finest_current_vox) &&
                    finest_current_vox > 1.0e-6f) {
                    return finest_current_vox;
                }
            }
        } catch (const std::exception& e) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::cerr << "[TSDF] failed to read SVRaster finest voxel size; "
                          << "falling back to Mapper.tsdf_voxel_size_m: "
                          << e.what() << "\n";
            }
        }
    }
    return std::max(1.0e-6f, sdf_params_.sdf_voxel_size_m_);
}

bool VoxelMapper::hasTsdfForSampling() const
{
    if (useSvrasterTsdfBackend()) {
        return voxel_model_ &&
               voxel_model_->hasSvrasterSdfField() &&
               voxel_model_->svrasterSdfWeights().defined() &&
               voxel_model_->svrasterSdfWeights().numel() > 0 &&
               voxel_model_->svrasterSdfWeights().max().item<float>() > 0.0f;
    }
    return sdf_mapper_ && sdf_mapper_->tsdf_layer().size() > 0;
}

bool VoxelMapper::prepareSvrasterTsdfInitContext(const std::shared_ptr<VoxelKeyframe>& kf)
{
    clearSvrasterTsdfInitContext();
    if (!kf || sensor_type_ != RGBD || !useSvrasterTsdfBackend() || !sdf_params_.tsdf_density_init_) {
        return false;
    }

    cv::Mat depth_meters;
    if (!depthMatToMeters(kf->img_auxiliary_undist_, depth_meters) || depth_meters.empty()) {
        return false;
    }
    if (depth_meters.channels() > 1) {
        cv::extractChannel(depth_meters, depth_meters, 0);
    }
    if (depth_meters.type() != CV_32FC1) {
        depth_meters.convertTo(depth_meters, CV_32FC1);
    }

    const float fx = kf->cam_.fx();
    const float fy = kf->cam_.fy();
    if (fx <= 1.0e-6f || fy <= 1.0e-6f) {
        return false;
    }

    sdf_state_.svraster_tsdf_init_depth_meters_ = depth_meters;
    sdf_state_.svraster_tsdf_init_Tcw_ = kf->getPosef();
    sdf_state_.svraster_tsdf_init_fx_ = fx;
    sdf_state_.svraster_tsdf_init_fy_ = fy;
    sdf_state_.svraster_tsdf_init_cx_ = kf->cam_.cx();
    sdf_state_.svraster_tsdf_init_cy_ = kf->cam_.cy();
    sdf_state_.svraster_tsdf_init_width_ = depth_meters.cols;
    sdf_state_.svraster_tsdf_init_height_ = depth_meters.rows;
    sdf_state_.svraster_tsdf_init_kfid_ = kf->fid_;
    sdf_state_.svraster_tsdf_init_context_valid_ = true;
    return true;
}

void VoxelMapper::clearSvrasterTsdfInitContext()
{
    sdf_state_.svraster_tsdf_init_context_valid_ = false;
    sdf_state_.svraster_tsdf_init_depth_meters_.release();
    sdf_state_.svraster_tsdf_init_Tcw_ = Sophus::SE3f();
    sdf_state_.svraster_tsdf_init_fx_ = 0.0f;
    sdf_state_.svraster_tsdf_init_fy_ = 0.0f;
    sdf_state_.svraster_tsdf_init_cx_ = 0.0f;
    sdf_state_.svraster_tsdf_init_cy_ = 0.0f;
    sdf_state_.svraster_tsdf_init_width_ = 0;
    sdf_state_.svraster_tsdf_init_height_ = 0;
    sdf_state_.svraster_tsdf_init_kfid_ = 0;
}

torch::Tensor VoxelMapper::computeSvrasterProjectiveDensityInitForGridPoints(
    const torch::Tensor& grid_points_world,
    float ray_interval_m)
{
    TORCH_CHECK(
        grid_points_world.defined() &&
        grid_points_world.dim() == 2 &&
        grid_points_world.size(1) == 3,
        "computeSvrasterProjectiveDensityInitForGridPoints expects grid_points_world [N,3]");

    const int64_t N = grid_points_world.size(0);
    auto out_opts = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(grid_points_world.device());
    torch::Tensor raw_init = torch::full({N, 1}, -10.0f, out_opts);
    if (N == 0) {
        return raw_init;
    }
    if (N == 0 ||
        !sdf_state_.svraster_tsdf_init_context_valid_ ||
        sdf_state_.svraster_tsdf_init_depth_meters_.empty() ||
        sdf_state_.svraster_tsdf_init_width_ <= 0 ||
        sdf_state_.svraster_tsdf_init_height_ <= 0) {
        return raw_init;
    }

    torch::NoGradGuard no_grad;
    torch::Tensor pts_world =
        grid_points_world.to(mDevice).to(torch::kFloat32).contiguous();

    Eigen::Matrix3f Rcw = sdf_state_.svraster_tsdf_init_Tcw_.rotationMatrix();
    Eigen::Vector3f tcw = sdf_state_.svraster_tsdf_init_Tcw_.translation();
    std::vector<float> R_data = {
        Rcw(0, 0), Rcw(0, 1), Rcw(0, 2),
        Rcw(1, 0), Rcw(1, 1), Rcw(1, 2),
        Rcw(2, 0), Rcw(2, 1), Rcw(2, 2)};
    std::vector<float> t_data = {tcw.x(), tcw.y(), tcw.z()};
    torch::Tensor R = torch::from_blob(
                          R_data.data(),
                          {3, 3},
                          torch::TensorOptions().dtype(torch::kFloat32))
                          .clone()
                          .to(mDevice);
    torch::Tensor t = torch::from_blob(
                          t_data.data(),
                          {1, 3},
                          torch::TensorOptions().dtype(torch::kFloat32))
                          .clone()
                          .to(mDevice);

    torch::Tensor pts_cam = torch::matmul(pts_world, R.t()) + t;
    torch::Tensor x = pts_cam.index({torch::indexing::Slice(), 0});
    torch::Tensor y = pts_cam.index({torch::indexing::Slice(), 1});
    torch::Tensor z = pts_cam.index({torch::indexing::Slice(), 2});

    torch::Tensor z_safe = z.clamp_min(1.0e-6f);
    torch::Tensor u = sdf_state_.svraster_tsdf_init_fx_ * x / z_safe + sdf_state_.svraster_tsdf_init_cx_;
    torch::Tensor v = sdf_state_.svraster_tsdf_init_fy_ * y / z_safe + sdf_state_.svraster_tsdf_init_cy_;
    const int W = sdf_state_.svraster_tsdf_init_width_;
    const int H = sdf_state_.svraster_tsdf_init_height_;
    torch::Tensor in_image =
        torch::isfinite(z) &
        (z > RGBD_min_depth_) &
        (u >= 0.0f) & (u < static_cast<float>(W)) &
        (v >= 0.0f) & (v < static_cast<float>(H));
    if (sdf_params_.svraster_tsdf_max_integration_distance_m_ > 0.0f) {
        in_image = in_image & (z < sdf_params_.svraster_tsdf_max_integration_distance_m_);
    }

    cv::Mat depth_for_tensor = sdf_state_.svraster_tsdf_init_depth_meters_;
    torch::Tensor depth =
        tensor_utils::cvMat2TorchTensor_Float32(depth_for_tensor, device_type_)
            .to(mDevice)
            .to(torch::kFloat32)
            .contiguous();
    if (depth.dim() != 2) {
        return raw_init;
    }

    torch::Tensor u_safe = torch::where(torch::isfinite(u), u, torch::zeros_like(u));
    torch::Tensor v_safe = torch::where(torch::isfinite(v), v, torch::zeros_like(v));
    torch::Tensor u_idx =
        torch::floor(u_safe)
            .clamp(0.0f, static_cast<float>(W - 1))
            .to(torch::kLong);
    torch::Tensor v_idx =
        torch::floor(v_safe)
            .clamp(0.0f, static_cast<float>(H - 1))
            .to(torch::kLong);
    torch::Tensor depth_flat = depth.reshape({H * W});
    torch::Tensor sampled_depth =
        depth_flat.index_select(0, (v_idx * W + u_idx).to(torch::kLong)).view({N});

    const float trunc_m =
        std::max(1.0e-6f, sdf_params_.tsdf_density_init_trunc_vox_ * tsdfMetricVoxelSize());
    torch::Tensor sdf = sampled_depth - z;
    torch::Tensor valid =
        in_image &
        torch::isfinite(sampled_depth) &
        (sampled_depth > RGBD_min_depth_) &
        (sampled_depth < RGBD_max_depth_) &
        (z < sampled_depth + trunc_m);
    if (!valid.any().item<bool>()) {
        return raw_init;
    }

    const float a = std::max(1.0e-4f, sdf_params_.tsdf_density_init_bell_a_);
    const float b = std::clamp(sdf_params_.tsdf_density_init_bell_b_, 1.0e-4f, 0.9999f);
    const float u_bell =
        (2.0f - b + 2.0f * std::sqrt(std::max(0.0f, 1.0f - b))) / b;
    const float sharpness = std::log(u_bell) / a;
    torch::Tensor F = torch::clamp(sdf / trunc_m, -1.0f, 1.0f);
    torch::Tensor sigmoid = torch::sigmoid(-sharpness * F);
    torch::Tensor alpha_unit = 4.0f * sigmoid * (1.0f - sigmoid);
    const float alpha_min = std::clamp(sdf_params_.tsdf_density_init_alpha_min_, 1.0e-6f, 0.999f);
    const float alpha_max = std::clamp(sdf_params_.tsdf_density_init_alpha_max_, alpha_min, 0.999f);
    torch::Tensor alpha =
        torch::clamp(alpha_min + (alpha_max - alpha_min) * alpha_unit,
                     1.0e-6f,
                     0.999f);

    constexpr float kSvrasterStepSzScale = 100.0f;
    const float interval = std::max(1.0e-6f, ray_interval_m);
    torch::Tensor density =
        -torch::log(torch::clamp(1.0f - alpha, 1.0e-6f, 1.0f)) /
        (kSvrasterStepSzScale * interval);
    torch::Tensor density_safe = torch::clamp(density, 1.0e-12f);
    torch::Tensor raw_low =
        (torch::log(density_safe) + 0.904689820196f) / 0.909090909091f;
    torch::Tensor raw =
        torch::where(density > 1.1f, density, raw_low)
            .clamp(-20.0f, 20.0f)
            .view({N, 1});

    raw_init = torch::where(valid.view({N, 1}), raw, raw_init).contiguous();

    static int printed = 0;
    if (printed < 5) {
        ++printed;
        std::cout << "[SVRASTER TSDF DENSITY INIT] kf=" << sdf_state_.svraster_tsdf_init_kfid_
                  << " N=" << N
                  << " valid=" << valid.sum().item<int64_t>()
                  << " trunc_m=" << trunc_m
                  << " interval=" << interval
                  << "\n";
    }

    return raw_init.to(grid_points_world.device()).contiguous();
}

VoxelMapper::TsdfSample VoxelMapper::sampleTsdfAtPointsWorld(const torch::Tensor& pts_world)
{
    TORCH_CHECK(
        pts_world.defined() &&
        pts_world.dim() == 2 &&
        pts_world.size(1) == 3,
        "VoxelMapper::sampleTsdfAtPointsWorld expects pts_world of shape [N,3]");

    const auto N = pts_world.size(0);
    TsdfSample out;

    // Preserve device of input
    const bool input_on_cuda = pts_world.device().is_cuda();
    const auto out_device    = pts_world.device();

    // Default outputs (unknown)
    out.tsdf    = torch::full({N}, std::numeric_limits<float>::quiet_NaN(),
                              torch::TensorOptions().dtype(torch::kFloat32).device(out_device));
    out.weight  = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(out_device));
    out.success = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(out_device));

    // Arbitrary-point sampling is provided by nvblox. The SVRaster backend is
    // sampled directly through grid point keys in sampleTsdfAtSvrasterGridCornersWorld().
    if (!useNvbloxTsdfBackend() || !sdf_mapper_) {
        return out;
    }
    nvblox::TsdfLayer& tsdf_layer = sdf_mapper_->tsdf_layer();
    if (tsdf_layer.size() == 0 || N == 0) {
        return out;
    }

    // Query on CPU (nvblox API expects std::vector positions)
    torch::Tensor pts_cpu = pts_world.to(torch::kCPU).contiguous();
    auto acc_pts = pts_cpu.accessor<float, 2>();

    std::vector<nvblox::Vector3f> positions_L;
    positions_L.reserve(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i) {
        positions_L.emplace_back(acc_pts[i][0], acc_pts[i][1], acc_pts[i][2]);
    }

    std::vector<nvblox::TsdfVoxel> voxels;
    std::vector<bool> success_flags;
    tsdf_layer.getVoxels(positions_L, &voxels, &success_flags);

    const size_t M_vox = voxels.size();
    const size_t M_suc = success_flags.size();

    // Fill CPU buffers first (fast, avoids per-element device writes)
    torch::Tensor tsdf_cpu   = torch::full({N}, std::numeric_limits<float>::quiet_NaN(),
                                           torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    torch::Tensor w_cpu      = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    torch::Tensor succ_cpu   = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));

    auto acc_tsdf = tsdf_cpu.accessor<float, 1>();
    auto acc_w    = w_cpu.accessor<float, 1>();
    auto acc_succ = succ_cpu.accessor<bool, 1>();

    int64_t num_success = 0;
    int64_t num_fail    = 0;

    // Weight statistics over successful reads
    float w_min = std::numeric_limits<float>::infinity();
    float w_max = 0.0f;
    double w_sum = 0.0;

    for (int64_t i = 0; i < N; ++i) {
        const bool have_voxel =
            (static_cast<size_t>(i) < M_vox) &&
            (static_cast<size_t>(i) < M_suc) &&
            success_flags[i];

        if (!have_voxel) {
            ++num_fail;
            continue;
        }

        const auto& v = voxels[i];
        acc_tsdf[i] = v.distance;
        acc_w[i]    = v.weight;
        acc_succ[i] = true;

        ++num_success;

        w_min = std::min(w_min, v.weight);
        w_max = std::max(w_max, v.weight);
        w_sum += static_cast<double>(v.weight);
    }

    // Optional: print once or occasionally
    {
        static int64_t printed = 0;
        if (printed < 5) {  // print first few calls to verify sanity
            ++printed;
            const double suc_ratio = (N > 0) ? (100.0 * double(num_success) / double(N)) : 0.0;
            const double mean_w    = (num_success > 0) ? (w_sum / double(num_success)) : 0.0;

            std::cout << "[TSDF SAMPLE] N=" << N
                      << " success=" << num_success << " (" << suc_ratio << "%)"
                      << " fail=" << num_fail
                      << " w_min=" << (num_success > 0 ? w_min : 0.0f)
                      << " w_max=" << (num_success > 0 ? w_max : 0.0f)
                      << " w_mean=" << mean_w
                      << " tsdf_voxel_size=" << tsdf_layer.voxel_size()
                      << std::endl;
        }
    }

    // Move to original device
    if (input_on_cuda) {
        out.tsdf    = tsdf_cpu.to(out_device);
        out.weight  = w_cpu.to(out_device);
        out.success = succ_cpu.to(out_device);
    } else {
        out.tsdf    = tsdf_cpu;
        out.weight  = w_cpu;
        out.success = succ_cpu;
    }

    return out;
}

VoxelMapper::TsdfSample VoxelMapper::sampleNvbloxTsdfAtPointsWorld(const torch::Tensor& pts_world)
{
    TORCH_CHECK(
        pts_world.defined() &&
        pts_world.dim() == 2 &&
        pts_world.size(1) == 3,
        "VoxelMapper::sampleNvbloxTsdfAtPointsWorld expects pts_world of shape [N,3]");

    const auto N = pts_world.size(0);
    TsdfSample out;
    const bool input_on_cuda = pts_world.device().is_cuda();
    const auto out_device = pts_world.device();

    out.tsdf = torch::full({N}, std::numeric_limits<float>::quiet_NaN(),
                           torch::TensorOptions().dtype(torch::kFloat32).device(out_device));
    out.weight = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(out_device));
    out.success = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(out_device));

    if (!sdf_mapper_ || N == 0) {
        return out;
    }
    nvblox::TsdfLayer& tsdf_layer = sdf_mapper_->tsdf_layer();
    if (tsdf_layer.size() == 0) {
        return out;
    }

    torch::Tensor pts_cpu = pts_world.to(torch::kCPU).contiguous();
    auto acc_pts = pts_cpu.accessor<float, 2>();

    std::vector<nvblox::Vector3f> positions_L;
    positions_L.reserve(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i) {
        positions_L.emplace_back(acc_pts[i][0], acc_pts[i][1], acc_pts[i][2]);
    }

    std::vector<nvblox::TsdfVoxel> voxels;
    std::vector<bool> success_flags;
    tsdf_layer.getVoxels(positions_L, &voxels, &success_flags);

    torch::Tensor tsdf_cpu = torch::full({N}, std::numeric_limits<float>::quiet_NaN(),
                                         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    torch::Tensor w_cpu = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    torch::Tensor succ_cpu = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));

    auto acc_tsdf = tsdf_cpu.accessor<float, 1>();
    auto acc_w = w_cpu.accessor<float, 1>();
    auto acc_succ = succ_cpu.accessor<bool, 1>();
    const size_t M_vox = voxels.size();
    const size_t M_suc = success_flags.size();

    for (int64_t i = 0; i < N; ++i) {
        const bool have_voxel =
            static_cast<size_t>(i) < M_vox &&
            static_cast<size_t>(i) < M_suc &&
            success_flags[i];
        if (!have_voxel) {
            continue;
        }
        const auto& v = voxels[i];
        acc_tsdf[i] = v.distance;
        acc_w[i] = v.weight;
        acc_succ[i] = true;
    }

    if (input_on_cuda) {
        out.tsdf = tsdf_cpu.to(out_device);
        out.weight = w_cpu.to(out_device);
        out.success = succ_cpu.to(out_device);
    } else {
        out.tsdf = tsdf_cpu;
        out.weight = w_cpu;
        out.success = succ_cpu;
    }
    return out;
}

torch::Tensor VoxelMapper::computeTsdfDensityInitForGridPoints(
    const torch::Tensor& grid_points_world,
    float ray_interval_m)
{
    TORCH_CHECK(
        grid_points_world.defined() &&
        grid_points_world.dim() == 2 &&
        grid_points_world.size(1) == 3,
        "computeTsdfDensityInitForGridPoints expects grid_points_world [N,3]");

    const int64_t N = grid_points_world.size(0);
    if (N == 0) {
        return torch::empty(
            {0, 1},
            torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(grid_points_world.device()));
    }

    auto opts = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(grid_points_world.device());
    torch::Tensor raw_init = torch::full({N, 1}, -10.0f, opts);

    if (!sdf_params_.tsdf_density_init_ || sensor_type_ != RGBD) {
        return raw_init;
    }
    if (useSvrasterTsdfBackend()) {
        return computeSvrasterProjectiveDensityInitForGridPoints(
            grid_points_world,
            ray_interval_m);
    }
    if (!hasTsdfForSampling()) {
        return raw_init;
    }

    TsdfSample sample = sampleTsdfAtPointsWorld(grid_points_world);
    torch::Tensor tsdf = sample.tsdf.to(grid_points_world.device()).to(torch::kFloat32);
    torch::Tensor weight = sample.weight.to(grid_points_world.device()).to(torch::kFloat32);
    torch::Tensor success = sample.success.to(grid_points_world.device()).to(torch::kBool);

    const float min_weight = std::max(0.0f, sdf_params_.tsdf_density_init_min_weight_);
    torch::Tensor valid =
        (success & (weight >= min_weight) & torch::isfinite(tsdf)).to(torch::kBool);
    const int64_t valid_count = valid.sum().item<int64_t>();
    if (valid_count == 0) {
        return raw_init;
    }

    const float tsdf_vox_size = tsdfMetricVoxelSize();
    const float trunc_m =
        std::max(1.0e-6f, sdf_params_.tsdf_density_init_trunc_vox_ * tsdf_vox_size);

    const float a = std::max(1.0e-4f, sdf_params_.tsdf_density_init_bell_a_);
    const float b = std::clamp(sdf_params_.tsdf_density_init_bell_b_, 1.0e-4f, 0.9999f);
    const float u = (2.0f - b + 2.0f * std::sqrt(std::max(0.0f, 1.0f - b))) / b;
    const float sharpness = std::log(u) / a;

    torch::Tensor F = torch::clamp(tsdf / trunc_m, -1.0f, 1.0f);
    torch::Tensor sigmoid = torch::sigmoid(-sharpness * F);
    torch::Tensor alpha_unit = 4.0f * sigmoid * (1.0f - sigmoid);

    const float alpha_min = std::clamp(sdf_params_.tsdf_density_init_alpha_min_, 1.0e-6f, 0.999f);
    const float alpha_max = std::clamp(sdf_params_.tsdf_density_init_alpha_max_, alpha_min, 0.999f);
    torch::Tensor alpha = alpha_min + (alpha_max - alpha_min) * alpha_unit;
    alpha = torch::clamp(alpha, 1.0e-6f, 0.999f);

    constexpr float kSvrasterStepSzScale = 100.0f;
    const float interval = std::max(1.0e-6f, ray_interval_m);
    torch::Tensor density =
        -torch::log(torch::clamp(1.0f - alpha, 1.0e-6f, 1.0f)) /
        (kSvrasterStepSzScale * interval);

    torch::Tensor density_safe = torch::clamp(density, 1.0e-12f);
    torch::Tensor raw_low =
        (torch::log(density_safe) + 0.904689820196f) / 0.909090909091f;
    torch::Tensor raw = torch::where(density > 1.1f, density, raw_low);
    raw = torch::clamp(raw, -20.0f, 20.0f).view({N, 1});

    raw_init = torch::where(valid.view({N, 1}), raw, raw_init).contiguous();

    {
        static int printed = 0;
        if (printed < 5) {
            ++printed;
            torch::Tensor raw_valid = raw.index({valid});
            torch::Tensor alpha_valid = alpha.index({valid});
            const float raw_min = raw_valid.numel() > 0 ? raw_valid.min().item<float>() : -10.0f;
            const float raw_max = raw_valid.numel() > 0 ? raw_valid.max().item<float>() : -10.0f;
            const float alpha_mean = alpha_valid.numel() > 0 ? alpha_valid.mean().item<float>() : 0.0f;
            std::cout << "[TSDF DENSITY INIT] N=" << N
                      << " valid=" << valid_count
                      << " trunc_m=" << trunc_m
                      << " bell_a=" << a
                      << " bell_b=" << b
                      << " sharpness=" << sharpness
                      << " alpha_min=" << alpha_min
                      << " alpha_max=" << alpha_max
                      << " alpha_mean_valid=" << alpha_mean
                      << " raw_valid_min=" << raw_min
                      << " raw_valid_max=" << raw_max
                      << "\n";
        }
    }

    return raw_init;
}

VoxelMapper::TsdfCornerSample VoxelMapper::sampleTsdfAtSvrasterGridCornersWorld(
    bool include_points_world)
{
    TORCH_CHECK(voxel_model_ != nullptr, "voxel_model_ is null");

    torch::Tensor grid_key = voxel_model_->gridPtsKey(); // [M,3] int64
    torch::Tensor vox_key  = voxel_model_->voxKey();     // [N,8] int64

    TORCH_CHECK(grid_key.defined() && grid_key.dim() == 2 && grid_key.size(1) == 3,
                "gridPtsKey must be [M,3]");
    TORCH_CHECK(vox_key.defined() && vox_key.dim() == 2 && vox_key.size(1) == 8,
                "voxKey must be [N,8]");

    const int64_t M = grid_key.size(0);
    const int64_t N = vox_key.size(0);

    TsdfCornerSample out;
    auto dev = voxel_model_->voxCenter().device();

    // Default (unknown)
    out.tsdf    = torch::full({N, 8}, std::numeric_limits<float>::quiet_NaN(),
                              torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    out.weight  = torch::zeros({N, 8}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    out.success = torch::zeros({N, 8}, torch::TensorOptions().dtype(torch::kBool).device(dev));
    out.points_world = torch::Tensor();

    if (N == 0 || M == 0 || !hasTsdfForSampling()) {
        return out;
    }

    // 1) grid_key -> world xyz
    torch::Tensor grid_xyz = voxel_model_->gridPointsWorld();
    if (grid_xyz.device() != dev) grid_xyz = grid_xyz.to(dev);

    TsdfSample g;
    if (useSvrasterTsdfBackend()) {
        voxel_model_->ensureSvrasterSdfField();
        torch::Tensor tsdf_grid = voxel_model_->svrasterSdfGridPts();
        torch::Tensor weight_grid = voxel_model_->svrasterSdfWeights();
        if (!tsdf_grid.defined() || !weight_grid.defined() ||
            tsdf_grid.size(0) != M || weight_grid.size(0) != M) {
            return out;
        }
        g.tsdf = tsdf_grid.to(dev).to(torch::kFloat32).view({M});
        g.weight = weight_grid.to(dev).to(torch::kFloat32).view({M});
        g.success = (g.weight > 0.0f).to(torch::kBool);
    } else {
        // 2) Sample nvblox TSDF at all SVRaster grid points
        g = sampleTsdfAtPointsWorld(grid_xyz); // g.tsdf,g.weight,g.success are [M]
    }

    // 3) Gather per-voxel corners using vox_key
    // Flatten indices: [N,8] -> [N*8]
    torch::Tensor idx = vox_key.to(dev).to(torch::kLong).reshape({-1}); // [N*8]
    TORCH_CHECK(idx.numel() == N * 8, "vox_key reshape mismatch");

    torch::Tensor tsdf8 = g.tsdf.index_select(0, idx).view({N, 8}).contiguous();
    torch::Tensor w8    = g.weight.index_select(0, idx).view({N, 8}).contiguous();
    torch::Tensor ok8   = g.success.index_select(0, idx).view({N, 8}).contiguous();
    out.tsdf    = tsdf8;
    out.weight  = w8;
    out.success = ok8;
    if (include_points_world) {
        out.points_world =
            grid_xyz.index_select(0, idx).view({N, 8, 3}).contiguous();
    }
    return out;
}

VoxelMapper::TsdfSupportMasks VoxelMapper::computeTsdfSupportMasksFromCorners(
    const TsdfCornerSample& corners,
    float min_weight,
    float surface_band_vox,
    int min_valid_corners) const
{
    TsdfSupportMasks out;
    if (!corners.tsdf.defined() ||
        !corners.weight.defined() ||
        !corners.success.defined() ||
        corners.tsdf.dim() != 2 ||
        corners.weight.dim() != 2 ||
        corners.success.dim() != 2 ||
        corners.tsdf.size(1) != 8 ||
        corners.weight.size(1) != 8 ||
        corners.success.size(1) != 8 ||
        corners.tsdf.size(0) != corners.weight.size(0) ||
        corners.tsdf.size(0) != corners.success.size(0)) {
        return out;
    }

    torch::Tensor tsdf8 = corners.tsdf.to(torch::kFloat32).contiguous();
    torch::Tensor w8 = corners.weight.to(tsdf8.device()).to(torch::kFloat32).contiguous();
    torch::Tensor ok8 = corners.success.to(tsdf8.device()).to(torch::kBool).contiguous();
    const int64_t N = tsdf8.size(0);
    const int valid_corner_count = std::max(1, std::min(8, min_valid_corners));
    const float min_w = std::max(0.0f, min_weight);
    const float band_m = std::max(0.0f, surface_band_vox) * tsdfMetricVoxelSize();

    torch::Tensor finite = torch::isfinite(tsdf8).to(torch::kBool);
    torch::Tensor corner_valid = (ok8 & finite & (w8 >= min_w)).to(torch::kBool);
    torch::Tensor valid_count = corner_valid.to(torch::kInt32).sum(/*dim=*/1);
    out.valid = (valid_count >= valid_corner_count).to(torch::kBool);

    torch::Tensor pos =
        (corner_valid & (tsdf8 > 0.0f)).to(torch::kBool).any(/*dim=*/1);
    torch::Tensor neg =
        (corner_valid & (tsdf8 < 0.0f)).to(torch::kBool).any(/*dim=*/1);
    torch::Tensor zero =
        (corner_valid & (torch::abs(tsdf8) <= 1.0e-6f))
            .to(torch::kBool)
            .any(/*dim=*/1);
    out.has_zero_crossing = (out.valid & (zero | (pos & neg))).to(torch::kBool);

    torch::Tensor abs_for_min =
        torch::where(
            corner_valid,
            torch::abs(tsdf8),
            torch::full_like(tsdf8, std::numeric_limits<float>::infinity()));
    torch::Tensor min_abs = std::get<0>(abs_for_min.min(/*dim=*/1));
    out.near_surface = (out.valid & (min_abs <= band_m)).to(torch::kBool);
    out.surface_support =
        (out.valid & (out.has_zero_crossing | out.near_surface)).to(torch::kBool);
    out.free_space =
        (out.valid & pos & (~neg) & (~zero) & (min_abs > band_m)).to(torch::kBool);

    return out;
}

void VoxelMapper::integrateKeyframeIntoSvrasterSdf(
    VoxelKeyframe& kf,
    const cv::Mat& depth_meters)
{
    if (!voxel_model_ || !sdf_params_.use_tsdf_mapping_ || sensor_type_ != RGBD ||
        !useSvrasterTsdfBackend() || depth_meters.empty()) {
        return;
    }
    if (!voxel_model_->gridPtsKey().defined() || voxel_model_->gridPtsKey().numel() == 0) {
        return;
    }

    cv::Mat depth32 = depth_meters;
    if (depth32.channels() > 1) {
        cv::extractChannel(depth32, depth32, 0);
    }
    if (depth32.type() != CV_32FC1) {
        depth32.convertTo(depth32, CV_32FC1);
    }

    const int H = depth32.rows;
    const int W = depth32.cols;
    if (H <= 0 || W <= 0) {
        return;
    }

    torch::NoGradGuard no_grad;
    voxel_model_->ensureSvrasterSdfField();

    torch::Tensor grid_world = voxel_model_->gridPointsWorld().to(mDevice).to(torch::kFloat32).contiguous();
    const int64_t M = grid_world.size(0);
    if (M == 0) {
        return;
    }

    Sophus::SE3f Tcw = kf.getPosef();
    Eigen::Matrix3f Rcw = Tcw.rotationMatrix();
    Eigen::Vector3f tcw = Tcw.translation();
    std::vector<float> R_data = {
        Rcw(0, 0), Rcw(0, 1), Rcw(0, 2),
        Rcw(1, 0), Rcw(1, 1), Rcw(1, 2),
        Rcw(2, 0), Rcw(2, 1), Rcw(2, 2)};
    std::vector<float> t_data = {tcw.x(), tcw.y(), tcw.z()};
    torch::Tensor R = torch::from_blob(
                          R_data.data(),
                          {3, 3},
                          torch::TensorOptions().dtype(torch::kFloat32))
                          .clone()
                          .to(mDevice);
    torch::Tensor t = torch::from_blob(
                          t_data.data(),
                          {1, 3},
                          torch::TensorOptions().dtype(torch::kFloat32))
                          .clone()
                          .to(mDevice);

    torch::Tensor grid_cam = torch::matmul(grid_world, R.t()) + t; // [M,3]
    torch::Tensor x = grid_cam.index({torch::indexing::Slice(), 0});
    torch::Tensor y = grid_cam.index({torch::indexing::Slice(), 1});
    torch::Tensor z = grid_cam.index({torch::indexing::Slice(), 2});

    const float fx = kf.cam_.fx();
    const float fy = kf.cam_.fy();
    const float cx = kf.cam_.cx();
    const float cy = kf.cam_.cy();
    sdf_state_.svraster_tsdf_last_context_valid_ = true;
    sdf_state_.svraster_tsdf_last_depth_meters_ = depth32.clone();
    sdf_state_.svraster_tsdf_last_Tcw_ = Tcw;
    sdf_state_.svraster_tsdf_last_fx_ = fx;
    sdf_state_.svraster_tsdf_last_fy_ = fy;
    sdf_state_.svraster_tsdf_last_cx_ = cx;
    sdf_state_.svraster_tsdf_last_cy_ = cy;
    sdf_state_.svraster_tsdf_last_width_ = W;
    sdf_state_.svraster_tsdf_last_height_ = H;
    sdf_state_.svraster_tsdf_last_kfid_ = kf.fid_;

    torch::Tensor z_safe = z.clamp_min(1.0e-6f);
    torch::Tensor u = fx * x / z_safe + cx;
    torch::Tensor v = fy * y / z_safe + cy;

    torch::Tensor in_image =
        torch::isfinite(z) &
        (z > RGBD_min_depth_) &
        (u >= 0.0f) & (u < static_cast<float>(W)) &
        (v >= 0.0f) & (v < static_cast<float>(H));
    if (sdf_params_.svraster_tsdf_max_integration_distance_m_ > 0.0f) {
        in_image = in_image & (z < sdf_params_.svraster_tsdf_max_integration_distance_m_);
    }

    cv::Mat depth_for_tensor = depth32;
    torch::Tensor depth = tensor_utils::cvMat2TorchTensor_Float32(depth_for_tensor, device_type_)
                              .to(mDevice)
                              .to(torch::kFloat32)
                              .contiguous();
    if (depth.dim() != 2) {
        return;
    }

    torch::Tensor u_safe = torch::where(torch::isfinite(u), u, torch::zeros_like(u));
    torch::Tensor v_safe = torch::where(torch::isfinite(v), v, torch::zeros_like(v));
    torch::Tensor u_idx =
        torch::floor(u_safe)
            .clamp(0.0f, static_cast<float>(W - 1))
            .to(torch::kLong);
    torch::Tensor v_idx =
        torch::floor(v_safe)
            .clamp(0.0f, static_cast<float>(H - 1))
            .to(torch::kLong);
    torch::Tensor depth_flat = depth.reshape({H * W});
    torch::Tensor sampled_depth =
        depth_flat.index_select(0, (v_idx * W + u_idx).to(torch::kLong)).view({M});

    const float trunc_m =
        std::max(1.0e-6f, sdf_params_.tsdf_density_init_trunc_vox_ * tsdfMetricVoxelSize());
    torch::Tensor sdf = sampled_depth - z;
    torch::Tensor valid =
        in_image &
        torch::isfinite(sampled_depth) &
        (sampled_depth > RGBD_min_depth_) &
        (sampled_depth < RGBD_max_depth_) &
        (z < sampled_depth + trunc_m);

    torch::Tensor tsdf_metric = torch::clamp(sdf, -trunc_m, trunc_m);
    torch::Tensor weights = valid.to(torch::kFloat32);
    if (sdf_params_.svraster_tsdf_inverse_square_weighting_) {
        torch::Tensor unit_near = torch::ones_like(z);
        torch::Tensor z_weight = z.clamp_min(1.0e-2f);
        torch::Tensor inverse_square = 1.0f / (z_weight * z_weight);
        torch::Tensor measurement_weight =
            torch::where(z <= 1.0e-2f, unit_near, inverse_square);
        measurement_weight = torch::where(
            torch::isfinite(measurement_weight),
            measurement_weight,
            torch::zeros_like(measurement_weight));
        weights = torch::where(valid, measurement_weight, torch::zeros_like(measurement_weight));
    }
    const bool has_valid_sdf_update = valid.any().item<bool>();
    voxel_model_->fuseSvrasterSdfGridSamples(
        tsdf_metric, weights, valid, sdf_params_.svraster_tsdf_max_weight_);
    if (has_valid_sdf_update && rerun_params_.enable_rerun_) {
        torch::Tensor Tcw_cpu = torch::empty(
            {4, 4},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        auto Tacc = Tcw_cpu.accessor<float, 2>();
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                Tacc[r][c] = Rcw(r, c);
            }
        }
        Tacc[0][3] = tcw.x();
        Tacc[1][3] = tcw.y();
        Tacc[2][3] = tcw.z();
        Tacc[3][0] = 0.0f;
        Tacc[3][1] = 0.0f;
        Tacc[3][2] = 0.0f;
        Tacc[3][3] = 1.0f;

        rerun_state_.rerun_gt_sdf_log_pending_ = true;
        rerun_state_.rerun_gt_sdf_pending_kfid_ = kf.fid_;
        rerun_state_.rerun_gt_sdf_pending_Tcw_cpu_ = Tcw_cpu;
        rerun_state_.rerun_gt_sdf_pending_fx_ = fx;
        rerun_state_.rerun_gt_sdf_pending_fy_ = fy;
        rerun_state_.rerun_gt_sdf_pending_cx_ = cx;
        rerun_state_.rerun_gt_sdf_pending_cy_ = cy;
        rerun_state_.rerun_gt_sdf_pending_width_ = W;
        rerun_state_.rerun_gt_sdf_pending_height_ = H;
    }

    if (sdf_params_.tsdf_density_init_) {
        torch::Tensor tsdf_all =
            voxel_model_->svrasterSdfGridPts().to(mDevice).to(torch::kFloat32).view({M});
        torch::Tensor weight_all =
            voxel_model_->svrasterSdfWeights().to(mDevice).to(torch::kFloat32).view({M});
        torch::Tensor geo_all =
            voxel_model_->geoGridPts().to(mDevice).to(torch::kFloat32).view({M});

        torch::Tensor init_valid =
            torch::isfinite(tsdf_all) &
            (weight_all >= std::max(0.0f, sdf_params_.tsdf_density_init_min_weight_)) &
            (geo_all <= -9.5f);

        if (init_valid.any().item<bool>()) {
            const float a = std::max(1.0e-4f, sdf_params_.tsdf_density_init_bell_a_);
            const float b = std::clamp(sdf_params_.tsdf_density_init_bell_b_, 1.0e-4f, 0.9999f);
            const float u_bell =
                (2.0f - b + 2.0f * std::sqrt(std::max(0.0f, 1.0f - b))) / b;
            const float sharpness = std::log(u_bell) / a;
            torch::Tensor F = torch::clamp(tsdf_all / trunc_m, -1.0f, 1.0f);
            torch::Tensor sigmoid = torch::sigmoid(-sharpness * F);
            torch::Tensor alpha_unit = 4.0f * sigmoid * (1.0f - sigmoid);
            const float alpha_min =
                std::clamp(sdf_params_.tsdf_density_init_alpha_min_, 1.0e-6f, 0.999f);
            const float alpha_max =
                std::clamp(sdf_params_.tsdf_density_init_alpha_max_, alpha_min, 0.999f);
            torch::Tensor alpha =
                torch::clamp(alpha_min + (alpha_max - alpha_min) * alpha_unit,
                             1.0e-6f,
                             0.999f);

            constexpr float kSvrasterStepSzScale = 100.0f;
            const float interval = std::max(1.0e-6f, voxel_model_->fixed_vox_size_);
            torch::Tensor density =
                -torch::log(torch::clamp(1.0f - alpha, 1.0e-6f, 1.0f)) /
                (kSvrasterStepSzScale * interval);
            torch::Tensor density_safe = torch::clamp(density, 1.0e-12f);
            torch::Tensor raw_low =
                (torch::log(density_safe) + 0.904689820196f) / 0.909090909091f;
            torch::Tensor raw = torch::where(density > 1.1f, density, raw_low)
                                    .clamp(-20.0f, 20.0f)
                                    .view({M, 1});
            voxel_model_->applyGeoGridRawInit(raw, init_valid);
        }
    }

    static int printed = 0;
    if (printed < 5) {
        ++printed;
        const int64_t valid_count = valid.sum().item<int64_t>();
        std::cout << "[SVRASTER TSDF] kf=" << kf.fid_
                  << " grid_pts=" << M
                  << " fused=" << valid_count
                  << " trunc_m=" << trunc_m
                  << " backend=" << sdf_params_.tsdf_backend_
                  << "\n";
    }
}

void VoxelMapper::integrateKeyframeSdfEvidenceVoxels(
    const std::shared_ptr<VoxelKeyframe>& kf,
    const cv::Mat& depth_meters)
{
    if (!kf || !voxel_model_ ||
        !sdf_params_.sdf_evidence_densify_ ||
        !sdf_params_.use_tsdf_mapping_ ||
        sensor_type_ != RGBD ||
        !useSvrasterTsdfBackend() ||
        depth_meters.empty() ||
        kf->image_width_ <= 0 ||
        kf->image_height_ <= 0) {
        return;
    }

    cv::Mat depth32 = depth_meters;
    if (depth32.channels() > 1) {
        cv::extractChannel(depth32, depth32, 0);
    }
    if (depth32.type() != CV_32FC1) {
        depth32.convertTo(depth32, CV_32FC1);
    }
    if (depth32.empty()) {
        return;
    }

    if (kf->cam_.model_id_ != sv::Camera::PINHOLE) {
        return;
    }

    const float fx = kf->cam_.fx();
    const float fy = kf->cam_.fy();
    const float cx = kf->cam_.cx();
    const float cy = kf->cam_.cy();
    if (!(fx > 0.0f && fy > 0.0f)) {
        return;
    }

    const float vox_m = std::max(1.0e-6f, tsdfMetricVoxelSize());
    const float trunc_m =
        std::max(1.0e-6f, sdf_params_.tsdf_density_init_trunc_vox_ * vox_m);
    const float sample_step = vox_m;

    const cv::Mat& image = kf->img_undist_;
    const bool has_color =
        !image.empty() &&
        image.rows == depth32.rows &&
        image.cols == depth32.cols &&
        image.channels() >= 3;
    auto read_color = [&](int y, int x, float& r, float& g, float& b) {
        r = g = b = 0.5f;
        if (!has_color) {
            return;
        }
        if (image.depth() == CV_32F) {
            const cv::Vec3f c = image.at<cv::Vec3f>(y, x);
            r = std::clamp(c[0], 0.0f, 1.0f);
            g = std::clamp(c[1], 0.0f, 1.0f);
            b = std::clamp(c[2], 0.0f, 1.0f);
        } else if (image.depth() == CV_8U) {
            const cv::Vec3b c = image.at<cv::Vec3b>(y, x);
            r = static_cast<float>(c[0]) / 255.0f;
            g = static_cast<float>(c[1]) / 255.0f;
            b = static_cast<float>(c[2]) / 255.0f;
        }
    };

    std::vector<float> pts_cam;
    std::vector<float> colors;
    const int samples_per_full_ray =
        std::max(1, static_cast<int>(std::ceil((2.0f * trunc_m) / sample_step)) + 1);
    pts_cam.reserve(static_cast<size_t>(1024) * static_cast<size_t>(samples_per_full_ray) * 3);
    colors.reserve(static_cast<size_t>(1024) * static_cast<size_t>(samples_per_full_ray) * 3);

    for (int y = 0; y < depth32.rows; ++y) {
        const float* depth_row = depth32.ptr<float>(y);
        for (int x = 0; x < depth32.cols; ++x) {
            const float d = depth_row[x];
            if (!std::isfinite(d) ||
                d <= RGBD_min_depth_ ||
                d >= RGBD_max_depth_) {
                continue;
            }
            if (sdf_params_.svraster_tsdf_max_integration_distance_m_ > 0.0f &&
                d > sdf_params_.svraster_tsdf_max_integration_distance_m_) {
                continue;
            }

            const float z_min = std::max(RGBD_min_depth_, d - trunc_m);
            const float z_max = std::min(RGBD_max_depth_, d + trunc_m);
            const int samples =
                std::max(1, static_cast<int>(std::ceil((z_max - z_min) / sample_step)) + 1);
            float r, g, b;
            read_color(y, x, r, g, b);
            for (int s = 0; s < samples; ++s) {
                const float alpha =
                    (samples == 1)
                        ? 0.0f
                        : static_cast<float>(s) / static_cast<float>(samples - 1);
                const float z = z_min + alpha * (z_max - z_min);
                if (!std::isfinite(z) ||
                    z <= RGBD_min_depth_ ||
                    z >= RGBD_max_depth_) {
                    continue;
                }
                const float x_cam = (static_cast<float>(x) - cx) / fx * z;
                const float y_cam = (static_cast<float>(y) - cy) / fy * z;
                pts_cam.push_back(x_cam);
                pts_cam.push_back(y_cam);
                pts_cam.push_back(z);
                colors.push_back(r);
                colors.push_back(g);
                colors.push_back(b);
            }
        }
    }

    if (pts_cam.empty()) {
        return;
    }

    if (rgbd_fill_render_holes_max_points_per_kf_ > 0 &&
        static_cast<int64_t>(pts_cam.size() / 3) >
            static_cast<int64_t>(rgbd_fill_render_holes_max_points_per_kf_)) {
        const int64_t total = static_cast<int64_t>(pts_cam.size() / 3);
        const int64_t keep_n = std::max<int64_t>(1, rgbd_fill_render_holes_max_points_per_kf_);
        std::vector<float> pts_keep;
        std::vector<float> colors_keep;
        pts_keep.reserve(static_cast<size_t>(keep_n) * 3);
        colors_keep.reserve(static_cast<size_t>(keep_n) * 3);
        const double step =
            (keep_n == 1)
                ? 0.0
                : static_cast<double>(total - 1) / static_cast<double>(keep_n - 1);
        for (int64_t i = 0; i < keep_n; ++i) {
            const int64_t src =
                (keep_n == 1)
                    ? total / 2
                    : std::min<int64_t>(
                          total - 1,
                          static_cast<int64_t>(std::llround(step * static_cast<double>(i))));
            for (int c = 0; c < 3; ++c) {
                pts_keep.push_back(pts_cam[static_cast<size_t>(3 * src + c)]);
                colors_keep.push_back(colors[static_cast<size_t>(3 * src + c)]);
            }
        }
        pts_cam.swap(pts_keep);
        colors.swap(colors_keep);
    }

    torch::Tensor points =
        torch::from_blob(
            pts_cam.data(),
            {static_cast<int64_t>(pts_cam.size() / 3), 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
            .clone()
            .to(device_type_);
    torch::Tensor color_tensor =
        torch::from_blob(
            colors.data(),
            {static_cast<int64_t>(colors.size() / 3), 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
            .clone()
            .to(device_type_);

    Sophus::SE3f Twc = kf->getPosef().inverse();
    torch::Tensor Twc_tensor =
        tensor_utils::EigenMatrix2TorchTensor(Twc.matrix(), device_type_).transpose(0, 1);
    voxel_utils::transformPoints(points, Twc_tensor);

    std::vector<sv::MiniCam> tr_cams;
    tr_cams.reserve(scene_->keyframes().size());
    for (const auto& kv : scene_->keyframes()) {
        if (kv.second) {
            tr_cams.push_back(
                kv.second->toMiniCam(kv.second->image_height_, kv.second->image_width_));
        }
    }

    std::unique_lock<std::mutex> lock_render(mutex_render_);
    voxel_model_->setNextRealInsertionRerunEntityPath("world/sdf_evidence/created");
    voxel_model_->setNextInsertionVoxelState(sv::VoxelRuntimeState::SdfEvidenceOnly);
    voxel_model_->increasePcd(points, color_tensor, getIteration(), tr_cams);
    voxel_model_->setNextInsertionVoxelState(sv::VoxelRuntimeState::ActiveRenderable);
    voxel_model_->setNextRealInsertionRerunEntityPath("");
}

void VoxelMapper::promoteSdfEvidenceVoxelsFromTsdfField()
{
    if (!voxel_model_ ||
        !scene_ ||
        !sdf_params_.sdf_evidence_densify_ ||
        !sdf_params_.use_tsdf_mapping_ ||
        sensor_type_ != RGBD ||
        !useSvrasterTsdfBackend()) {
        return;
    }

    torch::NoGradGuard no_grad;
    torch::Tensor evidence_mask = voxel_model_->sdfEvidenceOnlyMask();
    if (!evidence_mask.defined() || evidence_mask.numel() == 0 ||
        !evidence_mask.any().item<bool>()) {
        return;
    }

    TsdfCornerSample corners =
        sampleTsdfAtSvrasterGridCornersWorld(
            sdf_params_.sdf_evidence_zero_crossing_refresh_);
    if (!corners.tsdf.defined() ||
        corners.tsdf.dim() != 2 ||
        corners.tsdf.size(1) != 8) {
        return;
    }
    torch::Tensor tsdf8 = corners.tsdf.to(device_type_).to(torch::kFloat32).contiguous();
    torch::Tensor w8 = corners.weight.to(device_type_).to(torch::kFloat32).contiguous();
    torch::Tensor ok8 = corners.success.to(device_type_).to(torch::kBool).contiguous();
    const int64_t N = tsdf8.size(0);
    if (evidence_mask.numel() != N) {
        return;
    }

    evidence_mask = evidence_mask.to(device_type_).to(torch::kBool).contiguous().view({N});
    const float min_weight = std::max(0.0f, sdf_params_.tsdf_prune_min_weight_);
    const float surface_band =
        std::max(0.0f, sdf_params_.tsdf_prune_surface_band_vox_) *
        tsdfMetricVoxelSize();
    const float trunc_m =
        std::max(1.0e-6f, sdf_params_.tsdf_density_init_trunc_vox_ * tsdfMetricVoxelSize());
    torch::Tensor corner_valid = (ok8 & (w8 >= min_weight)).to(torch::kBool);
    torch::Tensor valid_count = corner_valid.to(torch::kInt32).sum(/*dim=*/1);
    torch::Tensor tsdf_valid_for_min =
        torch::where(corner_valid, tsdf8, torch::full_like(tsdf8, trunc_m));
    torch::Tensor tsdf_valid_for_max =
        torch::where(corner_valid, tsdf8, torch::full_like(tsdf8, -trunc_m));
    torch::Tensor min_tsdf = std::get<0>(tsdf_valid_for_min.min(/*dim=*/1));
    torch::Tensor max_tsdf = std::get<0>(tsdf_valid_for_max.max(/*dim=*/1));
    torch::Tensor surface_or_crossing_support =
        ((valid_count >= std::max(1, sdf_params_.tsdf_prune_min_valid_corners_)) &
         (min_tsdf <= surface_band) &
         (max_tsdf >= -surface_band))
            .to(torch::kBool);

    torch::Tensor edge_surface_support =
        torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    std::vector<torch::Tensor> zero_crossing_points;
    std::vector<torch::Tensor> zero_crossing_colors;
    if (sdf_params_.sdf_evidence_zero_crossing_refresh_ &&
        corners.points_world.defined() &&
        corners.points_world.dim() == 3 &&
        corners.points_world.size(0) == N &&
        corners.points_world.size(1) == 8 &&
        corners.points_world.size(2) == 3) {
        using torch::indexing::Slice;
        torch::Tensor points8 =
            corners.points_world.to(device_type_).to(torch::kFloat32).contiguous();
        torch::Tensor sh0 = voxel_model_->sh0().detach();
        torch::Tensor colors_all;
        if (sh0.defined() && sh0.numel() > 0) {
            sh0 = sh0.to(device_type_).to(torch::kFloat32);
            if (sh0.dim() == 3 && sh0.size(1) == 1 && sh0.size(2) == 3) {
                sh0 = sh0.view({sh0.size(0), 3});
            }
            if (sh0.dim() == 2 && sh0.size(0) == N && sh0.size(1) == 3) {
                constexpr float kSHC0 = 0.28209479177387814f;
                colors_all = torch::clamp(sh0 * kSHC0 + 0.5f, 0.0f, 1.0f).contiguous();
            }
        }
        if (!colors_all.defined()) {
            colors_all = torch::full(
                {N, 3},
                0.5f,
                torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
        }

        const std::array<std::array<int, 2>, 12> edges{{
            {{0, 1}}, {{0, 2}}, {{0, 4}},
            {{1, 3}}, {{1, 5}},
            {{2, 3}}, {{2, 6}},
            {{3, 7}},
            {{4, 5}}, {{4, 6}},
            {{5, 7}},
            {{6, 7}},
        }};

        for (const auto& edge : edges) {
            const int a = edge[0];
            const int b = edge[1];
            torch::Tensor v0 = tsdf8.index({Slice(), a});
            torch::Tensor v1 = tsdf8.index({Slice(), b});
            torch::Tensor edge_valid =
                (evidence_mask &
                 corner_valid.index({Slice(), a}) &
                 corner_valid.index({Slice(), b}))
                    .to(torch::kBool);
            torch::Tensor sign_change =
                (((v0 <= 0.0f) & (v1 >= 0.0f)) |
                 ((v0 >= 0.0f) & (v1 <= 0.0f)))
                    .to(torch::kBool);
            torch::Tensor edge_support =
                (edge_valid & sign_change).to(torch::kBool);
            edge_surface_support =
                (edge_surface_support | edge_support).to(torch::kBool);

            torch::Tensor edge_idx = torch::nonzero(edge_support).view({-1});
            if (!edge_idx.defined() || edge_idx.numel() == 0) {
                continue;
            }

            torch::Tensor v0_sel = v0.index_select(0, edge_idx).to(torch::kFloat32);
            torch::Tensor v1_sel = v1.index_select(0, edge_idx).to(torch::kFloat32);
            torch::Tensor denom = (v0_sel - v1_sel);
            torch::Tensor alpha_cross =
                torch::clamp(
                    torch::where(
                        torch::abs(denom) > 1.0e-8f,
                        v0_sel / denom,
                        torch::full_like(v0_sel, 0.5f)),
                    0.0f,
                    1.0f);
            torch::Tensor alpha = alpha_cross.view({-1, 1});

            torch::Tensor p0 = points8.index({edge_idx, a, Slice()});
            torch::Tensor p1 = points8.index({edge_idx, b, Slice()});
            torch::Tensor p = p0 + alpha * (p1 - p0);
            zero_crossing_points.push_back(p.contiguous());
            zero_crossing_colors.push_back(
                colors_all.index_select(0, edge_idx).contiguous());
        }
    }

    torch::Tensor promotion_support =
        sdf_params_.sdf_evidence_zero_crossing_refresh_
            ? edge_surface_support
            : surface_or_crossing_support;
    torch::Tensor eligible_evidence =
        (evidence_mask & promotion_support).to(torch::kBool);
    torch::Tensor promote_mask = eligible_evidence.to(torch::kBool).contiguous();
    {
        std::vector<sv::MiniCam> tr_cams;
        tr_cams.reserve(scene_->keyframes().size());
        for (const auto& kv : scene_->keyframes()) {
            if (kv.second) {
                tr_cams.push_back(
                    kv.second->toMiniCam(kv.second->image_height_, kv.second->image_width_));
            }
        }
        if (!tr_cams.empty()) {
            torch::Tensor octpath = voxel_model_->octPath();
            torch::Tensor centers = voxel_model_->voxCenter();
            torch::Tensor sizes = voxel_model_->voxSize();
            if (octpath.defined() && centers.defined() && sizes.defined() &&
                octpath.size(0) == N &&
                centers.dim() == 2 && centers.size(0) == N && centers.size(1) == 3 &&
                sizes.numel() == N) {
                octpath = octpath.to(device_type_).contiguous();
                centers = centers.to(device_type_).to(torch::kFloat32).contiguous();
                sizes = sizes.to(device_type_).to(torch::kFloat32).contiguous();
                if (sizes.dim() == 1) {
                    sizes = sizes.view({N, 1});
                } else if (sizes.dim() != 2 || sizes.size(1) != 1) {
                    sizes = sizes.reshape({N, 1});
                }

                at::Tensor rate =
                    sv::markSvrasterMaxSampRateDirect(tr_cams, octpath, centers, sizes);
                if (rate.dim() == 2 && rate.size(1) == 1) {
                    rate = rate.squeeze(1);
                }
                torch::Tensor visible = (rate.to(device_type_).to(torch::kFloat32) > 0.0f)
                                            .to(torch::kBool)
                                            .contiguous()
                                            .view({N});
                promote_mask = (promote_mask & visible).to(torch::kBool);

                if (opt_params_.filter_near_voxels_) {
                    const float near_thresh = 0.2f;
                    at::Tensor is_near =
                        sv::markSvrasterNearDirect(tr_cams, octpath, centers, sizes, near_thresh);
                    if (is_near.dim() == 2 && is_near.size(1) == 1) {
                        is_near = is_near.squeeze(1);
                    }
                    is_near =
                        is_near.to(device_type_).to(torch::kBool).contiguous().view({N});
                    promote_mask = (promote_mask & (~is_near)).to(torch::kBool);
                }
            }
        }
    }
    const int64_t promoted = promote_mask.sum().item<int64_t>();

    if (!zero_crossing_points.empty()) {
        torch::Tensor surface_points =
            torch::cat(zero_crossing_points, 0).to(device_type_).to(torch::kFloat32).contiguous();
        torch::Tensor surface_colors =
            torch::cat(zero_crossing_colors, 0).to(device_type_).to(torch::kFloat32).contiguous();
        std::vector<sv::MiniCam> tr_cams;
        tr_cams.reserve(scene_->keyframes().size());
        for (const auto& kv : scene_->keyframes()) {
            if (kv.second) {
                tr_cams.push_back(
                    kv.second->toMiniCam(kv.second->image_height_, kv.second->image_width_));
            }
        }
        if (surface_points.numel() > 0 && !tr_cams.empty()) {
            std::unique_lock<std::mutex> lock_render(mutex_render_);
            voxel_model_->setNextRealInsertionRerunEntityPath("world/sdf_evidence/created");
            voxel_model_->setNextInsertionVoxelState(sv::VoxelRuntimeState::ActiveRenderable);
            voxel_model_->increasePcd(surface_points, surface_colors, getIteration(), tr_cams);
            voxel_model_->setNextInsertionVoxelState(sv::VoxelRuntimeState::ActiveRenderable);
            voxel_model_->setNextRealInsertionRerunEntityPath("");
            if (rerun_params_.run_whole_run_) {
                rerun_state_.whole_run_live_voxels_dirty_ = true;
            }
        }
    }

    if (promoted > 0) {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        voxel_model_->promoteSdfEvidenceVoxels(promote_mask);
        if (rerun_params_.run_whole_run_) {
            rerun_state_.whole_run_live_voxels_dirty_ = true;
        }
    }
}

void VoxelMapper::refitSvrasterTsdfFromRegisteredKeyframes(const std::string& reason)
{
    if (!sdf_params_.svraster_tsdf_refit_on_topology_change_) {
        return;
    }
    if (!voxel_model_ || !scene_ || sensor_type_ != RGBD ||
        !sdf_params_.use_tsdf_mapping_ || !useSvrasterTsdfBackend()) {
        return;
    }
    if (!voxel_model_->gridPtsKey().defined() ||
        voxel_model_->gridPtsKey().numel() == 0) {
        return;
    }

    torch::NoGradGuard no_grad;
    voxel_model_->resetSvrasterSdfField();

    int fused_kfs = 0;
    for (const auto& kv : scene_->keyframes()) {
        if (!kv.second || kv.second->img_auxiliary_undist_.empty()) {
            continue;
        }
        cv::Mat depth_meters;
        if (!depthMatToMeters(kv.second->img_auxiliary_undist_, depth_meters)) {
            continue;
        }
        integrateKeyframeIntoSvrasterSdf(*kv.second, depth_meters);
        ++fused_kfs;
    }

    int64_t grid_pts = 0;
    float max_weight = 0.0f;
    try {
        if (voxel_model_->gridPtsKey().defined()) {
            grid_pts = voxel_model_->gridPtsKey().size(0);
        }
        torch::Tensor weights = voxel_model_->svrasterSdfWeights();
        if (weights.defined() && weights.numel() > 0) {
            max_weight = weights.max().item<float>();
        }
    } catch (...) {
        grid_pts = 0;
        max_weight = 0.0f;
    }

    if (rerun_params_.rerun_tsdf_unknown_voxels_) {
        rerun_state_.rerun_tsdf_unknown_dirty_ = true;
    }
    if (rerun_params_.run_floaters_) {
        rerun_state_.run_floaters_dirty_ = true;
    }

    // std::cout << "[SVRASTER TSDF REFIT] reason=" << reason
    //           << " keyframes=" << fused_kfs
    //           << " grid_pts=" << grid_pts
    //           << " max_weight=" << max_weight
    //           << "\n";
}

VoxelMapper::TsdfCornerSample VoxelMapper::sampleTsdfAtVoxelCornersWorld(
    const torch::Tensor& centers_world,
    const torch::Tensor& sizes_world)
{
    using torch::indexing::Slice;

    TORCH_CHECK(centers_world.defined() && centers_world.dim() == 2 && centers_world.size(1) == 3,
                "sampleTsdfAtVoxelCornersWorld expects centers_world [N,3]");
    TORCH_CHECK(sizes_world.defined(),
                "sampleTsdfAtVoxelCornersWorld expects sizes_world defined");

    const int64_t N = centers_world.size(0);
    TsdfCornerSample out;

    const auto dev = centers_world.device();

    // Normalize sizes to [N,1] on same device
    torch::Tensor sizes = sizes_world;
    if (sizes.dim() == 1) {
        TORCH_CHECK(sizes.size(0) == N, "sizes_world [N] mismatch with centers_world");
        sizes = sizes.view({N, 1});
    } else if (sizes.dim() == 2) {
        TORCH_CHECK(sizes.size(0) == N, "sizes_world [N,1] mismatch with centers_world");
        TORCH_CHECK(sizes.size(1) == 1, "sizes_world must have shape [N,1] if 2D");
    } else {
        TORCH_CHECK(false, "sizes_world must be [N] or [N,1]");
    }
    sizes = sizes.to(dev).contiguous();

    // Default outputs (unknown)
    out.tsdf    = torch::full({N, 8}, std::numeric_limits<float>::quiet_NaN(),
                              torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    out.weight  = torch::zeros({N, 8}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    out.success = torch::zeros({N, 8}, torch::TensorOptions().dtype(torch::kBool).device(dev));
    out.points_world = torch::Tensor();

    if (N == 0) return out;
    if (!sdf_mapper_) return out;
    nvblox::TsdfLayer& tsdf_layer = sdf_mapper_->tsdf_layer();
    if (tsdf_layer.size() == 0) return out;

    // Corner offsets (8,3) with all sign combinations.
    // Construct once, keep on CUDA (or whatever device centers are on).
    static torch::Tensor offsets_cache;
    static torch::Device cached_dev = torch::kCPU;

    if (!offsets_cache.defined() || cached_dev != dev) {
        // CPU literal then move
        offsets_cache = torch::tensor(
            {{-1.f, -1.f, -1.f},
             {-1.f, -1.f,  1.f},
             {-1.f,  1.f, -1.f},
             {-1.f,  1.f,  1.f},
             { 1.f, -1.f, -1.f},
             { 1.f, -1.f,  1.f},
             { 1.f,  1.f, -1.f},
             { 1.f,  1.f,  1.f}},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).to(dev).contiguous();
        cached_dev = dev;
    }
    const torch::Tensor offsets = offsets_cache; // [8,3] on dev

    // Build corners: [N,8,3] = centers[:,None,:] + 0.5*sizes[:,None,:]*offsets[None,:,:]
    torch::Tensor half = 0.5f * sizes;                 // [N,1]
    torch::Tensor corners = centers_world.contiguous().view({N, 1, 3})
                          + half.view({N, 1, 1}) * offsets.view({1, 8, 3}); // [N,8,3]
    torch::Tensor corners_flat = corners.view({N * 8, 3}).contiguous();     // [N*8,3]
    out.points_world = corners.contiguous();

    // Use your existing point sampler (returns [N*8])
    TsdfSample s = sampleTsdfAtPointsWorld(corners_flat);

    // Reshape back to [N,8]
    TORCH_CHECK(s.tsdf.defined() && s.tsdf.numel() == N * 8,
                "sampleTsdfAtPointsWorld returned unexpected tsdf size");
    TORCH_CHECK(s.weight.defined() && s.weight.numel() == N * 8,
                "sampleTsdfAtPointsWorld returned unexpected weight size");
    TORCH_CHECK(s.success.defined() && s.success.numel() == N * 8,
                "sampleTsdfAtPointsWorld returned unexpected success size");

    out.tsdf    = s.tsdf.view({N, 8}).contiguous();
    out.weight  = s.weight.view({N, 8}).contiguous();
    out.success = s.success.view({N, 8}).contiguous();

    // ------------------------------------------------------------
    // One-time sanity check: do corners match center ± size/2 ?
    // ------------------------------------------------------------
    {
        static bool printed_once = false;
        if (!printed_once) {
            printed_once = true;

            const int64_t i0 = 0;
            if (corners.defined() && corners.dim() == 3 && corners.size(0) > i0) {
                // Bring to CPU for printing
                auto c0_cpu = centers_world.index({i0}).to(torch::kCPU).contiguous();      // [3]
                auto s0_cpu = sizes_world.index({i0}).view({1}).to(torch::kCPU).contiguous(); // [1]
                auto crn_cpu = corners.index({i0}).to(torch::kCPU).contiguous();          // [8,3]

                const float cx = c0_cpu[0].item<float>();
                const float cy = c0_cpu[1].item<float>();
                const float cz = c0_cpu[2].item<float>();
                const float s  = s0_cpu[0].item<float>();
                const float h  = 0.5f * s;

                // Expected bounds
                const float ex_min_x = cx - h, ex_max_x = cx + h;
                const float ex_min_y = cy - h, ex_max_y = cy + h;
                const float ex_min_z = cz - h, ex_max_z = cz + h;

                // Observed bounds from corners
                auto min_xyz = std::get<0>(crn_cpu.min(/*dim=*/0, /*keepdim=*/false)); // [3]
                auto max_xyz = std::get<0>(crn_cpu.max(/*dim=*/0, /*keepdim=*/false)); // [3]

                const float ob_min_x = min_xyz[0].item<float>();
                const float ob_min_y = min_xyz[1].item<float>();
                const float ob_min_z = min_xyz[2].item<float>();

                const float ob_max_x = max_xyz[0].item<float>();
                const float ob_max_y = max_xyz[1].item<float>();
                const float ob_max_z = max_xyz[2].item<float>();

                auto absf = [](float v){ return v < 0.f ? -v : v; };
                const float eps = 1e-5f;

                std::cout << "[TSDF CORNER SANITY] i0=" << i0 << "\n";
                std::cout << "  center = [" << cx << ", " << cy << ", " << cz << "]\n";
                std::cout << "  size   = " << s << "  half=" << h << "\n";

                std::cout << "  expected min = [" << ex_min_x << ", " << ex_min_y << ", " << ex_min_z << "]\n";
                std::cout << "  expected max = [" << ex_max_x << ", " << ex_max_y << ", " << ex_max_z << "]\n";

                std::cout << "  observed min = [" << ob_min_x << ", " << ob_min_y << ", " << ob_min_z << "]\n";
                std::cout << "  observed max = [" << ob_max_x << ", " << ob_max_y << ", " << ob_max_z << "]\n";

                const bool ok_x = (absf(ob_min_x - ex_min_x) < eps) && (absf(ob_max_x - ex_max_x) < eps);
                const bool ok_y = (absf(ob_min_y - ex_min_y) < eps) && (absf(ob_max_y - ex_max_y) < eps);
                const bool ok_z = (absf(ob_min_z - ex_min_z) < eps) && (absf(ob_max_z - ex_max_z) < eps);

                std::cout << "  axis check: x=" << (ok_x ? "OK" : "FAIL")
                        << " y=" << (ok_y ? "OK" : "FAIL")
                        << " z=" << (ok_z ? "OK" : "FAIL")
                        << " (eps=" << eps << ")\n";

                std::cout << "  corners[8,3] =\n" << crn_cpu << "\n";
            } else {
                std::cout << "[TSDF CORNER SANITY] corners tensor not ready or empty.\n";
            }
        }
    }

    return out;
}

void VoxelMapper::runFinalSpecialPrune()
{
    if (!opt_params_.final_special_prune_enable_) {
        std::cout << "[FINAL/special_prune] disabled\n";
        return;
    }

    const int before_final_special = voxel_model_->numVoxels();
    if (before_final_special <= 0) {
        return;
    }

    auto centers = voxel_model_->voxCenter();
    auto sizes = voxel_model_->voxSize();
    if (sizes.defined() && sizes.dim() == 2 && sizes.size(1) == 1) {
        sizes = sizes.squeeze(1);
    } else if (sizes.defined() && sizes.dim() != 1) {
        sizes = sizes.reshape({-1});
    }
    auto prune_mask_final_special = torch::zeros(
        {before_final_special},
        torch::TensorOptions().dtype(torch::kBool).device(centers.device()));

    int64_t n_far_final_special = 0;
    int64_t n_near_final_special = 0;
    int64_t n_near_geom_final_special = 0;
    int64_t n_unstable_final_special = 0;
    bool far_valid_final_special = false;
    bool near_valid_final_special = false;
    bool unstable_valid_final_special = false;
    torch::Tensor unstable_mask_final_special;

    bool use_far_final_special = opt_params_.prune_far_voxels_;
    if (use_far_final_special) {
        const bool refreshed_dense_core =
            voxel_model_->refreshDenseCoreBBFromCurrentVoxels();
        if (!voxel_model_->hasDenseCoreBB()) {
            use_far_final_special = false;
            std::cout << "[FINAL/special_prune] dense-core refresh unavailable; "
                      << "skipping far pruning.\n";
        } else if (!refreshed_dense_core) {
            std::cout << "[FINAL/special_prune] dense-core refresh failed; "
                      << "using last available bbox.\n";
        }
    }

    if (use_far_final_special && voxel_model_->hasDenseCoreBB()) {
        auto bb_min = voxel_model_->denseCoreBBMin();
        auto bb_max = voxel_model_->denseCoreBBMax();
        if (centers.defined() && bb_min.defined() && bb_max.defined() &&
            centers.dim() == 2 && centers.size(1) == 3 &&
            centers.size(0) == before_final_special &&
            bb_min.numel() == 3 && bb_max.numel() == 3) {
            auto centers_f32 = centers.to(torch::kFloat32).contiguous();
            bb_min = bb_min.to(centers_f32.device()).to(torch::kFloat32).contiguous().view({1, 3});
            bb_max = bb_max.to(centers_f32.device()).to(torch::kFloat32).contiguous().view({1, 3});
            auto in_dense_core =
                (centers_f32 >= bb_min).all(/*dim=*/1) &
                (centers_f32 <= bb_max).all(/*dim=*/1);
            auto far_mask = (~in_dense_core.to(torch::kBool)).contiguous();
            n_far_final_special = far_mask.sum().item<int64_t>();
            prune_mask_final_special = (prune_mask_final_special | far_mask).to(torch::kBool);
            far_valid_final_special = true;
        }
    }

    {
        std::vector<sv::MiniCam> tr_cams;
        tr_cams.reserve(scene_->keyframes().size());
        for (const auto& kv : scene_->keyframes()) {
            tr_cams.push_back(kv.second->toMiniCam(
                kv.second->image_height_, kv.second->image_width_));
        }

        if (tr_cams.empty()) {
            n_near_final_special = 0;
            near_valid_final_special = true;
        } else {
            try {
                auto octpath = voxel_model_->octPath().contiguous();
                auto vox_center = voxel_model_->voxCenter().contiguous();
                auto vox_size = voxel_model_->voxSize().contiguous();

                TORCH_CHECK(octpath.size(0) == before_final_special,
                            "octpath length mismatch at final special prune");
                TORCH_CHECK(vox_center.size(0) == before_final_special,
                            "vox_center length mismatch at final special prune");
                TORCH_CHECK(vox_center.size(1) == 3,
                            "vox_center must be [N,3] at final special prune");
                if (vox_size.dim() == 1) {
                    vox_size = vox_size.view({before_final_special, 1});
                } else if (vox_size.dim() == 2) {
                    TORCH_CHECK(vox_size.size(0) == before_final_special,
                                "vox_size length mismatch at final special prune");
                } else {
                    TORCH_CHECK(false, "vox_size must be [N] or [N,1] at final special prune");
                }

                const float near_thresh = 0.2f;
                at::Tensor is_near =
                    sv::markSvrasterNearDirect(tr_cams, octpath, vox_center, vox_size, near_thresh);
                if (is_near.dim() == 2 && is_near.size(1) == 1) {
                    is_near = is_near.squeeze(1);
                }
                is_near = is_near.to(torch::kBool);
                at::Tensor is_near_geom = torch::zeros(
                    {before_final_special},
                    torch::TensorOptions().dtype(torch::kBool).device(is_near.device()));
                if (opt_params_.prune_near_voxels_geometric_) {
                    auto vox_center_f32 =
                        vox_center.to(is_near.device()).to(torch::kFloat32).contiguous();
                    auto vox_size_1d =
                        vox_size.to(is_near.device()).to(torch::kFloat32).contiguous();
                    if (vox_size_1d.dim() == 2 && vox_size_1d.size(1) == 1) {
                        vox_size_1d = vox_size_1d.squeeze(1);
                    } else if (vox_size_1d.dim() != 1) {
                        vox_size_1d = vox_size_1d.reshape({-1});
                    }
                    TORCH_CHECK(vox_size_1d.numel() == before_final_special,
                                "vox_size_1d.numel() != N in final special prune geometric near filter");

                    auto near_radius =
                        (torch::full_like(vox_size_1d, near_thresh) + 0.5f * vox_size_1d)
                            .contiguous();
                    auto near_radius_sq = (near_radius * near_radius).contiguous();

                    for (const auto& c : tr_cams) {
                        auto cam_pos =
                            c.position.to(is_near.device()).to(torch::kFloat32).view({1, 3});
                        auto d2 = (vox_center_f32 - cam_pos).pow(2).sum(/*dim=*/1);
                        is_near_geom =
                            (is_near_geom | (d2 <= near_radius_sq)).to(torch::kBool);
                    }
                }
                n_near_final_special = is_near.sum().item<int64_t>();
                n_near_geom_final_special = is_near_geom.sum().item<int64_t>();
                auto near_union = (is_near | is_near_geom).to(torch::kBool);
                prune_mask_final_special =
                    (prune_mask_final_special |
                     near_union.to(prune_mask_final_special.device()).to(torch::kBool)).to(torch::kBool);
                near_valid_final_special = true;
            } catch (const std::exception& e) {
                std::cerr << "[FINAL/special_prune] failed to compute near voxels: "
                          << e.what() << "\n";
            }
        }
    }

    if (opt_params_.prune_recent_unstable_) {
        try {
            torch::Tensor unstable_mask = torch::zeros(
                {before_final_special},
                torch::TensorOptions().dtype(torch::kBool).device(prune_mask_final_special.device()));

            if (pending_recent_unstable_prune_mask_.defined() &&
                pending_recent_unstable_prune_mask_.numel() == before_final_special) {
                torch::Tensor pending_unstable =
                    pending_recent_unstable_prune_mask_
                        .to(prune_mask_final_special.device())
                        .to(torch::kBool)
                        .contiguous()
                        .view({before_final_special});
                unstable_mask =
                    (unstable_mask | pending_unstable).to(torch::kBool);
            }

            unstable_mask_final_special = unstable_mask;
            n_unstable_final_special = unstable_mask.sum().item<int64_t>();
            prune_mask_final_special =
                (prune_mask_final_special | unstable_mask).to(torch::kBool);
            unstable_valid_final_special = true;
        } catch (const std::exception& e) {
            std::cerr << "[FINAL/special_prune] failed to compute unstable voxels: "
                      << e.what() << "\n";
        }
    }

    const int64_t n_selected_final_special =
        prune_mask_final_special.to(torch::kBool).sum().item<int64_t>();
    if (n_selected_final_special > 0) {
        torch::Tensor final_idx =
            prune_mask_final_special.to(torch::kBool).nonzero().squeeze(1);
        if (final_idx.defined() && final_idx.numel() > 0 &&
            centers.defined() && centers.dim() == 2 &&
            centers.size(0) == before_final_special &&
            sizes.defined() && sizes.numel() == before_final_special) {
            torch::Tensor final_centers =
                centers.index_select(0, final_idx.to(centers.device()).to(torch::kLong)).contiguous();
            torch::Tensor sizes_for_log = sizes;
            if (sizes_for_log.dim() == 2 && sizes_for_log.size(1) == 1) {
                sizes_for_log = sizes_for_log.squeeze(1);
            } else if (sizes_for_log.dim() != 1) {
                sizes_for_log = sizes_for_log.reshape({before_final_special});
            }
            torch::Tensor final_sizes =
                sizes_for_log.index_select(0, final_idx.to(sizes_for_log.device()).to(torch::kLong)).contiguous();
            torch::Tensor final_special_mask = torch::ones(
                {final_idx.numel()},
                torch::TensorOptions().dtype(torch::kBool).device(final_centers.device()));
            torch::Tensor final_unstable_mask;
            if (unstable_mask_final_special.defined() &&
                unstable_mask_final_special.numel() == before_final_special) {
                final_unstable_mask =
                    unstable_mask_final_special
                        .to(final_idx.device())
                        .to(torch::kBool)
                        .contiguous()
                        .view({before_final_special})
                        .index_select(0, final_idx.to(torch::kLong))
                        .contiguous();
            }
            appendWholeRunPrunedVoxels(
                getIteration(),
                final_centers,
                final_sizes,
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                final_special_mask);
            appendUnstablePrunedVoxels(
                getIteration(),
                final_centers,
                final_sizes,
                final_unstable_mask);
        }
    }

    if (n_selected_final_special > 0) {
        voxel_model_->pruning(prune_mask_final_special);
        if (rerun_params_.run_whole_run_) {
            torch::Tensor final_view_count =
                rerun_state_.whole_run_live_local_view_counts_cache_;
            if (final_view_count.defined() &&
                final_view_count.numel() != voxel_model_->numVoxels()) {
                final_view_count = torch::Tensor();
            }
            logWholeRunLiveVoxelsToRerun(
                getIteration(),
                voxel_model_->voxCenter(),
                voxel_model_->voxSize(),
                torch::Tensor(),
                final_view_count);
        }
        if (rerun_params_.rerun_tsdf_unknown_voxels_) {
            rerun_state_.rerun_tsdf_unknown_dirty_ = true;
        }
        if (rerun_params_.run_floaters_) {
            rerun_state_.run_floaters_dirty_ = true;
        }
    }

    const int after_final_special = voxel_model_->numVoxels();
    std::cout << "[FINAL/special_prune] before=" << before_final_special
              << " selected=" << n_selected_final_special
              << " removed=" << (before_final_special - after_final_special)
              << " far=" << (far_valid_final_special ? std::to_string(n_far_final_special) : std::string("N/A"))
              << " near=" << (near_valid_final_special ? std::to_string(n_near_final_special) : std::string("N/A"))
              << " near_geom=" << (near_valid_final_special ? std::to_string(n_near_geom_final_special) : std::string("N/A"))
              << " unstable=" << (unstable_valid_final_special ? std::to_string(n_unstable_final_special) : std::string("N/A"))
              << "\n";
}

void VoxelMapper::recordTsdfPruneAblation(
    const torch::Tensor& tsdf_prune_mask,
    const std::string& tag)
{
    if (!voxel_model_ || !tsdf_prune_mask.defined() || tsdf_prune_mask.numel() <= 0) {
        return;
    }

    const int64_t N = tsdf_prune_mask.numel();
    const torch::Device device = tsdf_prune_mask.device();
    torch::Tensor tsdf_mask =
        normalizeBoolMaskOrZeros(tsdf_prune_mask, N, device);

    torch::Tensor rgbd_points_mask =
        normalizeBoolMaskOrZeros(voxel_model_->orbVoxelMask(), N, device);
    torch::Tensor inactive_geo_mask =
        normalizeBoolMaskOrZeros(voxel_model_->inactiveGeoVoxelMask(), N, device);
    torch::Tensor rgbd_fill_mask =
        normalizeBoolMaskOrZeros(voxel_model_->rgbdFillRenderHolesVoxelMask(), N, device);

    // Source masks should be exclusive, but keep this robust for older runs.
    rgbd_points_mask =
        (rgbd_points_mask & (~inactive_geo_mask) & (~rgbd_fill_mask)).to(torch::kBool);

    const int64_t rgbd_points_live = rgbd_points_mask.sum().item<int64_t>();
    const int64_t inactive_geo_live = inactive_geo_mask.sum().item<int64_t>();
    const int64_t rgbd_fill_live = rgbd_fill_mask.sum().item<int64_t>();

    const int64_t rgbd_points_pruned =
        (tsdf_mask & rgbd_points_mask).sum().item<int64_t>();
    const int64_t inactive_geo_pruned =
        (tsdf_mask & inactive_geo_mask).sum().item<int64_t>();
    const int64_t rgbd_fill_pruned =
        (tsdf_mask & rgbd_fill_mask).sum().item<int64_t>();

    sdf_state_.tsdf_ablation_rgbd_points_live_last_ = rgbd_points_live;
    sdf_state_.tsdf_ablation_inactive_geo_live_last_ = inactive_geo_live;
    sdf_state_.tsdf_ablation_rgbd_fill_live_last_ = rgbd_fill_live;
    sdf_state_.tsdf_ablation_rgbd_points_pruned_by_sdf_ += rgbd_points_pruned;
    sdf_state_.tsdf_ablation_inactive_geo_pruned_by_sdf_ += inactive_geo_pruned;
    sdf_state_.tsdf_ablation_rgbd_fill_pruned_by_sdf_ += rgbd_fill_pruned;
    ++sdf_state_.tsdf_ablation_sdf_prune_passes_;

    // Verbose per-pass TSDF ablation logging is intentionally disabled.
    // The counters above are still accumulated and can be printed from
    // printTsdfPruneAblationSummary() if needed.
}

void VoxelMapper::printTsdfPruneAblationSummary(const std::string& tag) const
{
    const int64_t total_pruned_by_sdf =
        sdf_state_.tsdf_ablation_rgbd_points_pruned_by_sdf_ +
        sdf_state_.tsdf_ablation_inactive_geo_pruned_by_sdf_ +
        sdf_state_.tsdf_ablation_rgbd_fill_pruned_by_sdf_;

    std::cout << "[TSDF ABLATION SUMMARY] tag=" << tag
              << " sdf_prune_passes=" << sdf_state_.tsdf_ablation_sdf_prune_passes_
              << " total_sdf_pruned=" << total_pruned_by_sdf
              << "\n"
              << "  rgbd_points: inserted_roots=" << sdf_state_.tsdf_ablation_rgbd_points_created_
              << " lineage_created=" << sdf_state_.tsdf_ablation_rgbd_points_lineage_created_
              << " pruned_by_sdf=" << sdf_state_.tsdf_ablation_rgbd_points_pruned_by_sdf_
              << "\n"
              << "  inactive_geo_densify: inserted_roots=" << sdf_state_.tsdf_ablation_inactive_geo_created_
              << " lineage_created=" << sdf_state_.tsdf_ablation_inactive_geo_lineage_created_
              << " pruned_by_sdf=" << sdf_state_.tsdf_ablation_inactive_geo_pruned_by_sdf_
              << "\n"
              << "  rgbd_fill_render_holes: inserted_roots=" << sdf_state_.tsdf_ablation_rgbd_fill_created_
              << " lineage_created=" << sdf_state_.tsdf_ablation_rgbd_fill_lineage_created_
              << " pruned_by_sdf=" << sdf_state_.tsdf_ablation_rgbd_fill_pruned_by_sdf_
              << std::endl;
}
