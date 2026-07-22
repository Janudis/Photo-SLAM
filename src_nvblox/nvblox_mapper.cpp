#include "include_nvblox/nvblox_mapper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <nvblox/map/blox.h>
#include <nvblox/io/mesh_io.h>
#include <nvblox/mapper/mapper_params.h>

namespace photoslam {
namespace {

template <typename T>
T readValue(const cv::FileStorage& file, const char* key, const T& fallback)
{
    const cv::FileNode node = file[key];
    return node.empty() ? fallback : static_cast<T>(node);
}

bool readBool(const cv::FileStorage& file, const char* key, bool fallback)
{
    return readValue<int>(file, key, fallback ? 1 : 0) != 0;
}

bool finiteTransform(const Sophus::SE3f& transform)
{
    const Eigen::Matrix4f matrix = transform.matrix();
    return matrix.allFinite() &&
           std::abs(transform.rotationMatrix().determinant() - 1.0f) < 1e-3f;
}

}  // namespace

NvbloxMapper::NvbloxMapper(
    const std::filesystem::path& mapper_config_path,
    const std::filesystem::path& orb_config_path,
    const std::filesystem::path& output_directory)
    : output_directory_(output_directory),
      mesh_path_(output_directory_ / "nvblox_color_mesh.ply"),
      map_path_(output_directory_ / "nvblox_map.nvblx"),
      mapping_time_path_(output_directory_ / "NvbloxMappingTime.txt")
{
    std::filesystem::create_directories(output_directory_);
    std::ofstream(mapping_time_path_, std::ios::trunc).close();
    readMapperConfig(mapper_config_path);
    readCameraCalibration(orb_config_path);
    sampleGpuMemory();
    gpu_baseline_used_bytes_ = gpu_peak_used_bytes_;
    initializeMapper();
    sampleGpuMemory();
}

void NvbloxMapper::readMapperConfig(const std::filesystem::path& path)
{
    cv::FileStorage file(path.string(), cv::FileStorage::READ);
    if (!file.isOpened()) {
        throw std::runtime_error("Cannot open nvblox mapper config: " + path.string());
    }

    config_.voxel_size_m =
        readValue<float>(file, "Nvblox.voxel_size_m", config_.voxel_size_m);
    config_.max_integration_distance_m = readValue<float>(
        file,
        "Nvblox.max_integration_distance_m",
        config_.max_integration_distance_m);
    config_.truncation_distance_vox = readValue<float>(
        file,
        "Nvblox.truncation_distance_vox",
        config_.truncation_distance_vox);
    config_.max_weight =
        readValue<float>(file, "Nvblox.max_weight", config_.max_weight);
    config_.mesh_min_weight =
        readValue<float>(file, "Nvblox.mesh_min_weight", config_.mesh_min_weight);
    config_.raycast_subsampling_factor = readValue<int>(
        file,
        "Nvblox.raycast_subsampling_factor",
        config_.raycast_subsampling_factor);
    config_.frame_subsampling =
        readValue<int>(file, "Nvblox.frame_subsampling", config_.frame_subsampling);
    config_.color_frame_subsampling = readValue<int>(
        file,
        "Nvblox.color_frame_subsampling",
        config_.color_frame_subsampling);
    config_.mesh_update_interval = readValue<int>(
        file,
        "Nvblox.mesh_update_interval",
        config_.mesh_update_interval);
    config_.weld_mesh_vertices = readBool(
        file,
        "Nvblox.weld_mesh_vertices",
        config_.weld_mesh_vertices);
    config_.undistort_input =
        readBool(file, "Nvblox.undistort_input", config_.undistort_input);
    config_.integrate_color =
        readBool(file, "Nvblox.integrate_color", config_.integrate_color);
    config_.save_map = readBool(file, "Nvblox.save_map", config_.save_map);

    if (!(config_.voxel_size_m > 0.0f) ||
        !(config_.max_integration_distance_m > 0.0f) ||
        !(config_.truncation_distance_vox > 0.0f) ||
        !(config_.max_weight > 0.0f)) {
        throw std::runtime_error("Nvblox metric and TSDF parameters must be positive.");
    }
    config_.raycast_subsampling_factor =
        std::max(1, config_.raycast_subsampling_factor);
    config_.frame_subsampling = std::max(1, config_.frame_subsampling);
    config_.color_frame_subsampling =
        std::max(1, config_.color_frame_subsampling);
    config_.mesh_update_interval = std::max(0, config_.mesh_update_interval);
}

void NvbloxMapper::readCameraCalibration(const std::filesystem::path& path)
{
    cv::FileStorage file(path.string(), cv::FileStorage::READ);
    if (!file.isOpened()) {
        throw std::runtime_error("Cannot open ORB-SLAM camera config: " + path.string());
    }

    calibration_.fx = readValue<float>(file, "Camera1.fx", 0.0f);
    calibration_.fy = readValue<float>(file, "Camera1.fy", 0.0f);
    calibration_.cx = readValue<float>(file, "Camera1.cx", 0.0f);
    calibration_.cy = readValue<float>(file, "Camera1.cy", 0.0f);
    calibration_.width = readValue<int>(file, "Camera.width", 0);
    calibration_.height = readValue<int>(file, "Camera.height", 0);
    const float depth_factor =
        readValue<float>(file, "RGBD.DepthMapFactor", 1.0f);
    calibration_.depth_scale_to_m =
        std::abs(depth_factor) > 1e-6f ? 1.0f / depth_factor : 1.0f;

    calibration_.distortion = (cv::Mat_<float>(1, 5)
        << readValue<float>(file, "Camera1.k1", 0.0f),
           readValue<float>(file, "Camera1.k2", 0.0f),
           readValue<float>(file, "Camera1.p1", 0.0f),
           readValue<float>(file, "Camera1.p2", 0.0f),
           readValue<float>(file, "Camera1.k3", 0.0f));

    if (!(calibration_.fx > 0.0f) || !(calibration_.fy > 0.0f) ||
        calibration_.width <= 0 || calibration_.height <= 0) {
        throw std::runtime_error("ORB-SLAM camera config has invalid pinhole calibration.");
    }
}

void NvbloxMapper::initializeMapper()
{
    nvblox::BlockMemoryPoolParams pool_params;
    pool_params.memory_type = nvblox::MemoryType::kDevice;
    cuda_stream_ = std::make_shared<nvblox::CudaStreamOwning>();
    mapper_ = std::make_shared<nvblox::Mapper>(
        config_.voxel_size_m,
        pool_params,
        nvblox::ProjectiveLayerType::kTsdf,
        cuda_stream_);

    nvblox::MapperParams params;
    params.projective_integrator_params
        .projective_integrator_max_integration_distance_m =
        config_.max_integration_distance_m;
    params.projective_integrator_params
        .projective_integrator_truncation_distance_vox =
        config_.truncation_distance_vox;
    params.projective_integrator_params.projective_integrator_max_weight =
        config_.max_weight;
    params.view_calculator_params.raycast_subsampling_factor =
        config_.raycast_subsampling_factor;
    params.mesh_integrator_params.mesh_integrator_min_weight =
        config_.mesh_min_weight;
    params.mesh_integrator_params.mesh_integrator_weld_vertices =
        config_.weld_mesh_vertices;
    mapper_->setMapperParams(params);

    std::cout << "[NVBLOX] voxel_size=" << config_.voxel_size_m
              << "m truncation=" << config_.truncation_distance_vox
              << "vox max_distance=" << config_.max_integration_distance_m
              << "m frame_subsampling=" << config_.frame_subsampling
              << " raycast_subsampling=" << config_.raycast_subsampling_factor
              << std::endl;
}

void NvbloxMapper::updateUndistortionMaps(
    int width,
    int height,
    const cv::Mat& K)
{
    if (width == undistort_width_ && height == undistort_height_ &&
        !undistort_map_x_.empty()) {
        return;
    }
    cv::initUndistortRectifyMap(
        K,
        calibration_.distortion,
        cv::Mat::eye(3, 3, CV_32F),
        K,
        cv::Size(width, height),
        CV_32FC1,
        undistort_map_x_,
        undistort_map_y_);
    undistort_width_ = width;
    undistort_height_ = height;
}

bool NvbloxMapper::prepareImages(
    const cv::Mat& rgb,
    const cv::Mat& depth,
    cv::Mat& rgb_prepared,
    cv::Mat& depth_meters,
    nvblox::Camera& camera)
{
    if (rgb.empty() || depth.empty() || rgb.size() != depth.size()) {
        return false;
    }
    if (rgb.type() != CV_8UC3) {
        std::cerr << "[NVBLOX] expected an 8-bit three-channel RGB image."
                  << std::endl;
        return false;
    }

    if (depth.type() == CV_32FC1) {
        depth_meters = depth.clone();
    } else if (depth.type() == CV_16UC1) {
        depth.convertTo(depth_meters, CV_32FC1, calibration_.depth_scale_to_m);
    } else if (depth.type() == CV_64FC1) {
        depth.convertTo(depth_meters, CV_32FC1);
    } else {
        std::cerr << "[NVBLOX] unsupported depth type: " << depth.type()
                  << std::endl;
        return false;
    }

    const float scale_x =
        static_cast<float>(rgb.cols) / static_cast<float>(calibration_.width);
    const float scale_y =
        static_cast<float>(rgb.rows) / static_cast<float>(calibration_.height);
    const float fx = calibration_.fx * scale_x;
    const float fy = calibration_.fy * scale_y;
    const float cx = calibration_.cx * scale_x;
    const float cy = calibration_.cy * scale_y;
    const cv::Mat K = (cv::Mat_<float>(3, 3)
        << fx, 0.0f, cx,
           0.0f, fy, cy,
           0.0f, 0.0f, 1.0f);

    if (config_.undistort_input && cv::norm(calibration_.distortion) > 1e-8) {
        updateUndistortionMaps(rgb.cols, rgb.rows, K);
        cv::remap(
            rgb,
            rgb_prepared,
            undistort_map_x_,
            undistort_map_y_,
            cv::INTER_LINEAR,
            cv::BORDER_CONSTANT);
        cv::Mat depth_undistorted;
        cv::remap(
            depth_meters,
            depth_undistorted,
            undistort_map_x_,
            undistort_map_y_,
            cv::INTER_NEAREST,
            cv::BORDER_CONSTANT,
            cv::Scalar(0.0f));
        depth_meters = std::move(depth_undistorted);
    } else {
        rgb_prepared = rgb;
    }

    for (int row = 0; row < depth_meters.rows; ++row) {
        float* values = depth_meters.ptr<float>(row);
        for (int col = 0; col < depth_meters.cols; ++col) {
            if (!std::isfinite(values[col]) || values[col] <= 0.0f ||
                values[col] > config_.max_integration_distance_m) {
                values[col] = 0.0f;
            }
        }
    }

    camera = nvblox::Camera(fx, fy, cx, cy, rgb.cols, rgb.rows);
    return true;
}

void NvbloxMapper::uploadDepth(const cv::Mat& depth_meters)
{
    depth_host_image_.resizeAsync(
        depth_meters.rows,
        depth_meters.cols,
        *cuda_stream_);
    cuda_stream_->synchronize();
    for (int row = 0; row < depth_meters.rows; ++row) {
        std::memcpy(
            depth_host_image_.dataPtr() +
                static_cast<std::size_t>(row) * depth_meters.cols,
            depth_meters.ptr<float>(row),
            static_cast<std::size_t>(depth_meters.cols) * sizeof(float));
    }
    depth_image_.copyFromAsync(depth_host_image_, *cuda_stream_);
    cuda_stream_->synchronize();
}

void NvbloxMapper::uploadColor(const cv::Mat& rgb)
{
    static_assert(sizeof(nvblox::Color) == 3, "Unexpected nvblox Color layout");
    color_host_image_.resizeAsync(rgb.rows, rgb.cols, *cuda_stream_);
    cuda_stream_->synchronize();
    for (int row = 0; row < rgb.rows; ++row) {
        std::memcpy(
            color_host_image_.dataPtr() +
                static_cast<std::size_t>(row) * rgb.cols,
            rgb.ptr<cv::Vec3b>(row),
            static_cast<std::size_t>(rgb.cols) * sizeof(nvblox::Color));
    }
    color_image_.copyFromAsync(color_host_image_, *cuda_stream_);
    cuda_stream_->synchronize();
}

nvblox::Transform NvbloxMapper::toNvbloxTransform(const Sophus::SE3f& Tcw)
{
    const Sophus::SE3f Twc = Tcw.inverse();
    nvblox::Transform T_L_C = nvblox::Transform::Identity();
    T_L_C.linear() = Twc.rotationMatrix();
    T_L_C.translation() = Twc.translation();
    return T_L_C;
}

bool NvbloxMapper::integrateFrame(
    const cv::Mat& rgb,
    const cv::Mat& depth,
    const Sophus::SE3f& Tcw,
    int frame_index)
{
    if (frame_index % config_.frame_subsampling != 0) {
        ++stats_.skipped_by_subsampling;
        return false;
    }
    if (!finiteTransform(Tcw)) {
        return false;
    }

    const auto start = std::chrono::steady_clock::now();
    cv::Mat rgb_prepared;
    cv::Mat depth_meters;
    nvblox::Camera camera;
    if (!prepareImages(rgb, depth, rgb_prepared, depth_meters, camera)) {
        return false;
    }

    const nvblox::Transform T_L_C = toNvbloxTransform(Tcw);
    uploadDepth(depth_meters);
    mapper_->integrateDepth(depth_image_, T_L_C, camera);
    ++stats_.integrated_depth_frames;

    if (config_.integrate_color &&
        frame_index % config_.color_frame_subsampling == 0) {
        uploadColor(rgb_prepared);
        mapper_->integrateColor(color_image_, T_L_C, camera);
        ++stats_.integrated_color_frames;
    }

    if (config_.mesh_update_interval > 0 &&
        stats_.integrated_depth_frames % config_.mesh_update_interval == 0) {
        mapper_->updateColorMesh();
    }
    cuda_stream_->synchronize();

    const auto end = std::chrono::steady_clock::now();
    const double seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    stats_.integration_seconds += seconds;
    {
        std::ofstream out(mapping_time_path_, std::ios::app);
        out << std::fixed << std::setprecision(6) << seconds << "\n";
    }
    sampleGpuMemory();
    return true;
}

bool NvbloxMapper::saveOutputs()
{
    if (!mapper_) {
        return false;
    }

    const auto start = std::chrono::steady_clock::now();
    mapper_->updateColorMesh(nvblox::UpdateFullLayer::kYes);
    cuda_stream_->synchronize();

    bool success = nvblox::io::outputColorMeshLayerToPly(
        mapper_->color_mesh_layer(), mesh_path_.c_str());
    if (config_.save_map) {
        success = mapper_->saveLayerCake(map_path_.c_str()) && success;
    }

    stats_.tsdf_blocks = static_cast<std::size_t>(mapper_->tsdf_layer().numBlocks());
    stats_.allocated_tsdf_voxels =
        stats_.tsdf_blocks * static_cast<std::size_t>(nvblox::TsdfBlock::kNumVoxels);
    const auto end = std::chrono::steady_clock::now();
    stats_.finalization_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    sampleGpuMemory();

    std::cout << "[NVBLOX] integrated_frames=" << stats_.integrated_depth_frames
              << " tsdf_blocks=" << stats_.tsdf_blocks
              << " allocated_voxels=" << stats_.allocated_tsdf_voxels
              << " mesh=" << mesh_path_ << std::endl;
    return success;
}

void NvbloxMapper::sampleGpuMemory()
{
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        cudaGetLastError();
        return;
    }
    gpu_total_bytes_ = total_bytes;
    gpu_peak_used_bytes_ = std::max(gpu_peak_used_bytes_, total_bytes - free_bytes);
    const std::size_t delta = gpu_peak_used_bytes_ > gpu_baseline_used_bytes_
        ? gpu_peak_used_bytes_ - gpu_baseline_used_bytes_
        : 0;
    stats_.gpu_peak_delta_mb =
        static_cast<double>(delta) / (1024.0 * 1024.0);
    stats_.gpu_peak_used_mb =
        static_cast<double>(gpu_peak_used_bytes_) / (1024.0 * 1024.0);
}

double NvbloxMapper::fileSizeMb(const std::filesystem::path& path)
{
    if (!std::filesystem::exists(path)) {
        return 0.0;
    }
    return static_cast<double>(std::filesystem::file_size(path)) /
           (1024.0 * 1024.0);
}

void NvbloxMapper::saveRuntimeMetrics(const NvbloxRunSummary& summary) const
{
    std::ofstream out(output_directory_ / "runtime_metrics.json");
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"frames\": " << summary.input_frames << ",\n";
    out << "  \"tracked_frames\": " << summary.tracked_frames << ",\n";
    out << "  \"integrated_frames\": " << stats_.integrated_depth_frames << ",\n";
    out << "  \"keyframes\": " << summary.keyframes << ",\n";
    out << "  \"tsdf_blocks\": " << stats_.tsdf_blocks << ",\n";
    out << "  \"voxels\": " << stats_.allocated_tsdf_voxels << ",\n";
    out << "  \"tracking_seconds\": " << summary.tracking_seconds << ",\n";
    out << "  \"mapping_seconds\": " << stats_.integration_seconds << ",\n";
    out << "  \"finalization_seconds\": " << stats_.finalization_seconds << ",\n";
    out << "  \"total_seconds\": " << summary.total_seconds << ",\n";
    out << "  \"fps_hz\": "
        << (summary.total_seconds > 0.0
                ? static_cast<double>(summary.input_frames) / summary.total_seconds
                : 0.0)
        << ",\n";
    out << "  \"mapping_fps_hz\": "
        << (stats_.integration_seconds > 0.0
                ? static_cast<double>(stats_.integrated_depth_frames) /
                      stats_.integration_seconds
                : 0.0)
        << ",\n";
    out << "  \"map_path\": \"" << map_path_.string() << "\",\n";
    out << "  \"map_size_mb\": " << fileSizeMb(map_path_) << ",\n";
    out << "  \"mesh_path\": \"" << mesh_path_.string() << "\",\n";
    out << "  \"mesh_size_mb\": " << fileSizeMb(mesh_path_) << ",\n";
    out << "  \"gpu_memory_allocated_mb\": " << stats_.gpu_peak_delta_mb << ",\n";
    out << "  \"gpu_memory_reserved_mb\": " << stats_.gpu_peak_delta_mb << ",\n";
    out << "  \"gpu_memory_peak_used_mb\": " << stats_.gpu_peak_used_mb << "\n";
    out << "}\n";

    std::ofstream gpu_out(output_directory_ / "GpuPeakUsageMB.txt");
    gpu_out << std::fixed << std::setprecision(2);
    gpu_out << "Peak process delta (MB): " << stats_.gpu_peak_delta_mb << "\n";
    gpu_out << "Peak device used (MB): " << stats_.gpu_peak_used_mb << "\n";
}

}  // namespace photoslam
