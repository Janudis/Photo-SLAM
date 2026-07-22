#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>

#include <opencv2/core.hpp>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

#include <nvblox/core/cuda_stream.h>
#include <nvblox/mapper/mapper.h>
#include <nvblox/sensors/camera.h>
#include <nvblox/sensors/image.h>

namespace photoslam {

struct NvbloxMapperConfig {
    float voxel_size_m = 0.05f;
    float max_integration_distance_m = 7.0f;
    float truncation_distance_vox = 4.0f;
    float max_weight = 5.0f;
    float mesh_min_weight = 1e-4f;
    int raycast_subsampling_factor = 4;
    int frame_subsampling = 1;
    int color_frame_subsampling = 1;
    int mesh_update_interval = 0;
    bool weld_mesh_vertices = true;
    bool undistort_input = true;
    bool integrate_color = true;
    bool save_map = true;
};

struct NvbloxRunSummary {
    int input_frames = 0;
    int tracked_frames = 0;
    int keyframes = 0;
    double tracking_seconds = 0.0;
    double total_seconds = 0.0;
};

struct NvbloxMappingStats {
    int integrated_depth_frames = 0;
    int integrated_color_frames = 0;
    int skipped_by_subsampling = 0;
    double integration_seconds = 0.0;
    double finalization_seconds = 0.0;
    std::size_t tsdf_blocks = 0;
    std::size_t allocated_tsdf_voxels = 0;
    double gpu_peak_delta_mb = 0.0;
    double gpu_peak_used_mb = 0.0;
};

class NvbloxMapper {
public:
    NvbloxMapper(
        const std::filesystem::path& mapper_config_path,
        const std::filesystem::path& orb_config_path,
        const std::filesystem::path& output_directory);

    bool integrateFrame(
        const cv::Mat& rgb,
        const cv::Mat& depth,
        const Sophus::SE3f& Tcw,
        int frame_index);

    bool saveOutputs();
    void saveRuntimeMetrics(const NvbloxRunSummary& summary) const;

    const NvbloxMapperConfig& config() const { return config_; }
    const NvbloxMappingStats& stats() const { return stats_; }
    const std::filesystem::path& meshPath() const { return mesh_path_; }
    const std::filesystem::path& mapPath() const { return map_path_; }

private:
    struct CameraCalibration {
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
        int width = 0;
        int height = 0;
        float depth_scale_to_m = 1.0f;
        cv::Mat distortion;
    };

    void readMapperConfig(const std::filesystem::path& path);
    void readCameraCalibration(const std::filesystem::path& path);
    void initializeMapper();

    bool prepareImages(
        const cv::Mat& rgb,
        const cv::Mat& depth,
        cv::Mat& rgb_prepared,
        cv::Mat& depth_meters,
        nvblox::Camera& camera);
    void updateUndistortionMaps(int width, int height, const cv::Mat& K);
    void uploadDepth(const cv::Mat& depth_meters);
    void uploadColor(const cv::Mat& rgb);
    void sampleGpuMemory();

    static nvblox::Transform toNvbloxTransform(const Sophus::SE3f& Tcw);
    static double fileSizeMb(const std::filesystem::path& path);

    NvbloxMapperConfig config_;
    CameraCalibration calibration_;
    NvbloxMappingStats stats_;

    std::filesystem::path output_directory_;
    std::filesystem::path mesh_path_;
    std::filesystem::path map_path_;
    std::filesystem::path mapping_time_path_;

    std::shared_ptr<nvblox::CudaStream> cuda_stream_;
    std::shared_ptr<nvblox::Mapper> mapper_;
    nvblox::DepthImage depth_host_image_{nvblox::MemoryType::kHost};
    nvblox::ColorImage color_host_image_{nvblox::MemoryType::kHost};
    nvblox::DepthImage depth_image_{nvblox::MemoryType::kDevice};
    nvblox::ColorImage color_image_{nvblox::MemoryType::kDevice};

    cv::Mat undistort_map_x_;
    cv::Mat undistort_map_y_;
    int undistort_width_ = 0;
    int undistort_height_ = 0;

    std::size_t gpu_total_bytes_ = 0;
    std::size_t gpu_baseline_used_bytes_ = 0;
    std::size_t gpu_peak_used_bytes_ = 0;
};

}  // namespace photoslam
