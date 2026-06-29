#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <unistd.h> 

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <c10/cuda/CUDACachingAllocator.h>

#include "ORB-SLAM3/include/System.h"
#include "include_voxel/voxel_mapper.h"
#include "viewer/imgui_viewer.h"

static void loadReplicaImages(
    const std::filesystem::path& image_dir,
    const std::string& prefix,
    std::vector<std::string>& filenames)
{
    filenames.clear();
    if (!std::filesystem::exists(image_dir))
    {
        std::cerr << "[loadReplicaImages] directory does not exist: "
                  << image_dir << std::endl;
        return;
    }

    for (const auto& entry : std::filesystem::directory_iterator(image_dir))
    {
        if (!entry.is_regular_file())
            continue;

        const std::string name = entry.path().filename().string();
        if (name.rfind(prefix, 0) == 0)
            filenames.push_back(entry.path().string());
    }
    std::sort(filenames.begin(), filenames.end());
}

static void saveTrackingTime(const std::vector<float>& times, const std::string& path)
{
    std::ofstream out(path);
    for (float t : times)
        out << std::fixed << std::setprecision(4) << t << "\n";
}

struct GpuMemoryStats
{
    float reserved_mb = 0.0f;
    float allocated_mb = 0.0f;
};

static GpuMemoryStats getGpuPeakMemoryStats()
{
    GpuMemoryStats stats;
    if (!torch::cuda::is_available())
        return stats;

    namespace c10Alloc = c10::cuda::CUDACachingAllocator;
    c10Alloc::DeviceStats mem_stats = c10Alloc::getDeviceStats(0);

    stats.reserved_mb =
        mem_stats.reserved_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)].peak /
        (1024.0f * 1024.0f);
    stats.allocated_mb =
        mem_stats.allocated_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)].peak /
        (1024.0f * 1024.0f);
    return stats;
}

static void saveGpuPeakMemoryUsage(const std::filesystem::path& path)
{
    const GpuMemoryStats stats = getGpuPeakMemoryStats();
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    out << "Peak reserved (MB): " << stats.reserved_mb << "\n";
    out << "Peak allocated (MB): " << stats.allocated_mb << "\n";
}

static double fileSizeMb(const std::filesystem::path& path)
{
    if (!std::filesystem::exists(path))
        return 0.0;
    return static_cast<double>(std::filesystem::file_size(path)) /
           (1024.0 * 1024.0);
}

static void saveRuntimeMetrics(
    const std::filesystem::path& path,
    int frames,
    int keyframes,
    int voxels,
    int iterations,
    double total_seconds,
    const std::filesystem::path& map_path,
    const GpuMemoryStats& gpu_stats)
{
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());

    std::ofstream out(path);
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"frames\": " << frames << ",\n";
    out << "  \"keyframes\": " << keyframes << ",\n";
    out << "  \"voxels\": " << voxels << ",\n";
    out << "  \"iterations\": " << iterations << ",\n";
    out << "  \"total_seconds\": " << total_seconds << ",\n";
    out << "  \"fps_hz\": "
        << (total_seconds > 0.0 ? frames / total_seconds : 0.0) << ",\n";
    out << "  \"map_path\": \"" << map_path.string() << "\",\n";
    out << "  \"map_size_mb\": " << fileSizeMb(map_path) << ",\n";
    out << "  \"gpu_memory_allocated_mb\": " << gpu_stats.allocated_mb << ",\n";
    out << "  \"gpu_memory_reserved_mb\": " << gpu_stats.reserved_mb << "\n";
    out << "}\n";
}

int main(int argc, char** argv)
{
    if (argc != 6 && argc != 7)
    {
        std::cerr << "\nUsage: " << argv[0]
                  << " path_to_vocabulary"
                  << " path_to_ORB_SLAM3_settings"
                  << " path_to_voxel_mapper_settings"
                  << " path_to_sequence"
                  << " path_to_output_directory/"
                  << " (optional)no_viewer"
                  << std::endl;
        return 1;
    }

    const bool use_viewer = argc == 7 ? (std::string(argv[6]) != "no_viewer") : true;

    std::string output_directory = std::string(argv[5]);
    if (output_directory.back() != '/')
        output_directory += "/";
    std::filesystem::path output_dir(output_directory);
    const auto total_start = std::chrono::steady_clock::now();

    const std::filesystem::path seq_path(argv[4]);
    const std::filesystem::path image_dir = seq_path / "results";

    std::vector<std::string> rgb_files;
    std::vector<std::string> depth_files;
    loadReplicaImages(image_dir, "frame", rgb_files);
    loadReplicaImages(image_dir, "depth", depth_files);

    if (rgb_files.empty())
    {
        std::cerr << "[replica_rgbd_voxel] no RGB frames found in: "
                  << image_dir << std::endl;
        return 1;
    }
    if (rgb_files.size() != depth_files.size())
    {
        std::cerr << "[replica_rgbd_voxel] RGB/depth count mismatch: rgb="
                  << rgb_files.size() << " depth=" << depth_files.size()
                  << std::endl;
        return 1;
    }

    torch::DeviceType device_type =
        torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "[INFO] Using device: "
              << (device_type == torch::kCUDA ? "CUDA" : "CPU")
              << std::endl;

    std::shared_ptr<ORB_SLAM3::System> slam =
        std::make_shared<ORB_SLAM3::System>(
            argv[1],
            argv[2],
            ORB_SLAM3::System::RGBD,
            use_viewer);
    const float image_scale = slam->GetImageScale();

    std::shared_ptr<VoxelMapper> voxel_mapper =
        std::make_shared<VoxelMapper>(
            slam,
            std::filesystem::path(argv[3]),
            output_dir,
            0,
            device_type);

    std::thread training_thread(&VoxelMapper::run, voxel_mapper.get());

    std::thread viewer_thread;
    std::shared_ptr<ImGuiViewer> viewer;
    if (use_viewer)
    {
        viewer = std::make_shared<ImGuiViewer>(slam, voxel_mapper);
        viewer_thread = std::thread(&ImGuiViewer::run, viewer.get());
    }

    const int num_images = static_cast<int>(rgb_files.size());
    std::vector<float> tracking_times(num_images);
    int processed_frames = 0;

    std::cout << "\n-------\n";
    std::cout << "Start processing Replica RGB-D sequence ..." << std::endl;
    std::cout << "Images in the sequence: " << num_images << "\n\n";

    for (int i = 0; i < num_images; ++i)
    {
        if (slam->isShutDown())
            break;

        cv::Mat rgb = cv::imread(rgb_files[i], cv::IMREAD_UNCHANGED);
        if (rgb.empty())
        {
            std::cerr << "[replica_rgbd_voxel] failed to load RGB: "
                      << rgb_files[i] << std::endl;
            return 1;
        }
        cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);

        cv::Mat depth = cv::imread(depth_files[i], cv::IMREAD_UNCHANGED);
        if (depth.empty())
        {
            std::cerr << "[replica_rgbd_voxel] failed to load depth: "
                      << depth_files[i] << std::endl;
            return 1;
        }

        if (image_scale != 1.0f)
        {
            const int width = static_cast<int>(rgb.cols * image_scale);
            const int height = static_cast<int>(rgb.rows * image_scale);
            cv::resize(rgb, rgb, cv::Size(width, height));
            cv::resize(depth, depth, cv::Size(width, height));
        }

        const double timestamp = static_cast<double>(i);
        const auto start = std::chrono::steady_clock::now();
        slam->TrackRGBD(
            rgb,
            depth,
            timestamp,
            std::vector<ORB_SLAM3::IMU::Point>(),
            rgb_files[i]);
        const auto end = std::chrono::steady_clock::now();

        const float track_time =
            std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
        tracking_times[i] = track_time;
        ++processed_frames;

        // // Photo-SLAM's Replica RGB-D runner feeds frames as fast as TrackRGBD returns. Uncomment this block, the replica_fps constant, the unistd include, and the
        // // timestamp=i/replica_fps line above to reproduce the earlier fixed-30Hz pacing.
        // constexpr double replica_fps = 30.0;  // Uncomment when enabling fixed-FPS pacing.
        // constexpr double frame_dt = 1.0 / replica_fps;
        // if (track_time < frame_dt) {
        //     usleep(static_cast<useconds_t>((frame_dt - track_time) * 1e6));
        // }
    }

    slam->Shutdown();
    training_thread.join();

    const int final_iteration = voxel_mapper->getIteration();
    const std::filesystem::path shutdown_dir =
        output_dir / (std::to_string(final_iteration) + "_shutdown");
    const std::filesystem::path final_map_path =
        shutdown_dir / "ply" / "voxel_model" /
        ("iteration_" + std::to_string(final_iteration)) / "voxel_model.ply";
    const int keyframes =
        voxel_mapper->scene_ ? static_cast<int>(voxel_mapper->scene_->keyframes().size()) : 0;
    const int voxels =
        voxel_mapper->voxel_model_ ? voxel_mapper->voxel_model_->numVoxels() : 0;

    saveTrackingTime(tracking_times, (output_dir / "TrackingTime.txt").string());

    slam->SaveTrajectoryTUM((output_dir / "CameraTrajectory_TUM.txt").string());
    slam->SaveKeyFrameTrajectoryTUM((output_dir / "KeyFrameTrajectory_TUM.txt").string());
    slam->SaveTrajectoryEuRoC((output_dir / "CameraTrajectory_EuRoC.txt").string());
    slam->SaveKeyFrameTrajectoryEuRoC((output_dir / "KeyFrameTrajectory_EuRoC.txt").string());
    slam->SaveTrajectoryKITTI((output_dir / "CameraTrajectory_KITTI.txt").string());

    const auto total_end = std::chrono::steady_clock::now();
    const double total_wall_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            total_end - total_start).count();
    const double total_seconds = std::max(0.0, total_wall_seconds);
    const GpuMemoryStats gpu_stats = getGpuPeakMemoryStats();

    saveGpuPeakMemoryUsage(output_dir / "GpuPeakUsageMB.txt");
    saveGpuPeakMemoryUsage(shutdown_dir / "GpuPeakUsageMB.txt");
    saveRuntimeMetrics(
        shutdown_dir / "runtime_metrics.json",
        processed_frames,
        keyframes,
        voxels,
        final_iteration,
        total_seconds,
        final_map_path,
        gpu_stats);

    if (use_viewer)
        viewer_thread.join();

    return 0;
}
