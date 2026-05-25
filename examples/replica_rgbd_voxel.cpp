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

#include <pybind11/embed.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "ORB-SLAM3/include/System.h"
#include "include_voxel/voxel_mapper.h"
#include "viewer/imgui_viewer.h"

namespace py = pybind11;

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

static void saveGpuPeakMemoryUsage(const std::filesystem::path& path)
{
    namespace c10Alloc = c10::cuda::CUDACachingAllocator;
    c10Alloc::DeviceStats mem_stats = c10Alloc::getDeviceStats(0);

    const float max_reserved_mb =
        mem_stats.reserved_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)].peak /
        (1024.0f * 1024.0f);
    const float max_alloc_mb =
        mem_stats.allocated_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)].peak /
        (1024.0f * 1024.0f);

    std::ofstream out(path);
    out << "Peak reserved (MB): " << max_reserved_mb << "\n";
    out << "Peak allocated (MB): " << max_alloc_mb << "\n";
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

    py::initialize_interpreter();
    {
        py::gil_scoped_acquire gil;
        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("insert")(0, "../scripts_voxel");
        py::module_::import("scripts_voxel.python_svraster_bridge.renderer_wrapper");
        py::module_::import("torch.cuda");
    }
    py::gil_scoped_release release;

    std::string output_directory = std::string(argv[5]);
    if (output_directory.back() != '/')
        output_directory += "/";
    std::filesystem::path output_dir(output_directory);

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
    constexpr double replica_fps = 30.0;

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

        const double timestamp = static_cast<double>(i) / replica_fps;
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

        constexpr double frame_dt = 1.0 / replica_fps;
        if (track_time < frame_dt)
            usleep(static_cast<useconds_t>((frame_dt - track_time) * 1e6));
    }

    slam->Shutdown();
    training_thread.join();

    saveGpuPeakMemoryUsage(output_dir / "GpuPeakUsageMB.txt");
    saveTrackingTime(tracking_times, (output_dir / "TrackingTime.txt").string());

    slam->SaveTrajectoryTUM((output_dir / "CameraTrajectory_TUM.txt").string());
    slam->SaveKeyFrameTrajectoryTUM((output_dir / "KeyFrameTrajectory_TUM.txt").string());
    slam->SaveTrajectoryEuRoC((output_dir / "CameraTrajectory_EuRoC.txt").string());
    slam->SaveKeyFrameTrajectoryEuRoC((output_dir / "KeyFrameTrajectory_EuRoC.txt").string());
    slam->SaveTrajectoryKITTI((output_dir / "CameraTrajectory_KITTI.txt").string());

    if (use_viewer)
        viewer_thread.join();

    return 0;
}
