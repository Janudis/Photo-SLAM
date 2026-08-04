#include <torch/torch.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <c10/cuda/CUDACachingAllocator.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ORB-SLAM3/include/System.h"
#include "include_voxel/mapper_depth_registry.h"
#include "include_voxel/voxel_mapper.h"
#include "viewer/imgui_viewer.h"

namespace {

struct FrameRecord {
    double timestamp = 0.0;
    std::filesystem::path relative_image_path;
    std::filesystem::path relative_depth_path;
    std::filesystem::path relative_mapper_depth_path;
};

std::vector<FrameRecord> loadAssociation(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open Waymo RGB-D association: " + path.string());
    }

    std::vector<FrameRecord> frames;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') {
            continue;
        }
        std::istringstream stream(line);
        FrameRecord frame;
        double depth_timestamp = 0.0;
        std::string image_path;
        std::string depth_path;
        std::string mapper_depth_path;
        if (!(stream >> frame.timestamp >> image_path >> depth_timestamp >> depth_path)) {
            throw std::runtime_error("invalid Waymo RGB-D association line: " + line);
        }
        if (std::abs(frame.timestamp - depth_timestamp) > 1.0e-6) {
            throw std::runtime_error("Waymo RGB and LiDAR depth timestamps are not synchronized");
        }
        frame.relative_image_path = image_path;
        frame.relative_depth_path = depth_path;
        if (stream >> mapper_depth_path) {
            frame.relative_mapper_depth_path = mapper_depth_path;
        } else {
            frame.relative_mapper_depth_path = depth_path;
        }
        if (!frames.empty() && frame.timestamp <= frames.back().timestamp) {
            throw std::runtime_error("Waymo RGB-D timestamps must be strictly increasing");
        }
        frames.push_back(std::move(frame));
    }
    return frames;
}

void saveTrackingTimes(
    const std::vector<float>& times,
    const std::filesystem::path& output_path)
{
    std::ofstream output(output_path);
    for (const float time : times) {
        output << std::fixed << std::setprecision(6) << time << '\n';
    }
}

void saveGpuPeakMemoryUsage(const std::filesystem::path& output_path)
{
    namespace allocator = c10::cuda::CUDACachingAllocator;
    const allocator::DeviceStats stats = allocator::getDeviceStats(0);
    const auto aggregate = static_cast<int>(allocator::StatType::AGGREGATE);
    const double reserved_mb =
        stats.reserved_bytes[aggregate].peak / (1024.0 * 1024.0);
    const double allocated_mb =
        stats.allocated_bytes[aggregate].peak / (1024.0 * 1024.0);

    std::ofstream output(output_path);
    output << "Peak reserved (MB): " << reserved_mb << '\n';
    output << "Peak allocated (MB): " << allocated_mb << '\n';
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 6 && argc != 7) {
        std::cerr
            << "Usage: " << argv[0]
            << " vocabulary orb_settings voxel_settings prepared_sequence output_dir"
            << " [no_viewer]\n";
        return 1;
    }

    const std::filesystem::path vocabulary_path(argv[1]);
    const std::filesystem::path orb_settings_path(argv[2]);
    const std::filesystem::path voxel_settings_path(argv[3]);
    const std::filesystem::path sequence_root(argv[4]);
    const std::filesystem::path output_root(argv[5]);
    const bool use_viewer = argc != 7 || std::string(argv[6]) != "no_viewer";

    std::filesystem::create_directories(output_root);
    const std::vector<FrameRecord> frames =
        loadAssociation(sequence_root / "rgbd.txt");
    if (frames.empty()) {
        std::cerr << "No Waymo RGB-D frames found in " << sequence_root << '\n';
        return 1;
    }

    for (const FrameRecord& frame : frames) {
        sv::registerMapperDepthImage(
            (sequence_root / frame.relative_image_path).string(),
            (sequence_root / frame.relative_mapper_depth_path).string());
    }

    if (!torch::cuda::is_available()) {
        std::cerr << "[ERROR] waymo_rgbd_voxel requires a CUDA-capable GPU.\n";
        return 1;
    }
    const torch::DeviceType device_type = torch::kCUDA;
    std::cout << "[INFO] Using device: CUDA\n";

    auto slam = std::make_shared<ORB_SLAM3::System>(
        vocabulary_path.string(),
        orb_settings_path.string(),
        ORB_SLAM3::System::RGBD,
        use_viewer);
    auto mapper = std::make_shared<VoxelMapper>(
        slam,
        voxel_settings_path,
        output_root,
        0,
        device_type);
    mapper->setRuntimeFrameCount(static_cast<int>(frames.size()));

    std::thread mapping_thread(&VoxelMapper::run, mapper.get());
    std::shared_ptr<ImGuiViewer> viewer;
    std::thread viewer_thread;
    if (use_viewer) {
        viewer = std::make_shared<ImGuiViewer>(slam, mapper);
        viewer_thread = std::thread(&ImGuiViewer::run, viewer.get());
    }

    std::vector<float> tracking_times;
    tracking_times.reserve(frames.size());
    std::cout << "\n-------\nStart processing Waymo RGB-D sequence ...\n"
              << "Images in the sequence: " << frames.size() << "\n\n";

    for (std::size_t index = 0; index < frames.size(); ++index) {
        if (slam->isShutDown()) {
            break;
        }
        const std::filesystem::path image_path =
            sequence_root / frames[index].relative_image_path;
        const std::filesystem::path depth_path =
            sequence_root / frames[index].relative_depth_path;
        cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        cv::Mat depth = cv::imread(depth_path.string(), cv::IMREAD_UNCHANGED);
        if (image.empty() || depth.empty()) {
            std::cerr << "Failed to load Waymo RGB-D frame: "
                      << image_path << " / " << depth_path << '\n';
            slam->Shutdown();
            mapping_thread.join();
            if (use_viewer) {
                viewer_thread.join();
            }
            return 1;
        }
        if (depth.channels() != 1) {
            std::cerr << "Waymo LiDAR depth must be single-channel: " << depth_path << '\n';
            slam->Shutdown();
            mapping_thread.join();
            if (use_viewer) {
                viewer_thread.join();
            }
            return 1;
        }
        if (depth.type() != CV_32FC1) {
            depth.convertTo(depth, CV_32FC1);
        }
        if (image.size() != depth.size()) {
            cv::resize(image, image, depth.size(), 0.0, 0.0, cv::INTER_AREA);
        }
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        const auto start = std::chrono::steady_clock::now();
        {
            auto tracking_profile =
                mapper->profileLaptopModule("orb_tracking");
            slam->TrackRGBD(
                image,
                depth,
                frames[index].timestamp,
                std::vector<ORB_SLAM3::IMU::Point>(),
                image_path.string());
        }
        const auto end = std::chrono::steady_clock::now();
        const float tracking_time =
            std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
        tracking_times.push_back(tracking_time);

        double frame_period = 0.0;
        if (index + 1 < frames.size()) {
            frame_period = frames[index + 1].timestamp - frames[index].timestamp;
        } else if (index > 0) {
            frame_period = frames[index].timestamp - frames[index - 1].timestamp;
        }
        if (tracking_time < frame_period) {
            usleep(static_cast<useconds_t>((frame_period - tracking_time) * 1.0e6));
        }
    }

    slam->Shutdown();
    mapping_thread.join();
    if (use_viewer) {
        viewer_thread.join();
    }

    saveTrackingTimes(tracking_times, output_root / "TrackingTime.txt");
    saveGpuPeakMemoryUsage(output_root / "GpuPeakUsageMB.txt");
    slam->SaveTrajectoryTUM((output_root / "CameraTrajectory_TUM.txt").string());
    slam->SaveKeyFrameTrajectoryTUM(
        (output_root / "KeyFrameTrajectory_TUM.txt").string());
    slam->SaveTrajectoryEuRoC(
        (output_root / "CameraTrajectory_EuRoC.txt").string());
    slam->SaveKeyFrameTrajectoryEuRoC(
        (output_root / "KeyFrameTrajectory_EuRoC.txt").string());
    sv::clearMapperDepthImages();
    return 0;
}
