#include "include_nvblox/nvblox_runner.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime_api.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <glog/logging.h>

#include "ORB-SLAM3/include/System.h"
#include "ORB-SLAM3/include/Tracking.h"
#include "include_nvblox/nvblox_mapper.h"

namespace photoslam {
namespace {

struct RgbdSequence {
    std::vector<std::string> rgb_paths;
    std::vector<std::string> depth_paths;
    std::vector<double> timestamps;
    bool pace_to_timestamps = false;
};

bool loadTumSequence(
    const std::filesystem::path& sequence_directory,
    const std::filesystem::path& association_path,
    RgbdSequence& sequence)
{
    std::ifstream association(association_path);
    if (!association.is_open()) {
        std::cerr << "[NVBLOX] cannot open association file: "
                  << association_path << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(association, line)) {
        if (line.empty() || line.front() == '#') {
            continue;
        }
        std::stringstream stream(line);
        double rgb_timestamp = 0.0;
        double depth_timestamp = 0.0;
        std::string rgb_relative_path;
        std::string depth_relative_path;
        if (!(stream >> rgb_timestamp >> rgb_relative_path >>
              depth_timestamp >> depth_relative_path)) {
            continue;
        }
        sequence.rgb_paths.push_back(
            (sequence_directory / rgb_relative_path).string());
        sequence.depth_paths.push_back(
            (sequence_directory / depth_relative_path).string());
        sequence.timestamps.push_back(rgb_timestamp);
    }
    sequence.pace_to_timestamps = true;
    return !sequence.rgb_paths.empty() &&
           sequence.rgb_paths.size() == sequence.depth_paths.size();
}

void appendFilesWithPrefix(
    const std::filesystem::path& directory,
    const std::string& prefix,
    std::vector<std::string>& paths)
{
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string filename = entry.path().filename().string();
        if (filename.rfind(prefix, 0) == 0) {
            paths.push_back(entry.path().string());
        }
    }
    std::sort(paths.begin(), paths.end());
}

bool loadReplicaSequence(
    const std::filesystem::path& sequence_directory,
    RgbdSequence& sequence)
{
    const std::filesystem::path image_directory = sequence_directory / "results";
    if (!std::filesystem::is_directory(image_directory)) {
        std::cerr << "[NVBLOX] Replica image directory does not exist: "
                  << image_directory << std::endl;
        return false;
    }

    appendFilesWithPrefix(image_directory, "frame", sequence.rgb_paths);
    appendFilesWithPrefix(image_directory, "depth", sequence.depth_paths);
    if (sequence.rgb_paths.empty() ||
        sequence.rgb_paths.size() != sequence.depth_paths.size()) {
        return false;
    }
    sequence.timestamps.resize(sequence.rgb_paths.size());
    for (std::size_t index = 0; index < sequence.timestamps.size(); ++index) {
        sequence.timestamps[index] = static_cast<double>(index);
    }
    sequence.pace_to_timestamps = false;
    return true;
}

void saveTimes(const std::filesystem::path& path, const std::vector<double>& times)
{
    std::ofstream out(path);
    for (const double time : times) {
        out << std::fixed << std::setprecision(6) << time << "\n";
    }
}

bool trackingIsValid(int state)
{
    return state == ORB_SLAM3::Tracking::OK ||
           state == ORB_SLAM3::Tracking::OK_KLT;
}

void printUsage(NvbloxDataset dataset, const char* executable)
{
    std::cerr << "Usage: " << executable
              << " path_to_vocabulary"
              << " path_to_ORB_SLAM3_settings"
              << " path_to_nvblox_settings"
              << " path_to_sequence";
    if (dataset == NvbloxDataset::kTum) {
        std::cerr << " path_to_association";
    }
    std::cerr << " path_to_output_directory/" << std::endl;
}

}  // namespace

int runRgbdNvblox(NvbloxDataset dataset, int argc, char** argv)
{
    const int expected_arguments =
        dataset == NvbloxDataset::kTum ? 7 : 6;
    if (argc != expected_arguments) {
        printUsage(dataset, argv[0]);
        return 1;
    }

    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();

    int cuda_device_count = 0;
    if (cudaGetDeviceCount(&cuda_device_count) != cudaSuccess ||
        cuda_device_count <= 0) {
        std::cerr << "[NVBLOX] a CUDA-capable GPU is required." << std::endl;
        return 1;
    }

    const std::filesystem::path vocabulary_path(argv[1]);
    const std::filesystem::path orb_config_path(argv[2]);
    const std::filesystem::path nvblox_config_path(argv[3]);
    const std::filesystem::path sequence_directory(argv[4]);
    const std::filesystem::path output_directory(
        dataset == NvbloxDataset::kTum ? argv[6] : argv[5]);
    std::filesystem::create_directories(output_directory);

    RgbdSequence sequence;
    const bool loaded = dataset == NvbloxDataset::kTum
        ? loadTumSequence(sequence_directory, argv[5], sequence)
        : loadReplicaSequence(sequence_directory, sequence);
    if (!loaded) {
        std::cerr << "[NVBLOX] failed to load RGB-D sequence." << std::endl;
        return 1;
    }

    const auto total_start = std::chrono::steady_clock::now();
    auto slam = std::make_shared<ORB_SLAM3::System>(
        vocabulary_path.string(),
        orb_config_path.string(),
        ORB_SLAM3::System::RGBD);
    const float image_scale = slam->GetImageScale();

    NvbloxMapper mapper(
        nvblox_config_path,
        orb_config_path,
        output_directory);

    const int frame_count = static_cast<int>(sequence.rgb_paths.size());
    std::vector<double> tracking_times;
    tracking_times.reserve(frame_count);
    int processed_frames = 0;
    int tracked_frames = 0;
    double tracking_seconds = 0.0;

    std::cout << "\n-------\n"
              << "Start processing "
              << (dataset == NvbloxDataset::kTum ? "TUM" : "Replica")
              << " RGB-D sequence with ORB-SLAM + nvblox ...\n"
              << "Images in the sequence: " << frame_count << "\n\n";

    for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
        if (slam->isShutDown()) {
            break;
        }
        const auto frame_start = std::chrono::steady_clock::now();

        cv::Mat rgb = cv::imread(
            sequence.rgb_paths[frame_index], cv::IMREAD_UNCHANGED);
        cv::Mat depth = cv::imread(
            sequence.depth_paths[frame_index], cv::IMREAD_UNCHANGED);
        if (rgb.empty() || depth.empty()) {
            std::cerr << "[NVBLOX] failed to load frame " << frame_index
                      << std::endl;
            break;
        }
        cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);

        if (image_scale != 1.0f) {
            const cv::Size scaled_size(
                static_cast<int>(rgb.cols * image_scale),
                static_cast<int>(rgb.rows * image_scale));
            cv::resize(rgb, rgb, scaled_size, 0.0, 0.0, cv::INTER_LINEAR);
            cv::resize(depth, depth, scaled_size, 0.0, 0.0, cv::INTER_NEAREST);
        }

        const auto tracking_start = std::chrono::steady_clock::now();
        const Sophus::SE3f Tcw = slam->TrackRGBD(
            rgb,
            depth,
            sequence.timestamps[frame_index],
            std::vector<ORB_SLAM3::IMU::Point>(),
            sequence.rgb_paths[frame_index]);
        const auto tracking_end = std::chrono::steady_clock::now();
        const double tracking_time =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                tracking_end - tracking_start).count();
        tracking_times.push_back(tracking_time);
        tracking_seconds += tracking_time;
        ++processed_frames;

        if (trackingIsValid(slam->GetTrackingState())) {
            ++tracked_frames;
            mapper.integrateFrame(rgb, depth, Tcw, frame_index);
        }

        // The Gaussian/voxel mappers normally consume these BA notifications.
        // nvblox uses the current online pose directly, so retaining them only
        // grows an unbounded queue in this standalone experiment.
        slam->getAtlas()->clearMappingOperation();

        if (sequence.pace_to_timestamps && frame_index + 1 < frame_count) {
            const double target_period =
                sequence.timestamps[frame_index + 1] -
                sequence.timestamps[frame_index];
            const double elapsed =
                std::chrono::duration_cast<std::chrono::duration<double>>(
                    std::chrono::steady_clock::now() - frame_start).count();
            if (target_period > elapsed) {
                std::this_thread::sleep_for(
                    std::chrono::duration<double>(target_period - elapsed));
            }
        }
    }

    slam->Shutdown();
    const bool outputs_saved = mapper.saveOutputs();

    slam->SaveTrajectoryTUM(
        (output_directory / "CameraTrajectory_TUM.txt").string());
    slam->SaveKeyFrameTrajectoryTUM(
        (output_directory / "KeyFrameTrajectory_TUM.txt").string());
    slam->SaveTrajectoryEuRoC(
        (output_directory / "CameraTrajectory_EuRoC.txt").string());
    slam->SaveKeyFrameTrajectoryEuRoC(
        (output_directory / "KeyFrameTrajectory_EuRoC.txt").string());
    slam->SaveTrajectoryKITTI(
        (output_directory / "CameraTrajectory_KITTI.txt").string());
    saveTimes(output_directory / "TrackingTime.txt", tracking_times);

    const double total_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - total_start).count();
    NvbloxRunSummary summary;
    summary.input_frames = processed_frames;
    summary.tracked_frames = tracked_frames;
    summary.keyframes = static_cast<int>(slam->GetNumKeyframes());
    summary.tracking_seconds = tracking_seconds;
    summary.total_seconds = total_seconds;
    mapper.saveRuntimeMetrics(summary);

    return outputs_saved ? 0 : 1;
}

}  // namespace photoslam
