#include <torch/torch.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <c10/cuda/CUDACachingAllocator.h>

#include "ORB-SLAM3/include/System.h"
#include "include_voxel/voxel_mapper.h"
#include "include_voxel/viewer/voxel_imgui_viewer.h"

namespace {

struct EurocFrame
{
    std::filesystem::path image_path;
    double timestamp_seconds = 0.0;
};

struct EurocImuMeasurement
{
    double timestamp_seconds = 0.0;
    cv::Point3f gyroscope;
    cv::Point3f accelerometer;
};

std::string trimCarriageReturn(std::string value)
{
    if (!value.empty() && value.back() == '\r') {
        value.pop_back();
    }
    return value;
}

std::vector<EurocFrame> loadEurocCam0(
    const std::filesystem::path& sequence_path)
{
    const std::filesystem::path csv_path =
        sequence_path / "mav0" / "cam0" / "data.csv";
    const std::filesystem::path image_dir =
        sequence_path / "mav0" / "cam0" / "data";

    std::ifstream input(csv_path);
    if (!input) {
        throw std::runtime_error(
            "Could not open EuRoC cam0 index: " + csv_path.string());
    }

    std::vector<EurocFrame> frames;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty() || line.front() == '#') {
            continue;
        }

        const std::size_t comma = line.find(',');
        if (comma == std::string::npos) {
            throw std::runtime_error(
                "Malformed EuRoC cam0 index at line " +
                std::to_string(line_number));
        }

        const std::string timestamp_text = line.substr(0, comma);
        const std::string filename =
            trimCarriageReturn(line.substr(comma + 1));
        if (timestamp_text.empty() || filename.empty()) {
            throw std::runtime_error(
                "Incomplete EuRoC cam0 index at line " +
                std::to_string(line_number));
        }

        const long double timestamp_ns = std::stold(timestamp_text);
        EurocFrame frame;
        frame.timestamp_seconds =
            static_cast<double>(timestamp_ns * 1.0e-9L);
        frame.image_path = image_dir / filename;
        frames.push_back(std::move(frame));
    }

    if (frames.empty()) {
        throw std::runtime_error(
            "No EuRoC cam0 frames found in " + csv_path.string());
    }
    return frames;
}

std::vector<EurocImuMeasurement> loadEurocImu(
    const std::filesystem::path& sequence_path)
{
    const std::filesystem::path csv_path =
        sequence_path / "mav0" / "imu0" / "data.csv";
    std::ifstream input(csv_path);
    if (!input) {
        throw std::runtime_error(
            "Could not open EuRoC IMU index: " + csv_path.string());
    }

    std::vector<EurocImuMeasurement> measurements;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty() || line.front() == '#') {
            continue;
        }

        std::array<double, 7> values{};
        std::stringstream stream(line);
        std::string token;
        std::size_t column = 0;
        while (std::getline(stream, token, ',')) {
            if (column >= values.size()) {
                break;
            }
            values[column++] = std::stod(trimCarriageReturn(token));
        }
        if (column != values.size()) {
            throw std::runtime_error(
                "Malformed EuRoC IMU row at line " +
                std::to_string(line_number));
        }

        EurocImuMeasurement measurement;
        measurement.timestamp_seconds = values[0] * 1.0e-9;
        measurement.gyroscope = cv::Point3f(
            static_cast<float>(values[1]),
            static_cast<float>(values[2]),
            static_cast<float>(values[3]));
        measurement.accelerometer = cv::Point3f(
            static_cast<float>(values[4]),
            static_cast<float>(values[5]),
            static_cast<float>(values[6]));
        measurements.push_back(measurement);
    }

    if (measurements.empty()) {
        throw std::runtime_error(
            "No EuRoC IMU measurements found in " + csv_path.string());
    }
    return measurements;
}

void saveTrackingTime(
    const std::vector<float>& times,
    const std::filesystem::path& path)
{
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error(
            "Could not write tracking times: " + path.string());
    }
    for (const float time : times) {
        out << std::fixed << std::setprecision(8) << time << '\n';
    }
}

void saveGpuPeakMemoryUsage(const std::filesystem::path& path)
{
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }

    float max_reserved_mb = 0.0f;
    float max_allocated_mb = 0.0f;
    if (torch::cuda::is_available()) {
        namespace allocator = c10::cuda::CUDACachingAllocator;
        const allocator::DeviceStats stats =
            allocator::getDeviceStats(0);
        max_reserved_mb =
            stats.reserved_bytes.front().peak /
            (1024.0f * 1024.0f);
        max_allocated_mb =
            stats.allocated_bytes.front().peak /
            (1024.0f * 1024.0f);
    }

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error(
            "Could not write GPU memory metrics: " + path.string());
    }
    out << "Peak reserved (MB): " << max_reserved_mb << '\n';
    out << "Peak allocated (MB): " << max_allocated_mb << '\n';
}

void convertImageToMapperColorOrder(cv::Mat& image)
{
    if (image.channels() == 3) {
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2RGBA);
    } else if (image.channels() != 1) {
        throw std::runtime_error(
            "EuRoC image must have one, three, or four channels");
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 6 && argc != 7) {
        std::cerr
            << "Usage: " << argv[0]
            << " path_to_vocabulary"
            << " path_to_ORB_SLAM3_settings"
            << " path_to_voxel_mapper_settings"
            << " path_to_EuRoC_sequence"
            << " path_to_output_directory"
            << " (optional)no_viewer\n";
        return 1;
    }

    const bool use_viewer =
        argc != 7 || std::string(argv[6]) != "no_viewer";
    const std::filesystem::path sequence_path(argv[4]);
    const std::filesystem::path output_dir(argv[5]);
    std::filesystem::create_directories(output_dir);

    std::vector<EurocFrame> frames;
    std::vector<EurocImuMeasurement> imu_measurements;
    try {
        frames = loadEurocCam0(sequence_path);
        imu_measurements = loadEurocImu(sequence_path);
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return 1;
    }

    const torch::DeviceType device_type =
        torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "[INFO] Using device: "
              << (device_type == torch::kCUDA ? "CUDA" : "CPU")
              << '\n';

    auto slam = std::make_shared<ORB_SLAM3::System>(
        argv[1],
        argv[2],
        ORB_SLAM3::System::IMU_MONOCULAR,
        use_viewer);
    const float image_scale = slam->GetImageScale();

    auto mapper = std::make_shared<VoxelMapper>(
        slam,
        std::filesystem::path(argv[3]),
        output_dir,
        0,
        device_type);
    mapper->setRuntimeFrameCount(0);
    std::thread training_thread(&VoxelMapper::run, mapper.get());

    std::thread viewer_thread;
    std::shared_ptr<VoxelImGuiViewer> viewer;
    if (use_viewer) {
        viewer = std::make_shared<VoxelImGuiViewer>(slam, mapper);
        viewer_thread = std::thread(&VoxelImGuiViewer::run, viewer.get());
    }

    std::vector<float> tracking_times;
    tracking_times.reserve(frames.size());

    std::cout << "\n-------\n"
              << "Start processing EuRoC cam0 + IMU sequence ...\n"
              << "Images in the sequence: " << frames.size() << "\n"
              << "IMU measurements: " << imu_measurements.size() << "\n\n";

    std::size_t imu_index = 0;
    while (imu_index < imu_measurements.size() &&
           imu_measurements[imu_index].timestamp_seconds <=
               frames.front().timestamp_seconds) {
        ++imu_index;
    }
    if (imu_index > 0) {
        --imu_index;
    }

    bool input_error = false;
    for (std::size_t frame_index = 0;
         frame_index < frames.size();
         ++frame_index) {
        if (slam->isShutDown()) {
            break;
        }

        const EurocFrame& frame = frames[frame_index];
        cv::Mat image =
            cv::imread(frame.image_path.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "[ERROR] Failed to load image: "
                      << frame.image_path << '\n';
            input_error = true;
            break;
        }

        try {
            convertImageToMapperColorOrder(image);
        } catch (const std::exception& error) {
            std::cerr << "[ERROR] " << error.what() << ": "
                      << frame.image_path << '\n';
            input_error = true;
            break;
        }

        if (image_scale != 1.0f) {
            const int width =
                static_cast<int>(image.cols * image_scale);
            const int height =
                static_cast<int>(image.rows * image_scale);
            cv::resize(image, image, cv::Size(width, height));
        }

        std::vector<ORB_SLAM3::IMU::Point> frame_imu;
        if (frame_index > 0) {
            while (imu_index < imu_measurements.size() &&
                   imu_measurements[imu_index].timestamp_seconds <=
                       frame.timestamp_seconds) {
                const EurocImuMeasurement& measurement =
                    imu_measurements[imu_index];
                frame_imu.emplace_back(
                    measurement.accelerometer.x,
                    measurement.accelerometer.y,
                    measurement.accelerometer.z,
                    measurement.gyroscope.x,
                    measurement.gyroscope.y,
                    measurement.gyroscope.z,
                    measurement.timestamp_seconds);
                ++imu_index;
            }
        }

        const auto start = std::chrono::steady_clock::now();
        {
            auto tracking_profile =
                mapper->profileLaptopModule("orb_tracking");
            slam->TrackMonocular(
                image,
                frame.timestamp_seconds,
                frame_imu,
                frame.image_path.string());
        }
        const auto end = std::chrono::steady_clock::now();

        const float tracking_seconds =
            std::chrono::duration_cast<std::chrono::duration<float>>(
                end - start).count();
        tracking_times.push_back(tracking_seconds);
        mapper->setRuntimeFrameCount(
            static_cast<int>(tracking_times.size()));

        double frame_period = 0.0;
        if (frame_index + 1 < frames.size()) {
            frame_period =
                frames[frame_index + 1].timestamp_seconds -
                frame.timestamp_seconds;
        } else if (frame_index > 0) {
            frame_period =
                frame.timestamp_seconds -
                frames[frame_index - 1].timestamp_seconds;
        }
        if (tracking_seconds < frame_period) {
            std::this_thread::sleep_for(
                std::chrono::duration<double>(
                    frame_period - tracking_seconds));
        }
    }

    slam->Shutdown();
    training_thread.join();

    const std::filesystem::path shutdown_dir =
        output_dir /
        (std::to_string(mapper->getIteration()) + "_shutdown");
    saveGpuPeakMemoryUsage(output_dir / "GpuPeakUsageMB.txt");
    saveGpuPeakMemoryUsage(shutdown_dir / "GpuPeakUsageMB.txt");
    saveTrackingTime(tracking_times, output_dir / "TrackingTime.txt");
    saveTrackingTime(
        tracking_times,
        shutdown_dir / "TrackingTime.txt");

    const auto save_trajectories =
        [&](const std::filesystem::path& directory)
    {
        std::filesystem::create_directories(directory);
        slam->SaveTrajectoryTUM(
            (directory / "CameraTrajectory_TUM.txt").string());
        slam->SaveKeyFrameTrajectoryTUM(
            (directory / "KeyFrameTrajectory_TUM.txt").string());
        slam->SaveTrajectoryEuRoC(
            (directory / "CameraTrajectory_EuRoC.txt").string());
        slam->SaveKeyFrameTrajectoryEuRoC(
            (directory / "KeyFrameTrajectory_EuRoC.txt").string());
    };
    save_trajectories(output_dir);
    save_trajectories(shutdown_dir);

    if (use_viewer) {
        viewer_thread.join();
    }

    std::cout << "[EuRoC] Processed " << tracking_times.size()
              << " / " << frames.size()
              << " visual-inertial frames.\n";
    return input_error ? 1 : 0;
}
