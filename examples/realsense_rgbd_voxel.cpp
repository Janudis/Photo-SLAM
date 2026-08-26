#include <torch/torch.h>

#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <c10/cuda/CUDACachingAllocator.h>
#include <librealsense2/rs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "ORB-SLAM3/include/System.h"
#include "include_voxel/viewer/voxel_imgui_viewer.h"
#include "include_voxel/voxel_mapper.h"

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;

void handleSignal(int)
{
    g_stop_requested = 1;
}

struct StreamConfiguration
{
    int width = 640;
    int height = 360;
    int fps = 30;
};

StreamConfiguration readStreamConfiguration(
    const std::filesystem::path& settings_path)
{
    cv::FileStorage settings(settings_path.string(), cv::FileStorage::READ);
    if (!settings.isOpened()) {
        throw std::runtime_error(
            "Cannot open ORB-SLAM settings: " + settings_path.string());
    }

    StreamConfiguration stream;
    if (!settings["Camera.width"].empty()) {
        stream.width = static_cast<int>(settings["Camera.width"]);
    }
    if (!settings["Camera.height"].empty()) {
        stream.height = static_cast<int>(settings["Camera.height"]);
    }
    if (!settings["Camera.fps"].empty()) {
        stream.fps = static_cast<int>(settings["Camera.fps"]);
    }
    if (stream.width <= 0 || stream.height <= 0 || stream.fps <= 0) {
        throw std::runtime_error(
            "Camera.width, Camera.height, and Camera.fps must be positive");
    }
    return stream;
}

std::string scalarString(const double value)
{
    std::ostringstream stream;
    stream << std::setprecision(10) << value;
    return stream.str();
}

bool replaceYamlScalar(
    std::vector<std::string>& lines,
    const std::string& key,
    const std::string& value)
{
    bool replaced = false;
    for (std::string& line : lines) {
        const std::size_t first = line.find_first_not_of(" \t");
        if (first == std::string::npos || line.compare(first, key.size(), key) != 0) {
            continue;
        }
        const std::size_t after_key = first + key.size();
        const std::size_t colon = line.find(':', after_key);
        if (colon == std::string::npos ||
            line.substr(first, colon - first) != key) {
            continue;
        }

        const std::size_t comment = line.find('#', colon + 1);
        const std::string suffix = comment == std::string::npos
            ? std::string()
            : " " + line.substr(comment);
        line = line.substr(0, colon + 1) + " " + value + suffix;
        replaced = true;
    }
    return replaced;
}

std::filesystem::path writeRuntimeOrbSettings(
    const std::filesystem::path& template_path,
    const std::filesystem::path& output_directory,
    const rs2_intrinsics& intrinsics,
    const int fps)
{
    std::ifstream input(template_path);
    if (!input.is_open()) {
        throw std::runtime_error(
            "Cannot read ORB-SLAM settings template: " +
            template_path.string());
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        lines.push_back(line);
    }

    const std::vector<std::pair<std::string, std::string>> replacements = {
        {"Camera1.fx", scalarString(intrinsics.fx)},
        {"Camera1.fy", scalarString(intrinsics.fy)},
        {"Camera1.cx", scalarString(intrinsics.ppx)},
        {"Camera1.cy", scalarString(intrinsics.ppy)},
        {"Camera1.k1", scalarString(intrinsics.coeffs[0])},
        {"Camera1.k2", scalarString(intrinsics.coeffs[1])},
        {"Camera1.p1", scalarString(intrinsics.coeffs[2])},
        {"Camera1.p2", scalarString(intrinsics.coeffs[3])},
        {"Camera1.k3", scalarString(intrinsics.coeffs[4])},
        {"Camera.width", std::to_string(intrinsics.width)},
        {"Camera.height", std::to_string(intrinsics.height)},
        {"Camera.fps", std::to_string(fps)},
        {"Camera.RGB", "1"},
        // Depth is converted from device units to CV_32F metres before TrackRGBD.
        {"RGBD.DepthMapFactor", "1.0"},
    };
    for (const auto& [key, value] : replacements) {
        if (!replaceYamlScalar(lines, key, value)) {
            throw std::runtime_error(
                "Required key is missing from ORB-SLAM settings template: " + key);
        }
    }

    std::filesystem::create_directories(output_directory);
    const std::filesystem::path runtime_path =
        output_directory / "orb_realsense_runtime.yaml";
    std::ofstream output(runtime_path);
    if (!output.is_open()) {
        throw std::runtime_error(
            "Cannot write runtime ORB-SLAM settings: " +
            runtime_path.string());
    }
    for (const std::string& output_line : lines) {
        output << output_line << '\n';
    }
    return runtime_path;
}

void configureSensors(const rs2::device& device)
{
    for (rs2::sensor sensor : device.query_sensors()) {
        try {
            if (sensor.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE)) {
                sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1.0f);
            }
            if (sensor.is<rs2::depth_sensor>() &&
                sensor.supports(RS2_OPTION_EMITTER_ENABLED)) {
                sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 1.0f);
            }
        } catch (const rs2::error& error) {
            std::cerr << "[RealSense] Could not set sensor option: "
                      << error.what() << '\n';
        }
    }
}

void saveTrackingTime(
    const std::vector<float>& tracking_times,
    const std::filesystem::path& output_path)
{
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    std::ofstream output(output_path);
    for (const float tracking_time : tracking_times) {
        output << std::fixed << std::setprecision(6)
               << tracking_time << '\n';
    }
}

void saveGpuPeakMemoryUsage(const std::filesystem::path& output_path)
{
    if (!torch::cuda::is_available()) {
        return;
    }
    namespace allocator = c10::cuda::CUDACachingAllocator;
    const allocator::DeviceStats memory = allocator::getDeviceStats(0);
    const float reserved_mb =
        memory.reserved_bytes[
            static_cast<int>(allocator::StatType::AGGREGATE)].peak /
        (1024.0f * 1024.0f);
    const float allocated_mb =
        memory.allocated_bytes[
            static_cast<int>(allocator::StatType::AGGREGATE)].peak /
        (1024.0f * 1024.0f);

    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    std::ofstream output(output_path);
    output << "Peak reserved (MB): " << reserved_mb << '\n'
           << "Peak allocated (MB): " << allocated_mb << '\n';
}

void saveTrajectories(
    const std::shared_ptr<ORB_SLAM3::System>& slam,
    const std::filesystem::path& output_directory)
{
    std::filesystem::create_directories(output_directory);
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
}

std::string deviceInfo(
    const rs2::device& device,
    const rs2_camera_info field,
    const std::string& fallback = "unknown")
{
    return device.supports(field) ? device.get_info(field) : fallback;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 5 && argc != 6) {
        std::cerr
            << "\nUsage: " << argv[0]
            << " path_to_vocabulary"
            << " path_to_ORB_SLAM3_settings_template"
            << " path_to_voxel_mapping_settings"
            << " path_to_output_directory"
            << " (optional)no_viewer\n";
        return 1;
    }

    const bool use_viewer =
        argc != 6 || std::string(argv[5]) != "no_viewer";
    const std::filesystem::path orb_template(argv[2]);
    const std::filesystem::path voxel_settings(argv[3]);
    const std::filesystem::path output_directory(argv[4]);
    std::filesystem::create_directories(output_directory);

    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);

    try {
        const StreamConfiguration stream =
            readStreamConfiguration(orb_template);

        rs2::context context;
        if (context.query_devices().size() == 0) {
            throw std::runtime_error("No RealSense device is connected");
        }

        rs2::pipeline pipeline(context);
        rs2::config configuration;
        configuration.enable_stream(
            RS2_STREAM_COLOR,
            stream.width,
            stream.height,
            RS2_FORMAT_RGB8,
            stream.fps);
        configuration.enable_stream(
            RS2_STREAM_DEPTH,
            stream.width,
            stream.height,
            RS2_FORMAT_Z16,
            stream.fps);

        rs2::pipeline_profile profile = pipeline.start(configuration);
        const rs2::device device = profile.get_device();
        configureSensors(device);

        const std::string camera_name =
            deviceInfo(device, RS2_CAMERA_INFO_NAME);
        const std::string serial =
            deviceInfo(device, RS2_CAMERA_INFO_SERIAL_NUMBER);
        const std::string usb_type =
            deviceInfo(device, RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR);
        std::cout << "[RealSense] device=" << camera_name
                  << " serial=" << serial
                  << " usb=" << usb_type << '\n';
        if (usb_type.empty() || usb_type.front() != '3') {
            std::cerr
                << "[RealSense] WARNING: camera is not using SuperSpeed USB. "
                   "Use a USB 3 cable/port for performance measurements.\n";
        }

        const rs2::video_stream_profile color_profile =
            profile.get_stream(RS2_STREAM_COLOR)
                .as<rs2::video_stream_profile>();
        const rs2_intrinsics intrinsics = color_profile.get_intrinsics();
        const rs2::depth_sensor depth_sensor =
            device.first<rs2::depth_sensor>();
        const float depth_scale = depth_sensor.get_depth_scale();
        if (!(depth_scale > 0.0f)) {
            throw std::runtime_error("RealSense returned an invalid depth scale");
        }

        std::cout << std::fixed << std::setprecision(6)
                  << "[RealSense] stream=" << intrinsics.width << 'x'
                  << intrinsics.height << '@' << stream.fps
                  << " fx=" << intrinsics.fx
                  << " fy=" << intrinsics.fy
                  << " cx=" << intrinsics.ppx
                  << " cy=" << intrinsics.ppy
                  << " depth_scale=" << depth_scale << '\n';

        const std::filesystem::path runtime_orb_settings =
            writeRuntimeOrbSettings(
                orb_template,
                output_directory,
                intrinsics,
                stream.fps);
        std::cout << "[RealSense] runtime calibration: "
                  << runtime_orb_settings << '\n';

        const torch::DeviceType device_type =
            torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        std::cout << "[INFO] Using device: "
                  << (device_type == torch::kCUDA ? "CUDA" : "CPU")
                  << '\n';

        auto slam = std::make_shared<ORB_SLAM3::System>(
            argv[1],
            runtime_orb_settings.string(),
            ORB_SLAM3::System::RGBD,
            use_viewer);
        const float image_scale = slam->GetImageScale();

        auto mapper = std::make_shared<VoxelMapper>(
            slam,
            voxel_settings,
            output_directory,
            0,
            device_type);
        mapper->setRuntimeFrameCount(0);
        std::thread mapping_thread(&VoxelMapper::run, mapper.get());

        std::thread viewer_thread;
        std::shared_ptr<VoxelImGuiViewer> viewer;
        if (use_viewer) {
            viewer = std::make_shared<VoxelImGuiViewer>(slam, mapper);
            viewer_thread = std::thread(&VoxelImGuiViewer::run, viewer.get());
        }

        rs2::align align_to_color(RS2_STREAM_COLOR);
        std::vector<float> tracking_times;
        tracking_times.reserve(18000);
        bool stream_failed = false;

        std::cout
            << "\n-------\n"
            << "Start processing the live RealSense RGB-D stream.\n"
            << "Press Ctrl+C once to stop and export the map.\n\n";

        while (!g_stop_requested &&
               !slam->isShutDown()) {
            rs2::frameset frames;
            if (!pipeline.poll_for_frames(&frames)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            try {
                const rs2::frameset aligned = align_to_color.process(frames);
                const rs2::video_frame color_frame = aligned.get_color_frame();
                const rs2::depth_frame depth_frame = aligned.get_depth_frame();
                if (!color_frame || !depth_frame) {
                    continue;
                }

                cv::Mat color(
                    cv::Size(color_frame.get_width(), color_frame.get_height()),
                    CV_8UC3,
                    const_cast<void*>(color_frame.get_data()),
                    cv::Mat::AUTO_STEP);
                cv::Mat depth_raw(
                    cv::Size(depth_frame.get_width(), depth_frame.get_height()),
                    CV_16UC1,
                    const_cast<void*>(depth_frame.get_data()),
                    cv::Mat::AUTO_STEP);
                color = color.clone();
                cv::Mat depth_metres;
                depth_raw.convertTo(depth_metres, CV_32FC1, depth_scale);

                if (image_scale != 1.0f) {
                    const int width = static_cast<int>(
                        static_cast<float>(color.cols) * image_scale);
                    const int height = static_cast<int>(
                        static_cast<float>(color.rows) * image_scale);
                    cv::resize(color, color, cv::Size(width, height));
                    cv::resize(
                        depth_metres,
                        depth_metres,
                        cv::Size(width, height),
                        0.0,
                        0.0,
                        cv::INTER_NEAREST);
                }

                mapper->waitForInputQueueSlot();
                const auto start = std::chrono::steady_clock::now();
                {
                    auto tracking_profile =
                        mapper->profileLaptopModule("orb_tracking");
                    slam->TrackRGBD(
                        color,
                        depth_metres,
                        color_frame.get_timestamp() * 1.0e-3,
                        std::vector<ORB_SLAM3::IMU::Point>(),
                        "realsense_" +
                            std::to_string(color_frame.get_frame_number()));
                }
                const auto end = std::chrono::steady_clock::now();
                tracking_times.push_back(
                    std::chrono::duration_cast<std::chrono::duration<float>>(
                        end - start).count());
                mapper->setRuntimeFrameCount(
                    static_cast<int>(tracking_times.size()));
            } catch (const rs2::error& error) {
                std::cerr << "[RealSense] Streaming failed: "
                          << error.what() << '\n';
                stream_failed = true;
                break;
            }
        }

        try {
            pipeline.stop();
        } catch (const rs2::error& error) {
            std::cerr << "[RealSense] Pipeline stop failed: "
                      << error.what() << '\n';
        }

        std::cout << "[RealSense] Stopping SLAM and exporting results ...\n";
        slam->Shutdown();
        mapping_thread.join();

        const std::filesystem::path shutdown_directory =
            output_directory /
            (std::to_string(mapper->getIteration()) + "_shutdown");
        saveTrackingTime(
            tracking_times,
            output_directory / "TrackingTime.txt");
        saveTrackingTime(
            tracking_times,
            shutdown_directory / "TrackingTime.txt");
        saveGpuPeakMemoryUsage(
            output_directory / "GpuPeakUsageMB.txt");
        saveGpuPeakMemoryUsage(
            shutdown_directory / "GpuPeakUsageMB.txt");
        saveTrajectories(slam, output_directory);
        saveTrajectories(slam, shutdown_directory);

        if (use_viewer) {
            viewer_thread.join();
        }

        std::cout << "[RealSense] Processed " << tracking_times.size()
                  << " RGB-D frames. Results: "
                  << shutdown_directory << '\n';
        return stream_failed ? 1 : 0;
    } catch (const rs2::error& error) {
        std::cerr << "[RealSense] " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return 1;
    }
}
