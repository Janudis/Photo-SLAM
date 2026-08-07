#include <torch/torch.h>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <thread>
#include <filesystem>
#include <memory>
#include <iomanip>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ORB-SLAM3/include/System.h"
#include "include_voxel/voxel_mapper.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include "include_voxel/viewer/voxel_imgui_viewer.h"

void LoadImages(const std::filesystem::path &sequence_path,
                std::vector<std::string> &vstrImageFilenames,
                std::vector<double> &vTimestamps)
{
    std::filesystem::path image_list = sequence_path / "rgb.txt";
    if (!std::filesystem::exists(image_list))
    {
        image_list = sequence_path / "association.txt";
    }

    std::ifstream input(image_list);
    if (!input.is_open())
    {
        std::cerr << "[ERROR] Could not open monocular image list: "
                  << image_list << std::endl;
        return;
    }

    std::string line;
    while (std::getline(input, line))
    {
        if (line.empty() || line.front() == '#')
            continue;

        std::stringstream stream(line);
        double timestamp = 0.0;
        std::string rgb_path;
        if (stream >> timestamp >> rgb_path)
        {
            vTimestamps.push_back(timestamp);
            vstrImageFilenames.push_back(rgb_path);
        }
    }
}

void saveTrackingTime(const std::vector<float> &times, const std::string &path)
{
    std::ofstream out(path);
    for (float t : times)
        out << std::fixed << std::setprecision(8) << t << "\n";
    out.close();
}

void saveGpuPeakMemoryUsage(std::filesystem::path pathSave)
{
    namespace c10Alloc = c10::cuda::CUDACachingAllocator;
    c10Alloc::DeviceStats mem_stats = c10Alloc::getDeviceStats(0);

    float max_reserved_MB = mem_stats.reserved_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)].peak / (1024.0 * 1024.0);
    float max_alloc_MB = mem_stats.allocated_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)].peak / (1024.0 * 1024.0);

    std::ofstream out(pathSave);
    out << "Peak reserved (MB): " << max_reserved_MB << "\n";
    out << "Peak allocated (MB): " << max_alloc_MB << "\n";
    out.close();
}

int main(int argc, char **argv)
{
    if (argc != 6 && argc != 7)
    {
        std::cerr << std::endl
                  << "Usage: " << argv[0]
                  << " path_to_vocabulary"
                  << " path_to_ORB_SLAM3_settings"
                  << " path_to_voxel_mapper_settings"
                  << " path_to_sequence"
                  << " path_to_output_directory/"
                  << " (optional)no_viewer"
                  << std::endl;
        return 1;
    }

    bool use_viewer = true;
    if (argc == 7)
        use_viewer = (std::string(argv[6]) == "no_viewer" ? false : true);

    std::string output_directory = std::string(argv[5]);
    if (output_directory.back() != '/')
        output_directory += "/";
    std::filesystem::path output_dir(output_directory);

    std::vector<std::string> vstrImageFilenamesRGB;
    std::vector<double> vTimestamps;
    LoadImages(std::filesystem::path(argv[4]), vstrImageFilenamesRGB, vTimestamps);

    int nImages = vstrImageFilenamesRGB.size();
    if (nImages == 0)
    {
        std::cerr << std::endl
                  << "No images found in: " << argv[4] << std::endl;
        return 1;
    }

    torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "[INFO] Using device: " << (device_type == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

    std::shared_ptr<ORB_SLAM3::System> pSLAM =
        std::make_shared<ORB_SLAM3::System>(
            argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, use_viewer);

    float imageScale = pSLAM->GetImageScale();

    std::shared_ptr<VoxelMapper> pVoxelMapper =
        std::make_shared<VoxelMapper>(pSLAM,
                                      std::filesystem::path(argv[3]),
                                    //   std::filesystem::path(argv[4]),
                                      output_dir,
                                      0,
                                      device_type);
    pVoxelMapper->setRuntimeFrameCount(nImages);
    std::thread training_thd(&VoxelMapper::run, pVoxelMapper.get());

    std::thread viewer_thd;
    std::shared_ptr<VoxelImGuiViewer> pViewer;
    if (use_viewer)
    {
        pViewer = std::make_shared<VoxelImGuiViewer>(pSLAM, pVoxelMapper);
        viewer_thd = std::thread(&VoxelImGuiViewer::run, pViewer.get());
    }

    std::vector<float> vTimesTrack(nImages);

    std::cout << std::endl
              << "-------" << std::endl;
    std::cout << "Start processing sequence ..." << std::endl;
    std::cout << "Images in the sequence: " << nImages << std::endl
              << std::endl;

    for (int ni = 0; ni < nImages; ni++)
    {
        if (pSLAM->isShutDown())
            break;

        const std::string image_path = std::string(argv[4]) + "/" + vstrImageFilenamesRGB[ni];
        cv::Mat im = cv::imread(image_path, cv::IMREAD_UNCHANGED);

        if (im.empty())
        {
            std::cerr << std::endl
                      << "Failed to load image at: "
                      << image_path << std::endl;
            return 1;
        }
        cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
        double tframe = vTimestamps[ni];

        if (imageScale != 1.f)
        {
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
        }

        auto t1 = std::chrono::steady_clock::now();

        {
            auto tracking_profile =
                pVoxelMapper->profileLaptopModule("orb_tracking");
            pSLAM->TrackMonocular(
                im,
                tframe,
                std::vector<ORB_SLAM3::IMU::Point>(),
                image_path);
        }

        auto t2 = std::chrono::steady_clock::now();

        float ttrack = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

        double T = (ni < nImages - 1) ? (vTimestamps[ni + 1] - vTimestamps[ni]) : (vTimestamps[ni] - vTimestamps[ni - 1]);
        if (ttrack < T)
            usleep((T - ttrack) * 1e6);
    }

    pSLAM->Shutdown();
    training_thd.join();

    const std::filesystem::path shutdown_dir =
        output_dir /
        (std::to_string(pVoxelMapper->getIteration()) + "_shutdown");
    saveGpuPeakMemoryUsage(output_dir / "GpuPeakUsageMB.txt");
    saveGpuPeakMemoryUsage(shutdown_dir / "GpuPeakUsageMB.txt");
    saveTrackingTime(vTimesTrack, (output_dir / "TrackingTime.txt").string());

    const auto save_trajectories =
        [&](const std::filesystem::path& directory)
    {
        std::filesystem::create_directories(directory);
        pSLAM->SaveTrajectoryTUM(
            (directory / "CameraTrajectory_TUM.txt").string());
        pSLAM->SaveKeyFrameTrajectoryTUM(
            (directory / "KeyFrameTrajectory_TUM.txt").string());
        pSLAM->SaveTrajectoryEuRoC(
            (directory / "CameraTrajectory_EuRoC.txt").string());
        pSLAM->SaveKeyFrameTrajectoryEuRoC(
            (directory / "KeyFrameTrajectory_EuRoC.txt").string());
    };
    save_trajectories(output_dir);
    save_trajectories(shutdown_dir);

    if (use_viewer)
        viewer_thd.join();

    return 0;
}
