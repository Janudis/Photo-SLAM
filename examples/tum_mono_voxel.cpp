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

#include <pybind11/embed.h>

#include "ORB-SLAM3/include/System.h"
#include "include_voxel/voxel_mapper.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include "viewer/imgui_viewer.h"

void LoadImages(const std::string &strFile, std::vector<std::string> &vstrImageFilenames,
                std::vector<double> &vTimestamps)
{
    std::ifstream f(strFile.c_str());
    std::string s0;
    std::getline(f, s0);
    std::getline(f, s0);
    std::getline(f, s0);

    while (!f.eof())
    {
        std::string s;
        std::getline(f, s);
        if (!s.empty())
        {
            std::stringstream ss;
            ss << s;
            double t;
            std::string sRGB;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
}

void saveTrackingTime(const std::vector<float> &times, const std::string &path)
{
    std::ofstream out(path);
    for (float t : times)
        out << std::fixed << std::setprecision(4) << t << "\n";
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
    pybind11::initialize_interpreter();
    {
      // make sure Python torch.cuda is initialized first
      pybind11::gil_scoped_acquire gil;
      // add your voxel scripts to PYTHONPATH
      py::module_::import("sys").attr("path").attr("insert")(0, "../scripts_voxel");
      // import the wrapper – this will pull in PyTorch’s cuda module under Python
      py::module_::import("scripts_voxel.python_svraster_bridge.renderer_wrapper");
      // also explicitly touch torch.cuda
      py::module_::import("torch.cuda");
    }
    // now we can safely drop the GIL
    pybind11::gil_scoped_release release;

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
    LoadImages(std::string(argv[4]) + "/rgb.txt", vstrImageFilenamesRGB, vTimestamps);

    int nImages = vstrImageFilenamesRGB.size();
    if (nImages == 0)
    {
        std::cerr << std::endl
                  << "No images found in: " << std::string(argv[4]) + "/rgb.txt" << std::endl;
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
    std::thread training_thd(&VoxelMapper::run, pVoxelMapper.get());

    std::thread viewer_thd;
    std::shared_ptr<ImGuiViewer> pViewer;
    if (use_viewer)
    {
        pViewer = std::make_shared<ImGuiViewer>(pSLAM, pVoxelMapper);
        viewer_thd = std::thread(&ImGuiViewer::run, pViewer.get());
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

        cv::Mat im = cv::imread(std::string(argv[4]) + "/" + vstrImageFilenamesRGB[ni], cv::IMREAD_UNCHANGED);
        cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
        double tframe = vTimestamps[ni];

        if (im.empty())
        {
            std::cerr << std::endl
                      << "Failed to load image at: "
                      << std::string(argv[4]) << "/" << vstrImageFilenamesRGB[ni] << std::endl;
            return 1;
        }

        if (imageScale != 1.f)
        {
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
        }

        auto t1 = std::chrono::steady_clock::now();

        pSLAM->TrackMonocular(im, tframe, std::vector<ORB_SLAM3::IMU::Point>(), vstrImageFilenamesRGB[ni]);

        auto t2 = std::chrono::steady_clock::now();

        float ttrack = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

        double T = (ni < nImages - 1) ? (vTimestamps[ni + 1] - vTimestamps[ni]) : (vTimestamps[ni] - vTimestamps[ni - 1]);
        if (ttrack < T)
            usleep((T - ttrack) * 1e6);
    }

    pSLAM->Shutdown();
    training_thd.join();

    saveGpuPeakMemoryUsage(output_dir / "GpuPeakUsageMB.txt");
    saveTrackingTime(vTimesTrack, (output_dir / "TrackingTime.txt").string());

    pSLAM->SaveTrajectoryTUM((output_dir / "CameraTrajectory_TUM.txt").string());
    pSLAM->SaveKeyFrameTrajectoryTUM((output_dir / "KeyFrameTrajectory_TUM.txt").string());
    pSLAM->SaveTrajectoryEuRoC((output_dir / "CameraTrajectory_EuRoC.txt").string());
    pSLAM->SaveKeyFrameTrajectoryEuRoC((output_dir / "KeyFrameTrajectory_EuRoC.txt").string());

    if (use_viewer)
        viewer_thd.join();

    return 0;
}
