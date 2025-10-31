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
#include "viewer/imgui_viewer.h"

#include <c10/cuda/CUDACachingAllocator.h>

// ----------------- helpers -----------------

// association.txt -> rgb_path depth_path and timestamps.
// (same format Photo-SLAM expects)
static void LoadImagesRGBD(
    const std::string &strAssociationFilename,
    std::vector<std::string> &vstrImageFilenamesRGB,
    std::vector<std::string> &vstrImageFilenamesD,
    std::vector<double> &vTimestamps)
{
    std::ifstream fAssociation(strAssociationFilename.c_str());
    if (!fAssociation.is_open())
    {
        std::cerr << "[LoadImagesRGBD] couldn't open: " << strAssociationFilename << std::endl;
        return;
    }

    while (!fAssociation.eof())
    {
        std::string s;
        std::getline(fAssociation, s);
        if (s.empty()) continue;

        std::stringstream ss;
        ss << s;

        double t_rgb, t_d;
        std::string sRGB, sD;
        ss >> t_rgb;
        vTimestamps.push_back(t_rgb); // use RGB timestamp as frame time
        ss >> sRGB;
        vstrImageFilenamesRGB.push_back(sRGB);
        ss >> t_d;    // depth timestamp (not actually used separately here)
        ss >> sD;
        vstrImageFilenamesD.push_back(sD);
    }
}

static void saveTrackingTime(const std::vector<float> &times, const std::string &path)
{
    std::ofstream out(path);
    for (float t : times)
        out << std::fixed << std::setprecision(4) << t << "\n";
    out.close();
}

static void saveGpuPeakMemoryUsage(std::filesystem::path pathSave)
{
    namespace c10Alloc = c10::cuda::CUDACachingAllocator;
    c10Alloc::DeviceStats mem_stats = c10Alloc::getDeviceStats(0);

    float max_reserved_MB =
        mem_stats.reserved_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)].peak /
        (1024.0 * 1024.0);
    float max_alloc_MB =
        mem_stats.allocated_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)].peak /
        (1024.0 * 1024.0);

    std::ofstream out(pathSave);
    out << "Peak reserved (MB): " << max_reserved_MB << "\n";
    out << "Peak allocated (MB): " << max_alloc_MB << "\n";
    out.close();
}

// ----------------- main -----------------
int main(int argc, char **argv)
{
    namespace py = pybind11;

    // We mirror tum_mono_voxel: initialize Python first so torch.cuda is live
    py::initialize_interpreter();
    {
        py::gil_scoped_acquire gil;
        // add your voxel python scripts location
        py::module_::import("sys").attr("path").attr("insert")(0, "../scripts_voxel");
        // warm up renderer wrapper (spawns watchdog thread, imports torch.cuda, etc.)
        py::module_::import("scripts_voxel.python_svraster_bridge.renderer_wrapper");
        // explicitly touch torch.cuda so CUDA context is ready
        py::module_::import("torch.cuda");
    }
    py::gil_scoped_release release;

    // arg parsing
    // rgbd version originally:
    //   1 vocab
    //   2 ORB-SLAM3 settings
    //   3 voxel_mapper settings
    //   4 path_to_sequence
    //   5 path_to_association
    //   6 output_dir
    //   7 (optional)no_viewer
    if (argc != 7 && argc != 8)
    {
        std::cerr << "\nUsage: " << argv[0]
                  << " path_to_vocabulary"                   // 1
                  << " path_to_ORB_SLAM3_settings"           // 2
                  << " path_to_voxel_mapper_settings"        // 3  (was gaussian cfg)
                  << " path_to_sequence"                     // 4
                  << " path_to_association"                  // 5
                  << " path_to_output_directory/"            // 6
                  << " (optional)no_viewer"                  // 7
                  << std::endl;
        return 1;
    }

    bool use_viewer = true;
    if (argc == 8)
        use_viewer = (std::string(argv[7]) == "no_viewer" ? false : true);

    // prep output dir
    std::string output_directory = std::string(argv[6]);
    if (output_directory.back() != '/')
        output_directory += "/";
    std::filesystem::path output_dir(output_directory);

    // load image/depth filelists
    std::vector<std::string> vstrImageFilenamesRGB;
    std::vector<std::string> vstrImageFilenamesD;
    std::vector<double> vTimestamps;
    LoadImagesRGBD(std::string(argv[5]),
                   vstrImageFilenamesRGB,
                   vstrImageFilenamesD,
                   vTimestamps);

    if (vstrImageFilenamesRGB.empty())
    {
        std::cerr << "\nNo RGB images found from association file.\n";
        return 1;
    }
    if (vstrImageFilenamesD.size() != vstrImageFilenamesRGB.size())
    {
        std::cerr << "\nRGB/depth count mismatch.\n";
        return 1;
    }

    const int nImages = (int)vstrImageFilenamesRGB.size();

    // device selection
    torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "[INFO] Using device: "
              << (device_type == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

    // Create SLAM system (RGBD)
    // note: System signature in your mono voxel version is:
    // System(vocab, settings, sensor_type, use_viewer)
    // so we match that here.
    std::shared_ptr<ORB_SLAM3::System> pSLAM =
        std::make_shared<ORB_SLAM3::System>(
            argv[1], argv[2], ORB_SLAM3::System::RGBD, use_viewer);

    float imageScale = pSLAM->GetImageScale();

    // Create VoxelMapper (instead of GaussianMapper)
    // Constructor signature you used:
    //   VoxelMapper(pSLAM,
    //               voxel_cfg_path,
    //               output_dir,
    //               start_iter,
    //               device_type)
    std::shared_ptr<VoxelMapper> pVoxelMapper =
        std::make_shared<VoxelMapper>(pSLAM,
                                      std::filesystem::path(argv[3]),
                                      output_dir,
                                      0,
                                      device_type);

    std::thread training_thd(&VoxelMapper::run, pVoxelMapper.get());

    // Viewer thread (imgui)
    std::thread viewer_thd;
    std::shared_ptr<ImGuiViewer> pViewer;
    if (use_viewer)
    {
        pViewer = std::make_shared<ImGuiViewer>(pSLAM, pVoxelMapper);
        viewer_thd = std::thread(&ImGuiViewer::run, pViewer.get());
    }

    std::vector<float> vTimesTrack(nImages);

    std::cout << "\n-------\n";
    std::cout << "Start processing sequence ..." << std::endl;
    std::cout << "Images in the sequence: " << nImages << "\n\n";

    // main frame loop
    for (int ni = 0; ni < nImages; ni++)
    {
        if (pSLAM->isShutDown())
            break;

        // load RGB + depth
        cv::Mat imRGB = cv::imread(std::string(argv[4]) + "/" + vstrImageFilenamesRGB[ni],
                                   cv::IMREAD_UNCHANGED);
        if (imRGB.empty())
        {
            std::cerr << "\nFailed to load RGB image at: "
                      << std::string(argv[4]) << "/"
                      << vstrImageFilenamesRGB[ni] << std::endl;
            return 1;
        }
        cv::cvtColor(imRGB, imRGB, cv::COLOR_BGR2RGB);

        cv::Mat imD = cv::imread(std::string(argv[4]) + "/" + vstrImageFilenamesD[ni],
                                 cv::IMREAD_UNCHANGED);
        if (imD.empty())
        {
            std::cerr << "\nFailed to load depth image at: "
                      << std::string(argv[4]) << "/"
                      << vstrImageFilenamesD[ni] << std::endl;
            return 1;
        }

        double tframe = vTimestamps[ni];

        // scale if ORB-SLAM3 config uses scale
        if (imageScale != 1.f)
        {
            int width = imRGB.cols * imageScale;
            int height = imRGB.rows * imageScale;
            cv::resize(imRGB, imRGB, cv::Size(width, height));
            cv::resize(imD, imD, cv::Size(width, height));
        }

        auto t1 = std::chrono::steady_clock::now();

        // TrackRGBD signature:
        // TrackRGBD(imRGB, imD, timestamp, imuVec, imgName)
        pSLAM->TrackRGBD(imRGB,
                         imD,
                         tframe,
                         std::vector<ORB_SLAM3::IMU::Point>(),
                         vstrImageFilenamesRGB[ni]);

        auto t2 = std::chrono::steady_clock::now();

        float ttrack =
            std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

        // sleep to simulate real-time pacing, same pattern as mono voxel runner
        double T;
        if (ni < nImages - 1)
            T = vTimestamps[ni + 1] - vTimestamps[ni];
        else if (ni > 0)
            T = vTimestamps[ni] - vTimestamps[ni - 1];
        else
            T = 0.0;

        if (ttrack < T)
            usleep((T - ttrack) * 1e6);
    }

    // Shutdown SLAM + mapper thread
    pSLAM->Shutdown();
    training_thd.join();

    // Save GPU peak usage stats
    saveGpuPeakMemoryUsage(output_dir / "GpuPeakUsageMB.txt");

    // Save tracking time stats
    saveTrackingTime(vTimesTrack, (output_dir / "TrackingTime.txt").string());

    // Save trajectories
    pSLAM->SaveTrajectoryTUM((output_dir / "CameraTrajectory_TUM.txt").string());
    pSLAM->SaveKeyFrameTrajectoryTUM((output_dir / "KeyFrameTrajectory_TUM.txt").string());
    pSLAM->SaveTrajectoryEuRoC((output_dir / "CameraTrajectory_EuRoC.txt").string());
    pSLAM->SaveKeyFrameTrajectoryEuRoC((output_dir / "KeyFrameTrajectory_EuRoC.txt").string());
    pSLAM->SaveTrajectoryKITTI((output_dir / "CameraTrajectory_KITTI.txt").string());

    if (use_viewer)
        viewer_thd.join();

    return 0;
}
