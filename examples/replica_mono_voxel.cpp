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

#include <c10/cuda/CUDACachingAllocator.h>

#include "ORB-SLAM3/include/System.h"
#include "include_voxel/voxel_mapper.h"
#include "viewer/imgui_viewer.h"

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

// Replica loader: list all files in <seq>/results whose filename starts
// with `prefix` (e.g. "frame"), push full path, then sort.
static void LoadImagesReplica(const std::filesystem::path &pathImageDir,
                              std::vector<std::string> &vstrImageFilenames,
                              const std::string &prefix)
{
    vstrImageFilenames.clear();

    if (!std::filesystem::exists(pathImageDir))
    {
        std::cerr << "[LoadImagesReplica] directory does not exist: "
                  << pathImageDir << std::endl;
        return;
    }

    for (const auto &entry : std::filesystem::directory_iterator(pathImageDir))
    {
        if (!entry.is_regular_file())
            continue;

        const std::string name = entry.path().filename().string();
        // keep only files whose names start with prefix
        if (name.rfind(prefix, 0) == 0)
        {
            vstrImageFilenames.push_back(entry.path().string());
        }
    }

    std::sort(vstrImageFilenames.begin(), vstrImageFilenames.end());
}

// Save per-frame tracking time
static void saveTrackingTime(const std::vector<float> &vTimesTrack,
                             const std::string &pathOut)
{
    std::ofstream out(pathOut);
    float totaltime = 0.f;

    for (size_t i = 0; i < vTimesTrack.size(); ++i)
    {
        out << std::fixed << std::setprecision(8)
            << vTimesTrack[i] << "\n";
        totaltime += vTimesTrack[i];
    }

    out.close();
}

// Save CUDA peak mem stats
static void saveGpuPeakMemoryUsage(const std::filesystem::path &pathSave)
{
    namespace c10Alloc = c10::cuda::CUDACachingAllocator;
    c10Alloc::DeviceStats mem_stats = c10Alloc::getDeviceStats(0);

    float max_reserved_MB =
        mem_stats
            .reserved_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)]
            .peak /
        (1024.0f * 1024.0f);

    float max_alloc_MB =
        mem_stats
            .allocated_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)]
            .peak /
        (1024.0f * 1024.0f);

    std::ofstream out(pathSave);
    out << "Peak reserved (MB): " << max_reserved_MB << "\n";
    out << "Peak allocated (MB): " << max_alloc_MB << "\n";
    out.close();
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------

int main(int argc, char **argv)
{
    // Argument layout mirrors tum_mono_voxel.cpp:
    //
    //   argv[1] path_to_vocabulary
    //   argv[2] path_to_ORB_SLAM3_settings
    //   argv[3] path_to_voxel_mapper_settings
    //   argv[4] path_to_sequence            (Replica scene root)
    //   argv[5] path_to_output_directory/
    //   argv[6] (optional)no_viewer
    //
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

    // -------------------------------------------------------------------------
    // Normalize output directory (ensure trailing slash like tum runner)
    // -------------------------------------------------------------------------
    std::string output_directory = std::string(argv[5]);
    if (output_directory.back() != '/')
        output_directory += "/";
    std::filesystem::path output_dir(output_directory);

    // -------------------------------------------------------------------------
    // Load Replica RGB frames from <seq>/results/frame*.*
    // -------------------------------------------------------------------------
    std::filesystem::path seq_path(argv[4]);
    std::filesystem::path rgb_dir = seq_path / "results";

    std::vector<std::string> vstrImageFilenamesRGB;
    LoadImagesReplica(rgb_dir, vstrImageFilenamesRGB, "frame");

    const int nImages = static_cast<int>(vstrImageFilenamesRGB.size());
    if (nImages == 0)
    {
        std::cerr << std::endl
                  << "[ERROR] No images found in: "
                  << rgb_dir << std::endl;
        return 1;
    }

    // -------------------------------------------------------------------------
    // Pick device (GPU if available)
    // -------------------------------------------------------------------------
    torch::DeviceType device_type =
        torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "[INFO] Using device: "
              << (device_type == torch::kCUDA ? "CUDA" : "CPU")
              << std::endl;

    // -------------------------------------------------------------------------
    // Init ORB-SLAM3 (MONOCULAR, Replica is pure RGB sequence)
    // We pass `use_viewer` exactly like tum_mono_voxel does.
    // -------------------------------------------------------------------------
    std::shared_ptr<ORB_SLAM3::System> pSLAM =
        std::make_shared<ORB_SLAM3::System>(
            argv[1],                          // vocabulary
            argv[2],                          // ORB-SLAM3 settings (Replica yaml)
            ORB_SLAM3::System::MONOCULAR,     // sensor type
            use_viewer                        // internal ORB viewer on/off
        );

    float imageScale = pSLAM->GetImageScale();

    // -------------------------------------------------------------------------
    // Init VoxelMapper and training thread (same pattern as tum_mono_voxel)
    // -------------------------------------------------------------------------
    std::shared_ptr<VoxelMapper> pVoxelMapper =
        std::make_shared<VoxelMapper>(
            pSLAM,
            std::filesystem::path(argv[3]),   // voxel mapper config
            output_dir,
            0,                                // sequence idx
            device_type);

    pVoxelMapper->setRuntimeFrameCount(nImages);
    std::thread training_thd(&VoxelMapper::run, pVoxelMapper.get());

    // -------------------------------------------------------------------------
    // Optional ImGui viewer also mirrors tum_mono_voxel.cpp
    // -------------------------------------------------------------------------
    std::thread viewer_thd;
    std::shared_ptr<ImGuiViewer> pViewer;
    if (use_viewer)
    {
        pViewer = std::make_shared<ImGuiViewer>(pSLAM, pVoxelMapper);
        viewer_thd = std::thread(&ImGuiViewer::run, pViewer.get());
    }

    // -------------------------------------------------------------------------
    // Track per-frame processing time (same vector style as TUM)
    // -------------------------------------------------------------------------
    std::vector<float> vTimesTrack(nImages);

    std::cout << std::endl
              << "-------" << std::endl;
    std::cout << "Start processing sequence ..." << std::endl;
    std::cout << "Images in the sequence: " << nImages << std::endl
              << std::endl;

    // We'll pretend Replica runs at 30 FPS, same as many synthetic RGB sequences.
    // That means "timestamp" = frame_idx / 30.0.
    // That also lets us compute inter-frame spacing T like TUM code does.
    const double synth_fps = 30.0;

    // -------------------------------------------------------------------------
    // Main loop
    // -------------------------------------------------------------------------
    for (int ni = 0; ni < nImages; ++ni)
    {
        if (pSLAM->isShutDown())
            break;

        // Load RGB frame from disk
        cv::Mat im = cv::imread(vstrImageFilenamesRGB[ni], cv::IMREAD_UNCHANGED);
        if (im.empty())
        {
            std::cerr << std::endl
                      << "[ERROR] Failed to load image at: "
                      << vstrImageFilenamesRGB[ni]
                      << std::endl;
            break;
        }

        // BGR -> RGB to match pipeline convention
        cv::cvtColor(im, im, cv::COLOR_BGR2RGB);

        // Synthetic timestamp
        double tframe = static_cast<double>(ni) / synth_fps;

        // Downscale if ORB config requested it (Camera.newWidth/newHeight)
        if (imageScale != 1.f)
        {
            int width  = static_cast<int>(im.cols * imageScale);
            int height = static_cast<int>(im.rows * imageScale);
            cv::resize(im, im, cv::Size(width, height));
        }

        auto t1 = std::chrono::steady_clock::now();

        // Feed to SLAM -- matches TUM call signature
        {
            auto tracking_profile =
                pVoxelMapper->profileLaptopModule("orb_tracking");
            pSLAM->TrackMonocular(
                im,
                tframe,
                std::vector<ORB_SLAM3::IMU::Point>(),  // no IMU here
                vstrImageFilenamesRGB[ni]              // frame ID / name
            );
        }

        auto t2 = std::chrono::steady_clock::now();

        float ttrack = std::chrono::duration_cast<
                           std::chrono::duration<float>>(t2 - t1)
                           .count();
        vTimesTrack[ni] = ttrack;

        // Sleep to simulate real-time cadence, same style as TUM runner
        double T;
        if (ni < nImages - 1)
        {
            double tnext = static_cast<double>(ni + 1) / synth_fps;
            T = tnext - tframe;
        }
        else if (ni > 0)
        {
            double tprev = static_cast<double>(ni - 1) / synth_fps;
            T = tframe - tprev;
        }
        else
        {
            T = 0.0;
        }

        if (ttrack < T)
        {
            double us = (T - ttrack) * 1e6;
            if (us > 0.0)
                usleep(static_cast<useconds_t>(us));
        }
    }

    // -------------------------------------------------------------------------
    // Tear down (mirror tum_mono_voxel order)
    // -------------------------------------------------------------------------
    pSLAM->Shutdown();
    training_thd.join();

    const std::filesystem::path shutdown_dir =
        output_dir /
        (std::to_string(pVoxelMapper->getIteration()) + "_shutdown");

    // GPU peak usage
    saveGpuPeakMemoryUsage(output_dir / "GpuPeakUsageMB.txt");
    saveGpuPeakMemoryUsage(shutdown_dir / "GpuPeakUsageMB.txt");

    // Tracking time statistics
    saveTrackingTime(
        vTimesTrack,
        (output_dir / "TrackingTime.txt").string());

    // Preserve the trajectory beside the exact reconstruction that uses it.
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
    // KITTI is optional for Replica, usually not saved:
    // pSLAM->SaveTrajectoryKITTI(
    //     (output_dir / "CameraTrajectory_KITTI.txt").string());

    if (use_viewer)
        viewer_thd.join();

    return 0;
}
