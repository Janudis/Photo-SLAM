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
#include "include_voxel/viewer/voxel_imgui_viewer.h"

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
    const std::filesystem::path assoc_path(strAssociationFilename);
    if (!std::filesystem::exists(assoc_path) || !std::filesystem::is_regular_file(assoc_path))
    {
        std::cerr << "[LoadImagesRGBD] association path is not a regular file: "
                  << strAssociationFilename << std::endl;
        return;
    }

    std::ifstream fAssociation(assoc_path);
    if (!fAssociation.is_open())
    {
        std::cerr << "[LoadImagesRGBD] couldn't open: " << strAssociationFilename << std::endl;
        return;
    }

    std::string s;
    int malformed_lines = 0;
    while (std::getline(fAssociation, s))
    {
        if (s.empty() || s[0] == '#')
            continue;

        std::stringstream ss(s);

        double t_rgb = 0.0, t_d = 0.0;
        std::string sRGB, sD;
        if (!(ss >> t_rgb >> sRGB >> t_d >> sD))
        {
            malformed_lines++;
            continue;
        }

        vTimestamps.push_back(t_rgb); // use RGB timestamp as frame time
        vstrImageFilenamesRGB.push_back(sRGB);
        vstrImageFilenamesD.push_back(sD);
    }

    if (malformed_lines > 0)
    {
        std::cerr << "[LoadImagesRGBD] skipped malformed lines: " << malformed_lines << std::endl;
    }
}

static void saveTrackingTime(const std::vector<float> &times, const std::string &path)
{
    std::ofstream out(path);
    for (float t : times)
        out << std::fixed << std::setprecision(4) << t << "\n";
    out.close();
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

static void saveGpuPeakMemoryUsage(std::filesystem::path pathSave)
{
    const GpuMemoryStats stats = getGpuPeakMemoryStats();
    if (!pathSave.parent_path().empty())
        std::filesystem::create_directories(pathSave.parent_path());
    std::ofstream out(pathSave);
    out << "Peak reserved (MB): " << stats.reserved_mb << "\n";
    out << "Peak allocated (MB): " << stats.allocated_mb << "\n";
    out.close();
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

// ----------------- main -----------------
int main(int argc, char **argv)
{
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
    const auto total_start = std::chrono::steady_clock::now();

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
    pVoxelMapper->setRuntimeFrameCount(nImages);

    std::thread training_thd(&VoxelMapper::run, pVoxelMapper.get());

    // Viewer thread (imgui)
    std::thread viewer_thd;
    std::shared_ptr<VoxelImGuiViewer> pViewer;
    if (use_viewer)
    {
        pViewer = std::make_shared<VoxelImGuiViewer>(pSLAM, pVoxelMapper);
        viewer_thd = std::thread(&VoxelImGuiViewer::run, pViewer.get());
    }

    std::vector<float> vTimesTrack(nImages);
    int processed_frames = 0;

    std::cout << "\n-------\n";
    std::cout << "Start processing sequence ..." << std::endl;
    std::cout << "Images in the sequence: " << nImages << "\n\n";

    // main frame loop
    for (int ni = 0; ni < nImages; ni++)
    {
        if (pSLAM->isShutDown())
            break;

        // HI-SLAM2 bounds its producer/consumer queue to eight entries. The
        // ScanNet mapper enables an equivalent keyframe-level backpressure
        // limit so asynchronous ORB local mapping cannot outrun voxel mapping.
        pVoxelMapper->waitForInputQueueSlot();

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
        {
            auto tracking_profile =
                pVoxelMapper->profileLaptopModule("orb_tracking");
            pSLAM->TrackRGBD(imRGB,
                             imD,
                             tframe,
                             std::vector<ORB_SLAM3::IMU::Point>(),
                             vstrImageFilenamesRGB[ni]);
        }

        auto t2 = std::chrono::steady_clock::now();

        float ttrack =
            std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;
        ++processed_frames;

        // sleep to simulate real-time pacing, same pattern as mono voxel runner
        double T;
        if (ni < nImages - 1)
            T = vTimestamps[ni + 1] - vTimestamps[ni];
        else if (ni > 0)
            T = vTimestamps[ni] - vTimestamps[ni - 1];
        else
            T = 0.0;

        if (ttrack < T) {
            usleep(static_cast<useconds_t>((T - ttrack) * 1e6));
        }
    }

    // Shutdown SLAM + mapper thread
    pSLAM->Shutdown();
    training_thd.join();

    const int final_iteration = pVoxelMapper->getIteration();
    const std::filesystem::path shutdown_dir =
        output_dir / (std::to_string(final_iteration) + "_shutdown");
    const std::filesystem::path final_map_path =
        shutdown_dir / "ply" / "voxel_model" /
        ("iteration_" + std::to_string(final_iteration)) / "voxel_model.ply";
    const int keyframes =
        pVoxelMapper->scene_ ? static_cast<int>(pVoxelMapper->scene_->keyframes().size()) : 0;
    const int voxels =
        pVoxelMapper->voxel_model_ ? pVoxelMapper->voxel_model_->numVoxels() : 0;

    // Save tracking time stats
    saveTrackingTime(vTimesTrack, (output_dir / "TrackingTime.txt").string());

    // Preserve a run-local trajectory beside the matching reconstruction.
    const auto save_trajectories =
        [&](const std::filesystem::path& trajectory_dir)
    {
        std::filesystem::create_directories(trajectory_dir);
        pSLAM->SaveTrajectoryTUM(
            (trajectory_dir / "CameraTrajectory_TUM.txt").string());
        pSLAM->SaveKeyFrameTrajectoryTUM(
            (trajectory_dir / "KeyFrameTrajectory_TUM.txt").string());
        pSLAM->SaveTrajectoryEuRoC(
            (trajectory_dir / "CameraTrajectory_EuRoC.txt").string());
        pSLAM->SaveKeyFrameTrajectoryEuRoC(
            (trajectory_dir / "KeyFrameTrajectory_EuRoC.txt").string());
        pSLAM->SaveTrajectoryKITTI(
            (trajectory_dir / "CameraTrajectory_KITTI.txt").string());
    };
    save_trajectories(output_dir);
    save_trajectories(shutdown_dir);

    const auto total_end = std::chrono::steady_clock::now();
    const double total_wall_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            total_end - total_start).count();
    const double total_seconds = std::max(0.0, total_wall_seconds);
    const GpuMemoryStats gpu_stats = getGpuPeakMemoryStats();

    // Save GPU peak usage stats
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
        viewer_thd.join();

    return 0;
}
