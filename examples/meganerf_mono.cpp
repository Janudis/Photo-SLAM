/**
* Photo-SLAM runner for Mega-NeRF (building-pixsfm) monocular RGB
*
* Directory layout expected:
*   <ROOT>/
*     rgbs/               # images (jpg/png), arbitrary names
*     metadata/           # optional timestamps files
*       timestamps.txt        (format A)  : <filename> <timestamp>
*       timestamps_only.txt   (format B)  : one timestamp per line, same order as sorted rgbs
*
* Usage:
*   meganerf_mono path_to_vocab path_to_orbslam3_yaml path_to_voxel_yaml path_to_root path_to_out_dir/ (optional)no_viewer
*/

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
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ORB-SLAM3/include/System.h"
#include "include_voxel/voxel_mapper.h"
#include "viewer/imgui_viewer.h"

namespace fs = std::filesystem;

static bool ends_with(const std::string& s, const std::string& suf) {
    if (s.size() < suf.size()) return false;
    return std::equal(suf.rbegin(), suf.rend(), s.rbegin(),
                      [](char a, char b){ return std::tolower(a)==std::tolower(b);});
}

// Try multiple timestamp sources:
//  (A) <root>/metadata/timestamps.txt           lines: "<filename> <timestamp>"
//  (B) <root>/metadata/timestamps_only.txt      lines: "<timestamp>" (same order as sorted image list)
static void LoadImagesMegaNeRF(const fs::path& root,
                               std::vector<std::string>& vstrImageFilenames,
                               std::vector<double>& vTimestamps)
{
    const fs::path img_dir = root / "rgbs";
    const fs::path meta_dir = root / "metadata";
    if (!fs::exists(img_dir) || !fs::is_directory(img_dir))
        throw std::runtime_error("[MegaNeRF Loader] rgbs/ folder not found under: " + root.string());

    // 1) collect images
    std::vector<fs::path> images;
    for (auto& p : fs::directory_iterator(img_dir)) {
        if (!p.is_regular_file()) continue;
        const auto& ap = p.path();
        const std::string ext = ap.extension().string();
        if (ends_with(ext, ".jpg") || ends_with(ext, ".jpeg") || ends_with(ext, ".png") || ends_with(ext, ".bmp")) {
            images.push_back(ap);
        }
    }
    if (images.empty())
        throw std::runtime_error("[MegaNeRF Loader] No images in: " + img_dir.string());

    // Sort by filename for consistent ordering
    std::sort(images.begin(), images.end());

    // 2) read timestamps if available
    std::map<std::string, double> ts_by_name;
    bool have_byname = false, have_byorder = false;

    // (A) filename + timestamp
    {
        const fs::path fA = meta_dir / "timestamps.txt";
        if (fs::exists(fA)) {
            std::ifstream f(fA);
            std::string line;
            while (std::getline(f, line)) {
                if (line.empty()) continue;
                std::istringstream ss(line);
                std::string fname;
                double t;
                if (ss >> fname >> t) {
                    // key by plain filename (no directory), to be robust
                    ts_by_name[fs::path(fname).filename().string()] = t;
                }
            }
            have_byname = !ts_by_name.empty();
            if (have_byname)
                std::cout << "[MegaNeRF Loader] Loaded timestamps.txt with " << ts_by_name.size() << " entries.\n";
        }
    }

    // (B) order-only timestamps
    std::vector<double> ts_by_order;
    if (!have_byname) {
        const fs::path fB = meta_dir / "timestamps_only.txt";
        if (fs::exists(fB)) {
            std::ifstream f(fB);
            std::string line;
            while (std::getline(f, line)) {
                if (line.empty()) continue;
                std::istringstream ss(line);
                double t;
                if (ss >> t) ts_by_order.push_back(t);
            }
            have_byorder = (ts_by_order.size() == images.size());
            if (!have_byorder && !ts_by_order.empty()) {
                std::cerr << "[MegaNeRF Loader][warn] timestamps_only.txt count (" << ts_by_order.size()
                          << ") != num images (" << images.size() << "). Will ignore and synthesize.\n";
                ts_by_order.clear();
            }
            if (have_byorder)
                std::cout << "[MegaNeRF Loader] Loaded timestamps_only.txt with " << ts_by_order.size() << " entries.\n";
        }
    }

    // 3) build outputs
    vstrImageFilenames.reserve(images.size());
    vTimestamps.reserve(images.size());

    for (size_t i = 0; i < images.size(); ++i) {
        const fs::path& ap = images[i];
        vstrImageFilenames.push_back(fs::relative(ap, root).string()); // relative to <root> for display/path join

        double t = 0.0;
        if (have_byname) {
            const std::string key = ap.filename().string();
            auto it = ts_by_name.find(key);
            if (it != ts_by_name.end()) t = it->second;
            else {
                // If a few missing, synthesize locally
                t = i * 0.1; // 10 Hz default
            }
        } else if (have_byorder) {
            t = ts_by_order[i];
        } else {
            // fallback: synthesize 10 Hz
            t = i * 0.1;
        }
        vTimestamps.push_back(t);
    }

    std::cout << "[MegaNeRF Loader] Images: " << vstrImageFilenames.size()
              << " | timestamps source: "
              << (have_byname ? "timestamps.txt (by-name)"
                              : have_byorder ? "timestamps_only.txt (by-order)"
                                             : "synthetic @10Hz")
              << std::endl;
}

static void saveTrackingTime(std::vector<float> &vTimesTrack, const std::string &strSavePath)
{
    std::ofstream out(strSavePath);
    float totaltime = 0.f;
    for (float v : vTimesTrack) {
        out << std::fixed << std::setprecision(4) << v << '\n';
        totaltime += v;
    }
    out.close();
}

static void saveGpuPeakMemoryUsage(std::filesystem::path pathSave)
{
    namespace c10Alloc = c10::cuda::CUDACachingAllocator;
    c10Alloc::DeviceStats mem_stats = c10Alloc::getDeviceStats(0);

    c10Alloc::Stat reserved_bytes = mem_stats.reserved_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
    float max_reserved_MB = reserved_bytes.peak / (1024.0f * 1024.0f);

    c10Alloc::Stat alloc_bytes = mem_stats.allocated_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
    float max_alloc_MB = alloc_bytes.peak / (1024.0f * 1024.0f);

    std::ofstream out(pathSave);
    out << "Peak reserved (MB): " << max_reserved_MB << std::endl;
    out << "Peak allocated (MB): " << max_alloc_MB << std::endl;
    out.close();
}

int main(int argc, char **argv)
{
    if (argc != 6 && argc != 7)
    {
        std::cerr << "\nUsage: " << argv[0]
                  << " path_to_vocabulary"                   /*1*/
                  << " path_to_ORB_SLAM3_settings"           /*2*/
                  << " path_to_voxel_mapping_settings"    /*3*/
                  << " path_to_meganerf_root"                /*4*/ // e.g. scripts/building-pixsfm/train
                  << " path_to_output_directory/"            /*5*/
                  << " (optional)no_viewer"                  /*6*/
                  << std::endl;
        return 1;
    }
    bool use_viewer = true;
    if (argc == 7) use_viewer = (std::string(argv[6]) == "no_viewer" ? false : true);

    fs::path root(argv[4]);
    fs::path img_dir = root / "rgbs";
    fs::path meta_dir = root / "metadata";

    if (!fs::exists(root)) { std::cerr << "Root not found: " << root << std::endl; return 1; }
    if (!fs::exists(img_dir)) { std::cerr << "rgbs/ not found under: " << root << std::endl; return 1; }

    std::string output_directory = std::string(argv[5]);
    if (output_directory.back() != '/') output_directory += "/";
    fs::path output_dir(output_directory);

    // Load images & timestamps
    std::vector<std::string> vstrImageFilenamesRGB;
    std::vector<double> vTimestamps;
    try {
        LoadImagesMegaNeRF(root, vstrImageFilenamesRGB, vTimestamps);
    } catch (const std::exception& e) {
        std::cerr << "[MegaNeRF Loader] " << e.what() << std::endl;
        return 1;
    }

    const int nImages = static_cast<int>(vstrImageFilenamesRGB.size());
    if (nImages == 0) {
        std::cerr << "No images found." << std::endl;
        return 1;
    }

    // Device
    torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << (device_type == torch::kCUDA ? "CUDA available! Training on GPU." : "Training on CPU.") << std::endl;

    // Create SLAM system
    std::shared_ptr<ORB_SLAM3::System> pSLAM =
        std::make_shared<ORB_SLAM3::System>(
            argv[1], argv[2], ORB_SLAM3::System::MONOCULAR);
    float imageScale = pSLAM->GetImageScale();

    // Create VoxelMapper
    std::filesystem::path voxel_cfg_path(argv[3]);
    std::shared_ptr<VoxelMapper> pVoxelMapper =
        std::make_shared<VoxelMapper>(
            pSLAM, voxel_cfg_path, output_dir, 0, device_type);
    std::thread training_thd(&VoxelMapper::run, pVoxelMapper.get());

    // Viewer
    std::thread viewer_thd;
    std::shared_ptr<ImGuiViewer> pViewer;
    if (use_viewer) {
        pViewer = std::make_shared<ImGuiViewer>(pSLAM, pVoxelMapper);
        viewer_thd = std::thread(&ImGuiViewer::run, pViewer.get());
    }

    // Tracking time stats
    std::vector<float> vTimesTrack(nImages);

    std::cout << "\n-------\nStart processing sequence ...\nImages in the sequence: " << nImages << "\n\n";

    // Main loop
    cv::Mat im;
    for (int ni = 0; ni < nImages; ni++)
    {
        if (pSLAM->isShutDown()) break;

        const fs::path abs_img_path = root / vstrImageFilenamesRGB[ni];
        im = cv::imread(abs_img_path.string(), cv::IMREAD_UNCHANGED);
        if (im.empty()) {
            std::cerr << "Failed to load image: " << abs_img_path << std::endl;
            return 1;
        }

        // Mega-NeRF rgbs are typically already RGB; just ensure format is RGB for ORB-SLAM3 pipeline
        if (im.type() == CV_8UC3) {
            // If OpenCV read as BGR, convert to RGB for consistency (matches TUM runner)
            cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
        }

        if (imageScale != 1.f) {
            const int width  = static_cast<int>(im.cols * imageScale);
            const int height = static_cast<int>(im.rows * imageScale);
            cv::resize(im, im, cv::Size(width, height));
        }

        const double tframe = vTimestamps[ni];

        auto t1 = std::chrono::steady_clock::now();
        pSLAM->TrackMonocular(im, tframe, std::vector<ORB_SLAM3::IMU::Point>(), vstrImageFilenamesRGB[ni]);
        auto t2 = std::chrono::steady_clock::now();
        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = static_cast<float>(ttrack);

        // simulate real-time pacing
        double T = 0;
        if (ni < nImages - 1) T = vTimestamps[ni + 1] - tframe;
        else if (ni > 0)      T = tframe - vTimestamps[ni - 1];
        if (ttrack < T) usleep((T - ttrack) * 1e6);
    }

    // Stop threads
    pSLAM->Shutdown();
    training_thd.join();
    if (use_viewer) viewer_thd.join();

    // Peak GPU
    saveGpuPeakMemoryUsage(output_dir / "GpuPeakUsageMB.txt");

    // Tracking times
    saveTrackingTime(vTimesTrack, (output_dir / "TrackingTime.txt").string());

    // Save trajectories
    pSLAM->SaveTrajectoryTUM((output_dir / "CameraTrajectory_TUM.txt").string());
    pSLAM->SaveKeyFrameTrajectoryTUM((output_dir / "KeyFrameTrajectory_TUM.txt").string());
    pSLAM->SaveTrajectoryEuRoC((output_dir / "CameraTrajectory_EuRoC.txt").string());
    pSLAM->SaveKeyFrameTrajectoryEuRoC((output_dir / "KeyFrameTrajectory_EuRoC.txt").string());
    // pSLAM->SaveTrajectoryKITTI((output_dir / "CameraTrajectory_KITTI.txt").string());

    return 0;
}
