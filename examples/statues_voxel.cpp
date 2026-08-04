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
#include "viewer/imgui_viewer.h"

// jsoncpp
#include <json/json.h>

#include <regex>
#include <vector>
#include <utility>

// ----------------- helpers -----------------

static void LoadImages_TUM_RGBTXT(
    const std::string &strFile,
    std::vector<std::string> &vstrImageFilenames,
    std::vector<double> &vTimestamps)
{
    std::ifstream f(strFile.c_str());
    if (!f.is_open())
    {
        std::cerr << "[LoadImages] could not open: " << strFile << std::endl;
        return;
    }

    while (!f.eof())
    {
        std::string s;
        std::getline(f, s);
        if (s.empty()) continue;
        if (!s.empty() && s[0] == '#') continue;

        std::stringstream ss;
        ss << s;
        double t;
        std::string sRGB;
        ss >> t;
        ss >> sRGB;
        if (!ss.fail())
        {
            vTimestamps.push_back(t);
            vstrImageFilenames.push_back(sRGB);
        }
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
        mem_stats.reserved_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)].peak / (1024.0 * 1024.0);
    float max_alloc_MB =
        mem_stats.allocated_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)].peak / (1024.0 * 1024.0);

    std::ofstream out(pathSave);
    out << "Peak reserved (MB): " << max_reserved_MB << "\n";
    out << "Peak allocated (MB): " << max_alloc_MB << "\n";
    out.close();
}

/**
 * Create a TUM-style rgb.txt from a NeRF/SplatNav transforms.json.
 *
 * We only use frames[*].file_path to get the image order.
 * We generate synthetic timestamps t = i/fps so ORB-SLAM3 gets a monotonic timebase.
 *
 * If dataset_root/rgb.txt already exists, we reuse it.
 */
static std::filesystem::path EnsureRgbTxtFromTransforms(
    const std::filesystem::path& dataset_root,
    const std::filesystem::path& transforms_json_path,
    double fps = 30.0)
{
    namespace fs = std::filesystem;

    if (!fs::exists(transforms_json_path))
        throw std::runtime_error("transforms.json not found at: " + transforms_json_path.string());

    fs::path rgb_txt_path = dataset_root / "rgb.txt";
    if (fs::exists(rgb_txt_path))
    {
        std::cout << "[statues] Found existing rgb.txt: " << rgb_txt_path << " (reusing)\n";
        return rgb_txt_path;
    }

    // Parse transforms.json with jsoncpp
    std::ifstream in(transforms_json_path.string());
    if (!in.is_open())
        throw std::runtime_error("Failed to open transforms.json: " + transforms_json_path.string());

    Json::CharReaderBuilder builder;
    builder["collectComments"] = false;
    Json::Value root;
    std::string errs;
    if (!Json::parseFromStream(builder, in, &root, &errs))
        throw std::runtime_error("jsoncpp parse failed: " + errs);

    if (!root.isObject() || !root.isMember("frames") || !root["frames"].isArray())
        throw std::runtime_error("transforms.json missing 'frames' array.");

    const Json::Value& frames = root["frames"];
    if (frames.empty())
        throw std::runtime_error("transforms.json frames is empty.");

    std::ofstream out(rgb_txt_path.string());
    if (!out.is_open())
        throw std::runtime_error("Failed to create rgb.txt at: " + rgb_txt_path.string());

    out << "# generated from transforms.json\n";
    out << "# timestamp rgb_filename\n";
    out << "# (synthetic timestamps at fps=" << fps << ")\n";

    // Collect valid frames first
    std::vector<std::pair<int, std::string>> indexed; // (frame_id, relpath)
    indexed.reserve(frames.size());

    std::regex re(R"(frame_(\d+))");

    for (Json::ArrayIndex i = 0; i < frames.size(); ++i)
    {
        const Json::Value& fr = frames[i];
        if (!fr.isObject() || !fr.isMember("file_path") || !fr["file_path"].isString())
            continue;

        std::string rel = fr["file_path"].asString();
        fs::path img_path = dataset_root / rel;
        if (!fs::exists(img_path))
        {
            std::cerr << "[statues] WARNING: missing image: " << img_path << "\n";
            continue;
        }

        // Parse frame index from filename: frame_00325.png -> 325
        std::smatch m;
        int fid = -1;
        if (std::regex_search(rel, m, re) && m.size() >= 2)
            fid = std::stoi(m[1].str());

        // If parsing fails, still keep it but send it to the end deterministically
        if (fid < 0) fid = std::numeric_limits<int>::max();

        indexed.emplace_back(fid, rel);
    }

    if (indexed.empty())
        throw std::runtime_error("rgb.txt generation found 0 existing frames.");

    // Sort by frame index ascending (video order)
    std::sort(indexed.begin(), indexed.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Write timestamps monotonically after sorting
    size_t written = 0;
    for (const auto& [fid, rel] : indexed)
    {
        double t = (fps > 0.0) ? (double(written) / fps) : double(written);
        out << std::fixed << std::setprecision(6) << t << " " << rel << "\n";
        written++;
    }

    out.close();

    if (written == 0)
        throw std::runtime_error("rgb.txt generation wrote 0 frames (check file_path and dataset_root).");

    std::cout << "[statues] Created rgb.txt: " << rgb_txt_path << " (frames=" << written << ")\n";
    return rgb_txt_path;
}

// ----------------- main -----------------

int main(int argc, char **argv)
{
    namespace fs = std::filesystem;

    // Args:
    //   1 vocab
    //   2 ORB-SLAM3 settings (statues.yaml)
    //   3 voxel mapper settings (your voxel yaml)
    //   4 path_to_statues_dataset_root (contains transforms.json, images/, etc.)
    //   5 path_to_output_dir
    //   6 (optional) no_viewer
    if (argc != 6 && argc != 7)
    {
        std::cerr << "\nUsage: " << argv[0]
                  << " path_to_vocabulary"
                  << " path_to_ORB_SLAM3_settings"
                  << " path_to_voxel_mapper_settings"
                  << " path_to_statues_dataset_root"
                  << " path_to_output_directory/"
                  << " (optional)no_viewer\n";
        return 1;
    }

    bool use_viewer = true;
    if (argc == 7)
        use_viewer = (std::string(argv[6]) == "no_viewer" ? false : true);

    fs::path dataset_root(argv[4]);
    if (!fs::exists(dataset_root))
    {
        std::cerr << "[statues] dataset_root does not exist: " << dataset_root << "\n";
        return 1;
    }

    std::string output_directory = std::string(argv[5]);
    if (!output_directory.empty() && output_directory.back() != '/')
        output_directory += "/";
    fs::path output_dir(output_directory);

    // Create (or reuse) rgb.txt based on transforms.json
    fs::path rgb_txt = EnsureRgbTxtFromTransforms(dataset_root, dataset_root / "transforms.json", 30.0);

    // Load image list
    std::vector<std::string> vstrImageFilenamesRGB;
    std::vector<double> vTimestamps;
    LoadImages_TUM_RGBTXT(rgb_txt.string(), vstrImageFilenamesRGB, vTimestamps);

    const int nImages = (int)vstrImageFilenamesRGB.size();
    if (nImages == 0)
    {
        std::cerr << "[statues] No images found via rgb.txt: " << rgb_txt << "\n";
        return 1;
    }

    torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "[INFO] Using device: " << (device_type == torch::kCUDA ? "CUDA" : "CPU") << "\n";

    std::shared_ptr<ORB_SLAM3::System> pSLAM =
        std::make_shared<ORB_SLAM3::System>(
            argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, use_viewer);

    float imageScale = pSLAM->GetImageScale();

    std::shared_ptr<VoxelMapper> pVoxelMapper =
        std::make_shared<VoxelMapper>(
            pSLAM,
            fs::path(argv[3]),
            output_dir,
            0,
            device_type);
    pVoxelMapper->setRuntimeFrameCount(nImages);

    std::thread training_thd(&VoxelMapper::run, pVoxelMapper.get());

    std::thread viewer_thd;
    std::shared_ptr<ImGuiViewer> pViewer;
    if (use_viewer)
    {
        pViewer = std::make_shared<ImGuiViewer>(pSLAM, pVoxelMapper);
        viewer_thd = std::thread(&ImGuiViewer::run, pViewer.get());
    }

    std::vector<float> vTimesTrack(nImages);

    std::cout << "\n-------\n";
    std::cout << "Start processing sequence (statues) ...\n";
    std::cout << "Images: " << nImages << "\n\n";

    for (int ni = 0; ni < nImages; ni++)
    {
        if (pSLAM->isShutDown())
            break;

        fs::path img_path = dataset_root / vstrImageFilenamesRGB[ni];
        cv::Mat im = cv::imread(img_path.string(), cv::IMREAD_UNCHANGED);

        if (im.empty())
        {
            std::cerr << "\nFailed to load image at: " << img_path << "\n";
            return 1;
        }

        cv::cvtColor(im, im, cv::COLOR_BGR2RGB);

        if (imageScale != 1.f)
        {
            int width = int(im.cols * imageScale);
            int height = int(im.rows * imageScale);
            cv::resize(im, im, cv::Size(width, height));
        }

        double tframe = vTimestamps[ni];

        auto t1 = std::chrono::steady_clock::now();
        {
            auto tracking_profile =
                pVoxelMapper->profileLaptopModule("orb_tracking");
            pSLAM->TrackMonocular(
                im,
                tframe,
                std::vector<ORB_SLAM3::IMU::Point>(),
                vstrImageFilenamesRGB[ni]);
        }
        auto t2 = std::chrono::steady_clock::now();

        float ttrack = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

        double T = 0.0;
        if (ni < nImages - 1) T = (vTimestamps[ni + 1] - vTimestamps[ni]);
        else if (ni > 0)      T = (vTimestamps[ni] - vTimestamps[ni - 1]);

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
