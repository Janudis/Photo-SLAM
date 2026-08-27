#include "include_voxel/voxel_mapper.h"
#include "include_voxel/voxel_mapper_utils.h"
#include "include_voxel/voxel_mapper_supervision.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <opencv2/flann.hpp>
#include <random>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <c10/cuda/CUDACachingAllocator.h>

#include "ORB-SLAM3/include/Atlas.h"
#include "ORB-SLAM3/include/KeyFrame.h"
#include "ORB-SLAM3/include/MapPoint.h"

void VoxelMapper::logKeyframeCameraToRerunRecordings(
    const std::shared_ptr<VoxelKeyframe>& pkf,
    unsigned long kf_id,
    bool log_reconstruction_mesh)
{
    if (!pkf || !rerun_params_.enable_rerun_) {
        return;
    }
    const bool needs_camera_recording =
        (log_reconstruction_mesh && rerun_params_.rerun_reconstruction_mesh_) ||
        rerun_params_.run_whole_run_ ||
        rerun_params_.rerun_svrecon_debug_;
    if (!needs_camera_recording) {
        return;
    }

    const unsigned long rerun_kf_begin =
        static_cast<unsigned long>(std::max(0, rerun_params_.rerun_keyframe_start_));
    if (rerun_params_.rerun_max_keyframes_ > 0 &&
        (kf_id < rerun_kf_begin ||
         kf_id >= rerun_kf_begin + static_cast<unsigned long>(rerun_params_.rerun_max_keyframes_))) {
        return;
    }

    try {
        // ORB-SLAM stores Tcw. Use its inverse directly so cameras, ORB
        // MapPoints, and reconstructed geometry stay in the same world frame.
        const Eigen::Matrix4f T_W_C =
            pkf->getPosef().inverse().matrix();

        const sv::Camera& camera = scene_->cameras_.at(pkf->camera_id_);
        const float fx = static_cast<float>(camera.fx());
        const float fy = static_cast<float>(camera.fy());
        const float cx = static_cast<float>(camera.cx());
        const float cy = static_cast<float>(camera.cy());
        const int source_frame_id =
            voxel_utils::parseFrameIdFromPath(pkf->img_filename_);
        const std::vector<Eigen::Vector2f> kps_uv;
        const std::vector<int> track_ids;

        auto log_debug_camera =
            [&](bool enabled, const std::string& recording_name)
        {
            if (!enabled) {
                return;
            }
            sv::RerunVisualizerBridge::instance().visualizeDebugCamera(
                recording_name,
                T_W_C,
                pkf->img_undist_,
                kps_uv,
                track_ids,
                getIteration(),
                static_cast<int>(kf_id),
                fx, fy, cx, cy,
                source_frame_id);
        };

        log_debug_camera(
            log_reconstruction_mesh && rerun_params_.rerun_reconstruction_mesh_,
            "reconstruction_mesh");
        log_debug_camera(rerun_params_.run_whole_run_, "whole_run");
        log_debug_camera(rerun_params_.rerun_svrecon_debug_, "svrecon_debug");
    } catch (const c10::Error& e) {
        (void)e;
    } catch (const std::exception& e) {
        (void)e;
    }
}

void VoxelMapper::saveRerunRecordingsAtShutdown()
{
    if (!rerun_params_.enable_rerun_) {
        return;
    }

    ensureEmbeddedPythonRuntime(/*import_torch_cuda=*/false);

    const auto rrd_dir = result_dir_ / "rerun";
    std::filesystem::create_directories(rrd_dir);

    if (rerun_params_.run_whole_run_) {
        sv::RerunVisualizerBridge::instance().saveDebugRecording(
            "whole_run",
            (rrd_dir / "whole_run.rrd").string());
    }
    if (rerun_params_.rerun_svrecon_debug_) {
        sv::RerunVisualizerBridge::instance().saveDebugRecording(
            "svrecon_debug",
            (rrd_dir / "svrecon_debug.rrd").string());
    }
    if (rerun_params_.rerun_reconstruction_mesh_) {
        sv::RerunVisualizerBridge::instance().saveDebugRecording(
            "reconstruction_mesh",
            (rrd_dir / "run_reconstruction_mesh.rrd").string());
    }
    if (rerun_params_.rerun_maps_) {
        sv::RerunVisualizerBridge::instance().saveDebugRecording(
            "maps",
            (rrd_dir / "maps.rrd").string());
    }
}

void VoxelMapper::alignAndLogNvbloxReferenceMesh(
    const std::filesystem::path& shutdown_dir)
{
    if (!rerun_params_.enable_rerun_ ||
        !rerun_params_.run_whole_run_ ||
        !rerun_params_.rerun_nvblox_mesh_) {
        return;
    }

    const std::filesystem::path source_mesh =
        rerun_params_.rerun_nvblox_mesh_path_;
    const std::filesystem::path source_trajectory =
        source_mesh.parent_path() / "CameraTrajectory_TUM.txt";
    if (!std::filesystem::exists(source_mesh) ||
        !std::filesystem::exists(source_trajectory)) {
        std::cerr << "[RERUN/nvblox] missing reference mesh or trajectory: "
                  << source_mesh << ", " << source_trajectory << "\n";
        return;
    }

    const std::filesystem::path rerun_dir = shutdown_dir / "rerun";
    std::filesystem::create_directories(rerun_dir);
    const std::filesystem::path target_trajectory =
        rerun_dir / "monocular_keyframes_tum.txt";
    const std::filesystem::path aligned_mesh =
        rerun_dir / "nvblox_reference_aligned.ply";
    const std::filesystem::path alignment_report =
        rerun_dir / "nvblox_reference_alignment.txt";

    // Use the reference run's exact frame timestamps. Replica stores integer
    // frame IDs while ScanNet stores seconds, so deriving the convention from
    // filenames or FPS would make the trajectory association dataset-specific.
    std::vector<double> source_frame_timestamps;
    {
        std::ifstream source_poses(source_trajectory);
        std::string line;
        while (std::getline(source_poses, line)) {
            if (line.empty() || line.front() == '#') {
                continue;
            }
            std::istringstream fields(line);
            double timestamp = std::numeric_limits<double>::quiet_NaN();
            if (fields >> timestamp && std::isfinite(timestamp)) {
                source_frame_timestamps.push_back(timestamp);
            }
        }
    }

    std::ofstream trajectory(target_trajectory);
    if (!trajectory) {
        std::cerr << "[RERUN/nvblox] cannot write target trajectory: "
                  << target_trajectory << "\n";
        return;
    }
    trajectory << std::fixed << std::setprecision(9);
    std::size_t pose_count = 0;
    if (scene_) {
        for (const auto& [keyframe_id, keyframe] : scene_->keyframes()) {
            if (!keyframe || !keyframe->set_pose_) {
                continue;
            }
            double timestamp = keyframe->source_timestamp_;
            int source_frame_id = keyframe->source_frame_id_;
            if (source_frame_id < 0) {
                source_frame_id = voxel_utils::parseFrameIdFromPath(
                    keyframe->img_filename_);
            }
            if (source_frame_id >= 0 &&
                source_frame_id <
                    static_cast<int>(source_frame_timestamps.size())) {
                timestamp = source_frame_timestamps[
                    static_cast<std::size_t>(source_frame_id)];
            }
            if (!std::isfinite(timestamp)) {
                timestamp = voxel_utils::parseFrameTimestampFromPath(
                    keyframe->img_filename_);
            }
            if (!std::isfinite(timestamp)) {
                timestamp = static_cast<double>(keyframe_id);
            }

            const Sophus::SE3d camera_to_world = keyframe->getPose().inverse();
            const Eigen::Vector3d translation = camera_to_world.translation();
            const Eigen::Quaterniond rotation =
                camera_to_world.unit_quaternion().normalized();
            trajectory << timestamp << ' '
                       << translation.x() << ' '
                       << translation.y() << ' '
                       << translation.z() << ' '
                       << rotation.x() << ' '
                       << rotation.y() << ' '
                       << rotation.z() << ' '
                       << rotation.w() << '\n';
            ++pose_count;
        }
    }
    trajectory.close();
    if (pose_count < 4) {
        std::cerr << "[RERUN/nvblox] insufficient final keyframe poses: "
                  << pose_count << "\n";
        return;
    }

    sv::RerunVisualizerBridge& bridge =
        sv::RerunVisualizerBridge::instance();
    if (!bridge.alignReferencePlyMesh(
            source_mesh.string(),
            source_trajectory.string(),
            target_trajectory.string(),
            aligned_mesh.string(),
            alignment_report.string())) {
        std::cerr << "[RERUN/nvblox] Sim(3) alignment failed; reference mesh "
                     "was not logged.\n";
        return;
    }

    bridge.visualizeDebugPlyMesh(
        "whole_run",
        aligned_mesh.string(),
        getIteration(),
        "world/reference/nvblox_mesh",
        /*static_mesh=*/true);
}

void VoxelMapper::logLearnedDepthMapsToWholeRunRerun()
{
    if (!rerun_params_.enable_rerun_ ||
        !rerun_params_.run_whole_run_ || !scene_) {
        return;
    }

    struct DepthFrame
    {
        std::shared_ptr<VoxelKeyframe> keyframe;
        cv::Mat gt_depth;
    };
    std::vector<DepthFrame> frames;
    std::vector<std::pair<float, double>> scale_samples;

    for (const auto& [keyframe_id, keyframe] : scene_->keyframes()) {
        (void)keyframe_id;
        if (!keyframe || keyframe->monocular_depth_prior_.empty() ||
            keyframe->monocular_depth_source_ == sv::LearnedDepthSource::None) {
            continue;
        }

        cv::Mat gt_depth;
        voxel_eval::getKeyframeDepthMetersForEval(
            keyframe,
            keyframe->monocular_depth_prior_.rows,
            keyframe->monocular_depth_prior_.cols,
            gt_depth);
        if (!gt_depth.empty()) {
            const torch::Tensor model_depth = torch::from_blob(
                keyframe->monocular_depth_prior_.data,
                {keyframe->monocular_depth_prior_.rows,
                 keyframe->monocular_depth_prior_.cols},
                torch::TensorOptions().dtype(torch::kFloat32)).clone();
            voxel_eval::DepthScaleFitStats stats;
            if (voxel_eval::computeDepthScaleFitStats(
                    model_depth,
                    gt_depth,
                    std::max(1.0e-8f, RGBD_min_depth_),
                    1.0e6f,
                    stats)) {
                scale_samples.emplace_back(
                    stats.scale,
                    static_cast<double>(stats.overlap_count));
            }
        }
        frames.push_back({keyframe, std::move(gt_depth)});
    }
    if (frames.empty()) {
        return;
    }

    float model_to_metric_scale = 1.0f;
    const bool has_metric_scale =
        voxel_eval::computeWeightedMedianScale(
            scale_samples, model_to_metric_scale);
    std::cout << "[RERUN/depth] frames=" << frames.size()
              << " gt_frames=" << scale_samples.size()
              << " model_to_metric_scale="
              << (has_metric_scale ? model_to_metric_scale : 1.0f)
              << (has_metric_scale ? "" : " (unavailable)") << "\n";

    constexpr float viz_min = 0.0f;
    constexpr float viz_max = 6.0f;
    const float valid_min = std::max(1.0e-8f, RGBD_min_depth_);
    const float valid_max = std::min(RGBD_max_depth_, viz_max);
    sv::RerunVisualizerBridge& bridge =
        sv::RerunVisualizerBridge::instance();

    for (const DepthFrame& frame : frames) {
        const auto& keyframe = frame.keyframe;
        cv::Mat model_depth_metric;
        keyframe->monocular_depth_prior_.convertTo(
            model_depth_metric,
            CV_32FC1,
            has_metric_scale ? model_to_metric_scale : 1.0f);
        const cv::Mat model_rgb = voxel_eval::bgrToRgbImage(
            voxel_eval::colorizeDepthMatJet(
                model_depth_metric,
                valid_min,
                valid_max,
                viz_min,
                viz_max));
        const int iteration = keyframe->monocular_depth_prior_iteration_ >= 0
            ? keyframe->monocular_depth_prior_iteration_
            : getIteration();
        const char* model_entity =
            keyframe->monocular_depth_source_ == sv::LearnedDepthSource::TandemMvs
                ? "depth/model/mvs"
                : "depth/model/omnidata";
        bridge.visualizeDebugImage(
            "whole_run",
            model_rgb,
            iteration,
            static_cast<int>(keyframe->fid_),
            model_entity);

        if (!frame.gt_depth.empty()) {
            const cv::Mat gt_rgb = voxel_eval::bgrToRgbImage(
                voxel_eval::colorizeDepthMatJet(
                    frame.gt_depth,
                    valid_min,
                    valid_max,
                    viz_min,
                    viz_max));
            bridge.visualizeDebugImage(
                "whole_run",
                gt_rgb,
                iteration,
                static_cast<int>(keyframe->fid_),
                "depth/ground_truth");
        }
    }
}

namespace {
constexpr float kCommonEvalScale = 1.0f;
constexpr float kCommonEvalVoxelLength = 5.0f * kCommonEvalScale / 512.0f;
constexpr float kCommonEvalSdfTrunc = 0.04f * kCommonEvalScale;
constexpr float kCommonEvalDepthTrunc = 30.0f;
constexpr int kCommonEvalMedianKernel = 21; // Gaussian-SLAM evaluator.py uses 20; OpenCV requires an odd kernel.
constexpr float kCommonEvalDepthOutlierThreshold = 0.1f;
const Eigen::Vector3f kCommonEvalCompensation(
    0.0f * kCommonEvalScale / 512.0f,
    2.5f * kCommonEvalScale / 512.0f,
    -2.5f * kCommonEvalScale / 512.0f);

struct TriangleMeshRgb
{
    std::vector<Eigen::Vector3f> vertices;
    std::vector<std::array<uint32_t, 3>> faces;
    std::vector<std::array<uint8_t, 3>> colors;
};

static cv::Mat filterDepthOutliersLikeGaussianSlam(const cv::Mat& depth_map)
{
    CV_Assert(depth_map.type() == CV_32FC1);
    if (depth_map.empty()) return depth_map.clone();

    cv::Mat median_filtered = depth_map.clone();
    const int num_passes = std::max(1, kCommonEvalMedianKernel / 5);
    for (int pass = 0; pass < num_passes; ++pass)
    {
        cv::medianBlur(median_filtered, median_filtered, 5);
    }

    cv::Mat filtered = depth_map.clone();
    for (int y = 0; y < depth_map.rows; ++y)
    {
        const float* src_ptr = depth_map.ptr<float>(y);
        const float* med_ptr = median_filtered.ptr<float>(y);
        float* out_ptr = filtered.ptr<float>(y);
        for (int x = 0; x < depth_map.cols; ++x)
        {
            const float d = src_ptr[x];
            const float m = med_ptr[x];
            if (!std::isfinite(d) || d <= 0.0f)
            {
                out_ptr[x] = 0.0f;
                continue;
            }
            if (!std::isfinite(m) || m <= 0.0f) continue;
            if (std::abs(d - m) > kCommonEvalDepthOutlierThreshold) out_ptr[x] = m;
        }
    }
    return filtered;
}

static bool saveTriangleMeshPly(
    const std::filesystem::path& ply_path,
    const TriangleMeshRgb& mesh,
    bool write_vertex_colors = true)
{
    if (mesh.vertices.empty() || mesh.faces.empty()) return false;

    std::vector<float> vertices_xyz;
    vertices_xyz.reserve(mesh.vertices.size() * 3);
    for (const auto& v : mesh.vertices)
    {
        vertices_xyz.push_back(v.x());
        vertices_xyz.push_back(v.y());
        vertices_xyz.push_back(v.z());
    }

    std::vector<uint32_t> tri_idx;
    tri_idx.reserve(mesh.faces.size() * 3);
    for (const auto& f : mesh.faces)
    {
        tri_idx.push_back(f[0]);
        tri_idx.push_back(f[1]);
        tri_idx.push_back(f[2]);
    }

    std::vector<uint8_t> vertex_rgb;
    const bool has_colors =
        write_vertex_colors && mesh.colors.size() == mesh.vertices.size();
    if (has_colors)
    {
        vertex_rgb.reserve(mesh.colors.size() * 3);
        for (const auto& c : mesh.colors)
        {
            vertex_rgb.push_back(c[0]);
            vertex_rgb.push_back(c[1]);
            vertex_rgb.push_back(c[2]);
        }
    }

    std::filebuf fb;
    fb.open(ply_path, std::ios::out | std::ios::binary);
    std::ostream out(&fb);
    if (out.fail()) return false;

    tinyply::PlyFile ply;
    ply.add_properties_to_element(
        "vertex", {"x", "y", "z"},
        tinyply::Type::FLOAT32, static_cast<uint64_t>(mesh.vertices.size()),
        reinterpret_cast<uint8_t*>(vertices_xyz.data()),
        tinyply::Type::INVALID, 0);

    if (has_colors)
    {
        ply.add_properties_to_element(
            "vertex", {"red", "green", "blue"},
            tinyply::Type::UINT8, static_cast<uint64_t>(mesh.vertices.size()),
            reinterpret_cast<uint8_t*>(vertex_rgb.data()),
            tinyply::Type::INVALID, 0);
    }

    ply.add_properties_to_element(
        "face", {"vertex_indices"},
        tinyply::Type::UINT32, static_cast<uint64_t>(mesh.faces.size()),
        reinterpret_cast<uint8_t*>(tri_idx.data()),
        tinyply::Type::UINT8, 3);

    ply.write(out, true);
    fb.close();
    return true;
}

static void capTriangleMeshForExport(
    TriangleMeshRgb& mesh,
    std::size_t max_vertices,
    std::size_t max_faces,
    float quantization_eps)
{
    if (mesh.vertices.empty() || mesh.faces.empty()) return;
    if ((max_vertices == 0 || mesh.vertices.size() <= max_vertices) &&
        (max_faces == 0 || mesh.faces.size() <= max_faces)) {
        return;
    }

    struct QuantizedVertexKey
    {
        std::int64_t x = 0;
        std::int64_t y = 0;
        std::int64_t z = 0;

        bool operator==(const QuantizedVertexKey& other) const noexcept
        {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct QuantizedVertexKeyHash
    {
        std::size_t operator()(const QuantizedVertexKey& key) const noexcept
        {
            const std::uint64_t x = static_cast<std::uint64_t>(key.x) * 73856093ull;
            const std::uint64_t y = static_cast<std::uint64_t>(key.y) * 19349663ull;
            const std::uint64_t z = static_cast<std::uint64_t>(key.z) * 83492791ull;
            return static_cast<std::size_t>(x ^ y ^ z);
        }
    };

    const float eps = std::max(quantization_eps, 1.0e-8f);
    auto make_key = [eps](const Eigen::Vector3f& p) -> QuantizedVertexKey
    {
        return QuantizedVertexKey{
            static_cast<std::int64_t>(std::llround(static_cast<double>(p.x()) / eps)),
            static_cast<std::int64_t>(std::llround(static_cast<double>(p.y()) / eps)),
            static_cast<std::int64_t>(std::llround(static_cast<double>(p.z()) / eps))};
    };

    const bool has_colors = mesh.colors.size() == mesh.vertices.size();
    TriangleMeshRgb capped;
    capped.vertices.reserve(max_vertices > 0 ? std::min(max_vertices, mesh.vertices.size()) : mesh.vertices.size());
    capped.faces.reserve(max_faces > 0 ? std::min(max_faces, mesh.faces.size()) : mesh.faces.size());
    if (has_colors) {
        capped.colors.reserve(capped.vertices.capacity());
    }

    std::unordered_map<QuantizedVertexKey, uint32_t, QuantizedVertexKeyHash> vertex_map;
    vertex_map.reserve(capped.vertices.capacity());

    const std::size_t face_limit = max_faces > 0 ? max_faces : mesh.faces.size();
    for (const auto& face : mesh.faces)
    {
        if (capped.faces.size() >= face_limit) break;

        std::array<QuantizedVertexKey, 3> keys;
        std::array<uint32_t, 3> remapped{};
        bool valid_face = true;
        std::size_t new_vertices = 0;

        for (int i = 0; i < 3; ++i)
        {
            if (face[i] >= mesh.vertices.size()) {
                valid_face = false;
                break;
            }
            keys[i] = make_key(mesh.vertices[face[i]]);
            const auto it = vertex_map.find(keys[i]);
            if (it != vertex_map.end()) {
                remapped[i] = it->second;
                continue;
            }

            bool duplicate_new_key = false;
            for (int j = 0; j < i; ++j)
            {
                if (keys[i] == keys[j] && vertex_map.find(keys[i]) == vertex_map.end()) {
                    duplicate_new_key = true;
                    break;
                }
            }
            if (!duplicate_new_key) ++new_vertices;
        }
        if (!valid_face) continue;
        if (max_vertices > 0 && capped.vertices.size() + new_vertices > max_vertices) {
            continue;
        }

        for (int i = 0; i < 3; ++i)
        {
            const auto it = vertex_map.find(keys[i]);
            if (it != vertex_map.end()) {
                remapped[i] = it->second;
                continue;
            }

            const uint32_t new_index = static_cast<uint32_t>(capped.vertices.size());
            vertex_map.emplace(keys[i], new_index);
            remapped[i] = new_index;
            capped.vertices.push_back(mesh.vertices[face[i]]);
            if (has_colors) {
                capped.colors.push_back(mesh.colors[face[i]]);
            }
        }

        if (remapped[0] == remapped[1] ||
            remapped[1] == remapped[2] ||
            remapped[0] == remapped[2]) {
            continue;
        }
        capped.faces.push_back({remapped[0], remapped[1], remapped[2]});
    }

    if (!capped.vertices.empty() && !capped.faces.empty()) {
        mesh = std::move(capped);
    }
}

static void weldTriangleMeshVertices(TriangleMeshRgb& mesh, float quantization_eps = 1.0e-6f)
{
    if (mesh.vertices.empty() || mesh.faces.empty()) return;
    if (quantization_eps <= 0.0f) quantization_eps = 1.0e-6f;

    struct QuantizedVertexKey
    {
        std::int64_t x = 0;
        std::int64_t y = 0;
        std::int64_t z = 0;
        bool operator==(const QuantizedVertexKey& other) const noexcept
        {
            return x == other.x && y == other.y && z == other.z;
        }
    };
    struct QuantizedVertexHash
    {
        std::size_t operator()(const QuantizedVertexKey& key) const noexcept
        {
            std::size_t h = static_cast<std::size_t>(key.x);
            h ^= static_cast<std::size_t>(key.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= static_cast<std::size_t>(key.z) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };
    auto make_key = [&](const Eigen::Vector3f& v)
    {
        return QuantizedVertexKey{
            static_cast<std::int64_t>(std::llround(v.x() / quantization_eps)),
            static_cast<std::int64_t>(std::llround(v.y() / quantization_eps)),
            static_cast<std::int64_t>(std::llround(v.z() / quantization_eps))};
    };

    std::unordered_map<QuantizedVertexKey, uint32_t, QuantizedVertexHash> remap;
    remap.reserve(mesh.vertices.size());
    TriangleMeshRgb welded;
    welded.vertices.reserve(mesh.vertices.size());
    welded.colors.reserve(mesh.colors.size());
    welded.faces.reserve(mesh.faces.size());

    auto get_or_add = [&](uint32_t old_idx) -> uint32_t
    {
        if (old_idx >= mesh.vertices.size()) return 0;
        const QuantizedVertexKey key = make_key(mesh.vertices[old_idx]);
        auto it = remap.find(key);
        if (it != remap.end()) return it->second;
        const uint32_t new_idx = static_cast<uint32_t>(welded.vertices.size());
        remap.emplace(key, new_idx);
        welded.vertices.push_back(mesh.vertices[old_idx]);
        if (mesh.colors.size() == mesh.vertices.size()) {
            welded.colors.push_back(mesh.colors[old_idx]);
        }
        return new_idx;
    };

    for (const auto& face : mesh.faces)
    {
        const uint32_t a = get_or_add(face[0]);
        const uint32_t b = get_or_add(face[1]);
        const uint32_t c = get_or_add(face[2]);
        if (a == b || b == c || a == c) continue;
        welded.faces.push_back({a, b, c});
    }

    mesh = std::move(welded);
}

static void triangleMeshToTensors(
    const TriangleMeshRgb& mesh,
    torch::Tensor& vertices,
    torch::Tensor& colors,
    torch::Tensor& triangles)
{
    vertices = torch::empty(
        {static_cast<int64_t>(mesh.vertices.size()), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto v_acc = vertices.accessor<float, 2>();
    for (int64_t i = 0; i < vertices.size(0); ++i)
    {
        const auto& v = mesh.vertices[static_cast<std::size_t>(i)];
        v_acc[i][0] = v.x();
        v_acc[i][1] = v.y();
        v_acc[i][2] = v.z();
    }

    if (mesh.colors.size() == mesh.vertices.size())
    {
        colors = torch::empty(
            {static_cast<int64_t>(mesh.colors.size()), 3},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
        auto c_acc = colors.accessor<uint8_t, 2>();
        for (int64_t i = 0; i < colors.size(0); ++i)
        {
            const auto& c = mesh.colors[static_cast<std::size_t>(i)];
            c_acc[i][0] = c[0];
            c_acc[i][1] = c[1];
            c_acc[i][2] = c[2];
        }
    }
    else
    {
        colors = torch::Tensor();
    }

    triangles = torch::empty(
        {static_cast<int64_t>(mesh.faces.size()), 3},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto f_acc = triangles.accessor<int64_t, 2>();
    for (int64_t i = 0; i < triangles.size(0); ++i)
    {
        const auto& f = mesh.faces[static_cast<std::size_t>(i)];
        f_acc[i][0] = static_cast<int64_t>(f[0]);
        f_acc[i][1] = static_cast<int64_t>(f[1]);
        f_acc[i][2] = static_cast<int64_t>(f[2]);
    }
}

struct SparseTsdfVolume
{
    struct Voxel
    {
        float tsdf = 1.0f;
        uint16_t weight = 0;
        std::array<uint8_t, 3> color{0, 0, 0};
    };

    struct Key
    {
        int x;
        int y;
        int z;

        bool operator==(const Key& other) const noexcept
        {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct KeyHash
    {
        std::size_t operator()(const Key& key) const noexcept
        {
            const std::uint64_t x = static_cast<std::uint32_t>(key.x) * 73856093u;
            const std::uint64_t y = static_cast<std::uint32_t>(key.y) * 19349663u;
            const std::uint64_t z = static_cast<std::uint32_t>(key.z) * 83492791u;
            return static_cast<std::size_t>(x ^ y ^ z);
        }
    };

    explicit SparseTsdfVolume(float voxel_length_in, float sdf_trunc_in)
        : voxel_length(voxel_length_in),
          sdf_trunc(sdf_trunc_in)
    {}

    static Key fromVector(const Eigen::Vector3i& idx)
    {
        return Key{idx.x(), idx.y(), idx.z()};
    }

    static Eigen::Vector3i toVector(const Key& key)
    {
        return Eigen::Vector3i(key.x, key.y, key.z);
    }

    Eigen::Vector3f voxelCenter(const Eigen::Vector3i& idx) const
    {
        return (idx.cast<float>().array() + 0.5f).matrix() * voxel_length;
    }

    void integrate(
        const cv::Mat& color_rgb,
        const cv::Mat& depth_map,
        const std::vector<float>& intr,
        const Sophus::SE3f& Tcw,
        float depth_trunc)
    {
        CV_Assert(color_rgb.type() == CV_8UC3);
        CV_Assert(depth_map.type() == CV_32FC1);
        if (intr.size() < 4) throw std::runtime_error("SparseTsdfVolume::integrate: expected fx, fy, cx, cy");

        const float fx = intr[0];
        const float fy = intr[1];
        const float cx = intr[2];
        const float cy = intr[3];
        const Sophus::SE3f Twc = Tcw.inverse();
        const Eigen::Matrix3f Rwc = Twc.rotationMatrix();
        const Eigen::Vector3f twc = Twc.translation();
        const float step = voxel_length;

        for (int v = 0; v < depth_map.rows; ++v)
        {
            const float ry = (static_cast<float>(v) - cy) / fy;
            const float* depth_ptr = depth_map.ptr<float>(v);
            const cv::Vec3b* color_ptr = color_rgb.ptr<cv::Vec3b>(v);

            for (int u = 0; u < depth_map.cols; ++u)
            {
                const float depth = depth_ptr[u];
                if (!std::isfinite(depth) || depth <= 0.0f || depth > depth_trunc) continue;

                const float rx = (static_cast<float>(u) - cx) / fx;
                const float z_min = std::max(0.0f, depth - sdf_trunc);
                const float z_max = std::min(depth_trunc, depth + sdf_trunc);
                const cv::Vec3b rgb = color_ptr[u];
                Key last_key{std::numeric_limits<int>::min(), std::numeric_limits<int>::min(), std::numeric_limits<int>::min()};

                for (float zv = z_min; zv <= z_max + 1e-6f; zv += step)
                {
                    const Eigen::Vector3f p_cam(rx * zv, ry * zv, zv);
                    const Eigen::Vector3f p_world = Rwc * p_cam + twc;
                    const Eigen::Vector3i idx = (p_world / voxel_length).array().floor().cast<int>().matrix();
                    const Key key = fromVector(idx);
                    if (key == last_key) continue;
                    last_key = key;

                    const float sdf = depth - zv;
                    const float tsdf_sample = std::max(-1.0f, std::min(1.0f, sdf / sdf_trunc));
                    auto& voxel = voxels[key];
                    const float weight_old = static_cast<float>(voxel.weight);
                    const float weight_new = std::min(65535.0f, weight_old + 1.0f);
                    voxel.tsdf = (voxel.tsdf * weight_old + tsdf_sample) / weight_new;
                    for (int c = 0; c < 3; ++c)
                    {
                        const float fused = (static_cast<float>(voxel.color[c]) * weight_old
                                           + static_cast<float>(rgb[c])) / weight_new;
                        voxel.color[c] = static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(255.0f, fused))));
                    }
                    voxel.weight = static_cast<uint16_t>(weight_new);
                }
            }
        }
    }

    TriangleMeshRgb extractMesh(
        float min_weight = 1.0e-4f,
        std::size_t max_vertices = 0,
        std::size_t max_faces = 0) const
    {
        TriangleMeshRgb mesh;
        if (voxels.empty()) return mesh;
        min_weight = std::max(0.0f, min_weight);

        const std::array<Eigen::Vector3i, 8> corner_offsets = {
            Eigen::Vector3i(0, 0, 0), Eigen::Vector3i(1, 0, 0),
            Eigen::Vector3i(1, 1, 0), Eigen::Vector3i(0, 1, 0),
            Eigen::Vector3i(0, 0, 1), Eigen::Vector3i(1, 0, 1),
            Eigen::Vector3i(1, 1, 1), Eigen::Vector3i(0, 1, 1)};
        const std::array<std::array<int, 4>, 6> tetrahedra = {{
            {{0, 5, 1, 6}},
            {{0, 1, 2, 6}},
            {{0, 2, 3, 6}},
            {{0, 3, 7, 6}},
            {{0, 7, 4, 6}},
            {{0, 4, 5, 6}}
        }};
        const std::array<std::array<int, 2>, 6> tetra_edges = {{
            {{0, 1}}, {{1, 2}}, {{2, 0}}, {{0, 3}}, {{1, 3}}, {{2, 3}}
        }};

        struct CellKey
        {
            int x;
            int y;
            int z;
            bool operator==(const CellKey& other) const noexcept
            {
                return x == other.x && y == other.y && z == other.z;
            }
        };
        struct CellHash
        {
            std::size_t operator()(const CellKey& key) const noexcept
            {
                return KeyHash{}(Key{key.x, key.y, key.z});
            }
        };

        std::unordered_set<CellKey, CellHash> candidate_cells;
        candidate_cells.reserve(voxels.size() * 8);
        for (const auto& kv : voxels)
        {
            const Eigen::Vector3i base = toVector(kv.first);
            for (int dz = -1; dz <= 0; ++dz)
            {
                for (int dy = -1; dy <= 0; ++dy)
                {
                    for (int dx = -1; dx <= 0; ++dx)
                    {
                        const Eigen::Vector3i cell = base + Eigen::Vector3i(dx, dy, dz);
                        candidate_cells.insert(CellKey{cell.x(), cell.y(), cell.z()});
                    }
                }
            }
        }

        struct InterpVertex
        {
            Eigen::Vector3f pos;
            Eigen::Vector3f color;
        };

        auto cap_reached = [&]() -> bool
        {
            return (max_faces > 0 && mesh.faces.size() >= max_faces) ||
                   (max_vertices > 0 && mesh.vertices.size() + 3 > max_vertices);
        };

        auto append_triangle = [&](const InterpVertex& a, const InterpVertex& b, const InterpVertex& c) -> bool
        {
            if (cap_reached()) return false;
            const uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
            mesh.vertices.push_back(a.pos);
            mesh.vertices.push_back(b.pos);
            mesh.vertices.push_back(c.pos);
            auto clamp_color = [](const Eigen::Vector3f& col)
            {
                return std::array<uint8_t, 3>{
                    static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(255.0f, col.x())))),
                    static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(255.0f, col.y())))),
                    static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(255.0f, col.z()))))};
            };
            mesh.colors.push_back(clamp_color(a.color));
            mesh.colors.push_back(clamp_color(b.color));
            mesh.colors.push_back(clamp_color(c.color));
            mesh.faces.push_back({base, base + 1, base + 2});
            return true;
        };

        for (const auto& cell_key : candidate_cells)
        {
            if (cap_reached()) break;
            const Eigen::Vector3i cell(cell_key.x, cell_key.y, cell_key.z);
            std::array<float, 8> corner_tsdf{};
            std::array<uint16_t, 8> corner_weight{};
            std::array<Eigen::Vector3f, 8> corner_pos{};
            std::array<Eigen::Vector3f, 8> corner_color{};
            bool any_weight = false;

            for (int i = 0; i < 8; ++i)
            {
                const Key key = fromVector(cell + corner_offsets[i]);
                const auto it = voxels.find(key);
                if (it == voxels.end())
                {
                    corner_weight[i] = 0;
                    continue;
                }
                corner_tsdf[i] = it->second.tsdf;
                corner_weight[i] = it->second.weight;
                corner_pos[i] = voxelCenter(cell + corner_offsets[i]);
                corner_color[i] = Eigen::Vector3f(
                    static_cast<float>(it->second.color[0]),
                    static_cast<float>(it->second.color[1]),
                    static_cast<float>(it->second.color[2]));
                any_weight = any_weight || (static_cast<float>(corner_weight[i]) >= min_weight);
            }
            if (!any_weight) continue;

            for (const auto& tet : tetrahedra)
            {
                bool tet_valid = true;
                for (int local_idx = 0; local_idx < 4; ++local_idx)
                {
                    if (static_cast<float>(corner_weight[tet[local_idx]]) < min_weight)
                    {
                        tet_valid = false;
                        break;
                    }
                }
                if (!tet_valid) continue;

                std::vector<InterpVertex> poly_vertices;
                poly_vertices.reserve(4);
                for (const auto& edge : tetra_edges)
                {
                    const int a = tet[edge[0]];
                    const int b = tet[edge[1]];
                    const bool a_inside = corner_tsdf[a] < 0.0f;
                    const bool b_inside = corner_tsdf[b] < 0.0f;
                    if (a_inside == b_inside) continue;

                    const float denom = corner_tsdf[a] - corner_tsdf[b];
                    float t = 0.5f;
                    if (std::abs(denom) > 1e-8f) t = corner_tsdf[a] / denom;
                    t = std::max(0.0f, std::min(1.0f, t));
                    poly_vertices.push_back({
                        corner_pos[a] + t * (corner_pos[b] - corner_pos[a]),
                        corner_color[a] + t * (corner_color[b] - corner_color[a])});
                }

                if (poly_vertices.size() < 3) continue;
                if (poly_vertices.size() == 3)
                {
                    append_triangle(poly_vertices[0], poly_vertices[1], poly_vertices[2]);
                    if (cap_reached()) return mesh;
                    continue;
                }

                Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
                for (const auto& vtx : poly_vertices) centroid += vtx.pos;
                centroid /= static_cast<float>(poly_vertices.size());

                Eigen::Vector3f normal = (poly_vertices[1].pos - poly_vertices[0].pos)
                                       .cross(poly_vertices[2].pos - poly_vertices[0].pos);
                if (normal.norm() < 1e-8f && poly_vertices.size() >= 4)
                {
                    normal = (poly_vertices[2].pos - poly_vertices[0].pos)
                           .cross(poly_vertices[3].pos - poly_vertices[0].pos);
                }
                if (normal.norm() < 1e-8f) continue;
                normal.normalize();

                Eigen::Vector3f axis_u = poly_vertices[0].pos - centroid;
                if (axis_u.norm() < 1e-8f) axis_u = poly_vertices[1].pos - centroid;
                if (axis_u.norm() < 1e-8f) continue;
                axis_u.normalize();
                Eigen::Vector3f axis_v = normal.cross(axis_u);
                if (axis_v.norm() < 1e-8f) continue;
                axis_v.normalize();

                std::vector<int> order(poly_vertices.size());
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(), [&](int lhs, int rhs)
                {
                    const Eigen::Vector3f dl = poly_vertices[lhs].pos - centroid;
                    const Eigen::Vector3f dr = poly_vertices[rhs].pos - centroid;
                    const float al = std::atan2(dl.dot(axis_v), dl.dot(axis_u));
                    const float ar = std::atan2(dr.dot(axis_v), dr.dot(axis_u));
                    return al < ar;
                });

                append_triangle(poly_vertices[order[0]], poly_vertices[order[1]], poly_vertices[order[2]]);
                if (cap_reached()) return mesh;
                if (poly_vertices.size() == 4)
                {
                    append_triangle(poly_vertices[order[0]], poly_vertices[order[2]], poly_vertices[order[3]]);
                    if (cap_reached()) return mesh;
                }
            }
        }

        return mesh;
    }

    float voxel_length;
    float sdf_trunc;
    std::unordered_map<Key, Voxel, KeyHash> voxels;
};


int parseFrameIdFromPath(const std::string& path)
{
    const std::string name = std::filesystem::path(path).stem().string();
    std::string digits;
    for (char c : name) {
        if (std::isdigit(static_cast<unsigned char>(c))) {
            digits.push_back(c);
        }
    }
    if (digits.empty()) {
        return -1;
    }
    try {
        return std::stoi(digits);
    } catch (...) {
        return -1;
    }
}

int frameIdFromIntegerTimestamp(double timestamp)
{
    if (!std::isfinite(timestamp)) {
        return -1;
    }
    const double rounded = std::round(timestamp);
    if (std::abs(timestamp - rounded) > 1.0e-6 ||
        rounded < 0.0 ||
        rounded > static_cast<double>(std::numeric_limits<int>::max())) {
        return -1;
    }
    return static_cast<int>(rounded);
}

void saveKeyframeFrameIdMap(
    const std::map<std::size_t, std::shared_ptr<VoxelKeyframe>>& keyframes,
    const std::filesystem::path& path)
{
    std::ofstream out(path);
    if (!out.is_open()) {
        return;
    }
    out << "# ours_kf_id replica_frame_id\n";
    for (const auto& [kf_id, kf] : keyframes) {
        if (!kf) {
            continue;
        }
        int source_frame_id = kf->source_frame_id_;
        if (source_frame_id < 0) {
            source_frame_id = parseFrameIdFromPath(kf->img_filename_);
        }
        if (source_frame_id < 0) {
            source_frame_id = static_cast<int>(kf_id);
        }
        out << kf_id << " " << source_frame_id << "\n";
    }
}

static bool depthMatToMeters(const cv::Mat& depth_in, cv::Mat& depth_meters)
{
    if (depth_in.empty()) {
        return false;
    }

    cv::Mat d = depth_in;
    if (d.channels() > 1) {
        cv::extractChannel(d, d, 0);
    }

    if (d.type() == CV_32FC1) {
        depth_meters = d;
        return true;
    }
    if (d.type() == CV_16UC1) {
        // Handle both common 16-bit conventions:
        // - mm depth (TUM-style): depth_m = raw / 1000
        // - Replica-style uint16 encoding: depth_m = raw / 6553.5
        double max_val = 0.0;
        cv::minMaxLoc(d, nullptr, &max_val);
        const double scale = (max_val > 20000.0) ? (1.0 / 6553.5) : (1.0 / 1000.0);
        d.convertTo(depth_meters, CV_32FC1, scale);
        return true;
    }

    d.convertTo(depth_meters, CV_32FC1);
    return true;
}

static bool loadReplicaDepthFromRgbPath(const std::string& rgb_filename, cv::Mat& depth_meters)
{
    if (rgb_filename.empty()) {
        return false;
    }

    const std::filesystem::path rgb_path(rgb_filename);
    if (!std::filesystem::exists(rgb_path)) {
        return false;
    }

    const std::string name = rgb_path.filename().string();
    if (name.rfind("frame", 0) != 0) {
        return false;
    }

    const std::filesystem::path parent = rgb_path.parent_path();
    const std::string stem = rgb_path.stem().string();   // e.g., frame000123
    const std::string suffix_stem = (stem.size() > 5 ? stem.substr(5) : std::string());
    const std::string suffix_name = name.substr(5);      // e.g., 000123.jpg

    std::vector<std::filesystem::path> candidates;
    candidates.push_back(parent / ("depth" + suffix_name));
    if (!suffix_stem.empty()) {
        candidates.push_back(parent / ("depth" + suffix_stem + ".png"));
        candidates.push_back(parent / ("depth" + suffix_stem + ".exr"));
        candidates.push_back(parent / ("depth" + suffix_stem + ".tiff"));
        candidates.push_back(parent / ("depth" + suffix_stem + ".tif"));
    }

    for (const auto& p : candidates) {
        if (!std::filesystem::exists(p)) {
            continue;
        }
        const cv::Mat depth_raw = cv::imread(p.string(), cv::IMREAD_UNCHANGED);
        if (depth_raw.empty()) {
            continue;
        }
        return depthMatToMeters(depth_raw, depth_meters);
    }

    return false;
}

static bool loadTumDepthFromRgbPath(const std::string& rgb_filename, cv::Mat& depth_meters)
{
    if (rgb_filename.empty()) {
        return false;
    }

    const std::filesystem::path rgb_path(rgb_filename);
    if (!std::filesystem::exists(rgb_path)) {
        return false;
    }

    const std::filesystem::path rgb_dir = rgb_path.parent_path();
    if (rgb_dir.filename() != "rgb") {
        return false;
    }

    const std::filesystem::path dataset_root = rgb_dir.parent_path();
    const std::filesystem::path depth_dir = dataset_root / "depth";
    const std::filesystem::path depth_txt = dataset_root / "depth.txt";
    if (!std::filesystem::exists(depth_dir)) {
        return false;
    }

    const std::string stem = rgb_path.stem().string();
    const std::filesystem::path exact_depth_path = depth_dir / (stem + rgb_path.extension().string());
    if (std::filesystem::exists(exact_depth_path)) {
        const cv::Mat depth_raw = cv::imread(exact_depth_path.string(), cv::IMREAD_UNCHANGED);
        if (!depth_raw.empty()) {
            return depthMatToMeters(depth_raw, depth_meters);
        }
    }

    double rgb_ts = 0.0;
    try {
        rgb_ts = std::stod(stem);
    } catch (...) {
        return false;
    }

    struct TumDepthIndexEntry {
        double timestamp = 0.0;
        std::filesystem::path path;
    };

    static std::mutex s_tum_depth_cache_mutex;
    static std::unordered_map<std::string, std::vector<TumDepthIndexEntry>> s_tum_depth_cache;

    std::vector<TumDepthIndexEntry> depth_index;
    {
        std::lock_guard<std::mutex> lock(s_tum_depth_cache_mutex);
        auto it = s_tum_depth_cache.find(dataset_root.string());
        if (it == s_tum_depth_cache.end()) {
            std::vector<TumDepthIndexEntry> parsed;
            if (std::filesystem::exists(depth_txt)) {
                std::ifstream in(depth_txt);
                std::string line;
                while (std::getline(in, line)) {
                    if (line.empty() || line[0] == '#') {
                        continue;
                    }
                    std::istringstream iss(line);
                    double ts = 0.0;
                    std::string rel_path;
                    if (!(iss >> ts >> rel_path)) {
                        continue;
                    }
                    std::filesystem::path p = dataset_root / rel_path;
                    if (!std::filesystem::exists(p)) {
                        continue;
                    }
                    parsed.push_back({ts, p});
                }
            }
            std::sort(
                parsed.begin(),
                parsed.end(),
                [](const TumDepthIndexEntry& a, const TumDepthIndexEntry& b) {
                    return a.timestamp < b.timestamp;
                });
            it = s_tum_depth_cache.emplace(dataset_root.string(), std::move(parsed)).first;
        }
        depth_index = it->second;
    }

    if (depth_index.empty()) {
        return false;
    }

    auto lb = std::lower_bound(
        depth_index.begin(),
        depth_index.end(),
        rgb_ts,
        [](const TumDepthIndexEntry& e, double t) {
            return e.timestamp < t;
        });

    auto best_it = depth_index.end();
    double best_dt = std::numeric_limits<double>::infinity();
    if (lb != depth_index.end()) {
        best_it = lb;
        best_dt = std::abs(lb->timestamp - rgb_ts);
    }
    if (lb != depth_index.begin()) {
        auto prev = std::prev(lb);
        const double prev_dt = std::abs(prev->timestamp - rgb_ts);
        if (prev_dt < best_dt) {
            best_it = prev;
            best_dt = prev_dt;
        }
    }

    constexpr double kTumMaxDepthAssocDeltaSec = 0.05;
    if (best_it == depth_index.end() || !(best_dt <= kTumMaxDepthAssocDeltaSec)) {
        return false;
    }

    const cv::Mat depth_raw = cv::imread(best_it->path.string(), cv::IMREAD_UNCHANGED);
    if (depth_raw.empty()) {
        return false;
    }
    return depthMatToMeters(depth_raw, depth_meters);
}

static bool getKeyframeDepthMetersForEval(
    const std::shared_ptr<VoxelKeyframe>& pkf,
    int expected_h,
    int expected_w,
    cv::Mat& depth_meters)
{
    if (!pkf) {
        return false;
    }

    if (!pkf->img_auxiliary_undist_.empty()) {
        if (!depthMatToMeters(pkf->img_auxiliary_undist_, depth_meters)) {
            return false;
        }
    } else {
        if (!loadReplicaDepthFromRgbPath(pkf->img_filename_, depth_meters) &&
            !loadTumDepthFromRgbPath(pkf->img_filename_, depth_meters)) {
            return false;
        }
    }

    if (depth_meters.empty()) {
        return false;
    }

    if (depth_meters.rows != expected_h || depth_meters.cols != expected_w) {
        cv::resize(
            depth_meters,
            depth_meters,
            cv::Size(expected_w, expected_h),
            0.0,
            0.0,
            cv::INTER_NEAREST);
    }

    return true;
}

static inline torch::Tensor make_points_tensor_cpu_f32(
    const std::vector<Eigen::Vector3f>& pts)
{
    if (pts.empty()) {
        return torch::Tensor();
    }
    auto out = torch::empty(
        {static_cast<long>(pts.size()), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto acc = out.accessor<float, 2>();
    for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
        acc[i][0] = pts[i].x();
        acc[i][1] = pts[i].y();
        acc[i][2] = pts[i].z();
    }
    return out;
}

static inline torch::Tensor make_single_point_cpu_f32(const Eigen::Vector3d& p)
{
    return torch::tensor(
        {static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z())},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).view({1, 3});
}

static inline torch::Tensor color_u8(uint8_t r, uint8_t g, uint8_t b)
{
    return torch::tensor({r, g, b}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
}

static torch::Tensor normalizeRerunPointColors(torch::Tensor colors)
{
    if (!colors.defined() || colors.numel() == 0) {
        return colors;
    }
    if (colors.max().item<float>() > 1.5f) {
        colors = colors / 255.0f;
    }
    return colors.clamp(0.0f, 1.0f).contiguous();
}

} // namespace

void VoxelMapper::logWholeRunLiveVoxelsToRerun(
    int iteration,
    const torch::Tensor& centers_in,
    const torch::Tensor& sizes_in,
    const torch::Tensor& colors_in,
    bool log_whole_run,
    bool log_svrecon_debug)
{
    if (!rerun_params_.enable_rerun_ ||
        (!(rerun_params_.run_whole_run_ && log_whole_run) &&
         !(rerun_params_.rerun_svrecon_debug_ && log_svrecon_debug)) ||
        !voxel_model_) {
        return;
    }
    if (!centers_in.defined() || !sizes_in.defined() ||
        centers_in.dim() != 2 || centers_in.size(1) != 3) {
        return;
    }

    torch::NoGradGuard no_grad;

    const int64_t N = centers_in.size(0);
    const torch::Device dev = centers_in.device();
    std::vector<std::string> recordings;
    if (rerun_params_.run_whole_run_ && log_whole_run) {
        recordings.emplace_back("whole_run");
    }
    if (rerun_params_.rerun_svrecon_debug_ && log_svrecon_debug) {
        recordings.emplace_back("svrecon_debug");
    }
    torch::Tensor live_colors = colors_in;
    if (N > 0 &&
        (!live_colors.defined() || live_colors.numel() <= 0 ||
         live_colors.dim() != 2 || live_colors.size(0) != N ||
         (live_colors.size(1) != 3 && live_colors.size(1) != 4))) {
        torch::Tensor sh0 = voxel_model_->sh0();
        if (sh0.defined() && sh0.dim() == 2 && sh0.size(0) == N) {
            live_colors = (sh0 * sv::kSHC0 + 0.5f).clamp(0.0f, 1.0f).contiguous();
            // An SVRecon cell has no independent per-voxel opacity. Its rendered
            // alpha depends on ordered SDF samples along a camera ray, so the SDF
            // corner mean is not a meaningful box alpha. Display allocated cells
            // fully opaque; their learned colors still identify the current map.
            torch::Tensor col_rgba = torch::zeros(
                {N, 4}, live_colors.options());
            col_rgba.index_put_(
                {torch::indexing::Slice(), torch::indexing::Slice(0, 3)},
                live_colors);
            col_rgba.index_put_({torch::indexing::Slice(), 3}, 1.0f);
            live_colors = col_rgba.contiguous();
        }
    }
    torch::Tensor sizes = sizes_in;
    if (sizes.dim() == 1) {
        sizes = sizes.view({N, 1});
    } else if (sizes.dim() == 2 && sizes.size(1) == 1) {
        // ok
    } else {
        sizes = sizes.reshape({N, 1});
    }
    torch::Tensor levels = voxel_model_->octLevel();
    if (!levels.defined() || levels.numel() != N) {
        levels = torch::full(
            {N, 1},
            voxel_model_->insertionOctreeLevel(),
            torch::TensorOptions().dtype(torch::kInt32).device(dev));
    } else {
        levels = levels.to(dev).reshape({N, 1}).contiguous();
    }
    torch::Tensor grid_origin =
        (voxel_model_->SceneCenter() - 0.5f * voxel_model_->SceneExtent())
            .to(dev)
            .to(torch::kFloat32)
            .reshape({3})
            .contiguous();

    torch::Tensor orb_mask =
        voxel_utils::normalizeBoolMaskOrZeros(voxel_model_->orbVoxelMask(), N, dev);
    torch::Tensor inactive_geo_mask =
        voxel_utils::normalizeBoolMaskOrZeros(voxel_model_->inactiveGeoVoxelMask(), N, dev);
    torch::Tensor rgbd_fill_mask =
        voxel_utils::normalizeBoolMaskOrZeros(voxel_model_->rgbdFillRenderHolesVoxelMask(), N, dev);
    torch::Tensor monocular_rendered_depth_mask =
        voxel_utils::normalizeBoolMaskOrZeros(
            voxel_model_->monocularRenderedDepthVoxelMask(), N, dev);
    torch::Tensor monocular_mvs_mask =
        voxel_utils::normalizeBoolMaskOrZeros(
            voxel_model_->monocularMvsVoxelMask(), N, dev);
    torch::Tensor monocular_omnidata_mask =
        voxel_utils::normalizeBoolMaskOrZeros(
            voxel_model_->monocularOmnidataVoxelMask(), N, dev);
    torch::Tensor active_mask =
        voxel_utils::normalizeBoolMaskOrZeros(voxel_model_->activeRenderableMask(), N, dev);
    torch::Tensor active_source_count =
        (orb_mask.to(torch::kInt32) +
         inactive_geo_mask.to(torch::kInt32) +
         rgbd_fill_mask.to(torch::kInt32) +
         monocular_rendered_depth_mask.to(torch::kInt32) +
         monocular_mvs_mask.to(torch::kInt32) +
         monocular_omnidata_mask.to(torch::kInt32))
            .masked_select(active_mask);
    TORCH_CHECK(
        active_source_count.numel() == active_mask.sum().item<int64_t>() &&
            (active_source_count == 1).all().item<bool>(),
        "SVRecon Rerun provenance invariant failed: every active voxel must "
        "have exactly one source (ORB, inactive geometry, RGB-D, or "
        "monocular depth densification).");
    torch::Tensor orb_live_mask =
        (orb_mask & active_mask).to(torch::kBool);
    torch::Tensor rgbd_fill_live_mask =
        (rgbd_fill_mask & active_mask).to(torch::kBool);
    torch::Tensor inactive_geo_live_mask =
        (inactive_geo_mask & active_mask).to(torch::kBool);
    torch::Tensor monocular_rendered_depth_live_mask =
        (monocular_rendered_depth_mask & active_mask).to(torch::kBool);
    torch::Tensor monocular_mvs_live_mask =
        (monocular_mvs_mask & active_mask).to(torch::kBool);
    torch::Tensor monocular_omnidata_live_mask =
        (monocular_omnidata_mask & active_mask).to(torch::kBool);
    auto colors_for_indices =
        [&](const torch::Tensor& idx_in,
            const std::array<float, 4>& fallback_rgba) -> torch::Tensor
    {
        const int64_t K = idx_in.defined() ? idx_in.numel() : 0;
        torch::Tensor fallback = torch::zeros(
            {K, 4},
            torch::TensorOptions().dtype(torch::kFloat32).device(dev));
        if (K > 0) {
            fallback.index_put_({torch::indexing::Slice(), 0}, fallback_rgba[0]);
            fallback.index_put_({torch::indexing::Slice(), 1}, fallback_rgba[1]);
            fallback.index_put_({torch::indexing::Slice(), 2}, fallback_rgba[2]);
            fallback.index_put_({torch::indexing::Slice(), 3}, fallback_rgba[3]);
        }
        if (K <= 0 || !live_colors.defined() || live_colors.numel() <= 0 ||
            live_colors.dim() != 2 || live_colors.size(0) != N ||
            (live_colors.size(1) != 3 && live_colors.size(1) != 4)) {
            return fallback.contiguous();
        }
        torch::Tensor selected =
            live_colors.index_select(0, idx_in.to(live_colors.device()).to(torch::kLong))
                .to(dev)
                .to(torch::kFloat32)
                .contiguous();
        if (selected.size(1) == 4) {
            return selected;
        }
        torch::Tensor rgba = fallback.clone();
        rgba.index_put_(
            {torch::indexing::Slice(), torch::indexing::Slice(0, 3)},
            selected.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)}));
        return rgba.contiguous();
    };

    auto log_subset =
        [&](const torch::Tensor& mask_in,
            const std::string& entity_path,
            const std::array<float, 4>& fallback_rgba,
            bool force_log)
    {
        torch::Tensor mask = voxel_utils::normalizeBoolMaskOrZeros(mask_in, N, dev);
        torch::Tensor idx = mask.nonzero().squeeze(1);
        if ((!idx.defined() || idx.numel() <= 0) && !force_log) {
            return;
        }

        torch::Tensor centers_sel;
        torch::Tensor sizes_sel;
        torch::Tensor levels_sel;
        torch::Tensor colors_sel;
        if (!idx.defined() || idx.numel() <= 0) {
            centers_sel = torch::empty(
                {0, 3},
                torch::TensorOptions().dtype(torch::kFloat32).device(dev));
            sizes_sel = torch::empty(
                {0, 1},
                torch::TensorOptions().dtype(torch::kFloat32).device(dev));
            levels_sel = torch::empty(
                {0, 1},
                torch::TensorOptions().dtype(torch::kInt32).device(dev));
            colors_sel = torch::empty(
                {0, 4},
                torch::TensorOptions().dtype(torch::kFloat32).device(dev));
        } else {
            torch::Tensor idx_dev = idx.to(dev).to(torch::kLong);
            centers_sel = centers_in.index_select(0, idx_dev).contiguous();
            sizes_sel = sizes.index_select(0, idx_dev).contiguous();
            levels_sel = levels.index_select(0, idx_dev).contiguous();
            colors_sel = colors_for_indices(idx_dev, fallback_rgba);
        }

        for (const std::string& recording : recordings) {
            sv::RerunVisualizerBridge::instance().visualizeDebugVoxelGridMap(
                recording,
                centers_sel,
                sizes_sel,
                levels_sel,
                colors_sel,
                grid_origin,
                iteration,
                entity_path,
                1.0f);
        }
    };

    auto log_used_source =
        [&](const torch::Tensor& mask,
            const std::string& entity_path,
            bool& was_logged)
    {
        const bool has_voxels =
            mask.defined() && mask.numel() == N &&
            mask.any().item<bool>();
        if (!has_voxels && !was_logged) {
            return;
        }
        log_subset(
            mask,
            entity_path,
            {0.75f, 0.75f, 0.75f, 1.0f},
            true);
        was_logged = was_logged || has_voxels;
    };

    log_used_source(
        orb_live_mask,
        "world/svrecon/source/orb",
        rerun_state_.whole_run_logged_orb_source_);
    log_used_source(
        inactive_geo_live_mask,
        "world/svrecon/source/inactive",
        rerun_state_.whole_run_logged_inactive_source_);
    const std::string rgbd_source_entity = rgbd_tsdf_evidence_
        ? "world/svrecon/source/rgbd_tsdf_promoted"
        : "world/svrecon/source/rgbd_fill_render_holes";
    log_used_source(
        rgbd_fill_live_mask,
        rgbd_source_entity,
        rerun_state_.whole_run_logged_rgbd_source_);
    log_used_source(
        monocular_rendered_depth_live_mask,
        "world/svrecon/source/monocular_rendered_depth",
        rerun_state_.whole_run_logged_monocular_rendered_depth_source_);
    log_used_source(
        monocular_mvs_live_mask,
        "world/svrecon/source/monocular_mvs",
        rerun_state_.whole_run_logged_monocular_mvs_source_);
    log_used_source(
        monocular_omnidata_live_mask,
        "world/svrecon/source/monocular_omnidata",
        rerun_state_.whole_run_logged_monocular_omnidata_source_);
}

void VoxelMapper::logSvreconDebugVoxelMaskToRerun(
    int iteration,
    const torch::Tensor& mask_in,
    const std::string& entity_path)
{
    if (!rerun_params_.enable_rerun_ ||
        !rerun_params_.rerun_svrecon_debug_ || !voxel_model_) {
        return;
    }

    torch::NoGradGuard no_grad;
    torch::Tensor centers = voxel_model_->voxCenter();
    torch::Tensor sizes = voxel_model_->voxSize();
    if (!centers.defined() || !sizes.defined() ||
        centers.dim() != 2 || centers.size(1) != 3) {
        return;
    }

    const int64_t N = centers.size(0);
    const torch::Device dev = centers.device();
    torch::Tensor mask = voxel_utils::normalizeBoolMaskOrZeros(mask_in, N, dev);
    torch::Tensor idx = mask.nonzero().squeeze(1);
    const int64_t K = idx.defined() ? idx.numel() : 0;

    torch::Tensor selected_centers;
    torch::Tensor selected_sizes;
    torch::Tensor selected_levels;
    torch::Tensor selected_colors;
    if (K > 0) {
        torch::Tensor idx_dev = idx.to(dev).to(torch::kLong);
        selected_centers = centers.index_select(0, idx_dev).contiguous();
        if (sizes.dim() == 1) {
            sizes = sizes.view({N, 1});
        } else if (sizes.dim() != 2 || sizes.size(1) != 1) {
            sizes = sizes.reshape({N, 1});
        }
        selected_sizes = sizes.index_select(0, idx_dev).contiguous();
        torch::Tensor levels = voxel_model_->octLevel();
        if (levels.defined() && levels.numel() == N) {
            selected_levels =
                levels.to(dev).reshape({N, 1}).index_select(0, idx_dev).contiguous();
        } else {
            selected_levels = torch::full(
                {K, 1},
                voxel_model_->insertionOctreeLevel(),
                torch::TensorOptions().dtype(torch::kInt32).device(dev));
        }
        torch::Tensor sh0 = voxel_model_->sh0();
        if (sh0.defined() && sh0.dim() == 2 && sh0.size(0) == N) {
            torch::Tensor rgb =
                (sh0.index_select(0, idx_dev) * sv::kSHC0 + 0.5f)
                    .clamp(0.0f, 1.0f);
            selected_colors = torch::ones(
                {K, 4}, rgb.options());
            selected_colors.index_put_(
                {torch::indexing::Slice(), torch::indexing::Slice(0, 3)},
                rgb);
            selected_colors.index_put_({torch::indexing::Slice(), 3}, 0.8f);
        } else {
            selected_colors = torch::full(
                {K, 4},
                0.75f,
                torch::TensorOptions().dtype(torch::kFloat32).device(dev));
            selected_colors.index_put_({torch::indexing::Slice(), 3}, 0.8f);
        }
    } else {
        selected_centers = torch::empty(
            {0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
        selected_sizes = torch::empty(
            {0, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
        selected_levels = torch::empty(
            {0, 1}, torch::TensorOptions().dtype(torch::kInt32).device(dev));
        selected_colors = torch::empty(
            {0, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    }

    auto& bridge = sv::RerunVisualizerBridge::instance();
    torch::Tensor grid_origin =
        (voxel_model_->SceneCenter() - 0.5f * voxel_model_->SceneExtent())
            .to(dev)
            .to(torch::kFloat32)
            .reshape({3})
            .contiguous();
    bridge.visualizeDebugVoxelGridMap(
        "svrecon_debug",
        selected_centers,
        selected_sizes,
        selected_levels,
        selected_colors,
        grid_origin,
        iteration,
        entity_path,
        0.8f);
}

void VoxelMapper::appendWholeRunPrunedVoxels(
    int iteration,
    const torch::Tensor& centers_in,
    const torch::Tensor& sizes_in,
    const torch::Tensor& levels_in,
    const torch::Tensor& colors_in,
    const torch::Tensor& pruned_by_sdf_in,
    const torch::Tensor& pruned_by_surface_views_in,
    const torch::Tensor& pruned_by_near_camera_in,
    const torch::Tensor& pruned_by_far_in,
    const torch::Tensor& pruned_by_mvs_free_space_in,
    const torch::Tensor& pruned_by_final_refinement_in)
{
    if (!rerun_params_.enable_rerun_ ||
        (!rerun_params_.run_whole_run_ &&
         !rerun_params_.rerun_svrecon_debug_)) {
        return;
    }
    if (!centers_in.defined() || !sizes_in.defined() ||
        centers_in.dim() != 2 || centers_in.size(1) != 3 ||
        centers_in.size(0) <= 0) {
        return;
    }

    torch::NoGradGuard no_grad;

    const int64_t K = centers_in.size(0);
    std::vector<std::string> recordings;
    if (rerun_params_.run_whole_run_) {
        recordings.emplace_back("whole_run");
    }
    if (rerun_params_.rerun_svrecon_debug_) {
        recordings.emplace_back("svrecon_debug");
    }
    torch::Tensor centers =
        centers_in.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor sizes = sizes_in.detach().to(torch::kCPU).to(torch::kFloat32);
    if (sizes.dim() == 1) {
        sizes = sizes.view({K, 1});
    } else if (sizes.dim() == 2 && sizes.size(1) == 1) {
        // ok
    } else {
        sizes = sizes.reshape({K, 1});
    }
    sizes = sizes.contiguous();

    torch::Tensor levels;
    if (levels_in.defined() && levels_in.numel() == K) {
        levels =
            levels_in.detach().to(torch::kCPU).to(torch::kInt32).reshape({K, 1}).contiguous();
    } else {
        levels = torch::full(
            {K, 1},
            voxel_model_ ? voxel_model_->insertionOctreeLevel() : 0,
            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    }

    torch::Tensor colors;
    if (colors_in.defined() && colors_in.dim() == 2 &&
        colors_in.size(0) == K &&
        (colors_in.size(1) == 3 || colors_in.size(1) == 4)) {
        colors =
            colors_in.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
        if (colors.max().item<float>() > 1.5f) {
            colors = colors / 255.0f;
        }
        colors = colors.clamp(0.0f, 1.0f);
        if (colors.size(1) == 3) {
            torch::Tensor rgba = torch::ones(
                {K, 4}, colors.options());
            rgba.index_put_(
                {torch::indexing::Slice(), torch::indexing::Slice(0, 3)},
                colors);
            rgba.index_put_({torch::indexing::Slice(), 3}, 1.0f);
            colors = rgba.contiguous();
        }
    } else {
        colors = torch::full(
            {K, 4},
            0.75f,
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        colors.index_put_({torch::indexing::Slice(), 3}, 1.0f);
    }

    torch::Tensor sdf_mask = torch::zeros(
        {K},
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    if (pruned_by_sdf_in.defined() && pruned_by_sdf_in.numel() == K) {
        sdf_mask =
            pruned_by_sdf_in.detach()
                .to(torch::kCPU)
                .to(torch::kBool)
                .contiguous()
                .view({K});
    }
    torch::Tensor surface_views_mask = torch::zeros(
        {K},
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    if (pruned_by_surface_views_in.defined() &&
        pruned_by_surface_views_in.numel() == K) {
        surface_views_mask =
            pruned_by_surface_views_in.detach()
                .to(torch::kCPU)
                .to(torch::kBool)
                .contiguous()
                .view({K});
    }
    torch::Tensor near_camera_mask = torch::zeros(
        {K},
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    if (pruned_by_near_camera_in.defined() &&
        pruned_by_near_camera_in.numel() == K) {
        near_camera_mask =
            pruned_by_near_camera_in.detach()
                .to(torch::kCPU)
                .to(torch::kBool)
                .contiguous()
                .view({K});
    }
    torch::Tensor far_mask = torch::zeros(
        {K},
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    if (pruned_by_far_in.defined() &&
        pruned_by_far_in.numel() == K) {
        far_mask =
            pruned_by_far_in.detach()
                .to(torch::kCPU)
                .to(torch::kBool)
                .contiguous()
                .view({K});
    }
    torch::Tensor mvs_free_space_mask = torch::zeros(
        {K},
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    if (pruned_by_mvs_free_space_in.defined() &&
        pruned_by_mvs_free_space_in.numel() == K) {
        mvs_free_space_mask =
            pruned_by_mvs_free_space_in.detach()
                .to(torch::kCPU)
                .to(torch::kBool)
                .contiguous()
                .view({K});
    }
    torch::Tensor final_refinement_mask = torch::zeros(
        {K},
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    if (pruned_by_final_refinement_in.defined() &&
        pruned_by_final_refinement_in.numel() == K) {
        final_refinement_mask =
            pruned_by_final_refinement_in.detach()
                .to(torch::kCPU)
                .to(torch::kBool)
                .contiguous()
                .view({K});
    }
    // Keep concrete prune causes disjoint.
    surface_views_mask =
        (surface_views_mask & (~sdf_mask)).to(torch::kBool);
    near_camera_mask =
        (near_camera_mask & (~sdf_mask) & (~surface_views_mask))
            .to(torch::kBool);
    far_mask =
        (far_mask & (~sdf_mask) & (~surface_views_mask) &
         (~near_camera_mask))
            .to(torch::kBool);
    mvs_free_space_mask =
        (mvs_free_space_mask & (~sdf_mask) & (~surface_views_mask) &
         (~near_camera_mask) & (~far_mask))
            .to(torch::kBool);
    final_refinement_mask =
        (final_refinement_mask & (~sdf_mask) & (~surface_views_mask) &
         (~near_camera_mask) & (~far_mask) &
         (~mvs_free_space_mask))
            .to(torch::kBool);

    torch::Tensor grid_origin;
    if (voxel_model_) {
        grid_origin =
            (voxel_model_->SceneCenter() -
             0.5f * voxel_model_->SceneExtent())
                .detach()
                .to(torch::kCPU)
                .to(torch::kFloat32)
                .reshape({3})
                .contiguous();
    }

    auto append_source =
        [&](const torch::Tensor& mask,
            torch::Tensor& centers_accum,
            torch::Tensor& sizes_accum,
            torch::Tensor& levels_accum,
            torch::Tensor& colors_accum,
            const std::string& entity_path)
    {
        torch::Tensor idx = mask.nonzero().squeeze(1);
        if (!idx.defined() || idx.numel() <= 0) {
            return;
        }
        torch::Tensor centers_sel = centers.index_select(0, idx).contiguous();
        torch::Tensor sizes_sel = sizes.index_select(0, idx).contiguous();
        torch::Tensor levels_sel = levels.index_select(0, idx).contiguous();
        torch::Tensor colors_sel = colors.index_select(0, idx).contiguous();

        if (!centers_accum.defined()) {
            centers_accum = centers_sel;
            sizes_accum = sizes_sel;
            levels_accum = levels_sel;
            colors_accum = colors_sel;
        } else {
            centers_accum = torch::cat({centers_accum, centers_sel}, 0).contiguous();
            sizes_accum = torch::cat({sizes_accum, sizes_sel}, 0).contiguous();
            levels_accum = torch::cat({levels_accum, levels_sel}, 0).contiguous();
            colors_accum = torch::cat({colors_accum, colors_sel}, 0).contiguous();
        }

        for (const std::string& recording : recordings) {
            if (grid_origin.defined()) {
                sv::RerunVisualizerBridge::instance().visualizeDebugVoxelGridMap(
                    recording,
                    centers_accum,
                    sizes_accum,
                    levels_accum,
                    colors_accum,
                    grid_origin,
                    iteration,
                    entity_path,
                    1.0f);
            }
        }
    };

    if (!rerun_state_.whole_run_pruned_centers_accum_.defined()) {
        rerun_state_.whole_run_pruned_centers_accum_ = centers;
        rerun_state_.whole_run_pruned_sizes_accum_ = sizes;
        rerun_state_.whole_run_pruned_levels_accum_ = levels;
        rerun_state_.whole_run_pruned_colors_accum_ = colors;
    } else {
        rerun_state_.whole_run_pruned_centers_accum_ =
            torch::cat({rerun_state_.whole_run_pruned_centers_accum_, centers}, 0).contiguous();
        rerun_state_.whole_run_pruned_sizes_accum_ =
            torch::cat({rerun_state_.whole_run_pruned_sizes_accum_, sizes}, 0).contiguous();
        rerun_state_.whole_run_pruned_levels_accum_ =
            torch::cat({rerun_state_.whole_run_pruned_levels_accum_, levels}, 0).contiguous();
        rerun_state_.whole_run_pruned_colors_accum_ =
            torch::cat({rerun_state_.whole_run_pruned_colors_accum_, colors}, 0).contiguous();
    }

    for (const std::string& recording : recordings) {
        if (grid_origin.defined()) {
            sv::RerunVisualizerBridge::instance().visualizeDebugVoxelGridMap(
                recording,
                rerun_state_.whole_run_pruned_centers_accum_,
                rerun_state_.whole_run_pruned_sizes_accum_,
                rerun_state_.whole_run_pruned_levels_accum_,
                rerun_state_.whole_run_pruned_colors_accum_,
                grid_origin,
                iteration,
                "world/pruned_voxels",
                1.0f);
        }
    }

    append_source(
        sdf_mask,
        rerun_state_.whole_run_pruned_sdf_centers_accum_,
        rerun_state_.whole_run_pruned_sdf_sizes_accum_,
        rerun_state_.whole_run_pruned_sdf_levels_accum_,
        rerun_state_.whole_run_pruned_sdf_colors_accum_,
        "world/pruned_voxels/source/svrecon_sdf");
    append_source(
        surface_views_mask,
        rerun_state_.whole_run_pruned_surface_views_centers_accum_,
        rerun_state_.whole_run_pruned_surface_views_sizes_accum_,
        rerun_state_.whole_run_pruned_surface_views_levels_accum_,
        rerun_state_.whole_run_pruned_surface_views_colors_accum_,
        "world/pruned_voxels/source/surface_views");
    append_source(
        near_camera_mask,
        rerun_state_.whole_run_pruned_near_camera_centers_accum_,
        rerun_state_.whole_run_pruned_near_camera_sizes_accum_,
        rerun_state_.whole_run_pruned_near_camera_levels_accum_,
        rerun_state_.whole_run_pruned_near_camera_colors_accum_,
        "world/pruned_voxels/source/near_camera");
    append_source(
        far_mask,
        rerun_state_.whole_run_pruned_far_centers_accum_,
        rerun_state_.whole_run_pruned_far_sizes_accum_,
        rerun_state_.whole_run_pruned_far_levels_accum_,
        rerun_state_.whole_run_pruned_far_colors_accum_,
        "world/pruned_voxels/source/far");
    append_source(
        mvs_free_space_mask,
        rerun_state_.whole_run_pruned_mvs_free_space_centers_accum_,
        rerun_state_.whole_run_pruned_mvs_free_space_sizes_accum_,
        rerun_state_.whole_run_pruned_mvs_free_space_levels_accum_,
        rerun_state_.whole_run_pruned_mvs_free_space_colors_accum_,
        "world/pruned_voxels/source/mvs_free_space");
    append_source(
        final_refinement_mask,
        rerun_state_.whole_run_pruned_final_refinement_centers_accum_,
        rerun_state_.whole_run_pruned_final_refinement_sizes_accum_,
        rerun_state_.whole_run_pruned_final_refinement_levels_accum_,
        rerun_state_.whole_run_pruned_final_refinement_colors_accum_,
        "world/pruned_voxels/source/final_refinement");
}

void VoxelMapper::logCurrentOrbMapPointsToReconstructionRerun(int iteration)
{
    if (!rerun_params_.enable_rerun_ ||
        (!rerun_params_.rerun_reconstruction_mesh_ &&
         !rerun_params_.rerun_svrecon_debug_ &&
         !rerun_params_.run_whole_run_) ||
        !mpSLAM || !mpSLAM->getAtlas()) {
        return;
    }

    auto* pMap = mpSLAM->getAtlas()->GetCurrentMap();
    if (!pMap) {
        return;
    }

    {
        std::unique_lock<std::mutex> lock_map(pMap->mMutexMapUpdate);
        const std::vector<ORB_SLAM3::MapPoint*> map_points = pMap->GetAllMapPoints();
        for (auto* pMP : map_points) {
            if (!pMP || pMP->isBad()) {
                continue;
            }
            const Eigen::Vector3f pos = pMP->GetWorldPos();
            const Eigen::Vector3f color = pMP->GetColorRGB();
            if (!pos.allFinite() || !color.allFinite()) {
                continue;
            }
            rerun_state_.whole_run_orb_points_by_id_[
                static_cast<std::uint64_t>(pMP->mnId)] = {
                    pos.x(), pos.y(), pos.z(),
                    color.x(), color.y(), color.z()};
        }
    }

    std::vector<std::uint64_t> point_ids;
    point_ids.reserve(rerun_state_.whole_run_orb_points_by_id_.size());
    for (const auto& item : rerun_state_.whole_run_orb_points_by_id_) {
        point_ids.push_back(item.first);
    }
    std::sort(point_ids.begin(), point_ids.end());

    std::vector<float> pts;
    std::vector<float> cols;
    pts.reserve(point_ids.size() * 3);
    cols.reserve(point_ids.size() * 3);
    for (const std::uint64_t point_id : point_ids) {
        const auto& value =
            rerun_state_.whole_run_orb_points_by_id_.at(point_id);
        pts.insert(pts.end(), {value[0], value[1], value[2]});
        cols.insert(cols.end(), {value[3], value[4], value[5]});
    }

    const int64_t n_points = static_cast<int64_t>(pts.size() / 3);
    if (n_points == 0) {
        return;
    }
    auto points = torch::from_blob(
        pts.data(),
        {n_points, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
    auto colors = torch::from_blob(
        cols.data(),
        {n_points, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
    colors = normalizeRerunPointColors(colors);

    if (rerun_params_.rerun_reconstruction_mesh_) {
        sv::RerunVisualizerBridge::instance().visualizeDebugPoints3D(
            "reconstruction_mesh",
            points,
            colors,
            iteration,
            "world/orb/map_points",
            0.015f);
    }
    if (rerun_params_.rerun_svrecon_debug_) {
        sv::RerunVisualizerBridge::instance().visualizeDebugPoints3D(
            "svrecon_debug",
            points,
            colors,
            iteration,
            "world/orb/map_points",
            0.015f);
    }
    if (rerun_params_.run_whole_run_) {
        sv::RerunVisualizerBridge::instance().visualizeDebugPoints3D(
            "whole_run",
            points,
            colors,
            iteration,
            "world/orb/map_points",
            0.015f);
    }
}

void VoxelMapper::logCurrentOrbKeyframePosesToReconstructionRerun(int iteration)
{
    if (!rerun_params_.enable_rerun_ ||
        (!rerun_params_.rerun_reconstruction_mesh_ &&
         !rerun_params_.rerun_svrecon_debug_ &&
         !rerun_params_.run_whole_run_) ||
        !mpSLAM || !mpSLAM->getAtlas()) {
        return;
    }

    auto* pMap = mpSLAM->getAtlas()->GetCurrentMap();
    if (!pMap) {
        return;
    }

    std::vector<std::pair<unsigned long, Eigen::Matrix4f>> poses;
    {
        std::unique_lock<std::mutex> lock_map(pMap->mMutexMapUpdate);
        const std::vector<ORB_SLAM3::KeyFrame*> keyframes = pMap->GetAllKeyFrames();
        poses.reserve(keyframes.size());
        for (auto* pKF : keyframes) {
            if (!pKF || pKF->isBad()) {
                continue;
            }
            poses.emplace_back(pKF->mnId, pKF->GetPoseInverse().matrix());
        }
    }

    std::sort(
        poses.begin(),
        poses.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    const unsigned long begin =
        static_cast<unsigned long>(std::max(0, rerun_params_.rerun_keyframe_start_));
    const unsigned long end = rerun_params_.rerun_max_keyframes_ > 0
        ? begin + static_cast<unsigned long>(rerun_params_.rerun_max_keyframes_)
        : std::numeric_limits<unsigned long>::max();

    for (const auto& [kf_id, T_W_C] : poses) {
        if (kf_id < begin || kf_id >= end) {
            continue;
        }
        const auto previous = rerun_reconstruction_last_orb_poses_.find(kf_id);
        if (previous != rerun_reconstruction_last_orb_poses_.end() &&
            previous->second.isApprox(T_W_C, 1.0e-6f)) {
            continue;
        }
        if (rerun_params_.rerun_reconstruction_mesh_) {
            sv::RerunVisualizerBridge::instance().visualizeDebugCameraPose(
                "reconstruction_mesh",
                T_W_C,
                iteration,
                static_cast<int>(kf_id));
        }
        if (rerun_params_.rerun_svrecon_debug_) {
            sv::RerunVisualizerBridge::instance().visualizeDebugCameraPose(
                "svrecon_debug",
                T_W_C,
                iteration,
                static_cast<int>(kf_id));
        }
        if (rerun_params_.run_whole_run_) {
            sv::RerunVisualizerBridge::instance().visualizeDebugCameraPose(
                "whole_run",
                T_W_C,
                iteration,
                static_cast<int>(kf_id));
        }
        rerun_reconstruction_last_orb_poses_[kf_id] = T_W_C;
    }
}

void VoxelMapper::logReconstructionMeshToRerun(int iteration)
{
    if (!rerun_params_.enable_rerun_ || !rerun_params_.rerun_reconstruction_mesh_ ||
        !voxel_model_ || !scene_) {
        return;
    }
    if (rerun_params_.rerun_reconstruction_mesh_interval_ <= 0 ||
        (iteration % rerun_params_.rerun_reconstruction_mesh_interval_) != 0) {
        return;
    }

    torch::NoGradGuard no_grad;

    const auto& keyframes = scene_->keyframes();
    if (keyframes.empty()) return;

    auto centers_cpu = voxel_model_->voxCenter().detach().to(torch::kCPU).contiguous();
    if (!centers_cpu.defined() || centers_cpu.numel() == 0) return;

    const float current_sdf_voxel = std::max(1.0e-6f, sdfMetricVoxelSize());
    const float voxel_length =
        std::max(current_sdf_voxel, std::max(1.0e-6f, sdf_params_.sdf_voxel_size_m_));
    const float sdf_trunc =
        std::max(1.0e-6f, sdf_params_.sdf_init_trunc_vox_ * voxel_length);
    const float depth_trunc =
        (sdf_params_.sdf_init_max_depth_m_ > 0.0f)
            ? sdf_params_.sdf_init_max_depth_m_
            : kCommonEvalDepthTrunc;

    SparseTsdfVolume volume(voxel_length, sdf_trunc);

    bool froze_geo = false;
    auto unfreeze_geo = [&]() {
        if (froze_geo) {
            voxel_model_->unfreezeVoxGeo();
            froze_geo = false;
        }
    };
    try
    {
        voxel_model_->freezeVoxGeo();
        froze_geo = true;

        for (const auto& [kfid, pkf] : keyframes)
        {
            if (!pkf) continue;
            if (pkf->image_width_ <= 0 || pkf->image_height_ <= 0 || pkf->intr_.size() < 4) continue;
            const int image_height = pkf->image_height_;
            const int image_width = pkf->image_width_;

            const sv::MiniCam cam = pkf->toMiniCam(pkf->image_height_, pkf->image_width_);
            std::unordered_map<std::string, torch::Tensor> render_pkg =
                voxel_model_->render(
                    cam,
                    image_height,
                    image_width,
                    torch::Tensor(),
                    nullptr,
                    false,
                    1.0f,
                    true,
                    false,
                    false,
                    false,
                    false,
                    sv::RenderOpts{});
            if (render_pkg.empty()) continue;

            auto normalize_color_chw = [&](torch::Tensor t) {
                t = t.detach().contiguous();
                if (t.dim() == 4 && t.size(0) == 1) t = t.squeeze(0);
                if (t.dim() != 3) {
                    throw std::runtime_error("logReconstructionMeshToRerun: unexpected color tensor rank");
                }
                if (t.size(0) == 3 && t.size(1) == image_height && t.size(2) == image_width) {
                    return t.to(torch::kFloat32).contiguous();
                }
                if (t.size(0) == 3 && t.size(1) == image_width && t.size(2) == image_height) {
                    return t.transpose(1, 2).to(torch::kFloat32).contiguous();
                }
                if (t.size(0) == image_height && t.size(1) == image_width && t.size(2) == 3) {
                    return t.permute({2, 0, 1}).to(torch::kFloat32).contiguous();
                }
                if (t.size(0) == image_width && t.size(1) == image_height && t.size(2) == 3) {
                    return t.permute({2, 1, 0}).to(torch::kFloat32).contiguous();
                }
                std::ostringstream oss;
                oss << "logReconstructionMeshToRerun: unsupported color shape " << t.sizes();
                throw std::runtime_error(oss.str());
            };
            auto normalize_hw = [&](torch::Tensor t, const char* name) {
                t = t.detach().to(torch::kFloat32).contiguous();
                if (t.dim() == 3 && t.size(0) == 1) t = t.squeeze(0);
                if (t.dim() != 2) {
                    std::ostringstream oss;
                    oss << "logReconstructionMeshToRerun: unexpected " << name << " tensor rank";
                    throw std::runtime_error(oss.str());
                }
                if (t.size(0) == image_height && t.size(1) == image_width) {
                    return t.contiguous();
                }
                if (t.size(0) == image_width && t.size(1) == image_height) {
                    return t.transpose(0, 1).contiguous();
                }
                std::ostringstream oss;
                oss << "logReconstructionMeshToRerun: unsupported " << name << " shape " << t.sizes();
                throw std::runtime_error(oss.str());
            };

            auto it_color = render_pkg.find("color");
            auto it_depth = render_pkg.find("depth");
            if (it_color == render_pkg.end() || it_depth == render_pkg.end() ||
                !it_color->second.defined() || !it_depth->second.defined()) {
                continue;
            }

            torch::Tensor rendered_color = normalize_color_chw(it_color->second);
            torch::Tensor depth_pkg = it_depth->second.detach().contiguous();
            torch::Tensor rendered_depth;
            if (depth_pkg.dim() == 3 && depth_pkg.size(0) >= 3) {
                rendered_depth = normalize_hw(depth_pkg.index({2}), "depth[2]");
            } else {
                rendered_depth = normalize_hw(depth_pkg, "depth");
            }

            const auto mask_it = undistort_mask_.find(pkf->camera_id_);
            if (mask_it != undistort_mask_.end())
            {
                auto mask = mask_it->second;
                if (mask.dim() == 3) mask = mask.index({0});
                mask = normalize_hw(mask.to(rendered_depth.device()), "undistort_mask");
                rendered_color = rendered_color * mask.unsqueeze(0);
                rendered_depth = rendered_depth * mask;
            }

            auto color_cpu = rendered_color.clamp(0, 1)
                .mul(255.0f)
                .to(torch::kUInt8)
                .permute({1, 2, 0})
                .contiguous()
                .to(torch::kCPU);
            auto depth_cpu = rendered_depth
                .to(torch::kFloat32)
                .contiguous()
                .to(torch::kCPU);

            cv::Mat color_mat(
                image_height, image_width, CV_8UC3, color_cpu.data_ptr<uint8_t>());
            cv::Mat depth_mat(
                image_height, image_width, CV_32FC1, depth_cpu.data_ptr<float>());

            // The live Rerun mesh is a preview. Downsample before CPU TSDF
            // fusion; the shutdown evaluation mesh keeps its full-resolution
            // extraction path.
            constexpr int kLivePreviewDownsample = 2;
            const int fusion_width = std::max(1, image_width / kLivePreviewDownsample);
            const int fusion_height = std::max(1, image_height / kLivePreviewDownsample);
            cv::Mat color_for_fusion;
            cv::Mat depth_for_fusion;
            cv::resize(
                color_mat,
                color_for_fusion,
                cv::Size(fusion_width, fusion_height),
                0.0,
                0.0,
                cv::INTER_AREA);
            cv::resize(
                depth_mat,
                depth_for_fusion,
                cv::Size(fusion_width, fusion_height),
                0.0,
                0.0,
                cv::INTER_NEAREST);
            cv::Mat gt_depth_meters;
            if (getKeyframeDepthMetersForEval(
                    pkf, fusion_height, fusion_width, gt_depth_meters))
            {
                for (int y = 0; y < fusion_height; ++y)
                {
                    const float* gt_ptr = gt_depth_meters.ptr<float>(y);
                    float* depth_ptr = depth_for_fusion.ptr<float>(y);
                    for (int x = 0; x < fusion_width; ++x)
                    {
                        const float gt_depth = gt_ptr[x];
                        if (!std::isfinite(gt_depth) || gt_depth <= 0.0f) {
                            depth_ptr[x] = 0.0f;
                        }
                    }
                }
            }

            cv::Mat filtered_depth = filterDepthOutliersLikeGaussianSlam(depth_for_fusion);
            std::vector<float> fusion_intr = pkf->intr_;
            const float scale_x =
                static_cast<float>(fusion_width) / static_cast<float>(image_width);
            const float scale_y =
                static_cast<float>(fusion_height) / static_cast<float>(image_height);
            fusion_intr[0] *= scale_x;
            fusion_intr[1] *= scale_y;
            fusion_intr[2] *= scale_x;
            fusion_intr[3] *= scale_y;
            volume.integrate(
                color_for_fusion,
                filtered_depth,
                fusion_intr,
                pkf->getPosef(),
                depth_trunc);

        }

        TriangleMeshRgb mesh = volume.extractMesh(rerun_params_.rerun_reconstruction_mesh_min_weight_);
        if (mesh.vertices.empty() || mesh.faces.empty())
        {
            unfreeze_geo();
            return;
        }

        for (auto& v : mesh.vertices) v += kCommonEvalCompensation;
        if (rerun_params_.rerun_reconstruction_mesh_weld_vertices_) {
            weldTriangleMeshVertices(mesh);
        }
        const bool over_vertex_budget =
            rerun_params_.rerun_reconstruction_mesh_max_vertices_ > 0 &&
            mesh.vertices.size() > rerun_params_.rerun_reconstruction_mesh_max_vertices_;
        const bool over_face_budget =
            rerun_params_.rerun_reconstruction_mesh_max_faces_ > 0 &&
            mesh.faces.size() > rerun_params_.rerun_reconstruction_mesh_max_faces_;
        if (over_vertex_budget || over_face_budget) {
            unfreeze_geo();
            return;
        }

        torch::Tensor vertices;
        torch::Tensor colors;
        torch::Tensor triangles;
        triangleMeshToTensors(mesh, vertices, colors, triangles);

        unfreeze_geo();

        sv::RerunVisualizerBridge::instance().visualizeDebugTriangleMesh(
            "reconstruction_mesh",
            vertices,
            colors,
            triangles,
            iteration,
            "world/mesh/live");

    }
    catch (const std::exception& e)
    {
        unfreeze_geo();
        std::cerr << "[RERUN/reconstruction_mesh] failed: " << e.what() << "\n";
    }
}
