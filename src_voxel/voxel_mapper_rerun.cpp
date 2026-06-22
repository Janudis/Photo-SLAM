#include "include_voxel/voxel_mapper.h"
#include "include/stereo_vision.h"
#include "include_voxel/voxel_mapper_utils.h"

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

#include "ORB-SLAM3/include/Atlas.h"
#include "ORB-SLAM3/include/MapPoint.h"

namespace py = pybind11;

void VoxelMapper::logKeyframeCameraToRerunRecordings(
    const std::shared_ptr<VoxelKeyframe>& pkf,
    unsigned long kf_id,
    bool log_reconstruction_mesh)
{
    if (!pkf || !rerun_params_.enable_rerun_) {
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
        sv::MiniCam cam = pkf->toMiniCam(pkf->image_height_, pkf->image_width_);
        torch::Tensor c2w_cpu = cam.c2w.to(torch::kCPU).contiguous();
        TORCH_CHECK(c2w_cpu.sizes() == torch::IntArrayRef({4, 4}),
                    "MiniCam.c2w must be 4x4");

        Eigen::Matrix4f T_W_C;
        {
            float* data = c2w_cpu.data_ptr<float>();
            Eigen::Map<const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>
                T_row_major(data);
            T_W_C = T_row_major;
        }

        const sv::Camera& camera = scene_->cameras_.at(pkf->camera_id_);
        const float fx = static_cast<float>(camera.fx());
        const float fy = static_cast<float>(camera.fy());
        const float cx = static_cast<float>(camera.cx());
        const float cy = static_cast<float>(camera.cy());
        const int source_frame_id =
            voxel_utils::parseFrameIdFromPath(pkf->img_filename_);
        const std::vector<Eigen::Vector2f> kps_uv;
        const std::vector<int> track_ids;

        sv::RerunVisualizerBridge::instance().visualizeCamera(
            T_W_C,
            pkf->img_undist_,
            kps_uv,
            track_ids,
            getIteration(),
            static_cast<int>(kf_id),
            fx, fy, cx, cy,
            source_frame_id);

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

        log_debug_camera(rerun_params_.run_tsdf_pruned_, "tsdf_pruned");
        log_debug_camera(rerun_params_.run_sdf_pruned_nvblox_, "sdf_pruned_nvblox");
        log_debug_camera(
            log_reconstruction_mesh && rerun_params_.rerun_reconstruction_mesh_,
            "reconstruction_mesh");
        log_debug_camera(rerun_params_.rerun_tsdf_unknown_voxels_, "tsdf_unknown");
        if (rerun_params_.run_floaters_) {
            log_debug_camera(true, "floaters");
            rerun_state_.run_floaters_dirty_ = true;
        }
        log_debug_camera(rerun_params_.run_whole_run_, "whole_run");
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

    const auto rrd_dir = result_dir_ / "rerun";
    std::filesystem::create_directories(rrd_dir);

    if (rerun_params_.run_tsdf_pruned_) {
        sv::RerunVisualizerBridge::instance().saveDebugRecording(
            "tsdf_pruned",
            (rrd_dir / "run_tsdf_pruned.rrd").string());
    }
    if (rerun_params_.run_sdf_pruned_nvblox_) {
        sv::RerunVisualizerBridge::instance().visualizeDebugScalar(
            "sdf_pruned_nvblox",
            1.0,
            getIteration(),
            "world/timeline/end");
        sv::RerunVisualizerBridge::instance().saveDebugRecording(
            "sdf_pruned_nvblox",
            (rrd_dir / "run_sdf_pruned_nvblox.rrd").string());
    }
    if (rerun_params_.rerun_tsdf_unknown_voxels_) {
        if (rerun_state_.rerun_tsdf_unknown_dirty_) {
            logTsdfUnknownVoxelsToRerun(getIteration(), torch::Tensor());
            rerun_state_.rerun_tsdf_unknown_dirty_ = false;
        }
        sv::RerunVisualizerBridge::instance().saveDebugRecording(
            "tsdf_unknown",
            (rrd_dir / "run_tsdf_unknown.rrd").string());
    }
    if (rerun_params_.run_floaters_) {
        logFloatersToRerun(getIteration());
        rerun_state_.run_floaters_dirty_ = false;
        sv::RerunVisualizerBridge::instance().saveDebugRecording(
            "floaters",
            (rrd_dir / "floaters.rrd").string());
    }
    if (rerun_params_.run_whole_run_) {
        logWholeRunNvbloxMeshToRerun(0);
        sv::RerunVisualizerBridge::instance().saveDebugRecording(
            "whole_run",
            (rrd_dir / "whole_run.rrd").string());
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
        std::cout << "[TriangleMesh] capped mesh "
                  << " verts " << mesh.vertices.size() << " -> " << capped.vertices.size()
                  << " faces " << mesh.faces.size() << " -> " << capped.faces.size()
                  << " limits=(" << max_vertices << " verts, " << max_faces << " faces)"
                  << "\n";
        mesh = std::move(capped);
    } else {
        std::cout << "[TriangleMesh] mesh cap produced empty mesh; keeping full mesh.\n";
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

static void nvbloxColorMeshToTensors(
    const std::shared_ptr<const nvblox::ColorMesh>& mesh,
    torch::Tensor& vertices,
    torch::Tensor& colors,
    torch::Tensor& triangles)
{
    vertices = torch::Tensor();
    colors = torch::Tensor();
    triangles = torch::Tensor();
    if (!mesh) return;

    nvblox::CudaStreamOwning cuda_stream;
    std::vector<nvblox::Vector3f> verts =
        mesh->vertices.toVectorAsync(cuda_stream);
    std::vector<nvblox::Color> cols =
        mesh->vertex_appearances.toVectorAsync(cuda_stream);
    std::vector<int> tris =
        mesh->triangles.toVectorAsync(cuda_stream);
    cuda_stream.synchronize();

    if (verts.empty() || tris.empty()) return;

    vertices = torch::empty(
        {static_cast<int64_t>(verts.size()), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto v_acc = vertices.accessor<float, 2>();
    for (int64_t i = 0; i < vertices.size(0); ++i)
    {
        const auto& v = verts[static_cast<std::size_t>(i)];
        v_acc[i][0] = v.x();
        v_acc[i][1] = v.y();
        v_acc[i][2] = v.z();
    }

    if (cols.size() == verts.size())
    {
        colors = torch::empty(
            {static_cast<int64_t>(cols.size()), 3},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
        auto c_acc = colors.accessor<uint8_t, 2>();
        for (int64_t i = 0; i < colors.size(0); ++i)
        {
            const auto& c = cols[static_cast<std::size_t>(i)];
            c_acc[i][0] = c.r();
            c_acc[i][1] = c.g();
            c_acc[i][2] = c.b();
        }
    }

    const int64_t tri_count = static_cast<int64_t>(tris.size() / 3);
    if (tri_count <= 0) {
        vertices = torch::Tensor();
        colors = torch::Tensor();
        return;
    }
    triangles = torch::empty(
        {tri_count, 3},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto f_acc = triangles.accessor<int64_t, 2>();
    for (int64_t i = 0; i < tri_count; ++i)
    {
        f_acc[i][0] = static_cast<int64_t>(tris[static_cast<std::size_t>(3 * i + 0)]);
        f_acc[i][1] = static_cast<int64_t>(tris[static_cast<std::size_t>(3 * i + 1)]);
        f_acc[i][2] = static_cast<int64_t>(tris[static_cast<std::size_t>(3 * i + 2)]);
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

std::filesystem::path resolveNvbloxMeshPath(const std::string& configured_path)
{
    if (configured_path.empty()) {
        return {};
    }
    std::filesystem::path path(configured_path);
    if (path.has_extension()) {
        return path;
    }
    return path / "nvblox_color_mesh.ply";
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

static torch::Tensor normalizeBoolMaskOrZeros(
    torch::Tensor mask,
    const int64_t N,
    const torch::Device& device)
{
    if (!mask.defined() || mask.numel() != N) {
        return torch::zeros(
            {N},
            torch::TensorOptions().dtype(torch::kBool).device(device));
    }
    if (mask.dim() == 2 && mask.size(1) == 1) {
        mask = mask.squeeze(1);
    } else if (mask.dim() != 1) {
        mask = mask.reshape({N});
    }
    return mask.to(device).to(torch::kBool).contiguous();
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
    const torch::Tensor& colors_in)
{
    if (!rerun_params_.enable_rerun_ || !rerun_params_.run_whole_run_ || !voxel_model_) {
        return;
    }
    if (!centers_in.defined() || !sizes_in.defined() ||
        centers_in.dim() != 2 || centers_in.size(1) != 3) {
        return;
    }

    torch::NoGradGuard no_grad;

    const int64_t N = centers_in.size(0);
    const torch::Device dev = centers_in.device();
    torch::Tensor live_colors = colors_in;
    if (N > 0 &&
        (!live_colors.defined() || live_colors.numel() <= 0 ||
         live_colors.dim() != 2 || live_colors.size(0) != N ||
         (live_colors.size(1) != 3 && live_colors.size(1) != 4))) {
        torch::Tensor sh0 = voxel_model_->sh0();
        if (sh0.defined() && sh0.dim() == 2 && sh0.size(0) == N) {
            {
                py::gil_scoped_acquire gil;
                static py::module act_mod = py::module::import("src.utils.activation_utils");
                py::object rgb_py = act_mod.attr("shzero2rgb")(py::cast(sh0));
                live_colors = rgb_py.cast<torch::Tensor>().contiguous();
            }

            torch::Tensor density = voxel_model_->voxelDensityMean();
            if (density.defined() && density.numel() == N) {
                torch::Tensor d_cpu = density.view({-1}).to(torch::kCPU);
                const float d_min = d_cpu.min().item().toFloat();
                const float d_max = d_cpu.max().item().toFloat();
                const float range = d_max - d_min;
                torch::Tensor alpha_cpu;
                if (range < 1e-6f) {
                    alpha_cpu = torch::full_like(d_cpu, 0.8f);
                } else {
                    alpha_cpu = ((d_cpu - d_min) / range).clamp(0.05f, 1.0f);
                }

                torch::Tensor col_cpu = live_colors.to(torch::kCPU);
                if (col_cpu.dim() == 2 && col_cpu.size(0) == N && col_cpu.size(1) == 3) {
                    torch::Tensor col_rgba = torch::zeros({N, 4}, col_cpu.options());
                    col_rgba.index_put_(
                        {torch::indexing::Slice(), torch::indexing::Slice(0, 3)},
                        col_cpu);
                    col_rgba.index_put_({torch::indexing::Slice(), 3}, alpha_cpu);
                    live_colors = col_rgba.to(live_colors.device());
                } else if (col_cpu.dim() == 2 && col_cpu.size(0) == N && col_cpu.size(1) == 4) {
                    col_cpu.index_put_({torch::indexing::Slice(), 3}, alpha_cpu);
                    live_colors = col_cpu.to(live_colors.device());
                }
            }
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

    torch::Tensor orb_mask =
        normalizeBoolMaskOrZeros(voxel_model_->orbVoxelMask(), N, dev);
    torch::Tensor inactive_geo_mask =
        normalizeBoolMaskOrZeros(voxel_model_->inactiveGeoVoxelMask(), N, dev);
    torch::Tensor rgbd_fill_mask =
        normalizeBoolMaskOrZeros(voxel_model_->rgbdFillRenderHolesVoxelMask(), N, dev);
    orb_mask = (orb_mask & (~inactive_geo_mask) & (~rgbd_fill_mask)).to(torch::kBool);

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
            const std::array<float, 4>& fallback_rgba)
    {
        torch::Tensor mask = normalizeBoolMaskOrZeros(mask_in, N, dev);
        torch::Tensor idx = mask.nonzero().squeeze(1);

        torch::Tensor centers_sel;
        torch::Tensor sizes_sel;
        torch::Tensor colors_sel;
        if (!idx.defined() || idx.numel() <= 0) {
            centers_sel = torch::empty(
                {0, 3},
                torch::TensorOptions().dtype(torch::kFloat32).device(dev));
            sizes_sel = torch::empty(
                {0, 1},
                torch::TensorOptions().dtype(torch::kFloat32).device(dev));
            colors_sel = torch::empty(
                {0, 4},
                torch::TensorOptions().dtype(torch::kFloat32).device(dev));
        } else {
            torch::Tensor idx_dev = idx.to(dev).to(torch::kLong);
            centers_sel = centers_in.index_select(0, idx_dev).contiguous();
            sizes_sel = sizes.index_select(0, idx_dev).contiguous();
            colors_sel = colors_for_indices(idx_dev, fallback_rgba);
        }

        sv::RerunVisualizerBridge::instance().visualizeDebugVoxelBoxes(
            "whole_run",
            centers_sel,
            sizes_sel,
            colors_sel,
            iteration,
            entity_path);
    };

    torch::Tensor all_mask = torch::ones(
        {N},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    log_subset(all_mask, "world/voxels", {0.75f, 0.75f, 0.75f, 0.45f});
    log_subset(orb_mask, "world/voxels/source/orb", {0.1f, 0.8f, 1.0f, 0.7f});
    log_subset(inactive_geo_mask, "world/voxels/source/inactive_geo_densify", {0.7f, 0.45f, 0.2f, 0.7f});
    log_subset(rgbd_fill_mask, "world/voxels/source/rgbd_fill_render_holes", {0.95f, 0.25f, 0.85f, 0.7f});
}

void VoxelMapper::appendWholeRunPrunedVoxels(
    int iteration,
    const torch::Tensor& centers_in,
    const torch::Tensor& sizes_in,
    const torch::Tensor& pruned_by_tsdf_in,
    const torch::Tensor& pruned_by_near_in,
    const torch::Tensor& pruned_by_recent_unstable_in,
    const torch::Tensor& pruned_by_final_special_in)
{
    if (!rerun_params_.enable_rerun_ || !rerun_params_.run_whole_run_) {
        return;
    }
    if (!centers_in.defined() || !sizes_in.defined() ||
        centers_in.dim() != 2 || centers_in.size(1) != 3 ||
        centers_in.size(0) <= 0) {
        return;
    }

    torch::NoGradGuard no_grad;

    const int64_t K = centers_in.size(0);
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

    torch::Tensor colors = torch::zeros(
        {K, 4},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    colors.index_put_({torch::indexing::Slice(), 0}, 1.0f);
    colors.index_put_({torch::indexing::Slice(), 1}, 0.55f);
    colors.index_put_({torch::indexing::Slice(), 2}, 0.05f);
    colors.index_put_({torch::indexing::Slice(), 3}, 0.85f);

    torch::Tensor tsdf_mask = torch::zeros(
        {K},
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    if (pruned_by_tsdf_in.defined() && pruned_by_tsdf_in.numel() == K) {
        tsdf_mask =
            pruned_by_tsdf_in.detach()
                .to(torch::kCPU)
                .to(torch::kBool)
                .contiguous()
                .view({K});
    }
    torch::Tensor near_mask = torch::zeros(
        {K},
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    if (pruned_by_near_in.defined() && pruned_by_near_in.numel() == K) {
        near_mask =
            pruned_by_near_in.detach()
                .to(torch::kCPU)
                .to(torch::kBool)
                .contiguous()
                .view({K});
    }
    torch::Tensor recent_unstable_mask = torch::zeros(
        {K},
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    if (pruned_by_recent_unstable_in.defined() &&
        pruned_by_recent_unstable_in.numel() == K) {
        recent_unstable_mask =
            pruned_by_recent_unstable_in.detach()
                .to(torch::kCPU)
                .to(torch::kBool)
                .contiguous()
                .view({K});
    }
    torch::Tensor final_special_mask = torch::zeros(
        {K},
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    if (pruned_by_final_special_in.defined() &&
        pruned_by_final_special_in.numel() == K) {
        final_special_mask =
            pruned_by_final_special_in.detach()
                .to(torch::kCPU)
                .to(torch::kBool)
                .contiguous()
                .view({K});
    }
    torch::Tensor source_claimed =
        (tsdf_mask | near_mask | recent_unstable_mask | final_special_mask).to(torch::kBool);
    torch::Tensor svraster_mask = (~source_claimed).to(torch::kBool);

    auto append_source =
        [&](const torch::Tensor& mask,
            torch::Tensor& centers_accum,
            torch::Tensor& sizes_accum,
            torch::Tensor& colors_accum,
            const std::array<float, 4>& rgba,
            const std::string& entity_path)
    {
        torch::Tensor idx = mask.nonzero().squeeze(1);
        if (!idx.defined() || idx.numel() <= 0) {
            return;
        }
        torch::Tensor centers_sel = centers.index_select(0, idx).contiguous();
        torch::Tensor sizes_sel = sizes.index_select(0, idx).contiguous();
        torch::Tensor colors_sel = torch::zeros(
            {idx.numel(), 4},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        colors_sel.index_put_({torch::indexing::Slice(), 0}, rgba[0]);
        colors_sel.index_put_({torch::indexing::Slice(), 1}, rgba[1]);
        colors_sel.index_put_({torch::indexing::Slice(), 2}, rgba[2]);
        colors_sel.index_put_({torch::indexing::Slice(), 3}, rgba[3]);

        if (!centers_accum.defined()) {
            centers_accum = centers_sel;
            sizes_accum = sizes_sel;
            colors_accum = colors_sel;
        } else {
            centers_accum = torch::cat({centers_accum, centers_sel}, 0).contiguous();
            sizes_accum = torch::cat({sizes_accum, sizes_sel}, 0).contiguous();
            colors_accum = torch::cat({colors_accum, colors_sel}, 0).contiguous();
        }

        sv::RerunVisualizerBridge::instance().visualizeDebugVoxelBoxes(
            "whole_run",
            centers_accum,
            sizes_accum,
            colors_accum,
            iteration,
            entity_path);
    };

    if (!rerun_state_.whole_run_pruned_centers_accum_.defined()) {
        rerun_state_.whole_run_pruned_centers_accum_ = centers;
        rerun_state_.whole_run_pruned_sizes_accum_ = sizes;
        rerun_state_.whole_run_pruned_colors_accum_ = colors;
    } else {
        rerun_state_.whole_run_pruned_centers_accum_ =
            torch::cat({rerun_state_.whole_run_pruned_centers_accum_, centers}, 0).contiguous();
        rerun_state_.whole_run_pruned_sizes_accum_ =
            torch::cat({rerun_state_.whole_run_pruned_sizes_accum_, sizes}, 0).contiguous();
        rerun_state_.whole_run_pruned_colors_accum_ =
            torch::cat({rerun_state_.whole_run_pruned_colors_accum_, colors}, 0).contiguous();
    }

    sv::RerunVisualizerBridge::instance().visualizeDebugVoxelBoxes(
        "whole_run",
        rerun_state_.whole_run_pruned_centers_accum_,
        rerun_state_.whole_run_pruned_sizes_accum_,
        rerun_state_.whole_run_pruned_colors_accum_,
        iteration,
        "world/pruned");

    append_source(
        tsdf_mask,
        rerun_state_.whole_run_pruned_tsdf_centers_accum_,
        rerun_state_.whole_run_pruned_tsdf_sizes_accum_,
        rerun_state_.whole_run_pruned_tsdf_colors_accum_,
        {1.0f, 0.1f, 0.1f, 0.85f},
        "world/pruned/source/tsdf");
    append_source(
        svraster_mask,
        rerun_state_.whole_run_pruned_svraster_centers_accum_,
        rerun_state_.whole_run_pruned_svraster_sizes_accum_,
        rerun_state_.whole_run_pruned_svraster_colors_accum_,
        {0.2f, 0.55f, 1.0f, 0.85f},
        "world/pruned/source/svraster");
    append_source(
        near_mask,
        rerun_state_.whole_run_pruned_near_centers_accum_,
        rerun_state_.whole_run_pruned_near_sizes_accum_,
        rerun_state_.whole_run_pruned_near_colors_accum_,
        {1.0f, 0.85f, 0.05f, 0.85f},
        "world/pruned/source/near_voxels");
    append_source(
        recent_unstable_mask,
        rerun_state_.whole_run_pruned_recent_unstable_centers_accum_,
        rerun_state_.whole_run_pruned_recent_unstable_sizes_accum_,
        rerun_state_.whole_run_pruned_recent_unstable_colors_accum_,
        {0.85f, 0.2f, 1.0f, 0.85f},
        "world/pruned/source/recent_unstable");
    append_source(
        final_special_mask,
        rerun_state_.whole_run_pruned_final_special_centers_accum_,
        rerun_state_.whole_run_pruned_final_special_sizes_accum_,
        rerun_state_.whole_run_pruned_final_special_colors_accum_,
        {1.0f, 0.45f, 0.0f, 0.85f},
        "world/pruned/source/final_special_prune_enable");
}

torch::Tensor VoxelMapper::computeNvbloxProjectiveSdfForCorners(
    const torch::Tensor& corner_points,
    const std::string& nvblox_mesh_path)
{
    if (!corner_points.defined() ||
        corner_points.numel() == 0 ||
        corner_points.dim() != 3 ||
        corner_points.size(1) != 8 ||
        corner_points.size(2) != 3 ||
        nvblox_mesh_path.empty() ||
        !std::filesystem::exists(nvblox_mesh_path)) {
        return torch::Tensor();
    }

    const int64_t K = corner_points.size(0);
    torch::Tensor points_flat =
        corner_points.detach().to(torch::kCPU).to(torch::kFloat32).contiguous().view({K * 8, 3});
    torch::Tensor best_sdf = torch::full(
        {K * 8},
        std::numeric_limits<float>::quiet_NaN(),
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    torch::Tensor best_abs = torch::full(
        {K * 8},
        std::numeric_limits<float>::infinity(),
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    for (const auto& kv : scene_->keyframes()) {
        const auto& kf = kv.second;
        if (!kf || kf->image_width_ <= 0 || kf->image_height_ <= 0 ||
            kf->cam_.fx() <= 1.0e-6f || kf->cam_.fy() <= 1.0e-6f) {
            continue;
        }

        const Eigen::Matrix4f Tcw_eig = kf->getPosef().matrix();
        torch::Tensor Tcw_cpu = torch::empty(
            {4, 4},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        auto Tacc = Tcw_cpu.accessor<float, 2>();
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                Tacc[r][c] = Tcw_eig(r, c);
            }
        }

        torch::Tensor sdf =
            sv::RerunVisualizerBridge::instance().computeGtProjectiveSdf(
                points_flat,
                Tcw_cpu,
                kf->cam_.fx(),
                kf->cam_.fy(),
                kf->cam_.cx(),
                kf->cam_.cy(),
                kf->image_width_,
                kf->image_height_,
                nvblox_mesh_path,
                false,
                "",
                4);
        if (!sdf.defined() || sdf.numel() != K * 8) {
            continue;
        }
        sdf = sdf.to(torch::kCPU).to(torch::kFloat32).contiguous().view({K * 8});
        torch::Tensor finite = torch::isfinite(sdf);
        torch::Tensor abs_sdf = sdf.abs();
        torch::Tensor update = (finite & (abs_sdf < best_abs)).to(torch::kBool);
        best_sdf = torch::where(update, sdf, best_sdf);
        best_abs = torch::where(update, abs_sdf, best_abs);
    }

    return best_sdf.view({K, 8}).contiguous();
}

bool VoxelMapper::ensureRerunGtSdfGridCache(const std::string& mesh_path)
{
    if (!voxel_model_ || mesh_path.empty() || !std::filesystem::exists(mesh_path)) {
        return false;
    }
    torch::Tensor grid_key = voxel_model_->gridPtsKey();
    if (!grid_key.defined() || grid_key.numel() == 0 ||
        grid_key.dim() != 2 || grid_key.size(1) != 3) {
        return false;
    }

    torch::NoGradGuard no_grad;
    torch::Tensor grid_key_cpu =
        grid_key.to(torch::kCPU).to(torch::kLong).contiguous();
    const bool can_reuse_cache = !rerun_params_.rerun_tsdf_pruned_align_gt_to_slam_;
    const bool cache_valid =
        can_reuse_cache &&
        rerun_state_.rerun_gt_sdf_grid_pts_cpu_.defined() &&
        rerun_state_.rerun_gt_sdf_grid_keys_cpu_.defined() &&
        rerun_state_.rerun_gt_sdf_grid_mesh_path_ == mesh_path &&
        rerun_state_.rerun_gt_sdf_grid_pts_cpu_.numel() == grid_key_cpu.size(0) &&
        rerun_state_.rerun_gt_sdf_grid_keys_cpu_.sizes().vec() == grid_key_cpu.sizes().vec() &&
        torch::equal(rerun_state_.rerun_gt_sdf_grid_keys_cpu_, grid_key_cpu);
    if (cache_valid) {
        return true;
    }

    torch::Tensor grid_points = voxel_model_->gridPointsWorld();
    if (!grid_points.defined() || grid_points.numel() == 0 ||
        grid_points.dim() != 2 || grid_points.size(1) != 3 ||
        grid_points.size(0) != grid_key_cpu.size(0)) {
        return false;
    }

    torch::Tensor gt_sdf =
        sv::RerunVisualizerBridge::instance().computeGtSurfaceDistance(
            grid_points,
            mesh_path,
            rerun_params_.rerun_tsdf_pruned_align_gt_to_slam_,
            rerun_params_.rerun_tsdf_pruned_gt_traj_path_,
            rerun_params_.rerun_tsdf_pruned_align_min_pairs_);
    if (!gt_sdf.defined() || gt_sdf.numel() != grid_key_cpu.size(0)) {
        // std::cerr << "[RERUN/gt_sdf/cache] failed to compute GT SDF cache"
        //           << " grid_pts=" << grid_key_cpu.size(0)
        //           << " mesh=" << mesh_path << "\n";
        return false;
    }

    rerun_state_.rerun_gt_sdf_grid_keys_cpu_ = grid_key_cpu.clone();
    rerun_state_.rerun_gt_sdf_grid_pts_cpu_ =
        gt_sdf.to(torch::kCPU).to(torch::kFloat32).view({-1}).contiguous().clone();
    rerun_state_.rerun_gt_sdf_grid_mesh_path_ = mesh_path;
    rerun_state_.rerun_gt_sdf_grid_kfid_ = rerun_state_.rerun_gt_sdf_pending_kfid_;

    const int64_t finite =
        torch::isfinite(rerun_state_.rerun_gt_sdf_grid_pts_cpu_).sum().item<int64_t>();
    // std::cout << "[RERUN/gt_sdf/projective_cache] refreshed"
    //           << " grid_pts=" << rerun_state_.rerun_gt_sdf_grid_pts_cpu_.numel()
    //           << " finite=" << finite
    //           << " kf=" << rerun_state_.rerun_gt_sdf_grid_kfid_
    //           << " mesh=" << mesh_path
    //           << "\n";
    return true;
}

void VoxelMapper::logTsdfUnknownVoxelsToRerun(
    int iteration,
    const torch::Tensor& voxel_colors)
{
    if (!rerun_params_.enable_rerun_ || !rerun_params_.rerun_tsdf_unknown_voxels_ ||
        !voxel_model_) {
        return;
    }
    if (!hasTsdfForSampling()) {
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
    if (sizes.dim() == 1) {
        sizes = sizes.view({N, 1});
    } else if (sizes.dim() == 2 && sizes.size(1) == 1) {
        // ok
    } else {
        sizes = sizes.reshape({N, 1});
    }

    TsdfCornerSample c = sampleTsdfAtSvrasterGridCornersWorld();
    if (!c.tsdf.defined() || !c.weight.defined() || !c.success.defined() ||
        c.tsdf.size(0) != N || c.weight.size(0) != N || c.success.size(0) != N ||
        c.tsdf.dim() != 2 || c.tsdf.size(1) != 8 ||
        c.weight.dim() != 2 || c.weight.size(1) != 8 ||
        c.success.dim() != 2 || c.success.size(1) != 8) {
        return;
    }

    const torch::Device dev = centers.device();
    torch::Tensor ok8 = c.success.to(dev).to(torch::kBool);
    torch::Tensor w8 = c.weight.to(dev).to(torch::kFloat32);
    const float min_weight = std::max(0.0f, sdf_params_.tsdf_prune_min_weight_);
    torch::Tensor corner_valid = (ok8 & (w8 >= min_weight)).to(torch::kBool);
    torch::Tensor valid_count =
        corner_valid.to(torch::kInt32).sum(/*dim=*/1).to(torch::kInt32);
    torch::Tensor unknown_mask =
        (valid_count < sdf_params_.tsdf_prune_min_valid_corners_).to(torch::kBool);

    torch::Tensor orb_mask =
        normalizeBoolMaskOrZeros(voxel_model_->orbVoxelMask(), N, dev);
    torch::Tensor inactive_geo_mask =
        normalizeBoolMaskOrZeros(voxel_model_->inactiveGeoVoxelMask(), N, dev);
    torch::Tensor rgbd_fill_mask =
        normalizeBoolMaskOrZeros(voxel_model_->rgbdFillRenderHolesVoxelMask(), N, dev);

    const std::string root = "world/unknown_voxels";

    struct TsdfDebugKfContext
    {
        std::size_t kfid = 0;
        cv::Mat depth_meters;
        Eigen::Matrix3f Rcw = Eigen::Matrix3f::Identity();
        Eigen::Vector3f tcw = Eigen::Vector3f::Zero();
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
        int width = 0;
        int height = 0;
    };

    std::vector<TsdfDebugKfContext> tsdf_debug_kfs;
    if (scene_) {
        tsdf_debug_kfs.reserve(scene_->keyframes().size());
        for (const auto& kv : scene_->keyframes()) {
            const auto& kf = kv.second;
            if (!kf || kf->img_auxiliary_undist_.empty()) {
                continue;
            }
            cv::Mat depth_meters;
            if (!depthMatToMeters(kf->img_auxiliary_undist_, depth_meters) ||
                depth_meters.empty()) {
                continue;
            }
            if (depth_meters.channels() > 1) {
                cv::extractChannel(depth_meters, depth_meters, 0);
            }
            if (depth_meters.type() != CV_32FC1) {
                depth_meters.convertTo(depth_meters, CV_32FC1);
            }
            const float fx = kf->cam_.fx();
            const float fy = kf->cam_.fy();
            if (fx <= 1.0e-6f || fy <= 1.0e-6f) {
                continue;
            }
            TsdfDebugKfContext ctx;
            ctx.kfid = kf->fid_;
            ctx.depth_meters = depth_meters;
            Sophus::SE3f Tcw = kf->getPosef();
            ctx.Rcw = Tcw.rotationMatrix();
            ctx.tcw = Tcw.translation();
            ctx.fx = fx;
            ctx.fy = fy;
            ctx.cx = kf->cam_.cx();
            ctx.cy = kf->cam_.cy();
            ctx.width = depth_meters.cols;
            ctx.height = depth_meters.rows;
            tsdf_debug_kfs.push_back(std::move(ctx));
        }
    }

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

        if (K <= 0 || !voxel_colors.defined() || voxel_colors.numel() <= 0 ||
            voxel_colors.dim() != 2 || voxel_colors.size(0) != N ||
            (voxel_colors.size(1) != 3 && voxel_colors.size(1) != 4)) {
            return fallback.contiguous();
        }

        torch::Tensor idx = idx_in.to(voxel_colors.device()).to(torch::kLong);
        torch::Tensor selected =
            voxel_colors.index_select(0, idx).to(dev).to(torch::kFloat32).contiguous();
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
            const std::array<float, 4>& fallback_rgba)
    {
        torch::Tensor subset_mask = normalizeBoolMaskOrZeros(mask_in, N, dev);
        torch::Tensor idx = subset_mask.nonzero().squeeze(1);

        torch::Tensor centers_sel;
        torch::Tensor sizes_sel;
        torch::Tensor colors_sel;
        if (!idx.defined() || idx.numel() <= 0) {
            centers_sel = torch::empty(
                {0, 3},
                torch::TensorOptions().dtype(torch::kFloat32).device(dev));
            sizes_sel = torch::empty(
                {0, 1},
                torch::TensorOptions().dtype(torch::kFloat32).device(dev));
            colors_sel = torch::empty(
                {0, 4},
                torch::TensorOptions().dtype(torch::kFloat32).device(dev));
        } else {
            torch::Tensor idx_dev = idx.to(dev).to(torch::kLong);
            centers_sel = centers.index_select(0, idx_dev).contiguous();
            sizes_sel = sizes.index_select(0, idx_dev).contiguous();
            colors_sel = colors_for_indices(idx_dev, fallback_rgba);
        }

        sv::RerunVisualizerBridge::instance().visualizeDebugVoxelBoxes(
            "tsdf_unknown",
            centers_sel,
            sizes_sel,
            colors_sel,
            iteration,
            entity_path);
    };

    log_subset(
        unknown_mask,
        root,
        {0.45f, 0.45f, 0.45f, 0.45f});
    log_subset(
        (unknown_mask & orb_mask).to(torch::kBool),
        root + "/source/orb",
        {0.1f, 0.8f, 1.0f, 0.7f});
    log_subset(
        (unknown_mask & inactive_geo_mask).to(torch::kBool),
        root + "/source/inactive_geo_densify",
        {0.7f, 0.45f, 0.2f, 0.7f});
    log_subset(
        (unknown_mask & rgbd_fill_mask).to(torch::kBool),
        root + "/source/rgbd_fill_render_holes",
        {0.95f, 0.25f, 0.85f, 0.7f});

    auto log_unknown_corners =
        [&](const torch::Tensor& voxel_mask_in,
            const std::string& entity_path,
            const std::string& source_name)
    {
        torch::Tensor empty_points = torch::empty(
            {0, 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(dev));
        torch::Tensor empty_colors = torch::empty(
            {0, 4},
            torch::TensorOptions().dtype(torch::kFloat32).device(dev));

        if (!c.points_world.defined() || c.points_world.numel() == 0 ||
            c.points_world.dim() != 3 || c.points_world.size(0) != N ||
            c.points_world.size(1) != 8 || c.points_world.size(2) != 3) {
            sv::RerunVisualizerBridge::instance().visualizeDebugPoints3D(
                "tsdf_unknown",
                empty_points,
                empty_colors,
                iteration,
                entity_path,
                0.006f,
                {});
            return;
        }

        torch::Tensor voxel_mask =
            (normalizeBoolMaskOrZeros(voxel_mask_in, N, dev) & unknown_mask)
                .to(torch::kBool);
        torch::Tensor unknown_corner_mask =
            (voxel_mask.view({N, 1}) & (~corner_valid)).to(torch::kBool);
        torch::Tensor unknown_corner_pairs = unknown_corner_mask.nonzero();
        if (!unknown_corner_pairs.defined() || unknown_corner_pairs.numel() <= 0) {
            sv::RerunVisualizerBridge::instance().visualizeDebugPoints3D(
                "tsdf_unknown",
                empty_points,
                empty_colors,
                iteration,
                entity_path,
                0.006f,
                {});
            return;
        }

        torch::Tensor unknown_corner_vox_idx =
            unknown_corner_pairs.index({torch::indexing::Slice(), 0}).to(dev).to(torch::kLong);
        torch::Tensor unknown_corner_local_idx =
            unknown_corner_pairs.index({torch::indexing::Slice(), 1}).to(dev).to(torch::kLong);
        torch::Tensor unknown_corner_flat_idx =
            unknown_corner_vox_idx * 8 + unknown_corner_local_idx;

        torch::Tensor corner_points =
            c.points_world.to(dev).to(torch::kFloat32)
                .reshape({N * 8, 3})
                .index_select(0, unknown_corner_flat_idx)
                .contiguous();
        torch::Tensor tsdf_sel =
            c.tsdf.to(dev).to(torch::kFloat32)
                .reshape({N * 8})
                .index_select(0, unknown_corner_flat_idx)
                .contiguous();
        torch::Tensor weight_sel =
            w8.reshape({N * 8})
                .index_select(0, unknown_corner_flat_idx)
                .contiguous();

        torch::Tensor corner_colors = torch::zeros(
            {corner_points.size(0), 4},
            torch::TensorOptions().dtype(torch::kFloat32).device(dev));
        corner_colors.index_put_({torch::indexing::Slice(), 0}, 1.0f);
        corner_colors.index_put_({torch::indexing::Slice(), 3}, 1.0f);

        std::vector<std::string> labels;
        labels.reserve(static_cast<size_t>(corner_points.size(0)));
        torch::Tensor unknown_corner_vox_cpu =
            unknown_corner_vox_idx.to(torch::kCPU).to(torch::kLong).contiguous();
        torch::Tensor unknown_corner_local_cpu =
            unknown_corner_local_idx.to(torch::kCPU).to(torch::kLong).contiguous();
        torch::Tensor tsdf_cpu = tsdf_sel.to(torch::kCPU).to(torch::kFloat32).contiguous();
        torch::Tensor weight_cpu = weight_sel.to(torch::kCPU).to(torch::kFloat32).contiguous();
        torch::Tensor corner_points_cpu =
            corner_points.to(torch::kCPU).to(torch::kFloat32).contiguous();
        auto unknown_vox_acc = unknown_corner_vox_cpu.accessor<int64_t, 1>();
        auto unknown_corner_acc = unknown_corner_local_cpu.accessor<int64_t, 1>();
        auto tsdf_acc = tsdf_cpu.accessor<float, 1>();
        auto weight_acc = weight_cpu.accessor<float, 1>();
        auto point_acc = corner_points_cpu.accessor<float, 2>();
        const bool has_latest_tsdf_context =
            sdf_state_.svraster_tsdf_last_context_valid_ &&
            !sdf_state_.svraster_tsdf_last_depth_meters_.empty() &&
            sdf_state_.svraster_tsdf_last_width_ > 0 &&
            sdf_state_.svraster_tsdf_last_height_ > 0 &&
            sdf_state_.svraster_tsdf_last_fx_ > 1.0e-6f &&
            sdf_state_.svraster_tsdf_last_fy_ > 1.0e-6f;
        Eigen::Matrix3f latest_Rcw = Eigen::Matrix3f::Identity();
        Eigen::Vector3f latest_tcw = Eigen::Vector3f::Zero();
        if (has_latest_tsdf_context) {
            latest_Rcw = sdf_state_.svraster_tsdf_last_Tcw_.rotationMatrix();
            latest_tcw = sdf_state_.svraster_tsdf_last_Tcw_.translation();
        }
        const float latest_trunc_m =
            std::max(1.0e-6f, sdf_params_.tsdf_density_init_trunc_vox_ * tsdfMetricVoxelSize());

        struct LatestGateInfo
        {
            std::string reason;
            std::string label;
            bool would_update_but_weight_zero = false;
        };

        auto latest_gate_label =
            [&](int64_t i) -> LatestGateInfo
        {
            if (!has_latest_tsdf_context) {
                return {"no_latest_tsdf_context", "latest_gate=no_latest_tsdf_context", false};
            }
            const Eigen::Vector3f p_W(
                point_acc[i][0],
                point_acc[i][1],
                point_acc[i][2]);
            const Eigen::Vector3f p_C = latest_Rcw * p_W + latest_tcw;
            const float z = p_C.z();
            if (!std::isfinite(z) || z <= RGBD_min_depth_) {
                std::ostringstream oss;
                oss << "latest_gate=z_invalid latest_kf=" << sdf_state_.svraster_tsdf_last_kfid_
                    << " z=" << std::fixed << std::setprecision(5) << z;
                return {"z_invalid", oss.str(), false};
            }

            const float u = sdf_state_.svraster_tsdf_last_fx_ * p_C.x() / z + sdf_state_.svraster_tsdf_last_cx_;
            const float v = sdf_state_.svraster_tsdf_last_fy_ * p_C.y() / z + sdf_state_.svraster_tsdf_last_cy_;
            if (!std::isfinite(u) || !std::isfinite(v) ||
                u < 0.0f || u >= static_cast<float>(sdf_state_.svraster_tsdf_last_width_) ||
                v < 0.0f || v >= static_cast<float>(sdf_state_.svraster_tsdf_last_height_)) {
                std::ostringstream oss;
                oss << "latest_gate=out_of_image latest_kf=" << sdf_state_.svraster_tsdf_last_kfid_
                    << " u=" << std::fixed << std::setprecision(2) << u
                    << " v=" << v
                    << " z=" << std::setprecision(5) << z;
                return {"out_of_image", oss.str(), false};
            }

            const int uu = std::clamp(
                static_cast<int>(std::floor(u)),
                0,
                std::max(0, sdf_state_.svraster_tsdf_last_width_ - 1));
            const int vv = std::clamp(
                static_cast<int>(std::floor(v)),
                0,
                std::max(0, sdf_state_.svraster_tsdf_last_height_ - 1));
            const float sampled_depth =
                sdf_state_.svraster_tsdf_last_depth_meters_.at<float>(vv, uu);
            if (!std::isfinite(sampled_depth)) {
                std::ostringstream oss;
                oss << "latest_gate=depth_not_finite latest_kf="
                    << sdf_state_.svraster_tsdf_last_kfid_
                    << " px=(" << uu << "," << vv << ")";
                return {"depth_not_finite", oss.str(), false};
            }
            if (sampled_depth <= RGBD_min_depth_) {
                std::ostringstream oss;
                oss << "latest_gate=depth_le_min latest_kf="
                    << sdf_state_.svraster_tsdf_last_kfid_
                    << " depth=" << std::fixed << std::setprecision(5)
                    << sampled_depth
                    << " px=(" << uu << "," << vv << ")";
                return {"depth_le_min", oss.str(), false};
            }
            if (sampled_depth >= RGBD_max_depth_) {
                std::ostringstream oss;
                oss << "latest_gate=depth_ge_max latest_kf="
                    << sdf_state_.svraster_tsdf_last_kfid_
                    << " depth=" << std::fixed << std::setprecision(5)
                    << sampled_depth
                    << " px=(" << uu << "," << vv << ")";
                return {"depth_ge_max", oss.str(), false};
            }
            if (!(z < sampled_depth + latest_trunc_m)) {
                std::ostringstream oss;
                oss << "latest_gate=behind_surface_trunc latest_kf="
                    << sdf_state_.svraster_tsdf_last_kfid_
                    << " z=" << std::fixed << std::setprecision(5) << z
                    << " depth=" << sampled_depth
                    << " trunc=" << latest_trunc_m
                    << " px=(" << uu << "," << vv << ")";
                return {"behind_surface_trunc", oss.str(), false};
            }

            std::ostringstream oss;
            oss << "latest_gate=would_update latest_kf="
                << sdf_state_.svraster_tsdf_last_kfid_
                << " z=" << std::fixed << std::setprecision(5) << z
                << " depth=" << sampled_depth
                << " sdf=" << (sampled_depth - z)
                << " px=(" << uu << "," << vv << ")";
            bool would_update_but_weight_zero = false;
            if (weight_acc[i] <= 0.0f) {
                oss << " warning=would_update_but_weight_zero";
                would_update_but_weight_zero = true;
            }
            return {"would_update", oss.str(), would_update_but_weight_zero};
        };

        std::map<std::string, int64_t> reason_counts;
        int64_t would_update_but_weight_zero_count = 0;
        std::vector<std::string> sample_labels_for_stdout;
        constexpr int kMaxUnknownReasonSamples = 3;
        std::map<std::string, int64_t> history_counts;
        std::vector<std::string> history_samples_for_stdout;

        auto historical_gate_reason =
            [&](int64_t i) -> std::string
        {
            if (tsdf_debug_kfs.empty()) {
                return "no_keyframe_contexts";
            }
            const Eigen::Vector3f p_W(
                point_acc[i][0],
                point_acc[i][1],
                point_acc[i][2]);

            bool ever_front = false;
            bool ever_in_image = false;
            bool ever_finite_depth = false;
            bool ever_depth_range = false;
            for (const TsdfDebugKfContext& ctx : tsdf_debug_kfs) {
                if (ctx.width <= 0 || ctx.height <= 0 || ctx.depth_meters.empty()) {
                    continue;
                }
                const Eigen::Vector3f p_C = ctx.Rcw * p_W + ctx.tcw;
                const float z = p_C.z();
                if (!std::isfinite(z) || z <= RGBD_min_depth_) {
                    continue;
                }
                ever_front = true;

                const float u = ctx.fx * p_C.x() / z + ctx.cx;
                const float v = ctx.fy * p_C.y() / z + ctx.cy;
                if (!std::isfinite(u) || !std::isfinite(v) ||
                    u < 0.0f || u >= static_cast<float>(ctx.width) ||
                    v < 0.0f || v >= static_cast<float>(ctx.height)) {
                    continue;
                }
                ever_in_image = true;

                const int uu = std::clamp(
                    static_cast<int>(std::floor(u)),
                    0,
                    std::max(0, ctx.width - 1));
                const int vv = std::clamp(
                    static_cast<int>(std::floor(v)),
                    0,
                    std::max(0, ctx.height - 1));
                const float sampled_depth = ctx.depth_meters.at<float>(vv, uu);
                if (!std::isfinite(sampled_depth)) {
                    continue;
                }
                ever_finite_depth = true;
                if (sampled_depth <= RGBD_min_depth_ ||
                    sampled_depth >= RGBD_max_depth_) {
                    continue;
                }
                ever_depth_range = true;
                if (z < sampled_depth + latest_trunc_m) {
                    return "would_update_any_keyframe";
                }
            }

            if (ever_depth_range) {
                return "behind_surface_trunc_all_keyframes";
            }
            if (ever_finite_depth) {
                return "depth_out_of_range_all_keyframes";
            }
            if (ever_in_image) {
                return "depth_not_finite_all_keyframes";
            }
            if (ever_front) {
                return "out_of_image_all_keyframes";
            }
            return "z_invalid_all_keyframes";
        };

        for (int64_t i = 0; i < unknown_corner_vox_cpu.size(0); ++i) {
            const int64_t voxel_id = unknown_vox_acc[i];
            const int64_t corner_id = unknown_corner_acc[i];
            LatestGateInfo gate = latest_gate_label(i);
            reason_counts[gate.reason] += 1;
            if (gate.would_update_but_weight_zero) {
                ++would_update_but_weight_zero_count;
            }
            const std::string history_reason = historical_gate_reason(i);
            history_counts[history_reason] += 1;
            std::ostringstream rr_label;
            rr_label << "voxel_id=" << voxel_id
                     << " corner=" << corner_id
                     << " weight=" << std::fixed << std::setprecision(3)
                     << weight_acc[i]
                     << " " << gate.label
                     << " history_gate=" << history_reason;
            const std::string label = rr_label.str();

            std::ostringstream stdout_label;
            stdout_label << "voxel_id=" << voxel_id
                         << " corner=" << corner_id
                         << " latest=" << gate.reason
                         << " history=" << history_reason;
            const std::string compact_label = stdout_label.str();

            if (static_cast<int>(sample_labels_for_stdout.size()) < kMaxUnknownReasonSamples) {
                sample_labels_for_stdout.push_back(compact_label);
            }
            if (history_reason == "would_update_any_keyframe" &&
                static_cast<int>(history_samples_for_stdout.size()) < kMaxUnknownReasonSamples) {
                history_samples_for_stdout.push_back(compact_label);
            }
            labels.push_back(label);
        }

        {
            torch::Tensor voxel_mask =
                (normalizeBoolMaskOrZeros(voxel_mask_in, N, dev) & unknown_mask)
                    .to(torch::kBool);
            const int64_t unknown_voxels =
                voxel_mask.sum().item<int64_t>();
            std::ostringstream oss;
            oss << "[TSDF UNKNOWN REASONS] iter=" << iteration
                << " source=" << source_name
                << " unknown_voxels=" << unknown_voxels
                << " unknown_corners=" << unknown_corner_vox_cpu.size(0)
                << " latest_kf="
                << (has_latest_tsdf_context ? static_cast<long long>(sdf_state_.svraster_tsdf_last_kfid_) : -1)
                << " would_update_but_weight_zero="
                << would_update_but_weight_zero_count;
            for (const auto& kv : reason_counts) {
                oss << " " << kv.first << "=" << kv.second;
            }
            std::cout << oss.str() << "\n";
            for (const std::string& sample : sample_labels_for_stdout) {
                std::cout << "  [TSDF UNKNOWN SAMPLE] source="
                          << source_name << " " << sample << "\n";
            }
        }

        {
            std::ostringstream oss;
            oss << "[TSDF UNKNOWN HISTORY] iter=" << iteration
                << " source=" << source_name
                << " keyframes_checked=" << tsdf_debug_kfs.size()
                << " unknown_corners=" << unknown_corner_vox_cpu.size(0);
            for (const auto& kv : history_counts) {
                oss << " " << kv.first << "=" << kv.second;
            }
            std::cout << oss.str() << "\n";
            for (const std::string& sample : history_samples_for_stdout) {
                std::cout << "  [TSDF UNKNOWN HISTORY SAMPLE] source="
                          << source_name << " " << sample << "\n";
            }
        }

        sv::RerunVisualizerBridge::instance().visualizeDebugPoints3D(
            "tsdf_unknown",
            corner_points,
            corner_colors,
            iteration,
            entity_path,
            0.006f,
            labels);
    };

    log_unknown_corners(
        (unknown_mask & orb_mask).to(torch::kBool),
        root + "/source/orb/corners",
        "orb");
    log_unknown_corners(
        (unknown_mask & inactive_geo_mask).to(torch::kBool),
        root + "/source/inactive_geo_densify/corners",
        "inactive_geo_densify");
    log_unknown_corners(
        (unknown_mask & rgbd_fill_mask).to(torch::kBool),
        root + "/source/rgbd_fill_render_holes/corners",
        "rgbd_fill_render_holes");
}

void VoxelMapper::logFloatersToRerun(int iteration)
{
    if (!rerun_params_.enable_rerun_ || !rerun_params_.run_floaters_ || !voxel_model_) {
        return;
    }

    std::string gt_mesh_path = rerun_params_.rerun_tsdf_pruned_gt_mesh_path_;
    if (gt_mesh_path.empty()) {
        gt_mesh_path = rerun_params_.rerun_gt_mesh_path_;
    }
    if (gt_mesh_path.empty() || !std::filesystem::exists(gt_mesh_path)) {
        return;
    }
    if (!ensureRerunGtSdfGridCache(gt_mesh_path)) {
        return;
    }

    torch::NoGradGuard no_grad;

    torch::Tensor centers = voxel_model_->voxCenter();
    torch::Tensor sizes = voxel_model_->voxSize();
    torch::Tensor vox_key = voxel_model_->voxKey();
    if (!centers.defined() || !sizes.defined() || !vox_key.defined() ||
        centers.dim() != 2 || centers.size(1) != 3 ||
        vox_key.dim() != 2 || vox_key.size(1) != 8 ||
        centers.size(0) != vox_key.size(0) ||
        sizes.size(0) != centers.size(0) ||
        !rerun_state_.rerun_gt_sdf_grid_pts_cpu_.defined() ||
        rerun_state_.rerun_gt_sdf_grid_pts_cpu_.numel() <= 0) {
        return;
    }

    const int64_t N = centers.size(0);
    if (N <= 0) {
        return;
    }

    torch::Tensor sizes_view = sizes;
    if (sizes_view.dim() == 1) {
        sizes_view = sizes_view.view({N, 1});
    } else if (sizes_view.dim() == 2 && sizes_view.size(1) == 1) {
        // ok
    } else {
        sizes_view = sizes_view.reshape({N, 1});
    }

    torch::Tensor key_flat_cpu =
        vox_key.to(torch::kCPU).to(torch::kLong).reshape({-1});
    torch::Tensor gt_dist =
        rerun_state_.rerun_gt_sdf_grid_pts_cpu_
            .index_select(0, key_flat_cpu)
            .view({N, 8})
            .to(torch::kCPU)
            .to(torch::kFloat32)
            .contiguous();

    torch::Tensor sizes_cpu =
        sizes_view.to(torch::kCPU).to(torch::kFloat32).view({N}).contiguous();
    torch::Tensor gt_valid = torch::isfinite(gt_dist);
    torch::Tensor gt_all_valid = gt_valid.all(/*dim=*/1);
    torch::Tensor inf = torch::full_like(
        gt_dist,
        std::numeric_limits<float>::infinity());
    torch::Tensor gt_min_dist =
        std::get<0>(torch::where(gt_valid, gt_dist, inf).min(/*dim=*/1));
    const float tau_surface =
        std::max(0.0f, sdf_params_.tsdf_prune_surface_band_vox_) * tsdfMetricVoxelSize();
    torch::Tensor half_diag = (0.5f * std::sqrt(3.0f)) * sizes_cpu;
    torch::Tensor floater_mask_cpu =
        (gt_all_valid & (gt_min_dist > (tau_surface + half_diag))).to(torch::kBool);

    const torch::Device dev = centers.device();
    torch::Tensor floater_mask =
        floater_mask_cpu.to(dev).to(torch::kBool);
    torch::Tensor orb_mask =
        normalizeBoolMaskOrZeros(voxel_model_->orbVoxelMask(), N, dev);
    torch::Tensor inactive_geo_mask =
        normalizeBoolMaskOrZeros(voxel_model_->inactiveGeoVoxelMask(), N, dev);
    torch::Tensor rgbd_fill_mask =
        normalizeBoolMaskOrZeros(voxel_model_->rgbdFillRenderHolesVoxelMask(), N, dev);

    // Source masks should be exclusive in current runs, but keep ORB as the
    // residual source for older/debug runs where masks may overlap.
    orb_mask = (orb_mask & (~inactive_geo_mask) & (~rgbd_fill_mask)).to(torch::kBool);

    auto colors_for_indices =
        [&](const torch::Tensor& idx_in,
            const std::array<float, 4>& fallback_rgba,
            bool color_by_source) -> torch::Tensor
    {
        const int64_t K = idx_in.defined() ? idx_in.numel() : 0;
        torch::Tensor colors = torch::zeros(
            {K, 4},
            torch::TensorOptions().dtype(torch::kFloat32).device(dev));
        if (K <= 0) {
            return colors.contiguous();
        }

        colors.index_put_({torch::indexing::Slice(), 0}, fallback_rgba[0]);
        colors.index_put_({torch::indexing::Slice(), 1}, fallback_rgba[1]);
        colors.index_put_({torch::indexing::Slice(), 2}, fallback_rgba[2]);
        colors.index_put_({torch::indexing::Slice(), 3}, fallback_rgba[3]);
        if (!color_by_source) {
            return colors.contiguous();
        }

        torch::Tensor idx = idx_in.to(dev).to(torch::kLong);
        torch::Tensor orb_sel = orb_mask.index_select(0, idx);
        torch::Tensor inactive_sel = inactive_geo_mask.index_select(0, idx);
        torch::Tensor rgbd_fill_sel = rgbd_fill_mask.index_select(0, idx);

        colors.index_put_({orb_sel, 0}, 0.1f);
        colors.index_put_({orb_sel, 1}, 0.8f);
        colors.index_put_({orb_sel, 2}, 1.0f);
        colors.index_put_({orb_sel, 3}, 0.7f);

        colors.index_put_({inactive_sel, 0}, 0.7f);
        colors.index_put_({inactive_sel, 1}, 0.45f);
        colors.index_put_({inactive_sel, 2}, 0.2f);
        colors.index_put_({inactive_sel, 3}, 0.7f);

        colors.index_put_({rgbd_fill_sel, 0}, 0.95f);
        colors.index_put_({rgbd_fill_sel, 1}, 0.25f);
        colors.index_put_({rgbd_fill_sel, 2}, 0.85f);
        colors.index_put_({rgbd_fill_sel, 3}, 0.7f);
        return colors.contiguous();
    };

    auto log_subset =
        [&](const torch::Tensor& mask_in,
            const std::string& entity_path,
            const std::array<float, 4>& fallback_rgba,
            bool color_by_source)
    {
        torch::Tensor subset_mask = normalizeBoolMaskOrZeros(mask_in, N, dev);
        torch::Tensor idx = subset_mask.nonzero().squeeze(1);

        torch::Tensor centers_sel;
        torch::Tensor sizes_sel;
        torch::Tensor colors_sel;
        if (!idx.defined() || idx.numel() <= 0) {
            centers_sel = torch::empty(
                {0, 3},
                torch::TensorOptions().dtype(torch::kFloat32).device(dev));
            sizes_sel = torch::empty(
                {0, 1},
                torch::TensorOptions().dtype(torch::kFloat32).device(dev));
            colors_sel = torch::empty(
                {0, 4},
                torch::TensorOptions().dtype(torch::kFloat32).device(dev));
        } else {
            torch::Tensor idx_dev = idx.to(dev).to(torch::kLong);
            centers_sel = centers.index_select(0, idx_dev).contiguous();
            sizes_sel = sizes_view.index_select(0, idx_dev).contiguous();
            colors_sel = colors_for_indices(idx_dev, fallback_rgba, color_by_source);
        }

        sv::RerunVisualizerBridge::instance().visualizeDebugVoxelBoxes(
            "floaters",
            centers_sel,
            sizes_sel,
            colors_sel,
            iteration,
            entity_path);
    };

    const std::string root = "world/floaters";
    log_subset(
        floater_mask,
        root,
        {0.8f, 0.8f, 0.8f, 0.45f},
        true);
    log_subset(
        (floater_mask & orb_mask).to(torch::kBool),
        root + "/source/orb",
        {0.1f, 0.8f, 1.0f, 0.7f},
        false);
    log_subset(
        (floater_mask & inactive_geo_mask).to(torch::kBool),
        root + "/source/inactive_geo_densify",
        {0.7f, 0.45f, 0.2f, 0.7f},
        false);
    log_subset(
        (floater_mask & rgbd_fill_mask).to(torch::kBool),
        root + "/source/rgbd_fill_render_holes",
        {0.95f, 0.25f, 0.85f, 0.7f},
        false);
}

void VoxelMapper::appendAndLogOrbRawMapPcdToRerun(
    const std::map<point3D_id_t, Point3D>& pcd,
    int iteration)
{
    if (pcd.empty()) {
        return;
    }

    std::vector<float> pts;
    std::vector<float> cols;
    pts.reserve(pcd.size() * 3);
    cols.reserve(pcd.size() * 3);
    for (const auto& kv : pcd) {
        const auto& P = kv.second;
        pts.push_back(static_cast<float>(P.xyz_(0)));
        pts.push_back(static_cast<float>(P.xyz_(1)));
        pts.push_back(static_cast<float>(P.xyz_(2)));
        cols.push_back(static_cast<float>(P.color_(0)));
        cols.push_back(static_cast<float>(P.color_(1)));
        cols.push_back(static_cast<float>(P.color_(2)));
    }

    auto points = torch::from_blob(
        pts.data(),
        {static_cast<int64_t>(pcd.size()), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
    auto colors = torch::from_blob(
        cols.data(),
        {static_cast<int64_t>(pcd.size()), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
    colors = normalizeRerunPointColors(colors);

    if (!mono_prior_state_.orb_raw_pcd_points_accum_cpu_.defined() || mono_prior_state_.orb_raw_pcd_points_accum_cpu_.numel() == 0) {
        mono_prior_state_.orb_raw_pcd_points_accum_cpu_ = points.contiguous();
        mono_prior_state_.orb_raw_pcd_colors_accum_cpu_ = colors.contiguous();
    } else {
        mono_prior_state_.orb_raw_pcd_points_accum_cpu_ =
            torch::cat({mono_prior_state_.orb_raw_pcd_points_accum_cpu_, points.contiguous()}, 0).contiguous();
        mono_prior_state_.orb_raw_pcd_colors_accum_cpu_ =
            torch::cat({mono_prior_state_.orb_raw_pcd_colors_accum_cpu_, colors.contiguous()}, 0).contiguous();
    }

    sv::RerunVisualizerBridge::instance().visualizePoints3D(
        mono_prior_state_.orb_raw_pcd_points_accum_cpu_,
        mono_prior_state_.orb_raw_pcd_colors_accum_cpu_,
        iteration,
        "world/orb/raw_pcd",
        0.015f);
    // std::cout << "[rerun/orb] appended_raw_pcd_points=" << pcd.size()
    //           << " total_raw_pcd_points=" << mono_prior_state_.orb_raw_pcd_points_accum_cpu_.size(0)
    //           << " entity=world/orb/raw_pcd"
    //           << " iter=" << iteration
    //           << std::endl;
}

void VoxelMapper::appendAndLogOrbRawPointBatchToRerun(
    const std::vector<float>& points_flat,
    const std::vector<float>& colors_flat,
    int iteration)
{
    const int64_t n_points = static_cast<int64_t>(points_flat.size() / 3);
    if (n_points <= 0 || colors_flat.size() < static_cast<size_t>(3 * n_points)) {
        return;
    }

    auto points = torch::from_blob(
        const_cast<float*>(points_flat.data()),
        {n_points, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
    auto colors = torch::from_blob(
        const_cast<float*>(colors_flat.data()),
        {n_points, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
    colors = normalizeRerunPointColors(colors);

    if (!mono_prior_state_.orb_raw_pcd_points_accum_cpu_.defined() || mono_prior_state_.orb_raw_pcd_points_accum_cpu_.numel() == 0) {
        mono_prior_state_.orb_raw_pcd_points_accum_cpu_ = points.contiguous();
        mono_prior_state_.orb_raw_pcd_colors_accum_cpu_ = colors.contiguous();
    } else {
        mono_prior_state_.orb_raw_pcd_points_accum_cpu_ =
            torch::cat({mono_prior_state_.orb_raw_pcd_points_accum_cpu_, points.contiguous()}, 0).contiguous();
        mono_prior_state_.orb_raw_pcd_colors_accum_cpu_ =
            torch::cat({mono_prior_state_.orb_raw_pcd_colors_accum_cpu_, colors.contiguous()}, 0).contiguous();
    }

    sv::RerunVisualizerBridge::instance().visualizePoints3D(
        mono_prior_state_.orb_raw_pcd_points_accum_cpu_,
        mono_prior_state_.orb_raw_pcd_colors_accum_cpu_,
        iteration,
        "world/orb/raw_pcd",
        0.015f);
    // std::cout << "[rerun/orb] appended_raw_pcd_points=" << n_points
    //           << " total_raw_pcd_points=" << mono_prior_state_.orb_raw_pcd_points_accum_cpu_.size(0)
    //           << " entity=world/orb/raw_pcd"
    //           << " iter=" << iteration
    //           << std::endl;
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

    namespace py = pybind11;
    torch::NoGradGuard no_grad;

    const auto& keyframes = scene_->keyframes();
    if (keyframes.empty()) return;

    auto centers_cpu = voxel_model_->voxCenter().detach().to(torch::kCPU).contiguous();
    if (!centers_cpu.defined() || centers_cpu.numel() == 0) return;

    const float current_svraster_voxel = std::max(1.0e-6f, tsdfMetricVoxelSize());
    const float voxel_length =
        std::max(current_svraster_voxel, std::max(1.0e-6f, sdf_params_.sdf_voxel_size_m_));
    const float sdf_trunc =
        std::max(1.0e-6f, sdf_params_.tsdf_density_init_trunc_vox_ * voxel_length);
    const float depth_trunc =
        (sdf_params_.svraster_tsdf_max_integration_distance_m_ > 0.0f)
            ? sdf_params_.svraster_tsdf_max_integration_distance_m_
            : kCommonEvalDepthTrunc;

    SparseTsdfVolume volume(voxel_length, sdf_trunc);

    py::gil_scoped_acquire gil;
    py::object py_svm = voxel_model_->svm();
    if (py_svm.is_none()) {
        std::cout << "[RERUN/reconstruction_mesh] skipped: Python voxel model unavailable.\n";
        return;
    }

    static py::module_ torch_mod = py::module_::import("torch");
    bool froze_geo = false;
    py::object inference_ctx = torch_mod.attr("inference_mode")();
    try
    {
        torch_mod.attr("cuda").attr("empty_cache")();
        inference_ctx.attr("__enter__")();

        if (py::hasattr(py_svm, "freeze_vox_geo"))
        {
            py_svm.attr("freeze_vox_geo")();
            froze_geo = true;
        }

        std::size_t frame_idx = 0;
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

            cv::Mat depth_for_fusion = depth_mat.clone();
            cv::Mat gt_depth_meters;
            if (getKeyframeDepthMetersForEval(pkf, image_height, image_width, gt_depth_meters))
            {
                for (int y = 0; y < image_height; ++y)
                {
                    const float* gt_ptr = gt_depth_meters.ptr<float>(y);
                    float* depth_ptr = depth_for_fusion.ptr<float>(y);
                    for (int x = 0; x < image_width; ++x)
                    {
                        const float gt_depth = gt_ptr[x];
                        if (!std::isfinite(gt_depth) || gt_depth <= 0.0f) {
                            depth_ptr[x] = 0.0f;
                        }
                    }
                }
            }

            cv::Mat filtered_depth = filterDepthOutliersLikeGaussianSlam(depth_for_fusion);
            volume.integrate(
                color_mat.clone(),
                filtered_depth,
                pkf->intr_,
                pkf->getPosef(),
                depth_trunc);

            ++frame_idx;
        }

        TriangleMeshRgb mesh = volume.extractMesh(rerun_params_.rerun_reconstruction_mesh_min_weight_);
        if (mesh.vertices.empty() || mesh.faces.empty())
        {
            inference_ctx.attr("__exit__")(py::none(), py::none(), py::none());
            if (froze_geo && py::hasattr(py_svm, "unfreeze_vox_geo")) {
                py_svm.attr("unfreeze_vox_geo")();
            }
            std::cout << "[RERUN/reconstruction_mesh] iter=" << iteration
                      << " empty mesh after fusing " << frame_idx << " keyframes.\n";
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
            inference_ctx.attr("__exit__")(py::none(), py::none(), py::none());
            if (froze_geo && py::hasattr(py_svm, "unfreeze_vox_geo")) {
                py_svm.attr("unfreeze_vox_geo")();
            }
            std::cout << "[RERUN/reconstruction_mesh] iter=" << iteration
                      << " skipped over-budget full-scene mesh"
                      << " verts=" << mesh.vertices.size()
                      << " faces=" << mesh.faces.size()
                      << " max_vertices=" << rerun_params_.rerun_reconstruction_mesh_max_vertices_
                      << " max_faces=" << rerun_params_.rerun_reconstruction_mesh_max_faces_
                      << " voxel_length=" << voxel_length
                      << " current_svraster_voxel=" << current_svraster_voxel
                      << " configured_tsdf_voxel=" << sdf_params_.sdf_voxel_size_m_
                      << ". Increase Mapper.tsdf_voxel_size_m for a coarser debug mesh, "
                      << "or set the run_reconstruction_mesh max limits to 0 to disable this guard.\n";
            return;
        }

        torch::Tensor vertices;
        torch::Tensor colors;
        torch::Tensor triangles;
        triangleMeshToTensors(mesh, vertices, colors, triangles);

        inference_ctx.attr("__exit__")(py::none(), py::none(), py::none());
        if (froze_geo && py::hasattr(py_svm, "unfreeze_vox_geo")) {
            py_svm.attr("unfreeze_vox_geo")();
            froze_geo = false;
        }

        sv::RerunVisualizerBridge::instance().visualizeDebugNvbloxMesh(
            "reconstruction_mesh",
            vertices,
            colors,
            triangles,
            iteration,
            "world/mesh");

        std::cout << "[RERUN/reconstruction_mesh] iter=" << iteration
                  << " kfs=" << frame_idx
                  << " verts=" << mesh.vertices.size()
                  << " faces=" << mesh.faces.size()
                  << " voxel_length=" << voxel_length
                  << " current_svraster_voxel=" << current_svraster_voxel
                  << " configured_tsdf_voxel=" << sdf_params_.sdf_voxel_size_m_
                  << " sdf_trunc=" << sdf_trunc
                  << " depth_trunc=" << depth_trunc
                  << " min_weight=" << rerun_params_.rerun_reconstruction_mesh_min_weight_
                  << " weld=" << static_cast<int>(rerun_params_.rerun_reconstruction_mesh_weld_vertices_)
                  << " max_vertices=" << rerun_params_.rerun_reconstruction_mesh_max_vertices_
                  << " max_faces=" << rerun_params_.rerun_reconstruction_mesh_max_faces_
                  << "\n";
    }
    catch (const py::error_already_set& e)
    {
        try
        {
            inference_ctx.attr("__exit__")(py::none(), py::none(), py::none());
            torch_mod.attr("cuda").attr("empty_cache")();
        }
        catch (...) {}
        if (froze_geo && py::hasattr(py_svm, "unfreeze_vox_geo")) {
            py_svm.attr("unfreeze_vox_geo")();
        }
        std::cerr << "[RERUN/reconstruction_mesh] Python error: " << e.what() << "\n";
    }
    catch (const std::exception& e)
    {
        try
        {
            inference_ctx.attr("__exit__")(py::none(), py::none(), py::none());
            torch_mod.attr("cuda").attr("empty_cache")();
        }
        catch (...) {}
        if (froze_geo && py::hasattr(py_svm, "unfreeze_vox_geo")) {
            py_svm.attr("unfreeze_vox_geo")();
        }
        std::cerr << "[RERUN/reconstruction_mesh] failed: " << e.what() << "\n";
    }
}

void VoxelMapper::logNvbloxReconstructionMeshToRerun(int iteration)
{
    if (!rerun_params_.enable_rerun_ || !rerun_params_.rerun_reconstruction_mesh_ ||
        sensor_type_ != RGBD || !sdf_mapper_) {
        return;
    }
    if (rerun_params_.rerun_reconstruction_mesh_interval_ <= 0 ||
        (iteration % rerun_params_.rerun_reconstruction_mesh_interval_) != 0) {
        return;
    }

    try
    {
        sdf_mapper_->updateColorMesh(nvblox::UpdateFullLayer::kNo);

        nvblox::CudaStreamOwning cuda_stream;
        std::shared_ptr<const nvblox::ColorMesh> mesh =
            sdf_mapper_->color_mesh_layer().getMesh(cuda_stream);
        cuda_stream.synchronize();

        torch::Tensor vertices;
        torch::Tensor colors;
        torch::Tensor triangles;
        nvbloxColorMeshToTensors(mesh, vertices, colors, triangles);
        if (!vertices.defined() || !triangles.defined() ||
            vertices.numel() == 0 || triangles.numel() == 0) {
            std::cout << "[RERUN/reconstruction_mesh/nvblox] iter=" << iteration
                      << " empty nvblox mesh.\n";
            return;
        }

        const std::size_t num_vertices =
            static_cast<std::size_t>(vertices.size(0));
        const std::size_t num_faces =
            static_cast<std::size_t>(triangles.size(0));
        const bool over_vertex_budget =
            rerun_params_.rerun_reconstruction_mesh_max_vertices_ > 0 &&
            num_vertices > rerun_params_.rerun_reconstruction_mesh_max_vertices_;
        const bool over_face_budget =
            rerun_params_.rerun_reconstruction_mesh_max_faces_ > 0 &&
            num_faces > rerun_params_.rerun_reconstruction_mesh_max_faces_;
        if (over_vertex_budget || over_face_budget) {
            std::cout << "[RERUN/reconstruction_mesh/nvblox] iter=" << iteration
                      << " skipped over-budget mesh"
                      << " verts=" << num_vertices
                      << " faces=" << num_faces
                      << " max_vertices=" << rerun_params_.rerun_reconstruction_mesh_max_vertices_
                      << " max_faces=" << rerun_params_.rerun_reconstruction_mesh_max_faces_
                      << "\n";
            return;
        }

        sv::RerunVisualizerBridge::instance().visualizeDebugNvbloxMesh(
            "reconstruction_mesh",
            vertices,
            colors,
            triangles,
            iteration,
            "world/nvblox_mesh/live");

        std::cout << "[RERUN/reconstruction_mesh/nvblox] iter=" << iteration
                  << " verts=" << num_vertices
                  << " faces=" << num_faces
                  << " voxel_size=" << sdf_mapper_->voxel_size_m()
                  << "\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "[RERUN/reconstruction_mesh/nvblox] failed: "
                  << e.what() << "\n";
    }
}

void VoxelMapper::logWholeRunNvbloxMeshToRerun(int iteration)
{
    if (!rerun_params_.enable_rerun_ || !rerun_params_.run_whole_run_ ||
        sensor_type_ != RGBD || !sdf_mapper_) {
        return;
    }

    try
    {
        sdf_mapper_->updateColorMesh(nvblox::UpdateFullLayer::kNo);

        nvblox::CudaStreamOwning cuda_stream;
        std::shared_ptr<const nvblox::ColorMesh> mesh =
            sdf_mapper_->color_mesh_layer().getMesh(cuda_stream);
        cuda_stream.synchronize();

        torch::Tensor vertices;
        torch::Tensor colors;
        torch::Tensor triangles;
        nvbloxColorMeshToTensors(mesh, vertices, colors, triangles);
        if (!vertices.defined() || !triangles.defined() ||
            vertices.numel() == 0 || triangles.numel() == 0) {
            return;
        }

        sv::RerunVisualizerBridge::instance().visualizeDebugNvbloxMesh(
            "whole_run",
            vertices,
            colors,
            triangles,
            iteration,
            "world/nvblox_mesh");
    }
    catch (const std::exception& e)
    {
        std::cerr << "[RERUN/whole_run/nvblox] failed: "
                  << e.what() << "\n";
    }
}
