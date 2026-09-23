#include "include_voxel/voxel_mapper.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <ATen/ops/unique_dim.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "include_voxel/svrecon_marching_cubes_table.h"
#include "third_party/tinyply/tinyply.h"

namespace {

struct TriangleMeshRgb
{
    std::vector<Eigen::Vector3f> vertices;
    std::vector<std::array<uint32_t, 3>> faces;
    std::vector<std::array<uint8_t, 3>> colors;
};

bool saveTriangleMeshPly(
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

std::pair<torch::Tensor, torch::Tensor> marchingCubesVoxels(
    const torch::Tensor& unit_val,
    const torch::Tensor& unit_xyz,
    float iso)
{
    auto mask =
        (unit_val > iso).any(1) &
        (unit_val < iso).any(1) &
        ~torch::isnan(unit_val).any(1);
    auto filter_idx = torch::nonzero(mask).view({-1}).to(torch::kLong);
    if (filter_idx.numel() == 0) {
        return {
            torch::empty({0, 3}, unit_xyz.options()),
            torch::empty({0, 3}, unit_val.options().dtype(torch::kLong))};
    }

    auto values = unit_val.index_select(0, filter_idx).contiguous();
    auto xyz = unit_xyz.index_select(0, filter_idx).contiguous();
    const auto dev = values.device();
    const int64_t n_vox = values.size(0);

    static constexpr int64_t kEdgePairs[12][2] = {
        {0, 1}, {1, 5}, {5, 4}, {4, 0},
        {2, 3}, {3, 7}, {7, 6}, {6, 2},
        {0, 2}, {1, 3}, {5, 7}, {4, 6}};
    auto edges = torch::from_blob(
                     const_cast<int64_t*>(&kEdgePairs[0][0]),
                     {12, 2},
                     torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
                     .clone()
                     .to(dev);
    auto edge_a = edges.index({torch::indexing::Slice(), 0});
    auto edge_b = edges.index({torch::indexing::Slice(), 1});
    auto value_a = values.index_select(1, edge_a);
    auto value_b = values.index_select(1, edge_b);
    auto xyz_a = xyz.index_select(1, edge_a);
    auto xyz_b = xyz.index_select(1, edge_b);
    auto denom = value_b - value_a;
    auto ratio = (iso - value_a) / denom;
    ratio = torch::where(
        denom.abs() < 1.0e-9f,
        torch::full_like(ratio, 0.5f),
        ratio);
    auto edge_vertices =
        (xyz_a + ratio.unsqueeze(2) * (xyz_b - xyz_a)).contiguous();

    static constexpr int64_t kCubeIndexBases[8] = {1, 2, 4, 8, 16, 32, 64, 128};
    auto bases = torch::from_blob(
                     const_cast<int64_t*>(kCubeIndexBases),
                     {8},
                     torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
                     .clone()
                     .to(dev);
    auto cube_idx =
        ((values < iso).to(torch::kLong) * bases.view({1, 8})).sum(1).to(torch::kLong);
    auto tri_table = torch::from_blob(
                         const_cast<int64_t*>(&svrecon_mesh::kTriangleTable[0][0]),
                         {256, 15},
                         torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
                         .clone()
                         .to(dev);
    auto tri_idx = tri_table.index_select(0, cube_idx).contiguous();
    auto gather_idx = tri_idx.clamp_min(0).unsqueeze(2).expand({n_vox, 15, 3});
    auto faces_xyz = edge_vertices.gather(1, gather_idx);
    faces_xyz = faces_xyz.index({tri_idx != -1}).view({-1, 3}).contiguous();
    if (faces_xyz.numel() == 0) {
        return {
            torch::empty({0, 3}, unit_xyz.options()),
            torch::empty({0, 3}, unit_val.options().dtype(torch::kLong))};
    }

    auto unique_result = at::unique_dim(
        faces_xyz,
        /*dim=*/0,
        /*sorted=*/true,
        /*return_inverse=*/true,
        /*return_counts=*/false);
    auto vertices = std::get<0>(unique_result).contiguous();
    auto faces = std::get<1>(unique_result).view({-1, 3}).to(torch::kLong).contiguous();
    return {vertices, faces};
}

std::pair<torch::Tensor, torch::Tensor> marchingCubesGrid(
    const torch::Tensor& grid_pts_val,
    const torch::Tensor& grid_pts_xyz,
    const torch::Tensor& vox_key,
    float iso = 0.0f)
{
    constexpr int64_t kChunkSize = 1000000;
    std::vector<torch::Tensor> vertices_chunks;
    std::vector<torch::Tensor> faces_chunks;
    int64_t vertex_offset = 0;

    for (int64_t begin = 0; begin < vox_key.size(0); begin += kChunkSize) {
        const int64_t count = std::min(kChunkSize, vox_key.size(0) - begin);
        auto key = vox_key.narrow(0, begin, count).to(torch::kLong).contiguous();
        auto unit_val = grid_pts_val.index({key}).reshape({count, 8}).contiguous();
        auto unit_xyz = grid_pts_xyz.index({key}).reshape({count, 8, 3}).contiguous();
        auto [vertices, faces] = marchingCubesVoxels(unit_val, unit_xyz, iso);
        if (faces.numel() == 0) continue;
        vertices_chunks.push_back(vertices);
        faces_chunks.push_back(faces + vertex_offset);
        vertex_offset += vertices.size(0);
    }

    if (vertices_chunks.empty()) {
        return {
            torch::empty({0, 3}, grid_pts_xyz.options()),
            torch::empty({0, 3}, vox_key.options().dtype(torch::kLong))};
    }
    if (vertices_chunks.size() == 1) {
        return {vertices_chunks.front(), faces_chunks.front()};
    }

    auto all_vertices = torch::cat(vertices_chunks, 0).contiguous();
    auto all_faces = torch::cat(faces_chunks, 0).to(torch::kLong).contiguous();
    auto faces_xyz = all_vertices.index({all_faces}).reshape({-1, 3}).contiguous();
    auto unique_result = at::unique_dim(
        faces_xyz,
        /*dim=*/0,
        /*sorted=*/true,
        /*return_inverse=*/true,
        /*return_counts=*/false);
    return {
        std::get<0>(unique_result).contiguous(),
        std::get<1>(unique_result).view({-1, 3}).to(torch::kLong).contiguous()};
}

TriangleMeshRgb meshFromTensors(
    const torch::Tensor& vertices,
    const torch::Tensor& faces)
{
    TriangleMeshRgb mesh;
    auto vertices_cpu = vertices.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto faces_cpu = faces.detach().to(torch::kCPU).to(torch::kLong).contiguous();
    if (vertices_cpu.numel() == 0 || faces_cpu.numel() == 0) return mesh;

    auto verts = vertices_cpu.accessor<float, 2>();
    auto tris = faces_cpu.accessor<int64_t, 2>();
    mesh.vertices.reserve(static_cast<size_t>(vertices_cpu.size(0)));
    mesh.faces.reserve(static_cast<size_t>(faces_cpu.size(0)));
    for (int64_t i = 0; i < vertices_cpu.size(0); ++i) {
        mesh.vertices.emplace_back(verts[i][0], verts[i][1], verts[i][2]);
    }
    for (int64_t i = 0; i < faces_cpu.size(0); ++i) {
        mesh.faces.push_back({
            static_cast<uint32_t>(tris[i][0]),
            static_cast<uint32_t>(tris[i][1]),
            static_cast<uint32_t>(tris[i][2])});
    }
    return mesh;
}

torch::Tensor sampleBilinear(
    const torch::Tensor& image,
    const torch::Tensor& normalized_uv)
{
    const int64_t height = image.size(-2);
    const int64_t width = image.size(-1);
    auto options = torch::nn::functional::GridSampleFuncOptions()
                       .mode(torch::kBilinear)
                       .padding_mode(torch::kZeros)
                       .align_corners(false);
    return torch::nn::functional::grid_sample(
               image.contiguous().view({1, 1, height, width}),
               normalized_uv.contiguous().view({1, 1, -1, 2}),
               options)
        .flatten();
}

torch::Tensor sampleBilinearChannels(
    const torch::Tensor& image,
    const torch::Tensor& normalized_uv)
{
    const int64_t height = image.size(-2);
    const int64_t width = image.size(-1);
    const int64_t channels = image.numel() / (height * width);
    auto options = torch::nn::functional::GridSampleFuncOptions()
                       .mode(torch::kBilinear)
                       .padding_mode(torch::kZeros)
                       .align_corners(false);
    return torch::nn::functional::grid_sample(
               image.contiguous().view({1, channels, height, width}),
               normalized_uv.contiguous().view({1, 1, -1, 2}),
               options)
        .squeeze(0)
        .squeeze(1)
        .transpose(0, 1)
        .contiguous();
}

struct RenderedMeshView
{
    sv::MiniCam cam;
    torch::Tensor depth;
    torch::Tensor alpha;
};

struct RenderedMeshAllocationView
{
    cv::Mat depth;
    std::vector<float> intr;
    Sophus::SE3f Tcw;
};

struct SparseTsdfKey
{
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator==(const SparseTsdfKey& other) const noexcept
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct SparseTsdfKeyHash
{
    std::size_t operator()(const SparseTsdfKey& key) const noexcept
    {
        const std::uint64_t x = static_cast<std::uint32_t>(key.x) * 73856093u;
        const std::uint64_t y = static_cast<std::uint32_t>(key.y) * 19349663u;
        const std::uint64_t z = static_cast<std::uint32_t>(key.z) * 83492791u;
        return static_cast<std::size_t>(x ^ y ^ z);
    }
};

struct SparseRenderedTsdfGridData
{
    torch::Tensor grid_xyz;
    torch::Tensor voxel_keys;
    torch::Tensor cell_indices;
};

class SparseRenderedTsdfGrid
{
public:
    SparseRenderedTsdfGrid(float voxel_length, float sdf_trunc)
        : voxel_length_(voxel_length), sdf_trunc_(sdf_trunc)
    {}

    bool allocateFromKeyframe(
        const cv::Mat& depth_map,
        const std::vector<float>& intr,
        const Sophus::SE3f& Tcw,
        float depth_max)
    {
        CV_Assert(depth_map.type() == CV_32FC1);
        if (intr.size() < 4) {
            throw std::runtime_error(
                "SparseRenderedTsdfGrid::allocateFromKeyframe: expected fx, fy, cx, cy");
        }
        const std::size_t size_before = grid_keys_.size();

        const float fx = intr[0];
        const float fy = intr[1];
        const float cx = intr[2];
        const float cy = intr[3];
        const Sophus::SE3f Twc = Tcw.inverse();
        const Eigen::Matrix3f Rwc = Twc.rotationMatrix();
        const Eigen::Vector3f twc = Twc.translation();

        for (int v = 0; v < depth_map.rows; ++v) {
            const float ry = (static_cast<float>(v) - cy) / fy;
            const float* depth_ptr = depth_map.ptr<float>(v);
            for (int u = 0; u < depth_map.cols; ++u) {
                const float depth = depth_ptr[u];
                if (!std::isfinite(depth) || depth <= 0.0f || depth > depth_max) {
                    continue;
                }

                const float rx = (static_cast<float>(u) - cx) / fx;
                const float z_min = std::max(0.0f, depth - sdf_trunc_);
                const float z_max = std::min(depth_max, depth + sdf_trunc_);
                SparseTsdfKey last_key{
                    std::numeric_limits<int>::min(),
                    std::numeric_limits<int>::min(),
                    std::numeric_limits<int>::min()};

                for (float z = z_min; z <= z_max + 1.0e-6f; z += voxel_length_) {
                    const Eigen::Vector3f p_cam(rx * z, ry * z, z);
                    const Eigen::Vector3f p_world = Rwc * p_cam + twc;
                    const Eigen::Vector3i idx =
                        (p_world / voxel_length_).array().floor().cast<int>().matrix();
                    const SparseTsdfKey key{idx.x(), idx.y(), idx.z()};
                    if (key == last_key) continue;
                    last_key = key;

                    grid_keys_.insert(key);
                }
            }
        }
        return grid_keys_.size() > size_before;
    }

    std::size_t pointCount() const noexcept
    {
        return grid_keys_.size();
    }

    SparseRenderedTsdfGridData buildGrid(
        const torch::Device& device) const
    {
        std::unordered_map<SparseTsdfKey, int64_t, SparseTsdfKeyHash> point_index;
        point_index.reserve(grid_keys_.size());
        std::vector<float> xyz;
        xyz.reserve(grid_keys_.size() * 3);

        for (const auto& key : grid_keys_) {
            const int64_t index = static_cast<int64_t>(point_index.size());
            point_index.emplace(key, index);
            xyz.push_back((static_cast<float>(key.x) + 0.5f) * voxel_length_);
            xyz.push_back((static_cast<float>(key.y) + 0.5f) * voxel_length_);
            xyz.push_back((static_cast<float>(key.z) + 0.5f) * voxel_length_);
        }

        static constexpr int kCornerOffsets[8][3] = {
            {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
            {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};
        std::unordered_set<SparseTsdfKey, SparseTsdfKeyHash> candidate_cells;
        candidate_cells.reserve(point_index.size() * 8);
        for (const auto& [key, index] : point_index) {
            (void)index;
            for (int dz = -1; dz <= 0; ++dz) {
                for (int dy = -1; dy <= 0; ++dy) {
                    for (int dx = -1; dx <= 0; ++dx) {
                        candidate_cells.insert({key.x + dx, key.y + dy, key.z + dz});
                    }
                }
            }
        }

        std::vector<int64_t> voxel_keys;
        voxel_keys.reserve(candidate_cells.size() * 8);
        std::vector<int32_t> cell_indices;
        cell_indices.reserve(candidate_cells.size() * 3);
        for (const auto& cell : candidate_cells) {
            std::array<int64_t, 8> corners{};
            bool complete = true;
            for (int i = 0; i < 8; ++i) {
                const SparseTsdfKey key{
                    cell.x + kCornerOffsets[i][0],
                    cell.y + kCornerOffsets[i][1],
                    cell.z + kCornerOffsets[i][2]};
                const auto it = point_index.find(key);
                if (it == point_index.end()) {
                    complete = false;
                    break;
                }
                corners[i] = it->second;
            }
            if (!complete) continue;
            voxel_keys.insert(voxel_keys.end(), corners.begin(), corners.end());
            cell_indices.push_back(cell.x);
            cell_indices.push_back(cell.y);
            cell_indices.push_back(cell.z);
        }

        auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        auto cpu_long = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
        auto cpu_int = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
        if (xyz.empty() || voxel_keys.empty()) {
            return SparseRenderedTsdfGridData{
                torch::empty({0, 3}, cpu_float).to(device),
                torch::empty({0, 8}, cpu_long).to(device),
                torch::empty({0, 3}, cpu_int)};
        }
        auto xyz_tensor = torch::from_blob(
                              xyz.data(),
                              {static_cast<int64_t>(xyz.size() / 3), 3},
                              cpu_float)
                              .clone()
                              .to(device);
        auto key_tensor = torch::from_blob(
                              voxel_keys.data(),
                              {static_cast<int64_t>(voxel_keys.size() / 8), 8},
                              cpu_long)
                              .clone()
                              .to(device);
        auto cell_tensor = torch::from_blob(
                               cell_indices.data(),
                               {static_cast<int64_t>(cell_indices.size() / 3), 3},
                               cpu_int)
                               .clone();
        return SparseRenderedTsdfGridData{
            xyz_tensor.contiguous(),
            key_tensor.contiguous(),
            cell_tensor.contiguous()};
    }

private:
    float voxel_length_ = 0.05f;
    float sdf_trunc_ = 0.4f;
    std::unordered_set<SparseTsdfKey, SparseTsdfKeyHash> grid_keys_;
};

torch::Tensor projectNormalized(
    const torch::Tensor& points,
    const sv::MiniCam& cam)
{
    auto w2c = cam.w2c.to(points.device()).to(torch::kFloat32).contiguous();
    auto cam_xyz =
        torch::matmul(
            points,
            w2c.index({torch::indexing::Slice(0, 3),
                       torch::indexing::Slice(0, 3)}).transpose(0, 1)) +
        w2c.index({torch::indexing::Slice(0, 3), 3}).view({1, 3});
    auto z = cam_xyz.index({torch::indexing::Slice(), 2});
    return torch::stack({
        cam_xyz.index({torch::indexing::Slice(), 0}) /
                z / std::max(1.0e-8f, cam.tanfovx) +
            (2.0f * cam.cx / static_cast<float>(cam.width) - 1.0f),
        cam_xyz.index({torch::indexing::Slice(), 1}) /
                z / std::max(1.0e-8f, cam.tanfovy) +
            (2.0f * cam.cy / static_cast<float>(cam.height) - 1.0f)},
        1).contiguous();
}

struct FusedRenderedTsdf
{
    torch::Tensor tsdf;
    torch::Tensor keyframe_support;
};

struct RenderedTsdfSurfaceData
{
    torch::Tensor grid_xyz;
    torch::Tensor voxel_keys;
    torch::Tensor cell_indices;
    torch::Tensor fused_tsdf;
    torch::Tensor observed_cell_mask;
    torch::Tensor surface_cell_mask;
    int64_t rendered_view_count = 0;
    float requested_metric_voxel_size = 0.0f;
    float model_to_metric_scale = 1.0f;
    float voxel_size = 0.0f;
    float metric_voxel_size = 0.0f;
    float truncation = 0.0f;
    float min_keyframe_weight = 0.0f;
    float alpha_threshold = 0.0f;
    bool scale_aware = false;
    bool coverage_adapted = false;
};

FusedRenderedTsdf fuseRenderedTsdfWithSupport(
    const torch::Tensor& grid_xyz,
    const std::vector<RenderedMeshView>& views,
    float trunc_dist,
    float crop_border,
    float alpha_thres,
    bool use_unit_keyframe_weight)
{
    constexpr int64_t kPointChunk = 500000;
    std::vector<torch::Tensor> tsdf_chunks;
    std::vector<torch::Tensor> support_chunks;
    tsdf_chunks.reserve(
        static_cast<size_t>((grid_xyz.size(0) + kPointChunk - 1) / kPointChunk));
    support_chunks.reserve(tsdf_chunks.capacity());

    for (int64_t begin = 0; begin < grid_xyz.size(0); begin += kPointChunk) {
        const int64_t count = std::min(kPointChunk, grid_xyz.size(0) - begin);
        auto xyz = grid_xyz.narrow(0, begin, count).contiguous();
        auto weight = torch::zeros({count}, xyz.options().dtype(torch::kFloat32));
        auto keyframe_support = torch::zeros_like(weight);
        auto weighted_tsdf = torch::zeros_like(weight);

        for (const auto& view : views) {
            auto uv = projectNormalized(xyz, view.cam);
            auto camera_position =
                view.cam.position.to(xyz.device()).to(torch::kFloat32).view({1, 3});
            auto camera_lookat =
                view.cam.lookat.to(xyz.device()).to(torch::kFloat32).view({3, 1});
            auto xyz_depth =
                torch::matmul(xyz - camera_position, camera_lookat).flatten();
            auto projected =
                (uv.abs() <= (1.0f - crop_border)).all(1) & (xyz_depth > 0.0f);
            auto valid_idx = torch::nonzero(projected).view({-1}).to(torch::kLong);
            if (valid_idx.numel() == 0) continue;

            auto valid_uv = uv.index_select(0, valid_idx);
            auto sampled_depth = sampleBilinear(view.depth, valid_uv);
            auto valid_depth = xyz_depth.index_select(0, valid_idx);
            auto depth_idx = torch::nonzero(
                                 torch::isfinite(sampled_depth) & (sampled_depth > 0.0f))
                                 .view({-1})
                                 .to(torch::kLong);
            if (depth_idx.numel() == 0) continue;
            valid_idx = valid_idx.index_select(0, depth_idx);
            valid_uv = valid_uv.index_select(0, depth_idx);
            sampled_depth = sampled_depth.index_select(0, depth_idx);
            valid_depth = valid_depth.index_select(0, depth_idx);
            auto valid_sdf = (sampled_depth - valid_depth) / trunc_dist;

            auto trunc_idx = torch::nonzero(valid_sdf >= -1.0f)
                                 .view({-1})
                                 .to(torch::kLong);
            if (trunc_idx.numel() == 0) continue;
            valid_idx = valid_idx.index_select(0, trunc_idx);
            valid_uv = valid_uv.index_select(0, trunc_idx);
            valid_sdf = valid_sdf.index_select(0, trunc_idx).clamp(-1.0f, 1.0f);

            auto valid_alpha = sampleBilinear(view.alpha, valid_uv);
            auto alpha_idx = torch::nonzero(valid_alpha >= alpha_thres)
                                 .view({-1})
                                 .to(torch::kLong);
            if (alpha_idx.numel() == 0) continue;
            valid_idx = valid_idx.index_select(0, alpha_idx);
            valid_sdf = valid_sdf.index_select(0, alpha_idx);
            auto integration_weight = use_unit_keyframe_weight
                ? torch::ones_like(valid_sdf)
                : valid_alpha.index_select(0, alpha_idx);

            weight.index_add_(0, valid_idx, integration_weight);
            keyframe_support.index_add_(0, valid_idx, torch::ones_like(valid_sdf));
            weighted_tsdf.index_add_(
                0, valid_idx, integration_weight * valid_sdf);
        }
        auto nan = torch::full_like(weight, std::numeric_limits<float>::quiet_NaN());
        tsdf_chunks.push_back(torch::where(weight > 0.0f, weighted_tsdf / weight, nan));
        support_chunks.push_back(keyframe_support.contiguous());
    }
    return {
        torch::cat(tsdf_chunks, 0).contiguous(),
        torch::cat(support_chunks, 0).contiguous()};
}

torch::Tensor fuseRenderedTsdf(
    const torch::Tensor& grid_xyz,
    const std::vector<RenderedMeshView>& views,
    float trunc_dist,
    float crop_border,
    float alpha_thres)
{
    return fuseRenderedTsdfWithSupport(
               grid_xyz,
               views,
               trunc_dist,
               crop_border,
               alpha_thres,
               /*use_unit_keyframe_weight=*/false)
        .tsdf;
}

RenderedTsdfSurfaceData buildRenderedTsdfSurfaceData(
    const std::shared_ptr<sv::VoxelModel>& voxel_model,
    const std::map<std::size_t, std::shared_ptr<VoxelKeyframe>>& keyframes,
    const std::map<sv::camera_id_t, torch::Tensor>& undistort_masks,
    const VoxelRerunParameters& params,
    std::mutex& render_mutex,
    float depth_max,
    std::optional<float> model_to_metric_scale = std::nullopt)
{
    if (!voxel_model) {
        throw std::runtime_error(
            "buildRenderedTsdfSurfaceData: voxel model is not initialized");
    }
    if (keyframes.empty()) {
        throw std::runtime_error(
            "buildRenderedTsdfSurfaceData: no keyframes available");
    }

    torch::NoGradGuard no_grad;
    const float requested_metric_voxel_size = sv::kRenderedMeshVoxelSizeM;
    const bool scale_aware =
        params.rendered_mesh_scale_aware_ &&
        model_to_metric_scale.has_value() &&
        std::isfinite(*model_to_metric_scale) &&
        *model_to_metric_scale > 1.0e-6f;
    const float metric_scale = scale_aware ? *model_to_metric_scale : 1.0f;
    float voxel_length = requested_metric_voxel_size / metric_scale;
    const float min_keyframe_weight = sv::kRenderedMeshMinWeight;
    depth_max = std::max(1.0e-6f, depth_max);
    const float alpha_threshold = sv::kSvreconMeshAlphaThreshold;
    std::unique_lock<std::mutex> render_lock(render_mutex);
    bool froze_geo = false;
    auto unfreeze_geo = [&]() {
        if (froze_geo) {
            voxel_model->unfreezeVoxGeo();
            froze_geo = false;
        }
    };

    try {
        c10::cuda::CUDACachingAllocator::emptyCache();
        voxel_model->freezeVoxGeo();
        froze_geo = true;

        std::vector<RenderedMeshView> views;
        views.reserve(keyframes.size());
        std::vector<RenderedMeshAllocationView> allocation_views;
        allocation_views.reserve(keyframes.size());
        for (const auto& [kfid, pkf] : keyframes) {
            if (!pkf || !pkf->set_pose_ || pkf->intr_.size() < 4 ||
                pkf->image_height_ <= 0 || pkf->image_width_ <= 0) {
                continue;
            }
            const int height = pkf->image_height_;
            const int width = pkf->image_width_;
            const auto cam = pkf->toMiniCam(height, width);

            sv::RenderOpts render_opts;
            render_opts.output_depth = true;
            render_opts.output_T = true;
            auto render_pkg = voxel_model->render(
                cam,
                height,
                width,
                torch::Tensor(),
                nullptr,
                false,
                std::nullopt,
                true,
                false,
                true,
                false,
                false,
                render_opts);

            auto depth_it = render_pkg.find("raw_depth");
            auto transmittance_it = render_pkg.find("raw_T");
            if (depth_it == render_pkg.end() ||
                transmittance_it == render_pkg.end() ||
                !depth_it->second.defined() ||
                !transmittance_it->second.defined()) {
                std::cout
                    << "[mesh/rendered-TSDF-fixed] missing depth/transmittance for kf="
                    << kfid << ", skipping.\n";
                continue;
            }

            auto raw_depth =
                depth_it->second.detach().to(torch::kFloat32).contiguous();
            if (raw_depth.dim() == 4 && raw_depth.size(0) == 1) {
                raw_depth = raw_depth.squeeze(0);
            }
            if (raw_depth.dim() != 3 || raw_depth.size(0) < 3) {
                std::cout
                    << "[mesh/rendered-TSDF-fixed] expected SVRecon depth channels for kf="
                    << kfid << ", got " << raw_depth.sizes()
                    << ", skipping.\n";
                continue;
            }

            const int64_t depth_channel =
                params.svrecon_mesh_use_mean_depth_ ? 0 : 2;
            auto rendered_depth =
                raw_depth.index({depth_channel}).contiguous();
            auto rendered_transmittance =
                transmittance_it->second.detach()
                    .to(torch::kFloat32)
                    .contiguous();
            if (rendered_transmittance.dim() == 3 &&
                rendered_transmittance.size(0) == 1) {
                rendered_transmittance = rendered_transmittance.squeeze(0);
            }
            if (rendered_transmittance.dim() != 2 ||
                rendered_transmittance.size(0) != height ||
                rendered_transmittance.size(1) != width) {
                throw std::runtime_error(
                    "buildRenderedTsdfSurfaceData: invalid transmittance shape");
            }
            auto rendered_alpha =
                (1.0f - rendered_transmittance).clamp(0.0f, 1.0f);
            const auto valid_surface =
                torch::isfinite(rendered_depth) &
                torch::isfinite(rendered_alpha) &
                (rendered_depth > 0.0f) &
                (rendered_alpha >= alpha_threshold);
            rendered_depth = torch::where(
                valid_surface,
                rendered_depth,
                torch::zeros_like(rendered_depth));

            const auto mask_it = undistort_masks.find(pkf->camera_id_);
            if (mask_it != undistort_masks.end()) {
                auto mask = mask_it->second.to(rendered_depth.device());
                if (mask.dim() == 3) mask = mask.index({0});
                if (mask.dim() == 2 &&
                    mask.size(0) == width && mask.size(1) == height) {
                    mask = mask.transpose(0, 1);
                }
                if (mask.dim() != 2 || mask.size(0) != height ||
                    mask.size(1) != width) {
                    throw std::runtime_error(
                        "buildRenderedTsdfSurfaceData: invalid undistortion mask shape");
                }
                rendered_depth = rendered_depth * mask.to(torch::kFloat32);
            }

            auto depth_cpu = rendered_depth.contiguous().to(torch::kCPU);
            const cv::Mat depth_mat(
                height,
                width,
                CV_32FC1,
                depth_cpu.data_ptr<float>());
            allocation_views.push_back(RenderedMeshAllocationView{
                depth_mat.clone(), pkf->intr_, pkf->getPosef()});

            views.push_back(RenderedMeshView{
                cam,
                rendered_depth,
                rendered_alpha});
        }

        if (views.empty()) {
            unfreeze_geo();
            throw std::runtime_error(
                "buildRenderedTsdfSurfaceData: no valid rendered depth views");
        }

        const std::size_t max_grid_points =
            params.rendered_mesh_max_grid_points_;
        bool coverage_adapted = false;
        std::unique_ptr<SparseRenderedTsdfGrid> support_grid;
        for (int attempt = 0; attempt < 8; ++attempt) {
            const float sdf_trunc = std::max(
                voxel_length,
                sv::kRenderedMeshTruncVox * voxel_length);
            support_grid = std::make_unique<SparseRenderedTsdfGrid>(
                voxel_length, sdf_trunc);
            for (const auto& allocation_view : allocation_views) {
                support_grid->allocateFromKeyframe(
                    allocation_view.depth,
                    allocation_view.intr,
                    allocation_view.Tcw,
                    depth_max);
            }
            const std::size_t point_count = support_grid->pointCount();
            if (max_grid_points == 0 || point_count <= max_grid_points) {
                break;
            }

            const float coarsen = std::clamp(
                1.02f * std::sqrt(
                    static_cast<float>(point_count) /
                    static_cast<float>(max_grid_points)),
                1.05f,
                4.0f);
            voxel_length *= coarsen;
            coverage_adapted = true;
            std::cout
                << "[mesh/rendered-TSDF-fixed] grid budget exceeded: points="
                << point_count << " max=" << max_grid_points
                << " coarsening_voxel_size=" << voxel_length << "\n";
        }

        if (!support_grid ||
            (max_grid_points > 0 && support_grid->pointCount() > max_grid_points)) {
            unfreeze_geo();
            throw std::runtime_error(
                "buildRenderedTsdfSurfaceData: could not satisfy sparse-grid point budget");
        }

        const float sdf_trunc = std::max(
            voxel_length,
            sv::kRenderedMeshTruncVox * voxel_length);

        auto grid = support_grid->buildGrid(
            voxel_model->geoGridPts().device());
        if (grid.grid_xyz.numel() == 0 ||
            grid.voxel_keys.numel() == 0 || views.empty()) {
            unfreeze_geo();
            throw std::runtime_error(
                "buildRenderedTsdfSurfaceData: rendered depths allocated no complete TSDF cells");
        }

        auto fused = fuseRenderedTsdfWithSupport(
            grid.grid_xyz,
            views,
            sdf_trunc,
            /*crop_border=*/0.0f,
            alpha_threshold,
            /*use_unit_keyframe_weight=*/true);
        auto voxel_sdf = fused.tsdf.index({grid.voxel_keys});
        auto voxel_support =
            fused.keyframe_support.index({grid.voxel_keys});
        auto observed_cell_mask =
            torch::isfinite(voxel_sdf).all(1) &
            (std::get<0>(voxel_support.min(1)) >= min_keyframe_weight);
        auto surface_cell_mask =
            observed_cell_mask &
            (std::get<0>(voxel_sdf.min(1)) < 0.0f) &
            (std::get<0>(voxel_sdf.max(1)) > 0.0f);

        RenderedTsdfSurfaceData result;
        result.grid_xyz = std::move(grid.grid_xyz);
        result.voxel_keys = std::move(grid.voxel_keys);
        result.cell_indices = std::move(grid.cell_indices);
        result.fused_tsdf = std::move(fused.tsdf);
        result.observed_cell_mask = observed_cell_mask.contiguous();
        result.surface_cell_mask = surface_cell_mask.contiguous();
        result.rendered_view_count = static_cast<int64_t>(views.size());
        result.requested_metric_voxel_size = requested_metric_voxel_size;
        result.model_to_metric_scale = metric_scale;
        result.voxel_size = voxel_length;
        result.metric_voxel_size = voxel_length * metric_scale;
        result.truncation = sdf_trunc;
        result.min_keyframe_weight = min_keyframe_weight;
        result.alpha_threshold = alpha_threshold;
        result.scale_aware = scale_aware;
        result.coverage_adapted = coverage_adapted;
        unfreeze_geo();
        return result;
    } catch (...) {
        unfreeze_geo();
        throw;
    }
}

void setMeshColors(TriangleMeshRgb& mesh, const torch::Tensor& colors)
{
    auto colors_cpu =
        (colors.detach().to(torch::kCPU).to(torch::kFloat32).clamp(0.0f, 1.0f) * 255.0f)
            .round()
            .to(torch::kUInt8)
            .contiguous();
    if (colors_cpu.dim() != 2 || colors_cpu.size(0) != static_cast<int64_t>(mesh.vertices.size()) ||
        colors_cpu.size(1) != 3) {
        return;
    }
    auto acc = colors_cpu.accessor<uint8_t, 2>();
    mesh.colors.reserve(mesh.vertices.size());
    for (int64_t i = 0; i < colors_cpu.size(0); ++i) {
        mesh.colors.push_back({acc[i][0], acc[i][1], acc[i][2]});
    }
}

} // namespace
