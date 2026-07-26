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

    std::pair<torch::Tensor, torch::Tensor> buildGrid(
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
        }

        auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        auto cpu_long = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
        if (xyz.empty() || voxel_keys.empty()) {
            return {
                torch::empty({0, 3}, cpu_float).to(device),
                torch::empty({0, 8}, cpu_long).to(device)};
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
        return {xyz_tensor.contiguous(), key_tensor.contiguous()};
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

torch::Tensor VoxelMapper::colorizeRenderedMeshVertices(
    const torch::Tensor& vertices)
{
    auto points = vertices.detach().to(torch::kFloat32).contiguous();
    auto closest_color = torch::full(
        {points.size(0), 3}, 0.5f, points.options());
    auto closest_dist = torch::full(
        {points.size(0)}, std::numeric_limits<float>::infinity(), points.options());

    const auto keyframes = scene_->getAllKeyframes();
    for (const auto& [kf_id, pkf] : keyframes) {
        if (!pkf || !pkf->set_pose_) continue;
        const int height = std::max(1, pkf->image_height_);
        const int width = std::max(1, pkf->image_width_);
        const auto cam = pkf->toMiniCam(height, width);

        sv::RenderOpts render_opts;
        render_opts.output_depth = true;
        render_opts.output_T = true;
        auto render_pkg = voxel_model_->render(
            cam, height, width, torch::Tensor(), "sh0", false, std::nullopt,
            true, false, true, false, false, render_opts);
        auto color_it = render_pkg.find("color");
        auto depth_it = render_pkg.find("raw_depth");
        auto transmittance_it = render_pkg.find("raw_T");
        if (color_it == render_pkg.end() || depth_it == render_pkg.end() ||
            transmittance_it == render_pkg.end()) {
            continue;
        }

        auto raw_depth = depth_it->second.detach().to(points.device()).to(torch::kFloat32);
        if (raw_depth.dim() == 4 && raw_depth.size(0) == 1) raw_depth = raw_depth.squeeze(0);
        if (raw_depth.dim() != 3 || raw_depth.size(0) < 3) continue;
        const int64_t depth_channel = 2;
        auto frame_depth = raw_depth.index({depth_channel}).contiguous();
        auto frame_alpha =
            (1.0f - transmittance_it->second.detach()
                        .to(points.device()).to(torch::kFloat32)).contiguous();
        auto frame_color =
            color_it->second.detach().to(points.device()).to(torch::kFloat32).contiguous();

        auto uv = projectNormalized(points, cam);
        auto camera_position =
            cam.position.to(points.device()).to(torch::kFloat32).view({1, 3});
        auto camera_lookat =
            cam.lookat.to(points.device()).to(torch::kFloat32).view({3, 1});
        auto all_point_depth =
            torch::matmul(points - camera_position, camera_lookat).flatten();
        auto valid_idx = torch::nonzero(
                             (uv.abs() <= 1.0f).all(1) & (all_point_depth > 0.0f))
                             .view({-1}).to(torch::kLong);
        if (valid_idx.numel() == 0) continue;
        auto valid_uv = uv.index_select(0, valid_idx);
        auto sampled_alpha = sampleBilinear(frame_alpha, valid_uv);
        auto alpha_idx = torch::nonzero(
                             sampled_alpha >=
                                 rerun_params_.rendered_mesh_eval_alpha_thres_)
                             .view({-1}).to(torch::kLong);
        if (alpha_idx.numel() == 0) continue;
        valid_idx = valid_idx.index_select(0, alpha_idx);
        valid_uv = valid_uv.index_select(0, alpha_idx);

        auto sampled_depth = sampleBilinear(frame_depth, valid_uv);
        auto point_depth = all_point_depth.index_select(0, valid_idx);
        auto point_dist = (sampled_depth - point_depth).abs();
        auto better = point_dist < closest_dist.index_select(0, valid_idx);
        auto better_idx = torch::nonzero(better).view({-1}).to(torch::kLong);
        if (better_idx.numel() == 0) continue;
        valid_idx = valid_idx.index_select(0, better_idx);
        valid_uv = valid_uv.index_select(0, better_idx);
        point_dist = point_dist.index_select(0, better_idx);
        auto point_color = sampleBilinearChannels(frame_color, valid_uv);
        closest_dist.index_put_({valid_idx}, point_dist);
        closest_color.index_put_({valid_idx}, point_color);
    }
    return closest_color.contiguous();
}

void VoxelMapper::saveRenderedTsdfMeshPly(
    const std::filesystem::path& result_path)
{
    namespace fs = std::filesystem;
    torch::NoGradGuard no_grad;
    if (!voxel_model_ || !scene_) {
        std::cout << "[mesh/rendered-TSDF-fixed] skipped: mapper is not initialized.\n";
        return;
    }

    const auto& keyframes = scene_->keyframes();
    if (keyframes.empty()) {
        std::cout << "[mesh/rendered-TSDF-fixed] skipped: no keyframes available.\n";
        return;
    }
    if (!result_path.parent_path().empty()) {
        fs::create_directories(result_path.parent_path());
    }

    const float voxel_length =
        std::max(1.0e-6f, rerun_params_.rendered_mesh_eval_voxel_size_m_);
    const float min_keyframe_weight =
        std::max(0.0f, rerun_params_.rendered_mesh_eval_min_weight_);
    const float sdf_trunc = std::max(
        voxel_length,
        rerun_params_.rendered_mesh_eval_trunc_vox_ * voxel_length);
    const float depth_max =
        std::max(1.0e-6f, rerun_params_.rendered_mesh_eval_depth_max_m_);
    SparseRenderedTsdfGrid support_grid(voxel_length, sdf_trunc);

    std::unique_lock<std::mutex> render_lock(mutex_render_);
    bool froze_geo = false;
    auto unfreeze_geo = [&]() {
        if (froze_geo) {
            voxel_model_->unfreezeVoxGeo();
            froze_geo = false;
        }
    };

    try {
        c10::cuda::CUDACachingAllocator::emptyCache();
        voxel_model_->freezeVoxGeo();
        froze_geo = true;

        std::vector<RenderedMeshView> views;
        views.reserve(keyframes.size());
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
            auto render_pkg = voxel_model_->render(
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
            if (depth_it == render_pkg.end() || transmittance_it == render_pkg.end() ||
                !depth_it->second.defined() || !transmittance_it->second.defined()) {
                std::cout << "[mesh/rendered-TSDF-fixed] missing depth/transmittance for kf="
                          << kfid << ", skipping.\n";
                continue;
            }

            auto raw_depth = depth_it->second.detach().to(torch::kFloat32).contiguous();
            if (raw_depth.dim() == 4 && raw_depth.size(0) == 1) {
                raw_depth = raw_depth.squeeze(0);
            }
            if (raw_depth.dim() != 3 || raw_depth.size(0) < 3) {
                std::cout << "[mesh/rendered-TSDF-fixed] expected three rendered-depth channels for kf="
                          << kfid << ", got " << raw_depth.sizes() << ", skipping.\n";
                continue;
            }

            const int64_t depth_channel = 2;
            auto rendered_depth = raw_depth.index({depth_channel}).contiguous();
            auto rendered_transmittance =
                transmittance_it->second.detach().to(torch::kFloat32).contiguous();
            if (rendered_transmittance.dim() == 3 &&
                rendered_transmittance.size(0) == 1) {
                rendered_transmittance = rendered_transmittance.squeeze(0);
            }
            if (rendered_transmittance.dim() != 2 ||
                rendered_transmittance.size(0) != height ||
                rendered_transmittance.size(1) != width) {
                throw std::runtime_error(
                    "saveRenderedTsdfMeshPly: invalid transmittance shape");
            }
            auto rendered_alpha =
                (1.0f - rendered_transmittance).clamp(0.0f, 1.0f);
            const auto valid_surface =
                torch::isfinite(rendered_depth) &
                torch::isfinite(rendered_alpha) &
                (rendered_depth > 0.0f) &
                (rendered_alpha >=
                 rerun_params_.rendered_mesh_eval_alpha_thres_);
            rendered_depth = torch::where(
                valid_surface,
                rendered_depth,
                torch::zeros_like(rendered_depth));

            const auto mask_it = undistort_mask_.find(pkf->camera_id_);
            if (mask_it != undistort_mask_.end()) {
                auto mask = mask_it->second.to(rendered_depth.device());
                if (mask.dim() == 3) mask = mask.index({0});
                if (mask.dim() == 2 &&
                    mask.size(0) == width && mask.size(1) == height) {
                    mask = mask.transpose(0, 1);
                }
                if (mask.dim() != 2 || mask.size(0) != height || mask.size(1) != width) {
                    throw std::runtime_error(
                        "saveRenderedTsdfMeshPly: invalid undistortion mask shape");
                }
                rendered_depth = rendered_depth * mask.to(torch::kFloat32);
            }

            auto depth_cpu = rendered_depth.contiguous().to(torch::kCPU);
            cv::Mat depth_mat(height, width, CV_32FC1, depth_cpu.data_ptr<float>());
            support_grid.allocateFromKeyframe(
                depth_mat,
                pkf->intr_,
                pkf->getPosef(),
                depth_max);

            RenderedMeshView view;
            view.cam = cam;
            view.depth = rendered_depth;
            view.alpha = rendered_alpha;
            views.push_back(std::move(view));
        }

        auto [grid_xyz, vox_key] = support_grid.buildGrid(
            voxel_model_->geoGridPts().device());
        if (grid_xyz.numel() == 0 || vox_key.numel() == 0 || views.empty()) {
            unfreeze_geo();
            throw std::runtime_error(
                "saveRenderedTsdfMeshPly: rendered depths allocated no complete TSDF cells");
        }

        auto fused = fuseRenderedTsdfWithSupport(
            grid_xyz,
            views,
            sdf_trunc,
            /*crop_border=*/0.0f,
            rerun_params_.rendered_mesh_eval_alpha_thres_,
            /*use_unit_keyframe_weight=*/true);
        auto vox_sdf = fused.tsdf.index({vox_key});
        auto vox_support = fused.keyframe_support.index({vox_key});
        auto keep_mask =
            torch::isfinite(vox_sdf).all(1) &
            (std::get<0>(vox_support.min(1)) >= min_keyframe_weight) &
            (std::get<0>(vox_sdf.min(1)) < 0.0f) &
            (std::get<0>(vox_sdf.max(1)) > 0.0f);
        auto keep_idx = torch::nonzero(keep_mask).view({-1}).to(torch::kLong);
        vox_key = vox_key.index_select(0, keep_idx).contiguous();
        auto [vertices, faces] = marchingCubesGrid(
            -fused.tsdf,
            grid_xyz,
            vox_key,
            /*iso=*/0.0f);
        auto mesh = meshFromTensors(vertices, faces);
        if (mesh.vertices.empty() || mesh.faces.empty()) {
            unfreeze_geo();
            throw std::runtime_error(
                "saveRenderedTsdfMeshPly: zero-crossing TSDF produced an empty mesh");
        }
        setMeshColors(mesh, colorizeRenderedMeshVertices(vertices));
        if (!saveTriangleMeshPly(
                result_path,
                mesh,
                /*write_vertex_colors=*/true)) {
            unfreeze_geo();
            throw std::runtime_error(
                "saveRenderedTsdfMeshPly: failed to write mesh PLY");
        }
        unfreeze_geo();
        std::cout << "[mesh/rendered-TSDF-fixed] wrote " << result_path
                  << " depth_source="
                  << "median_opacity"
                  << " views=" << views.size()
                  << " voxel_size=" << voxel_length
                  << " truncation=" << sdf_trunc
                  << " min_keyframe_weight=" << min_keyframe_weight
                  << " alpha_threshold="
                  << rerun_params_.rendered_mesh_eval_alpha_thres_
                  << " candidate_grid_points=" << grid_xyz.size(0)
                  << " surface_grid_voxels=" << vox_key.size(0)
                  << " vertices=" << mesh.vertices.size()
                  << " faces=" << mesh.faces.size() << "\n";
    } catch (...) {
        unfreeze_geo();
        throw;
    }
}
