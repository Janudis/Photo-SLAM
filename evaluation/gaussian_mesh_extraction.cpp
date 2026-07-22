#include "include/gaussian_mapper.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <ATen/ops/unique_dim.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "include/gaussian_renderer.h"
#include "include_voxel/svrecon_marching_cubes_table.h"

namespace {

using torch::indexing::Slice;

struct GaussianTsdfView
{
    torch::Tensor w2c;
    torch::Tensor depth;
    torch::Tensor alpha;
    torch::Tensor color_cpu;
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    int width = 0;
    int height = 0;
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

torch::Tensor se3ToTensor(
    const Sophus::SE3f& pose,
    const torch::Device& device)
{
    const Eigen::Matrix4f matrix = pose.matrix();
    std::array<float, 16> row_major{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            row_major[static_cast<size_t>(r * 4 + c)] = matrix(r, c);
        }
    }
    return torch::from_blob(
               row_major.data(),
               {4, 4},
               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
        .clone()
        .to(device)
        .contiguous();
}

torch::Tensor normalizeImage2d(torch::Tensor image)
{
    if (image.dim() == 4 && image.size(0) == 1) image = image.squeeze(0);
    if (image.dim() == 3 && image.size(0) == 1) image = image.squeeze(0);
    TORCH_CHECK(image.dim() == 2, "expected a 2D image, got ", image.sizes());
    return image.contiguous();
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

std::pair<torch::Tensor, torch::Tensor> projectNormalized(
    const torch::Tensor& points,
    const GaussianTsdfView& view)
{
    auto cam_xyz =
        torch::matmul(points, view.w2c.index({Slice(0, 3), Slice(0, 3)}).transpose(0, 1)) +
        view.w2c.index({Slice(0, 3), 3}).view({1, 3});
    auto z = cam_xyz.index({Slice(), 2});
    auto u = view.fx * cam_xyz.index({Slice(), 0}) / z + view.cx;
    auto v = view.fy * cam_xyz.index({Slice(), 1}) / z + view.cy;
    auto uv = torch::stack({
        2.0f * (u + 0.5f) / static_cast<float>(view.width) - 1.0f,
        2.0f * (v + 0.5f) / static_cast<float>(view.height) - 1.0f}, 1);
    return {uv.contiguous(), z.contiguous()};
}

class SparseRenderedTsdfGrid
{
public:
    SparseRenderedTsdfGrid(float voxel_length, float sdf_trunc)
        : voxel_length_(voxel_length), sdf_trunc_(sdf_trunc)
    {}

    void allocateFromKeyframe(
        const cv::Mat& depth_map,
        const std::vector<float>& intr,
        const Sophus::SE3f& Tcw,
        float depth_max)
    {
        CV_Assert(depth_map.type() == CV_32FC1);
        if (intr.size() < 4) {
            throw std::runtime_error("Gaussian TSDF extraction requires pinhole intrinsics");
        }

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
                if (!std::isfinite(depth) || depth <= 0.0f || depth > depth_max) continue;

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
        for (const auto& item : point_index) {
            const auto& key = item.first;
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
            if (complete) voxel_keys.insert(voxel_keys.end(), corners.begin(), corners.end());
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
    float voxel_length_;
    float sdf_trunc_;
    std::unordered_set<SparseTsdfKey, SparseTsdfKeyHash> grid_keys_;
};

struct FusedRenderedTsdf
{
    torch::Tensor tsdf;
    torch::Tensor keyframe_support;
};

FusedRenderedTsdf fuseRenderedTsdf(
    const torch::Tensor& grid_xyz,
    const std::vector<GaussianTsdfView>& views,
    float trunc_dist,
    float alpha_threshold)
{
    constexpr int64_t kPointChunk = 500000;
    std::vector<torch::Tensor> tsdf_chunks;
    std::vector<torch::Tensor> support_chunks;

    for (int64_t begin = 0; begin < grid_xyz.size(0); begin += kPointChunk) {
        const int64_t count = std::min(kPointChunk, grid_xyz.size(0) - begin);
        auto xyz = grid_xyz.narrow(0, begin, count).contiguous();
        auto weight = torch::zeros({count}, xyz.options().dtype(torch::kFloat32));
        auto support = torch::zeros_like(weight);
        auto weighted_tsdf = torch::zeros_like(weight);

        for (const auto& view : views) {
            auto [uv, camera_depth] = projectNormalized(xyz, view);
            auto projected =
                torch::isfinite(camera_depth) & (camera_depth > 0.0f) &
                (uv.abs() <= 1.0f).all(1);
            auto valid_idx = torch::nonzero(projected).view({-1}).to(torch::kLong);
            if (valid_idx.numel() == 0) continue;

            auto valid_uv = uv.index_select(0, valid_idx);
            auto sampled_depth = sampleBilinear(view.depth, valid_uv);
            auto sampled_alpha = sampleBilinear(view.alpha, valid_uv);
            auto valid_depth = camera_depth.index_select(0, valid_idx);
            auto valid_sample =
                torch::isfinite(sampled_depth) & (sampled_depth > 0.0f) &
                torch::isfinite(sampled_alpha) & (sampled_alpha >= alpha_threshold);
            auto sample_idx = torch::nonzero(valid_sample).view({-1}).to(torch::kLong);
            if (sample_idx.numel() == 0) continue;

            valid_idx = valid_idx.index_select(0, sample_idx);
            sampled_depth = sampled_depth.index_select(0, sample_idx);
            valid_depth = valid_depth.index_select(0, sample_idx);
            auto sdf = (sampled_depth - valid_depth) / trunc_dist;
            auto trunc_idx = torch::nonzero(sdf >= -1.0f).view({-1}).to(torch::kLong);
            if (trunc_idx.numel() == 0) continue;

            valid_idx = valid_idx.index_select(0, trunc_idx);
            sdf = sdf.index_select(0, trunc_idx).clamp(-1.0f, 1.0f);
            auto unit_weight = torch::ones_like(sdf);
            weight.index_add_(0, valid_idx, unit_weight);
            support.index_add_(0, valid_idx, unit_weight);
            weighted_tsdf.index_add_(0, valid_idx, sdf);
        }

        auto nan = torch::full_like(weight, std::numeric_limits<float>::quiet_NaN());
        tsdf_chunks.push_back(torch::where(weight > 0.0f, weighted_tsdf / weight, nan));
        support_chunks.push_back(support);
    }
    return {torch::cat(tsdf_chunks, 0).contiguous(),
            torch::cat(support_chunks, 0).contiguous()};
}

std::pair<torch::Tensor, torch::Tensor> marchingCubesVoxels(
    const torch::Tensor& unit_val,
    const torch::Tensor& unit_xyz)
{
    auto mask =
        (unit_val > 0.0f).any(1) & (unit_val < 0.0f).any(1) &
        ~torch::isnan(unit_val).any(1);
    auto filter_idx = torch::nonzero(mask).view({-1}).to(torch::kLong);
    if (filter_idx.numel() == 0) {
        return {torch::empty({0, 3}, unit_xyz.options()),
                torch::empty({0, 3}, unit_val.options().dtype(torch::kLong))};
    }

    auto values = unit_val.index_select(0, filter_idx).contiguous();
    auto xyz = unit_xyz.index_select(0, filter_idx).contiguous();
    const auto device = values.device();
    const int64_t n_vox = values.size(0);

    static constexpr int64_t kEdgePairs[12][2] = {
        {0, 1}, {1, 5}, {5, 4}, {4, 0},
        {2, 3}, {3, 7}, {7, 6}, {6, 2},
        {0, 2}, {1, 3}, {5, 7}, {4, 6}};
    auto edges = torch::from_blob(
                     const_cast<int64_t*>(&kEdgePairs[0][0]),
                     {12, 2},
                     torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
                     .clone().to(device);
    auto edge_a = edges.index({Slice(), 0});
    auto edge_b = edges.index({Slice(), 1});
    auto value_a = values.index_select(1, edge_a);
    auto value_b = values.index_select(1, edge_b);
    auto xyz_a = xyz.index_select(1, edge_a);
    auto xyz_b = xyz.index_select(1, edge_b);
    auto denom = value_b - value_a;
    auto ratio = -value_a / denom;
    ratio = torch::where(
        denom.abs() < 1.0e-9f,
        torch::full_like(ratio, 0.5f),
        ratio);
    auto edge_vertices = xyz_a + ratio.unsqueeze(2) * (xyz_b - xyz_a);

    static constexpr int64_t kCubeIndexBases[8] = {1, 2, 4, 8, 16, 32, 64, 128};
    auto bases = torch::from_blob(
                     const_cast<int64_t*>(kCubeIndexBases),
                     {8},
                     torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
                     .clone().to(device);
    auto cube_idx = ((values < 0.0f).to(torch::kLong) * bases.view({1, 8}))
                        .sum(1).to(torch::kLong);
    auto tri_table = torch::from_blob(
                         const_cast<int64_t*>(&svrecon_mesh::kTriangleTable[0][0]),
                         {256, 15},
                         torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
                         .clone().to(device);
    auto tri_idx = tri_table.index_select(0, cube_idx).contiguous();
    auto gather_idx = tri_idx.clamp_min(0).unsqueeze(2).expand({n_vox, 15, 3});
    auto faces_xyz = edge_vertices.gather(1, gather_idx);
    faces_xyz = faces_xyz.index({tri_idx != -1}).view({-1, 3}).contiguous();
    if (faces_xyz.numel() == 0) {
        return {torch::empty({0, 3}, unit_xyz.options()),
                torch::empty({0, 3}, unit_val.options().dtype(torch::kLong))};
    }

    auto unique_result = at::unique_dim(
        faces_xyz, 0, true, true, false);
    return {std::get<0>(unique_result).contiguous(),
            std::get<1>(unique_result).view({-1, 3}).to(torch::kLong).contiguous()};
}

std::pair<torch::Tensor, torch::Tensor> marchingCubesGrid(
    const torch::Tensor& grid_values,
    const torch::Tensor& grid_xyz,
    const torch::Tensor& voxel_keys)
{
    constexpr int64_t kChunkSize = 1000000;
    std::vector<torch::Tensor> vertex_chunks;
    std::vector<torch::Tensor> face_chunks;
    int64_t vertex_offset = 0;

    for (int64_t begin = 0; begin < voxel_keys.size(0); begin += kChunkSize) {
        const int64_t count = std::min(kChunkSize, voxel_keys.size(0) - begin);
        auto keys = voxel_keys.narrow(0, begin, count).to(torch::kLong).contiguous();
        auto values = grid_values.index({keys}).reshape({count, 8}).contiguous();
        auto xyz = grid_xyz.index({keys}).reshape({count, 8, 3}).contiguous();
        auto [vertices, faces] = marchingCubesVoxels(values, xyz);
        if (faces.numel() == 0) continue;
        vertex_chunks.push_back(vertices);
        face_chunks.push_back(faces + vertex_offset);
        vertex_offset += vertices.size(0);
    }

    if (vertex_chunks.empty()) {
        return {torch::empty({0, 3}, grid_xyz.options()),
                torch::empty({0, 3}, voxel_keys.options().dtype(torch::kLong))};
    }
    auto vertices = torch::cat(vertex_chunks, 0).contiguous();
    auto faces = torch::cat(face_chunks, 0).to(torch::kLong).contiguous();
    auto faces_xyz = vertices.index({faces}).reshape({-1, 3}).contiguous();
    auto unique_result = at::unique_dim(faces_xyz, 0, true, true, false);
    return {std::get<0>(unique_result).contiguous(),
            std::get<1>(unique_result).view({-1, 3}).to(torch::kLong).contiguous()};
}

torch::Tensor colorizeVertices(
    const torch::Tensor& vertices,
    const std::vector<GaussianTsdfView>& views,
    float trunc_dist,
    float alpha_threshold)
{
    constexpr int64_t kPointChunk = 500000;
    std::vector<torch::Tensor> color_chunks;
    for (int64_t begin = 0; begin < vertices.size(0); begin += kPointChunk) {
        const int64_t count = std::min(kPointChunk, vertices.size(0) - begin);
        auto xyz = vertices.narrow(0, begin, count).contiguous();
        auto best_dist = torch::full(
            {count}, std::numeric_limits<float>::infinity(), xyz.options());
        auto best_color = torch::full({count, 3}, 0.5f, xyz.options());

        for (const auto& view : views) {
            auto [uv, camera_depth] = projectNormalized(xyz, view);
            auto projected =
                torch::isfinite(camera_depth) & (camera_depth > 0.0f) &
                (uv.abs() <= 1.0f).all(1);
            auto valid_idx = torch::nonzero(projected).view({-1}).to(torch::kLong);
            if (valid_idx.numel() == 0) continue;
            auto valid_uv = uv.index_select(0, valid_idx);
            auto sampled_depth = sampleBilinear(view.depth, valid_uv);
            auto sampled_alpha = sampleBilinear(view.alpha, valid_uv);
            auto vertex_depth = camera_depth.index_select(0, valid_idx);
            auto distance = (sampled_depth - vertex_depth).abs();
            auto valid =
                torch::isfinite(sampled_depth) & (sampled_depth > 0.0f) &
                (sampled_alpha >= alpha_threshold) & (distance <= trunc_dist);
            auto local_idx = torch::nonzero(valid).view({-1}).to(torch::kLong);
            if (local_idx.numel() == 0) continue;

            valid_idx = valid_idx.index_select(0, local_idx);
            valid_uv = valid_uv.index_select(0, local_idx);
            distance = distance.index_select(0, local_idx);
            auto better = distance < best_dist.index_select(0, valid_idx);
            auto better_idx = torch::nonzero(better).view({-1}).to(torch::kLong);
            if (better_idx.numel() == 0) continue;

            valid_idx = valid_idx.index_select(0, better_idx);
            valid_uv = valid_uv.index_select(0, better_idx);
            distance = distance.index_select(0, better_idx);
            auto frame_color = view.color_cpu.to(xyz.device(), torch::kFloat32);
            auto sampled_color = sampleBilinearChannels(frame_color, valid_uv);
            best_dist.index_put_({valid_idx}, distance);
            best_color.index_put_({valid_idx}, sampled_color);
        }
        color_chunks.push_back(best_color.contiguous());
    }
    return torch::cat(color_chunks, 0).contiguous();
}

bool saveBinaryPly(
    const std::filesystem::path& path,
    const torch::Tensor& vertices,
    const torch::Tensor& faces,
    const torch::Tensor& colors)
{
    auto vertices_cpu = vertices.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto faces_cpu = faces.detach().to(torch::kCPU).to(torch::kInt32).contiguous();
    auto colors_cpu =
        (colors.detach().to(torch::kCPU).to(torch::kFloat32).clamp(0.0f, 1.0f) * 255.0f)
            .round().to(torch::kUInt8).contiguous();
    if (vertices_cpu.numel() == 0 || faces_cpu.numel() == 0) return false;

    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out << "ply\n"
        << "format binary_little_endian 1.0\n"
        << "element vertex " << vertices_cpu.size(0) << "\n"
        << "property float x\nproperty float y\nproperty float z\n"
        << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        << "element face " << faces_cpu.size(0) << "\n"
        << "property list uchar int vertex_indices\n"
        << "end_header\n";

    auto v = vertices_cpu.accessor<float, 2>();
    auto c = colors_cpu.accessor<uint8_t, 2>();
    for (int64_t i = 0; i < vertices_cpu.size(0); ++i) {
        out.write(reinterpret_cast<const char*>(&v[i][0]), sizeof(float));
        out.write(reinterpret_cast<const char*>(&v[i][1]), sizeof(float));
        out.write(reinterpret_cast<const char*>(&v[i][2]), sizeof(float));
        out.write(reinterpret_cast<const char*>(&c[i][0]), sizeof(uint8_t));
        out.write(reinterpret_cast<const char*>(&c[i][1]), sizeof(uint8_t));
        out.write(reinterpret_cast<const char*>(&c[i][2]), sizeof(uint8_t));
    }

    auto f = faces_cpu.accessor<int32_t, 2>();
    const uint8_t count = 3;
    for (int64_t i = 0; i < faces_cpu.size(0); ++i) {
        out.write(reinterpret_cast<const char*>(&count), sizeof(uint8_t));
        out.write(reinterpret_cast<const char*>(&f[i][0]), sizeof(int32_t));
        out.write(reinterpret_cast<const char*>(&f[i][1]), sizeof(int32_t));
        out.write(reinterpret_cast<const char*>(&f[i][2]), sizeof(int32_t));
    }
    return static_cast<bool>(out);
}

} // namespace

void GaussianMapper::saveRenderedTsdfMeshPly(
    const std::filesystem::path& result_path)
{
    torch::NoGradGuard no_grad;
    if (!gaussians_ || !scene_) {
        throw std::runtime_error("Gaussian mapper is not initialized");
    }
    if (scene_->keyframes().empty()) {
        throw std::runtime_error("no Gaussian keyframes are available");
    }
    if (!result_path.parent_path().empty()) {
        std::filesystem::create_directories(result_path.parent_path());
    }

    const float voxel_length = rendered_mesh_eval_voxel_size_m_;
    const float trunc_dist = rendered_mesh_eval_trunc_vox_ * voxel_length;
    SparseRenderedTsdfGrid support_grid(voxel_length, trunc_dist);
    std::vector<GaussianTsdfView> views;
    views.reserve(scene_->keyframes().size());

    std::unique_lock<std::mutex> render_lock(mutex_render_);
    c10::cuda::CUDACachingAllocator::emptyCache();
    auto black_background = torch::zeros_like(background_);
    auto white_override = torch::ones(
        {gaussians_->getXYZ().size(0), 3},
        gaussians_->getXYZ().options().dtype(torch::kFloat32));

    for (const auto& [keyframe_id, keyframe] : scene_->keyframes()) {
        if (!keyframe || !keyframe->set_pose_ || !keyframe->set_camera_ ||
            keyframe->intr_.size() < 4 || keyframe->image_height_ <= 0 ||
            keyframe->image_width_ <= 0) {
            continue;
        }

        auto render_pkg = GaussianRenderer::render(
            keyframe,
            keyframe->image_height_,
            keyframe->image_width_,
            gaussians_,
            pipe_params_,
            background_,
            override_color_);
        auto alpha_pkg = GaussianRenderer::render(
            keyframe,
            keyframe->image_height_,
            keyframe->image_width_,
            gaussians_,
            pipe_params_,
            black_background,
            white_override,
            1.0f,
            true);

        auto depth = normalizeImage2d(
            std::get<1>(render_pkg).detach().to(torch::kFloat32));
        auto alpha_image = std::get<0>(alpha_pkg).detach().to(torch::kFloat32);
        TORCH_CHECK(
            alpha_image.dim() == 3 && alpha_image.size(0) == 3,
            "Gaussian alpha render has invalid shape ", alpha_image.sizes());
        auto alpha = alpha_image.index({0}).contiguous().clamp(0.0f, 1.0f);
        auto color = std::get<0>(render_pkg).detach().to(torch::kFloat32).contiguous();

        auto valid =
            torch::isfinite(depth) & (depth > RGBD_min_depth_) &
            (depth <= rendered_mesh_eval_depth_max_m_) &
            torch::isfinite(alpha) & (alpha >= rendered_mesh_eval_alpha_thres_);
        const auto mask_it = undistort_mask_.find(keyframe->camera_id_);
        if (mask_it != undistort_mask_.end()) {
            auto mask = mask_it->second.to(depth.device()).to(torch::kFloat32);
            if (mask.dim() == 3) {
                TORCH_CHECK(
                    mask.size(0) == 1 || mask.size(0) == 3,
                    "unsupported undistortion mask shape ", mask.sizes());
                mask = mask.index({0});
            }
            if (mask.dim() == 2 && mask.size(0) == keyframe->image_width_ &&
                mask.size(1) == keyframe->image_height_) {
                mask = mask.transpose(0, 1);
            }
            TORCH_CHECK(
                mask.dim() == 2 &&
                    mask.size(0) == depth.size(0) &&
                    mask.size(1) == depth.size(1),
                "undistortion mask shape ", mask.sizes(),
                " does not match rendered depth ", depth.sizes());
            valid &= mask > 0.5f;
        }
        depth = torch::where(valid, depth, torch::zeros_like(depth)).contiguous();
        alpha = torch::where(valid, alpha, torch::zeros_like(alpha)).contiguous();

        auto depth_cpu = depth.to(torch::kCPU).contiguous();
        cv::Mat depth_mat(
            keyframe->image_height_,
            keyframe->image_width_,
            CV_32FC1,
            depth_cpu.data_ptr<float>());
        support_grid.allocateFromKeyframe(
            depth_mat,
            keyframe->intr_,
            keyframe->getPosef(),
            rendered_mesh_eval_depth_max_m_);

        GaussianTsdfView view;
        view.w2c = se3ToTensor(keyframe->getPosef(), depth.device());
        view.depth = depth;
        view.alpha = alpha;
        view.color_cpu = color.to(torch::kCPU).contiguous();
        view.fx = keyframe->intr_[0];
        view.fy = keyframe->intr_[1];
        view.cx = keyframe->intr_[2];
        view.cy = keyframe->intr_[3];
        view.width = keyframe->image_width_;
        view.height = keyframe->image_height_;
        views.push_back(std::move(view));
    }

    if (views.empty()) {
        throw std::runtime_error("no valid rendered Gaussian views were produced");
    }
    auto [grid_xyz, voxel_keys] = support_grid.buildGrid(views.front().depth.device());
    if (grid_xyz.numel() == 0 || voxel_keys.numel() == 0) {
        throw std::runtime_error("rendered Gaussian depths allocated no complete TSDF cells");
    }

    auto fused = fuseRenderedTsdf(
        grid_xyz,
        views,
        trunc_dist,
        rendered_mesh_eval_alpha_thres_);
    auto voxel_sdf = fused.tsdf.index({voxel_keys});
    auto voxel_support = fused.keyframe_support.index({voxel_keys});
    auto keep =
        torch::isfinite(voxel_sdf).all(1) &
        (std::get<0>(voxel_support.min(1)) >= rendered_mesh_eval_min_weight_) &
        (std::get<0>(voxel_sdf.min(1)) < 0.0f) &
        (std::get<0>(voxel_sdf.max(1)) > 0.0f);
    auto keep_idx = torch::nonzero(keep).view({-1}).to(torch::kLong);
    voxel_keys = voxel_keys.index_select(0, keep_idx).contiguous();
    if (voxel_keys.numel() == 0) {
        throw std::runtime_error("Gaussian TSDF has no supported zero-crossing cells");
    }

    auto [vertices, faces] = marchingCubesGrid(-fused.tsdf, grid_xyz, voxel_keys);
    if (vertices.numel() == 0 || faces.numel() == 0) {
        throw std::runtime_error("Gaussian TSDF Marching Cubes produced an empty mesh");
    }
    auto colors = colorizeVertices(
        vertices,
        views,
        trunc_dist,
        rendered_mesh_eval_alpha_thres_);
    if (!saveBinaryPly(result_path, vertices, faces, colors)) {
        throw std::runtime_error("failed to write Gaussian surface mesh");
    }

    std::cout << "[Gaussian mesh/rendered-TSDF] wrote " << result_path
              << " views=" << views.size()
              << " voxel_size=" << voxel_length
              << " truncation=" << trunc_dist
              << " min_keyframe_weight=" << rendered_mesh_eval_min_weight_
              << " alpha_threshold=" << rendered_mesh_eval_alpha_thres_
              << " grid_points=" << grid_xyz.size(0)
              << " surface_cells=" << voxel_keys.size(0)
              << " vertices=" << vertices.size(0)
              << " faces=" << faces.size(0) << "\n";
}
