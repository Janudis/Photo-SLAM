#pragma once

#include "include_voxel/voxel_model.h"
#include <ATen/ops/isin.h>
#include <ATen/ops/searchsorted.h>
#include <ATen/ops/unique_dim.h>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <opencv2/flann.hpp>

namespace SVRECON_TV_COMPUTE {
void total_variation_bw(
    const torch::Tensor& grid_pts,
    const torch::Tensor& vox_key,
    float weight,
    const torch::Tensor& vox_size_inv,
    bool no_tv_s,
    bool tv_sparse,
    const torch::Tensor& grid_pts_grad);
} // namespace SVRECON_TV_COMPUTE

namespace SVRECON_GE_COMPUTE {
void grid_eikonal_bw(
    const torch::Tensor& grid_pts,
    const torch::Tensor& vox_key,
    const torch::Tensor& grid_voxel_coord,
    const torch::Tensor& grid_voxel_size,
    int32_t grid_res,
    const torch::Tensor& grid_mask,
    const torch::Tensor& grid_keys,
    const torch::Tensor& grid2voxel,
    const torch::Tensor& active_list,
    float weight,
    float vox_size_inv,
    bool no_tv_s,
    bool tv_sparse,
    const torch::Tensor& grid_pts_grad);
} // namespace SVRECON_GE_COMPUTE

namespace SVRECON_LS_COMPUTE {
void laplacian_smoothness_bw(
    const torch::Tensor& grid_pts,
    const torch::Tensor& vox_key,
    const torch::Tensor& grid_voxel_coord,
    const torch::Tensor& grid_voxel_size,
    int32_t grid_res,
    const torch::Tensor& grid_mask,
    const torch::Tensor& grid_keys,
    const torch::Tensor& grid2voxel,
    const torch::Tensor& active_list,
    float weight,
    float vox_size_inv,
    bool no_tv_s,
    bool tv_sparse,
    const torch::Tensor& grid_pts_grad);
} // namespace SVRECON_LS_COMPUTE

namespace SVRECON_UTILS {
torch::Tensor ijk_2_octpath(const torch::Tensor& ijk, const torch::Tensor& octlevel);
torch::Tensor octpath_2_ijk(const torch::Tensor& octpath, const torch::Tensor& octlevel);
std::tuple<at::Tensor, at::Tensor, at::Tensor> valid_gradient_table(
    const at::Tensor& vox_center,
    const at::Tensor& vox_size,
    const at::Tensor& scene_center,
    float inside_extent,
    int grid_res_pow2,
    const at::Tensor& is_leaf);
} // namespace SVRECON_UTILS

namespace SVRECON_ADAM_STEP {
void unbiased_adam_step(
    bool sparse,
    torch::Tensor& param,
    const torch::Tensor& grad,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    const double step,
    const double lr,
    const double beta1,
    const double beta2,
    const float eps);
} // namespace SVRECON_ADAM_STEP

namespace sv {
namespace {
torch::Tensor levelToVoxSize(
    const torch::Tensor& scene_extent,
    const torch::Tensor& octlevel)
{
    return scene_extent * torch::pow(
        torch::full_like(octlevel.to(torch::kFloat32), 2.0f),
        -octlevel.to(torch::kFloat32));
}

torch::Tensor voxSizeToLevel(
    const torch::Tensor& scene_extent,
    const torch::Tensor& vox_size)
{
    return -torch::log2(vox_size / scene_extent);
}

std::pair<torch::Tensor, torch::Tensor> decodeOctpath(
    const torch::Tensor& octpath,
    const torch::Tensor& octlevel,
    const torch::Tensor& scene_center,
    const torch::Tensor& scene_extent)
{
    torch::Tensor op = octpath.reshape({-1, 1}).contiguous();
    torch::Tensor lv = octlevel.reshape({-1, 1}).contiguous();
    torch::Tensor scene_min_xyz = scene_center - 0.5f * scene_extent;
    torch::Tensor vox_size = levelToVoxSize(scene_extent, lv);
    torch::Tensor vox_ijk = SVRECON_UTILS::octpath_2_ijk(op, lv);
    torch::Tensor vox_center = scene_min_xyz + (vox_ijk.to(torch::kFloat32) + 0.5f) * vox_size;
    return {vox_center.contiguous(), vox_size.contiguous()};
}

std::tuple<torch::Tensor, torch::Tensor> uniqueRowsWithInverse(const torch::Tensor& rows);

std::pair<torch::Tensor, torch::Tensor> genSvreconDenseOctpath(
    int outside_level,
    int n_level_inside,
    int max_num_levels,
    torch::Device dev)
{
    TORCH_CHECK(n_level_inside > 0,
                "genSvreconDenseOctpath: n_level_inside must be positive");
    TORCH_CHECK(outside_level >= 0 &&
                    outside_level + n_level_inside <= max_num_levels,
                "genSvreconDenseOctpath: invalid outside/inside levels");

    const int64_t dense_count =
        (n_level_inside > 1)
            ? (1LL << (3 * (n_level_inside - 1)))
            : 1LL;
    const int64_t total_count = 8LL * dense_count;

    std::vector<int64_t> octpaths;
    octpaths.reserve(static_cast<size_t>(total_count));

    auto append_path = [&](int64_t ordinal) {
        const int64_t eight_idx = ordinal / dense_count;
        const int64_t dense_idx = ordinal - eight_idx * dense_count;

        int64_t path = eight_idx << (3 * (max_num_levels - 1));
        for (int k = 0; k < outside_level; ++k) {
            path |= (eight_idx ^ 0b111LL) << (3 * (max_num_levels - (k + 2)));
        }
        if (n_level_inside > 1) {
            const int shift =
                3 * (max_num_levels - (outside_level + 1) - (n_level_inside - 1));
            path |= dense_idx << shift;
        }
        octpaths.push_back(path);
    };

    for (int64_t ordinal = 0; ordinal < total_count; ++ordinal) {
        append_path(ordinal);
    }

    auto cpu_opts = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
    torch::Tensor octpath =
        torch::from_blob(
            octpaths.data(),
            {static_cast<int64_t>(octpaths.size()), 1},
            cpu_opts)
            .clone()
            .to(dev)
            .contiguous();
    torch::Tensor octlevel = torch::full(
        {octpath.size(0), 1},
        static_cast<int8_t>(outside_level + n_level_inside),
        torch::TensorOptions().dtype(torch::kInt8).device(dev));
    return {octpath, octlevel.contiguous()};
}

std::pair<torch::Tensor, torch::Tensor> buildGridPtsLink(
    const torch::Tensor& octpath,
    const torch::Tensor& octlevel,
    int max_num_levels)
{
    torch::Tensor op = octpath.reshape({-1, 1}).contiguous();
    torch::Tensor lv = octlevel.reshape({-1, 1}).contiguous();
    torch::Tensor vox_ijk = SVRECON_UTILS::octpath_2_ijk(op, lv).to(torch::kLong).contiguous();
    torch::Tensor lv2max = (max_num_levels - lv.to(torch::kLong)).contiguous();
    torch::Tensor scale = torch::pow(
        torch::full_like(lv2max.to(torch::kFloat32), 2.0f),
        lv2max.to(torch::kFloat32)).to(torch::kLong).contiguous();

    torch::Tensor base_grid_ijk = (vox_ijk * scale).contiguous();

    // Equivalent to SVRecon's gen_gridpoints_coordinate(...).unique(dim=0),
    // without materializing an [N,8,3] int64 tensor. Large, weakly connected
    // scenes otherwise require hundreds of MiB for this single temporary.
    const int64_t coord_stride = (int64_t{1} << max_num_levels) + 1;
    const int64_t coord_stride_sq = coord_stride * coord_stride;
    torch::Tensor base_key =
        base_grid_ijk.index({torch::indexing::Slice(), 0}) * coord_stride_sq +
        base_grid_ijk.index({torch::indexing::Slice(), 1}) * coord_stride +
        base_grid_ijk.index({torch::indexing::Slice(), 2});
    const std::vector<int64_t> corner_coeff_values = {
        int64_t{0},
        int64_t{1},
        coord_stride,
        coord_stride + 1,
        coord_stride_sq,
        coord_stride_sq + 1,
        coord_stride_sq + coord_stride,
        coord_stride_sq + coord_stride + 1};
    torch::Tensor corner_coeff = torch::tensor(
        corner_coeff_values,
        torch::TensorOptions().dtype(torch::kLong).device(op.device()));
    torch::Tensor packed_grid_pts =
        base_key.view({-1, 1}) + scale.view({-1, 1}) * corner_coeff.view({1, 8});

    auto [packed_grid_pts_key, vox_key_flat] =
        uniqueRowsWithInverse(packed_grid_pts.reshape({-1, 1}));
    torch::Tensor packed = packed_grid_pts_key.reshape({-1});
    torch::Tensor grid_x = at::floor_divide(packed, coord_stride_sq);
    torch::Tensor remainder = at::remainder(packed, coord_stride_sq);
    torch::Tensor grid_y = at::floor_divide(remainder, coord_stride);
    torch::Tensor grid_z = at::remainder(remainder, coord_stride);
    torch::Tensor grid_pts_key =
        torch::stack({grid_x, grid_y, grid_z}, /*dim=*/1).to(torch::kLong).contiguous();
    torch::Tensor vox_key = vox_key_flat.reshape({-1, 8}).contiguous();
    return {grid_pts_key.contiguous(), vox_key};
}

std::tuple<torch::Tensor, torch::Tensor> uniqueRowsWithInverse(const torch::Tensor& rows)
{
    auto result = at::unique_dim(
        rows.contiguous(),
        /*dim=*/0,
        /*sorted=*/true,
        /*return_inverse=*/true,
        /*return_counts=*/false);
    return {std::get<0>(result).contiguous(), std::get<1>(result).contiguous()};
}

std::optional<std::pair<torch::Tensor, float>> mainSceneBoundPcdHeuristicCpp(
    const torch::Tensor& points_cpu,
    float pcd_density_rate)
{
    if (!points_cpu.defined() || points_cpu.numel() == 0 || points_cpu.size(0) < 1) {
        return std::nullopt;
    }

    auto pts = points_cpu.to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (pts.dim() != 2 || pts.size(1) != 3) {
        return std::nullopt;
    }

    auto center = std::get<0>(pts.median(/*dim=*/0, /*keepdim=*/false)).contiguous();
    auto dist = std::get<0>(
        (pts - center.view({1, 3})).abs().max(/*dim=*/1, /*keepdim=*/false))
        .to(torch::kFloat32)
        .contiguous();
    dist = std::get<0>(dist.sort(/*dim=*/0)).contiguous();

    const int64_t n = dist.size(0);
    if (n <= 0) {
        return std::nullopt;
    }

    auto idx = torch::arange(
        1, n + 1,
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto density = idx * (dist > 0.0f).to(torch::kFloat32) /
        ((2.0f * dist).pow(3) + 1e-6f);

    int64_t begin_idx = static_cast<int64_t>(
        std::llround(static_cast<double>(n) * 0.05));
    begin_idx = std::max<int64_t>(0, std::min<int64_t>(n - 1, begin_idx));

    auto tail = density.index({torch::indexing::Slice(begin_idx, torch::indexing::None)}).contiguous();
    if (tail.numel() <= 0) {
        return std::nullopt;
    }

    const int64_t max_idx = begin_idx + tail.argmax().item<int64_t>();
    const float max_density = density.index({max_idx}).item<float>();
    const float target_density = pcd_density_rate * max_density;
    auto right = density.index({torch::indexing::Slice(max_idx, torch::indexing::None)}).contiguous();
    auto below = torch::nonzero(right < target_density).view({-1}).contiguous();
    if (below.numel() <= 0) {
        return std::nullopt;
    }

    const int64_t target_idx = max_idx + below.index({0}).item<int64_t>();
    const float radius = dist.index({target_idx}).item<float>();
    if (!std::isfinite(radius) || radius <= 0.0f) {
        return std::nullopt;
    }

    return std::make_pair(center.contiguous(), radius);
}

torch::Tensor uniqueSorted1dLong(const torch::Tensor& values)
{
    if (!values.defined() || values.numel() == 0) {
        return values;
    }
    auto sorted = std::get<0>(values.to(torch::kLong).contiguous().sort(/*dim=*/0));
    if (sorted.numel() <= 1) {
        return sorted.contiguous();
    }
    using torch::indexing::Slice;
    auto keep = torch::empty_like(sorted, torch::kBool);
    keep.index_put_({0}, true);
    keep.index_put_(
        {Slice(1, torch::indexing::None)},
        sorted.index({Slice(1, torch::indexing::None)}) !=
            sorted.index({Slice(torch::indexing::None, -1)}));
    return sorted.index_select(0, torch::nonzero(keep).view({-1})).contiguous();
}

torch::Tensor aggregateVoxelCornersIntoGridPts(
    int64_t num_grid_pts,
    const torch::Tensor& vox_key,
    const torch::Tensor& vox_val)
{
    TORCH_CHECK(vox_key.dim() == 2 && vox_key.size(1) == 8,
                "aggregateVoxelCornersIntoGridPts expects vox_key [N,8]");
    TORCH_CHECK(vox_val.dim() >= 3 && vox_val.size(0) == vox_key.size(0) &&
                    vox_val.size(1) == 8,
                "aggregateVoxelCornersIntoGridPts expects vox_val [N,8,*]");
    std::vector<int64_t> out_shape{num_grid_pts};
    for (int64_t d = 2; d < vox_val.dim(); ++d) {
        out_shape.push_back(vox_val.size(d));
    }
    auto out = torch::zeros(
        out_shape,
        torch::TensorOptions().dtype(torch::kFloat32).device(vox_val.device()));
    torch::Tensor idx = vox_key.to(vox_val.device()).flatten().to(torch::kLong);
    torch::Tensor src = vox_val.flatten(0, 1).to(torch::kFloat32);
    out.index_add_(0, idx, src);
    torch::Tensor counts = torch::zeros(
        {num_grid_pts},
        torch::TensorOptions().dtype(torch::kFloat32).device(vox_val.device()));
    counts.index_add_(
        0,
        idx,
        torch::ones({idx.size(0)},
                    torch::TensorOptions().dtype(torch::kFloat32).device(vox_val.device())));
    std::vector<int64_t> count_shape{num_grid_pts};
    for (int64_t d = 1; d < out.dim(); ++d) {
        count_shape.push_back(1);
    }
    out = out / counts.clamp_min(1.0f).view(count_shape);
    return out.contiguous();
}

std::pair<torch::Tensor, torch::Tensor> genChildrenOctpath(
    const torch::Tensor& octpath,
    const torch::Tensor& octlevel,
    int max_num_levels)
{
    auto op = octpath.reshape({-1, 1}).to(torch::kInt64).contiguous();
    auto lv = octlevel.reshape({-1, 1}).to(torch::kInt64).contiguous() + 1;
    TORCH_CHECK(lv.max().item<int64_t>() <= max_num_levels,
                "Maximum level out of bound after subdivision");
    auto shift = (3 * (max_num_levels - lv)).to(torch::kFloat32);
    auto shift_scale = torch::pow(
        torch::full_like(shift, 2.0f),
        shift).to(torch::kInt64).contiguous();
    auto children = torch::arange(
        8,
        torch::TensorOptions().dtype(torch::kInt64).device(op.device())).view({1, 8});
    auto child_bits = children * shift_scale;
    auto child_path = (op.view({-1, 1}) + child_bits).reshape({-1, 1}).contiguous();
    auto child_level = lv.to(torch::kInt8).repeat_interleave(8, 0).contiguous();
    return {child_path, child_level};
}

torch::Tensor subdivideVoxelCornerValues(const torch::Tensor& vox_val)
{
    TORCH_CHECK(vox_val.dim() >= 3 && vox_val.size(1) == 8,
                "subdivideVoxelCornerValues expects [N,8,*]");
    std::vector<torch::Tensor> child_blocks;
    child_blocks.reserve(8);
    for (int c = 0; c < 8; ++c) {
        auto v0 = vox_val.index({torch::indexing::Slice(), c});
        std::vector<torch::Tensor> corners;
        corners.reserve(8);
        for (int q = 0; q < 8; ++q) {
            const int m = c ^ q;
            if (m == 0) {
                corners.push_back(v0);
            } else if (m == 1 || m == 2 || m == 4) {
                corners.push_back(0.5f * (v0 + vox_val.index({torch::indexing::Slice(), c ^ m})));
            } else if (m == 3) {
                corners.push_back(0.25f * (
                    v0 +
                    vox_val.index({torch::indexing::Slice(), c ^ 0b001}) +
                    vox_val.index({torch::indexing::Slice(), c ^ 0b010}) +
                    vox_val.index({torch::indexing::Slice(), c ^ 0b011})));
            } else if (m == 5) {
                corners.push_back(0.25f * (
                    v0 +
                    vox_val.index({torch::indexing::Slice(), c ^ 0b001}) +
                    vox_val.index({torch::indexing::Slice(), c ^ 0b100}) +
                    vox_val.index({torch::indexing::Slice(), c ^ 0b101})));
            } else if (m == 6) {
                corners.push_back(0.25f * (
                    v0 +
                    vox_val.index({torch::indexing::Slice(), c ^ 0b010}) +
                    vox_val.index({torch::indexing::Slice(), c ^ 0b100}) +
                    vox_val.index({torch::indexing::Slice(), c ^ 0b110})));
            } else {
                corners.push_back(vox_val.mean(/*dim=*/1, /*keepdim=*/false));
            }
        }
        child_blocks.push_back(torch::stack(corners, 1).contiguous());
    }
    auto stacked = torch::stack(child_blocks, 1).contiguous();
    std::vector<int64_t> out_shape{vox_val.size(0) * 8, 8};
    for (int64_t d = 2; d < vox_val.dim(); ++d) {
        out_shape.push_back(vox_val.size(d));
    }
    return stacked.reshape(out_shape).contiguous();
}

} // namespace

} // namespace sv
