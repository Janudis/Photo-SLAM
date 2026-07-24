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

void VoxelModel::setTopologySdfInitializationMode(const std::string& mode)
{
    if (mode == "orb_prior") {
        topology_sdf_init_mode_ = SdfInitMode::OrbPriorOnly;
    } else if (mode == "weak_positive") {
        topology_sdf_init_mode_ = SdfInitMode::WeakPositive;
    } else if (mode == "source_points") {
        topology_sdf_init_mode_ = SdfInitMode::SignedPointPrior;
    } else if (mode == "weak_surface_prior") {
        topology_sdf_init_mode_ = SdfInitMode::WeakSurfacePrior;
    } else {
        throw std::invalid_argument(
            "VoxelModel topology SDF initialization mode must be one of: "
            "orb_prior, weak_positive, source_points, weak_surface_prior");
    }
}

namespace {
constexpr float kSHC0 = 0.28209479177387814f;

torch::Tensor rgbToShZero(const torch::Tensor& rgb)
{
    return (rgb - 0.5f) / kSHC0;
}

torch::Tensor shZeroToRgb(const torch::Tensor& sh)
{
    return sh * kSHC0 + 0.5f;
}

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

VoxelModel::~VoxelModel() = default;

VoxelModel::VoxelModel(const int sh_degree)
    : active_sh_degree_(0)
{
    this->max_sh_degree_ = sh_degree;

    // Device
    if (torch::cuda::is_available())
        device_type_ = torch::kCUDA;
    else
        device_type_ = torch::kCPU;

    // Initialize all tensors on chosen device
    VOXEL_MODEL_INIT_TENSORS(this->device_type_);
}

VoxelModel::VoxelModel(const VoxelModelParams& model_params)
    : active_sh_degree_(0)
{
    this->max_sh_degree_ = model_params.sh_degree_;
    this->white_background_ = model_params.white_background_;
    this->black_background_ = false;

    // Device
    if (model_params.data_device_ == "cuda")
        device_type_ = torch::kCUDA;
    else
        device_type_ = torch::kCPU;

    // Initialize all tensors on chosen device
    VOXEL_MODEL_INIT_TENSORS(this->device_type_);
}


const torch::Tensor& sv::VoxelModel::geoGridPts() const { return _geo_grid_pts_; }
const torch::Tensor& sv::VoxelModel::sh0()        const { return sh0_; }
const torch::Tensor& sv::VoxelModel::shs()        const { return shs_; }

int64_t sv::VoxelModel::numGridPts() const {
    return (grid_pts_key_.defined() && grid_pts_key_.dim() > 0)
           ? grid_pts_key_.size(0)
           : 0;
}
torch::Tensor VoxelModel::gridPointsWorld() const
{
    TORCH_CHECK(grid_pts_key_.defined() &&
                    grid_pts_key_.dim() == 2 &&
                    grid_pts_key_.size(1) == 3,
                "gridPointsWorld: grid_pts_key_ must be [M,3]");
    TORCH_CHECK(scene_center_.defined() && scene_center_.numel() == 3,
                "gridPointsWorld: scene_center_ must be [3]");
    TORCH_CHECK(scene_extent_.defined() && scene_extent_.numel() == 1,
                "gridPointsWorld: scene_extent_ must be [1]");

    auto dev = grid_pts_key_.device();
    torch::Tensor scene_center =
        scene_center_.to(dev).to(torch::kFloat32).contiguous().view({3});
    torch::Tensor scene_extent =
        scene_extent_.to(dev).to(torch::kFloat32).contiguous().view({1});
    torch::Tensor scene_min = scene_center - 0.5f * scene_extent;

    const float finest_scale = std::ldexp(1.0f, -max_num_levels_);
    torch::Tensor finest_vox = scene_extent * finest_scale;
    return (scene_min.view({1, 3}) +
            grid_pts_key_.to(torch::kFloat32) * finest_vox.view({1, 1}))
        .contiguous();
}

std::tuple<torch::Tensor, torch::Tensor>
VoxelModel::buildSvreconExtractionGrid(int max_octree_level) const
{
    TORCH_CHECK(oct_path_.defined() && oct_level_.defined(),
                "buildSvreconExtractionGrid: octree topology is not initialized");
    TORCH_CHECK(scene_center_.defined() && scene_extent_.defined() && inside_extent_.defined(),
                "buildSvreconExtractionGrid: scene bounds are not initialized");

    auto octpath = oct_path_.reshape({-1, 1}).to(torch::kLong).contiguous();
    auto octlevel = oct_level_.reshape({-1, 1}).to(torch::kLong).contiguous();
    auto source_vox_key = vox_key_;
    if (is_leaf_.defined() && is_leaf_.size(0) == octpath.size(0)) {
        auto leaf_idx = torch::nonzero(is_leaf_.to(octpath.device())
                                           .to(torch::kBool).reshape({-1}))
                            .reshape({-1}).to(torch::kLong);
        octpath = octpath.index_select(0, leaf_idx);
        octlevel = octlevel.index_select(0, leaf_idx);
        source_vox_key = source_vox_key.index_select(
            0, leaf_idx.to(source_vox_key.device())).contiguous();
    }

    // Match SVRecon extract_mesh.py: retain cells touching the inside extent,
    // then clamp the adaptive topology to outside_level + final_lv.
    const auto current_grid_xyz = gridPointsWorld();
    const auto inside_min = scene_center_.view({1, 3}) - 0.5f * inside_extent_.view({1, 1});
    const auto inside_max = scene_center_.view({1, 3}) + 0.5f * inside_extent_.view({1, 1});
    const auto grid_inside =
        ((current_grid_xyz >= inside_min) & (current_grid_xyz <= inside_max)).all(1);
    const auto vox_inside = grid_inside.index({source_vox_key}).any(1);
    const auto inside_idx = torch::nonzero(vox_inside).view({-1}).to(torch::kLong);
    octpath = octpath.index_select(0, inside_idx);
    octlevel = octlevel.index_select(0, inside_idx);

    const int target_level = std::clamp(max_octree_level, 0, max_num_levels_);
    const int num_bits_to_mask = 3 * std::max(0, max_num_levels_ - target_level);
    if (num_bits_to_mask > 0) {
        const int64_t mask_scale = int64_t{1} << num_bits_to_mask;
        octpath = torch::floor_divide(octpath, mask_scale) * mask_scale;
    }
    octlevel = octlevel.clamp_max(target_level);

    auto topology = torch::cat({octpath, octlevel}, 1).contiguous();
    auto unique_result = at::unique_dim(
        topology,
        /*dim=*/0,
        /*sorted=*/true,
        /*return_inverse=*/false,
        /*return_counts=*/false);
    topology = std::get<0>(unique_result).contiguous();
    octpath = topology.index({torch::indexing::Slice(), 0}).view({-1, 1}).contiguous();
    octlevel = topology.index({torch::indexing::Slice(), 1})
                   .view({-1, 1})
                   .to(torch::kInt8)
                   .contiguous();

    auto [grid_pts_key, vox_key] =
        buildGridPtsLink(octpath, octlevel, max_num_levels_);
    const auto scene_min = scene_center_.view({1, 3}) - 0.5f * scene_extent_.view({1, 1});
    const float finest_scale = std::ldexp(1.0f, -max_num_levels_);
    const auto finest_vox = scene_extent_.view({1, 1}) * finest_scale;
    auto grid_xyz =
        (scene_min + grid_pts_key.to(torch::kFloat32) * finest_vox).contiguous();
    return {grid_xyz, vox_key.contiguous()};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
VoxelModel::buildSvreconDenseExtractionGrid(int inside_level) const
{
    TORCH_CHECK(scene_center_.defined() && scene_extent_.defined() && inside_extent_.defined(),
                "buildSvreconDenseExtractionGrid: scene bounds are not initialized");
    const int use_inside_level = std::clamp(
        inside_level,
        1,
        std::max(1, max_num_levels_ - outside_level_));
    auto dev = scene_center_.device();
    auto [octpath, octlevel] = genSvreconDenseOctpath(
        outside_level_,
        use_inside_level,
        max_num_levels_,
        dev);
    auto [grid_pts_key, vox_key] =
        buildGridPtsLink(octpath, octlevel, max_num_levels_);

    const auto scene_min = scene_center_.view({1, 3}) - 0.5f * scene_extent_.view({1, 1});
    const float finest_scale = std::ldexp(1.0f, -max_num_levels_);
    const auto finest_vox = scene_extent_.view({1, 1}) * finest_scale;
    auto grid_xyz =
        (scene_min + grid_pts_key.to(torch::kFloat32) * finest_vox).contiguous();

    // Match extract_mesh_progressive(): keep dense cells touching the inside box.
    const auto inside_min =
        scene_center_.view({1, 3}) - 0.5f * inside_extent_.view({1, 1});
    const auto inside_max =
        scene_center_.view({1, 3}) + 0.5f * inside_extent_.view({1, 1});
    const auto grid_inside = ((grid_xyz >= inside_min) & (grid_xyz <= inside_max)).all(1);
    const auto vox_inside = grid_inside.index({vox_key}).any(1);
    const auto inside_idx = torch::nonzero(vox_inside).view({-1}).to(torch::kLong);
    octpath = octpath.index_select(0, inside_idx).contiguous();
    octlevel = octlevel.index_select(0, inside_idx).contiguous();

    std::tie(grid_pts_key, vox_key) =
        buildGridPtsLink(octpath, octlevel, max_num_levels_);
    grid_xyz =
        (scene_min + grid_pts_key.to(torch::kFloat32) * finest_vox).contiguous();
    return {
        octpath,
        octlevel,
        grid_xyz,
        vox_key.contiguous()};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
VoxelModel::subdivideSvreconExtractionGrid(
    const torch::Tensor& octpath,
    const torch::Tensor& octlevel) const
{
    TORCH_CHECK(scene_center_.defined() && scene_extent_.defined(),
                "subdivideSvreconExtractionGrid: scene bounds are not initialized");
    TORCH_CHECK(octpath.defined() && octlevel.defined() &&
                    octpath.size(0) == octlevel.size(0),
                "subdivideSvreconExtractionGrid: invalid topology");
    auto [child_path, child_level] = genChildrenOctpath(
        octpath.to(scene_center_.device()).contiguous(),
        octlevel.to(scene_center_.device()).contiguous(),
        max_num_levels_);
    auto [grid_pts_key, vox_key] =
        buildGridPtsLink(child_path, child_level, max_num_levels_);
    const auto scene_min = scene_center_.view({1, 3}) - 0.5f * scene_extent_.view({1, 1});
    const float finest_scale = std::ldexp(1.0f, -max_num_levels_);
    const auto finest_vox = scene_extent_.view({1, 1}) * finest_scale;
    auto grid_xyz =
        (scene_min + grid_pts_key.to(torch::kFloat32) * finest_vox).contiguous();
    return {
        child_path.contiguous(),
        child_level.contiguous(),
        grid_xyz,
        vox_key.contiguous()};
}

std::tuple<torch::Tensor, torch::Tensor>
VoxelModel::querySdfTrilinear(const torch::Tensor& points_world) const
{
    auto dev = _geo_grid_pts_.defined() ? _geo_grid_pts_.device() : points_world.device();
    auto value_opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
    auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);

    if (!points_world.defined() || points_world.numel() == 0) {
        return {
            torch::empty({0}, value_opts),
            torch::empty({0}, bool_opts)
        };
    }

    torch::Tensor pts = points_world.to(dev).to(torch::kFloat32).reshape({-1, 3}).contiguous();
    const int64_t P = pts.size(0);
    torch::Tensor sdf_out = torch::zeros({P}, value_opts);
    torch::Tensor valid_out = torch::zeros({P}, bool_opts);

    if (!_geo_grid_pts_.defined() || _geo_grid_pts_.numel() == 0 ||
        !vox_key_.defined() || vox_key_.dim() != 2 || vox_key_.size(1) != 8 ||
        !oct_path_.defined() || !oct_level_.defined() ||
        oct_path_.numel() == 0 || oct_level_.numel() == 0 ||
        !scene_center_.defined() || scene_center_.numel() != 3 ||
        !scene_extent_.defined() || scene_extent_.numel() != 1 ||
        max_num_levels_ <= 0 || max_num_levels_ > 20) {
        return {sdf_out, valid_out};
    }

    torch::Tensor scene_center =
        scene_center_.to(dev).to(torch::kFloat32).contiguous().view({3});
    torch::Tensor scene_extent =
        scene_extent_.to(dev).to(torch::kFloat32).contiguous().view({1});
    torch::Tensor scene_min = scene_center - 0.5f * scene_extent;
    const float scene_extent_f =
        scene_extent_.detach().to(torch::kCPU).to(torch::kFloat32).view({1}).item<float>();
    if (!std::isfinite(scene_extent_f) || scene_extent_f <= 0.0f) {
        return {sdf_out, valid_out};
    }

    torch::Tensor octpath = oct_path_.to(dev).to(torch::kLong).view({-1}).contiguous();
    torch::Tensor octlevel = oct_level_.to(dev).to(torch::kLong).view({-1}).contiguous();
    torch::Tensor vox_key = vox_key_.to(dev).to(torch::kLong).contiguous();
    if (is_leaf_.defined() && is_leaf_.size(0) == octpath.size(0)) {
        auto leaf_idx = torch::nonzero(is_leaf_.to(dev).to(torch::kBool).reshape({-1}))
                            .reshape({-1}).to(torch::kLong);
        octpath = octpath.index_select(0, leaf_idx);
        octlevel = octlevel.index_select(0, leaf_idx);
        vox_key = vox_key.index_select(0, leaf_idx);
    }
    if (octpath.size(0) != octlevel.size(0) || octpath.size(0) != vox_key.size(0)) {
        return {sdf_out, valid_out};
    }

    torch::Tensor active_keys = (octpath * 256 + octlevel).to(torch::kLong).contiguous();
    auto sorted = torch::sort(active_keys);
    torch::Tensor sorted_keys = std::get<0>(sorted).contiguous();
    torch::Tensor sorted_voxel_idx = std::get<1>(sorted).to(torch::kLong).contiguous();

    torch::Tensor levels_cpu = octlevel.detach().to(torch::kCPU).contiguous();
    std::vector<int64_t> levels(
        levels_cpu.data_ptr<int64_t>(),
        levels_cpu.data_ptr<int64_t>() + levels_cpu.numel());
    std::sort(levels.begin(), levels.end(), std::greater<int64_t>());
    levels.erase(std::unique(levels.begin(), levels.end()), levels.end());

    torch::Tensor geo = _geo_grid_pts_.to(dev).to(torch::kFloat32).view({-1});

    for (const int64_t level : levels) {
        if (level < 0 || level > max_num_levels_) {
            continue;
        }
        const int64_t grid_dim = 1LL << static_cast<int>(level);
        const float vox_size_f = std::ldexp(scene_extent_f, -static_cast<int>(level));
        if (!std::isfinite(vox_size_f) || vox_size_f <= 0.0f) {
            continue;
        }

        torch::Tensor vox_size =
            torch::full({1}, vox_size_f, value_opts);
        torch::Tensor coord =
            (pts - scene_min.view({1, 3})) / vox_size.view({1, 1});
        torch::Tensor ijk = torch::floor(coord).to(torch::kLong).contiguous();
        torch::Tensor in_bounds =
            (ijk >= 0).all(/*dim=*/1) &
            (ijk < grid_dim).all(/*dim=*/1) &
            torch::logical_not(valid_out);
        if (!in_bounds.any().item<bool>()) {
            continue;
        }

        torch::Tensor candidate_point_idx =
            torch::nonzero(in_bounds).view({-1}).to(torch::kLong).contiguous();
        torch::Tensor ijk_candidates =
            ijk.index_select(0, candidate_point_idx).contiguous();
        torch::Tensor level_candidates =
            torch::full(
                {ijk_candidates.size(0), 1},
                static_cast<int64_t>(level),
                torch::TensorOptions().dtype(torch::kInt8).device(dev));
        torch::Tensor candidate_octpath =
            SVRECON_UTILS::ijk_2_octpath(ijk_candidates, level_candidates)
                .to(torch::kLong)
                .view({-1})
                .contiguous();
        torch::Tensor candidate_keys =
            (candidate_octpath * 256 + level).to(torch::kLong).contiguous();

        torch::Tensor pos =
            at::searchsorted(
                sorted_keys,
                candidate_keys,
                /*out_int32=*/false,
                /*right=*/false)
                .to(torch::kLong)
                .contiguous();
        torch::Tensor in_range = pos < sorted_keys.size(0);
        torch::Tensor pos_clamped =
            pos.clamp(0, std::max<int64_t>(0, sorted_keys.size(0) - 1)).contiguous();
        torch::Tensor key_at_pos = sorted_keys.index_select(0, pos_clamped);
        torch::Tensor matched =
            (in_range & (key_at_pos == candidate_keys)).to(torch::kBool);
        if (!matched.any().item<bool>()) {
            continue;
        }

        torch::Tensor matched_candidate_idx =
            torch::nonzero(matched).view({-1}).to(torch::kLong).contiguous();
        torch::Tensor sample_idx =
            candidate_point_idx.index_select(0, matched_candidate_idx).contiguous();
        torch::Tensor voxel_idx =
            sorted_voxel_idx.index_select(
                0,
                pos.index_select(0, matched_candidate_idx).to(torch::kLong)).contiguous();

        torch::Tensor ijk_matched =
            ijk_candidates.index_select(0, matched_candidate_idx).to(torch::kFloat32);
        torch::Tensor pts_matched = pts.index_select(0, sample_idx);
        torch::Tensor local_min =
            scene_min.view({1, 3}) + ijk_matched * vox_size.view({1, 1});
        torch::Tensor frac =
            ((pts_matched - local_min) / vox_size.view({1, 1})).clamp(0.0f, 1.0f);
        torch::Tensor fx = frac.index({torch::indexing::Slice(), 0});
        torch::Tensor fy = frac.index({torch::indexing::Slice(), 1});
        torch::Tensor fz = frac.index({torch::indexing::Slice(), 2});
        torch::Tensor one_x = 1.0f - fx;
        torch::Tensor one_y = 1.0f - fy;
        torch::Tensor one_z = 1.0f - fz;
        torch::Tensor weights = torch::stack(
            {
                one_x * one_y * one_z,
                one_x * one_y * fz,
                one_x * fy * one_z,
                one_x * fy * fz,
                fx * one_y * one_z,
                fx * one_y * fz,
                fx * fy * one_z,
                fx * fy * fz
            },
            /*dim=*/1);

        torch::Tensor corner_idx =
            vox_key.index_select(0, voxel_idx).reshape({-1}).contiguous();
        torch::Tensor corner_sdf =
            geo.index_select(0, corner_idx)
                .view({voxel_idx.size(0), 8})
                .contiguous();
        torch::Tensor sdf_interp =
            (corner_sdf * weights).sum(/*dim=*/1).contiguous();

        sdf_out.index_put_({sample_idx}, sdf_interp);
        valid_out.index_put_(
            {sample_idx},
            torch::ones({sample_idx.size(0)}, bool_opts));
    }

    return {sdf_out, valid_out};
}

torch::Tensor VoxelModel::voxelDensityMean() const
{
    TORCH_CHECK(_geo_grid_pts_.defined(), "_geo_grid_pts_ not defined");
    TORCH_CHECK(vox_key_.defined(), "vox_key_ not defined");
    // 1) Flatten grid scalar: [Mg,1] -> [Mg]
    auto geo_flat = _geo_grid_pts_.view({-1});  // [Mg]
    // 2) Flatten voxel keys: [Nv,8] -> [Nv*8] (long)
    auto vk_long = vox_key_.to(torch::kLong).view({-1}); // [Nv*8]
    // 3) Gather 8 corner densities per voxel:
    auto geo_corners = geo_flat.index_select(0, vk_long); // [Nv*8]
    // 4) Reshape to [Nv, 8] and average:
    const auto Nv = vox_key_.size(0);
    auto geo_per_voxel = geo_corners.view({Nv, 8}).mean(1); // [Nv]
    return geo_per_voxel;  // pre-activation densities per voxel
}

torch::Tensor VoxelModel::voxSize() const {
    torch::Tensor out = (size_.dim() == 1) ? size_.unsqueeze(1) : size_; // [N,1]
    // Light stats; item<>() syncs but is fine occasionally
    auto flat = out.view(-1);
    float minv = flat.min().item<float>();
    float maxv = flat.max().item<float>();
    float mean = flat.mean().item<float>();
    // std::cout << "[DBG][voxSize] shape=" << out.sizes()
    //           << " N=" << out.size(0)
    //           << " min/mean/max=" << minv << "/" << mean << "/" << maxv
    //           << std::endl;
    return out;
}

torch::Tensor VoxelModel::octLevel() const {
    // ensure [N,1] int8
    if (oct_level_.dim() == 1) return oct_level_.unsqueeze(1);
    return oct_level_;
}

torch::Tensor VoxelModel::octPath() const {
    return oct_path_;
}

int VoxelModel::numVoxels() const {
    return static_cast<int>(center_.size(0));
}

torch::Tensor VoxelModel::activeRenderableMask() const
{
    const int64_t N = center_.defined() ? center_.size(0) : 0;
    auto opts = torch::TensorOptions().dtype(torch::kBool).device(device_type_);
    if (N <= 0) {
        return torch::empty({0}, opts);
    }
    return torch::ones({N}, opts);
}

int VoxelModel::maxNumLevels() const {
    return max_num_levels_;
}

torch::Tensor VoxelModel::SceneCenter() const {
    return this->scene_center_;
}

torch::Tensor VoxelModel::SceneExtent() const {
    return this->scene_extent_;
}

float VoxelModel::insertionVoxSize() const
{
    if (vox_eff_.defined() && vox_eff_.numel() > 0) {
        return vox_eff_.detach().reshape({-1})[0].item<float>();
    }
    return fixed_vox_size_;
}

torch::Tensor VoxelModel::InsideExtent() const {
    return this->inside_extent_;
}

void VoxelModel::oneUpShDegree()
{
    // '''
    // sh_degree_add1 from svraster 
    // '''
    if (this->active_sh_degree_ < this->max_sh_degree_)
        this->active_sh_degree_ += 1;
}

void VoxelModel::setShDegree(const int sh)
{
    this->active_sh_degree_ = (sh > this->max_sh_degree_ ? this->max_sh_degree_ : sh);
}

const torch::Tensor& sv::VoxelModel::svrasterSdfGridPts() const { return svraster_sdf_grid_pts_; }
const torch::Tensor& sv::VoxelModel::svrasterSdfWeights() const { return svraster_sdf_weights_; }
bool VoxelModel::hasSvrasterSdfField() const
{
    return svraster_sdf_grid_pts_.defined() &&
           svraster_sdf_weights_.defined() &&
           svraster_sdf_grid_pts_.dim() == 2 &&
           svraster_sdf_weights_.dim() == 2 &&
           svraster_sdf_grid_pts_.size(1) == 1 &&
           svraster_sdf_weights_.size(1) == 1 &&
           grid_pts_key_.defined() &&
           svraster_sdf_grid_pts_.size(0) == grid_pts_key_.size(0) &&
           svraster_sdf_weights_.size(0) == grid_pts_key_.size(0);
}

void VoxelModel::setEmptySvrasterSdfField_()
{
    auto dev = _geo_grid_pts_.defined()
        ? _geo_grid_pts_.device()
        : torch::Device(device_type_);
    auto value_opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
    svraster_sdf_grid_pts_ = torch::empty({0, 1}, value_opts);
    svraster_sdf_weights_ = torch::empty({0, 1}, value_opts);
}

torch::Tensor VoxelModel::voxelCornerScalarFromGrid_(const torch::Tensor& grid_scalar) const
{
    const int64_t N = (vox_key_.defined() && vox_key_.dim() == 2) ? vox_key_.size(0) : 0;
    auto dev = (grid_scalar.defined() ? grid_scalar.device() :
                (_geo_grid_pts_.defined() ? _geo_grid_pts_.device() : torch::Device(device_type_)));
    auto value_opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
    if (N == 0 ||
        !grid_scalar.defined() ||
        grid_scalar.numel() == 0 ||
        !vox_key_.defined() ||
        vox_key_.dim() != 2 ||
        vox_key_.size(1) != 8) {
        return torch::zeros({N, 8, 1}, value_opts);
    }

    auto scalar = grid_scalar.to(dev).to(torch::kFloat32).reshape({-1, 1}).contiguous();
    return scalar.index({vox_key_.to(dev).to(torch::kLong)}).contiguous();
}

torch::Tensor VoxelModel::voxelGeoCorners() const
{
    torch::Tensor corners = voxelCornerScalarFromGrid_(_geo_grid_pts_);
    if (corners.defined() && corners.dim() == 3 && corners.size(2) == 1) {
        corners = corners.squeeze(2);
    }
    return corners.contiguous().detach();
}

torch::Tensor VoxelModel::voxelSdfWeightCorners() const
{
    torch::Tensor corners = voxelCornerScalarFromGrid_(svraster_sdf_weights_);
    if (corners.defined() && corners.dim() == 3 && corners.size(2) == 1) {
        corners = corners.squeeze(2);
    }
    return corners.contiguous().detach();
}

void VoxelModel::rebuildSvrasterSdfFieldFromVoxelCorners_(
    const torch::Tensor& voxel_sdf_values,
    const torch::Tensor& voxel_sdf_weights)
{
    torch::NoGradGuard no_grad;
    if (!grid_pts_key_.defined() ||
        grid_pts_key_.dim() != 2 ||
        grid_pts_key_.size(1) != 3 ||
        !vox_key_.defined() ||
        vox_key_.dim() != 2 ||
        vox_key_.size(1) != 8) {
        setEmptySvrasterSdfField_();
        return;
    }

    const int64_t M = grid_pts_key_.size(0);
    const int64_t N = vox_key_.size(0);
    auto dev = _geo_grid_pts_.defined() ? _geo_grid_pts_.device() : grid_pts_key_.device();
    auto value_opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
    if (M == 0) {
        svraster_sdf_grid_pts_ = torch::empty({0, 1}, value_opts);
        svraster_sdf_weights_ = torch::empty({0, 1}, value_opts);
        return;
    }
    if (N == 0 ||
        !voxel_sdf_values.defined() ||
        !voxel_sdf_weights.defined() ||
        voxel_sdf_values.size(0) != N ||
        voxel_sdf_values.size(1) != 8 ||
        voxel_sdf_weights.size(0) != N ||
        voxel_sdf_weights.size(1) != 8) {
        svraster_sdf_grid_pts_ = torch::zeros({M, 1}, value_opts);
        svraster_sdf_weights_ = torch::zeros({M, 1}, value_opts);
        return;
    }

    auto sdf_vox = voxel_sdf_values.to(dev).to(torch::kFloat32);
    auto w_vox = voxel_sdf_weights.to(dev).to(torch::kFloat32);
    if (sdf_vox.dim() == 2) {
        sdf_vox = sdf_vox.unsqueeze(-1);
    }
    if (w_vox.dim() == 2) {
        w_vox = w_vox.unsqueeze(-1);
    }

    svraster_sdf_grid_pts_ =
        aggregateVoxelCornersIntoGridPts(M, vox_key_.to(dev), sdf_vox).detach().contiguous();
    svraster_sdf_weights_ =
        aggregateVoxelCornersIntoGridPts(M, vox_key_.to(dev), w_vox).detach().contiguous();
}

void VoxelModel::ensureSvrasterSdfField()
{
    if (!grid_pts_key_.defined() || grid_pts_key_.dim() != 2 || grid_pts_key_.size(1) != 3) {
        setEmptySvrasterSdfField_();
        return;
    }

    const int64_t M = grid_pts_key_.size(0);
    auto tensor_dev = _geo_grid_pts_.defined() ? _geo_grid_pts_.device() : grid_pts_key_.device();
    auto value_opts = torch::TensorOptions().dtype(torch::kFloat32).device(tensor_dev);

    auto make_empty_values = [&]() {
        svraster_sdf_grid_pts_ = torch::zeros({M, 1}, value_opts);
        svraster_sdf_weights_ = torch::zeros({M, 1}, value_opts);
    };

    if (M == 0) {
        make_empty_values();
        return;
    }

    if (svraster_sdf_grid_pts_.defined() &&
        svraster_sdf_weights_.defined() &&
        svraster_sdf_grid_pts_.size(0) == M &&
        svraster_sdf_weights_.size(0) == M) {
        if (svraster_sdf_grid_pts_.device() != tensor_dev) {
            svraster_sdf_grid_pts_ = svraster_sdf_grid_pts_.to(tensor_dev).contiguous();
            svraster_sdf_weights_ = svraster_sdf_weights_.to(tensor_dev).contiguous();
        }
        return;
    }

    make_empty_values();
}

void VoxelModel::resetSvrasterSdfField()
{
    if (!grid_pts_key_.defined() || grid_pts_key_.dim() != 2 || grid_pts_key_.size(1) != 3) {
        setEmptySvrasterSdfField_();
        return;
    }

    const int64_t M = grid_pts_key_.size(0);
    auto tensor_dev = _geo_grid_pts_.defined() ? _geo_grid_pts_.device() : grid_pts_key_.device();
    auto value_opts = torch::TensorOptions().dtype(torch::kFloat32).device(tensor_dev);
    svraster_sdf_grid_pts_ = torch::zeros({M, 1}, value_opts);
    svraster_sdf_weights_ = torch::zeros({M, 1}, value_opts);
}

void VoxelModel::fuseSvrasterSdfGridSamples(
    const torch::Tensor& tsdf_values,
    const torch::Tensor& weights,
    const torch::Tensor& valid_mask,
    float max_weight)
{
    ensureSvrasterSdfField();
    if (!hasSvrasterSdfField() || grid_pts_key_.size(0) == 0) {
        return;
    }

    const int64_t M = grid_pts_key_.size(0);
    torch::NoGradGuard no_grad;
    auto dev = svraster_sdf_grid_pts_.device();
    torch::Tensor sample_tsdf = tsdf_values.to(dev).to(torch::kFloat32).reshape({M, 1});
    torch::Tensor sample_w = weights.to(dev).to(torch::kFloat32).reshape({M, 1});
    sample_w = torch::where(
        torch::isfinite(sample_w),
        sample_w,
        torch::zeros_like(sample_w));
    torch::Tensor valid = valid_mask.to(dev).to(torch::kBool).reshape({M, 1}) &
                          torch::isfinite(sample_tsdf) &
                          (sample_w > 0.0f);

    torch::Tensor old_w = svraster_sdf_weights_;
    torch::Tensor old_sdf = svraster_sdf_grid_pts_;
    torch::Tensor fused_w = old_w + sample_w;
    torch::Tensor fused_sdf =
        (old_sdf * old_w + sample_tsdf * sample_w) / fused_w.clamp_min(1.0e-6f);
    if (max_weight > 0.0f) {
        fused_w = fused_w.clamp_max(max_weight);
    }

    svraster_sdf_grid_pts_ = torch::where(valid, fused_sdf, old_sdf).contiguous();
    svraster_sdf_weights_ = torch::where(valid, fused_w, old_w).contiguous();
}

torch::Tensor VoxelModel::makeGeoGridInitRows_(
    const torch::Tensor& grid_pts_key_new,
    int64_t begin,
    int64_t end,
    float default_value)
{
    TORCH_CHECK(grid_pts_key_new.defined() &&
                    grid_pts_key_new.dim() == 2 &&
                    grid_pts_key_new.size(1) == 3,
                "makeGeoGridInitRows_: grid_pts_key_new must be [M,3]");
    TORCH_CHECK(begin >= 0 && end >= begin && end <= grid_pts_key_new.size(0),
                "makeGeoGridInitRows_: invalid row range");

    torch::Tensor key_rows =
        grid_pts_key_new.slice(/*dim=*/0, begin, end).contiguous();
    return makePointPriorSdfInitRowsForKeys_(key_rows, default_value);
}

void VoxelModel::appendSparseSupportPoints_(const torch::Tensor& points)
{
    if (!points.defined() || points.numel() == 0) {
        return;
    }
    auto dev = _geo_grid_pts_.defined()
        ? _geo_grid_pts_.device()
        : torch::Device(device_type_);
    torch::Tensor pts =
        points.to(dev).to(torch::kFloat32).reshape({-1, 3}).contiguous();
    if (pts.numel() == 0) {
        return;
    }

    if (!sparse_points_xyz_.defined() || sparse_points_xyz_.numel() == 0) {
        sparse_points_xyz_ = pts;
    } else {
        sparse_points_xyz_ =
            torch::cat({
                sparse_points_xyz_.to(dev).to(torch::kFloat32).reshape({-1, 3}).contiguous(),
                pts},
                0).contiguous();
    }

}

void VoxelModel::updateExistingSupportSdfFromPoints_(
    const torch::Tensor& support_xyz,
    const torch::Tensor& support_rgb,
    const std::vector<sv::MiniCam>& cams)
{
    if (!support_xyz.defined() ||
        support_xyz.numel() == 0 ||
        !grid_pts_key_.defined() ||
        grid_pts_key_.dim() != 2 ||
        grid_pts_key_.size(1) != 3 ||
        !_geo_grid_pts_.defined() ||
        _geo_grid_pts_.size(0) != grid_pts_key_.size(0)) {
        return;
    }

    auto dev = _geo_grid_pts_.device();
    torch::Tensor pts =
        support_xyz.to(dev).to(torch::kFloat32).reshape({-1, 3}).contiguous();
    if (pts.size(0) == 0) {
        return;
    }
    torch::Tensor finite_pts =
        torch::isfinite(pts).all(/*dim=*/1).to(torch::kBool);
    torch::Tensor finite_idx =
        torch::nonzero(finite_pts).view({-1}).to(torch::kLong);
    if (finite_idx.numel() == 0) {
        return;
    }
    pts = pts.index_select(0, finite_idx).contiguous();
    (void)support_rgb;

    torch::Tensor scene_min =
        scene_center_.to(dev).to(torch::kFloat32).contiguous().view({3}) -
        0.5f * scene_extent_.to(dev).to(torch::kFloat32).contiguous().view({1});
    const float finest_scale = std::ldexp(1.0f, -max_num_levels_);
    torch::Tensor finest_vox =
        scene_extent_.to(dev).to(torch::kFloat32).contiguous().view({1}) * finest_scale;
    torch::Tensor grid_xyz =
        scene_min.view({1, 3}) +
        grid_pts_key_.to(dev).to(torch::kFloat32) * finest_vox.view({1, 1});
    grid_xyz = grid_xyz.contiguous();

    const float radius_m =
        std::max(0.0f, sdf_initialization_orb_radius_vox_) * fixed_vox_size_;
    if (radius_m <= 0.0f) {
        return;
    }

    // This is an exact spatial rejection, not an approximation: no grid
    // corner outside the point-batch bounds expanded by the support radius can
    // pass the nearest-distance test below.
    torch::Tensor support_min =
        std::get<0>(pts.min(/*dim=*/0, /*keepdim=*/false)) - radius_m;
    torch::Tensor support_max =
        std::get<0>(pts.max(/*dim=*/0, /*keepdim=*/false)) + radius_m;
    torch::Tensor in_support_bounds =
        (grid_xyz >= support_min.view({1, 3})).all(/*dim=*/1) &
        (grid_xyz <= support_max.view({1, 3})).all(/*dim=*/1);
    torch::Tensor update_idx =
        torch::nonzero(in_support_bounds)
            .view({-1}).to(torch::kLong).contiguous();
    if (update_idx.numel() == 0) {
        return;
    }

    // Restore the visibility-first SVRecon path used by the archived fast
    // experiments. Keep projection on the tensor device and transfer only
    // visible corners to the CPU nearest-neighbor query.
    if (!cams.empty()) {
        torch::Tensor candidate_xyz =
            grid_xyz.index_select(0, update_idx).contiguous();
        torch::Tensor visible = torch::zeros(
            {candidate_xyz.size(0)},
            torch::TensorOptions().dtype(torch::kBool).device(dev));
        for (const auto& cam : cams) {
            if (cam.width <= 0 || cam.height <= 0 ||
                cam.fx <= 1.0e-6f || cam.fy <= 1.0e-6f ||
                !cam.w2c.defined() || cam.w2c.numel() < 16) {
                continue;
            }
            torch::Tensor w2c =
                cam.w2c.to(dev).to(torch::kFloat32).contiguous();
            torch::Tensor R = w2c.index({
                torch::indexing::Slice(0, 3),
                torch::indexing::Slice(0, 3)});
            torch::Tensor t =
                w2c.index({torch::indexing::Slice(0, 3), 3}).view({1, 3});
            torch::Tensor xc =
                torch::matmul(candidate_xyz, R.transpose(0, 1)) + t;
            torch::Tensor z = xc.index({torch::indexing::Slice(), 2});
            torch::Tensor z_safe = z.clamp_min(1.0e-6f);
            torch::Tensor u =
                cam.fx *
                    (xc.index({torch::indexing::Slice(), 0}) / z_safe) +
                cam.cx;
            torch::Tensor v =
                cam.fy *
                    (xc.index({torch::indexing::Slice(), 1}) / z_safe) +
                cam.cy;
            torch::Tensor in_img =
                (z > 1.0e-6f) &
                (u >= 0.0f) &
                (u < static_cast<float>(cam.width)) &
                (v >= 0.0f) &
                (v < static_cast<float>(cam.height));
            visible = visible | in_img.to(torch::kBool);
        }
        torch::Tensor visible_local =
            torch::nonzero(visible).view({-1}).to(torch::kLong).contiguous();
        if (visible_local.numel() == 0) {
            return;
        }
        update_idx =
            update_idx.index_select(0, visible_local).contiguous();
    }

    torch::Tensor query_xyz =
        grid_xyz.index_select(0, update_idx)
            .to(torch::kCPU)
            .to(torch::kFloat32)
            .contiguous();
    torch::Tensor support_cpu =
        pts.to(torch::kCPU).to(torch::kFloat32).contiguous();
    cv::Mat support_mat(
        static_cast<int>(support_cpu.size(0)),
        3,
        CV_32F,
        support_cpu.data_ptr<float>());
    cv::Mat query_mat(
        static_cast<int>(query_xyz.size(0)),
        3,
        CV_32F,
        query_xyz.data_ptr<float>());
    cv::Mat nearest_indices(query_mat.rows, 1, CV_32S);
    cv::Mat nearest_sq_dist(query_mat.rows, 1, CV_32F);
    cv::flann::Index point_index(
        support_mat,
        cv::flann::KDTreeIndexParams(4));
    point_index.knnSearch(
        query_mat,
        nearest_indices,
        nearest_sq_dist,
        1,
        cv::flann::SearchParams(64));

    torch::Tensor nearest =
        torch::from_blob(
            nearest_sq_dist.ptr<float>(),
            {static_cast<int64_t>(nearest_sq_dist.rows), 1},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
            .clone()
            .clamp_min_(0.0f)
            .sqrt_();
    torch::Tensor near_idx_cpu =
        torch::nonzero(nearest.view({-1}) <= radius_m)
            .view({-1}).to(torch::kLong).contiguous();
    if (near_idx_cpu.numel() == 0) {
        return;
    }
    update_idx =
        update_idx.index_select(0, near_idx_cpu.to(dev)).contiguous();
    torch::Tensor near_query_xyz =
        query_xyz.index_select(0, near_idx_cpu).to(dev).contiguous();
    torch::Tensor near_dist =
        nearest.index_select(0, near_idx_cpu).to(dev).contiguous();

    auto old_cams = sdf_init_cams_;
    sdf_init_cams_ = cams;
    torch::Tensor measured_sdf =
        (visibilitySignedPointPriorSdf_(
             near_query_xyz,
             near_dist,
             pts,
             std::max(1.0e-4f, 0.5f * fixed_vox_size_)) * 2.0f)
            .detach()
            .to(dev)
            .to(torch::kFloat32)
            .reshape({-1, 1})
            .contiguous();
    sdf_init_cams_ = old_cams;

    ensureSvrasterSdfField();
    if (!svraster_sdf_grid_pts_.defined() ||
        !svraster_sdf_weights_.defined() ||
        svraster_sdf_grid_pts_.size(0) != _geo_grid_pts_.size(0) ||
        svraster_sdf_weights_.size(0) != _geo_grid_pts_.size(0)) {
        resetSvrasterSdfField();
    }

    torch::NoGradGuard no_grad;
    torch::Tensor old_w =
        svraster_sdf_weights_.index_select(0, update_idx).to(torch::kFloat32).contiguous();
    torch::Tensor old_sdf =
        _geo_grid_pts_.detach().index_select(0, update_idx).to(torch::kFloat32).contiguous();
    torch::Tensor new_w = old_w + 1.0f;
    torch::Tensor fused_sdf =
        (old_sdf * old_w + measured_sdf) / new_w.clamp_min(1.0e-6f);

    _geo_grid_pts_.index_put_({update_idx}, fused_sdf);
    svraster_sdf_grid_pts_.index_put_({update_idx}, fused_sdf.detach());
    svraster_sdf_weights_.index_put_({update_idx}, new_w.detach().clamp_max(100.0f));
}

torch::Tensor VoxelModel::visibilitySignedPointPriorSdf_(
    const torch::Tensor& grid_xyz,
    const torch::Tensor& dist,
    const torch::Tensor& points,
    float surface_band) const
{
    TORCH_CHECK(grid_xyz.defined() && grid_xyz.dim() == 2 && grid_xyz.size(1) == 3,
                "visibilitySignedPointPriorSdf_: grid_xyz must be [N,3]");
    TORCH_CHECK(dist.defined() && dist.dim() == 2 && dist.size(1) == 1,
                "visibilitySignedPointPriorSdf_: dist must be [N,1]");

    const int64_t rows = grid_xyz.size(0);
    auto dev = grid_xyz.device();
    auto dist_cpu = dist.to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto signed_sdf_cpu = -dist_cpu.clone();
    if (rows == 0 || sdf_init_cams_.empty() ||
        !points.defined() || points.numel() == 0) {
        return signed_sdf_cpu.to(dev).contiguous();
    }

    torch::Tensor support =
        points.to(torch::kCPU).to(torch::kFloat32).reshape({-1, 3}).contiguous();
    if (support.numel() == 0) {
        return signed_sdf_cpu.to(dev).contiguous();
    }

    (void)surface_band;
    auto dist_acc = dist_cpu.accessor<float, 2>();
    std::vector<int64_t> candidate_rows;
    candidate_rows.reserve(static_cast<size_t>(rows));
    for (int64_t i = 0; i < rows; ++i) {
        // Match SVRecon's initialize_sdf_from_sfm(): initialize every grid
        // point as negative nearest-prior distance, then run visibility sign
        // flipping over all negative SDF values. Do not restrict signing to a
        // narrow surface band, otherwise far visible free-space can remain
        // incorrectly negative/occupied.
        if (std::isfinite(dist_acc[i][0]) && dist_acc[i][0] > 0.0f) {
            candidate_rows.push_back(i);
        }
    }
    if (candidate_rows.empty()) {
        return signed_sdf_cpu.to(dev).contiguous();
    }

    auto grid_cpu = grid_xyz.to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto grid_acc = grid_cpu.accessor<float, 2>();
    auto support_acc = support.accessor<float, 2>();
    auto sdf_acc = signed_sdf_cpu.accessor<float, 2>();
    const int64_t C = static_cast<int64_t>(candidate_rows.size());
    std::vector<uint8_t> was_visible(static_cast<size_t>(C), 0);
    std::vector<uint8_t> flip_positive(static_cast<size_t>(C), 0);

    const int bin_max = 64;
    const float eps = 1.0e-4f;
    const int64_t support_rows = support.size(0);

    for (const auto& cam : sdf_init_cams_) {
        if (cam.width <= 0 || cam.height <= 0 ||
            cam.fx <= 1.0e-6f || cam.fy <= 1.0e-6f ||
            !cam.w2c.defined() || cam.w2c.numel() < 16) {
            continue;
        }

        const int W = cam.width;
        const int H = cam.height;
        const float scale =
            static_cast<float>(bin_max) / static_cast<float>(std::max(W, H));
        const int bin_w = std::max(1, static_cast<int>(std::llround(W * scale)));
        const int bin_h = std::max(1, static_cast<int>(std::llround(H * scale)));
        const int64_t n_bins = static_cast<int64_t>(bin_w) * static_cast<int64_t>(bin_h);
        std::vector<float> depth_bins(
            static_cast<size_t>(n_bins),
            std::numeric_limits<float>::infinity());

        torch::Tensor w2c_cpu = cam.w2c.to(torch::kCPU).to(torch::kFloat32).contiguous();
        auto w2c = w2c_cpu.accessor<float, 2>();

        auto project = [&](const float x,
                           const float y,
                           const float z_world,
                           float& z_cam,
                           float& u,
                           float& v) -> bool {
            const float xc =
                w2c[0][0] * x + w2c[0][1] * y + w2c[0][2] * z_world + w2c[0][3];
            const float yc =
                w2c[1][0] * x + w2c[1][1] * y + w2c[1][2] * z_world + w2c[1][3];
            z_cam =
                w2c[2][0] * x + w2c[2][1] * y + w2c[2][2] * z_world + w2c[2][3];
            if (!std::isfinite(z_cam) || z_cam <= 1.0e-6f) {
                return false;
            }
            u = cam.fx * xc / z_cam + cam.cx;
            v = cam.fy * yc / z_cam + cam.cy;
            return std::isfinite(u) && std::isfinite(v) &&
                   u >= 0.0f && u < static_cast<float>(W) &&
                   v >= 0.0f && v < static_cast<float>(H);
        };

        for (int64_t i = 0; i < support_rows; ++i) {
            float z_cam = 0.0f;
            float u = 0.0f;
            float v = 0.0f;
            if (!project(support_acc[i][0], support_acc[i][1], support_acc[i][2],
                         z_cam, u, v)) {
                continue;
            }
            const int ui_bin = std::clamp(
                static_cast<int>(std::floor(u * static_cast<float>(bin_w) / static_cast<float>(W))),
                0,
                bin_w - 1);
            const int vi_bin = std::clamp(
                static_cast<int>(std::floor(v * static_cast<float>(bin_h) / static_cast<float>(H))),
                0,
                bin_h - 1);
            const int64_t bin_idx = static_cast<int64_t>(vi_bin) * bin_w + ui_bin;
            float& d0 = depth_bins[static_cast<size_t>(bin_idx)];
            if (z_cam < d0) {
                d0 = z_cam;
            }
        }

        for (int64_t local = 0; local < C; ++local) {
            const int64_t row = candidate_rows[static_cast<size_t>(local)];
            float z_cam = 0.0f;
            float u = 0.0f;
            float v = 0.0f;
            if (!project(grid_acc[row][0], grid_acc[row][1], grid_acc[row][2],
                         z_cam, u, v)) {
                continue;
            }
            was_visible[static_cast<size_t>(local)] = 1;
            const int ui_bin = std::clamp(
                static_cast<int>(std::floor(u * static_cast<float>(bin_w) / static_cast<float>(W))),
                0,
                bin_w - 1);
            const int vi_bin = std::clamp(
                static_cast<int>(std::floor(v * static_cast<float>(bin_h) / static_cast<float>(H))),
                0,
                bin_h - 1);
            const int64_t bin_idx = static_cast<int64_t>(vi_bin) * bin_w + ui_bin;
            const float front_depth = depth_bins[static_cast<size_t>(bin_idx)];
            if (z_cam < front_depth - eps) {
                flip_positive[static_cast<size_t>(local)] = 1;
            }
        }
    }

    for (int64_t local = 0; local < C; ++local) {
        const int64_t row = candidate_rows[static_cast<size_t>(local)];
        const bool positive =
            flip_positive[static_cast<size_t>(local)] ||
            !was_visible[static_cast<size_t>(local)];
        const float abs_dist = std::abs(dist_acc[row][0]);
        sdf_acc[row][0] = positive ? abs_dist : -abs_dist;
    }
    return signed_sdf_cpu.to(dev).contiguous();
}


torch::Tensor VoxelModel::makePointPriorSdfInitRowsForKeys_(
    const torch::Tensor& grid_pts_key_rows,
    float fallback_value)
{
    TORCH_CHECK(grid_pts_key_rows.defined() &&
                    grid_pts_key_rows.dim() == 2 &&
                    grid_pts_key_rows.size(1) == 3,
                "makePointPriorSdfInitRowsForKeys_: grid_pts_key_rows must be [M,3]");

    const int64_t rows = grid_pts_key_rows.size(0);
    auto opts = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(grid_pts_key_rows.device());
    if (rows == 0) {
        return torch::empty({0, 1}, opts).requires_grad_();
    }

    if (pending_sdf_init_mode_ == SdfInitMode::WeakPositive) {
        const float weak_sdf =
            std::max(
                1.0e-6f,
                positive_unknown_sdf_init_vox_ * fixed_vox_size_);
        return torch::full({rows, 1}, weak_sdf, opts).requires_grad_();
    }

    torch::Tensor points;
    if (sdf_init_local_support_points_.defined() &&
        sdf_init_local_support_points_.numel() > 0) {
        points = sdf_init_local_support_points_
                     .to(grid_pts_key_rows.device())
                     .to(torch::kFloat32)
                     .reshape({-1, 3})
                     .contiguous();
    } else {
        std::vector<torch::Tensor> support_parts;
        if (sparse_points_xyz_.defined() && sparse_points_xyz_.numel() > 0) {
            support_parts.push_back(sparse_points_xyz_);
        }
        std::vector<torch::Tensor> support_on_dev;
        support_on_dev.reserve(support_parts.size());
        for (const auto& p : support_parts) {
            support_on_dev.push_back(
                p.to(grid_pts_key_rows.device()).to(torch::kFloat32).reshape({-1, 3}).contiguous());
        }
        if (!support_on_dev.empty()) {
            points = torch::cat(support_on_dev, 0).contiguous();
        }
    }
    if (!points.defined() || points.numel() == 0) {
        return torch::full({rows, 1}, fallback_value, opts).requires_grad_();
    }

    auto dev = grid_pts_key_rows.device();
    torch::Tensor scene_center =
        scene_center_.to(dev).to(torch::kFloat32).contiguous().view({3});
    torch::Tensor scene_extent =
        scene_extent_.to(dev).to(torch::kFloat32).contiguous().view({1});
    torch::Tensor scene_min = scene_center - 0.5f * scene_extent;
    const float finest_scale = std::ldexp(1.0f, -max_num_levels_);
    torch::Tensor finest_vox = scene_extent * finest_scale;
    torch::Tensor grid_xyz =
        scene_min.view({1, 3}) +
        grid_pts_key_rows.to(dev).to(torch::kFloat32) * finest_vox.view({1, 1});
    grid_xyz = grid_xyz.contiguous();

    torch::NoGradGuard no_grad;
    points = points.to(dev).to(torch::kFloat32).contiguous().view({-1, 3});
    // Use the chunked CUDA nearest-prior calculation from the archived
    // SVRecon runs. It is equivalent to the KD-tree nearest-distance query but
    // avoids rebuilding and transferring a CPU index after every map update.
    const int64_t query_chunk = 2048;
    const int64_t point_chunk = 4096;
    std::vector<torch::Tensor> chunks;
    chunks.reserve(
        static_cast<size_t>((rows + query_chunk - 1) / query_chunk));
    for (int64_t begin = 0; begin < rows; begin += query_chunk) {
        const int64_t end = std::min<int64_t>(rows, begin + query_chunk);
        torch::Tensor query =
            grid_xyz.slice(/*dim=*/0, begin, end).contiguous();
        torch::Tensor best = torch::full(
            {end - begin, 1},
            std::numeric_limits<float>::infinity(),
            opts);
        for (int64_t point_begin = 0;
             point_begin < points.size(0);
             point_begin += point_chunk) {
            const int64_t point_end = std::min<int64_t>(
                points.size(0), point_begin + point_chunk);
            torch::Tensor point_rows =
                points.slice(/*dim=*/0, point_begin, point_end).contiguous();
            torch::Tensor distances = torch::cdist(query, point_rows);
            torch::Tensor nearest =
                std::get<0>(distances.min(/*dim=*/1, /*keepdim=*/true));
            best = torch::minimum(best, nearest);
        }
        chunks.push_back(best.contiguous());
    }
    torch::Tensor dist = torch::cat(chunks, 0).contiguous();
    const float surface_band = std::max(1.0e-4f, 0.5f * fixed_vox_size_);
    torch::Tensor signed_prior =
        visibilitySignedPointPriorSdf_(grid_xyz, dist, points, surface_band)
            .contiguous();
    // Match SVRecon's visibility-signed nearest-point initialization. Its
    // initializer scales the signed distance by two after assigning signs.
    torch::Tensor sdf = (signed_prior * 2.0f).contiguous();
    const float weak_delta_sdf =
        std::max(1.0e-7f, 2.5e-4f * fixed_vox_size_);
    const float weak_center_sdf = 2.0f * weak_delta_sdf;
    torch::Tensor weak_positive_sdf =
        (weak_center_sdf +
         weak_delta_sdf * torch::tanh(signed_prior / surface_band))
            .contiguous();
    torch::Tensor weak_signed_sdf =
        (weak_delta_sdf * torch::tanh(signed_prior / surface_band))
            .contiguous();
    if (pending_sdf_init_mode_ == SdfInitMode::WeakSurfacePrior) {
        // Match the useful property of SVRaster's raw density=-10: the new
        // candidate is barely visible but remains above the rasterizer's
        // alpha cutoff and can receive a photometric geometry gradient. Keep
        // every corner positive initially: the directional SDF ramp is a
        // trainable candidate, not a zero-crossing surface. Subdivision will
        // select it only after optimization moves part of the field negative.
        // At 5 cm and log_s=0.3 the ramp gives roughly the same alpha order as
        // density=-10, without introducing a second opacity parameter.
        sdf = weak_positive_sdf;
    }
    if (pending_sdf_init_mode_ == SdfInitMode::OrbPriorOnly) {
        const float radius_m =
            std::max(0.0f, sdf_initialization_orb_radius_vox_) * fixed_vox_size_;
        torch::Tensor local_orb = dist <= radius_m;
        // Only nearby ORB points provide a strong metric prior. Unsupported
        // corners retain only a tiny zero-centered signed ramp derived from the
        // ORB field, so inactive/RGB-D allocations can render weakly without
        // becoming geometric priors themselves.
        sdf = torch::where(local_orb, sdf, weak_signed_sdf).contiguous();
    }
    if (pending_sdf_init_mode_ == SdfInitMode::OrbPriorWeakCandidate) {
        const float radius_m =
            std::max(0.0f, sdf_initialization_orb_radius_vox_) * fixed_vox_size_;
        torch::Tensor local_orb = dist <= radius_m;
        // Block metadata is not renderable. A promoted near-surface candidate
        // receives the metric ORB prior only when ORB support is local;
        // otherwise it starts as a weak all-positive field. It can receive a
        // photometric gradient but cannot be treated as a surface or be
        // subdivided until optimization creates a strict sign crossing.
        sdf = torch::where(local_orb, sdf, weak_positive_sdf).contiguous();
    }
    return sdf.detach().requires_grad_();
}

void VoxelModel::setNextSdfInitializationGridSamples(
    const torch::Tensor& grid_points_world,
    const torch::Tensor& sdf_values)
{
    next_sdf_init_grid_keys_ = torch::Tensor();
    next_sdf_init_grid_values_ = torch::Tensor();
    if (!grid_points_world.defined() || !sdf_values.defined() ||
        grid_points_world.numel() == 0 ||
        grid_points_world.numel() / 3 != sdf_values.numel() ||
        !scene_center_.defined() || !scene_extent_.defined() ||
        max_num_levels_ <= 0 || max_num_levels_ > 20) {
        return;
    }

    torch::NoGradGuard no_grad;
    auto dev = scene_center_.device();
    torch::Tensor points = grid_points_world.to(dev).to(torch::kFloat32)
                               .reshape({-1, 3}).contiguous();
    torch::Tensor values = sdf_values.to(dev).to(torch::kFloat32)
                               .reshape({-1, 1}).contiguous();
    torch::Tensor scene_min =
        scene_center_.to(dev).to(torch::kFloat32).reshape({1, 3}) -
        0.5f * scene_extent_.to(dev).to(torch::kFloat32).reshape({1, 1});
    const float finest_voxel =
        scene_extent_.reshape({-1})[0].item<float>() *
        std::ldexp(1.0f, -max_num_levels_);
    if (!std::isfinite(finest_voxel) || finest_voxel <= 0.0f) {
        return;
    }

    torch::Tensor grid_float = (points - scene_min) / finest_voxel;
    torch::Tensor grid_keys = grid_float.round().to(torch::kInt64).contiguous();
    const int64_t grid_dim = 1LL << max_num_levels_;
    torch::Tensor valid =
        torch::isfinite(points).all(/*dim=*/1) &
        torch::isfinite(values).reshape({-1}) &
        ((grid_float - grid_keys.to(torch::kFloat32)).abs() < 1.0e-3f)
            .all(/*dim=*/1) &
        (grid_keys >= 0).all(/*dim=*/1) &
        (grid_keys <= grid_dim).all(/*dim=*/1);
    torch::Tensor keep = torch::nonzero(valid).reshape({-1}).to(torch::kLong);
    if (keep.numel() == 0) {
        return;
    }
    next_sdf_init_grid_keys_ = grid_keys.index_select(0, keep).contiguous();
    next_sdf_init_grid_values_ = values.index_select(0, keep).contiguous();
}

torch::Tensor VoxelModel::applyPendingSdfGridInitialization_(
    const torch::Tensor& grid_pts_key_rows,
    const torch::Tensor& initial_values) const
{
    if (!next_sdf_init_grid_keys_.defined() ||
        !next_sdf_init_grid_values_.defined() ||
        next_sdf_init_grid_keys_.numel() == 0 ||
        grid_pts_key_rows.numel() == 0) {
        return initial_values;
    }

    auto dev = grid_pts_key_rows.device();
    const int64_t key_base = (1LL << max_num_levels_) + 1LL;
    auto encode = [&](const torch::Tensor& keys_in) {
        using torch::indexing::Slice;
        torch::Tensor keys = keys_in.to(dev).to(torch::kInt64).contiguous();
        return ((keys.index({Slice(), 0}) * key_base +
                 keys.index({Slice(), 1})) * key_base +
                keys.index({Slice(), 2})).contiguous();
    };

    torch::Tensor pending_linear = encode(next_sdf_init_grid_keys_);
    auto sorted_result = pending_linear.sort(/*dim=*/0);
    torch::Tensor sorted_linear = std::get<0>(sorted_result).contiguous();
    torch::Tensor sorted_order = std::get<1>(sorted_result).to(torch::kLong);
    torch::Tensor row_linear = encode(grid_pts_key_rows);
    torch::Tensor positions = at::searchsorted(
        sorted_linear, row_linear, /*out_int32=*/false, /*right=*/false)
                                  .to(torch::kLong).contiguous();
    torch::Tensor in_range = positions < sorted_linear.size(0);
    torch::Tensor clamped = positions.clamp(
        0, std::max<int64_t>(0, sorted_linear.size(0) - 1));
    torch::Tensor matched =
        in_range & (sorted_linear.index_select(0, clamped) == row_linear);
    torch::Tensor row_idx = torch::nonzero(matched).reshape({-1}).to(torch::kLong);
    if (row_idx.numel() == 0) {
        return initial_values;
    }

    torch::Tensor pending_idx = sorted_order.index_select(
        0, positions.index_select(0, row_idx));
    torch::Tensor direct_values = next_sdf_init_grid_values_
                                      .to(dev).to(torch::kFloat32)
                                      .reshape({-1, 1})
                                      .index_select(0, pending_idx)
                                      .contiguous();
    torch::Tensor output = initial_values.clone().contiguous();
    output.index_put_({row_idx}, direct_values);
    return output.contiguous();
}

void VoxelModel::rebuildGeoGridForNewGridKeys_(
    const torch::Tensor& grid_pts_key_new,
    float default_value)
{
    TORCH_CHECK(grid_pts_key_new.defined() &&
                    grid_pts_key_new.dim() == 2 &&
                    grid_pts_key_new.size(1) == 3,
                "rebuildGeoGridForNewGridKeys_: grid_pts_key_new must be [M,3]");

    const int64_t M_new = grid_pts_key_new.size(0);
    auto dev = grid_pts_key_new.device();
    const bool old_valid =
        grid_pts_key_.defined() &&
        grid_pts_key_.dim() == 2 &&
        grid_pts_key_.size(1) == 3 &&
        _geo_grid_pts_.defined() &&
        _geo_grid_pts_.size(0) == grid_pts_key_.size(0);

    if (!old_valid || grid_pts_key_.size(0) == 0) {
        torch::Tensor new_geo =
            makePointPriorSdfInitRowsForKeys_(grid_pts_key_new.contiguous(), default_value)
                .detach()
                .contiguous();
        new_geo = applyPendingSdfGridInitialization_(
            grid_pts_key_new.contiguous(), new_geo).detach().contiguous();
        _geo_grid_pts_ = new_geo.contiguous().detach().requires_grad_(true);
        adam_geo_.exp_avg = torch::zeros_like(new_geo);
        adam_geo_.exp_avg_sq = torch::zeros_like(new_geo);
        if (svraster_sdf_grid_pts_.defined() || svraster_sdf_weights_.defined()) {
            svraster_sdf_grid_pts_ = torch::zeros_like(new_geo);
            svraster_sdf_weights_ = torch::zeros_like(new_geo);
        }
        return;
    }
    if (M_new == 0) {
        _geo_grid_pts_ = torch::empty(
            {0, 1},
            torch::TensorOptions().dtype(torch::kFloat32).device(dev)).requires_grad_(true);
        adam_geo_.exp_avg = torch::zeros_like(_geo_grid_pts_);
        adam_geo_.exp_avg_sq = torch::zeros_like(_geo_grid_pts_);
        if (svraster_sdf_grid_pts_.defined() || svraster_sdf_weights_.defined()) {
            svraster_sdf_grid_pts_ = torch::zeros_like(_geo_grid_pts_);
            svraster_sdf_weights_ = torch::zeros_like(_geo_grid_pts_);
        }
        return;
    }

    TORCH_CHECK(max_num_levels_ > 0 && max_num_levels_ <= 20,
                "rebuildGeoGridForNewGridKeys_: max_num_levels_ is too large for int64 key encoding");

    const int64_t grid_dim = 1LL << static_cast<int>(max_num_levels_);
    auto encode_keys = [&](const torch::Tensor& keys_in) {
        using torch::indexing::Slice;
        torch::Tensor keys = keys_in.to(dev).to(torch::kInt64).contiguous();
        return ((keys.index({Slice(), 0}) * grid_dim + keys.index({Slice(), 1})) *
                    grid_dim +
                keys.index({Slice(), 2}))
            .to(torch::kInt64)
            .contiguous();
    };

    torch::Tensor old_linear = encode_keys(grid_pts_key_);
    torch::Tensor new_linear = encode_keys(grid_pts_key_new);
    auto pos = at::searchsorted(
                   old_linear,
                   new_linear,
                   /*out_int32=*/false,
                   /*right=*/false)
                   .to(torch::kLong)
                   .contiguous();

    torch::Tensor in_range = pos < old_linear.size(0);
    torch::Tensor pos_clamped =
        pos.clamp(0, std::max<int64_t>(0, old_linear.size(0) - 1)).contiguous();
    torch::Tensor old_at_pos = old_linear.index_select(0, pos_clamped);
    torch::Tensor matched = (in_range & (old_at_pos == new_linear)).to(torch::kBool);
    torch::Tensor matched_new_idx =
        torch::nonzero(matched).view({-1}).to(torch::kLong).contiguous();
    torch::Tensor unmatched_new_idx =
        torch::nonzero(~matched).view({-1}).to(torch::kLong).contiguous();
    torch::Tensor matched_old_idx;

    torch::Tensor new_geo = torch::empty(
        {M_new, 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(dev));

    if (matched_new_idx.numel() > 0) {
        matched_old_idx =
            pos.index_select(0, matched_new_idx).to(torch::kLong).contiguous();
        torch::Tensor old_geo =
            _geo_grid_pts_.to(dev).to(torch::kFloat32).contiguous();
        if (old_geo.dim() == 1) {
            old_geo = old_geo.view({-1, 1}).contiguous();
        }
        new_geo.index_put_(
            {matched_new_idx},
            old_geo.index_select(0, matched_old_idx).contiguous());
    } else {
        matched_old_idx = torch::empty(
            {0}, torch::TensorOptions().dtype(torch::kLong).device(dev));
    }
    if (unmatched_new_idx.numel() > 0) {
        torch::Tensor unmatched_keys =
            grid_pts_key_new.index_select(0, unmatched_new_idx).contiguous();
        torch::Tensor unmatched_geo =
            makePointPriorSdfInitRowsForKeys_(unmatched_keys, default_value)
                .detach()
                .contiguous();
        unmatched_geo = applyPendingSdfGridInitialization_(
            unmatched_keys, unmatched_geo).detach().contiguous();
        new_geo.index_put_({unmatched_new_idx}, unmatched_geo);
    }

    auto zero_or_remap = [&](const torch::Tensor& old_state_in) -> torch::Tensor {
        torch::Tensor new_state = torch::zeros_like(new_geo);
        if (!old_state_in.defined() ||
            old_state_in.size(0) != old_linear.size(0) ||
            matched_new_idx.numel() == 0) {
            return new_state.contiguous();
        }
        torch::Tensor old_state = old_state_in.to(dev).to(torch::kFloat32).contiguous();
        if (old_state.dim() == 1) {
            old_state = old_state.view({-1, 1}).contiguous();
        }
        new_state.index_put_(
            {matched_new_idx},
            old_state.index_select(0, matched_old_idx).contiguous());
        return new_state.contiguous();
    };

    adam_geo_.exp_avg = zero_or_remap(adam_geo_.exp_avg);
    adam_geo_.exp_avg_sq = zero_or_remap(adam_geo_.exp_avg_sq);

    if (svraster_sdf_grid_pts_.defined() || svraster_sdf_weights_.defined()) {
        auto value_opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
        torch::Tensor new_sdf = torch::zeros({M_new, 1}, value_opts);
        torch::Tensor new_w = torch::zeros({M_new, 1}, value_opts);
        if (matched_new_idx.numel() > 0 &&
            svraster_sdf_grid_pts_.defined() &&
            svraster_sdf_weights_.defined() &&
            svraster_sdf_grid_pts_.size(0) == old_linear.size(0) &&
            svraster_sdf_weights_.size(0) == old_linear.size(0)) {
            torch::Tensor old_sdf =
                svraster_sdf_grid_pts_.to(dev).to(torch::kFloat32).contiguous();
            torch::Tensor old_w =
                svraster_sdf_weights_.to(dev).to(torch::kFloat32).contiguous();
            if (old_sdf.dim() == 1) {
                old_sdf = old_sdf.view({-1, 1}).contiguous();
            }
            if (old_w.dim() == 1) {
                old_w = old_w.view({-1, 1}).contiguous();
            }
            new_sdf.index_put_(
                {matched_new_idx},
                old_sdf.index_select(0, matched_old_idx).contiguous());
            new_w.index_put_(
                {matched_new_idx},
                old_w.index_select(0, matched_old_idx).contiguous());
        }
        svraster_sdf_grid_pts_ = new_sdf.contiguous();
        svraster_sdf_weights_ = new_w.contiguous();
    }

    _geo_grid_pts_ = new_geo.contiguous().detach().requires_grad_(true);
}

void VoxelModel::applyGeoGridRawInit(
    const torch::Tensor& raw_values,
    const torch::Tensor& valid_mask)
{
    if (!_geo_grid_pts_.defined() || _geo_grid_pts_.dim() != 2 || _geo_grid_pts_.size(1) != 1) {
        return;
    }
    const int64_t M = _geo_grid_pts_.size(0);
    if (M == 0) {
        return;
    }

    torch::NoGradGuard no_grad;
    auto dev = _geo_grid_pts_.device();
    torch::Tensor raw = raw_values.to(dev).to(torch::kFloat32).reshape({M, 1});
    torch::Tensor valid = valid_mask.to(dev).to(torch::kBool).reshape({M, 1}) &
                          torch::isfinite(raw);

    torch::Tensor updated = torch::where(valid, raw, _geo_grid_pts_).contiguous();
    _geo_grid_pts_.copy_(updated);
}

std::pair<torch::Tensor, torch::Tensor> VoxelModel::rgbdHoleSupportCellCenters(
    const torch::Tensor& surface_points_world) const
{
    if (!surface_points_world.defined() || surface_points_world.numel() == 0 ||
        !scene_min_t_.defined() || !vox_eff_.defined()) {
        return {torch::Tensor(), torch::Tensor()};
    }

    auto dev = scene_min_t_.device();
    torch::Tensor surface =
        surface_points_world.to(dev).to(torch::kFloat32).reshape({-1, 3}).contiguous();
    const float cell_size = vox_eff_.reshape({-1})[0].item<float>();
    if (surface.size(0) == 0 || !std::isfinite(cell_size) || cell_size <= 0.0f) {
        return {torch::Tensor(), torch::Tensor()};
    }

    torch::Tensor scene_min =
        scene_min_t_.to(dev).to(torch::kFloat32).reshape({1, 3}).contiguous();
    torch::Tensor surface_ijk =
        ((surface - scene_min) / cell_size).floor().to(torch::kLong).contiguous();
    // RGB-D render-hole filling allocates only the cell containing the measured
    // surface. Neighboring cells are not required merely to close a 2-D hole and
    // previously produced front/back layers of free-space candidates.
    torch::Tensor support_ijk = surface_ijk.contiguous();
    torch::Tensor source_idx =
        torch::arange(
            surface.size(0),
            torch::TensorOptions().dtype(torch::kLong).device(dev))
            .contiguous();
    const int64_t limit = 1LL << static_cast<int>(octlevel_);
    torch::Tensor valid =
        ((support_ijk >= 0) & (support_ijk < limit)).all(/*dim=*/1);
    support_ijk = support_ijk.index({valid}).contiguous();
    source_idx = source_idx.index({valid}).contiguous();
    if (support_ijk.size(0) == 0) {
        return {torch::Tensor(), torch::Tensor()};
    }
    return {
        (scene_min +
         (support_ijk.to(torch::kFloat32) + 0.5f) * cell_size)
            .contiguous(),
        source_idx};
}

int64_t VoxelModel::applyRgbdHoleSdfEvidence(
    const sv::MiniCam& cam,
    const torch::Tensor& depth_meters,
    const torch::Tensor& hole_mask,
    const float truncation_m)
{
    if (!_geo_grid_pts_.defined() || _geo_grid_pts_.numel() == 0 ||
        !grid_pts_key_.defined() || grid_pts_key_.size(0) != _geo_grid_pts_.size(0) ||
        !depth_meters.defined() || !hole_mask.defined() ||
        cam.width <= 0 || cam.height <= 0 ||
        cam.fx <= 1.0e-6f || cam.fy <= 1.0e-6f ||
        !cam.w2c.defined() || cam.w2c.numel() < 16) {
        return 0;
    }

    const int64_t expected_pixels =
        static_cast<int64_t>(cam.width) * static_cast<int64_t>(cam.height);
    if (depth_meters.numel() != expected_pixels || hole_mask.numel() != expected_pixels) {
        return 0;
    }

    auto dev = _geo_grid_pts_.device();
    torch::Tensor depth =
        depth_meters.to(dev).to(torch::kFloat32).reshape({-1}).contiguous();
    torch::Tensor holes =
        hole_mask.to(dev).to(torch::kBool).reshape({-1}).contiguous();
    // Grid corners around a selected surface cell project a few pixels away
    // from the original hole ray. Include that local image neighborhood when
    // assigning the shared corner SDF values.
    torch::Tensor hole_image = holes.view({cam.height, cam.width});
    torch::Tensor dilated_holes = hole_image.clone();
    constexpr int kHoleDilationRadius = 4;
    for (int dy = -kHoleDilationRadius; dy <= kHoleDilationRadius; ++dy) {
        const int dst_y0 = std::max(0, dy);
        const int dst_y1 = std::min(cam.height, cam.height + dy);
        const int src_y0 = std::max(0, -dy);
        const int src_y1 = std::min(cam.height, cam.height - dy);
        for (int dx = -kHoleDilationRadius; dx <= kHoleDilationRadius; ++dx) {
            const int dst_x0 = std::max(0, dx);
            const int dst_x1 = std::min(cam.width, cam.width + dx);
            const int src_x0 = std::max(0, -dx);
            const int src_x1 = std::min(cam.width, cam.width - dx);
            if (dst_y0 >= dst_y1 || dst_x0 >= dst_x1) {
                continue;
            }
            using torch::indexing::Slice;
            torch::Tensor dst = dilated_holes.index(
                {Slice(dst_y0, dst_y1), Slice(dst_x0, dst_x1)});
            torch::Tensor src = hole_image.index(
                {Slice(src_y0, src_y1), Slice(src_x0, src_x1)});
            dilated_holes.index_put_(
                {Slice(dst_y0, dst_y1), Slice(dst_x0, dst_x1)},
                dst | src);
        }
    }
    holes = dilated_holes.reshape({-1}).contiguous();

    torch::Tensor scene_min =
        scene_center_.to(dev).to(torch::kFloat32).reshape({3}) -
        0.5f * scene_extent_.to(dev).to(torch::kFloat32).reshape({1});
    const float finest_vox =
        scene_extent_.reshape({-1})[0].item<float>() *
        std::ldexp(1.0f, -max_num_levels_);
    torch::Tensor grid_xyz =
        scene_min.view({1, 3}) +
        grid_pts_key_.to(dev).to(torch::kFloat32) * finest_vox;

    torch::Tensor w2c = cam.w2c.to(dev).to(torch::kFloat32).contiguous();
    torch::Tensor R = w2c.index({torch::indexing::Slice(0, 3),
                                 torch::indexing::Slice(0, 3)});
    torch::Tensor t = w2c.index({torch::indexing::Slice(0, 3), 3}).view({1, 3});
    torch::Tensor camera_xyz = torch::matmul(grid_xyz, R.transpose(0, 1)) + t;
    torch::Tensor z = camera_xyz.index({torch::indexing::Slice(), 2});
    torch::Tensor z_safe = z.clamp_min(1.0e-6f);
    torch::Tensor u =
        (cam.fx * camera_xyz.index({torch::indexing::Slice(), 0}) / z_safe + cam.cx)
            .round().to(torch::kLong);
    torch::Tensor v =
        (cam.fy * camera_xyz.index({torch::indexing::Slice(), 1}) / z_safe + cam.cy)
            .round().to(torch::kLong);
    torch::Tensor in_image =
        (z > 1.0e-6f) &
        (u >= 0) & (u < cam.width) &
        (v >= 0) & (v < cam.height);
    torch::Tensor pixel =
        (v.clamp(0, cam.height - 1) * cam.width +
         u.clamp(0, cam.width - 1)).to(torch::kLong);
    torch::Tensor measured_depth = depth.index_select(0, pixel);
    torch::Tensor projected_hole = holes.index_select(0, pixel);
    torch::Tensor measured_sdf = measured_depth - z;
    const float trunc = std::max(1.0e-4f, truncation_m);
    torch::Tensor valid =
        in_image & projected_hole &
        torch::isfinite(measured_depth) &
        (measured_depth > 0.0f) &
        torch::isfinite(measured_sdf) &
        (measured_sdf.abs() <= trunc);
    torch::Tensor update_idx =
        torch::nonzero(valid).view({-1}).to(torch::kLong).contiguous();
    if (update_idx.numel() == 0) {
        return 0;
    }

    torch::Tensor sample =
        measured_sdf.index_select(0, update_idx)
            .clamp(-trunc, trunc).view({-1, 1}).contiguous();
    ensureSvrasterSdfField();
    torch::NoGradGuard no_grad;
    torch::Tensor old_w =
        svraster_sdf_weights_.index_select(0, update_idx)
            .to(torch::kFloat32).contiguous();
    torch::Tensor old_evidence =
        svraster_sdf_grid_pts_.index_select(0, update_idx)
            .to(torch::kFloat32).contiguous();
    torch::Tensor new_w = (old_w + 1.0f).clamp_max(100.0f);
    torch::Tensor fused =
        torch::where(
            old_w > 0.0f,
            (old_evidence * old_w + sample) / (old_w + 1.0f),
            sample)
            .contiguous();

    // Hole closure is an initialization/update of the learnable SDF from the
    // current valid RGB-D observation. Do not average the learnable geometry
    // against stale pre-loop or pre-subdivision evidence; retain the weighted
    // average only in the auxiliary evidence buffers.
    _geo_grid_pts_.index_put_({update_idx}, sample);
    svraster_sdf_grid_pts_.index_put_({update_idx}, fused);
    svraster_sdf_weights_.index_put_({update_idx}, new_w);
    if (adam_geo_.exp_avg.defined() &&
        adam_geo_.exp_avg.size(0) == _geo_grid_pts_.size(0)) {
        adam_geo_.exp_avg.index_put_({update_idx}, 0.0f);
    }
    if (adam_geo_.exp_avg_sq.defined() &&
        adam_geo_.exp_avg_sq.size(0) == _geo_grid_pts_.size(0)) {
        adam_geo_.exp_avg_sq.index_put_({update_idx}, 0.0f);
    }
    return update_idx.numel();
}

void VoxelModel::refreshSvreconLogSTargetFromVoxelSize(const bool initialize_current)
{
    if (!size_.defined() || size_.numel() == 0) {
        return;
    }

    const float min_vox_size = size_.detach().to(torch::kFloat32).min().item<float>();
    if (!std::isfinite(min_vox_size) || min_vox_size <= 0.0f) {
        return;
    }

    // SVRecon initializes its global SDF sharpness from the finest voxel size.
    constexpr float learning_thickness = 2.0f;
    const float target_log_s = 0.1f * std::log(
        std::log(99.0f) /
        (min_vox_size * learning_thickness * 2.0f));
    svrecon_log_s_target_ = target_log_s;

    float current_log_s = svrecon_log_s_current_;
    if (!std::isfinite(current_log_s) && log_s_.defined() && log_s_.numel() > 0) {
        current_log_s = log_s_.detach().reshape({-1})[0].item<float>();
    }
    if (initialize_current || !std::isfinite(current_log_s)) {
        torch::NoGradGuard no_grad;
        if (!log_s_.defined() || log_s_.numel() == 0) {
            log_s_ = torch::full(
                {1}, target_log_s,
                torch::TensorOptions().dtype(torch::kFloat32).device(size_.device()));
        } else {
            log_s_.fill_(target_log_s);
        }
        current_log_s = target_log_s;
        svrecon_log_s_current_ = target_log_s;
        adam_log_s_ = AdamGroupState{};
    }

}

void VoxelModel::advanceSvreconLogSTowardTarget(const float max_delta)
{
    if (max_delta <= 0.0f ||
        !std::isfinite(svrecon_log_s_target_) ||
        !log_s_.defined() || log_s_.numel() == 0) {
        return;
    }

    float current = svrecon_log_s_current_;
    if (!std::isfinite(current)) {
        current = log_s_.detach().reshape({-1})[0].item<float>();
        svrecon_log_s_current_ = current;
    }
    if (!std::isfinite(current) || current >= svrecon_log_s_target_) {
        return;
    }

    const float next = std::min(svrecon_log_s_target_, current + max_delta);
    torch::NoGradGuard no_grad;
    log_s_.fill_(next);
    svrecon_log_s_current_ = next;
}

namespace fs = std::filesystem;

void VoxelModel::createFromPcd(
    const std::map<point3D_id_t, Point3D>& pcd,
    const std::vector<sv::MiniCam>& cams)
{
    sdf_init_cams_ = cams;

    TORCH_CHECK(global_scene_extent_ > 0.f, "global_scene_extent_ must be set (>0).");
    TORCH_CHECK(fixed_vox_size_   > 0.f, "fixed_vox_size_ must be set (>0).");
    TORCH_CHECK(max_sh_degree_ >= 0, "max_sh_degree_ must be >= 0.");

    const int N = static_cast<int>(pcd.size());
    const int32_t current_kf_count =
        cams.empty() ? static_cast<int32_t>(-1) : static_cast<int32_t>(cams.size());
    if (N == 0) {
        std::cerr << "[createFromPcd] Empty PCD — nothing to do.\n";
        return;
    }

    auto dev = torch::kCUDA;
    outside_level_ = std::clamp(outside_level_, 0, max_num_levels_);

    // Pack inputs
    torch::Tensor xyz = torch::empty({N,3}, torch::dtype(torch::kFloat32).device(dev));
    torch::Tensor rgb = torch::empty({N,3}, torch::dtype(torch::kFloat32).device(dev));
    {
        int i = 0;
        for (const auto& kv : pcd) {
            const auto& P = kv.second;
            xyz[i][0] = P.xyz_(0);
            xyz[i][1] = P.xyz_(1);
            xyz[i][2] = P.xyz_(2);
            // P.color_ already scaled? If it’s 0..255, divide; if already 0..1, keep.
            // std::cout << "Point " << i << " color before scaling: "
            //           << P.color_(0) << ", " << P.color_(1) << ", " << P.color_(2) << "\n";
            rgb[i][0] = P.color_(0);
            rgb[i][1] = P.color_(1);
            rgb[i][2] = P.color_(2);
            ++i;
        }
    }
    // Accept both [0,1] and [0,255] color inputs (same policy as increasePcd).
    if (rgb.numel() > 0) {
        const float cmax = rgb.max().item<float>();
        if (cmax > 1.5f) {
            rgb.div_(255.0f);
        }
    }
    rgb.clamp_(0.0f, 1.0f);
    sparse_points_xyz_ = xyz.detach().to(torch::kFloat32).contiguous();
    sparse_points_color_ = rgb.detach().to(torch::kFloat32).contiguous();
    sdf_init_local_support_points_ = torch::Tensor();
    pending_sdf_init_mode_ = topology_sdf_init_mode_;

    // Reset accumulated real-point history. We seed it later from actually
    // inserted/filtered real voxels (not raw pre-filter PCD).
    real_pcd_points_accum_cpu_ = torch::empty(
        {0, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    // ------------------------------------------------------------------------
    // 0.5) Initialize global PCD bounds.
    // ------------------------------------------------------------------------
    {
        // Work on CPU for simplicity; this runs only once at initialization.
        auto xyz_cpu = xyz.to(torch::kCPU).contiguous();  // [N,3]

        auto min_res = xyz_cpu.min(/*dim=*/0, /*keepdim=*/false);
        auto max_res = xyz_cpu.max(/*dim=*/0, /*keepdim=*/false);

        torch::Tensor min_cpu = std::get<0>(min_res).contiguous();   // [3]
        torch::Tensor max_cpu = std::get<0>(max_res).contiguous();   // [3]

        torch::Tensor center_cpu;
        float radius = fixed_vox_size_;
        if (robust_scene_bounds_) {
            center_cpu =
                std::get<0>(xyz_cpu.median(/*dim=*/0, /*keepdim=*/false)).contiguous();
            torch::Tensor distances = std::get<0>(
                (xyz_cpu - center_cpu.view({1, 3}))
                    .abs()
                    .max(/*dim=*/1, /*keepdim=*/false))
                    .to(torch::kFloat32)
                    .contiguous();
            distances = std::get<0>(distances.sort(/*dim=*/0)).contiguous();
            const int64_t count = distances.size(0);
            if (count > 0) {
                const int64_t p90_index = std::clamp<int64_t>(
                    static_cast<int64_t>(std::llround(
                        0.90 * static_cast<double>(count - 1))),
                    0,
                    count - 1);
                radius = std::max(
                    radius,
                    distances.index({p90_index}).item<float>());
            }
        } else {
            center_cpu = ((min_cpu + max_cpu) * 0.5f).contiguous();
            torch::Tensor radius_cpu = ((max_cpu - min_cpu) * 0.5f).contiguous();
            radius = std::max(radius, radius_cpu.max().item<float>());
        }
        radius += fixed_vox_size_;
        const float derived_scene_extent = 2.0f * radius;
        global_scene_center_[0] = center_cpu[0].item<float>();
        global_scene_center_[1] = center_cpu[1].item<float>();
        global_scene_center_[2] = center_cpu[2].item<float>();
        // In SVRecon's bounded/object-centric mode outside_level=0. For online SLAM
        // we still need a stable bound; otherwise each newly observed region can fall
        // outside the first tight point-prior AABB and trigger repeated rebuilds.
        if (fixed_global_scene_layout_) {
            const float configured_inside_extent =
                configured_global_scene_extent_ /
                std::ldexp(1.0f, outside_level_);
            TORCH_CHECK(
                configured_inside_extent >= derived_scene_extent,
                "Model.global_scene_extent root (", configured_global_scene_extent_,
                " m) provides only ", configured_inside_extent,
                " m of inside extent at outside_level=", outside_level_,
                ", smaller than the initial map extent (", derived_scene_extent,
                " m). Increase the root extent or reduce outside_level.");
            global_scene_extent_ = configured_global_scene_extent_;
        } else {
            global_scene_extent_ = (outside_level_ == 0)
                ? std::max(global_scene_extent_, derived_scene_extent)
                : derived_scene_extent;
        }

        scene_center_ = center_cpu.to(dev).to(torch::kFloat32).contiguous();
        const float inside_extent = fixed_global_scene_layout_
            ? global_scene_extent_ / std::ldexp(1.0f, outside_level_)
            : global_scene_extent_;
        inside_extent_ = torch::tensor({inside_extent},
            torch::dtype(torch::kFloat32).device(dev)
        ).contiguous();
        scene_extent_ = torch::tensor(
            {fixed_global_scene_layout_
                 ? global_scene_extent_
                 : global_scene_extent_ * std::ldexp(1.0f, outside_level_)},
            torch::dtype(torch::kFloat32).device(dev)
        ).contiguous();
        scene_min_t_ = (scene_center_ - 0.5f * scene_extent_).contiguous();

        // Store as CUDA tensors so they match the rest of the model state
        global_pcd_min_ = min_cpu.to(dev).contiguous();              // [3]
        global_pcd_max_ = max_cpu.to(dev).contiguous();              // [3]
        has_global_pcd_bb_ = true;

    }
    // ------------------------------------------------------------------------
    // 1) Compute octlevel from vox_size (mirror points_init behavior:
    //    round/clamp).
    // ------------------------------------------------------------------------
    const int MAX_L = max_num_levels_;
    // Level as float (tensor) then rounded like points_init (nearest by default)
    auto vox_size_t = torch::full({1}, fixed_vox_size_, torch::dtype(torch::kFloat32).device(dev)); // [1]
    auto L_fp = voxSizeToLevel(scene_extent_, vox_size_t).round();                                   // [1] float
    auto L_clamped = L_fp.clamp_min(1).clamp_max(MAX_L).to(torch::kInt8).contiguous();               // [1] int8

    // cache scalar level
    octlevel_ = L_clamped.item<int8_t>();
    // cache [1,1] effective voxel size
    vox_eff_ = levelToVoxSize(scene_extent_, L_clamped.view({1,1})).view({1,1}).contiguous();
    // std::cout << "[createFromPcd] Using octlevel=" << octlevel[0][0].item<int>()
    //           << " (vox_size=" << vox_eff[0][0].item<float>() << " m) for fixed_vox_size_="
    //           << fixed_vox_size_ << " m.\n";

    // ------------------------------------------------------------------------
    // 2) Compute ijk with this level/voxel size (mirror points_init)
    // ------------------------------------------------------------------------
    // ijk = ((xyz - scene_min) / vox_size).long()
    auto vox_effN  = vox_eff_.expand({N,1});                                                         // [N,1]
    torch::Tensor ijk = ((xyz - scene_min_t_) / vox_effN).floor().to(torch::kLong);                  // [N,3]

    auto octlevelN = torch::full({N,1}, octlevel_, torch::dtype(torch::kInt8).device(dev)).contiguous(); // [N,1]
    auto L_long    = octlevelN.to(torch::kLong);

    // In libtorch C++, pass a vector to cat
    std::vector<torch::Tensor> cat_inputs{ijk, L_long};
    auto ijkl = torch::cat(cat_inputs, /*dim=*/1);     

    // Use Python torch.unique(dim=0, return_inverse=True) (simple & robust)
    auto [ijkl_unq, invmap] = uniqueRowsWithInverse(ijkl);

    torch::Tensor ijk_u, L_u; 
    auto parts = torch::split_with_sizes(ijkl_unq, {3, 1}, /*dim=*/1); 
    ijk_u = parts[0].contiguous(); L_u = parts[1].to(torch::kInt8).contiguous(); 
    L_u = L_u.to(torch::kInt8).contiguous(); // [Nu,1]
    int64_t Nu = ijk_u.size(0);

    auto rgb_u = torch::zeros({Nu,3}, torch::dtype(torch::kFloat32).device(dev));
    rgb_u.index_add_(0, invmap, rgb);
    auto counts = torch::bincount(invmap, {}, Nu).view({Nu,1}).clamp_min(1);
    rgb_u = rgb_u / counts;

    // Defensive bound check: ijk in [0, 2^L)
    const int L0 = L_u[0].item<int8_t>();               // all rows have same level
    const long limit = (1L << L0);
    TORCH_CHECK((ijk_u.index({torch::indexing::Slice(),0}) >= 0).all().item<bool>() &&
                (ijk_u.index({torch::indexing::Slice(),1}) >= 0).all().item<bool>() &&
                (ijk_u.index({torch::indexing::Slice(),2}) >= 0).all().item<bool>(),
                "Points below scene_min — enlarge global_scene_extent_ or filter.");
    TORCH_CHECK((ijk_u.index({torch::indexing::Slice(),0}) < limit).all().item<bool>() &&
                (ijk_u.index({torch::indexing::Slice(),1}) < limit).all().item<bool>() &&
                (ijk_u.index({torch::indexing::Slice(),2}) < limit).all().item<bool>(),
                "Points exceed scene bounds — enlarge global_scene_extent_ or filter.");

    // ------------------------------------------------------------------------
    // 3) utils: ijk -> octpath (no constructor calls)
    // ------------------------------------------------------------------------
    auto octpath = SVRECON_UTILS::ijk_2_octpath(ijk_u.contiguous(), L_u.contiguous()).contiguous(); // [Nu,1] int64
    // ------------------------------------------------------------------------
    // 3.5) Optional camera-based filtering for initialization candidates
    //      (same logic family as increasePcd insertion filter)
    // ------------------------------------------------------------------------
    if (!cams.empty() && octpath.size(0) > 0) {
        auto [vox_center, vox_size] = decodeOctpath(
            octpath.contiguous(),
            L_u.contiguous(),
            scene_center_.contiguous(),
            scene_extent_.contiguous());
        if (vox_size.dim() == 1) {
            vox_size = vox_size.view({-1, 1});
        }

        const int64_t Nu_before = octpath.size(0);
        at::Tensor rate = markSvreconMaxSampRateDirect(cams, octpath, vox_center, vox_size);
        if (rate.dim() == 2 && rate.size(1) == 1) {
            rate = rate.squeeze(1);
        }
        rate = rate.to(torch::kFloat32);
        at::Tensor kept = (rate > 0.0f);

        if (filter_near_voxels_) {
            const float near_thresh = 0.2f;
            at::Tensor is_near = markSvreconNearDirect(cams, octpath, vox_center, vox_size, near_thresh);
            if (is_near.dim() == 2 && is_near.size(1) == 1) {
                is_near = is_near.squeeze(1);
            }
            is_near = is_near.to(torch::kBool);
            kept = kept & (~is_near);
        }
        kept = kept.view({-1}).to(torch::kBool);
        auto idx = torch::nonzero(kept).view({-1});
        const int64_t K = idx.size(0);

        TORCH_CHECK(
            K > 0,
            "createFromPcd: no voxel-layout candidate is visible from the construction cameras");
        if (K < octpath.size(0)) {
            octpath = octpath.index_select(0, idx).contiguous();
            L_u     = L_u.index_select(0, idx).contiguous();
            ijk_u   = ijk_u.index_select(0, idx).contiguous();
            rgb_u   = rgb_u.index_select(0, idx).contiguous();
            Nu = octpath.size(0);
        }
    }
    // ------------------------------------------------------------------------
    // 4) Initialize learnables directly
    // ------------------------------------------------------------------------
    // Subdivision priority
    auto subdiv_p = torch::ones({Nu,1}, torch::dtype(torch::kFloat32).device(dev)).requires_grad_();

    // SH-0 from fused RGB
    auto sh0_dc = rgbToShZero(rgb_u.contiguous()).contiguous().requires_grad_(); // [Nu,3]

    // Higher-degree SH zeros
    const int n_sh_rest = (max_sh_degree_ + 1)*(max_sh_degree_ + 1) - 1;
    auto shs = torch::zeros({Nu, n_sh_rest, 3}, torch::dtype(torch::kFloat32).device(dev)).requires_grad_();

    // Grid link => allocate per-grid-point density
    auto [grid_pts_key, vox_key] = buildGridPtsLink(octpath, L_u.contiguous(), max_num_levels_);
    (void)vox_key;
    auto geo_grid = makeGeoGridInitRows_(
        grid_pts_key,
        /*begin=*/0,
        /*end=*/grid_pts_key.size(0),
        /*default_value=*/std::max(1.0e-6f, 2.0f * fixed_vox_size_));

    // ------------------------------------------------------------------------
    // 5) Initialize C++ members directly.
    // ------------------------------------------------------------------------
    this->oct_path_      = octpath.contiguous();
    this->oct_level_     = L_u.contiguous();
    {
        auto [center, size] = decodeOctpath(
            this->oct_path_.contiguous(),
            this->oct_level_.contiguous(),
            this->scene_center_.contiguous(),
            this->scene_extent_.contiguous());
        auto [grid_pts_key_now, vox_key_now] =
            buildGridPtsLink(this->oct_path_.contiguous(), this->oct_level_.contiguous(), max_num_levels_);
        this->center_        = center.contiguous();
        this->size_          = size.squeeze(1).contiguous();
        this->grid_pts_key_  = grid_pts_key_now.contiguous();
        this->vox_key_       = vox_key_now.contiguous();
    }
    this->vox_size_inv_  = 1.0f / size_;

    // learnables
    this->_geo_grid_pts_ = geo_grid.contiguous().detach().requires_grad_(true);
    ensureSvrasterSdfField();
    this->sh0_           = sh0_dc.contiguous().detach().requires_grad_(true);
    this->shs_           = shs.contiguous().detach().requires_grad_(true);
    this->subdiv_p_      = subdiv_p.contiguous().detach().requires_grad_(true);
    this->subdiv_meta_   = torch::zeros_like(this->subdiv_p_);
    this->is_leaf_       = torch::ones(
        {center_.size(0), 1},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    // stats buffer
    this->max_w_ = torch::zeros({center_.size(0), 1}, torch::dtype(torch::kFloat32).device(dev));
    
    this->is_orb_voxel_ = torch::ones(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    this->is_inactive_geo_voxel_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    this->is_rgbd_fill_render_holes_voxel_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    this->exist_since_iter_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kInt32).device(dev));
    this->exist_since_kf_ = torch::full(
        {center_.size(0)},
        current_kf_count,
        torch::TensorOptions().dtype(torch::kInt32).device(dev));

    // Keep the complete raw real-point history for diagnostics and source masks.
    real_pcd_points_accum_cpu_ = xyz.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();

    // Match SVRecon initialization while adapting sharpness to the actual
    // online octree resolution rather than a fixed offline scene scale.
    refreshSvreconLogSTargetFromVoxelSize(/*initialize_current=*/true);

    // Register with your optimizer
    VOXEL_MODEL_TENSORS_TO_VEC
}

void VoxelModel::increasePcd(
    std::vector<float> pcd_full,
    std::vector<float> colors,
    const int iteration,
    const std::vector<sv::MiniCam>& cams)
{
    sdf_init_cams_ = cams;
    torch::Tensor explicit_sdf_support_points = next_sdf_init_support_points_;
    next_sdf_init_support_points_ = torch::Tensor();
    struct PendingGridInitializationReset {
        torch::Tensor& keys;
        torch::Tensor& values;
        ~PendingGridInitializationReset()
        {
            keys = torch::Tensor();
            values = torch::Tensor();
        }
    } pending_grid_initialization_reset{
        next_sdf_init_grid_keys_, next_sdf_init_grid_values_};
    const int Nf = static_cast<int>(pcd_full.size());
    last_increase_pcd_stats_ = IncreasePcdStats{};
    if (Nf < 3 || colors.size() < 3) return;
    int N = Nf / 3;
    const int64_t raw_points_in = N;
    last_increase_pcd_stats_.raw_points_in = raw_points_in;
    const int32_t current_kf_count =
        cams.empty() ? static_cast<int32_t>(-1) : static_cast<int32_t>(cams.size());
    TORCH_CHECK(global_scene_extent_ > 0.f && fixed_vox_size_ > 0.f,
                "increasePcd: scene extent / fixed vox size not set.");
    TORCH_CHECK(oct_path_.defined() && oct_level_.defined() && center_.defined(),
                "increasePcd: voxel model not initialized; call createFromPcd first.");

    // ——— 0) Build CPU tensors from raw arrays, normalize RGB ————————
    torch::Tensor xyz_cpu = torch::from_blob(
        pcd_full.data(), { (int64_t)N, 3 },
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
    ).clone();

    torch::Tensor rgb_cpu = torch::from_blob(
        colors.data(), { (int64_t)N, 3 },
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
    ).clone();
    // Accept both [0,1] and [0,255] color inputs.
    if (rgb_cpu.numel() > 0) {
        const float cmax = rgb_cpu.max().item<float>();
        if (cmax > 1.5f) rgb_cpu.div_(255.0f);
    }
    rgb_cpu.clamp_(0.0f, 1.0f);

    N = static_cast<int>(xyz_cpu.size(0));
    const int64_t points_after_far_filter = N;
    last_increase_pcd_stats_.points_after_far_filter = points_after_far_filter;

    // Move to CUDA
    auto dev = torch::kCUDA;
    torch::Tensor xyz = xyz_cpu.to(dev);
    torch::Tensor rgb = rgb_cpu.to(dev);
    if (fixed_global_scene_layout_) {
        const torch::Tensor scene_max = scene_min_t_ + scene_extent_.view({1});
        torch::Tensor in_bounds =
            torch::isfinite(xyz).all(/*dim=*/1) &
            (xyz >= scene_min_t_.view({1, 3})).all(/*dim=*/1) &
            (xyz < scene_max.view({1, 3})).all(/*dim=*/1);
        in_bounds = in_bounds.to(torch::kBool).contiguous();
        if (!in_bounds.all().item<bool>()) {
            auto keep = torch::nonzero(in_bounds).view({-1});
            if (keep.numel() == 0) {
                last_increase_pcd_stats_.points_after_far_filter = 0;
                sdf_init_local_support_points_ = torch::Tensor();
                pending_sdf_init_mode_ = SdfInitMode::SignedPointPrior;
                pending_real_insert_rr_entity_path_.clear();
                return;
            }
            xyz = xyz.index_select(0, keep).contiguous();
            rgb = rgb.index_select(0, keep).contiguous();
            xyz_cpu = xyz.detach().to(torch::kCPU).contiguous();
            rgb_cpu = rgb.detach().to(torch::kCPU).contiguous();
            N = static_cast<int>(xyz.size(0));
            last_increase_pcd_stats_.points_after_far_filter = N;
        }
    }
    const bool add_as_orb =
        pending_real_insert_rr_entity_path_ == "world/orb/voxels_created";
    const bool add_as_inactive_geo =
        pending_real_insert_rr_entity_path_ == "world/voxels_inactive_geo_densify/created";
    const bool add_as_rgbd_fill_render_holes =
        pending_real_insert_rr_entity_path_ == "world/rgbd_fill_render_holes/created" ||
        pending_real_insert_rr_entity_path_ == "world/rgbd_surface_points/created" ||
        pending_real_insert_rr_entity_path_ == "world/rgbd_tsdf_evidence/promoted";
    const bool add_as_topology_support =
        add_as_inactive_geo ||
        add_as_rgbd_fill_render_holes;

    // Build the min/max AABB tensor once:
    at::Tensor aabb = torch::stack({
        torch::tensor({ global_scene_center_[0] - 0.5f*global_scene_extent_,
                        global_scene_center_[1] - 0.5f*global_scene_extent_,
                        global_scene_center_[2] - 0.5f*global_scene_extent_ }),
        torch::tensor({ global_scene_center_[0] + 0.5f*global_scene_extent_,
                        global_scene_center_[1] + 0.5f*global_scene_extent_,
                        global_scene_center_[2] + 0.5f*global_scene_extent_ })
    });

    // --- 2) Compute octlevel from fixed_vox_size_ (same as createFromPcd)
    const int MAX_L = max_num_levels_;

    auto vox_effN  = vox_eff_.expand({N,1});                                                         // [N,1]
    torch::Tensor ijk = ((xyz - scene_min_t_) / vox_effN).floor().to(torch::kLong);                  // [N,3]

    auto octlevelN = torch::full({N,1}, octlevel_, torch::dtype(torch::kInt8).device(dev)).contiguous(); // [N,1]
    auto L_long    = octlevelN.to(torch::kLong);

    // In libtorch C++, pass a vector to cat
    std::vector<torch::Tensor> cat_inputs{ijk, L_long};
    auto ijkl = torch::cat(cat_inputs, /*dim=*/1);                                                         // [N,4]

    auto [ijkl_unq, invmap] = uniqueRowsWithInverse(ijkl);
    torch::Tensor ijk_u, L_u;
    auto parts = torch::split_with_sizes(ijkl_unq, {3,1}, 1);
    ijk_u = parts[0].contiguous();                                                                    // [Nu,3]
    L_u   = parts[1].to(torch::kInt8).contiguous();                                                   // [Nu,1]
    int64_t Nu = ijk_u.size(0);
    const int64_t unique_voxel_candidates_before_insert_filter = Nu;

    // Defensive bound check: ijk in [0, 2^L)
    const int8_t L0 = L_u[0].item<int8_t>();      // all rows share the same level
    const long limit = (1L << L0);
    const bool in_low =
    (ijk_u.index({torch::indexing::Slice(),0}) >= 0).all().item<bool>() &&
    (ijk_u.index({torch::indexing::Slice(),1}) >= 0).all().item<bool>() &&
    (ijk_u.index({torch::indexing::Slice(),2}) >= 0).all().item<bool>();
    const bool in_high =
        (ijk_u.index({torch::indexing::Slice(),0}) < limit).all().item<bool>() &&
        (ijk_u.index({torch::indexing::Slice(),1}) < limit).all().item<bool>() &&
        (ijk_u.index({torch::indexing::Slice(),2}) < limit).all().item<bool>();
    if (!(in_low && in_high)) {
        std::cout << "[increasePcd] OOB detected — reinitializing via createFromPcd().\n";
        // A) Compute a conservative scene from this batch.
        {
            auto pts = xyz_cpu.to(torch::kCPU).to(torch::kFloat32).contiguous();
            auto center = std::get<0>(pts.median(/*dim=*/0, /*keepdim=*/false)).contiguous();
            auto dist = std::get<0>((pts - center.view({1, 3})).abs().max(/*dim=*/1, /*keepdim=*/false))
                .to(torch::kFloat32).contiguous();
            auto sorted = std::get<0>(dist.sort(/*dim=*/0, /*descending=*/false)).contiguous();
            const int64_t n = sorted.size(0);
            int64_t p90_idx = static_cast<int64_t>(std::llround(0.90 * static_cast<double>(std::max<int64_t>(0, n - 1))));
            p90_idx = std::max<int64_t>(0, std::min<int64_t>(p90_idx, std::max<int64_t>(0, n - 1)));
            float radius = (n > 0) ? sorted.index({p90_idx}).item<float>() : 1.0f;
            if (!std::isfinite(radius) || radius <= 0.0f) {
                radius = (n > 0) ? sorted.index({n - 1}).item<float>() : 1.0f;
            }
            if (!std::isfinite(radius) || radius <= 0.0f) {
                radius = 1.0f;
            }

            global_scene_center_[0] = center.index({0}).item<float>();
            global_scene_center_[1] = center.index({1}).item<float>();
            global_scene_center_[2] = center.index({2}).item<float>();
            global_scene_extent_ = 2.0f * radius;
        }
        // B) Convert current batch to a temporary map and call createFromPcd
        {
            std::map<point3D_id_t, Point3D> tmp;
            static point3D_id_t id_seed = 1;  // local seq; independent of COLMAP ids, etc.
            auto xyz_re = xyz_cpu.contiguous();
            auto rgb_re = rgb_cpu.contiguous();
            auto xyz_acc = xyz_re.accessor<float, 2>();
            auto rgb_acc = rgb_re.accessor<float, 2>();
            for (int i = 0; i < N; ++i) {
                Point3D P;
                // world coords
                P.xyz_(0) = xyz_acc[i][0];
                P.xyz_(1) = xyz_acc[i][1];
                P.xyz_(2) = xyz_acc[i][2];
                // rgb_cpu is normalized to [0,1]. Point3D stores uint8.
                P.color_(0) = static_cast<uint8_t>(std::round(std::clamp(rgb_acc[i][0], 0.0f, 1.0f) * 255.0f));
                P.color_(1) = static_cast<uint8_t>(std::round(std::clamp(rgb_acc[i][1], 0.0f, 1.0f) * 255.0f));
                P.color_(2) = static_cast<uint8_t>(std::round(std::clamp(rgb_acc[i][2], 0.0f, 1.0f) * 255.0f));
                tmp.emplace(id_seed++, P);
            }
            createFromPcd(tmp, cams);
        }
        // Log and exit this call
        sdf_init_local_support_points_ = torch::Tensor();
        pending_sdf_init_mode_ = SdfInitMode::SignedPointPrior;
        return;
    }

    auto rgb_u = torch::zeros({Nu,3}, torch::dtype(torch::kFloat32).device(dev));
    rgb_u.index_add_(0, invmap, rgb);
    auto counts = torch::bincount(invmap, {}, Nu).view({Nu,1}).clamp_min(1);
    rgb_u = rgb_u / counts;

    // ── 4) Build octpath for this batch ─────────────────────────────────────
    auto octpath_new = SVRECON_UTILS::ijk_2_octpath(ijk_u.contiguous(), L_u.contiguous()).contiguous();          // [Nu,1] int64

    const bool has_topology_support =
        (is_inactive_geo_voxel_.defined() &&
         is_inactive_geo_voxel_.numel() > 0 &&
         is_inactive_geo_voxel_.to(torch::kBool).any().item<bool>()) ||
        (is_rgbd_fill_render_holes_voxel_.defined() &&
         is_rgbd_fill_render_holes_voxel_.numel() > 0 &&
         is_rgbd_fill_render_holes_voxel_.to(torch::kBool).any().item<bool>());
    if (add_as_orb && has_topology_support) {
        updateExistingSupportSdfFromPoints_(
            xyz.contiguous(),
            rgb.contiguous(),
            cams);
    }

    // ---- Active insertion-time SVR-style filtering (kept separate from OLD block) ----
    // Keep this at insertion-time so bad candidates never enter the topology.
    // Prune-time filtering in VoxelMapper remains useful as a second cleanup stage.
    if (!cams.empty() && octpath_new.size(0) > 0) {
        // Decode voxel centers/sizes for current candidates.
        auto [vox_center, vox_size] = decodeOctpath(
            octpath_new.contiguous(),
            L_u.contiguous(),
            scene_center_.contiguous(),
            scene_extent_.contiguous());
        if (vox_size.dim() == 1) {
            vox_size = vox_size.view({-1, 1});
        }

        // 1) Visibility / sampling-rate filtering
        at::Tensor rate = markSvreconMaxSampRateDirect(cams, octpath_new, vox_center, vox_size);
        if (rate.dim() == 2 && rate.size(1) == 1) {
            rate = rate.squeeze(1);
        }
        rate = rate.to(torch::kFloat32);
        at::Tensor kept = (rate > 0.0f);

        // 2) Near filtering
        // NOTE: Upstream SVRaster layout initialization uses filter_near = -1 (disabled).
        if (filter_near_voxels_) {
            const float near_thresh = 0.2f;
            at::Tensor is_near =
                markSvreconNearDirect(cams, octpath_new, vox_center, vox_size, near_thresh);
            if (is_near.dim() == 2 && is_near.size(1) == 1) {
                is_near = is_near.squeeze(1);
            }
            is_near = is_near.to(torch::kBool);
            kept = kept & (~is_near);
        }

        kept = kept.view({-1}).to(torch::kBool);
        auto idx = torch::nonzero(kept).view({-1});
        int64_t K = idx.size(0);

        if (K < octpath_new.size(0)) {
            // Apply mask to ALL aligned tensors
            octpath_new = octpath_new.index_select(0, idx).contiguous(); // [K,1]
            L_u         = L_u.index_select(0, idx).contiguous();         // [K,1]
            ijk_u       = ijk_u.index_select(0, idx).contiguous();       // [K,3]
            rgb_u       = rgb_u.index_select(0, idx).contiguous();       // [K,3]
        }

        Nu = octpath_new.size(0); // update Nu after filtering

        TORCH_CHECK(L_u.sizes() == torch::IntArrayRef({Nu, 1}),
                    "L_u shape mismatch after insertion-time filtering");
        TORCH_CHECK(ijk_u.sizes() == torch::IntArrayRef({Nu, 3}),
                    "ijk_u shape mismatch after insertion-time filtering");
        TORCH_CHECK(rgb_u.sizes() == torch::IntArrayRef({Nu, 3}),
                    "rgb_u shape mismatch after insertion-time filtering");
	    }
	    // ---- end active insertion-time filtering ----
	    const int64_t unique_voxel_candidates_after_insert_filter = Nu;
    last_increase_pcd_stats_.unique_voxel_candidates_before_insert_filter =
        unique_voxel_candidates_before_insert_filter;
    last_increase_pcd_stats_.unique_voxel_candidates_after_insert_filter =
        unique_voxel_candidates_after_insert_filter;

    // ── 5) Dedup against existing voxels (across-batch) ─────────────────────
    auto octpath_old  = this->oct_path_.contiguous();                    // [No,1] int64
    auto octlevel_old = this->oct_level_.contiguous();                   // [No,1] int8
    const int64_t old_voxel_count = octpath_old.size(0);

    // Packed 1D key: (octpath<<8) | level
    auto key_new = octpath_new.view({-1}).to(torch::kInt64)
                    .mul(256)
                    .add(L_u.view({-1}).to(torch::kInt64));
    auto key_old_all = octpath_old.view({-1}).to(torch::kInt64)
                    .mul(256)
                    .add(octlevel_old.view({-1}).to(torch::kInt64));
    auto bool_opts_old = torch::TensorOptions().dtype(torch::kBool).device(dev);
    auto i32_opts_old = torch::TensorOptions().dtype(torch::kInt32).device(dev);
    if (!is_orb_voxel_.defined() || is_orb_voxel_.size(0) != octpath_old.size(0)) {
        is_orb_voxel_ = torch::zeros({octpath_old.size(0)}, bool_opts_old);
    } else if (is_orb_voxel_.device() != dev) {
        is_orb_voxel_ = is_orb_voxel_.to(dev);
    }
    torch::Tensor new_mask;
    if (octpath_old.numel() == 0) {
        new_mask = torch::ones({Nu}, torch::dtype(torch::kBool).device(dev));
    } else {
        auto is_dup = at::isin(key_new, key_old_all).to(torch::kBool);                                    // [Nu]
        new_mask = ~is_dup;

    }
    // --- Prevent re-adding a parent at L=base_L when children (L > base_L) already exist ---
    {
        if (octpath_old.numel() > 0) {
            auto dev = octpath_old.device();
            const int MAX_L  = max_num_levels_;            // must match svraster MAX_NUM_LEVELS
            const int base_L = static_cast<int>(octlevel_);// the level used by createFromPcd()

            TORCH_CHECK(base_L >= 1 && base_L <= MAX_L,
                        "[increasePcd] base_L (octlevel_) out of range: ", base_L,
                        " with MAX_L=", MAX_L);

            // Existing voxels whose level is strictly finer than base_L
            auto Lold_i64    = octlevel_old.view({-1}).to(torch::kInt64);     // [No]
            auto has_children= (Lold_i64 > base_L);                           // [No] bool

            if (has_children.any().item<bool>()) {

                // Mask that clears all octant bits *below* base_L.
                // Bits per level = 3; for a node at level L, its octant sits at shift = 3*(MAX_L - L).
                // To keep bits down to base_L (inclusive), clear the lowest 3*(MAX_L - base_L) bits.
                const int levels_below  = std::max(0, MAX_L - base_L);
                const int bits_to_clear = 3 * levels_below;
                long long lower_mask    = (bits_to_clear > 0) ? ((1LL << bits_to_clear) - 1LL) : 0LL;
                long long keep_mask_ll  = ~lower_mask;

                auto keep_mask = torch::full(
                    {1},
                    static_cast<int64_t>(keep_mask_ll),
                    torch::TensorOptions().dtype(torch::kInt64).device(dev)
                );

                // Compute the ancestor-at-base_L octpaths for those finer voxels
                auto op_old_i64  = octpath_old.view({-1}).to(torch::kInt64);   // [No]
                auto op_anc_base = (op_old_i64 & keep_mask);                   // [No]

                // Keep only rows where L_old > base_L
                auto sel = torch::nonzero(has_children).view({-1});            // [K]
                op_anc_base = op_anc_base.index_select(0, sel);                // [K]

                // Build the ancestor keys at base_L: ((octpath_anc_base<<8) | base_L)
                auto key_children_as_parent = op_anc_base.mul(256)
                                            .add(torch::full_like(op_anc_base,
                                                                static_cast<int64_t>(base_L)));

                // Unique + sorted to make isin faster
                auto unique_sorted_1d = [](const at::Tensor& t)->at::Tensor {
                    TORCH_CHECK(t.dim() == 1, "unique_sorted_1d expects a 1-D tensor");
                    if (t.numel() <= 1) return t.contiguous();
                    auto sort_res = t.sort(/*dim=*/0);
                    auto sorted   = std::get<0>(sort_res);
                    using torch::indexing::Slice;
                    auto keep = torch::empty_like(sorted, torch::kBool);
                    keep.index_put_({0}, true);
                    auto neq = sorted.index({Slice(1, torch::indexing::None)})
                            != sorted.index({Slice(torch::indexing::None, -1)});
                    keep.index_put_({Slice(1, torch::indexing::None)}, neq);
                    auto idx = torch::nonzero(keep).view({-1});
                    return sorted.index_select(0, idx).contiguous();
                };
                key_children_as_parent = unique_sorted_1d(key_children_as_parent);

                // If a candidate NEW (octpath, base_L) matches any ancestor of an existing finer voxel,
                // then inserting that parent would collide with existing children later.
                auto would_collide_parent =
                    at::isin(key_new, key_children_as_parent).to(torch::kBool);  // [Nu]

                if (would_collide_parent.any().item<bool>()) {
                    new_mask = new_mask & (~would_collide_parent);
                }
            }
        }
    }

    auto sel = torch::nonzero(new_mask).view({-1});
    const int64_t pending_support_updates = 0;
    const int64_t new_voxel_candidates = sel.size(0);
    const int64_t duplicate_existing_voxels =
        unique_voxel_candidates_after_insert_filter - new_voxel_candidates;
    last_increase_pcd_stats_.duplicate_existing_voxels = duplicate_existing_voxels;
    last_increase_pcd_stats_.new_voxels = new_voxel_candidates;
    last_increase_pcd_stats_.pending_promotions = 0;
    last_increase_pcd_stats_.pending_support_updates = pending_support_updates;
    if (add_as_orb) {
        // Keep one dedicated accumulated prior set. Inactive geometry and
        // RGB-D hole samples allocate cells but never enter this ORB set.
        appendSparseSupportPoints_(xyz);
    }
    pending_sdf_init_mode_ = topology_sdf_init_mode_;
    if (xyz_cpu.defined() && xyz_cpu.numel() > 0) {
        const bool uses_local_point_prior =
            pending_sdf_init_mode_ == SdfInitMode::SignedPointPrior ||
            pending_sdf_init_mode_ == SdfInitMode::WeakSurfacePrior;
        if (uses_local_point_prior) {
            sdf_init_local_support_points_ =
                explicit_sdf_support_points.defined() &&
                        explicit_sdf_support_points.numel() > 0
                    ? explicit_sdf_support_points.to(dev).to(torch::kFloat32)
                          .reshape({-1, 3}).contiguous()
                    : xyz.detach().contiguous();
        } else {
            sdf_init_local_support_points_ = torch::Tensor();
        }
        if (pending_sdf_init_mode_ == SdfInitMode::SignedPointPrior &&
            !add_as_orb) {
            appendSparseSupportPoints_(xyz);
        }
        auto new_pts_cpu = xyz_cpu.to(torch::kCPU).to(torch::kFloat32).contiguous();
        if (!real_pcd_points_accum_cpu_.defined() || real_pcd_points_accum_cpu_.numel() == 0) {
            real_pcd_points_accum_cpu_ = new_pts_cpu;
        } else {
            real_pcd_points_accum_cpu_ =
                torch::cat({real_pcd_points_accum_cpu_, new_pts_cpu}, 0).contiguous();
        }
    }
    if (sel.numel() == 0) {
        sdf_init_local_support_points_ = torch::Tensor();
        pending_sdf_init_mode_ = SdfInitMode::SignedPointPrior;
        pending_real_insert_rr_entity_path_.clear();
        return;
    }
    auto octpath_add = octpath_new.index_select(0, sel);                                                // [Nk,1]
    auto L_add       = L_u.index_select(0, sel);                                                         // [Nk,1]
    auto rgb_add     = rgb_u.index_select(0, sel);                                                       // [Nk,3]
    const int Nk = sel.size(0);

    ensureSvrasterSdfField();
    torch::Tensor old_vox_sdf_values =
        voxelCornerScalarFromGrid_(svraster_sdf_grid_pts_);
    torch::Tensor old_vox_sdf_weights =
        voxelCornerScalarFromGrid_(svraster_sdf_weights_);

    // ── 6) Append topology (old preserved) ──────────────────────────────────
    this->oct_path_ = torch::cat({octpath_old,  octpath_add}, 0).contiguous();
    this->oct_level_ = torch::cat({octlevel_old, L_add}, 0).contiguous();
    {
        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto i32_opts  = torch::TensorOptions().dtype(torch::kInt32).device(dev);
        if (!is_orb_voxel_.defined() || is_orb_voxel_.size(0) != octpath_old.size(0)) {
            is_orb_voxel_ = torch::zeros({octpath_old.size(0)}, bool_opts);
        } else if (is_orb_voxel_.device() != octpath_old.device()) {
            is_orb_voxel_ = is_orb_voxel_.to(octpath_old.device());
        }
        if (!is_inactive_geo_voxel_.defined() || is_inactive_geo_voxel_.size(0) != octpath_old.size(0)) {
            is_inactive_geo_voxel_ = torch::zeros({octpath_old.size(0)}, bool_opts);
        } else if (is_inactive_geo_voxel_.device() != octpath_old.device()) {
            is_inactive_geo_voxel_ = is_inactive_geo_voxel_.to(octpath_old.device());
        }
        if (!is_rgbd_fill_render_holes_voxel_.defined() ||
            is_rgbd_fill_render_holes_voxel_.size(0) != octpath_old.size(0)) {
            is_rgbd_fill_render_holes_voxel_ = torch::zeros({octpath_old.size(0)}, bool_opts);
        } else if (is_rgbd_fill_render_holes_voxel_.device() != octpath_old.device()) {
            is_rgbd_fill_render_holes_voxel_ =
                is_rgbd_fill_render_holes_voxel_.to(octpath_old.device());
        }
        if (!exist_since_iter_.defined() || exist_since_iter_.size(0) != octpath_old.size(0)) {
            exist_since_iter_ = torch::zeros({octpath_old.size(0)}, i32_opts);
        } else if (exist_since_iter_.device() != octpath_old.device()) {
            exist_since_iter_ = exist_since_iter_.to(octpath_old.device());
        }
        if (!exist_since_kf_.defined() || exist_since_kf_.size(0) != octpath_old.size(0)) {
            exist_since_kf_ = torch::full({octpath_old.size(0)}, static_cast<int32_t>(-1), i32_opts);
        } else if (exist_since_kf_.device() != octpath_old.device()) {
            exist_since_kf_ = exist_since_kf_.to(octpath_old.device());
        }
        if (Nk > 0) {
            auto orb_add_flag = torch::full({Nk}, add_as_orb, bool_opts);
            auto inactive_geo_add_flag = torch::full({Nk}, add_as_inactive_geo, bool_opts);
            auto rgbd_fill_render_holes_add_flag =
                torch::full({Nk}, add_as_rgbd_fill_render_holes, bool_opts);
            auto exist_since_add = torch::full(
                {Nk}, static_cast<int32_t>(iteration), i32_opts);
            auto exist_since_kf_add = torch::full(
                {Nk}, current_kf_count, i32_opts);
            is_orb_voxel_ = torch::cat({is_orb_voxel_, orb_add_flag}, 0).contiguous();
            is_inactive_geo_voxel_ = torch::cat({is_inactive_geo_voxel_, inactive_geo_add_flag}, 0).contiguous();
            is_rgbd_fill_render_holes_voxel_ =
                torch::cat({is_rgbd_fill_render_holes_voxel_, rgbd_fill_render_holes_add_flag}, 0).contiguous();
            exist_since_iter_ = torch::cat({exist_since_iter_, exist_since_add}, 0).contiguous();
            exist_since_kf_ = torch::cat({exist_since_kf_, exist_since_kf_add}, 0).contiguous();
        }
    }

    // ── 7) Append learnables for new rows ───────────────────────────────────
    // _subdiv_p
    auto subdiv_add = torch::ones({Nk,1}, torch::dtype(torch::kFloat32).device(dev));
    this->subdiv_p_ = torch::cat({this->subdiv_p_.detach(), subdiv_add}, 0)
                          .contiguous()
                          .detach()
                          .requires_grad_();
    if (!this->subdiv_meta_.defined() || this->subdiv_meta_.numel() == 0) {
        this->subdiv_meta_ = torch::zeros(
            {old_voxel_count, 1},
            torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    }
    this->subdiv_meta_ = torch::cat(
        {this->subdiv_meta_.to(dev).to(torch::kFloat32).contiguous(),
         torch::zeros({Nk, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(dev))},
        0).contiguous();

    // _sh0 from fused rgb
    torch::Tensor sh0_add = torch::empty({0, 3}, torch::dtype(torch::kFloat32).device(dev));
    if (Nk > 0) {
        sh0_add = rgbToShZero(rgb_add.contiguous()).contiguous(); // [Nk,3]
    }
    // _shs zeros
    const int n_sh_rest = (max_sh_degree_ + 1)*(max_sh_degree_ + 1) - 1;
    auto shs_add = torch::zeros({Nk, n_sh_rest, 3}, torch::dtype(torch::kFloat32).device(dev));

    // ── 8) Rebuild grid links; grow _geo_grid_pts only if grid expanded ─────
    auto [grid_pts_key_new, vox_key_new] =
        buildGridPtsLink(
            this->oct_path_.contiguous(),
            this->oct_level_.contiguous(),
            max_num_levels_);
    (void)vox_key_new;
    // ── 9) Append rows to optimizer param groups ────────────────────────────
    // Important: only call for non-empty additions.
    rebuildGeoGridForNewGridKeys_(grid_pts_key_new, /*default_value=*/-10.0f);
    sdf_init_local_support_points_ = torch::Tensor();
    pending_sdf_init_mode_ = SdfInitMode::SignedPointPrior;
    this->grid_pts_key_ = grid_pts_key_new.contiguous();
    this->vox_key_ = vox_key_new.contiguous();
    if (Nk > 0) {
        appendGroup_(/*group_idx=*/1, /*add_rows=*/sh0_add, &this->sh0_);
        appendGroup_(/*group_idx=*/2, /*add_rows=*/shs_add, &this->shs_);
    }

    // ── 10) Rebuild renderer fields from the C++ topology ───────────────────
    this->oct_path_      = this->oct_path_.contiguous();
    this->oct_level_     = this->oct_level_.contiguous();
    {
        auto [center, size] = decodeOctpath(
            this->oct_path_.contiguous(),
            this->oct_level_.contiguous(),
            this->scene_center_.contiguous(),
            this->scene_extent_.contiguous());
        this->center_        = center.contiguous();
        this->size_          = size.squeeze(1).contiguous();
        // Topology has not changed since the link built above. Reusing it avoids
        // a second full N x 8 corner expansion and CUDA unique_dim allocation.
        this->grid_pts_key_  = grid_pts_key_new.contiguous();
        this->vox_key_       = vox_key_new.contiguous();
    }
    this->vox_size_inv_  = 1.0f / size_;
    {
        const int64_t N_final = this->vox_key_.size(0);
        const int64_t N_added = std::max<int64_t>(0, N_final - old_voxel_count);
        auto dev_sdf = old_vox_sdf_values.defined()
            ? old_vox_sdf_values.device()
            : (_geo_grid_pts_.defined() ? _geo_grid_pts_.device() : torch::Device(device_type_));
        auto value_opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev_sdf);
        if (!old_vox_sdf_values.defined() || old_vox_sdf_values.size(0) != old_voxel_count) {
            old_vox_sdf_values = torch::zeros({old_voxel_count, 8, 1}, value_opts);
            old_vox_sdf_weights = torch::zeros({old_voxel_count, 8, 1}, value_opts);
        }
        auto added_sdf = torch::zeros({N_added, 8, 1}, value_opts);
        auto added_weights = torch::zeros({N_added, 8, 1}, value_opts);
        auto vox_sdf = torch::cat({old_vox_sdf_values, added_sdf}, 0).contiguous();
        auto vox_weights = torch::cat({old_vox_sdf_weights, added_weights}, 0).contiguous();
        rebuildSvrasterSdfFieldFromVoxelCorners_(vox_sdf, vox_weights);
    }
    // (optimizer params already set by appendGroup_; just ensure requires_grad)
    this->subdiv_p_ = this->subdiv_p_.contiguous();
    if (!this->subdiv_p_.requires_grad()) {
        this->subdiv_p_.set_requires_grad(true);
    }
    if (!this->subdiv_meta_.defined() || this->subdiv_meta_.size(0) != center_.size(0)) {
        this->subdiv_meta_ = torch::zeros(
            {center_.size(0), 1},
            torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    }
    if (!this->is_leaf_.defined() || this->is_leaf_.size(0) != center_.size(0)) {
        auto leaf_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto aligned = torch::ones({center_.size(0), 1}, leaf_opts);
        if (this->is_leaf_.defined() && this->is_leaf_.numel() > 0) {
            auto old = this->is_leaf_.to(dev).to(torch::kBool).reshape({-1, 1});
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        this->is_leaf_ = aligned.contiguous();
    }
    // stats buffer resize
    this->max_w_ = torch::zeros({center_.size(0), 1}, torch::dtype(torch::kFloat32).device(dev));
    // Keep provenance tensor aligned with current topology size.
    if (!is_orb_voxel_.defined() || is_orb_voxel_.size(0) != center_.size(0)) {
        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, bool_opts);
        if (is_orb_voxel_.defined() && is_orb_voxel_.numel() > 0) {
            auto old = is_orb_voxel_.to(dev).to(torch::kBool).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        is_orb_voxel_ = aligned;
    }
    if (!is_inactive_geo_voxel_.defined() || is_inactive_geo_voxel_.size(0) != center_.size(0)) {
        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, bool_opts);
        if (is_inactive_geo_voxel_.defined() && is_inactive_geo_voxel_.numel() > 0) {
            auto old = is_inactive_geo_voxel_.to(dev).to(torch::kBool).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        is_inactive_geo_voxel_ = aligned;
    }
    if (!is_rgbd_fill_render_holes_voxel_.defined() ||
        is_rgbd_fill_render_holes_voxel_.size(0) != center_.size(0)) {
        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, bool_opts);
        if (is_rgbd_fill_render_holes_voxel_.defined() &&
            is_rgbd_fill_render_holes_voxel_.numel() > 0) {
            auto old = is_rgbd_fill_render_holes_voxel_.to(dev).to(torch::kBool).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        is_rgbd_fill_render_holes_voxel_ = aligned;
    }
    // Keep exist_since_iter tensor aligned with current topology size.
    if (!exist_since_iter_.defined() || exist_since_iter_.size(0) != center_.size(0)) {
        auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, i32_opts);
        if (exist_since_iter_.defined() && exist_since_iter_.numel() > 0) {
            auto old = exist_since_iter_.to(dev).to(torch::kInt32).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        exist_since_iter_ = aligned;
    }
    if (!exist_since_kf_.defined() || exist_since_kf_.size(0) != center_.size(0)) {
        auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(dev);
        auto aligned = torch::full({center_.size(0)}, static_cast<int32_t>(-1), i32_opts);
        if (exist_since_kf_.defined() && exist_since_kf_.numel() > 0) {
            auto old = exist_since_kf_.to(dev).to(torch::kInt32).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        exist_since_kf_ = aligned;
    }
    // ── 10) Re-register with optimizer (new rows appended) ───────────────────
    VOXEL_MODEL_TENSORS_TO_VEC

    if (!pending_real_insert_rr_entity_path_.empty()) {
        pending_real_insert_rr_entity_path_.clear();
    }

}

void VoxelModel::increasePcd(
    torch::Tensor& new_point_cloud,
    torch::Tensor& new_colors,
    const int iteration,
    const std::vector<sv::MiniCam>& cams)
{
    // Follow GaussianModel::increasePcd(tensor) pattern, but reuse
    // our existing vector-based VoxelModel::increasePcd(..., cams).

    if (!new_point_cloud.defined() || !new_colors.defined())
        return;

    TORCH_CHECK(
        new_point_cloud.dim() == 2 && new_point_cloud.size(1) == 3,
        "VoxelModel::increasePcd(tensor): new_point_cloud must be [N,3]"
    );
    TORCH_CHECK(
        new_colors.dim() == 2 && new_colors.size(1) == 3,
        "VoxelModel::increasePcd(tensor): new_colors must be [N,3]"
    );
    TORCH_CHECK(
        new_point_cloud.size(0) == new_colors.size(0),
        "VoxelModel::increasePcd(tensor): points/colors size mismatch"
    );

    const int64_t N = new_point_cloud.size(0);
    if (N == 0)
        return;

    // Ensure CPU + contiguous
    auto xyz_cpu = new_point_cloud.to(torch::kCPU).contiguous();
    auto rgb_cpu = new_colors.to(torch::kCPU).contiguous();

    TORCH_CHECK(
        xyz_cpu.scalar_type() == torch::kFloat32 &&
        rgb_cpu.scalar_type() == torch::kFloat32,
        "VoxelModel::increasePcd(tensor): tensors must be float32"
    );

    // Flatten to 1D [3*N]
    auto xyz_flat = xyz_cpu.view({-1});  // [3N]
    auto rgb_flat = rgb_cpu.view({-1});  // [3N]

    std::vector<float> points(3 * N);
    std::vector<float> cols(3 * N);

    // Copy XYZ directly
    std::memcpy(
        points.data(),
        xyz_flat.data_ptr<float>(),
        points.size() * sizeof(float)
    );

    // Keep colors as-is; vector-based increasePcd now auto-detects [0,1] vs [0,255].
    const float* rgb_ptr = rgb_flat.data_ptr<float>();
    for (int64_t i = 0; i < static_cast<int64_t>(cols.size()); ++i) {
        cols[i] = rgb_ptr[i];
    }

    // Reuse the main point-cloud insertion pipeline.
    increasePcd(
        points, cols, iteration, cams);
}

void VoxelModel::createTrainer(float geo_lr, float sh0_lr, float shs_lr,
                               float beta1, float beta2, float eps,
                               const std::vector<int>& milestones,
                               float gamma,
                               float log_s_lr)
{
    optimizer_geo_lr_ = geo_lr;
    optimizer_sh0_lr_ = sh0_lr;
    optimizer_shs_lr_ = shs_lr;
    optimizer_log_s_lr_ = log_s_lr;
    optimizer_beta1_ = beta1;
    optimizer_beta2_ = beta2;
    optimizer_eps_ = eps;
    scheduler_milestones_ = milestones;
    std::sort(scheduler_milestones_.begin(), scheduler_milestones_.end());
    scheduler_gamma_ = gamma;
    scheduler_epoch_ = -1;
    optimizer_initialized_ = true;

    adam_geo_ = AdamGroupState{};
    adam_sh0_ = AdamGroupState{};
    adam_shs_ = AdamGroupState{};
    adam_log_s_ = AdamGroupState{};

    _geo_grid_pts_ = _geo_grid_pts_.contiguous().detach().requires_grad_(true);
    log_s_ = log_s_.defined() && log_s_.numel() > 0
        ? log_s_.contiguous().detach().requires_grad_(true)
        : torch::full({1}, 0.3f, _geo_grid_pts_.options()).requires_grad_(true);
    sh0_ = sh0_.contiguous().detach().requires_grad_(true);
    shs_ = shs_.contiguous().detach().requires_grad_(true);
}

std::tuple<double,double,double> VoxelModel::currentLearningRates() const {
    if (!optimizer_initialized_) {
        return {std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()};
    }
    return {optimizer_geo_lr_, optimizer_sh0_lr_, optimizer_shs_lr_};
}

void VoxelModel::appendGroup_(int group_idx,
                              const torch::Tensor& add_rows,
                              torch::Tensor* out_member_param) {
    torch::Tensor old_param = *out_member_param;
    torch::Tensor new_param =
        torch::cat({old_param, add_rows}, /*dim=*/0).contiguous().detach().requires_grad_(true);

    auto extend_state = [&](AdamGroupState& state) {
        if (!state.exp_avg.defined() || !state.exp_avg_sq.defined()) {
            return;
        }
        torch::Tensor zeros = torch::zeros_like(add_rows);
        state.exp_avg = torch::cat({state.exp_avg, zeros}, 0).contiguous();
        state.exp_avg_sq = torch::cat({state.exp_avg_sq, zeros}, 0).contiguous();
    };

    if (group_idx == 0) {
        extend_state(adam_geo_);
    } else if (group_idx == 1) {
        extend_state(adam_sh0_);
    } else if (group_idx == 2) {
        extend_state(adam_shs_);
    }

    *out_member_param = new_param;
}

VoxelModel::StatPkg
VoxelModel::computeTrainingStat(const std::vector<MiniCam>& cams) {
    // Mirrors SVAdaptive.compute_training_stat (but uses our renderer)
    freezeVoxGeo();

    const int64_t N = center_.size(0);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_type_);
    // auto max_w            = torch::zeros({N,1}, opts);
    this->max_w_.zero_();
    auto min_samp_interval = torch::full ({N,1}, 1e30f, opts);
    auto view_cnt         = torch::zeros({N,1}, opts);

    for (const auto& cam : cams) {
        // std::cout << "rendering cam " << cam.width << "x" << cam.height << "\n";
        // auto pkg = render(cam, torch::Tensor(), cam.height, cam.width, color_mode='dontcare', track_max_w=True);
        std::unordered_map<std::string, torch::Tensor> pkg;
        {
            torch::NoGradGuard no_grad;
            pkg = render(
                cam,
                cam.height,
                cam.width,
                torch::Tensor(),
                "dontcare",
                true);
        }
        
        if (!pkg.count("max_w") || !pkg.at("max_w").defined())
            continue;

        auto max_w_i = pkg["max_w"].to(device_type_);
        this->max_w_ = torch::maximum(this->max_w_, max_w_i);

        // visibility indices for current cam
        auto vis_idx = (max_w_i.squeeze(1) > 0).nonzero().squeeze(1);  // [K]

        if (vis_idx.numel() > 0) {
            // z distance along camera forward
            auto pos   = camPosition_(cam, device_type_);
            auto fwd   = camForward_(cam,   device_type_);
            auto vc    = center_.index({vis_idx});             // [K,3]
            auto zdist = ((vc - pos) * fwd).sum(1, true).abs(); // [K,1] (abs -> guard sign)

            float pix_size = camPixSize_(cam);                 // scalar
            auto samp_itv  = zdist * pix_size;                 // [K,1]

            // min over views
            auto cur = min_samp_interval.index({vis_idx});
            min_samp_interval.index_put_({vis_idx}, torch::minimum(cur, samp_itv));

            // view count
            view_cnt.index_put_({vis_idx}, view_cnt.index({vis_idx}) + 1);
        }
    }

    unfreezeVoxGeo();
    return { this->max_w_.contiguous(), min_samp_interval.contiguous(), view_cnt.contiguous() };
}

void VoxelModel::optimizerZeroGrad() {
    _geo_grid_pts_.mutable_grad() = torch::Tensor();
    log_s_.mutable_grad() = torch::Tensor();
    sh0_.mutable_grad() = torch::Tensor();
    shs_.mutable_grad() = torch::Tensor();
    subdiv_p_.mutable_grad() = torch::Tensor();
}

void VoxelModel::optimizerStep() {
    if (!optimizer_initialized_) {
        return;
    }
    torch::NoGradGuard no_grad;

    auto step_group = [&](torch::Tensor& param,
                          AdamGroupState& state,
                          const float lr) {
        torch::Tensor grad = param.grad();
        if (!grad.defined()) {
            return;
        }
        if (!grad.is_contiguous()) {
            grad = grad.contiguous();
        }
        if (!state.exp_avg.defined() ||
            !state.exp_avg_sq.defined() ||
            state.exp_avg.sizes() != param.sizes() ||
            state.exp_avg_sq.sizes() != param.sizes()) {
            state.exp_avg = torch::zeros_like(param);
            state.exp_avg_sq = torch::zeros_like(param);
            state.step = 0;
        }

        state.step += 1;
        SVRECON_ADAM_STEP::unbiased_adam_step(
            /*sparse=*/false,
            param,
            grad,
            state.exp_avg,
            state.exp_avg_sq,
            static_cast<double>(state.step),
            static_cast<double>(lr),
            static_cast<double>(optimizer_beta1_),
            static_cast<double>(optimizer_beta2_),
            optimizer_eps_);
    };

    step_group(_geo_grid_pts_, adam_geo_, optimizer_geo_lr_);
    if (optimizer_log_s_lr_ > 0.0f) {
        step_group(log_s_, adam_log_s_, optimizer_log_s_lr_);
    }
    step_group(sh0_, adam_sh0_, optimizer_sh0_lr_);
    step_group(shs_, adam_shs_, optimizer_shs_lr_);
}

void VoxelModel::schedulerStep()
{
    if (!optimizer_initialized_) {
        return;
    }
    scheduler_epoch_ += 1;
    if (std::binary_search(
            scheduler_milestones_.begin(),
            scheduler_milestones_.end(),
            static_cast<int>(scheduler_epoch_))) {
        optimizer_geo_lr_ *= scheduler_gamma_;
        optimizer_sh0_lr_ *= scheduler_gamma_;
        optimizer_shs_lr_ *= scheduler_gamma_;
        optimizer_log_s_lr_ *= scheduler_gamma_;
    }
}

void VoxelModel::pruning(const torch::Tensor& prune_mask) {
    auto mask = prune_mask.to(torch::kBool).to(device_type_).contiguous();
    if (mask.dim() == 2 && mask.size(1) == 1) {
        mask = mask.squeeze(1);
    }
    TORCH_CHECK(mask.dim() == 1, "pruning: mask must be [N] or [N,1]");
    const int64_t N_before = mask.size(0);

    auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(device_type_);
    auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(device_type_);
    auto ensure_bool = [&](torch::Tensor& t) {
        if (!t.defined() || t.size(0) != N_before) {
            t = torch::zeros({N_before}, bool_opts);
        } else if (t.device() != mask.device()) {
            t = t.to(mask.device());
        }
    };
    ensure_bool(is_orb_voxel_);
    ensure_bool(is_inactive_geo_voxel_);
    ensure_bool(is_rgbd_fill_render_holes_voxel_);
    if (!is_leaf_.defined() || is_leaf_.size(0) != N_before) {
        is_leaf_ = torch::ones({N_before, 1}, bool_opts);
    } else {
        is_leaf_ = is_leaf_.to(mask.device()).to(torch::kBool).reshape({N_before, 1});
    }
    // SVRecon retains fine-level internal parents for hierarchical continuity.
    // They are not rendered and must not be removed by leaf-surface pruning.
    mask = mask & is_leaf_.view({-1});
    if (!exist_since_iter_.defined() || exist_since_iter_.size(0) != N_before) {
        exist_since_iter_ = torch::zeros({N_before}, i32_opts);
    } else if (exist_since_iter_.device() != mask.device()) {
        exist_since_iter_ = exist_since_iter_.to(mask.device());
    }
    if (!exist_since_kf_.defined() || exist_since_kf_.size(0) != N_before) {
        exist_since_kf_ = torch::full({N_before}, static_cast<int32_t>(-1), i32_opts);
    } else if (exist_since_kf_.device() != mask.device()) {
        exist_since_kf_ = exist_since_kf_.to(mask.device());
    }

    auto kept_idx = torch::nonzero(~mask).view({-1}).to(torch::kLong).contiguous();
    if (kept_idx.numel() == 0) {
        return;
    }

    auto old_octpath = this->oct_path_.contiguous();
    auto old_octlevel = this->oct_level_.contiguous();
    auto old_vox_key = this->vox_key_.contiguous().clone();
    auto old_geo_grid_pts = this->_geo_grid_pts_.detach().contiguous();
    auto old_vox_grid_pts_val =
        old_geo_grid_pts.index({old_vox_key.to(old_geo_grid_pts.device())}).contiguous();
    ensureSvrasterSdfField();
    auto old_vox_sdf_values = voxelCornerScalarFromGrid_(svraster_sdf_grid_pts_);
    auto old_vox_sdf_weights = voxelCornerScalarFromGrid_(svraster_sdf_weights_);

    this->oct_path_ = old_octpath.index_select(0, kept_idx.to(old_octpath.device())).contiguous();
    this->oct_level_ = old_octlevel.index_select(0, kept_idx.to(old_octlevel.device())).contiguous();
    auto [new_center, new_size] = decodeOctpath(
        this->oct_path_.contiguous(),
        this->oct_level_.contiguous(),
        this->scene_center_.contiguous(),
        this->scene_extent_.contiguous());
    auto [new_grid_pts_key, new_vox_key] =
        buildGridPtsLink(this->oct_path_.contiguous(), this->oct_level_.contiguous(), max_num_levels_);
    this->center_ = new_center.contiguous();
    this->size_ = new_size.squeeze(1).contiguous();
    this->vox_size_inv_ = 1.0f / this->size_;
    this->grid_pts_key_ = new_grid_pts_key.contiguous();
    this->vox_key_ = new_vox_key.contiguous();

    auto kept_idx_sh0 = kept_idx.to(this->sh0_.device());
    auto old_subdiv_grad = this->subdiv_p_.grad();
    this->sh0_ = this->sh0_.detach().index_select(0, kept_idx_sh0).contiguous().requires_grad_(true);
    this->shs_ = this->shs_.detach().index_select(0, kept_idx_sh0).contiguous().requires_grad_(true);
    this->subdiv_p_ =
        this->subdiv_p_.detach().index_select(0, kept_idx.to(this->subdiv_p_.device()))
            .contiguous().requires_grad_(true);
    if (this->subdiv_meta_.defined() &&
        this->subdiv_meta_.dim() == 2 &&
        this->subdiv_meta_.size(0) == old_octpath.size(0)) {
        this->subdiv_meta_ =
            this->subdiv_meta_.to(this->subdiv_p_.device())
                .to(torch::kFloat32)
                .index_select(0, kept_idx.to(this->subdiv_p_.device()))
                .contiguous();
    } else {
        this->subdiv_meta_ = torch::zeros_like(this->subdiv_p_);
    }
    if (old_subdiv_grad.defined()) {
        this->subdiv_p_.mutable_grad() =
            old_subdiv_grad.index_select(0, kept_idx.to(old_subdiv_grad.device())).contiguous();
    }

    auto new_vox_val =
        old_vox_grid_pts_val.index_select(0, kept_idx.to(old_vox_grid_pts_val.device())).contiguous();
    this->_geo_grid_pts_ =
        aggregateVoxelCornersIntoGridPts(
            this->grid_pts_key_.size(0),
            this->vox_key_,
            new_vox_val).detach().contiguous().requires_grad_(true);
    auto kept_sdf =
        old_vox_sdf_values.index_select(0, kept_idx.to(old_vox_sdf_values.device())).contiguous();
    auto kept_weights =
        old_vox_sdf_weights.index_select(0, kept_idx.to(old_vox_sdf_weights.device())).contiguous();
    rebuildSvrasterSdfFieldFromVoxelCorners_(kept_sdf, kept_weights);

    this->max_w_ = torch::zeros(
        {center_.size(0), 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

    auto remap_bool = [&](torch::Tensor& t) {
        t = t.to(mask.device()).to(torch::kBool)
                .index_select(0, kept_idx.to(mask.device()))
                .contiguous();
    };
    remap_bool(is_orb_voxel_);
    remap_bool(is_inactive_geo_voxel_);
    remap_bool(is_rgbd_fill_render_holes_voxel_);
    is_leaf_ = is_leaf_.index_select(0, kept_idx.to(is_leaf_.device())).contiguous();
    exist_since_iter_ =
        exist_since_iter_.to(mask.device()).to(torch::kInt32)
            .index_select(0, kept_idx.to(mask.device())).contiguous();
    exist_since_kf_ =
        exist_since_kf_.to(mask.device()).to(torch::kInt32)
            .index_select(0, kept_idx.to(mask.device())).contiguous();
    VOXEL_MODEL_TENSORS_TO_VEC
}

void VoxelModel::subdividing(const torch::Tensor& subdivide_mask) {
    auto mask = subdivide_mask.to(torch::kBool).to(device_type_).contiguous();
    if (mask.dim() == 2 && mask.size(1) == 1) {
        mask = mask.squeeze(1);
    }
    TORCH_CHECK(mask.dim() == 1, "subdividing: mask must be [N] or [N,1]");
    const int64_t N_before = mask.size(0);

    auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(device_type_);
    auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(device_type_);
    auto ensure_bool = [&](torch::Tensor& t) {
        if (!t.defined() || t.size(0) != N_before) {
            t = torch::zeros({N_before}, bool_opts);
        } else if (t.device() != mask.device()) {
            t = t.to(mask.device());
        }
    };
    ensure_bool(is_orb_voxel_);
    ensure_bool(is_inactive_geo_voxel_);
    ensure_bool(is_rgbd_fill_render_holes_voxel_);
    if (!is_leaf_.defined() || is_leaf_.size(0) != N_before) {
        is_leaf_ = torch::ones({N_before, 1}, bool_opts);
    } else {
        is_leaf_ = is_leaf_.to(mask.device()).to(torch::kBool).reshape({N_before, 1});
    }
    mask = mask & is_leaf_.view({-1});
    if (!exist_since_iter_.defined() || exist_since_iter_.size(0) != N_before) {
        exist_since_iter_ = torch::zeros({N_before}, i32_opts);
    } else if (exist_since_iter_.device() != mask.device()) {
        exist_since_iter_ = exist_since_iter_.to(mask.device());
    }
    if (!exist_since_kf_.defined() || exist_since_kf_.size(0) != N_before) {
        exist_since_kf_ = torch::full({N_before}, static_cast<int32_t>(-1), i32_opts);
    } else if (exist_since_kf_.device() != mask.device()) {
        exist_since_kf_ = exist_since_kf_.to(mask.device());
    }
    auto orb_before = is_orb_voxel_.to(torch::kBool).contiguous();
    auto inactive_geo_before = is_inactive_geo_voxel_.to(torch::kBool).contiguous();
    auto rgbd_fill_render_holes_before =
        is_rgbd_fill_render_holes_voxel_.to(torch::kBool).contiguous();
    auto exist_since_before = exist_since_iter_.to(torch::kInt32).contiguous();
    auto exist_since_kf_before = exist_since_kf_.to(torch::kInt32).contiguous();

    auto subdiv_idx = torch::nonzero(mask).view({-1});
    if (subdiv_idx.numel() == 0) {
        return;
    }

    auto old_octpath = this->oct_path_.contiguous();
    auto old_octlevel = this->oct_level_.contiguous();
    auto depth_rel =
        old_octlevel.to(mask.device()).to(torch::kInt32).reshape({-1}) - outside_level_;
    auto parent_keep_mask = mask & (depth_rel >= 9);
    auto kept_idx = torch::nonzero((~mask) | parent_keep_mask)
                        .view({-1}).to(torch::kLong).contiguous();

    auto old_vox_key = this->vox_key_.contiguous().clone();
    auto old_geo_grid_pts = this->_geo_grid_pts_.detach().contiguous();
    auto old_vox_grid_pts_val =
        old_geo_grid_pts.index({old_vox_key.to(old_geo_grid_pts.device())}).contiguous();
    ensureSvrasterSdfField();
    auto old_vox_sdf_values = voxelCornerScalarFromGrid_(svraster_sdf_grid_pts_);
    auto old_vox_sdf_weights = voxelCornerScalarFromGrid_(svraster_sdf_weights_);

    auto [child_octpath, child_octlevel] =
        genChildrenOctpath(
            old_octpath.index_select(0, subdiv_idx.to(old_octpath.device())).contiguous(),
            old_octlevel.index_select(0, subdiv_idx.to(old_octlevel.device())).contiguous(),
            max_num_levels_);
    this->oct_path_ = torch::cat(
        {old_octpath.index_select(0, kept_idx.to(old_octpath.device())).contiguous(),
         child_octpath.to(old_octpath.device()).contiguous()}, 0).contiguous();
    this->oct_level_ = torch::cat(
        {old_octlevel.index_select(0, kept_idx.to(old_octlevel.device())).contiguous(),
         child_octlevel.to(old_octlevel.device()).contiguous()}, 0).contiguous();

    auto old_subdiv_grad = this->subdiv_p_.grad();
    auto subdiv_children =
        this->subdiv_p_.detach()
            .index_select(0, subdiv_idx.to(this->subdiv_p_.device()))
            .repeat_interleave(8, 0)
            .contiguous();
    this->subdiv_p_ = torch::cat(
        {this->subdiv_p_.detach().index_select(0, kept_idx.to(this->subdiv_p_.device())).contiguous(),
         subdiv_children}, 0).contiguous().requires_grad_(true);
    if (this->subdiv_meta_.defined() &&
        this->subdiv_meta_.dim() == 2 &&
        this->subdiv_meta_.size(0) == old_octpath.size(0)) {
        auto kept_meta =
            this->subdiv_meta_.to(this->subdiv_p_.device())
                .to(torch::kFloat32)
                .index_select(0, kept_idx.to(this->subdiv_p_.device()))
                .contiguous();
        auto child_meta =
            this->subdiv_meta_.to(this->subdiv_p_.device())
                .to(torch::kFloat32)
                .index_select(0, subdiv_idx.to(this->subdiv_p_.device()))
                .repeat_interleave(8, 0)
                .contiguous();
        this->subdiv_meta_ = torch::cat({kept_meta, child_meta}, 0).contiguous();
    } else {
        this->subdiv_meta_ = torch::zeros_like(this->subdiv_p_);
    }
    if (old_subdiv_grad.defined()) {
        this->subdiv_p_.mutable_grad() = torch::cat(
            {old_subdiv_grad.index_select(0, kept_idx.to(old_subdiv_grad.device())).contiguous(),
             subdiv_children.to(old_subdiv_grad.device()).contiguous()}, 0).contiguous();
    }

    auto sh0_children =
        this->sh0_.detach()
            .index_select(0, subdiv_idx.to(this->sh0_.device()))
            .repeat_interleave(8, 0)
            .contiguous();
    this->sh0_ = torch::cat(
        {this->sh0_.detach().index_select(0, kept_idx.to(this->sh0_.device())).contiguous(),
         sh0_children}, 0).contiguous().requires_grad_(true);

    auto shs_children =
        this->shs_.detach()
            .index_select(0, subdiv_idx.to(this->shs_.device()))
            .repeat_interleave(8, 0)
            .contiguous();
    this->shs_ = torch::cat(
        {this->shs_.detach().index_select(0, kept_idx.to(this->shs_.device())).contiguous(),
         shs_children}, 0).contiguous().requires_grad_(true);

    auto [new_center, new_size] = decodeOctpath(
        this->oct_path_.contiguous(),
        this->oct_level_.contiguous(),
        this->scene_center_.contiguous(),
        this->scene_extent_.contiguous());
    auto [new_grid_pts_key, new_vox_key] =
        buildGridPtsLink(this->oct_path_.contiguous(), this->oct_level_.contiguous(), max_num_levels_);
    this->center_ = new_center.contiguous();
    this->size_ = new_size.squeeze(1).contiguous();
    this->vox_size_inv_ = 1.0f / this->size_;
    this->grid_pts_key_ = new_grid_pts_key.contiguous();
    this->vox_key_ = new_vox_key.contiguous();

    auto kept_vox_val =
        old_vox_grid_pts_val.index_select(0, kept_idx.to(old_vox_grid_pts_val.device())).contiguous();
    auto subdiv_vox_val =
        subdivideVoxelCornerValues(
            old_vox_grid_pts_val.index_select(
                0, subdiv_idx.to(old_vox_grid_pts_val.device())).contiguous());
    auto new_vox_val = torch::cat({kept_vox_val, subdiv_vox_val}, 0).contiguous();
    this->_geo_grid_pts_ =
        aggregateVoxelCornersIntoGridPts(
            this->grid_pts_key_.size(0),
            this->vox_key_,
            new_vox_val).detach().contiguous().requires_grad_(true);
    {
        auto kept_sdf =
            old_vox_sdf_values.index_select(0, kept_idx.to(old_vox_sdf_values.device())).contiguous();
        auto subdiv_sdf =
            subdivideVoxelCornerValues(
                old_vox_sdf_values.index_select(
                    0, subdiv_idx.to(old_vox_sdf_values.device())).contiguous());
        auto new_sdf = torch::cat({kept_sdf, subdiv_sdf}, 0).contiguous();

        auto kept_weights =
            old_vox_sdf_weights.index_select(0, kept_idx.to(old_vox_sdf_weights.device())).contiguous();
        auto subdiv_weights =
            subdivideVoxelCornerValues(
                old_vox_sdf_weights.index_select(
                    0, subdiv_idx.to(old_vox_sdf_weights.device())).contiguous());
        auto new_weights = torch::cat({kept_weights, subdiv_weights}, 0).contiguous();
        rebuildSvrasterSdfFieldFromVoxelCorners_(new_sdf, new_weights);
    }
    this->max_w_ = torch::zeros(
        {center_.size(0), 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

    auto kept_leaf = is_leaf_.index_select(0, kept_idx.to(is_leaf_.device())).contiguous();
    auto kept_parent_local = parent_keep_mask.index_select(
        0, kept_idx.to(parent_keep_mask.device())).view({-1, 1});
    kept_leaf = kept_leaf & (~kept_parent_local);
    auto child_leaf = torch::ones(
        {subdiv_idx.numel() * 8, 1},
        torch::TensorOptions().dtype(torch::kBool).device(is_leaf_.device()));
    is_leaf_ = torch::cat({kept_leaf, child_leaf}, 0).contiguous();

    auto remap_child_bool = [&](torch::Tensor& t) {
        auto kept = t.to(mask.device()).to(torch::kBool)
                        .index_select(0, kept_idx.to(mask.device()))
                        .contiguous();
        auto child = t.to(mask.device()).to(torch::kBool)
                         .index_select(0, subdiv_idx.to(mask.device()))
                         .repeat_interleave(8, 0)
                         .contiguous();
        t = torch::cat({kept, child}, 0).to(device_type_).to(torch::kBool).contiguous();
    };
    remap_child_bool(is_orb_voxel_);
    remap_child_bool(is_inactive_geo_voxel_);
    remap_child_bool(is_rgbd_fill_render_holes_voxel_);

    auto kept_exist_iter =
        exist_since_before.to(mask.device()).index_select(0, kept_idx.to(mask.device())).contiguous();
    auto child_exist_iter =
        exist_since_before.to(mask.device()).index_select(0, subdiv_idx.to(mask.device()))
            .repeat_interleave(8, 0).contiguous();
    if (topology_birth_iter_ >= 0) {
        child_exist_iter = torch::full_like(child_exist_iter, topology_birth_iter_);
    }
    exist_since_iter_ =
        torch::cat({kept_exist_iter, child_exist_iter}, 0)
            .to(device_type_).to(torch::kInt32).contiguous();

    auto kept_exist_kf =
        exist_since_kf_before.to(mask.device()).index_select(0, kept_idx.to(mask.device())).contiguous();
    auto child_exist_kf =
        exist_since_kf_before.to(mask.device()).index_select(0, subdiv_idx.to(mask.device()))
            .repeat_interleave(8, 0).contiguous();
    if (topology_birth_kf_ >= 0) {
        child_exist_kf = torch::full_like(child_exist_kf, topology_birth_kf_);
    }
    exist_since_kf_ =
        torch::cat({kept_exist_kf, child_exist_kf}, 0)
            .to(device_type_).to(torch::kInt32).contiguous();
    VOXEL_MODEL_TENSORS_TO_VEC
}

torch::Tensor VoxelModel::subdivisionPriority() const {
    torch::Tensor p = this->subdiv_meta_;
    if (!p.defined() || p.numel() == 0) {
        return torch::Tensor();
    }
    if (p.dim() == 2 && p.size(1) == 1) p = p.squeeze(1);
    return p.contiguous();
}

void VoxelModel::accumulateSubdivisionPriority() {
    torch::Tensor g = this->subdiv_p_.grad();
    if (!g.defined() || !this->subdiv_p_.defined() || this->subdiv_p_.numel() == 0) {
        return;
    }
    if (!this->subdiv_meta_.defined() || this->subdiv_meta_.sizes() != this->subdiv_p_.sizes()) {
        this->subdiv_meta_ = torch::zeros_like(this->subdiv_p_);
    }
    this->subdiv_meta_ =
        (this->subdiv_meta_.to(g.device()).to(torch::kFloat32) +
         g.detach().to(torch::kFloat32))
            .contiguous();
}

void VoxelModel::resetSubdivisionPriority() {
    if (this->subdiv_meta_.defined()) {
        this->subdiv_meta_.zero_();
    }
    this->subdiv_p_.mutable_grad() = torch::Tensor();
}

void VoxelModel::freezeVoxGeo() {
    const int64_t N = center_.size(0);
    auto care_idx = torch::arange(N, torch::dtype(torch::kLong).device(device_type_));
    torch::NoGradGuard no_grad;
    frozen_vox_geo_ = gatherSvreconGeoParams(vox_key_, care_idx, _geo_grid_pts_)[0].contiguous();
    _geo_grid_pts_.set_requires_grad(false);
}

void VoxelModel::unfreezeVoxGeo() {
    frozen_vox_geo_.reset();              // make it undefined
    _geo_grid_pts_.set_requires_grad(true);
}

std::unordered_map<std::string, torch::Tensor> VoxelModel::render(
    const sv::MiniCam& cam,
    int im_height,
    int im_width,
    const torch::Tensor& gt_image,
    const char* color_mode,
    bool track_max_w,
    std::optional<float> ss,
    bool output_depth,
    bool output_normal,
    bool output_T,
    bool rand_bg,
    bool use_auto_exposure,
    const sv::RenderOpts& other_opt) const
{
    return renderSvreconDirect(
        cam,
        im_height,
        im_width,
        _geo_grid_pts_,
        sh0_,
        shs_,
        subdiv_p_,
        log_s_,
        oct_path_,
        is_leaf_,
        center_,
        size_,
        vox_key_,
        frozen_vox_geo_,
        active_sh_degree_,
        white_background_,
        black_background_,
        ss_,
        gt_image,
        color_mode,
        track_max_w,
        ss,
        output_depth,
        output_normal,
        output_T,
        rand_bg,
        use_auto_exposure,
        other_opt);
}

void VoxelModel::applyTvOnDensityField(float lambda_tv_density) {
    if (!_geo_grid_pts_.defined() || _geo_grid_pts_.numel() == 0 ||
        !vox_key_.defined() || vox_key_.numel() == 0 ||
        !vox_size_inv_.defined() || vox_size_inv_.numel() == 0) {
        return;
    }

    torch::Tensor grad = _geo_grid_pts_.grad();
    if (!grad.defined() || grad.sizes() != _geo_grid_pts_.sizes()) {
        _geo_grid_pts_.mutable_grad() = torch::zeros_like(_geo_grid_pts_);
        grad = _geo_grid_pts_.grad();
    } else if (!grad.is_contiguous()) {
        _geo_grid_pts_.mutable_grad() = grad.contiguous();
        grad = _geo_grid_pts_.grad();
    }
    SVRECON_TV_COMPUTE::total_variation_bw(
        _geo_grid_pts_,
        vox_key_,
        lambda_tv_density,
        vox_size_inv_,
        /*no_tv_s=*/true,
        /*tv_sparse=*/false,
        grad);
}

namespace {
struct SvreconRegularizerTable {
    torch::Tensor vox_key;
    torch::Tensor grid_voxel_coord;
    torch::Tensor grid_voxel_size;
    torch::Tensor grid_mask;
    torch::Tensor grid_keys;
    torch::Tensor grid2voxel;
    torch::Tensor active_list;
    int grid_level = -1;
    int grid_res = 0;
    float vox_size_inv = 0.0f;
    bool valid = false;
};
} // namespace

static torch::Tensor ensureGeoGridGrad(torch::Tensor& geo_grid_pts)
{
    torch::Tensor grad = geo_grid_pts.grad();
    if (!grad.defined() || grad.sizes() != geo_grid_pts.sizes()) {
        geo_grid_pts.mutable_grad() = torch::zeros_like(geo_grid_pts);
        grad = geo_grid_pts.grad();
    } else if (!grad.is_contiguous()) {
        geo_grid_pts.mutable_grad() = grad.contiguous();
        grad = geo_grid_pts.grad();
    }
    return grad;
}

SvreconRegularizerTable buildSvreconRegularizerTable(
    const torch::Tensor& center,
    const torch::Tensor& size,
    const torch::Tensor& vox_key,
    const torch::Tensor& oct_level,
    const torch::Tensor& is_leaf,
    const torch::Tensor& scene_center,
    const torch::Tensor& inside_extent,
    int outside_level,
    torch::DeviceType device_type)
{
    SvreconRegularizerTable out;
    if (!center.defined() || !size.defined() || !vox_key.defined() ||
        !oct_level.defined() || center.size(0) == 0 || vox_key.size(0) == 0 ||
        vox_key.dim() != 2 || vox_key.size(1) != 8) {
        return out;
    }

    auto dev = center.device();
    auto level_flat = oct_level.to(dev).reshape({-1}).to(torch::kInt64);
    if (level_flat.numel() == 0) {
        return out;
    }
    int grid_level =
        static_cast<int>(level_flat.max().item<int64_t>()) - outside_level;
    grid_level = std::clamp(grid_level, 1, 9);
    const int grid_res = 1 << grid_level;

    const float inside_extent_scalar =
        inside_extent.defined() && inside_extent.numel() > 0
            ? std::max(1.0e-6f, inside_extent.reshape({-1})[0].item<float>())
            : 1.0f;

    auto center_f = center.to(dev).to(torch::kFloat32).contiguous();
    auto size_f = size.to(dev).to(torch::kFloat32).reshape({-1}).contiguous();
    auto scene_center_f = scene_center.to(dev).to(torch::kFloat32).contiguous().view({3});
    auto grid_voxel_coord =
        (((center_f - size_f.view({-1, 1}) * 0.5f) -
          (scene_center_f - inside_extent_scalar * 0.5f)) /
         inside_extent_scalar) *
        static_cast<float>(grid_res);
    out.grid_voxel_coord = torch::round(grid_voxel_coord).contiguous();
    out.grid_voxel_size =
        torch::round((size_f / inside_extent_scalar) * static_cast<float>(grid_res))
            .contiguous();
    out.vox_key = vox_key.to(dev).to(torch::kLong).contiguous();
    auto leaf = is_leaf.defined() && is_leaf.size(0) == center_f.size(0)
        ? is_leaf.to(dev).to(torch::kBool).reshape({-1, 1}).contiguous()
        : torch::ones(
              {center_f.size(0), 1},
              torch::TensorOptions().dtype(torch::kBool).device(device_type));

    auto table = SVRECON_UTILS::valid_gradient_table(
        center_f,
        size_f,
        scene_center_f,
        inside_extent_scalar,
        grid_level,
        leaf);
    out.grid_mask = std::get<0>(table).to(dev).contiguous();
    auto grid_keys = std::get<1>(table).to(dev).to(torch::kInt32).contiguous();
    auto grid2voxel = std::get<2>(table).to(dev).to(torch::kInt32).contiguous();
    auto sort_pair = grid_keys.sort();
    out.grid_keys = std::get<0>(sort_pair).contiguous();
    out.grid2voxel = grid2voxel.index_select(0, std::get<1>(sort_pair)).contiguous();
    if (out.grid_keys.numel() == 0 || out.grid2voxel.numel() == 0) {
        return out;
    }
    out.active_list = torch::arange(
        out.grid_keys.size(0),
        torch::TensorOptions().dtype(torch::kInt32).device(dev)).contiguous();
    out.grid_level = grid_level;
    out.grid_res = grid_res;
    out.vox_size_inv = static_cast<float>(grid_res) / inside_extent_scalar;
    out.valid = true;
    return out;
}

void VoxelModel::applySvreconGridEikonalField(float lambda_ge_density)
{
    if (lambda_ge_density <= 0.0f ||
        !_geo_grid_pts_.defined() || _geo_grid_pts_.numel() == 0) {
        return;
    }

    torch::Tensor grad = ensureGeoGridGrad(_geo_grid_pts_);
    auto table = buildSvreconRegularizerTable(
        center_,
        size_,
        vox_key_,
        oct_level_,
        is_leaf_,
        scene_center_,
        inside_extent_,
        outside_level_,
        device_type_);
    if (!table.valid) {
        return;
    }

    SVRECON_GE_COMPUTE::grid_eikonal_bw(
        _geo_grid_pts_,
        table.vox_key,
        table.grid_voxel_coord,
        table.grid_voxel_size,
        table.grid_res,
        table.grid_mask,
        table.grid_keys,
        table.grid2voxel,
        table.active_list,
        lambda_ge_density,
        table.vox_size_inv,
        /*no_tv_s=*/true,
        /*tv_sparse=*/false,
        grad);
}

void VoxelModel::applySvreconLaplacianSmoothnessField(float lambda_ls_density)
{
    if (lambda_ls_density <= 0.0f ||
        !_geo_grid_pts_.defined() || _geo_grid_pts_.numel() == 0) {
        return;
    }

    torch::Tensor grad = ensureGeoGridGrad(_geo_grid_pts_);
    auto table = buildSvreconRegularizerTable(
        center_,
        size_,
        vox_key_,
        oct_level_,
        is_leaf_,
        scene_center_,
        inside_extent_,
        outside_level_,
        device_type_);
    if (!table.valid) {
        return;
    }

    SVRECON_LS_COMPUTE::laplacian_smoothness_bw(
        _geo_grid_pts_,
        table.vox_key,
        table.grid_voxel_coord,
        table.grid_voxel_size,
        table.grid_res,
        table.grid_mask,
        table.grid_keys,
        table.grid2voxel,
        table.active_list,
        lambda_ls_density,
        table.vox_size_inv,
        /*no_tv_s=*/true,
        /*tv_sparse=*/false,
        grad);
}

torch::Tensor VoxelModel::svreconLocalEikonalLoss(
    const float lambda_local_ge_density,
    const int min_inside_level) const
{
    if (lambda_local_ge_density <= 0.0f ||
        !_geo_grid_pts_.defined() || _geo_grid_pts_.numel() == 0 ||
        !vox_key_.defined() || vox_key_.numel() == 0 ||
        !oct_level_.defined() || oct_level_.numel() == 0 ||
        !size_.defined() || size_.numel() == 0) {
        return torch::zeros({}, _geo_grid_pts_.options());
    }

    auto levels = oct_level_.to(_geo_grid_pts_.device()).to(torch::kInt32).reshape({-1});
    auto mask = levels >= (outside_level_ + min_inside_level);
    if (is_leaf_.defined() && is_leaf_.size(0) == mask.size(0)) {
        mask = mask & is_leaf_.to(mask.device()).to(torch::kBool).reshape({-1});
    }
    auto idx = torch::nonzero(mask).reshape({-1}).to(torch::kLong);
    if (idx.numel() == 0) {
        return torch::zeros({}, _geo_grid_pts_.options());
    }

    auto keys = vox_key_.to(_geo_grid_pts_.device()).to(torch::kLong)
                    .index_select(0, idx).contiguous();
    auto corners = _geo_grid_pts_.index_select(0, keys.reshape({-1}))
                       .reshape({idx.numel(), 8});
    auto inv_size = (1.0f / size_.to(corners.device()).to(torch::kFloat32)
                                .reshape({-1}).index_select(0, idx))
                        .view({-1, 1});

    auto gx = 0.25f * (
        corners.index({torch::indexing::Slice(), torch::indexing::Slice(4, 8)}).sum(1, true) -
        corners.index({torch::indexing::Slice(), torch::indexing::Slice(0, 4)}).sum(1, true));
    auto gy = 0.25f * (
        corners.index_select(
            1,
            torch::tensor({2, 3, 6, 7},
                          torch::TensorOptions().dtype(torch::kLong).device(corners.device())))
            .sum(1, true) -
        corners.index_select(
            1,
            torch::tensor({0, 1, 4, 5},
                          torch::TensorOptions().dtype(torch::kLong).device(corners.device())))
            .sum(1, true));
    auto gz = 0.25f * (
        corners.index_select(
            1,
            torch::tensor({1, 3, 5, 7},
                          torch::TensorOptions().dtype(torch::kLong).device(corners.device())))
            .sum(1, true) -
        corners.index_select(
            1,
            torch::tensor({0, 2, 4, 6},
                          torch::TensorOptions().dtype(torch::kLong).device(corners.device())))
            .sum(1, true));
    auto grad_world = torch::cat({gx, gy, gz}, 1) * inv_size;
    return lambda_local_ge_density *
           torch::square(torch::linalg_vector_norm(grad_world, 2, {1}) - 1.0f).mean();
}

VoxelModel::SchedulerState VoxelModel::schedulerState() const
{
    SchedulerState state;
    state.valid = optimizer_initialized_;
    state.last_epoch = scheduler_epoch_;
    state.geo_lr = optimizer_geo_lr_;
    state.sh0_lr = optimizer_sh0_lr_;
    state.shs_lr = optimizer_shs_lr_;
    state.log_s_lr = optimizer_log_s_lr_;
    return state;
}

void VoxelModel::schedulerLoadState(const SchedulerState& state)
{
    if (!state.valid) {
        return;
    }
    scheduler_epoch_ = state.last_epoch;
    optimizer_geo_lr_ = state.geo_lr;
    optimizer_sh0_lr_ = state.sh0_lr;
    optimizer_shs_lr_ = state.shs_lr;
    optimizer_log_s_lr_ = state.log_s_lr;
}

/* static */ torch::Tensor
VoxelModel::camPosition_(const MiniCam& cam, torch::Device d) {
    // c2w: 4x4 or w2c inv; assume you have cam.c2w as float[4x4] or Tensor
    // position = c2w[0:3,3]
    auto c2w = cam.c2w.to(d).contiguous();              // (4,4)
    return c2w.index({torch::indexing::Slice(0,3), 3}); // (3)
}

/* static */ torch::Tensor
VoxelModel::camForward_(const MiniCam& cam, torch::Device d) {
    // forward = +Z axis of camera in world (c2w[0:3,2]); normalize.
    auto c2w = cam.c2w.to(d).contiguous();
    auto fwd = c2w.index({torch::indexing::Slice(0,3), 2}); // (3)
    auto nrm = fwd.norm().clamp_min(1e-8);
    return fwd / nrm;
}

/* static */ float
VoxelModel::camPixSize_(const MiniCam& cam) {
    // world distance per pixel per unit depth ≈ max(1/fx, 1/fy)
    // (fx,fy) are pixel focal lengths.
    float inv_fx = 1.0f / std::max(1e-8f, cam.fx);
    float inv_fy = 1.0f / std::max(1e-8f, cam.fy);
    return std::max(inv_fx, inv_fy);
}

// VoxelModel::savePly — WebGL viewer compatible (SH degree 1)
namespace {
constexpr int MAX_NUM_LEVELS = 16;

static inline std::array<int64_t,3> decode_ijk(uint64_t path, int lv) {
    path >>= (3 * (MAX_NUM_LEVELS - lv));
    int64_t i=0,j=0,k=0;
    for (int l=0; l<lv; ++l) {
        uint64_t bits = (path & 0x7u);
        i |= static_cast<int64_t>((bits >> 2) & 0x1u) << l;
        j |= static_cast<int64_t>((bits >> 1) & 0x1u) << l;
        k |= static_cast<int64_t>((bits >> 0) & 0x1u) << l;
        path >>= 3;
    }
    return {i,j,k};
}

static torch::Tensor decode_centers_from_octree(const torch::Tensor& octpath,
                                                const torch::Tensor& octlevel,
                                                const torch::Tensor& scene_center,
                                                const torch::Tensor& scene_extent) {
    auto op = octpath.contiguous().view({-1}).to(torch::kInt64);
    auto lv = octlevel.contiguous().view({-1}).to(torch::kInt32);

    const int64_t N = op.size(0);
    const float cx = scene_center[0].item<float>();
    const float cy = scene_center[1].item<float>();
    const float cz = scene_center[2].item<float>();
    const float extent = scene_extent.item<float>();

    const float minx = cx - 0.5f * extent;
    const float miny = cy - 0.5f * extent;
    const float minz = cz - 0.5f * extent;

    auto xyz = torch::empty({N,3}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    auto xyz_a = xyz.accessor<float,2>();
    const auto* op_ptr = op.data_ptr<int64_t>();
    const auto* lv_ptr = lv.data_ptr<int32_t>();

    for (int64_t n=0; n<N; ++n) {
        const uint64_t p = static_cast<uint64_t>(op_ptr[n]);
        const int l = lv_ptr[n];
        const auto ijk = decode_ijk(p, l);
        const float vox_size = std::ldexp(extent, -l);
        xyz_a[n][0] = minx + (static_cast<float>(ijk[0]) + 0.5f) * vox_size;
        xyz_a[n][1] = miny + (static_cast<float>(ijk[1]) + 0.5f) * vox_size;
        xyz_a[n][2] = minz + (static_cast<float>(ijk[2]) + 0.5f) * vox_size;
    }
    return xyz;
}

} // namespace

void VoxelModel::savePly(const std::filesystem::path& result_path)
{
    torch::NoGradGuard ng;
    namespace fs = std::filesystem;
    if (!result_path.parent_path().empty())
        fs::create_directories(result_path.parent_path());

    // Pull to CPU
    auto op_cpu   = oct_path_.detach().to(torch::kCPU).contiguous();        // [N]
    auto lv_cpu   = oct_level_.detach().to(torch::kCPU).contiguous();       // [N] or [N,1]
    auto sc_cpu   = scene_center_.detach().to(torch::kCPU).contiguous();    // [3]
    auto se_cpu   = scene_extent_.detach().to(torch::kCPU).contiguous();    // [1]
    auto sh0_cpu  = sh0_.detach().to(torch::kCPU).contiguous();             // [N,3] or [N,1,3]
    auto shs_cpu  = shs_.detach().to(torch::kCPU).contiguous();             // [N,K,3] (K>=0)
    auto voxkey   = vox_key_.detach().to(torch::kCPU).contiguous();         // [N,8]
    auto geo_cpu  = _geo_grid_pts_.detach().to(torch::kCPU).contiguous();   // [G]
    if (is_leaf_.defined() && is_leaf_.size(0) == op_cpu.size(0)) {
        auto leaf_idx = torch::nonzero(
            is_leaf_.detach().to(torch::kCPU).to(torch::kBool).reshape({-1}))
                            .reshape({-1}).to(torch::kLong);
        op_cpu = op_cpu.index_select(0, leaf_idx).contiguous();
        lv_cpu = lv_cpu.index_select(0, leaf_idx).contiguous();
        sh0_cpu = sh0_cpu.index_select(0, leaf_idx).contiguous();
        shs_cpu = shs_cpu.index_select(0, leaf_idx).contiguous();
        voxkey = voxkey.index_select(0, leaf_idx).contiguous();
    }

    // Decode centers (viewer’s computation)
    torch::Tensor xyz = decode_centers_from_octree(op_cpu, lv_cpu, sc_cpu, se_cpu); // [N,3]
    const int64_t N = xyz.size(0);
    if (N == 0) { std::cerr << "[savePly] No voxels.\n"; return; }

    // Flatten lv to [N]
    auto lv_i32 = lv_cpu.view({-1}).to(torch::kInt32).contiguous();
    auto lv_ptr = lv_i32.data_ptr<int32_t>();
    int lv_min =  999, lv_max = -999;
    for (int64_t n=0; n<N; ++n) { lv_min = std::min(lv_min, lv_ptr[n]); lv_max = std::max(lv_max, lv_ptr[n]); }

    // Build SH dc = f_dc_0..2
    torch::Tensor fdc;
    if (sh0_cpu.dim()==3 && sh0_cpu.size(1)==1 && sh0_cpu.size(2)==3)      fdc = sh0_cpu.view({N,3});
    else if (sh0_cpu.dim()==2 && sh0_cpu.size(1)==3)                        fdc = sh0_cpu;
    else { std::cerr << "[savePly] Unexpected sh0_ shape " << sh0_cpu.sizes() << "\n"; return; }
    fdc = fdc.to(torch::kFloat32).contiguous();

    // SH rest: enforce degree-1 → exactly 9 floats per voxel
    int64_t K = (shs_cpu.dim()>=2) ? shs_cpu.size(1) : 0;
    if (active_sh_degree_ < 1) {
        std::cerr << "[savePly] WARNING: active_sh_degree_ < 1; exporting degree-1 band as zeros.\n";
    }
    if (K < 3) {
        std::cerr << "[savePly] NOTE: shs_ has only " << K
                  << " coeffs per channel; padding to degree-1 (3) with zeros.\n";
    }
    torch::Tensor shs_band1;
    if (K >= 3 && shs_cpu.dim()==3 && shs_cpu.size(2)==3) {
        shs_band1 = shs_cpu.index({torch::indexing::Slice(),
                                   torch::indexing::Slice(0,3),
                                   torch::indexing::Slice()}).contiguous();  // [N,3,3]
    } else {
        // pad zeros to [N,3,3] if needed
        shs_band1 = torch::zeros({N,3,3}, torch::dtype(torch::kFloat32));
        if (K > 0 && shs_cpu.dim()==3 && shs_cpu.size(2)==3) {
            auto copyK = std::min<int64_t>(K, 3);
            shs_band1.index_put_({torch::indexing::Slice(),
                                  torch::indexing::Slice(0,copyK),
                                  torch::indexing::Slice()},
                                 shs_cpu.index({torch::indexing::Slice(),
                                                torch::indexing::Slice(0,copyK),
                                                torch::indexing::Slice()}).to(torch::kFloat32));
        }
    }
    auto frest = shs_band1.view({N, 9}).to(torch::kFloat32).contiguous(); // [N,9]

    // Vox key sanity
    if (!(voxkey.dim()==2 && voxkey.size(0)==N && voxkey.size(1)==8)) {
        std::cerr << "[savePly] ERROR: vox_key_ must be [N,8]; got " << voxkey.sizes() << "\n";
        return;
    }
    if (geo_cpu.dim()!=1) geo_cpu = geo_cpu.view({-1});
    const int64_t G = geo_cpu.size(0);

    std::vector<int64_t> keep_idx(static_cast<size_t>(N));
    std::iota(keep_idx.begin(), keep_idx.end(), 0);
    const int64_t M = (int64_t)keep_idx.size();
    auto idx_t = torch::from_blob(keep_idx.data(), {M}, torch::TensorOptions().dtype(torch::kLong)).clone();

    // Slice tensors
    xyz   = xyz.index_select(0, idx_t).contiguous();
    fdc   = fdc.index_select(0, idx_t).contiguous();
    frest = frest.index_select(0, idx_t).contiguous();
    voxkey= voxkey.index_select(0, idx_t).contiguous();

    // octpath / level to std::vector payloads
    std::vector<uint32_t> op_u32; op_u32.reserve(M);
    std::vector<uint8_t>  lv_u8;  lv_u8.reserve(M);
    {
        auto op64 = op_cpu.view({-1}).to(torch::kInt64);
        const int64_t* op_ptr = op64.data_ptr<int64_t>();
        for (auto i : keep_idx) op_u32.push_back(static_cast<uint32_t>(op_ptr[i]));
        for (auto i : keep_idx) lv_u8 .push_back(static_cast<uint8_t>(lv_ptr[i]));
    }

    // grid0..7_value
    std::array<torch::Tensor,8> grid_vals;
    int64_t out_of_range = 0;
    for (int c=0; c<8; ++c) {
        auto key_c = voxkey.index({torch::indexing::Slice(), c}).to(torch::kLong).contiguous();
        // Track OOR before clamping (debug)
        auto oor = (key_c < 0) | (key_c >= G);
        out_of_range += oor.sum().item<int64_t>();
        key_c = torch::clamp(key_c, 0, G-1);
        grid_vals[c] = geo_cpu.index_select(0, key_c).to(torch::kFloat32).contiguous(); // [M]
    }
    if (out_of_range > 0)
        std::cerr << "[savePly] WARNING: vox_key has " << out_of_range
                  << " out-of-range indices (clamped). Check syncFromPython().\n";

    // Write PLY (binary)
    std::filebuf fb;
    fb.open(result_path, std::ios::out | std::ios::binary);
    std::ostream out(&fb);
    if (out.fail()) throw std::runtime_error("savePly: open failed: " + result_path.string());

    tinyply::PlyFile ply;

    // x,y,z
    ply.add_properties_to_element(
        "vertex", {"x","y","z"},
        tinyply::Type::FLOAT32, M,
        reinterpret_cast<uint8_t*>(xyz.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // octpath (uint32)
    ply.add_properties_to_element(
        "vertex", {"octpath"},
        tinyply::Type::UINT32, M,
        reinterpret_cast<uint8_t*>(op_u32.data()),
        tinyply::Type::INVALID, 0);

    // octlevel (uint8)
    ply.add_properties_to_element(
        "vertex", {"octlevel"},
        tinyply::Type::UINT8, M,
        reinterpret_cast<uint8_t*>(lv_u8.data()),
        tinyply::Type::INVALID, 0);

    // f_dc_0..2
    {
        std::vector<std::string> names = {"f_dc_0","f_dc_1","f_dc_2"};
        ply.add_properties_to_element(
            "vertex", names,
            tinyply::Type::FLOAT32, M,
            reinterpret_cast<uint8_t*>(fdc.data_ptr<float>()),
            tinyply::Type::INVALID, 0);
    }

    // f_rest_0..8 (degree-1 only)
    {
        std::vector<std::string> names; names.reserve(9);
        for (int i=0; i<9; ++i) names.emplace_back("f_rest_" + std::to_string(i));
        ply.add_properties_to_element(
            "vertex", names,
            tinyply::Type::FLOAT32, M,
            reinterpret_cast<uint8_t*>(frest.data_ptr<float>()),
            tinyply::Type::INVALID, 0);
    }

    // grid0..7_value
    for (int c=0; c<8; ++c) {
        std::string nm = "grid" + std::to_string(c) + "_value";
        ply.add_properties_to_element(
            "vertex", {nm},
            tinyply::Type::FLOAT32, M,
            reinterpret_cast<uint8_t*>(grid_vals[c].data_ptr<float>()),
            tinyply::Type::INVALID, 0);
    }

    // Comments: handy for debugging in the viewer console
    try {
        const float cx = sc_cpu[0].item<float>(), cy = sc_cpu[1].item<float>(), cz = sc_cpu[2].item<float>();
        const float ex = se_cpu.item<float>();
        ply.get_comments().push_back("scene_center " + std::to_string(cx) + " " + std::to_string(cy) + " " + std::to_string(cz));
        ply.get_comments().push_back("scene_extent " + std::to_string(ex));
        ply.get_comments().push_back("active_sh_degree 1"); // we export degree-1
    } catch (...) {}

    ply.write(out, /*binary*/ true);
    fb.close();

}

bool VoxelModel::refreshDenseCoreBBFromCurrentVoxels()
{
    torch::Tensor pts_cpu;
    if (real_pcd_points_accum_cpu_.defined() &&
        real_pcd_points_accum_cpu_.numel() > 0) {
        pts_cpu = real_pcd_points_accum_cpu_;
    } else if (sparse_points_xyz_.defined() &&
               sparse_points_xyz_.numel() > 0) {
        pts_cpu = sparse_points_xyz_
                      .detach()
                      .to(torch::kCPU)
                      .to(torch::kFloat32)
                      .contiguous();
    }

    if (!pts_cpu.defined() || pts_cpu.numel() == 0 ||
        pts_cpu.dim() != 2 || pts_cpu.size(1) != 3) {
        return false;
    }

    auto bound = mainSceneBoundPcdHeuristicCpp(
        pts_cpu.contiguous(),
        dense_core_pcd_density_rate_);
    if (!bound) {
        return false;
    }

    const torch::Device dev =
        center_.defined() ? center_.device() :
        (scene_center_.defined() ? scene_center_.device() : torch::Device(torch::kCUDA));
    auto core_center =
        bound->first.to(dev).to(torch::kFloat32).contiguous().view({3});
    const float radius = bound->second;
    if (!std::isfinite(radius) || radius <= 0.0f) {
        return false;
    }

    auto core_radius =
        torch::full({3}, radius, torch::dtype(torch::kFloat32).device(dev));
    dense_core_bb_min_ = (core_center - core_radius).contiguous();
    dense_core_bb_max_ = (core_center + core_radius).contiguous();
    has_dense_core_bb_ = true;
    return true;
}

} // namespace sv
