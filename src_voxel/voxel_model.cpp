#include "include_voxel/voxel_model.h"
#include "src/adam_step.h"
#include "src/tv_compute.h"
#include "src/utils.h"
#include <ATen/ops/isin.h>
#include <ATen/ops/searchsorted.h>
#include <ATen/ops/unique_dim.h>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <array>
#include <cmath>
#include <limits>
#include <optional>

namespace sv {

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
    torch::Tensor vox_ijk = UTILS::octpath_2_ijk(op, lv);
    torch::Tensor vox_center = scene_min_xyz + (vox_ijk.to(torch::kFloat32) + 0.5f) * vox_size;
    return {vox_center.contiguous(), vox_size.contiguous()};
}

std::tuple<torch::Tensor, torch::Tensor> uniqueRowsWithInverse(const torch::Tensor& rows);

std::pair<torch::Tensor, torch::Tensor> buildGridPtsLink(
    const torch::Tensor& octpath,
    const torch::Tensor& octlevel,
    int max_num_levels)
{
    torch::Tensor op = octpath.reshape({-1, 1}).contiguous();
    torch::Tensor lv = octlevel.reshape({-1, 1}).contiguous();
    torch::Tensor vox_ijk = UTILS::octpath_2_ijk(op, lv).to(torch::kLong).contiguous();
    torch::Tensor lv2max = (max_num_levels - lv.to(torch::kLong)).contiguous();
    torch::Tensor scale = torch::pow(
        torch::full_like(lv2max.to(torch::kFloat32), 2.0f),
        lv2max.to(torch::kFloat32)).to(torch::kLong).contiguous();

    torch::Tensor base_grid_ijk = (vox_ijk * scale).view({-1, 1, 3}).contiguous();
    torch::Tensor subtree_shift = torch::tensor(
        {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
         {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}},
        torch::TensorOptions().dtype(torch::kLong).device(op.device()));
    torch::Tensor gridpts =
        base_grid_ijk + subtree_shift.view({1, 8, 3}) * scale.view({-1, 1, 1});

    auto [grid_pts_key, vox_key_flat] = uniqueRowsWithInverse(gridpts.reshape({-1, 3}));
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
    out.index_reduce_(
        /*dim=*/0,
        /*index=*/vox_key.to(vox_val.device()).flatten(),
        /*source=*/vox_val.flatten(0, 1).to(torch::kFloat32),
        /*reduce=*/"mean",
        /*include_self=*/false);
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

int VoxelModel::maxNumLevels() const {
    return max_num_levels_;
}

torch::Tensor VoxelModel::SceneCenter() const {
    return this->scene_center_;
}

torch::Tensor VoxelModel::SceneExtent() const {
    return this->scene_extent_;
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

    const int64_t rows = end - begin;
    auto make_default = [&]() {
        return torch::full(
                   {rows, 1},
                   default_value,
                   torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(grid_pts_key_new.device()))
            .contiguous()
            .detach()
            .requires_grad_();
    };

    if (rows == 0 || !geo_grid_init_callback_) {
        return make_default();
    }

    try {
        torch::Tensor key_add =
            grid_pts_key_new.slice(/*dim=*/0, begin, end).contiguous();
        auto dev = key_add.device();
        torch::Tensor scene_center =
            scene_center_.to(dev).to(torch::kFloat32).contiguous().view({3});
        torch::Tensor scene_extent =
            scene_extent_.to(dev).to(torch::kFloat32).contiguous().view({1});
        torch::Tensor scene_min = scene_center - 0.5f * scene_extent;

        const float finest_scale = std::ldexp(1.0f, -max_num_levels_);
        torch::Tensor finest_vox = scene_extent * finest_scale;
        torch::Tensor grid_xyz =
            scene_min.view({1, 3}) +
            key_add.to(torch::kFloat32) * finest_vox.view({1, 1});
        grid_xyz = grid_xyz.contiguous();

        torch::Tensor init = geo_grid_init_callback_(grid_xyz, fixed_vox_size_);
        if (!init.defined() || init.numel() != rows) {
            return make_default();
        }
        if (init.dim() == 1) {
            init = init.view({rows, 1});
        } else {
            init = init.reshape({rows, 1});
        }
        return init.to(grid_pts_key_new.device())
            .to(torch::kFloat32)
            .contiguous()
            .detach()
            .requires_grad_();
    } catch (const std::exception& e) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::cout << "[TSDF DENSITY INIT] callback failed; using default geo init: "
                      << e.what() << "\n";
        }
        return make_default();
    }
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

namespace fs = std::filesystem;

void VoxelModel::createFromPcd(
    const std::map<point3D_id_t, Point3D>& pcd,
    const std::vector<sv::MiniCam>& cams)
{
    std::cout << "VoxelModel::createFromPcd() called with " << pcd.size() << " points.\n";
    fill_empty_cells_done_ = false;
    fill_empty_cells_warmup_notified_ = false;
    last_artificial_iter_ = -1;
    art_key_before_iter_ = torch::empty(
        {0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

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

    // ------------------------------------------------------------------------
    // 0) Fixed global scene + desired voxel size
    // ------------------------------------------------------------------------
    auto dev = torch::kCUDA;
    artificial_centers_accum_viz_ = torch::empty(
        {0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    artificial_sizes_accum_viz_ = torch::empty(
        {0, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    scene_center_ = torch::tensor(
        {global_scene_center_[0], global_scene_center_[1], global_scene_center_[2]},
        torch::dtype(torch::kFloat32).device(dev)
    ).contiguous();                                                  // [3]
    // scene_extent_ = torch::tensor({global_scene_extent_},
    //     torch::dtype(torch::kFloat32).device(dev)
    // ).contiguous();   
    
    int   outside_level = 0;
    const float scene_extent_scalar = global_scene_extent_ * std::pow(2.0f, outside_level);
    scene_extent_ = torch::tensor({scene_extent_scalar},
        torch::dtype(torch::kFloat32).device(dev)).contiguous();
    inside_extent_ = torch::tensor({global_scene_extent_},
        torch::dtype(torch::kFloat32).device(dev)
    ).contiguous();    
    scene_min_t_  = (scene_center_ - 0.5f * scene_extent_).contiguous(); // [3]

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

    // Reset accumulated real-point history. We seed it later from actually
    // inserted/filtered real voxels (not raw pre-filter PCD).
    real_pcd_points_accum_cpu_ = torch::empty(
        {0, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    // ------------------------------------------------------------------------
    // 0.5) Initialize global PCD bounds (for later gating of fill_empty_cells_)
    // ------------------------------------------------------------------------
    {
        // Work on CPU for simplicity; this runs only once at initialization.
        auto xyz_cpu = xyz.to(torch::kCPU).contiguous();  // [N,3]

        auto min_res = xyz_cpu.min(/*dim=*/0, /*keepdim=*/false);
        auto max_res = xyz_cpu.max(/*dim=*/0, /*keepdim=*/false);

        torch::Tensor min_cpu = std::get<0>(min_res).contiguous();   // [3]
        torch::Tensor max_cpu = std::get<0>(max_res).contiguous();   // [3]

        // Store as CUDA tensors so they match the rest of the model state
        global_pcd_min_ = min_cpu.to(dev).contiguous();              // [3]
        global_pcd_max_ = max_cpu.to(dev).contiguous();              // [3]
        has_global_pcd_bb_ = true;

        std::cout << "[createFromPcd] global_pcd_min=" << global_pcd_min_
                  << " global_pcd_max=" << global_pcd_max_ << std::endl;
    }
    // ------------------------------------------------------------------------
    // 1) Compute octlevel from vox_size (mirror points_init behavior:
    //    round/clamp).
    // ------------------------------------------------------------------------
    // Dense-core estimation is deferred until real-point history is seeded
    // near the end of createFromPcd().
    has_dense_core_bb_ = false;

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
    // rgb_u.index_add_(0, invmap, rgb);
    // auto counts = torch::bincount(invmap, {}, Nu).view({Nu,1}).clamp_min(1);
    // rgb_u = rgb_u / counts;
    rgb_u.index_reduce_(
        /*dim=*/0,
        /*index=*/invmap,
        /*source=*/rgb,
        /*reduce=*/"mean",
        /*include_self=*/false
    );

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
    auto octpath = UTILS::ijk_2_octpath(ijk_u.contiguous(), L_u.contiguous()).contiguous(); // [Nu,1] int64

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
        at::Tensor rate = markSvrasterMaxSampRateDirect(cams, octpath, vox_center, vox_size);
        if (rate.dim() == 2 && rate.size(1) == 1) {
            rate = rate.squeeze(1);
        }
        rate = rate.to(torch::kFloat32);
        at::Tensor kept = (rate > 0.0f);

        if (filter_near_voxels_) {
            const float near_thresh = 0.2f;
            at::Tensor is_near = markSvrasterNearDirect(cams, octpath, vox_center, vox_size, near_thresh);
            if (is_near.dim() == 2 && is_near.size(1) == 1) {
                is_near = is_near.squeeze(1);
            }
            is_near = is_near.to(torch::kBool);
            kept = kept & (~is_near);
        }
        kept = kept.view({-1}).to(torch::kBool);
        auto idx = torch::nonzero(kept).view({-1});
        const int64_t K = idx.size(0);

        if (K > 0 && K < octpath.size(0)) {
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
        /*default_value=*/-10.0f);

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
    // stats buffer
    this->max_w_ = torch::zeros({center_.size(0), 1}, torch::dtype(torch::kFloat32).device(dev));
    
    // Initial map from PCD is treated as real geometry.
    this->is_artificial_voxel_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    // Initial topology is created directly from ORB-SLAM map points.
    this->is_orb_voxel_ = torch::ones(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    this->is_inactive_geo_voxel_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    this->is_rgbd_fill_render_holes_voxel_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    this->is_depthanything_fill_holes_voxel_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    this->is_promoted_artificial_voxel_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    this->total_promoted_artificial_voxels_ = 0;
    this->exist_since_iter_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kInt32).device(dev));
    this->exist_since_kf_ = torch::full(
        {center_.size(0)},
        current_kf_count,
        torch::TensorOptions().dtype(torch::kInt32).device(dev));
    this->geometrically_unstable_voxel_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    this->rendered_depth_candidate_voxel_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    this->rendered_depth_candidate_support_count_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kInt32).device(dev));
    this->rendered_depth_candidate_last_seen_kf_ = torch::full(
        {center_.size(0)},
        static_cast<int32_t>(-1),
        torch::TensorOptions().dtype(torch::kInt32).device(dev));
    this->rendered_depth_candidate_source_kind_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kInt32).device(dev));

    // Seed dense-core history from raw createFromPcd points (CPU).
    // This matches heuristic expectations better than regularized voxel centers.
    real_pcd_points_accum_cpu_ = xyz.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (max_real_pcd_points_ > 0 && real_pcd_points_accum_cpu_.size(0) > max_real_pcd_points_) {
        const int64_t total = real_pcd_points_accum_cpu_.size(0);
        auto idx = torch::linspace(
            0.0, static_cast<double>(total - 1), max_real_pcd_points_,
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).round().to(torch::kLong);
        real_pcd_points_accum_cpu_ = real_pcd_points_accum_cpu_.index_select(0, idx).contiguous();
    }

    // Compute dense-core once at create-time so insertion-time far filtering
    // can use it immediately if that option is enabled later.
    const bool refreshed_dense_core = refreshDenseCoreBBFromCurrentVoxels();
    if (!refreshed_dense_core || !has_dense_core_bb_) {
        has_dense_core_bb_ = false;
    }
    // Register with your optimizer
    VOXEL_MODEL_TENSORS_TO_VEC
}

void VoxelModel::increasePcd(
    std::vector<float> pcd_full,
    std::vector<float> colors,
    const int iteration,
    const std::vector<sv::MiniCam>& cams)
{
    const int Nf = static_cast<int>(pcd_full.size());
    last_increase_pcd_stats_ = IncreasePcdStats{};
    if (Nf < 3 || colors.size() < 3) return;
    int N = Nf / 3;
    const int64_t raw_points_in = N;
    last_increase_pcd_stats_.raw_points_in = raw_points_in;
    const int32_t current_kf_count =
        cams.empty() ? static_cast<int32_t>(-1) : static_cast<int32_t>(cams.size());
    const bool insert_rendered_depth_candidate = pending_insert_rendered_depth_candidate_;
    const bool insert_rendered_depth_candidate_as_real_protected =
        pending_insert_rendered_depth_candidate_as_real_protected_;
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
        // A) Compute scene from this batch using main_scene_bound_pcd_heuristic
        {
            bool updated_scene_from_cpp = false;
            if (auto bound = mainSceneBoundPcdHeuristicCpp(
                    xyz_cpu.contiguous(),
                    dense_core_pcd_density_rate_)) {
                auto center = bound->first.to(torch::kCPU).to(torch::kFloat32).contiguous();
                const float radius = bound->second;
                global_scene_center_[0] = center.index({0}).item<float>();
                global_scene_center_[1] = center.index({1}).item<float>();
                global_scene_center_[2] = center.index({2}).item<float>();
                global_scene_extent_ = 2.0f * radius;
                updated_scene_from_cpp = true;
            }

            if (!updated_scene_from_cpp) {
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
        return;
    }

    auto rgb_u = torch::zeros({Nu,3}, torch::dtype(torch::kFloat32).device(dev));
    rgb_u.index_reduce_(
        /*dim=*/0,
        /*index=*/invmap,
        /*source=*/rgb,
        /*reduce=*/"mean",
        /*include_self=*/false
    );

    // ── 4) Build octpath for this batch ─────────────────────────────────────
    auto octpath_new = UTILS::ijk_2_octpath(ijk_u.contiguous(), L_u.contiguous()).contiguous();          // [Nu,1] int64

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
        at::Tensor rate = markSvrasterMaxSampRateDirect(cams, octpath_new, vox_center, vox_size);
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
                markSvrasterNearDirect(cams, octpath_new, vox_center, vox_size, near_thresh);
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

    if (insert_rendered_depth_candidate &&
        rendered_depth_candidate_require_real_adjacency_ &&
        octpath_new.size(0) > 0) {
        auto unique_sorted_1d = [](const torch::Tensor& t) -> torch::Tensor {
            TORCH_CHECK(t.dim() == 1, "unique_sorted_1d expects a 1-D tensor");
            if (t.numel() <= 1) {
                return t.contiguous();
            }
            auto sort_pair = t.sort(/*dim=*/0);
            auto sorted = std::get<0>(sort_pair).contiguous();
            auto keep = torch::empty_like(sorted, torch::kBool);
            keep.index_put_({0}, true);
            auto neq =
                sorted.index({torch::indexing::Slice(1, torch::indexing::None)}) !=
                sorted.index({torch::indexing::Slice(torch::indexing::None, -1)});
            keep.index_put_({torch::indexing::Slice(1, torch::indexing::None)}, neq);
            auto keep_idx = torch::nonzero(keep).view({-1});
            return sorted.index_select(0, keep_idx).contiguous();
        };

        auto octpath_old_adj = this->oct_path_.contiguous();
        auto octlevel_old_adj = this->oct_level_.contiguous();
        if (octpath_old_adj.numel() == 0) {
            std::cout << "[increasePcd/real_adjacency] no existing topology, nothing to attach to.\n";
            pending_insert_rendered_depth_candidate_ = false;
            pending_insert_rendered_depth_candidate_source_kind_ = 0;
            pending_insert_rendered_depth_candidate_as_real_protected_ = false;
            pending_artificial_insert_rr_entity_path_.clear();
            return;
        }

        auto bool_opts_old_adj = torch::TensorOptions().dtype(torch::kBool).device(dev);
        if (!is_artificial_voxel_.defined() || is_artificial_voxel_.size(0) != octpath_old_adj.size(0)) {
            is_artificial_voxel_ = torch::zeros({octpath_old_adj.size(0)}, bool_opts_old_adj);
        } else if (is_artificial_voxel_.device() != dev) {
            is_artificial_voxel_ = is_artificial_voxel_.to(dev);
        }

        auto real_mask_adj = (~is_artificial_voxel_.to(dev).to(torch::kBool)).contiguous();
        auto real_idx_adj = torch::nonzero(real_mask_adj).view({-1});
        if (real_idx_adj.numel() == 0) {
            std::cout << "[increasePcd/real_adjacency] no real voxels available for attachment.\n";
            pending_insert_rendered_depth_candidate_ = false;
            pending_insert_rendered_depth_candidate_source_kind_ = 0;
            pending_insert_rendered_depth_candidate_as_real_protected_ = false;
            pending_artificial_insert_rr_entity_path_.clear();
            return;
        }

        const int base_L = static_cast<int>(L_u[0].item<int8_t>());
        TORCH_CHECK(base_L >= 1 && base_L <= max_num_levels_,
                    "[increasePcd/real_adjacency] base level out of range: ", base_L);

        auto real_path_adj = octpath_old_adj.index_select(0, real_idx_adj).contiguous();
        auto real_level_adj = octlevel_old_adj.index_select(0, real_idx_adj)
                                  .view({-1})
                                  .to(torch::kInt64)
                                  .contiguous();
        auto valid_real_level_mask = (real_level_adj >= static_cast<int64_t>(base_L));
        auto valid_real_level_idx = torch::nonzero(valid_real_level_mask).view({-1});
        if (valid_real_level_idx.numel() == 0) {
            std::cout << "[increasePcd/real_adjacency] no real voxels at or finer than candidate level.\n";
            pending_insert_rendered_depth_candidate_ = false;
            pending_insert_rendered_depth_candidate_source_kind_ = 0;
            pending_insert_rendered_depth_candidate_as_real_protected_ = false;
            pending_artificial_insert_rr_entity_path_.clear();
            return;
        }
        real_path_adj = real_path_adj.index_select(0, valid_real_level_idx).contiguous();

        const int levels_below = std::max(0, max_num_levels_ - base_L);
        const int bits_to_clear = 3 * levels_below;
        const int64_t lower_mask = (bits_to_clear > 0) ? ((1LL << bits_to_clear) - 1LL) : 0LL;
        const int64_t keep_mask_ll = ~lower_mask;
        auto keep_mask = torch::full(
            {1},
            keep_mask_ll,
            torch::TensorOptions().dtype(torch::kInt64).device(dev));

        auto real_base_path =
            (real_path_adj.view({-1}).to(torch::kInt64) & keep_mask).contiguous();
        real_base_path = unique_sorted_1d(real_base_path);
        auto real_base_key =
            real_base_path.mul(256).add(
                torch::full_like(real_base_path, static_cast<int64_t>(base_L)));
        real_base_key = unique_sorted_1d(real_base_key);
        if (real_base_key.numel() == 0) {
            std::cout << "[increasePcd/real_adjacency] no real base cells available after ancestor projection.\n";
            pending_insert_rendered_depth_candidate_ = false;
            pending_insert_rendered_depth_candidate_source_kind_ = 0;
            pending_insert_rendered_depth_candidate_as_real_protected_ = false;
            pending_artificial_insert_rr_entity_path_.clear();
            return;
        }

        real_base_path = real_base_path.view({-1, 1}).contiguous();
        auto L_real_base = torch::full(
            {real_base_path.size(0), 1},
            static_cast<int64_t>(base_L),
            torch::TensorOptions().dtype(torch::kInt8).device(dev));
        auto real_base_ijk =
            UTILS::octpath_2_ijk(real_base_path, L_real_base).to(torch::kLong).contiguous();

        std::vector<int64_t> shift_vals;
        const int adj_radius = std::max(1, rendered_depth_candidate_adjacency_radius_cells_);
        for (int dx = -adj_radius; dx <= adj_radius; ++dx) {
            for (int dy = -adj_radius; dy <= adj_radius; ++dy) {
                for (int dz = -adj_radius; dz <= adj_radius; ++dz) {
                    if (dx == 0 && dy == 0 && dz == 0) {
                        continue;
                    }
                    shift_vals.push_back(static_cast<int64_t>(dx));
                    shift_vals.push_back(static_cast<int64_t>(dy));
                    shift_vals.push_back(static_cast<int64_t>(dz));
                }
            }
        }
        auto side_shift = torch::from_blob(
            shift_vals.data(),
            {static_cast<int64_t>(shift_vals.size() / 3), 3},
            torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
                              .clone()
                              .to(dev)
                              .contiguous();

        auto adj_ijk =
            (real_base_ijk.unsqueeze(1) + side_shift.unsqueeze(0)).contiguous().view({-1, 3});
        const int64_t grid_limit = (1LL << base_L);
        auto in_bounds =
            (adj_ijk >= 0).all(1) &
            (adj_ijk < grid_limit).all(1);
        auto adj_keep_idx = torch::nonzero(in_bounds).view({-1});
        if (adj_keep_idx.numel() == 0) {
            std::cout << "[increasePcd/real_adjacency] no in-bounds adjacent cells available.\n";
            pending_insert_rendered_depth_candidate_ = false;
            pending_insert_rendered_depth_candidate_source_kind_ = 0;
            pending_insert_rendered_depth_candidate_as_real_protected_ = false;
            pending_artificial_insert_rr_entity_path_.clear();
            return;
        }
        adj_ijk = adj_ijk.index_select(0, adj_keep_idx).contiguous();

        adj_ijk = std::get<0>(uniqueRowsWithInverse(adj_ijk)).contiguous();
        if (adj_ijk.dim() == 1) {
            adj_ijk = adj_ijk.view({-1, 3});
        }

        auto L_adj = torch::full(
            {adj_ijk.size(0), 1},
            static_cast<int64_t>(base_L),
            torch::TensorOptions().dtype(torch::kInt8).device(dev));
        auto adj_path = UTILS::ijk_2_octpath(adj_ijk, L_adj).contiguous();
        auto adj_key = adj_path.view({-1}).to(torch::kInt64).mul(256).add(L_adj.view({-1}).to(torch::kInt64));
        adj_key = unique_sorted_1d(adj_key);

        auto cand_key =
            octpath_new.view({-1}).to(torch::kInt64).mul(256).add(L_u.view({-1}).to(torch::kInt64));
        auto keep_adjacent = at::isin(cand_key, adj_key).to(torch::kBool).contiguous();
        const int64_t kept_adjacent = keep_adjacent.sum().item<int64_t>();
        std::cout << "[increasePcd/real_adjacency] kept " << kept_adjacent
                  << "/" << octpath_new.size(0)
                  << " rendered-depth candidates"
                  << " real_base_cells=" << real_base_path.size(0)
                  << " radius_cells=" << adj_radius
                  << "\n";
        if (kept_adjacent == 0) {
            std::cout << "[increasePcd/filter] rendered-depth candidates rejected by real-adjacency gate.\n";
            pending_insert_rendered_depth_candidate_ = false;
            pending_insert_rendered_depth_candidate_source_kind_ = 0;
            pending_insert_rendered_depth_candidate_as_real_protected_ = false;
            pending_artificial_insert_rr_entity_path_.clear();
            return;
        }

        auto keep_adjacent_idx = torch::nonzero(keep_adjacent).view({-1});
        octpath_new = octpath_new.index_select(0, keep_adjacent_idx).contiguous();
        L_u = L_u.index_select(0, keep_adjacent_idx).contiguous();
        ijk_u = ijk_u.index_select(0, keep_adjacent_idx).contiguous();
        rgb_u = rgb_u.index_select(0, keep_adjacent_idx).contiguous();
        Nu = octpath_new.size(0);
    }

    // ── 5) Dedup against existing voxels (across-batch) ─────────────────────
    auto octpath_old  = this->oct_path_.contiguous();                    // [No,1] int64
    auto octlevel_old = this->oct_level_.contiguous();                   // [No,1] int8
    const int64_t old_voxel_count = octpath_old.size(0);
    ensureSvrasterSdfField();
    torch::Tensor old_vox_sdf_values = voxelCornerScalarFromGrid_(svraster_sdf_grid_pts_);
    torch::Tensor old_vox_sdf_weights = voxelCornerScalarFromGrid_(svraster_sdf_weights_);

    // Packed 1D key: (octpath<<8) | level
    auto key_new = octpath_new.view({-1}).to(torch::kInt64)
                    .mul(256)
                    .add(L_u.view({-1}).to(torch::kInt64));
    auto key_old_all = octpath_old.view({-1}).to(torch::kInt64)
                    .mul(256)
                    .add(octlevel_old.view({-1}).to(torch::kInt64));
    auto bool_opts_old = torch::TensorOptions().dtype(torch::kBool).device(dev);
    auto i32_opts_old = torch::TensorOptions().dtype(torch::kInt32).device(dev);
    if (!rendered_depth_candidate_voxel_.defined() ||
        rendered_depth_candidate_voxel_.size(0) != octpath_old.size(0)) {
        rendered_depth_candidate_voxel_ = torch::zeros({octpath_old.size(0)}, bool_opts_old);
    } else if (rendered_depth_candidate_voxel_.device() != dev) {
        rendered_depth_candidate_voxel_ = rendered_depth_candidate_voxel_.to(dev);
    }
    if (!rendered_depth_candidate_support_count_.defined() ||
        rendered_depth_candidate_support_count_.size(0) != octpath_old.size(0)) {
        rendered_depth_candidate_support_count_ = torch::zeros({octpath_old.size(0)}, i32_opts_old);
    } else if (rendered_depth_candidate_support_count_.device() != dev) {
        rendered_depth_candidate_support_count_ = rendered_depth_candidate_support_count_.to(dev);
    }
    if (!rendered_depth_candidate_last_seen_kf_.defined() ||
        rendered_depth_candidate_last_seen_kf_.size(0) != octpath_old.size(0)) {
        rendered_depth_candidate_last_seen_kf_ =
            torch::full({octpath_old.size(0)}, static_cast<int32_t>(-1), i32_opts_old);
    } else if (rendered_depth_candidate_last_seen_kf_.device() != dev) {
        rendered_depth_candidate_last_seen_kf_ = rendered_depth_candidate_last_seen_kf_.to(dev);
    }
    if (!rendered_depth_candidate_source_kind_.defined() ||
        rendered_depth_candidate_source_kind_.size(0) != octpath_old.size(0)) {
        rendered_depth_candidate_source_kind_ = torch::zeros({octpath_old.size(0)}, i32_opts_old);
    } else if (rendered_depth_candidate_source_kind_.device() != dev) {
        rendered_depth_candidate_source_kind_ = rendered_depth_candidate_source_kind_.to(dev);
    }
    torch::Tensor art_key_before = torch::empty({0}, key_old_all.options());
    if (octpath_old.numel() > 0 &&
        is_artificial_voxel_.defined() && is_artificial_voxel_.size(0) == octpath_old.size(0)) {
        auto art_before_mask = is_artificial_voxel_.to(dev).to(torch::kBool).contiguous();
        if (is_promoted_artificial_voxel_.defined() &&
            is_promoted_artificial_voxel_.size(0) == octpath_old.size(0)) {
            auto promoted_before_mask = is_promoted_artificial_voxel_.to(dev).to(torch::kBool).contiguous();
            art_before_mask = art_before_mask & (~promoted_before_mask);
        }
        auto idx_art_before = torch::nonzero(art_before_mask).view({-1});
        if (idx_art_before.numel() > 0) {
            art_key_before = key_old_all.index_select(0, idx_art_before).contiguous();
        }
    }
    if (last_artificial_iter_ != iteration) {
        last_artificial_iter_ = iteration;
        art_key_before_iter_ = art_key_before.defined()
            ? art_key_before.clone()
            : torch::empty({0}, key_old_all.options());
    } else if (!art_key_before_iter_.defined()) {
        art_key_before_iter_ = torch::empty({0}, key_old_all.options());
    }

    int64_t promoted_artificials_count = 0;
    auto long_opts_old = torch::TensorOptions().dtype(torch::kLong).device(dev);
    torch::Tensor promote_idx_deferred = torch::empty({0}, long_opts_old);
    torch::Tensor support_idx_deferred = torch::empty({0}, long_opts_old);
    torch::Tensor old_art_mask_for_promotion = torch::zeros({octpath_old.size(0)}, bool_opts_old);
    auto old_rendered_depth_candidate_mask =
        rendered_depth_candidate_voxel_.to(dev).to(torch::kBool).contiguous();
    if (octpath_old.numel() > 0) {
        if (!is_artificial_voxel_.defined() || is_artificial_voxel_.size(0) != octpath_old.size(0)) {
            is_artificial_voxel_ = torch::zeros({octpath_old.size(0)}, bool_opts_old);
        } else if (is_artificial_voxel_.device() != dev) {
            is_artificial_voxel_ = is_artificial_voxel_.to(dev);
        }
        old_art_mask_for_promotion = is_artificial_voxel_.to(torch::kBool).contiguous();
        if (is_promoted_artificial_voxel_.defined() &&
            is_promoted_artificial_voxel_.size(0) == octpath_old.size(0)) {
            auto promoted_before_mask = is_promoted_artificial_voxel_.to(dev).to(torch::kBool).contiguous();
            old_art_mask_for_promotion = old_art_mask_for_promotion & (~promoted_before_mask);
        }
        // Promote artificials regardless of when they were created:
        // only require "currently artificial and not already promoted".
        if (!enable_artificial_promotion_) {
            old_art_mask_for_promotion = torch::zeros_like(old_art_mask_for_promotion);
        }
        if (insert_rendered_depth_candidate) {
            old_art_mask_for_promotion = torch::zeros_like(old_art_mask_for_promotion);
        }
    }
    torch::Tensor new_mask;
    if (octpath_old.numel() == 0) {
        std::cout << "octpath_old is empty => all new.\n";
        new_mask = torch::ones({Nu}, torch::dtype(torch::kBool).device(dev));
    } else {
        auto is_dup = at::isin(key_new, key_old_all).to(torch::kBool);                                    // [Nu]
        new_mask = ~is_dup;

        // If a real observation hits an existing artificial voxel, defer promotion to
        // after real insertion in this call. For rendered-depth candidate insertion,
        // duplicates instead add support to existing candidates.
        if (is_dup.any().item<bool>()) {
            auto dup_idx_new = torch::nonzero(is_dup).view({-1});
            if (dup_idx_new.numel() > 0) {
                auto dup_key_new = key_new.index_select(0, dup_idx_new).contiguous();

                auto sort_pair = key_old_all.sort(/*dim=*/0);
                auto key_old_sorted = std::get<0>(sort_pair).contiguous();
                auto old_perm = std::get<1>(sort_pair).to(torch::kLong).contiguous();

                auto pos = at::searchsorted(
                        key_old_sorted,
                        dup_key_new,
                        /*out_int32=*/false,
                        /*right=*/false)
                    .to(torch::kLong).contiguous();

                auto in_range = (pos >= 0) & (pos < key_old_sorted.size(0));
                if (in_range.any().item<bool>()) {
                    auto in_idx = torch::nonzero(in_range).view({-1});
                    auto pos_in = pos.index_select(0, in_idx).contiguous();
                    auto key_in = dup_key_new.index_select(0, in_idx).contiguous();
                    auto key_hit = key_old_sorted.index_select(0, pos_in).contiguous();
                    auto exact = (key_hit == key_in);

                    if (exact.any().item<bool>()) {
                        auto ex_idx = torch::nonzero(exact).view({-1});
                        auto pos_ex = pos_in.index_select(0, ex_idx).contiguous();
                        auto old_idx = old_perm.index_select(0, pos_ex).contiguous();
                        if (insert_rendered_depth_candidate) {
                            auto can_support = old_rendered_depth_candidate_mask
                                .index_select(0, old_idx).to(torch::kBool).contiguous();
                            if (can_support.any().item<bool>()) {
                                auto support_rel_idx = torch::nonzero(can_support).view({-1});
                                auto support_idx = old_idx.index_select(0, support_rel_idx).contiguous();
                                support_idx_deferred = torch::cat(
                                    {support_idx_deferred, support_idx.to(torch::kLong)}, 0).contiguous();
                            }
                        } else {
                            auto can_promote = old_art_mask_for_promotion
                                .index_select(0, old_idx).to(torch::kBool).contiguous();
                            if (can_promote.any().item<bool>()) {
                                auto art_idx = torch::nonzero(can_promote).view({-1});
                                auto promote_idx = old_idx.index_select(0, art_idx).contiguous();
                                promote_idx_deferred = torch::cat(
                                    {promote_idx_deferred, promote_idx.to(torch::kLong)}, 0).contiguous();
                            }
                        }
                    }
                }
            }
        }
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

                // Promotion path:
                // Real observations arriving at parent level should promote overlapped child artificials to real,
                // otherwise artificial provenance keeps expanding after subdivision.
                // Rendered-depth candidate insertion uses the same overlap detection to
                // add support to subdivided descendants instead of promoting them.
                if (would_collide_parent.any().item<bool>()) {
                    auto collide_idx_new = torch::nonzero(would_collide_parent).view({-1});
                    auto collide_parent_keys = key_new.index_select(0, collide_idx_new).contiguous();
                    collide_parent_keys = unique_sorted_1d(collide_parent_keys);

                    auto key_child_rows = op_anc_base.mul(256)
                        .add(torch::full_like(op_anc_base, static_cast<int64_t>(base_L)));
                    auto child_under_collide =
                        at::isin(key_child_rows, collide_parent_keys).to(torch::kBool);

                    if (child_under_collide.any().item<bool>()) {
                        auto child_rel_idx = torch::nonzero(child_under_collide).view({-1});
                        auto old_idx_under = sel.index_select(0, child_rel_idx).contiguous();
                        if (insert_rendered_depth_candidate) {
                            auto can_support = old_rendered_depth_candidate_mask
                                .index_select(0, old_idx_under).to(torch::kBool).contiguous();
                            if (can_support.any().item<bool>()) {
                                auto support_rel_idx = torch::nonzero(can_support).view({-1});
                                auto support_idx = old_idx_under.index_select(0, support_rel_idx).contiguous();
                                support_idx_deferred = torch::cat(
                                    {support_idx_deferred, support_idx.to(torch::kLong)}, 0).contiguous();
                            }
                        } else {
                            auto can_promote = old_art_mask_for_promotion
                                .index_select(0, old_idx_under).to(torch::kBool).contiguous();
                            if (can_promote.any().item<bool>()) {
                                auto art_rel_idx = torch::nonzero(can_promote).view({-1});
                                auto promote_idx = old_idx_under.index_select(0, art_rel_idx).contiguous();
                                promote_idx_deferred = torch::cat(
                                    {promote_idx_deferred, promote_idx.to(torch::kLong)}, 0).contiguous();
                            }
                        }
                    }
                    new_mask = new_mask & (~would_collide_parent);
                }
            }
        }
    }

    if (promote_idx_deferred.numel() > 1) {
        auto sorted = std::get<0>(promote_idx_deferred.sort(/*dim=*/0));
        auto keep = torch::empty_like(sorted, torch::kBool);
        keep.index_put_({0}, true);
        auto neq = sorted.index({torch::indexing::Slice(1, torch::indexing::None)})
                != sorted.index({torch::indexing::Slice(torch::indexing::None, -1)});
        keep.index_put_({torch::indexing::Slice(1, torch::indexing::None)}, neq);
        auto keep_idx = torch::nonzero(keep).view({-1});
        promote_idx_deferred = sorted.index_select(0, keep_idx).contiguous();
    }
    if (support_idx_deferred.numel() > 1) {
        auto sorted = std::get<0>(support_idx_deferred.sort(/*dim=*/0));
        auto keep = torch::empty_like(sorted, torch::kBool);
        keep.index_put_({0}, true);
        auto neq = sorted.index({torch::indexing::Slice(1, torch::indexing::None)})
                != sorted.index({torch::indexing::Slice(torch::indexing::None, -1)});
        keep.index_put_({torch::indexing::Slice(1, torch::indexing::None)}, neq);
        auto keep_idx = torch::nonzero(keep).view({-1});
        support_idx_deferred = sorted.index_select(0, keep_idx).contiguous();
    }
    const int64_t pending_promotions = promote_idx_deferred.numel();
    const int64_t pending_support_updates = support_idx_deferred.numel();
    const int64_t new_voxel_candidates = new_mask.sum().item<int64_t>();
    const int64_t duplicate_existing_voxels =
        unique_voxel_candidates_after_insert_filter - new_voxel_candidates;
    last_increase_pcd_stats_.duplicate_existing_voxels = duplicate_existing_voxels;
    last_increase_pcd_stats_.new_voxels = new_voxel_candidates;
    last_increase_pcd_stats_.pending_promotions = pending_promotions;
    last_increase_pcd_stats_.pending_support_updates = pending_support_updates;
    if (!new_mask.any().item<bool>()) {
        if (pending_promotions == 0 && pending_support_updates == 0) {
            std::cout << "[increasePcd] No new voxels (all duplicates). Nothing appended.\n";
            pending_insert_rendered_depth_candidate_ = false;
            pending_insert_rendered_depth_candidate_source_kind_ = 0;
            pending_insert_rendered_depth_candidate_as_real_protected_ = false;
            pending_artificial_insert_rr_entity_path_.clear();
            return;
        }
        std::cout << "[increasePcd] No new voxels, continuing after deferred updates.\n";
    }
    auto sel = torch::nonzero(new_mask).view({-1});                                                     // [Nk]
    auto ijk_add     = ijk_u.index_select(0, sel);                                                      // [Nk,3]
    auto octpath_add = octpath_new.index_select(0, sel);                                                // [Nk,1]
    auto L_add       = L_u.index_select(0, sel);                                                         // [Nk,1]
    auto rgb_add     = rgb_u.index_select(0, sel);                                                       // [Nk,3]
    const int Nk = sel.size(0);

    int64_t Nm_added = 0;  // count of artificial voxels added later
    // ── 6) Append topology (old preserved) ──────────────────────────────────
    this->oct_path_ = torch::cat({octpath_old,  octpath_add}, 0).contiguous();
    this->oct_level_ = torch::cat({octlevel_old, L_add}, 0).contiguous();
    {
        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto i32_opts  = torch::TensorOptions().dtype(torch::kInt32).device(dev);
        if (!is_artificial_voxel_.defined() || is_artificial_voxel_.size(0) != octpath_old.size(0)) {
            is_artificial_voxel_ = torch::zeros({octpath_old.size(0)}, bool_opts);
        } else if (is_artificial_voxel_.device() != octpath_old.device()) {
            is_artificial_voxel_ = is_artificial_voxel_.to(octpath_old.device());
        }
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
        if (!is_depthanything_fill_holes_voxel_.defined() ||
            is_depthanything_fill_holes_voxel_.size(0) != octpath_old.size(0)) {
            is_depthanything_fill_holes_voxel_ = torch::zeros({octpath_old.size(0)}, bool_opts);
        } else if (is_depthanything_fill_holes_voxel_.device() != octpath_old.device()) {
            is_depthanything_fill_holes_voxel_ =
                is_depthanything_fill_holes_voxel_.to(octpath_old.device());
        }
        if (!is_promoted_artificial_voxel_.defined() || is_promoted_artificial_voxel_.size(0) != octpath_old.size(0)) {
            is_promoted_artificial_voxel_ = torch::zeros({octpath_old.size(0)}, bool_opts);
        } else if (is_promoted_artificial_voxel_.device() != octpath_old.device()) {
            is_promoted_artificial_voxel_ = is_promoted_artificial_voxel_.to(octpath_old.device());
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
        if (!geometrically_unstable_voxel_.defined() || geometrically_unstable_voxel_.size(0) != octpath_old.size(0)) {
            geometrically_unstable_voxel_ = torch::zeros({octpath_old.size(0)}, bool_opts);
        } else if (geometrically_unstable_voxel_.device() != octpath_old.device()) {
            geometrically_unstable_voxel_ = geometrically_unstable_voxel_.to(octpath_old.device());
        }
        if (!rendered_depth_candidate_voxel_.defined() ||
            rendered_depth_candidate_voxel_.size(0) != octpath_old.size(0)) {
            rendered_depth_candidate_voxel_ = torch::zeros({octpath_old.size(0)}, bool_opts);
        } else if (rendered_depth_candidate_voxel_.device() != octpath_old.device()) {
            rendered_depth_candidate_voxel_ = rendered_depth_candidate_voxel_.to(octpath_old.device());
        }
        if (!rendered_depth_candidate_support_count_.defined() ||
            rendered_depth_candidate_support_count_.size(0) != octpath_old.size(0)) {
            rendered_depth_candidate_support_count_ = torch::zeros({octpath_old.size(0)}, i32_opts);
        } else if (rendered_depth_candidate_support_count_.device() != octpath_old.device()) {
            rendered_depth_candidate_support_count_ = rendered_depth_candidate_support_count_.to(octpath_old.device());
        }
        if (!rendered_depth_candidate_last_seen_kf_.defined() ||
            rendered_depth_candidate_last_seen_kf_.size(0) != octpath_old.size(0)) {
            rendered_depth_candidate_last_seen_kf_ =
                torch::full({octpath_old.size(0)}, static_cast<int32_t>(-1), i32_opts);
        } else if (rendered_depth_candidate_last_seen_kf_.device() != octpath_old.device()) {
            rendered_depth_candidate_last_seen_kf_ = rendered_depth_candidate_last_seen_kf_.to(octpath_old.device());
        }
        if (!rendered_depth_candidate_source_kind_.defined() ||
            rendered_depth_candidate_source_kind_.size(0) != octpath_old.size(0)) {
            rendered_depth_candidate_source_kind_ = torch::zeros({octpath_old.size(0)}, i32_opts);
        } else if (rendered_depth_candidate_source_kind_.device() != octpath_old.device()) {
            rendered_depth_candidate_source_kind_ = rendered_depth_candidate_source_kind_.to(octpath_old.device());
        }
        if (Nk > 0) {
            const bool add_as_artificial_candidate =
                insert_rendered_depth_candidate && !insert_rendered_depth_candidate_as_real_protected;
            auto artificial_add_flag = torch::full(
                {Nk}, add_as_artificial_candidate, bool_opts);
            const bool add_as_orb =
                !insert_rendered_depth_candidate &&
                pending_real_insert_rr_entity_path_ == "world/orb/voxels_created";
            auto orb_add_flag = torch::full({Nk}, add_as_orb, bool_opts);
            const bool add_as_inactive_geo =
                !insert_rendered_depth_candidate &&
                pending_real_insert_rr_entity_path_ == "world/voxels_inactive_geo_densify/created";
            auto inactive_geo_add_flag = torch::full({Nk}, add_as_inactive_geo, bool_opts);
            const bool add_as_rgbd_fill_render_holes =
                !insert_rendered_depth_candidate &&
                pending_real_insert_rr_entity_path_ == "world/rgbd_fill_render_holes/created";
            auto rgbd_fill_render_holes_add_flag =
                torch::full({Nk}, add_as_rgbd_fill_render_holes, bool_opts);
            const bool add_as_depthanything_fill_holes =
                !add_as_artificial_candidate &&
                pending_real_insert_rr_entity_path_ == "world/mono_prior_fill_holes/created";
            auto depthanything_fill_holes_add_flag =
                torch::full({Nk}, add_as_depthanything_fill_holes, bool_opts);
            auto promoted_add_flag = torch::zeros({Nk}, bool_opts);
            auto exist_since_add = torch::full(
                {Nk}, static_cast<int32_t>(iteration), i32_opts);
            auto exist_since_kf_add = torch::full(
                {Nk}, current_kf_count, i32_opts);
            auto unstable_add_flag = torch::zeros({Nk}, bool_opts);
            auto rendered_depth_add_flag = torch::full(
                {Nk}, add_as_artificial_candidate, bool_opts);
            auto rendered_depth_support_add = torch::full(
                {Nk},
                static_cast<int32_t>(add_as_artificial_candidate ? 1 : 0),
                i32_opts);
            auto rendered_depth_last_seen_add = torch::full(
                {Nk},
                add_as_artificial_candidate ? current_kf_count : static_cast<int32_t>(-1),
                i32_opts);
            auto rendered_depth_source_add = torch::full(
                {Nk},
                static_cast<int32_t>(pending_insert_rendered_depth_candidate_source_kind_ > 0
                    ? pending_insert_rendered_depth_candidate_source_kind_
                    : 0),
                i32_opts);
            is_artificial_voxel_ = torch::cat({is_artificial_voxel_, artificial_add_flag}, 0).contiguous();
            is_orb_voxel_ = torch::cat({is_orb_voxel_, orb_add_flag}, 0).contiguous();
            is_inactive_geo_voxel_ = torch::cat({is_inactive_geo_voxel_, inactive_geo_add_flag}, 0).contiguous();
            is_rgbd_fill_render_holes_voxel_ =
                torch::cat({is_rgbd_fill_render_holes_voxel_, rgbd_fill_render_holes_add_flag}, 0).contiguous();
            is_depthanything_fill_holes_voxel_ =
                torch::cat({is_depthanything_fill_holes_voxel_, depthanything_fill_holes_add_flag}, 0).contiguous();
            is_promoted_artificial_voxel_ = torch::cat({is_promoted_artificial_voxel_, promoted_add_flag}, 0).contiguous();
            exist_since_iter_ = torch::cat({exist_since_iter_, exist_since_add}, 0).contiguous();
            exist_since_kf_ = torch::cat({exist_since_kf_, exist_since_kf_add}, 0).contiguous();
            geometrically_unstable_voxel_ = torch::cat({geometrically_unstable_voxel_, unstable_add_flag}, 0).contiguous();
            rendered_depth_candidate_voxel_ =
                torch::cat({rendered_depth_candidate_voxel_, rendered_depth_add_flag}, 0).contiguous();
            rendered_depth_candidate_support_count_ =
                torch::cat({rendered_depth_candidate_support_count_, rendered_depth_support_add}, 0).contiguous();
            rendered_depth_candidate_last_seen_kf_ =
                torch::cat({rendered_depth_candidate_last_seen_kf_, rendered_depth_last_seen_add}, 0).contiguous();
            rendered_depth_candidate_source_kind_ =
                torch::cat({rendered_depth_candidate_source_kind_, rendered_depth_source_add}, 0).contiguous();
        }
    }

    if (insert_rendered_depth_candidate && pending_support_updates > 0) {
        auto support_idx = support_idx_deferred
            .to(rendered_depth_candidate_support_count_.device())
            .to(torch::kLong)
            .contiguous();
        auto prev = rendered_depth_candidate_support_count_.index_select(0, support_idx);
        rendered_depth_candidate_support_count_.index_put_({support_idx}, prev + 1);
        rendered_depth_candidate_last_seen_kf_.index_put_(
            {support_idx},
            torch::full(
                {support_idx.size(0)},
                current_kf_count,
                torch::TensorOptions().dtype(torch::kInt32).device(rendered_depth_candidate_last_seen_kf_.device())));
    }

    // Apply deferred promotions after real insertion in this call.
    if (enable_artificial_promotion_ && pending_promotions > 0) {
        auto promote_idx = promote_idx_deferred.to(is_artificial_voxel_.device()).to(torch::kLong).contiguous();
        is_artificial_voxel_.index_put_({promote_idx}, false);
        is_promoted_artificial_voxel_.index_put_({promote_idx}, true);
        if (rendered_depth_candidate_voxel_.defined() &&
            rendered_depth_candidate_voxel_.size(0) == is_artificial_voxel_.size(0)) {
            rendered_depth_candidate_voxel_.index_put_({promote_idx}, false);
        }
        // Preserve insertion provenance after promotion so live debug views can
        // still identify voxels that originated from rendered-hole-fill.
        promoted_artificials_count = promote_idx.size(0);
        total_promoted_artificial_voxels_ += promoted_artificials_count;
    }

    // ── 7) Append learnables for new rows ───────────────────────────────────
    // _subdiv_p
    auto subdiv_add = torch::ones({Nk,1}, torch::dtype(torch::kFloat32).device(dev));
    this->subdiv_p_ = torch::cat({this->subdiv_p_.detach(), subdiv_add}, 0)
                          .contiguous()
                          .detach()
                          .requires_grad_();

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
    const int64_t M_prev = grid_pts_key_.defined() ? grid_pts_key_.size(0) : 0;
    const int64_t M_curr = grid_pts_key_new.size(0);

    // Prepare grow only when needed (group 0)
    torch::Tensor grow; // undefined by default
    if (M_curr > M_prev) {
        grow = makeGeoGridInitRows_(
            grid_pts_key_new,
            M_prev,
            M_curr,
            /*default_value=*/-10.0f);
    }
    // ── 9) Append rows to optimizer param groups ────────────────────────────
    // Important: only call for non-empty additions.
    if (grow.defined() && grow.size(0) > 0) {
        appendGroup_(/*group_idx=*/0, /*add_rows=*/grow, &this->_geo_grid_pts_);
    }
    if (Nk > 0) {
        appendGroup_(/*group_idx=*/1, /*add_rows=*/sh0_add, &this->sh0_);
        appendGroup_(/*group_idx=*/2, /*add_rows=*/shs_add, &this->shs_);
    }

    auto run_local_frontier_fill = [&]() -> int64_t {
        if (cams.empty()) {
            std::cout << "[increasePcd] local_frontier_fill: no cameras, skip.\n";
            return 0;
        }

        auto devL = scene_min_t_.device();
        const auto long_opts = torch::TensorOptions().dtype(torch::kLong).device(devL);
        const auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(devL);

        const int base_L = static_cast<int>(octlevel_);
        const int MAX_L  = max_num_levels_;
        TORCH_CHECK(base_L >= 1 && base_L <= MAX_L,
                    "[local_frontier_fill] base_L out of range: ", base_L);

        const int levels_below  = std::max(0, MAX_L - base_L);
        const int bits_to_clear = 3 * levels_below;
        long long lower_mask    = (bits_to_clear > 0) ? ((1LL << bits_to_clear) - 1LL) : 0LL;
        long long keep_mask_ll  = ~lower_mask;
        auto keep_mask = torch::full(
            {1},
            static_cast<int64_t>(keep_mask_ll),
            torch::TensorOptions().dtype(torch::kInt64).device(devL)
        );

        auto octpath_occ  = this->oct_path_.contiguous();
        auto octlevel_occ = this->oct_level_.contiguous();
        if (octpath_occ.numel() == 0) {
            return 0;
        }
        auto occ_path_i64  = octpath_occ.view({-1}).to(torch::kInt64).contiguous();
        auto occ_level_i64 = octlevel_occ.view({-1}).to(torch::kInt64).contiguous();
        auto key_occ_exact = occ_path_i64.mul(256).add(occ_level_i64);

        auto make_keep_mask_for_level = [&](int level) -> torch::Tensor {
            const int levels_below_l  = std::max(0, MAX_L - level);
            const int bits_to_clear_l = 3 * levels_below_l;
            long long lower_mask_l    = (bits_to_clear_l > 0) ? ((1LL << bits_to_clear_l) - 1LL) : 0LL;
            long long keep_mask_l_ll  = ~lower_mask_l;
            return torch::full(
                {1},
                static_cast<int64_t>(keep_mask_l_ll),
                torch::TensorOptions().dtype(torch::kInt64).device(devL));
        };

        auto unique_sorted_1d = [&](const torch::Tensor& t_in) -> torch::Tensor {
            auto t = t_in.contiguous().view({-1});
            if (t.numel() <= 1) return t;
            auto sorted = std::get<0>(t.sort(/*dim=*/0));
            auto keep = torch::empty_like(sorted, torch::kBool);
            keep.index_put_({0}, true);
            auto neq = sorted.index({torch::indexing::Slice(1, torch::indexing::None)})
                    != sorted.index({torch::indexing::Slice(torch::indexing::None, -1)});
            keep.index_put_({torch::indexing::Slice(1, torch::indexing::None)}, neq);
            auto idx = torch::nonzero(keep).view({-1});
            return sorted.index_select(0, idx).contiguous();
        };

        // A candidate at level L is considered occupied if:
        // 1) exact node exists at L, or
        // 2) an ancestor exists at coarser level, or
        // 3) any finer node projects to this node.
        auto occupied_hier_at_level = [&](const torch::Tensor& cand_path_i64, int level) -> torch::Tensor {
            auto cand_path = cand_path_i64.contiguous().view({-1}).to(torch::kInt64);
            if (cand_path.numel() == 0) {
                return torch::empty({0}, torch::TensorOptions().dtype(torch::kBool).device(devL));
            }
            auto cand_key = cand_path.mul(256).add(
                torch::full_like(cand_path, static_cast<int64_t>(level)));

            auto is_occ = at::isin(cand_key, key_occ_exact).to(torch::kBool);

            for (int anc_l = 1; anc_l < level; ++anc_l) {
                auto keep_mask_anc = make_keep_mask_for_level(anc_l);
                auto anc_path = (cand_path & keep_mask_anc);
                auto anc_key = anc_path.mul(256).add(
                    torch::full_like(anc_path, static_cast<int64_t>(anc_l)));
                auto has_anc = at::isin(anc_key, key_occ_exact).to(torch::kBool);
                is_occ = is_occ | has_anc;
            }

            auto finer_occ_mask = (occ_level_i64 > level);
            if (finer_occ_mask.any().item<bool>()) {
                auto finer_idx = torch::nonzero(finer_occ_mask).view({-1});
                auto finer_path = occ_path_i64.index_select(0, finer_idx).contiguous();
                auto keep_mask_l = make_keep_mask_for_level(level);
                auto finer_proj = (finer_path & keep_mask_l);
                auto finer_proj_key = finer_proj.mul(256).add(
                    torch::full_like(finer_proj, static_cast<int64_t>(level)));
                finer_proj_key = unique_sorted_1d(finer_proj_key);
                auto has_desc = at::isin(cand_key, finer_proj_key).to(torch::kBool);
                is_occ = is_occ | has_desc;
            }
            return is_occ.contiguous();
        };

        // Side-neighbor expansion only (no edge/corner fill).
        auto axis = torch::eye(3, long_opts);
        auto side_shift = torch::cat({axis, -axis}, 0).contiguous(); // [6,3]

        auto real_mask = torch::ones({octpath_occ.size(0)}, bool_opts);
        if (is_artificial_voxel_.defined() && is_artificial_voxel_.size(0) == octpath_occ.size(0)) {
            real_mask = (~is_artificial_voxel_.to(devL).to(torch::kBool)).contiguous();
        }
        auto seed_path_all = octpath_occ.contiguous();
        auto seed_level_all = octlevel_occ.contiguous();

        // Frontier-fill seeds: only real voxels in dense core (computed in createFromPcd).
        auto seed_mask = real_mask.clone();
        if (use_dense_core_neighbor_fill_) {
            if (has_dense_core_bb_ &&
                dense_core_bb_min_.defined() && dense_core_bb_max_.defined()) {
                auto [seed_center, seed_size_unused] = decodeOctpath(
                    seed_path_all.contiguous(),
                    seed_level_all.contiguous(),
                    scene_center_.contiguous(),
                    scene_extent_.contiguous());
                (void)seed_size_unused;
                auto bb_min = dense_core_bb_min_.to(devL).contiguous().view({1, 3});
                auto bb_max = dense_core_bb_max_.to(devL).contiguous().view({1, 3});
                auto in_dense_core =
                    (seed_center >= bb_min).all(/*dim=*/1) &
                    (seed_center <= bb_max).all(/*dim=*/1);
                seed_mask = seed_mask & in_dense_core.to(torch::kBool).contiguous();
            } else {
                std::cout << "[increasePcd] local_frontier_fill: dense-core bbox unavailable; "
                          << "fall back to real-only seeds.\n";
            }
        }

        auto seed_idx = torch::nonzero(seed_mask).view({-1});
        if (seed_idx.numel() == 0) {
            return 0;
        }
        auto seed_path = seed_path_all.index_select(0, seed_idx).contiguous();
        auto seed_level = seed_level_all.index_select(0, seed_idx).contiguous();
        auto seed_level_i64 = seed_level.view({-1}).to(torch::kInt64).contiguous();
        auto uniq_level = unique_sorted_1d(seed_level_i64);
        std::cout << "[increasePcd] local_frontier_fill seeds all="
                  << seed_path_all.size(0)
                  << " real=" << real_mask.sum().item<int64_t>()
                  << " active=" << seed_path.size(0) << "\n";

        std::vector<torch::Tensor> cand_path_list;
        std::vector<torch::Tensor> cand_level_list;
        auto uniq_level_cpu = uniq_level.to(torch::kCPU);
        for (int64_t i = 0; i < uniq_level_cpu.numel(); ++i) {
            const int lv = static_cast<int>(uniq_level_cpu[i].item<int64_t>());
            if (lv < 1 || lv > MAX_L) continue;

            auto level_mask = (seed_level_i64 == static_cast<int64_t>(lv));
            auto level_idx = torch::nonzero(level_mask).view({-1});
            if (level_idx.numel() == 0) continue;

            auto seed_path_lv = seed_path.index_select(0, level_idx).contiguous();
            auto L_seed_lv = torch::full(
                std::vector<int64_t>{seed_path_lv.size(0), 1},
                static_cast<int64_t>(lv),
                torch::dtype(torch::kInt8).device(devL)).contiguous();

            auto ijk_seed_lv =
                UTILS::octpath_2_ijk(seed_path_lv, L_seed_lv).to(torch::kLong).contiguous();
            TORCH_CHECK(
                ijk_seed_lv.dim() == 2 && ijk_seed_lv.size(1) == 3,
                "[increasePcd/local_frontier_fill] ijk_seed_lv must be [N,3], got ",
                ijk_seed_lv.sizes()
            );

            auto ijk_nbr_lv = (ijk_seed_lv.unsqueeze(1) + side_shift.unsqueeze(0))
                .contiguous().view({-1, 3});
            const int64_t grid_limit_lv = (1LL << lv);
            auto in_low_lv  = (ijk_nbr_lv >= 0).all(1);
            auto in_high_lv = (ijk_nbr_lv < grid_limit_lv).all(1);
            auto inb_lv     = in_low_lv & in_high_lv;
            if (!inb_lv.any().item<bool>()) continue;

            auto inb_idx_lv = torch::nonzero(inb_lv).view({-1});
            ijk_nbr_lv = ijk_nbr_lv.index_select(0, inb_idx_lv).contiguous();

            ijk_nbr_lv = std::get<0>(uniqueRowsWithInverse(ijk_nbr_lv.contiguous())).contiguous();
            if (ijk_nbr_lv.dim() == 1) {
                ijk_nbr_lv = ijk_nbr_lv.view({-1, 3});
            }
            if (ijk_nbr_lv.numel() == 0) continue;

            auto L_nbr_lv = torch::full(
                std::vector<int64_t>{ijk_nbr_lv.size(0), 1},
                static_cast<int64_t>(lv),
                torch::dtype(torch::kInt8).device(devL)).contiguous();
            auto path_nbr_lv = UTILS::ijk_2_octpath(ijk_nbr_lv, L_nbr_lv).contiguous();

            auto is_occ_lv = occupied_hier_at_level(path_nbr_lv.view({-1}).to(torch::kInt64), lv);
            auto miss_idx_lv = torch::nonzero(~is_occ_lv).view({-1});
            if (miss_idx_lv.numel() == 0) continue;

            cand_path_list.push_back(path_nbr_lv.index_select(0, miss_idx_lv).contiguous());
            cand_level_list.push_back(L_nbr_lv.index_select(0, miss_idx_lv).contiguous());
        }

        if (cand_path_list.empty()) {
            return 0;
        }

        auto octpath_box = torch::cat(cand_path_list, 0).contiguous();
        auto L_box = torch::cat(cand_level_list, 0).contiguous();
        {
            auto key_all = octpath_box.view({-1}).to(torch::kInt64).mul(256)
                         + L_box.view({-1}).to(torch::kInt64);
            auto sort_pair = key_all.sort(/*dim=*/0);
            auto key_sorted = std::get<0>(sort_pair);
            auto perm = std::get<1>(sort_pair);
            auto keep = torch::empty_like(key_sorted, torch::kBool);
            keep.index_put_({0}, true);
            if (key_sorted.numel() > 1) {
                auto neq = key_sorted.index({torch::indexing::Slice(1, torch::indexing::None)})
                        != key_sorted.index({torch::indexing::Slice(torch::indexing::None, -1)});
                keep.index_put_({torch::indexing::Slice(1, torch::indexing::None)}, neq);
            }
            auto ksorted = torch::nonzero(keep).view({-1});
            auto korig = perm.index_select(0, ksorted);
            octpath_box = octpath_box.index_select(0, korig).contiguous();
            L_box = L_box.index_select(0, korig).contiguous();
        }
        if (max_dense_core_fill_cells_ > 0 && octpath_box.size(0) > max_dense_core_fill_cells_) {
            octpath_box = octpath_box.index({
                torch::indexing::Slice(0, max_dense_core_fill_cells_)});
            L_box = L_box.index({
                torch::indexing::Slice(0, max_dense_core_fill_cells_)});
        }
        if (max_artificial_cells_ > 0 && octpath_box.size(0) > max_artificial_cells_) {
            octpath_box = octpath_box.index({
                torch::indexing::Slice(0, max_artificial_cells_)});
            L_box = L_box.index({
                torch::indexing::Slice(0, max_artificial_cells_)});
        }
        if (octpath_box.numel() == 0) {
            return 0;
        }

        std::cout << "[increasePcd] local_frontier_fill generated candidates="
                  << octpath_box.size(0) << "\n";

        auto ijk_box =
            UTILS::octpath_2_ijk(octpath_box, L_box).to(torch::kLong).contiguous();

        // Compare with CURRENT topology (which already includes real additions above).
        auto octpath_cur  = this->oct_path_.contiguous();
        auto octlevel_cur = this->oct_level_.contiguous();

        auto key_box = octpath_box.view({-1}).to(torch::kInt64).mul(256)
                     + L_box.view({-1}).to(torch::kInt64);
        auto key_cur = octpath_cur.view({-1}).to(torch::kInt64).mul(256)
                     + octlevel_cur.view({-1}).to(torch::kInt64);

        auto is_dup = at::isin(key_box, key_cur).to(torch::kBool);
        auto new_mask_box = ~is_dup;

        // Drop base-level parents of already-subdivided regions.
        {
            if (octpath_cur.numel() > 0) {
                auto Lold_i64     = octlevel_cur.view({-1}).to(torch::kInt64);
                auto has_children = (Lold_i64 > base_L);

                if (has_children.any().item<bool>()) {
                    auto op_old_i64  = octpath_cur.view({-1}).to(torch::kInt64);
                    auto op_anc_base = (op_old_i64 & keep_mask);
                    auto sel_child   = torch::nonzero(has_children).view({-1});
                    op_anc_base      = op_anc_base.index_select(0, sel_child);

                    auto key_children_as_parent = op_anc_base.mul(256)
                        .add(torch::full_like(op_anc_base, static_cast<int64_t>(base_L)));

                    auto would_collide_parent =
                        at::isin(key_box, key_children_as_parent).to(torch::kBool);
                    new_mask_box = new_mask_box & (~would_collide_parent);
                }
            }
        }

        if (!new_mask_box.any().item<bool>()) {
            return 0;
        }

        auto sel_local = torch::nonzero(new_mask_box).view({-1});
        const int64_t Nm_local = sel_local.size(0);

        auto octpath_add2 = octpath_box.index_select(0, sel_local);
        auto L_add2       = L_box.index_select(0, sel_local);
        {
            auto [add_center, add_size_unused] = decodeOctpath(
                octpath_add2.contiguous(),
                L_add2.contiguous(),
                scene_center_.contiguous(),
                scene_extent_.contiguous());
            (void)add_size_unused;
            bb_min_viz = std::get<0>(add_center.min(/*dim=*/0, /*keepdim=*/false)).contiguous();
            bb_max_viz = std::get<0>(add_center.max(/*dim=*/0, /*keepdim=*/false)).contiguous();
            sel_artificials_viz = torch::arange(
                Nm_local, torch::TensorOptions().dtype(torch::kLong).device(devL));
            ijk_box_viz = torch::empty({0, 3}, long_opts);
        }

        this->oct_path_ = torch::cat({octpath_cur,  octpath_add2}, 0).contiguous();
        this->oct_level_ = torch::cat({octlevel_cur, L_add2}, 0).contiguous();
        {
            auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(devL);
            auto i32_opts  = torch::TensorOptions().dtype(torch::kInt32).device(devL);
            if (!is_artificial_voxel_.defined() || is_artificial_voxel_.size(0) != octpath_cur.size(0)) {
                is_artificial_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
            } else if (is_artificial_voxel_.device() != devL) {
                is_artificial_voxel_ = is_artificial_voxel_.to(devL);
            }
            if (!is_orb_voxel_.defined() || is_orb_voxel_.size(0) != octpath_cur.size(0)) {
                is_orb_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
            } else if (is_orb_voxel_.device() != devL) {
                is_orb_voxel_ = is_orb_voxel_.to(devL);
            }
            if (!is_inactive_geo_voxel_.defined() || is_inactive_geo_voxel_.size(0) != octpath_cur.size(0)) {
                is_inactive_geo_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
            } else if (is_inactive_geo_voxel_.device() != devL) {
                is_inactive_geo_voxel_ = is_inactive_geo_voxel_.to(devL);
            }
            if (!is_rgbd_fill_render_holes_voxel_.defined() ||
                is_rgbd_fill_render_holes_voxel_.size(0) != octpath_cur.size(0)) {
                is_rgbd_fill_render_holes_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
            } else if (is_rgbd_fill_render_holes_voxel_.device() != devL) {
                is_rgbd_fill_render_holes_voxel_ = is_rgbd_fill_render_holes_voxel_.to(devL);
            }
            if (!is_depthanything_fill_holes_voxel_.defined() ||
                is_depthanything_fill_holes_voxel_.size(0) != octpath_cur.size(0)) {
                is_depthanything_fill_holes_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
            } else if (is_depthanything_fill_holes_voxel_.device() != devL) {
                is_depthanything_fill_holes_voxel_ = is_depthanything_fill_holes_voxel_.to(devL);
            }
            if (!is_promoted_artificial_voxel_.defined() || is_promoted_artificial_voxel_.size(0) != octpath_cur.size(0)) {
                is_promoted_artificial_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
            } else if (is_promoted_artificial_voxel_.device() != devL) {
                is_promoted_artificial_voxel_ = is_promoted_artificial_voxel_.to(devL);
            }
            if (!exist_since_iter_.defined() || exist_since_iter_.size(0) != octpath_cur.size(0)) {
                exist_since_iter_ = torch::zeros({octpath_cur.size(0)}, i32_opts);
            } else if (exist_since_iter_.device() != devL) {
                exist_since_iter_ = exist_since_iter_.to(devL);
            }
            if (!exist_since_kf_.defined() || exist_since_kf_.size(0) != octpath_cur.size(0)) {
                exist_since_kf_ = torch::full({octpath_cur.size(0)}, static_cast<int32_t>(-1), i32_opts);
            } else if (exist_since_kf_.device() != devL) {
                exist_since_kf_ = exist_since_kf_.to(devL);
            }
            if (!geometrically_unstable_voxel_.defined() || geometrically_unstable_voxel_.size(0) != octpath_cur.size(0)) {
                geometrically_unstable_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
            } else if (geometrically_unstable_voxel_.device() != devL) {
                geometrically_unstable_voxel_ = geometrically_unstable_voxel_.to(devL);
            }
            auto art_flag_add = torch::ones({Nm_local}, bool_opts);
            auto orb_flag_add = torch::zeros({Nm_local}, bool_opts);
            auto inactive_geo_flag_add = torch::zeros({Nm_local}, bool_opts);
            auto rgbd_fill_render_holes_flag_add = torch::zeros({Nm_local}, bool_opts);
            auto depthanything_fill_holes_flag_add = torch::zeros({Nm_local}, bool_opts);
            auto promoted_add_flag = torch::zeros({Nm_local}, bool_opts);
            auto exist_since_add = torch::full(
                {Nm_local}, static_cast<int32_t>(iteration), i32_opts);
            auto exist_since_kf_add = torch::full(
                {Nm_local}, current_kf_count, i32_opts);
            auto unstable_add_flag = torch::zeros({Nm_local}, bool_opts);
            is_artificial_voxel_ = torch::cat({is_artificial_voxel_, art_flag_add}, 0).contiguous();
            is_orb_voxel_ = torch::cat({is_orb_voxel_, orb_flag_add}, 0).contiguous();
            is_inactive_geo_voxel_ = torch::cat({is_inactive_geo_voxel_, inactive_geo_flag_add}, 0).contiguous();
            is_rgbd_fill_render_holes_voxel_ =
                torch::cat({is_rgbd_fill_render_holes_voxel_, rgbd_fill_render_holes_flag_add}, 0).contiguous();
            is_depthanything_fill_holes_voxel_ =
                torch::cat({is_depthanything_fill_holes_voxel_, depthanything_fill_holes_flag_add}, 0).contiguous();
            is_promoted_artificial_voxel_ = torch::cat({is_promoted_artificial_voxel_, promoted_add_flag}, 0).contiguous();
            exist_since_iter_ = torch::cat({exist_since_iter_, exist_since_add}, 0).contiguous();
            exist_since_kf_ = torch::cat({exist_since_kf_, exist_since_kf_add}, 0).contiguous();
            geometrically_unstable_voxel_ = torch::cat({geometrically_unstable_voxel_, unstable_add_flag}, 0).contiguous();
        }

        const int n_sh_rest_local = (max_sh_degree_ + 1)*(max_sh_degree_ + 1) - 1;
        auto shs_add2    = torch::zeros({Nm_local, n_sh_rest_local, 3}, torch::dtype(torch::kFloat32).device(devL)).contiguous();
        auto subdiv_add2 = torch::ones({Nm_local,1},                   torch::dtype(torch::kFloat32).device(devL)).contiguous();

        auto rgb_seed = torch::tensor(
            {artificial_bg_rgb_[0], artificial_bg_rgb_[1], artificial_bg_rgb_[2]},
            torch::dtype(torch::kFloat32).device(devL));
        if (rgb_add.defined() && rgb_add.numel() > 0) {
            rgb_seed = rgb_add.mean(/*dim=*/0, /*keepdim=*/false).clamp_(0.0f, 1.0f);
        }
        auto rgb_add2 = rgb_seed.view({1, 3}).repeat({Nm_local, 1}).contiguous();

        auto sh0_add2 = rgbToShZero(rgb_add2).contiguous();

        auto [grid_pts_key_new_local, vox_key_new_local] =
            buildGridPtsLink(
                this->oct_path_.contiguous(),
                this->oct_level_.contiguous(),
                max_num_levels_);
        (void)vox_key_new_local;
        const int64_t M_prev_local = grid_pts_key_.defined() ? grid_pts_key_.size(0) : 0;
        const int64_t M_curr_local = grid_pts_key_new_local.size(0);
        if (M_curr_local > M_prev_local) {
            auto grow2 = makeGeoGridInitRows_(
                grid_pts_key_new_local,
                M_prev_local,
                M_curr_local,
                /*default_value=*/-10.0f);
            appendGroup_(/*group_idx=*/0, grow2, &this->_geo_grid_pts_);
        }
        appendGroup_(/*group_idx=*/1, sh0_add2, &this->sh0_);
        appendGroup_(/*group_idx=*/2, shs_add2, &this->shs_);

        this->subdiv_p_ = torch::cat({this->subdiv_p_.detach(), subdiv_add2}, 0)
                              .contiguous()
                              .detach()
                              .requires_grad_(true);

        artificial_fill_happened_ = true;
        std::cout << "[increasePcd] local_frontier_fill added " << Nm_local
                  << " support voxels.\n";
        return Nm_local;
    };

    if (!insert_rendered_depth_candidate && !fill_empty_cells_ && use_local_frontier_fill_) {
        Nm_added += run_local_frontier_fill();
    }

    if (!insert_rendered_depth_candidate && fill_empty_cells_) {
        const bool warmup_reached = (iteration >= fill_empty_cells_warmup_iters_);
        if (!fill_empty_cells_done_ && !warmup_reached && !fill_empty_cells_warmup_notified_) {
            std::cout << "[increasePcd] fill_empty_cells_: waiting warmup (iter="
                      << iteration << " < " << fill_empty_cells_warmup_iters_ << ")\n";
            fill_empty_cells_warmup_notified_ = true;
        }
        const bool should_fill = (!fill_empty_cells_done_) && warmup_reached;
        if (should_fill) {
            bool did_local_fill = false;

            if (!did_local_fill) {
            // --- A) One-shot bbox from CURRENT batch heuristic (older behavior) ---
            torch::Tensor bb_min; // [3], world
            torch::Tensor bb_max; // [3], world
            bool bbox_ready = false;
            // Use accumulated real PCD (previous + current) for heuristic bbox estimation.
            torch::Tensor fill_pts_cpu = xyz_cpu.to(torch::kCPU).to(torch::kFloat32).contiguous();
            if (real_pcd_points_accum_cpu_.defined() && real_pcd_points_accum_cpu_.numel() > 0) {
                auto prev_pts_cpu =
                    real_pcd_points_accum_cpu_.to(torch::kCPU).to(torch::kFloat32).contiguous();
                if (prev_pts_cpu.dim() == 2 && prev_pts_cpu.size(1) == 3 && prev_pts_cpu.size(0) > 0) {
                    fill_pts_cpu = torch::cat({prev_pts_cpu, fill_pts_cpu}, 0).contiguous();
                }
            }
            std::cout << "[dense_core/refresh][fill] begin points="
                      << fill_pts_cpu.size(0)
                      << " rate=" << dense_core_pcd_density_rate_ << "\n";
            std::string heuristic_err_msg;
            bool crossing_ok = false;
            {
                auto pts_f32 = fill_pts_cpu.to(torch::kCPU).to(torch::kFloat32).contiguous();
                if (pts_f32.defined() && pts_f32.dim() == 2 && pts_f32.size(1) == 3 && pts_f32.size(0) > 0) {
                    auto center_cpu = std::get<0>(pts_f32.median(/*dim=*/0, /*keepdim=*/false)).contiguous(); // [3], CPU
                    auto dist = std::get<0>(
                        (pts_f32 - center_cpu.view({1, 3})).abs().max(/*dim=*/1, /*keepdim=*/false))
                                    .to(torch::kFloat32)
                                    .contiguous(); // [N]
                    dist = std::get<0>(dist.sort(/*dim=*/0)).contiguous();

                    const int64_t n = dist.size(0);
                    if (n > 0) {
                        auto idx = torch::arange(
                            1, n + 1,
                            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
                        auto nonzero = (dist > 0.0f).to(torch::kFloat32);
                        auto density = idx * nonzero / ((2.0f * dist).pow(3) + 1e-6f);

                        int64_t begin_idx = static_cast<int64_t>(std::llround(static_cast<double>(n) * 0.05));
                        begin_idx = std::max<int64_t>(0, std::min<int64_t>(n - 1, begin_idx));

                        auto tail = density.index({torch::indexing::Slice(begin_idx, torch::indexing::None)}).contiguous();
                        if (tail.numel() > 0) {
                            const int64_t max_idx = begin_idx + tail.argmax().item<int64_t>();
                            const float max_density = density.index({max_idx}).item<float>();
                            const float target_density = dense_core_pcd_density_rate_ * max_density;
                            auto right = density.index({torch::indexing::Slice(max_idx, torch::indexing::None)}).contiguous();
                            auto below = torch::nonzero(right < target_density).view({-1}).contiguous();
                            const int64_t crossing_count = below.numel();
                            const float right_min = right.min().item<float>();
                            const float right_last = right.index({-1}).item<float>();

                            std::cout << "[dense_core/refresh][fill/precheck] n=" << n
                                      << " begin_idx=" << begin_idx
                                      << " max_idx=" << max_idx
                                      << " max_density=" << max_density
                                      << " target_density=" << target_density
                                      << " right_min=" << right_min
                                      << " right_last=" << right_last
                                      << " crossing_count=" << crossing_count
                                      << "\n";

                            crossing_ok = (crossing_count > 0);
                        }
                    }
                }
            }

            if (!crossing_ok) {
                std::cout << "[dense_core/refresh][fill] no_density_crossing at rate="
                          << dense_core_pcd_density_rate_
                          << "; skipping dense-core heuristic this round.\n";
            } else {
                if (auto bound = mainSceneBoundPcdHeuristicCpp(
                        fill_pts_cpu.contiguous(),
                        dense_core_pcd_density_rate_)) {
                    auto center_t = bound->first
                        .to(torch::kCUDA)
                        .to(torch::kFloat32)
                        .contiguous()
                        .view({3});
                    const float radius_f = bound->second;
                    if (std::isfinite(radius_f) && radius_f > 0.0f) {
                        auto radius_t = torch::full(
                            {3}, radius_f, torch::dtype(torch::kFloat32).device(torch::kCUDA));
                        bb_min = (center_t - radius_t).contiguous();
                        bb_max = (center_t + radius_t).contiguous();
                        bbox_ready = true;
                        std::cout << "[dense_core/refresh][fill] done source=cpp radius="
                                  << radius_f << "\n";
                        std::cout << "[increasePcd] fill_empty_cells_: one-shot using accumulated-real-pcd heuristic bbox.\n";
                    } else {
                        std::cout << "[increasePcd] fill_empty_cells_: invalid heuristic radius="
                                  << radius_f << ".\n";
                    }
                } else {
                    heuristic_err_msg = "dense-core heuristic returned no valid center/radius";
                }
            }

            if (!bbox_ready) {
                if (!heuristic_err_msg.empty()) {
                    std::cout << "[increasePcd] fill_empty_cells_: dense-core heuristic failed:\n"
                              << heuristic_err_msg << "\n";
                }
                std::cout << "[increasePcd] fill_empty_cells_: bbox unavailable; will retry in next increasePcd.\n";
            }

            if (bbox_ready) {
            // Persist this recomputed dense-core so insertion-time filtering
            // uses the same latest bbox after warmup fill.
            dense_core_bb_min_ = bb_min.contiguous();
            dense_core_bb_max_ = bb_max.contiguous();
            has_dense_core_bb_ = true;
            std::cout << "[increasePcd] dense_core_bb_min=" << dense_core_bb_min_
                      << " dense_core_bb_max=" << dense_core_bb_max_ << std::endl;
            fill_empty_cells_done_ = true;

            // --- B) Convert bbox (world) -> dense ijk box at cached level/voxel size ---
            // Use cached scene_min_t_ and vox_eff_ (scalar size, but we kept it as [1,1]):
            auto vox_t = vox_eff_.mean()        // 0-dim CUDA float tensor
                            .view({1})
                            .repeat({3})       // [3]
                            .contiguous();     // CUDA float

            // Convert to indices, clamp to grid range [0, 2^L - 1]
            const int64_t grid_limit = (1LL << static_cast<int>(octlevel_)); // 2^L
            auto ijk_min = ((bb_min - scene_min_t_) / vox_t).floor().to(torch::kLong);
            auto ijk_max = ((bb_max - scene_min_t_) / vox_t).ceil().to(torch::kLong) - 1;
            // Clamp to valid grid
            auto zero = torch::zeros_like(ijk_min);
            auto lim  = torch::full_like(ijk_max, grid_limit - 1);
            ijk_min = torch::maximum(ijk_min, zero);
            ijk_max = torch::minimum(ijk_max, lim);

            auto lens = (ijk_max - ijk_min + 1);        // [3] Long on CUDA
            int64_t nx = lens[0].item<int64_t>();
            int64_t ny = lens[1].item<int64_t>();
            int64_t nz = lens[2].item<int64_t>();
            long double Nc_est = (long double)nx * ny * nz;

            int64_t cap = std::max<int64_t>(1, max_artificial_cells_);    // set a sensible default (e.g., 200k)
            int64_t stride = 1;
            if (Nc_est > cap) {
                long double s = std::cbrt(Nc_est / (long double)cap);
                stride = std::max<int64_t>(1, (int64_t)std::ceil(s));
            }

            // If degenerate (all out), skip
            if ((ijk_min <= ijk_max).all().item<bool>()) {
                // --- C) Enumerate dense cells & set-diff vs current SVM topology ---
                auto ir = torch::arange(ijk_min[0].item<int64_t>(), ijk_max[0].item<int64_t>() + 1,
                                        stride, torch::dtype(torch::kLong).device(torch::kCUDA));
                auto jr = torch::arange(ijk_min[1].item<int64_t>(), ijk_max[1].item<int64_t>() + 1,
                                        stride, torch::dtype(torch::kLong).device(torch::kCUDA));
                auto kr = torch::arange(ijk_min[2].item<int64_t>(), ijk_max[2].item<int64_t>() + 1,
                                        stride, torch::dtype(torch::kLong).device(torch::kCUDA));
                auto grids = torch::meshgrid({ir, jr, kr}, /*indexing=*/"ij");
                auto ijk_box = torch::stack(
                    { grids[0].contiguous().view({-1}),
                    grids[1].contiguous().view({-1}),
                    grids[2].contiguous().view({-1}) }, 1).contiguous(); // [Nc,3]

                if (max_artificial_cells_ > 0 && ijk_box.size(0) > max_artificial_cells_) {
                    std::cout << "[increasePcd] fill_empty_cells_: limiting artificial cells"<< std::endl;
                    ijk_box = ijk_box.index({torch::indexing::Slice(0, max_artificial_cells_)});
                }

                auto dev = ijk_box.device();
                auto L_box = torch::full({ijk_box.size(0),1}, octlevel_, torch::dtype(torch::kInt8).device(dev)).contiguous();

                // Build morton/octpath for those cells
                auto octpath_box = UTILS::ijk_2_octpath(ijk_box, L_box).contiguous(); // [Nc,1] int64
                
                // ------------------------------------------------------------------------
                // B) OPTIONAL: camera-based filtering (mark_max_samp_rate + mark_near)
                //     so artificial voxels are also consistent with SVRaster camera logic
                // ------------------------------------------------------------------------
                if (filter_near_voxels_ && !cams.empty()) {
                    // Decode voxel centers/sizes for artificial voxels
                    auto [vox_center_box, vox_size_box] = decodeOctpath(
                        octpath_box.contiguous(),
                        L_box.contiguous(),
                        scene_center_.contiguous(),
                        scene_extent_.contiguous());

                    auto Nc_before = octpath_box.size(0);

                    // 1) Visibility / sampling rate
                    at::Tensor rate_box =
                        markSvrasterMaxSampRateDirect(cams, octpath_box, vox_center_box, vox_size_box);
                    at::Tensor kept_cam = rate_box > 0;
                    int64_t n_rate_pos_box = kept_cam.sum().item<int64_t>();

                    // 2) Near filtering with same threshold as for PCD voxels
                    //    (you can pull this out as a const float near_thresh = ... at top)
                    const float near_thresh = filter_near_voxels_ ? 0.2f : -1.0f;
                    int64_t n_near_hit_box = 0;
                    if (near_thresh > 0.0f) {
                        at::Tensor is_near_box =
                            markSvrasterNearDirect(cams, octpath_box, vox_center_box, vox_size_box, near_thresh);
                        kept_cam = kept_cam & (~is_near_box);
                        n_near_hit_box = is_near_box.sum().item<int64_t>();
                    }

                    auto idx_box = torch::nonzero(kept_cam).view({-1});
                    int64_t K_box = idx_box.size(0);

                    if (K_box == 0) {
                        std::cout << "[increasePcd/fill_empty_cells] "
                                << "all artificial voxels filtered out by camera visibility / near; skipping."
                                << std::endl;
                        return;
                    }

                    if (K_box < octpath_box.size(0)) {
                        // Apply mask consistently to topology + ijk_box
                        octpath_box = octpath_box.index_select(0, idx_box).contiguous(); // [K_box,1]
                        L_box       = L_box.index_select(0, idx_box).contiguous();       // [K_box,1]
                        ijk_box     = ijk_box.index_select(0, idx_box).contiguous();     // [K_box,3]
                    }

                    std::cout << "[increasePcd/fill_empty_cells_cam] Nc_before=" << Nc_before
                            << " rate>0=" << n_rate_pos_box
                            << " near_hit=" << n_near_hit_box
                            << " kept_final=" << octpath_box.size(0) << std::endl;
                }

                // Compare with CURRENT topology (which already includes real additions above)
                auto octpath_cur  = this->oct_path_.contiguous();
                auto octlevel_cur = this->oct_level_.contiguous();

                auto key_box = octpath_box.view({-1}).to(torch::kInt64).mul(256)
                            + L_box.view({-1}).to(torch::kInt64);
                auto key_cur = octpath_cur.view({-1}).to(torch::kInt64).mul(256)
                            + octlevel_cur.view({-1}).to(torch::kInt64);

                auto is_dup = at::isin(key_box, key_cur).to(torch::kBool);
                auto new_mask_box = ~is_dup;

                // --- Also drop base-level parents of already-subdivided regions (L_old > base_L) ---
                {
                    const int MAX_L  = max_num_levels_;
                    const int base_L = static_cast<int>(octlevel_);

                    if (octpath_cur.numel() > 0) {
                        auto Lold_i64     = octlevel_cur.view({-1}).to(torch::kInt64);   // [No]
                        auto has_children = (Lold_i64 > base_L);                         // [No] bool

                        if (has_children.any().item<bool>()) {
                            // Build mask to zero out octant bits below base_L
                            const int levels_below  = std::max(0, MAX_L - base_L);
                            const int bits_to_clear = 3 * levels_below;
                            long long lower_mask    = (bits_to_clear > 0) ? ((1LL << bits_to_clear) - 1LL) : 0LL;
                            long long keep_mask_ll  = ~lower_mask;

                            auto dev = octpath_cur.device();
                            auto keep_mask = torch::full(
                                {1},
                                static_cast<int64_t>(keep_mask_ll),
                                torch::TensorOptions().dtype(torch::kInt64).device(dev)
                            );

                            // Compute ancestors at base_L for all L_old > base_L
                            auto op_old_i64  = octpath_cur.view({-1}).to(torch::kInt64);  // [No]
                            auto op_anc_base = (op_old_i64 & keep_mask);                  // [No]
                            auto sel_child   = torch::nonzero(has_children).view({-1});   // [K]
                            op_anc_base      = op_anc_base.index_select(0, sel_child);    // [K]

                            // Keys for those parents at base_L
                            auto key_children_as_parent = op_anc_base.mul(256)
                                                        .add(torch::full_like(op_anc_base,
                                                                            static_cast<int64_t>(base_L)));

                            // small helper
                            auto unique_sorted_1d = [](const at::Tensor& t)->at::Tensor {
                                TORCH_CHECK(t.dim() == 1, "unique_sorted_1d expects a 1-D tensor");
                                if (t.numel() <= 1) return t.contiguous();
                                auto sort_res = t.sort(0);
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

                            // Any base-level candidate that equals one of these parents must be skipped
                            auto would_collide_parent =
                                at::isin(key_box, key_children_as_parent).to(torch::kBool); // [Nc]

                            new_mask_box = new_mask_box & (~would_collide_parent);
                        }
                    }
                }

                // Proceed with the filtered candidates
                if (new_mask_box.any().item<bool>()) {
                    auto sel = torch::nonzero(new_mask_box).view({-1});
                    Nm_added = sel.size(0);

                    bb_min_viz       = bb_min;         // [3] (CUDA)
                    bb_max_viz       = bb_max;         // [3] (CUDA)
                    sel_artificials_viz= sel.clone();    // [Nm] indices into ijk_box
                    ijk_box_viz      = ijk_box.clone();// [Nc,3] (CUDA)

                    auto octpath_add2 = octpath_box.index_select(0, sel); // [Nm,1]
                    auto L_add2       = L_box.index_select(0, sel);       // [Nm,1]

                    // Append topology
                    this->oct_path_ = torch::cat({octpath_cur,  octpath_add2}, 0).contiguous();
                    this->oct_level_ = torch::cat({octlevel_cur, L_add2}, 0).contiguous();
                    {
                        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
                        auto i32_opts  = torch::TensorOptions().dtype(torch::kInt32).device(dev);
                        if (!is_artificial_voxel_.defined() || is_artificial_voxel_.size(0) != octpath_cur.size(0)) {
                            is_artificial_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
                        } else if (is_artificial_voxel_.device() != dev) {
                            is_artificial_voxel_ = is_artificial_voxel_.to(dev);
                        }
                        if (!is_orb_voxel_.defined() || is_orb_voxel_.size(0) != octpath_cur.size(0)) {
                            is_orb_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
                        } else if (is_orb_voxel_.device() != dev) {
                            is_orb_voxel_ = is_orb_voxel_.to(dev);
                        }
                        if (!is_inactive_geo_voxel_.defined() || is_inactive_geo_voxel_.size(0) != octpath_cur.size(0)) {
                            is_inactive_geo_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
                        } else if (is_inactive_geo_voxel_.device() != dev) {
                            is_inactive_geo_voxel_ = is_inactive_geo_voxel_.to(dev);
                        }
                        if (!is_rgbd_fill_render_holes_voxel_.defined() ||
                            is_rgbd_fill_render_holes_voxel_.size(0) != octpath_cur.size(0)) {
                            is_rgbd_fill_render_holes_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
                        } else if (is_rgbd_fill_render_holes_voxel_.device() != dev) {
                            is_rgbd_fill_render_holes_voxel_ = is_rgbd_fill_render_holes_voxel_.to(dev);
                        }
                        if (!is_depthanything_fill_holes_voxel_.defined() ||
                            is_depthanything_fill_holes_voxel_.size(0) != octpath_cur.size(0)) {
                            is_depthanything_fill_holes_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
                        } else if (is_depthanything_fill_holes_voxel_.device() != dev) {
                            is_depthanything_fill_holes_voxel_ = is_depthanything_fill_holes_voxel_.to(dev);
                        }
                        if (!is_promoted_artificial_voxel_.defined() || is_promoted_artificial_voxel_.size(0) != octpath_cur.size(0)) {
                            is_promoted_artificial_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
                        } else if (is_promoted_artificial_voxel_.device() != dev) {
                            is_promoted_artificial_voxel_ = is_promoted_artificial_voxel_.to(dev);
                        }
                        if (!exist_since_iter_.defined() || exist_since_iter_.size(0) != octpath_cur.size(0)) {
                            exist_since_iter_ = torch::zeros({octpath_cur.size(0)}, i32_opts);
                        } else if (exist_since_iter_.device() != dev) {
                            exist_since_iter_ = exist_since_iter_.to(dev);
                        }
                        if (!exist_since_kf_.defined() || exist_since_kf_.size(0) != octpath_cur.size(0)) {
                            exist_since_kf_ = torch::full({octpath_cur.size(0)}, static_cast<int32_t>(-1), i32_opts);
                        } else if (exist_since_kf_.device() != dev) {
                            exist_since_kf_ = exist_since_kf_.to(dev);
                        }
                        if (!geometrically_unstable_voxel_.defined() || geometrically_unstable_voxel_.size(0) != octpath_cur.size(0)) {
                            geometrically_unstable_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
                        } else if (geometrically_unstable_voxel_.device() != dev) {
                            geometrically_unstable_voxel_ = geometrically_unstable_voxel_.to(dev);
                        }
                        auto art_flag_add = torch::ones({Nm_added}, bool_opts);
                        auto orb_flag_add = torch::zeros({Nm_added}, bool_opts);
                        auto inactive_geo_flag_add = torch::zeros({Nm_added}, bool_opts);
                        auto rgbd_fill_render_holes_flag_add = torch::zeros({Nm_added}, bool_opts);
                        auto depthanything_fill_holes_flag_add = torch::zeros({Nm_added}, bool_opts);
                        auto promoted_add_flag = torch::zeros({Nm_added}, bool_opts);
                        auto exist_since_add = torch::full(
                            {Nm_added}, static_cast<int32_t>(iteration), i32_opts);
                        auto exist_since_kf_add = torch::full(
                            {Nm_added}, current_kf_count, i32_opts);
                        auto unstable_add_flag = torch::zeros({Nm_added}, bool_opts);
                        is_artificial_voxel_ = torch::cat({is_artificial_voxel_, art_flag_add}, 0).contiguous();
                        is_orb_voxel_ = torch::cat({is_orb_voxel_, orb_flag_add}, 0).contiguous();
                        is_inactive_geo_voxel_ = torch::cat({is_inactive_geo_voxel_, inactive_geo_flag_add}, 0).contiguous();
                        is_rgbd_fill_render_holes_voxel_ =
                            torch::cat({is_rgbd_fill_render_holes_voxel_, rgbd_fill_render_holes_flag_add}, 0).contiguous();
                        is_depthanything_fill_holes_voxel_ =
                            torch::cat({is_depthanything_fill_holes_voxel_, depthanything_fill_holes_flag_add}, 0).contiguous();
                        is_promoted_artificial_voxel_ = torch::cat({is_promoted_artificial_voxel_, promoted_add_flag}, 0).contiguous();
                        exist_since_iter_ = torch::cat({exist_since_iter_, exist_since_add}, 0).contiguous();
                        exist_since_kf_ = torch::cat({exist_since_kf_, exist_since_kf_add}, 0).contiguous();
                        geometrically_unstable_voxel_ = torch::cat({geometrically_unstable_voxel_, unstable_add_flag}, 0).contiguous();
                    }

                    // Prepare learnables for artificials
                    const int n_sh_rest = (max_sh_degree_ + 1)*(max_sh_degree_ + 1) - 1;
                    auto shs_add2    = torch::zeros({Nm_added, n_sh_rest, 3}, torch::dtype(torch::kFloat32).device(dev)).contiguous();
                    auto subdiv_add2 = torch::ones({Nm_added,1},             torch::dtype(torch::kFloat32).device(dev)).contiguous();

                    auto rgb_add2 = torch::empty({Nm_added,3}, torch::dtype(torch::kFloat32).device(dev));
                    rgb_add2.index_put_({torch::indexing::Slice(),0}, artificial_bg_rgb_[0]);
                    rgb_add2.index_put_({torch::indexing::Slice(),1}, artificial_bg_rgb_[1]);
                    rgb_add2.index_put_({torch::indexing::Slice(),2}, artificial_bg_rgb_[2]);

                    auto sh0_add2 = rgbToShZero(rgb_add2.contiguous()).contiguous();

                    // Grid growth and optimizer-preserving appends
                    auto [grid_pts_key_new, vox_key_new] =
                        buildGridPtsLink(
                            this->oct_path_.contiguous(),
                            this->oct_level_.contiguous(),
                            max_num_levels_);
                    (void)vox_key_new;
                    const int64_t M_prev = grid_pts_key_.defined() ? grid_pts_key_.size(0) : 0;
                    const int64_t M_curr = grid_pts_key_new.size(0);
                    if (M_curr > M_prev) {
                        auto grow = makeGeoGridInitRows_(
                            grid_pts_key_new,
                            M_prev,
                            M_curr,
                            /*default_value=*/-10.0f);
                        appendGroup_(/*group_idx=*/0, grow, &this->_geo_grid_pts_);
                    }
                    appendGroup_(/*group_idx=*/1, sh0_add2, &this->sh0_);
                    appendGroup_(/*group_idx=*/2, shs_add2, &this->shs_);

                    // subdiv_p (not in optimizer groups)
                    this->subdiv_p_ = torch::cat({this->subdiv_p_.detach(), subdiv_add2}, 0)
                                          .contiguous()
                                          .detach()
                                          .requires_grad_(true);
                    // NEW: mark that we actually added artificial voxels this call
                    artificial_fill_happened_ = true;
                }
                // Keep your existing “new voxels” logging in sync by adding Nm_added to Nk (as noted earlier)
                // ... (use last_added = Nk + Nm_added later)
            } // valid bbox
            } // bbox_ready
            } // !did_local_fill
            if (fill_empty_cells_done_) {
                std::cout << "[increasePcd] fill_empty_cells_ added " << Nm_added
                          << " artificial voxels.\n";
            }
        } // should_fill
    } // fill_empty_cells_

    // ── 10) Rebuild renderer fields from the C++ topology ───────────────────
    this->oct_path_      = this->oct_path_.contiguous();
    this->oct_level_     = this->oct_level_.contiguous();
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
    // stats buffer resize
    this->max_w_ = torch::zeros({center_.size(0), 1}, torch::dtype(torch::kFloat32).device(dev));
    // Keep provenance tensor aligned with current topology size.
    if (!is_artificial_voxel_.defined() || is_artificial_voxel_.size(0) != center_.size(0)) {
        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, bool_opts);
        if (is_artificial_voxel_.defined() && is_artificial_voxel_.numel() > 0) {
            auto old = is_artificial_voxel_.to(dev).to(torch::kBool).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        is_artificial_voxel_ = aligned;
        std::cout << "[artificial/provenance] realigned flag tensor to N="
                  << center_.size(0) << "\n";
    }
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
        std::cout << "[orb/provenance] realigned flag tensor to N="
                  << center_.size(0) << "\n";
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
        std::cout << "[inactive_geo/provenance] realigned flag tensor to N="
                  << center_.size(0) << "\n";
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
        std::cout << "[rgbd_fill_render_holes/provenance] realigned flag tensor to N="
                  << center_.size(0) << "\n";
    }
    if (!is_depthanything_fill_holes_voxel_.defined() ||
        is_depthanything_fill_holes_voxel_.size(0) != center_.size(0)) {
        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, bool_opts);
        if (is_depthanything_fill_holes_voxel_.defined() &&
            is_depthanything_fill_holes_voxel_.numel() > 0) {
            auto old = is_depthanything_fill_holes_voxel_.to(dev).to(torch::kBool).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        is_depthanything_fill_holes_voxel_ = aligned;
        std::cout << "[depthanything_fill_holes/provenance] realigned flag tensor to N="
                  << center_.size(0) << "\n";
    }
    if (!is_promoted_artificial_voxel_.defined() || is_promoted_artificial_voxel_.size(0) != center_.size(0)) {
        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, bool_opts);
        if (is_promoted_artificial_voxel_.defined() && is_promoted_artificial_voxel_.numel() > 0) {
            auto old = is_promoted_artificial_voxel_.to(dev).to(torch::kBool).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        is_promoted_artificial_voxel_ = aligned;
        std::cout << "[artificial/promotion] realigned flag tensor to N="
                  << center_.size(0) << "\n";
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
        std::cout << "[exist_since_iter] realigned tensor to N="
                  << center_.size(0) << "\n";
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
        std::cout << "[exist_since_kf] realigned tensor to N="
                  << center_.size(0) << "\n";
    }
    if (!geometrically_unstable_voxel_.defined() || geometrically_unstable_voxel_.size(0) != center_.size(0)) {
        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, bool_opts);
        if (geometrically_unstable_voxel_.defined() && geometrically_unstable_voxel_.numel() > 0) {
            auto old = geometrically_unstable_voxel_.to(dev).to(torch::kBool).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        geometrically_unstable_voxel_ = aligned;
        std::cout << "[geometrically_unstable] realigned tensor to N="
                  << center_.size(0) << "\n";
    }
    if (!rendered_depth_candidate_voxel_.defined() ||
        rendered_depth_candidate_voxel_.size(0) != center_.size(0)) {
        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, bool_opts);
        if (rendered_depth_candidate_voxel_.defined() &&
            rendered_depth_candidate_voxel_.numel() > 0) {
            auto old = rendered_depth_candidate_voxel_.to(dev).to(torch::kBool).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        rendered_depth_candidate_voxel_ = aligned;
        std::cout << "[rendered_depth_candidate] realigned flag tensor to N="
                  << center_.size(0) << "\n";
    }
    if (!rendered_depth_candidate_support_count_.defined() ||
        rendered_depth_candidate_support_count_.size(0) != center_.size(0)) {
        auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, i32_opts);
        if (rendered_depth_candidate_support_count_.defined() &&
            rendered_depth_candidate_support_count_.numel() > 0) {
            auto old = rendered_depth_candidate_support_count_.to(dev).to(torch::kInt32).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        rendered_depth_candidate_support_count_ = aligned;
        std::cout << "[rendered_depth_candidate] realigned support tensor to N="
                  << center_.size(0) << "\n";
    }
    if (!rendered_depth_candidate_last_seen_kf_.defined() ||
        rendered_depth_candidate_last_seen_kf_.size(0) != center_.size(0)) {
        auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(dev);
        auto aligned = torch::full({center_.size(0)}, static_cast<int32_t>(-1), i32_opts);
        if (rendered_depth_candidate_last_seen_kf_.defined() &&
            rendered_depth_candidate_last_seen_kf_.numel() > 0) {
            auto old = rendered_depth_candidate_last_seen_kf_.to(dev).to(torch::kInt32).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        rendered_depth_candidate_last_seen_kf_ = aligned;
        std::cout << "[rendered_depth_candidate] realigned last_seen tensor to N="
                  << center_.size(0) << "\n";
    }
    if (!rendered_depth_candidate_source_kind_.defined() ||
        rendered_depth_candidate_source_kind_.size(0) != center_.size(0)) {
        auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, i32_opts);
        if (rendered_depth_candidate_source_kind_.defined() &&
            rendered_depth_candidate_source_kind_.numel() > 0) {
            auto old = rendered_depth_candidate_source_kind_.to(dev).to(torch::kInt32).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        rendered_depth_candidate_source_kind_ = aligned;
        std::cout << "[rendered_depth_candidate] realigned source tensor to N="
                  << center_.size(0) << "\n";
    }

    // Keep dense-core history as accumulated raw PCD points (CPU), not voxel centers.
    if (!insert_rendered_depth_candidate && xyz_cpu.defined() && xyz_cpu.numel() > 0) {
        auto new_pts_cpu = xyz_cpu.to(torch::kCPU).to(torch::kFloat32).contiguous();
        if (!real_pcd_points_accum_cpu_.defined() || real_pcd_points_accum_cpu_.numel() == 0) {
            real_pcd_points_accum_cpu_ = new_pts_cpu;
        } else {
            real_pcd_points_accum_cpu_ =
                torch::cat({real_pcd_points_accum_cpu_, new_pts_cpu}, 0).contiguous();
        }
        if (max_real_pcd_points_ > 0 && real_pcd_points_accum_cpu_.size(0) > max_real_pcd_points_) {
            const int64_t total = real_pcd_points_accum_cpu_.size(0);
            auto idx = torch::linspace(
                0.0, static_cast<double>(total - 1), max_real_pcd_points_,
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
            ).round().to(torch::kLong);
            real_pcd_points_accum_cpu_ = real_pcd_points_accum_cpu_.index_select(0, idx).contiguous();
        }
    }

    // ── 10) Re-register with optimizer (new rows appended) ───────────────────
    VOXEL_MODEL_TENSORS_TO_VEC

    if (!pending_artificial_insert_rr_entity_path_.empty()) {
        pending_artificial_insert_rr_entity_path_.clear();
        pending_insert_rendered_depth_candidate_source_kind_ = 0;
        pending_insert_rendered_depth_candidate_as_real_protected_ = false;
    }

    if (!pending_real_insert_rr_entity_path_.empty()) {
        pending_real_insert_rr_entity_path_.clear();
    }
    pending_insert_rendered_depth_candidate_ = false;
    pending_insert_rendered_depth_candidate_source_kind_ = 0;
    pending_insert_rendered_depth_candidate_as_real_protected_ = false;

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

    // Reuse the main SVRaster-aware pipeline
    increasePcd(
        points, cols, iteration, cams);
}

bool VoxelModel::refreshDenseCoreBBFromCurrentVoxels()
{
    auto set_dense_core_from_center_radius = [&](const torch::Tensor& center_cpu_f32, float radius)->bool {
        if (!center_cpu_f32.defined() || center_cpu_f32.numel() != 3 || !std::isfinite(radius) || radius <= 0.0f) {
            return false;
        }
        auto dev = this->center_.defined() ? this->center_.device() : torch::Device(torch::kCUDA);
        auto core_center = center_cpu_f32.to(dev).to(torch::kFloat32).contiguous().view({3});
        auto core_radius_t = torch::full(
            {3}, radius, torch::dtype(torch::kFloat32).device(dev));
        dense_core_bb_min_ = (core_center - core_radius_t).contiguous();
        dense_core_bb_max_ = (core_center + core_radius_t).contiguous();
        has_dense_core_bb_ = true;
        return true;
    };

    torch::Tensor pts_cpu;
    // 1) Prefer accumulated raw real PCD history.
    if (real_pcd_points_accum_cpu_.defined() && real_pcd_points_accum_cpu_.numel() > 0) {
        pts_cpu = real_pcd_points_accum_cpu_.to(torch::kCPU).to(torch::kFloat32).contiguous();
    }

    // 2) Fallback: current REAL voxel centers (artificial excluded).
    if ((!pts_cpu.defined() || pts_cpu.numel() == 0 || pts_cpu.size(0) < 8) &&
        this->center_.defined() && this->center_.numel() > 0 &&
        this->center_.dim() == 2 && this->center_.size(1) == 3) {
        auto centers_cpu = this->center_.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
        torch::Tensor keep_mask_cpu = torch::ones(
            {centers_cpu.size(0)},
            torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
        if (is_artificial_voxel_.defined() && is_artificial_voxel_.size(0) == centers_cpu.size(0)) {
            auto real_mask_cpu = (~is_artificial_voxel_.to(torch::kCPU).to(torch::kBool).contiguous());
            keep_mask_cpu = (keep_mask_cpu & real_mask_cpu).to(torch::kBool);
        }
        auto keep_idx = torch::nonzero(keep_mask_cpu).view({-1}).contiguous();
        if (keep_idx.numel() > 0) {
            centers_cpu = centers_cpu.index_select(0, keep_idx).contiguous();
        } else {
            centers_cpu = torch::empty(
                {0, 3},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        }
        if (centers_cpu.defined() && centers_cpu.size(0) >= 8) {
            pts_cpu = centers_cpu;
        }
    }

    if (!pts_cpu.defined() || pts_cpu.numel() == 0 || pts_cpu.size(0) < 8) {
        return false;
    }

    // 3) SVRaster-style precheck (same equations) to detect "no crossing" before Python call.
    auto pts_f32 = pts_cpu.to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto center_cpu = std::get<0>(pts_f32.median(/*dim=*/0, /*keepdim=*/false)).contiguous(); // [3], CPU
    auto dist = std::get<0>(
        (pts_f32 - center_cpu.view({1, 3})).abs().max(/*dim=*/1, /*keepdim=*/false))
                    .to(torch::kFloat32)
                    .contiguous(); // [N]
    dist = std::get<0>(dist.sort(/*dim=*/0)).contiguous();

    const int64_t n = dist.size(0);
    if (n <= 0) {
        // std::cout << "[dense_core/refresh] failed: empty dist after preprocessing.\n";
        return false;
    }
    auto idx = torch::arange(
        1, n + 1,
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto nonzero = (dist > 0.0f).to(torch::kFloat32);
    auto density = idx * nonzero / ((2.0f * dist).pow(3) + 1e-6f);

    int64_t begin_idx = static_cast<int64_t>(std::llround(static_cast<double>(n) * 0.05));
    begin_idx = std::max<int64_t>(0, std::min<int64_t>(n - 1, begin_idx));

    auto tail = density.index({torch::indexing::Slice(begin_idx, torch::indexing::None)}).contiguous();
    if (tail.numel() <= 0) {
        // std::cout << "[dense_core/refresh] failed: empty density tail (begin_idx="
        //           << begin_idx << ", n=" << n << ").\n";
        return false;
    }

    const int64_t max_idx = begin_idx + tail.argmax().item<int64_t>();
    const float max_density = density.index({max_idx}).item<float>();
    const float target_density = dense_core_pcd_density_rate_ * max_density;
    auto right = density.index({torch::indexing::Slice(max_idx, torch::indexing::None)}).contiguous();
    auto below = torch::nonzero(right < target_density).view({-1}).contiguous();
    const int64_t crossing_count = below.numel();
    const float right_min = right.min().item<float>();
    const float right_last = right.index({-1}).item<float>();

    if (crossing_count == 0) {
        // std::cout << "[dense_core/refresh] no_density_crossing at rate="
        //           << dense_core_pcd_density_rate_
        //           << "; skipping python heuristic this round.\n";
        return false;
    }

    // 4) SVRaster pcd heuristic once the precheck says crossing exists.
    if (auto bound = mainSceneBoundPcdHeuristicCpp(
            pts_cpu.contiguous(),
            dense_core_pcd_density_rate_)) {
        auto core_center = bound->first.to(torch::kCPU).to(torch::kFloat32).contiguous().view({3});
        const float core_radius = bound->second;
        if (set_dense_core_from_center_radius(core_center, core_radius)) {
            // std::cout << "[dense_core/refresh] source=cpp points=" << pts_cpu.size(0)
            //           << " rate=" << dense_core_pcd_density_rate_
            //           << " radius=" << core_radius << "\n";
            // std::cout << "[dense_core/refresh] dense_core_bb_min=" << dense_core_bb_min_
            //           << " dense_core_bb_max=" << dense_core_bb_max_ << std::endl;
            return true;
        }
    }

    // std::cout << "[dense_core/refresh] failed to estimate dense-core bbox.\n";
    return false;
}

void VoxelModel::setGeometricallyUnstableMask(const torch::Tensor& mask)
{
    if (!center_.defined() || center_.numel() == 0) {
        geometrically_unstable_voxel_ = torch::empty(
            {0},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
        return;
    }

    auto m = mask;
    if (!m.defined()) {
        geometrically_unstable_voxel_ = torch::zeros(
            {center_.size(0)},
            torch::TensorOptions().dtype(torch::kBool).device(center_.device()));
        return;
    }
    if (m.dim() == 2 && m.size(1) == 1) {
        m = m.squeeze(1);
    }
    m = m.to(center_.device()).to(torch::kBool).contiguous().view({-1});
    if (m.numel() != center_.size(0)) {
        auto aligned = torch::zeros(
            {center_.size(0)},
            torch::TensorOptions().dtype(torch::kBool).device(center_.device()));
        const int64_t copy_n = std::min<int64_t>(aligned.size(0), m.size(0));
        if (copy_n > 0) {
            aligned.index_put_(
                {torch::indexing::Slice(0, copy_n)},
                m.index({torch::indexing::Slice(0, copy_n)}));
        }
        m = aligned;
    }
    geometrically_unstable_voxel_ = m.contiguous();
}

void VoxelModel::promoteRenderedDepthCandidates(const torch::Tensor& promote_mask)
{
    if (!center_.defined() || center_.numel() == 0) {
        return;
    }

    auto mask = promote_mask;
    if (!mask.defined()) {
        return;
    }
    if (mask.dim() == 2 && mask.size(1) == 1) {
        mask = mask.squeeze(1);
    }
    mask = mask.to(center_.device()).to(torch::kBool).contiguous().view({-1});
    if (mask.numel() != center_.size(0)) {
        return;
    }

    auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(center_.device());
    auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(center_.device());
    if (!is_artificial_voxel_.defined() || is_artificial_voxel_.size(0) != center_.size(0)) {
        is_artificial_voxel_ = torch::zeros({center_.size(0)}, bool_opts);
    }
    if (!is_promoted_artificial_voxel_.defined() || is_promoted_artificial_voxel_.size(0) != center_.size(0)) {
        is_promoted_artificial_voxel_ = torch::zeros({center_.size(0)}, bool_opts);
    }
    if (!rendered_depth_candidate_voxel_.defined() ||
        rendered_depth_candidate_voxel_.size(0) != center_.size(0)) {
        rendered_depth_candidate_voxel_ = torch::zeros({center_.size(0)}, bool_opts);
    }
    if (!rendered_depth_candidate_support_count_.defined() ||
        rendered_depth_candidate_support_count_.size(0) != center_.size(0)) {
        rendered_depth_candidate_support_count_ = torch::zeros({center_.size(0)}, i32_opts);
    }
    if (!rendered_depth_candidate_last_seen_kf_.defined() ||
        rendered_depth_candidate_last_seen_kf_.size(0) != center_.size(0)) {
        rendered_depth_candidate_last_seen_kf_ =
            torch::full({center_.size(0)}, static_cast<int32_t>(-1), i32_opts);
    }
    if (!rendered_depth_candidate_source_kind_.defined() ||
        rendered_depth_candidate_source_kind_.size(0) != center_.size(0)) {
        rendered_depth_candidate_source_kind_ = torch::zeros({center_.size(0)}, i32_opts);
    }

    auto effective_mask =
        (mask &
         rendered_depth_candidate_voxel_.to(center_.device()).to(torch::kBool) &
         is_artificial_voxel_.to(center_.device()).to(torch::kBool)).contiguous();
    auto idx = torch::nonzero(effective_mask).view({-1});
    if (idx.numel() == 0) {
        return;
    }

    is_artificial_voxel_.index_put_({idx}, false);
    is_promoted_artificial_voxel_.index_put_({idx}, true);
    rendered_depth_candidate_voxel_.index_put_({idx}, false);
    total_promoted_artificial_voxels_ += idx.size(0);
}

void VoxelModel::createTrainer(float geo_lr, float sh0_lr, float shs_lr,
                               float beta1, float beta2, float eps,
                               const std::vector<int>& milestones,
                               float gamma)
{
    optimizer_geo_lr_ = geo_lr;
    optimizer_sh0_lr_ = sh0_lr;
    optimizer_shs_lr_ = shs_lr;
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

    _geo_grid_pts_ = _geo_grid_pts_.contiguous().detach().requires_grad_(true);
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
    sh0_.mutable_grad() = torch::Tensor();
    shs_.mutable_grad() = torch::Tensor();
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
        ADAM_STEP::unbiased_adam_step(
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
    }
}

void VoxelModel::pruning(const torch::Tensor& prune_mask) {
    // accept [N] or [N,1] bool/byte
    auto mask = prune_mask.to(torch::kBool).to(device_type_).contiguous();
    if (mask.dim() == 2 && mask.size(1) == 1) {
        mask = mask.squeeze(1);
    }
    TORCH_CHECK(mask.dim() == 1, "pruning: mask must be [N] or [N,1]");
    const int64_t N_before = mask.size(0);

    if (!is_artificial_voxel_.defined() || is_artificial_voxel_.size(0) != N_before) {
        is_artificial_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (is_artificial_voxel_.device() != mask.device()) {
        is_artificial_voxel_ = is_artificial_voxel_.to(mask.device());
    }
    if (!is_orb_voxel_.defined() || is_orb_voxel_.size(0) != N_before) {
        is_orb_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (is_orb_voxel_.device() != mask.device()) {
        is_orb_voxel_ = is_orb_voxel_.to(mask.device());
    }
    if (!is_inactive_geo_voxel_.defined() || is_inactive_geo_voxel_.size(0) != N_before) {
        is_inactive_geo_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (is_inactive_geo_voxel_.device() != mask.device()) {
        is_inactive_geo_voxel_ = is_inactive_geo_voxel_.to(mask.device());
    }
    if (!is_rgbd_fill_render_holes_voxel_.defined() ||
        is_rgbd_fill_render_holes_voxel_.size(0) != N_before) {
        is_rgbd_fill_render_holes_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (is_rgbd_fill_render_holes_voxel_.device() != mask.device()) {
        is_rgbd_fill_render_holes_voxel_ = is_rgbd_fill_render_holes_voxel_.to(mask.device());
    }
    if (!is_depthanything_fill_holes_voxel_.defined() ||
        is_depthanything_fill_holes_voxel_.size(0) != N_before) {
        is_depthanything_fill_holes_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (is_depthanything_fill_holes_voxel_.device() != mask.device()) {
        is_depthanything_fill_holes_voxel_ =
            is_depthanything_fill_holes_voxel_.to(mask.device());
    }
    if (!is_promoted_artificial_voxel_.defined() || is_promoted_artificial_voxel_.size(0) != N_before) {
        is_promoted_artificial_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (is_promoted_artificial_voxel_.device() != mask.device()) {
        is_promoted_artificial_voxel_ = is_promoted_artificial_voxel_.to(mask.device());
    }
    if (!exist_since_iter_.defined() || exist_since_iter_.size(0) != N_before) {
        exist_since_iter_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
    } else if (exist_since_iter_.device() != mask.device()) {
        exist_since_iter_ = exist_since_iter_.to(mask.device());
    }
    if (!exist_since_kf_.defined() || exist_since_kf_.size(0) != N_before) {
        exist_since_kf_ = torch::full(
            {N_before},
            static_cast<int32_t>(-1),
            torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
    } else if (exist_since_kf_.device() != mask.device()) {
        exist_since_kf_ = exist_since_kf_.to(mask.device());
    }
    if (!geometrically_unstable_voxel_.defined() || geometrically_unstable_voxel_.size(0) != N_before) {
        geometrically_unstable_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (geometrically_unstable_voxel_.device() != mask.device()) {
        geometrically_unstable_voxel_ = geometrically_unstable_voxel_.to(mask.device());
    }
    if (!rendered_depth_candidate_voxel_.defined() ||
        rendered_depth_candidate_voxel_.size(0) != N_before) {
        rendered_depth_candidate_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (rendered_depth_candidate_voxel_.device() != mask.device()) {
        rendered_depth_candidate_voxel_ = rendered_depth_candidate_voxel_.to(mask.device());
    }
    if (!rendered_depth_candidate_support_count_.defined() ||
        rendered_depth_candidate_support_count_.size(0) != N_before) {
        rendered_depth_candidate_support_count_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
    } else if (rendered_depth_candidate_support_count_.device() != mask.device()) {
        rendered_depth_candidate_support_count_ = rendered_depth_candidate_support_count_.to(mask.device());
    }
    if (!rendered_depth_candidate_last_seen_kf_.defined() ||
        rendered_depth_candidate_last_seen_kf_.size(0) != N_before) {
        rendered_depth_candidate_last_seen_kf_ = torch::full(
            {N_before},
            static_cast<int32_t>(-1),
            torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
    } else if (rendered_depth_candidate_last_seen_kf_.device() != mask.device()) {
        rendered_depth_candidate_last_seen_kf_ = rendered_depth_candidate_last_seen_kf_.to(mask.device());
    }
    if (!rendered_depth_candidate_source_kind_.defined() ||
        rendered_depth_candidate_source_kind_.size(0) != N_before) {
        rendered_depth_candidate_source_kind_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
    } else if (rendered_depth_candidate_source_kind_.device() != mask.device()) {
        rendered_depth_candidate_source_kind_ = rendered_depth_candidate_source_kind_.to(mask.device());
    }
    auto art_before = is_artificial_voxel_.to(torch::kBool).contiguous();
    auto orb_before = is_orb_voxel_.to(torch::kBool).contiguous();
    auto inactive_geo_before = is_inactive_geo_voxel_.to(torch::kBool).contiguous();
    auto rgbd_fill_render_holes_before =
        is_rgbd_fill_render_holes_voxel_.to(torch::kBool).contiguous();
    auto depthanything_fill_holes_before =
        is_depthanything_fill_holes_voxel_.to(torch::kBool).contiguous();
    auto promoted_before = is_promoted_artificial_voxel_.to(torch::kBool).contiguous();
    auto exist_since_before = exist_since_iter_.to(torch::kInt32).contiguous();
    auto exist_since_kf_before = exist_since_kf_.to(torch::kInt32).contiguous();
    auto unstable_before = geometrically_unstable_voxel_.to(torch::kBool).contiguous();
    auto rendered_depth_before = rendered_depth_candidate_voxel_.to(torch::kBool).contiguous();
    auto rendered_depth_support_before = rendered_depth_candidate_support_count_.to(torch::kInt32).contiguous();
    auto rendered_depth_last_seen_before = rendered_depth_candidate_last_seen_kf_.to(torch::kInt32).contiguous();
    const int64_t n_art_before = art_before.sum().item<int64_t>();
    const int64_t n_promoted_before = promoted_before.sum().item<int64_t>();
    const int64_t n_prune_total = mask.sum().item<int64_t>();
    const int64_t n_prune_art = (mask & art_before).sum().item<int64_t>();
    const int64_t n_prune_real = n_prune_total - n_prune_art;
    auto old_octpath = this->oct_path_.contiguous();
    auto old_octlevel = this->oct_level_.contiguous();
    auto old_key_cpu = (old_octpath.view({-1}).to(torch::kInt64).mul(256)
                      + old_octlevel.view({-1}).to(torch::kInt64))
                      .to(torch::kCPU).contiguous();
    auto old_art_cpu = art_before.to(torch::kCPU).contiguous();
    auto old_orb_cpu = orb_before.to(torch::kCPU).contiguous();
    auto old_inactive_geo_cpu = inactive_geo_before.to(torch::kCPU).contiguous();
    auto old_rgbd_fill_render_holes_cpu =
        rgbd_fill_render_holes_before.to(torch::kCPU).contiguous();
    auto old_depthanything_fill_holes_cpu =
        depthanything_fill_holes_before.to(torch::kCPU).contiguous();
    auto old_promoted_cpu = promoted_before.to(torch::kCPU).contiguous();
    auto old_exist_since_cpu = exist_since_before.to(torch::kCPU).contiguous();
    auto old_exist_since_kf_cpu = exist_since_kf_before.to(torch::kCPU).contiguous();
    auto old_unstable_cpu = unstable_before.to(torch::kCPU).contiguous();
    auto old_rendered_depth_cpu = rendered_depth_before.to(torch::kCPU).contiguous();
    auto old_rendered_depth_support_cpu = rendered_depth_support_before.to(torch::kCPU).contiguous();
    auto old_rendered_depth_last_seen_cpu = rendered_depth_last_seen_before.to(torch::kCPU).contiguous();
    auto old_rendered_depth_source_cpu =
        rendered_depth_candidate_source_kind_.to(torch::kCPU).to(torch::kInt32).contiguous();

    auto kept_idx = torch::nonzero(~mask).view({-1}).to(torch::kLong).contiguous();
    if (kept_idx.numel() == 0) {
        return;
    }

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
    {
        auto kept_sdf = old_vox_sdf_values
            .index_select(0, kept_idx.to(old_vox_sdf_values.device()))
            .contiguous();
        auto kept_weights = old_vox_sdf_weights
            .index_select(0, kept_idx.to(old_vox_sdf_weights.device()))
            .contiguous();
        rebuildSvrasterSdfFieldFromVoxelCorners_(kept_sdf, kept_weights);
    }
    this->max_w_ = torch::zeros(
        {center_.size(0), 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

    auto new_key_cpu = (this->oct_path_.view({-1}).to(torch::kInt64).mul(256)
                      + this->oct_level_.view({-1}).to(torch::kInt64))
                      .to(torch::kCPU).contiguous();
    const int64_t N_after = new_key_cpu.size(0);

    auto art_after_cpu = torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto orb_after_cpu = torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto inactive_geo_after_cpu = torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto rgbd_fill_render_holes_after_cpu =
        torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto depthanything_fill_holes_after_cpu =
        torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto promoted_after_cpu = torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto exist_since_after_cpu = torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto exist_since_kf_after_cpu = torch::full(
        {N_after}, static_cast<int32_t>(-1),
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto unstable_after_cpu = torch::zeros(
        {N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto rendered_depth_after_cpu = torch::zeros(
        {N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto rendered_depth_support_after_cpu = torch::zeros(
        {N_after}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto rendered_depth_last_seen_after_cpu = torch::full(
        {N_after}, static_cast<int32_t>(-1),
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto rendered_depth_source_after_cpu = torch::zeros(
        {N_after}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));

    std::unordered_map<int64_t, int64_t> key_to_old_idx;
    key_to_old_idx.reserve(static_cast<size_t>(old_key_cpu.size(0) * 2 + 1));
    const int64_t* old_key_ptr = old_key_cpu.data_ptr<int64_t>();
    for (int64_t i = 0; i < old_key_cpu.size(0); ++i) {
        key_to_old_idx.emplace(old_key_ptr[i], i);
    }

    const bool* old_art_ptr = old_art_cpu.data_ptr<bool>();
    const bool* old_orb_ptr = old_orb_cpu.data_ptr<bool>();
    const bool* old_inactive_geo_ptr = old_inactive_geo_cpu.data_ptr<bool>();
    const bool* old_rgbd_fill_render_holes_ptr = old_rgbd_fill_render_holes_cpu.data_ptr<bool>();
    const bool* old_depthanything_fill_holes_ptr = old_depthanything_fill_holes_cpu.data_ptr<bool>();
    const bool* old_promoted_ptr = old_promoted_cpu.data_ptr<bool>();
    const int32_t* old_exist_since_ptr = old_exist_since_cpu.data_ptr<int32_t>();
    const int32_t* old_exist_since_kf_ptr = old_exist_since_kf_cpu.data_ptr<int32_t>();
    const bool* old_unstable_ptr = old_unstable_cpu.data_ptr<bool>();
    const bool* old_rendered_depth_ptr = old_rendered_depth_cpu.data_ptr<bool>();
    const int32_t* old_rendered_depth_support_ptr = old_rendered_depth_support_cpu.data_ptr<int32_t>();
    const int32_t* old_rendered_depth_last_seen_ptr = old_rendered_depth_last_seen_cpu.data_ptr<int32_t>();
    const int32_t* old_rendered_depth_source_ptr = old_rendered_depth_source_cpu.data_ptr<int32_t>();
    const int64_t* new_key_ptr = new_key_cpu.data_ptr<int64_t>();
    bool* art_after_ptr = art_after_cpu.data_ptr<bool>();
    bool* orb_after_ptr = orb_after_cpu.data_ptr<bool>();
    bool* inactive_geo_after_ptr = inactive_geo_after_cpu.data_ptr<bool>();
    bool* rgbd_fill_render_holes_after_ptr = rgbd_fill_render_holes_after_cpu.data_ptr<bool>();
    bool* depthanything_fill_holes_after_ptr = depthanything_fill_holes_after_cpu.data_ptr<bool>();
    bool* promoted_after_ptr = promoted_after_cpu.data_ptr<bool>();
    int32_t* exist_since_after_ptr = exist_since_after_cpu.data_ptr<int32_t>();
    int32_t* exist_since_kf_after_ptr = exist_since_kf_after_cpu.data_ptr<int32_t>();
    bool* unstable_after_ptr = unstable_after_cpu.data_ptr<bool>();
    bool* rendered_depth_after_ptr = rendered_depth_after_cpu.data_ptr<bool>();
    int32_t* rendered_depth_support_after_ptr = rendered_depth_support_after_cpu.data_ptr<int32_t>();
    int32_t* rendered_depth_last_seen_after_ptr = rendered_depth_last_seen_after_cpu.data_ptr<int32_t>();
    int32_t* rendered_depth_source_after_ptr = rendered_depth_source_after_cpu.data_ptr<int32_t>();

    int64_t matched_by_key = 0;
    for (int64_t i = 0; i < N_after; ++i) {
        auto it = key_to_old_idx.find(new_key_ptr[i]);
        if (it == key_to_old_idx.end()) {
            continue;
        }
        const int64_t old_idx = it->second;
        art_after_ptr[i] = old_art_ptr[old_idx];
        orb_after_ptr[i] = old_orb_ptr[old_idx];
        inactive_geo_after_ptr[i] = old_inactive_geo_ptr[old_idx];
        rgbd_fill_render_holes_after_ptr[i] = old_rgbd_fill_render_holes_ptr[old_idx];
        depthanything_fill_holes_after_ptr[i] = old_depthanything_fill_holes_ptr[old_idx];
        promoted_after_ptr[i] = old_promoted_ptr[old_idx];
        exist_since_after_ptr[i] = old_exist_since_ptr[old_idx];
        exist_since_kf_after_ptr[i] = old_exist_since_kf_ptr[old_idx];
        unstable_after_ptr[i] = old_unstable_ptr[old_idx];
        rendered_depth_after_ptr[i] = old_rendered_depth_ptr[old_idx];
        rendered_depth_support_after_ptr[i] = old_rendered_depth_support_ptr[old_idx];
        rendered_depth_last_seen_after_ptr[i] = old_rendered_depth_last_seen_ptr[old_idx];
        rendered_depth_source_after_ptr[i] = old_rendered_depth_source_ptr[old_idx];
        ++matched_by_key;
    }

    if (matched_by_key != N_after) {
        std::cout << "[PRUNE/provenance] WARNING: unmatched voxels after key remap: "
                  << (N_after - matched_by_key) << "/" << N_after << "\n";
    }

    is_artificial_voxel_ = art_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    is_orb_voxel_ = orb_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    is_inactive_geo_voxel_ = inactive_geo_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    is_rgbd_fill_render_holes_voxel_ =
        rgbd_fill_render_holes_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    is_depthanything_fill_holes_voxel_ =
        depthanything_fill_holes_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    is_promoted_artificial_voxel_ = promoted_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    exist_since_iter_ = exist_since_after_cpu.to(device_type_).to(torch::kInt32).contiguous();
    exist_since_kf_ = exist_since_kf_after_cpu.to(device_type_).to(torch::kInt32).contiguous();
    geometrically_unstable_voxel_ = unstable_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    rendered_depth_candidate_voxel_ =
        rendered_depth_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    rendered_depth_candidate_support_count_ =
        rendered_depth_support_after_cpu.to(device_type_).to(torch::kInt32).contiguous();
    rendered_depth_candidate_last_seen_kf_ =
        rendered_depth_last_seen_after_cpu.to(device_type_).to(torch::kInt32).contiguous();
    rendered_depth_candidate_source_kind_ =
        rendered_depth_source_after_cpu.to(device_type_).to(torch::kInt32).contiguous();
    const int64_t n_art_after = is_artificial_voxel_.sum().item<int64_t>();
    const int64_t n_promoted_after = is_promoted_artificial_voxel_.sum().item<int64_t>();
    (void)n_prune_total;
    (void)n_prune_art;
    (void)n_prune_real;
    (void)n_art_before;
    (void)n_art_after;
    (void)n_promoted_before;
    (void)n_promoted_after;
}

void VoxelModel::subdividing(const torch::Tensor& subdivide_mask) {
    auto mask = subdivide_mask.to(torch::kBool).to(device_type_).contiguous();
    if (mask.dim() == 2 && mask.size(1) == 1) {
        mask = mask.squeeze(1);
    }
    TORCH_CHECK(mask.dim() == 1, "subdividing: mask must be [N] or [N,1]");
    const int64_t N_before = mask.size(0);

    if (!is_artificial_voxel_.defined() || is_artificial_voxel_.size(0) != N_before) {
        is_artificial_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (is_artificial_voxel_.device() != mask.device()) {
        is_artificial_voxel_ = is_artificial_voxel_.to(mask.device());
    }
    if (!is_orb_voxel_.defined() || is_orb_voxel_.size(0) != N_before) {
        is_orb_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (is_orb_voxel_.device() != mask.device()) {
        is_orb_voxel_ = is_orb_voxel_.to(mask.device());
    }
    if (!is_inactive_geo_voxel_.defined() || is_inactive_geo_voxel_.size(0) != N_before) {
        is_inactive_geo_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (is_inactive_geo_voxel_.device() != mask.device()) {
        is_inactive_geo_voxel_ = is_inactive_geo_voxel_.to(mask.device());
    }
    if (!is_rgbd_fill_render_holes_voxel_.defined() ||
        is_rgbd_fill_render_holes_voxel_.size(0) != N_before) {
        is_rgbd_fill_render_holes_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (is_rgbd_fill_render_holes_voxel_.device() != mask.device()) {
        is_rgbd_fill_render_holes_voxel_ = is_rgbd_fill_render_holes_voxel_.to(mask.device());
    }
    if (!is_depthanything_fill_holes_voxel_.defined() ||
        is_depthanything_fill_holes_voxel_.size(0) != N_before) {
        is_depthanything_fill_holes_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (is_depthanything_fill_holes_voxel_.device() != mask.device()) {
        is_depthanything_fill_holes_voxel_ =
            is_depthanything_fill_holes_voxel_.to(mask.device());
    }
    if (!is_promoted_artificial_voxel_.defined() || is_promoted_artificial_voxel_.size(0) != N_before) {
        is_promoted_artificial_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (is_promoted_artificial_voxel_.device() != mask.device()) {
        is_promoted_artificial_voxel_ = is_promoted_artificial_voxel_.to(mask.device());
    }
    if (!exist_since_iter_.defined() || exist_since_iter_.size(0) != N_before) {
        exist_since_iter_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
    } else if (exist_since_iter_.device() != mask.device()) {
        exist_since_iter_ = exist_since_iter_.to(mask.device());
    }
    if (!exist_since_kf_.defined() || exist_since_kf_.size(0) != N_before) {
        exist_since_kf_ = torch::full(
            {N_before},
            static_cast<int32_t>(-1),
            torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
    } else if (exist_since_kf_.device() != mask.device()) {
        exist_since_kf_ = exist_since_kf_.to(mask.device());
    }
    if (!geometrically_unstable_voxel_.defined() || geometrically_unstable_voxel_.size(0) != N_before) {
        geometrically_unstable_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (geometrically_unstable_voxel_.device() != mask.device()) {
        geometrically_unstable_voxel_ = geometrically_unstable_voxel_.to(mask.device());
    }
    if (!rendered_depth_candidate_voxel_.defined() ||
        rendered_depth_candidate_voxel_.size(0) != N_before) {
        rendered_depth_candidate_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (rendered_depth_candidate_voxel_.device() != mask.device()) {
        rendered_depth_candidate_voxel_ = rendered_depth_candidate_voxel_.to(mask.device());
    }
    if (!rendered_depth_candidate_support_count_.defined() ||
        rendered_depth_candidate_support_count_.size(0) != N_before) {
        rendered_depth_candidate_support_count_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
    } else if (rendered_depth_candidate_support_count_.device() != mask.device()) {
        rendered_depth_candidate_support_count_ = rendered_depth_candidate_support_count_.to(mask.device());
    }
    if (!rendered_depth_candidate_last_seen_kf_.defined() ||
        rendered_depth_candidate_last_seen_kf_.size(0) != N_before) {
        rendered_depth_candidate_last_seen_kf_ = torch::full(
            {N_before},
            static_cast<int32_t>(-1),
            torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
    } else if (rendered_depth_candidate_last_seen_kf_.device() != mask.device()) {
        rendered_depth_candidate_last_seen_kf_ = rendered_depth_candidate_last_seen_kf_.to(mask.device());
    }
    if (!rendered_depth_candidate_source_kind_.defined() ||
        rendered_depth_candidate_source_kind_.size(0) != N_before) {
        rendered_depth_candidate_source_kind_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
    } else if (rendered_depth_candidate_source_kind_.device() != mask.device()) {
        rendered_depth_candidate_source_kind_ = rendered_depth_candidate_source_kind_.to(mask.device());
    }

    auto art_before = is_artificial_voxel_.to(torch::kBool).contiguous();
    auto orb_before = is_orb_voxel_.to(torch::kBool).contiguous();
    auto inactive_geo_before = is_inactive_geo_voxel_.to(torch::kBool).contiguous();
    auto rgbd_fill_render_holes_before =
        is_rgbd_fill_render_holes_voxel_.to(torch::kBool).contiguous();
    auto depthanything_fill_holes_before =
        is_depthanything_fill_holes_voxel_.to(torch::kBool).contiguous();
    auto promoted_before = is_promoted_artificial_voxel_.to(torch::kBool).contiguous();
    auto exist_since_before = exist_since_iter_.to(torch::kInt32).contiguous();
    auto exist_since_kf_before = exist_since_kf_.to(torch::kInt32).contiguous();
    auto unstable_before = geometrically_unstable_voxel_.to(torch::kBool).contiguous();
    auto rendered_depth_before = rendered_depth_candidate_voxel_.to(torch::kBool).contiguous();
    auto rendered_depth_support_before = rendered_depth_candidate_support_count_.to(torch::kInt32).contiguous();
    auto rendered_depth_last_seen_before = rendered_depth_candidate_last_seen_kf_.to(torch::kInt32).contiguous();
    const int64_t n_art_before = art_before.sum().item<int64_t>();
    const int64_t n_promoted_before = promoted_before.sum().item<int64_t>();

    auto subdiv_idx = torch::nonzero(mask).view({-1});
    const int64_t n_subdiv_parents = subdiv_idx.size(0);
    int64_t n_subdiv_art_parents = 0;
    int64_t n_subdiv_promoted_parents = 0;
    if (n_subdiv_parents > 0) {
        auto art_sel = art_before.index_select(0, subdiv_idx).contiguous();
        auto promoted_sel = promoted_before.index_select(0, subdiv_idx).contiguous();
        n_subdiv_art_parents = art_sel.sum().item<int64_t>();
        n_subdiv_promoted_parents = promoted_sel.sum().item<int64_t>();
    }

    auto old_octpath = this->oct_path_.contiguous();
    auto old_octlevel = this->oct_level_.contiguous();
    auto old_key_cpu = (old_octpath.view({-1}).to(torch::kInt64).mul(256)
                      + old_octlevel.view({-1}).to(torch::kInt64))
                      .to(torch::kCPU).contiguous();
    auto old_art_cpu = art_before.to(torch::kCPU).contiguous();
    auto old_orb_cpu = orb_before.to(torch::kCPU).contiguous();
    auto old_inactive_geo_cpu = inactive_geo_before.to(torch::kCPU).contiguous();
    auto old_rgbd_fill_render_holes_cpu =
        rgbd_fill_render_holes_before.to(torch::kCPU).contiguous();
    auto old_depthanything_fill_holes_cpu =
        depthanything_fill_holes_before.to(torch::kCPU).contiguous();
    auto old_promoted_cpu = promoted_before.to(torch::kCPU).contiguous();
    auto old_exist_since_cpu = exist_since_before.to(torch::kCPU).contiguous();
    auto old_exist_since_kf_cpu = exist_since_kf_before.to(torch::kCPU).contiguous();
    auto old_unstable_cpu = unstable_before.to(torch::kCPU).contiguous();
    auto old_rendered_depth_cpu = rendered_depth_before.to(torch::kCPU).contiguous();
    auto old_rendered_depth_support_cpu = rendered_depth_support_before.to(torch::kCPU).contiguous();
    auto old_rendered_depth_last_seen_cpu = rendered_depth_last_seen_before.to(torch::kCPU).contiguous();
    auto old_rendered_depth_source_cpu =
        rendered_depth_candidate_source_kind_.to(torch::kCPU).to(torch::kInt32).contiguous();

    auto kept_idx = torch::nonzero(~mask).view({-1}).to(torch::kLong).contiguous();
    if (subdiv_idx.numel() == 0) {
        return;
    }

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

    auto new_key_cpu = (this->oct_path_.view({-1}).to(torch::kInt64).mul(256)
                      + this->oct_level_.view({-1}).to(torch::kInt64))
                      .to(torch::kCPU).contiguous();
    const int64_t N_after = new_key_cpu.size(0);

    auto art_after_cpu = torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto orb_after_cpu = torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto inactive_geo_after_cpu = torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto rgbd_fill_render_holes_after_cpu =
        torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto depthanything_fill_holes_after_cpu =
        torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto promoted_after_cpu = torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto exist_since_after_cpu = torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto exist_since_kf_after_cpu = torch::full(
        {N_after}, static_cast<int32_t>(-1),
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto unstable_after_cpu = torch::zeros(
        {N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto rendered_depth_after_cpu = torch::zeros(
        {N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto rendered_depth_support_after_cpu = torch::zeros(
        {N_after}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto rendered_depth_last_seen_after_cpu = torch::full(
        {N_after}, static_cast<int32_t>(-1),
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto rendered_depth_source_after_cpu = torch::zeros(
        {N_after}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));

    std::unordered_map<int64_t, int64_t> key_to_old_idx;
    key_to_old_idx.reserve(static_cast<size_t>(old_key_cpu.size(0) * 2 + 1));
    const int64_t* old_key_ptr = old_key_cpu.data_ptr<int64_t>();
    for (int64_t i = 0; i < old_key_cpu.size(0); ++i) {
        key_to_old_idx.emplace(old_key_ptr[i], i);
    }

    const bool* old_art_ptr = old_art_cpu.data_ptr<bool>();
    const bool* old_orb_ptr = old_orb_cpu.data_ptr<bool>();
    const bool* old_inactive_geo_ptr = old_inactive_geo_cpu.data_ptr<bool>();
    const bool* old_rgbd_fill_render_holes_ptr = old_rgbd_fill_render_holes_cpu.data_ptr<bool>();
    const bool* old_depthanything_fill_holes_ptr = old_depthanything_fill_holes_cpu.data_ptr<bool>();
    const bool* old_promoted_ptr = old_promoted_cpu.data_ptr<bool>();
    const int32_t* old_exist_since_ptr = old_exist_since_cpu.data_ptr<int32_t>();
    const int32_t* old_exist_since_kf_ptr = old_exist_since_kf_cpu.data_ptr<int32_t>();
    const bool* old_unstable_ptr = old_unstable_cpu.data_ptr<bool>();
    const bool* old_rendered_depth_ptr = old_rendered_depth_cpu.data_ptr<bool>();
    const int32_t* old_rendered_depth_support_ptr = old_rendered_depth_support_cpu.data_ptr<int32_t>();
    const int32_t* old_rendered_depth_last_seen_ptr = old_rendered_depth_last_seen_cpu.data_ptr<int32_t>();
    const int32_t* old_rendered_depth_source_ptr = old_rendered_depth_source_cpu.data_ptr<int32_t>();
    const int64_t* new_key_ptr = new_key_cpu.data_ptr<int64_t>();
    bool* art_after_ptr = art_after_cpu.data_ptr<bool>();
    bool* orb_after_ptr = orb_after_cpu.data_ptr<bool>();
    bool* inactive_geo_after_ptr = inactive_geo_after_cpu.data_ptr<bool>();
    bool* rgbd_fill_render_holes_after_ptr = rgbd_fill_render_holes_after_cpu.data_ptr<bool>();
    bool* depthanything_fill_holes_after_ptr = depthanything_fill_holes_after_cpu.data_ptr<bool>();
    bool* promoted_after_ptr = promoted_after_cpu.data_ptr<bool>();
    int32_t* exist_since_after_ptr = exist_since_after_cpu.data_ptr<int32_t>();
    int32_t* exist_since_kf_after_ptr = exist_since_kf_after_cpu.data_ptr<int32_t>();
    bool* unstable_after_ptr = unstable_after_cpu.data_ptr<bool>();
    bool* rendered_depth_after_ptr = rendered_depth_after_cpu.data_ptr<bool>();
    int32_t* rendered_depth_support_after_ptr = rendered_depth_support_after_cpu.data_ptr<int32_t>();
    int32_t* rendered_depth_last_seen_after_ptr = rendered_depth_last_seen_after_cpu.data_ptr<int32_t>();
    int32_t* rendered_depth_source_after_ptr = rendered_depth_source_after_cpu.data_ptr<int32_t>();

    int64_t matched_exact = 0;
    int64_t matched_parent = 0;
    int64_t unmatched = 0;
    for (int64_t i = 0; i < N_after; ++i) {
        const int64_t key = new_key_ptr[i];
        auto it = key_to_old_idx.find(key);
        if (it != key_to_old_idx.end()) {
            const int64_t old_idx = it->second;
            art_after_ptr[i] = old_art_ptr[old_idx];
            orb_after_ptr[i] = old_orb_ptr[old_idx];
            inactive_geo_after_ptr[i] = old_inactive_geo_ptr[old_idx];
            rgbd_fill_render_holes_after_ptr[i] = old_rgbd_fill_render_holes_ptr[old_idx];
            depthanything_fill_holes_after_ptr[i] = old_depthanything_fill_holes_ptr[old_idx];
            promoted_after_ptr[i] = old_promoted_ptr[old_idx];
            exist_since_after_ptr[i] = old_exist_since_ptr[old_idx];
            exist_since_kf_after_ptr[i] = old_exist_since_kf_ptr[old_idx];
            unstable_after_ptr[i] = old_unstable_ptr[old_idx];
            rendered_depth_after_ptr[i] = old_rendered_depth_ptr[old_idx];
            rendered_depth_support_after_ptr[i] = old_rendered_depth_support_ptr[old_idx];
            rendered_depth_last_seen_after_ptr[i] = old_rendered_depth_last_seen_ptr[old_idx];
            rendered_depth_source_after_ptr[i] = old_rendered_depth_source_ptr[old_idx];
            ++matched_exact;
            continue;
        }

        const int64_t lv = key & 255LL;
        const int64_t path = key / 256LL;
        if (lv > 1) {
            const int parent_lv = static_cast<int>(lv - 1);
            const int levels_below_parent = std::max(0, max_num_levels_ - parent_lv);
            const int bits_to_clear = 3 * levels_below_parent;
            const int64_t lower_mask = (bits_to_clear > 0) ? ((1LL << bits_to_clear) - 1LL) : 0LL;
            const int64_t keep_mask = ~lower_mask;
            const int64_t parent_path = path & keep_mask;
            const int64_t parent_key = parent_path * 256LL + static_cast<int64_t>(parent_lv);

            auto it_parent = key_to_old_idx.find(parent_key);
            if (it_parent != key_to_old_idx.end()) {
                const int64_t old_idx = it_parent->second;
                art_after_ptr[i] = old_art_ptr[old_idx];
                orb_after_ptr[i] = old_orb_ptr[old_idx];
                inactive_geo_after_ptr[i] = old_inactive_geo_ptr[old_idx];
                rgbd_fill_render_holes_after_ptr[i] = old_rgbd_fill_render_holes_ptr[old_idx];
                depthanything_fill_holes_after_ptr[i] = old_depthanything_fill_holes_ptr[old_idx];
                promoted_after_ptr[i] = old_promoted_ptr[old_idx];
                if (topology_birth_iter_ >= 0) {
                    exist_since_after_ptr[i] = topology_birth_iter_;
                } else {
                    exist_since_after_ptr[i] = old_exist_since_ptr[old_idx];
                }
                if (topology_birth_kf_ >= 0) {
                    exist_since_kf_after_ptr[i] = topology_birth_kf_;
                } else {
                    exist_since_kf_after_ptr[i] = old_exist_since_kf_ptr[old_idx];
                }
                unstable_after_ptr[i] = old_unstable_ptr[old_idx];
                rendered_depth_after_ptr[i] = old_rendered_depth_ptr[old_idx];
                rendered_depth_support_after_ptr[i] = old_rendered_depth_support_ptr[old_idx];
                rendered_depth_last_seen_after_ptr[i] = old_rendered_depth_last_seen_ptr[old_idx];
                rendered_depth_source_after_ptr[i] = old_rendered_depth_source_ptr[old_idx];
                ++matched_parent;
                continue;
            }
        }

        ++unmatched;
    }
    // if (unmatched > 0) {
    //     std::cout << "[SUBDIV/provenance] WARNING: unmatched voxels after key/parent remap: "
    //               << unmatched << "/" << N_after << "\n";
    // }

    is_artificial_voxel_ = art_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    is_orb_voxel_ = orb_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    is_inactive_geo_voxel_ = inactive_geo_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    is_rgbd_fill_render_holes_voxel_ =
        rgbd_fill_render_holes_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    is_depthanything_fill_holes_voxel_ =
        depthanything_fill_holes_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    is_promoted_artificial_voxel_ = promoted_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    exist_since_iter_ = exist_since_after_cpu.to(device_type_).to(torch::kInt32).contiguous();
    exist_since_kf_ = exist_since_kf_after_cpu.to(device_type_).to(torch::kInt32).contiguous();
    geometrically_unstable_voxel_ = unstable_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    rendered_depth_candidate_voxel_ =
        rendered_depth_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    rendered_depth_candidate_support_count_ =
        rendered_depth_support_after_cpu.to(device_type_).to(torch::kInt32).contiguous();
    rendered_depth_candidate_last_seen_kf_ =
        rendered_depth_last_seen_after_cpu.to(device_type_).to(torch::kInt32).contiguous();
    rendered_depth_candidate_source_kind_ =
        rendered_depth_source_after_cpu.to(device_type_).to(torch::kInt32).contiguous();
    const int64_t n_art_after = is_artificial_voxel_.sum().item<int64_t>();
    const int64_t n_promoted_after = is_promoted_artificial_voxel_.sum().item<int64_t>();
    // std::cout << "[SUBDIV/artificial] N_before=" << N_before
    //           << " subdiv_parents=" << n_subdiv_parents
    //           << " subdiv_art_parents=" << n_subdiv_art_parents
    //           << " subdiv_promoted_parents=" << n_subdiv_promoted_parents
    //           << " matched_exact=" << matched_exact
    //           << " matched_parent=" << matched_parent
    //           << " unmatched=" << unmatched
    //           << " N_after=" << center_.size(0)
    //           << " art_before=" << n_art_before
    //           << " art_after=" << n_art_after
    //           << " promoted_before=" << n_promoted_before
    //           << " promoted_after=" << n_promoted_after
    //           << "\n";

    (void)n_subdiv_parents;
    (void)n_subdiv_art_parents;
    (void)n_subdiv_promoted_parents;
    (void)matched_exact;
    (void)matched_parent;
    (void)unmatched;
    (void)n_art_before;
    (void)n_art_after;
    (void)n_promoted_before;
    (void)n_promoted_after;
}

torch::Tensor VoxelModel::subdivisionPriority() const {
    auto g = this->subdiv_p_.grad();
    if (g.defined() && g.dim() == 2 && g.size(1) == 1) g = g.squeeze(1);
    return g;
}

void VoxelModel::resetSubdivisionPriority() {
    this->subdiv_p_.mutable_grad() = torch::Tensor();
}

void VoxelModel::freezeVoxGeo() {
    const int64_t N = center_.size(0);
    auto care_idx = torch::arange(N, torch::dtype(torch::kLong).device(device_type_));
    torch::NoGradGuard no_grad;
    frozen_vox_geo_ = gatherSvrasterGeoParams(vox_key_, care_idx, _geo_grid_pts_)[0].contiguous();
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
    return renderSvrasterDirect(
        cam,
        im_height,
        im_width,
        _geo_grid_pts_,
        sh0_,
        shs_,
        subdiv_p_,
        oct_path_,
        center_,
        size_,
        vox_key_,
        frozen_vox_geo_,
        active_sh_degree_,
        white_background_,
        black_background_,
        ss_,
        n_samp_per_vox_,
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
    TV_COMPUTE::total_variation_bw(
        _geo_grid_pts_,
        vox_key_,
        lambda_tv_density,
        vox_size_inv_,
        /*no_tv_s=*/true,
        /*tv_sparse=*/false,
        grad);
}

VoxelModel::SchedulerState VoxelModel::schedulerState() const
{
    SchedulerState state;
    state.valid = optimizer_initialized_;
    state.last_epoch = scheduler_epoch_;
    state.geo_lr = optimizer_geo_lr_;
    state.sh0_lr = optimizer_sh0_lr_;
    state.shs_lr = optimizer_shs_lr_;
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

// simple stride sampler for a vector of indices
static std::vector<int64_t> stride_sample(const std::vector<int64_t>& idxs, size_t want) {
    if (idxs.size() <= want) return idxs;
    std::vector<int64_t> out; out.reserve(want);
    double step = double(idxs.size()) / double(want);
    for (size_t k=0; k<want; ++k) {
        size_t pos = size_t(k * step);
        if (pos >= idxs.size()) pos = idxs.size()-1;
        out.push_back(idxs[pos]);
    }
    return out;
}
} // namespace

void VoxelModel::savePly(const std::filesystem::path& result_path)
{
    int target_max_voxels =1000000;
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

    // Stats (helpful when debugging “black”)
    auto fdc_min = std::get<0>(fdc.aminmax()); auto fdc_max = std::get<1>(fdc.aminmax());
    auto fre_min = std::get<0>(frest.aminmax()); auto fre_max = std::get<1>(frest.aminmax());
    auto geo_min = std::get<0>(geo_cpu.aminmax()); auto geo_max = std::get<1>(geo_cpu.aminmax());
    std::cout << "[savePly] N=" << N
              << "  lv:[" << lv_min << "," << lv_max << "]"
              << "  f_dc:[" << fdc_min.item<float>() << "," << fdc_max.item<float>() << "]"
              << "  f_rest:[" << fre_min.item<float>() << "," << fre_max.item<float>() << "]"
              << "  geo:[" << geo_min.item<float>() << "," << geo_max.item<float>() << "]\n";

    // AABB of xyz (just to spot wildly wrong scale/offset)
    auto xyz_min = std::get<0>(xyz.aminmax(0));
    auto xyz_max = std::get<1>(xyz.aminmax(0));
    std::cout << "[savePly] AABB min=("
              << xyz_min[0].item<float>() << "," << xyz_min[1].item<float>() << "," << xyz_min[2].item<float>()
              << ") max=("
              << xyz_max[0].item<float>() << "," << xyz_max[1].item<float>() << "," << xyz_max[2].item<float>()
              << ")\n";

    // Optional: level-aware downsample to keep ~target_max_voxels (coarse first)
    std::vector<int64_t> keep_idx;
    keep_idx.reserve(N);
    if (target_max_voxels > 0 && N > target_max_voxels) {
        std::unordered_map<int, std::vector<int64_t>> by_level;
        by_level.reserve(lv_max-lv_min+1);
        for (int64_t n=0; n<N; ++n) by_level[lv_ptr[n]].push_back(n);

        size_t remaining = target_max_voxels;
        for (int L = lv_min; L <= lv_max && remaining > 0; ++L) {
            auto& bucket = by_level[L];
            if (bucket.empty()) continue;
            if (bucket.size() <= remaining) {
                keep_idx.insert(keep_idx.end(), bucket.begin(), bucket.end());
                remaining -= bucket.size();
            } else {
                auto sampled = stride_sample(bucket, remaining);
                keep_idx.insert(keep_idx.end(), sampled.begin(), sampled.end());
                remaining = 0;
            }
        }
        std::sort(keep_idx.begin(), keep_idx.end());
        std::cout << "[savePly] Downsampled " << N << " → " << keep_idx.size()
                  << " voxels for viewer.\n";
    } else {
        keep_idx.resize(N); std::iota(keep_idx.begin(), keep_idx.end(), 0);
    }
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

    std::cout << "[savePly] Wrote " << M << " voxels to " << result_path
              << "  (viewer SH degree=1). "
              << (M>1200000 ? "NOTE: Consider lowering to ≤1M for FPS." : "")
              << "\n";
}

} // namespace sv
