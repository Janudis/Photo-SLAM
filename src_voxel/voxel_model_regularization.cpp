#include "src_voxel/voxel_model_internal.h"

namespace sv {

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
    // Dispatches the model tensors to the native SVRecon rasterizer.
    // SVRecon reference: SVRenderer.render.
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
    // Injects SVRecon's total-variation gradient into the shared SDF corners.
    // SVRecon reference: train.py calling grid_loss_bw.total_variation.
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
    // Injects the CUDA grid-Eikonal regularizer gradient.
    // SVRecon reference: train.py calling grid_loss_bw.grid_eikonal.
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
    // Injects the CUDA Laplacian smoothness regularizer gradient.
    // SVRecon reference: train.py calling grid_loss_bw.laplacian_smoothness.
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
    // Computes a differentiable finite-difference Eikonal loss on sufficiently
    // refined leaves. This local fallback is specific to the online mapper.
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
    // Snapshots scheduler state across topology-triggered trainer recreation.
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
    // Restores learning rates and epoch after topology-triggered recreation.
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

} // namespace sv
