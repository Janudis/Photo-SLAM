#include "include_voxel/voxel_model.h"
#include "include_voxel/rerun_utils.h"

namespace sv {

namespace {
void logLiveMaskedVoxels(
    const torch::Tensor& centers_all,
    const torch::Tensor& sizes_all,
    const torch::Tensor& mask,
    const torch::Tensor& live_colors,
    const int iteration,
    const std::string& entity_path,
    const std::array<float, 4>& fallback_rgba)
{
    if (!centers_all.defined() || centers_all.numel() == 0 ||
        !sizes_all.defined() || sizes_all.numel() == 0) {
        return;
    }
    if (!mask.defined() || mask.size(0) != centers_all.size(0)) {
        return;
    }

    auto live_mask = mask
        .to(centers_all.device())
        .to(torch::kBool)
        .contiguous()
        .view({-1});
    if (!live_mask.any().item<bool>()) {
        return;
    }

    auto idx = torch::nonzero(live_mask).view({-1});
    auto centers = centers_all.index_select(0, idx).contiguous();
    auto sizes = sizes_all.index_select(0, idx).view({-1, 1}).contiguous();
    torch::Tensor colors;
    if (live_colors.defined() && live_colors.numel() > 0 &&
        live_colors.dim() == 2 &&
        live_colors.size(0) == centers_all.size(0) &&
        (live_colors.size(1) == 3 || live_colors.size(1) == 4)) {
        colors = live_colors.to(centers.device()).index_select(0, idx).contiguous();
    } else {
        colors = torch::zeros(
            {centers.size(0), 4},
            torch::TensorOptions().dtype(torch::kFloat32).device(centers.device()));
        colors.index_put_({torch::indexing::Slice(), 0}, fallback_rgba[0]);
        colors.index_put_({torch::indexing::Slice(), 1}, fallback_rgba[1]);
        colors.index_put_({torch::indexing::Slice(), 2}, fallback_rgba[2]);
        colors.index_put_({torch::indexing::Slice(), 3}, fallback_rgba[3]);
    }

    sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
        centers,
        sizes,
        colors,
        iteration,
        entity_path);
}
} // namespace

void VoxelModel::logLiveOrbVoxels(const int iteration, const torch::Tensor& live_colors)
{
    logLiveMaskedVoxels(
        this->center_,
        this->size_,
        this->is_orb_voxel_,
        live_colors,
        iteration,
        "world/orb/voxels_created",
        {0.1f, 0.8f, 1.0f, 0.75f});
}

void VoxelModel::logLiveInactiveGeoVoxels(const int iteration, const torch::Tensor& live_colors)
{
    logLiveMaskedVoxels(
        this->center_,
        this->size_,
        this->is_inactive_geo_voxel_,
        live_colors,
        iteration,
        "world/voxels_inactive_geo_densify/created",
        {0.7f, 0.45f, 0.2f, 0.75f});
}

void VoxelModel::logLiveRgbdFillRenderHolesVoxels(const int iteration, const torch::Tensor& live_colors)
{
    logLiveMaskedVoxels(
        this->center_,
        this->size_,
        this->is_rgbd_fill_render_holes_voxel_,
        live_colors,
        iteration,
        "world/rgbd_fill_render_holes/created",
        {0.95f, 0.25f, 0.85f, 0.75f});
}

void VoxelModel::logLiveDepthAnythingFillHolesVoxels(const int iteration, const torch::Tensor& live_colors)
{
    logLiveMaskedVoxels(
        this->center_,
        this->size_,
        this->is_depthanything_fill_holes_voxel_,
        live_colors,
        iteration,
        "world/mono_prior_fill_holes/created",
        {0.2f, 0.95f, 0.45f, 0.75f});
}

void VoxelModel::logFinalartificialVoxels(const int iteration)
{
    (void)iteration;
    if (!this->center_.defined() || this->center_.numel() == 0) {
        return;
    }

    if (!is_artificial_voxel_.defined() || is_artificial_voxel_.size(0) != this->center_.size(0)) {
        return;
    }

    auto art_mask_raw = is_artificial_voxel_.to(this->center_.device()).to(torch::kBool).contiguous();
    auto art_mask_all = art_mask_raw.clone();
    auto promoted_mask = torch::zeros_like(art_mask_raw);
    if (is_promoted_artificial_voxel_.defined() &&
        is_promoted_artificial_voxel_.size(0) == this->center_.size(0)) {
        promoted_mask = is_promoted_artificial_voxel_
            .to(this->center_.device()).to(torch::kBool).contiguous();
        art_mask_all = art_mask_all & (~promoted_mask);
    }

    if (!art_mask_all.any().item<bool>()) {
        artificial_centers_accum_viz_ = torch::empty(
            {0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(this->center_.device()));
        artificial_sizes_accum_viz_ = torch::empty(
            {0, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(this->center_.device()));
        return;
    }

    auto idx_art_all = torch::nonzero(art_mask_all).view({-1});
    artificial_centers_accum_viz_ = this->center_.index_select(0, idx_art_all).contiguous();
    artificial_sizes_accum_viz_ = this->size_.index_select(0, idx_art_all).view({-1, 1}).contiguous();
}

void VoxelModel::logFinalPromotedartificialVoxels(const int iteration)
{
    (void)iteration;
    if (!this->center_.defined() || this->center_.numel() == 0) {
        return;
    }

    if (!is_promoted_artificial_voxel_.defined() ||
        is_promoted_artificial_voxel_.size(0) != this->center_.size(0)) {
        return;
    }

    auto promoted_mask = is_promoted_artificial_voxel_
        .to(this->center_.device()).to(torch::kBool).contiguous();
    if (!promoted_mask.any().item<bool>()) {
        return;
    }

    auto idx_promoted = torch::nonzero(promoted_mask).view({-1});
    auto promoted_centers = this->center_.index_select(0, idx_promoted).contiguous();
    auto promoted_sizes = this->size_.index_select(0, idx_promoted).view({-1, 1}).contiguous();

    auto rgba_promoted = torch::zeros(
        {promoted_centers.size(0), 4},
        torch::TensorOptions().dtype(torch::kFloat32).device(promoted_centers.device()));
    rgba_promoted.index_put_({torch::indexing::Slice(), 1}, 1.0f);
    rgba_promoted.index_put_({torch::indexing::Slice(), 3}, 0.70f);
}

} // namespace sv
