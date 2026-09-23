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
#include "src_voxel/voxel_mapper_rerun_helpers.h"
#include "src_voxel/voxel_mapper_rerun_tsdf_helpers.h"

void VoxelMapper::logWholeRunLiveVoxelsToRerun(
    int iteration,
    const torch::Tensor& centers_in,
    const torch::Tensor& sizes_in,
    const torch::Tensor& colors_in,
    bool log_whole_run,
    bool log_svrecon_debug,
    bool log_monocular_debug)
{
    if (!rerun_params_.enable_rerun_ ||
        (!(rerun_params_.run_whole_run_ && log_whole_run) &&
         !(rerun_params_.rerun_svrecon_debug_ && log_svrecon_debug) &&
         !(rerun_params_.rerun_monocular_debug_ && log_monocular_debug)) ||
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
    torch::Tensor active_mask =
        voxel_utils::normalizeBoolMaskOrZeros(voxel_model_->activeRenderableMask(), N, dev);
    torch::Tensor active_source_count =
        (orb_mask.to(torch::kInt32) +
         inactive_geo_mask.to(torch::kInt32) +
         rgbd_fill_mask.to(torch::kInt32) +
         monocular_rendered_depth_mask.to(torch::kInt32) +
         monocular_mvs_mask.to(torch::kInt32))
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
            bool force_log,
            const std::vector<std::string>& target_recordings)
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

        for (const std::string& recording : target_recordings) {
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
            true,
            recordings);
        was_logged = was_logged || has_voxels;
    };

    if (!recordings.empty()) {
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
    }

    if (rerun_params_.rerun_monocular_debug_ && log_monocular_debug) {
        const std::vector<std::string> compact_recording{"monocular_debug"};
        const torch::Tensor mvs_promoted_live_mask =
            monocular_mvs_tsdf_evidence_
            ? monocular_mvs_live_mask
            : torch::zeros_like(monocular_mvs_live_mask);
        log_subset(
            orb_live_mask,
            "world/svrecon_voxels/source/orb",
            {0.75f, 0.75f, 0.75f, 1.0f},
            true,
            compact_recording);
        log_subset(
            mvs_promoted_live_mask,
            "world/svrecon_voxels/source/mvs_promoted",
            {0.75f, 0.75f, 0.75f, 1.0f},
            true,
            compact_recording);
    }
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
         !rerun_params_.rerun_svrecon_debug_ &&
         !rerun_params_.rerun_monocular_debug_)) {
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
    std::vector<std::string> aggregate_recordings = recordings;
    if (rerun_params_.rerun_monocular_debug_) {
        aggregate_recordings.emplace_back("monocular_debug");
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

    for (const std::string& recording : aggregate_recordings) {
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
