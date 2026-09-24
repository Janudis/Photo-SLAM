#include "include_voxel/voxel_mapper.h"
#include "include_voxel/voxel_mapper_utils.h"
#include "include_voxel/voxel_mapper_supervision.h"
#include "include_voxel/mapper_depth_registry.h"
#include "include_voxel/tandem_mvs_backend.h"
#include <pybind11/embed.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <Eigen/Eigenvalues>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <opencv2/flann.hpp>
#include <queue>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <c10/cuda/CUDACachingAllocator.h>

#include "ORB-SLAM3/include/Atlas.h"
#include "ORB-SLAM3/include/MapPoint.h"

namespace py = pybind11;

namespace {
cv::Mat mapperDepthForKeyframe(
    const std::string& image_filename,
    const cv::Mat& tracking_depth,
    sv::Camera& camera)
{
    cv::Mat mapper_depth = sv::loadMapperDepthImage(image_filename);
    if (mapper_depth.empty()) {
        cv::Mat undistorted;
        camera.undistortImage(tracking_depth, undistorted);
        return undistorted;
    }
    if (mapper_depth.type() != CV_32FC1) {
        mapper_depth.convertTo(mapper_depth, CV_32FC1);
    }
    cv::Mat undistorted;
    cv::remap(
        mapper_depth,
        undistorted,
        camera.undistort_map1,
        camera.undistort_map2,
        cv::INTER_NEAREST);
    return undistorted;
}

std::filesystem::path resolveMapperResourcePath(
    const std::filesystem::path& config_file,
    const std::filesystem::path& configured_path)
{
    if (configured_path.empty() || configured_path.is_absolute()) {
        return configured_path;
    }
    if (std::filesystem::exists(configured_path)) {
        return std::filesystem::absolute(configured_path);
    }

    std::filesystem::path cursor =
        std::filesystem::absolute(config_file).parent_path();
    while (!cursor.empty()) {
        if (std::filesystem::exists(cursor / "CMakeLists.txt")) {
            return cursor / configured_path;
        }
        const std::filesystem::path parent = cursor.parent_path();
        if (parent == cursor) {
            break;
        }
        cursor = parent;
    }
    return configured_path;
}

void ensurePythonRuntimeInitialized(bool import_torch_cuda)
{
    static bool initialized_by_mapper = false;
    static PyThreadState* released_main_thread_state = nullptr;
    const auto configure_python_paths = []() {
        py::exec(R"PY(
import os
import site
import sys

_user_site = site.getusersitepackages()
site.addsitedir(_user_site)
for _path in (os.path.join(_user_site, "rerun_sdk"), _user_site):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)
)PY");
        py::module_::import("sys").attr("path").attr("insert")(0, "viewer");
        py::module_::import("sys").attr("path").attr("insert")(0, "../viewer");
    };

    if (!initialized_by_mapper && Py_IsInitialized() == 0) {
        py::initialize_interpreter(false);
        initialized_by_mapper = true;
        configure_python_paths();
        if (import_torch_cuda) {
            py::module_::import("torch.cuda");
        }
        released_main_thread_state = PyEval_SaveThread();
        return;
    }

    {
        py::gil_scoped_acquire gil;
        configure_python_paths();
        if (import_torch_cuda) {
            py::module_::import("torch.cuda");
        }
    }
    (void)released_main_thread_state;
}
} // namespace

void VoxelMapper::subdivideVoxelTopology(
    const int iter,
    const std::vector<sv::MiniCam>& tr_cams,
    sv::VoxelModel::StatPkg& stat)
{
                voxel_model_->setTopologyBirthContext(iter);
                const int before = voxel_model_->numVoxels();
                if (before == 0) {
                    return;
                } else {
                    bool did_subdivide = false;
                    int64_t n_normal_candidates = 0;
                    int64_t n_subdiv_normal_selected = 0;

                    // 1) One SVRecon subdivision pass for eligible voxels.
                    const int M = voxel_model_->numVoxels();
                    if (M > 0) {
                        auto vox_size = voxel_model_->voxSize(); // [M] or [M,1]
                        if (vox_size.dim() == 1) vox_size = vox_size.view({M,1});
                        else if (vox_size.dim() == 2 && vox_size.size(1) == 1) { /* ok */ }
                        else vox_size = vox_size.reshape({M,1});
                        auto vox_size_1d = vox_size.squeeze(1).contiguous();
                        auto octlv = voxel_model_->octLevel(); // [M] or [M,1]
                        if (octlv.defined() && octlv.dim() == 2 && octlv.size(1) == 1) {
                            octlv = octlv.squeeze(1);
                        }
                        auto octlv_i32 = octlv.to(torch::kInt32).contiguous();
                        const auto inside_level = octlv_i32 - svrecon_outside_level_;
                        const int level_stage_period = 2 * std::max(
                            1,
                            (opt_params_.prune_every_ > 0)
                                ? opt_params_.prune_every_
                                : opt_params_.adapt_every_);
                        const int coarse_level_limit =
                            9 - std::max(0, 2 - iter / level_stage_period);
                        // SVRecon progressively opens coarse levels 7, 8, and 9.
                        // Beyond the global association cap L=9, rendering may
                        // continue to L=10/11 but refinement is priority-limited.
                        auto non_finest =
                            (((inside_level < 9) &
                              (inside_level < coarse_level_limit)) |
                             ((inside_level >= 9) & (inside_level < 11))) &
                            (octlv_i32 < voxel_model_->maxNumLevels());
                        non_finest = non_finest.to(torch::kBool);

                        auto leaf_mask = voxel_model_->isLeaf();
                        if (!leaf_mask.defined() || leaf_mask.numel() != M) {
                            leaf_mask = torch::ones_like(non_finest, torch::kBool);
                        } else {
                            leaf_mask = leaf_mask.to(non_finest.device())
                                            .to(torch::kBool).reshape({M});
                        }
                        // Visibility is enforced during construction. Retained
                        // hierarchy parents are continuity-only and cannot split
                        // or render as leaves.
                        auto valid_mask_svrecon =
                            (non_finest.to(torch::kBool) & leaf_mask).to(torch::kBool);
                        auto normal_candidate_mask = valid_mask_svrecon.clone();
                        n_normal_candidates = normal_candidate_mask.sum().item<int64_t>();
                        torch::Tensor sdf_subdivide_candidate_mask;
                        torch::Tensor sdf_corners = voxel_model_->voxelGeoCorners();
                        if (sdf_corners.defined() &&
                            sdf_corners.numel() > 0 &&
                            sdf_corners.size(0) == M) {
                            sdf_corners =
                                sdf_corners
                                    .to(vox_size_1d.device())
                                    .to(torch::kFloat32)
                                    .contiguous();
                            torch::Tensor finite =
                                torch::isfinite(sdf_corners).to(torch::kBool);
                            torch::Tensor has_pos =
                                ((sdf_corners > 0.0f) & finite).any(/*dim=*/1);
                            torch::Tensor has_neg =
                                ((sdf_corners < 0.0f) & finite).any(/*dim=*/1);
                            torch::Tensor all_finite = finite.all(/*dim=*/1);
                            torch::Tensor has_surface =
                                (all_finite & has_pos & has_neg).to(torch::kBool);
                            // Subdivision requires an actual zero-level-set crossing.
                            // Same-sign cells in the near-zero learning band remain
                            // optimizable, but are not refined as surface geometry.
                            normal_candidate_mask =
                                (has_surface & valid_mask_svrecon)
                                    .to(torch::kBool);
                            sdf_subdivide_candidate_mask =
                                normal_candidate_mask.to(torch::kBool);
                            n_normal_candidates =
                                normal_candidate_mask.sum().item<int64_t>();
                        } else {
                            normal_candidate_mask =
                                torch::zeros_like(normal_candidate_mask, torch::kBool);
                            n_normal_candidates = 0;
                        }

                        // Priority: may be undefined/empty right after structural changes.
                        auto priority = voxel_model_->subdivisionPriority(); // [M]
                        if (!priority.defined() || priority.numel() != M) {
                            priority = torch::zeros(
                                {M},
                                torch::TensorOptions().dtype(torch::kFloat32).device(normal_candidate_mask.device()));
                        } else if (priority.dim() == 2 && priority.size(1) == 1) {
                            priority = priority.squeeze(1);
                        } else if (priority.dim() != 1) {
                            priority = priority.reshape({M});
                        }

                        auto normal_selected_mask =
                            torch::zeros_like(normal_candidate_mask, torch::kBool);
                        if (n_normal_candidates > 0) {
                            priority = priority * normal_candidate_mask.to(priority.scalar_type());

                            // In the offline SVRecon scene, inside level 9 is a
                            // stable global transition to priority refinement.
                            // Our online scene extent changes with the initial
                            // map, so that absolute level can correspond to a
                            // much finer physical scale. Establish the configured
                            // base voxel scale densely, then use accumulated
                            // SVRecon priority for every further refinement.
                            const float base_scale_threshold =
                                0.75f * std::max(
                                    1.0e-6f,
                                    voxel_model_->fixedVoxSize());
                            auto base_scale_candidates =
                                (normal_candidate_mask &
                                 (vox_size_1d >= base_scale_threshold))
                                    .to(torch::kBool);
                            auto refined_candidates =
                                (normal_candidate_mask &
                                 (vox_size_1d < base_scale_threshold))
                                    .to(torch::kBool);
                            normal_selected_mask = base_scale_candidates.clone();
                            if (opt_params_.subdivide_all_until_ > 0 &&
                                iter <= opt_params_.subdivide_all_until_) {
                                normal_selected_mask = normal_candidate_mask.clone();
                            }
                            auto fine_idx = torch::nonzero(
                                                refined_candidates &
                                                (~normal_selected_mask))
                                                .reshape({-1}).to(torch::kLong);
                            if (fine_idx.numel() > 0 && opt_params_.subdivide_prop_ > 0.0f) {
                                const int64_t fine_keep = std::max<int64_t>(
                                    1,
                                    static_cast<int64_t>(std::ceil(
                                        opt_params_.subdivide_prop_ *
                                        static_cast<float>(fine_idx.numel()))));
                                auto fine_priority = priority.index_select(0, fine_idx);
                                auto order = std::get<1>(fine_priority.sort(
                                    /*dim=*/0, /*descending=*/true));
                                auto chosen = fine_idx.index_select(
                                    0,
                                    order.index({torch::indexing::Slice(0, fine_keep)})
                                        .to(torch::kLong));
                                normal_selected_mask.index_put_({chosen}, true);
                            }
                            // SVRecon can retain a refined parent, so budget for
                            // eight additional cells per split, not seven.
                            const int64_t max_n_subdiv = std::max<int64_t>(
                                0,
                                (static_cast<int64_t>(opt_params_.subdivide_max_num_) -
                                 static_cast<int64_t>(M)) /
                                    8);
                            const int64_t selected_before_cap =
                                normal_selected_mask.sum().item<int64_t>();
                            if (selected_before_cap > max_n_subdiv) {
                                if (max_n_subdiv == 0) {
                                    normal_selected_mask =
                                        torch::zeros_like(
                                            normal_candidate_mask,
                                            torch::kBool);
                                } else {
                                    auto selected_idx =
                                        torch::nonzero(normal_selected_mask)
                                            .reshape({-1})
                                            .to(torch::kLong);
                                    auto selected_priority =
                                        priority.index_select(0, selected_idx);
                                    auto order = std::get<1>(
                                        selected_priority.sort(
                                            /*dim=*/0,
                                            /*descending=*/true));
                                    auto keep_idx = selected_idx.index_select(
                                        0,
                                        order.index({
                                            torch::indexing::Slice(
                                                0,
                                                max_n_subdiv)})
                                            .to(torch::kLong));
                                    normal_selected_mask =
                                        torch::zeros_like(
                                            normal_candidate_mask,
                                            torch::kBool);
                                    normal_selected_mask.index_put_(
                                        {keep_idx},
                                        true);
                                }
                            }

                            n_subdiv_normal_selected =
                                normal_selected_mask.sum().item<int64_t>();
                            if (n_subdiv_normal_selected > 0) {
		                                voxel_model_->subdividing(normal_selected_mask);
                                    if (rerun_params_.run_whole_run_ ||
                                        rerun_params_.rerun_svrecon_debug_ ||
                                        rerun_params_.rerun_monocular_debug_) {
                                        rerun_state_.whole_run_live_voxels_dirty_ = true;
                                    }
                                did_subdivide = true;
                            }
                        }
                        if (rerun_params_.enable_rerun_ &&
                            rerun_params_.rerun_svrecon_debug_) {
                            logSvreconDebugVoxelMaskToRerun(
                                iter,
                                sdf_subdivide_candidate_mask,
                                "world/svrecon/subdivide_candidates");
                            logSvreconDebugVoxelMaskToRerun(
                                iter,
                                normal_selected_mask,
                                "world/svrecon/subdivide_selected");
                        }
                    }

                    const int after = voxel_model_->numVoxels();

                    // 2) Apply post-subdivision bookkeeping.
                    if (did_subdivide) {
                        std::cout << "[SUBDIVIDING] " << std::setw(7) << before
                                  << " => "          << std::setw(7) << after
                                  << " (x" << std::fixed << std::setprecision(2)
                                  << (double)after / std::max(1, before) << ")\n";
                    }
                }
}

void VoxelMapper::adaptVoxelTopology(const int iter)
{
    if (disable_topology_changes_) return;

    const int prune_every = std::max(
        1, opt_params_.prune_every_ > 0
               ? opt_params_.prune_every_
               : opt_params_.adapt_every_);
    const int subdivide_every = std::max(
        1, opt_params_.subdivide_every_ > 0
               ? opt_params_.subdivide_every_
               : opt_params_.adapt_every_);
    bool need_pruning = iter >= opt_params_.prune_from_ &&
        iter % prune_every == 0 && iter <= opt_params_.prune_until_;
    const bool need_subdividing = iter >= opt_params_.subdivide_from_ &&
        iter % subdivide_every == 0;
    if (opt_params_.iterations_ > 0)
        need_pruning = need_pruning && iter <= opt_params_.iterations_ - 500;
    if (!need_pruning && !need_subdividing) return;

    std::vector<sv::MiniCam> cameras = incrementalMappingCameras();
    auto statistics = voxel_model_->computeTrainingStat(cameras);
    const auto scheduler_state = voxel_model_->schedulerState();
    if (need_pruning)
        pruneVoxelTopology(iter, prune_every, cameras, statistics);
    if (need_subdividing) {
        subdivideVoxelTopology(iter, cameras, statistics);
        voxel_model_->resetSubdivisionPriority();
    }
    voxel_model_->createTrainer(
        opt_params_.geo_lr_, opt_params_.sh0_lr_, opt_params_.shs_lr_,
        opt_params_.optim_beta1_, opt_params_.optim_beta2_,
        opt_params_.optim_eps_, opt_params_.lr_decay_ckpt_,
        opt_params_.lr_decay_mult_, opt_params_.log_s_lr_);
    voxel_model_->schedulerLoadState(scheduler_state);
    c10::cuda::CUDACachingAllocator::emptyCache();
    last_densify_iter_ = iter;
}
