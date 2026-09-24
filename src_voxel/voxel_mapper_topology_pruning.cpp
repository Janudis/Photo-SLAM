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
        py::module_::import("sys").attr("path").attr("insert")(0, "scripts");
        py::module_::import("sys").attr("path").attr("insert")(0, "../scripts");
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

void VoxelMapper::pruneVoxelTopology(
    const int iter,
    const int prune_every,
    const std::vector<sv::MiniCam>& tr_cams,
    sv::VoxelModel::StatPkg& stat)
{
                // 0) Refresh training statistics if topology changed since they were computed.
                {
                    const int N_cur = voxel_model_->numVoxels();
                    const bool stat_shape_ok =
                        stat.max_w.defined() &&
                        stat.min_samp_interval.defined() &&
                        stat.view_cnt.defined() &&
                        stat.max_w.size(0) == N_cur &&
                        stat.min_samp_interval.size(0) == N_cur &&
                        stat.view_cnt.size(0) == N_cur;
                    if (!stat_shape_ok) {
                        stat = voxel_model_->computeTrainingStat(tr_cams);
                    }
                }
                const int prune_all_iter =
                    std::max(1, opt_params_.prune_until_ - prune_every);
                const int prune_now_iter =
                    std::max(0, iter - prune_every);
                const float prune_iter_rate =
                    std::clamp(
                        static_cast<float>(prune_now_iter) /
                            static_cast<float>(prune_all_iter),
                        0.0f,
                        1.0f);
                const float prune_thres =
                    opt_params_.prune_thres_init_ +
                    std::max(0.0f,
                             opt_params_.prune_thres_final_ -
                                 opt_params_.prune_thres_init_) *
                        prune_iter_rate;
                // Separate threshold for at-target voxels.
                const float prune_thres_at_target = opt_params_.prune_thres_final_at_target_;
                const int ori_n = voxel_model_->numVoxels();
                const int N     = ori_n;
                const float target_vox_size = voxel_model_->fixedVoxSize();
                torch::Tensor observed_layout_mask; // [N] bool, voxels observed by the pruning camera set
                torch::Tensor prune_mask_near; // [N] bool, near-camera prune (unprotected)
                torch::Tensor prune_mask_far; // [N] bool, outside dense-core bound prune
                torch::Tensor prune_mask_mvs_supported; // [N] bool, MVS-confirmed surface cells
                torch::Tensor prune_mask_mvs_free_space; // [N] bool, multi-view free-space contradictions
                torch::Tensor prune_mask_default;         // [N] bool, default rules only

                torch::Device prune_device = mDevice;
                torch::Tensor centers_for_device = voxel_model_->voxCenter();
                if (centers_for_device.defined()) {
                    prune_device = centers_for_device.device();
                }
                auto bool_opts_prune =
                    torch::TensorOptions().dtype(torch::kBool).device(prune_device);
                torch::Tensor prune_mask_base = torch::zeros({N}, bool_opts_prune);
                auto prune_mask = prune_mask_base.clone();

                torch::Tensor sdf_prune_mask;
                torch::Tensor tsdf_surface_protect_mask;
                torch::Tensor svrecon_known_voxel_mask;
                float svrecon_surface_band_m =
                    std::max(1.0e-6f, 2.0f * sdfMetricVoxelSize());
                // 2) SVRecon SDF pruning.
                if (N > 0) {
                    try {
                            torch::Tensor centers_world = voxel_model_->voxCenter(); // [N,3]
                            torch::Tensor sizes_world   = voxel_model_->voxSize();   // [N,1] (your implementation)

                            if (centers_world.defined() &&
                                centers_world.dim() == 2 &&
                                centers_world.size(0) == N &&
                                centers_world.size(1) == 3 &&
                                sizes_world.defined() &&
                                sizes_world.size(0) == N)
                            {
                                // Gather SDF at the 8 voxel grid corners. In SVRecon mode
                                // the optimized geo grid is the SDF field.
                                torch::Tensor tsdf8 =
                                    voxel_model_->voxelGeoCorners()
                                        .to(prune_mask.device())
                                        .to(torch::kFloat32)
                                        .contiguous();

                                // Device alignment (should already match)
                                if (tsdf8.device() != prune_mask.device()) {
                                    tsdf8 = tsdf8.to(prune_mask.device());
                                }
                                torch::Tensor corner_valid;
                                torch::Tensor tsdf_unknown_mask;
                                torch::Tensor tsdf_free_mask;
                                torch::Tensor tsdf_occupied_mask;
                                torch::Tensor tsdf_surface_mask;

                                // Keep only an actual learned-SDF surface: the voxel must
                                // have finite corners on both strict sides of zero.
                                corner_valid = torch::isfinite(tsdf8).to(torch::kBool);
                                torch::Tensor valid_count =
                                    corner_valid.to(torch::kInt32).sum(/*dim=*/1);
                                torch::Tensor voxel_valid = (valid_count > 0).to(torch::kBool);
                                svrecon_known_voxel_mask =
                                    voxel_valid.to(prune_mask.device()).to(torch::kBool).contiguous();
                                torch::Tensor has_pos =
                                    ((tsdf8 > 0.0f) & corner_valid).any(/*dim=*/1);
                                torch::Tensor has_neg =
                                    ((tsdf8 < 0.0f) & corner_valid).any(/*dim=*/1);
                                torch::Tensor has_surface =
                                    ((valid_count == tsdf8.size(1)) & has_pos & has_neg)
                                        .to(torch::kBool);

                                    torch::Tensor abs_sdf = tsdf8.abs();
                                    torch::Tensor inf_like =
                                        torch::full_like(abs_sdf, std::numeric_limits<float>::infinity());
                                    torch::Tensor abs_valid =
                                        torch::where(corner_valid, abs_sdf, inf_like);
                                    torch::Tensor min_abs_sdf =
                                        std::get<0>(abs_valid.min(/*dim=*/1));

                                    float global_vox_size_min =
                                        std::max(1.0e-6f, voxel_model_->fixedVoxSize());
                                    torch::Tensor vox_size_for_thresh = voxel_model_->voxSize();
                                    if (vox_size_for_thresh.defined() && vox_size_for_thresh.numel() > 0) {
                                        global_vox_size_min = std::max(
                                            1.0e-6f,
                                            vox_size_for_thresh
                                                .to(prune_mask.device())
                                                .to(torch::kFloat32)
                                                .min()
                                                .item<float>());
                                    }
                                    torch::Tensor log_s =
                                        voxel_model_->svreconLogS()
                                            .to(prune_mask.device())
                                            .to(torch::kFloat32);
                                    torch::Tensor sdf_thresh_t =
                                        torch::log(torch::tensor(
                                            199.0f,
                                            torch::TensorOptions()
                                                .dtype(torch::kFloat32)
                                                .device(prune_mask.device()))) /
                                        torch::exp(10.0f * log_s);
                                    const float sdf_thresh = std::max(
                                        2.0f * global_vox_size_min,
                                        sdf_thresh_t.reshape({-1})[0].item<float>());
                                    svrecon_surface_band_m = sdf_thresh;

                                    torch::Tensor inside_mask =
                                        torch::ones({N},
                                            torch::TensorOptions()
                                                .dtype(torch::kBool)
                                                .device(prune_mask.device()));
                                    torch::Tensor scene_center = voxel_model_->SceneCenter();
                                    torch::Tensor inside_extent = voxel_model_->InsideExtent();
                                    if (!inside_extent.defined() || inside_extent.numel() == 0) {
                                        inside_extent = voxel_model_->SceneExtent();
                                    }
                                    if (scene_center.defined() &&
                                        inside_extent.defined() &&
                                        scene_center.numel() == 3 &&
                                        inside_extent.numel() > 0) {
                                        scene_center =
                                            scene_center.to(prune_mask.device())
                                                .to(torch::kFloat32)
                                                .contiguous()
                                                .view({1, 3});
                                        const float extent =
                                            std::max(1.0e-6f,
                                                     inside_extent
                                                         .to(prune_mask.device())
                                                         .to(torch::kFloat32)
                                                         .reshape({-1})[0]
                                                         .item<float>());
                                        torch::Tensor half_extent =
                                            torch::full({1, 3}, 0.5f * extent,
                                                torch::TensorOptions()
                                                    .dtype(torch::kFloat32)
                                                    .device(prune_mask.device()));
                                        torch::Tensor centers_for_inside =
                                            centers_world.to(prune_mask.device()).to(torch::kFloat32);
                                        inside_mask =
                                            ((centers_for_inside >= (scene_center - half_extent)).all(/*dim=*/1) &
                                             (centers_for_inside <= (scene_center + half_extent)).all(/*dim=*/1))
                                                .to(torch::kBool);
                                    }

                                    torch::Tensor near_surface =
                                        (min_abs_sdf <= sdf_thresh).to(torch::kBool);
                                    torch::Tensor all_positive =
                                        (voxel_valid & has_pos & (~has_neg)).to(torch::kBool);
                                    torch::Tensor all_negative =
                                        (voxel_valid & has_neg & (~has_pos)).to(torch::kBool);
                                    tsdf_unknown_mask = (~voxel_valid).to(torch::kBool);
                                    tsdf_free_mask =
                                        (all_positive & (~near_surface) & inside_mask).to(torch::kBool);
                                    tsdf_occupied_mask =
                                        (all_negative & (~near_surface) & inside_mask).to(torch::kBool);
                                    tsdf_surface_mask = has_surface.to(torch::kBool);

                                    // SVRecon SDF pruning: preserve both actual sign-changing
                                    // cells and the near-zero band in which a surface can still
                                    // move or form. Online SLAM has no fixed offline foreground
                                    // extent, so apply the same rule to every allocated cell.
                                    sdf_prune_mask =
                                        computeSvreconSdfPruneMask()
                                            .to(prune_mask.device())
                                            .to(torch::kBool);
                                tsdf_surface_protect_mask =
                                    tsdf_surface_mask.to(prune_mask.device()).to(torch::kBool).contiguous();

                                // Ensure device matches prune_mask
                                if (sdf_prune_mask.device() != prune_mask.device()) {
                                    sdf_prune_mask = sdf_prune_mask.to(prune_mask.device());
                                }
                                prune_mask = sdf_prune_mask.to(torch::kBool);

                                prune_mask_base = torch::zeros(
                                    {N},
                                    torch::TensorOptions()
                                        .dtype(torch::kBool)
                                        .device(prune_mask.device()));
                            }
                    } catch (const std::exception&) {}
                }

                // 3) Layout visibility / near filtering. This mirrors the
                // SVRecon octlayout_filtering utility: keep voxels with
                // mark_max_samp_rate > 0 and optionally remove mark_near hits.
                if (!tr_cams.empty() && N > 0) {
                    try {
                            at::Tensor octpath = voxel_model_->octPath().contiguous();      // [N,1] int64
                            at::Tensor L = voxel_model_->octLevel().contiguous();           // [N,1] int8
                            at::Tensor vox_center = voxel_model_->voxCenter().contiguous(); // [N,3]
                            at::Tensor vox_size = voxel_model_->voxSize().contiguous();     // [N,1]

                            // Basic sanity: same N
                            TORCH_CHECK(octpath.size(0) == N,
                                        "octpath.size(0) != N in pruning visibility filter");
                            TORCH_CHECK(L.size(0) == N,
                                        "octlevel.size(0) != N in pruning visibility filter");
                            TORCH_CHECK(vox_center.size(0) == N,
                                        "vox_center.size(0) != N in pruning visibility filter");
                            TORCH_CHECK(vox_center.size(1) == 3,
                                        "vox_center.size(1) must be 3");
                            if (vox_size.dim() == 1) {
                                vox_size = vox_size.view({N,1});
                            } else if (vox_size.dim() == 2) {
                                TORCH_CHECK(vox_size.size(0) == N,
                                            "vox_size.size(0) != N in pruning visibility filter");
                            } else {
                                TORCH_CHECK(false, "vox_size must be [N] or [N,1]");
                            }

                            auto Nu_before = octpath.size(0);
                            TORCH_CHECK(Nu_before == N,
                                        "octpath.size(0) != N before visibility filter");

                            // 1) visibility: rate > 0
                            at::Tensor rate =
                                sv::markSvreconMaxSampRateDirect(tr_cams, octpath, vox_center, vox_size);

                            if (rate.dim() == 2 && rate.size(1) == 1)
                                rate = rate.squeeze(1);
                            rate = rate.to(torch::kFloat32);

                            at::Tensor keep_rate = (rate > 0.0f).to(torch::kBool);   // [N]
                            // 2) near filtering:
                            //    a) SVRecon mark_near (camera-facing)
                            //    b) legacy geometric distance-to-camera test
                            //       for the old density path only.
                            const float near_thresh = 0.2f;
                            at::Tensor is_near = torch::zeros(
                                {N},
                                torch::TensorOptions().dtype(torch::kBool).device(keep_rate.device()));
                            at::Tensor is_near_geom = torch::zeros(
                                {N},
                                torch::TensorOptions().dtype(torch::kBool).device(keep_rate.device()));
                            if (near_thresh > 0.0f) {
                                is_near =
                                    sv::markSvreconNearDirect(tr_cams, octpath, vox_center, vox_size, near_thresh);
                                if (is_near.dim() == 2 && is_near.size(1) == 1)
                                    is_near = is_near.squeeze(1);
                                is_near = is_near.to(torch::kBool);
                                if (opt_params_.prune_near_voxels_geometric_) {
                                    auto vox_center_f32 =
                                        vox_center.to(keep_rate.device()).to(torch::kFloat32).contiguous();
                                    auto vox_size_1d =
                                        vox_size.to(keep_rate.device()).to(torch::kFloat32).contiguous();
                                    if (vox_size_1d.dim() == 2 && vox_size_1d.size(1) == 1) {
                                        vox_size_1d = vox_size_1d.squeeze(1);
                                    } else if (vox_size_1d.dim() != 1) {
                                        vox_size_1d = vox_size_1d.reshape({-1});
                                    }
                                    TORCH_CHECK(vox_size_1d.numel() == N,
                                                "vox_size_1d.numel() != N in pruning geometric near filter");

                                    auto near_radius =
                                        (torch::full_like(vox_size_1d, near_thresh) + 0.5f * vox_size_1d)
                                            .contiguous();
                                    auto near_radius_sq = (near_radius * near_radius).contiguous();

                                    for (const auto& c : tr_cams) {
                                        auto cam_pos =
                                            c.position.to(keep_rate.device()).to(torch::kFloat32).view({1, 3});
                                        auto d2 = (vox_center_f32 - cam_pos).pow(2).sum(/*dim=*/1);
                                        is_near_geom =
                                            (is_near_geom | (d2 <= near_radius_sq)).to(torch::kBool);
                                    }
                                }
                            }

                            auto prune_near_union = torch::zeros_like(is_near);
                            if (opt_params_.filter_near_voxels_) {
                                prune_near_union =
                                    (prune_near_union | is_near).to(torch::kBool);
                            }
                            if (opt_params_.prune_near_voxels_geometric_) {
                                prune_near_union =
                                    (prune_near_union | is_near_geom).to(torch::kBool);
                            }
                            keep_rate = keep_rate.view({-1}).to(torch::kBool);    // [N]
                            observed_layout_mask =
                                keep_rate.to(prune_mask.device()).to(torch::kBool).contiguous();
                            prune_mask_near =
                                prune_near_union.view({-1}).to(torch::kBool); // [N], near only

                            // Visibility remains diagnostic. Configured near
                            // filtering is applied below as a concrete prune cause.

                    } catch (const std::exception& e) {
                        std::cerr << "[PRUNE/visibility] exception: " << e.what() << "\n";
                    }
                }

                if (opt_params_.prune_far_voxels_ && N > 0) {
                    try {
                        voxel_model_->refreshDenseCoreBBFromCurrentVoxels();
                        if (voxel_model_->hasDenseCoreBB()) {
                            auto centers = voxel_model_->voxCenter();
                            auto bb_min = voxel_model_->denseCoreBBMin();
                            auto bb_max = voxel_model_->denseCoreBBMax();
                            if (centers.defined() && bb_min.defined() && bb_max.defined() &&
                                centers.dim() == 2 && centers.size(1) == 3 &&
                                centers.size(0) == N &&
                                bb_min.numel() == 3 && bb_max.numel() == 3) {
                                centers = centers.to(prune_mask.device()).to(torch::kFloat32).contiguous();
                                bb_min = bb_min.to(prune_mask.device()).to(torch::kFloat32).contiguous().view({1, 3});
                                bb_max = bb_max.to(prune_mask.device()).to(torch::kFloat32).contiguous().view({1, 3});
                                auto in_dense_core =
                                    (centers >= bb_min).all(/*dim=*/1) &
                                    (centers <= bb_max).all(/*dim=*/1);
                                prune_mask_far = (~in_dense_core.to(torch::kBool)).to(torch::kBool);
                            }
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "[PRUNE/far] exception: " << e.what() << "\n";
	                    }
		                }

                if (prune_mask_near.defined() &&
                    prune_mask_near.numel() == N) {
                    prune_mask =
                        (prune_mask.to(torch::kBool) |
                         prune_mask_near.to(prune_mask.device())
                             .to(torch::kBool))
                            .contiguous();
                }
                if (opt_params_.prune_far_voxels_ &&
                    prune_mask_far.defined() &&
                    prune_mask_far.numel() == N) {
                    prune_mask =
                        (prune_mask.to(torch::kBool) |
                         prune_mask_far.to(prune_mask.device())
                             .to(torch::kBool))
                            .contiguous();
                }

                if (opt_params_.prune_mvs_consistency_enable_ &&
                    sensor_type_ == MONOCULAR && N > 0) {
                    try {
                        const torch::Tensor centers_world =
                            voxel_model_->voxCenter();
                        const torch::Tensor sizes_world =
                            voxel_model_->voxSize();
                        sv::MonocularMvsPruneEvidence mvs_evidence =
                            computeMonocularMvsPruneEvidence(
                                centers_world, sizes_world);
                        if (mvs_evidence.supported.defined() &&
                            mvs_evidence.free_space.defined() &&
                            mvs_evidence.supported.numel() == N &&
                            mvs_evidence.free_space.numel() == N) {
                            prune_mask_mvs_supported =
                                mvs_evidence.supported
                                    .to(prune_mask.device())
                                    .to(torch::kBool)
                                    .contiguous();
                            prune_mask_mvs_free_space =
                                mvs_evidence.free_space
                                    .to(prune_mask.device())
                                    .to(torch::kBool)
                                    .contiguous();
                            auto zero_mask = torch::zeros_like(
                                prune_mask.to(torch::kBool));
                            torch::Tensor soft_candidates = zero_mask.clone();
                            if (sdf_prune_mask.defined() &&
                                sdf_prune_mask.numel() == N) {
                                soft_candidates =
                                    soft_candidates |
                                    sdf_prune_mask.to(prune_mask.device())
                                        .to(torch::kBool);
                            }
                            torch::Tensor hard_candidates =
                                prune_mask_base.to(prune_mask.device())
                                    .to(torch::kBool)
                                    .clone();
                            if (prune_mask_near.defined() &&
                                prune_mask_near.numel() == N) {
                                hard_candidates =
                                    hard_candidates |
                                    prune_mask_near.to(prune_mask.device())
                                        .to(torch::kBool);
                            }
                            if (prune_mask_far.defined() &&
                                prune_mask_far.numel() == N) {
                                hard_candidates =
                                    hard_candidates |
                                    prune_mask_far.to(prune_mask.device())
                                        .to(torch::kBool);
                            }

                            // Near/far are hard geometric filters. MVS support
                            // can protect learned-SDF candidates; independently
                            // confirmed free space contributes new candidates.
                            prune_mask =
                                (hard_candidates |
                                 (soft_candidates &
                                  (~prune_mask_mvs_supported)) |
                                 prune_mask_mvs_free_space)
                                    .to(torch::kBool)
                                    .contiguous();
                        }
                    } catch (const std::exception& e) {
                        std::cerr
                            << "[PRUNE/mvs] skipped: " << e.what() << "\n";
                    }
                }

                // Fine SVRecon parents are retained for level-9 continuity and
                // never participate in rendering or leaf-surface pruning.
                auto leaf_mask = voxel_model_->isLeaf();
                if (leaf_mask.defined() && leaf_mask.numel() == N) {
                    prune_mask = prune_mask.to(torch::kBool) &
                                 leaf_mask.to(prune_mask.device())
                                     .to(torch::kBool).reshape({-1});
                }
                // 4) Use the accumulated SVRecon pruning mask.
                prune_mask_default = prune_mask.to(torch::kBool);

                if (prune_mask.defined() && prune_mask.numel() == N) {
                    auto prune_idx = prune_mask.to(torch::kBool).nonzero().squeeze(1); // [K]
                    if (prune_idx.numel() > 0) {
                        auto centers_world = voxel_model_->voxCenter(); // [N,3]
                        auto sizes_world   = voxel_model_->voxSize();   // [N] or [N,1]
                        if (centers_world.defined() &&
                            centers_world.dim() == 2 &&
                            centers_world.size(0) == N &&
                            centers_world.size(1) == 3 &&
	                            sizes_world.defined() &&
	                            sizes_world.size(0) == N)
	                        {
	                            torch::Tensor pruned_centers = centers_world.index({prune_idx}).clone();
	                            torch::Tensor pruned_sizes = sizes_world.index({prune_idx}).clone();
                                torch::Tensor pruned_levels;
                                torch::Tensor oct_levels = voxel_model_->octLevel();
                                if (oct_levels.defined() &&
                                    oct_levels.numel() == N) {
                                    pruned_levels =
                                        oct_levels.reshape({N, 1})
                                            .index_select(
                                                0,
                                                prune_idx.to(oct_levels.device()).to(torch::kLong))
                                            .contiguous();
                                }
                                torch::Tensor pruned_colors;
                                torch::Tensor sh0 = voxel_model_->sh0();
                                if (sh0.defined() && sh0.dim() == 2 &&
                                    sh0.size(0) == N) {
                                    pruned_colors =
                                        (sh0.index_select(
                                             0,
                                             prune_idx.to(sh0.device()).to(torch::kLong)) *
                                             sv::kSHC0 +
                                         0.5f)
                                            .clamp(0.0f, 1.0f)
                                            .contiguous();
                                }
	                            torch::Tensor whole_run_pruned_by_tsdf;
                            if (sdf_prune_mask.defined() &&
                                sdf_prune_mask.numel() == N) {
                                whole_run_pruned_by_tsdf =
                                    sdf_prune_mask
                                        .to(prune_idx.device())
                                        .to(torch::kBool)
                                        .contiguous()
                                        .index_select(0, prune_idx.to(torch::kLong))
                                        .contiguous();
                            }
                            torch::Tensor whole_run_pruned_by_near;
                            if (prune_mask_near.defined() &&
                                prune_mask_near.numel() == N) {
                                whole_run_pruned_by_near =
                                    prune_mask_near
                                        .to(prune_idx.device())
                                        .to(torch::kBool)
                                        .contiguous()
                                        .index_select(
                                            0,
                                            prune_idx.to(torch::kLong))
                                        .contiguous();
                            }
                            torch::Tensor whole_run_pruned_by_far;
                            if (prune_mask_far.defined() &&
                                prune_mask_far.numel() == N) {
                                whole_run_pruned_by_far =
                                    prune_mask_far
                                        .to(prune_idx.device())
                                        .to(torch::kBool)
                                        .contiguous()
                                        .index_select(
                                            0,
                                            prune_idx.to(torch::kLong))
                                        .contiguous();
                            }
                            torch::Tensor whole_run_pruned_by_mvs_free_space;
                            if (prune_mask_mvs_free_space.defined() &&
                                prune_mask_mvs_free_space.numel() == N) {
                                whole_run_pruned_by_mvs_free_space =
                                    prune_mask_mvs_free_space
                                        .to(prune_idx.device())
                                        .to(torch::kBool)
                                        .contiguous()
                                        .index_select(
                                            0,
                                            prune_idx.to(torch::kLong))
                                        .contiguous();
                            }
                            appendWholeRunPrunedVoxels(
                                iter,
                                pruned_centers,
                                pruned_sizes,
                                pruned_levels,
                                pruned_colors,
                                whole_run_pruned_by_tsdf,
                                torch::Tensor(),
                                whole_run_pruned_by_near,
                                whole_run_pruned_by_far,
                                whole_run_pruned_by_mvs_free_space);
                        }
                    }
                }

                voxel_model_->pruning(prune_mask);
                if (rerun_params_.run_whole_run_ ||
                    rerun_params_.rerun_svrecon_debug_ ||
                    rerun_params_.rerun_monocular_debug_) {
                    rerun_state_.whole_run_live_voxels_dirty_ = true;
                }
                const int new_n = voxel_model_->numVoxels();
                // Verbose [PRUNE/TOTAL] diagnostics disabled.

                // If pruning changed the voxel set (or shapes don’t match), recompute stats
                const int M = voxel_model_->numVoxels();
                const bool shape_ok =
                    stat.min_samp_interval.defined() &&
                    stat.min_samp_interval.dim() == 2 &&
                    stat.min_samp_interval.size(0) == M &&
                    stat.min_samp_interval.size(1) == 1;
                if (new_n != ori_n || !shape_ok) {
                    stat = voxel_model_->computeTrainingStat(tr_cams);
                }
}
