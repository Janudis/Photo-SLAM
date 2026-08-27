#include "include_voxel/voxel_mapper.h"
#include "include_voxel/voxel_mapper_supervision.h"
#include "include_voxel/voxel_mapper_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

namespace {

torch::Tensor floatDepthMapToDevice(
    const cv::Mat& input,
    const torch::Device& device)
{
    if (input.empty() || input.channels() != 1) {
        return torch::Tensor();
    }
    cv::Mat float_input;
    if (input.type() == CV_32FC1) {
        float_input = input;
    } else {
        input.convertTo(float_input, CV_32FC1);
    }
    const cv::Mat continuous =
        float_input.isContinuous() ? float_input : float_input.clone();
    return torch::from_blob(
               const_cast<float*>(continuous.ptr<float>()),
               {continuous.rows, continuous.cols},
               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
        .clone()
        .to(device, torch::kFloat32)
        .contiguous();
}

} // namespace

float VoxelMapper::sdfMetricVoxelSize() const
{
    if (voxel_model_) {
        try {
            torch::Tensor sizes = voxel_model_->voxSize();
            if (sizes.defined() && sizes.numel() > 0) {
                const float finest_current_vox =
                    sizes.view({-1}).min().item<float>();
                if (std::isfinite(finest_current_vox) &&
                    finest_current_vox > 1.0e-6f) {
                    return finest_current_vox;
                }
            }
        } catch (const std::exception& e) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::cerr << "[SDF] failed to read finest voxel size; "
                          << "falling back to Mapper.sdf_voxel_size_m: "
                          << e.what() << "\n";
            }
        }
    }
    return std::max(1.0e-6f, sdf_params_.sdf_voxel_size_m_);
}

bool VoxelMapper::prepareProjectiveSdfInitContext(const std::shared_ptr<VoxelKeyframe>& kf)
{
    clearProjectiveSdfInitContext();
    if (!kf || sensor_type_ != RGBD) {
        return false;
    }

    cv::Mat depth_meters;
    if (!voxel_utils::depthMatToMeters(kf->img_auxiliary_undist_, depth_meters) ||
        depth_meters.empty()) {
        return false;
    }
    if (depth_meters.channels() > 1) {
        cv::extractChannel(depth_meters, depth_meters, 0);
    }
    if (depth_meters.type() != CV_32FC1) {
        depth_meters.convertTo(depth_meters, CV_32FC1);
    }

    const float fx = kf->cam_.fx();
    const float fy = kf->cam_.fy();
    if (fx <= 1.0e-6f || fy <= 1.0e-6f) {
        return false;
    }

    sdf_state_.projective_sdf_init_depth_meters_ = depth_meters;
    sdf_state_.projective_sdf_init_Tcw_ = kf->getPosef();
    sdf_state_.projective_sdf_init_fx_ = fx;
    sdf_state_.projective_sdf_init_fy_ = fy;
    sdf_state_.projective_sdf_init_cx_ = kf->cam_.cx();
    sdf_state_.projective_sdf_init_cy_ = kf->cam_.cy();
    sdf_state_.projective_sdf_init_width_ = depth_meters.cols;
    sdf_state_.projective_sdf_init_height_ = depth_meters.rows;
    sdf_state_.projective_sdf_init_kfid_ = kf->fid_;
    sdf_state_.projective_sdf_init_context_valid_ = true;
    return true;
}

void VoxelMapper::clearProjectiveSdfInitContext()
{
    sdf_state_.projective_sdf_init_context_valid_ = false;
    sdf_state_.projective_sdf_init_depth_meters_.release();
    sdf_state_.projective_sdf_init_Tcw_ = Sophus::SE3f();
    sdf_state_.projective_sdf_init_fx_ = 0.0f;
    sdf_state_.projective_sdf_init_fy_ = 0.0f;
    sdf_state_.projective_sdf_init_cx_ = 0.0f;
    sdf_state_.projective_sdf_init_cy_ = 0.0f;
    sdf_state_.projective_sdf_init_width_ = 0;
    sdf_state_.projective_sdf_init_height_ = 0;
    sdf_state_.projective_sdf_init_kfid_ = 0;
}

torch::Tensor VoxelMapper::computeProjectiveSdfInitForGridPoints(
    const torch::Tensor& grid_points_world,
    float /*ray_interval_m*/)
{
    TORCH_CHECK(
        grid_points_world.defined() &&
        grid_points_world.dim() == 2 &&
        grid_points_world.size(1) == 3,
        "computeProjectiveSdfInitForGridPoints expects grid_points_world [N,3]");

    const int64_t N = grid_points_world.size(0);
    auto out_opts = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(grid_points_world.device());
    torch::Tensor sdf_init =
        torch::full({N, 1}, std::numeric_limits<float>::quiet_NaN(), out_opts);
    if (N == 0 ||
        !sdf_state_.projective_sdf_init_context_valid_ ||
        sdf_state_.projective_sdf_init_depth_meters_.empty() ||
        sdf_state_.projective_sdf_init_width_ <= 0 ||
        sdf_state_.projective_sdf_init_height_ <= 0) {
        return sdf_init;
    }

    torch::NoGradGuard no_grad;
    torch::Tensor pts_world =
        grid_points_world.to(mDevice).to(torch::kFloat32).contiguous();

    Eigen::Matrix3f Rcw = sdf_state_.projective_sdf_init_Tcw_.rotationMatrix();
    Eigen::Vector3f tcw = sdf_state_.projective_sdf_init_Tcw_.translation();
    std::vector<float> R_data = {
        Rcw(0, 0), Rcw(0, 1), Rcw(0, 2),
        Rcw(1, 0), Rcw(1, 1), Rcw(1, 2),
        Rcw(2, 0), Rcw(2, 1), Rcw(2, 2)};
    std::vector<float> t_data = {tcw.x(), tcw.y(), tcw.z()};
    torch::Tensor R = torch::from_blob(
                          R_data.data(),
                          {3, 3},
                          torch::TensorOptions().dtype(torch::kFloat32))
                          .clone()
                          .to(mDevice);
    torch::Tensor t = torch::from_blob(
                          t_data.data(),
                          {1, 3},
                          torch::TensorOptions().dtype(torch::kFloat32))
                          .clone()
                          .to(mDevice);

    torch::Tensor pts_cam = torch::matmul(pts_world, R.t()) + t;
    torch::Tensor x = pts_cam.index({torch::indexing::Slice(), 0});
    torch::Tensor y = pts_cam.index({torch::indexing::Slice(), 1});
    torch::Tensor z = pts_cam.index({torch::indexing::Slice(), 2});

    torch::Tensor z_safe = z.clamp_min(1.0e-6f);
    torch::Tensor u =
        sdf_state_.projective_sdf_init_fx_ * x / z_safe +
        sdf_state_.projective_sdf_init_cx_;
    torch::Tensor v =
        sdf_state_.projective_sdf_init_fy_ * y / z_safe +
        sdf_state_.projective_sdf_init_cy_;
    const int W = sdf_state_.projective_sdf_init_width_;
    const int H = sdf_state_.projective_sdf_init_height_;
    torch::Tensor in_image =
        torch::isfinite(z) &
        (z > RGBD_min_depth_) &
        (u >= 0.0f) & (u < static_cast<float>(W)) &
        (v >= 0.0f) & (v < static_cast<float>(H));
    if (sdf_params_.sdf_init_max_depth_m_ > 0.0f) {
        in_image = in_image & (z < sdf_params_.sdf_init_max_depth_m_);
    }

    torch::Tensor depth =
        voxel_utils::cvMatToTorchTensorFloat32(
            sdf_state_.projective_sdf_init_depth_meters_,
            device_type_)
            .to(mDevice)
            .to(torch::kFloat32)
            .contiguous();
    if (depth.dim() != 2) {
        return sdf_init;
    }

    torch::Tensor u_safe =
        torch::where(torch::isfinite(u), u, torch::zeros_like(u));
    torch::Tensor v_safe =
        torch::where(torch::isfinite(v), v, torch::zeros_like(v));
    torch::Tensor u_idx =
        torch::floor(u_safe)
            .clamp(0.0f, static_cast<float>(W - 1))
            .to(torch::kLong);
    torch::Tensor v_idx =
        torch::floor(v_safe)
            .clamp(0.0f, static_cast<float>(H - 1))
            .to(torch::kLong);
    torch::Tensor depth_flat = depth.reshape({H * W});
    torch::Tensor sampled_depth =
        depth_flat.index_select(0, (v_idx * W + u_idx).to(torch::kLong)).view({N});

    const float trunc_m =
        std::max(1.0e-6f, sdf_params_.sdf_init_trunc_vox_ * sdfMetricVoxelSize());
    torch::Tensor sdf = torch::clamp(sampled_depth - z, -trunc_m, trunc_m);
    torch::Tensor valid =
        in_image &
        torch::isfinite(sampled_depth) &
        (sampled_depth > RGBD_min_depth_) &
        (sampled_depth < RGBD_max_depth_) &
        (z < sampled_depth + trunc_m);
    if (!valid.any().item<bool>()) {
        return sdf_init;
    }

    sdf_init = torch::where(
        valid.view({N, 1}),
        sdf.view({N, 1}),
        sdf_init.to(mDevice)).contiguous();

    return sdf_init.to(grid_points_world.device()).contiguous();
}

int64_t VoxelMapper::fuseProjectiveSdfInitFromKeyframe(
    const std::shared_ptr<VoxelKeyframe>& kf,
    const torch::Tensor& pixel_mask)
{
    const bool targeted_hole_update =
        pixel_mask.defined() && pixel_mask.numel() > 0;
    if ((!sdf_initialization_rgbd_projective_ && !targeted_hole_update) ||
        sensor_type_ != RGBD ||
        !voxel_model_ ||
        !prepareProjectiveSdfInitContext(kf)) {
        return 0;
    }

    torch::NoGradGuard no_grad;
    torch::Tensor grid_points = voxel_model_->gridPointsWorld();
    if (!grid_points.defined() || grid_points.numel() == 0) {
        clearProjectiveSdfInitContext();
        return 0;
    }

    torch::Tensor measured_sdf =
        computeProjectiveSdfInitForGridPoints(grid_points, sdfMetricVoxelSize());
    clearProjectiveSdfInitContext();
    torch::Tensor valid = torch::isfinite(measured_sdf).reshape({-1, 1});
    if (targeted_hole_update && valid.any().item<bool>()) {
        const float trunc_m = std::max(
            1.0e-6f,
            sdf_params_.sdf_init_trunc_vox_ * sdfMetricVoxelSize());
        // Hole densification supplies local surface support, not a full-ray
        // TSDF rewrite of the ORB field. Values clamped to +/-trunc_m lie
        // outside the measured surface band and are excluded here.
        valid = valid & (measured_sdf.abs() < trunc_m - 1.0e-6f);
        const sv::MiniCam camera =
            kf->toMiniCam(kf->image_height_, kf->image_width_);
        torch::Tensor w2c =
            camera.w2c.to(grid_points.device()).to(torch::kFloat32).contiguous();
        torch::Tensor points_cam =
            torch::matmul(
                grid_points.to(torch::kFloat32),
                w2c.index({torch::indexing::Slice(0, 3),
                           torch::indexing::Slice(0, 3)}).transpose(0, 1)) +
            w2c.index({torch::indexing::Slice(0, 3), 3}).view({1, 3});
        torch::Tensor z = points_cam.index({torch::indexing::Slice(), 2});
        torch::Tensor z_safe = z.clamp_min(1.0e-6f);
        torch::Tensor u =
            camera.fx * points_cam.index({torch::indexing::Slice(), 0}) / z_safe +
            camera.cx;
        torch::Tensor v =
            camera.fy * points_cam.index({torch::indexing::Slice(), 1}) / z_safe +
            camera.cy;
        torch::Tensor in_image =
            torch::isfinite(z) & (z > 0.0f) &
            (u >= 0.0f) & (u < static_cast<float>(camera.width)) &
            (v >= 0.0f) & (v < static_cast<float>(camera.height));
        torch::Tensor u_idx =
            torch::where(torch::isfinite(u), u, torch::zeros_like(u))
                .floor()
                .clamp(0.0f, static_cast<float>(camera.width - 1))
                .to(torch::kLong);
        torch::Tensor v_idx =
            torch::where(torch::isfinite(v), v, torch::zeros_like(v))
                .floor()
                .clamp(0.0f, static_cast<float>(camera.height - 1))
                .to(torch::kLong);
        torch::Tensor mask_flat =
            pixel_mask.to(grid_points.device()).to(torch::kBool).reshape({-1});
        TORCH_CHECK(
            mask_flat.numel() ==
                static_cast<int64_t>(camera.width) * camera.height,
            "fuseProjectiveSdfInitFromKeyframe: pixel mask size mismatch");
        torch::Tensor selected_pixel = mask_flat.index_select(
            0, (v_idx * camera.width + u_idx).to(torch::kLong));
        valid = valid & in_image.view({-1, 1}) & selected_pixel.view({-1, 1});
    }
    const int64_t observed = valid.sum().item<int64_t>();
    if (observed == 0) {
        return 0;
    }

    torch::Tensor weights = valid.to(torch::kFloat32);
    voxel_model_->fuseProjectiveSdfGridSamples(
        measured_sdf,
        weights,
        valid,
        /*max_weight=*/100.0f);
    torch::Tensor fused_weight = voxel_model_->fusedSdfWeights();
    voxel_model_->applyGeoGridRawInit(
        voxel_model_->fusedSdfGridPts(),
        targeted_hole_update ? valid : (fused_weight > 0.0f));
    return observed;
}

torch::Tensor VoxelMapper::computeSvreconSdfPruneMask(float* sdf_threshold_out)
{
    if (sdf_threshold_out) {
        *sdf_threshold_out = 0.0f;
    }

    torch::Device device = mDevice;
    if (voxel_model_ && voxel_model_->voxCenter().defined()) {
        device = voxel_model_->voxCenter().device();
    }
    auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(device);
    const int64_t N = voxel_model_ ? voxel_model_->numVoxels() : 0;
    if (!voxel_model_ || N <= 0) {
        return torch::empty({0}, bool_opts);
    }

    torch::Tensor sdf8 =
        voxel_model_->voxelGeoCorners().to(device).to(torch::kFloat32).contiguous();
    torch::Tensor leaf =
        voxel_model_->isLeaf().to(device).to(torch::kBool).reshape({-1});
    if (sdf8.dim() != 2 || sdf8.size(0) != N || sdf8.size(1) != 8 ||
        leaf.numel() != N) {
        return torch::zeros({N}, bool_opts);
    }

    torch::Tensor corner_valid = torch::isfinite(sdf8).to(torch::kBool);
    torch::Tensor valid_count =
        corner_valid.to(torch::kInt32).sum(/*dim=*/1);
    torch::Tensor has_pos = ((sdf8 > 0.0f) & corner_valid).any(/*dim=*/1);
    torch::Tensor has_neg = ((sdf8 < 0.0f) & corner_valid).any(/*dim=*/1);
    torch::Tensor has_surface =
        (valid_count == sdf8.size(1)) & has_pos & has_neg;

    torch::Tensor inf_like = torch::full_like(
        sdf8, std::numeric_limits<float>::infinity());
    torch::Tensor min_abs_sdf = std::get<0>(
        torch::where(corner_valid, sdf8.abs(), inf_like).min(/*dim=*/1));

    torch::Tensor sizes =
        voxel_model_->voxSize().to(device).to(torch::kFloat32).reshape({-1});
    float finest_voxel_size = std::max(1.0e-6f, voxel_model_->fixedVoxSize());
    if (sizes.numel() == N && N > 0) {
        finest_voxel_size = std::max(1.0e-6f, sizes.min().item<float>());
    }
    torch::Tensor log_s =
        voxel_model_->svreconLogS().to(device).to(torch::kFloat32).reshape({-1});
    float sdf_threshold = 2.0f * finest_voxel_size;
    if (log_s.numel() > 0) {
        const float sharpness_threshold =
            std::log(199.0f) / std::exp(10.0f * log_s[0].item<float>());
        sdf_threshold = std::max(sdf_threshold, sharpness_threshold);
    }
    if (sdf_threshold_out) {
        *sdf_threshold_out = sdf_threshold;
    }

    // This is the scheduled SVRecon SDF pruning rule, restricted to leaf cells:
    // no strict corner sign variation and outside the current near-surface band.
    return ((~has_surface) & (min_abs_sdf > sdf_threshold) & leaf)
        .to(torch::kBool)
        .contiguous();
}

torch::Tensor VoxelMapper::computeOnlineCovisibilityPruneMask(
    const std::vector<sv::MiniCam>& cameras,
    const torch::Tensor& view_count_in)
{
    const int64_t voxel_count = voxel_model_ ? voxel_model_->numVoxels() : 0;
    torch::Device device = mDevice;
    if (voxel_model_ && voxel_model_->voxCenter().defined()) {
        device = voxel_model_->voxCenter().device();
    }
    auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(device);
    auto prune_none = torch::zeros({std::max<int64_t>(0, voxel_count)}, bool_opts);
    if (!voxel_model_ || voxel_count <= 0) {
        return prune_none;
    }

    const int min_views = std::max(4, opt_params_.surface_min_views_);
    const int required_cameras = std::max(
        min_views,
        opt_params_.surface_view_window_size_);
    if (cameras.size() < static_cast<std::size_t>(required_cameras)) {
        std::cout << "[PRUNE/surface_views] skipped: cameras=" << cameras.size()
                  << " required=" << required_cameras << "\n";
        return prune_none;
    }
    if (!view_count_in.defined() || view_count_in.numel() != voxel_count) {
        std::cerr << "[PRUNE/surface_views] skipped: renderer view-count shape mismatch\n";
        return prune_none;
    }

    torch::Tensor view_count =
        view_count_in.to(device).to(torch::kFloat32).reshape({voxel_count});
    torch::Tensor leaf =
        voxel_model_->isLeaf().to(device).to(torch::kBool).reshape({voxel_count});
    torch::Tensor eligible = leaf.clone();

    // MonoGS performs one whole-map co-visibility pass, then limits later
    // passes to recently inserted primitives.
    if (surface_view_pruning_initialized_) {
        torch::Tensor birth_kf = voxel_model_->existSinceKf();
        if (birth_kf.defined() && birth_kf.numel() == voxel_count) {
            const int current_kf_count =
                scene_ ? static_cast<int>(scene_->keyframes().size()) : 0;
            const int recent_cutoff = std::max(0, current_kf_count - 2);
            eligible =
                eligible &
                (birth_kf.to(device).to(torch::kInt32).reshape({voxel_count}) >=
                 recent_cutoff);
        } else {
            std::cerr << "[PRUNE/surface_views] skipped: voxel birth metadata mismatch\n";
            return prune_none;
        }
    }
    torch::Tensor view_prune =
        (eligible & (view_count < static_cast<float>(min_views)))
            .to(torch::kBool)
            .contiguous();
    return view_prune;
}

void VoxelMapper::runPendingSurfaceViewPruning()
{
    if (!surfaceViewPruningReady()) {
        return;
    }

    auto pruning_profile = profileLaptopModule("surface_view_pruning");
    const std::vector<sv::MiniCam> cameras = surfaceViewPruningCameras();
    const int64_t voxel_count = voxel_model_->numVoxels();
    if (voxel_count <= 0) {
        surface_view_pending_keyframes_.clear();
        return;
    }

    torch::NoGradGuard no_grad;
    const bool initial_pass = !surface_view_pruning_initialized_;
    torch::Tensor view_count =
        voxel_model_->computeOcclusionAwareViewCount(cameras);
    torch::Tensor prune_mask =
        computeOnlineCovisibilityPruneMask(cameras, view_count);
    if (!prune_mask.defined() || prune_mask.numel() != voxel_count) {
        std::cerr
            << "[PRUNE/surface_views] skipped: visibility mask shape mismatch\n";
        return;
    }

    const torch::Device device = prune_mask.device();
    prune_mask = prune_mask.to(device).to(torch::kBool).reshape({voxel_count});

    // Optional MVS consistency can protect a co-visible surface candidate,
    // while its independent free-space candidates remain in scheduled pruning.
    int64_t mvs_protected = 0;
    if (opt_params_.prune_mvs_consistency_enable_) {
        sv::MonocularMvsPruneEvidence mvs_evidence =
            computeMonocularMvsPruneEvidence(
                voxel_model_->voxCenter(),
                voxel_model_->voxSize());
        if (mvs_evidence.supported.defined() &&
            mvs_evidence.supported.numel() == voxel_count) {
            torch::Tensor supported =
                mvs_evidence.supported.to(device).to(torch::kBool)
                    .reshape({voxel_count});
            mvs_protected = (prune_mask & supported).sum().item<int64_t>();
            prune_mask = (prune_mask & (~supported)).contiguous();
        }
    }

    torch::Tensor leaf =
        voxel_model_->isLeaf().to(device).to(torch::kBool)
            .reshape({voxel_count});
    torch::Tensor eligible = leaf.clone();
    if (!initial_pass) {
        torch::Tensor birth_kf = voxel_model_->existSinceKf();
        const int current_kf_count =
            static_cast<int>(scene_->keyframes().size());
        const int recent_cutoff = std::max(0, current_kf_count - 2);
        if (birth_kf.defined() && birth_kf.numel() == voxel_count) {
            eligible = eligible &
                (birth_kf.to(device).to(torch::kInt32)
                     .reshape({voxel_count}) >= recent_cutoff);
        }
    }

    const int64_t eligible_count = eligible.sum().item<int64_t>();
    const int64_t removed_count = prune_mask.sum().item<int64_t>();
    const int64_t supported_count =
        (eligible &
         (view_count.to(device).to(torch::kFloat32)
              .reshape({voxel_count}) >=
          static_cast<float>(std::max(4, opt_params_.surface_min_views_))))
            .sum()
            .item<int64_t>();

    if (removed_count > 0) {
        torch::Tensor prune_idx =
            torch::nonzero(prune_mask).reshape({-1}).to(torch::kLong);
        torch::Tensor centers = voxel_model_->voxCenter();
        torch::Tensor sizes = voxel_model_->voxSize();
        torch::Tensor levels = voxel_model_->octLevel();
        torch::Tensor colors;
        torch::Tensor sh0 = voxel_model_->sh0();
        if (sh0.defined() && sh0.dim() == 2 &&
            sh0.size(0) == voxel_count) {
            colors =
                (sh0.index_select(0, prune_idx.to(sh0.device())) *
                     sv::kSHC0 +
                 0.5f)
                    .clamp(0.0f, 1.0f)
                    .contiguous();
        }
        torch::Tensor source_surface_views = torch::ones(
            {removed_count},
            torch::TensorOptions().dtype(torch::kBool).device(device));
        appendWholeRunPrunedVoxels(
            getIteration(),
            centers.index_select(0, prune_idx.to(centers.device())),
            sizes.index_select(0, prune_idx.to(sizes.device())),
            levels.index_select(0, prune_idx.to(levels.device())),
            colors,
            torch::Tensor(),
            source_surface_views);

        voxel_model_->pruning(prune_mask);
        if (rerun_params_.run_whole_run_ ||
            rerun_params_.rerun_svrecon_debug_) {
            rerun_state_.whole_run_live_voxels_dirty_ = true;
        }
    }

    std::size_t trigger_keyframe = 0;
    if (!surface_view_pending_keyframes_.empty()) {
        trigger_keyframe = *std::max_element(
            surface_view_pending_keyframes_.begin(),
            surface_view_pending_keyframes_.end());
    }
    std::cout << "[PRUNE/surface_views] keyframe=" << trigger_keyframe
              << " window=" << cameras.size()
              << " initial=" << (initial_pass ? 1 : 0)
              << " eligible=" << eligible_count
              << " supported=" << supported_count
              << " mvs_protected=" << mvs_protected
              << " removed=" << removed_count
              << "\n";

    surface_view_pruning_initialized_ = true;
    surface_view_pending_keyframes_.clear();
}

void VoxelMapper::runFinalRefinement()
{
    if (!opt_params_.final_refinement_enable_) {
        std::cout << "[FINAL/refinement] disabled\n";
        return;
    }
    auto final_pruning_profile = profileLaptopModule("final_pruning");

    const int before = voxel_model_ ? voxel_model_->numVoxels() : 0;
    if (before <= 0) {
        return;
    }

    torch::Tensor centers = voxel_model_->voxCenter();
    torch::Device device = centers.device();
    auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(device);
    torch::Tensor sdf_prune =
        computeSvreconSdfPruneMask().to(device).to(torch::kBool);
    torch::Tensor surface_prune = torch::zeros({before}, bool_opts);
    torch::Tensor near_prune = torch::zeros({before}, bool_opts);
    torch::Tensor far_prune = torch::zeros({before}, bool_opts);
    FinalSurfacePruneStats surface_stats;

    surface_prune = computeRenderedTsdfSurfacePruneMask(&surface_stats);

    std::vector<sv::MiniCam> cameras;
    cameras.reserve(scene_->keyframes().size());
    for (const auto& item : scene_->keyframes()) {
        if (!item.second) {
            continue;
        }
        cameras.push_back(item.second->toMiniCam(
            item.second->image_height_,
            item.second->image_width_));
    }

    if (!cameras.empty()) {
        try {
            torch::Tensor octpath = voxel_model_->octPath().contiguous();
            torch::Tensor voxel_sizes = voxel_model_->voxSize().contiguous();
            if (voxel_sizes.dim() == 1) {
                voxel_sizes = voxel_sizes.view({before, 1});
            }
            if (opt_params_.filter_near_voxels_) {
                near_prune = sv::markSvreconNearDirect(
                    cameras,
                    octpath,
                    centers,
                    voxel_sizes,
                    0.2f);
                near_prune =
                    near_prune.to(device).to(torch::kBool).reshape({before});
            }

            if (opt_params_.prune_near_voxels_geometric_) {
                torch::Tensor voxel_size_1d =
                    voxel_sizes.to(device).to(torch::kFloat32)
                        .reshape({before});
                torch::Tensor radius =
                    torch::full_like(voxel_size_1d, 0.2f) +
                    0.5f * voxel_size_1d;
                torch::Tensor radius_sq = radius.square();
                torch::Tensor geometric_near =
                    torch::zeros({before}, bool_opts);
                torch::Tensor centers_f32 =
                    centers.to(device).to(torch::kFloat32).contiguous();
                for (const auto& camera : cameras) {
                    torch::Tensor camera_position =
                        camera.position.to(device).to(torch::kFloat32)
                            .view({1, 3});
                    torch::Tensor distance_sq =
                        (centers_f32 - camera_position).square().sum(/*dim=*/1);
                    geometric_near =
                        geometric_near | (distance_sq <= radius_sq);
                }
                near_prune = near_prune | geometric_near;
            }
        } catch (const std::exception& e) {
            near_prune.zero_();
            std::cerr << "[FINAL/refinement] near checks skipped: "
                      << e.what() << "\n";
        }
    }

    if (opt_params_.prune_far_voxels_) {
        try {
            voxel_model_->refreshDenseCoreBBFromCurrentVoxels();
            if (voxel_model_->hasDenseCoreBB()) {
                torch::Tensor bb_min =
                    voxel_model_->denseCoreBBMin().to(device)
                        .to(torch::kFloat32).view({1, 3});
                torch::Tensor bb_max =
                    voxel_model_->denseCoreBBMax().to(device)
                        .to(torch::kFloat32).view({1, 3});
                torch::Tensor centers_f32 =
                    centers.to(device).to(torch::kFloat32).contiguous();
                torch::Tensor in_dense_core =
                    (centers_f32 >= bb_min).all(/*dim=*/1) &
                    (centers_f32 <= bb_max).all(/*dim=*/1);
                far_prune = (~in_dense_core).to(torch::kBool);
            }
        } catch (const std::exception& e) {
            far_prune.zero_();
            std::cerr << "[FINAL/refinement] far check skipped: "
                      << e.what() << "\n";
        }
    }

    torch::Tensor leaf =
        voxel_model_->isLeaf().to(device).to(torch::kBool).reshape({before});
    sdf_prune = sdf_prune & leaf;
    surface_prune = surface_prune.to(device).to(torch::kBool) & leaf;
    near_prune = near_prune.to(device).to(torch::kBool) & leaf;
    far_prune = far_prune.to(device).to(torch::kBool) & leaf;
    if (surface_stats.rendered_tsdf_available) {
        // The fixed rendered-TSDF mesh is authoritative at shutdown. Protect
        // every adaptive cell intersecting that surface from the older final
        // SDF/near/far checks; all unsupported leaves are handled by the
        // rendered-TSDF mask itself.
        const torch::Tensor surface_supported = leaf & (~surface_prune);
        sdf_prune = sdf_prune & (~surface_supported);
        near_prune = near_prune & (~surface_supported);
        far_prune = far_prune & (~surface_supported);
    }
    torch::Tensor prune_mask =
        (sdf_prune | surface_prune | near_prune | far_prune)
            .to(torch::kBool)
            .contiguous();
    // Log one source per removed voxel. Scheduled surface-view pruning has
    // already run online; the stricter shutdown-only surface gate belongs to
    // final_refinement.
    const torch::Tensor reason_sdf = sdf_prune.contiguous();
    const torch::Tensor reason_near =
        (near_prune & (~reason_sdf)).contiguous();
    const torch::Tensor reason_far =
        (far_prune & (~reason_sdf) & (~reason_near)).contiguous();
    const torch::Tensor reason_final_refinement =
        (surface_prune & (~reason_sdf) & (~reason_near) & (~reason_far))
            .contiguous();
    const int64_t selected = prune_mask.sum().item<int64_t>();

    if (selected > 0 &&
        (rerun_params_.run_whole_run_ ||
         rerun_params_.rerun_svrecon_debug_)) {
        torch::Tensor indices = prune_mask.nonzero().squeeze(1);
        torch::Tensor sizes = voxel_model_->voxSize();
        torch::Tensor levels = voxel_model_->octLevel();
        torch::Tensor colors;
        torch::Tensor sh0 = voxel_model_->sh0();
        if (sh0.defined() && sh0.dim() == 2 && sh0.size(0) == before) {
            colors =
                (sh0.index_select(0, indices.to(sh0.device())) *
                     sv::kSHC0 +
                 0.5f)
                    .clamp(0.0f, 1.0f)
                    .contiguous();
        }
        auto select_cause = [&](const torch::Tensor& cause) {
            return cause.index_select(
                       0,
                       indices.to(cause.device()).to(torch::kLong))
                .to(centers.device())
                .to(torch::kBool)
                .contiguous();
        };
        appendWholeRunPrunedVoxels(
            getIteration(),
            centers.index_select(
                0, indices.to(centers.device()).to(torch::kLong)),
            sizes.index_select(
                0, indices.to(sizes.device()).to(torch::kLong)),
            levels.index_select(
                0, indices.to(levels.device()).to(torch::kLong)),
            colors,
            select_cause(reason_sdf),
            torch::Tensor(),
            select_cause(reason_near),
            select_cause(reason_far),
            torch::Tensor(),
            select_cause(reason_final_refinement));
    }

    if (selected > 0) {
        voxel_model_->pruning(prune_mask);
        if (rerun_params_.run_whole_run_ ||
            rerun_params_.rerun_svrecon_debug_) {
            logWholeRunLiveVoxelsToRerun(
                getIteration(),
                voxel_model_->voxCenter(),
                voxel_model_->voxSize(),
                torch::Tensor());
        }
    }

    const int after = voxel_model_->numVoxels();
    std::cout << "[FINAL/refinement] before=" << before
              << " rendered_views=" << surface_stats.rendered_view_count
              << " candidate_grid_points="
              << surface_stats.candidate_grid_point_count
              << " candidate_grid_cells="
              << surface_stats.candidate_grid_cell_count
              << " observed_grid_cells="
              << surface_stats.observed_grid_cell_count
              << " surface_grid_cells="
              << surface_stats.surface_grid_cell_count
              << " supported_voxels="
              << surface_stats.supported_voxel_count
              << " voxel_size=" << surface_stats.voxel_size
              << " truncation=" << surface_stats.truncation
              << " min_keyframe_weight="
              << surface_stats.min_keyframe_weight
              << " alpha_threshold=" << surface_stats.alpha_threshold
              << " tsdf_available="
              << (surface_stats.rendered_tsdf_available ? 1 : 0)
              << " selected=" << selected
              << " removed=" << (before - after)
              << " sdf=" << reason_sdf.sum().item<int64_t>()
              << " surface_support="
              << reason_final_refinement.sum().item<int64_t>()
              << " near=" << reason_near.sum().item<int64_t>()
              << " far=" << reason_far.sum().item<int64_t>()
              << "\n";
}
