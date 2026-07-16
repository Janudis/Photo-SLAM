#include "include_voxel/voxel_mapper.h"
#include "include_voxel/voxel_mapper_utils.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <unordered_map>
#include <vector>

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
        tensor_utils::cvMat2TorchTensor_Float32(
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

torch::Tensor VoxelMapper::computeSdfInitForGridPoints(
    const torch::Tensor& grid_points_world,
    float ray_interval_m)
{
    TORCH_CHECK(
        grid_points_world.defined() &&
        grid_points_world.dim() == 2 &&
        grid_points_world.size(1) == 3,
        "computeSdfInitForGridPoints expects grid_points_world [N,3]");

    const int64_t N = grid_points_world.size(0);
    auto opts = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(grid_points_world.device());
    torch::Tensor sdf_init =
        torch::full({N, 1}, std::numeric_limits<float>::quiet_NaN(), opts);
    if (N == 0 || sensor_type_ != RGBD) {
        return sdf_init;
    }

    if (sdf_state_.projective_sdf_init_context_valid_) {
        return computeProjectiveSdfInitForGridPoints(
            grid_points_world,
            ray_interval_m);
    }

    return sdf_init;
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

torch::Tensor VoxelMapper::computeFinalSurfaceConfidenceKeepMask(
    bool retain_connected)
{
    const char* log_tag = retain_connected
        ? "[FINAL/surface]"
        : "[PRUNE/surface_views]";
    const int64_t voxel_count = voxel_model_ ? voxel_model_->numVoxels() : 0;
    torch::Device device = mDevice;
    if (voxel_model_ && voxel_model_->voxCenter().defined()) {
        device = voxel_model_->voxCenter().device();
    }
    auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(device);
    if (voxel_count <= 0 || !voxel_model_) {
        return torch::empty({0}, bool_opts);
    }

    auto keep_all = torch::ones({voxel_count}, bool_opts);
    torch::Tensor centers =
        voxel_model_->voxCenter().to(device).to(torch::kFloat32).contiguous();
    torch::Tensor sizes =
        voxel_model_->voxSize().to(device).to(torch::kFloat32).reshape({voxel_count}).contiguous();
    torch::Tensor sdf =
        voxel_model_->voxelGeoCorners().to(device).to(torch::kFloat32).contiguous();
    torch::Tensor vox_key =
        voxel_model_->voxKey().to(device).to(torch::kLong).contiguous();
    if (centers.dim() != 2 || centers.size(0) != voxel_count || centers.size(1) != 3 ||
        sizes.numel() != voxel_count || sdf.dim() != 2 || sdf.size(0) != voxel_count ||
        sdf.size(1) != 8 || vox_key.dim() != 2 || vox_key.size(0) != voxel_count ||
        vox_key.size(1) != 8) {
        std::cerr << log_tag << " skipped: inconsistent voxel tensors\n";
        return keep_all;
    }

    torch::Tensor finite = torch::isfinite(sdf).all(/*dim=*/1);
    torch::Tensor sdf_min = std::get<0>(sdf.min(/*dim=*/1));
    torch::Tensor sdf_max = std::get<0>(sdf.max(/*dim=*/1));
    torch::Tensor sdf_span = sdf_max - sdf_min;
    const torch::Tensor corner_offsets = torch::tensor(
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
         {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 1.0f},
         {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 1.0f},
         {1.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    const torch::Tensor edge_begin = torch::tensor(
        {0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6},
        torch::TensorOptions().dtype(torch::kLong).device(device));
    const torch::Tensor edge_end = torch::tensor(
        {1, 2, 4, 3, 5, 3, 6, 7, 5, 6, 7, 7},
        torch::TensorOptions().dtype(torch::kLong).device(device));
    torch::Tensor corner_xyz =
        centers.unsqueeze(1) +
        (corner_offsets.unsqueeze(0) - 0.5f) * sizes.view({voxel_count, 1, 1});
    torch::Tensor sdf_begin = sdf.index_select(1, edge_begin);
    torch::Tensor sdf_end = sdf.index_select(1, edge_end);
    torch::Tensor has_positive = (sdf > 0.0f).any(/*dim=*/1);
    torch::Tensor has_negative = (sdf < 0.0f).any(/*dim=*/1);
    torch::Tensor strict_sign_change = has_positive & has_negative;

    // A surface edge must have endpoints on strict opposite sides of zero.
    // Merely touching zero from one side is not a crossing.
    torch::Tensor edge_crossing =
        ((sdf_begin > 0.0f) & (sdf_end < 0.0f)) |
        ((sdf_begin < 0.0f) & (sdf_end > 0.0f));
    torch::Tensor interp =
        sdf_begin.abs() / (sdf_begin.abs() + sdf_end.abs()).clamp_min(1.0e-8f);
    torch::Tensor edge_xyz_begin = corner_xyz.index_select(1, edge_begin);
    torch::Tensor edge_xyz_end = corner_xyz.index_select(1, edge_end);
    torch::Tensor edge_xyz =
        edge_xyz_begin + interp.unsqueeze(2) * (edge_xyz_end - edge_xyz_begin);
    torch::Tensor crossing_count =
        edge_crossing.to(torch::kFloat32).sum(/*dim=*/1).clamp_min(1.0f);
    torch::Tensor crossing_points =
        (edge_xyz * edge_crossing.unsqueeze(2).to(torch::kFloat32)).sum(/*dim=*/1) /
        crossing_count.unsqueeze(1);
    torch::Tensor strict_zero_crossing =
        finite & strict_sign_change & edge_crossing.any(/*dim=*/1);
    torch::Tensor span_pass =
        sdf_span >= opt_params_.final_surface_min_sdf_span_vox_ * sizes;
    torch::Tensor zero_crossing = strict_zero_crossing & span_pass;
    crossing_points = torch::where(
        zero_crossing.unsqueeze(1),
        crossing_points,
        centers);

    torch::Tensor surface_views = torch::zeros(
        {voxel_count},
        torch::TensorOptions().dtype(torch::kInt32).device(device));
    torch::Tensor observed_views = torch::zeros_like(surface_views);
    int depth_keyframes = 0;

    if (sensor_type_ == RGBD) {
        const torch::Tensor depth_tolerance =
            opt_params_.final_surface_depth_tolerance_vox_ * sizes;

        for (const auto& item : scene_->keyframes()) {
            const auto& kf = item.second;
            if (!kf || kf->img_auxiliary_undist_.empty()) {
                continue;
            }

            cv::Mat depth_meters;
            if (!voxel_utils::depthMatToMeters(kf->img_auxiliary_undist_, depth_meters) ||
                depth_meters.empty()) {
                continue;
            }
            if (depth_meters.channels() > 1) {
                cv::extractChannel(depth_meters, depth_meters, 0);
            }
            if (depth_meters.type() != CV_32FC1) {
                depth_meters.convertTo(depth_meters, CV_32FC1);
            }
            if (depth_meters.rows <= 0 || depth_meters.cols <= 0) {
                continue;
            }

            const int height = depth_meters.rows;
            const int width = depth_meters.cols;
            sv::MiniCam cam = kf->toMiniCam(height, width);
            if (!cam.w2c.defined() || cam.w2c.numel() < 16 ||
                cam.fx <= 1.0e-6f || cam.fy <= 1.0e-6f) {
                continue;
            }

            torch::Tensor depth = torch::from_blob(
                                      depth_meters.data,
                                      {height, width},
                                      torch::TensorOptions().dtype(torch::kFloat32))
                                      .clone()
                                      .to(device)
                                      .contiguous();
            torch::Tensor w2c =
                cam.w2c.to(device).to(torch::kFloat32).contiguous();
            torch::Tensor rotation = w2c.index({
                torch::indexing::Slice(0, 3),
                torch::indexing::Slice(0, 3)});
            torch::Tensor translation =
                w2c.index({torch::indexing::Slice(0, 3), 3}).view({1, 3});
            torch::Tensor camera_points =
                torch::matmul(crossing_points, rotation.transpose(0, 1)) + translation;
            torch::Tensor x = camera_points.index({torch::indexing::Slice(), 0});
            torch::Tensor y = camera_points.index({torch::indexing::Slice(), 1});
            torch::Tensor z = camera_points.index({torch::indexing::Slice(), 2});
            torch::Tensor z_safe = z.clamp_min(1.0e-6f);
            torch::Tensor u = cam.fx * x / z_safe + cam.cx;
            torch::Tensor v = cam.fy * y / z_safe + cam.cy;
            torch::Tensor in_image =
                torch::isfinite(z) & (z > std::max(1.0e-6f, RGBD_min_depth_)) &
                torch::isfinite(u) & torch::isfinite(v) &
                (u >= 0.0f) & (u < static_cast<float>(width)) &
                (v >= 0.0f) & (v < static_cast<float>(height));

            torch::Tensor u_index = torch::floor(
                torch::where(torch::isfinite(u), u, torch::zeros_like(u)))
                                        .clamp(0.0f, static_cast<float>(width - 1))
                                        .to(torch::kLong);
            torch::Tensor v_index = torch::floor(
                torch::where(torch::isfinite(v), v, torch::zeros_like(v)))
                                        .clamp(0.0f, static_cast<float>(height - 1))
                                        .to(torch::kLong);
            torch::Tensor measured_depth = depth.reshape({height * width}).index_select(
                0,
                v_index * width + u_index);
            torch::Tensor valid_depth =
                in_image & torch::isfinite(measured_depth) &
                (measured_depth > RGBD_min_depth_) &
                (measured_depth < RGBD_max_depth_);
            torch::Tensor surface_hit =
                valid_depth & ((z - measured_depth).abs() <= depth_tolerance);
            torch::Tensor observed =
                valid_depth & (z <= (measured_depth + depth_tolerance));

            surface_views += surface_hit.to(torch::kInt32);
            observed_views += observed.to(torch::kInt32);
            ++depth_keyframes;
        }
    }

    torch::Tensor strong_seed;
    torch::Tensor connected_candidate;
    if (depth_keyframes > 0) {
        strong_seed =
            zero_crossing &
            (surface_views >= opt_params_.final_surface_min_views_);
        connected_candidate = zero_crossing.clone();
    } else {
        std::vector<sv::MiniCam> cameras;
        cameras.reserve(scene_->keyframes().size());
        for (const auto& item : scene_->keyframes()) {
            if (item.second) {
                cameras.push_back(item.second->toMiniCam(
                    item.second->image_height_, item.second->image_width_));
            }
        }
        if (cameras.empty()) {
            std::cerr << log_tag << " skipped: no keyframes provide confidence evidence\n";
            return keep_all;
        }
        auto stat = voxel_model_->computeTrainingStat(cameras);
        torch::Tensor view_count =
            stat.view_cnt.to(device).to(torch::kFloat32).reshape({voxel_count});
        observed_views = view_count.to(torch::kInt32);
        surface_views = observed_views.clone();
        strong_seed =
            zero_crossing &
            (view_count >= static_cast<float>(opt_params_.final_surface_min_views_));
        connected_candidate = zero_crossing.clone();
    }

    const int64_t strong_count = strong_seed.sum().item<int64_t>();
    const int64_t candidate_count = connected_candidate.sum().item<int64_t>();
    const int64_t strict_crossing_count =
        strict_zero_crossing.sum().item<int64_t>();
    const int64_t span_reject_count =
        (strict_zero_crossing & (~span_pass)).sum().item<int64_t>();
    const int64_t no_depth_support_count =
        (zero_crossing & (surface_views <= 0)).sum().item<int64_t>();
    const int64_t below_min_views_count =
        (zero_crossing & (surface_views > 0) &
         (surface_views < opt_params_.final_surface_min_views_))
            .sum().item<int64_t>();
    if (strong_count <= 0) {
        std::cerr << log_tag << " skipped destructive refinement: no strong surface seeds"
                  << " strict_crossing=" << strict_crossing_count
                  << " span_reject=" << span_reject_count
                  << " zero_crossing=" << zero_crossing.sum().item<int64_t>()
                  << " no_depth_support=" << no_depth_support_count
                  << " below_min_views=" << below_min_views_count
                  << " candidates=" << candidate_count
                  << " depth_keyframes=" << depth_keyframes << "\n";
        return keep_all;
    }

    torch::Tensor keep = strong_seed.clone();
    if (retain_connected &&
        opt_params_.final_surface_keep_connected_ &&
        candidate_count > strong_count) {
        torch::Tensor candidate_cpu =
            connected_candidate.to(torch::kCPU).to(torch::kBool).contiguous();
        torch::Tensor strong_cpu =
            strong_seed.to(torch::kCPU).to(torch::kBool).contiguous();
        torch::Tensor vox_key_cpu =
            vox_key.to(torch::kCPU).to(torch::kLong).contiguous();
        const bool* candidate_ptr = candidate_cpu.data_ptr<bool>();
        const bool* strong_ptr = strong_cpu.data_ptr<bool>();
        const int64_t* vox_key_ptr = vox_key_cpu.data_ptr<int64_t>();

        std::vector<std::vector<int64_t>> adjacency(static_cast<size_t>(voxel_count));
        std::unordered_map<int64_t, std::vector<int64_t>> corner_owners;
        corner_owners.reserve(static_cast<size_t>(candidate_count) * 4);
        for (int64_t voxel = 0; voxel < voxel_count; ++voxel) {
            if (!candidate_ptr[voxel]) {
                continue;
            }
            for (int corner = 0; corner < 8; ++corner) {
                const int64_t key = vox_key_ptr[voxel * 8 + corner];
                auto& owners = corner_owners[key];
                for (const int64_t neighbor : owners) {
                    adjacency[static_cast<size_t>(voxel)].push_back(neighbor);
                    adjacency[static_cast<size_t>(neighbor)].push_back(voxel);
                }
                owners.push_back(voxel);
            }
        }

        std::vector<int32_t> distance(static_cast<size_t>(voxel_count), -1);
        std::deque<int64_t> frontier;
        for (int64_t voxel = 0; voxel < voxel_count; ++voxel) {
            if (strong_ptr[voxel]) {
                distance[static_cast<size_t>(voxel)] = 0;
                frontier.push_back(voxel);
            }
        }
        const int max_hops = std::max(0, opt_params_.final_surface_connected_hops_);
        while (!frontier.empty()) {
            const int64_t voxel = frontier.front();
            frontier.pop_front();
            const int current_distance = distance[static_cast<size_t>(voxel)];
            if (current_distance >= max_hops) {
                continue;
            }
            for (const int64_t neighbor : adjacency[static_cast<size_t>(voxel)]) {
                if (distance[static_cast<size_t>(neighbor)] >= 0) {
                    continue;
                }
                distance[static_cast<size_t>(neighbor)] = current_distance + 1;
                frontier.push_back(neighbor);
            }
        }

        torch::Tensor keep_cpu = torch::zeros(
            {voxel_count}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
        bool* keep_ptr = keep_cpu.data_ptr<bool>();
        for (int64_t voxel = 0; voxel < voxel_count; ++voxel) {
            keep_ptr[voxel] =
                candidate_ptr[voxel] && distance[static_cast<size_t>(voxel)] >= 0;
        }
        keep = keep_cpu.to(device).to(torch::kBool).contiguous();
    }

    const int64_t keep_count = keep.sum().item<int64_t>();
    std::cout << log_tag << " total=" << voxel_count
              << " depth_keyframes=" << depth_keyframes
              << " observed=" << (observed_views > 0).sum().item<int64_t>()
              << " strict_crossing=" << strict_crossing_count
              << " span_reject=" << span_reject_count
              << " zero_crossing=" << zero_crossing.sum().item<int64_t>()
              << " no_depth_support=" << no_depth_support_count
              << " below_min_views=" << below_min_views_count
              << " candidates=" << candidate_count
              << " strong=" << strong_count
              << " connected_saved=" << std::max<int64_t>(0, keep_count - strong_count)
              << " keep=" << keep_count
              << " prune=" << (voxel_count - keep_count)
              << "\n";
    return keep.to(device).to(torch::kBool).contiguous();
}

void VoxelMapper::runFinalSpecialPrune()
{
    if (!opt_params_.final_special_prune_enable_ &&
        !opt_params_.final_surface_prune_enable_) {
        std::cout << "[FINAL/prune] disabled\n";
        return;
    }

    const int before_final_special = voxel_model_->numVoxels();
    if (before_final_special <= 0) {
        return;
    }

    auto centers = voxel_model_->voxCenter();
    auto sizes = voxel_model_->voxSize();
    if (sizes.defined() && sizes.dim() == 2 && sizes.size(1) == 1) {
        sizes = sizes.squeeze(1);
    } else if (sizes.defined() && sizes.dim() != 1) {
        sizes = sizes.reshape({-1});
    }
    auto prune_mask_final_special = torch::zeros(
        {before_final_special},
        torch::TensorOptions().dtype(torch::kBool).device(centers.device()));

    int64_t n_near_final_special = 0;
    int64_t n_near_geom_final_special = 0;
    bool near_valid_final_special = false;

    if (opt_params_.final_special_prune_enable_) {
        std::vector<sv::MiniCam> tr_cams;
        tr_cams.reserve(scene_->keyframes().size());
        for (const auto& kv : scene_->keyframes()) {
            tr_cams.push_back(kv.second->toMiniCam(
                kv.second->image_height_,
                kv.second->image_width_));
        }

        if (tr_cams.empty()) {
            n_near_final_special = 0;
            near_valid_final_special = true;
        } else {
            try {
                auto octpath = voxel_model_->octPath().contiguous();
                auto vox_center = voxel_model_->voxCenter().contiguous();
                auto vox_size = voxel_model_->voxSize().contiguous();

                TORCH_CHECK(octpath.size(0) == before_final_special,
                            "octpath length mismatch at final special prune");
                TORCH_CHECK(vox_center.size(0) == before_final_special,
                            "vox_center length mismatch at final special prune");
                TORCH_CHECK(vox_center.size(1) == 3,
                            "vox_center must be [N,3] at final special prune");
                if (vox_size.dim() == 1) {
                    vox_size = vox_size.view({before_final_special, 1});
                } else if (vox_size.dim() == 2) {
                    TORCH_CHECK(vox_size.size(0) == before_final_special,
                                "vox_size length mismatch at final special prune");
                } else {
                    TORCH_CHECK(false, "vox_size must be [N] or [N,1] at final special prune");
                }

                const float near_thresh = 0.2f;
                at::Tensor is_near =
                    sv::markSvreconNearDirect(
                        tr_cams,
                        octpath,
                        vox_center,
                        vox_size,
                        near_thresh);
                if (is_near.dim() == 2 && is_near.size(1) == 1) {
                    is_near = is_near.squeeze(1);
                }
                is_near = is_near.to(torch::kBool);
                at::Tensor is_near_geom = torch::zeros(
                    {before_final_special},
                    torch::TensorOptions().dtype(torch::kBool).device(is_near.device()));
                if (opt_params_.prune_near_voxels_geometric_) {
                    auto vox_center_f32 =
                        vox_center.to(is_near.device()).to(torch::kFloat32).contiguous();
                    auto vox_size_1d =
                        vox_size.to(is_near.device()).to(torch::kFloat32).contiguous();
                    if (vox_size_1d.dim() == 2 && vox_size_1d.size(1) == 1) {
                        vox_size_1d = vox_size_1d.squeeze(1);
                    } else if (vox_size_1d.dim() != 1) {
                        vox_size_1d = vox_size_1d.reshape({-1});
                    }
                    TORCH_CHECK(vox_size_1d.numel() == before_final_special,
                                "vox_size_1d.numel() != N in final special prune geometric near filter");

                    auto near_radius =
                        (torch::full_like(vox_size_1d, near_thresh) + 0.5f * vox_size_1d)
                            .contiguous();
                    auto near_radius_sq = (near_radius * near_radius).contiguous();

                    for (const auto& c : tr_cams) {
                        auto cam_pos =
                            c.position.to(is_near.device()).to(torch::kFloat32).view({1, 3});
                        auto d2 = (vox_center_f32 - cam_pos).pow(2).sum(/*dim=*/1);
                        is_near_geom =
                            (is_near_geom | (d2 <= near_radius_sq)).to(torch::kBool);
                    }
                }
                n_near_final_special = is_near.sum().item<int64_t>();
                n_near_geom_final_special = is_near_geom.sum().item<int64_t>();
                auto near_union = (is_near | is_near_geom).to(torch::kBool);
                prune_mask_final_special =
                    (prune_mask_final_special |
                     near_union.to(prune_mask_final_special.device()).to(torch::kBool)).to(torch::kBool);
                near_valid_final_special = true;
            } catch (const std::exception& e) {
                std::cerr << "[FINAL/special_prune] failed to compute near voxels: "
                          << e.what() << "\n";
            }
        }
    }

    int64_t n_selected_surface_confidence = 0;
    if (opt_params_.final_surface_prune_enable_) {
        torch::Tensor surface_keep = computeFinalSurfaceConfidenceKeepMask();
        if (surface_keep.defined() && surface_keep.numel() == before_final_special) {
            torch::Tensor surface_prune =
                (~surface_keep.to(prune_mask_final_special.device()).to(torch::kBool))
                    .contiguous();
            n_selected_surface_confidence = surface_prune.sum().item<int64_t>();
            prune_mask_final_special =
                (prune_mask_final_special | surface_prune).to(torch::kBool);
        } else {
            std::cerr << "[FINAL/surface] skipped: confidence mask shape mismatch\n";
        }
    }

    const int64_t n_selected_final_special =
        prune_mask_final_special.to(torch::kBool).sum().item<int64_t>();
    if (n_selected_final_special > 0) {
        torch::Tensor final_idx =
            prune_mask_final_special.to(torch::kBool).nonzero().squeeze(1);
        if (final_idx.defined() && final_idx.numel() > 0 &&
            centers.defined() && centers.dim() == 2 &&
            centers.size(0) == before_final_special &&
            sizes.defined() && sizes.numel() == before_final_special) {
            torch::Tensor final_centers =
                centers.index_select(0, final_idx.to(centers.device()).to(torch::kLong)).contiguous();
            torch::Tensor sizes_for_log = sizes;
            if (sizes_for_log.dim() == 2 && sizes_for_log.size(1) == 1) {
                sizes_for_log = sizes_for_log.squeeze(1);
            } else if (sizes_for_log.dim() != 1) {
                sizes_for_log = sizes_for_log.reshape({before_final_special});
            }
            torch::Tensor final_sizes =
                sizes_for_log.index_select(0, final_idx.to(sizes_for_log.device()).to(torch::kLong)).contiguous();
            torch::Tensor final_special_mask = torch::ones(
                {final_idx.numel()},
                torch::TensorOptions().dtype(torch::kBool).device(final_centers.device()));
            appendWholeRunPrunedVoxels(
                getIteration(),
                final_centers,
                final_sizes,
                torch::Tensor(),
                torch::Tensor(),
                final_special_mask);
        }
    }

    if (n_selected_final_special > 0) {
        voxel_model_->pruning(prune_mask_final_special);
        if (rerun_params_.run_whole_run_ ||
            rerun_params_.rerun_svrecon_debug_) {
            logWholeRunLiveVoxelsToRerun(
                getIteration(),
                voxel_model_->voxCenter(),
                voxel_model_->voxSize(),
                torch::Tensor());
        }
    }

    const int after_final_special = voxel_model_->numVoxels();
    std::cout << "[FINAL/prune] before=" << before_final_special
              << " selected=" << n_selected_final_special
              << " removed=" << (before_final_special - after_final_special)
              << " surface_confidence=" << n_selected_surface_confidence
              << " near=" << (near_valid_final_special ? std::to_string(n_near_final_special) : std::string("N/A"))
              << " near_geom=" << (near_valid_final_special ? std::to_string(n_near_geom_final_special) : std::string("N/A"))
              << "\n";
}
