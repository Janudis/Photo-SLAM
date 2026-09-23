#include "src_voxel/voxel_model_internal.h"

namespace sv {

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
    // Refreshes existing learned corners near newly observed sparse support.
    // SVRecon reference: sdf_init_utils.initialize_sdf_from_sfm.
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
        sv::kSdfInitializationOrbRadiusVox * fixed_vox_size_;
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

    ensureFusedSdfField();
    if (!fused_sdf_grid_pts_.defined() ||
        !fused_sdf_weights_.defined() ||
        fused_sdf_grid_pts_.size(0) != _geo_grid_pts_.size(0) ||
        fused_sdf_weights_.size(0) != _geo_grid_pts_.size(0)) {
        resetFusedSdfField();
    }

    torch::NoGradGuard no_grad;
    torch::Tensor old_w =
        fused_sdf_weights_.index_select(0, update_idx).to(torch::kFloat32).contiguous();
    torch::Tensor old_sdf =
        _geo_grid_pts_.detach().index_select(0, update_idx).to(torch::kFloat32).contiguous();
    torch::Tensor new_w = old_w + 1.0f;
    torch::Tensor fused_sdf =
        (old_sdf * old_w + measured_sdf) / new_w.clamp_min(1.0e-6f);

    _geo_grid_pts_.index_put_({update_idx}, fused_sdf);
    fused_sdf_grid_pts_.index_put_({update_idx}, fused_sdf.detach());
    fused_sdf_weights_.index_put_({update_idx}, new_w.detach().clamp_max(100.0f));
}

torch::Tensor VoxelModel::visibilitySignedPointPriorSdf_(
    const torch::Tensor& grid_xyz,
    const torch::Tensor& dist,
    const torch::Tensor& points,
    float surface_band) const
{
    // Assigns point-distance signs using camera visibility before initializing
    // new corners. SVRecon reference: sdf_init_utils.initialize_sdf_from_sfm.
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
    // Produces initial learned SDF rows from the selected online source mode,
    // falling back to a weak prior when no trusted support is nearby.
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
        // Keep the useful property that a newly inserted voxel starts
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
            sv::kSdfInitializationOrbRadiusVox * fixed_vox_size_;
        torch::Tensor local_orb = dist <= radius_m;
        // Only nearby ORB points provide a strong metric prior. Unsupported
        // corners retain only a tiny zero-centered signed ramp derived from the
        // ORB field, so inactive/RGB-D allocations can render weakly without
        // becoming geometric priors themselves.
        sdf = torch::where(local_orb, sdf, weak_signed_sdf).contiguous();
    }
    if (pending_sdf_init_mode_ == SdfInitMode::OrbPriorWeakCandidate) {
        const float radius_m =
            sv::kSdfInitializationOrbRadiusVox * fixed_vox_size_;
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
    // Stages exact corner values for the next topology insertion, used when a
    // promoted evidence/MVS cell already has a geometric SDF estimate.
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
    // Rebuilds shared-corner parameters after topology changes while preserving
    // matching values, Adam state, and auxiliary fused-SDF evidence.
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
        if (fused_sdf_grid_pts_.defined() || fused_sdf_weights_.defined()) {
            fused_sdf_grid_pts_ = torch::zeros_like(new_geo);
            fused_sdf_weights_ = torch::zeros_like(new_geo);
        }
        return;
    }
    if (M_new == 0) {
        _geo_grid_pts_ = torch::empty(
            {0, 1},
            torch::TensorOptions().dtype(torch::kFloat32).device(dev)).requires_grad_(true);
        adam_geo_.exp_avg = torch::zeros_like(_geo_grid_pts_);
        adam_geo_.exp_avg_sq = torch::zeros_like(_geo_grid_pts_);
        if (fused_sdf_grid_pts_.defined() || fused_sdf_weights_.defined()) {
            fused_sdf_grid_pts_ = torch::zeros_like(_geo_grid_pts_);
            fused_sdf_weights_ = torch::zeros_like(_geo_grid_pts_);
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

    if (fused_sdf_grid_pts_.defined() || fused_sdf_weights_.defined()) {
        auto value_opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
        torch::Tensor new_sdf = torch::zeros({M_new, 1}, value_opts);
        torch::Tensor new_w = torch::zeros({M_new, 1}, value_opts);
        if (matched_new_idx.numel() > 0 &&
            fused_sdf_grid_pts_.defined() &&
            fused_sdf_weights_.defined() &&
            fused_sdf_grid_pts_.size(0) == old_linear.size(0) &&
            fused_sdf_weights_.size(0) == old_linear.size(0)) {
            torch::Tensor old_sdf =
                fused_sdf_grid_pts_.to(dev).to(torch::kFloat32).contiguous();
            torch::Tensor old_w =
                fused_sdf_weights_.to(dev).to(torch::kFloat32).contiguous();
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
        fused_sdf_grid_pts_ = new_sdf.contiguous();
        fused_sdf_weights_ = new_w.contiguous();
    }

    _geo_grid_pts_ = new_geo.contiguous().detach().requires_grad_(true);
}

void VoxelModel::applyGeoGridRawInit(
    const torch::Tensor& raw_values,
    const torch::Tensor& valid_mask)
{
    // Copies valid projective SDF samples into the learnable corner field.
    // This is the online RGB-D initialization bridge, not ongoing TSDF fusion.
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
    // Quantizes surface samples to one insertion-level cell per measurement and
    // returns the source-point mapping used by direct and learned-depth filling.
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

void VoxelModel::refreshSvreconLogSTargetFromVoxelSize(const bool initialize_current)
{
    // Recomputes NeuS sharpness from the finest active voxel size.
    // SVRecon reference: SVConstructor.model_init log_s initialization.
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
    if (initialize_current || !log_s_.defined() || log_s_.numel() == 0) {
        torch::NoGradGuard no_grad;
        if (!log_s_.defined() || log_s_.numel() == 0) {
            log_s_ = torch::full(
                {1}, target_log_s,
                torch::TensorOptions().dtype(torch::kFloat32).device(size_.device()));
        } else {
            log_s_.fill_(target_log_s);
        }
        adam_log_s_ = AdamGroupState{};
    }
}

} // namespace sv
