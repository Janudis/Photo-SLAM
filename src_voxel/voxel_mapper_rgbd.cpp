#include "src_voxel/voxel_mapper_internal.h"

torch::Tensor VoxelMapper::detectRgbdRenderHolePixels(
    const std::shared_ptr<VoxelKeyframe>& pkf,
    const torch::Tensor& depth,
    const int pixel_stride,
    const bool render_on_stride_grid,
    int64_t& valid_depth_pixels,
    int64_t& hole_pixels,
    torch::Tensor& full_hole_mask)
{
    valid_depth_pixels = 0;
    hole_pixels = 0;
    full_hole_mask = torch::Tensor();
    if (!pkf || !voxel_model_ || !depth.defined() ||
        pkf->image_height_ <= 0 || pkf->image_width_ <= 0) {
        return torch::Tensor();
    }

    const int stride = std::max(1, pixel_stride);
    const int render_scale = render_on_stride_grid ? stride : 1;
    const int render_height =
        (pkf->image_height_ + render_scale - 1) / render_scale;
    const int render_width =
        (pkf->image_width_ + render_scale - 1) / render_scale;
    sv::MiniCam render_camera =
        pkf->toMiniCam(pkf->image_height_, pkf->image_width_);
    if (render_scale > 1) {
        const float scale = static_cast<float>(render_scale);
        render_camera.width = render_width;
        render_camera.height = render_height;
        render_camera.fx /= scale;
        render_camera.fy /= scale;
        render_camera.cx /= scale;
        render_camera.cy /= scale;
        const float fovx = sv::focalToFov(
            render_camera.fx, render_width);
        const float fovy = sv::focalToFov(
            render_camera.fy, render_height);
        render_camera.tanfovx = std::tan(0.5f * fovx);
        render_camera.tanfovy = std::tan(0.5f * fovy);
        render_camera.pix_size =
            2.0f * render_camera.tanfovx /
            static_cast<float>(render_width);
    }

    std::unordered_map<std::string, torch::Tensor> render_pkg;
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        render_pkg = voxel_model_->render(
            render_camera,
            render_height,
            render_width,
            torch::Tensor(),
            "dontcare",
            false,
            std::nullopt,
            true,
            false,
            true,
            false,
            false,
            sv::RenderOpts{});
    }

    torch::Tensor render_depth_cpu;
    torch::Tensor render_alpha_cpu;
    torch::Tensor render_n_contrib_cpu;
    if (!voxel_utils::renderPkgToDepthAlphaMaps(
            render_pkg,
            render_height,
            render_width,
            render_depth_cpu,
            render_alpha_cpu,
            render_n_contrib_cpu)) {
        return torch::Tensor();
    }

    torch::Tensor depth_cpu =
        depth.detach().to(torch::kCPU).to(torch::kFloat32)
            .reshape({pkf->image_height_, pkf->image_width_}).contiguous();
    std::vector<uint8_t> selected_mask(
        static_cast<size_t>(pkf->image_height_) *
        static_cast<size_t>(pkf->image_width_),
        0);
    std::vector<uint8_t> all_holes_mask(selected_mask.size(), 0);
    std::vector<int64_t> selected;
    auto depth_acc = depth_cpu.accessor<float, 2>();
    auto render_depth_acc = render_depth_cpu.accessor<float, 2>();
    auto render_n_contrib_acc = render_n_contrib_cpu.accessor<int, 2>();
    for (int render_y = 0; render_y < render_height; ++render_y) {
        const int y = render_y * render_scale;
        for (int render_x = 0; render_x < render_width; ++render_x) {
            const int x = render_x * render_scale;
            const float z_rgbd = depth_acc[y][x];
            if (!std::isfinite(z_rgbd) ||
                z_rgbd <= RGBD_min_depth_ || z_rgbd >= RGBD_max_depth_) {
                continue;
            }
            ++valid_depth_pixels;
            const float z_render = render_depth_acc[render_y][render_x];
            const int n_contrib =
                render_n_contrib_acc[render_y][render_x];
            const bool no_rendered_depth =
                !std::isfinite(z_render) || z_render <= 1.0e-6f;
            const bool structural_hole =
                n_contrib <= 0 && no_rendered_depth;
            if (!structural_hole) {
                continue;
            }
            ++hole_pixels;
            all_holes_mask[
                static_cast<size_t>(y) * pkf->image_width_ + x] = 1;
            if (render_on_stride_grid ||
                ((x % stride) == 0 && (y % stride) == 0)) {
                selected.push_back(
                    static_cast<int64_t>(y) * pkf->image_width_ + x);
            }
        }
    }

    for (const int64_t idx : selected) {
        selected_mask[static_cast<size_t>(idx)] = 1;
    }
    full_hole_mask = torch::from_blob(
                         all_holes_mask.data(),
                         {static_cast<int64_t>(all_holes_mask.size())},
                         torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
                         .clone().to(device_type_).to(torch::kBool).contiguous();
    return torch::from_blob(
               selected_mask.data(),
               {static_cast<int64_t>(selected_mask.size())},
               torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
        .clone().to(device_type_).to(torch::kBool).contiguous();
}

void VoxelMapper::fillRgbdRenderHolesSdf(
    const std::shared_ptr<VoxelKeyframe>& pkf)
{
    if (!pkf || !voxel_model_ || !rgbd_fill_render_holes_ ||
        pkf->img_undist_.empty() || pkf->img_auxiliary_undist_.empty()) {
        return;
    }

    cv::cuda::GpuMat img_rgb_gpu, img_depth_gpu;
    img_rgb_gpu.upload(pkf->img_undist_);
    img_depth_gpu.upload(pkf->img_auxiliary_undist_);
    torch::Tensor rgb =
        voxel_utils::cvGpuMatToTorchTensorFloat32(img_rgb_gpu)
            .permute({1, 2, 0})
            .flatten(0, 1)
            .contiguous();
    torch::Tensor depth =
        voxel_utils::cvGpuMatToTorchTensorFloat32(img_depth_gpu)
            .flatten(0, 1)
            .contiguous();

    const sv::Camera& camera = scene_->cameras_.at(pkf->camera_id_);
    if (camera.model_id_ != sv::Camera::PINHOLE) {
        throw std::runtime_error(
            "[VoxelMapper] RGB-D render-hole filling supports pinhole cameras only.");
    }

    int64_t valid_depth_pixels = 0;
    int64_t hole_pixels = 0;
    torch::Tensor full_hole_mask;
    torch::Tensor hole_mask = detectRgbdRenderHolePixels(
        pkf,
        depth,
        rgbd_fill_render_holes_stride_,
        /*render_on_stride_grid=*/false,
        valid_depth_pixels,
        hole_pixels,
        full_hole_mask);
    if (!hole_mask.defined() || hole_mask.numel() == 0 ||
        !hole_mask.any().item<bool>()) {
        return;
    }

    torch::Tensor points3D_camera =
        voxel_utils::reprojectDepthPinholeVoxel(
            depth, pkf->intr_, pkf->image_width_);
    torch::Tensor surface_world =
        points3D_camera.index({hole_mask}).contiguous();
    torch::Tensor surface_colors = rgb.index({hole_mask}).contiguous();
    const Sophus::SE3f Twc = pkf->getPosef().inverse();
    torch::Tensor Twc_tensor =
        voxel_utils::eigenMatrixToTorchTensor(
            Twc.matrix(), device_type_).transpose(0, 1);
    voxel_utils::transformPoints(surface_world, Twc_tensor);

    // Match the established batched densification flow: collect all hole
    // samples from the cached keyframes and perform one topology update.
    if (!rgbd_fill_render_holes_cache_points_.defined() ||
        rgbd_fill_render_holes_cache_points_.dim() != 2 ||
        rgbd_fill_render_holes_cache_points_.size(0) == 0) {
        rgbd_fill_render_holes_cache_points_ = surface_world;
        rgbd_fill_render_holes_cache_colors_ = surface_colors;
    } else {
        rgbd_fill_render_holes_cache_points_ = torch::cat(
            {rgbd_fill_render_holes_cache_points_, surface_world}, 0);
        rgbd_fill_render_holes_cache_colors_ = torch::cat(
            {rgbd_fill_render_holes_cache_colors_, surface_colors}, 0);
    }
    if (rgbd_fill_render_holes_projective_sdf_) {
        rgbd_fill_render_holes_projective_cache_.emplace_back(
            pkf, hole_mask);
    }
}

void VoxelMapper::increasePcdByKeyframeInactiveGeoDensify(
    std::shared_ptr<VoxelKeyframe> pkf,
    const bool include_inactive_geo,
    const bool include_rgbd_hole_fill)
{
    torch::NoGradGuard no_grad;

    // Pose of camera in world frame
    Sophus::SE3f Twc = pkf->getPosef().inverse();

    switch (this->sensor_type_)
    {
    case MONOCULAR:
    {
        if (include_inactive_geo) {
            assert(pkf->kps_pixel_.size() % 2 == 0);
            int N = pkf->kps_pixel_.size() / 2;

            // Keypoints and local 3D (camera frame)
            torch::Tensor kps_pixel_tensor = torch::from_blob(
                pkf->kps_pixel_.data(),
                {N, 2},
                torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);

            torch::Tensor kps_point_local_tensor = torch::from_blob(
                pkf->kps_point_local_.data(),
                {N, 3},
                torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);

            torch::Tensor kps_has3D_tensor = torch::where(
                kps_point_local_tensor.index({torch::indexing::Slice(), 2}) > 0.0f,
                true,
                false);

            // RGB image → torch
            cv::cuda::GpuMat rgb_gpu;
            rgb_gpu.upload(pkf->img_undist_);
            torch::Tensor colors = voxel_utils::cvGpuMatToTorchTensorFloat32(rgb_gpu);
            colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();

            // Photo-SLAM’s neighborhood densification
            auto result =
                voxel_utils::monocularPinholeInactiveGeoDensifyBySearchingNeighborhoodKeypoints(
                    kps_pixel_tensor,
                    kps_has3D_tensor,
                    kps_point_local_tensor,
                    colors,
                    sv::kInactiveGeoDensifyMaxPixelDist,
                    pkf->intr_,
                    pkf->image_width_);

            torch::Tensor& points3D_valid = std::get<0>(result);
            torch::Tensor& colors_valid   = std::get<1>(result);

            // Transform points to world coordinates
            torch::Tensor Twc_tensor =
                voxel_utils::eigenMatrixToTorchTensor(
                    Twc.matrix(), device_type_).transpose(0, 1);
            voxel_utils::transformPoints(points3D_valid, Twc_tensor);

            // Add new points to the cache
            if (!depth_cache_points_.defined() ||
                depth_cache_points_.numel() == 0) {
                depth_cache_points_ = points3D_valid;
                depth_cache_colors_ = colors_valid;
            } else {
                depth_cache_points_ = torch::cat(
                    {depth_cache_points_, points3D_valid}, /*dim=*/0);
                depth_cache_colors_ = torch::cat(
                    {depth_cache_colors_, colors_valid}, /*dim=*/0);
            }
        }
    }
    break;

    case RGBD:
    {
        cv::cuda::GpuMat img_rgb_gpu, img_depth_gpu;
        img_rgb_gpu.upload(pkf->img_undist_);
        img_depth_gpu.upload(pkf->img_auxiliary_undist_);

        // cv::cuda::GpuMat → torch::Tensor
        torch::Tensor rgb = voxel_utils::cvGpuMatToTorchTensorFloat32(img_rgb_gpu);
        rgb = rgb.permute({1, 2, 0}).flatten(0, 1).contiguous();

        torch::Tensor depth = voxel_utils::cvGpuMatToTorchTensorFloat32(img_depth_gpu);
        depth = depth.flatten(0, 1).contiguous();

        sv::Camera& camera = scene_->cameras_.at(pkf->camera_id_);

        // Match Photo-SLAM inactive geometry: only tracked keypoints are
        // candidates for direct depth reprojection.
        torch::Tensor point_valid_flags = torch::full(
            {depth.size(0)},
            false,   // Note Photo-SLAM uses false here and then sets only around kps
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));

        int nkps_twice = pkf->kps_pixel_.size();
        int width      = pkf->image_width_;
        std::vector<int64_t> keypoint_indices;
        keypoint_indices.reserve(static_cast<size_t>(nkps_twice / 2));
        for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
            const int x = static_cast<int>(pkf->kps_pixel_[kpidx]);
            const int y = static_cast<int>(pkf->kps_pixel_[kpidx + 1]);
            if (x >= 0 && x < width && y >= 0 && y < pkf->image_height_) {
                keypoint_indices.push_back(
                    static_cast<int64_t>(y) * width + x);
            }
        }
        if (!keypoint_indices.empty()) {
            torch::Tensor keypoint_idx = torch::from_blob(
                keypoint_indices.data(),
                {static_cast<int64_t>(keypoint_indices.size())},
                torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
                .clone()
                .to(device_type_);
            point_valid_flags.index_fill_(0, keypoint_idx, true);
        }

        torch::Tensor valid_depth =
            torch::isfinite(depth) &
            (depth > RGBD_min_depth_) &
            (depth < RGBD_max_depth_);
        point_valid_flags = point_valid_flags & valid_depth;

        torch::Tensor inactive_geo_flags =
            include_inactive_geo
            ? point_valid_flags.clone()
            : torch::zeros_like(point_valid_flags);

        // Reproject to 3D (camera coordinates)
        torch::Tensor points3D_all;

        switch (camera.model_id_)
        {
        case sv::Camera::PINHOLE:
        {
            points3D_all = voxel_utils::reprojectDepthPinholeVoxel(
                depth,
                pkf->intr_,
                pkf->image_width_);
        }
        break;

        case sv::Camera::FISHEYE:
        {
            // TODO: support fisheye camera?
            throw std::runtime_error("[VoxelMapper] Fisheye cameras are not supported currently!");
        }
        break;

        default:
        {
            throw std::runtime_error("[VoxelMapper] Invalid camera model!");
        }
        break;
        }

        torch::Tensor points3D_inactive_geo = points3D_all.index({inactive_geo_flags});
        torch::Tensor colors_inactive_geo = rgb.index({inactive_geo_flags});
        // Transform to world coordinates
        torch::Tensor Twc_tensor =
            voxel_utils::eigenMatrixToTorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        voxel_utils::transformPoints(points3D_inactive_geo, Twc_tensor);

        if (points3D_inactive_geo.defined() &&
            points3D_inactive_geo.dim() == 2 &&
            points3D_inactive_geo.size(0) > 0) {
            if (!depth_cache_points_.defined() ||
                depth_cache_points_.dim() != 2 ||
                depth_cache_points_.size(0) == 0) {
                depth_cache_points_ = points3D_inactive_geo;
                depth_cache_colors_ = colors_inactive_geo;
            } else {
                depth_cache_points_ = torch::cat(
                    {depth_cache_points_, points3D_inactive_geo}, 0);
                depth_cache_colors_ = torch::cat(
                    {depth_cache_colors_, colors_inactive_geo}, 0);
            }
        }

        if (include_rgbd_hole_fill &&
            (rgbd_fill_render_holes_ || rgbd_tsdf_evidence_)) {
            rgbd_hole_fill_keyframe_cache_.push_back(pkf);
        }

    }
    break;

    default:
    {
        throw std::runtime_error("[VoxelMapper] Unsupported sensor type!");
    }
    break;
    }

    pkf->done_inactive_geo_densify_ = true;
    ++depth_cached_;

    if (depth_cached_ >= max_depth_cached_) {
        flushInactiveGeoCache();
    }
}

void VoxelMapper::flushInactiveGeoCache()
{
    const bool have_inactive_points =
        depth_cache_points_.defined() &&
        depth_cache_points_.dim() == 2 &&
        depth_cache_points_.size(0) > 0;
    if (depth_cached_ <= 0 && !have_inactive_points &&
        rgbd_hole_fill_keyframe_cache_.empty()) {
        return;
    }
    depth_cached_ = 0;

    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        std::vector<sv::MiniCam> tr_cams = incrementalMappingCameras();

        auto flush_points =
            [&](torch::Tensor& points,
                torch::Tensor& colors,
                const std::string& entity_path)
        {
            if (!points.defined() || points.dim() != 2 || points.size(0) <= 0) {
                points = torch::Tensor();
                colors = torch::Tensor();
                return;
            }
            voxel_model_->setNextRealInsertionRerunEntityPath(entity_path);
            voxel_model_->increasePcd(
                points,
                colors,
                getIteration(),
                tr_cams);
            voxel_model_->setNextRealInsertionRerunEntityPath("");
            if (voxel_model_->lastIncreasePcdStats().new_voxels > 0 &&
                (rerun_params_.run_whole_run_ ||
                 rerun_params_.rerun_svrecon_debug_ ||
                 rerun_params_.rerun_monocular_debug_)) {
                rerun_state_.whole_run_live_voxels_dirty_ = true;
            }
            points = torch::Tensor();
            colors = torch::Tensor();
        };

        // Commit inactive geometry before residual-hole detection.
        flush_points(
            depth_cache_points_,
            depth_cache_colors_,
            "world/voxels_inactive_geo_densify/created");
    }

    // Detect residual holes only after ORB and inactive topology are present.
    rgbd_hole_fill_ready_keyframes_.insert(
        rgbd_hole_fill_ready_keyframes_.end(),
        rgbd_hole_fill_keyframe_cache_.begin(),
        rgbd_hole_fill_keyframe_cache_.end());
    rgbd_hole_fill_keyframe_cache_.clear();
}

void VoxelMapper::processRgbdClosureCache()
{
    std::vector<std::shared_ptr<VoxelKeyframe>> rgbd_closure_kfs;
    rgbd_closure_kfs.swap(rgbd_hole_fill_ready_keyframes_);
    std::unordered_set<
        sv::RgbdTsdfGridKey,
        sv::RgbdTsdfGridKeyHash> affected_evidence_cells;
    for (const auto& closure_kf : rgbd_closure_kfs) {
        if (rgbd_fill_render_holes_) {
            fillRgbdRenderHolesSdf(closure_kf);
        }
        if (rgbd_tsdf_evidence_) {
            integrateRgbdTsdfEvidenceForRenderHoles(
                closure_kf,
                affected_evidence_cells);
        }
    }

    const bool have_direct_fill =
        rgbd_fill_render_holes_cache_points_.defined() &&
        rgbd_fill_render_holes_cache_points_.dim() == 2 &&
        rgbd_fill_render_holes_cache_points_.size(0) > 0;
    if (have_direct_fill) {
        sv::VoxelModel::IncreasePcdStats stats;
        {
            std::unique_lock<std::mutex> lock_render(mutex_render_);
            voxel_model_->setNextRealInsertionRerunEntityPath(
                "world/rgbd_fill_render_holes/created");
            voxel_model_->increasePcd(
                rgbd_fill_render_holes_cache_points_,
                rgbd_fill_render_holes_cache_colors_,
                getIteration(),
                incrementalMappingCameras());
            voxel_model_->setNextRealInsertionRerunEntityPath("");
            stats = voxel_model_->lastIncreasePcdStats();

            for (const auto& item :
                 rgbd_fill_render_holes_projective_cache_) {
                fuseProjectiveSdfInitFromKeyframe(item.first, item.second);
            }
        }
        if (stats.new_voxels > 0 &&
            (rerun_params_.run_whole_run_ ||
             rerun_params_.rerun_svrecon_debug_ ||
             rerun_params_.rerun_monocular_debug_)) {
            rerun_state_.whole_run_live_voxels_dirty_ = true;
        }
    }
    rgbd_fill_render_holes_cache_points_ = torch::Tensor();
    rgbd_fill_render_holes_cache_colors_ = torch::Tensor();
    rgbd_fill_render_holes_projective_cache_.clear();

    if (rgbd_tsdf_evidence_ && !affected_evidence_cells.empty()) {
        promoteRgbdTsdfEvidenceCells(affected_evidence_cells);
    }

    // Re-fuse after deferred evidence promotion or direct batched insertion so
    // newly allocated cells receive projective D-z corner values.
    if (initial_mapped_ && sdf_initialization_rgbd_projective_) {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        for (const auto& closure_kf : rgbd_closure_kfs) {
            fuseProjectiveSdfInitFromKeyframe(closure_kf);
        }
    }
}
