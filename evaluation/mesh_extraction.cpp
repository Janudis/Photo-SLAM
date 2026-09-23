#include "mesh_extraction_internal.h"

torch::Tensor VoxelMapper::colorizeSvreconMeshVertices(
    const torch::Tensor& vertices)
{
    auto points = vertices.detach().to(torch::kFloat32).contiguous();
    auto closest_color = torch::full(
        {points.size(0), 3}, 0.5f, points.options());
    auto closest_dist = torch::full(
        {points.size(0)}, std::numeric_limits<float>::infinity(), points.options());

    const auto keyframes = scene_->getAllKeyframes();
    for (const auto& [kf_id, pkf] : keyframes) {
        if (!pkf || !pkf->set_pose_) continue;
        const int height = std::max(1, pkf->image_height_);
        const int width = std::max(1, pkf->image_width_);
        const auto cam = pkf->toMiniCam(height, width);

        sv::RenderOpts render_opts;
        render_opts.output_depth = true;
        render_opts.output_T = true;
        auto render_pkg = voxel_model_->render(
            cam, height, width, torch::Tensor(), "sh0", false, std::nullopt,
            true, false, true, false, false, render_opts);
        auto color_it = render_pkg.find("color");
        auto depth_it = render_pkg.find("raw_depth");
        auto transmittance_it = render_pkg.find("raw_T");
        if (color_it == render_pkg.end() || depth_it == render_pkg.end() ||
            transmittance_it == render_pkg.end()) {
            continue;
        }

        auto raw_depth = depth_it->second.detach().to(points.device()).to(torch::kFloat32);
        if (raw_depth.dim() == 4 && raw_depth.size(0) == 1) raw_depth = raw_depth.squeeze(0);
        const int64_t depth_channel = rerun_params_.svrecon_mesh_use_mean_depth_
            ? 0
            : std::min<int64_t>(2, raw_depth.size(0) - 1);
        auto frame_depth = raw_depth.index({depth_channel}).contiguous();
        auto frame_alpha =
            (1.0f - transmittance_it->second.detach()
                        .to(points.device()).to(torch::kFloat32)).contiguous();
        auto frame_color =
            color_it->second.detach().to(points.device()).to(torch::kFloat32).contiguous();

        auto uv = projectNormalized(points, cam);
        auto camera_position =
            cam.position.to(points.device()).to(torch::kFloat32).view({1, 3});
        auto camera_lookat =
            cam.lookat.to(points.device()).to(torch::kFloat32).view({3, 1});
        auto all_point_depth =
            torch::matmul(points - camera_position, camera_lookat).flatten();
        auto valid_idx = torch::nonzero(
                             (uv.abs() <= 1.0f).all(1) & (all_point_depth > 0.0f))
                             .view({-1}).to(torch::kLong);
        if (valid_idx.numel() == 0) continue;
        auto valid_uv = uv.index_select(0, valid_idx);
        auto valid_points = points.index_select(0, valid_idx);

        auto sampled_alpha = sampleBilinear(frame_alpha, valid_uv);
        auto alpha_idx = torch::nonzero(
                             sampled_alpha > sv::kSvreconMeshAlphaThreshold)
                             .view({-1}).to(torch::kLong);
        if (alpha_idx.numel() == 0) continue;
        valid_idx = valid_idx.index_select(0, alpha_idx);
        valid_uv = valid_uv.index_select(0, alpha_idx);
        valid_points = valid_points.index_select(0, alpha_idx);

        auto sampled_depth = sampleBilinear(frame_depth, valid_uv);
        auto point_depth = all_point_depth.index_select(0, valid_idx);
        auto point_dist = (sampled_depth - point_depth).abs();
        auto better = point_dist < closest_dist.index_select(0, valid_idx);
        auto better_idx = torch::nonzero(better).view({-1}).to(torch::kLong);
        if (better_idx.numel() == 0) continue;
        valid_idx = valid_idx.index_select(0, better_idx);
        valid_uv = valid_uv.index_select(0, better_idx);
        point_dist = point_dist.index_select(0, better_idx);
        auto point_color = sampleBilinearChannels(frame_color, valid_uv);
        closest_dist.index_put_({valid_idx}, point_dist);
        closest_color.index_put_({valid_idx}, point_color);
    }
    return closest_color.contiguous();
}

void VoxelMapper::saveSvreconSdfMeshPly(const std::filesystem::path& result_path)
{
    namespace fs = std::filesystem;
    torch::NoGradGuard no_grad;
    if (!voxel_model_) {
        std::cout << "[SVRecon mesh/SDF] skipped: voxel model is not initialized.\n";
        return;
    }
    if (!result_path.parent_path().empty()) {
        fs::create_directories(result_path.parent_path());
    }

    auto grid_sdf = voxel_model_->geoGridPts().detach().view({-1}).contiguous();
    auto grid_xyz = voxel_model_->gridPointsWorld().detach().contiguous();
    auto vox_key = voxel_model_->voxKey().detach().to(torch::kLong).contiguous();

    auto [vertices, faces] = marchingCubesGrid(
        grid_sdf,
        grid_xyz,
        vox_key,
        /*iso=*/0.0f);
    auto mesh = meshFromTensors(vertices, faces);
    if (rerun_params_.svrecon_mesh_use_vert_color_ && vertices.numel() > 0) {
        std::unique_lock<std::mutex> render_lock(mutex_render_);
        setMeshColors(mesh, colorizeSvreconMeshVertices(vertices));
    }
    if (!saveTriangleMeshPly(
            result_path,
            mesh,
            rerun_params_.svrecon_mesh_use_vert_color_)) {
        throw std::runtime_error(
            "saveSvreconSdfMeshPly: direct learned-SDF extraction produced an empty mesh");
    }
    std::cout << "[SVRecon mesh/SDF] wrote " << result_path
              << " voxels=" << vox_key.size(0)
              << " vertices=" << mesh.vertices.size()
              << " faces=" << mesh.faces.size() << "\n";
}

void VoxelMapper::saveRenderedTsdfMeshPly(
    const std::filesystem::path& result_path)
{
    namespace fs = std::filesystem;
    if (!voxel_model_ || !scene_) {
        std::cout << "[mesh/rendered-TSDF-fixed] skipped: mapper is not initialized.\n";
        return;
    }

    const auto& keyframes = scene_->keyframes();
    if (keyframes.empty()) {
        std::cout << "[mesh/rendered-TSDF-fixed] skipped: no keyframes available.\n";
        return;
    }
    if (!result_path.parent_path().empty()) {
        fs::create_directories(result_path.parent_path());
    }

    auto surface = buildRenderedTsdfSurfaceData(
        voxel_model_,
        keyframes,
        undistort_mask_,
        rerun_params_,
        mutex_render_,
        sensor_type_ == MONOCULAR
            ? sv::kMonocularRenderedMeshDepthMaxM
            : sv::kRgbdRenderedMeshDepthMaxM,
        std::nullopt);
    auto keep_idx =
        torch::nonzero(surface.surface_cell_mask).view({-1}).to(torch::kLong);
    auto surface_voxel_keys =
        surface.voxel_keys.index_select(0, keep_idx).contiguous();
    auto [vertices, faces] = marchingCubesGrid(
        -surface.fused_tsdf,
        surface.grid_xyz,
        surface_voxel_keys,
        /*iso=*/0.0f);
    auto mesh = meshFromTensors(vertices, faces);
    if (mesh.vertices.empty() || mesh.faces.empty()) {
        throw std::runtime_error(
            "saveRenderedTsdfMeshPly: zero-crossing TSDF produced an empty mesh");
    }
    if (rerun_params_.svrecon_mesh_use_vert_color_) {
        std::unique_lock<std::mutex> render_lock(mutex_render_);
        setMeshColors(mesh, colorizeSvreconMeshVertices(vertices));
    }
    if (!saveTriangleMeshPly(
            result_path,
            mesh,
            rerun_params_.svrecon_mesh_use_vert_color_)) {
        throw std::runtime_error(
            "saveRenderedTsdfMeshPly: failed to write mesh PLY");
    }
    std::cout << "[mesh/rendered-TSDF-fixed] wrote " << result_path
              << " depth_source="
              << (rerun_params_.svrecon_mesh_use_mean_depth_
                      ? "mean_opacity"
                      : "median_opacity")
              << " views=" << surface.rendered_view_count
              << " requested_metric_voxel_size="
              << surface.requested_metric_voxel_size
              << " model_to_metric_scale=" << surface.model_to_metric_scale
              << " map_voxel_size=" << surface.voxel_size
              << " metric_voxel_size=" << surface.metric_voxel_size
              << " scale_aware=" << (surface.scale_aware ? 1 : 0)
              << " coverage_adapted=" << (surface.coverage_adapted ? 1 : 0)
              << " truncation=" << surface.truncation
              << " min_keyframe_weight=" << surface.min_keyframe_weight
              << " alpha_threshold=" << surface.alpha_threshold
              << " candidate_grid_points=" << surface.grid_xyz.size(0)
              << " surface_grid_voxels=" << surface_voxel_keys.size(0)
              << " vertices=" << mesh.vertices.size()
              << " faces=" << mesh.faces.size() << "\n";
}

torch::Tensor VoxelMapper::computeRenderedTsdfSurfacePruneMask(
    FinalSurfacePruneStats* stats_out)
{
    FinalSurfacePruneStats stats;
    const int64_t voxel_count = voxel_model_ ? voxel_model_->numVoxels() : 0;
    torch::Device device = mDevice;
    if (voxel_model_ && voxel_model_->voxCenter().defined()) {
        device = voxel_model_->voxCenter().device();
    }
    const auto bool_options =
        torch::TensorOptions().dtype(torch::kBool).device(device);
    const auto prune_none =
        torch::zeros({std::max<int64_t>(0, voxel_count)}, bool_options);
    auto finish = [&](const torch::Tensor& mask) {
        if (mask.defined()) {
            stats.surface_prune_count = mask.sum().item<int64_t>();
        }
        if (stats_out) {
            *stats_out = stats;
        }
        return mask;
    };

    if (!voxel_model_ || !scene_ || voxel_count <= 0 ||
        scene_->keyframes().empty()) {
        return finish(prune_none);
    }

    try {
        auto surface = buildRenderedTsdfSurfaceData(
            voxel_model_,
            scene_->keyframes(),
            undistort_mask_,
            rerun_params_,
            mutex_render_,
            sensor_type_ == MONOCULAR
                ? sv::kMonocularRenderedMeshDepthMaxM
                : sv::kRgbdRenderedMeshDepthMaxM);
        stats.rendered_view_count = surface.rendered_view_count;
        stats.candidate_grid_point_count = surface.grid_xyz.size(0);
        stats.candidate_grid_cell_count = surface.voxel_keys.size(0);
        stats.observed_grid_cell_count =
            surface.observed_cell_mask.sum().item<int64_t>();
        stats.surface_grid_cell_count =
            surface.surface_cell_mask.sum().item<int64_t>();
        stats.voxel_size = surface.voxel_size;
        stats.truncation = surface.truncation;
        stats.min_keyframe_weight = surface.min_keyframe_weight;
        stats.alpha_threshold = surface.alpha_threshold;

        if (stats.rendered_view_count <= 0 ||
            stats.surface_grid_cell_count <= 0) {
            return finish(prune_none);
        }

        auto surface_cell_rows =
            torch::nonzero(surface.surface_cell_mask)
                .view({-1})
                .to(torch::kLong)
                .to(torch::kCPU);
        auto surface_cells_cpu =
            surface.cell_indices
                .index_select(0, surface_cell_rows)
                .to(torch::kCPU)
                .to(torch::kInt32)
                .contiguous();

        // A fixed TSDF surface cell supports every adaptive leaf whose actual
        // world-space AABB intersects that cell. This is geometric overlap,
        // not a neighboring-cell expansion or halo.
        std::vector<SparseTsdfKey> surface_cells;
        surface_cells.reserve(
            static_cast<std::size_t>(surface_cells_cpu.size(0)));
        std::unordered_set<SparseTsdfKey, SparseTsdfKeyHash> surface_cell_set;
        surface_cell_set.reserve(
            static_cast<std::size_t>(surface_cells_cpu.size(0) * 2));
        auto fixed_cells = surface_cells_cpu.accessor<int32_t, 2>();
        for (int64_t i = 0; i < surface_cells_cpu.size(0); ++i) {
            const SparseTsdfKey key{
                fixed_cells[i][0],
                fixed_cells[i][1],
                fixed_cells[i][2]};
            surface_cells.push_back(key);
            surface_cell_set.insert(key);
        }

        auto centers_cpu =
            voxel_model_->voxCenter()
                .detach()
                .to(torch::kCPU)
                .to(torch::kFloat32)
                .contiguous();
        auto sizes_cpu =
            voxel_model_->voxSize()
                .detach()
                .to(torch::kCPU)
                .to(torch::kFloat32)
                .reshape({voxel_count})
                .contiguous();
        auto leaf_cpu =
            voxel_model_->isLeaf()
                .detach()
                .to(torch::kCPU)
                .to(torch::kBool)
                .reshape({voxel_count})
                .contiguous();
        auto supported_cpu = torch::zeros(
            {voxel_count},
            torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
        auto centers = centers_cpu.accessor<float, 2>();
        auto sizes = sizes_cpu.accessor<float, 1>();
        auto leaves = leaf_cpu.accessor<bool, 1>();
        auto supported = supported_cpu.accessor<bool, 1>();
        const double inv_voxel_size =
            1.0 / static_cast<double>(surface.voxel_size);
        constexpr double kBoundaryEpsilon = 1.0e-5;

        for (int64_t i = 0; i < voxel_count; ++i) {
            if (!leaves[i] || !std::isfinite(sizes[i]) || sizes[i] <= 0.0f ||
                !std::isfinite(centers[i][0]) ||
                !std::isfinite(centers[i][1]) ||
                !std::isfinite(centers[i][2])) {
                continue;
            }

            const double half_size = 0.5 * static_cast<double>(sizes[i]);
            int lower[3];
            int upper[3];
            for (int axis = 0; axis < 3; ++axis) {
                const double active_min =
                    static_cast<double>(centers[i][axis]) - half_size;
                const double active_max =
                    static_cast<double>(centers[i][axis]) + half_size;
                // Fixed cell k spans [(k + 0.5)v, (k + 1.5)v].
                lower[axis] = static_cast<int>(std::ceil(
                    active_min * inv_voxel_size - 1.5 + kBoundaryEpsilon));
                upper[axis] = static_cast<int>(std::floor(
                    active_max * inv_voxel_size - 0.5 - kBoundaryEpsilon));
            }

            const int64_t count_x =
                static_cast<int64_t>(upper[0]) - lower[0] + 1;
            const int64_t count_y =
                static_cast<int64_t>(upper[1]) - lower[1] + 1;
            const int64_t count_z =
                static_cast<int64_t>(upper[2]) - lower[2] + 1;
            const long double lattice_query_count =
                static_cast<long double>(std::max<int64_t>(0, count_x)) *
                static_cast<long double>(std::max<int64_t>(0, count_y)) *
                static_cast<long double>(std::max<int64_t>(0, count_z));

            bool intersects_surface = false;
            if (lattice_query_count <=
                static_cast<long double>(surface_cells.size())) {
                for (int x = lower[0];
                     x <= upper[0] && !intersects_surface;
                     ++x) {
                    for (int y = lower[1];
                         y <= upper[1] && !intersects_surface;
                         ++y) {
                        for (int z = lower[2]; z <= upper[2]; ++z) {
                            if (surface_cell_set.find({x, y, z}) !=
                                surface_cell_set.end()) {
                                intersects_surface = true;
                                break;
                            }
                        }
                    }
                }
            } else {
                for (const auto& cell : surface_cells) {
                    if (cell.x >= lower[0] && cell.x <= upper[0] &&
                        cell.y >= lower[1] && cell.y <= upper[1] &&
                        cell.z >= lower[2] && cell.z <= upper[2]) {
                        intersects_surface = true;
                        break;
                    }
                }
            }
            supported[i] = intersects_surface;
        }

        auto supported_device =
            supported_cpu.to(device).to(torch::kBool).contiguous();
        auto leaf_device =
            leaf_cpu.to(device).to(torch::kBool).contiguous();
        stats.supported_voxel_count =
            (supported_device & leaf_device).sum().item<int64_t>();
        stats.rendered_tsdf_available = true;
        return finish(
            (leaf_device & (~supported_device)).to(torch::kBool).contiguous());
    } catch (const std::exception& e) {
        std::cerr
            << "[FINAL/refinement] rendered-TSDF support unavailable; "
               "keeping the existing final map: "
            << e.what() << "\n";
        return finish(prune_none);
    }
}

void VoxelMapper::saveSvreconRenderedTsdfMeshPly(
    const std::filesystem::path& result_path)
{
    namespace fs = std::filesystem;
    torch::NoGradGuard no_grad;
    if (!voxel_model_ || !scene_) {
        std::cout << "[SVRecon mesh/rendered-TSDF] skipped: mapper is not initialized.\n";
        return;
    }
    if (!result_path.parent_path().empty()) {
        fs::create_directories(result_path.parent_path());
    }

    std::unique_lock<std::mutex> render_lock(mutex_render_);
    const auto mesh_device = voxel_model_->geoGridPts().device();
    std::vector<RenderedMeshView> views;
    const auto keyframes = scene_->getAllKeyframes();
    views.reserve(keyframes.size());
    for (const auto& [kf_id, pkf] : keyframes) {
        if (!pkf || !pkf->set_pose_) continue;
        const int height = std::max(1, pkf->image_height_);
        const int width = std::max(1, pkf->image_width_);
        const auto cam = pkf->toMiniCam(height, width);

        sv::RenderOpts render_opts;
        render_opts.output_depth = true;
        render_opts.output_T = true;
        auto render_pkg = voxel_model_->render(
            cam,
            height,
            width,
            torch::Tensor(),
            nullptr,
            false,
            std::nullopt,
            true,
            false,
            true,
            false,
            false,
            render_opts);
        auto depth_it = render_pkg.find("raw_depth");
        auto transmittance_it = render_pkg.find("raw_T");
        if (depth_it == render_pkg.end() || transmittance_it == render_pkg.end()) {
            continue;
        }

        auto raw_depth = depth_it->second.detach().to(mesh_device).to(torch::kFloat32);
        if (raw_depth.dim() == 4 && raw_depth.size(0) == 1) {
            raw_depth = raw_depth.squeeze(0);
        }
        torch::Tensor frame_depth;
        if (raw_depth.dim() >= 3) {
            const int64_t depth_channel = rerun_params_.svrecon_mesh_use_mean_depth_
                ? 0
                : std::min<int64_t>(2, raw_depth.size(0) - 1);
            frame_depth = raw_depth.index({depth_channel}).contiguous();
        } else {
            frame_depth = raw_depth.contiguous();
        }
        auto frame_alpha =
            (1.0f - transmittance_it->second.detach()
                        .to(mesh_device)
                        .to(torch::kFloat32))
                .contiguous();
        RenderedMeshView view;
        view.cam = cam;
        view.depth = frame_depth;
        view.alpha = frame_alpha;
        views.push_back(std::move(view));
    }
    if (views.empty()) {
        std::cout << "[SVRecon mesh/rendered-TSDF] skipped: no posed keyframes.\n";
        return;
    }

    const int max_inside_level = std::max(
        1, voxel_model_->maxNumLevels() - voxel_model_->outsideLevel());
    const int final_inside_level = std::clamp(
        sv::kSvreconMeshFinalLevel, 1, max_inside_level);
    const int init_inside_level = std::clamp(
        sv::kSvreconMeshInitialLevel, 1, final_inside_level);
    auto [octpath, octlevel, grid_xyz, vox_key] =
        voxel_model_->buildSvreconDenseExtractionGrid(init_inside_level);
    grid_xyz = grid_xyz.detach().to(torch::kFloat32).contiguous();
    vox_key = vox_key.detach().to(torch::kLong).contiguous();

    const float scene_extent =
        voxel_model_->SceneExtent().detach().reshape({-1})[0].item<float>();
    torch::Tensor grid_tsdf;
    float trunc_dist = 0.0f;
    std::cout << "[SVRecon mesh/rendered-TSDF] progressive levels "
              << init_inside_level << ".." << final_inside_level
              << " views=" << views.size() << "\n";
    for (int inside_level = init_inside_level;
         inside_level <= final_inside_level;
         ++inside_level) {
        const int trunc_inside_level = std::min(
            inside_level, sv::kSvreconMeshTruncationLevel);
        const int trunc_level = std::min(
            voxel_model_->maxNumLevels(),
            voxel_model_->outsideLevel() + trunc_inside_level);
        trunc_dist = sv::kSvreconMeshTruncationVox *
            std::ldexp(scene_extent, -trunc_level);
        std::cout << "[SVRecon mesh/rendered-TSDF] level=" << inside_level
                  << " voxels=" << vox_key.size(0)
                  << " truncation=" << trunc_dist << "\n";
        grid_tsdf = fuseRenderedTsdf(
            grid_xyz,
            views,
            trunc_dist,
            sv::kSvreconMeshCropBorder,
            sv::kSvreconMeshAlphaThreshold);

        if (inside_level >= final_inside_level) break;
        auto vox_tsdf = grid_tsdf.index({vox_key}).reshape({vox_key.size(0), 8});
        auto prune_mask =
            torch::isnan(vox_tsdf).any(1) |
            (std::get<0>(vox_tsdf.max(1)) < -sv::kSvreconMeshProgressivePrune) |
            (std::get<0>(vox_tsdf.min(1)) > sv::kSvreconMeshProgressivePrune);
        auto keep_idx = torch::nonzero(~prune_mask).view({-1}).to(torch::kLong);
        if (keep_idx.numel() == 0) {
            throw std::runtime_error(
                "saveSvreconRenderedTsdfMeshPly: progressive TSDF pruning removed every cell");
        }
        octpath = octpath.index_select(0, keep_idx).contiguous();
        octlevel = octlevel.index_select(0, keep_idx).contiguous();
        std::tie(octpath, octlevel, grid_xyz, vox_key) =
            voxel_model_->subdivideSvreconExtractionGrid(octpath, octlevel);
        grid_xyz = grid_xyz.detach().to(torch::kFloat32).contiguous();
        vox_key = vox_key.detach().to(torch::kLong).contiguous();
    }

    auto [vertices, faces] = marchingCubesGrid(
        grid_tsdf,
        grid_xyz,
        vox_key,
        /*iso=*/0.0f);
    auto mesh = meshFromTensors(vertices, faces);
    if (rerun_params_.svrecon_mesh_use_vert_color_ && vertices.numel() > 0) {
        setMeshColors(mesh, colorizeSvreconMeshVertices(vertices));
    }
    if (!saveTriangleMeshPly(
            result_path,
            mesh,
            rerun_params_.svrecon_mesh_use_vert_color_)) {
        throw std::runtime_error(
            "saveSvreconRenderedTsdfMeshPly: rendered-depth TSDF extraction produced an empty mesh");
    }
    std::cout << "[SVRecon mesh/rendered-TSDF] wrote " << result_path
              << " integrated_views=" << views.size()
              << " voxels=" << vox_key.size(0)
              << " truncation=" << trunc_dist
              << " vertices=" << mesh.vertices.size()
              << " faces=" << mesh.faces.size() << "\n";
}
