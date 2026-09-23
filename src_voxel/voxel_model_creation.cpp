#include "src_voxel/voxel_model_internal.h"

namespace sv {

namespace fs = std::filesystem;

void VoxelModel::createFromPcd(
    const std::map<point3D_id_t, Point3D>& pcd,
    const std::vector<sv::MiniCam>& cams)
{
    // Builds the initial sparse octree, shared SDF corners, and SH colors from
    // SLAM points. SVRecon references: SVConstructor.model_init and
    // sdf_init_utils.initialize_sdf_from_sfm; sparse online allocation is local.
    sdf_init_cams_ = cams;

    TORCH_CHECK(global_scene_extent_ > 0.f, "global_scene_extent_ must be set (>0).");
    TORCH_CHECK(fixed_vox_size_   > 0.f, "fixed_vox_size_ must be set (>0).");
    TORCH_CHECK(max_sh_degree_ >= 0, "max_sh_degree_ must be >= 0.");

    const int N = static_cast<int>(pcd.size());
    const int32_t current_kf_count =
        cams.empty() ? static_cast<int32_t>(-1) : static_cast<int32_t>(cams.size());
    if (N == 0) {
        std::cerr << "[createFromPcd] Empty PCD — nothing to do.\n";
        return;
    }

    auto dev = torch::kCUDA;
    outside_level_ = std::clamp(outside_level_, 0, max_num_levels_);

    // Pack inputs
    torch::Tensor xyz = torch::empty({N,3}, torch::dtype(torch::kFloat32).device(dev));
    torch::Tensor rgb = torch::empty({N,3}, torch::dtype(torch::kFloat32).device(dev));
    {
        int i = 0;
        for (const auto& kv : pcd) {
            const auto& P = kv.second;
            xyz[i][0] = P.xyz_(0);
            xyz[i][1] = P.xyz_(1);
            xyz[i][2] = P.xyz_(2);
            // P.color_ already scaled? If it’s 0..255, divide; if already 0..1, keep.
            // std::cout << "Point " << i << " color before scaling: "
            //           << P.color_(0) << ", " << P.color_(1) << ", " << P.color_(2) << "\n";
            rgb[i][0] = P.color_(0);
            rgb[i][1] = P.color_(1);
            rgb[i][2] = P.color_(2);
            ++i;
        }
    }
    // Accept both [0,1] and [0,255] color inputs (same policy as increasePcd).
    if (rgb.numel() > 0) {
        const float cmax = rgb.max().item<float>();
        if (cmax > 1.5f) {
            rgb.div_(255.0f);
        }
    }
    rgb.clamp_(0.0f, 1.0f);
    if (topology_sdf_init_mode_ == SdfInitMode::WeakSurfacePrior) {
        sparse_points_xyz_ = torch::empty(
            {0, 3},
            torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(dev));
        sdf_init_local_support_points_ =
            xyz.detach().to(torch::kFloat32).contiguous();
    } else {
        sparse_points_xyz_ =
            xyz.detach().to(torch::kFloat32).contiguous();
        sdf_init_local_support_points_ = torch::Tensor();
    }
    sparse_points_color_ = rgb.detach().to(torch::kFloat32).contiguous();
    pending_sdf_init_mode_ = topology_sdf_init_mode_;

    // Reset accumulated real-point history. We seed it later from actually
    // inserted/filtered real voxels (not raw pre-filter PCD).
    real_pcd_points_accum_cpu_ = torch::empty(
        {0, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    // ------------------------------------------------------------------------
    // 0.5) Initialize global PCD bounds.
    // ------------------------------------------------------------------------
    {
        // Work on CPU for simplicity; this runs only once at initialization.
        auto xyz_cpu = xyz.to(torch::kCPU).contiguous();  // [N,3]

        auto min_res = xyz_cpu.min(/*dim=*/0, /*keepdim=*/false);
        auto max_res = xyz_cpu.max(/*dim=*/0, /*keepdim=*/false);

        torch::Tensor min_cpu = std::get<0>(min_res).contiguous();   // [3]
        torch::Tensor max_cpu = std::get<0>(max_res).contiguous();   // [3]

        torch::Tensor center_cpu;
        float radius = fixed_vox_size_;
        if (robust_scene_bounds_) {
            center_cpu =
                std::get<0>(xyz_cpu.median(/*dim=*/0, /*keepdim=*/false)).contiguous();
            torch::Tensor distances = std::get<0>(
                (xyz_cpu - center_cpu.view({1, 3}))
                    .abs()
                    .max(/*dim=*/1, /*keepdim=*/false))
                    .to(torch::kFloat32)
                    .contiguous();
            distances = std::get<0>(distances.sort(/*dim=*/0)).contiguous();
            const int64_t count = distances.size(0);
            if (count > 0) {
                const int64_t p90_index = std::clamp<int64_t>(
                    static_cast<int64_t>(std::llround(
                        0.90 * static_cast<double>(count - 1))),
                    0,
                    count - 1);
                radius = std::max(
                    radius,
                    distances.index({p90_index}).item<float>());
            }
        } else {
            center_cpu = ((min_cpu + max_cpu) * 0.5f).contiguous();
            torch::Tensor radius_cpu = ((max_cpu - min_cpu) * 0.5f).contiguous();
            radius = std::max(radius, radius_cpu.max().item<float>());
        }
        radius += fixed_vox_size_;
        const float derived_scene_extent = 2.0f * radius;
        global_scene_center_[0] = center_cpu[0].item<float>();
        global_scene_center_[1] = center_cpu[1].item<float>();
        global_scene_center_[2] = center_cpu[2].item<float>();
        // In SVRecon's bounded/object-centric mode outside_level=0. For online SLAM
        // we still need a stable bound; otherwise each newly observed region can fall
        // outside the first tight point-prior AABB and trigger repeated rebuilds.
        if (fixed_global_scene_layout_) {
            const float configured_inside_extent =
                configured_global_scene_extent_ /
                std::ldexp(1.0f, outside_level_);
            TORCH_CHECK(
                configured_inside_extent >= derived_scene_extent,
                "Model.global_scene_extent root (", configured_global_scene_extent_,
                " m) provides only ", configured_inside_extent,
                " m of inside extent at outside_level=", outside_level_,
                ", smaller than the initial map extent (", derived_scene_extent,
                " m). Increase the root extent or reduce outside_level.");
            global_scene_extent_ = configured_global_scene_extent_;
        } else {
            global_scene_extent_ = (outside_level_ == 0)
                ? std::max(global_scene_extent_, derived_scene_extent)
                : derived_scene_extent;
        }

        scene_center_ = center_cpu.to(dev).to(torch::kFloat32).contiguous();
        const float inside_extent = fixed_global_scene_layout_
            ? global_scene_extent_ / std::ldexp(1.0f, outside_level_)
            : global_scene_extent_;
        inside_extent_ = torch::tensor({inside_extent},
            torch::dtype(torch::kFloat32).device(dev)
        ).contiguous();
        scene_extent_ = torch::tensor(
            {fixed_global_scene_layout_
                 ? global_scene_extent_
                 : global_scene_extent_ * std::ldexp(1.0f, outside_level_)},
            torch::dtype(torch::kFloat32).device(dev)
        ).contiguous();
        scene_min_t_ = (scene_center_ - 0.5f * scene_extent_).contiguous();

        // Store as CUDA tensors so they match the rest of the model state
        global_pcd_min_ = min_cpu.to(dev).contiguous();              // [3]
        global_pcd_max_ = max_cpu.to(dev).contiguous();              // [3]
        has_global_pcd_bb_ = true;

    }
    // ------------------------------------------------------------------------
    // 1) Compute octlevel from vox_size (mirror points_init behavior:
    //    round/clamp).
    // ------------------------------------------------------------------------
    const int MAX_L = max_num_levels_;
    // Level as float (tensor) then rounded like points_init (nearest by default)
    auto vox_size_t = torch::full({1}, fixed_vox_size_, torch::dtype(torch::kFloat32).device(dev)); // [1]
    auto L_fp = voxSizeToLevel(scene_extent_, vox_size_t).round();                                   // [1] float
    auto L_clamped = L_fp.clamp_min(1).clamp_max(MAX_L).to(torch::kInt8).contiguous();               // [1] int8

    // cache scalar level
    octlevel_ = L_clamped.item<int8_t>();
    // cache [1,1] effective voxel size
    vox_eff_ = levelToVoxSize(scene_extent_, L_clamped.view({1,1})).view({1,1}).contiguous();
    // std::cout << "[createFromPcd] Using octlevel=" << octlevel[0][0].item<int>()
    //           << " (vox_size=" << vox_eff[0][0].item<float>() << " m) for fixed_vox_size_="
    //           << fixed_vox_size_ << " m.\n";

    // ------------------------------------------------------------------------
    // 2) Compute ijk with this level/voxel size (mirror points_init)
    // ------------------------------------------------------------------------
    // ijk = ((xyz - scene_min) / vox_size).long()
    auto vox_effN  = vox_eff_.expand({N,1});                                                         // [N,1]
    torch::Tensor ijk = ((xyz - scene_min_t_) / vox_effN).floor().to(torch::kLong);                  // [N,3]

    auto octlevelN = torch::full({N,1}, octlevel_, torch::dtype(torch::kInt8).device(dev)).contiguous(); // [N,1]
    auto L_long    = octlevelN.to(torch::kLong);

    // In libtorch C++, pass a vector to cat
    std::vector<torch::Tensor> cat_inputs{ijk, L_long};
    auto ijkl = torch::cat(cat_inputs, /*dim=*/1);

    // Use Python torch.unique(dim=0, return_inverse=True) (simple & robust)
    auto [ijkl_unq, invmap] = uniqueRowsWithInverse(ijkl);

    torch::Tensor ijk_u, L_u;
    auto parts = torch::split_with_sizes(ijkl_unq, {3, 1}, /*dim=*/1);
    ijk_u = parts[0].contiguous(); L_u = parts[1].to(torch::kInt8).contiguous();
    L_u = L_u.to(torch::kInt8).contiguous(); // [Nu,1]
    int64_t Nu = ijk_u.size(0);

    auto rgb_u = torch::zeros({Nu,3}, torch::dtype(torch::kFloat32).device(dev));
    rgb_u.index_add_(0, invmap, rgb);
    auto counts = torch::bincount(invmap, {}, Nu).view({Nu,1}).clamp_min(1);
    rgb_u = rgb_u / counts;

    // Defensive bound check: ijk in [0, 2^L)
    const int L0 = L_u[0].item<int8_t>();               // all rows have same level
    const long limit = (1L << L0);
    TORCH_CHECK((ijk_u.index({torch::indexing::Slice(),0}) >= 0).all().item<bool>() &&
                (ijk_u.index({torch::indexing::Slice(),1}) >= 0).all().item<bool>() &&
                (ijk_u.index({torch::indexing::Slice(),2}) >= 0).all().item<bool>(),
                "Points below scene_min — enlarge global_scene_extent_ or filter.");
    TORCH_CHECK((ijk_u.index({torch::indexing::Slice(),0}) < limit).all().item<bool>() &&
                (ijk_u.index({torch::indexing::Slice(),1}) < limit).all().item<bool>() &&
                (ijk_u.index({torch::indexing::Slice(),2}) < limit).all().item<bool>(),
                "Points exceed scene bounds — enlarge global_scene_extent_ or filter.");

    // ------------------------------------------------------------------------
    // 3) utils: ijk -> octpath (no constructor calls)
    // ------------------------------------------------------------------------
    auto octpath = SVRECON_UTILS::ijk_2_octpath(ijk_u.contiguous(), L_u.contiguous()).contiguous(); // [Nu,1] int64
    // ------------------------------------------------------------------------
    // 3.5) Optional camera-based filtering for initialization candidates
    //      (same logic family as increasePcd insertion filter)
    // ------------------------------------------------------------------------
    if (!cams.empty() && octpath.size(0) > 0) {
        auto [vox_center, vox_size] = decodeOctpath(
            octpath.contiguous(),
            L_u.contiguous(),
            scene_center_.contiguous(),
            scene_extent_.contiguous());
        if (vox_size.dim() == 1) {
            vox_size = vox_size.view({-1, 1});
        }

        const int64_t Nu_before = octpath.size(0);
        at::Tensor rate = markSvreconMaxSampRateDirect(cams, octpath, vox_center, vox_size);
        if (rate.dim() == 2 && rate.size(1) == 1) {
            rate = rate.squeeze(1);
        }
        rate = rate.to(torch::kFloat32);
        at::Tensor kept = (rate > 0.0f);

        if (filter_near_voxels_) {
            const float near_thresh = 0.2f;
            at::Tensor is_near = markSvreconNearDirect(cams, octpath, vox_center, vox_size, near_thresh);
            if (is_near.dim() == 2 && is_near.size(1) == 1) {
                is_near = is_near.squeeze(1);
            }
            is_near = is_near.to(torch::kBool);
            kept = kept & (~is_near);
        }
        kept = kept.view({-1}).to(torch::kBool);
        auto idx = torch::nonzero(kept).view({-1});
        const int64_t K = idx.size(0);

        TORCH_CHECK(
            K > 0,
            "createFromPcd: no voxel-layout candidate is visible from the construction cameras");
        if (K < octpath.size(0)) {
            octpath = octpath.index_select(0, idx).contiguous();
            L_u     = L_u.index_select(0, idx).contiguous();
            ijk_u   = ijk_u.index_select(0, idx).contiguous();
            rgb_u   = rgb_u.index_select(0, idx).contiguous();
            Nu = octpath.size(0);
        }
    }
    // ------------------------------------------------------------------------
    // 4) Initialize learnables directly
    // ------------------------------------------------------------------------
    // Subdivision priority
    auto subdiv_p = torch::ones({Nu,1}, torch::dtype(torch::kFloat32).device(dev)).requires_grad_();

    // SH-0 from fused RGB
    auto sh0_dc = rgbToShZero(rgb_u.contiguous()).contiguous().requires_grad_(); // [Nu,3]

    // Higher-degree SH zeros
    const int n_sh_rest = (max_sh_degree_ + 1)*(max_sh_degree_ + 1) - 1;
    auto shs = torch::zeros({Nu, n_sh_rest, 3}, torch::dtype(torch::kFloat32).device(dev)).requires_grad_();

    // Grid link => allocate per-grid-point density
    auto [grid_pts_key, vox_key] = buildGridPtsLink(octpath, L_u.contiguous(), max_num_levels_);
    (void)vox_key;
    auto geo_grid = makeGeoGridInitRows_(
        grid_pts_key,
        /*begin=*/0,
        /*end=*/grid_pts_key.size(0),
        /*default_value=*/std::max(1.0e-6f, 2.0f * fixed_vox_size_));
    sdf_init_local_support_points_ = torch::Tensor();

    // ------------------------------------------------------------------------
    // 5) Initialize C++ members directly.
    // ------------------------------------------------------------------------
    this->oct_path_      = octpath.contiguous();
    this->oct_level_     = L_u.contiguous();
    {
        auto [center, size] = decodeOctpath(
            this->oct_path_.contiguous(),
            this->oct_level_.contiguous(),
            this->scene_center_.contiguous(),
            this->scene_extent_.contiguous());
        auto [grid_pts_key_now, vox_key_now] =
            buildGridPtsLink(this->oct_path_.contiguous(), this->oct_level_.contiguous(), max_num_levels_);
        this->center_        = center.contiguous();
        this->size_          = size.squeeze(1).contiguous();
        this->grid_pts_key_  = grid_pts_key_now.contiguous();
        this->vox_key_       = vox_key_now.contiguous();
    }
    this->vox_size_inv_  = 1.0f / size_;

    // learnables
    this->_geo_grid_pts_ = geo_grid.contiguous().detach().requires_grad_(true);
    ensureFusedSdfField();
    this->sh0_           = sh0_dc.contiguous().detach().requires_grad_(true);
    this->shs_           = shs.contiguous().detach().requires_grad_(true);
    this->subdiv_p_      = subdiv_p.contiguous().detach().requires_grad_(true);
    this->subdiv_meta_   = torch::zeros_like(this->subdiv_p_);
    this->is_leaf_       = torch::ones(
        {center_.size(0), 1},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    // stats buffer
    this->max_w_ = torch::zeros({center_.size(0), 1}, torch::dtype(torch::kFloat32).device(dev));

    this->is_orb_voxel_ = torch::ones(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    this->is_inactive_geo_voxel_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    this->is_rgbd_fill_render_holes_voxel_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    this->is_monocular_rendered_depth_voxel_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    this->is_monocular_mvs_voxel_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    this->exist_since_iter_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kInt32).device(dev));
    this->exist_since_kf_ = torch::full(
        {center_.size(0)},
        current_kf_count,
        torch::TensorOptions().dtype(torch::kInt32).device(dev));

    // Dense-core pruning must be estimated from cells that survived layout
    // filtering, not from raw points that were rejected as invisible or near.
    real_pcd_points_accum_cpu_ =
        center_.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();

    // Match SVRecon initialization while adapting sharpness to the actual
    // online octree resolution rather than a fixed offline scene scale.
    refreshSvreconLogSTargetFromVoxelSize(/*initialize_current=*/true);

    // Register with your optimizer
    VOXEL_MODEL_TENSORS_TO_VEC
}

} // namespace sv
