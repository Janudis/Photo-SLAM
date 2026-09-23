#include "src_voxel/voxel_model_internal.h"

namespace sv {

namespace fs = std::filesystem;

void VoxelModel::increasePcd(
    std::vector<float> pcd_full,
    std::vector<float> colors,
    const int iteration,
    const std::vector<sv::MiniCam>& cams,
    const bool clear_cuda_cache_before_parameter_append)
{
    // Incrementally inserts previously unseen source cells and remaps all
    // topology-aligned parameters and optimizer state. Offline SVRecon has no
    // direct equivalent because its topology is initialized before training.
    sdf_init_cams_ = cams;
    struct PendingGridInitializationReset {
        torch::Tensor& keys;
        torch::Tensor& values;
        ~PendingGridInitializationReset()
        {
            keys = torch::Tensor();
            values = torch::Tensor();
        }
    } pending_grid_initialization_reset{
        next_sdf_init_grid_keys_, next_sdf_init_grid_values_};
    const int Nf = static_cast<int>(pcd_full.size());
    last_increase_pcd_stats_ = IncreasePcdStats{};
    if (Nf < 3 || colors.size() < 3) return;
    int N = Nf / 3;
    const int64_t raw_points_in = N;
    last_increase_pcd_stats_.raw_points_in = raw_points_in;
    const int32_t current_kf_count =
        cams.empty() ? static_cast<int32_t>(-1) : static_cast<int32_t>(cams.size());
    TORCH_CHECK(global_scene_extent_ > 0.f && fixed_vox_size_ > 0.f,
                "increasePcd: scene extent / fixed vox size not set.");
    TORCH_CHECK(oct_path_.defined() && oct_level_.defined() && center_.defined(),
                "increasePcd: voxel model not initialized; call createFromPcd first.");

    // ——— 0) Build CPU tensors from raw arrays, normalize RGB ————————
    torch::Tensor xyz_cpu = torch::from_blob(
        pcd_full.data(), { (int64_t)N, 3 },
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
    ).clone();

    torch::Tensor rgb_cpu = torch::from_blob(
        colors.data(), { (int64_t)N, 3 },
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
    ).clone();
    // Accept both [0,1] and [0,255] color inputs.
    if (rgb_cpu.numel() > 0) {
        const float cmax = rgb_cpu.max().item<float>();
        if (cmax > 1.5f) rgb_cpu.div_(255.0f);
    }
    rgb_cpu.clamp_(0.0f, 1.0f);

    N = static_cast<int>(xyz_cpu.size(0));
    const int64_t points_after_far_filter = N;
    last_increase_pcd_stats_.points_after_far_filter = points_after_far_filter;

    // Move to CUDA
    auto dev = torch::kCUDA;
    torch::Tensor xyz = xyz_cpu.to(dev);
    torch::Tensor rgb = rgb_cpu.to(dev);
    if (fixed_global_scene_layout_) {
        const torch::Tensor scene_max = scene_min_t_ + scene_extent_.view({1});
        torch::Tensor in_bounds =
            torch::isfinite(xyz).all(/*dim=*/1) &
            (xyz >= scene_min_t_.view({1, 3})).all(/*dim=*/1) &
            (xyz < scene_max.view({1, 3})).all(/*dim=*/1);
        in_bounds = in_bounds.to(torch::kBool).contiguous();
        if (!in_bounds.all().item<bool>()) {
            auto keep = torch::nonzero(in_bounds).view({-1});
            if (keep.numel() == 0) {
                last_increase_pcd_stats_.points_after_far_filter = 0;
                sdf_init_local_support_points_ = torch::Tensor();
                pending_sdf_init_mode_ = SdfInitMode::SignedPointPrior;
                pending_real_insert_rr_entity_path_.clear();
                return;
            }
            xyz = xyz.index_select(0, keep).contiguous();
            rgb = rgb.index_select(0, keep).contiguous();
            xyz_cpu = xyz.detach().to(torch::kCPU).contiguous();
            rgb_cpu = rgb.detach().to(torch::kCPU).contiguous();
            N = static_cast<int>(xyz.size(0));
            last_increase_pcd_stats_.points_after_far_filter = N;
        }
    }
    const bool add_as_orb =
        pending_real_insert_rr_entity_path_ == "world/orb/voxels_created";
    const bool add_as_inactive_geo =
        pending_real_insert_rr_entity_path_ == "world/voxels_inactive_geo_densify/created";
    const bool add_as_rgbd_fill_render_holes =
        pending_real_insert_rr_entity_path_ == "world/rgbd_fill_render_holes/created" ||
        pending_real_insert_rr_entity_path_ == "world/rgbd_tsdf_evidence/promoted";
    const bool add_as_monocular_rendered_depth =
        pending_real_insert_rr_entity_path_ ==
            "world/monocular_rendered_depth/created" ||
        pending_real_insert_rr_entity_path_ ==
            "world/monocular_rendered_depth_evidence/promoted";
    const bool add_as_monocular_mvs =
        pending_real_insert_rr_entity_path_ ==
        "world/monocular_mvs/created";
    torch::Tensor orb_metric_support;
    if (add_as_orb &&
        topology_sdf_init_mode_ != SdfInitMode::WeakSurfacePrior) {
        orb_metric_support = xyz.detach().contiguous();
    }

    // Build the min/max AABB tensor once:
    at::Tensor aabb = torch::stack({
        torch::tensor({ global_scene_center_[0] - 0.5f*global_scene_extent_,
                        global_scene_center_[1] - 0.5f*global_scene_extent_,
                        global_scene_center_[2] - 0.5f*global_scene_extent_ }),
        torch::tensor({ global_scene_center_[0] + 0.5f*global_scene_extent_,
                        global_scene_center_[1] + 0.5f*global_scene_extent_,
                        global_scene_center_[2] + 0.5f*global_scene_extent_ })
    });

    // --- 2) Compute octlevel from fixed_vox_size_ (same as createFromPcd)
    const int MAX_L = max_num_levels_;

    auto vox_effN  = vox_eff_.expand({N,1});                                                         // [N,1]
    torch::Tensor ijk = ((xyz - scene_min_t_) / vox_effN).floor().to(torch::kLong);                  // [N,3]

    auto octlevelN = torch::full({N,1}, octlevel_, torch::dtype(torch::kInt8).device(dev)).contiguous(); // [N,1]
    auto L_long    = octlevelN.to(torch::kLong);

    // In libtorch C++, pass a vector to cat
    std::vector<torch::Tensor> cat_inputs{ijk, L_long};
    auto ijkl = torch::cat(cat_inputs, /*dim=*/1);                                                         // [N,4]

    auto [ijkl_unq, invmap] = uniqueRowsWithInverse(ijkl);
    torch::Tensor ijk_u, L_u;
    auto parts = torch::split_with_sizes(ijkl_unq, {3,1}, 1);
    ijk_u = parts[0].contiguous();                                                                    // [Nu,3]
    L_u   = parts[1].to(torch::kInt8).contiguous();                                                   // [Nu,1]
    int64_t Nu = ijk_u.size(0);
    const int64_t unique_voxel_candidates_before_insert_filter = Nu;

    // Defensive bound check: ijk in [0, 2^L)
    const int8_t L0 = L_u[0].item<int8_t>();      // all rows share the same level
    const long limit = (1L << L0);
    const bool in_low =
    (ijk_u.index({torch::indexing::Slice(),0}) >= 0).all().item<bool>() &&
    (ijk_u.index({torch::indexing::Slice(),1}) >= 0).all().item<bool>() &&
    (ijk_u.index({torch::indexing::Slice(),2}) >= 0).all().item<bool>();
    const bool in_high =
        (ijk_u.index({torch::indexing::Slice(),0}) < limit).all().item<bool>() &&
        (ijk_u.index({torch::indexing::Slice(),1}) < limit).all().item<bool>() &&
        (ijk_u.index({torch::indexing::Slice(),2}) < limit).all().item<bool>();
    if (!(in_low && in_high)) {
        std::cout << "[increasePcd] OOB detected — reinitializing via createFromPcd().\n";
        // A) Compute a conservative scene from this batch.
        {
            auto pts = xyz_cpu.to(torch::kCPU).to(torch::kFloat32).contiguous();
            auto center = std::get<0>(pts.median(/*dim=*/0, /*keepdim=*/false)).contiguous();
            auto dist = std::get<0>((pts - center.view({1, 3})).abs().max(/*dim=*/1, /*keepdim=*/false))
                .to(torch::kFloat32).contiguous();
            auto sorted = std::get<0>(dist.sort(/*dim=*/0, /*descending=*/false)).contiguous();
            const int64_t n = sorted.size(0);
            int64_t p90_idx = static_cast<int64_t>(std::llround(0.90 * static_cast<double>(std::max<int64_t>(0, n - 1))));
            p90_idx = std::max<int64_t>(0, std::min<int64_t>(p90_idx, std::max<int64_t>(0, n - 1)));
            float radius = (n > 0) ? sorted.index({p90_idx}).item<float>() : 1.0f;
            if (!std::isfinite(radius) || radius <= 0.0f) {
                radius = (n > 0) ? sorted.index({n - 1}).item<float>() : 1.0f;
            }
            if (!std::isfinite(radius) || radius <= 0.0f) {
                radius = 1.0f;
            }

            global_scene_center_[0] = center.index({0}).item<float>();
            global_scene_center_[1] = center.index({1}).item<float>();
            global_scene_center_[2] = center.index({2}).item<float>();
            global_scene_extent_ = 2.0f * radius;
        }
        // B) Convert current batch to a temporary map and call createFromPcd
        {
            std::map<point3D_id_t, Point3D> tmp;
            static point3D_id_t id_seed = 1;  // local seq; independent of COLMAP ids, etc.
            auto xyz_re = xyz_cpu.contiguous();
            auto rgb_re = rgb_cpu.contiguous();
            auto xyz_acc = xyz_re.accessor<float, 2>();
            auto rgb_acc = rgb_re.accessor<float, 2>();
            for (int i = 0; i < N; ++i) {
                Point3D P;
                // world coords
                P.xyz_(0) = xyz_acc[i][0];
                P.xyz_(1) = xyz_acc[i][1];
                P.xyz_(2) = xyz_acc[i][2];
                // rgb_cpu is normalized to [0,1]. Point3D stores uint8.
                P.color_(0) = static_cast<uint8_t>(std::round(std::clamp(rgb_acc[i][0], 0.0f, 1.0f) * 255.0f));
                P.color_(1) = static_cast<uint8_t>(std::round(std::clamp(rgb_acc[i][1], 0.0f, 1.0f) * 255.0f));
                P.color_(2) = static_cast<uint8_t>(std::round(std::clamp(rgb_acc[i][2], 0.0f, 1.0f) * 255.0f));
                tmp.emplace(id_seed++, P);
            }
            createFromPcd(tmp, cams);
        }
        // Log and exit this call
        sdf_init_local_support_points_ = torch::Tensor();
        pending_sdf_init_mode_ = SdfInitMode::SignedPointPrior;
        return;
    }

    auto rgb_u = torch::zeros({Nu,3}, torch::dtype(torch::kFloat32).device(dev));
    rgb_u.index_add_(0, invmap, rgb);
    auto counts = torch::bincount(invmap, {}, Nu).view({Nu,1}).clamp_min(1);
    rgb_u = rgb_u / counts;

    // ── 4) Build octpath for this batch ─────────────────────────────────────
    auto octpath_new = SVRECON_UTILS::ijk_2_octpath(ijk_u.contiguous(), L_u.contiguous()).contiguous();          // [Nu,1] int64

    // ---- Active insertion-time SVR-style filtering (kept separate from OLD block) ----
    // Keep this at insertion-time so bad candidates never enter the topology.
    // Prune-time filtering in VoxelMapper remains useful as a second cleanup stage.
    if (!cams.empty() && octpath_new.size(0) > 0) {
        // Decode voxel centers/sizes for current candidates.
        auto [vox_center, vox_size] = decodeOctpath(
            octpath_new.contiguous(),
            L_u.contiguous(),
            scene_center_.contiguous(),
            scene_extent_.contiguous());
        if (vox_size.dim() == 1) {
            vox_size = vox_size.view({-1, 1});
        }

        // 1) Visibility / sampling-rate filtering
        at::Tensor rate = markSvreconMaxSampRateDirect(cams, octpath_new, vox_center, vox_size);
        if (rate.dim() == 2 && rate.size(1) == 1) {
            rate = rate.squeeze(1);
        }
        rate = rate.to(torch::kFloat32);
        at::Tensor kept = (rate > 0.0f);

        // 2) Near filtering
        // Upstream layout initialization leaves near filtering disabled.
        if (filter_near_voxels_) {
            const float near_thresh = 0.2f;
            at::Tensor is_near =
                markSvreconNearDirect(cams, octpath_new, vox_center, vox_size, near_thresh);
            if (is_near.dim() == 2 && is_near.size(1) == 1) {
                is_near = is_near.squeeze(1);
            }
            is_near = is_near.to(torch::kBool);
            kept = kept & (~is_near);
        }

        kept = kept.view({-1}).to(torch::kBool);
        auto idx = torch::nonzero(kept).view({-1});
        int64_t K = idx.size(0);

        if (K < octpath_new.size(0)) {
            // Apply mask to ALL aligned tensors
            octpath_new = octpath_new.index_select(0, idx).contiguous(); // [K,1]
            L_u         = L_u.index_select(0, idx).contiguous();         // [K,1]
            ijk_u       = ijk_u.index_select(0, idx).contiguous();       // [K,3]
            rgb_u       = rgb_u.index_select(0, idx).contiguous();       // [K,3]
        }

        Nu = octpath_new.size(0); // update Nu after filtering

        TORCH_CHECK(L_u.sizes() == torch::IntArrayRef({Nu, 1}),
                    "L_u shape mismatch after insertion-time filtering");
        TORCH_CHECK(ijk_u.sizes() == torch::IntArrayRef({Nu, 3}),
                    "ijk_u shape mismatch after insertion-time filtering");
        TORCH_CHECK(rgb_u.sizes() == torch::IntArrayRef({Nu, 3}),
                    "rgb_u shape mismatch after insertion-time filtering");
	    }
	    // ---- end active insertion-time filtering ----
	    const int64_t unique_voxel_candidates_after_insert_filter = Nu;
    last_increase_pcd_stats_.unique_voxel_candidates_before_insert_filter =
        unique_voxel_candidates_before_insert_filter;
    last_increase_pcd_stats_.unique_voxel_candidates_after_insert_filter =
        unique_voxel_candidates_after_insert_filter;

    // ── 5) Dedup against existing voxels (across-batch) ─────────────────────
    auto octpath_old  = this->oct_path_.contiguous();                    // [No,1] int64
    auto octlevel_old = this->oct_level_.contiguous();                   // [No,1] int8
    const int64_t old_voxel_count = octpath_old.size(0);

    // Packed 1D key: (octpath<<8) | level
    auto key_new = octpath_new.view({-1}).to(torch::kInt64)
                    .mul(256)
                    .add(L_u.view({-1}).to(torch::kInt64));
    auto key_old_all = octpath_old.view({-1}).to(torch::kInt64)
                    .mul(256)
                    .add(octlevel_old.view({-1}).to(torch::kInt64));
    auto bool_opts_old = torch::TensorOptions().dtype(torch::kBool).device(dev);
    auto i32_opts_old = torch::TensorOptions().dtype(torch::kInt32).device(dev);
    if (!is_orb_voxel_.defined() || is_orb_voxel_.size(0) != octpath_old.size(0)) {
        is_orb_voxel_ = torch::zeros({octpath_old.size(0)}, bool_opts_old);
    } else if (is_orb_voxel_.device() != dev) {
        is_orb_voxel_ = is_orb_voxel_.to(dev);
    }
    torch::Tensor new_mask;
    if (octpath_old.numel() == 0) {
        new_mask = torch::ones({Nu}, torch::dtype(torch::kBool).device(dev));
    } else {
        auto is_dup = at::isin(key_new, key_old_all).to(torch::kBool);                                    // [Nu]
        new_mask = ~is_dup;

    }
    // --- Prevent re-adding a parent at L=base_L when children (L > base_L) already exist ---
    {
        if (octpath_old.numel() > 0) {
            auto dev = octpath_old.device();
            const int MAX_L  = max_num_levels_;            // must match SVRecon MAX_NUM_LEVELS
            const int base_L = static_cast<int>(octlevel_);// the level used by createFromPcd()

            TORCH_CHECK(base_L >= 1 && base_L <= MAX_L,
                        "[increasePcd] base_L (octlevel_) out of range: ", base_L,
                        " with MAX_L=", MAX_L);

            // Existing voxels whose level is strictly finer than base_L
            auto Lold_i64    = octlevel_old.view({-1}).to(torch::kInt64);     // [No]
            auto has_children= (Lold_i64 > base_L);                           // [No] bool

            if (has_children.any().item<bool>()) {

                // Mask that clears all octant bits *below* base_L.
                // Bits per level = 3; for a node at level L, its octant sits at shift = 3*(MAX_L - L).
                // To keep bits down to base_L (inclusive), clear the lowest 3*(MAX_L - base_L) bits.
                const int levels_below  = std::max(0, MAX_L - base_L);
                const int bits_to_clear = 3 * levels_below;
                long long lower_mask    = (bits_to_clear > 0) ? ((1LL << bits_to_clear) - 1LL) : 0LL;
                long long keep_mask_ll  = ~lower_mask;

                auto keep_mask = torch::full(
                    {1},
                    static_cast<int64_t>(keep_mask_ll),
                    torch::TensorOptions().dtype(torch::kInt64).device(dev)
                );

                // Compute the ancestor-at-base_L octpaths for those finer voxels
                auto op_old_i64  = octpath_old.view({-1}).to(torch::kInt64);   // [No]
                auto op_anc_base = (op_old_i64 & keep_mask);                   // [No]

                // Keep only rows where L_old > base_L
                auto sel = torch::nonzero(has_children).view({-1});            // [K]
                op_anc_base = op_anc_base.index_select(0, sel);                // [K]

                // Build the ancestor keys at base_L: ((octpath_anc_base<<8) | base_L)
                auto key_children_as_parent = op_anc_base.mul(256)
                                            .add(torch::full_like(op_anc_base,
                                                                static_cast<int64_t>(base_L)));

                // Unique + sorted to make isin faster
                auto unique_sorted_1d = [](const at::Tensor& t)->at::Tensor {
                    TORCH_CHECK(t.dim() == 1, "unique_sorted_1d expects a 1-D tensor");
                    if (t.numel() <= 1) return t.contiguous();
                    auto sort_res = t.sort(/*dim=*/0);
                    auto sorted   = std::get<0>(sort_res);
                    using torch::indexing::Slice;
                    auto keep = torch::empty_like(sorted, torch::kBool);
                    keep.index_put_({0}, true);
                    auto neq = sorted.index({Slice(1, torch::indexing::None)})
                            != sorted.index({Slice(torch::indexing::None, -1)});
                    keep.index_put_({Slice(1, torch::indexing::None)}, neq);
                    auto idx = torch::nonzero(keep).view({-1});
                    return sorted.index_select(0, idx).contiguous();
                };
                key_children_as_parent = unique_sorted_1d(key_children_as_parent);

                // If a candidate NEW (octpath, base_L) matches any ancestor of an existing finer voxel,
                // then inserting that parent would collide with existing children later.
                auto would_collide_parent =
                    at::isin(key_new, key_children_as_parent).to(torch::kBool);  // [Nu]

                if (would_collide_parent.any().item<bool>()) {
                    new_mask = new_mask & (~would_collide_parent);
                }
            }
        }
    }

    auto sel = torch::nonzero(new_mask).view({-1});
    const int64_t pending_support_updates = 0;
    const int64_t new_voxel_candidates = sel.size(0);
    const int64_t duplicate_existing_voxels =
        unique_voxel_candidates_after_insert_filter - new_voxel_candidates;
    last_increase_pcd_stats_.duplicate_existing_voxels = duplicate_existing_voxels;
    last_increase_pcd_stats_.new_voxels = new_voxel_candidates;
    last_increase_pcd_stats_.pending_promotions = 0;
    last_increase_pcd_stats_.pending_support_updates = pending_support_updates;
    if (add_as_orb) {
        // Weak monocular ORB candidates allocate topology, but only validated
        // support enters the metric SDF prior.
        appendSparseSupportPoints_(orb_metric_support);
    }
    pending_sdf_init_mode_ = topology_sdf_init_mode_;
    if (xyz_cpu.defined() && xyz_cpu.numel() > 0) {
        const bool uses_local_point_prior =
            pending_sdf_init_mode_ == SdfInitMode::SignedPointPrior ||
            pending_sdf_init_mode_ == SdfInitMode::WeakSurfacePrior;
        if (uses_local_point_prior) {
            sdf_init_local_support_points_ = xyz.detach().contiguous();
        } else {
            sdf_init_local_support_points_ = torch::Tensor();
        }
        if (pending_sdf_init_mode_ == SdfInitMode::SignedPointPrior &&
            !add_as_orb) {
            appendSparseSupportPoints_(xyz);
        }
    }
    if (sel.numel() == 0) {
        if (orb_metric_support.defined() &&
            orb_metric_support.numel() > 0) {
            updateExistingSupportSdfFromPoints_(
                orb_metric_support,
                torch::Tensor(),
                cams);
        }
        sdf_init_local_support_points_ = torch::Tensor();
        pending_sdf_init_mode_ = SdfInitMode::SignedPointPrior;
        pending_real_insert_rr_entity_path_.clear();
        return;
    }
    auto octpath_add = octpath_new.index_select(0, sel);                                                // [Nk,1]
    auto L_add       = L_u.index_select(0, sel);                                                         // [Nk,1]
    auto rgb_add     = rgb_u.index_select(0, sel);                                                       // [Nk,3]
    const int Nk = sel.size(0);
    if (Nk > 0) {
        auto [accepted_centers, accepted_sizes] = decodeOctpath(
            octpath_add.contiguous(),
            L_add.contiguous(),
            scene_center_.contiguous(),
            scene_extent_.contiguous());
        (void)accepted_sizes;
        auto accepted_centers_cpu =
            accepted_centers.detach()
                .to(torch::kCPU)
                .to(torch::kFloat32)
                .contiguous();
        if (!real_pcd_points_accum_cpu_.defined() ||
            real_pcd_points_accum_cpu_.numel() == 0) {
            real_pcd_points_accum_cpu_ = accepted_centers_cpu;
        } else {
            real_pcd_points_accum_cpu_ =
                torch::cat(
                    {real_pcd_points_accum_cpu_, accepted_centers_cpu},
                    0)
                    .contiguous();
        }
    }

    ensureFusedSdfField();
    torch::Tensor old_vox_sdf_values =
        voxelCornerScalarFromGrid_(fused_sdf_grid_pts_);
    torch::Tensor old_vox_sdf_weights =
        voxelCornerScalarFromGrid_(fused_sdf_weights_);

    // ── 6) Append topology (old preserved) ──────────────────────────────────
    this->oct_path_ = torch::cat({octpath_old,  octpath_add}, 0).contiguous();
    this->oct_level_ = torch::cat({octlevel_old, L_add}, 0).contiguous();
    {
        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto i32_opts  = torch::TensorOptions().dtype(torch::kInt32).device(dev);
        if (!is_orb_voxel_.defined() || is_orb_voxel_.size(0) != octpath_old.size(0)) {
            is_orb_voxel_ = torch::zeros({octpath_old.size(0)}, bool_opts);
        } else if (is_orb_voxel_.device() != octpath_old.device()) {
            is_orb_voxel_ = is_orb_voxel_.to(octpath_old.device());
        }
        if (!is_inactive_geo_voxel_.defined() || is_inactive_geo_voxel_.size(0) != octpath_old.size(0)) {
            is_inactive_geo_voxel_ = torch::zeros({octpath_old.size(0)}, bool_opts);
        } else if (is_inactive_geo_voxel_.device() != octpath_old.device()) {
            is_inactive_geo_voxel_ = is_inactive_geo_voxel_.to(octpath_old.device());
        }
        if (!is_rgbd_fill_render_holes_voxel_.defined() ||
            is_rgbd_fill_render_holes_voxel_.size(0) != octpath_old.size(0)) {
            is_rgbd_fill_render_holes_voxel_ = torch::zeros({octpath_old.size(0)}, bool_opts);
        } else if (is_rgbd_fill_render_holes_voxel_.device() != octpath_old.device()) {
            is_rgbd_fill_render_holes_voxel_ =
                is_rgbd_fill_render_holes_voxel_.to(octpath_old.device());
        }
        if (!is_monocular_rendered_depth_voxel_.defined() ||
            is_monocular_rendered_depth_voxel_.size(0) != octpath_old.size(0)) {
            is_monocular_rendered_depth_voxel_ =
                torch::zeros({octpath_old.size(0)}, bool_opts);
        } else if (is_monocular_rendered_depth_voxel_.device() !=
                   octpath_old.device()) {
            is_monocular_rendered_depth_voxel_ =
                is_monocular_rendered_depth_voxel_.to(octpath_old.device());
        }
        if (!is_monocular_mvs_voxel_.defined() ||
            is_monocular_mvs_voxel_.size(0) != octpath_old.size(0)) {
            is_monocular_mvs_voxel_ =
                torch::zeros({octpath_old.size(0)}, bool_opts);
        } else if (is_monocular_mvs_voxel_.device() !=
                   octpath_old.device()) {
            is_monocular_mvs_voxel_ =
                is_monocular_mvs_voxel_.to(octpath_old.device());
        }
        if (!exist_since_iter_.defined() || exist_since_iter_.size(0) != octpath_old.size(0)) {
            exist_since_iter_ = torch::zeros({octpath_old.size(0)}, i32_opts);
        } else if (exist_since_iter_.device() != octpath_old.device()) {
            exist_since_iter_ = exist_since_iter_.to(octpath_old.device());
        }
        if (!exist_since_kf_.defined() || exist_since_kf_.size(0) != octpath_old.size(0)) {
            exist_since_kf_ = torch::full({octpath_old.size(0)}, static_cast<int32_t>(-1), i32_opts);
        } else if (exist_since_kf_.device() != octpath_old.device()) {
            exist_since_kf_ = exist_since_kf_.to(octpath_old.device());
        }
        if (Nk > 0) {
            auto orb_add_flag = torch::full({Nk}, add_as_orb, bool_opts);
            auto inactive_geo_add_flag = torch::full({Nk}, add_as_inactive_geo, bool_opts);
            auto rgbd_fill_render_holes_add_flag =
                torch::full({Nk}, add_as_rgbd_fill_render_holes, bool_opts);
            auto monocular_rendered_depth_add_flag =
                torch::full({Nk}, add_as_monocular_rendered_depth, bool_opts);
            auto monocular_mvs_add_flag =
                torch::full({Nk}, add_as_monocular_mvs, bool_opts);
            auto exist_since_add = torch::full(
                {Nk}, static_cast<int32_t>(iteration), i32_opts);
            auto exist_since_kf_add = torch::full(
                {Nk}, current_kf_count, i32_opts);
            is_orb_voxel_ = torch::cat({is_orb_voxel_, orb_add_flag}, 0).contiguous();
            is_inactive_geo_voxel_ = torch::cat({is_inactive_geo_voxel_, inactive_geo_add_flag}, 0).contiguous();
            is_rgbd_fill_render_holes_voxel_ =
                torch::cat({is_rgbd_fill_render_holes_voxel_, rgbd_fill_render_holes_add_flag}, 0).contiguous();
            is_monocular_rendered_depth_voxel_ =
                torch::cat(
                    {is_monocular_rendered_depth_voxel_,
                     monocular_rendered_depth_add_flag},
                    0)
                    .contiguous();
            is_monocular_mvs_voxel_ =
                torch::cat(
                    {is_monocular_mvs_voxel_, monocular_mvs_add_flag},
                    0)
                    .contiguous();
            exist_since_iter_ = torch::cat({exist_since_iter_, exist_since_add}, 0).contiguous();
            exist_since_kf_ = torch::cat({exist_since_kf_, exist_since_kf_add}, 0).contiguous();
        }
    }

    // ── 7) Append learnables for new rows ───────────────────────────────────
    // _subdiv_p
    auto subdiv_add = torch::ones({Nk,1}, torch::dtype(torch::kFloat32).device(dev));
    this->subdiv_p_ = torch::cat({this->subdiv_p_.detach(), subdiv_add}, 0)
                          .contiguous()
                          .detach()
                          .requires_grad_();
    if (!this->subdiv_meta_.defined() || this->subdiv_meta_.numel() == 0) {
        this->subdiv_meta_ = torch::zeros(
            {old_voxel_count, 1},
            torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    }
    this->subdiv_meta_ = torch::cat(
        {this->subdiv_meta_.to(dev).to(torch::kFloat32).contiguous(),
         torch::zeros({Nk, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(dev))},
        0).contiguous();

    // _sh0 from fused rgb
    torch::Tensor sh0_add = torch::empty({0, 3}, torch::dtype(torch::kFloat32).device(dev));
    if (Nk > 0) {
        sh0_add = rgbToShZero(rgb_add.contiguous()).contiguous(); // [Nk,3]
    }
    // _shs zeros
    const int n_sh_rest = (max_sh_degree_ + 1)*(max_sh_degree_ + 1) - 1;
    auto shs_add = torch::zeros({Nk, n_sh_rest, 3}, torch::dtype(torch::kFloat32).device(dev));

    // ── 8) Rebuild grid links; grow _geo_grid_pts only if grid expanded ─────
    auto [grid_pts_key_new, vox_key_new] =
        buildGridPtsLink(
            this->oct_path_.contiguous(),
            this->oct_level_.contiguous(),
            max_num_levels_);
    (void)vox_key_new;
    // ── 9) Append rows to optimizer param groups ────────────────────────────
    // Important: only call for non-empty additions.
    rebuildGeoGridForNewGridKeys_(grid_pts_key_new, /*default_value=*/-10.0f);
    sdf_init_local_support_points_ = torch::Tensor();
    pending_sdf_init_mode_ = SdfInitMode::SignedPointPrior;
    this->grid_pts_key_ = grid_pts_key_new.contiguous();
    this->vox_key_ = vox_key_new.contiguous();
    if (Nk > 0) {
        appendGroup_(/*group_idx=*/1, /*add_rows=*/sh0_add, &this->sh0_);
        if (clear_cuda_cache_before_parameter_append) {
            c10::cuda::CUDACachingAllocator::emptyCache();
        }
        appendGroup_(/*group_idx=*/2, /*add_rows=*/shs_add, &this->shs_);
        if (clear_cuda_cache_before_parameter_append) {
            c10::cuda::CUDACachingAllocator::emptyCache();
        }
    }

    // ── 10) Rebuild renderer fields from the C++ topology ───────────────────
    this->oct_path_      = this->oct_path_.contiguous();
    this->oct_level_     = this->oct_level_.contiguous();
    {
        auto [center, size] = decodeOctpath(
            this->oct_path_.contiguous(),
            this->oct_level_.contiguous(),
            this->scene_center_.contiguous(),
            this->scene_extent_.contiguous());
        this->center_        = center.contiguous();
        this->size_          = size.squeeze(1).contiguous();
        // Topology has not changed since the link built above. Reusing it avoids
        // a second full N x 8 corner expansion and CUDA unique_dim allocation.
        this->grid_pts_key_  = grid_pts_key_new.contiguous();
        this->vox_key_       = vox_key_new.contiguous();
    }
    this->vox_size_inv_  = 1.0f / size_;
    {
        const int64_t N_final = this->vox_key_.size(0);
        const int64_t N_added = std::max<int64_t>(0, N_final - old_voxel_count);
        auto dev_sdf = old_vox_sdf_values.defined()
            ? old_vox_sdf_values.device()
            : (_geo_grid_pts_.defined() ? _geo_grid_pts_.device() : torch::Device(device_type_));
        auto value_opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev_sdf);
        if (!old_vox_sdf_values.defined() || old_vox_sdf_values.size(0) != old_voxel_count) {
            old_vox_sdf_values = torch::zeros({old_voxel_count, 8, 1}, value_opts);
            old_vox_sdf_weights = torch::zeros({old_voxel_count, 8, 1}, value_opts);
        }
        auto added_sdf = torch::zeros({N_added, 8, 1}, value_opts);
        auto added_weights = torch::zeros({N_added, 8, 1}, value_opts);
        auto vox_sdf = torch::cat({old_vox_sdf_values, added_sdf}, 0).contiguous();
        auto vox_weights = torch::cat({old_vox_sdf_weights, added_weights}, 0).contiguous();
        rebuildFusedSdfFieldFromVoxelCorners_(vox_sdf, vox_weights);
    }
    // (optimizer params already set by appendGroup_; just ensure requires_grad)
    this->subdiv_p_ = this->subdiv_p_.contiguous();
    if (!this->subdiv_p_.requires_grad()) {
        this->subdiv_p_.set_requires_grad(true);
    }
    if (!this->subdiv_meta_.defined() || this->subdiv_meta_.size(0) != center_.size(0)) {
        this->subdiv_meta_ = torch::zeros(
            {center_.size(0), 1},
            torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    }
    if (!this->is_leaf_.defined() || this->is_leaf_.size(0) != center_.size(0)) {
        auto leaf_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto aligned = torch::ones({center_.size(0), 1}, leaf_opts);
        if (this->is_leaf_.defined() && this->is_leaf_.numel() > 0) {
            auto old = this->is_leaf_.to(dev).to(torch::kBool).reshape({-1, 1});
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        this->is_leaf_ = aligned.contiguous();
    }
    // stats buffer resize
    this->max_w_ = torch::zeros({center_.size(0), 1}, torch::dtype(torch::kFloat32).device(dev));
    // Keep provenance tensor aligned with current topology size.
    if (!is_orb_voxel_.defined() || is_orb_voxel_.size(0) != center_.size(0)) {
        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, bool_opts);
        if (is_orb_voxel_.defined() && is_orb_voxel_.numel() > 0) {
            auto old = is_orb_voxel_.to(dev).to(torch::kBool).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        is_orb_voxel_ = aligned;
    }
    if (!is_inactive_geo_voxel_.defined() || is_inactive_geo_voxel_.size(0) != center_.size(0)) {
        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, bool_opts);
        if (is_inactive_geo_voxel_.defined() && is_inactive_geo_voxel_.numel() > 0) {
            auto old = is_inactive_geo_voxel_.to(dev).to(torch::kBool).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        is_inactive_geo_voxel_ = aligned;
    }
    if (!is_rgbd_fill_render_holes_voxel_.defined() ||
        is_rgbd_fill_render_holes_voxel_.size(0) != center_.size(0)) {
        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, bool_opts);
        if (is_rgbd_fill_render_holes_voxel_.defined() &&
            is_rgbd_fill_render_holes_voxel_.numel() > 0) {
            auto old = is_rgbd_fill_render_holes_voxel_.to(dev).to(torch::kBool).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        is_rgbd_fill_render_holes_voxel_ = aligned;
    }
    auto align_bool_metadata = [&](torch::Tensor& metadata) {
        if (metadata.defined() && metadata.size(0) == center_.size(0)) {
            metadata =
                metadata.to(dev).to(torch::kBool).contiguous();
            return;
        }
        auto bool_opts =
            torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, bool_opts);
        if (metadata.defined() && metadata.numel() > 0) {
            auto old =
                metadata.to(dev).to(torch::kBool).reshape({-1}).contiguous();
            const int64_t copy_n =
                std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        metadata = aligned.contiguous();
    };
    align_bool_metadata(is_monocular_rendered_depth_voxel_);
    align_bool_metadata(is_monocular_mvs_voxel_);
    // Keep exist_since_iter tensor aligned with current topology size.
    if (!exist_since_iter_.defined() || exist_since_iter_.size(0) != center_.size(0)) {
        auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, i32_opts);
        if (exist_since_iter_.defined() && exist_since_iter_.numel() > 0) {
            auto old = exist_since_iter_.to(dev).to(torch::kInt32).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        exist_since_iter_ = aligned;
    }
    if (!exist_since_kf_.defined() || exist_since_kf_.size(0) != center_.size(0)) {
        auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(dev);
        auto aligned = torch::full({center_.size(0)}, static_cast<int32_t>(-1), i32_opts);
        if (exist_since_kf_.defined() && exist_since_kf_.numel() > 0) {
            auto old = exist_since_kf_.to(dev).to(torch::kInt32).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        exist_since_kf_ = aligned;
    }
    if (orb_metric_support.defined() &&
        orb_metric_support.numel() > 0) {
        updateExistingSupportSdfFromPoints_(
            orb_metric_support,
            torch::Tensor(),
            cams);
    }

    // ── 10) Re-register with optimizer (new rows appended) ───────────────────
    VOXEL_MODEL_TENSORS_TO_VEC

    if (!pending_real_insert_rr_entity_path_.empty()) {
        pending_real_insert_rr_entity_path_.clear();
    }

}

void VoxelModel::increasePcd(
    torch::Tensor& new_point_cloud,
    torch::Tensor& new_colors,
    const int iteration,
    const std::vector<sv::MiniCam>& cams,
    const bool clear_cuda_cache_before_parameter_append)
{
    // Tensor adapter following Photo-SLAM GaussianModel::increasePcd; the
    // vector overload above remains the single topology-update implementation.

    if (!new_point_cloud.defined() || !new_colors.defined())
        return;

    TORCH_CHECK(
        new_point_cloud.dim() == 2 && new_point_cloud.size(1) == 3,
        "VoxelModel::increasePcd(tensor): new_point_cloud must be [N,3]"
    );
    TORCH_CHECK(
        new_colors.dim() == 2 && new_colors.size(1) == 3,
        "VoxelModel::increasePcd(tensor): new_colors must be [N,3]"
    );
    TORCH_CHECK(
        new_point_cloud.size(0) == new_colors.size(0),
        "VoxelModel::increasePcd(tensor): points/colors size mismatch"
    );

    const int64_t N = new_point_cloud.size(0);
    if (N == 0)
        return;

    // Ensure CPU + contiguous
    auto xyz_cpu = new_point_cloud.to(torch::kCPU).contiguous();
    auto rgb_cpu = new_colors.to(torch::kCPU).contiguous();

    TORCH_CHECK(
        xyz_cpu.scalar_type() == torch::kFloat32 &&
        rgb_cpu.scalar_type() == torch::kFloat32,
        "VoxelModel::increasePcd(tensor): tensors must be float32"
    );

    // Flatten to 1D [3*N]
    auto xyz_flat = xyz_cpu.view({-1});  // [3N]
    auto rgb_flat = rgb_cpu.view({-1});  // [3N]

    std::vector<float> points(3 * N);
    std::vector<float> cols(3 * N);

    // Copy XYZ directly
    std::memcpy(
        points.data(),
        xyz_flat.data_ptr<float>(),
        points.size() * sizeof(float)
    );

    // Keep colors as-is; vector-based increasePcd now auto-detects [0,1] vs [0,255].
    const float* rgb_ptr = rgb_flat.data_ptr<float>();
    for (int64_t i = 0; i < static_cast<int64_t>(cols.size()); ++i) {
        cols[i] = rgb_ptr[i];
    }

    // Reuse the main point-cloud insertion pipeline.
    increasePcd(
        points,
        cols,
        iteration,
        cams,
        clear_cuda_cache_before_parameter_append);
}

} // namespace sv
