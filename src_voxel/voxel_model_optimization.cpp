#include "src_voxel/voxel_model_internal.h"

namespace sv {

void VoxelModel::createTrainer(float geo_lr, float sh0_lr, float shs_lr,
                               float beta1, float beta2, float eps,
                               const std::vector<int>& milestones,
                               float gamma,
                               float log_s_lr)
{
    // Initializes C++ Adam groups and the milestone scheduler.
    // SVRecon reference: SVOptimizer.optimizer_init.
    optimizer_geo_lr_ = geo_lr;
    optimizer_sh0_lr_ = sh0_lr;
    optimizer_shs_lr_ = shs_lr;
    optimizer_log_s_lr_ = log_s_lr;
    optimizer_beta1_ = beta1;
    optimizer_beta2_ = beta2;
    optimizer_eps_ = eps;
    scheduler_milestones_ = milestones;
    std::sort(scheduler_milestones_.begin(), scheduler_milestones_.end());
    scheduler_gamma_ = gamma;
    scheduler_epoch_ = -1;
    optimizer_initialized_ = true;

    adam_geo_ = AdamGroupState{};
    adam_sh0_ = AdamGroupState{};
    adam_shs_ = AdamGroupState{};
    adam_log_s_ = AdamGroupState{};

    _geo_grid_pts_ = _geo_grid_pts_.contiguous().detach().requires_grad_(true);
    log_s_ = log_s_.defined() && log_s_.numel() > 0
        ? log_s_.contiguous().detach().requires_grad_(true)
        : torch::full({1}, 0.3f, _geo_grid_pts_.options()).requires_grad_(true);
    sh0_ = sh0_.contiguous().detach().requires_grad_(true);
    shs_ = shs_.contiguous().detach().requires_grad_(true);
}

void VoxelModel::appendGroup_(int group_idx,
                              const torch::Tensor& add_rows,
                              torch::Tensor* out_member_param) {
    torch::Tensor old_param = *out_member_param;
    torch::Tensor new_param =
        torch::cat({old_param, add_rows}, /*dim=*/0).contiguous().detach().requires_grad_(true);

    auto extend_state = [&](AdamGroupState& state) {
        if (!state.exp_avg.defined() || !state.exp_avg_sq.defined()) {
            return;
        }
        torch::Tensor zeros = torch::zeros_like(add_rows);
        state.exp_avg = torch::cat({state.exp_avg, zeros}, 0).contiguous();
        state.exp_avg_sq = torch::cat({state.exp_avg_sq, zeros}, 0).contiguous();
    };

    if (group_idx == 0) {
        extend_state(adam_geo_);
    } else if (group_idx == 1) {
        extend_state(adam_sh0_);
    } else if (group_idx == 2) {
        extend_state(adam_shs_);
    }

    *out_member_param = new_param;
}

VoxelModel::StatPkg
VoxelModel::computeTrainingStat(const std::vector<MiniCam>& cams) {
    // Computes maximum render weight, minimum sampling interval, and view count.
    // SVRecon reference: SVAdaptive.compute_training_stat.
    freezeVoxGeo();

    const int64_t N = center_.size(0);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_type_);
    // auto max_w            = torch::zeros({N,1}, opts);
    this->max_w_.zero_();
    auto min_samp_interval = torch::full ({N,1}, 1e30f, opts);
    auto view_cnt         = torch::zeros({N,1}, opts);

    for (const auto& cam : cams) {
        // std::cout << "rendering cam " << cam.width << "x" << cam.height << "\n";
        // auto pkg = render(cam, torch::Tensor(), cam.height, cam.width, color_mode='dontcare', track_max_w=True);
        std::unordered_map<std::string, torch::Tensor> pkg;
        {
            torch::NoGradGuard no_grad;
            pkg = render(
                cam,
                cam.height,
                cam.width,
                torch::Tensor(),
                "dontcare",
                true);
        }

        if (!pkg.count("max_w") || !pkg.at("max_w").defined())
            continue;

        auto max_w_i = pkg["max_w"].to(device_type_);
        this->max_w_ = torch::maximum(this->max_w_, max_w_i);

        // visibility indices for current cam
        auto vis_idx = (max_w_i.squeeze(1) > 0).nonzero().squeeze(1);  // [K]

        if (vis_idx.numel() > 0) {
            // z distance along camera forward
            auto pos   = camPosition_(cam, device_type_);
            auto fwd   = camForward_(cam,   device_type_);
            auto vc    = center_.index({vis_idx});             // [K,3]
            auto zdist = ((vc - pos) * fwd).sum(1, true).abs(); // [K,1] (abs -> guard sign)

            float pix_size = camPixSize_(cam);                 // scalar
            auto samp_itv  = zdist * pix_size;                 // [K,1]

            // min over views
            auto cur = min_samp_interval.index({vis_idx});
            min_samp_interval.index_put_({vis_idx}, torch::minimum(cur, samp_itv));

            // view count
            view_cnt.index_put_({vis_idx}, view_cnt.index({vis_idx}) + 1);
        }
    }

    unfreezeVoxGeo();
    return { this->max_w_.contiguous(), min_samp_interval.contiguous(), view_cnt.contiguous() };
}

torch::Tensor VoxelModel::computeOcclusionAwareViewCount(
    const std::vector<MiniCam>& cams)
{
    // MonoGS co-visibility counts a primitive only while post-compositing
    // transmittance remains above 0.5. The rasterizer reports that event once
    // per voxel and camera; this method accumulates it over the active window.
    freezeVoxGeo();

    const int64_t voxel_count = center_.size(0);
    auto opts = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(device_type_);
    torch::Tensor view_count = torch::zeros({voxel_count, 1}, opts);

    try {
        torch::NoGradGuard no_grad;
        RenderOpts render_opts;
        render_opts.track_occlusion_visibility = true;
        for (const auto& cam : cams) {
            auto pkg = render(
                cam,
                cam.height,
                cam.width,
                torch::Tensor(),
                "dontcare",
                false,
                std::nullopt,
                false,
                false,
                false,
                false,
                false,
                render_opts);
            auto visibility_it = pkg.find("occlusion_visible");
            if (visibility_it == pkg.end() ||
                !visibility_it->second.defined() ||
                visibility_it->second.numel() != voxel_count) {
                continue;
            }
            view_count += visibility_it->second
                              .to(device_type_)
                              .to(torch::kFloat32)
                              .reshape({voxel_count, 1});
        }
    } catch (...) {
        unfreezeVoxGeo();
        throw;
    }

    unfreezeVoxGeo();
    return view_count.contiguous();
}

void VoxelModel::optimizerZeroGrad() {
    // Clears gradients for the parameter groups used by the SVRecon renderer.
    // SVRecon reference: the optimizer zero_grad step in train.py.
    _geo_grid_pts_.mutable_grad() = torch::Tensor();
    log_s_.mutable_grad() = torch::Tensor();
    sh0_.mutable_grad() = torch::Tensor();
    shs_.mutable_grad() = torch::Tensor();
    subdiv_p_.mutable_grad() = torch::Tensor();
}

void VoxelModel::optimizerStep() {
    // Applies SVRecon's CUDA unbiased Adam update to each online parameter group.
    // SVRecon reference: svraster_cuda.sparse_adam.SparseAdam.step.
    if (!optimizer_initialized_) {
        return;
    }
    torch::NoGradGuard no_grad;

    auto step_group = [&](torch::Tensor& param,
                          AdamGroupState& state,
                          const float lr) {
        torch::Tensor grad = param.grad();
        if (!grad.defined()) {
            return;
        }
        if (!grad.is_contiguous()) {
            grad = grad.contiguous();
        }
        if (!state.exp_avg.defined() ||
            !state.exp_avg_sq.defined() ||
            state.exp_avg.sizes() != param.sizes() ||
            state.exp_avg_sq.sizes() != param.sizes()) {
            state.exp_avg = torch::zeros_like(param);
            state.exp_avg_sq = torch::zeros_like(param);
            state.step = 0;
        }

        state.step += 1;
        SVRECON_ADAM_STEP::unbiased_adam_step(
            /*sparse=*/false,
            param,
            grad,
            state.exp_avg,
            state.exp_avg_sq,
            static_cast<double>(state.step),
            static_cast<double>(lr),
            static_cast<double>(optimizer_beta1_),
            static_cast<double>(optimizer_beta2_),
            optimizer_eps_);
    };

    step_group(_geo_grid_pts_, adam_geo_, optimizer_geo_lr_);
    if (optimizer_log_s_lr_ > 0.0f) {
        step_group(log_s_, adam_log_s_, optimizer_log_s_lr_);
    }
    step_group(sh0_, adam_sh0_, optimizer_sh0_lr_);
    step_group(shs_, adam_shs_, optimizer_shs_lr_);
}

void VoxelModel::schedulerStep()
{
    // Advances the local multi-step learning-rate schedule once per iteration.
    if (!optimizer_initialized_) {
        return;
    }
    scheduler_epoch_ += 1;
    if (std::binary_search(
            scheduler_milestones_.begin(),
            scheduler_milestones_.end(),
            static_cast<int>(scheduler_epoch_))) {
        optimizer_geo_lr_ *= scheduler_gamma_;
        optimizer_sh0_lr_ *= scheduler_gamma_;
        optimizer_shs_lr_ *= scheduler_gamma_;
        optimizer_log_s_lr_ *= scheduler_gamma_;
    }
}

void VoxelModel::pruning(const torch::Tensor& prune_mask) {
    // Removes selected cells and rebuilds shared corners, provenance, evidence,
    // and optimizer state. SVRecon reference: SVAdaptive.pruning.
    auto mask = prune_mask.to(torch::kBool).to(device_type_).contiguous();
    if (mask.dim() == 2 && mask.size(1) == 1) {
        mask = mask.squeeze(1);
    }
    TORCH_CHECK(mask.dim() == 1, "pruning: mask must be [N] or [N,1]");
    const int64_t N_before = mask.size(0);

    auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(device_type_);
    auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(device_type_);
    auto ensure_bool = [&](torch::Tensor& t) {
        if (!t.defined() || t.size(0) != N_before) {
            t = torch::zeros({N_before}, bool_opts);
        } else if (t.device() != mask.device()) {
            t = t.to(mask.device());
        }
    };
    ensure_bool(is_orb_voxel_);
    ensure_bool(is_inactive_geo_voxel_);
    ensure_bool(is_rgbd_fill_render_holes_voxel_);
    ensure_bool(is_monocular_rendered_depth_voxel_);
    ensure_bool(is_monocular_mvs_voxel_);
    if (!is_leaf_.defined() || is_leaf_.size(0) != N_before) {
        is_leaf_ = torch::ones({N_before, 1}, bool_opts);
    } else {
        is_leaf_ = is_leaf_.to(mask.device()).to(torch::kBool).reshape({N_before, 1});
    }
    // SVRecon retains fine-level internal parents for hierarchical continuity.
    // They are not rendered and must not be removed by leaf-surface pruning.
    mask = mask & is_leaf_.view({-1});
    if (!exist_since_iter_.defined() || exist_since_iter_.size(0) != N_before) {
        exist_since_iter_ = torch::zeros({N_before}, i32_opts);
    } else if (exist_since_iter_.device() != mask.device()) {
        exist_since_iter_ = exist_since_iter_.to(mask.device());
    }
    if (!exist_since_kf_.defined() || exist_since_kf_.size(0) != N_before) {
        exist_since_kf_ = torch::full({N_before}, static_cast<int32_t>(-1), i32_opts);
    } else if (exist_since_kf_.device() != mask.device()) {
        exist_since_kf_ = exist_since_kf_.to(mask.device());
    }

    auto kept_idx = torch::nonzero(~mask).view({-1}).to(torch::kLong).contiguous();
    if (kept_idx.numel() == 0) {
        return;
    }

    auto old_octpath = this->oct_path_.contiguous();
    auto old_octlevel = this->oct_level_.contiguous();
    auto old_vox_key = this->vox_key_.contiguous().clone();
    auto old_geo_grid_pts = this->_geo_grid_pts_.detach().contiguous();
    auto old_vox_grid_pts_val =
        old_geo_grid_pts.index({old_vox_key.to(old_geo_grid_pts.device())}).contiguous();
    ensureFusedSdfField();
    auto old_vox_sdf_values = voxelCornerScalarFromGrid_(fused_sdf_grid_pts_);
    auto old_vox_sdf_weights = voxelCornerScalarFromGrid_(fused_sdf_weights_);

    this->oct_path_ = old_octpath.index_select(0, kept_idx.to(old_octpath.device())).contiguous();
    this->oct_level_ = old_octlevel.index_select(0, kept_idx.to(old_octlevel.device())).contiguous();
    auto [new_center, new_size] = decodeOctpath(
        this->oct_path_.contiguous(),
        this->oct_level_.contiguous(),
        this->scene_center_.contiguous(),
        this->scene_extent_.contiguous());
    auto [new_grid_pts_key, new_vox_key] =
        buildGridPtsLink(this->oct_path_.contiguous(), this->oct_level_.contiguous(), max_num_levels_);
    this->center_ = new_center.contiguous();
    this->size_ = new_size.squeeze(1).contiguous();
    this->vox_size_inv_ = 1.0f / this->size_;
    this->grid_pts_key_ = new_grid_pts_key.contiguous();
    this->vox_key_ = new_vox_key.contiguous();

    auto kept_idx_sh0 = kept_idx.to(this->sh0_.device());
    auto old_subdiv_grad = this->subdiv_p_.grad();
    this->sh0_ = this->sh0_.detach().index_select(0, kept_idx_sh0).contiguous().requires_grad_(true);
    this->shs_ = this->shs_.detach().index_select(0, kept_idx_sh0).contiguous().requires_grad_(true);
    this->subdiv_p_ =
        this->subdiv_p_.detach().index_select(0, kept_idx.to(this->subdiv_p_.device()))
            .contiguous().requires_grad_(true);
    if (this->subdiv_meta_.defined() &&
        this->subdiv_meta_.dim() == 2 &&
        this->subdiv_meta_.size(0) == old_octpath.size(0)) {
        this->subdiv_meta_ =
            this->subdiv_meta_.to(this->subdiv_p_.device())
                .to(torch::kFloat32)
                .index_select(0, kept_idx.to(this->subdiv_p_.device()))
                .contiguous();
    } else {
        this->subdiv_meta_ = torch::zeros_like(this->subdiv_p_);
    }
    if (old_subdiv_grad.defined()) {
        this->subdiv_p_.mutable_grad() =
            old_subdiv_grad.index_select(0, kept_idx.to(old_subdiv_grad.device())).contiguous();
    }

    auto new_vox_val =
        old_vox_grid_pts_val.index_select(0, kept_idx.to(old_vox_grid_pts_val.device())).contiguous();
    this->_geo_grid_pts_ =
        aggregateVoxelCornersIntoGridPts(
            this->grid_pts_key_.size(0),
            this->vox_key_,
            new_vox_val).detach().contiguous().requires_grad_(true);
    auto kept_sdf =
        old_vox_sdf_values.index_select(0, kept_idx.to(old_vox_sdf_values.device())).contiguous();
    auto kept_weights =
        old_vox_sdf_weights.index_select(0, kept_idx.to(old_vox_sdf_weights.device())).contiguous();
    rebuildFusedSdfFieldFromVoxelCorners_(kept_sdf, kept_weights);

    this->max_w_ = torch::zeros(
        {center_.size(0), 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

    auto remap_bool = [&](torch::Tensor& t) {
        t = t.to(mask.device()).to(torch::kBool)
                .index_select(0, kept_idx.to(mask.device()))
                .contiguous();
    };
    remap_bool(is_orb_voxel_);
    remap_bool(is_inactive_geo_voxel_);
    remap_bool(is_rgbd_fill_render_holes_voxel_);
    remap_bool(is_monocular_rendered_depth_voxel_);
    remap_bool(is_monocular_mvs_voxel_);
    is_leaf_ = is_leaf_.index_select(0, kept_idx.to(is_leaf_.device())).contiguous();
    exist_since_iter_ =
        exist_since_iter_.to(mask.device()).to(torch::kInt32)
            .index_select(0, kept_idx.to(mask.device())).contiguous();
    exist_since_kf_ =
        exist_since_kf_.to(mask.device()).to(torch::kInt32)
            .index_select(0, kept_idx.to(mask.device())).contiguous();
    VOXEL_MODEL_TENSORS_TO_VEC
}

void VoxelModel::subdividing(const torch::Tensor& subdivide_mask) {
    // Splits selected leaves into eight children and interpolates shared-corner
    // fields while retaining refined parents when required by SVRecon.
    // SVRecon reference: SVAdaptive.subdividing.
    auto mask = subdivide_mask.to(torch::kBool).to(device_type_).contiguous();
    if (mask.dim() == 2 && mask.size(1) == 1) {
        mask = mask.squeeze(1);
    }
    TORCH_CHECK(mask.dim() == 1, "subdividing: mask must be [N] or [N,1]");
    const int64_t N_before = mask.size(0);

    auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(device_type_);
    auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(device_type_);
    auto ensure_bool = [&](torch::Tensor& t) {
        if (!t.defined() || t.size(0) != N_before) {
            t = torch::zeros({N_before}, bool_opts);
        } else if (t.device() != mask.device()) {
            t = t.to(mask.device());
        }
    };
    ensure_bool(is_orb_voxel_);
    ensure_bool(is_inactive_geo_voxel_);
    ensure_bool(is_rgbd_fill_render_holes_voxel_);
    ensure_bool(is_monocular_rendered_depth_voxel_);
    ensure_bool(is_monocular_mvs_voxel_);
    if (!is_leaf_.defined() || is_leaf_.size(0) != N_before) {
        is_leaf_ = torch::ones({N_before, 1}, bool_opts);
    } else {
        is_leaf_ = is_leaf_.to(mask.device()).to(torch::kBool).reshape({N_before, 1});
    }
    mask = mask & is_leaf_.view({-1});
    if (!exist_since_iter_.defined() || exist_since_iter_.size(0) != N_before) {
        exist_since_iter_ = torch::zeros({N_before}, i32_opts);
    } else if (exist_since_iter_.device() != mask.device()) {
        exist_since_iter_ = exist_since_iter_.to(mask.device());
    }
    if (!exist_since_kf_.defined() || exist_since_kf_.size(0) != N_before) {
        exist_since_kf_ = torch::full({N_before}, static_cast<int32_t>(-1), i32_opts);
    } else if (exist_since_kf_.device() != mask.device()) {
        exist_since_kf_ = exist_since_kf_.to(mask.device());
    }
    auto exist_since_before = exist_since_iter_.to(torch::kInt32).contiguous();
    auto exist_since_kf_before = exist_since_kf_.to(torch::kInt32).contiguous();

    auto subdiv_idx = torch::nonzero(mask).view({-1});
    if (subdiv_idx.numel() == 0) {
        return;
    }

    auto old_octpath = this->oct_path_.contiguous();
    auto old_octlevel = this->oct_level_.contiguous();
    auto depth_rel =
        old_octlevel.to(mask.device()).to(torch::kInt32).reshape({-1}) - outside_level_;
    auto parent_keep_mask = mask & (depth_rel >= 9);
    auto kept_idx = torch::nonzero((~mask) | parent_keep_mask)
                        .view({-1}).to(torch::kLong).contiguous();

    auto old_vox_key = this->vox_key_.contiguous().clone();
    auto old_geo_grid_pts = this->_geo_grid_pts_.detach().contiguous();
    auto old_vox_grid_pts_val =
        old_geo_grid_pts.index({old_vox_key.to(old_geo_grid_pts.device())}).contiguous();
    ensureFusedSdfField();
    auto old_vox_sdf_values = voxelCornerScalarFromGrid_(fused_sdf_grid_pts_);
    auto old_vox_sdf_weights = voxelCornerScalarFromGrid_(fused_sdf_weights_);

    auto [child_octpath, child_octlevel] =
        genChildrenOctpath(
            old_octpath.index_select(0, subdiv_idx.to(old_octpath.device())).contiguous(),
            old_octlevel.index_select(0, subdiv_idx.to(old_octlevel.device())).contiguous(),
            max_num_levels_);
    this->oct_path_ = torch::cat(
        {old_octpath.index_select(0, kept_idx.to(old_octpath.device())).contiguous(),
         child_octpath.to(old_octpath.device()).contiguous()}, 0).contiguous();
    this->oct_level_ = torch::cat(
        {old_octlevel.index_select(0, kept_idx.to(old_octlevel.device())).contiguous(),
         child_octlevel.to(old_octlevel.device()).contiguous()}, 0).contiguous();

    auto old_subdiv_grad = this->subdiv_p_.grad();
    auto subdiv_children =
        this->subdiv_p_.detach()
            .index_select(0, subdiv_idx.to(this->subdiv_p_.device()))
            .repeat_interleave(8, 0)
            .contiguous();
    this->subdiv_p_ = torch::cat(
        {this->subdiv_p_.detach().index_select(0, kept_idx.to(this->subdiv_p_.device())).contiguous(),
         subdiv_children}, 0).contiguous().requires_grad_(true);
    if (this->subdiv_meta_.defined() &&
        this->subdiv_meta_.dim() == 2 &&
        this->subdiv_meta_.size(0) == old_octpath.size(0)) {
        auto kept_meta =
            this->subdiv_meta_.to(this->subdiv_p_.device())
                .to(torch::kFloat32)
                .index_select(0, kept_idx.to(this->subdiv_p_.device()))
                .contiguous();
        auto child_meta =
            this->subdiv_meta_.to(this->subdiv_p_.device())
                .to(torch::kFloat32)
                .index_select(0, subdiv_idx.to(this->subdiv_p_.device()))
                .repeat_interleave(8, 0)
                .contiguous();
        this->subdiv_meta_ = torch::cat({kept_meta, child_meta}, 0).contiguous();
    } else {
        this->subdiv_meta_ = torch::zeros_like(this->subdiv_p_);
    }
    if (old_subdiv_grad.defined()) {
        this->subdiv_p_.mutable_grad() = torch::cat(
            {old_subdiv_grad.index_select(0, kept_idx.to(old_subdiv_grad.device())).contiguous(),
             subdiv_children.to(old_subdiv_grad.device()).contiguous()}, 0).contiguous();
    }

    auto sh0_children =
        this->sh0_.detach()
            .index_select(0, subdiv_idx.to(this->sh0_.device()))
            .repeat_interleave(8, 0)
            .contiguous();
    this->sh0_ = torch::cat(
        {this->sh0_.detach().index_select(0, kept_idx.to(this->sh0_.device())).contiguous(),
         sh0_children}, 0).contiguous().requires_grad_(true);

    auto shs_children =
        this->shs_.detach()
            .index_select(0, subdiv_idx.to(this->shs_.device()))
            .repeat_interleave(8, 0)
            .contiguous();
    this->shs_ = torch::cat(
        {this->shs_.detach().index_select(0, kept_idx.to(this->shs_.device())).contiguous(),
         shs_children}, 0).contiguous().requires_grad_(true);

    auto [new_center, new_size] = decodeOctpath(
        this->oct_path_.contiguous(),
        this->oct_level_.contiguous(),
        this->scene_center_.contiguous(),
        this->scene_extent_.contiguous());
    auto [new_grid_pts_key, new_vox_key] =
        buildGridPtsLink(this->oct_path_.contiguous(), this->oct_level_.contiguous(), max_num_levels_);
    this->center_ = new_center.contiguous();
    this->size_ = new_size.squeeze(1).contiguous();
    this->vox_size_inv_ = 1.0f / this->size_;
    this->grid_pts_key_ = new_grid_pts_key.contiguous();
    this->vox_key_ = new_vox_key.contiguous();

    auto kept_vox_val =
        old_vox_grid_pts_val.index_select(0, kept_idx.to(old_vox_grid_pts_val.device())).contiguous();
    auto subdiv_vox_val =
        subdivideVoxelCornerValues(
            old_vox_grid_pts_val.index_select(
                0, subdiv_idx.to(old_vox_grid_pts_val.device())).contiguous());
    auto new_vox_val = torch::cat({kept_vox_val, subdiv_vox_val}, 0).contiguous();
    this->_geo_grid_pts_ =
        aggregateVoxelCornersIntoGridPts(
            this->grid_pts_key_.size(0),
            this->vox_key_,
            new_vox_val).detach().contiguous().requires_grad_(true);
    {
        auto kept_sdf =
            old_vox_sdf_values.index_select(0, kept_idx.to(old_vox_sdf_values.device())).contiguous();
        auto subdiv_sdf =
            subdivideVoxelCornerValues(
                old_vox_sdf_values.index_select(
                    0, subdiv_idx.to(old_vox_sdf_values.device())).contiguous());
        auto new_sdf = torch::cat({kept_sdf, subdiv_sdf}, 0).contiguous();

        auto kept_weights =
            old_vox_sdf_weights.index_select(0, kept_idx.to(old_vox_sdf_weights.device())).contiguous();
        auto subdiv_weights =
            subdivideVoxelCornerValues(
                old_vox_sdf_weights.index_select(
                    0, subdiv_idx.to(old_vox_sdf_weights.device())).contiguous());
        auto new_weights = torch::cat({kept_weights, subdiv_weights}, 0).contiguous();
        rebuildFusedSdfFieldFromVoxelCorners_(new_sdf, new_weights);
    }
    this->max_w_ = torch::zeros(
        {center_.size(0), 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

    auto kept_leaf = is_leaf_.index_select(0, kept_idx.to(is_leaf_.device())).contiguous();
    auto kept_parent_local = parent_keep_mask.index_select(
        0, kept_idx.to(parent_keep_mask.device())).view({-1, 1});
    kept_leaf = kept_leaf & (~kept_parent_local);
    auto child_leaf = torch::ones(
        {subdiv_idx.numel() * 8, 1},
        torch::TensorOptions().dtype(torch::kBool).device(is_leaf_.device()));
    is_leaf_ = torch::cat({kept_leaf, child_leaf}, 0).contiguous();

    auto remap_child_bool = [&](torch::Tensor& t) {
        auto kept = t.to(mask.device()).to(torch::kBool)
                        .index_select(0, kept_idx.to(mask.device()))
                        .contiguous();
        auto child = t.to(mask.device()).to(torch::kBool)
                         .index_select(0, subdiv_idx.to(mask.device()))
                         .repeat_interleave(8, 0)
                         .contiguous();
        t = torch::cat({kept, child}, 0).to(device_type_).to(torch::kBool).contiguous();
    };
    remap_child_bool(is_orb_voxel_);
    remap_child_bool(is_inactive_geo_voxel_);
    remap_child_bool(is_rgbd_fill_render_holes_voxel_);
    remap_child_bool(is_monocular_rendered_depth_voxel_);
    remap_child_bool(is_monocular_mvs_voxel_);

    auto kept_exist_iter =
        exist_since_before.to(mask.device()).index_select(0, kept_idx.to(mask.device())).contiguous();
    auto child_exist_iter =
        exist_since_before.to(mask.device()).index_select(0, subdiv_idx.to(mask.device()))
            .repeat_interleave(8, 0).contiguous();
    if (topology_birth_iter_ >= 0) {
        child_exist_iter = torch::full_like(child_exist_iter, topology_birth_iter_);
    }
    exist_since_iter_ =
        torch::cat({kept_exist_iter, child_exist_iter}, 0)
            .to(device_type_).to(torch::kInt32).contiguous();

    auto kept_exist_kf =
        exist_since_kf_before.to(mask.device()).index_select(0, kept_idx.to(mask.device())).contiguous();
    auto child_exist_kf =
        exist_since_kf_before.to(mask.device()).index_select(0, subdiv_idx.to(mask.device()))
            .repeat_interleave(8, 0).contiguous();
    // Subdivision refines an existing observation; it is not a newly
    // inserted primitive. Preserve the parent's keyframe age so online
    // co-visibility pruning does not reclassify the entire refined map as
    // recent and prune it using only the current camera window.
    exist_since_kf_ =
        torch::cat({kept_exist_kf, child_exist_kf}, 0)
            .to(device_type_).to(torch::kInt32).contiguous();
    VOXEL_MODEL_TENSORS_TO_VEC
}

torch::Tensor VoxelModel::subdivisionPriority() const {
    torch::Tensor p = this->subdiv_meta_;
    if (!p.defined() || p.numel() == 0) {
        return torch::Tensor();
    }
    if (p.dim() == 2 && p.size(1) == 1) p = p.squeeze(1);
    return p.contiguous();
}

void VoxelModel::accumulateSubdivisionPriority() {
    // Accumulates the dummy subdivision parameter gradient used for ranking.
    // SVRecon reference: train.py, `subdiv_meta += _subdiv_p.grad`.
    torch::Tensor g = this->subdiv_p_.grad();
    if (!g.defined() || !this->subdiv_p_.defined() || this->subdiv_p_.numel() == 0) {
        return;
    }
    if (!this->subdiv_meta_.defined() || this->subdiv_meta_.sizes() != this->subdiv_p_.sizes()) {
        this->subdiv_meta_ = torch::zeros_like(this->subdiv_p_);
    }
    this->subdiv_meta_ =
        (this->subdiv_meta_.to(g.device()).to(torch::kFloat32) +
         g.detach().to(torch::kFloat32))
            .contiguous();
}

void VoxelModel::resetSubdivisionPriority() {
    // Clears the accumulated ranking statistic after an adaptive update.
    if (this->subdiv_meta_.defined()) {
        this->subdiv_meta_.zero_();
    }
    this->subdiv_p_.mutable_grad() = torch::Tensor();
}

void VoxelModel::freezeVoxGeo() {
    // Pre-gathers shared corners per voxel for repeated no-gradient rendering.
    // SVRecon reference: SVRenderer.freeze_vox_geo.
    const int64_t N = center_.size(0);
    auto care_idx = torch::arange(N, torch::dtype(torch::kLong).device(device_type_));
    torch::NoGradGuard no_grad;
    frozen_vox_geo_ = gatherSvreconGeoParams(vox_key_, care_idx, _geo_grid_pts_)[0].contiguous();
    _geo_grid_pts_.set_requires_grad(false);
}

void VoxelModel::unfreezeVoxGeo() {
    // Releases the gathered cache and restores geometry gradients.
    // SVRecon reference: SVRenderer.unfreeze_vox_geo.
    frozen_vox_geo_.reset();              // make it undefined
    _geo_grid_pts_.set_requires_grad(true);
}

} // namespace sv
