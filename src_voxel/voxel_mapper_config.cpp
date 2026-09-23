#include "src_voxel/voxel_mapper_internal.h"

void VoxelMapper::readConfigFromFile(const std::filesystem::path& cfg_path)
{
    cv::FileStorage settings_file(cfg_path.string(), cv::FileStorage::READ);
    if (!settings_file.isOpened()) {
        std::cerr << "[VoxelMapper] Failed to open cfg: " << cfg_path << '\n';
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[VoxelMapper] Reading parameters from " << cfg_path << '\n';
    std::unique_lock<std::mutex> lock(mutex_settings_);

    // Model parameters
     model_params_.sh_degree_ =
         settings_file["Model.sh_degree"].operator int();
     model_params_.resolution_ =
         settings_file["Model.resolution"].operator float();
     model_params_.white_background_ =
         (settings_file["Model.white_background"].operator int()) != 0;
     model_params_.eval_ =
         (settings_file["Model.eval"].operator int()) != 0;

    /* ───────── PIPELINE FLAGS ───────── */
    z_near_ =
         settings_file["Camera.z_near"].operator float();
    if (!settings_file["Camera.z_far"].empty()) {
        z_far_ = std::max(
            z_near_,
            settings_file["Camera.z_far"].operator float());
    }
    cull_keyframes_ =
        (settings_file["Mapper.cull_keyframes"].operator int()) != 0;
    if (!settings_file["Mapper.keyframe_images_on_cpu"].empty()) {
        keyframe_images_on_cpu_ =
            (settings_file["Mapper.keyframe_images_on_cpu"].operator int()) != 0;
    }
    if (keyframe_images_on_cpu_) {
        std::cout << "[VoxelMapper] Caching keyframe RGB images and pyramids on CPU; "
                     "training images are transferred on demand.\n";
    }
    if (!settings_file["Mapper.input_queue_max_keyframes"].empty()) {
        input_queue_max_keyframes_ = std::max(
            0,
            settings_file["Mapper.input_queue_max_keyframes"].operator int());
    }
    if (!settings_file["Mapper.incremental_mapping_window_size"].empty()) {
        incremental_mapping_window_size_ = std::max(
            0,
            settings_file["Mapper.incremental_mapping_window_size"].operator int());
    }
    if (!settings_file["Mapper.loop_closure_reinsert_points"].empty()) {
        loop_closure_reinsert_points_ =
            (settings_file["Mapper.loop_closure_reinsert_points"].operator int()) != 0;
    }
    min_num_initial_map_kfs_ =
        static_cast<std::size_t>(settings_file["Mapper.min_num_initial_map_kfs"].operator int());
    new_keyframe_times_of_use_ =
        settings_file["Mapper.new_keyframe_times_of_use"].operator int();
    large_rot_th_ =
        settings_file["Mapper.large_rotation_threshold"].operator float();
    large_trans_th_ =
        settings_file["Mapper.large_translation_threshold"].operator float();
    local_BA_increased_times_of_use_ =
         settings_file["Mapper.local_BA_increased_times_of_use"].operator int();
    loop_closure_increased_times_of_use_ =
         settings_file["Mapper.loop_closure_increased_times_of_use_"].operator int();

    // Monocular configurations use the generic camera clipping range and do
    // not need RGB-D-specific entries. RGB-D configurations retain their
    // explicit sensor limits.
    RGBD_min_depth_ = z_near_;
    RGBD_max_depth_ = z_far_;
    if (!settings_file["RGBD.min_depth"].empty()) {
        RGBD_min_depth_ =
            settings_file["RGBD.min_depth"].operator float();
    }
    if (!settings_file["RGBD.max_depth"].empty()) {
        RGBD_max_depth_ =
            settings_file["RGBD.max_depth"].operator float();
    }

    inactive_geo_densify_ =
        (settings_file["Mapper.inactive_geo_densify"].operator int()) != 0;
    if (!settings_file["Mapper.monocular_mvs_densify"].empty()) {
        monocular_mvs_densify_ =
            (settings_file["Mapper.monocular_mvs_densify"].operator int()) != 0;
    }
    if (!settings_file["Mapper.monocular_mvs_model_dir"].empty()) {
        monocular_mvs_model_dir_ =
            settings_file["Mapper.monocular_mvs_model_dir"].operator std::string();
    }
    if (!settings_file["Mapper.monocular_mvs_depth_min_m"].empty()) {
        monocular_mvs_depth_min_m_ = std::max(
            1.0e-4f,
            settings_file["Mapper.monocular_mvs_depth_min_m"].operator float());
    }
    if (!settings_file["Mapper.monocular_mvs_depth_max_m"].empty()) {
        monocular_mvs_depth_max_m_ = std::max(
            monocular_mvs_depth_min_m_ + 1.0e-4f,
            settings_file["Mapper.monocular_mvs_depth_max_m"].operator float());
    }
    if (!settings_file["Mapper.monocular_mvs_empty_cache_before_launch"].empty()) {
        monocular_mvs_empty_cache_before_launch_ =
            settings_file["Mapper.monocular_mvs_empty_cache_before_launch"]
                .operator int() != 0;
    }
    if (!settings_file["Mapper.monocular_mvs_tsdf_evidence"].empty()) {
        monocular_mvs_tsdf_evidence_ =
            settings_file["Mapper.monocular_mvs_tsdf_evidence"]
                .operator int() != 0;
    }
    if (isMonocularMvsPipelineEnabled()) {
        if (monocular_mvs_model_dir_.empty()) {
            throw std::runtime_error(
                "Mapper.monocular_mvs_model_dir is required when "
                "TANDEM MVS densification or TSDF evidence is enabled");
        }
    }
    if (!settings_file["Mapper.allocate_orb_voxels"].empty()) {
        allocate_orb_voxels_ =
            (settings_file["Mapper.allocate_orb_voxels"].operator int()) != 0;
    }
    if (!settings_file["Model.outside_level"].empty()) {
        svrecon_outside_level_ =
            std::max(0, settings_file["Model.outside_level"].operator int());
    }
    if (!settings_file["Model.global_scene_extent"].empty()) {
        global_scene_extent_m_ = std::max(
            0.0f,
            settings_file["Model.global_scene_extent"].operator float());
    }
    if (!settings_file["Model.robust_scene_bounds"].empty()) {
        robust_scene_bounds_ =
            (settings_file["Model.robust_scene_bounds"].operator int()) != 0;
    }
    if (!settings_file["Mapper.sdf_initialization_rgbd_projective"].empty()) {
        sdf_initialization_rgbd_projective_ =
            (settings_file["Mapper.sdf_initialization_rgbd_projective"].operator int()) != 0;
    }
    if (!settings_file["Mapper.sdf_initialization_mode"].empty()) {
        sdf_initialization_mode_ = voxel_utils::toLowerCopy(
            settings_file["Mapper.sdf_initialization_mode"].operator std::string());
    }
    if (sdf_initialization_mode_ != "orb_prior" &&
        sdf_initialization_mode_ != "weak_positive" &&
        sdf_initialization_mode_ != "source_points" &&
        sdf_initialization_mode_ != "weak_surface_prior") {
        throw std::runtime_error(
            "[VoxelMapper] Mapper.sdf_initialization_mode must be one of: "
            "orb_prior, weak_positive, source_points, weak_surface_prior");
    }
    max_depth_cached_ =
        settings_file["Mapper.depth_cache"].operator int();
    if (!settings_file["Mapper.rgbd_fill_render_holes_initial_backfill"].empty()) {
        rgbd_fill_render_holes_initial_backfill_ =
            (settings_file["Mapper.rgbd_fill_render_holes_initial_backfill"].operator int()) != 0;
    }
    if (!settings_file["Mapper.rgbd_fill_render_holes"].empty()) {
        rgbd_fill_render_holes_ =
            (settings_file["Mapper.rgbd_fill_render_holes"].operator int()) != 0;
    }
    if (!settings_file["Mapper.rgbd_fill_render_holes_projective_sdf"].empty()) {
        rgbd_fill_render_holes_projective_sdf_ =
            (settings_file["Mapper.rgbd_fill_render_holes_projective_sdf"].operator int()) != 0;
    }
    if (!settings_file["Mapper.rgbd_fill_render_holes_stride"].empty()) {
        rgbd_fill_render_holes_stride_ =
            std::max(1, settings_file["Mapper.rgbd_fill_render_holes_stride"].operator int());
    }
    if (!settings_file["Mapper.rgbd_tsdf_evidence"].empty()) {
        rgbd_tsdf_evidence_ =
            (settings_file["Mapper.rgbd_tsdf_evidence"].operator int()) != 0;
    }
    if (!settings_file["Mapper.rgbd_tsdf_evidence_initial_backfill"].empty()) {
        rgbd_tsdf_evidence_initial_backfill_ =
            (settings_file["Mapper.rgbd_tsdf_evidence_initial_backfill"].operator int()) != 0;
    }
    if (!settings_file["Mapper.rgbd_tsdf_evidence_pixel_stride"].empty()) {
        rgbd_tsdf_evidence_pixel_stride_ = std::max(
            1,
            settings_file["Mapper.rgbd_tsdf_evidence_pixel_stride"].operator int());
    }
    if (!settings_file["Mapper.rgbd_tsdf_evidence_trunc_vox"].empty()) {
        rgbd_tsdf_evidence_trunc_vox_ = std::max(
            0.5f,
            settings_file["Mapper.rgbd_tsdf_evidence_trunc_vox"].operator float());
    }
    if (!settings_file["Mapper.rgbd_tsdf_evidence_max_weight"].empty()) {
        rgbd_tsdf_evidence_max_weight_ = std::max(
            1.0e-4f,
            settings_file["Mapper.rgbd_tsdf_evidence_max_weight"].operator float());
    }
    if (!settings_file["Mapper.rgbd_tsdf_evidence_promote_min_views"].empty()) {
        rgbd_tsdf_evidence_promote_min_views_ = std::max(
            1,
            settings_file["Mapper.rgbd_tsdf_evidence_promote_min_views"].operator int());
    }
    if (rgbd_tsdf_evidence_) {
        if (rgbd_fill_render_holes_ || sdf_initialization_rgbd_projective_) {
            std::cout
                << "[VoxelMapper] Mapper.rgbd_tsdf_evidence uses ORB, optional "
                   "inactive geometry, and evidence-only residual-hole fusion; "
                   "disabling direct RGB-D hole filling and global projective "
                   "SDF initialization.\n";
        }
        rgbd_fill_render_holes_ = false;
        sdf_initialization_rgbd_projective_ = false;
    }
    if (!settings_file["Mapper.sdf_voxel_size_m"].empty()) {
        sdf_params_.sdf_voxel_size_m_ =
            std::max(1.0e-4f, settings_file["Mapper.sdf_voxel_size_m"].operator float());
    }
    if (!settings_file["Mapper.sdf_init_trunc_vox"].empty()) {
        sdf_params_.sdf_init_trunc_vox_ =
            std::max(1.0e-3f, settings_file["Mapper.sdf_init_trunc_vox"].operator float());
    }
    if (!settings_file["Mapper.sdf_init_max_depth_m"].empty()) {
        sdf_params_.sdf_init_max_depth_m_ =
            std::max(0.0f, settings_file["Mapper.sdf_init_max_depth_m"].operator float());
    }

    pipe_params_.convert_SHs_ =
         (settings_file["Pipeline.convert_SHs"].operator int()) != 0;

    do_gaus_pyramid_training_ =
         (settings_file["GausPyramid.do"].operator int()) != 0;
    num_gaus_pyramid_sub_levels_ =
        settings_file["GausPyramid.num_sub_levels"].operator int();
    int sub_level_times_of_use =
        settings_file["GausPyramid.sub_level_times_of_use"].operator int();
    kf_gaus_pyramid_times_of_use_.resize(num_gaus_pyramid_sub_levels_);
    kf_gaus_pyramid_factors_.resize(num_gaus_pyramid_sub_levels_);
    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        kf_gaus_pyramid_times_of_use_[l] = sub_level_times_of_use;
        kf_gaus_pyramid_factors_[l] = std::pow(0.5f, num_gaus_pyramid_sub_levels_ - l);
    }

    /* ───────── OPTIMIZATION PARAMETERS ───────── */
    opt_params_.iterations_ =
        settings_file["Optimization.max_num_iterations"].operator int();
    opt_params_.geo_lr_ =
        settings_file["Optimization.geo_lr"].operator float();
    opt_params_.sh0_lr_ =
        settings_file["Optimization.sh0_lr"].operator float();
    opt_params_.shs_lr_ =
        settings_file["Optimization.shs_lr"].operator float();
    {
        cv::FileNode n = settings_file["Optimization.lr_decay_ckpt"];
        opt_params_.lr_decay_ckpt_.clear();
        if (!n.empty())
        {
            if (n.type() == cv::FileNode::SEQ) {
                // YAML: Optimization.lr_decay_ckpt: [5000, 10000, 20000]
                for (auto it = n.begin(); it != n.end(); ++it)
                    opt_params_.lr_decay_ckpt_.push_back((int)*it);
            } else if (n.isInt()) {
                // YAML: Optimization.lr_decay_ckpt: 10000
                opt_params_.lr_decay_ckpt_.push_back((int)n);
            } else if (n.isString()) {
                // YAML: Optimization.lr_decay_ckpt: "5000,10000,20000"
                std::string s = (std::string)n;
                std::stringstream ss(s);
                for (std::string tok; std::getline(ss, tok, ','); ) {
                    if (!tok.empty()) opt_params_.lr_decay_ckpt_.push_back(std::stoi(tok));
                }
            }
        }
    }
    opt_params_.optim_beta1_ =
        settings_file["Optimization.optim_beta1"].operator float();
    opt_params_.optim_beta2_ =
        settings_file["Optimization.optim_beta2"].operator float();
    opt_params_.optim_eps_ =
        settings_file["Optimization.optim_eps"].operator float();
    opt_params_.lr_decay_mult_ =
        settings_file["Optimization.lr_decay_mult"].operator float();

    if (!settings_file["Optimization.adapt_from"].empty()) {
        opt_params_.adapt_from_ =
            settings_file["Optimization.adapt_from"].operator int();
    }
    if (!settings_file["Optimization.adapt_every"].empty()) {
        opt_params_.adapt_every_ =
            settings_file["Optimization.adapt_every"].operator int();
    }
    if (!settings_file["Optimization.prune_every"].empty()) {
        opt_params_.prune_every_ =
            settings_file["Optimization.prune_every"].operator int();
    }
    if (!settings_file["Optimization.subdivide_every"].empty()) {
        opt_params_.subdivide_every_ =
            settings_file["Optimization.subdivide_every"].operator int();
    }
    if (!settings_file["Optimization.filter_near_voxels"].empty()) {
        opt_params_.filter_near_voxels_ =
            (settings_file["Optimization.filter_near_voxels"].operator int()) != 0;
    }
    if (!settings_file["Optimization.prune_far_voxels"].empty()) {
        opt_params_.prune_far_voxels_ =
            (settings_file["Optimization.prune_far_voxels"].operator int()) != 0;
    }
    opt_params_.prune_near_voxels_geometric_ =
        !settings_file["Optimization.prune_near_voxels_geometric"].empty() &&
        (settings_file["Optimization.prune_near_voxels_geometric"].operator int()) != 0;
    if (!settings_file["Optimization.prune_surface_views_enable"].empty()) {
        opt_params_.prune_surface_views_enable_ =
            (settings_file["Optimization.prune_surface_views_enable"].operator int()) != 0;
    }
    if (!settings_file["Optimization.prune_mvs_consistency_enable"].empty()) {
        opt_params_.prune_mvs_consistency_enable_ =
            settings_file["Optimization.prune_mvs_consistency_enable"]
                .operator int() != 0;
    }
    if (!settings_file["Optimization.final_refinement_enable"].empty()) {
        opt_params_.final_refinement_enable_ =
            (settings_file["Optimization.final_refinement_enable"].operator int()) != 0;
    }
    opt_params_.prune_from_ = !settings_file["Optimization.prune_from"].empty()
        ? settings_file["Optimization.prune_from"].operator int()
        : opt_params_.adapt_from_;
    opt_params_.prune_until_ =
        settings_file["Optimization.prune_until"].operator int();
    opt_params_.prune_thres_init_ =
        settings_file["Optimization.prune_thres_init"].operator float();
    opt_params_.prune_thres_final_ =
        settings_file["Optimization.prune_thres_final"].operator float();
    opt_params_.prune_thres_final_at_target_ =
        settings_file["Optimization.prune_thres_final_at_target"].operator float();
    opt_params_.subdivide_from_ = !settings_file["Optimization.subdivide_from"].empty()
        ? settings_file["Optimization.subdivide_from"].operator int()
        : opt_params_.adapt_from_;
    opt_params_.subdivide_all_until_ =
        settings_file["Optimization.subdivide_all_until"].operator int();
    opt_params_.subdivide_samp_thres_ =
        settings_file["Optimization.subdivide_samp_thres"].operator float();
    if (!settings_file["Optimization.subdivide_prop"].empty()) {
        opt_params_.subdivide_prop_ = std::clamp(
            settings_file["Optimization.subdivide_prop"].operator float(),
            0.0f,
            1.0f);
    }
    if (!settings_file["Optimization.subdivide_max_num"].empty()) {
        opt_params_.subdivide_max_num_ = std::max(
            1,
            settings_file["Optimization.subdivide_max_num"].operator int());
    }
    opt_params_.use_l1_ =
        (settings_file["Optimization.use_l1"].operator int()) != 0;
    opt_params_.use_huber_ =
        (settings_file["Optimization.use_huber"].operator int()) != 0;
    opt_params_.huber_thres_ =
        settings_file["Optimization.huber_thres"].operator float();
    if (opt_params_.use_l1_ && opt_params_.use_huber_) {
        std::cout << "[VoxelMapper] Both Optimization.use_l1 and Optimization.use_huber are enabled. "
                  << "Prioritizing L1 to match SVRecon." << std::endl;
    }

    opt_params_.lambda_tv_density_ =
        settings_file["Optimization.lambda_tv_density"].operator float();
    opt_params_.tv_from_ =
        settings_file["Optimization.tv_from"].operator int();
    opt_params_.tv_until_ =
        settings_file["Optimization.tv_until"].operator int();
    if (!settings_file["Optimization.lambda_ge_density"].empty()) {
        opt_params_.lambda_ge_density_ =
            std::max(0.0f, settings_file["Optimization.lambda_ge_density"].operator float());
    }
    if (!settings_file["Optimization.ge_from"].empty()) {
        opt_params_.ge_from_ = settings_file["Optimization.ge_from"].operator int();
    }
    if (!settings_file["Optimization.ge_until"].empty()) {
        opt_params_.ge_until_ = settings_file["Optimization.ge_until"].operator int();
    }
    if (!settings_file["Optimization.lambda_ls_density"].empty()) {
        opt_params_.lambda_ls_density_ =
            std::max(0.0f, settings_file["Optimization.lambda_ls_density"].operator float());
    }
    if (!settings_file["Optimization.ls_from"].empty()) {
        opt_params_.ls_from_ = settings_file["Optimization.ls_from"].operator int();
    }
    if (!settings_file["Optimization.ls_until"].empty()) {
        opt_params_.ls_until_ = settings_file["Optimization.ls_until"].operator int();
    }
    opt_params_.ss_aug_max_ = settings_file["Optimization.ss_aug_max"].operator float();
    opt_params_.lambda_R_concen_ = settings_file["Optimization.lambda_R_concen"].operator float();
    opt_params_.lambda_dist_ = settings_file["Optimization.lambda_dist"].operator float();
    if (!settings_file["Optimization.dist_from"].empty()) {
        opt_params_.dist_from_ = settings_file["Optimization.dist_from"].operator int();
    }
    opt_params_.lambda_T_concen_ = settings_file["Optimization.lambda_T_concen"].operator float();
    opt_params_.lambda_T_inside_ = settings_file["Optimization.lambda_T_inside"].operator float();
    opt_params_.lambda_normal_dmean_ = settings_file["Optimization.lambda_normal_dmean"].operator float();
    opt_params_.n_dmean_from_ = settings_file["Optimization.n_dmean_from"].operator int();
    opt_params_.n_dmean_end_ = settings_file["Optimization.n_dmean_end"].operator int();
    opt_params_.n_dmean_ks_ = settings_file["Optimization.n_dmean_ks"].operator int();
    opt_params_.n_dmean_tol_deg_ = settings_file["Optimization.n_dmean_tol_deg"].operator float();
    opt_params_.lambda_ssim_ = settings_file["Optimization.lambda_ssim"].operator float();

    opt_params_.lambda_sparse_depth_ = settings_file["Optimization.lambda_sparse_depth"].operator float();
    opt_params_.sparse_depth_until_ = settings_file["Optimization.sparse_depth_until"].operator int();
    if (!settings_file["Optimization.lambda_rgbd_depth"].empty()) {
        opt_params_.lambda_rgbd_depth_ = settings_file["Optimization.lambda_rgbd_depth"].operator float();
        opt_params_.rgbd_depth_from_ = settings_file["Optimization.rgbd_depth_from"].operator int();
        opt_params_.rgbd_depth_end_ = settings_file["Optimization.rgbd_depth_end"].operator int();
        opt_params_.rgbd_depth_end_mult_ = settings_file["Optimization.rgbd_depth_end_mult"].operator float();
        opt_params_.lambda_rgbd_normal_ = settings_file["Optimization.lambda_rgbd_normal"].operator float();
        opt_params_.rgbd_normal_from_ = settings_file["Optimization.rgbd_normal_from"].operator int();
        opt_params_.rgbd_normal_end_ = settings_file["Optimization.rgbd_normal_end"].operator int();
        opt_params_.rgbd_normal_end_mult_ = settings_file["Optimization.rgbd_normal_end_mult"].operator float();
        opt_params_.rgbd_normal_ks_ = settings_file["Optimization.rgbd_normal_ks"].operator int();
        opt_params_.rgbd_normal_tol_deg_ = settings_file["Optimization.rgbd_normal_tol_deg"].operator float();
    }
    if (!settings_file["Optimization.lambda_monocular_depth"].empty()) {
        opt_params_.lambda_monocular_depth_ = std::max(
            0.0f,
            settings_file["Optimization.lambda_monocular_depth"]
                .operator float());
    }
    if (!settings_file["Optimization.monocular_depth_from"].empty()) {
        opt_params_.monocular_depth_from_ = std::max(
            0,
            settings_file["Optimization.monocular_depth_from"]
                .operator int());
    }
    if (!settings_file["Optimization.monocular_depth_end"].empty()) {
        opt_params_.monocular_depth_end_ = std::max(
            opt_params_.monocular_depth_from_,
            settings_file["Optimization.monocular_depth_end"]
                .operator int());
    }
    if (!settings_file["Optimization.lambda_rgbd_sdf"].empty()) {
        opt_params_.lambda_rgbd_sdf_ = settings_file["Optimization.lambda_rgbd_sdf"].operator float();
    }
    if (!settings_file["Optimization.rgbd_sdf_from"].empty()) {
        opt_params_.rgbd_sdf_from_ = settings_file["Optimization.rgbd_sdf_from"].operator int();
    }
    if (!settings_file["Optimization.rgbd_sdf_end"].empty()) {
        opt_params_.rgbd_sdf_end_ = settings_file["Optimization.rgbd_sdf_end"].operator int();
    }
    if (!settings_file["Optimization.rgbd_sdf_end_mult"].empty()) {
        opt_params_.rgbd_sdf_end_mult_ = settings_file["Optimization.rgbd_sdf_end_mult"].operator float();
    }
    if (!settings_file["Optimization.rgbd_sdf_trunc_vox"].empty()) {
        opt_params_.rgbd_sdf_trunc_vox_ = settings_file["Optimization.rgbd_sdf_trunc_vox"].operator float();
    }
    if (!settings_file["Optimization.rgbd_sdf_max_samples"].empty()) {
        opt_params_.rgbd_sdf_max_samples_ = settings_file["Optimization.rgbd_sdf_max_samples"].operator int();
    }
    if (!settings_file["Optimization.rgbd_sdf_ray_pixels"].empty()) {
        opt_params_.rgbd_sdf_ray_pixels_ = settings_file["Optimization.rgbd_sdf_ray_pixels"].operator int();
    }
    if (!settings_file["Optimization.rgbd_sdf_free_samples"].empty()) {
        opt_params_.rgbd_sdf_free_samples_ = settings_file["Optimization.rgbd_sdf_free_samples"].operator int();
    }
    if (!settings_file["Optimization.rgbd_sdf_surface_samples"].empty()) {
        opt_params_.rgbd_sdf_surface_samples_ = settings_file["Optimization.rgbd_sdf_surface_samples"].operator int();
    }
    if (!settings_file["Optimization.rgbd_sdf_w_fs"].empty()) {
        opt_params_.rgbd_sdf_w_fs_ = settings_file["Optimization.rgbd_sdf_w_fs"].operator float();
    }
    if (!settings_file["Optimization.rgbd_sdf_w_center"].empty()) {
        opt_params_.rgbd_sdf_w_center_ = settings_file["Optimization.rgbd_sdf_w_center"].operator float();
    }
    if (!settings_file["Optimization.rgbd_sdf_w_tail"].empty()) {
        opt_params_.rgbd_sdf_w_tail_ = settings_file["Optimization.rgbd_sdf_w_tail"].operator float();
    }
    /* ───────── LOGGING PARAMETERS ───────── */
    training_report_interval_ =
        settings_file["Record.training_report_interval"].operator int();
    record_loop_ply_ =
        (settings_file["Record.record_loop_ply"].operator int()) != 0;
    rerun_params_.enable_rerun_ =
        (settings_file["Record.enable_rerun"].operator int()) != 0;
    rerun_params_.rerun_max_keyframes_ =
        settings_file["Record.rerun_max_keyframes"].operator int();
    rerun_params_.rerun_keyframe_start_ =
        std::max(0, settings_file["Record.rerun_keyframe_start"].operator int());
    rerun_params_.run_whole_run_ =
        !settings_file["Record.run_whole_run"].empty() &&
        (settings_file["Record.run_whole_run"].operator int()) != 0;
    rerun_params_.rerun_svrecon_debug_ =
        !settings_file["Record.rerun_svrecon_debug"].empty() &&
        (settings_file["Record.rerun_svrecon_debug"].operator int()) != 0;
    rerun_params_.rerun_monocular_debug_ =
        !settings_file["Record.rerun_monocular_debug"].empty() &&
        (settings_file["Record.rerun_monocular_debug"].operator int()) != 0;
    rerun_params_.rerun_gt_mesh_ =
        !settings_file["Record.rerun_gt_mesh"].empty() &&
        (settings_file["Record.rerun_gt_mesh"].operator int()) != 0;
    rerun_params_.rerun_gt_mesh_path_ =
        settings_file["Record.rerun_gt_mesh_path"].empty()
            ? std::string()
            : settings_file["Record.rerun_gt_mesh_path"].operator std::string();
    rerun_params_.save_progressive_rendered_tsdf_mesh_ =
        !settings_file["Record.save_progressive_rendered_tsdf_mesh"].empty() &&
        (settings_file["Record.save_progressive_rendered_tsdf_mesh"].operator int()) != 0;
    rerun_params_.rendered_mesh_voxel_size_m_ =
        settings_file["Record.rendered_mesh_voxel_size_m"].empty()
            ? 0.05f
            : std::max(
                  1.0e-6f,
                  settings_file["Record.rendered_mesh_voxel_size_m"].operator float());
    rerun_params_.rendered_mesh_scale_aware_ =
        !settings_file["Record.rendered_mesh_scale_aware"].empty() &&
        (settings_file["Record.rendered_mesh_scale_aware"].operator int()) != 0;
    rerun_params_.rendered_mesh_max_grid_points_ =
        settings_file["Record.rendered_mesh_max_grid_points"].empty()
            ? 2000000u
            : static_cast<std::size_t>(std::max(
                  0,
                  settings_file["Record.rendered_mesh_max_grid_points"].operator int()));
    rerun_params_.rendered_mesh_min_weight_ =
        settings_file["Record.rendered_mesh_min_weight"].empty()
            ? 2.0f
            : std::max(
                  0.0f,
                  settings_file["Record.rendered_mesh_min_weight"].operator float());
    rerun_params_.rendered_mesh_trunc_vox_ =
        settings_file["Record.rendered_mesh_trunc_vox"].empty()
            ? 8.0f
            : std::max(
                  1.0f,
                  settings_file["Record.rendered_mesh_trunc_vox"].operator float());
    rerun_params_.rendered_mesh_depth_max_m_ =
        settings_file["Record.rendered_mesh_depth_max_m"].empty()
            ? 5.0f
            : std::max(
                  1.0e-6f,
                  settings_file["Record.rendered_mesh_depth_max_m"].operator float());
    rerun_params_.svrecon_mesh_init_lv_ =
        settings_file["Record.svrecon_mesh_init_lv"].empty()
            ? 7
            : std::max(1, settings_file["Record.svrecon_mesh_init_lv"].operator int());
    rerun_params_.svrecon_mesh_final_lv_ =
        settings_file["Record.svrecon_mesh_final_lv"].empty()
            ? 10
            : std::max(1, settings_file["Record.svrecon_mesh_final_lv"].operator int());
    rerun_params_.svrecon_mesh_trunc_lv_ =
        settings_file["Record.svrecon_mesh_trunc_lv"].empty()
            ? 10
            : std::max(1, settings_file["Record.svrecon_mesh_trunc_lv"].operator int());
    rerun_params_.svrecon_mesh_trunc_vox_ =
        settings_file["Record.svrecon_mesh_trunc_vox"].empty()
            ? 5.0f
            : std::max(1.0e-6f, settings_file["Record.svrecon_mesh_trunc_vox"].operator float());
    rerun_params_.svrecon_mesh_pg_prune_ =
        settings_file["Record.svrecon_mesh_pg_prune"].empty()
            ? 0.6f
            : std::max(0.0f, settings_file["Record.svrecon_mesh_pg_prune"].operator float());
    rerun_params_.svrecon_mesh_crop_border_ =
        settings_file["Record.svrecon_mesh_crop_border"].empty()
            ? 0.01f
            : std::clamp(
                  settings_file["Record.svrecon_mesh_crop_border"].operator float(),
                  0.0f,
                  0.99f);
    rerun_params_.svrecon_mesh_alpha_thres_ =
        settings_file["Record.svrecon_mesh_alpha_thres"].empty()
            ? 0.5f
            : std::clamp(
                  settings_file["Record.svrecon_mesh_alpha_thres"].operator float(),
                  0.0f,
                  1.0f);
    rerun_params_.svrecon_mesh_use_mean_depth_ =
        !settings_file["Record.svrecon_mesh_use_mean_depth"].empty() &&
        (settings_file["Record.svrecon_mesh_use_mean_depth"].operator int()) != 0;
    rerun_params_.svrecon_mesh_use_vert_color_ =
        !settings_file["Record.svrecon_mesh_use_vert_color"].empty() &&
        (settings_file["Record.svrecon_mesh_use_vert_color"].operator int()) != 0;
    rerun_params_.rerun_reconstruction_mesh_ =
        !settings_file["Record.run_reconstruction_mesh"].empty() &&
        (settings_file["Record.run_reconstruction_mesh"].operator int()) != 0;
    rerun_params_.rerun_reconstruction_mesh_interval_ =
        settings_file["Record.run_reconstruction_mesh_interval"].empty()
            ? 200
            : std::max(1, settings_file["Record.run_reconstruction_mesh_interval"].operator int());
    rerun_params_.rerun_reconstruction_mesh_min_weight_ =
        settings_file["Record.run_reconstruction_mesh_min_weight"].empty()
            ? 1.0e-4f
            : std::max(0.0f, settings_file["Record.run_reconstruction_mesh_min_weight"].operator float());
    rerun_params_.rerun_reconstruction_mesh_weld_vertices_ =
        settings_file["Record.run_reconstruction_mesh_weld_vertices"].empty()
            ? true
            : (settings_file["Record.run_reconstruction_mesh_weld_vertices"].operator int()) != 0;
    rerun_params_.rerun_reconstruction_mesh_max_vertices_ =
        settings_file["Record.run_reconstruction_mesh_max_vertices"].empty()
            ? static_cast<std::size_t>(250000)
            : static_cast<std::size_t>(
                  std::max(0, settings_file["Record.run_reconstruction_mesh_max_vertices"].operator int()));
    rerun_params_.rerun_reconstruction_mesh_max_faces_ =
        settings_file["Record.run_reconstruction_mesh_max_faces"].empty()
            ? static_cast<std::size_t>(500000)
            : static_cast<std::size_t>(
                  std::max(0, settings_file["Record.run_reconstruction_mesh_max_faces"].operator int()));
    const bool any_rerun_recording_requested =
        rerun_params_.run_whole_run_ ||
        rerun_params_.rerun_svrecon_debug_ ||
        rerun_params_.rerun_monocular_debug_ ||
        rerun_params_.rerun_gt_mesh_ ||
        rerun_params_.rerun_reconstruction_mesh_;
#if !PHOTOSLAM_ENABLE_RERUN
    // Deployment builds exclude Rerun regardless of values in desktop YAMLs.
    rerun_params_.enable_rerun_ = false;
#endif
    rerun_params_.enable_rerun_ =
        rerun_params_.enable_rerun_ && any_rerun_recording_requested;
    if (!rerun_params_.enable_rerun_) {
        rerun_params_.run_whole_run_ = false;
        rerun_params_.rerun_svrecon_debug_ = false;
        rerun_params_.rerun_monocular_debug_ = false;
        rerun_params_.rerun_gt_mesh_ = false;
        rerun_params_.rerun_reconstruction_mesh_ = false;
    }
    // Viewer Parameters
     rendered_image_viewer_scale_ =
         settings_file["VoxelViewer.image_scale"].operator float();
     rendered_image_viewer_scale_main_ =
         settings_file["VoxelViewer.image_scale_main"].operator float();

}

void VoxelMapper::ensureEmbeddedPythonRuntime(bool import_torch_cuda)
{
    ensurePythonRuntimeInitialized(import_torch_cuda);
}
