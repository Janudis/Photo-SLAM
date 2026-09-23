#include "src_voxel/voxel_mapper_internal.h"

void VoxelMapper::trainForOneIteration()
{
    // 1) bump global iteration counter
    increaseIteration(1);
    auto iter_start_timing = std::chrono::steady_clock::now();

    sv::RenderOpts ropts;

    // 2) pick a random keyframe from the sliding window
    std::shared_ptr<VoxelKeyframe> viewpoint_cam = useOneRandomSlidingWindowKeyframe();
    if (!viewpoint_cam) {
        increaseIteration(-1);
        return;
    }
    const int iter = getIteration();
    int training_level = num_gaus_pyramid_sub_levels_;
    int image_height, image_width;
    torch::Tensor gt_image, mask;

    if (isdoingGausPyramidTraining())
         training_level = viewpoint_cam->getCurrentGausPyramidLevel();
    if (training_level == num_gaus_pyramid_sub_levels_) {
        image_height = viewpoint_cam->image_height_;
        image_width = viewpoint_cam->image_width_;
        gt_image = viewpoint_cam->original_image_
                            .to(mDevice);          // (3,H,W)
        mask = undistort_mask_[viewpoint_cam->camera_id_]
                                    .to(mDevice)
                                    .to(torch::kFloat32); // (3,H,W)
    }
    else {
        image_height = viewpoint_cam->gaus_pyramid_height_[training_level];
        image_width = viewpoint_cam->gaus_pyramid_width_[training_level];
        gt_image = viewpoint_cam->gaus_pyramid_original_image_[training_level].to(mDevice);
        mask = scene_->cameras_.at(viewpoint_cam->camera_id_).gaus_pyramid_undistort_mask_[training_level].to(mDevice).to(torch::kFloat32);
    }
    // 4) grow SH degree every 1000 iterations (locked during render)
    std::unique_lock<std::mutex> lock_render(mutex_render_);
    if (getIteration() % 1000 == 0 && default_sh_ < model_params_.sh_degree_)
    {
        default_sh_ += 1;
    }
    voxel_model_->setShDegree(default_sh_);

    // Keep the global SDF sharpness fixed at the value initialized from the
    // base voxel size. A local subdivision must not sharpen every remaining
    // coarse cell in the mixed-resolution online map.

    // Match SVRecon train.py: keep ss=1.0 early, then use augmentation
    // or remove ss so the model default is used.
    ropts.ss = 1.0f;
    if (iter > 1000) {
        if (opt_params_.ss_aug_max_ > 1.0f) {
            static thread_local std::mt19937 rng{std::random_device{}()};
            std::uniform_real_distribution<float> dist(1.0f, opt_params_.ss_aug_max_);
            ropts.ss = dist(rng);
        } else {
            ropts.ss = std::nullopt;
        }
    }

    const bool need_sparse_depth = (opt_params_.lambda_sparse_depth_ > 0.0f) && (iter <= opt_params_.sparse_depth_until_);
    const bool need_rgbd_depth =
        (sensor_type_ == RGBD) &&
        (opt_params_.lambda_rgbd_depth_ > 0.0f) &&
        (iter >= opt_params_.rgbd_depth_from_) &&
        (iter <= opt_params_.rgbd_depth_end_);
    const bool need_rgbd_sdf =
        (sensor_type_ == RGBD) &&
        (opt_params_.lambda_rgbd_sdf_ > 0.0f) &&
        (iter >= opt_params_.rgbd_sdf_from_) &&
        (iter <= opt_params_.rgbd_sdf_end_);
    const bool need_rgbd_mask = sensor_type_ == RGBD;
    const bool need_rgbd_normal =
        (sensor_type_ == RGBD) &&
        (opt_params_.lambda_rgbd_normal_ > 0.0f) &&
        (iter >= opt_params_.rgbd_normal_from_) &&
        (iter <= opt_params_.rgbd_normal_end_);
    const bool has_monocular_depth_prior =
        sensor_type_ == MONOCULAR &&
        viewpoint_cam->monocular_depth_source_ !=
            sv::LearnedDepthSource::None &&
        !viewpoint_cam->monocular_depth_prior_.empty() &&
        !viewpoint_cam->monocular_depth_confidence_.empty();
    const bool need_monocular_depth =
        has_monocular_depth_prior &&
        opt_params_.lambda_monocular_depth_ > 0.0f &&
        iter >= opt_params_.monocular_depth_from_ &&
        iter <= opt_params_.monocular_depth_end_;
    const bool need_monocular_normal =
        has_monocular_depth_prior &&
        sv::kLambdaMonocularNormal > 0.0f &&
        iter >= sv::kMonocularNormalFrom &&
        iter <= sv::kMonocularNormalEnd;
    const bool need_T_concen = (opt_params_.lambda_T_concen_ > 0.0f);
    const bool need_T_inside = (opt_params_.lambda_T_inside_ > 0.0f);
    const bool need_normal_dmean =
        (opt_params_.lambda_normal_dmean_ > 0.0f) &&
        (iter >= opt_params_.n_dmean_from_) &&
        (iter <= opt_params_.n_dmean_end_);
    ropts.output_T =
        need_T_concen || need_T_inside || need_sparse_depth || need_normal_dmean ||
        need_rgbd_depth || need_rgbd_normal || need_rgbd_mask ||
        need_monocular_depth || need_monocular_normal;
    ropts.output_depth =
        need_sparse_depth || need_normal_dmean || need_rgbd_depth ||
        need_monocular_depth;
    ropts.output_normal =
        need_normal_dmean || need_rgbd_normal || need_monocular_normal;

    // if (opt_params_.lambda_T_inside_ > 0.0f) {
    //     ropts.output_T = true;
    // }

    if (iter >= opt_params_.dist_from_ && opt_params_.lambda_dist_ > 0.0f) {
        ropts.lambda_dist = opt_params_.lambda_dist_;
    }

    if (iter >= opt_params_.rectifiy_from_ &&
        opt_params_.lambda_rectify_ > 0.0f) {
        ropts.lambda_ascending = -opt_params_.lambda_rectify_;
    } else if (iter >= opt_params_.ascending_from_ &&
               opt_params_.lambda_ascending_ > 0.0f) {
        ropts.lambda_ascending = opt_params_.lambda_ascending_;
    }

    if (iter > opt_params_.scaling_penalty_from_ &&
        iter <= opt_params_.scaling_penalty_end_ &&
        opt_params_.lambda_scaling_penalty_ > 0.0f) {
        auto vox_size = voxel_model_->voxSize();
        if (vox_size.defined() && vox_size.numel() > 0) {
            ropts.lambda_scaling_penalty = opt_params_.lambda_scaling_penalty_;
            ropts.min_voxel_size =
                vox_size.to(torch::kFloat32).min().item<float>();
        }
    }

    if (opt_params_.lambda_R_concen_ > 0.0f) {
        ropts.lambda_R_concen = opt_params_.lambda_R_concen_;
        ropts.gt_color = gt_image;
    }

    sv::MiniCam cam = viewpoint_cam->toMiniCam(image_height, image_width);

    std::unordered_map<std::string, torch::Tensor> render_pkg;
    {
        render_pkg = voxel_model_->render(
            cam,
            image_height,
            image_width,
            /* gt_image   */  gt_image,
            /* color_mode   */   nullptr,
            /* track_max_w   */  false,
            /* ss            */  ropts.ss,
            /* output_depth  */  ropts.output_depth,
            /* output_normal */  ropts.output_normal,
            /* output_T      */  ropts.output_T,
            /* rand_bg       */  false,
            /* use_auto_exp  */  false,
            ropts               // your struct (will be used for **other_opt-safe fields)
        );
    }
    if (render_pkg.empty() || !render_pkg.count("color") || !render_pkg.at("color").defined()) {
        return;
    }

    torch::Tensor rendered_image = render_pkg["color"].to(mDevice);
    torch::Tensor masked_image = rendered_image * mask;      // (1,3,H,W)

    // after render_pkg & rendered_image
    torch::Tensor depth_for_viz;   // declare here so it's visible later
    auto it_depth = render_pkg.find("depth");
    if (it_depth != render_pkg.end() && it_depth->second.defined()) {
        depth_for_viz = it_depth->second;  // keep on device for now
    }

    auto Ll1 = voxel_eval::l1Loss(masked_image, gt_image);
    auto mse = voxel_eval::mseLoss(masked_image, gt_image);

    // Match SVRecon's base photometric loss selection: L1, Huber, or MSE.
    torch::Tensor photo_loss;
    const char* photo_loss_name = nullptr;
    if (opt_params_.use_l1_) {
        photo_loss = Ll1;
        photo_loss_name = "L1";
    } else if (opt_params_.use_huber_) {
        photo_loss = voxel_eval::huberLoss(masked_image, gt_image, opt_params_.huber_thres_);
        photo_loss_name = "Huber";
    } else {
        photo_loss = mse;
        photo_loss_name = "MSE";
    }
    auto loss = photo_loss.clone();

    // --- Optional sparse/RGB-D depth regularization -----------------------------
    if (need_sparse_depth) {
        torch::Tensor depth_loss =
            computeSparseDepthLoss_Points(
            viewpoint_cam,   // which KF we are training on
            cam,             // MiniCam for this KF at current pyramid level
            image_width,
            image_height,
            render_pkg,
            iter);

        loss = loss + opt_params_.lambda_sparse_depth_ * depth_loss;
    }
    if (need_rgbd_depth) {
        torch::Tensor rgbd_depth_loss =
            computeRgbdDepthLoss(viewpoint_cam, cam, render_pkg, iter);

        loss = loss + opt_params_.lambda_rgbd_depth_ * rgbd_depth_loss;
    }
    if (need_rgbd_mask) {
        // SVRecon's lambda_mask supervises final transmittance using its
        // foreground mask. For RGB-D SLAM, valid measured depth is the
        // corresponding foreground observation.
        constexpr float kRgbdForegroundMaskWeight = 0.1f;
        torch::Tensor rgbd_mask_loss =
            computeRgbdMaskLoss(viewpoint_cam, cam, render_pkg);
        loss = loss + kRgbdForegroundMaskWeight * rgbd_mask_loss;
    }
    if (need_rgbd_sdf) {
        torch::Tensor rgbd_sdf_loss =
            computeRgbdSdfLoss(viewpoint_cam, cam, iter);

        loss = loss + opt_params_.lambda_rgbd_sdf_ * rgbd_sdf_loss;
    }
    if (need_rgbd_normal) {
        torch::Tensor rgbd_normal_loss =
            computeRgbdNormalLoss(viewpoint_cam, cam, render_pkg, iter);

        loss = loss + opt_params_.lambda_rgbd_normal_ * rgbd_normal_loss;
    }
    torch::Tensor monocular_depth_loss;
    if (need_monocular_depth) {
        monocular_depth_loss = computeMonocularDepthLoss(
            viewpoint_cam, render_pkg, iter);
        loss = loss +
            opt_params_.lambda_monocular_depth_ * monocular_depth_loss;
    }
    torch::Tensor monocular_normal_loss;
    if (need_monocular_normal) {
        monocular_normal_loss = computeMonocularNormalLoss(
            viewpoint_cam, render_pkg, iter);
        loss = loss +
            sv::kLambdaMonocularNormal * monocular_normal_loss;
    }
    torch::Tensor ssim_loss;
    if (opt_params_.lambda_ssim_ > 0.0f) {
        ssim_loss = voxel_eval::fastSsimLoss(masked_image, gt_image);
        loss += opt_params_.lambda_ssim_ * ssim_loss;
    }

    if (need_T_concen || need_T_inside) {
        auto it = render_pkg.find("raw_T");
        if (it != render_pkg.end() && it->second.defined()) {
            torch::Tensor raw_T = it->second;

            // SVRecon: loss += lambda_T_concen * prob_concen_loss(raw_T)
            if (need_T_concen) {
                torch::Tensor reg_concen = voxel_eval::probabilityConcentrationLoss(raw_T);
                loss = loss + opt_params_.lambda_T_concen_ * reg_concen;
            }

            // SVRecon: loss += lambda_T_inside * raw_T.square().mean()
            if (need_T_inside) {
                torch::Tensor reg_inside = raw_T.pow(2).mean();
                loss = loss + opt_params_.lambda_T_inside_ * reg_inside;
            }
        }
    }

    if (need_normal_dmean) {
        auto reg_normal_dmean = voxel_eval::normalDepthConsistencyLossSvrecon(
            cam,
            render_pkg,
            opt_params_.n_dmean_ks_,
            opt_params_.n_dmean_tol_deg_);
        loss = loss + opt_params_.lambda_normal_dmean_ * reg_normal_dmean;
    }

    // SVRecon applies local Eikonal regularization after the hierarchy reaches
    // inside level 9. The paper uses 1e-11 here versus 1e-8 for the coarse
    // global Eikonal term.
    const bool eikonal_enabled =
        !tail_refinement_active_ && opt_params_.lambda_ge_density_ > 0.f;
    if (eikonal_enabled) {
        loss = loss + voxel_model_->svreconLocalEikonalLoss(
            opt_params_.lambda_ge_density_ * 1.0e-3f,
            /*min_inside_level=*/9);
    }

    {
        voxel_model_->optimizerZeroGrad();
        loss.backward();

        if (opt_params_.lambda_tv_density_ > 0.f &&
            iter >= opt_params_.tv_from_ &&
            iter <= opt_params_.tv_until_) {
            voxel_model_->applyTvOnDensityField(opt_params_.lambda_tv_density_);
        }
        if (eikonal_enabled &&
            iter >= opt_params_.ge_from_ &&
            iter <= opt_params_.ge_until_) {
            const float ge_mult = std::pow(0.25f, std::min(iter / 2000, 2));
            voxel_model_->applySvreconGridEikonalField(
                opt_params_.lambda_ge_density_ * ge_mult);
        }
        if (opt_params_.lambda_ls_density_ > 0.f &&
            iter >= opt_params_.ls_from_ &&
            iter <= opt_params_.ls_until_) {
            const float ls_mult = std::pow(0.25f, std::min(iter / 2000, 2));
            voxel_model_->applySvreconLaplacianSmoothnessField(
                opt_params_.lambda_ls_density_ * ls_mult);
        }

        if (iter >= 500) {
            voxel_model_->accumulateSubdivisionPriority();
        }

        voxel_model_->optimizerStep();
    }

    runPendingSurfaceViewPruning();

    // This is a live diagnostic of the learned SDF, independent of the pruning
    // schedule. Recompute it after every optimizer step so its Rerun timeline
    // shows cells entering and leaving the current SVRecon prune set.
    if (rerun_params_.enable_rerun_ &&
        rerun_params_.rerun_svrecon_debug_) {
        torch::NoGradGuard no_grad;
        logSvreconDebugVoxelMaskToRerun(
            iter,
            computeSvreconSdfPruneMask(),
            "world/svrecon/sdf_prune_candidates");
    }

    adaptVoxelTopology(iter);
    // Update learning rate
    voxel_model_->schedulerStep();

    if (rerun_params_.enable_rerun_) {
        logReconstructionMeshToRerun(iter);
        const int svrecon_grid_snapshot_interval = std::max(
            1,
            std::max(training_report_interval_, opt_params_.adapt_every_));
        const bool periodic_svrecon_snapshot =
            (rerun_params_.rerun_svrecon_debug_ ||
             rerun_params_.rerun_monocular_debug_) &&
            (iter % svrecon_grid_snapshot_interval) == 0;
        const bool log_whole_run_live =
            rerun_params_.run_whole_run_ &&
            rerun_state_.whole_run_live_voxels_dirty_;
        const bool log_svrecon_debug_live =
            rerun_params_.rerun_svrecon_debug_ &&
            (!rerun_state_.svrecon_debug_has_source_snapshot_ ||
             rerun_state_.whole_run_live_voxels_dirty_ ||
             periodic_svrecon_snapshot);
        const bool log_monocular_debug_live =
            rerun_params_.rerun_monocular_debug_ &&
            (!rerun_state_.monocular_debug_has_source_snapshot_ ||
             rerun_state_.whole_run_live_voxels_dirty_ ||
             periodic_svrecon_snapshot);
        if (log_whole_run_live || log_svrecon_debug_live ||
            log_monocular_debug_live) {
            logWholeRunLiveVoxelsToRerun(
                iter,
                voxel_model_->voxCenter(),
                voxel_model_->voxSize(),
                torch::Tensor(),
                log_whole_run_live,
                log_svrecon_debug_live,
                log_monocular_debug_live);
            if (log_svrecon_debug_live) {
                rerun_state_.svrecon_debug_has_source_snapshot_ = true;
            }
            if (log_monocular_debug_live) {
                rerun_state_.monocular_debug_has_source_snapshot_ = true;
            }
            rerun_state_.whole_run_live_voxels_dirty_ = false;
        }
    }

    if (mDevice == torch::kCUDA) torch::cuda::synchronize();

    {
        torch::NoGradGuard no_grad;
        ema_loss_for_log_ = 0.4f * loss.item<float>() + 0.6f * ema_loss_for_log_;

        auto iter_end_timing = std::chrono::steady_clock::now();
        auto iter_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        iter_end_timing - iter_start_timing).count();

         // Log and save
         if (training_report_interval_ && (getIteration() % training_report_interval_ == 0))
             sv::VoxelTrainer::trainingReport(
                 getIteration(),
                 opt_params_.iterations_,
                 photo_loss,
                 photo_loss_name,
                 ssim_loss,
                 monocular_depth_loss,
                 monocular_normal_loss,
                 ema_loss_for_log_,
                 iter_time,
                 *voxel_model_,
                 *scene_,
                 pipe_params_,
                 background_
             );

        if (loop_closure_iteration_)
            loop_closure_iteration_ = false;

    }
}
