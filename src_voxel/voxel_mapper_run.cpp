#include "src_voxel/voxel_mapper_internal.h"

void VoxelMapper::run()
{
    sv::RerunVisualizerBridge::instance().setEnabled(rerun_params_.enable_rerun_);

    if (rerun_params_.enable_rerun_ && rerun_params_.rerun_gt_mesh_) {
        if (!rerun_params_.rerun_gt_mesh_path_.empty() &&
            std::filesystem::exists(rerun_params_.rerun_gt_mesh_path_)) {
            sv::RerunVisualizerBridge::instance().visualizePlyMesh(
                rerun_params_.rerun_gt_mesh_path_,
                0,
                "world/gt/mesh");
        } else {
            std::cerr << "[RERUN] GT mesh not found: "
                      << rerun_params_.rerun_gt_mesh_path_ << "\n";
        }
    }

    // First loop: Initial gaussian mapping
    while (!isStopped())
    {
        // Check conditions for initial mapping
        if (hasMetInitialMappingConditions())
        {
            mpSLAM->getAtlas()->clearMappingOperation();

            // Pull sparse SLAM map (get keyframes and map points)
            auto pMap = mpSLAM->getAtlas()->GetCurrentMap();
            std::vector<ORB_SLAM3::KeyFrame*> vKFs;
            std::vector<ORB_SLAM3::MapPoint*> vMPs;
            {
                std::unique_lock<std::mutex> lock_map(pMap->mMutexMapUpdate);
                vKFs = pMap->GetAllKeyFrames();
                vMPs = pMap->GetAllMapPoints();
                const unsigned long current_keyframe_id = pMap->GetMaxKFid();
                if (sensor_type_ == MONOCULAR) {
                    monocular_orb_inserted_point_ids_.clear();
                }
                for (const auto& pMP : vMPs)
                {
                     if (!pMP) {
                         continue;
                     }
                     if (sensor_type_ == MONOCULAR &&
                         !isMatureMonocularOrbMapPoint(
                             pMP, current_keyframe_id)) {
                         continue;
                     }
                     sv::Point3D point3D;
                     auto pos = pMP->GetWorldPos();
                     point3D.xyz_(0) = pos.x();
                     point3D.xyz_(1) = pos.y();
                     point3D.xyz_(2) = pos.z();
                     auto color = pMP->GetColorRGB();
                     point3D.color_(0) = color(0);
                     point3D.color_(1) = color(1);
                     point3D.color_(2) = color(2);
                     scene_->cachePoint3D(pMP->mnId, point3D);
                     if (sensor_type_ == MONOCULAR) {
                         monocular_orb_inserted_point_ids_.insert(pMP->mnId);
                     }
                 }
                // B) Create VoxelKeyframes from each SLAM KeyFrame
                for (const auto& pKF : vKFs)
                {
                    std::shared_ptr<VoxelKeyframe> new_kf = std::make_shared<VoxelKeyframe>(pKF->mnId, getIteration());
                    new_kf->source_timestamp_ = pKF->mTimeStamp;
                    new_kf->source_frame_id_ =
                        voxel_utils::parseFrameIdFromPath(pKF->mNameFile);
                    if (new_kf->source_frame_id_ < 0) {
                        new_kf->source_frame_id_ =
                            voxel_utils::frameIdFromIntegerTimestamp(
                                pKF->mTimeStamp);
                    }
                    new_kf->znear_ = z_near_;
                    // Pose
                    auto pose = pKF->GetPose();
                    new_kf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>()
                    );
                    cv::Mat imgRGB_undistorted, imgAux_undistorted;
                    // Camera
                    sv::Camera& camera = scene_->cameras_.at(pKF->mpCamera->GetId());
                    new_kf->setCameraParams(camera);

                    cv::Mat imgRGB = pKF->imgLeftRGB;
                    camera.undistortImage(imgRGB, imgRGB_undistorted);
                    // Auxiliary Image
                    cv::Mat imgAux = pKF->imgAuxiliary;
                    if (this->sensor_type_ == RGBD) {
                        imgAux_undistorted = mapperDepthForKeyframe(
                            pKF->mNameFile, imgAux, camera);
                    } else {
                        imgAux_undistorted = imgAux;
                    }

                    new_kf->original_image_ =
                        voxel_utils::cvMatToTorchTensorFloat32(imgRGB_undistorted,
                            keyframe_images_on_cpu_ ? torch::kCPU : device_type_);
                    new_kf->img_filename_ = pKF->mNameFile;
                    new_kf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
                    new_kf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
                    new_kf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;

                    // Compute transformations
                    // new_kf->computeTransformTensors(); //useless
                    scene_->addKeyframe(new_kf, &kfid_shuffled_);
                    latest_consumed_keyframe_id_.store(
                        std::max(
                            latest_consumed_keyframe_id_.load(std::memory_order_relaxed),
                            static_cast<long long>(pKF->mnId)),
                        std::memory_order_release);

                    increaseKeyframeTimesOfUse(new_kf, newKeyframeTimesOfUse());

                    // // Features for increasePcdByKeyframeInactiveGeoDensify
                    std::vector<float> pixels;
                    std::vector<float> pointsLocal;
                    pKF->GetKeypointInfo(pixels, pointsLocal);
                    new_kf->kps_pixel_ = std::move(pixels);
                    new_kf->kps_point_local_ = std::move(pointsLocal);
                    new_kf->img_undist_ = imgRGB_undistorted;
                    new_kf->img_auxiliary_undist_ = imgAux_undistorted;
                    if (isMonocularMvsPipelineEnabled()) {
                        captureMonocularMvsKeyframeMetadata(new_kf, pKF);
                    }

                    logKeyframeCameraToRerunRecordings(
                        new_kf,
                        pKF->mnId,
                        /*log_reconstruction_mesh=*/true);

                }
            }   // Mutex released

            // Prepare multi resolution images for training
            for (auto& kfit : scene_->keyframes()) {
                auto pkf = kfit.second;
                if (device_type_ == torch::kCUDA && !keyframe_images_on_cpu_) {
                    cv::cuda::GpuMat img_gpu;
                    img_gpu.upload(pkf->img_undist_);
                    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                        cv::cuda::GpuMat img_resized;
                        cv::cuda::resize(img_gpu, img_resized,
                                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                        pkf->gaus_pyramid_original_image_[l] =
                            voxel_utils::cvGpuMatToTorchTensorFloat32(img_resized);
                    }
                }
                else {
                    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                        cv::Mat img_resized;
                        cv::resize(pkf->img_undist_, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                        pkf->gaus_pyramid_original_image_[l] =
                            voxel_utils::cvMatToTorchTensorFloat32(img_resized,
                                keyframe_images_on_cpu_ ? torch::kCPU : device_type_);
                    }
                }
            }

            // Create MiniCams for all keyframes and use them for densification later
            std::vector<sv::MiniCam> tr_cams;
            tr_cams.reserve(scene_->keyframes().size());
            for (auto& kv : scene_->keyframes()) {
                // if (kv.second) tr_cams.push_back(kv.second->toMiniCam());
                auto& kf = *kv.second;
                // Use full-res here; you can choose a smaller level if you like.
                tr_cams.emplace_back(kf.toMiniCam(kf.image_height_, kf.image_width_));
            }

            //  Create voxel model & trainer setup
            {
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                if (sensor_type_ == RGBD && !allocate_orb_voxels_) {
                    throw std::runtime_error(
                        "RGB-D mapping requires Mapper.allocate_orb_voxels=1 for "
                        "the initial topology.");
                }
                voxel_model_->createFromPcd(scene_->cached_point_cloud_, tr_cams);
                if (rerun_params_.run_whole_run_ ||
                    rerun_params_.rerun_svrecon_debug_ ||
                    rerun_params_.rerun_monocular_debug_) {
                    rerun_state_.whole_run_live_voxels_dirty_ = true;
                }
                std::unique_lock<std::mutex> lock(mutex_settings_);
                voxel_model_->createTrainer(
                                            opt_params_.geo_lr_,
                                            opt_params_.sh0_lr_,
                                            opt_params_.shs_lr_,
                                            opt_params_.optim_beta1_,
                                            opt_params_.optim_beta2_,
                                            opt_params_.optim_eps_,
                                            opt_params_.lr_decay_ckpt_,
                                            opt_params_.lr_decay_mult_,
                                            opt_params_.log_s_lr_);
            }

            logCurrentOrbMapPointsToReconstructionRerun(getIteration());
            logCurrentOrbKeyframePosesToReconstructionRerun(getIteration());

            const bool do_inactive_geo_densify =
                isdoingInactiveGeoDensify();
            const bool do_initial_rgbd_completion =
                sensor_type_ == RGBD &&
                ((rgbd_tsdf_evidence_ &&
                  rgbd_tsdf_evidence_initial_backfill_) ||
                 (rgbd_fill_render_holes_ &&
                  rgbd_fill_render_holes_initial_backfill_));
            if (do_inactive_geo_densify || do_initial_rgbd_completion) {
                std::vector<std::shared_ptr<VoxelKeyframe>> initial_rgbd_kfs;
                initial_rgbd_kfs.reserve(scene_->keyframes().size());
                for (const auto& kv : scene_->keyframes()) {
                    if (kv.second && !kv.second->done_inactive_geo_densify_) {
                        initial_rgbd_kfs.push_back(kv.second);
                    }
                }
                const int previous_depth_cache_limit = max_depth_cached_;
                max_depth_cached_ = std::max(
                    1,
                    depth_cached_ + static_cast<int>(initial_rgbd_kfs.size()));
                for (const auto& pkf : initial_rgbd_kfs) {
                    increasePcdByKeyframeInactiveGeoDensify(
                        pkf,
                        /*include_inactive_geo=*/do_inactive_geo_densify,
                        /*include_rgbd_hole_fill=*/
                            do_initial_rgbd_completion);
                }
                flushInactiveGeoCache();
                processRgbdClosureCache();
                max_depth_cached_ = previous_depth_cache_limit;
            }
            if (sensor_type_ == MONOCULAR &&
                sv::kMonocularRenderedDepthDensify &&
                !isMonocularMvsPipelineEnabled()) {
                for (const auto& kv : scene_->keyframes()) {
                    if (kv.second) {
                        densifyMonocularFromRenderedDepth(kv.second);
                    }
                }
            }
            if (sensor_type_ == RGBD &&
                sdf_initialization_rgbd_projective_) {
                int64_t fused_observations = 0;
                int fused_keyframes = 0;
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                for (const auto& kv : scene_->keyframes()) {
                    if (!kv.second) {
                        continue;
                    }
                    const int64_t observed =
                        fuseProjectiveSdfInitFromKeyframe(kv.second);
                    if (observed > 0) {
                        fused_observations += observed;
                        ++fused_keyframes;
                    }
                }
                std::cout
                    << "[SDF/RGBD init] keyframes=" << fused_keyframes
                    << " grid_corner_observations=" << fused_observations
                    << "\n";
            }
            if (rerun_params_.enable_rerun_ &&
                (rerun_params_.run_whole_run_ ||
                 rerun_params_.rerun_svrecon_debug_ ||
                 rerun_params_.rerun_monocular_debug_)) {
                logWholeRunLiveVoxelsToRerun(
                    getIteration(),
                    voxel_model_->voxCenter(),
                    voxel_model_->voxSize(),
                    torch::Tensor(),
                    rerun_params_.run_whole_run_,
                    rerun_params_.rerun_svrecon_debug_,
                    rerun_params_.rerun_monocular_debug_);
                rerun_state_.svrecon_debug_has_source_snapshot_ =
                    rerun_params_.rerun_svrecon_debug_;
                rerun_state_.monocular_debug_has_source_snapshot_ =
                    rerun_params_.rerun_monocular_debug_;
                rerun_state_.whole_run_live_voxels_dirty_ = false;
            }
            if (opt_params_.prune_surface_views_enable_) {
                std::vector<std::shared_ptr<VoxelKeyframe>> initial_keyframes;
                initial_keyframes.reserve(scene_->keyframes().size());
                for (const auto& item : scene_->keyframes()) {
                    if (item.second) {
                        initial_keyframes.push_back(item.second);
                    }
                }
                markSurfaceViewPruningPending(initial_keyframes);
            }
            // One warm-up optimization step
            trainForOneIteration();

            initial_mapped_ = true;
            if (isMonocularMvsPipelineEnabled()) {
                std::vector<std::shared_ptr<VoxelKeyframe>> initial_mvs_kfs;
                initial_mvs_kfs.reserve(scene_->keyframes().size());
                for (const auto& item : scene_->keyframes()) {
                    if (item.second) {
                        initial_mvs_kfs.push_back(item.second);
                    }
                }
                scheduleLatestMonocularMvsKeyframe(initial_mvs_kfs);
            }
            input_backpressure_ready_.store(true, std::memory_order_release);
            break;  // Exit the initial mapping loop
        }
        else if (mpSLAM->isShutDown())
        {
            std::cout << "[VoxelMapper] Stopped before voxel initialization; "
                         "no voxel map to refine or export.\n";
            signalStop();
            return;
        }
        else
        {
            // Initial conditions not satisfied yet
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Second loop: Incremental voxel mapping
     int SLAM_stop_iter = 0;
     while (!isStopped()) {
        pollMonocularMvsDensification();
        // Check conditions for incremental mapping
        if (hasMetIncrementalMappingConditions()) {
            combineMappingOperations();
            if (cull_keyframes_)
                cullKeyframes();
        }

        // Invoke training once
        trainForOneIteration();

        if (mpSLAM->isShutDown()) {
            SLAM_stop_iter = getIteration();
            SLAM_ended_ = true;
        }

        const bool iteration_limit_reached =
            opt_params_.iterations_ > 0 &&
            getIteration() >= opt_params_.iterations_;
        if (SLAM_ended_ || iteration_limit_reached)
            break;
    }

    // Third loop: Tail optimization. Match GaussianMapper: scheduled topology
    // adaptation remains active while the post-SLAM optimization finishes.
    pollMonocularMvsDensification(/*wait_for_result=*/true);
    if (monocular_mvs_backend_) {
        monocular_mvs_backend_.reset();
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
    flushInactiveGeoCache();
    processRgbdClosureCache();
    int adapt_interval = opt_params_.adapt_every_;          // cfg.procedure.adapt_every
    int n_delay_iters  = adapt_interval * 0.8f;        // same heuristic as GS code
    const bool prev_tail_refinement_active = tail_refinement_active_;
    tail_refinement_active_ = true;
    while (getIteration() - SLAM_stop_iter <= n_delay_iters
        || (getIteration() % adapt_interval) <= n_delay_iters
        || isKeepingTraining() )
    {
        trainForOneIteration();
        // Re-read in case user changed cfg at runtime
        adapt_interval = opt_params_.adapt_every_;
        n_delay_iters  = adapt_interval * 0.8f;
    }
    tail_refinement_active_ = prev_tail_refinement_active;
    runFinalRefinement();

    // Save and clear
    const std::filesystem::path shutdown_dir =
        result_dir_ / (std::to_string(getIteration()) + "_shutdown");
    if (!config_file_path_.empty() && std::filesystem::exists(config_file_path_)) {
        try {
            std::filesystem::create_directories(shutdown_dir);
            std::filesystem::path config_copy_name = config_file_path_.filename();
            if (config_copy_name.empty()) {
                config_copy_name = "voxel_mapper.yaml";
            }
            std::filesystem::copy_file(
                config_file_path_,
                shutdown_dir / config_copy_name,
                std::filesystem::copy_options::overwrite_existing);
        } catch (const std::exception& e) {
            std::cerr << "[VoxelMapper] Failed to copy config file to shutdown folder: "
                      << e.what() << "\n";
        }
    }
    savePly(shutdown_dir / "ply");
    {
        const std::filesystem::path ply_dir =
            shutdown_dir / "ply" / "voxel_model" /
            ("iteration_" + std::to_string(getIteration()));
        const std::filesystem::path mesh_path = ply_dir / "voxel_surface_mesh.ply";
        const std::filesystem::path sdf_mesh_path =
            ply_dir / "voxel_surface_mesh_sdf.ply";
        const std::filesystem::path rendered_tsdf_mesh_path =
            ply_dir / "voxel_surface_mesh_rendered_tsdf.ply";

        const bool save_progressive_mesh =
            rerun_params_.save_progressive_rendered_tsdf_mesh_;
        {
            if (!save_progressive_mesh) {
                std::error_code remove_error;
                std::filesystem::remove(
                    rendered_tsdf_mesh_path,
                    remove_error);
                if (remove_error) {
                    std::cerr
                        << "[SVRecon mesh/rendered-TSDF] failed to remove stale export: "
                        << remove_error.message() << "\n";
                }
            }
            try {
                std::ofstream mesh_info(ply_dir / "mesh_reconstructions.txt");
                if (!mesh_info) {
                    throw std::runtime_error("failed to open mesh_reconstructions.txt");
                }
                mesh_info
                    << "voxel_model.ply: Native SVRecon octree cells with SH/color and corner SDF; viewer/model file, not a triangle mesh.\n"
                    << "voxel_surface_mesh.ply: Keyframe-weighted TSDF fusion of alpha-valid final rendered depths; optionally scale-aware with a coverage-adaptive sparse-grid budget.\n"
                    << "voxel_surface_mesh_sdf.ply: Direct zero-level Marching Cubes extraction from the optimized SVRecon corner SDF.\n";
                if (save_progressive_mesh) {
                    mesh_info
                        << "voxel_surface_mesh_rendered_tsdf.ply: Progressive SVRecon rendered-depth TSDF fusion followed by zero-level extraction.\n";
                }
            } catch (const std::exception& e) {
                std::cerr << "[mesh/info] shutdown export failed: "
                          << e.what() << "\n";
            }

            try {
                saveRenderedTsdfMeshPly(mesh_path);
                if (rerun_params_.enable_rerun_ &&
                    rerun_params_.rerun_reconstruction_mesh_ &&
                    std::filesystem::exists(mesh_path)) {
                    sv::RerunVisualizerBridge::instance().visualizeDebugPlyMesh(
                        "reconstruction_mesh",
                        mesh_path.string(),
                        getIteration(),
                        "world/mesh/final");
                }
            } catch (const std::exception& e) {
                std::cerr << "[mesh/rendered-TSDF-fixed] shutdown export failed: "
                          << e.what() << "\n";
            }
            c10::cuda::CUDACachingAllocator::emptyCache();

            if (save_progressive_mesh) {
                try {
                    saveSvreconRenderedTsdfMeshPly(rendered_tsdf_mesh_path);
                } catch (const std::exception& e) {
                    std::cerr << "[SVRecon mesh/rendered-TSDF] shutdown export failed: "
                              << e.what() << "\n";
                }
                c10::cuda::CUDACachingAllocator::emptyCache();
            }

            try {
                saveSvreconSdfMeshPly(sdf_mesh_path);
            } catch (const std::exception& e) {
                std::cerr << "[SVRecon mesh/SDF] shutdown export failed: "
                          << e.what() << "\n";
            }
        }
    }
    logLearnedDepthMapsToWholeRunRerun();
    saveRerunRecordingsAtShutdown();

    signalStop();
}
