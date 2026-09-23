#include "src_voxel/voxel_mapper_internal.h"

bool VoxelMapper::isMatureMonocularOrbMapPoint(
    ORB_SLAM3::MapPoint* map_point,
    const unsigned long current_keyframe_id) const
{
    if (!map_point || map_point->isBad() || map_point->mnFirstKFid < 0) {
        return false;
    }

    // These are ORB-SLAM3's own monocular recent-point culling criteria.
    const long age =
        static_cast<long>(current_keyframe_id) - map_point->mnFirstKFid;
    if (age < 3 ||
        map_point->GetFoundRatio() < 0.25f ||
        map_point->Observations() <= 2) {
        return false;
    }

    const Eigen::Vector3f position = map_point->GetWorldPos();
    const Eigen::Vector3f color = map_point->GetColorRGB();
    return position.allFinite() && color.allFinite();
}

void VoxelMapper::collectNewMonocularOrbSamplingSupport(
    std::vector<float>& points,
    std::vector<float>& colors)
{
    points.clear();
    colors.clear();
    if (sensor_type_ != MONOCULAR || !mpSLAM || !mpSLAM->getAtlas()) {
        return;
    }

    ORB_SLAM3::Map* map = mpSLAM->getAtlas()->GetCurrentMap();
    if (!map) {
        return;
    }

    std::vector<sv::point3D_id_t> newly_transferred_ids;
    {
        std::unique_lock<std::mutex> lock_map(map->mMutexMapUpdate);
        const unsigned long current_keyframe_id = map->GetMaxKFid();
        const std::vector<ORB_SLAM3::MapPoint*> map_points =
            map->GetAllMapPoints();
        newly_transferred_ids.reserve(map_points.size());

        for (ORB_SLAM3::MapPoint* map_point : map_points) {
            if (!isMatureMonocularOrbMapPoint(
                    map_point, current_keyframe_id)) {
                continue;
            }

            const Eigen::Vector3f position = map_point->GetWorldPos();
            const Eigen::Vector3f color = map_point->GetColorRGB();

            if (monocular_orb_inserted_point_ids_.count(map_point->mnId) != 0) {
                continue;
            }
            newly_transferred_ids.push_back(map_point->mnId);
            points.insert(
                points.end(),
                {position.x(), position.y(), position.z()});
            colors.insert(
                colors.end(),
                {color.x(), color.y(), color.z()});
        }
    }

    monocular_orb_inserted_point_ids_.insert(
        newly_transferred_ids.begin(), newly_transferred_ids.end());
}

void VoxelMapper::combineMappingOperations()
{
    // Get Mapping Operations
    while (mpSLAM->getAtlas()->hasMappingOperation()) {
        ORB_SLAM3::MappingOperation opr =
            mpSLAM->getAtlas()->getAndPopMappingOperation();
        switch (opr.meOperationType)
        {
        case ORB_SLAM3::MappingOperation::OprType::LocalMappingBA:
        {
            bool kf_changed = false;
            std::vector<std::shared_ptr<VoxelKeyframe>> new_densification_kfs;
            // Get new keyframes
            auto& associated_kfs = opr.associatedKeyFrames();
            // Add keyframes to the scene
            for (auto& kf : associated_kfs) {
                // Keyframe Id
                auto kfid = std::get<0>(kf);
                std::shared_ptr<VoxelKeyframe> pkf = scene_->getKeyframe(kfid);
                // If the keyframe is already in the scene, only update the pose.
                // Otherwise create a new one
                if (pkf) {
                    auto& pose = std::get<2>(kf);
                    pkf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>());
                //  pkf->computeTransformTensors();
                    // Give local BA keyframes times of use
                    increaseKeyframeTimesOfUse(pkf, local_BA_increased_times_of_use_);
                    kf_changed = true;
		                }
			        else {
				        handleNewKeyframe(kf);                   // still void
                    std::shared_ptr<VoxelKeyframe> new_pkf =
                        scene_->getKeyframe(kfid);
                    if (new_pkf) {
                        new_densification_kfs.push_back(new_pkf);
                    }
                    kf_changed = true;
			        }
	            }
            // Get new points
            auto& associated_points = opr.associatedMapPoints();
            auto& points = std::get<0>(associated_points);
            auto& colors = std::get<1>(associated_points);

            const int iter = getIteration();
            if (allocate_orb_voxels_ && initial_mapped_) {
                std::vector<float> allocation_points;
                std::vector<float> allocation_colors;
                if (sensor_type_ == MONOCULAR) {
                    collectNewMonocularOrbSamplingSupport(
                        allocation_points, allocation_colors);
                } else {
                    allocation_points = points;
                    allocation_colors = colors;
                }

                if (allocation_points.size() >= 3) {
                    torch::NoGradGuard no_grad;
                    std::unique_lock<std::mutex> lock_render(
                        mutex_render_);
                    std::vector<sv::MiniCam> tr_cams =
                        incrementalMappingCameras();
                    voxel_model_->setNextRealInsertionRerunEntityPath(
                        "world/orb/voxels_created");
                    {
                        voxel_model_->increasePcd(
                            std::move(allocation_points),
                            std::move(allocation_colors),
                            getIteration(),
                            tr_cams);
                    }
                    voxel_model_->setNextRealInsertionRerunEntityPath("");
                    if (voxel_model_->lastIncreasePcdStats().new_voxels > 0 &&
                        (rerun_params_.run_whole_run_ ||
                         rerun_params_.rerun_svrecon_debug_ ||
                         rerun_params_.rerun_monocular_debug_)) {
                        rerun_state_.whole_run_live_voxels_dirty_ = true;
                    }
		                }
			        }

            // Preserve the established scheduling: ORB topology first,
            // then inactive geometry, then residual sensor evidence/fill.
            const bool do_inactive_geo_densify =
                isdoingInactiveGeoDensify();
            const bool do_rgbd_completion =
                sensor_type_ == RGBD &&
                (rgbd_fill_render_holes_ || rgbd_tsdf_evidence_);
            if ((do_inactive_geo_densify || do_rgbd_completion) &&
                !new_densification_kfs.empty()) {
                for (const auto& pkf : new_densification_kfs) {
                    increasePcdByKeyframeInactiveGeoDensify(
                        pkf,
                        /*include_inactive_geo=*/do_inactive_geo_densify,
                        /*include_rgbd_hole_fill=*/do_rgbd_completion);
                }
            }
            if (sensor_type_ == MONOCULAR &&
                sv::kMonocularRenderedDepthDensify &&
                !isMonocularMvsPipelineEnabled()) {
                for (const auto& pkf : new_densification_kfs) {
                    densifyMonocularFromRenderedDepth(pkf);
                }
            }
            if (sensor_type_ == MONOCULAR &&
                isMonocularMvsPipelineEnabled()) {
                scheduleLatestMonocularMvsKeyframe(new_densification_kfs);
            }
            processRgbdClosureCache();
            markSurfaceViewPruningPending(new_densification_kfs);
        }
        break;

        case ORB_SLAM3::MappingOperation::OprType::LoopClosingBA:
        {
            std::cout << "[Voxel Mapper]Loop Closure Detected."
                    << std::endl;

            bool kf_changed = false;
            // Get the loop keyframe scale modification factor
            float loop_kf_scale = opr.mfScale;

            // Get new keyframes (scaled transformation applied in ORB-SLAM3)
            auto& associated_kfs = opr.associatedKeyFrames();
            std::vector<std::shared_ptr<VoxelKeyframe>> new_densification_kfs;
            // std::vector<std::shared_ptr<VoxelKeyframe>> kfs_for_bounding;

             // Mark the transformed points to avoid transforming more than once
             torch::Tensor point_not_transformed_flags =
                 torch::full(
                     {voxel_model_->center_.size(0)},
                     true,
                     torch::TensorOptions().device(device_type_).dtype(torch::kBool));
             if (record_loop_ply_)
                 savePly(result_dir_ / (std::to_string(getIteration()) + "_0_before_loop_correction"));
             int num_transformed = 0;
             // Add keyframes to the scene
             for (auto& kf : associated_kfs) {
                 // Keyframe Id
                 auto kfid = std::get<0>(kf);
                 std::shared_ptr<VoxelKeyframe> pkf = scene_->getKeyframe(kfid);
                 // In case new points are added in handleNewKeyframe()
                 int64_t num_new_points = voxel_model_->center_.size(0) - point_not_transformed_flags.size(0);
                 if (num_new_points > 0)
                     point_not_transformed_flags = torch::cat({
                         point_not_transformed_flags,
                         torch::full({num_new_points}, true, point_not_transformed_flags.options())},
                         /*dim=*/0);
                 // If kf is already in the scene, evaluate the change in pose,
                 // if too large we perform loop correction on its visible model points.
                 // If not in the scene, create a new one.
                 if (pkf) {
                     auto& pose = std::get<2>(kf);
                     // If is loop closure kf
 // if (std::get<4>(kf)) {
                         Sophus::SE3f original_pose = pkf->getPosef(); // original_pose = old, inv_pose = new
                         Sophus::SE3f inv_pose = pose.inverse();
                         Sophus::SE3f diff_pose = inv_pose * original_pose;
                         bool large_rot = !diff_pose.rotationMatrix().isApprox(
                             Eigen::Matrix3f::Identity(), large_rot_th_);
                         bool large_trans = !diff_pose.translation().isMuchSmallerThan(
                             1.0, large_trans_th_);
                         if (large_rot || large_trans) {
                             std::cout << "[Voxel Mapper]Large loop correction detected, transforming visible points of kf "
                                     << kfid << std::endl;
                             diff_pose.translation() -= inv_pose.translation(); // t = (R_new * t_old + t_new) - t_new
                             diff_pose.translation() *= loop_kf_scale;          // t = s * (R_new * t_old)
                             diff_pose.translation() += inv_pose.translation(); // t = (s * R_new * t_old) + t_new
                             torch::Tensor diff_pose_tensor =
                                 voxel_utils::eigenMatrixToTorchTensor(
                                     diff_pose.matrix(), device_type_).transpose(0, 1);
	                             // Give loop keyframes times of use
	                             increaseKeyframeTimesOfUse(pkf, loop_closure_increased_times_of_use_);
	                         }
	                     pkf->setPose(
	                         pose.unit_quaternion().cast<double>(),
	                         pose.translation().cast<double>());

		                    kf_changed = true;
	                 }
			                 else if (loop_closure_reinsert_points_) {
			                     handleNewKeyframe(kf);
			                     pkf = scene_->getKeyframe(kfid);
                         if (pkf) {
                             new_densification_kfs.push_back(pkf);
                         }
		                         kf_changed = true;
			                 }
	             }
             if (record_loop_ply_)
                 savePly(result_dir_ / (std::to_string(getIteration()) + "_1_after_loop_correction"));
	             // Get new points (scaled transformation applied in ORB-SLAM3, so this step is performed at last to avoid scaling twice)
             auto& associated_points = opr.associatedMapPoints();
             auto& points = std::get<0>(associated_points);
             auto& colors = std::get<1>(associated_points);

             // Add new points to the model
             const int iter = getIteration();
	             if (loop_closure_reinsert_points_ &&
	                 allocate_orb_voxels_ &&
	                 initial_mapped_) {
                std::vector<float> allocation_points;
                std::vector<float> allocation_colors;
                if (sensor_type_ == MONOCULAR) {
                    collectNewMonocularOrbSamplingSupport(
                        allocation_points, allocation_colors);
                } else {
                    allocation_points = points;
                    allocation_colors = colors;
                }

                if (allocation_points.size() >= 3) {
                    torch::NoGradGuard no_grad;
                    std::unique_lock<std::mutex> lock_render(
                        mutex_render_);

                    // Match Photo-SLAM behavior: insert loop-closure associated points.
                    std::vector<sv::MiniCam> tr_cams =
                        incrementalMappingCameras();
                    voxel_model_->setNextRealInsertionRerunEntityPath(
                        "world/orb/voxels_created");
                    {
                        voxel_model_->increasePcd(
                            std::move(allocation_points),
                            std::move(allocation_colors),
                            iter,
                            tr_cams);
                    }
                    voxel_model_->setNextRealInsertionRerunEntityPath("");
                    if (voxel_model_->lastIncreasePcdStats().new_voxels > 0 &&
                        (rerun_params_.run_whole_run_ ||
                         rerun_params_.rerun_svrecon_debug_ ||
                         rerun_params_.rerun_monocular_debug_)) {
                        rerun_state_.whole_run_live_voxels_dirty_ = true;
                    }
                }
			             }

                const bool do_inactive_geo_densify =
                    isdoingInactiveGeoDensify();
                const bool do_rgbd_completion =
                    sensor_type_ == RGBD &&
                    (rgbd_fill_render_holes_ || rgbd_tsdf_evidence_);
                if ((do_inactive_geo_densify || do_rgbd_completion) &&
                    !new_densification_kfs.empty()) {
                    for (const auto& pkf : new_densification_kfs) {
                        increasePcdByKeyframeInactiveGeoDensify(
                            pkf,
                            /*include_inactive_geo=*/do_inactive_geo_densify,
                            /*include_rgbd_hole_fill=*/do_rgbd_completion);
                    }
                }
                if (sensor_type_ == MONOCULAR &&
                    sv::kMonocularRenderedDepthDensify &&
                    !isMonocularMvsPipelineEnabled()) {
                    for (const auto& pkf : new_densification_kfs) {
                        densifyMonocularFromRenderedDepth(pkf);
                    }
                }
                if (sensor_type_ == MONOCULAR &&
                    isMonocularMvsPipelineEnabled()) {
                    scheduleLatestMonocularMvsKeyframe(
                        new_densification_kfs);
                }
                processRgbdClosureCache();
                markSurfaceViewPruningPending(new_densification_kfs);
			            // Mark this iteration
	            loop_closure_iteration_ = true;
         }

         break;

         case ORB_SLAM3::MappingOperation::OprType::ScaleRefinement:
         {
             std::cout << "[Voxel Mapper]Scale refinement Detected. Transforming all kfs and points..."
                       << std::endl;

	             float s = opr.mfScale;
	             Sophus::SE3f& T = opr.mT;
             if (initial_mapped_) {
                 // Apply the scaled transformation on gaussian model points
                 {
                     std::unique_lock<std::mutex> lock_render(mutex_render_);
	                 }
	             }
             else { // TODO: the workflow should not come here, delete this branch
                 // Apply the scaled transformation to the cached points
                 for (auto& pt : scene_->cached_point_cloud_) {
                     // pt <- (s * Ryw * pt + tyw)
                     auto& pt_xyz = pt.second.xyz_;
                     pt_xyz *= s;
                     pt_xyz = T.cast<double>() * pt_xyz;
                 }

                 // Apply the scaled transformation on gaussian keyframes
                 for (auto& kfit : scene_->keyframes()) {
                     std::shared_ptr<VoxelKeyframe> pkf = kfit.second;
                     Sophus::SE3f Twc = pkf->getPosef().inverse();
                     Twc.translation() *= s;
	                     Sophus::SE3f Tyc = T * Twc;
	                     Sophus::SE3f Tcy = Tyc.inverse();
	                     pkf->setPose(Tcy.unit_quaternion().cast<double>(), Tcy.translation().cast<double>());
                    //  pkf->computeTransformTensors();
                 }
             }
         }

         break;

         default:
         {
             throw std::runtime_error("MappingOperation type not supported!");
         }
         break;
         }
     }
     logCurrentOrbMapPointsToReconstructionRerun(getIteration());
     logCurrentOrbKeyframePosesToReconstructionRerun(getIteration());
 }

 bool VoxelMapper::hasMetInitialMappingConditions() {
     if (!mpSLAM->isShutDown() &&
         mpSLAM->GetNumKeyframes() >= min_num_initial_map_kfs_ &&
         mpSLAM->getAtlas()->hasMappingOperation())
         return true;

     bool conditions_met = false;
     return conditions_met;
}

bool VoxelMapper::hasMetIncrementalMappingConditions() {
     if (!mpSLAM->isShutDown() &&
         mpSLAM->getAtlas()->hasMappingOperation())
         return true;

     bool conditions_met = false;
     return conditions_met;
}

void VoxelMapper::generateKfidRandomShuffle()
{
     if (scene_->keyframes().empty())
         return;

     std::size_t nkfs = scene_->keyframes().size();
     kfid_shuffle_.resize(nkfs);
     std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);
     std::mt19937 g(rd_());
     std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

     kfid_shuffled_ = true;
}

std::shared_ptr<VoxelKeyframe> VoxelMapper::useOneRandomSlidingWindowKeyframe()
{
    // If no keyframes, return nullptr
    if (scene_->keyframes().empty())
        return nullptr;

    // If not shuffled yet, build shuffle
    if (!kfid_shuffled_)
        generateKfidRandomShuffle();

    std::shared_ptr<VoxelKeyframe> viewpoint_cam = nullptr;
    int random_cam_idx;

    if (kfid_shuffled_) {
        int start_shuffle_idx = kfid_shuffle_idx_;
        do {
            // Next shuffled idx
            ++kfid_shuffle_idx_;
            if (kfid_shuffle_idx_ >= kfid_shuffle_.size())
                kfid_shuffle_idx_ = 0;
            // Add 1 time of use to all kfs if they are all unavalible
            if (kfid_shuffle_idx_ == start_shuffle_idx)
                for (auto& kfit : scene_->keyframes())
                    increaseKeyframeTimesOfUse(kfit.second, 1);
            // Get viewpoint kf
            random_cam_idx = kfid_shuffle_[kfid_shuffle_idx_];
            auto random_cam_it = scene_->keyframes().begin();
            for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
                ++random_cam_it;
            viewpoint_cam = (*random_cam_it).second;
        } while (viewpoint_cam->remaining_times_of_use_ <= 0);
    }

    // Count used times
    auto viewpoint_fid = viewpoint_cam->fid_;
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
        kfs_used_times_[viewpoint_fid] = 1;
    else
        ++kfs_used_times_[viewpoint_fid];

    // Handle times of use
    --(viewpoint_cam->remaining_times_of_use_);

    return viewpoint_cam;
}

void VoxelMapper::cullKeyframes()
{
    // Ask ORB-SLAM3 which keyframe IDs are still “live”
    std::unordered_set<unsigned long> kfids =
        mpSLAM->getAtlas()->GetCurrentKeyFrameIds();

     std::vector<unsigned long> kfids_to_erase;
     std::size_t nkfs = scene_->keyframes().size();
     kfids_to_erase.reserve(nkfs);
     for (auto& kfit : scene_->keyframes()) {
         unsigned long kfid = kfit.first;
         if (kfids.find(kfid) == kfids.end()) {
             kfids_to_erase.emplace_back(kfid);
         }
     }

     for (auto& kfid : kfids_to_erase) {
         scene_->keyframes().erase(kfid);
     }
}

void VoxelMapper::handleNewKeyframe(
    std::tuple<
        unsigned long,    // 0: keyframe ID
        unsigned long,    // 1: camera ID
        Sophus::SE3f,     // 2: pose
        cv::Mat,          // 3: RGB image
        bool,             // 4: loop‐closure flag (unused here)
        cv::Mat,          // 5: auxiliary (unused here)
        std::vector<float>, // 6: keypoint pixel coords (unused here)
        std::vector<float>, // 7: keypoint local coords (unused here)
        std::string> &kf       // 8: image filename (relative or absolute)
)
{
    // ─── Create a new VoxelKeyframe, exactly like Photo-SLAM’s Gaussian case ─
    std::shared_ptr<VoxelKeyframe> pkf  = std::make_shared<VoxelKeyframe>(std::get<0>(kf), getIteration());
    pkf->znear_ = z_near_;
    // Pose
    auto& pose = std::get<2>(kf);
    pkf->setPose(
        pose.unit_quaternion().cast<double>(),
        pose.translation().cast<double>()
    );
    cv::Mat imgRGB_undistorted, imgAux_undistorted;
    // Camera
    sv::Camera& camera = scene_->cameras_.at(std::get<1>(kf));
    pkf->setCameraParams(camera);

    cv::Mat imgRGB = std::get<3>(kf);
    camera.undistortImage(imgRGB, imgRGB_undistorted);
    // Auxiliary Image
    cv::Mat imgAux = std::get<5>(kf);
    if (this->sensor_type_ == RGBD) {
        imgAux_undistorted = mapperDepthForKeyframe(
            std::get<8>(kf), imgAux, camera);
    } else {
        imgAux_undistorted = imgAux;
    }

    pkf->original_image_ =
        voxel_utils::cvMatToTorchTensorFloat32(imgRGB_undistorted,
            keyframe_images_on_cpu_ ? torch::kCPU : device_type_);
    pkf->img_filename_ = std::get<8>(kf);
    pkf->source_timestamp_ =
        voxel_utils::parseFrameTimestampFromPath(pkf->img_filename_);
    pkf->source_frame_id_ = voxel_utils::parseFrameIdFromPath(pkf->img_filename_);
    pkf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
    pkf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
    pkf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;

    // Add the new keyframe to the scene
    // pkf->computeTransformTensors();
    scene_->addKeyframe(pkf, &kfid_shuffled_);
    latest_consumed_keyframe_id_.store(
        std::max(
            latest_consumed_keyframe_id_.load(std::memory_order_relaxed),
            static_cast<long long>(std::get<0>(kf))),
        std::memory_order_release);

    // Give new keyframes times of use and add it to the training sliding window
    increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());

    // Get dense point cloud from the new keyframe to accelerate training
    pkf->img_undist_ = imgRGB_undistorted;
    pkf->img_auxiliary_undist_ = imgAux_undistorted;

    pkf->kps_pixel_ = std::move(std::get<6>(kf));
    pkf->kps_point_local_ = std::move(std::get<7>(kf));

    // Prepare multi resolution images for training
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

    logKeyframeCameraToRerunRecordings(
        pkf,
        std::get<0>(kf),
        /*log_reconstruction_mesh=*/true);

    if (rerun_params_.run_whole_run_ ||
        rerun_params_.rerun_svrecon_debug_ ||
        rerun_params_.rerun_monocular_debug_) {
        rerun_state_.whole_run_live_voxels_dirty_ = true;
    }

    if (initial_mapped_ && sensor_type_ == RGBD &&
        sdf_initialization_rgbd_projective_) {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        fuseProjectiveSdfInitFromKeyframe(pkf);
    }
}
