#include "src_voxel/voxel_mapper_internal.h"

bool VoxelMapper::getCurrentVoxelBounds(
    Eigen::Vector3f& minimum,
    Eigen::Vector3f& maximum) const
{
    std::unique_lock<std::mutex> lock_render(mutex_render_);
    if (!voxel_model_)
        return false;

    torch::NoGradGuard no_grad;
    torch::Tensor centers = voxel_model_->voxCenter().detach();
    if (!centers.defined() || centers.dim() != 2 ||
        centers.size(0) == 0 || centers.size(1) != 3)
        return false;

    centers = centers.to(torch::kCPU);
    const torch::Tensor minimum_tensor = std::get<0>(centers.min(0));
    const torch::Tensor maximum_tensor = std::get<0>(centers.max(0));
    minimum = Eigen::Vector3f(
        minimum_tensor[0].item<float>(),
        minimum_tensor[1].item<float>(),
        minimum_tensor[2].item<float>());
    maximum = Eigen::Vector3f(
        maximum_tensor[0].item<float>(),
        maximum_tensor[1].item<float>(),
        maximum_tensor[2].item<float>());
    return true;
}

cv::Mat VoxelMapper::renderFromPose(
    const Sophus::SE3f &Tcw,
    const int width,
    const int height,
    const bool main_vision)
{
    // Same guard as Photo-SLAM: no rendering before we have something
    if (!initial_mapped_ || getIteration() <= 0) {
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    }

    // Build a temporary keyframe for the viewer pose
    std::shared_ptr<VoxelKeyframe> pkf = std::make_shared<VoxelKeyframe>();
    // pkf->zfar_ = z_far_;   // only if you actually use z_far_ anywhere
    pkf->znear_ = z_near_;

    // Pose
    pkf->setPose(
        Tcw.unit_quaternion().cast<double>(),
        Tcw.translation().cast<double>());

    try {
        // Camera
        sv::Camera& camera = scene_->cameras_.at(viewer_camera_id_);
        pkf->setCameraParams(camera);
        // If your VoxelKeyframe has this (like GaussianKeyframe), call it:
        // pkf->computeTransformTensors();
    }
    catch (const std::out_of_range&) {
        throw std::runtime_error("[VoxelMapper::renderFromPose] KeyFrame Camera not found!");
    }

    // Build MiniCam for the viewer resolution
    sv::MiniCam cam = pkf->toMiniCam(height, width);

    // We don't want gradients in the viewer
    torch::NoGradGuard no_grad;

    // Call voxel_model_->render under the same render mutex
    std::unordered_map<std::string, torch::Tensor> pkg;
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);

        pkg = voxel_model_->render(
            cam,
            height,
            width,
            /* gt_image      */ torch::Tensor(),  // none
            /* color_mode    */ nullptr,
            /* track_max_w   */ false,
            /* ss            */ std::nullopt,
            /* output_depth  */ false,
            /* output_normal */ false,
            /* output_T      */ false,
            /* rand_bg       */ false,
            /* use_auto_exp  */ false,
            sv::RenderOpts{}   // default options
        );
    }

    // Check we actually got a color image
    auto it = pkg.find("color");
    if (it == pkg.end() || !it->second.defined()) {
        // Fallback: black image
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    }

    torch::Tensor color = it->second;  // expected shape [1,3,H,W] or [3,H,W]

    // Masking exactly like GaussianMapper
    torch::Tensor mask;
    if (main_vision) {
        mask = viewer_main_undistort_mask_[pkf->camera_id_];
    } else {
        mask = viewer_sub_undistort_mask_[pkf->camera_id_];
    }

    // Make sure mask is on the same device as color
    if (mask.device() != color.device()) {
        mask = mask.to(color.device());
    }

    // Both should be broadcastable: mask is usually [1,3,H,W] or [3,H,W]
    torch::Tensor masked_image = color * mask;

    // Reuse Photo-SLAM utility to convert to cv::Mat (float32 RGB)
    return voxel_utils::torchTensorToCvMatFloat32(masked_image);
}

// VoxelMapper::~VoxelMapper() {
//     // Explicitly reset any Python or Torch objects that may call Python at destruction
//     voxel_model_.reset();  // Deallocates all tensors and Python wrappers
//     mpSLAM.reset();
// }

int VoxelMapper::getIteration()
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    return iteration_;
}

std::vector<sv::MiniCam> VoxelMapper::incrementalMappingCameras() const
{
    const auto& keyframes = scene_->keyframes();
    const std::size_t limit = incremental_mapping_window_size_ > 0
        ? std::min<std::size_t>(
              static_cast<std::size_t>(incremental_mapping_window_size_),
              keyframes.size())
        : keyframes.size();

    std::vector<sv::MiniCam> cameras;
    cameras.reserve(limit);
    for (auto it = keyframes.rbegin();
         it != keyframes.rend() && cameras.size() < limit;
         ++it) {
        if (!it->second) {
            continue;
        }
        cameras.push_back(it->second->toMiniCam(
            it->second->image_height_, it->second->image_width_));
    }
    return cameras;
}

std::vector<sv::MiniCam> VoxelMapper::surfaceViewPruningCameras() const
{
    const auto& keyframes = scene_->keyframes();
    const std::size_t limit = std::min<std::size_t>(
        static_cast<std::size_t>(
            sv::kSurfaceViewWindowSize),
        keyframes.size());

    std::vector<sv::MiniCam> cameras;
    cameras.reserve(limit);
    for (auto it = keyframes.rbegin();
         it != keyframes.rend() && cameras.size() < limit;
         ++it) {
        if (!it->second) {
            continue;
        }
        cameras.push_back(it->second->toMiniCam(
            it->second->image_height_, it->second->image_width_));
    }
    return cameras;
}

void VoxelMapper::markSurfaceViewPruningPending(
    const std::vector<std::shared_ptr<VoxelKeyframe>>& keyframes)
{
    if (!opt_params_.prune_surface_views_enable_) {
        return;
    }
    for (const auto& keyframe : keyframes) {
        if (keyframe) {
            surface_view_pending_keyframes_.insert(keyframe->fid_);
        }
    }
}

bool VoxelMapper::surfaceViewPruningReady()
{
    if (!opt_params_.prune_surface_views_enable_ ||
        disable_topology_changes_ ||
        !scene_ ||
        !voxel_model_ ||
        surface_view_pending_keyframes_.empty()) {
        return false;
    }

    for (auto it = surface_view_pending_keyframes_.begin();
         it != surface_view_pending_keyframes_.end();) {
        const auto keyframe_it = scene_->keyframes().find(*it);
        if (keyframe_it == scene_->keyframes().end() || !keyframe_it->second) {
            it = surface_view_pending_keyframes_.erase(it);
            continue;
        }
        if (keyframe_it->second->remaining_times_of_use_ > 0) {
            return false;
        }
        ++it;
    }

    if (surface_view_pending_keyframes_.empty()) {
        return false;
    }
    return scene_->keyframes().size() >=
        static_cast<std::size_t>(
            sv::kSurfaceViewWindowSize);
}

void VoxelMapper::waitForInputQueueSlot()
{
    if (input_queue_max_keyframes_ <= 0) {
        return;
    }

    while (!isStopped() && !mpSLAM->isShutDown()) {
        const auto keyframe_ids = mpSLAM->getAtlas()->GetCurrentKeyFrameIds();
        if (keyframe_ids.empty()) {
            return;
        }

        if (!input_backpressure_ready_.load(std::memory_order_acquire)) {
            if (keyframe_ids.size() <= min_num_initial_map_kfs_) {
                return;
            }
        } else {
            const auto latest_orb_keyframe = static_cast<long long>(
                *std::max_element(keyframe_ids.begin(), keyframe_ids.end()));
            const auto latest_mapper_keyframe =
                latest_consumed_keyframe_id_.load(std::memory_order_acquire);
            if (latest_mapper_keyframe < 0 ||
                latest_orb_keyframe - latest_mapper_keyframe <=
                    input_queue_max_keyframes_) {
                return;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void VoxelMapper::increaseIteration(const int inc)
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    iteration_ += inc;
}

float VoxelMapper::geoLearningRateInit()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.geo_lr_;
}

float VoxelMapper::sh0LearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.sh0_lr_;
}

float VoxelMapper::shsLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.shs_lr_;
}

float VoxelMapper::lambdaSsim()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.lambda_ssim_;
}

int VoxelMapper::densifyInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.adapt_every_;
}

int VoxelMapper::newKeyframeTimesOfUse()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return new_keyframe_times_of_use_;
}

int VoxelMapper::stableNumIterExistence()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return stable_num_iter_existence_;
}

bool VoxelMapper::isKeepingTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return keep_training_;
}
bool VoxelMapper::isdoingGausPyramidTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return do_gaus_pyramid_training_;
}

bool VoxelMapper::isdoingInactiveGeoDensify()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return inactive_geo_densify_;
}

 void VoxelMapper::setKeepTraining(const bool keep)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     keep_training_ = keep;
 }
 VariableParameters VoxelMapper::getVaribleParameters()
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     VariableParameters params;
     params.geo_lr = opt_params_.geo_lr_;
     params.sh0_lr = opt_params_.sh0_lr_;
     params.shs_lr = opt_params_.shs_lr_;
     params.lambda_ssim = opt_params_.lambda_ssim_;
     params.densify_interval = opt_params_.adapt_every_;
     params.new_kf_times_of_use = new_keyframe_times_of_use_;
     params.stable_num_iter_existence = stable_num_iter_existence_;
     params.keep_training = keep_training_;
     params.do_gaus_pyramid_training = do_gaus_pyramid_training_;
     params.do_inactive_geo_densify = inactive_geo_densify_;
     return params;
 }

 void VoxelMapper::setVaribleParameters(const VariableParameters &params)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     opt_params_.geo_lr_ = params.geo_lr;
     opt_params_.sh0_lr_ = params.sh0_lr;
     opt_params_.shs_lr_ = params.shs_lr;
     opt_params_.lambda_ssim_ = params.lambda_ssim;
     opt_params_.adapt_every_ = params.densify_interval;
     new_keyframe_times_of_use_ = params.new_kf_times_of_use;
     stable_num_iter_existence_ = params.stable_num_iter_existence;
     keep_training_ = params.keep_training;
     do_gaus_pyramid_training_ = params.do_gaus_pyramid_training;
     inactive_geo_densify_ = params.do_inactive_geo_densify;
 }
