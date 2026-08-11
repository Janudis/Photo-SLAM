#include "include_voxel/voxel_mapper.h"

#include "include_voxel/tandem_mvs_backend.h"
#include "include_voxel/voxel_mapper_utils.h"

#include "ORB-SLAM3/include/KeyFrame.h"
#include "ORB-SLAM3/include/Map.h"
#include "ORB-SLAM3/include/MapPoint.h"

#include <c10/cuda/CUDACachingAllocator.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace {

cv::Mat toMvsBgr(
    const cv::Mat& mapper_image,
    const int width,
    const int height)
{
    if (mapper_image.empty() || width <= 0 || height <= 0) {
        throw std::runtime_error("TANDEM MVS received an empty keyframe image");
    }

    cv::Mat image_u8;
    if (mapper_image.depth() == CV_8U) {
        image_u8 = mapper_image;
    } else {
        const double scale =
            mapper_image.depth() == CV_32F || mapper_image.depth() == CV_64F
                ? 255.0
                : 1.0;
        mapper_image.convertTo(
            image_u8,
            CV_MAKETYPE(CV_8U, mapper_image.channels()),
            scale);
    }

    cv::Mat bgr;
    if (image_u8.channels() == 1) {
        cv::cvtColor(image_u8, bgr, cv::COLOR_GRAY2BGR);
    } else if (image_u8.channels() == 3) {
        // Mapper keyframes use RGB order; TANDEM's wrapper accepts BGR.
        cv::cvtColor(image_u8, bgr, cv::COLOR_RGB2BGR);
    } else if (image_u8.channels() == 4) {
        cv::cvtColor(image_u8, bgr, cv::COLOR_RGBA2BGR);
    } else {
        throw std::runtime_error(
            "TANDEM MVS keyframes must have one, three, or four channels");
    }

    cv::Mat resized;
    if (bgr.cols == width && bgr.rows == height) {
        resized = bgr.clone();
    } else {
        cv::resize(
            bgr,
            resized,
            cv::Size(width, height),
            0.0,
            0.0,
            cv::INTER_AREA);
    }
    return resized.isContinuous() ? resized : resized.clone();
}

Eigen::Matrix3f resizedIntrinsics(
    const VoxelKeyframe& keyframe,
    const int width,
    const int height)
{
    if (keyframe.intr_.size() < 4 || keyframe.image_width_ <= 0 ||
        keyframe.image_height_ <= 0) {
        throw std::runtime_error(
            "TANDEM MVS reference keyframe has invalid intrinsics");
    }

    const float sx = static_cast<float>(width) /
                     static_cast<float>(keyframe.image_width_);
    const float sy = static_cast<float>(height) /
                     static_cast<float>(keyframe.image_height_);
    Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
    K(0, 0) = sx * keyframe.intr_[0];
    K(1, 1) = sy * keyframe.intr_[1];
    K(0, 2) = 0.5f * static_cast<float>(width - 1) +
              sx * (keyframe.intr_[2] -
                    0.5f * static_cast<float>(keyframe.image_width_ - 1));
    K(1, 2) = 0.5f * static_cast<float>(height - 1) +
              sy * (keyframe.intr_[3] -
                    0.5f * static_cast<float>(keyframe.image_height_ - 1));
    return K;
}

void setMiniCamSnapshot(
    sv::MiniCam& camera,
    const Eigen::Matrix4f& camera_to_world,
    const Eigen::Matrix3f& intrinsics,
    const int width,
    const int height,
    const int frame_id)
{
    camera.width = width;
    camera.height = height;
    camera.fx = intrinsics(0, 0);
    camera.fy = intrinsics(1, 1);
    camera.cx = intrinsics(0, 2);
    camera.cy = intrinsics(1, 2);
    camera.c2w = voxel_utils::eigenMatrixToTorchTensor(
        camera_to_world, torch::kCPU).contiguous();
    camera.w2c = torch::linalg_inv(camera.c2w).contiguous();
    camera.frame_id = frame_id;
    camera.position = camera.c2w.index(
        {torch::indexing::Slice(0, 3), 3}).clone();
    camera.lookat = camera.c2w.index(
        {torch::indexing::Slice(0, 3), 2}).clone();
    const float lookat_norm = camera.lookat.norm().item<float>();
    if (lookat_norm > 1.0e-6f) {
        camera.lookat /= lookat_norm;
    }
    camera.tanfovx = std::tan(
        0.5f * sv::focalToFov(camera.fx, width));
    camera.tanfovy = std::tan(
        0.5f * sv::focalToFov(camera.fy, height));
    camera.pix_size = 2.0f * camera.tanfovx /
                      static_cast<float>(width);
}

long long frameOrder(const std::shared_ptr<VoxelKeyframe>& keyframe)
{
    if (!keyframe) {
        return -1;
    }
    return keyframe->source_frame_id_ >= 0
        ? static_cast<long long>(keyframe->source_frame_id_)
        : static_cast<long long>(keyframe->fid_);
}

bool computeTandemSparseDepthRange(
    ORB_SLAM3::KeyFrame* keyframe,
    const float depth_min,
    const float inverse_depth_quantile,
    const float depth_max_multiplier,
    float& depth_max,
    std::size_t& sparse_depth_count)
{
    depth_max = 0.0f;
    sparse_depth_count = 0;
    if (!keyframe || !(depth_min > 0.0f) ||
        !(inverse_depth_quantile > 0.0f && inverse_depth_quantile < 1.0f) ||
        !(depth_max_multiplier > 0.0f)) {
        return false;
    }

    const Sophus::SE3f camera_from_world = keyframe->GetPose();
    std::vector<float> inverse_depths;
    const std::vector<ORB_SLAM3::MapPoint*> map_points =
        keyframe->GetMapPointMatches();
    inverse_depths.reserve(map_points.size());
    for (ORB_SLAM3::MapPoint* map_point : map_points) {
        if (!map_point || map_point->isBad()) {
            continue;
        }
        const Eigen::Vector3f world = map_point->GetWorldPos();
        if (!world.allFinite()) {
            continue;
        }
        const float depth = (camera_from_world * world).z();
        if (!std::isfinite(depth) || depth <= 1.0e-6f) {
            continue;
        }
        inverse_depths.push_back(1.0f / depth);
    }
    if (inverse_depths.empty()) {
        return false;
    }

    // TANDEM derives its far plane from the sparse tracker in the same gauge
    // as the supplied camera poses. This keeps pure-monocular MVS aligned with
    // ORB-SLAM without introducing a separate metric or per-frame scale.
    const std::size_t quantile_index = std::min(
        inverse_depths.size() - 1,
        static_cast<std::size_t>(
            inverse_depth_quantile *
            static_cast<float>(inverse_depths.size())));
    std::nth_element(
        inverse_depths.begin(),
        inverse_depths.begin() + quantile_index,
        inverse_depths.end());
    const float inverse_depth = inverse_depths[quantile_index];
    if (!std::isfinite(inverse_depth) || inverse_depth <= 0.0f) {
        return false;
    }

    depth_max = depth_max_multiplier / inverse_depth;
    sparse_depth_count = inverse_depths.size();
    return std::isfinite(depth_max) && depth_max > depth_min;
}

bool mvsPoseChanged(
    const Eigen::Matrix4f& snapshot,
    const Eigen::Matrix4f& current,
    const float depth_scale)
{
    if (!snapshot.allFinite() || !current.allFinite()) {
        return true;
    }
    const float translation_change =
        (snapshot.block<3, 1>(0, 3) - current.block<3, 1>(0, 3)).norm();
    const Eigen::Matrix3f rotation_change =
        snapshot.block<3, 3>(0, 0).transpose() *
        current.block<3, 3>(0, 0);
    const float cos_angle = std::clamp(
        0.5f * (rotation_change.trace() - 1.0f), -1.0f, 1.0f);
    const float angle_change = std::acos(cos_angle);
    return translation_change > 1.0e-3f * std::max(1.0f, depth_scale) ||
           angle_change > 1.0e-3f;
}

struct DirectCornerSample
{
    Eigen::Vector3f world = Eigen::Vector3f::Zero();
    double sdf_sum = 0.0;
    int count = 0;
};

} // namespace

void VoxelMapper::captureMonocularMvsKeyframeMetadata(
    const std::shared_ptr<VoxelKeyframe>& pkf,
    ORB_SLAM3::KeyFrame* orb_keyframe)
{
    if (!pkf || !orb_keyframe) {
        return;
    }

    pkf->covisible_keyframes_.clear();
    const std::vector<ORB_SLAM3::KeyFrame*> covisible =
        orb_keyframe->GetVectorCovisibleKeyFrames();
    pkf->covisible_keyframes_.reserve(covisible.size());
    for (ORB_SLAM3::KeyFrame* source : covisible) {
        if (!source || source->isBad()) {
            continue;
        }
        pkf->covisible_keyframes_.emplace_back(
            source->mnId, orb_keyframe->GetWeight(source));
    }
    ORB_SLAM3::Map* map = orb_keyframe->GetMap();
    pkf->monocular_depth_anchor_pixels_.clear();
    pkf->monocular_depth_anchor_depths_.clear();
    if (map) {
        const std::vector<ORB_SLAM3::MapPoint*> map_points =
            orb_keyframe->GetMapPointMatches();
        const std::size_t count = std::min(
            map_points.size(), orb_keyframe->mvKeysUn.size());
        pkf->monocular_depth_anchor_pixels_.reserve(count);
        pkf->monocular_depth_anchor_depths_.reserve(count);
        const Sophus::SE3f camera_from_world = orb_keyframe->GetPose();
        const unsigned long current_keyframe_id = map->GetMaxKFid();
        for (std::size_t index = 0; index < count; ++index) {
            ORB_SLAM3::MapPoint* map_point = map_points[index];
            if (!isMatureMonocularOrbMapPoint(
                    map_point, current_keyframe_id)) {
                continue;
            }
            const cv::Point2f pixel = orb_keyframe->mvKeysUn[index].pt;
            const Eigen::Vector3f world = map_point->GetWorldPos();
            const float depth = (camera_from_world * world).z();
            if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y) ||
                pixel.x < 0.0f || pixel.y < 0.0f ||
                pixel.x >= static_cast<float>(pkf->image_width_) ||
                pixel.y >= static_cast<float>(pkf->image_height_) ||
                !std::isfinite(depth) || depth <= 1.0e-6f) {
                continue;
            }
            pkf->monocular_depth_anchor_pixels_.emplace_back(
                pixel.x, pixel.y);
            pkf->monocular_depth_anchor_depths_.push_back(depth);
        }
    }
    const bool pose_ready =
        map && (!monocular_mvs_requires_inertial_ba1_ ||
                map->GetIniertialBA1());

    pkf->monocular_mvs_depth_min_ = 0.0f;
    pkf->monocular_mvs_depth_max_ = 0.0f;
    pkf->monocular_mvs_sparse_depth_count_ = 0;
    bool depth_range_ready = false;
    if (monocular_mvs_depth_range_mode_ == "fixed") {
        pkf->monocular_mvs_depth_min_ = monocular_mvs_depth_min_m_;
        pkf->monocular_mvs_depth_max_ = monocular_mvs_depth_max_m_;
        depth_range_ready =
            monocular_mvs_depth_max_m_ > monocular_mvs_depth_min_m_;
    } else {
        pkf->monocular_mvs_depth_min_ = monocular_mvs_depth_min_scene_;
        depth_range_ready = computeTandemSparseDepthRange(
            orb_keyframe,
            monocular_mvs_depth_min_scene_,
            monocular_mvs_inverse_depth_quantile_,
            monocular_mvs_depth_max_multiplier_,
            pkf->monocular_mvs_depth_max_,
            pkf->monocular_mvs_sparse_depth_count_);
    }
    pkf->monocular_mvs_pose_ready_ = pose_ready && depth_range_ready;
}

bool VoxelMapper::isMonocularMvsPipelineEnabled() const
{
    return monocular_mvs_densify_ || monocular_mvs_tsdf_evidence_;
}

void VoxelMapper::refreshMonocularMvsKeyframeMetadata()
{
    if ((!isMonocularMvsPipelineEnabled() &&
         !monocular_omnidata_densify_) ||
        !mpSLAM || !mpSLAM->getAtlas()) {
        return;
    }
    ORB_SLAM3::Map* map = mpSLAM->getAtlas()->GetCurrentMap();
    if (!map) {
        return;
    }

    std::unique_lock<std::mutex> lock_map(map->mMutexMapUpdate);
    for (ORB_SLAM3::KeyFrame* orb_keyframe : map->GetAllKeyFrames()) {
        if (!orb_keyframe || orb_keyframe->isBad()) {
            continue;
        }
        captureMonocularMvsKeyframeMetadata(
            scene_->getKeyframe(orb_keyframe->mnId), orb_keyframe);
    }
}

std::vector<std::shared_ptr<VoxelKeyframe>>
VoxelMapper::selectMonocularMvsSourceKeyframes(
    const std::shared_ptr<VoxelKeyframe>& reference,
    int view_num) const
{
    std::vector<std::shared_ptr<VoxelKeyframe>> selected;
    if (view_num < 2) {
        view_num = monocular_mvs_view_num_;
    }
    if (!reference || view_num < 2) {
        return selected;
    }
    const std::size_t source_count =
        static_cast<std::size_t>(view_num - 1);

    int max_covisibility_weight = 0;
    for (const auto& item : reference->covisible_keyframes_) {
        max_covisibility_weight =
            std::max(max_covisibility_weight, item.second);
    }

    struct Candidate
    {
        std::shared_ptr<VoxelKeyframe> keyframe;
        float penalty = 0.0f;
    };
    std::vector<Candidate> candidates;
    const Eigen::Matrix4f reference_tcw =
        reference->getPosef().matrix();
    for (const auto& item : reference->covisible_keyframes_) {
        if (item.second <= static_cast<float>(max_covisibility_weight) / 3.0f) {
            continue;
        }
        std::shared_ptr<VoxelKeyframe> source =
            scene_->getKeyframe(item.first);
        if (!source || source == reference || source->img_undist_.empty()) {
            continue;
        }

        const Eigen::Matrix4f reference_from_source =
            reference_tcw * source->getPosef().inverse().matrix();
        const float translation =
            reference_from_source.block<3, 1>(0, 3).norm();
        const float trace =
            reference_from_source.block<3, 3>(0, 0).trace();
        const float rotation = std::sqrt(std::max(
            0.0f,
            2.0f * (1.0f - std::min(3.0f, trace) / 3.0f)));
        if (std::sqrt(
                translation * translation + rotation * rotation) < 0.20f) {
            continue;
        }

        const float rotation_penalty = rotation * rotation;
        const float translation_delta = translation - 0.25f;
        const float translation_penalty =
            (translation_delta < 0.0f ? 5.0f : 1.0f) *
            translation_delta * translation_delta;
        const float covisibility_penalty =
            item.second < static_cast<float>(max_covisibility_weight) / 2.5f
                ? 2.0f
                : 0.0f;
        candidates.push_back(
            {source,
             rotation_penalty + translation_penalty +
                 covisibility_penalty});
    }

    if (candidates.size() >= source_count) {
        std::sort(
            candidates.begin(), candidates.end(),
            [](const Candidate& lhs, const Candidate& rhs) {
                return lhs.penalty < rhs.penalty;
            });
        selected.reserve(source_count);
        for (std::size_t index = 0; index < source_count; ++index) {
            selected.push_back(candidates[index].keyframe);
        }
        return selected;
    }

    // SimpleMapping falls back to preceding keyframes, using every third
    // frame once the map is large enough. Preserve that policy, then fill any
    // remaining slots from the skipped preceding frames.
    std::vector<std::shared_ptr<VoxelKeyframe>> preceding;
    const long long reference_order = frameOrder(reference);
    for (const auto& item : scene_->keyframes()) {
        const std::shared_ptr<VoxelKeyframe>& source = item.second;
        if (!source || source == reference || source->img_undist_.empty() ||
            frameOrder(source) >= reference_order) {
            continue;
        }
        preceding.push_back(source);
    }
    std::sort(
        preceding.begin(), preceding.end(),
        [](const auto& lhs, const auto& rhs) {
            return frameOrder(lhs) > frameOrder(rhs);
        });

    std::unordered_set<unsigned long> used;
    const std::size_t step = preceding.size() >= 24 ? 3 : 1;
    for (std::size_t index = 0;
         index < preceding.size() && selected.size() < source_count;
         index += step) {
        selected.push_back(preceding[index]);
        used.insert(preceding[index]->fid_);
    }
    for (const auto& source : preceding) {
        if (selected.size() >= source_count) {
            break;
        }
        if (used.insert(source->fid_).second) {
            selected.push_back(source);
        }
    }
    if (selected.size() != source_count) {
        selected.clear();
    }
    return selected;
}

bool VoxelMapper::scheduleMonocularMvsDensification(
    const std::shared_ptr<VoxelKeyframe>& reference)
{
    if (!isMonocularMvsPipelineEnabled() || !monocular_mvs_backend_ ||
        !reference || !reference->monocular_mvs_pose_ready_ ||
        reference->img_undist_.empty() ||
        monocular_mvs_backend_->hasPending() ||
        monocular_mvs_scheduled_keyframes_.count(reference->fid_) != 0) {
        return false;
    }

    const float depth_min = reference->monocular_mvs_depth_min_;
    const float depth_max = reference->monocular_mvs_depth_max_;
    if (!std::isfinite(depth_min) || !std::isfinite(depth_max) ||
        !(depth_min > 0.0f && depth_max > depth_min)) {
        return false;
    }

    const std::vector<std::shared_ptr<VoxelKeyframe>> sources =
        selectMonocularMvsSourceKeyframes(reference);
    if (sources.size() + 1 !=
        static_cast<std::size_t>(monocular_mvs_view_num_)) {
        return false;
    }

    std::vector<std::shared_ptr<VoxelKeyframe>> views;
    views.reserve(static_cast<std::size_t>(monocular_mvs_view_num_));
    views.push_back(reference);
    views.insert(views.end(), sources.begin(), sources.end());

    std::vector<cv::Mat> bgr_images;
    std::vector<Eigen::Matrix4f> camera_to_world;
    bgr_images.reserve(views.size());
    camera_to_world.reserve(views.size());
    for (const auto& view : views) {
        bgr_images.push_back(toMvsBgr(
            view->img_undist_, monocular_mvs_width_, monocular_mvs_height_));
        camera_to_world.push_back(
            view->getPosef().inverse().matrix());
    }

    const Eigen::Matrix3f K = resizedIntrinsics(
        *reference, monocular_mvs_width_, monocular_mvs_height_);
    if (monocular_mvs_empty_cache_before_launch_) {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
    monocular_mvs_backend_->launch(
        bgr_images,
        K,
        camera_to_world,
        depth_min,
        depth_max,
        monocular_mvs_discard_percentage_);
    beginLaptopAsyncModule("mvs_inference", 1);

    monocular_mvs_pending_reference_ = reference;
    monocular_mvs_pending_c2w_ = camera_to_world.front();
    monocular_mvs_pending_view_ids_.clear();
    monocular_mvs_pending_view_ids_.reserve(views.size());
    for (const auto& view : views) {
        monocular_mvs_pending_view_ids_.push_back(view->fid_);
    }
    monocular_mvs_pending_view_c2w_ = camera_to_world;
    monocular_mvs_pending_depth_min_ = depth_min;
    monocular_mvs_pending_depth_max_ = depth_max;
    monocular_mvs_pending_camera_ = reference->toMiniCam(
        monocular_mvs_height_, monocular_mvs_width_);
    setMiniCamSnapshot(
        monocular_mvs_pending_camera_,
        monocular_mvs_pending_c2w_,
        K,
        monocular_mvs_width_,
        monocular_mvs_height_,
        static_cast<int>(reference->fid_));
    cv::cvtColor(
        bgr_images.front(),
        monocular_mvs_pending_reference_rgb_,
        cv::COLOR_BGR2RGB);
    monocular_mvs_scheduled_keyframes_.insert(reference->fid_);

    return true;
}

void VoxelMapper::scheduleLatestMonocularMvsKeyframe(
    const std::vector<std::shared_ptr<VoxelKeyframe>>& candidates)
{
    if (!isMonocularMvsPipelineEnabled() || !monocular_mvs_backend_ ||
        monocular_mvs_backend_->hasPending() || candidates.empty()) {
        return;
    }

    refreshMonocularMvsKeyframeMetadata();
    std::vector<std::shared_ptr<VoxelKeyframe>> newest_first = candidates;
    std::sort(
        newest_first.begin(), newest_first.end(),
        [](const auto& lhs, const auto& rhs) {
            return frameOrder(lhs) > frameOrder(rhs);
        });
    for (const auto& candidate : newest_first) {
        if (scheduleMonocularMvsDensification(candidate)) {
            return;
        }
    }
}

void VoxelMapper::pollMonocularMvsDensification(
    const bool wait_for_result)
{
    if (!isMonocularMvsPipelineEnabled() || !monocular_mvs_backend_ ||
        !monocular_mvs_backend_->hasPending()) {
        return;
    }

    std::optional<sv::TandemMvsResult> result =
        monocular_mvs_backend_->collect(wait_for_result);
    if (!result.has_value()) {
        return;
    }
    endLaptopAsyncModule("mvs_inference");

    bool poses_changed =
        monocular_mvs_pending_view_ids_.size() !=
            monocular_mvs_pending_view_c2w_.size();
    for (std::size_t index = 0;
         !poses_changed && index < monocular_mvs_pending_view_ids_.size();
         ++index) {
        const std::shared_ptr<VoxelKeyframe> view =
            scene_->getKeyframe(monocular_mvs_pending_view_ids_[index]);
        if (!view || mvsPoseChanged(
                         monocular_mvs_pending_view_c2w_[index],
                         view->getPosef().inverse().matrix(),
                         monocular_mvs_pending_depth_max_)) {
            poses_changed = true;
        }
    }
    if (poses_changed) {
        const std::shared_ptr<VoxelKeyframe> retry_reference =
            monocular_mvs_pending_reference_;
        if (retry_reference) {
            monocular_mvs_scheduled_keyframes_.erase(retry_reference->fid_);
        }
        monocular_mvs_pending_reference_.reset();
        monocular_mvs_pending_reference_rgb_.release();
        monocular_mvs_pending_view_ids_.clear();
        monocular_mvs_pending_view_c2w_.clear();
        monocular_mvs_pending_depth_min_ = 0.0f;
        monocular_mvs_pending_depth_max_ = 0.0f;
        if (!wait_for_result && retry_reference) {
            refreshMonocularMvsKeyframeMetadata();
            scheduleMonocularMvsDensification(retry_reference);
        }
        return;
    }
    integrateMonocularMvsDepth(*result);
    monocular_mvs_pending_reference_.reset();
    monocular_mvs_pending_reference_rgb_.release();
    monocular_mvs_pending_view_ids_.clear();
    monocular_mvs_pending_view_c2w_.clear();
    monocular_mvs_pending_depth_min_ = 0.0f;
    monocular_mvs_pending_depth_max_ = 0.0f;
}

void VoxelMapper::integrateMonocularMvsDepth(
    const sv::TandemMvsResult& result)
{
    cacheMonocularDepthPrior(
        monocular_mvs_pending_reference_,
        result.depth,
        result.confidence,
        sv::LearnedDepthSource::TandemMvs);
    if (monocular_mvs_tsdf_evidence_) {
        integrateMonocularMvsTsdfEvidence(result);
    } else {
        integrateMonocularLearnedDepth(
            result.depth,
            "MVS",
            "world/monocular_mvs/created",
            /*clear_cuda_cache_before_insertion=*/false);
    }
}

void VoxelMapper::cacheMonocularDepthPrior(
    const std::shared_ptr<VoxelKeyframe>& reference,
    const cv::Mat& depth,
    const cv::Mat& confidence,
    const sv::LearnedDepthSource source)
{
    if (!reference || depth.empty() || depth.type() != CV_32FC1) {
        throw std::runtime_error(
            "Cannot cache an invalid monocular depth prior");
    }

    cv::Mat confidence_float;
    if (confidence.empty()) {
        confidence_float = cv::Mat(
            depth.rows, depth.cols, CV_32FC1, cv::Scalar(1.0f));
    } else {
        if (confidence.rows != depth.rows || confidence.cols != depth.cols ||
            confidence.channels() != 1) {
            throw std::runtime_error(
                "Monocular depth confidence shape does not match depth");
        }
        if (confidence.type() == CV_32FC1) {
            confidence_float = confidence.clone();
        } else {
            confidence.convertTo(confidence_float, CV_32FC1);
        }
    }

    cv::Mat accepted_depth = depth.clone();
    int64_t accepted_pixels = 0;
    for (int y = 0; y < accepted_depth.rows; ++y) {
        float* depth_row = accepted_depth.ptr<float>(y);
        float* confidence_row = confidence_float.ptr<float>(y);
        for (int x = 0; x < accepted_depth.cols; ++x) {
            float& depth_value = depth_row[x];
            float& confidence_value = confidence_row[x];
            if (!std::isfinite(depth_value) || depth_value <= 1.0e-6f ||
                !std::isfinite(confidence_value)) {
                depth_value = 0.0f;
                confidence_value = 0.0f;
                continue;
            }
            confidence_value = std::clamp(confidence_value, 0.0f, 1.0f);
            if (confidence_value <= 0.0f) {
                depth_value = 0.0f;
                continue;
            }
            ++accepted_pixels;
        }
    }

    if (accepted_pixels == 0) {
        std::cerr
            << "[VoxelMapper] Learned depth prior has no valid pixels for "
               "keyframe "
            << reference->fid_ << "\n";
        return;
    }

    reference->monocular_depth_prior_ = std::move(accepted_depth);
    reference->monocular_depth_confidence_ = std::move(confidence_float);
    reference->monocular_depth_source_ = source;
    reference->monocular_depth_prior_iteration_ = getIteration();
}

void VoxelMapper::integrateMonocularLearnedDepth(
    const cv::Mat& depth_map,
    const std::string& source_name,
    const std::string& rerun_entity_path,
    const bool clear_cuda_cache_before_insertion)
{
    const std::shared_ptr<VoxelKeyframe> reference =
        monocular_mvs_pending_reference_;
    const int depth_width = monocular_mvs_pending_camera_.width;
    const int depth_height = monocular_mvs_pending_camera_.height;
    if (!reference || depth_map.empty() ||
        depth_map.type() != CV_32FC1 ||
        depth_map.cols != depth_width ||
        depth_map.rows != depth_height ||
        depth_width <= 0 || depth_height <= 0 ||
        !(monocular_mvs_pending_depth_min_ > 0.0f) ||
        !(monocular_mvs_pending_depth_max_ >
          monocular_mvs_pending_depth_min_) ||
        monocular_mvs_pending_reference_rgb_.empty()) {
        throw std::runtime_error(
            source_name + " returned an invalid filtered depth map");
    }
    auto integration_profile = profileLaptopModule(
        source_name == "Omnidata"
            ? "omnidata_depth_integration"
            : "mvs_depth_integration");

    std::unordered_map<std::string, torch::Tensor> render_pkg;
    {
        torch::NoGradGuard no_grad;
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        render_pkg = voxel_model_->render(
            monocular_mvs_pending_camera_,
            depth_height,
            depth_width,
            torch::Tensor(),
            "dontcare",
            /*track_max_w=*/false,
            std::nullopt,
            /*output_depth=*/true,
            /*output_normal=*/false,
            /*output_T=*/true,
            /*rand_bg=*/false,
            /*use_auto_exposure=*/false,
            sv::RenderOpts{});
    }

    torch::Tensor rendered_depth_cpu;
    torch::Tensor rendered_alpha_cpu;
    torch::Tensor n_contrib_cpu;
    if (!voxel_utils::renderPkgToDepthAlphaMaps(
            render_pkg,
            depth_height,
            depth_width,
            rendered_depth_cpu,
            rendered_alpha_cpu,
            n_contrib_cpu)) {
        throw std::runtime_error(
            "SVRecon did not return depth/contributor maps for " +
            source_name + " hole detection");
    }

    cv::Mat valid_camera_mask;
    if (!reference->cam_.undistort_mask.empty()) {
        cv::Mat mask_single_channel;
        if (reference->cam_.undistort_mask.channels() == 1) {
            mask_single_channel = reference->cam_.undistort_mask;
        } else {
            cv::extractChannel(
                reference->cam_.undistort_mask,
                mask_single_channel,
                0);
        }
        cv::resize(
            mask_single_channel,
            valid_camera_mask,
            cv::Size(depth_width, depth_height),
            0.0,
            0.0,
            cv::INTER_NEAREST);
    }

    const auto rendered_depth = rendered_depth_cpu.accessor<float, 2>();
    const auto contributors = n_contrib_cpu.accessor<int, 2>();
    const Eigen::Matrix3f K = resizedIntrinsics(
        *reference, depth_width, depth_height);
    const Eigen::Matrix3f rotation =
        monocular_mvs_pending_c2w_.block<3, 3>(0, 0);
    const Eigen::Vector3f translation =
        monocular_mvs_pending_c2w_.block<3, 1>(0, 3);

    std::vector<float> surface_points;
    std::vector<float> surface_colors;
    surface_points.reserve(
        static_cast<std::size_t>(depth_width) *
        static_cast<std::size_t>(depth_height));
    surface_colors.reserve(surface_points.capacity());
    for (int y = 0; y < depth_height; ++y) {
        const float* depth_row = depth_map.ptr<float>(y);
        for (int x = 0; x < depth_width; ++x) {
            const float depth = depth_row[x];
            if (!std::isfinite(depth) ||
                depth < monocular_mvs_pending_depth_min_ ||
                depth > monocular_mvs_pending_depth_max_) {
                continue;
            }
            if (!valid_camera_mask.empty()) {
                const float mask_value =
                    valid_camera_mask.depth() == CV_8U
                        ? static_cast<float>(valid_camera_mask.at<uint8_t>(y, x)) /
                              255.0f
                        : valid_camera_mask.at<float>(y, x);
                if (mask_value < 0.5f) {
                    continue;
                }
            }
            const float current_depth = rendered_depth[y][x];
            const bool structural_hole =
                contributors[y][x] <= 0 &&
                (!std::isfinite(current_depth) || current_depth <= 1.0e-6f);
            if (!structural_hole) {
                continue;
            }
            const Eigen::Vector3f camera_point(
                (static_cast<float>(x) - K(0, 2)) * depth / K(0, 0),
                (static_cast<float>(y) - K(1, 2)) * depth / K(1, 1),
                depth);
            const Eigen::Vector3f world_point =
                rotation * camera_point + translation;
            if (!world_point.allFinite()) {
                continue;
            }
            surface_points.insert(
                surface_points.end(),
                {world_point.x(), world_point.y(), world_point.z()});
            const cv::Vec3b rgb =
                monocular_mvs_pending_reference_rgb_.at<cv::Vec3b>(y, x);
            surface_colors.insert(
                surface_colors.end(),
                {static_cast<float>(rgb[0]) / 255.0f,
                 static_cast<float>(rgb[1]) / 255.0f,
                 static_cast<float>(rgb[2]) / 255.0f});
        }
    }

    if (surface_points.empty()) {
        return;
    }

    torch::Tensor surface_point_tensor = torch::from_blob(
        surface_points.data(),
        {static_cast<int64_t>(surface_points.size() / 3), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
    torch::Tensor surface_color_tensor = torch::from_blob(
        surface_colors.data(),
        {static_cast<int64_t>(surface_colors.size() / 3), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();

    torch::Tensor support_centers;
    torch::Tensor source_indices;
    torch::Tensor scene_center_cpu;
    float scene_extent = 0.0f;
    int insertion_level = 0;
    {
        torch::NoGradGuard no_grad;
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        std::tie(support_centers, source_indices) =
            voxel_model_->rgbdHoleSupportCellCenters(surface_point_tensor);
        scene_center_cpu = voxel_model_->SceneCenter()
                               .detach().to(torch::kCPU).to(torch::kFloat32)
                               .reshape({3}).contiguous();
        scene_extent = voxel_model_->SceneExtent()
                           .detach().to(torch::kCPU).to(torch::kFloat32)
                           .item<float>();
        insertion_level = voxel_model_->insertionOctreeLevel();
    }
    if (!support_centers.defined() || support_centers.numel() == 0 ||
        insertion_level <= 0 || insertion_level >= 30 ||
        !std::isfinite(scene_extent) || scene_extent <= 0.0f) {
        return;
    }

    support_centers = support_centers.detach().to(torch::kCPU)
                          .to(torch::kFloat32).reshape({-1, 3}).contiguous();
    source_indices = source_indices.detach().to(torch::kCPU)
                         .to(torch::kLong).reshape({-1}).contiguous();
    torch::Tensor support_colors = surface_color_tensor.index_select(
        0, source_indices).contiguous();

    const Eigen::Vector3f scene_center(
        scene_center_cpu[0].item<float>(),
        scene_center_cpu[1].item<float>(),
        scene_center_cpu[2].item<float>());
    const Eigen::Vector3f scene_min =
        scene_center - Eigen::Vector3f::Constant(0.5f * scene_extent);
    const int grid_dim = 1 << insertion_level;
    const float cell_size = scene_extent / static_cast<float>(grid_dim);
    const float truncation = std::max(
        1.0e-4f, sdf_params_.sdf_init_trunc_vox_ * cell_size);
    const Eigen::Matrix4f world_to_camera =
        monocular_mvs_pending_c2w_.inverse();

    std::unordered_set<sv::RgbdTsdfGridKey, sv::RgbdTsdfGridKeyHash>
        unique_cells;
    std::unordered_map<
        sv::RgbdTsdfGridKey,
        DirectCornerSample,
        sv::RgbdTsdfGridKeyHash> direct_samples;
    auto centers_accessor = support_centers.accessor<float, 2>();
    for (int64_t row = 0; row < support_centers.size(0); ++row) {
        const Eigen::Vector3f center(
            centers_accessor[row][0],
            centers_accessor[row][1],
            centers_accessor[row][2]);
        const Eigen::Vector3f grid = (center - scene_min) / cell_size;
        const sv::RgbdTsdfGridKey cell{
            static_cast<int>(std::floor(grid.x())),
            static_cast<int>(std::floor(grid.y())),
            static_cast<int>(std::floor(grid.z()))};
        if (cell.x < 0 || cell.y < 0 || cell.z < 0 ||
            cell.x >= grid_dim || cell.y >= grid_dim || cell.z >= grid_dim ||
            !unique_cells.insert(cell).second) {
            continue;
        }

        for (int dz = 0; dz <= 1; ++dz) {
            for (int dy = 0; dy <= 1; ++dy) {
                for (int dx = 0; dx <= 1; ++dx) {
                    const sv::RgbdTsdfGridKey corner{
                        cell.x + dx, cell.y + dy, cell.z + dz};
                    const Eigen::Vector3f world = scene_min +
                        cell_size * Eigen::Vector3f(
                            static_cast<float>(corner.x),
                            static_cast<float>(corner.y),
                            static_cast<float>(corner.z));
                    const Eigen::Vector4f camera_h =
                        world_to_camera * Eigen::Vector4f(
                            world.x(), world.y(), world.z(), 1.0f);
                    const float z = camera_h.z();
                    if (!std::isfinite(z) || z <= 1.0e-6f) {
                        continue;
                    }
                    const int x = static_cast<int>(std::lround(
                        K(0, 0) * camera_h.x() / z + K(0, 2)));
                    const int y = static_cast<int>(std::lround(
                        K(1, 1) * camera_h.y() / z + K(1, 2)));
                    if (x < 0 || x >= depth_width ||
                        y < 0 || y >= depth_height) {
                        continue;
                    }
                    const float measured = depth_map.at<float>(y, x);
                    const float sdf = measured - z;
                    if (!std::isfinite(measured) ||
                        measured < monocular_mvs_pending_depth_min_ ||
                        measured > monocular_mvs_pending_depth_max_ ||
                        !std::isfinite(sdf) || std::abs(sdf) > truncation) {
                        continue;
                    }
                    DirectCornerSample& sample = direct_samples[corner];
                    sample.world = world;
                    sample.sdf_sum += std::clamp(
                        sdf, -truncation, truncation);
                    ++sample.count;
                }
            }
        }
    }

    std::vector<float> corner_points;
    std::vector<float> corner_values;
    corner_points.reserve(direct_samples.size() * 3);
    corner_values.reserve(direct_samples.size());
    for (const auto& item : direct_samples) {
        const DirectCornerSample& sample = item.second;
        if (sample.count <= 0) {
            continue;
        }
        corner_points.insert(
            corner_points.end(),
            {sample.world.x(), sample.world.y(), sample.world.z()});
        corner_values.push_back(
            static_cast<float>(sample.sdf_sum /
                               static_cast<double>(sample.count)));
    }

    torch::Tensor corner_point_tensor;
    torch::Tensor corner_value_tensor;
    if (!corner_values.empty()) {
        corner_point_tensor = torch::from_blob(
            corner_points.data(),
            {static_cast<int64_t>(corner_values.size()), 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
        corner_value_tensor = torch::from_blob(
            corner_values.data(),
            {static_cast<int64_t>(corner_values.size()), 1},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
    }

    sv::VoxelModel::IncreasePcdStats stats;
    {
        torch::NoGradGuard no_grad;
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        if (corner_point_tensor.defined()) {
            voxel_model_->setNextSdfInitializationGridSamples(
                corner_point_tensor, corner_value_tensor);
        }
        voxel_model_->setNextRealInsertionRerunEntityPath(
            rerun_entity_path);
        voxel_model_->increasePcd(
            support_centers,
            support_colors,
            getIteration(),
            incrementalMappingCameras(),
            clear_cuda_cache_before_insertion);
        voxel_model_->setNextRealInsertionRerunEntityPath("");
        stats = voxel_model_->lastIncreasePcdStats();
    }

    if (stats.new_voxels > 0 &&
        (rerun_params_.run_whole_run_ ||
         rerun_params_.rerun_svrecon_debug_)) {
        rerun_state_.whole_run_live_voxels_dirty_ = true;
    }
}
