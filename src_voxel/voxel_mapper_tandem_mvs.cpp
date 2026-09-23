#include "include_voxel/voxel_mapper.h"

#include "include_voxel/tandem_mvs_backend.h"
#include "include_voxel/voxel_mapper_utils.h"

#include "ORB-SLAM3/include/KeyFrame.h"
#include "ORB-SLAM3/include/Map.h"
#include "ORB-SLAM3/include/MapPoint.h"

#include <c10/cuda/CUDACachingAllocator.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
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
    if (std::string_view(sv::kMonocularMvsDepthRangeMode) == "fixed") {
        pkf->monocular_mvs_depth_min_ = monocular_mvs_depth_min_m_;
        pkf->monocular_mvs_depth_max_ = monocular_mvs_depth_max_m_;
        depth_range_ready =
            monocular_mvs_depth_max_m_ > monocular_mvs_depth_min_m_;
    } else {
        pkf->monocular_mvs_depth_min_ = sv::kMonocularMvsDepthMinScene;
        depth_range_ready = computeTandemSparseDepthRange(
            orb_keyframe,
            sv::kMonocularMvsDepthMinScene,
            sv::kMonocularMvsInverseDepthQuantile,
            sv::kMonocularMvsDepthMaxMultiplier,
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
    if (!isMonocularMvsPipelineEnabled() || !mpSLAM || !mpSLAM->getAtlas()) {
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
        view_num = sv::kMonocularMvsViewNum;
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
        static_cast<std::size_t>(sv::kMonocularMvsViewNum)) {
        return false;
    }


    std::vector<std::shared_ptr<VoxelKeyframe>> views;
    views.reserve(static_cast<std::size_t>(sv::kMonocularMvsViewNum));
    views.push_back(reference);
    views.insert(views.end(), sources.begin(), sources.end());

    std::vector<cv::Mat> bgr_images;
    std::vector<Eigen::Matrix4f> camera_to_world;
    bgr_images.reserve(views.size());
    camera_to_world.reserve(views.size());
    for (const auto& view : views) {
        bgr_images.push_back(toMvsBgr(
            view->img_undist_, sv::kMonocularMvsWidth, sv::kMonocularMvsHeight));
        camera_to_world.push_back(
            view->getPosef().inverse().matrix());
    }

    const Eigen::Matrix3f K = resizedIntrinsics(
        *reference, sv::kMonocularMvsWidth, sv::kMonocularMvsHeight);
    if (monocular_mvs_empty_cache_before_launch_) {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
    monocular_mvs_backend_->launch(
        bgr_images,
        K,
        camera_to_world,
        depth_min,
        depth_max,
        sv::kMonocularMvsDiscardPercentage);
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
        sv::kMonocularMvsHeight, sv::kMonocularMvsWidth);
    setMiniCamSnapshot(
        monocular_mvs_pending_camera_,
        monocular_mvs_pending_c2w_,
        K,
        sv::kMonocularMvsWidth,
        sv::kMonocularMvsHeight,
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
