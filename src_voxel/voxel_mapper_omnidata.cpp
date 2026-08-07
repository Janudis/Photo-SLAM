#include "include_voxel/voxel_mapper.h"

#include "include_voxel/omnidata_depth_backend.h"

#include <c10/cuda/CUDACachingAllocator.h>

#include <Eigen/Cholesky>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace {

cv::Mat toOmnidataRgb(const cv::Mat& mapper_image)
{
    if (mapper_image.empty()) {
        throw std::runtime_error("Omnidata received an empty keyframe image");
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

    cv::Mat rgb;
    if (image_u8.channels() == 1) {
        cv::cvtColor(image_u8, rgb, cv::COLOR_GRAY2RGB);
    } else if (image_u8.channels() == 3) {
        rgb = image_u8;
    } else if (image_u8.channels() == 4) {
        cv::cvtColor(image_u8, rgb, cv::COLOR_RGBA2RGB);
    } else {
        throw std::runtime_error(
            "Omnidata keyframes must have one, three, or four channels");
    }
    return rgb.clone();
}

cv::Mat resizeRgb(const cv::Mat& rgb, const int width, const int height)
{
    cv::Mat resized;
    if (rgb.cols == width && rgb.rows == height) {
        resized = rgb.clone();
    } else {
        cv::resize(
            rgb,
            resized,
            cv::Size(width, height),
            0.0,
            0.0,
            cv::INTER_AREA);
    }
    return resized.isContinuous() ? resized : resized.clone();
}

Eigen::Matrix3f omnidataIntrinsics(
    const VoxelKeyframe& keyframe,
    const int width,
    const int height)
{
    if (keyframe.intr_.size() < 4 || keyframe.image_width_ <= 0 ||
        keyframe.image_height_ <= 0) {
        throw std::runtime_error(
            "Omnidata reference keyframe has invalid intrinsics");
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

void setOmnidataCameraSnapshot(
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

long long omnidataFrameOrder(
    const std::shared_ptr<VoxelKeyframe>& keyframe)
{
    if (!keyframe) {
        return -1;
    }
    return keyframe->source_frame_id_ >= 0
        ? static_cast<long long>(keyframe->source_frame_id_)
        : static_cast<long long>(keyframe->fid_);
}

float median(std::vector<float> values)
{
    if (values.empty()) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const std::size_t middle = values.size() / 2;
    std::nth_element(
        values.begin(), values.begin() + middle, values.end());
    float result = values[middle];
    if (values.size() % 2 == 0) {
        const auto lower = std::max_element(
            values.begin(), values.begin() + middle);
        result = 0.5f * (result + *lower);
    }
    return result;
}

float bilinearDepth(const cv::Mat& depth, const float x, const float y)
{
    if (depth.empty() || depth.type() != CV_32FC1 ||
        x < 0.0f || y < 0.0f ||
        x > static_cast<float>(depth.cols - 1) ||
        y > static_cast<float>(depth.rows - 1)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, depth.cols - 1);
    const int y1 = std::min(y0 + 1, depth.rows - 1);
    const float dx = x - static_cast<float>(x0);
    const float dy = y - static_cast<float>(y0);
    const float d00 = depth.at<float>(y0, x0);
    const float d10 = depth.at<float>(y0, x1);
    const float d01 = depth.at<float>(y1, x0);
    const float d11 = depth.at<float>(y1, x1);
    if (!std::isfinite(d00) || !std::isfinite(d10) ||
        !std::isfinite(d01) || !std::isfinite(d11)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return (1.0f - dy) * ((1.0f - dx) * d00 + dx * d10) +
           dy * ((1.0f - dx) * d01 + dx * d11);
}

Eigen::Vector4f bilinearScaleWeights(
    const float x,
    const float y,
    const int width,
    const int height)
{
    const float nx = width > 1
        ? std::clamp(x / static_cast<float>(width - 1), 0.0f, 1.0f)
        : 0.0f;
    const float ny = height > 1
        ? std::clamp(y / static_cast<float>(height - 1), 0.0f, 1.0f)
        : 0.0f;
    return Eigen::Vector4f(
        (1.0f - nx) * (1.0f - ny),
        nx * (1.0f - ny),
        (1.0f - nx) * ny,
        nx * ny);
}

struct AlignmentSample
{
    Eigen::Vector4f row = Eigen::Vector4f::Zero();
    float target_disparity = 0.0f;
};

struct AlignedOmnidataDepth
{
    cv::Mat depth;
    int anchor_count = 0;
    float median_relative_error =
        std::numeric_limits<float>::infinity();
};

std::optional<AlignedOmnidataDepth> alignOmnidataToOrbGauge(
    const VoxelKeyframe& keyframe,
    const cv::Mat& raw_depth,
    const int min_anchors,
    const float max_relative_error,
    const float depth_min,
    const float depth_max)
{
    if (raw_depth.empty() || raw_depth.type() != CV_32FC1 ||
        keyframe.image_width_ <= 0 || keyframe.image_height_ <= 0 ||
        keyframe.monocular_depth_anchor_pixels_.size() !=
            keyframe.monocular_depth_anchor_depths_.size()) {
        return std::nullopt;
    }

    std::vector<AlignmentSample> samples;
    std::vector<float> initial_ratios;
    samples.reserve(keyframe.monocular_depth_anchor_depths_.size());
    initial_ratios.reserve(samples.capacity());
    const float sx = static_cast<float>(raw_depth.cols) /
                     static_cast<float>(keyframe.image_width_);
    const float sy = static_cast<float>(raw_depth.rows) /
                     static_cast<float>(keyframe.image_height_);
    for (std::size_t index = 0;
         index < keyframe.monocular_depth_anchor_depths_.size();
         ++index) {
        const Eigen::Vector2f& pixel =
            keyframe.monocular_depth_anchor_pixels_[index];
        const float orb_depth =
            keyframe.monocular_depth_anchor_depths_[index];
        const float x = (pixel.x() + 0.5f) * sx - 0.5f;
        const float y = (pixel.y() + 0.5f) * sy - 0.5f;
        const float prior_depth = bilinearDepth(raw_depth, x, y);
        if (!std::isfinite(prior_depth) || prior_depth <= 1.0e-6f ||
            !std::isfinite(orb_depth) || orb_depth <= 1.0e-6f) {
            continue;
        }
        const float prior_disparity = 1.0f / prior_depth;
        const float target_disparity = 1.0f / orb_depth;
        AlignmentSample sample;
        sample.row = prior_disparity * bilinearScaleWeights(
            x, y, raw_depth.cols, raw_depth.rows);
        sample.target_disparity = target_disparity;
        samples.push_back(sample);
        initial_ratios.push_back(target_disparity / prior_disparity);
    }
    if (samples.size() < static_cast<std::size_t>(min_anchors)) {
        return std::nullopt;
    }

    const float initial_scale = median(initial_ratios);
    if (!std::isfinite(initial_scale) || initial_scale <= 1.0e-8f) {
        return std::nullopt;
    }
    Eigen::Vector4f scales = Eigen::Vector4f::Constant(initial_scale);

    // HI-SLAM2 optimizes a 2x2 bilinear disparity-scale field jointly with
    // dense geometry. Here ORB depths provide the same gauge constraint and a
    // short robust IRLS solve prevents individual sparse outliers dominating.
    for (int iteration = 0; iteration < 6; ++iteration) {
        std::vector<float> absolute_residuals;
        absolute_residuals.reserve(samples.size());
        for (const AlignmentSample& sample : samples) {
            absolute_residuals.push_back(std::abs(
                sample.row.dot(scales) - sample.target_disparity));
        }
        const float mad = median(absolute_residuals);
        const float huber_delta = std::max(
            1.0e-6f, 1.345f * 1.4826f * mad);

        Eigen::Matrix4f hessian = Eigen::Matrix4f::Zero();
        Eigen::Vector4f gradient = Eigen::Vector4f::Zero();
        for (const AlignmentSample& sample : samples) {
            const float residual =
                sample.row.dot(scales) - sample.target_disparity;
            const float absolute_residual = std::abs(residual);
            const float weight = absolute_residual <= huber_delta
                ? 1.0f
                : huber_delta / absolute_residual;
            hessian.noalias() +=
                weight * sample.row * sample.row.transpose();
            gradient.noalias() +=
                weight * sample.row * sample.target_disparity;
        }
        const float regularizer = std::max(
            1.0e-8f, 1.0e-7f * hessian.trace());
        hessian.diagonal().array() += regularizer;
        const Eigen::Vector4f updated = hessian.ldlt().solve(gradient);
        if (!updated.allFinite()) {
            return std::nullopt;
        }
        scales = updated;
    }
    if ((scales.array() <= 1.0e-8f).any()) {
        return std::nullopt;
    }

    std::vector<float> relative_errors;
    relative_errors.reserve(samples.size());
    for (const AlignmentSample& sample : samples) {
        const float estimate = sample.row.dot(scales);
        relative_errors.push_back(std::abs(
            estimate - sample.target_disparity) /
            std::max(sample.target_disparity, 1.0e-6f));
    }
    const float median_relative_error = median(relative_errors);
    if (!std::isfinite(median_relative_error) ||
        median_relative_error > max_relative_error) {
        return std::nullopt;
    }

    cv::Mat aligned = cv::Mat::zeros(
        raw_depth.rows, raw_depth.cols, CV_32FC1);
    for (int y = 0; y < raw_depth.rows; ++y) {
        const float* raw_row = raw_depth.ptr<float>(y);
        float* aligned_row = aligned.ptr<float>(y);
        for (int x = 0; x < raw_depth.cols; ++x) {
            const float raw = raw_row[x];
            const float scale = bilinearScaleWeights(
                static_cast<float>(x),
                static_cast<float>(y),
                raw_depth.cols,
                raw_depth.rows).dot(scales);
            if (!std::isfinite(raw) || raw <= 1.0e-6f ||
                !std::isfinite(scale) || scale <= 1.0e-8f) {
                continue;
            }
            const float depth = raw / scale;
            if (std::isfinite(depth) && depth >= depth_min &&
                depth <= depth_max) {
                aligned_row[x] = depth;
            }
        }
    }

    return AlignedOmnidataDepth{
        std::move(aligned),
        static_cast<int>(samples.size()),
        median_relative_error};
}

cv::Mat filterBySourceConsistency(
    const cv::Mat& reference_depth,
    const Eigen::Matrix3f& reference_k,
    const Eigen::Matrix4f& reference_c2w,
    const std::vector<cv::Mat>& source_depths,
    const std::vector<Eigen::Matrix3f>& source_intrinsics,
    const std::vector<Eigen::Matrix4f>& source_w2c,
    const int min_source_views,
    const float relative_tolerance,
    const float absolute_tolerance)
{
    cv::Mat filtered = cv::Mat::zeros(
        reference_depth.rows, reference_depth.cols, CV_32FC1);
    const Eigen::Matrix3f reference_rotation =
        reference_c2w.block<3, 3>(0, 0);
    const Eigen::Vector3f reference_translation =
        reference_c2w.block<3, 1>(0, 3);

    for (int y = 0; y < reference_depth.rows; ++y) {
        const float* reference_row = reference_depth.ptr<float>(y);
        float* filtered_row = filtered.ptr<float>(y);
        for (int x = 0; x < reference_depth.cols; ++x) {
            const float depth = reference_row[x];
            if (!std::isfinite(depth) || depth <= 1.0e-6f) {
                continue;
            }
            const Eigen::Vector3f reference_point(
                (static_cast<float>(x) - reference_k(0, 2)) *
                    depth / reference_k(0, 0),
                (static_cast<float>(y) - reference_k(1, 2)) *
                    depth / reference_k(1, 1),
                depth);
            const Eigen::Vector3f world =
                reference_rotation * reference_point +
                reference_translation;

            int supporting_views = 0;
            for (std::size_t source_index = 0;
                 source_index < source_depths.size();
                 ++source_index) {
                const Eigen::Vector4f source_h =
                    source_w2c[source_index] * Eigen::Vector4f(
                        world.x(), world.y(), world.z(), 1.0f);
                const float source_z = source_h.z();
                if (!std::isfinite(source_z) || source_z <= 1.0e-6f) {
                    continue;
                }
                const Eigen::Matrix3f& source_k =
                    source_intrinsics[source_index];
                const float source_x =
                    source_k(0, 0) * source_h.x() / source_z +
                    source_k(0, 2);
                const float source_y =
                    source_k(1, 1) * source_h.y() / source_z +
                    source_k(1, 2);
                const float measured = bilinearDepth(
                    source_depths[source_index], source_x, source_y);
                if (!std::isfinite(measured) || measured <= 1.0e-6f) {
                    continue;
                }
                const float tolerance = std::max(
                    absolute_tolerance,
                    relative_tolerance * source_z);
                if (std::abs(measured - source_z) <= tolerance) {
                    ++supporting_views;
                    if (supporting_views >= min_source_views) {
                        filtered_row[x] = depth;
                        break;
                    }
                }
            }
        }
    }
    return filtered;
}

} // namespace

bool VoxelMapper::scheduleMonocularOmnidataDensification(
    const std::shared_ptr<VoxelKeyframe>& reference)
{
    if (!monocular_omnidata_densify_ || !monocular_omnidata_backend_ ||
        !reference || !reference->monocular_mvs_pose_ready_ ||
        reference->img_undist_.empty() ||
        monocular_omnidata_backend_->hasPending() ||
        monocular_omnidata_scheduled_keyframes_.count(reference->fid_) != 0 ||
        reference->monocular_depth_anchor_depths_.size() <
            static_cast<std::size_t>(
                monocular_omnidata_min_alignment_anchors_)) {
        return false;
    }

    const std::vector<std::shared_ptr<VoxelKeyframe>> sources =
        selectMonocularMvsSourceKeyframes(
            reference, monocular_omnidata_view_num_);
    if (sources.size() + 1 !=
        static_cast<std::size_t>(monocular_omnidata_view_num_)) {
        return false;
    }
    int alignable_sources = 0;
    for (const auto& source : sources) {
        if (source && source->monocular_mvs_pose_ready_ &&
            source->monocular_depth_anchor_depths_.size() >=
                static_cast<std::size_t>(
                    monocular_omnidata_min_alignment_anchors_)) {
            ++alignable_sources;
        }
    }
    if (alignable_sources < monocular_omnidata_min_source_views_) {
        return false;
    }

    std::vector<std::shared_ptr<VoxelKeyframe>> views;
    views.reserve(static_cast<std::size_t>(monocular_omnidata_view_num_));
    views.push_back(reference);
    views.insert(views.end(), sources.begin(), sources.end());

    monocular_omnidata_pending_reference_ = reference;
    monocular_omnidata_pending_view_ids_.clear();
    monocular_omnidata_pending_view_ids_.reserve(views.size());
    std::vector<sv::OmnidataDepthInput> missing_inputs;
    for (const auto& view : views) {
        monocular_omnidata_pending_view_ids_.push_back(view->fid_);
        if (monocular_omnidata_raw_depth_cache_.count(view->fid_) == 0) {
            missing_inputs.push_back(
                {view->fid_, toOmnidataRgb(view->img_undist_)});
        }
    }
    monocular_omnidata_scheduled_keyframes_.insert(reference->fid_);

    if (missing_inputs.empty()) {
        integrateMonocularOmnidataDepth(sv::OmnidataDepthResult{});
        monocular_omnidata_pending_reference_.reset();
        monocular_omnidata_pending_view_ids_.clear();
        return true;
    }

    if (monocular_omnidata_empty_cache_before_launch_) {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
    monocular_omnidata_backend_->launch(
        missing_inputs,
        monocular_omnidata_width_,
        monocular_omnidata_height_,
        monocular_omnidata_depth_multiplier_);
    beginLaptopAsyncModule(
        "omnidata_inference",
        static_cast<std::uint64_t>(missing_inputs.size()));
    std::cout
        << "[MONO/Omnidata launch] kf=" << reference->fid_
        << " sources=" << sources.size()
        << " uncached_priors=" << missing_inputs.size()
        << " resolution=" << monocular_omnidata_width_
        << "x" << monocular_omnidata_height_ << "\n";
    return true;
}

void VoxelMapper::scheduleLatestMonocularOmnidataKeyframe(
    const std::vector<std::shared_ptr<VoxelKeyframe>>& candidates)
{
    if (!monocular_omnidata_densify_ || !monocular_omnidata_backend_ ||
        monocular_omnidata_backend_->hasPending() || candidates.empty()) {
        return;
    }
    refreshMonocularMvsKeyframeMetadata();
    std::vector<std::shared_ptr<VoxelKeyframe>> newest_first = candidates;
    std::sort(
        newest_first.begin(), newest_first.end(),
        [](const auto& lhs, const auto& rhs) {
            return omnidataFrameOrder(lhs) > omnidataFrameOrder(rhs);
        });
    for (const auto& candidate : newest_first) {
        if (scheduleMonocularOmnidataDensification(candidate)) {
            return;
        }
    }
}

void VoxelMapper::pollMonocularOmnidataDensification(
    const bool wait_for_result)
{
    if (!monocular_omnidata_densify_ || !monocular_omnidata_backend_ ||
        !monocular_omnidata_backend_->hasPending()) {
        return;
    }
    std::optional<sv::OmnidataDepthResult> result =
        monocular_omnidata_backend_->collect(wait_for_result);
    if (!result.has_value()) {
        return;
    }
    endLaptopAsyncModule("omnidata_inference");
    integrateMonocularOmnidataDepth(*result);
    monocular_omnidata_pending_reference_.reset();
    monocular_omnidata_pending_view_ids_.clear();
}

void VoxelMapper::integrateMonocularOmnidataDepth(
    const sv::OmnidataDepthResult& result)
{
    auto alignment_profile =
        profileLaptopModule("omnidata_alignment_filtering");
    for (const sv::OmnidataDepthFrame& frame : result.frames) {
        if (frame.relative_depth.empty() ||
            frame.relative_depth.type() != CV_32FC1 ||
            frame.relative_depth.cols != monocular_omnidata_width_ ||
            frame.relative_depth.rows != monocular_omnidata_height_) {
            throw std::runtime_error(
                "Omnidata returned an invalid relative depth map");
        }
        monocular_omnidata_raw_depth_cache_[frame.keyframe_id] =
            frame.relative_depth.clone();
    }

    const std::shared_ptr<VoxelKeyframe> reference =
        monocular_omnidata_pending_reference_;
    if (!reference || monocular_omnidata_pending_view_ids_.empty()) {
        return;
    }
    refreshMonocularMvsKeyframeMetadata();

    const auto reference_raw =
        monocular_omnidata_raw_depth_cache_.find(reference->fid_);
    if (reference_raw == monocular_omnidata_raw_depth_cache_.end()) {
        return;
    }
    const std::optional<AlignedOmnidataDepth> reference_aligned =
        alignOmnidataToOrbGauge(
            *reference,
            reference_raw->second,
            monocular_omnidata_min_alignment_anchors_,
            monocular_omnidata_max_alignment_rel_error_,
            reference->monocular_mvs_depth_min_,
            reference->monocular_mvs_depth_max_);
    if (!reference_aligned.has_value()) {
        std::cout
            << "[MONO/Omnidata skip] kf=" << reference->fid_
            << " reason=ORB_scale_alignment\n";
        return;
    }

    std::vector<cv::Mat> source_depths;
    std::vector<Eigen::Matrix3f> source_intrinsics;
    std::vector<Eigen::Matrix4f> source_w2c;
    for (const unsigned long source_id :
         monocular_omnidata_pending_view_ids_) {
        if (source_id == reference->fid_) {
            continue;
        }
        const std::shared_ptr<VoxelKeyframe> source =
            scene_->getKeyframe(source_id);
        const auto source_raw =
            monocular_omnidata_raw_depth_cache_.find(source_id);
        if (!source || !source->monocular_mvs_pose_ready_ ||
            source_raw == monocular_omnidata_raw_depth_cache_.end()) {
            continue;
        }
        const std::optional<AlignedOmnidataDepth> aligned =
            alignOmnidataToOrbGauge(
                *source,
                source_raw->second,
                monocular_omnidata_min_alignment_anchors_,
                monocular_omnidata_max_alignment_rel_error_,
                source->monocular_mvs_depth_min_,
                source->monocular_mvs_depth_max_);
        if (!aligned.has_value()) {
            continue;
        }
        source_depths.push_back(aligned->depth);
        source_intrinsics.push_back(omnidataIntrinsics(
            *source,
            monocular_omnidata_width_,
            monocular_omnidata_height_));
        source_w2c.push_back(source->getPosef().matrix());
    }
    if (source_depths.size() <
        static_cast<std::size_t>(monocular_omnidata_min_source_views_)) {
        std::cout
            << "[MONO/Omnidata skip] kf=" << reference->fid_
            << " reason=aligned_sources available=" << source_depths.size()
            << " required=" << monocular_omnidata_min_source_views_ << "\n";
        return;
    }

    float cell_size = 0.0f;
    {
        torch::NoGradGuard no_grad;
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        const int level = voxel_model_->insertionOctreeLevel();
        if (level > 0 && level < 30) {
            const float extent = voxel_model_->SceneExtent()
                .detach().to(torch::kCPU).item<float>();
            cell_size = extent / static_cast<float>(1 << level);
        }
    }
    const Eigen::Matrix3f reference_k = omnidataIntrinsics(
        *reference,
        monocular_omnidata_width_,
        monocular_omnidata_height_);
    const Eigen::Matrix4f reference_c2w =
        reference->getPosef().inverse().matrix();
    cv::Mat filtered_depth = filterBySourceConsistency(
        reference_aligned->depth,
        reference_k,
        reference_c2w,
        source_depths,
        source_intrinsics,
        source_w2c,
        monocular_omnidata_min_source_views_,
        monocular_omnidata_consistency_rel_tol_,
        monocular_omnidata_consistency_vox_ * cell_size);
    const int consistent_pixels = cv::countNonZero(filtered_depth > 0.0f);
    if (consistent_pixels <= 0) {
        std::cout
            << "[MONO/Omnidata skip] kf=" << reference->fid_
            << " reason=multiview_consistency\n";
        return;
    }

    const float alignment_confidence = std::clamp(
        1.0f - reference_aligned->median_relative_error /
            std::max(
                monocular_omnidata_max_alignment_rel_error_,
                1.0e-6f),
        0.0f,
        1.0f);
    cv::Mat supervision_confidence = cv::Mat::zeros(
        filtered_depth.rows, filtered_depth.cols, CV_32FC1);
    supervision_confidence.setTo(
        alignment_confidence,
        filtered_depth > 0.0f);
    cacheMonocularDepthPrior(
        reference,
        filtered_depth,
        supervision_confidence,
        sv::LearnedDepthSource::Omnidata);

    monocular_mvs_pending_reference_ = reference;
    monocular_mvs_pending_c2w_ = reference_c2w;
    monocular_mvs_pending_depth_min_ =
        reference->monocular_mvs_depth_min_;
    monocular_mvs_pending_depth_max_ =
        reference->monocular_mvs_depth_max_;
    monocular_mvs_pending_camera_ = reference->toMiniCam(
        monocular_omnidata_height_, monocular_omnidata_width_);
    setOmnidataCameraSnapshot(
        monocular_mvs_pending_camera_,
        reference_c2w,
        reference_k,
        monocular_omnidata_width_,
        monocular_omnidata_height_,
        static_cast<int>(reference->fid_));
    monocular_mvs_pending_reference_rgb_ = resizeRgb(
        toOmnidataRgb(reference->img_undist_),
        monocular_omnidata_width_,
        monocular_omnidata_height_);

    try {
        integrateMonocularLearnedDepth(
            filtered_depth,
            "Omnidata",
            "world/monocular_omnidata/created",
            /*clear_cuda_cache_before_insertion=*/true);
    } catch (...) {
        monocular_mvs_pending_reference_.reset();
        monocular_mvs_pending_reference_rgb_.release();
        monocular_mvs_pending_depth_min_ = 0.0f;
        monocular_mvs_pending_depth_max_ = 0.0f;
        throw;
    }
    monocular_mvs_pending_reference_.reset();
    monocular_mvs_pending_reference_rgb_.release();
    monocular_mvs_pending_depth_min_ = 0.0f;
    monocular_mvs_pending_depth_max_ = 0.0f;

    std::cout
        << "[MONO/Omnidata aligned] kf=" << reference->fid_
        << " anchors=" << reference_aligned->anchor_count
        << " median_rel_error="
        << reference_aligned->median_relative_error
        << " aligned_sources=" << source_depths.size()
        << " consistent_pixels=" << consistent_pixels << "\n";
}
