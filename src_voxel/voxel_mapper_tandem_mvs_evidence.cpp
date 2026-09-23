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
void VoxelMapper::integrateMonocularMvsDepth(
    const sv::TandemMvsResult& result)
{
    const std::shared_ptr<VoxelKeyframe> reference =
        monocular_mvs_pending_reference_;
    if (reference && !result.depth_dense.empty() &&
        result.depth_dense.type() == CV_32FC1) {
        reference->monocular_mvs_depth_dense_ = result.depth_dense.clone();
    }
    cacheMonocularDepthPrior(
        reference,
        result.depth,
        result.confidence,
        sv::LearnedDepthSource::TandemMvs);
    if (monocular_mvs_tsdf_evidence_) {
        integrateMonocularMvsTsdfEvidence(result);
    } else {
        integrateMonocularMvsSurfaceDepth(result.depth);
    }
    if (reference) {
        markSurfaceViewPruningPending({reference});
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

sv::MonocularMvsPruneEvidence
VoxelMapper::computeMonocularMvsPruneEvidence(
    const torch::Tensor& centers_world_in,
    const torch::Tensor& sizes_world_in)
{
    sv::MonocularMvsPruneEvidence result;
    if (!opt_params_.prune_mvs_consistency_enable_ || !scene_ ||
        !centers_world_in.defined() || !sizes_world_in.defined() ||
        centers_world_in.dim() != 2 || centers_world_in.size(1) != 3 ||
        centers_world_in.size(0) <= 0 ||
        sizes_world_in.size(0) != centers_world_in.size(0)) {
        return result;
    }

    torch::NoGradGuard no_grad;
    const int64_t voxel_count = centers_world_in.size(0);
    const torch::Device device = centers_world_in.device();
    const auto bool_options =
        torch::TensorOptions().dtype(torch::kBool).device(device);
    const auto count_options =
        torch::TensorOptions().dtype(torch::kInt32).device(device);

    result.supported = torch::zeros({voxel_count}, bool_options);
    result.free_space = torch::zeros({voxel_count}, bool_options);

    torch::Tensor centers_world =
        centers_world_in.detach().to(device).to(torch::kFloat32).contiguous();
    torch::Tensor sizes_world =
        sizes_world_in.detach().to(device).to(torch::kFloat32)
            .reshape({voxel_count}).contiguous();
    torch::Tensor support_count = torch::zeros({voxel_count}, count_options);
    torch::Tensor contradiction_count =
        torch::zeros({voxel_count}, count_options);

    const auto keyframes = scene_->getAllKeyframes();
    const cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
    constexpr int64_t kProjectionChunkSize = 262144;

    for (const auto& [keyframe_id, keyframe] : keyframes) {
        (void)keyframe_id;
        if (!keyframe ||
            keyframe->monocular_depth_source_ !=
                sv::LearnedDepthSource::TandemMvs ||
            keyframe->monocular_depth_prior_.empty() ||
            keyframe->monocular_depth_prior_.type() != CV_32FC1) {
            continue;
        }

        const cv::Mat& depth = keyframe->monocular_depth_prior_;
        const cv::Mat& confidence = keyframe->monocular_depth_confidence_;
        if (confidence.empty() || confidence.type() != CV_32FC1 ||
            confidence.size() != depth.size()) {
            continue;
        }

        // A 3x3 valid, locally smooth neighborhood prevents depth edges from
        // protecting or carving a whole octree cell.
        cv::Mat valid(depth.rows, depth.cols, CV_8UC1, cv::Scalar(0));
        cv::Mat depth_for_min(
            depth.rows,
            depth.cols,
            CV_32FC1,
            cv::Scalar(std::numeric_limits<float>::max()));
        cv::Mat depth_for_max(
            depth.rows,
            depth.cols,
            CV_32FC1,
            cv::Scalar(std::numeric_limits<float>::lowest()));
        int64_t valid_pixels = 0;
        for (int y = 0; y < depth.rows; ++y) {
            const float* depth_row = depth.ptr<float>(y);
            const float* confidence_row = confidence.ptr<float>(y);
            std::uint8_t* valid_row = valid.ptr<std::uint8_t>(y);
            float* min_row = depth_for_min.ptr<float>(y);
            float* max_row = depth_for_max.ptr<float>(y);
            for (int x = 0; x < depth.cols; ++x) {
                const float d = depth_row[x];
                const float c = confidence_row[x];
                if (!std::isfinite(d) || d <= 1.0e-6f ||
                    !std::isfinite(c) ||
                    c <= sv::kMonocularDepthConfidenceMin) {
                    continue;
                }
                valid_row[x] = 255;
                min_row[x] = d;
                max_row[x] = d;
                ++valid_pixels;
            }
        }
        if (valid_pixels == 0) {
            continue;
        }

        cv::Mat neighborhood_valid;
        cv::Mat local_min;
        cv::Mat local_max;
        cv::erode(
            valid,
            neighborhood_valid,
            kernel,
            cv::Point(-1, -1),
            1,
            cv::BORDER_CONSTANT,
            cv::Scalar(0));
        cv::erode(
            depth_for_min,
            local_min,
            kernel,
            cv::Point(-1, -1),
            1,
            cv::BORDER_CONSTANT,
            cv::Scalar(std::numeric_limits<float>::max()));
        cv::dilate(
            depth_for_max,
            local_max,
            kernel,
            cv::Point(-1, -1),
            1,
            cv::BORDER_CONSTANT,
            cv::Scalar(std::numeric_limits<float>::lowest()));
        cv::Mat local_span = cv::Mat::zeros(depth.size(), CV_32FC1);
        for (int y = 0; y < depth.rows; ++y) {
            const std::uint8_t* valid_row =
                neighborhood_valid.ptr<std::uint8_t>(y);
            const float* min_row = local_min.ptr<float>(y);
            const float* max_row = local_max.ptr<float>(y);
            float* span_row = local_span.ptr<float>(y);
            for (int x = 0; x < depth.cols; ++x) {
                if (valid_row[x] != 0) {
                    span_row[x] = std::max(0.0f, max_row[x] - min_row[x]);
                }
            }
        }

        torch::Tensor depth_map = torch::from_blob(
            depth.data,
            {depth.rows, depth.cols},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                .clone()
                .to(device)
                .reshape({-1});
        torch::Tensor confidence_map = torch::from_blob(
            confidence.data,
            {confidence.rows, confidence.cols},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                .clone()
                .to(device)
                .reshape({-1});
        torch::Tensor valid_map = torch::from_blob(
            neighborhood_valid.data,
            {neighborhood_valid.rows, neighborhood_valid.cols},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
                .clone()
                .to(device)
                .to(torch::kBool)
                .reshape({-1});
        torch::Tensor span_map = torch::from_blob(
            local_span.data,
            {local_span.rows, local_span.cols},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                .clone()
                .to(device)
                .reshape({-1});

        const Eigen::Matrix4f Tcw = keyframe->getPosef().matrix();
        Eigen::Matrix<float, 3, 4, Eigen::RowMajor> Tcw_3x4 =
            Tcw.block<3, 4>(0, 0);
        torch::Tensor transform = torch::from_blob(
            Tcw_3x4.data(),
            {3, 4},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                .clone()
                .to(device);
        torch::Tensor rotation = transform.index({
            torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
        torch::Tensor translation = transform.index({
            torch::indexing::Slice(), 3});
        const float camera_z_extent_scale =
            0.5f * Tcw.block<1, 3>(2, 0).cwiseAbs().sum();
        const Eigen::Matrix3f K =
            resizedIntrinsics(*keyframe, depth.cols, depth.rows);

        for (int64_t begin = 0; begin < voxel_count;
             begin += kProjectionChunkSize) {
            const int64_t end = std::min(
                voxel_count, begin + kProjectionChunkSize);
            const auto slice = torch::indexing::Slice(begin, end);
            torch::Tensor camera_xyz =
                torch::matmul(
                    centers_world.index({slice}),
                    rotation.transpose(0, 1)) +
                translation;
            torch::Tensor z = camera_xyz.index({torch::indexing::Slice(), 2});
            torch::Tensor positive_z = z > 1.0e-6f;
            torch::Tensor safe_z = torch::where(
                positive_z, z, torch::ones_like(z));
            torch::Tensor u_float =
                K(0, 0) *
                    camera_xyz.index({torch::indexing::Slice(), 0}) / safe_z +
                K(0, 2);
            torch::Tensor v_float =
                K(1, 1) *
                    camera_xyz.index({torch::indexing::Slice(), 1}) / safe_z +
                K(1, 2);
            torch::Tensor in_image =
                positive_z & torch::isfinite(u_float) & torch::isfinite(v_float) &
                (u_float >= 0.0f) &
                (u_float <= static_cast<float>(depth.cols - 1)) &
                (v_float >= 0.0f) &
                (v_float <= static_cast<float>(depth.rows - 1));
            torch::Tensor u =
                u_float.round().clamp(0, depth.cols - 1).to(torch::kLong);
            torch::Tensor v =
                v_float.round().clamp(0, depth.rows - 1).to(torch::kLong);
            torch::Tensor pixel_index = v * depth.cols + u;

            torch::Tensor measured = depth_map.index_select(0, pixel_index);
            torch::Tensor measured_confidence =
                confidence_map.index_select(0, pixel_index);
            torch::Tensor neighborhood_is_valid =
                valid_map.index_select(0, pixel_index);
            torch::Tensor local_depth_span =
                span_map.index_select(0, pixel_index);
            torch::Tensor voxel_size = sizes_world.index({slice});
            torch::Tensor tolerance =
                sv::kPruneMvsDepthToleranceVox * voxel_size;
            torch::Tensor observation_valid =
                in_image & neighborhood_is_valid & torch::isfinite(measured) &
                (measured > 1.0e-6f) &
                torch::isfinite(measured_confidence) &
                (measured_confidence >
                 sv::kMonocularDepthConfidenceMin) &
                (local_depth_span <= 2.0f * tolerance);

            torch::Tensor z_radius =
                camera_z_extent_scale * voxel_size;
            torch::Tensor z_min = z - z_radius;
            torch::Tensor z_max = z + z_radius;
            torch::Tensor supports_surface =
                observation_valid & (z_min <= measured + tolerance) &
                (z_max >= measured - tolerance);
            torch::Tensor contradicts_free_space =
                observation_valid & (z_max < measured - tolerance);

            support_count.index({slice}).add_(
                supports_surface.to(torch::kInt32));
            contradiction_count.index({slice}).add_(
                contradicts_free_space.to(torch::kInt32));
        }
    }

    result.supported =
        (support_count >= sv::kPruneMvsMinSupportingViews)
            .to(torch::kBool)
            .contiguous();
    result.free_space =
        ((contradiction_count >=
          sv::kPruneMvsMinContradictingViews) &
         (support_count == 0))
            .to(torch::kBool)
            .contiguous();
    return result;
}
