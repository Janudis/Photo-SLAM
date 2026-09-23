#include "include_voxel/voxel_mapper_supervision.h"
#include "include_voxel/voxel_mapper.h"
#include "include_voxel/voxel_mapper_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <regex>
#include <sstream>
#include <vector>

#include "ORB-SLAM3/include/Atlas.h"
#include "ORB-SLAM3/include/MapPoint.h"
#include "ORB-SLAM3/include/System.h"

namespace {

torch::Tensor supervisionZero(const torch::Device& device)
{
    return torch::zeros(
        {1},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
}

float supervisionScheduleMultiplier(
    const int iteration,
    const int iter_from,
    const int iter_end,
    const float end_multiplier)
{
    if (iter_end <= iter_from || end_multiplier == 1.0f) {
        return 1.0f;
    }
    const float ratio = std::clamp(
        static_cast<float>(iteration - iter_from) /
            static_cast<float>(iter_end - iter_from),
        0.0f,
        1.0f);
    return std::pow(end_multiplier, ratio);
}

torch::Tensor floatMatToTensor(
    const cv::Mat& input,
    const torch::Device& device)
{
    if (input.empty() || input.type() != CV_32FC1) {
        return torch::Tensor();
    }
    const cv::Mat continuous = input.isContinuous() ? input : input.clone();
    return torch::from_blob(
               const_cast<float*>(continuous.ptr<float>()),
               {continuous.rows, continuous.cols},
               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
        .clone()
        .to(device, torch::kFloat32)
        .contiguous();
}

torch::Tensor resizeMapBilinear(
    const torch::Tensor& map,
    const int64_t height,
    const int64_t width)
{
    if (!map.defined() || map.dim() != 2) {
        return torch::Tensor();
    }
    if (map.size(0) == height && map.size(1) == width) {
        return map;
    }
    return torch::nn::functional::interpolate(
               map.unsqueeze(0).unsqueeze(0),
               torch::nn::functional::InterpolateFuncOptions()
                   .size(std::vector<int64_t>{height, width})
                   .mode(torch::kBilinear)
                   .align_corners(false))
        .squeeze(0)
        .squeeze(0);
}

torch::Tensor validDepthContinuityMask(
    const torch::Tensor& depth,
    const torch::Tensor& valid,
    int kernel_size,
    const float max_relative_jump)
{
    using namespace torch::indexing;

    torch::Tensor support = voxel_eval::validDepthSupportMask(
        valid, kernel_size);
    if (max_relative_jump <= 0.0f || depth.dim() != 2 ||
        !support.any().item<bool>()) {
        return support;
    }

    kernel_size = std::max(3, kernel_size);
    if ((kernel_size % 2) == 0) {
        ++kernel_size;
    }
    const int64_t height = depth.size(0);
    const int64_t width = depth.size(1);
    const int64_t pad = kernel_size / 2;
    if (height <= 2 * pad || width <= 2 * pad) {
        return torch::zeros_like(
            valid, torch::TensorOptions().dtype(torch::kBool));
    }

    const torch::Tensor center = depth.index({
        Slice(pad, height - pad),
        Slice(pad, width - pad)});
    torch::Tensor continuous = support.index({
        Slice(pad, height - pad),
        Slice(pad, width - pad)}).clone();
    const torch::Tensor tolerance =
        max_relative_jump * center.abs().clamp_min(1.0e-6f);
    for (int64_t dy = -pad; dy <= pad; ++dy) {
        for (int64_t dx = -pad; dx <= pad; ++dx) {
            const torch::Tensor neighbor = depth.index({
                Slice(pad + dy, height - pad + dy),
                Slice(pad + dx, width - pad + dx)});
            continuous = continuous & ((neighbor - center).abs() <= tolerance);
        }
    }

    torch::Tensor result = torch::zeros_like(
        valid, torch::TensorOptions().dtype(torch::kBool));
    result.index_put_({
        Slice(pad, height - pad),
        Slice(pad, width - pad)}, continuous);
    return result;
}

bool learnedDepthPriorToTensors(
    const std::shared_ptr<VoxelKeyframe>& keyframe,
    const torch::Device& device,
    torch::Tensor& depth,
    torch::Tensor& confidence)
{
    depth = torch::Tensor();
    confidence = torch::Tensor();
    if (!keyframe || keyframe->monocular_depth_prior_.empty() ||
        keyframe->monocular_depth_confidence_.empty() ||
        keyframe->monocular_depth_prior_.size() !=
            keyframe->monocular_depth_confidence_.size()) {
        return false;
    }
    depth = floatMatToTensor(keyframe->monocular_depth_prior_, device);
    confidence = floatMatToTensor(
        keyframe->monocular_depth_confidence_, device);
    return depth.defined() && confidence.defined() &&
           depth.sizes() == confidence.sizes();
}

bool renderAlphaMap(
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    torch::Tensor& alpha)
{
    auto it = render_pkg.find("raw_T");
    if (it == render_pkg.end() || !it->second.defined()) {
        it = render_pkg.find("T");
    }
    if (it == render_pkg.end() || !it->second.defined()) {
        alpha = torch::Tensor();
        return false;
    }
    torch::Tensor transmittance =
        voxel_eval::transmittanceTensorToEvalMap(it->second);
    if (!transmittance.defined()) {
        alpha = torch::Tensor();
        return false;
    }
    alpha = (1.0f - transmittance).clamp(0.0f, 1.0f);
    return true;
}

} // namespace
torch::Tensor VoxelMapper::computeMonocularDepthLoss(
    const std::shared_ptr<VoxelKeyframe>& keyframe,
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    const int iteration)
{
    torch::Tensor zero = supervisionZero(mDevice);
    if (sensor_type_ != MONOCULAR ||
        opt_params_.lambda_monocular_depth_ <= 0.0f ||
        iteration < opt_params_.monocular_depth_from_ ||
        iteration > opt_params_.monocular_depth_end_ ||
        !keyframe ||
        keyframe->monocular_depth_source_ ==
            sv::LearnedDepthSource::None) {
        return zero;
    }

    torch::Tensor target_depth;
    torch::Tensor confidence;
    if (!learnedDepthPriorToTensors(
            keyframe, mDevice, target_depth, confidence)) {
        return zero;
    }

    auto depth_it = render_pkg.find("raw_depth");
    if (depth_it == render_pkg.end() || !depth_it->second.defined()) {
        depth_it = render_pkg.find("depth");
    }
    torch::Tensor rendered_depth_numerator;
    torch::Tensor alpha;
    if (depth_it == render_pkg.end() || !depth_it->second.defined() ||
        !renderAlphaMap(render_pkg, alpha)) {
        return zero;
    }
    rendered_depth_numerator = voxel_eval::tensorToEvalMapExactChannel(
        depth_it->second, 0);
    rendered_depth_numerator = resizeMapBilinear(
        rendered_depth_numerator,
        target_depth.size(0),
        target_depth.size(1));
    alpha = resizeMapBilinear(
        alpha, target_depth.size(0), target_depth.size(1));
    if (!rendered_depth_numerator.defined() || !alpha.defined()) {
        return zero;
    }
    // Match SVRecon's MASt3R metric-depth loss: resize accumulated depth and
    // alpha independently, then recover the alpha-normalized camera-Z depth.
    const torch::Tensor rendered_depth =
        rendered_depth_numerator / alpha.clamp_min(1.0e-4f);

    const float near_depth = std::max(1.0e-6f, keyframe->znear_);
    torch::Tensor valid =
        torch::isfinite(target_depth) &
        torch::isfinite(rendered_depth) &
        torch::isfinite(confidence) &
        (target_depth > near_depth) &
        (rendered_depth > near_depth) &
        (confidence > sv::kMonocularDepthConfidenceMin) &
        (alpha.detach() > sv::kMonocularDepthAlphaMin);
    if (!valid.any().item<bool>()) {
        return zero;
    }

    // SVRecon's MASt3R regularizer uses alpha-normalized metric depth and a
    // Cauchy residual. Confidence normalization keeps sparse accepted priors
    // comparable across MVS keyframes.
    torch::Tensor weights = confidence.clamp(0.0f, 1.0f).index({valid});
    const torch::Tensor residual =
        rendered_depth.index({valid}) - target_depth.index({valid});
    torch::Tensor loss =
        (weights * torch::log1p(residual.square())).sum() /
        weights.sum().clamp_min(1.0e-6f);
    return loss * supervisionScheduleMultiplier(
        iteration,
        opt_params_.monocular_depth_from_,
        opt_params_.monocular_depth_end_,
        sv::kMonocularDepthEndMultiplier);
}

torch::Tensor VoxelMapper::computeMonocularNormalLoss(
    const std::shared_ptr<VoxelKeyframe>& keyframe,
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    const int iteration)
{
    torch::Tensor zero = supervisionZero(mDevice);
    if (sensor_type_ != MONOCULAR ||
        sv::kLambdaMonocularNormal <= 0.0f ||
        iteration < sv::kMonocularNormalFrom ||
        iteration > sv::kMonocularNormalEnd ||
        !keyframe ||
        keyframe->monocular_depth_source_ ==
            sv::LearnedDepthSource::None) {
        return zero;
    }

    torch::Tensor target_depth;
    torch::Tensor confidence;
    if (!learnedDepthPriorToTensors(
            keyframe, mDevice, target_depth, confidence)) {
        return zero;
    }

    auto normal_it = render_pkg.find("raw_normal");
    if (normal_it == render_pkg.end() || !normal_it->second.defined()) {
        normal_it = render_pkg.find("normal");
    }
    torch::Tensor alpha;
    if (normal_it == render_pkg.end() || !normal_it->second.defined() ||
        !renderAlphaMap(render_pkg, alpha)) {
        return zero;
    }

    torch::Tensor rendered_normal =
        normal_it->second.to(mDevice, torch::kFloat32).contiguous();
    if (rendered_normal.dim() == 4 && rendered_normal.size(0) == 1) {
        rendered_normal = rendered_normal.squeeze(0);
    }
    if (rendered_normal.dim() != 3 || rendered_normal.size(0) < 3) {
        return zero;
    }
    if (rendered_normal.size(0) > 3) {
        rendered_normal = rendered_normal.index({
            torch::indexing::Slice(0, 3)}).contiguous();
    }

    const int64_t height = target_depth.size(0);
    const int64_t width = target_depth.size(1);
    if (rendered_normal.size(1) != height ||
        rendered_normal.size(2) != width) {
        rendered_normal = torch::nn::functional::interpolate(
            rendered_normal.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{height, width})
                .mode(torch::kBilinear)
                .align_corners(false)).squeeze(0);
    }
    alpha = resizeMapBilinear(alpha, height, width);
    if (!alpha.defined()) {
        return zero;
    }

    torch::Tensor target_normal;
    torch::Tensor target_valid;
    {
        torch::NoGradGuard no_grad;
        const float near_depth = std::max(1.0e-6f, keyframe->znear_);
        torch::Tensor valid_depth =
            torch::isfinite(target_depth) &
            torch::isfinite(confidence) &
            (target_depth > near_depth) &
            (confidence > sv::kMonocularDepthConfidenceMin);
        target_valid = validDepthContinuityMask(
            target_depth,
            valid_depth,
            sv::kMonocularNormalKernelSize,
            sv::kMonocularNormalMaxDepthJumpRelative);
        if (!target_valid.any().item<bool>()) {
            return zero;
        }

        const float tolerance_cos = std::cos(
            sv::kMonocularNormalToleranceDegrees *
            static_cast<float>(M_PI) / 180.0f);
        const sv::MiniCam prior_camera = keyframe->toMiniCam(
            static_cast<int>(height), static_cast<int>(width));
        // SVRecon flips depth-derived normals to face the observing camera
        // before comparing them with its differentiable rendered normals.
        target_normal = -voxel_eval::depthToNormal(
            prior_camera,
            target_depth.clamp_min(near_depth),
            sv::kMonocularNormalKernelSize,
            tolerance_cos);
        target_valid = target_valid & (target_normal != 0).any(0);
    }

    const torch::Tensor rendered_magnitude =
        rendered_normal.square().sum(0).sqrt();
    torch::Tensor valid =
        target_valid &
        torch::isfinite(rendered_normal).all(0) &
        (rendered_magnitude > 1.0e-6f) &
        (alpha.detach() > sv::kMonocularDepthAlphaMin);
    if (!valid.any().item<bool>()) {
        return zero;
    }

    rendered_normal = torch::nn::functional::normalize(
        rendered_normal,
        torch::nn::functional::NormalizeFuncOptions()
            .dim(0)
            .eps(1.0e-8));
    target_normal = torch::nn::functional::normalize(
        target_normal,
        torch::nn::functional::NormalizeFuncOptions()
            .dim(0)
            .eps(1.0e-8));
    const torch::Tensor cosine_error =
        1.0f - (rendered_normal * target_normal)
                   .sum(0)
                   .clamp(-1.0f, 1.0f);
    const torch::Tensor weights =
        confidence.clamp(0.0f, 1.0f).index({valid});
    torch::Tensor loss =
        (weights * cosine_error.index({valid})).sum() /
        weights.sum().clamp_min(1.0e-6f);
    return loss * supervisionScheduleMultiplier(
        iteration,
        sv::kMonocularNormalFrom,
        sv::kMonocularNormalEnd,
        sv::kMonocularNormalEndMultiplier);
}
