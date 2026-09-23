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

namespace voxel_eval {
namespace {

torch::Tensor gaussianWindow1d(
    const int window_size,
    const float sigma,
    const torch::DeviceType device_type)
{
    std::vector<float> values(window_size);
    for (int x = 0; x < window_size; ++x) {
        const int offset = x - window_size / 2;
        values[x] = std::exp(
            -offset * offset / (2.0f * sigma * sigma));
    }
    torch::Tensor gaussian = torch::tensor(
        values, torch::TensorOptions().device(device_type));
    return gaussian / gaussian.sum();
}

torch::Tensor createSsimWindow(
    const int window_size,
    const int64_t channels,
    const torch::DeviceType device_type)
{
    torch::Tensor window_1d =
        gaussianWindow1d(window_size, 1.5f, device_type).unsqueeze(1);
    torch::Tensor window_2d =
        window_1d.mm(window_1d.t()).to(torch::kFloat32)
            .unsqueeze(0).unsqueeze(0);
    return window_2d.expand(
        {channels, 1, window_size, window_size}).contiguous();
}

torch::Tensor ssimWithWindow(
    const torch::Tensor& image_1,
    const torch::Tensor& image_2,
    const torch::Tensor& window,
    const int window_size,
    const int64_t channels,
    const bool size_average)
{
    const int padding = window_size / 2;
    const auto options = torch::nn::functional::Conv2dFuncOptions()
        .padding(padding).groups(channels);
    torch::Tensor mu_1 = torch::nn::functional::conv2d(
        image_1, window, options);
    torch::Tensor mu_2 = torch::nn::functional::conv2d(
        image_2, window, options);
    torch::Tensor mu_1_sq = mu_1.square();
    torch::Tensor mu_2_sq = mu_2.square();
    torch::Tensor mu_1_mu_2 = mu_1 * mu_2;
    torch::Tensor sigma_1_sq = torch::nn::functional::conv2d(
        image_1 * image_1, window, options) - mu_1_sq;
    torch::Tensor sigma_2_sq = torch::nn::functional::conv2d(
        image_2 * image_2, window, options) - mu_2_sq;
    torch::Tensor sigma_12 = torch::nn::functional::conv2d(
        image_1 * image_2, window, options) - mu_1_mu_2;
    constexpr double kC1 = 0.01 * 0.01;
    constexpr double kC2 = 0.03 * 0.03;
    torch::Tensor ssim_map =
        ((2.0 * mu_1_mu_2 + kC1) * (2.0 * sigma_12 + kC2)) /
        ((mu_1_sq + mu_2_sq + kC1) *
         (sigma_1_sq + sigma_2_sq + kC2));
    return size_average
        ? ssim_map.mean()
        : ssim_map.mean(1).mean(1).mean(1);
}

torch::Tensor sparseLossMap2d(
    const torch::Tensor& input,
    const char* name)
{
    torch::Tensor map = input;
    if (map.dim() == 4 && map.size(0) == 1 && map.size(1) == 1) {
        return map.index({0, 0});
    }
    if (map.dim() == 3 && (map.size(0) == 1 || map.size(0) == 3)) {
        return map.index({0});
    }
    TORCH_CHECK(
        map.dim() == 2,
        name,
        " must have shape [H,W], [1,H,W], [3,H,W], or [1,1,H,W]; got ",
        map.sizes());
    return map;
}

} // namespace

torch::Tensor l1Loss(
    const torch::Tensor& prediction,
    const torch::Tensor& target)
{
    return torch::abs(prediction - target).mean();
}

torch::Tensor mseLoss(
    const torch::Tensor& prediction,
    const torch::Tensor& target)
{
    return torch::nn::functional::mse_loss(prediction, target);
}

torch::Tensor huberLoss(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    const float threshold)
{
    torch::Tensor absolute_error = (prediction - target).abs();
    torch::Tensor l1 = absolute_error.mean(0);
    torch::Tensor l2 = absolute_error.square().mean(0);
    return torch::where(
        l1 < threshold,
        l2,
        2.0f * threshold * l1 - threshold * threshold).mean();
}

torch::Tensor psnr(
    const torch::Tensor& prediction,
    const torch::Tensor& target)
{
    torch::Tensor mse = (prediction - target).square().mean();
    return 10.0f * torch::log10(1.0f / mse);
}

torch::Tensor ssim(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    const torch::DeviceType device_type,
    const int window_size,
    const bool size_average)
{
    const int64_t channels = prediction.size(-3);
    torch::Tensor window = createSsimWindow(
        window_size, channels, device_type).type_as(prediction);
    return ssimWithWindow(
        prediction,
        target,
        window,
        window_size,
        channels,
        size_average);
}

torch::Tensor fastSsimLoss(
    torch::Tensor prediction,
    torch::Tensor target)
{
    if (prediction.dim() == 3) {
        prediction = prediction.unsqueeze(0);
        target = target.unsqueeze(0);
    }
    const torch::DeviceType device_type =
        prediction.is_cuda() ? torch::kCUDA : torch::kCPU;
    return 1.0f - ssim(prediction, target, device_type);
}

torch::Tensor probabilityConcentrationLoss(const torch::Tensor& probability)
{
    return (probability.square() * (1.0f - probability).square()).mean();
}

torch::Tensor sparseDepthLoss(
    const torch::Tensor& raw_transmittance,
    const torch::Tensor& raw_depth,
    const torch::Tensor& sparse_uv,
    const torch::Tensor& sparse_depth)
{
    TORCH_CHECK(raw_transmittance.defined(), "raw transmittance is undefined");
    TORCH_CHECK(raw_depth.defined(), "raw depth is undefined");
    TORCH_CHECK(sparse_uv.defined(), "sparse UV coordinates are undefined");
    TORCH_CHECK(sparse_depth.defined(), "sparse depth is undefined");

    torch::Tensor depth = sparseLossMap2d(raw_depth, "raw depth");
    torch::Tensor transmittance = sparseLossMap2d(
        raw_transmittance, "raw transmittance");
    depth = depth / (1.0f - transmittance).clamp_min(1.0e-4f);

    TORCH_CHECK(
        sparse_uv.dim() == 2 && sparse_uv.size(1) == 2,
        "sparse UV coordinates must have shape [N,2], got ",
        sparse_uv.sizes());
    torch::Tensor target_depth = sparse_depth;
    if (target_depth.dim() == 2 && target_depth.size(1) == 1) {
        target_depth = target_depth.squeeze(1);
    }
    TORCH_CHECK(
        target_depth.dim() == 1,
        "sparse depth must have shape [N] or [N,1], got ",
        sparse_depth.sizes());
    TORCH_CHECK(
        target_depth.size(0) == sparse_uv.size(0),
        "sparse UV and depth counts differ: ",
        sparse_uv.size(0), " vs ", target_depth.size(0));

    torch::Tensor sampled_depth = torch::nn::functional::grid_sample(
        depth.unsqueeze(0).unsqueeze(0),
        sparse_uv.unsqueeze(0).unsqueeze(0),
        torch::nn::functional::GridSampleFuncOptions()
            .mode(torch::kBilinear)
            .padding_mode(torch::kZeros)
            .align_corners(false)).squeeze();
    return torch::nn::functional::smooth_l1_loss(
        sampled_depth,
        target_depth,
        torch::nn::functional::SmoothL1LossFuncOptions()
            .reduction(torch::kMean));
}

torch::Tensor tensorToEvalMap(const torch::Tensor& tensor, int preferred_channel)
{
    if (!tensor.defined()) {
        return torch::Tensor();
    }

    torch::Tensor d = tensor.detach().to(torch::kFloat32);
    if (d.dim() == 4 && d.size(0) == 1) {
        d = d.squeeze(0);  // [1,C,H,W] -> [C,H,W]
    }
    if (d.dim() == 3) {
        const int64_t C = d.size(0);
        if (C == 1) {
            d = d.squeeze(0);  // [1,H,W] -> [H,W]
        } else if (C > preferred_channel && preferred_channel >= 0) {
            d = d.index({preferred_channel});  // preferred channel
        } else if (C >= 1) {
            d = d.index({0});  // fallback to channel 0
        } else {
            return torch::Tensor();
        }
    }
    if (d.dim() != 2) {
        return torch::Tensor();
    }
    return d.contiguous();
}

torch::Tensor tensorToEvalMapExactChannel(
    const torch::Tensor& tensor,
    int required_channel)
{
    if (!tensor.defined()) {
        return torch::Tensor();
    }

    torch::Tensor d = tensor.detach().to(torch::kFloat32);
    if (d.dim() == 4 && d.size(0) == 1) {
        d = d.squeeze(0);  // [1,C,H,W] -> [C,H,W]
    }
    if (d.dim() == 3) {
        const int64_t C = d.size(0);
        if (C == 1) {
            if (required_channel != 0) {
                return torch::Tensor();
            }
            d = d.squeeze(0);  // [1,H,W] -> [H,W]
        } else {
            if (required_channel < 0 || required_channel >= C) {
                return torch::Tensor();
            }
            d = d.index({required_channel});
        }
    }
    if (d.dim() != 2) {
        return torch::Tensor();
    }
    return d.contiguous();
}

torch::Tensor depthTensorToEvalMap(const torch::Tensor& depth_tensor)
{
    // For GT-vs-render depth visualization, prefer the surface-like
    // median depth channel when available. This matches rendered mesh/fusion
    // utilities better than the alpha-weighted mean-depth channel.
    return tensorToEvalMap(depth_tensor, /*preferred_channel=*/2);
}

torch::Tensor transmittanceTensorToEvalMap(const torch::Tensor& t_tensor)
{
    return tensorToEvalMap(t_tensor, /*preferred_channel=*/0);
}

bool renderPkgToMetricDepthForEval(
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    torch::Tensor& pred_depth)
{
    pred_depth = torch::Tensor();

    auto it_depth = render_pkg.find("depth");
    if (it_depth == render_pkg.end() || !it_depth->second.defined()) {
        it_depth = render_pkg.find("raw_depth");
        if (it_depth == render_pkg.end() || !it_depth->second.defined()) {
            return false;
        }
    }

    pred_depth = depthTensorToEvalMap(it_depth->second);
    if (!pred_depth.defined()) {
        return false;
    }

    return true;
}

bool renderPkgToSparseDepthLossMap(
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    torch::Tensor& pred_depth)
{
    pred_depth = torch::Tensor();

    auto it_T = render_pkg.find("raw_T");
    if (it_T == render_pkg.end() || !it_T->second.defined()) {
        it_T = render_pkg.find("T");
        if (it_T == render_pkg.end() || !it_T->second.defined()) {
            return false;
        }
    }

    auto it_depth = render_pkg.find("raw_depth");
    if (it_depth == render_pkg.end() || !it_depth->second.defined()) {
        it_depth = render_pkg.find("depth");
        if (it_depth == render_pkg.end() || !it_depth->second.defined()) {
            return false;
        }
    }

    torch::Tensor raw_depth = tensorToEvalMapExactChannel(it_depth->second, 0);
    torch::Tensor raw_T = transmittanceTensorToEvalMap(it_T->second);
    if (!raw_depth.defined() || !raw_T.defined()) {
        return false;
    }
    if (raw_depth.sizes() != raw_T.sizes()) {
        return false;
    }

    pred_depth = raw_depth / (1.0f - raw_T).clamp_min(1e-4f);
    return pred_depth.defined();
}

bool computeSharedDepthVizRange(
    const torch::Tensor& pred_depth,
    const cv::Mat& gt_depth_meters,
    float valid_min_depth,
    float valid_max_depth,
    float& viz_min,
    float& viz_max)
{
    torch::Tensor pred = pred_depth.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor values = pred.masked_select(
        torch::isfinite(pred) &
        (pred > valid_min_depth) &
        (pred < valid_max_depth));

    if (!gt_depth_meters.empty()) {
        torch::Tensor gt = torch::from_blob(
            const_cast<float*>(gt_depth_meters.ptr<float>()),
            {gt_depth_meters.rows, gt_depth_meters.cols},
            torch::TensorOptions().dtype(torch::kFloat32)).clone();
        torch::Tensor gt_valid = gt.masked_select(
            torch::isfinite(gt) &
            (gt > valid_min_depth) &
            (gt < valid_max_depth));
        if (gt_valid.numel() > 0) {
            values = values.numel() > 0 ? torch::cat({values, gt_valid}, 0) : gt_valid;
        }
    }

    if (!values.defined() || values.numel() == 0) {
        return false;
    }

    if (values.numel() >= 10) {
        torch::Tensor q = torch::quantile(
            values,
            torch::tensor({0.03f, 0.97f}, torch::TensorOptions().dtype(torch::kFloat32)));
        viz_min = q[0].item<float>();
        viz_max = q[1].item<float>();
    } else {
        viz_min = values.min().item<float>();
        viz_max = values.max().item<float>();
    }

    if (!(viz_max > viz_min)) {
        viz_min = values.min().item<float>();
        viz_max = values.max().item<float>();
    }
    if (!(viz_max > viz_min)) {
        viz_max = viz_min + 1e-3f;
    }
    return true;
}

bool computeDepthScaleFitStats(
    const torch::Tensor& pred_depth,
    const cv::Mat& gt_depth_meters,
    float valid_min_depth,
    float valid_max_depth,
    DepthScaleFitStats& stats_out)
{
    stats_out = DepthScaleFitStats{};
    if (gt_depth_meters.empty()) {
        return false;
    }

    torch::Tensor pred = pred_depth.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor gt = torch::from_blob(
        const_cast<float*>(gt_depth_meters.ptr<float>()),
        {gt_depth_meters.rows, gt_depth_meters.cols},
        torch::TensorOptions().dtype(torch::kFloat32)).clone();

    torch::Tensor valid =
        torch::isfinite(pred) &
        torch::isfinite(gt) &
        (pred > valid_min_depth) &
        (pred < valid_max_depth) &
        (gt > valid_min_depth) &
        (gt < valid_max_depth);

    stats_out.overlap_count = valid.sum().item<int64_t>();
    if (stats_out.overlap_count > 0) {
        torch::Tensor pred_valid = pred.masked_select(valid);
        torch::Tensor gt_valid = gt.masked_select(valid);
        stats_out.pred_min = pred_valid.min().item<float>();
        stats_out.pred_max = pred_valid.max().item<float>();
        stats_out.gt_min = gt_valid.min().item<float>();
        stats_out.gt_max = gt_valid.max().item<float>();
    }

    if (stats_out.overlap_count < 100) {
        return false;
    }

    torch::Tensor ratio = gt.masked_select(valid) / pred.masked_select(valid).clamp_min(1e-6f);
    ratio = ratio.masked_select(torch::isfinite(ratio) & (ratio > 0.0f));
    stats_out.ratio_count_before_trim = ratio.numel();
    if (stats_out.ratio_count_before_trim < 100) {
        return false;
    }

    if (ratio.numel() >= 10) {
        torch::Tensor q = torch::quantile(
            ratio,
            torch::tensor(
                {0.05f, 0.25f, 0.50f, 0.75f, 0.95f},
                torch::TensorOptions().dtype(torch::kFloat32)));
        const float q_lo = q[0].item<float>();
        const float q_hi = q[4].item<float>();
        stats_out.ratio_q05 = q_lo;
        stats_out.ratio_q25 = q[1].item<float>();
        stats_out.ratio_q50 = q[2].item<float>();
        stats_out.ratio_q75 = q[3].item<float>();
        stats_out.ratio_q95 = q_hi;
        ratio = ratio.masked_select((ratio >= q_lo) & (ratio <= q_hi));
    }
    stats_out.ratio_count_after_trim = ratio.numel();
    if (stats_out.ratio_count_after_trim == 0) {
        return false;
    }

    const float scale = ratio.median().item<float>();
    if (!std::isfinite(scale) || scale <= 0.0f) {
        return false;
    }

    stats_out.scale = scale;
    stats_out.valid = true;
    return true;
}

bool computeWeightedMedianScale(
    const std::vector<std::pair<float, double>>& weighted_scales,
    float& scale_out)
{
    scale_out = 1.0f;
    if (weighted_scales.empty()) {
        return false;
    }

    std::vector<std::pair<float, double>> sorted = weighted_scales;
    std::sort(
        sorted.begin(),
        sorted.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    double total_weight = 0.0;
    for (const auto& item : sorted) {
        total_weight += std::max(0.0, item.second);
    }
    if (!(total_weight > 0.0)) {
        return false;
    }

    const double half_weight = 0.5 * total_weight;
    double accum_weight = 0.0;
    for (const auto& item : sorted) {
        accum_weight += std::max(0.0, item.second);
        if (accum_weight >= half_weight) {
            scale_out = item.first;
            return std::isfinite(scale_out) && scale_out > 0.0f;
        }
    }

    scale_out = sorted.back().first;
    return std::isfinite(scale_out) && scale_out > 0.0f;
}

cv::Mat depthTensorToCvMatFloat(const torch::Tensor& depth_tensor)
{
    torch::Tensor d = depth_tensor.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    CV_Assert(d.dim() == 2);
    cv::Mat depth_view(
        static_cast<int>(d.size(0)),
        static_cast<int>(d.size(1)),
        CV_32FC1,
        d.data_ptr<float>());
    return depth_view.clone();
}

bool saveMetricDepthPngMillimeters(
    const torch::Tensor& depth_meters,
    const std::filesystem::path& output_path,
    float valid_min_depth,
    float valid_max_depth)
{
    const cv::Mat depth = depthTensorToCvMatFloat(depth_meters);
    if (depth.empty()) {
        return false;
    }

    cv::Mat depth_mm(depth.rows, depth.cols, CV_16UC1, cv::Scalar(0));
    for (int y = 0; y < depth.rows; ++y) {
        const float* src = depth.ptr<float>(y);
        uint16_t* dst = depth_mm.ptr<uint16_t>(y);
        for (int x = 0; x < depth.cols; ++x) {
            const float z = src[x];
            if (!std::isfinite(z) || z <= valid_min_depth || z >= valid_max_depth) {
                continue;
            }
            dst[x] = static_cast<uint16_t>(std::clamp(
                std::lround(z * 1000.0f),
                1L,
                static_cast<long>(std::numeric_limits<uint16_t>::max())));
        }
    }

    std::filesystem::create_directories(output_path.parent_path());
    return cv::imwrite(output_path.string(), depth_mm);
}

cv::Mat colorizeDepthMatJet(
    const cv::Mat& depth_meters,
    float valid_min_depth,
    float valid_max_depth,
    float viz_min,
    float viz_max)
{
    CV_Assert(depth_meters.type() == CV_32FC1);
    const int H = depth_meters.rows;
    const int W = depth_meters.cols;

    cv::Mat gray(H, W, CV_8UC1, cv::Scalar(0));
    cv::Mat valid_mask(H, W, CV_8UC1, cv::Scalar(0));

    const float denom = std::max(1e-6f, viz_max - viz_min);
    for (int y = 0; y < H; ++y) {
        const float* src = depth_meters.ptr<float>(y);
        uint8_t* gray_ptr = gray.ptr<uint8_t>(y);
        uint8_t* mask_ptr = valid_mask.ptr<uint8_t>(y);
        for (int x = 0; x < W; ++x) {
            const float d = src[x];
            if (!std::isfinite(d) || d <= valid_min_depth || d >= valid_max_depth) {
                continue;
            }
            const float norm = std::clamp((d - viz_min) / denom, 0.0f, 1.0f);
            gray_ptr[x] = static_cast<uint8_t>(std::round(norm * 255.0f));
            mask_ptr[x] = 255;
        }
    }

    cv::Mat color_bgr;
    cv::applyColorMap(gray, color_bgr, cv::COLORMAP_JET);
    color_bgr.setTo(cv::Scalar(0, 0, 0), valid_mask == 0);
    return color_bgr;
}

cv::Mat colorizeFiniteScalarMatJet(
    const cv::Mat& values,
    float viz_min,
    float viz_max)
{
    return colorizeFiniteScalarMat(values, viz_min, viz_max, cv::COLORMAP_JET);
}

cv::Mat colorizeFiniteScalarMat(
    const cv::Mat& values,
    float viz_min,
    float viz_max,
    int colormap)
{
    CV_Assert(values.type() == CV_32FC1);
    const int H = values.rows;
    const int W = values.cols;

    cv::Mat gray(H, W, CV_8UC1, cv::Scalar(0));
    cv::Mat valid_mask(H, W, CV_8UC1, cv::Scalar(0));

    const float denom = std::max(1e-6f, viz_max - viz_min);
    for (int y = 0; y < H; ++y) {
        const float* src = values.ptr<float>(y);
        uint8_t* gray_ptr = gray.ptr<uint8_t>(y);
        uint8_t* mask_ptr = valid_mask.ptr<uint8_t>(y);
        for (int x = 0; x < W; ++x) {
            const float v = src[x];
            if (!std::isfinite(v)) {
                continue;
            }
            const float norm = std::clamp((v - viz_min) / denom, 0.0f, 1.0f);
            gray_ptr[x] = static_cast<uint8_t>(std::round(norm * 255.0f));
            mask_ptr[x] = 255;
        }
    }

    cv::Mat color_bgr;
    cv::applyColorMap(gray, color_bgr, colormap);
    color_bgr.setTo(cv::Scalar(0, 0, 0), valid_mask == 0);
    return color_bgr;
}

cv::Mat appendColormapLegendBar(
    const cv::Mat& image_bgr,
    float viz_min,
    float viz_max,
    const std::string& unit_suffix,
    int colormap,
    const std::string& high_label,
    const std::string& low_label)
{
    const int H = image_bgr.rows;
    const int bar_w = 24;
    const int pad = 8;
    const int legend_w = 180;

    cv::Mat gray(H, bar_w, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < H; ++y) {
        const float t = (H > 1) ? (1.0f - static_cast<float>(y) / static_cast<float>(H - 1)) : 1.0f;
        gray.row(y).setTo(cv::Scalar(static_cast<uint8_t>(std::round(std::clamp(t, 0.0f, 1.0f) * 255.0f))));
    }

    cv::Mat bar_bgr;
    cv::applyColorMap(gray, bar_bgr, colormap);

    cv::Mat legend(H, legend_w, CV_8UC3, cv::Scalar(0, 0, 0));
    bar_bgr.copyTo(legend(cv::Rect(pad, 0, bar_w, H)));

    const int text_x = pad + bar_w + 10;
    const double font_scale = 0.5;
    const int thickness = 1;
    const cv::Scalar white(255, 255, 255);
    const std::string max_text = "max " + std::string(cv::format("%.3f", viz_max)) + unit_suffix;
    const std::string min_text = "min " + std::string(cv::format("%.3f", viz_min)) + unit_suffix;

    cv::putText(legend, max_text, {text_x, 20}, cv::FONT_HERSHEY_SIMPLEX, font_scale, white, thickness, cv::LINE_AA);
    if (!high_label.empty()) {
        cv::putText(legend, high_label, {text_x, 40}, cv::FONT_HERSHEY_SIMPLEX, font_scale, white, thickness, cv::LINE_AA);
    }
    if (!low_label.empty()) {
        cv::putText(legend, low_label, {text_x, std::max(20, H - 24)}, cv::FONT_HERSHEY_SIMPLEX, font_scale, white, thickness, cv::LINE_AA);
    }
    cv::putText(legend, min_text, {text_x, std::max(16, H - 6)}, cv::FONT_HERSHEY_SIMPLEX, font_scale, white, thickness, cv::LINE_AA);

    cv::Mat out;
    cv::hconcat(std::vector<cv::Mat>{image_bgr, legend}, out);
    return out;
}

cv::Mat appendJetLegendBar(
    const cv::Mat& image_bgr,
    float viz_min,
    float viz_max,
    const std::string& unit_suffix)
{
    return appendColormapLegendBar(
        image_bgr,
        viz_min,
        viz_max,
        unit_suffix,
        cv::COLORMAP_JET,
        "red high",
        "blue low");
}

bool renderPkgToNormalForEval(
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    torch::Tensor& render_normal)
{
    render_normal = torch::Tensor();

    auto it_normal = render_pkg.find("raw_normal");
    if (it_normal == render_pkg.end() || !it_normal->second.defined()) {
        it_normal = render_pkg.find("normal");
        if (it_normal == render_pkg.end() || !it_normal->second.defined()) {
            return false;
        }
    }

    render_normal = it_normal->second.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (render_normal.dim() == 4 && render_normal.size(0) == 1) {
        render_normal = render_normal.squeeze(0);
    }
    if (render_normal.dim() != 3 || render_normal.size(0) < 3) {
        render_normal = torch::Tensor();
        return false;
    }
    if (render_normal.size(0) > 3) {
        render_normal = render_normal.index({torch::indexing::Slice(0, 3)}).contiguous();
    }
    return true;
}

torch::Tensor depthToNormal(
    const sv::MiniCam& cam,
    const torch::Tensor& depth,
    int ks,
    float tol_cos)
{
    using namespace torch::indexing;

    auto opts = depth.options();
    const int64_t H = depth.size(0);
    const int64_t W = depth.size(1);
    torch::Tensor normal = torch::zeros({3, H, W}, opts);
    if (H < 3 || W < 3) {
        return normal;
    }

    const float fx = (cam.fx > 1.0e-6f)
        ? cam.fx
        : (0.5f * static_cast<float>(W) / std::max(cam.tanfovx, 1.0e-6f));
    const float fy = (cam.fy > 1.0e-6f)
        ? cam.fy
        : (0.5f * static_cast<float>(H) / std::max(cam.tanfovy, 1.0e-6f));
    if (fx <= 1.0e-6f || fy <= 1.0e-6f) {
        return normal;
    }

    torch::Tensor uu = torch::arange(0, W, opts).view({1, W}).expand({H, W});
    torch::Tensor vv = torch::arange(0, H, opts).view({H, 1}).expand({H, W});
    torch::Tensor x = (uu - cam.cx) / fx;
    torch::Tensor y = (vv - cam.cy) / fy;
    torch::Tensor z = torch::ones_like(x);

    // SVRecon's compute_rd() keeps camera-ray z equal to one. Its rendered
    // RGB-D and TANDEM depth are camera-Z depth.
    // values rather than Euclidean ray ranges.
    torch::Tensor rd_cam = torch::stack({x, y, z}, 0);

    torch::Tensor c2w = cam.c2w.to(depth.device(), depth.scalar_type());
    torch::Tensor R = c2w.index({Slice(0, 3), Slice(0, 3)});
    torch::Tensor cam_pos = c2w.index({Slice(0, 3), 3}).view({3, 1, 1});
    torch::Tensor rd_world = torch::matmul(R, rd_cam.view({3, H * W})).view({3, H, W});

    torch::Tensor pts = cam_pos + rd_world * depth.unsqueeze(0);
    ks = std::max(3, ks);
    if ((ks % 2) == 0) {
        ks += 1;
    }
    const int64_t pad = ks / 2;
    const int64_t ks_1 = ks - 1;
    if (H <= ks_1 || W <= ks_1 || (H - 2 * pad) <= 0 || (W - 2 * pad) <= 0) {
        return normal;
    }

    torch::Tensor dx = pts.index({Slice(), Slice(pad, H - pad), Slice(ks_1, W)}) -
                       pts.index({Slice(), Slice(pad, H - pad), Slice(0, W - ks_1)});
    torch::Tensor dy = pts.index({Slice(), Slice(ks_1, H), Slice(pad, W - pad)}) -
                       pts.index({Slice(), Slice(0, H - ks_1), Slice(pad, W - pad)});
    torch::Tensor n_patch = torch::cross(dx, dy, 0);
    n_patch = torch::nn::functional::normalize(
        n_patch,
        torch::nn::functional::NormalizeFuncOptions().dim(0).eps(1.0e-12));
    normal.index_put_({Slice(), Slice(pad, H - pad), Slice(pad, W - pad)}, n_patch);

    if (tol_cos > 0.0f) {
        torch::Tensor pts_dir = torch::nn::functional::normalize(
            pts - cam_pos,
            torch::nn::functional::NormalizeFuncOptions().dim(0).eps(1.0e-12));
        torch::Tensor dot = (normal * pts_dir).sum(0);
        torch::Tensor mask = (dot > tol_cos).to(normal.scalar_type());
        normal = normal * mask.unsqueeze(0);
    }
    return normal;
}

torch::Tensor validDepthSupportMask(const torch::Tensor& valid, int ks)
{
    using namespace torch::indexing;

    if (valid.dim() != 2) {
        return torch::zeros_like(valid, torch::TensorOptions().dtype(torch::kBool));
    }

    ks = std::max(3, ks);
    if ((ks % 2) == 0) {
        ks += 1;
    }

    const int64_t H = valid.size(0);
    const int64_t W = valid.size(1);
    const int64_t pad = ks / 2;
    if (H <= 2 * pad || W <= 2 * pad) {
        return torch::zeros_like(valid, torch::TensorOptions().dtype(torch::kBool));
    }

    torch::Tensor valid_bool = valid.to(torch::kBool);
    torch::Tensor support = torch::zeros_like(valid_bool);
    torch::Tensor inner = torch::ones(
        {H - 2 * pad, W - 2 * pad},
        torch::TensorOptions().dtype(torch::kBool).device(valid.device()));

    for (int64_t dy = -pad; dy <= pad; ++dy) {
        for (int64_t dx = -pad; dx <= pad; ++dx) {
            inner = inner & valid_bool.index({
                Slice(pad + dy, H - pad + dy),
                Slice(pad + dx, W - pad + dx)});
        }
    }

    support.index_put_({Slice(pad, H - pad), Slice(pad, W - pad)}, inner);
    return support;
}

torch::Tensor normalDepthConsistencyLossSvrecon(
    const sv::MiniCam& cam,
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    int ks,
    float tol_deg)
{
    using namespace torch::indexing;

    auto it_T = render_pkg.find("raw_T");
    auto it_depth = render_pkg.find("raw_depth");
    auto it_normal = render_pkg.find("raw_normal");
    torch::Device zero_device = cam.c2w.device();
    if (it_T != render_pkg.end() && it_T->second.defined()) {
        zero_device = it_T->second.device();
    } else if (it_depth != render_pkg.end() && it_depth->second.defined()) {
        zero_device = it_depth->second.device();
    } else if (it_normal != render_pkg.end() && it_normal->second.defined()) {
        zero_device = it_normal->second.device();
    }

    if (it_T == render_pkg.end() ||
        it_depth == render_pkg.end() ||
        it_normal == render_pkg.end() ||
        !it_T->second.defined() ||
        !it_depth->second.defined() ||
        !it_normal->second.defined()) {
        return torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(zero_device));
    }

    torch::Tensor raw_T = it_T->second;
    if (raw_T.dim() == 4 && raw_T.size(0) == 1) {
        raw_T = raw_T.squeeze(0);
    }
    if (raw_T.dim() == 3 && raw_T.size(0) == 1) {
        raw_T = raw_T.squeeze(0);
    }
    if (raw_T.dim() != 2) {
        return torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(raw_T.device()));
    }
    torch::Tensor render_alpha = 1.0f - raw_T.detach();

    torch::Tensor raw_depth = it_depth->second;
    if (raw_depth.dim() == 4 && raw_depth.size(0) == 1) {
        raw_depth = raw_depth.squeeze(0);
    }
    if (raw_depth.dim() != 3 || raw_depth.size(0) < 1) {
        return torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(raw_depth.device()));
    }
    torch::Tensor render_depth = raw_depth.index({0});

    torch::Tensor render_normal = it_normal->second;
    if (render_normal.dim() == 4 && render_normal.size(0) == 1) {
        render_normal = render_normal.squeeze(0);
    }
    if (render_normal.dim() != 3 || render_normal.size(0) < 3) {
        return torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(render_normal.device()));
    }
    if (render_normal.size(0) > 3) {
        render_normal = render_normal.index({Slice(0, 3)});
    }

    constexpr float kPi = 3.14159265358979323846f;
    const float tol_cos = std::cos(tol_deg * kPi / 180.0f);
    torch::Tensor n_mean = depthToNormal(cam, render_depth, ks, tol_cos);

    torch::Tensor target = render_alpha.square();
    n_mean = n_mean * render_alpha.unsqueeze(0);
    torch::Tensor mask = (n_mean != 0).any(0);
    torch::Tensor loss_map =
        (target - (render_normal * n_mean).sum(0)) * mask.to(target.scalar_type());
    return loss_map.mean();
}

} // namespace voxel_eval
