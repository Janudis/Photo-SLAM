#include "include_voxel/voxel_mapper_evaluation.h"
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
    // For GT-vs-render depth visualization/evaluation, prefer the surface-like
    // median depth channel when available. This matches SVRaster mesh/fusion
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

torch::Tensor depth2normalSVRaster(
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

    torch::Tensor rd_cam = torch::stack({x, y, z}, 0);
    rd_cam = torch::nn::functional::normalize(
        rd_cam,
        torch::nn::functional::NormalizeFuncOptions().dim(0).eps(1.0e-12));

    torch::Tensor c2w = cam.c2w.to(depth.device(), depth.scalar_type());
    torch::Tensor R = c2w.index({Slice(0, 3), Slice(0, 3)});
    torch::Tensor cam_pos = c2w.index({Slice(0, 3), 3}).view({3, 1, 1});
    torch::Tensor rd_world = torch::matmul(R, rd_cam.view({3, H * W})).view({3, H, W});
    rd_world = torch::nn::functional::normalize(
        rd_world,
        torch::nn::functional::NormalizeFuncOptions().dim(0).eps(1.0e-12));

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
    torch::Tensor n_mean = depth2normalSVRaster(cam, render_depth, ks, tol_cos);

    torch::Tensor target = render_alpha.square();
    n_mean = n_mean * render_alpha.unsqueeze(0);
    torch::Tensor mask = (n_mean != 0).any(0);
    torch::Tensor loss_map =
        (target - (render_normal * n_mean).sum(0)) * mask.to(target.scalar_type());
    return loss_map.mean();
}

cv::Mat colorizeNormalMapBgr(const torch::Tensor& normal_tensor)
{
    torch::Tensor normal = normal_tensor.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (normal.dim() != 3 || normal.size(0) < 3) {
        return cv::Mat();
    }
    if (normal.size(0) > 3) {
        normal = normal.index({torch::indexing::Slice(0, 3)}).contiguous();
    }

    const int H = static_cast<int>(normal.size(1));
    const int W = static_cast<int>(normal.size(2));
    cv::Mat normal_rgb(H, W, CV_8UC3, cv::Scalar(0, 0, 0));
    auto acc = normal.accessor<float, 3>();
    for (int y = 0; y < H; ++y) {
        auto* row = normal_rgb.ptr<cv::Vec3b>(y);
        for (int x = 0; x < W; ++x) {
            float nx = acc[0][y][x];
            float ny = acc[1][y][x];
            float nz = acc[2][y][x];
            if (!std::isfinite(nx) || !std::isfinite(ny) || !std::isfinite(nz)) {
                continue;
            }
            const float norm = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (!(norm > 1e-6f)) {
                continue;
            }
            nx /= norm;
            ny /= norm;
            nz /= norm;
            row[x] = cv::Vec3b(
                static_cast<uint8_t>(std::round(std::clamp(0.5f * (nx + 1.0f), 0.0f, 1.0f) * 255.0f)),
                static_cast<uint8_t>(std::round(std::clamp(0.5f * (ny + 1.0f), 0.0f, 1.0f) * 255.0f)),
                static_cast<uint8_t>(std::round(std::clamp(0.5f * (nz + 1.0f), 0.0f, 1.0f) * 255.0f)));
        }
    }
    cv::Mat normal_bgr;
    cv::cvtColor(normal_rgb, normal_bgr, cv::COLOR_RGB2BGR);
    return normal_bgr;
}

torch::Tensor normalWorldToCameraForViz(
    const sv::MiniCam& cam,
    const torch::Tensor& normal_world)
{
    if (!normal_world.defined() || normal_world.dim() != 3 || normal_world.size(0) < 3) {
        return normal_world;
    }
    torch::Tensor normal = normal_world.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (normal.size(0) > 3) {
        normal = normal.index({torch::indexing::Slice(0, 3)}).contiguous();
    }
    torch::Tensor w2c = cam.w2c.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (w2c.dim() != 2 || w2c.size(0) < 3 || w2c.size(1) < 3) {
        return normal;
    }
    const int64_t H = normal.size(1);
    const int64_t W = normal.size(2);
    torch::Tensor Rcw = w2c.index({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)});
    return torch::matmul(Rcw, normal.view({3, H * W})).view({3, H, W}).contiguous();
}

cv::Mat blackRgbImage(int height, int width)
{
    if (height <= 0 || width <= 0) {
        return cv::Mat();
    }
    return cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
}

cv::Mat bgrToRgbImage(const cv::Mat& image_bgr)
{
    if (image_bgr.empty()) {
        return cv::Mat();
    }
    cv::Mat image_rgb;
    cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);
    return image_rgb;
}

cv::Mat chwRgbFloatTensorToU8Rgb(torch::Tensor image)
{
    if (!image.defined() || image.numel() == 0) {
        return cv::Mat();
    }
    image = image.detach().to(torch::kCPU).to(torch::kFloat32);
    if (image.dim() == 4 && image.size(0) == 1) {
        image = image.squeeze(0);
    }
    if (image.dim() != 3) {
        return cv::Mat();
    }
    if (image.size(0) == 3 || image.size(0) == 4) {
        image = image.index({torch::indexing::Slice(0, 3)})
                    .clamp(0.0f, 1.0f)
                    .permute({1, 2, 0})
                    .mul(255.0f)
                    .to(torch::kUInt8)
                    .contiguous();
    } else if (image.size(2) == 3 || image.size(2) == 4) {
        image = image.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 3)})
                    .clamp(0.0f, 1.0f)
                    .mul(255.0f)
                    .to(torch::kUInt8)
                    .contiguous();
    } else {
        return cv::Mat();
    }

    cv::Mat view(
        static_cast<int>(image.size(0)),
        static_cast<int>(image.size(1)),
        CV_8UC3,
        image.data_ptr<uint8_t>());
    return view.clone();
}

cv::Mat makeDepthGapMaskRgb(
    const cv::Mat& pred_depth_meters,
    const cv::Mat& gt_depth_meters,
    const torch::Tensor& eval_mask,
    float valid_min_depth,
    float valid_max_depth,
    double& gap_percent_out)
{
    gap_percent_out = -1.0;
    if (pred_depth_meters.empty() || gt_depth_meters.empty() ||
        pred_depth_meters.type() != CV_32FC1 || gt_depth_meters.type() != CV_32FC1 ||
        pred_depth_meters.rows != gt_depth_meters.rows ||
        pred_depth_meters.cols != gt_depth_meters.cols) {
        return cv::Mat();
    }

    torch::Tensor mask_cpu = eval_mask.detach().to(torch::kCPU).to(torch::kBool).contiguous();
    const bool have_mask =
        mask_cpu.dim() == 2 &&
        mask_cpu.size(0) == pred_depth_meters.rows &&
        mask_cpu.size(1) == pred_depth_meters.cols;

    cv::Mat out(pred_depth_meters.rows, pred_depth_meters.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    int64_t gt_valid_count = 0;
    int64_t gap_count = 0;
    for (int y = 0; y < pred_depth_meters.rows; ++y) {
        const float* pred_row = pred_depth_meters.ptr<float>(y);
        const float* gt_row = gt_depth_meters.ptr<float>(y);
        cv::Vec3b* out_row = out.ptr<cv::Vec3b>(y);
        const bool* mask_row = have_mask ? mask_cpu[y].data_ptr<bool>() : nullptr;
        for (int x = 0; x < pred_depth_meters.cols; ++x) {
            if (mask_row && !mask_row[x]) {
                continue;
            }
            const float pred = pred_row[x];
            const float gt = gt_row[x];
            const bool pred_valid =
                std::isfinite(pred) && pred > valid_min_depth && pred < valid_max_depth;
            const bool gt_valid =
                std::isfinite(gt) && gt > valid_min_depth && gt < valid_max_depth;
            if (!gt_valid) {
                continue;
            }
            ++gt_valid_count;
            if (!pred_valid) {
                ++gap_count;
                out_row[x] = cv::Vec3b(255, 0, 0); // RGB red: GT exists, render is missing.
            }
        }
    }
    if (gt_valid_count > 0) {
        gap_percent_out = 100.0 * static_cast<double>(gap_count) /
                          static_cast<double>(gt_valid_count);
    }
    return out;
}

bool sparseSamplesToDepthMat(
    const torch::Tensor& sparse_uv,
    const torch::Tensor& sparse_depth,
    int image_width,
    int image_height,
    cv::Mat& depth_meters)
{
    depth_meters = cv::Mat(
        image_height,
        image_width,
        CV_32FC1,
        cv::Scalar(std::numeric_limits<float>::quiet_NaN()));

    if (!sparse_uv.defined() || !sparse_depth.defined()) {
        return false;
    }
    if (sparse_uv.dim() != 2 || sparse_uv.size(1) != 2) {
        return false;
    }

    torch::Tensor uv = sparse_uv.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor depth = sparse_depth.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (depth.dim() == 2 && depth.size(1) == 1) {
        depth = depth.squeeze(1);
    }
    if (depth.dim() != 1 || depth.size(0) != uv.size(0)) {
        return false;
    }

    const auto uv_acc = uv.accessor<float, 2>();
    const auto depth_acc = depth.accessor<float, 1>();
    int64_t n_written = 0;
    for (int64_t i = 0; i < uv.size(0); ++i) {
        const float z = depth_acc[i];
        if (!std::isfinite(z) || z <= 0.0f) {
            continue;
        }

        const float px_f = 0.5f * (uv_acc[i][0] + 1.0f) * static_cast<float>(image_width);
        const float py_f = 0.5f * (uv_acc[i][1] + 1.0f) * static_cast<float>(image_height);
        const int px = static_cast<int>(std::lround(px_f));
        const int py = static_cast<int>(std::lround(py_f));
        if (px < 0 || px >= image_width || py < 0 || py >= image_height) {
            continue;
        }

        float& cell = depth_meters.at<float>(py, px);
        if (!std::isfinite(cell) || z < cell) {
            cell = z;
        }
        ++n_written;
    }

    return n_written > 0;
}

const std::string& runtimeOrbDepthDebugRunTag()
{
    static const std::string tag = []() {
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        return std::to_string(now);
    }();
    return tag;
}

std::filesystem::path runtimeOrbDepthDebugDir(
    const std::filesystem::path& result_root)
{
    return result_root / (".orb_depth_debug_" + runtimeOrbDepthDebugRunTag());
}

void copyPngFilesToDirectory(
    const std::filesystem::path& source_dir,
    const std::filesystem::path& target_dir)
{
    if (source_dir.empty() || target_dir.empty() ||
        !std::filesystem::exists(source_dir) ||
        !std::filesystem::is_directory(source_dir)) {
        return;
    }

    std::filesystem::create_directories(target_dir);
    for (const auto& entry : std::filesystem::directory_iterator(source_dir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".png") {
            continue;
        }
        std::error_code ec;
        std::filesystem::copy_file(
            entry.path(),
            target_dir / entry.path().filename(),
            std::filesystem::copy_options::overwrite_existing,
            ec);
    }
}

cv::Scalar relDepthErrorColorBgr(float rel_error)
{
    if (!std::isfinite(rel_error)) {
        return cv::Scalar(255, 0, 255);
    }
    if (rel_error <= 0.05f) {
        return cv::Scalar(0, 220, 0);
    }
    if (rel_error <= 0.15f) {
        return cv::Scalar(0, 220, 220);
    }
    if (rel_error <= 0.30f) {
        return cv::Scalar(0, 140, 255);
    }
    return cv::Scalar(0, 0, 255);
}

bool saveAccumulatedOrbDepthProjectionPng(
    const std::shared_ptr<ORB_SLAM3::System>& slam,
    const std::shared_ptr<VoxelKeyframe>& kf,
    const std::filesystem::path& output_dir,
    int iteration,
    float valid_min_depth,
    float valid_max_depth,
    const torch::Tensor& aligned_da_depth)
{
    if (!slam || !kf || output_dir.empty() ||
        kf->image_width_ <= 0 || kf->image_height_ <= 0 || kf->intr_.size() < 4) {
        return false;
    }

    ORB_SLAM3::Atlas* atlas = slam->getAtlas();
    if (!atlas) {
        return false;
    }

    const std::vector<ORB_SLAM3::MapPoint*> map_points = atlas->GetAllMapPoints();
    if (map_points.empty()) {
        return false;
    }

    const int W = kf->image_width_;
    const int H = kf->image_height_;
    const float fx = kf->intr_[0];
    const float fy = kf->intr_[1];
    const float cx = kf->intr_[2];
    const float cy = kf->intr_[3];
    if (!(fx > 1e-6f) || !(fy > 1e-6f)) {
        return false;
    }

    const Sophus::SE3f Tcw = kf->getPosef();
    const float z_min = std::max(valid_min_depth, std::max(kf->znear_, 1e-6f));
    const bool use_z_max = std::isfinite(valid_max_depth) && valid_max_depth > z_min;
    const float viz_valid_max_depth = use_z_max ? valid_max_depth : 1e6f;

    cv::Mat depth_meters(H, W, CV_32FC1, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    int64_t projected_points = 0;
    for (ORB_SLAM3::MapPoint* mp : map_points) {
        if (!mp || mp->isBad()) {
            continue;
        }

        const Eigen::Vector3f p_world = mp->GetWorldPos();
        if (!std::isfinite(p_world.x()) ||
            !std::isfinite(p_world.y()) ||
            !std::isfinite(p_world.z())) {
            continue;
        }

        const Eigen::Vector3f p_cam = Tcw * p_world;
        const float z = p_cam.z();
        if (!std::isfinite(z) || z <= z_min || (use_z_max && z >= valid_max_depth)) {
            continue;
        }

        const float u = fx * p_cam.x() / z + cx;
        const float v = fy * p_cam.y() / z + cy;
        if (!std::isfinite(u) || !std::isfinite(v)) {
            continue;
        }

        const int px = static_cast<int>(std::lround(u));
        const int py = static_cast<int>(std::lround(v));
        if (px < 0 || px >= W || py < 0 || py >= H) {
            continue;
        }

        float& dst = depth_meters.at<float>(py, px);
        if (!std::isfinite(dst) || z < dst) {
            dst = z;
        }
        ++projected_points;
    }

    if (projected_points <= 0) {
        return false;
    }

    const torch::Tensor depth_tensor = torch::from_blob(
        depth_meters.data,
        {H, W},
        torch::TensorOptions().dtype(torch::kFloat32)).clone();
    float viz_min = 0.0f;
    float viz_max = 1.0f;
    if (!computeSharedDepthVizRange(
            depth_tensor,
            cv::Mat(),
            valid_min_depth,
            viz_valid_max_depth,
            viz_min,
            viz_max)) {
        return false;
    }

    const cv::Mat depth_bgr = colorizeDepthMatJet(
        depth_meters,
        valid_min_depth,
        viz_valid_max_depth,
        viz_min,
        viz_max);

    std::ostringstream stem;
    stem << "kf_" << std::setw(5) << std::setfill('0') << kf->fid_;
    std::ostringstream iter_tag;
    iter_tag << "_densification_iter_"
             << std::setw(5) << std::setfill('0') << iteration;

    std::filesystem::create_directories(output_dir);
    bool wrote_any = cv::imwrite(
        (output_dir / (stem.str() + "_orb_depth" + iter_tag.str() + ".png")).string(),
        depth_bgr);

    if (!aligned_da_depth.defined()) {
        return wrote_any;
    }

    cv::Mat aligned_da_meters = depthTensorToCvMatFloat(aligned_da_depth);
    if (aligned_da_meters.empty()) {
        return wrote_any;
    }
    if (aligned_da_meters.rows != H || aligned_da_meters.cols != W) {
        cv::resize(
            aligned_da_meters,
            aligned_da_meters,
            cv::Size(W, H),
            0.0,
            0.0,
            cv::INTER_LINEAR);
    }

    float da_viz_min = 0.0f;
    float da_viz_max = 1.0f;
    const torch::Tensor aligned_da_tensor = torch::from_blob(
        aligned_da_meters.data,
        {H, W},
        torch::TensorOptions().dtype(torch::kFloat32)).clone();
    cv::Mat aligned_da_bgr;
    if (computeSharedDepthVizRange(
            aligned_da_tensor,
            cv::Mat(),
            valid_min_depth,
            viz_valid_max_depth,
            da_viz_min,
            da_viz_max)) {
        aligned_da_bgr = colorizeDepthMatJet(
            aligned_da_meters,
            valid_min_depth,
            viz_valid_max_depth,
            da_viz_min,
            da_viz_max);
    } else {
        aligned_da_bgr = cv::Mat(H, W, CV_8UC3, cv::Scalar(0, 0, 0));
    }

    cv::Mat rel_error(H, W, CV_32FC1, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    cv::Mat orb_on_da = aligned_da_bgr.clone();

    const int radius = std::clamp(std::min(W, H) / 260, 2, 4);
    int64_t compared_points = 0;
    for (int y = 0; y < H; ++y) {
        const float* orb_row = depth_meters.ptr<float>(y);
        const float* da_row = aligned_da_meters.ptr<float>(y);
        float* err_row = rel_error.ptr<float>(y);
        for (int x = 0; x < W; ++x) {
            const float z_orb = orb_row[x];
            if (!std::isfinite(z_orb) || z_orb <= valid_min_depth ||
                z_orb >= viz_valid_max_depth) {
                continue;
            }

            const float z_da = da_row[x];
            float rel = std::numeric_limits<float>::quiet_NaN();
            if (std::isfinite(z_da) && z_da > valid_min_depth &&
                z_da < viz_valid_max_depth) {
                rel = std::abs(z_orb - z_da) / std::max(z_orb, 1e-6f);
                err_row[x] = rel;
                ++compared_points;
            }

            cv::circle(
                orb_on_da,
                cv::Point(x, y),
                radius,
                relDepthErrorColorBgr(rel),
                cv::FILLED,
                cv::LINE_AA);
        }
    }

    wrote_any |= cv::imwrite(
        (output_dir / (stem.str() + "_orb_on_da_error" + iter_tag.str() + ".png")).string(),
        orb_on_da);

    if (compared_points > 0) {
        const cv::Mat rel_error_bgr = appendJetLegendBar(
            colorizeFiniteScalarMatJet(rel_error, 0.0f, 0.5f),
            0.0f,
            0.5f,
            " rel");
        wrote_any |= cv::imwrite(
            (output_dir / (stem.str() + "_orb_vs_da_rel_error" + iter_tag.str() + ".png")).string(),
            rel_error_bgr);
    }

    return wrote_any;
}

void saveDepthComparisonDebugPngs(
    const torch::Tensor& pred_depth,
    const cv::Mat& gt_depth_meters,
    float valid_min_depth,
    float valid_max_depth,
    const std::filesystem::path& rendered_path,
    const std::filesystem::path& gt_path,
    const std::filesystem::path& pair_path,
    std::optional<float> pred_to_gt_scale)
{
    torch::Tensor pred_depth_for_pair = pred_depth;
    bool used_alignment = false;
    if (pred_to_gt_scale.has_value() &&
        std::isfinite(*pred_to_gt_scale) &&
        *pred_to_gt_scale > 0.0f) {
        pred_depth_for_pair = pred_depth * (*pred_to_gt_scale);
        used_alignment = true;
    }

    float viz_min = 0.0f;
    float viz_max = 6.0f;
    const float viz_valid_max = std::min(valid_max_depth, viz_max);

    const cv::Mat pred_depth_meters = depthTensorToCvMatFloat(pred_depth_for_pair);
    const cv::Mat pred_bgr = colorizeDepthMatJet(
        pred_depth_meters,
        valid_min_depth,
        viz_valid_max,
        viz_min,
        viz_max);

    std::filesystem::create_directories(rendered_path.parent_path());
    cv::imwrite(rendered_path.string(), pred_bgr);

    if (gt_depth_meters.empty()) {
        return;
    }

    cv::Mat gt_bgr = colorizeDepthMatJet(
        gt_depth_meters,
        valid_min_depth,
        viz_valid_max,
        viz_min,
        viz_max);

    cv::Mat pair_bgr;
    cv::hconcat(std::vector<cv::Mat>{gt_bgr, pred_bgr}, pair_bgr);
    pair_bgr = appendJetLegendBar(pair_bgr, viz_min, viz_max, " m");

    std::filesystem::create_directories(gt_path.parent_path());
    cv::imwrite(gt_path.string(), gt_bgr);
    cv::imwrite(pair_path.string(), pair_bgr);
}

bool getKeyframeDepthMetersForEval(
    const std::shared_ptr<VoxelKeyframe>& pkf,
    int expected_h,
    int expected_w,
    cv::Mat& depth_meters)
{
    if (!pkf) {
        return false;
    }

    if (!pkf->img_auxiliary_undist_.empty()) {
        if (!voxel_utils::depthMatToMeters(pkf->img_auxiliary_undist_, depth_meters)) {
            return false;
        }
    } else {
        if (!voxel_utils::loadReplicaDepthFromRgbPath(pkf->img_filename_, depth_meters) &&
            !voxel_utils::loadTumDepthFromRgbPath(pkf->img_filename_, depth_meters)) {
            return false;
        }
    }

    if (depth_meters.empty()) {
        return false;
    }

    if (depth_meters.rows != expected_h || depth_meters.cols != expected_w) {
        cv::resize(
            depth_meters,
            depth_meters,
            cv::Size(expected_w, expected_h),
            0.0,
            0.0,
            cv::INTER_NEAREST);
    }

    return true;
}


} // namespace voxel_eval

bool VoxelMapper::buildSparseDepthFromMapPoints(
    const sv::MiniCam& cam,
    int image_width,
    int image_height,
    torch::Tensor& sparse_uv,
    torch::Tensor& sparse_depth)
{
    // 0) Pull global SLAM point cloud (world coords) from the scene
    const auto& pcd = scene_->cached_point_cloud_;
    const int64_t M_total = static_cast<int64_t>(pcd.size());
    if (M_total == 0) {
        return false;
    }

    // 1) Pack world points into a host vector [M_total, 3]
    std::vector<float> host_pts;
    host_pts.reserve(3 * M_total);
    for (const auto& kv : pcd) {
        const Point3D& P = kv.second;          // you already fill xyz_ in run()
        host_pts.push_back(static_cast<float>(P.xyz_(0)));
        host_pts.push_back(static_cast<float>(P.xyz_(1)));
        host_pts.push_back(static_cast<float>(P.xyz_(2)));
    }

    auto opts_host = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor pts_world_cpu = torch::from_blob(
        host_pts.data(),
        {M_total, 3},
        opts_host);

    // Move to device and own the memory (clone())
    auto opts_dev = torch::TensorOptions().dtype(torch::kFloat32).device(mDevice);
    torch::Tensor pts_world = pts_world_cpu.clone().to(mDevice);   // [M,3]

    // 2) Transform world → camera using cam.w2c  (SVRaster-style)
    //
    // We build homogeneous coordinates [M,4] and multiply by w2c^T:
    //   X_cam = X_world_h @ w2c^T
    //
    torch::Tensor ones = torch::ones({M_total, 1}, opts_dev);
    torch::Tensor pts_world_h = torch::cat({pts_world, ones}, /*dim=*/1); // [M,4]

    torch::Tensor w2c = cam.w2c.to(mDevice);                                // [4,4]
    torch::Tensor pts_cam_h =
        torch::matmul(pts_world_h, w2c.transpose(0, 1));                    // [M,4]
    torch::Tensor pts_cam = pts_cam_h.index(
        {torch::indexing::Slice(), torch::indexing::Slice(0, 3)});          // [M,3]

    torch::Tensor X = pts_cam.index({torch::indexing::Slice(), 0}); // [M]
    torch::Tensor Y = pts_cam.index({torch::indexing::Slice(), 1}); // [M]
    torch::Tensor Z = pts_cam.index({torch::indexing::Slice(), 2}); // [M]

    // 3) Compute intrinsics from tanFOV + cx,cy (exactly what rasterizer uses)
    const float W = static_cast<float>(image_width);
    const float H = static_cast<float>(image_height);

    const float fx = 0.5f * W / cam.tanfovx;
    const float fy = 0.5f * H / cam.tanfovy;

    // u,v in pixel coords
    torch::Tensor u = fx * X / Z + cam.cx;   // [M]
    torch::Tensor v = fy * Y / Z + cam.cy;   // [M]

    // 4) Visibility & image bounds
    torch::Tensor valid =
        (Z > 0.0f) &
        (u >= 0.0f) & (u <= (W - 1.0f)) &
        (v >= 0.0f) & (v <= (H - 1.0f));     // [M]

    torch::Tensor valid_idx = torch::nonzero(valid).squeeze(1); // [M_vis]
    const int64_t M_vis = valid_idx.size(0);
    if (M_vis == 0) {
        return false;
    }

    // 5) Subsample to at most N_max points (same spirit as RGB-D version)
    const int64_t N_max = 3000;
    torch::Tensor chosen_idx;
    if (M_vis <= N_max) {
        chosen_idx = valid_idx;
    } else {
        const int64_t stride = std::max<int64_t>(int64_t(1), M_vis / N_max);
        torch::Tensor arange_idx = torch::arange(
            0, M_vis, stride,
            torch::TensorOptions().dtype(torch::kLong).device(valid_idx.device()));
        if (arange_idx.size(0) > N_max) {
            arange_idx = arange_idx.slice(0, 0, N_max);
        }
        chosen_idx = valid_idx.index_select(0, arange_idx); // [N]
    }

    const int64_t N = chosen_idx.size(0);
    if (N == 0) {
        return false;
    }

    // 6) Gather u, v, Z for the chosen points
    torch::Tensor u_chosen = u.index_select(0, chosen_idx); // [N]
    torch::Tensor v_chosen = v.index_select(0, chosen_idx); // [N]
    torch::Tensor z_chosen = Z.index_select(0, chosen_idx); // [N]

    // 7) Match SVRaster Camera.project(): 2 * u / W - 1, 2 * v / H - 1.
    torch::Tensor u_ndc =
        2.0f * (u_chosen / W) - 1.0f;                      // [N]
    torch::Tensor v_ndc =
        2.0f * (v_chosen / H) - 1.0f;                      // [N]

    sparse_uv    = torch::stack({u_ndc, v_ndc}, /*dim=*/1); // [N,2]
    sparse_depth = z_chosen;                                // [N]

    return true;
}

torch::Tensor VoxelMapper::computeSparseDepthLoss_Points(
    const std::shared_ptr<VoxelKeyframe>& kf,
    const sv::MiniCam& cam,
    int image_width,
    int image_height,
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    int iteration)
{
    auto zero = torch::zeros(
        {1},
        torch::TensorOptions().dtype(torch::kFloat32).device(mDevice));

    // 0) Weight or schedule off -> no contribution
    if (opt_params_.lambda_sparse_depth_ <= 0.0f)
        return zero;

    loss_utils::SparseDepthLoss sparse_depth_loss(opt_params_.sparse_depth_until_);
    if (!sparse_depth_loss.isActive(iteration))
        return zero;

    // 1) Get raw_T / raw_depth (SVRaster-style)
    auto it_T = render_pkg.find("raw_T");
    if (it_T == render_pkg.end())
        it_T = render_pkg.find("T");          // fallback

    auto it_depth = render_pkg.find("raw_depth");
    if (it_depth == render_pkg.end())
        it_depth = render_pkg.find("depth");  // fallback

    if (it_T == render_pkg.end() || it_depth == render_pkg.end()) {
        return zero;
    }

    torch::Tensor raw_T     = it_T->second.to(mDevice);
    torch::Tensor raw_depth = it_depth->second.to(mDevice);

    // 2) Build (sparse_uv, sparse_depth) from SLAM 3D points (SVRaster-style)
    torch::Tensor sparse_uv;     // [N,2]
    torch::Tensor sparse_depth;  // [N]
    if (!buildSparseDepthFromMapPoints(cam, image_width, image_height,
                                       sparse_uv, sparse_depth)) {
        // No visible 3D points for this viewpoint
        return zero;
    }
    // Avoid N=1 shape corner case inside SparseDepthLoss (grid_sample(...).squeeze()).
    if (!sparse_uv.defined() || !sparse_depth.defined() ||
        sparse_uv.dim() != 2 || sparse_uv.size(0) < 2 ||
        sparse_depth.numel() < 2) {
        return zero;
    }

    // 3) Low-level SparseDepthLoss (exact math as SVRaster’s __call__)
    torch::Tensor depth_loss =
        sparse_depth_loss(raw_T, raw_depth, sparse_uv, sparse_depth);

    return depth_loss;
}

torch::Tensor VoxelMapper::computeRgbdDepthLoss(
    const std::shared_ptr<VoxelKeyframe>& kf,
    const sv::MiniCam& cam,
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    int iteration)
{
    auto zero = torch::zeros(
        {1},
        torch::TensorOptions().dtype(torch::kFloat32).device(mDevice));

    if (sensor_type_ != RGBD ||
        opt_params_.lambda_rgbd_depth_ <= 0.0f ||
        iteration < opt_params_.rgbd_depth_from_ ||
        iteration > opt_params_.rgbd_depth_end_ ||
        !kf ||
        kf->img_auxiliary_undist_.empty()) {
        return zero;
    }

    auto it_T = render_pkg.find("raw_T");
    if (it_T == render_pkg.end()) {
        it_T = render_pkg.find("T");
    }
    auto it_depth = render_pkg.find("raw_depth");
    if (it_depth == render_pkg.end()) {
        it_depth = render_pkg.find("depth");
    }
    if (it_T == render_pkg.end() || it_depth == render_pkg.end() ||
        !it_T->second.defined() || !it_depth->second.defined()) {
        return zero;
    }

    torch::Tensor raw_depth =
        it_depth->second.to(mDevice, torch::kFloat32).contiguous();
    if (raw_depth.dim() == 4 && raw_depth.size(0) == 1) {
        raw_depth = raw_depth.squeeze(0);
    }
    if (raw_depth.dim() == 2) {
        raw_depth = raw_depth.unsqueeze(0);
    }
    if (raw_depth.dim() != 3 || raw_depth.size(0) < 1) {
        return zero;
    }

    torch::Tensor raw_T =
        it_T->second.to(mDevice, torch::kFloat32).contiguous();
    if (raw_T.dim() == 4 && raw_T.size(0) == 1) {
        raw_T = raw_T.squeeze(0);
    }
    if (raw_T.dim() == 3 && raw_T.size(0) >= 1) {
        raw_T = raw_T.index({0});
    }
    if (raw_T.dim() != 2) {
        return zero;
    }

    const int H = static_cast<int>(raw_depth.size(1));
    const int W = static_cast<int>(raw_depth.size(2));
    if (raw_T.size(0) != H || raw_T.size(1) != W) {
        return zero;
    }

    cv::cuda::GpuMat depth_gpu;
    depth_gpu.upload(kf->img_auxiliary_undist_);
    torch::Tensor rgbd_depth =
        tensor_utils::cvGpuMat2TorchTensor_Float32(depth_gpu)
            .to(mDevice, torch::kFloat32)
            .contiguous();
    if (rgbd_depth.dim() == 3 && rgbd_depth.size(0) == 1) {
        rgbd_depth = rgbd_depth.squeeze(0);
    }
    if (rgbd_depth.dim() != 2) {
        return zero;
    }
    if (rgbd_depth.size(0) != H || rgbd_depth.size(1) != W) {
        rgbd_depth = torch::nn::functional::interpolate(
            rgbd_depth.unsqueeze(0).unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{H, W})
                .mode(torch::kNearest)).squeeze();
    }

    const float near_depth = std::max(1e-6f, cam.near);
    const float min_depth = std::max(RGBD_min_depth_, near_depth);
    torch::Tensor alpha = (1.0f - raw_T).clamp(0.0f, 1.0f);
    torch::Tensor valid =
        torch::isfinite(raw_depth.index({0})) &
        torch::isfinite(rgbd_depth) &
        (rgbd_depth > min_depth) &
        (rgbd_depth < RGBD_max_depth_) &
        (alpha > 0.8f);
    if (!valid.any().item<bool>()) {
        return zero;
    }

    torch::Tensor mask = valid.to(raw_depth.dtype());
    torch::Tensor render_inv =
        (1.0f / raw_depth.index({0}).clamp_min(near_depth)) * alpha;
    torch::Tensor target_inv =
        1.0f / rgbd_depth.clamp_min(near_depth);

    render_inv = render_inv * mask;
    target_inv = target_inv * mask;

    torch::Tensor loss = torch::nn::functional::mse_loss(
        render_inv,
        target_inv,
        torch::nn::functional::MSELossFuncOptions().reduction(torch::kMean));

    if (opt_params_.rgbd_depth_end_ <= opt_params_.rgbd_depth_from_ ||
        opt_params_.rgbd_depth_end_mult_ == 1.0f) {
        return loss;
    }

    const float ratio = std::clamp(
        static_cast<float>(iteration - opt_params_.rgbd_depth_from_) /
            static_cast<float>(opt_params_.rgbd_depth_end_ -
                               opt_params_.rgbd_depth_from_),
        0.0f,
        1.0f);
    const float mult = std::pow(opt_params_.rgbd_depth_end_mult_, ratio);
    return loss * mult;
}

torch::Tensor VoxelMapper::computeRgbdMaskLoss(
    const std::shared_ptr<VoxelKeyframe>& kf,
    const sv::MiniCam& cam,
    const std::unordered_map<std::string, torch::Tensor>& render_pkg)
{
    auto zero = torch::zeros(
        {1},
        torch::TensorOptions().dtype(torch::kFloat32).device(mDevice));

    if (sensor_type_ != RGBD || !kf || kf->img_auxiliary_undist_.empty()) {
        return zero;
    }

    auto it_T = render_pkg.find("raw_T");
    if (it_T == render_pkg.end()) {
        it_T = render_pkg.find("T");
    }
    if (it_T == render_pkg.end() || !it_T->second.defined()) {
        return zero;
    }

    torch::Tensor raw_T =
        it_T->second.to(mDevice, torch::kFloat32).contiguous();
    if (raw_T.dim() == 4 && raw_T.size(0) == 1) {
        raw_T = raw_T.squeeze(0);
    }
    if (raw_T.dim() == 3 && raw_T.size(0) >= 1) {
        raw_T = raw_T.index({0});
    }
    if (raw_T.dim() != 2) {
        return zero;
    }

    cv::cuda::GpuMat depth_gpu;
    depth_gpu.upload(kf->img_auxiliary_undist_);
    torch::Tensor rgbd_depth =
        tensor_utils::cvGpuMat2TorchTensor_Float32(depth_gpu)
            .to(mDevice, torch::kFloat32)
            .contiguous();
    if (rgbd_depth.dim() == 3 && rgbd_depth.size(0) == 1) {
        rgbd_depth = rgbd_depth.squeeze(0);
    }
    if (rgbd_depth.dim() != 2) {
        return zero;
    }
    if (rgbd_depth.sizes() != raw_T.sizes()) {
        rgbd_depth = torch::nn::functional::interpolate(
            rgbd_depth.unsqueeze(0).unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{raw_T.size(0), raw_T.size(1)})
                .mode(torch::kNearest)).squeeze();
    }

    const float min_depth = std::max(
        RGBD_min_depth_, std::max(1.0e-6f, cam.near));
    torch::Tensor valid =
        torch::isfinite(rgbd_depth) &
        (rgbd_depth > min_depth) &
        (rgbd_depth < RGBD_max_depth_);
    if (!valid.any().item<bool>()) {
        return zero;
    }

    // Equivalent to SVRecon's foreground-mask objective
    // ||T - (1-mask)||^2. Valid RGB-D samples have mask=1, hence target T=0.
    return raw_T.index({valid}).square().mean();
}

torch::Tensor VoxelMapper::computeRgbdSdfLoss(
    const std::shared_ptr<VoxelKeyframe>& kf,
    const sv::MiniCam& cam,
    int iteration)
{
    using torch::indexing::Slice;

    auto zero = torch::zeros(
        {1},
        torch::TensorOptions().dtype(torch::kFloat32).device(mDevice));

    if (sensor_type_ != RGBD ||
        opt_params_.lambda_rgbd_sdf_ <= 0.0f ||
        iteration < opt_params_.rgbd_sdf_from_ ||
        iteration > opt_params_.rgbd_sdf_end_ ||
        !kf ||
        kf->img_auxiliary_undist_.empty() ||
        !voxel_model_) {
        return zero;
    }

    if (!voxel_model_->geoGridPts().defined() ||
        voxel_model_->geoGridPts().numel() == 0) {
        return zero;
    }

    cv::Mat depth_meters;
    if (!voxel_utils::depthMatToMeters(kf->img_auxiliary_undist_, depth_meters) ||
        depth_meters.empty()) {
        return zero;
    }
    if (depth_meters.type() != CV_32FC1) {
        depth_meters.convertTo(depth_meters, CV_32FC1);
    }

    torch::Tensor rgbd_depth =
        torch::from_blob(
            depth_meters.data,
            {depth_meters.rows, depth_meters.cols},
            torch::TensorOptions().dtype(torch::kFloat32))
            .clone()
            .to(mDevice, torch::kFloat32)
            .contiguous();

    const int H = cam.height;
    const int W = cam.width;
    if (H <= 0 || W <= 0) {
        return zero;
    }
    if (rgbd_depth.size(0) != H || rgbd_depth.size(1) != W) {
        rgbd_depth = torch::nn::functional::interpolate(
            rgbd_depth.unsqueeze(0).unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{H, W})
                .mode(torch::kNearest)).squeeze();
    }

    const float near_depth = std::max(1e-6f, cam.near);
    const float min_depth = std::max(RGBD_min_depth_, near_depth);
    const float tau = std::max(
        1.0e-4f,
        opt_params_.rgbd_sdf_trunc_vox_ * voxel_model_->fixedVoxSize());
    const float center_band = 0.4f * tau;

    torch::Tensor valid_depth =
        torch::isfinite(rgbd_depth) &
        (rgbd_depth > min_depth) &
        (rgbd_depth < RGBD_max_depth_);
    if (!valid_depth.any().item<bool>()) {
        return zero;
    }

    const int free_samples = std::max(0, opt_params_.rgbd_sdf_free_samples_);
    const int surface_samples = std::max(0, opt_params_.rgbd_sdf_surface_samples_);
    const int samples_per_ray = free_samples + surface_samples;
    const int64_t requested_pixels =
        std::max<int64_t>(0, static_cast<int64_t>(opt_params_.rgbd_sdf_ray_pixels_));
    if (samples_per_ray <= 0 || requested_pixels <= 0 ||
        !std::isfinite(cam.fx) || !std::isfinite(cam.fy) ||
        std::abs(cam.fx) < 1.0e-6f || std::abs(cam.fy) < 1.0e-6f) {
        return zero;
    }

    torch::Tensor valid_uv =
        torch::nonzero(valid_depth).to(mDevice).to(torch::kLong).contiguous();
    if (!valid_uv.defined() || valid_uv.dim() != 2 || valid_uv.size(0) == 0) {
        return zero;
    }
    const int64_t valid_pixel_count = valid_uv.size(0);
    int64_t capped_pixels = requested_pixels;
    if (opt_params_.rgbd_sdf_max_samples_ > 0) {
        capped_pixels = std::min<int64_t>(
            capped_pixels,
            std::max<int64_t>(1, opt_params_.rgbd_sdf_max_samples_ / samples_per_ray));
    }
    const int64_t pixel_count =
        std::min<int64_t>(valid_pixel_count, capped_pixels);
    if (pixel_count <= 0) {
        return zero;
    }

    torch::Tensor pixel_row_idx;
    if (pixel_count < valid_pixel_count) {
        pixel_row_idx =
            torch::linspace(
                0.0,
                static_cast<double>(valid_pixel_count - 1),
                pixel_count,
                torch::TensorOptions().dtype(torch::kFloat32).device(mDevice))
                .round()
                .to(torch::kLong)
                .clamp(0, valid_pixel_count - 1)
                .contiguous();
    } else {
        pixel_row_idx =
            torch::arange(
                valid_pixel_count,
                torch::TensorOptions().dtype(torch::kLong).device(mDevice));
    }

    torch::Tensor uv = valid_uv.index_select(0, pixel_row_idx);
    torch::Tensor v_idx = uv.index({Slice(), 0});
    torch::Tensor u_idx = uv.index({Slice(), 1});
    torch::Tensor sampled_depth =
        rgbd_depth.view({H * W}).index_select(0, v_idx * W + u_idx).contiguous();

    std::vector<torch::Tensor> z_vec;
    std::vector<torch::Tensor> valid_vec;
    if (free_samples > 0) {
        torch::Tensor free_t =
            (torch::arange(
                 free_samples,
                 torch::TensorOptions().dtype(torch::kFloat32).device(mDevice)) +
             0.5f) /
            static_cast<float>(free_samples);
        torch::Tensor free_end =
            (sampled_depth - tau).clamp_min(min_depth);
        torch::Tensor free_span =
            (free_end - min_depth).clamp_min(0.0f);
        torch::Tensor z_free =
            min_depth + free_span.view({pixel_count, 1}) * free_t.view({1, free_samples});
        torch::Tensor valid_free =
            (free_span.view({pixel_count, 1}) > 1.0e-6f)
                .expand({pixel_count, free_samples});
        z_vec.push_back(z_free);
        valid_vec.push_back(valid_free);
    }
    if (surface_samples > 0) {
        torch::Tensor surface_t =
            (torch::arange(
                 surface_samples,
                 torch::TensorOptions().dtype(torch::kFloat32).device(mDevice)) +
             0.5f) /
            static_cast<float>(surface_samples);
        torch::Tensor z_surface =
            sampled_depth.view({pixel_count, 1}) - 1.5f * tau +
            (3.0f * tau) * surface_t.view({1, surface_samples});
        torch::Tensor valid_surface =
            (z_surface > min_depth) & (z_surface < RGBD_max_depth_);
        z_vec.push_back(z_surface);
        valid_vec.push_back(valid_surface);
    }

    torch::Tensor z_mat = torch::cat(z_vec, /*dim=*/1).contiguous();
    torch::Tensor sample_valid = torch::cat(valid_vec, /*dim=*/1).to(torch::kBool).contiguous();
    torch::Tensor depth_mat =
        sampled_depth.view({pixel_count, 1}).expand_as(z_mat).contiguous();
    torch::Tensor u_float =
        u_idx.to(torch::kFloat32).view({pixel_count, 1}).expand_as(z_mat);
    torch::Tensor v_float =
        v_idx.to(torch::kFloat32).view({pixel_count, 1}).expand_as(z_mat);
    torch::Tensor ray_x = (u_float - cam.cx) / cam.fx;
    torch::Tensor ray_y = (v_float - cam.cy) / cam.fy;

    torch::Tensor sample_mask =
        sample_valid &
        torch::isfinite(z_mat) &
        torch::isfinite(depth_mat) &
        torch::isfinite(ray_x) &
        torch::isfinite(ray_y);
    if (!sample_mask.any().item<bool>()) {
        return zero;
    }

    torch::Tensor z =
        z_mat.reshape({-1}).index({sample_mask.reshape({-1})}).contiguous();
    sampled_depth =
        depth_mat.reshape({-1}).index({sample_mask.reshape({-1})}).contiguous();
    torch::Tensor x =
        (ray_x.reshape({-1}).index({sample_mask.reshape({-1})}) * z).contiguous();
    torch::Tensor y =
        (ray_y.reshape({-1}).index({sample_mask.reshape({-1})}) * z).contiguous();
    torch::Tensor pts_cam = torch::stack({x, y, z}, /*dim=*/1).contiguous();

    torch::Tensor c2w = cam.c2w.to(mDevice, torch::kFloat32).contiguous();
    if (c2w.dim() != 2 || c2w.size(0) < 3 || c2w.size(1) < 4) {
        return zero;
    }
    torch::Tensor R = c2w.index({Slice(0, 3), Slice(0, 3)});
    torch::Tensor t = c2w.index({Slice(0, 3), 3}).view({1, 3});
    torch::Tensor pts_world = torch::matmul(pts_cam, R.t()) + t;

    torch::Tensor sdf_pred;
    torch::Tensor query_valid;
    try {
        std::tie(sdf_pred, query_valid) = voxel_model_->querySdfTrilinear(pts_world);
    } catch (const c10::Error&) {
        return zero;
    }
    if (!sdf_pred.defined() || !query_valid.defined() ||
        sdf_pred.numel() != z.numel() || query_valid.numel() != z.numel()) {
        return zero;
    }
    query_valid = query_valid.to(mDevice).to(torch::kBool).contiguous();
    if (!query_valid.any().item<bool>()) {
        return zero;
    }

    sdf_pred = sdf_pred.to(mDevice, torch::kFloat32).index({query_valid}).contiguous();
    z = z.index({query_valid}).contiguous();
    sampled_depth = sampled_depth.index({query_valid}).contiguous();

    torch::Tensor signed_dist =
        (sampled_depth - z).clamp(-tau, tau);
    torch::Tensor target_norm = signed_dist / tau;
    torch::Tensor pred_norm = sdf_pred / tau;

    torch::Tensor front = z < (sampled_depth - tau);
    torch::Tensor back = z > (sampled_depth + tau);
    torch::Tensor center =
        (z > (sampled_depth - center_band)) &
        (z < (sampled_depth + center_band));
    torch::Tensor excluded = torch::logical_or(torch::logical_or(front, back), center);
    torch::Tensor tail = torch::logical_not(excluded);

    auto masked_mse = [&](const torch::Tensor& values,
                          const torch::Tensor& mask) -> torch::Tensor {
        if (!mask.any().item<bool>()) {
            return zero;
        }
        return values.index({mask}).pow(2).mean();
    };

    torch::Tensor fs_loss =
        masked_mse(pred_norm - 1.0f, front);
    torch::Tensor center_loss =
        masked_mse(pred_norm - target_norm, center);
    torch::Tensor tail_loss =
        masked_mse(pred_norm - target_norm, tail);

    torch::Tensor loss =
        opt_params_.rgbd_sdf_w_fs_ * fs_loss +
        opt_params_.rgbd_sdf_w_center_ * center_loss +
        opt_params_.rgbd_sdf_w_tail_ * tail_loss;

    if (opt_params_.rgbd_sdf_end_ <= opt_params_.rgbd_sdf_from_ ||
        opt_params_.rgbd_sdf_end_mult_ == 1.0f) {
        return loss;
    }

    const float ratio = std::clamp(
        static_cast<float>(iteration - opt_params_.rgbd_sdf_from_) /
            static_cast<float>(opt_params_.rgbd_sdf_end_ -
                               opt_params_.rgbd_sdf_from_),
        0.0f,
        1.0f);
    const float mult = std::pow(opt_params_.rgbd_sdf_end_mult_, ratio);
    return loss * mult;
}

torch::Tensor VoxelMapper::computeRgbdNormalLoss(
    const std::shared_ptr<VoxelKeyframe>& kf,
    const sv::MiniCam& cam,
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    int iteration)
{
    auto zero = torch::zeros(
        {1},
        torch::TensorOptions().dtype(torch::kFloat32).device(mDevice));

    if (sensor_type_ != RGBD ||
        opt_params_.lambda_rgbd_normal_ <= 0.0f ||
        iteration < opt_params_.rgbd_normal_from_ ||
        iteration > opt_params_.rgbd_normal_end_ ||
        !kf ||
        kf->img_auxiliary_undist_.empty()) {
        return zero;
    }

    auto it_T = render_pkg.find("raw_T");
    if (it_T == render_pkg.end()) {
        it_T = render_pkg.find("T");
    }
    auto it_normal = render_pkg.find("raw_normal");
    if (it_normal == render_pkg.end()) {
        it_normal = render_pkg.find("normal");
    }
    if (it_T == render_pkg.end() || it_normal == render_pkg.end() ||
        !it_T->second.defined() || !it_normal->second.defined()) {
        return zero;
    }

    torch::Tensor raw_T =
        it_T->second.to(mDevice, torch::kFloat32).contiguous();
    if (raw_T.dim() == 4 && raw_T.size(0) == 1) {
        raw_T = raw_T.squeeze(0);
    }
    if (raw_T.dim() == 3 && raw_T.size(0) >= 1) {
        raw_T = raw_T.index({0});
    }
    if (raw_T.dim() != 2) {
        return zero;
    }

    torch::Tensor render_normal =
        it_normal->second.to(mDevice, torch::kFloat32).contiguous();
    if (render_normal.dim() == 4 && render_normal.size(0) == 1) {
        render_normal = render_normal.squeeze(0);
    }
    if (render_normal.dim() != 3 || render_normal.size(0) < 3) {
        return zero;
    }
    if (render_normal.size(0) > 3) {
        render_normal =
            render_normal.index({torch::indexing::Slice(0, 3)}).contiguous();
    }

    const int H = static_cast<int>(render_normal.size(1));
    const int W = static_cast<int>(render_normal.size(2));
    if (raw_T.size(0) != H || raw_T.size(1) != W) {
        return zero;
    }

    torch::Tensor target_normal;
    torch::Tensor valid_support;
    {
        torch::NoGradGuard no_grad;

        cv::cuda::GpuMat depth_gpu;
        depth_gpu.upload(kf->img_auxiliary_undist_);
        torch::Tensor rgbd_depth =
            tensor_utils::cvGpuMat2TorchTensor_Float32(depth_gpu)
                .to(mDevice, torch::kFloat32)
                .contiguous();
        if (rgbd_depth.dim() == 3 && rgbd_depth.size(0) == 1) {
            rgbd_depth = rgbd_depth.squeeze(0);
        }
        if (rgbd_depth.dim() != 2) {
            return zero;
        }
        if (rgbd_depth.size(0) != H || rgbd_depth.size(1) != W) {
            rgbd_depth = torch::nn::functional::interpolate(
                rgbd_depth.unsqueeze(0).unsqueeze(0),
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{H, W})
                    .mode(torch::kNearest)).squeeze();
        }

        const float near_depth = std::max(1e-6f, cam.near);
        const float min_depth = std::max(RGBD_min_depth_, near_depth);
        torch::Tensor valid_depth =
            torch::isfinite(rgbd_depth) &
            (rgbd_depth > min_depth) &
            (rgbd_depth < RGBD_max_depth_);
        valid_support =
            voxel_eval::validDepthSupportMask(valid_depth, opt_params_.rgbd_normal_ks_);
        if (!valid_support.any().item<bool>()) {
            return zero;
        }

        const float tol_cos = std::cos(
            opt_params_.rgbd_normal_tol_deg_ *
            static_cast<float>(M_PI) / 180.0f);
        target_normal = voxel_eval::depth2normalSVRaster(
            cam,
            rgbd_depth.clamp_min(near_depth),
            opt_params_.rgbd_normal_ks_,
            tol_cos);
    }

    torch::Tensor alpha = (1.0f - raw_T).clamp(0.0f, 1.0f);
    torch::Tensor mask =
        (target_normal != 0).any(0) &
        valid_support &
        (alpha > 0.8f);
    if (!mask.any().item<bool>()) {
        return zero;
    }

    torch::Tensor dot =
        (render_normal * target_normal).sum(0).clamp(-1.0f, 1.0f);
    torch::Tensor normal_angle = 1.0f - dot;
    // L1 normal term disabled to match SVRaster/HI-SLAM2-style cosine normal supervision.
    // torch::Tensor normal_l1 = (render_normal - target_normal).abs().sum(0);
    torch::Tensor mask_f = mask.to(render_normal.dtype());
    torch::Tensor denom = mask_f.sum().clamp_min(1.0f);
    // torch::Tensor loss = ((normal_angle + normal_l1) * mask_f).sum() / denom;
    torch::Tensor loss = (normal_angle * mask_f).sum() / denom;

    if (opt_params_.rgbd_normal_end_ <= opt_params_.rgbd_normal_from_ ||
        opt_params_.rgbd_normal_end_mult_ == 1.0f) {
        return loss;
    }

    const float ratio = std::clamp(
        static_cast<float>(iteration - opt_params_.rgbd_normal_from_) /
            static_cast<float>(opt_params_.rgbd_normal_end_ -
                               opt_params_.rgbd_normal_from_),
        0.0f,
        1.0f);
    const float mult = std::pow(opt_params_.rgbd_normal_end_mult_, ratio);
    return loss * mult;
}
