#pragma once

// Shared optimization-supervision and rendered-map utility declarations.

#include "include_voxel/mini_cam.h"
#include "include_voxel/voxel_keyframe.h"

#include <filesystem>
#include <limits>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

namespace ORB_SLAM3 { class System; }

namespace voxel_eval {

torch::Tensor l1Loss(
    const torch::Tensor& prediction,
    const torch::Tensor& target);
torch::Tensor mseLoss(
    const torch::Tensor& prediction,
    const torch::Tensor& target);
torch::Tensor huberLoss(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    float threshold);
torch::Tensor psnr(
    const torch::Tensor& prediction,
    const torch::Tensor& target);
torch::Tensor ssim(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    torch::DeviceType device_type,
    int window_size = 11,
    bool size_average = true);
torch::Tensor fastSsimLoss(
    torch::Tensor prediction,
    torch::Tensor target);
torch::Tensor probabilityConcentrationLoss(const torch::Tensor& probability);
torch::Tensor sparseDepthLoss(
    const torch::Tensor& raw_transmittance,
    const torch::Tensor& raw_depth,
    const torch::Tensor& sparse_uv,
    const torch::Tensor& sparse_depth);

struct DepthScaleFitStats
{
    bool valid = false;
    int64_t overlap_count = 0;
    int64_t ratio_count_before_trim = 0;
    int64_t ratio_count_after_trim = 0;
    float scale = std::numeric_limits<float>::quiet_NaN();
    float ratio_q05 = std::numeric_limits<float>::quiet_NaN();
    float ratio_q25 = std::numeric_limits<float>::quiet_NaN();
    float ratio_q50 = std::numeric_limits<float>::quiet_NaN();
    float ratio_q75 = std::numeric_limits<float>::quiet_NaN();
    float ratio_q95 = std::numeric_limits<float>::quiet_NaN();
    float pred_min = std::numeric_limits<float>::quiet_NaN();
    float pred_max = std::numeric_limits<float>::quiet_NaN();
    float gt_min = std::numeric_limits<float>::quiet_NaN();
    float gt_max = std::numeric_limits<float>::quiet_NaN();
};

torch::Tensor tensorToEvalMap(const torch::Tensor& tensor, int preferred_channel);
torch::Tensor tensorToEvalMapExactChannel(const torch::Tensor& tensor, int channel);
torch::Tensor depthTensorToEvalMap(const torch::Tensor& depth_tensor);
torch::Tensor transmittanceTensorToEvalMap(const torch::Tensor& t_tensor);
bool renderPkgToMetricDepthForEval(const std::unordered_map<std::string, torch::Tensor>& render_pkg, torch::Tensor& metric_depth);
bool renderPkgToSparseDepthLossMap(const std::unordered_map<std::string, torch::Tensor>& render_pkg, torch::Tensor& depth_loss_map);
bool computeSharedDepthVizRange(const torch::Tensor& pred_depth, const cv::Mat& gt_depth_meters, float valid_min_depth, float valid_max_depth, float& viz_min, float& viz_max);
bool computeDepthScaleFitStats(const torch::Tensor& pred_depth, const cv::Mat& gt_depth_meters, float valid_min_depth, float valid_max_depth, DepthScaleFitStats& stats_out);
bool computeWeightedMedianScale(const std::vector<std::pair<float, double>>& weighted_scales, float& scale_out);
cv::Mat depthTensorToCvMatFloat(const torch::Tensor& depth_tensor);
bool saveMetricDepthPngMillimeters(
    const torch::Tensor& depth_meters,
    const std::filesystem::path& output_path,
    float valid_min_depth,
    float valid_max_depth);
cv::Mat colorizeDepthMatJet(const cv::Mat& depth_meters, float valid_min_depth, float valid_max_depth, float viz_min, float viz_max);
cv::Mat colorizeFiniteScalarMat(const cv::Mat& values, float viz_min, float viz_max, int colormap);
cv::Mat colorizeFiniteScalarMatJet(const cv::Mat& values, float viz_min, float viz_max);
cv::Mat appendColormapLegendBar(const cv::Mat& image_bgr, float viz_min, float viz_max, const std::string& unit_suffix, int colormap, const std::string& high_label, const std::string& low_label);
cv::Mat appendJetLegendBar(const cv::Mat& image_bgr, float viz_min, float viz_max, const std::string& unit_suffix);
bool renderPkgToNormalForEval(const std::unordered_map<std::string, torch::Tensor>& render_pkg, torch::Tensor& render_normal);
torch::Tensor depthToNormal(const sv::MiniCam& cam, const torch::Tensor& depth, int ks, float tol_cos);
torch::Tensor validDepthSupportMask(const torch::Tensor& valid, int ks);
torch::Tensor normalDepthConsistencyLossSvrecon(const sv::MiniCam& cam, const std::unordered_map<std::string, torch::Tensor>& render_pkg, int ks, float tol_deg);
cv::Mat colorizeNormalMapBgr(const torch::Tensor& normal_tensor);
torch::Tensor normalWorldToCameraForViz(const sv::MiniCam& cam, const torch::Tensor& normal_world);
cv::Mat blackRgbImage(int height, int width);
cv::Mat bgrToRgbImage(const cv::Mat& image_bgr);
cv::Mat chwRgbFloatTensorToU8Rgb(torch::Tensor image);
cv::Mat makeDepthGapMaskRgb(const cv::Mat& pred_depth_meters, const cv::Mat& gt_depth_meters, const torch::Tensor& eval_mask, float valid_min_depth, float valid_max_depth, double& gap_percent_out);
bool sparseSamplesToDepthMat(const torch::Tensor& sparse_uv, const torch::Tensor& sparse_depth, int image_width, int image_height, cv::Mat& depth_meters);
std::filesystem::path runtimeOrbDepthDebugDir(const std::filesystem::path& result_root);
void copyPngFilesToDirectory(const std::filesystem::path& source_dir, const std::filesystem::path& target_dir);
bool saveAccumulatedOrbDepthProjectionPng(const std::shared_ptr<ORB_SLAM3::System>& slam, const std::shared_ptr<VoxelKeyframe>& kf, const std::filesystem::path& output_dir, int iteration, float valid_min_depth, float valid_max_depth, const torch::Tensor& aligned_da_depth = torch::Tensor());
void saveDepthComparisonDebugPngs(const torch::Tensor& pred_depth, const cv::Mat& gt_depth_meters, float valid_min_depth, float valid_max_depth, const std::filesystem::path& pred_path, const std::filesystem::path& gt_path, const std::filesystem::path& pair_path, std::optional<float> depth_scale = std::nullopt);
bool getKeyframeDepthMetersForEval(const std::shared_ptr<VoxelKeyframe>& pkf, int expected_h, int expected_w, cv::Mat& depth_meters);

} // namespace voxel_eval
