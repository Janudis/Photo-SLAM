#pragma once

#include "include_voxel/voxel_keyframe.h"

#include <filesystem>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

namespace voxel_utils {

inline constexpr int kRenderedCandidateSourceDepthInsert = 1;
inline constexpr int kRenderedCandidateSourceMonoPrior = 3;

std::string toLowerCopy(std::string s);
int parseFrameIdFromPath(const std::string& path);
int frameIdFromIntegerTimestamp(double timestamp);
void saveKeyframeFrameIdMap(
    const std::map<std::size_t, std::shared_ptr<VoxelKeyframe>>& keyframes,
    const std::filesystem::path& path);
std::filesystem::path resolveNvbloxMeshPath(const std::string& configured_path);

bool depthMatToMeters(const cv::Mat& depth_in, cv::Mat& depth_meters);
bool loadReplicaDepthFromRgbPath(const std::string& rgb_filename, cv::Mat& depth_meters);
bool loadTumDepthFromRgbPath(const std::string& rgb_filename, cv::Mat& depth_meters);

torch::Tensor normalizeBoolMaskOrZeros(
    torch::Tensor mask,
    int64_t N,
    const torch::Device& device);

int64_t tensorRowCount(const torch::Tensor& tensor);

void transformPoints(torch::Tensor& points, const torch::Tensor& Twc);
torch::Tensor reprojectDepthPinholeVoxel(
    const torch::Tensor& depth,
    const std::vector<float>& intr,
    int image_width);

bool renderPkgToDepthAlphaMaps(
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    int expected_h,
    int expected_w,
    torch::Tensor& depth_cpu,
    torch::Tensor& alpha_cpu,
    torch::Tensor& n_contrib_cpu);

} // namespace voxel_utils
