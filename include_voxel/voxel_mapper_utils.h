#pragma once

#include <filesystem>
#include <Eigen/Geometry>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <torch/torch.h>
#include <tuple>
#include <unordered_map>
#include <vector>

class VoxelKeyframe;

namespace voxel_utils {

// Photo-SLAM inactive-geometry CUDA implementation, localized for voxel_core.
std::tuple<torch::Tensor, torch::Tensor>
monocularPinholeInactiveGeoDensifyBySearchingNeighborhoodKeypoints(
    torch::Tensor& kps_pixel,
    torch::Tensor& kps_has3D,
    torch::Tensor& kps_point_local,
    torch::Tensor& colors,
    float max_pixel_dist,
    std::vector<float>& intr,
    int width);

std::string toLowerCopy(std::string s);
torch::Tensor cvMatToTorchTensorFloat32(
    cv::Mat& mat,
    torch::DeviceType device_type);
cv::Mat torchTensorToCvMatFloat32(torch::Tensor& tensor);
torch::Tensor cvGpuMatToTorchTensorFloat32(cv::cuda::GpuMat& mat);
torch::Tensor eigenMatrixToTorchTensor(
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix,
    torch::DeviceType device_type = torch::kCUDA);
int parseFrameIdFromPath(const std::string& path);
double parseFrameTimestampFromPath(const std::string& path);
int frameIdFromIntegerTimestamp(double timestamp);
void saveKeyframeFrameIdMap(
    const std::map<std::size_t, std::shared_ptr<VoxelKeyframe>>& keyframes,
    const std::filesystem::path& path);
bool depthMatToMeters(const cv::Mat& depth_in, cv::Mat& depth_meters);
bool loadReplicaDepthFromRgbPath(const std::string& rgb_filename, cv::Mat& depth_meters);
bool loadTumDepthFromRgbPath(const std::string& rgb_filename, cv::Mat& depth_meters);
bool loadScanNetDepthFromRgbPath(const std::string& rgb_filename, cv::Mat& depth_meters);

torch::Tensor normalizeBoolMaskOrZeros(
    torch::Tensor mask,
    int64_t N,
    const torch::Device& device);

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
