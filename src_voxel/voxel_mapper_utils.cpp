#include "include_voxel/voxel_mapper_utils.h"
#include "include_voxel/voxel_keyframe.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <opencv2/core/cuda.hpp>
#include <sstream>
#include <unordered_map>

namespace voxel_utils {
namespace {

torch::Tensor squeezeRenderMap2D(torch::Tensor tensor)
{
    if (!tensor.defined()) {
        return tensor;
    }
    if (tensor.dim() == 4 && tensor.size(0) == 1) {
        tensor = tensor.squeeze(0);
    }
    if (tensor.dim() == 3 && tensor.size(0) >= 1) {
        tensor = tensor.index({0});
    }
    return tensor.contiguous();
}

void gpuMatNoopDeleter(void*) {}

} // namespace

std::string toLowerCopy(std::string s) {
    std::transform(
        s.begin(),
        s.end(),
        s.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

torch::Tensor cvMatToTorchTensorFloat32(
    cv::Mat& mat,
    const torch::DeviceType device_type)
{
    torch::Tensor mat_tensor;
    torch::Tensor tensor;
    if (mat.channels() == 1) {
        mat_tensor = torch::from_blob(mat.data, {mat.rows, mat.cols});
        tensor = mat_tensor.clone().to(device_type);
    } else if (mat.channels() == 3) {
        mat_tensor = torch::from_blob(
            mat.data, {mat.rows, mat.cols, mat.channels()});
        tensor = mat_tensor.clone().to(device_type);
        tensor = tensor.permute({2, 0, 1});
    } else {
        std::cerr << "The mat has unsupported number of channels!" << std::endl;
    }
    return tensor.contiguous();
}

cv::Mat torchTensorToCvMatFloat32(torch::Tensor& tensor)
{
    cv::Mat mat;
    torch::Tensor mat_tensor = tensor.clone();
    if (mat_tensor.dim() == 2) {
        mat = cv::Mat(
            mat_tensor.size(0), mat_tensor.size(1), CV_32FC1,
            mat_tensor.data_ptr<float>());
    } else if (mat_tensor.dim() == 3) {
        mat_tensor = mat_tensor.detach().permute({1, 2, 0})
                         .to(torch::kCPU).contiguous();
        mat = cv::Mat(
            mat_tensor.size(0), mat_tensor.size(1), CV_32FC3,
            mat_tensor.data_ptr<float>());
    } else {
        std::cerr << "The tensor has unsupported number of dimensions!" << std::endl;
    }
    return mat.clone();
}

torch::Tensor cvGpuMatToTorchTensorFloat32(cv::cuda::GpuMat& mat)
{
    const int64_t step = mat.step / sizeof(float);
    torch::Tensor mat_tensor;
    torch::Tensor tensor;
    if (mat.channels() == 1) {
        mat_tensor = torch::from_blob(
            mat.data,
            {mat.rows, mat.cols},
            std::vector<int64_t>{step, 1},
            gpuMatNoopDeleter,
            torch::TensorOptions().device(torch::kCUDA));
        tensor = mat_tensor.clone();
    } else if (mat.channels() == 3) {
        mat_tensor = torch::from_blob(
            mat.data,
            {mat.rows, mat.cols, mat.channels()},
            std::vector<int64_t>{step, mat.channels(), 1},
            gpuMatNoopDeleter,
            torch::TensorOptions().device(torch::kCUDA));
        tensor = mat_tensor.clone().permute({2, 0, 1});
    } else {
        std::cerr << "The mat has unsupported number of channels!" << std::endl;
    }
    return tensor.contiguous();
}

torch::Tensor eigenMatrixToTorchTensor(
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix,
    const torch::DeviceType device_type)
{
    auto transposed = eigen_matrix;
    transposed.transposeInPlace();
    torch::Tensor tensor = torch::from_blob(
        transposed.data(),
        {eigen_matrix.rows(), eigen_matrix.cols()},
        torch::TensorOptions().dtype(torch::kFloat32)).clone();
    return tensor.to(device_type);
}

int parseFrameIdFromPath(const std::string& path)
{
    const std::string name = std::filesystem::path(path).stem().string();
    std::string digits;
    for (char c : name) {
        if (std::isdigit(static_cast<unsigned char>(c))) {
            digits.push_back(c);
        }
    }
    if (digits.empty()) {
        return -1;
    }
    try {
        return std::stoi(digits);
    } catch (...) {
        return -1;
    }
}
double parseFrameTimestampFromPath(const std::string& path)
{
    const std::string stem = std::filesystem::path(path).stem().string();
    if (!stem.empty()) {
        try {
            std::size_t consumed = 0;
            const double timestamp = std::stod(stem, &consumed);
            if (consumed == stem.size() && std::isfinite(timestamp)) {
                return timestamp;
            }
        } catch (...) {
        }
    }

    const int frame_id = parseFrameIdFromPath(path);
    return frame_id >= 0
        ? static_cast<double>(frame_id)
        : std::numeric_limits<double>::quiet_NaN();
}
int frameIdFromIntegerTimestamp(double timestamp)
{
    if (!std::isfinite(timestamp)) {
        return -1;
    }
    const double rounded = std::round(timestamp);
    if (std::abs(timestamp - rounded) > 1.0e-6 ||
        rounded < 0.0 ||
        rounded > static_cast<double>(std::numeric_limits<int>::max())) {
        return -1;
    }
    return static_cast<int>(rounded);
}
void saveKeyframeFrameIdMap(
    const std::map<std::size_t, std::shared_ptr<VoxelKeyframe>>& keyframes,
    const std::filesystem::path& path)
{
    std::ofstream out(path);
    if (!out.is_open()) {
        return;
    }
    out << "# ours_kf_id replica_frame_id\n";
    for (const auto& [kf_id, kf] : keyframes) {
        if (!kf) {
            continue;
        }
        int source_frame_id = kf->source_frame_id_;
        if (source_frame_id < 0) {
            source_frame_id = parseFrameIdFromPath(kf->img_filename_);
        }
        if (source_frame_id < 0) {
            source_frame_id = static_cast<int>(kf_id);
        }
        out << kf_id << " " << source_frame_id << "\n";
    }
}
bool depthMatToMeters(const cv::Mat& depth_in, cv::Mat& depth_meters)
{
    if (depth_in.empty()) {
        return false;
    }

    cv::Mat d = depth_in;
    if (d.channels() > 1) {
        cv::extractChannel(d, d, 0);
    }

    if (d.type() == CV_32FC1) {
        depth_meters = d;
        return true;
    }
    if (d.type() == CV_16UC1) {
        // Handle both common 16-bit conventions when dataset metadata is not
        // available:
        // - millimeter depth: depth_m = raw / 1000
        // - Replica-style uint16 encoding: depth_m = raw / 6553.5
        double max_val = 0.0;
        cv::minMaxLoc(d, nullptr, &max_val);
        const double scale = (max_val > 20000.0) ? (1.0 / 6553.5) : (1.0 / 1000.0);
        d.convertTo(depth_meters, CV_32FC1, scale);
        return true;
    }

    d.convertTo(depth_meters, CV_32FC1);
    return true;
}

static bool encodedDepthMatToMeters(
    const cv::Mat& depth_in,
    const double units_to_meters,
    cv::Mat& depth_meters)
{
    if (depth_in.empty()) {
        return false;
    }
    cv::Mat depth = depth_in;
    if (depth.channels() > 1) {
        cv::extractChannel(depth, depth, 0);
    }
    if (depth.type() == CV_16UC1) {
        depth.convertTo(depth_meters, CV_32FC1, units_to_meters);
        return true;
    }
    return depthMatToMeters(depth, depth_meters);
}

bool loadReplicaDepthFromRgbPath(const std::string& rgb_filename, cv::Mat& depth_meters)
{
    if (rgb_filename.empty()) {
        return false;
    }

    const std::filesystem::path rgb_path(rgb_filename);
    if (!std::filesystem::exists(rgb_path)) {
        return false;
    }

    const std::string name = rgb_path.filename().string();
    if (name.rfind("frame", 0) != 0) {
        return false;
    }

    const std::filesystem::path parent = rgb_path.parent_path();
    const std::string stem = rgb_path.stem().string();   // e.g., frame000123
    const std::string suffix_stem = (stem.size() > 5 ? stem.substr(5) : std::string());
    const std::string suffix_name = name.substr(5);      // e.g., 000123.jpg

    std::vector<std::filesystem::path> candidates;
    candidates.push_back(parent / ("depth" + suffix_name));
    if (!suffix_stem.empty()) {
        candidates.push_back(parent / ("depth" + suffix_stem + ".png"));
        candidates.push_back(parent / ("depth" + suffix_stem + ".exr"));
        candidates.push_back(parent / ("depth" + suffix_stem + ".tiff"));
        candidates.push_back(parent / ("depth" + suffix_stem + ".tif"));
    }

    for (const auto& p : candidates) {
        if (!std::filesystem::exists(p)) {
            continue;
        }
        const cv::Mat depth_raw = cv::imread(p.string(), cv::IMREAD_UNCHANGED);
        if (depth_raw.empty()) {
            continue;
        }
        return encodedDepthMatToMeters(
            depth_raw, 1.0 / 6553.5, depth_meters);
    }

    return false;
}

bool loadTumDepthFromRgbPath(const std::string& rgb_filename, cv::Mat& depth_meters)
{
    if (rgb_filename.empty()) {
        return false;
    }

    const std::filesystem::path rgb_path(rgb_filename);
    if (!std::filesystem::exists(rgb_path)) {
        return false;
    }

    const std::filesystem::path rgb_dir = rgb_path.parent_path();
    if (rgb_dir.filename() != "rgb") {
        return false;
    }

    const std::filesystem::path dataset_root = rgb_dir.parent_path();
    const std::filesystem::path depth_dir = dataset_root / "depth";
    const std::filesystem::path depth_txt = dataset_root / "depth.txt";
    if (!std::filesystem::exists(depth_dir)) {
        return false;
    }

    const std::string stem = rgb_path.stem().string();
    const std::filesystem::path exact_depth_path = depth_dir / (stem + rgb_path.extension().string());
    if (std::filesystem::exists(exact_depth_path)) {
        const cv::Mat depth_raw = cv::imread(exact_depth_path.string(), cv::IMREAD_UNCHANGED);
        if (!depth_raw.empty()) {
            return encodedDepthMatToMeters(
                depth_raw, 1.0 / 5000.0, depth_meters);
        }
    }

    double rgb_ts = 0.0;
    try {
        rgb_ts = std::stod(stem);
    } catch (...) {
        return false;
    }

    struct TumDepthIndexEntry {
        double timestamp = 0.0;
        std::filesystem::path path;
    };

    static std::mutex s_tum_depth_cache_mutex;
    static std::unordered_map<std::string, std::vector<TumDepthIndexEntry>> s_tum_depth_cache;

    std::vector<TumDepthIndexEntry> depth_index;
    {
        std::lock_guard<std::mutex> lock(s_tum_depth_cache_mutex);
        auto it = s_tum_depth_cache.find(dataset_root.string());
        if (it == s_tum_depth_cache.end()) {
            std::vector<TumDepthIndexEntry> parsed;
            if (std::filesystem::exists(depth_txt)) {
                std::ifstream in(depth_txt);
                std::string line;
                while (std::getline(in, line)) {
                    if (line.empty() || line[0] == '#') {
                        continue;
                    }
                    std::istringstream iss(line);
                    double ts = 0.0;
                    std::string rel_path;
                    if (!(iss >> ts >> rel_path)) {
                        continue;
                    }
                    std::filesystem::path p = dataset_root / rel_path;
                    if (!std::filesystem::exists(p)) {
                        continue;
                    }
                    parsed.push_back({ts, p});
                }
            }
            std::sort(
                parsed.begin(),
                parsed.end(),
                [](const TumDepthIndexEntry& a, const TumDepthIndexEntry& b) {
                    return a.timestamp < b.timestamp;
                });
            it = s_tum_depth_cache.emplace(dataset_root.string(), std::move(parsed)).first;
        }
        depth_index = it->second;
    }

    if (depth_index.empty()) {
        return false;
    }

    auto lb = std::lower_bound(
        depth_index.begin(),
        depth_index.end(),
        rgb_ts,
        [](const TumDepthIndexEntry& e, double t) {
            return e.timestamp < t;
        });

    auto best_it = depth_index.end();
    double best_dt = std::numeric_limits<double>::infinity();
    if (lb != depth_index.end()) {
        best_it = lb;
        best_dt = std::abs(lb->timestamp - rgb_ts);
    }
    if (lb != depth_index.begin()) {
        auto prev = std::prev(lb);
        const double prev_dt = std::abs(prev->timestamp - rgb_ts);
        if (prev_dt < best_dt) {
            best_it = prev;
            best_dt = prev_dt;
        }
    }

    constexpr double kTumMaxDepthAssocDeltaSec = 0.05;
    if (best_it == depth_index.end() || !(best_dt <= kTumMaxDepthAssocDeltaSec)) {
        return false;
    }

    const cv::Mat depth_raw = cv::imread(best_it->path.string(), cv::IMREAD_UNCHANGED);
    if (depth_raw.empty()) {
        return false;
    }
    return encodedDepthMatToMeters(
        depth_raw, 1.0 / 5000.0, depth_meters);
}

bool loadScanNetDepthFromRgbPath(
    const std::string& rgb_filename,
    cv::Mat& depth_meters)
{
    if (rgb_filename.empty()) {
        return false;
    }

    const std::filesystem::path rgb_path(rgb_filename);
    if (!std::filesystem::exists(rgb_path) ||
        rgb_path.parent_path().filename() != "color") {
        return false;
    }

    const std::filesystem::path depth_path =
        rgb_path.parent_path().parent_path() / "depth" /
        (rgb_path.stem().string() + ".png");
    if (!std::filesystem::exists(depth_path)) {
        return false;
    }

    const cv::Mat depth_raw =
        cv::imread(depth_path.string(), cv::IMREAD_UNCHANGED);
    return !depth_raw.empty() && encodedDepthMatToMeters(
        depth_raw, 1.0 / 1000.0, depth_meters);
}
torch::Tensor normalizeBoolMaskOrZeros(
    torch::Tensor mask,
    const int64_t N,
    const torch::Device& device)
{
    if (!mask.defined() || mask.numel() != N) {
        return torch::zeros(
            {N},
            torch::TensorOptions().dtype(torch::kBool).device(device));
    }
    if (mask.dim() == 2 && mask.size(1) == 1) {
        mask = mask.squeeze(1);
    } else if (mask.dim() != 1) {
        mask = mask.reshape({N});
    }
    return mask.to(device).to(torch::kBool).contiguous();
}

void transformPoints(torch::Tensor& points, const torch::Tensor& Twc)
{
    namespace idx = torch::indexing;

    if (!points.defined() || points.numel() == 0) {
        return;
    }

    const auto N = points.size(0);
    auto opts = torch::TensorOptions()
                    .dtype(points.dtype())
                    .device(points.device());
    torch::Tensor ones = torch::ones({N, 1}, opts);
    torch::Tensor pts_h = torch::cat({points, ones}, 1);
    torch::Tensor pts_w = torch::matmul(pts_h, Twc);
    points = pts_w.index({idx::Slice(), idx::Slice(0, 3)}).contiguous();
}

torch::Tensor reprojectDepthPinholeVoxel(
    const torch::Tensor& depth,
    const std::vector<float>& intr,
    int image_width)
{
    if (!depth.defined() || depth.dim() != 1 || image_width <= 0 || intr.size() < 4) {
        return torch::empty({0, 3}, depth.options().dtype(torch::kFloat32));
    }

    const int64_t N = depth.size(0);
    if (N <= 0 || N % image_width != 0) {
        return torch::empty({0, 3}, depth.options().dtype(torch::kFloat32));
    }

    const int64_t H = N / image_width;
    const int64_t W = image_width;
    const float fx = intr[0];
    const float fy = intr[1];
    const float cx = intr[2];
    const float cy = intr[3];
    if (std::abs(fx) <= 1e-8f || std::abs(fy) <= 1e-8f) {
        return torch::empty({0, 3}, depth.options().dtype(torch::kFloat32));
    }

    auto opts = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(depth.device());
    torch::Tensor u = torch::arange(W, opts).view({1, W}).repeat({H, 1}).flatten();
    torch::Tensor v = torch::arange(H, opts).view({H, 1}).repeat({1, W}).flatten();
    torch::Tensor z = depth.to(opts);
    torch::Tensor x = (u - cx) / fx * z;
    torch::Tensor y = (v - cy) / fy * z;
    return torch::stack({x, y, z}, 1).contiguous();
}

bool renderPkgToDepthAlphaMaps(
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    int expected_h,
    int expected_w,
    torch::Tensor& depth_cpu,
    torch::Tensor& alpha_cpu,
    torch::Tensor& n_contrib_cpu)
{
    depth_cpu = torch::Tensor();
    alpha_cpu = torch::Tensor();
    n_contrib_cpu = torch::Tensor();

    auto it_depth = render_pkg.find("raw_depth");
    if (it_depth == render_pkg.end()) {
        it_depth = render_pkg.find("depth");
    }
    auto it_T = render_pkg.find("raw_T");
    if (it_T == render_pkg.end()) {
        it_T = render_pkg.find("T");
    }
    auto it_n_contrib = render_pkg.find("n_contrib");
    if (it_n_contrib == render_pkg.end()) {
        it_n_contrib = render_pkg.find("raw_n_contrib");
    }
    if (it_depth == render_pkg.end() ||
        it_T == render_pkg.end() ||
        it_n_contrib == render_pkg.end() ||
        !it_depth->second.defined() ||
        !it_T->second.defined() ||
        !it_n_contrib->second.defined()) {
        return false;
    }

    torch::Tensor raw_depth = it_depth->second;
    if (raw_depth.dim() == 4 && raw_depth.size(0) == 1) {
        raw_depth = raw_depth.squeeze(0);
    }
    if (raw_depth.dim() != 3 || raw_depth.size(0) < 1) {
        return false;
    }

    torch::Tensor render_depth_raw =
        raw_depth.index({0}).detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor render_T =
        squeezeRenderMap2D(it_T->second).detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor render_n_contrib =
        squeezeRenderMap2D(it_n_contrib->second).detach().to(torch::kCPU).to(torch::kInt32).contiguous();

    if (render_depth_raw.dim() != 2 ||
        render_T.dim() != 2 ||
        render_n_contrib.dim() != 2) {
        return false;
    }
    if (render_depth_raw.size(0) != expected_h ||
        render_depth_raw.size(1) != expected_w ||
        render_T.size(0) != expected_h ||
        render_T.size(1) != expected_w ||
        render_n_contrib.size(0) != expected_h ||
        render_n_contrib.size(1) != expected_w) {
        return false;
    }

    alpha_cpu = (1.0f - render_T).contiguous();
    depth_cpu = (render_depth_raw / alpha_cpu.clamp_min(1e-6f)).contiguous();
    n_contrib_cpu = render_n_contrib.contiguous();
    return true;
}

} // namespace voxel_utils
