#include "include_voxel/voxel_mapper.h"
#include "include_voxel/voxel_mapper_evaluation.h"
#include "include/stereo_vision.h"

#include <pybind11/embed.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <opencv2/flann.hpp>
#include <random>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "ORB-SLAM3/include/Atlas.h"
#include "ORB-SLAM3/include/MapPoint.h"

namespace py = pybind11;

namespace {
static bool depthMatToMeters(const cv::Mat& depth_in, cv::Mat& depth_meters);
static bool loadReplicaDepthFromRgbPath(const std::string& rgb_filename, cv::Mat& depth_meters);
static bool loadTumDepthFromRgbPath(const std::string& rgb_filename, cv::Mat& depth_meters);
static bool computeSharedDepthVizRange(const torch::Tensor& pred_depth, const cv::Mat& gt_depth_meters, float valid_min_depth, float valid_max_depth, float& viz_min, float& viz_max);
static cv::Mat depthTensorToCvMatFloat(const torch::Tensor& depth_tensor);
static cv::Mat colorizeDepthMatJet(const cv::Mat& depth_meters, float valid_min_depth, float valid_max_depth, float viz_min, float viz_max);
static cv::Mat colorizeFiniteScalarMatJet(const cv::Mat& values, float viz_min, float viz_max);
static cv::Mat appendJetLegendBar(const cv::Mat& image_bgr, float viz_min, float viz_max, const std::string& unit_suffix);

constexpr int kRenderedCandidateSourceDepthInsert = 1;
constexpr int kRenderedCandidateSourceMonoPrior = 3;
constexpr float kMonoPriorFillHolesEmptyDepthEps = 1e-6f;
constexpr float kMonoPriorAlignInitialInlierRelErr = 0.50f;
constexpr float kMonoPriorAlignFinalInlierRelErr = 0.30f;

struct MonoPriorAlignmentStats {
    int64_t num_valid_anchors = 0;
    int64_t num_fit_anchors = 0;
    float sparse_depth_q05 = std::numeric_limits<float>::quiet_NaN();
    float sparse_depth_q95 = std::numeric_limits<float>::quiet_NaN();
    float scale = std::numeric_limits<float>::quiet_NaN();
    float shift = std::numeric_limits<float>::quiet_NaN();
    float median_rel_depth_error = std::numeric_limits<float>::quiet_NaN();
    float p90_rel_depth_error = std::numeric_limits<float>::quiet_NaN();
    float final_inlier_ratio = std::numeric_limits<float>::quiet_NaN();
};
std::string toLowerCopy(std::string s) {
    std::transform(
        s.begin(),
        s.end(),
        s.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}
bool isMetric3DModelId(const std::string& model_id) {
    const std::string model = toLowerCopy(model_id);
    return model.find("metric3d") != std::string::npos ||
           (model.find("depth-anything") != std::string::npos &&
            model.find("metric") != std::string::npos);
}
const char* monoPriorLogPrefix(const std::string& model_id) {
    return isMetric3DModelId(model_id) ? "[MetricDepth]" : "[DepthAnythingV2]";
}

std::string normalizeMonoPriorLossMode(std::string mode)
{
    mode = toLowerCopy(std::move(mode));
    if (mode == "orb_aligned" || mode == "orb-aligned" || mode == "slam" ||
        mode == "systems") {
        return "aligned";
    }
    if (mode == "sv" || mode == "svraster") {
        return "svraster";
    }
    if (mode == "geo" || mode == "geosvr") {
        return "geosvr";
    }
    return mode;
}

std::string normalizeMonoPriorNormalMode(std::string mode)
{
    mode = toLowerCopy(std::move(mode));
    if (mode == "orb" || mode == "orb_aligned" || mode == "orb-aligned" ||
        mode == "anchors" || mode == "keyframe" || mode == "aligned") {
        return "aligned";
    }
    if (mode == "geo" || mode == "geosvr") {
        return "geosvr";
    }
    return mode;
}

std::string normalizeMonoPriorDensifyAlignmentMode(std::string mode)
{
    mode = toLowerCopy(std::move(mode));
    if (mode == "render" || mode == "rendered_depth" || mode == "svraster") {
        return "rendered";
    }
    if (mode == "orb_aligned" || mode == "orb-aligned" || mode == "anchors" ||
        mode == "keyframe") {
        return "orb";
    }
    if (mode == "da_prior" || mode == "prior_da" ||
        mode == "prior-depth-anything" || mode == "prior_depth_anything" ||
        mode == "priorda") {
        return "da_prior";
    }
    return mode;
}

torch::Tensor squeezeRenderMap2D(torch::Tensor t) {
    if (!t.defined()) {
        return t;
    }
    if (t.dim() == 4 && t.size(0) == 1) t = t.squeeze(0);
    if (t.dim() == 3 && t.size(0) >= 1) t = t.index({0});
    return t.contiguous();
}
bool renderPkgToMonoPriorAlignmentMaps(
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
    if (it_depth == render_pkg.end()) it_depth = render_pkg.find("depth");
    auto it_T = render_pkg.find("raw_T");
    if (it_T == render_pkg.end()) it_T = render_pkg.find("T");
    auto it_n_contrib = render_pkg.find("n_contrib");
    if (it_n_contrib == render_pkg.end()) it_n_contrib = render_pkg.find("raw_n_contrib");
    if (it_depth == render_pkg.end() || it_T == render_pkg.end() ||
        it_n_contrib == render_pkg.end() ||
        !it_depth->second.defined() || !it_T->second.defined() ||
        !it_n_contrib->second.defined()) {
        return false;
    }

    torch::Tensor raw_depth = it_depth->second;
    if (raw_depth.dim() == 4 && raw_depth.size(0) == 1) raw_depth = raw_depth.squeeze(0);
    if (raw_depth.dim() != 3 || raw_depth.size(0) < 1) {
        return false;
    }

    torch::Tensor render_depth_raw =
        raw_depth.index({0}).detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor render_T =
        squeezeRenderMap2D(it_T->second).detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor render_n_contrib =
        squeezeRenderMap2D(it_n_contrib->second).detach().to(torch::kCPU).to(torch::kInt32).contiguous();
    if (render_depth_raw.dim() != 2 || render_T.dim() != 2 || render_n_contrib.dim() != 2) {
        return false;
    }
    if (render_depth_raw.size(0) != expected_h || render_depth_raw.size(1) != expected_w ||
        render_T.size(0) != expected_h || render_T.size(1) != expected_w ||
        render_n_contrib.size(0) != expected_h || render_n_contrib.size(1) != expected_w) {
        return false;
    }

    alpha_cpu = (1.0f - render_T).contiguous();
    depth_cpu = (render_depth_raw / alpha_cpu.clamp_min(1e-6f)).contiguous();
    n_contrib_cpu = render_n_contrib.contiguous();
    return true;
}

static torch::Tensor normalWorldToCameraForViz(
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

static cv::Mat blackRgbImage(int height, int width)
{
    if (height <= 0 || width <= 0) {
        return cv::Mat();
    }
    return cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
}

static cv::Mat bgrToRgbImage(const cv::Mat& image_bgr)
{
    if (image_bgr.empty()) {
        return cv::Mat();
    }
    cv::Mat image_rgb;
    cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);
    return image_rgb;
}

static cv::Mat chwRgbFloatTensorToU8Rgb(torch::Tensor image)
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

static cv::Mat makeDepthGapMaskRgb(
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

static bool sparseSamplesToDepthMat(
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

static const std::string& runtimeOrbDepthDebugRunTag()
{
    static const std::string tag = []() {
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        return std::to_string(now);
    }();
    return tag;
}

static std::filesystem::path runtimeOrbDepthDebugDir(
    const std::filesystem::path& result_root)
{
    return result_root / (".orb_depth_debug_" + runtimeOrbDepthDebugRunTag());
}

static void copyPngFilesToDirectory(
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

static cv::Scalar relDepthErrorColorBgr(float rel_error)
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

static bool saveAccumulatedOrbDepthProjectionPng(
    const std::shared_ptr<ORB_SLAM3::System>& slam,
    const std::shared_ptr<VoxelKeyframe>& kf,
    const std::filesystem::path& output_dir,
    int iteration,
    float valid_min_depth,
    float valid_max_depth,
    const torch::Tensor& aligned_da_depth = torch::Tensor())
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

static std::string monoPriorCacheKeyForKeyframe(const std::shared_ptr<VoxelKeyframe>& pkf)
{
    if (!pkf) {
        return "kf_invalid";
    }

    std::string key;
    if (!pkf->img_filename_.empty()) {
        key = std::filesystem::path(pkf->img_filename_).stem().string();
    }
    if (key.empty()) {
        std::ostringstream oss;
        oss << "kf_" << std::setw(6) << std::setfill('0') << pkf->fid_;
        key = oss.str();
    }

    key = std::regex_replace(key, std::regex(R"([^A-Za-z0-9_.-])"), "_");
    return key;
}

static bool buildSparseDepthFromKeyframeOrbAnchors(
    const std::shared_ptr<VoxelKeyframe>& kf,
    int image_width,
    int image_height,
    torch::Tensor& sparse_uv,
    torch::Tensor& sparse_depth)
{
    sparse_uv = torch::Tensor();
    sparse_depth = torch::Tensor();
    if (!kf || image_width <= 0 || image_height <= 0) {
        return false;
    }

    const size_t n_pix = kf->kps_pixel_.size() / 2;
    const size_t n_xyz = kf->kps_point_local_.size() / 3;
    const size_t N = std::min(n_pix, n_xyz);
    if (N == 0) {
        return false;
    }

    const float sx =
        (kf->image_width_ > 0)
            ? (static_cast<float>(image_width) / static_cast<float>(kf->image_width_))
            : 1.0f;
    const float sy =
        (kf->image_height_ > 0)
            ? (static_cast<float>(image_height) / static_cast<float>(kf->image_height_))
            : 1.0f;
    const float W = static_cast<float>(image_width);
    const float H = static_cast<float>(image_height);

    std::vector<float> uv_host;
    std::vector<float> depth_host;
    uv_host.reserve(2 * N);
    depth_host.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        const float u = kf->kps_pixel_[2 * i + 0] * sx;
        const float v = kf->kps_pixel_[2 * i + 1] * sy;
        const float z = kf->kps_point_local_[3 * i + 2];
        if (!std::isfinite(u) || !std::isfinite(v) || !std::isfinite(z) || z <= 0.0f) {
            continue;
        }
        if (u < 0.0f || u > (W - 1.0f) || v < 0.0f || v > (H - 1.0f)) {
            continue;
        }

        uv_host.push_back(2.0f * (u / W) - 1.0f);
        uv_host.push_back(2.0f * (v / H) - 1.0f);
        depth_host.push_back(z);
    }

    const int64_t M = static_cast<int64_t>(depth_host.size());
    if (M < 2) {
        return false;
    }

    sparse_uv = torch::from_blob(
        uv_host.data(),
        {M, 2},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
    sparse_depth = torch::from_blob(
        depth_host.data(),
        {M},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
    return true;
}

static bool getKeyframeDepthMetersForEval(
    const std::shared_ptr<VoxelKeyframe>& pkf,
    int expected_h,
    int expected_w,
    cv::Mat& depth_meters)
{
    if (!pkf) {
        return false;
    }

    if (!pkf->img_auxiliary_undist_.empty()) {
        if (!depthMatToMeters(pkf->img_auxiliary_undist_, depth_meters)) {
            return false;
        }
    } else {
        if (!loadReplicaDepthFromRgbPath(pkf->img_filename_, depth_meters) &&
            !loadTumDepthFromRgbPath(pkf->img_filename_, depth_meters)) {
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


int sampleGeoSvrPatchSize()
{
    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<int> dist(17, 31);
    return dist(rng);
}

static bool depthMatToMeters(const cv::Mat& depth_in, cv::Mat& depth_meters)
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
        // Handle both common 16-bit conventions:
        // - mm depth (TUM-style): depth_m = raw / 1000
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

static bool loadReplicaDepthFromRgbPath(const std::string& rgb_filename, cv::Mat& depth_meters)
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
        return depthMatToMeters(depth_raw, depth_meters);
    }

    return false;
}

static bool loadTumDepthFromRgbPath(const std::string& rgb_filename, cv::Mat& depth_meters)
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
            return depthMatToMeters(depth_raw, depth_meters);
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
    return depthMatToMeters(depth_raw, depth_meters);
}

static bool computeSharedDepthVizRange(
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

static cv::Mat depthTensorToCvMatFloat(const torch::Tensor& depth_tensor)
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

static cv::Mat colorizeDepthMatJet(
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

static cv::Mat colorizeFiniteScalarMat(
    const cv::Mat& values,
    float viz_min,
    float viz_max,
    int colormap);

static cv::Mat colorizeFiniteScalarMatJet(
    const cv::Mat& values,
    float viz_min,
    float viz_max)
{
    return colorizeFiniteScalarMat(values, viz_min, viz_max, cv::COLORMAP_JET);
}

static cv::Mat colorizeFiniteScalarMat(
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

static cv::Mat appendColormapLegendBar(
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

static cv::Mat appendJetLegendBar(
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

static bool sampleDenseDepthAtSparseUv(
    const torch::Tensor& dense_depth,
    const torch::Tensor& sparse_uv,
    torch::Tensor& sampled_depth)
{
    sampled_depth = torch::Tensor();
    if (!dense_depth.defined() || !sparse_uv.defined()) {
        return false;
    }

    torch::Tensor depth = dense_depth.detach().to(torch::kFloat32).contiguous();
    if (depth.dim() != 2) {
        return false;
    }
    if (sparse_uv.dim() != 2 || sparse_uv.size(1) != 2 || sparse_uv.size(0) < 1) {
        return false;
    }

    torch::Tensor depth_img = depth.unsqueeze(0).unsqueeze(0);        // [1,1,H,W]
    torch::Tensor grid = sparse_uv.detach()
        .to(depth.device())
        .to(torch::kFloat32)
        .contiguous()
        .unsqueeze(0)
        .unsqueeze(0);                                                 // [1,1,N,2]

    auto gs_opts = torch::nn::functional::GridSampleFuncOptions()
                       .mode(torch::kBilinear)
                       .padding_mode(torch::kZeros)
                       .align_corners(false);

    sampled_depth = torch::nn::functional::grid_sample(depth_img, grid, gs_opts).squeeze();
    if (!sampled_depth.defined()) {
        return false;
    }
    if (sampled_depth.dim() == 0) {
        sampled_depth = sampled_depth.unsqueeze(0);
    }
    return sampled_depth.dim() == 1;
}

static bool fitAffineScaleShift1D(
    const torch::Tensor& x,
    const torch::Tensor& y,
    float& scale,
    float& shift)
{
    scale = std::numeric_limits<float>::quiet_NaN();
    shift = std::numeric_limits<float>::quiet_NaN();
    if (!x.defined() || !y.defined() || x.numel() < 2 || y.numel() != x.numel()) {
        return false;
    }

    const torch::Tensor x_mean = x.mean();
    const torch::Tensor y_mean = y.mean();
    const torch::Tensor dx = x - x_mean;
    const torch::Tensor dy = y - y_mean;
    const torch::Tensor denom = (dx * dx).sum();
    const float denom_f = denom.item<float>();
    if (!std::isfinite(denom_f) || denom_f <= 1e-12f) {
        return false;
    }

    const torch::Tensor scale_t = (dx * dy).sum() / denom;
    const torch::Tensor shift_t = y_mean - scale_t * x_mean;
    scale = scale_t.item<float>();
    shift = shift_t.item<float>();
    return std::isfinite(scale) && std::isfinite(shift);
}

static bool fitScale1D(
    const torch::Tensor& x,
    const torch::Tensor& y,
    float& scale)
{
    scale = std::numeric_limits<float>::quiet_NaN();
    if (!x.defined() || !y.defined() || x.numel() < 2 || y.numel() != x.numel()) {
        return false;
    }

    const torch::Tensor denom = (x * x).sum();
    const float denom_f = denom.item<float>();
    if (!std::isfinite(denom_f) || denom_f <= 1e-12f) {
        return false;
    }

    const torch::Tensor scale_t = (x * y).sum() / denom;
    scale = scale_t.item<float>();
    return std::isfinite(scale) && scale > 0.0f;
}

static float quantileVector(std::vector<float> values, float q)
{
    values.erase(
        std::remove_if(
            values.begin(),
            values.end(),
            [](float v) { return !std::isfinite(v); }),
        values.end());
    if (values.empty()) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    q = std::clamp(q, 0.0f, 1.0f);
    std::sort(values.begin(), values.end());
    const float pos = q * static_cast<float>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) {
        return values[lo];
    }
    const float t = pos - static_cast<float>(lo);
    return values[lo] * (1.0f - t) + values[hi] * t;
}

static bool fitAffineScaleShiftWeighted1D(
    const std::vector<float>& x,
    const std::vector<float>& y,
    const std::vector<float>* weights,
    float& scale,
    float& shift)
{
    scale = std::numeric_limits<float>::quiet_NaN();
    shift = std::numeric_limits<float>::quiet_NaN();
    if (x.size() != y.size() || x.size() < 2) {
        return false;
    }

    double w_sum = 0.0;
    double x_sum = 0.0;
    double y_sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) {
            continue;
        }
        const double w = weights ? static_cast<double>((*weights)[i]) : 1.0;
        if (!(w > 0.0) || !std::isfinite(w)) {
            continue;
        }
        w_sum += w;
        x_sum += w * static_cast<double>(x[i]);
        y_sum += w * static_cast<double>(y[i]);
    }
    if (!(w_sum > 0.0)) {
        return false;
    }

    const double x_mean = x_sum / w_sum;
    const double y_mean = y_sum / w_sum;
    double denom = 0.0;
    double numer = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) {
            continue;
        }
        const double w = weights ? static_cast<double>((*weights)[i]) : 1.0;
        if (!(w > 0.0) || !std::isfinite(w)) {
            continue;
        }
        const double dx = static_cast<double>(x[i]) - x_mean;
        const double dy = static_cast<double>(y[i]) - y_mean;
        denom += w * dx * dx;
        numer += w * dx * dy;
    }
    if (!(denom > 1e-12) || !std::isfinite(denom)) {
        return false;
    }

    const double s = numer / denom;
    const double t = y_mean - s * x_mean;
    scale = static_cast<float>(s);
    shift = static_cast<float>(t);
    return std::isfinite(scale) && std::isfinite(shift);
}

static bool applyDepthAnythingAffineAlignment(
    const torch::Tensor& mono_prior,
    float scale,
    float shift,
    torch::Tensor& aligned_depth)
{
    aligned_depth = torch::Tensor();
    if (!mono_prior.defined() ||
        !std::isfinite(scale) ||
        !std::isfinite(shift) ||
        scale <= 0.0f) {
        return false;
    }

    torch::Tensor mono = mono_prior.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (mono.dim() == 3 && mono.size(0) == 1) {
        mono = mono.squeeze(0);
    }
    if (mono.dim() != 2) {
        return false;
    }

    torch::Tensor aligned_inv = mono * scale + shift;
    const torch::Tensor aligned_valid = torch::isfinite(aligned_inv) & (aligned_inv > 1e-6f);
    aligned_depth = torch::where(
        aligned_valid,
        1.0f / aligned_inv.clamp_min(1e-6f),
        torch::full_like(aligned_inv, std::numeric_limits<float>::quiet_NaN()));
    aligned_depth = aligned_depth.to(torch::kCPU).contiguous();
    return aligned_depth.defined() && aligned_depth.dim() == 2;
}

static bool applyMetricDepthAlignment(
    const torch::Tensor& mono_prior,
    float scale,
    float shift,
    torch::Tensor& aligned_depth)
{
    aligned_depth = torch::Tensor();
    if (!mono_prior.defined() ||
        !std::isfinite(scale) ||
        !std::isfinite(shift) ||
        scale <= 0.0f) {
        return false;
    }

    torch::Tensor mono = mono_prior.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (mono.dim() == 3 && mono.size(0) == 1) {
        mono = mono.squeeze(0);
    }
    if (mono.dim() != 2) {
        return false;
    }

    aligned_depth = mono * scale + shift;
    const torch::Tensor aligned_valid = torch::isfinite(aligned_depth) & (aligned_depth > 1e-6f);
    aligned_depth = torch::where(
        aligned_valid,
        aligned_depth,
        torch::full_like(aligned_depth, std::numeric_limits<float>::quiet_NaN()));
    aligned_depth = aligned_depth.to(torch::kCPU).contiguous();
    return aligned_depth.defined() && aligned_depth.dim() == 2;
}

static bool alignDepthAnythingPriorToSparseAnchors(
    const torch::Tensor& mono_prior,
    const torch::Tensor& sparse_uv,
    const torch::Tensor& sparse_depth,
    float cam_near,
    int min_sparse_anchors,
    torch::Tensor& aligned_depth,
    MonoPriorAlignmentStats& stats)
{
    aligned_depth = torch::Tensor();
    stats = MonoPriorAlignmentStats();

    if (!mono_prior.defined() || !sparse_uv.defined() || !sparse_depth.defined()) {
        return false;
    }

    torch::Tensor mono = mono_prior.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (mono.dim() == 3 && mono.size(0) == 1) {
        mono = mono.squeeze(0);
    }
    if (mono.dim() != 2) {
        return false;
    }

    torch::Tensor mono_samples;
    if (!sampleDenseDepthAtSparseUv(mono, sparse_uv, mono_samples)) {
        return false;
    }

    torch::Tensor sparse_depth_1d = sparse_depth.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (sparse_depth_1d.dim() == 2 && sparse_depth_1d.size(1) == 1) {
        sparse_depth_1d = sparse_depth_1d.squeeze(1);
    }
    if (sparse_depth_1d.dim() != 1 || sparse_depth_1d.size(0) != mono_samples.size(0)) {
        return false;
    }

    const float near_depth = std::max(1e-6f, cam_near);
    torch::Tensor valid =
        torch::isfinite(mono_samples) &
        torch::isfinite(sparse_depth_1d) &
        (mono_samples > 0.0f) &
        (sparse_depth_1d > near_depth);
    stats.num_valid_anchors = valid.sum().item<int64_t>();
    if (stats.num_valid_anchors < std::max(2, min_sparse_anchors)) {
        return false;
    }

    torch::Tensor Y = mono_samples.masked_select(valid).contiguous();
    torch::Tensor sparse_depth_valid = sparse_depth_1d.masked_select(valid);
    torch::Tensor Xref = 1.0f / sparse_depth_valid.clamp_min(near_depth);

    // Use the old robust median/MAD alignment only to identify obvious outlier anchors.
    const torch::Tensor Ymed = Y.median();
    const torch::Tensor Ys = (Y - Ymed).abs().mean().clamp_min(1e-6f);
    const torch::Tensor Xmed = Xref.median();
    const torch::Tensor Xs = (Xref - Xmed).abs().mean().clamp_min(1e-6f);
    const torch::Tensor init_inv = (Y - Ymed) * (Xs / Ys) + Xmed;
    const torch::Tensor init_depth = 1.0f / init_inv.clamp_min(1e-6f);
    const torch::Tensor init_rel_err =
        (init_depth - sparse_depth_valid).abs() / sparse_depth_valid.clamp_min(near_depth);
    torch::Tensor fit_mask =
        torch::isfinite(init_rel_err) &
        torch::isfinite(init_inv) &
        (init_inv > 1e-6f) &
        (init_rel_err <= kMonoPriorAlignInitialInlierRelErr);
    stats.num_fit_anchors = fit_mask.sum().item<int64_t>();
    if (stats.num_fit_anchors < std::max(2, min_sparse_anchors)) {
        fit_mask = torch::ones_like(fit_mask, torch::TensorOptions().dtype(torch::kBool));
        stats.num_fit_anchors = stats.num_valid_anchors;
    }

    const torch::Tensor Y_fit = Y.masked_select(fit_mask).contiguous();
    const torch::Tensor Xref_fit = Xref.masked_select(fit_mask).contiguous();
    if (!fitAffineScaleShift1D(Y_fit, Xref_fit, stats.scale, stats.shift)) {
        return false;
    }
    if (stats.scale <= 0.0f) {
        return false;
    }

    const torch::Tensor aligned_anchor_inv = Y * stats.scale + stats.shift;
    const torch::Tensor positive_anchor_inv =
        torch::isfinite(aligned_anchor_inv) & (aligned_anchor_inv > 1e-6f);
    if (positive_anchor_inv.sum().item<int64_t>() < std::max(2, min_sparse_anchors)) {
        return false;
    }
    const torch::Tensor aligned_anchor_depth =
        1.0f / aligned_anchor_inv.masked_select(positive_anchor_inv).clamp_min(1e-6f);
    const torch::Tensor sparse_depth_eval =
        sparse_depth_valid.masked_select(positive_anchor_inv).clamp_min(near_depth);
    torch::Tensor rel_err =
        (aligned_anchor_depth - sparse_depth_eval).abs() / sparse_depth_eval;
    rel_err = rel_err.masked_select(torch::isfinite(rel_err)).contiguous();
    if (rel_err.numel() < std::max(2, min_sparse_anchors)) {
        return false;
    }
    stats.median_rel_depth_error = rel_err.median().item<float>();
    if (rel_err.numel() >= 10) {
        stats.p90_rel_depth_error = torch::quantile(
            rel_err,
            torch::tensor(0.90f, torch::TensorOptions().dtype(torch::kFloat32)))
            .item<float>();
    } else {
        stats.p90_rel_depth_error = rel_err.max().item<float>();
    }
    stats.final_inlier_ratio =
        static_cast<float>((rel_err <= kMonoPriorAlignFinalInlierRelErr)
                               .sum()
                               .item<int64_t>()) /
        static_cast<float>(stats.num_valid_anchors);
    if (!std::isfinite(stats.median_rel_depth_error) ||
        !std::isfinite(stats.p90_rel_depth_error) ||
        !std::isfinite(stats.final_inlier_ratio)) {
        return false;
    }

    if (!applyDepthAnythingAffineAlignment(mono, stats.scale, stats.shift, aligned_depth)) {
        return false;
    }

    if (sparse_depth_valid.numel() >= 10) {
        torch::Tensor q = torch::quantile(
            sparse_depth_valid,
            torch::tensor({0.05f, 0.95f}, torch::TensorOptions().dtype(torch::kFloat32)));
        stats.sparse_depth_q05 = q[0].item<float>();
        stats.sparse_depth_q95 = q[1].item<float>();
    } else {
        stats.sparse_depth_q05 = sparse_depth_valid.min().item<float>();
        stats.sparse_depth_q95 = sparse_depth_valid.max().item<float>();
    }

    return aligned_depth.defined() &&
           aligned_depth.dim() == 2 &&
           std::isfinite(stats.sparse_depth_q05) &&
           std::isfinite(stats.sparse_depth_q95) &&
           stats.sparse_depth_q95 > stats.sparse_depth_q05;
}

static bool alignDepthAnythingPriorToRenderedDepth(
    const torch::Tensor& mono_prior,
    const torch::Tensor& rendered_depth,
    const torch::Tensor& rendered_alpha,
    const torch::Tensor& rendered_n_contrib,
    float cam_near,
    int min_rendered_anchors,
    torch::Tensor& aligned_depth,
    MonoPriorAlignmentStats& stats)
{
    aligned_depth = torch::Tensor();
    stats = MonoPriorAlignmentStats();

    if (!mono_prior.defined() || !rendered_depth.defined() ||
        !rendered_alpha.defined() || !rendered_n_contrib.defined()) {
        return false;
    }

    torch::Tensor mono =
        mono_prior.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (mono.dim() == 3 && mono.size(0) == 1) {
        mono = mono.squeeze(0);
    }
    torch::Tensor depth =
        rendered_depth.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor alpha =
        rendered_alpha.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor n_contrib =
        rendered_n_contrib.detach().to(torch::kCPU).to(torch::kInt32).contiguous();
    if (mono.dim() != 2 || depth.dim() != 2 || alpha.dim() != 2 ||
        n_contrib.dim() != 2 ||
        mono.sizes() != depth.sizes() ||
        mono.sizes() != alpha.sizes() ||
        mono.sizes() != n_contrib.sizes()) {
        return false;
    }

    const float near_depth = std::max(1e-6f, cam_near);
    torch::Tensor valid =
        torch::isfinite(mono) &
        torch::isfinite(depth) &
        torch::isfinite(alpha) &
        (mono > 0.0f) &
        (depth > near_depth) &
        (alpha > 0.5f) &
        (n_contrib > 0);
    stats.num_valid_anchors = valid.sum().item<int64_t>();
    if (stats.num_valid_anchors < std::max(2, min_rendered_anchors)) {
        return false;
    }

    torch::Tensor Y = mono.masked_select(valid).contiguous();
    torch::Tensor rendered_depth_valid = depth.masked_select(valid).contiguous();
    torch::Tensor Xref = 1.0f / rendered_depth_valid.clamp_min(near_depth);

    const torch::Tensor Ymed = Y.median();
    const torch::Tensor Ys = (Y - Ymed).abs().mean().clamp_min(1e-6f);
    const torch::Tensor Xmed = Xref.median();
    const torch::Tensor Xs = (Xref - Xmed).abs().mean().clamp_min(1e-6f);
    const torch::Tensor init_inv = (Y - Ymed) * (Xs / Ys) + Xmed;
    const torch::Tensor init_depth = 1.0f / init_inv.clamp_min(1e-6f);
    const torch::Tensor init_rel_err =
        (init_depth - rendered_depth_valid).abs() /
        rendered_depth_valid.clamp_min(near_depth);
    torch::Tensor fit_mask =
        torch::isfinite(init_rel_err) &
        torch::isfinite(init_inv) &
        (init_inv > 1e-6f) &
        (init_rel_err <= kMonoPriorAlignInitialInlierRelErr);
    stats.num_fit_anchors = fit_mask.sum().item<int64_t>();
    if (stats.num_fit_anchors < std::max(2, min_rendered_anchors)) {
        fit_mask = torch::ones_like(fit_mask, torch::TensorOptions().dtype(torch::kBool));
        stats.num_fit_anchors = stats.num_valid_anchors;
    }

    const torch::Tensor Y_fit = Y.masked_select(fit_mask).contiguous();
    const torch::Tensor Xref_fit = Xref.masked_select(fit_mask).contiguous();
    if (!fitAffineScaleShift1D(Y_fit, Xref_fit, stats.scale, stats.shift)) {
        return false;
    }
    if (stats.scale <= 0.0f) {
        return false;
    }

    const torch::Tensor aligned_anchor_inv = Y * stats.scale + stats.shift;
    const torch::Tensor positive_anchor_inv =
        torch::isfinite(aligned_anchor_inv) & (aligned_anchor_inv > 1e-6f);
    if (positive_anchor_inv.sum().item<int64_t>() < std::max(2, min_rendered_anchors)) {
        return false;
    }
    const torch::Tensor aligned_anchor_depth =
        1.0f / aligned_anchor_inv.masked_select(positive_anchor_inv).clamp_min(1e-6f);
    const torch::Tensor rendered_depth_eval =
        rendered_depth_valid.masked_select(positive_anchor_inv).clamp_min(near_depth);
    torch::Tensor rel_err =
        (aligned_anchor_depth - rendered_depth_eval).abs() / rendered_depth_eval;
    rel_err = rel_err.masked_select(torch::isfinite(rel_err)).contiguous();
    if (rel_err.numel() < std::max(2, min_rendered_anchors)) {
        return false;
    }

    stats.median_rel_depth_error = rel_err.median().item<float>();
    if (rel_err.numel() >= 10) {
        stats.p90_rel_depth_error = torch::quantile(
            rel_err,
            torch::tensor(0.90f, torch::TensorOptions().dtype(torch::kFloat32)))
            .item<float>();
    } else {
        stats.p90_rel_depth_error = rel_err.max().item<float>();
    }
    stats.final_inlier_ratio =
        static_cast<float>((rel_err <= kMonoPriorAlignFinalInlierRelErr)
                               .sum()
                               .item<int64_t>()) /
        static_cast<float>(stats.num_valid_anchors);
    if (!std::isfinite(stats.median_rel_depth_error) ||
        !std::isfinite(stats.p90_rel_depth_error) ||
        !std::isfinite(stats.final_inlier_ratio)) {
        return false;
    }

    if (!applyDepthAnythingAffineAlignment(mono, stats.scale, stats.shift, aligned_depth)) {
        return false;
    }

    if (rendered_depth_valid.numel() >= 10) {
        torch::Tensor q = torch::quantile(
            rendered_depth_valid,
            torch::tensor({0.05f, 0.95f}, torch::TensorOptions().dtype(torch::kFloat32)));
        stats.sparse_depth_q05 = q[0].item<float>();
        stats.sparse_depth_q95 = q[1].item<float>();
    } else {
        stats.sparse_depth_q05 = rendered_depth_valid.min().item<float>();
        stats.sparse_depth_q95 = rendered_depth_valid.max().item<float>();
    }

    return aligned_depth.defined() &&
           aligned_depth.dim() == 2 &&
           std::isfinite(stats.sparse_depth_q05) &&
           std::isfinite(stats.sparse_depth_q95) &&
           stats.sparse_depth_q95 > stats.sparse_depth_q05;
}

static bool alignMetricDepthPriorToSparseAnchors(
    const torch::Tensor& mono_prior,
    const torch::Tensor& sparse_uv,
    const torch::Tensor& sparse_depth,
    float cam_near,
    int min_sparse_anchors,
    torch::Tensor& aligned_depth,
    MonoPriorAlignmentStats& stats)
{
    aligned_depth = torch::Tensor();
    stats = MonoPriorAlignmentStats();

    if (!mono_prior.defined() || !sparse_uv.defined() || !sparse_depth.defined()) {
        return false;
    }

    torch::Tensor mono = mono_prior.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (mono.dim() == 3 && mono.size(0) == 1) {
        mono = mono.squeeze(0);
    }
    if (mono.dim() != 2) {
        return false;
    }

    torch::Tensor mono_samples;
    if (!sampleDenseDepthAtSparseUv(mono, sparse_uv, mono_samples)) {
        return false;
    }

    torch::Tensor sparse_depth_1d = sparse_depth.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (sparse_depth_1d.dim() == 2 && sparse_depth_1d.size(1) == 1) {
        sparse_depth_1d = sparse_depth_1d.squeeze(1);
    }
    if (sparse_depth_1d.dim() != 1 || sparse_depth_1d.size(0) != mono_samples.size(0)) {
        return false;
    }

    const float near_depth = std::max(1e-6f, cam_near);
    torch::Tensor valid =
        torch::isfinite(mono_samples) &
        torch::isfinite(sparse_depth_1d) &
        (mono_samples > near_depth) &
        (sparse_depth_1d > near_depth);
    stats.num_valid_anchors = valid.sum().item<int64_t>();
    if (stats.num_valid_anchors < std::max(2, min_sparse_anchors)) {
        return false;
    }

    torch::Tensor Y = mono_samples.masked_select(valid).contiguous();
    torch::Tensor sparse_depth_valid = sparse_depth_1d.masked_select(valid).contiguous();

    torch::Tensor init_scale_candidates =
        sparse_depth_valid / Y.clamp_min(near_depth);
    init_scale_candidates = init_scale_candidates.masked_select(
        torch::isfinite(init_scale_candidates) & (init_scale_candidates > 0.0f));
    if (!init_scale_candidates.defined() ||
        init_scale_candidates.numel() < std::max<int64_t>(2, min_sparse_anchors)) {
        return false;
    }
    const float init_scale = init_scale_candidates.median().item<float>();
    if (!std::isfinite(init_scale) || init_scale <= 0.0f) {
        return false;
    }

    const torch::Tensor init_depth = Y * init_scale;
    const torch::Tensor init_rel_err =
        (init_depth - sparse_depth_valid).abs() /
        sparse_depth_valid.clamp_min(near_depth);
    torch::Tensor fit_mask =
        torch::isfinite(init_rel_err) &
        (init_rel_err <= kMonoPriorAlignInitialInlierRelErr);
    stats.num_fit_anchors = fit_mask.sum().item<int64_t>();
    if (stats.num_fit_anchors < std::max(2, min_sparse_anchors)) {
        fit_mask = torch::ones_like(fit_mask, torch::TensorOptions().dtype(torch::kBool));
        stats.num_fit_anchors = stats.num_valid_anchors;
    }

    const torch::Tensor Y_fit = Y.masked_select(fit_mask).contiguous();
    const torch::Tensor depth_fit = sparse_depth_valid.masked_select(fit_mask).contiguous();
    stats.shift = 0.0f;
    if (!fitScale1D(Y_fit, depth_fit, stats.scale)) {
        return false;
    }

    const torch::Tensor aligned_anchor_depth = Y * stats.scale;
    torch::Tensor rel_err =
        (aligned_anchor_depth - sparse_depth_valid).abs() /
        sparse_depth_valid.clamp_min(near_depth);
    rel_err = rel_err.masked_select(torch::isfinite(rel_err)).contiguous();
    if (rel_err.numel() < std::max(2, min_sparse_anchors)) {
        return false;
    }

    stats.median_rel_depth_error = rel_err.median().item<float>();
    if (rel_err.numel() >= 10) {
        stats.p90_rel_depth_error = torch::quantile(
            rel_err,
            torch::tensor(0.90f, torch::TensorOptions().dtype(torch::kFloat32)))
            .item<float>();
    } else {
        stats.p90_rel_depth_error = rel_err.max().item<float>();
    }
    stats.final_inlier_ratio =
        static_cast<float>((rel_err <= kMonoPriorAlignFinalInlierRelErr)
                               .sum()
                               .item<int64_t>()) /
        static_cast<float>(stats.num_valid_anchors);
    if (!std::isfinite(stats.median_rel_depth_error) ||
        !std::isfinite(stats.p90_rel_depth_error) ||
        !std::isfinite(stats.final_inlier_ratio)) {
        return false;
    }

    if (!applyMetricDepthAlignment(mono, stats.scale, stats.shift, aligned_depth)) {
        return false;
    }

    if (sparse_depth_valid.numel() >= 10) {
        torch::Tensor q = torch::quantile(
            sparse_depth_valid,
            torch::tensor({0.05f, 0.95f}, torch::TensorOptions().dtype(torch::kFloat32)));
        stats.sparse_depth_q05 = q[0].item<float>();
        stats.sparse_depth_q95 = q[1].item<float>();
    } else {
        stats.sparse_depth_q05 = sparse_depth_valid.min().item<float>();
        stats.sparse_depth_q95 = sparse_depth_valid.max().item<float>();
    }

    return aligned_depth.defined() &&
           aligned_depth.dim() == 2 &&
           std::isfinite(stats.sparse_depth_q05) &&
           std::isfinite(stats.sparse_depth_q95) &&
           stats.sparse_depth_q95 > stats.sparse_depth_q05;
}

static bool alignMonoPriorToSparseAnchorsDaPrior(
    const torch::Tensor& mono_prior,
    const torch::Tensor& sparse_uv,
    const torch::Tensor& sparse_depth,
    const torch::Tensor& query_mask,
    bool mono_prior_is_metric_depth,
    float cam_near,
    int min_sparse_anchors,
    int knn_k,
    bool distance_weighting,
    float max_pixel_dist,
    torch::Tensor& aligned_depth,
    MonoPriorAlignmentStats& stats)
{
    aligned_depth = torch::Tensor();
    stats = MonoPriorAlignmentStats();

    if (!mono_prior.defined() || !sparse_uv.defined() || !sparse_depth.defined()) {
        return false;
    }

    torch::Tensor mono = mono_prior.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (mono.dim() == 3 && mono.size(0) == 1) {
        mono = mono.squeeze(0);
    }
    if (mono.dim() != 2) {
        return false;
    }
    const int H = static_cast<int>(mono.size(0));
    const int W = static_cast<int>(mono.size(1));
    if (H <= 0 || W <= 0) {
        return false;
    }

    torch::Tensor mono_samples;
    if (!sampleDenseDepthAtSparseUv(mono, sparse_uv, mono_samples)) {
        return false;
    }

    torch::Tensor sparse_depth_1d = sparse_depth.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (sparse_depth_1d.dim() == 2 && sparse_depth_1d.size(1) == 1) {
        sparse_depth_1d = sparse_depth_1d.squeeze(1);
    }
    torch::Tensor sparse_uv_cpu = sparse_uv.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (sparse_depth_1d.dim() != 1 ||
        sparse_uv_cpu.dim() != 2 ||
        sparse_uv_cpu.size(1) != 2 ||
        sparse_depth_1d.size(0) != mono_samples.size(0) ||
        sparse_uv_cpu.size(0) != mono_samples.size(0)) {
        return false;
    }

    struct DaPriorAnchor {
        float x = 0.0f;
        float y = 0.0f;
        float pred_disp = 0.0f;
        float target_disp = 0.0f;
        float target_depth = 0.0f;
    };

    const float near_depth = std::max(1e-6f, cam_near);
    auto mono_sample_acc = mono_samples.accessor<float, 1>();
    auto sparse_depth_acc = sparse_depth_1d.accessor<float, 1>();
    auto sparse_uv_acc = sparse_uv_cpu.accessor<float, 2>();

    std::vector<DaPriorAnchor> valid_anchors;
    valid_anchors.reserve(static_cast<size_t>(mono_samples.size(0)));
    std::vector<float> pred_disp_values;
    std::vector<float> target_disp_values;
    std::vector<float> sparse_depth_values;

    for (int64_t i = 0; i < mono_samples.size(0); ++i) {
        const float mono_value = mono_sample_acc[i];
        const float z = sparse_depth_acc[i];
        if (!std::isfinite(mono_value) || !std::isfinite(z) || z <= near_depth) {
            continue;
        }

        float pred_disp = std::numeric_limits<float>::quiet_NaN();
        if (mono_prior_is_metric_depth) {
            if (mono_value <= near_depth) {
                continue;
            }
            pred_disp = 1.0f / mono_value;
        } else {
            if (mono_value <= 0.0f) {
                continue;
            }
            pred_disp = mono_value;
        }
        const float target_disp = 1.0f / std::max(z, near_depth);
        if (!std::isfinite(pred_disp) || !std::isfinite(target_disp) ||
            pred_disp <= 0.0f || target_disp <= 0.0f) {
            continue;
        }

        const float x = 0.5f * (sparse_uv_acc[i][0] + 1.0f) * static_cast<float>(W);
        const float y = 0.5f * (sparse_uv_acc[i][1] + 1.0f) * static_cast<float>(H);
        if (!std::isfinite(x) || !std::isfinite(y) ||
            x < 0.0f || x > static_cast<float>(W - 1) ||
            y < 0.0f || y > static_cast<float>(H - 1)) {
            continue;
        }

        valid_anchors.push_back({x, y, pred_disp, target_disp, z});
        pred_disp_values.push_back(pred_disp);
        target_disp_values.push_back(target_disp);
        sparse_depth_values.push_back(z);
    }

    stats.num_valid_anchors = static_cast<int64_t>(valid_anchors.size());
    if (stats.num_valid_anchors < std::max(2, min_sparse_anchors)) {
        return false;
    }

    const float pred_med = quantileVector(pred_disp_values, 0.50f);
    const float target_med = quantileVector(target_disp_values, 0.50f);
    if (!std::isfinite(pred_med) || !std::isfinite(target_med)) {
        return false;
    }

    double pred_mad = 0.0;
    double target_mad = 0.0;
    for (size_t i = 0; i < pred_disp_values.size(); ++i) {
        pred_mad += std::abs(static_cast<double>(pred_disp_values[i]) - pred_med);
        target_mad += std::abs(static_cast<double>(target_disp_values[i]) - target_med);
    }
    pred_mad = std::max(pred_mad / static_cast<double>(pred_disp_values.size()), 1e-6);
    target_mad = std::max(target_mad / static_cast<double>(target_disp_values.size()), 1e-6);

    std::vector<size_t> fit_indices;
    fit_indices.reserve(valid_anchors.size());
    for (size_t i = 0; i < valid_anchors.size(); ++i) {
        const float init_disp =
            (valid_anchors[i].pred_disp - pred_med) *
                static_cast<float>(target_mad / pred_mad) +
            target_med;
        if (!std::isfinite(init_disp) || init_disp <= 1e-6f) {
            continue;
        }
        const float init_depth = 1.0f / init_disp;
        const float rel_err =
            std::abs(init_depth - valid_anchors[i].target_depth) /
            std::max(valid_anchors[i].target_depth, near_depth);
        if (std::isfinite(rel_err) && rel_err <= kMonoPriorAlignInitialInlierRelErr) {
            fit_indices.push_back(i);
        }
    }

    stats.num_fit_anchors = static_cast<int64_t>(fit_indices.size());
    if (stats.num_fit_anchors < std::max(2, min_sparse_anchors)) {
        fit_indices.clear();
        fit_indices.reserve(valid_anchors.size());
        for (size_t i = 0; i < valid_anchors.size(); ++i) {
            fit_indices.push_back(i);
        }
        stats.num_fit_anchors = stats.num_valid_anchors;
    }

    std::vector<float> fit_pred;
    std::vector<float> fit_target;
    fit_pred.reserve(fit_indices.size());
    fit_target.reserve(fit_indices.size());
    for (const size_t idx : fit_indices) {
        fit_pred.push_back(valid_anchors[idx].pred_disp);
        fit_target.push_back(valid_anchors[idx].target_disp);
    }
    if (!fitAffineScaleShiftWeighted1D(fit_pred, fit_target, nullptr, stats.scale, stats.shift)) {
        return false;
    }
    if (!std::isfinite(stats.scale) || !std::isfinite(stats.shift)) {
        return false;
    }

    std::vector<float> rel_errors;
    rel_errors.reserve(valid_anchors.size());
    int64_t final_inliers = 0;
    for (const auto& anchor : valid_anchors) {
        const float aligned_disp = anchor.pred_disp * stats.scale + stats.shift;
        if (!std::isfinite(aligned_disp) || aligned_disp <= 1e-6f) {
            continue;
        }
        const float aligned_z = 1.0f / aligned_disp;
        const float rel_err =
            std::abs(aligned_z - anchor.target_depth) /
            std::max(anchor.target_depth, near_depth);
        if (std::isfinite(rel_err)) {
            rel_errors.push_back(rel_err);
            if (rel_err <= kMonoPriorAlignFinalInlierRelErr) {
                ++final_inliers;
            }
        }
    }
    if (rel_errors.size() < static_cast<size_t>(std::max(2, min_sparse_anchors))) {
        return false;
    }
    stats.median_rel_depth_error = quantileVector(rel_errors, 0.50f);
    stats.p90_rel_depth_error = quantileVector(rel_errors, 0.90f);
    stats.final_inlier_ratio =
        static_cast<float>(final_inliers) /
        static_cast<float>(std::max<int64_t>(1, stats.num_valid_anchors));
    stats.sparse_depth_q05 = quantileVector(sparse_depth_values, 0.05f);
    stats.sparse_depth_q95 = quantileVector(sparse_depth_values, 0.95f);
    if (!std::isfinite(stats.median_rel_depth_error) ||
        !std::isfinite(stats.p90_rel_depth_error) ||
        !std::isfinite(stats.final_inlier_ratio) ||
        !std::isfinite(stats.sparse_depth_q05) ||
        !std::isfinite(stats.sparse_depth_q95) ||
        !(stats.sparse_depth_q95 > stats.sparse_depth_q05)) {
        return false;
    }

    const int local_k = std::min<int>(
        std::max(2, knn_k),
        static_cast<int>(fit_indices.size()));
    if (local_k < 2) {
        return false;
    }

    cv::Mat anchor_mat(static_cast<int>(fit_indices.size()), 2, CV_32F);
    std::vector<DaPriorAnchor> fit_anchors;
    fit_anchors.reserve(fit_indices.size());
    for (size_t i = 0; i < fit_indices.size(); ++i) {
        const DaPriorAnchor& anchor = valid_anchors[fit_indices[i]];
        fit_anchors.push_back(anchor);
        anchor_mat.at<float>(static_cast<int>(i), 0) = anchor.x;
        anchor_mat.at<float>(static_cast<int>(i), 1) = anchor.y;
    }

    torch::Tensor qmask;
    if (query_mask.defined()) {
        qmask = query_mask.detach().to(torch::kCPU).to(torch::kBool).contiguous();
        if (qmask.dim() != 2 || qmask.size(0) != H || qmask.size(1) != W) {
            return false;
        }
    }

    torch::Tensor aligned =
        torch::full({H, W}, std::numeric_limits<float>::quiet_NaN(),
                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto aligned_acc = aligned.accessor<float, 2>();
    auto mono_acc = mono.accessor<float, 2>();

    auto pred_disp_at = [&](int y, int x, float& pred_disp) -> bool {
        const float mono_value = mono_acc[y][x];
        if (!std::isfinite(mono_value)) {
            return false;
        }
        if (mono_prior_is_metric_depth) {
            if (mono_value <= near_depth) {
                return false;
            }
            pred_disp = 1.0f / mono_value;
        } else {
            if (mono_value <= 0.0f) {
                return false;
            }
            pred_disp = mono_value;
        }
        return std::isfinite(pred_disp) && pred_disp > 0.0f;
    };

    std::vector<cv::Point> query_pixels;
    query_pixels.reserve(query_mask.defined()
                             ? static_cast<size_t>(std::max<int64_t>(1, qmask.sum().item<int64_t>()))
                             : static_cast<size_t>(H) * static_cast<size_t>(W));
    if (qmask.defined()) {
        auto qmask_acc = qmask.accessor<bool, 2>();
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float pred_disp = 0.0f;
                if (qmask_acc[y][x] && pred_disp_at(y, x, pred_disp)) {
                    query_pixels.emplace_back(x, y);
                }
            }
        }
    } else {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float pred_disp = 0.0f;
                if (pred_disp_at(y, x, pred_disp)) {
                    query_pixels.emplace_back(x, y);
                }
            }
        }
    }

    if (query_pixels.empty()) {
        return false;
    }

    cv::Mat query_mat(static_cast<int>(query_pixels.size()), 2, CV_32F);
    for (size_t i = 0; i < query_pixels.size(); ++i) {
        query_mat.at<float>(static_cast<int>(i), 0) = static_cast<float>(query_pixels[i].x);
        query_mat.at<float>(static_cast<int>(i), 1) = static_cast<float>(query_pixels[i].y);
    }

    cv::flann::Index knn_index(anchor_mat, cv::flann::KDTreeIndexParams(4));
    cv::Mat indices;
    cv::Mat dists_sq;
    knn_index.knnSearch(
        query_mat,
        indices,
        dists_sq,
        local_k,
        cv::flann::SearchParams(32));

    constexpr float kDirectAnchorDistPx = 1e-3f;
    for (size_t qi = 0; qi < query_pixels.size(); ++qi) {
        const int x = query_pixels[qi].x;
        const int y = query_pixels[qi].y;
        float query_pred_disp = 0.0f;
        if (!pred_disp_at(y, x, query_pred_disp)) {
            continue;
        }

        float direct_depth = std::numeric_limits<float>::quiet_NaN();
        std::vector<float> local_pred;
        std::vector<float> local_target;
        std::vector<float> local_weights;
        local_pred.reserve(static_cast<size_t>(local_k));
        local_target.reserve(static_cast<size_t>(local_k));
        local_weights.reserve(static_cast<size_t>(local_k));

        double inv_dist_sum = 0.0;
        for (int k = 0; k < local_k; ++k) {
            const int anchor_idx = indices.at<int>(static_cast<int>(qi), k);
            if (anchor_idx < 0 || anchor_idx >= static_cast<int>(fit_anchors.size())) {
                continue;
            }
            const float dist =
                std::sqrt(std::max(0.0f, dists_sq.at<float>(static_cast<int>(qi), k)));
            if (max_pixel_dist > 0.0f && dist > max_pixel_dist) {
                continue;
            }
            const DaPriorAnchor& anchor = fit_anchors[static_cast<size_t>(anchor_idx)];
            if (dist <= kDirectAnchorDistPx) {
                direct_depth = anchor.target_depth;
                break;
            }
            const float inv_dist = 1.0f / std::max(dist, 1e-3f);
            inv_dist_sum += inv_dist;
            local_pred.push_back(anchor.pred_disp);
            local_target.push_back(anchor.target_disp);
            local_weights.push_back(inv_dist);
        }

        if (std::isfinite(direct_depth) && direct_depth > near_depth) {
            aligned_acc[y][x] = direct_depth;
            continue;
        }

        float local_scale = stats.scale;
        float local_shift = stats.shift;
        if (local_pred.size() >= 2) {
            std::vector<float>* weights_ptr = nullptr;
            if (distance_weighting && inv_dist_sum > 0.0) {
                for (float& w : local_weights) {
                    const float normalized = w / static_cast<float>(inv_dist_sum);
                    w = normalized * normalized;
                }
                weights_ptr = &local_weights;
            }
            if (!fitAffineScaleShiftWeighted1D(
                    local_pred,
                    local_target,
                    weights_ptr,
                    local_scale,
                    local_shift)) {
                local_scale = stats.scale;
                local_shift = stats.shift;
            }
        }

        float aligned_disp = query_pred_disp * local_scale + local_shift;
        if (!std::isfinite(aligned_disp) || aligned_disp <= 1e-6f) {
            aligned_disp = query_pred_disp * stats.scale + stats.shift;
        }
        if (std::isfinite(aligned_disp) && aligned_disp > 1e-6f) {
            const float z = 1.0f / aligned_disp;
            if (std::isfinite(z) && z > near_depth) {
                aligned_acc[y][x] = z;
            }
        }
    }

    for (const auto& anchor : fit_anchors) {
        const int x = std::clamp(static_cast<int>(std::round(anchor.x)), 0, W - 1);
        const int y = std::clamp(static_cast<int>(std::round(anchor.y)), 0, H - 1);
        aligned_acc[y][x] = anchor.target_depth;
    }

    const int64_t aligned_count =
        (torch::isfinite(aligned) & (aligned > near_depth)).sum().item<int64_t>();
    if (aligned_count <= 0) {
        return false;
    }

    aligned_depth = aligned.contiguous();
    return true;
}

static bool alignMetricDepthPriorToRenderedDepth(
    const torch::Tensor& mono_prior,
    const torch::Tensor& rendered_depth,
    const torch::Tensor& rendered_alpha,
    const torch::Tensor& rendered_n_contrib,
    float cam_near,
    int min_rendered_anchors,
    torch::Tensor& aligned_depth,
    MonoPriorAlignmentStats& stats)
{
    aligned_depth = torch::Tensor();
    stats = MonoPriorAlignmentStats();

    if (!mono_prior.defined() || !rendered_depth.defined() ||
        !rendered_alpha.defined() || !rendered_n_contrib.defined()) {
        return false;
    }

    torch::Tensor mono =
        mono_prior.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (mono.dim() == 3 && mono.size(0) == 1) {
        mono = mono.squeeze(0);
    }
    torch::Tensor depth =
        rendered_depth.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor alpha =
        rendered_alpha.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor n_contrib =
        rendered_n_contrib.detach().to(torch::kCPU).to(torch::kInt32).contiguous();
    if (mono.dim() != 2 || depth.dim() != 2 || alpha.dim() != 2 ||
        n_contrib.dim() != 2 ||
        mono.sizes() != depth.sizes() ||
        mono.sizes() != alpha.sizes() ||
        mono.sizes() != n_contrib.sizes()) {
        return false;
    }

    const float near_depth = std::max(1e-6f, cam_near);
    torch::Tensor valid =
        torch::isfinite(mono) &
        torch::isfinite(depth) &
        torch::isfinite(alpha) &
        (mono > near_depth) &
        (depth > near_depth) &
        (alpha > 0.5f) &
        (n_contrib > 0);
    stats.num_valid_anchors = valid.sum().item<int64_t>();
    if (stats.num_valid_anchors < std::max(2, min_rendered_anchors)) {
        return false;
    }

    torch::Tensor Y = mono.masked_select(valid).contiguous();
    torch::Tensor rendered_depth_valid = depth.masked_select(valid).contiguous();

    torch::Tensor init_scale_candidates =
        rendered_depth_valid / Y.clamp_min(near_depth);
    init_scale_candidates = init_scale_candidates.masked_select(
        torch::isfinite(init_scale_candidates) & (init_scale_candidates > 0.0f));
    if (!init_scale_candidates.defined() ||
        init_scale_candidates.numel() < std::max<int64_t>(2, min_rendered_anchors)) {
        return false;
    }
    const float init_scale = init_scale_candidates.median().item<float>();
    if (!std::isfinite(init_scale) || init_scale <= 0.0f) {
        return false;
    }

    const torch::Tensor init_depth = Y * init_scale;
    const torch::Tensor init_rel_err =
        (init_depth - rendered_depth_valid).abs() /
        rendered_depth_valid.clamp_min(near_depth);
    torch::Tensor fit_mask =
        torch::isfinite(init_rel_err) &
        (init_rel_err <= kMonoPriorAlignInitialInlierRelErr);
    stats.num_fit_anchors = fit_mask.sum().item<int64_t>();
    if (stats.num_fit_anchors < std::max(2, min_rendered_anchors)) {
        fit_mask = torch::ones_like(fit_mask, torch::TensorOptions().dtype(torch::kBool));
        stats.num_fit_anchors = stats.num_valid_anchors;
    }

    const torch::Tensor Y_fit = Y.masked_select(fit_mask).contiguous();
    const torch::Tensor depth_fit = rendered_depth_valid.masked_select(fit_mask).contiguous();
    stats.shift = 0.0f;
    if (!fitScale1D(Y_fit, depth_fit, stats.scale)) {
        return false;
    }

    const torch::Tensor aligned_anchor_depth = Y * stats.scale;
    torch::Tensor rel_err =
        (aligned_anchor_depth - rendered_depth_valid).abs() /
        rendered_depth_valid.clamp_min(near_depth);
    rel_err = rel_err.masked_select(torch::isfinite(rel_err)).contiguous();
    if (rel_err.numel() < std::max(2, min_rendered_anchors)) {
        return false;
    }

    stats.median_rel_depth_error = rel_err.median().item<float>();
    if (rel_err.numel() >= 10) {
        stats.p90_rel_depth_error = torch::quantile(
            rel_err,
            torch::tensor(0.90f, torch::TensorOptions().dtype(torch::kFloat32)))
            .item<float>();
    } else {
        stats.p90_rel_depth_error = rel_err.max().item<float>();
    }
    stats.final_inlier_ratio =
        static_cast<float>((rel_err <= kMonoPriorAlignFinalInlierRelErr)
                               .sum()
                               .item<int64_t>()) /
        static_cast<float>(stats.num_valid_anchors);
    if (!std::isfinite(stats.median_rel_depth_error) ||
        !std::isfinite(stats.p90_rel_depth_error) ||
        !std::isfinite(stats.final_inlier_ratio)) {
        return false;
    }

    if (!applyMetricDepthAlignment(mono, stats.scale, stats.shift, aligned_depth)) {
        return false;
    }

    if (rendered_depth_valid.numel() >= 10) {
        torch::Tensor q = torch::quantile(
            rendered_depth_valid,
            torch::tensor({0.05f, 0.95f}, torch::TensorOptions().dtype(torch::kFloat32)));
        stats.sparse_depth_q05 = q[0].item<float>();
        stats.sparse_depth_q95 = q[1].item<float>();
    } else {
        stats.sparse_depth_q05 = rendered_depth_valid.min().item<float>();
        stats.sparse_depth_q95 = rendered_depth_valid.max().item<float>();
    }

    return aligned_depth.defined() &&
           aligned_depth.dim() == 2 &&
           std::isfinite(stats.sparse_depth_q05) &&
           std::isfinite(stats.sparse_depth_q95) &&
           stats.sparse_depth_q95 > stats.sparse_depth_q05;
}

} // namespace

void VoxelMapper::readMonoPriorConfigFromSettings(const cv::FileStorage& settings_file)
{
    {
        cv::FileNode n = settings_file["Mapper.depthanything_densify_stride"];
        mono_prior_params_.depthanything_densify_stride_ =
            n.empty() ? 8 : std::max(1, static_cast<int>(n));
    }
    {
        cv::FileNode n = settings_file["Mapper.depthanything_densify_min_sparse_anchors"];
        mono_prior_params_.depthanything_densify_min_sparse_anchors_ =
            n.empty() ? 64 : std::max(2, static_cast<int>(n));
    }
    {
        cv::FileNode n = settings_file["Mapper.depthanything_densify_alignment_mode"];
        mono_prior_params_.depthanything_densify_alignment_mode_ =
            normalizeMonoPriorDensifyAlignmentMode(
                n.empty() ? std::string("orb") : static_cast<std::string>(n));
        if (mono_prior_params_.depthanything_densify_alignment_mode_ != "orb" &&
            mono_prior_params_.depthanything_densify_alignment_mode_ != "rendered" &&
            mono_prior_params_.depthanything_densify_alignment_mode_ != "da_prior") {
            std::cerr << "[MonoPrior] Unknown Mapper.depthanything_densify_alignment_mode='"
                      << mono_prior_params_.depthanything_densify_alignment_mode_
                      << "', falling back to 'orb'. Supported: orb, rendered, da_prior."
                      << std::endl;
            mono_prior_params_.depthanything_densify_alignment_mode_ = "orb";
        }
    }
    {
        cv::FileNode n = settings_file["Mapper.depthanything_da_prior_knn_k"];
        mono_prior_params_.depthanything_da_prior_knn_k_ =
            n.empty() ? 5 : std::max(2, static_cast<int>(n));
    }
    {
        cv::FileNode n = settings_file["Mapper.depthanything_da_prior_distance_weighting"];
        mono_prior_params_.depthanything_da_prior_distance_weighting_ =
            n.empty() ? true : ((n.operator int()) != 0);
    }
    {
        cv::FileNode n = settings_file["Mapper.depthanything_da_prior_max_pixel_dist"];
        mono_prior_params_.depthanything_da_prior_max_pixel_dist_ =
            n.empty() ? 0.0f : std::max(0.0f, static_cast<float>(n));
    }
    {
        cv::FileNode n = settings_file["Mapper.depthanything_fill_holes"];
        mono_prior_params_.depthanything_fill_holes_ = n.empty() ? false : ((n.operator int()) != 0);
    }
    {
        cv::FileNode n = settings_file["Mapper.depthanything_fill_holes_initial_backfill"];
        mono_prior_params_.depthanything_fill_holes_initial_backfill_ =
            n.empty() ? true : ((n.operator int()) != 0);
    }
    {
        cv::FileNode n = settings_file["Mapper.depthanything_fill_holes_warmup_iter"];
        mono_prior_params_.depthanything_fill_holes_warmup_iter_ =
            n.empty() ? 0 : std::max(0, static_cast<int>(n));
    }
    {
        cv::FileNode n = settings_file["Mapper.depthanything_fill_holes_warmup"];
        mono_prior_params_.depthanything_fill_holes_warmup_ =
            n.empty()
                ? (mono_prior_params_.depthanything_fill_holes_warmup_iter_ > 0)
                : ((n.operator int()) != 0);
    }
    {
        cv::FileNode n = settings_file["Optimization.mono_prior_model_id"];
        if (n.empty()) {
            n = settings_file["Optimization.depthanythingv2_model_id"];
        }
        mono_prior_params_.mono_prior_model_id_ =
            n.empty() ? std::string("depth-anything/Depth-Anything-V2-Small-hf")
                      : static_cast<std::string>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.mono_prior_loss_mode"];
        if (n.empty()) {
            n = settings_file["Optimization.depthanythingv2_loss_mode"];
        }
        mono_prior_params_.mono_prior_loss_mode_ =
            normalizeMonoPriorLossMode(n.empty() ? std::string("svraster")
                                                 : static_cast<std::string>(n));
        if (mono_prior_params_.mono_prior_loss_mode_ != "svraster" &&
            mono_prior_params_.mono_prior_loss_mode_ != "aligned" &&
            mono_prior_params_.mono_prior_loss_mode_ != "geosvr") {
            std::cerr << "[MonoPrior] Unknown Optimization.mono_prior_loss_mode='"
                      << mono_prior_params_.mono_prior_loss_mode_
                      << "', falling back to 'svraster'. Supported: svraster, aligned, geosvr."
                      << std::endl;
            mono_prior_params_.mono_prior_loss_mode_ = "svraster";
        }
    }
    if (monoPriorUsesMetricDepth() && mono_prior_params_.mono_prior_loss_mode_ == "svraster") {
        std::cerr << "[Metric3D] Optimization.mono_prior_loss_mode='svraster' is not "
                     "used for Metric3D priors. Falling back to 'aligned'."
                  << std::endl;
        mono_prior_params_.mono_prior_loss_mode_ = "aligned";
    }
    {
        cv::FileNode n = settings_file["Optimization.mono_prior_normal_mode"];
        if (n.empty()) {
            n = settings_file["Optimization.depthanythingv2_normal_mode"];
        }
        if (n.empty()) {
            n = settings_file["Optimization.lambda_depthanythingv2_normal_mode"];
        }
        if (n.empty()) {
            mono_prior_params_.mono_prior_normal_mode_ =
                (mono_prior_params_.mono_prior_loss_mode_ == "geosvr") ? "geosvr" : "aligned";
        } else {
            mono_prior_params_.mono_prior_normal_mode_ =
                normalizeMonoPriorNormalMode(static_cast<std::string>(n));
        }
        if (mono_prior_params_.mono_prior_normal_mode_ != "aligned" &&
            mono_prior_params_.mono_prior_normal_mode_ != "geosvr") {
            std::cerr << "[MonoPrior] Unknown Optimization.mono_prior_normal_mode='"
                      << mono_prior_params_.mono_prior_normal_mode_
                      << "', falling back to 'aligned'. Supported: aligned, geosvr."
                      << std::endl;
            mono_prior_params_.mono_prior_normal_mode_ = "aligned";
        }
    }
}

bool VoxelMapper::monoPriorUsesMetricDepth() const
{
    return isMetric3DModelId(mono_prior_params_.mono_prior_model_id_);
}

bool VoxelMapper::ensureMonoPriorForKeyframe(
    const std::shared_ptr<VoxelKeyframe>& kf)
{
    const char* log_prefix = monoPriorLogPrefix(mono_prior_params_.mono_prior_model_id_);
    if (!kf || !kf->original_image_.defined()) {
        return false;
    }
    if (kf->mono_prior_.defined() && kf->mono_prior_.numel() > 0) {
        if (kf->mono_prior_prepare_iter_ < 0) {
            kf->mono_prior_prepare_iter_ = getIteration();
        }
        return true;
    }

    ensureEmbeddedPythonRuntime(/*import_torch_cuda=*/true);

    try {
        py::gil_scoped_acquire gil;
        static py::object py_load_or_infer;
        if (!py_load_or_infer) {
            py_load_or_infer =
                py::module_::import("scripts_voxel.python_svraster_bridge.mono_prior_helper")
                    .attr("load_or_infer_mono_prior");
        }

        const std::filesystem::path depth_root =
            model_params_.source_path_ / "mono_priors" / "depthanythingv2";
        std::filesystem::create_directories(depth_root);

        torch::Tensor image_cpu =
            kf->original_image_.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
        const std::string cache_key = monoPriorCacheKeyForKeyframe(kf);

        py::object py_depth = py_load_or_infer(
            py::cast(image_cpu),
            py::str(depth_root.string()),
            py::str(cache_key),
            py::str(mono_prior_params_.mono_prior_model_id_),
            py::bool_(true),
            py::float_(kf->intr_.empty() ? 0.0f : static_cast<double>(kf->intr_[0])));

        torch::Tensor depth = py_depth.cast<torch::Tensor>()
                                  .detach()
                                  .to(torch::kCPU)
                                  .to(torch::kFloat32)
                                  .contiguous();
        if (depth.dim() == 3 && depth.size(0) == 1) {
            depth = depth.squeeze(0);
        }
        if (depth.dim() != 2) {
            std::cerr << log_prefix << " Unexpected prior shape for keyframe "
                      << kf->fid_ << ": " << depth.sizes() << std::endl;
            return false;
        }

        kf->mono_prior_ = depth;
        kf->mono_prior_prepare_iter_ = getIteration();
        return true;
    } catch (const py::error_already_set& e) {
        std::cerr << log_prefix << " Python exception while preparing prior for keyframe "
                  << kf->fid_ << ":\n" << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << log_prefix << " Failed to prepare prior for keyframe "
                  << kf->fid_ << ": " << e.what() << std::endl;
    }

    return false;
}

bool VoxelMapper::buildAlignedMonoPriorDepthForKeyframe(
    const std::shared_ptr<VoxelKeyframe>& kf,
    const sv::MiniCam& cam,
    int image_width,
    int image_height,
    torch::Tensor& aligned_depth)
{
    torch::NoGradGuard no_grad;
    aligned_depth = torch::Tensor();

    if (!kf || image_width <= 0 || image_height <= 0) {
        return false;
    }
    if (!ensureMonoPriorForKeyframe(kf) ||
        !kf->mono_prior_.defined() ||
        kf->mono_prior_.numel() == 0) {
        return false;
    }

    torch::Tensor mono_prior =
        kf->mono_prior_.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (mono_prior.dim() == 3 && mono_prior.size(0) == 1) {
        mono_prior = mono_prior.squeeze(0);
    }
    if (mono_prior.dim() != 2) {
        return false;
    }
    if (mono_prior.size(0) != image_height || mono_prior.size(1) != image_width) {
        mono_prior = torch::nn::functional::interpolate(
            mono_prior.unsqueeze(0).unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{image_height, image_width})
                .mode(torch::kBilinear)
                .align_corners(false)).squeeze().to(torch::kCPU).contiguous();
    }

    if (mono_prior_params_.depthanything_densify_alignment_mode_ == "da_prior" &&
        kf->mono_prior_aligned_depth_cache_.defined() &&
        kf->mono_prior_aligned_depth_cache_mode_ == "da_prior" &&
        kf->mono_prior_aligned_depth_cache_.dim() == 2 &&
        kf->mono_prior_aligned_depth_cache_.size(0) == image_height &&
        kf->mono_prior_aligned_depth_cache_.size(1) == image_width) {
        aligned_depth = kf->mono_prior_aligned_depth_cache_.detach()
                            .to(torch::kCPU)
                            .to(torch::kFloat32)
                            .contiguous();
        return true;
    }

    const bool use_metric3d = isMetric3DModelId(mono_prior_params_.mono_prior_model_id_);
    torch::Tensor sparse_uv;
    torch::Tensor sparse_depth;
    MonoPriorAlignmentStats stats;
    if (buildSparseDepthFromKeyframeOrbAnchors(
            kf,
            image_width,
            image_height,
            sparse_uv,
            sparse_depth) &&
        (((mono_prior_params_.depthanything_densify_alignment_mode_ == "da_prior") &&
          alignMonoPriorToSparseAnchorsDaPrior(
              mono_prior,
              sparse_uv,
              sparse_depth,
              torch::Tensor(),
              use_metric3d,
              cam.near,
              mono_prior_params_.depthanything_densify_min_sparse_anchors_,
              mono_prior_params_.depthanything_da_prior_knn_k_,
              mono_prior_params_.depthanything_da_prior_distance_weighting_,
              mono_prior_params_.depthanything_da_prior_max_pixel_dist_,
              aligned_depth,
              stats)) ||
         (use_metric3d &&
          alignMetricDepthPriorToSparseAnchors(
              mono_prior,
              sparse_uv,
              sparse_depth,
              cam.near,
              mono_prior_params_.depthanything_densify_min_sparse_anchors_,
              aligned_depth,
              stats)) ||
         (!use_metric3d &&
          alignDepthAnythingPriorToSparseAnchors(
              mono_prior,
              sparse_uv,
              sparse_depth,
              cam.near,
              mono_prior_params_.depthanything_densify_min_sparse_anchors_,
              aligned_depth,
              stats)))) {
        if (mono_prior_params_.depthanything_densify_alignment_mode_ == "da_prior") {
            kf->mono_prior_aligned_depth_cache_ =
                aligned_depth.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
            kf->mono_prior_aligned_depth_cache_mode_ = "da_prior";
            kf->mono_prior_aligned_depth_cache_iter_ = getIteration();
        }
        return aligned_depth.defined() && aligned_depth.dim() == 2;
    }

    if (mono_prior_state_.depthanything_global_alignment_valid_ &&
        ((use_metric3d &&
          applyMetricDepthAlignment(
              mono_prior,
              mono_prior_state_.depthanything_global_align_scale_,
              mono_prior_state_.depthanything_global_align_shift_,
              aligned_depth)) ||
         (!use_metric3d &&
          applyDepthAnythingAffineAlignment(
              mono_prior,
              mono_prior_state_.depthanything_global_align_scale_,
              mono_prior_state_.depthanything_global_align_shift_,
              aligned_depth)))) {
        return aligned_depth.defined() && aligned_depth.dim() == 2;
    }

    return false;
}

bool VoxelMapper::depthAnythingFillHolesWarmupReady()
{
    return !mono_prior_params_.depthanything_fill_holes_warmup_ ||
           mono_prior_params_.depthanything_fill_holes_warmup_iter_ <= 0 ||
           getIteration() >= mono_prior_params_.depthanything_fill_holes_warmup_iter_;
}

void VoxelMapper::queueDepthAnythingFillHolesKeyframe(
    const std::shared_ptr<VoxelKeyframe>& pkf)
{
    if (!pkf || pkf->mono_prior_first_apply_iter_ >= 0) {
        return;
    }
    if (mono_prior_state_.depthanything_fill_holes_pending_kfids_.insert(pkf->fid_).second) {
        mono_prior_state_.depthanything_fill_holes_pending_kfs_.push_back(pkf);
    }
}

void VoxelMapper::applyDepthAnythingFillHolesKeyframes(
    const std::vector<std::shared_ptr<VoxelKeyframe>>& keyframes,
    bool seed_global_alignment)
{
    if (!mono_prior_params_.depthanything_fill_holes_ || sensor_type_ != MONOCULAR || keyframes.empty()) {
        return;
    }

    if (seed_global_alignment) {
        for (const auto& pkf : keyframes) {
            if (pkf && pkf->mono_prior_first_apply_iter_ < 0) {
                updateMonoPriorGlobalAlignmentFromKeyframe(pkf);
            }
        }
    }

    for (const auto& pkf : keyframes) {
        if (!pkf || pkf->mono_prior_first_apply_iter_ >= 0) {
            continue;
        }
        const std::shared_ptr<VoxelKeyframe> scene_kf = scene_ ? scene_->getKeyframe(pkf->fid_) : nullptr;
        if (scene_kf && scene_kf.get() == pkf.get()) {
            increasePcdByKeyframeMonoPriorFillHoles(pkf);
        }
    }
}

void VoxelMapper::processDepthAnythingFillHolesWarmup()
{
    if (!mono_prior_params_.depthanything_fill_holes_ ||
        !mono_prior_params_.depthanything_fill_holes_warmup_ ||
        mono_prior_state_.depthanything_fill_holes_warmup_flushed_ ||
        !depthAnythingFillHolesWarmupReady()) {
        return;
    }

    std::vector<std::shared_ptr<VoxelKeyframe>> pending;
    pending.reserve(mono_prior_state_.depthanything_fill_holes_pending_kfs_.size());
    for (const auto& weak_pkf : mono_prior_state_.depthanything_fill_holes_pending_kfs_) {
        std::shared_ptr<VoxelKeyframe> pkf = weak_pkf.lock();
        if (!pkf || pkf->mono_prior_first_apply_iter_ >= 0) {
            continue;
        }
        const std::shared_ptr<VoxelKeyframe> scene_kf = scene_ ? scene_->getKeyframe(pkf->fid_) : nullptr;
        if (scene_kf && scene_kf.get() == pkf.get()) {
            pending.push_back(pkf);
        }
    }

    mono_prior_state_.depthanything_fill_holes_pending_kfs_.clear();
    mono_prior_state_.depthanything_fill_holes_pending_kfids_.clear();
    mono_prior_state_.depthanything_fill_holes_warmup_flushed_ = true;

    applyDepthAnythingFillHolesKeyframes(
        pending,
        /*seed_global_alignment=*/true);
}

void VoxelMapper::scheduleDepthAnythingFillHoles(
    const std::shared_ptr<VoxelKeyframe>& pkf)
{
    if (!mono_prior_params_.depthanything_fill_holes_ || !pkf || pkf->mono_prior_first_apply_iter_ >= 0) {
        return;
    }

    if (!depthAnythingFillHolesWarmupReady()) {
        queueDepthAnythingFillHolesKeyframe(pkf);
        return;
    }

    processDepthAnythingFillHolesWarmup();
    if (pkf->mono_prior_first_apply_iter_ < 0) {
        applyDepthAnythingFillHolesKeyframes(
            std::vector<std::shared_ptr<VoxelKeyframe>>{pkf},
            /*seed_global_alignment=*/false);
    }
}

torch::Tensor VoxelMapper::computeMonoPriorDepthLoss(
    const std::shared_ptr<VoxelKeyframe>& kf,
    const sv::MiniCam& cam,
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    int iteration)
{
    auto zero = torch::zeros(
        {1},
        torch::TensorOptions().dtype(torch::kFloat32).device(mDevice));

    if (opt_params_.lambda_depthanythingv2_ <= 0.0f) {
        return zero;
    }

    loss_utils::DepthAnythingv2Loss depthanything_loss(
        opt_params_.depthanythingv2_from_,
        opt_params_.depthanythingv2_end_,
        opt_params_.depthanythingv2_end_mult_);
    if (!depthanything_loss.isActive(iteration)) {
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
    if (it_T == render_pkg.end() || it_depth == render_pkg.end()) {
        return zero;
    }

    if (!ensureMonoPriorForKeyframe(kf) ||
        !kf->mono_prior_.defined() ||
        kf->mono_prior_.numel() == 0) {
        return zero;
    }

    torch::Tensor raw_T = it_T->second.to(mDevice, torch::kFloat32).contiguous();
    torch::Tensor raw_depth = it_depth->second.to(mDevice, torch::kFloat32).contiguous();
    torch::Tensor mono_depth =
        kf->mono_prior_.to(mDevice, torch::kFloat32).contiguous();

    if (mono_prior_params_.mono_prior_loss_mode_ == "geosvr") {
        loss_utils::DepthAnythingv2UncertaintyLoss geosvr_loss(
            opt_params_.depthanythingv2_from_,
            opt_params_.depthanythingv2_end_,
            opt_params_.depthanythingv2_end_mult_,
            opt_params_.depthanythingv2_overall_,
            opt_params_.depthanythingv2_alpha_adjust_);
        if (!geosvr_loss.isActive(iteration)) {
            return zero;
        }

        torch::Tensor level_weight;
        if (opt_params_.enable_da2_uncertainty_ &&
            iteration >= opt_params_.level_uncertainty_from_) {
            auto it_feat = render_pkg.find("raw_feat");
            if (it_feat == render_pkg.end() || !it_feat->second.defined() ||
                it_feat->second.numel() == 0) {
                it_feat = render_pkg.find("feat");
            }
            auto it_level_T = render_pkg.find("raw_T");
            if (it_level_T == render_pkg.end() || !it_level_T->second.defined()) {
                it_level_T = render_pkg.find("T");
            }

            if (it_feat != render_pkg.end() && it_level_T != render_pkg.end() &&
                it_feat->second.defined() && it_feat->second.numel() > 0 &&
                it_level_T->second.defined() && it_level_T->second.numel() > 0 &&
                !torch::isnan(it_feat->second).any().item<bool>()) {
                using namespace torch::indexing;

                torch::Tensor feat = it_feat->second.to(mDevice, torch::kFloat32).contiguous();
                torch::Tensor T_for_level =
                    it_level_T->second.to(mDevice, torch::kFloat32).contiguous();

                if (feat.dim() == 4 && feat.size(0) == 1) {
                    feat = feat.squeeze(0);
                }
                if (T_for_level.dim() == 4 && T_for_level.size(0) == 1) {
                    T_for_level = T_for_level.squeeze(0);
                }
                if (T_for_level.dim() == 3 && T_for_level.size(0) >= 1) {
                    T_for_level = T_for_level.index({0});
                }
                if (feat.dim() == 3 && feat.size(0) >= 1 && T_for_level.dim() == 2) {
                    torch::Tensor level_map =
                        (feat / (1.0f - T_for_level).clamp_min(0.1f)).squeeze().detach();
                    if (level_map.dim() == 2 &&
                        (level_map.size(0) != raw_depth.size(-2) ||
                         level_map.size(1) != raw_depth.size(-1))) {
                        level_map = torch::nn::functional::interpolate(
                            level_map.unsqueeze(0).unsqueeze(0),
                            torch::nn::functional::InterpolateFuncOptions()
                                .size(std::vector<int64_t>{raw_depth.size(-2), raw_depth.size(-1)})
                                .mode(torch::kNearest)).squeeze();
                    }
                    if (level_map.dim() == 2) {
                        torch::Tensor level_min = level_map.min();
                        torch::Tensor level_weight_2d =
                            (level_map.max() - level_min) /
                            (level_map - level_min).clamp_min(1.0f);
                        level_weight =
                            level_weight_2d.unsqueeze(0).pow(opt_params_.power_level_uncertainty_);
                    }
                }
            }
        }

        if (kf->mono_prior_first_depth_loss_iter_ < 0) {
            kf->mono_prior_first_depth_loss_iter_ = iteration;
        }
        return geosvr_loss(
            raw_T,
            raw_depth,
            mono_depth,
            cam.near,
            iteration,
            sampleGeoSvrPatchSize(),
            level_weight);
    }

    if (mono_prior_params_.mono_prior_loss_mode_ == "aligned") {
        using namespace torch::indexing;

        torch::Tensor depth = raw_depth;
        if (depth.dim() == 4 && depth.size(0) == 1) {
            depth = depth.squeeze(0);
        }
        if (depth.dim() == 2) {
            depth = depth.unsqueeze(0);
        }
        if (depth.dim() != 3 || depth.size(0) < 1) {
            return zero;
        }

        torch::Tensor T = raw_T;
        if (T.dim() == 4 && T.size(0) == 1) {
            T = T.squeeze(0);
        }
        if (T.dim() == 3 && T.size(0) >= 1) {
            T = T.index({0});
        }
        if (T.dim() != 2) {
            return zero;
        }

        const int H = static_cast<int>(depth.size(1));
        const int W = static_cast<int>(depth.size(2));
        torch::Tensor aligned_depth_cpu;
        if (!buildAlignedMonoPriorDepthForKeyframe(kf, cam, W, H, aligned_depth_cpu)) {
            return zero;
        }

        torch::Tensor target_depth =
            aligned_depth_cpu.to(mDevice, torch::kFloat32).contiguous();
        if (target_depth.dim() != 2) {
            return zero;
        }
        if (target_depth.size(0) != H || target_depth.size(1) != W) {
            target_depth = torch::nn::functional::interpolate(
                target_depth.unsqueeze(0).unsqueeze(0),
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{H, W})
                    .mode(torch::kBilinear)
                    .align_corners(false)).squeeze();
        }

        const float near_depth = std::max(1e-6f, cam.near);
        torch::Tensor alpha = (1.0f - T).clamp(0.0f, 1.0f);
        torch::Tensor render_depth = depth.index({0}) / alpha.clamp_min(1e-4f);
        torch::Tensor valid =
            torch::isfinite(render_depth) &
            torch::isfinite(target_depth) &
            (render_depth > near_depth) &
            (target_depth > near_depth) &
            (alpha > 0.5f);
        if (!valid.any().item<bool>()) {
            return zero;
        }

        torch::Tensor inv_render =
            (1.0f / render_depth.clamp_min(near_depth)).masked_select(valid);
        torch::Tensor inv_target =
            (1.0f / target_depth.clamp_min(near_depth)).masked_select(valid);
        torch::Tensor loss = (inv_render - inv_target).abs().mean();
        if (kf->mono_prior_first_depth_loss_iter_ < 0) {
            kf->mono_prior_first_depth_loss_iter_ = iteration;
        }

        if (opt_params_.depthanythingv2_end_ <= opt_params_.depthanythingv2_from_ ||
            opt_params_.depthanythingv2_end_mult_ == 1.0f) {
            return loss;
        }

        const float ratio = std::clamp(
            static_cast<float>(iteration - opt_params_.depthanythingv2_from_) /
                static_cast<float>(opt_params_.depthanythingv2_end_ -
                                   opt_params_.depthanythingv2_from_),
            0.0f,
            1.0f);
        const float mult = std::pow(opt_params_.depthanythingv2_end_mult_, ratio);
        return loss * mult;
    }

    if (kf->mono_prior_first_depth_loss_iter_ < 0) {
        kf->mono_prior_first_depth_loss_iter_ = iteration;
    }
    return depthanything_loss(raw_T, raw_depth, mono_depth, cam.near, iteration);
}

torch::Tensor VoxelMapper::computeMonoPriorNormalLoss(
    const std::shared_ptr<VoxelKeyframe>& kf,
    const sv::MiniCam& cam,
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    int iteration)
{
    const bool use_geosvr_normal = (mono_prior_params_.mono_prior_normal_mode_ == "geosvr");
    auto zero = torch::zeros(
        {1},
        torch::TensorOptions().dtype(torch::kFloat32).device(mDevice));

    if (opt_params_.lambda_depthanythingv2_normal_ <= 0.0f) {
        return zero;
    }
    if (iteration < opt_params_.depthanythingv2_normal_from_ ||
        iteration > opt_params_.depthanythingv2_normal_end_) {
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
    auto it_normal = render_pkg.find("raw_normal");
    if (it_normal == render_pkg.end()) {
        it_normal = render_pkg.find("normal");
    }
    if (it_T == render_pkg.end() || it_depth == render_pkg.end() || it_normal == render_pkg.end()) {
        return zero;
    }

    if (!ensureMonoPriorForKeyframe(kf) ||
        !kf->mono_prior_.defined() ||
        kf->mono_prior_.numel() == 0) {
        return zero;
    }

    torch::Tensor raw_T = it_T->second.to(mDevice, torch::kFloat32).contiguous();
    torch::Tensor raw_depth = it_depth->second.to(mDevice, torch::kFloat32).contiguous();
    torch::Tensor raw_normal = it_normal->second.to(mDevice, torch::kFloat32).contiguous();
    torch::Tensor mono_depth = kf->mono_prior_.to(mDevice, torch::kFloat32).contiguous();

    if (raw_depth.dim() == 4 && raw_depth.size(0) == 1) {
        raw_depth = raw_depth.squeeze(0);
    }
    if (raw_depth.dim() == 2) {
        raw_depth = raw_depth.unsqueeze(0);
    }
    if (raw_depth.dim() != 3) {
        return zero;
    }

    if (raw_T.dim() == 4 && raw_T.size(0) == 1) {
        raw_T = raw_T.squeeze(0);
    }
    if (raw_T.dim() == 2) {
        raw_T = raw_T.unsqueeze(0);
    }
    if (raw_T.dim() != 3) {
        return zero;
    }

    if (raw_normal.dim() == 4 && raw_normal.size(0) == 1) {
        raw_normal = raw_normal.squeeze(0);
    }
    if (raw_normal.dim() != 3 || raw_normal.size(0) < 3) {
        return zero;
    }
    if (raw_normal.size(0) > 3) {
        raw_normal = raw_normal.index({torch::indexing::Slice(0, 3)}).contiguous();
    }

    torch::Tensor Y = mono_depth;
    if (Y.dim() == 4 && Y.size(0) == 1) {
        Y = Y.squeeze(0);
    }
    if (Y.dim() == 3 && Y.size(0) == 1) {
        Y = Y.squeeze(0);
    }
    if (Y.dim() == 2) {
        Y = Y.unsqueeze(0).unsqueeze(0);
    } else if (Y.dim() == 3 && Y.size(0) == 1) {
        Y = Y.unsqueeze(0);
    }
    if (Y.dim() != 4) {
        return zero;
    }

    torch::Tensor alpha = 1.0f - raw_T.index({0});
    if (alpha.dim() == 2) {
        alpha = alpha.unsqueeze(0);
    }
    if (alpha.dim() != 3) {
        return zero;
    }
    if (alpha.size(0) > 1) {
        alpha = alpha.index({torch::indexing::Slice(0, 1)}).contiguous();
    }

    torch::Tensor target_normal;
    {
        torch::NoGradGuard no_grad;
        torch::Tensor target_depth;
        if (!use_geosvr_normal) {
            if (!buildAlignedMonoPriorDepthForKeyframe(
                    kf,
                    cam,
                    static_cast<int>(raw_normal.size(2)),
                    static_cast<int>(raw_normal.size(1)),
                    target_depth) ||
                !target_depth.defined() || target_depth.dim() != 2) {
                return zero;
            }
            const float tol_cos = std::cos(
                opt_params_.depthanythingv2_normal_tol_deg_ *
                static_cast<float>(M_PI) / 180.0f);
            target_normal = voxel_eval::depth2normalSVRaster(
                cam,
                target_depth.to(mDevice, torch::kFloat32).contiguous(),
                opt_params_.depthanythingv2_normal_ks_,
                tol_cos);
        } else {
            torch::Tensor mono = Y;
            if (mono.sizes().slice(2) != raw_normal.sizes().slice(1)) {
                mono = torch::nn::functional::interpolate(
                    mono,
                    torch::nn::functional::InterpolateFuncOptions()
                        .size(std::vector<int64_t>{raw_normal.size(1), raw_normal.size(2)})
                        .mode(torch::kBilinear)
                        .align_corners(false));
            }

            if (isMetric3DModelId(mono_prior_params_.mono_prior_model_id_)) {
                target_depth = mono.squeeze(0).squeeze(0).clamp_min(std::max(1e-6f, cam.near));
            } else {
                target_depth = (1.0f / mono.clamp_min(1e-3f)).squeeze(0).squeeze(0);
            }

            target_normal = voxel_eval::depth2normalSVRaster(
                cam,
                target_depth,
                3,
                -1.0f);
        }
    }

    torch::Tensor render_normal = raw_normal;

    torch::Tensor mask =
        (target_normal != 0).any(0).unsqueeze(0).repeat({3, 1, 1}) &
        (alpha > 0.8f).repeat({3, 1, 1});
    mask = mask.to(render_normal.dtype());
    if (!mask.any().item<bool>()) {
        return zero;
    }

    torch::Tensor dot =
        (render_normal * target_normal).sum(0).clamp(-1.0f, 1.0f);
    torch::Tensor loss_map = (1.0f - dot) * mask;
    // L1 normal term disabled to match SVRaster/HI-SLAM2-style cosine normal supervision.
    // torch::Tensor loss_map_abs =
    //     (render_normal - target_normal).abs().sum(0) * mask;
    // torch::Tensor loss = loss_map.mean() + loss_map_abs.mean();
    torch::Tensor loss = loss_map.mean();

    if (use_geosvr_normal) {
        return loss;
    }

    if (opt_params_.depthanythingv2_normal_end_ <= opt_params_.depthanythingv2_normal_from_ ||
        opt_params_.depthanythingv2_normal_end_mult_ == 1.0f) {
        return loss;
    }

    const float ratio = std::clamp(
        static_cast<float>(iteration - opt_params_.depthanythingv2_normal_from_) /
            static_cast<float>(opt_params_.depthanythingv2_normal_end_ -
                               opt_params_.depthanythingv2_normal_from_),
        0.0f,
        1.0f);
    const float mult = std::pow(opt_params_.depthanythingv2_normal_end_mult_, ratio);
    return loss * mult;
}

void VoxelMapper::accumulateMonoPriorGlobalAlignment(
    float scale,
    float shift,
    float weight)
{
    if (!std::isfinite(scale) || !std::isfinite(shift) || scale <= 0.0f) {
        return;
    }

    weight = std::max(1.0f, weight);
    if (!mono_prior_state_.depthanything_global_alignment_valid_ ||
        !std::isfinite(mono_prior_state_.depthanything_global_align_scale_) ||
        !std::isfinite(mono_prior_state_.depthanything_global_align_shift_) ||
        mono_prior_state_.depthanything_global_align_weight_ <= 0.0f) {
        mono_prior_state_.depthanything_global_align_scale_ = scale;
        mono_prior_state_.depthanything_global_align_shift_ = shift;
        mono_prior_state_.depthanything_global_align_weight_ = weight;
        mono_prior_state_.depthanything_global_align_observations_ = 1;
        mono_prior_state_.depthanything_global_alignment_valid_ = true;
        return;
    }

    const float old_weight = mono_prior_state_.depthanything_global_align_weight_;
    const float new_weight = old_weight + weight;
    mono_prior_state_.depthanything_global_align_scale_ =
        (mono_prior_state_.depthanything_global_align_scale_ * old_weight + scale * weight) / new_weight;
    mono_prior_state_.depthanything_global_align_shift_ =
        (mono_prior_state_.depthanything_global_align_shift_ * old_weight + shift * weight) / new_weight;
    mono_prior_state_.depthanything_global_align_weight_ = new_weight;
    ++mono_prior_state_.depthanything_global_align_observations_;
}

bool VoxelMapper::updateMonoPriorGlobalAlignmentFromKeyframe(
    const std::shared_ptr<VoxelKeyframe>& pkf)
{
    torch::NoGradGuard no_grad;

    if (!pkf || sensor_type_ != MONOCULAR) {
        return false;
    }

    const int H = pkf->image_height_;
    const int W = pkf->image_width_;
    if (H <= 0 || W <= 0) {
        return false;
    }
    if (!ensureMonoPriorForKeyframe(pkf) ||
        !pkf->mono_prior_.defined() ||
        pkf->mono_prior_.numel() == 0) {
        return false;
    }

    torch::Tensor mono_prior =
        pkf->mono_prior_.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (mono_prior.dim() == 3 && mono_prior.size(0) == 1) {
        mono_prior = mono_prior.squeeze(0);
    }
    if (mono_prior.dim() != 2) {
        return false;
    }
    if (mono_prior.size(0) != H || mono_prior.size(1) != W) {
        mono_prior = torch::nn::functional::interpolate(
            mono_prior.unsqueeze(0).unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{H, W})
                .mode(torch::kBilinear)
                .align_corners(false)).squeeze().to(torch::kCPU).contiguous();
    }

    torch::Tensor sparse_uv;
    torch::Tensor sparse_depth;
    if (!buildSparseDepthFromKeyframeOrbAnchors(pkf, W, H, sparse_uv, sparse_depth)) {
        // std::cout << "[mono_prior_align/global_seed] iter=" << getIteration()
        //           << " kf=" << pkf->fid_
        //           << " no valid ORB anchors"
        //           << std::endl;
        return false;
    }

    sv::MiniCam cam = pkf->toMiniCam(H, W);
    const bool use_metric3d = isMetric3DModelId(mono_prior_params_.mono_prior_model_id_);
    torch::Tensor aligned_depth;
    MonoPriorAlignmentStats stats;
    const bool aligned_ok = use_metric3d
        ? alignMetricDepthPriorToSparseAnchors(
              mono_prior,
              sparse_uv,
              sparse_depth,
              cam.near,
              mono_prior_params_.depthanything_densify_min_sparse_anchors_,
              aligned_depth,
              stats)
        : alignDepthAnythingPriorToSparseAnchors(
              mono_prior,
              sparse_uv,
              sparse_depth,
              cam.near,
              mono_prior_params_.depthanything_densify_min_sparse_anchors_,
              aligned_depth,
              stats);
    if (!aligned_ok) {
        // std::cout << "[mono_prior_align/global_seed] iter=" << getIteration()
        //           << " kf=" << pkf->fid_
        //           << " failed num_valid_anchors=" << stats.num_valid_anchors
        //           << " fit_anchors=" << stats.num_fit_anchors
        //           << " scale=" << stats.scale
        //           << " shift=" << stats.shift
        //           << " med_rel_err=" << stats.median_rel_depth_error
        //           << " p90_rel_err=" << stats.p90_rel_depth_error
        //           << " inlier_ratio=" << stats.final_inlier_ratio
        //           << std::endl;
        return false;
    }

    accumulateMonoPriorGlobalAlignment(
        stats.scale,
        stats.shift,
        static_cast<float>(std::max<int64_t>(1, stats.num_fit_anchors)));
    // std::cout << "[mono_prior_align/global_seed] iter=" << getIteration()
    //           << " kf=" << pkf->fid_
    //           << " accepted scale=" << stats.scale
    //           << " shift=" << stats.shift
    //           << " med_rel_err=" << stats.median_rel_depth_error
    //           << " p90_rel_err=" << stats.p90_rel_depth_error
    //           << " inlier_ratio=" << stats.final_inlier_ratio
    //           << " global_scale=" << mono_prior_state_.depthanything_global_align_scale_
    //           << " global_shift=" << mono_prior_state_.depthanything_global_align_shift_
    //           << " global_obs=" << mono_prior_state_.depthanything_global_align_observations_
    //           << std::endl;
    return true;
}

void VoxelMapper::increasePcdByKeyframeMonoPriorFillHoles(
    std::shared_ptr<VoxelKeyframe> pkf)
{
    torch::NoGradGuard no_grad;

    if (!mono_prior_params_.depthanything_fill_holes_ || !pkf || !voxel_model_) {
        return;
    }

    const int iter = getIteration();
    if (sensor_type_ != MONOCULAR) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    const int H = pkf->image_height_;
    const int W = pkf->image_width_;
    if (H <= 0 || W <= 0) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    sv::MiniCam cam = pkf->toMiniCam(H, W);

    auto finiteTensorStats = [](const torch::Tensor& t,
                                int64_t* count_out,
                                float* min_out,
                                float* max_out,
                                float* mean_out) {
        *count_out = 0;
        *min_out = std::numeric_limits<float>::quiet_NaN();
        *max_out = std::numeric_limits<float>::quiet_NaN();
        *mean_out = std::numeric_limits<float>::quiet_NaN();
        if (!t.defined() || t.numel() == 0) {
            return;
        }
        torch::Tensor cpu = t.to(torch::kCPU).to(torch::kFloat32).contiguous();
        torch::Tensor finite = torch::masked_select(cpu, torch::isfinite(cpu));
        if (!finite.defined() || finite.numel() == 0) {
            return;
        }
        *count_out = finite.numel();
        *min_out = finite.min().item<float>();
        *max_out = finite.max().item<float>();
        *mean_out = finite.mean().item<float>();
    };

    std::unordered_map<std::string, torch::Tensor> render_pkg;
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        render_pkg = voxel_model_->render(
            cam,
            H,
            W,
            torch::Tensor(),
            "dontcare",
            false,
            std::nullopt,
            true,
            false,
            true,
            false,
            false,
            sv::RenderOpts{});
    }
    torch::Tensor depth_cpu;
    torch::Tensor alpha_cpu;
    torch::Tensor n_contrib_cpu;
    if (!renderPkgToMonoPriorAlignmentMaps(
            render_pkg,
            H,
            W,
            depth_cpu,
            alpha_cpu,
            n_contrib_cpu)) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    if (!ensureMonoPriorForKeyframe(pkf) ||
        !pkf->mono_prior_.defined() ||
        pkf->mono_prior_.numel() == 0) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    torch::Tensor mono_prior =
        pkf->mono_prior_.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (mono_prior.dim() == 3 && mono_prior.size(0) == 1) {
        mono_prior = mono_prior.squeeze(0);
    }
    if (mono_prior.dim() != 2) {
        // std::cout << "[mono_prior_fill_holes/start] iter=" << iter
        //           << " kf=" << pkf->fid_
        //           << " invalid prior shape=" << mono_prior.sizes()
        //           << std::endl;
        updateRenderedDepthCandidateLifecycle();
        return;
    }
    if (mono_prior.size(0) != H || mono_prior.size(1) != W) {
        mono_prior = torch::nn::functional::interpolate(
                        mono_prior.unsqueeze(0).unsqueeze(0),
                        torch::nn::functional::InterpolateFuncOptions()
                            .size(std::vector<int64_t>{H, W})
                            .mode(torch::kBilinear)
                            .align_corners(false))
                         .squeeze()
                         .to(torch::kCPU)
                         .contiguous();
    }

    const bool use_metric3d = isMetric3DModelId(mono_prior_params_.mono_prior_model_id_);

    auto depth_acc = depth_cpu.accessor<float, 2>();
    auto n_contrib_acc = n_contrib_cpu.accessor<int, 2>();

    const int active_hole_max_n_contrib = 0;
    std::vector<uint8_t> hole_mask(static_cast<size_t>(H) * static_cast<size_t>(W), 0);
    auto flat = [W](int y, int x) -> int64_t {
        return static_cast<int64_t>(y) * static_cast<int64_t>(W) + static_cast<int64_t>(x);
    };

    int64_t hole_pixels = 0;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const float z = depth_acc[y][x];
            const int n_contrib = n_contrib_acc[y][x];
            const bool is_hole =
                (n_contrib <= active_hole_max_n_contrib) &&
                (!std::isfinite(z) || z <= kMonoPriorFillHolesEmptyDepthEps);
            if (is_hole) {
                hole_mask[static_cast<size_t>(flat(y, x))] = 1;
                ++hole_pixels;
            }
        }
    }
    if (hole_pixels <= 0) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    torch::Tensor da_prior_query_mask;
    if (mono_prior_params_.depthanything_densify_alignment_mode_ == "da_prior") {
        da_prior_query_mask = torch::from_blob(
                                  hole_mask.data(),
                                  {H, W},
                                  torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
                                  .clone()
                                  .to(torch::kBool);
    }

    torch::Tensor sparse_uv;
    torch::Tensor sparse_depth;
    torch::Tensor aligned_depth;
    MonoPriorAlignmentStats align_stats;
    bool used_rendered_alignment = false;
    bool used_da_prior_alignment = false;
    bool used_global_alignment = false;
    bool alignment_ok = false;
    if (mono_prior_params_.depthanything_densify_alignment_mode_ == "rendered") {
        alignment_ok = use_metric3d
            ? alignMetricDepthPriorToRenderedDepth(
                  mono_prior,
                  depth_cpu,
                  alpha_cpu,
                  n_contrib_cpu,
                  cam.near,
                  mono_prior_params_.depthanything_densify_min_sparse_anchors_,
                  aligned_depth,
                  align_stats)
            : alignDepthAnythingPriorToRenderedDepth(
                  mono_prior,
                  depth_cpu,
                  alpha_cpu,
                  n_contrib_cpu,
                  cam.near,
                  mono_prior_params_.depthanything_densify_min_sparse_anchors_,
                  aligned_depth,
                  align_stats);
        used_rendered_alignment = alignment_ok;
    }
    const bool has_sparse_anchors =
        buildSparseDepthFromKeyframeOrbAnchors(pkf, W, H, sparse_uv, sparse_depth);
    if (!alignment_ok && has_sparse_anchors) {
        if (mono_prior_params_.depthanything_densify_alignment_mode_ == "da_prior") {
            alignment_ok = alignMonoPriorToSparseAnchorsDaPrior(
                mono_prior,
                sparse_uv,
                sparse_depth,
                da_prior_query_mask,
                use_metric3d,
                cam.near,
                mono_prior_params_.depthanything_densify_min_sparse_anchors_,
                mono_prior_params_.depthanything_da_prior_knn_k_,
                mono_prior_params_.depthanything_da_prior_distance_weighting_,
                mono_prior_params_.depthanything_da_prior_max_pixel_dist_,
                aligned_depth,
                align_stats);
            used_da_prior_alignment = alignment_ok;
        }
        if (!alignment_ok) {
            alignment_ok = use_metric3d
                ? alignMetricDepthPriorToSparseAnchors(
                      mono_prior,
                      sparse_uv,
                      sparse_depth,
                      cam.near,
                      mono_prior_params_.depthanything_densify_min_sparse_anchors_,
                      aligned_depth,
                      align_stats)
                : alignDepthAnythingPriorToSparseAnchors(
                      mono_prior,
                      sparse_uv,
                      sparse_depth,
                      cam.near,
                      mono_prior_params_.depthanything_densify_min_sparse_anchors_,
                      aligned_depth,
                      align_stats);
        }
        if (alignment_ok) {
            accumulateMonoPriorGlobalAlignment(
                align_stats.scale,
                align_stats.shift,
                static_cast<float>(std::max<int64_t>(1, align_stats.num_fit_anchors)));
        }
    }
    if (!alignment_ok) {
        if (mono_prior_state_.depthanything_global_alignment_valid_ &&
            ((use_metric3d &&
              applyMetricDepthAlignment(
                  mono_prior,
                  mono_prior_state_.depthanything_global_align_scale_,
                  mono_prior_state_.depthanything_global_align_shift_,
                  aligned_depth)) ||
             (!use_metric3d &&
              applyDepthAnythingAffineAlignment(
                  mono_prior,
                  mono_prior_state_.depthanything_global_align_scale_,
                  mono_prior_state_.depthanything_global_align_shift_,
                  aligned_depth)))) {
            used_global_alignment = true;
            align_stats.scale = mono_prior_state_.depthanything_global_align_scale_;
            align_stats.shift = mono_prior_state_.depthanything_global_align_shift_;
        } else {
            std::cout << "[mono_prior_alignment] path=fill_holes"
                      << " iter=" << iter
                      << " kf=" << pkf->fid_
                      << " requested=" << mono_prior_params_.depthanything_densify_alignment_mode_
                      << " used=none"
                      << " fallback=0"
                      << " failed=1"
                      << " has_sparse_anchors=" << (has_sparse_anchors ? 1 : 0)
                      << " anchors=" << align_stats.num_valid_anchors
                      << " fit_anchors=" << align_stats.num_fit_anchors
                      << " min_required=" << mono_prior_params_.depthanything_densify_min_sparse_anchors_
                      << " global_valid=" << (mono_prior_state_.depthanything_global_alignment_valid_ ? 1 : 0)
                      << std::endl;
            updateRenderedDepthCandidateLifecycle();
            return;
        }
    }
    const char* fill_holes_alignment_source =
        used_rendered_alignment ? "rendered" :
        (used_da_prior_alignment ? "da_prior" :
         (used_global_alignment ? "global" : "orb"));
    const bool fill_holes_alignment_fallback =
        mono_prior_params_.depthanything_densify_alignment_mode_ != fill_holes_alignment_source;
    std::cout << "[mono_prior_alignment] path=fill_holes"
              << " iter=" << iter
              << " kf=" << pkf->fid_
              << " requested=" << mono_prior_params_.depthanything_densify_alignment_mode_
              << " used=" << fill_holes_alignment_source
              << " fallback=" << (fill_holes_alignment_fallback ? 1 : 0)
              << " anchors=" << align_stats.num_valid_anchors
              << " fit_anchors=" << align_stats.num_fit_anchors
              << " scale=" << align_stats.scale
              << " shift=" << align_stats.shift
              << " med_rel_err=" << align_stats.median_rel_depth_error
              << " p90_rel_err=" << align_stats.p90_rel_depth_error
              << " inlier_ratio=" << align_stats.final_inlier_ratio
              << std::endl;
    int64_t prior_finite_count = 0;
    int64_t aligned_finite_count = 0;
    float prior_min = std::numeric_limits<float>::quiet_NaN();
    float prior_max = std::numeric_limits<float>::quiet_NaN();
    float prior_mean = std::numeric_limits<float>::quiet_NaN();
    float aligned_min = std::numeric_limits<float>::quiet_NaN();
    float aligned_max = std::numeric_limits<float>::quiet_NaN();
    float aligned_mean = std::numeric_limits<float>::quiet_NaN();
    finiteTensorStats(mono_prior, &prior_finite_count, &prior_min, &prior_max, &prior_mean);
    finiteTensorStats(aligned_depth, &aligned_finite_count, &aligned_min, &aligned_max, &aligned_mean);

    auto image_cpu = pkf->original_image_.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (image_cpu.dim() == 4 && image_cpu.size(0) == 1) {
        image_cpu = image_cpu.squeeze(0);
    }
    if (image_cpu.dim() != 3 || image_cpu.size(0) < 3 ||
        image_cpu.size(1) != H || image_cpu.size(2) != W) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }
    if (image_cpu.size(0) > 3) {
        image_cpu = image_cpu.index({torch::indexing::Slice(0, 3)}).contiguous();
    }

    auto image_acc = image_cpu.accessor<float, 3>();

    torch::Tensor valid_mask =
        torch::isfinite(aligned_depth) &
        (aligned_depth > std::max(cam.near, 1e-4f));
    if (std::isfinite(RGBD_max_depth_) && RGBD_max_depth_ > 0.0f) {
        valid_mask &= (aligned_depth < RGBD_max_depth_);
    }
    {
        auto hole_mask_tensor = torch::from_blob(
            hole_mask.data(),
            {H, W},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
        valid_mask &= hole_mask_tensor.to(torch::kBool);
    }
    auto mask_it = undistort_mask_.find(pkf->camera_id_);
    if (mask_it != undistort_mask_.end() && mask_it->second.defined()) {
        torch::Tensor eval_mask = mask_it->second.detach().to(torch::kCPU).to(torch::kFloat32);
        if (eval_mask.dim() == 4 && eval_mask.size(0) == 1) {
            eval_mask = eval_mask.squeeze(0);
        }
        if (eval_mask.dim() == 3) {
            eval_mask = eval_mask.index({0});
        }
        if (eval_mask.dim() == 2 &&
            eval_mask.size(0) == H &&
            eval_mask.size(1) == W) {
            valid_mask &= (eval_mask > 0.5f);
        }
    }
    const int64_t valid_hole_pixels_after_mask = valid_mask.sum().item<int64_t>();

    const float fx = (cam.fx > 1e-6f)
        ? cam.fx
        : (0.5f * static_cast<float>(W) / std::max(cam.tanfovx, 1e-6f));
    const float fy = (cam.fy > 1e-6f)
        ? cam.fy
        : (0.5f * static_cast<float>(H) / std::max(cam.tanfovy, 1e-6f));
    if (fx <= 1e-6f || fy <= 1e-6f) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    const int stride = std::max(1, mono_prior_params_.depthanything_densify_stride_);
    std::vector<int64_t> selected_idx;
    selected_idx.reserve(
        static_cast<size_t>((H + stride - 1) / stride) *
        static_cast<size_t>((W + stride - 1) / stride));
    auto valid_acc = valid_mask.accessor<bool, 2>();
    for (int y = 0; y < H; y += stride) {
        for (int x = 0; x < W; x += stride) {
            if (valid_acc[y][x]) {
                selected_idx.push_back(static_cast<int64_t>(y) * static_cast<int64_t>(W) + x);
            }
        }
    }

    // std::cout << "[mono_prior_fill_holes/start] iter=" << iter
    //           << " kf=" << pkf->fid_
    //           << " anchors=" << align_stats.num_valid_anchors
    //           << " fit_anchors=" << align_stats.num_fit_anchors
    //           << " prior_finite=" << prior_finite_count
    //           << " prior_range=[" << prior_min << "," << prior_max << "]"
    //           << " prior_mean=" << prior_mean
    //           << " aligned_finite=" << aligned_finite_count
    //           << " aligned_range=[" << aligned_min << "," << aligned_max << "]"
    //           << " aligned_mean=" << aligned_mean
    //           << " hole_pixels=" << hole_pixels
    //           << " valid_hole_pixels=" << valid_hole_pixels_after_mask
    //           << " selected_pixels=" << selected_idx.size()
    //           << " stride=" << stride
    //           << " alignment_source="
    //           << (used_rendered_alignment ? "rendered" :
    //               (used_global_alignment ? "global" : "orb"))
    //           << " align_scale=" << align_stats.scale
    //           << " align_shift=" << align_stats.shift
    //           << " med_rel_err=" << align_stats.median_rel_depth_error
    //           << " p90_rel_err=" << align_stats.p90_rel_depth_error
    //           << " inlier_ratio=" << align_stats.final_inlier_ratio
    //           << " global_scale=" << mono_prior_state_.depthanything_global_align_scale_
    //           << " global_shift=" << mono_prior_state_.depthanything_global_align_shift_
    //           << " global_obs=" << mono_prior_state_.depthanything_global_align_observations_
    //           << " anchor_depth_stats=[" << align_stats.sparse_depth_q05
    //           << "," << align_stats.sparse_depth_q95 << "]"
    //           << std::endl;

    if (selected_idx.empty()) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    const Sophus::SE3f Twc = pkf->getPosef().inverse();
    auto aligned_depth_acc = aligned_depth.accessor<float, 2>();

    std::vector<float> candidate_points_world;
    std::vector<float> candidate_colors;
    candidate_points_world.reserve(selected_idx.size() * 3);
    candidate_colors.reserve(selected_idx.size() * 3);
    float candidate_depth_min = std::numeric_limits<float>::infinity();
    float candidate_depth_max = 0.0f;
    Eigen::Vector3f candidate_world_min(
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity());
    Eigen::Vector3f candidate_world_max(
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity());
    std::vector<std::string> sample_candidates;
    sample_candidates.reserve(3);

    for (const int64_t flat_idx : selected_idx) {
        const int y = static_cast<int>(flat_idx / W);
        const int x = static_cast<int>(flat_idx % W);
        const float z = aligned_depth_acc[y][x];
        if (!std::isfinite(z) || z <= 0.0f) {
            continue;
        }

        const float Xc = (static_cast<float>(x) - cam.cx) / fx * z;
        const float Yc = (static_cast<float>(y) - cam.cy) / fy * z;
        const Eigen::Vector3f p_cam(Xc, Yc, z);
        const Eigen::Vector3f p_world = Twc * p_cam;
        if (!std::isfinite(p_world.x()) ||
            !std::isfinite(p_world.y()) ||
            !std::isfinite(p_world.z())) {
            continue;
        }

        candidate_points_world.push_back(p_world.x());
        candidate_points_world.push_back(p_world.y());
        candidate_points_world.push_back(p_world.z());
        candidate_colors.push_back(std::clamp(image_acc[0][y][x], 0.0f, 1.0f));
        candidate_colors.push_back(std::clamp(image_acc[1][y][x], 0.0f, 1.0f));
        candidate_colors.push_back(std::clamp(image_acc[2][y][x], 0.0f, 1.0f));
        candidate_depth_min = std::min(candidate_depth_min, z);
        candidate_depth_max = std::max(candidate_depth_max, z);
        candidate_world_min = candidate_world_min.cwiseMin(p_world);
        candidate_world_max = candidate_world_max.cwiseMax(p_world);
        if (sample_candidates.size() < 3) {
            std::ostringstream oss;
            oss << "(u=" << x
                << ",v=" << y
                << ",z=" << z
                << ",pw=[" << p_world.x() << "," << p_world.y() << "," << p_world.z() << "])";
            sample_candidates.push_back(oss.str());
        }
    }

    const int64_t num_candidates = static_cast<int64_t>(candidate_points_world.size() / 3);
    // std::cout << "[mono_prior_fill_holes/candidates] iter=" << iter
    //           << " kf=" << pkf->fid_
    //           << " valid_candidates=" << num_candidates
    //           << " depth_range=["
    //           << (num_candidates > 0 ? candidate_depth_min : std::numeric_limits<float>::quiet_NaN())
    //           << ","
    //           << (num_candidates > 0 ? candidate_depth_max : std::numeric_limits<float>::quiet_NaN())
    //           << "]"
    //           << " world_bbox_min=["
    //           << (num_candidates > 0 ? candidate_world_min.x() : std::numeric_limits<float>::quiet_NaN()) << ","
    //           << (num_candidates > 0 ? candidate_world_min.y() : std::numeric_limits<float>::quiet_NaN()) << ","
    //           << (num_candidates > 0 ? candidate_world_min.z() : std::numeric_limits<float>::quiet_NaN()) << "]"
    //           << " world_bbox_max=["
    //           << (num_candidates > 0 ? candidate_world_max.x() : std::numeric_limits<float>::quiet_NaN()) << ","
    //           << (num_candidates > 0 ? candidate_world_max.y() : std::numeric_limits<float>::quiet_NaN()) << ","
    //           << (num_candidates > 0 ? candidate_world_max.z() : std::numeric_limits<float>::quiet_NaN()) << "]"
    //           << std::endl;
    // for (size_t i = 0; i < sample_candidates.size(); ++i) {
    //     std::cout << "[mono_prior_fill_holes/sample] iter=" << iter
    //               << " kf=" << pkf->fid_
    //               << " idx=" << i
    //               << " " << sample_candidates[i]
    //               << std::endl;
    // }

    if (num_candidates > 0) {
        saveAccumulatedOrbDepthProjectionPng(
            mpSLAM,
            pkf,
            runtimeOrbDepthDebugDir(result_dir_),
            iter,
            RGBD_min_depth_,
            RGBD_max_depth_,
            aligned_depth);

        auto points_tensor = torch::from_blob(
            candidate_points_world.data(),
            {num_candidates, 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone().to(device_type_);
        auto colors_tensor = torch::from_blob(
            candidate_colors.data(),
            {num_candidates, 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone().to(device_type_);

        std::vector<sv::MiniCam> tr_cams;
        tr_cams.reserve(scene_->keyframes().size());
        for (auto& kv : scene_->keyframes()) {
            if (kv.second) {
                tr_cams.push_back(
                    kv.second->toMiniCam(kv.second->image_height_, kv.second->image_width_));
            }
        }

        std::unique_lock<std::mutex> lock_render(mutex_render_);
        const bool log_depthanything_fill_holes_created_voxels =
            true;
        if (log_depthanything_fill_holes_created_voxels) {
            voxel_model_->setNextRealInsertionRerunEntityPath(
                "world/mono_prior_fill_holes/created");
        }
        voxel_model_->setRenderedDepthCandidateRealAdjacency(
            false,
            1);
        voxel_model_->setNextRenderedDepthCandidateInsertion(
            true,
            "",
            kRenderedCandidateSourceMonoPrior,
            /*insert_as_real_protected=*/true);
        voxel_model_->increasePcd(points_tensor, colors_tensor, iter, tr_cams);
        if (pkf->mono_prior_first_apply_iter_ < 0) {
            pkf->mono_prior_first_apply_iter_ = iter;
        }
        voxel_model_->setNextRenderedDepthCandidateInsertion(false);
        if (log_depthanything_fill_holes_created_voxels) {
            voxel_model_->setNextRealInsertionRerunEntityPath("");
        }
        const sv::VoxelModel::IncreasePcdStats insert_stats = voxel_model_->lastIncreasePcdStats();
        // std::cout << "[mono_prior_fill_holes/insert] iter=" << iter
        //           << " kf=" << pkf->fid_
        //           << " raw_points_in=" << insert_stats.raw_points_in
        //           << " points_after_far_filter=" << insert_stats.points_after_far_filter
        //           << " unique_before_filter=" << insert_stats.unique_voxel_candidates_before_insert_filter
        //           << " unique_after_filter=" << insert_stats.unique_voxel_candidates_after_insert_filter
        //           << " duplicate_existing_voxels=" << insert_stats.duplicate_existing_voxels
        //           << " new_voxels=" << insert_stats.new_voxels
        //           << " pending_promotions=" << insert_stats.pending_promotions
        //           << " pending_support_updates=" << insert_stats.pending_support_updates
        //           << std::endl;
    }

    updateRenderedDepthCandidateLifecycle();
}
