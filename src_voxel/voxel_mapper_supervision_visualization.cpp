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

    bool loaded_external_depth = false;
    if (!pkf->img_auxiliary_undist_.empty()) {
        if (!voxel_utils::depthMatToMeters(pkf->img_auxiliary_undist_, depth_meters)) {
            return false;
        }
    } else {
        if (!voxel_utils::loadReplicaDepthFromRgbPath(pkf->img_filename_, depth_meters) &&
            !voxel_utils::loadTumDepthFromRgbPath(pkf->img_filename_, depth_meters) &&
            !voxel_utils::loadScanNetDepthFromRgbPath(pkf->img_filename_, depth_meters)) {
            return false;
        }
        loaded_external_depth = true;
    }

    if (depth_meters.empty()) {
        return false;
    }

    // Monocular reference depth is loaded in the source-image frame, while
    // mapper keyframes and MVS predictions use the undistorted camera frame.
    if (loaded_external_depth &&
        !pkf->cam_.undistort_map1.empty() &&
        !pkf->cam_.undistort_map2.empty()) {
        if (depth_meters.size() != pkf->cam_.undistort_map1.size()) {
            cv::resize(
                depth_meters,
                depth_meters,
                pkf->cam_.undistort_map1.size(),
                0.0,
                0.0,
                cv::INTER_NEAREST);
        }
        cv::Mat undistorted_depth;
        cv::remap(
            depth_meters,
            undistorted_depth,
            pkf->cam_.undistort_map1,
            pkf->cam_.undistort_map2,
            cv::INTER_NEAREST,
            cv::BORDER_CONSTANT,
            cv::Scalar(0.0f));
        depth_meters = std::move(undistorted_depth);
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
