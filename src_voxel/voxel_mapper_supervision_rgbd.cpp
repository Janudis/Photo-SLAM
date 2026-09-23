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
        const sv::Point3D& P = kv.second;          // you already fill xyz_ in run()
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

    // 2) Transform world to camera using cam.w2c.
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

    // 7) Match the renderer projection: 2 * u / W - 1, 2 * v / H - 1.
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

    if (iteration > opt_params_.sparse_depth_until_)
        return zero;

    // 1) Get raw_T / raw_depth from the renderer.
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

    // 2) Build (sparse_uv, sparse_depth) from SLAM 3D points.
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

    // 3) Apply the low-level sparse depth loss.
    torch::Tensor depth_loss = voxel_eval::sparseDepthLoss(
        raw_T, raw_depth, sparse_uv, sparse_depth);

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
        voxel_utils::cvGpuMatToTorchTensorFloat32(depth_gpu)
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
        voxel_utils::cvGpuMatToTorchTensorFloat32(depth_gpu)
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
            voxel_utils::cvGpuMatToTorchTensorFloat32(depth_gpu)
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
        target_normal = voxel_eval::depthToNormal(
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
    // L1 normal term is disabled in favor of cosine normal supervision.
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
