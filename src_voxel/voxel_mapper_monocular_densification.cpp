#include "include_voxel/voxel_mapper.h"
#include "include_voxel/voxel_mapper_utils.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

constexpr float kReliableRenderedAlpha = 0.95f;
constexpr int kMinimumLocalDepthNeighbors = 2;

sv::MiniCam scaledRenderCamera(
    const std::shared_ptr<VoxelKeyframe>& keyframe,
    const int stride,
    int& render_height,
    int& render_width)
{
    render_height =
        (keyframe->image_height_ + stride - 1) / stride;
    render_width =
        (keyframe->image_width_ + stride - 1) / stride;

    sv::MiniCam camera =
        keyframe->toMiniCam(
            keyframe->image_height_, keyframe->image_width_);
    if (stride <= 1) {
        return camera;
    }

    const float scale = static_cast<float>(stride);
    camera.width = render_width;
    camera.height = render_height;
    camera.fx /= scale;
    camera.fy /= scale;
    camera.cx /= scale;
    camera.cy /= scale;
    const float fov_x =
        sv::focalToFov(camera.fx, render_width);
    const float fov_y =
        sv::focalToFov(camera.fy, render_height);
    camera.tanfovx = std::tan(0.5f * fov_x);
    camera.tanfovy = std::tan(0.5f * fov_y);
    camera.pix_size =
        2.0f * camera.tanfovx / static_cast<float>(render_width);
    return camera;
}

cv::Mat floatColorImage(const cv::Mat& image)
{
    if (image.empty()) {
        return cv::Mat();
    }

    cv::Mat color = image;
    if (color.channels() == 1) {
        cv::cvtColor(color, color, cv::COLOR_GRAY2BGR);
    }
    if (color.channels() != 3) {
        return cv::Mat();
    }

    cv::Mat color_float;
    const double scale = color.depth() == CV_8U ? 1.0 / 255.0 : 1.0;
    color.convertTo(color_float, CV_32FC3, scale);
    return color_float;
}

float localMedian(std::vector<float>& values)
{
    const std::size_t middle = values.size() / 2;
    std::nth_element(
        values.begin(), values.begin() + middle, values.end());
    float median = values[middle];
    if (values.size() % 2 == 0) {
        const auto lower =
            std::max_element(values.begin(), values.begin() + middle);
        median = 0.5f * (median + *lower);
    }
    return median;
}

} // namespace

void VoxelMapper::densifyMonocularFromRenderedDepth(
    const std::shared_ptr<VoxelKeyframe>& pkf)
{
    if (!monocular_rendered_depth_densify_ ||
        sensor_type_ != MONOCULAR || !pkf || !voxel_model_ ||
        voxel_model_->numVoxels() <= 0 || pkf->img_undist_.empty() ||
        pkf->intr_.size() < 4 || pkf->image_height_ <= 0 ||
        pkf->image_width_ <= 0) {
        return;
    }
    auto densification_profile =
        profileLaptopModule("monocular_rendered_depth_densification");

    const int stride =
        std::max(1, monocular_rendered_depth_pixel_stride_);
    int render_height = 0;
    int render_width = 0;
    sv::MiniCam render_camera =
        scaledRenderCamera(
            pkf, stride, render_height, render_width);

    std::unordered_map<std::string, torch::Tensor> render_pkg;
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        torch::NoGradGuard no_grad;
        render_pkg = voxel_model_->render(
            render_camera,
            render_height,
            render_width,
            torch::Tensor(),
            "dontcare",
            /*track_max_w=*/true,
            std::nullopt,
            /*output_depth=*/true,
            /*output_normal=*/false,
            /*output_T=*/true,
            /*rand_bg=*/false,
            /*use_auto_exposure=*/false,
            sv::RenderOpts{});

        auto max_weight_it = render_pkg.find("max_w");
        if (max_weight_it != render_pkg.end() &&
            max_weight_it->second.defined() &&
            max_weight_it->second.numel() ==
                voxel_model_->numVoxels()) {
            // Orbeez-SLAM counts a sample only when its compositing weight
            // alpha*T exceeds 0.1. max_w is that same quantity per voxel.
            voxel_model_->accumulateMonocularRenderObservation(
                max_weight_it->second,
                monocular_rendered_depth_min_weight_);
        }
    }

    torch::Tensor depth_cpu;
    torch::Tensor alpha_cpu;
    torch::Tensor n_contrib_cpu;
    if (!voxel_utils::renderPkgToDepthAlphaMaps(
            render_pkg,
            render_height,
            render_width,
            depth_cpu,
            alpha_cpu,
            n_contrib_cpu)) {
        validateMonocularRenderedDepthVoxels();
        return;
    }

    const cv::Mat color_float = floatColorImage(pkf->img_undist_);
    if (color_float.empty()) {
        validateMonocularRenderedDepthVoxels();
        return;
    }

    auto depth = depth_cpu.accessor<float, 2>();
    auto alpha = alpha_cpu.accessor<float, 2>();
    auto n_contrib = n_contrib_cpu.accessor<int, 2>();
    std::vector<uint8_t> reliable(
        static_cast<std::size_t>(render_height) *
            static_cast<std::size_t>(render_width),
        0);
    for (int y = 0; y < render_height; ++y) {
        for (int x = 0; x < render_width; ++x) {
            const float z = depth[y][x];
            const bool valid =
                n_contrib[y][x] > 0 &&
                alpha[y][x] > kReliableRenderedAlpha &&
                std::isfinite(z) &&
                z > z_near_ && z < z_far_;
            reliable[
                static_cast<std::size_t>(y) *
                    static_cast<std::size_t>(render_width) +
                static_cast<std::size_t>(x)] =
                valid ? 1 : 0;
        }
    }

    std::vector<float> points;
    std::vector<float> colors;
    points.reserve(
        static_cast<std::size_t>(render_height) *
        static_cast<std::size_t>(render_width));
    colors.reserve(points.capacity());

    const float fx = pkf->intr_[0];
    const float fy = pkf->intr_[1];
    const float cx = pkf->intr_[2];
    const float cy = pkf->intr_[3];
    const Sophus::SE3f world_from_camera =
        pkf->getPosef().inverse();

    int64_t local_hole_candidates = 0;
    for (int render_y = 0; render_y < render_height; ++render_y) {
        for (int render_x = 0; render_x < render_width; ++render_x) {
            // Only an empty rasterization pixel is a structural hole. A
            // low-opacity contributor remains an optimization problem.
            if (n_contrib[render_y][render_x] > 0) {
                continue;
            }

            std::vector<float> neighboring_depths;
            neighboring_depths.reserve(8);
            for (int dy = -1; dy <= 1; ++dy) {
                const int y = render_y + dy;
                if (y < 0 || y >= render_height) {
                    continue;
                }
                for (int dx = -1; dx <= 1; ++dx) {
                    const int x = render_x + dx;
                    if ((dx == 0 && dy == 0) ||
                        x < 0 || x >= render_width) {
                        continue;
                    }
                    const std::size_t index =
                        static_cast<std::size_t>(y) *
                            static_cast<std::size_t>(render_width) +
                        static_cast<std::size_t>(x);
                    if (reliable[index] != 0) {
                        neighboring_depths.push_back(depth[y][x]);
                    }
                }
            }
            if (neighboring_depths.size() <
                kMinimumLocalDepthNeighbors) {
                continue;
            }

            // MonoGS uses rendered pseudo-depth for monocular densification.
            // We restrict it to a local 3x3 median and deliberately provide no
            // global-median fallback at completely unobserved pixels.
            const float z = localMedian(neighboring_depths);
            if (!std::isfinite(z) || z <= z_near_ || z >= z_far_) {
                continue;
            }

            const int pixel_x = std::min(
                render_x * stride, pkf->image_width_ - 1);
            const int pixel_y = std::min(
                render_y * stride, pkf->image_height_ - 1);
            const Eigen::Vector3f camera_point(
                (static_cast<float>(pixel_x) - cx) * z / fx,
                (static_cast<float>(pixel_y) - cy) * z / fy,
                z);
            const Eigen::Vector3f world_point =
                world_from_camera * camera_point;
            if (!world_point.allFinite()) {
                continue;
            }

            const cv::Vec3f color =
                color_float.at<cv::Vec3f>(pixel_y, pixel_x);
            for (int axis = 0; axis < 3; ++axis) {
                points.push_back(world_point[axis]);
                colors.push_back(
                    std::clamp(color[axis], 0.0f, 1.0f));
            }
            ++local_hole_candidates;
        }
    }

    int64_t inserted = 0;
    if (!points.empty()) {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        torch::NoGradGuard no_grad;
        voxel_model_->setNextRealInsertionRerunEntityPath(
            "world/monocular_rendered_depth/created");
        voxel_model_->increasePcd(
            std::move(points),
            std::move(colors),
            getIteration(),
            {pkf->toMiniCam(
                pkf->image_height_, pkf->image_width_)});
        voxel_model_->setNextRealInsertionRerunEntityPath("");
        inserted = voxel_model_->lastIncreasePcdStats().new_voxels;
        if (inserted > 0 &&
            (rerun_params_.run_whole_run_ ||
             rerun_params_.rerun_svrecon_debug_)) {
            rerun_state_.whole_run_live_voxels_dirty_ = true;
        }
    }

    validateMonocularRenderedDepthVoxels();
    if (inserted > 0) {
        std::cout
            << "[MONO/rendered_depth] kf=" << pkf->fid_
            << " local_holes=" << local_hole_candidates
            << " inserted=" << inserted
            << "\n";
    }
}

void VoxelMapper::validateMonocularRenderedDepthVoxels(
    const bool final_pass)
{
    if (!monocular_rendered_depth_densify_ ||
        sensor_type_ != MONOCULAR || !voxel_model_) {
        return;
    }
    auto validation_profile = profileLaptopModule(
        final_pass
            ? "monocular_covisibility_final_pruning"
            : "monocular_covisibility_validation");

    std::unique_lock<std::mutex> lock_render(mutex_render_);
    torch::NoGradGuard no_grad;

    const int64_t count = voxel_model_->numVoxels();
    if (count <= 0) {
        return;
    }
    torch::Tensor provisional =
        voxel_model_->monocularProvisionalVoxelMask();
    torch::Tensor support =
        voxel_model_->monocularRenderSupportHits();
    torch::Tensor opportunities =
        voxel_model_->monocularRenderOpportunities();
    if (!provisional.defined() || !support.defined() ||
        !opportunities.defined() ||
        provisional.numel() != count ||
        support.numel() != count ||
        opportunities.numel() != count) {
        return;
    }

    provisional =
        provisional.to(device_type_).to(torch::kBool).reshape({count});
    support =
        support.to(device_type_).to(torch::kInt32).reshape({count});
    opportunities =
        opportunities.to(device_type_).to(torch::kInt32).reshape({count});
    torch::Tensor ready = final_pass
        ? provisional
        : provisional &
              (opportunities >=
               monocular_rendered_depth_window_size_);
    if (!ready.any().item<bool>()) {
        return;
    }

    torch::Tensor stable =
        ready &
        (support >= monocular_rendered_depth_min_views_);
    torch::Tensor rejected = ready & (~stable);
    const int64_t ready_count = ready.sum().item<int64_t>();
    const int64_t stable_count = stable.sum().item<int64_t>();
    const int64_t rejected_count = rejected.sum().item<int64_t>();

    // Clear provisional state before pruning so rejected cells can pass
    // through the regular topology remapping path.
    voxel_model_->resolveMonocularProvisionalVoxels(ready);
    if (rejected_count > 0) {
        logMonocularCoVisibilityPrunedVoxels(rejected);
        voxel_model_->pruning(rejected);
        if (rerun_params_.run_whole_run_ ||
            rerun_params_.rerun_svrecon_debug_) {
            rerun_state_.whole_run_live_voxels_dirty_ = true;
        }
    }

    std::cout
        << "[MONO/covisibility] final=" << (final_pass ? 1 : 0)
        << " evaluated=" << ready_count
        << " stable=" << stable_count
        << " pruned=" << rejected_count
        << "\n";
}

void VoxelMapper::logMonocularCoVisibilityPrunedVoxels(
    const torch::Tensor& prune_mask)
{
    if (!rerun_params_.enable_rerun_ ||
        (!rerun_params_.run_whole_run_ &&
         !rerun_params_.rerun_svrecon_debug_) ||
        !voxel_model_ || !prune_mask.defined()) {
        return;
    }

    const int64_t count = voxel_model_->numVoxels();
    if (prune_mask.numel() != count) {
        return;
    }
    torch::Tensor indices =
        prune_mask.to(torch::kBool).reshape({count})
            .nonzero().squeeze(1);
    if (!indices.defined() || indices.numel() <= 0) {
        return;
    }

    torch::Tensor centers = voxel_model_->voxCenter();
    torch::Tensor sizes = voxel_model_->voxSize();
    torch::Tensor levels = voxel_model_->octLevel();
    torch::Tensor colors;
    torch::Tensor sh0 = voxel_model_->sh0();
    if (sh0.defined() && sh0.dim() == 2 &&
        sh0.size(0) == count) {
        colors =
            (sh0.index_select(
                 0, indices.to(sh0.device()).to(torch::kLong)) *
                 sv::kSHC0 +
             0.5f)
                .clamp(0.0f, 1.0f)
                .contiguous();
    }

    torch::Tensor selected_indices =
        indices.to(centers.device()).to(torch::kLong);
    const int64_t selected_count = selected_indices.numel();
    torch::Tensor cause = torch::ones(
        {selected_count},
        torch::TensorOptions()
            .dtype(torch::kBool)
            .device(centers.device()));
    appendWholeRunPrunedVoxels(
        getIteration(),
        centers.index_select(0, selected_indices).contiguous(),
        sizes.index_select(
            0, selected_indices.to(sizes.device())).contiguous(),
        levels.index_select(
            0, selected_indices.to(levels.device())).contiguous(),
        colors,
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        cause);
}
