#include "include_voxel/voxel_mapper.h"
#include "include_voxel/voxel_mapper_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

constexpr float kReliableRenderedAlpha = 0.95f;
constexpr int kMinimumLocalDepthNeighbors = 2;
constexpr int kMinimumObservedCornersForPromotion = 4;

struct RenderedDepthFrameCellObservation
{
    cv::Vec3f color_sum{0.0f, 0.0f, 0.0f};
    std::uint32_t color_count = 0;
    std::array<double, 8> corner_sdf_sum{};
    std::array<std::uint32_t, 8> corner_sdf_count{};
};

struct RenderedDepthFrameCornerObservation
{
    double sdf_sum = 0.0;
    std::uint32_t count = 0;
};

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
    const float fov_x = sv::focalToFov(camera.fx, render_width);
    const float fov_y = sv::focalToFov(camera.fy, render_height);
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

bool containsKeyframe(
    const std::vector<sv::camera_id_t>& keyframes,
    const sv::camera_id_t keyframe)
{
    return std::find(keyframes.begin(), keyframes.end(), keyframe) !=
           keyframes.end();
}

} // namespace

void VoxelMapper::resetMonocularRenderedDepthEvidenceIfLayoutChanged()
{
    if (!voxel_model_ || voxel_model_->insertionOctreeLevel() <= 0) {
        monocular_rendered_depth_corner_evidence_.clear();
        monocular_rendered_depth_cell_evidence_.clear();
        monocular_rendered_depth_layout_cell_size_ = 0.0f;
        monocular_rendered_depth_layout_grid_dim_ = 0;
        return;
    }

    const torch::Tensor center_cpu = voxel_model_->SceneCenter()
        .detach().to(torch::kCPU).to(torch::kFloat32).reshape({3}).contiguous();
    const float extent = voxel_model_->SceneExtent()
        .detach().to(torch::kCPU).to(torch::kFloat32).item<float>();
    const int level = voxel_model_->insertionOctreeLevel();
    if (!std::isfinite(extent) || extent <= 0.0f || level <= 0 || level >= 30) {
        return;
    }

    const int grid_dim = 1 << level;
    const float cell_size = extent / static_cast<float>(grid_dim);
    const Eigen::Vector3f center(
        center_cpu[0].item<float>(),
        center_cpu[1].item<float>(),
        center_cpu[2].item<float>());
    const Eigen::Vector3f scene_min =
        center - Eigen::Vector3f::Constant(0.5f * extent);
    const float tolerance = std::max(1.0e-6f, 1.0e-4f * cell_size);
    const bool changed =
        monocular_rendered_depth_layout_grid_dim_ != grid_dim ||
        std::abs(monocular_rendered_depth_layout_cell_size_ - cell_size) >
            tolerance ||
        (monocular_rendered_depth_layout_scene_min_ - scene_min)
                .cwiseAbs().maxCoeff() > tolerance;
    if (changed) {
        monocular_rendered_depth_corner_evidence_.clear();
        monocular_rendered_depth_cell_evidence_.clear();
        monocular_rendered_depth_layout_scene_min_ = scene_min;
        monocular_rendered_depth_layout_cell_size_ = cell_size;
        monocular_rendered_depth_layout_grid_dim_ = grid_dim;
    }
}

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
        profileLaptopModule("monocular_rendered_depth_evidence_fusion");

    resetMonocularRenderedDepthEvidenceIfLayoutChanged();
    if (!(monocular_rendered_depth_layout_cell_size_ > 0.0f) ||
        monocular_rendered_depth_layout_grid_dim_ <= 0) {
        return;
    }

    const int stride =
        std::max(1, monocular_rendered_depth_pixel_stride_);
    int render_height = 0;
    int render_width = 0;
    const sv::MiniCam render_camera =
        scaledRenderCamera(pkf, stride, render_height, render_width);

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
            /*track_max_w=*/false,
            std::nullopt,
            /*output_depth=*/true,
            /*output_normal=*/false,
            /*output_T=*/true,
            /*rand_bg=*/false,
            /*use_auto_exposure=*/false,
            sv::RenderOpts{});
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
        return;
    }

    const cv::Mat color_float = floatColorImage(pkf->img_undist_);
    if (color_float.empty()) {
        return;
    }

    auto depth = depth_cpu.accessor<float, 2>();
    auto alpha = alpha_cpu.accessor<float, 2>();
    auto n_contrib = n_contrib_cpu.accessor<int, 2>();
    std::vector<std::uint8_t> reliable(
        static_cast<std::size_t>(render_height) *
            static_cast<std::size_t>(render_width),
        0);
    for (int y = 0; y < render_height; ++y) {
        for (int x = 0; x < render_width; ++x) {
            const float z = depth[y][x];
            const bool valid =
                n_contrib[y][x] > 0 &&
                alpha[y][x] > kReliableRenderedAlpha &&
                std::isfinite(z) && z > z_near_ && z < z_far_;
            reliable[
                static_cast<std::size_t>(y) *
                    static_cast<std::size_t>(render_width) +
                static_cast<std::size_t>(x)] = valid ? 1 : 0;
        }
    }

    const float fx = pkf->intr_[0];
    const float fy = pkf->intr_[1];
    const float cx = pkf->intr_[2];
    const float cy = pkf->intr_[3];
    if (!(fx > 0.0f) || !(fy > 0.0f)) {
        return;
    }
    const Sophus::SE3f world_from_camera = pkf->getPosef().inverse();
    const Sophus::SE3f camera_from_world = pkf->getPosef();
    const float cell_size = monocular_rendered_depth_layout_cell_size_;
    const float truncation = std::max(
        cell_size,
        monocular_rendered_depth_evidence_trunc_vox_ * cell_size);
    const float half_sampling_band = 0.5f * truncation;
    const int evidence_samples =
        std::max(2, monocular_rendered_depth_evidence_samples_);

    std::unordered_map<
        sv::RgbdTsdfGridKey,
        RenderedDepthFrameCellObservation,
        sv::RgbdTsdfGridKeyHash> frame_cells;
    int64_t structural_holes = 0;
    int64_t sampled_holes = 0;
    int64_t generated_samples = 0;
    for (int render_y = 0; render_y < render_height; ++render_y) {
        for (int render_x = 0; render_x < render_width; ++render_x) {
            if (n_contrib[render_y][render_x] > 0) {
                continue;
            }
            ++structural_holes;

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

            const float depth_hypothesis = localMedian(neighboring_depths);
            if (!std::isfinite(depth_hypothesis) ||
                depth_hypothesis <= z_near_ ||
                depth_hypothesis >= z_far_) {
                continue;
            }

            const int pixel_x = std::min(
                render_x * stride, pkf->image_width_ - 1);
            const int pixel_y = std::min(
                render_y * stride, pkf->image_height_ - 1);
            const float x_normalized =
                (static_cast<float>(pixel_x) - cx) / fx;
            const float y_normalized =
                (static_cast<float>(pixel_y) - cy) / fy;
            const cv::Vec3f color =
                color_float.at<cv::Vec3f>(pixel_y, pixel_x);

            std::unordered_set<
                sv::RgbdTsdfGridKey,
                sv::RgbdTsdfGridKeyHash> gap_cells;
            for (int sample_index = 0;
                 sample_index < evidence_samples;
                 ++sample_index) {
                const float fraction = static_cast<float>(sample_index) /
                    static_cast<float>(evidence_samples - 1);
                const float sample_depth =
                    depth_hypothesis - half_sampling_band +
                    2.0f * half_sampling_band * fraction;
                if (!std::isfinite(sample_depth) ||
                    sample_depth <= z_near_ || sample_depth >= z_far_) {
                    continue;
                }
                ++generated_samples;

                const Eigen::Vector3f camera_point(
                    x_normalized * sample_depth,
                    y_normalized * sample_depth,
                    sample_depth);
                const Eigen::Vector3f world_point =
                    world_from_camera * camera_point;
                if (!world_point.allFinite()) {
                    continue;
                }
                const Eigen::Vector3i index =
                    ((world_point -
                      monocular_rendered_depth_layout_scene_min_) /
                     cell_size)
                        .array()
                        .floor()
                        .cast<int>();
                if ((index.array() < 0).any() ||
                    (index.array() >=
                     monocular_rendered_depth_layout_grid_dim_).any()) {
                    continue;
                }
                gap_cells.insert({index.x(), index.y(), index.z()});
            }
            if (gap_cells.empty()) {
                continue;
            }
            ++sampled_holes;

            for (const auto& key : gap_cells) {
                RenderedDepthFrameCellObservation& observation =
                    frame_cells[key];
                observation.color_sum += color;
                ++observation.color_count;

                const auto corner_keys =
                    sv::rgbdTsdfCellCornerKeys(key);
                for (std::size_t corner_index = 0;
                     corner_index < corner_keys.size();
                     ++corner_index) {
                    const auto& corner_key = corner_keys[corner_index];
                    const Eigen::Vector3f corner_world =
                        monocular_rendered_depth_layout_scene_min_ +
                        cell_size * Eigen::Vector3f(
                            static_cast<float>(corner_key.x),
                            static_cast<float>(corner_key.y),
                            static_cast<float>(corner_key.z));
                    const Eigen::Vector3f corner_camera =
                        camera_from_world * corner_world;
                    if (!corner_camera.allFinite() ||
                        corner_camera.z() <= 1.0e-6f) {
                        continue;
                    }
                    const float signed_distance = std::clamp(
                        depth_hypothesis - corner_camera.z(),
                        -truncation,
                        truncation);
                    observation.corner_sdf_sum[corner_index] +=
                        static_cast<double>(signed_distance);
                    ++observation.corner_sdf_count[corner_index];
                }
            }
        }
    }
    if (frame_cells.empty()) {
        return;
    }

    std::vector<sv::RgbdTsdfGridKey> frame_keys;
    std::vector<float> center_values;
    frame_keys.reserve(frame_cells.size());
    center_values.reserve(frame_cells.size() * 3);
    for (const auto& item : frame_cells) {
        const auto& key = item.first;
        const Eigen::Vector3f center =
            monocular_rendered_depth_layout_scene_min_ +
            cell_size * Eigen::Vector3f(
                static_cast<float>(key.x) + 0.5f,
                static_cast<float>(key.y) + 0.5f,
                static_cast<float>(key.z) + 0.5f);
        frame_keys.push_back(key);
        center_values.insert(
            center_values.end(), {center.x(), center.y(), center.z()});
    }
    torch::Tensor centers = torch::from_blob(
        center_values.data(),
        {static_cast<int64_t>(frame_keys.size()), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
        .clone().to(device_type_).contiguous();
    torch::Tensor active;
    {
        torch::NoGradGuard no_grad;
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        active = std::get<1>(voxel_model_->querySdfTrilinear(centers))
                     .detach().to(torch::kCPU).to(torch::kBool).contiguous();
    }
    const bool* active_ptr = active.data_ptr<bool>();
    for (std::size_t index = 0; index < frame_keys.size(); ++index) {
        if (active_ptr[index]) {
            frame_cells.erase(frame_keys[index]);
            monocular_rendered_depth_cell_evidence_.erase(frame_keys[index]);
        }
    }
    if (frame_cells.empty()) {
        return;
    }

    std::unordered_map<
        sv::RgbdTsdfGridKey,
        RenderedDepthFrameCornerObservation,
        sv::RgbdTsdfGridKeyHash> frame_corners;
    for (const auto& item : frame_cells) {
        const auto corner_keys = sv::rgbdTsdfCellCornerKeys(item.first);
        for (std::size_t corner_index = 0;
             corner_index < corner_keys.size();
             ++corner_index) {
            const std::uint32_t count =
                item.second.corner_sdf_count[corner_index];
            if (count == 0) {
                continue;
            }
            RenderedDepthFrameCornerObservation& observation =
                frame_corners[corner_keys[corner_index]];
            observation.sdf_sum +=
                item.second.corner_sdf_sum[corner_index] /
                static_cast<double>(count);
            ++observation.count;
        }
    }

    std::unordered_set<
        sv::RgbdTsdfGridKey,
        sv::RgbdTsdfGridKeyHash> updated_corners;
    for (const auto& item : frame_corners) {
        if (item.second.count == 0) {
            continue;
        }
        const float sample = static_cast<float>(
            item.second.sdf_sum /
            static_cast<double>(item.second.count));
        if (!std::isfinite(sample)) {
            continue;
        }
        sv::RgbdTsdfCornerEvidence& evidence =
            monocular_rendered_depth_corner_evidence_[item.first];
        const float new_weight = std::min(
            evidence.weight + 1.0f,
            monocular_rendered_depth_evidence_max_weight_);
        const float retained_weight = std::max(0.0f, new_weight - 1.0f);
        evidence.distance = retained_weight > 0.0f
            ? (evidence.distance * retained_weight + sample) / new_weight
            : sample;
        evidence.weight = new_weight;
        updated_corners.insert(item.first);
    }

    const sv::camera_id_t keyframe_id =
        static_cast<sv::camera_id_t>(pkf->fid_);
    for (const auto& item : frame_cells) {
        const auto corners = sv::rgbdTsdfCellCornerKeys(item.first);
        const bool observed = std::any_of(
            corners.begin(), corners.end(), [&](const auto& corner) {
                return updated_corners.find(corner) != updated_corners.end();
            });
        if (!observed) {
            continue;
        }
        sv::RgbdTsdfCellEvidence& evidence =
            monocular_rendered_depth_cell_evidence_[item.first];
        if (!containsKeyframe(evidence.observed_keyframes, keyframe_id)) {
            evidence.observed_keyframes.push_back(keyframe_id);
            if (item.second.color_count > 0) {
                evidence.color_sum +=
                    item.second.color_sum /
                    static_cast<float>(item.second.color_count);
                ++evidence.color_observations;
            }
        }
    }

    std::unordered_set<
        sv::RgbdTsdfGridKey,
        sv::RgbdTsdfGridKeyHash> affected_cells;
    affected_cells.reserve(frame_cells.size() + updated_corners.size() * 8);
    for (const auto& item : frame_cells) {
        if (monocular_rendered_depth_cell_evidence_.find(item.first) !=
            monocular_rendered_depth_cell_evidence_.end()) {
            affected_cells.insert(item.first);
        }
    }
    for (const auto& corner : updated_corners) {
        for (int dz = 0; dz <= 1; ++dz) {
            for (int dy = 0; dy <= 1; ++dy) {
                for (int dx = 0; dx <= 1; ++dx) {
                    const sv::RgbdTsdfGridKey cell{
                        corner.x - dx,
                        corner.y - dy,
                        corner.z - dz};
                    if (cell.x < 0 || cell.y < 0 || cell.z < 0 ||
                        cell.x >= monocular_rendered_depth_layout_grid_dim_ ||
                        cell.y >= monocular_rendered_depth_layout_grid_dim_ ||
                        cell.z >= monocular_rendered_depth_layout_grid_dim_) {
                        continue;
                    }
                    if (monocular_rendered_depth_cell_evidence_.find(cell) !=
                        monocular_rendered_depth_cell_evidence_.end()) {
                        affected_cells.insert(cell);
                    }
                }
            }
        }
    }

    const int64_t promoted =
        promoteMonocularRenderedDepthEvidenceCells(affected_cells);
    std::cout
        << "[MONO/rendered_evidence] kf=" << pkf->fid_
        << " structural_holes=" << structural_holes
        << " sampled_holes=" << sampled_holes
        << " ray_samples=" << generated_samples
        << " frame_cells=" << frame_cells.size()
        << " pending_cells="
        << monocular_rendered_depth_cell_evidence_.size()
        << " promoted=" << promoted
        << "\n";
}

void VoxelMapper::logMonocularRenderedDepthEvidenceCellsToRerun(
    const int iteration,
    const std::vector<sv::RgbdTsdfGridKey>& cells,
    const std::string& entity_path)
{
    if (!rerun_params_.enable_rerun_ ||
        !rerun_params_.rerun_svrecon_debug_ || !voxel_model_ ||
        !(monocular_rendered_depth_layout_cell_size_ > 0.0f)) {
        return;
    }

    std::vector<std::int32_t> indices;
    std::vector<float> colors;
    indices.reserve(cells.size() * 3);
    colors.reserve(cells.size() * 4);
    for (const auto& key : cells) {
        indices.insert(indices.end(), {key.x, key.y, key.z});
        cv::Vec3f color(0.75f, 0.75f, 0.75f);
        const auto evidence =
            monocular_rendered_depth_cell_evidence_.find(key);
        if (evidence != monocular_rendered_depth_cell_evidence_.end() &&
            evidence->second.color_observations > 0) {
            color = evidence->second.color_sum /
                static_cast<float>(evidence->second.color_observations);
        }
        colors.insert(colors.end(), {
            std::clamp(color[0], 0.0f, 1.0f),
            std::clamp(color[1], 0.0f, 1.0f),
            std::clamp(color[2], 0.0f, 1.0f),
            0.8f});
    }

    sv::RerunVisualizerBridge::instance().visualizeDebugVoxelGridIndices(
        "svrecon_debug",
        indices,
        colors,
        monocular_rendered_depth_layout_scene_min_,
        monocular_rendered_depth_layout_cell_size_,
        static_cast<std::int32_t>(voxel_model_->insertionOctreeLevel()),
        iteration,
        entity_path,
        0.8f);
}

int64_t VoxelMapper::promoteMonocularRenderedDepthEvidenceCells(
    const std::unordered_set<
        sv::RgbdTsdfGridKey,
        sv::RgbdTsdfGridKeyHash>& affected_cells)
{
    if (!monocular_rendered_depth_densify_ || !voxel_model_ ||
        monocular_rendered_depth_cell_evidence_.empty() ||
        affected_cells.empty()) {
        return 0;
    }
    auto promotion_profile =
        profileLaptopModule("monocular_rendered_depth_evidence_promotion");

    const int iteration = getIteration();
    const bool log_evidence_snapshot =
        rerun_params_.enable_rerun_ &&
        rerun_params_.rerun_svrecon_debug_;
    std::vector<sv::RgbdTsdfGridKey> cells_to_evaluate;
    if (log_evidence_snapshot) {
        cells_to_evaluate.reserve(
            monocular_rendered_depth_cell_evidence_.size());
        for (const auto& item :
             monocular_rendered_depth_cell_evidence_) {
            cells_to_evaluate.push_back(item.first);
        }
    } else {
        cells_to_evaluate.reserve(affected_cells.size());
        cells_to_evaluate.insert(
            cells_to_evaluate.end(),
            affected_cells.begin(),
            affected_cells.end());
    }

    std::vector<sv::RgbdTsdfGridKey> promoted_cells;
    std::vector<sv::RgbdTsdfGridKey> incomplete_corner_cells;
    std::vector<sv::RgbdTsdfGridKey> waiting_view_cells;
    std::vector<sv::RgbdTsdfGridKey> waiting_baseline_cells;
    std::vector<sv::RgbdTsdfGridKey> zero_crossing_cells;
    std::vector<sv::RgbdTsdfGridKey> rejected_non_surface_cells;
    std::vector<sv::RgbdTsdfGridKey> affected_rejected_cells;
    std::vector<float> centers;
    std::vector<float> colors;
    std::unordered_set<
        sv::RgbdTsdfGridKey,
        sv::RgbdTsdfGridKeyHash> direct_corner_keys;
    std::vector<float> direct_corner_points;
    std::vector<float> direct_corner_values;

    for (const auto& key : cells_to_evaluate) {
        const auto cell_item =
            monocular_rendered_depth_cell_evidence_.find(key);
        if (cell_item == monocular_rendered_depth_cell_evidence_.end()) {
            continue;
        }
        const bool affected =
            affected_cells.find(key) != affected_cells.end();
        const auto corner_keys = sv::rgbdTsdfCellCornerKeys(key);
        int observed_corner_count = 0;
        bool has_positive = false;
        bool has_negative = false;
        for (const auto& corner_key : corner_keys) {
            const auto corner =
                monocular_rendered_depth_corner_evidence_.find(corner_key);
            if (corner == monocular_rendered_depth_corner_evidence_.end() ||
                corner->second.weight <
                    monocular_rendered_depth_evidence_promote_min_weight_ ||
                !std::isfinite(corner->second.distance)) {
                continue;
            }
            ++observed_corner_count;
            has_positive = has_positive || corner->second.distance > 0.0f;
            has_negative = has_negative || corner->second.distance < 0.0f;
        }
        if (observed_corner_count <
            kMinimumObservedCornersForPromotion) {
            if (log_evidence_snapshot) {
                incomplete_corner_cells.push_back(key);
            }
            continue;
        }
        if (static_cast<int>(
                cell_item->second.observed_keyframes.size()) <
            monocular_rendered_depth_evidence_promote_min_views_) {
            if (log_evidence_snapshot) {
                waiting_view_cells.push_back(key);
            }
            continue;
        }

        const Eigen::Vector3f center =
            monocular_rendered_depth_layout_scene_min_ +
            monocular_rendered_depth_layout_cell_size_ * Eigen::Vector3f(
                static_cast<float>(key.x) + 0.5f,
                static_cast<float>(key.y) + 0.5f,
                static_cast<float>(key.z) + 0.5f);
        std::vector<Eigen::Vector3f> camera_centers;
        camera_centers.reserve(
            cell_item->second.observed_keyframes.size());
        for (const sv::camera_id_t keyframe_id :
             cell_item->second.observed_keyframes) {
            const std::shared_ptr<VoxelKeyframe> keyframe =
                scene_->getKeyframe(
                    static_cast<std::size_t>(keyframe_id));
            if (!keyframe) {
                continue;
            }
            camera_centers.push_back(
                keyframe->getPosef().inverse().translation());
        }
        bool sufficient_baseline = false;
        for (std::size_t first = 0;
             first < camera_centers.size() && !sufficient_baseline;
             ++first) {
            for (std::size_t second = first + 1;
                 second < camera_centers.size();
                 ++second) {
                const float baseline =
                    (camera_centers[first] - camera_centers[second]).norm();
                const float mean_depth = 0.5f *
                    ((center - camera_centers[first]).norm() +
                     (center - camera_centers[second]).norm());
                if (std::isfinite(baseline) &&
                    std::isfinite(mean_depth) && mean_depth > 1.0e-6f &&
                    baseline >=
                        monocular_rendered_depth_evidence_min_baseline_ratio_ *
                            mean_depth) {
                    sufficient_baseline = true;
                    break;
                }
            }
        }
        if (!sufficient_baseline) {
            if (log_evidence_snapshot) {
                waiting_baseline_cells.push_back(key);
            }
            continue;
        }
        if (!has_positive || !has_negative) {
            if (log_evidence_snapshot) {
                rejected_non_surface_cells.push_back(key);
            }
            if (affected) {
                affected_rejected_cells.push_back(key);
            }
            continue;
        }
        if (log_evidence_snapshot) {
            zero_crossing_cells.push_back(key);
        }
        if (!affected) {
            continue;
        }

        centers.insert(
            centers.end(), {center.x(), center.y(), center.z()});
        const cv::Vec3f color =
            cell_item->second.color_observations > 0
            ? cell_item->second.color_sum /
                static_cast<float>(cell_item->second.color_observations)
            : cv::Vec3f(0.5f, 0.5f, 0.5f);
        colors.insert(colors.end(), {
            std::clamp(color[0], 0.0f, 1.0f),
            std::clamp(color[1], 0.0f, 1.0f),
            std::clamp(color[2], 0.0f, 1.0f)});
        promoted_cells.push_back(key);

        for (const auto& corner_key : corner_keys) {
            const auto corner =
                monocular_rendered_depth_corner_evidence_.find(corner_key);
            if (corner == monocular_rendered_depth_corner_evidence_.end() ||
                corner->second.weight <
                    monocular_rendered_depth_evidence_promote_min_weight_ ||
                !std::isfinite(corner->second.distance) ||
                !direct_corner_keys.insert(corner_key).second) {
                continue;
            }
            const Eigen::Vector3f world =
                monocular_rendered_depth_layout_scene_min_ +
                monocular_rendered_depth_layout_cell_size_ *
                    Eigen::Vector3f(
                        static_cast<float>(corner_key.x),
                        static_cast<float>(corner_key.y),
                        static_cast<float>(corner_key.z));
            direct_corner_points.insert(
                direct_corner_points.end(),
                {world.x(), world.y(), world.z()});
            direct_corner_values.push_back(corner->second.distance);
        }
    }

    if (log_evidence_snapshot) {
        logMonocularRenderedDepthEvidenceCellsToRerun(
            iteration,
            incomplete_corner_cells,
            "world/monocular_rendered_depth_evidence/incomplete_corners");
        logMonocularRenderedDepthEvidenceCellsToRerun(
            iteration,
            waiting_view_cells,
            "world/monocular_rendered_depth_evidence/waiting_views");
        logMonocularRenderedDepthEvidenceCellsToRerun(
            iteration,
            waiting_baseline_cells,
            "world/monocular_rendered_depth_evidence/waiting_baseline");
        logMonocularRenderedDepthEvidenceCellsToRerun(
            iteration,
            zero_crossing_cells,
            "world/monocular_rendered_depth_evidence/zero_crossing");
        logMonocularRenderedDepthEvidenceCellsToRerun(
            iteration,
            rejected_non_surface_cells,
            "world/monocular_rendered_depth_evidence/rejected_non_surface");
    }
    for (const auto& key : affected_rejected_cells) {
        monocular_rendered_depth_cell_evidence_.erase(key);
    }
    if (promoted_cells.empty()) {
        if (log_evidence_snapshot) {
            logMonocularRenderedDepthEvidenceCellsToRerun(
                iteration,
                {},
                "world/monocular_rendered_depth_evidence/promoted");
        }
        return 0;
    }

    torch::Tensor center_tensor = torch::from_blob(
        centers.data(),
        {static_cast<int64_t>(promoted_cells.size()), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
        .clone();
    torch::Tensor color_tensor = torch::from_blob(
        colors.data(),
        {static_cast<int64_t>(promoted_cells.size()), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
        .clone();
    torch::Tensor corner_point_tensor = torch::from_blob(
        direct_corner_points.data(),
        {static_cast<int64_t>(direct_corner_values.size()), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
        .clone();
    torch::Tensor corner_value_tensor = torch::from_blob(
        direct_corner_values.data(),
        {static_cast<int64_t>(direct_corner_values.size()), 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
        .clone();

    sv::VoxelModel::IncreasePcdStats stats;
    {
        torch::NoGradGuard no_grad;
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        voxel_model_->setNextSdfInitializationGridSamples(
            corner_point_tensor, corner_value_tensor);
        voxel_model_->setNextRealInsertionRerunEntityPath(
            "world/monocular_rendered_depth_evidence/promoted");
        voxel_model_->increasePcd(
            center_tensor,
            color_tensor,
            getIteration(),
            incrementalMappingCameras());
        voxel_model_->setNextRealInsertionRerunEntityPath("");
        stats = voxel_model_->lastIncreasePcdStats();
    }

    torch::Tensor active_after_insertion;
    {
        torch::NoGradGuard no_grad;
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        active_after_insertion =
            std::get<1>(voxel_model_->querySdfTrilinear(
                center_tensor.to(device_type_).contiguous()))
                .detach().to(torch::kCPU).to(torch::kBool).contiguous();
    }
    const bool* active_after_ptr = active_after_insertion.data_ptr<bool>();
    std::vector<sv::RgbdTsdfGridKey> activated_cells;
    activated_cells.reserve(promoted_cells.size());
    for (std::size_t index = 0; index < promoted_cells.size(); ++index) {
        if (active_after_ptr[index]) {
            activated_cells.push_back(promoted_cells[index]);
        }
    }
    if (log_evidence_snapshot) {
        logMonocularRenderedDepthEvidenceCellsToRerun(
            iteration,
            activated_cells,
            "world/monocular_rendered_depth_evidence/promoted");
    }
    for (const auto& key : activated_cells) {
        monocular_rendered_depth_cell_evidence_.erase(key);
    }
    if (stats.new_voxels > 0 &&
        (rerun_params_.run_whole_run_ ||
         rerun_params_.rerun_svrecon_debug_)) {
        rerun_state_.whole_run_live_voxels_dirty_ = true;
    }
    return stats.new_voxels;
}
