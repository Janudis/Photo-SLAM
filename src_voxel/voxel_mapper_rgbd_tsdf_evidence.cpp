#include "include_voxel/voxel_mapper.h"
#include "include_voxel/voxel_mapper_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace {

constexpr int kMinObservedCornersForPromotion = 4;

struct FrameCellObservation {
    cv::Vec3f color_sum{0.0f, 0.0f, 0.0f};
    std::uint32_t color_count = 0;
};

void raycastGridSegment(
    const Eigen::Vector3f& start_world,
    const Eigen::Vector3f& end_world,
    const Eigen::Vector3f& scene_min,
    const float cell_size,
    const int grid_dim,
    std::vector<sv::RgbdTsdfGridKey>& cells)
{
    if (!(cell_size > 0.0f) || grid_dim <= 0) {
        return;
    }

    const Eigen::Vector3f start = (start_world - scene_min) / cell_size;
    const Eigen::Vector3f end = (end_world - scene_min) / cell_size;
    const Eigen::Vector3f direction = end - start;
    Eigen::Vector3i current = start.array().floor().cast<int>();
    const Eigen::Vector3i target = end.array().floor().cast<int>();
    Eigen::Vector3i step = Eigen::Vector3i::Zero();
    Eigen::Vector3f t_max = Eigen::Vector3f::Constant(
        std::numeric_limits<float>::infinity());
    Eigen::Vector3f t_delta = Eigen::Vector3f::Constant(
        std::numeric_limits<float>::infinity());

    for (int axis = 0; axis < 3; ++axis) {
        if (std::abs(direction[axis]) <= 1.0e-8f) {
            continue;
        }
        step[axis] = direction[axis] > 0.0f ? 1 : -1;
        const float next_boundary = step[axis] > 0
            ? static_cast<float>(current[axis] + 1)
            : static_cast<float>(current[axis]);
        t_max[axis] = (next_boundary - start[axis]) / direction[axis];
        t_delta[axis] = 1.0f / std::abs(direction[axis]);
    }

    constexpr int kMaxTraversalCells = 256;
    for (int iteration = 0; iteration < kMaxTraversalCells; ++iteration) {
        if ((current.array() >= 0).all() &&
            (current.array() < grid_dim).all()) {
            cells.push_back({current.x(), current.y(), current.z()});
        }
        if (current == target) {
            break;
        }

        int axis = 0;
        if (t_max.y() < t_max.x()) axis = 1;
        if (t_max.z() < t_max[axis]) axis = 2;
        if (!std::isfinite(t_max[axis]) || t_max[axis] > 1.0f + 1.0e-6f) {
            break;
        }
        current[axis] += step[axis];
        t_max[axis] += t_delta[axis];
    }
}

bool containsKeyframe(
    const std::vector<camera_id_t>& keyframes,
    const camera_id_t keyframe)
{
    return std::find(keyframes.begin(), keyframes.end(), keyframe) !=
           keyframes.end();
}

}  // namespace

void VoxelMapper::resetRgbdTsdfEvidenceIfLayoutChanged()
{
    if (!voxel_model_ || voxel_model_->insertionOctreeLevel() <= 0) {
        rgbd_tsdf_corner_evidence_.clear();
        rgbd_tsdf_cell_evidence_.clear();
        rgbd_tsdf_layout_cell_size_ = 0.0f;
        rgbd_tsdf_layout_grid_dim_ = 0;
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
        rgbd_tsdf_layout_grid_dim_ != grid_dim ||
        std::abs(rgbd_tsdf_layout_cell_size_ - cell_size) > tolerance ||
        (rgbd_tsdf_layout_scene_min_ - scene_min).cwiseAbs().maxCoeff() > tolerance;
    if (changed) {
        rgbd_tsdf_corner_evidence_.clear();
        rgbd_tsdf_cell_evidence_.clear();
        rgbd_tsdf_layout_scene_min_ = scene_min;
        rgbd_tsdf_layout_cell_size_ = cell_size;
        rgbd_tsdf_layout_grid_dim_ = grid_dim;
    }
}

void VoxelMapper::integrateRgbdTsdfEvidenceForRenderHoles(
    const std::shared_ptr<VoxelKeyframe>& pkf,
    std::unordered_set<
        sv::RgbdTsdfGridKey,
        sv::RgbdTsdfGridKeyHash>& affected_cells)
{
    if (!rgbd_tsdf_evidence_ || !pkf || !voxel_model_ ||
        sensor_type_ != RGBD || pkf->img_undist_.empty() ||
        pkf->img_auxiliary_undist_.empty()) {
        return;
    }
    auto evidence_profile =
        profileLaptopModule("rgbd_tsdf_evidence_fusion");

    resetRgbdTsdfEvidenceIfLayoutChanged();
    if (rgbd_tsdf_layout_cell_size_ <= 0.0f ||
        rgbd_tsdf_layout_grid_dim_ <= 0) {
        return;
    }

    cv::Mat depth_meters;
    if (!voxel_utils::depthMatToMeters(
            pkf->img_auxiliary_undist_, depth_meters) || depth_meters.empty()) {
        return;
    }
    if (depth_meters.channels() > 1) {
        cv::extractChannel(depth_meters, depth_meters, 0);
    }
    if (depth_meters.type() != CV_32FC1) {
        depth_meters.convertTo(depth_meters, CV_32FC1);
    }

    cv::Mat rgb_float;
    if (pkf->img_undist_.channels() == 3) {
        pkf->img_undist_.convertTo(rgb_float, CV_32FC3);
    } else if (pkf->img_undist_.channels() == 1) {
        cv::Mat gray;
        pkf->img_undist_.convertTo(gray, CV_32FC1);
        cv::cvtColor(gray, rgb_float, cv::COLOR_GRAY2RGB);
    } else {
        return;
    }
    double color_max = 0.0;
    cv::minMaxLoc(rgb_float.reshape(1), nullptr, &color_max);
    if (color_max > 1.5) {
        rgb_float *= 1.0f / 255.0f;
    }

    const int height = std::min(depth_meters.rows, rgb_float.rows);
    const int width = std::min(depth_meters.cols, rgb_float.cols);
    if (height != pkf->image_height_ || width != pkf->image_width_) {
        return;
    }

    torch::Tensor depth_tensor = torch::from_blob(
        depth_meters.data,
        {height, width},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
        .clone().to(device_type_).reshape({-1}).contiguous();
    int64_t valid_depth_pixels = 0;
    int64_t hole_pixels = 0;
    torch::Tensor full_hole_mask;
    torch::Tensor selected_holes = detectRgbdRenderHolePixels(
        pkf,
        depth_tensor,
        rgbd_tsdf_evidence_pixel_stride_,
        /*render_on_stride_grid=*/true,
        valid_depth_pixels,
        hole_pixels,
        full_hole_mask);
    if (!selected_holes.defined() || selected_holes.numel() == 0) {
        return;
    }
    selected_holes = selected_holes.detach().to(torch::kCPU)
                         .to(torch::kBool).reshape({height, width}).contiguous();
    if (!selected_holes.any().item<bool>()) {
        return;
    }

    const sv::MiniCam cam = pkf->toMiniCam(height, width);
    if (cam.fx <= 1.0e-6f || cam.fy <= 1.0e-6f) {
        return;
    }
    const Sophus::SE3f Tcw = pkf->getPosef();
    const Sophus::SE3f Twc = Tcw.inverse();
    const float truncation = std::max(
        rgbd_tsdf_layout_cell_size_,
        rgbd_tsdf_evidence_trunc_vox_ * rgbd_tsdf_layout_cell_size_);
    const float max_depth = RGBD_max_depth_;
    torch::Tensor selected_indices =
        torch::nonzero(selected_holes.reshape({-1}))
            .reshape({-1}).to(torch::kInt64).contiguous();
    const auto selected_indices_acc =
        selected_indices.accessor<int64_t, 1>();

    std::unordered_map<
        sv::RgbdTsdfGridKey,
        FrameCellObservation,
        sv::RgbdTsdfGridKeyHash> frame_cells;
    std::vector<sv::RgbdTsdfGridKey> traversed_cells;
    for (int64_t selected_index = 0;
         selected_index < selected_indices.size(0);
         ++selected_index) {
        const int64_t flat_index = selected_indices_acc[selected_index];
        const int y = static_cast<int>(
            flat_index / static_cast<int64_t>(width));
        const int x = static_cast<int>(
            flat_index % static_cast<int64_t>(width));
        const float y_normalized = (static_cast<float>(y) - cam.cy) / cam.fy;
        const float depth = depth_meters.at<float>(y, x);
        if (!std::isfinite(depth) || depth <= RGBD_min_depth_ ||
            depth >= max_depth) {
            continue;
        }
        const float x_normalized = (static_cast<float>(x) - cam.cx) / cam.fx;
        const float start_depth = std::max(RGBD_min_depth_, depth - truncation);
        const float end_depth = std::min(max_depth, depth + truncation);
        if (end_depth <= start_depth) {
            continue;
        }
        const Eigen::Vector3f start_world = Twc * Eigen::Vector3f(
            x_normalized * start_depth,
            y_normalized * start_depth,
            start_depth);
        const Eigen::Vector3f end_world = Twc * Eigen::Vector3f(
            x_normalized * end_depth,
            y_normalized * end_depth,
            end_depth);
        traversed_cells.clear();
        raycastGridSegment(
            start_world,
            end_world,
            rgbd_tsdf_layout_scene_min_,
            rgbd_tsdf_layout_cell_size_,
            rgbd_tsdf_layout_grid_dim_,
            traversed_cells);
        const cv::Vec3f color = rgb_float.at<cv::Vec3f>(y, x);
        for (const auto& key : traversed_cells) {
            FrameCellObservation& observation = frame_cells[key];
            observation.color_sum += color;
            ++observation.color_count;
        }
    }
    if (frame_cells.empty()) {
        return;
    }

    // Evidence never competes with an already-active octree cell. This also
    // removes stale evidence if ORB/inactive geometry claimed the cell later.
    std::vector<sv::RgbdTsdfGridKey> frame_keys;
    std::vector<float> center_values;
    frame_keys.reserve(frame_cells.size());
    center_values.reserve(frame_cells.size() * 3);
    for (const auto& item : frame_cells) {
        const auto& key = item.first;
        const Eigen::Vector3f center = rgbd_tsdf_layout_scene_min_ +
            rgbd_tsdf_layout_cell_size_ * Eigen::Vector3f(
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
            rgbd_tsdf_cell_evidence_.erase(frame_keys[index]);
        }
    }
    if (frame_cells.empty()) {
        return;
    }

    std::unordered_set<
        sv::RgbdTsdfGridKey,
        sv::RgbdTsdfGridKeyHash> frame_corner_keys;
    for (const auto& item : frame_cells) {
        for (const auto& corner : sv::rgbdTsdfCellCornerKeys(item.first)) {
            frame_corner_keys.insert(corner);
        }
    }

    std::unordered_set<
        sv::RgbdTsdfGridKey,
        sv::RgbdTsdfGridKeyHash> updated_corners;
    for (const auto& key : frame_corner_keys) {
        const Eigen::Vector3f world = rgbd_tsdf_layout_scene_min_ +
            rgbd_tsdf_layout_cell_size_ * Eigen::Vector3f(
                static_cast<float>(key.x),
                static_cast<float>(key.y),
                static_cast<float>(key.z));
        const Eigen::Vector3f camera = Tcw * world;
        if (!std::isfinite(camera.z()) || camera.z() <= 1.0e-6f) {
            continue;
        }
        const int x = static_cast<int>(std::lround(
            cam.fx * camera.x() / camera.z() + cam.cx));
        const int y = static_cast<int>(std::lround(
            cam.fy * camera.y() / camera.z() + cam.cy));
        if (x < 0 || x >= width || y < 0 || y >= height) {
            continue;
        }
        const float measured_depth = depth_meters.at<float>(y, x);
        if (!std::isfinite(measured_depth) ||
            measured_depth <= RGBD_min_depth_ || measured_depth >= max_depth) {
            continue;
        }

        const float measured_sdf = measured_depth - camera.z();
        if (measured_sdf < -truncation) {
            continue;
        }
        const float sample = std::clamp(measured_sdf, -truncation, truncation);
        const float measurement_weight = camera.z() <= 1.0e-6f
            ? 1.0f
            : 1.0f / (camera.z() * camera.z());
        if (!std::isfinite(measurement_weight) || measurement_weight <= 0.0f) {
            continue;
        }

        sv::RgbdTsdfCornerEvidence& evidence = rgbd_tsdf_corner_evidence_[key];
        const float denominator = evidence.weight + measurement_weight;
        evidence.distance = evidence.weight > 0.0f
            ? (evidence.distance * evidence.weight + sample * measurement_weight) /
                  denominator
            : sample;
        evidence.weight = std::min(
            denominator, rgbd_tsdf_evidence_max_weight_);
        updated_corners.insert(key);
    }

    const camera_id_t keyframe_id = static_cast<camera_id_t>(pkf->fid_);
    for (const auto& item : frame_cells) {
        const auto corners = sv::rgbdTsdfCellCornerKeys(item.first);
        const bool observed = std::any_of(
            corners.begin(), corners.end(), [&](const auto& corner) {
                return updated_corners.find(corner) != updated_corners.end();
            });
        if (!observed) {
            continue;
        }
        sv::RgbdTsdfCellEvidence& evidence = rgbd_tsdf_cell_evidence_[item.first];
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
        sv::RgbdTsdfGridKeyHash> frame_affected_cells;
    frame_affected_cells.reserve(
        frame_cells.size() + updated_corners.size() * 8);

    // Recheck cells traversed by this keyframe. Only cells with accumulated
    // evidence are relevant to promotion.
    for (const auto& item : frame_cells) {
        if (rgbd_tsdf_cell_evidence_.find(item.first) !=
            rgbd_tsdf_cell_evidence_.end()) {
            frame_affected_cells.insert(item.first);
        }
    }

    // Grid corners are shared by up to eight cells. Updating one corner changes
    // every existing evidence cell incident to it, even when that neighboring
    // cell was first traversed by an earlier keyframe.
    for (const auto& corner : updated_corners) {
        for (int dz = 0; dz <= 1; ++dz) {
            for (int dy = 0; dy <= 1; ++dy) {
                for (int dx = 0; dx <= 1; ++dx) {
                    const sv::RgbdTsdfGridKey cell{
                        corner.x - dx,
                        corner.y - dy,
                        corner.z - dz};
                    if (cell.x < 0 || cell.y < 0 || cell.z < 0 ||
                        cell.x >= rgbd_tsdf_layout_grid_dim_ ||
                        cell.y >= rgbd_tsdf_layout_grid_dim_ ||
                        cell.z >= rgbd_tsdf_layout_grid_dim_) {
                        continue;
                    }
                    if (rgbd_tsdf_cell_evidence_.find(cell) !=
                        rgbd_tsdf_cell_evidence_.end()) {
                        frame_affected_cells.insert(cell);
                    }
                }
            }
        }
    }

    affected_cells.insert(
        frame_affected_cells.begin(),
        frame_affected_cells.end());
}

void VoxelMapper::logRgbdTsdfEvidenceCellsToRerun(
    const int iteration,
    const std::vector<sv::RgbdTsdfGridKey>& cells,
    const std::string& entity_path)
{
    if (!rerun_params_.enable_rerun_ ||
        !rerun_params_.rerun_svrecon_debug_ ||
        !voxel_model_ ||
        rgbd_tsdf_layout_cell_size_ <= 0.0f) {
        return;
    }

    std::vector<std::int32_t> indices;
    std::vector<float> colors;
    indices.reserve(cells.size() * 3);
    colors.reserve(cells.size() * 4);
    const int level = voxel_model_->insertionOctreeLevel();

    for (const auto& key : cells) {
        indices.insert(indices.end(), {key.x, key.y, key.z});

        cv::Vec3f color(0.75f, 0.75f, 0.75f);
        const auto evidence = rgbd_tsdf_cell_evidence_.find(key);
        if (evidence != rgbd_tsdf_cell_evidence_.end() &&
            evidence->second.color_observations > 0) {
            color =
                evidence->second.color_sum /
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
        rgbd_tsdf_layout_scene_min_,
        rgbd_tsdf_layout_cell_size_,
        static_cast<std::int32_t>(level),
        iteration,
        entity_path,
        0.8f);
}

void VoxelMapper::promoteRgbdTsdfEvidenceCells(
    const std::unordered_set<
        sv::RgbdTsdfGridKey,
        sv::RgbdTsdfGridKeyHash>& affected_cells)
{
    if (!rgbd_tsdf_evidence_ || !voxel_model_ ||
        rgbd_tsdf_cell_evidence_.empty() || affected_cells.empty()) {
        return;
    }
    auto promotion_profile =
        profileLaptopModule("rgbd_tsdf_evidence_promotion");

    std::vector<sv::RgbdTsdfGridKey> promoted_cells;
    std::vector<sv::RgbdTsdfGridKey> rejected_non_surface_cells;
    std::vector<sv::RgbdTsdfGridKey> affected_rejected_non_surface_cells;
    std::vector<sv::RgbdTsdfGridKey> incomplete_corner_cells;
    std::vector<sv::RgbdTsdfGridKey> waiting_view_cells;
    std::vector<sv::RgbdTsdfGridKey> zero_crossing_cells;
    std::vector<float> centers;
    std::vector<float> colors;
    std::unordered_set<
        sv::RgbdTsdfGridKey,
        sv::RgbdTsdfGridKeyHash> direct_corner_keys;
    std::vector<float> direct_corner_points;
    std::vector<float> direct_corner_values;
    const int iteration = getIteration();
    const bool log_evidence_snapshot =
        rerun_params_.enable_rerun_ &&
        rerun_params_.rerun_svrecon_debug_;

    std::vector<sv::RgbdTsdfGridKey> cells_to_evaluate;
    if (log_evidence_snapshot) {
        // Complete Rerun snapshots intentionally retain the diagnostic full
        // scan. Normal mapping evaluates only cells changed by this keyframe.
        cells_to_evaluate.reserve(rgbd_tsdf_cell_evidence_.size());
        for (const auto& item : rgbd_tsdf_cell_evidence_) {
            cells_to_evaluate.push_back(item.first);
        }
    } else {
        cells_to_evaluate.reserve(affected_cells.size());
        for (const auto& key : affected_cells) {
            cells_to_evaluate.push_back(key);
        }
    }

    for (const auto& key : cells_to_evaluate) {
        const auto item = rgbd_tsdf_cell_evidence_.find(key);
        if (item == rgbd_tsdf_cell_evidence_.end()) {
            continue;
        }
        const bool affected =
            affected_cells.find(key) != affected_cells.end();
        const sv::RgbdTsdfCellEvidence& cell_evidence = item->second;
        const auto corner_keys = sv::rgbdTsdfCellCornerKeys(key);
        bool has_positive = false;
        bool has_negative = false;
        int observed_corner_count = 0;
        for (const auto& corner_key : corner_keys) {
            const auto corner = rgbd_tsdf_corner_evidence_.find(corner_key);
            if (corner == rgbd_tsdf_corner_evidence_.end() ||
                corner->second.weight <= 0.0f ||
                !std::isfinite(corner->second.distance)) {
                continue;
            }
            ++observed_corner_count;
            has_positive = has_positive || corner->second.distance > 0.0f;
            has_negative = has_negative || corner->second.distance < 0.0f;
        }
        if (observed_corner_count < kMinObservedCornersForPromotion) {
            if (log_evidence_snapshot) {
                incomplete_corner_cells.push_back(key);
            }
            continue;
        }
        if (static_cast<int>(cell_evidence.observed_keyframes.size()) <
            rgbd_tsdf_evidence_promote_min_views_) {
            if (log_evidence_snapshot) {
                waiting_view_cells.push_back(key);
            }
            continue;
        }
        if (!has_positive || !has_negative) {
            if (log_evidence_snapshot) {
                rejected_non_surface_cells.push_back(key);
            }
            if (affected) {
                affected_rejected_non_surface_cells.push_back(key);
            }
            continue;
        }
        if (log_evidence_snapshot) {
            zero_crossing_cells.push_back(key);
        }
        if (!affected) {
            continue;
        }

        const Eigen::Vector3f center = rgbd_tsdf_layout_scene_min_ +
            rgbd_tsdf_layout_cell_size_ * Eigen::Vector3f(
                static_cast<float>(key.x) + 0.5f,
                static_cast<float>(key.y) + 0.5f,
                static_cast<float>(key.z) + 0.5f);
        centers.insert(centers.end(), {center.x(), center.y(), center.z()});
        const cv::Vec3f color = cell_evidence.color_observations > 0
            ? cell_evidence.color_sum /
                  static_cast<float>(cell_evidence.color_observations)
            : cv::Vec3f(0.5f, 0.5f, 0.5f);
        colors.insert(colors.end(), {
            std::clamp(color[0], 0.0f, 1.0f),
            std::clamp(color[1], 0.0f, 1.0f),
            std::clamp(color[2], 0.0f, 1.0f)});
        promoted_cells.push_back(key);

        for (const auto& corner_key : corner_keys) {
            const auto corner = rgbd_tsdf_corner_evidence_.find(corner_key);
            if (corner == rgbd_tsdf_corner_evidence_.end() ||
                corner->second.weight <= 0.0f ||
                !std::isfinite(corner->second.distance)) {
                continue;
            }
            if (!direct_corner_keys.insert(corner_key).second) {
                continue;
            }
            const Eigen::Vector3f world = rgbd_tsdf_layout_scene_min_ +
                rgbd_tsdf_layout_cell_size_ * Eigen::Vector3f(
                    static_cast<float>(corner_key.x),
                    static_cast<float>(corner_key.y),
                    static_cast<float>(corner_key.z));
            direct_corner_points.insert(
                direct_corner_points.end(), {world.x(), world.y(), world.z()});
            direct_corner_values.push_back(corner->second.distance);
        }
    }
    if (log_evidence_snapshot) {
        logRgbdTsdfEvidenceCellsToRerun(
            iteration,
            incomplete_corner_cells,
            "world/rgbd_tsdf_evidence/incomplete_corners");
        logRgbdTsdfEvidenceCellsToRerun(
            iteration,
            waiting_view_cells,
            "world/rgbd_tsdf_evidence/waiting_views");
        logRgbdTsdfEvidenceCellsToRerun(
            iteration,
            zero_crossing_cells,
            "world/rgbd_tsdf_evidence/zero_crossing");
        logRgbdTsdfEvidenceCellsToRerun(
            iteration,
            rejected_non_surface_cells,
            "world/rgbd_tsdf_evidence/rejected_non_surface");
    }
    for (const auto& key : affected_rejected_non_surface_cells) {
        rgbd_tsdf_cell_evidence_.erase(key);
    }
    if (promoted_cells.empty()) {
        if (log_evidence_snapshot) {
            logRgbdTsdfEvidenceCellsToRerun(
                iteration,
                {},
                "world/rgbd_tsdf_evidence/promoted");
        }
        return;
    }

    torch::Tensor center_tensor = torch::from_blob(
        centers.data(),
        {static_cast<int64_t>(promoted_cells.size()), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
    torch::Tensor color_tensor = torch::from_blob(
        colors.data(),
        {static_cast<int64_t>(promoted_cells.size()), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
    torch::Tensor corner_points = torch::from_blob(
        direct_corner_points.data(),
        {static_cast<int64_t>(direct_corner_values.size()), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
    torch::Tensor corner_values = torch::from_blob(
        direct_corner_values.data(),
        {static_cast<int64_t>(direct_corner_values.size()), 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();

    sv::VoxelModel::IncreasePcdStats stats;
    {
        torch::NoGradGuard no_grad;
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        voxel_model_->setNextSdfInitializationGridSamples(
            corner_points, corner_values);
        voxel_model_->setNextRealInsertionRerunEntityPath(
            "world/rgbd_tsdf_evidence/promoted");
        voxel_model_->increasePcd(
            center_tensor,
            color_tensor,
            getIteration(),
            incrementalMappingCameras());
        voxel_model_->setNextRealInsertionRerunEntityPath("");
        stats = voxel_model_->lastIncreasePcdStats();
    }

    // Remove evidence only for cells that now resolve to active topology. Cells
    // rejected by insertion-time geometry filters remain evidence-only.
    torch::Tensor active;
    {
        torch::NoGradGuard no_grad;
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        active = std::get<1>(voxel_model_->querySdfTrilinear(
                    center_tensor.to(device_type_).contiguous()))
                     .detach().to(torch::kCPU).to(torch::kBool).contiguous();
    }
    const bool* active_ptr = active.data_ptr<bool>();
    std::vector<sv::RgbdTsdfGridKey> activated_cells;
    activated_cells.reserve(promoted_cells.size());
    for (std::size_t index = 0; index < promoted_cells.size(); ++index) {
        if (active_ptr[index]) {
            activated_cells.push_back(promoted_cells[index]);
        }
    }
    if (log_evidence_snapshot) {
        logRgbdTsdfEvidenceCellsToRerun(
            iteration,
            activated_cells,
            "world/rgbd_tsdf_evidence/promoted");
    }
    for (const auto& key : activated_cells) {
        rgbd_tsdf_cell_evidence_.erase(key);
    }
    if (stats.new_voxels > 0 &&
        (rerun_params_.run_whole_run_ || rerun_params_.rerun_svrecon_debug_)) {
        rerun_state_.whole_run_live_voxels_dirty_ = true;
    }
}
