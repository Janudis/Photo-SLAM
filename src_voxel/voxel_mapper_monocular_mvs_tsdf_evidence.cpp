#include "include_voxel/voxel_mapper.h"

#include "include_voxel/tandem_mvs_backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace {

constexpr int kMinObservedCornersForMvsPromotion = 4;

struct MvsFrameCellObservation
{
    cv::Vec3f color_sum{0.0f, 0.0f, 0.0f};
    std::uint32_t color_count = 0;
};

void raycastMvsEvidenceGridSegment(
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

bool containsMvsReference(
    const std::vector<sv::camera_id_t>& keyframes,
    const sv::camera_id_t keyframe)
{
    return std::find(keyframes.begin(), keyframes.end(), keyframe) !=
           keyframes.end();
}

} // namespace

void VoxelMapper::resetMonocularMvsTsdfEvidenceIfLayoutChanged()
{
    if (!voxel_model_ || voxel_model_->insertionOctreeLevel() <= 0) {
        monocular_mvs_tsdf_corner_evidence_.clear();
        monocular_mvs_tsdf_cell_evidence_.clear();
        monocular_mvs_tsdf_layout_cell_size_ = 0.0f;
        monocular_mvs_tsdf_layout_grid_dim_ = 0;
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
        monocular_mvs_tsdf_layout_grid_dim_ != grid_dim ||
        std::abs(monocular_mvs_tsdf_layout_cell_size_ - cell_size) >
            tolerance ||
        (monocular_mvs_tsdf_layout_scene_min_ - scene_min)
                .cwiseAbs().maxCoeff() > tolerance;
    if (changed) {
        monocular_mvs_tsdf_corner_evidence_.clear();
        monocular_mvs_tsdf_cell_evidence_.clear();
        monocular_mvs_tsdf_layout_scene_min_ = scene_min;
        monocular_mvs_tsdf_layout_cell_size_ = cell_size;
        monocular_mvs_tsdf_layout_grid_dim_ = grid_dim;
    }
}

void VoxelMapper::integrateMonocularMvsTsdfEvidence(
    const sv::TandemMvsResult& result)
{
    const std::shared_ptr<VoxelKeyframe> reference =
        monocular_mvs_pending_reference_;
    const int width = monocular_mvs_pending_camera_.width;
    const int height = monocular_mvs_pending_camera_.height;
    if (!monocular_mvs_tsdf_evidence_ || !voxel_model_ || !reference ||
        result.depth.empty() || result.depth.type() != CV_32FC1 ||
        result.depth.cols != width || result.depth.rows != height ||
        result.confidence.empty() ||
        result.confidence.cols != width || result.confidence.rows != height ||
        result.confidence.channels() != 1 ||
        monocular_mvs_pending_reference_rgb_.empty() ||
        monocular_mvs_pending_reference_rgb_.type() != CV_8UC3) {
        throw std::runtime_error(
            "TANDEM returned invalid depth/confidence for MVS TSDF evidence");
    }
    auto fusion_profile =
        profileLaptopModule("mvs_tsdf_evidence_fusion");

    cv::Mat confidence;
    if (result.confidence.type() == CV_32FC1) {
        confidence = result.confidence;
    } else {
        result.confidence.convertTo(confidence, CV_32FC1);
    }

    cv::Mat valid_camera_mask;
    if (!reference->cam_.undistort_mask.empty()) {
        cv::Mat mask_single_channel;
        if (reference->cam_.undistort_mask.channels() == 1) {
            mask_single_channel = reference->cam_.undistort_mask;
        } else {
            cv::extractChannel(
                reference->cam_.undistort_mask,
                mask_single_channel,
                0);
        }
        cv::resize(
            mask_single_channel,
            valid_camera_mask,
            cv::Size(width, height),
            0.0,
            0.0,
            cv::INTER_NEAREST);
        const double scale = valid_camera_mask.depth() == CV_8U
            ? 1.0 / 255.0
            : 1.0;
        valid_camera_mask.convertTo(
            valid_camera_mask, CV_32FC1, scale);
    }

    resetMonocularMvsTsdfEvidenceIfLayoutChanged();
    if (!(monocular_mvs_tsdf_layout_cell_size_ > 0.0f) ||
        monocular_mvs_tsdf_layout_grid_dim_ <= 0 ||
        !(monocular_mvs_pending_camera_.fx > 0.0f) ||
        !(monocular_mvs_pending_camera_.fy > 0.0f)) {
        return;
    }

    const float truncation = std::max(
        monocular_mvs_tsdf_layout_cell_size_,
        monocular_mvs_tsdf_evidence_trunc_vox_ *
            monocular_mvs_tsdf_layout_cell_size_);
    const Eigen::Matrix3f rotation =
        monocular_mvs_pending_c2w_.block<3, 3>(0, 0);
    const Eigen::Vector3f translation =
        monocular_mvs_pending_c2w_.block<3, 1>(0, 3);
    const Eigen::Matrix4f world_to_camera =
        monocular_mvs_pending_c2w_.inverse();
    const float fx = monocular_mvs_pending_camera_.fx;
    const float fy = monocular_mvs_pending_camera_.fy;
    const float cx = monocular_mvs_pending_camera_.cx;
    const float cy = monocular_mvs_pending_camera_.cy;

    std::unordered_map<
        sv::RgbdTsdfGridKey,
        MvsFrameCellObservation,
        sv::RgbdTsdfGridKeyHash> frame_cells;
    std::vector<sv::RgbdTsdfGridKey> traversed_cells;
    const int stride = monocular_mvs_tsdf_evidence_pixel_stride_;
    for (int y = 0; y < height; y += stride) {
        const float* depth_row = result.depth.ptr<float>(y);
        const float* confidence_row = confidence.ptr<float>(y);
        for (int x = 0; x < width; x += stride) {
            const float depth = depth_row[x];
            const float pixel_confidence = confidence_row[x];
            if (!std::isfinite(depth) ||
                depth < monocular_mvs_pending_depth_min_ ||
                depth > monocular_mvs_pending_depth_max_ ||
                !std::isfinite(pixel_confidence) ||
                pixel_confidence <= 0.0f ||
                (!valid_camera_mask.empty() &&
                 valid_camera_mask.at<float>(y, x) < 0.5f)) {
                continue;
            }

            const float x_normalized =
                (static_cast<float>(x) - cx) / fx;
            const float y_normalized =
                (static_cast<float>(y) - cy) / fy;
            const float start_depth = std::max(
                monocular_mvs_pending_depth_min_, depth - truncation);
            const float end_depth = std::min(
                monocular_mvs_pending_depth_max_, depth + truncation);
            if (!(end_depth > start_depth)) {
                continue;
            }

            const Eigen::Vector3f start_camera(
                x_normalized * start_depth,
                y_normalized * start_depth,
                start_depth);
            const Eigen::Vector3f end_camera(
                x_normalized * end_depth,
                y_normalized * end_depth,
                end_depth);
            const Eigen::Vector3f start_world =
                rotation * start_camera + translation;
            const Eigen::Vector3f end_world =
                rotation * end_camera + translation;
            traversed_cells.clear();
            raycastMvsEvidenceGridSegment(
                start_world,
                end_world,
                monocular_mvs_tsdf_layout_scene_min_,
                monocular_mvs_tsdf_layout_cell_size_,
                monocular_mvs_tsdf_layout_grid_dim_,
                traversed_cells);

            const cv::Vec3b rgb =
                monocular_mvs_pending_reference_rgb_.at<cv::Vec3b>(y, x);
            const cv::Vec3f color(
                static_cast<float>(rgb[0]) / 255.0f,
                static_cast<float>(rgb[1]) / 255.0f,
                static_cast<float>(rgb[2]) / 255.0f);
            for (const auto& key : traversed_cells) {
                MvsFrameCellObservation& observation = frame_cells[key];
                observation.color_sum += color;
                ++observation.color_count;
            }
        }
    }
    if (frame_cells.empty()) {
        return;
    }

    // Full-image evidence is independent of the rendered hole mask. Existing
    // active cells are excluded only in 3D, so an incorrect foreground ORB
    // cell cannot suppress evidence at the correct depth farther along a ray.
    std::vector<sv::RgbdTsdfGridKey> frame_keys;
    std::vector<float> center_values;
    frame_keys.reserve(frame_cells.size());
    center_values.reserve(frame_cells.size() * 3);
    for (const auto& item : frame_cells) {
        const sv::RgbdTsdfGridKey& key = item.first;
        const Eigen::Vector3f center =
            monocular_mvs_tsdf_layout_scene_min_ +
            monocular_mvs_tsdf_layout_cell_size_ * Eigen::Vector3f(
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
            monocular_mvs_tsdf_cell_evidence_.erase(frame_keys[index]);
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
        const Eigen::Vector3f world =
            monocular_mvs_tsdf_layout_scene_min_ +
            monocular_mvs_tsdf_layout_cell_size_ * Eigen::Vector3f(
                static_cast<float>(key.x),
                static_cast<float>(key.y),
                static_cast<float>(key.z));
        const Eigen::Vector4f camera_h =
            world_to_camera * Eigen::Vector4f(
                world.x(), world.y(), world.z(), 1.0f);
        const float z = camera_h.z();
        if (!std::isfinite(z) || z <= 1.0e-6f) {
            continue;
        }
        const int x = static_cast<int>(std::lround(
            fx * camera_h.x() / z + cx));
        const int y = static_cast<int>(std::lround(
            fy * camera_h.y() / z + cy));
        if (x < 0 || x >= width || y < 0 || y >= height ||
            (!valid_camera_mask.empty() &&
             valid_camera_mask.at<float>(y, x) < 0.5f)) {
            continue;
        }

        const float measured_depth = result.depth.at<float>(y, x);
        const float measured_confidence = confidence.at<float>(y, x);
        if (!std::isfinite(measured_depth) ||
            measured_depth < monocular_mvs_pending_depth_min_ ||
            measured_depth > monocular_mvs_pending_depth_max_ ||
            !std::isfinite(measured_confidence) ||
            measured_confidence <= 0.0f) {
            continue;
        }
        const float measured_sdf = measured_depth - z;
        if (!std::isfinite(measured_sdf) || measured_sdf < -truncation) {
            continue;
        }
        const float sample = std::clamp(
            measured_sdf, -truncation, truncation);
        const float measurement_weight = std::clamp(
            measured_confidence, 0.0f, 1.0f);
        if (!(measurement_weight > 0.0f)) {
            continue;
        }

        sv::RgbdTsdfCornerEvidence& evidence =
            monocular_mvs_tsdf_corner_evidence_[key];
        const float denominator = evidence.weight + measurement_weight;
        evidence.distance = evidence.weight > 0.0f
            ? (evidence.distance * evidence.weight +
               sample * measurement_weight) / denominator
            : sample;
        evidence.weight = std::min(
            denominator, monocular_mvs_tsdf_evidence_max_weight_);
        updated_corners.insert(key);
    }

    const sv::camera_id_t reference_id =
        static_cast<sv::camera_id_t>(reference->fid_);
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
            monocular_mvs_tsdf_cell_evidence_[item.first];
        if (!containsMvsReference(
                evidence.observed_keyframes, reference_id)) {
            evidence.observed_keyframes.push_back(reference_id);
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
        if (monocular_mvs_tsdf_cell_evidence_.find(item.first) !=
            monocular_mvs_tsdf_cell_evidence_.end()) {
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
                        cell.x >= monocular_mvs_tsdf_layout_grid_dim_ ||
                        cell.y >= monocular_mvs_tsdf_layout_grid_dim_ ||
                        cell.z >= monocular_mvs_tsdf_layout_grid_dim_) {
                        continue;
                    }
                    if (monocular_mvs_tsdf_cell_evidence_.find(cell) !=
                        monocular_mvs_tsdf_cell_evidence_.end()) {
                        affected_cells.insert(cell);
                    }
                }
            }
        }
    }
    promoteMonocularMvsTsdfEvidenceCells(affected_cells);
}

void VoxelMapper::logMonocularMvsTsdfEvidenceCellsToRerun(
    const int iteration,
    const std::vector<sv::RgbdTsdfGridKey>& cells,
    const std::string& entity_path)
{
    if (!rerun_params_.enable_rerun_ ||
        !rerun_params_.rerun_svrecon_debug_ || !voxel_model_ ||
        !(monocular_mvs_tsdf_layout_cell_size_ > 0.0f)) {
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
            monocular_mvs_tsdf_cell_evidence_.find(key);
        if (evidence != monocular_mvs_tsdf_cell_evidence_.end() &&
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
        monocular_mvs_tsdf_layout_scene_min_,
        monocular_mvs_tsdf_layout_cell_size_,
        static_cast<std::int32_t>(voxel_model_->insertionOctreeLevel()),
        iteration,
        entity_path,
        0.8f);
}

void VoxelMapper::promoteMonocularMvsTsdfEvidenceCells(
    const std::unordered_set<
        sv::RgbdTsdfGridKey,
        sv::RgbdTsdfGridKeyHash>& affected_cells)
{
    if (!monocular_mvs_tsdf_evidence_ || !voxel_model_ ||
        monocular_mvs_tsdf_cell_evidence_.empty() ||
        affected_cells.empty()) {
        return;
    }
    auto promotion_profile =
        profileLaptopModule("mvs_tsdf_evidence_promotion");

    const int iteration = getIteration();
    const bool log_evidence_snapshot =
        rerun_params_.enable_rerun_ &&
        rerun_params_.rerun_svrecon_debug_;
    std::vector<sv::RgbdTsdfGridKey> cells_to_evaluate;
    if (log_evidence_snapshot) {
        cells_to_evaluate.reserve(
            monocular_mvs_tsdf_cell_evidence_.size());
        for (const auto& item : monocular_mvs_tsdf_cell_evidence_) {
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
            monocular_mvs_tsdf_cell_evidence_.find(key);
        if (cell_item == monocular_mvs_tsdf_cell_evidence_.end()) {
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
                monocular_mvs_tsdf_corner_evidence_.find(corner_key);
            if (corner == monocular_mvs_tsdf_corner_evidence_.end() ||
                corner->second.weight <
                    monocular_mvs_tsdf_evidence_promote_min_weight_ ||
                !std::isfinite(corner->second.distance)) {
                continue;
            }
            ++observed_corner_count;
            has_positive = has_positive || corner->second.distance > 0.0f;
            has_negative = has_negative || corner->second.distance < 0.0f;
        }
        if (observed_corner_count < kMinObservedCornersForMvsPromotion) {
            if (log_evidence_snapshot) {
                incomplete_corner_cells.push_back(key);
            }
            continue;
        }
        if (static_cast<int>(
                cell_item->second.observed_keyframes.size()) <
            monocular_mvs_tsdf_evidence_promote_min_views_) {
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

        const Eigen::Vector3f center =
            monocular_mvs_tsdf_layout_scene_min_ +
            monocular_mvs_tsdf_layout_cell_size_ * Eigen::Vector3f(
                static_cast<float>(key.x) + 0.5f,
                static_cast<float>(key.y) + 0.5f,
                static_cast<float>(key.z) + 0.5f);
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
                monocular_mvs_tsdf_corner_evidence_.find(corner_key);
            if (corner == monocular_mvs_tsdf_corner_evidence_.end() ||
                corner->second.weight <
                    monocular_mvs_tsdf_evidence_promote_min_weight_ ||
                !std::isfinite(corner->second.distance) ||
                !direct_corner_keys.insert(corner_key).second) {
                continue;
            }
            const Eigen::Vector3f world =
                monocular_mvs_tsdf_layout_scene_min_ +
                monocular_mvs_tsdf_layout_cell_size_ * Eigen::Vector3f(
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
        logMonocularMvsTsdfEvidenceCellsToRerun(
            iteration,
            incomplete_corner_cells,
            "world/monocular_mvs_tsdf_evidence/incomplete_corners");
        logMonocularMvsTsdfEvidenceCellsToRerun(
            iteration,
            waiting_view_cells,
            "world/monocular_mvs_tsdf_evidence/waiting_views");
        logMonocularMvsTsdfEvidenceCellsToRerun(
            iteration,
            zero_crossing_cells,
            "world/monocular_mvs_tsdf_evidence/zero_crossing");
        logMonocularMvsTsdfEvidenceCellsToRerun(
            iteration,
            rejected_non_surface_cells,
            "world/monocular_mvs_tsdf_evidence/rejected_non_surface");
    }
    for (const auto& key : affected_rejected_cells) {
        monocular_mvs_tsdf_cell_evidence_.erase(key);
    }
    if (promoted_cells.empty()) {
        if (log_evidence_snapshot) {
            logMonocularMvsTsdfEvidenceCellsToRerun(
                iteration,
                {},
                "world/monocular_mvs_tsdf_evidence/promoted");
        }
        return;
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
        // Preserve the existing MVS provenance contract for promoted cells.
        voxel_model_->setNextRealInsertionRerunEntityPath(
            "world/monocular_mvs/created");
        voxel_model_->increasePcd(
            center_tensor,
            color_tensor,
            getIteration(),
            incrementalMappingCameras());
        voxel_model_->setNextRealInsertionRerunEntityPath("");
        stats = voxel_model_->lastIncreasePcdStats();
    }

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
        logMonocularMvsTsdfEvidenceCellsToRerun(
            iteration,
            activated_cells,
            "world/monocular_mvs_tsdf_evidence/promoted");
    }
    for (const auto& key : activated_cells) {
        monocular_mvs_tsdf_cell_evidence_.erase(key);
    }
    if (stats.new_voxels > 0 &&
        (rerun_params_.run_whole_run_ ||
         rerun_params_.rerun_svrecon_debug_)) {
        rerun_state_.whole_run_live_voxels_dirty_ = true;
    }
}
