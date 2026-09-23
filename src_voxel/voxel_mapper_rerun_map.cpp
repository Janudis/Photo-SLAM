#include "include_voxel/voxel_mapper.h"
#include "include_voxel/voxel_mapper_utils.h"
#include "include_voxel/voxel_mapper_supervision.h"
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
#include <c10/cuda/CUDACachingAllocator.h>

#include "ORB-SLAM3/include/Atlas.h"
#include "ORB-SLAM3/include/KeyFrame.h"
#include "ORB-SLAM3/include/MapPoint.h"
#include "src_voxel/voxel_mapper_rerun_helpers.h"
#include "src_voxel/voxel_mapper_rerun_tsdf_helpers.h"

void VoxelMapper::logCurrentOrbMapPointsToReconstructionRerun(int iteration)
{
    if (!rerun_params_.enable_rerun_ ||
        (!rerun_params_.rerun_reconstruction_mesh_ &&
         !rerun_params_.rerun_svrecon_debug_ &&
         !rerun_params_.rerun_monocular_debug_ &&
         !rerun_params_.run_whole_run_) ||
        !mpSLAM || !mpSLAM->getAtlas()) {
        return;
    }

    auto* pMap = mpSLAM->getAtlas()->GetCurrentMap();
    if (!pMap) {
        return;
    }

    {
        std::unique_lock<std::mutex> lock_map(pMap->mMutexMapUpdate);
        const std::vector<ORB_SLAM3::MapPoint*> map_points = pMap->GetAllMapPoints();
        for (auto* pMP : map_points) {
            if (!pMP || pMP->isBad()) {
                continue;
            }
            const Eigen::Vector3f pos = pMP->GetWorldPos();
            const Eigen::Vector3f color = pMP->GetColorRGB();
            if (!pos.allFinite() || !color.allFinite()) {
                continue;
            }
            rerun_state_.whole_run_orb_points_by_id_[
                static_cast<std::uint64_t>(pMP->mnId)] = {
                    pos.x(), pos.y(), pos.z(),
                    color.x(), color.y(), color.z()};
        }
    }

    std::vector<std::uint64_t> point_ids;
    point_ids.reserve(rerun_state_.whole_run_orb_points_by_id_.size());
    for (const auto& item : rerun_state_.whole_run_orb_points_by_id_) {
        point_ids.push_back(item.first);
    }
    std::sort(point_ids.begin(), point_ids.end());

    std::vector<float> pts;
    std::vector<float> cols;
    pts.reserve(point_ids.size() * 3);
    cols.reserve(point_ids.size() * 3);
    for (const std::uint64_t point_id : point_ids) {
        const auto& value =
            rerun_state_.whole_run_orb_points_by_id_.at(point_id);
        pts.insert(pts.end(), {value[0], value[1], value[2]});
        cols.insert(cols.end(), {value[3], value[4], value[5]});
    }

    const int64_t n_points = static_cast<int64_t>(pts.size() / 3);
    if (n_points == 0) {
        return;
    }
    auto points = torch::from_blob(
        pts.data(),
        {n_points, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
    auto colors = torch::from_blob(
        cols.data(),
        {n_points, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
    colors = normalizeRerunPointColors(colors);

    if (rerun_params_.rerun_reconstruction_mesh_) {
        sv::RerunVisualizerBridge::instance().visualizeDebugPoints3D(
            "reconstruction_mesh",
            points,
            colors,
            iteration,
            "world/orb/map_points",
            0.015f);
    }
    if (rerun_params_.rerun_svrecon_debug_) {
        sv::RerunVisualizerBridge::instance().visualizeDebugPoints3D(
            "svrecon_debug",
            points,
            colors,
            iteration,
            "world/orb/map_points",
            0.015f);
    }
    if (rerun_params_.run_whole_run_) {
        sv::RerunVisualizerBridge::instance().visualizeDebugPoints3D(
            "whole_run",
            points,
            colors,
            iteration,
            "world/orb/map_points",
            0.015f);
    }
    if (rerun_params_.rerun_monocular_debug_) {
        sv::RerunVisualizerBridge::instance().visualizeDebugPoints3D(
            "monocular_debug",
            points,
            colors,
            iteration,
            "world/orb_points",
            0.015f);
    }
}

void VoxelMapper::logCurrentOrbKeyframePosesToReconstructionRerun(int iteration)
{
    if (!rerun_params_.enable_rerun_ ||
        (!rerun_params_.rerun_reconstruction_mesh_ &&
         !rerun_params_.rerun_svrecon_debug_ &&
         !rerun_params_.rerun_monocular_debug_ &&
         !rerun_params_.run_whole_run_) ||
        !mpSLAM || !mpSLAM->getAtlas()) {
        return;
    }

    auto* pMap = mpSLAM->getAtlas()->GetCurrentMap();
    if (!pMap) {
        return;
    }

    std::vector<std::pair<unsigned long, Eigen::Matrix4f>> poses;
    {
        std::unique_lock<std::mutex> lock_map(pMap->mMutexMapUpdate);
        const std::vector<ORB_SLAM3::KeyFrame*> keyframes = pMap->GetAllKeyFrames();
        poses.reserve(keyframes.size());
        for (auto* pKF : keyframes) {
            if (!pKF || pKF->isBad()) {
                continue;
            }
            poses.emplace_back(pKF->mnId, pKF->GetPoseInverse().matrix());
        }
    }

    std::sort(
        poses.begin(),
        poses.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    const unsigned long begin =
        static_cast<unsigned long>(std::max(0, rerun_params_.rerun_keyframe_start_));
    const unsigned long end = rerun_params_.rerun_max_keyframes_ > 0
        ? begin + static_cast<unsigned long>(rerun_params_.rerun_max_keyframes_)
        : std::numeric_limits<unsigned long>::max();

    for (const auto& [kf_id, T_W_C] : poses) {
        if (kf_id < begin || kf_id >= end) {
            continue;
        }
        const auto previous = rerun_reconstruction_last_orb_poses_.find(kf_id);
        if (previous != rerun_reconstruction_last_orb_poses_.end() &&
            previous->second.isApprox(T_W_C, 1.0e-6f)) {
            continue;
        }
        if (rerun_params_.rerun_reconstruction_mesh_) {
            sv::RerunVisualizerBridge::instance().visualizeDebugCameraPose(
                "reconstruction_mesh",
                T_W_C,
                iteration,
                static_cast<int>(kf_id));
        }
        if (rerun_params_.rerun_svrecon_debug_) {
            sv::RerunVisualizerBridge::instance().visualizeDebugCameraPose(
                "svrecon_debug",
                T_W_C,
                iteration,
                static_cast<int>(kf_id));
        }
        if (rerun_params_.run_whole_run_) {
            sv::RerunVisualizerBridge::instance().visualizeDebugCameraPose(
                "whole_run",
                T_W_C,
                iteration,
                static_cast<int>(kf_id));
        }
        if (rerun_params_.rerun_monocular_debug_) {
            sv::RerunVisualizerBridge::instance().visualizeDebugCameraPose(
                "monocular_debug",
                T_W_C,
                iteration,
                static_cast<int>(kf_id));
        }
        rerun_reconstruction_last_orb_poses_[kf_id] = T_W_C;
    }
}

void VoxelMapper::logReconstructionMeshToRerun(int iteration)
{
    if (!rerun_params_.enable_rerun_ || !rerun_params_.rerun_reconstruction_mesh_ ||
        !voxel_model_ || !scene_) {
        return;
    }
    if (rerun_params_.rerun_reconstruction_mesh_interval_ <= 0 ||
        (iteration % rerun_params_.rerun_reconstruction_mesh_interval_) != 0) {
        return;
    }

    torch::NoGradGuard no_grad;

    const auto& keyframes = scene_->keyframes();
    if (keyframes.empty()) return;

    auto centers_cpu = voxel_model_->voxCenter().detach().to(torch::kCPU).contiguous();
    if (!centers_cpu.defined() || centers_cpu.numel() == 0) return;

    const float current_sdf_voxel = std::max(1.0e-6f, sdfMetricVoxelSize());
    const float voxel_length =
        std::max(current_sdf_voxel, std::max(1.0e-6f, sdf_params_.sdf_voxel_size_m_));
    const float sdf_trunc =
        std::max(1.0e-6f, sdf_params_.sdf_init_trunc_vox_ * voxel_length);
    const float depth_trunc =
        (sdf_params_.sdf_init_max_depth_m_ > 0.0f)
            ? sdf_params_.sdf_init_max_depth_m_
            : kCommonEvalDepthTrunc;

    SparseTsdfVolume volume(voxel_length, sdf_trunc);

    bool froze_geo = false;
    auto unfreeze_geo = [&]() {
        if (froze_geo) {
            voxel_model_->unfreezeVoxGeo();
            froze_geo = false;
        }
    };
    try
    {
        voxel_model_->freezeVoxGeo();
        froze_geo = true;

        for (const auto& [kfid, pkf] : keyframes)
        {
            if (!pkf) continue;
            if (pkf->image_width_ <= 0 || pkf->image_height_ <= 0 || pkf->intr_.size() < 4) continue;
            const int image_height = pkf->image_height_;
            const int image_width = pkf->image_width_;

            const sv::MiniCam cam = pkf->toMiniCam(pkf->image_height_, pkf->image_width_);
            std::unordered_map<std::string, torch::Tensor> render_pkg =
                voxel_model_->render(
                    cam,
                    image_height,
                    image_width,
                    torch::Tensor(),
                    nullptr,
                    false,
                    1.0f,
                    true,
                    false,
                    false,
                    false,
                    false,
                    sv::RenderOpts{});
            if (render_pkg.empty()) continue;

            auto normalize_color_chw = [&](torch::Tensor t) {
                t = t.detach().contiguous();
                if (t.dim() == 4 && t.size(0) == 1) t = t.squeeze(0);
                if (t.dim() != 3) {
                    throw std::runtime_error("logReconstructionMeshToRerun: unexpected color tensor rank");
                }
                if (t.size(0) == 3 && t.size(1) == image_height && t.size(2) == image_width) {
                    return t.to(torch::kFloat32).contiguous();
                }
                if (t.size(0) == 3 && t.size(1) == image_width && t.size(2) == image_height) {
                    return t.transpose(1, 2).to(torch::kFloat32).contiguous();
                }
                if (t.size(0) == image_height && t.size(1) == image_width && t.size(2) == 3) {
                    return t.permute({2, 0, 1}).to(torch::kFloat32).contiguous();
                }
                if (t.size(0) == image_width && t.size(1) == image_height && t.size(2) == 3) {
                    return t.permute({2, 1, 0}).to(torch::kFloat32).contiguous();
                }
                std::ostringstream oss;
                oss << "logReconstructionMeshToRerun: unsupported color shape " << t.sizes();
                throw std::runtime_error(oss.str());
            };
            auto normalize_hw = [&](torch::Tensor t, const char* name) {
                t = t.detach().to(torch::kFloat32).contiguous();
                if (t.dim() == 3 && t.size(0) == 1) t = t.squeeze(0);
                if (t.dim() != 2) {
                    std::ostringstream oss;
                    oss << "logReconstructionMeshToRerun: unexpected " << name << " tensor rank";
                    throw std::runtime_error(oss.str());
                }
                if (t.size(0) == image_height && t.size(1) == image_width) {
                    return t.contiguous();
                }
                if (t.size(0) == image_width && t.size(1) == image_height) {
                    return t.transpose(0, 1).contiguous();
                }
                std::ostringstream oss;
                oss << "logReconstructionMeshToRerun: unsupported " << name << " shape " << t.sizes();
                throw std::runtime_error(oss.str());
            };

            auto it_color = render_pkg.find("color");
            auto it_depth = render_pkg.find("depth");
            if (it_color == render_pkg.end() || it_depth == render_pkg.end() ||
                !it_color->second.defined() || !it_depth->second.defined()) {
                continue;
            }

            torch::Tensor rendered_color = normalize_color_chw(it_color->second);
            torch::Tensor depth_pkg = it_depth->second.detach().contiguous();
            torch::Tensor rendered_depth;
            if (depth_pkg.dim() == 3 && depth_pkg.size(0) >= 3) {
                rendered_depth = normalize_hw(depth_pkg.index({2}), "depth[2]");
            } else {
                rendered_depth = normalize_hw(depth_pkg, "depth");
            }

            const auto mask_it = undistort_mask_.find(pkf->camera_id_);
            if (mask_it != undistort_mask_.end())
            {
                auto mask = mask_it->second;
                if (mask.dim() == 3) mask = mask.index({0});
                mask = normalize_hw(mask.to(rendered_depth.device()), "undistort_mask");
                rendered_color = rendered_color * mask.unsqueeze(0);
                rendered_depth = rendered_depth * mask;
            }

            auto color_cpu = rendered_color.clamp(0, 1)
                .mul(255.0f)
                .to(torch::kUInt8)
                .permute({1, 2, 0})
                .contiguous()
                .to(torch::kCPU);
            auto depth_cpu = rendered_depth
                .to(torch::kFloat32)
                .contiguous()
                .to(torch::kCPU);

            cv::Mat color_mat(
                image_height, image_width, CV_8UC3, color_cpu.data_ptr<uint8_t>());
            cv::Mat depth_mat(
                image_height, image_width, CV_32FC1, depth_cpu.data_ptr<float>());

            // The live Rerun mesh is a preview. Downsample before CPU TSDF
            // fusion; the shutdown mesh keeps its full-resolution
            // extraction path.
            constexpr int kLivePreviewDownsample = 2;
            const int fusion_width = std::max(1, image_width / kLivePreviewDownsample);
            const int fusion_height = std::max(1, image_height / kLivePreviewDownsample);
            cv::Mat color_for_fusion;
            cv::Mat depth_for_fusion;
            cv::resize(
                color_mat,
                color_for_fusion,
                cv::Size(fusion_width, fusion_height),
                0.0,
                0.0,
                cv::INTER_AREA);
            cv::resize(
                depth_mat,
                depth_for_fusion,
                cv::Size(fusion_width, fusion_height),
                0.0,
                0.0,
                cv::INTER_NEAREST);
            cv::Mat gt_depth_meters;
            if (getKeyframeDepthMetersForEval(
                    pkf, fusion_height, fusion_width, gt_depth_meters))
            {
                for (int y = 0; y < fusion_height; ++y)
                {
                    const float* gt_ptr = gt_depth_meters.ptr<float>(y);
                    float* depth_ptr = depth_for_fusion.ptr<float>(y);
                    for (int x = 0; x < fusion_width; ++x)
                    {
                        const float gt_depth = gt_ptr[x];
                        if (!std::isfinite(gt_depth) || gt_depth <= 0.0f) {
                            depth_ptr[x] = 0.0f;
                        }
                    }
                }
            }

            cv::Mat filtered_depth = filterDepthOutliersLikeGaussianSlam(depth_for_fusion);
            std::vector<float> fusion_intr = pkf->intr_;
            const float scale_x =
                static_cast<float>(fusion_width) / static_cast<float>(image_width);
            const float scale_y =
                static_cast<float>(fusion_height) / static_cast<float>(image_height);
            fusion_intr[0] *= scale_x;
            fusion_intr[1] *= scale_y;
            fusion_intr[2] *= scale_x;
            fusion_intr[3] *= scale_y;
            volume.integrate(
                color_for_fusion,
                filtered_depth,
                fusion_intr,
                pkf->getPosef(),
                depth_trunc);

        }

        TriangleMeshRgb mesh = volume.extractMesh(rerun_params_.rerun_reconstruction_mesh_min_weight_);
        if (mesh.vertices.empty() || mesh.faces.empty())
        {
            unfreeze_geo();
            return;
        }

        for (auto& v : mesh.vertices) v += kCommonEvalCompensation;
        if (rerun_params_.rerun_reconstruction_mesh_weld_vertices_) {
            weldTriangleMeshVertices(mesh);
        }
        const bool over_vertex_budget =
            rerun_params_.rerun_reconstruction_mesh_max_vertices_ > 0 &&
            mesh.vertices.size() > rerun_params_.rerun_reconstruction_mesh_max_vertices_;
        const bool over_face_budget =
            rerun_params_.rerun_reconstruction_mesh_max_faces_ > 0 &&
            mesh.faces.size() > rerun_params_.rerun_reconstruction_mesh_max_faces_;
        if (over_vertex_budget || over_face_budget) {
            unfreeze_geo();
            return;
        }

        torch::Tensor vertices;
        torch::Tensor colors;
        torch::Tensor triangles;
        triangleMeshToTensors(mesh, vertices, colors, triangles);

        unfreeze_geo();

        sv::RerunVisualizerBridge::instance().visualizeDebugTriangleMesh(
            "reconstruction_mesh",
            vertices,
            colors,
            triangles,
            iteration,
            "world/mesh/live");

    }
    catch (const std::exception& e)
    {
        unfreeze_geo();
        std::cerr << "[RERUN/reconstruction_mesh] failed: " << e.what() << "\n";
    }
}
