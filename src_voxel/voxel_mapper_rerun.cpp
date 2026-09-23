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

void VoxelMapper::logKeyframeCameraToRerunRecordings(
    const std::shared_ptr<VoxelKeyframe>& pkf,
    unsigned long kf_id,
    bool log_reconstruction_mesh)
{
    if (!pkf || !rerun_params_.enable_rerun_) {
        return;
    }
    const bool needs_camera_recording =
        (log_reconstruction_mesh && rerun_params_.rerun_reconstruction_mesh_) ||
        rerun_params_.run_whole_run_ ||
        rerun_params_.rerun_svrecon_debug_ ||
        rerun_params_.rerun_monocular_debug_;
    if (!needs_camera_recording) {
        return;
    }

    const unsigned long rerun_kf_begin =
        static_cast<unsigned long>(std::max(0, rerun_params_.rerun_keyframe_start_));
    if (rerun_params_.rerun_max_keyframes_ > 0 &&
        (kf_id < rerun_kf_begin ||
         kf_id >= rerun_kf_begin + static_cast<unsigned long>(rerun_params_.rerun_max_keyframes_))) {
        return;
    }

    try {
        // ORB-SLAM stores Tcw. Use its inverse directly so cameras, ORB
        // MapPoints, and reconstructed geometry stay in the same world frame.
        const Eigen::Matrix4f T_W_C =
            pkf->getPosef().inverse().matrix();

        const sv::Camera& camera = scene_->cameras_.at(pkf->camera_id_);
        const float fx = static_cast<float>(camera.fx());
        const float fy = static_cast<float>(camera.fy());
        const float cx = static_cast<float>(camera.cx());
        const float cy = static_cast<float>(camera.cy());
        const int source_frame_id =
            voxel_utils::parseFrameIdFromPath(pkf->img_filename_);
        const std::vector<Eigen::Vector2f> kps_uv;
        const std::vector<int> track_ids;

        auto log_debug_camera =
            [&](bool enabled, const std::string& recording_name)
        {
            if (!enabled) {
                return;
            }
            sv::RerunVisualizerBridge::instance().visualizeDebugCamera(
                recording_name,
                T_W_C,
                pkf->img_undist_,
                kps_uv,
                track_ids,
                getIteration(),
                static_cast<int>(kf_id),
                fx, fy, cx, cy,
                source_frame_id);
        };

        log_debug_camera(
            log_reconstruction_mesh && rerun_params_.rerun_reconstruction_mesh_,
            "reconstruction_mesh");
        log_debug_camera(rerun_params_.run_whole_run_, "whole_run");
        log_debug_camera(rerun_params_.rerun_svrecon_debug_, "svrecon_debug");
        log_debug_camera(
            rerun_params_.rerun_monocular_debug_, "monocular_debug");
    } catch (const c10::Error& e) {
        (void)e;
    } catch (const std::exception& e) {
        (void)e;
    }
}

void VoxelMapper::saveRerunRecordingsAtShutdown()
{
    if (!rerun_params_.enable_rerun_) {
        return;
    }

    ensureEmbeddedPythonRuntime(/*import_torch_cuda=*/false);

    const auto rrd_dir = result_dir_ / "rerun";
    std::filesystem::create_directories(rrd_dir);

    if (rerun_params_.run_whole_run_) {
        sv::RerunVisualizerBridge::instance().saveDebugRecording(
            "whole_run",
            (rrd_dir / "whole_run.rrd").string());
    }
    if (rerun_params_.rerun_svrecon_debug_) {
        sv::RerunVisualizerBridge::instance().saveDebugRecording(
            "svrecon_debug",
            (rrd_dir / "svrecon_debug.rrd").string());
    }
    if (rerun_params_.rerun_monocular_debug_) {
        sv::RerunVisualizerBridge::instance().saveDebugRecording(
            "monocular_debug",
            (rrd_dir / "svrecon_debug_monocular.rrd").string());
    }
    if (rerun_params_.rerun_reconstruction_mesh_) {
        sv::RerunVisualizerBridge::instance().saveDebugRecording(
            "reconstruction_mesh",
            (rrd_dir / "run_reconstruction_mesh.rrd").string());
    }
}

void VoxelMapper::logLearnedDepthMapsToWholeRunRerun()
{
    if (!rerun_params_.enable_rerun_ ||
        !rerun_params_.run_whole_run_ || !scene_) {
        return;
    }

    struct DepthFrame
    {
        std::shared_ptr<VoxelKeyframe> keyframe;
        cv::Mat gt_depth;
    };
    std::vector<DepthFrame> frames;
    std::vector<std::pair<float, double>> scale_samples;

    for (const auto& [keyframe_id, keyframe] : scene_->keyframes()) {
        (void)keyframe_id;
        if (!keyframe || keyframe->monocular_depth_prior_.empty() ||
            keyframe->monocular_depth_source_ == sv::LearnedDepthSource::None) {
            continue;
        }

        cv::Mat gt_depth;
        voxel_eval::getKeyframeDepthMetersForEval(
            keyframe,
            keyframe->monocular_depth_prior_.rows,
            keyframe->monocular_depth_prior_.cols,
            gt_depth);
        if (!gt_depth.empty()) {
            const torch::Tensor model_depth = torch::from_blob(
                keyframe->monocular_depth_prior_.data,
                {keyframe->monocular_depth_prior_.rows,
                 keyframe->monocular_depth_prior_.cols},
                torch::TensorOptions().dtype(torch::kFloat32)).clone();
            voxel_eval::DepthScaleFitStats stats;
            if (voxel_eval::computeDepthScaleFitStats(
                    model_depth,
                    gt_depth,
                    std::max(1.0e-8f, RGBD_min_depth_),
                    1.0e6f,
                    stats)) {
                scale_samples.emplace_back(
                    stats.scale,
                    static_cast<double>(stats.overlap_count));
            }
        }
        frames.push_back({keyframe, std::move(gt_depth)});
    }
    if (frames.empty()) {
        return;
    }

    float model_to_metric_scale = 1.0f;
    const bool has_metric_scale =
        voxel_eval::computeWeightedMedianScale(
            scale_samples, model_to_metric_scale);
    constexpr float viz_min = 0.0f;
    constexpr float viz_max = 6.0f;
    const float valid_min = std::max(1.0e-8f, RGBD_min_depth_);
    const float valid_max = std::min(RGBD_max_depth_, viz_max);
    sv::RerunVisualizerBridge& bridge =
        sv::RerunVisualizerBridge::instance();

    for (const DepthFrame& frame : frames) {
        const auto& keyframe = frame.keyframe;
        cv::Mat model_depth_metric;
        keyframe->monocular_depth_prior_.convertTo(
            model_depth_metric,
            CV_32FC1,
            has_metric_scale ? model_to_metric_scale : 1.0f);
        const cv::Mat model_rgb = voxel_eval::bgrToRgbImage(
            voxel_eval::colorizeDepthMatJet(
                model_depth_metric,
                valid_min,
                valid_max,
                viz_min,
                viz_max));
        const int iteration = keyframe->monocular_depth_prior_iteration_ >= 0
            ? keyframe->monocular_depth_prior_iteration_
            : getIteration();
        bridge.visualizeDebugImage(
            "whole_run",
            model_rgb,
            iteration,
            static_cast<int>(keyframe->fid_),
            "depth/model/mvs");

        if (!frame.gt_depth.empty()) {
            const cv::Mat gt_rgb = voxel_eval::bgrToRgbImage(
                voxel_eval::colorizeDepthMatJet(
                    frame.gt_depth,
                    valid_min,
                    valid_max,
                    viz_min,
                    viz_max));
            bridge.visualizeDebugImage(
                "whole_run",
                gt_rgb,
                iteration,
                static_cast<int>(keyframe->fid_),
                "depth/ground_truth");
        }
    }
}
