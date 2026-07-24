#pragma once

#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>
#include <string>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <torch/torch.h>
#include <pybind11/numpy.h>

namespace sv {

/**
 * Thin C++ → Python bridge to the Python RerunVisualizer.
 *
 * Internally this calls:
 *   scripts_voxel/python_rerun_bridge/visualizer_wrapper.RerunVisualizer
 *
 * It does NOT assume ownership of the Python interpreter; we just acquire
 * the GIL and import the module. You already embed Python for SVRaster.
 */
class RerunVisualizerBridge {
public:
    static RerunVisualizerBridge& instance();

    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }

    // Initialize Python-side RerunVisualizer (no-op if already initialized).
    void init(const std::string& app_id = "PhotoSLAM-SVRaster",
              bool spawn_viewer = true);

    void saveRecording(const std::string& path);
    void saveDebugRecording(const std::string& recording_name,
                            const std::string& path);

    // Log pose + image + 2D keypoints (Photo-SLAM / cuVSLAM style).
    //
    // T_W_C         : world_T_cam, 4x4 float.
    // image_rgb     : tracking image in pipeline RGB order (cv::Mat).
    // keypoints_uv  : vector of (u,v) pixel coords.
    // track_ids     : optional vector of track IDs (same length as keypoints_uv).
    // iteration     : mapper iteration used as the Rerun timeline.
    // keyframe_id   : stable SLAM keyframe id used as the Rerun entity name.
    void visualizeCamera(
        const Eigen::Matrix4f& T_W_C,
        const cv::Mat& image_rgb,
        const std::vector<Eigen::Vector2f>& keypoints_uv,
        const std::vector<int>& track_ids,
        int iteration,
        int keyframe_id,
        float fx,
        float fy,
        float cx,
        float cy,
        int source_frame_id = -1
    );

    void visualizeDebugCamera(
        const std::string& recording_name,
        const Eigen::Matrix4f& T_W_C,
        const cv::Mat& image_rgb,
        const std::vector<Eigen::Vector2f>& keypoints_uv,
        const std::vector<int>& track_ids,
        int iteration,
        int keyframe_id,
        float fx,
        float fy,
        float cx,
        float cy,
        int source_frame_id = -1
    );

    void visualizeDebugCameraPose(
        const std::string& recording_name,
        const Eigen::Matrix4f& T_W_C,
        int iteration,
        int keyframe_id
    );

    // Log a triangle mesh for debug visualization.
    //
    // vertices  : [N,3] float tensor on any device.
    // colors    : [N,3] float or uint8 data in tensor (optional, can be empty).
    // triangles : [M,3] int tensor (indices).
    void visualizeMesh(
        const torch::Tensor& vertices,
        const torch::Tensor& colors,
        const torch::Tensor& triangles
    );

    void visualizeVoxelBoxes(
        const torch::Tensor& centers,   // [N,3] float
        const torch::Tensor& sizes,     // [N] or [N,1] or [N,3] float
        const torch::Tensor& colors,     // [N,3] float or uint8, can be undefined
        int iteration,
        const std::string& entity_path = "world/voxels"
    );

    void visualizeDebugVoxelBoxes(
        const std::string& recording_name,
        const torch::Tensor& centers,   // [N,3] float
        const torch::Tensor& sizes,     // [N] or [N,1] or [N,3] float
        const torch::Tensor& colors,     // [N,3] float or uint8, can be undefined
        int iteration,
        const std::string& entity_path = "world/voxels"
    );

    // Log sparse, axis-aligned octree leaves using Rerun's VoxelGridMap.
    // Each octree level is emitted under entity_path/level_<L>, since one
    // VoxelGridMap entity has a single voxel size.
    void visualizeDebugVoxelGridMap(
        const std::string& recording_name,
        const torch::Tensor& centers,    // [N,3] float world centers
        const torch::Tensor& sizes,      // [N] or [N,1] float edge lengths
        const torch::Tensor& levels,     // [N] or [N,1] integer octree levels
        const torch::Tensor& colors,     // [N,3]/[N,4] optimized/fused colors
        const torch::Tensor& grid_origin,// [3] world-space minimum corner
        int iteration,
        const std::string& entity_path = "world/voxels",
        float opacity = 0.8f
    );

    // Deferred-friendly fixed-level VoxelGridMap input. Integer grid indices
    // and optional RGB/RGBA colors are retained during mapping; tensor and
    // Rerun conversion happens only when the recording is flushed.
    void visualizeDebugVoxelGridIndices(
        const std::string& recording_name,
        const std::vector<std::int32_t>& indices_xyz, // flat [N*3]
        const std::vector<float>& colors,             // flat [N*3] or [N*4]
        const Eigen::Vector3f& grid_origin,
        float voxel_size,
        std::int32_t level,
        int iteration,
        const std::string& entity_path = "world/voxels",
        float opacity = 0.8f
    );

    void visualizeTriangleMesh(
        const torch::Tensor& vertices,   // [N,3], float32
        const torch::Tensor& colors,     // [N,3], float32 or uint8, or undefined
        const torch::Tensor& triangles,  // [M,3], int32/int64
        int iteration
    );
    void visualizeDebugTriangleMesh(
        const std::string& recording_name,
        const torch::Tensor& vertices,   // [N,3], float32
        const torch::Tensor& colors,     // [N,3], float32 or uint8, or undefined
        const torch::Tensor& triangles,  // [M,3], int32/int64
        int iteration,
        const std::string& entity_path = "world/mesh"
    );

    void visualizePlyMesh(
        const std::string& ply_path,
        int iteration,
        const std::string& entity_path = "world/sdf_mesh/live");
    void visualizeDebugPlyMesh(
        const std::string& recording_name,
        const std::string& ply_path,
        int iteration,
        const std::string& entity_path = "world/mesh/reference");
    void visualizeSVRasterMesh(
        const torch::Tensor& centers,   // [N,3] float
        const torch::Tensor& sizes,     // [N] or [N,1] or [N,3] float
        const torch::Tensor& colors,     // [N,3] float or uint8, can be undefined
        int iteration
    );

    void visualizePoints3D(
    const torch::Tensor& points_xyz,     // [N,3] float32
    const torch::Tensor& colors,         // optional [N,3] uint8 or float
    int iteration,
    const std::string& entity_path,
    float radius = 0.02f,
    const std::vector<std::string>& labels = {});

    void visualizeDebugPoints3D(
        const std::string& recording_name,
        const torch::Tensor& points_xyz,     // [N,3] float32
        const torch::Tensor& colors,         // optional [N,3]/[N,4] uint8 or float
        int iteration,
        const std::string& entity_path,
        float radius = 0.02f,
        const std::vector<std::string>& labels = {});

    void visualizeLineStrip3D(
        const torch::Tensor& points_xyz,     // [N,3] float32
        const torch::Tensor& color_rgb,      // optional [3] uint8/float
        int iteration,
        const std::string& entity_path,
        float radius = 0.01f);

    void visualizeScalar(
        double value,
        int iteration,
        const std::string& entity_path);

    void visualizeDebugScalar(
        const std::string& recording_name,
        double value,
        int iteration,
        const std::string& entity_path);

    void visualizeMapsFrameRecording(
        const std::string& recording_name,
        int keyframe_id,
        int iteration,
        const cv::Mat& gt_rgb,
        const cv::Mat& rendered_rgb,
        const cv::Mat& rgb_error,
        const cv::Mat& gt_depth_rgb,
        const cv::Mat& rendered_depth_rgb,
        const cv::Mat& depth_error_rgb,
        const cv::Mat& depth_gap_rgb,
        const cv::Mat& gt_normal_rgb,
        const cv::Mat& rendered_normal_rgb,
        const cv::Mat& normal_error_rgb,
        double psnr,
        double ssim,
        double depth_l1_m,
        double depth_gap_percent,
        double normal_mean_deg);

    void visualizeDebugGtSdfMesh(
        const std::string& recording_name,
        const std::string& gt_mesh_path,
        bool align_gt_to_slam,
        const std::string& gt_traj_path,
        int align_min_pairs,
        int iteration);

    torch::Tensor computeGtSignedDistance(
        const torch::Tensor& points_xyz,     // [N,3] float
        const std::string& gt_mesh_path,
        bool align_gt_to_slam,
        const std::string& gt_traj_path,
        int align_min_pairs);

    torch::Tensor computeGtSurfaceDistance(
        const torch::Tensor& points_xyz,     // [N,3] float
        const std::string& gt_mesh_path,
        bool align_gt_to_slam,
        const std::string& gt_traj_path,
        int align_min_pairs);

    torch::Tensor computeGtProjectiveSdf(
        const torch::Tensor& points_xyz,     // [N,3] float world points
        const torch::Tensor& Tcw,            // [4,4] float world-to-camera
        float fx,
        float fy,
        float cx,
        float cy,
        int width,
        int height,
        const std::string& gt_mesh_path,
        bool align_gt_to_slam,
        const std::string& gt_traj_path,
        int align_min_pairs);

    void visualizeSdfVoxelsRecording(
        const std::string& recording_name,
        const torch::Tensor& centers,        // [N,3] float
        const torch::Tensor& sizes,          // [N] or [N,1] or [N,3] float, full edge length
        const torch::Tensor& corner_points,  // [N,8,3] float
        const torch::Tensor& computed_sdf,   // [N,8] float metric SDF
        const torch::Tensor& sdf_weights,    // [N,8] float
        const torch::Tensor& corner_density, // [N,8] float raw density values
        const torch::Tensor& gt_sdf,         // [N,8] float GT mesh SDF
        const torch::Tensor& voxel_colors,   // optional [N,3] or [N,4] colors
        const torch::Tensor& voxel_ids,      // optional [N] int ids
        const torch::Tensor& source_sdf_mask,      // optional [N] bool
        const torch::Tensor& source_svraster_mask, // optional [N] bool
        int iteration,
        const std::string& gt_mesh_path,
        bool align_gt_to_slam,
        const std::string& gt_traj_path,
        int align_min_pairs,
        float surface_band_m,
        float min_weight,
        bool log_gt_mesh,
        const std::string& entity_path = "world/voxels_sdf_pruned");

private:
    RerunVisualizerBridge();
    void ensureInitialized();
    bool deferDebugCall(std::function<void()> call);
    void flushDeferredDebugCalls();

    bool enabled_ = true;
    bool initialized_ = false;
    bool flushing_deferred_debug_calls_ = false;
    std::mutex deferred_debug_mutex_;
    std::vector<std::function<void()>> deferred_debug_calls_;

    // Python object: instance of python_rerun_bridge.visualizer_wrapper.RerunVisualizer
    struct PyImpl;
    PyImpl* impl_ = nullptr;
};

} // namespace sv
