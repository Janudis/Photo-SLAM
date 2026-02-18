#pragma once

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

    // Initialize Python-side RerunVisualizer (no-op if already initialized).
    void init(const std::string& app_id = "PhotoSLAM-SVRaster",
              bool spawn_viewer = true);

    void saveRecording(const std::string& path);

    // Log pose + image + 2D keypoints (Photo-SLAM / cuVSLAM style).
    //
    // T_W_C         : world_T_cam, 4x4 float.
    // image_bgr     : tracking image (cv::Mat, BGR or RGB).
    // keypoints_uv  : vector of (u,v) pixel coords.
    // track_ids     : optional vector of track IDs (same length as keypoints_uv).
    void visualizeCamera(
        const Eigen::Matrix4f& T_W_C,
        const cv::Mat& image_bgr,
        const std::vector<Eigen::Vector2f>& keypoints_uv,
        const std::vector<int>& track_ids,
        int iteration,
        float fx,
        float fy,
        float cx,
        float cy
    );

    // Log a mesh (e.g. TSDF mesh or voxel mesh) similar to nvblox's visualize_nvblox.
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

    void visualizeNvbloxMesh(
        const torch::Tensor& vertices,   // [N,3], float32
        const torch::Tensor& colors,     // [N,3], float32 or uint8, or undefined
        const torch::Tensor& triangles,  // [M,3], int32/int64
        int iteration
    );

    void visualizeNvbloxPlyMesh(const std::string& ply_path, int iteration);
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
    float radius = 0.02f);

    void visualizeLineStrip3D(
        const torch::Tensor& points_xyz,     // [N,3] float32
        const torch::Tensor& color_rgb,      // optional [3] uint8/float
        int iteration,
        const std::string& entity_path,
        float radius = 0.01f);

private:
    RerunVisualizerBridge();
    void ensureInitialized();

    bool initialized_ = false;

    // Python object: instance of python_rerun_bridge.visualizer_wrapper.RerunVisualizer
    struct PyImpl;
    PyImpl* impl_ = nullptr;
};

} // namespace sv
