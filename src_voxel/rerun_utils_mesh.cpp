#include "include_voxel/rerun_utils.h"
#include "src_voxel/rerun_utils_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <Eigen/Geometry>          // for Eigen::Quaternionf
#include <opencv2/imgproc.hpp>     // for cv::cvtColor

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace {
// Small helper to pretty-print cv::Mat type, e.g. "CV_8UC3"
std::string cvTypeToString(int type) {
    int depth = type & CV_MAT_DEPTH_MASK;
    int chans = 1 + (type >> CV_CN_SHIFT);

    std::string depthStr;
    switch (depth) {
        case CV_8U:  depthStr = "8U";  break;
        case CV_8S:  depthStr = "8S";  break;
        case CV_16U: depthStr = "16U"; break;
        case CV_16S: depthStr = "16S"; break;
        case CV_32S: depthStr = "32S"; break;
        case CV_32F: depthStr = "32F"; break;
        case CV_64F: depthStr = "64F"; break;
        default:     depthStr = "User"; break;
    }
    std::ostringstream oss;
    oss << "CV_" << depthStr << "C" << chans;
    return oss.str();
}

} // anonymous namespace

namespace sv {
void RerunVisualizerBridge::visualizeTriangleMesh(
    const torch::Tensor& vertices,
    const torch::Tensor& colors,
    const torch::Tensor& triangles,
    int iteration
) {
    ensureInitialized();
    if (!impl_) return;
    if (!vertices.defined() || vertices.numel() == 0) {
        // nothing to log
        return;
    }

    py::gil_scoped_acquire gil;

    // ---- vertices: [N,3] float32 ----
    auto v_cpu = vertices.contiguous().to(torch::kCPU);
    TORCH_CHECK(v_cpu.dim() == 2 && v_cpu.size(1) == 3,
                "vertices must be [N,3]");
    auto v_sizes = v_cpu.sizes(); // (N,3)

    std::vector<ssize_t> v_shape{
        static_cast<ssize_t>(v_sizes[0]),
        static_cast<ssize_t>(v_sizes[1])
    };
    std::vector<ssize_t> v_strides{
        static_cast<ssize_t>(sizeof(float) * v_sizes[1]),
        static_cast<ssize_t>(sizeof(float))
    };

    py::array vertices_np(py::buffer_info(
        v_cpu.data_ptr<float>(),
        sizeof(float),
        py::format_descriptor<float>::format(),
        2,
        v_shape,
        v_strides
    ));

    // ---- colors: [N,3] uint8 or float32, may be empty/undefined ----
    py::object colors_np = py::none();
    if (colors.defined() && colors.numel() > 0) {
        auto c_cpu = colors.contiguous().to(torch::kCPU);
        TORCH_CHECK(c_cpu.dim() == 2 && c_cpu.size(1) == 3,
                    "colors must be [N,3]");

        auto c_sizes = c_cpu.sizes();
        std::vector<ssize_t> c_shape{
            static_cast<ssize_t>(c_sizes[0]),
            static_cast<ssize_t>(c_sizes[1])
        };
        std::vector<ssize_t> c_strides{
            static_cast<ssize_t>(c_cpu.element_size() * c_sizes[1]),
            static_cast<ssize_t>(c_cpu.element_size())
        };

        py::array c_np(py::buffer_info(
            c_cpu.data_ptr(),
            c_cpu.element_size(),
            c_cpu.dtype() == torch::kUInt8
                ? py::format_descriptor<uint8_t>::format()
                : py::format_descriptor<float>::format(),
            2,
            c_shape,
            c_strides
        ));
        colors_np = c_np;
    }

    // ---- triangles: [M,3] int32/int64 ----
    TORCH_CHECK(triangles.defined(), "triangles must be defined");
    auto f_cpu = triangles.contiguous().to(torch::kCPU);
    TORCH_CHECK(f_cpu.dim() == 2 && f_cpu.size(1) == 3,
                "triangles must be [M,3]");

    auto f_sizes = f_cpu.sizes();
    std::vector<ssize_t> f_shape{
        static_cast<ssize_t>(f_sizes[0]),
        static_cast<ssize_t>(f_sizes[1])
    };
    std::vector<ssize_t> f_strides{
        static_cast<ssize_t>(f_cpu.element_size() * f_sizes[1]),
        static_cast<ssize_t>(f_cpu.element_size())
    };

    py::array triangles_np(py::buffer_info(
        f_cpu.data_ptr(),
        f_cpu.element_size(),
        f_cpu.dtype() == torch::kInt32
            ? py::format_descriptor<int32_t>::format()
            : py::format_descriptor<int64_t>::format(),
        2,
        f_shape,
        f_strides
    ));

    // ---- call Python ----
    try {
        impl_->visualizer.attr("visualize_triangle_mesh")(
            vertices_np,
            colors_np,
            triangles_np,
            iteration
        );
    } catch (const py::error_already_set& e) {
        std::cerr << "[RERUN] Python error in visualizeTriangleMesh: "
                  << e.what() << std::endl;
    }
}

void RerunVisualizerBridge::visualizeDebugTriangleMesh(
    const std::string& recording_name,
    const torch::Tensor& vertices,
    const torch::Tensor& colors,
    const torch::Tensor& triangles,
    int iteration,
    const std::string& entity_path
) {
    ensureInitialized();
    if (!impl_) return;
    if (!vertices.defined() || vertices.numel() == 0) {
        return;
    }

    py::gil_scoped_acquire gil;

    auto v_cpu = vertices.contiguous().to(torch::kCPU);
    TORCH_CHECK(v_cpu.dim() == 2 && v_cpu.size(1) == 3,
                "vertices must be [N,3]");
    auto v_sizes = v_cpu.sizes();
    std::vector<ssize_t> v_shape{
        static_cast<ssize_t>(v_sizes[0]),
        static_cast<ssize_t>(v_sizes[1])
    };
    std::vector<ssize_t> v_strides{
        static_cast<ssize_t>(sizeof(float) * v_sizes[1]),
        static_cast<ssize_t>(sizeof(float))
    };
    py::array vertices_np(py::buffer_info(
        v_cpu.data_ptr<float>(),
        sizeof(float),
        py::format_descriptor<float>::format(),
        2,
        v_shape,
        v_strides
    ));

    py::object colors_np = py::none();
    if (colors.defined() && colors.numel() > 0) {
        auto c_cpu = colors.contiguous().to(torch::kCPU);
        TORCH_CHECK(c_cpu.dim() == 2 && c_cpu.size(1) == 3,
                    "colors must be [N,3]");
        auto c_sizes = c_cpu.sizes();
        std::vector<ssize_t> c_shape{
            static_cast<ssize_t>(c_sizes[0]),
            static_cast<ssize_t>(c_sizes[1])
        };
        std::vector<ssize_t> c_strides{
            static_cast<ssize_t>(c_cpu.element_size() * c_sizes[1]),
            static_cast<ssize_t>(c_cpu.element_size())
        };
        py::array c_np(py::buffer_info(
            c_cpu.data_ptr(),
            c_cpu.element_size(),
            c_cpu.dtype() == torch::kUInt8
                ? py::format_descriptor<uint8_t>::format()
                : py::format_descriptor<float>::format(),
            2,
            c_shape,
            c_strides
        ));
        colors_np = c_np;
    }

    TORCH_CHECK(triangles.defined(), "triangles must be defined");
    auto f_cpu = triangles.contiguous().to(torch::kCPU);
    TORCH_CHECK(f_cpu.dim() == 2 && f_cpu.size(1) == 3,
                "triangles must be [M,3]");
    auto f_sizes = f_cpu.sizes();
    std::vector<ssize_t> f_shape{
        static_cast<ssize_t>(f_sizes[0]),
        static_cast<ssize_t>(f_sizes[1])
    };
    std::vector<ssize_t> f_strides{
        static_cast<ssize_t>(f_cpu.element_size() * f_sizes[1]),
        static_cast<ssize_t>(f_cpu.element_size())
    };
    py::array triangles_np(py::buffer_info(
        f_cpu.data_ptr(),
        f_cpu.element_size(),
        f_cpu.dtype() == torch::kInt32
            ? py::format_descriptor<int32_t>::format()
            : py::format_descriptor<int64_t>::format(),
        2,
        f_shape,
        f_strides
    ));

    try {
        impl_->visualizer.attr("visualize_triangle_mesh_recording")(
            py::str(recording_name),
            vertices_np,
            colors_np,
            triangles_np,
            iteration,
            py::str(entity_path)
        );
    } catch (const py::error_already_set& e) {
        std::cerr << "[RERUN] Python error in visualizeDebugTriangleMesh: "
                  << e.what() << std::endl;
    }
}

void RerunVisualizerBridge::visualizePlyMesh(
    const std::string& ply_path,
    int iteration,
    const std::string& entity_path
) {
    ensureInitialized();
    if (!impl_) return;

    py::gil_scoped_acquire gil;
    try {
        impl_->visualizer.attr("visualize_ply_mesh")(
            py::str(ply_path),
            iteration,
            py::str(entity_path)
        );
    } catch (const py::error_already_set& e) {
        std::cerr << "[RERUN] Python error in visualizePlyMesh: "
                  << e.what() << std::endl;
    }
}

void RerunVisualizerBridge::visualizeDebugPlyMesh(
    const std::string& recording_name,
    const std::string& ply_path,
    int iteration,
    const std::string& entity_path,
    bool static_mesh
) {
    ensureInitialized();
    if (!impl_) return;

    py::gil_scoped_acquire gil;
    try {
        impl_->visualizer.attr("visualize_ply_mesh_recording")(
            py::str(recording_name),
            py::str(ply_path),
            iteration,
            py::str(entity_path),
            static_mesh
        );
    } catch (const py::error_already_set& e) {
        std::cerr << "[RERUN] Python error in visualizeDebugPlyMesh: "
                  << e.what() << std::endl;
    }
}

} // namespace sv
