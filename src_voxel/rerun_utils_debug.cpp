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
void RerunVisualizerBridge::visualizeDebugImage(
    const std::string& recording_name,
    const cv::Mat& image_rgb,
    int iteration,
    int keyframe_id,
    const std::string& entity_path)
{
    if (!enabled_ || image_rgb.empty() || image_rgb.type() != CV_8UC3) {
        return;
    }

    cv::Mat image = image_rgb.clone();
    if (deferDebugCall(
            [this,
             recording_name,
             image,
             iteration,
             keyframe_id,
             entity_path]() {
                this->visualizeDebugImage(
                    recording_name,
                    image,
                    iteration,
                    keyframe_id,
                    entity_path);
            })) {
        return;
    }

    ensureInitialized();
    if (!impl_) return;

    py::gil_scoped_acquire gil;
    py::array image_np(py::buffer_info(
        image.data,
        sizeof(uint8_t),
        py::format_descriptor<uint8_t>::format(),
        3,
        {
            static_cast<ssize_t>(image.rows),
            static_cast<ssize_t>(image.cols),
            static_cast<ssize_t>(3)
        },
        {
            static_cast<ssize_t>(image.step),
            static_cast<ssize_t>(3),
            static_cast<ssize_t>(1)
        }));

    try {
        impl_->visualizer.attr("visualize_image_recording")(
            py::str(recording_name),
            image_np,
            py::str(entity_path),
            iteration,
            keyframe_id);
    } catch (const py::error_already_set& e) {
        std::cerr << "[RERUN] Python error in visualizeDebugImage: "
                  << e.what() << std::endl;
    }
}

void RerunVisualizerBridge::visualizePoints3D(
    const torch::Tensor& points_xyz,
    const torch::Tensor& colors,
    int iteration,
    const std::string& entity_path,
    float radius,
    const std::vector<std::string>& labels
) {
    ensureInitialized();
    if (!impl_) return;
    if (!points_xyz.defined()) return;

    py::gil_scoped_acquire gil;

    auto p_cpu = points_xyz.contiguous().to(torch::kCPU);
    TORCH_CHECK(p_cpu.dim() == 2 && p_cpu.size(1) == 3, "points_xyz must be [N,3]");

    auto ps = p_cpu.sizes();
    std::vector<ssize_t> p_shape{ps[0], ps[1]};
    std::vector<ssize_t> p_strides{
        static_cast<ssize_t>(sizeof(float) * ps[1]),
        static_cast<ssize_t>(sizeof(float))
    };

    py::array points_np(py::buffer_info(
        p_cpu.data_ptr<float>(),
        sizeof(float),
        py::format_descriptor<float>::format(),
        2,
        p_shape,
        p_strides
    ));

    py::object colors_np = py::none();
    if (colors.defined() && colors.numel() > 0) {
        auto c_cpu = colors.contiguous().to(torch::kCPU);
        TORCH_CHECK(c_cpu.dim() == 2, "colors must be 2D [N,3] or [N,4]");
        TORCH_CHECK(c_cpu.size(0) == p_cpu.size(0),
                    "colors must have same N as points_xyz");
        TORCH_CHECK(c_cpu.size(1) == 3 || c_cpu.size(1) == 4,
                    "colors must be [N,3] or [N,4]");

        auto cs = c_cpu.sizes();
        std::vector<ssize_t> c_shape{cs[0], cs[1]};
        std::vector<ssize_t> c_strides{
            static_cast<ssize_t>(c_cpu.element_size() * cs[1]),
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

    py::object labels_py = py::none();
    if (!labels.empty()) {
        if (static_cast<int64_t>(labels.size()) == p_cpu.size(0)) {
            py::list py_labels;
            for (const auto& label : labels) {
                py_labels.append(py::str(label));
            }
            labels_py = py_labels;
        } else {
            std::cerr << "[RERUN] visualizePoints3D labels length mismatch: labels="
                      << labels.size() << " points=" << p_cpu.size(0)
                      << " (ignoring labels)\n";
        }
    }

    impl_->visualizer.attr("visualize_points3d")(
        points_np,
        colors_np,
        radius,
        py::str(entity_path),
        iteration,
        labels_py
    );
}

void RerunVisualizerBridge::visualizeDebugPoints3D(
    const std::string& recording_name,
    const torch::Tensor& points_xyz,
    const torch::Tensor& colors,
    int iteration,
    const std::string& entity_path,
    float radius,
    const std::vector<std::string>& labels
) {
    if (!enabled_) return;
    if (!points_xyz.defined()) return;
    auto points_copy =
        points_xyz.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto colors_copy = colors.defined()
        ? colors.detach().to(torch::kCPU).contiguous()
        : torch::Tensor();
    if (deferDebugCall(
            [this,
             recording_name,
             points_copy,
             colors_copy,
             iteration,
             entity_path,
             radius,
             labels]() {
                this->visualizeDebugPoints3D(
                    recording_name,
                    points_copy,
                    colors_copy,
                    iteration,
                    entity_path,
                    radius,
                    labels);
            })) {
        return;
    }
    ensureInitialized();
    if (!impl_) return;
    if (!points_xyz.defined()) return;

    py::gil_scoped_acquire gil;

    auto p_cpu = points_xyz.contiguous().to(torch::kCPU);
    TORCH_CHECK(p_cpu.dim() == 2 && p_cpu.size(1) == 3, "points_xyz must be [N,3]");

    auto ps = p_cpu.sizes();
    std::vector<ssize_t> p_shape{ps[0], ps[1]};
    std::vector<ssize_t> p_strides{
        static_cast<ssize_t>(sizeof(float) * ps[1]),
        static_cast<ssize_t>(sizeof(float))
    };

    py::array points_np(py::buffer_info(
        p_cpu.data_ptr<float>(),
        sizeof(float),
        py::format_descriptor<float>::format(),
        2,
        p_shape,
        p_strides
    ));

    py::object colors_np = py::none();
    if (colors.defined() && colors.numel() > 0) {
        auto c_cpu = colors.contiguous().to(torch::kCPU);
        TORCH_CHECK(c_cpu.dim() == 2, "colors must be 2D [N,3] or [N,4]");
        TORCH_CHECK(c_cpu.size(0) == p_cpu.size(0),
                    "colors must have same N as points_xyz");
        TORCH_CHECK(c_cpu.size(1) == 3 || c_cpu.size(1) == 4,
                    "colors must be [N,3] or [N,4]");

        auto cs = c_cpu.sizes();
        std::vector<ssize_t> c_shape{cs[0], cs[1]};
        std::vector<ssize_t> c_strides{
            static_cast<ssize_t>(c_cpu.element_size() * cs[1]),
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

    py::object labels_py = py::none();
    if (!labels.empty()) {
        if (static_cast<int64_t>(labels.size()) == p_cpu.size(0)) {
            py::list py_labels;
            for (const auto& label : labels) {
                py_labels.append(py::str(label));
            }
            labels_py = py_labels;
        } else {
            std::cerr << "[RERUN] visualizeDebugPoints3D labels length mismatch: labels="
                      << labels.size() << " points=" << p_cpu.size(0)
                      << " (ignoring labels)\n";
        }
    }

    try {
        impl_->visualizer.attr("visualize_points3d_recording")(
            py::str(recording_name),
            points_np,
            colors_np,
            radius,
            py::str(entity_path),
            iteration,
            labels_py
        );
    } catch (const py::error_already_set& e) {
        std::cerr << "[RERUN] Python error in visualizeDebugPoints3D: "
                  << e.what() << std::endl;
    }
}

void RerunVisualizerBridge::visualizeLineStrip3D(
    const torch::Tensor& points_xyz,
    const torch::Tensor& color_rgb,
    int iteration,
    const std::string& entity_path,
    float radius
) {
    ensureInitialized();
    if (!impl_) return;
    if (!points_xyz.defined() || points_xyz.numel() == 0) return;

    py::gil_scoped_acquire gil;

    auto p_cpu = points_xyz.contiguous().to(torch::kCPU);
    TORCH_CHECK(p_cpu.dim() == 2 && p_cpu.size(1) == 3, "points_xyz must be [N,3]");

    auto ps = p_cpu.sizes();
    std::vector<ssize_t> p_shape{ps[0], ps[1]};
    std::vector<ssize_t> p_strides{
        static_cast<ssize_t>(sizeof(float) * ps[1]),
        static_cast<ssize_t>(sizeof(float))
    };

    py::array points_np(py::buffer_info(
        p_cpu.data_ptr<float>(),
        sizeof(float),
        py::format_descriptor<float>::format(),
        2,
        p_shape,
        p_strides
    ));

    py::object color_np = py::none();
    if (color_rgb.defined() && color_rgb.numel() > 0) {
        auto c = color_rgb.contiguous().to(torch::kCPU).view({-1});
        TORCH_CHECK(c.numel() >= 3, "color_rgb must have at least 3 values");
        // pass as numpy [3]
        py::array_t<uint8_t> cu8(static_cast<py::ssize_t>(3));
        if (c.dtype() == torch::kUInt8) {
            auto cptr = c.data_ptr<uint8_t>();
            auto b = cu8.mutable_unchecked<1>();
            b(0)=cptr[0]; b(1)=cptr[1]; b(2)=cptr[2];
        } else {
            // assume float-ish [0,1], convert safely
            auto cf = c.to(torch::kFloat32);
            auto cptr = cf.data_ptr<float>();
            auto b = cu8.mutable_unchecked<1>();
            b(0)=uint8_t(std::max(0.f,std::min(1.f,cptr[0]))*255.f);
            b(1)=uint8_t(std::max(0.f,std::min(1.f,cptr[1]))*255.f);
            b(2)=uint8_t(std::max(0.f,std::min(1.f,cptr[2]))*255.f);
        }
        color_np = cu8;
    }

    impl_->visualizer.attr("visualize_linestrip3d")(
        points_np,
        color_np,
        radius,
        py::str(entity_path),
        iteration
    );
}

void RerunVisualizerBridge::visualizeScalar(
    double value,
    int iteration,
    const std::string& entity_path)
{
    ensureInitialized();
    if (!impl_) return;

    py::gil_scoped_acquire gil;
    impl_->visualizer.attr("visualize_scalar")(
        value,
        py::str(entity_path),
        iteration
    );
}

void RerunVisualizerBridge::visualizeDebugScalar(
    const std::string& recording_name,
    double value,
    int iteration,
    const std::string& entity_path)
{
    ensureInitialized();
    if (!impl_) return;

    py::gil_scoped_acquire gil;
    try {
        impl_->visualizer.attr("visualize_scalar_recording")(
            py::str(recording_name),
            value,
            py::str(entity_path),
            iteration
        );
    } catch (const py::error_already_set& e) {
        std::cerr << "[RERUN] Python error in visualizeDebugScalar: "
                  << e.what() << std::endl;
    }
}

void RerunVisualizerBridge::visualizeDebugGtSdfMesh(
    const std::string& recording_name,
    const std::string& gt_mesh_path,
    bool align_gt_to_slam,
    const std::string& gt_traj_path,
    int align_min_pairs,
    int iteration)
{
    ensureInitialized();
    if (!impl_ || gt_mesh_path.empty()) return;

    py::gil_scoped_acquire gil;
    try {
        impl_->visualizer.attr("visualize_gt_sdf_mesh_recording")(
            py::str(recording_name),
            py::str(gt_mesh_path),
            align_gt_to_slam,
            py::str(gt_traj_path),
            align_min_pairs,
            iteration,
            py::str("world/gt/mesh"));
    } catch (const py::error_already_set& e) {
        std::cerr << "[RERUN] Python error in visualizeDebugGtSdfMesh: "
                  << e.what() << std::endl;
    }
}

torch::Tensor RerunVisualizerBridge::computeGtSignedDistance(
    const torch::Tensor& points_xyz,
    const std::string& gt_mesh_path,
    bool align_gt_to_slam,
    const std::string& gt_traj_path,
    int align_min_pairs)
{
    ensureInitialized();
    if (!impl_ ||
        !points_xyz.defined() ||
        points_xyz.numel() == 0 ||
        points_xyz.dim() != 2 ||
        points_xyz.size(1) != 3) {
        return torch::Tensor();
    }

    py::gil_scoped_acquire gil;
    try {
        torch::Tensor points_cpu =
            points_xyz.contiguous().to(torch::kCPU).to(torch::kFloat32);
        std::vector<ssize_t> shape = {
            static_cast<ssize_t>(points_cpu.size(0)),
            static_cast<ssize_t>(points_cpu.size(1))};
        std::vector<ssize_t> strides = {
            static_cast<ssize_t>(points_cpu.stride(0) * points_cpu.element_size()),
            static_cast<ssize_t>(points_cpu.stride(1) * points_cpu.element_size())};
        py::array points_np(py::buffer_info(
            points_cpu.data_ptr<float>(),
            sizeof(float),
            py::format_descriptor<float>::format(),
            2,
            shape,
            strides));

        py::object result = impl_->visualizer.attr("compute_gt_signed_distance")(
            points_np,
            py::str(gt_mesh_path),
            align_gt_to_slam,
            py::str(gt_traj_path),
            align_min_pairs);
        if (result.is_none()) {
            return torch::Tensor();
        }

        py::array_t<float, py::array::c_style | py::array::forcecast> sdf_np =
            result.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        py::buffer_info info = sdf_np.request();
        if (info.ndim != 1 || info.shape.empty()) {
            return torch::Tensor();
        }
        const int64_t N = static_cast<int64_t>(info.shape[0]);
        if (N != points_cpu.size(0)) {
            std::cerr << "[RERUN] computeGtSignedDistance returned " << N
                      << " values for " << points_cpu.size(0) << " points.\n";
            return torch::Tensor();
        }
        return torch::from_blob(
                   info.ptr,
                   {N},
                   torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
            .clone();
    } catch (const py::error_already_set& e) {
        std::cerr << "[RERUN] Python error in computeGtSignedDistance: "
                  << e.what() << std::endl;
    }
    return torch::Tensor();
}

torch::Tensor RerunVisualizerBridge::computeGtSurfaceDistance(
    const torch::Tensor& points_xyz,
    const std::string& gt_mesh_path,
    bool align_gt_to_slam,
    const std::string& gt_traj_path,
    int align_min_pairs)
{
    ensureInitialized();
    if (!impl_) return torch::Tensor();
    if (!points_xyz.defined() || points_xyz.numel() == 0) return torch::Tensor();

    py::gil_scoped_acquire gil;
    try {
        torch::Tensor pts_cpu =
            points_xyz.contiguous().to(torch::kCPU).to(torch::kFloat32);
        TORCH_CHECK(pts_cpu.dim() == 2 && pts_cpu.size(1) == 3,
                    "points_xyz must be [N,3]");

        const int64_t N = pts_cpu.size(0);
        std::vector<ssize_t> shape{static_cast<ssize_t>(N), 3};
        std::vector<ssize_t> strides{
            static_cast<ssize_t>(pts_cpu.stride(0) * pts_cpu.element_size()),
            static_cast<ssize_t>(pts_cpu.stride(1) * pts_cpu.element_size())
        };
        py::array pts_np(py::buffer_info(
            pts_cpu.data_ptr<float>(),
            sizeof(float),
            py::format_descriptor<float>::format(),
            2,
            shape,
            strides));

        py::object result = impl_->visualizer.attr("compute_gt_surface_distance")(
            pts_np,
            py::str(gt_mesh_path),
            align_gt_to_slam,
            py::str(gt_traj_path),
            align_min_pairs);
        if (result.is_none()) {
            return torch::Tensor();
        }

        py::array arr = py::cast<py::array>(result);
        py::buffer_info info = arr.request();
        if (info.ndim != 1 || info.shape[0] != N) {
            std::cerr << "[RERUN] computeGtSurfaceDistance returned "
                      << info.shape[0] << " values, expected " << N << std::endl;
            return torch::Tensor();
        }
        return torch::from_blob(
                   info.ptr,
                   {N},
                   torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
            .clone();
    } catch (const py::error_already_set& e) {
        std::cerr << "[RERUN] Python error in computeGtSurfaceDistance: "
                  << e.what() << std::endl;
    }
    return torch::Tensor();
}

torch::Tensor RerunVisualizerBridge::computeGtProjectiveSdf(
    const torch::Tensor& points_xyz,
    const torch::Tensor& Tcw,
    float fx,
    float fy,
    float cx,
    float cy,
    int width,
    int height,
    const std::string& gt_mesh_path,
    bool align_gt_to_slam,
    const std::string& gt_traj_path,
    int align_min_pairs)
{
    ensureInitialized();
    if (!impl_ ||
        !points_xyz.defined() ||
        points_xyz.numel() == 0 ||
        points_xyz.dim() != 2 ||
        points_xyz.size(1) != 3 ||
        !Tcw.defined() ||
        Tcw.numel() != 16) {
        return torch::Tensor();
    }

    py::gil_scoped_acquire gil;
    try {
        struct TensorArray {
            torch::Tensor tensor;
            py::array array;
        };
        auto make_array = [](const torch::Tensor& tensor) -> TensorArray {
            TensorArray out;
            out.tensor = tensor.contiguous().to(torch::kCPU).to(torch::kFloat32);
            std::vector<ssize_t> shape;
            std::vector<ssize_t> strides;
            shape.reserve(static_cast<size_t>(out.tensor.dim()));
            strides.reserve(static_cast<size_t>(out.tensor.dim()));
            for (int64_t d = 0; d < out.tensor.dim(); ++d) {
                shape.push_back(static_cast<ssize_t>(out.tensor.size(d)));
                strides.push_back(static_cast<ssize_t>(out.tensor.stride(d) * out.tensor.element_size()));
            }
            out.array = py::array(py::buffer_info(
                out.tensor.data_ptr<float>(),
                sizeof(float),
                py::format_descriptor<float>::format(),
                static_cast<ssize_t>(out.tensor.dim()),
                shape,
                strides));
            return out;
        };

        TensorArray points_np = make_array(points_xyz);
        TensorArray Tcw_np = make_array(Tcw.reshape({4, 4}));

        py::object result = impl_->visualizer.attr("compute_gt_projective_sdf")(
            points_np.array,
            Tcw_np.array,
            fx,
            fy,
            cx,
            cy,
            width,
            height,
            py::str(gt_mesh_path),
            align_gt_to_slam,
            py::str(gt_traj_path),
            align_min_pairs);
        if (result.is_none()) {
            return torch::Tensor();
        }

        py::array_t<float, py::array::c_style | py::array::forcecast> sdf_np =
            result.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        py::buffer_info info = sdf_np.request();
        if (info.ndim != 1 || info.shape.empty()) {
            return torch::Tensor();
        }
        const int64_t N = static_cast<int64_t>(info.shape[0]);
        if (N != points_np.tensor.size(0)) {
            std::cerr << "[RERUN] computeGtProjectiveSdf returned " << N
                      << " values for " << points_np.tensor.size(0) << " points.\n";
            return torch::Tensor();
        }
        return torch::from_blob(
                   info.ptr,
                   {N},
                   torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
            .clone();
    } catch (const py::error_already_set& e) {
        std::cerr << "[RERUN] Python error in computeGtProjectiveSdf: "
                  << e.what() << std::endl;
    }
    return torch::Tensor();
}

} // namespace sv
