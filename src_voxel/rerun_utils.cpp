#include "include_voxel/rerun_utils.h"

#include <cstring>

#include <Eigen/Geometry>          // for Eigen::Quaternionf
#include <opencv2/imgproc.hpp>     // for cv::cvtColor, cv::COLOR_BGR2RGB

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

struct RerunVisualizerBridge::PyImpl {
    py::object visualizer;  // instance of RerunVisualizer
};

RerunVisualizerBridge& RerunVisualizerBridge::instance() {
    static RerunVisualizerBridge inst;
    return inst;
}

RerunVisualizerBridge::RerunVisualizerBridge() = default;

void RerunVisualizerBridge::init(const std::string& app_id, bool spawn_viewer) {
    if (!enabled_) {
        return;
    }
    if (initialized_) {
        return;
    }

    py::gil_scoped_acquire gil;

    // Import our Python visualizer
    py::module_ mod = py::module_::import(
        "scripts_voxel.python_rerun_bridge.visualizer_wrapper"
    );
    py::object cls = mod.attr("RerunVisualizer");
    py::object vis = cls(app_id, spawn_viewer);

    impl_ = new PyImpl();
    impl_->visualizer = vis;

    initialized_ = true;
}

void RerunVisualizerBridge::ensureInitialized() {
    if (!enabled_) {
        return;
    }
    if (!initialized_) {
        init();
    }
}

void RerunVisualizerBridge::saveRecording(const std::string& path) {
    ensureInitialized();
    if (!impl_) return;

    py::gil_scoped_acquire gil;
    try {
        impl_->visualizer.attr("save_recording")(py::str(path));
    } catch (const py::error_already_set& e) {
        std::cerr << "[RERUN] Python error in saveRecording: "
                  << e.what() << std::endl;
    }
}

void RerunVisualizerBridge::visualizeCamera(
    const Eigen::Matrix4f& T_W_C,
    const cv::Mat& img_bgr_or_gray,
    const std::vector<Eigen::Vector2f>& kps_uv,
    const std::vector<int>& track_ids,
    int iteration,
    int keyframe_id,
    float fx, float fy,
    float cx, float cy
) {
    ensureInitialized();
    if (!impl_) return;
    if (img_bgr_or_gray.empty()) {
        std::cout << "[RERUN] visualizeCamera: input image is empty\n";
        return;
    }

    // std::cout << "[RERUN] Input cv::Mat: "
    //           << "size=" << img_bgr_or_gray.cols << "x" << img_bgr_or_gray.rows
    //           << ", channels=" << img_bgr_or_gray.channels()
    //           << ", type=" << cvTypeToString(img_bgr_or_gray.type())
    //           << ", step=" << img_bgr_or_gray.step
    //           << ", isContinuous=" << (img_bgr_or_gray.isContinuous() ? "true" : "false")
    //           << std::endl;

    // 1) Ensure RGB, contiguous
    cv::Mat img_rgb;
    if (img_bgr_or_gray.channels() == 3) {
        cv::cvtColor(img_bgr_or_gray, img_rgb, cv::COLOR_BGR2RGB);
    } else if (img_bgr_or_gray.channels() == 1) {
        cv::cvtColor(img_bgr_or_gray, img_rgb, cv::COLOR_GRAY2RGB);
    } else {
        img_rgb = img_bgr_or_gray.clone();
    }

    if (!img_rgb.isContinuous()) {
        img_rgb = img_rgb.clone();
    }

    // std::cout << "[RERUN] After BGR->RGB conversion: "
    //           << "size=" << img_rgb.cols << "x" << img_rgb.rows
    //           << ", channels=" << img_rgb.channels()
    //           << ", type=" << cvTypeToString(img_rgb.type())
    //           << ", step=" << img_rgb.step
    //           << ", isContinuous=" << (img_rgb.isContinuous() ? "true" : "false")
    //           << std::endl;

    // 2) Convert to 8-bit for visualization if needed
    cv::Mat img_u8;
    if (img_rgb.type() == CV_8UC3) {
        img_u8 = img_rgb;  // already fine
    } else if (img_rgb.type() == CV_32FC3) {
        // Assume standard normalized [0,1] floats.
        // Clamp and scale to [0,255].
        cv::Mat img_clamped;
        cv::min(img_rgb, 1.0, img_clamped);   // clamp upper bound
        cv::max(img_clamped, 0.0, img_clamped); // clamp lower bound
        img_clamped.convertTo(img_u8, CV_8UC3, 255.0);
    } else {
        std::cerr << "[RERUN] Unexpected img_rgb type; converting to CV_8UC3 with scale=255.\n";
        img_rgb.convertTo(img_u8, CV_8UC3, 255.0);
    }

    if (!img_u8.isContinuous()) {
        img_u8 = img_u8.clone();
    }

    // std::cout << "[RERUN] For Rerun: "
    //           << "size=" << img_u8.cols << "x" << img_u8.rows
    //           << ", channels=" << img_u8.channels()
    //           << ", type=" << cvTypeToString(img_u8.type())
    //           << ", step=" << img_u8.step
    //           << ", isContinuous=" << (img_u8.isContinuous() ? "true" : "false")
    //           << std::endl;

    const int H = img_u8.rows;
    const int W = img_u8.cols;
    const int C = img_u8.channels();
    const size_t row_stride_bytes = img_u8.step;

    // 3) Pose → translation + quaternion
    Eigen::Vector3f t = T_W_C.block<3,1>(0, 3);
    Eigen::Matrix3f R = T_W_C.block<3,3>(0, 0);
    Eigen::Quaternionf q(R);

    py::gil_scoped_acquire gil;

    py::array_t<float> t_np({3});
    py::array_t<float> q_np({4});
    {
        auto bt = t_np.mutable_unchecked<1>();
        auto bq = q_np.mutable_unchecked<1>();
        for (int i = 0; i < 3; ++i) bt(i) = t(i);
        bq(0) = q.x();
        bq(1) = q.y();
        bq(2) = q.z();
        bq(3) = q.w();
    }

    // 4) Image → numpy using explicit strides (H, W, C) in bytes
    std::vector<ssize_t> shape   = { H, W, C };
    std::vector<ssize_t> strides = {
        static_cast<ssize_t>(row_stride_bytes), // bytes between rows
        static_cast<ssize_t>(C),                // bytes between cols (3 uint8 per pixel)
        static_cast<ssize_t>(1)                 // bytes between channels
    };

    py::buffer_info img_buf(
        img_u8.data,
        sizeof(uint8_t),
        py::format_descriptor<uint8_t>::format(),
        3,
        shape,
        strides,
        false  // not readonly
    );
    py::array img_np(img_buf);

    try {
        impl_->visualizer.attr("visualize_cuvslam")(
            t_np,
            q_np,
            img_np,
            py::none(),  // points_uv
            py::none(),   // track_ids
            iteration,
            keyframe_id,
            fx,
            fy,
            cx,
            cy
        );
    } catch (const py::error_already_set& e) {
        std::cerr << "[RERUN] Python error in visualizeCamera: "
                  << e.what() << std::endl;
    }
}

void RerunVisualizerBridge::visualizeVoxelBoxes(
    const torch::Tensor& centers,
    const torch::Tensor& sizes,
    const torch::Tensor& colors,
    int iteration,
    const std::string& entity_path
) {
    ensureInitialized();
    if (!impl_) return;
    if (!centers.defined() || centers.numel() == 0) return;

    py::gil_scoped_acquire gil;

    auto c_cpu = centers.contiguous().to(torch::kCPU);
    TORCH_CHECK(c_cpu.dim() == 2 && c_cpu.size(1) == 3, "centers must be [N,3]");

    // sizes → half_sizes [N,3]
    torch::Tensor half_sizes_cpu;
    {
        auto s = sizes;
        TORCH_CHECK(s.defined(), "sizes must be defined");
        s = s.contiguous().to(torch::kCPU);

        if (s.dim() == 1) {
            // [N] → [N,3] isotropic
            s = s.view({s.size(0), 1}).expand({s.size(0), 3});
        } else if (s.dim() == 2 && s.size(1) == 1) {
            s = s.expand({s.size(0), 3});
        } else {
            TORCH_CHECK(s.dim() == 2 && s.size(1) == 3,
                        "sizes must be [N], [N,1], or [N,3]");
        }
        half_sizes_cpu = 0.5f * s;
    }

    auto c_sizes = c_cpu.sizes();
    std::vector<ssize_t> centers_shape{c_sizes[0], c_sizes[1]};
    std::vector<ssize_t> centers_strides{
        static_cast<ssize_t>(sizeof(float) * c_sizes[1]),
        static_cast<ssize_t>(sizeof(float))
    };

    py::array centers_np(py::buffer_info(
        c_cpu.data_ptr<float>(),
        sizeof(float),
        py::format_descriptor<float>::format(),
        2,
        centers_shape,
        centers_strides
    ));

    auto h_sizes = half_sizes_cpu;
    auto h_sizes_sizes = h_sizes.sizes();
    std::vector<ssize_t> hs_shape{h_sizes_sizes[0], h_sizes_sizes[1]};
    std::vector<ssize_t> hs_strides{
        static_cast<ssize_t>(sizeof(float) * h_sizes_sizes[1]),
        static_cast<ssize_t>(sizeof(float))
    };

    py::array half_sizes_np(py::buffer_info(
        h_sizes.data_ptr<float>(),
        sizeof(float),
        py::format_descriptor<float>::format(),
        2,
        hs_shape,
        hs_strides
    ));

    py::object colors_np = py::none();
    if (colors.defined() && colors.numel() > 0) {
        auto col_cpu   = colors.contiguous().to(torch::kCPU);
        auto col_sizes = col_cpu.sizes();
        TORCH_CHECK(
            col_cpu.dim() == 2 && (col_cpu.size(1) == 3 || col_cpu.size(1) == 4),
            "colors must be [N,3] or [N,4]"
        );

        std::vector<ssize_t> col_shape{col_sizes[0], col_sizes[1]};
        std::vector<ssize_t> col_strides{
            static_cast<ssize_t>(col_cpu.element_size() * col_sizes[1]),
            static_cast<ssize_t>(col_cpu.element_size())
        };

        py::array tmp(py::buffer_info(
            col_cpu.data_ptr(),
            col_cpu.element_size(),
            col_cpu.dtype() == torch::kUInt8
                ? py::format_descriptor<uint8_t>::format()
                : py::format_descriptor<float>::format(),
            2,
            col_shape,
            col_strides
        ));
        colors_np = tmp;
    }

    impl_->visualizer.attr("visualize_voxels_boxes")(
        centers_np, half_sizes_np, colors_np,
        py::str(entity_path),              // entity_path
        1000000,                      // max_boxes
        iteration                    // iteration
    );
}

void RerunVisualizerBridge::visualizeNvbloxMesh(
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
        impl_->visualizer.attr("visualize_nvblox")(
            vertices_np,
            colors_np,
            triangles_np,
            iteration
        );
    } catch (const py::error_already_set& e) {
        std::cerr << "[RERUN] Python error in visualizeNvbloxMesh: "
                  << e.what() << std::endl;
    }
}

void RerunVisualizerBridge::visualizeNvbloxPlyMesh(
    const std::string& ply_path,
    int iteration,
    const std::string& entity_path
) {
    ensureInitialized();
    if (!impl_) return;

    py::gil_scoped_acquire gil;
    try {
        impl_->visualizer.attr("visualize_nvblox_ply")(
            py::str(ply_path),
            iteration,
            py::str(entity_path)
        );
    } catch (const py::error_already_set& e) {
        std::cerr << "[RERUN] Python error in visualizeNvbloxPlyMesh: "
                  << e.what() << std::endl;
    }
}

void RerunVisualizerBridge::visualizeSVRasterMesh(
    const torch::Tensor& centers,
    const torch::Tensor& sizes,
    const torch::Tensor& colors,
    int iteration
) {
    ensureInitialized();
    if (!impl_) return;
    if (!centers.defined() || centers.numel() == 0) return;

    py::gil_scoped_acquire gil;

    auto c_cpu = centers.contiguous().to(torch::kCPU);
    TORCH_CHECK(c_cpu.dim() == 2 && c_cpu.size(1) == 3, "centers must be [N,3]");

    // centers → numpy
    auto c_sizes = c_cpu.sizes();
    std::vector<ssize_t> centers_shape{c_sizes[0], c_sizes[1]};
    std::vector<ssize_t> centers_strides{
        static_cast<ssize_t>(sizeof(float) * c_sizes[1]),
        static_cast<ssize_t>(sizeof(float))
    };
    py::array centers_np(py::buffer_info(
        c_cpu.data_ptr<float>(),
        sizeof(float),
        py::format_descriptor<float>::format(),
        2,
        centers_shape,
        centers_strides
    ));

    // sizes → half_sizes [N,3], then numpy
    torch::Tensor half_sizes_cpu;
    {
        auto s = sizes;
        TORCH_CHECK(s.defined(), "sizes must be defined");
        s = s.contiguous().to(torch::kCPU);

        if (s.dim() == 1) {
            s = s.view({s.size(0), 1}).expand({s.size(0), 3});
        } else if (s.dim() == 2 && s.size(1) == 1) {
            s = s.expand({s.size(0), 3});
        } else {
            TORCH_CHECK(s.dim() == 2 && s.size(1) == 3,
                        "sizes must be [N], [N,1], or [N,3]");
        }
        half_sizes_cpu = 0.5f * s;
    }

    auto hs_sizes = half_sizes_cpu.sizes();
    std::vector<ssize_t> hs_shape{hs_sizes[0], hs_sizes[1]};
    std::vector<ssize_t> hs_strides{
        static_cast<ssize_t>(sizeof(float) * hs_sizes[1]),
        static_cast<ssize_t>(sizeof(float))
    };
    py::array half_sizes_np(py::buffer_info(
        half_sizes_cpu.data_ptr<float>(),
        sizeof(float),
        py::format_descriptor<float>::format(),
        2,
        hs_shape,
        hs_strides
    ));

    py::object colors_np = py::none();
    if (colors.defined() && colors.numel() > 0) {
        auto col_cpu = colors.contiguous().to(torch::kCPU);
        TORCH_CHECK(col_cpu.dim() == 2 && col_cpu.size(1) == 3,
                    "colors must be [N,3]");

        auto col_sizes = col_cpu.sizes();
        std::vector<ssize_t> col_shape{col_sizes[0], col_sizes[1]};
        std::vector<ssize_t> col_strides{
            static_cast<ssize_t>(col_cpu.element_size() * col_sizes[1]),
            static_cast<ssize_t>(col_cpu.element_size())
        };

        py::array tmp(py::buffer_info(
            col_cpu.data_ptr(),
            col_cpu.element_size(),
            col_cpu.dtype() == torch::kUInt8
                ? py::format_descriptor<uint8_t>::format()
                : py::format_descriptor<float>::format(),
            2,
            col_shape,
            col_strides
        ));
        colors_np = tmp;
    }

    try {
        impl_->visualizer.attr("visualize_voxels_mesh")(
            centers_np,
            half_sizes_np,
            colors_np,
            20000,
            iteration
        );
    } catch (const py::error_already_set& e) {
        std::cerr << "[RERUN] Python error in visualizeSVRasterMesh: "
                  << e.what() << std::endl;
    }
}

void RerunVisualizerBridge::visualizePoints3D(
    const torch::Tensor& points_xyz,
    const torch::Tensor& colors,
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

    impl_->visualizer.attr("visualize_points3d")(
        points_np,
        colors_np,
        radius,
        py::str(entity_path),
        iteration
    );
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
        py::array_t<uint8_t> cu8({3});
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

} // namespace sv
