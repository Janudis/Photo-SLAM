#include "include_voxel/rerun_utils.h"

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
    if (!Py_IsInitialized()) {
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

bool RerunVisualizerBridge::deferDebugCall(std::function<void()> call) {
    std::lock_guard<std::mutex> lock(deferred_debug_mutex_);
    if (initialized_ || flushing_deferred_debug_calls_) {
        return false;
    }
    deferred_debug_calls_.push_back(std::move(call));
    return true;
}

void RerunVisualizerBridge::flushDeferredDebugCalls() {
    std::vector<std::function<void()>> calls;
    {
        std::lock_guard<std::mutex> lock(deferred_debug_mutex_);
        flushing_deferred_debug_calls_ = true;
        calls.swap(deferred_debug_calls_);
    }
    for (auto& call : calls) {
        call();
    }
    {
        std::lock_guard<std::mutex> lock(deferred_debug_mutex_);
        flushing_deferred_debug_calls_ = false;
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

void RerunVisualizerBridge::saveDebugRecording(
    const std::string& recording_name,
    const std::string& path)
{
    ensureInitialized();
    if (!impl_) return;
    flushDeferredDebugCalls();

    py::gil_scoped_acquire gil;
    try {
        impl_->visualizer.attr("save_debug_recording")(
            py::str(recording_name),
            py::str(path));
    } catch (const py::error_already_set& e) {
        std::cerr << "[RERUN] Python error in saveDebugRecording: "
                  << e.what() << std::endl;
    }
}

void RerunVisualizerBridge::visualizeCamera(
    const Eigen::Matrix4f& T_W_C,
    const cv::Mat& img_rgb_or_gray,
    const std::vector<Eigen::Vector2f>& kps_uv,
    const std::vector<int>& track_ids,
    int iteration,
    int keyframe_id,
    float fx, float fy,
    float cx, float cy,
    int source_frame_id
) {
    ensureInitialized();
    if (!impl_) return;
    if (img_rgb_or_gray.empty()) {
        std::cout << "[RERUN] visualizeCamera: input image is empty\n";
        return;
    }

    // std::cout << "[RERUN] Input cv::Mat: "
    //           << "size=" << img_rgb_or_gray.cols << "x" << img_rgb_or_gray.rows
    //           << ", channels=" << img_rgb_or_gray.channels()
    //           << ", type=" << cvTypeToString(img_rgb_or_gray.type())
    //           << ", step=" << img_rgb_or_gray.step
    //           << ", isContinuous=" << (img_rgb_or_gray.isContinuous() ? "true" : "false")
    //           << std::endl;

    // 1) Ensure RGB, contiguous. The mapper/examples convert OpenCV BGR input
    // to RGB before tracking, and VoxelKeyframe stores RGB throughout.
    cv::Mat img_rgb;
    if (img_rgb_or_gray.channels() == 3) {
        img_rgb = img_rgb_or_gray;
    } else if (img_rgb_or_gray.channels() == 1) {
        cv::cvtColor(img_rgb_or_gray, img_rgb, cv::COLOR_GRAY2RGB);
    } else {
        img_rgb = img_rgb_or_gray.clone();
    }

    if (!img_rgb.isContinuous()) {
        img_rgb = img_rgb.clone();
    }

    // std::cout << "[RERUN] RGB image for Rerun: "
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

    const py::tuple t_py = py::make_tuple(t.x(), t.y(), t.z());
    const py::tuple q_py = py::make_tuple(q.x(), q.y(), q.z(), q.w());

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
            t_py,
            q_py,
            img_np,
            py::none(),  // points_uv
            py::none(),   // track_ids
            iteration,
            keyframe_id,
            fx,
            fy,
            cx,
            cy,
            source_frame_id
        );
    } catch (const py::error_already_set& e) {
        std::cerr << "[RERUN] Python error in visualizeCamera: "
                  << e.what() << std::endl;
    }
}

void RerunVisualizerBridge::visualizeDebugCamera(
    const std::string& recording_name,
    const Eigen::Matrix4f& T_W_C,
    const cv::Mat& img_rgb_or_gray,
    const std::vector<Eigen::Vector2f>&,
    const std::vector<int>&,
    int iteration,
    int keyframe_id,
    float fx, float fy,
    float cx, float cy,
    int source_frame_id
) {
    if (!enabled_) return;
    cv::Mat image_copy = img_rgb_or_gray.clone();
    if (deferDebugCall(
            [this,
             recording_name,
             T_W_C,
             image_copy,
             iteration,
             keyframe_id,
             fx,
             fy,
             cx,
             cy,
             source_frame_id]() {
                this->visualizeDebugCamera(
                    recording_name,
                    T_W_C,
                    image_copy,
                    {},
                    {},
                    iteration,
                    keyframe_id,
                    fx,
                    fy,
                    cx,
                    cy,
                    source_frame_id);
            })) {
        return;
    }
    ensureInitialized();
    if (!impl_) return;
    if (img_rgb_or_gray.empty()) {
        std::cout << "[RERUN] visualizeDebugCamera: input image is empty\n";
        return;
    }

    cv::Mat img_rgb;
    if (img_rgb_or_gray.channels() == 3) {
        img_rgb = img_rgb_or_gray;
    } else if (img_rgb_or_gray.channels() == 1) {
        cv::cvtColor(img_rgb_or_gray, img_rgb, cv::COLOR_GRAY2RGB);
    } else {
        img_rgb = img_rgb_or_gray.clone();
    }
    if (!img_rgb.isContinuous()) {
        img_rgb = img_rgb.clone();
    }

    cv::Mat img_u8;
    if (img_rgb.type() == CV_8UC3) {
        img_u8 = img_rgb;
    } else if (img_rgb.type() == CV_32FC3) {
        cv::Mat img_clamped;
        cv::min(img_rgb, 1.0, img_clamped);
        cv::max(img_clamped, 0.0, img_clamped);
        img_clamped.convertTo(img_u8, CV_8UC3, 255.0);
    } else {
        img_rgb.convertTo(img_u8, CV_8UC3, 255.0);
    }
    if (!img_u8.isContinuous()) {
        img_u8 = img_u8.clone();
    }

    const int H = img_u8.rows;
    const int W = img_u8.cols;
    const int C = img_u8.channels();
    const size_t row_stride_bytes = img_u8.step;

    Eigen::Vector3f t = T_W_C.block<3,1>(0, 3);
    Eigen::Matrix3f R = T_W_C.block<3,3>(0, 0);
    Eigen::Quaternionf q(R);

    py::gil_scoped_acquire gil;

    const py::tuple t_py = py::make_tuple(t.x(), t.y(), t.z());
    const py::tuple q_py = py::make_tuple(q.x(), q.y(), q.z(), q.w());

    std::vector<ssize_t> shape = {H, W, C};
    std::vector<ssize_t> strides = {
        static_cast<ssize_t>(row_stride_bytes),
        static_cast<ssize_t>(C),
        static_cast<ssize_t>(1)
    };
    py::array img_np(py::buffer_info(
        img_u8.data,
        sizeof(uint8_t),
        py::format_descriptor<uint8_t>::format(),
        3,
        shape,
        strides,
        false));

    try {
        impl_->visualizer.attr("visualize_cuvslam_recording")(
            py::str(recording_name),
            t_py,
            q_py,
            img_np,
            py::none(),
            py::none(),
            iteration,
            keyframe_id,
            fx,
            fy,
            cx,
            cy,
            source_frame_id);
    } catch (const py::error_already_set& e) {
        std::cerr << "[RERUN] Python error in visualizeDebugCamera: "
                  << e.what() << std::endl;
    }
}

void RerunVisualizerBridge::visualizeDebugCameraPose(
    const std::string& recording_name,
    const Eigen::Matrix4f& T_W_C,
    int iteration,
    int keyframe_id
) {
    if (!enabled_) return;
    if (deferDebugCall(
            [this, recording_name, T_W_C, iteration, keyframe_id]() {
                this->visualizeDebugCameraPose(
                    recording_name, T_W_C, iteration, keyframe_id);
            })) {
        return;
    }
    ensureInitialized();
    if (!impl_) return;

    const Eigen::Vector3f t = T_W_C.block<3, 1>(0, 3);
    const Eigen::Quaternionf q(T_W_C.block<3, 3>(0, 0));

    py::gil_scoped_acquire gil;
    const py::tuple t_py = py::make_tuple(t.x(), t.y(), t.z());
    const py::tuple q_py = py::make_tuple(q.x(), q.y(), q.z(), q.w());

    try {
        impl_->visualizer.attr("visualize_camera_pose_recording")(
            py::str(recording_name),
            t_py,
            q_py,
            iteration,
            keyframe_id);
    } catch (const py::error_already_set& e) {
        std::cerr << "[RERUN] Python error in visualizeDebugCameraPose: "
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
    if (!centers.defined()) return;

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

void RerunVisualizerBridge::visualizeDebugVoxelBoxes(
    const std::string& recording_name,
    const torch::Tensor& centers,
    const torch::Tensor& sizes,
    const torch::Tensor& colors,
    int iteration,
    const std::string& entity_path
) {
    ensureInitialized();
    if (!impl_) return;
    if (!centers.defined()) return;

    py::gil_scoped_acquire gil;

    auto c_cpu = centers.contiguous().to(torch::kCPU);
    TORCH_CHECK(c_cpu.dim() == 2 && c_cpu.size(1) == 3, "centers must be [N,3]");

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
        auto col_cpu = colors.contiguous().to(torch::kCPU);
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

    try {
        impl_->visualizer.attr("visualize_voxels_boxes_recording")(
            py::str(recording_name),
            centers_np,
            half_sizes_np,
            colors_np,
            py::str(entity_path),
            1000000,
            iteration,
            py::none());
    } catch (const py::error_already_set& e) {
        std::cerr << "[RERUN] Python error in visualizeDebugVoxelBoxes: "
                  << e.what() << std::endl;
    }
}

void RerunVisualizerBridge::visualizeDebugVoxelGridMap(
    const std::string& recording_name,
    const torch::Tensor& centers,
    const torch::Tensor& sizes,
    const torch::Tensor& levels,
    const torch::Tensor& colors,
    const torch::Tensor& grid_origin,
    int iteration,
    const std::string& entity_path,
    float opacity
) {
    if (!enabled_) return;
    if (!centers.defined() || !sizes.defined() ||
        !levels.defined() || !grid_origin.defined()) {
        return;
    }
    auto centers_copy =
        centers.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto sizes_copy =
        sizes.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto levels_copy =
        levels.detach().to(torch::kCPU).to(torch::kInt32).contiguous();
    auto colors_copy = colors.defined()
        ? colors.detach().to(torch::kCPU).contiguous()
        : torch::Tensor();
    auto origin_copy =
        grid_origin.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (deferDebugCall(
            [this,
             recording_name,
             centers_copy,
             sizes_copy,
             levels_copy,
             colors_copy,
             origin_copy,
             iteration,
             entity_path,
             opacity]() {
                this->visualizeDebugVoxelGridMap(
                    recording_name,
                    centers_copy,
                    sizes_copy,
                    levels_copy,
                    colors_copy,
                    origin_copy,
                    iteration,
                    entity_path,
                    opacity);
            })) {
        return;
    }
    ensureInitialized();
    if (!impl_ || !centers.defined() || !sizes.defined() ||
        !levels.defined() || !grid_origin.defined()) {
        return;
    }

    py::gil_scoped_acquire gil;

    auto centers_cpu =
        centers.detach().to(torch::kCPU).to(torch::kFloat32).reshape({-1, 3}).contiguous();
    const int64_t num_voxels = centers_cpu.size(0);
    auto sizes_cpu =
        sizes.detach().to(torch::kCPU).to(torch::kFloat32).reshape({-1}).contiguous();
    auto levels_cpu =
        levels.detach().to(torch::kCPU).to(torch::kInt32).reshape({-1}).contiguous();
    auto origin_cpu =
        grid_origin.detach().to(torch::kCPU).to(torch::kFloat32).reshape({3}).contiguous();

    TORCH_CHECK(
        sizes_cpu.numel() == num_voxels,
        "VoxelGridMap sizes must have one value per center");
    TORCH_CHECK(
        levels_cpu.numel() == num_voxels,
        "VoxelGridMap levels must have one value per center");

    auto tensor_to_numpy_2d = [](torch::Tensor tensor) {
        const auto tensor_sizes = tensor.sizes();
        return py::array(py::buffer_info(
            tensor.data_ptr(),
            tensor.element_size(),
            tensor.scalar_type() == torch::kInt32
                ? py::format_descriptor<int32_t>::format()
                : py::format_descriptor<float>::format(),
            2,
            {
                static_cast<ssize_t>(tensor_sizes[0]),
                static_cast<ssize_t>(tensor_sizes[1])
            },
            {
                static_cast<ssize_t>(tensor.element_size() * tensor_sizes[1]),
                static_cast<ssize_t>(tensor.element_size())
            }
        ));
    };
    auto tensor_to_numpy_1d = [](torch::Tensor tensor) {
        return py::array(py::buffer_info(
            tensor.data_ptr(),
            tensor.element_size(),
            tensor.scalar_type() == torch::kInt32
                ? py::format_descriptor<int32_t>::format()
                : py::format_descriptor<float>::format(),
            1,
            {static_cast<ssize_t>(tensor.numel())},
            {static_cast<ssize_t>(tensor.element_size())}
        ));
    };

    py::array centers_np = tensor_to_numpy_2d(centers_cpu);
    py::array sizes_np = tensor_to_numpy_1d(sizes_cpu);
    py::array levels_np = tensor_to_numpy_1d(levels_cpu);
    py::array origin_np = tensor_to_numpy_1d(origin_cpu);

    py::object colors_np = py::none();
    torch::Tensor colors_cpu;
    if (colors.defined() && colors.numel() > 0) {
        colors_cpu = colors.detach().to(torch::kCPU).contiguous();
        TORCH_CHECK(
            colors_cpu.dim() == 2 &&
                colors_cpu.size(0) == num_voxels &&
                (colors_cpu.size(1) == 3 || colors_cpu.size(1) == 4),
            "VoxelGridMap colors must be [N,3] or [N,4]");
        if (colors_cpu.scalar_type() != torch::kUInt8) {
            colors_cpu = colors_cpu.to(torch::kFloat32).contiguous();
        }
        const auto color_sizes = colors_cpu.sizes();
        py::array colors_array(py::buffer_info(
            colors_cpu.data_ptr(),
            colors_cpu.element_size(),
            colors_cpu.scalar_type() == torch::kUInt8
                ? py::format_descriptor<uint8_t>::format()
                : py::format_descriptor<float>::format(),
            2,
            {
                static_cast<ssize_t>(color_sizes[0]),
                static_cast<ssize_t>(color_sizes[1])
            },
            {
                static_cast<ssize_t>(
                    colors_cpu.element_size() * color_sizes[1]),
                static_cast<ssize_t>(colors_cpu.element_size())
            }
        ));
        colors_np = colors_array;
    }

    try {
        impl_->visualizer.attr("visualize_voxel_grid_map_recording")(
            py::str(recording_name),
            centers_np,
            sizes_np,
            levels_np,
            colors_np,
            origin_np,
            py::str(entity_path),
            iteration,
            opacity);
    } catch (const py::error_already_set& e) {
        std::cerr << "[RERUN] Python error in visualizeDebugVoxelGridMap: "
                  << e.what() << std::endl;
    }
}

void RerunVisualizerBridge::visualizeDebugVoxelGridIndices(
    const std::string& recording_name,
    const std::vector<std::int32_t>& indices_xyz,
    const std::vector<float>& colors,
    const Eigen::Vector3f& grid_origin,
    float voxel_size,
    std::int32_t level,
    int iteration,
    const std::string& entity_path,
    float opacity
) {
    if (!enabled_ || indices_xyz.size() % 3 != 0 ||
        !std::isfinite(voxel_size) || voxel_size <= 0.0f) {
        return;
    }
    const std::size_t num_voxels = indices_xyz.size() / 3;
    const std::size_t color_channels = num_voxels > 0
        ? colors.size() / num_voxels
        : 0;
    if (!colors.empty() &&
        (colors.size() % std::max<std::size_t>(1, num_voxels) != 0 ||
         (color_channels != 3 && color_channels != 4))) {
        return;
    }

    if (deferDebugCall(
            [this,
             recording_name,
             indices_xyz,
             colors,
             grid_origin,
             voxel_size,
             level,
             iteration,
             entity_path,
             opacity]() {
                this->visualizeDebugVoxelGridIndices(
                    recording_name,
                    indices_xyz,
                    colors,
                    grid_origin,
                    voxel_size,
                    level,
                    iteration,
                    entity_path,
                    opacity);
            })) {
        return;
    }

    auto int_options =
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto float_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor indices = num_voxels == 0
        ? torch::empty({0, 3}, int_options)
        : torch::from_blob(
              const_cast<std::int32_t*>(indices_xyz.data()),
              {static_cast<int64_t>(num_voxels), 3},
              int_options).clone();
    torch::Tensor origin = torch::tensor(
        {grid_origin.x(), grid_origin.y(), grid_origin.z()},
        float_options);
    torch::Tensor centers =
        origin.view({1, 3}) +
        (indices.to(torch::kFloat32) + 0.5f) * voxel_size;
    torch::Tensor sizes = torch::full(
        {static_cast<int64_t>(num_voxels), 1},
        voxel_size,
        float_options);
    torch::Tensor levels = torch::full(
        {static_cast<int64_t>(num_voxels), 1},
        level,
        int_options);
    torch::Tensor color_tensor;
    if (!colors.empty()) {
        color_tensor = torch::from_blob(
            const_cast<float*>(colors.data()),
            {
                static_cast<int64_t>(num_voxels),
                static_cast<int64_t>(color_channels)
            },
            float_options).clone();
    }
    visualizeDebugVoxelGridMap(
        recording_name,
        centers,
        sizes,
        levels,
        color_tensor,
        origin,
        iteration,
        entity_path,
        opacity);
}

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
    const std::string& entity_path
) {
    ensureInitialized();
    if (!impl_) return;

    py::gil_scoped_acquire gil;
    try {
        impl_->visualizer.attr("visualize_ply_mesh_recording")(
            py::str(recording_name),
            py::str(ply_path),
            iteration,
            py::str(entity_path)
        );
    } catch (const py::error_already_set& e) {
        std::cerr << "[RERUN] Python error in visualizeDebugPlyMesh: "
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

void RerunVisualizerBridge::visualizeMapsFrameRecording(
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
    double normal_mean_deg)
{
    ensureInitialized();
    if (!impl_) return;

    auto mat_to_uint8_rgb_array = [](const cv::Mat& input) -> py::array_t<uint8_t> {
        if (input.empty()) {
            return py::array_t<uint8_t>({0, 0, 3});
        }

        cv::Mat rgb;
        if (input.channels() == 3) {
            rgb = input;
        } else if (input.channels() == 1) {
            cv::cvtColor(input, rgb, cv::COLOR_GRAY2RGB);
        } else if (input.channels() == 4) {
            cv::cvtColor(input, rgb, cv::COLOR_RGBA2RGB);
        } else {
            return py::array_t<uint8_t>({0, 0, 3});
        }

        cv::Mat u8;
        if (rgb.type() == CV_8UC3) {
            u8 = rgb;
        } else if (rgb.type() == CV_32FC3) {
            cv::Mat clamped;
            cv::min(rgb, 1.0, clamped);
            cv::max(clamped, 0.0, clamped);
            clamped.convertTo(u8, CV_8UC3, 255.0);
        } else {
            rgb.convertTo(u8, CV_8UC3);
        }

        if (!u8.isContinuous()) {
            u8 = u8.clone();
        }

        const int H = u8.rows;
        const int W = u8.cols;
        py::array_t<uint8_t> arr({H, W, 3});
        std::memcpy(arr.mutable_data(), u8.data, static_cast<size_t>(H) * static_cast<size_t>(W) * 3);
        return arr;
    };

    py::gil_scoped_acquire gil;
    try {
        impl_->visualizer.attr("visualize_maps_frame_recording")(
            py::str(recording_name),
            keyframe_id,
            iteration,
            mat_to_uint8_rgb_array(gt_rgb),
            mat_to_uint8_rgb_array(rendered_rgb),
            mat_to_uint8_rgb_array(rgb_error),
            mat_to_uint8_rgb_array(gt_depth_rgb),
            mat_to_uint8_rgb_array(rendered_depth_rgb),
            mat_to_uint8_rgb_array(depth_error_rgb),
            mat_to_uint8_rgb_array(depth_gap_rgb),
            mat_to_uint8_rgb_array(gt_normal_rgb),
            mat_to_uint8_rgb_array(rendered_normal_rgb),
            mat_to_uint8_rgb_array(normal_error_rgb),
            psnr,
            ssim,
            depth_l1_m,
            depth_gap_percent,
            normal_mean_deg);
    } catch (const py::error_already_set& e) {
        std::cerr << "[RERUN] Python error in visualizeMapsFrameRecording: "
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

void RerunVisualizerBridge::visualizeSdfVoxelsRecording(
    const std::string& recording_name,
    const torch::Tensor& centers,
    const torch::Tensor& sizes,
    const torch::Tensor& corner_points,
    const torch::Tensor& computed_sdf,
    const torch::Tensor& sdf_weights,
    const torch::Tensor& corner_density,
    const torch::Tensor& gt_sdf,
    const torch::Tensor& voxel_colors,
    const torch::Tensor& voxel_ids,
    const torch::Tensor& source_sdf_mask,
    const torch::Tensor& source_svraster_mask,
    int iteration,
    const std::string& gt_mesh_path,
    bool align_gt_to_slam,
    const std::string& gt_traj_path,
    int align_min_pairs,
    float surface_band_m,
    float min_weight,
    bool log_gt_mesh,
    const std::string& entity_path)
{
    ensureInitialized();
    if (!impl_) return;
    if (!centers.defined() || centers.numel() == 0 ||
        !sizes.defined() || sizes.numel() == 0 ||
        !corner_points.defined() || corner_points.numel() == 0 ||
        !computed_sdf.defined() || computed_sdf.numel() == 0 ||
        !sdf_weights.defined() || sdf_weights.numel() == 0 ||
        !corner_density.defined() || corner_density.numel() == 0 ||
        !gt_sdf.defined() || gt_sdf.numel() == 0) {
        return;
    }

    py::gil_scoped_acquire gil;
    try {
        struct TensorArray {
            torch::Tensor tensor;
            py::array array;
        };
        auto make_float_array = [](const torch::Tensor& tensor) -> TensorArray {
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

        TensorArray centers_np = make_float_array(centers);
        TensorArray sizes_np = make_float_array(sizes);
        TensorArray corner_points_np = make_float_array(corner_points);
        TensorArray computed_sdf_np = make_float_array(computed_sdf);
        TensorArray sdf_weights_np = make_float_array(sdf_weights);
        TensorArray corner_density_np = make_float_array(corner_density);
        TensorArray gt_sdf_np = make_float_array(gt_sdf);

        py::object voxel_colors_obj = py::none();
        TensorArray voxel_colors_np;
        if (voxel_colors.defined() && voxel_colors.numel() > 0) {
            voxel_colors_np = make_float_array(voxel_colors);
            voxel_colors_obj = voxel_colors_np.array;
        }

        py::object voxel_ids_obj = py::none();
        torch::Tensor voxel_ids_cpu;
        py::array voxel_ids_np;
        if (voxel_ids.defined() && voxel_ids.numel() > 0) {
            voxel_ids_cpu = voxel_ids.contiguous().to(torch::kCPU).to(torch::kInt64).view({-1});
            std::vector<ssize_t> shape{static_cast<ssize_t>(voxel_ids_cpu.size(0))};
            std::vector<ssize_t> strides{static_cast<ssize_t>(voxel_ids_cpu.element_size())};
            voxel_ids_np = py::array(py::buffer_info(
                voxel_ids_cpu.data_ptr<int64_t>(),
                sizeof(int64_t),
                py::format_descriptor<int64_t>::format(),
                1,
                shape,
                strides));
            voxel_ids_obj = voxel_ids_np;
        }

        py::object source_sdf_obj = py::none();
        py::object source_svraster_obj = py::none();
        torch::Tensor source_sdf_cpu;
        torch::Tensor source_svraster_cpu;
        py::array source_sdf_np;
        py::array source_svraster_np;
        if (source_sdf_mask.defined() && source_sdf_mask.numel() > 0) {
            source_sdf_cpu = source_sdf_mask.contiguous().to(torch::kCPU).to(torch::kBool).view({-1});
            std::vector<ssize_t> shape{static_cast<ssize_t>(source_sdf_cpu.size(0))};
            std::vector<ssize_t> strides{static_cast<ssize_t>(source_sdf_cpu.element_size())};
            source_sdf_np = py::array(py::buffer_info(
                source_sdf_cpu.data_ptr<bool>(),
                sizeof(bool),
                py::format_descriptor<bool>::format(),
                1,
                shape,
                strides));
            source_sdf_obj = source_sdf_np;
        }
        if (source_svraster_mask.defined() && source_svraster_mask.numel() > 0) {
            source_svraster_cpu = source_svraster_mask.contiguous().to(torch::kCPU).to(torch::kBool).view({-1});
            std::vector<ssize_t> shape{static_cast<ssize_t>(source_svraster_cpu.size(0))};
            std::vector<ssize_t> strides{static_cast<ssize_t>(source_svraster_cpu.element_size())};
            source_svraster_np = py::array(py::buffer_info(
                source_svraster_cpu.data_ptr<bool>(),
                sizeof(bool),
                py::format_descriptor<bool>::format(),
                1,
                shape,
                strides));
            source_svraster_obj = source_svraster_np;
        }

        impl_->visualizer.attr("visualize_sdf_voxels_recording")(
            py::str(recording_name),
            centers_np.array,
            sizes_np.array,
            corner_points_np.array,
            computed_sdf_np.array,
            sdf_weights_np.array,
            corner_density_np.array,
            gt_sdf_np.array,
            voxel_colors_obj,
            voxel_ids_obj,
            source_sdf_obj,
            source_svraster_obj,
            iteration,
            py::str(gt_mesh_path),
            align_gt_to_slam,
            py::str(gt_traj_path),
            align_min_pairs,
            surface_band_m,
            min_weight,
            log_gt_mesh,
            py::str(entity_path));
    } catch (const py::error_already_set& e) {
        std::cerr << "[RERUN] Python error in visualizeSdfVoxelsRecording: "
                  << e.what() << std::endl;
    }
}

} // namespace sv
