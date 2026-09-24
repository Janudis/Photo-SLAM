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
        "rerun_visualization.visualizer_wrapper"
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

} // namespace sv
