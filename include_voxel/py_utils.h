#pragma once
#include <pybind11/embed.h>
#include <pybind11/cast.h>        // <-- for py::cast
#include "mini_cam.h"
#include <pybind11/numpy.h>

namespace sv {
    template <typename T>
    pybind11::array tensor_to_numpy_typed(const torch::Tensor& t) {
        auto t_cpu = t.contiguous().to(torch::kCPU);
        std::vector<ssize_t> shape(t_cpu.sizes().begin(), t_cpu.sizes().end());
        std::vector<ssize_t> strides(shape.size());
        ssize_t stride = sizeof(T);
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }

        return pybind11::array(pybind11::buffer_info(
            t_cpu.data_ptr<T>(), sizeof(T),
            pybind11::format_descriptor<T>::format(),
            shape.size(), shape, strides));
    }

    inline pybind11::array tensor_to_numpy(const torch::Tensor& t) {
        if (t.dtype() == torch::kFloat32)
            return tensor_to_numpy_typed<float>(t);
        else if (t.dtype() == torch::kInt64)
            return tensor_to_numpy_typed<int64_t>(t);
        else
            throw std::runtime_error("Unsupported tensor dtype in tensor_to_numpy.");
    }

    inline pybind11::object MiniCam_to_py(const MiniCam& cam) {
        namespace py = pybind11;
        try {
            py::module_ torch = py::module_::import("torch");
            py::module_ m = py::module_::import("scripts_voxel.python_svraster_bridge.mini_cam");
            py::object MiniCamClass = m.attr("MiniCam");

            py::object w2c_py = torch.attr("from_numpy")(tensor_to_numpy(cam.w2c.cpu()));
            py::object c2w_py = torch.attr("from_numpy")(tensor_to_numpy(cam.c2w.cpu()));

            py::object py_cam = MiniCamClass(
                cam.width,
                cam.height,
                w2c_py,
                c2w_py,
                cam.tanfovx,
                cam.tanfovy,
                cam.cx,
                cam.cy,
                cam.cam_mode
            );

            // Keep the Python-side camera contract aligned with SVRaster.
            py_cam.attr("near") = py::float_(cam.near);
            py_cam.attr("frame_id") = py::int_(cam.frame_id);
    
            return py_cam;
    
        } catch (const py::error_already_set& e) {
            std::cerr << "[ERROR] Python exception in MiniCam_to_py():\n" << e.what() << std::endl;
            std::terminate();
        }
    }

    inline pybind11::array_t<uint8_t> tensorToNumpyRGB(const torch::Tensor &img) {
        // must be H×W×3 uint8
        TORCH_CHECK(img.dtype() == torch::kUInt8, "tensorToNumpyRGB expects uint8 tensor");
        TORCH_CHECK(img.is_contiguous(),     "tensorToNumpyRGB expects contiguous tensor");
        auto sizes = img.sizes();
        ssize_t H = sizes[0];
        ssize_t W = sizes[1];
        ssize_t C = sizes[2];
        TORCH_CHECK(C == 3, "tensorToNumpyRGB expects 3 channels");

        // Strides in bytes
        ssize_t stride_H = img.stride(0) * sizeof(uint8_t);
        ssize_t stride_W = img.stride(1) * sizeof(uint8_t);
        ssize_t stride_C = img.stride(2) * sizeof(uint8_t);

        return pybind11::array_t<uint8_t>(
            {H, W, C},                           // shape
            {stride_H, stride_W, stride_C},     // strides
            img.data_ptr<uint8_t>()              // data ptr
        );
    }
       
} // namespace sv
