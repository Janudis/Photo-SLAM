#pragma once

#include "include_voxel/voxel_mapper.h"
#include "include_voxel/voxel_mapper_utils.h"
#include "include_voxel/voxel_mapper_supervision.h"
#include "include_voxel/mapper_depth_registry.h"
#include "include_voxel/tandem_mvs_backend.h"
#include <pybind11/embed.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <Eigen/Eigenvalues>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <opencv2/flann.hpp>
#include <queue>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <c10/cuda/CUDACachingAllocator.h>

#include "ORB-SLAM3/include/Atlas.h"
#include "ORB-SLAM3/include/MapPoint.h"

namespace py = pybind11;

namespace {
cv::Mat mapperDepthForKeyframe(
    const std::string& image_filename,
    const cv::Mat& tracking_depth,
    sv::Camera& camera)
{
    cv::Mat mapper_depth = sv::loadMapperDepthImage(image_filename);
    if (mapper_depth.empty()) {
        cv::Mat undistorted;
        camera.undistortImage(tracking_depth, undistorted);
        return undistorted;
    }
    if (mapper_depth.type() != CV_32FC1) {
        mapper_depth.convertTo(mapper_depth, CV_32FC1);
    }
    cv::Mat undistorted;
    cv::remap(
        mapper_depth,
        undistorted,
        camera.undistort_map1,
        camera.undistort_map2,
        cv::INTER_NEAREST);
    return undistorted;
}

std::filesystem::path resolveMapperResourcePath(
    const std::filesystem::path& config_file,
    const std::filesystem::path& configured_path)
{
    if (configured_path.empty() || configured_path.is_absolute()) {
        return configured_path;
    }
    if (std::filesystem::exists(configured_path)) {
        return std::filesystem::absolute(configured_path);
    }

    std::filesystem::path cursor =
        std::filesystem::absolute(config_file).parent_path();
    while (!cursor.empty()) {
        if (std::filesystem::exists(cursor / "CMakeLists.txt")) {
            return cursor / configured_path;
        }
        const std::filesystem::path parent = cursor.parent_path();
        if (parent == cursor) {
            break;
        }
        cursor = parent;
    }
    return configured_path;
}

void ensurePythonRuntimeInitialized(bool import_torch_cuda)
{
    static bool initialized_by_mapper = false;
    static PyThreadState* released_main_thread_state = nullptr;
    const auto configure_python_paths = []() {
        py::exec(R"PY(
import os
import site
import sys

_user_site = site.getusersitepackages()
site.addsitedir(_user_site)
for _path in (os.path.join(_user_site, "rerun_sdk"), _user_site):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)
)PY");
        py::module_::import("sys").attr("path").attr("insert")(0, "scripts_voxel");
        py::module_::import("sys").attr("path").attr("insert")(0, "../scripts_voxel");
    };

    if (!initialized_by_mapper && Py_IsInitialized() == 0) {
        py::initialize_interpreter(false);
        initialized_by_mapper = true;
        configure_python_paths();
        if (import_torch_cuda) {
            py::module_::import("torch.cuda");
        }
        released_main_thread_state = PyEval_SaveThread();
        return;
    }

    {
        py::gil_scoped_acquire gil;
        configure_python_paths();
        if (import_torch_cuda) {
            py::module_::import("torch.cuda");
        }
    }
    (void)released_main_thread_state;
}
} // namespace
