#pragma once

#include "include_voxel/rerun_utils.h"

#include <pybind11/pybind11.h>

namespace sv {

struct RerunVisualizerBridge::PyImpl
{
    pybind11::object visualizer;
};

} // namespace sv
