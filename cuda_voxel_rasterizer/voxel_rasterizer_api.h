#pragma once
#include <torch/extension.h>

at::Tensor voxel_rasterizer_forward(
    torch::Tensor vox,
    torch::Tensor K,
    torch::Tensor Twc,
    int64_t H,
    int64_t W);
