#pragma once

#include <torch/torch.h>

namespace sv {

inline constexpr float kSHC0 = 0.28209479177387814f;

inline torch::Tensor rgbToShZero(const torch::Tensor& rgb)
{
    return (rgb - 0.5f) / kSHC0;
}

inline torch::Tensor shZeroToRgb(const torch::Tensor& sh)
{
    return sh * kSHC0 + 0.5f;
}

}
