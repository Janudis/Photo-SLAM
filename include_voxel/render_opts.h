#pragma once
#include <optional>
#include <torch/torch.h>

namespace sv {

struct RenderOpts {
    bool        track_max_w = false;
    std::optional<float> lambda_R_concen = std::nullopt;
    bool        output_T = false;
    bool        output_depth = false;
    std::optional<float> ss = 1.0f;
    bool        rand_bg = false;
    bool        use_auto_exposure = false;
    bool        output_normal = false;
    std::optional<float> lambda_dist = std::nullopt;
    std::optional<float> lambda_ascending = std::nullopt;
    torch::Tensor gt_color = torch::Tensor();
};

} // namespace sv
