/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
 *
 * Photo-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Photo-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Photo-SLAM.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <vector>

#include <torch/torch.h>

namespace loss_utils
{

inline torch::Tensor l1_loss(torch::Tensor &network_output, torch::Tensor &gt)
{
    return torch::abs(network_output - gt).mean();
}

inline torch::Tensor l2_loss(torch::Tensor &network_output, torch::Tensor &gt) {
    // Equivalent to torch.nn.functional.mse_loss(x, y, reduction='mean')
    // auto diff = network_output - gt;
    // return diff.mul(diff).mean();
    return torch::nn::functional::mse_loss(network_output, gt);
}
inline torch::Tensor psnr(torch::Tensor &img1, torch::Tensor &img2)
{
    auto mse = torch::pow(img1 - img2, 2).mean();
    return 10.0f * torch::log10(1.0f / mse);
}

/** def psnr(img1, img2):
 *     mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
 *     return 20 * torch.log10(1.0 / torch.sqrt(mse))
 */
inline torch::Tensor psnr_gaussian_splatting(torch::Tensor &img1, torch::Tensor &img2)
{
    auto mse = torch::pow(img1 - img2, 2).view({img1.size(0) , -1}).mean(1, /*keepdim=*/true);
    return 20.0f * torch::log10(1.0f / torch::sqrt(mse)).mean();
}

inline torch::Tensor gaussian(
    int window_size,
    float sigma,
    torch::DeviceType device_type = torch::kCUDA)
{
    std::vector<float> gauss_values(window_size);
    for (int x = 0; x < window_size; ++x) {
        int temp = x - window_size / 2;
        gauss_values[x] = std::exp(-temp * temp / (2.0f * sigma * sigma));
    }
    torch::Tensor gauss = torch::tensor(
        gauss_values,
        torch::TensorOptions().device(device_type));
    return gauss / gauss.sum();
}

inline torch::autograd::Variable create_window(
    int window_size,
    int64_t channel,
    torch::DeviceType device_type = torch::kCUDA)
{
    auto _1D_window = gaussian(window_size, 1.5f, device_type).unsqueeze(1);
    auto _2D_window = _1D_window.mm(_1D_window.t()).to(torch::kFloat).unsqueeze(0).unsqueeze(0);
    auto window = torch::autograd::Variable(_2D_window.expand({channel, 1, window_size, window_size}).contiguous());
    return window;
}

inline torch::Tensor _ssim(
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::autograd::Variable &window,
    int window_size,
    int64_t channel,
    bool size_average = true)
{
    int window_size_half = window_size / 2;
    auto mu1 = torch::nn::functional::conv2d(img1, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel));
    auto mu2 = torch::nn::functional::conv2d(img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel));

    auto mu1_sq = mu1.pow(2);
    auto mu2_sq = mu2.pow(2);
    auto mu1_mu2 = mu1 * mu2;

    auto sigma1_sq = torch::nn::functional::conv2d(img1 * img1, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel))
                    - mu1_sq;
    auto sigma2_sq = torch::nn::functional::conv2d(img2 * img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel))
                    - mu2_sq;
    auto sigma12 = torch::nn::functional::conv2d(img1 * img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel))
                    - mu1_mu2;

    auto C1 = 0.01 * 0.01;
    auto C2 = 0.03 * 0.03;

    auto ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2));

    if (size_average)
        return ssim_map.mean();
    else
        return ssim_map.mean(1).mean(1).mean(1);
}

inline torch::Tensor ssim(
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::DeviceType device_type = torch::kCUDA,
    int window_size = 11,
    bool size_average = true)
{
    auto channel = img1.size(-3);
    auto window = create_window(window_size, channel, device_type);

    // window = window.to(img1.device());
    window = window.type_as(img1);

    return _ssim(img1, img2, window, window_size, channel, size_average);
}

inline torch::Tensor fast_ssim_loss(torch::Tensor x, torch::Tensor y)
{
    // Match Python: inputs are [C,H,W] or [B,C,H,W]; fused_ssim expects [B,C,H,W].
    if (x.dim() == 3)
    {
        x = x.unsqueeze(0);  // [1,C,H,W]
        y = y.unsqueeze(0);
    }

    const bool is_train = x.requires_grad() || y.requires_grad();

    // Fallback: use your existing SSIM implementation (slower).
    // Note: your ssim() already expects 4D input [B,C,H,W].
    torch::DeviceType device_type =
        x.is_cuda() ? torch::kCUDA : torch::kCPU;
    torch::Tensor ssim_val = ssim(x, y, device_type);

    // Return loss = 1 - SSIM
    return 1.0f - ssim_val;
}

struct SparseDepthLoss
{
    // Last iteration where this loss is active (inclusive)
    int iter_end_;

    explicit SparseDepthLoss(int iter_end)
        : iter_end_(iter_end)
    {}

    inline bool isActive(int iteration) const
    {
        return iteration <= iter_end_;
    }

    /**
     * C++ version of SVRaster's SparseDepthLoss.__call__.
     *
     * Python reference:
     *
     *   depth = raw_depth[0] / (1 - raw_T).clamp_min_(1e-4)
     *   rend_sparse_depth = grid_sample(depth[None], sparse_uv[None,None], ...)
     *   return smooth_l1_loss(rend_sparse_depth, sparse_depth)
     *
     * Arguments:
     *   raw_T         : Tensor containing raw transmittance T, shape ~ [1,H,W] or [1,1,H,W]
     *   raw_depth     : Tensor containing raw accumulated depth, shape ~ [1,H,W] or [1,1,H,W]
     *   sparse_uv     : Tensor of sampling locations in NDC, shape [N,2], in [-1,1]
     *   sparse_depth  : Tensor of ground-truth depths, shape [N] or [N,1]
     *
     * Returns:
     *   Smooth L1 loss (Huber) between rendered sparse depths and target sparse depths.
     */
    inline torch::Tensor operator()(const torch::Tensor& raw_T,
                                    const torch::Tensor& raw_depth,
                                    const torch::Tensor& sparse_uv,
                                    const torch::Tensor& sparse_depth) const
    {
        TORCH_CHECK(raw_T.defined(),    "SparseDepthLoss: raw_T is undefined");
        TORCH_CHECK(raw_depth.defined(),"SparseDepthLoss: raw_depth is undefined");
        TORCH_CHECK(sparse_uv.defined(),"SparseDepthLoss: sparse_uv is undefined");
        TORCH_CHECK(sparse_depth.defined(),"SparseDepthLoss: sparse_depth is undefined");

        // ---- 1) Compute per-pixel depth = raw_depth / (1 - T) -----------------------
        torch::Tensor depth = raw_depth;

        // Accept [1,1,H,W], [1,H,W], [H,W], and also [3,H,W] -> use first slice (SVRaster-style [0])
        if (depth.dim() == 4 && depth.size(0) == 1 && depth.size(1) == 1)
        {
            // [1,1,H,W] -> [H,W]
            depth = depth.index({0, 0});
        }
        else if (depth.dim() == 3)
        {
            if (depth.size(0) == 1)
            {
                // [1,H,W] -> [H,W]
                depth = depth.index({0});
            }
            else if (depth.size(0) == 3)
            {
                // [3,H,W] -> take first slice [H,W]
                // mirrors Python: depth = raw_depth[0] / ...
                depth = depth.index({0});
            }
            else
            {
                TORCH_CHECK(false,
                            "SparseDepthLoss: raw_depth 3D tensor must have size(0)==1 or 3; got ",
                            depth.sizes());
            }
        }
        else if (depth.dim() == 2)
        {
            // already [H,W], do nothing
        }
        else
        {
            TORCH_CHECK(false,
                        "SparseDepthLoss: raw_depth must have shape [1,H,W], [1,1,H,W], [H,W] "
                        "or [3,H,W]; got ",
                        depth.sizes());
        }

        // Bring raw_T to [H,W] with analogous logic
        torch::Tensor T = raw_T;
        if (T.dim() == 4 && T.size(0) == 1 && T.size(1) == 1)
        {
            T = T.index({0, 0});          // [H,W]
        }
        else if (T.dim() == 3)
        {
            if (T.size(0) == 1)
            {
                T = T.index({0});         // [H,W]
            }
            else if (T.size(0) == 3)
            {
                // same reasoning: take first slice
                T = T.index({0});         // [H,W]
            }
            else
            {
                TORCH_CHECK(false,
                            "SparseDepthLoss: raw_T 3D tensor must have size(0)==1 or 3; got ",
                            T.sizes());
            }
        }
        else if (T.dim() == 2)
        {
            // [H,W], ok
        }
        else
        {
            TORCH_CHECK(false,
                        "SparseDepthLoss: raw_T must have shape [1,H,W], [1,1,H,W], [H,W] "
                        "or [3,H,W]; got ",
                        T.sizes());
        }

        torch::Tensor denom = (1.0f - T).clamp_min(1e-4f);  // (1 - T).clamp_min_(1e-4)
        depth = depth / denom;                              // [H,W]

        // ---- 2) Prepare sparse_uv and sparse_depth ---------------------------------
        // sparse_uv: [N,2] with coordinates in [-1,1]
        TORCH_CHECK(sparse_uv.dim() == 2 && sparse_uv.size(1) == 2,
                    "SparseDepthLoss: sparse_uv must have shape [N,2], got ",
                    sparse_uv.sizes());

        // sparse_depth: [N] or [N,1]
        torch::Tensor target_depth = sparse_depth;
        if (target_depth.dim() == 2 && target_depth.size(1) == 1)
        {
            target_depth = target_depth.squeeze(1);         // [N]
        }
        TORCH_CHECK(target_depth.dim() == 1,
                    "SparseDepthLoss: sparse_depth must have shape [N] or [N,1], got ",
                    sparse_depth.sizes());

        const auto N = sparse_uv.size(0);
        TORCH_CHECK(target_depth.size(0) == N,
                    "SparseDepthLoss: sparse_uv and sparse_depth must agree in N; got ",
                    N, " vs ", target_depth.size(0));

        // ---- 3) grid_sample(depth[None], sparse_uv[None,None], ...) ----------------
        //
        // depth: [H,W]          -> [1,1,H,W]
        // sparse_uv: [N,2]      -> [1,1,N,2]
        // result: [1,1,1,N]     -> squeeze() -> [N]

        torch::Tensor depth_img = depth.unsqueeze(0).unsqueeze(0);   // [1,1,H,W]
        torch::Tensor grid      = sparse_uv.unsqueeze(0).unsqueeze(0); // [1,1,N,2]

        auto gs_opts = torch::nn::functional::GridSampleFuncOptions()
                           .mode(torch::kBilinear)
                           .padding_mode(torch::kZeros)
                           .align_corners(false);

        torch::Tensor rend_sparse_depth =
            torch::nn::functional::grid_sample(depth_img, grid, gs_opts).squeeze(); // [N]

        // ---- 4) Smooth L1 loss between rendered and target sparse depths -----------

        auto loss = torch::nn::functional::smooth_l1_loss(
            rend_sparse_depth,
            target_depth,
            torch::nn::functional::SmoothL1LossFuncOptions().reduction(torch::kMean));

        return loss;
    }
};

}