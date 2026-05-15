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

#include <algorithm>
#include <cmath>
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

inline torch::Tensor huber_loss(torch::Tensor &network_output,
                                torch::Tensor &gt,
                                float thres = 0.03f)
{
    auto abs_err = (network_output - gt).abs();
    auto l1 = abs_err.mean(0);
    auto l2 = abs_err.square().mean(0);
    auto loss = torch::where(
        l1 < thres,
        l2,
        2.0f * thres * l1 - thres * thres);
    return loss.mean();
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

inline torch::Tensor prob_concen_loss(torch::Tensor prob)
{
    // SVRaster: (prob^2 * (1-prob)^2).mean()
    return (prob.square() * (1.0f - prob).square()).mean();
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

struct DepthAnythingv2Loss
{
    int iter_from_;
    int iter_end_;
    float end_mult_;

    DepthAnythingv2Loss(int iter_from, int iter_end, float end_mult)
        : iter_from_(iter_from),
          iter_end_(iter_end),
          end_mult_(end_mult)
    {}

    inline bool isActive(int iteration) const
    {
        return iteration >= iter_from_ && iteration <= iter_end_;
    }

    inline torch::Tensor operator()(const torch::Tensor& raw_T,
                                    const torch::Tensor& raw_depth,
                                    const torch::Tensor& mono_depth,
                                    float cam_near,
                                    int iteration) const
    {
        TORCH_CHECK(raw_T.defined(), "DepthAnythingv2Loss: raw_T is undefined");
        TORCH_CHECK(raw_depth.defined(), "DepthAnythingv2Loss: raw_depth is undefined");
        TORCH_CHECK(mono_depth.defined(), "DepthAnythingv2Loss: mono_depth is undefined");

        if (!isActive(iteration))
        {
            return torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(raw_depth.device()));
        }

        torch::Tensor depth = raw_depth;
        if (depth.dim() == 4 && depth.size(0) == 1)
        {
            depth = depth.squeeze(0);
        }
        if (depth.dim() == 2)
        {
            depth = depth.unsqueeze(0);
        }
        TORCH_CHECK(depth.dim() == 3,
                    "DepthAnythingv2Loss: raw_depth must be [C,H,W] or [1,C,H,W], got ",
                    depth.sizes());

        torch::Tensor T = raw_T;
        if (T.dim() == 4 && T.size(0) == 1)
        {
            T = T.squeeze(0);
        }
        if (T.dim() == 2)
        {
            T = T.unsqueeze(0);
        }
        TORCH_CHECK(T.dim() == 3,
                    "DepthAnythingv2Loss: raw_T must be [C,H,W] or [1,C,H,W], got ",
                    T.sizes());

        torch::Tensor invdepth = 1.0f / depth.unsqueeze(1).clamp_min(std::max(1e-6f, cam_near));
        const int64_t ref_idx = std::min<int64_t>(2, invdepth.size(0) - 1);
        torch::Tensor X = invdepth.index({0}).unsqueeze(0);
        torch::Tensor Xref = invdepth.index({ref_idx}).unsqueeze(0);

        torch::Tensor alpha = 1.0f - T.index({0}).unsqueeze(0).unsqueeze(0);

        torch::Tensor Y = mono_depth;
        if (Y.dim() == 4 && Y.size(0) == 1)
        {
            Y = Y.squeeze(0);
        }
        if (Y.dim() == 3 && Y.size(0) == 1)
        {
            Y = Y.squeeze(0);
        }
        if (Y.dim() == 2)
        {
            Y = Y.unsqueeze(0).unsqueeze(0);
        }
        else if (Y.dim() == 3 && Y.size(0) == 1)
        {
            Y = Y.unsqueeze(0);
        }
        TORCH_CHECK(Y.dim() == 4,
                    "DepthAnythingv2Loss: mono_depth must be [H,W], [1,H,W], or [1,1,H,W], got ",
                    mono_depth.sizes());

        if (Y.sizes().slice(2) != X.sizes().slice(2))
        {
            Y = torch::nn::functional::interpolate(
                Y,
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{X.size(2), X.size(3)})
                    .mode(torch::kBilinear)
                    .align_corners(false));
        }

        X = X * alpha;

        torch::Tensor target;
        {
            torch::NoGradGuard no_grad;
            const torch::Tensor Ymed = Y.median();
            const torch::Tensor Ys = (Y - Ymed).abs().mean().clamp_min(1e-6f);
            const torch::Tensor Xmed = Xref.median();
            const torch::Tensor Xs = (Xref - Xmed).abs().mean().clamp_min(1e-6f);
            target = (Y - Ymed) * (Xs / Ys) + Xmed;
        }

        torch::Tensor mask = (target > 0.01f) & (alpha > 0.5f);
        mask = mask.to(X.dtype());
        X = X * mask;
        target = target * mask;

        torch::Tensor loss = torch::nn::functional::mse_loss(
            X,
            target,
            torch::nn::functional::MSELossFuncOptions().reduction(torch::kMean));

        if (iter_end_ <= iter_from_ || end_mult_ == 1.0f)
        {
            return loss;
        }

        const float ratio = std::clamp(
            static_cast<float>(iteration - iter_from_) /
                static_cast<float>(iter_end_ - iter_from_),
            0.0f,
            1.0f);
        const float mult = std::pow(end_mult_, ratio);
        return loss * mult;
    }
};

inline torch::Tensor normalize_patch_tensor(
    const torch::Tensor& input,
    c10::optional<torch::Tensor> mean = c10::nullopt,
    c10::optional<torch::Tensor> std = c10::nullopt)
{
    torch::Tensor input_mean =
        mean.has_value() ? *mean : torch::mean(input, /*dim=*/1, /*keepdim=*/true);
    torch::Tensor input_std =
        std.has_value() ? *std : torch::std(input, /*dim=*/1, /*unbiased=*/true, /*keepdim=*/true);
    torch::Tensor global_std =
        torch::std(input.reshape({-1}) + 1e-5f).clamp_min(1e-6f);
    return (input - input_mean) / (input_std + 1e-2f * global_std);
}

inline torch::Tensor patchify_tensor(const torch::Tensor& input, int patch_size)
{
    torch::Tensor x = input;
    if (x.dim() == 2) {
        x = x.unsqueeze(0).unsqueeze(0);  // [1,1,H,W]
    } else if (x.dim() == 3) {
        x = x.unsqueeze(1);               // [N,1,H,W]
    }
    TORCH_CHECK(x.dim() == 4,
                "patchify_tensor expects [N,C,H,W], got ", input.sizes());
    torch::Tensor unfolded = torch::nn::functional::unfold(
        x,
        torch::nn::functional::UnfoldFuncOptions(
            {patch_size, patch_size}).stride({patch_size, patch_size}));
    return unfolded.permute({0, 2, 1}).reshape({-1, x.size(1) * patch_size * patch_size});
}

inline torch::Tensor margin_l2_loss_weighted(
    const torch::Tensor& network_output,
    const torch::Tensor& gt,
    float margin,
    const torch::Tensor& weight = torch::Tensor())
{
    torch::Tensor mask = (network_output - gt).abs() > margin;
    if (!mask.any().item<bool>()) {
        return torch::zeros(
            {},
            torch::TensorOptions().dtype(network_output.dtype()).device(network_output.device()));
    }
    torch::Tensor sq = (network_output - gt).pow(2);
    if (weight.defined()) {
        sq = sq * weight;
    }
    return sq.masked_select(mask).mean();
}

inline torch::Tensor patch_norm_mse_loss_geosvr(
    const torch::Tensor& input,
    const torch::Tensor& target,
    c10::optional<int> patch_size = c10::nullopt,
    float margin = 0.0f,
    const torch::Tensor& weight = torch::Tensor())
{
    torch::Tensor input_patches;
    torch::Tensor target_patches;
    torch::Tensor weight_patches;

    if (patch_size.has_value()) {
        input_patches = normalize_patch_tensor(
            patchify_tensor(input, *patch_size));
        target_patches = normalize_patch_tensor(
            patchify_tensor(target, *patch_size));
        if (weight.defined()) {
            weight_patches = patchify_tensor(weight, *patch_size);
        }
    } else {
        input_patches = normalize_patch_tensor(input.view({input.size(0), -1}));
        target_patches = normalize_patch_tensor(target.view({target.size(0), -1}));
        if (weight.defined()) {
            weight_patches = weight.view({weight.size(0), -1});
        }
    }

    return margin_l2_loss_weighted(input_patches, target_patches, margin, weight_patches);
}

inline torch::Tensor patch_norm_mse_loss_global_geosvr(
    const torch::Tensor& input,
    const torch::Tensor& target,
    c10::optional<int> patch_size = c10::nullopt,
    float margin = 0.0f,
    const torch::Tensor& weight = torch::Tensor())
{
    torch::Tensor input_patches;
    torch::Tensor target_patches;
    torch::Tensor weight_patches;

    if (patch_size.has_value()) {
        torch::Tensor input_std = torch::std(input).detach();
        torch::Tensor target_std = torch::std(target).detach();
        input_patches = normalize_patch_tensor(
            patchify_tensor(input, *patch_size),
            c10::nullopt,
            input_std);
        target_patches = normalize_patch_tensor(
            patchify_tensor(target, *patch_size),
            c10::nullopt,
            target_std);
        if (weight.defined()) {
            weight_patches = patchify_tensor(weight, *patch_size);
        }
    } else {
        torch::Tensor input_std = torch::std(input).detach();
        torch::Tensor target_std = torch::std(target).detach();
        input_patches = normalize_patch_tensor(
            input.view({input.size(0), -1}),
            c10::nullopt,
            input_std);
        target_patches = normalize_patch_tensor(
            target.view({target.size(0), -1}),
            c10::nullopt,
            target_std);
        if (weight.defined()) {
            weight_patches = weight.view({weight.size(0), -1});
        }
    }

    return margin_l2_loss_weighted(input_patches, target_patches, margin, weight_patches);
}

struct DepthAnythingv2UncertaintyLoss
{
    int iter_from_;
    int iter_end_;
    float end_mult_;
    bool overall_;
    bool alpha_adjust_;

    DepthAnythingv2UncertaintyLoss(
        int iter_from,
        int iter_end,
        float end_mult,
        bool overall,
        bool alpha_adjust)
        : iter_from_(iter_from),
          iter_end_(iter_end),
          end_mult_(end_mult),
          overall_(overall),
          alpha_adjust_(alpha_adjust)
    {}

    inline bool isActive(int iteration) const
    {
        return iteration >= iter_from_ && iteration <= iter_end_;
    }

    inline torch::Tensor operator()(const torch::Tensor& raw_T,
                                    const torch::Tensor& raw_depth,
                                    const torch::Tensor& mono_depth,
                                    float cam_near,
                                    int iteration,
                                    int patch_size,
                                    const torch::Tensor& level_weight = torch::Tensor()) const
    {
        TORCH_CHECK(raw_T.defined(), "DepthAnythingv2UncertaintyLoss: raw_T is undefined");
        TORCH_CHECK(raw_depth.defined(), "DepthAnythingv2UncertaintyLoss: raw_depth is undefined");
        TORCH_CHECK(mono_depth.defined(), "DepthAnythingv2UncertaintyLoss: mono_depth is undefined");

        if (!isActive(iteration)) {
            return torch::zeros(
                {},
                torch::TensorOptions().dtype(torch::kFloat32).device(raw_depth.device()));
        }

        torch::Tensor depth = raw_depth;
        if (depth.dim() == 4 && depth.size(0) == 1) depth = depth.squeeze(0);
        if (depth.dim() == 2) depth = depth.unsqueeze(0);
        TORCH_CHECK(depth.dim() == 3,
                    "DepthAnythingv2UncertaintyLoss: raw_depth must be [C,H,W] or [1,C,H,W], got ",
                    depth.sizes());
        depth = depth.clamp_min(std::max(1e-6f, cam_near));

        torch::Tensor T = raw_T;
        if (T.dim() == 4 && T.size(0) == 1) T = T.squeeze(0);
        if (T.dim() == 2) T = T.unsqueeze(0);
        TORCH_CHECK(T.dim() == 3,
                    "DepthAnythingv2UncertaintyLoss: raw_T must be [C,H,W] or [1,C,H,W], got ",
                    T.sizes());

        torch::Tensor alpha = 1.0f - T.index({0}).unsqueeze(0);
        torch::Tensor depth_map = depth.index({0}).unsqueeze(0);
        if (alpha_adjust_) {
            depth_map = depth_map / alpha.clamp_min(1e-3f).detach();
        }

        torch::Tensor mono = mono_depth;
        if (mono.dim() == 4 && mono.size(0) == 1) mono = mono.squeeze(0);
        if (mono.dim() == 3 && mono.size(0) == 1) mono = mono.squeeze(0);
        if (mono.dim() == 2) mono = mono.unsqueeze(0);
        TORCH_CHECK(mono.dim() == 3,
                    "DepthAnythingv2UncertaintyLoss: mono_depth must be [H,W], [1,H,W], or [1,1,H,W], got ",
                    mono_depth.sizes());

        if (mono.sizes().slice(1) != depth_map.sizes().slice(1)) {
            mono = torch::nn::functional::interpolate(
                mono.unsqueeze(0),
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{depth_map.size(1), depth_map.size(2)})
                    .mode(torch::kBilinear)
                    .align_corners(false)).squeeze(0);
        }

        torch::Tensor weight = level_weight;
        if (weight.defined()) {
            if (weight.dim() == 4 && weight.size(0) == 1) weight = weight.squeeze(0);
            if (weight.dim() == 2) weight = weight.unsqueeze(0);
            if (weight.dim() == 3 && weight.sizes().slice(1) != depth_map.sizes().slice(1)) {
                weight = torch::nn::functional::interpolate(
                    weight.unsqueeze(0),
                    torch::nn::functional::InterpolateFuncOptions()
                        .size(std::vector<int64_t>{depth_map.size(1), depth_map.size(2)})
                        .mode(torch::kNearest)).squeeze(0);
            }
        }

        torch::Tensor mask = alpha < 0.5f;
        torch::Tensor valid = ~mask;
        if (valid.any().item<bool>()) {
            torch::Tensor mono_fill = mono.masked_select(valid).mean();
            torch::Tensor depth_fill = depth_map.masked_select(valid).mean();
            mono = mono.clone();
            depth_map = depth_map.clone();
            mono.masked_fill_(mask, mono_fill.item<float>());
            depth_map.masked_fill_(mask, depth_fill.item<float>());
        }

        torch::Tensor loss = patch_norm_mse_loss_global_geosvr(
            depth_map.unsqueeze(0),
            mono.unsqueeze(0),
            patch_size,
            0.001f,
            weight);
        loss = loss + 0.1f * patch_norm_mse_loss_geosvr(
            depth_map.unsqueeze(0),
            mono.unsqueeze(0),
            patch_size,
            0.001f,
            weight);

        if (overall_) {
            torch::Tensor inv_depth = 1.0f / depth_map.clamp_min(std::max(1e-6f, cam_near));
            torch::Tensor inv_mono = 1.0f / mono.clamp_min(std::max(1e-6f, cam_near));
            loss = loss + 0.1f * patch_norm_mse_loss_global_geosvr(
                inv_depth.unsqueeze(0),
                inv_mono.unsqueeze(0),
                patch_size,
                0.001f,
                weight);
        }

        if (iter_end_ <= iter_from_ || end_mult_ == 1.0f) {
            return loss;
        }

        const float ratio = std::clamp(
            static_cast<float>(iteration - iter_from_) /
                static_cast<float>(iter_end_ - iter_from_),
            0.0f,
            1.0f);
        const float mult = std::pow(end_mult_, ratio);
        return loss * mult;
    }
};

}
