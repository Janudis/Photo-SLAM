#include "include_voxel/voxel_renderer.h"

#include <algorithm>
#include <iostream>
#include <tuple>

#include <ATen/ops/upsample_bilinear2d.h>    // at::upsample_bilinear2d


namespace sv {

using namespace torch::nn::functional;

VoxelRenderer::VoxelRenderer(const VoxelModel* model)
: M_(model) {
    TORCH_CHECK(M_ != nullptr, "VoxelRenderer: model pointer must not be null.");
}

void VoxelRenderer::freezeVoxGeo() {
    TORCH_CHECK(M_->_geo_grid_pts_.defined(), "freezeVoxGeo: _geo_grid_pts_ undefined");
    TORCH_CHECK(M_->vox_key_.defined() && M_->vox_key_.dim()==2 && M_->vox_key_.size(1)==8,
                "freezeVoxGeo: vox_key_ must be [N,8]");

    torch::NoGradGuard guard;

    const int64_t N = M_->oct_path_.size(0);
    auto idx_all = torch::arange(N, M_->vox_key_.options().dtype(torch::kLong)); // [N] CUDA

    // EXACT mirror of python:
    frozen_vox_geo_ = sv::rasterizer::GatherGeoParams(
        M_->vox_key_,          // [N,8]
        idx_all,               // [N]
        M_->_geo_grid_pts_     // tri-interp grid param
    ).contiguous();

    if (M_->_geo_grid_pts_.requires_grad()) {
        const_cast<torch::Tensor&>(M_->_geo_grid_pts_).set_requires_grad(false);
    }
}

void VoxelRenderer::unfreezeVoxGeo() {
    frozen_vox_geo_.reset();
    if (M_->_geo_grid_pts_.defined()) {
        const_cast<torch::Tensor&>(M_->_geo_grid_pts_).set_requires_grad(true);
    }
}

RenderOutput VoxelRenderer::render(
    const MiniCam& cam,
    int image_height,
    int image_width,
    const torch::Tensor& gt_image, 
    const char* color_mode,
    bool track_max_w,
    std::optional<float> ss,
    bool output_depth,
    bool output_normal,
    bool output_T,
    bool rand_bg,
    bool use_auto_exposure,
    const RenderOpts& other_opt
) const
{
    RenderOutput out;

    // Early out if no voxels:
    if (!M_->oct_path_.defined() || M_->oct_path_.size(0) == 0) {
        return out; // empty
    }

    const int64_t N = M_->oct_path_.size(0);
    TORCH_CHECK(M_->center_.size(0) == N && M_->size_.size(0) == N &&
                M_->sh0_.size(0) == N && M_->shs_.size(0) == N,
                "Topology mismatch: N=", N,
                " center=", M_->center_.sizes(),
                " size=",   M_->size_.sizes(),
                " sh0=",    M_->sh0_.sizes(),
                " shs=",    M_->shs_.sizes());
    if (frozen_vox_geo_.defined()) {
        TORCH_CHECK(frozen_vox_geo_.size(0) == N, "Frozen geo stale: ", frozen_vox_geo_.sizes(), " vs N=", N);
    }

    // Supersampling & resized intrinsics (match SVRaster logic)
    const float ss_value = ss.has_value()
                        ? *ss
                        : (other_opt.ss.has_value() ? *other_opt.ss : M_->ss_);
    // 2) Sizes & intrinsic scaling (w_src, h_src, w, h, w_ss, h_ss)
    const int w_src = image_width;
    const int h_src = image_height;
    const int w = int(std::round(w_src * ss_value));
    const int h = int(std::round(h_src * ss_value));
    const float w_ss = float(w) / w_src;
    const float h_ss = float(h) / h_src;

    // Prepare gt_color like SVRaster's: if ss != 1.0 and 'gt_color' in other_opt, resize it first
    const auto dev = M_->center_.device();
    torch::Tensor rs_gt_color = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(dev)); // leave empty unless valid
    if (other_opt.gt_color.defined() && other_opt.gt_color.numel() > 0) {
        // Start from kwargs gt_color (do NOT use positional gt_image here)
        auto gt = other_opt.gt_color;
        // Ensure channel-first [3,H,W]; if you stored HWC anywhere, fix it here
        if (gt.dim() == 3 && gt.size(0) != 3 && gt.size(2) == 3) {
            // HWC -> CHW
            gt = gt.permute({2, 0, 1}).contiguous();
        }
        // Enforce float32
        if (gt.dtype() != torch::kFloat32) {
            // If it was uint8, scale to [0,1]; otherwise just cast
            if (gt.dtype() == torch::kUInt8) {
                gt = gt.to(torch::kFloat32).div_(255.0f);
            } else {
                gt = gt.to(torch::kFloat32);
            }
        }
        // Move to same CUDA device
        if (gt.device() != dev) {
            gt = gt.to(dev, /*non_blocking=*/true);
        }
        // Resize only when ss != 1.0 (SVRaster behavior)
        if (std::abs(ss_value - 1.f) > 1e-6f) {
            gt = resize_rendering(gt, h, w);  // returns CHW
        }
        rs_gt_color = gt.contiguous();  // make sure it's contiguous for data_ptr<float>()
    }

    // 3) n_samp_per_vox = other_opt.get('n_samp_per_vox', self.n_samp_per_vox)
    const int n_samp_per_vox = other_opt.n_samp_per_vox.has_value()
                            ? *other_opt.n_samp_per_vox
                            : M_->n_samp_per_vox_;

    // Define vox_fn (runs on CPU calling CUDA ops), returns per-voxel params
    auto vox_fn = [&](const torch::Tensor& idx,           // [K] in-frustum indices (int64)
                    const torch::Tensor& cam_pos,       // [3]
                    const char*        cmode,           // color_mode
                    const torch::Tensor& viewdir = torch::Tensor())       // [K,3] or undefined
                    -> sv::rasterizer::VoxParams
    {
        using namespace torch::indexing;

        const int64_t N = M_->oct_path_.size(0); 
        auto float_opts = M_->center_.options().dtype(torch::kFloat32);
        auto long_opts  = M_->center_.options().dtype(torch::kLong);

        // SVRaster passes the active subset `idx` to gather; allow empty → fall back to all
        torch::Tensor care_idx = (idx.defined() && idx.numel() > 0)
                            ? idx
                            : torch::arange(N, long_opts);

        sv::rasterizer::VoxParams vp;

        // -----------------------
        // 1) Gather voxel geos
        // -----------------------
        // Optional: invalidate stale frozen snapshot if topology changed
        if (frozen_vox_geo_.defined() && frozen_vox_geo_.size(0) != N) {
            std::cout << "[VoxelRenderer] Warning: frozen_vox_geo_ size mismatch ("
                      << frozen_vox_geo_.size(0) << " vs " << N << "); unfreezing." << std::endl;
            const_cast<torch::Tensor&>(frozen_vox_geo_) = torch::Tensor();
        }

        if (frozen_vox_geo_.defined() && frozen_vox_geo_.numel() > 0) {
            vp.geos = frozen_vox_geo_;
        } else {
            auto geos = sv::rasterizer::GatherGeoParams(
                M_->vox_key_,   // [N,8]
                care_idx,       // [K] (ignored for output shape; backend returns [N,8,*])
                M_->_geo_grid_pts_);
            TORCH_CHECK(geos.size(0) == N, "GatherGeoParams must return [N,...]. Got ", geos.sizes());
            vp.geos = geos.contiguous();
        }

        // -----------------------
        // 2) Compute voxel colors (SH_eval)
        // -----------------------
        std::string mode = (cmode && *cmode) ? std::string(cmode) : std::string("sh");
        int active_sh_degree = M_->active_sh_degree_;
        if (mode.rfind("sh", 0) == 0) {
            if (mode.size() >= 3) {
                int parsed = mode[2] - '0';
                if (parsed >= 0 && parsed <= M_->max_sh_degree_) active_sh_degree = parsed;
            }
            mode = "sh";
        }

        if (mode == "sh") {
            auto viewdir_arg = (viewdir.defined() && viewdir.numel() > 0) ? viewdir : torch::Tensor();
            auto rgbs = sv::rasterizer::SH_eval(
                active_sh_degree,
                care_idx,             // [K]
                M_->center_,          // [N,3]
                cam_pos,              // [3]
                viewdir_arg,          // [K,3] or undef
                M_->sh0_,             // [N,3]
                M_->shs_);            // [N,*,3]
            TORCH_CHECK(rgbs.size(0) == N && rgbs.size(1) == 3, "SH_eval must return [N,3]. Got ", rgbs.sizes());
            vp.rgbs = rgbs.contiguous();
        }
          else if (mode == "rand") {
            auto float_opts = M_->center_.options().dtype(torch::kFloat32);
            vp.rgbs = torch::rand({N, 3}, float_opts);
        } else if (mode == "dontcare") {
            auto float_opts = M_->center_.options().dtype(torch::kFloat32);
            vp.rgbs = torch::empty({N, 3}, float_opts);
        } else {
            TORCH_CHECK(false, "Unsupported color_mode: ", mode);
        }

        // -----------------------
        // 3) Subdivision priority
        // -----------------------
        if (M_->subdiv_p_.defined() && M_->subdiv_p_.numel() > 0) {
            auto sp = (M_->subdiv_p_.dim() == 1) ? M_->subdiv_p_.view({-1, 1}) : M_->subdiv_p_;
            TORCH_CHECK(sp.size(0) == N && sp.size(1) == 1, "subdiv_p must be [N,1], got ", sp.sizes());
            vp.subdiv_p = sp.contiguous();
        } else {
            vp.subdiv_p = torch::ones({N, 1}, float_opts);
        }

        return vp;
    };

    // Build raster settings
    sv::rasterizer::RasterSettings rs;
    rs.color_mode    = (color_mode ? std::string(color_mode) : std::string("sh"));
    rs.n_samp_per_vox = n_samp_per_vox;
    rs.image_width   = w;
    rs.image_height  = h;
    rs.tanfovx       = cam.tanfovx;
    rs.tanfovy       = cam.tanfovy;
    rs.cx            = cam.cx * w_ss;
    rs.cy            = cam.cy * h_ss;
    rs.w2c_matrix    = cam.w2c;
    rs.c2w_matrix    = cam.c2w;
    rs.bg_color      = static_cast<float>(M_->white_background_); // 1.0 for white bg, else 0.0
    rs.near          = cam.near;                                   // keep from MiniCam
    rs.need_depth    = output_depth;
    rs.need_normal   = output_normal;
    rs.track_max_w   = track_max_w;
    // Regularization hooks
    if (other_opt.lambda_R_concen)  rs.lambda_R_concen  = *other_opt.lambda_R_concen;
    if (other_opt.lambda_ascending) rs.lambda_ascending = *other_opt.lambda_ascending;
    if (other_opt.lambda_dist)      rs.lambda_dist      = *other_opt.lambda_dist;
    // Ground-truth color for concentration loss (resize if SS>1)
    if (rs_gt_color.defined())      rs.gt_color = rs_gt_color;  // CHW, float32, on CUDA

    // Rasterize
    torch::Tensor color, depth, normal, T, max_w;
    std::tie(color, depth, normal, T, max_w) =
        sv::rasterizer::rasterize_voxels(   // <-- check API name/signature
            rs,
            M_->oct_path_,                   // [N,1] int64 or [N]? keep consistent with your wrapper
            M_->center_,                     // [N,3]
            M_->size_.view({-1,1}),          // [N,1] float
            vox_fn);

    // post-process + pack
    out.raw_color  = color;
    out.raw_depth  = depth;
    out.raw_normal = normal;
    out.raw_T      = T;
    out.max_w      = max_w;

    // Background composition (parity to Python)
    if (rand_bg) {
        auto noise = torch::rand_like(color);
        color = color + T * noise;
    } else if (!M_->white_background_ && !M_->black_background_) {
        // “gray-world” style: add T * mean(color)
        auto mean_c = color.mean({1,2}, /*keepdim=*/true); // [3,1,1]
        color = color + T * mean_c;
    }

    // if (use_auto_exposure && cam.auto_exposure_enabled) {
    //     // If you have an exposure routine on MiniCam, apply it:
    //     color = cam.auto_exposure_apply(color);  // <-- check API
    // }

    // If SS was used, resize back to source size:
    if (h != h_src || w != w_src) {
        if (color.defined() && color.numel()>0) color = resize_rendering(color, h_src, w_src);
        if (depth.defined() && depth.numel()>0) depth = resize_rendering(depth, h_src, w_src);
        if (normal.defined()&& normal.numel()>0) normal = resize_rendering(normal, h_src, w_src);
        if (T.defined()     && T.numel()>0)      T      = resize_rendering(T, h_src, w_src);
    }

    color.clamp_(0.0f, 1.0f);
    out.color  = color;
    out.depth  = output_depth  ? depth  : torch::Tensor();
    out.normal = output_normal ? normal : torch::Tensor();
    out.T      = output_T      ? T      : torch::Tensor();
    return out;
}

torch::Tensor VoxelRenderer::resize_rendering(const torch::Tensor& t, int out_h, int out_w) {
    if (!t.defined() || t.numel()==0) return t;
    if ((int)t.size(-2) == out_h && (int)t.size(-1) == out_w) return t;
    // expects [C,H,W]; upsample op expects NCHW
    auto in = t;
    bool squeezed2D = false;
    if (in.dim()==2) {                // Depth/T: [H,W] -> [1,H,W]
        in = in.unsqueeze(0);
        squeezed2D = true;
    }
    if (in.dim()==3) {                // [C,H,W] -> [1,C,H,W]
        in = in.unsqueeze(0);
    }
    // at::upsample_bilinear2d(input, output_size, align_corners, scales_h, scales_w)
    auto out = at::upsample_bilinear2d(in, {out_h, out_w}, /*align_corners=*/false, c10::nullopt, c10::nullopt);
    out = out.squeeze(0);             // back to [C,H,W] or [1,H,W]
    if (squeezed2D) out = out.squeeze(0);  // back to [H,W]
    return out.contiguous();
}

} // namespace sv
