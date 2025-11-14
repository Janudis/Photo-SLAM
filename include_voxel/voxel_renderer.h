#pragma once
#include <torch/torch.h>
#include <optional>
#include <tuple>
#include <string>

#include "include_voxel/mini_cam.h"
#include "include_voxel/render_opts.h"
#include "include_voxel/voxel_rasterizer.h"  // sv::rasterizer::RasterSettings, rasterize_voxels
// #include "include_voxel/voxel_model.h"

namespace sv {

class VoxelModel;

/** Output package to mirror the Python renderer’s dictionary. */
struct RenderOutput {
    torch::Tensor color;     // [3,H,W]
    torch::Tensor depth;     // [1,H,W] or empty
    torch::Tensor normal;    // [3,H,W] or empty
    torch::Tensor T;         // [1,H,W] or empty
    torch::Tensor max_w;     // [N,1]   (per-voxel stat) if track_max_w

    // “raw_*” before SS downsample (for parity with Python)
    torch::Tensor raw_color;
    torch::Tensor raw_depth;
    torch::Tensor raw_normal;
    torch::Tensor raw_T;
};

/**
 * VoxelRenderer: a pure-C++ renderer aligned with SVRaster’s SVRenderer.
 *
 * It *reads* fields from a VoxelModel (non-owning pointer), gathers per-voxel
 * geo params (either frozen or on-the-fly), evaluates SH colors, builds
 * RasterSettings, calls rasterizer, and returns a RenderOutput.
 */
class VoxelRenderer {
public:
    explicit VoxelRenderer(const VoxelModel* model);

    // Freeze/unfreeze grid-point geo:
    void freezeVoxGeo();    // pre-gather grid-point params to per-voxel [N,8,*]
    void unfreezeVoxGeo();  // resume training on _geo_grid_pts_

    // Main render entry:
    RenderOutput render(
        const MiniCam& cam,
        int image_height,
        int image_width,
        const at::Tensor& gt_image = at::Tensor(),
        const char* color_mode = nullptr,
        bool track_max_w = false,
        std::optional<float> ss = std::nullopt,
        bool output_depth = false,
        bool output_normal = false,
        bool output_T = false,
        bool rand_bg = false,
        bool use_auto_exposure = false,
        const RenderOpts& other_opt = RenderOpts()
    ) const;

private:
    // Helpers
    static torch::Tensor resize_rendering(const torch::Tensor& t, int out_h, int out_w);

private:
    const VoxelModel* M_;               // non-owning
    mutable torch::Tensor frozen_vox_geo_; // [N,8,*] when frozen; undefined otherwise
};

} // namespace sv
