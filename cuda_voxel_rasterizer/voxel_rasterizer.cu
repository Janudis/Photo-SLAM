#include "forward.h"
#include "backward.h"
#include <torch/extension.h>

using torch::Tensor;

// === Forward ===============================================================
Tensor voxel_rasterize_forward(Tensor vox,          // [M,16] layout we define below
                               Tensor K, Tensor Twc,
                               int H, int W) {
    /*
     * VOX LAYOUT we pick (float32):
     * [0:3]  : xyz center
     * [3]    : half-length (voxel size /2)
     * [4:12] : 8 densities
     * [12:15]: rgb (we ignore SH & feats for smoke test)
     * [15]   : octree level  (float, will cast to int)
     *
     * dims = 16
     */
    TORCH_CHECK(vox.dim()==2 && vox.size(1)>=16, "vox tensor bad shape");
    int64_t M = vox.size(0);

    // split
    Tensor vox_centers = vox.slice(1,0,3).contiguous();
    Tensor vox_lengths = vox.slice(1,3,4).squeeze(1).contiguous();
    Tensor geos        = vox.slice(1,4,12).contiguous();
    Tensor rgbs        = vox.slice(1,12,15).contiguous();
    Tensor levels_f    = vox.slice(1,15,16).squeeze(1).contiguous();

    // simplistic octree_path = levels << something | idx
    // for now just use sequential ids so ordering works.
    Tensor octree_paths = torch::arange(M, vox.options().dtype(torch::kInt64));

    // empty placeholders
    Tensor feats       = torch::empty({M,0}, vox.options());
    Tensor geomBuffer  = torch::empty({0}, torch::TensorOptions().dtype(torch::kByte).device(vox.device()));
    Tensor background  = torch::tensor({0.f,0.f,0.f}, vox.options());

    // call SVRaster
    auto out_tuple = FORWARD::rasterize_voxels(
            /*vox_geo_mode*/      2,             // tri-interp order 2
            /*density_mode*/      0,             // EXP_LINEAR_11_MODE
            W, H,
            /*tan_fovx*/          1.f,           // placeholder
            /*tan_fovy*/          1.f,
            /*cx,cy*/             0.f, 0.f,
            /*w2c*/               K,             // we’ll ignore intern.
            /*c2w*/               Twc,
            background,
            /*cam_mode*/          0,             // perspective
            /*need_depth*/        false,
            /*need_distortion*/   false,
            /*need_normal*/       false,
            /*track_max_w*/       false,

            octree_paths,
            vox_centers,
            vox_lengths,
            geos,
            rgbs,
            feats,
            geomBuffer,
            /*debug*/             false);

    Tensor out_color = std::get<3>(out_tuple);   // 3×H×W
    return out_color;
}

// === Backward (stub for now) ===============================================
std::vector<Tensor> voxel_rasterize_backward(Tensor /*dL_dI*/,
                                             Tensor /*vox*/, Tensor /*K*/, Tensor /*Twc*/,
                                             int /*H*/, int /*W*/) {
    TORCH_CHECK(false, "voxel rasterizer backward not implemented yet");
    return {};
}
