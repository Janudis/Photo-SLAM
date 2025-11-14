#pragma once
#include <torch/torch.h>
#include <tuple>
#include <string>
#include "include_voxel/mini_cam.h"
#include "include_voxel/voxel_model.h"

namespace sv::rasterizer {

// ---- Small POD matching SVRaster’s RasterSettings ----
struct RasterSettings {
    std::string color_mode;
    int   n_samp_per_vox;
    int   image_width;
    int   image_height;
    float tanfovx;
    float tanfovy;
    float cx;
    float cy;
    torch::Tensor w2c_matrix; // [4,4] float32 CUDA
    torch::Tensor c2w_matrix; // [4,4] float32 CUDA
    float bg_color = 0.f;
    float near = 0.01f;
    bool  need_depth = false;
    bool  need_normal = false;
    bool  track_max_w = false;
    // optional (regularizers)
    float lambda_R_concen = 0.f;
    float lambda_ascending = 0.f;
    float lambda_dist = 0.f;
    torch::Tensor gt_color; // [3,H,W] if used
    bool debug = false;
};

// Vox params struct your renderer’s vox_fn returns
struct VoxParams {
    torch::Tensor geos;      // [N,8] or [K,8] if subselect
    torch::Tensor rgbs;      // [N,3]
    torch::Tensor subdiv_p;  // [N,1]
};

// Autograd functions exposed as simple helpers:
torch::Tensor SH_eval(
    int active_sh_degree,
    const torch::Tensor& idx,          // [K] long or empty
    const torch::Tensor& vox_centers,  // [N,3] (or viewdirs if provided)
    const torch::Tensor& cam_pos,      // [3]
    const torch::Tensor& viewdir,      // [K,3] or undefined
    const torch::Tensor& sh0,          // [N,3]
    const torch::Tensor& shs           // [N, M-1, 3]
);

// Gather geo params with triinterp, autograd-enabled:
torch::Tensor GatherGeoParams(
    const torch::Tensor& vox_key,   // [N,8] long
    const torch::Tensor& care_idx,  // [K] long
    const torch::Tensor& grid_pts   // [num_grid_pts]
);

// One-time preprocess to build geomBuffer & n_duplicates:
std::tuple<torch::Tensor, torch::Tensor> rasterize_preprocess(
    int image_width, int image_height,
    float tanfovx, float tanfovy,
    float cx, float cy,
    const torch::Tensor& w2c, const torch::Tensor& c2w,
    float near_plane,
    const torch::Tensor& octree_paths,  // [N] or [N,1] long
    const torch::Tensor& vox_centers,   // [N,3] float
    const torch::Tensor& vox_lengths,   // [N,1] float
    bool debug);

// The main differentiable voxel rasterizer:
//   rs – settings (above)
//   octree_paths – [N] or [N,1] long
//   vox_centers  – [N,3] float
//   vox_lengths  – [N,1] float
//   vox_fn(idx,cam_pos,color_mode,viewdir) → VoxParams
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_voxels(
    const RasterSettings& rs,
    const torch::Tensor& octree_paths,
    const torch::Tensor& vox_centers,
    const torch::Tensor& vox_lengths,
    const std::function<VoxParams(const torch::Tensor&, const torch::Tensor&, const char*, const torch::Tensor&)>& vox_fn
);

// in include_voxel/voxel_rasterizer.h (inside namespace sv::rasterizer)
torch::Tensor mark_n_duplicates( /* same signature as above but without return_buffer */ );
torch::Tensor mark_max_samp_rate(const std::vector<sv::MiniCam>& cams,
                                 const torch::Tensor& octree_paths,
                                 const torch::Tensor& vox_centers,
                                 const torch::Tensor& vox_lengths,
                                 float near);
torch::Tensor mark_near(const std::vector<sv::MiniCam>& cams,
                        const torch::Tensor& octree_paths,
                        const torch::Tensor& vox_centers,
                        const torch::Tensor& vox_lengths,
                        float near);

                        
// Integer <-> octpath helpers (optional):
torch::Tensor ijk_2_octpath(const torch::Tensor& ijk, const torch::Tensor& octlevel);
torch::Tensor octpath_2_ijk(const torch::Tensor& octpath, const torch::Tensor& octlevel);

} // namespace sv::rasterizer
