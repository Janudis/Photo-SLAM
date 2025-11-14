#pragma once
#include <torch/torch.h>
#include "include_voxel/mini_cam.h"
#include "include_voxel/voxel_rasterizer.h"
#include "include_voxel/voxel_model.h"

namespace sv::act {

/** rgb in [N,3], range ~[0,1] -> SH0 “DC” coef */
inline at::Tensor rgb2shzero(const at::Tensor& rgb) {
    TORCH_CHECK(rgb.dim()==2 && rgb.size(1)==3, "rgb must be [N,3]");
    // (x - 0.5) / 0.28209479177387814
    return (rgb - 0.5) / 0.28209479177387814;
}

/** inverse of rgb2shzero */
inline at::Tensor shzero2rgb(const at::Tensor& sh0) {
    TORCH_CHECK(sh0.dim()==2 && sh0.size(1)==3, "sh0 must be [N,3]");
    return (sh0 * 0.28209479177387814) + 0.5;
}

} // namespace sv::act

namespace sv::oct {

inline constexpr int MAX_NUM_LEVELS = 16; // keep in sync with VoxelModel::max_num_levels_
// -------- Scalar/tensor level<->vox_size --------

/** level -> voxel size: vox_size = scene_extent * 2^{-L} */
at::Tensor level2voxsize(const at::Tensor& scene_extent /*[1] float*/,
                         const at::Tensor& level       /*[N,1] int8 or int64*/);

/** vox_size -> level (float): L = -log2(vox_size / scene_extent) */
at::Tensor voxsize2levelf(const at::Tensor& scene_extent /*[1] float*/,
                          const at::Tensor& vox_size     /*[N,1] float*/);

// -------- xyz<->octpath / centers / sizes --------

/** Given xyz, build octpath at levels; throws if out-of-bounds. */
at::Tensor xyz2octpath(const at::Tensor& xyz        /*[N,3] float*/,
                       const at::Tensor& octlevel   /*[N,1] int8*/,
                       const at::Tensor& scene_center /*[3] float*/,
                       const at::Tensor& scene_extent /*[1] float*/);

/** Decode centers and sizes from octpath/level */
std::pair<at::Tensor, at::Tensor> octpath_decoding(const at::Tensor& octpath  /*[N,1] int64*/,
                                                  const at::Tensor& octlevel /*[N,1] int8*/,
                                                  const at::Tensor& scene_center /*[3]*/,
                                                  const at::Tensor& scene_extent /*[1]*/);

// -------- Grid-points / links (voxel corners) --------

/** Generate the 8 corner integer coords at finest level for each voxel. */
at::Tensor gen_gridpoints_coordinate(const at::Tensor& octpath  /*[N,1] int64*/,
                           const at::Tensor& octlevel /*[N,1] int8*/);

/** Convert integer grid coordinates to world xyz (finest level spacing). */
at::Tensor compute_gridpoints_xyz(const at::Tensor& gridpts_ijk /*[M,3] int64*/,
                              const at::Tensor& scene_center /*[3]*/,
                              const at::Tensor& scene_extent /*[1]*/);

/** Build unique grid key and index each voxel’s 8 corners. */
std::pair<at::Tensor, at::Tensor> build_grid_pts_link(const at::Tensor& octpath  /*[N,1] int64*/,
                                                       const at::Tensor& octlevel /*[N,1] int8*/);

// -------- Children / clamping --------

/** Subdivide each voxel into 8 children. */
std::pair<at::Tensor, at::Tensor> gen_children(const at::Tensor& octpath /*[N,1] int64*/,
                                                const at::Tensor& octlevel/*[N,1] int8*/);

/** Mask lower bits to clamp_levels; return unique (octpath, octlevel). */
std::pair<at::Tensor, at::Tensor> clamp_level(const at::Tensor& octpath /*[N,1] int64*/,
                                               const at::Tensor& octlevel/*[N,1] int8*/,
                                               int max_lv);

} // namespace sv::oct
