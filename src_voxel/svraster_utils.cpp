#include "include_voxel/svraster_utils.h"
#include <unordered_map>
#include <tuple>

namespace sv::oct {

static inline at::Tensor to_i8_col(const at::Tensor& L) {
    if (L.dtype() == at::kChar) return L.contiguous();
    if (L.dtype() == at::kLong) return L.to(at::kChar);
    TORCH_CHECK(false, "level tensor must be int8 or int64");
}

// Ensure integer levels as a column [N,1], dtype int64 (ldexp wants integer exp).
static inline at::Tensor to_level_col(const at::Tensor& L) {
    TORCH_CHECK(L.defined(), "level tensor undefined");
    TORCH_CHECK(L.is_cuda(), "level tensor must be CUDA");
    at::Tensor Li = L;
    // Accept [N] or [N,1]
    if (Li.dim() == 1) {
        Li = Li.view({Li.size(0), 1});
    } else {
        TORCH_CHECK(Li.dim() == 2 && Li.size(1) == 1, "level must be [N] or [N,1], got ", Li.sizes());
    }
    // SVRaster stores octlevel as int8; ldexp expects integer exponents → int64 is safe.
    if (Li.dtype() == at::kChar)      Li = Li.to(at::kLong);
    else if (Li.dtype() == at::kLong) /* ok */;
    else TORCH_CHECK(false, "level dtype must be int8 or int64, got ", Li.dtype());
    return Li.contiguous();
}

// Ensure vox_size is float column [*,1] (handles [1], [N], [N,1]).
static inline at::Tensor to_voxsize_col(const at::Tensor& V, at::ScalarType dtype, c10::Device dev) {
    TORCH_CHECK(V.defined(), "vox_size undefined");
    at::Tensor Vs = V.to(dev, dtype);
    if (Vs.dim() == 0) {              // scalar -> [1,1]
        Vs = Vs.view({1, 1});
    } else if (Vs.dim() == 1) {       // [N] -> [N,1]
        Vs = Vs.view({Vs.size(0), 1});
    } else {
        TORCH_CHECK(Vs.dim() == 2 && Vs.size(1) == 1, "vox_size must be [N], [N,1], or [1], got ", Vs.sizes());
    }
    return Vs.contiguous();
}

// --- API (mirror SVRaster) -----------------------------------------------

// Python: level_2_vox_size(scene_extent:[1], octlevel:[N,1]) -> [N,1]
at::Tensor level2voxsize(const at::Tensor& scene_extent, const at::Tensor& level) {
    TORCH_CHECK(scene_extent.defined() && scene_extent.is_cuda(), "scene_extent must be CUDA");
    TORCH_CHECK(scene_extent.numel() == 1, "scene_extent must be [1], got ", scene_extent.sizes());
    auto dev   = scene_extent.device();
    auto s_f32 = scene_extent.to(dev, at::kFloat).view({1,1}); // [1,1] for broadcast

    // Accept int8/int64 and either [N] or [N,1]
    auto Lcol = to_level_col(level);                             // [N,1] int64
    // ldexp(x, e) computes x * 2**e → we want 2**(-L)
    auto negL = -Lcol;
    // Broadcast: [1,1] ldexp [N,1] → [N,1]
    auto vox_size = at::ldexp(s_f32, negL);                      // [N,1], float32
    return vox_size.contiguous();
}

// Python: vox_size_2_level(scene_extent:[1], vox_size:[N,1]) -> [N,1] (float)
at::Tensor voxsize2levelf(const at::Tensor& scene_extent, const at::Tensor& vox_size) {
    TORCH_CHECK(scene_extent.defined() && scene_extent.is_cuda(), "scene_extent must be CUDA");
    TORCH_CHECK(scene_extent.numel() == 1, "scene_extent must be [1], got ", scene_extent.sizes());
    auto dev   = scene_extent.device();
    auto s_f32 = scene_extent.to(dev, at::kFloat);               // [1]

    // Accept [1], [N], [N,1]; cast to float column on same device
    auto Vcol = to_voxsize_col(vox_size, at::kFloat, dev);       // [*,1]
    // -log2(vox_size / scene_extent)
    auto Lf = -at::log2(Vcol / s_f32);
    return Lf.contiguous();
}

at::Tensor xyz2octpath(const at::Tensor& xyz, const at::Tensor& octlevel,
                       const at::Tensor& scene_center, const at::Tensor& scene_extent) {
    TORCH_CHECK(xyz.dim()==2 && xyz.size(1)==3, "xyz [N,3]");
    TORCH_CHECK(scene_center.numel()==3 && scene_extent.numel()==1, "scene shapes");
    auto scene_min = scene_center - 0.5 * scene_extent; // [3]
    auto vox_size  = level2voxsize(scene_extent, octlevel); // [N,1]
    auto ijk = ((xyz - scene_min) / vox_size).floor().to(at::kLong); // [N,3]

    // bound checks: 0 <= ijk < 2^L
    auto L = to_i8_col(octlevel).to(at::kLong);            // [N,1]
    auto limit = at::bitwise_left_shift(
                at::ones_like(L, at::kLong), L);        // 1 << L  → [N,1]
    TORCH_CHECK((ijk < limit).all().item<bool>(), "xyz exceed scene bounds");

    return sv::rasterizer::ijk_2_octpath(ijk, to_i8_col(octlevel));
}

std::pair<at::Tensor, at::Tensor> octpath_decoding(const at::Tensor& octpath,
                                                 const at::Tensor& octlevel,
                                                 const at::Tensor& scene_center,
                                                 const at::Tensor& scene_extent) {
    auto vox_ijk  = sv::rasterizer::octpath_2_ijk(octpath, to_i8_col(octlevel)); // [N,3] int64
    auto vox_size = level2voxsize(scene_extent, octlevel);                      // [N,1] float
    auto scene_min = scene_center - 0.5 * scene_extent;                         // [3]
    auto vox_center = scene_min + (vox_ijk.to(at::kFloat) + 0.5) * vox_size;   // [N,3]
    return {vox_center.contiguous(), vox_size.contiguous()};
}

at::Tensor gen_gridpoints_coordinate(const at::Tensor& octpath,
                                     const at::Tensor& octlevel)
{
    TORCH_CHECK(octpath.dtype() == at::kLong, "octpath must be int64 [N,1]");
    TORCH_CHECK(octlevel.dtype() == at::kChar, "octlevel must be int8 [N,1]");
    TORCH_CHECK(octpath.sizes() == octlevel.sizes(), "octpath/octlevel shape mismatch");

    const auto dev = octpath.device();
    const int64_t N = octpath.size(0);

    // 1) Decode voxel ijk at its own level: [N,3] int64
    auto vox_ijk = sv::rasterizer::octpath_2_ijk(octpath, octlevel); // [N,3] int64

    // 2) shift = (MAX_NUM_LEVELS - L)  (upcast to int64)
    const int MAXLV = sv::oct::MAX_NUM_LEVELS;
    auto lv2max = (MAXLV - octlevel.to(at::kLong)).contiguous();     // [N,1] int64

    // 3) base_grid_ijk = (vox_ijk << lv2max)  => [N,3]
    //    (broadcast [N,3] << [N,1])
    auto base_grid_ijk = at::bitwise_left_shift(vox_ijk, lv2max);     // [N,3] int64

    // 4) subtree_shift_int64 = [[0,0,0],[1,0,0],...,[1,1,1]]
    static const int64_t OFFS[8][3] = {
        {0,0,0}, {0,0,1}, {0,1,0}, {0,1,1},
        {1,0,0}, {1,0,1}, {1,1,0}, {1,1,1}
    };
    // Make a tiny device tensor once per call (cheap)
    auto offs = at::from_blob((void*)OFFS, {8,3}, at::TensorOptions().dtype(at::kLong))
                    .clone().to(dev); // [8,3] int64 on device

    // 5) Allocate the output: [N,8,3] int64
    auto gridpts = at::empty({N, 8, 3}, at::TensorOptions().dtype(at::kLong).device(dev));

    // 6) Fill each corner slice without making a giant broadcasted temp
    //    Equivalent to: base_grid_ijk.view(N,1,3) + (offs << lv2max.view(N,1,1))
    for (int v = 0; v < 8; ++v) {
        // scaled_v = (offs[v] << lv2max) → [N,3]  (offs[v] is [1,3], lv2max is [N,1])
        auto offs_v = offs.select(0, v).unsqueeze(0);       // [1,3]
        auto scaled_v = at::bitwise_left_shift(offs_v, lv2max); // [N,3]
        auto gp_v = base_grid_ijk + scaled_v;               // [N,3]
        gridpts.select(1, v).copy_(gp_v);                   // write into [:,v,:]
    }

    return gridpts.contiguous(); // [N,8,3] int64
}

at::Tensor compute_gridpoints_xyz(const at::Tensor& gridpts_ijk,
                                  const at::Tensor& scene_center,
                                  const at::Tensor& scene_extent) {
    TORCH_CHECK(gridpts_ijk.dim()==2 && gridpts_ijk.size(1)==3, "gridpts_ijk [N,3] int64");
    TORCH_CHECK(gridpts_ijk.dtype()==at::kLong, "gridpts_ijk must be int64");

    auto dev = scene_center.device();
    auto fopts = scene_center.options().dtype(at::kFloat);

    auto scene_center_f = scene_center.to(fopts);
    auto scene_extent_f = scene_extent.to(fopts);
    auto scene_min = scene_center_f - 0.5 * scene_extent_f;

    // SVRaster: level_2_vox_size(scene_extent, torch.tensor(MAX_NUM_LEVELS, int64, cuda))
    auto lvl = at::scalar_tensor((int64_t)MAX_NUM_LEVELS,
                                 at::TensorOptions().dtype(at::kLong).device(dev));
    // ensure level2voxsize returns float32 on same device (scalar or [1])
    auto finest = level2voxsize(scene_extent, lvl).to(at::kFloat);     // scalar/[1]

    // gridpts_ijk is int64; cast to float32 for multiply
    auto grid_f = gridpts_ijk.to(at::kFloat);                          // [N,3] float
    auto gridxyz = scene_min + grid_f * finest;                        // [N,3]
    return gridxyz.contiguous();
}

std::pair<at::Tensor, at::Tensor>
build_grid_pts_link(const at::Tensor& octpath, const at::Tensor& octlevel) {
    TORCH_CHECK(octpath.sizes() == octlevel.sizes(),
                "build_grid_pts_link: octpath and octlevel shape mismatch");
    // 1) corners @ finest level
    auto gp = gen_gridpoints_coordinate(octpath, octlevel).contiguous();   // [N,8,3] int64
    auto gp_flat = gp.reshape({-1, 3});                          // [N*8,3]

    // Mirror Python: unique(dim=0, return_inverse=True), default sorted=False
    at::Tensor grid_pts_key, inv;
    // at::unique_dim returns (output, inverse, counts) in C++ API
    auto tup = at::unique_dim(gp_flat, /*dim=*/0,
                                /*sorted=*/false,
                                /*return_inverse=*/true,
                                /*return_counts=*/false);
    grid_pts_key = std::get<0>(tup).contiguous();  // [M,3] int64
    inv          = std::get<1>(tup).contiguous();  // [N*8] int64

    // 2) reshape inverse to [N,8] (exact SVRaster semantics: use -1 not octpath.size(0))
    auto vox_key = inv.reshape({-1, 8}).contiguous();            // [N,8] int64
    return {grid_pts_key, vox_key};
}

std::pair<at::Tensor, at::Tensor> gen_children(const at::Tensor& octpath, const at::Tensor& octlevel) {
    using namespace sv;
    auto N = octpath.size(0);
    auto dev = octpath.device();
    TORCH_CHECK(octpath.dtype()==at::kLong && octlevel.dtype()==at::kChar, "types");
    auto Lp1 = to_i8_col(octlevel) + 1;
    TORCH_CHECK(Lp1.max().item<int8_t>() <= MAX_NUM_LEVELS, "max level out of bound after subdivision.");
    auto children = at::arange(0,8, at::TensorOptions().dtype(at::kLong).device(dev)); // [8]
    // bit position for each voxel after inc level
    auto bit = (sv::oct::MAX_NUM_LEVELS - Lp1.to(at::kLong));     // [N,1]
    auto base = octpath.view({N,1}).to(at::kLong);                // [N,1]

    // prepare shifts of shape [N,8]
    auto shift_N8 = (3 * bit).expand({N,1}).repeat({1,8});        // [N,8]
    auto children_N8 = children.view({1,8}).expand({N,8});        // [N,8]
    auto child_bits = at::bitwise_left_shift(children_N8, shift_N8); // [N,8]

    // OR with base broadcast to [N,8]
    auto base_N8 = base.expand({N,8});
    auto child_paths = at::bitwise_or(base_N8, child_bits);       // [N,8]

    auto out_path  = child_paths.reshape({-1,1}).contiguous();    // [N*8,1]
    auto out_level = Lp1.repeat_interleave(8, 0).contiguous();    // [N*8,1]
    return std::make_pair(out_path, out_level);
}

std::pair<at::Tensor, at::Tensor>
clamp_level(const at::Tensor& octpath, const at::Tensor& octlevel, int max_lv) {
    TORCH_CHECK(octpath.sizes() == octlevel.sizes(),
                "clamp_level: octpath and octlevel shape mismatch");
    TORCH_CHECK(max_lv >= 1 && max_lv <= sv::oct::MAX_NUM_LEVELS, "max_lv out of bounds");

    const int num_bits = 3 * std::max(0, sv::oct::MAX_NUM_LEVELS - max_lv);
    // Mask path and clamp level
    auto masked  = at::bitwise_left_shift(
                     at::bitwise_right_shift(octpath, num_bits), num_bits
                   ).contiguous();                                // [N,1] int64
    auto L_i8    = octlevel.clamp_max(max_lv).to(at::kChar).contiguous(); // [N,1] int8

    // Stack like Python: shape [2, N, 1], then unique over dim=1 (voxel axis)
    auto st = at::stack({masked, L_i8.to(at::kLong)}, /*dim=*/0); // [2,N,1], cast to long to unique together
    auto tup = at::unique_dim(st, /*dim=*/1,
                              /*sorted=*/true,
                              /*return_inverse=*/false,
                              /*return_counts=*/false);
    auto uniq = std::get<0>(tup).contiguous();    // [2, M, 1] int64
    auto out_octpath = uniq.index({0}).contiguous();                 // [M,1] int64
    auto out_octlevel = uniq.index({1}).to(at::kChar).contiguous();  // [M,1] int8
    return {out_octpath, out_octlevel};
}

} // namespace sv::oct
