#include "include_voxel/voxel_rasterizer.h"
#include <torch/torch.h>
#include <tuple>
#include <stdexcept>

// === SVRaster CUDA headers you copied ===
#include "cuda_voxel_rasterizer/sh_compute.h"
#include "cuda_voxel_rasterizer/geo_params_gather.h"
#include "cuda_voxel_rasterizer/preprocess.h"
#include "cuda_voxel_rasterizer/forward.h"
#include "cuda_voxel_rasterizer/backward.h"
#include "cuda_voxel_rasterizer/utils.h"

// Bring CUDA symbols into scope (namespaces taken from the headers)
using SH_COMPUTE::sh_compute;
using SH_COMPUTE::sh_compute_bw;

using GEO_PARAMS_GATHER::gather_triinterp_geo_params;
using GEO_PARAMS_GATHER::gather_triinterp_geo_params_bw;

// NOTE: we DO NOT bring PREPROCESS/ FORWARD/ BACKWARD into scope by name,
// because we have local functions with the same names; we’ll call them with
// explicit "::PREPROCESS::..." / "::FORWARD::..." / "::BACKWARD::..." to avoid clashes.

namespace sv::rasterizer {
using namespace torch::indexing;

// ----------------- Preprocess (calls CUDA) -----------------
std::tuple<torch::Tensor, torch::Tensor> rasterize_preprocess(
    int image_width, int image_height,
    float tanfovx, float tanfovy,
    float cx, float cy,
    const torch::Tensor& w2c, const torch::Tensor& c2w,
    float near_plane,
    const torch::Tensor& octree_paths,
    const torch::Tensor& vox_centers,
    const torch::Tensor& vox_lengths,
    bool debug) 
{
    // Call CUDA preprocess, not ourselves:
    return ::PREPROCESS::rasterize_preprocess(
        image_width, image_height, tanfovx, tanfovy, cx, cy, w2c, c2w, near_plane,
        octree_paths, vox_centers, vox_lengths, debug
    );
}

// ----------------- Main differentiable rasterizer -----------------
struct RasterizeVoxelsFunction : public torch::autograd::Function<RasterizeVoxelsFunction> {
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        RasterSettings rs,
        torch::Tensor geomBuffer,
        torch::Tensor octree_paths,
        torch::Tensor vox_centers,
        torch::Tensor vox_lengths,
        torch::Tensor geos,
        torch::Tensor rgbs,
        torch::Tensor subdiv_p
    ) {
        const bool need_distortion = rs.lambda_dist > 0;

        int R = 0;  // num_rendered (CUDA returns int first)
        torch::Tensor binningBuffer, imgBuffer;
        torch::Tensor out_color, out_depth, out_normal, out_T, max_w;

        // Call the CUDA forward (fully-qualified to avoid picking our wrapper)
        std::tie(R, binningBuffer, imgBuffer, out_color, out_depth, out_normal, out_T, max_w) =
            ::FORWARD::rasterize_voxels(
                rs.n_samp_per_vox,
                rs.image_width, rs.image_height,
                rs.tanfovx, rs.tanfovy,
                rs.cx, rs.cy,
                rs.w2c_matrix, rs.c2w_matrix,
                rs.bg_color,
                rs.need_depth, need_distortion, rs.need_normal, rs.track_max_w,
                octree_paths, vox_centers, vox_lengths, geos, rgbs,
                geomBuffer,
                rs.debug
            );

        // Save scalars / flags
        ctx->saved_data["R"]               = R;
        ctx->saved_data["n_samp_per_vox"]  = rs.n_samp_per_vox;
        ctx->saved_data["image_width"]     = rs.image_width;
        ctx->saved_data["image_height"]    = rs.image_height;
        ctx->saved_data["tanfovx"]         = rs.tanfovx;
        ctx->saved_data["tanfovy"]         = rs.tanfovy;
        ctx->saved_data["cx"]              = rs.cx;
        ctx->saved_data["cy"]              = rs.cy;
        ctx->saved_data["bg_color"]        = rs.bg_color;
        ctx->saved_data["lambda_R_concen"] = rs.lambda_R_concen;
        ctx->saved_data["lambda_ascending"]= rs.lambda_ascending;
        ctx->saved_data["lambda_dist"]     = rs.lambda_dist;
        ctx->saved_data["need_depth"]      = rs.need_depth;
        ctx->saved_data["need_normal"]     = rs.need_normal;
        ctx->saved_data["debug"]           = rs.debug;
        // ctx->saved_data["w2c_matrix"]           = rs.w2c_matrix;
        // ctx->saved_data["c2w_matrix"]           = rs.c2w_matrix;
        // ctx->saved_data["gt_color"]           = rs.gt_color;

        // Save tensors for backward
        ctx->save_for_backward({
            octree_paths, vox_centers, vox_lengths, geos, rgbs,
            geomBuffer, binningBuffer, imgBuffer, out_T, out_depth, out_normal,
            rs.w2c_matrix, rs.c2w_matrix, rs.gt_color
        });

        // mark_non_differentiable expects a list
        ctx->mark_non_differentiable({max_w});

        return { out_color, out_depth, out_normal, out_T, max_w };
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        auto dL_dout_color  = grad_outputs[0];
        auto dL_dout_depth  = grad_outputs[1];
        auto dL_dout_normal = grad_outputs[2];
        auto dL_dout_T      = grad_outputs[3];
        // grad_outputs[4] (max_w) is non-diff

        auto saved = ctx->get_saved_variables();
        int i = 0;
        auto octree_paths = saved[i++];
        auto vox_centers  = saved[i++];
        auto vox_lengths  = saved[i++];
        auto geos         = saved[i++];
        auto rgbs         = saved[i++];
        auto geomBuffer   = saved[i++];
        auto binningBuffer= saved[i++];
        auto imgBuffer    = saved[i++];
        auto out_T        = saved[i++];
        auto out_depth    = saved[i++];
        auto out_normal   = saved[i++];
        auto w2c          = saved[i++];
        auto c2w          = saved[i++];
        auto gt_color     = saved[i++];

        const int R                  = (int)ctx->saved_data["R"].toInt();
        const int n_samp_per_vox     = (int)ctx->saved_data["n_samp_per_vox"].toInt();
        const int image_width        = (int)ctx->saved_data["image_width"].toInt();
        const int image_height       = (int)ctx->saved_data["image_height"].toInt();
        const float tanfovx          = (float)ctx->saved_data["tanfovx"].toDouble();
        const float tanfovy          = (float)ctx->saved_data["tanfovy"].toDouble();
        const float cx               = (float)ctx->saved_data["cx"].toDouble();
        const float cy               = (float)ctx->saved_data["cy"].toDouble();
        const float bg_color         = (float)ctx->saved_data["bg_color"].toDouble();
        const float lambda_R_concen  = (float)ctx->saved_data["lambda_R_concen"].toDouble();
        const float lambda_ascending = (float)ctx->saved_data["lambda_ascending"].toDouble();
        const float lambda_dist      = (float)ctx->saved_data["lambda_dist"].toDouble();
        const bool need_depth        =        ctx->saved_data["need_depth"].toBool();
        const bool need_normal       =        ctx->saved_data["need_normal"].toBool();
        const bool debug             =        ctx->saved_data["debug"].toBool();
        // const float w2c               = (float)ctx->saved_data["w2c_matrix"].toDouble();
        // const float c2w               = (float)ctx->saved_data["c2w_matrix"].toDouble();
        // const float gt_color               = (float)ctx->saved_data["gt_color"].toDouble();

        torch::Tensor dL_dgeos, dL_drgbs, subdiv_p_bw;
        std::tie(dL_dgeos, dL_drgbs, subdiv_p_bw) = ::BACKWARD::rasterize_voxels_backward(
            R,                    // <-- pass int R here
            n_samp_per_vox, image_width, image_height,
            tanfovx, tanfovy, cx, cy,
            w2c, c2w, bg_color,
            octree_paths, vox_centers, vox_lengths, geos, rgbs,
            geomBuffer, binningBuffer, imgBuffer, out_T,
            dL_dout_color, dL_dout_depth, dL_dout_normal, dL_dout_T,
            lambda_R_concen, gt_color, lambda_ascending, lambda_dist,
            need_depth, need_normal, out_depth, out_normal,
            debug
        );

        torch::autograd::variable_list grads(8);
        grads[0] = torch::Tensor(); // rs (struct; no grad)
        grads[1] = torch::Tensor(); // geomBuffer
        grads[2] = torch::Tensor(); // octree_paths
        grads[3] = torch::Tensor(); // vox_centers
        grads[4] = torch::Tensor(); // vox_lengths
        grads[5] = dL_dgeos;        // geos
        grads[6] = dL_drgbs;        // rgbs
        grads[7] = subdiv_p_bw;     // subdiv_p
        return grads;
    }
};

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_voxels( // <-- this is the high-level wrapper (same name as CUDA one)
    const RasterSettings& rs,
    const torch::Tensor& octree_paths,
    const torch::Tensor& vox_centers,
    const torch::Tensor& vox_lengths,
    const std::function<VoxParams(const torch::Tensor&, const torch::Tensor&, const char*, const torch::Tensor&)>& vox_fn
) {

    // const auto N = octree_paths.numel();
    const int64_t N = octree_paths.size(0);
    TORCH_CHECK(vox_centers.size(0) == N, "Size mismatched: vox_centers[0] != N");
    TORCH_CHECK(vox_lengths.numel() == N, "Size mismatched: vox_lengths.numel() != N");
    TORCH_CHECK(vox_centers.dim()==2 && vox_centers.size(1)==3, "Expect vox_centers [N,3]");
    // Device parity (exactly as python ‘Device mismatch’ checks)
    auto dev = octree_paths.device();
    TORCH_CHECK(rs.w2c_matrix.device()==dev, "Device mismatch: w2c_matrix");
    TORCH_CHECK(rs.c2w_matrix.device()==dev, "Device mismatch: c2w_matrix");
    TORCH_CHECK(vox_centers.device()==dev,   "Device mismatch: vox_centers");
    TORCH_CHECK(vox_lengths.device()==dev,   "Device mismatch: vox_lengths");

    torch::Tensor n_dup, geomBuffer;
    std::tie(n_dup, geomBuffer) = rasterize_preprocess(
        rs.image_width, rs.image_height,
        rs.tanfovx, rs.tanfovy,
        rs.cx, rs.cy,
        rs.w2c_matrix, rs.c2w_matrix,
        rs.near,
        octree_paths, vox_centers, vox_lengths,
        rs.debug
    );

    // Build in-frustum indices (like Python)
    auto in_frusts_idx = torch::nonzero(n_dup > 0).squeeze(1);

    // Camera position (3,)
    auto cam_pos = rs.c2w_matrix.index({Slice(0,3), 3}).contiguous();

    // Ask the model (must return full-N tensors even though we pass a subset)
    auto vp = vox_fn(in_frusts_idx, cam_pos, rs.color_mode.c_str(), torch::Tensor());  // viewdir = None

    // Keep the same N-shape checks
    TORCH_CHECK(vp.geos.dim() >= 2 && vp.geos.size(0) == N && vp.geos.size(1) == 8, "geos must be [N,8,*]");
    TORCH_CHECK(vp.rgbs.dim() == 2 && vp.rgbs.size(0) == N && vp.rgbs.size(1) == 3, "rgbs must be [N,3]");
    TORCH_CHECK(vp.subdiv_p.dim() == 2 && vp.subdiv_p.size(0) == N && vp.subdiv_p.size(1) == 1, "subdiv_p must be [N,1]");

    if (rs.lambda_R_concen > 0.f) {
        TORCH_CHECK(rs.gt_color.defined() && rs.gt_color.dim()==3 &&
                    rs.gt_color.size(0)==3 &&
                    rs.gt_color.size(1)==rs.image_height &&
                    rs.gt_color.size(2)==rs.image_width,
                    "Expect gt_color in shape [3,H,W]");
        TORCH_CHECK(rs.gt_color.device()==dev, "Device mismatch: gt_color.");
    }

    auto outputs = RasterizeVoxelsFunction::apply(
        rs, geomBuffer, octree_paths, vox_centers, vox_lengths,
        vp.geos, vp.rgbs, vp.subdiv_p
    );
    return { outputs[0], outputs[1], outputs[2], outputs[3], outputs[4] };
}

// ----------------- SH_eval (autograd) -----------------
struct SHEvalFunction : public torch::autograd::Function<SHEvalFunction> {
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                                 int active_sh_degree,
                                 torch::Tensor idx,
                                 torch::Tensor vox_centers,
                                 torch::Tensor cam_pos,
                                 torch::Tensor viewdir,
                                 torch::Tensor sh0,
                                 torch::Tensor shs) {
        if (viewdir.defined() && viewdir.numel() > 0) {
            vox_centers = viewdir;
            cam_pos = torch::zeros_like(cam_pos);
        }

        auto rgbs = sh_compute(active_sh_degree, idx, vox_centers, cam_pos, sh0, shs);

        ctx->saved_data["active_sh_degree"] = active_sh_degree;
        ctx->saved_data["M"] = 1 + shs.size(1);

        torch::autograd::variable_list to_save;
        to_save.reserve(4);
        to_save.push_back(idx);
        to_save.push_back(vox_centers);
        to_save.push_back(cam_pos);
        to_save.push_back(rgbs);
        ctx->save_for_backward(to_save);
        return rgbs;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                   torch::autograd::variable_list grad_outputs) {
        auto dL_drgbs = grad_outputs[0];

        auto saved = ctx->get_saved_variables();
        auto idx         = saved[0];
        auto vox_centers = saved[1];
        auto cam_pos     = saved[2];
        auto rgbs        = saved[3];

        const int active_sh_degree = (int)ctx->saved_data["active_sh_degree"].toInt();
        const int M                = (int)ctx->saved_data["M"].toInt();

        torch::Tensor dL_dsh0, dL_dshs;
        std::tie(dL_dsh0, dL_dshs) = sh_compute_bw(
            active_sh_degree, M, idx, vox_centers, cam_pos, rgbs, dL_drgbs);

        return {
            torch::Tensor(), // active_sh_degree
            torch::Tensor(), // idx
            torch::Tensor(), // vox_centers
            torch::Tensor(), // cam_pos
            torch::Tensor(), // viewdir
            dL_dsh0,         // sh0
            dL_dshs          // shs
        };
    }
};

torch::Tensor SH_eval(
    int active_sh_degree,
    const torch::Tensor& idx,
    const torch::Tensor& vox_centers,
    const torch::Tensor& cam_pos,
    const torch::Tensor& viewdir,   // may be undefined/empty
    const torch::Tensor& sh0,
    const torch::Tensor& shs
) {
    auto opts_f = vox_centers.options().dtype(torch::kFloat32);
    auto opts_l = vox_centers.options().dtype(torch::kLong);

    // Ensure defined tensors for Autograd.apply
    torch::Tensor safe_idx     = idx.defined()     ? idx     : torch::empty({0}, opts_l);
    torch::Tensor safe_viewdir = (viewdir.defined() ? viewdir : torch::empty({0,3}, opts_f));

    return SHEvalFunction::apply(
        active_sh_degree, safe_idx, vox_centers, cam_pos, safe_viewdir, sh0, shs
    );
}

// ----------------- GatherGeoParams (autograd) -----------------
struct GatherGeoParamsFunction : public torch::autograd::Function<GatherGeoParamsFunction> {
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                                 torch::Tensor vox_key,
                                 torch::Tensor care_idx,
                                 torch::Tensor grid_pts) {
        TORCH_CHECK(vox_key.dim()==2 && vox_key.size(1)==8, "vox_key [N,8]");
        TORCH_CHECK(care_idx.dim()==1, "care_idx [K]");

        auto geo = gather_triinterp_geo_params(vox_key, care_idx, grid_pts);

        ctx->saved_data["num_grid_pts"] = (int64_t)grid_pts.numel();
        torch::autograd::variable_list to_save;
        to_save.reserve(2);
        to_save.push_back(vox_key);
        to_save.push_back(care_idx);
        ctx->save_for_backward(to_save);
        return geo;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                   torch::autograd::variable_list grad_outputs) {
        auto dL_dgeo = grad_outputs[0];
        auto saved = ctx->get_saved_variables();
        auto vox_key  = saved[0];
        auto care_idx = saved[1];
        const int64_t num_grid_pts = ctx->saved_data["num_grid_pts"].toInt();

        auto dL_dgrid_pts = gather_triinterp_geo_params_bw(vox_key, care_idx, num_grid_pts, dL_dgeo);

        torch::autograd::variable_list grads(3);
        grads[0] = torch::Tensor();   // d(vox_key)
        grads[1] = torch::Tensor();   // d(care_idx)
        grads[2] = dL_dgrid_pts;      // d(grid_pts)
        return grads;
    }
};

torch::Tensor GatherGeoParams(
    const torch::Tensor& vox_key,
    const torch::Tensor& care_idx,
    const torch::Tensor& grid_pts
) {
    return GatherGeoParamsFunction::apply(vox_key, care_idx, grid_pts);
}

// parity helper: python mark_n_duplicates(...)
torch::Tensor mark_n_duplicates(
    int image_width, int image_height,
    float tanfovx, float tanfovy,
    float cx, float cy,
    const torch::Tensor& w2c, const torch::Tensor& c2w, float near_plane,
    const torch::Tensor& octree_paths,
    const torch::Tensor& vox_centers,
    const torch::Tensor& vox_lengths,
    bool /*return_buffer*/ = false,          // kept for parity; we return only n_dup here
    bool debug = false)
{
    torch::Tensor n_dup, geom;
    std::tie(n_dup, geom) = rasterize_preprocess(
        image_width, image_height,
        tanfovx, tanfovy, cx, cy,
        w2c, c2w, near_plane,
        octree_paths, vox_centers, vox_lengths,
        debug
    );
    return n_dup; // if you need geom, add an overload returning the pair
}


// python: mark_max_samp_rate(cameras, octree_paths, vox_centers, vox_lengths, near=0.02)
torch::Tensor mark_max_samp_rate(
    const std::vector<sv::MiniCam>& cams,
    const torch::Tensor& octree_paths,
    const torch::Tensor& vox_centers,
    const torch::Tensor& vox_lengths,
    float near =0.02f)
{
    const auto dev = octree_paths.device();
    const auto N   = octree_paths.size(0);

    auto max_rate = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));

    for (const auto& cam : cams) {
        // preprocess for this camera
        auto n_dup = mark_n_duplicates(
            cam.width, cam.height,
            cam.tanfovx, cam.tanfovy,
            cam.cx, cam.cy,
            cam.w2c, cam.c2w, near,
            octree_paths, vox_centers, vox_lengths,
            /*return_buffer=*/false, /*debug=*/false);

        // z distance along camera forward
        auto zdist_all = ((vox_centers - cam.position) * cam.lookat).sum(-1); // [N]

        // visible & in front of near plane
        auto vis_mask = (n_dup > 0) & (zdist_all > near);
        auto vis_idx  = torch::nonzero(vis_mask).squeeze(1); // [K]

        if (vis_idx.numel() == 0) continue;

        auto zdist = zdist_all.index_select(0, vis_idx); // [K]
        auto samp_interval = zdist * cam.pix_size;       // [K]
        auto vlen = vox_lengths.index_select(0, vis_idx).squeeze(1); // [K]

        // per-voxel sampling rate
        auto samp_rate = vlen / samp_interval;          // [K]

        // max over views:
        auto cur_max = max_rate.index_select(0, vis_idx);     // [K]
        auto new_max = torch::maximum(cur_max, samp_rate);    // [K]
        max_rate.index_put_({vis_idx}, new_max);
    }
    return max_rate.contiguous();
}

// python: mark_near(cameras, octree_paths, vox_centers, vox_lengths, near=0.2)
torch::Tensor mark_near(
    const std::vector<sv::MiniCam>& cams,
    const torch::Tensor& octree_paths,
    const torch::Tensor& vox_centers,
    const torch::Tensor& vox_lengths,
    float near =0.2f)
{
    const auto dev = octree_paths.device();
    const auto N   = octree_paths.size(0);

    auto is_near = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(dev));

    for (const auto& cam : cams) {
        // preprocess for this camera
        auto n_dup = mark_n_duplicates(
            cam.width, cam.height,
            cam.tanfovx, cam.tanfovy,
            cam.cx, cam.cy,
            cam.w2c, cam.c2w, near,
            octree_paths, vox_centers, vox_lengths,
            /*return_buffer=*/false, /*debug=*/false);

        auto vis_idx = torch::nonzero(n_dup > 0).squeeze(1); // [K]
        if (vis_idx.numel() == 0) continue;

        auto vc_vis   = vox_centers.index_select(0, vis_idx); // [K,3]
        auto zdist    = ((vc_vis - cam.position) * cam.lookat).sum(-1); // [K]
        auto vlen_vis = vox_lengths.index_select(0, vis_idx).squeeze(1); // [K]

        auto near_mask = zdist <= (near + vlen_vis); // [K] bool

        // OR into the running mask
        auto cur = is_near.index_select(0, vis_idx);        // [K] bool
        is_near.index_put_({vis_idx}, cur | near_mask);
    }
    return is_near.contiguous();
}

// ----------------- Integer path helpers (call CUDA utils) -----------------
torch::Tensor ijk_2_octpath(const torch::Tensor& ijk, const torch::Tensor& octlevel) {
    TORCH_CHECK(ijk.defined() && octlevel.defined(), "ijk_2_octpath: inputs must be tensors");
    TORCH_CHECK(ijk.dim() == 2 && ijk.size(1) == 3, "ijk_2_octpath: ijk must be [N,3]");
    TORCH_CHECK(ijk.dtype() == torch::kInt64,       "ijk_2_octpath: ijk must be int64");
    TORCH_CHECK(octlevel.dtype() == torch::kInt8,   "ijk_2_octpath: octlevel must be int8");
    TORCH_CHECK(ijk.numel() == octlevel.numel() * 3,
                "ijk_2_octpath: ijk.numel() must equal 3 * octlevel.numel()");

    // Forward to CUDA implementation (utils.cu / utils.h)
    return ::UTILS::ijk_2_octpath(ijk, octlevel);
}

torch::Tensor octpath_2_ijk(const torch::Tensor& octpath, const torch::Tensor& octlevel) {
    TORCH_CHECK(octpath.defined() && octlevel.defined(), "octpath_2_ijk: inputs must be tensors");
    TORCH_CHECK(octpath.dtype() == torch::kInt64,       "octpath_2_ijk: octpath must be int64");
    TORCH_CHECK(octlevel.dtype() == torch::kInt8,       "octpath_2_ijk: octlevel must be int8");
    TORCH_CHECK(octpath.numel() == octlevel.numel(),
                "octpath_2_ijk: octpath.numel() must equal octlevel.numel()");

    // Forward to CUDA implementation (utils.cu / utils.h)
    return ::UTILS::octpath_2_ijk(octpath, octlevel);
}

} // namespace sv::rasterizer
