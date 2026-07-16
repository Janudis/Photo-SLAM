#include "include_voxel/voxel_svrecon_rasterizer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <ATen/TensorIndexing.h>
#include <torch/nn/functional/upsampling.h>

#define PREPROCESS SVRECON_PREPROCESS
#define FORWARD SVRECON_FORWARD
#define BACKWARD SVRECON_BACKWARD
#define RASTER_STATE SVRECON_RASTER_STATE
#define GEO_PARAMS_GATHER SVRECON_GEO_PARAMS_GATHER
#define SH_COMPUTE SVRECON_SH_COMPUTE
#define TV_COMPUTE SVRECON_TV_COMPUTE
#define VG_COMPUTE SVRECON_VG_COMPUTE
#define GE_COMPUTE SVRECON_GE_COMPUTE
#define LS_COMPUTE SVRECON_LS_COMPUTE
#define PL_COMPUTE SVRECON_PL_COMPUTE
#define UTILS SVRECON_UTILS
#define ADAM_STEP SVRECON_ADAM_STEP

#include "third_party/SVRecon/cuda/src/backward.h"
#include "third_party/SVRecon/cuda/src/config.h"
#include "third_party/SVRecon/cuda/src/forward.h"
#include "third_party/SVRecon/cuda/src/geo_params_gather.h"
#include "third_party/SVRecon/cuda/src/preprocess.h"
#include "third_party/SVRecon/cuda/src/raster_state.h"
#include "third_party/SVRecon/cuda/src/sh_compute.h"

namespace sv {
namespace {

torch::Tensor emptyCudaLike(const torch::Tensor& ref, at::IntArrayRef shape) {
    return torch::empty(shape, ref.options().device(ref.device()));
}

torch::Tensor definedOrZerosLike(const torch::Tensor& grad, const torch::Tensor& ref) {
    if (grad.defined()) {
        return grad.contiguous();
    }
    return torch::zeros_like(ref);
}

torch::Tensor resizeRendering(
    const torch::Tensor& render,
    const std::vector<int64_t>& size,
    const std::string& mode = "bilinear") {
    namespace F = torch::nn::functional;
    torch::Tensor input;
    int squeeze_dims = 0;
    if (render.dim() == 2) {
        input = render.unsqueeze(0).unsqueeze(0);
        squeeze_dims = 2;
    } else if (render.dim() == 3) {
        input = render.unsqueeze(0);
        squeeze_dims = 1;
    } else if (render.dim() == 4) {
        input = render;
    } else {
        return render;
    }
    auto opts = F::InterpolateFuncOptions().size(size);
    if (mode == "nearest") {
        opts = opts.mode(torch::kNearest);
    } else {
        opts = opts.mode(torch::kBilinear).align_corners(false);
    }
    auto out = F::interpolate(input, opts);
    if (squeeze_dims == 2) {
        return out.squeeze(0).squeeze(0);
    }
    if (squeeze_dims == 1) {
        return out.squeeze(0);
    }
    return out;
}

std::string normalizeColorMode(const char* color_mode, int& active_sh_degree) {
    if (color_mode == nullptr) {
        return "sh";
    }
    std::string mode(color_mode);
    if (mode.empty()) {
        return "sh";
    }
    if (mode.rfind("sh", 0) == 0) {
        if (mode.size() > 2) {
            active_sh_degree = std::stoi(mode.substr(2, 1));
        }
        return "sh";
    }
    return mode;
}

class SvreconGatherGeoFunction : public torch::autograd::Function<SvreconGatherGeoFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor vox_key,
        torch::Tensor care_idx,
        torch::Tensor grid_pts) {
        TORCH_CHECK(vox_key.dim() == 2 && vox_key.size(1) == 8,
                    "SvreconGatherGeoFunction: vox_key must be [N,8]");
        TORCH_CHECK(care_idx.dim() == 1,
                    "SvreconGatherGeoFunction: care_idx must be [K]");
        TORCH_CHECK(grid_pts.numel() == grid_pts.size(0),
                    "SvreconGatherGeoFunction: grid_pts must be [M] or [M,1]");
        torch::Tensor grid_flat = grid_pts.contiguous().view({-1, 1});
        torch::Tensor geo = GEO_PARAMS_GATHER::gather_triinterp_geo_params(
            vox_key.contiguous(),
            care_idx.contiguous(),
            grid_flat);
        ctx->saved_data["num_grid_pts"] = grid_flat.size(0);
        ctx->save_for_backward({vox_key.contiguous(), care_idx.contiguous()});
        return geo;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto vox_key = saved[0];
        auto care_idx = saved[1];
        const int num_grid_pts = ctx->saved_data["num_grid_pts"].toInt();
        torch::Tensor dgeo = grad_outputs[0].defined()
            ? grad_outputs[0].contiguous()
            : torch::empty({0}, vox_key.options().dtype(torch::kFloat32));
        torch::Tensor dgrid = GEO_PARAMS_GATHER::gather_triinterp_geo_params_bw(
            vox_key,
            care_idx,
            num_grid_pts,
            dgeo);
        return {torch::Tensor(), torch::Tensor(), dgrid};
    }
};

class SvreconSHFunction : public torch::autograd::Function<SvreconSHFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        int active_sh_degree,
        torch::Tensor idx,
        torch::Tensor vox_centers,
        torch::Tensor cam_pos,
        torch::Tensor sh0,
        torch::Tensor shs) {
        if (!idx.defined()) {
            idx = torch::empty({0}, vox_centers.options().dtype(torch::kInt64));
        }
        torch::Tensor rgb = SH_COMPUTE::sh_compute(
            active_sh_degree,
            idx.contiguous(),
            vox_centers.contiguous(),
            cam_pos.contiguous(),
            sh0.contiguous(),
            shs.contiguous());
        ctx->saved_data["active_sh_degree"] = active_sh_degree;
        ctx->saved_data["M"] = 1 + shs.size(1);
        ctx->save_for_backward({idx.contiguous(), vox_centers.contiguous(),
                                cam_pos.contiguous(), rgb});
        return rgb;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto idx = saved[0];
        auto vox_centers = saved[1];
        auto cam_pos = saved[2];
        auto rgb = saved[3];
        torch::Tensor drgb = definedOrZerosLike(grad_outputs[0], rgb);
        auto grads = SH_COMPUTE::sh_compute_bw(
            ctx->saved_data["active_sh_degree"].toInt(),
            ctx->saved_data["M"].toInt(),
            idx,
            vox_centers,
            cam_pos,
            rgb,
            drgb);
        return {torch::Tensor(), torch::Tensor(), torch::Tensor(),
                torch::Tensor(), std::get<0>(grads), std::get<1>(grads)};
    }
};

class SvreconVoxelsFunction : public torch::autograd::Function<SvreconVoxelsFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext* ctx,
        SvreconRasterizationSettings raster_settings,
        torch::Tensor geomBuffer,
        torch::Tensor octree_paths,
        torch::Tensor vox_centers,
        torch::Tensor vox_lengths,
        torch::Tensor geos,
        torch::Tensor rgbs,
        torch::Tensor vox_feats,
        torch::Tensor subdiv_p,
        torch::Tensor s_val) {
        const bool need_distortion = raster_settings.lambda_dist > 0.0f;
        auto result = FORWARD::rasterize_voxels(
            VOX_TRIINTERP1_MODE,
            SDF_MODE,
            raster_settings.image_width,
            raster_settings.image_height,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.cx,
            raster_settings.cy,
            raster_settings.w2c_matrix.contiguous(),
            raster_settings.c2w_matrix.contiguous(),
            raster_settings.bg_color.contiguous(),
            raster_settings.cam_mode,
            raster_settings.need_depth,
            need_distortion,
            raster_settings.need_normal,
            raster_settings.track_max_w,
            octree_paths.contiguous(),
            vox_centers.contiguous(),
            vox_lengths.contiguous(),
            geos.contiguous(),
            rgbs.contiguous(),
            vox_feats.contiguous(),
            geomBuffer.contiguous(),
            raster_settings.debug,
            s_val.contiguous());

        int num_rendered = std::get<0>(result);
        torch::Tensor binningBuffer = std::get<1>(result);
        torch::Tensor imgBuffer = std::get<2>(result);
        torch::Tensor out_color = std::get<3>(result);
        torch::Tensor out_depth = std::get<4>(result);
        torch::Tensor out_normal = std::get<5>(result);
        torch::Tensor out_T = std::get<6>(result);
        torch::Tensor max_w = std::get<7>(result);
        torch::Tensor out_sdf0 = std::get<8>(result);
        torch::Tensor out_feat = std::get<9>(result);
        auto image_state = RASTER_STATE::unpack_ImageState(
            raster_settings.image_width,
            raster_settings.image_height,
            imgBuffer);
        torch::Tensor n_contrib = std::get<2>(image_state);

        ctx->saved_data["num_rendered"] = num_rendered;
        ctx->saved_data["image_width"] = raster_settings.image_width;
        ctx->saved_data["image_height"] = raster_settings.image_height;
        ctx->saved_data["tanfovx"] = raster_settings.tanfovx;
        ctx->saved_data["tanfovy"] = raster_settings.tanfovy;
        ctx->saved_data["cx"] = raster_settings.cx;
        ctx->saved_data["cy"] = raster_settings.cy;
        ctx->saved_data["lambda_R_concen"] = raster_settings.lambda_R_concen;
        ctx->saved_data["lambda_ascending"] = raster_settings.lambda_ascending;
        ctx->saved_data["lambda_dist"] = raster_settings.lambda_dist;
        ctx->saved_data["need_depth"] = raster_settings.need_depth;
        ctx->saved_data["need_normal"] = raster_settings.need_normal;
        ctx->saved_data["debug"] = raster_settings.debug;
        ctx->save_for_backward({
            raster_settings.w2c_matrix.contiguous(),
            raster_settings.c2w_matrix.contiguous(),
            raster_settings.bg_color.contiguous(),
            raster_settings.gt_color.defined() ? raster_settings.gt_color.contiguous()
                                               : emptyCudaLike(out_color, {0}),
            octree_paths.contiguous(),
            vox_centers.contiguous(),
            vox_lengths.contiguous(),
            geos.contiguous(),
            rgbs.contiguous(),
            geomBuffer.contiguous(),
            binningBuffer.contiguous(),
            imgBuffer.contiguous(),
            out_T.contiguous(),
            out_depth.contiguous(),
            out_normal.contiguous(),
            s_val.contiguous(),
            out_sdf0.contiguous()});

        return {out_color, out_depth, out_normal, out_T, max_w, n_contrib, out_feat};
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto w2c = saved[0];
        auto c2w = saved[1];
        auto bg_color = saved[2];
        auto gt_color = saved[3];
        auto octree_paths = saved[4];
        auto vox_centers = saved[5];
        auto vox_lengths = saved[6];
        auto geos = saved[7];
        auto rgbs = saved[8];
        auto geomBuffer = saved[9];
        auto binningBuffer = saved[10];
        auto imgBuffer = saved[11];
        auto out_T = saved[12];
        auto out_depth = saved[13];
        auto out_normal = saved[14];
        auto s_val = saved[15];
        auto out_sdf0 = saved[16];

        torch::Tensor dcolor = definedOrZerosLike(grad_outputs[0], rgbs.new_zeros({3, 1, 1}));
        if (grad_outputs[0].defined()) {
            dcolor = grad_outputs[0].contiguous();
        }
        torch::Tensor ddepth = definedOrZerosLike(grad_outputs[1], out_depth);
        torch::Tensor dnormal = definedOrZerosLike(grad_outputs[2], out_normal);
        torch::Tensor dT = definedOrZerosLike(grad_outputs[3], out_T);

        auto bw = BACKWARD::rasterize_voxels_backward(
            ctx->saved_data["num_rendered"].toInt(),
            VOX_TRIINTERP1_MODE,
            SDF_MODE,
            ctx->saved_data["image_width"].toInt(),
            ctx->saved_data["image_height"].toInt(),
            static_cast<float>(ctx->saved_data["tanfovx"].toDouble()),
            static_cast<float>(ctx->saved_data["tanfovy"].toDouble()),
            static_cast<float>(ctx->saved_data["cx"].toDouble()),
            static_cast<float>(ctx->saved_data["cy"].toDouble()),
            w2c,
            c2w,
            bg_color,
            octree_paths,
            vox_centers,
            vox_lengths,
            geos,
            rgbs,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            out_T,
            dcolor,
            ddepth,
            dnormal,
            dT,
            static_cast<float>(ctx->saved_data["lambda_R_concen"].toDouble()),
            gt_color,
            static_cast<float>(ctx->saved_data["lambda_ascending"].toDouble()),
            static_cast<float>(ctx->saved_data["lambda_dist"].toDouble()),
            ctx->saved_data["need_depth"].toBool(),
            ctx->saved_data["need_normal"].toBool(),
            out_depth,
            out_normal,
            ctx->saved_data["debug"].toBool(),
            s_val,
            out_sdf0);

        return {
            torch::Tensor(), // raster settings
            torch::Tensor(), // geomBuffer
            torch::Tensor(), // octree_paths
            torch::Tensor(), // vox_centers
            torch::Tensor(), // vox_lengths
            std::get<0>(bw),
            std::get<1>(bw),
            torch::Tensor(), // vox_feats
            std::get<2>(bw),
            std::get<3>(bw)};
    }
};

torch::autograd::tensor_list rasterizeSvreconVoxels(
    SvreconRasterizationSettings& settings,
    torch::Tensor geomBuffer,
    torch::Tensor octree_paths,
    torch::Tensor vox_centers,
    torch::Tensor vox_lengths,
    torch::Tensor geos,
    torch::Tensor rgbs,
    torch::Tensor vox_feats,
    torch::Tensor subdiv_p,
    torch::Tensor s_val) {
    return SvreconVoxelsFunction::apply(
        settings,
        geomBuffer,
        octree_paths,
        vox_centers,
        vox_lengths,
        geos,
        rgbs,
        vox_feats,
        subdiv_p,
        s_val);
}

} // namespace

torch::autograd::tensor_list gatherSvreconGeoParams(
    torch::Tensor& vox_key,
    torch::Tensor& care_idx,
    torch::Tensor& grid_pts) {
    torch::Tensor out = SvreconGatherGeoFunction::apply(vox_key, care_idx, grid_pts);
    return {out};
}

torch::autograd::tensor_list evalSvreconSH(
    int active_sh_degree,
    torch::Tensor& idx,
    torch::Tensor& vox_centers,
    torch::Tensor& cam_pos,
    torch::Tensor& sh0,
    torch::Tensor& shs) {
    torch::Tensor out = SvreconSHFunction::apply(
        active_sh_degree, idx, vox_centers, cam_pos, sh0, shs);
    return {out};
}

std::unordered_map<std::string, torch::Tensor> renderSvreconDirect(
    const MiniCam& cam,
    int im_height,
    int im_width,
    torch::Tensor sdf_grid_pts,
    torch::Tensor sh0,
    torch::Tensor shs,
    torch::Tensor subdiv_p,
    torch::Tensor log_s,
    torch::Tensor oct_path,
    torch::Tensor is_leaf,
    torch::Tensor center,
    torch::Tensor vox_size,
    torch::Tensor vox_key,
    torch::Tensor frozen_vox_geo,
    int active_sh_degree,
    bool white_background,
    bool black_background,
    float default_ss,
    const torch::Tensor& gt_image,
    const char* color_mode,
    bool track_max_w,
    std::optional<float> ss,
    bool output_depth,
    bool output_normal,
    bool output_T,
    bool rand_bg,
    bool use_auto_exposure,
    const RenderOpts& other_opt) {
    (void)gt_image;
    (void)use_auto_exposure;

    std::unordered_map<std::string, torch::Tensor> render_pkg;
    if (!center.defined() || center.numel() == 0) {
        return render_pkg;
    }

    torch::Device device = center.device();
    center = center.to(device).contiguous();
    vox_size = vox_size.to(device).contiguous();
    if (vox_size.dim() == 1) {
        vox_size = vox_size.view({-1, 1});
    }
    oct_path = oct_path.to(device).contiguous().view({-1});
    if (!is_leaf.defined() || is_leaf.numel() != center.size(0)) {
        is_leaf = torch::ones(
            {center.size(0)},
            torch::TensorOptions().dtype(torch::kUInt8).device(device));
    } else {
        is_leaf = is_leaf.to(device).to(torch::kUInt8).contiguous().view({-1});
    }
    vox_key = vox_key.to(device).contiguous();
    sdf_grid_pts = sdf_grid_pts.to(device).contiguous().view({-1, 1});
    sh0 = sh0.to(device).contiguous();
    shs = shs.to(device).contiguous();
    subdiv_p = subdiv_p.to(device).contiguous().view({-1, 1});
    log_s = log_s.to(device).contiguous().view({-1});
    if (log_s.numel() == 0) {
        log_s = torch::full({1}, 0.3f, center.options());
    }

    const float render_ss =
        ss.has_value() ? *ss : (other_opt.ss.has_value() ? *other_opt.ss : default_ss);
    const int w_src = im_width;
    const int h_src = im_height;
    const int w = std::max(1, static_cast<int>(std::lround(static_cast<float>(w_src) * render_ss)));
    const int h = std::max(1, static_cast<int>(std::lround(static_cast<float>(h_src) * render_ss)));
    const float w_ss = static_cast<float>(w) / std::max(1, w_src);
    const float h_ss = static_cast<float>(h) / std::max(1, h_src);

    int active_degree = active_sh_degree;
    const std::string mode = normalizeColorMode(color_mode, active_degree);

    SvreconRasterizationSettings settings;
    settings.color_mode = mode;
    settings.image_width = w;
    settings.image_height = h;
    settings.tanfovx = cam.tanfovx;
    settings.tanfovy = cam.tanfovy;
    settings.cx = cam.cx * w_ss;
    settings.cy = cam.cy * h_ss;
    settings.w2c_matrix = cam.w2c.to(device).contiguous();
    settings.c2w_matrix = cam.c2w.to(device).contiguous();
    const float bg = white_background ? 1.0f : 0.0f;
    settings.bg_color = torch::full({3}, bg, center.options());
    settings.cam_mode = CAM_PERSP;
    settings.near = 0.02f;
    settings.need_depth = output_depth || other_opt.output_depth;
    settings.need_normal = output_normal || other_opt.output_normal;
    settings.track_max_w = track_max_w || other_opt.track_max_w;
    settings.lambda_R_concen = other_opt.lambda_R_concen.value_or(0.0f);
    settings.lambda_ascending = other_opt.lambda_ascending.value_or(0.0f);
    settings.lambda_dist = other_opt.lambda_dist.value_or(0.0f);
    settings.gt_color = other_opt.gt_color.defined()
        ? other_opt.gt_color.to(device).contiguous()
        : torch::empty({0}, center.options());

    auto prep = PREPROCESS::rasterize_preprocess(
        settings.image_width,
        settings.image_height,
        settings.tanfovx,
        settings.tanfovy,
        settings.cx,
        settings.cy,
        settings.w2c_matrix,
        settings.c2w_matrix,
        settings.cam_mode,
        settings.near,
        oct_path,
        center,
        vox_size,
        is_leaf,
        settings.debug);
    torch::Tensor n_duplicates = std::get<0>(prep);
    torch::Tensor geomBuffer = std::get<1>(prep);
    torch::Tensor in_frusts_idx = torch::nonzero(n_duplicates > 0).view({-1}).to(torch::kLong);

    torch::Tensor geos;
    if (frozen_vox_geo.defined() && frozen_vox_geo.numel() > 0) {
        geos = frozen_vox_geo.to(device).contiguous();
    } else {
        geos = SvreconGatherGeoFunction::apply(vox_key, in_frusts_idx, sdf_grid_pts);
    }

    torch::Tensor rgbs;
    if (mode == "sh") {
        torch::Tensor cam_pos = settings.c2w_matrix.index({torch::indexing::Slice(0, 3), 3}).contiguous();
        rgbs = SvreconSHFunction::apply(active_degree, in_frusts_idx, center, cam_pos, sh0, shs);
    } else if (mode == "rand") {
        rgbs = torch::rand({center.size(0), 3}, center.options());
    } else if (mode == "dontcare") {
        rgbs = torch::empty({center.size(0), 3}, center.options());
    } else {
        throw std::runtime_error("Unsupported SVRecon color mode: " + mode);
    }

    torch::Tensor vox_feats = other_opt.vox_feats.defined() && other_opt.vox_feats.numel() > 0
        ? other_opt.vox_feats.to(device).contiguous()
        : torch::empty({center.size(0), 0}, center.options());
    torch::Tensor s_val = torch::exp(log_s * 10.0f).contiguous();

    auto result = rasterizeSvreconVoxels(
        settings,
        geomBuffer,
        oct_path,
        center,
        vox_size,
        geos,
        rgbs,
        vox_feats,
        subdiv_p,
        s_val);

    torch::Tensor color = result[0];
    torch::Tensor depth = result[1];
    torch::Tensor normal = result[2];
    torch::Tensor T = result[3];
    torch::Tensor max_w = result[4];
    torch::Tensor n_contrib = result[5];
    torch::Tensor feat = result[6];
    const bool dontcare_color = (mode == "dontcare");

    if (!dontcare_color && rand_bg) {
        color = color + T * torch::rand_like(color);
    } else if (!dontcare_color && !white_background && !black_background) {
        color = color + T * color.mean({1, 2}, true);
    }

    auto insert_tensor = [&](const std::string& key, torch::Tensor value, bool keep) {
        if (!keep || !value.defined()) {
            return;
        }
        render_pkg[key] = value;
        render_pkg["raw_" + key] = value;
    };

    insert_tensor("color", color, !dontcare_color);
    insert_tensor("depth", depth, settings.need_depth);
    insert_tensor("normal", normal, settings.need_normal);
    insert_tensor("T", T, output_T || other_opt.output_T);
    render_pkg["max_w"] = max_w;
    render_pkg["n_contrib"] = n_contrib;
    render_pkg["raw_n_contrib"] = n_contrib;
    if (feat.defined() && feat.numel() > 0) {
        render_pkg["feat"] = feat;
        render_pkg["raw_feat"] = feat;
    }

    const std::vector<int64_t> target_size{h_src, w_src};
    for (const auto& key : {"color", "depth", "normal", "T", "n_contrib", "feat"}) {
        auto it = render_pkg.find(key);
        if (it == render_pkg.end() || !it->second.defined() || it->second.dim() < 2) {
            continue;
        }
        if (it->second.size(-2) == h_src && it->second.size(-1) == w_src) {
            continue;
        }
        if (std::string(key) == "n_contrib") {
            it->second = resizeRendering(it->second.to(torch::kFloat32), target_size, "nearest")
                             .round()
                             .to(torch::kInt32);
        } else if (std::string(key) == "feat") {
            it->second = resizeRendering(it->second, target_size, "nearest");
        } else {
            it->second = resizeRendering(it->second, target_size);
        }
    }

    auto color_it = render_pkg.find("color");
    if (color_it != render_pkg.end() && color_it->second.defined()) {
        color_it->second = color_it->second.clamp(0.0, 1.0);
    }
    return render_pkg;
}

torch::Tensor markSvreconMaxSampRateDirect(
    const std::vector<MiniCam>& cameras,
    const torch::Tensor& octree_paths,
    const torch::Tensor& vox_centers,
    const torch::Tensor& vox_lengths,
    float near) {
    if (!octree_paths.defined() || octree_paths.numel() == 0) {
        return torch::empty({0}, vox_centers.options().dtype(torch::kFloat32));
    }
    torch::Tensor centers = vox_centers.contiguous();
    torch::Tensor lengths = vox_lengths.dim() == 2
        ? vox_lengths.squeeze(1).contiguous()
        : vox_lengths.contiguous().view({-1});
    const int64_t N = centers.size(0);
    torch::Device device = centers.device();
    torch::Tensor max_samp_rate =
        torch::zeros({N}, centers.options().dtype(torch::kFloat32));
    torch::Tensor is_leaf =
        torch::ones({N}, torch::TensorOptions().dtype(torch::kUInt8).device(device));

    for (const auto& cam : cameras) {
        auto prep = PREPROCESS::rasterize_preprocess(
            cam.width,
            cam.height,
            cam.tanfovx,
            cam.tanfovy,
            cam.cx,
            cam.cy,
            cam.w2c.to(device).contiguous(),
            cam.c2w.to(device).contiguous(),
            CAM_PERSP,
            near,
            octree_paths.contiguous(),
            centers,
            vox_lengths.contiguous(),
            is_leaf,
            /*debug=*/false);
        torch::Tensor n_duplicates = std::get<0>(prep);
        torch::Tensor pos = cam.position.to(device).to(torch::kFloat32).view({1, 3});
        torch::Tensor lookat = cam.lookat.to(device).to(torch::kFloat32).view({1, 3});
        torch::Tensor zdist = ((centers - pos) * lookat).sum(-1);
        torch::Tensor vis_idx =
            torch::nonzero((n_duplicates > 0) & (zdist > near)).view({-1}).to(torch::kLong);
        if (vis_idx.numel() == 0) {
            continue;
        }
        torch::Tensor zdist_vis = zdist.index_select(0, vis_idx);
        torch::Tensor samp_interval = zdist_vis * cam.pix_size;
        torch::Tensor samp_rate = lengths.index_select(0, vis_idx) / samp_interval;
        torch::Tensor old_rate = max_samp_rate.index_select(0, vis_idx);
        max_samp_rate.index_put_({vis_idx}, torch::maximum(old_rate, samp_rate));
    }
    return max_samp_rate.contiguous();
}

torch::Tensor markSvreconNearDirect(
    const std::vector<MiniCam>& cameras,
    const torch::Tensor& octree_paths,
    const torch::Tensor& vox_centers,
    const torch::Tensor& vox_lengths,
    float near) {
    if (!octree_paths.defined() || octree_paths.numel() == 0) {
        return torch::empty({0}, vox_centers.options().dtype(torch::kBool));
    }
    torch::Tensor centers = vox_centers.contiguous();
    torch::Tensor lengths = vox_lengths.dim() == 2
        ? vox_lengths.squeeze(1).contiguous()
        : vox_lengths.contiguous().view({-1});
    const int64_t N = centers.size(0);
    torch::Device device = centers.device();
    torch::Tensor is_near =
        torch::zeros({N}, centers.options().dtype(torch::kBool));
    torch::Tensor is_leaf =
        torch::ones({N}, torch::TensorOptions().dtype(torch::kUInt8).device(device));

    for (const auto& cam : cameras) {
        auto prep = PREPROCESS::rasterize_preprocess(
            cam.width,
            cam.height,
            cam.tanfovx,
            cam.tanfovy,
            cam.cx,
            cam.cy,
            cam.w2c.to(device).contiguous(),
            cam.c2w.to(device).contiguous(),
            CAM_PERSP,
            near,
            octree_paths.contiguous(),
            centers,
            vox_lengths.contiguous(),
            is_leaf,
            /*debug=*/false);
        torch::Tensor n_duplicates = std::get<0>(prep);
        torch::Tensor vis_idx =
            torch::nonzero(n_duplicates > 0).view({-1}).to(torch::kLong);
        if (vis_idx.numel() == 0) {
            continue;
        }
        torch::Tensor pos = cam.position.to(device).to(torch::kFloat32).view({1, 3});
        torch::Tensor lookat = cam.lookat.to(device).to(torch::kFloat32).view({1, 3});
        torch::Tensor center_vis = centers.index_select(0, vis_idx);
        torch::Tensor zdist = ((center_vis - pos) * lookat).sum(-1);
        torch::Tensor cond = zdist <= (near + lengths.index_select(0, vis_idx));
        torch::Tensor old = is_near.index_select(0, vis_idx);
        is_near.index_put_({vis_idx}, old | cond.to(torch::kBool));
    }
    return is_near.contiguous();
}

} // namespace sv
