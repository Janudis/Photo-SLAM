#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "include_voxel/mini_cam.h"
#include "include_voxel/render_opts.h"

namespace sv {

struct SvreconRasterizationSettings {
    std::string color_mode = "sh";
    std::string vox_geo_mode = "triinterp1";
    std::string density_mode = "sdf";
    int image_width = 0;
    int image_height = 0;
    float tanfovx = 0.0f;
    float tanfovy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    torch::Tensor w2c_matrix;
    torch::Tensor c2w_matrix;
    torch::Tensor bg_color;
    int cam_mode = 0;
    float near = 0.02f;
    bool need_depth = false;
    bool need_normal = false;
    bool track_max_w = false;
    bool track_occlusion_visibility = false;
    float lambda_R_concen = 0.0f;
    float lambda_ascending = 0.0f;
    float lambda_dist = 0.0f;
    torch::Tensor gt_color;
    bool debug = false;
};

torch::autograd::tensor_list gatherSvreconGeoParams(
    torch::Tensor& vox_key,
    torch::Tensor& care_idx,
    torch::Tensor& grid_pts);

torch::autograd::tensor_list evalSvreconSH(
    int active_sh_degree,
    torch::Tensor& idx,
    torch::Tensor& vox_centers,
    torch::Tensor& cam_pos,
    torch::Tensor& sh0,
    torch::Tensor& shs);

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
    const torch::Tensor& gt_image = torch::Tensor(),
    const char* color_mode = nullptr,
    bool track_max_w = false,
    std::optional<float> ss = std::nullopt,
    bool output_depth = false,
    bool output_normal = false,
    bool output_T = false,
    bool rand_bg = false,
    bool use_auto_exposure = false,
    const RenderOpts& other_opt = RenderOpts());

torch::Tensor markSvreconMaxSampRateDirect(
    const std::vector<MiniCam>& cameras,
    const torch::Tensor& octree_paths,
    const torch::Tensor& vox_centers,
    const torch::Tensor& vox_lengths,
    float near = 0.02f);

torch::Tensor markSvreconNearDirect(
    const std::vector<MiniCam>& cameras,
    const torch::Tensor& octree_paths,
    const torch::Tensor& vox_centers,
    const torch::Tensor& vox_lengths,
    float near = 0.2f);

} // namespace sv
