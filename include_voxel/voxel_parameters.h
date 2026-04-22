#pragma once

#include <string>
#include <filesystem>
#include <vector>   

class VoxelModelParams 
{
public:
    VoxelModelParams(
        std::filesystem::path source_path = "",
        std::filesystem::path model_path = "",
        std::filesystem::path exec_path = "",
        int sh_degree = 3,
        std::string images = "images",
        float resolution = -1.0f,
        bool white_background = false,
        std::string data_device = "cuda",
        bool eval = false);

public:
    int sh_degree_;
    std::filesystem::path source_path_;
    std::filesystem::path model_path_;
    std::string images_;
    float resolution_;
    bool white_background_;
    std::string data_device_;
    bool eval_;
};

class VoxelPipelineParams 
{
public:
    VoxelPipelineParams(
        bool convert_SHs = false);

public:
    bool convert_SHs_;
};

class VoxelOptimizationParams 
{
public:
    VoxelOptimizationParams(
        int iterations = 20000,
        float geo_lr      = 0.025f,    // cfg.optimizer.geo_lr
        float sh0_lr      = 0.010f,    // cfg.optimizer.sh0_lr
        float shs_lr      = 0.00025f,  // cfg.optimizer.shs_lr
        // Adam hyper-params (match SVRaster)
        float optim_beta1 = 0.1f,
        float optim_beta2 = 0.999f,
        float optim_eps   = 1e-15f,
        // MultiStep decay (same pattern as cfg.optimizer.lr_decay_ckpt & lr_decay_mult)
        std::vector<int> lr_decay_ckpt = {19000}, // milestones in iterations
        float lr_decay_mult = 0.1f,                            // gamma
        // --- Adaptive procedure (SV style) ---
        int   adapt_from_            = 800,
        int   adapt_every_           = 1000,
        int   prune_every_           = 0,         // 0: fallback to adapt_every_
        int   subdivide_every_       = 0,         // 0: fallback to adapt_every_
        int   densify_cooldown_iters_ = 0,      // skip prune/subdivide for very recent voxels
        bool  filter_near_voxels_ = true,
        bool  filter_far_voxels_on_insert_ = true,
        bool  prune_far_voxels_ = true,
        bool  prune_near_voxels_geometric_ = false,
        bool  prune_recompute_dense_core_ = true, // 1: recompute at prune time, 0: reuse latest cached bbox
        bool  prune_recent_unstable_ = false,
        int   prune_recent_keyframes_ = 3,      // GS-SLAM style: "inserted within last K keyframes"
        int   prune_recent_min_views_real_ = 0,
        int   prune_recent_min_views_artificial_ = 0,
        int   rendered_depth_candidate_promote_min_support_ = 3,
        int   rendered_depth_candidate_prune_kf_age_ = 3,
        int   prune_min_kf_age_ = 0,            // prune only if current_kf - exist_since_kf >= this
        bool  prune_surface_keep_enable_ = true,
        bool  prune_surface_keep_use_view_ = true,
        bool  prune_surface_keep_use_size_ = true,
        bool  final_special_prune_enable_ = true,
        // Pruning
        int   prune_until_           = 18000,
        float prune_thres_init_      = 1e-4f,
        float prune_thres_final_     = 5e-2f,
        float prune_thres_final_at_target_ = 5e-2f,
        float prune_thres_init_artificial_ = 3e-4f,
        float prune_thres_final_artificial_ = 1e-1f,
        // Subdivision
        int   subdivide_until_       = 15000,
        int   subdivide_all_until_   = 0,
        float subdivide_samp_thres_  = 1.0f,
        float subdivide_prop_        = 0.05f,        // top 5% by priority
        float subdivide_samp_thres_at_target_ = 1.0f,
        float subdivide_prop_at_target_ = 0.05f,
        int   subdivide_max_num_     = 10000000,   // hard cap
        bool  subdivide_force_to_target_size_ = false,
        float subdivide_target_vox_size_ = 0.05f,
        float lambda_dssim = 0.2f,
        bool  use_l1 = false,
        bool  use_huber = false,
        float huber_thres = 0.03f,
        float lambda_tv_density = 1e-10f,
        int   tv_from = 0,
        int   tv_until = 10000,
        float ss_aug_max = 1.5,
        float lambda_R_concen = 0.01,
        float lambda_dist = 0.1,
        int   dist_from = 10000,
        float lambda_T_concen = 0.01,
        float lambda_T_inside = 0.01,
        float lambda_normal_dmean = 0.0f,
        int   n_dmean_from = 10000,
        int   n_dmean_end = 20000,
        int   n_dmean_ks = 3,
        float n_dmean_tol_deg = 90.0f,
        float lambda_ssim = 0.02f,
        float lambda_sparse_depth = 0.0f,
        int   sparse_depth_until = 1000,
        float lambda_depthanythingv2 = 0.0f,
        int   depthanythingv2_from = 3000,
        int   depthanythingv2_end = 20000,
        float depthanythingv2_end_mult = 0.1f,
        float lambda_depthanythingv2_normal = 0.0f,
        int   depthanythingv2_normal_from = 3000,
        int   depthanythingv2_normal_end = 20000,
        float depthanythingv2_normal_end_mult = 0.1f
    );

public:
    int iterations_;
    float geo_lr_;
    float sh0_lr_;
    float shs_lr_;
    float optim_beta1_;
    float optim_beta2_;
    float optim_eps_;
    std::vector<int> lr_decay_ckpt_;
    float lr_decay_mult_;
    int adapt_from_;
    int adapt_every_;
    int prune_every_;
    int subdivide_every_;
    int densify_cooldown_iters_;
    bool filter_near_voxels_;
    bool filter_far_voxels_on_insert_;
    bool prune_far_voxels_;
    bool prune_near_voxels_geometric_;
    bool prune_recompute_dense_core_;
    bool prune_recent_unstable_;
    int prune_recent_keyframes_;
    int prune_recent_min_views_real_;
    int prune_recent_min_views_artificial_;
    int rendered_depth_candidate_promote_min_support_;
    int rendered_depth_candidate_prune_kf_age_;
    int prune_min_kf_age_;
    bool prune_surface_keep_enable_;
    bool prune_surface_keep_use_view_;
    bool prune_surface_keep_use_size_;
    bool final_special_prune_enable_;
    int prune_until_;
    float prune_thres_init_;
    float prune_thres_final_;
    float prune_thres_final_at_target_;
    float prune_thres_init_artificial_;
    float prune_thres_final_artificial_;
    int subdivide_until_;
    int subdivide_all_until_;
    float subdivide_samp_thres_;
    float subdivide_prop_;
    float subdivide_samp_thres_at_target_;
    float subdivide_prop_at_target_;
    int subdivide_max_num_;
    bool subdivide_artificial_requires_promotion_ = false;
    bool subdivide_force_to_target_size_;
    float subdivide_target_vox_size_;
    float lambda_dssim_;
    bool use_l1_;
    bool use_huber_;
    float huber_thres_;
    float lambda_tv_density_;
    int   tv_from_;
    int   tv_until_;
    float ss_aug_max_;
    float lambda_R_concen_;
    float lambda_dist_;
    int   dist_from_;
    float lambda_T_concen_;
    float lambda_T_inside_;
    float lambda_normal_dmean_;
    int   n_dmean_from_;
    int   n_dmean_end_;
    int   n_dmean_ks_;
    float n_dmean_tol_deg_;
    float lambda_ssim_;
    float lambda_sparse_depth_;
    int   sparse_depth_until_;
    float lambda_depthanythingv2_;
    int   depthanythingv2_from_;
    int   depthanythingv2_end_;
    float depthanythingv2_end_mult_;
    float lambda_depthanythingv2_normal_;
    int   depthanythingv2_normal_from_;
    int   depthanythingv2_normal_end_;
    float depthanythingv2_normal_end_mult_;
};
