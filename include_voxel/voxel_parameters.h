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
        // Adam hyperparameters used by the SVRecon optimizer.
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
        bool  filter_near_voxels_ = true,
        bool  prune_far_voxels_ = true,
        bool  prune_near_voxels_geometric_ = false,
        // Pruning
        int   prune_from_            = 1000,
        int   prune_until_           = 18000,
        float prune_thres_init_      = 1e-4f,
        float prune_thres_final_     = 5e-2f,
        float prune_thres_final_at_target_ = 5e-2f,
        // Subdivision
        int   subdivide_from_        = 250,
        int   subdivide_all_until_   = 0,
        float subdivide_samp_thres_  = 1.0f,
        float subdivide_prop_        = 0.05f,
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
        bool  enable_da2_uncertainty = true,
        int   level_uncertainty_from = 0,
        float power_level_uncertainty = 1.0f,
        float lambda_ascending = 0.0f,
        int   ascending_from = 0,
        float lambda_rectify = 1e-6f,
        int   rectifiy_from = 0,
        float lambda_scaling_penalty = 1e-6f,
        int   scaling_penalty_from = 0,
        int   scaling_penalty_end = 20000,
        int   multi_view_weight_from_iter = 1000000000,
        int   multi_view_interval = 1,
        float multi_view_anneal_scale = 0.0f,
        float multi_view_ncc_weight = 0.05f,
        float multi_view_geo_weight = 0.01f,
        int   multi_view_patch_size = 3,
        int   multi_view_sample_num = 10240000,
        float multi_view_pixel_noise_th = 1.0f,
        float voxel_dropout_min = 0.5f
    );

public:
    int iterations_;
    float geo_lr_;
    float sh0_lr_;
    float shs_lr_;
    float log_s_lr_ = 0.0f;
    float optim_beta1_;
    float optim_beta2_;
    float optim_eps_;
    std::vector<int> lr_decay_ckpt_;
    float lr_decay_mult_;
    int adapt_from_;
    int adapt_every_;
    int prune_every_;
    int subdivide_every_;
    bool filter_near_voxels_;
    bool prune_far_voxels_;
    bool prune_near_voxels_geometric_;
    bool prune_surface_views_enable_ = false;
    int surface_min_views_ = 4;
    int surface_view_window_size_ = 10;
    bool prune_mvs_consistency_enable_ = false;
    int prune_mvs_min_supporting_views_ = 2;
    int prune_mvs_min_contradicting_views_ = 2;
    float prune_mvs_depth_tolerance_vox_ = 1.5f;
    bool final_refinement_enable_ = false;
    int prune_from_;
    int prune_until_;
    float prune_thres_init_;
    float prune_thres_final_;
    float prune_thres_final_at_target_;
    int subdivide_from_;
    int subdivide_all_until_;
    float subdivide_samp_thres_;
    float subdivide_prop_;
    int subdivide_max_num_ = 1500000;
    bool use_l1_;
    bool use_huber_;
    float huber_thres_;
    float lambda_tv_density_;
    int   tv_from_;
    int   tv_until_;
    float lambda_ge_density_ = 0.0f;
    int   ge_from_ = 0;
    int   ge_until_ = 0;
    float lambda_ls_density_ = 0.0f;
    int   ls_from_ = 0;
    int   ls_until_ = 0;
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
    float lambda_rgbd_depth_ = 0.0f;
    int   rgbd_depth_from_ = 3000;
    int   rgbd_depth_end_ = 20000;
    float rgbd_depth_end_mult_ = 0.1f;
    float lambda_monocular_depth_ = 0.0f;
    int   monocular_depth_from_ = 1000;
    int   monocular_depth_end_ = 15000;
    float monocular_depth_end_mult_ = 0.1f;
    float monocular_depth_alpha_min_ = 0.5f;
    float monocular_depth_confidence_min_ = 0.0f;
    float lambda_rgbd_sdf_ = 0.0f;
    int   rgbd_sdf_from_ = 0;
    int   rgbd_sdf_end_ = 15000;
    float rgbd_sdf_end_mult_ = 0.1f;
    float rgbd_sdf_trunc_vox_ = 4.0f;
    int   rgbd_sdf_max_samples_ = 40000;
    int   rgbd_sdf_ray_pixels_ = 1024;
    int   rgbd_sdf_free_samples_ = 4;
    int   rgbd_sdf_surface_samples_ = 8;
    float rgbd_sdf_w_fs_ = 5.0f;
    float rgbd_sdf_w_center_ = 20.0f;
    float rgbd_sdf_w_tail_ = 1.0f;
    bool  enable_da2_uncertainty_;
    int   level_uncertainty_from_;
    float power_level_uncertainty_;
    float lambda_ascending_;
    int   ascending_from_;
    float lambda_rectify_;
    int   rectifiy_from_;
    float lambda_scaling_penalty_;
    int   scaling_penalty_from_;
    int   scaling_penalty_end_;
    int   multi_view_weight_from_iter_;
    int   multi_view_interval_;
    float multi_view_anneal_scale_;
    float multi_view_ncc_weight_;
    float multi_view_geo_weight_;
    int   multi_view_patch_size_;
    int   multi_view_sample_num_;
    float multi_view_pixel_noise_th_;
    float voxel_dropout_min_;
    float lambda_rgbd_normal_ = 0.0f;
    int   rgbd_normal_from_ = 3000;
    int   rgbd_normal_end_ = 20000;
    float rgbd_normal_end_mult_ = 0.1f;
    int   rgbd_normal_ks_ = 3;
    float rgbd_normal_tol_deg_ = 90.0f;
    float lambda_monocular_normal_ = 0.0f;
    int   monocular_normal_from_ = 1000;
    int   monocular_normal_end_ = 15000;
    float monocular_normal_end_mult_ = 0.1f;
    int   monocular_normal_ks_ = 3;
    float monocular_normal_tol_deg_ = 90.0f;
    float monocular_normal_max_depth_jump_rel_ = 0.05f;
};
