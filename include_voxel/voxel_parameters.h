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
        // Pruning
        int   prune_until_           = 18000,
        float prune_thres_init_      = 1e-4f,
        float prune_thres_final_     = 5e-2f,
        // Subdivision
        int   subdivide_until_       = 15000,
        int   subdivide_all_until_   = 0,
        float subdivide_samp_thres_  = 1.0f,
        float subdivide_prop_        = 0.05f,        // top 5% by priority
        int   subdivide_max_num_     = 10000000,   // hard cap
        float lambda_dssim = 0.2f,
        float lambda_tv_density = 1e-10f,
        int   tv_from = 0,
        int   tv_until = 10000,
        float ss_aug_max = 1.5,
        float lambda_R_concen = 0.01,
        float lambda_dist = 0.1,
        float lambda_T_inside = 0.01
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
    int prune_until_;
    float prune_thres_init_;
    float prune_thres_final_;
    int subdivide_until_;
    int subdivide_all_until_;
    float subdivide_samp_thres_;
    float subdivide_prop_;
    int subdivide_max_num_;
    float lambda_dssim_;
    float lambda_tv_density_;
    int   tv_from_;
    int   tv_until_;
    float ss_aug_max_;
    float lambda_R_concen_;
    float lambda_dist_;
    float lambda_T_inside_;
};
