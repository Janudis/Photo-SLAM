#pragma once

#include <string>
#include <filesystem>

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
        bool convert_SHs = false,
        bool compute_cov3D = false);

public:
    bool convert_SHs_;
    bool compute_cov3D_;
};

class VoxelOptimizationParams 
{
public:
    VoxelOptimizationParams(
        int iterations = 30'000,
        float geo_lr_init = 0.00016f,
        float geo_lr_final = 0.0000016f,
        float geo_lr_delay_mult = 0.01f,
        float geo_lr_max_steps = 30'000,
        // float meta_accum_lr = 0.1f,
        float sh0_lr = 0.00125f,
        float shs_lr = 0.0000625f, // 1/20 of SH₀
        float lambda_dssim = 0.2f,
        int densification_interval = 100,
        int subdiv_from = 500,
        int subdiv_every = 100,
        int subdiv_until = 15000,
        float subdiv_quantile = 0.8f,
        float subdiv_gradient_threshold = 1e-4f,
        float subdivide_samp_thres = 1.0f,
        int   subdivide_max_num    = 10'000'000,
        float subdivide_target_scale_ = 90.f,
        int   subdivide_all_until_    = 0,
        int prune_from = 500,
        int prune_every = 100,
        int prune_until = 18000,
        float prune_threshold_init = 0.0001f,
        float prune_threshold_final = 0.05f,
        // int min_voxels = 512,
        int opacity_reset_interval = 3000,
        int densify_from_iter = 500,
        int densify_until_iter = 15'000,
        float densify_grad_threshold = 0.0002f,

        int geo_warmup_iters = 1000,
        int sh0_warmup_iters = 1000,
        float geo_weight_decay = 1e-4f
    );

public:
    int iterations_;
    float geo_lr_init_;
    float geo_lr_final_;
    float geo_lr_delay_mult_;
    int geo_lr_max_steps_;
    float meta_accum_lr_;
    float geo_lr_;
    float sh0_lr_;
    float shs_lr_;

    int densification_interval_;
    int subdiv_from_;
    int subdiv_every_;
    int subdiv_until_;
    float subdiv_quantile_;
    float subdiv_gradient_threshold_;
    float subdivide_samp_thres_;
    int subdivide_max_num_;
    float subdivide_target_scale_;
    int subdivide_all_until_;
    int prune_from_;
    int prune_every_;
    int prune_until_;
    float prune_threshold_init_;
    float prune_threshold_final_;
    // int min_voxels_;
    float percent_dense_;
    float lambda_dssim_;
    int opacity_reset_interval_;
    int densify_from_iter_;
    int densify_until_iter_;
    float densify_grad_threshold_;

    int geo_warmup_iters_;      // how many iters to train geometry only
    int sh0_warmup_iters_;      // after that, train SH₀ only
    float geo_weight_decay_;    // small L2 on geo to stabilize
};
