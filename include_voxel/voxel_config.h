#pragma once
#include <string>

namespace sv {

struct VoxelScheduleConfig {
    // Subdivision
    int subdiv_from = 4;
    int subdiv_every = 10;
    int subdiv_until = 30000;
    float subdiv_quantile = 0.8f;
    float subdiv_gradient_threshold = 1e-4f;

    // Pruning
    int prune_from = 10;
    int prune_every = 5;
    int prune_until = 30000;
    float prune_threshold_init = 1e-4f;
    float prune_threshold_final = 0.05f;
    int min_voxels = 512;

    // Learning
    int max_num_iterations = 30100;
    int densification_interval_ = 100;
    float lr = 1e-2f;
    float meta_accum_lr = 0.1f;

    // Loss weights
    float lambda_photo    = 1.0f;
    float lambda_ssim     = 0.02f;
    float lambda_T_concen = 0.01f;
    float lambda_T_inside = 0.01f;
};
} // namespace sv