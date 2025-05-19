#pragma once
#include <string>

namespace sv {

struct VoxelScheduleConfig {
    // Subdivision
    int subdiv_from = 4;
    int subdiv_every = 5;
    int subdiv_until = 40;
    float subdiv_quantile = 0.8f;
    float subdiv_gradient_threshold = 1e-4f;

    // Pruning
    int prune_from = 10;
    int prune_every = 5;
    int prune_until = 50;
    float prune_threshold_init = 1e-4f;
    float prune_threshold_final = 0.05f;
    int min_voxels = 512;

    int meta_warmup_iters = 10;      // ← optional warm-up before pruning
    float lr = 1e-2f;                // ← used if you want learning rate configurable
    float meta_accum_lr = 0.1f;

    float lambda_photo   = 1.0f;
    float lambda_ssim    = 0.02f;
    float lambda_T_concen = 0.01f;
    float lambda_T_inside = 0.01f;

};

} // namespace sv