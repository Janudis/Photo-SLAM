#pragma once

#include <torch/torch.h>

#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

class VoxelKeyframe;

struct VoxelMonoPriorParameters
{
    std::string mono_prior_model_id_ = "depth-anything/Depth-Anything-V2-Small-hf";
    std::string mono_prior_loss_mode_ = "svraster";
    std::string mono_prior_normal_mode_ = "aligned";

    int depthanything_densify_stride_ = 8;
    int depthanything_densify_min_sparse_anchors_ = 64;
    std::string depthanything_densify_alignment_mode_ = "orb";
    int depthanything_da_prior_knn_k_ = 5;
    bool depthanything_da_prior_distance_weighting_ = true;
    float depthanything_da_prior_max_pixel_dist_ = 0.0f;
    bool depthanything_fill_holes_ = false;
    bool depthanything_fill_holes_initial_backfill_ = true;
    bool depthanything_fill_holes_warmup_ = false;
    int depthanything_fill_holes_warmup_iter_ = 0;
};

struct VoxelMonoPriorState
{
    bool depthanything_fill_holes_warmup_flushed_ = false;
    std::vector<std::weak_ptr<VoxelKeyframe>> depthanything_fill_holes_pending_kfs_;
    std::unordered_set<std::size_t> depthanything_fill_holes_pending_kfids_;
    torch::Tensor orb_raw_pcd_points_accum_cpu_;
    torch::Tensor orb_raw_pcd_colors_accum_cpu_;
    bool depthanything_global_alignment_valid_ = false;
    float depthanything_global_align_scale_ = std::numeric_limits<float>::quiet_NaN();
    float depthanything_global_align_shift_ = std::numeric_limits<float>::quiet_NaN();
    float depthanything_global_align_weight_ = 0.0f;
    int depthanything_global_align_observations_ = 0;
};
