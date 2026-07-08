#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <cstdint>
#include <limits>
#include <string>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

struct VoxelSdfParameters
{
    bool use_tsdf_mapping_ = false;
    bool use_tsdf_pruning_ = false;
    bool use_tsdf_planning_ = false;
    std::string tsdf_backend_ = "svraster";
    float sdf_voxel_size_m_ = 0.05f;
    float tsdf_prune_min_weight_ = 1.0f;
    float tsdf_prune_surface_band_vox_ = 2.0f;
    int tsdf_prune_min_valid_corners_ = 8;
    bool tsdf_protect_surface_band_from_pruning_ = true;
    bool tsdf_density_init_ = false;
    float tsdf_density_init_min_weight_ = 1.0f;
    float tsdf_density_init_trunc_vox_ = 4.0f;
    float svraster_tsdf_max_integration_distance_m_ = 4.0f;
    bool svraster_tsdf_inverse_square_weighting_ = true;
    float svraster_tsdf_max_weight_ = 5.0f;
    bool svraster_tsdf_refit_on_topology_change_ = true;
    bool sdf_evidence_densify_ = false;
    bool sdf_evidence_zero_crossing_refresh_ = false;
    bool tsdf_subdivide_near_zero_crossing_ = false;
    float tsdf_subdivide_min_weight_ = 0.5f;
    float tsdf_subdivide_surface_band_vox_ = 2.0f;
    int tsdf_subdivide_min_valid_corners_ = 2;
    float tsdf_density_init_bell_a_ = 0.1f;
    float tsdf_density_init_bell_b_ = 0.5f;
    float tsdf_density_init_alpha_min_ = 0.0002f;
    float tsdf_density_init_alpha_max_ = 0.5f;
};

struct VoxelSdfState
{
    bool svraster_tsdf_init_context_valid_ = false;
    cv::Mat svraster_tsdf_init_depth_meters_;
    Sophus::SE3f svraster_tsdf_init_Tcw_;
    float svraster_tsdf_init_fx_ = 0.0f;
    float svraster_tsdf_init_fy_ = 0.0f;
    float svraster_tsdf_init_cx_ = 0.0f;
    float svraster_tsdf_init_cy_ = 0.0f;
    int svraster_tsdf_init_width_ = 0;
    int svraster_tsdf_init_height_ = 0;
    std::size_t svraster_tsdf_init_kfid_ = 0;

    bool svraster_tsdf_last_context_valid_ = false;
    cv::Mat svraster_tsdf_last_depth_meters_;
    Sophus::SE3f svraster_tsdf_last_Tcw_;
    float svraster_tsdf_last_fx_ = 0.0f;
    float svraster_tsdf_last_fy_ = 0.0f;
    float svraster_tsdf_last_cx_ = 0.0f;
    float svraster_tsdf_last_cy_ = 0.0f;
    int svraster_tsdf_last_width_ = 0;
    int svraster_tsdf_last_height_ = 0;
    std::size_t svraster_tsdf_last_kfid_ = 0;

    int64_t tsdf_ablation_rgbd_points_created_ = 0;
    int64_t tsdf_ablation_inactive_geo_created_ = 0;
    int64_t tsdf_ablation_rgbd_fill_created_ = 0;
    int64_t tsdf_ablation_rgbd_points_lineage_created_ = 0;
    int64_t tsdf_ablation_inactive_geo_lineage_created_ = 0;
    int64_t tsdf_ablation_rgbd_fill_lineage_created_ = 0;
    int64_t tsdf_ablation_rgbd_points_live_last_ = 0;
    int64_t tsdf_ablation_inactive_geo_live_last_ = 0;
    int64_t tsdf_ablation_rgbd_fill_live_last_ = 0;
    int64_t tsdf_ablation_rgbd_points_pruned_by_sdf_ = 0;
    int64_t tsdf_ablation_inactive_geo_pruned_by_sdf_ = 0;
    int64_t tsdf_ablation_rgbd_fill_pruned_by_sdf_ = 0;
    int64_t tsdf_ablation_sdf_prune_passes_ = 0;
};
