#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <cstdint>
#include <array>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

struct SdfEvidenceOctreeKey
{
    int64_t octpath = 0;
    int8_t octlevel = 0;

    bool operator==(const SdfEvidenceOctreeKey& other) const noexcept
    {
        return octpath == other.octpath && octlevel == other.octlevel;
    }
};

struct SdfEvidenceOctreeKeyHash
{
    std::size_t operator()(const SdfEvidenceOctreeKey& key) const noexcept
    {
        std::size_t seed = std::hash<int64_t>{}(key.octpath);
        seed ^= std::hash<int>{}(static_cast<int>(key.octlevel)) +
                0x9e3779b9U + (seed << 6U) + (seed >> 2U);
        return seed;
    }
};

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
    int sdf_evidence_promote_min_views_ = 2;
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

    std::unordered_map<
        SdfEvidenceOctreeKey,
        std::vector<std::size_t>,
        SdfEvidenceOctreeKeyHash> sdf_evidence_observed_keyframes_;
    bool sdf_evidence_layout_valid_ = false;
    std::array<float, 3> sdf_evidence_scene_center_{{0.0f, 0.0f, 0.0f}};
    float sdf_evidence_scene_extent_ = 0.0f;
    int sdf_evidence_base_level_ = 0;
};
