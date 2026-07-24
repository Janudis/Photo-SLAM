#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <limits>
#include <string>

struct VoxelRerunParameters
{
    bool enable_rerun_ = true;
    int rerun_max_keyframes_ = -1;
    int rerun_keyframe_start_ = 0;
    bool run_whole_run_ = false;
    bool rerun_svrecon_debug_ = false;
    bool rerun_gt_mesh_ = false;
    std::string rerun_gt_mesh_path_;
    bool save_rendered_mesh_eval_ = true;
    bool rerun_rendered_mesh_eval_ = false;
    float rendered_mesh_eval_voxel_size_m_ = 0.05f;
    float rendered_mesh_eval_min_weight_ = 2.0f;
    float rendered_mesh_eval_trunc_vox_ = 8.0f;
    float rendered_mesh_eval_depth_max_m_ = 5.0f;
    int svrecon_mesh_init_lv_ = 7;
    int svrecon_mesh_final_lv_ = 10;
    int svrecon_mesh_trunc_lv_ = 10;
    float svrecon_mesh_trunc_vox_ = 5.0f;
    float svrecon_mesh_pg_prune_ = 0.6f;
    float svrecon_mesh_crop_border_ = 0.01f;
    float svrecon_mesh_alpha_thres_ = 0.5f;
    bool svrecon_mesh_use_mean_depth_ = false;
    bool svrecon_mesh_use_vert_color_ = false;
    bool rerun_reconstruction_mesh_ = false;
    int rerun_reconstruction_mesh_interval_ = 200;
    float rerun_reconstruction_mesh_min_weight_ = 1.0e-4f;
    bool rerun_reconstruction_mesh_weld_vertices_ = true;
    std::size_t rerun_reconstruction_mesh_max_vertices_ = 250000;
    std::size_t rerun_reconstruction_mesh_max_faces_ = 500000;
    bool rerun_maps_ = false;
    int rerun_maps_stride_ = 1;
};

struct VoxelRerunState
{
    bool whole_run_live_voxels_dirty_ = true;
    bool svrecon_debug_has_source_snapshot_ = false;

    torch::Tensor whole_run_pruned_centers_accum_;
    torch::Tensor whole_run_pruned_sizes_accum_;
    torch::Tensor whole_run_pruned_levels_accum_;
    torch::Tensor whole_run_pruned_colors_accum_;
    torch::Tensor whole_run_pruned_sdf_centers_accum_;
    torch::Tensor whole_run_pruned_sdf_sizes_accum_;
    torch::Tensor whole_run_pruned_sdf_levels_accum_;
    torch::Tensor whole_run_pruned_sdf_colors_accum_;
    torch::Tensor whole_run_pruned_surface_views_centers_accum_;
    torch::Tensor whole_run_pruned_surface_views_sizes_accum_;
    torch::Tensor whole_run_pruned_surface_views_levels_accum_;
    torch::Tensor whole_run_pruned_surface_views_colors_accum_;
    torch::Tensor whole_run_pruned_final_surface_centers_accum_;
    torch::Tensor whole_run_pruned_final_surface_sizes_accum_;
    torch::Tensor whole_run_pruned_final_surface_levels_accum_;
    torch::Tensor whole_run_pruned_final_surface_colors_accum_;

    torch::Tensor rerun_gt_sdf_grid_keys_cpu_;
    torch::Tensor rerun_gt_sdf_grid_pts_cpu_;
    std::string rerun_gt_sdf_grid_mesh_path_;
};
