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
    bool run_tsdf_pruned_ = false;
    bool rerun_tsdf_unknown_voxels_ = false;
    bool rerun_unstable_ = false;
    bool run_floaters_ = false;
    bool run_whole_run_ = false;
    bool run_sdf_pruned_nvblox_ = false;
    int run_floaters_stride_ = 1;

    bool rerun_tsdf_pruned_log_gt_mesh_ = false;
    bool rerun_tsdf_pruned_align_gt_to_slam_ = false;
    std::string rerun_tsdf_pruned_gt_mesh_path_;
    std::string rerun_tsdf_pruned_gt_traj_path_;
    int rerun_tsdf_pruned_align_min_pairs_ = 10;

    bool save_nvblox_mesh_eval_ = false;
    bool load_saved_nvblox_mesh_ = false;
    bool rerun_nvblox_mesh_ = false;
    std::string saved_nvblox_mesh_path_;
    bool rerun_gt_mesh_ = false;
    std::string rerun_gt_mesh_path_;
    bool save_rendered_mesh_eval_ = true;
    bool rerun_rendered_mesh_eval_ = false;
    float rendered_mesh_eval_voxel_size_m_ = 0.05f;
    float rendered_mesh_eval_min_weight_ = 2.0f;
    float rendered_mesh_eval_trunc_vox_ = 8.0f;
    float rendered_mesh_eval_depth_max_m_ = 5.0f;
    float rendered_mesh_eval_alpha_thres_ = 0.5f;
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
    bool rerun_tsdf_unknown_dirty_ = false;
    bool run_floaters_dirty_ = false;
    bool whole_run_live_voxels_dirty_ = true;

    torch::Tensor whole_run_pruned_centers_accum_;
    torch::Tensor whole_run_pruned_sizes_accum_;
    torch::Tensor whole_run_pruned_colors_accum_;
    torch::Tensor whole_run_pruned_tsdf_centers_accum_;
    torch::Tensor whole_run_pruned_tsdf_sizes_accum_;
    torch::Tensor whole_run_pruned_tsdf_colors_accum_;
    torch::Tensor whole_run_pruned_svraster_centers_accum_;
    torch::Tensor whole_run_pruned_svraster_sizes_accum_;
    torch::Tensor whole_run_pruned_svraster_colors_accum_;
    torch::Tensor whole_run_pruned_near_centers_accum_;
    torch::Tensor whole_run_pruned_near_sizes_accum_;
    torch::Tensor whole_run_pruned_near_colors_accum_;
    torch::Tensor whole_run_pruned_recent_unstable_centers_accum_;
    torch::Tensor whole_run_pruned_recent_unstable_sizes_accum_;
    torch::Tensor whole_run_pruned_recent_unstable_colors_accum_;
    torch::Tensor whole_run_pruned_final_special_centers_accum_;
    torch::Tensor whole_run_pruned_final_special_sizes_accum_;
    torch::Tensor whole_run_pruned_final_special_colors_accum_;
    torch::Tensor whole_run_live_local_view_counts_cache_;
    torch::Tensor unstable_pruned_centers_accum_;
    torch::Tensor unstable_pruned_sizes_accum_;
    torch::Tensor unstable_pruned_colors_accum_;

    torch::Tensor rerun_gt_sdf_grid_keys_cpu_;
    torch::Tensor rerun_gt_sdf_grid_pts_cpu_;
    std::string rerun_gt_sdf_grid_mesh_path_;
    std::size_t rerun_gt_sdf_grid_kfid_ = std::numeric_limits<std::size_t>::max();
    bool rerun_gt_sdf_log_pending_ = false;
    std::size_t rerun_gt_sdf_pending_kfid_ = 0;
    torch::Tensor rerun_gt_sdf_pending_Tcw_cpu_;
    float rerun_gt_sdf_pending_fx_ = 0.0f;
    float rerun_gt_sdf_pending_fy_ = 0.0f;
    float rerun_gt_sdf_pending_cx_ = 0.0f;
    float rerun_gt_sdf_pending_cy_ = 0.0f;
    int rerun_gt_sdf_pending_width_ = 0;
    int rerun_gt_sdf_pending_height_ = 0;
};
