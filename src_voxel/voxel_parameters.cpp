#include "include_voxel/voxel_parameters.h"

VoxelModelParams::VoxelModelParams(
    std::filesystem::path source_path,
    std::filesystem::path model_path,
    std::filesystem::path exec_path,
    int sh_degree,
    std::string images,
    float resolution,
    bool white_background,
    std::string data_device,
    bool eval)
    : source_path_(source_path),
      model_path_(model_path),
      sh_degree_(sh_degree),
      images_(images),
      resolution_(resolution),
      white_background_(white_background),
      data_device_(data_device),
      eval_(eval)
 {
     if (source_path.is_absolute())
         source_path_ = source_path;
     else
         source_path_ = exec_path / source_path;
 
     if (model_path.is_absolute())
         model_path_ = model_path;
     else
         model_path_ = exec_path / model_path;
 }

VoxelPipelineParams::VoxelPipelineParams(bool convert_SHs, bool compute_cov3D)
     : convert_SHs_(convert_SHs), compute_cov3D_(compute_cov3D)
{}

VoxelOptimizationParams::VoxelOptimizationParams(
    int iterations,
    float geo_lr_init,
    float geo_lr_final,
    float geo_lr_delay_mult,
    float geo_lr_max_steps,
    float meta_accum_lr,
    float sh0_lr,
    float shs_lr,
    float lambda_dssim,
    int densification_interval,
    int subdiv_from,
    int subdiv_every,
    int subdiv_until,
    float subdiv_quantile,
    float subdiv_gradient_threshold,
    int prune_from,
    int prune_every,
    int prune_until,
    float prune_threshold_init,
    float prune_threshold_final,
    int min_voxels,
    int opacity_reset_interval,
    int densify_from_iter,
    int densify_until_iter,
    float densify_grad_threshold)
    :   iterations_(iterations),
        geo_lr_init_(geo_lr_init),
        geo_lr_final_(geo_lr_final),
        geo_lr_delay_mult_(geo_lr_delay_mult),
        geo_lr_max_steps_(geo_lr_max_steps),  // default value, can be adjusted later
        meta_accum_lr_(meta_accum_lr),
        sh0_lr_(sh0_lr),
        shs_lr_(shs_lr),
        lambda_dssim_(lambda_dssim),
        densification_interval_(densification_interval),
        subdiv_from_(subdiv_from),
        subdiv_every_(subdiv_every),
        subdiv_until_(subdiv_until),
        subdiv_quantile_(subdiv_quantile),
        subdiv_gradient_threshold_(subdiv_gradient_threshold),
        prune_from_(prune_from),
        prune_every_(prune_every),
        prune_until_(prune_until),
        prune_threshold_init_(prune_threshold_init),
        prune_threshold_final_(prune_threshold_final),
        min_voxels_(min_voxels),
        opacity_reset_interval_(opacity_reset_interval),
        densify_from_iter_(densify_from_iter),
        densify_until_iter_(densify_until_iter),
        densify_grad_threshold_(densify_grad_threshold)
{}
