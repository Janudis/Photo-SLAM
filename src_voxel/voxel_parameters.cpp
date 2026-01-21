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

VoxelPipelineParams::VoxelPipelineParams(bool convert_SHs)
     : convert_SHs_(convert_SHs)
{}

VoxelOptimizationParams::VoxelOptimizationParams(
    int iterations,
    float geo_lr,
    float sh0_lr,
    float shs_lr,
    float optim_beta1,
    float optim_beta2,
    float optim_eps,
    std::vector<int> lr_decay_ckpt,
    float lr_decay_mult,
    int adapt_from,
    int adapt_every,
    int prune_until,
    float prune_thres_init,
    float prune_thres_final,
    int subdivide_until,
    int subdivide_all_until,
    float subdivide_samp_thres,
    float subdivide_prop,
    int subdivide_max_num,
    float lambda_dssim,
    float lambda_tv_density,
    int   tv_from,
    int   tv_until,
    float ss_aug_max,
    float lambda_R_concen,
    float lambda_dist,
    float lambda_T_inside,
    float lambda_ssim,
    float lambda_sparse_depth,
    int   sparse_depth_until
)
    :   iterations_(iterations),
        geo_lr_(geo_lr),
        sh0_lr_(sh0_lr),
        shs_lr_(shs_lr),
        optim_beta1_(optim_beta1),
        optim_beta2_(optim_beta2),
        optim_eps_(optim_eps),
        lr_decay_ckpt_(lr_decay_ckpt),
        lr_decay_mult_(lr_decay_mult),
        adapt_from_(adapt_from),
        adapt_every_(adapt_every),
        prune_until_(prune_until),
        prune_thres_init_(prune_thres_init),
        prune_thres_final_(prune_thres_final),
        subdivide_until_(subdivide_until),
        subdivide_all_until_(subdivide_all_until),
        subdivide_samp_thres_(subdivide_samp_thres),
        subdivide_prop_(subdivide_prop),
        subdivide_max_num_(subdivide_max_num),
        lambda_dssim_(lambda_dssim),
        lambda_tv_density_(lambda_tv_density),
        tv_from_(tv_from),
        tv_until_(tv_until),
        ss_aug_max_(ss_aug_max),
        lambda_R_concen_(lambda_R_concen),
        lambda_dist_(lambda_dist),
        lambda_T_inside_(lambda_T_inside),
        lambda_ssim_(lambda_ssim),
        lambda_sparse_depth_(lambda_sparse_depth),
        sparse_depth_until_(sparse_depth_until)
{}
