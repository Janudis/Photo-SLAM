#include "include_voxel/voxel_model.h"

namespace py = pybind11;
namespace sv {
//------------------------------------------------------------------------------
// Constructor: initialize all voxel‐related tensors as empty leaf tensors
//------------------------------------------------------------------------------
VoxelModel::VoxelModel(const int sh_degree)
    : active_sh_degree_(0), spatial_lr_scale_(0.0),
      lr_delay_steps_(0), lr_delay_mult_(1.0), max_steps_(1000000)
{
    this->max_sh_degree_ = sh_degree;

    // Device
    if (torch::cuda::is_available())
        device_type_ = torch::kCUDA;
    else
        device_type_ = torch::kCPU;

    // Initialize all tensors on chosen device
    VOXEL_MODEL_INIT_TENSORS(device_type_);
}

VoxelModel::VoxelModel(const VoxelModelParams& model_params)
    : active_sh_degree_(0), spatial_lr_scale_(0.0),
      lr_delay_steps_(0), lr_delay_mult_(1.0), max_steps_(1000000)
{
    this->max_sh_degree_ = model_params.sh_degree_;

    // Device
    if (model_params.data_device_ == "cuda")
        device_type_ = torch::kCUDA;
    else
        device_type_ = torch::kCPU;

    // Initialize all tensors on chosen device
    VOXEL_MODEL_INIT_TENSORS(device_type_);
}

void VoxelModel::oneUpShDegree()
{
    if (this->active_sh_degree_ < this->max_sh_degree_)
        this->active_sh_degree_ += 1;
}

void VoxelModel::setShDegree(const int sh)
{
    this->active_sh_degree_ = (sh > this->max_sh_degree_ ? this->max_sh_degree_ : sh);
}

void VoxelModel::createFromPcd(
    const std::map<point3D_id_t, Point3D>& pcd,
    const float spatial_lr_scale)
{
    /* ------------------------------------------------------------------ 1. */
    this->spatial_lr_scale_ = spatial_lr_scale;
    /* ------------------------------------------------------------------ 2. */
    const int64_t num_points = static_cast<int64_t>(pcd.size());
    /* ------------------------------------------------------------------ 3. */
    torch::Tensor fused_point_cloud = torch::zeros(
        {num_points, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    torch::Tensor color = torch::zeros(
        {num_points, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

    auto pcd_it = pcd.begin();
    for (int64_t point_idx = 0; point_idx < num_points; ++point_idx) {
        const Point3D& point = (*pcd_it).second;
        fused_point_cloud.index({point_idx, 0}) = point.xyz_(0);
        fused_point_cloud.index({point_idx, 1}) = point.xyz_(1);
        fused_point_cloud.index({point_idx, 2}) = point.xyz_(2);
        color.index({point_idx, 0}) = point.color_(0);
        color.index({point_idx, 1}) = point.color_(1);
        color.index({point_idx, 2}) = point.color_(2);
        ++pcd_it;
    }

    /* ------------------------------------------------------------------ 4. */
    torch::Tensor fused_color = sh_utils::RGB2SH(color);
    const int temp = this->max_sh_degree_ + 1;
    torch::Tensor features = torch::zeros(
        {fused_color.size(0), 3, temp * temp},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    features.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(0, 3),
         0}) = fused_color;
    features.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(3, features.size(1)),
         torch::indexing::Slice(1, features.size(2))}) = 0.0f;

    /* ------------------------------------------------------------------ 5.  (voxel-specific) */
    // sizes: initialise to constant voxel length (0.05 m here)
    this->center_ = fused_point_cloud;   // (num_points,3) on device_type_
    const float default_voxel_size = 0.05f;
    this->size_ = torch::full(
        {num_points},
        default_voxel_size,
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

    // geo_ 8-vector: (σ_xx σ_xy σ_xz σ_yy σ_yz σ_zz density pad)
    this->geo_ = torch::zeros({num_points, 8},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    const float var = 0.25f * default_voxel_size * default_voxel_size;
    this->geo_.index_put_({torch::indexing::Slice(), 0}, var);
    this->geo_.index_put_({torch::indexing::Slice(), 3}, var);
    this->geo_.index_put_({torch::indexing::Slice(), 5}, var);
    const float init_density = 1.5f;
    this->geo_.index_put_({torch::indexing::Slice(), 6}, init_density);
    this->geo_.index_put_({torch::indexing::Slice(), 7}, 0.0f);           // slot 7 pad
    this->geo_.set_requires_grad(true);                                              // <---

    // std::cout << "createPcd [DBG] geo_[0] = " << geo_[0] << std::endl;

    /* ------------------------------------------------------------------ 6. */
    this->sh0_ = features.index({torch::indexing::Slice(),
                                 torch::indexing::Slice(),
                                 torch::indexing::Slice(0, 1)})
                     .transpose(1, 2)
                     .contiguous()
                     .requires_grad_();

    this->shs_ = features.index({torch::indexing::Slice(),
                                 torch::indexing::Slice(),
                                 torch::indexing::Slice(1, features.size(2))})
                     .transpose(1, 2)
                     .contiguous()
                     .requires_grad_();

    // /* ------------------------------------------------------------------ 7.  (voxel-specific) */
    // // opacity stored in logit space – start at p = 0.8
    // torch::Tensor opacities = general_utils::inverse_sigmoid(
    //     0.8f * torch::ones({num_points, 1},
    //             torch::TensorOptions().dtype(torch::kFloat32).device(device_type_)));
    // // this->opacity_ = opacities.view({num_points}).requires_grad_();

    /* ------------------------------------------------------------------ 8.  (voxel-specific bookkeeping) */
    this->oct_path_  = torch::arange(
        0, num_points,
        torch::TensorOptions().dtype(torch::kLong).device(device_type_));
    this->oct_level_ = torch::zeros(
        {num_points},
        torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
    this->exist_since_iter_ = torch::zeros(
        {num_points},
        torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
    this->subdiv_meta_ = torch::zeros(
        {num_points},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_)).requires_grad_();
    this->subdiv_meta_.retain_grad();
    this->subdiv_p_ = torch::zeros_like(this->subdiv_meta_).requires_grad_();
    this->subdiv_p_.retain_grad();
    this->subdiv_p_grad_buffer_ = torch::zeros_like(this->subdiv_p_);

    /* ------------------------------------------------------------------ 9. */
    VOXEL_MODEL_TENSORS_TO_VEC
}

void VoxelModel::increasePcd(std::vector<float> points,
                             std::vector<float> colors,
                             const int iteration)
{
    /* ------------------------------------------------------------------ 0. */
    assert(points.size() == colors.size());
    assert(points.size() % 3 == 0);
    auto num_new_points = static_cast<int64_t>(points.size() / 3);
    if (num_new_points == 0) 
        return;

    /* ------------------------------------------------------------------ 1.  copy vectors → tensors (CPU → device) */
    torch::Tensor new_point_cloud = torch::from_blob(
        points.data(), {num_new_points, 3},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);

    torch::Tensor new_colors = torch::from_blob(
        colors.data(), {num_new_points, 3},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);

    /* optional cache of sparse points – unchanged from Gaussian */
    if (sparse_points_xyz_.size(0) == 0) {
        sparse_points_xyz_   = new_point_cloud;
        sparse_points_color_ = new_colors;
    } else {
        sparse_points_xyz_   = torch::cat({sparse_points_xyz_,   new_point_cloud}, 0);
        sparse_points_color_ = torch::cat({sparse_points_color_, new_colors},   0);
    }

    /* ------------------------------------------------------------------ 2. SH feature construction (verbatim) */
    torch::Tensor new_fused_colors = sh_utils::RGB2SH(new_colors);
    const int temp = this->max_sh_degree_ + 1;
    torch::Tensor features = torch::zeros(
        {new_fused_colors.size(0), 3, temp * temp},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    features.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(0, 3),
         0}) = new_fused_colors;
    features.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(3, features.size(1)),
         torch::indexing::Slice(1, features.size(2))}) = 0.0f;

    /* ------------------------------------------------------------------ 3. voxel-specific attribute init */
    const float default_vox_size = 0.05f;                                         // <--- voxel-specific
    torch::Tensor new_sizes = torch::full(
        {num_new_points}, default_vox_size,
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_)); // <---

    // geo 8-vector with diagonal cov = (l/2)^2, others zero
    torch::Tensor new_geo = torch::zeros(
        {num_new_points, 8},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_)); // <---
    const float var = 0.25f * default_vox_size * default_vox_size;                // <---
    new_geo.index_put_({torch::indexing::Slice(), 0}, var);                       // σ_xx
    new_geo.index_put_({torch::indexing::Slice(), 3}, var);                       // σ_yy
    new_geo.index_put_({torch::indexing::Slice(), 5}, var);                       // σ_zz
    const float init_density = 1.5f;
    new_geo.index_put_({torch::indexing::Slice(), 6}, init_density);
    new_geo.index_put_({torch::indexing::Slice(), 7}, 0.0f);
    
    // std::cout << "increadePcd [DBG] geo_[0] = " << geo_[0] << std::endl;

    // SH terms
    torch::Tensor new_sh0 = features.index({torch::indexing::Slice(),
                                            torch::indexing::Slice(),
                                            torch::indexing::Slice(0, 1)})
                                .transpose(1, 2)
                                .contiguous();
    torch::Tensor new_shs = features.index({torch::indexing::Slice(),
                                            torch::indexing::Slice(),
                                            torch::indexing::Slice(1, features.size(2))})
                                .transpose(1, 2)
                                .contiguous();

    // opacity logits start from p = 0.8
    // torch::Tensor new_opacity = general_utils::inverse_sigmoid(
    //     0.8f *
    //     torch::ones({num_new_points, 1},
    //         torch::TensorOptions().dtype(torch::kFloat32).device(device_type_)))
    //         .view({num_new_points});                                              // <---

    // octree bookkeeping
    torch::Tensor new_oct_paths = torch::arange(
        this->center_.size(0),
        this->center_.size(0) + num_new_points,
        torch::TensorOptions().dtype(torch::kLong).device(device_type_));   // <---
    torch::Tensor new_oct_lvls = torch::zeros(
        {num_new_points},
        torch::TensorOptions().dtype(torch::kInt32).device(device_type_));  // <---
    torch::Tensor new_meta      = torch::zeros(
        {num_new_points},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_)); // <---
    torch::Tensor new_subdiv_p  = torch::zeros_like(new_meta);                     // <---
    torch::Tensor new_exist_iter= torch::full(
        {num_new_points}, iteration,
        torch::TensorOptions().dtype(torch::kInt32).device(device_type_));   // <---

    /* ------------------------------------------------------------------ 4. concatenate to existing tensors */
    this->center_      = torch::cat({this->center_,   new_point_cloud}, 0).requires_grad_(false);
    this->size_        = torch::cat({this->size_,     new_sizes},       0).requires_grad_(false); // <---
    this->geo_         = torch::cat({this->geo_,      new_geo},         0).requires_grad_(true);
    this->sh0_         = torch::cat({this->sh0_,      new_sh0},         0).requires_grad_(true);
    this->shs_         = torch::cat({this->shs_,      new_shs},         0).requires_grad_(true);
    // this->opacity_     = torch::cat({this->opacity_,  new_opacity},     0).requires_grad_(true);  // <---
    this->oct_path_    = torch::cat({this->oct_path_, new_oct_paths},   0);                       // <---
    this->oct_level_   = torch::cat({this->oct_level_,new_oct_lvls},    0);                       // <---
    this->exist_since_iter_ = torch::cat({this->exist_since_iter_, new_exist_iter}, 0);           // <---
    this->subdiv_meta_ = torch::cat({this->subdiv_meta_, new_meta},     0).requires_grad_(true);
    this->subdiv_meta_.retain_grad();
    this->subdiv_p_    = torch::cat({this->subdiv_p_,  new_subdiv_p},   0).requires_grad_(true);
    this->subdiv_p_.retain_grad();

    /* ------------------------------------------------------------------ 5. resize gradient buffer if needed */
    if (this->subdiv_p_grad_buffer_.numel() != this->subdiv_p_.numel()) {
        torch::Tensor new_buf = torch::zeros_like(this->subdiv_p_);
        if (this->subdiv_p_grad_buffer_.numel() > 0)
            new_buf.narrow(0, 0, this->subdiv_p_grad_buffer_.numel())
                   .copy_(this->subdiv_p_grad_buffer_);
        this->subdiv_p_grad_buffer_ = std::move(new_buf);
    }
}

void VoxelModel::increasePcd(torch::Tensor& new_point_cloud, torch::Tensor& new_colors, const int iteration)
{
    auto num_new_points = new_point_cloud.size(0);
    if (num_new_points == 0)
        return;

    if (sparse_points_xyz_.size(0) == 0) {
        sparse_points_xyz_ = new_point_cloud;
        sparse_points_color_ = new_colors;
    }
    else {
        sparse_points_xyz_ = torch::cat({sparse_points_xyz_, new_point_cloud}, /*dim=*/0);
        sparse_points_color_ = torch::cat({sparse_points_color_, new_colors}, /*dim=*/0);
    }

    /* ------------------------------------------------------------------ 1 : SH colours */
    torch::Tensor new_fused_colors = sh_utils::RGB2SH(new_colors);
    auto temp = this->max_sh_degree_ + 1;
    torch::Tensor features = torch::zeros(
        {new_fused_colors.size(0), 3, temp * temp},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    features.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(0, 3),
         0}) = new_fused_colors;
    features.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(3, features.size(1)),
         torch::indexing::Slice(1, features.size(2))}) = 0.0f;

    /* ------------------------------------------------------------------ 2 : default voxel attributes */
    const float default_vox_size = 0.05f;
    torch::Tensor new_size = torch::full(
        {num_new_points}, default_vox_size,
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

    torch::Tensor new_geo = torch::zeros(
        {num_new_points, 8}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    const float var = 0.25f * default_vox_size * default_vox_size;
    new_geo.index_put_({torch::indexing::Slice(), 0}, var);
    new_geo.index_put_({torch::indexing::Slice(), 3}, var);
    new_geo.index_put_({torch::indexing::Slice(), 5}, var);

    torch::Tensor new_sh0 = features.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(),
        torch::indexing::Slice(0,1)})
        .transpose(1,2).contiguous();                  // (P , 1 , 3) → (P ,3)

    torch::Tensor new_shs = features.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(),
        torch::indexing::Slice(1, features.size(2))})
        .transpose(1,2).contiguous();                  // (P ,  (M^2-1) , 3)

    /* opacity logits at p = 0.8 */
    // torch::Tensor op_init = general_utils::inverse_sigmoid(
    //     0.8f * torch::ones({num_new_points,1},
    //         torch::TensorOptions().dtype(torch::kFloat32).device(device_type_)));
    // torch::Tensor new_opacity = op_init.view({num_new_points});

    /* ------------------------------------------------------------------ 3 : book-keeping tensors */
    torch::Tensor new_oct   = torch::arange(
        center_.size(0), center_.size(0)+num_new_points,
        torch::TensorOptions().dtype(torch::kLong).device(device_type_));

    torch::Tensor new_lvl   = torch::zeros({num_new_points},
        torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

    torch::Tensor new_meta  = torch::zeros({num_new_points},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    torch::Tensor new_subP  = torch::zeros_like(new_meta);

    torch::Tensor new_exist = torch::full(
        {num_new_points}, iteration,
        torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

    /* ------------------------------------------------------------------ 4 : concatenate into the model  */
    center_      = torch::cat({center_,   new_point_cloud     }, 0);          // (no-grad)
    size_        = torch::cat({size_,     new_size            }, 0);
    geo_         = torch::cat({geo_,      new_geo             }, 0).requires_grad_(true);
    sh0_         = torch::cat({sh0_,      new_sh0             }, 0).requires_grad_(true);
    shs_         = torch::cat({shs_,      new_shs             }, 0).requires_grad_(true);
    // opacity_     = torch::cat({opacity_,  new_opacity         }, 0).requires_grad_(true);
    oct_path_    = torch::cat({oct_path_, new_oct             }, 0);
    oct_level_   = torch::cat({oct_level_,new_lvl             }, 0);
    exist_since_iter_ = torch::cat({exist_since_iter_, new_exist}, 0);
    subdiv_meta_ = torch::cat({subdiv_meta_, new_meta }, 0).requires_grad_(true);
    subdiv_p_    = torch::cat({subdiv_p_,    new_subP }, 0).requires_grad_(true);
    subdiv_meta_.retain_grad();
    subdiv_p_.retain_grad();

    /* ------------------------------------------------------------------ 5 : keep grad-buffer sized */
    if (subdiv_p_grad_buffer_.numel() != subdiv_p_.numel())
        subdiv_p_grad_buffer_ = torch::zeros_like(subdiv_p_);

}

torch::Tensor VoxelModel::replaceTensorToOptimizer(torch::Tensor& tensor, int tensor_idx)
{
    auto& param = this->optimizer_->param_groups()[tensor_idx].params()[0];
    auto& state = optimizer_->state();
    auto key = c10::guts::to_string(param.unsafeGetTensorImpl());
    auto& stored_state = static_cast<torch::optim::AdamParamState&>(*state[key]);
    auto new_state = std::make_unique<torch::optim::AdamParamState>();
    new_state->step(stored_state.step());
    new_state->exp_avg(torch::zeros_like(tensor));
    new_state->exp_avg_sq(torch::zeros_like(tensor));
    // new_state->max_exp_avg_sq(stored_state.max_exp_avg_sq().clone()); // needed only when options.amsgrad(true), which is false by default

    state.erase(key);
    param = tensor.requires_grad_();
    key = c10::guts::to_string(param.unsafeGetTensorImpl());
    state[key] = std::move(new_state);

    auto optimizable_tensors = param;
    return optimizable_tensors;
}

void VoxelModel::scaledTransformationPostfix(torch::Tensor& new_geo)
{
    // We only need to re-wire the geometry key (geo_) back into Adam (param_group 0).
    torch::Tensor optimizable_geo = this->replaceTensorToOptimizer(new_geo, /*group*/0);
    this->geo_ = optimizable_geo;
    // Update our helper vector so future replaceTensorToOptimizer calls still work:
    this->Tensor_vec_geo_ = { this->geo_ };
    // Note: we do *not* have a scaling tensor in the optimizer for voxels,
    // so we omit any scaling_ logic here.
}

void VoxelModel::applyScaledTransformation(
    const float s,
    const Sophus::SE3f T)
{
    torch::NoGradGuard no_grad;
    // pt <- (s * Ryw * pt + tyw)
    this->center_ *= s;

    torch::Tensor T_tensor =
        tensor_utils::EigenMatrix2TorchTensor(T.matrix(), device_type_).transpose(0, 1);
    // use the same helper as Photo-SLAM to apply SE3
    torch::Tensor ones = torch::ones({center_.size(0), 1},
                                 center_.options().dtype(torch::kFloat32));
    torch::Tensor center_h = torch::cat({center_, ones}, 1);   // (N,4)
    center_ = (T_tensor.matmul(center_h.t())).t()
                .index({torch::indexing::Slice(),
                        torch::indexing::Slice(0,3)});

    // we don’t have a “scaling_” tensor in Adam—our size_ lives outside—so scale size_ here
    this->size_ *= s;

    // geo_ stores variances along diag; those scale with s²
    const float var_scale = s * s;
    this->geo_.index_put_({torch::indexing::Slice(), 0},
        this->geo_.index({torch::indexing::Slice(), 0}) * var_scale);
    this->geo_.index_put_({torch::indexing::Slice(), 3},
        this->geo_.index({torch::indexing::Slice(), 3}) * var_scale);
    this->geo_.index_put_({torch::indexing::Slice(), 5},
        this->geo_.index({torch::indexing::Slice(), 5}) * var_scale);

    scaledTransformationPostfix(this->geo_);
}

void VoxelModel::scaledTransformVisiblePointsOfKeyframe(
    torch::Tensor& point_not_transformed_flags,  // (N) bool
    torch::Tensor& diff_pose,                    // (4×4)  SE(3) in world
    torch::Tensor& kf_world_view_transform,      // unused (placeholder)
    torch::Tensor& kf_full_proj_transform,       // unused (placeholder)
    const int      kf_creation_iter,
    const int      stable_num_iter_existence,
    int&           num_transformed,
    const float    scale)
{
    torch::NoGradGuard no_grad;

    // 1) grab our “points” and (no rots for voxels)
    torch::Tensor points = this->center_;

    // 2) find “unstable” (just‐born) voxels
    torch::Tensor point_unstable_flags = torch::abs(
        this->exist_since_iter_ - kf_creation_iter
    ) < stable_num_iter_existence;

    // 3) those still eligible to transform
    torch::Tensor to_check = 
        point_not_transformed_flags.logical_and(~point_unstable_flags);
    if (!to_check.any().item<bool>()) return;

    // 4) (we cheat: treat *all* to_check as visible)
    torch::Tensor to_transform = to_check;
    if (!to_transform.any().item<bool>()) return;

    // 5) apply diff_pose to centers[to_transform]
    torch::Tensor idx = to_transform.nonzero().view(-1);
    num_transformed += idx.numel();

    // homogeneous
    torch::Tensor sel_centers = points.index({idx});          // (M,3)
    torch::Tensor ones = torch::ones(
        {idx.size(0),1},
        torch::TensorOptions().dtype(points.dtype()).device(points.device())
    );
    torch::Tensor centers_h = torch::cat({sel_centers, ones}, 1); // (M,4)
    torch::Tensor updated = 
        (diff_pose.matmul(centers_h.t()))
        .t()
        .index({torch::indexing::Slice(), torch::indexing::Slice(0,3)}); // (M,3)

    this->center_.index_put_({idx}, updated);

    // 6) scale size_ and geo_ diagonals
    this->size_.index_put_({idx}, this->size_.index({idx}) * scale);
    float vs = scale*scale;
    this->geo_.index_put_({idx, 0}, this->geo_.index({idx, 0}) * vs);
    this->geo_.index_put_({idx, 3}, this->geo_.index({idx, 3}) * vs);
    this->geo_.index_put_({idx, 5}, this->geo_.index({idx, 5}) * vs);

    // 7) mark done
    point_not_transformed_flags.index_put_({idx}, false);

    // === Postfix: re-wire our geo_ back into Adam’s param-group[0] ===
    torch::Tensor optimizable_geo = this->replaceTensorToOptimizer(
        this->geo_, /*tensor_idx=*/0
    );
    this->geo_ = optimizable_geo;
    this->Tensor_vec_geo_ = { this->geo_ };
}

void VoxelModel::trainingSetup(const VoxelOptimizationParams& training_args)
{
    // exactly as in Photo-SLAM:
    setPercentDense(training_args.percent_dense_);
    this->subdiv_p_grad_buffer_ = torch::zeros_like(this->subdiv_p_);

    torch::optim::AdamOptions adam_options;
    adam_options.set_lr(0.0);
    adam_options.eps() = 1e-15;

    // start with geometry group (analogous to xyz_ in Photo-SLAM)
    this->optimizer_.reset(
        new torch::optim::Adam(Tensor_vec_geo_, adam_options));
    optimizer_->param_groups()[0]
        .options()
        .set_lr(training_args.geo_lr_init_ * this->spatial_lr_scale_);

    // add SH-DC group
    optimizer_->add_param_group(Tensor_vec_sh0_);
    optimizer_->param_groups()[1]
        .options()
        .set_lr(training_args.sh0_lr_);

    // add higher-order SH group
    optimizer_->add_param_group(Tensor_vec_shs_);
    optimizer_->param_groups()[2]
        .options()
        .set_lr(training_args.sh0_lr_/ 20.0);

    // // add opacity group
    // optimizer_->add_param_group(Tensor_vec_opacity_);
    // optimizer_->param_groups()[3]
    //     .options()
    //     .set_lr(training_args.opacity_lr_);

    // store schedule parameters
    lr_init_    = training_args.geo_lr_init_  * this->spatial_lr_scale_;
    lr_final_   = training_args.geo_lr_final_ * this->spatial_lr_scale_;
    lr_delay_mult_  = training_args.geo_lr_delay_mult_;
    max_steps_      = training_args.geo_lr_max_steps_;
}

//------------------------------------------------------------------------------
// Update the optimizer’s LR for “geo” param group at iteration `step`.
//------------------------------------------------------------------------------
float VoxelModel::updateLearningRate(int step)
{
    float lr = this->exponLrFunc(step);
    optimizer_->param_groups()[0].options().set_lr(lr);
    return lr;
}
//------------------------------------------------------------------------------
// Manually set per-group LRs at runtime (mirrors GaussianModel setters).
//------------------------------------------------------------------------------
void VoxelModel::setGeoLearningRate(float lr) 
{
    optimizer_->param_groups()[0].options().set_lr(lr);
}
void VoxelModel::setSh0LearningRate(float lr) 
{
    optimizer_->param_groups()[1].options().set_lr(lr);
}
void VoxelModel::setShsLearningRate(float lr) 
{
    optimizer_->param_groups()[2].options().set_lr(lr);
}

//------------------------------------------------------------------------------
// Prune (remove) voxels that do not satisfy mask_keep (shape (N,) bool).
// Mirrors GaussianModel::prunePoints but simpler because we only track geo_, sh0_, shs_, opacity_, subdiv.
//------------------------------------------------------------------------------
void VoxelModel::prune(const torch::Tensor& mask_keep)
{
    // mask_keep is (N,) bool on same device
    auto mk = mask_keep.to(device_type_);
    center_      = center_.index({mk}).requires_grad_(false);
    size_        = size_.index({mk}).requires_grad_(false);
    geo_         = geo_.index({mk}).requires_grad_(true);
    sh0_         = sh0_.index({mk}).requires_grad_(true);
    shs_         = shs_.index({mk}).requires_grad_(true);
    // opacity_     = opacity_.index({mk}).requires_grad_(true);
    oct_path_    = oct_path_.index({mk});
    oct_level_   = oct_level_.index({mk});
    subdiv_meta_ = subdiv_meta_.index({mk}).requires_grad_(true);
    subdiv_meta_.retain_grad();
    subdiv_p_    = subdiv_p_.index({mk}).requires_grad_(true);
    subdiv_p_.retain_grad();
    subdiv_p_grad_buffer_ = torch::zeros_like(subdiv_p_);

    // Any other stateful buffers (e.g. training‐stats) would be pruned similarly.
}

//------------------------------------------------------------------------------
// Subdivide: mirror VoxelTrainer::subdivide(), but as a member of VoxelModel.
// Each “parent” voxel spawns 8 children. The logic is identical to the old VoxelTrainer.
//------------------------------------------------------------------------------
void VoxelModel::subdivide(const torch::Tensor& mask)
{
    TORCH_CHECK(mask.dtype() == torch::kBool, "subdivide(): mask must be boolean");

    torch::NoGradGuard no_grad;
    auto device = center_.device();

    // 1) Find indices of parents to subdivide
    torch::Tensor idx_parent = mask.nonzero().view(-1);
    if (idx_parent.numel() == 0) return;

    // Validate bounds
    {
        int64_t N = center_.size(0);
        auto max_idx = idx_parent.max().item<int64_t>();
        auto min_idx = idx_parent.min().item<int64_t>();
        if (min_idx < 0 || max_idx >= N) {
            throw std::runtime_error("subdivide(): parent index out of bounds");
        }
    }

    // Parent levels & filter out those already at max octree level
    torch::Tensor parent_level = oct_level_.index({idx_parent});            // (P,)
    torch::Tensor below_max    = parent_level < torch::full_like(parent_level, static_cast<int>(sv::MAX_OCT_LEVEL));
    if (!below_max.any().item<bool>()) return;
    idx_parent = idx_parent.index({below_max});
    parent_level = parent_level.index({below_max});

    // 2) Keep indices that are not subdivided
    torch::Tensor idx_keep = (~mask).nonzero().view(-1);

    // Helpers to index‐preserve
    auto keep_t = [&](const torch::Tensor& t) {
        return t.index({idx_keep});
    };
    auto gather  = [&](const torch::Tensor& t) {
        return t.index({idx_parent});
    };

    // 3) Extract “kept” (non‐parents) voxel attributes
    torch::Tensor center_k      = keep_t(center_);
    torch::Tensor size_k        = keep_t(size_);
    torch::Tensor geo_k         = keep_t(geo_);
    torch::Tensor sh0_k         = keep_t(sh0_);
    torch::Tensor shs_k         = keep_t(shs_);
    // torch::Tensor opacity_k     = keep_t(opacity_);
    torch::Tensor oct_k         = keep_t(oct_path_);
    torch::Tensor lvl_k         = keep_t(oct_level_);
    torch::Tensor meta_k        = keep_t(subdiv_meta_);

    // 4) Gather parent attributes
    torch::Tensor c_par         = gather(center_);       // (P,3)
    torch::Tensor s_par         = gather(size_);         // (P)
    torch::Tensor geo_par       = gather(geo_);          // (P,8)
    torch::Tensor sh0_par       = gather(sh0_);          // (P,3)
    torch::Tensor shs_par       = gather(shs_);          // (P,45,3)
    // torch::Tensor op_par        = gather(opacity_);      // (P)
    torch::Tensor path_p        = oct_path_.index({idx_parent});     // (P)
    // parent_level is already computed

    const int64_t n_parent = idx_parent.size(0);
    const int64_t n_child  = n_parent * 8;

    // 5) Compute child size: half of parent, clamped to MIN_VOX_SIZE
    torch::Tensor s_child = (s_par * 0.5f).repeat({8});   // (8P)
    s_child = torch::clamp(s_child,
                           torch::full_like(s_child, MIN_VOX_SIZE),
                           torch::full_like(s_child, std::numeric_limits<float>::infinity()));

    // 6) Compute child centers: parent centers ± 0.25 * parent_size
    static const float offs_data[8][3] = {
        {-0.25f, -0.25f, -0.25f},
        { 0.25f, -0.25f, -0.25f},
        {-0.25f,  0.25f, -0.25f},
        { 0.25f,  0.25f, -0.25f},
        {-0.25f, -0.25f,  0.25f},
        { 0.25f, -0.25f,  0.25f},
        {-0.25f,  0.25f,  0.25f},
        { 0.25f,  0.25f,  0.25f},
    };
    torch::Tensor offs = torch::from_blob((float*)offs_data, {8,3}, torch::TensorOptions().dtype(torch::kFloat32).device(device)).clone();  
    // Broadcast parent centers and add offsets * parent_size
    torch::Tensor c_child = c_par.repeat_interleave(8, /*dim=*/0)
                          + (offs.repeat({n_parent, 1}) * s_par.repeat_interleave(8).unsqueeze(1));  // (8P,3)

    // 7) Inherit attributes from parent to children
    auto repeat8 = [&](const torch::Tensor& t)->torch::Tensor {
        return t.repeat_interleave(8, /*dim=*/0);
    };
    torch::Tensor geo_child   = repeat8(geo_par);        // (8P,8)
    const float init_density = 1.5f;
    geo_child.index_put_({torch::indexing::Slice(), 6}, init_density);
    geo_child.index_put_({torch::indexing::Slice(), 7}, 0.0f);  
    
    torch::Tensor sh0_child   = repeat8(sh0_par);        // (8P,3)
    torch::Tensor shs_child   = repeat8(shs_par);        // (8P,45,3)
    // torch::Tensor op_child    = repeat8(op_par);         // (8P)

    // 8) Child levels & octree paths
    torch::Tensor lvl_child = parent_level.repeat_interleave(8) + 1;                       // (8P)
    torch::Tensor idx8      = torch::arange(8, torch::TensorOptions().dtype(torch::kLong).device(device)).repeat({n_parent});  // (8P)
    torch::Tensor path_child = ((path_p * 8).repeat_interleave(8)) | idx8;  // (8P)

    // 9) Child subdiv_meta = zeros
    torch::Tensor meta_child = torch::zeros({n_child}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // 10) Concatenate “keep” and “child” groups  
    center_      = torch::cat({center_k,     c_child},      /*dim=*/0).requires_grad_(false);
    size_        = torch::cat({size_k,       s_child},      /*dim=*/0).requires_grad_(false);
    geo_         = torch::cat({geo_k,        geo_child},    /*dim=*/0).requires_grad_(true);
    sh0_         = torch::cat({sh0_k,        sh0_child},    /*dim=*/0).requires_grad_(true);
    shs_         = torch::cat({shs_k,        shs_child},    /*dim=*/0).requires_grad_(true);
    // opacity_     = torch::cat({opacity_k,    op_child},     /*dim=*/0).requires_grad_(true);
    oct_path_    = torch::cat({oct_k,        path_child},   /*dim=*/0);
    oct_level_   = torch::cat({lvl_k,        lvl_child},    /*dim=*/0);
    subdiv_meta_ = torch::cat({meta_k,       meta_child},   /*dim=*/0).requires_grad_(true);
    subdiv_meta_.retain_grad();

    // Reset subdiv_p_: new zero tensor of same shape
    subdiv_p_    = torch::zeros_like(subdiv_meta_).requires_grad_(true);
    subdiv_p_.retain_grad();

    // 11) Rebuild subdiv_p_grad_buffer_
    {
        auto old_buf = subdiv_p_grad_buffer_;
        const int64_t oldN = old_buf.numel();
        const int64_t newN = subdiv_p_.numel();
        if (oldN != newN) {
            auto new_buf = torch::zeros_like(subdiv_p_);
            if (oldN > 0) {
                new_buf.narrow(0, 0, oldN).copy_(old_buf);
            }
            subdiv_p_grad_buffer_ = std::move(new_buf);
        }
    }
}

//------------------------------------------------------------------------------
// Replace the current subdiv_meta_ with `updated` (leaf, requires_grad).
//------------------------------------------------------------------------------
void VoxelModel::setSubdivMeta(const torch::Tensor& updated)
{
    subdiv_meta_ = updated.clone().to(device_type_).requires_grad_(true);
    subdiv_meta_.retain_grad();
    // if (auto g = subdiv_meta_.grad(); g.defined()) {
    //     g.detach_();
    // }
}

//------------------------------------------------------------------------------
// Retrieve the gradient of subdiv_p_ (must have run backward).
//------------------------------------------------------------------------------
torch::Tensor VoxelModel::getSubdivPriorityGrad() const
{
    if (!subdiv_p_.defined() || !subdiv_p_.requires_grad()) {
        throw std::runtime_error("[VoxelModel] subdiv_p is not initialized for grad.");
    }
    auto grad = subdiv_p_.grad();
    if (!grad.defined()) {
        throw std::runtime_error("[VoxelModel] subdiv_p.grad() is undefined; call backward first.");
    }
    return grad;
}

//------------------------------------------------------------------------------
// Scatter‐add `parent_grads` back into subdiv_p_grad_buffer_ at indices `parent_idx`.
//------------------------------------------------------------------------------
void VoxelModel::accumulateSubdivGradients(const torch::Tensor& parent_idx,
                                           const torch::Tensor& parent_grads)
{
    if (!subdiv_p_grad_buffer_.defined()) {
        // same shape and dtype as subdiv_p_  → 1-D
        subdiv_p_grad_buffer_ = torch::zeros_like(subdiv_p_);
    }
    TORCH_CHECK(subdiv_p_grad_buffer_.sizes() == subdiv_p_.sizes(),
                "accumulateSubdivGradients: size mismatch");

    subdiv_p_grad_buffer_.scatter_add_(
        /*dim=*/0,
        /*index=*/parent_idx.to(device_type_),   // (K)
        /*src=*/parent_grads.to(device_type_)    // (K)
    );
}

//------------------------------------------------------------------------------
// Return all optimizable parameters for the outer training loop.
// Mirrors GaussianModel::parameters() → returns {geo_, sh0_, shs_}.
//------------------------------------------------------------------------------
std::vector<torch::Tensor> VoxelModel::parameters() const
{
    return { geo_, sh0_, shs_, subdiv_p_ };
}

//------------------------------------------------------------------------------
// Lookup any named tensor by string (for debugging or inspection).
// Mirrors VoxelTrainer::get_tensor.
//------------------------------------------------------------------------------
torch::Tensor VoxelModel::getTensor(const std::string& name) const
{
    if (name == "center")       return center_;
    else if (name == "size")    return size_;
    else if (name == "geo")     return geo_;
    else if (name == "sh0")     return sh0_;
    else if (name == "shs")     return shs_;
    // else if (name == "opacity") return opacity_;
    else if (name == "octpath") return oct_path_;
    else if (name == "octlevel")return oct_level_;
    else if (name == "subdiv_meta") return subdiv_meta_;
    else if (name == "subdiv_p")    return subdiv_p_;
    else throw std::runtime_error("VoxelModel::getTensor(): unknown name \"" + name + "\"");
}

std::unordered_map<std::string, torch::Tensor>
VoxelModel::render(
    const MiniCam&                   cam,
    const py::array_t<uint8_t>&      rgb_image,
    const std::string&               output_dir
) const {
    // 1) Acquire GIL and import the Python function once
    py::gil_scoped_acquire gil;
    // static py::object py_render = py::module_::import(
    // "scripts_voxel.python_svraster_bridge.renderer_wrapper")
    //     .attr("render");
    static py::object py_render;
    if (!py_render) {  // initialize exactly once
        try {
            py_render = py::module_::import("scripts_voxel.python_svraster_bridge.renderer_wrapper")
                            .attr("render");
            std::cerr << "[INFO] Successfully imported renderer_wrapper." << std::endl;
        }
        catch (const py::error_already_set& e) {
            std::cerr << "[PYBIND11] Exception while importing renderer_wrapper:\n"
                    << e.what() << std::endl;
            return {};  // Early return if import fails
        }
    }
    // 2) Sanity check
    if (center_.numel() == 0) {
        std::cout << "[INFO] Skipping render — empty voxel data\n";
        return {};
    }
    if (!subdiv_p_.defined() || !subdiv_p_.isfinite().all().item<bool>()) {
        throw std::runtime_error("Invalid subdiv_p_ before rendering.");
    }

    // 3) Pack all voxel buffers into a Python dict
    py::dict d;
    // torch::Tensor cov6   = geo_.slice(/*dim=*/1, /*start=*/0, /*end=*/6);   // (N,6)
    // // torch::Tensor density= opacity_.sigmoid().unsqueeze(1);                // (N,1)
    // torch::Tensor density = geo_.slice(/*dim=*/1, /*start=*/6, /*end=*/7); // (N,1)
    // // torch::Tensor pad1   = torch::zeros_like(density);                      // (N,1)
    // torch::Tensor pad1 = torch::zeros({density.size(0),1}, density.options());
    // torch::Tensor geos8  = torch::cat({cov6, density, pad1}, /*dim=*/1);    // (N,8)
    d["geos"]        = py::cast(geo_);
    d["colors"]      = py::cast(sh0_);
    d["shs"]         = py::cast(shs_);
    d["subdiv_p"]    = py::cast(subdiv_p_);
    d["octpaths"]    = py::cast(oct_path_.cpu());
    d["centers"]     = py::cast(center_.cpu());
    d["vox_lengths"] = py::cast(size_.cpu());
    d["octlevels"]   = py::cast(oct_level_.cpu());
    d["subdiv_meta"] = py::cast(subdiv_meta_.cpu());

    // 4) Build the Python MiniCam
    py::object py_cam = MiniCam_to_py(cam);

    // 5) Call into Python
    py::object out = py_render(py_cam, d, rgb_image, output_dir);

    // 6) Unpack exactly the "rgb" tensor
    torch::Tensor rgb_t = out.attr("get")("rgb").cast<torch::Tensor>();
    return { { "rgb", rgb_t } };
}

float VoxelModel::percentDense()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return percent_dense_;
}

void VoxelModel::setPercentDense(const float percent_dense)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    percent_dense_ = percent_dense;
}

float VoxelModel::exponLrFunc(int step) 
{
    if (step < 0 || (lr_init_ == 0.0f && lr_final_ == 0.0f)) {
        return 0.0f;
    }
    float delay_rate;
    if (lr_delay_steps_ > 0) 
        delay_rate = lr_delay_mult_ + (1.0f - lr_delay_mult_) * std::sin(M_PI_2f32 * std::clamp(static_cast<float>(step) / lr_delay_steps_, 0.0f, 1.0f));
    else
        delay_rate = 1.0f;
    float t = std::clamp(static_cast<float>(step) / max_steps_, 0.0f, 1.0f);
    float log_lerp = std::exp(std::log(lr_init_) * (1.0f - t) + std::log(lr_final_) * t);
    return delay_rate * log_lerp;
}

} // namespace sv

