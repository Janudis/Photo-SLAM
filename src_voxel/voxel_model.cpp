// voxel_model.cpp
// Mirrors GaussianModel.cpp but adapted for SVRaster voxels.

#include "include_voxel/voxel_model.h"
#include "include_voxel/mini_cam.h"                 // for MiniCam_to_py(cam)
#include <pybind11/embed.h>                         // for Python bridge
#include <torch/extension.h>
#include <iostream>

namespace py = pybind11;
namespace sv {

//------------------------------------------------------------------------------
// Constructor: initialize all voxel‐related tensors as empty leaf tensors
//------------------------------------------------------------------------------
VoxelModel::VoxelModel(int sh_degree)
    : active_sh_degree_(0),
      max_sh_degree_(sh_degree),
      lr_delay_steps_(0),
      lr_delay_mult_(1.0),
      max_steps_(1'000'000)
{
    // Device selection
    if (torch::cuda::is_available()) {
        device_type_ = torch::kCUDA;
    } else {
        device_type_ = torch::kCPU;
    }

    // Initialize all tensors on chosen device
    VOXEL_MODEL_INIT_TENSORS(device_type_);
}

VoxelModel::VoxelModel(const VoxelModelParams& model_params)
    : active_sh_degree_(0),
      max_sh_degree_(model_params.sh_degree_),
      lr_delay_steps_(0),
      lr_delay_mult_(1.0),
      max_steps_(1'000'000)
{
    // Device selection based on model_params
    if (model_params.data_device_ == "cuda") {
        device_type_ = torch::kCUDA;
    } else {
        device_type_ = torch::kCPU;
    }

    // Initialize all tensors on chosen device
    VOXEL_MODEL_INIT_TENSORS(device_type_);
}

//------------------------------------------------------------------------------
// “Increase” the voxel point cloud by providing a list of new points/colors.
// Builds exactly like GaussianModel::increasePcd, but uses SVRaster’s voxel fields.
//------------------------------------------------------------------------------
void VoxelModel::increasePcd(const std::vector<Eigen::Vector3f>& pts,
                             const std::vector<Eigen::Vector3f>& cols,
                             int /*iteration*/)
{
    if (pts.empty()) return;
    torch::NoGradGuard no_grad;
    auto device = center_.device();
    const int64_t P = static_cast<int64_t>(pts.size());

    // 1) Create new‐voxel attributes (centers, sizes, geo, sh0, shs, opacity, oct, lvl, meta, subdiv_p)
    torch::Tensor new_centers  = torch::empty({P, 3},   torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor new_sizes    = torch::full({P}, 0.05f, torch::TensorOptions().dtype(torch::kFloat32).device(device)); 
    torch::Tensor new_geo      = torch::zeros({P, 8},   torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor new_sh0      = torch::empty({P, 3},   torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor new_shs      = torch::zeros({P, 45, 3},torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor new_opacity  = torch::ones({P},       torch::TensorOptions().dtype(torch::kFloat32).device(device)) * 0.8f;
    // Unique octree path IDs: just use increasing integers
    torch::Tensor new_oct      = torch::arange(center_.size(0), center_.size(0) + P, torch::TensorOptions().dtype(torch::kLong).device(device));
    torch::Tensor new_lvl      = torch::zeros({P},       torch::TensorOptions().dtype(torch::kInt32).device(device));
    torch::Tensor new_meta     = torch::zeros({P},       torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor new_subdivP  = torch::zeros({P},       torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // Fill in new_centers and new_sh0 from pts, cols
    {
        auto ctr_acc = new_centers.accessor<float,2>();
        auto col_acc = new_sh0.accessor<float,2>();
        for (int64_t i = 0; i < P; ++i) {
            ctr_acc[i][0] = pts[i].x();
            ctr_acc[i][1] = pts[i].y();
            ctr_acc[i][2] = pts[i].z();
            col_acc[i][0] = cols[i].x();
            col_acc[i][1] = cols[i].y();
            col_acc[i][2] = cols[i].z();
        }
    }

    // 2) Concatenate to existing tensors; re‐enable requires_grad on leaf
    center_      = torch::cat({center_,   new_centers},  /*dim=*/0).requires_grad_(false);
    size_        = torch::cat({size_,     new_sizes  },  /*dim=*/0).requires_grad_(false);
    geo_         = torch::cat({geo_,      new_geo    },  /*dim=*/0).requires_grad_(true);
    sh0_         = torch::cat({sh0_,      new_sh0    },  /*dim=*/0).requires_grad_(true);
    shs_         = torch::cat({shs_,      new_shs    },  /*dim=*/0).requires_grad_(true);
    opacity_     = torch::cat({opacity_,  new_opacity},  /*dim=*/0).requires_grad_(true);
    oct_path_    = torch::cat({oct_path_, new_oct   },  /*dim=*/0);
    oct_level_   = torch::cat({oct_level_,new_lvl   },  /*dim=*/0);
    subdiv_meta_ = torch::cat({subdiv_meta_, new_meta}, /*dim=*/0).requires_grad_(true);
    subdiv_meta_.retain_grad();
    subdiv_p_    = torch::cat({subdiv_p_,  new_subdivP},/*dim=*/0).requires_grad_(true);
    subdiv_p_.retain_grad();

    // Extend the subdiv_p_grad_buffer_ accordingly
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
    float spatial_lr_scale)
{
    // 1) Store the scene extent for later use (e.g. in LR scheduling or densification)
    this->spatial_lr_scale_ = spatial_lr_scale;

    // 2) Number of input 3D points = number of voxels we will create
    const int64_t N = static_cast<int64_t>(pcd.size());
    if (N == 0) {
        // Leave all member‐tensors as their default empty values
        return;
    }

    // 3) Allocate a CPU tensor for centers and colors, then move to device
    //    We will fill them row‐by‐row from the ordered map.
    torch::TensorOptions opts = torch::TensorOptions()
                                    .dtype(torch::kFloat32)
                                    .device(device_type_);
    torch::Tensor centers_cpu = torch::empty({N, 3}, opts);
    torch::Tensor colors_cpu  = torch::empty({N, 3}, opts);

    {
        // Fill `centers_cpu` and `colors_cpu` from `pcd`
        auto it = pcd.begin();
        for (int64_t i = 0; i < N; ++i, ++it) {
            const Point3D& pt = it->second;
            centers_cpu.index_put_({i, 0}, pt.xyz_(0));
            centers_cpu.index_put_({i, 1}, pt.xyz_(1));
            centers_cpu.index_put_({i, 2}, pt.xyz_(2));
            // Color in [0,255], convert to float in [0,1]
            colors_cpu.index_put_({i, 0}, pt.color_(0) / 255.0f);
            colors_cpu.index_put_({i, 1}, pt.color_(1) / 255.0f);
            colors_cpu.index_put_({i, 2}, pt.color_(2) / 255.0f);
        }
    }

    // 4) Move centers and colors to the correct device
    center_ = centers_cpu.to(device_type_);
    // Each voxel default size: 0.05 m (same as in mapper). 
    // If you want to pull from VoxelModelParams instead, replace 0.05f with model_params_.initial_voxel_size
    float default_voxel_size = 0.05f;
    size_   = torch::full({N}, default_voxel_size, opts).to(device_type_);

    // 5) Initialize "geo" = 8‐component tensor per voxel (covariance + pad) to zero.
    geo_ = torch::zeros({N, 8}, opts).requires_grad_(true);

    // 6) Convert each color to spherical‐harmonic DC (sh_utils::RGB2SH returns [N, 3])
    //    Then build sh0_ = fused_color * Y₀
    torch::Tensor colors_device = colors_cpu.to(device_type_);
    torch::Tensor fused_sh       = sh_utils::RGB2SH(colors_device);
    // fused_sh has shape [N, 3]. We interpret that as the “SH‐DC” term.
    sh0_ = fused_sh.clone().requires_grad_(true);

    // 7) Initialize all higher‐degree SH coefficients to zero: [N, 45, 3] → we can zero them.
    shs_ = torch::zeros({N, 45, 3}, opts).requires_grad_(true);

    // 8) Initialize opacity to 0.8 for each voxel, then invert‐sigmoid so we can train in logit‐space:
    //    We want initial “probability” = 0.8. inverse_sigmoid(0.8) is a logit. 
    //    If you don't have inverse_sigmoid, you can simply store 0.8 directly and train in “probability” space,
    //    so long as your loss functions expect [0,1]. Here we follow GaussianModel’s pattern and do:
    torch::Tensor init_prob = torch::full({N, 1}, 0.8f, opts);
    // general_utils::inverse_sigmoid() expects shape [N,1] → returns logits
    torch::Tensor opacity_logits = general_utils::inverse_sigmoid(init_prob).view({N});
    opacity_ = opacity_logits.clone().requires_grad_(true);

    // 9) Build octree‐tracking tensors: path = [0..N-1], level = zeros
    oct_path_  = torch::arange(0, N, torch::TensorOptions().dtype(torch::kLong).device(device_type_));
    oct_level_ = torch::zeros({N}, torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

    // 10) exist_since_iter_ = zero for every newly‐created voxel
    exist_since_iter_ = torch::zeros(
        {N}, 
        torch::TensorOptions().dtype(torch::kInt32).device(device_type_)
    );

    // 11) subdiv_meta_ = zeros → this is a trainable “priority”‐value for each voxel
    subdiv_meta_ = torch::zeros({N}, opts).requires_grad_(true);
    subdiv_meta_.retain_grad();

    // 12) subdiv_p_ = zeros (same shape) → also trainable
    subdiv_p_ = torch::zeros({N}, opts).requires_grad_(true);
    subdiv_p_.retain_grad();

    // 13) Prepare the “gradient buffer” for subdividing later
    subdiv_p_grad_buffer_ = torch::zeros_like(subdiv_p_).to(device_type_);
}

//------------------------------------------------------------------------------
// Tensor‐based increasePcd overload (if SLAM supplies Tensors directly).
//------------------------------------------------------------------------------
void VoxelModel::increasePcd(torch::Tensor& new_centers,
                             torch::Tensor& new_colors,
                             const int /*iteration*/)
{
    const int64_t P = new_centers.size(0);
    if (P == 0) return;
    torch::NoGradGuard no_grad;
    auto device = center_.device();

    // Prepare the same per‐voxel attributes as above
    torch::Tensor new_sizes   = torch::full({P}, 0.05f, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor new_geo     = torch::zeros({P, 8}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor new_sh0     = new_colors.to(device);
    torch::Tensor new_shs     = torch::zeros({P, 45, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor new_opacity = torch::ones({P}, torch::TensorOptions().dtype(torch::kFloat32).device(device)) * 0.8f;
    torch::Tensor new_oct     = torch::arange(center_.size(0), center_.size(0) + P, torch::TensorOptions().dtype(torch::kLong).device(device));
    torch::Tensor new_lvl     = torch::zeros({P}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    torch::Tensor new_meta    = torch::zeros({P}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor new_subdivP = torch::zeros({P}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // Concatenate everything
    center_      = torch::cat({center_,   new_centers.to(device)},   /*dim=*/0).requires_grad_(false);
    size_        = torch::cat({size_,     new_sizes},               /*dim=*/0).requires_grad_(false);
    geo_         = torch::cat({geo_,      new_geo},                 /*dim=*/0).requires_grad_(true);
    sh0_         = torch::cat({sh0_,      new_sh0},                 /*dim=*/0).requires_grad_(true);
    shs_         = torch::cat({shs_,      new_shs},                 /*dim=*/0).requires_grad_(true);
    opacity_     = torch::cat({opacity_,  new_opacity},             /*dim=*/0).requires_grad_(true);
    oct_path_    = torch::cat({oct_path_, new_oct},                 /*dim=*/0);
    oct_level_   = torch::cat({oct_level_,new_lvl},                 /*dim=*/0);
    subdiv_meta_ = torch::cat({subdiv_meta_, new_meta},             /*dim=*/0).requires_grad_(true);
    subdiv_meta_.retain_grad();
    subdiv_p_    = torch::cat({subdiv_p_,  new_subdivP},            /*dim=*/0).requires_grad_(true);
    subdiv_p_.retain_grad();

    // Extend subdiv_p_grad_buffer_
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
// Apply a global scale + SE3 transform to ALL voxel centers (no gradient).
// Mirrors GaussianModel::applyScaledTransformation.
//------------------------------------------------------------------------------
void VoxelModel::applyScaledTransformation(const float s, const Sophus::SE3f& T)
{
    torch::NoGradGuard no_grad;
    // 1) Scale all centers
    center_ *= s;

    // 2) Build a (4×4) torch Tensor from T.matrix()
    torch::Tensor T_tensor = tensor_utils::EigenMatrix2TorchTensor(T.matrix(), device_type_).transpose(0, 1);

    // 3) Convert centers to homogeneous and apply SE3: [R,t] on each (no gradient)
    //    center_hom = (center_, 1).T  → warping
    auto N = center_.size(0);
    auto ones = torch::ones({N, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    auto centers_h = torch::cat({center_, ones}, /*dim=*/1);          // (N,4)
    auto centers_t = (T_tensor.matmul(centers_h.transpose(0,1))).transpose(0,1).index({torch::indexing::Slice(), torch::indexing::Slice(0,3)});
    center_ = centers_t.clone();  // (N,3)

    // 4) No need to touch size_, geo_, sh0_, shs_, opacity_ because they are intrinsic.
    //    Only subdiv_meta_ and subdiv_p_ remain the same.
}

void VoxelModel::scaledTransformVisiblePointsOfKeyframe(
    torch::Tensor&     point_not_transformed_flags,
    torch::Tensor&     diff_pose,                 // (4,4) homogeneous
    torch::Tensor&     kf_world_view_transform,   // (4,4)
    torch::Tensor&     kf_full_proj_transform,    // (4,4)
    const int          kf_creation_iter,
    const int          stable_num_iter_existence,
    int&               num_transformed,
    const float        scale)
{
    torch::NoGradGuard no_grad;
    // 1) Determine “unstable” voxels: those created too recently
    torch::Tensor unstable = torch::abs(exist_since_iter_ - kf_creation_iter) < stable_num_iter_existence;

    // 2) Compute visibility: rasterize only non‐transformed & non-unstable voxels against this keyframe
    //    (For simplicity, we assume SVRaster’s Python wrapper does the actual frustum check.)
    //    Here we just combine the flags:
    torch::Tensor to_check = (point_not_transformed_flags.logical_and(~unstable));

    if (!to_check.any().item<bool>()) {
        return;
    }

    // 3) Build a per-voxel “voxel_data” dict and do a single backward pass to find visible subset.
    //    However, Photo-SLAM’s C++ GaussianModel uses a custom C++ function “scaleAndTransformThenMarkVisiblePoints(...)”.
    //    We can cheat: call the Python wrapper with a special “visibility_only” mode to get a boolean mask H of size (N,) of visible voxels.
    //    Let’s assume a helper function exists:
    ///
    ///   torch::Tensor visible_mask = getVisibleVoxelMask(
    ///       to_check, kf_world_view_transform, kf_full_proj_transform);
    ///
    //    For now, we’ll pretend every “to_check” voxel is visible.
    torch::Tensor visible_mask = to_check;  

    // 4) Build a combined mask of “to_transform” = visible_mask
    torch::Tensor to_transform = visible_mask;  // (N,) bool

    if (!to_transform.any().item<bool>()) {
        return;
    }

    // 5) Apply diff_pose to those voxel centers: center[to_transform] = diff_pose * [center;1]
    auto idx = to_transform.nonzero().view(-1);  // indices of voxels to update
    num_transformed += idx.numel();

    auto centers_sel = center_.index({idx});  // (M,3)
    auto ones = torch::ones({(int)idx.numel(), 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    auto centers_h = torch::cat({centers_sel, ones}, /*dim=*/1);      // (M,4)
    auto updated = (diff_pose.matmul(centers_h.transpose(0,1))).transpose(0,1).index({torch::indexing::Slice(), torch::indexing::Slice(0,3)});
    center_.index_put_({idx}, updated);

    // 6) Scale (if needed) is baked into diff_pose’s translation; we assume diff_pose already has scale on R,t.

    // 7) Mark those voxels as “transformed”
    point_not_transformed_flags.index_put_({idx}, false);

    // 8) No need to update geo_, sh0_, shs_, opacity_ here.
    //    Only “center_” and, if we had scale‐dependent fields, we’d update them.

    // NOTE: since this is NoGrad, we do not rewire optimizer. We assume positions are not learnable after loop closure.
}

//------------------------------------------------------------------------------
// Prepare optimizer: mirror GaussianModel::trainingSetup.
//   - We create an Adam optimizer with three param groups: [geo_], [sh0_], [shs_].
//   - Adjust their initial LRs from `opt_params`.
//------------------------------------------------------------------------------
void VoxelModel::trainingSetup(const VoxelOptimizationParams& opt)
{
    std::lock_guard<std::mutex> lk(mutex_settings_);

    torch::optim::AdamOptions adam_opts(0.0);
    adam_opts.eps(1e-15);
    optimizer_.reset(new torch::optim::Adam({geo_}, adam_opts));

    // 1) geo‐group
    optimizer_->param_groups()[0].options().set_lr(opt.geo_lr_init_);

    // 2) sh0‐group
    optimizer_->add_param_group(torch::optim::OptimizerParamGroup({sh0_}));
    optimizer_->param_groups()[1].options().set_lr(opt.sh0_lr_);

    // 3) shs‐group
    optimizer_->add_param_group(torch::optim::OptimizerParamGroup({shs_}));
    optimizer_->param_groups()[2].options().set_lr(opt.shs_lr_);

    // Save schedule parameters
    geo_lr_init_      = opt.geo_lr_init_;
    geo_lr_final_     = opt.geo_lr_final_;
    lr_delay_mult_    = opt.geo_lr_delay_mult_;
    max_steps_        = opt.geo_lr_max_steps_;
    // If desired, we could also add opacity_ as group[3], subdiv_p_ as group[4], etc.
}

//------------------------------------------------------------------------------
// Exponential learning‐rate schedule (mirrors GaussianModel::exponLrFunc).
//------------------------------------------------------------------------------
float VoxelModel::exponLrFunc(int step) 
{
    if (step < 0 || (geo_lr_init_ == 0.0f && geo_lr_final_ == 0.0f)) {
        return 0.0f;
    }
    float delay_rate;
    if (lr_delay_steps_ > 0) 
        delay_rate = lr_delay_mult_ + (1.0f - lr_delay_mult_) * std::sin(M_PI_2f32 * std::clamp(static_cast<float>(step) / lr_delay_steps_, 0.0f, 1.0f));
    else
        delay_rate = 1.0f;
    float t = std::clamp(static_cast<float>(step) / max_steps_, 0.0f, 1.0f);
    float log_lerp = std::exp(std::log(geo_lr_init_) * (1.0f - t) + std::log(geo_lr_final_) * t);
    return delay_rate * log_lerp;
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
    size_        = size_.index({mk});
    geo_         = geo_.index({mk}).requires_grad_(true);
    sh0_         = sh0_.index({mk}).requires_grad_(true);
    shs_         = shs_.index({mk}).requires_grad_(true);
    opacity_     = opacity_.index({mk}).requires_grad_(true);
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
    torch::Tensor opacity_k     = keep_t(opacity_);
    torch::Tensor oct_k         = keep_t(oct_path_);
    torch::Tensor lvl_k         = keep_t(oct_level_);
    torch::Tensor meta_k        = keep_t(subdiv_meta_);

    // 4) Gather parent attributes
    torch::Tensor c_par         = gather(center_);       // (P,3)
    torch::Tensor s_par         = gather(size_);         // (P)
    torch::Tensor geo_par       = gather(geo_);          // (P,8)
    torch::Tensor sh0_par       = gather(sh0_);          // (P,3)
    torch::Tensor shs_par       = gather(shs_);          // (P,45,3)
    torch::Tensor op_par        = gather(opacity_);      // (P)
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
    torch::Tensor sh0_child   = repeat8(sh0_par);        // (8P,3)
    torch::Tensor shs_child   = repeat8(shs_par);        // (8P,45,3)
    torch::Tensor op_child    = repeat8(op_par);         // (8P)

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
    opacity_     = torch::cat({opacity_k,    op_child},     /*dim=*/0).requires_grad_(true);
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
    if (auto g = subdiv_meta_.grad(); g.defined()) {
        g.detach_();
    }
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
        subdiv_p_grad_buffer_ = torch::zeros_like(subdiv_p_);
    }
    TORCH_CHECK(subdiv_p_grad_buffer_.sizes() == subdiv_p_.sizes(),
                "accumulateSubdivGradients: size mismatch");
    subdiv_p_grad_buffer_.scatter_add_(
        /*dim=*/0,
        /*index=*/parent_idx.to(device_type_),
        /*src=*/parent_grads.to(device_type_)
    );
}

//------------------------------------------------------------------------------
// Return all optimizable parameters for the outer training loop.
// Mirrors GaussianModel::parameters() → returns {geo_, sh0_, shs_}.
//------------------------------------------------------------------------------
std::vector<torch::Tensor> VoxelModel::parameters() const
{
    return { geo_, sh0_, shs_ };
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
    else if (name == "opacity") return opacity_;
    else if (name == "octpath") return oct_path_;
    else if (name == "octlevel")return oct_level_;
    else if (name == "subdiv_meta") return subdiv_meta_;
    else if (name == "subdiv_p")    return subdiv_p_;
    else throw std::runtime_error("VoxelModel::getTensor(): unknown name \"" + name + "\"");
}

//------------------------------------------------------------------------------
// Save all voxel tensors to a Torch file (CPU) for checkpointing.
//------------------------------------------------------------------------------
void VoxelModel::saveTorch(const std::filesystem::path& p) const
{
    std::vector<torch::Tensor> pack = {
        center_.cpu(),
        size_.cpu(),
        geo_.cpu(),
        sh0_.cpu(),
        shs_.cpu(),
        opacity_.cpu(),
        oct_path_.cpu(),
        oct_level_.cpu(),
        subdiv_meta_.cpu(),
        subdiv_p_.cpu()
    };
    torch::save(pack, p.string());
}

std::unordered_map<std::string, torch::Tensor>
// VoxelModel::render(
//     const VoxelKeyframe& kf,
//     const VoxelPipelineParams& /*pipeParams*/,      // pass through if needed
//     const torch::Tensor& background,
//     const torch::Tensor& override_color,
//     const std::string& output_dir
// ) const
// {
//     // Exactly the same logic you had in old VoxelTrainer::render,
//     // except now use `this->center_`, `this->geo_`, `this->sh0_`, etc.,
//     // instead of the trainer's member variables.

//     py::gil_scoped_acquire gil;
//     static py::object py_render = py::module_::import(
//         "scripts_voxel.python_svraster_bridge.renderer_wrapper")
//         .attr("render");

//     // If no voxels, skip:
//     if (center_.numel() == 0) {
//         std::cout << "[INFO] Skipping render — empty voxel data\n";
//         return {};
//     }
//     // Sanity check:
//     if (!subdiv_p_.defined() || !subdiv_p_.isfinite().all().item<bool>()) {
//         std::cerr << "[ERROR] subdiv_p_ contains NaNs or is not finite.\n";
//         std::cerr << " - numel: " << subdiv_p_.numel() << "\n";
//         throw std::runtime_error("Invalid subdiv_p_ before rendering.");
//     }

//     // Pack all voxel tensors into a Python dict:
//     py::dict dict;
//     dict["geos"]       = py::cast(geo_);               // (N, 8)
//     dict["colors"]     = py::cast(sh0_);              // (N, 3)
//     dict["shs"]        = py::cast(shs_);              // (N, 45, 3)
//     dict["opacities"]  = py::cast(opacity_);          // (N,)
//     dict["subdiv_p"]   = py::cast(subdiv_p_);         // (N,)
//     dict["octpaths"]   = py::cast(oct_path_.cpu());  
//     dict["centers"]    = py::cast(center_.cpu());
//     dict["vox_lengths"]= py::cast(size_.cpu());
//     dict["octlevels"]  = py::cast(oct_level_.cpu());
//     dict["subdiv_meta"]= py::cast(subdiv_meta_.cpu());

//     // Build a Python MiniCam from VoxelKeyframe:
//     py::object py_cam = MiniCam_to_py(kf.toMiniCam());

//     // Call the Python bridge and get back a Python dict (or map):
//     py::object out = py_render(py_cam, dict, /*rgb_image=*/nullptr, output_dir);

//     // The old wrapper returned only “rgb”, but if you now return more fields
//     // (e.g. viewspace pts, visibility_mask, radii), unpack them here.
//     torch::Tensor rgb_t = out.attr("get")("rgb").cast<torch::Tensor>();

//     return { { "rgb", rgb_t }
//            /*, { "viewspace", viewspace_t },
//               { "vis_mask", vis_mask },
//               { "radii", radii_t } */
//            };
// }

VoxelModel::render(
    const MiniCam&                   cam,
    const py::array_t<uint8_t>&      rgb_image,
    const std::string&               output_dir
) const {
  // 1) Acquire GIL and import the Python function once
  py::gil_scoped_acquire gil;
  static py::object py_render = py::module_::import(
    "scripts_voxel.python_svraster_bridge.renderer_wrapper")
      .attr("render");

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
  d["geos"]        = py::cast(geo_);
  d["colors"]      = py::cast(sh0_);
  d["shs"]         = py::cast(shs_);
  d["opacities"]   = py::cast(opacity_);
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

} // namespace sv
