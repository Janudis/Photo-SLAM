#include "include_voxel/voxel_model.h"
#include <fstream>
#include <sstream>

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
    VOXEL_MODEL_INIT_TENSORS(this->device_type_);
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
    VOXEL_MODEL_INIT_TENSORS(this->device_type_);
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

    /* constants (same as before) */
    constexpr float kPadding     = 0.05f;
    constexpr float kTargetVoxel = 0.05f;
    constexpr float kGeoInit     = 4.0f;
    constexpr float kSh0Init     = 0.5f;
    constexpr float kShsInit     = 0.0f;

    this->spatial_lr_scale_ = spatial_lr_scale;

    /* 1) copy PCD → CUDA (unchanged) */
    const int N_pts = static_cast<int>(pcd.size());
    torch::Tensor pts  = torch::zeros({N_pts,3}, torch::kFloat32).to(device_type_);
    torch::Tensor cols = torch::zeros({N_pts,3}, torch::kFloat32).to(device_type_);
    {
        int i = 0;
        for (const auto& kv : pcd) {
            const Point3D& p = kv.second;
            pts[i]  = torch::tensor({p.xyz_(0), p.xyz_(1), p.xyz_(2)},
                                    torch::kFloat32).to(device_type_);
            cols[i] = torch::tensor({p.color_(0), p.color_(1), p.color_(2)},
                                    torch::kFloat32).to(device_type_);
            ++i;
        }
    }

    /* 2) scene bounds & extent (unchanged) */
    auto mn = std::get<0>(pts.min(0));
    auto mx = std::get<0>(pts.max(0));
    float half_side     = (mx - mn).max().item<float>();
    float inside_extent = 2.f * half_side + kPadding;
    float scene_extent  = inside_extent * std::pow(2.f, float(outside_level_));
    torch::Tensor scene_center = 0.5f * (mn + mx);
    torch::Tensor extent       = torch::tensor({scene_extent},
                                 torch::TensorOptions().dtype(torch::kFloat32)
                                                        .device(device_type_));

    /* 3) choose leaf level from target voxel size (unchanged) */
    int leaf_level = std::ceil(std::log2(scene_extent / kTargetVoxel));
    torch::Tensor octlevel0 = torch::full({N_pts,1}, leaf_level,
        torch::TensorOptions().dtype(torch::kInt8).device(device_type_));

    /* 4) xyz → octpath (unchanged) */
    py::gil_scoped_acquire gil;
    static py::module octree = []{
        py::module sys = py::module::import("sys");
        sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
        return py::module::import("src.utils.octree_utils");
    }();
    torch::Tensor octpath0 = octree.attr("xyz_2_octpath")(
                                 pts, octlevel0, scene_center, extent
                             ).cast<torch::Tensor>().to(device_type_);

    /* 5) collapse duplicates (unchanged) */
    torch::Tensor keys = torch::cat({octpath0.to(torch::kLong).view({-1,1}),
                                     octlevel0.to(torch::kLong)}, 1);
    auto uq = at::unique_dim(keys.cpu(), 0, /*sorted=*/true,
                             /*return_inverse=*/true, /*return_counts=*/true);
    torch::Tensor uniq_keys = std::get<0>(uq).to(device_type_, torch::kLong);
    torch::Tensor inv_id    = std::get<1>(uq).to(device_type_, torch::kLong);
    torch::Tensor counts    = std::get<2>(uq).to(torch::kFloat32)
                                             .unsqueeze(1).to(device_type_);
    const int64_t M = uniq_keys.size(0);

    torch::Tensor rgb_sum = torch::zeros({M,3}, cols.options());
    rgb_sum.index_add_(0, inv_id, cols);
    torch::Tensor rgb_avg = (rgb_sum / counts).detach();

    torch::Tensor up_octpath  = uniq_keys.select(1,0).view({-1,1});
    torch::Tensor up_octlevel = uniq_keys.select(1,1)
                                       .to(torch::kInt8).view({-1,1});

    /* 6) ▶ CHANGED: decode octpath (GPU tensors, no .cpu()) */
    auto dec = octree.attr("octpath_decoding")(
                   up_octpath, up_octlevel,     // stay on CUDA
                   scene_center, extent         // already CUDA
               ).cast<std::tuple<torch::Tensor,torch::Tensor>>();
    torch::Tensor vox_centers = std::get<0>(dec).to(device_type_);
    torch::Tensor vox_size    = std::get<1>(dec).squeeze(1).to(device_type_);

    py::module_ np = py::module_::import("numpy");
    // save raw point‐cloud + colors
    np.attr("save")("/home/dimitris/Photo-SLAM/debug_pts.npy",
                    tensor_to_numpy(pts.cpu()));
    np.attr("save")("/home/dimitris/Photo-SLAM/debug_cols.npy",
                    tensor_to_numpy(cols.cpu()));
    // save voxel cell centres + edge lengths
    np.attr("save")("/home/dimitris/Photo-SLAM/debug_vox_centers.npy",
                    tensor_to_numpy(vox_centers.cpu()));
    np.attr("save")("/home/dimitris/Photo-SLAM/debug_vox_size.npy",
                    tensor_to_numpy(vox_size.cpu()));

    /* 7) save basics (unchanged) */
    center_       = vox_centers;
    size_         = vox_size;
    vox_size_inv_ = 1.0f / size_;
    oct_path_     = up_octpath.to(torch::kLong);
    oct_level_ = up_octlevel.to(torch::kInt8)         // <<< int8!
                            .view({M}).clone();      // keep contiguous

    /* 8) ▶ CHANGED: grid-point links (inputs stay on CUDA) */
    py::tuple link = octree.attr("build_grid_pts_link")(up_octpath, up_octlevel);
    grid_pts_key_  = link[0].cast<torch::Tensor>().to(device_type_);
    vox_key_       = link[1].cast<torch::Tensor>().to(device_type_);
    int64_t G      = grid_pts_key_.size(0);

    /* 9) allocate learnables (unchanged) */
    _geo_grid_pts = torch::full({G,1}, kGeoInit,
                                torch::kFloat32).to(device_type_)
                     .requires_grad_(true);

    torch::Tensor sh0_rgb = torch::full({M,3}, kSh0Init,
                                        torch::kFloat32).to(device_type_);
    sh0_ = sh_utils::RGB2SH(sh0_rgb).view({M,1,3}).requires_grad_(true);

    int K = (max_sh_degree_+1)*(max_sh_degree_+1) - 1;
    shs_ = torch::full({M,K,3}, kShsInit,
                       torch::kFloat32).to(device_type_)
           .requires_grad_(true);

    subdiv_p_            = torch::ones({M,1},
                                       torch::kFloat32).to(device_type_)
                             .requires_grad_(true);
    subdiv_meta_         = torch::zeros({M,1},
                                        torch::kFloat32).to(device_type_)
                              .requires_grad_(true);

    // subdiv_p_grad_buffer_= torch::zeros_like(subdiv_p_);

    subdiv_meta_.retain_grad();
    subdiv_p_   .retain_grad();

    // ─── **new**: allocate our two stats‐buffers on the same device ─────────
    xyz_gradient_accum_ = torch::zeros({M,1},
        torch::TensorOptions()
          .dtype(torch::kFloat32)
          .device(device_type_));
    denom_ = torch::zeros_like(xyz_gradient_accum_);

    voxel_error_sum_  = torch::zeros({M,1},
                                 torch::TensorOptions()
                                     .dtype(torch::kFloat32)
                                     .device(device_type_));
    voxel_hit_count_  = torch::zeros_like(voxel_error_sum_);

    /* 10) register tensors with optimizer (unchanged) */
    VOXEL_MODEL_TENSORS_TO_VEC
}

// void VoxelModel::accumulateError(const torch::Tensor& vis_idx,
//                                  const torch::Tensor& err)
// {
//     TORCH_CHECK(vis_idx.dtype()==torch::kLong && vis_idx.dim()==1);
//     TORCH_CHECK(err.dim()==1 && err.numel()==vis_idx.numel());

//     // make sure tensors are 1-D float32 on the right device
//     auto idx = vis_idx.to(device_type_);
//     auto e   = err.to(device_type_, /*non_blocking=*/true);

//     // grow stat buffers if we just subdivided
//     if (voxel_error_sum_.size(0) != center_.size(0))
//     {
//         int64_t M = center_.size(0);
//         voxel_error_sum_ = torch::zeros({M,1}, voxel_error_sum_.options());
//         voxel_hit_count_ = torch::zeros_like(voxel_error_sum_);
//     }

//     voxel_error_sum_.index_add_(0, idx, e.unsqueeze(1));    // Σ err
//     voxel_hit_count_.index_add_(0, idx,
//                                 torch::ones_like(e).unsqueeze(1)); // +1
// }

// torch::Tensor VoxelModel::errorNormalized() const
// {
//     torch::Tensor avg = voxel_error_sum_ / (voxel_hit_count_ + 1e-6f);
//     float mn = avg.min().item<float>(),  mx = avg.max().item<float>();
//     return (avg - mn) / (mx - mn + 1e-6f);   // ∈[0,1]
// }

// void VoxelModel::createFromPcd(
//     const std::map<point3D_id_t, Point3D>& pcd,
//     float spatial_lr_scale)
// {
//     constexpr float kPadding     = 0.05f;
//     constexpr float kTargetVoxel = 0.1f;
//     this->spatial_lr_scale_ = spatial_lr_scale;

//     // 1) Compute scene bounds
//     const int N_pts = static_cast<int>(pcd.size());
//     torch::Tensor pts = torch::zeros({N_pts,3}, torch::kFloat32).to(device_type_);
//     torch::Tensor cols = torch::zeros({N_pts,3}, torch::kFloat32).to(device_type_);
//     {
//         int i = 0;
//         for (const auto& kv : pcd) {
//             const auto& p = kv.second;
//             pts[i] = torch::tensor({p.xyz_(0), p.xyz_(1), p.xyz_(2)}, torch::kFloat32).to(device_type_);
//             cols[i] = torch::tensor({p.color_(0), p.color_(1), p.color_(2)},
//                         torch::kFloat32).to(device_type_);
//             ++i;
//         }
//     }
//     auto mn = std::get<0>(pts.min(0));
//     auto mx = std::get<0>(pts.max(0));
    
//     // 2) Scene center and extent
//     auto half_side = (mx - mn).max().item<float>() * 0.5f;
//     auto padded_half = half_side + kPadding;
//     // scene_extent spans both inside and outside levels
//     torch::Tensor scene_center = 0.5f * (mn + mx);
//     torch::Tensor scene_extent = torch::tensor({2.f * padded_half * std::pow(2.f, float(outside_level_))},
//                                                torch::TensorOptions().dtype(torch::kFloat32)
//                                                                               .device(device_type_));

//     // 3) Determine leaf level for uniform grid
//     int leaf_level = std::ceil(std::log2(scene_extent.item<float>() / kTargetVoxel));

//     // 4) Generate dense octree paths at that level
//     py::gil_scoped_acquire gil;
//     static py::module octree_utils = []{
//         py::module sys = py::module::import("sys");
//         sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
//         return py::module::import("src.utils.octree_utils");
//     }();
//     py::tuple dense = octree_utils.attr("gen_octpath_dense")(outside_level_, leaf_level);
//     torch::Tensor octpath = dense[0].cast<torch::Tensor>().to(device_type_);
//     torch::Tensor octlevel = dense[1].cast<torch::Tensor>().to(device_type_);

//     // 5) Decode uniform grid to get voxel centers and sizes
//     auto dec = octree_utils.attr("octpath_decoding")(octpath, octlevel,
//                                                        scene_center, scene_extent)
//                    .cast<std::tuple<torch::Tensor, torch::Tensor>>();
//     torch::Tensor vox_centers = std::get<0>(dec);
//     torch::Tensor vox_size    = std::get<1>(dec).view({-1,1});

//     // // 6) Save debug if needed
//     // py::module_ np = py::module_::import("numpy");
//     // // save raw point‐cloud + colors
//     // np.attr("save")("/home/dimitris/Photo-SLAM/debug_pts.npy",
//     //                 tensor_to_numpy(pts.cpu()));
//     // np.attr("save")("/home/dimitris/Photo-SLAM/debug_cols.npy",
//     //                 tensor_to_numpy(cols.cpu()));
//     // // save voxel cell centres + edge lengths
//     // np.attr("save")("/home/dimitris/Photo-SLAM/debug_vox_centers.npy",
//     //                 tensor_to_numpy(vox_centers.cpu()));
//     // np.attr("save")("/home/dimitris/Photo-SLAM/debug_vox_size.npy",
//     //                 tensor_to_numpy(vox_size.cpu()));

//     // 7) Store basics
//     center_   = vox_centers;
//     size_     = vox_size;
//     vox_size_inv_ = 1.0f / vox_size;
//     oct_path_  = octpath.to(torch::kLong);
//     oct_level_ = octlevel.to(torch::kInt8).view({-1}).clone();

//     // 8) Build grid-point links
//     py::tuple link = octree_utils.attr("build_grid_pts_link")(octpath, octlevel);
//     grid_pts_key_ = link[0].cast<torch::Tensor>().to(device_type_);
//     vox_key_      = link[1].cast<torch::Tensor>().to(device_type_);
//     int64_t G     = grid_pts_key_.size(0);

//     // 9) Allocate learnables uniformly
//     _geo_grid_pts = torch::full({G,1}, /*geo init*/ 4.0f,
//                                 torch::kFloat32).to(device_type_).requires_grad_(true);
//     int64_t N = octpath.size(0);
//     sh0_  = torch::zeros({N,1,3}, torch::kFloat32).to(device_type_).requires_grad_(true);
//     shs_  = torch::zeros({N,(max_sh_degree_+1)*(max_sh_degree_+1)-1,3},
//                           torch::kFloat32).to(device_type_).requires_grad_(true);
//     subdiv_p_    = torch::ones({N,1}, torch::kFloat32).to(device_type_).requires_grad_(true);
//     subdiv_meta_ = torch::zeros({N,1}, torch::kFloat32).to(device_type_).requires_grad_(true);
//     subdiv_meta_.retain_grad(); subdiv_p_.retain_grad();

//     xyz_gradient_accum_ = torch::zeros({N,1}, torch::kFloat32).to(device_type_);
//     denom_              = torch::zeros_like(xyz_gradient_accum_);

//     // 10) Register with optimizer
//     VOXEL_MODEL_TENSORS_TO_VEC
// }

// void VoxelModel::createFromPcd(
//     const std::map<point3D_id_t, Point3D>& pcd,
//     float spatial_lr_scale)
// {
//     /* ── 1. AABB from point-cloud ───────────────────────────────────────── */
//     std::array<float,3> mn , mx ;
//     bool first = true;
//     for (auto const& kv : pcd) {
//         const auto& p = kv.second;
//         std::array<float,3> pt = { float(p.xyz_(0)),
//                                    float(p.xyz_(1)),
//                                    float(p.xyz_(2)) };
//         if (first) { mn = mx = pt; first = false; }
//         else {
//             for (int i = 0; i < 3; ++i) {
//                 mn[i] = std::min(mn[i], pt[i]);
//                 mx[i] = std::max(mx[i], pt[i]);
//             }
//         }
//     }
//     std::vector<std::array<float,3>> bounding = { mn, mx };

//     /* ── 2. Python: SVRaster constructor + config.init ──────────────────── */
//     py::gil_scoped_acquire gil;

//     static py::module cons_mod = []{
//         py::module sys = py::module::import("sys");
//         sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
//         return py::module::import("src.sparse_voxel_gears.constructor");
//     }();
//     py::object svc = cons_mod.attr("SVConstructor")();

//     py::module cfg_mod  = py::module::import("src.config");
//     py::object cfg_init = cfg_mod.attr("cfg").attr("init");

//     /* ── 3. Build inside-uniform + outside shells exactly like SVRaster ─── */
//     svc.attr("model_init")(bounding, cfg_init /* <- second arg required */);

//     /* ── 4. Pull back CUDA tensors ──────────────────────────────────────── */
//     center_       = svc.attr("vox_center").cast<torch::Tensor>();
//     size_         = svc.attr("vox_size").cast<torch::Tensor>().view(-1);
//     vox_size_inv_ = 1.0f / size_;

//     oct_path_     = svc.attr("octpath").cast<torch::Tensor>().to(torch::kLong);
//     oct_level_    = svc.attr("octlevel").cast<torch::Tensor>()
//                                          .to(torch::kInt8).view(-1);

//     grid_pts_key_ = svc.attr("grid_pts_key").cast<torch::Tensor>().to(torch::kLong);
//     vox_key_      = svc.attr("vox_key").cast<torch::Tensor>().to(torch::kLong);

//     /* ── 5. Learnable buffers provided by SVRaster ──────────────────────── */
//     _geo_grid_pts = svc.attr("_geo_grid_pts").cast<torch::Tensor>();
//     sh0_          = svc.attr("_sh0").cast<torch::Tensor>();
//     shs_          = svc.attr("_shs").cast<torch::Tensor>();
//     subdiv_p_     = svc.attr("_subdiv_p").cast<torch::Tensor>();
//     subdiv_meta_  = svc.attr("subdiv_meta").cast<torch::Tensor>();

//     subdiv_meta_.retain_grad();
//     subdiv_p_.retain_grad();

//     /* ── 6. Stats buffers (zeros) ───────────────────────────────────────── */
//     xyz_gradient_accum_ = torch::zeros_like(_geo_grid_pts);
//     denom_              = torch::zeros_like(_geo_grid_pts);

//     this->spatial_lr_scale_ = spatial_lr_scale;

//     /* ── 7. Register with optimizer ─────────────────────────────────────── */
//     VOXEL_MODEL_TENSORS_TO_VEC
// }

// void VoxelModel::increasePcd(torch::Tensor& new_point_cloud, torch::Tensor& new_colors, const int iteration)
// {
//     auto num_new_points = new_point_cloud.size(0);
//     if (num_new_points == 0)
//         return;

//     if (sparse_points_xyz_.size(0) == 0) {
//         sparse_points_xyz_ = new_point_cloud;
//         sparse_points_color_ = new_colors;
//     }
//     else {
//         sparse_points_xyz_ = torch::cat({sparse_points_xyz_, new_point_cloud}, /*dim=*/0);
//         sparse_points_color_ = torch::cat({sparse_points_color_, new_colors}, /*dim=*/0);
//     }

//     /* ------------------------------------------------------------------ 1 : SH colours */
//     torch::Tensor new_fused_colors = sh_utils::RGB2SH(new_colors);
//     auto temp = this->max_sh_degree_ + 1;
//     torch::Tensor features = torch::zeros(
//         {new_fused_colors.size(0), 3, temp * temp},
//         torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
//     features.index(
//         {torch::indexing::Slice(),
//          torch::indexing::Slice(0, 3),
//          0}) = new_fused_colors;
//     features.index(
//         {torch::indexing::Slice(),
//          torch::indexing::Slice(3, features.size(1)),
//          torch::indexing::Slice(1, features.size(2))}) = 0.0f;

//     /* ------------------------------------------------------------------ 2 : default voxel attributes */
//     const float default_vox_size = 0.05f;
//     torch::Tensor new_size = torch::full(
//         {num_new_points}, default_vox_size,
//         torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

//     torch::Tensor new_geo = torch::zeros(
//         {num_new_points, 8}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
//     const float var = 0.25f * default_vox_size * default_vox_size;
//     new_geo.index_put_({torch::indexing::Slice(), 0}, var);
//     new_geo.index_put_({torch::indexing::Slice(), 3}, var);
//     new_geo.index_put_({torch::indexing::Slice(), 5}, var);

//     torch::Tensor new_sh0 = features.index({
//         torch::indexing::Slice(),
//         torch::indexing::Slice(),
//         torch::indexing::Slice(0,1)})
//         .transpose(1,2).contiguous();                  // (P , 1 , 3) → (P ,3)

//     torch::Tensor new_shs = features.index({
//         torch::indexing::Slice(),
//         torch::indexing::Slice(),
//         torch::indexing::Slice(1, features.size(2))})
//         .transpose(1,2).contiguous();                  // (P ,  (M^2-1) , 3)

//     /* opacity logits at p = 0.8 */
//     // torch::Tensor op_init = general_utils::inverse_sigmoid(
//     //     0.8f * torch::ones({num_new_points,1},
//     //         torch::TensorOptions().dtype(torch::kFloat32).device(device_type_)));
//     // torch::Tensor new_opacity = op_init.view({num_new_points});

//     /* ------------------------------------------------------------------ 3 : book-keeping tensors */
//     torch::Tensor new_oct   = torch::arange(
//         center_.size(0), center_.size(0)+num_new_points,
//         torch::TensorOptions().dtype(torch::kLong).device(device_type_));

//     torch::Tensor new_lvl   = torch::zeros({num_new_points},
//         torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

//     torch::Tensor new_meta  = torch::zeros({num_new_points},
//         torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
//     torch::Tensor new_subP  = torch::zeros_like(new_meta);

//     torch::Tensor new_exist = torch::full(
//         {num_new_points}, iteration,
//         torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

//     /* ------------------------------------------------------------------ 4 : concatenate into the model  */
//     center_      = torch::cat({center_,   new_point_cloud     }, 0);          // (no-grad)
//     size_        = torch::cat({size_,     new_size            }, 0);
//     geo_         = torch::cat({geo_,      new_geo             }, 0).requires_grad_(true);
//     sh0_         = torch::cat({sh0_,      new_sh0             }, 0).requires_grad_(true);
//     shs_         = torch::cat({shs_,      new_shs             }, 0).requires_grad_(true);
//     // opacity_     = torch::cat({opacity_,  new_opacity         }, 0).requires_grad_(true);
//     oct_path_    = torch::cat({oct_path_, new_oct             }, 0);
//     oct_level_   = torch::cat({oct_level_,new_lvl             }, 0);
//     exist_since_iter_ = torch::cat({exist_since_iter_, new_exist}, 0);
//     subdiv_meta_ = torch::cat({subdiv_meta_, new_meta }, 0).requires_grad_(true);
//     subdiv_p_    = torch::cat({subdiv_p_,    new_subP }, 0).requires_grad_(true);
//     subdiv_meta_.retain_grad();
//     subdiv_p_.retain_grad();

//     /* ------------------------------------------------------------------ 5 : keep grad-buffer sized */
//     if (subdiv_p_grad_buffer_.numel() != subdiv_p_.numel())
//         subdiv_p_grad_buffer_ = torch::zeros_like(subdiv_p_);

// }

// void VoxelModel::increasePcd(std::vector<float> points,
//                              std::vector<float> colors,
//                              const int iteration)
// {
//     /* ------------------------------------------------------------------ 0 : basic sanity */
//     assert(points.size() == colors.size());
//     assert(points.size() % 3 == 0);
//     const int64_t P = static_cast<int64_t>(points.size() / 3);
//     if (P == 0) return;

//     /* ------------------------------------------------------------------ 1 : vectors → CUDA tensors */
//     torch::Tensor new_xyz = torch::from_blob(points.data(),  {P,3},
//                               torch::dtype(torch::kFloat32)).to(device_type_);
//     torch::Tensor new_rgb = torch::from_blob(colors.data(),  {P,3},
//                               torch::dtype(torch::kFloat32)).to(device_type_);

//     /* keep a CPU-copy for inspection / debugging (optional) */
//     if (sparse_points_xyz_.numel() == 0) {
//         sparse_points_xyz_   = new_xyz;
//         sparse_points_color_ = new_rgb;
//     } else {
//         sparse_points_xyz_   = torch::cat({sparse_points_xyz_,   new_xyz}, 0);
//         sparse_points_color_ = torch::cat({sparse_points_color_, new_rgb}, 0);
//     }

//     /* ------------------------------------------------------------------ 2 : RGB → SH coefficients (degree ≤ max_sh_degree_) */
//     torch::Tensor sh0_dc = sh_utils::RGB2SH(new_rgb);           // (P,3)
//     const int Lplus1 = max_sh_degree_ + 1;                      // (L+1)^2 coeffs per channel
//     const int nCoeffs = Lplus1 * Lplus1 - 1;                    // excl. DC

//     torch::Tensor new_sh0 = sh0_dc.view({P,1,3})                // (P,1,3)    ← DC term only
//                                    .contiguous();               // (requires_grad_ set later)

//     torch::Tensor new_shs = torch::zeros({P, nCoeffs, 3},       // higher-order SH (start @ 0)
//                                    sh0_dc.options());           // ^ same dtype/device

//     /* ------------------------------------------------------------------ 3 : voxel geometry / density */
//     const float rho0 = 1.0f;                                    // initial density value
//     const float logd = std::log1p(rho0);                        // ρ̂  (because ρ = exp(ρ̂) – 1)
//     torch::Tensor new_geo = torch::full({P, 8}, logd,           // **uniform log-density**
//                                 torch::TensorOptions()
//                                 .dtype(torch::kFloat32)
//                                 .device(device_type_))              // (P,8)
//                                .requires_grad_();               // learnable

//     const float vox_len = 0.05f;                                // edge-length
//     torch::Tensor new_size = torch::full({P}, vox_len,
//                                sh0_dc.options());

//     /* ------------------------------------------------------------------ 4 : bookkeeping tensors */
//     torch::Tensor new_oct  = torch::arange(center_.size(0), center_.size(0)+P,
//                                torch::dtype(torch::kInt64).device(device_type_));
//     torch::Tensor new_lvl  = torch::zeros({P},
//                                torch::dtype(torch::kInt32).device(device_type_));
//     torch::Tensor new_meta = torch::zeros({P},
//                                torch::dtype(torch::kFloat32).device(device_type_));
//     torch::Tensor new_subP = torch::zeros_like(new_meta);
//     torch::Tensor new_exist= torch::full({P}, iteration,
//                                torch::dtype(torch::kInt32).device(device_type_));

//     /* ------------------------------------------------------------------ 5 : concatenate into model buffers */
//     center_   = torch::cat({center_,   new_xyz  }, 0);          // (no grad)
//     size_     = torch::cat({size_,     new_size }, 0);
//     geo_      = torch::cat({geo_,      new_geo  }, 0);
//     sh0_      = torch::cat({sh0_,      new_sh0  }, 0);
//     shs_      = torch::cat({shs_,      new_shs  }, 0);
//     oct_path_ = torch::cat({oct_path_, new_oct  }, 0);
//     oct_level_= torch::cat({oct_level_,new_lvl  }, 0);
//     subdiv_meta_ = torch::cat({subdiv_meta_, new_meta}, 0);
//     subdiv_p_    = torch::cat({subdiv_p_,    new_subP}, 0);

//     /* enable gradients where needed */
//     geo_.requires_grad_(true);
//     sh0_.requires_grad_(true);
//     shs_.requires_grad_(true);
//     subdiv_meta_.requires_grad_(true).retain_grad();
//     subdiv_p_.requires_grad_(true).retain_grad();

//     /* ------------------------------------------------------------------ 6 : keep grad buffer sized */
//     if (subdiv_p_grad_buffer_.numel() != subdiv_p_.numel())
//         subdiv_p_grad_buffer_ = torch::zeros_like(subdiv_p_);
// }

// torch::Tensor VoxelModel::replaceTensorToOptimizer(torch::Tensor& tensor, int tensor_idx)
// {
//     auto& param = this->optimizer_->param_groups()[tensor_idx].params()[0];
//     auto& state = optimizer_->state();
//     auto key = c10::guts::to_string(param.unsafeGetTensorImpl());
//     auto& stored_state = static_cast<torch::optim::AdamParamState&>(*state[key]);
//     auto new_state = std::make_unique<torch::optim::AdamParamState>();
//     new_state->step(stored_state.step());
//     new_state->exp_avg(torch::zeros_like(tensor));
//     new_state->exp_avg_sq(torch::zeros_like(tensor));
//     // new_state->max_exp_avg_sq(stored_state.max_exp_avg_sq().clone()); // needed only when options.amsgrad(true), which is false by default

//     state.erase(key);
//     param = tensor.requires_grad_();
//     key = c10::guts::to_string(param.unsafeGetTensorImpl());
//     state[key] = std::move(new_state);

//     auto optimizable_tensors = param;
//     return optimizable_tensors;
// }

// void VoxelModel::scaledTransformationPostfix(torch::Tensor& new_geo)
// {
//     // We only need to re-wire the geometry key (geo_) back into Adam (param_group 0).
//     torch::Tensor optimizable_geo = this->replaceTensorToOptimizer(new_geo, /*group*/0);
//     this->geo_ = optimizable_geo;
//     // Update our helper vector so future replaceTensorToOptimizer calls still work:
//     this->Tensor_vec_geo_ = { this->geo_ };
//     // Note: we do *not* have a scaling tensor in the optimizer for voxels,
//     // so we omit any scaling_ logic here.
// }

// void VoxelModel::applyScaledTransformation(
//     const float s,
//     const Sophus::SE3f T)
// {
//     torch::NoGradGuard no_grad;
//     // pt <- (s * Ryw * pt + tyw)
//     this->center_ *= s;

//     torch::Tensor T_tensor =
//         tensor_utils::EigenMatrix2TorchTensor(T.matrix(), device_type_).transpose(0, 1);
//     // use the same helper as Photo-SLAM to apply SE3
//     torch::Tensor ones = torch::ones({center_.size(0), 1},
//                                  center_.options().dtype(torch::kFloat32));
//     torch::Tensor center_h = torch::cat({center_, ones}, 1);   // (N,4)
//     center_ = (T_tensor.matmul(center_h.t())).t()
//                 .index({torch::indexing::Slice(),
//                         torch::indexing::Slice(0,3)});

//     // we don’t have a “scaling_” tensor in Adam—our size_ lives outside—so scale size_ here
//     this->size_ *= s;

//     // geo_ stores variances along diag; those scale with s²
//     const float var_scale = s * s;
//     this->geo_.index_put_({torch::indexing::Slice(), 0},
//         this->geo_.index({torch::indexing::Slice(), 0}) * var_scale);
//     this->geo_.index_put_({torch::indexing::Slice(), 3},
//         this->geo_.index({torch::indexing::Slice(), 3}) * var_scale);
//     this->geo_.index_put_({torch::indexing::Slice(), 5},
//         this->geo_.index({torch::indexing::Slice(), 5}) * var_scale);

//     scaledTransformationPostfix(this->geo_);
// }

// void VoxelModel::scaledTransformVisiblePointsOfKeyframe(
//     torch::Tensor& point_not_transformed_flags,  // (N) bool
//     torch::Tensor& diff_pose,                    // (4×4)  SE(3) in world
//     torch::Tensor& kf_world_view_transform,      // unused (placeholder)
//     torch::Tensor& kf_full_proj_transform,       // unused (placeholder)
//     const int      kf_creation_iter,
//     const int      stable_num_iter_existence,
//     int&           num_transformed,
//     const float    scale)
// {
//     torch::NoGradGuard no_grad;

//     // 1) grab our “points” and (no rots for voxels)
//     torch::Tensor points = this->center_;

//     // 2) find “unstable” (just‐born) voxels
//     torch::Tensor point_unstable_flags = torch::abs(
//         this->exist_since_iter_ - kf_creation_iter
//     ) < stable_num_iter_existence;

//     // 3) those still eligible to transform
//     torch::Tensor to_check = 
//         point_not_transformed_flags.logical_and(~point_unstable_flags);
//     if (!to_check.any().item<bool>()) return;

//     // 4) (we cheat: treat *all* to_check as visible)
//     torch::Tensor to_transform = to_check;
//     if (!to_transform.any().item<bool>()) return;

//     // 5) apply diff_pose to centers[to_transform]
//     torch::Tensor idx = to_transform.nonzero().view(-1);
//     num_transformed += idx.numel();

//     // homogeneous
//     torch::Tensor sel_centers = points.index({idx});          // (M,3)
//     torch::Tensor ones = torch::ones(
//         {idx.size(0),1},
//         torch::TensorOptions().dtype(points.dtype()).device(points.device())
//     );
//     torch::Tensor centers_h = torch::cat({sel_centers, ones}, 1); // (M,4)
//     torch::Tensor updated = 
//         (diff_pose.matmul(centers_h.t()))
//         .t()
//         .index({torch::indexing::Slice(), torch::indexing::Slice(0,3)}); // (M,3)

//     this->center_.index_put_({idx}, updated);

//     // 6) scale size_ and geo_ diagonals
//     this->size_.index_put_({idx}, this->size_.index({idx}) * scale);
//     float vs = scale*scale;
//     this->geo_.index_put_({idx, 0}, this->geo_.index({idx, 0}) * vs);
//     this->geo_.index_put_({idx, 3}, this->geo_.index({idx, 3}) * vs);
//     this->geo_.index_put_({idx, 5}, this->geo_.index({idx, 5}) * vs);

//     // 7) mark done
//     point_not_transformed_flags.index_put_({idx}, false);

//     // === Postfix: re-wire our geo_ back into Adam’s param-group[0] ===
//     torch::Tensor optimizable_geo = this->replaceTensorToOptimizer(
//         this->geo_, /*tensor_idx=*/0
//     );
//     this->geo_ = optimizable_geo;
//     this->Tensor_vec_geo_ = { this->geo_ };
// }

void VoxelModel::trainingSetup(const VoxelOptimizationParams& training_args)
{
    // exactly as in Photo-SLAM:
    setPercentDense(training_args.percent_dense_);
    // this->subdiv_p_grad_buffer_ = torch::zeros_like(this->subdiv_p_);

    torch::optim::AdamOptions adam_options;
    adam_options.set_lr(0.0);
    adam_options.eps() = 1e-15;
    // adam_options.weight_decay() = training_args.geo_weight_decay_;

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
void VoxelModel::setGeoLearningRate(float geo_lr) 
{
    optimizer_->param_groups()[0].options().set_lr(geo_lr * this->spatial_lr_scale_);
}
void VoxelModel::setSh0LearningRate(float sh0_lr) 
{
    optimizer_->param_groups()[1].options().set_lr(sh0_lr);
}
void VoxelModel::setShsLearningRate(float shs_lr) 
{
    optimizer_->param_groups()[2].options().set_lr(shs_lr);
}

void VoxelModel::prune(const torch::Tensor& mask_keep)
{
    auto mk = mask_keep.to(torch::kBool).view(-1);
    TORCH_CHECK(mk.numel() == center_.size(0),
                "[VoxelModel::prune] mask length mismatch");

    int64_t n_left = mk.sum().item<int64_t>();
    if (n_left == 0) { std::cerr<<"[WARN] prune(): nothing left, skipping\n"; return; }
    if (n_left == center_.size(0)) return;

    auto idx_cuda = [&](const torch::Tensor& T, bool need_grad=false) {
      auto src = T.to(device_type_);
      auto out = src.index({mk});
      return need_grad ? out.requires_grad_(true) : out;
    };

    // 1) Leaf‐level tensors
    center_          = idx_cuda(center_           );
    size_            = idx_cuda(size_             );
    sh0_             = idx_cuda(sh0_,      true);
    shs_             = idx_cuda(shs_,      true);
    subdiv_meta_     = idx_cuda(subdiv_meta_,true);
    subdiv_p_        = idx_cuda(subdiv_p_,   true);

    oct_path_         = oct_path_.to(device_type_).index({mk});
    oct_level_        = oct_level_.to(device_type_).index({mk});

    // subdiv_p_grad_buffer_ = torch::zeros_like(subdiv_p_);
    sh0_.retain_grad();
    shs_.retain_grad();
    subdiv_meta_.retain_grad();
    subdiv_p_.retain_grad();

    // ─── prune our two stats‐buffers ───────────────
    xyz_gradient_accum_ = idx_cuda(xyz_gradient_accum_);
    denom_              = idx_cuda(denom_);

    voxel_error_sum_  = idx_cuda(voxel_error_sum_);
    voxel_hit_count_  = idx_cuda(voxel_hit_count_);

    // 2) Rebuild grid (same as before)
    if (oct_path_.numel()==0) {
      grid_pts_key_ = torch::empty({0},   torch::kLong).to(device_type_);
      vox_key_      = torch::empty({0,8}, torch::kLong).to(device_type_);
      _geo_grid_pts = torch::empty({0,1}, torch::kFloat32)
                         .to(device_type_).requires_grad_(true);
    } else {
        auto path_in = oct_path_.view({-1,1}).contiguous();          // int64
        auto lvl_in  = oct_level_.to(torch::kInt8)                   // <<< cast
                                .view({-1,1}).contiguous();        // int8

        py::gil_scoped_acquire gil;
        static py::module octree = py::module::import("src.utils.octree_utils");

        // both tensors already contiguous; pass as-is
        py::tuple link = octree.attr("build_grid_pts_link")(path_in, lvl_in);

        grid_pts_key_ = link[0].cast<torch::Tensor>()
                            .to(device_type_, torch::kLong);
        vox_key_      = link[1].cast<torch::Tensor>()
                            .to(device_type_, torch::kLong);

        const int64_t G = grid_pts_key_.size(0);
        if (_geo_grid_pts.size(0) != G)
            _geo_grid_pts = torch::full({G,1}, 4.f,
                            torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .device(device_type_))
                            .requires_grad_(true);
    }
    vox_size_inv_ = 1.0f / size_;
    rebuildOptimizer();
    // VOXEL_MODEL_TENSORS_TO_VEC;
}

void VoxelModel::subdivide(const torch::Tensor& mask_parent)
{
    TORCH_CHECK(mask_parent.dtype()==torch::kBool,
                "subdivide mask must be bool");

    auto idx_par = mask_parent.nonzero().view(-1);
    if (idx_par.numel()==0) return;

    auto parent_level = oct_level_.index({idx_par});
    auto below_max    = parent_level < MAX_OCT_LEVEL;
    idx_par           = idx_par.index({below_max});
    parent_level      = parent_level.index({below_max});
    if (idx_par.numel()==0) return;

    auto idx_keep = (~mask_parent).nonzero().view(-1);
    auto keep   = [&](const torch::Tensor& T){ return T.index({idx_keep}); };
    auto parent = [&](const torch::Tensor& T){ return T.index({idx_par}); };

    // kept voxels
    auto center_k  = keep(center_);
    auto size_k    = keep(size_);
    auto sh0_k     = keep(sh0_);
    auto shs_k     = keep(shs_);
    auto path_k    = keep(oct_path_);
    auto lvl_k     = keep(oct_level_);
    auto meta_k    = keep(subdiv_meta_);

    // --- keep our two stats‐buffers
    auto grad_k  = keep(xyz_gradient_accum_);
    auto denom_k = keep(denom_);

    // parents to split
    auto c_par     = parent(center_);
    auto s_par     = parent(size_);
    auto sh0_par   = parent(sh0_);
    auto shs_par   = parent(shs_);
    auto path_par  = parent(oct_path_);

    int64_t P = idx_par.numel(), C = P*8;
    auto s_child = (s_par*0.5f).repeat_interleave(8);

    static const float offs_data[8][3] = {
      {-0.25f,-0.25f,-0.25f},{+0.25f,-0.25f,-0.25f},
      {-0.25f,+0.25f,-0.25f},{+0.25f,+0.25f,-0.25f},
      {-0.25f,-0.25f,+0.25f},{+0.25f,-0.25f,+0.25f},
      {-0.25f,+0.25f,+0.25f},{+0.25f,+0.25f,+0.25f},
    };
    auto offs = torch::from_blob((void*)offs_data,{8,3},
                torch::TensorOptions().dtype(torch::kFloat32))
                .to(device_type_).repeat({P,1});
    auto c_child = c_par.repeat_interleave(8,0)
                   + offs*s_par.repeat_interleave(8,0).unsqueeze(1);

    auto rep8 = [&](const torch::Tensor& T){
                  return T.repeat_interleave(8,0);
                };
    auto sh0_child = rep8(sh0_par);
    auto shs_child = rep8(shs_par);
    auto meta_child= torch::zeros({C,1},torch::kFloat32)
                      .to(device_type_);

    auto lvl_child = parent_level.repeat_interleave(8)+1;
    auto idx8      = torch::arange(8,torch::kLong)
                      .to(device_type_).repeat({P});
    auto path_child= ((path_par*8).repeat_interleave(8))|idx8;

    // **new** child buffers → zeros
    auto grad_child  = torch::zeros({C,1}, torch::kFloat32)
                            .to(device_type_);
    auto denom_child = torch::zeros_like(grad_child);

    // concat keep + children
    center_      = torch::cat({center_k,   c_child},    0).detach();
    size_        = torch::cat({size_k,     s_child},    0).detach();
    sh0_         = torch::cat({sh0_k,      sh0_child},  0).contiguous().requires_grad_(true);
    shs_         = torch::cat({shs_k,      shs_child},  0).contiguous().requires_grad_(true);
    oct_path_    = torch::cat({path_k,     path_child}, 0);
    oct_level_   = torch::cat({lvl_k,      lvl_child},  0);
    subdiv_meta_ = torch::cat({meta_k,     meta_child}, 0).contiguous().requires_grad_(true);

    subdiv_p_            = torch::zeros_like(subdiv_meta_).requires_grad_(true);
    // subdiv_p_grad_buffer_= torch::zeros_like(subdiv_p_);
    subdiv_p_.retain_grad();
    subdiv_meta_.retain_grad();

    // ─── grow our two stats‐buffers ────────────────────────
    xyz_gradient_accum_ = torch::cat({grad_k,  grad_child }, 0);
    denom_              = torch::cat({denom_k, denom_child}, 0);

    auto err_k   = keep(voxel_error_sum_);
    auto cnt_k   = keep(voxel_hit_count_);
    auto err_ch  = torch::zeros({C,1}, torch::kFloat32).to(device_type_);
    auto cnt_ch  = torch::zeros_like(err_ch);
    voxel_error_sum_  = torch::cat({err_k,  err_ch}, 0);
    voxel_hit_count_  = torch::cat({cnt_k,  cnt_ch}, 0);

    // rebuild grid exactly as in prune()
    {
        auto path_in = oct_path_.view({-1,1}).contiguous();          // int64
        auto lvl_in  = oct_level_.to(torch::kInt8)                   // <<< cast
                                .view({-1,1}).contiguous();        // int8
        py::gil_scoped_acquire gil;
        static py::module octree = py::module::import("src.utils.octree_utils");
        py::tuple link = octree.attr("build_grid_pts_link")(path_in, lvl_in);
        grid_pts_key_ = link[0].cast<torch::Tensor>()
                            .to(device_type_,torch::kLong);
        vox_key_      = link[1].cast<torch::Tensor>()
                            .to(device_type_,torch::kLong);

        int64_t G = grid_pts_key_.size(0);
        if (_geo_grid_pts.size(0)!=G)
        _geo_grid_pts = torch::full({G,1},4.0f,
                            torch::TensorOptions()
                            .dtype(torch::kFloat32)
                            .device(device_type_))
                        .requires_grad_(true);
    }

    vox_size_inv_ = 1.0f / size_;
    rebuildOptimizer();
    // VOXEL_MODEL_TENSORS_TO_VEC;
}

void VoxelModel::setSubdivMeta(const torch::Tensor& updated)
{
    subdiv_meta_ = updated.clone().to(device_type_).requires_grad_(true);
    subdiv_meta_.retain_grad();
    // if (auto g = subdiv_meta_.grad(); g.defined()) {
    //     g.detach_();
    // }
}

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

void VoxelModel::accumulateSubdivGradients(const torch::Tensor& parent_idx,
                                           const torch::Tensor& parent_grads)
{
    TORCH_CHECK(subdiv_meta_.defined(), "subdiv_meta not ready");
    subdiv_meta_.index_add_(0,
        parent_idx.to(device_type_),
        parent_grads.to(device_type_));
}

TrainingStat VoxelModel::computeTrainingStat(
        const std::vector<MiniCam>& cameras,
        const py::array_t<uint8_t>& rgb_image)
{
    const int64_t N = center_.size(0);
    torch::Tensor max_w            = torch::zeros({N,1}, torch::kFloat32).to(device_type_);
    torch::Tensor min_samp_interval= torch::full({N,1}, 1e30f, torch::kFloat32).to(device_type_);
    torch::Tensor view_cnt         = torch::zeros({N,1}, torch::kFloat32).to(device_type_);

    bool rg_backup = _geo_grid_pts.requires_grad();
    _geo_grid_pts.set_requires_grad(false);

    for (const MiniCam& cam_in : cameras)
    {
        //–– prepare camera on correct device
        MiniCam cam = cam_in;
        cam.c2w = cam.c2w.to(device_type_);
        cam.w2c = cam.w2c.to(device_type_);

        //–– render only max_w
        auto pkg = render(cam, rgb_image, "");
        torch::cuda::synchronize();
        torch::Tensor mw_i = pkg["max_w"].to(device_type_);      // (N,1)
        max_w = torch::maximum(max_w, mw_i);

        //–– derive visibility directly from max_w
        auto vis_mask = (mw_i.view(-1) > 0);                    // bool[N]
        auto vis_idx  = vis_mask.nonzero().squeeze(1);          // (K,)
        if (vis_idx.numel() == 0) continue;

        //–– sample-interval as before
        auto cam_pos = cam.c2w.index({torch::indexing::Slice(0,3),3}).to(device_type_);
        auto lookat  = cam.c2w.index({torch::indexing::Slice(0,3),2}).to(device_type_).neg();
        lookat = lookat / lookat.norm();
        float pix_size = 2.0f * cam.tanfovy / float(cam.height);

        auto zdist = ((center_.index({vis_idx}) - cam_pos) * lookat)
                         .sum(-1, /*keepdim=*/true);           // (K,1)
        auto samp_interval = zdist * pix_size;                  // (K,1)

        min_samp_interval.index_put_(
            {vis_idx},
            torch::minimum(min_samp_interval.index({vis_idx}),
                           samp_interval));

        view_cnt.index_put_({vis_idx},
                            view_cnt.index({vis_idx}) + 1);
    }

    _geo_grid_pts.set_requires_grad(rg_backup);
    return {max_w, min_samp_interval, view_cnt};
}

void VoxelModel::rebuildOptimizer()
{
    // 1) fetch current LRs from each group, or fall back to lr_init_
    float geo_lr = lr_init_;
    float sh0_lr = 0.0f;
    float shs_lr = 0.0f;

    if (optimizer_) {
        // group 0: geometry
        {
            auto &opts = static_cast<const torch::optim::AdamOptions&>(
                optimizer_->param_groups()[0].options());
            geo_lr = opts.get_lr();
        }
        // group 1: sh0
        {
            auto &opts = static_cast<const torch::optim::AdamOptions&>(
                optimizer_->param_groups()[1].options());
            sh0_lr = opts.get_lr();
        }
        // group 2: shs
        {
            auto &opts = static_cast<const torch::optim::AdamOptions&>(
                optimizer_->param_groups()[2].options());
            shs_lr = opts.get_lr();
        }
    }

    // 2) reset the old optimizer (drops its state)
    optimizer_.reset();

    // 3) re-collect parameters into the three groups
    VOXEL_MODEL_TENSORS_TO_VEC

    // 4) create a fresh Adam with identical LRs
    torch::optim::AdamOptions adam_opt(/*lr=*/0.0);
    adam_opt.eps() = 1e-15;

    optimizer_ = std::make_unique<torch::optim::Adam>(Tensor_vec_geo_, adam_opt);
    optimizer_->param_groups()[0].options().set_lr(geo_lr);

    optimizer_->add_param_group(Tensor_vec_sh0_);
    optimizer_->param_groups()[1].options().set_lr(sh0_lr);

    optimizer_->add_param_group(Tensor_vec_shs_);
    optimizer_->param_groups()[2].options().set_lr(shs_lr);
}

//------------------------------------------------------------------------------
// Return all optimizable parameters for the outer training loop.
// Mirrors GaussianModel::parameters() → returns {geo_, sh0_, shs_}.
//------------------------------------------------------------------------------
std::vector<torch::Tensor> VoxelModel::parameters() const
{
    return { geo_, sh0_, shs_, subdiv_p_ };
}

std::unordered_map<std::string, torch::Tensor>
VoxelModel::render(const MiniCam&              cam,
                   const py::array_t<uint8_t>& rgb_image,
                   const std::string&          output_dir) const
{
    /* --------------------------------------------------------------------- */
    /* 1) import the python entry-point once                                 */
    /* --------------------------------------------------------------------- */
    py::gil_scoped_acquire gil;
    static py::object py_render;
    if (!py_render) {
        try {
            py_render = py::module_::import(
                            "scripts_voxel.python_svraster_bridge.renderer_wrapper")
                            .attr("render");
            std::cerr << "[INFO] Python-side renderer imported OK.\n";
        } catch (const py::error_already_set& e) {
            std::cerr << "[PYBIND11] Could not import renderer_wrapper:\n"
                      << e.what() << std::endl;
            return {};
        }
    }

    /* --------------------------------------------------------------------- */
    /* 2) quick bail-out if there is nothing to render                       */
    /* --------------------------------------------------------------------- */
    if (center_.numel() == 0) {
        std::cout << "[INFO] Skipping render – voxel buffer empty.\n";
        return {};
    }

    /* --------------------------------------------------------------------- */
    /* 3) pack voxel buffers into a python dict                              */
    /* --------------------------------------------------------------------- */
    py::dict d;
    d["colors"]        = py::cast(sh0_);
    d["shs"]           = py::cast(shs_);
    d["subdiv_p"]      = py::cast(subdiv_p_);
    d["octpaths"]      = py::cast(oct_path_.cpu());
    d["centers"]       = py::cast(center_.cpu());
    d["vox_lengths"]   = py::cast(size_.cpu());
    d["octlevels"]     = py::cast(oct_level_.cpu());
    d["subdiv_meta"]   = py::cast(subdiv_meta_.cpu());
    d["sh_degree"]     = py::int_(active_sh_degree_);
    d["_geo_grid_pts"] = py::cast(_geo_grid_pts);
    d["vox_key"]       = py::cast(vox_key_.cpu());
    d["vox_size_inv"]  = py::cast(vox_size_inv_.cpu());

    /* --------------------------------------------------------------------- */
    /* 4) call python                                                        */
    /* --------------------------------------------------------------------- */
    py::object py_cam  = MiniCam_to_py(cam);
    py::object py_out;
    try {
        py_out = py_render(py_cam, d, rgb_image, output_dir);
    } catch (const py::error_already_set& e) {
        std::cerr << "[PYBIND11] Exception thrown by renderer:\n"
                  << e.what() << std::endl;
        return {};
    }

    /* --------------------------------------------------------------------- */
    /* 5) the python function *should* return a dict                         */
    /* --------------------------------------------------------------------- */
    if (!py::isinstance<py::dict>(py_out)) {
        std::cerr << "[WARN] renderer_wrapper.render returned "
                     "a non-dict object - skipping frame.\n";
        return {};
    }
    py::dict out_dict = py_out.cast<py::dict>();

    /* --------------------------------------------------------------------- */
    /* 6) copy every tensor into a C++ map                                   */
    /* --------------------------------------------------------------------- */
    std::unordered_map<std::string, torch::Tensor> pkg;
    // std::cout << "\n[DBG]  content coming back from Python:\n";
    for (auto item : out_dict) {
        const std::string key = py::str(item.first);

        // try to cast – even if isinstance fails we fall back to a try/catch
        torch::Tensor t;
        bool is_tensor = true;
        try {
            t = item.second.cast<torch::Tensor>();
        } catch (...) {
            is_tensor = false;
        }

        // std::cout << "    • " << key << "  : ";
        if (!is_tensor) {
            std::cout << "(not a tensor)\n";
            continue;
        }
        // std::cout << (t.defined() ? "tensor  " : "UNDEFINED ")
        //           << t.sizes() << '\n';

        if (t.defined())
            pkg.emplace(key, std::move(t));
    }

    /* --------------------------------------------------------------------- */
    /* 7) sanity-check – we really want at least “color”                     */
    /* --------------------------------------------------------------------- */
    if (!pkg.count("color") || !pkg["color"].defined()) {
        std::cerr << "[ERROR] Python renderer did not supply a valid "
                     "\"color\" tensor – skipping frame.\n";
        pkg.clear();                  // signal failure to caller
    }

    // 7) Build visibility index from max_w directly
    auto max_w = pkg["max_w"].view(-1);
    auto vis_idx = (max_w > 0).nonzero().squeeze(1);
    pkg["idx"] = vis_idx;
    
    return pkg;                       // may be empty on error
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

