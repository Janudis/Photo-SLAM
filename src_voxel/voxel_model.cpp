#include "include_voxel/voxel_model.h"
#include <fstream>
#include <sstream>

namespace py = pybind11;
namespace sv {

struct __attribute__((visibility("hidden"))) VoxelModel::PyState {
    py::object svm;        // Python SparseVoxelModel
    py::object optimizer_py;  // Python SparseAdam

    ~PyState() {
        py::gil_scoped_acquire gil;
        svm = py::none();
        optimizer_py = py::none();
    }
};
VoxelModel::~VoxelModel() {
    py::gil_scoped_acquire gil;
    py_.reset(); // drops optimizer & svm safely under GIL
}

VoxelModel::VoxelModel(const int sh_degree)
    : active_sh_degree_(0)
{
    this->max_sh_degree_ = sh_degree;

    // Device
    if (torch::cuda::is_available())
        device_type_ = torch::kCUDA;
    else
        device_type_ = torch::kCPU;

    // Initialize all tensors on chosen device
    VOXEL_MODEL_INIT_TENSORS(this->device_type_);
    py_ = std::make_unique<PyState>();
}

VoxelModel::VoxelModel(const VoxelModelParams& model_params)
    : active_sh_degree_(0)
{
    this->max_sh_degree_ = model_params.sh_degree_;

    // Device
    if (model_params.data_device_ == "cuda")
        device_type_ = torch::kCUDA;
    else
        device_type_ = torch::kCPU;

    // Initialize all tensors on chosen device
    VOXEL_MODEL_INIT_TENSORS(this->device_type_);
    py_ = std::make_unique<PyState>();
}

void VoxelModel::oneUpShDegree()
{
    // '''
    // sh_degree_add1 from svraster 
    // '''
    if (this->active_sh_degree_ < this->max_sh_degree_)
        this->active_sh_degree_ += 1;
}

void VoxelModel::setShDegree(const int sh)
{
    this->active_sh_degree_ = (sh > this->max_sh_degree_ ? this->max_sh_degree_ : sh);
}

void VoxelModel::createFromPcd(
    const std::map<point3D_id_t, Point3D>& pcd)
{
    // '''
    // offline approach with dense grid
    // '''
    std::cout << "VoxelModel::createFromPcd() called with "
              << pcd.size() << " points." << std::endl;
    namespace py = pybind11;

    // ─── 1) Pack PCD into two [N×3] CUDA tensors ────────────────────────
    const int N = (int)pcd.size();
    torch::Tensor xyz = torch::empty({N,3},
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(device_type_));
    torch::Tensor rgb = torch::empty({N,3},
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(device_type_));
    {
        int i = 0;
        for (auto& [id, P] : pcd) {
            xyz[i] = torch::tensor(
                      {P.xyz_(0), P.xyz_(1), P.xyz_(2)},
                      torch::kFloat32);
            // normalize 0–1 range
            rgb[i] = torch::tensor(
                      {P.color_(0), P.color_(1), P.color_(2)},
                      torch::kFloat32) / 255.0f;
            ++i;
        }
    }

    // ─── 2) Use KNOWN inflated bounding directly (no file, no Python utils) ──
    auto opts_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor bounding_cpu = torch::empty({2,3}, opts_cpu);
    // Row 0 = min, Row 1 = max
    bounding_cpu[0][0] = -10.748001f;  bounding_cpu[0][1] = -29.842281f;  bounding_cpu[0][2] = -0.293899f;
    bounding_cpu[1][0] =  56.780504f;  bounding_cpu[1][1] =  18.760179f;  bounding_cpu[1][2] = 66.683914f;

    // Move to your device for model_init:
    torch::Tensor bounding = bounding_cpu.to(device_type_);
    torch::Tensor scene_min = bounding_cpu[0].to(device_type_);
    torch::Tensor scene_max = bounding_cpu[1].to(device_type_);
    this->scene_center_ = 0.5f * (scene_min + scene_max);
    float extent_side  = (scene_max - scene_min).max().item<float>();
    this->scene_extent_       = torch::tensor({extent_side},
                                 torch::TensorOptions().dtype(torch::kFloat32)
                                                        .device(device_type_));
    float float_level = -std::log2(0.05f / extent_side);
    int level = std::round(float_level);
    std::cout << "level = " << level
              << " (float_level = " << float_level << ")" << std::endl;

    // ─── 3) Call into Python’s SparseVoxelModel.model_init() ───────────────
    py::gil_scoped_acquire gil;
    static py::module svm_mod = []{
        py::module sys = py::module::import("sys");
        sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
        return py::module::import("src.sparse_voxel_model");
    }();
    py::object SVM    = svm_mod.attr("SparseVoxelModel");
    py_->svm = SVM(py::arg("sh_degree")        = max_sh_degree_,
                        py::arg("black_background") = true);

    py_->svm.attr("model_init")(
        py::arg("bounding")        = bounding,
        py::arg("outside_level")   = 0,
        py::arg("init_n_level")    = 6,
        py::arg("init_out_ratio")  = 2,
        py::arg("sh_degree_init")  = 3,
        py::arg("geo_init")        = -10.0f,
        py::arg("sh0_init")        = 0.5f,
        py::arg("shs_init")        = 0.0f
    );

    // ─── 4) Pull back all core tensors from the Python object ───────────
    auto fetch = [&](const char* name){
        return py_->svm.attr(name).cast<torch::Tensor>().contiguous();
    };
    this->oct_path_      = fetch("octpath");
    this->oct_level_     = fetch("octlevel");
    this->center_        = fetch("vox_center");
    this->size_          = fetch("vox_size").squeeze(1);
    this->vox_size_inv_  = 1.0f / size_;
    this->grid_pts_key_  = fetch("grid_pts_key");
    this->vox_key_       = fetch("vox_key");      
    // ─── 5) Copy over the learnable fields ─────────────────────────────
    this->_geo_grid_pts_ = fetch("_geo_grid_pts");
    this->sh0_ = fetch("_sh0").view({-1,1,3});
    this->shs_ = fetch("_shs");
    this->subdiv_p_ = fetch("_subdiv_p");
    // subdiv_p_   .retain_grad();
    // ─── 6) Stats buffers (exactly as before) ───────────────────────────
    this->max_w_ = torch::zeros({center_.size(0), 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

    // std::cout << " center_ " << center_ << std::endl;
    // After you fetch center_ (M×3) and size_ (M,)
    auto c_cpu = center_.cpu();
    auto s_cpu = size_.cpu();
    torch::Tensor bb_min_eff = (c_cpu - 0.5 * s_cpu.unsqueeze(1)).amin(/*dim=*/0);
    torch::Tensor bb_max_eff = (c_cpu + 0.5 * s_cpu.unsqueeze(1)).amax(/*dim=*/0);
    this->bb_min_eff_ = bb_min_eff.to(device_type_);
    this->bb_max_eff_ = bb_max_eff.to(device_type_);
    std::cout << std::fixed << std::setprecision(6)
            << "[VOXEL GRID AABB] min:[" << bb_min_eff[0].item<float>() << ","
                                        << bb_min_eff[1].item<float>() << ","
                                        << bb_min_eff[2].item<float>() << "]  "
            << "max:[" << bb_max_eff[0].item<float>() << ","
                        << bb_max_eff[1].item<float>() << ","
                        << bb_max_eff[2].item<float>() << "]\n";

    py::module_ np = py::module_::import("numpy");
    // save raw point‐cloud + colors
    np.attr("save")("/home/dimitris/Photo-SLAM/debug_pts.npy",
                    tensor_to_numpy(xyz.cpu()));
    np.attr("save")("/home/dimitris/Photo-SLAM/debug_cols.npy",
                    tensor_to_numpy(rgb.cpu()));
    // save voxel cell centres + edge lengths
    np.attr("save")("/home/dimitris/Photo-SLAM/debug_vox_centers.npy",
                    tensor_to_numpy(center_.cpu()));
    np.attr("save")("/home/dimitris/Photo-SLAM/debug_vox_size.npy",
                    tensor_to_numpy(size_.cpu()));
    torch::Tensor sh0_cpu = sh0_.view({-1,3}).cpu().contiguous();
    np.attr("save")("/home/dimitris/Photo-SLAM/debug_vox_color.npy",
                    tensor_to_numpy(sh0_cpu));

    VOXEL_MODEL_TENSORS_TO_VEC
}

// void VoxelModel::createFromPcd(const std::map<point3D_id_t, Point3D>& pcd)
// {
//     // '''
//     // dense grid
//     // '''
//     std::cout << "VoxelModel::createFromPcd() called with "
//               << pcd.size() << " points." << std::endl;
//     namespace py = pybind11;

//     // ─── 1) Pack PCD into two [N×3] CUDA tensors ────────────────────────
//     const int N = (int)pcd.size();
//     torch::Tensor xyz = torch::empty({N,3},
//         torch::TensorOptions()
//             .dtype(torch::kFloat32)
//             .device(device_type_));
//     torch::Tensor rgb = torch::empty({N,3},
//         torch::TensorOptions()
//             .dtype(torch::kFloat32)
//             .device(device_type_));
//     {
//         int i = 0;
//         for (auto& [id, P] : pcd) {
//             xyz[i] = torch::tensor(
//                       {P.xyz_(0), P.xyz_(1), P.xyz_(2)},
//                       torch::kFloat32);
//             // normalize 0–1 range
//             rgb[i] = torch::tensor(
//                       {P.color_(0), P.color_(1), P.color_(2)},
//                       torch::kFloat32) / 255.0f;
//             ++i;
//         }
//     }

//     // ─── 2) Ask Python to compute the bounding box via decide_main_bounding ──
//     py::gil_scoped_acquire gil;
//     static py::module bu_mod = []{
//         py::module sys = py::module::import("sys");
//         sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
//         return py::module::import("src.utils.bounding_utils");
//     }();
//     py::object decide_main_bounding = bu_mod.attr("decide_main_bounding");

//     // xyz.mul_(6.0f);
//     // Pass the point cloud to Python as a (N,3) float32 tensor on CPU
//     torch::Tensor xyz_cpu = xyz.cpu().contiguous();
//     // 1) Wrap the CPU tensor as a Python torch.Tensor
//     py::object torch_tensor_py = py::cast(xyz_cpu);
//     // 2) Call its .numpy() in Python to get a NumPy array
//     py::object np_array = torch_tensor_py.attr("numpy")();
//     // 3) Build a SimpleNamespace(points=ndarray) so .points works in Python
//     py::object SimpleNS = py::module_::import("types").attr("SimpleNamespace");
//     py::object pcd_obj  = SimpleNS(py::arg("points") = np_array);

//     // py::list tr_cams;
//     // py::list py_cams; 

//     // for (auto& kf : keyframes)
//     // {
//     //     /* 1.  Position ------------------------------------------------------ */
//     //     const Eigen::Vector3d& p = kf->t_;               // translation part
//     //     torch::Tensor pos_cpu = torch::tensor(
//     //         {static_cast<float>(p.x()),
//     //         static_cast<float>(p.y()),
//     //         static_cast<float>(p.z())},
//     //         torch::TensorOptions().dtype(torch::kFloat32));   // stays on CPU
                                                            
//     //     /* 2.  Forward (+Z) direction in world space ------------------------- */
//     //     Eigen::Vector3d fwd_eig = kf->R_quaternion_ * Eigen::Vector3d(0,0,1);
//     //     fwd_eig.normalize();
//     //     torch::Tensor look_cpu = torch::tensor(
//     //         {static_cast<float>(fwd_eig.x()),
//     //         static_cast<float>(fwd_eig.y()),
//     //         static_cast<float>(fwd_eig.z())},
//     //         torch::TensorOptions().dtype(torch::kFloat32));

//     //     /* 3.  Wrap as NumPy for SimpleNamespace ----------------------------- */
//     //     py::object np_pos    = py::cast(pos_cpu).attr("numpy")();
//     //     py::object np_lookat = py::cast(look_cpu).attr("numpy")();

//     //     tr_cams.append(
//     //         SimpleNS(py::arg("position") = np_pos,
//     //                 py::arg("lookat")   = np_lookat)
//     //     );
//     //     // build a *real* MiniCam
//     //     sv::MiniCam mc  = kf->toMiniCam();          // C++ struct
//     //     py::object cam_py = MiniCam_to_py(mc);      // Python object
//     //     cam_py.attr("w2c") = cam_py.attr("w2c").attr("cuda")();
//     //     cam_py.attr("c2w") = cam_py.attr("c2w").attr("cuda")();
//     //     cam_py.attr("position") = cam_py.attr("c2w")[py::make_tuple(py::slice(0,3,1), 3)];
//     //     cam_py.attr("lookat")   = cam_py.attr("c2w")[py::make_tuple(py::slice(0,3,1), 2)];
//     //     py_cams.append(cam_py);                     // keep for model_init()
//     // }

//     // py::list all_tr_cams = build_tr_cams_from_file_light(cam_pose_txt_path, 1.0f);
//     // py::list all_py_cams = build_full_py_cams_from_file(cam_pose_txt_path, 1.0f);

//     py::object bounding_py = decide_main_bounding(
//         py::arg("bound_mode")      = "pcd",   // or "default" / "camera_max" …
//         py::arg("pcd_density_rate")= 0.1,
//         py::arg("bound_scale")     = 1.0,
//         py::arg("pcd")             = pcd_obj  // just needs .points attribute-like
//     );
//     // py::object bounding_py = decide_main_bounding(
//     //     py::arg("bound_mode")      = "default",   // or "default" / "camera_max" …
//     //     py::arg("tr_cams")             = all_tr_cams  // just needs .points attribute-like
//     // );
//     // 1) cast to py::array_t<float> so we can access the buffer
//     auto arr = bounding_py.cast<py::array_t<float>>();
//     py::buffer_info info = arr.request();
//     // 2) build a CPU tensor that wraps that data
//     //    shape should be [2,3] here
//     std::vector<int64_t> shape{ (int64_t)info.shape[0],
//                                 (int64_t)info.shape[1] };
//     auto options = torch::TensorOptions()
//                     .dtype(torch::kFloat32)
//                     .device(torch::kCPU);
//     torch::Tensor bounding_cpu = torch::from_blob(
//         info.ptr, shape, options);
//     // 3) clone + move to your GPU (or whatever device_type_ is)
//     torch::Tensor bounding = bounding_cpu.clone()
//                                 .to(device_type_);

//     torch::Tensor scene_min = bounding_cpu[0].to(device_type_);
//     torch::Tensor scene_max = bounding_cpu[1].to(device_type_);
//     this->scene_center_ = 0.5f * (scene_min + scene_max);
//     float extent_side  = (scene_max - scene_min).max().item<float>();
//     // extent_side *= 3.0f;
//     this->scene_extent_       = torch::tensor({extent_side},
//                                  torch::TensorOptions().dtype(torch::kFloat32)
//                                                         .device(device_type_));
//     // std::cout << "[Debug] scene_extent = " << extent_side << std::endl;
//     float float_level = -std::log2(0.05f / extent_side);
//     int level = std::round(float_level);
//     std::cout << "level = " << level
//               << " (float_level = " << float_level << ")" << std::endl;
//     // const int init_level = 6;
//     // // option A: using std::pow
//     // float vox_size_pow   = extent_side * std::pow(2.0f, -init_level);
//     // // option B: using std::ldexp (more efficient for integer exponents)
//     // float vox_size_ldexp = std::ldexp(extent_side, -init_level);
//     // std::cout << "[Debug] scene_extent = " << extent_side
//     //         << ", init_n_level = " << init_level
//     //         << "\n         voxel_size_pow   = " << vox_size_pow
//     //         << "\n         voxel_size_ldexp = " << vox_size_ldexp
//     //         << std::endl;

//     // ─── 3) Call into Python’s SparseVoxelModel.model_init() ────────────
//     // py::gil_scoped_acquire gil;
//     static py::module svm_mod = []{
//         py::module sys = py::module::import("sys");
//         sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
//         return py::module::import("src.sparse_voxel_model");
//     }();
//     py::object SVM     = svm_mod.attr("SparseVoxelModel");
//     py_->svm = SVM(
//         py::arg("sh_degree")        = max_sh_degree_,
//         py::arg("black_background") = true
//     );
//     py_->svm.attr("model_init")(
//         py::arg("bounding")        = bounding,
//         py::arg("outside_level")   = 0,
//         // py::arg("init_n_level")    = level,
//         py::arg("init_n_level")    = 6,
//         py::arg("init_out_ratio")  = 2,
//         py::arg("sh_degree_init")  = 3,
//         py::arg("geo_init")        = 0.1f,
//         py::arg("sh0_init")        = 0.5f,
//         py::arg("shs_init")        = 0.0f
//         // py::arg("cameras")  = all_py_cams
//     );

//     // ─── 4) Pull back all core tensors from the Python object ───────────
//     auto fetch = [&](const char* name){
//         return py_->svm.attr(name).cast<torch::Tensor>().contiguous();
//     };
//     this->oct_path_      = fetch("octpath");
//     this->oct_level_     = fetch("octlevel");
//     this->center_        = fetch("vox_center");
//     this->size_          = fetch("vox_size").squeeze(1);
//     this->vox_size_inv_  = 1.0f / size_;
//     this->grid_pts_key_  = fetch("grid_pts_key");
//     this->vox_key_       = fetch("vox_key");      
//     // ─── 5) Copy over the learnable fields ─────────────────────────────
//     this->_geo_grid_pts_ = fetch("_geo_grid_pts");
//     this->sh0_ = fetch("_sh0").view({-1,1,3});
//     this->shs_ = fetch("_shs");
//     this->subdiv_p_ = fetch("_subdiv_p");
//     // subdiv_p_   .retain_grad();
//     // ─── 6) Stats buffers (exactly as before) ───────────────────────────
//     this->max_w_ = torch::zeros({center_.size(0), 1},
//         torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

//     py::module_ np = py::module_::import("numpy");
//     // save raw point‐cloud + colors
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_pts.npy",
//                     tensor_to_numpy(xyz.cpu()));
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_cols.npy",
//                     tensor_to_numpy(rgb.cpu()));
//     // save voxel cell centres + edge lengths
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_vox_centers.npy",
//                     tensor_to_numpy(center_.cpu()));
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_vox_size.npy",
//                     tensor_to_numpy(size_.cpu()));
//     torch::Tensor sh0_cpu = sh0_.view({-1,3}).cpu().contiguous();
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_vox_color.npy",
//                     tensor_to_numpy(sh0_cpu));

//     // ─── 7) Register with the optimizer, etc. ──────────────────────────
//     VOXEL_MODEL_TENSORS_TO_VEC
// }

// void VoxelModel::createFromPcd(
//     const std::map<point3D_id_t, Point3D>& pcd,
//     float spatial_lr_scale,
//     const std::vector<std::shared_ptr<VoxelKeyframe>>& keyframes,
//     const std::string& cam_pose_txt_path)
// {
//     // '''
//     // dense grid
//     // '''
//     std::cout << "VoxelModel::createFromPcd() called with "
//               << pcd.size() << " points." << std::endl;
//     namespace py = pybind11;
//     this->spatial_lr_scale_ = spatial_lr_scale;

//     // ─── 1) Pack PCD into two [N×3] CUDA tensors ────────────────────────
//     const int N = (int)pcd.size();
//     torch::Tensor xyz = torch::empty({N,3},
//         torch::TensorOptions()
//             .dtype(torch::kFloat32)
//             .device(device_type_));
//     torch::Tensor rgb = torch::empty({N,3},
//         torch::TensorOptions()
//             .dtype(torch::kFloat32)
//             .device(device_type_));
//     {
//         int i = 0;
//         for (auto& [id, P] : pcd) {
//             xyz[i] = torch::tensor(
//                       {P.xyz_(0), P.xyz_(1), P.xyz_(2)},
//                       torch::kFloat32);
//             // normalize 0–1 range
//             rgb[i] = torch::tensor(
//                       {P.color_(0), P.color_(1), P.color_(2)},
//                       torch::kFloat32) / 255.0f;
//             ++i;
//         }
//     }

//     // 2) Compute PCD AABB in world units
//     auto mn = std::get<0>(xyz.min(0));
//     auto mx = std::get<0>(xyz.max(0));
//     torch::Tensor center = 0.5f * (mn + mx);
//     float side_raw = (mx - mn).max().item<float>();

//     // Inflate the box (choose your factor)
//     const float bound_scale = 1.0f;          // ← make this a cfg knob
//     float side = side_raw * bound_scale;     // cube with equal sides

//     this->scene_center = center.to(device_type_);
//     this->scene_extent = torch::tensor({side},
//                         torch::TensorOptions()
//                             .dtype(torch::kFloat32)
//                             .device(device_type_));

//     // Build [min,max] bounding tensor for model_init
//     torch::Tensor scene_min = this->scene_center - 0.5f * this->scene_extent;
//     torch::Tensor scene_max = this->scene_center + 0.5f * this->scene_extent;
//     torch::Tensor bounding = torch::stack({scene_min, scene_max}).contiguous();

//     py::gil_scoped_acquire gil;
//     // std::cout << "[Debug] scene_extent = " << extent_side << std::endl;
//     // float float_level = -std::log2(0.2f / side);
//     // int level = std::round(float_level);
//     // std::cout << "level = " << level
//     //           << " (float_level = " << float_level << ")" << std::endl;

//     // ─── 3) Call into Python’s SparseVoxelModel.model_init() ────────────
//     // py::gil_scoped_acquire gil;
//     static py::module svm_mod = []{
//         py::module sys = py::module::import("sys");
//         sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
//         return py::module::import("src.sparse_voxel_model");
//     }();
//     auto SVM     = svm_mod.attr("SparseVoxelModel");
//     py::object svm = SVM(
//         py::arg("sh_degree")        = max_sh_degree_,
//         py::arg("black_background") = true
//     );
//     svm.attr("model_init")(
//         py::arg("bounding")        = bounding,
//         py::arg("outside_level")   = 0,
//         // py::arg("init_n_level")    = level,
//         py::arg("init_n_level")    = 6,
//         py::arg("init_out_ratio")  = 2,
//         py::arg("sh_degree_init")  = 3,
//         py::arg("geo_init")        = 0.1f,
//         py::arg("sh0_init")        = 0.5f,
//         py::arg("shs_init")        = 0.0f
//         // py::arg("cameras")  = all_py_cams
//     );

//     // ─── 4) Pull back all core tensors from the Python object ───────────
//     auto fetch = [&](const char* name){
//         return svm.attr(name).cast<torch::Tensor>()
//                   .contiguous()
//                   .to(device_type_);
//     };

//     oct_path_      = fetch("octpath").to(torch::kLong);
//     oct_level_     = fetch("octlevel").to(torch::kInt8);
//     center_        = fetch("vox_center");
//     size_          = fetch("vox_size").squeeze(1);
//     vox_size_inv_  = 1.0f / size_;
//     grid_pts_key_  = fetch("grid_pts_key");
//     vox_key_       = fetch("vox_key");         

//     // ─── 5) Copy over the learnable fields ─────────────────────────────
//     _geo_grid_pts_ = fetch("_geo_grid_pts")
//                       .detach().requires_grad_(true);
//     sh0_ = fetch("_sh0").view({-1,1,3})
//               .detach().requires_grad_(true);
//     shs_ = fetch("_shs")
//               .detach().requires_grad_(true);
//     subdiv_p_ = fetch("_subdiv_p")
//                    .detach().requires_grad_(true);
//     // we still need subdiv_meta_ locally
//     subdiv_meta_ = torch::zeros_like(subdiv_p_)
//                       .requires_grad_(true);
//     // subdiv_p_   .retain_grad();
//     // subdiv_meta_.retain_grad();

//     py::module_ np = py::module_::import("numpy");
//     // save raw point‐cloud + colors
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_pts.npy",
//                     tensor_to_numpy(xyz.cpu()));
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_cols.npy",
//                     tensor_to_numpy(rgb.cpu()));
//     // save voxel cell centres + edge lengths
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_vox_centers.npy",
//                     tensor_to_numpy(center_.cpu()));
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_vox_size.npy",
//                     tensor_to_numpy(size_.cpu()));

//     torch::Tensor sh0_cpu = sh0_.view({-1,3}).cpu().contiguous();
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_vox_color.npy",
//                     tensor_to_numpy(sh0_cpu));

//     // ─── 6) Stats buffers (exactly as before) ───────────────────────────
//     xyz_gradient_accum_ = torch::zeros_like(subdiv_p_);
//     denom_              = torch::zeros_like(subdiv_p_);
//     voxel_error_sum_    = torch::zeros_like(subdiv_p_);
//     voxel_hit_count_    = torch::zeros_like(subdiv_p_);
//     sh0_accum_ = torch::zeros({center_.size(0),3}, xyz.options());
//     hit_count_ = torch::zeros({center_.size(0),1}, xyz.options());

//     // ─── 7) Register with the optimizer, etc. ──────────────────────────
//     VOXEL_MODEL_TENSORS_TO_VEC
// }

// void VoxelModel::createFromPcd(
//     const std::map<point3D_id_t, Point3D>& pcd)
// {
//     // '''
//     // points_init
//     // '''
//     std::cout << "VoxelModel::createFromPcd() called with "
//               << pcd.size() << " points." << std::endl;
//     namespace py = pybind11;

//     /* 1) copy PCD → CUDA (unchanged) */
//     const int N_pts = static_cast<int>(pcd.size());
//     torch::Tensor pts  = torch::zeros({N_pts,3}, torch::kFloat32).to(device_type_);
//     torch::Tensor cols = torch::zeros({N_pts,3}, torch::kFloat32).to(device_type_);
//     {
//         int i = 0;
//         for (const auto& kv : pcd) {
//             const Point3D& p = kv.second;
//             pts[i]  = torch::tensor({p.xyz_(0), p.xyz_(1), p.xyz_(2)},
//                                     torch::kFloat32).to(device_type_);
//             cols[i] = torch::tensor({p.color_(0), p.color_(1), p.color_(2)},
//                                     torch::kFloat32).to(device_type_);
//             ++i;
//         }
//     }

//     /* 2) scene bounds & extent (unchanged) */
//     auto mn = std::get<0>(pts.min(0));
//     auto mx = std::get<0>(pts.max(0));
//     float raw_extent     = (mx - mn).max().item<float>();
//     float padded_extent = raw_extent * 1.01f;   // ← 1% padding
//     this->scene_center_ = (mn + mx) * 0.5f;
//     this->scene_extent_ = torch::tensor({padded_extent},
//                                  torch::TensorOptions().dtype(torch::kFloat32)
//                                                         .device(device_type_));
//     // float expected_vox_size =  std::ldexp(padded_extent, -6);

//     // py::gil_scoped_acquire gil;
//     // static py::module bu_mod = []{
//     //     py::module sys = py::module::import("sys");
//     //     sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
//     //     return py::module::import("src.utils.bounding_utils");
//     // }();
//     // py::object decide_main_bounding = bu_mod.attr("decide_main_bounding");
//     // // Pass the point cloud to Python as a (N,3) float32 tensor on CPU
//     // torch::Tensor xyz_cpu = pts.cpu().contiguous();
//     // // 1) Wrap the CPU tensor as a Python torch.Tensor
//     // py::object torch_tensor_py = py::cast(xyz_cpu);
//     // // 2) Call its .numpy() in Python to get a NumPy array
//     // py::object np_array = torch_tensor_py.attr("numpy")();
//     // // 3) Build a SimpleNamespace(points=ndarray) so .points works in Python
//     // py::object SimpleNS = py::module_::import("types").attr("SimpleNamespace");
//     // py::object pcd_obj  = SimpleNS(py::arg("points") = np_array);
//     // py::object bounding_py = decide_main_bounding(
//     //     py::arg("bound_mode")      = "pcd",   // or "default" / "camera_max" …
//     //     py::arg("pcd_density_rate")= 0.1,
//     //     py::arg("bound_scale")     = 1.0,
//     //     py::arg("pcd")             = pcd_obj  // just needs .points attribute-like
//     // );
//     // auto arr = bounding_py.cast<py::array_t<float>>();
//     // py::buffer_info info = arr.request();
//     // //    shape should be [2,3] here
//     // std::vector<int64_t> shape{ (int64_t)info.shape[0],
//     //                             (int64_t)info.shape[1] };
//     // auto options = torch::TensorOptions()
//     //                 .dtype(torch::kFloat32)
//     //                 .device(torch::kCPU);
//     // torch::Tensor bounding_cpu = torch::from_blob(
//     //     info.ptr, shape, options);
//     // // 3) clone + move to your GPU (or whatever device_type_ is)
//     // torch::Tensor bounding = bounding_cpu.clone()
//     //                             .to(device_type_);
//     // torch::Tensor scene_min = bounding_cpu[0].to(device_type_);
//     // torch::Tensor scene_max = bounding_cpu[1].to(device_type_);
//     // this->scene_center_ = (scene_min + scene_max) * 0.5f;
//     // float extent_side  = (scene_max - scene_min).max().item<float>();
//     // // extent_side *= 3.0f;
//     // this->scene_extent_       = torch::tensor({extent_side},
//     //                              torch::TensorOptions().dtype(torch::kFloat32)
//     //                                                     .device(device_type_));

//     std::cout << "[Debug] scene_extent = " <<  this->scene_extent_ << std::endl;
//     // ─── 3) Call into Python’s SparseVoxelModel.model_init() ────────────
//     py::gil_scoped_acquire gil;
//     static py::module svm_mod = []{
//         py::module sys = py::module::import("sys");
//         sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
//         return py::module::import("src.sparse_voxel_model");
//     }();
//     py::object SVM     = svm_mod.attr("SparseVoxelModel");
//     py_->svm = SVM(
//         py::arg("sh_degree")        = max_sh_degree_,
//         py::arg("black_background") = true
//     );
    
//     py_->svm.attr("points_init")(
//         py::arg("scene_center")      = py::cast(this->scene_center_),
//         py::arg("scene_extent")      = py::cast(this->scene_extent_),
//         py::arg("xyz")               = py::cast(pts),              // already on CUDA
//         py::arg("expected_vox_size") = 0.05f,      
//         py::arg("density")           = 0.1f,
//         py::arg("rgb")               = py::cast(cols),              // already on CUDA
//         py::arg("shs")               = 0.0
//     );

//     // ─── 4) Pull back all core tensors from the Python object ───────────
//      auto fetch = [&](const char* name){
//         return py_->svm.attr(name).cast<torch::Tensor>().contiguous();
//     };
//     this->oct_path_      = fetch("octpath");
//     this->oct_level_     = fetch("octlevel");
//     this->center_        = fetch("vox_center");
//     this->size_          = fetch("vox_size").squeeze(1);
//     this->vox_size_inv_  = 1.0f / size_;
//     this->grid_pts_key_  = fetch("grid_pts_key");
//     this->vox_key_       = fetch("vox_key");      
//     // ─── 5) Copy over the learnable fields ─────────────────────────────
//     this->_geo_grid_pts_ = fetch("_geo_grid_pts");
//     this->sh0_ = fetch("_sh0").view({-1,1,3});
//     this->shs_ = fetch("_shs");
//     this->subdiv_p_ = fetch("_subdiv_p");
//     // subdiv_p_   .retain_grad();
//     // ─── 6) Stats buffers (exactly as before) ───────────────────────────
//     this->max_w_ = torch::zeros({center_.size(0), 1},
//         torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

//     py::module_ np = py::module_::import("numpy");
//     // save raw point‐cloud + colors
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_pts.npy",
//                     tensor_to_numpy(pts.cpu()));
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_cols.npy",
//                     tensor_to_numpy(cols.cpu()));
//     // save voxel cell centres + edge lengths
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_vox_centers.npy",
//                     tensor_to_numpy(center_.cpu()));
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_vox_size.npy",
//                     tensor_to_numpy(size_.cpu()));
//     torch::Tensor sh0_cpu = sh0_.view({-1,3}).cpu().contiguous();
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_vox_color.npy",
//                     tensor_to_numpy(sh0_cpu));

//     VOXEL_MODEL_TENSORS_TO_VEC
// }

// void VoxelModel::createFromPcd(
//     const std::map<point3D_id_t, Point3D>& pcd)
// {
//     // '''
//     // offline approach points_init
//     // '''
//     std::cout << "VoxelModel::createFromPcd() called with "
//               << pcd.size() << " points." << std::endl;
//     namespace py = pybind11;

//     /* 1) copy PCD → CUDA (unchanged) */
//     const int N_pts = static_cast<int>(pcd.size());
//     torch::Tensor pts  = torch::zeros({N_pts,3}, torch::kFloat32).to(device_type_);
//     torch::Tensor cols = torch::zeros({N_pts,3}, torch::kFloat32).to(device_type_);
//     {
//         int i = 0;
//         for (const auto& kv : pcd) {
//             const Point3D& p = kv.second;
//             pts[i]  = torch::tensor({p.xyz_(0), p.xyz_(1), p.xyz_(2)},
//                                     torch::kFloat32).to(device_type_);
//             cols[i] = torch::tensor({p.color_(0), p.color_(1), p.color_(2)},
//                                     torch::kFloat32).to(device_type_);
//             ++i;
//         }
//     }

//     // // ─── 2) Use KNOWN inflated bounding directly (no file, no Python utils) ──
//     // auto opts_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
//     // torch::Tensor bounding_cpu = torch::empty({2,3}, opts_cpu);
//     // // Row 0 = min, Row 1 = max
//     // bounding_cpu[0][0] = -10.748001f;  bounding_cpu[0][1] = -29.842281f;  bounding_cpu[0][2] = -0.293899f;
//     // bounding_cpu[1][0] =  56.780504f;  bounding_cpu[1][1] =  13.760179f;  bounding_cpu[1][2] = 66.683914f;

//     // // Move to your device
//     // torch::Tensor bounding = bounding_cpu.to(device_type_);
//     // torch::Tensor scene_min = bounding_cpu[0].to(device_type_);
//     // torch::Tensor scene_max = bounding_cpu[1].to(device_type_);
//     // this->scene_center_ = 0.5f * (scene_min + scene_max);
//     // float extent_side  = (scene_max - scene_min).max().item<float>();
//     // this->scene_extent_       = torch::tensor({extent_side},
//     //                              torch::TensorOptions().dtype(torch::kFloat32)
//     //                                                     .device(device_type_));
//     // float float_level = -std::log2(0.05f / extent_side);
//     // int level = std::round(float_level);
//     // std::cout << "level = " << level
//     //           << " (float_level = " << float_level << ")" << std::endl;

//     auto mn = std::get<0>(pts.min(0));
//     auto mx = std::get<0>(pts.max(0));
//     float raw_extent     = (mx - mn).max().item<float>();
//     float padded_extent = raw_extent * 1.01f;   // ← 1% padding
//     this->scene_center_ = (mn + mx) * 0.5f;
//     this->scene_extent_ = torch::tensor({padded_extent},
//                                  torch::TensorOptions().dtype(torch::kFloat32)
//                                                         .device(device_type_));

//     std::cout << "[Debug] scene_extent = " <<  this->scene_extent_ << std::endl;
//     // ─── 3) Call into Python’s SparseVoxelModel() ────────────
//     py::gil_scoped_acquire gil;
//     static py::module svm_mod = []{
//         py::module sys = py::module::import("sys");
//         sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
//         return py::module::import("src.sparse_voxel_model");
//     }();
//     py::object SVM     = svm_mod.attr("SparseVoxelModel");
//     py_->svm = SVM(
//         py::arg("sh_degree")        = max_sh_degree_,
//         py::arg("black_background") = true
//     );
    
//     py_->svm.attr("points_init")(
//         py::arg("scene_center")      = py::cast(this->scene_center_),
//         py::arg("scene_extent")      = py::cast(this->scene_extent_),
//         py::arg("xyz")               = py::cast(pts),              // already on CUDA
//         py::arg("expected_vox_size") = 0.05f,      
//         py::arg("density")           = -10.0f,
//         py::arg("rgb")               = py::cast(cols),              // already on CUDA
//         py::arg("shs")               = 0.0
//     );

//     // ─── 4) Pull back all core tensors from the Python object ───────────
//      auto fetch = [&](const char* name){
//         return py_->svm.attr(name).cast<torch::Tensor>().contiguous();
//     };
//     this->oct_path_      = fetch("octpath");
//     this->oct_level_     = fetch("octlevel");
//     this->center_        = fetch("vox_center");
//     this->size_          = fetch("vox_size").squeeze(1);
//     this->vox_size_inv_  = 1.0f / size_;
//     this->grid_pts_key_  = fetch("grid_pts_key");
//     this->vox_key_       = fetch("vox_key");      
//     // ─── 5) Copy over the learnable fields ─────────────────────────────
//     this->_geo_grid_pts_ = fetch("_geo_grid_pts");
//     this->sh0_ = fetch("_sh0").view({-1,1,3});
//     this->shs_ = fetch("_shs");
//     this->subdiv_p_ = fetch("_subdiv_p");
//     // subdiv_p_   .retain_grad();
//     // ─── 6) Stats buffers (exactly as before) ───────────────────────────
//     this->max_w_ = torch::zeros({center_.size(0), 1},
//         torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

//     py::module_ np = py::module_::import("numpy");
//     // save raw point‐cloud + colors
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_pts.npy",
//                     tensor_to_numpy(pts.cpu()));
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_cols.npy",
//                     tensor_to_numpy(cols.cpu()));
//     // save voxel cell centres + edge lengths
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_vox_centers.npy",
//                     tensor_to_numpy(center_.cpu()));
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_vox_size.npy",
//                     tensor_to_numpy(size_.cpu()));
//     torch::Tensor sh0_cpu = sh0_.view({-1,3}).cpu().contiguous();
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_vox_color.npy",
//                     tensor_to_numpy(sh0_cpu));

//     VOXEL_MODEL_TENSORS_TO_VEC
// }

void VoxelModel::increasePcd(
    std::vector<float>                                 pts_vec,
    std::vector<float>                                 col_vec,
    int                                                /*iteration*/
    // const std::vector<std::shared_ptr<VoxelKeyframe>>& /*unused*/
)
{
    // 1) Sanity
    if (pts_vec.empty()) {
        std::cout << "[DEBUG] increasePcd: no new points, returning\n";
        return;
    }
    TORCH_CHECK((pts_vec.size() == col_vec.size()) && (pts_vec.size() % 3 == 0),
                "increasePcd(): points/colours size mismatch");
    const int N = int(pts_vec.size()/3);
    // std::cout << "[DEBUG] increasePcd: N_new_pts=" << N << "\n";

    // 2) Wrap new points & colours on CUDA
    auto xyz_new = torch::from_blob(pts_vec.data(), {N,3}, torch::kFloat32)
                       .to(device_type_);
    auto rgb_new = torch::from_blob(col_vec.data(), {N,3}, torch::kFloat32)
                       .to(device_type_) / 255.0f;

    // torch::Tensor inside =
    //     (xyz_new >= bb_min_eff_).all(1) & (xyz_new <= bb_max_eff_).all(1);
    // int64_t n_oob = (~inside).sum().item<int64_t>();
    // if (n_oob > 0) {
    //     std::cout << "[increasePcd] " << n_oob << " / " << xyz_new.size(0)
    //             << " points outside EFFECTIVE grid AABB\n";
    // }

    // std::cout << "[DEBUG] old scene_center=" << this->scene_center
    //             << "  extent=" << this->scene_extent << "\n";
    // // 3) Split into inside vs outside the *old* AABB
    // {
    //     // bounds = [scene_center ± 0.5*scene_extent]
    //     auto half_ext = 0.5 * this->scene_extent;
    //     auto min_b    = this->scene_center - half_ext;  // [3]
    //     auto max_b    = this->scene_center + half_ext;  // [3]

    //     // per-point mask
    //     auto ge_min   = xyz_new.ge(min_b);
    //     auto le_max   = xyz_new.le(max_b);
    //     auto inside   = (ge_min.logical_and(le_max)).all(1);  // [N]
    //     auto outside  = inside.logical_not();

    //     auto idxs     = outside.nonzero().view(-1);  // [M]
    //     const int M   = int(idxs.size(0));
    //     if (M == 0) {
    //         std::cout << "[DEBUG] increasePcd: all new pts inside old AABB, skipping\n";
    //         return;
    //     }

    //     // select only the outside points
    //     xyz_new = xyz_new.index_select(0, idxs);
    //     rgb_new = rgb_new.index_select(0, idxs);
    //     std::cout << "[DEBUG] increasePcd: M_outside=" << M << "\n";
    // }

    // // 4) Expand your AABB to include those outside points
    // {
    //     // recompute AABB from old bounds + xyz_new
    //     auto all_pts = torch::cat(torch::TensorList{ this->scene_center.unsqueeze(0), xyz_new }, 0);
    //     // but we really need the true min/max over world coords:
    //     auto mn      = std::get<0>(all_pts.min(0));
    //     auto mx      = std::get<0>(all_pts.max(0));
    //     float raw    = (mx - mn).max().item<float>();
    //     float pad    = raw * 1.01f;  // 1% padding

    //     this->scene_center = (mn + mx) * 0.5f;   // [3]
    //     this->scene_extent = torch::tensor({pad},
    //                                torch::TensorOptions()
    //                                  .dtype(torch::kFloat32)
    //                                  .device(device_type_));
    //     std::cout << "[DEBUG] new scene_center=" << this->scene_center
    //               << "  extent=" << this->scene_extent << "\n";
    // }
    // torch::cuda::synchronize();

    // // 5) Call into Python’s SparseVoxelModel.points_init() just on those outside pts
    // py::object svm;
    // {
    //     py::gil_scoped_acquire gil;
    //     static py::object SVM = []{
    //         auto sys = py::module::import("sys");
    //         sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
    //         return py::module::import("src.sparse_voxel_model")
    //                    .attr("SparseVoxelModel");
    //     }();
    //     svm = SVM(py::arg("sh_degree")        = max_sh_degree_,
    //               py::arg("black_background") = true);

    //     // choose the same init‐level you used in createFromPcd (here: 4)
    //     float expected_vox_size = std::ldexp(this->scene_extent.item<float>(), -6);
    //     svm.attr("points_init")(
    //         py::arg("scene_center")      = py::cast(this->scene_center),
    //         py::arg("scene_extent")      = py::cast(this->scene_extent),
    //         py::arg("xyz")               = py::cast(xyz_new),
    //         py::arg("expected_vox_size") = 0.03f,
    //         py::arg("density")           = 0.1f,
    //         py::arg("rgb")               = py::cast(rgb_new),
    //         py::arg("shs")               = 0.0f
    //     );
    // }
    // torch::cuda::synchronize();

    // // 6) Pull back *only* the new‐voxel tensors
    // auto fetch = [svm,this](const char* name){
    //     py::gil_scoped_acquire gil2;
    //     return svm.attr(name)
    //               .cast<torch::Tensor>()
    //               .contiguous()
    //               .to(device_type_);
    // };
    // auto center_new   = fetch("vox_center");         // [M',3]
    // auto size_new     = fetch("vox_size").squeeze(1);// [M']
    // auto sh0_new      = fetch("_sh0").view({-1,1,3}); // [M',1,3]
    // auto shs_new      = fetch("_shs");               // [M',K,3]
    // auto geo_new      = fetch("_geo_grid_pts");      // [G',1]
    // auto subdiv_new   = fetch("_subdiv_p");          // [M',1]
    // auto grid_new     = fetch("grid_pts_key");       // [G',3]
    // auto vox_key_new  = fetch("vox_key");            // [M',8]
    // auto octpath_new  = fetch("octpath").to(torch::kLong);//[M',1]
    // auto octlevel_new = fetch("octlevel").to(torch::kInt8);//[M',1]

    // // // Move a copy to CPU so we can print it
    // // auto octp_cpu  = octpath_new.view(-1).to(torch::kCPU);
    // // auto octl_cpu  = octlevel_new.view(-1).to(torch::kCPU);
    // // // Print just the first 10 entries (so you don’t drown your console):
    // // const int S = std::min<int>(10, octp_cpu.size(0));
    // // std::cout << "[DEBUG] octpath_new[0.." << S-1 << "]: ";
    // // for (int i = 0; i < S; ++i) {
    // //   std::cout << octp_cpu[i].item<int64_t>() << " ";
    // // }
    // // std::cout << "\n";

    // // std::cout << "[DEBUG] octlevel_new[0.." << S-1 << "]: ";
    // // for (int i = 0; i < S; ++i) {
    // //   std::cout << octl_cpu[i].item<int>() << " ";
    // // }
    // // std::cout << "\n";
    // // // If you want a quick check for overlap with your existing grid:
    // // {
    // //   // Pack old and new (path,level) into a 64-bit key so you can compare.
    // //   auto old_path = this->oct_path_.select(1,0).to(torch::kLong).to(torch::kCPU);
    // //   auto old_lvl  = this->oct_level_.select(1,0).to(torch::kLong).to(torch::kCPU);
    // //   std::unordered_set<int64_t> old_keys;
    // //   old_keys.reserve(old_path.size(0));
    // //   for (int i = 0; i < old_path.size(0); ++i)
    // //     old_keys.insert( (old_path[i].item<int64_t>() << 8) | old_lvl[i].item<int64_t>() );

    // //   int coll = 0;
    // //   for (int i = 0; i < octp_cpu.size(0); ++i) {
    // //     int64_t key = (octp_cpu[i].item<int64_t>() << 8)
    // //                 | octl_cpu[i].item<int64_t>();
    // //     if (old_keys.count(key)) ++coll;
    // //   }
    // //   std::cout << "[DEBUG] octpath/octlevel collisions with existing voxels: "
    // //             << coll << " / " << octp_cpu.size(0) << "\n";
    // // }

    {
        py::gil_scoped_acquire gil;
        auto np = py::module::import("numpy");
        // Convert this batch (already filtered to outside points) to NumPy once
        py::object np_pts     = tensor_to_numpy(xyz_new.cpu());                // Nnew×3
        py::object np_cols    = tensor_to_numpy(rgb_new.cpu());                // Nnew×3
        // py::object np_centers = tensor_to_numpy(center_new.cpu());             // Mnew×3
        // py::object np_sizes   = tensor_to_numpy(size_new.unsqueeze(1).cpu());  // Mnew×1
        // py::object np_voxcols = tensor_to_numpy(sh0_new.view({-1,3}).cpu());   // Mnew×3

        // Helper: append 'batch' to file 'path' (or create it if missing)
        auto append_save = [&](const char* path, const py::object& batch) {
            try {
                py::object old = np.attr("load")(path);
                // Make sure dims line up (especially for sizes)
                py::object cat = np.attr("concatenate")(py::make_tuple(old, batch),
                                                        py::arg("axis") = 0);
                np.attr("save")(path, cat);
            } catch (const py::error_already_set&) {
                // File missing / unreadable -> start with this batch
                np.attr("save")(path, batch);
            }
        };
        // Append this batch to the SAME filenames you already use
        append_save("/home/dimitris/Photo-SLAM/debug_new_pts.npy",          np_pts);
        append_save("/home/dimitris/Photo-SLAM/debug_new_cols.npy",         np_cols);
        // append_save("/home/dimitris/Photo-SLAM/debug_new_vox_centers.npy",  np_centers);
        // append_save("/home/dimitris/Photo-SLAM/debug_new_vox_size.npy",     np_sizes);
        // append_save("/home/dimitris/Photo-SLAM/debug_new_vox_color.npy",    np_voxcols);
    }

    // // 7) Append them into your optimizer & buffers
    // densificationPostfix(geo_new, sh0_new, shs_new, subdiv_new);

    // // how many grid‐pts we had before:
    // int64_t prev_G = grid_pts_key_.size(0);

    // center_       = torch::cat({ center_,   center_new   }, 0);
    // size_         = torch::cat({ size_,     size_new     }, 0);
    // grid_pts_key_ = torch::cat({ grid_pts_key_, grid_new }, 0);

    // vox_key_new.add_(prev_G);
    // vox_key_      = torch::cat({ vox_key_, vox_key_new }, 0);

    // // keep oct_path_/oct_level_ 2D
    // if (oct_path_.dim()==1)  oct_path_  = oct_path_.unsqueeze(1);
    // if (oct_level_.dim()==1) oct_level_ = oct_level_.unsqueeze(1);

    // oct_path_  = torch::cat({ oct_path_,  octpath_new  }, 0);
    // oct_level_ = torch::cat({ oct_level_, octlevel_new }, 0);

    // // 8) Grow stats by M'
    // int64_t Mprime = center_new.size(0);
    // auto zeros = torch::zeros({Mprime,1}, voxel_error_sum_.options());
    // voxel_error_sum_    = torch::cat({ voxel_error_sum_,    zeros }, 0);
    // voxel_hit_count_    = torch::cat({ voxel_hit_count_,    zeros }, 0);
    // xyz_gradient_accum_ = torch::cat({ xyz_gradient_accum_, zeros }, 0);
    // denom_              = torch::cat({ denom_,              zeros }, 0);

    // std::cout << "[increasePcd] +"<< Mprime <<" new voxels → total "<< center_.size(0) <<"\n";
    // c10::cuda::CUDACachingAllocator::emptyCache();
}

// void VoxelModel::increasePcd(
//     std::vector<float>                                 pts_vec,
//     std::vector<float>                                 col_vec,
//     int                                                /*iteration*/,
//     const std::vector<std::shared_ptr<VoxelKeyframe>>& /*unused*/)
// {
//     namespace py = pybind11;

//     // 0) Sanity
//     if (pts_vec.empty()) {
//         std::cout << "[DEBUG] increasePcd(incremental): no new points, returning\n";
//         return;
//     }
//     TORCH_CHECK((pts_vec.size() == col_vec.size()) && (pts_vec.size() % 3 == 0),
//                 "increasePcd(incremental): points/colours size mismatch");
//     const int N = int(pts_vec.size()/3);
//     std::cout << "[DEBUG] increasePcd(incremental): N_new_pts=" << N << "\n";

//     // Wrap batch
//     auto xyz_all = torch::from_blob(pts_vec.data(), {N,3}, torch::kFloat32).to(device_type_);
//     auto rgb_all = torch::from_blob(col_vec.data(), {N,3}, torch::kFloat32).to(device_type_) / 255.0f;

//     // 1) Filter points outside the *old* AABB; if none, return (no growth)
//     auto need_expand = [&]() -> bool {
//         if (!(scene_center.defined() && scene_extent.defined() && scene_extent.numel()==1))
//             return true; // first time
//         auto half_ext = 0.5 * scene_extent;
//         auto min_b    = scene_center - half_ext;
//         auto max_b    = scene_center + half_ext;
//         auto inside   = xyz_all.ge(min_b).logical_and(xyz_all.le(max_b)).all(1);
//         auto outside  = inside.logical_not();
//         auto idxs     = outside.nonzero().view(-1);
//         if (idxs.size(0) == 0) {
//             std::cout << "[DEBUG] increasePcd(incremental): all new pts inside AABB, nothing to add.\n";
//             return false;
//         }
//         // keep only outside (we only need them for logging/extents; the grid comes from model_init)
//         xyz_all = xyz_all.index_select(0, idxs);
//         rgb_all = rgb_all.index_select(0, idxs);
//         std::cout << "[DEBUG] incremental: M_outside=" << idxs.size(0) << "\n";
//         return true;
//     }();
//     if (!need_expand) return;

//     // 2) Expand AABB = union(old AABB, outside-pts AABB) (+1% pad), never shrink
//     {
//         torch::Tensor prev_min, prev_max;
//         if (scene_center.defined() && scene_extent.defined() && scene_extent.numel()==1) {
//             auto half_ext = 0.5 * scene_extent;
//             prev_min = scene_center - half_ext;
//             prev_max = scene_center + half_ext;
//         } else {
//             prev_min = std::get<0>(xyz_all.min(0));
//             prev_max = std::get<0>(xyz_all.max(0));
//         }
//         auto mn_new = std::get<0>(xyz_all.min(0));
//         auto mx_new = std::get<0>(xyz_all.max(0));
//         auto mn = torch::min(prev_min, mn_new);
//         auto mx = torch::max(prev_max, mx_new);

//         float side = (mx - mn).max().item<float>();
//         float pad  = side * 1.01f;

//         scene_center = (mn + mx) * 0.5f;
//         scene_extent = torch::tensor({pad},
//                           torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

//         std::cout << "[DEBUG] incremental: new scene_center=" << scene_center
//                   << "  extent=" << scene_extent << "\n";
//     }
//     torch::cuda::synchronize();

//     // 3) Build a *dense* grid over the expanded bound (uniform level stays the same)
//     py::object svm;
//     torch::Tensor new_octpath, new_octlevel, new_center, new_size, new_grid_pts_key, new_vox_key;
//     torch::Tensor new_geo_pts, new_sh0, new_shs, new_subdiv_p;
//     {
//         py::gil_scoped_acquire gil;
//         static py::object SVM = []{
//             auto sys = py::module::import("sys");
//             sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
//             return py::module::import("src.sparse_voxel_model").attr("SparseVoxelModel");
//         }();
//         svm = SVM(py::arg("sh_degree")        = max_sh_degree_,
//                   py::arg("black_background") = true);

//         auto scene_min = scene_center - 0.5 * scene_extent;
//         auto scene_max = scene_center + 0.5 * scene_extent;
//         auto bounding  = torch::stack({ scene_min, scene_max });

//         // Keep init_n_level the same across calls (resolution consistency)
//         svm.attr("model_init")(
//             py::arg("bounding")       = bounding,
//             py::arg("outside_level")  = 0,
//             py::arg("init_n_level")   = 6,
//             py::arg("init_out_ratio") = 2.0,
//             py::arg("sh_degree_init") = 3,
//             py::arg("geo_init")       = 0.0f,
//             py::arg("sh0_init")       = 0.5f,
//             py::arg("shs_init")       = 0.0f
//         );

//         auto fetch = [&](const char* name){
//             return svm.attr(name).cast<torch::Tensor>().contiguous().to(device_type_);
//         };
//         new_octpath      = fetch("octpath").to(torch::kLong);      // [V',1]
//         new_octlevel     = fetch("octlevel").to(torch::kInt8);     // [V',1] uniform
//         new_center       = fetch("vox_center");                    // [V',3]
//         new_size         = fetch("vox_size").squeeze(1);           // [V']
//         new_grid_pts_key = fetch("grid_pts_key");                  // [G',3] int
//         new_vox_key      = fetch("vox_key");                       // [V',8] index into G'
//         new_geo_pts      = fetch("_geo_grid_pts");                 // [G',1]
//         new_sh0          = fetch("_sh0").view({-1,1,3});           // [V',1,3]
//         new_shs          = fetch("_shs");                          // [V',K,3]
//         new_subdiv_p     = fetch("_subdiv_p");                     // [V',1]
//     }
//     torch::cuda::synchronize();

//     // 4) If no previous grid, just register everything via densificationPostfix
//     if (!(center_.defined() && center_.numel() > 0)) {
//         // Append all new grid points + voxels as the initial model
//         int64_t Gadd = new_grid_pts_key.size(0);
//         int64_t Vadd = new_center.size(0);

//         // Optimizer append (Photo-SLAM style)
//         densificationPostfix(new_geo_pts, new_sh0, new_shs, new_subdiv_p);

//         // Geometry/meta append
//         int64_t prev_G = grid_pts_key_.defined() ? grid_pts_key_.size(0) : 0;
//         if (grid_pts_key_.defined() && grid_pts_key_.numel() > 0)
//             grid_pts_key_ = torch::cat({grid_pts_key_, new_grid_pts_key}, 0);
//         else
//             grid_pts_key_ = new_grid_pts_key;

//         // vox_key indices must be offset by prev_G
//         auto vox_key_adj = new_vox_key + prev_G;

//         if (center_.defined() && center_.numel() > 0) {
//             center_    = torch::cat({center_, new_center}, 0);
//             size_      = torch::cat({size_, new_size}, 0);
//             vox_key_   = torch::cat({vox_key_, vox_key_adj}, 0);
//             oct_path_  = torch::cat({oct_path_, new_octpath}, 0);
//             oct_level_ = torch::cat({oct_level_, new_octlevel}, 0);
//         } else {
//             center_    = new_center;
//             size_      = new_size;
//             vox_key_   = vox_key_adj;
//             oct_path_  = new_octpath;
//             oct_level_ = new_octlevel;
//         }
//         vox_size_inv_ = 1.0f / size_;

//         // Stats
//         auto zeros = torch::zeros({Vadd,1}, new_subdiv_p.options());
//         voxel_error_sum_    = zeros.clone();
//         voxel_hit_count_    = zeros.clone();
//         xyz_gradient_accum_ = zeros.clone();
//         denom_              = zeros.clone();
//         sh0_accum_          = torch::zeros({center_.size(0),3}, center_.options());
//         hit_count_          = torch::zeros({center_.size(0),1}, center_.options());

//         VOXEL_MODEL_TENSORS_TO_VEC;
//         std::cout << "[increasePcd(incremental)] init model → total voxels " << center_.size(0)
//                   << " (+ " << Vadd << ")\n";
//         torch::cuda::synchronize();
//         return;
//     }

//     // 5) Incremental: select only *new* voxels (not present before)
//     // 5.1 Build old voxel key set
//     std::unordered_set<int64_t> old_vox_set;
//     {
//         auto op = oct_path_.view(-1).to(torch::kLong).cpu();
//         auto ol = oct_level_.view(-1).to(torch::kInt8).cpu();
//         old_vox_set.reserve(op.size(0) * 2);
//         for (int i=0; i<op.size(0); ++i) {
//             int64_t key = (op[i].item<int64_t>() << 8) | (int64_t)ol[i].item<int8_t>();
//             old_vox_set.insert(key);
//         }
//     }

//     // 5.2 Mark new voxels = those whose (octpath, level) are not in old set
//     auto np = new_octpath.view(-1).to(torch::kLong).cpu();
//     auto nl = new_octlevel.view(-1).to(torch::kInt8).cpu();
//     std::vector<int64_t> add_vox_idx;
//     add_vox_idx.reserve(np.size(0)/4);
//     for (int j=0; j<np.size(0); ++j) {
//         int64_t key = (np[j].item<int64_t>() << 8) | (int64_t)nl[j].item<int8_t>();
//         if (old_vox_set.find(key) == old_vox_set.end())
//             add_vox_idx.push_back(j);
//     }
//     if (add_vox_idx.empty()) {
//         std::cout << "[DEBUG] incremental: expanded bound created no new voxels (grid matched old). Nothing to add.\n";
//         return;
//     }
//     auto add_vox_t = torch::tensor(add_vox_idx, torch::TensorOptions().dtype(torch::kLong)).to(device_type_);
//     const int64_t Vadd = add_vox_t.size(0);

//     // 6) Collect the grid points needed *only* for these new voxels,
//     //    dedupe, and map to the combined grid (reusing existing grid points).
//     // 6.1 Old grid point map: pack (x,y,z) → old index
//     auto old_gpk_cpu = grid_pts_key_.to(torch::kCPU); // [G,3] int
//     std::unordered_map<int64_t,int64_t> old_gpt_to_idx;
//     old_gpt_to_idx.reserve(old_gpk_cpu.size(0) * 2);

//     auto pack_grid = [](int64_t x, int64_t y, int64_t z) -> int64_t {
//         // bias to handle negatives; supports ~±1M
//         const int64_t off = (1ll<<20);
//         return ((x+off) << 42) | ((y+off) << 21) | (z+off);
//     };
//     {
//         auto acc = old_gpk_cpu.accessor<int32_t,2>();
//         for (int64_t i=0; i<old_gpk_cpu.size(0); ++i) {
//             int64_t key = pack_grid(acc[i][0], acc[i][1], acc[i][2]);
//             old_gpt_to_idx.emplace(key, i);
//         }
//     }

//     // 6.2 Gather candidate grid-point indices from new_vox_key[add_vox]
//     auto add_vox_key = new_vox_key.index_select(0, add_vox_t).to(torch::kCPU); // [Vadd,8]
//     std::unordered_map<int64_t,int64_t> newG_to_combined; // map (index into new G) → combined index
//     newG_to_combined.reserve(add_vox_key.size(0) * 8);

//     std::vector<int64_t> add_grid_idx; // indices into new_grid_pts_key to append
//     {
//         auto gpk_new_cpu = new_grid_pts_key.to(torch::kCPU); // [G',3]
//         auto gacc = gpk_new_cpu.accessor<int32_t,2>();
//         auto vkacc = add_vox_key.accessor<int64_t,2>();
//         int64_t next_combined = grid_pts_key_.size(0); // starting index for appends

//         for (int64_t v=0; v<add_vox_key.size(0); ++v) {
//             for (int c=0; c<8; ++c) {
//                 int64_t g = vkacc[v][c];
//                 if (newG_to_combined.find(g) != newG_to_combined.end()) continue;

//                 int32_t gx = gacc[g][0], gy = gacc[g][1], gz = gacc[g][2];
//                 int64_t key = pack_grid(gx, gy, gz);

//                 auto it_old = old_gpt_to_idx.find(key);
//                 if (it_old != old_gpt_to_idx.end()) {
//                     newG_to_combined[g] = it_old->second; // reuse old grid point
//                 } else {
//                     newG_to_combined[g] = next_combined;  // will append
//                     add_grid_idx.push_back(g);
//                     ++next_combined;
//                 }
//             }
//         }
//     }

//     // 6.3 Build adjusted vox_key that points into the combined grid
//     std::vector<int64_t> vox_key_flat;
//     vox_key_flat.resize(Vadd * 8);
//     {
//         auto vkacc = add_vox_key.accessor<int64_t,2>();
//         for (int64_t v=0; v<Vadd; ++v) {
//             for (int c=0; c<8; ++c) {
//                 int64_t g = vkacc[v][c];
//                 vox_key_flat[size_t(v)*8 + c] = newG_to_combined[g];
//             }
//         }
//     }
//     auto vox_key_add = torch::from_blob(vox_key_flat.data(), {Vadd,8}, torch::kLong).clone().to(device_type_);

//     // 6.4 Select new grid-points tensors for the ones we must append
//     torch::Tensor add_grid_t;
//     if (!add_grid_idx.empty()) {
//         add_grid_t = torch::tensor(add_grid_idx, torch::TensorOptions().dtype(torch::kLong)).to(device_type_);
//     }

//     torch::Tensor grid_pts_key_add, geo_add;
//     if (!add_grid_idx.empty()) {
//         grid_pts_key_add = new_grid_pts_key.index_select(0, add_grid_t); // [Gadd,3]
//         geo_add          = new_geo_pts     .index_select(0, add_grid_t); // [Gadd,1]
//     } else {
//         grid_pts_key_add = torch::empty({0,3}, new_grid_pts_key.options());
//         geo_add          = torch::empty({0,1}, new_geo_pts.options());
//     }
//     const int64_t Gadd = grid_pts_key_add.size(0);

//     // 7) Select per-voxel tensors for the voxels we add
//     auto center_add    = new_center   .index_select(0, add_vox_t); // [Vadd,3]
//     auto size_add      = new_size     .index_select(0, add_vox_t); // [Vadd]
//     auto sh0_add       = new_sh0      .index_select(0, add_vox_t); // [Vadd,1,3]
//     auto shs_add       = new_shs      .index_select(0, add_vox_t); // [Vadd,K,3]
//     auto subdiv_p_add  = new_subdiv_p .index_select(0, add_vox_t); // [Vadd,1]
//     auto octpath_add   = new_octpath  .index_select(0, add_vox_t); // [Vadd,1]
//     auto octlevel_add  = new_octlevel .index_select(0, add_vox_t); // [Vadd,1]

//     // 8) Append to optimizer-managed params (Photo-SLAM style)
//     densificationPostfix(geo_add, sh0_add, shs_add, subdiv_p_add);

//     // 9) Append geometry/meta arrays
//     // 9.1 Append grid points (if any)
//     if (Gadd > 0) {
//         if (grid_pts_key_.defined() && grid_pts_key_.numel()>0)
//             grid_pts_key_ = torch::cat({grid_pts_key_, grid_pts_key_add}, 0);
//         else
//             grid_pts_key_ = grid_pts_key_add;
//     }
//     // 9.2 Append voxel records
//     center_    = torch::cat({center_, center_add}, 0);
//     size_      = torch::cat({size_,   size_add  }, 0);
//     vox_key_   = torch::cat({vox_key_, vox_key_add}, 0);
//     oct_path_  = torch::cat({oct_path_,  octpath_add }, 0);
//     oct_level_ = torch::cat({oct_level_, octlevel_add}, 0);
//     vox_size_inv_ = 1.0f / size_;

//     // 10) Grow stats by Vadd
//     auto zeros = torch::zeros({Vadd,1}, subdiv_p_add.options());
//     voxel_error_sum_    = torch::cat({voxel_error_sum_,    zeros}, 0);
//     voxel_hit_count_    = torch::cat({voxel_hit_count_,    zeros}, 0);
//     xyz_gradient_accum_ = torch::cat({xyz_gradient_accum_, zeros}, 0);
//     denom_              = torch::cat({denom_,              zeros}, 0);
//     sh0_accum_          = torch::cat({sh0_accum_, torch::zeros({Vadd,3}, sh0_accum_.options())}, 0);
//     hit_count_          = torch::cat({hit_count_, torch::zeros({Vadd,1}, hit_count_.options())}, 0);

//     VOXEL_MODEL_TENSORS_TO_VEC;

//     std::cout << "[increasePcd(incremental)] +" << Vadd
//               << " voxels, +" << Gadd << " grid pts → totals V="
//               << center_.size(0) << "  G=" << grid_pts_key_.size(0) << "\n";
//     torch::cuda::synchronize();
// }

// void VoxelModel::increasePcd(
//     std::vector<float>                                 pts_vec,
//     std::vector<float>                                 col_vec,
//     int                                                /*iteration*/,
//     const std::vector<std::shared_ptr<VoxelKeyframe>>& /*unused*/)
// {
//     // 1) Sanity
//     if (pts_vec.empty()) {
//         std::cout << "[DEBUG] increasePcd: no new points, returning\n";
//         return;
//     }
//     TORCH_CHECK((pts_vec.size() == col_vec.size()) && (pts_vec.size() % 3 == 0),
//                 "increasePcd(): points/colours size mismatch");
//     int N = static_cast<int>(pts_vec.size() / 3);
//     std::cout << "[DEBUG] increasePcd: N_new_pts=" << N << "\n";

    // 2) Wrap new points & colors on CUDA
    // auto xyz_new = torch::from_blob(pts_vec.data(), {N,3}, torch::kFloat32)
    //                    .to(device_type_);
    // auto rgb_new = torch::from_blob(col_vec.data(), {N,3}, torch::kFloat32)
    //                    .to(device_type_) / 255.0f;

//     // 3) Recompute scene bounds over ALL existing + new pts
//     {
//         auto all_pts = torch::cat(torch::TensorList{ center_, xyz_new }, /*dim=*/0);
//         auto mn      = std::get<0>(all_pts.min(0));
//         auto mx      = std::get<0>(all_pts.max(0));
//         float raw    = (mx - mn).max().item<float>();
//         float pad    = raw * 1.01f;
//         scene_center = (mn + mx) * 0.5f;
//         scene_extent = torch::tensor({pad},
//                             torch::TensorOptions()
//                                 .dtype(torch::kFloat32)
//                                 .device(device_type_));
//         std::cout << "[DEBUG] new scene_center=" << scene_center
//                   << "  extent=" << scene_extent << "\n";
//     }
//     torch::cuda::synchronize();
    
//     std::cout << " expected_vox_size=" << std::ldexp(scene_extent.item<float>(), -6)
//               << "\n";
//     // 4) Call into Python to initialize voxels for all NEW points
//     namespace py = pybind11;

//     // Create SVM instance and run points_init while holding the GIL.
//     py::object svm;
//     {
//         py::gil_scoped_acquire gil;
//         static py::object SVM = []{
//             auto sys = py::module::import("sys");
//             sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
//             return py::module::import("src.sparse_voxel_model")
//                     .attr("SparseVoxelModel");
//         }();

//         svm = SVM(py::arg("sh_degree")        = max_sh_degree_,
//                 py::arg("black_background") = true);

//         svm.attr("points_init")(
//             py::arg("scene_center")      = py::cast(scene_center),
//             py::arg("scene_extent")      = py::cast(scene_extent),
//             py::arg("xyz")               = py::cast(xyz_new),
//             py::arg("expected_vox_size") = 0.05f,
//             py::arg("density")           = 0.1f,
//             py::arg("rgb")               = py::cast(rgb_new),
//             py::arg("shs")               = 0.0f
//         );
//     } // GIL is automatically released here

//     torch::cuda::synchronize();

//     // 5) Fetch back all voxel outputs from Python, capturing svm by copy
//     auto fetch = [svm,&device_type_=this->device_type_](const char* name){
//         py::gil_scoped_acquire gil2;
//         return svm.attr(name)
//                   .cast<torch::Tensor>()
//                   .contiguous()
//                   .to(device_type_);
//     };
//     auto center_new   = fetch("vox_center");       // [V',3]
//     auto size_new     = fetch("vox_size").squeeze(1); // [V']
//     auto sh0_new      = fetch("_sh0").view({-1,1,3}); // [V',1,3]
//     auto shs_new      = fetch("_shs");             // [V',K,3]
//     auto geo_new      = fetch("_geo_grid_pts");    // [G',1]
//     auto subdiv_new   = fetch("_subdiv_p");        // [V',1]
//     auto grid_new     = fetch("grid_pts_key");     // [G',3]
//     auto vox_key_new  = fetch("vox_key");          // [V',8]
//     auto octpath_new  = fetch("octpath").to(torch::kLong);  // [V',1]
//     auto octlevel_new = fetch("octlevel").to(torch::kInt8); // [V',1]

//     torch::cuda::synchronize();

//     // 5b) Sanitize newly created voxel tensors before using them
//     auto isfinite_rows = [&](const torch::Tensor& t, int64_t flat_last_dims) {
//         // flattens the last `flat_last_dims` dims per-voxel, then checks all finite
//         auto v = t;
//         if (flat_last_dims > 0) {
//             std::vector<int64_t> shape = t.sizes().vec();
//             int64_t vox = shape[0];
//             int64_t rest = 1;
//             for (int i = 1; i < (int)shape.size(); ++i) rest *= shape[i];
//             v = t.view({vox, rest});
//         }
//         return torch::isfinite(v).all(1);
//     };

//     auto ok_center = isfinite_rows(center_new, /*flat_last_dims=*/1);        // [V']
//     auto ok_size   = torch::isfinite(size_new) & size_new.gt(1e-8);          // [V']
//     auto ok_sh0    = isfinite_rows(sh0_new,   /*flat_last_dims=*/2);         // [V']
//     auto ok_shs    = isfinite_rows(shs_new,   /*flat_last_dims=*/2);         // [V']
//     auto ok_all    = ok_center & ok_size & ok_sh0 & ok_shs;

//     // If anything bad, drop it now (prevents poison entering the model)
//     if (ok_all.sum().item<int64_t>() != ok_all.size(0)) {
//         auto keep = torch::nonzero(ok_all).view(-1).to(device_type_);
//         center_new   = center_new.index_select(0, keep);
//         size_new     = size_new  .index_select(0, keep);
//         sh0_new      = sh0_new   .index_select(0, keep);
//         shs_new      = shs_new   .index_select(0, keep);
//         geo_new      = geo_new   .index_select(0, keep);
//         subdiv_new   = subdiv_new.index_select(0, keep);
//         grid_new     = grid_new  .index_select(0, keep);
//         vox_key_new  = vox_key_new.index_select(0, keep);
//         octpath_new  = octpath_new.index_select(0, keep);
//         octlevel_new = octlevel_new.index_select(0, keep);

//         std::cout << "[DEBUG] sanitize: dropped "
//                 << (ok_all.size(0) - keep.size(0)) << " invalid voxels\n";
//     }

//     // 6) Identify genuinely new voxels via octree-key collision
//     auto make_key = [](torch::Tensor path, torch::Tensor lvl){
//         auto p = path.to(torch::kLong).view(-1);
//         auto l = lvl .to(torch::kLong).view(-1);
//         auto p8 = p.bitwise_left_shift((int64_t)8);
//         return p8 + l;
//     };

//     auto old_path = (oct_path_.dim()==1)  ? oct_path_.unsqueeze(1)  : oct_path_;
//     auto old_lvl  = (oct_level_.dim()==1) ? oct_level_.unsqueeze(1) : oct_level_;

//     auto old_keys = make_key(old_path,  old_lvl).cpu();
//     auto new_keys = make_key(octpath_new, octlevel_new).cpu();

//     std::unordered_set<int64_t> seen;
//     seen.reserve(old_keys.size(0));
//     for (int i = 0; i < old_keys.size(0); ++i)
//         seen.insert(old_keys[i].item<int64_t>());

//     std::vector<int64_t> idxs;
//     idxs.reserve(new_keys.size(0));
//     for (int i = 0; i < new_keys.size(0); ++i) {
//         int64_t k = new_keys[i].item<int64_t>();
//         if (!seen.count(k)) idxs.push_back(i);
//     }
//     int M = (int)idxs.size();
//     if (M == 0) {
//         std::cout << "[DEBUG] increasePcd: no brand‑new voxels, skipping\n";
//         return;
//     }

//     // 7) Gather only those brand‑new entries
//     auto sel = torch::from_blob(idxs.data(), {M}, torch::kLong)
//                    .to(device_type_);
//     auto center_add   = center_new .index_select(0, sel);
//     auto size_add     = size_new   .index_select(0, sel);
//     auto sh0_add      = sh0_new    .index_select(0, sel);
//     auto shs_add      = shs_new    .index_select(0, sel);
//     auto geo_add      = geo_new    .index_select(0, sel);
//     auto subdiv_add   = subdiv_new .index_select(0, sel);
//     auto grid_add     = grid_new   .index_select(0, sel);
//     auto vox_key_add  = vox_key_new.index_select(0, sel);
//     auto octpath_add  = octpath_new.index_select(0, sel);
//     auto octlevel_add = octlevel_new.index_select(0, sel);

//     torch::cuda::synchronize();

//     // 8) Extend your trainable nets
//     densificationPostfix(geo_add, sh0_add, shs_add, subdiv_add);
//     torch::cuda::synchronize();

//     // 9) Append into all internal buffers
//     int64_t prev_G = grid_pts_key_.size(0);

//     center_       = torch::cat(torch::TensorList{ center_,       center_add  }, /*dim=*/0);
//     size_         = torch::cat(torch::TensorList{ size_,         size_add    }, /*dim=*/0);
//     grid_pts_key_ = torch::cat(torch::TensorList{ grid_pts_key_, grid_add    }, /*dim=*/0);

//     vox_key_add.add_(prev_G);
//     vox_key_ = torch::cat(torch::TensorList{ vox_key_, vox_key_add }, /*dim=*/0);

//     if (oct_path_.dim()  == 1) oct_path_  = oct_path_.unsqueeze(1);
//     if (oct_level_.dim() == 1) oct_level_ = oct_level_.unsqueeze(1);

//     oct_path_  = torch::cat(torch::TensorList{ oct_path_,  octpath_add  }, /*dim=*/0);
//     oct_level_ = torch::cat(torch::TensorList{ oct_level_, octlevel_add }, /*dim=*/0);

//     torch::cuda::synchronize();
//     std::cout << "[DEBUG] appended M=" << M
//               << "  total_vox=" << center_.size(0) << "\n";

//     // 10) Grow stats by M
//     auto zeros = torch::zeros({M,1}, voxel_error_sum_.options());
//     voxel_error_sum_    = torch::cat(torch::TensorList{ voxel_error_sum_,    zeros }, /*dim=*/0);
//     voxel_hit_count_    = torch::cat(torch::TensorList{ voxel_hit_count_,    zeros }, /*dim=*/0);
//     xyz_gradient_accum_ = torch::cat(torch::TensorList{ xyz_gradient_accum_, zeros }, /*dim=*/0);
//     denom_              = torch::cat(torch::TensorList{ denom_,              zeros }, /*dim=*/0);

//     std::cout << "[increasePcd] +"<< M <<" new voxels → total "<< center_.size(0) <<"\n";
//     c10::cuda::CUDACachingAllocator::emptyCache();
// }

// void VoxelModel::increasePcd(std::vector<float>               pts_vec,
//                              std::vector<float>               col_vec,
//                              int /*iteration*/,
//                              const std::vector<std::shared_ptr<VoxelKeyframe>>& /*unused*/)
// {
//     /* ──────────────────── sanity ──────────────────── */
//     if (pts_vec.empty()) return;
//     TORCH_CHECK(pts_vec.size() == col_vec.size() && pts_vec.size() % 3 == 0,
//                 "increasePcd(): points / colours size mismatch");
//     const int M = static_cast<int>(pts_vec.size()/3);

//     /* ──────────────────── build CUDA tensors ─────────────────── */
//     torch::Tensor xyz = torch::from_blob(pts_vec.data(), {M,3},
//                          torch::kFloat32).to(device_type_);
//     torch::Tensor rgb = torch::from_blob(col_vec.data(), {M,3},
//                          torch::kFloat32).to(device_type_);

//     /* ──────────────────── octree encoding ────────────────────── */
//     torch::Tensor octlevel = torch::full({M,1}, leaf_level_,   // stored in createFromPcd()
//                          torch::TensorOptions().dtype(torch::kInt8)
//                                              .device(device_type_));

//     namespace py = pybind11;
//     py::gil_scoped_acquire gil;
//     static py::object octree = []{
//         py::module sys = py::module::import("sys");
//         sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
//         return py::module::import("src.utils.octree_utils");
//     }();

//     torch::Tensor octpath = octree.attr("xyz_2_octpath")(
//                                 xyz, octlevel,
//                                 scene_center, scene_extent
//                             ).cast<torch::Tensor>().to(device_type_);

//     /* ── collapse duplicates inside the *batch* (colours averaged) ── */
//     torch::Tensor key_mat = torch::cat({octpath.to(torch::kLong).view({-1,1}),
//                                         octlevel.to(torch::kLong)}, 1); // (M,2)
//     auto uq = at::unique_dim(key_mat.cpu(), /*dim=*/0, /*sorted=*/false,
//                              /*return_inverse=*/true, /*return_counts=*/true);
//     torch::Tensor uniq_keys = std::get<0>(uq).to(device_type_, torch::kLong);   // (B,2)
//     torch::Tensor inv_id    = std::get<1>(uq).to(device_type_, torch::kLong);   // (M)
//     torch::Tensor counts    = std::get<2>(uq).to(device_type_, torch::kFloat32)
//                                         .unsqueeze(1).to(device_type_);         // (B,1)
//     const int64_t B = uniq_keys.size(0);

//     torch::Tensor rgb_sum = torch::zeros({B,3}, rgb.options());
//     rgb_sum.index_add_(0, inv_id, rgb);
//     torch::Tensor rgb_avg = rgb_sum / counts;                                   // (B,3)

//     torch::Tensor up_octpath  = uniq_keys.select(1,0).view({-1,1});
//     torch::Tensor up_octlevel = uniq_keys.select(1,1)
//                                           .to(torch::kInt8).view({-1,1});

//     /* ──────────────────── split into old / new voxels ──────────────────── */
//     auto pack_key = [](const torch::Tensor& path, const torch::Tensor& lvl){
//         return (path.to(torch::kLong).bitwise_left_shift(8) +
//                 lvl.to(torch::kLong));
//     };
//     torch::Tensor packed = pack_key(up_octpath, up_octlevel);                   // (B,1)
//     std::vector<int64_t> new_rows_cpu;  new_rows_cpu.reserve(B);
//     std::vector<int64_t> new_keys_cpu;  new_keys_cpu.reserve(B);

//     for (int i = 0; i < B; ++i)
//     {
//         int64_t k = packed[i].item<int64_t>();
//         auto it = voxel_hash_.find(k);
//         if (it == voxel_hash_.end()) {                    // *** brand‑new voxel ***
//             new_rows_cpu.push_back(i);
//             new_keys_cpu.push_back(k);
//         } else {                                          // *** existing voxel  ***
//             int j = it->second;                           // row in tensors
//             /* simple running average of DC colour */
//             auto rgb_i = rgb_avg[i];
//             auto sh_i  = sh_utils::RGB2SH(rgb_i).view({1,3});
//             sh0_[j] = 0.5f * (sh0_[j] + sh_i);
//             voxel_hit_count_[j] += 1;
//         }
//     }
//     const int Nnew = static_cast<int>(new_rows_cpu.size());
//     if (Nnew == 0) return;                                // nothing to add

//     torch::Tensor sel = torch::from_blob(new_rows_cpu.data(), {Nnew},
//                                          torch::kLong).to(device_type_);
//     torch::Tensor new_octpath  = up_octpath .index_select(0, sel);
//     torch::Tensor new_octlevel = up_octlevel.index_select(0, sel);
//     torch::Tensor new_rgb      = rgb_avg    .index_select(0, sel);

//     /* ──────────────────── decode to centre / size ─────────────────────── */
//     auto dec = octree.attr("octpath_decoding")(
//                    new_octpath, new_octlevel,
//                    scene_center, scene_extent
//                ).cast<std::tuple<torch::Tensor,torch::Tensor>>();
//     torch::Tensor new_center = std::get<0>(dec).to(device_type_);              // (N,3)
//     torch::Tensor new_size   = std::get<1>(dec).squeeze(1).to(device_type_);   // (N)

//     /* ─────────────────── grid‑point links (duplicates allowed) ─────────── */
//     py::tuple link = octree.attr("build_grid_pts_link")(new_octpath,
//                                                         new_octlevel);
//     torch::Tensor new_grid_pts = link[0].cast<torch::Tensor>().to(device_type_);
//     torch::Tensor new_vox_key  = link[1].cast<torch::Tensor>().to(device_type_);

//     /* ─────────────────── allocate trainables for the newcomers ─────────── */
//     /* geo */
//     torch::Tensor new_geo = torch::full({new_grid_pts.size(0),1}, kGeoInit,
//                            torch::kFloat32).to(device_type_);
//     /* sh‑0 (DC colour) */
//     torch::Tensor new_sh0 = sh_utils::RGB2SH(new_rgb).view({-1,1,3});
//     /* sh‑rest */
//     int K = (max_sh_degree_+1)*(max_sh_degree_+1) - 1;
//     torch::Tensor new_shs = torch::full({Nnew,K,3}, kShsInit,
//                            torch::kFloat32).to(device_type_);
//     /* subdivision priority */
//     torch::Tensor new_subdiv = torch::ones({Nnew,1},
//                            torch::kFloat32).to(device_type_);

//     /* append to optimiser ► updates sh0_, shs_, _geo_grid_pts_, subdiv_p_ */
//     densificationPostfix(new_geo, new_sh0, new_shs, new_subdiv);

//     /* ─────────────────── append non‑trainable arrays ──────────────────── */
//     int64_t Vold = center_.size(0);
//     int64_t Gold = grid_pts_key_.size(0);

//     center_        = torch::cat({center_, new_center}, 0);
//     size_          = torch::cat({size_,   new_size},   0);
//     vox_size_inv_  = 1.0f / size_;
//     oct_path_      = torch::cat({oct_path_,  new_octpath.to(torch::kLong)}, 0);
//     oct_level_     = torch::cat({oct_level_, new_octlevel.view({-1})}, 0);

//     /* shift new_vox_key indices to account for the grid_pts already stored */
//     new_vox_key.add_(Gold);
//     vox_key_       = torch::cat({vox_key_,       new_vox_key}, 0);
//     grid_pts_key_  = torch::cat({grid_pts_key_,  new_grid_pts}, 0);

//     /* statistics buffers – enlarge with zeros */
//     auto z_row = torch::zeros({Nnew,1}, voxel_error_sum_.options());
//     voxel_error_sum_ = torch::cat({voxel_error_sum_, z_row.clone()}, 0);
//     voxel_hit_count_ = torch::cat({voxel_hit_count_, z_row.clone()}, 0);
//     xyz_gradient_accum_ = torch::cat({xyz_gradient_accum_, z_row.clone()}, 0);
//     denom_              = torch::cat({denom_,              z_row.clone()}, 0);

//     /* ─────────────────── update hash‑table ─────────────────────────────── */
//     for (int i = 0; i < Nnew; ++i)
//         voxel_hash_[new_keys_cpu[i]] = Vold + i;

//     /* (optional) tiny log */
//     std::cout << "[increasePcd]  +"<<Nnew<<" vox | "
//               << "total "<<center_.size(0)<<" vox, "
//               << grid_pts_key_.size(0)<<" grid‑pts\n";
// }

// void VoxelModel::increasePcd(std::vector<float> pts,
//                              std::vector<float> cols,
//                              const int iteration,
//                              const std::vector<std::shared_ptr<VoxelKeyframe>>& kfs)
// {
//     if (pts.empty()) return;                     // nothing to do

//     /* ───────────── 0.  build CUDA tensors for *all* points  ────────── */
//     const int M_new = static_cast<int>(pts.size() / 3);

//     torch::Tensor xyz_new = torch::from_blob(
//         const_cast<float*>(pts.data()), {M_new,3},
//         torch::TensorOptions().dtype(torch::kFloat32))
//         .to(device_type_);

//     torch::Tensor rgb_new = torch::from_blob(
//         const_cast<float*>(cols.data()), {M_new,3},
//         torch::TensorOptions().dtype(torch::kFloat32))
//         .to(device_type_) / 255.0f;

//     const auto old_v = center_.size(0);
//     const auto new_v = M_new;
//     std::cout << "[increasePcd] #old_voxels=" << old_v
//                 << "  #new_pts="   << new_v << std::endl;
                
//     /* ───────────── 1.  concatenate with *existing* voxel centres ───── */
//     torch::Tensor xyz_all, rgb_all;
//     {
//         // voxel centres live in center_  ⇒  (V,3)
//         torch::Tensor old_xyz = center_;
//         // torch::Tensor old_rgb = sh0_.view({-1,3});
//         // torch::Tensor old_rgb = sh_utils::SH2RGB(sh0_.view({-1, 3}));  
//         torch::Tensor old_rgb = 0.282095f * sh0_.view({-1,3}) + 0.5f;

//         xyz_all = torch::cat({old_xyz, xyz_new}, 0);
//         rgb_all = torch::cat({old_rgb, rgb_new}, 0);

//         std::cout << "[increasePcd] xyz_all.shape=" 
//             << xyz_all.size(0) << "×" << xyz_all.size(1)
//             << "  rgb_all.shape="
//             << rgb_all.size(0) << "×" << rgb_all.size(1)
//             << std::endl;
//     }

//     py::gil_scoped_acquire gil;
//     // ––– 2.a ask Python for a bounding box on *all* points
//     torch::Tensor xyz_cpu = xyz_all.cpu().contiguous();
//     // 1) Wrap the CPU tensor as a Python torch.Tensor
//     py::object torch_tensor_py = py::cast(xyz_cpu);
//     // 2) Call its .numpy() in Python to get a NumPy array
//     py::object np_array = torch_tensor_py.attr("numpy")();
//     // 3) Build a SimpleNamespace(points=ndarray) so .points works in Python
//     py::object SimpleNS = py::module_::import("types").attr("SimpleNamespace");
//     py::object pcd_obj  = SimpleNS(py::arg("points") = np_array);

//     py::list tr_cams;
//     py::list py_cams; 
//     for (auto& kf : kfs)
//     {
//         /* 1.  Position ------------------------------------------------------ */
//         const Eigen::Vector3d& p = kf->t_;               // translation part
//         torch::Tensor pos_cpu = torch::tensor(
//             {static_cast<float>(p.x()),
//             static_cast<float>(p.y()),
//             static_cast<float>(p.z())},
//             torch::TensorOptions().dtype(torch::kFloat32));   // stays on CPU
                                                            
//         /* 2.  Forward (+Z) direction in world space ------------------------- */
//         Eigen::Vector3d fwd_eig = kf->R_quaternion_ * Eigen::Vector3d(0,0,1);
//         fwd_eig.normalize();
//         torch::Tensor look_cpu = torch::tensor(
//             {static_cast<float>(fwd_eig.x()),
//             static_cast<float>(fwd_eig.y()),
//             static_cast<float>(fwd_eig.z())},
//             torch::TensorOptions().dtype(torch::kFloat32));

//         /* 3.  Wrap as NumPy for SimpleNamespace ----------------------------- */
//         py::object np_pos    = py::cast(pos_cpu).attr("numpy")();
//         py::object np_lookat = py::cast(look_cpu).attr("numpy")();

//         tr_cams.append(
//             SimpleNS(py::arg("position") = np_pos,
//                     py::arg("lookat")   = np_lookat)
//         );
//         // build a *real* MiniCam
//         sv::MiniCam mc  = kf->toMiniCam();          // C++ struct
//         py::object cam_py = MiniCam_to_py(mc);      // Python object
//         cam_py.attr("w2c") = cam_py.attr("w2c").attr("cuda")();
//         cam_py.attr("c2w") = cam_py.attr("c2w").attr("cuda")();
//         cam_py.attr("position") = cam_py.attr("c2w")[py::make_tuple(py::slice(0,3,1), 3)];
//         cam_py.attr("lookat")   = cam_py.attr("c2w")[py::make_tuple(py::slice(0,3,1), 2)];
//         py_cams.append(cam_py);                     // keep for model_init()
//     }

//     py::module bu_mod     = py::module_::import("src.utils.bounding_utils");
//     py::object decide_bb  = bu_mod.attr("decide_main_bounding");
//     py::object bb_py      = decide_bb(
//         py::arg("bound_mode")      = "pcd",
//         py::arg("pcd_density_rate")= 0.1,
//         py::arg("bound_scale")     = 1.0,
//         py::arg("pcd")             = pcd_obj
//     );
//     auto arr   = bb_py.cast<py::array_t<float>>();
//     auto info  = arr.request();
//     torch::Tensor bounding = torch::from_blob(
//         info.ptr, {(int64_t)info.shape[0],(int64_t)info.shape[1]},
//         torch::TensorOptions().dtype(torch::kFloat32))
//         .clone().to(device_type_);

//     /* ––– 2.b run `SparseVoxelModel.model_init()` again ––––––––––––––– */
//     py::module svm_mod = py::module_::import("src.sparse_voxel_model");
//     py::object svm = svm_mod.attr("SparseVoxelModel")(
//         py::arg("sh_degree")        = max_sh_degree_,
//         py::arg("black_background") = true
//     );

//     svm.attr("model_init")(
//         py::arg("bounding")        = bounding,
//         py::arg("outside_level")   = 0,
//         py::arg("init_n_level")    = 6,
//         py::arg("init_out_ratio")  = 2,
//         py::arg("sh_degree_init")  = 3,
//         py::arg("geo_init")        = 0.1f,
//         py::arg("sh0_init")        = 0.5f,
//         py::arg("shs_init")        = 0.0f,
//         py::arg("cameras")         = py_cams
//     );

//     /* ───────────── 3.  fetch the *new* grid description  ───────────── */
//     auto fetch = [&](const char* name){
//         return svm.attr(name).cast<torch::Tensor>()
//                   .contiguous().to(device_type_);
//     };
//     torch::Tensor vox_key_new = fetch("vox_key");    // (Vnew,4)  (path,level)
//     torch::Tensor sh0_new     = fetch("_sh0").view({-1,3}); // (Vnew,3)
//     torch::Tensor shs_new     = fetch("_shs");       // same shape as before
//     torch::Tensor geo_new     = fetch("_geo_grid_pts");
//     torch::Tensor subdiv_new  = fetch("_subdiv_p");

//     /* ───────────── 4.  build a map  old-key → index  ──────────────── */
//     std::unordered_map<int64_t,int> old2idx;
//     {
//         torch::Tensor key_flat =
//             vox_key_.select(1,0).to(torch::kLong) * 256 +
//             vox_key_.select(1,1).to(torch::kLong);   // pack path+level
//         auto cpu = key_flat.cpu();
//         for (int i=0;i<cpu.size(0);++i)
//             old2idx[cpu[i].item<int64_t>()] = i;
//     }

//     /* ───────────── 5.  copy old learnables into the new grid ───────── */
//     torch::Tensor key_flat_new =
//         vox_key_new.select(1,0).to(torch::kLong) * 256 +
//         vox_key_new.select(1,1).to(torch::kLong);

//     torch::Tensor hit_count =
//         torch::zeros({key_flat_new.size(0),1}, sh0_new.options());

//     for (int i=0;i<key_flat_new.size(0);++i)
//     {
//         int64_t k = key_flat_new[i].item<int64_t>();
//         auto it = old2idx.find(k);
//         if (it == old2idx.end()) continue;           // brand-new voxel

//         int j = it->second;                          // index in *old* grid
//         // running mean for colour
//         sh0_new[i] = (sh0_new[i] + sh0_.view({-1,3})[j]) * 0.5f;

//         // copy the rest verbatim
//         shs_new[i]        = shs_[j];
//         geo_new[i]        = _geo_grid_pts[j];
//         subdiv_new[i]     = subdiv_p_[j];
//         hit_count[i]      = voxel_hit_count_[j] + 1;
//     }

//     /* ───────────── 6.  replace the internal state  ─────────────────── */
//     vox_key_        = vox_key_new;
//     center_         = fetch("vox_center");
//     size_           = fetch("vox_size").squeeze(1);
//     vox_size_inv_   = 1.0f / size_;
//     oct_path_       = fetch("octpath").to(torch::kLong);
//     oct_level_      = fetch("octlevel").to(torch::kInt8);

//     sh0_            = sh0_new.view({-1,1,3}).detach().requires_grad_(true);
//     shs_            = shs_new.detach().requires_grad_(true);
//     _geo_grid_pts   = geo_new.detach().requires_grad_(true);
//     subdiv_p_       = subdiv_new.detach().requires_grad_(true);

//     // update hit counter (used by colour running mean)
//     voxel_hit_count_= hit_count;

//     /* keep grads */
//     subdiv_p_.retain_grad();
//     subdiv_meta_ = torch::zeros_like(subdiv_p_).requires_grad_(true);
//     subdiv_meta_.retain_grad();

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

void VoxelModel::initOptimizer(float geo_lr, float sh0_lr, float shs_lr,
                               float beta1, float beta2, float eps) {
    py::gil_scoped_acquire gil;
    py::module sadam = py::module::import("svraster_cuda.sparse_adam");
    py::object SparseAdam = sadam.attr("SparseAdam");

    // Three param groups like train.py (only params + lr per group)
    py::list groups;

    {
        py::dict g0;
        g0["params"] = py::make_tuple(py_->svm.attr("_geo_grid_pts"));
        g0["lr"]     = geo_lr;
        groups.append(g0);
    }
    {
        py::dict g1;
        g1["params"] = py::make_tuple(py_->svm.attr("_sh0"));
        g1["lr"]     = sh0_lr;
        groups.append(g1);
    }
    {
        py::dict g2;
        g2["params"] = py::make_tuple(py_->svm.attr("_shs"));
        g2["lr"]     = shs_lr;
        groups.append(g2);
    }

    // SparseAdam(params, lr=..., betas=(...), eps=...)
    // Call with positionals to avoid keyword syntax:
    py_->optimizer_py = SparseAdam(groups,
                              0.0,                              // lr
                              py::make_tuple(beta1, beta2),     // betas
                              eps);                              // eps
}

void VoxelModel::rebuildOptimizer(float geo_lr, float sh0_lr, float shs_lr,
                                  float beta1, float beta2, float eps) {
    // Same as initOptimizer, but you may want to preserve your current LRs.
    initOptimizer(geo_lr, sh0_lr, shs_lr, beta1, beta2, eps);
}

void VoxelModel::optimizerZeroGrad() {
    py::gil_scoped_acquire gil;
    if (!py_->optimizer_py.is_none())
        py_->optimizer_py.attr("zero_grad")(py::arg("set_to_none") = true);
}

void VoxelModel::optimizerStep() {
    py::gil_scoped_acquire gil;
    if (!py_->optimizer_py.is_none())
        py_->optimizer_py.attr("step")();
}

void VoxelModel::setLearningRates(float geo_lr, float sh0_lr, float shs_lr) {
    py::gil_scoped_acquire gil;
    if (py_->optimizer_py.is_none()) return;
    py::list groups = py_->optimizer_py.attr("param_groups");
    auto g0 = groups[0].cast<py::dict>();
    auto g1 = groups[1].cast<py::dict>();
    auto g2 = groups[2].cast<py::dict>();
    g0["lr"] = geo_lr;
    g1["lr"] = sh0_lr;
    g2["lr"] = shs_lr;
}

float VoxelModel::multiStepDecay(int iter, float base_lr,
                            const std::vector<int>& milestones,
                            float gamma /* e.g. 0.33f */)
{
    if (milestones.empty() || gamma <= 0.f || gamma >= 1.f) return base_lr;
    int passed = 0;
    for (int m : milestones) if (m > 0 && iter >= m) ++passed;
    if (passed == 0) return base_lr;
    return base_lr * std::pow(gamma, passed);
}

/* static */ torch::Tensor
VoxelModel::camPosition_(const MiniCam& cam, torch::Device d) {
    // c2w: 4x4 or w2c inv; assume you have cam.c2w as float[4x4] or Tensor
    // position = c2w[0:3,3]
    auto c2w = cam.c2w.to(d).contiguous();              // (4,4)
    return c2w.index({torch::indexing::Slice(0,3), 3}); // (3)
}

/* static */ torch::Tensor
VoxelModel::camForward_(const MiniCam& cam, torch::Device d) {
    // forward = +Z axis of camera in world (c2w[0:3,2]); normalize.
    auto c2w = cam.c2w.to(d).contiguous();
    auto fwd = c2w.index({torch::indexing::Slice(0,3), 2}); // (3)
    auto nrm = fwd.norm().clamp_min(1e-8);
    return fwd / nrm;
}

/* static */ float
VoxelModel::camPixSize_(const MiniCam& cam) {
    // world distance per pixel per unit depth ≈ max(1/fx, 1/fy)
    // (fx,fy) are pixel focal lengths.
    float inv_fx = 1.0f / std::max(1e-8f, cam.fx);
    float inv_fy = 1.0f / std::max(1e-8f, cam.fy);
    return std::max(inv_fx, inv_fy);
}

void VoxelModel::syncFromPython_() {
    py::gil_scoped_acquire gil;

    auto fetch = [&](const char* name){
        return py_->svm.attr(name).cast<torch::Tensor>().contiguous();
    };

    // Octree / indexing
    oct_path_   = fetch("octpath").to(torch::kLong);
    oct_level_  = fetch("octlevel").to(torch::kInt8);
    vox_key_    = fetch("vox_key");               // [N,8] long
    center_     = fetch("vox_center");            // [N,3] float
    size_       = fetch("vox_size").squeeze(1);   // [N] or [N,1]→[N]

    // Learnables
    _geo_grid_pts_ = fetch("_geo_grid_pts");
    sh0_           = fetch("_sh0").view({-1,1,3});
    shs_           = fetch("_shs");
    subdiv_p_      = fetch("_subdiv_p");

    // Resize any side buffers bound to voxel count
    max_w_ = torch::zeros({center_.size(0), 1},
              torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
}

VoxelModel::StatPkg
VoxelModel::computeTrainingStat(const std::vector<MiniCam>& cams) {
    // Mirrors SVAdaptive.compute_training_stat (but uses our renderer)
    freezeVoxGeo();

    const int64_t N = center_.size(0);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_type_);
    auto max_w            = torch::zeros({N,1}, opts);
    auto min_samp_interval= torch::full ({N,1}, 1e30f, opts);
    auto view_cnt         = torch::zeros({N,1}, opts);

    for (const auto& cam : cams) {
        // Render with track_max_w; our wrapper already returns "max_w"
        // (your VoxelModel::render already returns 'max_w' in pkg)
        py::array dummy_rgb; // not needed for weight tracking; safe empty
        auto pkg = render(cam, dummy_rgb);
        if (!pkg.count("max_w") || !pkg.at("max_w").defined())
            continue;

        auto max_w_i = pkg["max_w"].to(device_type_);
        max_w = torch::maximum(max_w, max_w_i);

        // visibility indices for current cam
        auto vis_idx = (max_w_i.squeeze(1) > 0).nonzero().squeeze(1);  // [K]

        if (vis_idx.numel() > 0) {
            // z distance along camera forward
            auto pos   = camPosition_(cam, device_type_);
            auto fwd   = camForward_(cam,   device_type_);
            auto vc    = center_.index({vis_idx});             // [K,3]
            auto zdist = ((vc - pos) * fwd).sum(1, true).abs(); // [K,1] (abs -> guard sign)

            float pix_size = camPixSize_(cam);                 // scalar
            auto samp_itv  = zdist * pix_size;                 // [K,1]

            // min over views
            auto cur = min_samp_interval.index({vis_idx});
            min_samp_interval.index_put_({vis_idx}, torch::minimum(cur, samp_itv));

            // view count
            view_cnt.index_put_({vis_idx}, view_cnt.index({vis_idx}) + 1);
        }
    }

    unfreezeVoxGeo();
    return { max_w.contiguous(), min_samp_interval.contiguous(), view_cnt.contiguous() };
}

void VoxelModel::pruning(const torch::Tensor& prune_mask) {
    py::gil_scoped_acquire gil;
    // accept [N] or [N,1] bool/byte
    auto mask = prune_mask.to(torch::kBool).to(device_type_).contiguous();
    py_->svm.attr("pruning")(mask);
    syncFromPython_();
}

void VoxelModel::subdividing(const torch::Tensor& subdivide_mask) {
    py::gil_scoped_acquire gil;
    auto mask = subdivide_mask.to(torch::kBool).to(device_type_).contiguous();
    py_->svm.attr("subdividing")(mask);
    syncFromPython_();
}

torch::Tensor VoxelModel::subdivisionPriority() const {
    py::gil_scoped_acquire gil;
    return py_->svm.attr("subdivision_priority")
                    .cast<torch::Tensor>()
                    .to(device_type_)
                    .contiguous();
}

torch::Tensor VoxelModel::voxSize() const {
    // ensure [N,1]
    if (size_.dim() == 1) return size_.unsqueeze(1);
    return size_;
}

torch::Tensor VoxelModel::octLevel() const {
    // ensure [N,1] int8
    if (oct_level_.dim() == 1) return oct_level_.unsqueeze(1);
    return oct_level_;
}

int VoxelModel::numVoxels() const {
    return static_cast<int>(center_.size(0));
}

int VoxelModel::maxNumLevels() const {
    return max_num_levels_;
}

void VoxelModel::resetSubdivisionPriority() {
    py::gil_scoped_acquire gil;
    if (py_->svm.attr("reset_subdivision_priority").is_none()) return;
    py_->svm.attr("reset_subdivision_priority")();
}

void VoxelModel::freezeVoxGeo() {
    py::gil_scoped_acquire gil;
    if (py_->svm.attr("freeze_vox_geo").is_none()) return;
    py_->svm.attr("freeze_vox_geo")();
}

void VoxelModel::unfreezeVoxGeo() {
    py::gil_scoped_acquire gil;
    if (py_->svm.attr("unfreeze_vox_geo").is_none()) return;
    py_->svm.attr("unfreeze_vox_geo")();
}

std::unordered_map<std::string, torch::Tensor>
VoxelModel::render(const MiniCam&              cam,
                   const py::array_t<uint8_t>& rgb_image) const
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
    // if (_geo_grid_pts_=! 0.1) {
    //     std::cout << "[INFO] _geo_grid_pts_ " << _geo_grid_pts_ << std::endl;
    //     std::cout << "[INFO] this->_geo_grid_pts_ " << this->_geo_grid_pts_ << std::endl;
    // }
    /* --------------------------------------------------------------------- */
    /* 3) pack voxel buffers into a python dict                              */
    /* --------------------------------------------------------------------- */
    py::dict d;
    // d["geo_grid_pts"] = py::cast(_geo_grid_pts_);
    // d["sh0"]        = py::cast(sh0_);
    // d["shs"]           = py::cast(shs_);
    // d["subdiv_p"]      = py::cast(subdiv_p_);
    // d["octpath"]      = py::cast(oct_path_);
    // d["center"]       = py::cast(center_);
    // d["vox_size"]  = py::cast(size_);
    // d["vox_key"]   = py::cast(vox_key_);
    // d["active_sh_degree"]     = py::int_(active_sh_degree_);
    d["geo_grid_pts"]      = py_->svm.attr("_geo_grid_pts");
    d["sh0"]               = py_->svm.attr("_sh0");
    d["shs"]               = py_->svm.attr("_shs");
    d["subdiv_p"]          = py_->svm.attr("_subdiv_p");        // gradient used as priority
    d["octpath"]           = py_->svm.attr("octpath");
    d["octlevel"]          = py_->svm.attr("octlevel");
    d["center"]            = py_->svm.attr("vox_center");
    d["vox_size"]          = py_->svm.attr("vox_size");
    d["vox_key"]           = py_->svm.attr("vox_key");
    d["active_sh_degree"]  = py::int_(active_sh_degree_);

    /* --------------------------------------------------------------------- */
    /* 4) call python                                                        */
    /* --------------------------------------------------------------------- */
    py::object py_cam  = MiniCam_to_py(cam);
    py::object py_out;
    try {
        py_out = py_render(py_cam, d, rgb_image);
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
    return pkg;                       // may be empty on error
}

float VoxelModel::paramL2(const char* name) {
    py::gil_scoped_acquire gil;
    auto t = py_->svm.attr(name).cast<torch::Tensor>();
    return t.norm().item<float>();
}

float VoxelModel::gradL2(const char* name) {
    py::gil_scoped_acquire gil;
    auto p = py_->svm.attr(name).cast<torch::Tensor>();
    auto g = p.grad(); // C++ API: p.grad() (or from py: p.attr("grad"))
    if (!g.defined()) return 0.f;
    return g.norm().item<float>();
}

} // namespace sv

