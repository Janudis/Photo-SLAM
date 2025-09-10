#include "include_voxel/voxel_model.h"
#include <fstream>
#include <sstream>

namespace py = pybind11;
using namespace py::literals;  // enables "name"_a syntax
namespace sv {

struct __attribute__((visibility("hidden"))) VoxelModel::PyState {
    py::object svm;        // Python SparseVoxelModel
    py::object optimizer_py;  // Python SparseAdam
    py::object scheduler_py; 

    ~PyState() {
        py::gil_scoped_acquire gil;
        svm = py::none();
        optimizer_py = py::none();
        scheduler_py = py::none();
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

void VoxelModel::createFromPcd(const std::map<point3D_id_t, Point3D>& pcd)
{
    // '''
    // dense grid
    // '''
    std::cout << "VoxelModel::createPcd() called with "
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

    // ─── 2) Ask Python to compute the bounding box via decide_main_bounding ──
    py::gil_scoped_acquire gil;
    static py::module bu_mod = []{
        py::module sys = py::module::import("sys");
        sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
        return py::module::import("src.utils.bounding_utils");
    }();
    py::object decide_main_bounding = bu_mod.attr("decide_main_bounding");

    // xyz.mul_(6.0f);
    // Pass the point cloud to Python as a (N,3) float32 tensor on CPU
    torch::Tensor xyz_cpu = xyz.cpu().contiguous();
    // 1) Wrap the CPU tensor as a Python torch.Tensor
    py::object torch_tensor_py = py::cast(xyz_cpu);
    // 2) Call its .numpy() in Python to get a NumPy array
    py::object np_array = torch_tensor_py.attr("numpy")();
    // 3) Build a SimpleNamespace(points=ndarray) so .points works in Python
    py::object SimpleNS = py::module_::import("types").attr("SimpleNamespace");
    py::object pcd_obj  = SimpleNS(py::arg("points") = np_array);

    // py::list tr_cams;
    // py::list py_cams; 

    // for (auto& kf : keyframes)
    // {
    //     /* 1.  Position ------------------------------------------------------ */
    //     const Eigen::Vector3d& p = kf->t_;               // translation part
    //     torch::Tensor pos_cpu = torch::tensor(
    //         {static_cast<float>(p.x()),
    //         static_cast<float>(p.y()),
    //         static_cast<float>(p.z())},
    //         torch::TensorOptions().dtype(torch::kFloat32));   // stays on CPU
                                                            
    //     /* 2.  Forward (+Z) direction in world space ------------------------- */
    //     Eigen::Vector3d fwd_eig = kf->R_quaternion_ * Eigen::Vector3d(0,0,1);
    //     fwd_eig.normalize();
    //     torch::Tensor look_cpu = torch::tensor(
    //         {static_cast<float>(fwd_eig.x()),
    //         static_cast<float>(fwd_eig.y()),
    //         static_cast<float>(fwd_eig.z())},
    //         torch::TensorOptions().dtype(torch::kFloat32));

    //     /* 3.  Wrap as NumPy for SimpleNamespace ----------------------------- */
    //     py::object np_pos    = py::cast(pos_cpu).attr("numpy")();
    //     py::object np_lookat = py::cast(look_cpu).attr("numpy")();

    //     tr_cams.append(
    //         SimpleNS(py::arg("position") = np_pos,
    //                 py::arg("lookat")   = np_lookat)
    //     );
    //     // build a *real* MiniCam
    //     sv::MiniCam mc  = kf->toMiniCam();          // C++ struct
    //     py::object cam_py = MiniCam_to_py(mc);      // Python object
    //     cam_py.attr("w2c") = cam_py.attr("w2c").attr("cuda")();
    //     cam_py.attr("c2w") = cam_py.attr("c2w").attr("cuda")();
    //     cam_py.attr("position") = cam_py.attr("c2w")[py::make_tuple(py::slice(0,3,1), 3)];
    //     cam_py.attr("lookat")   = cam_py.attr("c2w")[py::make_tuple(py::slice(0,3,1), 2)];
    //     py_cams.append(cam_py);                     // keep for model_init()
    // }

    // py::list all_tr_cams = build_tr_cams_from_file_light(cam_pose_txt_path, 1.0f);
    // py::list all_py_cams = build_full_py_cams_from_file(cam_pose_txt_path, 1.0f);

    py::object bounding_py = decide_main_bounding(
        py::arg("bound_mode")      = "pcd",   // or "default" / "camera_max" …
        py::arg("pcd_density_rate")= 0.1,
        py::arg("bound_scale")     = 1.0,
        py::arg("pcd")             = pcd_obj  // just needs .points attribute-like
    );
    // py::object bounding_py = decide_main_bounding(
    //     py::arg("bound_mode")      = "default",   // or "default" / "camera_max" …
    //     py::arg("tr_cams")             = all_tr_cams  // just needs .points attribute-like
    // );
    // 1) cast to py::array_t<float> so we can access the buffer
    auto arr = bounding_py.cast<py::array_t<float>>();
    py::buffer_info info = arr.request();
    // 2) build a CPU tensor that wraps that data
    //    shape should be [2,3] here
    std::vector<int64_t> shape{ (int64_t)info.shape[0],
                                (int64_t)info.shape[1] };
    auto options = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(torch::kCPU);
    torch::Tensor bounding_cpu = torch::from_blob(
        info.ptr, shape, options);
    // 3) clone + move to your GPU (or whatever device_type_ is)
    torch::Tensor bounding = bounding_cpu.clone()
                                .to(device_type_);

    torch::Tensor scene_min = bounding_cpu[0].to(device_type_);
    torch::Tensor scene_max = bounding_cpu[1].to(device_type_);
    this->scene_center_ = 0.5f * (scene_min + scene_max);
    float extent_side  = (scene_max - scene_min).max().item<float>();
    // extent_side *= 3.0f;
    this->scene_extent_       = torch::tensor({extent_side},
                                 torch::TensorOptions().dtype(torch::kFloat32)
                                                        .device(device_type_));
    // std::cout << "[Debug] scene_extent = " << extent_side << std::endl;
    float float_level = -std::log2(0.05f / extent_side);
    int level = std::round(float_level);
    // std::cout << "level = " << level
    //           << " (float_level = " << float_level << ")" << std::endl;

    // const int init_level = 6;
    // // option A: using std::pow
    // float vox_size_pow   = extent_side * std::pow(2.0f, -init_level);
    // // option B: using std::ldexp (more efficient for integer exponents)
    // float vox_size_ldexp = std::ldexp(extent_side, -init_level);
    // std::cout << "[Debug] scene_extent = " << extent_side
    //         << ", init_n_level = " << init_level
    //         << "\n         voxel_size_pow   = " << vox_size_pow
    //         << "\n         voxel_size_ldexp = " << vox_size_ldexp
    //         << std::endl;

    // ─── 3) Call into Python’s SparseVoxelModel.model_init() ────────────
    // py::gil_scoped_acquire gil;
    static py::module svm_mod = []{
        py::module sys = py::module::import("sys");
        sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
        return py::module::import("src.sparse_voxel_model");
    }();
    py::object SVM     = svm_mod.attr("SparseVoxelModel");
    py_->svm = SVM(
        py::arg("sh_degree")        = max_sh_degree_,
        py::arg("black_background") = true
    );
    py_->svm.attr("model_init")(
        py::arg("bounding")        = bounding,
        py::arg("outside_level")   = 0,
        // py::arg("init_n_level")    = level,
        py::arg("init_n_level")    = 6,
        py::arg("init_out_ratio")  = 2,
        py::arg("sh_degree_init")  = 3,
        py::arg("geo_init")        = -10.0f,
        py::arg("sh0_init")        = 0.5f,
        py::arg("shs_init")        = 0.0f
        // py::arg("cameras")  = all_py_cams
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
    this->_geo_grid_pts_ = fetch("_geo_grid_pts").requires_grad_(true);
    this->sh0_ = fetch("_sh0").requires_grad_(true);
    this->shs_ = fetch("_shs").requires_grad_(true);
    this->subdiv_p_ = fetch("_subdiv_p").requires_grad_(true);
    // subdiv_p_   .retain_grad();
    // ─── 6) Stats buffers (exactly as before) ───────────────────────────
    this->max_w_ = torch::zeros({center_.size(0), 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

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

    // ─── 7) Register with the optimizer, etc. ──────────────────────────
    VOXEL_MODEL_TENSORS_TO_VEC
}

void VoxelModel::increasePcd(std::vector<float> pcd_full,
                             std::vector<float> colors,
                             const int /*iteration*/)
{
    // '''
    // dense grid
    // '''
    namespace py = pybind11;

    const int N = static_cast<int>(pcd_full.size() / 3);
    torch::Tensor xyz  = torch::from_blob(pcd_full.data(), {N,3}, torch::kFloat32).clone().to(device_type_);
    torch::Tensor cols = torch::from_blob(colors.data(),   {N,3}, torch::kFloat32).clone().to(device_type_);

    // ─── 2) Ask Python to compute the bounding box via decide_main_bounding ──
    py::gil_scoped_acquire gil;
    static py::module bu_mod = []{
        py::module sys = py::module::import("sys");
        sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
        return py::module::import("src.utils.bounding_utils");
    }();
    py::object decide_main_bounding = bu_mod.attr("decide_main_bounding");

    // Pass the point cloud to Python as a (N,3) float32 tensor on CPU
    torch::Tensor xyz_cpu = xyz.cpu().contiguous();
    // 1) Wrap the CPU tensor as a Python torch.Tensor
    py::object torch_tensor_py = py::cast(xyz_cpu);
    // 2) Call its .numpy() in Python to get a NumPy array
    py::object np_array = torch_tensor_py.attr("numpy")();
    // 3) Build a SimpleNamespace(points=ndarray) so .points works in Python
    py::object SimpleNS = py::module_::import("types").attr("SimpleNamespace");
    py::object pcd_obj  = SimpleNS(py::arg("points") = np_array);

    py::object bounding_py = decide_main_bounding(
        py::arg("bound_mode")      = "pcd",   // or "default" / "camera_max" …
        py::arg("pcd_density_rate")= 0.1,
        py::arg("bound_scale")     = 1.0,
        py::arg("pcd")             = pcd_obj  // just needs .points attribute-like
    );
    // 1) cast to py::array_t<float> so we can access the buffer
    auto arr = bounding_py.cast<py::array_t<float>>();
    py::buffer_info info = arr.request();
    // 2) build a CPU tensor that wraps that data
    //    shape should be [2,3] here
    std::vector<int64_t> shape{ (int64_t)info.shape[0],
                                (int64_t)info.shape[1] };
    auto options = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(torch::kCPU);
    torch::Tensor bounding_cpu = torch::from_blob(
        info.ptr, shape, options);
    // 3) clone + move to your GPU (or whatever device_type_ is)
    torch::Tensor bounding = bounding_cpu.clone()
                                .to(device_type_);

    torch::Tensor scene_min = bounding_cpu[0].to(device_type_);
    torch::Tensor scene_max = bounding_cpu[1].to(device_type_);
    this->scene_center_ = 0.5f * (scene_min + scene_max);
    float extent_side  = (scene_max - scene_min).max().item<float>();
    this->scene_extent_       = torch::tensor({extent_side},
                                 torch::TensorOptions().dtype(torch::kFloat32)
                                                        .device(device_type_));
    // std::cout << "[Debug] scene_extent = " << extent_side << std::endl;
    // float float_level = -std::log2(0.05f / extent_side);
    // int level = std::round(float_level);
    // std::cout << "level = " << level
    //           << " (float_level = " << float_level << ")" << std::endl;

    // ─── 3) Call into Python’s SparseVoxelModel.model_init() ────────────
    // py::gil_scoped_acquire gil;
    static py::module svm_mod = []{
        py::module sys = py::module::import("sys");
        sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
        return py::module::import("src.sparse_voxel_model");
    }();
    py::object SVM     = svm_mod.attr("SparseVoxelModel");
    py_->svm = SVM(
        py::arg("sh_degree")        = max_sh_degree_,
        py::arg("black_background") = true
    );
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
        return py_->svm.attr(name).cast<torch::Tensor>();
        // .contiguous();
    };
    this->oct_path_      = fetch("octpath");
    this->oct_level_     = fetch("octlevel");
    this->center_        = fetch("vox_center");
    this->size_          = fetch("vox_size").squeeze(1);
    this->vox_size_inv_  = 1.0f / size_;
    this->grid_pts_key_  = fetch("grid_pts_key");
    this->vox_key_       = fetch("vox_key");      
    // ─── 5) Copy over the learnable fields ─────────────────────────────
    this->_geo_grid_pts_ = fetch("_geo_grid_pts").requires_grad_(true);
    this->sh0_ = fetch("_sh0").requires_grad_(true);
    this->shs_ = fetch("_shs").requires_grad_(true);
    this->subdiv_p_ = fetch("_subdiv_p").requires_grad_(true);
    // subdiv_p_   .retain_grad();
    // ─── 6) Stats buffers (exactly as before) ───────────────────────────
    this->max_w_ = torch::zeros({center_.size(0), 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

    // // ─── 7) Register with the optimizer, etc. ──────────────────────────
    // VOXEL_MODEL_TENSORS_TO_VEC
    syncFromPython_();
}

// void VoxelModel::createFromPcd(
//     const std::map<point3D_id_t, Point3D>& pcd)
// {
//     // '''
//     // offline approach with dense grid
//     // '''
//     std::cout << "VoxelModel::createPcd() called with "
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
//     // ─── 2) Use KNOWN inflated bounding directly (no file, no Python utils) ──
//     auto opts_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
//     torch::Tensor bounding_cpu = torch::empty({2,3}, opts_cpu);
//     // Row 0 = min, Row 1 = max
//     bounding_cpu[0][0] = -10.748001f;  bounding_cpu[0][1] = -25.842281f;  bounding_cpu[0][2] = -0.403899f;
//     bounding_cpu[1][0] =  56.780504f;  bounding_cpu[1][1] =  15.760179f;  bounding_cpu[1][2] = 60.683914f;

//     // Move to your device for model_init:
//     torch::Tensor bounding = bounding_cpu.to(device_type_);
//     torch::Tensor scene_min = bounding_cpu[0].to(device_type_);
//     torch::Tensor scene_max = bounding_cpu[1].to(device_type_);
//     this->scene_center_ = 0.5f * (scene_min + scene_max);
//     float extent_side  = (scene_max - scene_min).max().item<float>();
//     this->scene_extent_       = torch::tensor({extent_side},
//                                  torch::TensorOptions().dtype(torch::kFloat32)
//                                                         .device(device_type_));
//     float float_level = -std::log2(0.05f / extent_side);
//     int level = std::round(float_level);
//     std::cout << "level = " << level
//               << " (float_level = " << float_level << ")" << std::endl;

//     // ─── 3) Call into Python’s SparseVoxelModel.model_init() ───────────────
//     py::gil_scoped_acquire gil;
//     static py::module svm_mod = []{
//         py::module sys = py::module::import("sys");
//         sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
//         return py::module::import("src.sparse_voxel_model");
//     }();
//     py::object SVM    = svm_mod.attr("SparseVoxelModel");
//     py_->svm = SVM(py::arg("sh_degree")        = max_sh_degree_);
//     py_->svm.attr("model_init")(
//         py::arg("bounding")        = bounding,
//         py::arg("outside_level")   = 0,
//         py::arg("init_n_level")    = 6,
//         py::arg("init_out_ratio")  = 2,
//         py::arg("sh_degree_init")  = 3,
//         py::arg("geo_init")        = -10.0f,
//         py::arg("sh0_init")        = 0.5f,
//         py::arg("shs_init")        = 0.0f
//     );

//     // {
//     //     py::gil_scoped_acquire gil;
//     //     auto sh0 = py_->svm.attr("_sh0");
//     //     auto shs = py_->svm.attr("_shs");
//     //     auto geo = py_->svm.attr("_geo_grid_pts");
//     //     sh0.attr("requires_grad_")(true);
//     //     shs.attr("requires_grad_")(true);
//     //     geo.attr("requires_grad_")(true);
//     // }
//     // ─── 4) Pull back all core tensors from the Python object ───────────
//     auto fetch = [&](const char* name){
//         return py_->svm.attr(name).cast<torch::Tensor>();
//         // .contiguous();
//     };
//     this->oct_path_      = fetch("octpath");
//     this->oct_level_     = fetch("octlevel");
//     this->center_        = fetch("vox_center");
//     this->size_          = fetch("vox_size").squeeze(1);
//     this->vox_size_inv_  = 1.0f / this->size_;
//     this->grid_pts_key_  = fetch("grid_pts_key");
//     this->vox_key_       = fetch("vox_key");      
//     // ─── 5) Copy over the learnable fields ─────────────────────────────
//     this->_geo_grid_pts_ = fetch("_geo_grid_pts").requires_grad_(true);
//     // this->sh0_ = fetch("_sh0").view({-1,1,3}).requires_grad_(true);
//     this->sh0_ = fetch("_sh0").requires_grad_(true);
//     // this->sh0_ = this->sh0_.view({-1,1,3});
//     this->shs_ = fetch("_shs").requires_grad_(true);
//     this->subdiv_p_ = fetch("_subdiv_p").requires_grad_(true);

//     // this->_geo_grid_pts_ = fetch("_geo_grid_pts");
//     // this->sh0_ = fetch("_sh0").view({-1,1,3});
//     // this->shs_ = fetch("_shs");
//     // this->subdiv_p_ = fetch("_subdiv_p");
//     // ─── 6) Stats buffers (exactly as before) ───────────────────────────
//     this->max_w_ = torch::zeros({center_.size(0), 1},
//         torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

//     // std::cout << " center_ " << center_ << std::endl;
//     // After you fetch center_ (M×3) and size_ (M,)
//     auto c_cpu = center_.cpu();
//     auto s_cpu = size_.cpu();
//     torch::Tensor bb_min_eff = (c_cpu - 0.5 * s_cpu.unsqueeze(1)).amin(/*dim=*/0);
//     torch::Tensor bb_max_eff = (c_cpu + 0.5 * s_cpu.unsqueeze(1)).amax(/*dim=*/0);
//     this->bb_min_eff_ = bb_min_eff.to(device_type_);
//     this->bb_max_eff_ = bb_max_eff.to(device_type_);
//     std::cout << std::fixed << std::setprecision(6)
//             << "[VOXEL GRID AABB] min:[" << bb_min_eff[0].item<float>() << ","
//                                         << bb_min_eff[1].item<float>() << ","
//                                         << bb_min_eff[2].item<float>() << "]  "
//             << "max:[" << bb_max_eff[0].item<float>() << ","
//                         << bb_max_eff[1].item<float>() << ","
//                         << bb_max_eff[2].item<float>() << "]\n";

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

//     VOXEL_MODEL_TENSORS_TO_VEC
// }

// void VoxelModel::createFromPcd(
//     const std::map<point3D_id_t, Point3D>& pcd)
// {
//     // '''
//     // offline approach points_init
//     // '''
//     std::cout << "VoxelModel::createPcd() called with "
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

//     auto mn = std::get<0>(pts.min(0));
//     auto mx = std::get<0>(pts.max(0));
//     float raw_extent     = (mx - mn).max().item<float>();
//     float padded_extent = raw_extent * 1.01f;   // ← 1% padding
//     this->scene_center_ = (mn + mx) * 0.5f;
//     this->scene_extent_ = torch::tensor({padded_extent},
//                                  torch::TensorOptions().dtype(torch::kFloat32)
//                                                         .device(device_type_));

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
//     // this->scene_extent_       = torch::tensor({extent_side},
//     //                              torch::TensorOptions().dtype(torch::kFloat32)
//     //                                                     .device(device_type_));

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
//         return py_->svm.attr(name).cast<torch::Tensor>();
//         // .contiguous();
//     };
//     this->oct_path_      = fetch("octpath");
//     this->oct_level_     = fetch("octlevel");
//     this->center_        = fetch("vox_center");
//     this->size_          = fetch("vox_size").squeeze(1);
//     this->vox_size_inv_  = 1.0f / size_;
//     this->grid_pts_key_  = fetch("grid_pts_key");
//     this->vox_key_       = fetch("vox_key");      
//     // ─── 5) Copy over the learnable fields ─────────────────────────────
    // this->_geo_grid_pts_ = fetch("_geo_grid_pts").requires_grad_(true);
    // this->sh0_ = fetch("_sh0").requires_grad_(true);
    // this->shs_ = fetch("_shs").requires_grad_(true);
    // this->subdiv_p_ = fetch("_subdiv_p").requires_grad_(true);
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

// // --- helpers for hashing keys we can match across rebuilds ------------
// namespace {
// struct PairHash {
//     template<class T1,class T2>
//     std::size_t operator()(const std::pair<T1,T2>& p) const noexcept {
//         auto h1 = std::hash<T1>{}(p.first);
//         auto h2 = std::hash<T2>{}(p.second);
//         return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1<<6) + (h1>>2));
//     }
// };
// struct Tuple3Hash {
//     std::size_t operator()(const std::tuple<int64_t,int64_t,int64_t>& t) const noexcept {
//         auto h = std::hash<int64_t>{}(std::get<0>(t));
//         auto h2= std::hash<int64_t>{}(std::get<1>(t));
//         auto h3= std::hash<int64_t>{}(std::get<2>(t));
//         h ^= h2 + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
//         h ^= h3 + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
//         return h;
//     }
// };
// } // namespace

// // ---------- debug helpers ----------
// static std::string shp(const torch::Tensor& t) {
//     if (!t.defined()) return "<undef>";
//     std::ostringstream oss; oss << '[';
//     for (int i=0;i<t.dim();++i){ if(i) oss<<','; oss<<t.size(i); }
//     oss << ']'; return oss.str();
// }
// static std::string dev(const torch::Tensor& t) {
//     return t.defined() ? t.device().str() : "<undef>";
// }
// static void dump_param(const char* name, const torch::Tensor& p) {
//     std::cout << "    " << name
//               << " def=" << (p.defined()?1:0)
//               << " req=" << (p.defined()?p.requires_grad():false)
//               << " shape=" << shp(p)
//               << " dtype=" << (p.defined()?c10::toString(p.scalar_type()):"<undef>")
//               << " dev=" << dev(p);
//     torch::Tensor g;
//     try { if (p.defined()) g = p.grad(); } catch(...) {}
//     std::cout << "  | grad def=" << (g.defined()?1:0)
//               << " shape=" << shp(g) << "\n";
// }

// void VoxelModel::increasePcd(std::vector<float> pcd_full,
//                              std::vector<float> colors,
//                              const int /*iteration*/)
// {
//     namespace py = pybind11;
//     if (pcd_full.empty()) return;
//     if (pcd_full.size() != colors.size()) { std::cerr << "[increasepcd] points/colors size mismatch\n"; return; }
//     if ((pcd_full.size() % 3) != 0)        { std::cerr << "[increasepcd] points.size() not divisible by 3\n"; return; }

//     const int Nnew = (int)pcd_full.size()/3;
//     torch::Tensor new_pts  = torch::from_blob(pcd_full.data(), {Nnew,3}, torch::kFloat32).clone().to(device_type_);
//     torch::Tensor new_cols = torch::from_blob(colors.data(),   {Nnew,3}, torch::kFloat32).clone().to(device_type_);

//     // ---- bounds: union with current scene bbox ----
//     auto mn = std::get<0>(new_pts.min(0)).detach().to(torch::kCPU);
//     auto mx = std::get<0>(new_pts.max(0)).detach().to(torch::kCPU);
//     auto pcd_min = mn, pcd_max = mx;

//     torch::Tensor old_center_cpu = this->scene_center_.defined()
//         ? this->scene_center_.detach().to(torch::kCPU)
//         : (pcd_min + pcd_max) * 0.5f;

//     torch::Tensor old_extent_cpu = this->scene_extent_.defined()
//         ? this->scene_extent_.detach().to(torch::kCPU)
//         : torch::tensor({(pcd_max - pcd_min).max().item<float>()}, torch::kFloat32);

//     auto old_half = 0.5f * old_extent_cpu;
//     auto old_min  = old_center_cpu - old_half;
//     auto old_max  = old_center_cpu + old_half;
//     auto uni_min    = torch::min(old_min, pcd_min);
//     auto uni_max    = torch::max(old_max, pcd_max);
//     auto uni_center = (uni_min + uni_max) * 0.5f;
//     float uni_extent = (uni_max - uni_min).max().item<float>() * 1.01f;
//     if (uni_extent < 1e-3f) uni_extent = 1e-3f;

//     this->scene_center_ = uni_center.to(device_type_);
//     this->scene_extent_ = torch::tensor({uni_extent},
//                        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

//     // ---- snapshot "old" state (for transfer) ----
//     torch::Tensor old_op, old_lv, old_sh0, old_shs, old_subdiv, old_gkey, old_geogrid, old_center, old_cols_like;
//     if (this->oct_path_.defined())       old_op      = this->oct_path_.detach().cpu().contiguous();
//     if (this->oct_level_.defined())      old_lv      = this->oct_level_.detach().cpu().contiguous();
//     if (this->grid_pts_key_.defined())   old_gkey    = this->grid_pts_key_.detach().cpu().contiguous();
//     if (this->_geo_grid_pts_.defined())  old_geogrid = this->_geo_grid_pts_.detach().cpu().contiguous();
//     if (this->center_.defined())         old_center  = this->center_.detach().clone();      // [Nold,3] (device)
//     if (this->sh0_.defined())            old_sh0     = this->sh0_.detach().clone();         // [Nold,3] (device)
//     if (this->shs_.defined())            old_shs     = this->shs_.detach().cpu().contiguous();
//     if (this->subdiv_p_.defined())       old_subdiv  = this->subdiv_p_.detach().cpu().contiguous();

//     // old colors for seeding (clamp to [0,1] just in case)
//     if (old_sh0.defined()) {
//         torch::Tensor c = old_sh0;
//         if (c.dim()==3 && c.size(1)==1 && c.size(2)==3) c = c.view({c.size(0),3});
//         old_cols_like = c.clamp(0.f, 1.f);
//     }

//     // ---- build seed cloud = [ new_pts ; old voxel centers ] ----
//     torch::Tensor seed_pts  = new_pts;
//     torch::Tensor seed_cols = new_cols;
//     if (old_center.defined()) {
//         seed_pts  = torch::cat({seed_pts,  old_center.to(device_type_)}, 0);
//         if (old_cols_like.defined())
//             seed_cols = torch::cat({seed_cols, old_cols_like.to(device_type_)}, 0);
//         else
//             seed_cols = torch::cat({seed_cols, torch::zeros_like(old_center)}, 0);
//     }

//     std::cout << "[increasepcd:APPEND] seed: new=" << Nnew
//               << "  oldCenters=" << (old_center.defined()? old_center.size(0):0)
//               << "  total=" << seed_pts.size(0)
//               << "  extent≈" << uni_extent << "\n";

//     // ---- rebuild SVM with seed cloud ----
//     {
//         py::gil_scoped_acquire gil;
//         static py::module svm_mod = []{
//             py::module sys = py::module::import("sys");
//             sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
//             return py::module::import("src.sparse_voxel_model");
//         }();
//         py::object SVM = svm_mod.attr("SparseVoxelModel");
//         py_->svm = SVM(py::arg("sh_degree")        = max_sh_degree_,
//                        py::arg("black_background") = true);

//         py_->svm.attr("points_init")(
//             py::arg("scene_center")      = py::cast(scene_center_),
//             py::arg("scene_extent")      = py::cast(scene_extent_),
//             py::arg("xyz")               = py::cast(seed_pts),
//             py::arg("expected_vox_size") = 0.05f,
//             py::arg("density")           = -10.0f,
//             py::arg("rgb")               = py::cast(seed_cols),
//             py::arg("shs")               = 0.0f
//         );
//     }

//     // ---- fetch new tensors ----
//     auto fetch = [&](const char* name){ return py_->svm.attr(name).cast<torch::Tensor>(); };
//     torch::Tensor new_op       = fetch("octpath");
//     torch::Tensor new_lv       = fetch("octlevel");
//     torch::Tensor new_gkey     = fetch("grid_pts_key");
//     torch::Tensor new_geogrid  = fetch("_geo_grid_pts");
//     torch::Tensor new_sh0      = fetch("_sh0");
//     torch::Tensor new_shs      = fetch("_shs");
//     torch::Tensor new_subdiv   = fetch("_subdiv_p");

//     // ---- robust transfer (same as your guarded version) ----
//     auto sh0_to_flat = [](torch::Tensor t)->torch::Tensor {
//         if (t.defined() && t.dim()==3 && t.size(1)==1 && t.size(2)==3) return t.view({t.size(0),3});
//         return t;
//     };
//     auto shs_to_flat = [](torch::Tensor t)->torch::Tensor {
//         if (t.defined() && t.dim()==3) return t.view({t.size(0), t.size(1)*t.size(2)});
//         return t;
//     };
//     auto transfer_rows = [](torch::Tensor& dst, const torch::Tensor& src,
//                             const std::vector<std::pair<int64_t,int64_t>>& pairs)->int64_t {
//         if (!dst.defined() || !src.defined()) return 0;
//         auto dst_cpu = dst.detach().cpu().contiguous();
//         auto src_cpu = src.detach().cpu().contiguous();
//         if (dst_cpu.dim()==1) dst_cpu = dst_cpu.view({dst_cpu.size(0),1});
//         if (src_cpu.dim()==1) src_cpu = src_cpu.view({src_cpu.size(0),1});
//         const int64_t Nd = dst_cpu.size(0), Ns = src_cpu.size(0);
//         if (Nd==0 || Ns==0) return 0;
//         const int64_t de = dst_cpu.numel()/std::max<int64_t>(Nd,1);
//         const int64_t se = src_cpu.numel()/std::max<int64_t>(Ns,1);
//         if (de != se) return 0;
//         auto d2 = dst_cpu.view({Nd,de});
//         auto s2 = src_cpu.view({Ns,se});
//         int64_t hit=0;
//         for (auto [i_new, j_old] : pairs) if (0<=i_new && i_new<Nd && 0<=j_old && j_old<Ns) { d2[i_new]=s2[j_old]; ++hit; }
//         dst = d2.view(dst.sizes()).to(dst.device());
//         return hit;
//     };

//     int64_t vox_hits=0, grid_hits=0;
//     if (old_op.defined() && old_lv.defined() && new_op.defined() && new_lv.defined()) {
//         auto o_op = old_op.to(torch::kInt64).view({-1});
//         auto o_lv = old_lv.to(torch::kInt64).view({-1});
//         std::unordered_map<std::pair<int64_t,int64_t>, int64_t, PairHash> old_map;
//         old_map.reserve(o_op.size(0));
//         for (int64_t i=0;i<o_op.size(0);++i)
//             old_map.emplace(std::make_pair(o_op[i].item<int64_t>(), o_lv[i].item<int64_t>()), i);

//         auto n_op = new_op.detach().cpu().to(torch::kInt64).view({-1});
//         auto n_lv = new_lv.detach().cpu().to(torch::kInt64).view({-1});
//         std::vector<std::pair<int64_t,int64_t>> pairs; pairs.reserve(n_op.size(0));
//         for (int64_t i=0;i<n_op.size(0);++i) {
//             auto it = old_map.find(std::make_pair(n_op[i].item<int64_t>(), n_lv[i].item<int64_t>()));
//             if (it != old_map.end()) pairs.emplace_back(i, it->second);
//         }

//         auto new_sh0_f = sh0_to_flat(new_sh0);
//         auto old_sh0_f = sh0_to_flat(old_sh0);
//         vox_hits += transfer_rows(new_sh0_f, old_sh0_f, pairs);
//         new_sh0 = new_sh0_f;

//         int64_t R=0;
//         if (new_shs.defined()) R = (new_shs.dim()==3)? new_shs.size(1) : (new_shs.dim()==2? new_shs.size(1)/3 : 0);
//         auto new_shs_f = shs_to_flat(new_shs);
//         auto old_shs_f = shs_to_flat(old_shs);
//         vox_hits += transfer_rows(new_shs_f, old_shs_f, pairs);
//         if (new_shs_f.defined() && R>0) new_shs = new_shs_f.view({new_shs_f.size(0), R, 3});

//         if (new_subdiv.defined() && new_subdiv.dim()==2 && new_subdiv.size(1)==1) new_subdiv = new_subdiv.squeeze(1);
//         if (old_subdiv.defined() && old_subdiv.dim()==2 && old_subdiv.size(1)==1) old_subdiv = old_subdiv.squeeze(1);
//         vox_hits += transfer_rows(new_subdiv, old_subdiv, pairs);
//         if (vox_hits) std::cout << "[increasepcd:APPEND] transferred vox=" << vox_hits << "\n";
//     }

//     if (old_gkey.defined() && old_geogrid.defined() && new_gkey.defined() && new_geogrid.defined()) {
//         auto o_key = old_gkey.to(torch::kInt64).view({-1,3});
//         std::unordered_map<std::tuple<int64_t,int64_t,int64_t>, int64_t, Tuple3Hash> gmap;
//         gmap.reserve(o_key.size(0));
//         for (int64_t i=0;i<o_key.size(0);++i)
//             gmap.emplace(std::make_tuple(o_key[i][0].item<int64_t>(),
//                                          o_key[i][1].item<int64_t>(),
//                                          o_key[i][2].item<int64_t>()), i);

//         auto n_key     = new_gkey.detach().cpu().to(torch::kInt64).view({-1,3});
//         auto n_geo_cpu = new_geogrid.detach().cpu().contiguous();
//         for (int64_t i=0;i<n_key.size(0) && i<n_geo_cpu.size(0); ++i) {
//             auto it = gmap.find(std::make_tuple(n_key[i][0].item<int64_t>(),
//                                                 n_key[i][1].item<int64_t>(),
//                                                 n_key[i][2].item<int64_t>()));
//             if (it != gmap.end()) { n_geo_cpu[i] = old_geogrid[it->second]; ++grid_hits; }
//         }
//         if (grid_hits) std::cout << "[increasepcd:APPEND] transferred grid=" << grid_hits << "\n";
//         new_geogrid = n_geo_cpu.to(device_type_);
//     }

//     // ---- push transferred fields back to Python (canonical shapes) ----
//     {
//         py::gil_scoped_acquire gil;
//         if (new_sh0.defined() && new_sh0.dim()==3 && new_sh0.size(1)==1 && new_sh0.size(2)==3)
//             new_sh0 = new_sh0.view({new_sh0.size(0),3});
//         if (new_shs.defined() && new_shs.dim()==2 && (new_shs.size(1)%3)==0)
//             new_shs = new_shs.view({new_shs.size(0), new_shs.size(1)/3, 3});
//         if (new_subdiv.defined() && new_subdiv.dim()==2 && new_subdiv.size(1)==1)
//             new_subdiv = new_subdiv.squeeze(1);

//         py_->svm.attr("_sh0")          = new_sh0;
//         py_->svm.attr("_shs")          = new_shs;
//         py_->svm.attr("_subdiv_p")     = new_subdiv;
//         py_->svm.attr("_geo_grid_pts") = new_geogrid;
//     }

//     // ---- sync back to C++ ----
//     syncFromPython_();
// }

// void VoxelModel::increasePcd(std::vector<float> pcd_full,
//                              std::vector<float> colors,
//                              const int /*iteration*/)
// {
//     namespace py = pybind11;
//     if (pcd_full.empty()) return;
//     if (pcd_full.size() != colors.size()) { std::cerr << "[increasepcd] points/colors size mismatch\n"; return; }
//     if ((pcd_full.size() % 3) != 0)        { std::cerr << "[increasepcd] points.size() not divisible by 3\n"; return; }

//     const int N = static_cast<int>(pcd_full.size() / 3);
//     torch::Tensor pts  = torch::from_blob(pcd_full.data(), {N,3}, torch::kFloat32).clone().to(device_type_);
//     torch::Tensor cols = torch::from_blob(colors.data(),   {N,3}, torch::kFloat32).clone().to(device_type_);

//     // --- 1) expand scene bounds with the new points (small pad) ---
//     auto mn = std::get<0>(pts.min(0)).detach().to(torch::kCPU);
//     auto mx = std::get<0>(pts.max(0)).detach().to(torch::kCPU);

//     torch::Tensor old_center_cpu = scene_center_.defined()
//         ? scene_center_.detach().to(torch::kCPU)
//         : (mn + mx) * 0.5f;

//     torch::Tensor old_extent_cpu = scene_extent_.defined()
//         ? scene_extent_.detach().to(torch::kCPU)
//         : torch::tensor({(mx - mn).max().item<float>()}, torch::kFloat32);

//     auto half     = 0.5f * old_extent_cpu;
//     auto old_min  = old_center_cpu - half;
//     auto old_max  = old_center_cpu + half;
//     auto uni_min  = torch::min(old_min, mn);
//     auto uni_max  = torch::max(old_max, mx);
//     auto uni_ctr  = (uni_min + uni_max) * 0.5f;
//     float uni_ext = (uni_max - uni_min).max().item<float>() * 1.01f;
//     if (uni_ext < 1e-3f) uni_ext = 1e-3f;

//     scene_center_ = uni_ctr.to(device_type_);
//     scene_extent_ = torch::tensor({uni_ext},
//                       torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

//     // --- 2) snapshot OLD learnables/keys (for transfer) ---
//     torch::Tensor old_op, old_lv, old_sh0, old_shs, old_subdiv, old_gkey, old_geogrid;
//     if (oct_path_.defined())       old_op      = oct_path_.detach().cpu().contiguous();
//     if (oct_level_.defined())      old_lv      = oct_level_.detach().cpu().contiguous();
//     if (sh0_.defined())            old_sh0     = sh0_.detach().cpu().contiguous();
//     if (shs_.defined())            old_shs     = shs_.detach().cpu().contiguous();
//     if (subdiv_p_.defined())       old_subdiv  = subdiv_p_.detach().cpu().contiguous();
//     if (grid_pts_key_.defined())   old_gkey    = grid_pts_key_.detach().cpu().contiguous();
//     if (_geo_grid_pts_.defined())  old_geogrid = _geo_grid_pts_.detach().cpu().contiguous();

//     // --- 3) choose expected_vox_size to match current model ---
//     float expected_vox = 0.05f;
//     if (size_.defined() && size_.numel() > 0) {
//         // median() returns (values, indices) in some versions; handle both
//         torch::Tensor s = size_.detach().to(torch::kCPU).view({-1});
//         float med = 0.05f;
//         if (s.numel() == 1) {
//             med = s.item<float>();
//         } else {
//             auto sorted = std::get<0>(s.sort());
//             med = sorted[sorted.size(0)/2].item<float>();
//         }
//         // stay within sensible bounds relative to current scene extent
//         float max_allowed = std::max(1e-5f, scene_extent_.item<float>() / (1 << 12)); // conservative
//         float min_allowed = 1e-5f;
//         expected_vox = std::clamp(med, min_allowed, max_allowed);
//     }

//     std::cout << "[increasepcd:RESET] new-only init: N=" << N
//               << "  center=" << scene_center_
//               << "  extent≈" << expected_vox * (1 << 8)  // quick hint of level
//               << "\n"
//               << "    pts=" << shp(pts) << " dev=" << dev(pts)
//               << "  cols=" << shp(cols) << " dev=" << dev(cols) << "\n";

//     // --- 4) rebuild SVM from ONLY the new points (let Python compute geometry) ---
//     {
//         py::gil_scoped_acquire gil;
//         static py::module svm_mod = []{
//             py::module sys = py::module::import("sys");
//             sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
//             return py::module::import("src.sparse_voxel_model");
//         }();
//         py::object SVM = svm_mod.attr("SparseVoxelModel");
//         py_->svm = SVM(py::arg("sh_degree")        = max_sh_degree_,
//                        py::arg("black_background") = true);

//         // IMPORTANT: do not assign read-only attrs like vox_center/vox_size/etc.
//         py_->svm.attr("points_init")(
//             py::arg("scene_center")      = py::cast(scene_center_),
//             py::arg("scene_extent")      = py::cast(scene_extent_),
//             py::arg("xyz")               = py::cast(pts),
//             py::arg("expected_vox_size") = expected_vox,
//             py::arg("density")           = -10.0f,
//             py::arg("rgb")               = py::cast(cols),
//             py::arg("shs")               = 0.0f
//         );
//     }

//     // --- 5) fetch NEW tensors from SVM (for transfer) ---
//     auto fetch = [&](const char* name){ return py_->svm.attr(name).cast<torch::Tensor>(); };
//     torch::Tensor new_op       = fetch("octpath");
//     torch::Tensor new_lv       = fetch("octlevel");
//     torch::Tensor new_gkey     = fetch("grid_pts_key");
//     torch::Tensor new_geogrid  = fetch("_geo_grid_pts");
//     torch::Tensor new_sh0      = fetch("_sh0");
//     torch::Tensor new_shs      = fetch("_shs");
//     torch::Tensor new_subdiv   = fetch("_subdiv_p");

//     auto sh0_to_flat = [](torch::Tensor t)->torch::Tensor {
//         if (t.defined() && t.dim()==3 && t.size(1)==1 && t.size(2)==3) return t.view({t.size(0),3});
//         return t;
//     };
//     auto shs_to_flat = [](torch::Tensor t)->torch::Tensor {
//         if (t.defined() && t.dim()==3) return t.view({t.size(0), t.size(1)*t.size(2)});
//         return t;
//     };

//     auto transfer_rows = [](torch::Tensor& dst, const torch::Tensor& src,
//                             const std::vector<std::pair<int64_t,int64_t>>& pairs)->int64_t {
//         if (!dst.defined() || !src.defined()) return 0;
//         auto dst_cpu = dst.detach().cpu().contiguous();
//         auto src_cpu = src.detach().cpu().contiguous();
//         if (dst_cpu.dim()==1) dst_cpu = dst_cpu.view({dst_cpu.size(0),1});
//         if (src_cpu.dim()==1) src_cpu = src_cpu.view({src_cpu.size(0),1});
//         const int64_t Nd = dst_cpu.size(0), Ns = src_cpu.size(0);
//         if (Nd==0 || Ns==0) return 0;
//         const int64_t de = dst_cpu.numel()/std::max<int64_t>(Nd,1);
//         const int64_t se = src_cpu.numel()/std::max<int64_t>(Ns,1);
//         if (de != se) return 0;
//         auto d2 = dst_cpu.view({Nd,de});
//         auto s2 = src_cpu.view({Ns,se});
//         int64_t hit=0;
//         for (auto [i_new, j_old] : pairs) if (0<=i_new && i_new<Nd && 0<=j_old && j_old<Ns) { d2[i_new]=s2[j_old]; ++hit; }
//         dst = d2.view(dst.sizes()).to(dst.device());
//         return hit;
//     };

//     // --- 6) transfer by exact (octpath, octlevel) matches ---
//     int64_t vox_hits = 0, grid_hits = 0;
//     if (old_op.defined() && old_lv.defined() && new_op.defined() && new_lv.defined()) {
//         auto o_op = old_op.to(torch::kInt64).view({-1});
//         auto o_lv = old_lv.to(torch::kInt64).view({-1});
//         std::unordered_map<std::pair<int64_t,int64_t>, int64_t, PairHash> old_map;
//         old_map.reserve(o_op.size(0));
//         for (int64_t i=0;i<o_op.size(0);++i)
//             old_map.emplace(std::make_pair(o_op[i].item<int64_t>(), o_lv[i].item<int64_t>()), i);

//         auto n_op = new_op.detach().cpu().to(torch::kInt64).view({-1});
//         auto n_lv = new_lv.detach().cpu().to(torch::kInt64).view({-1});
//         std::vector<std::pair<int64_t,int64_t>> pairs; pairs.reserve(n_op.size(0));
//         for (int64_t i=0;i<n_op.size(0);++i) {
//             auto it = old_map.find(std::make_pair(n_op[i].item<int64_t>(), n_lv[i].item<int64_t>()));
//             if (it != old_map.end()) pairs.emplace_back(i, it->second);
//         }

//         // SH0
//         auto new_sh0_f = sh0_to_flat(new_sh0);
//         auto old_sh0_f = sh0_to_flat(old_sh0);
//         vox_hits += transfer_rows(new_sh0_f, old_sh0_f, pairs);
//         new_sh0 = new_sh0_f;

//         // SHs
//         int64_t R=0;
//         if (new_shs.defined()) R = (new_shs.dim()==3)? new_shs.size(1) : (new_shs.dim()==2? new_shs.size(1)/3 : 0);
//         auto new_shs_f = shs_to_flat(new_shs);
//         auto old_shs_f = shs_to_flat(old_shs);
//         vox_hits += transfer_rows(new_shs_f, old_shs_f, pairs);
//         if (new_shs_f.defined() && R>0) new_shs = new_shs_f.view({new_shs_f.size(0), R, 3});

//         // subdiv priority
//         if (new_subdiv.defined() && new_subdiv.dim()==2 && new_subdiv.size(1)==1) new_subdiv = new_subdiv.squeeze(1);
//         if (old_subdiv.defined() && old_subdiv.dim()==2 && old_subdiv.size(1)==1) old_subdiv = old_subdiv.squeeze(1);
//         vox_hits += transfer_rows(new_subdiv, old_subdiv, pairs);
//     }

//     // --- 7) transfer geometry grid by key triplets (int64[*,3]) ---
//     if (old_gkey.defined() && old_geogrid.defined() && new_gkey.defined() && new_geogrid.defined()) {
//         auto o_key = old_gkey.to(torch::kInt64).view({-1,3});
//         std::unordered_map<std::tuple<int64_t,int64_t,int64_t>, int64_t, Tuple3Hash> gmap;
//         gmap.reserve(o_key.size(0));
//         for (int64_t i=0;i<o_key.size(0);++i)
//             gmap.emplace(std::make_tuple(o_key[i][0].item<int64_t>(),
//                                          o_key[i][1].item<int64_t>(),
//                                          o_key[i][2].item<int64_t>()), i);

//         auto n_key     = new_gkey.detach().cpu().to(torch::kInt64).view({-1,3});
//         auto n_geo_cpu = new_geogrid.detach().cpu().contiguous();
//         for (int64_t i=0;i<n_key.size(0) && i<n_geo_cpu.size(0); ++i) {
//             auto it = gmap.find(std::make_tuple(n_key[i][0].item<int64_t>(),
//                                                 n_key[i][1].item<int64_t>(),
//                                                 n_key[i][2].item<int64_t>()));
//             if (it != gmap.end()) { n_geo_cpu[i] = old_geogrid[it->second]; ++grid_hits; }
//         }
//         new_geogrid = n_geo_cpu.to(device_type_);
//     }

//     if (vox_hits || grid_hits)
//         std::cout << "[increasepcd:RESET] transferred vox=" << vox_hits
//                   << " grid=" << grid_hits << "\n";

//     // --- 8) write back ONLY writable learnables (canonical shapes) ---
//     {
//         py::gil_scoped_acquire gil;
//         if (new_sh0.defined() && new_sh0.dim()==3 && new_sh0.size(1)==1 && new_sh0.size(2)==3)
//             new_sh0 = new_sh0.view({new_sh0.size(0),3});
//         if (new_shs.defined() && new_shs.dim()==2 && (new_shs.size(1)%3)==0)
//             new_shs = new_shs.view({new_shs.size(0), new_shs.size(1)/3, 3});
//         if (new_subdiv.defined() && new_subdiv.dim()==2 && new_subdiv.size(1)==1)
//             new_subdiv = new_subdiv.squeeze(1);

//         py_->svm.attr("_sh0")          = new_sh0;      // [N,3]
//         py_->svm.attr("_shs")          = new_shs;      // [N,R,3]
//         py_->svm.attr("_subdiv_p")     = new_subdiv;   // [N]
//         py_->svm.attr("_geo_grid_pts") = new_geogrid;  // [M,1]
//     }

//     // --- 9) pull everything back to C++ ---
//     syncFromPython_();
// }

void VoxelModel::syncFromPython_() {
    py::gil_scoped_acquire gil;

    auto fetch = [&](const char* name){
        return py_->svm.attr(name).cast<torch::Tensor>();
        // .contiguous();
    };

    // Octree / indexing
    this->oct_path_   = fetch("octpath");
    this->oct_level_  = fetch("octlevel");
    this->vox_key_    = fetch("vox_key");               // [N,8] long
    this->center_     = fetch("vox_center");            // [N,3] float
    this->size_       = fetch("vox_size").squeeze(1);   // [N] or [N,1]→[N]
    this->vox_size_inv_  = 1.0f / this->size_;
    this->grid_pts_key_  = fetch("grid_pts_key");
    // Learnables
    this->_geo_grid_pts_ = fetch("_geo_grid_pts").requires_grad_(true);
    this->sh0_ = fetch("_sh0").requires_grad_(true);
    // this->sh0_ = this->sh0_.view({-1,1,3});
    this->shs_ = fetch("_shs").requires_grad_(true);
    this->subdiv_p_ = fetch("_subdiv_p").requires_grad_(true);
    // Resize any side buffers bound to voxel count
    this->max_w_ = torch::zeros({center_.size(0), 1},
              torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

    VOXEL_MODEL_TENSORS_TO_VEC
}

void VoxelModel::createTrainer(float geo_lr, float sh0_lr, float shs_lr,
                               float beta1, float beta2, float eps,
                               const std::vector<int>& milestones,
                               float gamma)
{
    py::gil_scoped_acquire gil;

    // --- Optimizer (SparseAdam) ---
    py::module sadam = py::module::import("svraster_cuda.sparse_adam");
    py::object SparseAdam = sadam.attr("SparseAdam");

    // Build param groups with per-group learning rates (SVRaster style)
    py::list groups;
    {
        py::dict g0;
        g0["params"] = py::make_tuple(this->_geo_grid_pts_);
        g0["lr"]     = geo_lr;
        groups.append(g0);
    }
    {
        py::dict g1;
        g1["params"] = py::make_tuple(this->sh0_);
        g1["lr"]     = sh0_lr;
        groups.append(g1);
    }
    {
        py::dict g2;
        g2["params"] = py::make_tuple(this->shs_);
        g2["lr"]     = shs_lr;
        groups.append(g2);
    }

    // Same call style as Python (group-specific lrs; global lr unused)
    py_->optimizer_py = SparseAdam(groups,
                                   "betas"_a = py::make_tuple(beta1, beta2),
                                   "eps"_a   = eps);

    // --- Scheduler (MultiStepLR) ---
    py::module lr_sched = py::module::import("torch.optim.lr_scheduler");
    py_->scheduler_py = lr_sched.attr("MultiStepLR")(
        py_->optimizer_py,
        "milestones"_a = py::cast(milestones),
        "gamma"_a = gamma
    );
}

std::tuple<double,double,double> VoxelModel::currentLearningRates() const {
    py::gil_scoped_acquire gil;
    if (py_->optimizer_py.is_none()) {
        return {std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()};
    }
    py::list groups = py_->optimizer_py.attr("param_groups");
    auto g0 = groups[0].cast<py::dict>();
    auto g1 = groups[1].cast<py::dict>();
    auto g2 = groups[2].cast<py::dict>();
    double geo = g0["lr"].cast<double>();
    double sh0 = g1["lr"].cast<double>();
    double shs = g2["lr"].cast<double>();
    return {geo, sh0, shs};
}

VoxelModel::StatPkg
VoxelModel::computeTrainingStat(const std::vector<MiniCam>& cams) {
    // Mirrors SVAdaptive.compute_training_stat (but uses our renderer)
    freezeVoxGeo();

    const int64_t N = center_.size(0);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_type_);
    // auto max_w            = torch::zeros({N,1}, opts);
    this->max_w_.zero_();
    auto min_samp_interval= torch::full ({N,1}, 1e30f, opts);
    auto view_cnt         = torch::zeros({N,1}, opts);

    for (const auto& cam : cams) {
        // Render with track_max_w; our wrapper already returns "max_w"
        // (your VoxelModel::render already returns 'max_w' in pkg)
        auto pkg = render(cam);
        if (!pkg.count("max_w") || !pkg.at("max_w").defined())
            continue;

        auto max_w_i = pkg["max_w"].to(device_type_);
        this->max_w_ = torch::maximum(this->max_w_, max_w_i);

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
    return { this->max_w_.contiguous(), min_samp_interval.contiguous(), view_cnt.contiguous() };
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

void VoxelModel::schedulerStep()
{
    py::gil_scoped_acquire gil;
    if (!py_->scheduler_py.is_none())
        py_->scheduler_py.attr("step")();
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

// torch::Tensor VoxelModel::subdivisionPriority() const {
//     py::gil_scoped_acquire gil;
//     return py_->svm.attr("subdivision_priority")
//                     .cast<torch::Tensor>()
//                     .to(device_type_)
//                     .contiguous();
// }

// void VoxelModel::resetSubdivisionPriority() {
//     py::gil_scoped_acquire gil;
//     if (py_->svm.attr("reset_subdivision_priority").is_none()) return;
//     py_->svm.attr("reset_subdivision_priority")();
// }

// void VoxelModel::freezeVoxGeo() {
//     py::gil_scoped_acquire gil;
//     if (py_->svm.attr("freeze_vox_geo").is_none()) return;
//     py_->svm.attr("freeze_vox_geo")();
// }

// void VoxelModel::unfreezeVoxGeo() {
//     py::gil_scoped_acquire gil;
//     if (py_->svm.attr("unfreeze_vox_geo").is_none()) return;
//     py_->svm.attr("unfreeze_vox_geo")();
// }

torch::Tensor VoxelModel::subdivisionPriority() const {
    auto g = this->subdiv_p_.grad();
    if (g.defined() && g.dim() == 2 && g.size(1) == 1) g = g.squeeze(1);
    return g;
}

void VoxelModel::resetSubdivisionPriority() {
    this->subdiv_p_.mutable_grad() = torch::Tensor();
}

void VoxelModel::freezeVoxGeo() {
    py::gil_scoped_acquire gil;

    // care_idx = torch.arange(self.num_voxels, device="cuda")
    const int64_t N = this->center_.size(0);
    auto care_idx = torch::arange(
        N, torch::TensorOptions().dtype(torch::kLong).device(device_type_));

    // with torch.no_grad(): GatherGeoParams.apply(...)
    py::object no_grad_cm = py::module_::import("torch").attr("no_grad")();
    no_grad_cm.attr("__enter__")();
    try {
        py::object Gather = py::module_::import("svraster_cuda.renderer").attr("GatherGeoParams");
        // returns a torch.Tensor
        py::object frozen = Gather.attr("apply")(this->vox_key_, care_idx, this->_geo_grid_pts_);

        // Persist on the Python SVM so vox_fn() will use it
        py_->svm.attr("frozen_vox_geo") = frozen;

        // Disable grads on grid points parameter (same as Python)
        this->_geo_grid_pts_.set_requires_grad(false);

        no_grad_cm.attr("__exit__")(py::none(), py::none(), py::none());
    } catch (...) {
        // make sure we exit the context on error too
        no_grad_cm.attr("__exit__")(py::none(), py::none(), py::none());
        throw;
    }
}

// VoxelModel::unfreezeVoxGeo — mirror SVRenderer.unfreeze_vox_geo
void VoxelModel::unfreezeVoxGeo() {
    py::gil_scoped_acquire gil;

    // Remove the attribute entirely (hasattr() must return False)
    if (py::hasattr(py_->svm, "frozen_vox_geo")) {
        py::delattr(py_->svm, "frozen_vox_geo");
    }
    this->_geo_grid_pts_.set_requires_grad(true);
}

std::unordered_map<std::string, torch::Tensor>
VoxelModel::render(const MiniCam&              cam) const
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
    // d["geo_grid_pts"]      = py_->svm.attr("_geo_grid_pts");
    // d["sh0"]               = py_->svm.attr("_sh0");
    // d["shs"]               = py_->svm.attr("_shs");
    // d["subdiv_p"]          = py_->svm.attr("_subdiv_p");        // gradient used as priority
    // d["octpath"]           = py_->svm.attr("octpath");
    // d["octlevel"]          = py_->svm.attr("octlevel");
    // d["center"]            = py_->svm.attr("vox_center");
    // d["vox_size"]          = py_->svm.attr("vox_size");
    // d["vox_key"]           = py_->svm.attr("vox_key");
    // d["active_sh_degree"]  = py::int_(this->active_sh_degree_);
    d["geo_grid_pts"]      = this->_geo_grid_pts_;
    d["sh0"]               = this->sh0_;
    d["shs"]               = this->shs_;
    d["subdiv_p"]          = this->subdiv_p_;     // gradient used as priority
    d["octpath"]           = this->oct_path_;
    d["octlevel"]          = this->oct_level_;
    d["center"]            = this->center_;
    d["vox_size"]          = this->size_;
    d["vox_key"]           = this->vox_key_;
    d["active_sh_degree"]  = py::int_(this->active_sh_degree_);

    /* --------------------------------------------------------------------- */
    /* 4) call python                                                        */
    /* --------------------------------------------------------------------- */
    py::object py_cam  = MiniCam_to_py(cam);
    py::object py_out;
    try {
        py_out = py_render(py_cam, d);
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

void VoxelModel::applyTvOnDensityField(float lambda_tv_density) {
    // Must NOT be inside a NoGrad guard — we want to add to _geo_grid_pts.grad
    py::gil_scoped_acquire gil;
    try {
        // Uses SVProperties.apply_tv_on_density_field(self, weight)
        // which sets up grad if needed and calls svraster_cuda.grid_loss_bw.total_variation
        py_->svm.attr("apply_tv_on_density_field")(lambda_tv_density);
    } catch (const py::error_already_set& e) {
        std::cerr << "[TV] apply_tv_on_density_field failed: " << e.what() << std::endl;
    }
}

py::object VoxelModel::schedulerStateDict()
{
    py::gil_scoped_acquire gil;
    if (py_->scheduler_py.is_none()) return py::none();
    return py_->scheduler_py.attr("state_dict")();
}

void VoxelModel::schedulerLoadStateDict(const py::object& state)
{
    if (state.is_none()) return;
    py::gil_scoped_acquire gil;
    if (!py_->scheduler_py.is_none())
        py_->scheduler_py.attr("load_state_dict")(state);
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

torch::Tensor VoxelModel::voxSize() const {
    torch::Tensor out = (size_.dim() == 1) ? size_.unsqueeze(1) : size_; // [N,1]
    // Light stats; item<>() syncs but is fine occasionally
    auto flat = out.view(-1);
    float minv = flat.min().item<float>();
    float maxv = flat.max().item<float>();
    float mean = flat.mean().item<float>();
    // std::cout << "[DBG][voxSize] shape=" << out.sizes()
    //           << " N=" << out.size(0)
    //           << " min/mean/max=" << minv << "/" << mean << "/" << maxv
    //           << std::endl;
    return out;
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

torch::Tensor VoxelModel::SceneCenter() const {
    return this->scene_center_;
}

torch::Tensor VoxelModel::SceneExtent() const {
    return this->scene_extent_;
}

float VoxelModel::paramL2(const char* name) {
    py::gil_scoped_acquire gil;
    auto t = py_->svm.attr(name).cast<torch::Tensor>();
    return t.defined()? t.norm().item<float>() : 0.f;
}

float VoxelModel::gradL2(const char* name) {
    py::gil_scoped_acquire gil;
    auto p = py_->svm.attr(name).cast<torch::Tensor>();
    auto g = p.grad();
    return (g.defined()? g.norm().item<float>() : 0.f);
}

void VoxelModel::debugParamChain() {
    py::gil_scoped_acquire gil;

    auto p_geo = py_->svm.attr("_geo_grid_pts").cast<torch::Tensor>();
    auto p_sh0 = py_->svm.attr("_sh0").cast<torch::Tensor>();
    auto p_shs = py_->svm.attr("_shs").cast<torch::Tensor>();

    auto g_geo = p_geo.grad();
    auto g_sh0 = p_sh0.grad();
    auto g_shs = p_shs.grad();

    auto ptr = [](const torch::Tensor& t)->uintptr_t { return (uintptr_t)t.data_ptr(); };
    auto nrm = [](const torch::Tensor& t)->double { return (t.defined() ? t.norm().item<double>() : -1.0); };

    std::cout << std::fixed << std::setprecision(6)
              << "[autograd] _geo  req=" << p_geo.requires_grad()
              << "  p="  << ptr(p_geo)
              << "  ||p||="   << nrm(p_geo)
              << "  grad? "   << (g_geo.defined())
              << "  ||g||="   << nrm(g_geo) << "\n"
              << "[autograd] _sh0  req=" << p_sh0.requires_grad()
              << "  p="  << ptr(p_sh0)
              << "  ||p||="   << nrm(p_sh0)
              << "  grad? "   << (g_sh0.defined())
              << "  ||g||="   << nrm(g_sh0) << "\n"
              << "[autograd] _shs  req=" << p_shs.requires_grad()
              << "  p="  << ptr(p_shs)
              << "  ||p||="   << nrm(p_shs)
              << "  grad? "   << (g_shs.defined())
              << "  ||g||="   << nrm(g_shs) << std::endl;
}

void VoxelModel::debugOptimizer() {
    py::gil_scoped_acquire gil;
    if (py_->optimizer_py.is_none()) {
        std::cout << "[optim] None\n"; return;
    }
    py::list groups = py_->optimizer_py.attr("param_groups");
    for (ssize_t i = 0; i < (ssize_t)groups.size(); ++i) {
        auto g = groups[i].cast<py::dict>();
        double lr = py::float_(g["lr"]);
        py::tuple params = g["params"].cast<py::tuple>();
        std::cout << "[optim] group " << i << " lr=" << lr
                  << " nparams=" << params.size() << "\n";
        for (ssize_t j = 0; j < (ssize_t)params.size(); ++j) {
            auto t = py::reinterpret_borrow<py::object>(params[j]).cast<torch::Tensor>();
            auto tg = t.grad();
            std::cout << "   - shape=" << t.sizes()
                      << " req=" << t.requires_grad()
                      << " ||p||=" << (t.defined()? t.norm().item<double>(): -1.0)
                      << " grad? " << tg.defined()
                      << " ||g||=" << (tg.defined()? tg.norm().item<double>(): -1.0)
                      << "\n";
        }
    }
}

torch::Tensor VoxelModel::snapParam(const char* name) {
    py::gil_scoped_acquire gil;
    auto t = py_->svm.attr(name).cast<torch::Tensor>();
    return t.detach().clone();               // CPU clone if you prefer
}

double VoxelModel::deltaFrom(const char* name, const torch::Tensor& prev) {
    py::gil_scoped_acquire gil;
    auto t = py_->svm.attr(name).cast<torch::Tensor>();
    auto d = (t - prev).norm().item<double>();
    auto n = t.norm().item<double>();
    return (n > 0.0 ? d / n : 0.0);
}

// void VoxelModel::savePly(const std::filesystem::path& result_path)
// {
//     torch::NoGradGuard no_grad;

//     // --- Gather per-voxel attributes ---
//     // positions
//     torch::Tensor xyz = center_.detach().cpu().contiguous();            // [N,3] float
//     // dummy normals
//     torch::Tensor normals = torch::zeros_like(xyz);                      // [N,3] float
//     // size (edge length)
//     torch::Tensor size = size_.detach().unsqueeze(1).cpu().contiguous(); // [N,1] float
//     // level
//     torch::Tensor level = oct_level_.detach().view({-1,1}).cpu().contiguous(); // [N,1] int8

//     // color from SH0
//     torch::Tensor rgb = sh0_.detach();
//     if (rgb.dim() == 3 && rgb.size(1) == 1 && rgb.size(2) == 3) {
//         rgb = rgb.view({-1, 3});
//     }
//     // Clamp to displayable range and convert to u8
//     rgb = rgb.clamp(0, 1).mul(255).to(torch::kUInt8).cpu().contiguous(); // [N,3] u8

//     // --- Open the file ---
//     std::filebuf fb_binary;
//     fb_binary.open(result_path, std::ios::out | std::ios::binary);
//     std::ostream outstream_binary(&fb_binary);
//     if (outstream_binary.fail())
//         throw std::runtime_error("failed to open " + result_path.string());

//     tinyply::PlyFile ply;

//     // xyz
//     ply.add_properties_to_element(
//         "vertex", {"x","y","z"},
//         tinyply::Type::FLOAT32, xyz.size(0),
//         reinterpret_cast<uint8_t*>(xyz.data_ptr<float>()),
//         tinyply::Type::INVALID, 0);

//     // normals
//     ply.add_properties_to_element(
//         "vertex", {"nx","ny","nz"},
//         tinyply::Type::FLOAT32, normals.size(0),
//         reinterpret_cast<uint8_t*>(normals.data_ptr<float>()),
//         tinyply::Type::INVALID, 0);

//     // size
//     ply.add_properties_to_element(
//         "vertex", {"size"},
//         tinyply::Type::FLOAT32, size.size(0),
//         reinterpret_cast<uint8_t*>(size.data_ptr<float>()),
//         tinyply::Type::INVALID, 0);

//     // level (int8 -> store as int8)
//     ply.add_properties_to_element(
//         "vertex", {"level"},
//         tinyply::Type::INT8, level.size(0),
//         reinterpret_cast<uint8_t*>(level.data_ptr<int8_t>()),
//         tinyply::Type::INVALID, 0);

//     // color
//     ply.add_properties_to_element(
//         "vertex", {"red","green","blue"},
//         tinyply::Type::UINT8, rgb.size(0),
//         reinterpret_cast<uint8_t*>(rgb.data_ptr<uint8_t>()),
//         tinyply::Type::INVALID, 0);

//     // write
//     ply.write(outstream_binary, /*isBinary*/ true);
//     fb_binary.close();
// }

// VoxelModel::savePly — WebGL viewer compatible (SH degree 1)
namespace {
constexpr int MAX_NUM_LEVELS = 16;

static inline std::array<int64_t,3> decode_ijk(uint64_t path, int lv) {
    path >>= (3 * (MAX_NUM_LEVELS - lv));
    int64_t i=0,j=0,k=0;
    for (int l=0; l<lv; ++l) {
        uint64_t bits = (path & 0x7u);
        i |= static_cast<int64_t>((bits >> 2) & 0x1u) << l;
        j |= static_cast<int64_t>((bits >> 1) & 0x1u) << l;
        k |= static_cast<int64_t>((bits >> 0) & 0x1u) << l;
        path >>= 3;
    }
    return {i,j,k};
}

// // --- helper: encode (i,j,k,L) into the SAME bit layout used by decode_ijk() ---
// static inline uint64_t encode_ijk_to_octpath(int i, int j, int k, int L) {
//     // decode_ijk() expects bits starting at position 3*(MAX_NUM_LEVELS - L)
//     const int base = 3 * (MAX_NUM_LEVELS - L);
//     uint64_t p = 0ull;
//     for (int l = 0; l < L; ++l) {
//         uint64_t bits =
//             (uint64_t)(((i >> l) & 1) << 2) |
//             (uint64_t)(((j >> l) & 1) << 1) |
//             (uint64_t)(((k >> l) & 1) << 0);
//         p |= (bits << (base + 3 * l));
//     }
//     return p;
// }

static torch::Tensor decode_centers_from_octree(const torch::Tensor& octpath,
                                                const torch::Tensor& octlevel,
                                                const torch::Tensor& scene_center,
                                                const torch::Tensor& scene_extent) {
    auto op = octpath.contiguous().view({-1}).to(torch::kInt64);
    auto lv = octlevel.contiguous().view({-1}).to(torch::kInt32);

    const int64_t N = op.size(0);
    const float cx = scene_center[0].item<float>();
    const float cy = scene_center[1].item<float>();
    const float cz = scene_center[2].item<float>();
    const float extent = scene_extent.item<float>();

    const float minx = cx - 0.5f * extent;
    const float miny = cy - 0.5f * extent;
    const float minz = cz - 0.5f * extent;

    auto xyz = torch::empty({N,3}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    auto xyz_a = xyz.accessor<float,2>();
    const auto* op_ptr = op.data_ptr<int64_t>();
    const auto* lv_ptr = lv.data_ptr<int32_t>();

    for (int64_t n=0; n<N; ++n) {
        const uint64_t p = static_cast<uint64_t>(op_ptr[n]);
        const int l = lv_ptr[n];
        const auto ijk = decode_ijk(p, l);
        const float vox_size = std::ldexp(extent, -l);
        xyz_a[n][0] = minx + (static_cast<float>(ijk[0]) + 0.5f) * vox_size;
        xyz_a[n][1] = miny + (static_cast<float>(ijk[1]) + 0.5f) * vox_size;
        xyz_a[n][2] = minz + (static_cast<float>(ijk[2]) + 0.5f) * vox_size;
    }
    return xyz;
}

// simple stride sampler for a vector of indices
static std::vector<int64_t> stride_sample(const std::vector<int64_t>& idxs, size_t want) {
    if (idxs.size() <= want) return idxs;
    std::vector<int64_t> out; out.reserve(want);
    double step = double(idxs.size()) / double(want);
    for (size_t k=0; k<want; ++k) {
        size_t pos = size_t(k * step);
        if (pos >= idxs.size()) pos = idxs.size()-1;
        out.push_back(idxs[pos]);
    }
    return out;
}
} // namespace

void VoxelModel::savePly(const std::filesystem::path& result_path)
{
    int target_max_voxels =1000000;
    torch::NoGradGuard ng;
    namespace fs = std::filesystem;
    if (!result_path.parent_path().empty())
        fs::create_directories(result_path.parent_path());

    // Pull to CPU
    auto op_cpu   = oct_path_.detach().to(torch::kCPU).contiguous();        // [N]
    auto lv_cpu   = oct_level_.detach().to(torch::kCPU).contiguous();       // [N] or [N,1]
    auto sc_cpu   = scene_center_.detach().to(torch::kCPU).contiguous();    // [3]
    auto se_cpu   = scene_extent_.detach().to(torch::kCPU).contiguous();    // [1]
    auto sh0_cpu  = sh0_.detach().to(torch::kCPU).contiguous();             // [N,3] or [N,1,3]
    auto shs_cpu  = shs_.detach().to(torch::kCPU).contiguous();             // [N,K,3] (K>=0)
    auto voxkey   = vox_key_.detach().to(torch::kCPU).contiguous();         // [N,8]
    auto geo_cpu  = _geo_grid_pts_.detach().to(torch::kCPU).contiguous();   // [G]

    // Decode centers (viewer’s computation)
    torch::Tensor xyz = decode_centers_from_octree(op_cpu, lv_cpu, sc_cpu, se_cpu); // [N,3]
    const int64_t N = xyz.size(0);
    if (N == 0) { std::cerr << "[savePly] No voxels.\n"; return; }

    // Flatten lv to [N]
    auto lv_i32 = lv_cpu.view({-1}).to(torch::kInt32).contiguous();
    auto lv_ptr = lv_i32.data_ptr<int32_t>();
    int lv_min =  999, lv_max = -999;
    for (int64_t n=0; n<N; ++n) { lv_min = std::min(lv_min, lv_ptr[n]); lv_max = std::max(lv_max, lv_ptr[n]); }

    // Build SH dc = f_dc_0..2
    torch::Tensor fdc;
    if (sh0_cpu.dim()==3 && sh0_cpu.size(1)==1 && sh0_cpu.size(2)==3)      fdc = sh0_cpu.view({N,3});
    else if (sh0_cpu.dim()==2 && sh0_cpu.size(1)==3)                        fdc = sh0_cpu;
    else { std::cerr << "[savePly] Unexpected sh0_ shape " << sh0_cpu.sizes() << "\n"; return; }
    fdc = fdc.to(torch::kFloat32).contiguous();

    // SH rest: enforce degree-1 → exactly 9 floats per voxel
    int64_t K = (shs_cpu.dim()>=2) ? shs_cpu.size(1) : 0;
    if (active_sh_degree_ < 1) {
        std::cerr << "[savePly] WARNING: active_sh_degree_ < 1; exporting degree-1 band as zeros.\n";
    }
    if (K < 3) {
        std::cerr << "[savePly] NOTE: shs_ has only " << K
                  << " coeffs per channel; padding to degree-1 (3) with zeros.\n";
    }
    torch::Tensor shs_band1;
    if (K >= 3 && shs_cpu.dim()==3 && shs_cpu.size(2)==3) {
        shs_band1 = shs_cpu.index({torch::indexing::Slice(),
                                   torch::indexing::Slice(0,3),
                                   torch::indexing::Slice()}).contiguous();  // [N,3,3]
    } else {
        // pad zeros to [N,3,3] if needed
        shs_band1 = torch::zeros({N,3,3}, torch::dtype(torch::kFloat32));
        if (K > 0 && shs_cpu.dim()==3 && shs_cpu.size(2)==3) {
            auto copyK = std::min<int64_t>(K, 3);
            shs_band1.index_put_({torch::indexing::Slice(),
                                  torch::indexing::Slice(0,copyK),
                                  torch::indexing::Slice()},
                                 shs_cpu.index({torch::indexing::Slice(),
                                                torch::indexing::Slice(0,copyK),
                                                torch::indexing::Slice()}).to(torch::kFloat32));
        }
    }
    auto frest = shs_band1.view({N, 9}).to(torch::kFloat32).contiguous(); // [N,9]

    // Vox key sanity
    if (!(voxkey.dim()==2 && voxkey.size(0)==N && voxkey.size(1)==8)) {
        std::cerr << "[savePly] ERROR: vox_key_ must be [N,8]; got " << voxkey.sizes() << "\n";
        return;
    }
    if (geo_cpu.dim()!=1) geo_cpu = geo_cpu.view({-1});
    const int64_t G = geo_cpu.size(0);

    // Stats (helpful when debugging “black”)
    auto fdc_min = std::get<0>(fdc.aminmax()); auto fdc_max = std::get<1>(fdc.aminmax());
    auto fre_min = std::get<0>(frest.aminmax()); auto fre_max = std::get<1>(frest.aminmax());
    auto geo_min = std::get<0>(geo_cpu.aminmax()); auto geo_max = std::get<1>(geo_cpu.aminmax());
    std::cout << "[savePly] N=" << N
              << "  lv:[" << lv_min << "," << lv_max << "]"
              << "  f_dc:[" << fdc_min.item<float>() << "," << fdc_max.item<float>() << "]"
              << "  f_rest:[" << fre_min.item<float>() << "," << fre_max.item<float>() << "]"
              << "  geo:[" << geo_min.item<float>() << "," << geo_max.item<float>() << "]\n";

    // AABB of xyz (just to spot wildly wrong scale/offset)
    auto xyz_min = std::get<0>(xyz.aminmax(0));
    auto xyz_max = std::get<1>(xyz.aminmax(0));
    std::cout << "[savePly] AABB min=("
              << xyz_min[0].item<float>() << "," << xyz_min[1].item<float>() << "," << xyz_min[2].item<float>()
              << ") max=("
              << xyz_max[0].item<float>() << "," << xyz_max[1].item<float>() << "," << xyz_max[2].item<float>()
              << ")\n";

    // Optional: level-aware downsample to keep ~target_max_voxels (coarse first)
    std::vector<int64_t> keep_idx;
    keep_idx.reserve(N);
    if (target_max_voxels > 0 && N > target_max_voxels) {
        std::unordered_map<int, std::vector<int64_t>> by_level;
        by_level.reserve(lv_max-lv_min+1);
        for (int64_t n=0; n<N; ++n) by_level[lv_ptr[n]].push_back(n);

        size_t remaining = target_max_voxels;
        for (int L = lv_min; L <= lv_max && remaining > 0; ++L) {
            auto& bucket = by_level[L];
            if (bucket.empty()) continue;
            if (bucket.size() <= remaining) {
                keep_idx.insert(keep_idx.end(), bucket.begin(), bucket.end());
                remaining -= bucket.size();
            } else {
                auto sampled = stride_sample(bucket, remaining);
                keep_idx.insert(keep_idx.end(), sampled.begin(), sampled.end());
                remaining = 0;
            }
        }
        std::sort(keep_idx.begin(), keep_idx.end());
        std::cout << "[savePly] Downsampled " << N << " → " << keep_idx.size()
                  << " voxels for viewer.\n";
    } else {
        keep_idx.resize(N); std::iota(keep_idx.begin(), keep_idx.end(), 0);
    }
    const int64_t M = (int64_t)keep_idx.size();
    auto idx_t = torch::from_blob(keep_idx.data(), {M}, torch::TensorOptions().dtype(torch::kLong)).clone();

    // Slice tensors
    xyz   = xyz.index_select(0, idx_t).contiguous();
    fdc   = fdc.index_select(0, idx_t).contiguous();
    frest = frest.index_select(0, idx_t).contiguous();
    voxkey= voxkey.index_select(0, idx_t).contiguous();

    // octpath / level to std::vector payloads
    std::vector<uint32_t> op_u32; op_u32.reserve(M);
    std::vector<uint8_t>  lv_u8;  lv_u8.reserve(M);
    {
        auto op64 = op_cpu.view({-1}).to(torch::kInt64);
        const int64_t* op_ptr = op64.data_ptr<int64_t>();
        for (auto i : keep_idx) op_u32.push_back(static_cast<uint32_t>(op_ptr[i]));
        for (auto i : keep_idx) lv_u8 .push_back(static_cast<uint8_t>(lv_ptr[i]));
    }

    // grid0..7_value
    std::array<torch::Tensor,8> grid_vals;
    int64_t out_of_range = 0;
    for (int c=0; c<8; ++c) {
        auto key_c = voxkey.index({torch::indexing::Slice(), c}).to(torch::kLong).contiguous();
        // Track OOR before clamping (debug)
        auto oor = (key_c < 0) | (key_c >= G);
        out_of_range += oor.sum().item<int64_t>();
        key_c = torch::clamp(key_c, 0, G-1);
        grid_vals[c] = geo_cpu.index_select(0, key_c).to(torch::kFloat32).contiguous(); // [M]
    }
    if (out_of_range > 0)
        std::cerr << "[savePly] WARNING: vox_key has " << out_of_range
                  << " out-of-range indices (clamped). Check syncFromPython().\n";

    // Write PLY (binary)
    std::filebuf fb;
    fb.open(result_path, std::ios::out | std::ios::binary);
    std::ostream out(&fb);
    if (out.fail()) throw std::runtime_error("savePly: open failed: " + result_path.string());

    tinyply::PlyFile ply;

    // x,y,z
    ply.add_properties_to_element(
        "vertex", {"x","y","z"},
        tinyply::Type::FLOAT32, M,
        reinterpret_cast<uint8_t*>(xyz.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // octpath (uint32)
    ply.add_properties_to_element(
        "vertex", {"octpath"},
        tinyply::Type::UINT32, M,
        reinterpret_cast<uint8_t*>(op_u32.data()),
        tinyply::Type::INVALID, 0);

    // octlevel (uint8)
    ply.add_properties_to_element(
        "vertex", {"octlevel"},
        tinyply::Type::UINT8, M,
        reinterpret_cast<uint8_t*>(lv_u8.data()),
        tinyply::Type::INVALID, 0);

    // f_dc_0..2
    {
        std::vector<std::string> names = {"f_dc_0","f_dc_1","f_dc_2"};
        ply.add_properties_to_element(
            "vertex", names,
            tinyply::Type::FLOAT32, M,
            reinterpret_cast<uint8_t*>(fdc.data_ptr<float>()),
            tinyply::Type::INVALID, 0);
    }

    // f_rest_0..8 (degree-1 only)
    {
        std::vector<std::string> names; names.reserve(9);
        for (int i=0; i<9; ++i) names.emplace_back("f_rest_" + std::to_string(i));
        ply.add_properties_to_element(
            "vertex", names,
            tinyply::Type::FLOAT32, M,
            reinterpret_cast<uint8_t*>(frest.data_ptr<float>()),
            tinyply::Type::INVALID, 0);
    }

    // grid0..7_value
    for (int c=0; c<8; ++c) {
        std::string nm = "grid" + std::to_string(c) + "_value";
        ply.add_properties_to_element(
            "vertex", {nm},
            tinyply::Type::FLOAT32, M,
            reinterpret_cast<uint8_t*>(grid_vals[c].data_ptr<float>()),
            tinyply::Type::INVALID, 0);
    }

    // Comments: handy for debugging in the viewer console
    try {
        const float cx = sc_cpu[0].item<float>(), cy = sc_cpu[1].item<float>(), cz = sc_cpu[2].item<float>();
        const float ex = se_cpu.item<float>();
        ply.get_comments().push_back("scene_center " + std::to_string(cx) + " " + std::to_string(cy) + " " + std::to_string(cz));
        ply.get_comments().push_back("scene_extent " + std::to_string(ex));
        ply.get_comments().push_back("active_sh_degree 1"); // we export degree-1
    } catch (...) {}

    ply.write(out, /*binary*/ true);
    fb.close();

    std::cout << "[savePly] Wrote " << M << " voxels to " << result_path
              << "  (viewer SH degree=1). "
              << (M>1200000 ? "NOTE: Consider lowering to ≤1M for FPS." : "")
              << "\n";
}

// void VoxelModel::initOptimizer(float geo_lr, float sh0_lr, float shs_lr,
//                                float beta1, float beta2, float eps) {
//     py::gil_scoped_acquire gil;
//     py::module sadam = py::module::import("svraster_cuda.sparse_adam");
//     py::object SparseAdam = sadam.attr("SparseAdam");

//     py::list groups;
//     {
//         py::dict g0; g0["params"] = py::make_tuple(py_->svm.attr("_geo_grid_pts")); g0["lr"] = geo_lr; groups.append(g0);
//     }
//     {
//         py::dict g1; g1["params"] = py::make_tuple(py_->svm.attr("_sh0"));          g1["lr"] = sh0_lr; groups.append(g1);
//     }
//     {
//         py::dict g2; g2["params"] = py::make_tuple(py_->svm.attr("_shs"));          g2["lr"] = shs_lr; groups.append(g2);
//     }

//     // Match Python: SparseAdam(groups, betas=(beta1,beta2), eps=eps)
//     // (global lr is unused because each group has its own 'lr')
//     py_->optimizer_py = SparseAdam(groups,
//                                    "betas"_a = py::make_tuple(beta1, beta2),
//                                    "eps"_a   = eps);
// }

// void VoxelModel::rebuildOptimizer(float geo_lr, float sh0_lr, float shs_lr,
//                                   float beta1, float beta2, float eps) {
//     {
//         py::gil_scoped_acquire gil;
//         py_->svm.attr("_geo_grid_pts").attr("requires_grad_")(true);
//         py_->svm.attr("_sh0").attr("requires_grad_")(true);
//         py_->svm.attr("_shs").attr("requires_grad_")(true);
//     }
//     // Same as initOptimizer, but you may want to preserve your current LRs.
//     initOptimizer(geo_lr, sh0_lr, shs_lr, beta1, beta2, eps);
// }

// void VoxelModel::setLearningRates(float geo_lr, float sh0_lr, float shs_lr) {
//     py::gil_scoped_acquire gil;
//     if (py_->optimizer_py.is_none()) return;
//     py::list groups = py_->optimizer_py.attr("param_groups");
//     auto g0 = groups[0].cast<py::dict>();
//     auto g1 = groups[1].cast<py::dict>();
//     auto g2 = groups[2].cast<py::dict>();
//     g0["lr"] = geo_lr;
//     g1["lr"] = sh0_lr;
//     g2["lr"] = shs_lr;
// }

// float VoxelModel::multiStepDecay(int iter, float base_lr,
//                             const std::vector<int>& milestones,
//                             float gamma /* e.g. 0.33f */)
// {
//     if (milestones.empty() || gamma <= 0.f || gamma >= 1.f) return base_lr;
//     int passed = 0;
//     for (int m : milestones) if (m > 0 && iter >= m) ++passed;
//     if (passed == 0) return base_lr;
//     return base_lr * std::pow(gamma, passed);
// }

} // namespace sv

