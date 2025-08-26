#include "include_voxel/voxel_model.h"
#include <fstream>
#include <sstream>

// ------------------------------------------------------------------
// insert these *statics* at the top of your .cpp file (outside any function):
static std::vector<float> debug_all_new_pts;
static std::vector<float> debug_all_new_cols;
static std::vector<float> debug_all_new_centers;
static std::vector<float> debug_all_new_sizes;
static std::vector<float> debug_all_new_voxcols;
// ------------------------------------------------------------------

static std::string shape_str(const torch::Tensor& t)
{
    std::ostringstream os; os << "(";
    for (int d=0; d<t.dim(); ++d){ os << t.size(d);
        if (d+1<t.dim()) os << ","; }
    os << ")"; return os.str();
}

static torch::Tensor select_new(const torch::Tensor& t,
                                const torch::Tensor& mask)
{
    // mask is (V',1) bool on CUDA;  result keeps dims of t
    torch::Tensor idx = torch::nonzero(mask).squeeze(1);  // 1‑D int64
    return t.index_select(0, idx);
}

static std::string nice(const torch::Tensor& t,
                        int head = 3, int tail = 3)
{
    std::ostringstream os;
    os << "(";
    for (int d = 0; d < t.dim(); ++d) {
        os << t.size(d);
        if (d + 1 < t.dim()) os << ",";
    }
    os << ")";
    if (t.dim() == 2 && t.size(1) == 8) {           // vox_key_
        auto cpu = t.to(torch::kCPU);
        os << "  [";
        for (int i = 0; i < head && i < cpu.size(0); ++i)
            os << cpu[i][0].item<int64_t>() << ",";
        os << "...";
        for (int i = std::max<int>(head, cpu.size(0)-tail);
                 i < cpu.size(0); ++i)
            os << cpu[i][0].item<int64_t>() << ",";
        os << "]";
    }
    return os.str();
}

static std::string row_info(const torch::Tensor& t, int n = 6)
{
    std::ostringstream os;
    auto cpu = t.flatten().cpu();
    os << "[";
    for (int i = 0; i < std::min<int>(cpu.size(0), n); ++i)
        os << cpu[i].item<float>() << (i+1<n?",":"");
    if (cpu.size(0) > n) os << "...";
    os << "]";
    return os.str();
}

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

// void VoxelModel::createFromPcd(
//     const std::map<point3D_id_t, Point3D>& pcd,
//     float spatial_lr_scale,
//     const std::vector<std::shared_ptr<VoxelKeyframe>>& keyframes)
// {
//     // '''
//     // octree
//     // '''
//     /* constants (same as before) */
//     constexpr float kPadding     = 0.05f;
//     constexpr float kTargetVoxel = 0.05f;
//     constexpr float kGeoInit     = 4.0f;
//     constexpr float kSh0Init     = 0.5f;
//     constexpr float kShsInit     = 0.0f;

//     this->spatial_lr_scale_ = spatial_lr_scale;

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
//     // std::cout << "mn = " << mn << ", mx = " << mx << std::endl;
//     float half_side     = (mx - mn).max().item<float>();
//     float inside_extent = 2.f * half_side + kPadding;
//     float scene_extent  = inside_extent * std::pow(2.f, float(outside_level_));
//     torch::Tensor scene_center = 0.5f * (mn + mx);
//     torch::Tensor extent       = torch::tensor({scene_extent},
//                                  torch::TensorOptions().dtype(torch::kFloat32)
//                                                         .device(device_type_));
//     // torch::Tensor extent       = torch::tensor({half_side},
//     //                             torch::TensorOptions().dtype(torch::kFloat32)
//     //                                                 .device(device_type_));

//     /* 3) choose leaf level from target voxel size (unchanged) */
//     int leaf_level = std::ceil(std::log2(scene_extent / kTargetVoxel));
//     torch::Tensor octlevel0 = torch::full({N_pts,1}, leaf_level,
//         torch::TensorOptions().dtype(torch::kInt8).device(device_type_));

//     leaf_level_    = leaf_level;       // after you compute leaf_level
//     scene_center  = scene_center;     // keep on CUDA
//     scene_extent  = extent;           // (1,) CUDA

//     /* 4) xyz → octpath (unchanged) */
//     py::gil_scoped_acquire gil;
//     static py::module octree = []{
//         py::module sys = py::module::import("sys");
//         sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
//         return py::module::import("src.utils.octree_utils");
//     }();
//     torch::Tensor octpath0 = octree.attr("xyz_2_octpath")(
//                                  pts, octlevel0, scene_center, extent
//                              ).cast<torch::Tensor>().to(device_type_);

//     /* 5) collapse duplicates (unchanged) */
//     torch::Tensor keys = torch::cat({octpath0.to(torch::kLong).view({-1,1}),
//                                      octlevel0.to(torch::kLong)}, 1);
//     auto uq = at::unique_dim(keys.cpu(), 0, /*sorted=*/true,
//                              /*return_inverse=*/true, /*return_counts=*/true);
//     torch::Tensor uniq_keys = std::get<0>(uq).to(device_type_, torch::kLong);
//     torch::Tensor inv_id    = std::get<1>(uq).to(device_type_, torch::kLong);
//     torch::Tensor counts    = std::get<2>(uq).to(torch::kFloat32)
//                                              .unsqueeze(1).to(device_type_);
//     const int64_t M = uniq_keys.size(0);

//     torch::Tensor rgb_sum = torch::zeros({M,3}, cols.options());
//     rgb_sum.index_add_(0, inv_id, cols);
//     torch::Tensor rgb_avg = (rgb_sum / counts).detach();

//     torch::Tensor up_octpath  = uniq_keys.select(1,0).view({-1,1});
//     torch::Tensor up_octlevel = uniq_keys.select(1,1)
//                                        .to(torch::kInt8).view({-1,1});

//     /* 6) ▶ CHANGED: decode octpath (GPU tensors, no .cpu()) */
//     auto dec = octree.attr("octpath_decoding")(
//                    up_octpath, up_octlevel,     // stay on CUDA
//                    scene_center, extent         // already CUDA
//                ).cast<std::tuple<torch::Tensor,torch::Tensor>>();
//     torch::Tensor vox_centers = std::get<0>(dec).to(device_type_);
//     torch::Tensor vox_size    = std::get<1>(dec).squeeze(1).to(device_type_);

//     py::module_ np = py::module_::import("numpy");
//     // save raw point‐cloud + colors
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_pts.npy",
//                     tensor_to_numpy(pts.cpu()));
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_cols.npy",
//                     tensor_to_numpy(cols.cpu()));
//     // save voxel cell centres + edge lengths
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_vox_centers.npy",
//                     tensor_to_numpy(vox_centers.cpu()));
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_vox_size.npy",
//                     tensor_to_numpy(vox_size.cpu()));

//     /* 7) save basics (unchanged) */
//     center_       = vox_centers;
//     size_         = vox_size;
//     vox_size_inv_ = 1.0f / size_;
//     oct_path_     = up_octpath.to(torch::kLong);
//     oct_level_ = up_octlevel.to(torch::kInt8)         // <<< int8!
//                             .view({M}).clone();      // keep contiguous

//     /* 8) ▶ CHANGED: grid-point links (inputs stay on CUDA) */
//     py::tuple link = octree.attr("build_grid_pts_link")(up_octpath, up_octlevel);
//     grid_pts_key_  = link[0].cast<torch::Tensor>().to(device_type_);
//     vox_key_       = link[1].cast<torch::Tensor>().to(device_type_);
//     int64_t G      = grid_pts_key_.size(0);

//     /* 9) allocate learnables (unchanged) */
//     _geo_grid_pts_ = torch::full({G,1}, kGeoInit,
//                                 torch::kFloat32).to(device_type_)
//                      .requires_grad_(true);

//     torch::Tensor sh0_rgb = torch::full({M,3}, kSh0Init,
//                                         torch::kFloat32).to(device_type_);
//     sh0_ = sh_utils::RGB2SH(sh0_rgb).view({M,1,3}).requires_grad_(true);

//     int K = (max_sh_degree_+1)*(max_sh_degree_+1) - 1;
//     shs_ = torch::full({M,K,3}, kShsInit,
//                        torch::kFloat32).to(device_type_)
//            .requires_grad_(true);

//     subdiv_p_            = torch::ones({M,1},
//                                        torch::kFloat32).to(device_type_)
//                              .requires_grad_(true);
//     subdiv_meta_         = torch::zeros({M,1},
//                                         torch::kFloat32).to(device_type_)
//                               .requires_grad_(true);
//     // subdiv_p_grad_buffer_= torch::zeros_like(subdiv_p_);
//     subdiv_meta_.retain_grad();
//     subdiv_p_   .retain_grad();

//     // ─── **new**: allocate our two stats‐buffers on the same device ─────────
//     xyz_gradient_accum_ = torch::zeros({M,1},
//         torch::TensorOptions()
//           .dtype(torch::kFloat32)
//           .device(device_type_));
//     denom_ = torch::zeros_like(xyz_gradient_accum_);

//     voxel_error_sum_  = torch::zeros({M,1},
//                                  torch::TensorOptions()
//                                      .dtype(torch::kFloat32)
//                                      .device(device_type_));
//     voxel_hit_count_  = torch::zeros_like(voxel_error_sum_);

//     /* 10) register tensors with optimizer (unchanged) */
//     VOXEL_MODEL_TENSORS_TO_VEC

//     voxel_hash_.clear();
//     for (int i = 0; i < M; ++i) {
//         int64_t packed = ((up_octpath[i].item<int64_t>() << 8)
//                         | up_octlevel[i].item<int8_t>());
//         voxel_hash_[packed] = i;
//     }
// }

struct CameraHeader {
    int   width, height;
    float fx, fy, cx, cy;
    float znear, zfar;
    bool  valid = false;
};

struct PoseRec {
    int                    fid;
    Eigen::Vector3d        t_wc;   // camera→world
    Eigen::Quaterniond     q_wc;   // camera→world
};

// ——— Parse the txt (header + poses) ————————————————————————
static void read_pose_file(
    const std::string&        path,
    CameraHeader&             cam,
    std::vector<PoseRec>&     poses)
{
    std::ifstream ifs(path);
    if (!ifs.is_open())
        throw std::runtime_error("Cannot open pose file: " + path);

    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            std::istringstream iss(line);
            std::string tag;
            iss >> tag;
            if (tag == "#") iss >> tag;
            if (tag == "CAMERA") {
                cam.valid = true;
                iss >> cam.width >> cam.height
                    >> cam.fx >> cam.fy
                    >> cam.cx >> cam.cy
                    >> cam.znear >> cam.zfar;
            }
            continue;
        }
        PoseRec p;
        double tx, ty, tz, qx, qy, qz, qw;
        std::istringstream iss(line);
        if (!(iss >> p.fid >> tx >> ty >> tz
                  >> qx >> qy >> qz >> qw))
            continue;
        p.t_wc = {tx, ty, tz};
        p.q_wc = Eigen::Quaterniond(qw, qx, qy, qz);
        p.q_wc.normalize();
        poses.push_back(p);
    }
    if (!cam.valid)
        throw std::runtime_error("Missing CAMERA header in: " + path);
}

// ——— Build light cams for bounding ——————————————————————
static pybind11::list build_tr_cams_from_file_light(
    const std::string& path,
    float world_scale = 1.0f)
{
    namespace py = pybind11;
    py::object SimpleNS =
        py::module_::import("types").attr("SimpleNamespace");

    CameraHeader cam_hdr;
    std::vector<PoseRec> poses;
    read_pose_file(path, cam_hdr, poses);

    py::list out;
    for (auto &p : poses) {
        Eigen::Vector3d look = p.q_wc.toRotationMatrix().col(2);
        look.normalize();
        Eigen::Vector3d pos = p.t_wc * world_scale;

        auto pos_cpu  = torch::tensor(
            {float(pos.x()), float(pos.y()), float(pos.z())},
            torch::TensorOptions().dtype(torch::kFloat32));
        auto look_cpu = torch::tensor(
            {float(look.x()), float(look.y()), float(look.z())},
            torch::TensorOptions().dtype(torch::kFloat32));

        py::object np_pos    = py::cast(pos_cpu).attr("numpy")();
        py::object np_lookat = py::cast(look_cpu).attr("numpy")();

        out.append(SimpleNS(py::arg("position")=np_pos,
                            py::arg("lookat")  =np_lookat));
    }
    return out;
}

// ——— Build full SVRaster MiniCams for model_init —————————
static pybind11::list build_full_py_cams_from_file(
    const std::string& path,
    float               world_scale = 1.0f)
{
    namespace py = pybind11;

    CameraHeader cam_hdr;
    std::vector<PoseRec> poses;
    read_pose_file(path, cam_hdr, poses);

    // synthesize a temporary C++ Camera from header
    sv::Camera cpp_cam;
    cpp_cam.camera_id_ = 0;
    cpp_cam.model_id_  = 1; // pinhole
    cpp_cam.width_     = cam_hdr.width;
    cpp_cam.height_    = cam_hdr.height;
    cpp_cam.params_    = {cam_hdr.fx, cam_hdr.fy, cam_hdr.cx, cam_hdr.cy};

    py::list out;
    for (auto &p : poses) {
        Eigen::Matrix4f c2w = Eigen::Matrix4f::Identity();
        c2w.block<3,3>(0,0) = p.q_wc.toRotationMatrix().cast<float>();
        c2w.block<3,1>(0,3) = (p.t_wc * world_scale).cast<float>();

        auto c2w_cpu = torch::from_blob(
            c2w.data(), {4,4},
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone();

        // build C++ MiniCam then bridge to Python
        sv::MiniCam mc = sv::fromCamera(cpp_cam, c2w_cpu, p.fid);
        mc.near = cam_hdr.znear;

        py::object cam_py = MiniCam_to_py(mc);
        cam_py.attr("w2c")      = cam_py.attr("w2c").attr("cuda")();
        cam_py.attr("c2w")      = cam_py.attr("c2w").attr("cuda")();
        cam_py.attr("position") = cam_py.attr("c2w")
            [py::make_tuple(py::slice(0,3,1), 3)];
        cam_py.attr("lookat")   = cam_py.attr("c2w")
            [py::make_tuple(py::slice(0,3,1), 2)];

        out.append(cam_py);
    }
    return out;
}

// void VoxelModel::createFromPcd(
//     const std::map<point3D_id_t, Point3D>& pcd,
//     float spatial_lr_scale)
// {
//     // '''
//     // offline approach
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

//     // ─── 2) Use KNOWN inflated bounding directly (no file, no Python utils) ──
//     auto opts_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
//     torch::Tensor bounding_cpu = torch::empty({2,3}, opts_cpu);
//     // Row 0 = min, Row 1 = max
//     bounding_cpu[0][0] = -10.748001f;  bounding_cpu[0][1] = -28.842281f;  bounding_cpu[0][2] = -0.293899f;
//     bounding_cpu[1][0] =  56.780504f;  bounding_cpu[1][1] =  13.760179f;  bounding_cpu[1][2] = 66.683914f;

//     // Move to your device for model_init:
//     torch::Tensor bounding = bounding_cpu.to(device_type_);
//     torch::Tensor scene_min = bounding_cpu[0].to(device_type_);
//     torch::Tensor scene_max = bounding_cpu[1].to(device_type_);
//     this->scene_center = 0.5f * (scene_min + scene_max);
//     float extent_side  = (scene_max - scene_min).max().item<float>();
//     this->scene_extent       = torch::tensor({extent_side},
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
//     auto SVM     = svm_mod.attr("SparseVoxelModel");
//     py::object svm = SVM(py::arg("sh_degree")        = max_sh_degree_,
//                         py::arg("black_background") = true);

//     svm.attr("model_init")(
//         py::arg("bounding")        = bounding,
//         py::arg("outside_level")   = 0,
//         py::arg("init_n_level")    = 6,
//         py::arg("init_out_ratio")  = 2,
//         py::arg("sh_degree_init")  = 3,
//         py::arg("geo_init")        = 0.1f,
//         py::arg("sh0_init")        = 0.5f,
//         py::arg("shs_init")        = 0.0f
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

//     // ─── 6) Stats buffers (exactly as before) ───────────────────────────
//     xyz_gradient_accum_ = torch::zeros_like(subdiv_p_);
//     denom_              = torch::zeros_like(subdiv_p_);
//     voxel_error_sum_    = torch::zeros_like(subdiv_p_);
//     voxel_hit_count_    = torch::zeros_like(subdiv_p_);

//     // ─── 7) Register with the optimizer, etc. ──────────────────────────
//     VOXEL_MODEL_TENSORS_TO_VEC
// }

void VoxelModel::createFromPcd(
    const std::map<point3D_id_t, Point3D>& pcd
    // float spatial_lr_scale
    // const std::vector<std::shared_ptr<VoxelKeyframe>>& keyframes,
    // const std::string& cam_pose_txt_path
)
{
    // '''
    // dense grid
    // '''
    std::cout << "VoxelModel::createFromPcd() called with "
              << pcd.size() << " points." << std::endl;
    namespace py = pybind11;
    // this->spatial_lr_scale_ = spatial_lr_scale;

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
    this->scene_center = 0.5f * (scene_min + scene_max);
    float extent_side  = (scene_max - scene_min).max().item<float>();
    // extent_side *= 3.0f;
    this->scene_extent       = torch::tensor({extent_side},
                                 torch::TensorOptions().dtype(torch::kFloat32)
                                                        .device(device_type_));
    // std::cout << "[Debug] scene_extent = " << extent_side << std::endl;
    float float_level = -std::log2(0.05f / extent_side);
    int level = std::round(float_level);
    std::cout << "level = " << level
              << " (float_level = " << float_level << ")" << std::endl;
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
    auto SVM     = svm_mod.attr("SparseVoxelModel");
    py::object svm = SVM(
        py::arg("sh_degree")        = max_sh_degree_,
        py::arg("black_background") = true
    );
    svm.attr("model_init")(
        py::arg("bounding")        = bounding,
        py::arg("outside_level")   = 0,
        py::arg("init_n_level")    = level,
        // py::arg("init_n_level")    = 6,
        py::arg("init_out_ratio")  = 2,
        py::arg("sh_degree_init")  = 3,
        py::arg("geo_init")        = 0.1f,
        py::arg("sh0_init")        = 0.5f,
        py::arg("shs_init")        = 0.0f
        // py::arg("cameras")  = all_py_cams
    );

    // ─── 4) Pull back all core tensors from the Python object ───────────
    auto fetch = [&](const char* name){
        return svm.attr(name).cast<torch::Tensor>()
                  .contiguous()
                  .to(device_type_);
    };

    oct_path_      = fetch("octpath").to(torch::kLong);
    oct_level_     = fetch("octlevel").to(torch::kInt8);
    center_        = fetch("vox_center");
    size_          = fetch("vox_size").squeeze(1);
    vox_size_inv_  = 1.0f / size_;
    grid_pts_key_  = fetch("grid_pts_key");
    vox_key_       = fetch("vox_key");         

    // ─── 5) Copy over the learnable fields ─────────────────────────────
    _geo_grid_pts_ = fetch("_geo_grid_pts")
                      .detach().requires_grad_(true);
    sh0_ = fetch("_sh0").view({-1,1,3})
              .detach().requires_grad_(true);
    shs_ = fetch("_shs")
              .detach().requires_grad_(true);
    subdiv_p_ = fetch("_subdiv_p")
                   .detach().requires_grad_(true);
    // we still need subdiv_meta_ locally
    subdiv_meta_ = torch::zeros_like(subdiv_p_)
                      .requires_grad_(true);
    // subdiv_p_   .retain_grad();
    // subdiv_meta_.retain_grad();

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

    // ─── 6) Stats buffers (exactly as before) ───────────────────────────
    xyz_gradient_accum_ = torch::zeros_like(subdiv_p_);
    denom_              = torch::zeros_like(subdiv_p_);
    voxel_error_sum_    = torch::zeros_like(subdiv_p_);
    voxel_hit_count_    = torch::zeros_like(subdiv_p_);
    sh0_accum_ = torch::zeros({center_.size(0),3}, xyz.options());
    hit_count_ = torch::zeros({center_.size(0),1}, xyz.options());

    // ─── 7) Register with the optimizer, etc. ──────────────────────────
    VOXEL_MODEL_TENSORS_TO_VEC
}

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
//     const std::map<point3D_id_t, Point3D>& pcd,
//     float spatial_lr_scale,
//     const std::vector<std::shared_ptr<VoxelKeyframe>>& keyframes,
//     const std::string& cam_pose_txt_path)
// {
//     // '''
//     // points_init
//     // '''
//     std::cout << "VoxelModel::createFromPcd() called with "
//               << pcd.size() << " points." << std::endl;
//     namespace py = pybind11;
//     this->spatial_lr_scale_ = spatial_lr_scale;

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
//     // std::cout
//     // << "[DEBUG] cols.dims=" << cols.dim()
//     // << "   shape=("
//     // << cols.size(0) << "," << cols.size(1)
//     // << ")\n";
//     /* 2) scene bounds & extent (unchanged) */
//     auto mn = std::get<0>(pts.min(0));
//     auto mx = std::get<0>(pts.max(0));
//     float raw_extent     = (mx - mn).max().item<float>();
//     float padded_extent = raw_extent * 1.01f;   // ← 1% padding
//     this->scene_center = (mn + mx) * 0.5f;
//     this->scene_extent       = torch::tensor({padded_extent},
//                                  torch::TensorOptions().dtype(torch::kFloat32)
//                                                         .device(device_type_));
//     float expected_vox_size =  std::ldexp(padded_extent, -6);
//     // ─── 3) Call into Python’s SparseVoxelModel.model_init() ────────────
//     py::gil_scoped_acquire gil;
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
    
//     svm.attr("points_init")(
//         py::arg("scene_center")      = py::cast(this->scene_center),
//         py::arg("scene_extent")      = py::cast(this->scene_extent),
//         py::arg("xyz")               = py::cast(pts),              // already on CUDA
//         py::arg("expected_vox_size") = 0.03f,      // tune as you like
//         py::arg("density")           = 0.1f,
//         py::arg("rgb")               = py::cast(cols),              // already on CUDA
//         py::arg("shs")               = 0.0
//     );

//     // ─── 4) Pull back all core tensors from the Python object ───────────
//     auto fetch = [&](const char* name){
//         return svm.attr(name).cast<torch::Tensor>()
//                   .contiguous()
//                   .to(device_type_);
//     };

//     this->oct_path_      = fetch("octpath").to(torch::kLong);
//     this->oct_level_     = fetch("octlevel").to(torch::kInt8);
//     this->center_        = fetch("vox_center");
//     this->size_          = fetch("vox_size").squeeze(1);
//     // vox_size_inv_  = 1.0f / size_;
//     this->grid_pts_key_  = fetch("grid_pts_key");
//     this->vox_key_       = fetch("vox_key");         

//     // ─── 5) Copy over the learnable fields ─────────────────────────────
//     this->_geo_grid_pts_ = fetch("_geo_grid_pts")
//                       .detach().requires_grad_(true);
//     this->sh0_ = fetch("_sh0").view({-1,1,3}).detach().requires_grad_(true);
//     this->shs_ = fetch("_shs")
//               .detach().requires_grad_(true);
//     this->subdiv_p_ = fetch("_subdiv_p")
//                    .detach().requires_grad_(true);
//     // we still need subdiv_meta_ locally
//     this->subdiv_meta_ = torch::zeros_like(subdiv_p_)
//                       .requires_grad_(true);
//     // this->subdiv_p_   .retain_grad();
//     // this->subdiv_meta_.retain_grad();

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

//     // ─── 6) Stats buffers (exactly as before) ───────────────────────────
//     xyz_gradient_accum_ = torch::zeros_like(subdiv_p_);
//     denom_              = torch::zeros_like(subdiv_p_);
//     voxel_error_sum_    = torch::zeros_like(subdiv_p_);
//     voxel_hit_count_    = torch::zeros_like(subdiv_p_);
    
//     // ─── 7) Register with the optimizer, etc. ──────────────────────────
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

// void VoxelModel::scaledTransformationPostfix(torch::Tensor& new_center)
// {
//     // If you registered voxel centers in the optimizer, swap the tensor inside it
//     // to preserve optimizer state (moments). Otherwise just assign.
//     // Adjust kOPT_CENTER to the correct param-group index (or set to -1 to skip).
//     static constexpr int kOPT_CENTER = 0;  // <-- set this to your actual param-group for centers

//     torch::Tensor opt_center = new_center;

//     if (optimizer_ && kOPT_CENTER >= 0) {
//         opt_center = replaceTensorToOptimizer(new_center, kOPT_CENTER);
//     }

//     center_ = opt_center;

//     // If you also registered 'size_' in the optimizer, you can do the same swap here.
//     // Otherwise, the direct assignment in applyScaledTransformation is fine.
//     // Example (uncomment and set the right group index if needed):
//     //
//     // static constexpr int kOPT_SIZE = -1; // set to real index, or leave -1 to skip
//     // if (optimizer_ && kOPT_SIZE >= 0) {
//     //     size_ = replaceTensorToOptimizer(size_, kOPT_SIZE);
//     // }
//     // vox_size_inv_ = 1.0f / size_;

//     // If you keep convenience vectors for training, refresh them here (optional):
//     // Tensor_vec_center_ = {center_};
// }

// void VoxelModel::transformPoints(torch::Tensor& points, const torch::Tensor& transformmatrix)
// {
//     using torch::indexing::Slice;

//     // --- Basic checks ---
//     if (points.dim() != 2 || points.size(1) != 3) {
//         TORCH_CHECK(false, "points must have shape (N, 3), got ", points.sizes());
//     }
//     TORCH_CHECK(transformmatrix.dim() == 2 &&
//                 transformmatrix.size(0) == 4 &&
//                 transformmatrix.size(1) == 4,
//                 "transformmatrix must be 4x4, got ", transformmatrix.sizes());

//     // This implementation runs on CPU. If inputs are on CUDA, move to CPU.
//     // (If you prefer to keep device, you can remove the .cpu() lines and let ATen
//     //  run on whatever device the tensors already are.)
//     auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

//     torch::Tensor pts = points.contiguous();
//     if (pts.scalar_type() != torch::kFloat32) pts = pts.to(torch::kFloat32);
//     if (pts.device().is_cuda()) pts = pts.cpu();

//     torch::Tensor T = transformmatrix.contiguous();
//     if (T.scalar_type() != torch::kFloat32) T = T.to(torch::kFloat32);
//     if (T.device().is_cuda()) T = T.cpu();

//     const int64_t N = pts.size(0);
//     if (N == 0) {
//         // Nothing to do; just ensure points is float32 CPU contiguous
//         if (!points.is_cpu() || points.scalar_type() != torch::kFloat32 || !points.is_contiguous()) {
//             points = pts;
//         }
//         return;
//     }

//     // Build homogeneous coordinates: [x y z 1]
//     torch::Tensor ones = torch::ones({N, 1}, opts);
//     torch::Tensor pts_h = torch::cat({pts, ones}, /*dim=*/1);  // (N,4)

//     // Row-vector convention: out = pts_h @ T^T  → (N,4)
//     torch::Tensor out_h = torch::mm(pts_h, T.transpose(0, 1));

//     // Keep xyz
//     torch::Tensor out_xyz = out_h.index({Slice(), Slice(0, 3)});  // (N,3)

//     // Write back (stay on CPU float32; caller can move to CUDA if needed)
//     points = out_xyz.contiguous();
// }

// void VoxelModel::applyScaledTransformation(const float s, const Sophus::SE3f T)
// {
//     torch::NoGradGuard no_grad;

//     // 1) scale + rigid-transform voxel centers (world space)
//     center_ *= s;
//     torch::Tensor T_tensor =
//         tensor_utils::EigenMatrix2TorchTensor(T.matrix(), device_type_)
//             .transpose(0, 1); // 4x4, row-major expected by your helper
//     transformPoints(center_, T_tensor);

//     // 2) scale voxel sizes (edge lengths) and refresh inverse
//     size_ *= s;
//     vox_size_inv_ = 1.0f / size_;

//     // 3) (optional) if your geometry samples are in world space, transform them too
//     if (_geo_grid_pts_.defined() && _geo_grid_pts_.numel() > 0) {
//         _geo_grid_pts_ *= s;
//         transformPoints(_geo_grid_pts_, T_tensor);
//     }

//     // 4) push center_ update through optimizer param tensor (if registered)
//     scaledTransformationPostfix(center_);
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
        // .set_lr(training_args.geo_lr_init_ * this->spatial_lr_scale_);
        .set_lr(training_args.geo_lr_init_);

    // optimizer_->add_param_group(Tensor_vec_geo_);
    // optimizer_->param_groups()[0]
    //     .options()
    //     .set_lr(training_args.geo_lr_init_);

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
    // lr_init_    = training_args.geo_lr_init_  * this->spatial_lr_scale_;
    // lr_final_   = training_args.geo_lr_final_ * this->spatial_lr_scale_;
    lr_init_    = training_args.geo_lr_init_;
    lr_final_   = training_args.geo_lr_final_;
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
      _geo_grid_pts_ = torch::empty({0,1}, torch::kFloat32)
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
        if (_geo_grid_pts_.size(0) != G)
            _geo_grid_pts_ = torch::full({G,1}, 4.f,
                            torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .device(device_type_))
                            .requires_grad_(true);
    }
    // vox_size_inv_ = 1.0f / size_;
    // rebuildOptimizer();
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
        if (_geo_grid_pts_.size(0)!=G)
        _geo_grid_pts_ = torch::full({G,1},4.0f,
                            torch::TensorOptions()
                            .dtype(torch::kFloat32)
                            .device(device_type_))
                        .requires_grad_(true);
    }

    // vox_size_inv_ = 1.0f / size_;
    // rebuildOptimizer();
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

    bool rg_backup = _geo_grid_pts_.requires_grad();
    _geo_grid_pts_.set_requires_grad(false);

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

    _geo_grid_pts_.set_requires_grad(rg_backup);
    return {max_w, min_samp_interval, view_cnt};
}

// void VoxelModel::rebuildOptimizer()
// {
//     // 1) fetch current LRs from each group, or fall back to lr_init_
//     float geo_lr = lr_init_;
//     float sh0_lr = 0.0f;
//     float shs_lr = 0.0f;

//     if (optimizer_) {
//         // group 0: geometry
//         {
//             auto &opts = static_cast<const torch::optim::AdamOptions&>(
//                 optimizer_->param_groups()[0].options());
//             geo_lr = opts.get_lr();
//         }
//         // group 1: sh0
//         {
//             auto &opts = static_cast<const torch::optim::AdamOptions&>(
//                 optimizer_->param_groups()[1].options());
//             sh0_lr = opts.get_lr();
//         }
//         // group 2: shs
//         {
//             auto &opts = static_cast<const torch::optim::AdamOptions&>(
//                 optimizer_->param_groups()[2].options());
//             shs_lr = opts.get_lr();
//         }
//     }

//     // 2) reset the old optimizer (drops its state)
//     optimizer_.reset();

//     // 3) re-collect parameters into the three groups
//     VOXEL_MODEL_TENSORS_TO_VEC

//     // 4) create a fresh Adam with identical LRs
//     torch::optim::AdamOptions adam_opt(/*lr=*/0.0);
//     adam_opt.eps() = 1e-15;

//     optimizer_ = std::make_unique<torch::optim::Adam>(Tensor_vec_geo_, adam_opt);
//     optimizer_->param_groups()[0].options().set_lr(geo_lr);

//     optimizer_->add_param_group(Tensor_vec_sh0_);
//     optimizer_->param_groups()[1].options().set_lr(sh0_lr);

//     optimizer_->add_param_group(Tensor_vec_shs_);
//     optimizer_->param_groups()[2].options().set_lr(shs_lr);
// }

void VoxelModel::densificationPostfix(
    torch::Tensor& geo_new,
    torch::Tensor& sh0_new,
    torch::Tensor& shs_new,
    torch::Tensor& subdiv_p_new)
{
    // cat_tensors_to_optimizer
    std::vector<torch::Tensor> optimizable_tensors(3);
    std::vector<torch::Tensor> tensors_dict = {
        geo_new,
        sh0_new,
        shs_new
    };
    auto& param_groups = this->optimizer_->param_groups();
    auto& state = this->optimizer_->state();
    for (int group_idx = 0; group_idx < 3; ++group_idx) {
        auto& group = param_groups[group_idx];
        assert(group.params().size() == 1);
        auto& extension_tensor = tensors_dict[group_idx];
        auto& param = group.params()[0];
        auto key = c10::guts::to_string(param.unsafeGetTensorImpl());
        if (state.find(key) != state.end()) {
            auto& stored_state = static_cast<torch::optim::AdamParamState&>(*state[key]);
            auto new_state = std::make_unique<torch::optim::AdamParamState>();
            new_state->step(stored_state.step());
            new_state->exp_avg(torch::cat({stored_state.exp_avg().clone(), torch::zeros_like(extension_tensor)}, /*dim=*/0));
            new_state->exp_avg_sq(torch::cat({stored_state.exp_avg_sq().clone(), torch::zeros_like(extension_tensor)}, /*dim=*/0));
            // new_state->max_exp_avg_sq(stored_state.max_exp_avg_sq().clone());  // needed only when options.amsgrad(true), which is false by default

            state.erase(key);
            param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
            key = c10::guts::to_string(param.unsafeGetTensorImpl());
            state[key] = std::move(new_state);

            optimizable_tensors[group_idx] = param;
        }
        else {
            param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
            optimizable_tensors[group_idx] = param;
        }
    }

    this->_geo_grid_pts_ = optimizable_tensors[0];
    this->sh0_ = optimizable_tensors[1];
    this->shs_ = optimizable_tensors[2];

    // subdiv_p_ is not in the optimizer; just cat as before
    this->subdiv_p_ = torch::cat({this->subdiv_p_, subdiv_p_new}, 0);

    VOXEL_MODEL_TENSORS_TO_VEC   // keep your “all‑the‑params” vector in sync
}

// ── single‑shot sanity checker ────────────────────────────────────────
void VoxelModel::check_consistency(int where) const
{
    auto rows = [](const torch::Tensor& x){ return x.defined()?x.size(0):-1; };
    std::cout << "\n🔎  CONSISTENCY @" << where
              << "  rows  center:"   << rows(center_)
              << "  sh0:"            << rows(sh0_)
              << "  shs:"            << rows(shs_)
              << "  geo:"            << rows(_geo_grid_pts_)
              << "  subdiv_p:"       << rows(subdiv_p_)
              << "  vox_key:"        << rows(vox_key_)            << '\n';

    TORCH_CHECK( sh0_.size(0)==shs_.size(0) &&
                 sh0_.size(0)==subdiv_p_.size(0) &&
                 sh0_.size(0)==_geo_grid_pts_.size(0) &&
                 sh0_.size(0)==vox_key_.size(0),
                 "row‑count mismatch (centre=", rows(center_),
                 " sh0=", rows(sh0_), " shs=", rows(shs_), ")");
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

    // Only print scene_center/extent if they've been set
    if (scene_center.defined() && scene_center.numel() == 3 &&
        scene_extent.defined() && scene_extent.numel() == 1)
    {
        auto sc = scene_center.cpu();
        // std::cout << "[DBG render] scene_center = ["
        //           << sc[0].item<float>() << ", "
        //           << sc[1].item<float>() << ", "
        //           << sc[2].item<float>() << "]  "
        //           << "scene_extent = " << scene_extent.item<float>() << "\n";
    }
    // Print the shapes of the core voxel tensors
    // std::cout << "[DBG render] centers      = " << center_.sizes() << "\n"
    //           << "              vox_lengths  = " << size_.sizes()   << "\n"
    //           << "              colors (sh0) = " << sh0_.sizes()    << "\n"
    //           << "              shs          = " << shs_.sizes()    << "\n"
    //           << "              subdiv_p     = " << subdiv_p_.sizes() << "\n"
    //           << "              octpaths     = " 
    //               << oct_path_.to(torch::kLong).sizes() << "\n"
    //           << "              octlevels    = " 
    //               << oct_level_.to(torch::kInt8).sizes() << "\n"
    //           << "              subdiv_meta  = " << subdiv_meta_.sizes() << "\n"
    //           << "              geo_grid_pts = " << _geo_grid_pts_.sizes() << "\n"
    //           << "              vox_key      = " << vox_key_.sizes() << "\n";
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
    d["_geo_grid_pts"] = py::cast(_geo_grid_pts_);
    d["vox_key"]       = py::cast(vox_key_.cpu());
    // d["vox_size_inv"]  = py::cast(vox_size_inv_.cpu());

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

