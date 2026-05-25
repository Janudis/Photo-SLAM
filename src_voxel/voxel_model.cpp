#include "include_voxel/voxel_model.h"
#include "include_voxel/rerun_utils.h"
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <array>
#include <cmath>

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
    this->white_background_ = model_params.white_background_;
    this->black_background_ = false;

    // Device
    if (model_params.data_device_ == "cuda")
        device_type_ = torch::kCUDA;
    else
        device_type_ = torch::kCPU;

    // Initialize all tensors on chosen device
    VOXEL_MODEL_INIT_TENSORS(this->device_type_);
    py_ = std::make_unique<PyState>();
}

py::object VoxelModel::svm() const
{
    py::gil_scoped_acquire gil;
    if (!py_ || py_->svm.is_none()) {
        return py::none();
    }
    return py_->svm;
}

int64_t sv::VoxelModel::numGridPts() const {
    return (grid_pts_key_.defined() && grid_pts_key_.dim() > 0)
           ? grid_pts_key_.size(0)
           : 0;
}

const torch::Tensor& sv::VoxelModel::geoGridPts() const { return _geo_grid_pts_; }
const torch::Tensor& sv::VoxelModel::sh0()        const { return sh0_; }
const torch::Tensor& sv::VoxelModel::shs()        const { return shs_; }

torch::Tensor VoxelModel::voxelDensityMean() const
{
    TORCH_CHECK(_geo_grid_pts_.defined(), "_geo_grid_pts_ not defined");
    TORCH_CHECK(vox_key_.defined(), "vox_key_ not defined");

    // 1) Flatten grid scalar: [Mg,1] -> [Mg]
    auto geo_flat = _geo_grid_pts_.view({-1});  // [Mg]

    // 2) Flatten voxel keys: [Nv,8] -> [Nv*8] (long)
    auto vk_long = vox_key_.to(torch::kLong).view({-1}); // [Nv*8]

    // 3) Gather 8 corner densities per voxel:
    auto geo_corners = geo_flat.index_select(0, vk_long); // [Nv*8]

    // 4) Reshape to [Nv, 8] and average:
    const auto Nv = vox_key_.size(0);
    auto geo_per_voxel = geo_corners.view({Nv, 8}).mean(1); // [Nv]

    return geo_per_voxel;  // pre-activation densities per voxel
}

torch::Tensor VoxelModel::makeGeoGridInitRows_(
    const torch::Tensor& grid_pts_key_new,
    int64_t begin,
    int64_t end,
    float default_value)
{
    TORCH_CHECK(grid_pts_key_new.defined() &&
                    grid_pts_key_new.dim() == 2 &&
                    grid_pts_key_new.size(1) == 3,
                "makeGeoGridInitRows_: grid_pts_key_new must be [M,3]");
    TORCH_CHECK(begin >= 0 && end >= begin && end <= grid_pts_key_new.size(0),
                "makeGeoGridInitRows_: invalid row range");

    const int64_t rows = end - begin;
    auto make_default = [&]() {
        return torch::full(
                   {rows, 1},
                   default_value,
                   torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(grid_pts_key_new.device()))
            .contiguous()
            .detach()
            .requires_grad_();
    };

    if (rows == 0 || !geo_grid_init_callback_) {
        return make_default();
    }

    try {
        torch::Tensor key_add =
            grid_pts_key_new.slice(/*dim=*/0, begin, end).contiguous();
        auto dev = key_add.device();
        torch::Tensor scene_center =
            scene_center_.to(dev).to(torch::kFloat32).contiguous().view({3});
        torch::Tensor scene_extent =
            scene_extent_.to(dev).to(torch::kFloat32).contiguous().view({1});
        torch::Tensor scene_min = scene_center - 0.5f * scene_extent;

        const float finest_scale = std::ldexp(1.0f, -max_num_levels_);
        torch::Tensor finest_vox = scene_extent * finest_scale;
        torch::Tensor grid_xyz =
            scene_min.view({1, 3}) +
            key_add.to(torch::kFloat32) * finest_vox.view({1, 1});
        grid_xyz = grid_xyz.contiguous();

        torch::Tensor init = geo_grid_init_callback_(grid_xyz, fixed_vox_size_);
        if (!init.defined() || init.numel() != rows) {
            return make_default();
        }
        if (init.dim() == 1) {
            init = init.view({rows, 1});
        } else {
            init = init.reshape({rows, 1});
        }
        return init.to(grid_pts_key_new.device())
            .to(torch::kFloat32)
            .contiguous()
            .detach()
            .requires_grad_();
    } catch (const std::exception& e) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::cout << "[TSDF DENSITY INIT] callback failed; using default geo init: "
                      << e.what() << "\n";
        }
        return make_default();
    }
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

torch::Tensor VoxelModel::octPath() const {
    return oct_path_;
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

void VoxelModel::logParamSignature(const char* tag)
{
    auto dump = [&](const char* name, const torch::Tensor& t) {
        if (!t.defined()) {
            std::cerr << "[" << tag << "] " << name << ": <undefined>\n";
            return;
        }
        auto flat = t.detach().flatten().to(torch::kCPU);
        const int64_t n = flat.numel();
        float mean = n ? flat.mean().item<float>() : 0.f;
        float minv = n ? flat.min().item<float>()  : 0.f;
        float maxv = n ? flat.max().item<float>()  : 0.f;
        float f0   = n ? flat.index({0}).item<float>() : 0.f;
        float f1   = n ? flat.index({std::min<int64_t>(1,n-1)}).item<float>() : 0.f;
        float fN   = n ? flat.index({n-1}).item<float>() : 0.f;
        std::cerr << "[" << tag << "] " << name
                  << " n=" << n
                  << " mean=" << mean
                  << " min=" << minv
                  << " max=" << maxv
                  << " samples=[" << f0 << "," << f1 << ",..," << fN << "]"
                  << " ptr=" << (const void*)t.data_ptr() << "\n";
    };

    // Topology too (so we see if the grid changed):
    std::cerr << "[" << tag << "] voxels=" << center_.size(0)
              << " gridPts=" << _geo_grid_pts_.size(0) << "\n";

    dump("_geo_grid_pts_", _geo_grid_pts_);
    dump("sh0_",           sh0_);
    dump("shs_",           shs_);
    dump("subdiv_p_",      subdiv_p_);
}

namespace fs = std::filesystem;
inline void logAABB_minmax(const torch::Tensor& min_xyz,
                           const torch::Tensor& max_xyz,
                           std::string_view tag,
                           int iter,
                           const fs::path& outdir)
{
    auto mn = min_xyz.detach().to(torch::kCPU).contiguous();
    auto mx = max_xyz.detach().to(torch::kCPU).contiguous();

    const float minx = mn[0].item<float>();
    const float miny = mn[1].item<float>();
    const float minz = mn[2].item<float>();
    const float maxx = mx[0].item<float>();
    const float maxy = mx[1].item<float>();
    const float maxz = mx[2].item<float>();

    fs::create_directories(outdir);
    const fs::path csv = outdir / "atlas_boxes.csv";
    const bool newfile = !fs::exists(csv);

    std::ofstream f(csv, std::ios::app);
    if (newfile) f << "iter,tag,minx,miny,minz,maxx,maxy,maxz\n";
    f << iter << ',' << tag << ','
      << minx << ',' << miny << ',' << minz << ','
      << maxx << ',' << maxy << ',' << maxz << '\n';
}

inline void logAABB_center_extent(const torch::Tensor& scene_center,
                                  const torch::Tensor& scene_extent, // [1]
                                  std::string_view tag,
                                  int iter,
                                  const fs::path& outdir)
{
    auto sc = scene_center.detach().to(torch::kCPU).contiguous();
    auto se = scene_extent.detach().to(torch::kCPU).contiguous();
    const float cx = sc[0].item<float>();
    const float cy = sc[1].item<float>();
    const float cz = sc[2].item<float>();
    const float ex = 0.5f * se[0].item<float>();

    torch::Tensor min_xyz = torch::tensor({cx - ex, cy - ex, cz - ex});
    torch::Tensor max_xyz = torch::tensor({cx + ex, cy + ex, cz + ex});
    logAABB_minmax(min_xyz, max_xyz, tag, iter, outdir);
}

// convenience for a bounding tensor shaped [2,3] (min row, max row)
inline void logAABB_bounding2x3(const torch::Tensor& bounding_2x3,
                                std::string_view tag,
                                int iter,
                                const fs::path& outdir)
{
    logAABB_minmax(bounding_2x3[0], bounding_2x3[1], tag, iter, outdir);
}

static inline void print_aabb(const char* name,
                              const float mn[3], const float mx[3]) {
    const float sx = mx[0] - mn[0];
    const float sy = mx[1] - mn[1];
    const float sz = mx[2] - mn[2];
    const float vol = std::max(0.f, sx) * std::max(0.f, sy) * std::max(0.f, sz);
    std::cout << std::fixed << std::setprecision(6)
              << "[AABB " << name << "]  "
              << "min=(" << mn[0] << "," << mn[1] << "," << mn[2] << ")  "
              << "max=(" << mx[0] << "," << mx[1] << "," << mx[2] << ")  "
              << "size=(" << sx << "," << sy << "," << sz << ")  "
              << "vol=" << vol << "\n";
}

// void VoxelModel::createFromPcd(const std::map<point3D_id_t, Point3D>& pcd, const std::vector<sv::MiniCam>& cams)
// {
//     namespace py = pybind11;
//     std::cout << "VoxelModel::createFromPcd() called with " << pcd.size() << " points.\n";

//     TORCH_CHECK(global_scene_extent_ > 0.f, "global_scene_extent_ must be set (>0).");
//     TORCH_CHECK(fixed_vox_size_   > 0.f, "fixed_vox_size_ must be set (>0).");
//     TORCH_CHECK(max_sh_degree_ >= 0, "max_sh_degree_ must be >= 0.");

//     const int N = static_cast<int>(pcd.size());
//     if (N == 0) { std::cerr << "[createFromPcd] Empty PCD — nothing to do.\n"; return; }

//     auto dev = torch::kCUDA;

//     // --- 0) Pack PCD to CPU tensors (xyz in world, rgb in [0..1]) -------------------
//     torch::Tensor xyz_cpu = torch::empty({N,3}, torch::dtype(torch::kFloat32).device(torch::kCPU));
//     torch::Tensor rgb_cpu = torch::empty({N,3}, torch::dtype(torch::kFloat32).device(torch::kCPU));
//     {
//         int i = 0;
//         for (const auto& kv : pcd) {
//             const auto& P = kv.second;
//             xyz_cpu[i][0] = P.xyz_(0);
//             xyz_cpu[i][1] = P.xyz_(1);
//             xyz_cpu[i][2] = P.xyz_(2);
//             // P.color_ already scaled? If it’s 0..255, divide; if already 0..1, keep.
//             rgb_cpu[i][0] = P.color_(0);  // adapt if needed
//             rgb_cpu[i][1] = P.color_(1);
//             rgb_cpu[i][2] = P.color_(2);
//             ++i;
//         }
//     }

//     // --- 1) Estimate a compact main-scene bbox from the PCD --------------------------
//     py::gil_scoped_acquire gil;
//     // Keep this order identical to the original working version:
//     static py::module svr_utils = py::module::import("svraster_cuda").attr("utils");
//     // Insert path *once* via the svm_mod lambda, then any src.* imports are safe.
//     static py::module svm_mod = []{
//         py::module sys = py::module::import("sys");
//         sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
//         return py::module::import("src.sparse_voxel_model");
//     }();
//     // Now it's safe to import from src.utils.*
//     static py::module act_mod     = py::module::import("src.utils.activation_utils");
//     static py::module oct_utils   = py::module::import("src.utils.octree_utils");
//     static py::module bound_utils = py::module::import("src.utils.bounding_utils"); // only if you use it
//     static py::module types       = py::module::import("types");
//     static py::module torch_mod   = py::module::import("torch");

//     py::object xyz_cpu_py = py::cast(xyz_cpu.contiguous());
//     py::object ns         = types.attr("SimpleNamespace")("points"_a = xyz_cpu_py.attr("numpy")());
//     // tweak 0.1 → your density fraction / percentile as you like
//     py::tuple cr = bound_utils.attr("main_scene_bound_pcd_heuristic")(ns, py::float_(0.1f));
//     // back to tensors, CUDA
//     py::object center_t_py = torch_mod.attr("from_numpy")(cr[0]);
//     auto center_t = center_t_py.cast<torch::Tensor>().to(dev).contiguous();        // [3]
//     float radius  = cr[1].cast<float>();
//     auto radius_t = torch::full({3}, radius, torch::dtype(torch::kFloat32).device(dev));

//     scene_center_ = torch::tensor(
//         {global_scene_center_[0], global_scene_center_[1], global_scene_center_[2]},
//         torch::dtype(torch::kFloat32).device(dev)).contiguous();                                             
//     int   outside_level = 0;
//     const float scene_extent_scalar = global_scene_extent_ * std::pow(2.0f, outside_level);
//     scene_extent_ = torch::tensor({scene_extent_scalar},
//         torch::dtype(torch::kFloat32).device(dev)).contiguous();
//     inside_extent_ = torch::tensor({global_scene_extent_},
//         torch::dtype(torch::kFloat32).device(dev)
//     ).contiguous();    
//     scene_min_t_  = (scene_center_ - 0.5f * scene_extent_).contiguous(); // [3]

//     // --- 2) Fix base level from fixed_vox_size_ and cache effective voxel size ------
//     const int MAX_L = max_num_levels_;
//     auto vox_size_t = torch::full({1}, fixed_vox_size_, torch::dtype(torch::kFloat32).device(dev));
//     py::object L_fp_py = oct_utils.attr("vox_size_2_level")(py::cast(scene_extent_), py::cast(vox_size_t));
//     auto L_fp      = L_fp_py.cast<torch::Tensor>().round();
//     auto L_clamped = L_fp.clamp_min(1).clamp_max(MAX_L).to(torch::kInt8).contiguous();  // [1]
//     octlevel_      = L_clamped.item<int8_t>();

//     py::object vox_eff_py = oct_utils.attr("level_2_vox_size")(
//         py::cast(scene_extent_), py::cast(L_clamped.view({1,1})));
//     vox_eff_ = vox_eff_py.cast<torch::Tensor>().view({1,1}).contiguous();               // [1,1]

//     // --- 3) Convert bbox → dense ijk range at base level -----------------------------
//     auto bb_min = (scene_center_ - radius_t).contiguous();
//     auto bb_max = (scene_center_ + radius_t).contiguous();

//     auto vox_t = vox_eff_.mean().view({1}).repeat({3}).contiguous(); // [3] CUDA float
//     const int64_t grid_limit = (1LL << static_cast<int>(octlevel_));
//     auto ijk_min = ((bb_min - scene_min_t_) / vox_t).floor().to(torch::kLong);
//     auto ijk_max = ((bb_max - scene_min_t_) / vox_t).ceil().to(torch::kLong) - 1;

//     // Clamp
//     auto zero = torch::zeros_like(ijk_min);
//     auto lim  = torch::full_like(ijk_max, grid_limit - 1);
//     ijk_min = torch::maximum(ijk_min, zero);
//     ijk_max = torch::minimum(ijk_max, lim);

//     if (!(ijk_min <= ijk_max).all().item<bool>()) {
//         std::cerr << "[createFromPcd] Degenerate bbox → no cells.\n";
//         return;
//     }

//     // Enumerate ijk box
//     auto ir = torch::arange(ijk_min[0].item<int64_t>(), ijk_max[0].item<int64_t>() + 1,
//                             torch::dtype(torch::kLong).device(dev));
//     auto jr = torch::arange(ijk_min[1].item<int64_t>(), ijk_max[1].item<int64_t>() + 1,
//                             torch::dtype(torch::kLong).device(dev));
//     auto kr = torch::arange(ijk_min[2].item<int64_t>(), ijk_max[2].item<int64_t>() + 1,
//                             torch::dtype(torch::kLong).device(dev));
//     auto grids  = torch::meshgrid({ir, jr, kr}, /*indexing=*/"ij");
//     auto ijk_box= torch::stack(
//         { grids[0].contiguous().view({-1}),
//           grids[1].contiguous().view({-1}),
//           grids[2].contiguous().view({-1}) }, 1).contiguous(); // [Nc,3]
//     int64_t Nc = ijk_box.size(0);
//     auto L_box = torch::full({Nc,1}, octlevel_, torch::dtype(torch::kInt8).device(dev)).contiguous();

//     // Morton/octpath for those cells
//     py::object octpath_py = svr_utils.attr("ijk_2_octpath")(py::cast(ijk_box), py::cast(L_box));
//     auto octpath = octpath_py.cast<torch::Tensor>().contiguous(); // [Nc,1] int64
    
//     if (!cams.empty())
//     {
//         py::gil_scoped_acquire gil;
//         static py::module oct_utils = py::module::import("src.utils.octree_utils");
//         static py::module svr_mod   = py::module::import("svraster_cuda").attr("renderer");

//         // Decode voxel centers/sizes for filtering
//         py::tuple dec = oct_utils.attr("octpath_decoding")(
//             py::cast(octpath.contiguous()),
//             py::cast(L_box.contiguous()),
//             py::cast(scene_center_.contiguous()),
//             py::cast(scene_extent_.contiguous())
//         );
//         at::Tensor vox_center = dec[0].cast<at::Tensor>(); // [Nu,3]
//         at::Tensor vox_size   = dec[1].cast<at::Tensor>(); // [Nu,1]
//         std::cout << "octpath dev="   << (octpath.is_cuda() ? "cuda" : "cpu") 
//             << " dtype="        << octpath.dtype() << "\n";
//         std::cout << "vox_center dev="<< (vox_center.is_cuda() ? "cuda" : "cpu")
//                 << " dtype="        << vox_center.dtype() << "\n";
//         std::cout << "vox_size dev="  << (vox_size.is_cuda() ? "cuda" : "cpu")
//                 << " dtype="        << vox_size.dtype() << "\n";

//         // Build Python list of MiniCams (ensure CUDA tensors)
//         py::list py_cams;
//         py::module torch_mod = py::module::import("torch");
//         py::object py_cuda = torch_mod.attr("device")("cuda");
//         auto move_attr_to_cuda_if_tensor = [&](py::object& obj, const char* name){
//             if (py::hasattr(obj, name)) {
//                 py::object t = obj.attr(name);
//                 // Only tensors have .is_cuda/.to; defensive check:
//                 if (py::hasattr(t, "is_cuda") && !py::bool_(t.attr("is_cuda"))) {
//                     obj.attr(name) = t.attr("to")(py_cuda);  // or t.attr("to")("cuda")
//                 }
//             }
//         };
//         for (const auto& c : cams) {
//             py::object py_cam = MiniCam_to_py(c);
//             // matrices
//             move_attr_to_cuda_if_tensor(py_cam, "w2c");
//             move_attr_to_cuda_if_tensor(py_cam, "c2w");
//             // vectors used by mark_near / other helpers
//             move_attr_to_cuda_if_tensor(py_cam, "position");
//             move_attr_to_cuda_if_tensor(py_cam, "lookat");
//             py_cams.append(py_cam);
//         }

//         auto Nu_before = octpath.size(0);
//         // mark_max_samp_rate -> keep rate>0
//         at::Tensor rate = svr_mod.attr("mark_max_samp_rate")(
//             py_cams,
//             py::cast(octpath),
//             py::cast(vox_center),
//             py::cast(vox_size)
//         ).cast<at::Tensor>();

//         at::Tensor kept = rate > 0;
//         int64_t n_rate_pos = kept.sum().item<int64_t>();
//         // optional near filtering
//         const float near_thresh = 0.2f;
//         int64_t n_near_hit = 0;
//         if (near_thresh > 0.0f) {
//             at::Tensor is_near = svr_mod.attr("mark_near")(
//                 py_cams,
//                 py::cast(octpath),
//                 py::cast(vox_center),
//                 py::cast(vox_size),
//                 py::float_(near_thresh)
//             ).cast<at::Tensor>();
//             kept = kept & (~is_near);
//             n_near_hit = is_near.sum().item<int64_t>();
//         }

//         auto idx = torch::nonzero(kept).view({-1});
//         int64_t K = idx.size(0);
//         // Apply mask (all tensors together, no reshapes)
//         if (K > 0 && K < octpath.size(0)) {
//             octpath = octpath.index_select(0, idx).contiguous();   // [K,1]
//             L_box   = L_box.index_select(0, idx).contiguous();     // [K,1] already
//         }
//         // Recompute sizes and assert consistency
//         Nc = octpath.size(0);
//         // Debug prints
//         std::cout << "[filter] Nu_before=" << Nu_before
//                 << " rate>0=" << n_rate_pos
//                 << " near_hit=" << n_near_hit
//                 << " kept_final=" << Nc << std::endl;
//     }

//     // --- 4) Create or reuse SVM; set scene & topology -------------------------------
//     if (!py_->svm || py_->svm.is_none()) {
//         py::object SVM = svm_mod.attr("SparseVoxelModel");
//         py_->svm = SVM(py::arg("sh_degree") = max_sh_degree_);
//     }
//     py_->svm.attr("scene_center")  = scene_center_.contiguous();
//     py_->svm.attr("scene_extent")  = scene_extent_.contiguous();
//     py_->svm.attr("inside_extent") = inside_extent_.contiguous();
//     py_->svm.attr("octpath")       = octpath;
//     py_->svm.attr("octlevel")      = L_box;

//     // --- 5) Initialize learnables ----------------------------------------------------
//     // Subdivision priority
//     auto subdiv_p = torch::ones({Nc,1}, torch::dtype(torch::kFloat32).device(dev)).requires_grad_();
//     py_->svm.attr("_subdiv_p") = subdiv_p.detach().requires_grad_();

//     // SH-0: use mean RGB of the input PCD as a simple prior (or a constant)
//     auto rgb_mean = rgb_cpu.to(dev).mean(0, /*keepdim=*/false).contiguous(); // [3]
//     py::object sh0_dc_py = act_mod.attr("rgb2shzero")(py::cast(rgb_mean.view({1,3})));
//     auto sh0_dc = sh0_dc_py.cast<torch::Tensor>().expand({Nc,3}).contiguous();
//     py_->svm.attr("_sh0") = sh0_dc.detach().requires_grad_();

//     // Higher-degree SH zeros
//     const int n_sh_rest = (max_sh_degree_ + 1)*(max_sh_degree_ + 1) - 1;
//     auto shs = torch::zeros({Nc, n_sh_rest, 3}, torch::dtype(torch::kFloat32).device(dev));
//     py_->svm.attr("_shs") = shs.detach().requires_grad_();

//     // Geometry: grid density (-10) over all grid points
//     torch::Tensor grid_pts_key = py_->svm.attr("grid_pts_key").cast<torch::Tensor>(); // [Mg,3] int64
//     auto geo_grid = torch::full({grid_pts_key.size(0), 1}, -10.0f,
//                                 torch::dtype(torch::kFloat32).device(dev));
//     py_->svm.attr("_geo_grid_pts") = geo_grid.detach().requires_grad_();

//     // Activate initial SH degree (e.g., min(3, max_sh_degree_))
//     py_->svm.attr("active_sh_degree") = py::int_(std::min(max_sh_degree_, 3));

//     // --- 6) Sync back to C++ members & optimizer groups -----------------------------
//     auto fetch = [&](const char* name){ return py_->svm.attr(name).cast<torch::Tensor>().contiguous(); };
//     this->oct_path_      = fetch("octpath");
//     this->oct_level_     = fetch("octlevel");
//     this->center_        = fetch("vox_center");
//     this->size_          = fetch("vox_size").squeeze(1);
//     this->vox_size_inv_  = 1.0f / size_;
//     this->grid_pts_key_  = fetch("grid_pts_key");
//     this->vox_key_       = fetch("vox_key");

//     this->_geo_grid_pts_ = fetch("_geo_grid_pts").requires_grad_(true);
//     this->sh0_           = fetch("_sh0").requires_grad_(true);
//     this->shs_           = fetch("_shs").requires_grad_(true);
//     this->subdiv_p_      = fetch("_subdiv_p").requires_grad_(true);

//     // stats buffer
//     this->max_w_ = torch::zeros({center_.size(0), 1}, torch::dtype(torch::kFloat32).device(dev));

//     // Register with optimizer
//     VOXEL_MODEL_TENSORS_TO_VEC

//     std::cout << "[createFromPcd] Seeded " << Nc << " voxels from bbox @ level "
//               << static_cast<int>(octlevel_) << " (vox_size="
//               << vox_eff_.item<float>() << " m)\n";
// }

// void VoxelModel::increasePcd(std::vector<float> pcd_full,
//                              std::vector<float> colors,
//                              const int /*iteration*/, const std::vector<sv::MiniCam>& cams)
// {
//     namespace py = pybind11;

//     const int Nf = static_cast<int>(pcd_full.size());
//     if (Nf < 3 || colors.size() < 3) return;
//     const int N = Nf / 3;

//     std::cout << "VoxelModel::increasePcd() called with " << N << " points.\n";
//     TORCH_CHECK(global_scene_extent_ > 0.f && fixed_vox_size_ > 0.f,
//                 "increasePcd: scene extent / fixed vox size not set.");
//     TORCH_CHECK(py_->svm && !py_->svm.is_none(),
//                 "increasePcd: SVM not initialized; call createFromPcd first.");

//     // --- 0) Pack batch to CPU tensors (xyz in world, rgb in [0..1 or your scale]) ---
//     torch::Tensor xyz_cpu = torch::from_blob(
//         pcd_full.data(), { (int64_t)N, 3 },
//         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
//     ).clone();

//     torch::Tensor rgb_cpu = torch::from_blob(
//         colors.data(), { (int64_t)N, 3 },
//         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
//     ).clone(); // .div_(255.0f);  // uncomment if input colors are 0..255

//     auto dev = torch::kCUDA;

//     // --- 1) Get a compact bbox via the Python heuristic (median + density) -----------
//     py::gil_scoped_acquire gil;
//     static py::module bound_utils = py::module::import("src.utils.bounding_utils");
//     static py::module types       = py::module::import("types");
//     static py::module torch_mod   = py::module::import("torch");
//     static py::module svr_utils   = py::module::import("svraster_cuda").attr("utils");
//     static py::module act_mod     = py::module::import("src.utils.activation_utils");

//     py::object xyz_cpu_py = py::cast(xyz_cpu.contiguous());
//     py::object ns         = types.attr("SimpleNamespace")("points"_a = xyz_cpu_py.attr("numpy")());

//     // tune 0.1 to your needs; keep consistent with createFromPcd()
//     py::tuple cr = bound_utils.attr("main_scene_bound_pcd_heuristic")(ns, py::float_(0.1f));

//     py::object center_t_py = torch_mod.attr("from_numpy")(cr[0]);
//     auto center_t = center_t_py.cast<torch::Tensor>().to(dev).contiguous();  // [3]
//     float radius  = cr[1].cast<float>();
//     auto radius_t = torch::full({3}, radius, torch::dtype(torch::kFloat32).device(dev));

//     // --- 2) Convert bbox → dense ijk block at cached base level ----------------------
//     // Assumes createFromPcd() has set scene_min_t_ and vox_eff_ (base level size)
//     auto bb_min = (center_t - radius_t).contiguous();
//     auto bb_max = (center_t + radius_t).contiguous();

//     auto vox_t = vox_eff_.mean().view({1}).repeat({3}).contiguous(); // [3] CUDA float
//     const int64_t grid_limit = (1LL << static_cast<int>(octlevel_));

//     auto ijk_min = ((bb_min - scene_min_t_) / vox_t).floor().to(torch::kLong);
//     auto ijk_max = ((bb_max - scene_min_t_) / vox_t).ceil().to(torch::kLong) - 1;

//     // Clamp to valid grid range
//     auto zero = torch::zeros_like(ijk_min);
//     auto lim  = torch::full_like(ijk_max, grid_limit - 1);
//     ijk_min = torch::maximum(ijk_min, zero);
//     ijk_max = torch::minimum(ijk_max, lim);

//     if (!(ijk_min <= ijk_max).all().item<bool>()) {
//         std::cout << "[increasePcd] Heuristic bbox produced no in-bounds cells; skipping.\n";
//         return;
//     }

//     // Enumerate the ijk block
//     auto ir = torch::arange(ijk_min[0].item<int64_t>(), ijk_max[0].item<int64_t>() + 1,
//                             torch::dtype(torch::kLong).device(dev));
//     auto jr = torch::arange(ijk_min[1].item<int64_t>(), ijk_max[1].item<int64_t>() + 1,
//                             torch::dtype(torch::kLong).device(dev));
//     auto kr = torch::arange(ijk_min[2].item<int64_t>(), ijk_max[2].item<int64_t>() + 1,
//                             torch::dtype(torch::kLong).device(dev));
//     auto grids   = torch::meshgrid({ir, jr, kr}, /*indexing=*/"ij");
//     auto ijk_box = torch::stack(
//         { grids[0].contiguous().view({-1}),
//           grids[1].contiguous().view({-1}),
//           grids[2].contiguous().view({-1}) }, 1).contiguous(); // [Nc,3]
//     int64_t Nc = ijk_box.size(0);

//     auto L_box = torch::full({Nc,1}, octlevel_, torch::dtype(torch::kInt8).device(dev)).contiguous();

//     // Build morton/octpath for candidates
//     py::object octpath_box_py = svr_utils.attr("ijk_2_octpath")(py::cast(ijk_box), py::cast(L_box));
//     auto octpath_box = octpath_box_py.cast<torch::Tensor>().contiguous(); // [Nc,1] int64

//     // ---- (NEW) SVR-style filtering with cameras ----
//     if (!cams.empty()) {
//         py::gil_scoped_acquire gil;
//         static py::module oct_utils = py::module::import("src.utils.octree_utils");
//         static py::module svr_mod   = py::module::import("svraster_cuda").attr("renderer");
//         static py::module torch_mod = py::module::import("torch");

//         // Decode voxel centers/sizes
//         py::tuple dec = oct_utils.attr("octpath_decoding")(
//             py::cast(octpath_box),
//             py::cast(L_box),
//             py::cast(scene_center_.contiguous()),
//             py::cast(scene_extent_.contiguous())
//         );
//         at::Tensor vox_center = dec[0].cast<at::Tensor>(); // [Nu,3] cuda
//         at::Tensor vox_size   = dec[1].cast<at::Tensor>(); // [Nu,1] cuda

//         // Build Python list of CUDA MiniCams
//         py::list py_cams;
//         py::object py_cuda = torch_mod.attr("device")("cuda");
//         auto move_attr_to_cuda_if_tensor = [&](py::object& obj, const char* name){
//             if (py::hasattr(obj, name)) {
//                 py::object t = obj.attr(name);
//                 if (py::hasattr(t, "is_cuda") && !py::bool_(t.attr("is_cuda"))) {
//                     obj.attr(name) = t.attr("to")(py_cuda);
//                 }
//             }
//         };
//         for (const auto& c : cams) {
//             py::object py_cam = MiniCam_to_py(c);
//             move_attr_to_cuda_if_tensor(py_cam, "w2c");
//             move_attr_to_cuda_if_tensor(py_cam, "c2w");
//             move_attr_to_cuda_if_tensor(py_cam, "position");
//             move_attr_to_cuda_if_tensor(py_cam, "lookat");
//             py_cams.append(py_cam);
//         }

//         auto Nu_before = octpath_box.size(0);

//         // rate > 0
//         at::Tensor rate = svr_mod.attr("mark_max_samp_rate")(
//             py_cams, py::cast(octpath_box), py::cast(vox_center), py::cast(vox_size)
//         ).cast<at::Tensor>();
//         at::Tensor kept = rate > 0;
//         int64_t n_rate_pos = kept.sum().item<int64_t>();

//         // near filtering (same threshold you used in createFromPcd)
//         const float near_thresh = 0.2f;
//         int64_t n_near_hit = 0;
//         if (near_thresh > 0.0f) {
//             at::Tensor is_near = svr_mod.attr("mark_near")(
//                 py_cams, py::cast(octpath_box), py::cast(vox_center), py::cast(vox_size),
//                 py::float_(near_thresh)
//             ).cast<at::Tensor>();
//             kept = kept & (~is_near);
//             n_near_hit = is_near.sum().item<int64_t>();
//         }

//         auto idx = torch::nonzero(kept).view({-1});
//         int64_t K = idx.size(0);

//         if (K == 0) {
//             std::cout << "[increasePcd/filter] all candidates filtered out, nothing to add.\n";
//             return;
//         }

//         if (K < octpath_box.size(0)) {
//             // Apply mask to ALL aligned tensors
//             octpath_box = octpath_box.index_select(0, idx).contiguous(); // [K,1]
//             L_box         = L_box.index_select(0, idx).contiguous();         // [K,1]
//         }

//         Nc = octpath_box.size(0); // update Nu after filtering

//         // Sanity
//         TORCH_CHECK(L_box.sizes() == torch::IntArrayRef({Nc,1}), "L_box shape mismatch after filtering");

//         std::cout << "[increasePcd/filter] Nu_before=" << Nu_before
//                 << " rate>0=" << n_rate_pos
//                 << " near_hit=" << n_near_hit
//                 << " kept_final=" << Nc << std::endl;
//     }
//     // ---- end filtering ----

//     // --- 3) Dedup against existing topology + avoid base-level parents if children exist
//     auto octpath_cur  = py_->svm.attr("octpath").cast<torch::Tensor>().contiguous(); // [No,1]
//     auto octlevel_cur = py_->svm.attr("octlevel").cast<torch::Tensor>().contiguous(); // [No,1]

//     auto key_box = octpath_box.view({-1}).to(torch::kInt64).mul(256)
//                  + L_box.view({-1}).to(torch::kInt64);

//     torch::Tensor new_mask;
//     if (octpath_cur.numel() == 0) {
//         new_mask = torch::ones({Nc}, torch::dtype(torch::kBool).device(dev));
//     } else {
//         static py::module torch_mod_local = py::module::import("torch");
//         auto key_cur = octpath_cur.view({-1}).to(torch::kInt64).mul(256)
//                      + octlevel_cur.view({-1}).to(torch::kInt64);
//         py::object isin_py = torch_mod_local.attr("isin")(py::cast(key_box), py::cast(key_cur));
//         auto is_dup = isin_py.cast<torch::Tensor>().to(torch::kBool); // [Nc]
//         new_mask = ~is_dup;

//         // Drop base-level parents of already-subdivided regions (L_old > base_L)
//         const int MAX_L  = max_num_levels_;
//         const int base_L = static_cast<int>(octlevel_);
//         if (octpath_cur.numel() > 0) {
//             auto Lold_i64     = octlevel_cur.view({-1}).to(torch::kInt64);
//             auto has_children = (Lold_i64 > base_L);
//             if (has_children.any().item<bool>()) {
//                 // Clear octant bits below base_L and build parent keys at base_L
//                 const int levels_below  = std::max(0, MAX_L - base_L);
//                 const int bits_to_clear = 3 * levels_below;
//                 long long lower_mask    = (bits_to_clear > 0) ? ((1LL << bits_to_clear) - 1LL) : 0LL;
//                 long long keep_mask_ll  = ~lower_mask;
//                 auto keep_mask = torch::full({1}, static_cast<int64_t>(keep_mask_ll),
//                     torch::TensorOptions().dtype(torch::kInt64).device(dev));

//                 auto op_old_i64  = octpath_cur.view({-1}).to(torch::kInt64);
//                 auto op_anc_base = (op_old_i64 & keep_mask);

//                 auto sel_child = torch::nonzero(has_children).view({-1});
//                 op_anc_base    = op_anc_base.index_select(0, sel_child);

//                 auto key_children_as_parent = op_anc_base.mul(256)
//                                            + torch::full_like(op_anc_base, static_cast<int64_t>(base_L));

//                 // unique + sorted helper
//                 auto unique_sorted_1d = [](const at::Tensor& t)->at::Tensor {
//                     TORCH_CHECK(t.dim() == 1, "unique_sorted_1d expects a 1-D tensor");
//                     if (t.numel() <= 1) return t.contiguous();
//                     auto sort_res = t.sort(0);
//                     auto sorted   = std::get<0>(sort_res);
//                     using torch::indexing::Slice;
//                     auto keep = torch::empty_like(sorted, torch::kBool);
//                     keep.index_put_({0}, true);
//                     auto neq = sorted.index({Slice(1, torch::indexing::None)})
//                              != sorted.index({Slice(torch::indexing::None, -1)});
//                     keep.index_put_({Slice(1, torch::indexing::None)}, neq);
//                     auto idx = torch::nonzero(keep).view({-1});
//                     return sorted.index_select(0, idx).contiguous();
//                 };
//                 key_children_as_parent = unique_sorted_1d(key_children_as_parent);

//                 py::object isin_py2 = torch_mod_local.attr("isin")(
//                     py::cast(key_box), py::cast(key_children_as_parent));
//                 auto would_collide_parent = isin_py2.cast<torch::Tensor>().to(torch::kBool); // [Nc]
//                 new_mask = new_mask & (~would_collide_parent);
//             }
//         }
//     }

//     if (!new_mask.any().item<bool>()) {
//         std::cout << "[increasePcd] No new voxels from bbox.\n";
//         return;
//     }

//     // Select additions
//     auto sel        = torch::nonzero(new_mask).view({-1}); // [Nk]
//     auto octpath_add= octpath_box.index_select(0, sel);    // [Nk,1]
//     auto L_add      = L_box.index_select(0, sel);          // [Nk,1]
//     const int Nk    = sel.size(0);

//     // --- 4) Append topology ----------------------------------------------------------
//     py_->svm.attr("octpath")  = torch::cat({octpath_cur,  octpath_add}, 0).contiguous();
//     py_->svm.attr("octlevel") = torch::cat({octlevel_cur, L_add},       0).contiguous();

//     // --- 5) Append learnables for new rows -------------------------------------------
//     // _subdiv_p
//     auto subdiv_old = py_->svm.attr("_subdiv_p").cast<torch::Tensor>();
//     auto subdiv_add = torch::ones({Nk,1}, torch::dtype(torch::kFloat32).device(dev));
//     py_->svm.attr("_subdiv_p") = torch::cat({subdiv_old, subdiv_add}, 0)
//                                  .contiguous().detach().requires_grad_();

//     // _sh0: simple prior from this batch's mean color (or constant)
//     auto rgb_mean = rgb_cpu.to(dev).mean(0, /*keepdim=*/false).contiguous(); // [3]
//     py::object sh0_add_py = act_mod.attr("rgb2shzero")(py::cast(rgb_mean.view({1,3})));
//     auto sh0_add_bcast = sh0_add_py.cast<torch::Tensor>().expand({Nk,3}).contiguous();
//     auto sh0_old = py_->svm.attr("_sh0").cast<torch::Tensor>();
//     py_->svm.attr("_sh0") = torch::cat({sh0_old, sh0_add_bcast}, 0)
//                             .contiguous().detach().requires_grad_();

//     // _shs: zeros
//     const int n_sh_rest = (max_sh_degree_ + 1)*(max_sh_degree_ + 1) - 1;
//     auto shs_old = py_->svm.attr("_shs").cast<torch::Tensor>();
//     auto shs_add = torch::zeros({Nk, n_sh_rest, 3}, torch::dtype(torch::kFloat32).device(dev));
//     py_->svm.attr("_shs") = torch::cat({shs_old, shs_add}, 0)
//                             .contiguous().detach().requires_grad_();

//     // --- 6) Rebuild grid links; grow _geo_grid_pts only if grid expanded -------------
//     auto grid_pts_key_new = py_->svm.attr("grid_pts_key").cast<torch::Tensor>(); // [M,3]
//     const int64_t M_prev  = grid_pts_key_.defined() ? grid_pts_key_.size(0) : 0;
//     const int64_t M_curr  = grid_pts_key_new.size(0);

//     if (M_curr > M_prev) {
//         auto grow = torch::full({M_curr - M_prev, 1}, -10.0f,
//                         torch::dtype(torch::kFloat32).device(dev))
//                     .contiguous().detach().requires_grad_();
//         appendGroup_(/*group_idx=*/0, grow, "_geo_grid_pts", &this->_geo_grid_pts_);
//     }
//     // Append SH params to optimizer groups (topology done above)
//     appendGroup_(/*group_idx=*/1, sh0_add_bcast, "_sh0", &this->sh0_);
//     appendGroup_(/*group_idx=*/2, shs_add,       "_shs", &this->shs_);

//     // --- 7) Sync mirrors to C++ for renderer ----------------------------------------
//     auto fetch = [&](const char* name){ return py_->svm.attr(name).cast<torch::Tensor>().contiguous(); };
//     this->oct_path_      = fetch("octpath");
//     this->oct_level_     = fetch("octlevel");
//     this->center_        = fetch("vox_center");
//     this->size_          = fetch("vox_size").squeeze(1);
//     this->vox_size_inv_  = 1.0f / size_;
//     this->grid_pts_key_  = fetch("grid_pts_key");
//     this->vox_key_       = fetch("vox_key");
//     this->subdiv_p_      = py_->svm.attr("_subdiv_p").cast<torch::Tensor>().requires_grad_(true);

//     // stats buffer resize
//     this->max_w_ = torch::zeros({center_.size(0), 1}, torch::dtype(torch::kFloat32).device(dev));

//     // --- 8) Re-register with optimizer ----------------------------------------------
//     VOXEL_MODEL_TENSORS_TO_VEC

//     std::cout << "[increasePcd] Added " << Nk
//               << " bbox voxels at level " << static_cast<int>(octlevel_) << ". "
//               << "Total now: " << this->oct_path_.size(0) << "\n";
// }

void VoxelModel::applyTsdfTransparency(const torch::Tensor& tsdf_mask, float geo_value)
{
    TORCH_CHECK(tsdf_mask.defined(),
                "applyTsdfTransparency: tsdf_mask must be defined");
    TORCH_CHECK(tsdf_mask.dim() == 1,
                "applyTsdfTransparency: tsdf_mask must be 1D [N]");
    TORCH_CHECK(tsdf_mask.size(0) == center_.size(0),
                "applyTsdfTransparency: tsdf_mask.size(0) must equal num voxels");

    TORCH_CHECK(vox_key_.defined(),
                "applyTsdfTransparency: vox_key_ must be defined");
    TORCH_CHECK(_geo_grid_pts_.defined(),
                "applyTsdfTransparency: _geo_grid_pts_ must be defined");

    // [N] bool mask over voxels
    torch::Tensor mask = tsdf_mask.to(torch::kBool);
    if (!mask.any().item<bool>()) {
        std::cout << "[TSDF TRANS] no TSDF voxels marked, nothing to do.\n";
        return;
    }

    if (mask.device() != vox_key_.device()) {
        mask = mask.to(vox_key_.device());
    }

    torch::Tensor vox_key = vox_key_;  // [N,K]
    TORCH_CHECK(vox_key.dim() == 2,
                "applyTsdfTransparency: vox_key_ expected to be [N,K]");

    const auto N = vox_key.size(0);
    const auto K = vox_key.size(1);
    TORCH_CHECK(mask.size(0) == N,
                "applyTsdfTransparency: tsdf_mask and vox_key_ N mismatch");

    // ---- voxel-level stats ----
    int64_t n_vox_tsdf = mask.sum().item<int64_t>();   // #voxels marked by TSDF

    // Expand voxel mask to [N,K], then select all grid indices of marked voxels
    torch::Tensor mask2d   = mask.view({N, 1}).expand_as(vox_key); // [N,K]
    torch::Tensor flat_idx = vox_key.masked_select(mask2d);        // [M]

    if (flat_idx.numel() == 0) {
        std::cout << "[TSDF TRANS] tsdf_mask has true entries but flat_idx is empty – check vox_key_.\n";
        return;
    }

    // ---- unique grid indices (because grid points are shared by voxels) ----
    // 1) sort the flat indices
    auto sort_res          = flat_idx.sort();              // (values, indices)
    torch::Tensor sorted   = std::get<0>(sort_res);        // [M]
    auto M                 = sorted.size(0);

    // 2) build a mask that keeps only the first occurrence of each value
    torch::Tensor flat_idx_unique;
    if (M == 0) {
        flat_idx_unique = sorted;
    } else {
        auto opts_bool = torch::TensorOptions()
                            .dtype(torch::kBool)
                            .device(sorted.device());
        torch::Tensor mask = torch::empty({M}, opts_bool);

        // mask[0] = true (always keep first element)
        mask.index_put_({0}, true);

        if (M > 1) {
            // tail = sorted[1:]
            auto tail = sorted.slice(/*dim=*/0, /*start=*/1, /*end=*/M);
            // head = sorted[:-1]
            auto head = sorted.slice(/*dim=*/0, /*start=*/0, /*end=*/M - 1);
            // neq = (tail != head)
            auto neq  = tail.ne(head);                      // [M-1] bool

            // mask[1:] = neq
            using torch::indexing::Slice;
            mask.index_put_({Slice(1, torch::indexing::None)}, neq);
        }

        // 3) select unique values
        flat_idx_unique = sorted.masked_select(mask);       // [M_unique]
    }

    // geometry field: [Mg,1]
    torch::Tensor geo = _geo_grid_pts_;
    TORCH_CHECK(geo.dim() == 2 && geo.size(1) == 1,
                "applyTsdfTransparency: _geo_grid_pts_ must be [Mg,1]");
    TORCH_CHECK(flat_idx.dtype() == torch::kLong,
                "applyTsdfTransparency: vox_key_ must be int64 indices");

    if (flat_idx_unique.device() != geo.device()) {
        flat_idx_unique = flat_idx_unique.to(geo.device());
    }

    // ---- BEFORE modification stats ----
    torch::Tensor geo_sel_before      = geo.index_select(0, flat_idx_unique); // [M_unique,1]
    torch::Tensor geo_sel_before_flat = geo_sel_before.view({-1});           // [M_unique]

    float dens_min_before = geo_sel_before_flat.min().item().toFloat();
    float dens_max_before = geo_sel_before_flat.max().item().toFloat();

    double avg_grid_per_voxel =
        static_cast<double>(flat_idx_unique.numel()) /
        std::max<int64_t>(1, n_vox_tsdf);

    std::cout << "[TSDF TRANS] voxels_marked=" << n_vox_tsdf
              << "  K_per_voxel=" << K
              << "  grid_pts_selected=" << flat_idx.numel()
              << "  unique_grid_pts=" << flat_idx_unique.numel()
              << "  avg_unique_grid_per_voxel=" << avg_grid_per_voxel << "\n";

    std::cout << "[TSDF TRANS] BEFORE: density in those grid pts: "
              << "min=" << dens_min_before
              << ", max=" << dens_max_before
              << "  -> setting all to " << geo_value << "\n";

    // ---- Apply transparency without autograd tracking ----
    {
        torch::NoGradGuard no_grad;
        // geo: [Mg,1]; we write along dim=0, using flat_idx_unique as row indices
        geo.index_fill_(0, flat_idx_unique, geo_value);
    }

    // ---- AFTER modification stats ----
    torch::Tensor geo_sel_after      = geo.index_select(0, flat_idx_unique);
    torch::Tensor geo_sel_after_flat = geo_sel_after.view({-1});

    float dens_min_after = geo_sel_after_flat.min().item().toFloat();
    float dens_max_after = geo_sel_after_flat.max().item().toFloat();

    std::cout << "[TSDF TRANS] AFTER: density in those grid pts: "
              << "min=" << dens_min_after
              << ", max=" << dens_max_after << "\n";

    // write back
    _geo_grid_pts_ = geo;
}

void VoxelModel::applySingleVoxelTsdfTransparency(int64_t voxel_idx, float geo_value)
{
    TORCH_CHECK(vox_key_.defined(),
                "applySingleVoxelTsdfTransparency: vox_key_ must be defined");
    TORCH_CHECK(_geo_grid_pts_.defined(),
                "applySingleVoxelTsdfTransparency: _geo_grid_pts_ must be defined");
    TORCH_CHECK(voxel_idx >= 0 && voxel_idx < vox_key_.size(0),
                "applySingleVoxelTsdfTransparency: voxel_idx out of range");

    torch::Tensor geo = _geo_grid_pts_;  // [Mg,1]
    TORCH_CHECK(geo.dim() == 2 && geo.size(1) == 1,
                "applySingleVoxelTsdfTransparency: _geo_grid_pts_ must be [Mg,1]");

    torch::Tensor vox_key = vox_key_;    // [N, K]
    TORCH_CHECK(vox_key.dim() == 2,
                "applySingleVoxelTsdfTransparency: vox_key_ must be [N,K]");

    const auto N = vox_key.size(0);
    const auto K = vox_key.size(1);

    TORCH_CHECK(voxel_idx < N,
                "applySingleVoxelTsdfTransparency: voxel_idx >= N");

    // --- Get the K grid indices for this voxel ---
    auto idx_options = torch::TensorOptions().dtype(torch::kLong).device(vox_key.device());
    torch::Tensor vox_idx_tensor = torch::tensor({voxel_idx}, idx_options);  // [1]

    // row: [1, K] -> flatten to [K]
    torch::Tensor grid_idx_row = vox_key.index_select(0, vox_idx_tensor).view({-1}); // [K]
    TORCH_CHECK(grid_idx_row.dtype() == torch::kLong,
                "applySingleVoxelTsdfTransparency: vox_key_ must contain int64 indices");

    if (grid_idx_row.device() != geo.device()) {
        grid_idx_row = grid_idx_row.to(geo.device());
    }

    // ---- BEFORE modification stats ----
    torch::Tensor geo_sel_before      = geo.index_select(0, grid_idx_row);  // [K,1]
    torch::Tensor geo_sel_before_flat = geo_sel_before.view({-1});         // [K]

    float dens_min_before = geo_sel_before_flat.min().item<float>();
    float dens_max_before = geo_sel_before_flat.max().item<float>();

    std::cout << "[TSDF DEBUG] single voxel_idx=" << voxel_idx
              << " grid_pts=" << grid_idx_row.numel()
              << "  BEFORE density: min=" << dens_min_before
              << " max=" << dens_max_before
              << "  -> setting all to " << geo_value << "\n";

    // ---- Apply transparency without autograd tracking ----
    {
        torch::NoGradGuard no_grad;
        // geo: [Mg,1]; write along dim=0, using grid_idx_row as row indices
        geo.index_fill_(0, grid_idx_row, geo_value);
    }

    // ---- AFTER modification stats ----
    torch::Tensor geo_sel_after      = geo.index_select(0, grid_idx_row);
    torch::Tensor geo_sel_after_flat = geo_sel_after.view({-1});

    float dens_min_after = geo_sel_after_flat.min().item<float>();
    float dens_max_after = geo_sel_after_flat.max().item<float>();

    std::cout << "[TSDF DEBUG] single voxel_idx=" << voxel_idx
              << " AFTER density: min=" << dens_min_after
              << " max=" << dens_max_after << "\n";

    // write back
    _geo_grid_pts_ = geo;
}

void VoxelModel::createFromPcd(
    const std::map<point3D_id_t, Point3D>& pcd,
    const std::vector<sv::MiniCam>& cams)
{
    namespace py = pybind11;
    std::cout << "VoxelModel::createFromPcd() called with " << pcd.size() << " points.\n";
    fill_empty_cells_done_ = false;
    fill_empty_cells_warmup_notified_ = false;
    last_artificial_iter_ = -1;
    art_key_before_iter_ = torch::empty(
        {0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    TORCH_CHECK(global_scene_extent_ > 0.f, "global_scene_extent_ must be set (>0).");
    TORCH_CHECK(fixed_vox_size_   > 0.f, "fixed_vox_size_ must be set (>0).");
    TORCH_CHECK(max_sh_degree_ >= 0, "max_sh_degree_ must be >= 0.");

    const int N = static_cast<int>(pcd.size());
    const int32_t current_kf_count =
        cams.empty() ? static_cast<int32_t>(-1) : static_cast<int32_t>(cams.size());
    if (N == 0) {
        std::cerr << "[createFromPcd] Empty PCD — nothing to do.\n";
        return;
    }

    // ------------------------------------------------------------------------
    // 0) Fixed global scene + desired voxel size
    // ------------------------------------------------------------------------
    auto dev = torch::kCUDA;
    artificial_centers_accum_viz_ = torch::empty(
        {0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    artificial_sizes_accum_viz_ = torch::empty(
        {0, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    inactive_geo_centers_accum_viz_ = torch::empty(
        {0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    inactive_geo_sizes_accum_viz_ = torch::empty(
        {0, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    inactive_geo_rgba_accum_viz_ = torch::empty(
        {0, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    orb_created_centers_accum_viz_ = torch::empty(
        {0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    orb_created_sizes_accum_viz_ = torch::empty(
        {0, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    orb_created_rgba_accum_viz_ = torch::empty(
        {0, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    depthanything_created_centers_accum_viz_ = torch::empty(
        {0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    depthanything_created_sizes_accum_viz_ = torch::empty(
        {0, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    depthanything_created_rgba_accum_viz_ = torch::empty(
        {0, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    depthanything_fill_holes_created_centers_accum_viz_ = torch::empty(
        {0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    depthanything_fill_holes_created_sizes_accum_viz_ = torch::empty(
        {0, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    depthanything_fill_holes_created_rgba_accum_viz_ = torch::empty(
        {0, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    rendered_depth_created_centers_accum_viz_ = torch::empty(
        {0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    rendered_depth_created_sizes_accum_viz_ = torch::empty(
        {0, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    rendered_depth_created_rgba_accum_viz_ = torch::empty(
        {0, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    scene_center_ = torch::tensor(
        {global_scene_center_[0], global_scene_center_[1], global_scene_center_[2]},
        torch::dtype(torch::kFloat32).device(dev)
    ).contiguous();                                                  // [3]
    // scene_extent_ = torch::tensor({global_scene_extent_},
    //     torch::dtype(torch::kFloat32).device(dev)
    // ).contiguous();   
    
    int   outside_level = 0;
    const float scene_extent_scalar = global_scene_extent_ * std::pow(2.0f, outside_level);
    scene_extent_ = torch::tensor({scene_extent_scalar},
        torch::dtype(torch::kFloat32).device(dev)).contiguous();
    inside_extent_ = torch::tensor({global_scene_extent_},
        torch::dtype(torch::kFloat32).device(dev)
    ).contiguous();    
    scene_min_t_  = (scene_center_ - 0.5f * scene_extent_).contiguous(); // [3]

    // Pack inputs
    torch::Tensor xyz = torch::empty({N,3}, torch::dtype(torch::kFloat32).device(dev));
    torch::Tensor rgb = torch::empty({N,3}, torch::dtype(torch::kFloat32).device(dev));
    {
        int i = 0;
        for (const auto& kv : pcd) {
            const auto& P = kv.second;
            xyz[i][0] = P.xyz_(0);
            xyz[i][1] = P.xyz_(1);
            xyz[i][2] = P.xyz_(2);
            // P.color_ already scaled? If it’s 0..255, divide; if already 0..1, keep.
            // std::cout << "Point " << i << " color before scaling: "
            //           << P.color_(0) << ", " << P.color_(1) << ", " << P.color_(2) << "\n";
            rgb[i][0] = P.color_(0);
            rgb[i][1] = P.color_(1);
            rgb[i][2] = P.color_(2);
            ++i;
        }
    }
    // Accept both [0,1] and [0,255] color inputs (same policy as increasePcd).
    if (rgb.numel() > 0) {
        const float cmax = rgb.max().item<float>();
        if (cmax > 1.5f) {
            rgb.div_(255.0f);
        }
    }
    rgb.clamp_(0.0f, 1.0f);

    // Reset accumulated real-point history. We seed it later from actually
    // inserted/filtered real voxels (not raw pre-filter PCD).
    real_pcd_points_accum_cpu_ = torch::empty(
        {0, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    // ------------------------------------------------------------------------
    // 0.5) Initialize global PCD bounds (for later gating of fill_empty_cells_)
    // ------------------------------------------------------------------------
    {
        // Work on CPU for simplicity; this runs only once at initialization.
        auto xyz_cpu = xyz.to(torch::kCPU).contiguous();  // [N,3]

        auto min_res = xyz_cpu.min(/*dim=*/0, /*keepdim=*/false);
        auto max_res = xyz_cpu.max(/*dim=*/0, /*keepdim=*/false);

        torch::Tensor min_cpu = std::get<0>(min_res).contiguous();   // [3]
        torch::Tensor max_cpu = std::get<0>(max_res).contiguous();   // [3]

        // Store as CUDA tensors so they match the rest of the model state
        global_pcd_min_ = min_cpu.to(dev).contiguous();              // [3]
        global_pcd_max_ = max_cpu.to(dev).contiguous();              // [3]
        has_global_pcd_bb_ = true;

        std::cout << "[createFromPcd] global_pcd_min=" << global_pcd_min_
                  << " global_pcd_max=" << global_pcd_max_ << std::endl;
    }
    // ------------------------------------------------------------------------
    // 1) Compute octlevel from vox_size using utils.vox_size_2_level
    //    (mirror points_init behavior: round/clamp)
    // ------------------------------------------------------------------------
    py::gil_scoped_acquire gil;
    static py::module svr_utils = py::module::import("svraster_cuda").attr("utils");
    static py::module svm_mod = []{
        py::module sys = py::module::import("sys");
        sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
        return py::module::import("src.sparse_voxel_model");
    }();
    static py::module act_mod   = py::module::import("src.utils.activation_utils");
    static py::module oct_utils = py::module::import("src.utils.octree_utils");
    static py::module torch_mod = py::module::import("torch");
    static py::module svr_mod   = py::module::import("svraster_cuda").attr("renderer");

    // Dense-core estimation is deferred until real-point history is seeded
    // near the end of createFromPcd().
    has_dense_core_bb_ = false;

    const int MAX_L = max_num_levels_;
    // Level as float (tensor) then rounded like points_init (nearest by default)
    auto vox_size_t = torch::full({1}, fixed_vox_size_, torch::dtype(torch::kFloat32).device(dev)); // [1]
    py::object L_fp_py = oct_utils.attr("vox_size_2_level")(py::cast(scene_extent_), py::cast(vox_size_t));
    auto L_fp = L_fp_py.cast<torch::Tensor>().round();                                               // [1] float
    auto L_clamped = L_fp.clamp_min(1).clamp_max(MAX_L).to(torch::kInt8).contiguous();               // [1] int8

    // cache scalar level
    octlevel_ = L_clamped.item<int8_t>();
    // cache [1,1] effective voxel size
    py::object vox_eff_py = oct_utils.attr("level_2_vox_size")(
        py::cast(scene_extent_), 
        // level_2_vox_size expects [N,1]; give it [1,1] then keep it cached
        py::cast(L_clamped.view({1,1}))
    );
    vox_eff_ = vox_eff_py.cast<torch::Tensor>().view({1,1}).contiguous();    
    // std::cout << "[createFromPcd] Using octlevel=" << octlevel[0][0].item<int>()
    //           << " (vox_size=" << vox_eff[0][0].item<float>() << " m) for fixed_vox_size_="
    //           << fixed_vox_size_ << " m.\n";

    // ------------------------------------------------------------------------
    // 2) Compute ijk with this level/voxel size (mirror points_init)
    // ------------------------------------------------------------------------
    // ijk = ((xyz - scene_min) / vox_size).long()
    auto vox_effN  = vox_eff_.expand({N,1});                                                         // [N,1]
    torch::Tensor ijk = ((xyz - scene_min_t_) / vox_effN).floor().to(torch::kLong);                  // [N,3]

    auto octlevelN = torch::full({N,1}, octlevel_, torch::dtype(torch::kInt8).device(dev)).contiguous(); // [N,1]
    auto L_long    = octlevelN.to(torch::kLong);

    // In libtorch C++, pass a vector to cat
    std::vector<torch::Tensor> cat_inputs{ijk, L_long};
    auto ijkl = torch::cat(cat_inputs, /*dim=*/1);     

    // Use Python torch.unique(dim=0, return_inverse=True) (simple & robust)
    torch::Tensor ijkl_unq, invmap;
    {
        py::tuple uniq = torch_mod.attr("unique")(
            py::cast(ijkl.contiguous()),
            py::arg("dim") = 0,
            py::arg("sorted") = true,
            py::arg("return_inverse") = true
        );
        ijkl_unq = uniq[0].cast<torch::Tensor>().contiguous(); // [Nu,4]
        invmap   = uniq[1].cast<torch::Tensor>().contiguous(); // [N]
    }

    torch::Tensor ijk_u, L_u; 
    auto parts = torch::split_with_sizes(ijkl_unq, {3, 1}, /*dim=*/1); 
    ijk_u = parts[0].contiguous(); L_u = parts[1].to(torch::kInt8).contiguous(); 
    L_u = L_u.to(torch::kInt8).contiguous(); // [Nu,1]
    int64_t Nu = ijk_u.size(0);

    auto rgb_u = torch::zeros({Nu,3}, torch::dtype(torch::kFloat32).device(dev));
    // rgb_u.index_add_(0, invmap, rgb);
    // auto counts = torch::bincount(invmap, {}, Nu).view({Nu,1}).clamp_min(1);
    // rgb_u = rgb_u / counts;
    rgb_u.index_reduce_(
        /*dim=*/0,
        /*index=*/invmap,
        /*source=*/rgb,
        /*reduce=*/"mean",
        /*include_self=*/false
    );

    // Defensive bound check: ijk in [0, 2^L)
    const int L0 = L_u[0].item<int8_t>();               // all rows have same level
    const long limit = (1L << L0);
    TORCH_CHECK((ijk_u.index({torch::indexing::Slice(),0}) >= 0).all().item<bool>() &&
                (ijk_u.index({torch::indexing::Slice(),1}) >= 0).all().item<bool>() &&
                (ijk_u.index({torch::indexing::Slice(),2}) >= 0).all().item<bool>(),
                "Points below scene_min — enlarge global_scene_extent_ or filter.");
    TORCH_CHECK((ijk_u.index({torch::indexing::Slice(),0}) < limit).all().item<bool>() &&
                (ijk_u.index({torch::indexing::Slice(),1}) < limit).all().item<bool>() &&
                (ijk_u.index({torch::indexing::Slice(),2}) < limit).all().item<bool>(),
                "Points exceed scene bounds — enlarge global_scene_extent_ or filter.");

    // ------------------------------------------------------------------------
    // 3) utils: ijk -> octpath (no constructor calls)
    // ------------------------------------------------------------------------
    py::object octpath_py = svr_utils.attr("ijk_2_octpath")(py::cast(ijk_u.contiguous()),
                                                            py::cast(L_u.contiguous()));
    auto octpath = octpath_py.cast<torch::Tensor>().contiguous();            // [Nu,1] int64

    // ------------------------------------------------------------------------
    // 3.5) Optional camera-based filtering for initialization candidates
    //      (same logic family as increasePcd insertion filter)
    // ------------------------------------------------------------------------
    if (!cams.empty() && octpath.size(0) > 0) {
        py::tuple dec = oct_utils.attr("octpath_decoding")(
            py::cast(octpath.contiguous()),
            py::cast(L_u.contiguous()),
            py::cast(scene_center_.contiguous()),
            py::cast(scene_extent_.contiguous()));
        at::Tensor vox_center = dec[0].cast<at::Tensor>().contiguous(); // [Nu,3]
        at::Tensor vox_size   = dec[1].cast<at::Tensor>().contiguous(); // [Nu,1] or [Nu]
        if (vox_size.dim() == 1) {
            vox_size = vox_size.view({-1, 1});
        }

        py::list py_cams;
        py::object py_cuda = torch_mod.attr("device")("cuda");
        auto move_attr_to_cuda_if_tensor = [&](py::object& obj, const char* name) {
            if (py::hasattr(obj, name)) {
                py::object t = obj.attr(name);
                if (py::hasattr(t, "is_cuda") && !py::bool_(t.attr("is_cuda"))) {
                    obj.attr(name) = t.attr("to")(py_cuda);
                }
            }
        };
        for (const auto& c : cams) {
            py::object py_cam = MiniCam_to_py(c);
            move_attr_to_cuda_if_tensor(py_cam, "w2c");
            move_attr_to_cuda_if_tensor(py_cam, "c2w");
            move_attr_to_cuda_if_tensor(py_cam, "position");
            move_attr_to_cuda_if_tensor(py_cam, "lookat");
            py_cams.append(py_cam);
        }

        const int64_t Nu_before = octpath.size(0);
        at::Tensor rate = svr_mod.attr("mark_max_samp_rate")(
            py_cams,
            py::cast(octpath),
            py::cast(vox_center),
            py::cast(vox_size)).cast<at::Tensor>();
        if (rate.dim() == 2 && rate.size(1) == 1) {
            rate = rate.squeeze(1);
        }
        rate = rate.to(torch::kFloat32);
        at::Tensor kept = (rate > 0.0f);
        const int64_t n_rate_pos = kept.sum().item<int64_t>();

        const float near_thresh = 0.2f;
        int64_t n_near_hit = 0;
        if (near_thresh > 0.0f) {
            at::Tensor is_near = svr_mod.attr("mark_near")(
                py_cams,
                py::cast(octpath),
                py::cast(vox_center),
                py::cast(vox_size),
                py::float_(near_thresh)).cast<at::Tensor>();
            if (is_near.dim() == 2 && is_near.size(1) == 1) {
                is_near = is_near.squeeze(1);
            }
            is_near = is_near.to(torch::kBool);
            kept = kept & (~is_near);
            n_near_hit = is_near.sum().item<int64_t>();
        }

        kept = kept.view({-1}).to(torch::kBool);
        auto idx = torch::nonzero(kept).view({-1});
        const int64_t K = idx.size(0);

        if (K > 0 && K < octpath.size(0)) {
            octpath = octpath.index_select(0, idx).contiguous();
            L_u     = L_u.index_select(0, idx).contiguous();
            ijk_u   = ijk_u.index_select(0, idx).contiguous();
            rgb_u   = rgb_u.index_select(0, idx).contiguous();
            Nu = octpath.size(0);
        }
        // std::cout << "[createFromPcd/filter] Nu_before=" << Nu_before
        //           << " rate>0=" << n_rate_pos
        //           << " near_hit=" << n_near_hit
        //           << " kept_final=" << Nu << "\n";
    }
    
    // Create/reuse SVM (DO NOT call model_init/points_init/...)
    if (!py_->svm || py_->svm.is_none()) {
        py::object SVM = svm_mod.attr("SparseVoxelModel");
        py_->svm = SVM(
            py::arg("sh_degree") = max_sh_degree_,
            py::arg("white_background") = white_background_,
            py::arg("black_background") = black_background_);
    }
    py_->svm.attr("white_background") = py::bool_(white_background_);
    py_->svm.attr("black_background") = py::bool_(black_background_);

    // scene tensors (single fixed scene; no outside shells => inside_extent == scene_extent)
    py_->svm.attr("scene_center")  = scene_center_.contiguous();   // [3]
    py_->svm.attr("scene_extent")  = scene_extent_.contiguous(); // [1]
    // py_->svm.attr("inside_extent") = scene_extent_.contiguous(); // [1]
    py_->svm.attr("inside_extent") = inside_extent_.contiguous(); // [1]

    // Topology
    py_->svm.attr("octpath")  = octpath;                          // [Nu,1] int64
    py_->svm.attr("octlevel") = L_u.contiguous();
    // py_->svm.attr("octlevel") = L_u.view({Nu,1}).contiguous();    // [Nu,1] int8

    // ------------------------------------------------------------------------
    // 4) Initialize learnables directly
    // ------------------------------------------------------------------------
    // Subdivision priority
    auto subdiv_p = torch::ones({Nu,1}, torch::dtype(torch::kFloat32).device(dev)).requires_grad_();
    py_->svm.attr("_subdiv_p")     = subdiv_p.detach().requires_grad_(); 

    // SH-0 from fused RGB
    py::object sh0_dc_py = act_mod.attr("rgb2shzero")(py::cast(rgb_u.contiguous()));
    auto sh0_dc = sh0_dc_py.cast<torch::Tensor>().contiguous().requires_grad_(); // [Nu,3]
    py_->svm.attr("_sh0")          = sh0_dc.detach().requires_grad_(); 

    // Higher-degree SH zeros
    const int n_sh_rest = (max_sh_degree_ + 1)*(max_sh_degree_ + 1) - 1;
    auto shs = torch::zeros({Nu, n_sh_rest, 3}, torch::dtype(torch::kFloat32).device(dev)).requires_grad_();
    py_->svm.attr("_shs")          = shs.detach().requires_grad_(); 

    // Active SH degree
    py_->svm.attr("active_sh_degree") = py::int_(std::min(max_sh_degree_, 3));

    // Grid link => allocate per-grid-point density
    torch::Tensor grid_pts_key = py_->svm.attr("grid_pts_key").cast<torch::Tensor>(); // [Mg,3] int64
    auto geo_grid = makeGeoGridInitRows_(
        grid_pts_key,
        /*begin=*/0,
        /*end=*/grid_pts_key.size(0),
        /*default_value=*/-10.0f);
    py_->svm.attr("_geo_grid_pts") = geo_grid.detach().requires_grad_();

    // ------------------------------------------------------------------------
    // 5) Sync to C++ members
    // ------------------------------------------------------------------------
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

    // learnables
    this->_geo_grid_pts_ = fetch("_geo_grid_pts").requires_grad_(true);
    this->sh0_           = fetch("_sh0").requires_grad_(true);
    this->shs_           = fetch("_shs").requires_grad_(true);
    this->subdiv_p_      = fetch("_subdiv_p").requires_grad_(true);

    // stats buffer
    this->max_w_ = torch::zeros({center_.size(0), 1}, torch::dtype(torch::kFloat32).device(dev));
    if (this->center_.defined() && this->size_.defined() &&
        this->center_.dim() == 2 && this->center_.size(1) == 3 &&
        this->center_.size(0) > 0) {
        orb_created_centers_accum_viz_ = this->center_.detach().clone().contiguous();
        orb_created_sizes_accum_viz_ =
            this->size_.detach().view({-1, 1}).clone().contiguous();
        auto alpha_orb = torch::full(
            {orb_created_centers_accum_viz_.size(0), 1},
            0.95f,
            torch::TensorOptions().dtype(torch::kFloat32).device(dev));
        orb_created_rgba_accum_viz_ =
            torch::cat({rgb_u.clamp(0.0f, 1.0f).contiguous(), alpha_orb}, 1).contiguous();

        if (max_inactive_geo_viz_accum_ > 0 &&
            orb_created_centers_accum_viz_.size(0) > max_inactive_geo_viz_accum_) {
            const int64_t total_orb = orb_created_centers_accum_viz_.size(0);
            auto keep = torch::linspace(
                0.0,
                static_cast<double>(total_orb - 1),
                max_inactive_geo_viz_accum_,
                torch::TensorOptions().dtype(torch::kFloat32).device(dev))
                .round().to(torch::kLong);
            orb_created_centers_accum_viz_ =
                orb_created_centers_accum_viz_.index_select(0, keep).contiguous();
            orb_created_sizes_accum_viz_ =
                orb_created_sizes_accum_viz_.index_select(0, keep).contiguous();
            orb_created_rgba_accum_viz_ =
                orb_created_rgba_accum_viz_.index_select(0, keep).contiguous();
        }

        // std::cout << "[rerun/orb] logged_initial_voxels="
        //           << orb_created_centers_accum_viz_.size(0)
        //           << " entity=world/orb/voxels_created"
        //           << std::endl;
    }
    // Initial map from PCD is treated as real geometry.
    this->is_artificial_voxel_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    // Initial topology is created directly from ORB-SLAM map points.
    this->is_orb_voxel_ = torch::ones(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    this->is_inactive_geo_voxel_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    this->is_rgbd_fill_render_holes_voxel_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    this->is_depthanything_fill_holes_voxel_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    this->is_promoted_artificial_voxel_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    this->total_promoted_artificial_voxels_ = 0;
    this->exist_since_iter_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kInt32).device(dev));
    this->exist_since_kf_ = torch::full(
        {center_.size(0)},
        current_kf_count,
        torch::TensorOptions().dtype(torch::kInt32).device(dev));
    this->geometrically_unstable_voxel_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    this->rendered_depth_candidate_voxel_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(dev));
    this->rendered_depth_candidate_support_count_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kInt32).device(dev));
    this->rendered_depth_candidate_last_seen_kf_ = torch::full(
        {center_.size(0)},
        static_cast<int32_t>(-1),
        torch::TensorOptions().dtype(torch::kInt32).device(dev));
    this->rendered_depth_candidate_source_kind_ = torch::zeros(
        {center_.size(0)},
        torch::TensorOptions().dtype(torch::kInt32).device(dev));

    // Seed dense-core history from raw createFromPcd points (CPU).
    // This matches heuristic expectations better than regularized voxel centers.
    real_pcd_points_accum_cpu_ = xyz.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (max_real_pcd_points_ > 0 && real_pcd_points_accum_cpu_.size(0) > max_real_pcd_points_) {
        const int64_t total = real_pcd_points_accum_cpu_.size(0);
        auto idx = torch::linspace(
            0.0, static_cast<double>(total - 1), max_real_pcd_points_,
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).round().to(torch::kLong);
        real_pcd_points_accum_cpu_ = real_pcd_points_accum_cpu_.index_select(0, idx).contiguous();
    }

    // Compute dense-core once at create-time so insertion-time far filtering
    // can use it immediately if that option is enabled later.
    // std::cout << "[dense_core/refresh][create] begin points="
    //           << real_pcd_points_accum_cpu_.size(0)
    //           << " rate=" << dense_core_pcd_density_rate_ << "\n";
    const bool refreshed_dense_core = refreshDenseCoreBBFromCurrentVoxels();
    if (!refreshed_dense_core || !has_dense_core_bb_) {
        has_dense_core_bb_ = false;
        // std::cout << "[dense_core/refresh][create] done has_bb=0 updated=0\n";
        // std::cout << "[createFromPcd] dense-core refresh unavailable; dense-core bbox not set.\n";
    } else {
        // std::cout << "[dense_core/refresh][create] done has_bb=1 updated=1\n";
        // std::cout << "[createFromPcd] dense_core_bb_min=" << dense_core_bb_min_
        //           << " dense_core_bb_max=" << dense_core_bb_max_ << std::endl;
    }

    // Register with your optimizer
    VOXEL_MODEL_TENSORS_TO_VEC

    // inc = 0 for the initial snapshot (or any monotonic counter you keep)
    // const int64_t inc0 = 0;
    // 1) Log the global fixed scene AABB once
    // rrLogGlobalSceneAABB(inc0);
    // 2) Log ALL voxels after creation
    // rrLogVoxelBoxes(this->center_, this->size_, /*max_boxes_for_viz=*/10000000000, "vox_all", inc0);
}

void VoxelModel::increasePcd(
    std::vector<float> pcd_full,
    std::vector<float> colors,
    const int iteration,
    const std::vector<sv::MiniCam>& cams)
{
    namespace py = pybind11;
    const int Nf = static_cast<int>(pcd_full.size());
    last_increase_pcd_stats_ = IncreasePcdStats{};
    if (Nf < 3 || colors.size() < 3) return;
    int N = Nf / 3;
    const int64_t raw_points_in = N;
    last_increase_pcd_stats_.raw_points_in = raw_points_in;
    const int32_t current_kf_count =
        cams.empty() ? static_cast<int32_t>(-1) : static_cast<int32_t>(cams.size());
    const bool insert_rendered_depth_candidate = pending_insert_rendered_depth_candidate_;
    const bool insert_rendered_depth_candidate_as_real_protected =
        pending_insert_rendered_depth_candidate_as_real_protected_;
    // std::cout << "VoxelModel::increasePcd() called with " << N << " points).\n";
    TORCH_CHECK(global_scene_extent_ > 0.f && fixed_vox_size_ > 0.f,
                "increasePcd: scene extent / fixed vox size not set.");
    TORCH_CHECK(py_->svm && !py_->svm.is_none(),
                "increasePcd: SVM not initialized; call createFromPcd first.");

    // ——— 0) Build CPU tensors from raw arrays, normalize RGB ————————
    torch::Tensor xyz_cpu = torch::from_blob(
        pcd_full.data(), { (int64_t)N, 3 },
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
    ).clone();

    torch::Tensor rgb_cpu = torch::from_blob(
        colors.data(), { (int64_t)N, 3 },
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
    ).clone();
    // Accept both [0,1] and [0,255] color inputs.
    if (rgb_cpu.numel() > 0) {
        const float cmax = rgb_cpu.max().item<float>();
        if (cmax > 1.5f) rgb_cpu.div_(255.0f);
    }
    rgb_cpu.clamp_(0.0f, 1.0f);

    // Optional insertion-time far filtering by current dense-core bbox.
    if (filter_far_voxels_on_insert_) {
        const int64_t n_before = xyz_cpu.size(0);
        if (hasDenseCoreBB()) {
            auto bb_min_cpu = dense_core_bb_min_.to(torch::kCPU).to(torch::kFloat32).contiguous().view({1, 3});
            auto bb_max_cpu = dense_core_bb_max_.to(torch::kCPU).to(torch::kFloat32).contiguous().view({1, 3});
            auto in_dense_core =
                (xyz_cpu >= bb_min_cpu).all(/*dim=*/1) &
                (xyz_cpu <= bb_max_cpu).all(/*dim=*/1);
            in_dense_core = in_dense_core.to(torch::kBool).contiguous();
            const int64_t n_kept = in_dense_core.sum().item<int64_t>();
            if (n_kept < n_before) {
                auto keep_idx = torch::nonzero(in_dense_core).view({-1}).contiguous();
                xyz_cpu = xyz_cpu.index_select(0, keep_idx).contiguous();
                rgb_cpu = rgb_cpu.index_select(0, keep_idx).contiguous();
            }
            // std::cout << "[increasePcd/filter_far_insert] kept "
            //           << n_kept << "/" << n_before << " points.\n";
        }
        if (xyz_cpu.numel() == 0 || xyz_cpu.size(0) == 0) {
            return;
        }
    }

    N = static_cast<int>(xyz_cpu.size(0));
    const int64_t points_after_far_filter = N;
    last_increase_pcd_stats_.points_after_far_filter = points_after_far_filter;

    // Move to CUDA
    auto dev = torch::kCUDA;
    torch::Tensor xyz = xyz_cpu.to(dev);
    torch::Tensor rgb = rgb_cpu.to(dev);

    // Build the min/max AABB tensor once:
    at::Tensor aabb = torch::stack({
        torch::tensor({ global_scene_center_[0] - 0.5f*global_scene_extent_,
                        global_scene_center_[1] - 0.5f*global_scene_extent_,
                        global_scene_center_[2] - 0.5f*global_scene_extent_ }),
        torch::tensor({ global_scene_center_[0] + 0.5f*global_scene_extent_,
                        global_scene_center_[1] + 0.5f*global_scene_extent_,
                        global_scene_center_[2] + 0.5f*global_scene_extent_ })
    });

    // --- 2) Compute octlevel from fixed_vox_size_ (same as createFromPcd)
    py::gil_scoped_acquire gil;
    static py::module svr_utils = py::module::import("svraster_cuda").attr("utils");
    static py::module svm_mod = []{
        py::module sys = py::module::import("sys");
        sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
        return py::module::import("src.sparse_voxel_model");
    }();
    static py::module act_mod   = py::module::import("src.utils.activation_utils");
    static py::module oct_utils = py::module::import("src.utils.octree_utils");
    static py::module torch_mod = py::module::import("torch");
    const int MAX_L = max_num_levels_;

    auto vox_effN  = vox_eff_.expand({N,1});                                                         // [N,1]
    torch::Tensor ijk = ((xyz - scene_min_t_) / vox_effN).floor().to(torch::kLong);                  // [N,3]
    // torch::Tensor ijk = ((xyz - scene_min_t_) / vox_effN).to(torch::kLong); 

    auto octlevelN = torch::full({N,1}, octlevel_, torch::dtype(torch::kInt8).device(dev)).contiguous(); // [N,1]
    auto L_long    = octlevelN.to(torch::kLong);

    // In libtorch C++, pass a vector to cat
    std::vector<torch::Tensor> cat_inputs{ijk, L_long};
    auto ijkl = torch::cat(cat_inputs, /*dim=*/1);                                                         // [N,4]

    torch::Tensor ijkl_unq, invmap;
    {
        py::tuple uniq = torch_mod.attr("unique")(
            py::cast(ijkl.contiguous()),
            py::arg("dim") = 0,
            py::arg("sorted") = true,
            py::arg("return_inverse") = true
        );
        ijkl_unq = uniq[0].cast<torch::Tensor>().contiguous(); // [Nu,4]
        invmap   = uniq[1].cast<torch::Tensor>().contiguous(); // [N]
    }
    torch::Tensor ijk_u, L_u;
    auto parts = torch::split_with_sizes(ijkl_unq, {3,1}, 1);
    ijk_u = parts[0].contiguous();                                                                    // [Nu,3]
    L_u   = parts[1].to(torch::kInt8).contiguous();                                                   // [Nu,1]
    int64_t Nu = ijk_u.size(0);
    const int64_t unique_voxel_candidates_before_insert_filter = Nu;

    // Defensive bound check: ijk in [0, 2^L)
    const int8_t L0 = L_u[0].item<int8_t>();      // all rows share the same level
    const long limit = (1L << L0);
    // TORCH_CHECK(
    //     (ijk_u.index({torch::indexing::Slice(),0}) >= 0).all().item<bool>() &&
    //     (ijk_u.index({torch::indexing::Slice(),1}) >= 0).all().item<bool>() &&
    //     (ijk_u.index({torch::indexing::Slice(),2}) >= 0).all().item<bool>(),
    //     "increasePcd: points below scene_min — enlarge global_scene_extent_ or filter.");
    // TORCH_CHECK(
    //     (ijk_u.index({torch::indexing::Slice(),0}) < limit).all().item<bool>() &&
    //     (ijk_u.index({torch::indexing::Slice(),1}) < limit).all().item<bool>() &&
    //     (ijk_u.index({torch::indexing::Slice(),2}) < limit).all().item<bool>(),
    //     "increasePcd: points exceed scene bounds — enlarge global_scene_extent_ or filter.");
    const bool in_low =
    (ijk_u.index({torch::indexing::Slice(),0}) >= 0).all().item<bool>() &&
    (ijk_u.index({torch::indexing::Slice(),1}) >= 0).all().item<bool>() &&
    (ijk_u.index({torch::indexing::Slice(),2}) >= 0).all().item<bool>();
    const bool in_high =
        (ijk_u.index({torch::indexing::Slice(),0}) < limit).all().item<bool>() &&
        (ijk_u.index({torch::indexing::Slice(),1}) < limit).all().item<bool>() &&
        (ijk_u.index({torch::indexing::Slice(),2}) < limit).all().item<bool>();
    if (!(in_low && in_high)) {
        std::cout << "[increasePcd] OOB detected — reinitializing via createFromPcd().\n";
        // A) Compute scene from this batch using main_scene_bound_pcd_heuristic
        {
            bool updated_scene_from_py = false;
            try {
                py::gil_scoped_acquire gil;
                static py::module bound_utils = py::module::import("src.utils.bounding_utils");
                static py::module types       = py::module::import("types");

                // xyz_cpu is a CPU float32 tensor with shape [N,3]
                py::object ns = types.attr("SimpleNamespace")(
                    "points"_a = py::cast(xyz_cpu.contiguous()).attr("numpy")()
                );
                py::tuple cr = bound_utils.attr("main_scene_bound_pcd_heuristic")(
                    ns, py::float_(dense_core_pcd_density_rate_));
                std::vector<double> c_vec = py::cast<std::vector<double>>(cr[0]);
                TORCH_CHECK(c_vec.size() == 3, "main_scene_bound_pcd_heuristic: center must have 3 elements");
                float radius = py::cast<float>(cr[1]);
                if (std::isfinite(radius) && radius > 0.0f) {
                    global_scene_center_[0] = static_cast<float>(c_vec[0]);
                    global_scene_center_[1] = static_cast<float>(c_vec[1]);
                    global_scene_center_[2] = static_cast<float>(c_vec[2]);
                    global_scene_extent_    = 2.0f * radius;
                    updated_scene_from_py = true;
                }
            } catch (const std::exception&) {
                updated_scene_from_py = false;
            }

            if (!updated_scene_from_py) {
                auto pts = xyz_cpu.to(torch::kCPU).to(torch::kFloat32).contiguous();
                auto center = std::get<0>(pts.median(/*dim=*/0, /*keepdim=*/false)).contiguous();
                auto dist = std::get<0>((pts - center.view({1, 3})).abs().max(/*dim=*/1, /*keepdim=*/false))
                    .to(torch::kFloat32).contiguous();
                auto sorted = std::get<0>(dist.sort(/*dim=*/0, /*descending=*/false)).contiguous();
                const int64_t n = sorted.size(0);
                int64_t p90_idx = static_cast<int64_t>(std::llround(0.90 * static_cast<double>(std::max<int64_t>(0, n - 1))));
                p90_idx = std::max<int64_t>(0, std::min<int64_t>(p90_idx, std::max<int64_t>(0, n - 1)));
                float radius = (n > 0) ? sorted.index({p90_idx}).item<float>() : 1.0f;
                if (!std::isfinite(radius) || radius <= 0.0f) {
                    radius = (n > 0) ? sorted.index({n - 1}).item<float>() : 1.0f;
                }
                if (!std::isfinite(radius) || radius <= 0.0f) {
                    radius = 1.0f;
                }

                global_scene_center_[0] = center.index({0}).item<float>();
                global_scene_center_[1] = center.index({1}).item<float>();
                global_scene_center_[2] = center.index({2}).item<float>();
                global_scene_extent_ = 2.0f * radius;
            }
        }
        // B) Reset Python SVM so createFromPcd builds fresh state
        {
            py::gil_scoped_acquire gil;
            py_->svm = py::none();
        }
        // C) Convert current batch to a temporary map and call createFromPcd
        {
            std::map<point3D_id_t, Point3D> tmp;
            static point3D_id_t id_seed = 1;  // local seq; independent of COLMAP ids, etc.
            auto xyz_re = xyz_cpu.contiguous();
            auto rgb_re = rgb_cpu.contiguous();
            auto xyz_acc = xyz_re.accessor<float, 2>();
            auto rgb_acc = rgb_re.accessor<float, 2>();
            for (int i = 0; i < N; ++i) {
                Point3D P;
                // world coords
                P.xyz_(0) = xyz_acc[i][0];
                P.xyz_(1) = xyz_acc[i][1];
                P.xyz_(2) = xyz_acc[i][2];
                // rgb_cpu is normalized to [0,1]. Point3D stores uint8.
                P.color_(0) = static_cast<uint8_t>(std::round(std::clamp(rgb_acc[i][0], 0.0f, 1.0f) * 255.0f));
                P.color_(1) = static_cast<uint8_t>(std::round(std::clamp(rgb_acc[i][1], 0.0f, 1.0f) * 255.0f));
                P.color_(2) = static_cast<uint8_t>(std::round(std::clamp(rgb_acc[i][2], 0.0f, 1.0f) * 255.0f));
                tmp.emplace(id_seed++, P);
            }
            createFromPcd(tmp, cams);
        }
        // Log and exit this call
        return;
    }

    auto rgb_u = torch::zeros({Nu,3}, torch::dtype(torch::kFloat32).device(dev));
    // rgb_u.index_add_(0, invmap, rgb);
    // auto counts = torch::bincount(invmap, {}, Nu).view({Nu,1}).clamp_min(1);
    // rgb_u = rgb_u / counts;
    rgb_u.index_reduce_(
        /*dim=*/0,
        /*index=*/invmap,
        /*source=*/rgb,
        /*reduce=*/"mean",
        /*include_self=*/false
    );

    // ── 4) Build octpath for this batch ─────────────────────────────────────
    py::object octpath_py = svr_utils.attr("ijk_2_octpath")(py::cast(ijk_u), py::cast(L_u));
    auto octpath_new = octpath_py.cast<torch::Tensor>().contiguous();                                   // [Nu,1] int64

    // ---- Active insertion-time SVR-style filtering (kept separate from OLD block) ----
    // Keep this at insertion-time so bad candidates never enter the topology.
    // Prune-time filtering in VoxelMapper remains useful as a second cleanup stage.
    if (!cams.empty() && octpath_new.size(0) > 0) {
        static py::module svr_mod   = py::module::import("svraster_cuda").attr("renderer");
        static py::module oct_utils = py::module::import("src.utils.octree_utils");

        // Decode voxel centers/sizes for current candidates.
        py::tuple dec = oct_utils.attr("octpath_decoding")(
            py::cast(octpath_new.contiguous()),
            py::cast(L_u.contiguous()),
            py::cast(scene_center_.contiguous()),
            py::cast(scene_extent_.contiguous())
        );
        at::Tensor vox_center = dec[0].cast<at::Tensor>().contiguous(); // [Nu,3]
        at::Tensor vox_size   = dec[1].cast<at::Tensor>().contiguous(); // [Nu,1] or [Nu]
        if (vox_size.dim() == 1) {
            vox_size = vox_size.view({-1, 1});
        }

        // Build Python list of CUDA MiniCams.
        py::list py_cams;
        py::object py_cuda = torch_mod.attr("device")("cuda");
        auto move_attr_to_cuda_if_tensor = [&](py::object& obj, const char* name) {
            if (py::hasattr(obj, name)) {
                py::object t = obj.attr(name);
                if (py::hasattr(t, "is_cuda") && !py::bool_(t.attr("is_cuda"))) {
                    obj.attr(name) = t.attr("to")(py_cuda);
                }
            }
        };
        for (const auto& c : cams) {
            py::object py_cam = MiniCam_to_py(c);
            move_attr_to_cuda_if_tensor(py_cam, "w2c");
            move_attr_to_cuda_if_tensor(py_cam, "c2w");
            move_attr_to_cuda_if_tensor(py_cam, "position");
            move_attr_to_cuda_if_tensor(py_cam, "lookat");
            py_cams.append(py_cam);
        }

        const int64_t Nu_before = octpath_new.size(0);

        // 1) Visibility / sampling-rate filtering
        at::Tensor rate = svr_mod.attr("mark_max_samp_rate")(
            py_cams,
            py::cast(octpath_new),
            py::cast(vox_center),
            py::cast(vox_size)
        ).cast<at::Tensor>();
        if (rate.dim() == 2 && rate.size(1) == 1) {
            rate = rate.squeeze(1);
        }
        rate = rate.to(torch::kFloat32);
        at::Tensor kept = (rate > 0.0f);
        int64_t n_rate_pos = kept.sum().item<int64_t>();

        // 2) Near filtering
        // NOTE: Upstream SVRaster layout initialization uses filter_near = -1 (disabled).
        const float near_thresh = 0.2f;
        int64_t n_near_hit = 0;
        if (near_thresh > 0.0f) {
            at::Tensor is_near = svr_mod.attr("mark_near")(
                py_cams,
                py::cast(octpath_new),
                py::cast(vox_center),
                py::cast(vox_size),
                py::float_(near_thresh)
            ).cast<at::Tensor>();
            if (is_near.dim() == 2 && is_near.size(1) == 1) {
                is_near = is_near.squeeze(1);
            }
            is_near = is_near.to(torch::kBool);
            kept = kept & (~is_near);
            n_near_hit = is_near.sum().item<int64_t>();
        }

        kept = kept.view({-1}).to(torch::kBool);
        auto idx = torch::nonzero(kept).view({-1});
        int64_t K = idx.size(0);

        if (K == 0) {
            std::cout << "[increasePcd/filter] all candidates filtered out, nothing to add.\n";
            return;
        }

        if (K < octpath_new.size(0)) {
            // Apply mask to ALL aligned tensors
            octpath_new = octpath_new.index_select(0, idx).contiguous(); // [K,1]
            L_u         = L_u.index_select(0, idx).contiguous();         // [K,1]
            ijk_u       = ijk_u.index_select(0, idx).contiguous();       // [K,3]
            rgb_u       = rgb_u.index_select(0, idx).contiguous();       // [K,3]
        }

        Nu = octpath_new.size(0); // update Nu after filtering

        TORCH_CHECK(L_u.sizes() == torch::IntArrayRef({Nu, 1}),
                    "L_u shape mismatch after insertion-time filtering");
        TORCH_CHECK(ijk_u.sizes() == torch::IntArrayRef({Nu, 3}),
                    "ijk_u shape mismatch after insertion-time filtering");
        TORCH_CHECK(rgb_u.sizes() == torch::IntArrayRef({Nu, 3}),
                    "rgb_u shape mismatch after insertion-time filtering");

        // std::cout << "[increasePcd/filter] Nu_before=" << Nu_before
        //           << " rate>0=" << n_rate_pos
        //           << " near_hit=" << n_near_hit
        //           << " kept_final=" << Nu << std::endl;
    }
    // ---- end active insertion-time filtering ----
    const int64_t unique_voxel_candidates_after_insert_filter = Nu;
    last_increase_pcd_stats_.unique_voxel_candidates_before_insert_filter =
        unique_voxel_candidates_before_insert_filter;
    last_increase_pcd_stats_.unique_voxel_candidates_after_insert_filter =
        unique_voxel_candidates_after_insert_filter;

    if (insert_rendered_depth_candidate &&
        rendered_depth_candidate_require_real_adjacency_ &&
        octpath_new.size(0) > 0) {
        auto unique_sorted_1d = [](const torch::Tensor& t) -> torch::Tensor {
            TORCH_CHECK(t.dim() == 1, "unique_sorted_1d expects a 1-D tensor");
            if (t.numel() <= 1) {
                return t.contiguous();
            }
            auto sort_pair = t.sort(/*dim=*/0);
            auto sorted = std::get<0>(sort_pair).contiguous();
            auto keep = torch::empty_like(sorted, torch::kBool);
            keep.index_put_({0}, true);
            auto neq =
                sorted.index({torch::indexing::Slice(1, torch::indexing::None)}) !=
                sorted.index({torch::indexing::Slice(torch::indexing::None, -1)});
            keep.index_put_({torch::indexing::Slice(1, torch::indexing::None)}, neq);
            auto keep_idx = torch::nonzero(keep).view({-1});
            return sorted.index_select(0, keep_idx).contiguous();
        };

        auto octpath_old_adj = py_->svm.attr("octpath").cast<torch::Tensor>().contiguous();
        auto octlevel_old_adj = py_->svm.attr("octlevel").cast<torch::Tensor>().contiguous();
        if (octpath_old_adj.numel() == 0) {
            std::cout << "[increasePcd/real_adjacency] no existing topology, nothing to attach to.\n";
            pending_insert_rendered_depth_candidate_ = false;
            pending_insert_rendered_depth_candidate_source_kind_ = 0;
            pending_insert_rendered_depth_candidate_as_real_protected_ = false;
            pending_artificial_insert_rr_entity_path_.clear();
            return;
        }

        auto bool_opts_old_adj = torch::TensorOptions().dtype(torch::kBool).device(dev);
        if (!is_artificial_voxel_.defined() || is_artificial_voxel_.size(0) != octpath_old_adj.size(0)) {
            is_artificial_voxel_ = torch::zeros({octpath_old_adj.size(0)}, bool_opts_old_adj);
        } else if (is_artificial_voxel_.device() != dev) {
            is_artificial_voxel_ = is_artificial_voxel_.to(dev);
        }

        auto real_mask_adj = (~is_artificial_voxel_.to(dev).to(torch::kBool)).contiguous();
        auto real_idx_adj = torch::nonzero(real_mask_adj).view({-1});
        if (real_idx_adj.numel() == 0) {
            std::cout << "[increasePcd/real_adjacency] no real voxels available for attachment.\n";
            pending_insert_rendered_depth_candidate_ = false;
            pending_insert_rendered_depth_candidate_source_kind_ = 0;
            pending_insert_rendered_depth_candidate_as_real_protected_ = false;
            pending_artificial_insert_rr_entity_path_.clear();
            return;
        }

        const int base_L = static_cast<int>(L_u[0].item<int8_t>());
        TORCH_CHECK(base_L >= 1 && base_L <= max_num_levels_,
                    "[increasePcd/real_adjacency] base level out of range: ", base_L);

        auto real_path_adj = octpath_old_adj.index_select(0, real_idx_adj).contiguous();
        auto real_level_adj = octlevel_old_adj.index_select(0, real_idx_adj)
                                  .view({-1})
                                  .to(torch::kInt64)
                                  .contiguous();
        auto valid_real_level_mask = (real_level_adj >= static_cast<int64_t>(base_L));
        auto valid_real_level_idx = torch::nonzero(valid_real_level_mask).view({-1});
        if (valid_real_level_idx.numel() == 0) {
            std::cout << "[increasePcd/real_adjacency] no real voxels at or finer than candidate level.\n";
            pending_insert_rendered_depth_candidate_ = false;
            pending_insert_rendered_depth_candidate_source_kind_ = 0;
            pending_insert_rendered_depth_candidate_as_real_protected_ = false;
            pending_artificial_insert_rr_entity_path_.clear();
            return;
        }
        real_path_adj = real_path_adj.index_select(0, valid_real_level_idx).contiguous();

        const int levels_below = std::max(0, max_num_levels_ - base_L);
        const int bits_to_clear = 3 * levels_below;
        const int64_t lower_mask = (bits_to_clear > 0) ? ((1LL << bits_to_clear) - 1LL) : 0LL;
        const int64_t keep_mask_ll = ~lower_mask;
        auto keep_mask = torch::full(
            {1},
            keep_mask_ll,
            torch::TensorOptions().dtype(torch::kInt64).device(dev));

        auto real_base_path =
            (real_path_adj.view({-1}).to(torch::kInt64) & keep_mask).contiguous();
        real_base_path = unique_sorted_1d(real_base_path);
        auto real_base_key =
            real_base_path.mul(256).add(
                torch::full_like(real_base_path, static_cast<int64_t>(base_L)));
        real_base_key = unique_sorted_1d(real_base_key);
        if (real_base_key.numel() == 0) {
            std::cout << "[increasePcd/real_adjacency] no real base cells available after ancestor projection.\n";
            pending_insert_rendered_depth_candidate_ = false;
            pending_insert_rendered_depth_candidate_source_kind_ = 0;
            pending_insert_rendered_depth_candidate_as_real_protected_ = false;
            pending_artificial_insert_rr_entity_path_.clear();
            return;
        }

        real_base_path = real_base_path.view({-1, 1}).contiguous();
        auto L_real_base = torch::full(
            {real_base_path.size(0), 1},
            static_cast<int64_t>(base_L),
            torch::TensorOptions().dtype(torch::kInt8).device(dev));
        auto real_base_ijk = svr_utils.attr("octpath_2_ijk")(
            py::cast(real_base_path),
            py::cast(L_real_base)).cast<torch::Tensor>().to(torch::kLong).contiguous();

        std::vector<int64_t> shift_vals;
        const int adj_radius = std::max(1, rendered_depth_candidate_adjacency_radius_cells_);
        for (int dx = -adj_radius; dx <= adj_radius; ++dx) {
            for (int dy = -adj_radius; dy <= adj_radius; ++dy) {
                for (int dz = -adj_radius; dz <= adj_radius; ++dz) {
                    if (dx == 0 && dy == 0 && dz == 0) {
                        continue;
                    }
                    shift_vals.push_back(static_cast<int64_t>(dx));
                    shift_vals.push_back(static_cast<int64_t>(dy));
                    shift_vals.push_back(static_cast<int64_t>(dz));
                }
            }
        }
        auto side_shift = torch::from_blob(
            shift_vals.data(),
            {static_cast<int64_t>(shift_vals.size() / 3), 3},
            torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
                              .clone()
                              .to(dev)
                              .contiguous();

        auto adj_ijk =
            (real_base_ijk.unsqueeze(1) + side_shift.unsqueeze(0)).contiguous().view({-1, 3});
        const int64_t grid_limit = (1LL << base_L);
        auto in_bounds =
            (adj_ijk >= 0).all(1) &
            (adj_ijk < grid_limit).all(1);
        auto adj_keep_idx = torch::nonzero(in_bounds).view({-1});
        if (adj_keep_idx.numel() == 0) {
            std::cout << "[increasePcd/real_adjacency] no in-bounds adjacent cells available.\n";
            pending_insert_rendered_depth_candidate_ = false;
            pending_insert_rendered_depth_candidate_source_kind_ = 0;
            pending_insert_rendered_depth_candidate_as_real_protected_ = false;
            pending_artificial_insert_rr_entity_path_.clear();
            return;
        }
        adj_ijk = adj_ijk.index_select(0, adj_keep_idx).contiguous();

        py::object uniq_adj = torch_mod.attr("unique")(
            py::cast(adj_ijk),
            py::arg("dim") = 0,
            py::arg("sorted") = true);
        adj_ijk = uniq_adj.cast<torch::Tensor>().contiguous();
        if (adj_ijk.dim() == 1) {
            adj_ijk = adj_ijk.view({-1, 3});
        }

        auto L_adj = torch::full(
            {adj_ijk.size(0), 1},
            static_cast<int64_t>(base_L),
            torch::TensorOptions().dtype(torch::kInt8).device(dev));
        auto adj_path = svr_utils.attr("ijk_2_octpath")(
            py::cast(adj_ijk),
            py::cast(L_adj)).cast<torch::Tensor>().contiguous();
        auto adj_key = adj_path.view({-1}).to(torch::kInt64).mul(256).add(L_adj.view({-1}).to(torch::kInt64));
        adj_key = unique_sorted_1d(adj_key);

        auto cand_key =
            octpath_new.view({-1}).to(torch::kInt64).mul(256).add(L_u.view({-1}).to(torch::kInt64));
        auto keep_adjacent = torch_mod.attr("isin")(
            py::cast(cand_key),
            py::cast(adj_key)).cast<torch::Tensor>().to(torch::kBool).contiguous();
        const int64_t kept_adjacent = keep_adjacent.sum().item<int64_t>();
        std::cout << "[increasePcd/real_adjacency] kept " << kept_adjacent
                  << "/" << octpath_new.size(0)
                  << " rendered-depth candidates"
                  << " real_base_cells=" << real_base_path.size(0)
                  << " radius_cells=" << adj_radius
                  << "\n";
        if (kept_adjacent == 0) {
            std::cout << "[increasePcd/filter] rendered-depth candidates rejected by real-adjacency gate.\n";
            pending_insert_rendered_depth_candidate_ = false;
            pending_insert_rendered_depth_candidate_source_kind_ = 0;
            pending_insert_rendered_depth_candidate_as_real_protected_ = false;
            pending_artificial_insert_rr_entity_path_.clear();
            return;
        }

        auto keep_adjacent_idx = torch::nonzero(keep_adjacent).view({-1});
        octpath_new = octpath_new.index_select(0, keep_adjacent_idx).contiguous();
        L_u = L_u.index_select(0, keep_adjacent_idx).contiguous();
        ijk_u = ijk_u.index_select(0, keep_adjacent_idx).contiguous();
        rgb_u = rgb_u.index_select(0, keep_adjacent_idx).contiguous();
        Nu = octpath_new.size(0);
    }

    // ── 5) Dedup against existing voxels (across-batch) ─────────────────────
    auto octpath_old  = py_->svm.attr("octpath").cast<torch::Tensor>().contiguous();                    // [No,1] int64
    auto octlevel_old = py_->svm.attr("octlevel").cast<torch::Tensor>().contiguous();                   // [No,1] int8
    const int64_t old_voxel_count = octpath_old.size(0);

    // Packed 1D key: (octpath<<8) | level
    auto key_new = octpath_new.view({-1}).to(torch::kInt64)
                    .mul(256)
                    .add(L_u.view({-1}).to(torch::kInt64));
    auto key_old_all = octpath_old.view({-1}).to(torch::kInt64)
                    .mul(256)
                    .add(octlevel_old.view({-1}).to(torch::kInt64));
    auto bool_opts_old = torch::TensorOptions().dtype(torch::kBool).device(dev);
    auto i32_opts_old = torch::TensorOptions().dtype(torch::kInt32).device(dev);
    if (!rendered_depth_candidate_voxel_.defined() ||
        rendered_depth_candidate_voxel_.size(0) != octpath_old.size(0)) {
        rendered_depth_candidate_voxel_ = torch::zeros({octpath_old.size(0)}, bool_opts_old);
    } else if (rendered_depth_candidate_voxel_.device() != dev) {
        rendered_depth_candidate_voxel_ = rendered_depth_candidate_voxel_.to(dev);
    }
    if (!rendered_depth_candidate_support_count_.defined() ||
        rendered_depth_candidate_support_count_.size(0) != octpath_old.size(0)) {
        rendered_depth_candidate_support_count_ = torch::zeros({octpath_old.size(0)}, i32_opts_old);
    } else if (rendered_depth_candidate_support_count_.device() != dev) {
        rendered_depth_candidate_support_count_ = rendered_depth_candidate_support_count_.to(dev);
    }
    if (!rendered_depth_candidate_last_seen_kf_.defined() ||
        rendered_depth_candidate_last_seen_kf_.size(0) != octpath_old.size(0)) {
        rendered_depth_candidate_last_seen_kf_ =
            torch::full({octpath_old.size(0)}, static_cast<int32_t>(-1), i32_opts_old);
    } else if (rendered_depth_candidate_last_seen_kf_.device() != dev) {
        rendered_depth_candidate_last_seen_kf_ = rendered_depth_candidate_last_seen_kf_.to(dev);
    }
    if (!rendered_depth_candidate_source_kind_.defined() ||
        rendered_depth_candidate_source_kind_.size(0) != octpath_old.size(0)) {
        rendered_depth_candidate_source_kind_ = torch::zeros({octpath_old.size(0)}, i32_opts_old);
    } else if (rendered_depth_candidate_source_kind_.device() != dev) {
        rendered_depth_candidate_source_kind_ = rendered_depth_candidate_source_kind_.to(dev);
    }
    torch::Tensor art_key_before = torch::empty({0}, key_old_all.options());
    if (octpath_old.numel() > 0 &&
        is_artificial_voxel_.defined() && is_artificial_voxel_.size(0) == octpath_old.size(0)) {
        auto art_before_mask = is_artificial_voxel_.to(dev).to(torch::kBool).contiguous();
        if (is_promoted_artificial_voxel_.defined() &&
            is_promoted_artificial_voxel_.size(0) == octpath_old.size(0)) {
            auto promoted_before_mask = is_promoted_artificial_voxel_.to(dev).to(torch::kBool).contiguous();
            art_before_mask = art_before_mask & (~promoted_before_mask);
        }
        auto idx_art_before = torch::nonzero(art_before_mask).view({-1});
        if (idx_art_before.numel() > 0) {
            art_key_before = key_old_all.index_select(0, idx_art_before).contiguous();
        }
    }
    if (last_artificial_iter_ != iteration) {
        last_artificial_iter_ = iteration;
        art_key_before_iter_ = art_key_before.defined()
            ? art_key_before.clone()
            : torch::empty({0}, key_old_all.options());
    } else if (!art_key_before_iter_.defined()) {
        art_key_before_iter_ = torch::empty({0}, key_old_all.options());
    }

    int64_t promoted_artificials_count = 0;
    auto long_opts_old = torch::TensorOptions().dtype(torch::kLong).device(dev);
    torch::Tensor promote_idx_deferred = torch::empty({0}, long_opts_old);
    torch::Tensor support_idx_deferred = torch::empty({0}, long_opts_old);
    torch::Tensor old_art_mask_for_promotion = torch::zeros({octpath_old.size(0)}, bool_opts_old);
    auto old_rendered_depth_candidate_mask =
        rendered_depth_candidate_voxel_.to(dev).to(torch::kBool).contiguous();
    if (octpath_old.numel() > 0) {
        if (!is_artificial_voxel_.defined() || is_artificial_voxel_.size(0) != octpath_old.size(0)) {
            is_artificial_voxel_ = torch::zeros({octpath_old.size(0)}, bool_opts_old);
        } else if (is_artificial_voxel_.device() != dev) {
            is_artificial_voxel_ = is_artificial_voxel_.to(dev);
        }
        old_art_mask_for_promotion = is_artificial_voxel_.to(torch::kBool).contiguous();
        if (is_promoted_artificial_voxel_.defined() &&
            is_promoted_artificial_voxel_.size(0) == octpath_old.size(0)) {
            auto promoted_before_mask = is_promoted_artificial_voxel_.to(dev).to(torch::kBool).contiguous();
            old_art_mask_for_promotion = old_art_mask_for_promotion & (~promoted_before_mask);
        }
        // Promote artificials regardless of when they were created:
        // only require "currently artificial and not already promoted".
        if (!enable_artificial_promotion_) {
            old_art_mask_for_promotion = torch::zeros_like(old_art_mask_for_promotion);
        }
        if (insert_rendered_depth_candidate) {
            old_art_mask_for_promotion = torch::zeros_like(old_art_mask_for_promotion);
        }
    }
    torch::Tensor new_mask;
    if (octpath_old.numel() == 0) {
        std::cout << "octpath_old is empty => all new.\n";
        new_mask = torch::ones({Nu}, torch::dtype(torch::kBool).device(dev));
    } else {
        py::object isin_py = torch_mod.attr("isin")(py::cast(key_new), py::cast(key_old_all));
        auto is_dup = isin_py.cast<torch::Tensor>().to(torch::kBool);                                    // [Nu]
        new_mask = ~is_dup;

        // If a real observation hits an existing artificial voxel, defer promotion to
        // after real insertion in this call. For rendered-depth candidate insertion,
        // duplicates instead add support to existing candidates.
        if (is_dup.any().item<bool>()) {
            auto dup_idx_new = torch::nonzero(is_dup).view({-1});
            if (dup_idx_new.numel() > 0) {
                auto dup_key_new = key_new.index_select(0, dup_idx_new).contiguous();

                auto sort_pair = key_old_all.sort(/*dim=*/0);
                auto key_old_sorted = std::get<0>(sort_pair).contiguous();
                auto old_perm = std::get<1>(sort_pair).to(torch::kLong).contiguous();

                auto pos = torch_mod.attr("searchsorted")(
                    py::cast(key_old_sorted), py::cast(dup_key_new),
                    py::arg("right") = false).cast<torch::Tensor>()
                    .to(torch::kLong).contiguous();

                auto in_range = (pos >= 0) & (pos < key_old_sorted.size(0));
                if (in_range.any().item<bool>()) {
                    auto in_idx = torch::nonzero(in_range).view({-1});
                    auto pos_in = pos.index_select(0, in_idx).contiguous();
                    auto key_in = dup_key_new.index_select(0, in_idx).contiguous();
                    auto key_hit = key_old_sorted.index_select(0, pos_in).contiguous();
                    auto exact = (key_hit == key_in);

                    if (exact.any().item<bool>()) {
                        auto ex_idx = torch::nonzero(exact).view({-1});
                        auto pos_ex = pos_in.index_select(0, ex_idx).contiguous();
                        auto old_idx = old_perm.index_select(0, pos_ex).contiguous();
                        if (insert_rendered_depth_candidate) {
                            auto can_support = old_rendered_depth_candidate_mask
                                .index_select(0, old_idx).to(torch::kBool).contiguous();
                            if (can_support.any().item<bool>()) {
                                auto support_rel_idx = torch::nonzero(can_support).view({-1});
                                auto support_idx = old_idx.index_select(0, support_rel_idx).contiguous();
                                support_idx_deferred = torch::cat(
                                    {support_idx_deferred, support_idx.to(torch::kLong)}, 0).contiguous();
                            }
                        } else {
                            auto can_promote = old_art_mask_for_promotion
                                .index_select(0, old_idx).to(torch::kBool).contiguous();
                            if (can_promote.any().item<bool>()) {
                                auto art_idx = torch::nonzero(can_promote).view({-1});
                                auto promote_idx = old_idx.index_select(0, art_idx).contiguous();
                                promote_idx_deferred = torch::cat(
                                    {promote_idx_deferred, promote_idx.to(torch::kLong)}, 0).contiguous();
                            }
                        }
                    }
                }
            }
        }
    }
    // --- Prevent re-adding a parent at L=base_L when children (L > base_L) already exist ---
    {
        if (octpath_old.numel() > 0) {
            auto dev = octpath_old.device();
            const int MAX_L  = max_num_levels_;            // must match svraster MAX_NUM_LEVELS
            const int base_L = static_cast<int>(octlevel_);// the level used by createFromPcd()

            TORCH_CHECK(base_L >= 1 && base_L <= MAX_L,
                        "[increasePcd] base_L (octlevel_) out of range: ", base_L,
                        " with MAX_L=", MAX_L);

            // Existing voxels whose level is strictly finer than base_L
            auto Lold_i64    = octlevel_old.view({-1}).to(torch::kInt64);     // [No]
            auto has_children= (Lold_i64 > base_L);                           // [No] bool

            if (has_children.any().item<bool>()) {

                // Mask that clears all octant bits *below* base_L.
                // Bits per level = 3; for a node at level L, its octant sits at shift = 3*(MAX_L - L).
                // To keep bits down to base_L (inclusive), clear the lowest 3*(MAX_L - base_L) bits.
                const int levels_below  = std::max(0, MAX_L - base_L);
                const int bits_to_clear = 3 * levels_below;
                long long lower_mask    = (bits_to_clear > 0) ? ((1LL << bits_to_clear) - 1LL) : 0LL;
                long long keep_mask_ll  = ~lower_mask;

                auto keep_mask = torch::full(
                    {1},
                    static_cast<int64_t>(keep_mask_ll),
                    torch::TensorOptions().dtype(torch::kInt64).device(dev)
                );

                // Compute the ancestor-at-base_L octpaths for those finer voxels
                auto op_old_i64  = octpath_old.view({-1}).to(torch::kInt64);   // [No]
                auto op_anc_base = (op_old_i64 & keep_mask);                   // [No]

                // Keep only rows where L_old > base_L
                auto sel = torch::nonzero(has_children).view({-1});            // [K]
                op_anc_base = op_anc_base.index_select(0, sel);                // [K]

                // Build the ancestor keys at base_L: ((octpath_anc_base<<8) | base_L)
                auto key_children_as_parent = op_anc_base.mul(256)
                                            .add(torch::full_like(op_anc_base,
                                                                static_cast<int64_t>(base_L)));

                // Unique + sorted to make isin faster
                auto unique_sorted_1d = [](const at::Tensor& t)->at::Tensor {
                    TORCH_CHECK(t.dim() == 1, "unique_sorted_1d expects a 1-D tensor");
                    if (t.numel() <= 1) return t.contiguous();
                    auto sort_res = t.sort(/*dim=*/0);
                    auto sorted   = std::get<0>(sort_res);
                    using torch::indexing::Slice;
                    auto keep = torch::empty_like(sorted, torch::kBool);
                    keep.index_put_({0}, true);
                    auto neq = sorted.index({Slice(1, torch::indexing::None)})
                            != sorted.index({Slice(torch::indexing::None, -1)});
                    keep.index_put_({Slice(1, torch::indexing::None)}, neq);
                    auto idx = torch::nonzero(keep).view({-1});
                    return sorted.index_select(0, idx).contiguous();
                };
                key_children_as_parent = unique_sorted_1d(key_children_as_parent);

                // If a candidate NEW (octpath, base_L) matches any ancestor of an existing finer voxel,
                // then inserting that parent would collide with existing children later.
                py::object isin_py = torch_mod.attr("isin")(
                    py::cast(key_new), py::cast(key_children_as_parent));
                auto would_collide_parent = isin_py.cast<torch::Tensor>().to(torch::kBool);  // [Nu]

                // Promotion path:
                // Real observations arriving at parent level should promote overlapped child artificials to real,
                // otherwise artificial provenance keeps expanding after subdivision.
                // Rendered-depth candidate insertion uses the same overlap detection to
                // add support to subdivided descendants instead of promoting them.
                if (would_collide_parent.any().item<bool>()) {
                    auto collide_idx_new = torch::nonzero(would_collide_parent).view({-1});
                    auto collide_parent_keys = key_new.index_select(0, collide_idx_new).contiguous();
                    collide_parent_keys = unique_sorted_1d(collide_parent_keys);

                    auto key_child_rows = op_anc_base.mul(256)
                        .add(torch::full_like(op_anc_base, static_cast<int64_t>(base_L)));
                    auto child_under_collide = torch_mod.attr("isin")(
                        py::cast(key_child_rows), py::cast(collide_parent_keys))
                        .cast<torch::Tensor>().to(torch::kBool);

                    if (child_under_collide.any().item<bool>()) {
                        auto child_rel_idx = torch::nonzero(child_under_collide).view({-1});
                        auto old_idx_under = sel.index_select(0, child_rel_idx).contiguous();
                        if (insert_rendered_depth_candidate) {
                            auto can_support = old_rendered_depth_candidate_mask
                                .index_select(0, old_idx_under).to(torch::kBool).contiguous();
                            if (can_support.any().item<bool>()) {
                                auto support_rel_idx = torch::nonzero(can_support).view({-1});
                                auto support_idx = old_idx_under.index_select(0, support_rel_idx).contiguous();
                                support_idx_deferred = torch::cat(
                                    {support_idx_deferred, support_idx.to(torch::kLong)}, 0).contiguous();
                            }
                        } else {
                            auto can_promote = old_art_mask_for_promotion
                                .index_select(0, old_idx_under).to(torch::kBool).contiguous();
                            if (can_promote.any().item<bool>()) {
                                auto art_rel_idx = torch::nonzero(can_promote).view({-1});
                                auto promote_idx = old_idx_under.index_select(0, art_rel_idx).contiguous();
                                promote_idx_deferred = torch::cat(
                                    {promote_idx_deferred, promote_idx.to(torch::kLong)}, 0).contiguous();
                            }
                        }
                    }
                    new_mask = new_mask & (~would_collide_parent);
                }
            }
        }
    }

    if (promote_idx_deferred.numel() > 1) {
        auto sorted = std::get<0>(promote_idx_deferred.sort(/*dim=*/0));
        auto keep = torch::empty_like(sorted, torch::kBool);
        keep.index_put_({0}, true);
        auto neq = sorted.index({torch::indexing::Slice(1, torch::indexing::None)})
                != sorted.index({torch::indexing::Slice(torch::indexing::None, -1)});
        keep.index_put_({torch::indexing::Slice(1, torch::indexing::None)}, neq);
        auto keep_idx = torch::nonzero(keep).view({-1});
        promote_idx_deferred = sorted.index_select(0, keep_idx).contiguous();
    }
    if (support_idx_deferred.numel() > 1) {
        auto sorted = std::get<0>(support_idx_deferred.sort(/*dim=*/0));
        auto keep = torch::empty_like(sorted, torch::kBool);
        keep.index_put_({0}, true);
        auto neq = sorted.index({torch::indexing::Slice(1, torch::indexing::None)})
                != sorted.index({torch::indexing::Slice(torch::indexing::None, -1)});
        keep.index_put_({torch::indexing::Slice(1, torch::indexing::None)}, neq);
        auto keep_idx = torch::nonzero(keep).view({-1});
        support_idx_deferred = sorted.index_select(0, keep_idx).contiguous();
    }
    const int64_t pending_promotions = promote_idx_deferred.numel();
    const int64_t pending_support_updates = support_idx_deferred.numel();
    const int64_t new_voxel_candidates = new_mask.sum().item<int64_t>();
    const int64_t duplicate_existing_voxels =
        unique_voxel_candidates_after_insert_filter - new_voxel_candidates;
    last_increase_pcd_stats_.duplicate_existing_voxels = duplicate_existing_voxels;
    last_increase_pcd_stats_.new_voxels = new_voxel_candidates;
    last_increase_pcd_stats_.pending_promotions = pending_promotions;
    last_increase_pcd_stats_.pending_support_updates = pending_support_updates;
    // std::cout << "[increasePcd/quantize] raw_points_in=" << raw_points_in
    //           << " points_after_far_filter=" << points_after_far_filter
    //           << " unique_voxel_candidates=" << unique_voxel_candidates_before_insert_filter
    //           << " kept_after_insert_filter=" << unique_voxel_candidates_after_insert_filter
    //           << " duplicate_existing_voxels=" << duplicate_existing_voxels
    //           << " new_voxels=" << new_voxel_candidates
    //           << " pending_promotions=" << pending_promotions
    //           << " pending_support_updates=" << pending_support_updates
    //           << std::endl;
    if (!new_mask.any().item<bool>()) {
        if (pending_promotions == 0 && pending_support_updates == 0) {
            std::cout << "[increasePcd] No new voxels (all duplicates). Nothing appended.\n";
            pending_insert_rendered_depth_candidate_ = false;
            pending_insert_rendered_depth_candidate_source_kind_ = 0;
            pending_insert_rendered_depth_candidate_as_real_protected_ = false;
            pending_artificial_insert_rr_entity_path_.clear();
            return;
        }
        std::cout << "[increasePcd] No new voxels, continuing after deferred updates.\n";
    }
    auto sel = torch::nonzero(new_mask).view({-1});                                                     // [Nk]
    auto ijk_add     = ijk_u.index_select(0, sel);                                                      // [Nk,3]
    auto octpath_add = octpath_new.index_select(0, sel);                                                // [Nk,1]
    auto L_add       = L_u.index_select(0, sel);                                                         // [Nk,1]
    auto rgb_add     = rgb_u.index_select(0, sel);                                                       // [Nk,3]
    const int Nk = sel.size(0);

    int64_t Nm_added = 0;  // count of artificial voxels added later
    // ── 6) Append topology (old preserved) ──────────────────────────────────
    py_->svm.attr("octpath")  = torch::cat({octpath_old,  octpath_add}, 0).contiguous();
    py_->svm.attr("octlevel") = torch::cat({octlevel_old, L_add},       0).contiguous();
    {
        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto i32_opts  = torch::TensorOptions().dtype(torch::kInt32).device(dev);
        if (!is_artificial_voxel_.defined() || is_artificial_voxel_.size(0) != octpath_old.size(0)) {
            is_artificial_voxel_ = torch::zeros({octpath_old.size(0)}, bool_opts);
        } else if (is_artificial_voxel_.device() != octpath_old.device()) {
            is_artificial_voxel_ = is_artificial_voxel_.to(octpath_old.device());
        }
        if (!is_orb_voxel_.defined() || is_orb_voxel_.size(0) != octpath_old.size(0)) {
            is_orb_voxel_ = torch::zeros({octpath_old.size(0)}, bool_opts);
        } else if (is_orb_voxel_.device() != octpath_old.device()) {
            is_orb_voxel_ = is_orb_voxel_.to(octpath_old.device());
        }
        if (!is_inactive_geo_voxel_.defined() || is_inactive_geo_voxel_.size(0) != octpath_old.size(0)) {
            is_inactive_geo_voxel_ = torch::zeros({octpath_old.size(0)}, bool_opts);
        } else if (is_inactive_geo_voxel_.device() != octpath_old.device()) {
            is_inactive_geo_voxel_ = is_inactive_geo_voxel_.to(octpath_old.device());
        }
        if (!is_rgbd_fill_render_holes_voxel_.defined() ||
            is_rgbd_fill_render_holes_voxel_.size(0) != octpath_old.size(0)) {
            is_rgbd_fill_render_holes_voxel_ = torch::zeros({octpath_old.size(0)}, bool_opts);
        } else if (is_rgbd_fill_render_holes_voxel_.device() != octpath_old.device()) {
            is_rgbd_fill_render_holes_voxel_ =
                is_rgbd_fill_render_holes_voxel_.to(octpath_old.device());
        }
        if (!is_depthanything_fill_holes_voxel_.defined() ||
            is_depthanything_fill_holes_voxel_.size(0) != octpath_old.size(0)) {
            is_depthanything_fill_holes_voxel_ = torch::zeros({octpath_old.size(0)}, bool_opts);
        } else if (is_depthanything_fill_holes_voxel_.device() != octpath_old.device()) {
            is_depthanything_fill_holes_voxel_ =
                is_depthanything_fill_holes_voxel_.to(octpath_old.device());
        }
        if (!is_promoted_artificial_voxel_.defined() || is_promoted_artificial_voxel_.size(0) != octpath_old.size(0)) {
            is_promoted_artificial_voxel_ = torch::zeros({octpath_old.size(0)}, bool_opts);
        } else if (is_promoted_artificial_voxel_.device() != octpath_old.device()) {
            is_promoted_artificial_voxel_ = is_promoted_artificial_voxel_.to(octpath_old.device());
        }
        if (!exist_since_iter_.defined() || exist_since_iter_.size(0) != octpath_old.size(0)) {
            exist_since_iter_ = torch::zeros({octpath_old.size(0)}, i32_opts);
        } else if (exist_since_iter_.device() != octpath_old.device()) {
            exist_since_iter_ = exist_since_iter_.to(octpath_old.device());
        }
        if (!exist_since_kf_.defined() || exist_since_kf_.size(0) != octpath_old.size(0)) {
            exist_since_kf_ = torch::full({octpath_old.size(0)}, static_cast<int32_t>(-1), i32_opts);
        } else if (exist_since_kf_.device() != octpath_old.device()) {
            exist_since_kf_ = exist_since_kf_.to(octpath_old.device());
        }
        if (!geometrically_unstable_voxel_.defined() || geometrically_unstable_voxel_.size(0) != octpath_old.size(0)) {
            geometrically_unstable_voxel_ = torch::zeros({octpath_old.size(0)}, bool_opts);
        } else if (geometrically_unstable_voxel_.device() != octpath_old.device()) {
            geometrically_unstable_voxel_ = geometrically_unstable_voxel_.to(octpath_old.device());
        }
        if (!rendered_depth_candidate_voxel_.defined() ||
            rendered_depth_candidate_voxel_.size(0) != octpath_old.size(0)) {
            rendered_depth_candidate_voxel_ = torch::zeros({octpath_old.size(0)}, bool_opts);
        } else if (rendered_depth_candidate_voxel_.device() != octpath_old.device()) {
            rendered_depth_candidate_voxel_ = rendered_depth_candidate_voxel_.to(octpath_old.device());
        }
        if (!rendered_depth_candidate_support_count_.defined() ||
            rendered_depth_candidate_support_count_.size(0) != octpath_old.size(0)) {
            rendered_depth_candidate_support_count_ = torch::zeros({octpath_old.size(0)}, i32_opts);
        } else if (rendered_depth_candidate_support_count_.device() != octpath_old.device()) {
            rendered_depth_candidate_support_count_ = rendered_depth_candidate_support_count_.to(octpath_old.device());
        }
        if (!rendered_depth_candidate_last_seen_kf_.defined() ||
            rendered_depth_candidate_last_seen_kf_.size(0) != octpath_old.size(0)) {
            rendered_depth_candidate_last_seen_kf_ =
                torch::full({octpath_old.size(0)}, static_cast<int32_t>(-1), i32_opts);
        } else if (rendered_depth_candidate_last_seen_kf_.device() != octpath_old.device()) {
            rendered_depth_candidate_last_seen_kf_ = rendered_depth_candidate_last_seen_kf_.to(octpath_old.device());
        }
        if (!rendered_depth_candidate_source_kind_.defined() ||
            rendered_depth_candidate_source_kind_.size(0) != octpath_old.size(0)) {
            rendered_depth_candidate_source_kind_ = torch::zeros({octpath_old.size(0)}, i32_opts);
        } else if (rendered_depth_candidate_source_kind_.device() != octpath_old.device()) {
            rendered_depth_candidate_source_kind_ = rendered_depth_candidate_source_kind_.to(octpath_old.device());
        }
        if (Nk > 0) {
            const bool add_as_artificial_candidate =
                insert_rendered_depth_candidate && !insert_rendered_depth_candidate_as_real_protected;
            auto artificial_add_flag = torch::full(
                {Nk}, add_as_artificial_candidate, bool_opts);
            const bool add_as_orb =
                !insert_rendered_depth_candidate &&
                pending_real_insert_rr_entity_path_ == "world/orb/voxels_created";
            auto orb_add_flag = torch::full({Nk}, add_as_orb, bool_opts);
            const bool add_as_inactive_geo =
                !insert_rendered_depth_candidate &&
                pending_real_insert_rr_entity_path_ == "world/voxels_inactive_geo_densify/created";
            auto inactive_geo_add_flag = torch::full({Nk}, add_as_inactive_geo, bool_opts);
            const bool add_as_rgbd_fill_render_holes =
                !insert_rendered_depth_candidate &&
                pending_real_insert_rr_entity_path_ == "world/rgbd_fill_render_holes/created";
            auto rgbd_fill_render_holes_add_flag =
                torch::full({Nk}, add_as_rgbd_fill_render_holes, bool_opts);
            const bool add_as_depthanything_fill_holes =
                !add_as_artificial_candidate &&
                pending_real_insert_rr_entity_path_ == "world/mono_prior_fill_holes/created";
            auto depthanything_fill_holes_add_flag =
                torch::full({Nk}, add_as_depthanything_fill_holes, bool_opts);
            auto promoted_add_flag = torch::zeros({Nk}, bool_opts);
            auto exist_since_add = torch::full(
                {Nk}, static_cast<int32_t>(iteration), i32_opts);
            auto exist_since_kf_add = torch::full(
                {Nk}, current_kf_count, i32_opts);
            auto unstable_add_flag = torch::zeros({Nk}, bool_opts);
            auto rendered_depth_add_flag = torch::full(
                {Nk}, add_as_artificial_candidate, bool_opts);
            auto rendered_depth_support_add = torch::full(
                {Nk},
                static_cast<int32_t>(add_as_artificial_candidate ? 1 : 0),
                i32_opts);
            auto rendered_depth_last_seen_add = torch::full(
                {Nk},
                add_as_artificial_candidate ? current_kf_count : static_cast<int32_t>(-1),
                i32_opts);
            auto rendered_depth_source_add = torch::full(
                {Nk},
                static_cast<int32_t>(pending_insert_rendered_depth_candidate_source_kind_ > 0
                    ? pending_insert_rendered_depth_candidate_source_kind_
                    : 0),
                i32_opts);
            is_artificial_voxel_ = torch::cat({is_artificial_voxel_, artificial_add_flag}, 0).contiguous();
            is_orb_voxel_ = torch::cat({is_orb_voxel_, orb_add_flag}, 0).contiguous();
            is_inactive_geo_voxel_ = torch::cat({is_inactive_geo_voxel_, inactive_geo_add_flag}, 0).contiguous();
            is_rgbd_fill_render_holes_voxel_ =
                torch::cat({is_rgbd_fill_render_holes_voxel_, rgbd_fill_render_holes_add_flag}, 0).contiguous();
            is_depthanything_fill_holes_voxel_ =
                torch::cat({is_depthanything_fill_holes_voxel_, depthanything_fill_holes_add_flag}, 0).contiguous();
            is_promoted_artificial_voxel_ = torch::cat({is_promoted_artificial_voxel_, promoted_add_flag}, 0).contiguous();
            exist_since_iter_ = torch::cat({exist_since_iter_, exist_since_add}, 0).contiguous();
            exist_since_kf_ = torch::cat({exist_since_kf_, exist_since_kf_add}, 0).contiguous();
            geometrically_unstable_voxel_ = torch::cat({geometrically_unstable_voxel_, unstable_add_flag}, 0).contiguous();
            rendered_depth_candidate_voxel_ =
                torch::cat({rendered_depth_candidate_voxel_, rendered_depth_add_flag}, 0).contiguous();
            rendered_depth_candidate_support_count_ =
                torch::cat({rendered_depth_candidate_support_count_, rendered_depth_support_add}, 0).contiguous();
            rendered_depth_candidate_last_seen_kf_ =
                torch::cat({rendered_depth_candidate_last_seen_kf_, rendered_depth_last_seen_add}, 0).contiguous();
            rendered_depth_candidate_source_kind_ =
                torch::cat({rendered_depth_candidate_source_kind_, rendered_depth_source_add}, 0).contiguous();
        }
    }

    if (insert_rendered_depth_candidate && pending_support_updates > 0) {
        auto support_idx = support_idx_deferred
            .to(rendered_depth_candidate_support_count_.device())
            .to(torch::kLong)
            .contiguous();
        auto prev = rendered_depth_candidate_support_count_.index_select(0, support_idx);
        rendered_depth_candidate_support_count_.index_put_({support_idx}, prev + 1);
        rendered_depth_candidate_last_seen_kf_.index_put_(
            {support_idx},
            torch::full(
                {support_idx.size(0)},
                current_kf_count,
                torch::TensorOptions().dtype(torch::kInt32).device(rendered_depth_candidate_last_seen_kf_.device())));
    }

    // Apply deferred promotions after real insertion in this call.
    if (enable_artificial_promotion_ && pending_promotions > 0) {
        auto promote_idx = promote_idx_deferred.to(is_artificial_voxel_.device()).to(torch::kLong).contiguous();
        is_artificial_voxel_.index_put_({promote_idx}, false);
        is_promoted_artificial_voxel_.index_put_({promote_idx}, true);
        if (rendered_depth_candidate_voxel_.defined() &&
            rendered_depth_candidate_voxel_.size(0) == is_artificial_voxel_.size(0)) {
            rendered_depth_candidate_voxel_.index_put_({promote_idx}, false);
        }
        // Preserve insertion provenance after promotion so live debug views can
        // still identify voxels that originated from rendered-hole-fill.
        promoted_artificials_count = promote_idx.size(0);
        total_promoted_artificial_voxels_ += promoted_artificials_count;
    }

    // ── 7) Append learnables for new rows ───────────────────────────────────
    // _subdiv_p
    auto subdiv_old = py_->svm.attr("_subdiv_p").cast<torch::Tensor>();
    auto subdiv_add = torch::ones({Nk,1}, torch::dtype(torch::kFloat32).device(dev));
    py_->svm.attr("_subdiv_p") = torch::cat({subdiv_old, subdiv_add}, 0)
                                .contiguous().detach().requires_grad_();

    // _sh0 from fused rgb
    torch::Tensor sh0_add = torch::empty({0, 3}, torch::dtype(torch::kFloat32).device(dev));
    if (Nk > 0) {
        py::object sh0_add_py = act_mod.attr("rgb2shzero")(py::cast(rgb_add.contiguous())); // [Nk,3]
        sh0_add = sh0_add_py.cast<torch::Tensor>().contiguous();
    }
    auto sh0_old = py_->svm.attr("_sh0").cast<torch::Tensor>();
    py_->svm.attr("_sh0")      = torch::cat({sh0_old,    sh0_add   }, 0)
                                    .contiguous().detach().requires_grad_();

    // _shs zeros
    const int n_sh_rest = (max_sh_degree_ + 1)*(max_sh_degree_ + 1) - 1;
    auto shs_old = py_->svm.attr("_shs").cast<torch::Tensor>();
    auto shs_add = torch::zeros({Nk, n_sh_rest, 3}, torch::dtype(torch::kFloat32).device(dev));
    py_->svm.attr("_shs")      = torch::cat({shs_old,    shs_add   }, 0)
                                .contiguous().detach().requires_grad_();

    // ── 8) Rebuild grid links; grow _geo_grid_pts only if grid expanded ─────
    // Trigger SVProperties recompute via property access
    torch::Tensor grid_pts_key_new = py_->svm.attr("grid_pts_key").cast<torch::Tensor>();               // [M,3] int64
    const int64_t M_prev = grid_pts_key_.defined() ? grid_pts_key_.size(0) : 0;
    const int64_t M_curr = grid_pts_key_new.size(0);

    // Prepare grow only when needed (group 0)
    torch::Tensor grow; // undefined by default
    if (M_curr > M_prev) {
        grow = makeGeoGridInitRows_(
            grid_pts_key_new,
            M_prev,
            M_curr,
            /*default_value=*/-10.0f);
    }
    // ── 9) Append rows to optimizer param groups (and keep py svm in sync) ──
    // Important: only call for non-empty additions.
    if (grow.defined() && grow.size(0) > 0) {
        appendGroup_(/*group_idx=*/0, /*add_rows=*/grow, /*svm_field_name=*/"_geo_grid_pts",
                    &this->_geo_grid_pts_);
    }
    if (Nk > 0) {
        appendGroup_(/*group_idx=*/1, /*add_rows=*/sh0_add, /*svm_field_name=*/"_sh0",
                    &this->sh0_);
        appendGroup_(/*group_idx=*/2, /*add_rows=*/shs_add, /*svm_field_name=*/"_shs",
                    &this->shs_);
    }

    auto run_local_frontier_fill = [&]() -> int64_t {
        if (cams.empty()) {
            std::cout << "[increasePcd] local_frontier_fill: no cameras, skip.\n";
            return 0;
        }

        auto devL = scene_min_t_.device();
        const auto long_opts = torch::TensorOptions().dtype(torch::kLong).device(devL);
        const auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(devL);

        const int base_L = static_cast<int>(octlevel_);
        const int MAX_L  = max_num_levels_;
        TORCH_CHECK(base_L >= 1 && base_L <= MAX_L,
                    "[local_frontier_fill] base_L out of range: ", base_L);

        const int levels_below  = std::max(0, MAX_L - base_L);
        const int bits_to_clear = 3 * levels_below;
        long long lower_mask    = (bits_to_clear > 0) ? ((1LL << bits_to_clear) - 1LL) : 0LL;
        long long keep_mask_ll  = ~lower_mask;
        auto keep_mask = torch::full(
            {1},
            static_cast<int64_t>(keep_mask_ll),
            torch::TensorOptions().dtype(torch::kInt64).device(devL)
        );

        auto octpath_occ  = py_->svm.attr("octpath").cast<torch::Tensor>().contiguous();
        auto octlevel_occ = py_->svm.attr("octlevel").cast<torch::Tensor>().contiguous();
        if (octpath_occ.numel() == 0) {
            return 0;
        }
        auto occ_path_i64  = octpath_occ.view({-1}).to(torch::kInt64).contiguous();
        auto occ_level_i64 = octlevel_occ.view({-1}).to(torch::kInt64).contiguous();
        auto key_occ_exact = occ_path_i64.mul(256).add(occ_level_i64);

        auto make_keep_mask_for_level = [&](int level) -> torch::Tensor {
            const int levels_below_l  = std::max(0, MAX_L - level);
            const int bits_to_clear_l = 3 * levels_below_l;
            long long lower_mask_l    = (bits_to_clear_l > 0) ? ((1LL << bits_to_clear_l) - 1LL) : 0LL;
            long long keep_mask_l_ll  = ~lower_mask_l;
            return torch::full(
                {1},
                static_cast<int64_t>(keep_mask_l_ll),
                torch::TensorOptions().dtype(torch::kInt64).device(devL));
        };

        auto unique_sorted_1d = [&](const torch::Tensor& t_in) -> torch::Tensor {
            auto t = t_in.contiguous().view({-1});
            if (t.numel() <= 1) return t;
            auto sorted = std::get<0>(t.sort(/*dim=*/0));
            auto keep = torch::empty_like(sorted, torch::kBool);
            keep.index_put_({0}, true);
            auto neq = sorted.index({torch::indexing::Slice(1, torch::indexing::None)})
                    != sorted.index({torch::indexing::Slice(torch::indexing::None, -1)});
            keep.index_put_({torch::indexing::Slice(1, torch::indexing::None)}, neq);
            auto idx = torch::nonzero(keep).view({-1});
            return sorted.index_select(0, idx).contiguous();
        };

        // A candidate at level L is considered occupied if:
        // 1) exact node exists at L, or
        // 2) an ancestor exists at coarser level, or
        // 3) any finer node projects to this node.
        auto occupied_hier_at_level = [&](const torch::Tensor& cand_path_i64, int level) -> torch::Tensor {
            auto cand_path = cand_path_i64.contiguous().view({-1}).to(torch::kInt64);
            if (cand_path.numel() == 0) {
                return torch::empty({0}, torch::TensorOptions().dtype(torch::kBool).device(devL));
            }
            auto cand_key = cand_path.mul(256).add(
                torch::full_like(cand_path, static_cast<int64_t>(level)));

            auto is_occ = torch_mod.attr("isin")(
                py::cast(cand_key), py::cast(key_occ_exact))
                .cast<torch::Tensor>().to(torch::kBool);

            for (int anc_l = 1; anc_l < level; ++anc_l) {
                auto keep_mask_anc = make_keep_mask_for_level(anc_l);
                auto anc_path = (cand_path & keep_mask_anc);
                auto anc_key = anc_path.mul(256).add(
                    torch::full_like(anc_path, static_cast<int64_t>(anc_l)));
                auto has_anc = torch_mod.attr("isin")(
                    py::cast(anc_key), py::cast(key_occ_exact))
                    .cast<torch::Tensor>().to(torch::kBool);
                is_occ = is_occ | has_anc;
            }

            auto finer_occ_mask = (occ_level_i64 > level);
            if (finer_occ_mask.any().item<bool>()) {
                auto finer_idx = torch::nonzero(finer_occ_mask).view({-1});
                auto finer_path = occ_path_i64.index_select(0, finer_idx).contiguous();
                auto keep_mask_l = make_keep_mask_for_level(level);
                auto finer_proj = (finer_path & keep_mask_l);
                auto finer_proj_key = finer_proj.mul(256).add(
                    torch::full_like(finer_proj, static_cast<int64_t>(level)));
                finer_proj_key = unique_sorted_1d(finer_proj_key);
                auto has_desc = torch_mod.attr("isin")(
                    py::cast(cand_key), py::cast(finer_proj_key))
                    .cast<torch::Tensor>().to(torch::kBool);
                is_occ = is_occ | has_desc;
            }
            return is_occ.contiguous();
        };

        // Side-neighbor expansion only (no edge/corner fill).
        auto axis = torch::eye(3, long_opts);
        auto side_shift = torch::cat({axis, -axis}, 0).contiguous(); // [6,3]

        auto real_mask = torch::ones({octpath_occ.size(0)}, bool_opts);
        if (is_artificial_voxel_.defined() && is_artificial_voxel_.size(0) == octpath_occ.size(0)) {
            real_mask = (~is_artificial_voxel_.to(devL).to(torch::kBool)).contiguous();
        }
        auto seed_path_all = octpath_occ.contiguous();
        auto seed_level_all = octlevel_occ.contiguous();

        // Frontier-fill seeds: only real voxels in dense core (computed in createFromPcd).
        auto seed_mask = real_mask.clone();
        if (use_dense_core_neighbor_fill_) {
            if (has_dense_core_bb_ &&
                dense_core_bb_min_.defined() && dense_core_bb_max_.defined()) {
                py::tuple dec_seed = oct_utils.attr("octpath_decoding")(
                    py::cast(seed_path_all.contiguous()),
                    py::cast(seed_level_all.contiguous()),
                    py::cast(scene_center_.contiguous()),
                    py::cast(scene_extent_.contiguous())
                );
                auto seed_center = dec_seed[0].cast<at::Tensor>().contiguous(); // [N,3]
                auto bb_min = dense_core_bb_min_.to(devL).contiguous().view({1, 3});
                auto bb_max = dense_core_bb_max_.to(devL).contiguous().view({1, 3});
                auto in_dense_core =
                    (seed_center >= bb_min).all(/*dim=*/1) &
                    (seed_center <= bb_max).all(/*dim=*/1);
                seed_mask = seed_mask & in_dense_core.to(torch::kBool).contiguous();
            } else {
                std::cout << "[increasePcd] local_frontier_fill: dense-core bbox unavailable; "
                          << "fall back to real-only seeds.\n";
            }
        }

        auto seed_idx = torch::nonzero(seed_mask).view({-1});
        if (seed_idx.numel() == 0) {
            return 0;
        }
        auto seed_path = seed_path_all.index_select(0, seed_idx).contiguous();
        auto seed_level = seed_level_all.index_select(0, seed_idx).contiguous();
        auto seed_level_i64 = seed_level.view({-1}).to(torch::kInt64).contiguous();
        auto uniq_level = unique_sorted_1d(seed_level_i64);
        std::cout << "[increasePcd] local_frontier_fill seeds all="
                  << seed_path_all.size(0)
                  << " real=" << real_mask.sum().item<int64_t>()
                  << " active=" << seed_path.size(0) << "\n";

        std::vector<torch::Tensor> cand_path_list;
        std::vector<torch::Tensor> cand_level_list;
        auto uniq_level_cpu = uniq_level.to(torch::kCPU);
        for (int64_t i = 0; i < uniq_level_cpu.numel(); ++i) {
            const int lv = static_cast<int>(uniq_level_cpu[i].item<int64_t>());
            if (lv < 1 || lv > MAX_L) continue;

            auto level_mask = (seed_level_i64 == static_cast<int64_t>(lv));
            auto level_idx = torch::nonzero(level_mask).view({-1});
            if (level_idx.numel() == 0) continue;

            auto seed_path_lv = seed_path.index_select(0, level_idx).contiguous();
            auto L_seed_lv = torch::full(
                std::vector<int64_t>{seed_path_lv.size(0), 1},
                static_cast<int64_t>(lv),
                torch::dtype(torch::kInt8).device(devL)).contiguous();

            auto ijk_seed_lv = svr_utils.attr("octpath_2_ijk")(
                py::cast(seed_path_lv), py::cast(L_seed_lv))
                .cast<torch::Tensor>().to(torch::kLong).contiguous();
            TORCH_CHECK(
                ijk_seed_lv.dim() == 2 && ijk_seed_lv.size(1) == 3,
                "[increasePcd/local_frontier_fill] ijk_seed_lv must be [N,3], got ",
                ijk_seed_lv.sizes()
            );

            auto ijk_nbr_lv = (ijk_seed_lv.unsqueeze(1) + side_shift.unsqueeze(0))
                .contiguous().view({-1, 3});
            const int64_t grid_limit_lv = (1LL << lv);
            auto in_low_lv  = (ijk_nbr_lv >= 0).all(1);
            auto in_high_lv = (ijk_nbr_lv < grid_limit_lv).all(1);
            auto inb_lv     = in_low_lv & in_high_lv;
            if (!inb_lv.any().item<bool>()) continue;

            auto inb_idx_lv = torch::nonzero(inb_lv).view({-1});
            ijk_nbr_lv = ijk_nbr_lv.index_select(0, inb_idx_lv).contiguous();

            py::object uq_nbr = torch_mod.attr("unique")(
                py::cast(ijk_nbr_lv.contiguous()),
                py::arg("dim") = 0,
                py::arg("sorted") = true
            );
            ijk_nbr_lv = uq_nbr.cast<torch::Tensor>().contiguous();
            if (ijk_nbr_lv.dim() == 1) {
                ijk_nbr_lv = ijk_nbr_lv.view({-1, 3});
            }
            if (ijk_nbr_lv.numel() == 0) continue;

            auto L_nbr_lv = torch::full(
                std::vector<int64_t>{ijk_nbr_lv.size(0), 1},
                static_cast<int64_t>(lv),
                torch::dtype(torch::kInt8).device(devL)).contiguous();
            auto path_nbr_lv = svr_utils.attr("ijk_2_octpath")(
                py::cast(ijk_nbr_lv), py::cast(L_nbr_lv))
                .cast<torch::Tensor>().contiguous();

            auto is_occ_lv = occupied_hier_at_level(path_nbr_lv.view({-1}).to(torch::kInt64), lv);
            auto miss_idx_lv = torch::nonzero(~is_occ_lv).view({-1});
            if (miss_idx_lv.numel() == 0) continue;

            cand_path_list.push_back(path_nbr_lv.index_select(0, miss_idx_lv).contiguous());
            cand_level_list.push_back(L_nbr_lv.index_select(0, miss_idx_lv).contiguous());
        }

        if (cand_path_list.empty()) {
            return 0;
        }

        auto octpath_box = torch::cat(cand_path_list, 0).contiguous();
        auto L_box = torch::cat(cand_level_list, 0).contiguous();
        {
            auto key_all = octpath_box.view({-1}).to(torch::kInt64).mul(256)
                         + L_box.view({-1}).to(torch::kInt64);
            auto sort_pair = key_all.sort(/*dim=*/0);
            auto key_sorted = std::get<0>(sort_pair);
            auto perm = std::get<1>(sort_pair);
            auto keep = torch::empty_like(key_sorted, torch::kBool);
            keep.index_put_({0}, true);
            if (key_sorted.numel() > 1) {
                auto neq = key_sorted.index({torch::indexing::Slice(1, torch::indexing::None)})
                        != key_sorted.index({torch::indexing::Slice(torch::indexing::None, -1)});
                keep.index_put_({torch::indexing::Slice(1, torch::indexing::None)}, neq);
            }
            auto ksorted = torch::nonzero(keep).view({-1});
            auto korig = perm.index_select(0, ksorted);
            octpath_box = octpath_box.index_select(0, korig).contiguous();
            L_box = L_box.index_select(0, korig).contiguous();
        }
        if (max_dense_core_fill_cells_ > 0 && octpath_box.size(0) > max_dense_core_fill_cells_) {
            octpath_box = octpath_box.index({
                torch::indexing::Slice(0, max_dense_core_fill_cells_)});
            L_box = L_box.index({
                torch::indexing::Slice(0, max_dense_core_fill_cells_)});
        }
        if (max_artificial_cells_ > 0 && octpath_box.size(0) > max_artificial_cells_) {
            octpath_box = octpath_box.index({
                torch::indexing::Slice(0, max_artificial_cells_)});
            L_box = L_box.index({
                torch::indexing::Slice(0, max_artificial_cells_)});
        }
        if (octpath_box.numel() == 0) {
            return 0;
        }

        std::cout << "[increasePcd] local_frontier_fill generated candidates="
                  << octpath_box.size(0) << "\n";

        auto ijk_box = svr_utils.attr("octpath_2_ijk")(
            py::cast(octpath_box), py::cast(L_box)).cast<torch::Tensor>()
            .to(torch::kLong).contiguous();

        // Compare with CURRENT topology (which already includes real additions above).
        auto octpath_cur  = py_->svm.attr("octpath").cast<torch::Tensor>().contiguous();
        auto octlevel_cur = py_->svm.attr("octlevel").cast<torch::Tensor>().contiguous();

        auto key_box = octpath_box.view({-1}).to(torch::kInt64).mul(256)
                     + L_box.view({-1}).to(torch::kInt64);
        auto key_cur = octpath_cur.view({-1}).to(torch::kInt64).mul(256)
                     + octlevel_cur.view({-1}).to(torch::kInt64);

        auto is_dup = torch_mod.attr("isin")(py::cast(key_box), py::cast(key_cur))
                      .cast<torch::Tensor>().to(torch::kBool);
        auto new_mask_box = ~is_dup;

        // Drop base-level parents of already-subdivided regions.
        {
            if (octpath_cur.numel() > 0) {
                auto Lold_i64     = octlevel_cur.view({-1}).to(torch::kInt64);
                auto has_children = (Lold_i64 > base_L);

                if (has_children.any().item<bool>()) {
                    auto op_old_i64  = octpath_cur.view({-1}).to(torch::kInt64);
                    auto op_anc_base = (op_old_i64 & keep_mask);
                    auto sel_child   = torch::nonzero(has_children).view({-1});
                    op_anc_base      = op_anc_base.index_select(0, sel_child);

                    auto key_children_as_parent = op_anc_base.mul(256)
                        .add(torch::full_like(op_anc_base, static_cast<int64_t>(base_L)));

                    auto would_collide_parent = torch_mod.attr("isin")(
                        py::cast(key_box), py::cast(key_children_as_parent))
                        .cast<torch::Tensor>().to(torch::kBool);
                    new_mask_box = new_mask_box & (~would_collide_parent);
                }
            }
        }

        if (!new_mask_box.any().item<bool>()) {
            return 0;
        }

        auto sel_local = torch::nonzero(new_mask_box).view({-1});
        const int64_t Nm_local = sel_local.size(0);

        auto octpath_add2 = octpath_box.index_select(0, sel_local);
        auto L_add2       = L_box.index_select(0, sel_local);
        {
            py::tuple dec_add = oct_utils.attr("octpath_decoding")(
                py::cast(octpath_add2.contiguous()),
                py::cast(L_add2.contiguous()),
                py::cast(scene_center_.contiguous()),
                py::cast(scene_extent_.contiguous())
            );
            at::Tensor add_center = dec_add[0].cast<at::Tensor>().contiguous(); // [Nm,3]
            bb_min_viz = std::get<0>(add_center.min(/*dim=*/0, /*keepdim=*/false)).contiguous();
            bb_max_viz = std::get<0>(add_center.max(/*dim=*/0, /*keepdim=*/false)).contiguous();
            sel_artificials_viz = torch::arange(
                Nm_local, torch::TensorOptions().dtype(torch::kLong).device(devL));
            ijk_box_viz = torch::empty({0, 3}, long_opts);
        }

        py_->svm.attr("octpath")  = torch::cat({octpath_cur,  octpath_add2}, 0).contiguous();
        py_->svm.attr("octlevel") = torch::cat({octlevel_cur, L_add2},       0).contiguous();
        {
            auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(devL);
            auto i32_opts  = torch::TensorOptions().dtype(torch::kInt32).device(devL);
            if (!is_artificial_voxel_.defined() || is_artificial_voxel_.size(0) != octpath_cur.size(0)) {
                is_artificial_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
            } else if (is_artificial_voxel_.device() != devL) {
                is_artificial_voxel_ = is_artificial_voxel_.to(devL);
            }
            if (!is_orb_voxel_.defined() || is_orb_voxel_.size(0) != octpath_cur.size(0)) {
                is_orb_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
            } else if (is_orb_voxel_.device() != devL) {
                is_orb_voxel_ = is_orb_voxel_.to(devL);
            }
            if (!is_inactive_geo_voxel_.defined() || is_inactive_geo_voxel_.size(0) != octpath_cur.size(0)) {
                is_inactive_geo_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
            } else if (is_inactive_geo_voxel_.device() != devL) {
                is_inactive_geo_voxel_ = is_inactive_geo_voxel_.to(devL);
            }
            if (!is_rgbd_fill_render_holes_voxel_.defined() ||
                is_rgbd_fill_render_holes_voxel_.size(0) != octpath_cur.size(0)) {
                is_rgbd_fill_render_holes_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
            } else if (is_rgbd_fill_render_holes_voxel_.device() != devL) {
                is_rgbd_fill_render_holes_voxel_ = is_rgbd_fill_render_holes_voxel_.to(devL);
            }
            if (!is_depthanything_fill_holes_voxel_.defined() ||
                is_depthanything_fill_holes_voxel_.size(0) != octpath_cur.size(0)) {
                is_depthanything_fill_holes_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
            } else if (is_depthanything_fill_holes_voxel_.device() != devL) {
                is_depthanything_fill_holes_voxel_ = is_depthanything_fill_holes_voxel_.to(devL);
            }
            if (!is_promoted_artificial_voxel_.defined() || is_promoted_artificial_voxel_.size(0) != octpath_cur.size(0)) {
                is_promoted_artificial_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
            } else if (is_promoted_artificial_voxel_.device() != devL) {
                is_promoted_artificial_voxel_ = is_promoted_artificial_voxel_.to(devL);
            }
            if (!exist_since_iter_.defined() || exist_since_iter_.size(0) != octpath_cur.size(0)) {
                exist_since_iter_ = torch::zeros({octpath_cur.size(0)}, i32_opts);
            } else if (exist_since_iter_.device() != devL) {
                exist_since_iter_ = exist_since_iter_.to(devL);
            }
            if (!exist_since_kf_.defined() || exist_since_kf_.size(0) != octpath_cur.size(0)) {
                exist_since_kf_ = torch::full({octpath_cur.size(0)}, static_cast<int32_t>(-1), i32_opts);
            } else if (exist_since_kf_.device() != devL) {
                exist_since_kf_ = exist_since_kf_.to(devL);
            }
            if (!geometrically_unstable_voxel_.defined() || geometrically_unstable_voxel_.size(0) != octpath_cur.size(0)) {
                geometrically_unstable_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
            } else if (geometrically_unstable_voxel_.device() != devL) {
                geometrically_unstable_voxel_ = geometrically_unstable_voxel_.to(devL);
            }
            auto art_flag_add = torch::ones({Nm_local}, bool_opts);
            auto orb_flag_add = torch::zeros({Nm_local}, bool_opts);
            auto inactive_geo_flag_add = torch::zeros({Nm_local}, bool_opts);
            auto rgbd_fill_render_holes_flag_add = torch::zeros({Nm_local}, bool_opts);
            auto depthanything_fill_holes_flag_add = torch::zeros({Nm_local}, bool_opts);
            auto promoted_add_flag = torch::zeros({Nm_local}, bool_opts);
            auto exist_since_add = torch::full(
                {Nm_local}, static_cast<int32_t>(iteration), i32_opts);
            auto exist_since_kf_add = torch::full(
                {Nm_local}, current_kf_count, i32_opts);
            auto unstable_add_flag = torch::zeros({Nm_local}, bool_opts);
            is_artificial_voxel_ = torch::cat({is_artificial_voxel_, art_flag_add}, 0).contiguous();
            is_orb_voxel_ = torch::cat({is_orb_voxel_, orb_flag_add}, 0).contiguous();
            is_inactive_geo_voxel_ = torch::cat({is_inactive_geo_voxel_, inactive_geo_flag_add}, 0).contiguous();
            is_rgbd_fill_render_holes_voxel_ =
                torch::cat({is_rgbd_fill_render_holes_voxel_, rgbd_fill_render_holes_flag_add}, 0).contiguous();
            is_depthanything_fill_holes_voxel_ =
                torch::cat({is_depthanything_fill_holes_voxel_, depthanything_fill_holes_flag_add}, 0).contiguous();
            is_promoted_artificial_voxel_ = torch::cat({is_promoted_artificial_voxel_, promoted_add_flag}, 0).contiguous();
            exist_since_iter_ = torch::cat({exist_since_iter_, exist_since_add}, 0).contiguous();
            exist_since_kf_ = torch::cat({exist_since_kf_, exist_since_kf_add}, 0).contiguous();
            geometrically_unstable_voxel_ = torch::cat({geometrically_unstable_voxel_, unstable_add_flag}, 0).contiguous();
        }

        const int n_sh_rest_local = (max_sh_degree_ + 1)*(max_sh_degree_ + 1) - 1;
        auto shs_add2    = torch::zeros({Nm_local, n_sh_rest_local, 3}, torch::dtype(torch::kFloat32).device(devL)).contiguous();
        auto subdiv_add2 = torch::ones({Nm_local,1},                   torch::dtype(torch::kFloat32).device(devL)).contiguous();

        auto rgb_seed = torch::tensor(
            {artificial_bg_rgb_[0], artificial_bg_rgb_[1], artificial_bg_rgb_[2]},
            torch::dtype(torch::kFloat32).device(devL));
        if (rgb_add.defined() && rgb_add.numel() > 0) {
            rgb_seed = rgb_add.mean(/*dim=*/0, /*keepdim=*/false).clamp_(0.0f, 1.0f);
        }
        auto rgb_add2 = rgb_seed.view({1, 3}).repeat({Nm_local, 1}).contiguous();

        py::object sh0_add2_py = act_mod.attr("rgb2shzero")(py::cast(rgb_add2));
        auto sh0_add2 = sh0_add2_py.cast<torch::Tensor>().contiguous();

        auto grid_pts_key_new_local = py_->svm.attr("grid_pts_key").cast<torch::Tensor>();
        const int64_t M_prev_local = grid_pts_key_.defined() ? grid_pts_key_.size(0) : 0;
        const int64_t M_curr_local = grid_pts_key_new_local.size(0);
        if (M_curr_local > M_prev_local) {
            auto grow2 = makeGeoGridInitRows_(
                grid_pts_key_new_local,
                M_prev_local,
                M_curr_local,
                /*default_value=*/-10.0f);
            appendGroup_(/*group_idx=*/0, grow2, "_geo_grid_pts", &this->_geo_grid_pts_);
        }
        appendGroup_(/*group_idx=*/1, sh0_add2, "_sh0", &this->sh0_);
        appendGroup_(/*group_idx=*/2, shs_add2, "_shs", &this->shs_);

        auto subdiv_old2 = py_->svm.attr("_subdiv_p").cast<torch::Tensor>();
        py_->svm.attr("_subdiv_p") = torch::cat({subdiv_old2, subdiv_add2}, 0)
                                     .contiguous().detach().requires_grad_();
        this->subdiv_p_ = py_->svm.attr("_subdiv_p").cast<torch::Tensor>().requires_grad_(true);

        artificial_fill_happened_ = true;
        std::cout << "[increasePcd] local_frontier_fill added " << Nm_local
                  << " support voxels.\n";
        return Nm_local;
    };

    if (!insert_rendered_depth_candidate && !fill_empty_cells_ && use_local_frontier_fill_) {
        Nm_added += run_local_frontier_fill();
    }

    if (!insert_rendered_depth_candidate && fill_empty_cells_) {
        const bool warmup_reached = (iteration >= fill_empty_cells_warmup_iters_);
        if (!fill_empty_cells_done_ && !warmup_reached && !fill_empty_cells_warmup_notified_) {
            std::cout << "[increasePcd] fill_empty_cells_: waiting warmup (iter="
                      << iteration << " < " << fill_empty_cells_warmup_iters_ << ")\n";
            fill_empty_cells_warmup_notified_ = true;
        }
        const bool should_fill = (!fill_empty_cells_done_) && warmup_reached;
        if (should_fill) {
            bool did_local_fill = false;

            if (!did_local_fill) {
            // --- A) One-shot bbox from CURRENT batch heuristic (older behavior) ---
            torch::Tensor bb_min; // [3], world
            torch::Tensor bb_max; // [3], world
            bool bbox_ready = false;
            // Use accumulated real PCD (previous + current) for heuristic bbox estimation.
            torch::Tensor fill_pts_cpu = xyz_cpu.to(torch::kCPU).to(torch::kFloat32).contiguous();
            if (real_pcd_points_accum_cpu_.defined() && real_pcd_points_accum_cpu_.numel() > 0) {
                auto prev_pts_cpu =
                    real_pcd_points_accum_cpu_.to(torch::kCPU).to(torch::kFloat32).contiguous();
                if (prev_pts_cpu.dim() == 2 && prev_pts_cpu.size(1) == 3 && prev_pts_cpu.size(0) > 0) {
                    fill_pts_cpu = torch::cat({prev_pts_cpu, fill_pts_cpu}, 0).contiguous();
                }
            }
            std::cout << "[dense_core/refresh][fill] begin points="
                      << fill_pts_cpu.size(0)
                      << " rate=" << dense_core_pcd_density_rate_ << "\n";
            std::string py_err_msg;
            bool crossing_ok = false;
            {
                auto pts_f32 = fill_pts_cpu.to(torch::kCPU).to(torch::kFloat32).contiguous();
                if (pts_f32.defined() && pts_f32.dim() == 2 && pts_f32.size(1) == 3 && pts_f32.size(0) > 0) {
                    auto center_cpu = std::get<0>(pts_f32.median(/*dim=*/0, /*keepdim=*/false)).contiguous(); // [3], CPU
                    auto dist = std::get<0>(
                        (pts_f32 - center_cpu.view({1, 3})).abs().max(/*dim=*/1, /*keepdim=*/false))
                                    .to(torch::kFloat32)
                                    .contiguous(); // [N]
                    dist = std::get<0>(dist.sort(/*dim=*/0)).contiguous();

                    const int64_t n = dist.size(0);
                    if (n > 0) {
                        auto idx = torch::arange(
                            1, n + 1,
                            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
                        auto nonzero = (dist > 0.0f).to(torch::kFloat32);
                        auto density = idx * nonzero / ((2.0f * dist).pow(3) + 1e-6f);

                        int64_t begin_idx = static_cast<int64_t>(std::llround(static_cast<double>(n) * 0.05));
                        begin_idx = std::max<int64_t>(0, std::min<int64_t>(n - 1, begin_idx));

                        auto tail = density.index({torch::indexing::Slice(begin_idx, torch::indexing::None)}).contiguous();
                        if (tail.numel() > 0) {
                            const int64_t max_idx = begin_idx + tail.argmax().item<int64_t>();
                            const float max_density = density.index({max_idx}).item<float>();
                            const float target_density = dense_core_pcd_density_rate_ * max_density;
                            auto right = density.index({torch::indexing::Slice(max_idx, torch::indexing::None)}).contiguous();
                            auto below = torch::nonzero(right < target_density).view({-1}).contiguous();
                            const int64_t crossing_count = below.numel();
                            const float right_min = right.min().item<float>();
                            const float right_last = right.index({-1}).item<float>();

                            std::cout << "[dense_core/refresh][fill/precheck] n=" << n
                                      << " begin_idx=" << begin_idx
                                      << " max_idx=" << max_idx
                                      << " max_density=" << max_density
                                      << " target_density=" << target_density
                                      << " right_min=" << right_min
                                      << " right_last=" << right_last
                                      << " crossing_count=" << crossing_count
                                      << "\n";

                            crossing_ok = (crossing_count > 0);
                        }
                    }
                }
            }

            if (!crossing_ok) {
                std::cout << "[dense_core/refresh][fill] no_density_crossing at rate="
                          << dense_core_pcd_density_rate_
                          << "; skipping python heuristic this round.\n";
            } else {
                try {
                    static py::module bound_utils = py::module::import("src.utils.bounding_utils");
                    static py::module types       = py::module::import("types");

                    py::object ns = types.attr("SimpleNamespace")(
                        "points"_a = py::cast(fill_pts_cpu.contiguous()).attr("numpy")());
                    py::tuple cr = bound_utils.attr("main_scene_bound_pcd_heuristic")(
                        ns, py::float_(dense_core_pcd_density_rate_));

                    py::object center_t_py = torch_mod.attr("from_numpy")(cr[0]); // CPU tensor
                    auto center_t = center_t_py.cast<torch::Tensor>()
                        .to(torch::kCUDA).to(torch::kFloat32).contiguous().view({3});
                    const float radius_f = cr[1].cast<float>();
                    if (std::isfinite(radius_f) && radius_f > 0.0f) {
                        auto radius_t = torch::full(
                            {3}, radius_f, torch::dtype(torch::kFloat32).device(torch::kCUDA));
                        bb_min = (center_t - radius_t).contiguous();
                        bb_max = (center_t + radius_t).contiguous();
                        bbox_ready = true;
                        std::cout << "[dense_core/refresh][fill] done source=python radius="
                                  << radius_f << "\n";
                        std::cout << "[increasePcd] fill_empty_cells_: one-shot using accumulated-real-pcd heuristic bbox.\n";
                    } else {
                        std::cout << "[increasePcd] fill_empty_cells_: invalid heuristic radius="
                                  << radius_f << ".\n";
                    }
                } catch (const py::error_already_set& e) {
                    py_err_msg = e.what();
                } catch (const std::exception& e) {
                    py_err_msg = e.what();
                } catch (...) {
                    py_err_msg = "unknown_exception";
                }
            }

            if (!bbox_ready) {
                if (!py_err_msg.empty()) {
                    std::cout << "[increasePcd] fill_empty_cells_: python heuristic failed:\n"
                              << py_err_msg << "\n";
                }
                std::cout << "[increasePcd] fill_empty_cells_: bbox unavailable; will retry in next increasePcd.\n";
            }

            if (bbox_ready) {
            // Persist this recomputed dense-core so insertion-time filtering
            // uses the same latest bbox after warmup fill.
            dense_core_bb_min_ = bb_min.contiguous();
            dense_core_bb_max_ = bb_max.contiguous();
            has_dense_core_bb_ = true;
            std::cout << "[increasePcd] dense_core_bb_min=" << dense_core_bb_min_
                      << " dense_core_bb_max=" << dense_core_bb_max_ << std::endl;
            {
                auto bbox_center = (0.5f * (bb_min + bb_max)).view({1, 3}).contiguous();
                auto bbox_size = (bb_max - bb_min).view({1, 3}).contiguous();
                auto bbox_rgba = torch::zeros(
                    {1, 4},
                    torch::TensorOptions().dtype(torch::kFloat32).device(bbox_center.device()));
                bbox_rgba.index_put_({0, 0}, 1.0f);
                bbox_rgba.index_put_({0, 1}, 0.75f);
                bbox_rgba.index_put_({0, 3}, 0.18f);
                sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
                    bbox_center,
                    bbox_size,
                    bbox_rgba,
                    iteration,
                    "world/fill_empty_cells/raw_cube_aabb");
            }
            fill_empty_cells_done_ = true;

            // --- B) Convert bbox (world) -> dense ijk box at cached level/voxel size ---
            // Use cached scene_min_t_ and vox_eff_ (scalar size, but we kept it as [1,1]):
            auto vox_t = vox_eff_.mean()        // 0-dim CUDA float tensor
                            .view({1})
                            .repeat({3})       // [3]
                            .contiguous();     // CUDA float

            // Convert to indices, clamp to grid range [0, 2^L - 1]
            const int64_t grid_limit = (1LL << static_cast<int>(octlevel_)); // 2^L
            auto ijk_min = ((bb_min - scene_min_t_) / vox_t).floor().to(torch::kLong);
            auto ijk_max = ((bb_max - scene_min_t_) / vox_t).ceil().to(torch::kLong) - 1;
            // auto ijk_min = ((bb_min - scene_min_t_) / vox_t).to(torch::kLong);
            // auto ijk_max = ((bb_max - scene_min_t_) / vox_t).to(torch::kLong);
            // auto ijk_min = (((bb_min - scene_min_t_) / vox_t) - 0.5f).ceil().to(torch::kLong); // for not getting out of the bounding box
            // auto ijk_max = (((bb_max - scene_min_t_) / vox_t) - 0.5f).floor().to(torch::kLong);

            // Clamp to valid grid
            auto zero = torch::zeros_like(ijk_min);
            auto lim  = torch::full_like(ijk_max, grid_limit - 1);
            // auto lim  = torch::full_like(ijk_max, grid_limit);
            ijk_min = torch::maximum(ijk_min, zero);
            ijk_max = torch::minimum(ijk_max, lim);

            auto lens = (ijk_max - ijk_min + 1);        // [3] Long on CUDA
            int64_t nx = lens[0].item<int64_t>();
            int64_t ny = lens[1].item<int64_t>();
            int64_t nz = lens[2].item<int64_t>();
            long double Nc_est = (long double)nx * ny * nz;

            int64_t cap = std::max<int64_t>(1, max_artificial_cells_);    // set a sensible default (e.g., 200k)
            int64_t stride = 1;
            if (Nc_est > cap) {
                long double s = std::cbrt(Nc_est / (long double)cap);
                stride = std::max<int64_t>(1, (int64_t)std::ceil(s));
            }

            // If degenerate (all out), skip
            if ((ijk_min <= ijk_max).all().item<bool>()) {
                // --- C) Enumerate dense cells & set-diff vs current SVM topology ---
                auto ir = torch::arange(ijk_min[0].item<int64_t>(), ijk_max[0].item<int64_t>() + 1,
                                        stride, torch::dtype(torch::kLong).device(torch::kCUDA));
                auto jr = torch::arange(ijk_min[1].item<int64_t>(), ijk_max[1].item<int64_t>() + 1,
                                        stride, torch::dtype(torch::kLong).device(torch::kCUDA));
                auto kr = torch::arange(ijk_min[2].item<int64_t>(), ijk_max[2].item<int64_t>() + 1,
                                        stride, torch::dtype(torch::kLong).device(torch::kCUDA));
                auto grids = torch::meshgrid({ir, jr, kr}, /*indexing=*/"ij");
                auto ijk_box = torch::stack(
                    { grids[0].contiguous().view({-1}),
                    grids[1].contiguous().view({-1}),
                    grids[2].contiguous().view({-1}) }, 1).contiguous(); // [Nc,3]

                if (max_artificial_cells_ > 0 && ijk_box.size(0) > max_artificial_cells_) {
                    std::cout << "[increasePcd] fill_empty_cells_: limiting artificial cells"<< std::endl;
                    ijk_box = ijk_box.index({torch::indexing::Slice(0, max_artificial_cells_)});
                }

                auto dev = ijk_box.device();
                auto L_box = torch::full({ijk_box.size(0),1}, octlevel_, torch::dtype(torch::kInt8).device(dev)).contiguous();

                // Build morton/octpath for those cells
                py::object octpath_box_py = svr_utils.attr("ijk_2_octpath")(py::cast(ijk_box), py::cast(L_box));
                auto octpath_box = octpath_box_py.cast<torch::Tensor>().contiguous(); // [Nc,1] int64
                
                // ------------------------------------------------------------------------
                // B) OPTIONAL: camera-based filtering (mark_max_samp_rate + mark_near)
                //     so artificial voxels are also consistent with SVRaster camera logic
                // ------------------------------------------------------------------------
                if (filter_near_voxels_ && !cams.empty()) {
                    py::gil_scoped_acquire gil;
                    static py::module svr_mod   = py::module::import("svraster_cuda").attr("renderer");
                    static py::module oct_utils = py::module::import("src.utils.octree_utils");
                    // torch_mod already defined as static at top of increasePcd

                    // Decode voxel centers/sizes for artificial voxels
                    py::tuple dec_box = oct_utils.attr("octpath_decoding")(
                        py::cast(octpath_box.contiguous()),
                        py::cast(L_box.contiguous()),
                        py::cast(scene_center_.contiguous()),
                        py::cast(scene_extent_.contiguous())
                    );
                    at::Tensor vox_center_box = dec_box[0].cast<at::Tensor>(); // [Nc,3] cuda
                    at::Tensor vox_size_box   = dec_box[1].cast<at::Tensor>(); // [Nc,1] cuda

                    // Build Python list of CUDA MiniCams (same pattern as above)
                    py::list py_cams;
                    py::object py_cuda = torch_mod.attr("device")("cuda");
                    auto move_attr_to_cuda_if_tensor = [&](py::object& obj, const char* name){
                        if (py::hasattr(obj, name)) {
                            py::object t = obj.attr(name);
                            if (py::hasattr(t, "is_cuda") && !py::bool_(t.attr("is_cuda"))) {
                                obj.attr(name) = t.attr("to")(py_cuda);
                            }
                        }
                    };
                    for (const auto& c : cams) {
                        py::object py_cam = MiniCam_to_py(c);
                        move_attr_to_cuda_if_tensor(py_cam, "w2c");
                        move_attr_to_cuda_if_tensor(py_cam, "c2w");
                        move_attr_to_cuda_if_tensor(py_cam, "position");
                        move_attr_to_cuda_if_tensor(py_cam, "lookat");
                        py_cams.append(py_cam);
                    }

                    auto Nc_before = octpath_box.size(0);

                    // 1) Visibility / sampling rate
                    at::Tensor rate_box = svr_mod.attr("mark_max_samp_rate")(
                        py_cams,
                        py::cast(octpath_box),
                        py::cast(vox_center_box),
                        py::cast(vox_size_box)
                    ).cast<at::Tensor>();
                    at::Tensor kept_cam = rate_box > 0;
                    int64_t n_rate_pos_box = kept_cam.sum().item<int64_t>();

                    // 2) Near filtering with same threshold as for PCD voxels
                    //    (you can pull this out as a const float near_thresh = ... at top)
                    const float near_thresh = filter_near_voxels_ ? 0.2f : -1.0f;
                    int64_t n_near_hit_box = 0;
                    if (near_thresh > 0.0f) {
                        at::Tensor is_near_box = svr_mod.attr("mark_near")(
                            py_cams,
                            py::cast(octpath_box),
                            py::cast(vox_center_box),
                            py::cast(vox_size_box),
                            py::float_(near_thresh)
                        ).cast<at::Tensor>();
                        kept_cam = kept_cam & (~is_near_box);
                        n_near_hit_box = is_near_box.sum().item<int64_t>();
                    }

                    auto idx_box = torch::nonzero(kept_cam).view({-1});
                    int64_t K_box = idx_box.size(0);

                    if (K_box == 0) {
                        std::cout << "[increasePcd/fill_empty_cells] "
                                << "all artificial voxels filtered out by camera visibility / near; skipping."
                                << std::endl;
                        return;
                    }

                    if (K_box < octpath_box.size(0)) {
                        // Apply mask consistently to topology + ijk_box
                        octpath_box = octpath_box.index_select(0, idx_box).contiguous(); // [K_box,1]
                        L_box       = L_box.index_select(0, idx_box).contiguous();       // [K_box,1]
                        ijk_box     = ijk_box.index_select(0, idx_box).contiguous();     // [K_box,3]
                    }

                    std::cout << "[increasePcd/fill_empty_cells_cam] Nc_before=" << Nc_before
                            << " rate>0=" << n_rate_pos_box
                            << " near_hit=" << n_near_hit_box
                            << " kept_final=" << octpath_box.size(0) << std::endl;
                }

                // Compare with CURRENT topology (which already includes real additions above)
                auto octpath_cur  = py_->svm.attr("octpath").cast<torch::Tensor>().contiguous();
                auto octlevel_cur = py_->svm.attr("octlevel").cast<torch::Tensor>().contiguous();

                auto key_box = octpath_box.view({-1}).to(torch::kInt64).mul(256)
                            + L_box.view({-1}).to(torch::kInt64);
                auto key_cur = octpath_cur.view({-1}).to(torch::kInt64).mul(256)
                            + octlevel_cur.view({-1}).to(torch::kInt64);

                py::object isin_py = torch_mod.attr("isin")(py::cast(key_box), py::cast(key_cur));
                auto is_dup   = isin_py.cast<torch::Tensor>().to(torch::kBool);
                auto new_mask_box = ~is_dup;

                // --- Also drop base-level parents of already-subdivided regions (L_old > base_L) ---
                {
                    const int MAX_L  = max_num_levels_;
                    const int base_L = static_cast<int>(octlevel_);

                    if (octpath_cur.numel() > 0) {
                        auto Lold_i64     = octlevel_cur.view({-1}).to(torch::kInt64);   // [No]
                        auto has_children = (Lold_i64 > base_L);                         // [No] bool

                        if (has_children.any().item<bool>()) {
                            // Build mask to zero out octant bits below base_L
                            const int levels_below  = std::max(0, MAX_L - base_L);
                            const int bits_to_clear = 3 * levels_below;
                            long long lower_mask    = (bits_to_clear > 0) ? ((1LL << bits_to_clear) - 1LL) : 0LL;
                            long long keep_mask_ll  = ~lower_mask;

                            auto dev = octpath_cur.device();
                            auto keep_mask = torch::full(
                                {1},
                                static_cast<int64_t>(keep_mask_ll),
                                torch::TensorOptions().dtype(torch::kInt64).device(dev)
                            );

                            // Compute ancestors at base_L for all L_old > base_L
                            auto op_old_i64  = octpath_cur.view({-1}).to(torch::kInt64);  // [No]
                            auto op_anc_base = (op_old_i64 & keep_mask);                  // [No]
                            auto sel_child   = torch::nonzero(has_children).view({-1});   // [K]
                            op_anc_base      = op_anc_base.index_select(0, sel_child);    // [K]

                            // Keys for those parents at base_L
                            auto key_children_as_parent = op_anc_base.mul(256)
                                                        .add(torch::full_like(op_anc_base,
                                                                            static_cast<int64_t>(base_L)));

                            // small helper
                            auto unique_sorted_1d = [](const at::Tensor& t)->at::Tensor {
                                TORCH_CHECK(t.dim() == 1, "unique_sorted_1d expects a 1-D tensor");
                                if (t.numel() <= 1) return t.contiguous();
                                auto sort_res = t.sort(0);
                                auto sorted   = std::get<0>(sort_res);
                                using torch::indexing::Slice;
                                auto keep = torch::empty_like(sorted, torch::kBool);
                                keep.index_put_({0}, true);
                                auto neq = sorted.index({Slice(1, torch::indexing::None)})
                                        != sorted.index({Slice(torch::indexing::None, -1)});
                                keep.index_put_({Slice(1, torch::indexing::None)}, neq);
                                auto idx = torch::nonzero(keep).view({-1});
                                return sorted.index_select(0, idx).contiguous();
                            };

                            key_children_as_parent = unique_sorted_1d(key_children_as_parent);

                            // Any base-level candidate that equals one of these parents must be skipped
                            py::object isin_py2 = torch_mod.attr("isin")(
                                py::cast(key_box), py::cast(key_children_as_parent));
                            auto would_collide_parent = isin_py2.cast<torch::Tensor>().to(torch::kBool); // [Nc]

                            new_mask_box = new_mask_box & (~would_collide_parent);
                        }
                    }
                }

                // Proceed with the filtered candidates
                if (new_mask_box.any().item<bool>()) {
                    auto sel = torch::nonzero(new_mask_box).view({-1});
                    Nm_added = sel.size(0);

                    bb_min_viz       = bb_min;         // [3] (CUDA)
                    bb_max_viz       = bb_max;         // [3] (CUDA)
                    sel_artificials_viz= sel.clone();    // [Nm] indices into ijk_box
                    ijk_box_viz      = ijk_box.clone();// [Nc,3] (CUDA)

                    auto octpath_add2 = octpath_box.index_select(0, sel); // [Nm,1]
                    auto L_add2       = L_box.index_select(0, sel);       // [Nm,1]

                    // Append topology
                    py_->svm.attr("octpath")  = torch::cat({octpath_cur,  octpath_add2}, 0).contiguous();
                    py_->svm.attr("octlevel") = torch::cat({octlevel_cur, L_add2},       0).contiguous();
                    {
                        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
                        auto i32_opts  = torch::TensorOptions().dtype(torch::kInt32).device(dev);
                        if (!is_artificial_voxel_.defined() || is_artificial_voxel_.size(0) != octpath_cur.size(0)) {
                            is_artificial_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
                        } else if (is_artificial_voxel_.device() != dev) {
                            is_artificial_voxel_ = is_artificial_voxel_.to(dev);
                        }
                        if (!is_orb_voxel_.defined() || is_orb_voxel_.size(0) != octpath_cur.size(0)) {
                            is_orb_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
                        } else if (is_orb_voxel_.device() != dev) {
                            is_orb_voxel_ = is_orb_voxel_.to(dev);
                        }
                        if (!is_inactive_geo_voxel_.defined() || is_inactive_geo_voxel_.size(0) != octpath_cur.size(0)) {
                            is_inactive_geo_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
                        } else if (is_inactive_geo_voxel_.device() != dev) {
                            is_inactive_geo_voxel_ = is_inactive_geo_voxel_.to(dev);
                        }
                        if (!is_rgbd_fill_render_holes_voxel_.defined() ||
                            is_rgbd_fill_render_holes_voxel_.size(0) != octpath_cur.size(0)) {
                            is_rgbd_fill_render_holes_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
                        } else if (is_rgbd_fill_render_holes_voxel_.device() != dev) {
                            is_rgbd_fill_render_holes_voxel_ = is_rgbd_fill_render_holes_voxel_.to(dev);
                        }
                        if (!is_depthanything_fill_holes_voxel_.defined() ||
                            is_depthanything_fill_holes_voxel_.size(0) != octpath_cur.size(0)) {
                            is_depthanything_fill_holes_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
                        } else if (is_depthanything_fill_holes_voxel_.device() != dev) {
                            is_depthanything_fill_holes_voxel_ = is_depthanything_fill_holes_voxel_.to(dev);
                        }
                        if (!is_promoted_artificial_voxel_.defined() || is_promoted_artificial_voxel_.size(0) != octpath_cur.size(0)) {
                            is_promoted_artificial_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
                        } else if (is_promoted_artificial_voxel_.device() != dev) {
                            is_promoted_artificial_voxel_ = is_promoted_artificial_voxel_.to(dev);
                        }
                        if (!exist_since_iter_.defined() || exist_since_iter_.size(0) != octpath_cur.size(0)) {
                            exist_since_iter_ = torch::zeros({octpath_cur.size(0)}, i32_opts);
                        } else if (exist_since_iter_.device() != dev) {
                            exist_since_iter_ = exist_since_iter_.to(dev);
                        }
                        if (!exist_since_kf_.defined() || exist_since_kf_.size(0) != octpath_cur.size(0)) {
                            exist_since_kf_ = torch::full({octpath_cur.size(0)}, static_cast<int32_t>(-1), i32_opts);
                        } else if (exist_since_kf_.device() != dev) {
                            exist_since_kf_ = exist_since_kf_.to(dev);
                        }
                        if (!geometrically_unstable_voxel_.defined() || geometrically_unstable_voxel_.size(0) != octpath_cur.size(0)) {
                            geometrically_unstable_voxel_ = torch::zeros({octpath_cur.size(0)}, bool_opts);
                        } else if (geometrically_unstable_voxel_.device() != dev) {
                            geometrically_unstable_voxel_ = geometrically_unstable_voxel_.to(dev);
                        }
                        auto art_flag_add = torch::ones({Nm_added}, bool_opts);
                        auto orb_flag_add = torch::zeros({Nm_added}, bool_opts);
                        auto inactive_geo_flag_add = torch::zeros({Nm_added}, bool_opts);
                        auto rgbd_fill_render_holes_flag_add = torch::zeros({Nm_added}, bool_opts);
                        auto depthanything_fill_holes_flag_add = torch::zeros({Nm_added}, bool_opts);
                        auto promoted_add_flag = torch::zeros({Nm_added}, bool_opts);
                        auto exist_since_add = torch::full(
                            {Nm_added}, static_cast<int32_t>(iteration), i32_opts);
                        auto exist_since_kf_add = torch::full(
                            {Nm_added}, current_kf_count, i32_opts);
                        auto unstable_add_flag = torch::zeros({Nm_added}, bool_opts);
                        is_artificial_voxel_ = torch::cat({is_artificial_voxel_, art_flag_add}, 0).contiguous();
                        is_orb_voxel_ = torch::cat({is_orb_voxel_, orb_flag_add}, 0).contiguous();
                        is_inactive_geo_voxel_ = torch::cat({is_inactive_geo_voxel_, inactive_geo_flag_add}, 0).contiguous();
                        is_rgbd_fill_render_holes_voxel_ =
                            torch::cat({is_rgbd_fill_render_holes_voxel_, rgbd_fill_render_holes_flag_add}, 0).contiguous();
                        is_depthanything_fill_holes_voxel_ =
                            torch::cat({is_depthanything_fill_holes_voxel_, depthanything_fill_holes_flag_add}, 0).contiguous();
                        is_promoted_artificial_voxel_ = torch::cat({is_promoted_artificial_voxel_, promoted_add_flag}, 0).contiguous();
                        exist_since_iter_ = torch::cat({exist_since_iter_, exist_since_add}, 0).contiguous();
                        exist_since_kf_ = torch::cat({exist_since_kf_, exist_since_kf_add}, 0).contiguous();
                        geometrically_unstable_voxel_ = torch::cat({geometrically_unstable_voxel_, unstable_add_flag}, 0).contiguous();
                    }

                    // Prepare learnables for artificials
                    const int n_sh_rest = (max_sh_degree_ + 1)*(max_sh_degree_ + 1) - 1;
                    auto shs_add2    = torch::zeros({Nm_added, n_sh_rest, 3}, torch::dtype(torch::kFloat32).device(dev)).contiguous();
                    auto subdiv_add2 = torch::ones({Nm_added,1},             torch::dtype(torch::kFloat32).device(dev)).contiguous();

                    auto rgb_add2 = torch::empty({Nm_added,3}, torch::dtype(torch::kFloat32).device(dev));
                    rgb_add2.index_put_({torch::indexing::Slice(),0}, artificial_bg_rgb_[0]);
                    rgb_add2.index_put_({torch::indexing::Slice(),1}, artificial_bg_rgb_[1]);
                    rgb_add2.index_put_({torch::indexing::Slice(),2}, artificial_bg_rgb_[2]);

                    py::object sh0_add2_py = act_mod.attr("rgb2shzero")(py::cast(rgb_add2.contiguous()));
                    auto sh0_add2 = sh0_add2_py.cast<torch::Tensor>().contiguous();

                    // Grid growth and optimizer-preserving appends
                    auto grid_pts_key_new = py_->svm.attr("grid_pts_key").cast<torch::Tensor>();
                    const int64_t M_prev = grid_pts_key_.defined() ? grid_pts_key_.size(0) : 0;
                    const int64_t M_curr = grid_pts_key_new.size(0);
                    if (M_curr > M_prev) {
                        auto grow = makeGeoGridInitRows_(
                            grid_pts_key_new,
                            M_prev,
                            M_curr,
                            /*default_value=*/-10.0f);
                        appendGroup_(/*group_idx=*/0, grow, "_geo_grid_pts", &this->_geo_grid_pts_);
                    }
                    appendGroup_(/*group_idx=*/1, sh0_add2, "_sh0", &this->sh0_);
                    appendGroup_(/*group_idx=*/2, shs_add2, "_shs", &this->shs_);

                    // subdiv_p (not in optimizer groups)
                    auto subdiv_old2 = py_->svm.attr("_subdiv_p").cast<torch::Tensor>();
                    py_->svm.attr("_subdiv_p") = torch::cat({subdiv_old2, subdiv_add2}, 0)
                                                .contiguous().detach().requires_grad_();
                    this->subdiv_p_ = py_->svm.attr("_subdiv_p").cast<torch::Tensor>().requires_grad_(true);
                    // std::cout << "[increasePcd] Added " << Nm_added << " artificial voxels (pcd bbox).\n";
                    
                    // NEW: mark that we actually added artificial voxels this call
                    artificial_fill_happened_ = true;
                }
                // Keep your existing “new voxels” logging in sync by adding Nm_added to Nk (as noted earlier)
                // ... (use last_added = Nk + Nm_added later)
            } // valid bbox
            } // bbox_ready
            } // !did_local_fill
            if (fill_empty_cells_done_) {
                std::cout << "[increasePcd] fill_empty_cells_ added " << Nm_added
                          << " artificial voxels.\n";
            }
        } // should_fill
    } // fill_empty_cells_

    // ── 10) Pull back the rest for renderer (topology/derived fields) ───────
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
    // (optimizer params already set by appendGroup_; just ensure requires_grad)
    this->subdiv_p_      = py_->svm.attr("_subdiv_p").cast<torch::Tensor>().requires_grad_(true);
    // stats buffer resize
    this->max_w_ = torch::zeros({center_.size(0), 1}, torch::dtype(torch::kFloat32).device(dev));
    // Keep provenance tensor aligned with current topology size.
    if (!is_artificial_voxel_.defined() || is_artificial_voxel_.size(0) != center_.size(0)) {
        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, bool_opts);
        if (is_artificial_voxel_.defined() && is_artificial_voxel_.numel() > 0) {
            auto old = is_artificial_voxel_.to(dev).to(torch::kBool).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        is_artificial_voxel_ = aligned;
        std::cout << "[artificial/provenance] realigned flag tensor to N="
                  << center_.size(0) << "\n";
    }
    if (!is_orb_voxel_.defined() || is_orb_voxel_.size(0) != center_.size(0)) {
        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, bool_opts);
        if (is_orb_voxel_.defined() && is_orb_voxel_.numel() > 0) {
            auto old = is_orb_voxel_.to(dev).to(torch::kBool).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        is_orb_voxel_ = aligned;
        std::cout << "[orb/provenance] realigned flag tensor to N="
                  << center_.size(0) << "\n";
    }
    if (!is_inactive_geo_voxel_.defined() || is_inactive_geo_voxel_.size(0) != center_.size(0)) {
        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, bool_opts);
        if (is_inactive_geo_voxel_.defined() && is_inactive_geo_voxel_.numel() > 0) {
            auto old = is_inactive_geo_voxel_.to(dev).to(torch::kBool).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        is_inactive_geo_voxel_ = aligned;
        std::cout << "[inactive_geo/provenance] realigned flag tensor to N="
                  << center_.size(0) << "\n";
    }
    if (!is_rgbd_fill_render_holes_voxel_.defined() ||
        is_rgbd_fill_render_holes_voxel_.size(0) != center_.size(0)) {
        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, bool_opts);
        if (is_rgbd_fill_render_holes_voxel_.defined() &&
            is_rgbd_fill_render_holes_voxel_.numel() > 0) {
            auto old = is_rgbd_fill_render_holes_voxel_.to(dev).to(torch::kBool).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        is_rgbd_fill_render_holes_voxel_ = aligned;
        std::cout << "[rgbd_fill_render_holes/provenance] realigned flag tensor to N="
                  << center_.size(0) << "\n";
    }
    if (!is_depthanything_fill_holes_voxel_.defined() ||
        is_depthanything_fill_holes_voxel_.size(0) != center_.size(0)) {
        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, bool_opts);
        if (is_depthanything_fill_holes_voxel_.defined() &&
            is_depthanything_fill_holes_voxel_.numel() > 0) {
            auto old = is_depthanything_fill_holes_voxel_.to(dev).to(torch::kBool).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        is_depthanything_fill_holes_voxel_ = aligned;
        std::cout << "[depthanything_fill_holes/provenance] realigned flag tensor to N="
                  << center_.size(0) << "\n";
    }
    if (!is_promoted_artificial_voxel_.defined() || is_promoted_artificial_voxel_.size(0) != center_.size(0)) {
        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, bool_opts);
        if (is_promoted_artificial_voxel_.defined() && is_promoted_artificial_voxel_.numel() > 0) {
            auto old = is_promoted_artificial_voxel_.to(dev).to(torch::kBool).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        is_promoted_artificial_voxel_ = aligned;
        std::cout << "[artificial/promotion] realigned flag tensor to N="
                  << center_.size(0) << "\n";
    }
    // Keep exist_since_iter tensor aligned with current topology size.
    if (!exist_since_iter_.defined() || exist_since_iter_.size(0) != center_.size(0)) {
        auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, i32_opts);
        if (exist_since_iter_.defined() && exist_since_iter_.numel() > 0) {
            auto old = exist_since_iter_.to(dev).to(torch::kInt32).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        exist_since_iter_ = aligned;
        std::cout << "[exist_since_iter] realigned tensor to N="
                  << center_.size(0) << "\n";
    }
    if (!exist_since_kf_.defined() || exist_since_kf_.size(0) != center_.size(0)) {
        auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(dev);
        auto aligned = torch::full({center_.size(0)}, static_cast<int32_t>(-1), i32_opts);
        if (exist_since_kf_.defined() && exist_since_kf_.numel() > 0) {
            auto old = exist_since_kf_.to(dev).to(torch::kInt32).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        exist_since_kf_ = aligned;
        std::cout << "[exist_since_kf] realigned tensor to N="
                  << center_.size(0) << "\n";
    }
    if (!geometrically_unstable_voxel_.defined() || geometrically_unstable_voxel_.size(0) != center_.size(0)) {
        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, bool_opts);
        if (geometrically_unstable_voxel_.defined() && geometrically_unstable_voxel_.numel() > 0) {
            auto old = geometrically_unstable_voxel_.to(dev).to(torch::kBool).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        geometrically_unstable_voxel_ = aligned;
        std::cout << "[geometrically_unstable] realigned tensor to N="
                  << center_.size(0) << "\n";
    }
    if (!rendered_depth_candidate_voxel_.defined() ||
        rendered_depth_candidate_voxel_.size(0) != center_.size(0)) {
        auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, bool_opts);
        if (rendered_depth_candidate_voxel_.defined() &&
            rendered_depth_candidate_voxel_.numel() > 0) {
            auto old = rendered_depth_candidate_voxel_.to(dev).to(torch::kBool).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        rendered_depth_candidate_voxel_ = aligned;
        std::cout << "[rendered_depth_candidate] realigned flag tensor to N="
                  << center_.size(0) << "\n";
    }
    if (!rendered_depth_candidate_support_count_.defined() ||
        rendered_depth_candidate_support_count_.size(0) != center_.size(0)) {
        auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, i32_opts);
        if (rendered_depth_candidate_support_count_.defined() &&
            rendered_depth_candidate_support_count_.numel() > 0) {
            auto old = rendered_depth_candidate_support_count_.to(dev).to(torch::kInt32).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        rendered_depth_candidate_support_count_ = aligned;
        std::cout << "[rendered_depth_candidate] realigned support tensor to N="
                  << center_.size(0) << "\n";
    }
    if (!rendered_depth_candidate_last_seen_kf_.defined() ||
        rendered_depth_candidate_last_seen_kf_.size(0) != center_.size(0)) {
        auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(dev);
        auto aligned = torch::full({center_.size(0)}, static_cast<int32_t>(-1), i32_opts);
        if (rendered_depth_candidate_last_seen_kf_.defined() &&
            rendered_depth_candidate_last_seen_kf_.numel() > 0) {
            auto old = rendered_depth_candidate_last_seen_kf_.to(dev).to(torch::kInt32).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        rendered_depth_candidate_last_seen_kf_ = aligned;
        std::cout << "[rendered_depth_candidate] realigned last_seen tensor to N="
                  << center_.size(0) << "\n";
    }
    if (!rendered_depth_candidate_source_kind_.defined() ||
        rendered_depth_candidate_source_kind_.size(0) != center_.size(0)) {
        auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(dev);
        auto aligned = torch::zeros({center_.size(0)}, i32_opts);
        if (rendered_depth_candidate_source_kind_.defined() &&
            rendered_depth_candidate_source_kind_.numel() > 0) {
            auto old = rendered_depth_candidate_source_kind_.to(dev).to(torch::kInt32).contiguous();
            const int64_t copy_n = std::min<int64_t>(old.size(0), aligned.size(0));
            if (copy_n > 0) {
                aligned.index_put_(
                    {torch::indexing::Slice(0, copy_n)},
                    old.index({torch::indexing::Slice(0, copy_n)}));
            }
        }
        rendered_depth_candidate_source_kind_ = aligned;
        std::cout << "[rendered_depth_candidate] realigned source tensor to N="
                  << center_.size(0) << "\n";
    }

    // Keep dense-core history as accumulated raw PCD points (CPU), not voxel centers.
    if (!insert_rendered_depth_candidate && xyz_cpu.defined() && xyz_cpu.numel() > 0) {
        auto new_pts_cpu = xyz_cpu.to(torch::kCPU).to(torch::kFloat32).contiguous();
        if (!real_pcd_points_accum_cpu_.defined() || real_pcd_points_accum_cpu_.numel() == 0) {
            real_pcd_points_accum_cpu_ = new_pts_cpu;
        } else {
            real_pcd_points_accum_cpu_ =
                torch::cat({real_pcd_points_accum_cpu_, new_pts_cpu}, 0).contiguous();
        }
        if (max_real_pcd_points_ > 0 && real_pcd_points_accum_cpu_.size(0) > max_real_pcd_points_) {
            const int64_t total = real_pcd_points_accum_cpu_.size(0);
            auto idx = torch::linspace(
                0.0, static_cast<double>(total - 1), max_real_pcd_points_,
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
            ).round().to(torch::kLong);
            real_pcd_points_accum_cpu_ = real_pcd_points_accum_cpu_.index_select(0, idx).contiguous();
        }
    }

    // ── 10) Re-register with optimizer (new rows appended) ───────────────────
    VOXEL_MODEL_TENSORS_TO_VEC

    if (!pending_artificial_insert_rr_entity_path_.empty()) {
        if (Nk > 0) {
            const int64_t new_end = old_voxel_count + static_cast<int64_t>(Nk);
            if (this->center_.defined() && this->size_.defined() &&
                new_end <= this->center_.size(0) &&
                new_end <= this->size_.size(0)) {
                auto idx_new = torch::arange(
                    old_voxel_count,
                    new_end,
                    torch::TensorOptions().dtype(torch::kLong).device(this->center_.device()));
                auto centers_new = this->center_.index_select(0, idx_new).contiguous();
                auto sizes_new = this->size_.index_select(0, idx_new).view({-1, 1}).contiguous();
                auto rgb_new = rgb_add.clamp(0.0f, 1.0f).contiguous();
                auto alpha_new = torch::full(
                    {Nk, 1},
                    0.85f,
                    torch::TensorOptions().dtype(torch::kFloat32).device(rgb_new.device()));
                auto rgba_new = torch::cat({rgb_new, alpha_new}, 1).contiguous();
                auto* accum_centers = &rendered_depth_created_centers_accum_viz_;
                auto* accum_sizes = &rendered_depth_created_sizes_accum_viz_;
                auto* accum_rgba = &rendered_depth_created_rgba_accum_viz_;
                const char* rerun_tag = "[rerun/rendered_depth_insert]";

                if (!accum_centers->defined() || accum_centers->numel() == 0) {
                    *accum_centers = centers_new.clone();
                    *accum_sizes = sizes_new.clone();
                    *accum_rgba = rgba_new.clone();
                } else {
                    *accum_centers = torch::cat({*accum_centers, centers_new}, 0).contiguous();
                    *accum_sizes = torch::cat({*accum_sizes, sizes_new}, 0).contiguous();
                    *accum_rgba = torch::cat({*accum_rgba, rgba_new}, 0).contiguous();
                }

                if (max_rendered_depth_viz_accum_ > 0 &&
                    accum_centers->size(0) > max_rendered_depth_viz_accum_) {
                    const int64_t total = accum_centers->size(0);
                    auto keep = torch::linspace(
                        0.0,
                        static_cast<double>(total - 1),
                        max_rendered_depth_viz_accum_,
                        torch::TensorOptions().dtype(torch::kFloat32).device(accum_centers->device()))
                        .round().to(torch::kLong);
                    *accum_centers = accum_centers->index_select(0, keep).contiguous();
                    *accum_sizes = accum_sizes->index_select(0, keep).contiguous();
                    *accum_rgba = accum_rgba->index_select(0, keep).contiguous();
                }

                sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
                    *accum_centers,
                    *accum_sizes,
                    *accum_rgba,
                    iteration,
                    pending_artificial_insert_rr_entity_path_);
                // std::cout << rerun_tag << " logged_created_voxels_batch=" << Nk
                //           << " accumulated=" << accum_centers->size(0)
                //           << " entity=" << pending_artificial_insert_rr_entity_path_
                //           << std::endl;
            }
        } else {
            // std::cout << "[rerun/rendered_depth_insert] no created voxels to log"
            //           << " entity=" << pending_artificial_insert_rr_entity_path_
            //           << std::endl;
        }
        pending_artificial_insert_rr_entity_path_.clear();
        pending_insert_rendered_depth_candidate_source_kind_ = 0;
        pending_insert_rendered_depth_candidate_as_real_protected_ = false;
    }

    if (!pending_real_insert_rr_entity_path_.empty()) {
        if (Nk > 0) {
            const int64_t new_real_end = old_voxel_count + static_cast<int64_t>(Nk);
            if (this->center_.defined() && this->size_.defined() &&
                new_real_end <= this->center_.size(0) &&
                new_real_end <= this->size_.size(0)) {
                auto idx_new_real = torch::arange(
                    old_voxel_count,
                    new_real_end,
                    torch::TensorOptions().dtype(torch::kLong).device(this->center_.device()));
                auto centers_new_real = this->center_.index_select(0, idx_new_real).contiguous();
                auto sizes_new_real = this->size_.index_select(0, idx_new_real).view({-1, 1}).contiguous();
                auto rgb_new_real = rgb_add.clamp(0.0f, 1.0f).contiguous();
                auto alpha_new_real = torch::full(
                    {Nk, 1},
                    0.95f,
                    torch::TensorOptions().dtype(torch::kFloat32).device(rgb_new_real.device()));
                auto rgba_new_real = torch::cat({rgb_new_real, alpha_new_real}, 1).contiguous();

                const bool is_depthanything_entity =
                    pending_real_insert_rr_entity_path_ == "world/mono_prior_densify/created";
                const bool is_depthanything_fill_holes_entity =
                    pending_real_insert_rr_entity_path_ == "world/mono_prior_fill_holes/created";
                const bool is_orb_entity =
                    pending_real_insert_rr_entity_path_ == "world/orb/voxels_created";
                const bool is_inactive_geo_entity =
                    pending_real_insert_rr_entity_path_ == "world/voxels_inactive_geo_densify/created";
                const bool is_rgbd_fill_render_holes_entity =
                    pending_real_insert_rr_entity_path_ == "world/rgbd_fill_render_holes/created";
                const char* rerun_real_tag = is_depthanything_entity
                    ? "[rerun/depthanything_densify]"
                    : (is_depthanything_fill_holes_entity
                           ? "[rerun/depthanything_fill_holes]"
                           : (is_orb_entity
                                  ? "[rerun/orb]"
                                  : (is_inactive_geo_entity
                                         ? "[rerun/inactive_geo_densify]"
                                         : (is_rgbd_fill_render_holes_entity
                                                ? "[rerun/rgbd_fill_render_holes]"
                                                : "[rerun/real_insert]"))));

                if (is_orb_entity || is_inactive_geo_entity ||
                    is_rgbd_fill_render_holes_entity ||
                    is_depthanything_fill_holes_entity) {
                    // Source topics are logged from the live voxel model in the
                    // training loop so they stay synchronized with /voxels.
                    (void)rerun_real_tag;
                } else {
                    auto* accum_centers_real = is_depthanything_entity
                        ? &depthanything_created_centers_accum_viz_
                        : (is_depthanything_fill_holes_entity
                               ? &depthanything_fill_holes_created_centers_accum_viz_
                               : &inactive_geo_centers_accum_viz_);
                    auto* accum_sizes_real = is_depthanything_entity
                        ? &depthanything_created_sizes_accum_viz_
                        : (is_depthanything_fill_holes_entity
                               ? &depthanything_fill_holes_created_sizes_accum_viz_
                               : &inactive_geo_sizes_accum_viz_);
                    auto* accum_rgba_real = is_depthanything_entity
                        ? &depthanything_created_rgba_accum_viz_
                        : (is_depthanything_fill_holes_entity
                               ? &depthanything_fill_holes_created_rgba_accum_viz_
                               : &inactive_geo_rgba_accum_viz_);

                    if (!accum_centers_real->defined() ||
                        accum_centers_real->numel() == 0) {
                        *accum_centers_real = centers_new_real.clone();
                        *accum_sizes_real = sizes_new_real.clone();
                        *accum_rgba_real = rgba_new_real.clone();
                    } else {
                        *accum_centers_real = torch::cat(
                            {*accum_centers_real, centers_new_real}, 0).contiguous();
                        *accum_sizes_real = torch::cat(
                            {*accum_sizes_real, sizes_new_real}, 0).contiguous();
                        *accum_rgba_real = torch::cat(
                            {*accum_rgba_real, rgba_new_real}, 0).contiguous();
                    }

                    if (max_inactive_geo_viz_accum_ > 0 &&
                        accum_centers_real->size(0) > max_inactive_geo_viz_accum_) {
                        const int64_t total_inactive_geo = accum_centers_real->size(0);
                        auto keep = torch::linspace(
                            0.0,
                            static_cast<double>(total_inactive_geo - 1),
                            max_inactive_geo_viz_accum_,
                            torch::TensorOptions().dtype(torch::kFloat32).device(accum_centers_real->device())
                        ).round().to(torch::kLong);
                        *accum_centers_real =
                            accum_centers_real->index_select(0, keep).contiguous();
                        *accum_sizes_real =
                            accum_sizes_real->index_select(0, keep).contiguous();
                        *accum_rgba_real =
                            accum_rgba_real->index_select(0, keep).contiguous();
                    }

                    sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
                        *accum_centers_real,
                        *accum_sizes_real,
                        *accum_rgba_real,
                        iteration,
                        pending_real_insert_rr_entity_path_);
                    // std::cout << rerun_real_tag
                    //           << " logged_created_voxels_batch=" << Nk
                    //           << " accumulated=" << accum_centers_real->size(0)
                    //           << " entity=" << pending_real_insert_rr_entity_path_
                    //           << std::endl;
                }
            } else {
                // std::cout << "[rerun/inactive_geo_densify] skipped logging due to topology size mismatch"
                //           << " old_voxel_count=" << old_voxel_count
                //           << " new_real_voxels=" << Nk
                //           << " center_count=" << (this->center_.defined() ? this->center_.size(0) : 0)
                //           << " size_count=" << (this->size_.defined() ? this->size_.size(0) : 0)
                //           << " entity=" << pending_real_insert_rr_entity_path_
                //           << std::endl;
            }
        } else {
            const bool is_depthanything_entity =
                pending_real_insert_rr_entity_path_ == "world/mono_prior_densify/created";
            const bool is_depthanything_fill_holes_entity =
                pending_real_insert_rr_entity_path_ == "world/mono_prior_fill_holes/created";
            const bool is_orb_entity =
                pending_real_insert_rr_entity_path_ == "world/orb/voxels_created";
            // std::cout << (is_depthanything_entity
            //                  ? "[rerun/depthanything_densify] no created voxels to log"
            //                  : (is_depthanything_fill_holes_entity
            //                         ? "[rerun/depthanything_fill_holes] no created voxels to log"
            //                         : (is_orb_entity
            //                                ? "[rerun/orb] no created voxels to log"
            //                                : "[rerun/inactive_geo_densify] no created voxels to log")))
            //           << " entity=" << pending_real_insert_rr_entity_path_
            //           << std::endl;
        }
        pending_real_insert_rr_entity_path_.clear();
    }
    pending_insert_rendered_depth_candidate_ = false;
    pending_insert_rendered_depth_candidate_source_kind_ = 0;
    pending_insert_rendered_depth_candidate_as_real_protected_ = false;

    // 3) Log ONLY the newly-added voxels (orange)s
    const int64_t last_added = Nk + Nm_added;
    if (last_added > 0) {
        auto start = this->center_.size(0) - last_added;
        auto idx = torch::arange(start, this->center_.size(0),
                                this->center_.options().dtype(torch::kLong));
        auto centers_new = this->center_.index_select(0, idx);
        auto size_new    = this->size_.index_select(0, idx);
    }

    // Log artificial voxels on dedicated Rerun topics for debugging:
    // all CURRENT artificial voxels (exclude promoted-to-real),
    //    plus a dedicated topic for promoted voxels.
    if (this->center_.defined() && this->center_.numel() > 0 &&
        is_artificial_voxel_.defined() && is_artificial_voxel_.size(0) == this->center_.size(0)) {
        auto art_mask_raw = is_artificial_voxel_.to(this->center_.device()).to(torch::kBool).contiguous();
        auto promoted_mask_all = torch::zeros_like(art_mask_raw);
        if (is_promoted_artificial_voxel_.defined() &&
            is_promoted_artificial_voxel_.size(0) == this->center_.size(0)) {
            promoted_mask_all = is_promoted_artificial_voxel_
                .to(this->center_.device()).to(torch::kBool).contiguous();
        }
        auto art_mask_all = (art_mask_raw & (~promoted_mask_all)).contiguous();

        auto idx_art_all = torch::nonzero(art_mask_all).view({-1});
        auto idx_promoted_all = torch::nonzero(promoted_mask_all).view({-1});
        auto real_mask_all = (~art_mask_raw).contiguous();
        auto idx_real_all = torch::nonzero(real_mask_all).view({-1});

        // Enforce disjoint visualization by key (octpath, octlevel).
        if (idx_art_all.numel() > 0 && idx_real_all.numel() > 0) {
            auto key_all = this->oct_path_.view({-1}).to(torch::kInt64).mul(256)
                         + this->oct_level_.view({-1}).to(torch::kInt64);

            auto key_art = key_all.index_select(0, idx_art_all).contiguous();
            auto key_real = key_all.index_select(0, idx_real_all).contiguous();

            auto art_in_real = torch_mod.attr("isin")(py::cast(key_art), py::cast(key_real))
                .cast<torch::Tensor>().to(torch::kBool).contiguous();
            if (art_in_real.any().item<bool>()) {
                auto keep_art = torch::nonzero(~art_in_real).view({-1});
                const int64_t dropped = art_in_real.sum().item<int64_t>();
                idx_art_all = idx_art_all.index_select(0, keep_art).contiguous();
                std::cout << "[artificial/viz] dropped " << dropped
                          << " artificial rows overlapping real keys.\n";
            }

            if (idx_art_all.numel() > 0) {
                key_art = key_all.index_select(0, idx_art_all).contiguous();
                auto real_in_art = torch_mod.attr("isin")(py::cast(key_real), py::cast(key_art))
                    .cast<torch::Tensor>().to(torch::kBool).contiguous();
                if (real_in_art.any().item<bool>()) {
                    auto keep_real = torch::nonzero(~real_in_art).view({-1});
                    const int64_t dropped = real_in_art.sum().item<int64_t>();
                    idx_real_all = idx_real_all.index_select(0, keep_real).contiguous();
                    std::cout << "[artificial/viz] dropped " << dropped
                              << " real rows overlapping artificial keys.\n";
                }
            }
        }

        if (idx_art_all.numel() > 0) {
            artificial_centers_accum_viz_ = this->center_.index_select(0, idx_art_all).contiguous();
            artificial_sizes_accum_viz_   = this->size_.index_select(0, idx_art_all).view({-1, 1}).contiguous();

            if (max_artificial_viz_accum_ > 0 && artificial_centers_accum_viz_.size(0) > max_artificial_viz_accum_) {
                const int64_t total_art = artificial_centers_accum_viz_.size(0);
                auto keep = torch::linspace(
                    0.0, static_cast<double>(total_art - 1), max_artificial_viz_accum_,
                    torch::TensorOptions().dtype(torch::kFloat32).device(artificial_centers_accum_viz_.device())
                ).round().to(torch::kLong);
                artificial_centers_accum_viz_ = artificial_centers_accum_viz_.index_select(0, keep).contiguous();
                artificial_sizes_accum_viz_   = artificial_sizes_accum_viz_.index_select(0, keep).contiguous();
            }

            auto rgba_all = torch::zeros(
                {artificial_centers_accum_viz_.size(0), 4},
                torch::TensorOptions().dtype(torch::kFloat32).device(artificial_centers_accum_viz_.device()));
            rgba_all.index_put_({torch::indexing::Slice(), 0}, 1.0f);   // R
            rgba_all.index_put_({torch::indexing::Slice(), 2}, 1.0f);   // B -> magenta
            rgba_all.index_put_({torch::indexing::Slice(), 3}, 0.65f);  // alpha

            // sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
            //     artificial_centers_accum_viz_,
            //     artificial_sizes_accum_viz_,
            //     rgba_all,
            //     iteration,
            //     "world/voxels_artificials/all");

            // latest = newly-created artificials in this call:
            // current artificials minus artificials that already existed before this call.
            auto key_all = this->oct_path_.view({-1}).to(torch::kInt64).mul(256)
                         + this->oct_level_.view({-1}).to(torch::kInt64);
            auto key_art_all = key_all.index_select(0, idx_art_all).contiguous();
            auto is_old_art = torch::zeros(
                {key_art_all.size(0)},
                torch::TensorOptions().dtype(torch::kBool).device(key_art_all.device()));
            torch::Tensor art_key_before_for_latest = art_key_before_iter_;
            if (art_key_before_for_latest.defined() &&
                art_key_before_for_latest.device() != key_art_all.device()) {
                art_key_before_for_latest = art_key_before_for_latest.to(key_art_all.device());
            }
            if (art_key_before_for_latest.defined() && art_key_before_for_latest.numel() > 0) {
                is_old_art = torch_mod.attr("isin")(py::cast(key_art_all), py::cast(art_key_before_for_latest))
                    .cast<torch::Tensor>().to(torch::kBool).contiguous();
            }
            auto idx_latest_local = torch::nonzero(~is_old_art).view({-1});
            if (idx_latest_local.numel() > 0) {
                auto idx_art_latest = idx_art_all.index_select(0, idx_latest_local).contiguous();
                auto centers_art_latest = this->center_.index_select(0, idx_art_latest).contiguous();
                auto sizes_art_latest = this->size_.index_select(0, idx_art_latest).view({-1, 1}).contiguous();
                auto rgba_art_latest = torch::zeros(
                    {centers_art_latest.size(0), 4},
                    torch::TensorOptions().dtype(torch::kFloat32).device(centers_art_latest.device()));
                rgba_art_latest.index_put_({torch::indexing::Slice(), 0}, 1.0f);   // R
                rgba_art_latest.index_put_({torch::indexing::Slice(), 2}, 1.0f);   // B -> magenta
                rgba_art_latest.index_put_({torch::indexing::Slice(), 3}, 0.95f);  // alpha

                // sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
                //     centers_art_latest,
                //     sizes_art_latest,
                //     rgba_art_latest,
                //     iteration,
                //     "world/voxels_artificials/latest");
            }
        } else {
            artificial_centers_accum_viz_ = torch::empty(
                {0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(this->center_.device()));
            artificial_sizes_accum_viz_ = torch::empty(
                {0, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(this->center_.device()));
        }

        if (idx_promoted_all.numel() > 0) {
            auto promoted_centers = this->center_.index_select(0, idx_promoted_all).contiguous();
            auto promoted_sizes = this->size_.index_select(0, idx_promoted_all).view({-1, 1}).contiguous();

            if (max_artificial_viz_accum_ > 0 && promoted_centers.size(0) > max_artificial_viz_accum_) {
                const int64_t total_promoted = promoted_centers.size(0);
                auto keep = torch::linspace(
                    0.0, static_cast<double>(total_promoted - 1), max_artificial_viz_accum_,
                    torch::TensorOptions().dtype(torch::kFloat32).device(promoted_centers.device())
                ).round().to(torch::kLong);
                promoted_centers = promoted_centers.index_select(0, keep).contiguous();
                promoted_sizes = promoted_sizes.index_select(0, keep).contiguous();
            }

            auto rgba_promoted = torch::zeros(
                {promoted_centers.size(0), 4},
                torch::TensorOptions().dtype(torch::kFloat32).device(promoted_centers.device()));
            rgba_promoted.index_put_({torch::indexing::Slice(), 1}, 1.0f);   // G
            rgba_promoted.index_put_({torch::indexing::Slice(), 3}, 0.70f);  // alpha
        }

        // real_voxels_debug topic disabled by request
    }

}

void VoxelModel::increasePcd(
    torch::Tensor& new_point_cloud,
    torch::Tensor& new_colors,
    const int iteration,
    const std::vector<sv::MiniCam>& cams)
{
    // Follow GaussianModel::increasePcd(tensor) pattern, but reuse
    // our existing vector-based VoxelModel::increasePcd(..., cams).

    if (!new_point_cloud.defined() || !new_colors.defined())
        return;

    TORCH_CHECK(
        new_point_cloud.dim() == 2 && new_point_cloud.size(1) == 3,
        "VoxelModel::increasePcd(tensor): new_point_cloud must be [N,3]"
    );
    TORCH_CHECK(
        new_colors.dim() == 2 && new_colors.size(1) == 3,
        "VoxelModel::increasePcd(tensor): new_colors must be [N,3]"
    );
    TORCH_CHECK(
        new_point_cloud.size(0) == new_colors.size(0),
        "VoxelModel::increasePcd(tensor): points/colors size mismatch"
    );

    const int64_t N = new_point_cloud.size(0);
    if (N == 0)
        return;

    // Ensure CPU + contiguous
    auto xyz_cpu = new_point_cloud.to(torch::kCPU).contiguous();
    auto rgb_cpu = new_colors.to(torch::kCPU).contiguous();

    TORCH_CHECK(
        xyz_cpu.scalar_type() == torch::kFloat32 &&
        rgb_cpu.scalar_type() == torch::kFloat32,
        "VoxelModel::increasePcd(tensor): tensors must be float32"
    );

    // Flatten to 1D [3*N]
    auto xyz_flat = xyz_cpu.view({-1});  // [3N]
    auto rgb_flat = rgb_cpu.view({-1});  // [3N]

    std::vector<float> points(3 * N);
    std::vector<float> cols(3 * N);

    // Copy XYZ directly
    std::memcpy(
        points.data(),
        xyz_flat.data_ptr<float>(),
        points.size() * sizeof(float)
    );

    // Keep colors as-is; vector-based increasePcd now auto-detects [0,1] vs [0,255].
    const float* rgb_ptr = rgb_flat.data_ptr<float>();
    for (int64_t i = 0; i < static_cast<int64_t>(cols.size()); ++i) {
        cols[i] = rgb_ptr[i];
    }

    // Reuse the main SVRaster-aware pipeline
    increasePcd(
        points, cols, iteration, cams);
}

bool VoxelModel::refreshDenseCoreBBFromCurrentVoxels()
{
    auto set_dense_core_from_center_radius = [&](const torch::Tensor& center_cpu_f32, float radius)->bool {
        if (!center_cpu_f32.defined() || center_cpu_f32.numel() != 3 || !std::isfinite(radius) || radius <= 0.0f) {
            return false;
        }
        auto dev = this->center_.defined() ? this->center_.device() : torch::Device(torch::kCUDA);
        auto core_center = center_cpu_f32.to(dev).to(torch::kFloat32).contiguous().view({3});
        auto core_radius_t = torch::full(
            {3}, radius, torch::dtype(torch::kFloat32).device(dev));
        dense_core_bb_min_ = (core_center - core_radius_t).contiguous();
        dense_core_bb_max_ = (core_center + core_radius_t).contiguous();
        has_dense_core_bb_ = true;
        return true;
    };

    torch::Tensor pts_cpu;
    // 1) Prefer accumulated raw real PCD history.
    if (real_pcd_points_accum_cpu_.defined() && real_pcd_points_accum_cpu_.numel() > 0) {
        pts_cpu = real_pcd_points_accum_cpu_.to(torch::kCPU).to(torch::kFloat32).contiguous();
    }

    // 2) Fallback: current REAL voxel centers (artificial excluded).
    if ((!pts_cpu.defined() || pts_cpu.numel() == 0 || pts_cpu.size(0) < 8) &&
        this->center_.defined() && this->center_.numel() > 0 &&
        this->center_.dim() == 2 && this->center_.size(1) == 3) {
        auto centers_cpu = this->center_.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
        torch::Tensor keep_mask_cpu = torch::ones(
            {centers_cpu.size(0)},
            torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
        if (is_artificial_voxel_.defined() && is_artificial_voxel_.size(0) == centers_cpu.size(0)) {
            auto real_mask_cpu = (~is_artificial_voxel_.to(torch::kCPU).to(torch::kBool).contiguous());
            keep_mask_cpu = (keep_mask_cpu & real_mask_cpu).to(torch::kBool);
        }
        auto keep_idx = torch::nonzero(keep_mask_cpu).view({-1}).contiguous();
        if (keep_idx.numel() > 0) {
            centers_cpu = centers_cpu.index_select(0, keep_idx).contiguous();
        } else {
            centers_cpu = torch::empty(
                {0, 3},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        }
        if (centers_cpu.defined() && centers_cpu.size(0) >= 8) {
            pts_cpu = centers_cpu;
        }
    }

    if (!pts_cpu.defined() || pts_cpu.numel() == 0 || pts_cpu.size(0) < 8) {
        return false;
    }

    // 3) SVRaster-style precheck (same equations) to detect "no crossing" before Python call.
    auto pts_f32 = pts_cpu.to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto center_cpu = std::get<0>(pts_f32.median(/*dim=*/0, /*keepdim=*/false)).contiguous(); // [3], CPU
    auto dist = std::get<0>(
        (pts_f32 - center_cpu.view({1, 3})).abs().max(/*dim=*/1, /*keepdim=*/false))
                    .to(torch::kFloat32)
                    .contiguous(); // [N]
    dist = std::get<0>(dist.sort(/*dim=*/0)).contiguous();

    const int64_t n = dist.size(0);
    if (n <= 0) {
        // std::cout << "[dense_core/refresh] failed: empty dist after preprocessing.\n";
        return false;
    }
    auto idx = torch::arange(
        1, n + 1,
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto nonzero = (dist > 0.0f).to(torch::kFloat32);
    auto density = idx * nonzero / ((2.0f * dist).pow(3) + 1e-6f);

    int64_t begin_idx = static_cast<int64_t>(std::llround(static_cast<double>(n) * 0.05));
    begin_idx = std::max<int64_t>(0, std::min<int64_t>(n - 1, begin_idx));

    auto tail = density.index({torch::indexing::Slice(begin_idx, torch::indexing::None)}).contiguous();
    if (tail.numel() <= 0) {
        // std::cout << "[dense_core/refresh] failed: empty density tail (begin_idx="
        //           << begin_idx << ", n=" << n << ").\n";
        return false;
    }

    const int64_t max_idx = begin_idx + tail.argmax().item<int64_t>();
    const float max_density = density.index({max_idx}).item<float>();
    const float target_density = dense_core_pcd_density_rate_ * max_density;
    auto right = density.index({torch::indexing::Slice(max_idx, torch::indexing::None)}).contiguous();
    auto below = torch::nonzero(right < target_density).view({-1}).contiguous();
    const int64_t crossing_count = below.numel();
    const float right_min = right.min().item<float>();
    const float right_last = right.index({-1}).item<float>();

    // std::cout << "[dense_core/refresh][precheck] n=" << n
    //           << " begin_idx=" << begin_idx
    //           << " max_idx=" << max_idx
    //           << " max_density=" << max_density
    //           << " target_density=" << target_density
    //           << " right_min=" << right_min
    //           << " right_last=" << right_last
    //           << " crossing_count=" << crossing_count
    //           << "\n";

    if (crossing_count == 0) {
        // std::cout << "[dense_core/refresh] no_density_crossing at rate="
        //           << dense_core_pcd_density_rate_
        //           << "; skipping python heuristic this round.\n";
        return false;
    }

    // 4) Python heuristic (SVRaster) once precheck says crossing exists.
    std::string py_err_msg;
    try {
        py::gil_scoped_acquire gil;
        static py::module bound_utils = py::module::import("src.utils.bounding_utils");
        static py::module types = py::module::import("types");
        static py::module torch_mod = py::module::import("torch");

        py::object ns = types.attr("SimpleNamespace")(
            "points"_a = py::cast(pts_cpu.contiguous()).attr("numpy")());
        py::tuple cr = bound_utils.attr("main_scene_bound_pcd_heuristic")(
            ns, py::float_(dense_core_pcd_density_rate_));

        py::object core_center_py = torch_mod.attr("from_numpy")(cr[0]);
        auto core_center = core_center_py.cast<torch::Tensor>()
            .to(torch::kCPU).to(torch::kFloat32).contiguous().view({3}); // [3]
        const float core_radius = cr[1].cast<float>();
        if (set_dense_core_from_center_radius(core_center, core_radius)) {
            // std::cout << "[dense_core/refresh] source=python points=" << pts_cpu.size(0)
            //           << " rate=" << dense_core_pcd_density_rate_
            //           << " radius=" << core_radius << "\n";
            // std::cout << "[dense_core/refresh] dense_core_bb_min=" << dense_core_bb_min_
            //           << " dense_core_bb_max=" << dense_core_bb_max_ << std::endl;
            return true;
        }
        py_err_msg = "heuristic returned invalid center/radius";
    } catch (const py::error_already_set& e) {
        py_err_msg = e.what();
    } catch (const std::exception& e) {
        py_err_msg = e.what();
    } catch (...) {
        py_err_msg = "unknown_exception";
    }

    // std::cout << "[dense_core/refresh] python heuristic failed: "
    //           << py_err_msg << "\n";
    // std::cout << "[dense_core/refresh] failed to estimate dense-core bbox.\n";
    return false;
}

void VoxelModel::logDenseCoreBBoxToRerun(
    int iteration,
    const std::string& entity_path) const
{
    if (!has_dense_core_bb_ ||
        !dense_core_bb_min_.defined() ||
        !dense_core_bb_max_.defined()) {
        return;
    }

    auto bb_min = dense_core_bb_min_.to(torch::kFloat32).contiguous().view({1, 3});
    auto bb_max = dense_core_bb_max_.to(torch::kFloat32).contiguous().view({1, 3});
    auto bbox_center = (0.5f * (bb_min + bb_max)).contiguous();
    auto bbox_size = (bb_max - bb_min).contiguous();
    auto bbox_rgba = torch::zeros(
        {1, 4},
        torch::TensorOptions().dtype(torch::kFloat32).device(bbox_center.device()));
    bbox_rgba.index_put_({0, 1}, 1.0f);
    bbox_rgba.index_put_({0, 2}, 1.0f);
    bbox_rgba.index_put_({0, 3}, 0.22f);

    sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
        bbox_center,
        bbox_size,
        bbox_rgba,
        iteration,
        entity_path);
}

void VoxelModel::setGeometricallyUnstableMask(const torch::Tensor& mask)
{
    if (!center_.defined() || center_.numel() == 0) {
        geometrically_unstable_voxel_ = torch::empty(
            {0},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
        return;
    }

    auto m = mask;
    if (!m.defined()) {
        geometrically_unstable_voxel_ = torch::zeros(
            {center_.size(0)},
            torch::TensorOptions().dtype(torch::kBool).device(center_.device()));
        return;
    }
    if (m.dim() == 2 && m.size(1) == 1) {
        m = m.squeeze(1);
    }
    m = m.to(center_.device()).to(torch::kBool).contiguous().view({-1});
    if (m.numel() != center_.size(0)) {
        auto aligned = torch::zeros(
            {center_.size(0)},
            torch::TensorOptions().dtype(torch::kBool).device(center_.device()));
        const int64_t copy_n = std::min<int64_t>(aligned.size(0), m.size(0));
        if (copy_n > 0) {
            aligned.index_put_(
                {torch::indexing::Slice(0, copy_n)},
                m.index({torch::indexing::Slice(0, copy_n)}));
        }
        m = aligned;
    }
    geometrically_unstable_voxel_ = m.contiguous();
}

void VoxelModel::promoteRenderedDepthCandidates(const torch::Tensor& promote_mask)
{
    if (!center_.defined() || center_.numel() == 0) {
        return;
    }

    auto mask = promote_mask;
    if (!mask.defined()) {
        return;
    }
    if (mask.dim() == 2 && mask.size(1) == 1) {
        mask = mask.squeeze(1);
    }
    mask = mask.to(center_.device()).to(torch::kBool).contiguous().view({-1});
    if (mask.numel() != center_.size(0)) {
        return;
    }

    auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(center_.device());
    auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(center_.device());
    if (!is_artificial_voxel_.defined() || is_artificial_voxel_.size(0) != center_.size(0)) {
        is_artificial_voxel_ = torch::zeros({center_.size(0)}, bool_opts);
    }
    if (!is_promoted_artificial_voxel_.defined() || is_promoted_artificial_voxel_.size(0) != center_.size(0)) {
        is_promoted_artificial_voxel_ = torch::zeros({center_.size(0)}, bool_opts);
    }
    if (!rendered_depth_candidate_voxel_.defined() ||
        rendered_depth_candidate_voxel_.size(0) != center_.size(0)) {
        rendered_depth_candidate_voxel_ = torch::zeros({center_.size(0)}, bool_opts);
    }
    if (!rendered_depth_candidate_support_count_.defined() ||
        rendered_depth_candidate_support_count_.size(0) != center_.size(0)) {
        rendered_depth_candidate_support_count_ = torch::zeros({center_.size(0)}, i32_opts);
    }
    if (!rendered_depth_candidate_last_seen_kf_.defined() ||
        rendered_depth_candidate_last_seen_kf_.size(0) != center_.size(0)) {
        rendered_depth_candidate_last_seen_kf_ =
            torch::full({center_.size(0)}, static_cast<int32_t>(-1), i32_opts);
    }
    if (!rendered_depth_candidate_source_kind_.defined() ||
        rendered_depth_candidate_source_kind_.size(0) != center_.size(0)) {
        rendered_depth_candidate_source_kind_ = torch::zeros({center_.size(0)}, i32_opts);
    }

    auto effective_mask =
        (mask &
         rendered_depth_candidate_voxel_.to(center_.device()).to(torch::kBool) &
         is_artificial_voxel_.to(center_.device()).to(torch::kBool)).contiguous();
    auto idx = torch::nonzero(effective_mask).view({-1});
    if (idx.numel() == 0) {
        return;
    }

    is_artificial_voxel_.index_put_({idx}, false);
    is_promoted_artificial_voxel_.index_put_({idx}, true);
    rendered_depth_candidate_voxel_.index_put_({idx}, false);
    total_promoted_artificial_voxels_ += idx.size(0);
}

namespace {
void logLiveMaskedVoxels(
    const torch::Tensor& centers_all,
    const torch::Tensor& sizes_all,
    const torch::Tensor& mask,
    const torch::Tensor& live_colors,
    const int64_t max_viz_count,
    const int iteration,
    const std::string& entity_path,
    const std::array<float, 4>& fallback_rgba)
{
    if (!centers_all.defined() || centers_all.numel() == 0 ||
        !sizes_all.defined() || sizes_all.numel() == 0) {
        return;
    }
    if (!mask.defined() || mask.size(0) != centers_all.size(0)) {
        return;
    }

    auto live_mask = mask
        .to(centers_all.device())
        .to(torch::kBool)
        .contiguous()
        .view({-1});
    if (!live_mask.any().item<bool>()) {
        return;
    }

    auto idx = torch::nonzero(live_mask).view({-1});
    auto centers = centers_all.index_select(0, idx).contiguous();
    auto sizes = sizes_all.index_select(0, idx).view({-1, 1}).contiguous();
    torch::Tensor colors;
    if (live_colors.defined() && live_colors.numel() > 0 &&
        live_colors.dim() == 2 &&
        live_colors.size(0) == centers_all.size(0) &&
        (live_colors.size(1) == 3 || live_colors.size(1) == 4)) {
        colors = live_colors.to(centers.device()).index_select(0, idx).contiguous();
    } else {
        colors = torch::zeros(
            {centers.size(0), 4},
            torch::TensorOptions().dtype(torch::kFloat32).device(centers.device()));
        colors.index_put_({torch::indexing::Slice(), 0}, fallback_rgba[0]);
        colors.index_put_({torch::indexing::Slice(), 1}, fallback_rgba[1]);
        colors.index_put_({torch::indexing::Slice(), 2}, fallback_rgba[2]);
        colors.index_put_({torch::indexing::Slice(), 3}, fallback_rgba[3]);
    }

    if (max_viz_count > 0 && centers.size(0) > max_viz_count) {
        const int64_t total = centers.size(0);
        auto keep = torch::linspace(
            0.0,
            static_cast<double>(total - 1),
            max_viz_count,
            torch::TensorOptions().dtype(torch::kFloat32).device(centers.device()))
            .round()
            .to(torch::kLong);
        centers = centers.index_select(0, keep).contiguous();
        sizes = sizes.index_select(0, keep).contiguous();
        colors = colors.index_select(0, keep).contiguous();
    }

    sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
        centers,
        sizes,
        colors,
        iteration,
        entity_path);
}
} // namespace

void VoxelModel::logLiveOrbVoxels(const int iteration, const torch::Tensor& live_colors)
{
    logLiveMaskedVoxels(
        this->center_,
        this->size_,
        this->is_orb_voxel_,
        live_colors,
        max_inactive_geo_viz_accum_,
        iteration,
        "world/orb/voxels_created",
        {0.1f, 0.8f, 1.0f, 0.75f});
}

void VoxelModel::logLiveInactiveGeoVoxels(const int iteration, const torch::Tensor& live_colors)
{
    logLiveMaskedVoxels(
        this->center_,
        this->size_,
        this->is_inactive_geo_voxel_,
        live_colors,
        max_inactive_geo_viz_accum_,
        iteration,
        "world/voxels_inactive_geo_densify/created",
        {0.7f, 0.45f, 0.2f, 0.75f});
}

void VoxelModel::logLiveRgbdFillRenderHolesVoxels(const int iteration, const torch::Tensor& live_colors)
{
    logLiveMaskedVoxels(
        this->center_,
        this->size_,
        this->is_rgbd_fill_render_holes_voxel_,
        live_colors,
        max_inactive_geo_viz_accum_,
        iteration,
        "world/rgbd_fill_render_holes/created",
        {0.95f, 0.25f, 0.85f, 0.75f});
}

void VoxelModel::logLiveDepthAnythingFillHolesVoxels(const int iteration, const torch::Tensor& live_colors)
{
    logLiveMaskedVoxels(
        this->center_,
        this->size_,
        this->is_depthanything_fill_holes_voxel_,
        live_colors,
        max_inactive_geo_viz_accum_,
        iteration,
        "world/mono_prior_fill_holes/created",
        {0.2f, 0.95f, 0.45f, 0.75f});
}

void VoxelModel::logFinalartificialVoxels(const int iteration)
{
    if (!this->center_.defined() || this->center_.numel() == 0) {
        return;
    }

    if (!is_artificial_voxel_.defined() || is_artificial_voxel_.size(0) != this->center_.size(0)) {
        return;
    }

    auto art_mask_raw = is_artificial_voxel_.to(this->center_.device()).to(torch::kBool).contiguous();
    auto art_mask_all = art_mask_raw.clone();
    auto promoted_mask = torch::zeros_like(art_mask_raw);
    if (is_promoted_artificial_voxel_.defined() &&
        is_promoted_artificial_voxel_.size(0) == this->center_.size(0)) {
        promoted_mask = is_promoted_artificial_voxel_
            .to(this->center_.device()).to(torch::kBool).contiguous();
        art_mask_all = art_mask_all & (~promoted_mask);
    }
    auto real_mask_all = (~art_mask_raw).contiguous();

    const int64_t n_total = this->center_.size(0);
    const int64_t n_art_raw = art_mask_raw.sum().item<int64_t>();
    const int64_t n_art_final = art_mask_all.sum().item<int64_t>();
    const int64_t n_real = real_mask_all.sum().item<int64_t>();
    const int64_t n_art_promoted_overlap = (art_mask_raw & promoted_mask).sum().item<int64_t>();
    const int64_t n_art_real_overlap = (art_mask_raw & real_mask_all).sum().item<int64_t>();
    int64_t n_promoted = 0;
    if (is_promoted_artificial_voxel_.defined() &&
        is_promoted_artificial_voxel_.size(0) == this->center_.size(0)) {
        n_promoted = is_promoted_artificial_voxel_
            .to(this->center_.device()).to(torch::kBool).sum().item<int64_t>();
    }

    if (!art_mask_all.any().item<bool>()) {
        artificial_centers_accum_viz_ = torch::empty(
            {0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(this->center_.device()));
        artificial_sizes_accum_viz_ = torch::empty(
            {0, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(this->center_.device()));
        return;
    }

    auto idx_art_all = torch::nonzero(art_mask_all).view({-1});
    artificial_centers_accum_viz_ = this->center_.index_select(0, idx_art_all).contiguous();
    artificial_sizes_accum_viz_   = this->size_.index_select(0, idx_art_all).view({-1, 1}).contiguous();

    if (max_artificial_viz_accum_ > 0 && artificial_centers_accum_viz_.size(0) > max_artificial_viz_accum_) {
        const int64_t total_art = artificial_centers_accum_viz_.size(0);
        auto keep = torch::linspace(
            0.0, static_cast<double>(total_art - 1), max_artificial_viz_accum_,
            torch::TensorOptions().dtype(torch::kFloat32).device(artificial_centers_accum_viz_.device())
        ).round().to(torch::kLong);
        artificial_centers_accum_viz_ = artificial_centers_accum_viz_.index_select(0, keep).contiguous();
        artificial_sizes_accum_viz_   = artificial_sizes_accum_viz_.index_select(0, keep).contiguous();
    }

    auto rgba_all = torch::zeros(
        {artificial_centers_accum_viz_.size(0), 4},
        torch::TensorOptions().dtype(torch::kFloat32).device(artificial_centers_accum_viz_.device()));
    rgba_all.index_put_({torch::indexing::Slice(), 0}, 1.0f);   // R
    rgba_all.index_put_({torch::indexing::Slice(), 2}, 1.0f);   // B -> magenta
    rgba_all.index_put_({torch::indexing::Slice(), 3}, 0.65f);  // alpha

    // sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
    //     artificial_centers_accum_viz_,
    //     artificial_sizes_accum_viz_,
    //     rgba_all,
    //     iteration,
    //     "world/voxels_artificials/all");
}

void VoxelModel::logFinalPromotedartificialVoxels(const int iteration)
{
    if (!this->center_.defined() || this->center_.numel() == 0) {
        return;
    }

    if (!is_promoted_artificial_voxel_.defined() ||
        is_promoted_artificial_voxel_.size(0) != this->center_.size(0)) {
        return;
    }

    auto promoted_mask = is_promoted_artificial_voxel_
        .to(this->center_.device()).to(torch::kBool).contiguous();
    if (!promoted_mask.any().item<bool>()) {
        return;
    }

    auto idx_promoted = torch::nonzero(promoted_mask).view({-1});
    auto promoted_centers = this->center_.index_select(0, idx_promoted).contiguous();
    auto promoted_sizes = this->size_.index_select(0, idx_promoted).view({-1, 1}).contiguous();

    if (max_artificial_viz_accum_ > 0 && promoted_centers.size(0) > max_artificial_viz_accum_) {
        const int64_t total_promoted = promoted_centers.size(0);
        auto keep = torch::linspace(
            0.0, static_cast<double>(total_promoted - 1), max_artificial_viz_accum_,
            torch::TensorOptions().dtype(torch::kFloat32).device(promoted_centers.device())
        ).round().to(torch::kLong);
        promoted_centers = promoted_centers.index_select(0, keep).contiguous();
        promoted_sizes = promoted_sizes.index_select(0, keep).contiguous();
    }

    auto rgba_promoted = torch::zeros(
        {promoted_centers.size(0), 4},
        torch::TensorOptions().dtype(torch::kFloat32).device(promoted_centers.device()));
    rgba_promoted.index_put_({torch::indexing::Slice(), 1}, 1.0f);   // G
    rgba_promoted.index_put_({torch::indexing::Slice(), 3}, 0.70f);  // alpha

    // sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
    //     promoted_centers,
    //     promoted_sizes,
    //     rgba_promoted,
    //     iteration,
    //     "world/voxels_artificials/promoted");
}

void VoxelModel::syncFromPython_() {
    py::gil_scoped_acquire gil;

    auto fetch = [&](const char* name){
        return py_->svm.attr(name).cast<torch::Tensor>().contiguous();
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

void VoxelModel::appendGroup_(int group_idx,
                              const torch::Tensor& add_rows,
                              const char* svm_field_name,
                              torch::Tensor* out_member_param) {
    namespace py = pybind11;
    py::gil_scoped_acquire gil;

    if (!py_->optimizer_py || py_->optimizer_py.is_none()) return;

    // 1) param_groups[i]['params'][0]
    py::list groups = py_->optimizer_py.attr("param_groups");
    auto g = groups[group_idx].cast<py::dict>();
    py::list gparams = g["params"].cast<py::list>();
    torch::Tensor old_param = gparams[0].cast<torch::Tensor>();

    // 2) state dict (tensor-keyed)
    py::dict state = py_->optimizer_py.attr("state").cast<py::dict>();
    bool had_state = state.contains(py::cast(old_param));

    // 3) Build new concatenated param (cat along dim=0)
    // torch::Tensor new_param = torch::cat({old_param, add_rows}, /*dim=*/0).contiguous().requires_grad_(true);
    torch::Tensor new_param = torch::cat({old_param, add_rows}, /*dim=*/0).contiguous().detach().requires_grad_(true); // leaf

    // 4) Move/extend state if present
    if (had_state) {
        py::dict old_s = state[py::cast(old_param)].cast<py::dict>();
        // Copy step
        py::object step_obj = old_s["step"];
        // Extend exp_avg / exp_avg_sq
        torch::Tensor exp_avg    = old_s["exp_avg"].cast<torch::Tensor>();
        torch::Tensor exp_avg_sq = old_s["exp_avg_sq"].cast<torch::Tensor>();
        torch::Tensor z = torch::zeros_like(add_rows);

        torch::Tensor exp_avg_new    = torch::cat({exp_avg,    z}, 0).contiguous();
        torch::Tensor exp_avg_sq_new = torch::cat({exp_avg_sq, z}, 0).contiguous();

        // Construct new state
        py::dict new_s;
        new_s["step"]       = step_obj;         // keep same scalar
        new_s["exp_avg"]    = exp_avg_new;
        new_s["exp_avg_sq"] = exp_avg_sq_new;

        // Swap state key from old tensor to new tensor
        state.attr("__delitem__")(py::cast(old_param));
        state[py::cast(new_param)] = new_s;
    }
    // If no state existed yet, that's fine — SparseAdam lazily initializes on first .step()

    // 5) Rebind group param to the *same* new tensor object
    gparams[0] = py::cast(new_param);
    g["params"] = gparams;     // write back (usually not necessary, but explicit)

    // 6) Keep Python SVM and C++ members in sync with the *same* tensor object
    py_->svm.attr(svm_field_name) = new_param;
    *out_member_param = new_param;
}

void VoxelModel::debugAssertTopologyConsistent(const char* where) const {
    // 1) topology must not contain duplicate (octpath,octlevel) pairs
    TORCH_CHECK(oct_path_.defined() && oct_level_.defined(),
                "[", where, "] topology tensors undefined");
    TORCH_CHECK(oct_path_.dim()==2 && oct_path_.size(1)==1,
                "[", where, "] oct_path_ must be [N,1]");
    TORCH_CHECK(oct_level_.dim()==2 && oct_level_.size(1)==1,
                "[", where, "] oct_level_ must be [N,1]");
    TORCH_CHECK(oct_path_.size(0) == oct_level_.size(0),
                "[", where, "] oct_path_/oct_level_ length mismatch");

    auto key = oct_path_.view({-1}).to(torch::kInt64).mul(256)
            + oct_level_.view({-1}).to(torch::kInt64);
    auto key_flat = key.view({-1});
    int64_t unique_count = key_flat.numel();
    if (key_flat.numel() > 1) {
        auto sorted = std::get<0>(key_flat.sort(/*dim=*/0));
        using torch::indexing::Slice;
        auto neigh_diff = sorted.index({Slice(1, torch::indexing::None)})
                        != sorted.index({Slice(torch::indexing::None, -1)});
        unique_count = neigh_diff.sum().item<int64_t>() + 1;
    }
    TORCH_CHECK(unique_count == key_flat.numel(),
                "[", where, "] Duplicate (octpath, level) found in topology");

    // 2) learnables must match topology length
    const auto M = oct_path_.size(0);
    auto bad = [&](const torch::Tensor& t){ return !t.defined() || t.size(0)!=M; };

    TORCH_CHECK(!bad(subdiv_p_), "[", where, "] _subdiv_p length mismatch vs topology");
    TORCH_CHECK(!bad(sh0_),      "[", where, "] _sh0 length mismatch vs topology");
    TORCH_CHECK(!bad(shs_),      "[", where, "] _shs length mismatch vs topology");

    // (optional) grid points can be different length, so no check for _geo_grid_pts_
}

VoxelModel::StatPkg
VoxelModel::computeTrainingStat(const std::vector<MiniCam>& cams) {
    // Mirrors SVAdaptive.compute_training_stat (but uses our renderer)
    freezeVoxGeo();

    const int64_t N = center_.size(0);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_type_);
    // auto max_w            = torch::zeros({N,1}, opts);
    this->max_w_.zero_();
    auto min_samp_interval = torch::full ({N,1}, 1e30f, opts);
    auto view_cnt         = torch::zeros({N,1}, opts);

    for (const auto& cam : cams) {
        // std::cout << "rendering cam " << cam.width << "x" << cam.height << "\n";
        // auto pkg = render(cam, torch::Tensor(), cam.height, cam.width, color_mode='dontcare', track_max_w=True);
        auto pkg = render(
            cam, 
            cam.height, 
            cam.width, 
            torch::Tensor(),
            "dontcare", 
            true);
        
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
    if (mask.dim() == 2 && mask.size(1) == 1) {
        mask = mask.squeeze(1);
    }
    TORCH_CHECK(mask.dim() == 1, "pruning: mask must be [N] or [N,1]");
    const int64_t N_before = mask.size(0);

    if (!is_artificial_voxel_.defined() || is_artificial_voxel_.size(0) != N_before) {
        is_artificial_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (is_artificial_voxel_.device() != mask.device()) {
        is_artificial_voxel_ = is_artificial_voxel_.to(mask.device());
    }
    if (!is_orb_voxel_.defined() || is_orb_voxel_.size(0) != N_before) {
        is_orb_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (is_orb_voxel_.device() != mask.device()) {
        is_orb_voxel_ = is_orb_voxel_.to(mask.device());
    }
    if (!is_inactive_geo_voxel_.defined() || is_inactive_geo_voxel_.size(0) != N_before) {
        is_inactive_geo_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (is_inactive_geo_voxel_.device() != mask.device()) {
        is_inactive_geo_voxel_ = is_inactive_geo_voxel_.to(mask.device());
    }
    if (!is_rgbd_fill_render_holes_voxel_.defined() ||
        is_rgbd_fill_render_holes_voxel_.size(0) != N_before) {
        is_rgbd_fill_render_holes_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (is_rgbd_fill_render_holes_voxel_.device() != mask.device()) {
        is_rgbd_fill_render_holes_voxel_ = is_rgbd_fill_render_holes_voxel_.to(mask.device());
    }
    if (!is_depthanything_fill_holes_voxel_.defined() ||
        is_depthanything_fill_holes_voxel_.size(0) != N_before) {
        is_depthanything_fill_holes_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (is_depthanything_fill_holes_voxel_.device() != mask.device()) {
        is_depthanything_fill_holes_voxel_ =
            is_depthanything_fill_holes_voxel_.to(mask.device());
    }
    if (!is_promoted_artificial_voxel_.defined() || is_promoted_artificial_voxel_.size(0) != N_before) {
        is_promoted_artificial_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (is_promoted_artificial_voxel_.device() != mask.device()) {
        is_promoted_artificial_voxel_ = is_promoted_artificial_voxel_.to(mask.device());
    }
    if (!exist_since_iter_.defined() || exist_since_iter_.size(0) != N_before) {
        exist_since_iter_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
    } else if (exist_since_iter_.device() != mask.device()) {
        exist_since_iter_ = exist_since_iter_.to(mask.device());
    }
    if (!exist_since_kf_.defined() || exist_since_kf_.size(0) != N_before) {
        exist_since_kf_ = torch::full(
            {N_before},
            static_cast<int32_t>(-1),
            torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
    } else if (exist_since_kf_.device() != mask.device()) {
        exist_since_kf_ = exist_since_kf_.to(mask.device());
    }
    if (!geometrically_unstable_voxel_.defined() || geometrically_unstable_voxel_.size(0) != N_before) {
        geometrically_unstable_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (geometrically_unstable_voxel_.device() != mask.device()) {
        geometrically_unstable_voxel_ = geometrically_unstable_voxel_.to(mask.device());
    }
    if (!rendered_depth_candidate_voxel_.defined() ||
        rendered_depth_candidate_voxel_.size(0) != N_before) {
        rendered_depth_candidate_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (rendered_depth_candidate_voxel_.device() != mask.device()) {
        rendered_depth_candidate_voxel_ = rendered_depth_candidate_voxel_.to(mask.device());
    }
    if (!rendered_depth_candidate_support_count_.defined() ||
        rendered_depth_candidate_support_count_.size(0) != N_before) {
        rendered_depth_candidate_support_count_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
    } else if (rendered_depth_candidate_support_count_.device() != mask.device()) {
        rendered_depth_candidate_support_count_ = rendered_depth_candidate_support_count_.to(mask.device());
    }
    if (!rendered_depth_candidate_last_seen_kf_.defined() ||
        rendered_depth_candidate_last_seen_kf_.size(0) != N_before) {
        rendered_depth_candidate_last_seen_kf_ = torch::full(
            {N_before},
            static_cast<int32_t>(-1),
            torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
    } else if (rendered_depth_candidate_last_seen_kf_.device() != mask.device()) {
        rendered_depth_candidate_last_seen_kf_ = rendered_depth_candidate_last_seen_kf_.to(mask.device());
    }
    if (!rendered_depth_candidate_source_kind_.defined() ||
        rendered_depth_candidate_source_kind_.size(0) != N_before) {
        rendered_depth_candidate_source_kind_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
    } else if (rendered_depth_candidate_source_kind_.device() != mask.device()) {
        rendered_depth_candidate_source_kind_ = rendered_depth_candidate_source_kind_.to(mask.device());
    }
    auto art_before = is_artificial_voxel_.to(torch::kBool).contiguous();
    auto orb_before = is_orb_voxel_.to(torch::kBool).contiguous();
    auto inactive_geo_before = is_inactive_geo_voxel_.to(torch::kBool).contiguous();
    auto rgbd_fill_render_holes_before =
        is_rgbd_fill_render_holes_voxel_.to(torch::kBool).contiguous();
    auto depthanything_fill_holes_before =
        is_depthanything_fill_holes_voxel_.to(torch::kBool).contiguous();
    auto promoted_before = is_promoted_artificial_voxel_.to(torch::kBool).contiguous();
    auto exist_since_before = exist_since_iter_.to(torch::kInt32).contiguous();
    auto exist_since_kf_before = exist_since_kf_.to(torch::kInt32).contiguous();
    auto unstable_before = geometrically_unstable_voxel_.to(torch::kBool).contiguous();
    auto rendered_depth_before = rendered_depth_candidate_voxel_.to(torch::kBool).contiguous();
    auto rendered_depth_support_before = rendered_depth_candidate_support_count_.to(torch::kInt32).contiguous();
    auto rendered_depth_last_seen_before = rendered_depth_candidate_last_seen_kf_.to(torch::kInt32).contiguous();
    const int64_t n_art_before = art_before.sum().item<int64_t>();
    const int64_t n_promoted_before = promoted_before.sum().item<int64_t>();
    const int64_t n_prune_total = mask.sum().item<int64_t>();
    const int64_t n_prune_art = (mask & art_before).sum().item<int64_t>();
    const int64_t n_prune_real = n_prune_total - n_prune_art;
    auto old_octpath = py_->svm.attr("octpath").cast<torch::Tensor>().contiguous();
    auto old_octlevel = py_->svm.attr("octlevel").cast<torch::Tensor>().contiguous();
    auto old_key_cpu = (old_octpath.view({-1}).to(torch::kInt64).mul(256)
                      + old_octlevel.view({-1}).to(torch::kInt64))
                      .to(torch::kCPU).contiguous();
    auto old_art_cpu = art_before.to(torch::kCPU).contiguous();
    auto old_orb_cpu = orb_before.to(torch::kCPU).contiguous();
    auto old_inactive_geo_cpu = inactive_geo_before.to(torch::kCPU).contiguous();
    auto old_rgbd_fill_render_holes_cpu =
        rgbd_fill_render_holes_before.to(torch::kCPU).contiguous();
    auto old_depthanything_fill_holes_cpu =
        depthanything_fill_holes_before.to(torch::kCPU).contiguous();
    auto old_promoted_cpu = promoted_before.to(torch::kCPU).contiguous();
    auto old_exist_since_cpu = exist_since_before.to(torch::kCPU).contiguous();
    auto old_exist_since_kf_cpu = exist_since_kf_before.to(torch::kCPU).contiguous();
    auto old_unstable_cpu = unstable_before.to(torch::kCPU).contiguous();
    auto old_rendered_depth_cpu = rendered_depth_before.to(torch::kCPU).contiguous();
    auto old_rendered_depth_support_cpu = rendered_depth_support_before.to(torch::kCPU).contiguous();
    auto old_rendered_depth_last_seen_cpu = rendered_depth_last_seen_before.to(torch::kCPU).contiguous();
    auto old_rendered_depth_source_cpu =
        rendered_depth_candidate_source_kind_.to(torch::kCPU).to(torch::kInt32).contiguous();

    py_->svm.attr("pruning")(mask);
    syncFromPython_();

    auto new_key_cpu = (this->oct_path_.view({-1}).to(torch::kInt64).mul(256)
                      + this->oct_level_.view({-1}).to(torch::kInt64))
                      .to(torch::kCPU).contiguous();
    const int64_t N_after = new_key_cpu.size(0);

    auto art_after_cpu = torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto orb_after_cpu = torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto inactive_geo_after_cpu = torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto rgbd_fill_render_holes_after_cpu =
        torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto depthanything_fill_holes_after_cpu =
        torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto promoted_after_cpu = torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto exist_since_after_cpu = torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto exist_since_kf_after_cpu = torch::full(
        {N_after}, static_cast<int32_t>(-1),
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto unstable_after_cpu = torch::zeros(
        {N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto rendered_depth_after_cpu = torch::zeros(
        {N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto rendered_depth_support_after_cpu = torch::zeros(
        {N_after}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto rendered_depth_last_seen_after_cpu = torch::full(
        {N_after}, static_cast<int32_t>(-1),
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto rendered_depth_source_after_cpu = torch::zeros(
        {N_after}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));

    std::unordered_map<int64_t, int64_t> key_to_old_idx;
    key_to_old_idx.reserve(static_cast<size_t>(old_key_cpu.size(0) * 2 + 1));
    const int64_t* old_key_ptr = old_key_cpu.data_ptr<int64_t>();
    for (int64_t i = 0; i < old_key_cpu.size(0); ++i) {
        key_to_old_idx.emplace(old_key_ptr[i], i);
    }

    const bool* old_art_ptr = old_art_cpu.data_ptr<bool>();
    const bool* old_orb_ptr = old_orb_cpu.data_ptr<bool>();
    const bool* old_inactive_geo_ptr = old_inactive_geo_cpu.data_ptr<bool>();
    const bool* old_rgbd_fill_render_holes_ptr = old_rgbd_fill_render_holes_cpu.data_ptr<bool>();
    const bool* old_depthanything_fill_holes_ptr = old_depthanything_fill_holes_cpu.data_ptr<bool>();
    const bool* old_promoted_ptr = old_promoted_cpu.data_ptr<bool>();
    const int32_t* old_exist_since_ptr = old_exist_since_cpu.data_ptr<int32_t>();
    const int32_t* old_exist_since_kf_ptr = old_exist_since_kf_cpu.data_ptr<int32_t>();
    const bool* old_unstable_ptr = old_unstable_cpu.data_ptr<bool>();
    const bool* old_rendered_depth_ptr = old_rendered_depth_cpu.data_ptr<bool>();
    const int32_t* old_rendered_depth_support_ptr = old_rendered_depth_support_cpu.data_ptr<int32_t>();
    const int32_t* old_rendered_depth_last_seen_ptr = old_rendered_depth_last_seen_cpu.data_ptr<int32_t>();
    const int32_t* old_rendered_depth_source_ptr = old_rendered_depth_source_cpu.data_ptr<int32_t>();
    const int64_t* new_key_ptr = new_key_cpu.data_ptr<int64_t>();
    bool* art_after_ptr = art_after_cpu.data_ptr<bool>();
    bool* orb_after_ptr = orb_after_cpu.data_ptr<bool>();
    bool* inactive_geo_after_ptr = inactive_geo_after_cpu.data_ptr<bool>();
    bool* rgbd_fill_render_holes_after_ptr = rgbd_fill_render_holes_after_cpu.data_ptr<bool>();
    bool* depthanything_fill_holes_after_ptr = depthanything_fill_holes_after_cpu.data_ptr<bool>();
    bool* promoted_after_ptr = promoted_after_cpu.data_ptr<bool>();
    int32_t* exist_since_after_ptr = exist_since_after_cpu.data_ptr<int32_t>();
    int32_t* exist_since_kf_after_ptr = exist_since_kf_after_cpu.data_ptr<int32_t>();
    bool* unstable_after_ptr = unstable_after_cpu.data_ptr<bool>();
    bool* rendered_depth_after_ptr = rendered_depth_after_cpu.data_ptr<bool>();
    int32_t* rendered_depth_support_after_ptr = rendered_depth_support_after_cpu.data_ptr<int32_t>();
    int32_t* rendered_depth_last_seen_after_ptr = rendered_depth_last_seen_after_cpu.data_ptr<int32_t>();
    int32_t* rendered_depth_source_after_ptr = rendered_depth_source_after_cpu.data_ptr<int32_t>();

    int64_t matched_by_key = 0;
    for (int64_t i = 0; i < N_after; ++i) {
        auto it = key_to_old_idx.find(new_key_ptr[i]);
        if (it == key_to_old_idx.end()) {
            continue;
        }
        const int64_t old_idx = it->second;
        art_after_ptr[i] = old_art_ptr[old_idx];
        orb_after_ptr[i] = old_orb_ptr[old_idx];
        inactive_geo_after_ptr[i] = old_inactive_geo_ptr[old_idx];
        rgbd_fill_render_holes_after_ptr[i] = old_rgbd_fill_render_holes_ptr[old_idx];
        depthanything_fill_holes_after_ptr[i] = old_depthanything_fill_holes_ptr[old_idx];
        promoted_after_ptr[i] = old_promoted_ptr[old_idx];
        exist_since_after_ptr[i] = old_exist_since_ptr[old_idx];
        exist_since_kf_after_ptr[i] = old_exist_since_kf_ptr[old_idx];
        unstable_after_ptr[i] = old_unstable_ptr[old_idx];
        rendered_depth_after_ptr[i] = old_rendered_depth_ptr[old_idx];
        rendered_depth_support_after_ptr[i] = old_rendered_depth_support_ptr[old_idx];
        rendered_depth_last_seen_after_ptr[i] = old_rendered_depth_last_seen_ptr[old_idx];
        rendered_depth_source_after_ptr[i] = old_rendered_depth_source_ptr[old_idx];
        ++matched_by_key;
    }

    if (matched_by_key != N_after) {
        std::cout << "[PRUNE/provenance] WARNING: unmatched voxels after key remap: "
                  << (N_after - matched_by_key) << "/" << N_after << "\n";
    }

    is_artificial_voxel_ = art_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    is_orb_voxel_ = orb_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    is_inactive_geo_voxel_ = inactive_geo_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    is_rgbd_fill_render_holes_voxel_ =
        rgbd_fill_render_holes_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    is_depthanything_fill_holes_voxel_ =
        depthanything_fill_holes_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    is_promoted_artificial_voxel_ = promoted_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    exist_since_iter_ = exist_since_after_cpu.to(device_type_).to(torch::kInt32).contiguous();
    exist_since_kf_ = exist_since_kf_after_cpu.to(device_type_).to(torch::kInt32).contiguous();
    geometrically_unstable_voxel_ = unstable_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    rendered_depth_candidate_voxel_ =
        rendered_depth_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    rendered_depth_candidate_support_count_ =
        rendered_depth_support_after_cpu.to(device_type_).to(torch::kInt32).contiguous();
    rendered_depth_candidate_last_seen_kf_ =
        rendered_depth_last_seen_after_cpu.to(device_type_).to(torch::kInt32).contiguous();
    rendered_depth_candidate_source_kind_ =
        rendered_depth_source_after_cpu.to(device_type_).to(torch::kInt32).contiguous();
    const int64_t n_art_after = is_artificial_voxel_.sum().item<int64_t>();
    const int64_t n_promoted_after = is_promoted_artificial_voxel_.sum().item<int64_t>();
    (void)n_prune_total;
    (void)n_prune_art;
    (void)n_prune_real;
    (void)n_art_before;
    (void)n_art_after;
    (void)n_promoted_before;
    (void)n_promoted_after;
}

void VoxelModel::subdividing(const torch::Tensor& subdivide_mask) {
    py::gil_scoped_acquire gil;
    auto mask = subdivide_mask.to(torch::kBool).to(device_type_).contiguous();
    if (mask.dim() == 2 && mask.size(1) == 1) {
        mask = mask.squeeze(1);
    }
    TORCH_CHECK(mask.dim() == 1, "subdividing: mask must be [N] or [N,1]");
    const int64_t N_before = mask.size(0);

    if (!is_artificial_voxel_.defined() || is_artificial_voxel_.size(0) != N_before) {
        is_artificial_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (is_artificial_voxel_.device() != mask.device()) {
        is_artificial_voxel_ = is_artificial_voxel_.to(mask.device());
    }
    if (!is_orb_voxel_.defined() || is_orb_voxel_.size(0) != N_before) {
        is_orb_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (is_orb_voxel_.device() != mask.device()) {
        is_orb_voxel_ = is_orb_voxel_.to(mask.device());
    }
    if (!is_inactive_geo_voxel_.defined() || is_inactive_geo_voxel_.size(0) != N_before) {
        is_inactive_geo_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (is_inactive_geo_voxel_.device() != mask.device()) {
        is_inactive_geo_voxel_ = is_inactive_geo_voxel_.to(mask.device());
    }
    if (!is_rgbd_fill_render_holes_voxel_.defined() ||
        is_rgbd_fill_render_holes_voxel_.size(0) != N_before) {
        is_rgbd_fill_render_holes_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (is_rgbd_fill_render_holes_voxel_.device() != mask.device()) {
        is_rgbd_fill_render_holes_voxel_ = is_rgbd_fill_render_holes_voxel_.to(mask.device());
    }
    if (!is_depthanything_fill_holes_voxel_.defined() ||
        is_depthanything_fill_holes_voxel_.size(0) != N_before) {
        is_depthanything_fill_holes_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (is_depthanything_fill_holes_voxel_.device() != mask.device()) {
        is_depthanything_fill_holes_voxel_ =
            is_depthanything_fill_holes_voxel_.to(mask.device());
    }
    if (!is_promoted_artificial_voxel_.defined() || is_promoted_artificial_voxel_.size(0) != N_before) {
        is_promoted_artificial_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (is_promoted_artificial_voxel_.device() != mask.device()) {
        is_promoted_artificial_voxel_ = is_promoted_artificial_voxel_.to(mask.device());
    }
    if (!exist_since_iter_.defined() || exist_since_iter_.size(0) != N_before) {
        exist_since_iter_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
    } else if (exist_since_iter_.device() != mask.device()) {
        exist_since_iter_ = exist_since_iter_.to(mask.device());
    }
    if (!exist_since_kf_.defined() || exist_since_kf_.size(0) != N_before) {
        exist_since_kf_ = torch::full(
            {N_before},
            static_cast<int32_t>(-1),
            torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
    } else if (exist_since_kf_.device() != mask.device()) {
        exist_since_kf_ = exist_since_kf_.to(mask.device());
    }
    if (!geometrically_unstable_voxel_.defined() || geometrically_unstable_voxel_.size(0) != N_before) {
        geometrically_unstable_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (geometrically_unstable_voxel_.device() != mask.device()) {
        geometrically_unstable_voxel_ = geometrically_unstable_voxel_.to(mask.device());
    }
    if (!rendered_depth_candidate_voxel_.defined() ||
        rendered_depth_candidate_voxel_.size(0) != N_before) {
        rendered_depth_candidate_voxel_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    } else if (rendered_depth_candidate_voxel_.device() != mask.device()) {
        rendered_depth_candidate_voxel_ = rendered_depth_candidate_voxel_.to(mask.device());
    }
    if (!rendered_depth_candidate_support_count_.defined() ||
        rendered_depth_candidate_support_count_.size(0) != N_before) {
        rendered_depth_candidate_support_count_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
    } else if (rendered_depth_candidate_support_count_.device() != mask.device()) {
        rendered_depth_candidate_support_count_ = rendered_depth_candidate_support_count_.to(mask.device());
    }
    if (!rendered_depth_candidate_last_seen_kf_.defined() ||
        rendered_depth_candidate_last_seen_kf_.size(0) != N_before) {
        rendered_depth_candidate_last_seen_kf_ = torch::full(
            {N_before},
            static_cast<int32_t>(-1),
            torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
    } else if (rendered_depth_candidate_last_seen_kf_.device() != mask.device()) {
        rendered_depth_candidate_last_seen_kf_ = rendered_depth_candidate_last_seen_kf_.to(mask.device());
    }
    if (!rendered_depth_candidate_source_kind_.defined() ||
        rendered_depth_candidate_source_kind_.size(0) != N_before) {
        rendered_depth_candidate_source_kind_ = torch::zeros(
            {N_before},
            torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
    } else if (rendered_depth_candidate_source_kind_.device() != mask.device()) {
        rendered_depth_candidate_source_kind_ = rendered_depth_candidate_source_kind_.to(mask.device());
    }

    auto art_before = is_artificial_voxel_.to(torch::kBool).contiguous();
    auto orb_before = is_orb_voxel_.to(torch::kBool).contiguous();
    auto inactive_geo_before = is_inactive_geo_voxel_.to(torch::kBool).contiguous();
    auto rgbd_fill_render_holes_before =
        is_rgbd_fill_render_holes_voxel_.to(torch::kBool).contiguous();
    auto depthanything_fill_holes_before =
        is_depthanything_fill_holes_voxel_.to(torch::kBool).contiguous();
    auto promoted_before = is_promoted_artificial_voxel_.to(torch::kBool).contiguous();
    auto exist_since_before = exist_since_iter_.to(torch::kInt32).contiguous();
    auto exist_since_kf_before = exist_since_kf_.to(torch::kInt32).contiguous();
    auto unstable_before = geometrically_unstable_voxel_.to(torch::kBool).contiguous();
    auto rendered_depth_before = rendered_depth_candidate_voxel_.to(torch::kBool).contiguous();
    auto rendered_depth_support_before = rendered_depth_candidate_support_count_.to(torch::kInt32).contiguous();
    auto rendered_depth_last_seen_before = rendered_depth_candidate_last_seen_kf_.to(torch::kInt32).contiguous();
    const int64_t n_art_before = art_before.sum().item<int64_t>();
    const int64_t n_promoted_before = promoted_before.sum().item<int64_t>();

    auto subdiv_idx = torch::nonzero(mask).view({-1});
    const int64_t n_subdiv_parents = subdiv_idx.size(0);
    int64_t n_subdiv_art_parents = 0;
    int64_t n_subdiv_promoted_parents = 0;
    if (n_subdiv_parents > 0) {
        auto art_sel = art_before.index_select(0, subdiv_idx).contiguous();
        auto promoted_sel = promoted_before.index_select(0, subdiv_idx).contiguous();
        n_subdiv_art_parents = art_sel.sum().item<int64_t>();
        n_subdiv_promoted_parents = promoted_sel.sum().item<int64_t>();
    }

    auto old_octpath = py_->svm.attr("octpath").cast<torch::Tensor>().contiguous();
    auto old_octlevel = py_->svm.attr("octlevel").cast<torch::Tensor>().contiguous();
    auto old_key_cpu = (old_octpath.view({-1}).to(torch::kInt64).mul(256)
                      + old_octlevel.view({-1}).to(torch::kInt64))
                      .to(torch::kCPU).contiguous();
    auto old_art_cpu = art_before.to(torch::kCPU).contiguous();
    auto old_orb_cpu = orb_before.to(torch::kCPU).contiguous();
    auto old_inactive_geo_cpu = inactive_geo_before.to(torch::kCPU).contiguous();
    auto old_rgbd_fill_render_holes_cpu =
        rgbd_fill_render_holes_before.to(torch::kCPU).contiguous();
    auto old_depthanything_fill_holes_cpu =
        depthanything_fill_holes_before.to(torch::kCPU).contiguous();
    auto old_promoted_cpu = promoted_before.to(torch::kCPU).contiguous();
    auto old_exist_since_cpu = exist_since_before.to(torch::kCPU).contiguous();
    auto old_exist_since_kf_cpu = exist_since_kf_before.to(torch::kCPU).contiguous();
    auto old_unstable_cpu = unstable_before.to(torch::kCPU).contiguous();
    auto old_rendered_depth_cpu = rendered_depth_before.to(torch::kCPU).contiguous();
    auto old_rendered_depth_support_cpu = rendered_depth_support_before.to(torch::kCPU).contiguous();
    auto old_rendered_depth_last_seen_cpu = rendered_depth_last_seen_before.to(torch::kCPU).contiguous();
    auto old_rendered_depth_source_cpu =
        rendered_depth_candidate_source_kind_.to(torch::kCPU).to(torch::kInt32).contiguous();

    py_->svm.attr("subdividing")(mask);
    syncFromPython_();

    auto new_key_cpu = (this->oct_path_.view({-1}).to(torch::kInt64).mul(256)
                      + this->oct_level_.view({-1}).to(torch::kInt64))
                      .to(torch::kCPU).contiguous();
    const int64_t N_after = new_key_cpu.size(0);

    auto art_after_cpu = torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto orb_after_cpu = torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto inactive_geo_after_cpu = torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto rgbd_fill_render_holes_after_cpu =
        torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto depthanything_fill_holes_after_cpu =
        torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto promoted_after_cpu = torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto exist_since_after_cpu = torch::zeros({N_after}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto exist_since_kf_after_cpu = torch::full(
        {N_after}, static_cast<int32_t>(-1),
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto unstable_after_cpu = torch::zeros(
        {N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto rendered_depth_after_cpu = torch::zeros(
        {N_after}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto rendered_depth_support_after_cpu = torch::zeros(
        {N_after}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto rendered_depth_last_seen_after_cpu = torch::full(
        {N_after}, static_cast<int32_t>(-1),
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto rendered_depth_source_after_cpu = torch::zeros(
        {N_after}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));

    std::unordered_map<int64_t, int64_t> key_to_old_idx;
    key_to_old_idx.reserve(static_cast<size_t>(old_key_cpu.size(0) * 2 + 1));
    const int64_t* old_key_ptr = old_key_cpu.data_ptr<int64_t>();
    for (int64_t i = 0; i < old_key_cpu.size(0); ++i) {
        key_to_old_idx.emplace(old_key_ptr[i], i);
    }

    const bool* old_art_ptr = old_art_cpu.data_ptr<bool>();
    const bool* old_orb_ptr = old_orb_cpu.data_ptr<bool>();
    const bool* old_inactive_geo_ptr = old_inactive_geo_cpu.data_ptr<bool>();
    const bool* old_rgbd_fill_render_holes_ptr = old_rgbd_fill_render_holes_cpu.data_ptr<bool>();
    const bool* old_depthanything_fill_holes_ptr = old_depthanything_fill_holes_cpu.data_ptr<bool>();
    const bool* old_promoted_ptr = old_promoted_cpu.data_ptr<bool>();
    const int32_t* old_exist_since_ptr = old_exist_since_cpu.data_ptr<int32_t>();
    const int32_t* old_exist_since_kf_ptr = old_exist_since_kf_cpu.data_ptr<int32_t>();
    const bool* old_unstable_ptr = old_unstable_cpu.data_ptr<bool>();
    const bool* old_rendered_depth_ptr = old_rendered_depth_cpu.data_ptr<bool>();
    const int32_t* old_rendered_depth_support_ptr = old_rendered_depth_support_cpu.data_ptr<int32_t>();
    const int32_t* old_rendered_depth_last_seen_ptr = old_rendered_depth_last_seen_cpu.data_ptr<int32_t>();
    const int32_t* old_rendered_depth_source_ptr = old_rendered_depth_source_cpu.data_ptr<int32_t>();
    const int64_t* new_key_ptr = new_key_cpu.data_ptr<int64_t>();
    bool* art_after_ptr = art_after_cpu.data_ptr<bool>();
    bool* orb_after_ptr = orb_after_cpu.data_ptr<bool>();
    bool* inactive_geo_after_ptr = inactive_geo_after_cpu.data_ptr<bool>();
    bool* rgbd_fill_render_holes_after_ptr = rgbd_fill_render_holes_after_cpu.data_ptr<bool>();
    bool* depthanything_fill_holes_after_ptr = depthanything_fill_holes_after_cpu.data_ptr<bool>();
    bool* promoted_after_ptr = promoted_after_cpu.data_ptr<bool>();
    int32_t* exist_since_after_ptr = exist_since_after_cpu.data_ptr<int32_t>();
    int32_t* exist_since_kf_after_ptr = exist_since_kf_after_cpu.data_ptr<int32_t>();
    bool* unstable_after_ptr = unstable_after_cpu.data_ptr<bool>();
    bool* rendered_depth_after_ptr = rendered_depth_after_cpu.data_ptr<bool>();
    int32_t* rendered_depth_support_after_ptr = rendered_depth_support_after_cpu.data_ptr<int32_t>();
    int32_t* rendered_depth_last_seen_after_ptr = rendered_depth_last_seen_after_cpu.data_ptr<int32_t>();
    int32_t* rendered_depth_source_after_ptr = rendered_depth_source_after_cpu.data_ptr<int32_t>();

    int64_t matched_exact = 0;
    int64_t matched_parent = 0;
    int64_t unmatched = 0;
    for (int64_t i = 0; i < N_after; ++i) {
        const int64_t key = new_key_ptr[i];
        auto it = key_to_old_idx.find(key);
        if (it != key_to_old_idx.end()) {
            const int64_t old_idx = it->second;
            art_after_ptr[i] = old_art_ptr[old_idx];
            orb_after_ptr[i] = old_orb_ptr[old_idx];
            inactive_geo_after_ptr[i] = old_inactive_geo_ptr[old_idx];
            rgbd_fill_render_holes_after_ptr[i] = old_rgbd_fill_render_holes_ptr[old_idx];
            depthanything_fill_holes_after_ptr[i] = old_depthanything_fill_holes_ptr[old_idx];
            promoted_after_ptr[i] = old_promoted_ptr[old_idx];
            exist_since_after_ptr[i] = old_exist_since_ptr[old_idx];
            exist_since_kf_after_ptr[i] = old_exist_since_kf_ptr[old_idx];
            unstable_after_ptr[i] = old_unstable_ptr[old_idx];
            rendered_depth_after_ptr[i] = old_rendered_depth_ptr[old_idx];
            rendered_depth_support_after_ptr[i] = old_rendered_depth_support_ptr[old_idx];
            rendered_depth_last_seen_after_ptr[i] = old_rendered_depth_last_seen_ptr[old_idx];
            rendered_depth_source_after_ptr[i] = old_rendered_depth_source_ptr[old_idx];
            ++matched_exact;
            continue;
        }

        const int64_t lv = key & 255LL;
        const int64_t path = key / 256LL;
        if (lv > 1) {
            const int parent_lv = static_cast<int>(lv - 1);
            const int levels_below_parent = std::max(0, max_num_levels_ - parent_lv);
            const int bits_to_clear = 3 * levels_below_parent;
            const int64_t lower_mask = (bits_to_clear > 0) ? ((1LL << bits_to_clear) - 1LL) : 0LL;
            const int64_t keep_mask = ~lower_mask;
            const int64_t parent_path = path & keep_mask;
            const int64_t parent_key = parent_path * 256LL + static_cast<int64_t>(parent_lv);

            auto it_parent = key_to_old_idx.find(parent_key);
            if (it_parent != key_to_old_idx.end()) {
                const int64_t old_idx = it_parent->second;
                art_after_ptr[i] = old_art_ptr[old_idx];
                orb_after_ptr[i] = old_orb_ptr[old_idx];
                inactive_geo_after_ptr[i] = old_inactive_geo_ptr[old_idx];
                rgbd_fill_render_holes_after_ptr[i] = old_rgbd_fill_render_holes_ptr[old_idx];
                depthanything_fill_holes_after_ptr[i] = old_depthanything_fill_holes_ptr[old_idx];
                promoted_after_ptr[i] = old_promoted_ptr[old_idx];
                if (topology_birth_iter_ >= 0) {
                    exist_since_after_ptr[i] = topology_birth_iter_;
                } else {
                    exist_since_after_ptr[i] = old_exist_since_ptr[old_idx];
                }
                if (topology_birth_kf_ >= 0) {
                    exist_since_kf_after_ptr[i] = topology_birth_kf_;
                } else {
                    exist_since_kf_after_ptr[i] = old_exist_since_kf_ptr[old_idx];
                }
                unstable_after_ptr[i] = old_unstable_ptr[old_idx];
                rendered_depth_after_ptr[i] = old_rendered_depth_ptr[old_idx];
                rendered_depth_support_after_ptr[i] = old_rendered_depth_support_ptr[old_idx];
                rendered_depth_last_seen_after_ptr[i] = old_rendered_depth_last_seen_ptr[old_idx];
                rendered_depth_source_after_ptr[i] = old_rendered_depth_source_ptr[old_idx];
                ++matched_parent;
                continue;
            }
        }

        ++unmatched;
    }

    // if (unmatched > 0) {
    //     std::cout << "[SUBDIV/provenance] WARNING: unmatched voxels after key/parent remap: "
    //               << unmatched << "/" << N_after << "\n";
    // }

    is_artificial_voxel_ = art_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    is_orb_voxel_ = orb_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    is_inactive_geo_voxel_ = inactive_geo_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    is_rgbd_fill_render_holes_voxel_ =
        rgbd_fill_render_holes_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    is_depthanything_fill_holes_voxel_ =
        depthanything_fill_holes_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    is_promoted_artificial_voxel_ = promoted_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    exist_since_iter_ = exist_since_after_cpu.to(device_type_).to(torch::kInt32).contiguous();
    exist_since_kf_ = exist_since_kf_after_cpu.to(device_type_).to(torch::kInt32).contiguous();
    geometrically_unstable_voxel_ = unstable_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    rendered_depth_candidate_voxel_ =
        rendered_depth_after_cpu.to(device_type_).to(torch::kBool).contiguous();
    rendered_depth_candidate_support_count_ =
        rendered_depth_support_after_cpu.to(device_type_).to(torch::kInt32).contiguous();
    rendered_depth_candidate_last_seen_kf_ =
        rendered_depth_last_seen_after_cpu.to(device_type_).to(torch::kInt32).contiguous();
    rendered_depth_candidate_source_kind_ =
        rendered_depth_source_after_cpu.to(device_type_).to(torch::kInt32).contiguous();
    const int64_t n_art_after = is_artificial_voxel_.sum().item<int64_t>();
    const int64_t n_promoted_after = is_promoted_artificial_voxel_.sum().item<int64_t>();
    // std::cout << "[SUBDIV/artificial] N_before=" << N_before
    //           << " subdiv_parents=" << n_subdiv_parents
    //           << " subdiv_art_parents=" << n_subdiv_art_parents
    //           << " subdiv_promoted_parents=" << n_subdiv_promoted_parents
    //           << " matched_exact=" << matched_exact
    //           << " matched_parent=" << matched_parent
    //           << " unmatched=" << unmatched
    //           << " N_after=" << center_.size(0)
    //           << " art_before=" << n_art_before
    //           << " art_after=" << n_art_after
    //           << " promoted_before=" << n_promoted_before
    //           << " promoted_after=" << n_promoted_after
    //           << "\n";

    (void)n_subdiv_parents;
    (void)n_subdiv_art_parents;
    (void)n_subdiv_promoted_parents;
    (void)matched_exact;
    (void)matched_parent;
    (void)unmatched;
    (void)n_art_before;
    (void)n_art_after;
    (void)n_promoted_before;
    (void)n_promoted_after;
}

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

    const int64_t N = center_.size(0);
    auto care_idx = torch::arange(N, torch::dtype(torch::kLong).device(device_type_));

    auto torch_mod = py::module_::import("torch");
    auto no_grad   = torch_mod.attr("no_grad")();
    no_grad.attr("__enter__")();
    try {
        auto renderer = py::module_::import("svraster_cuda.renderer");
        auto Gather   = renderer.attr("GatherGeoParams");
        // returns a torch.Tensor
        py::object py_frozen = Gather.attr("apply")(vox_key_, care_idx, _geo_grid_pts_);
        frozen_vox_geo_ = py_frozen.cast<torch::Tensor>().contiguous();
        _geo_grid_pts_.set_requires_grad(false);
        no_grad.attr("__exit__")(py::none(), py::none(), py::none());
    } catch (...) {
        no_grad.attr("__exit__")(py::none(), py::none(), py::none());
        throw;
    }
}

void VoxelModel::unfreezeVoxGeo() {
    py::gil_scoped_acquire gil;
    frozen_vox_geo_.reset();              // make it undefined
    _geo_grid_pts_.set_requires_grad(true);
}

static inline py::dict make_kwargs_from(
    const sv::RenderOpts& a,
    // explicit function params (after color_mode)
    bool track_max_w,
    std::optional<float> ss,
    bool output_depth,
    bool output_normal,
    bool output_T,
    bool rand_bg,
    bool use_auto_exposure)
{
    py::dict kw;

    // 1) Start with struct — these become **other_opt
    if (a.lambda_dist.has_value())       kw["lambda_dist"]      = *a.lambda_dist;
    if (a.lambda_ascending.has_value())  kw["lambda_ascending"] = *a.lambda_ascending;
    if (a.lambda_scaling_penalty.has_value()) kw["lambda_scaling_penalty"] = *a.lambda_scaling_penalty;
    if (a.min_voxel_size.has_value())    kw["min_voxel_size"]   = *a.min_voxel_size;
    if (a.lambda_R_concen.has_value())  kw["lambda_R_concen"] = *a.lambda_R_concen;
    if (a.gt_color.defined())            kw["gt_color"]         = a.gt_color;
    if (a.vox_feats.defined())           kw["vox_feats"]        = a.vox_feats;
    // If you have other non-overlapping keys (e.g., n_samp_per_vox), add them here as well:
    // if (a.n_samp_per_vox.has_value()) kw["n_samp_per_vox"] = *a.n_samp_per_vox;

    // Overlapping names: add **only** to kwargs (we won’t pass them positionally)
    // struct values (if set) first…
    if (a.track_max_w)        kw["track_max_w"]       = true;
    if (a.ss.has_value())     kw["ss"]                = *a.ss;
    if (a.output_depth)       kw["output_depth"]      = true;
    if (a.output_normal)      kw["output_normal"]     = true;
    if (a.output_T)           kw["output_T"]          = true;
    if (a.rand_bg)            kw["rand_bg"]           = true;
    if (a.use_auto_exposure)  kw["use_auto_exposure"] = true;

    // …then override with explicit function arguments if caller set them.
    if (track_max_w)          kw["track_max_w"]       = true;
    if (ss.has_value())       kw["ss"]                = *ss;
    if (output_depth)         kw["output_depth"]      = true;
    if (output_normal)        kw["output_normal"]     = true;
    if (output_T)             kw["output_T"]          = true;
    if (rand_bg)              kw["rand_bg"]           = true;
    if (use_auto_exposure)    kw["use_auto_exposure"] = true;

    return kw;
}

// std::unordered_map<std::string, torch::Tensor> VoxelModel::render(const MiniCam& cam, torch::Tensor gt_image, int image_height, int image_width, float ss, bool track_max_w) const
std::unordered_map<std::string, torch::Tensor> VoxelModel::render(
    const sv::MiniCam& cam,
    int im_height,
    int im_width,
    const torch::Tensor& gt_image,
    const char* color_mode,
    bool track_max_w,
    std::optional<float> ss,
    bool output_depth,
    bool output_normal,
    bool output_T,
    bool rand_bg,
    bool use_auto_exposure,
    const sv::RenderOpts& other_opt) const
{
    /* 1) import the python entry-point once                                 */
    py::gil_scoped_acquire gil;
    static py::object py_render;
    if (!py_render) {
        try {
            py_render = py::module_::import(
                            "scripts_voxel.python_svraster_bridge.renderer_wrapper")
                            .attr("render");
            // std::cerr << "[INFO] Python-side renderer imported OK.\n";
        } catch (const py::error_already_set& e) {
            std::cerr << "[PYBIND11] Could not import renderer_wrapper:\n"
                      << e.what() << std::endl;
            return {};
        }
    }

    py::dict d;
    d["_geo_grid_pts"]      = this->_geo_grid_pts_;
    d["sh0"]               = this->sh0_;
    d["shs"]               = this->shs_;
    d["subdiv_p"]          = this->subdiv_p_;
    d["octpath"]           = this->oct_path_;
    d["octlevel"]          = this->oct_level_;
    d["center"]            = this->center_;
    d["vox_size"]          = this->size_;
    d["vox_key"]           = this->vox_key_;
    d["active_sh_degree"]  = py::int_(this->active_sh_degree_);

    d["white_background"] = py::bool_(this->white_background_);
    d["black_background"] = py::bool_(this->black_background_);
    d["ss"]               = py::float_(this->ss_);       // default SS (used if ss=None)
    d["n_samp_per_vox"]   = py::int_(this->n_samp_per_vox_);
    d["num_voxels"]        = static_cast<long long>(this->center_.size(0));

    // Optional: expose pre-gathered geo if present
    if (this->frozen_vox_geo_.defined() && this->frozen_vox_geo_.numel() > 0) {
        d["frozen_vox_geo"] = this->frozen_vox_geo_;
    }

    // Build kwargs (**other_opt). IMPORTANT: do NOT pass overlapping args positionally.
    py::dict kwargs = make_kwargs_from(
        other_opt, track_max_w, ss, output_depth, output_normal,
        output_T, rand_bg, use_auto_exposure);
    // std::cout << "kwargs" << kwargs << std::endl;   
    /* 4) call python                                                        */
    py::object py_cam  = MiniCam_to_py(cam);
    py::object py_out;
    py::tuple args = py::make_tuple(
        d,
        py_cam,
        im_height,
        im_width,
        gt_image,
        color_mode
    ); 
    try {
        py_out = py_render(*args, **kwargs);
        // py_out = py_render(
        //     d,
        //     py_cam,
        //     im_height,
        //     im_width,
        //     gt_image,
        //     color_mode,
        //     py::kwargs(kwargs) );
        // py_out = py_render(py_cam, d, gt_image, image_height, image_width, ss, track_max_w);
    } catch (const py::error_already_set& e) {
        std::cerr << "[PYBIND11] Exception thrown by renderer:\n"
                  << e.what() << std::endl;
        return {};
    }
    py::dict out_dict = py_out.cast<py::dict>();

    /* 6) copy every tensor into a C++ map                                   */
    std::unordered_map<std::string, torch::Tensor> pkg;
    for (auto item : out_dict) {
        const std::string key = py::str(item.first);
        torch::Tensor t;
        bool is_tensor = true;
        try {
            t = item.second.cast<torch::Tensor>();
        } catch (...) {
            is_tensor = false;
        }
        if (t.defined())
            pkg.emplace(key, std::move(t));
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

static inline uint64_t encode_ijk(int64_t i, int64_t j, int64_t k, int lv) {
    uint64_t p = 0;
    for (int l = lv - 1; l >= 0; --l) {
        const uint64_t b2 = (static_cast<uint64_t>(i >> l) & 0x1ull);
        const uint64_t b1 = (static_cast<uint64_t>(j >> l) & 0x1ull);
        const uint64_t b0 = (static_cast<uint64_t>(k >> l) & 0x1ull);
        p = (p << 3) | (b2 << 2) | (b1 << 1) | b0;
    }
    p <<= (3 * (MAX_NUM_LEVELS - lv));
    return p;
}

struct VoxelOccKey {
    uint64_t path;
    uint8_t level;
    bool operator==(const VoxelOccKey& o) const {
        return path == o.path && level == o.level;
    }
};

struct VoxelOccKeyHash {
    size_t operator()(const VoxelOccKey& k) const {
        // FNV-like mix
        uint64_t h = 1469598103934665603ull;
        h ^= k.path;  h *= 1099511628211ull;
        h ^= static_cast<uint64_t>(k.level); h *= 1099511628211ull;
        return static_cast<size_t>(h);
    }
};

struct GridCornerKey {
    int64_t x;
    int64_t y;
    int64_t z;
    bool operator==(const GridCornerKey& o) const {
        return x == o.x && y == o.y && z == o.z;
    }
};

struct GridCornerKeyHash {
    size_t operator()(const GridCornerKey& k) const {
        uint64_t h = 1469598103934665603ull;
        h ^= static_cast<uint64_t>(k.x * 0x9e3779b97f4a7c15ull); h *= 1099511628211ull;
        h ^= static_cast<uint64_t>(k.y * 0xbf58476d1ce4e5b9ull); h *= 1099511628211ull;
        h ^= static_cast<uint64_t>(k.z * 0x94d049bb133111ebull); h *= 1099511628211ull;
        return static_cast<size_t>(h);
    }
};

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

void VoxelModel::savePlannerNPZ(const std::filesystem::path& npz_path,
                               int target_max_voxels /*=1000000*/)
{
    torch::NoGradGuard ng;
    namespace fs = std::filesystem;
    namespace py = pybind11;

    if (!npz_path.parent_path().empty())
        fs::create_directories(npz_path.parent_path());

    // We export the *SVRaster model state* from the embedded Python SparseVoxelModel.
    py::gil_scoped_acquire gil;

    if (!py_ || py_->svm.is_none()) {
        std::cerr << "[savePlannerNPZ] ERROR: py_->svm is None. Nothing to export.\n";
        return;
    }

    py::object svm = py_->svm;
    py::module_ np = py::module_::import("numpy");

    auto to_numpy = [](py::handle t) -> py::object {
        // Expect torch.Tensor-like objects.
        py::object x = py::reinterpret_borrow<py::object>(t);

        if (py::hasattr(x, "detach")) x = x.attr("detach")();
        if (py::hasattr(x, "contiguous")) x = x.attr("contiguous")();
        if (py::hasattr(x, "cpu")) x = x.attr("cpu")();
        if (py::hasattr(x, "numpy")) return x.attr("numpy")();

        throw std::runtime_error("to_numpy(): object has no .numpy() (not a torch.Tensor?)");
    };

    // ---- Required SVRaster state (needed to render) ----
    // These attributes exist in your pipeline because you assign them in createFromPcd().
    py::object scene_center   = to_numpy(svm.attr("scene_center"));
    py::object scene_extent   = to_numpy(svm.attr("scene_extent"));
    py::object inside_extent  = to_numpy(svm.attr("inside_extent"));

    py::object octpath        = to_numpy(svm.attr("octpath"));
    py::object octlevel       = to_numpy(svm.attr("octlevel"));

    py::object geo_grid_pts   = to_numpy(svm.attr("_geo_grid_pts"));
    py::object sh0            = to_numpy(svm.attr("_sh0"));
    py::object shs            = to_numpy(svm.attr("_shs"));
    py::object subdiv_p       = to_numpy(svm.attr("_subdiv_p"));

    int active_sh_degree = 0;
    int max_sh_degree    = this->max_sh_degree_;
    try { active_sh_degree = svm.attr("active_sh_degree").cast<int>(); } catch (...) {}
    try { max_sh_degree    = svm.attr("max_sh_degree").cast<int>(); }    catch (...) {}

    // ---- Optional planner caches (useful for quick box/occupancy visualizations) ----
    // We export these too, but they are *not* sufficient to render SVRaster by themselves.
    py::object vox_center   = py::none();
    py::object vox_halfsize = py::none();
    py::object vox_level_u8 = py::none();

    try {
        // vox_center: [N,3] float
        py::object vc_t = svm.attr("vox_center");
        // vox_size:   [N,1] float (or [N])
        py::object vs_t = svm.attr("vox_size");

        // Convert to torch tensors on CPU first, then numpy.
        // We also compute half-size.
        py::object vc_cpu = vc_t.attr("detach")().attr("contiguous")().attr("cpu")();
        py::object vs_cpu = vs_t.attr("detach")().attr("contiguous")().attr("cpu")();

        // half-size = 0.5 * vox_size
        py::object half_cpu = py::module_::import("torch").attr("mul")(vs_cpu, 0.5);

        // level as uint8 for planner
        py::object lv_cpu = svm.attr("octlevel").attr("detach")().attr("contiguous")().attr("cpu")();
        py::object lv_u8  = lv_cpu.attr("to")(py::module_::import("torch").attr("uint8"));

        // NOTE: We intentionally DO NOT downsample the model state here.
        // If you want downsampling for planner only, do it downstream in Python/OMPL code.

        vox_center   = vc_cpu.attr("numpy")();
        vox_halfsize = half_cpu.attr("numpy")();
        vox_level_u8 = lv_u8.attr("numpy")();
    } catch (...) {
        // If vox_center/vox_size are not available, we still export the full SVRaster state.
    }

    try {
        np.attr("savez_compressed")(
            npz_path.string(),
            // --- SVRaster reconstruction keys ---
            py::arg("scene_center")      = scene_center,
            py::arg("scene_extent")      = scene_extent,
            py::arg("inside_extent")     = inside_extent,
            py::arg("octpath")           = octpath,
            py::arg("octlevel")          = octlevel,
            py::arg("geo_grid_pts")      = geo_grid_pts,
            py::arg("sh0")               = sh0,
            py::arg("shs")               = shs,
            py::arg("subdiv_p")          = subdiv_p,
            py::arg("active_sh_degree")  = active_sh_degree,
            py::arg("max_sh_degree")     = max_sh_degree,
            // --- Optional planner caches ---
            py::arg("vox_center")        = vox_center,
            py::arg("vox_half_size")     = vox_halfsize,
            py::arg("vox_level_u8")      = vox_level_u8
        );

        std::cout << "[savePlannerNPZ] Wrote SVRaster model state to " << npz_path << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "[savePlannerNPZ] ERROR: numpy savez_compressed failed: " << e.what() << "\n";
    }
}

} // namespace sv
