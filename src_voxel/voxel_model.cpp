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
    // std::cout << std::fixed << std::setprecision(6)
    //           << "[AABB " << name << "]  "
    //           << "min=(" << mn[0] << "," << mn[1] << "," << mn[2] << ")  "
    //           << "max=(" << mx[0] << "," << mx[1] << "," << mx[2] << ")  "
    //           << "size=(" << sx << "," << sy << "," << sz << ")  "
    //           << "vol=" << vol << "\n";
}

// static inline std::string rrTimestamp() {
//     std::time_t t = std::time(nullptr);
//     std::tm tm;
//     localtime_r(&t, &tm);
//     char buf[32];
//     std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
//     return buf;
// }

// inline void sv::VoxelModel::rrInitOnce() {
//     if (rr_initialized_) return;
//     rr_ = std::make_unique<rerun::RecordingStream>("PhotoSLAM");
//     const std::string f = "/home/dimitris/Photo-SLAM/rerun/debug_run_" + rrTimestamp() + ".rrd";
//     rr_->save(f).exit_on_failure();  // offline
//     rr_initialized_ = true;
//     // optional: std::cout << "[Rerun] writing to " << f << "\n";
// }

// static inline std::string rrSanitize(std::string s) {
//     for (char& ch : s)
//         if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch=='_' || ch=='-' || ch=='/'))
//             ch = '_';
//     return s;
// }

// void sv::VoxelModel::rrLogPointsAndAABB(
//     int iteration,                        // only for label text
//     const at::Tensor& xyz,                // [N,3] (on any device)
//     const at::Tensor& rgb,                // [N,3] or empty
//     const at::Tensor& bounding_2x3,       // [2,3] min/max (CPU or GPU)
//     const std::string& tag,               // e.g., "init_raw", "init_used", "increase_raw", "increase_used"
//     int64_t max_points_for_viz,           // subsample cap; 0 => no subsampling
//     bool log_points,                      // log point cloud?
//     bool log_box,                         // log AABB?
//     int64_t inc                           // required: single time axis
// ){
//     rrInitOnce();

//     const std::string tag_safe = rrSanitize(tag);

//     // ---- Points ----
//     if (log_points && xyz.defined() && xyz.numel() >= 3) {
//         at::Tensor P = xyz.detach().to(at::kCPU).to(at::kFloat).contiguous();
//         if (P.dim()==2 && P.size(1)==3) {
//             at::Tensor C;
//             if (rgb.defined() && rgb.sizes()==P.sizes())
//                 C = rgb.detach().clamp(0,1).to(at::kCPU).to(at::kFloat).contiguous();
//             else
//                 C = at::ones({P.size(0),3}, P.options());

//             const int64_t N = P.size(0);
//             const int64_t stride = (max_points_for_viz>0 && N>max_points_for_viz)
//                                    ? ((N + max_points_for_viz - 1) / max_points_for_viz)
//                                    : 1;

//             std::vector<rerun::Vec3D> positions; positions.reserve((N + stride - 1) / stride);
//             std::vector<rerun::Rgba32> colors;    colors.reserve((N + stride - 1) / stride);

//             const float* p = P.data_ptr<float>();
//             const float* c = C.data_ptr<float>();
//             for (int64_t i=0; i<N; i+=stride) {
//                 positions.emplace_back(p[3*i+0], p[3*i+1], p[3*i+2]);
//                 colors.emplace_back(
//                     (uint8_t)std::lround(255.f * c[3*i+0]),
//                     (uint8_t)std::lround(255.f * c[3*i+1]),
//                     (uint8_t)std::lround(255.f * c[3*i+2]),
//                     255
//                 );
//             }

//             rr_->set_time_sequence("inc", inc);
//             rr_->log("world/points/" + tag_safe,
//                      rerun::Points3D(positions).with_colors(colors));
//         }
//     }

//     // ---- Box ----
//     if (log_box && bounding_2x3.defined() && bounding_2x3.numel()==6) {
//         at::Tensor B = bounding_2x3.detach().to(at::kCPU).to(at::kFloat).contiguous();
//         if (B.dim()==2 && B.size(0)==2 && B.size(1)==3) {
//             const float minx = B[0][0].item<float>();
//             const float miny = B[0][1].item<float>();
//             const float minz = B[0][2].item<float>();
//             const float maxx = B[1][0].item<float>();
//             const float maxy = B[1][1].item<float>();
//             const float maxz = B[1][2].item<float>();

//             const rerun::Vec3D mins  {minx, miny, minz};
//             const rerun::Vec3D sizes {std::max(1e-6f, maxx-minx),
//                                       std::max(1e-6f, maxy-miny),
//                                       std::max(1e-6f, maxz-minz)};

//             const bool is_raw = (tag.find("raw") != std::string::npos);
//             const rerun::Rgba32 col = is_raw ? rerun::Rgba32(255,0,0,255)   // raw = red
//                                              : rerun::Rgba32(0,255,0,255);  // used/union = green

//             std::ostringstream lbl; lbl << tag << "@" << iteration;

//             rr_->set_time_sequence("inc", inc);
//             rr_->log("world/boxes/" + tag_safe,
//                      rerun::Boxes3D::from_mins_and_sizes({mins}, {sizes})
//                         .with_labels({lbl.str()})
//                         .with_colors({col}));
//         }
//     }
// }

// void sv::VoxelModel::rrLogVoxelBoxes(
//     const at::Tensor& centers,     // [N,3], any device
//     const at::Tensor& size,        // [N],   any device (edge length per voxel)
//     int64_t max_boxes_for_viz,     // e.g., 20000
//     const std::string& tag,        // e.g., "vox_all", "vox_new"
//     int64_t inc
// ){
//     rrInitOnce();

//     if (!centers.defined() || !size.defined()) return;
//     at::Tensor C = centers.detach().to(at::kCPU).to(at::kFloat).contiguous();
//     at::Tensor S = size   .detach().to(at::kCPU).to(at::kFloat).contiguous();

//     TORCH_CHECK(C.dim()==2 && C.size(1)==3, "centers must be [N,3]");
//     TORCH_CHECK(S.dim()==1 && S.size(0)==C.size(0), "size must be [N]");

//     int64_t N = C.size(0);
//     int64_t stride = (max_boxes_for_viz>0 && N>max_boxes_for_viz)
//                      ? ((N + max_boxes_for_viz - 1) / max_boxes_for_viz)
//                      : 1;

//     std::vector<rerun::Vec3D> mins;  mins.reserve((N + stride - 1) / stride);
//     std::vector<rerun::Vec3D> sizes; sizes.reserve((N + stride - 1) / stride);

//     const float* pc = C.data_ptr<float>();
//     const float* ps = S.data_ptr<float>();
//     for (int64_t i=0; i<N; i+=stride) {
//         float sz = std::max(1e-6f, ps[i]);
//         float hx = 0.5f * sz;
//         mins.emplace_back(pc[3*i+0] - hx, pc[3*i+1] - hx, pc[3*i+2] - hx);
//         sizes.emplace_back(sz, sz, sz);
//     }

//     // color: orange for "new", translucent blue otherwise
//     const rerun::Rgba32 col = (tag.find("new") != std::string::npos)
//         ? rerun::Rgba32(255, 165,   0, 128)  // orange, semi
//         : rerun::Rgba32(  0, 128, 255,  64); // blue, more translucent

//     rr_->set_time_sequence("inc", inc);
//     rr_->log("world/voxels/" + rrSanitize(tag),
//              rerun::Boxes3D::from_mins_and_sizes(mins, sizes)
//                  .with_colors({col}));
// }

// inline void sv::VoxelModel::rrLogGlobalSceneAABB(int64_t inc)
// {
//     rrInitOnce();
//     const float e = global_scene_extent_;
//     const rerun::Vec3D mins {
//         global_scene_center_[0] - 0.5f*e,
//         global_scene_center_[1] - 0.5f*e,
//         global_scene_center_[2] - 0.5f*e,
//     };
//     const rerun::Vec3D sizes { std::max(1e-6f, e), std::max(1e-6f, e), std::max(1e-6f, e) };

//     rr_->set_time_sequence("inc", inc);
//     rr_->log("world/boxes/global_scene",
//              rerun::Boxes3D::from_mins_and_sizes({mins}, {sizes})
//                  .with_labels({"global_scene"})
//                  .with_colors({rerun::Rgba32(0,255,0,255)})); // solid green
// }

// void sv::VoxelModel::rrLogVoxelBoxesWithShColor(
//     const at::Tensor& centers,   // [N,3]
//     const at::Tensor& size,      // [N] or [N,1]
//     const at::Tensor& sh0,       // [N,3] (DC SH -> RGB)
//     int64_t max_points_for_viz,  // e.g. 50'000
//     const std::string& tag,      // e.g. "vox_all", "vox_pruned_final"
//     int64_t inc
// ) {
//     rrInitOnce();  // make sure rr_ is created (offline .rrd file)

//     if (!centers.defined() || !size.defined() || !sh0.defined())
//         return;

//     TORCH_CHECK(centers.dim() == 2 && centers.size(1) == 3,
//                 "centers must be [N,3]");
//     TORCH_CHECK(sh0.dim() == 2 && sh0.size(1) == 3,
//                 "sh0 must be [N,3]");

//     int64_t N = centers.size(0);
//     TORCH_CHECK(size.dim() == 1 || (size.dim() == 2 && size.size(1) == 1),
//                 "size must be [N] or [N,1]");
//     TORCH_CHECK(sh0.size(0) == N,
//                 "sh0 and centers must have same length");

//     if (N == 0) return;

//     // --- Move to CPU & prepare tensors --------------------------------------
//     at::Tensor C = centers.detach().to(at::kCPU).to(at::kFloat).contiguous(); // [N,3]
//     at::Tensor S = size   .detach().to(at::kCPU).to(at::kFloat).contiguous(); // [N] or [N,1]
//     if (S.dim() == 2 && S.size(1) == 1)
//         S = S.view({S.size(0)});  // [N]

//     // SH0 -> RGB via your Python helper
//     at::Tensor rgb_cpu;
//     {
//         py::gil_scoped_acquire gil;
//         static py::module act_mod =
//             py::module::import("src.utils.activation_utils");
//         at::Tensor sh0_cpu = sh0.detach()
//                                    .to(at::kCPU)
//                                    .to(at::kFloat)
//                                    .contiguous();  // [N,3]

//         py::object rgb_py = act_mod.attr("shzero2rgb")(py::cast(sh0_cpu));
//         rgb_cpu = rgb_py.cast<at::Tensor>().contiguous().to(at::kFloat); // [N,3]
//     }
//     TORCH_CHECK(rgb_cpu.dim() == 2 && rgb_cpu.size(0) == N && rgb_cpu.size(1) == 3,
//                 "shzero2rgb returned wrong shape");

//     // --- Subsample if needed -------------------------------------------------
//     int64_t stride = (max_points_for_viz > 0 && N > max_points_for_viz)
//                      ? ((N + max_points_for_viz - 1) / max_points_for_viz)
//                      : 1;

//     std::vector<rerun::datatypes::Vec3D> points;
//     std::vector<float> radii;
//     std::vector<rerun::datatypes::Rgba32> colors;

//     points.reserve((N + stride - 1) / stride);
//     radii .reserve((N + stride - 1) / stride);
//     colors.reserve((N + stride - 1) / stride);

//     const float* pc = C.data_ptr<float>();        // [N,3]
//     const float* ps = S.data_ptr<float>();        // [N]
//     const float* pr = rgb_cpu.data_ptr<float>();  // [N,3]

//     for (int64_t i = 0; i < N; i += stride) {
//         float sz = std::max(1e-6f, ps[i]);   // voxel edge length
//         float r  = sz * 0.5f;                // radius for visualization

//         float x = pc[3 * i + 0];
//         float y = pc[3 * i + 1];
//         float z = pc[3 * i + 2];

//         points.emplace_back(x, y, z);
//         radii.emplace_back(r);

//         float cr = std::clamp(pr[3 * i + 0], 0.0f, 1.0f);
//         float cg = std::clamp(pr[3 * i + 1], 0.0f, 1.0f);
//         float cb = std::clamp(pr[3 * i + 2], 0.0f, 1.0f);

//         colors.emplace_back(
//             static_cast<uint8_t>(std::lround(255.f * cr)),
//             static_cast<uint8_t>(std::lround(255.f * cg)),
//             static_cast<uint8_t>(std::lround(255.f * cb)),
//             200  // alpha
//         );
//     }

//     // --- Log as Points3D -----------------------------------------------------
//     rr_->set_time_sequence("inc", inc);

//     auto pts = rerun::archetypes::Points3D(points)
//                    .with_radii(radii)
//                    .with_colors(colors);

//     rr_->log(
//         "world/voxels/" + rrSanitize(tag),
//         pts
//     );
// }

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

void VoxelModel::createFromPcd(const std::map<point3D_id_t, Point3D>& pcd, const std::vector<sv::MiniCam>& cams)
{
    namespace py = pybind11;
    std::cout << "VoxelModel::createFromPcd() called with " << pcd.size() << " points.\n";

    TORCH_CHECK(global_scene_extent_ > 0.f, "global_scene_extent_ must be set (>0).");
    TORCH_CHECK(fixed_vox_size_   > 0.f, "fixed_vox_size_ must be set (>0).");
    TORCH_CHECK(max_sh_degree_ >= 0, "max_sh_degree_ must be >= 0.");

    const int N = static_cast<int>(pcd.size());
    if (N == 0) {
        std::cerr << "[createFromPcd] Empty PCD — nothing to do.\n";
        return;
    }

    // ------------------------------------------------------------------------
    // 0) Fixed global scene + desired voxel size
    // ------------------------------------------------------------------------
    auto dev = torch::kCUDA;
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
        static py::module torch_mod = py::module::import("torch");
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
    
    // Create/reuse SVM (DO NOT call model_init/points_init/...)
    if (!py_->svm || py_->svm.is_none()) {
        py::object SVM = svm_mod.attr("SparseVoxelModel");
        py_->svm = SVM(py::arg("sh_degree") = max_sh_degree_);
    }

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
    auto geo_grid = torch::full({grid_pts_key.size(0), 1}, -10.0f,
                                torch::dtype(torch::kFloat32).device(dev)).requires_grad_();
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

    // Register with your optimizer
    VOXEL_MODEL_TENSORS_TO_VEC

    // inc = 0 for the initial snapshot (or any monotonic counter you keep)
    // const int64_t inc0 = 0;
    // 1) Log the global fixed scene AABB once
    // rrLogGlobalSceneAABB(inc0);
    // 2) Log ALL voxels after creation
    // rrLogVoxelBoxes(this->center_, this->size_, /*max_boxes_for_viz=*/10000000000, "vox_all", inc0);
}

void VoxelModel::increasePcd(std::vector<float> pcd_full,
                             std::vector<float> colors,
                             const int /*iteration*/, const std::vector<sv::MiniCam>& cams)
{
    namespace py = pybind11;
    static int64_t inc_counter = 1;  // or make it a member if you prefer
    const int Nf = static_cast<int>(pcd_full.size());
    if (Nf < 3 || colors.size() < 3) return;
    const int N = Nf / 3;
    std::cout << "VoxelModel::increasePcd() called with " << N << " points).\n";
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
    ).clone().div_(255.0f);

    // Move to CUDA
    auto dev = torch::kCUDA;
    auto cpu = torch::kCPU;
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
    // rrLogPointsAndAABB(/*iteration=*/0, xyz, rgb, aabb,
    //                 /*tag=*/"increase_raw", /*cap=*/10000000000,
    //                 /*log_points=*/true, /*log_box=*/false, /*inc=*/inc_counter);

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
            py::gil_scoped_acquire gil;
            static py::module bound_utils = py::module::import("src.utils.bounding_utils");
            static py::module types       = py::module::import("types");
            static py::module torch_mod   = py::module::import("torch");

            // xyz_cpu is a CPU float32 tensor with shape [N,3]
            py::object ns = types.attr("SimpleNamespace")(
                "points"_a = py::cast(xyz_cpu.contiguous()).attr("numpy")()
            );
            py::tuple cr = bound_utils.attr("main_scene_bound_pcd_heuristic")(ns, py::float_(0.1f));
            // Extract center as a C++ vector (handles float32/float64 seamlessly)
            std::vector<double> c_vec = py::cast<std::vector<double>>(cr[0]);
            TORCH_CHECK(c_vec.size() == 3, "main_scene_bound_pcd_heuristic: center must have 3 elements");
            float radius = py::cast<float>(cr[1]);
            // Update global scene (cube side = 2*radius)
            global_scene_center_[0] = static_cast<float>(c_vec[0]);
            global_scene_center_[1] = static_cast<float>(c_vec[1]);
            global_scene_center_[2] = static_cast<float>(c_vec[2]);
            global_scene_extent_    = 2.0f * radius;
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
            for (int i = 0; i < N; ++i) {
                Point3D P;
                // world coords
                P.xyz_(0) = pcd_full[3*i + 0];
                P.xyz_(1) = pcd_full[3*i + 1];
                P.xyz_(2) = pcd_full[3*i + 2];
                // colors[] are the original 0..255 values (floats)
                P.color_(0) = colors[3*i + 0];
                P.color_(1) = colors[3*i + 1];
                P.color_(2) = colors[3*i + 2];
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

    // // ---- (NEW) SVR-style filtering with cameras ----
    // if (!cams.empty()) {
    //     py::gil_scoped_acquire gil;
    //     static py::module oct_utils = py::module::import("src.utils.octree_utils");
    //     static py::module svr_mod   = py::module::import("svraster_cuda").attr("renderer");
    //     static py::module torch_mod = py::module::import("torch");

    //     // Decode voxel centers/sizes
    //     py::tuple dec = oct_utils.attr("octpath_decoding")(
    //         py::cast(octpath_new),
    //         py::cast(L_u),
    //         py::cast(scene_center_.contiguous()),
    //         py::cast(scene_extent_.contiguous())
    //     );
    //     at::Tensor vox_center = dec[0].cast<at::Tensor>(); // [Nu,3] cuda
    //     at::Tensor vox_size   = dec[1].cast<at::Tensor>(); // [Nu,1] cuda

    //     // Build Python list of CUDA MiniCams
    //     py::list py_cams;
    //     py::object py_cuda = torch_mod.attr("device")("cuda");
    //     auto move_attr_to_cuda_if_tensor = [&](py::object& obj, const char* name){
    //         if (py::hasattr(obj, name)) {
    //             py::object t = obj.attr(name);
    //             if (py::hasattr(t, "is_cuda") && !py::bool_(t.attr("is_cuda"))) {
    //                 obj.attr(name) = t.attr("to")(py_cuda);
    //             }
    //         }
    //     };
    //     for (const auto& c : cams) {
    //         py::object py_cam = MiniCam_to_py(c);
    //         move_attr_to_cuda_if_tensor(py_cam, "w2c");
    //         move_attr_to_cuda_if_tensor(py_cam, "c2w");
    //         move_attr_to_cuda_if_tensor(py_cam, "position");
    //         move_attr_to_cuda_if_tensor(py_cam, "lookat");
    //         py_cams.append(py_cam);
    //     }

    //     auto Nu_before = octpath_new.size(0);

    //     // rate > 0
    //     at::Tensor rate = svr_mod.attr("mark_max_samp_rate")(
    //         py_cams, py::cast(octpath_new), py::cast(vox_center), py::cast(vox_size)
    //     ).cast<at::Tensor>();
    //     at::Tensor kept = rate > 0;
    //     int64_t n_rate_pos = kept.sum().item<int64_t>();

    //     // near filtering (same threshold you used in createFromPcd)
    //     const float near_thresh = 1.0f;
    //     int64_t n_near_hit = 0;
    //     if (near_thresh > 0.0f) {
    //         at::Tensor is_near = svr_mod.attr("mark_near")(
    //             py_cams, py::cast(octpath_new), py::cast(vox_center), py::cast(vox_size),
    //             py::float_(near_thresh)
    //         ).cast<at::Tensor>();
    //         kept = kept & (~is_near);
    //         n_near_hit = is_near.sum().item<int64_t>();
    //     }

    //     auto idx = torch::nonzero(kept).view({-1});
    //     int64_t K = idx.size(0);

    //     if (K == 0) {
    //         std::cout << "[increasePcd/filter] all candidates filtered out, nothing to add.\n";
    //         return;
    //     }

    //     if (K < octpath_new.size(0)) {
    //         // Apply mask to ALL aligned tensors
    //         octpath_new = octpath_new.index_select(0, idx).contiguous(); // [K,1]
    //         L_u         = L_u.index_select(0, idx).contiguous();         // [K,1]
    //         rgb_u       = rgb_u.index_select(0, idx).contiguous();       // [K,3]
    //     }

    //     Nu = octpath_new.size(0); // update Nu after filtering

    //     // Sanity
    //     TORCH_CHECK(L_u.sizes() == torch::IntArrayRef({Nu,1}), "L_u shape mismatch after filtering");
    //     TORCH_CHECK(rgb_u.sizes() == torch::IntArrayRef({Nu,3}), "rgb_u shape mismatch after filtering");

    //     std::cout << "[increasePcd/filter] Nu_before=" << Nu_before
    //             << " rate>0=" << n_rate_pos
    //             << " near_hit=" << n_near_hit
    //             << " kept_final=" << Nu << std::endl;
    // }
    // // ---- end filtering ----

    // ── 5) Dedup against existing voxels (across-batch) ─────────────────────
    auto octpath_old  = py_->svm.attr("octpath").cast<torch::Tensor>().contiguous();                    // [No,1] int64
    auto octlevel_old = py_->svm.attr("octlevel").cast<torch::Tensor>().contiguous();                   // [No,1] int8

    // Packed 1D key: (octpath<<8) | level
    auto key_new = octpath_new.view({-1}).to(torch::kInt64)
                    .mul(256)
                    .add(L_u.view({-1}).to(torch::kInt64));

    torch::Tensor new_mask;
    if (octpath_old.numel() == 0) {
        std::cout << "octpath_old is empty => all new.\n";
        new_mask = torch::ones({Nu}, torch::dtype(torch::kBool).device(dev));
    } else {
        auto key_old = octpath_old.view({-1}).to(torch::kInt64)
                        .mul(256)
                        .add(octlevel_old.view({-1}).to(torch::kInt64));
        py::object isin_py = torch_mod.attr("isin")(py::cast(key_new), py::cast(key_old));
        auto is_dup = isin_py.cast<torch::Tensor>().to(torch::kBool);                                    // [Nu]
        new_mask = ~is_dup;
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

                // Remove those candidates from insertion
                new_mask = new_mask & (~would_collide_parent);
            }
        }
    }

    if (!new_mask.any().item<bool>()) {
        std::cout << "[increasePcd] No new voxels (all duplicates). Nothing appended.\n";
        return;
    }
    auto sel = torch::nonzero(new_mask).view({-1});                                                     // [Nk]
    auto octpath_add = octpath_new.index_select(0, sel);                                                // [Nk,1]
    auto L_add       = L_u.index_select(0, sel);                                                         // [Nk,1]
    auto rgb_add     = rgb_u.index_select(0, sel);                                                       // [Nk,3]
    const int Nk = sel.size(0);

    int64_t Nm_added = 0;  // count of artifact voxels added later
    // ── 6) Append topology (old preserved) ──────────────────────────────────
    py_->svm.attr("octpath")  = torch::cat({octpath_old,  octpath_add}, 0).contiguous();
    py_->svm.attr("octlevel") = torch::cat({octlevel_old, L_add},       0).contiguous();

    // ── 7) Append learnables for new rows ───────────────────────────────────
    // _subdiv_p
    auto subdiv_old = py_->svm.attr("_subdiv_p").cast<torch::Tensor>();
    auto subdiv_add = torch::ones({Nk,1}, torch::dtype(torch::kFloat32).device(dev));
    py_->svm.attr("_subdiv_p") = torch::cat({subdiv_old, subdiv_add}, 0)
                                .contiguous().detach().requires_grad_();

    // _sh0 from fused rgb
    py::object sh0_add_py = act_mod.attr("rgb2shzero")(py::cast(rgb_add.contiguous())); // [Nk,3]
    auto sh0_add = sh0_add_py.cast<torch::Tensor>().contiguous();
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
        grow = torch::full({M_curr - M_prev, 1}, -10.0f,
                torch::dtype(torch::kFloat32).device(dev))
            .contiguous().detach().requires_grad_();
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

    if (fill_empty_cells_) {
        // ---- 0) Early-exit heuristic: only fill when global PCD bounds expand ----
        // xyz_cpu is already [N,3] on CPU for this batch
        auto xyz_cpu_contig = xyz_cpu.contiguous();
        auto local_min_res = xyz_cpu_contig.min(/*dim=*/0, /*keepdim=*/false);
        auto local_max_res = xyz_cpu_contig.max(/*dim=*/0, /*keepdim=*/false);
        torch::Tensor local_min_cpu = std::get<0>(local_min_res); // [3], CPU
        torch::Tensor local_max_cpu = std::get<0>(local_max_res); // [3], CPU

        bool  should_fill = false;
        float margin = 2.0f * vox_eff_.item<float>();   // e.g. twice base voxel size

        if (!has_global_pcd_bb_) {
            // First time: we don't have bounds yet → we must fill once and initialize bbox.
            should_fill       = true;
            global_pcd_min_   = local_min_cpu.to(torch::kCUDA).contiguous();
            global_pcd_max_   = local_max_cpu.to(torch::kCUDA).contiguous();
            has_global_pcd_bb_ = true;

            std::cout << "[increasePcd] fill_empty_cells_: initializing global bbox\n";
        } else {
            // Compare against current global bounds
            auto gmin_cpu = global_pcd_min_.to(torch::kCPU).contiguous();
            auto gmax_cpu = global_pcd_max_.to(torch::kCPU).contiguous();

            auto acc_gmin = gmin_cpu.accessor<float, 1>();  // [3]
            auto acc_gmax = gmax_cpu.accessor<float, 1>();
            auto acc_lmin = local_min_cpu.accessor<float, 1>();
            auto acc_lmax = local_max_cpu.accessor<float, 1>();

            bool expanded = false;
            for (int d = 0; d < 3; ++d) {
                if (acc_lmin[d] < acc_gmin[d] - margin ||
                    acc_lmax[d] > acc_gmax[d] + margin) {
                    expanded = true;
                    break;
                }
            }

            if (!expanded) {
                // Entire new PCD is inside (or very close to) existing global bbox.
                // → Skip fill_empty_cells_ to avoid re-creating pruned padding voxels.
                std::cout << "[increasePcd] fill_empty_cells_: skipped (pcd inside global bbox)\n";
                // IMPORTANT: do *not* return from increasePcd; just skip filling.
                should_fill = false;
            } else {
                // New batch extends global bbox → update it and allow filling.
                torch::Tensor new_gmin = torch::min(gmin_cpu, local_min_cpu);
                torch::Tensor new_gmax = torch::max(gmax_cpu, local_max_cpu);
                global_pcd_min_ = new_gmin.to(torch::kCUDA).contiguous();
                global_pcd_max_ = new_gmax.to(torch::kCUDA).contiguous();

                should_fill = true;
                std::cout << "[increasePcd] fill_empty_cells_: global bbox expanded\n";
            }
        }

        // If we decided not to fill, just skip the rest of this block.
        if (!should_fill) {
            // Nothing else to do for fill_empty_cells_; rest of increasePcd continues.
        } else {
            // --- A) Build a bbox from the pcd via Python heuristic (median + density) ---
            // Use the raw batch on CPU (xyz_cpu); the function expects an object with a `.points` ndarray
            static py::module bound_utils = py::module::import("src.utils.bounding_utils");
            static py::module types       = py::module::import("types");
            static py::module torch_mod   = py::module::import("torch");

            py::object xyz_cpu_py = py::cast(xyz_cpu.contiguous());             // torch CPU tensor
            py::object ns         = types.attr("SimpleNamespace")(
                                    "points"_a = xyz_cpu_py.attr("numpy")()); // pcd.points = numpy

            // Call Python heuristic
            // py::tuple cr = bound_utils.attr("main_scene_bound_pcd_heuristic")(
            //     ns, py::float_(0.1)   // expose a float member/const for this
            // );

            torch::Tensor center_t;   // CUDA tensor [3]
            float          radius_f;  // scalar radius
            try {
                // Try Python heuristic
                py::tuple cr = bound_utils.attr("main_scene_bound_pcd_heuristic")(ns, py::float_(0.1f));
                if (py::len(cr) == 2) {
                    // cr[0]: numpy array (shape [3]), cr[1]: float
                    py::object center_t_py = torch_mod.attr("from_numpy")(cr[0]); // torch CPU
                    center_t = center_t_py.cast<torch::Tensor>().to(torch::kCUDA).contiguous();
                    radius_f = cr[1].cast<float>();
                } else {
                    throw std::runtime_error("heuristic returned unexpected output");
                }
            } catch (...) {
                std::cout << "[increasePcd] fill_empty_cells_: Python heuristic failed, using C++ fallback.\n";
                // ----- Fallback: compute center/radius purely in C++ -----
                // center = mean of points (CPU -> CUDA)
                auto center_cpu = xyz_cpu.mean(/*dim=*/0, /*keepdim=*/false).contiguous();    // [3], CPU
                center_t = center_cpu.to(torch::kCUDA);
                // radius = max L2 distance to center, with a floor of 3 * vox_size
                torch::Tensor diffs = (xyz_cpu - center_cpu).contiguous();                   // [N,3], CPU
                torch::Tensor dists = diffs.norm(2, /*dim=*/1);                              // [N], CPU
                float maxdist = dists.numel() ? dists.max().item<float>() : 0.0f;
                radius_f = std::max(3.0f * vox_eff_.item<float>(), maxdist);
            }

            // Convert back to Torch (CUDA) so we can do all math in tensors
            // py::object center_t_py = torch_mod.attr("from_numpy")(cr[0]);
            // auto center_t = center_t_py.cast<torch::Tensor>().to(torch::kCUDA).contiguous();      // [3]
            // float radius_f = cr[1].cast<float>();
            auto radius_t  = torch::full({3}, radius_f, torch::dtype(torch::kFloat32).device(torch::kCUDA));

            auto bb_min = (center_t - radius_t).contiguous();   // [3], world coords
            auto bb_max = (center_t + radius_t).contiguous();   // [3]

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

            int64_t cap = std::max<int64_t>(1, max_artifact_cells_);    // set a sensible default (e.g., 200k)
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

                if (max_artifact_cells_ > 0 && ijk_box.size(0) > max_artifact_cells_) {
                    std::cout << "[increasePcd] fill_empty_cells_: limiting artifact cells"<< std::endl;
                    ijk_box = ijk_box.index({torch::indexing::Slice(0, max_artifact_cells_)});
                }

                auto dev = ijk_box.device();
                auto L_box = torch::full({ijk_box.size(0),1}, octlevel_, torch::dtype(torch::kInt8).device(dev)).contiguous();

                // Build morton/octpath for those cells
                py::object octpath_box_py = svr_utils.attr("ijk_2_octpath")(py::cast(ijk_box), py::cast(L_box));
                auto octpath_box = octpath_box_py.cast<torch::Tensor>().contiguous(); // [Nc,1] int64
                
                // // ------------------------------------------------------------------------
                // // B) OPTIONAL: camera-based filtering (mark_max_samp_rate + mark_near)
                // //     so artifact voxels are also consistent with SVRaster camera logic
                // // ------------------------------------------------------------------------
                // if (!cams.empty()) {
                //     py::gil_scoped_acquire gil;
                //     static py::module svr_mod   = py::module::import("svraster_cuda").attr("renderer");
                //     static py::module oct_utils = py::module::import("src.utils.octree_utils");
                //     // torch_mod already defined as static at top of increasePcd

                //     // Decode voxel centers/sizes for artifact voxels
                //     py::tuple dec_box = oct_utils.attr("octpath_decoding")(
                //         py::cast(octpath_box.contiguous()),
                //         py::cast(L_box.contiguous()),
                //         py::cast(scene_center_.contiguous()),
                //         py::cast(scene_extent_.contiguous())
                //     );
                //     at::Tensor vox_center_box = dec_box[0].cast<at::Tensor>(); // [Nc,3] cuda
                //     at::Tensor vox_size_box   = dec_box[1].cast<at::Tensor>(); // [Nc,1] cuda

                //     // Build Python list of CUDA MiniCams (same pattern as above)
                //     py::list py_cams;
                //     py::object py_cuda = torch_mod.attr("device")("cuda");
                //     auto move_attr_to_cuda_if_tensor = [&](py::object& obj, const char* name){
                //         if (py::hasattr(obj, name)) {
                //             py::object t = obj.attr(name);
                //             if (py::hasattr(t, "is_cuda") && !py::bool_(t.attr("is_cuda"))) {
                //                 obj.attr(name) = t.attr("to")(py_cuda);
                //             }
                //         }
                //     };
                //     for (const auto& c : cams) {
                //         py::object py_cam = MiniCam_to_py(c);
                //         move_attr_to_cuda_if_tensor(py_cam, "w2c");
                //         move_attr_to_cuda_if_tensor(py_cam, "c2w");
                //         move_attr_to_cuda_if_tensor(py_cam, "position");
                //         move_attr_to_cuda_if_tensor(py_cam, "lookat");
                //         py_cams.append(py_cam);
                //     }

                //     auto Nc_before = octpath_box.size(0);

                //     // 1) Visibility / sampling rate
                //     at::Tensor rate_box = svr_mod.attr("mark_max_samp_rate")(
                //         py_cams,
                //         py::cast(octpath_box),
                //         py::cast(vox_center_box),
                //         py::cast(vox_size_box)
                //     ).cast<at::Tensor>();
                //     at::Tensor kept_cam = rate_box > 0;
                //     int64_t n_rate_pos_box = kept_cam.sum().item<int64_t>();

                //     // 2) Near filtering with same threshold as for PCD voxels
                //     //    (you can pull this out as a const float near_thresh = ... at top)
                //     const float near_thresh = 1.0f;    // or 5.0f if you prefer
                //     int64_t n_near_hit_box = 0;
                //     if (near_thresh > 0.0f) {
                //         at::Tensor is_near_box = svr_mod.attr("mark_near")(
                //             py_cams,
                //             py::cast(octpath_box),
                //             py::cast(vox_center_box),
                //             py::cast(vox_size_box),
                //             py::float_(near_thresh)
                //         ).cast<at::Tensor>();
                //         kept_cam = kept_cam & (~is_near_box);
                //         n_near_hit_box = is_near_box.sum().item<int64_t>();
                //     }

                //     auto idx_box = torch::nonzero(kept_cam).view({-1});
                //     int64_t K_box = idx_box.size(0);

                //     if (K_box == 0) {
                //         std::cout << "[increasePcd/fill_empty_cells] "
                //                 << "all artifact voxels filtered out by camera visibility / near; skipping."
                //                 << std::endl;
                //         return;
                //     }

                //     if (K_box < octpath_box.size(0)) {
                //         // Apply mask consistently to topology + ijk_box
                //         octpath_box = octpath_box.index_select(0, idx_box).contiguous(); // [K_box,1]
                //         L_box       = L_box.index_select(0, idx_box).contiguous();       // [K_box,1]
                //         ijk_box     = ijk_box.index_select(0, idx_box).contiguous();     // [K_box,3]
                //     }

                //     std::cout << "[increasePcd/fill_empty_cells_cam] Nc_before=" << Nc_before
                //             << " rate>0=" << n_rate_pos_box
                //             << " near_hit=" << n_near_hit_box
                //             << " kept_final=" << octpath_box.size(0) << std::endl;
                // }

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
                    sel_artifacts_viz= sel.clone();    // [Nm] indices into ijk_box
                    ijk_box_viz      = ijk_box.clone();// [Nc,3] (CUDA)

                    auto octpath_add2 = octpath_box.index_select(0, sel); // [Nm,1]
                    auto L_add2       = L_box.index_select(0, sel);       // [Nm,1]

                    // Append topology
                    py_->svm.attr("octpath")  = torch::cat({octpath_cur,  octpath_add2}, 0).contiguous();
                    py_->svm.attr("octlevel") = torch::cat({octlevel_cur, L_add2},       0).contiguous();

                    // Prepare learnables for artifacts
                    const int n_sh_rest = (max_sh_degree_ + 1)*(max_sh_degree_ + 1) - 1;
                    auto shs_add2    = torch::zeros({Nm_added, n_sh_rest, 3}, torch::dtype(torch::kFloat32).device(dev)).contiguous();
                    auto subdiv_add2 = torch::ones({Nm_added,1},             torch::dtype(torch::kFloat32).device(dev)).contiguous();

                    auto rgb_add2 = torch::empty({Nm_added,3}, torch::dtype(torch::kFloat32).device(dev));
                    rgb_add2.index_put_({torch::indexing::Slice(),0}, artifact_bg_rgb_[0]);
                    rgb_add2.index_put_({torch::indexing::Slice(),1}, artifact_bg_rgb_[1]);
                    rgb_add2.index_put_({torch::indexing::Slice(),2}, artifact_bg_rgb_[2]);

                    py::object sh0_add2_py = act_mod.attr("rgb2shzero")(py::cast(rgb_add2.contiguous()));
                    auto sh0_add2 = sh0_add2_py.cast<torch::Tensor>().contiguous();

                    // Grid growth and optimizer-preserving appends
                    auto grid_pts_key_new = py_->svm.attr("grid_pts_key").cast<torch::Tensor>();
                    const int64_t M_prev = grid_pts_key_.defined() ? grid_pts_key_.size(0) : 0;
                    const int64_t M_curr = grid_pts_key_new.size(0);
                    if (M_curr > M_prev) {
                        auto grow = torch::full({M_curr - M_prev, 1}, -10.0f,
                                    torch::dtype(torch::kFloat32).device(dev))
                                    .contiguous().detach().requires_grad_();
                        appendGroup_(/*group_idx=*/0, grow, "_geo_grid_pts", &this->_geo_grid_pts_);
                    }
                    appendGroup_(/*group_idx=*/1, sh0_add2, "_sh0", &this->sh0_);
                    appendGroup_(/*group_idx=*/2, shs_add2, "_shs", &this->shs_);

                    // subdiv_p (not in optimizer groups)
                    auto subdiv_old2 = py_->svm.attr("_subdiv_p").cast<torch::Tensor>();
                    py_->svm.attr("_subdiv_p") = torch::cat({subdiv_old2, subdiv_add2}, 0)
                                                .contiguous().detach().requires_grad_();
                    this->subdiv_p_ = py_->svm.attr("_subdiv_p").cast<torch::Tensor>().requires_grad_(true);
                    // std::cout << "[increasePcd] Added " << Nm_added << " artifact voxels (pcd bbox).\n";
                    
                    // NEW: mark that we actually added artifact voxels this call
                    artifact_fill_happened_ = true;
                }
                // Keep your existing “new voxels” logging in sync by adding Nm_added to Nk (as noted earlier)
                // ... (use last_added = Nk + Nm_added later)
            } // valid bbox
        } //should_fill
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

    // ── 10) Re-register with optimizer (new rows appended) ───────────────────
    VOXEL_MODEL_TENSORS_TO_VEC

    // 1) Always (re)log the global scene box so timeline scrubbing shows context
    // rrLogGlobalSceneAABB(inc_counter);
    // 2) Log ALL voxels
    // rrLogVoxelBoxes(this->center_, this->size_, /*max_boxes_for_viz=*/10000000000, "vox_all", inc_counter);
    // 3) Log ONLY the newly-added voxels (orange)s
    const int64_t last_added = Nk + Nm_added;
    if (last_added > 0) {
        auto start = this->center_.size(0) - last_added;
        auto idx = torch::arange(start, this->center_.size(0),
                                this->center_.options().dtype(torch::kLong));
        auto centers_new = this->center_.index_select(0, idx);
        auto size_new    = this->size_.index_select(0, idx);
        // rrLogVoxelBoxes(centers_new, size_new, /*max*/10000000000, "vox_new", inc_counter);
    }

    // --- One-shot visualization of the algorithm ---
    static bool viz_once_done = false;
    if (fill_empty_cells_ && !viz_once_done && Nm_added > 0 &&
        bb_min_viz.defined() && bb_max_viz.defined() &&
        sel_artifacts_viz.defined() && ijk_box_viz.defined()) {

        std::vector<torch::Tensor> aabb_list{bb_min_viz, bb_max_viz};
        at::Tensor aabb = torch::stack(aabb_list).contiguous();

        // rrLogPointsAndAABB(0, xyz, rgb, aabb,
        //     "viz/pcd_and_bbox", 1000000, /*log_points=*/true, /*log_box=*/true, inc_counter);

        auto ijk_art  = ijk_box_viz.index_select(0, sel_artifacts_viz).to(torch::kFloat32);
        auto vox_t_viz= vox_eff_.mean().view({1}).repeat({3});
        auto art_pts  = (scene_min_t_ + (ijk_art + 0.5f) * vox_t_viz).contiguous();

        auto art_rgb  = torch::empty({art_pts.size(0), 3},
                            torch::dtype(torch::kFloat32).device(art_pts.device()));
        art_rgb.index_put_({torch::indexing::Slice(),0}, artifact_bg_rgb_[0]);
        art_rgb.index_put_({torch::indexing::Slice(),1}, artifact_bg_rgb_[1]);
        art_rgb.index_put_({torch::indexing::Slice(),2}, artifact_bg_rgb_[2]);

        // rrLogPointsAndAABB(0, art_pts, art_rgb, aabb,
        //     "viz/artifact_points", 200000, /*log_points=*/true, /*log_box=*/false, inc_counter);

        const int64_t total = this->center_.size(0);
        const int64_t last_added = Nk + Nm_added;
        if (last_added > 0 && total >= last_added) {
            auto optsL = this->center_.options().dtype(torch::kLong);
            auto idx_real_new = (Nm_added > 0)
                ? torch::arange(total - last_added, total - Nm_added, optsL)
                : torch::empty({0}, optsL);
            auto idx_art_new  = (Nm_added > 0)
                ? torch::arange(total - Nm_added, total, optsL)
                : torch::empty({0}, optsL);

            // if (idx_real_new.numel() > 0) {
            //     auto c_real = this->center_.index_select(0, idx_real_new);
            //     auto s_real = this->size_.index_select(0,   idx_real_new);
            //     rrLogVoxelBoxes(c_real, s_real, 200000, "viz/vox_new_real", inc_counter);
            // }
            // if (idx_art_new.numel() > 0) {
            //     auto c_art = this->center_.index_select(0, idx_art_new);
            //     auto s_art = this->size_.index_select(0,   idx_art_new);
            //     rrLogVoxelBoxes(c_art, s_art, 200000, "viz/vox_new_artifacts", inc_counter);
            // }
        }
        viz_once_done = true;
    }

    ++inc_counter;
    // std::cout << "[increasePcd] Appended " << Nk << " new voxels. Total now: "
    //         << this->oct_path_.size(0) << "\n";
    // std::cout << "[increasePcd] Appended " << (Nk + Nm_added)
    //       << " voxels this call (real=" << Nk
    //       << ", artifact=" << Nm_added << "). Total now: "
    //       << this->oct_path_.size(0) << "\n";
}

void VoxelModel::increasePcd(torch::Tensor& new_point_cloud,
                             torch::Tensor& new_colors,
                             const int iteration)
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

    // Colors: current pipeline (RGBD branch) gives us [0,1] floats.
    // The existing vector-based increasePcd expects 0..255 and divides by 255.0f.
    // So we scale up to 0..255 here to keep behaviour consistent.
    const float* rgb_ptr = rgb_flat.data_ptr<float>();
    for (int64_t i = 0; i < static_cast<int64_t>(cols.size()); ++i) {
        cols[i] = rgb_ptr[i] * 255.0f;
    }

    // For inactive geo densify we can safely pass an empty cam list:
    // we already know points are valid (near keypoints with good depth),
    // so we do not need extra mark_max_samp_rate / mark_near filtering.
    std::vector<sv::MiniCam> empty_cams;

    // Reuse the main SVRaster-aware pipeline
    increasePcd(points, cols, iteration, empty_cams);
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
    if (a.lambda_R_concen.has_value())  kw["lambda_R_concen"] = *a.lambda_R_concen;
    if (a.gt_color.defined())            kw["gt_color"]         = a.gt_color;
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
            std::cerr << "[INFO] Python-side renderer imported OK.\n";
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

