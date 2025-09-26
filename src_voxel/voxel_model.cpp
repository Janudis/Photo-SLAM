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

int64_t sv::VoxelModel::numGridPts() const {
    return (grid_pts_key_.defined() && grid_pts_key_.dim() > 0)
           ? grid_pts_key_.size(0)
           : 0;
}

const torch::Tensor& sv::VoxelModel::geoGridPts() const { return _geo_grid_pts_; }
const torch::Tensor& sv::VoxelModel::sh0()        const { return sh0_; }
const torch::Tensor& sv::VoxelModel::shs()        const { return shs_; }

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

static inline std::string rrTimestamp() {
    std::time_t t = std::time(nullptr);
    std::tm tm;
    localtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    return buf;
}

inline void sv::VoxelModel::rrInitOnce() {
    if (rr_initialized_) return;
    rr_ = std::make_unique<rerun::RecordingStream>("PhotoSLAM");
    const std::string f = "/home/dimitris/Photo-SLAM/rerun/debug_run_" + rrTimestamp() + ".rrd";
    rr_->save(f).exit_on_failure();  // offline
    rr_initialized_ = true;
    // optional: std::cout << "[Rerun] writing to " << f << "\n";
}

static inline std::string rrSanitize(std::string s) {
    for (char& ch : s)
        if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch=='_' || ch=='-' || ch=='/'))
            ch = '_';
    return s;
}

void sv::VoxelModel::rrLogPointsAndAABB(
    int iteration,                        // only for label text
    const at::Tensor& xyz,                // [N,3] (on any device)
    const at::Tensor& rgb,                // [N,3] or empty
    const at::Tensor& bounding_2x3,       // [2,3] min/max (CPU or GPU)
    const std::string& tag,               // e.g., "init_raw", "init_used", "increase_raw", "increase_used"
    int64_t max_points_for_viz,           // subsample cap; 0 => no subsampling
    bool log_points,                      // log point cloud?
    bool log_box,                         // log AABB?
    int64_t inc                           // required: single time axis
){
    rrInitOnce();

    const std::string tag_safe = rrSanitize(tag);

    // ---- Points ----
    if (log_points && xyz.defined() && xyz.numel() >= 3) {
        at::Tensor P = xyz.detach().to(at::kCPU).to(at::kFloat).contiguous();
        if (P.dim()==2 && P.size(1)==3) {
            at::Tensor C;
            if (rgb.defined() && rgb.sizes()==P.sizes())
                C = rgb.detach().clamp(0,1).to(at::kCPU).to(at::kFloat).contiguous();
            else
                C = at::ones({P.size(0),3}, P.options());

            const int64_t N = P.size(0);
            const int64_t stride = (max_points_for_viz>0 && N>max_points_for_viz)
                                   ? ((N + max_points_for_viz - 1) / max_points_for_viz)
                                   : 1;

            std::vector<rerun::Vec3D> positions; positions.reserve((N + stride - 1) / stride);
            std::vector<rerun::Rgba32> colors;    colors.reserve((N + stride - 1) / stride);

            const float* p = P.data_ptr<float>();
            const float* c = C.data_ptr<float>();
            for (int64_t i=0; i<N; i+=stride) {
                positions.emplace_back(p[3*i+0], p[3*i+1], p[3*i+2]);
                colors.emplace_back(
                    (uint8_t)std::lround(255.f * c[3*i+0]),
                    (uint8_t)std::lround(255.f * c[3*i+1]),
                    (uint8_t)std::lround(255.f * c[3*i+2]),
                    255
                );
            }

            rr_->set_time_sequence("inc", inc);
            rr_->log("world/points/" + tag_safe,
                     rerun::Points3D(positions).with_colors(colors));
        }
    }

    // ---- Box ----
    if (log_box && bounding_2x3.defined() && bounding_2x3.numel()==6) {
        at::Tensor B = bounding_2x3.detach().to(at::kCPU).to(at::kFloat).contiguous();
        if (B.dim()==2 && B.size(0)==2 && B.size(1)==3) {
            const float minx = B[0][0].item<float>();
            const float miny = B[0][1].item<float>();
            const float minz = B[0][2].item<float>();
            const float maxx = B[1][0].item<float>();
            const float maxy = B[1][1].item<float>();
            const float maxz = B[1][2].item<float>();

            const rerun::Vec3D mins  {minx, miny, minz};
            const rerun::Vec3D sizes {std::max(1e-6f, maxx-minx),
                                      std::max(1e-6f, maxy-miny),
                                      std::max(1e-6f, maxz-minz)};

            const bool is_raw = (tag.find("raw") != std::string::npos);
            const rerun::Rgba32 col = is_raw ? rerun::Rgba32(255,0,0,255)   // raw = red
                                             : rerun::Rgba32(0,255,0,255);  // used/union = green

            std::ostringstream lbl; lbl << tag << "@" << iteration;

            rr_->set_time_sequence("inc", inc);
            rr_->log("world/boxes/" + tag_safe,
                     rerun::Boxes3D::from_mins_and_sizes({mins}, {sizes})
                        .with_labels({lbl.str()})
                        .with_colors({col}));
        }
    }
}

void sv::VoxelModel::rrLogVoxelBoxes(
    const at::Tensor& centers,     // [N,3], any device
    const at::Tensor& size,        // [N],   any device (edge length per voxel)
    int64_t max_boxes_for_viz,     // e.g., 20000
    const std::string& tag,        // e.g., "vox_all", "vox_new"
    int64_t inc
){
    rrInitOnce();

    if (!centers.defined() || !size.defined()) return;
    at::Tensor C = centers.detach().to(at::kCPU).to(at::kFloat).contiguous();
    at::Tensor S = size   .detach().to(at::kCPU).to(at::kFloat).contiguous();

    TORCH_CHECK(C.dim()==2 && C.size(1)==3, "centers must be [N,3]");
    TORCH_CHECK(S.dim()==1 && S.size(0)==C.size(0), "size must be [N]");

    int64_t N = C.size(0);
    int64_t stride = (max_boxes_for_viz>0 && N>max_boxes_for_viz)
                     ? ((N + max_boxes_for_viz - 1) / max_boxes_for_viz)
                     : 1;

    std::vector<rerun::Vec3D> mins;  mins.reserve((N + stride - 1) / stride);
    std::vector<rerun::Vec3D> sizes; sizes.reserve((N + stride - 1) / stride);

    const float* pc = C.data_ptr<float>();
    const float* ps = S.data_ptr<float>();
    for (int64_t i=0; i<N; i+=stride) {
        float sz = std::max(1e-6f, ps[i]);
        float hx = 0.5f * sz;
        mins.emplace_back(pc[3*i+0] - hx, pc[3*i+1] - hx, pc[3*i+2] - hx);
        sizes.emplace_back(sz, sz, sz);
    }

    // color: orange for "new", translucent blue otherwise
    const rerun::Rgba32 col = (tag.find("new") != std::string::npos)
        ? rerun::Rgba32(255, 165,   0, 128)  // orange, semi
        : rerun::Rgba32(  0, 128, 255,  64); // blue, more translucent

    rr_->set_time_sequence("inc", inc);
    rr_->log("world/voxels/" + rrSanitize(tag),
             rerun::Boxes3D::from_mins_and_sizes(mins, sizes)
                 .with_colors({col}));
}

inline void sv::VoxelModel::rrLogGlobalSceneAABB(int64_t inc)
{
    rrInitOnce();
    const float e = global_scene_extent_;
    const rerun::Vec3D mins {
        global_scene_center_[0] - 0.5f*e,
        global_scene_center_[1] - 0.5f*e,
        global_scene_center_[2] - 0.5f*e,
    };
    const rerun::Vec3D sizes { std::max(1e-6f, e), std::max(1e-6f, e), std::max(1e-6f, e) };

    rr_->set_time_sequence("inc", inc);
    rr_->log("world/boxes/global_scene",
             rerun::Boxes3D::from_mins_and_sizes({mins}, {sizes})
                 .with_labels({"global_scene"})
                 .with_colors({rerun::Rgba32(0,255,0,255)})); // solid green
}

void VoxelModel::createFromPcd(const std::map<point3D_id_t, Point3D>& pcd)
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
    const auto scene_center = torch::tensor(
        {global_scene_center_[0], global_scene_center_[1], global_scene_center_[2]},
        torch::dtype(torch::kFloat32).device(dev));                            // [3]
    const auto scene_extent_t = torch::tensor(
        {global_scene_extent_}, torch::dtype(torch::kFloat32).device(dev));    // [1]
    const auto scene_min = scene_center - 0.5f * scene_extent_t;               // [3]

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
            rgb[i][0] = P.color_(0) / 255.0f;
            rgb[i][1] = P.color_(1) / 255.0f;
            rgb[i][2] = P.color_(2) / 255.0f;
            ++i;
        }
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
    py::object L_fp_py = oct_utils.attr("vox_size_2_level")(py::cast(scene_extent_t), py::cast(vox_size_t));
    auto L_fp = L_fp_py.cast<torch::Tensor>();  // [1] float

    // Rounding mode: "nearest" (you can expose this as a config if you like)
    L_fp = L_fp.round();
    // Clamp to [1, MAX_L]
    auto L_clamped = L_fp.clamp_min(1).clamp_max(MAX_L).to(torch::kInt8);     // [1] int8
    // Broadcast to all points -> [N,1]
    auto octlevel = L_clamped.view({1,1}).repeat({N,1}).contiguous();         // [N,1] int8
    // Effective voxel size (tensor) = level_2_vox_size(scene_extent, octlevel)
    py::object vox_eff_py = oct_utils.attr("level_2_vox_size")(py::cast(scene_extent_t), py::cast(octlevel));
    auto vox_eff = vox_eff_py.cast<torch::Tensor>().contiguous();             // [N,1] float (same value in all rows)
    // std::cout << "[createFromPcd] Using octlevel=" << octlevel[0][0].item<int>()
    //           << " (vox_size=" << vox_eff[0][0].item<float>() << " m) for fixed_vox_size_="
    //           << fixed_vox_size_ << " m.\n";

    // ------------------------------------------------------------------------
    // 2) Compute ijk with this level/voxel size (mirror points_init)
    // ------------------------------------------------------------------------
    // ijk = ((xyz - scene_min) / vox_size).long()
    torch::Tensor ijk = ((xyz - scene_min) / vox_eff).floor().to(torch::kLong); // [N,3]

    // Dedup (ijk, octlevel) and fuse RGB (scatter mean)
    auto L_long = octlevel.to(torch::kLong);             // [N,1]
    auto ijkl   = torch::cat({ijk, L_long}, 1);          // [N,4]

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
    const int Nu = ijk_u.size(0);

    auto rgb_u = torch::zeros({Nu,3}, torch::dtype(torch::kFloat32).device(dev));
    rgb_u.index_add_(0, invmap, rgb);
    auto counts = torch::bincount(invmap, {}, Nu).view({Nu,1}).clamp_min(1);
    rgb_u = rgb_u / counts;

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
        py_->svm = SVM(py::arg("sh_degree") = max_sh_degree_,
                       py::arg("black_background") = true);
    }

    // scene tensors (single fixed scene; no outside shells => inside_extent == scene_extent)
    py_->svm.attr("scene_center")  = scene_center.contiguous();   // [3]
    py_->svm.attr("scene_extent")  = scene_extent_t.contiguous(); // [1]
    py_->svm.attr("inside_extent") = scene_extent_t.contiguous(); // [1]

    // Topology
    py_->svm.attr("octpath")  = octpath;                          // [Nu,1] int64
    py_->svm.attr("octlevel") = L_u.view({Nu,1}).contiguous();    // [Nu,1] int8

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
    const int64_t inc0 = 0;
    // 1) Log the global fixed scene AABB once
    rrLogGlobalSceneAABB(inc0);
    // 2) Log ALL voxels after creation
    rrLogVoxelBoxes(this->center_, this->size_, /*max_boxes_for_viz=*/200000, "vox_all", inc0);
}

void VoxelModel::increasePcd(std::vector<float> pcd_full,
                             std::vector<float> colors,
                             const int /*iteration*/)
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
    torch::Tensor xyz = xyz_cpu.to(dev);
    torch::Tensor rgb = rgb_cpu.to(dev);

    // ——— 1) Fixed scene tensors ————————————————————————————————
    const auto scene_center = torch::tensor(
        {global_scene_center_[0], global_scene_center_[1], global_scene_center_[2]},
        torch::dtype(torch::kFloat32).device(dev));                            // [3]
    const auto scene_extent_t = torch::tensor(
        {global_scene_extent_}, torch::dtype(torch::kFloat32).device(dev));    // [1]
    const auto scene_min = scene_center - 0.5f * scene_extent_t;               // [3]

    // Build the min/max AABB tensor once:
    at::Tensor aabb = torch::stack({
        torch::tensor({ global_scene_center_[0] - 0.5f*global_scene_extent_,
                        global_scene_center_[1] - 0.5f*global_scene_extent_,
                        global_scene_center_[2] - 0.5f*global_scene_extent_ }),
        torch::tensor({ global_scene_center_[0] + 0.5f*global_scene_extent_,
                        global_scene_center_[1] + 0.5f*global_scene_extent_,
                        global_scene_center_[2] + 0.5f*global_scene_extent_ })
    });
    rrLogPointsAndAABB(/*iteration=*/0, xyz, rgb, aabb,
                    /*tag=*/"increase_raw", /*cap=*/200000,
                    /*log_points=*/true, /*log_box=*/false, /*inc=*/inc_counter);

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

    // ── 2) Compute octlevel from fixed_vox_size_ (nearest), effective vox size
    auto vox_size_req = torch::full({1}, fixed_vox_size_, torch::dtype(torch::kFloat32).device(dev)); // [1]
    py::object L_fp_py = oct_utils.attr("vox_size_2_level")(py::cast(scene_extent_t), py::cast(vox_size_req));
    auto L_fp = L_fp_py.cast<torch::Tensor>().round();                                                // nearest
    auto L_clamped = L_fp.clamp_min(1).clamp_max(MAX_L).to(torch::kInt8);                             // [1] int8
    auto octlevel  = L_clamped.view({1,1}).repeat({N,1}).contiguous();                                // [N,1] int8

    py::object vox_eff_py = oct_utils.attr("level_2_vox_size")(py::cast(scene_extent_t), py::cast(octlevel));
    auto vox_eff = vox_eff_py.cast<torch::Tensor>().contiguous();                                     // [N,1] float
    // std::cout << "[increasePcd] Using octlevel=" << octlevel[0][0].item<int>()
    //           << " (vox_size=" << vox_eff[0][0].item<float>() << " m) for fixed_vox_size_="
    //           << fixed_vox_size_ << " m.\n";

    // ── 3) Compute ijk, dedup within-batch, fuse RGB ────────────────────────
    torch::Tensor ijk = ((xyz - scene_min) / vox_eff).floor().to(torch::kLong);                       // [N,3]
    auto L_long = octlevel.to(torch::kLong);                                                          // [N,1]
    auto ijkl   = torch::cat({ijk, L_long}, 1);                                                       // [N,4]

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
    const int Nu = ijk_u.size(0);

    // Defensive bound check: ijk in [0, 2^L)
    const int8_t L0 = L_u[0].item<int8_t>();      // all rows share the same level
    const long limit = (1L << L0);
    TORCH_CHECK(
        (ijk_u.index({torch::indexing::Slice(),0}) >= 0).all().item<bool>() &&
        (ijk_u.index({torch::indexing::Slice(),1}) >= 0).all().item<bool>() &&
        (ijk_u.index({torch::indexing::Slice(),2}) >= 0).all().item<bool>(),
        "increasePcd: points below scene_min — enlarge global_scene_extent_ or filter.");
    TORCH_CHECK(
        (ijk_u.index({torch::indexing::Slice(),0}) < limit).all().item<bool>() &&
        (ijk_u.index({torch::indexing::Slice(),1}) < limit).all().item<bool>() &&
        (ijk_u.index({torch::indexing::Slice(),2}) < limit).all().item<bool>(),
        "increasePcd: points exceed scene bounds — enlarge global_scene_extent_ or filter.");

    auto rgb_u = torch::zeros({Nu,3}, torch::dtype(torch::kFloat32).device(dev));
    rgb_u.index_add_(0, invmap, rgb);
    auto counts = torch::bincount(invmap, {}, Nu).view({Nu,1}).clamp_min(1);
    rgb_u = rgb_u / counts;

    // ── 4) Build octpath for this batch ─────────────────────────────────────
    py::object octpath_py = svr_utils.attr("ijk_2_octpath")(py::cast(ijk_u), py::cast(L_u));
    auto octpath_new = octpath_py.cast<torch::Tensor>().contiguous();                                   // [Nu,1] int64

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

    if (!new_mask.any().item<bool>()) {
        std::cout << "[increasePcd] No new voxels (all duplicates). Nothing appended.\n";
        return;
    }

    auto sel = torch::nonzero(new_mask).view({-1});                                                     // [Nk]
    auto octpath_add = octpath_new.index_select(0, sel);                                                // [Nk,1]
    auto L_add       = L_u.index_select(0, sel);                                                         // [Nk,1]
    auto rgb_add     = rgb_u.index_select(0, sel);                                                       // [Nk,3]
    const int Nk = sel.size(0);

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


    // auto geo_old = py_->svm.attr("_geo_grid_pts").cast<torch::Tensor>();
    // if (M_curr > M_prev) {
    //     auto grow = torch::full({M_curr - M_prev, 1}, -10.0f, torch::dtype(torch::kFloat32).device(dev));
    //     py_->svm.attr("_geo_grid_pts") = torch::cat({geo_old, grow}, 0)
    //                                 .contiguous().detach().requires_grad_();
    // }
    // // ── 9) Pull back for renderer; keep exact field names/types ─────────────
    // auto fetch = [&](const char* name){
    //     return py_->svm.attr(name).cast<torch::Tensor>().contiguous();
    // };
    // this->oct_path_      = fetch("octpath");
    // this->oct_level_     = fetch("octlevel");
    // this->center_        = fetch("vox_center");
    // this->size_          = fetch("vox_size").squeeze(1);
    // this->vox_size_inv_  = 1.0f / size_;
    // this->grid_pts_key_  = fetch("grid_pts_key");
    // this->vox_key_       = fetch("vox_key");
    // this->_geo_grid_pts_ = fetch("_geo_grid_pts").requires_grad_(true);
    // this->sh0_           = fetch("_sh0").requires_grad_(true);
    // this->shs_           = fetch("_shs").requires_grad_(true);
    // this->subdiv_p_      = fetch("_subdiv_p").requires_grad_(true);
    // this->max_w_ = torch::zeros({center_.size(0), 1}, torch::dtype(torch::kFloat32).device(dev));

    // ── 10) Re-register with optimizer (new rows appended) ───────────────────
    VOXEL_MODEL_TENSORS_TO_VEC

    std::cout << "[increasePcd] Appended " << Nk << " new voxels. Total now: "
              << this->oct_path_.size(0) << "\n";
    // 1) Always (re)log the global scene box so timeline scrubbing shows context
    rrLogGlobalSceneAABB(inc_counter);
    // 2) Log ALL voxels
    rrLogVoxelBoxes(this->center_, this->size_, /*max_boxes_for_viz=*/200000, "vox_all", inc_counter);
    // 3) Log ONLY the newly-added voxels (orange)
    if (sel.defined() && sel.numel() > 0) {
        auto centers_new = this->center_.index_select(0, torch::arange(
            this->center_.size(0) - sel.size(0), this->center_.size(0),
            /*step=*/1, this->center_.options().dtype(torch::kLong).device(this->center_.device())));
        auto size_new = this->size_.index_select(0, torch::arange(
            this->size_.size(0) - sel.size(0), this->size_.size(0),
            /*step=*/1, this->size_.options().dtype(torch::kLong).device(this->size_.device())));
        // If you kept "sel" before the concat, you can also select directly from the append block.

        rrLogVoxelBoxes(centers_new, size_new, /*max*/200000, "vox_new", inc_counter);
    }
    ++inc_counter;
}

// void VoxelModel::createFromPcd(const std::map<point3D_id_t, Point3D>& pcd)
// {
//     // '''
//     // dense grid
//     // '''
//     std::cout << "VoxelModel::createPcd() called with "
//               << pcd.size() << " points." << std::endl;
//     namespace py = pybind11;

//     // ─── 0) Reset per-scene stuff ───────────────────────────────────────
//     rr_inc_step_ = 0;                 // (if you use Rerun time axis)
//     const int64_t inc0 = 0;

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
//     // std::cout << "this->scene_center_" << this->scene_center_ << std::endl;
//     // std::cout << "this->scene_extent_" << this->scene_extent_ << std::endl;
//     logAABB_bounding2x3(
//         bounding_cpu, "init_raw", /*iter=*/0,
//         std::filesystem::path("/home/dimitris/Photo-SLAM/results/tum_voxel/rgbd_dataset_freiburg1_desk/atlas_debug"));
//     // std::cout << "[Debug] scene_extent = " << extent_side << std::endl;
//     // float float_level = -std::log2(0.05f / extent_side);
//     // int level = std::round(float_level);
//     // std::cout << "level = " << level
//     //           << " (float_level = " << float_level << ")" << std::endl;
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
//     //  log the raw bounding from Python (min/max)
//     rrLogPointsAndAABB(/*iteration=*/0, xyz, rgb, bounding_cpu,
//                        /*tag=*/"init_raw",
//                        /*max_points_for_viz=*/200000,
//                        /*log_points=*/true, /*log_box=*/true,
//                        /*inc=*/inc0);    
//     // rrLogVoxelBoxes(center_, size_, /*max_boxes=*/20000, "boxes", inc0);

//     const float extent_val = this->scene_extent_.item<float>();
//     // 2) Build the used min/max as CPU tensors (shape [3]), using the scalar
//     torch::Tensor used_min_cpu = (this->scene_center_ - 0.5f * extent_val).to(torch::kCPU);
//     torch::Tensor used_max_cpu = (this->scene_center_ + 0.5f * extent_val).to(torch::kCPU);

//     // 3) Record current inside AABB (host copies)
//     for (int d = 0; d < 3; ++d) {
//         cur_min_[d] = used_min_cpu[d].item<float>();
//         cur_max_[d] = used_max_cpu[d].item<float>();
//     }
//     have_cur_aabb_ = true;

//     torch::Tensor used_box_cpu = torch::stack({used_min_cpu, used_max_cpu}); // [2,3]
    
//     // 4) Fix the global lattice
//     const int kInitLevel = 6; 
//     global_vox_       = extent_val / float(1 << kInitLevel);  // <-- was scene_extent_ / ...
//     global_origin_[0]= used_min_cpu[0].item<float>();
//     global_origin_[1]= used_min_cpu[1].item<float>();
//     global_origin_[2]= used_min_cpu[2].item<float>();
//     global_grid_ok_   = true;

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
//         // py::arg("bounding")        = bounding,
//         py::arg("bounding")        = used_box_cpu.to(device_type_),
//         py::arg("outside_level")   = 0,
//         // py::arg("init_n_level")    = level,
//         py::arg("init_n_level")    = 6,
//         py::arg("init_out_ratio")  = 2,
//         py::arg("sh_degree_init")  = 3,
//         py::arg("geo_init")        = -10.0f,
//         py::arg("sh0_init")        = 0.5f,
//         py::arg("shs_init")        = 0.0f
//         // py::arg("cameras")  = all_py_cams
//     );

    // // ─── 4) Pull back all core tensors from the Python object ───────────
    // auto fetch = [&](const char* name){
    //     return py_->svm.attr(name).cast<torch::Tensor>().contiguous();
    // };
    // this->oct_path_      = fetch("octpath");
    // this->oct_level_     = fetch("octlevel");
    // this->center_        = fetch("vox_center");
    // this->size_          = fetch("vox_size").squeeze(1);
    // this->vox_size_inv_  = 1.0f / size_;
    // this->grid_pts_key_  = fetch("grid_pts_key");
    // this->vox_key_       = fetch("vox_key");      
    // // ─── 5) Copy over the learnable fields ─────────────────────────────
    // this->_geo_grid_pts_ = fetch("_geo_grid_pts").requires_grad_(true);
    // this->sh0_ = fetch("_sh0").requires_grad_(true);
    // this->shs_ = fetch("_shs").requires_grad_(true);
    // this->subdiv_p_ = fetch("_subdiv_p").requires_grad_(true);
    // // subdiv_p_   .retain_grad();
    // // ─── 6) Stats buffers (exactly as before) ───────────────────────────
    // this->max_w_ = torch::zeros({center_.size(0), 1},
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
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_pts_all.npy",
//                     tensor_to_numpy(xyz.cpu()));
//     np.attr("save")("/home/dimitris/Photo-SLAM/debug_cols_all.npy",
//                     tensor_to_numpy(rgb.cpu()));

//     // ─── 7) Register with the optimizer, etc. ──────────────────────────
//     VOXEL_MODEL_TENSORS_TO_VEC
// }

// void VoxelModel::increasePcd(std::vector<float> pcd_full,
//                              std::vector<float> colors,
//                              const int /*iteration*/)
// {
//     const int Nf = static_cast<int>(pcd_full.size());
//     if (Nf < 3 || colors.size() < 3) return;
//     const int N = Nf / 3;
//     std::cout << "VoxelModel::increasePcd() called with "
//               << Nf << " floats (" << N << " points).\n";

//     // ─────────────────────────────────────────────────────────────────────
//     // 0) Early-out guard: if empty or no bbox expansion needed
//     //    (we’ll *still* compute a “raw” bbox via Python to match createFromPcd)
//     // ─────────────────────────────────────────────────────────────────────

//     // Build a CPU torch tensor (N,3) from incoming points (like createFromPcd)
//     torch::Tensor xyz_cpu = torch::from_blob(
//         pcd_full.data(),
//         { (int64_t)N, (int64_t)3 },
//         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
//     ).clone(); // clone so lifetime is independent of pcd_full
//     torch::Tensor rgb_cpu;
//     rgb_cpu = torch::from_blob(
//         colors.data(), { (int64_t)N, (int64_t)3 },
//         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
//     ).clone().div_(255.0f);
//     // ─────────────────────────────────────────────────────────────────────
//     // 1) Ask Python to compute a fresh bounding from the *incoming points*
//     //    the same way as in createFromPcd (pcd mode).
//     // ─────────────────────────────────────────────────────────────────────
//     namespace py = pybind11;
//     py::gil_scoped_acquire gil;

//     static py::module bu_mod = []{
//         py::module sys = py::module::import("sys");
//         sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
//         return py::module::import("src.utils.bounding_utils");
//     }();
//     py::object decide_main_bounding = bu_mod.attr("decide_main_bounding");

//     // Wrap xyz_cpu as Python tensor → numpy → SimpleNamespace(points=…)
//     py::object torch_tensor_py = py::cast(xyz_cpu);
//     py::object np_array = torch_tensor_py.attr("numpy")();
//     py::object SimpleNS = py::module_::import("types").attr("SimpleNamespace");
//     py::object pcd_obj  = SimpleNS(py::arg("points") = np_array);
//     py::object bounding_py = decide_main_bounding(
//         py::arg("bound_mode")       = "pcd",
//         py::arg("pcd_density_rate") = 0.1,
//         py::arg("bound_scale")      = 1.0,
//         py::arg("pcd")              = pcd_obj
//     );
//     // Convert py ndarray [2,3] → torch CPU tensor
//     auto arr       = bounding_py.cast<py::array_t<float>>();
//     py::buffer_info info = arr.request();
//     TORCH_CHECK(info.ndim == 2 && info.shape[0] == 2 && info.shape[1] == 3,
//                 "decide_main_bounding returned invalid shape");
//     torch::Tensor bounding_cpu = torch::from_blob(
//         info.ptr, { (int64_t)2, (int64_t)3 },
//         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
//     ).clone();

//     // ---- LOG: new raw bbox from Python (red) ----
//     int64_t inc = ++rr_inc_step_;  // advance a global “time” for Rerun
//     rrLogPointsAndAABB(
//         /*iteration=*/static_cast<int>(inc),
//         /*xyz=*/xyz_cpu, /*rgb=*/rgb_cpu,
//         /*bounding_2x3=*/bounding_cpu,
//         /*tag=*/"increase_raw",
//         /*max_points_for_viz=*/200000,
//         /*log_points=*/true, /*log_box=*/true,
//         /*inc=*/inc
//     );

//     // Convert to the “used” cube the same way as in createFromPcd:
//     // used_min/used_max are center +/- 0.5*extent_side where extent_side = max(range per axis)
//     torch::Tensor scene_min_cpu = bounding_cpu[0];
//     torch::Tensor scene_max_cpu = bounding_cpu[1];
//     torch::Tensor center_cpu    = 0.5f * (scene_min_cpu + scene_max_cpu);
//     float extent_side           = (scene_max_cpu - scene_min_cpu).max().item<float>();
//     torch::Tensor used_min_cpu  = center_cpu - 0.5f * extent_side;
//     torch::Tensor used_max_cpu  = center_cpu + 0.5f * extent_side;

//     auto cpu_opt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
//     // Previous USED box (what we are currently training in)
//     torch::Tensor prev_used_cpu = torch::tensor(
//         { {cur_min_[0], cur_min_[1], cur_min_[2]},
//         {cur_max_[0], cur_max_[1], cur_max_[2]} }, cpu_opt);

//     // ---- LOG: previously used box (green) ----
//     rrLogPointsAndAABB(
//         /*iteration=*/static_cast<int>(inc),
//         /*xyz=*/xyz_cpu, /*rgb=*/rgb_cpu,
//         /*bounding_2x3=*/prev_used_cpu,
//         /*tag=*/"increase_prev_used",
//         /*max_points_for_viz=*/200000,
//         /*log_points=*/false, /*log_box=*/true,
//         /*inc=*/inc
//     );

//     // ─────────────────────────────────────────────────────────────────────
//     // 2) UNION with the previously *used* box (cur_min_/cur_max_)
//     // ─────────────────────────────────────────────────────────────────────
//     float new_used_min[3], new_used_max[3];
//     for (int d=0; d<3; ++d) {
//         const float cand_min = used_min_cpu[d].item<float>();
//         const float cand_max = used_max_cpu[d].item<float>();
//         new_used_min[d] = std::min(cur_min_[d], cand_min);
//         new_used_max[d] = std::max(cur_max_[d], cand_max);
//     }
//     // If union == current (within small epsilon), nothing to do
//     auto nearly_equal = [](float a, float b, float eps=1e-6f){ return std::fabs(a-b) <= eps; };
//     bool same_box = true;
//     for (int d=0; d<3; ++d) {
//         if (!nearly_equal(new_used_min[d], cur_min_[d]) ||
//             !nearly_equal(new_used_max[d], cur_max_[d])) {
//             same_box = false; break;
//         }
//     }
//     if (same_box) {
//         // Still fully inside (union didn’t grow) → keep training
//         return;
//     }
//     // ─────────────────────────────────────────────────────────────────────
//     // 3) Prepare NEW union box tensors & new lattice (fixed level)
//     // ─────────────────────────────────────────────────────────────────────
//     // Fixed init level (e.g., 6 ⇒ 64^3) to keep memory bounded
//     static constexpr int kInitLevel = 6;

//     // Make the union cubic (like “used”): take side = max(range) over d,
//     // centered at union center.
//     const float ucen[3] = {
//         0.5f * (new_used_min[0] + new_used_max[0]),
//         0.5f * (new_used_min[1] + new_used_max[1]),
//         0.5f * (new_used_min[2] + new_used_max[2])
//     };
//     const float urange[3] = {
//         new_used_max[0] - new_used_min[0],
//         new_used_max[1] - new_used_min[1],
//         new_used_max[2] - new_used_min[2]
//     };
//     float uni_extent = std::max(std::max(urange[0], urange[1]), urange[2]);
//     float uni_min[3] = { ucen[0] - 0.5f*uni_extent,
//                          ucen[1] - 0.5f*uni_extent,
//                          ucen[2] - 0.5f*uni_extent };
//     float uni_max[3] = { ucen[0] + 0.5f*uni_extent,
//                          ucen[1] + 0.5f*uni_extent,
//                          ucen[2] + 0.5f*uni_extent };

//     // New voxel edge length from fixed level
//     const float new_vox = uni_extent / float(1 << kInitLevel);

//     // Save old lattice & min for corner-world reconstruction
//     const float old_vox = global_vox_;
//     const float old_min[3] = { cur_min_[0], cur_min_[1], cur_min_[2] };

//     // ─────────────────────────────────────────────────────────────────────
//     // 5) Re-init Python SVM with the **union** box (fixed kInitLevel)
//     // ─────────────────────────────────────────────────────────────────────
//     {
//         static py::module svm_mod = []{
//             py::module sys = py::module::import("sys");
//             sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
//             return py::module::import("src.sparse_voxel_model");
//         }();
//         py::object SVM = svm_mod.attr("SparseVoxelModel");
//         py_->svm = SVM(py::arg("sh_degree")        = max_sh_degree_,
//                        py::arg("black_background") = true);

//         auto cpu_opt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
//         torch::Tensor union_bound_cpu = torch::tensor(
//             { {uni_min[0], uni_min[1], uni_min[2]},
//               {uni_max[0], uni_max[1], uni_max[2]} }, cpu_opt);

//         // ---- LOG: union box (green) ----
//         rrLogPointsAndAABB(
//             /*iteration=*/static_cast<int>(inc),
//             /*xyz=*/xyz_cpu, /*rgb=*/rgb_cpu,
//             /*bounding_2x3=*/union_bound_cpu,
//             /*tag=*/"increase_union_used",
//             /*max_points_for_viz=*/200000,
//             /*log_points=*/false, /*log_box=*/true,
//             /*inc=*/inc
//         );

//         py_->svm.attr("model_init")(
//             py::arg("bounding")       = union_bound_cpu.to(device_type_),
//             py::arg("outside_level")  = 0,
//             py::arg("init_n_level")   = kInitLevel,
//             py::arg("init_out_ratio") = 2,
//             py::arg("sh_degree_init") = 3,
//             py::arg("geo_init")       = -10.0f,
//             py::arg("sh0_init")       = 0.5f,
//             py::arg("shs_init")       = 0.0f
//         );

//         auto fetch = [&](const char* name){ return py_->svm.attr(name).cast<torch::Tensor>(); };
//         oct_path_      = fetch("octpath");
//         oct_level_     = fetch("octlevel");
//         center_        = fetch("vox_center");                  // [Nnew,3]
//         size_          = fetch("vox_size").squeeze(1);         // [Nnew]
//         vox_size_inv_  = 1.0f / size_;
//         grid_pts_key_  = fetch("grid_pts_key");                // [NgNew,3]
//         vox_key_       = fetch("vox_key");

//         _geo_grid_pts_ = fetch("_geo_grid_pts").requires_grad_(true);
//         sh0_           = fetch("_sh0").requires_grad_(true);
//         shs_           = fetch("_shs").requires_grad_(true);
//         subdiv_p_      = fetch("_subdiv_p").requires_grad_(true);

//         max_w_ = torch::zeros({center_.size(0), 1},
//                               torch::dtype(torch::kFloat32).device(device_type_));
//     }
//     // rrLogVoxelBoxes(center_, size_, /*max_boxes=*/20000, "boxes", inc);
//     // Update NEW globals & AABB to UNION
//     global_vox_       = new_vox;
//     global_origin_[0] = uni_min[0];
//     global_origin_[1] = uni_min[1];
//     global_origin_[2] = uni_min[2];
//     for (int d=0; d<3; ++d) { cur_min_[d] = uni_min[d]; cur_max_[d] = uni_max[d]; }
//     have_cur_aabb_ = true;
// }

// void VoxelModel::increasePcd(std::vector<float> pcd_full,
//                              std::vector<float> colors,
//                              const int /*iteration*/)
// {
//     // '''
//     // dense grid
//     // '''
//     namespace py = pybind11;

//     const int N = static_cast<int>(pcd_full.size() / 3);
//     torch::Tensor xyz  = torch::from_blob(pcd_full.data(), {N,3}, torch::kFloat32).clone().to(device_type_);
//     torch::Tensor cols = torch::from_blob(colors.data(),   {N,3}, torch::kFloat32).clone().to(device_type_);

//     // ─── 2) Ask Python to compute the bounding box via decide_main_bounding ──
//     py::gil_scoped_acquire gil;
//     static py::module bu_mod = []{
//         py::module sys = py::module::import("sys");
//         sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
//         return py::module::import("src.utils.bounding_utils");
//     }();
//     py::object decide_main_bounding = bu_mod.attr("decide_main_bounding");

//     // Pass the point cloud to Python as a (N,3) float32 tensor on CPU
//     torch::Tensor xyz_cpu = xyz.cpu().contiguous();
//     // 1) Wrap the CPU tensor as a Python torch.Tensor
//     py::object torch_tensor_py = py::cast(xyz_cpu);
//     // 2) Call its .numpy() in Python to get a NumPy array
//     py::object np_array = torch_tensor_py.attr("numpy")();
//     // 3) Build a SimpleNamespace(points=ndarray) so .points works in Python
//     py::object SimpleNS = py::module_::import("types").attr("SimpleNamespace");
//     py::object pcd_obj  = SimpleNS(py::arg("points") = np_array);

//     py::object bounding_py = decide_main_bounding(
//         py::arg("bound_mode")      = "pcd",   // or "default" / "camera_max" …
//         py::arg("pcd_density_rate")= 0.1,
//         py::arg("bound_scale")     = 1.0,
//         py::arg("pcd")             = pcd_obj  // just needs .points attribute-like
//     );
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
//     this->scene_extent_       = torch::tensor({extent_side},
//                                  torch::TensorOptions().dtype(torch::kFloat32)
//                                                         .device(device_type_));
//     // std::cout << "[Debug] scene_extent = " << extent_side << std::endl;
//     // float float_level = -std::log2(0.05f / extent_side);
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
//     py::object SVM     = svm_mod.attr("SparseVoxelModel");
//     py_->svm = SVM(
//         py::arg("sh_degree")        = max_sh_degree_,
//         py::arg("black_background") = true
//     );
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

//     // ─── 4) Pull back all core tensors from the Python object ───────────
//     auto fetch = [&](const char* name){
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
//     this->_geo_grid_pts_ = fetch("_geo_grid_pts").requires_grad_(true);
//     this->sh0_ = fetch("_sh0").requires_grad_(true);
//     this->shs_ = fetch("_shs").requires_grad_(true);
//     this->subdiv_p_ = fetch("_subdiv_p").requires_grad_(true);
//     // subdiv_p_   .retain_grad();
//     // ─── 6) Stats buffers (exactly as before) ───────────────────────────
//     this->max_w_ = torch::zeros({center_.size(0), 1},
//         torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

//     // // ─── 7) Register with the optimizer, etc. ──────────────────────────
//     // VOXEL_MODEL_TENSORS_TO_VEC
//     syncFromPython_();
// }

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
    torch::Tensor new_param = torch::cat({old_param, add_rows}, /*dim=*/0).contiguous().requires_grad_(true);

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

