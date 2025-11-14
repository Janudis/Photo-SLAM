#include "include_voxel/voxel_model.h"
#include <fstream>
#include <sstream>

// namespace py = pybind11;
// using namespace py::literals;  // enables "name"_a syntax
namespace sv {

// struct __attribute__((visibility("hidden"))) VoxelModel::PyState {
//     py::object svm;        // Python SparseVoxelModel
//     py::object optimizer_py;  // Python SparseAdam
//     py::object scheduler_py; 

//     ~PyState() {
//         py::gil_scoped_acquire gil;
//         svm = py::none();
//         optimizer_py = py::none();
//         scheduler_py = py::none();
//     }
// };
// VoxelModel::~VoxelModel() {
//     py::gil_scoped_acquire gil;
//     py_.reset(); // drops optimizer & svm safely under GIL
// }

VoxelModel::~VoxelModel() = default;   // <-- define here, once

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
    // py_ = std::make_unique<PyState>();
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
    // py_ = std::make_unique<PyState>();
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

// === VoxelModel: SVRaster-parity accessors & lazy caches ===

int64_t sv::VoxelModel::num_grid_pts() const {
    // SVRaster: len(self.grid_pts_key)
    return grid_pts_key().defined() ? grid_pts_key().size(0) : 0;
}

sv::VoxelModel::DerivedSignature sv::VoxelModel::signature() const {
    DerivedSignature s;
    s.num_voxels    = (oct_path_.defined() ? oct_path_.size(0) : 0);
    s.octpath_impl  = (oct_path_.defined() ? oct_path_.unsafeGetTensorImpl() : nullptr);
    s.octlevel_impl = (oct_level_.defined() ? oct_level_.unsafeGetTensorImpl() : nullptr);
    return s;
}

void sv::VoxelModel::_check_derived_voxel_attr() const {
    const auto sig = signature();
    const bool need =
        (_check_derived_voxel_attr_signature_ != sig) ||
        !center_.defined() || !size_.defined() ||
        !grid_pts_key_.defined() || !vox_key_.defined();

    if (!need) return;

    TORCH_CHECK(oct_path_.defined() && oct_level_.defined(),
        "_check_derived_voxel_attr requires oct_path_ and oct_level_ to be set.");

    at::Tensor vox_center, vox_size;
    std::tie(vox_center, vox_size) = sv::oct::octpath_decoding(
        oct_path_.contiguous(),           // [N,1] int64
        oct_level_.contiguous(),          // [N,1] int8
        scene_center_.contiguous(),       // [3]   float32
        scene_extent_.contiguous());      // [1]   float32

    at::Tensor grid_key, vkey;
    std::tie(grid_key, vkey) = sv::oct::build_grid_pts_link(
        oct_path_.contiguous(),           // [N,1] int64
        oct_level_.contiguous());         // [N,1] int8

    center_       = vox_center.contiguous();  // [N,3] float32
    size_         = vox_size.contiguous();    // [N,1] float32
    grid_pts_key_ = grid_key.contiguous();    // [M,3] int64
    vox_key_      = vkey.contiguous();        // [N,8] int64

    _check_derived_voxel_attr_signature_ = sig;
}

const at::Tensor& sv::VoxelModel::vox_center()   const { _check_derived_voxel_attr(); return center_; }
const at::Tensor& sv::VoxelModel::vox_size()     const { _check_derived_voxel_attr(); return size_; }
const at::Tensor& sv::VoxelModel::grid_pts_key() const { _check_derived_voxel_attr(); return grid_pts_key_; }
const at::Tensor& sv::VoxelModel::vox_key()      const { _check_derived_voxel_attr(); return vox_key_; }

const at::Tensor& sv::VoxelModel::vox_size_inv() const {
    const auto sig = signature();
    const bool need = (_vox_size_inv_signature_ != sig) || !vox_size_inv_.defined();
    if (need) {
        // triggers _check_derived_voxel_attr() via vox_size()
        vox_size_inv_ = (1.0f / vox_size()).contiguous(); // [N,1]
        _vox_size_inv_signature_ = sig;
    }
    return vox_size_inv_;
}

const at::Tensor& sv::VoxelModel::grid_pts_xyz() const {
    const auto sig = signature();
    const bool need = (_grid_pts_xyz_signature_ != sig) || !_grid_pts_xyz_.defined();
    if (need) {
        _grid_pts_xyz_ = sv::oct::compute_gridpoints_xyz(
            grid_pts_key(),               // [M,3] int64
            scene_center_.contiguous(),   // [3]
            scene_extent_.contiguous()    // [1]
        ).contiguous();                   // [M,3] float32
        _grid_pts_xyz_signature_ = sig;
    }
    return _grid_pts_xyz_;
}

using torch::indexing::Slice;
// NumPy-style median for 1-D tensor: average the two middle values when N is even.
static at::Tensor median1d_numpy(const at::Tensor& v1d_cpu) {
    TORCH_CHECK(v1d_cpu.dim()==1 && v1d_cpu.device().is_cpu(),
                "median1d_numpy expects 1-D CPU tensor");
    auto sorted = std::get<0>(v1d_cpu.sort(0));   // [N]
    const int64_t N = sorted.size(0);
    TORCH_CHECK(N > 0, "median of empty tensor");

    if (N & 1) {
        // odd
        return sorted.index({N/2});
    } else {
        // even -> average the two middles
        auto a = sorted.index({N/2 - 1});
        auto b = sorted.index({N/2});
        return (a + b) * 0.5f;
    }
}
// Mirrors Python main_scene_bound_pcd_heuristic
static std::pair<at::Tensor, float>
compute_scene_bound_heuristic(const at::Tensor& xyz_any, float pcd_density_rate) {
    TORCH_CHECK(xyz_any.dim()==2 && xyz_any.size(1)==3, "xyz must be [N,3]");
    const auto dev  = xyz_any.device();
    const auto N    = xyz_any.size(0);
    TORCH_CHECK(N > 0, "empty point cloud");

    // Work on CPU for sorting/median
    auto xyz_cpu = xyz_any.to(torch::kCPU).contiguous();

    // 1) center = median(points, axis=0)  (NumPy semantics)
    auto x_med = median1d_numpy(xyz_cpu.index({Slice(), 0}));
    auto y_med = median1d_numpy(xyz_cpu.index({Slice(), 1}));
    auto z_med = median1d_numpy(xyz_cpu.index({Slice(), 2}));
    std::vector<at::Tensor> meds = {x_med, y_med, z_med};
    at::Tensor center_cpu = at::stack(meds).contiguous(); // [3], CPU

    // 2) dist = max(abs(points - center), axis=1)   (L∞ distance)
    at::Tensor diffs = (xyz_cpu - center_cpu.unsqueeze(0)).abs_(); // [N,3]
    at::Tensor dist  = std::get<0>(diffs.max(/*dim=*/1));          // [N], CPU

    // 3) sort distances ascending
    at::Tensor dist_sorted = std::get<0>(dist.sort(0));            // [N]

    // 4) density = (1+arange(N))*(dist>0) / ((2*dist)^3 + 1e-6)
    at::Tensor idx1 = at::arange(1, N+1, dist_sorted.options().dtype(at::kFloat)); // [N]
    at::Tensor positive = dist_sorted.gt(0).to(at::kFloat);                         // [N]
    at::Tensor denom = (2.0f * dist_sorted).pow(3).add_(1e-6f);                     // [N]
    at::Tensor density = idx1.mul_(positive).div_(denom);                           // [N]

    // 5) begin_idx = round(0.05 * N)
    int64_t begin_idx = static_cast<int64_t>(std::llround(0.05 * static_cast<double>(N)));
    begin_idx = std::min<int64_t>(std::max<int64_t>(begin_idx, 0), N-1);

    // 6) max_idx = begin_idx + argmax(density[begin_idx:])
    at::Tensor density_tail = density.index({Slice(begin_idx, torch::indexing::None)});
    int64_t argmax_tail = std::get<1>(density_tail.max(0)).item<int64_t>();
    int64_t max_idx = begin_idx + argmax_tail;

    // 7) target_density = pcd_density_rate * density[max_idx]
    float target_density = pcd_density_rate * density.index({max_idx}).item<float>();

    // 8) target_idx = first index after max where density < target_density
    at::Tensor tail_after_peak = density.index({Slice(max_idx, torch::indexing::None)});
    at::Tensor below = tail_after_peak.lt(target_density).nonzero(); // [K,1] or empty
    int64_t target_idx = N - 1;
    if (below.numel() > 0) {
        target_idx = max_idx + below.index({0, 0}).item<int64_t>();
    }

    // 9) radius
    float radius_f = dist_sorted.index({target_idx}).item<float>();

    // Return center on original device
    at::Tensor center_t = center_cpu.to(dev, /*non_blocking=*/true).contiguous(); // [3] on dev
    return {center_t, radius_f};
}

void VoxelModel::createFromPcd(const std::map<point3D_id_t, Point3D>& pcd,
                               const std::vector<sv::MiniCam>& cams)
{
    std::cout << "VoxelModel::createFromPcd() called with " << pcd.size() << " points.\n";

    TORCH_CHECK(global_scene_extent_ > 0.f, "global_scene_extent_ must be set (>0).");
    TORCH_CHECK(fixed_vox_size_     > 0.f, "fixed_vox_size_ must be set (>0).");
    TORCH_CHECK(max_sh_degree_     >= 0,   "max_sh_degree_ must be >= 0.");

    const int N = static_cast<int>(pcd.size());
    if (N == 0) {
        std::cerr << "[createFromPcd] Empty PCD — nothing to do.\n";
        return;
    }

    auto dev = torch::kCUDA;
    auto f32 = torch::dtype(torch::kFloat32).device(dev);
    auto i64 = torch::dtype(torch::kLong).device(dev);
    auto i8  = torch::dtype(torch::kInt8).device(dev);

    // ---- 0) Scene (center/extent, cached tensors) ----------------------------------
    scene_center_ = torch::tensor(
        {global_scene_center_[0], global_scene_center_[1], global_scene_center_[2]},
        torch::dtype(torch::kFloat32).device(dev)
    ).contiguous(); // [3]

    const int  outside_level = 0;
    const float scene_extent_scalar = global_scene_extent_ * std::pow(2.0f, outside_level);
    scene_extent_ = torch::tensor({scene_extent_scalar}, torch::dtype(torch::kFloat32).device(dev)).contiguous(); // [1]
    inside_extent_ = torch::tensor({global_scene_extent_}, torch::dtype(torch::kFloat32).device(dev)).contiguous(); // [1]
    scene_min_t_ = (scene_center_ - 0.5f * scene_extent_).contiguous(); // [3]

    // Pack PCD to tensors
    torch::Tensor xyz = torch::empty({N,3}, torch::dtype(torch::kFloat32).device(dev));
    torch::Tensor rgb = torch::empty({N,3}, torch::dtype(torch::kFloat32).device(dev));
    {
        int i = 0;
        for (const auto& kv : pcd) {
            const auto& P = kv.second;
            xyz[i][0] = P.xyz_(0); xyz[i][1] = P.xyz_(1); xyz[i][2] = P.xyz_(2);
            rgb[i][0] = P.color_(0); rgb[i][1] = P.color_(1); rgb[i][2] = P.color_(2);
            ++i;
        }
    }
    // ---- 1) Choose voxel level from target voxel size ------------------------------
    const int MAX_L = max_num_levels_; // keep in sync with SVRaster MAX_NUM_LEVELS
    auto vox_size_t = torch::full({1}, fixed_vox_size_, torch::dtype(torch::kFloat32).device(dev));          // [1,1]
    auto L_fp       = sv::oct::voxsize2levelf(scene_extent_, vox_size_t).round();                      // [1,1]
    auto L_clamped  = L_fp.clamp_min(1).clamp_max(MAX_L).to(torch::kInt8).contiguous();                   // [1,1] int8
    octlevel_       = L_clamped.item<int8_t>();                                                        // scalar cache
    vox_eff_    = sv::oct::level2voxsize(scene_extent_, L_clamped.view({1,1})).view({1,1}).contiguous();       // [1,1]

    // ---- 2) Quantize to ijk at this level; unique; fuse colors per voxel ----------
    auto vox_effN = vox_eff_.expand({N,1}); // [N,1]
    at::Tensor ijk = ((xyz - scene_min_t_) / vox_effN).floor().to(at::kLong); 

    at::Tensor L_N      = at::full({N,1}, octlevel_, at::dtype(torch::kInt8).device(dev)).contiguous(); // [N,1]
    at::Tensor L_N_long = L_N.to(at::kLong);
    at::Tensor ijkl     = at::cat({ijk, L_N_long}, /*dim=*/1); // [N,4] = (i,j,k,L)

    at::Tensor ijkl_unq, invmap;
        {
            auto tup = at::unique_dim(ijkl.contiguous(), /*dim=*/0, /*sorted=*/true, /*return_inverse=*/true);
            ijkl_unq = std::get<0>(tup).contiguous(); // [Nu,4]
            invmap   = std::get<1>(tup).contiguous(); // [N]
        }

    torch::Tensor ijk_u, L_u; 
    auto parts = torch::split_with_sizes(ijkl_unq, {3, 1}, /*dim=*/1); 
    ijk_u = parts[0].contiguous(); L_u = parts[1].to(torch::kInt8).contiguous(); 
    L_u = L_u.to(torch::kInt8).contiguous(); // [Nu,1]
    int64_t Nu = ijk_u.size(0);

    at::Tensor rgb_u = at::zeros({Nu,3}, torch::dtype(torch::kFloat32).device(dev));
    rgb_u.index_reduce_(/*dim=*/0, /*index=*/invmap, /*source=*/rgb, /*reduce=*/"mean", /*include_self=*/false);

    // Bounds: 0 <= ijk < 2^L
    const int  L0    = L_u[0].item<int8_t>();        // (all rows same level here)
    const long limit = (1L << L0);
    TORCH_CHECK((ijk_u.index({at::indexing::Slice(),0}) >= 0).all().item<bool>() &&
                (ijk_u.index({at::indexing::Slice(),1}) >= 0).all().item<bool>() &&
                (ijk_u.index({at::indexing::Slice(),2}) >= 0).all().item<bool>(),
                "Points below scene_min — enlarge global_scene_extent_ or filter.");
    TORCH_CHECK((ijk_u.index({at::indexing::Slice(),0}) < limit).all().item<bool>() &&
                (ijk_u.index({at::indexing::Slice(),1}) < limit).all().item<bool>() &&
                (ijk_u.index({at::indexing::Slice(),2}) < limit).all().item<bool>(),
                "Points exceed scene bounds — enlarge global_scene_extent_ or filter.");

    // ---- 3) ijk -> octpath (pure C++) ---------------------------------------------
    at::Tensor octpath = sv::rasterizer::ijk_2_octpath(ijk_u.contiguous(), L_u.contiguous()); // [Nu,1] int64

    // ---- 3b) Optional filtering (mark_max_samp_rate / mark_near) -------------------
    if (!cams.empty()) {
        at::Tensor vox_center, vox_size;
        std::tie(vox_center, vox_size) = sv::oct::octpath_decoding(octpath, L_u, scene_center_, scene_extent_); // [Nu,3],[Nu,1]

        const int64_t Nu_before = octpath.size(0);
        at::Tensor rate     = sv::rasterizer::mark_max_samp_rate(cams, octpath, vox_center, vox_size, /*near=*/0.02f); // [Nu]
        at::Tensor kept     = rate > 0;
        const float near_th = 0.2f;
        const int64_t n_rate_pos = kept.sum().item<int64_t>();
        int64_t n_near_hit = 0;

        if (near_th > 0.f) {
            at::Tensor is_near = sv::rasterizer::mark_near(cams, octpath, vox_center, vox_size, near_th); // [Nu] bool
            kept = kept & (~is_near);
            n_near_hit = is_near.sum().item<int64_t>();
        }

        auto idx = at::nonzero(kept).view({-1});
        if (idx.size(0) > 0 && idx.size(0) < octpath.size(0)) {
            octpath = octpath.index_select(0, idx).contiguous();
            L_u     = L_u.index_select(0, idx).contiguous();
            rgb_u   = rgb_u.index_select(0, idx).contiguous();
        }
        Nu = octpath.size(0);

        std::cout << "[filter] Nu_before=" << Nu_before
                  << " rate>0=" << n_rate_pos
                  << " near_hit=" << n_near_hit
                  << " kept_final=" << Nu << std::endl;
    }

    // --- assign topology tensors FIRST (SVRaster signature drives lazy caches)
    oct_path_  = octpath.contiguous();  // [N,1] int64
    oct_level_ = L_u.contiguous();      // [N,1] int8

    // Invalidate derived caches (let lazy props rebuild them)
    _check_derived_voxel_attr_signature_ = {};
    _vox_size_inv_signature_ = {};
    _grid_pts_xyz_signature_ = {};
    center_        = at::Tensor();
    size_          = at::Tensor();
    grid_pts_key_  = at::Tensor();
    vox_key_       = at::Tensor();
    vox_size_inv_  = at::Tensor();
    _grid_pts_xyz_ = at::Tensor();

    (void)vox_center();   // fills center_
    (void)vox_size();     // fills size_
    (void)grid_pts_key(); // fills grid_pts_key_
    (void)vox_key();      // fills vox_key_
    (void)vox_size_inv();// fills vox_size_inv_

    std::cout << "[createFromPcd] Nu=" << Nu
        << " vox_key_.shape=" << vox_key_.sizes() 
        << " center_shape=" << center_.sizes() 
        << " size_shape=" << size_.sizes() 
        << " grid_pts_key_shape=" << grid_pts_key_.sizes()
        << " vox_size_inv_shape=" << vox_size_inv_.sizes()
        << " _grid_pts_xyz_shape=" << _grid_pts_xyz_.sizes()
        << std::endl;

    // --- init learnables that depend on N (number of voxels)
    const int n_sh_rest = (max_sh_degree_ + 1)*(max_sh_degree_ + 1) - 1;
    subdiv_p_ = at::ones({Nu,1}, f32).requires_grad_();                     // [N,1]
    sh0_      = sv::act::rgb2shzero(rgb_u.contiguous()).contiguous().requires_grad_();  // [N,3]
    shs_      = at::zeros({Nu, n_sh_rest, 3}, f32).requires_grad_();        // [N,M-1,3]
    // --- init grid field AFTER topology (depends on M = num_grid_pts)
    const int64_t M = num_grid_pts();                                       // triggers lazy link build
    _geo_grid_pts_ = at::full({M,1}, -10.0f, f32).requires_grad_();         // [M,1]
    // housekeeping
    active_sh_degree_ = std::min(max_sh_degree_, 3);
    max_w_ = at::zeros({Nu,1}, f32);

    // Register with your optimizer
    VOXEL_MODEL_TENSORS_TO_VEC;
    // Optional visualization hooks:
    // const int64_t inc0 = 0;
    // rrLogGlobalSceneAABB(inc0);
    // rrLogVoxelBoxes(this->center_, this->size_, /*max_boxes_for_viz=*/10000000000, "vox_all", inc0);
}

void VoxelModel::increasePcd(std::vector<float> pcd_full,
                             std::vector<float> colors,
                             const int /*iteration*/,
                             const std::vector<sv::MiniCam>& cams)
{
    static int64_t inc_counter = 1;

    const int Nf = static_cast<int>(pcd_full.size());
    if (Nf < 3 || colors.size() < 3) return;

    const int N = Nf / 3;
    std::cout << "VoxelModel::increasePcd() called with " << N << " points.\n";

    TORCH_CHECK(global_scene_extent_ > 0.f && fixed_vox_size_ > 0.f,
                "increasePcd: scene extent / fixed vox size not set.");
    // Must have a valid base created (we rely on cached scene_, vox_eff_, octlevel_)
    TORCH_CHECK(this->oct_path_.defined() && this->oct_level_.defined(),
                "increasePcd: topology not initialized; call createFromPcd() first.");

    auto dev = torch::kCUDA;
    auto f32 = torch::dtype(torch::kFloat32).device(dev);
    auto i64 = torch::dtype(torch::kLong).device(dev);
    auto i8  = torch::dtype(torch::kInt8).device(dev);

    // --- 0) Convert raw arrays to CUDA tensors (XYZ in world, RGB normalized [0,1]) ----
    at::Tensor xyz = at::from_blob(pcd_full.data(), {(int64_t)N, 3},
                        at::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                        .clone().to(dev);
    at::Tensor rgb = at::from_blob(colors.data(), {(int64_t)N, 3},
                        at::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                        .clone().div_(255.0f).to(dev);
    // --- 1) Quantize to ijk at cached level/voxel size (same as createFromPcd) --------
    TORCH_CHECK(vox_eff_.defined() && vox_eff_.numel()==1, "increasePcd: vox_eff_ not cached.");
    TORCH_CHECK(octlevel_ >= 1, "increasePcd: invalid cached octlevel_.");
    auto vox_effN = vox_eff_.expand({N,1});                       // [N,1]
    at::Tensor ijk = ((xyz - scene_min_t_) / vox_effN).to(at::kLong);  // [N,3]

    at::Tensor L_N      = at::full({N,1}, (int8_t)octlevel_, at::dtype(torch::kInt8).device(dev)); // [N,1] i8
    at::Tensor L_N_long = L_N.to(at::kLong);
    at::Tensor ijkl     = at::cat({ijk, L_N_long}, 1);               // [N,4] = (i,j,k,L)

    // unique (ijk,L) with inverse to fuse colors per-voxel
    at::Tensor ijkl_unq, invmap;
    {
        auto tup = at::unique_dim(ijkl.contiguous(), /*dim=*/0, /*sorted=*/true, /*return_inverse=*/true);
        ijkl_unq = std::get<0>(tup).contiguous(); // [Nu,4]
        invmap   = std::get<1>(tup).contiguous(); // [N]
    }

    // at::Tensor ijk_u = ijkl_unq.index({at::indexing::Slice(), at::indexing::Slice(0,3)}).contiguous();              // [Nu,3]
    // at::Tensor L_u   = ijkl_unq.index({at::indexing::Slice(), 3}).to(torch::kInt8).view({-1,1}).contiguous();          // [Nu,1]
    // int64_t Nu       = ijk_u.size(0);
    torch::Tensor ijk_u, L_u;
    auto parts = torch::split_with_sizes(ijkl_unq, {3,1}, 1);
    ijk_u = parts[0].contiguous();                                                                    // [Nu,3]
    L_u   = parts[1].to(torch::kInt8).contiguous();                                                   // [Nu,1]
    int64_t Nu = ijk_u.size(0);

    at::Tensor rgb_u = at::zeros({Nu,3}, torch::dtype(torch::kFloat32).device(dev));
    rgb_u.index_reduce_(/*dim=*/0, /*index=*/invmap, /*source=*/rgb, /*reduce=*/"mean", /*include_self=*/false);

    // simple bounds check 0 <= ijk < 2^L
    const int  L0    = L_u[0].item<int8_t>();
    const long limit = (1L << L0);
    bool ok_low  =
        (ijk_u.index({at::indexing::Slice(),0}) >= 0).all().item<bool>() &&
        (ijk_u.index({at::indexing::Slice(),1}) >= 0).all().item<bool>() &&
        (ijk_u.index({at::indexing::Slice(),2}) >= 0).all().item<bool>();
    bool ok_high =
        (ijk_u.index({at::indexing::Slice(),0}) < limit).all().item<bool>() &&
        (ijk_u.index({at::indexing::Slice(),1}) < limit).all().item<bool>() &&
        (ijk_u.index({at::indexing::Slice(),2}) < limit).all().item<bool>();
    if (!(ok_low && ok_high)) {
        std::cout << "[increasePcd] Points out of cached scene bounds; skipping batch.\n";
        return;
    }

    // --- 2) Build octpath for these candidates --------------------------------------
    at::Tensor octpath_new = sv::rasterizer::ijk_2_octpath(ijk_u.contiguous(), L_u.contiguous()); // [Nu,1] i64

    // --- 3) Optional camera-based filtering (rate>0 & not near) ---------------------
    if (!cams.empty()) {
        // Precompute centers/sizes for filtering
        at::Tensor vox_center, vox_size;
        std::tie(vox_center, vox_size) = sv::oct::octpath_decoding(octpath_new, L_u, scene_center_, scene_extent_);

        const int64_t Nu_before = octpath_new.size(0);
        at::Tensor rate = sv::rasterizer::mark_max_samp_rate(cams, octpath_new, vox_center, vox_size, /*near*/0.02f); // [Nu]
        at::Tensor kept = rate > 0;

        const float near_thresh = 0.2f;
        int64_t n_rate_pos = kept.sum().item<int64_t>();
        int64_t n_near_hit = 0;

        if (near_thresh > 0.f) {
            at::Tensor is_near = sv::rasterizer::mark_near(cams, octpath_new, vox_center, vox_size, near_thresh); // [Nu]
            kept = kept & (~is_near);
            n_near_hit = is_near.sum().item<int64_t>();
        }

        auto idx = at::nonzero(kept).view({-1});
        if (idx.numel() == 0) {
            std::cout << "[increasePcd/filter] all candidates filtered out.\n";
            return;
        }
        if (idx.numel() < octpath_new.size(0)) {
            octpath_new = octpath_new.index_select(0, idx).contiguous();
            L_u         = L_u.index_select(0, idx).contiguous();
            rgb_u       = rgb_u.index_select(0, idx).contiguous();
            Nu          = octpath_new.size(0);
        }

        std::cout << "[increasePcd/filter] Nu_before=" << Nu_before
                  << " rate>0=" << n_rate_pos
                  << " near_hit=" << n_near_hit
                  << " kept_final=" << Nu << std::endl;
    }

    // --- 4) Deduplicate vs existing topology (no Python, fast) ----------------------
    // keys: (octpath<<8)|level
    auto octpath_old  = oct_path_.contiguous();                    // [No,1] int64
    auto octlevel_old = oct_level_.contiguous();                   // [No,1] int8

    // at::Tensor key_new = octpath_new.view({-1}).to(torch::kInt64)
    //                         .mul(256)
    //                         .add(L_u.view({-1}).to(torch::kInt64));                                // [Nu]
    //  at::Tensor key_old;
    //  {
    //      TORCH_CHECK(oct_path_.defined() && oct_level_.defined(), "existing topology undefined");
    //      key_old = oct_path_.view({-1}).to(torch::kInt64).mul(256)
    //              + oct_level_.view({-1}).to(torch::kInt64);                                // [No]
    //  }

    // at::Tensor new_mask;
    // if (key_old.numel() == 0) {
    //     new_mask = at::ones({Nu}, at::dtype(at::kBool).device(dev));
    // } else {
    //     // sort old, then searchsorted new, then exact-match check
    //     auto sort_res       = key_old.sort(/*dim=*/0);
    //     at::Tensor old_sorted = std::get<0>(sort_res);                                  // [No]
    //     at::Tensor pos        = at::searchsorted(old_sorted, key_new, /*right=*/false); // [Nu]
    //     // build equality check safely
    //     auto No = old_sorted.size(0);
    //     at::Tensor in_bounds = pos < No;
    //     at::Tensor pos_clamped = at::minimum(pos, at::full_like(pos, No-1));
    //     at::Tensor gathered    = old_sorted.index_select(0, pos_clamped);
    //     at::Tensor eq          = (gathered == key_new);
    //     new_mask = (~(in_bounds & eq)).to(at::kBool);                                    // [Nu] true => not in old
    // }

    // if (!new_mask.any().item<bool>()) {
    //     std::cout << "[increasePcd] No new voxels (all duplicates).\n";
    //     return;
    // }

    // auto sel      = at::nonzero(new_mask).view({-1});           // [Nk]
    // const int Nk  = (int)sel.size(0);
    // at::Tensor octpath_add = octpath_new.index_select(0, sel);  // [Nk,1]
    // at::Tensor L_add       = L_u.index_select(0, sel);          // [Nk,1]
    // at::Tensor rgb_add     = rgb_u.index_select(0, sel);        // [Nk,3]

    // // --- 5) (Optional) prevent re-adding base-level parents if finer children exist --
    // // If your pipeline subdivides (L_old > base_L), disallow adding their base parent.
    // {
    //     if (oct_level_.numel() > 0) {
    //         const int MAX_L  = max_num_levels_;
    //         const int base_L = static_cast<int>(octlevel_);
    //         auto Lold_i64     = oct_level_.view({-1}).to(torch::kInt64); // [No]
    //         auto has_children = (Lold_i64 > base_L);
    //         if (has_children.any().item<bool>()) {
    //             const int levels_below  = std::max(0, MAX_L - base_L);
    //             const int bits_to_clear = 3 * levels_below;
    //             long long lower_mask = (bits_to_clear > 0) ? ((1LL << bits_to_clear) - 1LL) : 0LL;
    //             long long keep_mask_ll = ~lower_mask;

    //             auto op_old_i64  = oct_path_.view({-1}).to(torch::kInt64);           // [No]
    //             auto keep_mask    = at::full({1}, (int64_t)keep_mask_ll,
    //                                   at::TensorOptions().dtype(torch::kInt64).device(dev));
    //             auto op_anc_base  = (op_old_i64 & keep_mask);                      // [No]
    //             auto sel_child    = at::nonzero(has_children).view({-1});          // [K]
    //             op_anc_base       = op_anc_base.index_select(0, sel_child);        // [K]
    //             auto key_children_as_parent =
    //                 op_anc_base.mul(256).add(at::full_like(op_anc_base, (int64_t)base_L));

    //             // check collisions for the *selected* new keys only
    //             auto key_new_sel = key_new.index_select(0, sel);                   // [Nk]
    //             auto sort_res2   = key_children_as_parent.sort(0);
    //             auto set_sorted  = std::get<0>(sort_res2);                         // [M]
    //             auto pos2        = at::searchsorted(set_sorted, key_new_sel, false); // [Nk]
    //             auto M           = set_sorted.size(0);
    //             if (M > 0) {
    //                 auto inb   = pos2 < M;
    //                 auto poscl = at::minimum(pos2, at::full_like(pos2, M-1));
    //                 auto got   = set_sorted.index_select(0, poscl);
    //                 auto eq2   = (got == key_new_sel);
    //                 auto coll  = (inb & eq2);                                      // [Nk]
    //                 if (coll.any().item<bool>()) {
    //                     auto keep = (~coll).to(at::kBool);
    //                     // shrink selection consistently
    //                     sel        = sel.index_select(0, at::nonzero(keep).view({-1}));
    //                     octpath_add= octpath_add.index_select(0, at::nonzero(keep).view({-1}));
    //                     L_add      = L_add.index_select(0, at::nonzero(keep).view({-1}));
    //                     rgb_add    = rgb_add.index_select(0, at::nonzero(keep).view({-1}));
    //                     // recompute Nk
    //                     // (don’t forget to refresh Nk for later logs)
    //                 }
    //             }
    //         }
    //     }
    // }
    // const int Nk_final = (int)octpath_add.size(0);
    // if (Nk_final == 0) {
    //     std::cout << "[increasePcd] Nothing to append after parent/child collision filtering.\n";
    //     return;
    // }

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
        auto a = key_new.contiguous().to(torch::kInt64);
        auto b = key_old.contiguous().to(torch::kInt64);
        // bool tensor of shape like `a`: true where a[i] ∈ b
        auto is_dup = at::isin(a, b, /*assume_unique=*/false, /*invert=*/false).to(torch::kBool);                                   // [Nu]
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

                // // If a candidate NEW (octpath, base_L) matches any ancestor of an existing finer voxel,
                // // then inserting that parent would collide with existing children later.
                // py::object isin_py = torch_mod.attr("isin")(
                //     py::cast(key_new), py::cast(key_children_as_parent));
                // auto would_collide_parent = isin_py.cast<torch::Tensor>().to(torch::kBool);  // [Nu]
                // inputs: key_new, key_old  (both 1-D, e.g. int64)
                // ensures correct dtype/layout
                auto c = key_new.contiguous().to(torch::kInt64);
                auto d = key_children_as_parent.contiguous().to(torch::kInt64);
                // bool tensor of shape like `a`: true where a[i] ∈ b
                auto would_collide_parent = at::isin(c, d, /*assume_unique=*/false, /*invert=*/false).to(torch::kBool);

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

    // --- 6) Append topology ---------------------------------------------------------
    // at::Tensor octpath_cur  = at::cat({this->oct_path_,  octpath_add}, 0).contiguous();
    // at::Tensor octlevel_cur = at::cat({this->oct_level_, L_add},       0).contiguous();
    oct_path_  = at::cat({octpath_old,  octpath_add}, 0).contiguous();
    oct_level_ = at::cat({octlevel_old, L_add},       0).contiguous();

    at::Tensor subdiv_add = at::ones({Nk,1}, f32).contiguous();
    subdiv_p_ = at::cat({subdiv_p_, subdiv_add}, 0).contiguous().detach().requires_grad_();

    // --- 7) Init learnables for NEW voxels only
    at::Tensor sh0_add    = sv::act::rgb2shzero(rgb_add.contiguous()).contiguous();
    auto sh0_old = sh0_;
    at::Tensor sh0_curr   = at::cat({sh0_old, sh0_add}, 0).contiguous().detach().requires_grad_(); // [N, M, 3]

    const int n_sh_rest = (max_sh_degree_ + 1)*(max_sh_degree_ + 1) - 1;
    at::Tensor shs_add    = at::zeros({Nk, n_sh_rest, 3}, f32).contiguous();
    auto shs_old = shs_;
    at::Tensor shs_curr = at::cat({shs_old, shs_add}, 0).contiguous().detach().requires_grad_(); // [N+Nk, M, 3]

    // === SVRaster-style: TOPOLOGY FIRST, LAZY DERIVED ===
    // Save M_prev BEFORE topology change
    const int64_t M_prev = grid_pts_key_.size(0); //num_grid_pts(); // may trigger a build if invalid, OK
    // std::cout<<"M_prev increasePcd="<<M_prev<<std::endl;

    // 6b) invalidate derived caches
    _check_derived_voxel_attr_signature_ = {};
    _vox_size_inv_signature_             = {};
    _grid_pts_xyz_signature_             = {};
    center_        = at::Tensor();
    size_          = at::Tensor();
    grid_pts_key_  = at::Tensor(); // becomes news
    vox_key_       = at::Tensor();
    vox_size_inv_  = at::Tensor();
    _grid_pts_xyz_ = at::Tensor();

    (void)vox_center();   // fills center_
    (void)vox_size();     // fills size_
    (void)grid_pts_key(); // fills grid_pts_key_
    (void)vox_key();      // fills vox_key_
    (void)vox_size_inv();// fills vox_size_inv_

    // std::cout << "[increasePcd] Nu=" << Nu
    //     << " vox_key_.shape=" << vox_key_.sizes() 
    //     << " center_shape=" << center_.sizes() 
    //     << " size_shape=" << size_.sizes() 
    //     << " grid_pts_key_shape=" << grid_pts_key_.sizes()
    //     << " vox_size_inv_shape=" << vox_size_inv_.sizes()
    //     << " _grid_pts_xyz_shape=" << _grid_pts_xyz_.sizes()
    //     << std::endl;

    // --- 8) Grow geo grid AFTER topology (depends on M)
    const int64_t M_curr = num_grid_pts(); // triggers grid link build lazily
    // std::cout<<"M_curr increasePcd="<<M_curr<<std::endl;

    // if (M_curr > M_prev) {
    //     at::Tensor grow = at::full({M_curr - M_prev, 1}, -10.0f, f32)
    //                           .contiguous().detach().requires_grad_();
    //     appendGroup_(/*group_idx=*/0, /*add_rows=*/grow, "_geo_grid_pts", &this->_geo_grid_pts_);
    // }
    // appendGroup_(/*group_idx=*/1, /*add_rows=*/sh0_add, "_sh0", &this->sh0_);
    // appendGroup_(/*group_idx=*/2, /*add_rows=*/shs_add, "_shs", &this->shs_);
    
    // Prepare grow only when needed (group 0)
    torch::Tensor grow; // undefined by default
    if (M_curr > M_prev) {
        grow = torch::full({M_curr - M_prev, 1}, -10.0f,
                torch::dtype(torch::kFloat32).device(dev))
            .contiguous().detach().requires_grad_();
    }
    // Important: only call for non-empty additions.
    if (grow.defined() && grow.size(0) > 0) {
        // std::cout << "increasePcd: geo_grid_pts_ " << this->_geo_grid_pts_.sizes() << std::endl;
        appendGroup_(/*group_idx=*/0, /*add_rows=*/grow, /*svm_field_name=*/"_geo_grid_pts",
                    &this->_geo_grid_pts_);
    }
    if (Nk > 0) {
        // std::cout << "increasePcd: sh0_ " << this->sh0_.sizes() << std::endl;
        // std::cout << "increasePcd: shs_ " << this->shs_.sizes() << std::endl;
        appendGroup_(/*group_idx=*/1, /*add_rows=*/sh0_add, /*svm_field_name=*/"_sh0",
                    &this->sh0_);
        appendGroup_(/*group_idx=*/2, /*add_rows=*/shs_add, /*svm_field_name=*/"_shs",
                    &this->shs_);
    }

    // --- 11) (Optional) fill_empty_cells_ via C++ heuristic -------------------------
    int64_t Nm_added = 0;
    if (fill_empty_cells_) {
        // Compute AABB from this batch (mean center + max L2 radius, min radius = 3*vox_size)
        at::Tensor xyz_cpu = xyz.to(torch::kCPU).contiguous();
        auto res = compute_scene_bound_heuristic(xyz, /*pcd_density_rate=*/0.1f);
        at::Tensor center_t = res.first;  // [3] on CUDA if xyz was CUDA
        float radius_f      = res.second;
        
        at::Tensor radius_t = at::full({3}, radius_f, torch::dtype(torch::kFloat32).device(dev));
        at::Tensor bb_min   = (center_t - radius_t).contiguous();                   // [3]
        at::Tensor bb_max   = (center_t + radius_t).contiguous();                   // [3]

        // Build dense box of ijk at base level (stride to cap count)
        auto vox_t = vox_eff_.mean()        // 0-dim CUDA float tensor
                .view({1})
                .repeat({3})       // [3]
                .contiguous();     // CUDA float

        const int64_t grid_limit = (1LL << static_cast<int>(octlevel_));
        at::Tensor ijk_min = ((bb_min - scene_min_t_) / vox_t).floor().to(at::kLong);
        at::Tensor ijk_max = ((bb_max - scene_min_t_) / vox_t).ceil().to(at::kLong) - 1;

        auto zero = at::zeros_like(ijk_min);
        auto lim  = at::full_like(ijk_max, grid_limit - 1);
        ijk_min = at::maximum(ijk_min, zero);
        ijk_max = at::minimum(ijk_max, lim);

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

        if ((ijk_min <= ijk_max).all().item<bool>()) {
            auto ir = at::arange(ijk_min[0].item<int64_t>(), ijk_max[0].item<int64_t>() + 1,
                                 at::dtype(at::kLong).device(dev));
            auto jr = at::arange(ijk_min[1].item<int64_t>(), ijk_max[1].item<int64_t>() + 1,
                                 at::dtype(at::kLong).device(dev));
            auto kr = at::arange(ijk_min[2].item<int64_t>(), ijk_max[2].item<int64_t>() + 1,
                                 at::dtype(at::kLong).device(dev));

            auto grids = at::meshgrid({ir, jr, kr}, /*indexing=*/"ij");
            at::Tensor ijk_box = at::stack(
                { grids[0].contiguous().view({-1}),
                  grids[1].contiguous().view({-1}),
                  grids[2].contiguous().view({-1}) }, 1).contiguous(); // [Nc,3]

            std::cout << "inceease pcd ijk_box size before stride: " << ijk_box.size(0) << std::endl;
            if (max_artifact_cells_ > 0 && ijk_box.size(0) > max_artifact_cells_) {
                std::cout << "[increasePcd] fill_empty_cells_: limiting artifact cells"<< std::endl;
                ijk_box = ijk_box.index({torch::indexing::Slice(0, max_artifact_cells_)});
            }

            at::Tensor L_box = at::full({ijk_box.size(0),1}, octlevel_,
                                        at::dtype(torch::kInt8).device(dev)).contiguous();
            at::Tensor octpath_box = sv::rasterizer::ijk_2_octpath(ijk_box, L_box); // [Nc,1]
            
            auto octpath_cur  = oct_path_;
            auto octlevel_cur = oct_level_;
        
            at::Tensor key_box     = octpath_box.view({-1}).to(torch::kInt64).mul(256)
                                   + L_box.view({-1}).to(torch::kInt64);
            auto key_cur = octpath_cur.view({-1}).to(torch::kInt64).mul(256)
                        + octlevel_cur.view({-1}).to(torch::kInt64);

            // Dedup against current topology
            auto sort_res3     = key_cur.sort(0);
            auto cur_sorted    = std::get<0>(sort_res3);
            auto pos3          = at::searchsorted(cur_sorted, key_box, false);
            auto No3           = cur_sorted.size(0);
            auto inb3          = pos3 < No3;
            auto poscl3        = at::minimum(pos3, at::full_like(pos3, No3-1));
            auto got3          = cur_sorted.index_select(0, poscl3);
            at::Tensor dup3    = inb3 & (got3 == key_box);
            at::Tensor new_mask_box = (~dup3).to(at::kBool);

            // Remove base parents of existing finer children (same logic as above)
            {
                const int MAX_L  = max_num_levels_;
                const int base_L = static_cast<int>(octlevel_);
                if (octpath_cur.numel() > 0) {
                    auto Lold_i64     = octlevel_cur.view({-1}).to(torch::kInt64);
                    auto has_children = (Lold_i64 > base_L);

                    if (has_children.any().item<bool>()) {
                        const int levels_below  = std::max(0, MAX_L - base_L);
                        const int bits_to_clear = 3 * levels_below;
                        long long lower_mask    = (bits_to_clear > 0) ? ((1LL << bits_to_clear) - 1LL) : 0LL;
                        long long keep_mask_ll  = ~lower_mask;

                        auto keep_mask   = at::full({1}, static_cast<int64_t>(keep_mask_ll),
                                            at::TensorOptions().dtype(torch::kInt64).device(dev));
                        auto op_old_i64  = octpath_cur.view({-1}).to(torch::kInt64);
                        auto op_anc_base = (op_old_i64 & keep_mask); 
                        auto sel_child   = at::nonzero(has_children).view({-1});
                        op_anc_base      = op_anc_base.index_select(0, sel_child);

                        auto parents_key = op_anc_base.mul(256).add(at::full_like(op_anc_base, static_cast<int64_t>(base_L)));

                        auto sort_res4   = parents_key.sort(0);
                        auto par_sorted  = std::get<0>(sort_res4);
                        auto pos4        = at::searchsorted(par_sorted, key_box, false);
                        auto Mp          = par_sorted.size(0);
                        if (Mp > 0) {
                            auto inb4   = pos4 < Mp;
                            auto poscl4 = at::minimum(pos4, at::full_like(pos4, Mp-1));
                            auto got4   = par_sorted.index_select(0, poscl4);
                            auto coll   = inb4 & (got4 == key_box);
                            new_mask_box = new_mask_box & (~coll);
                        }
                    }
                }
            }
            if (new_mask_box.any().item<bool>()) {
                auto sel         = at::nonzero(new_mask_box).view({-1});
                Nm_added          = sel.size(0);
                auto octpath_add2      = octpath_box.index_select(0, sel);
                auto L_add2       = L_box.index_select(0, sel);

                // Append topology AGAIN (still no derived recompute)
                this->oct_path_  = at::cat({octpath_cur,  octpath_add2}, 0).contiguous();
                this->oct_level_ = at::cat({octlevel_cur, L_add2},  0).contiguous();

                // Prepare learnables for artifacts
                const int n_sh_rest = (max_sh_degree_ + 1)*(max_sh_degree_ + 1) - 1;
                at::Tensor shs_add2    = at::zeros({Nm_added, n_sh_rest, 3}, f32).contiguous();
                at::Tensor subdiv_add2 = at::ones({Nm_added,1}, f32).contiguous();
                at::Tensor rgb_add2    = at::empty({Nm_added,3}, f32);
                rgb_add2.index_put_({at::indexing::Slice(),0}, artifact_bg_rgb_[0]);
                rgb_add2.index_put_({at::indexing::Slice(),1}, artifact_bg_rgb_[1]);
                rgb_add2.index_put_({at::indexing::Slice(),2}, artifact_bg_rgb_[2]);
                at::Tensor sh0_add2    = sv::act::rgb2shzero(rgb_add2).contiguous();

                // Invalidate caches again
                _check_derived_voxel_attr_signature_ = {};
                _vox_size_inv_signature_             = {};
                _grid_pts_xyz_signature_             = {};
                center_        = at::Tensor();
                size_          = at::Tensor();
                grid_pts_key_  = at::Tensor();
                vox_key_       = at::Tensor();
                vox_size_inv_  = at::Tensor();
                _grid_pts_xyz_ = at::Tensor();

                (void)vox_center();   // fills center_
                (void)vox_size();     // fills size_
                (void)grid_pts_key(); // fills grid_pts_key_
                (void)vox_key();      // fills vox_key_
                (void)vox_size_inv();// fills vox_size_inv_

                // Grow geo grid for the second topology change
                const int64_t M_prev2 = M_curr;
                const int64_t M_curr2 = num_grid_pts(); // triggers lazy link build
                if (M_curr2 > M_prev2) {
                    at::Tensor grow2 = at::full({M_curr2 - M_prev2, 1}, -10.0f, f32)
                                        .contiguous().detach().requires_grad_();
                    appendGroup_(/*group_idx=*/0, grow2, "_geo_grid_pts", &this->_geo_grid_pts_);
                }
                appendGroup_(/*group_idx=*/1, sh0_add2, "_sh0", &this->sh0_);
                appendGroup_(/*group_idx=*/2, shs_add2, "_shs", &this->shs_);
                this->subdiv_p_ = at::cat({subdiv_p_, subdiv_add2}, 0).contiguous().detach().requires_grad_();

            }
        }
    }

    // Refresh stats buffer for final N
    this->max_w_ = at::zeros({this->oct_path_.size(0), 1}, f32);
    VOXEL_MODEL_TENSORS_TO_VEC;

    // --- optional viz (commented) ---------------------------------------------------
    // rrLogGlobalSceneAABB(inc_counter);
    // rrLogVoxelBoxes(this->center_, this->size_, /*cap*/10000000000, "vox_all", inc_counter);
    // const int64_t last_added = (int64_t)Nk_final + Nm_added;
    // if (last_added > 0) { ... log just-added voxels ... }

    ++inc_counter;
}

void VoxelModel::syncFromPython_() {
    // No Python SVM anymore — just refresh derived fields & param vectors from C++ state.

    // Ensure tensors are defined (in case called early)
    if (!center_.defined() || !size_.defined() || !sh0_.defined() || !shs_.defined() ||
        !_geo_grid_pts_.defined() || !oct_path_.defined() || !oct_level_.defined()) {
        VOXEL_MODEL_INIT_TENSORS(device_type_);
    }

    // Recompute cached/derived fields
    if (size_.numel() > 0) {
        vox_size_inv_ = 1.0f / size_;
    } else {
        vox_size_inv_ = size_; // empty, keeps device/dtype
    }

    max_w_ = torch::zeros({center_.size(0), 1},
              torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

    // Ensure learnables are leaf tensors with grad
    if (_geo_grid_pts_.defined()) _geo_grid_pts_ = _geo_grid_pts_.contiguous().detach().requires_grad_(true);
    if (sh0_.defined())           sh0_           = sh0_.contiguous().detach().requires_grad_(true);
    if (shs_.defined())           shs_           = shs_.contiguous().detach().requires_grad_(true);
    if (subdiv_p_.defined())      subdiv_p_      = subdiv_p_.contiguous().detach().requires_grad_(true);

    // Refresh the parameter vectors used by optimizer handling
    VOXEL_MODEL_TENSORS_TO_VEC
}

// void VoxelModel::createTrainer(float geo_lr, float sh0_lr, float shs_lr,
//                                float beta1, float beta2, float eps,
//                                const std::vector<int>& milestones,
//                                float gamma)
// {
//     py::gil_scoped_acquire gil;

//     // --- Optimizer (SparseAdam) ---
//     py::module sadam = py::module::import("svraster_cuda.sparse_adam");
//     py::object SparseAdam = sadam.attr("SparseAdam");

//     // Build param groups with per-group learning rates (SVRaster style)
//     py::list groups;
//     {
//         py::dict g0;
//         g0["params"] = py::make_tuple(this->_geo_grid_pts_);
//         g0["lr"]     = geo_lr;
//         groups.append(g0);
//     }
//     {
//         py::dict g1;
//         g1["params"] = py::make_tuple(this->sh0_);
//         g1["lr"]     = sh0_lr;
//         groups.append(g1);
//     }
//     {
//         py::dict g2;
//         g2["params"] = py::make_tuple(this->shs_);
//         g2["lr"]     = shs_lr;
//         groups.append(g2);
//     }

//     // Same call style as Python (group-specific lrs; global lr unused)
//     py_->optimizer_py = SparseAdam(groups,
//                                    "betas"_a = py::make_tuple(beta1, beta2),
//                                    "eps"_a   = eps);

//     // --- Scheduler (MultiStepLR) ---
//     py::module lr_sched = py::module::import("torch.optim.lr_scheduler");
//     py_->scheduler_py = lr_sched.attr("MultiStepLR")(
//         py_->optimizer_py,
//         "milestones"_a = py::cast(milestones),
//         "gamma"_a = gamma
//     );
// }

void VoxelModel::createTrainer(float geo_lr, float sh0_lr, float shs_lr,
                               float beta1, float beta2, float eps,
                               const std::vector<int>& milestones,
                               float gamma)
{
  optimizer_ = std::make_unique<sv::optim::SparseAdam>();

  // group 0: _geo_grid_pts
  {
    sv::optim::ParamGroup g;
    g.lr = geo_lr; g.beta1 = beta1; g.beta2 = beta2; g.eps = eps;
    g.biased = false; g.sparse = false;
    g.params.push_back(&_geo_grid_pts_);
    optimizer_->add_param_group(g);
  }
  // group 1: sh0
  {
    sv::optim::ParamGroup g;
    g.lr = sh0_lr; g.beta1 = beta1; g.beta2 = beta2; g.eps = eps;
    g.biased = false; g.sparse = false;
    g.params.push_back(&sh0_);
    optimizer_->add_param_group(g);
  }
  // group 2: shs
  {
    sv::optim::ParamGroup g;
    g.lr = shs_lr; g.beta1 = beta1; g.beta2 = beta2; g.eps = eps;
    g.biased = false; g.sparse = false;
    g.params.push_back(&shs_);
    optimizer_->add_param_group(g);
  }

  scheduler_ = std::make_unique<sv::optim::MultiStepLR>(
      optimizer_.get(), milestones, gamma);
  sched_milestones_ = milestones;
  sched_gamma_ = gamma;
}

// std::tuple<double,double,double> VoxelModel::currentLearningRates() const {
//     py::gil_scoped_acquire gil;
//     if (py_->optimizer_py.is_none()) {
//         return {std::numeric_limits<double>::quiet_NaN(),
//                 std::numeric_limits<double>::quiet_NaN(),
//                 std::numeric_limits<double>::quiet_NaN()};
//     }
//     py::list groups = py_->optimizer_py.attr("param_groups");
//     auto g0 = groups[0].cast<py::dict>();
//     auto g1 = groups[1].cast<py::dict>();
//     auto g2 = groups[2].cast<py::dict>();
//     double geo = g0["lr"].cast<double>();
//     double sh0 = g1["lr"].cast<double>();
//     double shs = g2["lr"].cast<double>();
//     return {geo, sh0, shs};
// }

// void VoxelModel::appendGroup_(int group_idx,
//                               const torch::Tensor& add_rows,
//                               const char* /*svm_field_name*/,
//                               torch::Tensor* out_member_param) {
//     namespace py = pybind11;
//     py::gil_scoped_acquire gil;

//     if (!py_->optimizer_py || py_->optimizer_py.is_none()) {
//         // No python optimizer attached; just grow the C++ member and return.
//         if (out_member_param && out_member_param->defined()) {
//             auto old_param = *out_member_param;
//             auto new_param = torch::cat({old_param, add_rows}, 0).contiguous().detach().requires_grad_(true);
//             *out_member_param = new_param;
//         }
//         return;
//     }

//     // 1) param_groups[i]['params'][0]
//     py::list groups = py_->optimizer_py.attr("param_groups");
//     auto g = groups[group_idx].cast<py::dict>();
//     py::list gparams = g["params"].cast<py::list>();
//     torch::Tensor old_param = gparams[0].cast<torch::Tensor>();

//     // 2) state dict (tensor-keyed)
//     py::dict state = py_->optimizer_py.attr("state").cast<py::dict>();
//     const bool had_state = state.contains(py::cast(old_param));

//     // 3) Build new concatenated param (leaf)
//     torch::Tensor new_param = torch::cat({old_param, add_rows}, 0).contiguous().detach().requires_grad_(true);

//     // 4) Move/extend state if present
//     if (had_state) {
//         py::dict old_s = state[py::cast(old_param)].cast<py::dict>();
//         py::object step_obj = old_s["step"];
//         torch::Tensor exp_avg    = old_s["exp_avg"].cast<torch::Tensor>();
//         torch::Tensor exp_avg_sq = old_s["exp_avg_sq"].cast<torch::Tensor>();
//         torch::Tensor z = torch::zeros_like(add_rows);

//         torch::Tensor exp_avg_new    = torch::cat({exp_avg,    z}, 0).contiguous();
//         torch::Tensor exp_avg_sq_new = torch::cat({exp_avg_sq, z}, 0).contiguous();

//         py::dict new_s;
//         new_s["step"]       = step_obj;
//         new_s["exp_avg"]    = exp_avg_new;
//         new_s["exp_avg_sq"] = exp_avg_sq_new;

//         // Swap state key from old tensor to new tensor
//         state.attr("__delitem__")(py::cast(old_param));
//         state[py::cast(new_param)] = new_s;
//     }

//     // 5) Rebind group param to the same new tensor object
//     gparams[0] = py::cast(new_param);
//     g["params"] = gparams;

//     // 6) Keep C++ member in sync with the *same* tensor object
//     if (out_member_param) *out_member_param = new_param;
// }

void VoxelModel::appendGroup_(int group_idx,
                              const torch::Tensor& add_rows,
                              const char* /*svm_field_name*/,
                              torch::Tensor* out_member_param)
{
  TORCH_CHECK(out_member_param && out_member_param->defined(),
              "appendGroup_: target tensor must be defined.");
  torch::Tensor& member_param = *out_member_param;

  // Make rows match device & dtype
  auto rows = add_rows.to(member_param.options()).contiguous();

  torch::NoGradGuard ng;

  // Build new concatenated leaf
  auto base = member_param.detach().contiguous();
  torch::Tensor new_param = torch::cat({base, rows}, 0).contiguous();
  new_param.set_requires_grad(true);

  // Extend optimizer state (by TensorImpl key), no pointer rebinding!
  if (optimizer_) {
    optimizer_->on_param_concatenated(member_param, rows, new_param);
  }

  // Replace the member tensor (the address of the member object stays stable)
  member_param = std::move(new_param);
}

VoxelModel::StatPkg
VoxelModel::computeTrainingStat(const std::vector<MiniCam>& cams) {
    // Mirrors SVAdaptive.compute_training_stat (but uses our renderer)
    const int64_t N = center_.size(0);
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(device_type_);

    // Reset / init
    this->max_w_.zero_();                               // [N,1] already allocated in model
    auto min_samp_interval = torch::full({N,1}, 1e30f, opts_f);
    auto view_cnt          = torch::zeros({N,1}, opts_f);

    // Renderer bound to this model (so vox_fn sees model fields)
    sv::VoxelRenderer renderer(this);

    // 1) freeze_vox_geo()
    renderer.freezeVoxGeo();

    for (const auto& cam : cams) {
        // std::cout << "rendering cam " << cam.width << "x" << cam.height << "\n";
        // auto pkg = render(cam, torch::Tensor(), cam.height, cam.width, color_mode='dontcare', track_max_w=True);
        // auto pkg = render(
        //     cam, 
        //     cam.height, 
        //     cam.width, 
        //     torch::Tensor(),
        //     "dontcare", 
        //     true);
        sv::RenderOutput pkg = renderer.render(
            cam,
            /*im_height*/ cam.height,
            /*im_width*/  cam.width,
            /*gt_image*/  torch::Tensor(),   // none
            /*color_mode*/ "dontcare",
            /*track_max_w*/ true);
        
        // if (!pkg.count("max_w") || !pkg.at("max_w").defined())
        //     continue;
        if (!pkg.max_w.defined() || pkg.max_w.numel() == 0) {
            continue;
        }

        // auto max_w_i = pkg["max_w"].to(device_type_);
        auto max_w_i = pkg.max_w.to(device_type_);
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

    renderer.unfreezeVoxGeo();
    return { this->max_w_.contiguous(), min_samp_interval.contiguous(), view_cnt.contiguous() };
}

// void VoxelModel::optimizerZeroGrad() {
//     py::gil_scoped_acquire gil;
//     if (!py_->optimizer_py.is_none())
//         py_->optimizer_py.attr("zero_grad")(py::arg("set_to_none") = true);
// }

// void VoxelModel::optimizerStep() {
//     py::gil_scoped_acquire gil;
//     if (!py_->optimizer_py.is_none())
//         py_->optimizer_py.attr("step")();
// }

// void VoxelModel::schedulerStep()
// {
//     py::gil_scoped_acquire gil;
//     if (!py_->scheduler_py.is_none())
//         py_->scheduler_py.attr("step")();
// }

// py::object VoxelModel::schedulerStateDict()
// {
//     py::gil_scoped_acquire gil;
//     if (py_->scheduler_py.is_none()) return py::none();
//     return py_->scheduler_py.attr("state_dict")();
// }

// void VoxelModel::schedulerLoadStateDict(const py::object& state)
// {
//     if (state.is_none()) return;
//     py::gil_scoped_acquire gil;
//     if (!py_->scheduler_py.is_none())
//         py_->scheduler_py.attr("load_state_dict")(state);
// }

void VoxelModel::optimizerZeroGrad() { if (optimizer_) optimizer_->zero_grad(true); }
void VoxelModel::optimizerStep()     { if (optimizer_) optimizer_->step(); }
void VoxelModel::schedulerStep()     { if (scheduler_) scheduler_->step(); } // call once per iter like train.py

sv::optim::MultiStepLRState VoxelModel::schedulerStateDict() const {
  sv::optim::MultiStepLRState st;
  if (scheduler_) st = scheduler_->state_dict();
  else {
    st.last_epoch = -1;
    st.gamma = sched_gamma_;
    st.milestones = sched_milestones_;
  }
  return st;
}

void VoxelModel::schedulerLoadStateDict(const sv::optim::MultiStepLRState& state) {
  if (!scheduler_) {
    // recreate if needed to honor incoming state
    scheduler_ = std::make_unique<sv::optim::MultiStepLR>(optimizer_.get(),
                                                          state.milestones,
                                                          state.gamma);
  }
  scheduler_->load_state_dict(state);
}

// void VoxelModel::pruning(const torch::Tensor& prune_mask) {
//     py::gil_scoped_acquire gil;
//     // accept [N] or [N,1] bool/byte
//     auto mask = prune_mask.to(torch::kBool).to(device_type_).contiguous();
//     py_->svm.attr("pruning")(mask);
//     syncFromPython_();
// }

// void VoxelModel::subdividing(const torch::Tensor& subdivide_mask) {
//     py::gil_scoped_acquire gil;
//     auto mask = subdivide_mask.to(torch::kBool).to(device_type_).contiguous();
//     py_->svm.attr("subdividing")(mask);
//     syncFromPython_();
// }

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

// mask / cat / perm in one place
inline at::Tensor mask_cat_perm(const at::Tensor& tensor,
                                const c10::optional<at::Tensor>& kept_idx = c10::nullopt,
                                const c10::optional<at::Tensor>& cat_tensor = c10::nullopt,
                                const c10::optional<at::Tensor>& perm = c10::nullopt) {
    TORCH_CHECK(tensor.defined(), "mask_cat_perm: input undefined");
    at::Tensor out = tensor;
    if (kept_idx.has_value()) {
        out = out.index_select(0, kept_idx.value().to(out.device()));
    }
    if (cat_tensor.has_value()) {
        TORCH_CHECK(cat_tensor.value().defined(), "mask_cat_perm: cat_tensor undefined");
        out = at::cat({out, cat_tensor.value().to(out.device())}, /*dim=*/0);
    }
    if (perm.has_value()) {
        TORCH_CHECK(perm.value().numel() == out.size(0), "mask_cat_perm: perm size mismatch");
        out = out.index_select(0, perm.value().to(out.device()));
    }
    return out.contiguous();
}

// Aggregate per-voxel corner values [N,8,*] into grid-points [M,*] via mean reduce.
// vox_key: [N,8] (grid-point indices)
inline at::Tensor agg_voxel_into_grid_pts(
    int64_t num_grid_pts,
    const at::Tensor& vox_key,     // [N,8] int64
    const at::Tensor& vox_val)     // [N,8,*] float
{
    TORCH_CHECK(vox_key.dtype() == at::kLong, "vox_key must be int64");
    TORCH_CHECK(vox_key.dim()==2 && vox_key.size(1)==8, "vox_key [N,8]");
    TORCH_CHECK(vox_val.dim()>=2 && vox_val.size(1)==8, "vox_val [N,8,*]");

    auto dev = vox_val.device();
    const auto N = vox_key.size(0);

    // Flatten voxel corners
    auto idx_flat = vox_key.reshape({N*8});                         // [N*8]
    auto val_flat = vox_val.reshape({N*8, -1});                     // [N*8, C]

    // Sums per grid point
    auto sums = at::zeros({num_grid_pts, val_flat.size(1)}, val_flat.options());
    sums.index_add_(0, idx_flat, val_flat);

    // Counts per grid point
    auto ones = at::ones({N*8}, val_flat.options().dtype(at::kFloat));
    auto counts = at::zeros({num_grid_pts}, val_flat.options().dtype(at::kFloat));
    counts.index_add_(0, idx_flat, ones);

    // Avoid div-by-zero; where count==0 keep 0
    counts = counts.clamp_min(1.0f).unsqueeze(1); // [M,1]
    auto mean = sums / counts;                    // [M,C]

    // reshape back to [M,*]
    std::vector<int64_t> shape;
    shape.push_back(num_grid_pts);
    for (int d=2; d<vox_val.dim(); ++d) shape.push_back(vox_val.size(d));
    return mean.reshape(shape).contiguous();
}

// Trilinear “child” interpolation for voxel corner data, mirroring Python version.
// Input:  vox_val [N,8,*]  -> Output: [8N,8,*]
inline at::Tensor subdivide_by_interp(const at::Tensor& vox_val) {
    TORCH_CHECK(vox_val.dim()>=2 && vox_val.size(1)==8, "subdivide_by_interp: vox_val [N,8,*]");
    auto dev = vox_val.device();
    const auto N = vox_val.size(0);

    // new_vox_val: [N,8,8,*]
    std::vector<int64_t> new_shape = {N, 8, vox_val.size(1)};
    for (int d=2; d<vox_val.dim(); ++d) new_shape.push_back(vox_val.size(d));
    at::Tensor new_vox_val = at::zeros(new_shape, vox_val.options());

    // main_idx 0..7; XOR masks as in Python
    auto main_idx = at::arange(8, at::TensorOptions().dtype(at::kLong).device(dev));
    // handy views
    auto V = vox_val; // [N,8,*]

    // helper lambda to set child corner values
    auto set_corner = [&](int dst_off, int src_off, double w_self, int xor1=-1, double w1=0, int xor2=-1, double w2=0, int xor3=-1, double w3=0) {
        // new_vox_val[:, main_idx, main_idx^dst_mask] = combination
        auto dst_corner = main_idx ^ dst_off;
        auto src0 = V.index({at::indexing::Slice(), main_idx});
        at::Tensor out = w_self * src0;
        if (xor1>=0) out = out + w1 * V.index({at::indexing::Slice(), main_idx ^ xor1});
        if (xor2>=0) out = out + w2 * V.index({at::indexing::Slice(), main_idx ^ xor2});
        if (xor3>=0) out = out + w3 * V.index({at::indexing::Slice(), main_idx ^ xor3});
        new_vox_val.index_put_({at::indexing::Slice(), main_idx, dst_corner}, out);
    };

    // Direct copies
    new_vox_val.index_put_({at::indexing::Slice(), main_idx, main_idx}, V.index({at::indexing::Slice(), main_idx}));

    // Edges (0.5 blends)
    set_corner(0b001, 0, 0.5, 0b001, 0.5);
    set_corner(0b010, 0, 0.5, 0b010, 0.5);
    set_corner(0b100, 0, 0.5, 0b100, 0.5);

    // Faces (0.25 blends)
    set_corner(0b011, 0, 0.25, 0b001, 0.25, 0b010, 0.25, 0b011, 0.25);
    set_corner(0b101, 0, 0.25, 0b001, 0.25, 0b100, 0.25, 0b101, 0.25);
    set_corner(0b110, 0, 0.25, 0b010, 0.25, 0b100, 0.25, 0b110, 0.25);

    // Center (mean of 8 corners)
    auto mean8 = V.mean(/*dim=*/1, /*keepdim=*/true); // [N,1,*]
    new_vox_val.index_put_({at::indexing::Slice(), main_idx, main_idx ^ 0b111}, mean8);

    // reshape to [8N,8,*]
    std::vector<int64_t> final_shape = {N*8, 8};
    for (int d=2; d<vox_val.dim(); ++d) final_shape.push_back(vox_val.size(d));
    return new_vox_val.reshape(final_shape).contiguous();
}

void VoxelModel::pruning(const torch::Tensor& prune_mask_in) {
    at::NoGradGuard ng;
    // accept [N] or [N,1]; to bool+device
    auto mask = prune_mask_in.to(torch::kBool).to(device_type_).contiguous();
    if (mask.dim()==2 && mask.size(1)==1) mask = mask.squeeze(1);
    TORCH_CHECK(mask.dim()==1 && mask.size(0)==center_.size(0), "pruning: mask shape mismatch");

    // indices to keep
    auto kept_idx = (~mask).nonzero().squeeze(1);
    if (kept_idx.numel()==0) return;

    // save old vox_key to gather old grid-pt values
    auto old_vox_key = vox_key_.clone(); // [N,8]

    // ---- Per-voxel ATTRIBUTES (non-trainable): octpath/octlevel (centers/sizes will be recomputed)
    oct_path_  = mask_cat_perm(oct_path_,  kept_idx);
    oct_level_ = mask_cat_perm(oct_level_, kept_idx);

    // recompute centers/sizes from octree
    {
        auto pair = sv::oct::octpath_decoding(oct_path_, oct_level_, scene_center_, scene_extent_);
        center_ = pair.first.contiguous();
        size_   = pair.second.squeeze(1).contiguous(); // [N]
        vox_size_inv_ = 1.0f / size_;
    }

    // rebuild grid links from new octree
    {
        auto pair = sv::oct::build_grid_pts_link(oct_path_, oct_level_);
        grid_pts_key_ = pair.first.contiguous(); // [M,3]
        vox_key_      = pair.second.contiguous(); // [N,8]
    }

    // ---- Per-voxel PARAMS (trainable): sh0_, shs_
    if (sh0_.defined()) {
        auto ori = sh0_.detach();
        sh0_ = mask_cat_perm(ori, kept_idx).contiguous().detach().requires_grad_(true);
    }
    if (shs_.defined()) {
        auto ori = shs_.detach();
        shs_ = mask_cat_perm(ori, kept_idx).contiguous().detach().requires_grad_(true);
    }

    // ---- Subdivision priority (keep and preserve grad if present)
    if (subdiv_p_.defined()) {
        auto ori = subdiv_p_;
        auto new_subdiv = mask_cat_perm(ori, kept_idx);
        if (ori.grad().defined()) {
            auto new_grad = mask_cat_perm(ori.grad(), kept_idx);
            new_subdiv = new_subdiv.contiguous().detach().set_requires_grad(true);
            new_subdiv.mutable_grad() = new_grad.contiguous();
        } else {
            new_subdiv = new_subdiv.contiguous().detach().set_requires_grad(true);
        }
        subdiv_p_ = new_subdiv;
    }

    // ---- Grid-point PARAMS (trainable on corners): _geo_grid_pts_
    if (_geo_grid_pts_.defined()) {
        // gather old per-voxel-corner values: [N,8,*]
        auto ori_grid = _geo_grid_pts_.detach();
        auto ori_vox_grid_pts_val = ori_grid.index({old_vox_key}); // [N,8,*]
        // keep only survivors
        auto new_vox_val = ori_vox_grid_pts_val.index_select(0, kept_idx);
        // aggregate into new grid-points
        const int64_t M = grid_pts_key_.size(0);
        auto new_param = agg_voxel_into_grid_pts(M, vox_key_, new_vox_val)
                            .contiguous().detach().requires_grad_(true);
        _geo_grid_pts_ = new_param;
    }

    // resize side buffers
    max_w_ = torch::zeros({center_.size(0), 1},
              torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

    VOXEL_MODEL_TENSORS_TO_VEC
}

void VoxelModel::subdividing(const torch::Tensor& subdivide_mask_in) {
    at::NoGradGuard ng;
    auto mask = subdivide_mask_in.to(torch::kBool).to(device_type_).contiguous();
    if (mask.dim()==2 && mask.size(1)==1) mask = mask.squeeze(1);
    TORCH_CHECK(mask.dim()==1 && mask.size(0)==center_.size(0), "subdividing: mask shape mismatch");

    auto kept_idx      = (~mask).nonzero().squeeze(1); // survivors
    auto subdivide_idx = mask.nonzero().squeeze(1);    // to split
    if (subdivide_idx.numel()==0) return;

    auto old_vox_key = vox_key_.clone(); // [N,8]

    // ---- Per-voxel ATTRIBUTES with SPECIAL subdivision for octree fields
    // children octpath/level for subdivided subset
    auto child_pair = sv::oct::gen_children(
        oct_path_.index_select(0, subdivide_idx),
        oct_level_.index_select(0, subdivide_idx)
    );
    auto child_path  = child_pair.first;  // [8K,1] int64
    auto child_level = child_pair.second; // [8K,1] int8

    // oct_path_, oct_level_ := keep + children
    oct_path_  = mask_cat_perm(oct_path_,  kept_idx, child_path);
    oct_level_ = mask_cat_perm(oct_level_, kept_idx, child_level);

    // recompute centers/sizes from new octree
    {
        auto pair = sv::oct::octpath_decoding(oct_path_, oct_level_, scene_center_, scene_extent_);
        center_ = pair.first.contiguous();
        size_   = pair.second.squeeze(1).contiguous();
        vox_size_inv_ = 1.0f / size_;
    }

    // rebuild grid links from new octree
    {
        auto pair = sv::oct::build_grid_pts_link(oct_path_, oct_level_);
        grid_pts_key_ = pair.first.contiguous();
        vox_key_      = pair.second.contiguous();
    }

    // ---- Per-voxel PARAMS: duplicate subdivided entries 8x, then keep+cat
    if (sh0_.defined()) {
        auto ori = sh0_.detach();
        auto dup = ori.index_select(0, subdivide_idx).repeat_interleave(8, /*dim=*/0);
        sh0_ = mask_cat_perm(ori, kept_idx, dup).contiguous().detach().requires_grad_(true);
    }
    if (shs_.defined()) {
        auto ori = shs_.detach();
        auto dup = ori.index_select(0, subdivide_idx).repeat_interleave(8, /*dim=*/0);
        shs_ = mask_cat_perm(ori, kept_idx, dup).contiguous().detach().requires_grad_(true);
    }

    // ---- Subdivision priority (preserve grad)
    if (subdiv_p_.defined()) {
        auto ori = subdiv_p_;
        auto dup = ori.index_select(0, subdivide_idx).repeat_interleave(8, /*dim=*/0);
        auto new_subdiv = mask_cat_perm(ori, kept_idx, dup);
        if (ori.grad().defined()) {
            auto gdup = ori.grad().index_select(0, subdivide_idx).repeat_interleave(8, 0);
            auto new_grad = mask_cat_perm(ori.grad(), kept_idx, gdup);
            new_subdiv = new_subdiv.contiguous().detach().set_requires_grad(true);
            new_subdiv.mutable_grad() = new_grad.contiguous();
        } else {
            new_subdiv = new_subdiv.contiguous().detach().set_requires_grad(true);
        }
        subdiv_p_ = new_subdiv;
    }

    // ---- Grid-point PARAMS: gather→interpolate children→keep+cat→aggregate
    if (_geo_grid_pts_.defined()) {
        auto ori_grid = _geo_grid_pts_.detach();
        auto ori_vox_grid_pts_val = ori_grid.index({old_vox_key});          // [N,8,*]
        auto sub_vox_vals = subdivide_by_interp(
            ori_vox_grid_pts_val.index_select(0, subdivide_idx));            // [8K,8,*]
        auto new_vox_val = mask_cat_perm(
            ori_vox_grid_pts_val, kept_idx, sub_vox_vals);                   // [(N-K)+8K,8,*]
        const int64_t M = grid_pts_key_.size(0);
        auto new_param = agg_voxel_into_grid_pts(M, vox_key_, new_vox_val)
                            .contiguous().detach().requires_grad_(true);
        _geo_grid_pts_ = new_param;
    }

    // resize side buffers
    max_w_ = torch::zeros({center_.size(0), 1},
              torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

    VOXEL_MODEL_TENSORS_TO_VEC
}

torch::Tensor VoxelModel::subdivisionPriority() const {
    auto g = this->subdiv_p_.grad();
    if (g.defined() && g.dim() == 2 && g.size(1) == 1) g = g.squeeze(1);
    return g;
}

void VoxelModel::resetSubdivisionPriority() {
    this->subdiv_p_.mutable_grad() = torch::Tensor();
}

// void VoxelModel::freezeVoxGeo() {
//     py::gil_scoped_acquire gil;

//     const int64_t N = center_.size(0);
//     auto care_idx = torch::arange(N, torch::dtype(torch::kLong).device(device_type_));

//     auto torch_mod = py::module_::import("torch");
//     auto no_grad   = torch_mod.attr("no_grad")();
//     no_grad.attr("__enter__")();
//     try {
//         auto renderer = py::module_::import("svraster_cuda.renderer");
//         auto Gather   = renderer.attr("GatherGeoParams");
//         // returns a torch.Tensor
//         py::object py_frozen = Gather.attr("apply")(vox_key_, care_idx, _geo_grid_pts_);
//         frozen_vox_geo_ = py_frozen.cast<torch::Tensor>().contiguous();
//         _geo_grid_pts_.set_requires_grad(false);
//         no_grad.attr("__exit__")(py::none(), py::none(), py::none());
//     } catch (...) {
//         no_grad.attr("__exit__")(py::none(), py::none(), py::none());
//         throw;
//     }
// }

// void VoxelModel::freezeVoxGeo() {
//     // Freeze grid-point params and pre-gather them per voxel (no Python).
//     TORCH_CHECK(_geo_grid_pts_.defined(), "freezeVoxGeo: _geo_grid_pts_ is undefined");
//     TORCH_CHECK(vox_key_.defined() && vox_key_.dim()==2 && vox_key_.size(1)==8,
//                 "freezeVoxGeo: vox_key_ must be [N,8]");

//     torch::NoGradGuard guard;

//     // Gather per-voxel 8-corner values: [N,8,*]
//     // (Equivalent to svraster_cuda.renderer.GatherGeoParams.apply(...))
//     frozen_vox_geo_ = _geo_grid_pts_.index({vox_key_}).contiguous();

//     // Stop training grid-point params while frozen
//     _geo_grid_pts_.set_requires_grad(false);
// }

// void VoxelModel::unfreezeVoxGeo() {
//     // py::gil_scoped_acquire gil;
//     frozen_vox_geo_.reset();              // make it undefined
//     _geo_grid_pts_.set_requires_grad(true);
// }

// static inline py::dict make_kwargs_from(
//     const sv::RenderOpts& a,
//     // explicit function params (after color_mode)
//     bool track_max_w,
//     std::optional<float> ss,
//     bool output_depth,
//     bool output_normal,
//     bool output_T,
//     bool rand_bg,
//     bool use_auto_exposure)
// {
//     py::dict kw;

//     // 1) Start with struct — these become **other_opt
//     if (a.lambda_dist.has_value())       kw["lambda_dist"]      = *a.lambda_dist;
//     if (a.lambda_ascending.has_value())  kw["lambda_ascending"] = *a.lambda_ascending;
//     if (a.lambda_R_concen.has_value())  kw["lambda_R_concen"] = *a.lambda_R_concen;
//     if (a.gt_color.defined())            kw["gt_color"]         = a.gt_color;
//     // If you have other non-overlapping keys (e.g., n_samp_per_vox), add them here as well:
//     // if (a.n_samp_per_vox.has_value()) kw["n_samp_per_vox"] = *a.n_samp_per_vox;

//     // Overlapping names: add **only** to kwargs (we won’t pass them positionally)
//     // struct values (if set) first…
//     if (a.track_max_w)        kw["track_max_w"]       = true;
//     if (a.ss.has_value())     kw["ss"]                = *a.ss;
//     if (a.output_depth)       kw["output_depth"]      = true;
//     if (a.output_normal)      kw["output_normal"]     = true;
//     if (a.output_T)           kw["output_T"]          = true;
//     if (a.rand_bg)            kw["rand_bg"]           = true;
//     if (a.use_auto_exposure)  kw["use_auto_exposure"] = true;

//     // …then override with explicit function arguments if caller set them.
//     if (track_max_w)          kw["track_max_w"]       = true;
//     if (ss.has_value())       kw["ss"]                = *ss;
//     if (output_depth)         kw["output_depth"]      = true;
//     if (output_normal)        kw["output_normal"]     = true;
//     if (output_T)             kw["output_T"]          = true;
//     if (rand_bg)              kw["rand_bg"]           = true;
//     if (use_auto_exposure)    kw["use_auto_exposure"] = true;

//     return kw;
// }

// // std::unordered_map<std::string, torch::Tensor> VoxelModel::render(const MiniCam& cam, torch::Tensor gt_image, int image_height, int image_width, float ss, bool track_max_w) const
// std::unordered_map<std::string, torch::Tensor> VoxelModel::render(
//     const sv::MiniCam& cam,
//     int im_height,
//     int im_width,
//     const torch::Tensor& gt_image,
//     const char* color_mode,
//     bool track_max_w,
//     std::optional<float> ss,
//     bool output_depth,
//     bool output_normal,
//     bool output_T,
//     bool rand_bg,
//     bool use_auto_exposure,
//     const sv::RenderOpts& other_opt) const
// {
//     /* 1) import the python entry-point once                                 */
//     py::gil_scoped_acquire gil;
//     static py::object py_render;
//     if (!py_render) {
//         try {
//             py_render = py::module_::import(
//                             "scripts_voxel.python_svraster_bridge.renderer_wrapper")
//                             .attr("render");
//             std::cerr << "[INFO] Python-side renderer imported OK.\n";
//         } catch (const py::error_already_set& e) {
//             std::cerr << "[PYBIND11] Could not import renderer_wrapper:\n"
//                       << e.what() << std::endl;
//             return {};
//         }
//     }

//     py::dict d;
//     d["_geo_grid_pts"]      = this->_geo_grid_pts_;
//     d["sh0"]               = this->sh0_;
//     d["shs"]               = this->shs_;
//     d["subdiv_p"]          = this->subdiv_p_;
//     d["octpath"]           = this->oct_path_;
//     d["octlevel"]          = this->oct_level_;
//     d["center"]            = this->center_;
//     d["vox_size"]          = this->size_;
//     d["vox_key"]           = this->vox_key_;
//     d["active_sh_degree"]  = py::int_(this->active_sh_degree_);

//     d["white_background"] = py::bool_(this->white_background_);
//     d["black_background"] = py::bool_(this->black_background_);
//     d["ss"]               = py::float_(this->ss_);       // default SS (used if ss=None)
//     d["n_samp_per_vox"]   = py::int_(this->n_samp_per_vox_);
//     d["num_voxels"]        = static_cast<long long>(this->center_.size(0));

//     // Optional: expose pre-gathered geo if present
//     if (this->frozen_vox_geo_.defined() && this->frozen_vox_geo_.numel() > 0) {
//         d["frozen_vox_geo"] = this->frozen_vox_geo_;
//     }

//     // Build kwargs (**other_opt). IMPORTANT: do NOT pass overlapping args positionally.
//     py::dict kwargs = make_kwargs_from(
//         other_opt, track_max_w, ss, output_depth, output_normal,
//         output_T, rand_bg, use_auto_exposure);
//     // std::cout << "kwargs" << kwargs << std::endl;   
//     /* 4) call python                                                        */
//     py::object py_cam  = MiniCam_to_py(cam);
//     py::object py_out;
//     py::tuple args = py::make_tuple(
//         d,
//         py_cam,
//         im_height,
//         im_width,
//         gt_image,
//         color_mode
//     ); 
//     try {
//         py_out = py_render(*args, **kwargs);
//         // py_out = py_render(
//         //     d,
//         //     py_cam,
//         //     im_height,
//         //     im_width,
//         //     gt_image,
//         //     color_mode,
//         //     py::kwargs(kwargs) );
//         // py_out = py_render(py_cam, d, gt_image, image_height, image_width, ss, track_max_w);
//     } catch (const py::error_already_set& e) {
//         std::cerr << "[PYBIND11] Exception thrown by renderer:\n"
//                   << e.what() << std::endl;
//         return {};
//     }
//     py::dict out_dict = py_out.cast<py::dict>();

//     /* 6) copy every tensor into a C++ map                                   */
//     std::unordered_map<std::string, torch::Tensor> pkg;
//     for (auto item : out_dict) {
//         const std::string key = py::str(item.first);
//         torch::Tensor t;
//         bool is_tensor = true;
//         try {
//             t = item.second.cast<torch::Tensor>();
//         } catch (...) {
//             is_tensor = false;
//         }
//         if (t.defined())
//             pkg.emplace(key, std::move(t));
//     }   
//     return pkg;                       // may be empty on error
// }

// void VoxelModel::applyTvOnDensityField(float lambda_tv_density) {
//     // Must NOT be inside a NoGrad guard — we want to add to _geo_grid_pts.grad
//     py::gil_scoped_acquire gil;
//     try {
//         // Uses SVProperties.apply_tv_on_density_field(self, weight)
//         // which sets up grad if needed and calls svraster_cuda.grid_loss_bw.total_variation
//         py_->svm.attr("apply_tv_on_density_field")(lambda_tv_density);
//     } catch (const py::error_already_set& e) {
//         std::cerr << "[TV] apply_tv_on_density_field failed: " << e.what() << std::endl;
//     }
// }

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

// float VoxelModel::paramL2(const char* name) {
//     py::gil_scoped_acquire gil;
//     auto t = py_->svm.attr(name).cast<torch::Tensor>();
//     return t.defined()? t.norm().item<float>() : 0.f;
// }

// float VoxelModel::gradL2(const char* name) {
//     py::gil_scoped_acquire gil;
//     auto p = py_->svm.attr(name).cast<torch::Tensor>();
//     auto g = p.grad();
//     return (g.defined()? g.norm().item<float>() : 0.f);
// }

// void VoxelModel::debugParamChain() {
//     py::gil_scoped_acquire gil;

//     auto p_geo = py_->svm.attr("_geo_grid_pts").cast<torch::Tensor>();
//     auto p_sh0 = py_->svm.attr("_sh0").cast<torch::Tensor>();
//     auto p_shs = py_->svm.attr("_shs").cast<torch::Tensor>();

//     auto g_geo = p_geo.grad();
//     auto g_sh0 = p_sh0.grad();
//     auto g_shs = p_shs.grad();

//     auto ptr = [](const torch::Tensor& t)->uintptr_t { return (uintptr_t)t.data_ptr(); };
//     auto nrm = [](const torch::Tensor& t)->double { return (t.defined() ? t.norm().item<double>() : -1.0); };

//     std::cout << std::fixed << std::setprecision(6)
//               << "[autograd] _geo  req=" << p_geo.requires_grad()
//               << "  p="  << ptr(p_geo)
//               << "  ||p||="   << nrm(p_geo)
//               << "  grad? "   << (g_geo.defined())
//               << "  ||g||="   << nrm(g_geo) << "\n"
//               << "[autograd] _sh0  req=" << p_sh0.requires_grad()
//               << "  p="  << ptr(p_sh0)
//               << "  ||p||="   << nrm(p_sh0)
//               << "  grad? "   << (g_sh0.defined())
//               << "  ||g||="   << nrm(g_sh0) << "\n"
//               << "[autograd] _shs  req=" << p_shs.requires_grad()
//               << "  p="  << ptr(p_shs)
//               << "  ||p||="   << nrm(p_shs)
//               << "  grad? "   << (g_shs.defined())
//               << "  ||g||="   << nrm(g_shs) << std::endl;
// }

// void VoxelModel::debugOptimizer() {
//     py::gil_scoped_acquire gil;
//     if (py_->optimizer_py.is_none()) {
//         std::cout << "[optim] None\n"; return;
//     }
//     py::list groups = py_->optimizer_py.attr("param_groups");
//     for (ssize_t i = 0; i < (ssize_t)groups.size(); ++i) {
//         auto g = groups[i].cast<py::dict>();
//         double lr = py::float_(g["lr"]);
//         py::tuple params = g["params"].cast<py::tuple>();
//         std::cout << "[optim] group " << i << " lr=" << lr
//                   << " nparams=" << params.size() << "\n";
//         for (ssize_t j = 0; j < (ssize_t)params.size(); ++j) {
//             auto t = py::reinterpret_borrow<py::object>(params[j]).cast<torch::Tensor>();
//             auto tg = t.grad();
//             std::cout << "   - shape=" << t.sizes()
//                       << " req=" << t.requires_grad()
//                       << " ||p||=" << (t.defined()? t.norm().item<double>(): -1.0)
//                       << " grad? " << tg.defined()
//                       << " ||g||=" << (tg.defined()? tg.norm().item<double>(): -1.0)
//                       << "\n";
//         }
//     }
// }

// torch::Tensor VoxelModel::snapParam(const char* name) {
//     py::gil_scoped_acquire gil;
//     auto t = py_->svm.attr(name).cast<torch::Tensor>();
//     return t.detach().clone();               // CPU clone if you prefer
// }

// double VoxelModel::deltaFrom(const char* name, const torch::Tensor& prev) {
//     py::gil_scoped_acquire gil;
//     auto t = py_->svm.attr(name).cast<torch::Tensor>();
//     auto d = (t - prev).norm().item<double>();
//     auto n = t.norm().item<double>();
//     return (n > 0.0 ? d / n : 0.0);
// }

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

} // namespace sv

