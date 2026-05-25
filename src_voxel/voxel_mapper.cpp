#include "include_voxel/voxel_mapper.h"
#include "include/stereo_vision.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <Eigen/Eigenvalues>
#include <numeric>
#include <opencv2/flann.hpp>
#include <queue>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "ORB-SLAM3/include/Atlas.h"
#include "ORB-SLAM3/include/MapPoint.h"

namespace py = pybind11;
std::ofstream loss_log_;
std::ofstream loss_l1_log_;
std::ofstream loss_ssim_log_;
std::ofstream loss_l2_log_;

namespace {
constexpr int kRenderedCandidateSourceDepthInsert = 1;
constexpr int kRenderedCandidateSourceMonoPrior = 3;
constexpr float kMonoPriorFillHolesEmptyDepthEps = 1e-6f;
constexpr float kMonoPriorAlignInitialInlierRelErr = 0.50f;
constexpr float kMonoPriorAlignFinalInlierRelErr = 0.30f;

struct MonoPriorAlignmentStats {
    int64_t num_valid_anchors = 0;
    int64_t num_fit_anchors = 0;
    float sparse_depth_q05 = std::numeric_limits<float>::quiet_NaN();
    float sparse_depth_q95 = std::numeric_limits<float>::quiet_NaN();
    float scale = std::numeric_limits<float>::quiet_NaN();
    float shift = std::numeric_limits<float>::quiet_NaN();
    float median_rel_depth_error = std::numeric_limits<float>::quiet_NaN();
    float p90_rel_depth_error = std::numeric_limits<float>::quiet_NaN();
    float final_inlier_ratio = std::numeric_limits<float>::quiet_NaN();
};

std::string toLowerCopy(std::string s) {
    std::transform(
        s.begin(),
        s.end(),
        s.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool isMetric3DModelId(const std::string& model_id) {
    const std::string model = toLowerCopy(model_id);
    return model.find("metric3d") != std::string::npos ||
           (model.find("depth-anything") != std::string::npos &&
            model.find("metric") != std::string::npos);
}

std::string monoPriorDebugStemFromModelId(const std::string& model_id) {
    (void)model_id;
    return "mono_prior";
}

const char* monoPriorLogPrefix(const std::string& model_id) {
    return isMetric3DModelId(model_id) ? "[MetricDepth]" : "[DepthAnythingV2]";
}

std::string normalizeMonoPriorLossMode(std::string mode) {
    mode = toLowerCopy(std::move(mode));
    if (mode == "orb_aligned" || mode == "orb-aligned" || mode == "slam" ||
        mode == "systems") {
        return "aligned";
    }
    if (mode == "sv" || mode == "svraster") {
        return "svraster";
    }
    if (mode == "geo" || mode == "geosvr") {
        return "geosvr";
    }
    return mode;
}

std::string normalizeMonoPriorNormalMode(std::string mode) {
    mode = toLowerCopy(std::move(mode));
    if (mode == "orb" || mode == "orb_aligned" || mode == "orb-aligned" ||
        mode == "anchors" || mode == "keyframe" || mode == "aligned") {
        return "aligned";
    }
    if (mode == "geo" || mode == "geosvr") {
        return "geosvr";
    }
    return mode;
}

std::string normalizeMonoPriorDensifyAlignmentMode(std::string mode) {
    mode = toLowerCopy(std::move(mode));
    if (mode == "render" || mode == "rendered_depth" || mode == "svraster") {
        return "rendered";
    }
    if (mode == "orb_aligned" || mode == "orb-aligned" || mode == "anchors" ||
        mode == "keyframe") {
        return "orb";
    }
    if (mode == "da_prior" || mode == "prior_da" ||
        mode == "prior-depth-anything" || mode == "prior_depth_anything" ||
        mode == "priorda") {
        return "da_prior";
    }
    return mode;
}

int sampleGeoSvrPatchSize()
{
    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<int> dist(17, 31);
    return dist(rng);
}

torch::Tensor squeezeRenderMap2D(torch::Tensor t) {
    if (!t.defined()) {
        return t;
    }
    if (t.dim() == 4 && t.size(0) == 1) t = t.squeeze(0);
    if (t.dim() == 3 && t.size(0) >= 1) t = t.index({0});
    return t.contiguous();
}

bool renderPkgToMonoPriorAlignmentMaps(
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    int expected_h,
    int expected_w,
    torch::Tensor& depth_cpu,
    torch::Tensor& alpha_cpu,
    torch::Tensor& n_contrib_cpu)
{
    depth_cpu = torch::Tensor();
    alpha_cpu = torch::Tensor();
    n_contrib_cpu = torch::Tensor();

    auto it_depth = render_pkg.find("raw_depth");
    if (it_depth == render_pkg.end()) it_depth = render_pkg.find("depth");
    auto it_T = render_pkg.find("raw_T");
    if (it_T == render_pkg.end()) it_T = render_pkg.find("T");
    auto it_n_contrib = render_pkg.find("n_contrib");
    if (it_n_contrib == render_pkg.end()) it_n_contrib = render_pkg.find("raw_n_contrib");
    if (it_depth == render_pkg.end() || it_T == render_pkg.end() ||
        it_n_contrib == render_pkg.end() ||
        !it_depth->second.defined() || !it_T->second.defined() ||
        !it_n_contrib->second.defined()) {
        return false;
    }

    torch::Tensor raw_depth = it_depth->second;
    if (raw_depth.dim() == 4 && raw_depth.size(0) == 1) raw_depth = raw_depth.squeeze(0);
    if (raw_depth.dim() != 3 || raw_depth.size(0) < 1) {
        return false;
    }

    torch::Tensor render_depth_raw =
        raw_depth.index({0}).detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor render_T =
        squeezeRenderMap2D(it_T->second).detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor render_n_contrib =
        squeezeRenderMap2D(it_n_contrib->second).detach().to(torch::kCPU).to(torch::kInt32).contiguous();
    if (render_depth_raw.dim() != 2 || render_T.dim() != 2 || render_n_contrib.dim() != 2) {
        return false;
    }
    if (render_depth_raw.size(0) != expected_h || render_depth_raw.size(1) != expected_w ||
        render_T.size(0) != expected_h || render_T.size(1) != expected_w ||
        render_n_contrib.size(0) != expected_h || render_n_contrib.size(1) != expected_w) {
        return false;
    }

    alpha_cpu = (1.0f - render_T).contiguous();
    depth_cpu = (render_depth_raw / alpha_cpu.clamp_min(1e-6f)).contiguous();
    n_contrib_cpu = render_n_contrib.contiguous();
    return true;
}

}

void saveTensor(const torch::Tensor &t,
                const std::string &tag,
                const std::string &dbg_dir,
                int iter,
                int image_id)
{
    auto img = t.squeeze(0)
                 .permute({1,2,0})
               // optional gamma:
               // .clamp(0.0f, 1.0f).pow(1.0f/2.2f)
                 .mul(255.0f).clamp(0.0f,255.0f)
                 .to(torch::kUInt8)
                 .contiguous()
                 .cpu();
    int H = img.size(0), W = img.size(1);
    cv::Mat rgb(H, W, CV_8UC3, img.data_ptr());
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    std::ostringstream ss;
    ss << dbg_dir << "/" 
       << tag 
       << "_iter" << std::setw(6) << std::setfill('0') << iter
       << "_img"  << std::setw(3) << std::setfill('0') << image_id
       << ".png";
    cv::imwrite(ss.str(), bgr);
}

inline void saveDebugImage(torch::Tensor tensor, const std::string& path) {
    torch::NoGradGuard ng;
    namespace fs = std::filesystem;

    const fs::path out_path(path);
    if (!out_path.parent_path().empty())
        fs::create_directories(out_path.parent_path());

    // Normalize to (C,H,W)
    tensor = tensor.detach();
    if (tensor.dim() == 4 && tensor.size(0) == 1) tensor = tensor.squeeze(0); // (1,3,H,W)->(3,H,W)
    if (tensor.dim() == 2)                        tensor = tensor.unsqueeze(0); // (H,W)->(1,H,W)
    if (tensor.dim() == 3 && tensor.size(0) == 1) tensor = tensor.expand({3, tensor.size(1), tensor.size(2)});
    if (tensor.dim() != 3 || tensor.size(0) != 3) {
        std::cerr << "[saveDebugImage_fast] bad shape " << tensor.sizes() << "\n";
        return;
    }

    // GPU-side quantize to uint8 to reduce D2H bandwidth 4x
    if (tensor.device().is_cuda()) {
        if (tensor.dtype() != torch::kUInt8)
            tensor = tensor.clamp(0, 1).mul(255).to(torch::kUInt8); // still on GPU

        // Copy into pinned host memory
        auto cpu_u8 = torch::empty_like(
            tensor, tensor.options().device(torch::kCPU).pinned_memory(true));

        // For correctness keep this blocking; if you offload imwrite to a thread,
        // switch to non_blocking=true and synchronize appropriately there.
        cpu_u8.copy_(tensor, /*non_blocking=*/false);
        tensor = cpu_u8;
    } else {
        if (tensor.dtype() != torch::kUInt8)
            tensor = tensor.clamp(0, 1).mul(255).to(torch::kUInt8);
    }

    // HWC for OpenCV
    tensor = tensor.permute({1, 2, 0}).contiguous();
    cv::Mat rgb(tensor.size(0), tensor.size(1), CV_8UC3, tensor.data_ptr<uint8_t>());

    // Convert to BGR (skip this if your tensors are already BGR)
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);

    // Encoder params
    std::vector<int> params;
    const std::string ext = out_path.extension().string();
    if (ext == ".png" || ext == ".PNG") {
        params = {cv::IMWRITE_PNG_COMPRESSION, 1};
    } else if (ext == ".jpg" || ext == ".jpeg" || ext == ".JPG" || ext == ".JPEG") {
        params = {cv::IMWRITE_JPEG_QUALITY, 90};
    }

    if (!cv::imwrite(out_path.string(), bgr, params)) {
        std::cerr << "[saveDebugImage_fast] imwrite failed: " << out_path << "\n";
    }
}

static inline void extendAABB(Eigen::Vector3f& mn, Eigen::Vector3f& mx,
                              const Eigen::Vector3f& p) {
    mn = mn.cwiseMin(p);
    mx = mx.cwiseMax(p);
}

static inline void extendAABB_with_flat_xyz(Eigen::Vector3f& mn, Eigen::Vector3f& mx,
                                            const std::vector<float>& flat_xyz) {
    const size_t n = flat_xyz.size();
    if (n < 3) return;
    // If mn is not initialized yet, seed from the first triplet
    if (!std::isfinite(mn.x())) {
        mn = Eigen::Vector3f(flat_xyz[0], flat_xyz[1], flat_xyz[2]);
        mx = mn;
    }
    for (size_t i = 0; i + 2 < n; i += 3) {
        Eigen::Vector3f p(flat_xyz[i+0], flat_xyz[i+1], flat_xyz[i+2]);
        extendAABB(mn, mx, p);
    }
}

const char* sensorTypeToString(SystemSensorType sensor_type)
{
    switch (sensor_type)
    {
    case MONOCULAR: return "MONOCULAR";
    case STEREO:    return "STEREO";
    case RGBD:      return "RGBD";
    default:        return "UNKNOWN";
    }
}

int64_t tensorRowCount(const torch::Tensor& tensor)
{
    if (!tensor.defined() || tensor.numel() == 0) return 0;
    if (tensor.dim() == 0) return 1;
    if (tensor.dim() == 1) return tensor.size(0);
    return tensor.size(0);
}

namespace {

constexpr float kCommonEvalScale = 1.0f;
constexpr float kCommonEvalVoxelLength = 5.0f * kCommonEvalScale / 512.0f;
constexpr float kCommonEvalSdfTrunc = 0.04f * kCommonEvalScale;
constexpr float kCommonEvalDepthTrunc = 30.0f;
constexpr int kCommonEvalMedianKernel = 21; // Gaussian-SLAM evaluator.py uses 20; OpenCV requires an odd kernel.
constexpr float kCommonEvalDepthOutlierThreshold = 0.1f;
const Eigen::Vector3f kCommonEvalCompensation(
    0.0f * kCommonEvalScale / 512.0f,
    2.5f * kCommonEvalScale / 512.0f,
    -2.5f * kCommonEvalScale / 512.0f);

struct TriangleMeshRgb
{
    std::vector<Eigen::Vector3f> vertices;
    std::vector<std::array<uint32_t, 3>> faces;
    std::vector<std::array<uint8_t, 3>> colors;
};

static cv::Mat filterDepthOutliersLikeGaussianSlam(const cv::Mat& depth_map)
{
    CV_Assert(depth_map.type() == CV_32FC1);
    if (depth_map.empty()) return depth_map.clone();

    cv::Mat median_filtered = depth_map.clone();
    const int num_passes = std::max(1, kCommonEvalMedianKernel / 5);
    for (int pass = 0; pass < num_passes; ++pass)
    {
        cv::medianBlur(median_filtered, median_filtered, 5);
    }

    cv::Mat filtered = depth_map.clone();
    for (int y = 0; y < depth_map.rows; ++y)
    {
        const float* src_ptr = depth_map.ptr<float>(y);
        const float* med_ptr = median_filtered.ptr<float>(y);
        float* out_ptr = filtered.ptr<float>(y);
        for (int x = 0; x < depth_map.cols; ++x)
        {
            const float d = src_ptr[x];
            const float m = med_ptr[x];
            if (!std::isfinite(d) || d <= 0.0f)
            {
                out_ptr[x] = 0.0f;
                continue;
            }
            if (!std::isfinite(m) || m <= 0.0f) continue;
            if (std::abs(d - m) > kCommonEvalDepthOutlierThreshold) out_ptr[x] = m;
        }
    }
    return filtered;
}

static bool saveTriangleMeshPly(const std::filesystem::path& ply_path, const TriangleMeshRgb& mesh)
{
    if (mesh.vertices.empty() || mesh.faces.empty()) return false;

    std::vector<float> vertices_xyz;
    vertices_xyz.reserve(mesh.vertices.size() * 3);
    for (const auto& v : mesh.vertices)
    {
        vertices_xyz.push_back(v.x());
        vertices_xyz.push_back(v.y());
        vertices_xyz.push_back(v.z());
    }

    std::vector<uint32_t> tri_idx;
    tri_idx.reserve(mesh.faces.size() * 3);
    for (const auto& f : mesh.faces)
    {
        tri_idx.push_back(f[0]);
        tri_idx.push_back(f[1]);
        tri_idx.push_back(f[2]);
    }

    std::vector<uint8_t> vertex_rgb;
    const bool has_colors = mesh.colors.size() == mesh.vertices.size();
    if (has_colors)
    {
        vertex_rgb.reserve(mesh.colors.size() * 3);
        for (const auto& c : mesh.colors)
        {
            vertex_rgb.push_back(c[0]);
            vertex_rgb.push_back(c[1]);
            vertex_rgb.push_back(c[2]);
        }
    }

    std::filebuf fb;
    fb.open(ply_path, std::ios::out | std::ios::binary);
    std::ostream out(&fb);
    if (out.fail()) return false;

    tinyply::PlyFile ply;
    ply.add_properties_to_element(
        "vertex", {"x", "y", "z"},
        tinyply::Type::FLOAT32, static_cast<uint64_t>(mesh.vertices.size()),
        reinterpret_cast<uint8_t*>(vertices_xyz.data()),
        tinyply::Type::INVALID, 0);

    if (has_colors)
    {
        ply.add_properties_to_element(
            "vertex", {"red", "green", "blue"},
            tinyply::Type::UINT8, static_cast<uint64_t>(mesh.vertices.size()),
            reinterpret_cast<uint8_t*>(vertex_rgb.data()),
            tinyply::Type::INVALID, 0);
    }

    ply.add_properties_to_element(
        "face", {"vertex_indices"},
        tinyply::Type::UINT32, static_cast<uint64_t>(mesh.faces.size()),
        reinterpret_cast<uint8_t*>(tri_idx.data()),
        tinyply::Type::UINT8, 3);

    ply.write(out, true);
    fb.close();
    return true;
}

struct SparseTsdfVolume
{
    struct Voxel
    {
        float tsdf = 1.0f;
        uint16_t weight = 0;
        std::array<uint8_t, 3> color{0, 0, 0};
    };

    struct Key
    {
        int x;
        int y;
        int z;

        bool operator==(const Key& other) const noexcept
        {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct KeyHash
    {
        std::size_t operator()(const Key& key) const noexcept
        {
            const std::uint64_t x = static_cast<std::uint32_t>(key.x) * 73856093u;
            const std::uint64_t y = static_cast<std::uint32_t>(key.y) * 19349663u;
            const std::uint64_t z = static_cast<std::uint32_t>(key.z) * 83492791u;
            return static_cast<std::size_t>(x ^ y ^ z);
        }
    };

    explicit SparseTsdfVolume(float voxel_length_in, float sdf_trunc_in)
        : voxel_length(voxel_length_in),
          sdf_trunc(sdf_trunc_in)
    {}

    static Key fromVector(const Eigen::Vector3i& idx)
    {
        return Key{idx.x(), idx.y(), idx.z()};
    }

    static Eigen::Vector3i toVector(const Key& key)
    {
        return Eigen::Vector3i(key.x, key.y, key.z);
    }

    Eigen::Vector3f voxelCenter(const Eigen::Vector3i& idx) const
    {
        return (idx.cast<float>().array() + 0.5f).matrix() * voxel_length;
    }

    void integrate(
        const cv::Mat& color_rgb,
        const cv::Mat& depth_map,
        const std::vector<float>& intr,
        const Sophus::SE3f& Tcw,
        float depth_trunc)
    {
        CV_Assert(color_rgb.type() == CV_8UC3);
        CV_Assert(depth_map.type() == CV_32FC1);
        if (intr.size() < 4) throw std::runtime_error("SparseTsdfVolume::integrate: expected fx, fy, cx, cy");

        const float fx = intr[0];
        const float fy = intr[1];
        const float cx = intr[2];
        const float cy = intr[3];
        const Sophus::SE3f Twc = Tcw.inverse();
        const Eigen::Matrix3f Rwc = Twc.rotationMatrix();
        const Eigen::Vector3f twc = Twc.translation();
        const float step = voxel_length;

        for (int v = 0; v < depth_map.rows; ++v)
        {
            const float ry = (static_cast<float>(v) - cy) / fy;
            const float* depth_ptr = depth_map.ptr<float>(v);
            const cv::Vec3b* color_ptr = color_rgb.ptr<cv::Vec3b>(v);

            for (int u = 0; u < depth_map.cols; ++u)
            {
                const float depth = depth_ptr[u];
                if (!std::isfinite(depth) || depth <= 0.0f || depth > depth_trunc) continue;

                const float rx = (static_cast<float>(u) - cx) / fx;
                const float z_min = std::max(0.0f, depth - sdf_trunc);
                const float z_max = std::min(depth_trunc, depth + sdf_trunc);
                const cv::Vec3b rgb = color_ptr[u];
                Key last_key{std::numeric_limits<int>::min(), std::numeric_limits<int>::min(), std::numeric_limits<int>::min()};

                for (float zv = z_min; zv <= z_max + 1e-6f; zv += step)
                {
                    const Eigen::Vector3f p_cam(rx * zv, ry * zv, zv);
                    const Eigen::Vector3f p_world = Rwc * p_cam + twc;
                    const Eigen::Vector3i idx = (p_world / voxel_length).array().floor().cast<int>().matrix();
                    const Key key = fromVector(idx);
                    if (key == last_key) continue;
                    last_key = key;

                    const float sdf = depth - zv;
                    const float tsdf_sample = std::max(-1.0f, std::min(1.0f, sdf / sdf_trunc));
                    auto& voxel = voxels[key];
                    const float weight_old = static_cast<float>(voxel.weight);
                    const float weight_new = std::min(65535.0f, weight_old + 1.0f);
                    voxel.tsdf = (voxel.tsdf * weight_old + tsdf_sample) / weight_new;
                    for (int c = 0; c < 3; ++c)
                    {
                        const float fused = (static_cast<float>(voxel.color[c]) * weight_old
                                           + static_cast<float>(rgb[c])) / weight_new;
                        voxel.color[c] = static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(255.0f, fused))));
                    }
                    voxel.weight = static_cast<uint16_t>(weight_new);
                }
            }
        }
    }

    TriangleMeshRgb extractMesh() const
    {
        TriangleMeshRgb mesh;
        if (voxels.empty()) return mesh;

        const std::array<Eigen::Vector3i, 8> corner_offsets = {
            Eigen::Vector3i(0, 0, 0), Eigen::Vector3i(1, 0, 0),
            Eigen::Vector3i(1, 1, 0), Eigen::Vector3i(0, 1, 0),
            Eigen::Vector3i(0, 0, 1), Eigen::Vector3i(1, 0, 1),
            Eigen::Vector3i(1, 1, 1), Eigen::Vector3i(0, 1, 1)};
        const std::array<std::array<int, 4>, 6> tetrahedra = {{
            {{0, 5, 1, 6}},
            {{0, 1, 2, 6}},
            {{0, 2, 3, 6}},
            {{0, 3, 7, 6}},
            {{0, 7, 4, 6}},
            {{0, 4, 5, 6}}
        }};
        const std::array<std::array<int, 2>, 6> tetra_edges = {{
            {{0, 1}}, {{1, 2}}, {{2, 0}}, {{0, 3}}, {{1, 3}}, {{2, 3}}
        }};

        struct CellKey
        {
            int x;
            int y;
            int z;
            bool operator==(const CellKey& other) const noexcept
            {
                return x == other.x && y == other.y && z == other.z;
            }
        };
        struct CellHash
        {
            std::size_t operator()(const CellKey& key) const noexcept
            {
                return KeyHash{}(Key{key.x, key.y, key.z});
            }
        };

        std::unordered_set<CellKey, CellHash> candidate_cells;
        candidate_cells.reserve(voxels.size() * 8);
        for (const auto& kv : voxels)
        {
            const Eigen::Vector3i base = toVector(kv.first);
            for (int dz = -1; dz <= 0; ++dz)
            {
                for (int dy = -1; dy <= 0; ++dy)
                {
                    for (int dx = -1; dx <= 0; ++dx)
                    {
                        const Eigen::Vector3i cell = base + Eigen::Vector3i(dx, dy, dz);
                        candidate_cells.insert(CellKey{cell.x(), cell.y(), cell.z()});
                    }
                }
            }
        }

        struct InterpVertex
        {
            Eigen::Vector3f pos;
            Eigen::Vector3f color;
        };

        auto append_triangle = [&](const InterpVertex& a, const InterpVertex& b, const InterpVertex& c)
        {
            const uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
            mesh.vertices.push_back(a.pos);
            mesh.vertices.push_back(b.pos);
            mesh.vertices.push_back(c.pos);
            auto clamp_color = [](const Eigen::Vector3f& col)
            {
                return std::array<uint8_t, 3>{
                    static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(255.0f, col.x())))),
                    static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(255.0f, col.y())))),
                    static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(255.0f, col.z()))))};
            };
            mesh.colors.push_back(clamp_color(a.color));
            mesh.colors.push_back(clamp_color(b.color));
            mesh.colors.push_back(clamp_color(c.color));
            mesh.faces.push_back({base, base + 1, base + 2});
        };

        for (const auto& cell_key : candidate_cells)
        {
            const Eigen::Vector3i cell(cell_key.x, cell_key.y, cell_key.z);
            std::array<float, 8> corner_tsdf{};
            std::array<uint16_t, 8> corner_weight{};
            std::array<Eigen::Vector3f, 8> corner_pos{};
            std::array<Eigen::Vector3f, 8> corner_color{};
            bool any_weight = false;

            for (int i = 0; i < 8; ++i)
            {
                const Key key = fromVector(cell + corner_offsets[i]);
                const auto it = voxels.find(key);
                if (it == voxels.end())
                {
                    corner_weight[i] = 0;
                    continue;
                }
                corner_tsdf[i] = it->second.tsdf;
                corner_weight[i] = it->second.weight;
                corner_pos[i] = voxelCenter(cell + corner_offsets[i]);
                corner_color[i] = Eigen::Vector3f(
                    static_cast<float>(it->second.color[0]),
                    static_cast<float>(it->second.color[1]),
                    static_cast<float>(it->second.color[2]));
                any_weight = any_weight || (corner_weight[i] > 0);
            }
            if (!any_weight) continue;

            for (const auto& tet : tetrahedra)
            {
                bool tet_valid = true;
                for (int local_idx = 0; local_idx < 4; ++local_idx)
                {
                    if (corner_weight[tet[local_idx]] == 0)
                    {
                        tet_valid = false;
                        break;
                    }
                }
                if (!tet_valid) continue;

                std::vector<InterpVertex> poly_vertices;
                poly_vertices.reserve(4);
                for (const auto& edge : tetra_edges)
                {
                    const int a = tet[edge[0]];
                    const int b = tet[edge[1]];
                    const bool a_inside = corner_tsdf[a] < 0.0f;
                    const bool b_inside = corner_tsdf[b] < 0.0f;
                    if (a_inside == b_inside) continue;

                    const float denom = corner_tsdf[a] - corner_tsdf[b];
                    float t = 0.5f;
                    if (std::abs(denom) > 1e-8f) t = corner_tsdf[a] / denom;
                    t = std::max(0.0f, std::min(1.0f, t));
                    poly_vertices.push_back({
                        corner_pos[a] + t * (corner_pos[b] - corner_pos[a]),
                        corner_color[a] + t * (corner_color[b] - corner_color[a])});
                }

                if (poly_vertices.size() < 3) continue;
                if (poly_vertices.size() == 3)
                {
                    append_triangle(poly_vertices[0], poly_vertices[1], poly_vertices[2]);
                    continue;
                }

                Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
                for (const auto& vtx : poly_vertices) centroid += vtx.pos;
                centroid /= static_cast<float>(poly_vertices.size());

                Eigen::Vector3f normal = (poly_vertices[1].pos - poly_vertices[0].pos)
                                       .cross(poly_vertices[2].pos - poly_vertices[0].pos);
                if (normal.norm() < 1e-8f && poly_vertices.size() >= 4)
                {
                    normal = (poly_vertices[2].pos - poly_vertices[0].pos)
                           .cross(poly_vertices[3].pos - poly_vertices[0].pos);
                }
                if (normal.norm() < 1e-8f) continue;
                normal.normalize();

                Eigen::Vector3f axis_u = poly_vertices[0].pos - centroid;
                if (axis_u.norm() < 1e-8f) axis_u = poly_vertices[1].pos - centroid;
                if (axis_u.norm() < 1e-8f) continue;
                axis_u.normalize();
                Eigen::Vector3f axis_v = normal.cross(axis_u);
                if (axis_v.norm() < 1e-8f) continue;
                axis_v.normalize();

                std::vector<int> order(poly_vertices.size());
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(), [&](int lhs, int rhs)
                {
                    const Eigen::Vector3f dl = poly_vertices[lhs].pos - centroid;
                    const Eigen::Vector3f dr = poly_vertices[rhs].pos - centroid;
                    const float al = std::atan2(dl.dot(axis_v), dl.dot(axis_u));
                    const float ar = std::atan2(dr.dot(axis_v), dr.dot(axis_u));
                    return al < ar;
                });

                append_triangle(poly_vertices[order[0]], poly_vertices[order[1]], poly_vertices[order[2]]);
                if (poly_vertices.size() == 4)
                {
                    append_triangle(poly_vertices[order[0]], poly_vertices[order[2]], poly_vertices[order[3]]);
                }
            }
        }

        return mesh;
    }

    float voxel_length;
    float sdf_trunc;
    std::unordered_map<Key, Voxel, KeyHash> voxels;
};

std::mutex g_dumpkf_mutex;

static void saveKfPng_fromFloatRGB(const cv::Mat& im_float_rgb,   // CV_32FC3 in [0..1] RGB
                                   int fid,
                                   const std::filesystem::path& imgs_dir)
{
    // convert float[0..1] RGB -> 8-bit BGR
    cv::Mat tmp8, bgr8;
    im_float_rgb.convertTo(tmp8, CV_8UC3, 255.0);
    cv::cvtColor(tmp8, bgr8, cv::COLOR_RGB2BGR);
    std::ostringstream oss;
    oss << "kf_" << fid << ".png";
    const auto img_path = (imgs_dir / oss.str()).string();
    bool ok = cv::imwrite(img_path, bgr8);
    // std::cout << "[saveKfPng] " << img_path << " ok=" << std::boolalpha << ok
    //           << " mean(BGR)=" << cv::mean(bgr8) << std::endl;
}

// Ported from SVRaster camera.depth2normal:
// third_party/svraster/src/cameras.py::CameraBase.depth2normal
static torch::Tensor depth2normalSVRaster(
    const sv::MiniCam& cam,
    const torch::Tensor& depth,
    int ks,
    float tol_cos)
{
    using namespace torch::indexing;
    auto opts = depth.options();
    const int64_t H = depth.size(0);
    const int64_t W = depth.size(1);
    auto normal = torch::zeros({3, H, W}, opts);
    if (H < 3 || W < 3) return normal;

    const float fx = (cam.fx > 1e-6f)
        ? cam.fx
        : (0.5f * static_cast<float>(W) / std::max(cam.tanfovx, 1e-6f));
    const float fy = (cam.fy > 1e-6f)
        ? cam.fy
        : (0.5f * static_cast<float>(H) / std::max(cam.tanfovy, 1e-6f));
    if (fx <= 1e-6f || fy <= 1e-6f) return normal;

    auto uu = torch::arange(0, W, opts).view({1, W}).expand({H, W});
    auto vv = torch::arange(0, H, opts).view({H, 1}).expand({H, W});
    auto x = (uu - cam.cx) / fx;
    auto y = (vv - cam.cy) / fy;
    auto z = torch::ones_like(x);

    auto rd_cam = torch::stack({x, y, z}, 0);
    rd_cam = torch::nn::functional::normalize(
        rd_cam,
        torch::nn::functional::NormalizeFuncOptions().dim(0).eps(1e-12));

    auto c2w = cam.c2w.to(depth.device(), depth.scalar_type());
    auto R = c2w.index({Slice(0, 3), Slice(0, 3)});
    auto cam_pos = c2w.index({Slice(0, 3), 3}).view({3, 1, 1});
    auto rd_world = torch::matmul(R, rd_cam.view({3, H * W})).view({3, H, W});
    rd_world = torch::nn::functional::normalize(
        rd_world,
        torch::nn::functional::NormalizeFuncOptions().dim(0).eps(1e-12));

    auto pts = cam_pos + rd_world * depth.unsqueeze(0);
    ks = std::max(3, ks);
    if ((ks % 2) == 0) ks += 1;
    const int64_t pad = ks / 2;
    const int64_t ks_1 = ks - 1;
    if (H <= ks_1 || W <= ks_1 || (H - 2 * pad) <= 0 || (W - 2 * pad) <= 0) return normal;

    auto dx = pts.index({Slice(), Slice(pad, H - pad), Slice(ks_1, W)}) -
              pts.index({Slice(), Slice(pad, H - pad), Slice(0, W - ks_1)});
    auto dy = pts.index({Slice(), Slice(ks_1, H), Slice(pad, W - pad)}) -
              pts.index({Slice(), Slice(0, H - ks_1), Slice(pad, W - pad)});
    auto n_patch = torch::cross(dx, dy, 0);
    n_patch = torch::nn::functional::normalize(
        n_patch,
        torch::nn::functional::NormalizeFuncOptions().dim(0).eps(1e-12));
    normal.index_put_({Slice(), Slice(pad, H - pad), Slice(pad, W - pad)}, n_patch);

    if (tol_cos > 0.0f) {
        auto pts_dir = torch::nn::functional::normalize(
            pts - cam_pos,
            torch::nn::functional::NormalizeFuncOptions().dim(0).eps(1e-12));
        auto dot = (normal * pts_dir).sum(0);
        auto mask = (dot > tol_cos).to(normal.scalar_type());
        normal = normal * mask.unsqueeze(0);
    }
    return normal;
}

static torch::Tensor validDepthSupportMask(
    const torch::Tensor& valid,
    int ks)
{
    using namespace torch::indexing;
    if (valid.dim() != 2) {
        return torch::zeros_like(valid, torch::TensorOptions().dtype(torch::kBool));
    }

    ks = std::max(3, ks);
    if ((ks % 2) == 0) ks += 1;

    const int64_t H = valid.size(0);
    const int64_t W = valid.size(1);
    const int64_t pad = ks / 2;
    if (H <= 2 * pad || W <= 2 * pad) {
        return torch::zeros_like(valid, torch::TensorOptions().dtype(torch::kBool));
    }

    auto valid_bool = valid.to(torch::kBool);
    auto support = torch::zeros_like(valid_bool);
    auto inner = torch::ones(
        {H - 2 * pad, W - 2 * pad},
        torch::TensorOptions().dtype(torch::kBool).device(valid.device()));

    for (int64_t dy = -pad; dy <= pad; ++dy) {
        for (int64_t dx = -pad; dx <= pad; ++dx) {
            inner = inner & valid_bool.index({
                Slice(pad + dy, H - pad + dy),
                Slice(pad + dx, W - pad + dx)});
        }
    }

    support.index_put_({Slice(pad, H - pad), Slice(pad, W - pad)}, inner);
    return support;
}

// Ported from SVRaster normal-depth consistency:
// third_party/svraster/src/utils/loss_utils.py::NormalDepthConsistencyLoss
static torch::Tensor normalDepthConsistencyLossSVRaster(
    const sv::MiniCam& cam,
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    const int ks,
    const float tol_deg)
{
    using namespace torch::indexing;

    auto it_T = render_pkg.find("raw_T");
    auto it_depth = render_pkg.find("raw_depth");
    auto it_normal = render_pkg.find("raw_normal");
    auto zero_device = cam.c2w.device();
    if (it_T != render_pkg.end() && it_T->second.defined()) zero_device = it_T->second.device();
    else if (it_depth != render_pkg.end() && it_depth->second.defined()) zero_device = it_depth->second.device();
    else if (it_normal != render_pkg.end() && it_normal->second.defined()) zero_device = it_normal->second.device();

    if (it_T == render_pkg.end() || it_depth == render_pkg.end() || it_normal == render_pkg.end()) {
        return torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(zero_device));
    }
    if (!it_T->second.defined() || !it_depth->second.defined() || !it_normal->second.defined()) {
        return torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(zero_device));
    }

    auto raw_T = it_T->second;
    if (raw_T.dim() == 4 && raw_T.size(0) == 1) raw_T = raw_T.squeeze(0);
    if (raw_T.dim() == 3 && raw_T.size(0) == 1) raw_T = raw_T.squeeze(0);
    if (raw_T.dim() != 2) {
        return torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(raw_T.device()));
    }
    auto render_alpha = 1.0f - raw_T.detach();

    auto raw_depth = it_depth->second;
    if (raw_depth.dim() == 4 && raw_depth.size(0) == 1) raw_depth = raw_depth.squeeze(0);
    if (raw_depth.dim() != 3 || raw_depth.size(0) < 1) {
        return torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(raw_depth.device()));
    }
    auto render_depth = raw_depth.index({0});

    auto render_normal = it_normal->second;
    if (render_normal.dim() == 4 && render_normal.size(0) == 1) render_normal = render_normal.squeeze(0);
    if (render_normal.dim() != 3 || render_normal.size(0) < 3) {
        return torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(render_normal.device()));
    }
    if (render_normal.size(0) > 3) render_normal = render_normal.index({Slice(0, 3)});

    constexpr float kPi = 3.14159265358979323846f;
    const float tol_cos = std::cos(tol_deg * kPi / 180.0f);
    auto n_mean = depth2normalSVRaster(cam, render_depth, ks, tol_cos);

    auto target = render_alpha.square();
    n_mean = n_mean * render_alpha.unsqueeze(0);
    auto mask = (n_mean != 0).any(0);
    auto loss_map = (target - (render_normal * n_mean).sum(0)) * mask.to(target.scalar_type());
    return loss_map.mean();
}

// template<typename KFMap, typename CamContainer>
// void dumpKeyframesForProjectionFile(const KFMap& kfmap,
//                                     const CamContainer& cameras,
//                                     const std::filesystem::path& out_dir)
// {
//     std::lock_guard<std::mutex> lk(g_dumpkf_mutex);
//     std::filesystem::create_directories(out_dir);
//     std::filesystem::create_directories(out_dir / "imgs");
//     const auto tmp_file = out_dir / "keyframes_proj.tmp";
//     const auto out_file = out_dir / "keyframes_proj.txt";
//     std::ofstream os(tmp_file, std::ios::trunc);
//     if (!os) { std::cerr << "[dumpKF] cannot open " << tmp_file << '\n'; return; }
//     size_t n_lines = 0;
//     for (const auto& kv : kfmap) {
//         const auto& kf_ptr = kv.second;
//         if (!kf_ptr) continue;
//         const auto cam_id = kf_ptr->camera_id_;
//         auto cam_it = cameras.find(cam_id);
//         if (cam_it == cameras.end()) continue;
//         const sv::Camera& cam = cam_it->second;
//         const int   W  = kf_ptr->image_width_;
//         const int   H  = kf_ptr->image_height_;
//         const float fx = cam.fx(), fy = cam.fy(), cx = cam.cx(), cy = cam.cy();
//         const Eigen::Matrix4f Tcw = kf_ptr->getWorld2View2(kf_ptr->trans_, kf_ptr->scale_);
//         std::ostringstream oss;
//         oss << "imgs/kf_" << kf_ptr->fid_ << ".png";
//         const std::string rel_img = oss.str();
//         os << kf_ptr->fid_ << ' '
//            << W << ' ' << H << ' '
//            << std::setprecision(9) << fx << ' ' << fy << ' ' << cx << ' ' << cy << ' ';
//         for (int r = 0; r < 4; ++r)
//             for (int c = 0; c < 4; ++c)
//                 os << Tcw(r,c) << ' ';
//         os << rel_img << '\n';
//         ++n_lines;
//     }
//     os.close();
//     std::error_code ec;
//     std::filesystem::rename(tmp_file, out_file, ec);
//     if (ec) {
//         std::filesystem::copy_file(tmp_file, out_file,
//                                    std::filesystem::copy_options::overwrite_existing, ec);
//         std::filesystem::remove(tmp_file);
//     }
//     // std::cout << "[dumpKF] wrote " << out_file << " (" << n_lines << " lines)\n";
// }
} // namespace

inline void write_npy_float32(const std::string &path, const torch::Tensor &tensor)
{
    // Ensure tensor is contiguous and on CPU
    auto t = tensor.to(torch::kCPU).contiguous();

    if (t.dtype() != torch::kFloat32) {
        throw std::runtime_error("[write_npy_float32] Tensor must be float32.");
    }
    if (t.dim() < 1) {
        throw std::runtime_error("[write_npy_float32] Tensor must have at least 1 dimension.");
    }

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("[write_npy_float32] Failed to open file: " + path);
    }

    // Prepare NPY header
    // Magic string: \x93NUMPY
    const char magic[] = "\x93NUMPY";
    file.write(magic, 6);

    // Version 1.0
    unsigned char major = 1;
    unsigned char minor = 0;
    file.write(reinterpret_cast<char *>(&major), 1);
    file.write(reinterpret_cast<char *>(&minor), 1);

    // Build shape string, e.g. "(100, 3)"
    std::ostringstream shape_stream;
    shape_stream << "(";
    for (int i = 0; i < t.dim(); i++) {
        shape_stream << t.size(i);
        if (i < t.dim() - 1)
            shape_stream << ", ";
    }
    if (t.dim() == 1) {
        shape_stream << ",";
    }
    shape_stream << ")";

    // Little-endian, float32, fortran_order=False
    std::ostringstream header_stream;
    header_stream << "{'descr': '<f4', 'fortran_order': False, 'shape': "
                  << shape_stream.str() << ", }";
    std::string header = header_stream.str();

    // Pad header to 16-byte alignment
    int header_len = header.size() + 1; // +1 for newline
    int padding = 16 - ((10 + header_len) % 16);
    header.append(padding, ' ');
    header += "\n";

    // Write header length (2 bytes, little-endian)
    uint16_t header_size = static_cast<uint16_t>(header.size());
    file.write(reinterpret_cast<char *>(&header_size), 2);

    // Write header
    file.write(header.c_str(), header.size());

    // Write raw data
    file.write(reinterpret_cast<const char *>(t.data_ptr()), t.numel() * sizeof(float));
    file.close();
}

inline torch::Tensor read_npy_float32(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("[read_npy_float32] Failed to open file: " + path);
    }

    // Check magic string
    char magic[6];
    file.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") {
        throw std::runtime_error("[read_npy_float32] Invalid NPY file magic string: " + path);
    }

    // Read version
    unsigned char major, minor;
    file.read(reinterpret_cast<char *>(&major), 1);
    file.read(reinterpret_cast<char *>(&minor), 1);
    if (!(major == 1 && minor == 0)) {
        throw std::runtime_error("[read_npy_float32] Only NPY v1.0 supported.");
    }

    // Read header length (little-endian uint16)
    uint16_t header_len;
    file.read(reinterpret_cast<char *>(&header_len), 2);

    // Read header content
    std::vector<char> header_buf(header_len);
    file.read(header_buf.data(), header_len);
    std::string header(header_buf.begin(), header_buf.end());

    // Parse shape from header
    auto pos1 = header.find("(");
    auto pos2 = header.find(")");
    if (pos1 == std::string::npos || pos2 == std::string::npos || pos2 <= pos1) {
        throw std::runtime_error("[read_npy_float32] Failed to parse shape.");
    }
    std::string shape_str = header.substr(pos1 + 1, pos2 - pos1 - 1);

    // Tokenize numbers in shape
    std::vector<int64_t> dims;
    std::stringstream ss(shape_str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        std::stringstream num(item);
        int64_t val;
        if (num >> val)
            dims.push_back(val);
    }

    // Count elements
    int64_t numel = 1;
    for (auto d : dims)
        numel *= d;

    // Read data
    torch::Tensor tensor = torch::empty(dims, torch::kFloat32);
    file.read(reinterpret_cast<char *>(tensor.data_ptr()), numel * sizeof(float));
    file.close();

    return tensor;
}

static void save_initial_pcd_npy(
    const std::string& dir,
    const std::map<point3D_id_t, Point3D>& pcd)
{
    const int N = (int)pcd.size();
    torch::Tensor xyz = torch::empty({N,3}, torch::kFloat32);
    torch::Tensor rgb = torch::empty({N,3}, torch::kFloat32);
    torch::Tensor ids = torch::empty({N},   torch::kInt64);   // CPU

    int64_t* ids_ptr = ids.data_ptr<int64_t>();

    int i=0;
    for (const auto& kv : pcd) {
        const auto id = (int64_t)kv.first;
        const auto& P = kv.second;

        xyz.index_put_({i,0}, (float)P.xyz_(0));
        xyz.index_put_({i,1}, (float)P.xyz_(1));
        xyz.index_put_({i,2}, (float)P.xyz_(2));

        rgb.index_put_({i,0}, (float)P.color_(0));
        rgb.index_put_({i,1}, (float)P.color_(1));
        rgb.index_put_({i,2}, (float)P.color_(2));

        ids_ptr[i] = id;
        ++i;
    }

    // make sure the directory exists on your side

    write_npy_float32(dir + "/initial_xyz.npy", xyz);
    write_npy_float32(dir + "/initial_rgb.npy", rgb);
    torch::save(ids, dir + "/initial_ids.pt");
}

// pcd_log.h (continued)
static void log_increase_batch_npy(
    const std::string& dir,
    const std::vector<float>& points_flat,
    const std::vector<float>& colors_flat,
    int iter, int batch_idx)
{
    if (points_flat.size() % 3 != 0 || colors_flat.size() % 3 != 0)
        throw std::runtime_error("log_increase_batch_npy: flat vectors must be multiples of 3");

    const int Np = (int)(points_flat.size() / 3);
    const int Nc = (int)(colors_flat.size() / 3);
    if (Np != Nc)
        throw std::runtime_error("log_increase_batch_npy: points and colors count mismatch");

    // Ensure .../batches exists (C++17)
    #if __has_include(<filesystem>)
    #include <filesystem>
    std::filesystem::create_directories(dir + "/batches");
    #endif

    torch::Tensor xyz = torch::from_blob(
        const_cast<float*>(points_flat.data()),
        {Np, 3},
        torch::TensorOptions().dtype(torch::kFloat32))
        .clone(); // clone so we own memory

    torch::Tensor rgb = torch::from_blob(
        const_cast<float*>(colors_flat.data()),
        {Nc, 3},
        torch::TensorOptions().dtype(torch::kFloat32))
        .clone();

    char fn_xyz[512], fn_rgb[512], fn_meta[512];
    std::snprintf(fn_xyz,  sizeof(fn_xyz),  "%s/batches/xyz_%06d.npy",  dir.c_str(), batch_idx);
    std::snprintf(fn_rgb,  sizeof(fn_rgb),  "%s/batches/rgb_%06d.npy",  dir.c_str(), batch_idx);
    std::snprintf(fn_meta, sizeof(fn_meta), "%s/batches/meta_%06d.txt", dir.c_str(), batch_idx);

    write_npy_float32(fn_xyz, xyz);
    write_npy_float32(fn_rgb, rgb);

    std::ofstream m(fn_meta);
    m << "iter " << iter << "\n";
}

// Load initial + all batches/* (xyz_XXXXXX.npy + rgb_XXXXXX.npy)
static std::map<point3D_id_t, Point3D>
load_full_pcd_from_logs(const std::string& dir)
{
    std::map<point3D_id_t, Point3D> out;

    // --- 1) initial blobs ------------------------------------------------
    auto xyz0 = read_npy_float32(dir + "/initial_xyz.npy"); // [N,3] float32
    auto rgb0 = read_npy_float32(dir + "/initial_rgb.npy"); // [N,3] float32

    torch::Tensor ids0;
    torch::load(ids0, dir + "/initial_ids.pt");             // [N] int64
    auto ids_ptr = ids0.data_ptr<int64_t>();

    const int64_t N0 = xyz0.size(0);
    for (int64_t i = 0; i < N0; ++i) {
        Point3D P;
        // If Point3D uses Vector3d, keep (double) casts
        P.xyz_(0)   = (double)xyz0.index({i,0}).item<float>();
        P.xyz_(1)   = (double)xyz0.index({i,1}).item<float>();
        P.xyz_(2)   = (double)xyz0.index({i,2}).item<float>();
        P.color_(0) = (double)rgb0.index({i,0}).item<float>();
        P.color_(1) = (double)rgb0.index({i,1}).item<float>();
        P.color_(2) = (double)rgb0.index({i,2}).item<float>();

        point3D_id_t id = (point3D_id_t)ids_ptr[i];
        out[id] = P;
    }

    // Next ID for batch points (append-only)
    point3D_id_t next_id = out.empty() ? 0 : (std::prev(out.end())->first + 1);

    // --- 2) batches/* ----------------------------------------------------
    const std::string batches_dir = dir + "/batches";
    if (std::filesystem::exists(batches_dir)) {
        // Collect xyz files, sort by numeric index
        std::vector<std::pair<int, std::string>> xyz_files; // (idx, path)
        std::regex rex(R"(xyz_(\d+)\.npy)");

        for (auto& p : std::filesystem::directory_iterator(batches_dir)) {
            if (!p.is_regular_file()) continue;
            const auto name = p.path().filename().string();
            std::smatch m;
            if (std::regex_match(name, m, rex)) {
                int idx = std::stoi(m[1]);
                xyz_files.emplace_back(idx, p.path().string());
            }
        }
        std::sort(xyz_files.begin(), xyz_files.end(),
                  [](auto& a, auto& b){ return a.first < b.first; });

        for (auto& [idx, xfile] : xyz_files) {
            // find matching rgb file
            std::string rfile = xfile;
            if (auto pos = rfile.rfind("xyz_"); pos != std::string::npos) {
                rfile.replace(pos, 3, "rgb");
            } else {
                // fallback: construct explicit path
                char buf[512];
                std::snprintf(buf, sizeof(buf), "%s/rgb_%06d.npy", batches_dir.c_str(), idx);
                rfile = buf;
            }

            // read
            auto xb = read_npy_float32(xfile);  // [Nb,3]
            auto rb = read_npy_float32(rfile);  // [Nb,3]
            const int64_t Nb = xb.size(0);

            // sanity: shapes must match
            if (rb.size(0) != Nb || xb.size(1) != 3 || rb.size(1) != 3) {
                throw std::runtime_error("Batch shape mismatch at index " + std::to_string(idx));
            }

            for (int64_t i = 0; i < Nb; ++i, ++next_id) {
                Point3D P;
                P.xyz_(0)   = (double)xb.index({i,0}).item<float>();
                P.xyz_(1)   = (double)xb.index({i,1}).item<float>();
                P.xyz_(2)   = (double)xb.index({i,2}).item<float>();
                P.color_(0) = (double)rb.index({i,0}).item<float>();
                P.color_(1) = (double)rb.index({i,1}).item<float>();
                P.color_(2) = (double)rb.index({i,2}).item<float>();
                out[next_id] = P;
            }
        }
    }

    return out;
}

// // --- header-scope constants (top of voxel_mapper.cpp) ---
// static constexpr int HMAP_R_MAX = 6;   // cap splat radius (pixels)
// static constexpr float Z_EPS = 1e-6f;
// static torch::Tensor approxGeomFromCentersAndSize(const sv::MiniCam& cam,
//                                                   const torch::Tensor& vox_center, // [N,3], CPU ok
//                                                   const torch::Tensor& vox_size,   // [N],   CPU ok
//                                                   int H, int W)
// {
//     auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
//     auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
//     torch::Tensor geom = torch::full({H, W}, (int64_t)-1, opts_i64);
//     torch::Tensor zbuf = torch::full({H, W}, std::numeric_limits<float>::infinity(), opts_f32);

//     torch::Tensor centers = vox_center.detach().to(torch::kCPU).contiguous();
//     torch::Tensor sizes   = vox_size.detach().to(torch::kCPU).contiguous();

//     torch::Tensor w2c = cam.w2c.detach().to(torch::kCPU).contiguous(); // [4,4]
//     const float fx = cam.fx, fy = cam.fy, cx = cam.cx, cy = cam.cy;

//     int64_t N = centers.size(0);
//     auto xyz1 = torch::cat({centers, torch::ones({N,1}, opts_f32)}, 1);
//     auto cam_xyz1 = torch::matmul(xyz1, w2c.t());
//     auto X = cam_xyz1.index({torch::indexing::Slice(), 0});
//     auto Y = cam_xyz1.index({torch::indexing::Slice(), 1});
//     auto Z = cam_xyz1.index({torch::indexing::Slice(), 2});

//     auto X_a = X.data_ptr<float>();
//     auto Y_a = Y.data_ptr<float>();
//     auto Z_a = Z.data_ptr<float>();
//     auto S_a = sizes.data_ptr<float>();
//     auto zbuf_a = zbuf.data_ptr<float>();
//     auto geom_a = geom.data_ptr<int64_t>();

//     auto in_bounds = [&](int u, int v) { return (u>=0 && u<W && v>=0 && v<H); };

//     for (int64_t i = 0; i < N; ++i) {
//         float z = Z_a[i];
//         if (z <= cam.near) continue;
//         float u_f = fx * (X_a[i] / z) + cx;
//         float v_f = fy * (Y_a[i] / z) + cy;

//         // projected *radius* in pixels ~ 0.5 * f * (size/z)
//         int ru = std::max(1, (int)std::ceil(0.5f * fx * (S_a[i] / std::abs(z))));
//         int rv = std::max(1, (int)std::ceil(0.5f * fy * (S_a[i] / std::abs(z))));

//         int u0 = (int)std::floor(u_f + 0.5f);
//         int v0 = (int)std::floor(v_f + 0.5f);

//         for (int dv = -rv; dv <= rv; ++dv) {
//             int vv = v0 + dv; if (vv < 0 || vv >= H) continue;
//             for (int du = -ru; du <= ru; ++du) {
//                 int uu = u0 + du; if (uu < 0 || uu >= W) continue;
//                 // optional: circular mask
//                 if ((du*du)/(float)(ru*ru) + (dv*dv)/(float)(rv*rv) > 1.0f) continue;
//                 int idx = vv * W + uu;
//                 if (z < zbuf_a[idx]) { zbuf_a[idx] = z; geom_a[idx] = i; }
//             }
//         }
//     }
//     return geom; // int64 [H,W], -1 empty
// }

VoxelMapper::VoxelMapper(std::shared_ptr<ORB_SLAM3::System> pSLAM,
                         const std::filesystem::path& config_file_path,
                        std::filesystem::path result_dir,
                        int seed,
                        torch::DeviceType device_type)
    : mpSLAM(pSLAM),
      initial_mapped_(false),
      interrupt_training_(false),
      stopped_(false),
      iteration_(0),
      ema_loss_for_log_(0.0f),
      SLAM_ended_(false),
      loop_closure_iteration_(false),
      min_num_initial_map_kfs_(15UL),
      large_rot_th_(1e-1f),
      large_trans_th_(1e-2f),
      training_report_interval_(0)
{
    std::srand(seed);
    torch::manual_seed(seed);
    loss_log_.open("loss.csv", std::ios::out);
    loss_ssim_log_.open("loss_ssim.csv", std::ios::out);
    loss_l1_log_.open("loss_l1.csv", std::ios::out);
    loss_l2_log_.open("loss_l2.csv", std::ios::out);

    if (device_type == torch::kCUDA && torch::cuda::is_available()) {
        std::cout << "[VoxelMapper] CUDA available! Training on GPU." << std::endl;
        device_type_ = torch::kCUDA;
        mDevice = torch::Device(torch::kCUDA);
        model_params_.data_device_ = "cuda";
    } else {
        std::cout << "[VoxelMapper] Training on CPU." << std::endl;
        device_type_ = torch::kCPU;
        mDevice = torch::Device(torch::kCPU);
        model_params_.data_device_ = "cpu";
    }

    // result_dir_ = mOutDir;
    result_dir_ = result_dir;
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir);
    config_file_path_ = config_file_path;
    readConfigFromFile(config_file_path);

    // Background & override color setup
    std::vector<float> bg_color = {0.0f, 0.0f, 0.0f};  // no white-background logic needed
    if (model_params_.white_background_)
         bg_color = {1.0f, 1.0f, 1.0f};
     else
         bg_color = {0.0f, 0.0f, 0.0f};
    background_ = torch::tensor(bg_color,
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    override_color_ = torch::empty(0, torch::TensorOptions().device(device_type_));

    voxel_model_ = std::make_shared<sv::VoxelModel>(model_params_);
    voxel_model_->setFilterNearVoxels(opt_params_.filter_near_voxels_);
    voxel_model_->setFilterFarVoxelsOnInsert(opt_params_.filter_far_voxels_on_insert_);
    voxel_model_->setRenderedDepthCandidateRealAdjacency(
        rendered_depth_insert_require_real_adjacency_,
        rendered_depth_insert_adjacency_radius_cells_);
    voxel_model_->setGeoGridInitCallback(
        [this](const torch::Tensor& grid_points_world, float ray_interval_m) {
            return this->computeTsdfDensityInitForGridPoints(
                grid_points_world,
                ray_interval_m);
        });

    std::cout << "[VoxelMapper] Voxel cfg: fixed_vox_size=" << voxel_model_->fixed_vox_size_
              << " fill_empty_cells=" << static_cast<int>(voxel_model_->fill_empty_cells_)
              << " fill_warmup_iters=" << voxel_model_->fill_empty_cells_warmup_iters_
              << " local_frontier=" << static_cast<int>(voxel_model_->use_local_frontier_fill_)
              << " dense_core_neighbor_fill=" << static_cast<int>(voxel_model_->use_dense_core_neighbor_fill_)
              << " dense_rate=" << voxel_model_->dense_core_pcd_density_rate_
              << " max_artificial=" << voxel_model_->max_artificial_cells_
              << " fill_filter_near=" << static_cast<int>(voxel_model_->filterNearVoxels())
              << " filter_far_insert=" << static_cast<int>(voxel_model_->filterFarVoxelsOnInsert())
              << " save_rendered_mesh_eval=" << static_cast<int>(save_rendered_mesh_eval_)
              << " rendered_mesh_backend=" << rendered_mesh_backend_
              << " prune_recompute_dense_core=" << static_cast<int>(opt_params_.prune_recompute_dense_core_)
              << " promote_artificial=" << static_cast<int>(voxel_model_->enableArtificialPromotion())
              << " tsdf_density_init=" << static_cast<int>(tsdf_density_init_)
              << "\n";

    scene_       = std::make_shared<sv::VoxelScene>(model_params_);
    size_t N = scene_->keyframes().size();
    best_loss_per_kf_.assign(N,  std::numeric_limits<float>::infinity());
    worst_loss_per_kf_.assign(N, -std::numeric_limits<float>::infinity());
    extrema_dir_ = result_dir_ / "extrema";
    {
        std::error_code ec;
        std::filesystem::remove_all(extrema_dir_, ec); // best-effort cleanup
    }
    std::filesystem::create_directories(extrema_dir_);

    switch (pSLAM->getSensorType()) {
    case ORB_SLAM3::System::MONOCULAR:
    case ORB_SLAM3::System::IMU_MONOCULAR:
    {
        this->sensor_type_ = MONOCULAR;
    }
    break;
    case ORB_SLAM3::System::STEREO:
    case ORB_SLAM3::System::IMU_STEREO:
    {
        this->sensor_type_ = STEREO;
        this->stereo_baseline_length_ = pSLAM->getSettings()->b();
        this->stereo_cv_sgm_ = cv::cuda::createStereoSGM(
            this->stereo_min_disparity_,
            this->stereo_num_disparity_);
        this->stereo_Q_ = pSLAM->getSettings()->Q().clone();
        stereo_Q_.convertTo(stereo_Q_, CV_32FC3, 1.0);
    }
    break;
    case ORB_SLAM3::System::RGBD:
    case ORB_SLAM3::System::IMU_RGBD:
    {
        this->sensor_type_ = RGBD;
    }
    break;
    default:
    {
        throw std::runtime_error("[Gaussian Mapper]Unsupported sensor type!");
    }
    break;
    }

    // std::cout << "[inactive_geo_densify/config] enabled=" << inactive_geo_densify_
    //           << " sensor=" << sensorTypeToString(sensor_type_)
    //           << " depth_cache=" << max_depth_cached_
    //           << " mono_max_pixel_dist=" << monocular_inactive_geo_densify_max_pixel_dist_
    //           << " rgbd_depth_range=[" << RGBD_min_depth_ << "," << RGBD_max_depth_ << "]"
    //           << std::endl;
    // std::cout << "[rendered_depth_insert/config] enabled=" << rendered_depth_insert_
    //           << " sensor=" << sensorTypeToString(sensor_type_)
    //           << " stride=" << rendered_depth_insert_stride_
    //           << " frontier_radius_px=" << rendered_depth_insert_frontier_radius_px_
    //           << " max_points_per_kf=" << rendered_depth_insert_max_points_per_kf_
    //           << " normal_offset_vox=" << rendered_depth_insert_normal_offset_vox_
    //           << " require_real_adjacency=" << (rendered_depth_insert_require_real_adjacency_ ? 1 : 0)
    //           << " adjacency_radius_cells=" << rendered_depth_insert_adjacency_radius_cells_
    //           << " promote_min_support=" << opt_params_.rendered_depth_candidate_promote_min_support_
    //           << " prune_kf_age=" << opt_params_.rendered_depth_candidate_prune_kf_age_
    //           << " rerun=" << (rerun_rendered_depth_insert_ ? 1 : 0)
    //           << std::endl;
    // std::cout << "[mono_prior_densify/config] enabled=" << depthanything_densify_
    //           << " sensor=" << sensorTypeToString(sensor_type_)
    //           << " stride=" << depthanything_densify_stride_
    //           << " min_sparse_anchors=" << depthanything_densify_min_sparse_anchors_
    //           << " alignment_mode=" << depthanything_densify_alignment_mode_
    //           << std::endl;
    // std::cout << "[mono_prior_fill_holes/config] enabled=" << depthanything_fill_holes_
    //           << " sensor=" << sensorTypeToString(sensor_type_)
    //           << " stride=" << depthanything_densify_stride_
    //           << " min_sparse_anchors=" << depthanything_densify_min_sparse_anchors_
    //           << " alignment_mode=" << depthanything_densify_alignment_mode_
    //           << " initial_backfill=" << (depthanything_fill_holes_initial_backfill_ ? 1 : 0)
    //           << " empty_depth_eps=" << kMonoPriorFillHolesEmptyDepthEps
    //           << std::endl;
    // std::cout << "[mono_prior_regularization/config] loss_mode="
    //           << mono_prior_loss_mode_
    //           << " normal_mode=" << mono_prior_normal_mode_
    //           << " model_id=" << mono_prior_model_id_
    //           << " lambda_depth=" << opt_params_.lambda_depthanythingv2_
    //           << " lambda_normal=" << opt_params_.lambda_depthanythingv2_normal_
    //           << std::endl;

    // /* Load every ORB-SLAM3 camera, convert to Camera, pre–compute            */
    auto settings = pSLAM->getSettings();   
    cv::Size SLAM_im_size = settings->newImSize();
    UndistortParams undistort_params(
        SLAM_im_size,
        settings->camera1DistortionCoef()
    );
    auto vpCameras = pSLAM->getAtlas()->GetAllCameras();
    for (auto& SLAM_camera : vpCameras) {
        sv::Camera camera;
        camera.camera_id_ = SLAM_camera->GetId();
        if (SLAM_camera->GetType() == ORB_SLAM3::GeometricCamera::CAM_PINHOLE) {
            camera.setModelId(sv::Camera::CameraModelType::PINHOLE);
            float SLAM_fx = SLAM_camera->getParameter(0);
            float SLAM_fy = SLAM_camera->getParameter(1);
            float SLAM_cx = SLAM_camera->getParameter(2);
            float SLAM_cy = SLAM_camera->getParameter(3);

            // Old K, i.e. K in SLAM
            cv::Mat K = (
                cv::Mat_<float>(3, 3)
                    << SLAM_fx, 0.f, SLAM_cx,
                        0.f, SLAM_fy, SLAM_cy,
                        0.f, 0.f, 1.f
            );
            camera.width_ = undistort_params.old_size_.width;
            float x_ratio = static_cast<float>(camera.width_) / undistort_params.old_size_.width;
            camera.height_ = undistort_params.old_size_.height;
            float y_ratio = static_cast<float>(camera.height_) / undistort_params.old_size_.height;

            camera.num_gaus_pyramid_sub_levels_ = num_gaus_pyramid_sub_levels_;
            camera.gaus_pyramid_width_.resize(num_gaus_pyramid_sub_levels_);
            camera.gaus_pyramid_height_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                camera.gaus_pyramid_width_[l] = camera.width_ * this->kf_gaus_pyramid_factors_[l];
                camera.gaus_pyramid_height_[l] = camera.height_ * this->kf_gaus_pyramid_factors_[l];
            }

            camera.params_[0]/*new fx*/= SLAM_fx * x_ratio;
            camera.params_[1]/*new fy*/= SLAM_fy * y_ratio;
            camera.params_[2]/*new cx*/= SLAM_cx * x_ratio;
            camera.params_[3]/*new cy*/= SLAM_cy * y_ratio;

            cv::Mat K_new = (
                cv::Mat_<float>(3, 3)
                    << camera.params_[0], 0.f, camera.params_[2],
                        0.f, camera.params_[1], camera.params_[3],
                        0.f, 0.f, 1.f
            );

            // Undistortion
            if (this->sensor_type_ == MONOCULAR || this->sensor_type_ == RGBD)
                undistort_params.dist_coeff_.copyTo(camera.dist_coeff_);

            camera.initUndistortRectifyMapAndMask(K, SLAM_im_size, K_new, true);

            undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    camera.undistort_mask, device_type_);

            cv::Mat viewer_sub_undistort_mask;
            int viewer_image_height_ = camera.height_ * rendered_image_viewer_scale_;
            int viewer_image_width_ = camera.width_ * rendered_image_viewer_scale_;
            cv::resize(camera.undistort_mask, viewer_sub_undistort_mask,
                    cv::Size(viewer_image_width_, viewer_image_height_));
            viewer_sub_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_sub_undistort_mask, device_type_);

            cv::Mat viewer_main_undistort_mask;
            int viewer_image_height_main_ = camera.height_ * rendered_image_viewer_scale_main_;
            int viewer_image_width_main_ = camera.width_ * rendered_image_viewer_scale_main_;
            cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                    cv::Size(viewer_image_width_main_, viewer_image_height_main_));
            viewer_main_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_main_undistort_mask, device_type_);

            if (this->sensor_type_ == STEREO) {
                camera.stereo_bf_ = stereo_baseline_length_ * camera.params_[0];
                if (this->stereo_Q_.cols != 4) {
                    this->stereo_Q_ = cv::Mat(4, 4, CV_32FC1);
                    this->stereo_Q_.setTo(0.0f);
                    this->stereo_Q_.at<float>(0, 0) = 1.0f;
                    this->stereo_Q_.at<float>(0, 3) = -camera.params_[2];
                    this->stereo_Q_.at<float>(1, 1) = 1.0f;
                    this->stereo_Q_.at<float>(1, 3) = -camera.params_[3];
                    this->stereo_Q_.at<float>(2, 3) = camera.params_[0];
                    this->stereo_Q_.at<float>(3, 2) = 1.0f / stereo_baseline_length_;
                }
            }
        }
        else if (SLAM_camera->GetType() == ORB_SLAM3::GeometricCamera::CAM_FISHEYE) {
            camera.setModelId(sv::Camera::CameraModelType::FISHEYE);
        }
        else {
            camera.setModelId(sv::Camera::CameraModelType::INVALID);
        }

        if (!viewer_camera_id_set_) {
            viewer_camera_id_ = camera.camera_id_;
            viewer_camera_id_set_ = true;
        }
        this->scene_->addCamera(camera);
    }
}

void VoxelMapper::readConfigFromFile(const std::filesystem::path& cfg_path)
{
    cv::FileStorage settings_file(cfg_path.string(), cv::FileStorage::READ);
    if (!settings_file.isOpened()) {
        std::cerr << "[VoxelMapper] Failed to open cfg: " << cfg_path << '\n';
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[VoxelMapper] Reading parameters from " << cfg_path << '\n';
    std::unique_lock<std::mutex> lock(mutex_settings_);

    // Model parameters
     model_params_.sh_degree_ =
         settings_file["Model.sh_degree"].operator int();
     model_params_.resolution_ =
         settings_file["Model.resolution"].operator float();
     model_params_.white_background_ =
         (settings_file["Model.white_background"].operator int()) != 0;
     model_params_.eval_ =
         (settings_file["Model.eval"].operator int()) != 0;

    /* ───────── PIPELINE FLAGS ───────── */
    z_near_ =
         settings_file["Camera.z_near"].operator float();
    {
        cv::FileNode n = settings_file["Monocular.inactive_geo_densify_max_pixel_dist"];
        if (!n.empty()) {
            monocular_inactive_geo_densify_max_pixel_dist_ = n.operator float();
        }
    }
    cull_keyframes_ =
        (settings_file["Mapper.cull_keyframes"].operator int()) != 0;
    min_num_initial_map_kfs_ =
        static_cast<std::size_t>(settings_file["Mapper.min_num_initial_map_kfs"].operator int());
    new_keyframe_times_of_use_ =
        settings_file["Mapper.new_keyframe_times_of_use"].operator int();
    large_rot_th_ =
        settings_file["Mapper.large_rotation_threshold"].operator float();
    large_trans_th_ =
        settings_file["Mapper.large_translation_threshold"].operator float();
    local_BA_increased_times_of_use_ = 
         settings_file["Mapper.local_BA_increased_times_of_use"].operator int();
    loop_closure_increased_times_of_use_ = 
         settings_file["Mapper.loop_closure_increased_times_of_use_"].operator int();

    RGBD_min_depth_ =
        settings_file["RGBD.min_depth"].operator float();
    RGBD_max_depth_ =
        settings_file["RGBD.max_depth"].operator float();

    inactive_geo_densify_ =
        (settings_file["Mapper.inactive_geo_densify"].operator int()) != 0;
    max_depth_cached_ =
        settings_file["Mapper.depth_cache"].operator int();
    {
        cv::FileNode n = settings_file["Mapper.rgbd_fill_render_holes"];
        rgbd_fill_render_holes_ = n.empty() ? false : ((n.operator int()) != 0);
    }
    {
        cv::FileNode n = settings_file["Mapper.rgbd_fill_render_holes_stride"];
        rgbd_fill_render_holes_stride_ = n.empty() ? 2 : std::max(1, static_cast<int>(n));
    }
    {
        cv::FileNode n = settings_file["Mapper.rgbd_fill_render_holes_max_points_per_kf"];
        rgbd_fill_render_holes_max_points_per_kf_ =
            n.empty() ? 20000 : std::max(0, static_cast<int>(n));
    }
    {
        cv::FileNode n = settings_file["Mapper.use_tsdf_mapping"];
        use_tsdf_mapping_ = n.empty() ? false : ((n.operator int()) != 0);
    }
    {
        cv::FileNode n = settings_file["Mapper.use_tsdf_pruning"];
        use_tsdf_pruning_ = n.empty() ? false : ((n.operator int()) != 0);
    }
    {
        cv::FileNode n = settings_file["Mapper.use_tsdf_planning"];
        use_tsdf_planning_ = n.empty() ? false : ((n.operator int()) != 0);
    }
    {
        cv::FileNode n = settings_file["Mapper.tsdf_voxel_size_m"];
        sdf_voxel_size_m_ = n.empty() ? 0.05f : std::max(1.0e-4f, static_cast<float>(n));
    }
    {
        cv::FileNode n = settings_file["Mapper.tsdf_prune_min_weight"];
        tsdf_prune_min_weight_ = n.empty() ? 1.0f : std::max(0.0f, static_cast<float>(n));
    }
    {
        cv::FileNode n = settings_file["Mapper.tsdf_prune_surface_band_vox"];
        tsdf_prune_surface_band_vox_ = n.empty() ? 2.0f : std::max(0.0f, static_cast<float>(n));
    }
    {
        cv::FileNode n = settings_file["Mapper.tsdf_prune_min_valid_corners"];
        tsdf_prune_min_valid_corners_ =
            n.empty() ? 8 : std::max(1, std::min(8, static_cast<int>(n)));
    }
    {
        cv::FileNode n = settings_file["Mapper.tsdf_density_init"];
        tsdf_density_init_ = n.empty() ? false : ((n.operator int()) != 0);
    }
    {
        cv::FileNode n = settings_file["Mapper.tsdf_density_init_min_weight"];
        tsdf_density_init_min_weight_ =
            n.empty() ? tsdf_prune_min_weight_ : std::max(0.0f, static_cast<float>(n));
    }
    {
        cv::FileNode n = settings_file["Mapper.tsdf_density_init_trunc_vox"];
        tsdf_density_init_trunc_vox_ =
            n.empty() ? 4.0f : std::max(1.0e-3f, static_cast<float>(n));
    }
    {
        cv::FileNode n = settings_file["Mapper.tsdf_density_init_bell_a"];
        tsdf_density_init_bell_a_ =
            n.empty() ? 0.1f : std::max(1.0e-4f, static_cast<float>(n));
    }
    {
        cv::FileNode n = settings_file["Mapper.tsdf_density_init_bell_b"];
        tsdf_density_init_bell_b_ =
            n.empty() ? 0.5f : std::clamp(static_cast<float>(n), 1.0e-4f, 0.9999f);
    }
    {
        cv::FileNode n = settings_file["Mapper.tsdf_density_init_alpha_min"];
        tsdf_density_init_alpha_min_ =
            n.empty() ? 0.0002f : std::clamp(static_cast<float>(n), 1.0e-6f, 0.999f);
    }
    {
        cv::FileNode n = settings_file["Mapper.tsdf_density_init_alpha_max"];
        tsdf_density_init_alpha_max_ =
            n.empty() ? 0.5f : std::clamp(static_cast<float>(n), 1.0e-6f, 0.999f);
    }
    if (tsdf_density_init_alpha_max_ < tsdf_density_init_alpha_min_) {
        std::swap(tsdf_density_init_alpha_max_, tsdf_density_init_alpha_min_);
    }
    {
        cv::FileNode n = settings_file["Mapper.depthanything_densify"];
        depthanything_densify_ = n.empty() ? false : ((n.operator int()) != 0);
    }
    {
        cv::FileNode n = settings_file["Mapper.depthanything_densify_stride"];
        depthanything_densify_stride_ = n.empty() ? 8 : std::max(1, static_cast<int>(n));
    }
    {
        cv::FileNode n = settings_file["Mapper.depthanything_densify_min_sparse_anchors"];
        depthanything_densify_min_sparse_anchors_ = n.empty() ? 64 : std::max(2, static_cast<int>(n));
    }
    {
        cv::FileNode n = settings_file["Mapper.depthanything_densify_alignment_mode"];
        depthanything_densify_alignment_mode_ =
            normalizeMonoPriorDensifyAlignmentMode(
                n.empty() ? std::string("orb") : static_cast<std::string>(n));
        if (depthanything_densify_alignment_mode_ != "orb" &&
            depthanything_densify_alignment_mode_ != "rendered" &&
            depthanything_densify_alignment_mode_ != "da_prior") {
            std::cerr << "[MonoPrior] Unknown Mapper.depthanything_densify_alignment_mode='"
                      << depthanything_densify_alignment_mode_
                      << "', falling back to 'orb'. Supported: orb, rendered, da_prior."
                      << std::endl;
            depthanything_densify_alignment_mode_ = "orb";
        }
    }
    {
        cv::FileNode n = settings_file["Mapper.depthanything_da_prior_knn_k"];
        depthanything_da_prior_knn_k_ = n.empty() ? 5 : std::max(2, static_cast<int>(n));
    }
    {
        cv::FileNode n = settings_file["Mapper.depthanything_da_prior_distance_weighting"];
        depthanything_da_prior_distance_weighting_ = n.empty() ? true : ((n.operator int()) != 0);
    }
    {
        cv::FileNode n = settings_file["Mapper.depthanything_da_prior_max_pixel_dist"];
        depthanything_da_prior_max_pixel_dist_ = n.empty() ? 0.0f : std::max(0.0f, static_cast<float>(n));
    }
    {
        cv::FileNode n = settings_file["Mapper.depthanything_fill_holes"];
        depthanything_fill_holes_ = n.empty() ? false : ((n.operator int()) != 0);
    }
    {
        cv::FileNode n = settings_file["Mapper.depthanything_fill_holes_initial_backfill"];
        depthanything_fill_holes_initial_backfill_ =
            n.empty() ? true : ((n.operator int()) != 0);
    }
    {
        cv::FileNode n = settings_file["Mapper.depthanything_fill_holes_warmup_iter"];
        depthanything_fill_holes_warmup_iter_ =
            n.empty() ? 0 : std::max(0, static_cast<int>(n));
    }
    {
        cv::FileNode n = settings_file["Mapper.depthanything_fill_holes_warmup"];
        depthanything_fill_holes_warmup_ =
            n.empty()
                ? (depthanything_fill_holes_warmup_iter_ > 0)
                : ((n.operator int()) != 0);
    }
    {
        cv::FileNode n = settings_file["Mapper.rendered_depth_insert"];
        rendered_depth_insert_ = n.empty() ? false : ((n.operator int()) != 0);
    }
    {
        cv::FileNode n = settings_file["Mapper.rendered_depth_insert_stride"];
        rendered_depth_insert_stride_ = n.empty() ? 4 : std::max(1, static_cast<int>(n));
    }
    {
        cv::FileNode n = settings_file["Mapper.rendered_depth_insert_frontier_radius_px"];
        rendered_depth_insert_frontier_radius_px_ = n.empty() ? 1 : std::max(1, static_cast<int>(n));
    }
    {
        cv::FileNode n = settings_file["Mapper.rendered_depth_insert_max_points_per_kf"];
        rendered_depth_insert_max_points_per_kf_ = n.empty() ? 3000 : std::max(1, static_cast<int>(n));
    }
    {
        cv::FileNode n = settings_file["Mapper.rendered_depth_insert_normal_offset_vox"];
        rendered_depth_insert_normal_offset_vox_ = n.empty() ? 0.5f : std::max(0.0f, static_cast<float>(n));
    }
    {
        cv::FileNode n = settings_file["Mapper.rendered_depth_insert_require_real_adjacency"];
        rendered_depth_insert_require_real_adjacency_ =
            n.empty() ? true : ((n.operator int()) != 0);
    }
    {
        cv::FileNode n = settings_file["Mapper.rendered_depth_insert_adjacency_radius_cells"];
        rendered_depth_insert_adjacency_radius_cells_ =
            n.empty() ? 1 : std::max(1, static_cast<int>(n));
    }

    pipe_params_.convert_SHs_ =
         (settings_file["Pipeline.convert_SHs"].operator int()) != 0;

    do_gaus_pyramid_training_ =
         (settings_file["GausPyramid.do"].operator int()) != 0;
    num_gaus_pyramid_sub_levels_ =
        settings_file["GausPyramid.num_sub_levels"].operator int();
    int sub_level_times_of_use =
        settings_file["GausPyramid.sub_level_times_of_use"].operator int();
    kf_gaus_pyramid_times_of_use_.resize(num_gaus_pyramid_sub_levels_);
    kf_gaus_pyramid_factors_.resize(num_gaus_pyramid_sub_levels_);
    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        kf_gaus_pyramid_times_of_use_[l] = sub_level_times_of_use;
        kf_gaus_pyramid_factors_[l] = std::pow(0.5f, num_gaus_pyramid_sub_levels_ - l);
    }
    
    /* ───────── OPTIMIZATION PARAMETERS ───────── */
    opt_params_.iterations_ =
        settings_file["Optimization.max_num_iterations"].operator int();
    opt_params_.geo_lr_ =
        settings_file["Optimization.geo_lr"].operator float();
    opt_params_.sh0_lr_ =
        settings_file["Optimization.sh0_lr"].operator float();
    opt_params_.shs_lr_ =
        settings_file["Optimization.shs_lr"].operator float();
    {
        cv::FileNode n = settings_file["Optimization.lr_decay_ckpt"];
        opt_params_.lr_decay_ckpt_.clear();
        if (!n.empty())
        {
            if (n.type() == cv::FileNode::SEQ) {
                // YAML: Optimization.lr_decay_ckpt: [5000, 10000, 20000]
                for (auto it = n.begin(); it != n.end(); ++it)
                    opt_params_.lr_decay_ckpt_.push_back((int)*it);
            } else if (n.isInt()) {
                // YAML: Optimization.lr_decay_ckpt: 10000
                opt_params_.lr_decay_ckpt_.push_back((int)n);
            } else if (n.isString()) {
                // YAML: Optimization.lr_decay_ckpt: "5000,10000,20000"
                std::string s = (std::string)n;
                std::stringstream ss(s);
                for (std::string tok; std::getline(ss, tok, ','); ) {
                    if (!tok.empty()) opt_params_.lr_decay_ckpt_.push_back(std::stoi(tok));
                }
            }
        }
    }
    opt_params_.optim_beta1_ =
        settings_file["Optimization.optim_beta1"].operator float();
    opt_params_.optim_beta2_ =
        settings_file["Optimization.optim_beta2"].operator float();
    opt_params_.optim_eps_ =
        settings_file["Optimization.optim_eps"].operator float();
    opt_params_.lr_decay_mult_ =
        settings_file["Optimization.lr_decay_mult"].operator float();

    opt_params_.adapt_from_ =
        settings_file["Optimization.adapt_from"].operator int();
    opt_params_.adapt_every_ =
        settings_file["Optimization.adapt_every"].operator int();
    opt_params_.prune_every_ =
        settings_file["Optimization.prune_every"].operator int();
    opt_params_.subdivide_every_ =
        settings_file["Optimization.subdivide_every"].operator int();
    opt_params_.densify_cooldown_iters_ =
        settings_file["Optimization.densify_cooldown_iters"].operator int();
    {
        cv::FileNode n = settings_file["Optimization.filter_near_voxels"];
        opt_params_.filter_near_voxels_ = n.empty() ? true : ((n.operator int()) != 0);
    }
    opt_params_.filter_far_voxels_on_insert_ =
        (settings_file["Optimization.filter_far_voxels_on_insert"].operator int()) != 0;
    opt_params_.prune_far_voxels_ =
        (settings_file["Optimization.prune_far_voxels"].operator int()) != 0;
    {
        cv::FileNode n = settings_file["Optimization.prune_near_voxels_geometric"];
        opt_params_.prune_near_voxels_geometric_ = n.empty() ? false : ((n.operator int()) != 0);
    }
    {
        cv::FileNode n = settings_file["Optimization.prune_recompute_dense_core"];
        opt_params_.prune_recompute_dense_core_ = n.empty() ? true : ((n.operator int()) != 0);
    }
    opt_params_.prune_recent_unstable_ =
        (settings_file["Optimization.prune_recent_unstable"].operator int()) != 0;
    opt_params_.prune_recent_keyframes_ =
        settings_file["Optimization.prune_recent_keyframes"].operator int();
    opt_params_.prune_recent_min_views_real_ =
        settings_file["Optimization.prune_recent_min_views_real"].operator int();
    opt_params_.prune_recent_min_views_artificial_ =
        settings_file["Optimization.prune_recent_min_views_artificial"].operator int();
    {
        cv::FileNode n = settings_file["Optimization.rendered_depth_candidate_promote_min_support"];
        opt_params_.rendered_depth_candidate_promote_min_support_ =
            n.empty() ? 3 : std::max(1, static_cast<int>(n));
    }
    {
        cv::FileNode n = settings_file["Optimization.rendered_depth_candidate_prune_kf_age"];
        opt_params_.rendered_depth_candidate_prune_kf_age_ =
            n.empty() ? 3 : std::max(1, static_cast<int>(n));
    }
    opt_params_.prune_min_kf_age_ =
        settings_file["Optimization.prune_min_kf_age"].operator int();
    {
        cv::FileNode n = settings_file["Optimization.final_special_prune_enable"];
        opt_params_.final_special_prune_enable_ = n.empty() ? true : ((n.operator int()) != 0);
    }
    opt_params_.prune_until_ =
        settings_file["Optimization.prune_until"].operator int();
    opt_params_.prune_thres_init_ =
        settings_file["Optimization.prune_thres_init"].operator float();
    opt_params_.prune_thres_final_ =
        settings_file["Optimization.prune_thres_final"].operator float();
    opt_params_.prune_thres_final_at_target_ =
        settings_file["Optimization.prune_thres_final_at_target"].operator float();
    opt_params_.prune_thres_init_artificial_ =
        settings_file["Optimization.prune_thres_init_artificial"].operator float();
    opt_params_.prune_thres_final_artificial_ =
        settings_file["Optimization.prune_thres_final_artificial"].operator float();

    opt_params_.subdivide_until_ =
        settings_file["Optimization.subdivide_until"].operator int();
    opt_params_.subdivide_all_until_ =
        settings_file["Optimization.subdivide_all_until"].operator int();
    opt_params_.subdivide_samp_thres_ =
        settings_file["Optimization.subdivide_samp_thres"].operator float();
    opt_params_.subdivide_prop_ =
        settings_file["Optimization.subdivide_prop"].operator float();
    opt_params_.subdivide_samp_thres_at_target_ =
        settings_file["Optimization.subdivide_samp_thres_at_target"].operator float();
    opt_params_.subdivide_prop_at_target_ =
        settings_file["Optimization.subdivide_prop_at_target"].operator float();
    opt_params_.subdivide_max_num_ =
        settings_file["Optimization.subdivide_max_num"].operator int();
    {
        cv::FileNode n = settings_file["Optimization.subdivide_artificial_requires_promotion"];
        opt_params_.subdivide_artificial_requires_promotion_ =
            n.empty() ? false : ((n.operator int()) != 0);
    }
    opt_params_.subdivide_force_to_target_size_ =
        (settings_file["Optimization.subdivide_force_to_target_size"].operator int()) != 0;
    opt_params_.subdivide_target_vox_size_ =
        settings_file["Optimization.subdivide_target_vox_size"].operator float();

    opt_params_.lambda_dssim_ =
        settings_file["Optimization.lambda_dssim"].operator float();
    {
        cv::FileNode n = settings_file["Optimization.use_l1"];
        opt_params_.use_l1_ = n.empty() ? false : (static_cast<int>(n) != 0);
    }
    {
        cv::FileNode n = settings_file["Optimization.use_huber"];
        opt_params_.use_huber_ = n.empty() ? false : (static_cast<int>(n) != 0);
    }
    {
        cv::FileNode n = settings_file["Optimization.huber_thres"];
        opt_params_.huber_thres_ = n.empty() ? 0.03f : static_cast<float>(n);
    }
    if (opt_params_.use_l1_ && opt_params_.use_huber_) {
        std::cout << "[VoxelMapper] Both Optimization.use_l1 and Optimization.use_huber are enabled. "
                  << "Prioritizing L1 to match SVRaster." << std::endl;
    }

    opt_params_.lambda_tv_density_ =
        settings_file["Optimization.lambda_tv_density"].operator float();
    opt_params_.tv_from_ =
        settings_file["Optimization.tv_from"].operator int();
    opt_params_.tv_until_ =
        settings_file["Optimization.tv_until"].operator int();

    opt_params_.ss_aug_max_ = settings_file["Optimization.ss_aug_max"].operator float();
    opt_params_.lambda_R_concen_ = settings_file["Optimization.lambda_R_concen"].operator float();
    opt_params_.lambda_dist_ = settings_file["Optimization.lambda_dist"].operator float();
    {
        cv::FileNode n = settings_file["Optimization.dist_from"];
        if (!n.empty()) {
            opt_params_.dist_from_ = static_cast<int>(n);
        }
    }
    opt_params_.lambda_T_concen_ = settings_file["Optimization.lambda_T_concen"].operator float();
    opt_params_.lambda_T_inside_ = settings_file["Optimization.lambda_T_inside"].operator float();
    opt_params_.lambda_normal_dmean_ = settings_file["Optimization.lambda_normal_dmean"].operator float();
    opt_params_.n_dmean_from_ = settings_file["Optimization.n_dmean_from"].operator int();
    opt_params_.n_dmean_end_ = settings_file["Optimization.n_dmean_end"].operator int();
    opt_params_.n_dmean_ks_ = settings_file["Optimization.n_dmean_ks"].operator int();
    opt_params_.n_dmean_tol_deg_ = settings_file["Optimization.n_dmean_tol_deg"].operator float();
    opt_params_.lambda_ssim_ = settings_file["Optimization.lambda_ssim"].operator float();

    opt_params_.lambda_sparse_depth_ = settings_file["Optimization.lambda_sparse_depth"].operator float();
    opt_params_.sparse_depth_until_ = settings_file["Optimization.sparse_depth_until"].operator int();
    {
        cv::FileNode n = settings_file["Optimization.lambda_depthanythingv2"];
        opt_params_.lambda_depthanythingv2_ = n.empty() ? 0.0f : static_cast<float>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.depthanythingv2_from"];
        opt_params_.depthanythingv2_from_ = n.empty() ? 3000 : static_cast<int>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.depthanythingv2_end"];
        opt_params_.depthanythingv2_end_ = n.empty() ? 20000 : static_cast<int>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.depthanythingv2_end_mult"];
        opt_params_.depthanythingv2_end_mult_ = n.empty() ? 0.1f : static_cast<float>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.lambda_rgbd_depth"];
        opt_params_.lambda_rgbd_depth_ = n.empty() ? 0.0f : static_cast<float>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.rgbd_depth_from"];
        opt_params_.rgbd_depth_from_ = n.empty() ? 3000 : static_cast<int>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.rgbd_depth_end"];
        opt_params_.rgbd_depth_end_ = n.empty() ? 20000 : static_cast<int>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.rgbd_depth_end_mult"];
        opt_params_.rgbd_depth_end_mult_ = n.empty() ? 0.1f : static_cast<float>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.depthanythingv2_overall"];
        opt_params_.depthanythingv2_overall_ =
            n.empty() ? false : (static_cast<int>(n) != 0);
    }
    {
        cv::FileNode n = settings_file["Optimization.depthanythingv2_alpha_adjust"];
        opt_params_.depthanythingv2_alpha_adjust_ =
            n.empty() ? false : (static_cast<int>(n) != 0);
    }
    {
        cv::FileNode n = settings_file["Optimization.enable_da2_uncertainty"];
        opt_params_.enable_da2_uncertainty_ =
            n.empty() ? true : (static_cast<int>(n) != 0);
    }
    {
        cv::FileNode n = settings_file["Optimization.level_uncertainty_from"];
        opt_params_.level_uncertainty_from_ = n.empty() ? 0 : static_cast<int>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.power_level_uncertainty"];
        opt_params_.power_level_uncertainty_ = n.empty() ? 1.0f : static_cast<float>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.lambda_ascending"];
        opt_params_.lambda_ascending_ = n.empty() ? 0.0f : static_cast<float>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.ascending_from"];
        opt_params_.ascending_from_ = n.empty() ? 0 : static_cast<int>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.lambda_rectify"];
        opt_params_.lambda_rectify_ = n.empty() ? 1e-6f : static_cast<float>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.rectifiy_from"];
        opt_params_.rectifiy_from_ = n.empty() ? 0 : static_cast<int>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.lambda_scaling_penalty"];
        opt_params_.lambda_scaling_penalty_ = n.empty() ? 1e-6f : static_cast<float>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.scaling_penalty_from"];
        opt_params_.scaling_penalty_from_ = n.empty() ? 0 : static_cast<int>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.scaling_penalty_end"];
        opt_params_.scaling_penalty_end_ = n.empty() ? 20000 : static_cast<int>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.multi_view_weight_from_iter"];
        opt_params_.multi_view_weight_from_iter_ =
            n.empty() ? 1000000000 : static_cast<int>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.multi_view_interval"];
        opt_params_.multi_view_interval_ = n.empty() ? 1 : static_cast<int>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.multi_view_anneal_scale"];
        opt_params_.multi_view_anneal_scale_ = n.empty() ? 0.0f : static_cast<float>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.multi_view_ncc_weight"];
        opt_params_.multi_view_ncc_weight_ = n.empty() ? 0.05f : static_cast<float>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.multi_view_geo_weight"];
        opt_params_.multi_view_geo_weight_ = n.empty() ? 0.01f : static_cast<float>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.multi_view_patch_size"];
        opt_params_.multi_view_patch_size_ = n.empty() ? 3 : static_cast<int>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.multi_view_sample_num"];
        opt_params_.multi_view_sample_num_ = n.empty() ? 10240000 : static_cast<int>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.multi_view_pixel_noise_th"];
        opt_params_.multi_view_pixel_noise_th_ = n.empty() ? 1.0f : static_cast<float>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.voxel_dropout_min"];
        opt_params_.voxel_dropout_min_ = n.empty() ? 0.5f : static_cast<float>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.lambda_depthanythingv2_normal"];
        opt_params_.lambda_depthanythingv2_normal_ = n.empty() ? 0.0f : static_cast<float>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.depthanythingv2_normal_from"];
        opt_params_.depthanythingv2_normal_from_ = n.empty() ? 3000 : static_cast<int>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.depthanythingv2_normal_end"];
        opt_params_.depthanythingv2_normal_end_ = n.empty() ? 20000 : static_cast<int>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.depthanythingv2_normal_end_mult"];
        opt_params_.depthanythingv2_normal_end_mult_ = n.empty() ? 0.1f : static_cast<float>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.depthanythingv2_normal_ks"];
        opt_params_.depthanythingv2_normal_ks_ =
            n.empty() ? opt_params_.n_dmean_ks_ : static_cast<int>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.depthanythingv2_normal_tol_deg"];
        opt_params_.depthanythingv2_normal_tol_deg_ =
            n.empty() ? opt_params_.n_dmean_tol_deg_ : static_cast<float>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.lambda_rgbd_normal"];
        opt_params_.lambda_rgbd_normal_ = n.empty() ? 0.0f : static_cast<float>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.rgbd_normal_from"];
        opt_params_.rgbd_normal_from_ = n.empty() ? 3000 : static_cast<int>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.rgbd_normal_end"];
        opt_params_.rgbd_normal_end_ = n.empty() ? 20000 : static_cast<int>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.rgbd_normal_end_mult"];
        opt_params_.rgbd_normal_end_mult_ = n.empty() ? 0.1f : static_cast<float>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.rgbd_normal_ks"];
        opt_params_.rgbd_normal_ks_ = n.empty() ? 3 : static_cast<int>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.rgbd_normal_tol_deg"];
        opt_params_.rgbd_normal_tol_deg_ = n.empty() ? 90.0f : static_cast<float>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.mono_prior_model_id"];
        if (n.empty()) {
            n = settings_file["Optimization.depthanythingv2_model_id"];
        }
        mono_prior_model_id_ =
            n.empty() ? std::string("depth-anything/Depth-Anything-V2-Small-hf")
                      : static_cast<std::string>(n);
    }
    {
        cv::FileNode n = settings_file["Optimization.mono_prior_loss_mode"];
        if (n.empty()) {
            n = settings_file["Optimization.depthanythingv2_loss_mode"];
        }
        mono_prior_loss_mode_ =
            normalizeMonoPriorLossMode(n.empty() ? std::string("svraster")
                                                     : static_cast<std::string>(n));
        if (mono_prior_loss_mode_ != "svraster" &&
            mono_prior_loss_mode_ != "aligned" &&
            mono_prior_loss_mode_ != "geosvr") {
            std::cerr << "[MonoPrior] Unknown Optimization.mono_prior_loss_mode='"
                      << mono_prior_loss_mode_
                      << "', falling back to 'svraster'. Supported: svraster, aligned, geosvr."
                      << std::endl;
            mono_prior_loss_mode_ = "svraster";
        }
    }
    if (isMetric3DModelId(mono_prior_model_id_) &&
        mono_prior_loss_mode_ == "svraster") {
        std::cerr << "[Metric3D] Optimization.mono_prior_loss_mode='svraster' is not "
                     "used for Metric3D priors. Falling back to 'aligned'."
                  << std::endl;
            mono_prior_loss_mode_ = "aligned";
    }
    {
        cv::FileNode n = settings_file["Optimization.mono_prior_normal_mode"];
        if (n.empty()) {
            n = settings_file["Optimization.depthanythingv2_normal_mode"];
        }
        if (n.empty()) {
            n = settings_file["Optimization.lambda_depthanythingv2_normal_mode"];
        }
        if (n.empty()) {
            mono_prior_normal_mode_ =
                (mono_prior_loss_mode_ == "geosvr") ? "geosvr" : "aligned";
        } else {
            mono_prior_normal_mode_ =
                normalizeMonoPriorNormalMode(static_cast<std::string>(n));
        }
        if (mono_prior_normal_mode_ != "aligned" &&
            mono_prior_normal_mode_ != "geosvr") {
            std::cerr << "[MonoPrior] Unknown Optimization.mono_prior_normal_mode='"
                      << mono_prior_normal_mode_
                      << "', falling back to 'aligned'. Supported: aligned, geosvr."
                      << std::endl;
            mono_prior_normal_mode_ = "aligned";
        }
    }
    if (opt_params_.multi_view_weight_from_iter_ < 1000000000 &&
        (opt_params_.multi_view_ncc_weight_ > 0.0f ||
         opt_params_.multi_view_geo_weight_ > 0.0f)) {
        std::cerr << "[GeoSVR] Multi-view NCC/geometry regularization is configured, "
                     "but it is not wired into the current Photo-SLAM training loop yet. "
                     "The configs are parsed, but the term is currently inactive."
                  << std::endl;
    }

    /* ───────── LOGGING PARAMETERS ───────── */
    training_report_interval_ =
        settings_file["Record.training_report_interval"].operator int();
    keyframe_record_interval_ =
        settings_file["Record.keyframe_record_interval"].operator int();
    all_keyframes_record_interval_ =
        settings_file["Record.all_keyframes_record_interval"].operator int();
    record_rendered_image_ =
        (settings_file["Record.record_rendered_image"].operator int()) != 0;
    record_ground_truth_image_ =
        (settings_file["Record.record_ground_truth_image"].operator int()) != 0;
    record_loss_image_ =
        (settings_file["Record.record_loss_image"].operator int()) != 0;
    {
        cv::FileNode n = settings_file["Record.enable_rerun"];
        enable_rerun_ = n.empty() ? true : (static_cast<int>(n) != 0);
    }
    {
        cv::FileNode n = settings_file["Record.rerun_final_only"];
        rerun_final_only_ = n.empty() ? false : (static_cast<int>(n) != 0);
    }
    {
        cv::FileNode n = settings_file["Record.rerun_max_keyframes"];
        rerun_max_keyframes_ = n.empty() ? -1 : static_cast<int>(n);
    }
    {
        cv::FileNode n = settings_file["Record.rerun_rendered_depth_insert"];
        rerun_rendered_depth_insert_ = n.empty() ? true : (static_cast<int>(n) != 0);
    }
    {
        cv::FileNode n = settings_file["Record.save_nvblox_mesh_eval"];
        save_nvblox_mesh_eval_ = n.empty() ? false : (static_cast<int>(n) != 0);
    }
    {
        cv::FileNode n = settings_file["Record.load_saved_nvblox_mesh"];
        load_saved_nvblox_mesh_ = n.empty() ? false : (static_cast<int>(n) != 0);
    }
    {
        cv::FileNode n = settings_file["Record.rerun_nvblox_mesh"];
        rerun_nvblox_mesh_ = n.empty() ? false : (static_cast<int>(n) != 0);
    }
    {
        cv::FileNode n = settings_file["Record.saved_nvblox_mesh_path"];
        saved_nvblox_mesh_path_ = n.empty() ? std::string() : static_cast<std::string>(n);
    }
    {
        cv::FileNode n = settings_file["Record.save_rendered_mesh_eval"];
        save_rendered_mesh_eval_ = n.empty() ? true : (static_cast<int>(n) != 0);
    }
    {
        cv::FileNode n = settings_file["Record.rerun_rendered_mesh_eval"];
        rerun_rendered_mesh_eval_ = n.empty() ? false : (static_cast<int>(n) != 0);
    }
    {
        cv::FileNode n = settings_file["Record.rendered_mesh_backend"];
        rendered_mesh_backend_ = n.empty() ? 0 : static_cast<int>(n);
        rendered_mesh_backend_ = std::max(0, std::min(1, rendered_mesh_backend_));
    }
    {
        cv::FileNode n = settings_file["Record.record_depth_metrics"];
        record_depth_metrics_ = n.empty() ? false : (static_cast<int>(n) != 0);
    }
    {
        cv::FileNode n = settings_file["Record.depth_f1_threshold_m"];
        depth_f1_threshold_m_ = n.empty() ? 0.01f : static_cast<float>(n);
    }
    // Viewer Parameters
     rendered_image_viewer_scale_ =
         settings_file["VoxelViewer.image_scale"].operator float();
     rendered_image_viewer_scale_main_ =
         settings_file["VoxelViewer.image_scale_main"].operator float();

    // std::cout << "\n[CFG] Parsed Optimization Parameters:" << std::endl;
    // // std::cout << "  lr:                       " << opt_params_.position_lr_final_ << std::endl;
}

// sv::Camera  →  nvblox::Camera
inline nvblox::Camera toNvbloxCamera(const VoxelKeyframe& kf)
{
    const sv::Camera& cam = kf.cam_;
    // These should be the full undistorted image resolution.
    const int width  = kf.image_width_;
    const int height = kf.image_height_;

    return nvblox::Camera(
        cam.fx(), cam.fy(),   // fu, fv
        cam.cx(), cam.cy(),   // cu, cv
        width, height         // width, height
    );
}

// VoxelKeyframe (Tcw_) → nvblox::Transform T_L_C  (camera → layer/world)
inline nvblox::Transform toNvbloxTransform(VoxelKeyframe& kf)
{
    // Tcw_ is world→cam  ⇒  Twc = Tcw_.inverse() is cam→world
    Sophus::SE3f Tcw_f = kf.getPosef();       // SE3f world→cam
    Sophus::SE3f Twc_f = Tcw_f.inverse();     // SE3f cam→world

    nvblox::Transform T_L_C;
    T_L_C.linear()      = Twc_f.rotationMatrix();
    T_L_C.translation() = Twc_f.translation();
    return T_L_C;  // layer frame L == world frame W
}

// cv::Mat (CV_32FC1, meters) → nvblox::DepthImage (device)
inline void cvDepthToNvbloxDepth(const cv::Mat& depth_meters,
                                 nvblox::DepthImage* depth_img)
{
    using namespace nvblox;
    CHECK(depth_meters.type() == CV_32FC1);

    const int rows = depth_meters.rows;
    const int cols = depth_meters.cols;

    // Allocate / resize on device
    *depth_img = DepthImage(rows, cols, MemoryType::kDevice);

    DepthImage host_img(rows, cols, MemoryType::kHost);
    for (int r = 0; r < rows; ++r) {
        const float* src_row = depth_meters.ptr<float>(r);
        for (int c = 0; c < cols; ++c) {
            host_img(r, c) = src_row[c];
        }
    }

    // Host → device (synchronous)
    depth_img->copyFrom(host_img);
}

// Build an NVBlox camera whose resolution matches the depth image,
// and whose intrinsics are a scaled version of sv::Camera intrinsics.
inline nvblox::Camera makeNvbloxCameraFromDepthAndSvCam(
    const cv::Mat& depth_meters,
    const sv::Camera& cam)
{
    const int width  = depth_meters.cols;
    const int height = depth_meters.rows;

    CHECK(width  > 0);
    CHECK(height > 0);

    // Start from the Photo-SLAM camera intrinsics.
    float fx = cam.fx();
    float fy = cam.fy();
    float cx = cam.cx();
    float cy = cam.cy();

    const float cam_w = static_cast<float>(cam.width());
    const float cam_h = static_cast<float>(cam.height());

    // If Photo-SLAM uses a different internal resolution (e.g. 318x255),
    // rescale intrinsics to match the actual depth image size.
    if (cam_w > 0.0f && cam_h > 0.0f &&
        (static_cast<int>(cam_w) != width ||
         static_cast<int>(cam_h) != height))
    {
        const float scale_x = static_cast<float>(width)  / cam_w;
        const float scale_y = static_cast<float>(height) / cam_h;

        fx *= scale_x;
        fy *= scale_y;
        cx *= scale_x;
        cy *= scale_y;

        // std::cout << "[NVBLOX] Rescaling intrinsics for depth image: "
        //           << "cam(" << cam.width() << "x" << cam.height()
        //           << ") -> img(" << width << "x" << height << ") "
        //           << "scale_x=" << scale_x << " scale_y=" << scale_y
        //           << std::endl;
    }

    return nvblox::Camera(fx, fy, cx, cy, width, height);
}

static bool makeRgbdDepthRerunImage(
    const cv::Mat& depth_in,
    float valid_min_depth,
    float valid_max_depth,
    cv::Mat& depth_viz_bgr)
{
    if (depth_in.empty()) {
        return false;
    }

    cv::Mat depth = depth_in;
    if (depth.channels() > 1) {
        cv::extractChannel(depth, depth, 0);
    }
    if (depth.type() != CV_32FC1) {
        depth.convertTo(depth, CV_32FC1);
    }

    cv::Mat valid_mask(depth.rows, depth.cols, CV_8UC1, cv::Scalar(0));
    double min_depth = std::numeric_limits<double>::infinity();
    double max_depth = -std::numeric_limits<double>::infinity();
    int valid_count = 0;
    for (int y = 0; y < depth.rows; ++y) {
        const float* depth_ptr = depth.ptr<float>(y);
        uint8_t* mask_ptr = valid_mask.ptr<uint8_t>(y);
        for (int x = 0; x < depth.cols; ++x) {
            const float z = depth_ptr[x];
            if (!std::isfinite(z) ||
                z <= valid_min_depth ||
                z >= valid_max_depth) {
                continue;
            }
            mask_ptr[x] = 255;
            min_depth = std::min(min_depth, static_cast<double>(z));
            max_depth = std::max(max_depth, static_cast<double>(z));
            ++valid_count;
        }
    }
    if (valid_count == 0 || !(max_depth > min_depth)) {
        return false;
    }

    cv::Mat gray(depth.rows, depth.cols, CV_8UC1, cv::Scalar(0));
    const float denom = static_cast<float>(std::max(1e-6, max_depth - min_depth));
    for (int y = 0; y < depth.rows; ++y) {
        const float* depth_ptr = depth.ptr<float>(y);
        const uint8_t* mask_ptr = valid_mask.ptr<uint8_t>(y);
        uint8_t* gray_ptr = gray.ptr<uint8_t>(y);
        for (int x = 0; x < depth.cols; ++x) {
            if (mask_ptr[x] == 0) {
                continue;
            }
            const float t = std::clamp(
                (depth_ptr[x] - static_cast<float>(min_depth)) / denom,
                0.0f,
                1.0f);
            gray_ptr[x] = static_cast<uint8_t>(std::round(t * 255.0f));
        }
    }

    cv::applyColorMap(gray, depth_viz_bgr, cv::COLORMAP_JET);
    depth_viz_bgr.setTo(cv::Scalar(0, 0, 0), valid_mask == 0);
    return true;
}

void VoxelMapper::initializeNvbloxMapper()
{
    using namespace nvblox;

    // 1) Configure where TSDF lives (device is standard)
    BlockMemoryPoolParams pool_params;
    pool_params.memory_type = MemoryType::kDevice;

    // 2) Create mapper that integrates TSDF (no freespace / occupancy)
    auto cuda_stream = std::make_shared<CudaStreamOwning>();
    sdf_mapper_ = std::make_shared<Mapper>(
        sdf_voxel_size_m_,
        pool_params,
        ProjectiveLayerType::kTsdf,   // TSDF only
        cuda_stream
    );

    // 3) Set mapper params (defaults + small tweaks)
    MapperParams mapper_params;  // default-constructed

    mapper_params.esdf_integrator_params.esdf_integrator_max_distance_m = 5.0f;
    mapper_params.projective_integrator_params
        .projective_integrator_max_integration_distance_m = 4.0f;

    sdf_mapper_->setMapperParams(mapper_params);

    // Use every depth pixel when deciding which TSDF blocks to allocate.
    // The nvblox default subsamples this raycast, which can miss isolated
    // RGB-D points that we still insert into SVRaster.
    auto& tsdf_int = sdf_mapper_->tsdf_integrator();
    tsdf_int.view_calculator().raycast_subsampling_factor(1);

    // Now override appearance integrator settings explicitly.
    auto& color_int = sdf_mapper_->color_integrator();
    color_int.sphere_tracing_ray_subsampling_factor(1);
    color_int.view_calculator().raycast_subsampling_factor(1);

    // (Optional) if you ever use feature integration:
    // auto& feat_int = sdf_mapper_->feature_integrator();
    // feat_int.sphere_tracing_ray_subsampling_factor(1);
    // feat_int.view_calculator().raycast_subsampling_factor(1);
    // std::cout << "[NVBLOX] tsdf_integrator view_calculator.raycast_subsampling_factor = "
    //           << tsdf_int.view_calculator().raycast_subsampling_factor() << "\n";
    // std::cout << "[NVBLOX] color_integrator sphere_tracing_ray_subsampling_factor = "
    //           << color_int.sphere_tracing_ray_subsampling_factor() << "\n";
    // std::cout << "[NVBLOX] color_integrator view_calculator.raycast_subsampling_factor = "
    //           << color_int.view_calculator().raycast_subsampling_factor() << "\n";

    // --- DEBUG: print TSDF decay free distance and approximate truncation ---
    {
        nvblox::TsdfDecayIntegrator tsdf_decay;
        nvblox::FreespaceIntegrator freespace;
        std::cout << "[TEST] TsdfDecayIntegrator.free_distance_vox() = "
                << tsdf_decay.free_distance_vox() << " vox\n";
        std::cout << "[TEST] FreespaceIntegrator.max_tsdf_distance_for_occupancy_m() = "
                << freespace.max_tsdf_distance_for_occupancy_m() << " m\n";
    // // After sdf_mapper_ is constructed and setMapperParams() has been called:
    //     auto& decay = sdf_mapper_->tsdf_decay_integrator();
    //     std::cout << "[NVBLOX] TsdfDecayIntegrator.free_distance_vox() = "
    //               << decay.free_distance_vox() << " vox\n";

    //     auto& freespace = sdf_mapper_->freespace_integrator();
    //     std::cout << "[NVBLOX] FreespaceIntegrator.max_tsdf_distance_for_occupancy_m() = "
    //               << freespace.max_tsdf_distance_for_occupancy_m() << " m\n";
    }
}

void cvRgbToNvbloxColor(const cv::Mat& rgb_image,
                        nvblox::ColorImage* color_img,
                        nvblox::CudaStream* /*stream*/)
{
    CHECK(color_img != nullptr);
    CHECK(!rgb_image.empty());

    cv::Mat rgb_u8;
    if (rgb_image.type() == CV_8UC3) {
        rgb_u8 = rgb_image;
    } else {
        // Photo-SLAM often stores float images; convert to [0,255]
        rgb_image.convertTo(rgb_u8, CV_8UC3, 255.0);
    }

    const int rows = rgb_u8.rows;
    const int cols = rgb_u8.cols;

    // Host buffer
    nvblox::ColorImage color_host(rows, cols, nvblox::MemoryType::kHost);
    for (int v = 0; v < rows; ++v) {
        const cv::Vec3b* row_ptr = rgb_u8.ptr<cv::Vec3b>(v);
        for (int u = 0; u < cols; ++u) {
            const cv::Vec3b& c = row_ptr[u];
            nvblox::Color col;

            // OpenCV BGR → NVBlox RGB
            col.r() = c[2];
            col.g() = c[1];
            col.b() = c[0];

            color_host(v, u) = col;
        }
    }

    // Allocate device image and copy
    *color_img = nvblox::ColorImage(rows, cols, nvblox::MemoryType::kDevice);
    // copyFrom(ImageBase) – no stream parameter in this overload
    color_img->copyFrom(color_host);
}

void VoxelMapper::integrateKeyframeIntoNvblox(
    VoxelKeyframe& kf,
    const cv::Mat& depth_meters)
{
    if (!sdf_mapper_) {
        return;
    }

    const sv::Camera& cam = kf.cam_;

    // Depth resolution drives NVBlox camera resolution
    const int depth_w = depth_meters.cols;
    const int depth_h = depth_meters.rows;

    if (depth_w <= 0 || depth_h <= 0) {
        std::cout << "[NVBLOX] Warning: depth_meters has invalid size ("
                  << depth_w << "x" << depth_h << "), skipping integration.\n";
        return;
    }

    // Just for debugging, look at both RGB and depth sizes
    const int img_w = kf.img_undist_.cols;
    const int img_h = kf.img_undist_.rows;

    // std::cout << "[NVBLOX] integrate KF: "
    //           << "depth " << depth_w << "x" << depth_h
    //           << "  rgb "   << img_w   << "x" << img_h
    //           << "  cam.width()="  << cam.width()
    //           << " cam.height()=" << cam.height()
    //           << std::endl;

    // Build NVBlox camera whose resolution matches the depth image
    nvblox::Camera nvb_cam = makeNvbloxCameraFromDepthAndSvCam(depth_meters, cam);

    // 3) Pose (camera → world == layer)
    nvblox::Transform T_L_C = toNvbloxTransform(kf);

    // 4) Depth image → NVBlox
    static nvblox::DepthImage depth_img(nvblox::MemoryType::kDevice);
    cvDepthToNvbloxDepth(depth_meters, &depth_img);

    // Integrate TSDF
    sdf_mapper_->integrateDepth(depth_img, T_L_C, nvb_cam);

    // --- Color integration (optional) ---
    const cv::Mat& rgb_undistorted = kf.img_undist_;  // parallel color image

    if (!rgb_undistorted.empty()) {
        // It is nice (but not strictly required) that RGB matches depth size
        if (rgb_undistorted.cols != depth_w || rgb_undistorted.rows != depth_h) {
            std::cout << "[NVBLOX] Warning: RGB size ("
                      << rgb_undistorted.cols << "x" << rgb_undistorted.rows
                      << ") != depth size (" << depth_w << "x" << depth_h
                      << "). Color integration may be inconsistent." << std::endl;
        }

        static nvblox::ColorImage color_img(nvblox::MemoryType::kDevice);
        cvRgbToNvbloxColor(rgb_undistorted, &color_img, /*stream=*/nullptr);

        sdf_mapper_->integrateColor(color_img, T_L_C, nvb_cam);
    } else {
        std::cout << "[NVBLOX] kf.img_undist_ is empty, skipping color integration.\n";
    }
}

bool VoxelMapper::queryEsdfAtWorld(const Eigen::Vector3d& p_W, float& dist_out) const
{
    if (!sdf_mapper_) return false;

    nvblox::EsdfLayer& esdf_layer = sdf_mapper_->esdf_layer();
    if (esdf_layer.size() == 0) return false;

    std::vector<nvblox::Vector3f> positions;
    positions.emplace_back(
        static_cast<float>(p_W.x()),
        static_cast<float>(p_W.y()),
        static_cast<float>(p_W.z())
    );

    std::vector<nvblox::EsdfVoxel> voxels;
    std::vector<bool> success;
    esdf_layer.getVoxels(positions, &voxels, &success);

    if (success.empty() || !success[0]) return false;

    const nvblox::EsdfVoxel& v = voxels[0];

    // If voxel never observed, treat as unknown.
    if (!v.observed) return false;

    const float voxel_size = sdf_mapper_->voxel_size_m();  // Mapper::voxel_size_m()
    const float dist_m = std::sqrt(std::max(0.0f, v.squared_distance_vox)) * voxel_size;

    dist_out = v.is_inside ? -dist_m : dist_m;
    return true;
}

VoxelMapper::TsdfSample VoxelMapper::sampleTsdfAtPointsWorld(const torch::Tensor& pts_world)
{
    TORCH_CHECK(
        pts_world.defined() &&
        pts_world.dim() == 2 &&
        pts_world.size(1) == 3,
        "VoxelMapper::sampleTsdfAtPointsWorld expects pts_world of shape [N,3]");

    const auto N = pts_world.size(0);
    TsdfSample out;

    // Preserve device of input
    const bool input_on_cuda = pts_world.device().is_cuda();
    const auto out_device    = pts_world.device();

    // Default outputs (unknown)
    out.tsdf    = torch::full({N}, std::numeric_limits<float>::quiet_NaN(),
                              torch::TensorOptions().dtype(torch::kFloat32).device(out_device));
    out.weight  = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(out_device));
    out.success = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(out_device));

    // If TSDF is unavailable, return unknowns.
    if (!sdf_mapper_) {
        return out;
    }
    nvblox::TsdfLayer& tsdf_layer = sdf_mapper_->tsdf_layer();
    if (tsdf_layer.size() == 0 || N == 0) {
        return out;
    }

    // Query on CPU (nvblox API expects std::vector positions)
    torch::Tensor pts_cpu = pts_world.to(torch::kCPU).contiguous();
    auto acc_pts = pts_cpu.accessor<float, 2>();

    std::vector<nvblox::Vector3f> positions_L;
    positions_L.reserve(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i) {
        positions_L.emplace_back(acc_pts[i][0], acc_pts[i][1], acc_pts[i][2]);
    }

    std::vector<nvblox::TsdfVoxel> voxels;
    std::vector<bool> success_flags;
    tsdf_layer.getVoxels(positions_L, &voxels, &success_flags);

    const size_t M_vox = voxels.size();
    const size_t M_suc = success_flags.size();

    // Fill CPU buffers first (fast, avoids per-element device writes)
    torch::Tensor tsdf_cpu   = torch::full({N}, std::numeric_limits<float>::quiet_NaN(),
                                           torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    torch::Tensor w_cpu      = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    torch::Tensor succ_cpu   = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));

    auto acc_tsdf = tsdf_cpu.accessor<float, 1>();
    auto acc_w    = w_cpu.accessor<float, 1>();
    auto acc_succ = succ_cpu.accessor<bool, 1>();

    int64_t num_success = 0;
    int64_t num_fail    = 0;

    // Weight statistics over successful reads
    float w_min = std::numeric_limits<float>::infinity();
    float w_max = 0.0f;
    double w_sum = 0.0;

    for (int64_t i = 0; i < N; ++i) {
        const bool have_voxel =
            (static_cast<size_t>(i) < M_vox) &&
            (static_cast<size_t>(i) < M_suc) &&
            success_flags[i];

        if (!have_voxel) {
            ++num_fail;
            continue;
        }

        const auto& v = voxels[i];
        acc_tsdf[i] = v.distance;
        acc_w[i]    = v.weight;
        acc_succ[i] = true;

        ++num_success;

        w_min = std::min(w_min, v.weight);
        w_max = std::max(w_max, v.weight);
        w_sum += static_cast<double>(v.weight);
    }

    // Optional: print once or occasionally
    {
        static int64_t printed = 0;
        if (printed < 5) {  // print first few calls to verify sanity
            ++printed;
            const double suc_ratio = (N > 0) ? (100.0 * double(num_success) / double(N)) : 0.0;
            const double mean_w    = (num_success > 0) ? (w_sum / double(num_success)) : 0.0;

            std::cout << "[TSDF SAMPLE] N=" << N
                      << " success=" << num_success << " (" << suc_ratio << "%)"
                      << " fail=" << num_fail
                      << " w_min=" << (num_success > 0 ? w_min : 0.0f)
                      << " w_max=" << (num_success > 0 ? w_max : 0.0f)
                      << " w_mean=" << mean_w
                      << " tsdf_voxel_size=" << tsdf_layer.voxel_size()
                      << std::endl;
        }
    }

    // Move to original device
    if (input_on_cuda) {
        out.tsdf    = tsdf_cpu.to(out_device);
        out.weight  = w_cpu.to(out_device);
        out.success = succ_cpu.to(out_device);
    } else {
        out.tsdf    = tsdf_cpu;
        out.weight  = w_cpu;
        out.success = succ_cpu;
    }

    return out;
}

torch::Tensor VoxelMapper::computeTsdfDensityInitForGridPoints(
    const torch::Tensor& grid_points_world,
    float ray_interval_m)
{
    TORCH_CHECK(
        grid_points_world.defined() &&
        grid_points_world.dim() == 2 &&
        grid_points_world.size(1) == 3,
        "computeTsdfDensityInitForGridPoints expects grid_points_world [N,3]");

    const int64_t N = grid_points_world.size(0);
    if (N == 0) {
        return torch::empty(
            {0, 1},
            torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(grid_points_world.device()));
    }

    auto opts = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(grid_points_world.device());
    torch::Tensor raw_init = torch::full({N, 1}, -10.0f, opts);

    if (!tsdf_density_init_ ||
        sensor_type_ != RGBD ||
        !sdf_mapper_ ||
        sdf_mapper_->tsdf_layer().size() == 0) {
        return raw_init;
    }

    TsdfSample sample = sampleTsdfAtPointsWorld(grid_points_world);
    torch::Tensor tsdf = sample.tsdf.to(grid_points_world.device()).to(torch::kFloat32);
    torch::Tensor weight = sample.weight.to(grid_points_world.device()).to(torch::kFloat32);
    torch::Tensor success = sample.success.to(grid_points_world.device()).to(torch::kBool);

    const float min_weight = std::max(0.0f, tsdf_density_init_min_weight_);
    torch::Tensor valid =
        (success & (weight >= min_weight) & torch::isfinite(tsdf)).to(torch::kBool);
    const int64_t valid_count = valid.sum().item<int64_t>();
    if (valid_count == 0) {
        return raw_init;
    }

    const float tsdf_vox_size = std::max(
        1.0e-6f,
        static_cast<float>(sdf_mapper_->tsdf_layer().voxel_size()));
    const float trunc_m =
        std::max(1.0e-6f, tsdf_density_init_trunc_vox_ * tsdf_vox_size);

    const float a = std::max(1.0e-4f, tsdf_density_init_bell_a_);
    const float b = std::clamp(tsdf_density_init_bell_b_, 1.0e-4f, 0.9999f);
    const float u = (2.0f - b + 2.0f * std::sqrt(std::max(0.0f, 1.0f - b))) / b;
    const float sharpness = std::log(u) / a;

    torch::Tensor F = torch::clamp(tsdf / trunc_m, -1.0f, 1.0f);
    torch::Tensor sigmoid = torch::sigmoid(-sharpness * F);
    torch::Tensor alpha_unit = 4.0f * sigmoid * (1.0f - sigmoid);

    const float alpha_min = std::clamp(tsdf_density_init_alpha_min_, 1.0e-6f, 0.999f);
    const float alpha_max = std::clamp(tsdf_density_init_alpha_max_, alpha_min, 0.999f);
    torch::Tensor alpha = alpha_min + (alpha_max - alpha_min) * alpha_unit;
    alpha = torch::clamp(alpha, 1.0e-6f, 0.999f);

    constexpr float kSvrasterStepSzScale = 100.0f;
    const float interval = std::max(1.0e-6f, ray_interval_m);
    torch::Tensor density =
        -torch::log(torch::clamp(1.0f - alpha, 1.0e-6f, 1.0f)) /
        (kSvrasterStepSzScale * interval);

    torch::Tensor density_safe = torch::clamp(density, 1.0e-12f);
    torch::Tensor raw_low =
        (torch::log(density_safe) + 0.904689820196f) / 0.909090909091f;
    torch::Tensor raw = torch::where(density > 1.1f, density, raw_low);
    raw = torch::clamp(raw, -20.0f, 20.0f).view({N, 1});

    raw_init = torch::where(valid.view({N, 1}), raw, raw_init).contiguous();

    {
        static int printed = 0;
        if (printed < 5) {
            ++printed;
            torch::Tensor raw_valid = raw.index({valid});
            torch::Tensor alpha_valid = alpha.index({valid});
            const float raw_min = raw_valid.numel() > 0 ? raw_valid.min().item<float>() : -10.0f;
            const float raw_max = raw_valid.numel() > 0 ? raw_valid.max().item<float>() : -10.0f;
            const float alpha_mean = alpha_valid.numel() > 0 ? alpha_valid.mean().item<float>() : 0.0f;
            std::cout << "[TSDF DENSITY INIT] N=" << N
                      << " valid=" << valid_count
                      << " trunc_m=" << trunc_m
                      << " bell_a=" << a
                      << " bell_b=" << b
                      << " sharpness=" << sharpness
                      << " alpha_min=" << alpha_min
                      << " alpha_max=" << alpha_max
                      << " alpha_mean_valid=" << alpha_mean
                      << " raw_valid_min=" << raw_min
                      << " raw_valid_max=" << raw_max
                      << "\n";
        }
    }

    return raw_init;
}

static torch::Tensor computeSvrasterGridPointsWorld(
    const torch::Tensor& grid_pts_key,   // [M,3] int64
    const torch::Tensor& scene_center,   // [3] float
    const torch::Tensor& scene_extent,   // [1] float
    int max_num_levels)                  // e.g. 16
{
    TORCH_CHECK(grid_pts_key.defined() && grid_pts_key.dim() == 2 && grid_pts_key.size(1) == 3,
                "grid_pts_key must be [M,3] int64");
    TORCH_CHECK(scene_center.defined() && scene_center.numel() == 3,
                "scene_center must be [3]");
    TORCH_CHECK(scene_extent.defined() && scene_extent.numel() == 1,
                "scene_extent must be [1]");

    auto dev = scene_center.device();
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(dev);

    // scene_min = scene_center - 0.5 * scene_extent
    torch::Tensor scene_min = scene_center.to(opts_f).view({3}) - 0.5f * scene_extent.to(opts_f).view({1});

    // finest_vox_size = scene_extent * 2^{-MAX_NUM_LEVELS}
    // (grid_pts_key are integer coords on the finest grid; corners, not centers)
    const float scale = std::ldexp(1.0f, -max_num_levels);  // 2^{-L}
    torch::Tensor finest_vox = scene_extent.to(opts_f) * scale; // [1]

    // grid_xyz = scene_min + grid_pts_key * finest_vox_size
    torch::Tensor grid_xyz =
        scene_min.view({1,3}) + grid_pts_key.to(dev).to(torch::kFloat32) * finest_vox.view({1,1});

    return grid_xyz.contiguous(); // [M,3]
}

// Grid-point-based: gather 8 corner TSDFs for each SVRaster voxel using vox_key_
VoxelMapper::TsdfCornerSample VoxelMapper::sampleTsdfAtSvrasterGridCornersWorld()
{
    TORCH_CHECK(voxel_model_ != nullptr, "voxel_model_ is null");

    torch::Tensor grid_key = voxel_model_->gridPtsKey(); // [M,3] int64
    torch::Tensor vox_key  = voxel_model_->voxKey();     // [N,8] int64

    TORCH_CHECK(grid_key.defined() && grid_key.dim() == 2 && grid_key.size(1) == 3,
                "gridPtsKey must be [M,3]");
    TORCH_CHECK(vox_key.defined() && vox_key.dim() == 2 && vox_key.size(1) == 8,
                "voxKey must be [N,8]");

    const int64_t M = grid_key.size(0);
    const int64_t N = vox_key.size(0);

    TsdfCornerSample out;
    auto dev = voxel_model_->voxCenter().device();

    // Default (unknown)
    out.tsdf    = torch::full({N, 8}, std::numeric_limits<float>::quiet_NaN(),
                              torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    out.weight  = torch::zeros({N, 8}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    out.success = torch::zeros({N, 8}, torch::TensorOptions().dtype(torch::kBool).device(dev));
    out.points_world = torch::Tensor();

    if (!sdf_mapper_ || sdf_mapper_->tsdf_layer().size() == 0 || N == 0 || M == 0) {
        return out;
    }

    // 1) grid_key -> world xyz
    torch::Tensor scene_center = voxel_model_->SceneCenter(); // [3]
    torch::Tensor scene_extent = voxel_model_->SceneExtent(); // [1]
    const int Lmax = voxel_model_->maxNumLevels();

    torch::Tensor grid_xyz = computeSvrasterGridPointsWorld(grid_key, scene_center, scene_extent, Lmax);
    if (grid_xyz.device() != dev) grid_xyz = grid_xyz.to(dev);

    // 2) Sample TSDF at all grid points
    TsdfSample g = sampleTsdfAtPointsWorld(grid_xyz); // g.tsdf,g.weight,g.success are [M]

    // 3) Gather per-voxel corners using vox_key
    // Flatten indices: [N,8] -> [N*8]
    torch::Tensor idx = vox_key.to(dev).to(torch::kLong).reshape({-1}); // [N*8]
    TORCH_CHECK(idx.numel() == N * 8, "vox_key reshape mismatch");

    // // Defensive bounds check (optional; can be expensive, use only while debugging)
    // TORCH_CHECK((idx >= 0).all().item<bool>() && (idx < M).all().item<bool>(),
    //             "vox_key contains out-of-range indices (must be in [0, M))");

    torch::Tensor tsdf8 = g.tsdf.index_select(0, idx).view({N, 8}).contiguous();
    torch::Tensor w8    = g.weight.index_select(0, idx).view({N, 8}).contiguous();
    torch::Tensor ok8   = g.success.index_select(0, idx).view({N, 8}).contiguous();
    torch::Tensor points8 =
        grid_xyz.index_select(0, idx).view({N, 8, 3}).contiguous();

    out.tsdf    = tsdf8;
    out.weight  = w8;
    out.success = ok8;
    out.points_world = points8;
    return out;
}

VoxelMapper::TsdfCornerSample VoxelMapper::sampleTsdfAtVoxelCornersWorld(
    const torch::Tensor& centers_world,
    const torch::Tensor& sizes_world)
{
    using torch::indexing::Slice;

    TORCH_CHECK(centers_world.defined() && centers_world.dim() == 2 && centers_world.size(1) == 3,
                "sampleTsdfAtVoxelCornersWorld expects centers_world [N,3]");
    TORCH_CHECK(sizes_world.defined(),
                "sampleTsdfAtVoxelCornersWorld expects sizes_world defined");

    const int64_t N = centers_world.size(0);
    TsdfCornerSample out;

    const auto dev = centers_world.device();

    // Normalize sizes to [N,1] on same device
    torch::Tensor sizes = sizes_world;
    if (sizes.dim() == 1) {
        TORCH_CHECK(sizes.size(0) == N, "sizes_world [N] mismatch with centers_world");
        sizes = sizes.view({N, 1});
    } else if (sizes.dim() == 2) {
        TORCH_CHECK(sizes.size(0) == N, "sizes_world [N,1] mismatch with centers_world");
        TORCH_CHECK(sizes.size(1) == 1, "sizes_world must have shape [N,1] if 2D");
    } else {
        TORCH_CHECK(false, "sizes_world must be [N] or [N,1]");
    }
    sizes = sizes.to(dev).contiguous();

    // Default outputs (unknown)
    out.tsdf    = torch::full({N, 8}, std::numeric_limits<float>::quiet_NaN(),
                              torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    out.weight  = torch::zeros({N, 8}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    out.success = torch::zeros({N, 8}, torch::TensorOptions().dtype(torch::kBool).device(dev));
    out.points_world = torch::Tensor();

    if (N == 0) return out;
    if (!sdf_mapper_) return out;
    nvblox::TsdfLayer& tsdf_layer = sdf_mapper_->tsdf_layer();
    if (tsdf_layer.size() == 0) return out;

    // Corner offsets (8,3) with all sign combinations.
    // Construct once, keep on CUDA (or whatever device centers are on).
    static torch::Tensor offsets_cache;
    static torch::Device cached_dev = torch::kCPU;

    if (!offsets_cache.defined() || cached_dev != dev) {
        // CPU literal then move
        offsets_cache = torch::tensor(
            {{-1.f, -1.f, -1.f},
             {-1.f, -1.f,  1.f},
             {-1.f,  1.f, -1.f},
             {-1.f,  1.f,  1.f},
             { 1.f, -1.f, -1.f},
             { 1.f, -1.f,  1.f},
             { 1.f,  1.f, -1.f},
             { 1.f,  1.f,  1.f}},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).to(dev).contiguous();
        cached_dev = dev;
    }
    const torch::Tensor offsets = offsets_cache; // [8,3] on dev

    // Build corners: [N,8,3] = centers[:,None,:] + 0.5*sizes[:,None,:]*offsets[None,:,:]
    torch::Tensor half = 0.5f * sizes;                 // [N,1]
    torch::Tensor corners = centers_world.contiguous().view({N, 1, 3})
                          + half.view({N, 1, 1}) * offsets.view({1, 8, 3}); // [N,8,3]
    torch::Tensor corners_flat = corners.view({N * 8, 3}).contiguous();     // [N*8,3]
    out.points_world = corners.contiguous();

    // Use your existing point sampler (returns [N*8])
    TsdfSample s = sampleTsdfAtPointsWorld(corners_flat);

    // Reshape back to [N,8]
    TORCH_CHECK(s.tsdf.defined() && s.tsdf.numel() == N * 8,
                "sampleTsdfAtPointsWorld returned unexpected tsdf size");
    TORCH_CHECK(s.weight.defined() && s.weight.numel() == N * 8,
                "sampleTsdfAtPointsWorld returned unexpected weight size");
    TORCH_CHECK(s.success.defined() && s.success.numel() == N * 8,
                "sampleTsdfAtPointsWorld returned unexpected success size");

    out.tsdf    = s.tsdf.view({N, 8}).contiguous();
    out.weight  = s.weight.view({N, 8}).contiguous();
    out.success = s.success.view({N, 8}).contiguous();

    // ------------------------------------------------------------
    // One-time sanity check: do corners match center ± size/2 ?
    // ------------------------------------------------------------
    {
        static bool printed_once = false;
        if (!printed_once) {
            printed_once = true;

            const int64_t i0 = 0;
            if (corners.defined() && corners.dim() == 3 && corners.size(0) > i0) {
                // Bring to CPU for printing
                auto c0_cpu = centers_world.index({i0}).to(torch::kCPU).contiguous();      // [3]
                auto s0_cpu = sizes_world.index({i0}).view({1}).to(torch::kCPU).contiguous(); // [1]
                auto crn_cpu = corners.index({i0}).to(torch::kCPU).contiguous();          // [8,3]

                const float cx = c0_cpu[0].item<float>();
                const float cy = c0_cpu[1].item<float>();
                const float cz = c0_cpu[2].item<float>();
                const float s  = s0_cpu[0].item<float>();
                const float h  = 0.5f * s;

                // Expected bounds
                const float ex_min_x = cx - h, ex_max_x = cx + h;
                const float ex_min_y = cy - h, ex_max_y = cy + h;
                const float ex_min_z = cz - h, ex_max_z = cz + h;

                // Observed bounds from corners
                auto min_xyz = std::get<0>(crn_cpu.min(/*dim=*/0, /*keepdim=*/false)); // [3]
                auto max_xyz = std::get<0>(crn_cpu.max(/*dim=*/0, /*keepdim=*/false)); // [3]

                const float ob_min_x = min_xyz[0].item<float>();
                const float ob_min_y = min_xyz[1].item<float>();
                const float ob_min_z = min_xyz[2].item<float>();

                const float ob_max_x = max_xyz[0].item<float>();
                const float ob_max_y = max_xyz[1].item<float>();
                const float ob_max_z = max_xyz[2].item<float>();

                auto absf = [](float v){ return v < 0.f ? -v : v; };
                const float eps = 1e-5f;

                std::cout << "[TSDF CORNER SANITY] i0=" << i0 << "\n";
                std::cout << "  center = [" << cx << ", " << cy << ", " << cz << "]\n";
                std::cout << "  size   = " << s << "  half=" << h << "\n";

                std::cout << "  expected min = [" << ex_min_x << ", " << ex_min_y << ", " << ex_min_z << "]\n";
                std::cout << "  expected max = [" << ex_max_x << ", " << ex_max_y << ", " << ex_max_z << "]\n";

                std::cout << "  observed min = [" << ob_min_x << ", " << ob_min_y << ", " << ob_min_z << "]\n";
                std::cout << "  observed max = [" << ob_max_x << ", " << ob_max_y << ", " << ob_max_z << "]\n";

                const bool ok_x = (absf(ob_min_x - ex_min_x) < eps) && (absf(ob_max_x - ex_max_x) < eps);
                const bool ok_y = (absf(ob_min_y - ex_min_y) < eps) && (absf(ob_max_y - ex_max_y) < eps);
                const bool ok_z = (absf(ob_min_z - ex_min_z) < eps) && (absf(ob_max_z - ex_max_z) < eps);

                std::cout << "  axis check: x=" << (ok_x ? "OK" : "FAIL")
                        << " y=" << (ok_y ? "OK" : "FAIL")
                        << " z=" << (ok_z ? "OK" : "FAIL")
                        << " (eps=" << eps << ")\n";

                std::cout << "  corners[8,3] =\n" << crn_cpu << "\n";
            } else {
                std::cout << "[TSDF CORNER SANITY] corners tensor not ready or empty.\n";
            }
        }
    }

    return out;
}

static torch::Tensor normalizeBoolMaskOrZeros(
    torch::Tensor mask,
    const int64_t N,
    const torch::Device& device)
{
    if (!mask.defined() || mask.numel() != N) {
        return torch::zeros(
            {N},
            torch::TensorOptions().dtype(torch::kBool).device(device));
    }
    if (mask.dim() == 2 && mask.size(1) == 1) {
        mask = mask.squeeze(1);
    } else if (mask.dim() != 1) {
        mask = mask.reshape({N});
    }
    return mask.to(device).to(torch::kBool).contiguous();
}

void VoxelMapper::recordTsdfPruneAblation(
    const torch::Tensor& tsdf_prune_mask,
    const std::string& tag)
{
    if (!voxel_model_ || !tsdf_prune_mask.defined() || tsdf_prune_mask.numel() <= 0) {
        return;
    }

    const int64_t N = tsdf_prune_mask.numel();
    const torch::Device device = tsdf_prune_mask.device();
    torch::Tensor tsdf_mask =
        normalizeBoolMaskOrZeros(tsdf_prune_mask, N, device);

    torch::Tensor rgbd_points_mask =
        normalizeBoolMaskOrZeros(voxel_model_->orbVoxelMask(), N, device);
    torch::Tensor inactive_geo_mask =
        normalizeBoolMaskOrZeros(voxel_model_->inactiveGeoVoxelMask(), N, device);
    torch::Tensor rgbd_fill_mask =
        normalizeBoolMaskOrZeros(voxel_model_->rgbdFillRenderHolesVoxelMask(), N, device);

    // Source masks should be exclusive, but keep this robust for older runs.
    rgbd_points_mask =
        (rgbd_points_mask & (~inactive_geo_mask) & (~rgbd_fill_mask)).to(torch::kBool);

    const int64_t rgbd_points_live = rgbd_points_mask.sum().item<int64_t>();
    const int64_t inactive_geo_live = inactive_geo_mask.sum().item<int64_t>();
    const int64_t rgbd_fill_live = rgbd_fill_mask.sum().item<int64_t>();

    const int64_t rgbd_points_pruned =
        (tsdf_mask & rgbd_points_mask).sum().item<int64_t>();
    const int64_t inactive_geo_pruned =
        (tsdf_mask & inactive_geo_mask).sum().item<int64_t>();
    const int64_t rgbd_fill_pruned =
        (tsdf_mask & rgbd_fill_mask).sum().item<int64_t>();

    tsdf_ablation_rgbd_points_live_last_ = rgbd_points_live;
    tsdf_ablation_inactive_geo_live_last_ = inactive_geo_live;
    tsdf_ablation_rgbd_fill_live_last_ = rgbd_fill_live;
    tsdf_ablation_rgbd_points_pruned_by_sdf_ += rgbd_points_pruned;
    tsdf_ablation_inactive_geo_pruned_by_sdf_ += inactive_geo_pruned;
    tsdf_ablation_rgbd_fill_pruned_by_sdf_ += rgbd_fill_pruned;
    ++tsdf_ablation_sdf_prune_passes_;

    std::cout << "[TSDF ABLATION] iter=" << getIteration()
              << " tag=" << tag
              << " rgbd_points live=" << rgbd_points_live
              << " sdf_pruned=" << rgbd_points_pruned
              << " cum_sdf_pruned=" << tsdf_ablation_rgbd_points_pruned_by_sdf_
              << " inactive_geo live=" << inactive_geo_live
              << " sdf_pruned=" << inactive_geo_pruned
              << " cum_sdf_pruned=" << tsdf_ablation_inactive_geo_pruned_by_sdf_
              << " rgbd_fill_render_holes live=" << rgbd_fill_live
              << " sdf_pruned=" << rgbd_fill_pruned
              << " cum_sdf_pruned=" << tsdf_ablation_rgbd_fill_pruned_by_sdf_
              << std::endl;
}

void VoxelMapper::printTsdfPruneAblationSummary(const std::string& tag) const
{
    std::cout << "[TSDF ABLATION SUMMARY] tag=" << tag
              << " sdf_prune_passes=" << tsdf_ablation_sdf_prune_passes_
              << "\n"
              << "  rgbd_points: inserted_roots=" << tsdf_ablation_rgbd_points_created_
              << " lineage_created=" << tsdf_ablation_rgbd_points_lineage_created_
              << " pruned_by_sdf=" << tsdf_ablation_rgbd_points_pruned_by_sdf_
              << "\n"
              << "  inactive_geo_densify: inserted_roots=" << tsdf_ablation_inactive_geo_created_
              << " lineage_created=" << tsdf_ablation_inactive_geo_lineage_created_
              << " pruned_by_sdf=" << tsdf_ablation_inactive_geo_pruned_by_sdf_
              << "\n"
              << "  rgbd_fill_render_holes: inserted_roots=" << tsdf_ablation_rgbd_fill_created_
              << " lineage_created=" << tsdf_ablation_rgbd_fill_lineage_created_
              << " pruned_by_sdf=" << tsdf_ablation_rgbd_fill_pruned_by_sdf_
              << std::endl;
}

static inline torch::Tensor make_points_tensor_cpu_f32(
    const std::vector<Eigen::Vector3f>& pts
) {
    if (pts.empty()) {
        return torch::Tensor();
    }
    auto out = torch::empty(
        {(long)pts.size(), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
    );
    auto acc = out.accessor<float, 2>();
    for (int i = 0; i < (int)pts.size(); ++i) {
        acc[i][0] = pts[i].x();
        acc[i][1] = pts[i].y();
        acc[i][2] = pts[i].z();
    }
    return out;
}

static inline torch::Tensor make_single_point_cpu_f32(const Eigen::Vector3d& p) {
    return torch::tensor(
        {(float)p.x(), (float)p.y(), (float)p.z()},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
    ).view({1, 3});
}

static inline torch::Tensor color_u8(uint8_t r, uint8_t g, uint8_t b) {
    return torch::tensor({r, g, b}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
}

static torch::Tensor normalizeRerunPointColors(torch::Tensor colors)
{
    if (!colors.defined() || colors.numel() == 0) {
        return colors;
    }
    if (colors.max().item<float>() > 1.5f) {
        colors = colors / 255.0f;
    }
    return colors.clamp(0.0f, 1.0f).contiguous();
}

void VoxelMapper::appendAndLogOrbRawMapPcdToRerun(
    const std::map<point3D_id_t, Point3D>& pcd,
    int iteration)
{
    if (pcd.empty()) {
        return;
    }

    std::vector<float> pts;
    std::vector<float> cols;
    pts.reserve(pcd.size() * 3);
    cols.reserve(pcd.size() * 3);
    for (const auto& kv : pcd) {
        const auto& P = kv.second;
        pts.push_back(static_cast<float>(P.xyz_(0)));
        pts.push_back(static_cast<float>(P.xyz_(1)));
        pts.push_back(static_cast<float>(P.xyz_(2)));
        cols.push_back(static_cast<float>(P.color_(0)));
        cols.push_back(static_cast<float>(P.color_(1)));
        cols.push_back(static_cast<float>(P.color_(2)));
    }

    auto points = torch::from_blob(
        pts.data(),
        {static_cast<int64_t>(pcd.size()), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
    auto colors = torch::from_blob(
        cols.data(),
        {static_cast<int64_t>(pcd.size()), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
    colors = normalizeRerunPointColors(colors);

    if (!orb_raw_pcd_points_accum_cpu_.defined() || orb_raw_pcd_points_accum_cpu_.numel() == 0) {
        orb_raw_pcd_points_accum_cpu_ = points.contiguous();
        orb_raw_pcd_colors_accum_cpu_ = colors.contiguous();
    } else {
        orb_raw_pcd_points_accum_cpu_ =
            torch::cat({orb_raw_pcd_points_accum_cpu_, points.contiguous()}, 0).contiguous();
        orb_raw_pcd_colors_accum_cpu_ =
            torch::cat({orb_raw_pcd_colors_accum_cpu_, colors.contiguous()}, 0).contiguous();
    }

    sv::RerunVisualizerBridge::instance().visualizePoints3D(
        orb_raw_pcd_points_accum_cpu_,
        orb_raw_pcd_colors_accum_cpu_,
        iteration,
        "world/orb/raw_pcd",
        0.015f);
    // std::cout << "[rerun/orb] appended_raw_pcd_points=" << pcd.size()
    //           << " total_raw_pcd_points=" << orb_raw_pcd_points_accum_cpu_.size(0)
    //           << " entity=world/orb/raw_pcd"
    //           << " iter=" << iteration
    //           << std::endl;
}

void VoxelMapper::appendAndLogOrbRawPointBatchToRerun(
    const std::vector<float>& points_flat,
    const std::vector<float>& colors_flat,
    int iteration)
{
    const int64_t n_points = static_cast<int64_t>(points_flat.size() / 3);
    if (n_points <= 0 || colors_flat.size() < static_cast<size_t>(3 * n_points)) {
        return;
    }

    auto points = torch::from_blob(
        const_cast<float*>(points_flat.data()),
        {n_points, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
    auto colors = torch::from_blob(
        const_cast<float*>(colors_flat.data()),
        {n_points, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
    colors = normalizeRerunPointColors(colors);

    if (!orb_raw_pcd_points_accum_cpu_.defined() || orb_raw_pcd_points_accum_cpu_.numel() == 0) {
        orb_raw_pcd_points_accum_cpu_ = points.contiguous();
        orb_raw_pcd_colors_accum_cpu_ = colors.contiguous();
    } else {
        orb_raw_pcd_points_accum_cpu_ =
            torch::cat({orb_raw_pcd_points_accum_cpu_, points.contiguous()}, 0).contiguous();
        orb_raw_pcd_colors_accum_cpu_ =
            torch::cat({orb_raw_pcd_colors_accum_cpu_, colors.contiguous()}, 0).contiguous();
    }

    sv::RerunVisualizerBridge::instance().visualizePoints3D(
        orb_raw_pcd_points_accum_cpu_,
        orb_raw_pcd_colors_accum_cpu_,
        iteration,
        "world/orb/raw_pcd",
        0.015f);
    // std::cout << "[rerun/orb] appended_raw_pcd_points=" << n_points
    //           << " total_raw_pcd_points=" << orb_raw_pcd_points_accum_cpu_.size(0)
    //           << " entity=world/orb/raw_pcd"
    //           << " iter=" << iteration
    //           << std::endl;
}

void VoxelMapper::run()
{
    /* expose our helper scripts to the embedded Python side */
    py::gil_scoped_acquire gil;
    py::module_::import("sys").attr("path").attr("insert")(0, "../scripts_voxel");

    sv::RerunVisualizerBridge::instance().setEnabled(enable_rerun_);

    // Initialize Rerun in "headless" mode (no viewer window).
    sv::RerunVisualizerBridge::instance().init(
        "PhotoSLAM-SVRaster",
        /*spawn_viewer=*/false
    );

    if (enable_rerun_ && rerun_nvblox_mesh_ && load_saved_nvblox_mesh_) {
        if (!saved_nvblox_mesh_path_.empty() &&
            std::filesystem::exists(saved_nvblox_mesh_path_)) {
            sv::RerunVisualizerBridge::instance().visualizeNvbloxPlyMesh(
                saved_nvblox_mesh_path_,
                0,
                "world/nvblox_mesh/reference");
            std::cout << "[NVBLOX] loaded saved reference mesh into Rerun: "
                      << saved_nvblox_mesh_path_ << "\n";
        } else {
            std::cerr << "[NVBLOX] saved reference mesh not found: "
                      << saved_nvblox_mesh_path_ << "\n";
        }
    }

    const bool need_nvblox =
        use_tsdf_mapping_ ||
        use_tsdf_pruning_ ||
        tsdf_density_init_ ||
        use_tsdf_planning_ ||
        save_nvblox_mesh_eval_ ||
        (enable_rerun_ && rerun_nvblox_mesh_ && !load_saved_nvblox_mesh_);
    if (sensor_type_ == RGBD && need_nvblox)
    {
        initializeNvbloxMapper();
    }
    // First loop: Initial gaussian mapping
    while (!isStopped())
    {
        // Check conditions for initial mapping
        if (hasMetInitialMappingConditions())
        {
            mpSLAM->getAtlas()->clearMappingOperation();

            // Pull sparse SLAM map (get keyframes and map points)
            auto pMap = mpSLAM->getAtlas()->GetCurrentMap();
            std::vector<ORB_SLAM3::KeyFrame*> vKFs;
            std::vector<ORB_SLAM3::MapPoint*> vMPs;
            {
                std::unique_lock<std::mutex> lock_map(pMap->mMutexMapUpdate);
                vKFs = pMap->GetAllKeyFrames();
                vMPs = pMap->GetAllMapPoints();
                for (const auto& pMP : vMPs)
                {
                     Point3D point3D;
                     auto pos = pMP->GetWorldPos();
                     point3D.xyz_(0) = pos.x();
                     point3D.xyz_(1) = pos.y();
                     point3D.xyz_(2) = pos.z();
                     auto color = pMP->GetColorRGB();
                     point3D.color_(0) = color(0);
                     point3D.color_(1) = color(1);
                     point3D.color_(2) = color(2);
                     scene_->cachePoint3D(pMP->mnId, point3D);
                 }
                // B) Create VoxelKeyframes from each SLAM KeyFrame
                for (const auto& pKF : vKFs)
                {
                    std::shared_ptr<VoxelKeyframe> new_kf = std::make_shared<VoxelKeyframe>(pKF->mnId, getIteration());
                    new_kf->znear_ = z_near_;
                    // Pose
                    auto pose = pKF->GetPose();
                    new_kf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>()
                    );
                    cv::Mat imgRGB_undistorted, imgAux_undistorted;
                    // Camera
                    sv::Camera& camera = scene_->cameras_.at(pKF->mpCamera->GetId());
                    new_kf->setCameraParams(camera);

                    // Image (left if STEREO)
                    cv::Mat imgRGB = pKF->imgLeftRGB;
                    // camera.undistortImage(imgRGB, imgRGB_undistorted);
                    if (this->sensor_type_ == STEREO)
                            imgRGB_undistorted = imgRGB;
                        else
                            camera.undistortImage(imgRGB, imgRGB_undistorted);
                    // Auxiliary Image
                    cv::Mat imgAux = pKF->imgAuxiliary;
                    if (this->sensor_type_ == RGBD)
                        camera.undistortImage(imgAux, imgAux_undistorted);
                    else
                        imgAux_undistorted = imgAux;

                    new_kf->original_image_ =
                        tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
                    // std::cout << "new_kf->original_image_" << new_kf->original_image_ << std::endl;
                    new_kf->img_filename_ = pKF->mNameFile;
                    new_kf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
                    new_kf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
                    new_kf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;

                    // Compute transformations
                    // new_kf->computeTransformTensors(); //useless
                    scene_->addKeyframe(new_kf, &kfid_shuffled_);

                    increaseKeyframeTimesOfUse(new_kf, newKeyframeTimesOfUse());

                    // // Features for increasePcdByKeyframeInactiveGeoDensify
                    std::vector<float> pixels;
                    std::vector<float> pointsLocal;
                    pKF->GetKeypointInfo(pixels, pointsLocal);
                    new_kf->kps_pixel_ = std::move(pixels);
                    new_kf->kps_point_local_ = std::move(pointsLocal);
                    new_kf->img_undist_ = imgRGB_undistorted;
                    new_kf->img_auxiliary_undist_ = imgAux_undistorted;

                    // ── Log this initial keyframe to Rerun as a camera ──
                    try {
                        const unsigned long kf_id = pKF->mnId;

                        // Use full resolution of this keyframe
                        const int image_height = new_kf->image_height_;
                        const int image_width  = new_kf->image_width_;

                        // Build MiniCam (this uses Tcw_ internally and sets c2w/w2c)
                        sv::MiniCam cam = new_kf->toMiniCam(image_height, image_width);

                        // Intrinsics for Rerun
                        const float fx = static_cast<float>(camera.fx());
                        const float fy = static_cast<float>(camera.fy());
                        const float cx = static_cast<float>(camera.cx());
                        const float cy = static_cast<float>(camera.cy());

                        // cam.c2w is a 4x4 torch::Tensor on CPU or CUDA; move to CPU & map to Eigen
                        torch::Tensor c2w_cpu = cam.c2w.to(torch::kCPU).contiguous();
                        TORCH_CHECK(c2w_cpu.sizes() == torch::IntArrayRef({4, 4}),
                                    "MiniCam.c2w must be 4x4");

                        Eigen::Matrix4f T_W_C;
                        {
                            float* data = c2w_cpu.data_ptr<float>();
                            Eigen::Map<const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>
                                T_row_major(data);
                            T_W_C = T_row_major;
                        }

                        // In RGB-D hole-fill debug runs, show the actual depth map
                        // sampled by rgbd_fill_render_holes instead of the RGB keyframe.
                        cv::Mat depth_rerun_img;
                        const cv::Mat* track_img = &new_kf->img_undist_;
                        if (sensor_type_ == RGBD && rgbd_fill_render_holes_ &&
                            makeRgbdDepthRerunImage(
                                new_kf->img_auxiliary_undist_,
                                RGBD_min_depth_,
                                RGBD_max_depth_,
                                depth_rerun_img)) {
                            track_img = &depth_rerun_img;
                        }

                        constexpr unsigned long rerun_kf_begin = 50;
                        if (enable_rerun_ && !rerun_final_only_ &&
                            (rerun_max_keyframes_ <= 0 ||
                             (kf_id >= rerun_kf_begin &&
                              kf_id < rerun_kf_begin +
                                      static_cast<unsigned long>(rerun_max_keyframes_)))) {
                            sv::RerunVisualizerBridge::instance().visualizeCamera(
                                T_W_C,
                                *track_img,
                                std::vector<Eigen::Vector2f>{},  // no 2D keypoints for now
                                std::vector<int>{},              // no track ids
                                getIteration(),
                                static_cast<int>(kf_id),
                                fx, fy, cx, cy
                            );
                        }
                    } catch (const c10::Error& e) {
                        (void)e;
                        // std::cerr << "[RERUN] Torch error in visualizeCamera (initial KFs): "
                        //         << e.msg() << std::endl;
                    } catch (const std::exception& e) {
                        (void)e;
                        // std::cerr << "[RERUN] Exception in visualizeCamera (initial KFs): "
                        //         << e.what() << std::endl;
                    }

                    // ─── nvblox: integrate this keyframe’s depth into TSDF ───
                    if (sensor_type_ == RGBD && need_nvblox) {
                        // Assume imgAux_undistorted is a depth map aligned to imgRGB_undistorted
                        cv::Mat depth_meters;
                        if (imgAux_undistorted.type() == CV_32FC1) {
                            // already in meters
                            depth_meters = imgAux_undistorted;
                        } else if (imgAux_undistorted.type() == CV_16UC1) {
                            // common RealSense-style millimeters → meters conversion
                            imgAux_undistorted.convertTo(depth_meters, CV_32FC1, 1.0 / 1000.0);
                        } else {
                            // fallback: convert to float, assume already in meters scale
                            imgAux_undistorted.convertTo(depth_meters, CV_32FC1);
                        }
                        integrateKeyframeIntoNvblox(*new_kf, depth_meters);
                    }

                }
            }   // Mutex released

            // Prepare multi resolution images for training
            for (auto& kfit : scene_->keyframes()) {
                auto pkf = kfit.second;
                if (device_type_ == torch::kCUDA) {
                    cv::cuda::GpuMat img_gpu;
                    img_gpu.upload(pkf->img_undist_);
                    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                        cv::cuda::GpuMat img_resized;
                        cv::cuda::resize(img_gpu, img_resized,
                                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                        pkf->gaus_pyramid_original_image_[l] =
                            tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
                        // std::cout << "pkf->gaus_pyramid_original_image_[l]" << pkf->gaus_pyramid_original_image_[l] << std::endl;
                    }
                }
                else {
                    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                        cv::Mat img_resized;
                        cv::resize(pkf->img_undist_, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                        pkf->gaus_pyramid_original_image_[l] =
                            tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
                    }
                }
            }

            // C) Create MiniCams for all keyframes and use them for densification later
            std::vector<sv::MiniCam> tr_cams;
            tr_cams.reserve(scene_->keyframes().size());
            for (auto& kv : scene_->keyframes()) {
                // if (kv.second) tr_cams.push_back(kv.second->toMiniCam());
                auto& kf = *kv.second;
                // Use full-res here; you can choose a smaller level if you like.
                tr_cams.emplace_back(kf.toMiniCam(kf.image_height_, kf.image_width_));
            }
            if (!tr_cams.empty()) {
                const auto& c0 = tr_cams.front();
                // position and lookat are 1D tensors of size 3 on CPU
                auto px = c0.position.index({0}).item<float>();
                auto py = c0.position.index({1}).item<float>();
                auto pz = c0.position.index({2}).item<float>();
                auto lx = c0.lookat.index({0}).item<float>();
                auto ly = c0.lookat.index({1}).item<float>();
                auto lz = c0.lookat.index({2}).item<float>();
                // std::cout << "[MiniCam debug] first cam: "
                //         << "pos=(" << px << "," << py << "," << pz << ") "
                //         << "lookat=(" << lx << "," << ly << "," << lz << ") "
                //         << "pix_size=" << c0.pix_size
                //         << std::endl;
            }
            // D) Create voxel model & trainer setup
            if (enable_rerun_ && !rerun_final_only_) {
                appendAndLogOrbRawMapPcdToRerun(
                    scene_->cached_point_cloud_,
                    getIteration());
            }
            {
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                // scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
                // std::cout << "[VoxelMapper] Scene extent: " 
                //             << scene_->cameras_extent_ << std::endl;
                // save_initial_pcd_npy(result_dir_, scene_->cached_point_cloud_);
                // auto restored = load_full_pcd_from_logs((result_dir_ / "offline_experiment").string());
                // voxel_model_->createFromPcd(std::move(restored));
                voxel_model_->createFromPcd(
                    scene_->cached_point_cloud_,
                    tr_cams);
                if (sensor_type_ == RGBD) {
                    const int64_t roots_created = voxel_model_->numVoxels();
                    tsdf_ablation_rgbd_points_created_ += roots_created;
                    tsdf_ablation_rgbd_points_lineage_created_ += roots_created;
                }
                std::unique_lock<std::mutex> lock(mutex_settings_);
                voxel_model_->createTrainer(
                                            opt_params_.geo_lr_,
                                            opt_params_.sh0_lr_,
                                            opt_params_.shs_lr_,
                                            opt_params_.optim_beta1_,
                                            opt_params_.optim_beta2_,
                                            opt_params_.optim_eps_,
                                            opt_params_.lr_decay_ckpt_,
                                            opt_params_.lr_decay_mult_);
            }

            if (sensor_type_ == RGBD && inactive_geo_densify_) {
                std::vector<std::shared_ptr<VoxelKeyframe>> initial_rgbd_kfs;
                initial_rgbd_kfs.reserve(scene_->keyframes().size());
                for (const auto& kv : scene_->keyframes()) {
                    if (kv.second && !kv.second->done_inactive_geo_densify_) {
                        initial_rgbd_kfs.push_back(kv.second);
                    }
                }
                for (const auto& pkf : initial_rgbd_kfs) {
                    increasePcdByKeyframeInactiveGeoDensify(pkf);
                }
            }

            if (sensor_type_ == MONOCULAR &&
                depthanything_fill_holes_ &&
                depthanything_fill_holes_initial_backfill_) {
                std::vector<std::shared_ptr<VoxelKeyframe>> initial_fill_kfs;
                initial_fill_kfs.reserve(scene_->keyframes().size());
                for (const auto& kv : scene_->keyframes()) {
                    if (kv.second) {
                        initial_fill_kfs.push_back(kv.second);
                    }
                }

                // std::cout << "[mono_prior_fill_holes/initial_backfill] start"
                //           << " kfs=" << initial_fill_kfs.size()
                //           << " min_initial_map_kfs=" << min_num_initial_map_kfs_
                //           << " iter=" << getIteration()
                //           << std::endl;
                if (depthAnythingFillHolesWarmupReady()) {
                    applyDepthAnythingFillHolesKeyframes(
                        initial_fill_kfs,
                        /*seed_global_alignment=*/true);
                } else {
                    for (const auto& pkf : initial_fill_kfs) {
                        queueDepthAnythingFillHolesKeyframe(pkf);
                    }
                }
                // std::cout << "[mono_prior_fill_holes/initial_backfill] done"
                //           << " kfs=" << initial_fill_kfs.size()
                //           << " iter=" << getIteration()
                //           << std::endl;
            }

            // One warm-up optimization step
            trainForOneIteration();

            initial_mapped_ = true;
            break;  // Exit the initial mapping loop
        }
        else if (mpSLAM->isShutDown())
        {
            break;
        }
        else
        {
            // Initial conditions not satisfied yet
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Second loop: Incremental voxel mapping
     int SLAM_stop_iter = 0;
     while (!isStopped()) {
        // Check conditions for incremental mapping
        if (hasMetIncrementalMappingConditions()) {
            combineMappingOperations();
            if (cull_keyframes_)
                cullKeyframes();
        }
 
        processDepthAnythingFillHolesWarmup();

        // Invoke training once
        trainForOneIteration();

        if (mpSLAM->isShutDown()) {
            SLAM_stop_iter = getIteration();
            SLAM_ended_ = true;
        }

        if (SLAM_ended_ || getIteration() >= opt_params_.iterations_)
            break;
    }
    
    // // Third loop: Tail gaussian optimization
    int adapt_interval = opt_params_.adapt_every_;          // cfg.procedure.adapt_every
    int n_delay_iters  = adapt_interval * 0.8f;        // same heuristic as GS code
    while (getIteration() - SLAM_stop_iter <= n_delay_iters
        || (getIteration() % adapt_interval) <= n_delay_iters
        || isKeepingTraining() )
    {
        processDepthAnythingFillHolesWarmup();
        trainForOneIteration();
        // Re-read in case user changed cfg at runtime
        adapt_interval = opt_params_.adapt_every_;
        n_delay_iters  = adapt_interval * 0.8f;
    }

    // Final one-shot special prune after tail optimization:
    // remove voxels that are far from dense-core, near cameras,
    // and/or still above the target voxel size.
    if (opt_params_.final_special_prune_enable_) {
        const int before_final_special = voxel_model_->numVoxels();
        if (before_final_special > 0) {
            auto centers = voxel_model_->voxCenter();
            auto sizes = voxel_model_->voxSize();
            if (sizes.defined() && sizes.dim() == 2 && sizes.size(1) == 1) {
                sizes = sizes.squeeze(1);
            } else if (sizes.defined() && sizes.dim() != 1) {
                sizes = sizes.reshape({-1});
            }
            auto prune_mask_final_special = torch::zeros(
                {before_final_special},
                torch::TensorOptions().dtype(torch::kBool).device(centers.device()));

            int64_t n_far_final_special = 0;
            int64_t n_near_final_special = 0;
            int64_t n_above_target_final_special = 0;
            int64_t n_tsdf_final_special = 0;
            int64_t n_tsdf_valid_final_special = 0;
            int64_t n_tsdf_unknown_final_special = 0;
            int64_t n_tsdf_surface_final_special = 0;
            int64_t n_tsdf_occupied_final_special = 0;
            bool far_valid_final_special = false;
            bool near_valid_final_special = false;
            bool tsdf_valid_final_special = false;
            torch::Tensor tsdf_free_mask_final_special;

            bool use_far_final_special = opt_params_.prune_far_voxels_;
            if (use_far_final_special) {
                if (opt_params_.prune_recompute_dense_core_) {
                    const bool refreshed_dense_core =
                        voxel_model_->refreshDenseCoreBBFromCurrentVoxels();
                    if (!voxel_model_->hasDenseCoreBB()) {
                        use_far_final_special = false;
                        std::cout << "[FINAL/special_prune] dense-core refresh unavailable; "
                                  << "skipping far pruning.\n";
                    } else if (!refreshed_dense_core) {
                        std::cout << "[FINAL/special_prune] dense-core refresh failed; "
                                  << "using last available bbox.\n";
                    }
                } else {
                    if (!voxel_model_->hasDenseCoreBB()) {
                        use_far_final_special = false;
                        std::cout << "[FINAL/special_prune] cached dense-core bbox unavailable; "
                                  << "skipping far pruning.\n";
                    } else {
                        std::cout << "[FINAL/special_prune] using cached dense-core bbox "
                                  << "(no recompute).\n";
                    }
                }
                if (enable_rerun_ && !rerun_final_only_ && voxel_model_->hasDenseCoreBB()) {
                    voxel_model_->logDenseCoreBBoxToRerun(
                        getIteration(),
                        "world/dense_core/used_for_prune");
                }
            }

            if (use_far_final_special && voxel_model_->hasDenseCoreBB()) {
                auto bb_min = voxel_model_->denseCoreBBMin();
                auto bb_max = voxel_model_->denseCoreBBMax();
                if (centers.defined() && bb_min.defined() && bb_max.defined() &&
                    centers.dim() == 2 && centers.size(1) == 3 &&
                    centers.size(0) == before_final_special &&
                    bb_min.numel() == 3 && bb_max.numel() == 3) {
                    auto centers_f32 = centers.to(torch::kFloat32).contiguous();
                    bb_min = bb_min.to(centers_f32.device()).to(torch::kFloat32).contiguous().view({1, 3});
                    bb_max = bb_max.to(centers_f32.device()).to(torch::kFloat32).contiguous().view({1, 3});
                    auto in_dense_core =
                        (centers_f32 >= bb_min).all(/*dim=*/1) &
                        (centers_f32 <= bb_max).all(/*dim=*/1);
                    auto far_mask = (~in_dense_core.to(torch::kBool)).contiguous();
                    n_far_final_special = far_mask.sum().item<int64_t>();
                    prune_mask_final_special = (prune_mask_final_special | far_mask).to(torch::kBool);
                    far_valid_final_special = true;
                }
            }

            if (sizes.defined() && sizes.numel() == before_final_special) {
                auto sizes_f32 = sizes.to(centers.device()).to(torch::kFloat32).contiguous();
                auto above_target_mask =
                    (sizes_f32 > opt_params_.subdivide_target_vox_size_).to(torch::kBool).contiguous();
                n_above_target_final_special = above_target_mask.sum().item<int64_t>();
                prune_mask_final_special =
                    (prune_mask_final_special | above_target_mask).to(torch::kBool);
            }

            {
                std::vector<sv::MiniCam> tr_cams;
                tr_cams.reserve(scene_->keyframes().size());
                for (const auto& kv : scene_->keyframes()) {
                    tr_cams.push_back(kv.second->toMiniCam(
                        kv.second->image_height_, kv.second->image_width_));
                }

                if (tr_cams.empty()) {
                    n_near_final_special = 0;
                    near_valid_final_special = true;
                } else {
                    try {
                        py::gil_scoped_acquire gil;
                        static py::module svr_mod = py::module::import("svraster_cuda").attr("renderer");
                        static py::module torch_mod = py::module::import("torch");

                        auto svm = voxel_model_->svm();
                        auto octpath = svm.attr("octpath").cast<torch::Tensor>().contiguous();
                        auto vox_center = svm.attr("vox_center").cast<torch::Tensor>().contiguous();
                        auto vox_size = svm.attr("vox_size").cast<torch::Tensor>().contiguous();

                        TORCH_CHECK(octpath.size(0) == before_final_special,
                                    "octpath length mismatch at final special prune");
                        TORCH_CHECK(vox_center.size(0) == before_final_special,
                                    "vox_center length mismatch at final special prune");
                        TORCH_CHECK(vox_center.size(1) == 3,
                                    "vox_center must be [N,3] at final special prune");
                        if (vox_size.dim() == 1) {
                            vox_size = vox_size.view({before_final_special, 1});
                        } else if (vox_size.dim() == 2) {
                            TORCH_CHECK(vox_size.size(0) == before_final_special,
                                        "vox_size length mismatch at final special prune");
                        } else {
                            TORCH_CHECK(false, "vox_size must be [N] or [N,1] at final special prune");
                        }

                        py::list py_cams;
                        py::object py_cuda = torch_mod.attr("device")("cuda");
                        auto move_attr_to_cuda_if_tensor =
                            [&](py::object& obj, const char* name) {
                                if (py::hasattr(obj, name)) {
                                    py::object t = obj.attr(name);
                                    if (py::hasattr(t, "is_cuda") && !py::bool_(t.attr("is_cuda"))) {
                                        obj.attr(name) = t.attr("to")(py_cuda);
                                    }
                                }
                            };

                        for (const auto& c : tr_cams) {
                            py::object py_cam = MiniCam_to_py(c);
                            move_attr_to_cuda_if_tensor(py_cam, "w2c");
                            move_attr_to_cuda_if_tensor(py_cam, "c2w");
                            move_attr_to_cuda_if_tensor(py_cam, "position");
                            move_attr_to_cuda_if_tensor(py_cam, "lookat");
                            py_cams.append(py_cam);
                        }

                        const float near_thresh = 0.2f;
                        at::Tensor is_near = svr_mod.attr("mark_near")(
                            py_cams,
                            py::cast(octpath),
                            py::cast(vox_center),
                            py::cast(vox_size),
                            py::float_(near_thresh)
                        ).cast<at::Tensor>();
                        if (is_near.dim() == 2 && is_near.size(1) == 1) {
                            is_near = is_near.squeeze(1);
                        }
                        is_near = is_near.to(torch::kBool);
                        n_near_final_special = is_near.sum().item<int64_t>();
                        prune_mask_final_special =
                            (prune_mask_final_special |
                             is_near.to(prune_mask_final_special.device()).to(torch::kBool)).to(torch::kBool);
                        near_valid_final_special = true;
                    } catch (const std::exception& e) {
                        std::cerr << "[FINAL/special_prune] failed to compute near voxels: "
                                  << e.what() << "\n";
                    }
                }
            }

            if (sensor_type_ == RGBD && use_tsdf_pruning_) {
                if (sdf_mapper_ && sdf_mapper_->tsdf_layer().size() > 0 &&
                    sizes.defined() && sizes.numel() == before_final_special) {
                    try {
                        TsdfCornerSample c = sampleTsdfAtSvrasterGridCornersWorld();
                        torch::Tensor tsdf8 = c.tsdf;    // [N,8]
                        torch::Tensor w8 = c.weight;     // [N,8]
                        torch::Tensor ok8 = c.success;   // [N,8]

                        if (tsdf8.defined() && w8.defined() && ok8.defined() &&
                            tsdf8.dim() == 2 && tsdf8.size(0) == before_final_special &&
                            tsdf8.size(1) == 8 &&
                            w8.sizes() == tsdf8.sizes() &&
                            ok8.sizes() == tsdf8.sizes()) {
                            if (tsdf8.device() != prune_mask_final_special.device()) {
                                tsdf8 = tsdf8.to(prune_mask_final_special.device());
                                w8 = w8.to(prune_mask_final_special.device());
                                ok8 = ok8.to(prune_mask_final_special.device());
                            }

                            const float min_weight = tsdf_prune_min_weight_;
                            const float tau_surface =
                                std::max(0.0f, tsdf_prune_surface_band_vox_) *
                                sdf_mapper_->tsdf_layer().voxel_size();
                            const float sign_eps = 1.0e-6f;

                            torch::Tensor corner_valid =
                                (ok8 & (w8 >= min_weight)).to(torch::kBool);
                            torch::Tensor valid_count =
                                corner_valid.to(torch::kInt32).sum(/*dim=*/1);
                            torch::Tensor voxel_valid =
                                (valid_count >= tsdf_prune_min_valid_corners_).to(torch::kBool);

                            torch::Tensor has_pos =
                                ((tsdf8 > sign_eps) & corner_valid).any(/*dim=*/1);
                            torch::Tensor has_neg =
                                ((tsdf8 < -sign_eps) & corner_valid).any(/*dim=*/1);
                            torch::Tensor has_surface =
                                (has_pos & has_neg).to(torch::kBool);

                            torch::Tensor abs_tsdf = tsdf8.abs();
                            torch::Tensor inf_like =
                                torch::full_like(abs_tsdf, std::numeric_limits<float>::infinity());
                            torch::Tensor abs_valid =
                                torch::where(corner_valid, abs_tsdf, inf_like);
                            torch::Tensor min_abs_tsdf =
                                std::get<0>(abs_valid.min(/*dim=*/1));

                            torch::Tensor sizes_for_tsdf = sizes;
                            if (sizes_for_tsdf.dim() == 2 && sizes_for_tsdf.size(1) == 1) {
                                sizes_for_tsdf = sizes_for_tsdf.squeeze(1);
                            } else if (sizes_for_tsdf.dim() != 1) {
                                sizes_for_tsdf = sizes_for_tsdf.reshape({before_final_special});
                            }
                            sizes_for_tsdf = sizes_for_tsdf
                                .to(prune_mask_final_special.device())
                                .to(torch::kFloat32)
                                .contiguous();

                            torch::Tensor half_voxel_diag =
                                (0.5f * std::sqrt(3.0f)) * sizes_for_tsdf;
                            torch::Tensor conservative_free_thresh =
                                torch::full_like(half_voxel_diag, tau_surface) + half_voxel_diag;
                            torch::Tensor strong_far_from_surface =
                                (min_abs_tsdf > conservative_free_thresh).to(torch::kBool);

                            torch::Tensor all_positive =
                                (voxel_valid & has_pos & (~has_neg)).to(torch::kBool);
                            torch::Tensor all_negative =
                                (voxel_valid & has_neg & (~has_pos)).to(torch::kBool);
                            torch::Tensor tsdf_free_mask =
                                (all_positive & strong_far_from_surface).to(torch::kBool);
                            torch::Tensor tsdf_occupied_mask =
                                (all_negative & strong_far_from_surface).to(torch::kBool);
                            torch::Tensor tsdf_surface_mask =
                                (voxel_valid & (has_surface | (~strong_far_from_surface)))
                                    .to(torch::kBool);
                            torch::Tensor tsdf_unknown_mask =
                                (~voxel_valid).to(torch::kBool);
                            tsdf_free_mask_final_special = tsdf_free_mask.clone();
                            recordTsdfPruneAblation(
                                tsdf_free_mask,
                                "final_special_prune");

                            n_tsdf_final_special = tsdf_free_mask.sum().item<int64_t>();
                            n_tsdf_valid_final_special = voxel_valid.sum().item<int64_t>();
                            n_tsdf_unknown_final_special = tsdf_unknown_mask.sum().item<int64_t>();
                            n_tsdf_surface_final_special = tsdf_surface_mask.sum().item<int64_t>();
                            n_tsdf_occupied_final_special =
                                tsdf_occupied_mask.sum().item<int64_t>();
                            tsdf_valid_final_special = true;

                            prune_mask_final_special =
                                (prune_mask_final_special |
                                 tsdf_free_mask.to(prune_mask_final_special.device()))
                                    .to(torch::kBool);

                            std::cout << "[FINAL/special_prune/tsdf] iter=" << getIteration()
                                      << " N=" << before_final_special
                                      << " voxel_valid=" << n_tsdf_valid_final_special
                                      << " free_space=" << n_tsdf_final_special
                                      << " occupied_side=" << n_tsdf_occupied_final_special
                                      << " surface_band=" << n_tsdf_surface_final_special
                                      << " unknown=" << n_tsdf_unknown_final_special
                                      << " min_weight=" << min_weight
                                      << " min_valid_corners=" << tsdf_prune_min_valid_corners_
                                      << " surface_band_m=" << tau_surface
                                      << "\n";
                        } else {
                            std::cout << "[FINAL/special_prune/tsdf] shape mismatch; skipping.\n";
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "[FINAL/special_prune/tsdf] failed: "
                                  << e.what() << "\n";
                    }
                } else {
                    std::cout << "[FINAL/special_prune/tsdf] skipped"
                              << " mapper=" << (sdf_mapper_ ? 1 : 0)
                              << " tsdf_blocks="
                              << (sdf_mapper_ ? sdf_mapper_->tsdf_layer().size() : 0)
                              << " sizes_ok="
                              << (sizes.defined() && sizes.numel() == before_final_special)
                              << "\n";
                }
            }

            const int64_t n_selected_final_special =
                prune_mask_final_special.to(torch::kBool).sum().item<int64_t>();
            if (enable_rerun_ && n_selected_final_special > 0) {
                auto log_final_special_mask =
                    [&](const torch::Tensor& mask_in,
                        float r,
                        float g,
                        float b,
                        float a,
                        const std::string& entity_path,
                        const std::string& label)
                {
                    if (!mask_in.defined() || mask_in.numel() != before_final_special ||
                        !centers.defined() || !sizes.defined()) {
                        return;
                    }
                    torch::Tensor mask = mask_in.to(centers.device()).to(torch::kBool).view({-1});
                    torch::Tensor idx = mask.nonzero().squeeze(1);
                    if (idx.numel() <= 0) {
                        return;
                    }

                    torch::Tensor centers_sel = centers.index_select(0, idx).contiguous();
                    torch::Tensor sizes_for_log = sizes;
                    if (sizes_for_log.dim() == 1) {
                        sizes_for_log = sizes_for_log.view({sizes_for_log.size(0), 1});
                    } else if (!(sizes_for_log.dim() == 2 && sizes_for_log.size(1) == 1)) {
                        sizes_for_log = sizes_for_log.reshape({sizes_for_log.size(0), 1});
                    }
                    torch::Tensor sizes_sel = sizes_for_log.index_select(0, idx).contiguous();
                    torch::Tensor colors = torch::zeros({idx.size(0), 4}, centers_sel.options());
                    colors.index_put_({torch::indexing::Slice(), 0}, r);
                    colors.index_put_({torch::indexing::Slice(), 1}, g);
                    colors.index_put_({torch::indexing::Slice(), 2}, b);
                    colors.index_put_({torch::indexing::Slice(), 3}, a);

                    std::cout << "[FINAL/special_prune/rerun] visualizing "
                              << idx.size(0) << " " << label << " voxels\n";
                    sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
                        centers_sel,
                        sizes_sel,
                        colors,
                        getIteration(),
                        entity_path);
                };

                log_final_special_mask(
                    prune_mask_final_special,
                    1.0f, 0.65f, 0.0f, 0.65f,
                    "world/final_special_prune/selected",
                    "final-special selected");
                if (tsdf_free_mask_final_special.defined()) {
                    log_final_special_mask(
                        tsdf_free_mask_final_special,
                        0.0f, 0.35f, 1.0f, 0.75f,
                        "world/final_special_prune/tsdf_free",
                        "final-special TSDF-free");
                }
            }
            if (n_selected_final_special > 0) {
                voxel_model_->pruning(prune_mask_final_special);
            }
            const int after_final_special = voxel_model_->numVoxels();
            std::cout << "[FINAL/special_prune] before=" << before_final_special
                      << " selected=" << n_selected_final_special
                      << " removed=" << (before_final_special - after_final_special)
                      << " far=" << (far_valid_final_special ? std::to_string(n_far_final_special) : std::string("N/A"))
                      << " near=" << (near_valid_final_special ? std::to_string(n_near_final_special) : std::string("N/A"))
                      << " above_target=" << n_above_target_final_special
                      << " tsdf_free=" << (tsdf_valid_final_special ? std::to_string(n_tsdf_final_special) : std::string("N/A"))
                      << " tsdf_valid=" << (tsdf_valid_final_special ? std::to_string(n_tsdf_valid_final_special) : std::string("N/A"))
                      << " tsdf_unknown=" << (tsdf_valid_final_special ? std::to_string(n_tsdf_unknown_final_special) : std::string("N/A"))
                      << "\n";
        }
    } else {
        std::cout << "[FINAL/special_prune] disabled\n";
    }
    if (sensor_type_ == RGBD && use_tsdf_pruning_) {
        printTsdfPruneAblationSummary("shutdown");
    }
    // {
    //     // 1) Check if we had a recent artificial fill (fill_empty_cells_)
    //     bool need_forced_densify = false;
    //     if (last_artificial_fill_iter_ >= 0 && last_densify_iter_ >= 0) {
    //         if (last_artificial_fill_iter_ > last_densify_iter_) {
    //             need_forced_densify = true;
    //             std::cout << "[VoxelMapper] Tail: artificial fill after last densify "
    //                     << "(last_fill_iter=" << last_artificial_fill_iter_
    //                     << ", last_densify_iter=" << last_densify_iter_
    //                     << ") -> will force one densification.\n";
    //         }
    //     }
    //     // 2) If needed, run ONE forced densification step (prune+subdivide)
    //     if (need_forced_densify) {
    //         // Backup optimization parameters and last_artificial_fill_iter_
    //         auto   backup_opt        = opt_params_;
    //         auto   backup_last_fill  = last_artificial_fill_iter_;
    //         const int cur_iter_before = getIteration();

    //         // Configure opt_params_ so that the NEXT iteration always densifies:
    //         //   - adapt_every_ = 1 → meet_adapt_period every iteration
    //         //   - adapt_from_ ≤ current_iter
    //         //   - prune_until_ / subdivide_until_ ≥ current_iter+1
    //         opt_params_.adapt_every_     = 1;
    //         opt_params_.adapt_from_      =
    //             std::min(backup_opt.adapt_from_, cur_iter_before);
    //         opt_params_.prune_until_     =
    //             std::max(backup_opt.prune_until_, cur_iter_before + 1);
    //         opt_params_.subdivide_until_ =
    //             std::max(backup_opt.subdivide_until_, cur_iter_before + 1);

    //         // Disable cooldown for this forced step so densification actually runs
    //         last_artificial_fill_iter_ = -1;

    //         // This call WILL:
    //         //  - do one training iteration
    //         //  - trigger prune/subdivide according to the modified opt_params_
    //         trainForOneIteration();

    //         // Restore original parameters and state
    //         opt_params_             = backup_opt;
    //         last_artificial_fill_iter_ = backup_last_fill;

    //         std::cout << "[VoxelMapper] Tail: forced densification step done at iter "
    //                 << getIteration() << "\n";
    //     }
    //     // 3) Tail optimization: run 300 extra iterations AFTER densification (if any)
    //     const int tail_extra_iters = 300;  // can move to YAML later
    //     const int start_tail_iter  = getIteration();
    //     const int end_tail_iter    = start_tail_iter + tail_extra_iters;
    //     while (getIteration() < end_tail_iter) {
    //         trainForOneIteration();
    //     }
    // }

    // ─────────────────────────────────────────────────────────────
    // TSDF-based transparency AFTER all training is done
    // ─────────────────────────────────────────────────────────────
    // if (sensor_type_ == RGBD && use_tsdf_mapping_) {
    //     applyFinalTsdfTransparency();
    // }

    // if (have_bounds_) {
    //     std::cout.setf(std::ios::fixed);
    //     std::cout << std::setprecision(6)
    //             << "[AABB:final] min:[" << aabb_min_.x() << "," << aabb_min_.y() << "," << aabb_min_.z()
    //             << "] max:[" << aabb_max_.x() << "," << aabb_max_.y() << "," << aabb_max_.z() << "]\n";
    // }

    // ------------------------------------------------------------
    // One-shot offline planning (in-process ESDF) + Rerun visualization
    // ------------------------------------------------------------
    if (!planned_once_ && sensor_type_ == RGBD && use_tsdf_planning_ && sdf_mapper_) {
        // Choose a stable iteration stamp for rerun. Best: last keyframe id or last training iter.
        // Here we use last keyframe id if possible; fallback to 0.
        int iteration0 = 0;
        if (scene_ && !scene_->keyframes().empty()) {
            iteration0 = (int)scene_->keyframes().rbegin()->first; // assumes ordered map
        }

        // 1) Update ESDF once (full layer) after mapping is complete
        sdf_mapper_->updateEsdf(nvblox::UpdateFullLayer::kYes);
        cudaDeviceSynchronize();

        auto printLayerAabb = [&](const auto& layer, const char* name) {
            const std::vector<nvblox::Index3D> blocks = layer.getAllBlockIndices();
            std::cout << "[NVBLOX] " << name << " blocks=" << blocks.size() << "\n";
            if (blocks.empty()) return;

            nvblox::Index3D mn = blocks[0], mx = blocks[0];
            for (const auto& b : blocks) {
                mn = mn.cwiseMin(b);
                mx = mx.cwiseMax(b);
            }

            const float vs = layer.voxel_size(); // meters
            const int vps = 8;                   // VoxelBlock::kVoxelsPerSide
            const float bs = vs * float(vps);    // block size in meters

            Eigen::Vector3f min_m(mn.x() * bs, mn.y() * bs, mn.z() * bs);
            Eigen::Vector3f max_m((mx.x() + 1) * bs, (mx.y() + 1) * bs, (mx.z() + 1) * bs);

            std::cout << "[NVBLOX] " << name << " AABB(m): ["
                    << min_m.transpose() << "] -> ["
                    << max_m.transpose() << "] voxel_size=" << vs << "\n";
        };

        printLayerAabb(sdf_mapper_->esdf_layer(), "ESDF");
        printLayerAabb(sdf_mapper_->tsdf_layer(), "TSDF");

        // 2) Bounds from ESDF blocks AABB (planner domain)
        PlanBounds b;
        {
            auto& layer = sdf_mapper_->esdf_layer();
            const std::vector<nvblox::Index3D> blocks = layer.getAllBlockIndices();
            TORCH_CHECK(!blocks.empty(), "ESDF has no blocks.");

            nvblox::Index3D mn = blocks[0], mx = blocks[0];
            for (const auto& bi : blocks) {
                mn = mn.cwiseMin(bi);
                mx = mx.cwiseMax(bi);
            }

            const float vs = layer.voxel_size();
            const double bs = double(vs) * 8.0; // 8 voxels/side
            const double pad = 0.5;             // meters padding around ESDF AABB

            b.min_x = mn.x() * bs - pad;
            b.min_y = mn.y() * bs - pad;
            b.min_z = mn.z() * bs - pad;

            b.max_x = (mx.x() + 1) * bs + pad;
            b.max_y = (mx.y() + 1) * bs + pad;
            b.max_z = (mx.z() + 1) * bs + pad;
        }

        // 3) Start pose from first keyframe
        Eigen::Vector3d start_pos;
        Eigen::Quaterniond start_q = Eigen::Quaterniond::Identity();
        {
            auto kf0_it = scene_->keyframes().begin();
            TORCH_CHECK(kf0_it != scene_->keyframes().end(), "No keyframes in scene");
            auto kf0 = kf0_it->second;
            TORCH_CHECK(kf0, "First keyframe is null");

            sv::MiniCam c0 = kf0->toMiniCam(kf0->image_height_, kf0->image_width_);
            torch::Tensor c2w = c0.c2w.to(torch::kCPU).contiguous(); // [4,4]
            TORCH_CHECK(c2w.sizes() == torch::IntArrayRef({4,4}), "MiniCam.c2w must be [4,4]");

            start_pos = Eigen::Vector3d(
                c2w[0][3].item<double>(),
                c2w[1][3].item<double>(),
                c2w[2][3].item<double>()
            );

            // Optional: orientation
            // Eigen::Matrix3d R;
            // for (int r=0;r<3;++r) for (int c=0;c<3;++c) R(r,c) = c2w[r][c].item<double>();
            // start_q = Eigen::Quaterniond(R);
        }

        // 4) Goal pose: last keyframe position (current choice)
        Eigen::Vector3d goal_pos;
        {
            auto& kfs = scene_->keyframes();
            TORCH_CHECK(!kfs.empty(), "No keyframes in scene");

            auto it = std::prev(kfs.end());
            auto kf_last = it->second;
            TORCH_CHECK(kf_last, "Last keyframe is null");

            sv::MiniCam cl = kf_last->toMiniCam(kf_last->image_height_, kf_last->image_width_);
            auto c2w = cl.c2w.to(torch::kCPU).contiguous();

            goal_pos = Eigen::Vector3d(
                c2w[0][3].item<double>(),
                c2w[1][3].item<double>(),
                c2w[2][3].item<double>()
            );
        }

        // 5) Instantiate planner ONCE with bounds + validity function
        if (!planner_) {
            planner_ = std::make_unique<VoxelPlanner>(b);

            const double robot_radius = static_cast<double>(planner_clearance_m_);
            planner_->setValidityFunction([this, robot_radius](const Eigen::Vector3d& p) -> bool {
                float d = 0.f;
                const bool ok = this->queryEsdfAtWorld(p, d);
                if (!ok) return false;                          // unknown => invalid (fail closed)
                return (static_cast<double>(d) >= robot_radius); // free if clearance satisfied
            });
        }

        auto in_bounds = [&](const Eigen::Vector3d& p){
            return (p.x() >= b.min_x && p.x() <= b.max_x &&
                    p.y() >= b.min_y && p.y() <= b.max_y &&
                    p.z() >= b.min_z && p.z() <= b.max_z);
        };

        std::cout << "[Planner] start: " << start_pos.transpose()
                << " in_bounds=" << in_bounds(start_pos) << "\n";
        std::cout << "[Planner] goal:  " << goal_pos.transpose()
                << " in_bounds=" << in_bounds(goal_pos) << "\n";
        std::cout << "[Planner] bounds: "
                << "["<<b.min_x<<","<<b.max_x<<"] "
                << "["<<b.min_y<<","<<b.max_y<<"] "
                << "["<<b.min_z<<","<<b.max_z<<"]\n";

        // ---------------------------------------------------------
        // A) Rerun: log start & goal markers
        // ---------------------------------------------------------
        {
            torch::Tensor start_t = make_single_point_cpu_f32(start_pos);
            torch::Tensor goal_t  = make_single_point_cpu_f32(goal_pos);

            // Color them explicitly so it's always obvious.
            torch::Tensor green = torch::tensor({0,255,0}, torch::TensorOptions().dtype(torch::kUInt8)).view({1,3});
            torch::Tensor red   = torch::tensor({255,0,0}, torch::TensorOptions().dtype(torch::kUInt8)).view({1,3});

            sv::RerunVisualizerBridge::instance().visualizePoints3D(
                start_t, green, iteration0, "world/planner/start", 0.08f);

            sv::RerunVisualizerBridge::instance().visualizePoints3D(
                goal_t, red, iteration0, "world/planner/goal", 0.08f);
        }

        // --- ESDF sanity at start/goal ---
        auto dbg_esdf = [&](const char* tag, const Eigen::Vector3d& p) {
            float d = 0.f;
            const bool ok = this->queryEsdfAtWorld(p, d);
            std::cout << "[Planner] ESDF " << tag
                    << " ok=" << ok
                    << " d=" << d
                    << " clearance=" << planner_clearance_m_
                    << "\n";
        };
        dbg_esdf("start", start_pos);
        dbg_esdf("goal",  goal_pos);

        // ---------------------------------------------------------
        // B) Rerun: log "ESDF-known free samples" (key visual)
        // ---------------------------------------------------------
        {
            const int N = 25000; // dense cloud for a good "voxblox-like" free-space feeling
            std::vector<Eigen::Vector3f> free_pts;
            free_pts.reserve(N);

            std::mt19937 rng(0);
            std::uniform_real_distribution<double> ux(b.min_x, b.max_x);
            std::uniform_real_distribution<double> uy(b.min_y, b.max_y);
            std::uniform_real_distribution<double> uz(b.min_z, b.max_z);

            for (int i = 0; i < N; ++i) {
                Eigen::Vector3d p(ux(rng), uy(rng), uz(rng));
                float d = 0.f;
                const bool ok = this->queryEsdfAtWorld(p, d);
                if (!ok) continue;                        // ESDF unknown -> ignore
                if (d < planner_clearance_m_) continue;    // not free enough

                free_pts.emplace_back(p.cast<float>());
            }

            std::cout << "[Planner] free_samples kept=" << free_pts.size() << " / " << N << "\n";

            torch::Tensor pts = make_points_tensor_cpu_f32(free_pts);
            if (pts.defined() && pts.numel() > 0) {
                // Light gray points to mimic "free space" feel.
                torch::Tensor gray = torch::full({(long)free_pts.size(), 3}, 180,
                    torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));

                sv::RerunVisualizerBridge::instance().visualizePoints3D(
                    pts, gray, iteration0, "world/esdf/free_samples", 0.01f);
            }
        }

        // Optional: quick stats (keep if you like)
        {
            const int N = 3000;
            int ok_cnt = 0;
            int valid_cnt = 0;

            std::mt19937 rng(0);
            std::uniform_real_distribution<double> ux(b.min_x, b.max_x);
            std::uniform_real_distribution<double> uy(b.min_y, b.max_y);
            std::uniform_real_distribution<double> uz(b.min_z, b.max_z);

            for (int i = 0; i < N; ++i) {
                Eigen::Vector3d p(ux(rng), uy(rng), uz(rng));
                float d = 0.f;
                const bool ok = this->queryEsdfAtWorld(p, d);
                ok_cnt += ok ? 1 : 0;
                if (!ok) continue;
                if (d >= planner_clearance_m_) valid_cnt++;
            }

            std::cout << "[Planner] samples N=" << N
                    << " esdf_ok=" << ok_cnt << " (" << (ok_cnt/(double)N) << ")"
                    << " valid=" << valid_cnt << " (" << (valid_cnt/(double)N) << ")"
                    << "\n";
        }

        // 6) Plan
        const double solve_seconds = 10.0;
        PlanResult res = planner_->plan(start_pos, start_q, goal_pos, solve_seconds);

        if (res.success) {
            const auto out_path = (result_dir_ / "planned_path_xyz.txt").string();
            std::ofstream f(out_path);
            for (const auto& p : res.waypoints) {
                f << p.x() << " " << p.y() << " " << p.z() << "\n";
            }
            std::cout << "[Planner] success. waypoints=" << res.waypoints.size()
                    << " length=" << res.length << "\n";

            // ---------------------------------------------------------
            // A) Rerun: log planned path (static)
            // ---------------------------------------------------------
            if (res.waypoints.size() >= 2) {
                std::vector<Eigen::Vector3f> path_pts;
                path_pts.reserve(res.waypoints.size());
                for (const auto& p : res.waypoints) {
                    path_pts.emplace_back((float)p.x(), (float)p.y(), (float)p.z());
                }

                torch::Tensor path_t = make_points_tensor_cpu_f32(path_pts);

                // Path color: green
                torch::Tensor green = color_u8(0, 255, 0);

                sv::RerunVisualizerBridge::instance().visualizeLineStrip3D(
                    path_t, green, iteration0, "world/planner/path", 0.02f);

                // ---------------------------------------------------------
                // Robot playback along the path (SplatNav-like)
                // We log "world/planner/robot" at successive iter steps.
                // ---------------------------------------------------------
                {
                    // robot marker color: blue
                    torch::Tensor blue = torch::tensor({0, 120, 255}, torch::TensorOptions().dtype(torch::kUInt8))
                                            .view({1,3});

                    // How many frames to animate:
                    // If you want smoother motion, resample between waypoints later.
                    int base_iter = iteration0 + 1;

                    for (int i = 0; i < (int)path_pts.size(); ++i) {
                        Eigen::Vector3d p(path_pts[i].x(), path_pts[i].y(), path_pts[i].z());
                        torch::Tensor robot_t = make_single_point_cpu_f32(p);

                        sv::RerunVisualizerBridge::instance().visualizePoints3D(
                            robot_t, blue, base_iter + i, "world/planner/robot", (float)planner_clearance_m_);
                    }
                }
            }

        } else {
            std::cout << "[Planner] failed.\n";

            // Optional: log straight line in red to show "why planning matters".
            // (Leave commented if you prefer clean output.)
            /*
            std::vector<Eigen::Vector3f> seg;
            seg.emplace_back((float)start_pos.x(), (float)start_pos.y(), (float)start_pos.z());
            seg.emplace_back((float)goal_pos.x(),  (float)goal_pos.y(),  (float)goal_pos.z());
            torch::Tensor seg_t = make_points_tensor_cpu_f32(seg);
            torch::Tensor red = color_u8(255,0,0);
            sv::RerunVisualizerBridge::instance().visualizeLineStrip3D(
                seg_t, red, iteration0, "world/planner/failed_straight_line", 0.02f);
            */
        }

        planned_once_ = true;
    }

    // Final voxel summary at shutdown (for quick diagnostics).
    {
        const int64_t n_total_end = static_cast<int64_t>(voxel_model_->numVoxels());
        int64_t n_above_target_end = 0;
        int64_t n_at_or_below_target_end = 0;

        auto vox_size_end = voxel_model_->voxSize();
        if (vox_size_end.defined()) {
            if (vox_size_end.dim() == 2 && vox_size_end.size(1) == 1) {
                vox_size_end = vox_size_end.squeeze(1);
            } else if (vox_size_end.dim() != 1) {
                vox_size_end = vox_size_end.reshape({-1});
            }
            if (vox_size_end.numel() == n_total_end) {
                auto above_target_end =
                    (vox_size_end > opt_params_.subdivide_target_vox_size_).to(torch::kBool);
                n_above_target_end = above_target_end.sum().item<int64_t>();
                n_at_or_below_target_end = n_total_end - n_above_target_end;
            }
        }

        int64_t n_far_end = -1;
        bool far_count_valid = false;
        if (voxel_model_->hasDenseCoreBB()) {
            auto centers_end = voxel_model_->voxCenter();
            auto bb_min_end = voxel_model_->denseCoreBBMin();
            auto bb_max_end = voxel_model_->denseCoreBBMax();
            if (centers_end.defined() && bb_min_end.defined() && bb_max_end.defined() &&
                centers_end.dim() == 2 && centers_end.size(1) == 3 &&
                centers_end.size(0) == n_total_end &&
                bb_min_end.numel() == 3 && bb_max_end.numel() == 3) {
                centers_end = centers_end.to(torch::kFloat32).contiguous();
                bb_min_end = bb_min_end.to(centers_end.device()).to(torch::kFloat32).contiguous().view({1, 3});
                bb_max_end = bb_max_end.to(centers_end.device()).to(torch::kFloat32).contiguous().view({1, 3});
                auto in_dense_core_end =
                    (centers_end >= bb_min_end).all(/*dim=*/1) &
                    (centers_end <= bb_max_end).all(/*dim=*/1);
                auto far_mask_end = (~in_dense_core_end.to(torch::kBool)).contiguous();
                n_far_end = far_mask_end.sum().item<int64_t>();
                far_count_valid = true;
            }
        }

        int64_t n_near_end = -1;
        bool near_count_valid = false;
        if (n_total_end == 0) {
            n_near_end = 0;
            near_count_valid = true;
        } else {
            std::vector<sv::MiniCam> tr_cams;
            tr_cams.reserve(scene_->keyframes().size());
            for (const auto& kv : scene_->keyframes()) {
                tr_cams.push_back(kv.second->toMiniCam(
                    kv.second->image_height_, kv.second->image_width_));
            }

            if (!tr_cams.empty()) {
                try {
                    py::gil_scoped_acquire gil;
                    static py::module svr_mod = py::module::import("svraster_cuda").attr("renderer");
                    static py::module torch_mod = py::module::import("torch");

                    auto svm = voxel_model_->svm();
                    auto octpath = svm.attr("octpath").cast<torch::Tensor>().contiguous();
                    auto vox_center = svm.attr("vox_center").cast<torch::Tensor>().contiguous();
                    auto vox_size = svm.attr("vox_size").cast<torch::Tensor>().contiguous();

                    TORCH_CHECK(octpath.size(0) == n_total_end, "octpath length mismatch at final summary");
                    TORCH_CHECK(vox_center.size(0) == n_total_end, "vox_center length mismatch at final summary");
                    TORCH_CHECK(vox_center.size(1) == 3, "vox_center must be [N,3] at final summary");
                    if (vox_size.dim() == 1) {
                        vox_size = vox_size.view({n_total_end, 1});
                    } else if (vox_size.dim() == 2) {
                        TORCH_CHECK(vox_size.size(0) == n_total_end, "vox_size length mismatch at final summary");
                    } else {
                        TORCH_CHECK(false, "vox_size must be [N] or [N,1] at final summary");
                    }

                    py::list py_cams;
                    py::object py_cuda = torch_mod.attr("device")("cuda");
                    auto move_attr_to_cuda_if_tensor =
                        [&](py::object& obj, const char* name) {
                            if (py::hasattr(obj, name)) {
                                py::object t = obj.attr(name);
                                if (py::hasattr(t, "is_cuda") && !py::bool_(t.attr("is_cuda"))) {
                                    obj.attr(name) = t.attr("to")(py_cuda);
                                }
                            }
                        };

                    for (const auto& c : tr_cams) {
                        py::object py_cam = MiniCam_to_py(c);
                        move_attr_to_cuda_if_tensor(py_cam, "w2c");
                        move_attr_to_cuda_if_tensor(py_cam, "c2w");
                        move_attr_to_cuda_if_tensor(py_cam, "position");
                        move_attr_to_cuda_if_tensor(py_cam, "lookat");
                        py_cams.append(py_cam);
                    }

                    const float near_thresh = 0.2f;
                    at::Tensor is_near = svr_mod.attr("mark_near")(
                        py_cams,
                        py::cast(octpath),
                        py::cast(vox_center),
                        py::cast(vox_size),
                        py::float_(near_thresh)
                    ).cast<at::Tensor>();
                    if (is_near.dim() == 2 && is_near.size(1) == 1) {
                        is_near = is_near.squeeze(1);
                    }
                    auto near_mask_end = is_near.to(torch::kBool).contiguous();
                    n_near_end = near_mask_end.sum().item<int64_t>();
                    near_count_valid = true;
                } catch (const std::exception& e) {
                    std::cerr << "[FINAL/vox] failed to compute near count: "
                              << e.what() << "\n";
                }
            } else {
                n_near_end = 0;
                near_count_valid = true;
            }
        }

        std::cout << "[FINAL/vox] total=" << n_total_end
                  << " above_target=" << n_above_target_end
                  << " at_or_below_target=" << n_at_or_below_target_end;
        if (far_count_valid) {
            std::cout << " far=" << n_far_end;
        } else {
            std::cout << " far=N/A";
        }
        if (near_count_valid) {
            std::cout << " near=" << n_near_end;
        } else {
            std::cout << " near=N/A";
        }
        std::cout << " target_vox_size=" << opt_params_.subdivide_target_vox_size_
                  << "\n";
    }

    // Final-only Rerun mode: log only final voxel sets at shutdown.
    if (enable_rerun_ && rerun_final_only_) {
        const int final_iter = getIteration();
        torch::Tensor centers_all = voxel_model_->voxCenter(); // [N,3]
        torch::Tensor sizes_all   = voxel_model_->voxSize();   // [N] or [N,1]
        if (sizes_all.defined() && sizes_all.dim() == 1) {
            sizes_all = sizes_all.view({-1, 1});
        }

        torch::Tensor colors_all;
        {
            torch::Tensor sh0 = voxel_model_->sh0();
            {
                py::gil_scoped_acquire gil2;
                static py::module act_mod = py::module::import("src.utils.activation_utils");
                py::object rgb_py = act_mod.attr("shzero2rgb")(py::cast(sh0));
                colors_all = rgb_py.cast<torch::Tensor>().contiguous();
            }

            torch::Tensor density = voxel_model_->voxelDensityMean();
            if (density.defined() && density.numel() == centers_all.size(0)) {
                auto d_cpu = density.view({-1}).to(torch::kCPU);
                float d_min = d_cpu.min().item().toFloat();
                float d_max = d_cpu.max().item().toFloat();
                float eps   = 1e-6f;
                float range = d_max - d_min;

                torch::Tensor alpha_cpu;
                if (range < eps) {
                    alpha_cpu = torch::full_like(d_cpu, 0.8f);
                } else {
                    alpha_cpu = (d_cpu - d_min) / range;
                    alpha_cpu = alpha_cpu.clamp(0.05f, 1.0f);
                }
                auto col_cpu = colors_all.to(torch::kCPU);
                TORCH_CHECK(col_cpu.dim() == 2 &&
                            col_cpu.size(0) == alpha_cpu.size(0),
                            "colors and density must have same N");
                if (col_cpu.size(1) == 3) {
                    auto N = col_cpu.size(0);
                    auto col_rgba = torch::zeros({N, 4}, col_cpu.options());
                    col_rgba.index_put_(
                        {torch::indexing::Slice(), torch::indexing::Slice(0, 3)},
                        col_cpu
                    );
                    col_rgba.index_put_(
                        {torch::indexing::Slice(), 3},
                        alpha_cpu
                    );
                    colors_all = col_rgba.to(colors_all.device());
                } else if (col_cpu.size(1) == 4) {
                    col_cpu.index_put_(
                        {torch::indexing::Slice(), 3},
                        alpha_cpu
                    );
                    colors_all = col_cpu.to(colors_all.device());
                } else {
                    TORCH_CHECK(false, "colors must be [N,3] or [N,4]");
                }
            }
        }

        // 1) All final voxels
        sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
            centers_all, sizes_all, colors_all, final_iter, "world/voxels_final/all");

        // 2) Final voxels above target size
        torch::Tensor sizes_1d = sizes_all;
        if (sizes_1d.defined() && sizes_1d.dim() == 2 && sizes_1d.size(1) == 1) {
            sizes_1d = sizes_1d.squeeze(1);
        } else if (sizes_1d.defined() && sizes_1d.dim() != 1) {
            sizes_1d = sizes_1d.reshape({-1});
        }
        if (sizes_1d.defined() && sizes_1d.numel() == centers_all.size(0)) {
            auto above_target_mask =
                (sizes_1d > opt_params_.subdivide_target_vox_size_).to(torch::kBool).contiguous();
            auto idx_above_target = torch::nonzero(above_target_mask).view({-1});
            if (idx_above_target.numel() > 0) {
                auto centers_above = centers_all.index_select(0, idx_above_target).contiguous();
                auto sizes_above   = sizes_all.index_select(0, idx_above_target).contiguous();
                auto colors_above = torch::zeros(
                    {centers_above.size(0), 4},
                    torch::TensorOptions().dtype(torch::kFloat32).device(centers_above.device()));
                colors_above.index_put_({torch::indexing::Slice(), 0}, 1.0f); // R
                colors_above.index_put_({torch::indexing::Slice(), 3}, 0.85f);
                sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
                    centers_above, sizes_above, colors_above, final_iter,
                    "world/voxels_final/above_target");
            }
        }

        voxel_model_->logFinalPromotedartificialVoxels(final_iter);
    }

    // Save and clear
    renderAndRecordAllKeyframes("_shutdown");
    savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
    {
        const std::filesystem::path ply_dir =
            result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply" /
            "voxel_model" / ("iteration_" + std::to_string(getIteration()));
        const std::filesystem::path eval_mesh_path = ply_dir / "voxel_surface_mesh.ply";

        const bool want_rendered_mesh =
            save_rendered_mesh_eval_ || (enable_rerun_ && rerun_rendered_mesh_eval_);
        if (want_rendered_mesh) {
            try {
                saveRenderedTsdfMeshPly(eval_mesh_path);
                if (enable_rerun_ && rerun_rendered_mesh_eval_ &&
                    std::filesystem::exists(eval_mesh_path)) {
                    sv::RerunVisualizerBridge::instance().visualizeNvbloxPlyMesh(
                        eval_mesh_path.string(),
                        getIteration(),
                        "world/voxel_model_mesh/final");
                }
            } catch (const std::exception& e) {
                std::cerr << "[saveRenderedTsdfMeshPly] shutdown export failed: "
                          << e.what() << "\n";
            }
        } else {
            std::cout << "[saveRenderedTsdfMeshPly] skipped: disabled by "
                         "Record.save_rendered_mesh_eval and Record.rerun_rendered_mesh_eval.\n";
        }
    }
    {
        const bool want_nvblox_mesh =
            sensor_type_ == RGBD &&
            !load_saved_nvblox_mesh_ &&
            (save_nvblox_mesh_eval_ || (enable_rerun_ && rerun_nvblox_mesh_));
        if (want_nvblox_mesh && sdf_mapper_) {
            try {
                sdf_mapper_->updateColorMesh();
                const std::filesystem::path nvblox_dir =
                    result_dir_ / (std::to_string(getIteration()) + "_shutdown") /
                    "ply" / "nvblox";
                std::filesystem::create_directories(nvblox_dir);
                const std::filesystem::path nvblox_mesh_path =
                    nvblox_dir / "nvblox_color_mesh.ply";

                nvblox::io::outputColorMeshLayerToPly(
                    sdf_mapper_->color_mesh_layer(),
                    nvblox_mesh_path.string());

                if (enable_rerun_ && rerun_nvblox_mesh_) {
                    sv::RerunVisualizerBridge::instance().visualizeNvbloxPlyMesh(
                        nvblox_mesh_path.string(),
                        getIteration(),
                        "world/nvblox_mesh/final");
                }

                std::cout << "[NVBLOX] saved final mesh: "
                          << nvblox_mesh_path << "\n";
            } catch (const std::exception& e) {
                std::cerr << "[NVBLOX] final mesh export failed: "
                          << e.what() << "\n";
            }
        } else if (save_nvblox_mesh_eval_) {
            std::cout << "[NVBLOX] final mesh skipped: nvblox mapper was not initialized.\n";
        }
    }
    savePlannerNPZ(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "planner.npz");
    writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

    const auto rrd_dir  = result_dir_ / "rerun";
    std::filesystem::create_directories(rrd_dir);
    const auto rrd_path = (rrd_dir / "recording.rrd").string();
    sv::RerunVisualizerBridge::instance().saveRecording(rrd_path);

    signalStop();
 }

 // ---------- debug helpers ----------
static std::string shp(const torch::Tensor& t) {
    if (!t.defined()) return "<undef>";
    std::ostringstream oss; oss << '[';
    for (int i=0;i<t.dim();++i){ if(i) oss<<','; oss<<t.size(i); }
    oss << ']'; return oss.str();
}
static std::string dev(const torch::Tensor& t) {
    return t.defined() ? t.device().str() : "<undef>";
}
static void dump_param(const char* name, const torch::Tensor& p) {
    std::cout << "    " << name
              << " def=" << (p.defined()?1:0)
              << " req=" << (p.defined()?p.requires_grad():false)
              << " shape=" << shp(p)
              << " dtype=" << (p.defined()?c10::toString(p.scalar_type()):"<undef>")
              << " dev=" << dev(p);
    torch::Tensor g;
    try { if (p.defined()) g = p.grad(); } catch(...) {}
    std::cout << "  | grad def=" << (g.defined()?1:0)
              << " shape=" << shp(g) << "\n";
}

void VoxelMapper::savePhotometricErrorHeatmapAsPng(
    const torch::Tensor& error_tensor,    // [H,W] or [1,H,W] or [3,H,W]
    const std::filesystem::path& path)
{
    if (!error_tensor.defined()) {
        std::cout << "[PHOTO-ERR] tensor is undefined, skip save\n";
        return;
    }

    torch::Tensor e = error_tensor.detach()
                                     .to(torch::kCPU)
                                     .to(torch::kFloat32);

    // Handle shapes [C,H,W] → [H,W]
    if (e.dim() == 3) {
        const auto C = e.size(0);
        if (C == 1) {
            e = e.squeeze(0);          // [1,H,W] -> [H,W]
        } else if (C >= 3) {
            // Average over channels to get scalar error
            e = e.mean(0);             // [3,H,W] -> [H,W]
        } else {
            std::cout << "[PHOTO-ERR] unexpected 3D shape: "
                      << e.sizes() << ", skip save\n";
            return;
        }
    } else if (e.dim() != 2) {
        std::cout << "[PHOTO-ERR] expected [H,W] or [C,H,W], got dim="
                  << e.dim() << ", skip save\n";
        return;
    }

    e = e.contiguous();  // [H,W]

    // Mask valid pixels (error > 0); you can make this looser/tighter
    torch::Tensor mask = e > 0.0f;
    if (!mask.any().item<bool>()) {
        std::cout << "[PHOTO-ERR] all errors <= 0, skip save\n";
        return;
    }

    // Use quantiles to clip outliers (3%–97%), similar to depth viz
    torch::Tensor e_valid = e.masked_select(mask);
    if (e_valid.numel() < 10) {
        std::cout << "[PHOTO-ERR] too few valid pixels, skip save\n";
        return;
    }

    torch::Tensor q = torch::quantile(
        e_valid,
        torch::tensor({0.03f, 0.97f}, torch::kFloat32)
    );
    float e_min = q[0].item<float>();
    float e_max = q[1].item<float>();

    if (!(e_max > e_min)) {
        std::cout << "[PHOTO-ERR] invalid quantile range: min=" << e_min
                  << " max=" << e_max << ", skip save\n";
        return;
    }

    // Normalize to [0,1]
    torch::Tensor x = (e - e_min) / (e_max - e_min);
    x = x.clamp(0.0f, 1.0f);

    // Map to [0,255]
    torch::Tensor x_u8 = (x * 255.0f).to(torch::kUInt8).contiguous();
    int H = static_cast<int>(x_u8.size(0));
    int W = static_cast<int>(x_u8.size(1));

    cv::Mat gray(H, W, CV_8UC1, x_u8.data_ptr<uint8_t>());

    // Colorize with VIRIDIS
    cv::Mat color_bgr;
    cv::applyColorMap(gray, color_bgr, cv::COLORMAP_VIRIDIS);

    // Zero-out invalid pixels (where mask==0)
    torch::Tensor mask_u8 = mask.to(torch::kUInt8).contiguous();
    cv::Mat mask_cv(H, W, CV_8UC1, mask_u8.data_ptr<uint8_t>());
    color_bgr.setTo(cv::Scalar(0, 0, 0), mask_cv == 0);

    std::filesystem::create_directories(path.parent_path());
    cv::imwrite(path.string(), color_bgr);
    // if (!cv::imwrite(path.string(), color_bgr)) {
    //     std::cerr << "[PHOTO-ERR] failed to write PNG: " << path << "\n";
    // } else {
    //     std::cout << "[PHOTO-ERR] wrote heatmap PNG: " << path << "\n";
    // }
}

void VoxelMapper::saveDepthTensorAsPng(
    const torch::Tensor& depth_tensor,
    const std::filesystem::path& path)
{
    if (!depth_tensor.defined()) {
        std::cout << "[DEPTH VIZ] tensor is undefined, skip save\n";
        return;
    }

    torch::Tensor d = depth_tensor.detach().to(torch::kCPU).to(torch::kFloat32);
    if (d.dim() == 4 && d.size(0) == 1) {
        d = d.squeeze(0);
    }
    if (d.dim() == 3) {
        const auto C = d.size(0);
        if (C == 1) {
            d = d.squeeze(0);
        } else if (C >= 3) {
            d = d.index({2});
        } else if (C >= 1) {
            d = d.index({0});
        } else {
            return;
        }
    } else if (d.dim() != 2) {
        return;
    }

    d = d.contiguous();
    torch::Tensor mask = torch::isfinite(d) & (d > 0.0f);
    if (!mask.any().item<bool>()) {
        std::cout << "[DEPTH VIZ] all depth invalid, skip save\n";
        return;
    }

    torch::Tensor d_valid = d.masked_select(mask);
    torch::Tensor q = torch::quantile(
        d_valid,
        torch::tensor({0.03f, 0.97f}, torch::TensorOptions().dtype(torch::kFloat32))
    );
    float d_min = q[0].item<float>();
    float d_max = q[1].item<float>();
    if (!(d_max > d_min)) {
        d_min = d_valid.min().item<float>();
        d_max = d_valid.max().item<float>();
    }
    if (!(d_max > d_min)) {
        return;
    }

    torch::Tensor x = (d - d_min) / (d_max - d_min);
    x = x.clamp(0.0f, 1.0f);
    torch::Tensor x_u8 = (x * 255.0f).to(torch::kUInt8).contiguous();
    torch::Tensor mask_u8 = mask.to(torch::kUInt8).contiguous();

    const int H = static_cast<int>(x_u8.size(0));
    const int W = static_cast<int>(x_u8.size(1));
    cv::Mat gray(H, W, CV_8UC1, x_u8.data_ptr<uint8_t>());
    cv::Mat mask_cv(H, W, CV_8UC1, mask_u8.data_ptr<uint8_t>());

    cv::Mat color_bgr;
    cv::applyColorMap(gray, color_bgr, cv::COLORMAP_JET);
    color_bgr.setTo(cv::Scalar(0, 0, 0), mask_cv == 0);

    std::filesystem::create_directories(path.parent_path());
    cv::imwrite(path.string(), color_bgr);
    // if (!cv::imwrite(path.string(), final_img)) {
    //     std::cerr << "[DEPTH VIZ] failed to write depth PNG to: " 
    //               << path << "\n";
    // } else {
    //     std::cout << "[DEPTH VIZ] wrote depth PNG: " << path << "\n";
    // }
}

static torch::Tensor tensorToEvalMap(const torch::Tensor& tensor, int preferred_channel)
{
    if (!tensor.defined()) {
        return torch::Tensor();
    }

    torch::Tensor d = tensor.detach().to(torch::kFloat32);
    if (d.dim() == 4 && d.size(0) == 1) {
        d = d.squeeze(0);  // [1,C,H,W] -> [C,H,W]
    }
    if (d.dim() == 3) {
        const int64_t C = d.size(0);
        if (C == 1) {
            d = d.squeeze(0);  // [1,H,W] -> [H,W]
        } else if (C > preferred_channel && preferred_channel >= 0) {
            d = d.index({preferred_channel});  // preferred channel
        } else if (C >= 1) {
            d = d.index({0});  // fallback to channel 0
        } else {
            return torch::Tensor();
        }
    }
    if (d.dim() != 2) {
        return torch::Tensor();
    }
    return d.contiguous();
}

static torch::Tensor tensorToEvalMapExactChannel(
    const torch::Tensor& tensor,
    int required_channel)
{
    if (!tensor.defined()) {
        return torch::Tensor();
    }

    torch::Tensor d = tensor.detach().to(torch::kFloat32);
    if (d.dim() == 4 && d.size(0) == 1) {
        d = d.squeeze(0);  // [1,C,H,W] -> [C,H,W]
    }
    if (d.dim() == 3) {
        const int64_t C = d.size(0);
        if (C == 1) {
            if (required_channel != 0) {
                return torch::Tensor();
            }
            d = d.squeeze(0);  // [1,H,W] -> [H,W]
        } else {
            if (required_channel < 0 || required_channel >= C) {
                return torch::Tensor();
            }
            d = d.index({required_channel});
        }
    }
    if (d.dim() != 2) {
        return torch::Tensor();
    }
    return d.contiguous();
}

static torch::Tensor depthTensorToEvalMap(const torch::Tensor& depth_tensor)
{
    // For GT-vs-render depth visualization/evaluation, prefer the surface-like
    // median depth channel when available. This matches SVRaster mesh/fusion
    // utilities better than the alpha-weighted mean-depth channel.
    return tensorToEvalMap(depth_tensor, /*preferred_channel=*/2);
}

static torch::Tensor transmittanceTensorToEvalMap(const torch::Tensor& t_tensor)
{
    return tensorToEvalMap(t_tensor, /*preferred_channel=*/0);
}

static bool renderPkgToMetricDepthForEval(
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    torch::Tensor& pred_depth)
{
    pred_depth = torch::Tensor();

    auto it_depth = render_pkg.find("depth");
    if (it_depth == render_pkg.end() || !it_depth->second.defined()) {
        it_depth = render_pkg.find("raw_depth");
        if (it_depth == render_pkg.end() || !it_depth->second.defined()) {
            return false;
        }
    }

    pred_depth = depthTensorToEvalMap(it_depth->second);
    if (!pred_depth.defined()) {
        return false;
    }

    return true;
}

static bool renderPkgToSparseDepthLossMap(
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    torch::Tensor& pred_depth)
{
    pred_depth = torch::Tensor();

    auto it_T = render_pkg.find("raw_T");
    if (it_T == render_pkg.end() || !it_T->second.defined()) {
        it_T = render_pkg.find("T");
        if (it_T == render_pkg.end() || !it_T->second.defined()) {
            return false;
        }
    }

    auto it_depth = render_pkg.find("raw_depth");
    if (it_depth == render_pkg.end() || !it_depth->second.defined()) {
        it_depth = render_pkg.find("depth");
        if (it_depth == render_pkg.end() || !it_depth->second.defined()) {
            return false;
        }
    }

    torch::Tensor raw_depth = tensorToEvalMapExactChannel(it_depth->second, 0);
    torch::Tensor raw_T = transmittanceTensorToEvalMap(it_T->second);
    if (!raw_depth.defined() || !raw_T.defined()) {
        return false;
    }
    if (raw_depth.sizes() != raw_T.sizes()) {
        return false;
    }

    pred_depth = raw_depth / (1.0f - raw_T).clamp_min(1e-4f);
    return pred_depth.defined();
}

static bool renderPkgToMonoPriorDebugMaps(
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    const torch::Tensor& mono_depth,
    float cam_near,
    torch::Tensor& mono_prior_resized,
    torch::Tensor& aligned_target,
    torch::Tensor& rendered_loss_ref)
{
    mono_prior_resized = torch::Tensor();
    aligned_target = torch::Tensor();
    rendered_loss_ref = torch::Tensor();

    if (!mono_depth.defined() || mono_depth.numel() == 0) {
        return false;
    }

    auto it_T = render_pkg.find("raw_T");
    if (it_T == render_pkg.end() || !it_T->second.defined()) {
        it_T = render_pkg.find("T");
        if (it_T == render_pkg.end() || !it_T->second.defined()) {
            return false;
        }
    }

    auto it_depth = render_pkg.find("raw_depth");
    if (it_depth == render_pkg.end() || !it_depth->second.defined()) {
        it_depth = render_pkg.find("depth");
        if (it_depth == render_pkg.end() || !it_depth->second.defined()) {
            return false;
        }
    }

    torch::Tensor depth = it_depth->second.detach().to(torch::kFloat32).contiguous();
    if (depth.dim() == 4 && depth.size(0) == 1) {
        depth = depth.squeeze(0);
    }
    if (depth.dim() == 2) {
        depth = depth.unsqueeze(0);
    }
    if (depth.dim() != 3) {
        return false;
    }

    torch::Tensor T = it_T->second.detach().to(depth.device()).to(torch::kFloat32).contiguous();
    if (T.dim() == 4 && T.size(0) == 1) {
        T = T.squeeze(0);
    }
    if (T.dim() == 2) {
        T = T.unsqueeze(0);
    }
    if (T.dim() != 3) {
        return false;
    }

    torch::Tensor Y = mono_depth.detach().to(depth.device()).to(torch::kFloat32).contiguous();
    if (Y.dim() == 4 && Y.size(0) == 1) {
        Y = Y.squeeze(0);
    }
    if (Y.dim() == 3 && Y.size(0) == 1) {
        Y = Y.squeeze(0);
    }
    if (Y.dim() == 2) {
        Y = Y.unsqueeze(0).unsqueeze(0);
    } else if (Y.dim() == 3 && Y.size(0) == 1) {
        Y = Y.unsqueeze(0);
    }
    if (Y.dim() != 4) {
        return false;
    }

    torch::Tensor invdepth =
        1.0f / depth.unsqueeze(1).clamp_min(std::max(1e-6f, cam_near));
    const int64_t ref_idx = std::min<int64_t>(2, invdepth.size(0) - 1);
    torch::Tensor X = invdepth.index({0}).unsqueeze(0);
    torch::Tensor Xref = invdepth.index({ref_idx}).unsqueeze(0);
    torch::Tensor alpha = 1.0f - T.index({0}).unsqueeze(0).unsqueeze(0);

    if (Y.sizes().slice(2) != X.sizes().slice(2)) {
        Y = torch::nn::functional::interpolate(
            Y,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{X.size(2), X.size(3)})
                .mode(torch::kBilinear)
                .align_corners(false));
    }

    torch::Tensor target;
    {
        torch::NoGradGuard no_grad;
        const torch::Tensor Ymed = Y.median();
        const torch::Tensor Ys = (Y - Ymed).abs().mean().clamp_min(1e-6f);
        const torch::Tensor Xmed = Xref.median();
        const torch::Tensor Xs = (Xref - Xmed).abs().mean().clamp_min(1e-6f);
        target = (Y - Ymed) * (Xs / Ys) + Xmed;
    }

    torch::Tensor render_ref = X * alpha;
    torch::Tensor valid_mask = (target > 0.01f) & (alpha > 0.5f);
    const torch::Tensor nan_like = torch::full_like(render_ref, std::numeric_limits<float>::quiet_NaN());

    mono_prior_resized = Y.squeeze().detach().to(torch::kCPU).contiguous();
    aligned_target = torch::where(valid_mask, target, nan_like)
                         .squeeze()
                         .detach()
                         .to(torch::kCPU)
                         .contiguous();
    rendered_loss_ref = torch::where(valid_mask, render_ref, nan_like)
                            .squeeze()
                            .detach()
                            .to(torch::kCPU)
                            .contiguous();

    return mono_prior_resized.defined() &&
           aligned_target.defined() &&
           rendered_loss_ref.defined() &&
           mono_prior_resized.dim() == 2 &&
           aligned_target.dim() == 2 &&
           rendered_loss_ref.dim() == 2;
}

static bool computeSharedDepthVizRange(
    const torch::Tensor& pred_depth,
    const cv::Mat& gt_depth_meters,
    float valid_min_depth,
    float valid_max_depth,
    float& viz_min,
    float& viz_max)
{
    torch::Tensor pred = pred_depth.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor values = pred.masked_select(
        torch::isfinite(pred) &
        (pred > valid_min_depth) &
        (pred < valid_max_depth));

    if (!gt_depth_meters.empty()) {
        torch::Tensor gt = torch::from_blob(
            const_cast<float*>(gt_depth_meters.ptr<float>()),
            {gt_depth_meters.rows, gt_depth_meters.cols},
            torch::TensorOptions().dtype(torch::kFloat32)).clone();
        torch::Tensor gt_valid = gt.masked_select(
            torch::isfinite(gt) &
            (gt > valid_min_depth) &
            (gt < valid_max_depth));
        if (gt_valid.numel() > 0) {
            values = values.numel() > 0 ? torch::cat({values, gt_valid}, 0) : gt_valid;
        }
    }

    if (!values.defined() || values.numel() == 0) {
        return false;
    }

    if (values.numel() >= 10) {
        torch::Tensor q = torch::quantile(
            values,
            torch::tensor({0.03f, 0.97f}, torch::TensorOptions().dtype(torch::kFloat32)));
        viz_min = q[0].item<float>();
        viz_max = q[1].item<float>();
    } else {
        viz_min = values.min().item<float>();
        viz_max = values.max().item<float>();
    }

    if (!(viz_max > viz_min)) {
        viz_min = values.min().item<float>();
        viz_max = values.max().item<float>();
    }
    if (!(viz_max > viz_min)) {
        viz_max = viz_min + 1e-3f;
    }
    return true;
}

struct DepthScaleFitStats
{
    bool valid = false;
    int64_t overlap_count = 0;
    int64_t ratio_count_before_trim = 0;
    int64_t ratio_count_after_trim = 0;
    float scale = std::numeric_limits<float>::quiet_NaN();
    float ratio_q05 = std::numeric_limits<float>::quiet_NaN();
    float ratio_q25 = std::numeric_limits<float>::quiet_NaN();
    float ratio_q50 = std::numeric_limits<float>::quiet_NaN();
    float ratio_q75 = std::numeric_limits<float>::quiet_NaN();
    float ratio_q95 = std::numeric_limits<float>::quiet_NaN();
    float pred_min = std::numeric_limits<float>::quiet_NaN();
    float pred_max = std::numeric_limits<float>::quiet_NaN();
    float gt_min = std::numeric_limits<float>::quiet_NaN();
    float gt_max = std::numeric_limits<float>::quiet_NaN();
};

static bool computeDepthScaleFitStats(
    const torch::Tensor& pred_depth,
    const cv::Mat& gt_depth_meters,
    float valid_min_depth,
    float valid_max_depth,
    DepthScaleFitStats& stats_out)
{
    stats_out = DepthScaleFitStats{};
    if (gt_depth_meters.empty()) {
        return false;
    }

    torch::Tensor pred = pred_depth.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor gt = torch::from_blob(
        const_cast<float*>(gt_depth_meters.ptr<float>()),
        {gt_depth_meters.rows, gt_depth_meters.cols},
        torch::TensorOptions().dtype(torch::kFloat32)).clone();

    torch::Tensor valid =
        torch::isfinite(pred) &
        torch::isfinite(gt) &
        (pred > valid_min_depth) &
        (pred < valid_max_depth) &
        (gt > valid_min_depth) &
        (gt < valid_max_depth);

    stats_out.overlap_count = valid.sum().item<int64_t>();
    if (stats_out.overlap_count > 0) {
        torch::Tensor pred_valid = pred.masked_select(valid);
        torch::Tensor gt_valid = gt.masked_select(valid);
        stats_out.pred_min = pred_valid.min().item<float>();
        stats_out.pred_max = pred_valid.max().item<float>();
        stats_out.gt_min = gt_valid.min().item<float>();
        stats_out.gt_max = gt_valid.max().item<float>();
    }

    if (stats_out.overlap_count < 100) {
        return false;
    }

    torch::Tensor ratio = gt.masked_select(valid) / pred.masked_select(valid).clamp_min(1e-6f);
    ratio = ratio.masked_select(torch::isfinite(ratio) & (ratio > 0.0f));
    stats_out.ratio_count_before_trim = ratio.numel();
    if (stats_out.ratio_count_before_trim < 100) {
        return false;
    }

    if (ratio.numel() >= 10) {
        torch::Tensor q = torch::quantile(
            ratio,
            torch::tensor(
                {0.05f, 0.25f, 0.50f, 0.75f, 0.95f},
                torch::TensorOptions().dtype(torch::kFloat32)));
        const float q_lo = q[0].item<float>();
        const float q_hi = q[4].item<float>();
        stats_out.ratio_q05 = q_lo;
        stats_out.ratio_q25 = q[1].item<float>();
        stats_out.ratio_q50 = q[2].item<float>();
        stats_out.ratio_q75 = q[3].item<float>();
        stats_out.ratio_q95 = q_hi;
        ratio = ratio.masked_select((ratio >= q_lo) & (ratio <= q_hi));
    }
    stats_out.ratio_count_after_trim = ratio.numel();
    if (stats_out.ratio_count_after_trim == 0) {
        return false;
    }

    const float scale = ratio.median().item<float>();
    if (!std::isfinite(scale) || scale <= 0.0f) {
        return false;
    }

    stats_out.scale = scale;
    stats_out.valid = true;
    return true;
}

static bool computeWeightedMedianScale(
    const std::vector<std::pair<float, double>>& weighted_scales,
    float& scale_out)
{
    scale_out = 1.0f;
    if (weighted_scales.empty()) {
        return false;
    }

    std::vector<std::pair<float, double>> sorted = weighted_scales;
    std::sort(
        sorted.begin(),
        sorted.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    double total_weight = 0.0;
    for (const auto& item : sorted) {
        total_weight += std::max(0.0, item.second);
    }
    if (!(total_weight > 0.0)) {
        return false;
    }

    const double half_weight = 0.5 * total_weight;
    double accum_weight = 0.0;
    for (const auto& item : sorted) {
        accum_weight += std::max(0.0, item.second);
        if (accum_weight >= half_weight) {
            scale_out = item.first;
            return std::isfinite(scale_out) && scale_out > 0.0f;
        }
    }

    scale_out = sorted.back().first;
    return std::isfinite(scale_out) && scale_out > 0.0f;
}

static cv::Mat depthTensorToCvMatFloat(const torch::Tensor& depth_tensor)
{
    torch::Tensor d = depth_tensor.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    CV_Assert(d.dim() == 2);
    cv::Mat depth_view(
        static_cast<int>(d.size(0)),
        static_cast<int>(d.size(1)),
        CV_32FC1,
        d.data_ptr<float>());
    return depth_view.clone();
}

static cv::Mat colorizeDepthMatJet(
    const cv::Mat& depth_meters,
    float valid_min_depth,
    float valid_max_depth,
    float viz_min,
    float viz_max)
{
    CV_Assert(depth_meters.type() == CV_32FC1);
    const int H = depth_meters.rows;
    const int W = depth_meters.cols;

    cv::Mat gray(H, W, CV_8UC1, cv::Scalar(0));
    cv::Mat valid_mask(H, W, CV_8UC1, cv::Scalar(0));

    const float denom = std::max(1e-6f, viz_max - viz_min);
    for (int y = 0; y < H; ++y) {
        const float* src = depth_meters.ptr<float>(y);
        uint8_t* gray_ptr = gray.ptr<uint8_t>(y);
        uint8_t* mask_ptr = valid_mask.ptr<uint8_t>(y);
        for (int x = 0; x < W; ++x) {
            const float d = src[x];
            if (!std::isfinite(d) || d <= valid_min_depth || d >= valid_max_depth) {
                continue;
            }
            const float norm = std::clamp((d - viz_min) / denom, 0.0f, 1.0f);
            gray_ptr[x] = static_cast<uint8_t>(std::round(norm * 255.0f));
            mask_ptr[x] = 255;
        }
    }

    cv::Mat color_bgr;
    cv::applyColorMap(gray, color_bgr, cv::COLORMAP_JET);
    color_bgr.setTo(cv::Scalar(0, 0, 0), valid_mask == 0);
    return color_bgr;
}

static cv::Mat colorizeFiniteScalarMatJet(
    const cv::Mat& values,
    float viz_min,
    float viz_max)
{
    CV_Assert(values.type() == CV_32FC1);
    const int H = values.rows;
    const int W = values.cols;

    cv::Mat gray(H, W, CV_8UC1, cv::Scalar(0));
    cv::Mat valid_mask(H, W, CV_8UC1, cv::Scalar(0));

    const float denom = std::max(1e-6f, viz_max - viz_min);
    for (int y = 0; y < H; ++y) {
        const float* src = values.ptr<float>(y);
        uint8_t* gray_ptr = gray.ptr<uint8_t>(y);
        uint8_t* mask_ptr = valid_mask.ptr<uint8_t>(y);
        for (int x = 0; x < W; ++x) {
            const float v = src[x];
            if (!std::isfinite(v)) {
                continue;
            }
            const float norm = std::clamp((v - viz_min) / denom, 0.0f, 1.0f);
            gray_ptr[x] = static_cast<uint8_t>(std::round(norm * 255.0f));
            mask_ptr[x] = 255;
        }
    }

    cv::Mat color_bgr;
    cv::applyColorMap(gray, color_bgr, cv::COLORMAP_JET);
    color_bgr.setTo(cv::Scalar(0, 0, 0), valid_mask == 0);
    return color_bgr;
}

static cv::Mat appendJetLegendBar(
    const cv::Mat& image_bgr,
    float viz_min,
    float viz_max,
    const std::string& unit_suffix)
{
    const int H = image_bgr.rows;
    const int bar_w = 24;
    const int pad = 8;
    const int legend_w = 180;

    cv::Mat gray(H, bar_w, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < H; ++y) {
        const float t = (H > 1) ? (1.0f - static_cast<float>(y) / static_cast<float>(H - 1)) : 1.0f;
        gray.row(y).setTo(cv::Scalar(static_cast<uint8_t>(std::round(std::clamp(t, 0.0f, 1.0f) * 255.0f))));
    }

    cv::Mat bar_bgr;
    cv::applyColorMap(gray, bar_bgr, cv::COLORMAP_JET);

    cv::Mat legend(H, legend_w, CV_8UC3, cv::Scalar(0, 0, 0));
    bar_bgr.copyTo(legend(cv::Rect(pad, 0, bar_w, H)));

    const int text_x = pad + bar_w + 10;
    const double font_scale = 0.5;
    const int thickness = 1;
    const cv::Scalar white(255, 255, 255);
    const std::string max_text = "max " + std::string(cv::format("%.3f", viz_max)) + unit_suffix;
    const std::string min_text = "min " + std::string(cv::format("%.3f", viz_min)) + unit_suffix;

    cv::putText(legend, max_text, {text_x, 20}, cv::FONT_HERSHEY_SIMPLEX, font_scale, white, thickness, cv::LINE_AA);
    cv::putText(legend, "red high", {text_x, 40}, cv::FONT_HERSHEY_SIMPLEX, font_scale, white, thickness, cv::LINE_AA);
    cv::putText(legend, "blue low", {text_x, std::max(20, H - 24)}, cv::FONT_HERSHEY_SIMPLEX, font_scale, white, thickness, cv::LINE_AA);
    cv::putText(legend, min_text, {text_x, std::max(16, H - 6)}, cv::FONT_HERSHEY_SIMPLEX, font_scale, white, thickness, cv::LINE_AA);

    cv::Mat out;
    cv::hconcat(std::vector<cv::Mat>{image_bgr, legend}, out);
    return out;
}

static bool renderPkgToNormalForEval(
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    torch::Tensor& render_normal)
{
    render_normal = torch::Tensor();

    auto it_normal = render_pkg.find("raw_normal");
    if (it_normal == render_pkg.end() || !it_normal->second.defined()) {
        it_normal = render_pkg.find("normal");
        if (it_normal == render_pkg.end() || !it_normal->second.defined()) {
            return false;
        }
    }

    render_normal = it_normal->second.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (render_normal.dim() == 4 && render_normal.size(0) == 1) {
        render_normal = render_normal.squeeze(0);
    }
    if (render_normal.dim() != 3 || render_normal.size(0) < 3) {
        render_normal = torch::Tensor();
        return false;
    }
    if (render_normal.size(0) > 3) {
        render_normal = render_normal.index({torch::indexing::Slice(0, 3)}).contiguous();
    }
    return true;
}

static cv::Mat colorizeNormalMapBgr(const torch::Tensor& normal_tensor)
{
    torch::Tensor normal = normal_tensor.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (normal.dim() != 3 || normal.size(0) < 3) {
        return cv::Mat();
    }
    if (normal.size(0) > 3) {
        normal = normal.index({torch::indexing::Slice(0, 3)}).contiguous();
    }

    const int H = static_cast<int>(normal.size(1));
    const int W = static_cast<int>(normal.size(2));
    cv::Mat out(H, W, CV_8UC3, cv::Scalar(0, 0, 0));
    auto acc = normal.accessor<float, 3>();
    for (int y = 0; y < H; ++y) {
        auto* row = out.ptr<cv::Vec3b>(y);
        for (int x = 0; x < W; ++x) {
            float nx = acc[0][y][x];
            float ny = acc[1][y][x];
            float nz = acc[2][y][x];
            if (!std::isfinite(nx) || !std::isfinite(ny) || !std::isfinite(nz)) {
                continue;
            }
            const float norm = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (!(norm > 1e-6f)) {
                continue;
            }
            nx /= norm;
            ny /= norm;
            nz /= norm;
            row[x] = cv::Vec3b(
                static_cast<uint8_t>(std::round(std::clamp(0.5f * (nz + 1.0f), 0.0f, 1.0f) * 255.0f)),
                static_cast<uint8_t>(std::round(std::clamp(0.5f * (ny + 1.0f), 0.0f, 1.0f) * 255.0f)),
                static_cast<uint8_t>(std::round(std::clamp(0.5f * (nx + 1.0f), 0.0f, 1.0f) * 255.0f)));
        }
    }
    return out;
}

static bool sparseSamplesToDepthMat(
    const torch::Tensor& sparse_uv,
    const torch::Tensor& sparse_depth,
    int image_width,
    int image_height,
    cv::Mat& depth_meters)
{
    depth_meters = cv::Mat(
        image_height,
        image_width,
        CV_32FC1,
        cv::Scalar(std::numeric_limits<float>::quiet_NaN()));

    if (!sparse_uv.defined() || !sparse_depth.defined()) {
        return false;
    }
    if (sparse_uv.dim() != 2 || sparse_uv.size(1) != 2) {
        return false;
    }

    torch::Tensor uv = sparse_uv.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor depth = sparse_depth.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (depth.dim() == 2 && depth.size(1) == 1) {
        depth = depth.squeeze(1);
    }
    if (depth.dim() != 1 || depth.size(0) != uv.size(0)) {
        return false;
    }

    const auto uv_acc = uv.accessor<float, 2>();
    const auto depth_acc = depth.accessor<float, 1>();
    int64_t n_written = 0;
    for (int64_t i = 0; i < uv.size(0); ++i) {
        const float z = depth_acc[i];
        if (!std::isfinite(z) || z <= 0.0f) {
            continue;
        }

        const float px_f = 0.5f * (uv_acc[i][0] + 1.0f) * static_cast<float>(image_width);
        const float py_f = 0.5f * (uv_acc[i][1] + 1.0f) * static_cast<float>(image_height);
        const int px = static_cast<int>(std::lround(px_f));
        const int py = static_cast<int>(std::lround(py_f));
        if (px < 0 || px >= image_width || py < 0 || py >= image_height) {
            continue;
        }

        float& cell = depth_meters.at<float>(py, px);
        if (!std::isfinite(cell) || z < cell) {
            cell = z;
        }
        ++n_written;
    }

    return n_written > 0;
}

static const std::string& runtimeOrbDepthDebugRunTag()
{
    static const std::string tag = []() {
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        return std::to_string(now);
    }();
    return tag;
}

static std::filesystem::path runtimeOrbDepthDebugDir(
    const std::filesystem::path& result_root)
{
    return result_root / (".orb_depth_debug_" + runtimeOrbDepthDebugRunTag());
}

static void copyPngFilesToDirectory(
    const std::filesystem::path& source_dir,
    const std::filesystem::path& target_dir)
{
    if (source_dir.empty() || target_dir.empty() ||
        !std::filesystem::exists(source_dir) ||
        !std::filesystem::is_directory(source_dir)) {
        return;
    }

    std::filesystem::create_directories(target_dir);
    for (const auto& entry : std::filesystem::directory_iterator(source_dir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".png") {
            continue;
        }
        std::error_code ec;
        std::filesystem::copy_file(
            entry.path(),
            target_dir / entry.path().filename(),
            std::filesystem::copy_options::overwrite_existing,
            ec);
    }
}

static cv::Scalar relDepthErrorColorBgr(float rel_error)
{
    if (!std::isfinite(rel_error)) {
        return cv::Scalar(255, 0, 255);
    }
    if (rel_error <= 0.05f) {
        return cv::Scalar(0, 220, 0);
    }
    if (rel_error <= 0.15f) {
        return cv::Scalar(0, 220, 220);
    }
    if (rel_error <= 0.30f) {
        return cv::Scalar(0, 140, 255);
    }
    return cv::Scalar(0, 0, 255);
}

static bool saveAccumulatedOrbDepthProjectionPng(
    const std::shared_ptr<ORB_SLAM3::System>& slam,
    const std::shared_ptr<VoxelKeyframe>& kf,
    const std::filesystem::path& output_dir,
    int iteration,
    float valid_min_depth,
    float valid_max_depth,
    const torch::Tensor& aligned_da_depth = torch::Tensor())
{
    if (!slam || !kf || output_dir.empty() ||
        kf->image_width_ <= 0 || kf->image_height_ <= 0 || kf->intr_.size() < 4) {
        return false;
    }

    ORB_SLAM3::Atlas* atlas = slam->getAtlas();
    if (!atlas) {
        return false;
    }

    const std::vector<ORB_SLAM3::MapPoint*> map_points = atlas->GetAllMapPoints();
    if (map_points.empty()) {
        return false;
    }

    const int W = kf->image_width_;
    const int H = kf->image_height_;
    const float fx = kf->intr_[0];
    const float fy = kf->intr_[1];
    const float cx = kf->intr_[2];
    const float cy = kf->intr_[3];
    if (!(fx > 1e-6f) || !(fy > 1e-6f)) {
        return false;
    }

    const Sophus::SE3f Tcw = kf->getPosef();
    const float z_min = std::max(valid_min_depth, std::max(kf->znear_, 1e-6f));
    const bool use_z_max = std::isfinite(valid_max_depth) && valid_max_depth > z_min;
    const float viz_valid_max_depth = use_z_max ? valid_max_depth : 1e6f;

    cv::Mat depth_meters(H, W, CV_32FC1, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    int64_t projected_points = 0;
    for (ORB_SLAM3::MapPoint* mp : map_points) {
        if (!mp || mp->isBad()) {
            continue;
        }

        const Eigen::Vector3f p_world = mp->GetWorldPos();
        if (!std::isfinite(p_world.x()) ||
            !std::isfinite(p_world.y()) ||
            !std::isfinite(p_world.z())) {
            continue;
        }

        const Eigen::Vector3f p_cam = Tcw * p_world;
        const float z = p_cam.z();
        if (!std::isfinite(z) || z <= z_min || (use_z_max && z >= valid_max_depth)) {
            continue;
        }

        const float u = fx * p_cam.x() / z + cx;
        const float v = fy * p_cam.y() / z + cy;
        if (!std::isfinite(u) || !std::isfinite(v)) {
            continue;
        }

        const int px = static_cast<int>(std::lround(u));
        const int py = static_cast<int>(std::lround(v));
        if (px < 0 || px >= W || py < 0 || py >= H) {
            continue;
        }

        float& dst = depth_meters.at<float>(py, px);
        if (!std::isfinite(dst) || z < dst) {
            dst = z;
        }
        ++projected_points;
    }

    if (projected_points <= 0) {
        return false;
    }

    const torch::Tensor depth_tensor = torch::from_blob(
        depth_meters.data,
        {H, W},
        torch::TensorOptions().dtype(torch::kFloat32)).clone();
    float viz_min = 0.0f;
    float viz_max = 1.0f;
    if (!computeSharedDepthVizRange(
            depth_tensor,
            cv::Mat(),
            valid_min_depth,
            viz_valid_max_depth,
            viz_min,
            viz_max)) {
        return false;
    }

    const cv::Mat depth_bgr = colorizeDepthMatJet(
        depth_meters,
        valid_min_depth,
        viz_valid_max_depth,
        viz_min,
        viz_max);

    std::ostringstream stem;
    stem << "kf_" << std::setw(5) << std::setfill('0') << kf->fid_;
    std::ostringstream iter_tag;
    iter_tag << "_densification_iter_"
             << std::setw(5) << std::setfill('0') << iteration;

    std::filesystem::create_directories(output_dir);
    bool wrote_any = cv::imwrite(
        (output_dir / (stem.str() + "_orb_depth" + iter_tag.str() + ".png")).string(),
        depth_bgr);

    if (!aligned_da_depth.defined()) {
        return wrote_any;
    }

    cv::Mat aligned_da_meters = depthTensorToCvMatFloat(aligned_da_depth);
    if (aligned_da_meters.empty()) {
        return wrote_any;
    }
    if (aligned_da_meters.rows != H || aligned_da_meters.cols != W) {
        cv::resize(
            aligned_da_meters,
            aligned_da_meters,
            cv::Size(W, H),
            0.0,
            0.0,
            cv::INTER_LINEAR);
    }

    float da_viz_min = 0.0f;
    float da_viz_max = 1.0f;
    const torch::Tensor aligned_da_tensor = torch::from_blob(
        aligned_da_meters.data,
        {H, W},
        torch::TensorOptions().dtype(torch::kFloat32)).clone();
    cv::Mat aligned_da_bgr;
    if (computeSharedDepthVizRange(
            aligned_da_tensor,
            cv::Mat(),
            valid_min_depth,
            viz_valid_max_depth,
            da_viz_min,
            da_viz_max)) {
        aligned_da_bgr = colorizeDepthMatJet(
            aligned_da_meters,
            valid_min_depth,
            viz_valid_max_depth,
            da_viz_min,
            da_viz_max);
    } else {
        aligned_da_bgr = cv::Mat(H, W, CV_8UC3, cv::Scalar(0, 0, 0));
    }

    cv::Mat rel_error(H, W, CV_32FC1, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    cv::Mat orb_on_da = aligned_da_bgr.clone();

    const int radius = std::clamp(std::min(W, H) / 260, 2, 4);
    int64_t compared_points = 0;
    for (int y = 0; y < H; ++y) {
        const float* orb_row = depth_meters.ptr<float>(y);
        const float* da_row = aligned_da_meters.ptr<float>(y);
        float* err_row = rel_error.ptr<float>(y);
        for (int x = 0; x < W; ++x) {
            const float z_orb = orb_row[x];
            if (!std::isfinite(z_orb) || z_orb <= valid_min_depth ||
                z_orb >= viz_valid_max_depth) {
                continue;
            }

            const float z_da = da_row[x];
            float rel = std::numeric_limits<float>::quiet_NaN();
            if (std::isfinite(z_da) && z_da > valid_min_depth &&
                z_da < viz_valid_max_depth) {
                rel = std::abs(z_orb - z_da) / std::max(z_orb, 1e-6f);
                err_row[x] = rel;
                ++compared_points;
            }

            cv::circle(
                orb_on_da,
                cv::Point(x, y),
                radius,
                relDepthErrorColorBgr(rel),
                cv::FILLED,
                cv::LINE_AA);
        }
    }

    wrote_any |= cv::imwrite(
        (output_dir / (stem.str() + "_orb_on_da_error" + iter_tag.str() + ".png")).string(),
        orb_on_da);

    if (compared_points > 0) {
        const cv::Mat rel_error_bgr = appendJetLegendBar(
            colorizeFiniteScalarMatJet(rel_error, 0.0f, 0.5f),
            0.0f,
            0.5f,
            " rel");
        wrote_any |= cv::imwrite(
            (output_dir / (stem.str() + "_orb_vs_da_rel_error" + iter_tag.str() + ".png")).string(),
            rel_error_bgr);
    }

    return wrote_any;
}

static bool sampleDenseDepthAtSparseUv(
    const torch::Tensor& dense_depth,
    const torch::Tensor& sparse_uv,
    torch::Tensor& sampled_depth)
{
    sampled_depth = torch::Tensor();
    if (!dense_depth.defined() || !sparse_uv.defined()) {
        return false;
    }

    torch::Tensor depth = dense_depth.detach().to(torch::kFloat32).contiguous();
    if (depth.dim() != 2) {
        return false;
    }
    if (sparse_uv.dim() != 2 || sparse_uv.size(1) != 2 || sparse_uv.size(0) < 1) {
        return false;
    }

    torch::Tensor depth_img = depth.unsqueeze(0).unsqueeze(0);        // [1,1,H,W]
    torch::Tensor grid = sparse_uv.detach()
        .to(depth.device())
        .to(torch::kFloat32)
        .contiguous()
        .unsqueeze(0)
        .unsqueeze(0);                                                 // [1,1,N,2]

    auto gs_opts = torch::nn::functional::GridSampleFuncOptions()
                       .mode(torch::kBilinear)
                       .padding_mode(torch::kZeros)
                       .align_corners(false);

    sampled_depth = torch::nn::functional::grid_sample(depth_img, grid, gs_opts).squeeze();
    if (!sampled_depth.defined()) {
        return false;
    }
    if (sampled_depth.dim() == 0) {
        sampled_depth = sampled_depth.unsqueeze(0);
    }
    return sampled_depth.dim() == 1;
}

static bool fitAffineScaleShift1D(
    const torch::Tensor& x,
    const torch::Tensor& y,
    float& scale,
    float& shift)
{
    scale = std::numeric_limits<float>::quiet_NaN();
    shift = std::numeric_limits<float>::quiet_NaN();
    if (!x.defined() || !y.defined() || x.numel() < 2 || y.numel() != x.numel()) {
        return false;
    }

    const torch::Tensor x_mean = x.mean();
    const torch::Tensor y_mean = y.mean();
    const torch::Tensor dx = x - x_mean;
    const torch::Tensor dy = y - y_mean;
    const torch::Tensor denom = (dx * dx).sum();
    const float denom_f = denom.item<float>();
    if (!std::isfinite(denom_f) || denom_f <= 1e-12f) {
        return false;
    }

    const torch::Tensor scale_t = (dx * dy).sum() / denom;
    const torch::Tensor shift_t = y_mean - scale_t * x_mean;
    scale = scale_t.item<float>();
    shift = shift_t.item<float>();
    return std::isfinite(scale) && std::isfinite(shift);
}

static bool fitScale1D(
    const torch::Tensor& x,
    const torch::Tensor& y,
    float& scale)
{
    scale = std::numeric_limits<float>::quiet_NaN();
    if (!x.defined() || !y.defined() || x.numel() < 2 || y.numel() != x.numel()) {
        return false;
    }

    const torch::Tensor denom = (x * x).sum();
    const float denom_f = denom.item<float>();
    if (!std::isfinite(denom_f) || denom_f <= 1e-12f) {
        return false;
    }

    const torch::Tensor scale_t = (x * y).sum() / denom;
    scale = scale_t.item<float>();
    return std::isfinite(scale) && scale > 0.0f;
}

static float quantileVector(std::vector<float> values, float q)
{
    values.erase(
        std::remove_if(
            values.begin(),
            values.end(),
            [](float v) { return !std::isfinite(v); }),
        values.end());
    if (values.empty()) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    q = std::clamp(q, 0.0f, 1.0f);
    std::sort(values.begin(), values.end());
    const float pos = q * static_cast<float>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) {
        return values[lo];
    }
    const float t = pos - static_cast<float>(lo);
    return values[lo] * (1.0f - t) + values[hi] * t;
}

static bool fitAffineScaleShiftWeighted1D(
    const std::vector<float>& x,
    const std::vector<float>& y,
    const std::vector<float>* weights,
    float& scale,
    float& shift)
{
    scale = std::numeric_limits<float>::quiet_NaN();
    shift = std::numeric_limits<float>::quiet_NaN();
    if (x.size() != y.size() || x.size() < 2) {
        return false;
    }

    double w_sum = 0.0;
    double x_sum = 0.0;
    double y_sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) {
            continue;
        }
        const double w = weights ? static_cast<double>((*weights)[i]) : 1.0;
        if (!(w > 0.0) || !std::isfinite(w)) {
            continue;
        }
        w_sum += w;
        x_sum += w * static_cast<double>(x[i]);
        y_sum += w * static_cast<double>(y[i]);
    }
    if (!(w_sum > 0.0)) {
        return false;
    }

    const double x_mean = x_sum / w_sum;
    const double y_mean = y_sum / w_sum;
    double denom = 0.0;
    double numer = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) {
            continue;
        }
        const double w = weights ? static_cast<double>((*weights)[i]) : 1.0;
        if (!(w > 0.0) || !std::isfinite(w)) {
            continue;
        }
        const double dx = static_cast<double>(x[i]) - x_mean;
        const double dy = static_cast<double>(y[i]) - y_mean;
        denom += w * dx * dx;
        numer += w * dx * dy;
    }
    if (!(denom > 1e-12) || !std::isfinite(denom)) {
        return false;
    }

    const double s = numer / denom;
    const double t = y_mean - s * x_mean;
    scale = static_cast<float>(s);
    shift = static_cast<float>(t);
    return std::isfinite(scale) && std::isfinite(shift);
}

static bool applyDepthAnythingAffineAlignment(
    const torch::Tensor& mono_prior,
    float scale,
    float shift,
    torch::Tensor& aligned_depth)
{
    aligned_depth = torch::Tensor();
    if (!mono_prior.defined() ||
        !std::isfinite(scale) ||
        !std::isfinite(shift) ||
        scale <= 0.0f) {
        return false;
    }

    torch::Tensor mono = mono_prior.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (mono.dim() == 3 && mono.size(0) == 1) {
        mono = mono.squeeze(0);
    }
    if (mono.dim() != 2) {
        return false;
    }

    torch::Tensor aligned_inv = mono * scale + shift;
    const torch::Tensor aligned_valid = torch::isfinite(aligned_inv) & (aligned_inv > 1e-6f);
    aligned_depth = torch::where(
        aligned_valid,
        1.0f / aligned_inv.clamp_min(1e-6f),
        torch::full_like(aligned_inv, std::numeric_limits<float>::quiet_NaN()));
    aligned_depth = aligned_depth.to(torch::kCPU).contiguous();
    return aligned_depth.defined() && aligned_depth.dim() == 2;
}

static bool applyMetricDepthAlignment(
    const torch::Tensor& mono_prior,
    float scale,
    float shift,
    torch::Tensor& aligned_depth)
{
    aligned_depth = torch::Tensor();
    if (!mono_prior.defined() ||
        !std::isfinite(scale) ||
        !std::isfinite(shift) ||
        scale <= 0.0f) {
        return false;
    }

    torch::Tensor mono = mono_prior.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (mono.dim() == 3 && mono.size(0) == 1) {
        mono = mono.squeeze(0);
    }
    if (mono.dim() != 2) {
        return false;
    }

    aligned_depth = mono * scale + shift;
    const torch::Tensor aligned_valid = torch::isfinite(aligned_depth) & (aligned_depth > 1e-6f);
    aligned_depth = torch::where(
        aligned_valid,
        aligned_depth,
        torch::full_like(aligned_depth, std::numeric_limits<float>::quiet_NaN()));
    aligned_depth = aligned_depth.to(torch::kCPU).contiguous();
    return aligned_depth.defined() && aligned_depth.dim() == 2;
}

static bool alignDepthAnythingPriorToSparseAnchors(
    const torch::Tensor& mono_prior,
    const torch::Tensor& sparse_uv,
    const torch::Tensor& sparse_depth,
    float cam_near,
    int min_sparse_anchors,
    torch::Tensor& aligned_depth,
    MonoPriorAlignmentStats& stats)
{
    aligned_depth = torch::Tensor();
    stats = MonoPriorAlignmentStats();

    if (!mono_prior.defined() || !sparse_uv.defined() || !sparse_depth.defined()) {
        return false;
    }

    torch::Tensor mono = mono_prior.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (mono.dim() == 3 && mono.size(0) == 1) {
        mono = mono.squeeze(0);
    }
    if (mono.dim() != 2) {
        return false;
    }

    torch::Tensor mono_samples;
    if (!sampleDenseDepthAtSparseUv(mono, sparse_uv, mono_samples)) {
        return false;
    }

    torch::Tensor sparse_depth_1d = sparse_depth.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (sparse_depth_1d.dim() == 2 && sparse_depth_1d.size(1) == 1) {
        sparse_depth_1d = sparse_depth_1d.squeeze(1);
    }
    if (sparse_depth_1d.dim() != 1 || sparse_depth_1d.size(0) != mono_samples.size(0)) {
        return false;
    }

    const float near_depth = std::max(1e-6f, cam_near);
    torch::Tensor valid =
        torch::isfinite(mono_samples) &
        torch::isfinite(sparse_depth_1d) &
        (mono_samples > 0.0f) &
        (sparse_depth_1d > near_depth);
    stats.num_valid_anchors = valid.sum().item<int64_t>();
    if (stats.num_valid_anchors < std::max(2, min_sparse_anchors)) {
        return false;
    }

    torch::Tensor Y = mono_samples.masked_select(valid).contiguous();
    torch::Tensor sparse_depth_valid = sparse_depth_1d.masked_select(valid);
    torch::Tensor Xref = 1.0f / sparse_depth_valid.clamp_min(near_depth);

    // Use the old robust median/MAD alignment only to identify obvious outlier anchors.
    const torch::Tensor Ymed = Y.median();
    const torch::Tensor Ys = (Y - Ymed).abs().mean().clamp_min(1e-6f);
    const torch::Tensor Xmed = Xref.median();
    const torch::Tensor Xs = (Xref - Xmed).abs().mean().clamp_min(1e-6f);
    const torch::Tensor init_inv = (Y - Ymed) * (Xs / Ys) + Xmed;
    const torch::Tensor init_depth = 1.0f / init_inv.clamp_min(1e-6f);
    const torch::Tensor init_rel_err =
        (init_depth - sparse_depth_valid).abs() / sparse_depth_valid.clamp_min(near_depth);
    torch::Tensor fit_mask =
        torch::isfinite(init_rel_err) &
        torch::isfinite(init_inv) &
        (init_inv > 1e-6f) &
        (init_rel_err <= kMonoPriorAlignInitialInlierRelErr);
    stats.num_fit_anchors = fit_mask.sum().item<int64_t>();
    if (stats.num_fit_anchors < std::max(2, min_sparse_anchors)) {
        fit_mask = torch::ones_like(fit_mask, torch::TensorOptions().dtype(torch::kBool));
        stats.num_fit_anchors = stats.num_valid_anchors;
    }

    const torch::Tensor Y_fit = Y.masked_select(fit_mask).contiguous();
    const torch::Tensor Xref_fit = Xref.masked_select(fit_mask).contiguous();
    if (!fitAffineScaleShift1D(Y_fit, Xref_fit, stats.scale, stats.shift)) {
        return false;
    }
    if (stats.scale <= 0.0f) {
        return false;
    }

    const torch::Tensor aligned_anchor_inv = Y * stats.scale + stats.shift;
    const torch::Tensor positive_anchor_inv =
        torch::isfinite(aligned_anchor_inv) & (aligned_anchor_inv > 1e-6f);
    if (positive_anchor_inv.sum().item<int64_t>() < std::max(2, min_sparse_anchors)) {
        return false;
    }
    const torch::Tensor aligned_anchor_depth =
        1.0f / aligned_anchor_inv.masked_select(positive_anchor_inv).clamp_min(1e-6f);
    const torch::Tensor sparse_depth_eval =
        sparse_depth_valid.masked_select(positive_anchor_inv).clamp_min(near_depth);
    torch::Tensor rel_err =
        (aligned_anchor_depth - sparse_depth_eval).abs() / sparse_depth_eval;
    rel_err = rel_err.masked_select(torch::isfinite(rel_err)).contiguous();
    if (rel_err.numel() < std::max(2, min_sparse_anchors)) {
        return false;
    }
    stats.median_rel_depth_error = rel_err.median().item<float>();
    if (rel_err.numel() >= 10) {
        stats.p90_rel_depth_error = torch::quantile(
            rel_err,
            torch::tensor(0.90f, torch::TensorOptions().dtype(torch::kFloat32)))
            .item<float>();
    } else {
        stats.p90_rel_depth_error = rel_err.max().item<float>();
    }
    stats.final_inlier_ratio =
        static_cast<float>((rel_err <= kMonoPriorAlignFinalInlierRelErr)
                               .sum()
                               .item<int64_t>()) /
        static_cast<float>(stats.num_valid_anchors);
    if (!std::isfinite(stats.median_rel_depth_error) ||
        !std::isfinite(stats.p90_rel_depth_error) ||
        !std::isfinite(stats.final_inlier_ratio)) {
        return false;
    }

    if (!applyDepthAnythingAffineAlignment(mono, stats.scale, stats.shift, aligned_depth)) {
        return false;
    }

    if (sparse_depth_valid.numel() >= 10) {
        torch::Tensor q = torch::quantile(
            sparse_depth_valid,
            torch::tensor({0.05f, 0.95f}, torch::TensorOptions().dtype(torch::kFloat32)));
        stats.sparse_depth_q05 = q[0].item<float>();
        stats.sparse_depth_q95 = q[1].item<float>();
    } else {
        stats.sparse_depth_q05 = sparse_depth_valid.min().item<float>();
        stats.sparse_depth_q95 = sparse_depth_valid.max().item<float>();
    }

    return aligned_depth.defined() &&
           aligned_depth.dim() == 2 &&
           std::isfinite(stats.sparse_depth_q05) &&
           std::isfinite(stats.sparse_depth_q95) &&
           stats.sparse_depth_q95 > stats.sparse_depth_q05;
}

static bool alignDepthAnythingPriorToRenderedDepth(
    const torch::Tensor& mono_prior,
    const torch::Tensor& rendered_depth,
    const torch::Tensor& rendered_alpha,
    const torch::Tensor& rendered_n_contrib,
    float cam_near,
    int min_rendered_anchors,
    torch::Tensor& aligned_depth,
    MonoPriorAlignmentStats& stats)
{
    aligned_depth = torch::Tensor();
    stats = MonoPriorAlignmentStats();

    if (!mono_prior.defined() || !rendered_depth.defined() ||
        !rendered_alpha.defined() || !rendered_n_contrib.defined()) {
        return false;
    }

    torch::Tensor mono =
        mono_prior.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (mono.dim() == 3 && mono.size(0) == 1) {
        mono = mono.squeeze(0);
    }
    torch::Tensor depth =
        rendered_depth.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor alpha =
        rendered_alpha.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor n_contrib =
        rendered_n_contrib.detach().to(torch::kCPU).to(torch::kInt32).contiguous();
    if (mono.dim() != 2 || depth.dim() != 2 || alpha.dim() != 2 ||
        n_contrib.dim() != 2 ||
        mono.sizes() != depth.sizes() ||
        mono.sizes() != alpha.sizes() ||
        mono.sizes() != n_contrib.sizes()) {
        return false;
    }

    const float near_depth = std::max(1e-6f, cam_near);
    torch::Tensor valid =
        torch::isfinite(mono) &
        torch::isfinite(depth) &
        torch::isfinite(alpha) &
        (mono > 0.0f) &
        (depth > near_depth) &
        (alpha > 0.5f) &
        (n_contrib > 0);
    stats.num_valid_anchors = valid.sum().item<int64_t>();
    if (stats.num_valid_anchors < std::max(2, min_rendered_anchors)) {
        return false;
    }

    torch::Tensor Y = mono.masked_select(valid).contiguous();
    torch::Tensor rendered_depth_valid = depth.masked_select(valid).contiguous();
    torch::Tensor Xref = 1.0f / rendered_depth_valid.clamp_min(near_depth);

    const torch::Tensor Ymed = Y.median();
    const torch::Tensor Ys = (Y - Ymed).abs().mean().clamp_min(1e-6f);
    const torch::Tensor Xmed = Xref.median();
    const torch::Tensor Xs = (Xref - Xmed).abs().mean().clamp_min(1e-6f);
    const torch::Tensor init_inv = (Y - Ymed) * (Xs / Ys) + Xmed;
    const torch::Tensor init_depth = 1.0f / init_inv.clamp_min(1e-6f);
    const torch::Tensor init_rel_err =
        (init_depth - rendered_depth_valid).abs() /
        rendered_depth_valid.clamp_min(near_depth);
    torch::Tensor fit_mask =
        torch::isfinite(init_rel_err) &
        torch::isfinite(init_inv) &
        (init_inv > 1e-6f) &
        (init_rel_err <= kMonoPriorAlignInitialInlierRelErr);
    stats.num_fit_anchors = fit_mask.sum().item<int64_t>();
    if (stats.num_fit_anchors < std::max(2, min_rendered_anchors)) {
        fit_mask = torch::ones_like(fit_mask, torch::TensorOptions().dtype(torch::kBool));
        stats.num_fit_anchors = stats.num_valid_anchors;
    }

    const torch::Tensor Y_fit = Y.masked_select(fit_mask).contiguous();
    const torch::Tensor Xref_fit = Xref.masked_select(fit_mask).contiguous();
    if (!fitAffineScaleShift1D(Y_fit, Xref_fit, stats.scale, stats.shift)) {
        return false;
    }
    if (stats.scale <= 0.0f) {
        return false;
    }

    const torch::Tensor aligned_anchor_inv = Y * stats.scale + stats.shift;
    const torch::Tensor positive_anchor_inv =
        torch::isfinite(aligned_anchor_inv) & (aligned_anchor_inv > 1e-6f);
    if (positive_anchor_inv.sum().item<int64_t>() < std::max(2, min_rendered_anchors)) {
        return false;
    }
    const torch::Tensor aligned_anchor_depth =
        1.0f / aligned_anchor_inv.masked_select(positive_anchor_inv).clamp_min(1e-6f);
    const torch::Tensor rendered_depth_eval =
        rendered_depth_valid.masked_select(positive_anchor_inv).clamp_min(near_depth);
    torch::Tensor rel_err =
        (aligned_anchor_depth - rendered_depth_eval).abs() / rendered_depth_eval;
    rel_err = rel_err.masked_select(torch::isfinite(rel_err)).contiguous();
    if (rel_err.numel() < std::max(2, min_rendered_anchors)) {
        return false;
    }

    stats.median_rel_depth_error = rel_err.median().item<float>();
    if (rel_err.numel() >= 10) {
        stats.p90_rel_depth_error = torch::quantile(
            rel_err,
            torch::tensor(0.90f, torch::TensorOptions().dtype(torch::kFloat32)))
            .item<float>();
    } else {
        stats.p90_rel_depth_error = rel_err.max().item<float>();
    }
    stats.final_inlier_ratio =
        static_cast<float>((rel_err <= kMonoPriorAlignFinalInlierRelErr)
                               .sum()
                               .item<int64_t>()) /
        static_cast<float>(stats.num_valid_anchors);
    if (!std::isfinite(stats.median_rel_depth_error) ||
        !std::isfinite(stats.p90_rel_depth_error) ||
        !std::isfinite(stats.final_inlier_ratio)) {
        return false;
    }

    if (!applyDepthAnythingAffineAlignment(mono, stats.scale, stats.shift, aligned_depth)) {
        return false;
    }

    if (rendered_depth_valid.numel() >= 10) {
        torch::Tensor q = torch::quantile(
            rendered_depth_valid,
            torch::tensor({0.05f, 0.95f}, torch::TensorOptions().dtype(torch::kFloat32)));
        stats.sparse_depth_q05 = q[0].item<float>();
        stats.sparse_depth_q95 = q[1].item<float>();
    } else {
        stats.sparse_depth_q05 = rendered_depth_valid.min().item<float>();
        stats.sparse_depth_q95 = rendered_depth_valid.max().item<float>();
    }

    return aligned_depth.defined() &&
           aligned_depth.dim() == 2 &&
           std::isfinite(stats.sparse_depth_q05) &&
           std::isfinite(stats.sparse_depth_q95) &&
           stats.sparse_depth_q95 > stats.sparse_depth_q05;
}

static bool alignMetricDepthPriorToSparseAnchors(
    const torch::Tensor& mono_prior,
    const torch::Tensor& sparse_uv,
    const torch::Tensor& sparse_depth,
    float cam_near,
    int min_sparse_anchors,
    torch::Tensor& aligned_depth,
    MonoPriorAlignmentStats& stats)
{
    aligned_depth = torch::Tensor();
    stats = MonoPriorAlignmentStats();

    if (!mono_prior.defined() || !sparse_uv.defined() || !sparse_depth.defined()) {
        return false;
    }

    torch::Tensor mono = mono_prior.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (mono.dim() == 3 && mono.size(0) == 1) {
        mono = mono.squeeze(0);
    }
    if (mono.dim() != 2) {
        return false;
    }

    torch::Tensor mono_samples;
    if (!sampleDenseDepthAtSparseUv(mono, sparse_uv, mono_samples)) {
        return false;
    }

    torch::Tensor sparse_depth_1d = sparse_depth.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (sparse_depth_1d.dim() == 2 && sparse_depth_1d.size(1) == 1) {
        sparse_depth_1d = sparse_depth_1d.squeeze(1);
    }
    if (sparse_depth_1d.dim() != 1 || sparse_depth_1d.size(0) != mono_samples.size(0)) {
        return false;
    }

    const float near_depth = std::max(1e-6f, cam_near);
    torch::Tensor valid =
        torch::isfinite(mono_samples) &
        torch::isfinite(sparse_depth_1d) &
        (mono_samples > near_depth) &
        (sparse_depth_1d > near_depth);
    stats.num_valid_anchors = valid.sum().item<int64_t>();
    if (stats.num_valid_anchors < std::max(2, min_sparse_anchors)) {
        return false;
    }

    torch::Tensor Y = mono_samples.masked_select(valid).contiguous();
    torch::Tensor sparse_depth_valid = sparse_depth_1d.masked_select(valid).contiguous();

    torch::Tensor init_scale_candidates =
        sparse_depth_valid / Y.clamp_min(near_depth);
    init_scale_candidates = init_scale_candidates.masked_select(
        torch::isfinite(init_scale_candidates) & (init_scale_candidates > 0.0f));
    if (!init_scale_candidates.defined() ||
        init_scale_candidates.numel() < std::max<int64_t>(2, min_sparse_anchors)) {
        return false;
    }
    const float init_scale = init_scale_candidates.median().item<float>();
    if (!std::isfinite(init_scale) || init_scale <= 0.0f) {
        return false;
    }

    const torch::Tensor init_depth = Y * init_scale;
    const torch::Tensor init_rel_err =
        (init_depth - sparse_depth_valid).abs() /
        sparse_depth_valid.clamp_min(near_depth);
    torch::Tensor fit_mask =
        torch::isfinite(init_rel_err) &
        (init_rel_err <= kMonoPriorAlignInitialInlierRelErr);
    stats.num_fit_anchors = fit_mask.sum().item<int64_t>();
    if (stats.num_fit_anchors < std::max(2, min_sparse_anchors)) {
        fit_mask = torch::ones_like(fit_mask, torch::TensorOptions().dtype(torch::kBool));
        stats.num_fit_anchors = stats.num_valid_anchors;
    }

    const torch::Tensor Y_fit = Y.masked_select(fit_mask).contiguous();
    const torch::Tensor depth_fit = sparse_depth_valid.masked_select(fit_mask).contiguous();
    stats.shift = 0.0f;
    if (!fitScale1D(Y_fit, depth_fit, stats.scale)) {
        return false;
    }

    const torch::Tensor aligned_anchor_depth = Y * stats.scale;
    torch::Tensor rel_err =
        (aligned_anchor_depth - sparse_depth_valid).abs() /
        sparse_depth_valid.clamp_min(near_depth);
    rel_err = rel_err.masked_select(torch::isfinite(rel_err)).contiguous();
    if (rel_err.numel() < std::max(2, min_sparse_anchors)) {
        return false;
    }

    stats.median_rel_depth_error = rel_err.median().item<float>();
    if (rel_err.numel() >= 10) {
        stats.p90_rel_depth_error = torch::quantile(
            rel_err,
            torch::tensor(0.90f, torch::TensorOptions().dtype(torch::kFloat32)))
            .item<float>();
    } else {
        stats.p90_rel_depth_error = rel_err.max().item<float>();
    }
    stats.final_inlier_ratio =
        static_cast<float>((rel_err <= kMonoPriorAlignFinalInlierRelErr)
                               .sum()
                               .item<int64_t>()) /
        static_cast<float>(stats.num_valid_anchors);
    if (!std::isfinite(stats.median_rel_depth_error) ||
        !std::isfinite(stats.p90_rel_depth_error) ||
        !std::isfinite(stats.final_inlier_ratio)) {
        return false;
    }

    if (!applyMetricDepthAlignment(mono, stats.scale, stats.shift, aligned_depth)) {
        return false;
    }

    if (sparse_depth_valid.numel() >= 10) {
        torch::Tensor q = torch::quantile(
            sparse_depth_valid,
            torch::tensor({0.05f, 0.95f}, torch::TensorOptions().dtype(torch::kFloat32)));
        stats.sparse_depth_q05 = q[0].item<float>();
        stats.sparse_depth_q95 = q[1].item<float>();
    } else {
        stats.sparse_depth_q05 = sparse_depth_valid.min().item<float>();
        stats.sparse_depth_q95 = sparse_depth_valid.max().item<float>();
    }

    return aligned_depth.defined() &&
           aligned_depth.dim() == 2 &&
           std::isfinite(stats.sparse_depth_q05) &&
           std::isfinite(stats.sparse_depth_q95) &&
           stats.sparse_depth_q95 > stats.sparse_depth_q05;
}

static bool alignMonoPriorToSparseAnchorsDaPrior(
    const torch::Tensor& mono_prior,
    const torch::Tensor& sparse_uv,
    const torch::Tensor& sparse_depth,
    const torch::Tensor& query_mask,
    bool mono_prior_is_metric_depth,
    float cam_near,
    int min_sparse_anchors,
    int knn_k,
    bool distance_weighting,
    float max_pixel_dist,
    torch::Tensor& aligned_depth,
    MonoPriorAlignmentStats& stats)
{
    aligned_depth = torch::Tensor();
    stats = MonoPriorAlignmentStats();

    if (!mono_prior.defined() || !sparse_uv.defined() || !sparse_depth.defined()) {
        return false;
    }

    torch::Tensor mono = mono_prior.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (mono.dim() == 3 && mono.size(0) == 1) {
        mono = mono.squeeze(0);
    }
    if (mono.dim() != 2) {
        return false;
    }
    const int H = static_cast<int>(mono.size(0));
    const int W = static_cast<int>(mono.size(1));
    if (H <= 0 || W <= 0) {
        return false;
    }

    torch::Tensor mono_samples;
    if (!sampleDenseDepthAtSparseUv(mono, sparse_uv, mono_samples)) {
        return false;
    }

    torch::Tensor sparse_depth_1d = sparse_depth.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (sparse_depth_1d.dim() == 2 && sparse_depth_1d.size(1) == 1) {
        sparse_depth_1d = sparse_depth_1d.squeeze(1);
    }
    torch::Tensor sparse_uv_cpu = sparse_uv.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (sparse_depth_1d.dim() != 1 ||
        sparse_uv_cpu.dim() != 2 ||
        sparse_uv_cpu.size(1) != 2 ||
        sparse_depth_1d.size(0) != mono_samples.size(0) ||
        sparse_uv_cpu.size(0) != mono_samples.size(0)) {
        return false;
    }

    struct DaPriorAnchor {
        float x = 0.0f;
        float y = 0.0f;
        float pred_disp = 0.0f;
        float target_disp = 0.0f;
        float target_depth = 0.0f;
    };

    const float near_depth = std::max(1e-6f, cam_near);
    auto mono_sample_acc = mono_samples.accessor<float, 1>();
    auto sparse_depth_acc = sparse_depth_1d.accessor<float, 1>();
    auto sparse_uv_acc = sparse_uv_cpu.accessor<float, 2>();

    std::vector<DaPriorAnchor> valid_anchors;
    valid_anchors.reserve(static_cast<size_t>(mono_samples.size(0)));
    std::vector<float> pred_disp_values;
    std::vector<float> target_disp_values;
    std::vector<float> sparse_depth_values;

    for (int64_t i = 0; i < mono_samples.size(0); ++i) {
        const float mono_value = mono_sample_acc[i];
        const float z = sparse_depth_acc[i];
        if (!std::isfinite(mono_value) || !std::isfinite(z) || z <= near_depth) {
            continue;
        }

        float pred_disp = std::numeric_limits<float>::quiet_NaN();
        if (mono_prior_is_metric_depth) {
            if (mono_value <= near_depth) {
                continue;
            }
            pred_disp = 1.0f / mono_value;
        } else {
            if (mono_value <= 0.0f) {
                continue;
            }
            pred_disp = mono_value;
        }
        const float target_disp = 1.0f / std::max(z, near_depth);
        if (!std::isfinite(pred_disp) || !std::isfinite(target_disp) ||
            pred_disp <= 0.0f || target_disp <= 0.0f) {
            continue;
        }

        const float x = 0.5f * (sparse_uv_acc[i][0] + 1.0f) * static_cast<float>(W);
        const float y = 0.5f * (sparse_uv_acc[i][1] + 1.0f) * static_cast<float>(H);
        if (!std::isfinite(x) || !std::isfinite(y) ||
            x < 0.0f || x > static_cast<float>(W - 1) ||
            y < 0.0f || y > static_cast<float>(H - 1)) {
            continue;
        }

        valid_anchors.push_back({x, y, pred_disp, target_disp, z});
        pred_disp_values.push_back(pred_disp);
        target_disp_values.push_back(target_disp);
        sparse_depth_values.push_back(z);
    }

    stats.num_valid_anchors = static_cast<int64_t>(valid_anchors.size());
    if (stats.num_valid_anchors < std::max(2, min_sparse_anchors)) {
        return false;
    }

    const float pred_med = quantileVector(pred_disp_values, 0.50f);
    const float target_med = quantileVector(target_disp_values, 0.50f);
    if (!std::isfinite(pred_med) || !std::isfinite(target_med)) {
        return false;
    }

    double pred_mad = 0.0;
    double target_mad = 0.0;
    for (size_t i = 0; i < pred_disp_values.size(); ++i) {
        pred_mad += std::abs(static_cast<double>(pred_disp_values[i]) - pred_med);
        target_mad += std::abs(static_cast<double>(target_disp_values[i]) - target_med);
    }
    pred_mad = std::max(pred_mad / static_cast<double>(pred_disp_values.size()), 1e-6);
    target_mad = std::max(target_mad / static_cast<double>(target_disp_values.size()), 1e-6);

    std::vector<size_t> fit_indices;
    fit_indices.reserve(valid_anchors.size());
    for (size_t i = 0; i < valid_anchors.size(); ++i) {
        const float init_disp =
            (valid_anchors[i].pred_disp - pred_med) *
                static_cast<float>(target_mad / pred_mad) +
            target_med;
        if (!std::isfinite(init_disp) || init_disp <= 1e-6f) {
            continue;
        }
        const float init_depth = 1.0f / init_disp;
        const float rel_err =
            std::abs(init_depth - valid_anchors[i].target_depth) /
            std::max(valid_anchors[i].target_depth, near_depth);
        if (std::isfinite(rel_err) && rel_err <= kMonoPriorAlignInitialInlierRelErr) {
            fit_indices.push_back(i);
        }
    }

    stats.num_fit_anchors = static_cast<int64_t>(fit_indices.size());
    if (stats.num_fit_anchors < std::max(2, min_sparse_anchors)) {
        fit_indices.clear();
        fit_indices.reserve(valid_anchors.size());
        for (size_t i = 0; i < valid_anchors.size(); ++i) {
            fit_indices.push_back(i);
        }
        stats.num_fit_anchors = stats.num_valid_anchors;
    }

    std::vector<float> fit_pred;
    std::vector<float> fit_target;
    fit_pred.reserve(fit_indices.size());
    fit_target.reserve(fit_indices.size());
    for (const size_t idx : fit_indices) {
        fit_pred.push_back(valid_anchors[idx].pred_disp);
        fit_target.push_back(valid_anchors[idx].target_disp);
    }
    if (!fitAffineScaleShiftWeighted1D(fit_pred, fit_target, nullptr, stats.scale, stats.shift)) {
        return false;
    }
    if (!std::isfinite(stats.scale) || !std::isfinite(stats.shift)) {
        return false;
    }

    std::vector<float> rel_errors;
    rel_errors.reserve(valid_anchors.size());
    int64_t final_inliers = 0;
    for (const auto& anchor : valid_anchors) {
        const float aligned_disp = anchor.pred_disp * stats.scale + stats.shift;
        if (!std::isfinite(aligned_disp) || aligned_disp <= 1e-6f) {
            continue;
        }
        const float aligned_z = 1.0f / aligned_disp;
        const float rel_err =
            std::abs(aligned_z - anchor.target_depth) /
            std::max(anchor.target_depth, near_depth);
        if (std::isfinite(rel_err)) {
            rel_errors.push_back(rel_err);
            if (rel_err <= kMonoPriorAlignFinalInlierRelErr) {
                ++final_inliers;
            }
        }
    }
    if (rel_errors.size() < static_cast<size_t>(std::max(2, min_sparse_anchors))) {
        return false;
    }
    stats.median_rel_depth_error = quantileVector(rel_errors, 0.50f);
    stats.p90_rel_depth_error = quantileVector(rel_errors, 0.90f);
    stats.final_inlier_ratio =
        static_cast<float>(final_inliers) /
        static_cast<float>(std::max<int64_t>(1, stats.num_valid_anchors));
    stats.sparse_depth_q05 = quantileVector(sparse_depth_values, 0.05f);
    stats.sparse_depth_q95 = quantileVector(sparse_depth_values, 0.95f);
    if (!std::isfinite(stats.median_rel_depth_error) ||
        !std::isfinite(stats.p90_rel_depth_error) ||
        !std::isfinite(stats.final_inlier_ratio) ||
        !std::isfinite(stats.sparse_depth_q05) ||
        !std::isfinite(stats.sparse_depth_q95) ||
        !(stats.sparse_depth_q95 > stats.sparse_depth_q05)) {
        return false;
    }

    const int local_k = std::min<int>(
        std::max(2, knn_k),
        static_cast<int>(fit_indices.size()));
    if (local_k < 2) {
        return false;
    }

    cv::Mat anchor_mat(static_cast<int>(fit_indices.size()), 2, CV_32F);
    std::vector<DaPriorAnchor> fit_anchors;
    fit_anchors.reserve(fit_indices.size());
    for (size_t i = 0; i < fit_indices.size(); ++i) {
        const DaPriorAnchor& anchor = valid_anchors[fit_indices[i]];
        fit_anchors.push_back(anchor);
        anchor_mat.at<float>(static_cast<int>(i), 0) = anchor.x;
        anchor_mat.at<float>(static_cast<int>(i), 1) = anchor.y;
    }

    torch::Tensor qmask;
    if (query_mask.defined()) {
        qmask = query_mask.detach().to(torch::kCPU).to(torch::kBool).contiguous();
        if (qmask.dim() != 2 || qmask.size(0) != H || qmask.size(1) != W) {
            return false;
        }
    }

    torch::Tensor aligned =
        torch::full({H, W}, std::numeric_limits<float>::quiet_NaN(),
                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto aligned_acc = aligned.accessor<float, 2>();
    auto mono_acc = mono.accessor<float, 2>();

    auto pred_disp_at = [&](int y, int x, float& pred_disp) -> bool {
        const float mono_value = mono_acc[y][x];
        if (!std::isfinite(mono_value)) {
            return false;
        }
        if (mono_prior_is_metric_depth) {
            if (mono_value <= near_depth) {
                return false;
            }
            pred_disp = 1.0f / mono_value;
        } else {
            if (mono_value <= 0.0f) {
                return false;
            }
            pred_disp = mono_value;
        }
        return std::isfinite(pred_disp) && pred_disp > 0.0f;
    };

    std::vector<cv::Point> query_pixels;
    query_pixels.reserve(query_mask.defined()
                             ? static_cast<size_t>(std::max<int64_t>(1, qmask.sum().item<int64_t>()))
                             : static_cast<size_t>(H) * static_cast<size_t>(W));
    if (qmask.defined()) {
        auto qmask_acc = qmask.accessor<bool, 2>();
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float pred_disp = 0.0f;
                if (qmask_acc[y][x] && pred_disp_at(y, x, pred_disp)) {
                    query_pixels.emplace_back(x, y);
                }
            }
        }
    } else {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float pred_disp = 0.0f;
                if (pred_disp_at(y, x, pred_disp)) {
                    query_pixels.emplace_back(x, y);
                }
            }
        }
    }

    if (query_pixels.empty()) {
        return false;
    }

    cv::Mat query_mat(static_cast<int>(query_pixels.size()), 2, CV_32F);
    for (size_t i = 0; i < query_pixels.size(); ++i) {
        query_mat.at<float>(static_cast<int>(i), 0) = static_cast<float>(query_pixels[i].x);
        query_mat.at<float>(static_cast<int>(i), 1) = static_cast<float>(query_pixels[i].y);
    }

    cv::flann::Index knn_index(anchor_mat, cv::flann::KDTreeIndexParams(4));
    cv::Mat indices;
    cv::Mat dists_sq;
    knn_index.knnSearch(
        query_mat,
        indices,
        dists_sq,
        local_k,
        cv::flann::SearchParams(32));

    constexpr float kDirectAnchorDistPx = 1e-3f;
    for (size_t qi = 0; qi < query_pixels.size(); ++qi) {
        const int x = query_pixels[qi].x;
        const int y = query_pixels[qi].y;
        float query_pred_disp = 0.0f;
        if (!pred_disp_at(y, x, query_pred_disp)) {
            continue;
        }

        float direct_depth = std::numeric_limits<float>::quiet_NaN();
        std::vector<float> local_pred;
        std::vector<float> local_target;
        std::vector<float> local_weights;
        local_pred.reserve(static_cast<size_t>(local_k));
        local_target.reserve(static_cast<size_t>(local_k));
        local_weights.reserve(static_cast<size_t>(local_k));

        double inv_dist_sum = 0.0;
        for (int k = 0; k < local_k; ++k) {
            const int anchor_idx = indices.at<int>(static_cast<int>(qi), k);
            if (anchor_idx < 0 || anchor_idx >= static_cast<int>(fit_anchors.size())) {
                continue;
            }
            const float dist =
                std::sqrt(std::max(0.0f, dists_sq.at<float>(static_cast<int>(qi), k)));
            if (max_pixel_dist > 0.0f && dist > max_pixel_dist) {
                continue;
            }
            const DaPriorAnchor& anchor = fit_anchors[static_cast<size_t>(anchor_idx)];
            if (dist <= kDirectAnchorDistPx) {
                direct_depth = anchor.target_depth;
                break;
            }
            const float inv_dist = 1.0f / std::max(dist, 1e-3f);
            inv_dist_sum += inv_dist;
            local_pred.push_back(anchor.pred_disp);
            local_target.push_back(anchor.target_disp);
            local_weights.push_back(inv_dist);
        }

        if (std::isfinite(direct_depth) && direct_depth > near_depth) {
            aligned_acc[y][x] = direct_depth;
            continue;
        }

        float local_scale = stats.scale;
        float local_shift = stats.shift;
        if (local_pred.size() >= 2) {
            std::vector<float>* weights_ptr = nullptr;
            if (distance_weighting && inv_dist_sum > 0.0) {
                for (float& w : local_weights) {
                    const float normalized = w / static_cast<float>(inv_dist_sum);
                    w = normalized * normalized;
                }
                weights_ptr = &local_weights;
            }
            if (!fitAffineScaleShiftWeighted1D(
                    local_pred,
                    local_target,
                    weights_ptr,
                    local_scale,
                    local_shift)) {
                local_scale = stats.scale;
                local_shift = stats.shift;
            }
        }

        float aligned_disp = query_pred_disp * local_scale + local_shift;
        if (!std::isfinite(aligned_disp) || aligned_disp <= 1e-6f) {
            aligned_disp = query_pred_disp * stats.scale + stats.shift;
        }
        if (std::isfinite(aligned_disp) && aligned_disp > 1e-6f) {
            const float z = 1.0f / aligned_disp;
            if (std::isfinite(z) && z > near_depth) {
                aligned_acc[y][x] = z;
            }
        }
    }

    for (const auto& anchor : fit_anchors) {
        const int x = std::clamp(static_cast<int>(std::round(anchor.x)), 0, W - 1);
        const int y = std::clamp(static_cast<int>(std::round(anchor.y)), 0, H - 1);
        aligned_acc[y][x] = anchor.target_depth;
    }

    const int64_t aligned_count =
        (torch::isfinite(aligned) & (aligned > near_depth)).sum().item<int64_t>();
    if (aligned_count <= 0) {
        return false;
    }

    aligned_depth = aligned.contiguous();
    return true;
}

static bool alignMetricDepthPriorToRenderedDepth(
    const torch::Tensor& mono_prior,
    const torch::Tensor& rendered_depth,
    const torch::Tensor& rendered_alpha,
    const torch::Tensor& rendered_n_contrib,
    float cam_near,
    int min_rendered_anchors,
    torch::Tensor& aligned_depth,
    MonoPriorAlignmentStats& stats)
{
    aligned_depth = torch::Tensor();
    stats = MonoPriorAlignmentStats();

    if (!mono_prior.defined() || !rendered_depth.defined() ||
        !rendered_alpha.defined() || !rendered_n_contrib.defined()) {
        return false;
    }

    torch::Tensor mono =
        mono_prior.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (mono.dim() == 3 && mono.size(0) == 1) {
        mono = mono.squeeze(0);
    }
    torch::Tensor depth =
        rendered_depth.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor alpha =
        rendered_alpha.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor n_contrib =
        rendered_n_contrib.detach().to(torch::kCPU).to(torch::kInt32).contiguous();
    if (mono.dim() != 2 || depth.dim() != 2 || alpha.dim() != 2 ||
        n_contrib.dim() != 2 ||
        mono.sizes() != depth.sizes() ||
        mono.sizes() != alpha.sizes() ||
        mono.sizes() != n_contrib.sizes()) {
        return false;
    }

    const float near_depth = std::max(1e-6f, cam_near);
    torch::Tensor valid =
        torch::isfinite(mono) &
        torch::isfinite(depth) &
        torch::isfinite(alpha) &
        (mono > near_depth) &
        (depth > near_depth) &
        (alpha > 0.5f) &
        (n_contrib > 0);
    stats.num_valid_anchors = valid.sum().item<int64_t>();
    if (stats.num_valid_anchors < std::max(2, min_rendered_anchors)) {
        return false;
    }

    torch::Tensor Y = mono.masked_select(valid).contiguous();
    torch::Tensor rendered_depth_valid = depth.masked_select(valid).contiguous();

    torch::Tensor init_scale_candidates =
        rendered_depth_valid / Y.clamp_min(near_depth);
    init_scale_candidates = init_scale_candidates.masked_select(
        torch::isfinite(init_scale_candidates) & (init_scale_candidates > 0.0f));
    if (!init_scale_candidates.defined() ||
        init_scale_candidates.numel() < std::max<int64_t>(2, min_rendered_anchors)) {
        return false;
    }
    const float init_scale = init_scale_candidates.median().item<float>();
    if (!std::isfinite(init_scale) || init_scale <= 0.0f) {
        return false;
    }

    const torch::Tensor init_depth = Y * init_scale;
    const torch::Tensor init_rel_err =
        (init_depth - rendered_depth_valid).abs() /
        rendered_depth_valid.clamp_min(near_depth);
    torch::Tensor fit_mask =
        torch::isfinite(init_rel_err) &
        (init_rel_err <= kMonoPriorAlignInitialInlierRelErr);
    stats.num_fit_anchors = fit_mask.sum().item<int64_t>();
    if (stats.num_fit_anchors < std::max(2, min_rendered_anchors)) {
        fit_mask = torch::ones_like(fit_mask, torch::TensorOptions().dtype(torch::kBool));
        stats.num_fit_anchors = stats.num_valid_anchors;
    }

    const torch::Tensor Y_fit = Y.masked_select(fit_mask).contiguous();
    const torch::Tensor depth_fit = rendered_depth_valid.masked_select(fit_mask).contiguous();
    stats.shift = 0.0f;
    if (!fitScale1D(Y_fit, depth_fit, stats.scale)) {
        return false;
    }

    const torch::Tensor aligned_anchor_depth = Y * stats.scale;
    torch::Tensor rel_err =
        (aligned_anchor_depth - rendered_depth_valid).abs() /
        rendered_depth_valid.clamp_min(near_depth);
    rel_err = rel_err.masked_select(torch::isfinite(rel_err)).contiguous();
    if (rel_err.numel() < std::max(2, min_rendered_anchors)) {
        return false;
    }

    stats.median_rel_depth_error = rel_err.median().item<float>();
    if (rel_err.numel() >= 10) {
        stats.p90_rel_depth_error = torch::quantile(
            rel_err,
            torch::tensor(0.90f, torch::TensorOptions().dtype(torch::kFloat32)))
            .item<float>();
    } else {
        stats.p90_rel_depth_error = rel_err.max().item<float>();
    }
    stats.final_inlier_ratio =
        static_cast<float>((rel_err <= kMonoPriorAlignFinalInlierRelErr)
                               .sum()
                               .item<int64_t>()) /
        static_cast<float>(stats.num_valid_anchors);
    if (!std::isfinite(stats.median_rel_depth_error) ||
        !std::isfinite(stats.p90_rel_depth_error) ||
        !std::isfinite(stats.final_inlier_ratio)) {
        return false;
    }

    if (!applyMetricDepthAlignment(mono, stats.scale, stats.shift, aligned_depth)) {
        return false;
    }

    if (rendered_depth_valid.numel() >= 10) {
        torch::Tensor q = torch::quantile(
            rendered_depth_valid,
            torch::tensor({0.05f, 0.95f}, torch::TensorOptions().dtype(torch::kFloat32)));
        stats.sparse_depth_q05 = q[0].item<float>();
        stats.sparse_depth_q95 = q[1].item<float>();
    } else {
        stats.sparse_depth_q05 = rendered_depth_valid.min().item<float>();
        stats.sparse_depth_q95 = rendered_depth_valid.max().item<float>();
    }

    return aligned_depth.defined() &&
           aligned_depth.dim() == 2 &&
           std::isfinite(stats.sparse_depth_q05) &&
           std::isfinite(stats.sparse_depth_q95) &&
           stats.sparse_depth_q95 > stats.sparse_depth_q05;
}

static void saveDepthComparisonDebugPngs(
    const torch::Tensor& pred_depth,
    const cv::Mat& gt_depth_meters,
    float valid_min_depth,
    float valid_max_depth,
    const std::filesystem::path& rendered_path,
    const std::filesystem::path& gt_path,
    const std::filesystem::path& pair_path,
    std::optional<float> pred_to_gt_scale = std::nullopt)
{
    torch::Tensor pred_depth_for_pair = pred_depth;
    bool used_alignment = false;
    if (pred_to_gt_scale.has_value() &&
        std::isfinite(*pred_to_gt_scale) &&
        *pred_to_gt_scale > 0.0f) {
        pred_depth_for_pair = pred_depth * (*pred_to_gt_scale);
        used_alignment = true;
    }

    float viz_min = 0.0f;
    float viz_max = 1.0f;
    if (!computeSharedDepthVizRange(
            pred_depth_for_pair,
            gt_depth_meters,
            valid_min_depth,
            valid_max_depth,
            viz_min,
            viz_max)) {
        return;
    }

    const cv::Mat pred_depth_meters = depthTensorToCvMatFloat(pred_depth_for_pair);
    const cv::Mat pred_bgr = colorizeDepthMatJet(
        pred_depth_meters,
        valid_min_depth,
        valid_max_depth,
        viz_min,
        viz_max);

    std::filesystem::create_directories(rendered_path.parent_path());
    cv::imwrite(rendered_path.string(), pred_bgr);

    if (gt_depth_meters.empty()) {
        return;
    }

    cv::Mat gt_bgr = colorizeDepthMatJet(
        gt_depth_meters,
        valid_min_depth,
        valid_max_depth,
        viz_min,
        viz_max);

    cv::Mat pair_bgr;
    cv::hconcat(std::vector<cv::Mat>{gt_bgr, pred_bgr}, pair_bgr);
    pair_bgr = appendJetLegendBar(pair_bgr, viz_min, viz_max, " m");

    std::filesystem::create_directories(gt_path.parent_path());
    cv::imwrite(gt_path.string(), gt_bgr);
    cv::imwrite(pair_path.string(), pair_bgr);
}

static bool depthMatToMeters(const cv::Mat& depth_in, cv::Mat& depth_meters)
{
    if (depth_in.empty()) {
        return false;
    }

    cv::Mat d = depth_in;
    if (d.channels() > 1) {
        cv::extractChannel(d, d, 0);
    }

    if (d.type() == CV_32FC1) {
        depth_meters = d;
        return true;
    }
    if (d.type() == CV_16UC1) {
        // Handle both common 16-bit conventions:
        // - mm depth (TUM-style): depth_m = raw / 1000
        // - Replica-style uint16 encoding: depth_m = raw / 6553.5
        double max_val = 0.0;
        cv::minMaxLoc(d, nullptr, &max_val);
        const double scale = (max_val > 20000.0) ? (1.0 / 6553.5) : (1.0 / 1000.0);
        d.convertTo(depth_meters, CV_32FC1, scale);
        return true;
    }

    d.convertTo(depth_meters, CV_32FC1);
    return true;
}

static bool loadReplicaDepthFromRgbPath(const std::string& rgb_filename, cv::Mat& depth_meters)
{
    if (rgb_filename.empty()) {
        return false;
    }

    const std::filesystem::path rgb_path(rgb_filename);
    if (!std::filesystem::exists(rgb_path)) {
        return false;
    }

    const std::string name = rgb_path.filename().string();
    if (name.rfind("frame", 0) != 0) {
        return false;
    }

    const std::filesystem::path parent = rgb_path.parent_path();
    const std::string stem = rgb_path.stem().string();   // e.g., frame000123
    const std::string suffix_stem = (stem.size() > 5 ? stem.substr(5) : std::string());
    const std::string suffix_name = name.substr(5);      // e.g., 000123.jpg

    std::vector<std::filesystem::path> candidates;
    candidates.push_back(parent / ("depth" + suffix_name));
    if (!suffix_stem.empty()) {
        candidates.push_back(parent / ("depth" + suffix_stem + ".png"));
        candidates.push_back(parent / ("depth" + suffix_stem + ".exr"));
        candidates.push_back(parent / ("depth" + suffix_stem + ".tiff"));
        candidates.push_back(parent / ("depth" + suffix_stem + ".tif"));
    }

    for (const auto& p : candidates) {
        if (!std::filesystem::exists(p)) {
            continue;
        }
        const cv::Mat depth_raw = cv::imread(p.string(), cv::IMREAD_UNCHANGED);
        if (depth_raw.empty()) {
            continue;
        }
        return depthMatToMeters(depth_raw, depth_meters);
    }

    return false;
}

static bool loadTumDepthFromRgbPath(const std::string& rgb_filename, cv::Mat& depth_meters)
{
    if (rgb_filename.empty()) {
        return false;
    }

    const std::filesystem::path rgb_path(rgb_filename);
    if (!std::filesystem::exists(rgb_path)) {
        return false;
    }

    const std::filesystem::path rgb_dir = rgb_path.parent_path();
    if (rgb_dir.filename() != "rgb") {
        return false;
    }

    const std::filesystem::path dataset_root = rgb_dir.parent_path();
    const std::filesystem::path depth_dir = dataset_root / "depth";
    const std::filesystem::path depth_txt = dataset_root / "depth.txt";
    if (!std::filesystem::exists(depth_dir)) {
        return false;
    }

    const std::string stem = rgb_path.stem().string();
    const std::filesystem::path exact_depth_path = depth_dir / (stem + rgb_path.extension().string());
    if (std::filesystem::exists(exact_depth_path)) {
        const cv::Mat depth_raw = cv::imread(exact_depth_path.string(), cv::IMREAD_UNCHANGED);
        if (!depth_raw.empty()) {
            return depthMatToMeters(depth_raw, depth_meters);
        }
    }

    double rgb_ts = 0.0;
    try {
        rgb_ts = std::stod(stem);
    } catch (...) {
        return false;
    }

    struct TumDepthIndexEntry {
        double timestamp = 0.0;
        std::filesystem::path path;
    };

    static std::mutex s_tum_depth_cache_mutex;
    static std::unordered_map<std::string, std::vector<TumDepthIndexEntry>> s_tum_depth_cache;

    std::vector<TumDepthIndexEntry> depth_index;
    {
        std::lock_guard<std::mutex> lock(s_tum_depth_cache_mutex);
        auto it = s_tum_depth_cache.find(dataset_root.string());
        if (it == s_tum_depth_cache.end()) {
            std::vector<TumDepthIndexEntry> parsed;
            if (std::filesystem::exists(depth_txt)) {
                std::ifstream in(depth_txt);
                std::string line;
                while (std::getline(in, line)) {
                    if (line.empty() || line[0] == '#') {
                        continue;
                    }
                    std::istringstream iss(line);
                    double ts = 0.0;
                    std::string rel_path;
                    if (!(iss >> ts >> rel_path)) {
                        continue;
                    }
                    std::filesystem::path p = dataset_root / rel_path;
                    if (!std::filesystem::exists(p)) {
                        continue;
                    }
                    parsed.push_back({ts, p});
                }
            }
            std::sort(
                parsed.begin(),
                parsed.end(),
                [](const TumDepthIndexEntry& a, const TumDepthIndexEntry& b) {
                    return a.timestamp < b.timestamp;
                });
            it = s_tum_depth_cache.emplace(dataset_root.string(), std::move(parsed)).first;
        }
        depth_index = it->second;
    }

    if (depth_index.empty()) {
        return false;
    }

    auto lb = std::lower_bound(
        depth_index.begin(),
        depth_index.end(),
        rgb_ts,
        [](const TumDepthIndexEntry& e, double t) {
            return e.timestamp < t;
        });

    auto best_it = depth_index.end();
    double best_dt = std::numeric_limits<double>::infinity();
    if (lb != depth_index.end()) {
        best_it = lb;
        best_dt = std::abs(lb->timestamp - rgb_ts);
    }
    if (lb != depth_index.begin()) {
        auto prev = std::prev(lb);
        const double prev_dt = std::abs(prev->timestamp - rgb_ts);
        if (prev_dt < best_dt) {
            best_it = prev;
            best_dt = prev_dt;
        }
    }

    constexpr double kTumMaxDepthAssocDeltaSec = 0.05;
    if (best_it == depth_index.end() || !(best_dt <= kTumMaxDepthAssocDeltaSec)) {
        return false;
    }

    const cv::Mat depth_raw = cv::imread(best_it->path.string(), cv::IMREAD_UNCHANGED);
    if (depth_raw.empty()) {
        return false;
    }
    return depthMatToMeters(depth_raw, depth_meters);
}

static std::string monoPriorCacheKeyForKeyframe(const std::shared_ptr<VoxelKeyframe>& pkf)
{
    if (!pkf) {
        return "kf_invalid";
    }

    std::string key;
    if (!pkf->img_filename_.empty()) {
        key = std::filesystem::path(pkf->img_filename_).stem().string();
    }
    if (key.empty()) {
        std::ostringstream oss;
        oss << "kf_" << std::setw(6) << std::setfill('0') << pkf->fid_;
        key = oss.str();
    }

    key = std::regex_replace(key, std::regex(R"([^A-Za-z0-9_.-])"), "_");
    return key;
}

static bool buildSparseDepthFromKeyframeOrbAnchors(
    const std::shared_ptr<VoxelKeyframe>& kf,
    int image_width,
    int image_height,
    torch::Tensor& sparse_uv,
    torch::Tensor& sparse_depth)
{
    sparse_uv = torch::Tensor();
    sparse_depth = torch::Tensor();
    if (!kf || image_width <= 0 || image_height <= 0) {
        return false;
    }

    const size_t n_pix = kf->kps_pixel_.size() / 2;
    const size_t n_xyz = kf->kps_point_local_.size() / 3;
    const size_t N = std::min(n_pix, n_xyz);
    if (N == 0) {
        return false;
    }

    const float sx =
        (kf->image_width_ > 0)
            ? (static_cast<float>(image_width) / static_cast<float>(kf->image_width_))
            : 1.0f;
    const float sy =
        (kf->image_height_ > 0)
            ? (static_cast<float>(image_height) / static_cast<float>(kf->image_height_))
            : 1.0f;
    const float W = static_cast<float>(image_width);
    const float H = static_cast<float>(image_height);

    std::vector<float> uv_host;
    std::vector<float> depth_host;
    uv_host.reserve(2 * N);
    depth_host.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        const float u = kf->kps_pixel_[2 * i + 0] * sx;
        const float v = kf->kps_pixel_[2 * i + 1] * sy;
        const float z = kf->kps_point_local_[3 * i + 2];
        if (!std::isfinite(u) || !std::isfinite(v) || !std::isfinite(z) || z <= 0.0f) {
            continue;
        }
        if (u < 0.0f || u > (W - 1.0f) || v < 0.0f || v > (H - 1.0f)) {
            continue;
        }

        uv_host.push_back(2.0f * (u / W) - 1.0f);
        uv_host.push_back(2.0f * (v / H) - 1.0f);
        depth_host.push_back(z);
    }

    const int64_t M = static_cast<int64_t>(depth_host.size());
    if (M < 2) {
        return false;
    }

    sparse_uv = torch::from_blob(
        uv_host.data(),
        {M, 2},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
    sparse_depth = torch::from_blob(
        depth_host.data(),
        {M},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
    return true;
}

static bool getKeyframeDepthMetersForEval(
    const std::shared_ptr<VoxelKeyframe>& pkf,
    int expected_h,
    int expected_w,
    cv::Mat& depth_meters)
{
    if (!pkf) {
        return false;
    }

    if (!pkf->img_auxiliary_undist_.empty()) {
        if (!depthMatToMeters(pkf->img_auxiliary_undist_, depth_meters)) {
            return false;
        }
    } else {
        if (!loadReplicaDepthFromRgbPath(pkf->img_filename_, depth_meters) &&
            !loadTumDepthFromRgbPath(pkf->img_filename_, depth_meters)) {
            return false;
        }
    }

    if (depth_meters.empty()) {
        return false;
    }

    if (depth_meters.rows != expected_h || depth_meters.cols != expected_w) {
        cv::resize(
            depth_meters,
            depth_meters,
            cv::Size(expected_w, expected_h),
            0.0,
            0.0,
            cv::INTER_NEAREST);
    }

    return true;
}

bool VoxelMapper::ensureMonoPriorForKeyframe(
    const std::shared_ptr<VoxelKeyframe>& kf)
{
    const char* log_prefix = monoPriorLogPrefix(mono_prior_model_id_);
    if (!kf || !kf->original_image_.defined()) {
        return false;
    }
    if (kf->mono_prior_.defined() && kf->mono_prior_.numel() > 0) {
        if (kf->mono_prior_prepare_iter_ < 0) {
            kf->mono_prior_prepare_iter_ = getIteration();
        }
        return true;
    }

    try {
        py::gil_scoped_acquire gil;
        static py::object py_load_or_infer;
        if (!py_load_or_infer) {
            py_load_or_infer =
                py::module_::import("scripts_voxel.python_svraster_bridge.mono_prior_helper")
                    .attr("load_or_infer_mono_prior");
        }

        const std::filesystem::path depth_root =
            model_params_.source_path_ / "mono_priors" / "depthanythingv2";
        std::filesystem::create_directories(depth_root);

        torch::Tensor image_cpu =
            kf->original_image_.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
        const std::string cache_key = monoPriorCacheKeyForKeyframe(kf);

        py::object py_depth = py_load_or_infer(
            py::cast(image_cpu),
            py::str(depth_root.string()),
            py::str(cache_key),
            py::str(mono_prior_model_id_),
            py::bool_(true),
            py::float_(kf->intr_.empty() ? 0.0f : static_cast<double>(kf->intr_[0])));

        torch::Tensor depth = py_depth.cast<torch::Tensor>()
                                  .detach()
                                  .to(torch::kCPU)
                                  .to(torch::kFloat32)
                                  .contiguous();
        if (depth.dim() == 3 && depth.size(0) == 1) {
            depth = depth.squeeze(0);
        }
        if (depth.dim() != 2) {
            std::cerr << log_prefix << " Unexpected prior shape for keyframe "
                      << kf->fid_ << ": " << depth.sizes() << std::endl;
            return false;
        }

        kf->mono_prior_ = depth;
        kf->mono_prior_prepare_iter_ = getIteration();
        return true;
    } catch (const py::error_already_set& e) {
        std::cerr << log_prefix << " Python exception while preparing prior for keyframe "
                  << kf->fid_ << ":\n" << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << log_prefix << " Failed to prepare prior for keyframe "
                  << kf->fid_ << ": " << e.what() << std::endl;
    }

    return false;
}

bool VoxelMapper::buildAlignedMonoPriorDepthForKeyframe(
    const std::shared_ptr<VoxelKeyframe>& kf,
    const sv::MiniCam& cam,
    int image_width,
    int image_height,
    torch::Tensor& aligned_depth)
{
    torch::NoGradGuard no_grad;
    aligned_depth = torch::Tensor();

    if (!kf || image_width <= 0 || image_height <= 0) {
        return false;
    }
    if (!ensureMonoPriorForKeyframe(kf) ||
        !kf->mono_prior_.defined() ||
        kf->mono_prior_.numel() == 0) {
        return false;
    }

    torch::Tensor mono_prior =
        kf->mono_prior_.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (mono_prior.dim() == 3 && mono_prior.size(0) == 1) {
        mono_prior = mono_prior.squeeze(0);
    }
    if (mono_prior.dim() != 2) {
        return false;
    }
    if (mono_prior.size(0) != image_height || mono_prior.size(1) != image_width) {
        mono_prior = torch::nn::functional::interpolate(
            mono_prior.unsqueeze(0).unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{image_height, image_width})
                .mode(torch::kBilinear)
                .align_corners(false)).squeeze().to(torch::kCPU).contiguous();
    }

    if (depthanything_densify_alignment_mode_ == "da_prior" &&
        kf->mono_prior_aligned_depth_cache_.defined() &&
        kf->mono_prior_aligned_depth_cache_mode_ == "da_prior" &&
        kf->mono_prior_aligned_depth_cache_.dim() == 2 &&
        kf->mono_prior_aligned_depth_cache_.size(0) == image_height &&
        kf->mono_prior_aligned_depth_cache_.size(1) == image_width) {
        aligned_depth = kf->mono_prior_aligned_depth_cache_.detach()
                            .to(torch::kCPU)
                            .to(torch::kFloat32)
                            .contiguous();
        return true;
    }

    const bool use_metric3d = isMetric3DModelId(mono_prior_model_id_);
    torch::Tensor sparse_uv;
    torch::Tensor sparse_depth;
    MonoPriorAlignmentStats stats;
    if (buildSparseDepthFromKeyframeOrbAnchors(
            kf,
            image_width,
            image_height,
            sparse_uv,
            sparse_depth) &&
        (((depthanything_densify_alignment_mode_ == "da_prior") &&
          alignMonoPriorToSparseAnchorsDaPrior(
              mono_prior,
              sparse_uv,
              sparse_depth,
              torch::Tensor(),
              use_metric3d,
              cam.near,
              depthanything_densify_min_sparse_anchors_,
              depthanything_da_prior_knn_k_,
              depthanything_da_prior_distance_weighting_,
              depthanything_da_prior_max_pixel_dist_,
              aligned_depth,
              stats)) ||
         (use_metric3d &&
          alignMetricDepthPriorToSparseAnchors(
              mono_prior,
              sparse_uv,
              sparse_depth,
              cam.near,
              depthanything_densify_min_sparse_anchors_,
              aligned_depth,
              stats)) ||
         (!use_metric3d &&
          alignDepthAnythingPriorToSparseAnchors(
              mono_prior,
              sparse_uv,
              sparse_depth,
              cam.near,
              depthanything_densify_min_sparse_anchors_,
              aligned_depth,
              stats)))) {
        if (depthanything_densify_alignment_mode_ == "da_prior") {
            kf->mono_prior_aligned_depth_cache_ =
                aligned_depth.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
            kf->mono_prior_aligned_depth_cache_mode_ = "da_prior";
            kf->mono_prior_aligned_depth_cache_iter_ = getIteration();
        }
        return aligned_depth.defined() && aligned_depth.dim() == 2;
    }

    if (depthanything_global_alignment_valid_ &&
        ((use_metric3d &&
          applyMetricDepthAlignment(
              mono_prior,
              depthanything_global_align_scale_,
              depthanything_global_align_shift_,
              aligned_depth)) ||
         (!use_metric3d &&
          applyDepthAnythingAffineAlignment(
              mono_prior,
              depthanything_global_align_scale_,
              depthanything_global_align_shift_,
              aligned_depth)))) {
        return aligned_depth.defined() && aligned_depth.dim() == 2;
    }

    return false;
}

bool VoxelMapper::depthAnythingFillHolesWarmupReady()
{
    return !depthanything_fill_holes_warmup_ ||
           depthanything_fill_holes_warmup_iter_ <= 0 ||
           getIteration() >= depthanything_fill_holes_warmup_iter_;
}

void VoxelMapper::queueDepthAnythingFillHolesKeyframe(
    const std::shared_ptr<VoxelKeyframe>& pkf)
{
    if (!pkf || pkf->mono_prior_first_apply_iter_ >= 0) {
        return;
    }
    if (depthanything_fill_holes_pending_kfids_.insert(pkf->fid_).second) {
        depthanything_fill_holes_pending_kfs_.push_back(pkf);
    }
}

void VoxelMapper::applyDepthAnythingFillHolesKeyframes(
    const std::vector<std::shared_ptr<VoxelKeyframe>>& keyframes,
    bool seed_global_alignment)
{
    if (!depthanything_fill_holes_ || sensor_type_ != MONOCULAR || keyframes.empty()) {
        return;
    }

    if (seed_global_alignment) {
        for (const auto& pkf : keyframes) {
            if (pkf && pkf->mono_prior_first_apply_iter_ < 0) {
                updateMonoPriorGlobalAlignmentFromKeyframe(pkf);
            }
        }
    }

    for (const auto& pkf : keyframes) {
        if (!pkf || pkf->mono_prior_first_apply_iter_ >= 0) {
            continue;
        }
        const std::shared_ptr<VoxelKeyframe> scene_kf = scene_ ? scene_->getKeyframe(pkf->fid_) : nullptr;
        if (scene_kf && scene_kf.get() == pkf.get()) {
            increasePcdByKeyframeMonoPriorFillHoles(pkf);
        }
    }
}

void VoxelMapper::processDepthAnythingFillHolesWarmup()
{
    if (!depthanything_fill_holes_ ||
        !depthanything_fill_holes_warmup_ ||
        depthanything_fill_holes_warmup_flushed_ ||
        !depthAnythingFillHolesWarmupReady()) {
        return;
    }

    std::vector<std::shared_ptr<VoxelKeyframe>> pending;
    pending.reserve(depthanything_fill_holes_pending_kfs_.size());
    for (const auto& weak_pkf : depthanything_fill_holes_pending_kfs_) {
        std::shared_ptr<VoxelKeyframe> pkf = weak_pkf.lock();
        if (!pkf || pkf->mono_prior_first_apply_iter_ >= 0) {
            continue;
        }
        const std::shared_ptr<VoxelKeyframe> scene_kf = scene_ ? scene_->getKeyframe(pkf->fid_) : nullptr;
        if (scene_kf && scene_kf.get() == pkf.get()) {
            pending.push_back(pkf);
        }
    }

    depthanything_fill_holes_pending_kfs_.clear();
    depthanything_fill_holes_pending_kfids_.clear();
    depthanything_fill_holes_warmup_flushed_ = true;

    applyDepthAnythingFillHolesKeyframes(
        pending,
        /*seed_global_alignment=*/true);
}

void VoxelMapper::scheduleDepthAnythingFillHoles(
    const std::shared_ptr<VoxelKeyframe>& pkf)
{
    if (!depthanything_fill_holes_ || !pkf || pkf->mono_prior_first_apply_iter_ >= 0) {
        return;
    }

    if (!depthAnythingFillHolesWarmupReady()) {
        queueDepthAnythingFillHolesKeyframe(pkf);
        return;
    }

    processDepthAnythingFillHolesWarmup();
    if (pkf->mono_prior_first_apply_iter_ < 0) {
        applyDepthAnythingFillHolesKeyframes(
            std::vector<std::shared_ptr<VoxelKeyframe>>{pkf},
            /*seed_global_alignment=*/false);
    }
}

bool VoxelMapper::buildSparseDepthFromMapPoints(
    const sv::MiniCam& cam,
    int image_width,
    int image_height,
    torch::Tensor& sparse_uv,
    torch::Tensor& sparse_depth)
{
    // 0) Pull global SLAM point cloud (world coords) from the scene
    const auto& pcd = scene_->cached_point_cloud_;
    const int64_t M_total = static_cast<int64_t>(pcd.size());
    if (M_total == 0) {
        return false;
    }

    // 1) Pack world points into a host vector [M_total, 3]
    std::vector<float> host_pts;
    host_pts.reserve(3 * M_total);
    for (const auto& kv : pcd) {
        const Point3D& P = kv.second;          // you already fill xyz_ in run()
        host_pts.push_back(static_cast<float>(P.xyz_(0)));
        host_pts.push_back(static_cast<float>(P.xyz_(1)));
        host_pts.push_back(static_cast<float>(P.xyz_(2)));
    }

    auto opts_host = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor pts_world_cpu = torch::from_blob(
        host_pts.data(),
        {M_total, 3},
        opts_host);

    // Move to device and own the memory (clone())
    auto opts_dev = torch::TensorOptions().dtype(torch::kFloat32).device(mDevice);
    torch::Tensor pts_world = pts_world_cpu.clone().to(mDevice);   // [M,3]

    // 2) Transform world → camera using cam.w2c  (SVRaster-style)
    //
    // We build homogeneous coordinates [M,4] and multiply by w2c^T:
    //   X_cam = X_world_h @ w2c^T
    //
    torch::Tensor ones = torch::ones({M_total, 1}, opts_dev);
    torch::Tensor pts_world_h = torch::cat({pts_world, ones}, /*dim=*/1); // [M,4]

    torch::Tensor w2c = cam.w2c.to(mDevice);                                // [4,4]
    torch::Tensor pts_cam_h =
        torch::matmul(pts_world_h, w2c.transpose(0, 1));                    // [M,4]
    torch::Tensor pts_cam = pts_cam_h.index(
        {torch::indexing::Slice(), torch::indexing::Slice(0, 3)});          // [M,3]

    torch::Tensor X = pts_cam.index({torch::indexing::Slice(), 0}); // [M]
    torch::Tensor Y = pts_cam.index({torch::indexing::Slice(), 1}); // [M]
    torch::Tensor Z = pts_cam.index({torch::indexing::Slice(), 2}); // [M]

    // 3) Compute intrinsics from tanFOV + cx,cy (exactly what rasterizer uses)
    const float W = static_cast<float>(image_width);
    const float H = static_cast<float>(image_height);

    const float fx = 0.5f * W / cam.tanfovx;
    const float fy = 0.5f * H / cam.tanfovy;

    // u,v in pixel coords
    torch::Tensor u = fx * X / Z + cam.cx;   // [M]
    torch::Tensor v = fy * Y / Z + cam.cy;   // [M]

    // 4) Visibility & image bounds
    torch::Tensor valid =
        (Z > 0.0f) &
        (u >= 0.0f) & (u <= (W - 1.0f)) &
        (v >= 0.0f) & (v <= (H - 1.0f));     // [M]

    torch::Tensor valid_idx = torch::nonzero(valid).squeeze(1); // [M_vis]
    const int64_t M_vis = valid_idx.size(0);
    if (M_vis == 0) {
        return false;
    }

    // 5) Subsample to at most N_max points (same spirit as RGB-D version)
    const int64_t N_max = 3000;
    torch::Tensor chosen_idx;
    if (M_vis <= N_max) {
        chosen_idx = valid_idx;
    } else {
        const int64_t stride = std::max<int64_t>(int64_t(1), M_vis / N_max);
        torch::Tensor arange_idx = torch::arange(
            0, M_vis, stride,
            torch::TensorOptions().dtype(torch::kLong).device(valid_idx.device()));
        if (arange_idx.size(0) > N_max) {
            arange_idx = arange_idx.slice(0, 0, N_max);
        }
        chosen_idx = valid_idx.index_select(0, arange_idx); // [N]
    }

    const int64_t N = chosen_idx.size(0);
    if (N == 0) {
        return false;
    }

    // 6) Gather u, v, Z for the chosen points
    torch::Tensor u_chosen = u.index_select(0, chosen_idx); // [N]
    torch::Tensor v_chosen = v.index_select(0, chosen_idx); // [N]
    torch::Tensor z_chosen = Z.index_select(0, chosen_idx); // [N]

    // 7) Match SVRaster Camera.project(): 2 * u / W - 1, 2 * v / H - 1.
    torch::Tensor u_ndc =
        2.0f * (u_chosen / W) - 1.0f;                      // [N]
    torch::Tensor v_ndc =
        2.0f * (v_chosen / H) - 1.0f;                      // [N]

    sparse_uv    = torch::stack({u_ndc, v_ndc}, /*dim=*/1); // [N,2]
    sparse_depth = z_chosen;                                // [N]

    return true;
}

torch::Tensor VoxelMapper::computeSparseDepthLoss_Points(
    const std::shared_ptr<VoxelKeyframe>& kf,
    const sv::MiniCam& cam,
    int image_width,
    int image_height,
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    int iteration)
{
    auto zero = torch::zeros(
        {1},
        torch::TensorOptions().dtype(torch::kFloat32).device(mDevice));

    // 0) Weight or schedule off -> no contribution
    if (opt_params_.lambda_sparse_depth_ <= 0.0f)
        return zero;

    loss_utils::SparseDepthLoss sparse_depth_loss(opt_params_.sparse_depth_until_);
    if (!sparse_depth_loss.isActive(iteration))
        return zero;

    // 1) Get raw_T / raw_depth (SVRaster-style)
    auto it_T = render_pkg.find("raw_T");
    if (it_T == render_pkg.end())
        it_T = render_pkg.find("T");          // fallback

    auto it_depth = render_pkg.find("raw_depth");
    if (it_depth == render_pkg.end())
        it_depth = render_pkg.find("depth");  // fallback

    if (it_T == render_pkg.end() || it_depth == render_pkg.end()) {
        return zero;
    }

    torch::Tensor raw_T     = it_T->second.to(mDevice);
    torch::Tensor raw_depth = it_depth->second.to(mDevice);

    // 2) Build (sparse_uv, sparse_depth) from SLAM 3D points (SVRaster-style)
    torch::Tensor sparse_uv;     // [N,2]
    torch::Tensor sparse_depth;  // [N]
    if (!buildSparseDepthFromMapPoints(cam, image_width, image_height,
                                       sparse_uv, sparse_depth)) {
        // No visible 3D points for this viewpoint
        return zero;
    }
    // Avoid N=1 shape corner case inside SparseDepthLoss (grid_sample(...).squeeze()).
    if (!sparse_uv.defined() || !sparse_depth.defined() ||
        sparse_uv.dim() != 2 || sparse_uv.size(0) < 2 ||
        sparse_depth.numel() < 2) {
        return zero;
    }

    // 3) Low-level SparseDepthLoss (exact math as SVRaster’s __call__)
    torch::Tensor depth_loss =
        sparse_depth_loss(raw_T, raw_depth, sparse_uv, sparse_depth);

    return depth_loss;
}

torch::Tensor VoxelMapper::computeMonoPriorDepthLoss(
    const std::shared_ptr<VoxelKeyframe>& kf,
    const sv::MiniCam& cam,
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    int iteration)
{
    auto zero = torch::zeros(
        {1},
        torch::TensorOptions().dtype(torch::kFloat32).device(mDevice));

    if (opt_params_.lambda_depthanythingv2_ <= 0.0f) {
        return zero;
    }

    loss_utils::DepthAnythingv2Loss depthanything_loss(
        opt_params_.depthanythingv2_from_,
        opt_params_.depthanythingv2_end_,
        opt_params_.depthanythingv2_end_mult_);
    if (!depthanything_loss.isActive(iteration)) {
        return zero;
    }

    auto it_T = render_pkg.find("raw_T");
    if (it_T == render_pkg.end()) {
        it_T = render_pkg.find("T");
    }
    auto it_depth = render_pkg.find("raw_depth");
    if (it_depth == render_pkg.end()) {
        it_depth = render_pkg.find("depth");
    }
    if (it_T == render_pkg.end() || it_depth == render_pkg.end()) {
        return zero;
    }

    if (!ensureMonoPriorForKeyframe(kf) ||
        !kf->mono_prior_.defined() ||
        kf->mono_prior_.numel() == 0) {
        return zero;
    }

    torch::Tensor raw_T = it_T->second.to(mDevice, torch::kFloat32).contiguous();
    torch::Tensor raw_depth = it_depth->second.to(mDevice, torch::kFloat32).contiguous();
    torch::Tensor mono_depth =
        kf->mono_prior_.to(mDevice, torch::kFloat32).contiguous();

    if (mono_prior_loss_mode_ == "geosvr") {
        loss_utils::DepthAnythingv2UncertaintyLoss geosvr_loss(
            opt_params_.depthanythingv2_from_,
            opt_params_.depthanythingv2_end_,
            opt_params_.depthanythingv2_end_mult_,
            opt_params_.depthanythingv2_overall_,
            opt_params_.depthanythingv2_alpha_adjust_);
        if (!geosvr_loss.isActive(iteration)) {
            return zero;
        }

        torch::Tensor level_weight;
        if (opt_params_.enable_da2_uncertainty_ &&
            iteration >= opt_params_.level_uncertainty_from_) {
            auto it_feat = render_pkg.find("raw_feat");
            if (it_feat == render_pkg.end() || !it_feat->second.defined() ||
                it_feat->second.numel() == 0) {
                it_feat = render_pkg.find("feat");
            }
            auto it_level_T = render_pkg.find("raw_T");
            if (it_level_T == render_pkg.end() || !it_level_T->second.defined()) {
                it_level_T = render_pkg.find("T");
            }

            if (it_feat != render_pkg.end() && it_level_T != render_pkg.end() &&
                it_feat->second.defined() && it_feat->second.numel() > 0 &&
                it_level_T->second.defined() && it_level_T->second.numel() > 0 &&
                !torch::isnan(it_feat->second).any().item<bool>()) {
                using namespace torch::indexing;

                torch::Tensor feat = it_feat->second.to(mDevice, torch::kFloat32).contiguous();
                torch::Tensor T_for_level =
                    it_level_T->second.to(mDevice, torch::kFloat32).contiguous();

                if (feat.dim() == 4 && feat.size(0) == 1) {
                    feat = feat.squeeze(0);
                }
                if (T_for_level.dim() == 4 && T_for_level.size(0) == 1) {
                    T_for_level = T_for_level.squeeze(0);
                }
                if (T_for_level.dim() == 3 && T_for_level.size(0) >= 1) {
                    T_for_level = T_for_level.index({0});
                }
                if (feat.dim() == 3 && feat.size(0) >= 1 && T_for_level.dim() == 2) {
                    torch::Tensor level_map =
                        (feat / (1.0f - T_for_level).clamp_min(0.1f)).squeeze().detach();
                    if (level_map.dim() == 2 &&
                        (level_map.size(0) != raw_depth.size(-2) ||
                         level_map.size(1) != raw_depth.size(-1))) {
                        level_map = torch::nn::functional::interpolate(
                            level_map.unsqueeze(0).unsqueeze(0),
                            torch::nn::functional::InterpolateFuncOptions()
                                .size(std::vector<int64_t>{raw_depth.size(-2), raw_depth.size(-1)})
                                .mode(torch::kNearest)).squeeze();
                    }
                    if (level_map.dim() == 2) {
                        torch::Tensor level_min = level_map.min();
                        torch::Tensor level_weight_2d =
                            (level_map.max() - level_min) /
                            (level_map - level_min).clamp_min(1.0f);
                        level_weight =
                            level_weight_2d.unsqueeze(0).pow(opt_params_.power_level_uncertainty_);
                    }
                }
            }
        }

        if (kf->mono_prior_first_depth_loss_iter_ < 0) {
            kf->mono_prior_first_depth_loss_iter_ = iteration;
        }
        return geosvr_loss(
            raw_T,
            raw_depth,
            mono_depth,
            cam.near,
            iteration,
            sampleGeoSvrPatchSize(),
            level_weight);
    }

    if (mono_prior_loss_mode_ == "aligned") {
        using namespace torch::indexing;

        torch::Tensor depth = raw_depth;
        if (depth.dim() == 4 && depth.size(0) == 1) {
            depth = depth.squeeze(0);
        }
        if (depth.dim() == 2) {
            depth = depth.unsqueeze(0);
        }
        if (depth.dim() != 3 || depth.size(0) < 1) {
            return zero;
        }

        torch::Tensor T = raw_T;
        if (T.dim() == 4 && T.size(0) == 1) {
            T = T.squeeze(0);
        }
        if (T.dim() == 3 && T.size(0) >= 1) {
            T = T.index({0});
        }
        if (T.dim() != 2) {
            return zero;
        }

        const int H = static_cast<int>(depth.size(1));
        const int W = static_cast<int>(depth.size(2));
        torch::Tensor aligned_depth_cpu;
        if (!buildAlignedMonoPriorDepthForKeyframe(kf, cam, W, H, aligned_depth_cpu)) {
            return zero;
        }

        torch::Tensor target_depth =
            aligned_depth_cpu.to(mDevice, torch::kFloat32).contiguous();
        if (target_depth.dim() != 2) {
            return zero;
        }
        if (target_depth.size(0) != H || target_depth.size(1) != W) {
            target_depth = torch::nn::functional::interpolate(
                target_depth.unsqueeze(0).unsqueeze(0),
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{H, W})
                    .mode(torch::kBilinear)
                    .align_corners(false)).squeeze();
        }

        const float near_depth = std::max(1e-6f, cam.near);
        torch::Tensor alpha = (1.0f - T).clamp(0.0f, 1.0f);
        torch::Tensor render_depth = depth.index({0}) / alpha.clamp_min(1e-4f);
        torch::Tensor valid =
            torch::isfinite(render_depth) &
            torch::isfinite(target_depth) &
            (render_depth > near_depth) &
            (target_depth > near_depth) &
            (alpha > 0.5f);
        if (!valid.any().item<bool>()) {
            return zero;
        }

        torch::Tensor inv_render =
            (1.0f / render_depth.clamp_min(near_depth)).masked_select(valid);
        torch::Tensor inv_target =
            (1.0f / target_depth.clamp_min(near_depth)).masked_select(valid);
        torch::Tensor loss = (inv_render - inv_target).abs().mean();
        if (kf->mono_prior_first_depth_loss_iter_ < 0) {
            kf->mono_prior_first_depth_loss_iter_ = iteration;
        }

        if (opt_params_.depthanythingv2_end_ <= opt_params_.depthanythingv2_from_ ||
            opt_params_.depthanythingv2_end_mult_ == 1.0f) {
            return loss;
        }

        const float ratio = std::clamp(
            static_cast<float>(iteration - opt_params_.depthanythingv2_from_) /
                static_cast<float>(opt_params_.depthanythingv2_end_ -
                                   opt_params_.depthanythingv2_from_),
            0.0f,
            1.0f);
        const float mult = std::pow(opt_params_.depthanythingv2_end_mult_, ratio);
        return loss * mult;
    }

    if (kf->mono_prior_first_depth_loss_iter_ < 0) {
        kf->mono_prior_first_depth_loss_iter_ = iteration;
    }
    return depthanything_loss(raw_T, raw_depth, mono_depth, cam.near, iteration);
}

torch::Tensor VoxelMapper::computeRgbdDepthLoss(
    const std::shared_ptr<VoxelKeyframe>& kf,
    const sv::MiniCam& cam,
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    int iteration)
{
    auto zero = torch::zeros(
        {1},
        torch::TensorOptions().dtype(torch::kFloat32).device(mDevice));

    if (sensor_type_ != RGBD ||
        opt_params_.lambda_rgbd_depth_ <= 0.0f ||
        iteration < opt_params_.rgbd_depth_from_ ||
        iteration > opt_params_.rgbd_depth_end_ ||
        !kf ||
        kf->img_auxiliary_undist_.empty()) {
        return zero;
    }

    auto it_T = render_pkg.find("raw_T");
    if (it_T == render_pkg.end()) {
        it_T = render_pkg.find("T");
    }
    auto it_depth = render_pkg.find("raw_depth");
    if (it_depth == render_pkg.end()) {
        it_depth = render_pkg.find("depth");
    }
    if (it_T == render_pkg.end() || it_depth == render_pkg.end() ||
        !it_T->second.defined() || !it_depth->second.defined()) {
        return zero;
    }

    torch::Tensor raw_depth =
        it_depth->second.to(mDevice, torch::kFloat32).contiguous();
    if (raw_depth.dim() == 4 && raw_depth.size(0) == 1) {
        raw_depth = raw_depth.squeeze(0);
    }
    if (raw_depth.dim() == 2) {
        raw_depth = raw_depth.unsqueeze(0);
    }
    if (raw_depth.dim() != 3 || raw_depth.size(0) < 1) {
        return zero;
    }

    torch::Tensor raw_T =
        it_T->second.to(mDevice, torch::kFloat32).contiguous();
    if (raw_T.dim() == 4 && raw_T.size(0) == 1) {
        raw_T = raw_T.squeeze(0);
    }
    if (raw_T.dim() == 3 && raw_T.size(0) >= 1) {
        raw_T = raw_T.index({0});
    }
    if (raw_T.dim() != 2) {
        return zero;
    }

    const int H = static_cast<int>(raw_depth.size(1));
    const int W = static_cast<int>(raw_depth.size(2));
    if (raw_T.size(0) != H || raw_T.size(1) != W) {
        return zero;
    }

    cv::cuda::GpuMat depth_gpu;
    depth_gpu.upload(kf->img_auxiliary_undist_);
    torch::Tensor rgbd_depth =
        tensor_utils::cvGpuMat2TorchTensor_Float32(depth_gpu)
            .to(mDevice, torch::kFloat32)
            .contiguous();
    if (rgbd_depth.dim() == 3 && rgbd_depth.size(0) == 1) {
        rgbd_depth = rgbd_depth.squeeze(0);
    }
    if (rgbd_depth.dim() != 2) {
        return zero;
    }
    if (rgbd_depth.size(0) != H || rgbd_depth.size(1) != W) {
        rgbd_depth = torch::nn::functional::interpolate(
            rgbd_depth.unsqueeze(0).unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{H, W})
                .mode(torch::kNearest)).squeeze();
    }

    const float near_depth = std::max(1e-6f, cam.near);
    const float min_depth = std::max(RGBD_min_depth_, near_depth);
    torch::Tensor alpha = (1.0f - raw_T).clamp(0.0f, 1.0f);
    torch::Tensor valid =
        torch::isfinite(raw_depth.index({0})) &
        torch::isfinite(rgbd_depth) &
        (rgbd_depth > min_depth) &
        (rgbd_depth < RGBD_max_depth_) &
        (alpha > 0.8f);
    if (!valid.any().item<bool>()) {
        return zero;
    }

    torch::Tensor mask = valid.to(raw_depth.dtype());
    torch::Tensor render_inv =
        (1.0f / raw_depth.index({0}).clamp_min(near_depth)) * alpha;
    torch::Tensor target_inv =
        1.0f / rgbd_depth.clamp_min(near_depth);

    render_inv = render_inv * mask;
    target_inv = target_inv * mask;

    torch::Tensor loss = torch::nn::functional::mse_loss(
        render_inv,
        target_inv,
        torch::nn::functional::MSELossFuncOptions().reduction(torch::kMean));

    if (opt_params_.rgbd_depth_end_ <= opt_params_.rgbd_depth_from_ ||
        opt_params_.rgbd_depth_end_mult_ == 1.0f) {
        return loss;
    }

    const float ratio = std::clamp(
        static_cast<float>(iteration - opt_params_.rgbd_depth_from_) /
            static_cast<float>(opt_params_.rgbd_depth_end_ -
                               opt_params_.rgbd_depth_from_),
        0.0f,
        1.0f);
    const float mult = std::pow(opt_params_.rgbd_depth_end_mult_, ratio);
    return loss * mult;
}

torch::Tensor VoxelMapper::computeMonoPriorNormalLoss(
    const std::shared_ptr<VoxelKeyframe>& kf,
    const sv::MiniCam& cam,
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    int iteration)
{
    const bool use_geosvr_normal = (mono_prior_normal_mode_ == "geosvr");
    auto zero = torch::zeros(
        {1},
        torch::TensorOptions().dtype(torch::kFloat32).device(mDevice));

    if (opt_params_.lambda_depthanythingv2_normal_ <= 0.0f) {
        return zero;
    }
    if (iteration < opt_params_.depthanythingv2_normal_from_ ||
        iteration > opt_params_.depthanythingv2_normal_end_) {
        return zero;
    }

    auto it_T = render_pkg.find("raw_T");
    if (it_T == render_pkg.end()) {
        it_T = render_pkg.find("T");
    }
    auto it_depth = render_pkg.find("raw_depth");
    if (it_depth == render_pkg.end()) {
        it_depth = render_pkg.find("depth");
    }
    auto it_normal = render_pkg.find("raw_normal");
    if (it_normal == render_pkg.end()) {
        it_normal = render_pkg.find("normal");
    }
    if (it_T == render_pkg.end() || it_depth == render_pkg.end() || it_normal == render_pkg.end()) {
        return zero;
    }

    if (!ensureMonoPriorForKeyframe(kf) ||
        !kf->mono_prior_.defined() ||
        kf->mono_prior_.numel() == 0) {
        return zero;
    }

    torch::Tensor raw_T = it_T->second.to(mDevice, torch::kFloat32).contiguous();
    torch::Tensor raw_depth = it_depth->second.to(mDevice, torch::kFloat32).contiguous();
    torch::Tensor raw_normal = it_normal->second.to(mDevice, torch::kFloat32).contiguous();
    torch::Tensor mono_depth = kf->mono_prior_.to(mDevice, torch::kFloat32).contiguous();

    if (raw_depth.dim() == 4 && raw_depth.size(0) == 1) {
        raw_depth = raw_depth.squeeze(0);
    }
    if (raw_depth.dim() == 2) {
        raw_depth = raw_depth.unsqueeze(0);
    }
    if (raw_depth.dim() != 3) {
        return zero;
    }

    if (raw_T.dim() == 4 && raw_T.size(0) == 1) {
        raw_T = raw_T.squeeze(0);
    }
    if (raw_T.dim() == 2) {
        raw_T = raw_T.unsqueeze(0);
    }
    if (raw_T.dim() != 3) {
        return zero;
    }

    if (raw_normal.dim() == 4 && raw_normal.size(0) == 1) {
        raw_normal = raw_normal.squeeze(0);
    }
    if (raw_normal.dim() != 3 || raw_normal.size(0) < 3) {
        return zero;
    }
    if (raw_normal.size(0) > 3) {
        raw_normal = raw_normal.index({torch::indexing::Slice(0, 3)}).contiguous();
    }

    torch::Tensor Y = mono_depth;
    if (Y.dim() == 4 && Y.size(0) == 1) {
        Y = Y.squeeze(0);
    }
    if (Y.dim() == 3 && Y.size(0) == 1) {
        Y = Y.squeeze(0);
    }
    if (Y.dim() == 2) {
        Y = Y.unsqueeze(0).unsqueeze(0);
    } else if (Y.dim() == 3 && Y.size(0) == 1) {
        Y = Y.unsqueeze(0);
    }
    if (Y.dim() != 4) {
        return zero;
    }

    torch::Tensor alpha = 1.0f - raw_T.index({0});
    if (alpha.dim() == 2) {
        alpha = alpha.unsqueeze(0);
    }
    if (alpha.dim() != 3) {
        return zero;
    }
    if (alpha.size(0) > 1) {
        alpha = alpha.index({torch::indexing::Slice(0, 1)}).contiguous();
    }

    torch::Tensor target_normal;
    {
        torch::NoGradGuard no_grad;
        torch::Tensor target_depth;
        if (!use_geosvr_normal) {
            if (!buildAlignedMonoPriorDepthForKeyframe(
                    kf,
                    cam,
                    static_cast<int>(raw_normal.size(2)),
                    static_cast<int>(raw_normal.size(1)),
                    target_depth) ||
                !target_depth.defined() || target_depth.dim() != 2) {
                return zero;
            }
            const float tol_cos = std::cos(
                opt_params_.depthanythingv2_normal_tol_deg_ *
                static_cast<float>(M_PI) / 180.0f);
            target_normal = depth2normalSVRaster(
                cam,
                target_depth.to(mDevice, torch::kFloat32).contiguous(),
                opt_params_.depthanythingv2_normal_ks_,
                tol_cos);
        } else {
            torch::Tensor mono = Y;
            if (mono.sizes().slice(2) != raw_normal.sizes().slice(1)) {
                mono = torch::nn::functional::interpolate(
                    mono,
                    torch::nn::functional::InterpolateFuncOptions()
                        .size(std::vector<int64_t>{raw_normal.size(1), raw_normal.size(2)})
                        .mode(torch::kBilinear)
                        .align_corners(false));
            }

            if (isMetric3DModelId(mono_prior_model_id_)) {
                target_depth = mono.squeeze(0).squeeze(0).clamp_min(std::max(1e-6f, cam.near));
            } else {
                target_depth = (1.0f / mono.clamp_min(1e-3f)).squeeze(0).squeeze(0);
            }

            target_normal = depth2normalSVRaster(
                cam,
                target_depth,
                3,
                -1.0f);
        }
    }

    torch::Tensor render_normal = raw_normal;

    torch::Tensor mask =
        (target_normal != 0).any(0).unsqueeze(0).repeat({3, 1, 1}) &
        (alpha > 0.8f).repeat({3, 1, 1});
    mask = mask.to(render_normal.dtype());
    if (!mask.any().item<bool>()) {
        return zero;
    }

    torch::Tensor dot =
        (render_normal * target_normal).sum(0).clamp(-1.0f, 1.0f);
    torch::Tensor loss_map = (1.0f - dot) * mask;
    // L1 normal term disabled to match SVRaster/HI-SLAM2-style cosine normal supervision.
    // torch::Tensor loss_map_abs =
    //     (render_normal - target_normal).abs().sum(0) * mask;
    // torch::Tensor loss = loss_map.mean() + loss_map_abs.mean();
    torch::Tensor loss = loss_map.mean();

    if (use_geosvr_normal) {
        return loss;
    }

    if (opt_params_.depthanythingv2_normal_end_ <= opt_params_.depthanythingv2_normal_from_ ||
        opt_params_.depthanythingv2_normal_end_mult_ == 1.0f) {
        return loss;
    }

    const float ratio = std::clamp(
        static_cast<float>(iteration - opt_params_.depthanythingv2_normal_from_) /
            static_cast<float>(opt_params_.depthanythingv2_normal_end_ -
                               opt_params_.depthanythingv2_normal_from_),
        0.0f,
        1.0f);
    const float mult = std::pow(opt_params_.depthanythingv2_normal_end_mult_, ratio);
    return loss * mult;
}

torch::Tensor VoxelMapper::computeRgbdNormalLoss(
    const std::shared_ptr<VoxelKeyframe>& kf,
    const sv::MiniCam& cam,
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    int iteration)
{
    auto zero = torch::zeros(
        {1},
        torch::TensorOptions().dtype(torch::kFloat32).device(mDevice));

    if (sensor_type_ != RGBD ||
        opt_params_.lambda_rgbd_normal_ <= 0.0f ||
        iteration < opt_params_.rgbd_normal_from_ ||
        iteration > opt_params_.rgbd_normal_end_ ||
        !kf ||
        kf->img_auxiliary_undist_.empty()) {
        return zero;
    }

    auto it_T = render_pkg.find("raw_T");
    if (it_T == render_pkg.end()) {
        it_T = render_pkg.find("T");
    }
    auto it_normal = render_pkg.find("raw_normal");
    if (it_normal == render_pkg.end()) {
        it_normal = render_pkg.find("normal");
    }
    if (it_T == render_pkg.end() || it_normal == render_pkg.end() ||
        !it_T->second.defined() || !it_normal->second.defined()) {
        return zero;
    }

    torch::Tensor raw_T =
        it_T->second.to(mDevice, torch::kFloat32).contiguous();
    if (raw_T.dim() == 4 && raw_T.size(0) == 1) {
        raw_T = raw_T.squeeze(0);
    }
    if (raw_T.dim() == 3 && raw_T.size(0) >= 1) {
        raw_T = raw_T.index({0});
    }
    if (raw_T.dim() != 2) {
        return zero;
    }

    torch::Tensor render_normal =
        it_normal->second.to(mDevice, torch::kFloat32).contiguous();
    if (render_normal.dim() == 4 && render_normal.size(0) == 1) {
        render_normal = render_normal.squeeze(0);
    }
    if (render_normal.dim() != 3 || render_normal.size(0) < 3) {
        return zero;
    }
    if (render_normal.size(0) > 3) {
        render_normal =
            render_normal.index({torch::indexing::Slice(0, 3)}).contiguous();
    }

    const int H = static_cast<int>(render_normal.size(1));
    const int W = static_cast<int>(render_normal.size(2));
    if (raw_T.size(0) != H || raw_T.size(1) != W) {
        return zero;
    }

    torch::Tensor target_normal;
    torch::Tensor valid_support;
    {
        torch::NoGradGuard no_grad;

        cv::cuda::GpuMat depth_gpu;
        depth_gpu.upload(kf->img_auxiliary_undist_);
        torch::Tensor rgbd_depth =
            tensor_utils::cvGpuMat2TorchTensor_Float32(depth_gpu)
                .to(mDevice, torch::kFloat32)
                .contiguous();
        if (rgbd_depth.dim() == 3 && rgbd_depth.size(0) == 1) {
            rgbd_depth = rgbd_depth.squeeze(0);
        }
        if (rgbd_depth.dim() != 2) {
            return zero;
        }
        if (rgbd_depth.size(0) != H || rgbd_depth.size(1) != W) {
            rgbd_depth = torch::nn::functional::interpolate(
                rgbd_depth.unsqueeze(0).unsqueeze(0),
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{H, W})
                    .mode(torch::kNearest)).squeeze();
        }

        const float near_depth = std::max(1e-6f, cam.near);
        const float min_depth = std::max(RGBD_min_depth_, near_depth);
        torch::Tensor valid_depth =
            torch::isfinite(rgbd_depth) &
            (rgbd_depth > min_depth) &
            (rgbd_depth < RGBD_max_depth_);
        valid_support =
            validDepthSupportMask(valid_depth, opt_params_.rgbd_normal_ks_);
        if (!valid_support.any().item<bool>()) {
            return zero;
        }

        const float tol_cos = std::cos(
            opt_params_.rgbd_normal_tol_deg_ *
            static_cast<float>(M_PI) / 180.0f);
        target_normal = depth2normalSVRaster(
            cam,
            rgbd_depth.clamp_min(near_depth),
            opt_params_.rgbd_normal_ks_,
            tol_cos);
    }

    torch::Tensor alpha = (1.0f - raw_T).clamp(0.0f, 1.0f);
    torch::Tensor mask =
        (target_normal != 0).any(0) &
        valid_support &
        (alpha > 0.8f);
    if (!mask.any().item<bool>()) {
        return zero;
    }

    torch::Tensor dot =
        (render_normal * target_normal).sum(0).clamp(-1.0f, 1.0f);
    torch::Tensor normal_angle = 1.0f - dot;
    // L1 normal term disabled to match SVRaster/HI-SLAM2-style cosine normal supervision.
    // torch::Tensor normal_l1 = (render_normal - target_normal).abs().sum(0);
    torch::Tensor mask_f = mask.to(render_normal.dtype());
    torch::Tensor denom = mask_f.sum().clamp_min(1.0f);
    // torch::Tensor loss = ((normal_angle + normal_l1) * mask_f).sum() / denom;
    torch::Tensor loss = (normal_angle * mask_f).sum() / denom;

    if (opt_params_.rgbd_normal_end_ <= opt_params_.rgbd_normal_from_ ||
        opt_params_.rgbd_normal_end_mult_ == 1.0f) {
        return loss;
    }

    const float ratio = std::clamp(
        static_cast<float>(iteration - opt_params_.rgbd_normal_from_) /
            static_cast<float>(opt_params_.rgbd_normal_end_ -
                               opt_params_.rgbd_normal_from_),
        0.0f,
        1.0f);
    const float mult = std::pow(opt_params_.rgbd_normal_end_mult_, ratio);
    return loss * mult;
}

bool VoxelMapper::buildSparseDepthFromRGBD(
    const std::shared_ptr<VoxelKeyframe>& kf,
    torch::Tensor& sparse_uv,
    torch::Tensor& sparse_depth)
{
    // 1) Check that this keyframe actually has RGB-D depth
    if (kf->img_auxiliary_undist_.empty()) {
        // std::cout << "[SparseDepth] kf fid=" << kf->fid_
        //           << " has EMPTY img_auxiliary_undist_ (no RGB-D)\n";
        return false;
    }

    const int H = kf->image_height_;
    const int W = kf->image_width_;

    // 2) Upload cv::Mat depth to GPU and convert to torch::Tensor
    cv::cuda::GpuMat depth_gpu;
    depth_gpu.upload(kf->img_auxiliary_undist_);

    torch::Tensor depth =
        tensor_utils::cvGpuMat2TorchTensor_Float32(depth_gpu).to(mDevice);

    // Expected shapes: [H,W] or [1,H,W]
    if (depth.dim() == 3 && depth.size(0) == 1) {
        depth = depth.squeeze(0);        // [H,W]
    }
    else if (depth.dim() != 2) {
        // std::cerr << "[SparseDepth] Unexpected depth tensor shape for fid="
        //           << kf->fid_ << " : " << depth.sizes() << std::endl;
        return false;
    }

    // Debug: print depth stats
    auto dmin = depth.min().item<float>();
    auto dmax = depth.max().item<float>();
    // std::cout << "[SparseDepth] fid=" << kf->fid_
    //           << " depth min=" << dmin
    //           << " max=" << dmax << std::endl;

    // 3) Build a validity mask based on depth range
    //    (reuse Photo-SLAM's RGBD_min_depth_ / RGBD_max_depth_ thresholds)
    //    You may want to relax RGBD_min_depth_ to >0 first for debugging.
    torch::Tensor valid_mask =
        (depth > RGBD_min_depth_) & (depth < RGBD_max_depth_);   // [H,W]

    torch::Tensor flat_mask = valid_mask.view({-1});             // [H*W]
    torch::Tensor valid_idx = torch::nonzero(flat_mask).squeeze(1); // [M]

    const auto M = valid_idx.size(0);
    // std::cout << "[SparseDepth] fid=" << kf->fid_
    //           << " valid M=" << M << " (H*W=" << (H*W) << ")\n";

    if (M == 0) {
        // No valid depth pixels in this frame
        return false;
    }

    // 4) Subsample to at most N_max points to keep training cost reasonable
    const int64_t N_max = 3000;
    torch::Tensor chosen_idx;
    if (M <= N_max) {
        chosen_idx = valid_idx;
    } else {
        const int64_t stride = std::max<int64_t>(1, M / N_max);
        torch::Tensor arange_idx = torch::arange(
            0, M, stride,
            torch::TensorOptions().dtype(torch::kLong).device(valid_idx.device()));
        if (arange_idx.size(0) > N_max) {
            arange_idx = arange_idx.slice(0, 0, N_max);
        }
        chosen_idx = valid_idx.index_select(0, arange_idx);  // [N <= N_max]
    }

    const auto N = chosen_idx.size(0);
    if (N == 0) {
        // std::cout << "[SparseDepth] fid=" << kf->fid_
        //           << " selected N=0 after subsampling\n";
        return false;
    }

    // 5) Convert flat indices to (u,v) pixel coordinates
    torch::Tensor idx_u = chosen_idx.remainder(W);   // [N]
    torch::Tensor idx_v = chosen_idx / W;           // [N]

    // 6) Match SVRaster Camera.project() normalization for SparseDepthLoss.
    torch::Tensor u_ndc =
        2.0f * (idx_u.to(torch::kFloat32) / float(W)) - 1.0f;
    torch::Tensor v_ndc =
        2.0f * (idx_v.to(torch::kFloat32) / float(H)) - 1.0f;

    sparse_uv = torch::stack({u_ndc, v_ndc}, /*dim=*/1);  // [N,2]

    // 7) Gather depths at those indices
    torch::Tensor depth_flat = depth.view({H * W});        // [H*W]
    sparse_depth = depth_flat.index_select(0, chosen_idx); // [N]

    sparse_uv    = sparse_uv.to(mDevice);
    sparse_depth = sparse_depth.to(mDevice);

    // std::cout << "[SparseDepth] fid=" << kf->fid_
    //           << " final N=" << N << " depth_sparse min="
    //           << sparse_depth.min().item<float>()
    //           << " max=" << sparse_depth.max().item<float>()
    //           << std::endl;

    return true;
}

void VoxelMapper::debugDepthStats(const cv::Mat& depth_meters, int kf_id)
{
    if (depth_meters.empty() || depth_meters.type() != CV_32FC1) {
        // std::cout << "[DEPTH DEBUG] KF " << kf_id
        //           << " depth empty or wrong type: " << depth_meters.type() << "\n";
        return;
    }

    double min_val, max_val;
    cv::minMaxLoc(depth_meters, &min_val, &max_val);

    // Count valid depths ( >0 ) and zeros
    int valid_count = 0;
    int zero_count  = 0;
    double sum_valid = 0.0;

    const int H = depth_meters.rows;
    const int W = depth_meters.cols;

    for (int v = 0; v < H; ++v) {
        const float* row = depth_meters.ptr<float>(v);
        for (int u = 0; u < W; ++u) {
            float d = row[u];
            if (d > 0.0f && std::isfinite(d)) {
                valid_count++;
                sum_valid += d;
            } else if (d == 0.0f) {
                zero_count++;
            }
        }
    }

    double mean_valid = (valid_count > 0) ? (sum_valid / valid_count) : 0.0;

    // std::cout << "[DEPTH DEBUG] KF " << kf_id
    //           << " size=" << W << "x" << H
    //           << "  min=" << min_val
    //           << "  max=" << max_val
    //           << "  mean_valid=" << mean_valid
    //           << "  valid=" << valid_count
    //           << "  zeros=" << zero_count << "\n";
}

torch::Tensor VoxelMapper::computeSparseDepthLoss(
    const std::shared_ptr<VoxelKeyframe>& kf,
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    int iteration)
{
    auto zero = torch::zeros(
        {1},
        torch::TensorOptions().dtype(torch::kFloat32).device(mDevice));

    // 0) Weight or schedule off -> no contribution
    if (opt_params_.lambda_sparse_depth_ <= 0.0f)
        return zero;

    loss_utils::SparseDepthLoss sparse_depth_loss(opt_params_.sparse_depth_until_);
    if (!sparse_depth_loss.isActive(iteration))
        return zero;

    // 1) Try to get raw_T / raw_depth (SVRaster-style) or fall back to T / depth (your wrapper)
    auto it_T = render_pkg.find("raw_T");
    auto it_depth = render_pkg.find("raw_depth");

    if (it_T == render_pkg.end() || it_depth == render_pkg.end() ||
        !it_T->second.defined() || !it_depth->second.defined())
    {
        std::cerr << "[SparseDepth] raw_T/raw_depth missing or undefined at iter "
                  << iteration
                  << " (did you forget to enable output_T/output_depth in RenderOpts?)\n";
        return zero;
    }

    torch::Tensor raw_T     = it_T->second.to(mDevice);
    torch::Tensor raw_depth = it_depth->second.to(mDevice);

    // 2) Build sparse_uv and sparse_depth from RGB-D for this keyframe
    torch::Tensor sparse_uv;     // [N,2]
    torch::Tensor sparse_depth;  // [N]
    if (!buildSparseDepthFromRGBD(kf, sparse_uv, sparse_depth)) {
        std::cout << "No valid depth points" << std::endl;
        // No valid sparse depth points for this keyframe
        // (no RGB-D or all invalid)
        return zero;
    }
    // Avoid N=1 shape corner case inside SparseDepthLoss (grid_sample(...).squeeze()).
    if (!sparse_uv.defined() || !sparse_depth.defined() ||
        sparse_uv.dim() != 2 || sparse_uv.size(0) < 2 ||
        sparse_depth.numel() < 2) {
        return zero;
    }

    // 3) Low-level SparseDepthLoss (SVRaster math)
    torch::Tensor depth_loss =
        sparse_depth_loss(raw_T, raw_depth, sparse_uv, sparse_depth);

    // Optional debug
    // std::cout << "[SparseDepth] iter=" << iteration
    //           << " N=" << sparse_depth.size(0)
    //           << " loss=" << depth_loss.item<float>() << std::endl;

    return depth_loss;
}

// void VoxelMapper::applyFinalTsdfTransparency()
// {
//     if (!(sensor_type_ == RGBD && use_tsdf_mapping_)) {
//         std::cout << "[TSDF FINAL] RGBD/TSDF disabled, skipping final transparency.\n";
//         return;
//     }
//     if (!sdf_mapper_ || sdf_mapper_->tsdf_layer().size() == 0) {
//         std::cout << "[TSDF FINAL] no TSDF data, skipping final transparency.\n";
//         return;
//     }
//     std::unique_lock<std::mutex> lock_render(mutex_render_);

//     // ────────────────────────────────────────────────
//     // DEBUG: visualize *NVblox* TSDF voxels with distance < -0.15 (inside)
//     // ────────────────────────────────────────────────
//     {
//         nvblox::TsdfLayer& tsdf_layer = sdf_mapper_->tsdf_layer();
//         if (tsdf_layer.size() == 0) {
//             std::cout << "[TSDF DEBUG NVBLOX < 0.0] empty TSDF layer, skipping.\n";
//         } else {
//             const float block_size    = tsdf_layer.block_size();
//             const float voxel_size    = tsdf_layer.voxel_size();
//             const float inside_thresh = 0.0f;  // "definitely inside"
//             const float min_weight    = 1e-3f;   // consider as observed

//             using nvblox::Index3D;
//             using nvblox::TsdfBlock;
//             constexpr int kVoxelsPerSide = TsdfBlock::kVoxelsPerSide;

//             // 1) Enumerate all voxel centers in the TSDF layer
//             std::vector<Index3D> block_indices = tsdf_layer.getAllBlockIndices();
//             std::vector<nvblox::Vector3f> positions_L;
//             positions_L.reserve(
//                 block_indices.size() *
//                 kVoxelsPerSide * kVoxelsPerSide * kVoxelsPerSide);

//             for (const Index3D& block_index : block_indices) {
//                 // NOTE: we do NOT dereference block->voxels here.
//                 for (int vx = 0; vx < kVoxelsPerSide; ++vx) {
//                     for (int vy = 0; vy < kVoxelsPerSide; ++vy) {
//                         for (int vz = 0; vz < kVoxelsPerSide; ++vz) {
//                             nvblox::Index3D voxel_index(vx, vy, vz);
//                             nvblox::Vector3f center =
//                                 nvblox::getCenterPositionFromBlockIndexAndVoxelIndex(
//                                     block_size, block_index, voxel_index);
//                             positions_L.push_back(center);
//                         }
//                     }
//                 }
//             }

//             if (positions_L.empty()) {
//                 std::cout << "[TSDF DEBUG NVBLOX < 0.0] no voxel centers, skipping.\n";
//             } else {
//                 // 2) Query TSDF values for all these centers (safe w.r.t. GPU)
//                 std::vector<nvblox::TsdfVoxel> voxels;
//                 std::vector<bool>              success_flags;
//                 tsdf_layer.getVoxels(positions_L, &voxels, &success_flags);

//                 // 3) Filter by TSDF value and weight, build visualization tensors
//                 std::vector<float> centers_vec;  // x,y,z
//                 std::vector<float> sizes_vec;    // edge length
//                 centers_vec.reserve(positions_L.size() * 3);
//                 sizes_vec.reserve(positions_L.size());

//                 const size_t M = voxels.size();
//                 for (size_t i = 0; i < M; ++i) {
//                     if (!success_flags[i]) continue;
//                     const auto& v = voxels[i];
//                     if (v.weight < min_weight) continue;
//                     if (v.distance > inside_thresh) continue;  // want < 0.0

//                     const auto& c = positions_L[i];
//                     centers_vec.push_back(c.x());
//                     centers_vec.push_back(c.y());
//                     centers_vec.push_back(c.z());
//                     sizes_vec.push_back(voxel_size);
//                 }

//                 const int64_t K = static_cast<int64_t>(centers_vec.size() / 3);
//                 std::cout << "[TSDF DEBUG NVBLOX < 0.0] K=" << K
//                           << " TSDF voxels with distance < "
//                           << inside_thresh << " (NVblox grid)\n";

//                 if (K > 0) {
//                     auto opts = torch::TensorOptions()
//                                     .dtype(torch::kFloat32)
//                                     .device(torch::kCPU);

//                     torch::Tensor centers_tsdf =
//                         torch::from_blob(centers_vec.data(), {K, 3}, opts).clone();
//                     torch::Tensor sizes_tsdf =
//                         torch::from_blob(sizes_vec.data(), {K, 1}, opts).clone();

//                     // [K,4] RGBA: red-ish for inside
//                     torch::Tensor colors_tsdf =
//                         torch::zeros({K, 4}, centers_tsdf.options());
//                     using torch::indexing::Slice;
//                     colors_tsdf.index_put_({Slice(), 0}, 1.0f);  // R
//                     colors_tsdf.index_put_({Slice(), 3}, 0.6f);  // alpha

//                     sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
//                         centers_tsdf,
//                         sizes_tsdf,
//                         colors_tsdf,
//                         getIteration(),
//                         "world/nvblox_tsdf_voxels_neg015"
//                     );
//                 }
//             }
//         }
//     }
//     // ────────────────────────────────────────────────
//     // DEBUG: visualize *NVblox* TSDF voxels with distance == 0.0
//     // ────────────────────────────────────────────────
//     {
//         nvblox::TsdfLayer& tsdf_layer = sdf_mapper_->tsdf_layer();
//         if (tsdf_layer.size() == 0) {
//             std::cout << "[TSDF DEBUG NVBLOX 0.0] empty TSDF layer, skipping.\n";
//         } else {
//             const float block_size  = tsdf_layer.block_size();
//             const float voxel_size  = tsdf_layer.voxel_size();
//             const float target_tsdf = 0.0f;
//             const float min_weight  = 1e-3f;
//             const float eps         = 0.02f; 

//             using nvblox::Index3D;
//             using nvblox::TsdfBlock;
//             constexpr int kVoxelsPerSide = TsdfBlock::kVoxelsPerSide;

//             // 1) Build positions for ALL TSDF voxel centers
//             std::vector<Index3D>   block_indices = tsdf_layer.getAllBlockIndices();
//             std::vector<nvblox::Vector3f> positions_L;
//             positions_L.reserve(block_indices.size() * kVoxelsPerSide * kVoxelsPerSide * kVoxelsPerSide);

//             for (const Index3D& block_index : block_indices) {
//                 TsdfBlock::ConstPtr block = tsdf_layer.getBlockAtIndex(block_index);
//                 if (!block) continue;

//                 for (int vx = 0; vx < kVoxelsPerSide; ++vx) {
//                     for (int vy = 0; vy < kVoxelsPerSide; ++vy) {
//                         for (int vz = 0; vz < kVoxelsPerSide; ++vz) {
//                             nvblox::Index3D voxel_index(vx, vy, vz);
//                             nvblox::Vector3f center =
//                                 nvblox::getCenterPositionFromBlockIndexAndVoxelIndex(
//                                     block_size, block_index, voxel_index);
//                             positions_L.push_back(center);
//                         }
//                     }
//                 }
//             }

//             if (positions_L.empty()) {
//                 std::cout << "[TSDF DEBUG NVBLOX 0.0] no voxel centers, skipping.\n";
//             } else {
//                 // 2) Query TSDF values for all these centers
//                 std::vector<nvblox::TsdfVoxel> voxels;
//                 std::vector<bool>              success_flags;
//                 tsdf_layer.getVoxels(positions_L, &voxels, &success_flags);

//                 // 3) Collect those with distance == 0.0 and enough weight
//                 std::vector<float> centers_vec;  // x,y,z
//                 std::vector<float> sizes_vec;    // one per voxel
//                 centers_vec.reserve(positions_L.size() * 3);
//                 sizes_vec.reserve(positions_L.size());

//                 const size_t M = voxels.size();
//                 for (size_t i = 0; i < M; ++i) {
//                     if (!success_flags[i]) continue;
//                     const auto& v = voxels[i];
//                     if (v.weight < min_weight) continue;
//                     // if (v.distance != target_tsdf) continue;
//                     // 2) Keep a *band* around zero, not exact equality
//                     if (std::fabs(v.distance - target_tsdf) > eps) continue;

//                     const auto& c = positions_L[i];
//                     centers_vec.push_back(c.x());
//                     centers_vec.push_back(c.y());
//                     centers_vec.push_back(c.z());
//                     sizes_vec.push_back(voxel_size);
//                 }

//                 const int64_t K = static_cast<int64_t>(centers_vec.size() / 3);
//                 std::cout << "[TSDF DEBUG NVBLOX 0] K=" << K
//                         << " TSDF voxels with distance == "
//                         << target_tsdf << " (NVblox grid)\n";

//                 if (K > 0) {
//                     auto opts = torch::TensorOptions()
//                                     .dtype(torch::kFloat32)
//                                     .device(torch::kCPU);

//                     torch::Tensor centers_tsdf =
//                         torch::from_blob(centers_vec.data(), {K, 3}, opts).clone();
//                     torch::Tensor sizes_tsdf =
//                         torch::from_blob(sizes_vec.data(), {K, 1}, opts).clone();

//                     torch::Tensor colors_tsdf =
//                         torch::zeros({K, 4}, centers_tsdf.options());
//                     using torch::indexing::Slice;
//                     colors_tsdf.index_put_({Slice(), 1}, 1.0f);  // G
//                     colors_tsdf.index_put_({Slice(), 2}, 1.0f);  // B
//                     colors_tsdf.index_put_({Slice(), 3}, 0.6f);  // alpha

//                     sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
//                         centers_tsdf,
//                         sizes_tsdf,
//                         colors_tsdf,
//                         getIteration(),
//                         "world/nvblox_tsdf_voxels_0p0"
//                     );
//                 }
//             }
//         }
//     }
//     // ────────────────────────────────────────────────
//     // DEBUG: visualize *NVblox* TSDF voxels with distance > 0.15 (outside)
//     // ────────────────────────────────────────────────
//     {
//         nvblox::TsdfLayer& tsdf_layer = sdf_mapper_->tsdf_layer();
//         if (tsdf_layer.size() == 0) {
//             std::cout << "[TSDF DEBUG NVBLOX > 0.15] empty TSDF layer, skipping.\n";
//         } else {
//             const float block_size    = tsdf_layer.block_size();
//             const float voxel_size    = tsdf_layer.voxel_size();
//             const float inside_thresh = 0.15f;  // "definitely inside"
//             const float min_weight    = 1e-3f;   // consider as observed

//             using nvblox::Index3D;
//             using nvblox::TsdfBlock;
//             constexpr int kVoxelsPerSide = TsdfBlock::kVoxelsPerSide;

//             // 1) Enumerate all voxel centers in the TSDF layer
//             std::vector<Index3D> block_indices = tsdf_layer.getAllBlockIndices();
//             std::vector<nvblox::Vector3f> positions_L;
//             positions_L.reserve(
//                 block_indices.size() *
//                 kVoxelsPerSide * kVoxelsPerSide * kVoxelsPerSide);

//             for (const Index3D& block_index : block_indices) {
//                 // NOTE: we do NOT dereference block->voxels here.
//                 for (int vx = 0; vx < kVoxelsPerSide; ++vx) {
//                     for (int vy = 0; vy < kVoxelsPerSide; ++vy) {
//                         for (int vz = 0; vz < kVoxelsPerSide; ++vz) {
//                             nvblox::Index3D voxel_index(vx, vy, vz);
//                             nvblox::Vector3f center =
//                                 nvblox::getCenterPositionFromBlockIndexAndVoxelIndex(
//                                     block_size, block_index, voxel_index);
//                             positions_L.push_back(center);
//                         }
//                     }
//                 }
//             }

//             if (positions_L.empty()) {
//                 std::cout << "[TSDF DEBUG NVBLOX < 0.0] no voxel centers, skipping.\n";
//             } else {
//                 // 2) Query TSDF values for all these centers (safe w.r.t. GPU)
//                 std::vector<nvblox::TsdfVoxel> voxels;
//                 std::vector<bool>              success_flags;
//                 tsdf_layer.getVoxels(positions_L, &voxels, &success_flags);

//                 // 3) Filter by TSDF value and weight, build visualization tensors
//                 std::vector<float> centers_vec;  // x,y,z
//                 std::vector<float> sizes_vec;    // edge length
//                 centers_vec.reserve(positions_L.size() * 3);
//                 sizes_vec.reserve(positions_L.size());

//                 const size_t M = voxels.size();
//                 for (size_t i = 0; i < M; ++i) {
//                     if (!success_flags[i]) continue;
//                     const auto& v = voxels[i];
//                     if (v.weight < min_weight) continue;
//                     if (v.distance < inside_thresh) continue;  

//                     const auto& c = positions_L[i];
//                     centers_vec.push_back(c.x());
//                     centers_vec.push_back(c.y());
//                     centers_vec.push_back(c.z());
//                     sizes_vec.push_back(voxel_size);
//                 }

//                 const int64_t K = static_cast<int64_t>(centers_vec.size() / 3);
//                 std::cout << "[TSDF DEBUG NVBLOX < 0.0] K=" << K
//                           << " TSDF voxels with distance < "
//                           << inside_thresh << " (NVblox grid)\n";

//                 if (K > 0) {
//                     auto opts = torch::TensorOptions()
//                                     .dtype(torch::kFloat32)
//                                     .device(torch::kCPU);

//                     torch::Tensor centers_tsdf =
//                         torch::from_blob(centers_vec.data(), {K, 3}, opts).clone();
//                     torch::Tensor sizes_tsdf =
//                         torch::from_blob(sizes_vec.data(), {K, 1}, opts).clone();

//                     // [K,4] RGBA: red-ish for inside
//                     torch::Tensor colors_tsdf =
//                         torch::zeros({K, 4}, centers_tsdf.options());
//                     using torch::indexing::Slice;
//                     colors_tsdf.index_put_({Slice(), 0}, 1.0f);  // R
//                     colors_tsdf.index_put_({Slice(), 3}, 0.6f);  // alpha

//                     sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
//                         centers_tsdf,
//                         sizes_tsdf,
//                         colors_tsdf,
//                         getIteration(),
//                         "world/nvblox_tsdf_voxels_p015"
//                     );
//                 }
//             }
//         }
//     }
//     // ────────────────────────────────────────────────
//     // sampleTsdfAtPointsWorld
//     // ────────────────────────────────────────────────
//     torch::Tensor centers_world = voxel_model_->voxCenter();  // [N,3]
//     if (!centers_world.defined() || centers_world.numel() == 0) {
//         std::cout << "[TSDF FINAL] no voxels, skipping final transparency.\n";
//         return;
//     }
//     const int N = centers_world.size(0);
//     // Sample TSDF for all voxel centers
//     torch::Tensor tsdf_vals = sampleTsdfAtPointsWorld(centers_world);  // [N]
//     // 1) finite mask
//     torch::Tensor tsdf_finite_mask = tsdf_vals.isfinite();  // [N]
//     if (!tsdf_finite_mask.any().item<bool>()) {
//         std::cout << "[TSDF FINAL] no finite TSDF values, skipping.\n";
//         return;
//     }
//     // // ────────────────────────────────────────────────
//     // // DEBUG: visualize voxels with TSDF == 0.2
//     // // ────────────────────────────────────────────────
//     // {
//     //     const float target_tsdf = 0.2f;    // exact value you want to inspect
//     //     // mask for finite TSDF AND exactly equal to target_tsdf
//     //     torch::Tensor eq_mask = tsdf_finite_mask & tsdf_vals.eq(target_tsdf); // [N]
//     //     eq_mask = eq_mask.to(torch::kBool);

//     //     auto n_eq = eq_mask.sum().item<int64_t>();
//     //     if (n_eq == 0) {
//     //         std::cout << "[TSDF DEBUG 0.2] no voxels with TSDF == "
//     //                 << target_tsdf << ", skipping visualization.\n";
//     //     } else {
//     //         // indices of those voxels
//     //         torch::Tensor idx_eq = eq_mask.nonzero().squeeze(1);  // [K]
//     //         const int64_t K = idx_eq.size(0);

//     //         // TSDF stats only on that exact set
//     //         torch::Tensor tsdf_eq_vals = tsdf_vals.index_select(0, idx_eq);  // [K]
//     //         float tsdf_eq_min  = tsdf_eq_vals.min().item<float>();
//     //         float tsdf_eq_max  = tsdf_eq_vals.max().item<float>();
//     //         float tsdf_eq_mean = tsdf_eq_vals.mean().item<float>();

//     //         std::cout << "[TSDF DEBUG 0.2] K=" << K
//     //                 << " tsdf_eq_min="  << tsdf_eq_min
//     //                 << " tsdf_eq_max="  << tsdf_eq_max
//     //                 << " tsdf_eq_mean=" << tsdf_eq_mean
//     //                 << " (TSDF == " << target_tsdf << ")\n";

//     //         // ---- build centers & sizes for visualization ----
//     //         torch::Tensor centers_0p2 =
//     //             centers_world.index_select(0, idx_eq).clone(); // [K,3]

//     //         torch::Tensor sizes_all = voxel_model_->voxSize();  // [N] or [N,1]
//     //         if (sizes_all.dim() == 1) {
//     //             sizes_all = sizes_all.view({N, 1});
//     //         } else if (sizes_all.dim() == 2 && sizes_all.size(1) == 1) {
//     //             // ok
//     //         } else {
//     //             sizes_all = sizes_all.reshape({N, 1});
//     //         }
//     //         torch::Tensor sizes_0p2 =
//     //             sizes_all.index_select(0, idx_eq).clone();       // [K,1]

//     //         // RGBA: bright green, semi-transparent
//     //         torch::Tensor colors_0p2 =
//     //             torch::zeros({K, 4}, centers_0p2.options());  // [K,4]
//     //         using torch::indexing::Slice;
//     //         colors_0p2.index_put_({Slice(), 1}, 1.0f);  // G channel
//     //         colors_0p2.index_put_({Slice(), 3}, 0.8f);  // alpha

//     //         std::cout << "[TSDF DEBUG 0.2] visualizing " << K
//     //                 << " voxels as 'world/voxels_tsdf_0p2' in rerun\n";

//     //         sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
//     //             centers_0p2,
//     //             sizes_0p2,
//     //             colors_0p2,
//     //             getIteration(),
//     //             "world/voxels_tsdf_0p2"
//     //         );
//     //     }
//     // }

//     // 2) NVBlox freespace threshold
//     nvblox::FreespaceIntegrator freespace;
//     const float tsdf_free_thresh_m = freespace.max_tsdf_distance_for_occupancy_m(); //0.15
//     const float tsdf_margin_m      = 0.0f;
//     const float thresh             = tsdf_free_thresh_m + tsdf_margin_m;

//     torch::Tensor tsdf_free_mask = tsdf_vals > thresh;        // [N]
//     torch::Tensor tsdf_mask      = (tsdf_finite_mask & tsdf_free_mask).to(torch::kBool);

//     const int64_t n_tsdf = tsdf_mask.sum().item<int64_t>();
//     if (n_tsdf == 0) {
//         std::cout << "[TSDF FINAL] no voxels classified as free, nothing to fade.\n";
//         return;
//     }

//     // Optional stats (only over finite TSDF)
//     auto tsdf_finite = tsdf_vals.masked_select(tsdf_finite_mask);
//     float tsdf_min  = tsdf_finite.min().item<float>();
//     float tsdf_max  = tsdf_finite.max().item<float>();
//     float tsdf_mean = tsdf_finite.mean().item<float>();
//     std::cout << "[TSDF FINAL] N=" << N
//               << " tsdf_min="  << tsdf_min
//               << " tsdf_max="  << tsdf_max
//               << " tsdf_mean=" << tsdf_mean
//               << " free_mask_sum=" << n_tsdf
//               << " thresh=" << thresh
//               << "\n";
//     // ------------------------------------------------------------------
//     // DEBUG MODE: operate on ONE voxel: the furthest free-space voxel
//     // ------------------------------------------------------------------
//     {
//         // indices of free-space voxels
//         torch::Tensor idx_free = tsdf_mask.nonzero().squeeze(1);  // [K]
//         TORCH_CHECK(idx_free.numel() > 0, "tsdf_mask has no true entries unexpectedly.");

//         // TSDF values only for free voxels
//         torch::Tensor tsdf_free_vals = tsdf_vals.index_select(0, idx_free); // [K]

//         // argmax over that subset
//         auto max_pair      = tsdf_free_vals.max(0);                  // (values, indices)
//         int64_t local_arg  = std::get<1>(max_pair).item<int64_t>();  // index in [0..K-1]
//         int64_t voxel_idx  = idx_free[local_arg].item<int64_t>();    // index in [0..N-1]

//         float tsdf_vox = tsdf_vals[voxel_idx].item<float>();
//         std::cout << "[TSDF DEBUG] picked voxel_idx=" << voxel_idx
//                   << " with tsdf=" << tsdf_vox << " (max over free-space set)\n";

//         // --- visualize this voxel in Rerun ---
//         // Make a 1-element index tensor
//         auto idx_options = torch::TensorOptions().dtype(torch::kLong).device(centers_world.device());
//         torch::Tensor idx_single = torch::tensor({voxel_idx}, idx_options); // [1]

//         torch::Tensor center_debug = centers_world.index_select(0, idx_single); // [1,3]

//         torch::Tensor size_all = voxel_model_->voxSize(); // [N] or [N,1]
//         if (size_all.dim() == 1) {
//             size_all = size_all.view({N, 1});
//         } else if (size_all.dim() == 2 && size_all.size(1) == 1) {
//             // ok
//         } else {
//             size_all = size_all.reshape({N, 1});
//         }
//         torch::Tensor size_debug = size_all.index_select(0, idx_single); // [1,1]

//         // RGBA: red, semi-transparent
//         torch::Tensor colors_debug = torch::zeros({1, 4}, center_debug.options());
//         colors_debug.index_put_({0, 0}, 1.0f);  // R
//         colors_debug.index_put_({0, 3}, 0.8f);  // alpha

//         sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
//             center_debug,
//             size_debug,
//             colors_debug,
//             getIteration(),
//             "world/tsdf_debug_voxel"  // separate entity
//         );

//         // ---- make ONLY this voxel transparent via its 8 grid points ----
//         const float geo_value_tsdf_free = -30.0f;  // or whatever you're using
//         voxel_model_->applySingleVoxelTsdfTransparency(voxel_idx, geo_value_tsdf_free);

//         std::cout << "[TSDF DEBUG] applied transparency to single voxel_idx="
//                   << voxel_idx << "\n";

//         // IMPORTANT: early return here so we don't call the bulk path
//         // return;
//     }

//     // Push log-density of those free voxels down to a very low value
//     const float geo_value_tsdf_free = -30.0f;   //
//     voxel_model_->applyTsdfTransparency(tsdf_mask, geo_value_tsdf_free);

//     // --- OPTIONAL: debug visualization of TSDF-free voxels as a separate entity ---
//     try {
//         auto idx = tsdf_mask.nonzero().squeeze(1);   // [K]
//         if (idx.numel() > 0) {
//             torch::Tensor centers_tsdf = centers_world.index({idx}).clone();  // [K,3]
//             torch::Tensor sizes_tsdf   = voxel_model_->voxSize();             // [N] or [N,1]
//             if (sizes_tsdf.dim() == 1) {
//                 sizes_tsdf = sizes_tsdf.view({N, 1});
//             } else if (sizes_tsdf.dim() == 2 && sizes_tsdf.size(1) == 1) {
//                 // ok
//             } else {
//                 sizes_tsdf = sizes_tsdf.reshape({N, 1});
//             }
//             sizes_tsdf = sizes_tsdf.index({idx}).clone();                     // [K,1]

//             const auto K = centers_tsdf.size(0);
//             torch::Tensor colors_tsdf = torch::zeros({K, 4}, centers_tsdf.options());
//             // e.g. blue-transparent
//             colors_tsdf.index_put_({torch::indexing::Slice(), 2}, 1.0f);  // B
//             colors_tsdf.index_put_({torch::indexing::Slice(), 3}, 0.7f);  // alpha

//             std::cout << "[TSDF FINAL] visualizing " << K
//                       << " TSDF-free voxels as 'world/voxels_tsdf' in rerun\n";

//             sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
//                 centers_tsdf,
//                 sizes_tsdf,
//                 colors_tsdf,
//                 getIteration(),           // current iter
//                 "world/voxels_tsdf_transparent"       // separate entity
//             );
//         }
//     } catch (const std::exception& e) {
//         std::cerr << "[TSDF FINAL] rerun visualization error: " << e.what() << "\n";
//     }
// }

// Add training with optimization loop
void VoxelMapper::trainForOneIteration()
{
    // 1) bump global iteration counter
    increaseIteration(1);
    auto iter_start_timing = std::chrono::steady_clock::now();

    sv::RenderOpts ropts;

    // 2) pick a random keyframe from the sliding window
    std::shared_ptr<VoxelKeyframe> viewpoint_cam = useOneRandomSlidingWindowKeyframe();
    if (!viewpoint_cam) {
        increaseIteration(-1);
        return;
    }
    writeKeyframeUsedTimes(result_dir_ / "used_times");

    const int iter = getIteration();
    int training_level = num_gaus_pyramid_sub_levels_;
    int image_height, image_width;
    torch::Tensor gt_image, mask;

    if (isdoingGausPyramidTraining())
         training_level = viewpoint_cam->getCurrentGausPyramidLevel();
    if (training_level == num_gaus_pyramid_sub_levels_) {
        // std::cout << "training full res\n";
        image_height = viewpoint_cam->image_height_;
        image_width = viewpoint_cam->image_width_;
        gt_image = viewpoint_cam->original_image_
                            .to(mDevice);          // (3,H,W)
        mask = undistort_mask_[viewpoint_cam->camera_id_]
                                    .to(mDevice)
                                    .to(torch::kFloat32); // (3,H,W)
    }
    else {
        image_height = viewpoint_cam->gaus_pyramid_height_[training_level];
        image_width = viewpoint_cam->gaus_pyramid_width_[training_level];
        gt_image = viewpoint_cam->gaus_pyramid_original_image_[training_level].to(mDevice); 
        mask = scene_->cameras_.at(viewpoint_cam->camera_id_).gaus_pyramid_undistort_mask_[training_level].to(mDevice).to(torch::kFloat32);
    }
    // std::cout << "gt_image" << gt_image << std::endl;
    // 4) grow SH degree every 1000 iterations (locked during render)
    std::unique_lock<std::mutex> lock_render(mutex_render_);
    if (getIteration() % 1000 == 0 && default_sh_ < model_params_.sh_degree_)
    {
        default_sh_ += 1;
        std::cout << "[VoxelMapper] SH degree: " << default_sh_ << std::endl;
    }    
    voxel_model_->setShDegree(default_sh_);

    // // Follow SVRaster train.py:
    // // keep ss=1.0 early, then switch to augmentation or the model default.
    // ropts.ss = 1.0f;
    // if (iter > 1000) {
    //     if (opt_params_.ss_aug_max_ > 1.0f) {
    //         static thread_local std::mt19937 rng{std::random_device{}()};
    //         std::uniform_real_distribution<float> dist(1.0f, opt_params_.ss_aug_max_);
    //         ropts.ss = dist(rng);
    //     } else {
    //         ropts.ss = std::nullopt;
    //     }
    // }

    // Use default super-sampling option (enable after 1000 iters)
    if (iter > 200) {
        if (opt_params_.ss_aug_max_ > 1.0f) {
            static thread_local std::mt19937 rng{std::random_device{}()};
            std::uniform_real_distribution<float> dist(1.0f, opt_params_.ss_aug_max_);
            ropts.ss = dist(rng);                 // tr_render_opt['ss'] = U(1, ss_aug_max)
        } else {
            ropts.ss = std::nullopt;              // pop('ss') -> use model default self.ss
        }
    } else {
        ropts.ss = 1.0f;                           // disable supersampling at first
    }
    ropts.ss = 1.0f;  

    const bool need_sparse_depth = (opt_params_.lambda_sparse_depth_ > 0.0f) && (iter <= opt_params_.sparse_depth_until_);
    const bool need_depthanythingv2 =
        (opt_params_.lambda_depthanythingv2_ > 0.0f) &&
        (iter >= opt_params_.depthanythingv2_from_) &&
        (iter <= opt_params_.depthanythingv2_end_);
    const bool need_rgbd_depth =
        (sensor_type_ == RGBD) &&
        (opt_params_.lambda_rgbd_depth_ > 0.0f) &&
        (iter >= opt_params_.rgbd_depth_from_) &&
        (iter <= opt_params_.rgbd_depth_end_);
    const bool need_mono_prior_normal =
        (opt_params_.lambda_depthanythingv2_normal_ > 0.0f) &&
        (iter >= opt_params_.depthanythingv2_normal_from_) &&
        (iter <= opt_params_.depthanythingv2_normal_end_);
    const bool need_rgbd_normal =
        (sensor_type_ == RGBD) &&
        (opt_params_.lambda_rgbd_normal_ > 0.0f) &&
        (iter >= opt_params_.rgbd_normal_from_) &&
        (iter <= opt_params_.rgbd_normal_end_);
    const bool need_T_concen = (opt_params_.lambda_T_concen_ > 0.0f);
    const bool need_T_inside = (opt_params_.lambda_T_inside_ > 0.0f);
    const bool need_normal_dmean =
        (opt_params_.lambda_normal_dmean_ > 0.0f) &&
        (iter >= opt_params_.n_dmean_from_) &&
        (iter <= opt_params_.n_dmean_end_);
    ropts.output_T =
        need_T_concen || need_T_inside || need_sparse_depth || need_normal_dmean ||
        need_depthanythingv2 || need_rgbd_depth || need_mono_prior_normal ||
        need_rgbd_normal;
    ropts.output_depth = need_sparse_depth || need_normal_dmean || need_depthanythingv2 ||
                         need_rgbd_depth || need_mono_prior_normal;
    ropts.output_normal = need_normal_dmean || need_mono_prior_normal || need_rgbd_normal;

    // if (opt_params_.lambda_T_inside_ > 0.0f) {
    //     ropts.output_T = true;
    // }

    if (iter >= opt_params_.dist_from_ && opt_params_.lambda_dist_ > 0.0f) {
        ropts.lambda_dist = opt_params_.lambda_dist_;
    }

    if (iter >= opt_params_.rectifiy_from_ &&
        opt_params_.lambda_rectify_ > 0.0f) {
        ropts.lambda_ascending = -opt_params_.lambda_rectify_;
    } else if (iter >= opt_params_.ascending_from_ &&
               opt_params_.lambda_ascending_ > 0.0f) {
        ropts.lambda_ascending = opt_params_.lambda_ascending_;
    }

    if (iter > opt_params_.scaling_penalty_from_ &&
        iter <= opt_params_.scaling_penalty_end_ &&
        opt_params_.lambda_scaling_penalty_ > 0.0f) {
        auto vox_size = voxel_model_->voxSize();
        if (vox_size.defined() && vox_size.numel() > 0) {
            ropts.lambda_scaling_penalty = opt_params_.lambda_scaling_penalty_;
            ropts.min_voxel_size =
                vox_size.to(torch::kFloat32).min().item<float>();
        }
    }

    if (opt_params_.lambda_R_concen_ > 0.0f) {
        ropts.lambda_R_concen = opt_params_.lambda_R_concen_;
        ropts.gt_color = gt_image;
    }

    if (mono_prior_loss_mode_ == "geosvr" &&
        need_depthanythingv2 &&
        opt_params_.enable_da2_uncertainty_ &&
        iter >= opt_params_.level_uncertainty_from_) {
        ropts.vox_feats = voxel_model_->octLevel().detach().to(torch::kFloat32).contiguous();
    }

    // ) build a MiniCam out of this keyframe
    // std::cout << "build minicam image size: " << image_width << "x" << image_height << std::endl;
    sv::MiniCam cam = viewpoint_cam->toMiniCam(image_height, image_width);
    // tr_cams.push_back(cam); // for densification later

    // std::cout << "ropts.track_max_w = " << ropts.track_max_w << std::endl;
    auto render_pkg = voxel_model_->render(
        cam,
        image_height,
        image_width,
        /* gt_image   */  gt_image,            
        /* color_mode   */   nullptr,             
        /* track_max_w   */  true,
        /* ss            */  std::nullopt,
        /* output_depth  */  ropts.output_depth,
        /* output_normal */  ropts.output_normal,
        /* output_T      */  ropts.output_T,
        /* rand_bg       */  false,
        /* use_auto_exp  */  false,
        ropts               // your struct (will be used for **other_opt-safe fields)
    );
    if (render_pkg.empty() || !render_pkg.count("color") || !render_pkg.at("color").defined()) {
        std::cout << "voxel_mapper render: pkg empty" << std::endl;
        return;
    }
    // // keep running max_w stats for pruning diagnostics (if returned)
    // if (render_pkg.count("max_w") && render_pkg.at("max_w").defined()) {
    //     voxel_model_->max_w_ = torch::maximum(
    //         voxel_model_->max_w_, render_pkg["max_w"].to(mDevice));
    // }

    torch::Tensor rendered_image = render_pkg["color"].to(mDevice);
    torch::Tensor masked_image = rendered_image * mask;      // (1,3,H,W)
    // torch::Tensor masked_gt   = gt_image * mask;
    if (!rendered_image.requires_grad()) {
        std::cerr << "[warn] rendered_image.requires_grad == false; grad_fn="
                << (rendered_image.grad_fn() ? "set" : "NULL") << "\n";
    }

    // after render_pkg & rendered_image
    torch::Tensor depth_for_viz;   // declare here so it's visible later
    auto it_depth = render_pkg.find("depth");
    if (it_depth != render_pkg.end() && it_depth->second.defined()) {
        depth_for_viz = it_depth->second;  // keep on device for now
    }

    auto Ll1 = loss_utils::l1_loss(masked_image, gt_image);
    auto mse = loss_utils::l2_loss(masked_image, gt_image);

    // Match SVRaster's base photometric loss selection: L1, Huber, or MSE.
    torch::Tensor photo_loss;
    if (opt_params_.use_l1_) {
        photo_loss = Ll1;
    } else if (opt_params_.use_huber_) {
        photo_loss = loss_utils::huber_loss(masked_image, gt_image, opt_params_.huber_thres_);
    } else {
        photo_loss = mse;
    }
    auto loss = photo_loss.clone();

    // optional use loss from the original photoslam paper
    float lambda_dssim = lambdaDssim();
    auto photoslam_loss = (1.0 - lambda_dssim) * Ll1
            + lambda_dssim * (1.0 - loss_utils::ssim(masked_image, gt_image, mDevice.type()));

    // Diagnostics for occupancy-related terms (printed periodically below)
    float dbg_sparse_depth_raw = 0.0f;
    float dbg_sparse_depth_w   = 0.0f;
    bool  dbg_sparse_depth_on  = false;
    float dbg_depthanything_raw = 0.0f;
    float dbg_depthanything_w   = 0.0f;
    bool  dbg_depthanything_on  = false;
    float dbg_rgbd_depth_raw = 0.0f;
    float dbg_rgbd_depth_w   = 0.0f;
    bool  dbg_rgbd_depth_on  = false;
    float dbg_depthanything_normal_raw = 0.0f;
    float dbg_depthanything_normal_w   = 0.0f;
    bool  dbg_depthanything_normal_on  = false;
    float dbg_rgbd_normal_raw = 0.0f;
    float dbg_rgbd_normal_w   = 0.0f;
    bool  dbg_rgbd_normal_on  = false;
    float dbg_t_concen_raw     = 0.0f;
    float dbg_t_concen_w       = 0.0f;
    bool  dbg_t_concen_on      = false;
    float dbg_t_inside_raw     = 0.0f;
    float dbg_t_inside_w       = 0.0f;
    bool  dbg_t_inside_on      = false;

    // --- Sparse depth regularization (SVRaster-style) ----------------------------
    if (need_sparse_depth) {
        torch::Tensor depth_loss =
            // computeSparseDepthLoss(viewpoint_cam, render_pkg, iter);
            computeSparseDepthLoss_Points(
            viewpoint_cam,   // which KF we are training on
            cam,             // MiniCam for this KF at current pyramid level
            image_width,
            image_height,
            render_pkg,
            iter);

        float dl = depth_loss.item<float>();
        dbg_sparse_depth_raw = dl;
        dbg_sparse_depth_w   = opt_params_.lambda_sparse_depth_ * dl;
        dbg_sparse_depth_on  = true;
        // if (dl > 0.0f || iter % 100 == 0) {
        //     std::cout << "[iter " << iter << "] sparse depth loss = " << dl << std::endl;
        // }
        loss = loss + opt_params_.lambda_sparse_depth_ * depth_loss;
    }
    if (need_depthanythingv2) {
        torch::Tensor dense_depth_loss = computeMonoPriorDepthLoss(
            viewpoint_cam,
            cam,
            render_pkg,
            iter);

        const float dl = dense_depth_loss.item<float>();
        dbg_depthanything_raw = dl;
        dbg_depthanything_w   = opt_params_.lambda_depthanythingv2_ * dl;
        dbg_depthanything_on  = true;
        loss = loss + opt_params_.lambda_depthanythingv2_ * dense_depth_loss;
    }
    if (need_rgbd_depth) {
        torch::Tensor rgbd_depth_loss =
            computeRgbdDepthLoss(viewpoint_cam, cam, render_pkg, iter);

        const float dl = rgbd_depth_loss.item<float>();
        dbg_rgbd_depth_raw = dl;
        dbg_rgbd_depth_w   = opt_params_.lambda_rgbd_depth_ * dl;
        dbg_rgbd_depth_on  = true;
        loss = loss + opt_params_.lambda_rgbd_depth_ * rgbd_depth_loss;
    }
    if (need_mono_prior_normal) {
        torch::Tensor dense_normal_loss = computeMonoPriorNormalLoss(
            viewpoint_cam,
            cam,
            render_pkg,
            iter);

        const float nl = dense_normal_loss.item<float>();
        dbg_depthanything_normal_raw = nl;
        dbg_depthanything_normal_w =
            opt_params_.lambda_depthanythingv2_normal_ * nl;
        dbg_depthanything_normal_on = true;
        loss = loss + opt_params_.lambda_depthanythingv2_normal_ * dense_normal_loss;
    }
    if (need_rgbd_normal) {
        torch::Tensor rgbd_normal_loss =
            computeRgbdNormalLoss(viewpoint_cam, cam, render_pkg, iter);

        const float nl = rgbd_normal_loss.item<float>();
        dbg_rgbd_normal_raw = nl;
        dbg_rgbd_normal_w = opt_params_.lambda_rgbd_normal_ * nl;
        dbg_rgbd_normal_on = true;
        loss = loss + opt_params_.lambda_rgbd_normal_ * rgbd_normal_loss;
    }
    if (opt_params_.lambda_ssim_ > 0.0f) {
        loss += opt_params_.lambda_ssim_ * loss_utils::fast_ssim_loss(masked_image, gt_image);
        // std::cout << "[iter " << iter << "] "
        //           << "MSE: " << mse.item<float>()
        //           << " SSIM_loss: " << (opt_params_.lambda_ssim_ * loss_utils::fast_ssim_loss(masked_image, gt_image)).item<float>()
        //           << " loss: " << loss.item<float>() << "\n";
    }

    if (need_T_concen || need_T_inside) {
        auto it = render_pkg.find("raw_T");
        if (it != render_pkg.end() && it->second.defined()) {
            torch::Tensor raw_T = it->second;

            // SVRaster: loss += lambda_T_concen * prob_concen_loss(raw_T)
            if (need_T_concen) {
                torch::Tensor reg_concen = loss_utils::prob_concen_loss(raw_T);
                dbg_t_concen_raw = reg_concen.item<float>();
                dbg_t_concen_w   = opt_params_.lambda_T_concen_ * dbg_t_concen_raw;
                dbg_t_concen_on  = true;
                loss = loss + opt_params_.lambda_T_concen_ * reg_concen;
            }

            // SVRaster: loss += lambda_T_inside * raw_T.square().mean()
            if (need_T_inside) {
                torch::Tensor reg_inside = raw_T.pow(2).mean();
                dbg_t_inside_raw = reg_inside.item<float>();
                dbg_t_inside_w   = opt_params_.lambda_T_inside_ * dbg_t_inside_raw;
                dbg_t_inside_on  = true;
                loss = loss + opt_params_.lambda_T_inside_ * reg_inside;
            }
        } else {
            std::cerr << "[warn] raw_T not in render_pkg (output_T might be off)\n";
        }
    }

    if (need_normal_dmean) {
        auto reg_normal_dmean = normalDepthConsistencyLossSVRaster(
            cam,
            render_pkg,
            opt_params_.n_dmean_ks_,
            opt_params_.n_dmean_tol_deg_);
        loss = loss + opt_params_.lambda_normal_dmean_ * reg_normal_dmean;
    }

    voxel_model_->optimizerZeroGrad();   // move this BEFORE backward
    {
        py::gil_scoped_release no_gil;
        loss.backward();
    }

    if (opt_params_.lambda_tv_density_ > 0.f &&
        iter >= opt_params_.tv_from_ &&
        iter <= opt_params_.tv_until_) {
        voxel_model_->applyTvOnDensityField(opt_params_.lambda_tv_density_);
    }

    voxel_model_->optimizerStep();   // <-- the actual update

    // --- debug: store near voxels for rerun ---
    torch::Tensor debug_near_centers;  // [K,3]
    torch::Tensor debug_near_sizes;    // [K,1] or [K]
    bool debug_has_near = false;
    torch::Tensor debug_near_geom_centers;  // [K_geom,3]
    torch::Tensor debug_near_geom_sizes;    // [K_geom,1] or [K_geom]
    bool debug_has_near_geom = false;
    torch::Tensor debug_tsdf_centers;   // [K_tsdf,3]
    torch::Tensor debug_tsdf_sizes;     // [K_tsdf,1] or [K_tsdf]
    bool debug_has_tsdf = false;
    torch::Tensor debug_tsdf_free_centers;
    torch::Tensor debug_tsdf_free_sizes;
    bool debug_has_tsdf_free = false;
    torch::Tensor debug_tsdf_occupied_centers;
    torch::Tensor debug_tsdf_occupied_sizes;
    bool debug_has_tsdf_occupied = false;
    torch::Tensor debug_tsdf_surface_centers;
    torch::Tensor debug_tsdf_surface_sizes;
    bool debug_has_tsdf_surface = false;
    torch::Tensor debug_tsdf_unknown_centers;
    torch::Tensor debug_tsdf_unknown_sizes;
    bool debug_has_tsdf_unknown = false;
    torch::Tensor debug_tsdf_free_corner_points;
    torch::Tensor debug_tsdf_occupied_corner_points;
    torch::Tensor debug_tsdf_surface_corner_points;
    torch::Tensor debug_tsdf_unknown_corner_points;
    torch::Tensor debug_tsdf_unknown_corner_colors;
    bool debug_has_tsdf_free_corners = false;
    bool debug_has_tsdf_occupied_corners = false;
    bool debug_has_tsdf_surface_corners = false;
    bool debug_has_tsdf_unknown_corners = false;
    torch::Tensor debug_pruned_centers; // [K_prune,3]
    torch::Tensor debug_pruned_sizes;   // [K_prune,1] or [K_prune]
    bool debug_has_pruned = false;
    torch::Tensor debug_far_pruned_centers; // [K_far,3]
    torch::Tensor debug_far_pruned_sizes;   // [K_far,1] or [K_far]
    bool debug_has_far_pruned = false;
    {
        // Densification for increasePcd
        const int prune_every =
            std::max(1, (opt_params_.prune_every_ > 0) ? opt_params_.prune_every_ : opt_params_.adapt_every_);
        const int subdivide_every =
            std::max(1, (opt_params_.subdivide_every_ > 0) ? opt_params_.subdivide_every_ : opt_params_.adapt_every_);
        const bool meet_prune_period =
            (iter >= opt_params_.adapt_from_) && (iter % prune_every == 0);
        const bool meet_subdivide_period =
            (iter >= opt_params_.adapt_from_) && (iter % subdivide_every == 0);

        bool need_pruning =
            meet_prune_period && (iter <= opt_params_.prune_until_);
        bool need_subdividing =
            meet_subdivide_period &&
            (iter <= opt_params_.subdivide_until_) &&
            (voxel_model_->numVoxels() < opt_params_.subdivide_max_num_);

        if (need_pruning || need_subdividing)
        {
            std::cout << "[DENSIFY/schedule] iter=" << iter
                      << " adapt_from=" << opt_params_.adapt_from_
                      << " prune_every=" << prune_every
                      << " subdivide_every=" << subdivide_every
                      << " need_prune=" << (need_pruning ? 1 : 0)
                      << " need_subdivide=" << (need_subdividing ? 1 : 0)
                      << "\n";

            // // NEW: cooldown after fill_empty_cells_ artificial creation
            // constexpr int64_t kMinItersAfterFill = 200;  // tune or move to YAML
            // if (last_artificial_fill_iter_ >= 0) {
            //     int64_t dt = static_cast<int64_t>(iter) - last_artificial_fill_iter_;
            //     if (dt < kMinItersAfterFill) {
            //         std::cout << "[VoxelMapper] skipping prune/subdiv at iter "
            //                 << iter << " (dt=" << dt
            //                 << " < " << kMinItersAfterFill
            //                 << " since last artificial fill)\n";
            //         // Skip densification for this iteration, but keep training, etc.
            //         return;   // exit trainForOneIteration() here
            //     }
            // }
            // // Build list of training cameras (use all current keyframes)
            std::vector<sv::MiniCam> tr_cams; 
            tr_cams.reserve(scene_->keyframes().size());
            std::cout << "keyframes size: " << scene_->keyframes().size() << std::endl;
            // std::cout << "image size: " << image_width << "x" << image_height << std::endl;
            for (auto& kv : scene_->keyframes()) {
                // std::cout << "keyframe id: " << kv.first << std::endl;
                // std::cout << "image_height_ : " << kv.second->image_height_ << std::endl;
                if (kv.second) {
                    tr_cams.push_back(
                        kv.second->toMiniCam(kv.second->image_height_, kv.second->image_width_));
                }
            }

            auto stat = voxel_model_->computeTrainingStat(tr_cams);
            py::object sched_state = voxel_model_->schedulerStateDict();
            auto flatten_colvec = [](torch::Tensor t) {
                if (t.defined() && t.dim() == 2 && t.size(1) == 1) {
                    t = t.squeeze(1);
                }
                return t.contiguous().view({-1});
            };

            std::cout << "[densify:stat] N=" << voxel_model_->numVoxels()
                    << "  max_w=" << shp(stat.max_w)
                    << "  min_itv=" << shp(stat.min_samp_interval)
                    << "  view_cnt=" << shp(stat.view_cnt) << "\n";

            // ---------------- PRUNE ----------------
            auto run_pruning = [&]() {
                {
                    const int N_cur = voxel_model_->numVoxels();
                    const bool stat_shape_ok =
                        stat.max_w.defined() &&
                        stat.min_samp_interval.defined() &&
                        stat.view_cnt.defined() &&
                        stat.max_w.size(0) == N_cur &&
                        stat.min_samp_interval.size(0) == N_cur &&
                        stat.view_cnt.size(0) == N_cur;
                    if (!stat_shape_ok) {
                        stat = voxel_model_->computeTrainingStat(tr_cams);
                    }
                }
                const float t1 = opt_params_.prune_thres_final_;
                const float ta1 = opt_params_.prune_thres_final_artificial_;
                const float prune_thres = t1; // fixed threshold
                // Separate threshold for at-target voxels.
                const float prune_thres_at_target = opt_params_.prune_thres_final_at_target_;
                const float prune_thres_artificial = ta1; // fixed threshold
                const int ori_n = voxel_model_->numVoxels();
                const int N     = ori_n;
                torch::Tensor prune_mask_vis; // [N] bool, set when visibility filter runs
                torch::Tensor prune_mask_near; // [N] bool, near-camera prune (unprotected)
                torch::Tensor prune_mask_recent_unstable; // [N] bool, young unstable voxels
                torch::Tensor prune_mask_default;         // [N] bool, default rules only
                torch::Tensor prune_mask_real_outside_dense_core; // [N] bool, real outliers outside dense-core
                torch::Tensor prune_mask_gslam_unstable;  // [N] bool, GS-SLAM unstable (real + artificial)
                torch::Tensor prune_mask_gslam_real;      // [N] bool, GS-SLAM unstable real only
                torch::Tensor prune_mask_gslam_artificial;// [N] bool, GS-SLAM unstable artificial only

                // --- 1) base Photo-SLAM / SVRaster pruning (max_w) ---
                auto max_w_1d = flatten_colvec(stat.max_w).to(torch::kFloat32); // [N]
                auto art_mask_for_base = voxel_model_->artificialMask();
                if (art_mask_for_base.defined()) {
                    art_mask_for_base = flatten_colvec(
                        art_mask_for_base.to(max_w_1d.device()).to(torch::kBool));
                }
                if (!art_mask_for_base.defined() || art_mask_for_base.numel() != N) {
                    art_mask_for_base = torch::zeros(
                        {N},
                        torch::TensorOptions().dtype(torch::kBool).device(max_w_1d.device()));
                }
                const int64_t n_artificial_total = art_mask_for_base.sum().item<int64_t>();
                auto rendered_depth_candidate_mask = voxel_model_->renderedDepthCandidateMask();
                if (rendered_depth_candidate_mask.defined()) {
                    rendered_depth_candidate_mask = flatten_colvec(
                        rendered_depth_candidate_mask.to(max_w_1d.device()).to(torch::kBool));
                }
                if (!rendered_depth_candidate_mask.defined() ||
                    rendered_depth_candidate_mask.numel() != N) {
                    rendered_depth_candidate_mask = torch::zeros(
                        {N},
                        torch::TensorOptions().dtype(torch::kBool).device(max_w_1d.device()));
                }
                auto rendered_depth_support_count = voxel_model_->renderedDepthCandidateSupportCount();
                if (rendered_depth_support_count.defined()) {
                    rendered_depth_support_count = flatten_colvec(
                        rendered_depth_support_count.to(max_w_1d.device()).to(torch::kInt32));
                }
                if (!rendered_depth_support_count.defined() ||
                    rendered_depth_support_count.numel() != N) {
                    rendered_depth_support_count = torch::zeros(
                        {N},
                        torch::TensorOptions().dtype(torch::kInt32).device(max_w_1d.device()));
                }
                auto rendered_depth_last_seen_kf = voxel_model_->renderedDepthCandidateLastSeenKf();
                if (rendered_depth_last_seen_kf.defined()) {
                    rendered_depth_last_seen_kf = flatten_colvec(
                        rendered_depth_last_seen_kf.to(max_w_1d.device()).to(torch::kInt32));
                }
                if (!rendered_depth_last_seen_kf.defined() ||
                    rendered_depth_last_seen_kf.numel() != N) {
                    rendered_depth_last_seen_kf = torch::full(
                        {N},
                        static_cast<int32_t>(-1),
                        torch::TensorOptions().dtype(torch::kInt32).device(max_w_1d.device()));
                }
                const int32_t rendered_depth_kf_now = static_cast<int32_t>(tr_cams.size());
                auto rendered_depth_under_support =
                    (rendered_depth_support_count <
                     opt_params_.rendered_depth_candidate_promote_min_support_).to(torch::kBool);
                auto rendered_depth_seen_valid =
                    (rendered_depth_last_seen_kf >= 0).to(torch::kBool);
                auto rendered_depth_age_kf =
                    (torch::full(
                        {N},
                        rendered_depth_kf_now,
                        torch::TensorOptions().dtype(torch::kInt32).device(max_w_1d.device())) -
                     rendered_depth_last_seen_kf).to(torch::kInt32);
                auto rendered_depth_candidate_young_protect =
                    (rendered_depth_candidate_mask &
                     rendered_depth_under_support &
                     rendered_depth_seen_valid &
                     (rendered_depth_age_kf < opt_params_.rendered_depth_candidate_prune_kf_age_))
                        .to(torch::kBool);
                auto rendered_depth_candidate_stale_prune =
                    (rendered_depth_candidate_mask &
                     rendered_depth_under_support &
                     rendered_depth_seen_valid &
                     (rendered_depth_age_kf >= opt_params_.rendered_depth_candidate_prune_kf_age_))
                        .to(torch::kBool);
                const int64_t n_rendered_depth_young_protect =
                    rendered_depth_candidate_young_protect.sum().item<int64_t>();
                const int64_t n_rendered_depth_stale_prune =
                    rendered_depth_candidate_stale_prune.sum().item<int64_t>();
                auto real_mask_for_base = (~art_mask_for_base).to(torch::kBool);
                auto in_target_size_mask = torch::zeros(
                    {N},
                    torch::TensorOptions().dtype(torch::kBool).device(max_w_1d.device()));
                auto vox_size_1d = voxel_model_->voxSize();
                if (vox_size_1d.defined()) {
                    if (vox_size_1d.dim() == 2 && vox_size_1d.size(1) == 1) {
                        vox_size_1d = vox_size_1d.squeeze(1);
                    } else if (vox_size_1d.dim() != 1) {
                        vox_size_1d = vox_size_1d.reshape({-1});
                    }
                    vox_size_1d = vox_size_1d.to(max_w_1d.device()).to(torch::kFloat32).contiguous();
                    if (vox_size_1d.numel() == N) {
                        in_target_size_mask =
                            (vox_size_1d <= opt_params_.subdivide_target_vox_size_).to(torch::kBool);
                    }
                }
                const int64_t n_in_target_size = in_target_size_mask.sum().item<int64_t>();
                const float prune_thres_real_in_target =
                    std::max(prune_thres, prune_thres_at_target);
                const float prune_thres_artificial_in_target =
                    std::max(prune_thres_artificial, prune_thres_at_target);
                auto max_w_real = max_w_1d.to(torch::kFloat32).contiguous();
                auto prune_thres_real_vec =
                    torch::full({N}, prune_thres,
                        torch::TensorOptions().dtype(torch::kFloat32).device(max_w_1d.device()));
                auto prune_thres_artificial_vec =
                    torch::full({N}, prune_thres_artificial,
                        torch::TensorOptions().dtype(torch::kFloat32).device(max_w_1d.device()));
                if (n_in_target_size > 0) {
                    prune_thres_real_vec = torch::where(
                        in_target_size_mask,
                        torch::full_like(prune_thres_real_vec, prune_thres_real_in_target),
                        prune_thres_real_vec);
                    prune_thres_artificial_vec = torch::where(
                        in_target_size_mask,
                        torch::full_like(prune_thres_artificial_vec, prune_thres_artificial_in_target),
                        prune_thres_artificial_vec);
                }
                auto prune_mask_base_real_raw =
                    ((max_w_1d < prune_thres_real_vec) & real_mask_for_base).to(torch::kBool);
                auto prune_mask_base_artificial_raw =
                    ((max_w_1d < prune_thres_artificial_vec) & art_mask_for_base).to(torch::kBool);
                prune_mask_base_artificial_raw =
                    (prune_mask_base_artificial_raw & (~rendered_depth_candidate_young_protect))
                        .to(torch::kBool);

                auto prune_mask_base_real = prune_mask_base_real_raw.to(torch::kBool);
                auto prune_mask_base_artificial = prune_mask_base_artificial_raw.to(torch::kBool);
                auto prune_mask_base_real_at_target =
                    (prune_mask_base_real & in_target_size_mask).to(torch::kBool);
                auto prune_mask_base_artificial_at_target =
                    (prune_mask_base_artificial & in_target_size_mask).to(torch::kBool);
                auto prune_mask_base_real_at_target_extra =
                    ((max_w_real >= prune_thres) &
                     (max_w_real < prune_thres_real_in_target) &
                     real_mask_for_base &
                     in_target_size_mask).to(torch::kBool);
                auto prune_mask_base_artificial_at_target_extra =
                    ((max_w_real >= prune_thres_artificial) &
                     (max_w_real < prune_thres_artificial_in_target) &
                     art_mask_for_base &
                     in_target_size_mask).to(torch::kBool);
                torch::Tensor prune_mask_base =
                    (prune_mask_base_real |
                     prune_mask_base_artificial |
                     rendered_depth_candidate_stale_prune).to(torch::kBool); // [N]
                auto prune_mask = prune_mask_base.clone(); // [N] bool
                const int n_prune_base =
                    prune_mask_base.defined()
                        ? (int)prune_mask_base.sum().item<int64_t>()
                        : -1;

                int n_prune_tsdf   = 0;
                int n_prune_union  = n_prune_base;
                int n_prune_tsdf_only = 0;
                int n_prune_overlap   = 0;
                // NEW: declare tsdf_prune_mask here, default undefined
                torch::Tensor tsdf_prune_mask;
                if (sensor_type_ == RGBD && use_tsdf_pruning_) {
                    if (N > 0 && sdf_mapper_ && sdf_mapper_->tsdf_layer().size() > 0) {
                        try {
                            torch::Tensor centers_world = voxel_model_->voxCenter(); // [N,3]
                            torch::Tensor sizes_world   = voxel_model_->voxSize();   // [N,1] (your implementation)

                            if (centers_world.defined() &&
                                centers_world.dim() == 2 &&
                                centers_world.size(0) == N &&
                                centers_world.size(1) == 3 &&
                                sizes_world.defined() &&
                                sizes_world.size(0) == N)
                            {
                                // Sample TSDF at 8 corners per voxel
                                // TsdfCornerSample c = sampleTsdfAtVoxelCornersWorld(centers_world, sizes_world);
                                TsdfCornerSample c = sampleTsdfAtSvrasterGridCornersWorld();
                                torch::Tensor tsdf8   = c.tsdf;    // [N,8]
                                torch::Tensor w8      = c.weight;  // [N,8]
                                torch::Tensor ok8     = c.success; // [N,8] bool

                                // Device alignment (should already match)
                                if (tsdf8.device() != prune_mask.device()) {
                                    tsdf8 = tsdf8.to(prune_mask.device());
                                    w8    = w8.to(prune_mask.device());
                                    ok8   = ok8.to(prune_mask.device());
                                }

                                // SVRecon-style SDF pruning:
                                // - gather 8 corner SDF values per voxel
                                // - keep zero-crossing voxels
                                // - keep voxels close to the zero surface
                                // - prune only sufficiently observed voxels far from the zero surface
                                const float min_weight = tsdf_prune_min_weight_;
                                const float tau_surface =
                                    std::max(0.0f, tsdf_prune_surface_band_vox_) *
                                    sdf_mapper_->tsdf_layer().voxel_size();
                                const float sign_eps = 1.0e-6f;

                                torch::Tensor corner_valid = (ok8 & (w8 >= min_weight)).to(torch::kBool); // [N,8]
                                torch::Tensor valid_count =
                                    corner_valid.to(torch::kInt32).sum(/*dim=*/1); // [N]
                                torch::Tensor voxel_valid =
                                    (valid_count >= tsdf_prune_min_valid_corners_).to(torch::kBool); // [N]

                                torch::Tensor has_pos =
                                    ((tsdf8 > sign_eps) & corner_valid).any(/*dim=*/1); // [N]
                                torch::Tensor has_neg =
                                    ((tsdf8 < -sign_eps) & corner_valid).any(/*dim=*/1); // [N]
                                torch::Tensor has_surface =
                                    (has_pos & has_neg).to(torch::kBool); // [N], zero crossing in valid corners

                                torch::Tensor abs_tsdf = tsdf8.abs();
                                torch::Tensor inf_like =
                                    torch::full_like(abs_tsdf, std::numeric_limits<float>::infinity());
                                torch::Tensor abs_valid =
                                    torch::where(corner_valid, abs_tsdf, inf_like);
                                torch::Tensor min_abs_tsdf =
                                    std::get<0>(abs_valid.min(/*dim=*/1)); // [N]
                                torch::Tensor far_from_surface =
                                    (min_abs_tsdf > tau_surface).to(torch::kBool); // [N]

                                torch::Tensor sizes_for_tsdf = sizes_world;
                                if (sizes_for_tsdf.dim() == 2 && sizes_for_tsdf.size(1) == 1) {
                                    sizes_for_tsdf = sizes_for_tsdf.squeeze(1);
                                } else if (sizes_for_tsdf.dim() != 1) {
                                    sizes_for_tsdf = sizes_for_tsdf.reshape({N});
                                }
                                sizes_for_tsdf = sizes_for_tsdf
                                    .to(prune_mask.device())
                                    .to(torch::kFloat32)
                                    .contiguous();
                                torch::Tensor half_voxel_diag =
                                    (0.5f * std::sqrt(3.0f)) * sizes_for_tsdf; // [N]
                                torch::Tensor conservative_free_thresh =
                                    torch::full_like(half_voxel_diag, tau_surface) + half_voxel_diag;
                                torch::Tensor strong_far_from_surface =
                                    (min_abs_tsdf > conservative_free_thresh).to(torch::kBool); // [N]

                                torch::Tensor all_positive =
                                    (voxel_valid & has_pos & (~has_neg)).to(torch::kBool);
                                torch::Tensor all_negative =
                                    (voxel_valid & has_neg & (~has_pos)).to(torch::kBool);
                                torch::Tensor tsdf_unknown_mask =
                                    (~voxel_valid).to(torch::kBool);
                                torch::Tensor tsdf_free_mask =
                                    (all_positive & strong_far_from_surface).to(torch::kBool);
                                torch::Tensor tsdf_occupied_mask =
                                    (all_negative & strong_far_from_surface).to(torch::kBool);
                                torch::Tensor tsdf_surface_mask =
                                    (voxel_valid & (has_surface | (~strong_far_from_surface)))
                                        .to(torch::kBool);

                                // Conservative external-TSDF pruning:
                                // prune only strong positive/free-space voxels. Negative-side voxels
                                // are kept for now because nvblox sign alone is not enough to prove
                                // they are useless scene geometry.
                                tsdf_prune_mask =
                                    tsdf_free_mask.to(torch::kBool); // [N]

                                // Ensure device matches prune_mask
                                if (tsdf_prune_mask.device() != prune_mask.device()) {
                                    tsdf_prune_mask = tsdf_prune_mask.to(prune_mask.device());
                                }
                                recordTsdfPruneAblation(
                                    tsdf_prune_mask,
                                    "regular_prune");

                                // Stats
                                n_prune_tsdf = (int)tsdf_prune_mask.sum().item<int64_t>();

                                // Additional diagnostics (optional, but useful early)
                                {
                                    int64_t n_valid = voxel_valid.sum().item<int64_t>();
                                    int64_t n_has_surface = has_surface.sum().item<int64_t>();
                                    int64_t n_far = far_from_surface.sum().item<int64_t>();
                                    int64_t n_strong_far =
                                        strong_far_from_surface.sum().item<int64_t>();
                                    int64_t n_min_valid =
                                        (valid_count >= tsdf_prune_min_valid_corners_).sum().item<int64_t>();
                                    int64_t n_free = tsdf_free_mask.sum().item<int64_t>();
                                    int64_t n_occupied = tsdf_occupied_mask.sum().item<int64_t>();
                                    int64_t n_surface_band = tsdf_surface_mask.sum().item<int64_t>();
                                    int64_t n_unknown = tsdf_unknown_mask.sum().item<int64_t>();
                                    float mean_half_diag =
                                        half_voxel_diag.numel() > 0
                                            ? half_voxel_diag.mean().item<float>()
                                            : 0.0f;
                                    std::cout << "[TSDF CORNER PRUNE] iter=" << iter
                                            << " N=" << N
                                            << " voxel_valid=" << n_valid
                                            << " min_valid_corners_vox=" << n_min_valid
                                            << " has_surface=" << n_has_surface
                                            << " far_from_surface=" << n_far
                                            << " strong_far_from_surface=" << n_strong_far
                                            << " free_space=" << n_free
                                            << " occupied_side=" << n_occupied
                                            << " surface_band=" << n_surface_band
                                            << " unknown=" << n_unknown
                                            << " prune_tsdf_sum=" << n_prune_tsdf
                                            << " min_weight=" << min_weight
                                            << " min_valid_corners=" << tsdf_prune_min_valid_corners_
                                            << " surface_band_m=" << tau_surface
                                            << " mean_half_voxel_diag_m=" << mean_half_diag
                                            << std::endl;
                                }

                                // Union + overlap statistics
                                auto prune_mask_union = prune_mask | tsdf_prune_mask;      // [N]
                                auto overlap_mask     = prune_mask_base & tsdf_prune_mask; // [N]

                                n_prune_union     = (int)prune_mask_union.sum().item<int64_t>();
                                n_prune_overlap   = (int)overlap_mask.sum().item<int64_t>();
                                n_prune_tsdf_only = n_prune_union - n_prune_base;
                                std::cout << "[TSDF CORNER PRUNE] union_stats"
                                          << " base=" << n_prune_base
                                          << " tsdf=" << n_prune_tsdf
                                          << " overlap=" << n_prune_overlap
                                          << " tsdf_only=" << n_prune_tsdf_only
                                          << " union=" << n_prune_union
                                          << std::endl;

                                // Save debug voxels pruned by TSDF for rerun visualization
                                auto tsdf_idx = tsdf_prune_mask.nonzero().squeeze(1); // [K]
                                if (tsdf_idx.numel() > 0) {
                                    debug_tsdf_centers = centers_world.index({tsdf_idx}).clone(); // [K,3]
                                    // sizes_world is [N,1] already
                                    debug_tsdf_sizes   = sizes_world.index({tsdf_idx}).clone();   // [K,1]
                                    debug_has_tsdf     = true;
                                    std::cout << "[DEBUG TSDF] saved " << tsdf_idx.size(0)
                                              << " SDF-pruned voxels for rerun visualization\n";
                                } else {
                                    debug_has_tsdf = false;
                                }

                                auto save_tsdf_class_debug =
                                    [&](const torch::Tensor& mask_in,
                                        torch::Tensor& centers_out,
                                        torch::Tensor& sizes_out,
                                        bool& has_out,
                                        const char* label)
                                {
                                    torch::Tensor mask = mask_in.to(torch::kBool);
                                    if (mask.device() != centers_world.device()) {
                                        mask = mask.to(centers_world.device());
                                    }

                                    auto idx = mask.nonzero().squeeze(1); // [K]
                                    if (idx.numel() > 0) {
                                        centers_out = centers_world.index_select(0, idx).clone(); // [K,3]
                                        sizes_out = sizes_world.index_select(0, idx).clone();     // [K,1]
                                        has_out = true;
                                        std::cout << "[DEBUG TSDF] saved " << idx.size(0)
                                                  << " " << label
                                                  << " voxels for rerun visualization\n";
                                    } else {
                                        centers_out = torch::Tensor();
                                        sizes_out = torch::Tensor();
                                        has_out = false;
                                    }
                                };

                                save_tsdf_class_debug(
                                    tsdf_free_mask,
                                    debug_tsdf_free_centers,
                                    debug_tsdf_free_sizes,
                                    debug_has_tsdf_free,
                                    "free-space");
                                save_tsdf_class_debug(
                                    tsdf_occupied_mask,
                                    debug_tsdf_occupied_centers,
                                    debug_tsdf_occupied_sizes,
                                    debug_has_tsdf_occupied,
                                    "occupied-side");
                                save_tsdf_class_debug(
                                    tsdf_surface_mask,
                                    debug_tsdf_surface_centers,
                                    debug_tsdf_surface_sizes,
                                    debug_has_tsdf_surface,
                                    "surface-band");
                                save_tsdf_class_debug(
                                    tsdf_unknown_mask,
                                    debug_tsdf_unknown_centers,
                                    debug_tsdf_unknown_sizes,
                                    debug_has_tsdf_unknown,
                                    "unknown");

                                auto save_tsdf_corner_debug =
                                    [&](const torch::Tensor& mask_in,
                                        torch::Tensor& points_out,
                                        torch::Tensor* colors_out,
                                        bool& has_out,
                                        float r,
                                        float g,
                                        float b,
                                        float a,
                                        const char* label,
                                        bool color_by_corner_valid)
                                {
                                    has_out = false;
                                    points_out = torch::Tensor();
                                    if (colors_out) {
                                        *colors_out = torch::Tensor();
                                    }
                                    if (!c.points_world.defined() ||
                                        c.points_world.dim() != 3 ||
                                        c.points_world.size(0) != N ||
                                        c.points_world.size(1) != 8 ||
                                        c.points_world.size(2) != 3) {
                                        return;
                                    }

                                    torch::Tensor mask = mask_in.to(torch::kBool);
                                    if (mask.device() != c.points_world.device()) {
                                        mask = mask.to(c.points_world.device());
                                    }
                                    torch::Tensor idx = mask.nonzero().squeeze(1);
                                    if (idx.numel() <= 0) {
                                        return;
                                    }

                                    constexpr int64_t kMaxTsdfSampleDebugVoxels = 50000;
                                    const int64_t original_voxels = idx.numel();
                                    if (original_voxels > kMaxTsdfSampleDebugVoxels) {
                                        idx = idx.slice(0, 0, kMaxTsdfSampleDebugVoxels);
                                    }

                                    points_out =
                                        c.points_world.index_select(0, idx)
                                            .reshape({-1, 3})
                                            .clone();
                                    has_out = points_out.defined() && points_out.numel() > 0;
                                    if (!has_out) {
                                        return;
                                    }

                                    if (colors_out) {
                                        torch::Tensor colors = torch::zeros(
                                            {points_out.size(0), 4},
                                            points_out.options());
                                        if (color_by_corner_valid) {
                                            torch::Tensor valid_corners =
                                                corner_valid
                                                    .to(c.points_world.device())
                                                    .index_select(0, idx)
                                                    .reshape({-1})
                                                    .to(torch::kBool);
                                            // Unknown voxels are colored per corner:
                                            // cyan = nvblox has a valid TSDF value there,
                                            // red = no valid TSDF sample at that corner.
                                            colors.index_put_({torch::indexing::Slice(), 0}, 1.0f);
                                            colors.index_put_({torch::indexing::Slice(), 3}, a);
                                            colors.index_put_({valid_corners, 0}, 0.0f);
                                            colors.index_put_({valid_corners, 1}, 0.9f);
                                            colors.index_put_({valid_corners, 2}, 1.0f);
                                        } else {
                                            colors.index_put_({torch::indexing::Slice(), 0}, r);
                                            colors.index_put_({torch::indexing::Slice(), 1}, g);
                                            colors.index_put_({torch::indexing::Slice(), 2}, b);
                                            colors.index_put_({torch::indexing::Slice(), 3}, a);
                                        }
                                        *colors_out = colors;
                                    }

                                    std::cout << "[DEBUG TSDF] saved "
                                              << points_out.size(0)
                                              << " " << label
                                              << " corner samples for rerun visualization";
                                    if (original_voxels > kMaxTsdfSampleDebugVoxels) {
                                        std::cout << " (capped from "
                                                  << original_voxels * 8
                                                  << " samples)";
                                    }
                                    std::cout << "\n";
                                };

                                save_tsdf_corner_debug(
                                    tsdf_free_mask,
                                    debug_tsdf_free_corner_points,
                                    nullptr,
                                    debug_has_tsdf_free_corners,
                                    0.0f, 0.9f, 1.0f, 0.95f,
                                    "free-space",
                                    false);
                                save_tsdf_corner_debug(
                                    tsdf_occupied_mask,
                                    debug_tsdf_occupied_corner_points,
                                    nullptr,
                                    debug_has_tsdf_occupied_corners,
                                    1.0f, 0.0f, 1.0f, 0.95f,
                                    "occupied-side",
                                    false);
                                save_tsdf_corner_debug(
                                    tsdf_surface_mask,
                                    debug_tsdf_surface_corner_points,
                                    nullptr,
                                    debug_has_tsdf_surface_corners,
                                    0.0f, 1.0f, 0.0f, 0.95f,
                                    "surface-band",
                                    false);
                                save_tsdf_corner_debug(
                                    tsdf_unknown_mask,
                                    debug_tsdf_unknown_corner_points,
                                    &debug_tsdf_unknown_corner_colors,
                                    debug_has_tsdf_unknown_corners,
                                    0.5f, 0.5f, 0.5f, 0.95f,
                                    "unknown",
                                    true);

                                // Use union as final prune mask
                                prune_mask = prune_mask_union;
                            } else {
                                std::cout << "[TSDF CORNER PRUNE] centers/sizes shape mismatch, skipping.\n";
                            }
                        } catch (const std::exception& e) {
                            std::cerr << "[TSDF CORNER PRUNE] exception: " << e.what() << "\n";
                        }
                    }
                }

                int64_t n_prune_near_front = 0;
                int64_t n_prune_near_geom = 0;

                // ---------------- NEW: SVRaster-like VISIBILITY / NEAR filtering ----------------
                // This mimics octlayout_filtering(...) using mark_max_samp_rate + mark_near,
                // but we apply it via pruning on the current model layout.
                if (!tr_cams.empty() && N > 0) {
                    try {
                        py::gil_scoped_acquire gil;
                        static py::module_ svr_mod =
                            py::module_::import("svraster_cuda").attr("renderer");
                        static py::module_ torch_mod =
                            py::module_::import("torch");

                        // Access Python SparseVoxelModel
                        py::object py_svm = voxel_model_->svm();
                        if (!py_svm.is_none()) {

                            py::object py_octpath   = py_svm.attr("octpath");
                            py::object py_octlv     = py_svm.attr("octlevel");
                            py::object py_vox_center= py_svm.attr("vox_center");
                            py::object py_vox_size  = py_svm.attr("vox_size");

                            at::Tensor octpath = py_octpath.cast<at::Tensor>().contiguous();     // [N,1] int64
                            at::Tensor L       = py_octlv.cast<at::Tensor>().contiguous();       // [N,1] int8 or int64
                            at::Tensor vox_center = py_vox_center.cast<at::Tensor>().contiguous(); // [N,3]
                            at::Tensor vox_size   = py_vox_size.cast<at::Tensor>().contiguous();   // [N,1] or [N]

                            // Basic sanity: same N
                            TORCH_CHECK(octpath.size(0) == N,
                                        "octpath.size(0) != N in pruning visibility filter");
                            TORCH_CHECK(L.size(0) == N,
                                        "octlevel.size(0) != N in pruning visibility filter");
                            TORCH_CHECK(vox_center.size(0) == N,
                                        "vox_center.size(0) != N in pruning visibility filter");
                            TORCH_CHECK(vox_center.size(1) == 3,
                                        "vox_center.size(1) must be 3");
                            if (vox_size.dim() == 1) {
                                vox_size = vox_size.view({N,1});
                            } else if (vox_size.dim() == 2) {
                                TORCH_CHECK(vox_size.size(0) == N,
                                            "vox_size.size(0) != N in pruning visibility filter");
                            } else {
                                TORCH_CHECK(false, "vox_size must be [N] or [N,1]");
                            }

                            // Build Python list of CUDA MiniCams
                            py::list py_cams;
                            py::object py_cuda = torch_mod.attr("device")("cuda");

                            auto move_attr_to_cuda_if_tensor =
                                [&](py::object& obj, const char* name){
                                    if (py::hasattr(obj, name)) {
                                        py::object t = obj.attr(name);
                                        if (py::hasattr(t, "is_cuda") &&
                                            !py::bool_(t.attr("is_cuda"))) {
                                            obj.attr(name) = t.attr("to")(py_cuda);
                                        }
                                    }
                                };

                            for (const auto& c : tr_cams) {
                                py::object py_cam = MiniCam_to_py(c);
                                move_attr_to_cuda_if_tensor(py_cam, "w2c");
                                move_attr_to_cuda_if_tensor(py_cam, "c2w");
                                move_attr_to_cuda_if_tensor(py_cam, "position");
                                move_attr_to_cuda_if_tensor(py_cam, "lookat");
                                py_cams.append(py_cam);
                            }

                            auto Nu_before = octpath.size(0);
                            TORCH_CHECK(Nu_before == N,
                                        "octpath.size(0) != N before visibility filter");

                            // 1) visibility: rate > 0
                            at::Tensor rate = svr_mod.attr("mark_max_samp_rate")(
                                py_cams,
                                py::cast(octpath),
                                py::cast(vox_center),
                                py::cast(vox_size)
                            ).cast<at::Tensor>();        // [N,1] or [N]

                            if (rate.dim() == 2 && rate.size(1) == 1)
                                rate = rate.squeeze(1);
                            rate = rate.to(torch::kFloat32);

                            at::Tensor keep_rate = (rate > 0.0f).to(torch::kBool);   // [N]
                            int64_t n_rate_pos = keep_rate.sum().item<int64_t>();

                            // 2) near filtering:
                            //    a) SVRaster mark_near (camera-facing)
                            //    b) geometric distance-to-camera test (front or behind)
                            const float near_thresh = 0.2f;
                            int64_t n_near_hit = 0;
                            int64_t n_near_geom_hit = 0;
                            at::Tensor is_near = torch::zeros(
                                {N},
                                torch::TensorOptions().dtype(torch::kBool).device(keep_rate.device()));
                            at::Tensor is_near_geom = torch::zeros(
                                {N},
                                torch::TensorOptions().dtype(torch::kBool).device(keep_rate.device()));
                            if (near_thresh > 0.0f) {
                                is_near = svr_mod.attr("mark_near")(
                                    py_cams,
                                    py::cast(octpath),
                                    py::cast(vox_center),
                                    py::cast(vox_size),
                                    py::float_(near_thresh)
                                ).cast<at::Tensor>();       // [N,1] or [N]
                                if (is_near.dim() == 2 && is_near.size(1) == 1)
                                    is_near = is_near.squeeze(1);
                                is_near = is_near.to(torch::kBool);
                                n_near_hit = is_near.sum().item<int64_t>();

                                if (opt_params_.prune_near_voxels_geometric_) {
                                    auto vox_center_f32 =
                                        vox_center.to(keep_rate.device()).to(torch::kFloat32).contiguous();
                                    auto vox_size_1d =
                                        vox_size.to(keep_rate.device()).to(torch::kFloat32).contiguous();
                                    if (vox_size_1d.dim() == 2 && vox_size_1d.size(1) == 1) {
                                        vox_size_1d = vox_size_1d.squeeze(1);
                                    } else if (vox_size_1d.dim() != 1) {
                                        vox_size_1d = vox_size_1d.reshape({-1});
                                    }
                                    TORCH_CHECK(vox_size_1d.numel() == N,
                                                "vox_size_1d.numel() != N in pruning geometric near filter");

                                    auto near_radius =
                                        (torch::full_like(vox_size_1d, near_thresh) + 0.5f * vox_size_1d)
                                            .contiguous();
                                    auto near_radius_sq = (near_radius * near_radius).contiguous();

                                    for (const auto& c : tr_cams) {
                                        auto cam_pos =
                                            c.position.to(keep_rate.device()).to(torch::kFloat32).view({1, 3});
                                        auto d2 = (vox_center_f32 - cam_pos).pow(2).sum(/*dim=*/1);
                                        is_near_geom =
                                            (is_near_geom | (d2 <= near_radius_sq)).to(torch::kBool);
                                    }
                                    n_near_geom_hit = is_near_geom.sum().item<int64_t>();
                                }
                            }

                            auto prune_near_union = (is_near | is_near_geom).to(torch::kBool);
                            n_prune_near_front = n_near_hit;
                            n_prune_near_geom = n_near_geom_hit;

                            auto near_geom_idx = is_near_geom.nonzero().squeeze(1);  // [K_geom]
                            if (near_geom_idx.numel() > 0) {
                                debug_near_geom_centers = vox_center.index({near_geom_idx}).clone(); // [K_geom,3]
                                debug_near_geom_sizes   = vox_size.index({near_geom_idx}).clone();   // [K_geom,1] or [K_geom]
                                debug_has_near_geom     = true;
                                std::cout << "[DEBUG NEAR GEOM] saved " << near_geom_idx.size(0)
                                          << " geometric-near voxels for rerun visualization\n";
                            } else {
                                debug_has_near_geom = false;
                            }

                            // --- DEBUG: save near voxels for rerun visualization ---
                            auto near_idx = prune_near_union.nonzero().squeeze(1);  // [K]
                            if (near_idx.numel() > 0) {
                                debug_near_centers = vox_center.index({near_idx}).clone();  // [K,3]
                                debug_near_sizes   = vox_size.index({near_idx}).clone();    // [K,1] or [K]
                                debug_has_near     = true;

                                std::cout << "[DEBUG NEAR] saved " << near_idx.size(0)
                                        << " near voxels for rerun visualization"
                                        << " (front=" << n_prune_near_front
                                        << ", geom=" << n_prune_near_geom << ")\n";
                            } else {
                                debug_has_near = false;
                            }

                            keep_rate = keep_rate.view({-1}).to(torch::kBool);    // [N]
                            prune_mask_vis = (~keep_rate);     // [N], visibility only
                            prune_mask_near = prune_near_union.view({-1}).to(torch::kBool); // [N], near only

                            // Combine with existing prune_mask
                            prune_mask = prune_mask | prune_mask_vis;

                        }

                    } catch (const std::exception& e) {
                        std::cerr << "[PRUNE/visibility] exception: " << e.what() << "\n";
                    }
                }

                // Keep a copy of default pruning terms before extra custom rules.
                prune_mask_default = prune_mask.to(torch::kBool);

                int64_t n_prune_base_real = prune_mask_base_real.defined()
                    ? prune_mask_base_real.sum().item<int64_t>() : 0;
                int64_t n_prune_base_artificial = prune_mask_base_artificial.defined()
                    ? prune_mask_base_artificial.sum().item<int64_t>() : 0;
                int64_t n_prune_base_real_cand = prune_mask_base_real_raw.defined()
                    ? prune_mask_base_real_raw.sum().item<int64_t>() : 0;
                int64_t n_prune_real_outside_dense_core = 0;
                int64_t n_prune_gslam_real = 0;
                int64_t n_prune_gslam_artificial = 0;
                int64_t n_prune_base_real_at_target = 0;
                int64_t n_prune_base_artificial_at_target = 0;
                int64_t n_prune_base_real_at_target_extra = 0;
                int64_t n_prune_base_artificial_at_target_extra = 0;
                int64_t n_prune_base_real_above_target = 0;
                int64_t n_prune_base_artificial_above_target = 0;
                int64_t n_prune_base_real_pre_gates = 0;
                int64_t n_prune_base_real_at_target_pre_gates = 0;
                int64_t n_prune_base_real_above_target_pre_gates = 0;
                int64_t n_real_at_target_total = 0;
                int64_t n_real_above_target_total = 0;
                int64_t n_artificial_at_target_total = 0;
                int64_t n_artificial_above_target_total = 0;
                int64_t n_real_recent_cooldown = 0;
                int64_t n_real_blocked_cooldown = 0;
                int64_t n_prune_blocked_kf_age = 0;
                int64_t n_prune_near = 0;
                const int64_t n_promoted_artificial_total = voxel_model_->totalPromotedartificialCount();

                if (N > 0) {
                    bool use_far_prune_this_round = opt_params_.prune_far_voxels_;
                    if (opt_params_.prune_far_voxels_) {
                        bool refreshed_dense_core = false;
                        if (opt_params_.prune_recompute_dense_core_) {
                            // std::cout << "[dense_core/refresh][prune] begin\n";
                            refreshed_dense_core =
                                voxel_model_->refreshDenseCoreBBFromCurrentVoxels();
                            // std::cout << "[dense_core/refresh][prune] done has_bb="
                            //           << (voxel_model_->hasDenseCoreBB() ? 1 : 0)
                            //           << " updated=" << (refreshed_dense_core ? 1 : 0) << "\n";
                            if (!voxel_model_->hasDenseCoreBB()) {
                                use_far_prune_this_round = false;
                                // std::cout << "[PRUNE/far] dense-core refresh unavailable; skipping real_far pruning this round.\n";
                            } else if (!refreshed_dense_core) {
                                // std::cout << "[PRUNE/far] dense-core refresh failed; using last available dense-core bbox.\n";
                            }
                        } else {
                            // std::cout << "[dense_core/refresh][prune] skipped (reuse cached bbox)\n";
                            if (!voxel_model_->hasDenseCoreBB()) {
                                use_far_prune_this_round = false;
                                // std::cout << "[PRUNE/far] cached dense-core bbox unavailable; skipping real_far pruning this round.\n";
                            } else {
                                // std::cout << "[PRUNE/far] using cached dense-core bbox (no recompute).\n";
                            }
                        }
                        if (enable_rerun_ && !rerun_final_only_ && voxel_model_->hasDenseCoreBB()) {
                            voxel_model_->logDenseCoreBBoxToRerun(
                                getIteration(),
                                "world/dense_core/used_for_prune");
                        }
                    }
                    prune_mask_real_outside_dense_core = torch::zeros(
                        {N}, torch::TensorOptions().dtype(torch::kBool).device(prune_mask.device()));
                    if (!prune_mask_near.defined() || prune_mask_near.numel() != N) {
                        prune_mask_near = torch::zeros(
                            {N}, torch::TensorOptions().dtype(torch::kBool).device(prune_mask.device()));
                    } else if (prune_mask_near.device() != prune_mask.device()) {
                        prune_mask_near = prune_mask_near.to(prune_mask.device()).to(torch::kBool).contiguous();
                    }
                    prune_mask_gslam_unstable = torch::zeros(
                        {N}, torch::TensorOptions().dtype(torch::kBool).device(prune_mask.device()));
                    prune_mask_gslam_real = torch::zeros(
                        {N}, torch::TensorOptions().dtype(torch::kBool).device(prune_mask.device()));
                    prune_mask_gslam_artificial = torch::zeros(
                        {N}, torch::TensorOptions().dtype(torch::kBool).device(prune_mask.device()));
                    auto geometrically_unstable_mask = torch::zeros(
                        {N}, torch::TensorOptions().dtype(torch::kBool).device(prune_mask.device()));

                    auto art_mask = art_mask_for_base.to(prune_mask.device()).to(torch::kBool).contiguous();
                    auto real_mask = real_mask_for_base.to(prune_mask.device()).to(torch::kBool).contiguous();
                    auto view_cnt = stat.view_cnt;
                    auto exist_since_kf = voxel_model_->existSinceKf();
                    if (view_cnt.defined() && exist_since_kf.defined()) {
                        view_cnt = flatten_colvec(view_cnt.to(prune_mask.device()).to(torch::kFloat32));
                        exist_since_kf = flatten_colvec(
                            exist_since_kf.to(prune_mask.device()).to(torch::kInt32));

                        if (view_cnt.numel() == N &&
                            exist_since_kf.numel() == N) {
                            if (use_far_prune_this_round && voxel_model_->hasDenseCoreBB()) {
                                auto centers = voxel_model_->voxCenter();
                                auto bb_min = voxel_model_->denseCoreBBMin();
                                auto bb_max = voxel_model_->denseCoreBBMax();
                                if (centers.defined() && bb_min.defined() && bb_max.defined() &&
                                    centers.dim() == 2 && centers.size(1) == 3 &&
                                    centers.size(0) == N &&
                                    bb_min.numel() == 3 && bb_max.numel() == 3) {
                                    centers = centers.to(prune_mask.device()).to(torch::kFloat32).contiguous();
                                    bb_min = bb_min.to(prune_mask.device()).to(torch::kFloat32).contiguous().view({1, 3});
                                    bb_max = bb_max.to(prune_mask.device()).to(torch::kFloat32).contiguous().view({1, 3});
                                    auto in_dense_core =
                                        (centers >= bb_min).all(/*dim=*/1) &
                                        (centers <= bb_max).all(/*dim=*/1);
                                    prune_mask_real_outside_dense_core =
                                        (real_mask & (~in_dense_core.to(torch::kBool))).to(torch::kBool);
                                }
                            }

                            if (opt_params_.prune_recent_unstable_) {
                                const int32_t current_kf_count = static_cast<int32_t>(tr_cams.size());
                                const int recent_kf_span = std::max(0, opt_params_.prune_recent_keyframes_);
                                auto born_valid_mask = (exist_since_kf >= 0).to(torch::kBool);
                                auto age_kf = (current_kf_count - exist_since_kf).to(torch::kInt32);
                                auto recent_mask =
                                    (born_valid_mask & (age_kf <= recent_kf_span)).to(torch::kBool);
                                auto other_view_cnt = torch::clamp_min(view_cnt - 1.0f, 0.0f);

                                auto prune_recent_real =
                                    recent_mask &
                                    real_mask &
                                    (other_view_cnt < static_cast<float>(opt_params_.prune_recent_min_views_real_));
                                auto prune_recent_artificial =
                                    recent_mask &
                                    art_mask &
                                    (other_view_cnt < static_cast<float>(opt_params_.prune_recent_min_views_artificial_));

                                prune_mask_gslam_real = prune_recent_real.to(torch::kBool);
                                prune_mask_gslam_artificial = prune_recent_artificial.to(torch::kBool);
                                prune_mask_gslam_unstable =
                                    (prune_mask_gslam_real | prune_mask_gslam_artificial).to(torch::kBool);
                                geometrically_unstable_mask = prune_mask_gslam_unstable.clone();
                            }
                        }
                    }

                    voxel_model_->setGeometricallyUnstableMask(geometrically_unstable_mask);
                    prune_mask_recent_unstable = prune_mask_gslam_unstable;
                    prune_mask = (prune_mask_default |
                                  prune_mask_near |
                                  prune_mask_real_outside_dense_core |
                                  prune_mask_gslam_unstable)
                        .to(torch::kBool);

                    n_prune_near = prune_mask_near.sum().item<int64_t>();
                    n_prune_real_outside_dense_core = prune_mask_real_outside_dense_core.sum().item<int64_t>();
                    n_prune_gslam_real = prune_mask_gslam_real.sum().item<int64_t>();
                    n_prune_gslam_artificial = prune_mask_gslam_artificial.sum().item<int64_t>();
                } else {
                    voxel_model_->setGeometricallyUnstableMask(torch::zeros(
                        {0},
                        torch::TensorOptions().dtype(torch::kBool).device(prune_mask.device())));
                }

                // Snapshot threshold-prune counts before KF-age / cooldown gates.
                n_prune_base_real_pre_gates = prune_mask_base_real.defined()
                    ? prune_mask_base_real.sum().item<int64_t>() : 0;
                n_prune_base_real_at_target_pre_gates =
                    (prune_mask_base_real.to(torch::kBool) &
                     in_target_size_mask.to(torch::kBool)).sum().item<int64_t>();
                n_prune_base_real_above_target_pre_gates =
                    (prune_mask_base_real.to(torch::kBool) &
                     (~in_target_size_mask.to(torch::kBool))).sum().item<int64_t>();

                // Keyframe-age gate: only prune voxels that are old enough in KF age.
                if (opt_params_.prune_min_kf_age_ > 0 && N > 0) {
                    auto exist_since_kf = voxel_model_->existSinceKf();
                    if (exist_since_kf.defined()) {
                        exist_since_kf = flatten_colvec(
                            exist_since_kf.to(prune_mask.device()).to(torch::kInt32));
                        if (exist_since_kf.numel() == N) {
                            auto kf_now = torch::full(
                                {N},
                                static_cast<int32_t>(tr_cams.size()),
                                torch::TensorOptions().dtype(torch::kInt32).device(prune_mask.device()));
                            auto born_valid_mask = (exist_since_kf >= 0).to(torch::kBool);
                            auto age_kf = (kf_now - exist_since_kf).to(torch::kInt32);
                            auto mature_kf_mask =
                                ((~born_valid_mask) |
                                 (age_kf >= opt_params_.prune_min_kf_age_)).to(torch::kBool);

                            auto prune_mask_before_kf_age = prune_mask.to(torch::kBool);
                            auto protected_mask = torch::ones(
                                {N},
                                torch::TensorOptions().dtype(torch::kBool).device(prune_mask.device()));
                            if (prune_mask_real_outside_dense_core.defined() &&
                                prune_mask_real_outside_dense_core.numel() == N) {
                                protected_mask =
                                    (~prune_mask_real_outside_dense_core.to(torch::kBool)).to(torch::kBool);
                            }
                            if (prune_mask_near.defined() && prune_mask_near.numel() == N) {
                                protected_mask =
                                    (protected_mask & (~prune_mask_near.to(torch::kBool))).to(torch::kBool);
                            }
                            if (prune_mask_gslam_unstable.defined() &&
                                prune_mask_gslam_unstable.numel() == N) {
                                protected_mask =
                                    (protected_mask & (~prune_mask_gslam_unstable.to(torch::kBool))).to(torch::kBool);
                            }
                            n_prune_blocked_kf_age =
                                (prune_mask_before_kf_age & protected_mask & (~mature_kf_mask))
                                    .sum().item<int64_t>();

                            prune_mask_base_real =
                                (prune_mask_base_real.to(torch::kBool) & mature_kf_mask).to(torch::kBool);
                            prune_mask_base_artificial =
                                (prune_mask_base_artificial.to(torch::kBool) & mature_kf_mask).to(torch::kBool);
                            prune_mask_base_real_at_target =
                                (prune_mask_base_real_at_target.to(torch::kBool) & mature_kf_mask).to(torch::kBool);
                            prune_mask_base_artificial_at_target =
                                (prune_mask_base_artificial_at_target.to(torch::kBool) & mature_kf_mask).to(torch::kBool);
                            prune_mask_base_real_at_target_extra =
                                (prune_mask_base_real_at_target_extra.to(torch::kBool) & mature_kf_mask).to(torch::kBool);
                            prune_mask_base_artificial_at_target_extra =
                                (prune_mask_base_artificial_at_target_extra.to(torch::kBool) & mature_kf_mask).to(torch::kBool);
                            prune_mask_default =
                                (prune_mask_default.to(torch::kBool) & mature_kf_mask).to(torch::kBool);
                            prune_mask = (prune_mask_default.to(torch::kBool) |
                                          prune_mask_near.to(torch::kBool) |
                                          prune_mask_real_outside_dense_core.to(torch::kBool) |
                                          prune_mask_gslam_unstable.to(torch::kBool)).to(torch::kBool);
                        }
                    }
                }

                // Cooldown: do not prune voxels that were created too recently.
                if (opt_params_.densify_cooldown_iters_ > 0 && N > 0) {
                    auto exist_since_iter = voxel_model_->existSinceIter();
                    if (exist_since_iter.defined()) {
                        exist_since_iter = flatten_colvec(
                            exist_since_iter.to(prune_mask.device()).to(torch::kInt32));
                        if (exist_since_iter.numel() == N) {
                            auto iter_now = torch::full(
                                {N},
                                static_cast<int>(iter),
                                torch::TensorOptions().dtype(torch::kInt32).device(prune_mask.device()));
                            auto age = iter_now - exist_since_iter; // [N], in iterations
                            auto mature_mask = (age >= opt_params_.densify_cooldown_iters_).to(torch::kBool);
                            auto recent_mask = (~mature_mask).to(torch::kBool);
                            n_real_recent_cooldown =
                                (real_mask_for_base.to(prune_mask.device()).to(torch::kBool) & recent_mask)
                                    .sum().item<int64_t>();
                            auto prune_mask_before_cooldown = prune_mask.to(torch::kBool);
                            auto protected_mask = torch::ones(
                                {N},
                                torch::TensorOptions().dtype(torch::kBool).device(prune_mask.device()));
                            if (prune_mask_real_outside_dense_core.defined() &&
                                prune_mask_real_outside_dense_core.numel() == N) {
                                protected_mask =
                                    (~prune_mask_real_outside_dense_core.to(torch::kBool)).to(torch::kBool);
                            }
                            if (prune_mask_near.defined() && prune_mask_near.numel() == N) {
                                protected_mask =
                                    (protected_mask & (~prune_mask_near.to(torch::kBool))).to(torch::kBool);
                            }
                            if (prune_mask_gslam_unstable.defined() &&
                                prune_mask_gslam_unstable.numel() == N) {
                                protected_mask =
                                    (protected_mask & (~prune_mask_gslam_unstable.to(torch::kBool))).to(torch::kBool);
                            }
                            n_real_blocked_cooldown =
                                (prune_mask_before_cooldown &
                                 real_mask_for_base.to(prune_mask.device()).to(torch::kBool) &
                                 recent_mask &
                                 protected_mask).sum().item<int64_t>();
                            prune_mask_base_real =
                                (prune_mask_base_real.to(torch::kBool) & mature_mask).to(torch::kBool);
                            prune_mask_base_artificial =
                                (prune_mask_base_artificial.to(torch::kBool) & mature_mask).to(torch::kBool);
                            prune_mask_base_real_at_target =
                                (prune_mask_base_real_at_target.to(torch::kBool) & mature_mask).to(torch::kBool);
                            prune_mask_base_artificial_at_target =
                                (prune_mask_base_artificial_at_target.to(torch::kBool) & mature_mask).to(torch::kBool);
                            prune_mask_base_real_at_target_extra =
                                (prune_mask_base_real_at_target_extra.to(torch::kBool) & mature_mask).to(torch::kBool);
                            prune_mask_base_artificial_at_target_extra =
                                (prune_mask_base_artificial_at_target_extra.to(torch::kBool) & mature_mask).to(torch::kBool);
                            prune_mask_default =
                                (prune_mask_default.to(torch::kBool) & mature_mask).to(torch::kBool);
                            prune_mask = (prune_mask_default.to(torch::kBool) |
                                          prune_mask_near.to(torch::kBool) |
                                          prune_mask_real_outside_dense_core.to(torch::kBool) |
                                          prune_mask_gslam_unstable.to(torch::kBool)).to(torch::kBool);
                        }
                    }
                }

                n_prune_base_real = prune_mask_base_real.defined()
                    ? prune_mask_base_real.sum().item<int64_t>() : 0;
                n_prune_base_artificial = prune_mask_base_artificial.defined()
                    ? prune_mask_base_artificial.sum().item<int64_t>() : 0;
                n_prune_base_real_at_target = prune_mask_base_real_at_target.defined()
                    ? prune_mask_base_real_at_target.sum().item<int64_t>() : 0;
                n_prune_base_artificial_at_target = prune_mask_base_artificial_at_target.defined()
                    ? prune_mask_base_artificial_at_target.sum().item<int64_t>() : 0;
                n_prune_base_real_at_target_extra = prune_mask_base_real_at_target_extra.defined()
                    ? prune_mask_base_real_at_target_extra.sum().item<int64_t>() : 0;
                n_prune_base_artificial_at_target_extra = prune_mask_base_artificial_at_target_extra.defined()
                    ? prune_mask_base_artificial_at_target_extra.sum().item<int64_t>() : 0;
                auto in_target_size_mask_final = in_target_size_mask.to(torch::kBool).contiguous();
                auto above_target_size_mask_final = (~in_target_size_mask_final).to(torch::kBool);
                n_prune_base_real_above_target =
                    (prune_mask_base_real.to(torch::kBool) & above_target_size_mask_final).sum().item<int64_t>();
                n_prune_base_artificial_above_target =
                    (prune_mask_base_artificial.to(torch::kBool) & above_target_size_mask_final).sum().item<int64_t>();
                n_real_at_target_total =
                    (real_mask_for_base.to(torch::kBool) & in_target_size_mask_final).sum().item<int64_t>();
                n_real_above_target_total =
                    (real_mask_for_base.to(torch::kBool) & above_target_size_mask_final).sum().item<int64_t>();
                n_artificial_at_target_total =
                    (art_mask_for_base.to(torch::kBool) & in_target_size_mask_final).sum().item<int64_t>();
                n_artificial_above_target_total =
                    (art_mask_for_base.to(torch::kBool) & above_target_size_mask_final).sum().item<int64_t>();
                n_prune_real_outside_dense_core = prune_mask_real_outside_dense_core.defined()
                    ? prune_mask_real_outside_dense_core.sum().item<int64_t>() : 0;
                n_prune_gslam_real = prune_mask_gslam_real.defined()
                    ? prune_mask_gslam_real.sum().item<int64_t>() : 0;
                n_prune_gslam_artificial = prune_mask_gslam_artificial.defined()
                    ? prune_mask_gslam_artificial.sum().item<int64_t>() : 0;
                const int64_t n_prune_total_selected = prune_mask.defined()
                    ? prune_mask.to(torch::kBool).sum().item<int64_t>() : 0;
                std::cout << "[PRUNE/real] thres_now=" << prune_thres
                          << " thres_at_target=" << prune_thres_real_in_target
                          << " vox_at_target=" << n_real_at_target_total
                          << " vox_above_target=" << n_real_above_target_total
                          << " by_thres_cand=" << n_prune_base_real_cand
                          << " by_thres_pre_gates=" << n_prune_base_real_pre_gates
                          << " by_thres_at_target_pre_gates=" << n_prune_base_real_at_target_pre_gates
                          << " by_thres_above_target_pre_gates=" << n_prune_base_real_above_target_pre_gates
                          << " by_thres_total=" << n_prune_base_real
                          << " by_thres_at_target=" << n_prune_base_real_at_target
                          << " by_thres_above_target=" << n_prune_base_real_above_target
                          << " extra_at_target=" << n_prune_base_real_at_target_extra
                          << " gslam=" << n_prune_gslam_real
                          << " near=" << n_prune_near
                          << " near_front=" << n_prune_near_front
                          << " near_geom=" << n_prune_near_geom
                          << " real_far=" << n_prune_real_outside_dense_core
                          << " kf_age_blocked=" << n_prune_blocked_kf_age
                          << " cooldown_recent=" << n_real_recent_cooldown
                          << "\n";
                std::cout << "[PRUNE/artificial] thres_now=" << prune_thres_artificial
                          << " thres_at_target=" << prune_thres_artificial_in_target
                          << " vox_at_target=" << n_artificial_at_target_total
                          << " vox_above_target=" << n_artificial_above_target_total
                          << " by_thres_total=" << n_prune_base_artificial << "/" << n_artificial_total
                          << " by_thres_at_target=" << n_prune_base_artificial_at_target
                          << " by_thres_above_target=" << n_prune_base_artificial_above_target
                          << " extra_at_target=" << n_prune_base_artificial_at_target_extra
                          << " rendered_depth_young_protected=" << n_rendered_depth_young_protect
                          << " rendered_depth_stale=" << n_rendered_depth_stale_prune
                          << " gslam=" << n_prune_gslam_artificial
                          << " promoted_total=" << n_promoted_artificial_total
                          << "\n";

                // Save final pruned voxels (all criteria merged) for rerun visualization.
                if (prune_mask.defined() && prune_mask.numel() == N) {
                    auto prune_idx = prune_mask.to(torch::kBool).nonzero().squeeze(1); // [K]
                    if (prune_idx.numel() > 0) {
                        auto centers_world = voxel_model_->voxCenter(); // [N,3]
                        auto sizes_world   = voxel_model_->voxSize();   // [N] or [N,1]
                        if (centers_world.defined() &&
                            centers_world.dim() == 2 &&
                            centers_world.size(0) == N &&
                            centers_world.size(1) == 3 &&
                            sizes_world.defined() &&
                            sizes_world.size(0) == N)
                        {
                            debug_pruned_centers = centers_world.index({prune_idx}).clone();
                            debug_pruned_sizes   = sizes_world.index({prune_idx}).clone();
                            debug_has_pruned     = true;

                            // Save far-only pruned voxels for a dedicated rerun topic.
                            debug_has_far_pruned = false;
                            if (prune_mask_real_outside_dense_core.defined() &&
                                prune_mask_real_outside_dense_core.numel() == N) {
                                auto far_pruned_mask =
                                    (prune_mask_real_outside_dense_core.to(torch::kBool) &
                                     prune_mask.to(torch::kBool)).to(torch::kBool);
                                auto far_idx = far_pruned_mask.nonzero().squeeze(1);
                                if (far_idx.numel() > 0) {
                                    debug_far_pruned_centers = centers_world.index({far_idx}).clone();
                                    debug_far_pruned_sizes   = sizes_world.index({far_idx}).clone();
                                    debug_has_far_pruned     = true;
                                    std::cout << "[DEBUG FAR] saved " << far_idx.size(0)
                                              << " far-pruned voxels for rerun visualization\n";
                                }
                            }
                        } else {
                            debug_has_pruned = false;
                            debug_has_far_pruned = false;
                        }
                    } else {
                        debug_has_pruned = false;
                        debug_has_far_pruned = false;
                    }
                } else {
                    debug_has_pruned = false;
                    debug_has_far_pruned = false;
                }

                voxel_model_->pruning(prune_mask);
                const int new_n = voxel_model_->numVoxels();
                std::cout << "[PRUNE/TOTAL] " << std::setw(7) << ori_n
                          << " => "          << std::setw(7) << new_n
                          << " (x" << std::fixed << std::setprecision(2)
                          << (double)new_n / std::max(1, ori_n) << ")"
                          << " selected=" << n_prune_total_selected
                          << " removed=" << (ori_n - new_n) << "\n";

                // If pruning changed the voxel set (or shapes don’t match), recompute stats
                const int M = voxel_model_->numVoxels();
                const bool shape_ok =
                    stat.min_samp_interval.defined() &&
                    stat.min_samp_interval.dim() == 2 &&
                    stat.min_samp_interval.size(0) == M &&
                    stat.min_samp_interval.size(1) == 1;
                if (new_n != ori_n || !shape_ok) {
                    stat = voxel_model_->computeTrainingStat(tr_cams);
                }
            };

            // ---------------- SUBDIVIDE ----------------
            if (need_subdividing) {
                voxel_model_->setTopologyBirthContext(iter, static_cast<int>(tr_cams.size()));
                const int before = voxel_model_->numVoxels();
                if (before == 0) {
                    std::cout << "[SUBDIV:skip] M==0\n";
                } else {
                    auto vox_size_before = voxel_model_->voxSize(); // [N] or [N,1]
                    if (vox_size_before.dim() == 2 && vox_size_before.size(1) == 1) {
                        vox_size_before = vox_size_before.squeeze(1);
                    } else if (vox_size_before.dim() != 1) {
                        vox_size_before = vox_size_before.reshape({-1});
                    }
                    const double vox_size_min_before =
                        (vox_size_before.numel() > 0) ? vox_size_before.min().item<double>() : 0.0;
                    const double vox_size_max_before =
                        (vox_size_before.numel() > 0) ? vox_size_before.max().item<double>() : 0.0;

                    bool did_subdivide = false;
                    int aggressive_rounds = 0;
                    int64_t n_subdiv_aggressive_total = 0;
                    int64_t n_normal_candidates = 0;
                    int64_t n_subdiv_normal_selected = 0;
                    int64_t n_rendered_depth_blocked_aggressive_total = 0;
                    int64_t n_rendered_depth_blocked_normal = 0;
                    int64_t n_artificial_blocked_aggressive_total = 0;
                    int64_t n_artificial_blocked_normal = 0;

                    auto rendered_depth_candidate_mask_for = [&](int64_t expected_n,
                                                                 const torch::Device& dev) {
                        auto m = voxel_model_->renderedDepthCandidateMask();
                        if (!m.defined() || m.numel() == 0) {
                            return torch::zeros(
                                {expected_n},
                                torch::TensorOptions().dtype(torch::kBool).device(dev));
                        }
                        if (m.dim() == 2 && m.size(1) == 1) {
                            m = m.squeeze(1);
                        }
                        m = m.to(dev).to(torch::kBool).contiguous().view({-1});
                        if (m.numel() != expected_n) {
                            return torch::zeros(
                                {expected_n},
                                torch::TensorOptions().dtype(torch::kBool).device(dev));
                        }
                        return m;
                    };
                    auto artificial_subdiv_block_mask_for = [&](int64_t expected_n,
                                                                const torch::Device& dev) {
                        if (!opt_params_.subdivide_artificial_requires_promotion_) {
                            return torch::zeros(
                                {expected_n},
                                torch::TensorOptions().dtype(torch::kBool).device(dev));
                        }
                        auto m = voxel_model_->artificialMask();
                        if (!m.defined() || m.numel() == 0) {
                            return torch::zeros(
                                {expected_n},
                                torch::TensorOptions().dtype(torch::kBool).device(dev));
                        }
                        if (m.dim() == 2 && m.size(1) == 1) {
                            m = m.squeeze(1);
                        }
                        m = m.to(dev).to(torch::kBool).contiguous().view({-1});
                        if (m.numel() != expected_n) {
                            return torch::zeros(
                                {expected_n},
                                torch::TensorOptions().dtype(torch::kBool).device(dev));
                        }
                        return m;
                    };
                    auto source_counts_for_current_voxels =
                        [&]() -> std::array<int64_t, 3>
                    {
                        const int64_t n = voxel_model_->numVoxels();
                        if (n <= 0) {
                            return {0, 0, 0};
                        }
                        const torch::Device dev = voxel_model_->voxCenter().device();
                        torch::Tensor rgbd_points_mask =
                            normalizeBoolMaskOrZeros(voxel_model_->orbVoxelMask(), n, dev);
                        torch::Tensor inactive_geo_mask =
                            normalizeBoolMaskOrZeros(voxel_model_->inactiveGeoVoxelMask(), n, dev);
                        torch::Tensor rgbd_fill_mask =
                            normalizeBoolMaskOrZeros(voxel_model_->rgbdFillRenderHolesVoxelMask(), n, dev);
                        rgbd_points_mask =
                            (rgbd_points_mask & (~inactive_geo_mask) & (~rgbd_fill_mask))
                                .to(torch::kBool);
                        return {
                            rgbd_points_mask.sum().item<int64_t>(),
                            inactive_geo_mask.sum().item<int64_t>(),
                            rgbd_fill_mask.sum().item<int64_t>()};
                    };
                    auto selected_source_counts =
                        [&](const torch::Tensor& selected_mask) -> std::array<int64_t, 3>
                    {
                        if (!selected_mask.defined() || selected_mask.numel() <= 0) {
                            return {0, 0, 0};
                        }
                        const int64_t n = selected_mask.numel();
                        const torch::Device dev = selected_mask.device();
                        torch::Tensor selected =
                            normalizeBoolMaskOrZeros(selected_mask, n, dev);
                        torch::Tensor rgbd_points_mask =
                            normalizeBoolMaskOrZeros(voxel_model_->orbVoxelMask(), n, dev);
                        torch::Tensor inactive_geo_mask =
                            normalizeBoolMaskOrZeros(voxel_model_->inactiveGeoVoxelMask(), n, dev);
                        torch::Tensor rgbd_fill_mask =
                            normalizeBoolMaskOrZeros(voxel_model_->rgbdFillRenderHolesVoxelMask(), n, dev);
                        rgbd_points_mask =
                            (rgbd_points_mask & (~inactive_geo_mask) & (~rgbd_fill_mask))
                                .to(torch::kBool);
                        return {
                            (selected & rgbd_points_mask).sum().item<int64_t>(),
                            (selected & inactive_geo_mask).sum().item<int64_t>(),
                            (selected & rgbd_fill_mask).sum().item<int64_t>()};
                    };
                    auto account_subdivision_lineage_births =
                        [&](const std::array<int64_t, 3>& before_counts,
                            const std::array<int64_t, 3>& parent_counts,
                            const char* tag)
                    {
                        const std::array<int64_t, 3> after_counts =
                            source_counts_for_current_voxels();
                        const int64_t rgbd_children =
                            std::max<int64_t>(0, after_counts[0] - before_counts[0] + parent_counts[0]);
                        const int64_t inactive_children =
                            std::max<int64_t>(0, after_counts[1] - before_counts[1] + parent_counts[1]);
                        const int64_t rgbd_fill_children =
                            std::max<int64_t>(0, after_counts[2] - before_counts[2] + parent_counts[2]);
                        tsdf_ablation_rgbd_points_lineage_created_ += rgbd_children;
                        tsdf_ablation_inactive_geo_lineage_created_ += inactive_children;
                        tsdf_ablation_rgbd_fill_lineage_created_ += rgbd_fill_children;
                        if (rgbd_children > 0 || inactive_children > 0 || rgbd_fill_children > 0) {
                            std::cout << "[TSDF ABLATION/subdiv] iter=" << iter
                                      << " tag=" << tag
                                      << " child_voxels rgbd_points=" << rgbd_children
                                      << " inactive_geo=" << inactive_children
                                      << " rgbd_fill_render_holes=" << rgbd_fill_children
                                      << "\n";
                        }
                    };

                    auto compute_max_n_subdiv = [&]() -> int {
                        return std::round(
                            (opt_params_.subdivide_max_num_ - voxel_model_->numVoxels()) / 7.0);
                    };

                    // Stage 1: if force-to-target is enabled, repeatedly subdivide
                    // above-target voxels until none remain (or cap / finest-level stops us).
                    if (opt_params_.subdivide_force_to_target_size_) {
                        while (true) {
                            const int Ma = voxel_model_->numVoxels();
                            if (Ma <= 0) break;

                            auto vox_size_a = voxel_model_->voxSize(); // [Ma] or [Ma,1]
                            if (vox_size_a.dim() == 2 && vox_size_a.size(1) == 1) {
                                vox_size_a = vox_size_a.squeeze(1);
                            } else if (vox_size_a.dim() != 1) {
                                vox_size_a = vox_size_a.reshape({-1});
                            }

                            auto octlv_a = voxel_model_->octLevel(); // [Ma] or [Ma,1]
                            if (octlv_a.defined() && octlv_a.dim() == 2 && octlv_a.size(1) == 1) {
                                octlv_a = octlv_a.squeeze(1);
                            }
                            auto non_finest_a =
                                (octlv_a.to(torch::kInt32) < voxel_model_->maxNumLevels()); // [Ma] bool

                            auto rendered_depth_candidate_mask_a =
                                rendered_depth_candidate_mask_for(Ma, vox_size_a.device());
                            auto artificial_subdiv_block_mask_a =
                                artificial_subdiv_block_mask_for(Ma, vox_size_a.device());
                            const int64_t n_rendered_depth_blocked_aggressive =
                                rendered_depth_candidate_mask_a.sum().item<int64_t>();
                            const int64_t n_artificial_blocked_aggressive =
                                artificial_subdiv_block_mask_a.sum().item<int64_t>();
                            auto aggressive_candidate_mask =
                                ((vox_size_a > opt_params_.subdivide_target_vox_size_) &
                                 non_finest_a &
                                 (~rendered_depth_candidate_mask_a) &
                                 (~artificial_subdiv_block_mask_a))
                                    .to(torch::kBool);
                            const int64_t n_aggressive_candidates =
                                aggressive_candidate_mask.sum().item<int64_t>();
                            if (n_aggressive_candidates <= 0) {
                                break;
                            }

                            int max_n_subdiv = compute_max_n_subdiv();
                            if (max_n_subdiv <= 0) {
                                std::cout << "[SUBDIV:skip] cap reached during aggressive stage (max_n_subdiv<=0)\n";
                                break;
                            }

                            auto aggressive_selected_mask = aggressive_candidate_mask.clone();
                            if (n_aggressive_candidates > max_n_subdiv) {
                                // If capped, keep largest voxels first.
                                auto pos_idx = aggressive_candidate_mask.nonzero().squeeze(1); // [K]
                                auto pos_sizes = vox_size_a.index({pos_idx});                  // [K]
                                auto sort_pair = pos_sizes.sort(/*dim=*/0, /*descending=*/true);
                                auto order_desc = std::get<1>(sort_pair).to(torch::kLong).contiguous();
                                auto keep_local = order_desc.index(
                                    {torch::indexing::Slice(0, max_n_subdiv)}).contiguous();
                                auto keep_idx = pos_idx.index_select(0, keep_local).contiguous();
                                aggressive_selected_mask =
                                    torch::zeros_like(aggressive_candidate_mask, torch::kBool);
                                aggressive_selected_mask.index_put_({keep_idx}, true);
                            }

                            const int64_t n_aggressive_selected =
                                aggressive_selected_mask.sum().item<int64_t>();
                            if (n_aggressive_selected <= 0) {
                                break;
                            }

                            ++aggressive_rounds;
                            n_subdiv_aggressive_total += n_aggressive_selected;
                            n_rendered_depth_blocked_aggressive_total +=
                                n_rendered_depth_blocked_aggressive;
                            n_artificial_blocked_aggressive_total +=
                                n_artificial_blocked_aggressive;
                            std::cout << "[SUBDIV/aggressive] round=" << aggressive_rounds
                                      << " candidates=" << n_aggressive_candidates
                                      << " rendered_depth_blocked="
                                      << n_rendered_depth_blocked_aggressive
                                      << " artificial_blocked="
                                      << n_artificial_blocked_aggressive
                                      << " selected=" << n_aggressive_selected << "\n";

                            const std::array<int64_t, 3> source_counts_before_subdiv =
                                source_counts_for_current_voxels();
                            const std::array<int64_t, 3> selected_parent_counts =
                                selected_source_counts(aggressive_selected_mask);
                            voxel_model_->subdividing(aggressive_selected_mask);
                            account_subdivision_lineage_births(
                                source_counts_before_subdiv,
                                selected_parent_counts,
                                "aggressive");
                            did_subdivide = true;
                        }
                    }

                    // Stage 2: one SVRaster subdivision pass for remaining voxels,
                    // after the above-target voxels have been forced down to target.
                    const int M = voxel_model_->numVoxels();
                    if (M > 0) {
                        auto min_samp_interval = stat.min_samp_interval; // [M,1]
                        if (did_subdivide ||
                            !min_samp_interval.defined() ||
                            min_samp_interval.size(0) != M) {
                            stat = voxel_model_->computeTrainingStat(tr_cams);
                            min_samp_interval = stat.min_samp_interval;
                        }
                        if (min_samp_interval.dim() == 1) {
                            min_samp_interval = min_samp_interval.view({M,1});
                        }

                        const float subdivide_samp_thres_now =
                            opt_params_.subdivide_force_to_target_size_
                                ? opt_params_.subdivide_samp_thres_at_target_
                                : opt_params_.subdivide_samp_thres_;
                        const double subdivide_prop_now = std::clamp(
                            static_cast<double>(
                                opt_params_.subdivide_force_to_target_size_
                                    ? opt_params_.subdivide_prop_at_target_
                                    : opt_params_.subdivide_prop_),
                            0.0, 1.0);

                        auto size_thres = min_samp_interval * subdivide_samp_thres_now; // [M,1]
                        auto vox_size = voxel_model_->voxSize(); // [M] or [M,1]
                        if (vox_size.dim() == 1) vox_size = vox_size.view({M,1});
                        else if (vox_size.dim() == 2 && vox_size.size(1) == 1) { /* ok */ }
                        else vox_size = vox_size.reshape({M,1});
                        auto vox_size_1d = vox_size.squeeze(1).contiguous();

                        auto large_enough = (vox_size * 0.5 > size_thres).squeeze(1); // [M] bool
                        auto octlv = voxel_model_->octLevel(); // [M] or [M,1]
                        if (octlv.defined() && octlv.dim() == 2 && octlv.size(1) == 1) {
                            octlv = octlv.squeeze(1);
                        }
                        auto non_finest =
                            (octlv.to(torch::kInt32) < voxel_model_->maxNumLevels()); // [M] bool

                        auto rendered_depth_candidate_mask =
                            rendered_depth_candidate_mask_for(M, vox_size_1d.device());
                        auto artificial_subdiv_block_mask =
                            artificial_subdiv_block_mask_for(M, vox_size_1d.device());
                        n_rendered_depth_blocked_normal =
                            rendered_depth_candidate_mask.sum().item<int64_t>();
                        n_artificial_blocked_normal =
                            artificial_subdiv_block_mask.sum().item<int64_t>();
                        auto valid_mask_svraster =
                            (large_enough &
                             non_finest &
                             (~rendered_depth_candidate_mask) &
                             (~artificial_subdiv_block_mask))
                                .to(torch::kBool); // [M]
                        auto normal_candidate_mask = valid_mask_svraster.clone();
                        if (opt_params_.subdivide_force_to_target_size_) {
                            normal_candidate_mask =
                                (valid_mask_svraster &
                                 (vox_size_1d <= opt_params_.subdivide_target_vox_size_))
                                    .to(torch::kBool);
                        }
                        n_normal_candidates = normal_candidate_mask.sum().item<int64_t>();

                        // Priority: may be undefined/empty right after structural changes.
                        auto priority = voxel_model_->subdivisionPriority(); // [M]
                        if (!priority.defined() || priority.numel() != M) {
                            priority = torch::zeros(
                                {M},
                                torch::TensorOptions().dtype(torch::kFloat32).device(normal_candidate_mask.device()));
                        } else if (priority.dim() == 2 && priority.size(1) == 1) {
                            priority = priority.squeeze(1);
                        } else if (priority.dim() != 1) {
                            priority = priority.reshape({M});
                        }

                        std::cout << "[SUBDIV:prep] N=" << M
                                  << " normal_candidates=" << n_normal_candidates
                                  << " rendered_depth_blocked=" << n_rendered_depth_blocked_normal
                                  << " artificial_blocked=" << n_artificial_blocked_normal
                                  << " samp_thres_now=" << subdivide_samp_thres_now
                                  << " prop_now=" << subdivide_prop_now
                                  << " force_target=" << (opt_params_.subdivide_force_to_target_size_ ? 1 : 0)
                                  << "\n";

                        if (n_normal_candidates > 0) {
                            priority = priority * normal_candidate_mask.to(priority.scalar_type());

                            auto normal_selected_mask =
                                torch::zeros_like(normal_candidate_mask, torch::kBool);
                            if (iter <= opt_params_.subdivide_all_until_) {
                                normal_selected_mask = normal_candidate_mask.clone();
                            } else {
                                auto pos_idx = normal_candidate_mask.nonzero().squeeze(1); // [K]
                                auto pos_vals = priority.index({pos_idx});                 // [K]
                                double q = std::max(0.0, 1.0 - subdivide_prop_now);
                                auto thres = (pos_vals.numel() > 0)
                                        ? pos_vals.quantile(q)
                                        : torch::tensor(0.0, pos_vals.options());
                                if (pos_vals.numel() > 0) {
                                    auto pick = (pos_vals > thres); // [K]
                                    normal_selected_mask.index_put_({pos_idx}, pick);
                                    normal_selected_mask =
                                        (normal_selected_mask & normal_candidate_mask).to(torch::kBool);
                                }
                            }

                            int max_n_subdiv = compute_max_n_subdiv();
                            if (max_n_subdiv <= 0) {
                                std::cout << "[SUBDIV:skip] cap reached before normal stage (max_n_subdiv<=0)\n";
                                normal_selected_mask =
                                    torch::zeros_like(normal_candidate_mask, torch::kBool);
                            } else {
                                int64_t num_sel = normal_selected_mask.sum().item<int64_t>();
                                if (num_sel > max_n_subdiv) {
                                    auto pos_idx = normal_selected_mask.nonzero().squeeze(1); // [K]
                                    auto pos_vals = priority.index({pos_idx});                // [K]
                                    auto sort_pair = pos_vals.sort(/*dim=*/0, /*descending=*/true);
                                    auto order_desc = std::get<1>(sort_pair).to(torch::kLong).contiguous();
                                    auto keep_local = order_desc.index(
                                        {torch::indexing::Slice(0, max_n_subdiv)}).contiguous();
                                    auto keep_idx = pos_idx.index_select(0, keep_local).contiguous();
                                    normal_selected_mask =
                                        torch::zeros_like(normal_candidate_mask, torch::kBool);
                                    normal_selected_mask.index_put_({keep_idx}, true);
                                }
                            }

                            n_subdiv_normal_selected =
                                normal_selected_mask.sum().item<int64_t>();
                            if (n_subdiv_normal_selected > 0) {
                                const std::array<int64_t, 3> source_counts_before_subdiv =
                                    source_counts_for_current_voxels();
                                const std::array<int64_t, 3> selected_parent_counts =
                                    selected_source_counts(normal_selected_mask);
                                voxel_model_->subdividing(normal_selected_mask);
                                account_subdivision_lineage_births(
                                    source_counts_before_subdiv,
                                    selected_parent_counts,
                                    "normal");
                                did_subdivide = true;
                            }
                        }
                    }

                    auto vox_size_after = voxel_model_->voxSize(); // [N] or [N,1]
                    if (vox_size_after.dim() == 2 && vox_size_after.size(1) == 1) {
                        vox_size_after = vox_size_after.squeeze(1);
                    } else if (vox_size_after.dim() != 1) {
                        vox_size_after = vox_size_after.reshape({-1});
                    }
                    const double vox_size_min_after =
                        (vox_size_after.numel() > 0) ? vox_size_after.min().item<double>() : 0.0;
                    const double vox_size_max_after =
                        (vox_size_after.numel() > 0) ? vox_size_after.max().item<double>() : 0.0;
                    const int after = voxel_model_->numVoxels();

                    const int64_t n_subdiv_total_selected =
                        n_subdiv_aggressive_total + n_subdiv_normal_selected;
                    if (opt_params_.subdivide_force_to_target_size_) {
                        std::cout << "[SUBDIV/target] target_vox_size="
                                  << opt_params_.subdivide_target_vox_size_
                                  << " aggressive_rounds=" << aggressive_rounds
                                  << " selected_aggressive=" << n_subdiv_aggressive_total
                                  << " rendered_depth_blocked_aggressive="
                                  << n_rendered_depth_blocked_aggressive_total
                                  << " artificial_blocked_aggressive="
                                  << n_artificial_blocked_aggressive_total
                                  << " candidates_normal=" << n_normal_candidates
                                  << " rendered_depth_blocked_normal="
                                  << n_rendered_depth_blocked_normal
                                  << " artificial_blocked_normal="
                                  << n_artificial_blocked_normal
                                  << " selected_normal=" << n_subdiv_normal_selected
                                  << " selected_total=" << n_subdiv_total_selected
                                  << "\n";
                    } else {
                        std::cout << "[SUBDIV/selection] selected_total=" << n_subdiv_total_selected
                                  << " selected_aggressive=0"
                                  << " rendered_depth_blocked_aggressive="
                                  << n_rendered_depth_blocked_aggressive_total
                                  << " artificial_blocked_aggressive="
                                  << n_artificial_blocked_aggressive_total
                                  << " rendered_depth_blocked_normal="
                                  << n_rendered_depth_blocked_normal
                                  << " artificial_blocked_normal="
                                  << n_artificial_blocked_normal
                                  << " selected_normal=" << n_subdiv_normal_selected
                                  << "\n";
                    }

                    if (did_subdivide) {
                        std::cout << "[SUBDIVIDING] " << std::setw(7) << before
                                  << " => "          << std::setw(7) << after
                                  << " (x" << std::fixed << std::setprecision(2)
                                  << (double)after / std::max(1, before) << ")\n";
                    } else {
                        std::cout << "[SUBDIV:skip] selected_total=0\n";
                    }
                    std::cout << "[SUBDIV/size] before_min=" << vox_size_min_before
                              << " before_max=" << vox_size_max_before
                              << " after_min=" << vox_size_min_after
                              << " after_max=" << vox_size_max_after
                              << "\n";
                }
            }
            if (need_pruning) {
                run_pruning();
            }
            // Keep SVRaster behavior: clear accumulated subdivision priority
            // after each adapt round that enters the subdivision branch.
            if (need_subdividing) {
                voxel_model_->resetSubdivisionPriority();
            }
            voxel_model_->createTrainer(
                opt_params_.geo_lr_,
                opt_params_.sh0_lr_,
                opt_params_.shs_lr_,
                opt_params_.optim_beta1_,
                opt_params_.optim_beta2_,
                opt_params_.optim_eps_,
                opt_params_.lr_decay_ckpt_,
                opt_params_.lr_decay_mult_
            );
            voxel_model_->schedulerLoadStateDict(sched_state);
            // Empty CUDA cache as SV does
            {
                py::gil_scoped_acquire gil;
                py::module_ torch_mod = py::module_::import("torch");
                torch_mod.attr("cuda").attr("empty_cache")();
            }
            last_densify_iter_ = iter;
        }
    }
    // Update learning rate
    voxel_model_->schedulerStep();

    if (enable_rerun_ && !rerun_final_only_) {
        // ----- 1) FULL VOXELS (unchanged) -----
        torch::Tensor centers_all = voxel_model_->voxCenter(); // [N,3]
        torch::Tensor sizes_all   = voxel_model_->voxSize();   // [N] or [N,1]
        // colors from SH0 + density as before
        torch::Tensor colors_all;
        {
            torch::Tensor sh0 = voxel_model_->sh0();
            {
                py::gil_scoped_acquire gil2;
                static py::module act_mod = py::module::import("src.utils.activation_utils");
                py::object rgb_py = act_mod.attr("shzero2rgb")(py::cast(sh0));
                colors_all = rgb_py.cast<torch::Tensor>().contiguous();
            }

            torch::Tensor density = voxel_model_->voxelDensityMean();
            if (density.defined() && density.numel() == centers_all.size(0)) {
                auto d_cpu = density.view({-1}).to(torch::kCPU);
                float d_min = d_cpu.min().item().toFloat();
                float d_max = d_cpu.max().item().toFloat();
                float eps   = 1e-6f;
                float range = d_max - d_min;

                torch::Tensor alpha_cpu;
                if (range < eps) {
                    alpha_cpu = torch::full_like(d_cpu, 0.8f);
                } else {
                    alpha_cpu = (d_cpu - d_min) / range;
                    alpha_cpu = alpha_cpu.clamp(0.05f, 1.0f);
                }
                auto col_cpu = colors_all.to(torch::kCPU);
                TORCH_CHECK(col_cpu.dim() == 2 &&
                            col_cpu.size(0) == alpha_cpu.size(0),
                            "colors and density must have same N");
                if (col_cpu.size(1) == 3) {
                    auto N = col_cpu.size(0);
                    auto col_rgba = torch::zeros({N, 4}, col_cpu.options());
                    col_rgba.index_put_(
                        {torch::indexing::Slice(), torch::indexing::Slice(0, 3)},
                        col_cpu
                    );
                    col_rgba.index_put_(
                        {torch::indexing::Slice(), 3},
                        alpha_cpu
                    );
                    colors_all = col_rgba.to(colors_all.device());
                } else if (col_cpu.size(1) == 4) {
                    col_cpu.index_put_(
                        {torch::indexing::Slice(), 3},
                        alpha_cpu
                    );
                    colors_all = col_cpu.to(colors_all.device());
                } else {
                    TORCH_CHECK(false, "colors must be [N,3] or [N,4]");
                }
            }
        }
        // Log full voxel field sparsely to keep Rerun memory bounded on long runs.
        // if ((iter % 20) == 0) {
        //     sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
        //         centers_all, sizes_all, colors_all, iter
        //     );
        // }
        sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
            centers_all, sizes_all, colors_all, iter
        );

        // visualize real-only voxels each iteration:
        // real from PCD + promoted-artificial voxels
        {
            auto art_mask_all = voxel_model_->artificialMask();
            auto promoted_mask_all = voxel_model_->promotedartificialMask();
            auto real_mask_all = torch::ones(
                {centers_all.size(0)},
                torch::TensorOptions().dtype(torch::kBool).device(centers_all.device()));

            if (art_mask_all.defined()) {
                if (art_mask_all.dim() == 2 && art_mask_all.size(1) == 1) {
                    art_mask_all = art_mask_all.squeeze(1);
                }
                art_mask_all = art_mask_all.to(centers_all.device()).to(torch::kBool).contiguous().view({-1});
                if (art_mask_all.numel() == centers_all.size(0)) {
                    real_mask_all = real_mask_all & (~art_mask_all);
                }
            }
            if (promoted_mask_all.defined()) {
                if (promoted_mask_all.dim() == 2 && promoted_mask_all.size(1) == 1) {
                    promoted_mask_all = promoted_mask_all.squeeze(1);
                }
                promoted_mask_all = promoted_mask_all.to(centers_all.device()).to(torch::kBool).contiguous().view({-1});
                if (promoted_mask_all.numel() == centers_all.size(0)) {
                    real_mask_all = real_mask_all | promoted_mask_all;
                }
            }

            auto real_idx = torch::nonzero(real_mask_all).view({-1});
            if (real_idx.numel() > 0) {
                auto centers_real = centers_all.index_select(0, real_idx).contiguous();
                auto sizes_real = sizes_all.index_select(0, real_idx).contiguous();
                torch::Tensor colors_real;
                if (colors_all.defined() && colors_all.numel() > 0 &&
                    colors_all.dim() == 2 && colors_all.size(0) == centers_all.size(0)) {
                    colors_real = colors_all.index_select(0, real_idx).contiguous();
                }
                // sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
                //     centers_real, sizes_real, colors_real, iter, "world/voxels_real"
                // );
            }
        }
        // Keep artificials/all synchronized with final topology state of this iteration.
        // increasePcd() logs artificial topics at insertion time (pre-adapt); this call
        // rewrites artificials/all after prune/subdivide so it matches /voxels.
        voxel_model_->logLiveOrbVoxels(iter, colors_all);
        voxel_model_->logLiveInactiveGeoVoxels(iter, colors_all);
        voxel_model_->logLiveRgbdFillRenderHolesVoxels(iter, colors_all);
        voxel_model_->logLiveDepthAnythingFillHolesVoxels(iter, colors_all);
        voxel_model_->logFinalartificialVoxels(iter);
        voxel_model_->logFinalPromotedartificialVoxels(iter);

        // ----- 2) NEAR VOXELS (debug overlay) -----
        if (debug_has_near &&
            debug_near_centers.defined() &&
            debug_near_centers.numel() > 0)
        {
            auto centers_near = debug_near_centers;         // [K,3] CUDA or CPU
            auto sizes_near   = debug_near_sizes;           // [K,1] or [K]

        // ensure sizes_near is [K,1] on CPU
        if (sizes_near.dim() == 1) {
            sizes_near = sizes_near.view({sizes_near.size(0), 1});
        } else if (sizes_near.dim() == 2 && sizes_near.size(1) == 1) {
            // ok
        } else {
            sizes_near = sizes_near.reshape({sizes_near.size(0), 1});
        }

        auto K = centers_near.size(0);
        torch::Tensor colors_near = torch::zeros({K, 4}, centers_near.options());
        colors_near.index_put_({torch::indexing::Slice(), 0}, 1.0f);  // R
        colors_near.index_put_({torch::indexing::Slice(), 3}, 0.7f);  // alpha

        std::cout << "[DEBUG NEAR] visualizing " << K
                << " near voxels in rerun (red boxes)\n";

        sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
            centers_near,
            sizes_near,
            colors_near,
            iter,
            "world/voxels_near"    // <-- separate entity in blueprint
        );
        }
        // ----- 2b) GEOMETRIC-NEAR VOXELS (debug overlay) -----
        if (debug_has_near_geom &&
            debug_near_geom_centers.defined() &&
            debug_near_geom_centers.numel() > 0)
        {
            auto centers_near_geom = debug_near_geom_centers; // [K,3]
            auto sizes_near_geom   = debug_near_geom_sizes;   // [K,1] or [K]

            if (sizes_near_geom.dim() == 1) {
                sizes_near_geom = sizes_near_geom.view({sizes_near_geom.size(0), 1});
            } else if (!(sizes_near_geom.dim() == 2 && sizes_near_geom.size(1) == 1)) {
                sizes_near_geom = sizes_near_geom.reshape({sizes_near_geom.size(0), 1});
            }

            auto Kg = centers_near_geom.size(0);
            torch::Tensor colors_near_geom = torch::zeros({Kg, 4}, centers_near_geom.options());
            colors_near_geom.index_put_({torch::indexing::Slice(), 0}, 1.0f);  // R
            colors_near_geom.index_put_({torch::indexing::Slice(), 1}, 0.5f);  // G
            colors_near_geom.index_put_({torch::indexing::Slice(), 3}, 0.8f);  // alpha

            std::cout << "[DEBUG NEAR GEOM] visualizing " << Kg
                      << " geometric-near voxels in rerun\n";

            sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
                centers_near_geom,
                sizes_near_geom,
                colors_near_geom,
                iter,
                "world/voxels_near_geometric");
        }
        auto visualize_tsdf_class_boxes =
            [&](const torch::Tensor& centers_in,
                const torch::Tensor& sizes_in,
                bool has,
                float r,
                float g,
                float b,
                float a,
                const std::string& entity_path,
                const std::string& label)
        {
            if (!has || !centers_in.defined() || centers_in.numel() == 0) {
                return;
            }

            auto centers = centers_in;
            auto sizes = sizes_in;
            if (!sizes.defined() || sizes.numel() == 0) {
                return;
            }
            if (sizes.dim() == 1) {
                sizes = sizes.view({sizes.size(0), 1});
            } else if (!(sizes.dim() == 2 && sizes.size(1) == 1)) {
                sizes = sizes.reshape({sizes.size(0), 1});
            }

            auto K = centers.size(0);
            torch::Tensor colors = torch::zeros({K, 4}, centers.options());
            colors.index_put_({torch::indexing::Slice(), 0}, r);
            colors.index_put_({torch::indexing::Slice(), 1}, g);
            colors.index_put_({torch::indexing::Slice(), 2}, b);
            colors.index_put_({torch::indexing::Slice(), 3}, a);

            std::cout << "[DEBUG TSDF] visualizing " << K
                      << " " << label << " voxels in rerun\n";

            sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
                centers,
                sizes,
                colors,
                iter,
                entity_path);
        };

        visualize_tsdf_class_boxes(
            debug_tsdf_free_centers,
            debug_tsdf_free_sizes,
            debug_has_tsdf_free,
            0.0f, 0.9f, 1.0f, 0.45f,
            "world/tsdf/free_space",
            "TSDF free-space");
        visualize_tsdf_class_boxes(
            debug_tsdf_occupied_centers,
            debug_tsdf_occupied_sizes,
            debug_has_tsdf_occupied,
            1.0f, 0.0f, 1.0f, 0.45f,
            "world/tsdf/occupied_side",
            "TSDF occupied-side");
        visualize_tsdf_class_boxes(
            debug_tsdf_surface_centers,
            debug_tsdf_surface_sizes,
            debug_has_tsdf_surface,
            0.0f, 1.0f, 0.0f, 0.35f,
            "world/tsdf/surface_band",
            "TSDF surface-band");
        visualize_tsdf_class_boxes(
            debug_tsdf_unknown_centers,
            debug_tsdf_unknown_sizes,
            debug_has_tsdf_unknown,
            0.5f, 0.5f, 0.5f, 0.22f,
            "world/tsdf/unknown",
            "TSDF unknown");

        auto visualize_tsdf_corner_points =
            [&](const torch::Tensor& points_in,
                const torch::Tensor& colors_in,
                bool has,
                float r,
                float g,
                float b,
                float a,
                const std::string& entity_path,
                const std::string& label)
        {
            if (!has || !points_in.defined() || points_in.numel() == 0) {
                return;
            }
            auto points = points_in;
            if (points.dim() != 2 || points.size(1) != 3) {
                return;
            }

            torch::Tensor colors = colors_in;
            if (!colors.defined() || colors.numel() == 0) {
                colors = torch::zeros({points.size(0), 4}, points.options());
                colors.index_put_({torch::indexing::Slice(), 0}, r);
                colors.index_put_({torch::indexing::Slice(), 1}, g);
                colors.index_put_({torch::indexing::Slice(), 2}, b);
                colors.index_put_({torch::indexing::Slice(), 3}, a);
            }

            std::cout << "[DEBUG TSDF] visualizing " << points.size(0)
                      << " " << label << " corner samples in rerun\n";
            sv::RerunVisualizerBridge::instance().visualizePoints3D(
                points,
                colors,
                iter,
                entity_path,
                0.012f);
        };

        visualize_tsdf_corner_points(
            debug_tsdf_free_corner_points,
            torch::Tensor(),
            debug_has_tsdf_free_corners,
            0.0f, 0.9f, 1.0f, 0.95f,
            "world/tsdf_samples/free_space_corners",
            "TSDF free-space");
        visualize_tsdf_corner_points(
            debug_tsdf_occupied_corner_points,
            torch::Tensor(),
            debug_has_tsdf_occupied_corners,
            1.0f, 0.0f, 1.0f, 0.95f,
            "world/tsdf_samples/occupied_side_corners",
            "TSDF occupied-side");
        visualize_tsdf_corner_points(
            debug_tsdf_surface_corner_points,
            torch::Tensor(),
            debug_has_tsdf_surface_corners,
            0.0f, 1.0f, 0.0f, 0.95f,
            "world/tsdf_samples/surface_band_corners",
            "TSDF surface-band");
        visualize_tsdf_corner_points(
            debug_tsdf_unknown_corner_points,
            debug_tsdf_unknown_corner_colors,
            debug_has_tsdf_unknown_corners,
            0.5f, 0.5f, 0.5f, 0.95f,
            "world/tsdf_samples/unknown_corners",
            "TSDF unknown");

        // ----- 3) SDF/TSDF-PRUNED VOXELS (debug overlay) -----
        if (debug_has_tsdf &&
            debug_tsdf_centers.defined() &&
            debug_tsdf_centers.numel() > 0)
        {
            auto centers_tsdf = debug_tsdf_centers;   // [K_tsdf,3]
            auto sizes_tsdf   = debug_tsdf_sizes;     // [K_tsdf,1] or [K_tsdf]

        // ensure sizes_tsdf is [K_tsdf,1]
        if (sizes_tsdf.dim() == 1) {
            sizes_tsdf = sizes_tsdf.view({sizes_tsdf.size(0), 1});
        } else if (sizes_tsdf.dim() == 2 && sizes_tsdf.size(1) == 1) {
            // ok
        } else {
            sizes_tsdf = sizes_tsdf.reshape({sizes_tsdf.size(0), 1});
        }

        auto Kt = centers_tsdf.size(0);
        // visualize with a different color, e.g. blue
        torch::Tensor colors_tsdf = torch::zeros({Kt, 4}, centers_tsdf.options());
        colors_tsdf.index_put_({torch::indexing::Slice(), 2}, 1.0f);  // B = 1
        colors_tsdf.index_put_({torch::indexing::Slice(), 3}, 0.7f);  // alpha = 0.7

        std::cout << "[DEBUG TSDF] visualizing " << Kt
                << " SDF-pruned voxels in rerun (blue boxes)\n";

        sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
            centers_tsdf,
            sizes_tsdf,
            colors_tsdf,
            iter,
            "world/voxels_sdf_pruned"
        );
        }
        // ----- 4) FINAL-PRUNED VOXELS (debug overlay) -----
        if (debug_has_pruned &&
            debug_pruned_centers.defined() &&
            debug_pruned_centers.numel() > 0)
        {
            auto centers_pruned = debug_pruned_centers; // [K_prune,3]
            auto sizes_pruned   = debug_pruned_sizes;   // [K_prune,1] or [K_prune]

        if (sizes_pruned.dim() == 1) {
            sizes_pruned = sizes_pruned.view({sizes_pruned.size(0), 1});
        } else if (sizes_pruned.dim() == 2 && sizes_pruned.size(1) == 1) {
            // ok
        } else {
            sizes_pruned = sizes_pruned.reshape({sizes_pruned.size(0), 1});
        }

        auto Kp = centers_pruned.size(0);
        torch::Tensor colors_pruned = torch::zeros({Kp, 4}, centers_pruned.options());
        colors_pruned.index_put_({torch::indexing::Slice(), 0}, 1.0f);  // R
        colors_pruned.index_put_({torch::indexing::Slice(), 1}, 1.0f);  // G (yellow)
        colors_pruned.index_put_({torch::indexing::Slice(), 3}, 0.8f);  // alpha

        // sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
        //     centers_pruned,
        //     sizes_pruned,
        //     colors_pruned,
        //     iter,
        //     "world/voxels_pruned"   // separate entity path
        // );
        }

        // ----- 6) FAR-ONLY PRUNED VOXELS (debug overlay) -----
        if (debug_has_far_pruned &&
            debug_far_pruned_centers.defined() &&
            debug_far_pruned_centers.numel() > 0)
        {
            auto centers_far_pruned = debug_far_pruned_centers; // [K_far,3]
            auto sizes_far_pruned   = debug_far_pruned_sizes;   // [K_far,1] or [K_far]

            if (sizes_far_pruned.dim() == 1) {
                sizes_far_pruned = sizes_far_pruned.view({sizes_far_pruned.size(0), 1});
            } else if (sizes_far_pruned.dim() == 2 && sizes_far_pruned.size(1) == 1) {
                // ok
            } else {
                sizes_far_pruned = sizes_far_pruned.reshape({sizes_far_pruned.size(0), 1});
            }

            auto Kf = centers_far_pruned.size(0);
            torch::Tensor colors_far_pruned = torch::zeros({Kf, 4}, centers_far_pruned.options());
            colors_far_pruned.index_put_({torch::indexing::Slice(), 0}, 1.0f);  // R
            colors_far_pruned.index_put_({torch::indexing::Slice(), 3}, 0.9f);  // alpha

            std::cout << "[DEBUG FAR] visualizing " << Kf
                      << " far-pruned voxels in rerun\n";

            sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
                centers_far_pruned,
                sizes_far_pruned,
                colors_far_pruned,
                iter,
                "world/far_voxels");
        }

    }

    if (mDevice == torch::kCUDA) torch::cuda::synchronize();

    {
        torch::NoGradGuard no_grad;
        ema_loss_for_log_ = 0.4f * loss.item<float>() + 0.6f * ema_loss_for_log_;

        if (keyframe_record_interval_ &&
            getIteration() % keyframe_record_interval_ == 0)
            recordKeyframeRendered(
                masked_image,
                gt_image,
                viewpoint_cam->fid_,
                result_dir_, result_dir_, result_dir_
            );
        auto iter_end_timing = std::chrono::steady_clock::now();
        auto iter_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        iter_end_timing - iter_start_timing).count();

         // Log and save
         if (training_report_interval_ && (getIteration() % training_report_interval_ == 0))
             sv::VoxelTrainer::trainingReport(
                 getIteration(),
                 opt_params_.iterations_,
                 Ll1,
                 loss,
                 ema_loss_for_log_,
                 mse,
                 iter_time,
                 *voxel_model_,
                 *scene_,
                 pipe_params_,
                 background_
             );

        // if (training_report_interval_ && (getIteration() % training_report_interval_ == 0)) {
        //     std::cout << std::fixed << std::setprecision(6)
        //               << "[loss_terms] iter=" << getIteration()
        //               << " sparse_depth(on=" << (dbg_sparse_depth_on ? 1 : 0)
        //               << " raw=" << dbg_sparse_depth_raw
        //               << " w=" << dbg_sparse_depth_w << ")"
        //               << " depthanythingv2(on=" << (dbg_depthanything_on ? 1 : 0)
        //               << " raw=" << dbg_depthanything_raw
        //               << " w=" << dbg_depthanything_w << ")"
        //               << " T_concen(on=" << (dbg_t_concen_on ? 1 : 0)
        //               << " raw=" << dbg_t_concen_raw
        //               << " w=" << dbg_t_concen_w << ")"
        //               << " T_inside(on=" << (dbg_t_inside_on ? 1 : 0)
        //               << " raw=" << dbg_t_inside_raw
        //               << " w=" << dbg_t_inside_w << ")"
        //               << std::endl;
        // }

        if ((all_keyframes_record_interval_ && getIteration() % all_keyframes_record_interval_ == 0)
            )
        {
            renderAndRecordAllKeyframes();
            savePly(result_dir_ / std::to_string(iteration_) / "ply");
        }
        
        if (loop_closure_iteration_)
            loop_closure_iteration_ = false;

        // Extract scalars for csv logging
        const float l1_val    = Ll1.item<float>();
        const float l2_val    = mse.item<float>();
        const float photo_val = photo_loss.item<float>();
        const float train_loss_val = loss.item<float>();
        // Combined CSV (iter, l1, l2, train_loss)
        if (loss_log_) {
            loss_log_ << std::fixed << std::setprecision(6)
                    << iter << ',' << l1_val << ',' << l2_val << ',' << train_loss_val << '\n';
        }
        if (loss_l1_log_) { loss_l1_log_ << std::fixed << std::setprecision(6) << iter << ',' << l1_val << '\n'; }
        if (loss_l2_log_) { loss_l2_log_ << std::fixed << std::setprecision(6) << iter << ',' << l2_val << '\n'; }
        if (loss_ssim_log_) { loss_ssim_log_ << std::fixed << std::setprecision(6) << iter << ',' << photo_val << '\n'; }
        if ((iter % 50) == 0) {
            loss_log_.flush();
            loss_l1_log_.flush();
            loss_l2_log_.flush();
            loss_ssim_log_.flush();
        }

        float loss_val = train_loss_val;
        int fid = viewpoint_cam->fid_;                 // key-frame ID being trained
        // --- 1) ensure our vectors are big enough ---
        if (fid >= static_cast<int>(best_loss_per_kf_.size())) {
            size_t newSize = fid + 1;
            best_loss_per_kf_.resize(newSize,
                                    std::numeric_limits<float>::infinity());
            worst_loss_per_kf_.resize(newSize,
                                    -std::numeric_limits<float>::infinity());
        }
        // references into the right slot
        float &best  = best_loss_per_kf_[fid];
        float &worst = worst_loss_per_kf_[fid];
        // --- 2) update “best” for this KF ---
        if (loss_val < best) {
            best = loss_val;
            auto kf_dir = extrema_dir_ / ("kf" + std::to_string(fid));
            std::filesystem::create_directories(kf_dir);
            saveTensor(gt_image,     "best_gt",     kf_dir.string(), iteration_, fid);
            saveTensor(masked_image, "best_masked", kf_dir.string(), iteration_, fid);
        }
        // --- 3) update “worst” for this KF ---
        if (loss_val > worst) {
            worst = loss_val;
            auto kf_dir = extrema_dir_ / ("kf" + std::to_string(fid));
            std::filesystem::create_directories(kf_dir);
            saveTensor(gt_image,     "worst_gt",     kf_dir.string(), iteration_, fid);
            saveTensor(masked_image, "worst_masked", kf_dir.string(), iteration_, fid);
        }
    }
}

void VoxelMapper::combineMappingOperations()
{
    // Get Mapping Operations
    while (mpSLAM->getAtlas()->hasMappingOperation()) {
        ORB_SLAM3::MappingOperation opr =
            mpSLAM->getAtlas()->getAndPopMappingOperation();
        switch (opr.meOperationType)
        {
        case ORB_SLAM3::MappingOperation::OprType::LocalMappingBA:
        {
        bool kf_changed = false;
            // std::cout << "[Gaussian Mapper]Local BA Detected."
            //           << std::endl;
            // Get new keyframes
            auto& associated_kfs = opr.associatedKeyFrames();
            // Add keyframes to the scene
            for (auto& kf : associated_kfs) {
                // Keyframe Id
                auto kfid = std::get<0>(kf);
                std::shared_ptr<VoxelKeyframe> pkf = scene_->getKeyframe(kfid);
                // If the keyframe is already in the scene, only update the pose.
                // Otherwise create a new one
                if (pkf) {
                //  std::cout << "if pkf" << std::endl;
                    auto& pose = std::get<2>(kf);
                    pkf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>());
                //  pkf->computeTransformTensors();
                    // Give local BA keyframes times of use
                    increaseKeyframeTimesOfUse(pkf, local_BA_increased_times_of_use_);
                    kf_changed = true;
                }
                else {
                // std::cout << "no pkf" << std::endl;
                handleNewKeyframe(kf);                   // still void
                pkf = scene_->getKeyframe(kfid);
                if (pkf) {
                    // Its original_image_ is Float32 RGB in [0..1], shape (3,H,W)
                    torch::Tensor chw = pkf->original_image_.detach().cpu().clamp(0,1);
                    torch::Tensor hwc = chw.permute({1,2,0}).contiguous(); // (H,W,3), float32
                    // Copy into a CV_32FC3 Mat (RGB)
                    const int H = hwc.size(0);
                    const int W = hwc.size(1);
                    cv::Mat img_float(H, W, CV_32FC3);
                    std::memcpy(img_float.data, hwc.data_ptr<float>(), H*W*3*sizeof(float));
                    static const auto proj_dir = result_dir_ / "proj_debug";
                    static const auto imgs_dir = proj_dir / "imgs";
                    std::filesystem::create_directories(imgs_dir);
                    saveKfPng_fromFloatRGB(img_float, kfid, imgs_dir);
                }
                }
            }
            // Get new points
            auto& associated_points = opr.associatedMapPoints();
            auto& points = std::get<0>(associated_points);
            auto& colors = std::get<1>(associated_points);

            // Add new points to the model
            const int iter = getIteration();
            if (initial_mapped_ && points.size() >= 30) {
                torch::NoGradGuard no_grad;
                std::unique_lock<std::mutex> lock_render(mutex_render_);
            // log_increase_batch_npy(result_dir_, points, colors, getIteration(), next_batch_index_);
            // ++next_batch_index_;
            //  voxel_model_->increasePcd(points, colors, getIteration(), kfs_for_bounding);
                // py::object sched_state = voxel_model_->schedulerStateDict();

                // Build training camera list from the keyframes we keep in the scene
                std::vector<sv::MiniCam> tr_cams;
                tr_cams.reserve(scene_->keyframes().size());
                for (auto& kv : scene_->keyframes()) {
                    // OLD:
                    // if (kv.second) tr_cams.push_back(kv.second->toMiniCam());
                    if (kv.second) {
                        tr_cams.push_back(
                            kv.second->toMiniCam(kv.second->image_height_, kv.second->image_width_));
                    }
                }
                if (points.size() >= 30) {
                    if (enable_rerun_ && !rerun_final_only_) {
                        appendAndLogOrbRawPointBatchToRerun(
                            points,
                            colors,
                            iter);
                        voxel_model_->setNextRealInsertionRerunEntityPath(
                            "world/orb/voxels_created");
                    }
                    voxel_model_->increasePcd(
                        points,
                        colors,
                        getIteration(),
                        tr_cams);
                    if (sensor_type_ == RGBD) {
                        const sv::VoxelModel::IncreasePcdStats insert_stats =
                            voxel_model_->lastIncreasePcdStats();
                        tsdf_ablation_rgbd_points_created_ += insert_stats.new_voxels;
                        tsdf_ablation_rgbd_points_lineage_created_ += insert_stats.new_voxels;
                    }
                    if (enable_rerun_ && !rerun_final_only_) {
                        voxel_model_->setNextRealInsertionRerunEntityPath("");
                    }
                    if (voxel_model_ && voxel_model_->consumeartificialFillFlag()) {
                        last_artificial_fill_iter_ = static_cast<int64_t>(iter);
                        std::cout << "[VoxelMapper] artificial fill happened at iter "
                                << iter << "\n";
                    }
                }
                // voxel_model_->createTrainer(
                //                             opt_params_.geo_lr_,
                //                             opt_params_.sh0_lr_,
                //                             opt_params_.shs_lr_,
                //                             opt_params_.optim_beta1_,
                //                             opt_params_.optim_beta2_,
                //                             opt_params_.optim_eps_,
                //                             opt_params_.lr_decay_ckpt_,
                //                             opt_params_.lr_decay_mult_);
                // voxel_model_->schedulerLoadStateDict(sched_state);
            }

            // // Only try to grow if model was initialized and we have any new points
            // // if (initial_mapped_ && !points.empty()) {
            // if (initial_mapped_ && points.size() >= 30) {
            //     // 1) Get current bound
            //     torch::Tensor sc  = voxel_model_->SceneCenter();  // [3] if set
            //     torch::Tensor se  = voxel_model_->SceneExtent();  // [1] if set
            //     // If the model hasn't been initialized yet (shouldn’t happen here), skip.
            //     if (sc.defined() && se.defined() && sc.numel()==3 && se.numel()==1) {
            //         auto sc_cpu = sc.detach().to(torch::kCPU);
            //         auto se_cpu = se.detach().to(torch::kCPU);
            //         // scene_min = center - 0.5 * extent;  scene_max = center + 0.5 * extent
            //         float cx = sc_cpu[0].item<float>();
            //         float cy = sc_cpu[1].item<float>();
            //         float cz = sc_cpu[2].item<float>();
            //         float ex = se_cpu[0].item<float>() * 0.5f;
            //         float minx = cx - ex, maxx = cx + ex;
            //         float miny = cy - ex, maxy = cy + ex;
            //         float minz = cz - ex, maxz = cz + ex;
            //         const float eps = 1e-6f; // tiny numerical slack
            //         // 2) Keep only points outside current AABB
            //         std::vector<float> out_pts;
            //         std::vector<float> out_cols;
            //         out_pts.reserve(points.size());  // upper bound
            //         out_cols.reserve(colors.size());
            //         const size_t N = points.size() / 3;
            //         for (size_t i = 0; i < N; ++i) {
            //             float x = points[3*i + 0];
            //             float y = points[3*i + 1];
            //             float z = points[3*i + 2];
            //             bool outside =
            //                 (x < minx - eps) || (x > maxx + eps) ||
            //                 (y < miny - eps) || (y > maxy + eps) ||
            //                 (z < minz - eps) || (z > maxz + eps);
            //             if (outside) {
            //                 out_pts.push_back(x);
            //                 out_pts.push_back(y);
            //                 out_pts.push_back(z);

            //                 out_cols.push_back(colors[3*i + 0]);
            //                 out_cols.push_back(colors[3*i + 1]);
            //                 out_cols.push_back(colors[3*i + 2]);
            //             }
            //         }
            //         // 3) Only grow if enough out-of-bounds points
            //         const int MIN_OUTSIDE = 10; // your threshold
            //         if ((int)(out_pts.size()/3) >= MIN_OUTSIDE) {
            //             // Optional: ensure we don't move the 'min side' if you want perfect key stability
            //             // (Counts of which side is violated; skip if it would force min to move)
            //             // bool touches_min_side = (min of any coord < minX/Y/Z);
            //             // if (touches_min_side) { /* decide: defer / tile / rebase */ }
            //             torch::NoGradGuard no_grad;
            //             std::unique_lock<std::mutex> lock_render(mutex_render_);
            //             // 4) Actually insert *only* the outside points
            //             voxel_model_->increasePcd(out_pts, out_cols, getIteration());
            //             // voxel_model_->increasePcd(points, colors, getIteration());
            //         }
            //     }
            // }
            // if (kf_changed) {
            //     dumpKeyframesForProjectionFile(scene_->keyframes(), scene_->cameras_,
            //                                 result_dir_ / "proj_debug");
            // }
        }
        break;

        case ORB_SLAM3::MappingOperation::OprType::LoopClosingBA:
        {
            std::cout << "[Voxel Mapper]Loop Closure Detected."
                    << std::endl;

            bool kf_changed = false;
            // Get the loop keyframe scale modification factor
            float loop_kf_scale = opr.mfScale;

            // Get new keyframes (scaled transformation applied in ORB-SLAM3)
            auto& associated_kfs = opr.associatedKeyFrames();

            // std::vector<std::shared_ptr<VoxelKeyframe>> kfs_for_bounding;

             // Mark the transformed points to avoid transforming more than once
             torch::Tensor point_not_transformed_flags =
                 torch::full(
                     {voxel_model_->center_.size(0)},
                     true,
                     torch::TensorOptions().device(device_type_).dtype(torch::kBool));
            //  if (record_loop_ply_)
            //      savePly(result_dir_ / (std::to_string(getIteration()) + "_0_before_loop_correction"));
             int num_transformed = 0;
             // Add keyframes to the scene
             for (auto& kf : associated_kfs) {
                 // Keyframe Id
                 auto kfid = std::get<0>(kf);
                 std::shared_ptr<VoxelKeyframe> pkf = scene_->getKeyframe(kfid);
                 // In case new points are added in handleNewKeyframe()
                 int64_t num_new_points = voxel_model_->center_.size(0) - point_not_transformed_flags.size(0);
                 if (num_new_points > 0)
                     point_not_transformed_flags = torch::cat({
                         point_not_transformed_flags,
                         torch::full({num_new_points}, true, point_not_transformed_flags.options())},
                         /*dim=*/0);
                 // If kf is already in the scene, evaluate the change in pose,
                 // if too large we perform loop correction on its visible model points.
                 // If not in the scene, create a new one.
                 if (pkf) {
                     auto& pose = std::get<2>(kf);
                     // If is loop closure kf
 // if (std::get<4>(kf)) {
 // renderAndRecordKeyframe(pkf, result_dir_, "_0_before_loop_correction");
                         Sophus::SE3f original_pose = pkf->getPosef(); // original_pose = old, inv_pose = new
                         Sophus::SE3f inv_pose = pose.inverse();
                         Sophus::SE3f diff_pose = inv_pose * original_pose;
                         bool large_rot = !diff_pose.rotationMatrix().isApprox(
                             Eigen::Matrix3f::Identity(), large_rot_th_);
                         bool large_trans = !diff_pose.translation().isMuchSmallerThan(
                             1.0, large_trans_th_);
                         if (large_rot || large_trans) {
                             std::cout << "[Voxel Mapper]Large loop correction detected, transforming visible points of kf "
                                     << kfid << std::endl;
                             diff_pose.translation() -= inv_pose.translation(); // t = (R_new * t_old + t_new) - t_new
                             diff_pose.translation() *= loop_kf_scale;          // t = s * (R_new * t_old)
                             diff_pose.translation() += inv_pose.translation(); // t = (s * R_new * t_old) + t_new
                             torch::Tensor diff_pose_tensor =
                                 tensor_utils::EigenMatrix2TorchTensor(
                                     diff_pose.matrix(), device_type_).transpose(0, 1);
                            //  {
                            //      std::unique_lock<std::mutex> lock_render(mutex_render_);
                            //      voxel_model_->scaledTransformVisiblePointsOfKeyframe(
                            //          point_not_transformed_flags,
                            //          diff_pose_tensor,
                            //          pkf->world_view_transform_,
                            //          pkf->full_proj_transform_,
                            //          pkf->creation_iter_,
                            //          stableNumIterExistence(),
                            //          num_transformed,
                            //          loop_kf_scale); // selected xyz *= s
                            //  }
                             // Give loop keyframes times of use
                             increaseKeyframeTimesOfUse(pkf, loop_closure_increased_times_of_use_);
 // renderAndRecordKeyframe(pkf, result_dir_, "_1_after_loop_transforming_points");
 // std::cout<<num_transformed<<std::endl;
                         }
 // }
                     pkf->setPose(
                         pose.unit_quaternion().cast<double>(),
                         pose.translation().cast<double>());
                    //  pkf->computeTransformTensors();
 // if (std::get<4>(kf)) renderAndRecordKeyframe(pkf, result_dir_, "_2_after_pose_correction");

                    kf_changed = true;
                 }
                 else {
                    //  std::cout << "no pkf again" << std::endl;
                     handleNewKeyframe(kf);
                     pkf = scene_->getKeyframe(kfid);
                 }
             }
            //  if (record_loop_ply_)
            //      savePly(result_dir_ / (std::to_string(getIteration()) + "_1_after_loop_correction"));
 // keyframesToJson(result_dir_ / (std::to_string(getIteration()) + "_0_before_loop_correction"));
 
             // Get new points (scaled transformation applied in ORB-SLAM3, so this step is performed at last to avoid scaling twice)
             auto& associated_points = opr.associatedMapPoints();
             auto& points = std::get<0>(associated_points);
             auto& colors = std::get<1>(associated_points);

             // Add new points to the model
             const int iter = getIteration();
             if (initial_mapped_ && points.size() >= 30) {
                std::cout << "adds new points" << std::endl;
                extendAABB_with_flat_xyz(aabb_min_, aabb_max_, points);  // points = std::vector<float>
                have_bounds_ = true;
                torch::NoGradGuard no_grad;
                std::unique_lock<std::mutex> lock_render(mutex_render_);

                // Match Photo-SLAM behavior: insert loop-closure associated points.
                std::vector<sv::MiniCam> tr_cams;
                tr_cams.reserve(scene_->keyframes().size());
                for (auto& kv : scene_->keyframes()) {
                    if (kv.second) {
                        tr_cams.push_back(
                            kv.second->toMiniCam(
                                kv.second->image_height_,
                                kv.second->image_width_));
                    }
                }
                if (points.size() >= 30) {
                    if (enable_rerun_ && !rerun_final_only_) {
                        appendAndLogOrbRawPointBatchToRerun(
                            points,
                            colors,
                            iter);
                        voxel_model_->setNextRealInsertionRerunEntityPath(
                            "world/orb/voxels_created");
                    }
                    voxel_model_->increasePcd(
                        points,
                        colors,
                        iter,
                        tr_cams);
                    if (enable_rerun_ && !rerun_final_only_) {
                        voxel_model_->setNextRealInsertionRerunEntityPath("");
                    }
                    if (voxel_model_ && voxel_model_->consumeartificialFillFlag()) {
                        last_artificial_fill_iter_ = static_cast<int64_t>(iter);
                        std::cout << "[VoxelMapper] artificial fill happened at iter "
                                << iter << "\n";
                    }
                }
             }

            // Mark this iteration
            loop_closure_iteration_ = true;
            // if (kf_changed) {
            //     dumpKeyframesForProjectionFile(scene_->keyframes(), scene_->cameras_,
            //                                 result_dir_ / "proj_debug");
            // }
         }
         break;
 
         case ORB_SLAM3::MappingOperation::OprType::ScaleRefinement:
         {
             std::cout << "[Voxel Mapper]Scale refinement Detected. Transforming all kfs and points..."
                       << std::endl;
 
             float s = opr.mfScale;
             Sophus::SE3f& T = opr.mT;
             if (initial_mapped_) {
                 // Apply the scaled transformation on gaussian model points
                 {
                     std::unique_lock<std::mutex> lock_render(mutex_render_);
                    //  voxel_model_->applyScaledTransformation(s, T);
                 }
                 // Apply the scaled transformation to the scene
                //  scene_->applyScaledTransformation(s, T);
             }
             else { // TODO: the workflow should not come here, delete this branch
                 // Apply the scaled transformation to the cached points
                 for (auto& pt : scene_->cached_point_cloud_) {
                     // pt <- (s * Ryw * pt + tyw)
                     auto& pt_xyz = pt.second.xyz_;
                     pt_xyz *= s;
                     pt_xyz = T.cast<double>() * pt_xyz;
                 }
 
                 // Apply the scaled transformation on gaussian keyframes
                 for (auto& kfit : scene_->keyframes()) {
                     std::shared_ptr<VoxelKeyframe> pkf = kfit.second;
                     Sophus::SE3f Twc = pkf->getPosef().inverse();
                     Twc.translation() *= s;
                     Sophus::SE3f Tyc = T * Twc;
                     Sophus::SE3f Tcy = Tyc.inverse();
                     std::cout << "ScaleRefinement: kf " << Tcy.translation() << std::endl;
                     pkf->setPose(Tcy.unit_quaternion().cast<double>(), Tcy.translation().cast<double>());
                    //  pkf->computeTransformTensors();
                 }
             }
         }
         break;
 
         default:
         {
             throw std::runtime_error("MappingOperation type not supported!");
         }
         break;
         }
     }
 }

 bool VoxelMapper::hasMetInitialMappingConditions() {
     if (!mpSLAM->isShutDown() &&
         mpSLAM->GetNumKeyframes() >= min_num_initial_map_kfs_ &&
         mpSLAM->getAtlas()->hasMappingOperation())
         return true;
 
     bool conditions_met = false;
     return conditions_met;
}

bool VoxelMapper::hasMetIncrementalMappingConditions() {
     if (!mpSLAM->isShutDown() &&
         mpSLAM->getAtlas()->hasMappingOperation())
         return true;
 
     bool conditions_met = false;
     return conditions_met;
}

void VoxelMapper::generateKfidRandomShuffle()
{
     if (scene_->keyframes().empty())
         return;
 
     std::size_t nkfs = scene_->keyframes().size();
     kfid_shuffle_.resize(nkfs);
     std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);
     std::mt19937 g(rd_());
     std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);
 
     kfid_shuffled_ = true;
}

std::shared_ptr<VoxelKeyframe> VoxelMapper::useOneRandomSlidingWindowKeyframe()
{
    // If no keyframes, return nullptr
    if (scene_->keyframes().empty())
        return nullptr;

    // If not shuffled yet, build shuffle
    if (!kfid_shuffled_)
        generateKfidRandomShuffle();

    std::shared_ptr<VoxelKeyframe> viewpoint_cam = nullptr;
    int random_cam_idx;

    if (kfid_shuffled_) {
        int start_shuffle_idx = kfid_shuffle_idx_;
        do {
            // Next shuffled idx
            ++kfid_shuffle_idx_;
            if (kfid_shuffle_idx_ >= kfid_shuffle_.size())
                kfid_shuffle_idx_ = 0;
            // Add 1 time of use to all kfs if they are all unavalible
            if (kfid_shuffle_idx_ == start_shuffle_idx)
                for (auto& kfit : scene_->keyframes())
                    increaseKeyframeTimesOfUse(kfit.second, 1);
            // Get viewpoint kf
            random_cam_idx = kfid_shuffle_[kfid_shuffle_idx_];
            auto random_cam_it = scene_->keyframes().begin();
            for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
                ++random_cam_it;
            viewpoint_cam = (*random_cam_it).second;
        } while (viewpoint_cam->remaining_times_of_use_ <= 0);
    }

    // Count used times
    auto viewpoint_fid = viewpoint_cam->fid_;
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
        kfs_used_times_[viewpoint_fid] = 1;
    else
        ++kfs_used_times_[viewpoint_fid];
    
    // Handle times of use
    --(viewpoint_cam->remaining_times_of_use_);

    return viewpoint_cam;
}

std::shared_ptr<VoxelKeyframe> VoxelMapper::useOneRandomKeyframe()
 {
     if (scene_->keyframes().empty())
         return nullptr;
 
     // Get randomly
     int nkfs = static_cast<int>(scene_->keyframes().size());
     int random_cam_idx = std::rand() / ((RAND_MAX + 1u) / nkfs);
     auto random_cam_it = scene_->keyframes().begin();
     for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
         ++random_cam_it;
     std::shared_ptr<VoxelKeyframe> viewpoint_cam = (*random_cam_it).second;
 
     // Count used times
     auto viewpoint_fid = viewpoint_cam->fid_;
     if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
         kfs_used_times_[viewpoint_fid] = 1;
     else
         ++kfs_used_times_[viewpoint_fid];
 
     return viewpoint_cam;
 }

void VoxelMapper::cullKeyframes()
{
    // Ask ORB-SLAM3 which keyframe IDs are still “live”
    std::unordered_set<unsigned long> kfids =
        mpSLAM->getAtlas()->GetCurrentKeyFrameIds();

     std::vector<unsigned long> kfids_to_erase;
     std::size_t nkfs = scene_->keyframes().size();
     kfids_to_erase.reserve(nkfs);
     for (auto& kfit : scene_->keyframes()) {
         unsigned long kfid = kfit.first;
         if (kfids.find(kfid) == kfids.end()) {
             kfids_to_erase.emplace_back(kfid);
         }
     }
 
     for (auto& kfid : kfids_to_erase) {
         scene_->keyframes().erase(kfid);
     }
}

void VoxelMapper::handleNewKeyframe(
    std::tuple<
        unsigned long,    // 0: keyframe ID
        unsigned long,    // 1: camera ID
        Sophus::SE3f,     // 2: pose
        cv::Mat,          // 3: RGB image
        bool,             // 4: loop‐closure flag (unused here)
        cv::Mat,          // 5: auxiliary (unused here)
        std::vector<float>, // 6: keypoint pixel coords (unused here)
        std::vector<float>, // 7: keypoint local coords (unused here)
        std::string> &kf       // 8: image filename (relative or absolute)
)
{
    // ─── Create a new VoxelKeyframe, exactly like Photo-SLAM’s Gaussian case ─
    std::shared_ptr<VoxelKeyframe> pkf  = std::make_shared<VoxelKeyframe>(std::get<0>(kf), getIteration());
    pkf->znear_ = z_near_;
    // Pose
    auto& pose = std::get<2>(kf);
    pkf->setPose(
        pose.unit_quaternion().cast<double>(),
        pose.translation().cast<double>()
    );
    cv::Mat imgRGB_undistorted, imgAux_undistorted;
    // Camera
    sv::Camera& camera = scene_->cameras_.at(std::get<1>(kf));
    pkf->setCameraParams(camera);

    // Image (left if STEREO)
    cv::Mat imgRGB = std::get<3>(kf);
    if (this->sensor_type_ == STEREO)
        imgRGB_undistorted = imgRGB;
    else
        camera.undistortImage(imgRGB, imgRGB_undistorted);
    // Auxiliary Image
    cv::Mat imgAux = std::get<5>(kf);
    if (this->sensor_type_ == RGBD)
        camera.undistortImage(imgAux, imgAux_undistorted);
    else
        imgAux_undistorted = imgAux;

    pkf->original_image_ =
        tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
    pkf->img_filename_ = std::get<8>(kf);
    pkf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
    pkf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
    pkf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
     
    // Add the new keyframe to the scene
    // pkf->computeTransformTensors();
    scene_->addKeyframe(pkf, &kfid_shuffled_);

    // Give new keyframes times of use and add it to the training sliding window
    increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());

    // Get dense point cloud from the new keyframe to accelerate training
    pkf->img_undist_ = imgRGB_undistorted;
    pkf->img_auxiliary_undist_ = imgAux_undistorted;

    pkf->kps_pixel_ = std::move(std::get<6>(kf));
    pkf->kps_point_local_ = std::move(std::get<7>(kf));
    if (isdoingInactiveGeoDensify())
        increasePcdByKeyframeInactiveGeoDensify(pkf);
    if (depthanything_densify_)
        increasePcdByKeyframeMonoPrior(pkf);
    if (depthanything_fill_holes_)
        scheduleDepthAnythingFillHoles(pkf);
    if (rendered_depth_insert_)
        increasePcdByKeyframeRenderedDepthInsertion(pkf);

    // Prepare multi resolution images for training
    if (device_type_ == torch::kCUDA) {
        cv::cuda::GpuMat img_gpu;
        img_gpu.upload(pkf->img_undist_);
        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::cuda::GpuMat img_resized;
            cv::cuda::resize(img_gpu, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
        }
    }
    else {
        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::Mat img_resized;
            cv::resize(pkf->img_undist_, img_resized,
                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
        }
    }

    try {
        const unsigned long kf_id = std::get<0>(kf);
        // Build MiniCam at native resolution
        const int image_height = pkf->image_height_;
        const int image_width  = pkf->image_width_;
        sv::MiniCam cam = pkf->toMiniCam(image_height, image_width);        // cam.c2w is a 4x4 torch::Tensor, typically on CUDA; move to CPU.
        torch::Tensor c2w_cpu = cam.c2w.to(torch::kCPU).contiguous();
        TORCH_CHECK(c2w_cpu.sizes() == torch::IntArrayRef({4, 4}),
                    "MiniCam.c2w must be 4x4");
        // Torch is row-major; Eigen is column-major by default.
        // Map the data as a row-major Eigen matrix and then copy it into a normal Matrix4f.
        Eigen::Matrix4f T_W_C;
        {
            float* data = c2w_cpu.data_ptr<float>();
            Eigen::Map<const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>> T_row_major(data);
            T_W_C = T_row_major;
        }
        // Now t and R are correct
        Eigen::Vector3f t = T_W_C.block<3,1>(0, 3);
        Eigen::Matrix3f R = T_W_C.block<3,3>(0, 0);
        Eigen::Quaternionf q(R);

        // In RGB-D hole-fill debug runs, show the actual depth map
        // sampled by rgbd_fill_render_holes instead of the RGB keyframe.
        cv::Mat depth_rerun_img;
        const cv::Mat* track_img = &pkf->img_undist_;
        if (sensor_type_ == RGBD && rgbd_fill_render_holes_ &&
            makeRgbdDepthRerunImage(
                pkf->img_auxiliary_undist_,
                RGBD_min_depth_,
                RGBD_max_depth_,
                depth_rerun_img)) {
            track_img = &depth_rerun_img;
        }

        // Intrinsics for Rerun
        const float fx = static_cast<float>(camera.fx());
        const float fy = static_cast<float>(camera.fy());
        const float cx = static_cast<float>(camera.cx());
        const float cy = static_cast<float>(camera.cy());

        // For now, don't send any 2D keypoints (only pose + image).
        std::vector<Eigen::Vector2f> kps_uv;
        std::vector<int>             track_ids;

        constexpr unsigned long rerun_kf_begin = 50;
        if (enable_rerun_ && !rerun_final_only_ &&
            (rerun_max_keyframes_ <= 0 ||
             (kf_id >= rerun_kf_begin &&
              kf_id < rerun_kf_begin +
                      static_cast<unsigned long>(rerun_max_keyframes_)))) {
            sv::RerunVisualizerBridge::instance().visualizeCamera(
                T_W_C,
                *track_img,
                std::vector<Eigen::Vector2f>{},
                std::vector<int>{},
                getIteration(),
                static_cast<int>(kf_id),
                fx, fy, cx, cy
            );
        }
    } catch (const c10::Error& e) {
        (void)e;
        // std::cerr << "[RERUN] Torch error in visualizeCamera: "
        //           << e.msg() << std::endl;
    } catch (const std::exception& e) {
        (void)e;
        // std::cerr << "[RERUN] Exception in visualizeCamera: "
        //           << e.what() << std::endl;
    }

    // // ─── nvblox: integrate this new keyframe into TSDF ───
    const bool need_nvblox =
        use_tsdf_mapping_ ||
        use_tsdf_pruning_ ||
        tsdf_density_init_ ||
        use_tsdf_planning_ ||
        save_nvblox_mesh_eval_ ||
        (enable_rerun_ && rerun_nvblox_mesh_ && !load_saved_nvblox_mesh_);
    if (sensor_type_ == RGBD && need_nvblox) {
        cv::Mat depth_meters;
        if (pkf->img_auxiliary_undist_.type() == CV_32FC1) {
            depth_meters = pkf->img_auxiliary_undist_;
        } else if (pkf->img_auxiliary_undist_.type() == CV_16UC1) {
            pkf->img_auxiliary_undist_.convertTo(depth_meters, CV_32FC1, 1.0 / 1000.0);
        } else {
            pkf->img_auxiliary_undist_.convertTo(depth_meters, CV_32FC1);
        }
        debugDepthStats(depth_meters, static_cast<int>(std::get<0>(kf)));
        integrateKeyframeIntoNvblox(*pkf, depth_meters);
    }
}

// -----------------------------------------------------------------------------
// 1) Transform 3D points by a 4x4 pose matrix Twc
//    points: [N,3], Twc: [4,4], post-multiplied as row vectors.
// -----------------------------------------------------------------------------
void transformPoints(torch::Tensor& points, torch::Tensor& Twc)
{
    namespace idx = torch::indexing;

    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3,
                "transformPoints: points must be [N,3]");
    TORCH_CHECK(Twc.dim() == 2 && Twc.size(0) == 4 && Twc.size(1) == 4,
                "transformPoints: Twc must be [4,4]");

    auto device = points.device();
    auto N      = points.size(0);

    auto opts = torch::TensorOptions()
                    .dtype(points.dtype())
                    .device(device);

    // Homogeneous coordinates [N,4]
    torch::Tensor ones = torch::ones({N, 1}, opts);
    torch::Tensor pts_h = torch::cat({points, ones}, /*dim=*/1);  // [N,4]

    // Row-vector convention: [N,4] * [4,4] -> [N,4]
    torch::Tensor pts_w = torch::matmul(pts_h, Twc);              // [N,4]

    // Drop homogeneous coordinate
    points = pts_w.index({idx::Slice(), idx::Slice(0, 3)}).contiguous();
}

// -----------------------------------------------------------------------------
// 2) Reproject depth map (pinhole) to 3D points in camera coordinates
//
// depth:            flattened [H*W] depth (meters) on device (CUDA or CPU)
// point_valid_flags:[H*W] bool, currently not used here (kept for signature)
// intr:             fx, fy, cx, cy
// image_width:      W
//
// Return: [H*W,3] tensor of (X,Y,Z) in camera coordinates. You can later
//          mask it with point_valid_flags (as Photo-SLAM does).
// -----------------------------------------------------------------------------
torch::Tensor reprojectDepthPinholeVoxel(
    torch::Tensor& depth,
    torch::Tensor& point_valid_flags,
    std::vector<float>& intr,
    int image_width)
{
    namespace idx = torch::indexing;

    TORCH_CHECK(depth.dim() == 1,
                "reprojectDepthPinholeVoxel: expected depth to be 1-D flattened [H*W]");
    TORCH_CHECK(intr.size() >= 4,
                "reprojectDepthPinholeVoxel: intr must contain at least {fx, fy, cx, cy}");
    TORCH_CHECK(image_width > 0,
                "reprojectDepthPinholeVoxel: image_width must be > 0");

    auto device = depth.device();
    auto opts   = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(device);

    const int64_t N = depth.size(0);
    TORCH_CHECK(N % image_width == 0,
                "reprojectDepthPinholeVoxel: depth.size(0) not divisible by image_width");
    const int64_t H = N / image_width;
    const int64_t W = image_width;

    const float fx = intr[0];
    const float fy = intr[1];
    const float cx = intr[2];
    const float cy = intr[3];

    // Build pixel grid
    torch::Tensor u = torch::arange(W, opts)            // [W]
                          .view({1, W})                 // [1,W]
                          .repeat({H, 1});              // [H,W]
    torch::Tensor v = torch::arange(H, opts)            // [H]
                          .view({H, 1})                 // [H,1]
                          .repeat({1, W});              // [H,W]

    u = u.flatten();                                    // [H*W]
    v = v.flatten();                                    // [H*W]

    // Depth
    torch::Tensor z = depth.to(opts);                   // [H*W]

    // Avoid division by zero if intrinsics are weird
    TORCH_CHECK(std::abs(fx) > 1e-8f && std::abs(fy) > 1e-8f,
                "reprojectDepthPinholeVoxel: fx/fy must be non-zero");

    torch::Tensor x = (u - cx) / fx * z;                // [H*W]
    torch::Tensor y = (v - cy) / fy * z;                // [H*W]

    // Stack to [H*W,3]
    torch::Tensor points = torch::stack({x, y, z}, /*dim=*/1);  // [H*W,3]

    // point_valid_flags is kept only for signature compatibility
    (void)point_valid_flags;

    return points.contiguous();
}

void VoxelMapper::increasePcdByKeyframeInactiveGeoDensify(
    std::shared_ptr<VoxelKeyframe> pkf)
{
    // auto start_timing = std::chrono::steady_clock::now();
    torch::NoGradGuard no_grad;

    const int iter = getIteration();
    const int64_t num_kps = static_cast<int64_t>(pkf->kps_pixel_.size() / 2);
    const int64_t cache_points_before = tensorRowCount(depth_cache_points_);
    // std::cout << "[inactive_geo_densify/start] iter=" << iter
    //           << " kf=" << pkf->fid_
    //           << " sensor=" << sensorTypeToString(sensor_type_)
    //           << " num_kps=" << num_kps
    //           << " cache_slots=" << depth_cached_ << "/" << max_depth_cached_
    //           << " cache_points=" << cache_points_before
    //           << std::endl;

    int64_t added_points = 0;

    // Pose of camera in world frame
    Sophus::SE3f Twc = pkf->getPosef().inverse();

    switch (this->sensor_type_)
    {
    case MONOCULAR:
    {
        // savePly(result_dir_ / (std::to_string(getIteration()) + "_" +
        //                       std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));

        assert(pkf->kps_pixel_.size() % 2 == 0);
        int N = pkf->kps_pixel_.size() / 2;

        // Keypoints and local 3D (camera frame)
        torch::Tensor kps_pixel_tensor = torch::from_blob(
            pkf->kps_pixel_.data(),
            {N, 2},
            torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);

        torch::Tensor kps_point_local_tensor = torch::from_blob(
            pkf->kps_point_local_.data(),
            {N, 3},
            torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);

        torch::Tensor kps_has3D_tensor = torch::where(
            kps_point_local_tensor.index({torch::indexing::Slice(), 2}) > 0.0f,
            true,
            false);
        const int64_t num_kps_with_3d = kps_has3D_tensor.sum().item<int64_t>();

        // RGB image → torch
        cv::cuda::GpuMat rgb_gpu;
        rgb_gpu.upload(pkf->img_undist_);
        torch::Tensor colors = tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_gpu);
        colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();

        // Photo-SLAM’s neighborhood densification
        auto result =
            monocularPinholeInactiveGeoDensifyBySearchingNeighborhoodKeypoints(
                kps_pixel_tensor,
                kps_has3D_tensor,
                kps_point_local_tensor,
                colors,
                monocular_inactive_geo_densify_max_pixel_dist_,
                pkf->intr_,
                pkf->image_width_);

        torch::Tensor& points3D_valid = std::get<0>(result);
        torch::Tensor& colors_valid   = std::get<1>(result);
        added_points = tensorRowCount(points3D_valid);
        const int64_t densified_missing_points =
            std::max<int64_t>(0, added_points - num_kps_with_3d);

        static bool logged_monocular_impl = false;
        if (!logged_monocular_impl) {
            // std::cout << "[inactive_geo_densify/monocular] using CUDA neighborhood depth propagation "
            //           << "from src/stereo_vision.cu"
            //           << std::endl;
            logged_monocular_impl = true;
        }
        // std::cout << "[inactive_geo_densify/monocular] iter=" << iter
        //           << " kf=" << pkf->fid_
        //           << " keypoints=" << N
        //           << " keypoints_with_orb_3d=" << num_kps_with_3d
        //           << " keypoints_without_orb_3d=" << (N - num_kps_with_3d)
        //           << " max_pixel_dist=" << monocular_inactive_geo_densify_max_pixel_dist_
        //           << " returned_points=" << added_points
        //           << " densified_missing_points=" << densified_missing_points
        //           << std::endl;

        // Transform points to world coordinates
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);

        // Add new points to the cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        } else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid},   /*dim=*/0);
        }

        // savePly(result_dir_ / (std::to_string(getIteration()) + "_" +
        //                       std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;

    case STEREO:
    {
        // savePly(result_dir_ / (std::to_string(getIteration()) + "_" +
        //                       std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));

        cv::cuda::GpuMat rgb_left_gpu, rgb_right_gpu;
        cv::cuda::GpuMat gray_left_gpu, gray_right_gpu;

        rgb_left_gpu.upload(pkf->img_undist_);
        rgb_right_gpu.upload(pkf->img_auxiliary_undist_);

        // RGB → gray
        cv::cuda::cvtColor(rgb_left_gpu,  gray_left_gpu,  cv::COLOR_RGB2GRAY);
        cv::cuda::cvtColor(rgb_right_gpu, gray_right_gpu, cv::COLOR_RGB2GRAY);

        // float → uint8
        gray_left_gpu.convertTo(gray_left_gpu,   CV_8UC1, 255.0);
        gray_right_gpu.convertTo(gray_right_gpu, CV_8UC1, 255.0);

        // Compute disparity
        cv::cuda::GpuMat cv_disp;
        stereo_cv_sgm_->compute(gray_left_gpu, gray_right_gpu, cv_disp);
        cv_disp.convertTo(cv_disp, CV_32F, 1.0 / 16.0);

        // Reproject to 3D
        cv::cuda::GpuMat cv_points3D;
        cv::cuda::reprojectImageTo3D(cv_disp, cv_points3D, stereo_Q_, 3);

        // To torch
        torch::Tensor disp = tensor_utils::cvGpuMat2TorchTensor_Float32(cv_disp);
        disp = disp.flatten(0, 1).contiguous();

        torch::Tensor points3D =
            tensor_utils::cvGpuMat2TorchTensor_Float32(cv_points3D);
        points3D = points3D.permute({1, 2, 0}).flatten(0, 1).contiguous();

        torch::Tensor colors =
            tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_left_gpu);
        colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();

        // Keep only points near tracked keypoints + valid disparity range
        torch::Tensor point_valid_flags = torch::full(
            {disp.size(0)},
            false,
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));

        int nkps_twice = pkf->kps_pixel_.size();
        int width      = pkf->image_width_;
        for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
            int idx = static_cast<int>(pkf->kps_pixel_[kpidx]) +
                      static_cast<int>(pkf->kps_pixel_[kpidx + 1]) * width;
            point_valid_flags[idx] = true;
        }

        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(
                disp > static_cast<float>(stereo_cv_sgm_->getMinDisparity()),
                true,
                false));

        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(
                disp < static_cast<float>(stereo_cv_sgm_->getNumDisparities()),
                true,
                false));
        const int64_t num_valid_pixels = point_valid_flags.sum().item<int64_t>();

        torch::Tensor points3D_valid = points3D.index({point_valid_flags});
        torch::Tensor colors_valid   = colors.index({point_valid_flags});
        added_points = tensorRowCount(points3D_valid);
        // std::cout << "[inactive_geo_densify/stereo] iter=" << iter
        //           << " kf=" << pkf->fid_
        //           << " keypoints=" << (nkps_twice / 2)
        //           << " valid_pixels_after_disp_filter=" << num_valid_pixels
        //           << " added_points=" << added_points
        //           << std::endl;

        // Transform to world
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);

        // Cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        } else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid},   /*dim=*/0);
        }

        // savePly(result_dir_ / (std::to_string(getIteration()) + "_" +
        //                       std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;

    case RGBD:
    {
        // savePly(result_dir_ / (std::to_string(getIteration()) + "_" +
        //                       std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));

        cv::cuda::GpuMat img_rgb_gpu, img_depth_gpu;
        img_rgb_gpu.upload(pkf->img_undist_);
        img_depth_gpu.upload(pkf->img_auxiliary_undist_);

        // cv::cuda::GpuMat → torch::Tensor
        torch::Tensor rgb = tensor_utils::cvGpuMat2TorchTensor_Float32(img_rgb_gpu);
        rgb = rgb.permute({1, 2, 0}).flatten(0, 1).contiguous();

        torch::Tensor depth = tensor_utils::cvGpuMat2TorchTensor_Float32(img_depth_gpu);
        depth = depth.flatten(0, 1).contiguous();

        sv::Camera& camera = scene_->cameras_.at(pkf->camera_id_);

        // Filter depth using tracked keypoints + RGBD_min/max. This preserves
        // the original sparse RGB-D insertion path.
        torch::Tensor point_valid_flags = torch::full(
            {depth.size(0)},
            false,   // Note Photo-SLAM uses false here and then sets only around kps
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));

        int nkps_twice = pkf->kps_pixel_.size();
        int width      = pkf->image_width_;
        for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
            int idx = static_cast<int>(pkf->kps_pixel_[kpidx]) +
                      static_cast<int>(pkf->kps_pixel_[kpidx + 1]) * width;
            point_valid_flags[idx] = true;
        }

        // Debug: how many pixels do we mark around kps?
        auto num_kps          = nkps_twice / 2;
        auto num_flags_before = point_valid_flags.sum().item<int64_t>();

        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(depth > RGBD_min_depth_, true, false));
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(depth < RGBD_max_depth_, true, false));
        
        torch::Tensor inactive_geo_flags = point_valid_flags.clone();
        torch::Tensor rgbd_render_hole_flags = torch::zeros_like(point_valid_flags);
        auto num_flags_after_depth = point_valid_flags.sum().item<int64_t>();

        int64_t rgbd_render_hole_pixels = 0;
        int64_t rgbd_render_hole_selected = 0;
        if (rgbd_fill_render_holes_ && voxel_model_ &&
            pkf->image_height_ > 0 && pkf->image_width_ > 0) {
            std::unordered_map<std::string, torch::Tensor> render_pkg;
            {
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                render_pkg = voxel_model_->render(
                    pkf->toMiniCam(pkf->image_height_, pkf->image_width_),
                    pkf->image_height_,
                    pkf->image_width_,
                    torch::Tensor(),
                    "dontcare",
                    false,
                    std::nullopt,
                    true,
                    false,
                    true,
                    false,
                    false,
                    sv::RenderOpts{});
            }

            torch::Tensor render_depth_cpu;
            torch::Tensor render_alpha_cpu;
            torch::Tensor render_n_contrib_cpu;
            if (renderPkgToMonoPriorAlignmentMaps(
                    render_pkg,
                    pkf->image_height_,
                    pkf->image_width_,
                    render_depth_cpu,
                    render_alpha_cpu,
                    render_n_contrib_cpu)) {
                torch::Tensor depth_cpu =
                    depth.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
                if (depth_cpu.dim() == 1) {
                    depth_cpu = depth_cpu.view({pkf->image_height_, pkf->image_width_});
                }

                std::vector<uint8_t> fill_mask(
                    static_cast<size_t>(pkf->image_height_) *
                    static_cast<size_t>(pkf->image_width_),
                    0);
                auto depth_acc = depth_cpu.accessor<float, 2>();
                auto render_depth_acc = render_depth_cpu.accessor<float, 2>();
                auto render_n_contrib_acc = render_n_contrib_cpu.accessor<int, 2>();

                const int active_hole_max_n_contrib = 0;
                const int stride = std::max(1, rgbd_fill_render_holes_stride_);
                std::vector<int64_t> selected_idx;
                selected_idx.reserve(
                    static_cast<size_t>((pkf->image_height_ + stride - 1) / stride) *
                    static_cast<size_t>((pkf->image_width_ + stride - 1) / stride));

                for (int y = 0; y < pkf->image_height_; ++y) {
                    for (int x = 0; x < pkf->image_width_; ++x) {
                        const float z_rgbd = depth_acc[y][x];
                        if (!std::isfinite(z_rgbd) ||
                            z_rgbd <= RGBD_min_depth_ ||
                            z_rgbd >= RGBD_max_depth_) {
                            continue;
                        }

                        const float z_render = render_depth_acc[y][x];
                        const int n_contrib = render_n_contrib_acc[y][x];
                        const bool render_hole =
                            (n_contrib <= active_hole_max_n_contrib) &&
                            (!std::isfinite(z_render) ||
                             z_render <= kMonoPriorFillHolesEmptyDepthEps);
                        if (!render_hole) {
                            continue;
                        }

                        ++rgbd_render_hole_pixels;
                        if ((x % stride) == 0 && (y % stride) == 0) {
                            selected_idx.push_back(
                                static_cast<int64_t>(y) *
                                static_cast<int64_t>(pkf->image_width_) +
                                static_cast<int64_t>(x));
                        }
                    }
                }

                if (rgbd_fill_render_holes_max_points_per_kf_ > 0 &&
                    static_cast<int>(selected_idx.size()) >
                        rgbd_fill_render_holes_max_points_per_kf_) {
                    std::vector<int64_t> keep;
                    keep.reserve(static_cast<size_t>(
                        rgbd_fill_render_holes_max_points_per_kf_));
                    if (rgbd_fill_render_holes_max_points_per_kf_ == 1) {
                        keep.push_back(selected_idx[selected_idx.size() / 2]);
                    } else {
                        const double step =
                            static_cast<double>(selected_idx.size() - 1) /
                            static_cast<double>(
                                rgbd_fill_render_holes_max_points_per_kf_ - 1);
                        for (int i = 0; i < rgbd_fill_render_holes_max_points_per_kf_; ++i) {
                            const size_t idx = static_cast<size_t>(
                                std::llround(step * static_cast<double>(i)));
                            keep.push_back(selected_idx[
                                std::min(idx, selected_idx.size() - 1)]);
                        }
                    }
                    selected_idx.swap(keep);
                }

                for (const int64_t idx : selected_idx) {
                    fill_mask[static_cast<size_t>(idx)] = 1;
                }
                rgbd_render_hole_selected = static_cast<int64_t>(selected_idx.size());

                if (rgbd_render_hole_selected > 0) {
                    torch::Tensor fill_flags =
                        torch::from_blob(
                            fill_mask.data(),
                            {depth.size(0)},
                            torch::TensorOptions()
                                .dtype(torch::kUInt8)
                                .device(torch::kCPU))
                            .clone()
                            .to(device_type_)
                            .to(torch::kBool);
                    rgbd_render_hole_flags =
                        (fill_flags & (~inactive_geo_flags)).to(torch::kBool);
                    point_valid_flags =
                        (inactive_geo_flags | rgbd_render_hole_flags).to(torch::kBool);
                    rgbd_render_hole_selected =
                        rgbd_render_hole_flags.sum().item<int64_t>();
                }
            }
        }

        const int64_t num_flags_after_rgbd_holes =
            point_valid_flags.sum().item<int64_t>();

        // Reproject to 3D (camera coordinates)
        torch::Tensor points3D_all;

        switch (camera.model_id_)
        {
        case Camera::PINHOLE:
        {
            points3D_all = reprojectDepthPinholeVoxel(
                depth,
                point_valid_flags,
                pkf->intr_,
                pkf->image_width_);
        }
        break;

        case Camera::FISHEYE:
        {
            // TODO: support fisheye camera?
            throw std::runtime_error("[VoxelMapper] Fisheye cameras are not supported currently!");
        }
        break;

        default:
        {
            throw std::runtime_error("[VoxelMapper] Invalid camera model!");
        }
        break;
        }

        torch::Tensor points3D_inactive_geo = points3D_all.index({inactive_geo_flags});
        torch::Tensor colors_inactive_geo = rgb.index({inactive_geo_flags});
        torch::Tensor points3D_rgbd_holes = points3D_all.index({rgbd_render_hole_flags});
        torch::Tensor colors_rgbd_holes = rgb.index({rgbd_render_hole_flags});
        const int64_t inactive_geo_points = tensorRowCount(points3D_inactive_geo);
        const int64_t rgbd_hole_points = tensorRowCount(points3D_rgbd_holes);
        added_points = inactive_geo_points + rgbd_hole_points;
        std::cout << "[inactive_geo_densify/rgbd] iter=" << iter
                  << " kf=" << pkf->fid_
                  << " keypoints=" << num_kps
                  << " valid_pixels_after_kps=" << num_flags_before
                  << " valid_pixels_after_depth_filter=" << num_flags_after_depth
                  << " render_hole_pixels=" << rgbd_render_hole_pixels
                  << " render_hole_selected=" << rgbd_render_hole_selected
                  << " valid_pixels_after_rgbd_holes=" << num_flags_after_rgbd_holes
                  << " inactive_geo_points=" << inactive_geo_points
                  << " rgbd_hole_points=" << rgbd_hole_points
                  << " added_points=" << added_points
                  << " rgbd_hole_stride=" << rgbd_fill_render_holes_stride_
                  << " depth_range=[" << RGBD_min_depth_ << "," << RGBD_max_depth_ << "]"
                  << std::endl;

        // Transform to world coordinates
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_inactive_geo, Twc_tensor);
        transformPoints(points3D_rgbd_holes, Twc_tensor);

        // Cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_inactive_geo;
            depth_cache_colors_ = colors_inactive_geo;
            rgbd_fill_render_holes_cache_points_ = points3D_rgbd_holes;
            rgbd_fill_render_holes_cache_colors_ = colors_rgbd_holes;
        } else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_inactive_geo}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_inactive_geo},   /*dim=*/0);
            rgbd_fill_render_holes_cache_points_ =
                torch::cat({rgbd_fill_render_holes_cache_points_, points3D_rgbd_holes}, /*dim=*/0);
            rgbd_fill_render_holes_cache_colors_ =
                torch::cat({rgbd_fill_render_holes_cache_colors_, colors_rgbd_holes},   /*dim=*/0);
        }

        // savePly(result_dir_ / (std::to_string(getIteration()) + "_" +
        //                       std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;

    default:
    {
        throw std::runtime_error("[VoxelMapper] Unsupported sensor type!");
    }
    break;
    }

    pkf->done_inactive_geo_densify_ = true;
    const int next_depth_cached = depth_cached_ + 1;
    const int64_t cache_points_after =
        tensorRowCount(depth_cache_points_) +
        tensorRowCount(rgbd_fill_render_holes_cache_points_);
    const bool will_flush_cache = next_depth_cached >= max_depth_cached_;
    // std::cout << "[inactive_geo_densify/cache] iter=" << iter
    //           << " kf=" << pkf->fid_
    //           << " added_points=" << added_points
    //           << " cache_slots=" << next_depth_cached << "/" << max_depth_cached_
    //           << " cache_points=" << cache_points_after
    //           << " flush=" << (will_flush_cache ? 1 : 0)
    //           << std::endl;
    ++depth_cached_;

    if (depth_cached_ >= max_depth_cached_) {
        depth_cached_ = 0;

        // Add new points to the voxel model
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        std::vector<sv::MiniCam> tr_cams;
        tr_cams.reserve(scene_->keyframes().size());
        for (auto& kv : scene_->keyframes()) {
            if (kv.second) {
                tr_cams.push_back(
                    kv.second->toMiniCam(kv.second->image_height_, kv.second->image_width_));
            }
        }
        auto flush_real_cache =
            [&](torch::Tensor& cache_points,
                torch::Tensor& cache_colors,
                const std::string& entity_path)
        {
            if (!cache_points.defined() || cache_points.dim() != 2 || cache_points.size(0) <= 0) {
                return;
            }
            const bool log_created_voxels = enable_rerun_ && !rerun_final_only_;
            if (log_created_voxels) {
                voxel_model_->setNextRealInsertionRerunEntityPath(entity_path);
            }
            voxel_model_->increasePcd(
                cache_points,
                cache_colors,
                getIteration(),
                tr_cams);
            const sv::VoxelModel::IncreasePcdStats insert_stats =
                voxel_model_->lastIncreasePcdStats();
            if (entity_path == "world/voxels_inactive_geo_densify/created") {
                tsdf_ablation_inactive_geo_created_ += insert_stats.new_voxels;
                tsdf_ablation_inactive_geo_lineage_created_ += insert_stats.new_voxels;
            } else if (entity_path == "world/rgbd_fill_render_holes/created") {
                tsdf_ablation_rgbd_fill_created_ += insert_stats.new_voxels;
                tsdf_ablation_rgbd_fill_lineage_created_ += insert_stats.new_voxels;
            }
            if (log_created_voxels) {
                voxel_model_->setNextRealInsertionRerunEntityPath("");
            }
            cache_points = torch::Tensor();
            cache_colors = torch::Tensor();
        };

        if ((depth_cache_points_.defined() && depth_cache_points_.dim() == 2 && depth_cache_points_.size(0) > 0) ||
            (rgbd_fill_render_holes_cache_points_.defined() &&
             rgbd_fill_render_holes_cache_points_.dim() == 2 &&
             rgbd_fill_render_holes_cache_points_.size(0) > 0)) {
            // std::cout << "[inactive_geo_densify/flush] iter=" << iter
            //           << " kf=" << pkf->fid_
            //           << " flushing_points=" << depth_cache_points_.size(0)
            //           << " cameras_for_filtering=" << tr_cams.size()
            //           << std::endl;
            flush_real_cache(
                depth_cache_points_,
                depth_cache_colors_,
                "world/voxels_inactive_geo_densify/created");
            flush_real_cache(
                rgbd_fill_render_holes_cache_points_,
                rgbd_fill_render_holes_cache_colors_,
                "world/rgbd_fill_render_holes/created");
        } else {
            std::cout << "[inactive_geo_densify/flush] iter=" << iter
                      << " kf=" << pkf->fid_
                      << " no cached points to insert"
                      << std::endl;
        }
    }

    // auto end_timing = std::chrono::steady_clock::now();
    // auto completion_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    //     end_timing - start_timing).count();
    // std::cout << "[VoxelMapper] increasePcdByKeyframeInactiveGeoDensify() takes "
    //           << completion_time << " ms" << std::endl;
}

void VoxelMapper::accumulateMonoPriorGlobalAlignment(
    float scale,
    float shift,
    float weight)
{
    if (!std::isfinite(scale) || !std::isfinite(shift) || scale <= 0.0f) {
        return;
    }

    weight = std::max(1.0f, weight);
    if (!depthanything_global_alignment_valid_ ||
        !std::isfinite(depthanything_global_align_scale_) ||
        !std::isfinite(depthanything_global_align_shift_) ||
        depthanything_global_align_weight_ <= 0.0f) {
        depthanything_global_align_scale_ = scale;
        depthanything_global_align_shift_ = shift;
        depthanything_global_align_weight_ = weight;
        depthanything_global_align_observations_ = 1;
        depthanything_global_alignment_valid_ = true;
        return;
    }

    const float old_weight = depthanything_global_align_weight_;
    const float new_weight = old_weight + weight;
    depthanything_global_align_scale_ =
        (depthanything_global_align_scale_ * old_weight + scale * weight) / new_weight;
    depthanything_global_align_shift_ =
        (depthanything_global_align_shift_ * old_weight + shift * weight) / new_weight;
    depthanything_global_align_weight_ = new_weight;
    ++depthanything_global_align_observations_;
}

bool VoxelMapper::updateMonoPriorGlobalAlignmentFromKeyframe(
    const std::shared_ptr<VoxelKeyframe>& pkf)
{
    torch::NoGradGuard no_grad;

    if (!pkf || sensor_type_ != MONOCULAR) {
        return false;
    }

    const int H = pkf->image_height_;
    const int W = pkf->image_width_;
    if (H <= 0 || W <= 0) {
        return false;
    }
    if (!ensureMonoPriorForKeyframe(pkf) ||
        !pkf->mono_prior_.defined() ||
        pkf->mono_prior_.numel() == 0) {
        return false;
    }

    torch::Tensor mono_prior =
        pkf->mono_prior_.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (mono_prior.dim() == 3 && mono_prior.size(0) == 1) {
        mono_prior = mono_prior.squeeze(0);
    }
    if (mono_prior.dim() != 2) {
        return false;
    }
    if (mono_prior.size(0) != H || mono_prior.size(1) != W) {
        mono_prior = torch::nn::functional::interpolate(
            mono_prior.unsqueeze(0).unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{H, W})
                .mode(torch::kBilinear)
                .align_corners(false)).squeeze().to(torch::kCPU).contiguous();
    }

    torch::Tensor sparse_uv;
    torch::Tensor sparse_depth;
    if (!buildSparseDepthFromKeyframeOrbAnchors(pkf, W, H, sparse_uv, sparse_depth)) {
        // std::cout << "[mono_prior_align/global_seed] iter=" << getIteration()
        //           << " kf=" << pkf->fid_
        //           << " no valid ORB anchors"
        //           << std::endl;
        return false;
    }

    sv::MiniCam cam = pkf->toMiniCam(H, W);
    const bool use_metric3d = isMetric3DModelId(mono_prior_model_id_);
    torch::Tensor aligned_depth;
    MonoPriorAlignmentStats stats;
    const bool aligned_ok = use_metric3d
        ? alignMetricDepthPriorToSparseAnchors(
              mono_prior,
              sparse_uv,
              sparse_depth,
              cam.near,
              depthanything_densify_min_sparse_anchors_,
              aligned_depth,
              stats)
        : alignDepthAnythingPriorToSparseAnchors(
              mono_prior,
              sparse_uv,
              sparse_depth,
              cam.near,
              depthanything_densify_min_sparse_anchors_,
              aligned_depth,
              stats);
    if (!aligned_ok) {
        // std::cout << "[mono_prior_align/global_seed] iter=" << getIteration()
        //           << " kf=" << pkf->fid_
        //           << " failed num_valid_anchors=" << stats.num_valid_anchors
        //           << " fit_anchors=" << stats.num_fit_anchors
        //           << " scale=" << stats.scale
        //           << " shift=" << stats.shift
        //           << " med_rel_err=" << stats.median_rel_depth_error
        //           << " p90_rel_err=" << stats.p90_rel_depth_error
        //           << " inlier_ratio=" << stats.final_inlier_ratio
        //           << std::endl;
        return false;
    }

    accumulateMonoPriorGlobalAlignment(
        stats.scale,
        stats.shift,
        static_cast<float>(std::max<int64_t>(1, stats.num_fit_anchors)));
    // std::cout << "[mono_prior_align/global_seed] iter=" << getIteration()
    //           << " kf=" << pkf->fid_
    //           << " accepted scale=" << stats.scale
    //           << " shift=" << stats.shift
    //           << " med_rel_err=" << stats.median_rel_depth_error
    //           << " p90_rel_err=" << stats.p90_rel_depth_error
    //           << " inlier_ratio=" << stats.final_inlier_ratio
    //           << " global_scale=" << depthanything_global_align_scale_
    //           << " global_shift=" << depthanything_global_align_shift_
    //           << " global_obs=" << depthanything_global_align_observations_
    //           << std::endl;
    return true;
}

void VoxelMapper::increasePcdByKeyframeMonoPrior(
    std::shared_ptr<VoxelKeyframe> pkf)
{
    torch::NoGradGuard no_grad;

    if (!depthanything_densify_ || !pkf || !voxel_model_) {
        return;
    }

    const int iter = getIteration();
    if (sensor_type_ != MONOCULAR) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    const int H = pkf->image_height_;
    const int W = pkf->image_width_;
    if (H <= 0 || W <= 0) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    if (!ensureMonoPriorForKeyframe(pkf) ||
        !pkf->mono_prior_.defined() ||
        pkf->mono_prior_.numel() == 0) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    sv::MiniCam cam = pkf->toMiniCam(H, W);

    auto finiteTensorStats = [](const torch::Tensor& t,
                                int64_t* count_out,
                                float* min_out,
                                float* max_out,
                                float* mean_out) {
        *count_out = 0;
        *min_out = std::numeric_limits<float>::quiet_NaN();
        *max_out = std::numeric_limits<float>::quiet_NaN();
        *mean_out = std::numeric_limits<float>::quiet_NaN();
        if (!t.defined() || t.numel() == 0) {
            return;
        }
        torch::Tensor finite = torch::masked_select(
            t.to(torch::kCPU).to(torch::kFloat32).contiguous(),
            torch::isfinite(t.to(torch::kCPU).to(torch::kFloat32).contiguous()));
        if (!finite.defined() || finite.numel() == 0) {
            return;
        }
        *count_out = finite.numel();
        *min_out = finite.min().item<float>();
        *max_out = finite.max().item<float>();
        *mean_out = finite.mean().item<float>();
    };

    torch::Tensor mono_prior = pkf->mono_prior_.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (mono_prior.dim() == 3 && mono_prior.size(0) == 1) {
        mono_prior = mono_prior.squeeze(0);
    }
    if (mono_prior.dim() != 2) {
        // std::cout << "[mono_prior_densify/start] iter=" << iter
        //           << " kf=" << pkf->fid_
        //           << " invalid prior shape=" << mono_prior.sizes()
        //           << std::endl;
        updateRenderedDepthCandidateLifecycle();
        return;
    }
    if (mono_prior.size(0) != H || mono_prior.size(1) != W) {
        mono_prior = torch::nn::functional::interpolate(
            mono_prior.unsqueeze(0).unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{H, W})
                .mode(torch::kBilinear)
                .align_corners(false)).squeeze().to(torch::kCPU).contiguous();
    }

    const bool use_metric3d = isMetric3DModelId(mono_prior_model_id_);
    torch::Tensor da_prior_query_mask;
    if (depthanything_densify_alignment_mode_ == "da_prior") {
        da_prior_query_mask = torch::zeros(
            {H, W},
            torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
        auto qmask_acc = da_prior_query_mask.accessor<bool, 2>();
        const int stride = std::max(1, depthanything_densify_stride_);
        for (int y = 0; y < H; y += stride) {
            for (int x = 0; x < W; x += stride) {
                qmask_acc[y][x] = true;
            }
        }
    }
    torch::Tensor sparse_uv;
    torch::Tensor sparse_depth;
    torch::Tensor aligned_depth;
    MonoPriorAlignmentStats align_stats;
    bool used_rendered_alignment = false;
    bool used_da_prior_alignment = false;
    bool used_global_alignment = false;
    bool alignment_ok = false;
    torch::Tensor rendered_align_depth_cpu;
    torch::Tensor rendered_align_alpha_cpu;
    torch::Tensor rendered_align_n_contrib_cpu;
    if (depthanything_densify_alignment_mode_ == "rendered") {
        std::unordered_map<std::string, torch::Tensor> render_pkg;
        {
            std::unique_lock<std::mutex> lock_render(mutex_render_);
            render_pkg = voxel_model_->render(
                cam,
                H,
                W,
                torch::Tensor(),
                "dontcare",
                false,
                std::nullopt,
                true,
                false,
                true,
                false,
                false,
                sv::RenderOpts{});
        }
        if (renderPkgToMonoPriorAlignmentMaps(
                render_pkg,
                H,
                W,
                rendered_align_depth_cpu,
                rendered_align_alpha_cpu,
                rendered_align_n_contrib_cpu)) {
            alignment_ok = use_metric3d
                ? alignMetricDepthPriorToRenderedDepth(
                      mono_prior,
                      rendered_align_depth_cpu,
                      rendered_align_alpha_cpu,
                      rendered_align_n_contrib_cpu,
                      cam.near,
                      depthanything_densify_min_sparse_anchors_,
                      aligned_depth,
                      align_stats)
                : alignDepthAnythingPriorToRenderedDepth(
                      mono_prior,
                      rendered_align_depth_cpu,
                      rendered_align_alpha_cpu,
                      rendered_align_n_contrib_cpu,
                      cam.near,
                      depthanything_densify_min_sparse_anchors_,
                      aligned_depth,
                      align_stats);
            used_rendered_alignment = alignment_ok;
        }
    }
    const bool has_sparse_anchors =
        buildSparseDepthFromKeyframeOrbAnchors(pkf, W, H, sparse_uv, sparse_depth);
    if (!alignment_ok && has_sparse_anchors) {
        if (depthanything_densify_alignment_mode_ == "da_prior") {
            alignment_ok = alignMonoPriorToSparseAnchorsDaPrior(
                mono_prior,
                sparse_uv,
                sparse_depth,
                da_prior_query_mask,
                use_metric3d,
                cam.near,
                depthanything_densify_min_sparse_anchors_,
                depthanything_da_prior_knn_k_,
                depthanything_da_prior_distance_weighting_,
                depthanything_da_prior_max_pixel_dist_,
                aligned_depth,
                align_stats);
            used_da_prior_alignment = alignment_ok;
        }
        if (!alignment_ok) {
            alignment_ok = use_metric3d
                ? alignMetricDepthPriorToSparseAnchors(
                      mono_prior,
                      sparse_uv,
                      sparse_depth,
                      cam.near,
                      depthanything_densify_min_sparse_anchors_,
                      aligned_depth,
                      align_stats)
                : alignDepthAnythingPriorToSparseAnchors(
                      mono_prior,
                      sparse_uv,
                      sparse_depth,
                      cam.near,
                      depthanything_densify_min_sparse_anchors_,
                      aligned_depth,
                      align_stats);
        }
        if (alignment_ok) {
            accumulateMonoPriorGlobalAlignment(
                align_stats.scale,
                align_stats.shift,
                static_cast<float>(std::max<int64_t>(1, align_stats.num_fit_anchors)));
        }
    }
    if (!alignment_ok) {
        if (depthanything_global_alignment_valid_ &&
            ((use_metric3d &&
              applyMetricDepthAlignment(
                  mono_prior,
                  depthanything_global_align_scale_,
                  depthanything_global_align_shift_,
                  aligned_depth)) ||
             (!use_metric3d &&
              applyDepthAnythingAffineAlignment(
                  mono_prior,
                  depthanything_global_align_scale_,
                  depthanything_global_align_shift_,
                  aligned_depth)))) {
            used_global_alignment = true;
            align_stats.scale = depthanything_global_align_scale_;
            align_stats.shift = depthanything_global_align_shift_;
        } else {
            std::cout << "[mono_prior_alignment] path=densify"
                      << " iter=" << iter
                      << " kf=" << pkf->fid_
                      << " requested=" << depthanything_densify_alignment_mode_
                      << " used=none"
                      << " fallback=0"
                      << " failed=1"
                      << " has_sparse_anchors=" << (has_sparse_anchors ? 1 : 0)
                      << " anchors=" << align_stats.num_valid_anchors
                      << " fit_anchors=" << align_stats.num_fit_anchors
                      << " min_required=" << depthanything_densify_min_sparse_anchors_
                      << " global_valid=" << (depthanything_global_alignment_valid_ ? 1 : 0)
                      << std::endl;
            updateRenderedDepthCandidateLifecycle();
            return;
        }
    }
    const char* densify_alignment_source =
        used_rendered_alignment ? "rendered" :
        (used_da_prior_alignment ? "da_prior" :
         (used_global_alignment ? "global" : "orb"));
    const bool densify_alignment_fallback =
        depthanything_densify_alignment_mode_ != densify_alignment_source;
    std::cout << "[mono_prior_alignment] path=densify"
              << " iter=" << iter
              << " kf=" << pkf->fid_
              << " requested=" << depthanything_densify_alignment_mode_
              << " used=" << densify_alignment_source
              << " fallback=" << (densify_alignment_fallback ? 1 : 0)
              << " anchors=" << align_stats.num_valid_anchors
              << " fit_anchors=" << align_stats.num_fit_anchors
              << " scale=" << align_stats.scale
              << " shift=" << align_stats.shift
              << " med_rel_err=" << align_stats.median_rel_depth_error
              << " p90_rel_err=" << align_stats.p90_rel_depth_error
              << " inlier_ratio=" << align_stats.final_inlier_ratio
              << std::endl;
    int64_t prior_finite_count = 0;
    int64_t aligned_finite_count = 0;
    float prior_min = std::numeric_limits<float>::quiet_NaN();
    float prior_max = std::numeric_limits<float>::quiet_NaN();
    float prior_mean = std::numeric_limits<float>::quiet_NaN();
    float aligned_min = std::numeric_limits<float>::quiet_NaN();
    float aligned_max = std::numeric_limits<float>::quiet_NaN();
    float aligned_mean = std::numeric_limits<float>::quiet_NaN();
    finiteTensorStats(mono_prior, &prior_finite_count, &prior_min, &prior_max, &prior_mean);
    finiteTensorStats(aligned_depth, &aligned_finite_count, &aligned_min, &aligned_max, &aligned_mean);

    auto image_cpu = pkf->original_image_.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (image_cpu.dim() == 4 && image_cpu.size(0) == 1) image_cpu = image_cpu.squeeze(0);
    if (image_cpu.dim() != 3 || image_cpu.size(0) < 3 ||
        image_cpu.size(1) != H || image_cpu.size(2) != W) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }
    if (image_cpu.size(0) > 3) {
        image_cpu = image_cpu.index({torch::indexing::Slice(0, 3)}).contiguous();
    }

    torch::Tensor valid_mask =
        torch::isfinite(aligned_depth) &
        (aligned_depth > std::max(cam.near, 1e-4f));
    if (std::isfinite(RGBD_max_depth_) && RGBD_max_depth_ > 0.0f) {
        valid_mask &= (aligned_depth < RGBD_max_depth_);
    }

    auto mask_it = undistort_mask_.find(pkf->camera_id_);
    if (mask_it != undistort_mask_.end() && mask_it->second.defined()) {
        torch::Tensor eval_mask = mask_it->second.detach().to(torch::kCPU).to(torch::kFloat32);
        if (eval_mask.dim() == 4 && eval_mask.size(0) == 1) {
            eval_mask = eval_mask.squeeze(0);
        }
        if (eval_mask.dim() == 3) {
            eval_mask = eval_mask.index({0});
        }
        if (eval_mask.dim() == 2 &&
            eval_mask.size(0) == H &&
            eval_mask.size(1) == W) {
            valid_mask &= (eval_mask > 0.5f);
        }
    }
    const int64_t valid_pixels_after_mask = valid_mask.sum().item<int64_t>();

    const float fx = (cam.fx > 1e-6f)
        ? cam.fx
        : (0.5f * static_cast<float>(W) / std::max(cam.tanfovx, 1e-6f));
    const float fy = (cam.fy > 1e-6f)
        ? cam.fy
        : (0.5f * static_cast<float>(H) / std::max(cam.tanfovy, 1e-6f));
    if (fx <= 1e-6f || fy <= 1e-6f) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    const int stride = std::max(1, depthanything_densify_stride_);
    std::vector<int64_t> selected_idx;
    selected_idx.reserve(
        static_cast<size_t>((H + stride - 1) / stride) *
        static_cast<size_t>((W + stride - 1) / stride));
    auto valid_acc = valid_mask.accessor<bool, 2>();
    for (int y = 0; y < H; y += stride) {
        for (int x = 0; x < W; x += stride) {
            if (valid_acc[y][x]) {
                selected_idx.push_back(static_cast<int64_t>(y) * static_cast<int64_t>(W) + x);
            }
        }
    }

    // std::cout << "[mono_prior_densify/start] iter=" << iter
    //           << " kf=" << pkf->fid_
    //           << " anchors=" << align_stats.num_valid_anchors
    //           << " fit_anchors=" << align_stats.num_fit_anchors
    //           << " prior_finite=" << prior_finite_count
    //           << " prior_range=[" << prior_min << "," << prior_max << "]"
    //           << " prior_mean=" << prior_mean
    //           << " aligned_finite=" << aligned_finite_count
    //           << " aligned_range=[" << aligned_min << "," << aligned_max << "]"
    //           << " aligned_mean=" << aligned_mean
    //           << " valid_pixels=" << valid_pixels_after_mask
    //           << " selected_pixels=" << selected_idx.size()
    //           << " stride=" << stride
    //           << " alignment_source="
    //           << (used_rendered_alignment ? "rendered" :
    //               (used_global_alignment ? "global" : "orb"))
    //           << " align_scale=" << align_stats.scale
    //           << " align_shift=" << align_stats.shift
    //           << " med_rel_err=" << align_stats.median_rel_depth_error
    //           << " p90_rel_err=" << align_stats.p90_rel_depth_error
    //           << " inlier_ratio=" << align_stats.final_inlier_ratio
    //           << " global_scale=" << depthanything_global_align_scale_
    //           << " global_shift=" << depthanything_global_align_shift_
    //           << " global_obs=" << depthanything_global_align_observations_
    //           << " anchor_depth_stats=[" << align_stats.sparse_depth_q05
    //           << "," << align_stats.sparse_depth_q95 << "]"
    //           << std::endl;

    if (selected_idx.empty()) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    const Sophus::SE3f Twc = pkf->getPosef().inverse();
    auto depth_acc = aligned_depth.accessor<float, 2>();
    auto image_acc = image_cpu.accessor<float, 3>();

    std::vector<float> candidate_points_world;
    std::vector<float> candidate_colors;
    candidate_points_world.reserve(selected_idx.size() * 3);
    candidate_colors.reserve(selected_idx.size() * 3);
    float candidate_depth_min = std::numeric_limits<float>::infinity();
    float candidate_depth_max = 0.0f;
    Eigen::Vector3f candidate_world_min(
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity());
    Eigen::Vector3f candidate_world_max(
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity());
    std::vector<std::string> sample_candidates;
    sample_candidates.reserve(3);

    for (const int64_t flat_idx : selected_idx) {
        const int y = static_cast<int>(flat_idx / W);
        const int x = static_cast<int>(flat_idx % W);
        const float z = depth_acc[y][x];
        if (!std::isfinite(z) || z <= 0.0f) {
            continue;
        }

        const float Xc = (static_cast<float>(x) - cam.cx) / fx * z;
        const float Yc = (static_cast<float>(y) - cam.cy) / fy * z;
        const Eigen::Vector3f p_cam(Xc, Yc, z);
        const Eigen::Vector3f p_world = Twc * p_cam;
        if (!std::isfinite(p_world.x()) ||
            !std::isfinite(p_world.y()) ||
            !std::isfinite(p_world.z())) {
            continue;
        }

        candidate_points_world.push_back(p_world.x());
        candidate_points_world.push_back(p_world.y());
        candidate_points_world.push_back(p_world.z());
        candidate_colors.push_back(std::clamp(image_acc[0][y][x], 0.0f, 1.0f));
        candidate_colors.push_back(std::clamp(image_acc[1][y][x], 0.0f, 1.0f));
        candidate_colors.push_back(std::clamp(image_acc[2][y][x], 0.0f, 1.0f));
        candidate_depth_min = std::min(candidate_depth_min, z);
        candidate_depth_max = std::max(candidate_depth_max, z);
        candidate_world_min = candidate_world_min.cwiseMin(p_world);
        candidate_world_max = candidate_world_max.cwiseMax(p_world);
        if (sample_candidates.size() < 3) {
            std::ostringstream oss;
            oss << "(u=" << x
                << ",v=" << y
                << ",z=" << z
                << ",pw=[" << p_world.x() << "," << p_world.y() << "," << p_world.z() << "])";
            sample_candidates.push_back(oss.str());
        }
    }

    const int64_t num_candidates = static_cast<int64_t>(candidate_points_world.size() / 3);
    // std::cout << "[mono_prior_densify/candidates] iter=" << iter
    //           << " kf=" << pkf->fid_
    //           << " valid_candidates=" << num_candidates
    //           << " depth_range=["
    //           << (num_candidates > 0 ? candidate_depth_min : std::numeric_limits<float>::quiet_NaN())
    //           << ","
    //           << (num_candidates > 0 ? candidate_depth_max : std::numeric_limits<float>::quiet_NaN())
    //           << "]"
    //           << " world_bbox_min=["
    //           << (num_candidates > 0 ? candidate_world_min.x() : std::numeric_limits<float>::quiet_NaN()) << ","
    //           << (num_candidates > 0 ? candidate_world_min.y() : std::numeric_limits<float>::quiet_NaN()) << ","
    //           << (num_candidates > 0 ? candidate_world_min.z() : std::numeric_limits<float>::quiet_NaN()) << "]"
    //           << " world_bbox_max=["
    //           << (num_candidates > 0 ? candidate_world_max.x() : std::numeric_limits<float>::quiet_NaN()) << ","
    //           << (num_candidates > 0 ? candidate_world_max.y() : std::numeric_limits<float>::quiet_NaN()) << ","
    //           << (num_candidates > 0 ? candidate_world_max.z() : std::numeric_limits<float>::quiet_NaN()) << "]"
    //           << std::endl;
    // for (size_t i = 0; i < sample_candidates.size(); ++i) {
    //     std::cout << "[mono_prior_densify/sample] iter=" << iter
    //               << " kf=" << pkf->fid_
    //               << " idx=" << i
    //               << " " << sample_candidates[i]
    //               << std::endl;
    // }

    if (num_candidates > 0) {
        auto points_tensor = torch::from_blob(
            candidate_points_world.data(),
            {num_candidates, 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone().to(device_type_);
        auto colors_tensor = torch::from_blob(
            candidate_colors.data(),
            {num_candidates, 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone().to(device_type_);

        std::vector<sv::MiniCam> tr_cams;
        tr_cams.reserve(scene_->keyframes().size());
        for (auto& kv : scene_->keyframes()) {
            if (kv.second) {
                tr_cams.push_back(
                    kv.second->toMiniCam(kv.second->image_height_, kv.second->image_width_));
            }
        }

        std::unique_lock<std::mutex> lock_render(mutex_render_);
        const bool log_depthanything_created_voxels =
            enable_rerun_ && !rerun_final_only_;
        if (log_depthanything_created_voxels) {
            voxel_model_->setNextRealInsertionRerunEntityPath(
                "world/mono_prior_densify/created");
        }
        voxel_model_->setRenderedDepthCandidateRealAdjacency(
            false,
            1);
        voxel_model_->setNextRenderedDepthCandidateInsertion(
            true,
            "",
            kRenderedCandidateSourceMonoPrior,
            /*insert_as_real_protected=*/true);
        voxel_model_->increasePcd(points_tensor, colors_tensor, iter, tr_cams);
        if (pkf->mono_prior_first_apply_iter_ < 0) {
            pkf->mono_prior_first_apply_iter_ = iter;
        }
        voxel_model_->setNextRenderedDepthCandidateInsertion(false);
        if (log_depthanything_created_voxels) {
            voxel_model_->setNextRealInsertionRerunEntityPath("");
        }
        const sv::VoxelModel::IncreasePcdStats insert_stats = voxel_model_->lastIncreasePcdStats();
        // std::cout << "[mono_prior_densify/insert] iter=" << iter
        //           << " kf=" << pkf->fid_
        //           << " raw_points_in=" << insert_stats.raw_points_in
        //           << " points_after_far_filter=" << insert_stats.points_after_far_filter
        //           << " unique_before_filter=" << insert_stats.unique_voxel_candidates_before_insert_filter
        //           << " unique_after_filter=" << insert_stats.unique_voxel_candidates_after_insert_filter
        //           << " duplicate_existing_voxels=" << insert_stats.duplicate_existing_voxels
        //           << " new_voxels=" << insert_stats.new_voxels
        //           << " pending_promotions=" << insert_stats.pending_promotions
        //           << " pending_support_updates=" << insert_stats.pending_support_updates
        //           << std::endl;
    }

    updateRenderedDepthCandidateLifecycle();
}

void VoxelMapper::increasePcdByKeyframeMonoPriorFillHoles(
    std::shared_ptr<VoxelKeyframe> pkf)
{
    torch::NoGradGuard no_grad;

    if (!depthanything_fill_holes_ || !pkf || !voxel_model_) {
        return;
    }

    const int iter = getIteration();
    if (sensor_type_ != MONOCULAR) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    const int H = pkf->image_height_;
    const int W = pkf->image_width_;
    if (H <= 0 || W <= 0) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    sv::MiniCam cam = pkf->toMiniCam(H, W);

    auto finiteTensorStats = [](const torch::Tensor& t,
                                int64_t* count_out,
                                float* min_out,
                                float* max_out,
                                float* mean_out) {
        *count_out = 0;
        *min_out = std::numeric_limits<float>::quiet_NaN();
        *max_out = std::numeric_limits<float>::quiet_NaN();
        *mean_out = std::numeric_limits<float>::quiet_NaN();
        if (!t.defined() || t.numel() == 0) {
            return;
        }
        torch::Tensor cpu = t.to(torch::kCPU).to(torch::kFloat32).contiguous();
        torch::Tensor finite = torch::masked_select(cpu, torch::isfinite(cpu));
        if (!finite.defined() || finite.numel() == 0) {
            return;
        }
        *count_out = finite.numel();
        *min_out = finite.min().item<float>();
        *max_out = finite.max().item<float>();
        *mean_out = finite.mean().item<float>();
    };

    std::unordered_map<std::string, torch::Tensor> render_pkg;
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        render_pkg = voxel_model_->render(
            cam,
            H,
            W,
            torch::Tensor(),
            "dontcare",
            false,
            std::nullopt,
            true,
            false,
            true,
            false,
            false,
            sv::RenderOpts{});
    }
    torch::Tensor depth_cpu;
    torch::Tensor alpha_cpu;
    torch::Tensor n_contrib_cpu;
    if (!renderPkgToMonoPriorAlignmentMaps(
            render_pkg,
            H,
            W,
            depth_cpu,
            alpha_cpu,
            n_contrib_cpu)) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    if (!ensureMonoPriorForKeyframe(pkf) ||
        !pkf->mono_prior_.defined() ||
        pkf->mono_prior_.numel() == 0) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    torch::Tensor mono_prior =
        pkf->mono_prior_.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (mono_prior.dim() == 3 && mono_prior.size(0) == 1) {
        mono_prior = mono_prior.squeeze(0);
    }
    if (mono_prior.dim() != 2) {
        // std::cout << "[mono_prior_fill_holes/start] iter=" << iter
        //           << " kf=" << pkf->fid_
        //           << " invalid prior shape=" << mono_prior.sizes()
        //           << std::endl;
        updateRenderedDepthCandidateLifecycle();
        return;
    }
    if (mono_prior.size(0) != H || mono_prior.size(1) != W) {
        mono_prior = torch::nn::functional::interpolate(
                        mono_prior.unsqueeze(0).unsqueeze(0),
                        torch::nn::functional::InterpolateFuncOptions()
                            .size(std::vector<int64_t>{H, W})
                            .mode(torch::kBilinear)
                            .align_corners(false))
                         .squeeze()
                         .to(torch::kCPU)
                         .contiguous();
    }

    const bool use_metric3d = isMetric3DModelId(mono_prior_model_id_);

    auto depth_acc = depth_cpu.accessor<float, 2>();
    auto n_contrib_acc = n_contrib_cpu.accessor<int, 2>();

    const int active_hole_max_n_contrib = 0;
    std::vector<uint8_t> hole_mask(static_cast<size_t>(H) * static_cast<size_t>(W), 0);
    auto flat = [W](int y, int x) -> int64_t {
        return static_cast<int64_t>(y) * static_cast<int64_t>(W) + static_cast<int64_t>(x);
    };

    int64_t hole_pixels = 0;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const float z = depth_acc[y][x];
            const int n_contrib = n_contrib_acc[y][x];
            const bool is_hole =
                (n_contrib <= active_hole_max_n_contrib) &&
                (!std::isfinite(z) || z <= kMonoPriorFillHolesEmptyDepthEps);
            if (is_hole) {
                hole_mask[static_cast<size_t>(flat(y, x))] = 1;
                ++hole_pixels;
            }
        }
    }
    if (hole_pixels <= 0) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    torch::Tensor da_prior_query_mask;
    if (depthanything_densify_alignment_mode_ == "da_prior") {
        da_prior_query_mask = torch::from_blob(
                                  hole_mask.data(),
                                  {H, W},
                                  torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
                                  .clone()
                                  .to(torch::kBool);
    }

    torch::Tensor sparse_uv;
    torch::Tensor sparse_depth;
    torch::Tensor aligned_depth;
    MonoPriorAlignmentStats align_stats;
    bool used_rendered_alignment = false;
    bool used_da_prior_alignment = false;
    bool used_global_alignment = false;
    bool alignment_ok = false;
    if (depthanything_densify_alignment_mode_ == "rendered") {
        alignment_ok = use_metric3d
            ? alignMetricDepthPriorToRenderedDepth(
                  mono_prior,
                  depth_cpu,
                  alpha_cpu,
                  n_contrib_cpu,
                  cam.near,
                  depthanything_densify_min_sparse_anchors_,
                  aligned_depth,
                  align_stats)
            : alignDepthAnythingPriorToRenderedDepth(
                  mono_prior,
                  depth_cpu,
                  alpha_cpu,
                  n_contrib_cpu,
                  cam.near,
                  depthanything_densify_min_sparse_anchors_,
                  aligned_depth,
                  align_stats);
        used_rendered_alignment = alignment_ok;
    }
    const bool has_sparse_anchors =
        buildSparseDepthFromKeyframeOrbAnchors(pkf, W, H, sparse_uv, sparse_depth);
    if (!alignment_ok && has_sparse_anchors) {
        if (depthanything_densify_alignment_mode_ == "da_prior") {
            alignment_ok = alignMonoPriorToSparseAnchorsDaPrior(
                mono_prior,
                sparse_uv,
                sparse_depth,
                da_prior_query_mask,
                use_metric3d,
                cam.near,
                depthanything_densify_min_sparse_anchors_,
                depthanything_da_prior_knn_k_,
                depthanything_da_prior_distance_weighting_,
                depthanything_da_prior_max_pixel_dist_,
                aligned_depth,
                align_stats);
            used_da_prior_alignment = alignment_ok;
        }
        if (!alignment_ok) {
            alignment_ok = use_metric3d
                ? alignMetricDepthPriorToSparseAnchors(
                      mono_prior,
                      sparse_uv,
                      sparse_depth,
                      cam.near,
                      depthanything_densify_min_sparse_anchors_,
                      aligned_depth,
                      align_stats)
                : alignDepthAnythingPriorToSparseAnchors(
                      mono_prior,
                      sparse_uv,
                      sparse_depth,
                      cam.near,
                      depthanything_densify_min_sparse_anchors_,
                      aligned_depth,
                      align_stats);
        }
        if (alignment_ok) {
            accumulateMonoPriorGlobalAlignment(
                align_stats.scale,
                align_stats.shift,
                static_cast<float>(std::max<int64_t>(1, align_stats.num_fit_anchors)));
        }
    }
    if (!alignment_ok) {
        if (depthanything_global_alignment_valid_ &&
            ((use_metric3d &&
              applyMetricDepthAlignment(
                  mono_prior,
                  depthanything_global_align_scale_,
                  depthanything_global_align_shift_,
                  aligned_depth)) ||
             (!use_metric3d &&
              applyDepthAnythingAffineAlignment(
                  mono_prior,
                  depthanything_global_align_scale_,
                  depthanything_global_align_shift_,
                  aligned_depth)))) {
            used_global_alignment = true;
            align_stats.scale = depthanything_global_align_scale_;
            align_stats.shift = depthanything_global_align_shift_;
        } else {
            std::cout << "[mono_prior_alignment] path=fill_holes"
                      << " iter=" << iter
                      << " kf=" << pkf->fid_
                      << " requested=" << depthanything_densify_alignment_mode_
                      << " used=none"
                      << " fallback=0"
                      << " failed=1"
                      << " has_sparse_anchors=" << (has_sparse_anchors ? 1 : 0)
                      << " anchors=" << align_stats.num_valid_anchors
                      << " fit_anchors=" << align_stats.num_fit_anchors
                      << " min_required=" << depthanything_densify_min_sparse_anchors_
                      << " global_valid=" << (depthanything_global_alignment_valid_ ? 1 : 0)
                      << std::endl;
            updateRenderedDepthCandidateLifecycle();
            return;
        }
    }
    const char* fill_holes_alignment_source =
        used_rendered_alignment ? "rendered" :
        (used_da_prior_alignment ? "da_prior" :
         (used_global_alignment ? "global" : "orb"));
    const bool fill_holes_alignment_fallback =
        depthanything_densify_alignment_mode_ != fill_holes_alignment_source;
    std::cout << "[mono_prior_alignment] path=fill_holes"
              << " iter=" << iter
              << " kf=" << pkf->fid_
              << " requested=" << depthanything_densify_alignment_mode_
              << " used=" << fill_holes_alignment_source
              << " fallback=" << (fill_holes_alignment_fallback ? 1 : 0)
              << " anchors=" << align_stats.num_valid_anchors
              << " fit_anchors=" << align_stats.num_fit_anchors
              << " scale=" << align_stats.scale
              << " shift=" << align_stats.shift
              << " med_rel_err=" << align_stats.median_rel_depth_error
              << " p90_rel_err=" << align_stats.p90_rel_depth_error
              << " inlier_ratio=" << align_stats.final_inlier_ratio
              << std::endl;
    int64_t prior_finite_count = 0;
    int64_t aligned_finite_count = 0;
    float prior_min = std::numeric_limits<float>::quiet_NaN();
    float prior_max = std::numeric_limits<float>::quiet_NaN();
    float prior_mean = std::numeric_limits<float>::quiet_NaN();
    float aligned_min = std::numeric_limits<float>::quiet_NaN();
    float aligned_max = std::numeric_limits<float>::quiet_NaN();
    float aligned_mean = std::numeric_limits<float>::quiet_NaN();
    finiteTensorStats(mono_prior, &prior_finite_count, &prior_min, &prior_max, &prior_mean);
    finiteTensorStats(aligned_depth, &aligned_finite_count, &aligned_min, &aligned_max, &aligned_mean);

    auto image_cpu = pkf->original_image_.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (image_cpu.dim() == 4 && image_cpu.size(0) == 1) {
        image_cpu = image_cpu.squeeze(0);
    }
    if (image_cpu.dim() != 3 || image_cpu.size(0) < 3 ||
        image_cpu.size(1) != H || image_cpu.size(2) != W) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }
    if (image_cpu.size(0) > 3) {
        image_cpu = image_cpu.index({torch::indexing::Slice(0, 3)}).contiguous();
    }

    auto image_acc = image_cpu.accessor<float, 3>();

    torch::Tensor valid_mask =
        torch::isfinite(aligned_depth) &
        (aligned_depth > std::max(cam.near, 1e-4f));
    if (std::isfinite(RGBD_max_depth_) && RGBD_max_depth_ > 0.0f) {
        valid_mask &= (aligned_depth < RGBD_max_depth_);
    }
    {
        auto hole_mask_tensor = torch::from_blob(
            hole_mask.data(),
            {H, W},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
        valid_mask &= hole_mask_tensor.to(torch::kBool);
    }
    auto mask_it = undistort_mask_.find(pkf->camera_id_);
    if (mask_it != undistort_mask_.end() && mask_it->second.defined()) {
        torch::Tensor eval_mask = mask_it->second.detach().to(torch::kCPU).to(torch::kFloat32);
        if (eval_mask.dim() == 4 && eval_mask.size(0) == 1) {
            eval_mask = eval_mask.squeeze(0);
        }
        if (eval_mask.dim() == 3) {
            eval_mask = eval_mask.index({0});
        }
        if (eval_mask.dim() == 2 &&
            eval_mask.size(0) == H &&
            eval_mask.size(1) == W) {
            valid_mask &= (eval_mask > 0.5f);
        }
    }
    const int64_t valid_hole_pixels_after_mask = valid_mask.sum().item<int64_t>();

    const float fx = (cam.fx > 1e-6f)
        ? cam.fx
        : (0.5f * static_cast<float>(W) / std::max(cam.tanfovx, 1e-6f));
    const float fy = (cam.fy > 1e-6f)
        ? cam.fy
        : (0.5f * static_cast<float>(H) / std::max(cam.tanfovy, 1e-6f));
    if (fx <= 1e-6f || fy <= 1e-6f) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    const int stride = std::max(1, depthanything_densify_stride_);
    std::vector<int64_t> selected_idx;
    selected_idx.reserve(
        static_cast<size_t>((H + stride - 1) / stride) *
        static_cast<size_t>((W + stride - 1) / stride));
    auto valid_acc = valid_mask.accessor<bool, 2>();
    for (int y = 0; y < H; y += stride) {
        for (int x = 0; x < W; x += stride) {
            if (valid_acc[y][x]) {
                selected_idx.push_back(static_cast<int64_t>(y) * static_cast<int64_t>(W) + x);
            }
        }
    }

    // std::cout << "[mono_prior_fill_holes/start] iter=" << iter
    //           << " kf=" << pkf->fid_
    //           << " anchors=" << align_stats.num_valid_anchors
    //           << " fit_anchors=" << align_stats.num_fit_anchors
    //           << " prior_finite=" << prior_finite_count
    //           << " prior_range=[" << prior_min << "," << prior_max << "]"
    //           << " prior_mean=" << prior_mean
    //           << " aligned_finite=" << aligned_finite_count
    //           << " aligned_range=[" << aligned_min << "," << aligned_max << "]"
    //           << " aligned_mean=" << aligned_mean
    //           << " hole_pixels=" << hole_pixels
    //           << " valid_hole_pixels=" << valid_hole_pixels_after_mask
    //           << " selected_pixels=" << selected_idx.size()
    //           << " stride=" << stride
    //           << " alignment_source="
    //           << (used_rendered_alignment ? "rendered" :
    //               (used_global_alignment ? "global" : "orb"))
    //           << " align_scale=" << align_stats.scale
    //           << " align_shift=" << align_stats.shift
    //           << " med_rel_err=" << align_stats.median_rel_depth_error
    //           << " p90_rel_err=" << align_stats.p90_rel_depth_error
    //           << " inlier_ratio=" << align_stats.final_inlier_ratio
    //           << " global_scale=" << depthanything_global_align_scale_
    //           << " global_shift=" << depthanything_global_align_shift_
    //           << " global_obs=" << depthanything_global_align_observations_
    //           << " anchor_depth_stats=[" << align_stats.sparse_depth_q05
    //           << "," << align_stats.sparse_depth_q95 << "]"
    //           << std::endl;

    if (selected_idx.empty()) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    const Sophus::SE3f Twc = pkf->getPosef().inverse();
    auto aligned_depth_acc = aligned_depth.accessor<float, 2>();

    std::vector<float> candidate_points_world;
    std::vector<float> candidate_colors;
    candidate_points_world.reserve(selected_idx.size() * 3);
    candidate_colors.reserve(selected_idx.size() * 3);
    float candidate_depth_min = std::numeric_limits<float>::infinity();
    float candidate_depth_max = 0.0f;
    Eigen::Vector3f candidate_world_min(
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity());
    Eigen::Vector3f candidate_world_max(
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity());
    std::vector<std::string> sample_candidates;
    sample_candidates.reserve(3);

    for (const int64_t flat_idx : selected_idx) {
        const int y = static_cast<int>(flat_idx / W);
        const int x = static_cast<int>(flat_idx % W);
        const float z = aligned_depth_acc[y][x];
        if (!std::isfinite(z) || z <= 0.0f) {
            continue;
        }

        const float Xc = (static_cast<float>(x) - cam.cx) / fx * z;
        const float Yc = (static_cast<float>(y) - cam.cy) / fy * z;
        const Eigen::Vector3f p_cam(Xc, Yc, z);
        const Eigen::Vector3f p_world = Twc * p_cam;
        if (!std::isfinite(p_world.x()) ||
            !std::isfinite(p_world.y()) ||
            !std::isfinite(p_world.z())) {
            continue;
        }

        candidate_points_world.push_back(p_world.x());
        candidate_points_world.push_back(p_world.y());
        candidate_points_world.push_back(p_world.z());
        candidate_colors.push_back(std::clamp(image_acc[0][y][x], 0.0f, 1.0f));
        candidate_colors.push_back(std::clamp(image_acc[1][y][x], 0.0f, 1.0f));
        candidate_colors.push_back(std::clamp(image_acc[2][y][x], 0.0f, 1.0f));
        candidate_depth_min = std::min(candidate_depth_min, z);
        candidate_depth_max = std::max(candidate_depth_max, z);
        candidate_world_min = candidate_world_min.cwiseMin(p_world);
        candidate_world_max = candidate_world_max.cwiseMax(p_world);
        if (sample_candidates.size() < 3) {
            std::ostringstream oss;
            oss << "(u=" << x
                << ",v=" << y
                << ",z=" << z
                << ",pw=[" << p_world.x() << "," << p_world.y() << "," << p_world.z() << "])";
            sample_candidates.push_back(oss.str());
        }
    }

    const int64_t num_candidates = static_cast<int64_t>(candidate_points_world.size() / 3);
    // std::cout << "[mono_prior_fill_holes/candidates] iter=" << iter
    //           << " kf=" << pkf->fid_
    //           << " valid_candidates=" << num_candidates
    //           << " depth_range=["
    //           << (num_candidates > 0 ? candidate_depth_min : std::numeric_limits<float>::quiet_NaN())
    //           << ","
    //           << (num_candidates > 0 ? candidate_depth_max : std::numeric_limits<float>::quiet_NaN())
    //           << "]"
    //           << " world_bbox_min=["
    //           << (num_candidates > 0 ? candidate_world_min.x() : std::numeric_limits<float>::quiet_NaN()) << ","
    //           << (num_candidates > 0 ? candidate_world_min.y() : std::numeric_limits<float>::quiet_NaN()) << ","
    //           << (num_candidates > 0 ? candidate_world_min.z() : std::numeric_limits<float>::quiet_NaN()) << "]"
    //           << " world_bbox_max=["
    //           << (num_candidates > 0 ? candidate_world_max.x() : std::numeric_limits<float>::quiet_NaN()) << ","
    //           << (num_candidates > 0 ? candidate_world_max.y() : std::numeric_limits<float>::quiet_NaN()) << ","
    //           << (num_candidates > 0 ? candidate_world_max.z() : std::numeric_limits<float>::quiet_NaN()) << "]"
    //           << std::endl;
    // for (size_t i = 0; i < sample_candidates.size(); ++i) {
    //     std::cout << "[mono_prior_fill_holes/sample] iter=" << iter
    //               << " kf=" << pkf->fid_
    //               << " idx=" << i
    //               << " " << sample_candidates[i]
    //               << std::endl;
    // }

    if (num_candidates > 0) {
        saveAccumulatedOrbDepthProjectionPng(
            mpSLAM,
            pkf,
            runtimeOrbDepthDebugDir(result_dir_),
            iter,
            RGBD_min_depth_,
            RGBD_max_depth_,
            aligned_depth);

        auto points_tensor = torch::from_blob(
            candidate_points_world.data(),
            {num_candidates, 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone().to(device_type_);
        auto colors_tensor = torch::from_blob(
            candidate_colors.data(),
            {num_candidates, 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone().to(device_type_);

        std::vector<sv::MiniCam> tr_cams;
        tr_cams.reserve(scene_->keyframes().size());
        for (auto& kv : scene_->keyframes()) {
            if (kv.second) {
                tr_cams.push_back(
                    kv.second->toMiniCam(kv.second->image_height_, kv.second->image_width_));
            }
        }

        std::unique_lock<std::mutex> lock_render(mutex_render_);
        const bool log_depthanything_fill_holes_created_voxels =
            enable_rerun_ && !rerun_final_only_;
        if (log_depthanything_fill_holes_created_voxels) {
            voxel_model_->setNextRealInsertionRerunEntityPath(
                "world/mono_prior_fill_holes/created");
        }
        voxel_model_->setRenderedDepthCandidateRealAdjacency(
            false,
            1);
        voxel_model_->setNextRenderedDepthCandidateInsertion(
            true,
            "",
            kRenderedCandidateSourceMonoPrior,
            /*insert_as_real_protected=*/true);
        voxel_model_->increasePcd(points_tensor, colors_tensor, iter, tr_cams);
        if (pkf->mono_prior_first_apply_iter_ < 0) {
            pkf->mono_prior_first_apply_iter_ = iter;
        }
        voxel_model_->setNextRenderedDepthCandidateInsertion(false);
        if (log_depthanything_fill_holes_created_voxels) {
            voxel_model_->setNextRealInsertionRerunEntityPath("");
        }
        const sv::VoxelModel::IncreasePcdStats insert_stats = voxel_model_->lastIncreasePcdStats();
        // std::cout << "[mono_prior_fill_holes/insert] iter=" << iter
        //           << " kf=" << pkf->fid_
        //           << " raw_points_in=" << insert_stats.raw_points_in
        //           << " points_after_far_filter=" << insert_stats.points_after_far_filter
        //           << " unique_before_filter=" << insert_stats.unique_voxel_candidates_before_insert_filter
        //           << " unique_after_filter=" << insert_stats.unique_voxel_candidates_after_insert_filter
        //           << " duplicate_existing_voxels=" << insert_stats.duplicate_existing_voxels
        //           << " new_voxels=" << insert_stats.new_voxels
        //           << " pending_promotions=" << insert_stats.pending_promotions
        //           << " pending_support_updates=" << insert_stats.pending_support_updates
        //           << std::endl;
    }

    updateRenderedDepthCandidateLifecycle();
}

void VoxelMapper::increasePcdByKeyframeRenderedDepthInsertion(
    std::shared_ptr<VoxelKeyframe> pkf)
{
    torch::NoGradGuard no_grad;

    if (!pkf || !voxel_model_) {
        return;
    }

    const int iter = getIteration();
    if (sensor_type_ != MONOCULAR) {
        static bool logged_non_mono = false;
        if (!logged_non_mono) {
            std::cout << "[rendered_depth_insert] skipping: current implementation is MONOCULAR-only.\n";
            logged_non_mono = true;
        }
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    const int H = pkf->image_height_;
    const int W = pkf->image_width_;
    if (H <= 0 || W <= 0) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    sv::MiniCam cam = pkf->toMiniCam(H, W);
    std::unordered_map<std::string, torch::Tensor> render_pkg;
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        render_pkg = voxel_model_->render(
            cam,
            H,
            W,
            torch::Tensor(),
            "dontcare",
            false,
            std::nullopt,
            true,
            true,
            true,
            false,
            false,
            sv::RenderOpts{});
    }

    auto it_depth = render_pkg.find("raw_depth");
    if (it_depth == render_pkg.end()) it_depth = render_pkg.find("depth");
    auto it_normal = render_pkg.find("raw_normal");
    if (it_normal == render_pkg.end()) it_normal = render_pkg.find("normal");
    if (it_depth == render_pkg.end() || it_normal == render_pkg.end() ||
        !it_depth->second.defined() || !it_normal->second.defined()) {
        std::cout << "[rendered_depth_insert/start] iter=" << iter
                  << " kf=" << pkf->fid_
                  << " render outputs missing depth/normal; skipping"
                  << std::endl;
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    auto render_depth = it_depth->second;
    if (render_depth.dim() == 4 && render_depth.size(0) == 1) render_depth = render_depth.squeeze(0);
    if (render_depth.dim() == 3 && render_depth.size(0) >= 1) render_depth = render_depth.index({0});
    if (render_depth.dim() != 2) {
        std::cout << "[rendered_depth_insert/start] iter=" << iter
                  << " kf=" << pkf->fid_
                  << " invalid depth shape=" << render_depth.sizes()
                  << std::endl;
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    auto render_normal = it_normal->second;
    if (render_normal.dim() == 4 && render_normal.size(0) == 1) render_normal = render_normal.squeeze(0);
    if (render_normal.dim() != 3 || render_normal.size(0) < 3) {
        std::cout << "[rendered_depth_insert/start] iter=" << iter
                  << " kf=" << pkf->fid_
                  << " invalid normal shape=" << render_normal.sizes()
                  << std::endl;
        updateRenderedDepthCandidateLifecycle();
        return;
    }
    if (render_normal.size(0) > 3) {
        render_normal = render_normal.index({torch::indexing::Slice(0, 3)});
    }

    auto depth_cpu = render_depth.to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto normal_cpu = render_normal.to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto image_cpu = pkf->original_image_.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (image_cpu.dim() == 4 && image_cpu.size(0) == 1) image_cpu = image_cpu.squeeze(0).contiguous();

    if (depth_cpu.size(0) != H || depth_cpu.size(1) != W ||
        normal_cpu.size(1) != H || normal_cpu.size(2) != W ||
        image_cpu.dim() != 3 || image_cpu.size(0) < 3 ||
        image_cpu.size(1) != H || image_cpu.size(2) != W) {
        std::cout << "[rendered_depth_insert/start] iter=" << iter
                  << " kf=" << pkf->fid_
                  << " shape mismatch depth=" << depth_cpu.sizes()
                  << " normal=" << normal_cpu.sizes()
                  << " image=" << image_cpu.sizes()
                  << std::endl;
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    auto depth_acc = depth_cpu.accessor<float, 2>();
    auto normal_acc = normal_cpu.accessor<float, 3>();
    auto image_acc = image_cpu.accessor<float, 3>();

    std::vector<int64_t> selected_frontier_idx;
    selected_frontier_idx.reserve(static_cast<size_t>(rendered_depth_insert_max_points_per_kf_));
    int64_t hit_pixels = 0;
    int64_t frontier_hit_pixels = 0;
    const int frontier_radius = std::max(1, rendered_depth_insert_frontier_radius_px_);
    const int stride = std::max(1, rendered_depth_insert_stride_);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const float z = depth_acc[y][x];
            if (!std::isfinite(z) || z <= 0.0f) {
                continue;
            }
            ++hit_pixels;

            bool has_miss_neighbor = false;
            for (int dy = -frontier_radius; dy <= frontier_radius && !has_miss_neighbor; ++dy) {
                const int yy = y + dy;
                if (yy < 0 || yy >= H) continue;
                for (int dx = -frontier_radius; dx <= frontier_radius; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    const int xx = x + dx;
                    if (xx < 0 || xx >= W) continue;
                    const float z_nb = depth_acc[yy][xx];
                    if (!std::isfinite(z_nb) || z_nb <= 0.0f) {
                        has_miss_neighbor = true;
                        break;
                    }
                }
            }
            if (!has_miss_neighbor) continue;

            ++frontier_hit_pixels;
            if ((x % stride) == 0 && (y % stride) == 0) {
                selected_frontier_idx.push_back(static_cast<int64_t>(y) * static_cast<int64_t>(W) + x);
            }
        }
    }

    if (rendered_depth_insert_max_points_per_kf_ > 0 &&
        static_cast<int>(selected_frontier_idx.size()) > rendered_depth_insert_max_points_per_kf_) {
        std::vector<int64_t> keep;
        keep.reserve(static_cast<size_t>(rendered_depth_insert_max_points_per_kf_));
        if (rendered_depth_insert_max_points_per_kf_ == 1) {
            keep.push_back(selected_frontier_idx[selected_frontier_idx.size() / 2]);
        } else {
            const double step = static_cast<double>(selected_frontier_idx.size() - 1) /
                                static_cast<double>(rendered_depth_insert_max_points_per_kf_ - 1);
            for (int i = 0; i < rendered_depth_insert_max_points_per_kf_; ++i) {
                const size_t idx = static_cast<size_t>(std::llround(step * static_cast<double>(i)));
                keep.push_back(selected_frontier_idx[std::min(idx, selected_frontier_idx.size() - 1)]);
            }
        }
        selected_frontier_idx.swap(keep);
    }

    std::cout << "[rendered_depth_insert/start] iter=" << iter
              << " kf=" << pkf->fid_
              << " hit_pixels=" << hit_pixels
              << " frontier_hit_pixels=" << frontier_hit_pixels
              << " selected_frontier_pixels=" << selected_frontier_idx.size()
              << " stride=" << stride
              << " frontier_radius_px=" << frontier_radius
              << std::endl;

    if (selected_frontier_idx.empty()) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    const float fx = (cam.fx > 1e-6f)
        ? cam.fx
        : (0.5f * static_cast<float>(W) / std::max(cam.tanfovx, 1e-6f));
    const float fy = (cam.fy > 1e-6f)
        ? cam.fy
        : (0.5f * static_cast<float>(H) / std::max(cam.tanfovy, 1e-6f));
    if (fx <= 1e-6f || fy <= 1e-6f) {
        updateRenderedDepthCandidateLifecycle();
        return;
    }

    const Sophus::SE3f Twc = pkf->getPosef().inverse();
    const Eigen::Vector3f cam_pos_world = Twc.translation();
    const float offset_m =
        std::max(0.0f, rendered_depth_insert_normal_offset_vox_) * voxel_model_->fixedVoxSize();

    std::vector<float> frontier_points_world;
    std::vector<float> candidate_points_world;
    std::vector<float> candidate_colors;
    frontier_points_world.reserve(selected_frontier_idx.size() * 3);
    candidate_points_world.reserve(selected_frontier_idx.size() * 3);
    candidate_colors.reserve(selected_frontier_idx.size() * 3);

    for (const int64_t flat_idx : selected_frontier_idx) {
        const int y = static_cast<int>(flat_idx / W);
        const int x = static_cast<int>(flat_idx % W);
        const float z = depth_acc[y][x];
        if (!std::isfinite(z) || z <= 0.0f) {
            continue;
        }

        Eigen::Vector3f normal_world(
            normal_acc[0][y][x],
            normal_acc[1][y][x],
            normal_acc[2][y][x]);
        if (!std::isfinite(normal_world.x()) ||
            !std::isfinite(normal_world.y()) ||
            !std::isfinite(normal_world.z())) {
            continue;
        }
        const float normal_norm = normal_world.norm();
        if (!(normal_norm > 1e-6f)) {
            continue;
        }
        normal_world /= normal_norm;

        const float Xc = (static_cast<float>(x) - cam.cx) / fx * z;
        const float Yc = (static_cast<float>(y) - cam.cy) / fy * z;
        const Eigen::Vector3f p_cam(Xc, Yc, z);
        Eigen::Vector3f p_world = Twc * p_cam;
        if (!std::isfinite(p_world.x()) ||
            !std::isfinite(p_world.y()) ||
            !std::isfinite(p_world.z())) {
            continue;
        }

        Eigen::Vector3f ray_dir = p_world - cam_pos_world;
        const float ray_norm = ray_dir.norm();
        if (!(ray_norm > 1e-6f)) {
            continue;
        }
        ray_dir /= ray_norm;
        if (normal_world.dot(ray_dir) < 0.0f) {
            normal_world = -normal_world;
        }

        const Eigen::Vector3f p_candidate = p_world + offset_m * normal_world;
        if (!std::isfinite(p_candidate.x()) ||
            !std::isfinite(p_candidate.y()) ||
            !std::isfinite(p_candidate.z())) {
            continue;
        }

        frontier_points_world.push_back(p_world.x());
        frontier_points_world.push_back(p_world.y());
        frontier_points_world.push_back(p_world.z());

        candidate_points_world.push_back(p_candidate.x());
        candidate_points_world.push_back(p_candidate.y());
        candidate_points_world.push_back(p_candidate.z());

        candidate_colors.push_back(std::clamp(image_acc[0][y][x], 0.0f, 1.0f));
        candidate_colors.push_back(std::clamp(image_acc[1][y][x], 0.0f, 1.0f));
        candidate_colors.push_back(std::clamp(image_acc[2][y][x], 0.0f, 1.0f));
    }

    const int64_t num_candidates = static_cast<int64_t>(candidate_points_world.size() / 3);
    std::cout << "[rendered_depth_insert/candidates] iter=" << iter
              << " kf=" << pkf->fid_
              << " valid_candidates=" << num_candidates
              << " normal_offset_m=" << offset_m
              << std::endl;

    if (num_candidates > 0 && enable_rerun_ && !rerun_final_only_ && rerun_rendered_depth_insert_) {
        auto pts_frontier = torch::from_blob(
            frontier_points_world.data(),
            {num_candidates, 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
        auto pts_candidate = torch::from_blob(
            candidate_points_world.data(),
            {num_candidates, 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
        auto colors_candidate = torch::from_blob(
            candidate_colors.data(),
            {num_candidates, 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
        auto frontier_color = torch::full(
            {num_candidates, 3},
            0.0f,
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        frontier_color.index_put_({torch::indexing::Slice(), 1}, 1.0f);
        frontier_color.index_put_({torch::indexing::Slice(), 2}, 1.0f);
        sv::RerunVisualizerBridge::instance().visualizePoints3D(
            pts_frontier,
            frontier_color,
            iter,
            "world/rendered_depth_insert/frontier_points",
            0.01f);
        sv::RerunVisualizerBridge::instance().visualizePoints3D(
            pts_candidate,
            colors_candidate,
            iter,
            "world/rendered_depth_insert/candidate_points",
            0.0125f);
    }

    if (num_candidates > 0) {
        auto points_tensor = torch::from_blob(
            candidate_points_world.data(),
            {num_candidates, 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone().to(device_type_);
        auto colors_tensor = torch::from_blob(
            candidate_colors.data(),
            {num_candidates, 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone().to(device_type_);

        std::vector<sv::MiniCam> tr_cams;
        tr_cams.reserve(scene_->keyframes().size());
        for (auto& kv : scene_->keyframes()) {
            if (kv.second) {
                tr_cams.push_back(
                    kv.second->toMiniCam(kv.second->image_height_, kv.second->image_width_));
            }
        }

        std::unique_lock<std::mutex> lock_render(mutex_render_);
        voxel_model_->setRenderedDepthCandidateRealAdjacency(
            rendered_depth_insert_require_real_adjacency_,
            rendered_depth_insert_adjacency_radius_cells_);
        voxel_model_->setNextRenderedDepthCandidateInsertion(
            true,
            (enable_rerun_ && !rerun_final_only_ && rerun_rendered_depth_insert_)
                ? "world/rendered_depth_insert/created"
                : "",
            kRenderedCandidateSourceDepthInsert);
        voxel_model_->increasePcd(points_tensor, colors_tensor, iter, tr_cams);
        voxel_model_->setNextRenderedDepthCandidateInsertion(false);
    }

    updateRenderedDepthCandidateLifecycle();
}

void VoxelMapper::updateRenderedDepthCandidateLifecycle()
{
    if ((!rendered_depth_insert_ && !depthanything_densify_ && !depthanything_fill_holes_) || !voxel_model_) {
        return;
    }

    const int iter = getIteration();
    std::unique_lock<std::mutex> lock_render(mutex_render_);

    auto candidate_mask = voxel_model_->renderedDepthCandidateMask();
    if (!candidate_mask.defined() || candidate_mask.numel() == 0) {
        return;
    }

    auto flatten_mask = [](torch::Tensor t) {
        if (t.dim() == 2 && t.size(1) == 1) t = t.squeeze(1);
        return t.contiguous().view({-1});
    };

    auto centers = voxel_model_->voxCenter();
    auto sizes = voxel_model_->voxSize();
    if (!centers.defined() || !sizes.defined()) {
        return;
    }
    candidate_mask = flatten_mask(candidate_mask.to(centers.device()).to(torch::kBool));
    if (candidate_mask.numel() != centers.size(0)) {
        return;
    }

    auto support_count = voxel_model_->renderedDepthCandidateSupportCount();
    auto last_seen_kf = voxel_model_->renderedDepthCandidateLastSeenKf();
    auto source_kind = voxel_model_->renderedDepthCandidateSourceKind();
    if (!support_count.defined() || !last_seen_kf.defined()) {
        return;
    }
    support_count = flatten_mask(support_count.to(centers.device()).to(torch::kInt32));
    last_seen_kf = flatten_mask(last_seen_kf.to(centers.device()).to(torch::kInt32));
    if (!source_kind.defined()) {
        source_kind = torch::zeros(
            {candidate_mask.numel()},
            torch::TensorOptions().dtype(torch::kInt32).device(centers.device()));
    } else {
        source_kind = flatten_mask(source_kind.to(centers.device()).to(torch::kInt32));
    }
    if (support_count.numel() != candidate_mask.numel() ||
        last_seen_kf.numel() != candidate_mask.numel() ||
        source_kind.numel() != candidate_mask.numel()) {
        return;
    }

    auto box_sizes = sizes;
    if (box_sizes.dim() == 1) {
        box_sizes = box_sizes.view({-1, 1});
    }
    auto log_candidate_boxes = [&](const torch::Tensor& mask,
                                   const std::string& entity_path,
                                   const std::array<float, 4>& rgba) {
        auto idx = torch::nonzero(mask).view({-1});
        if (idx.numel() == 0) return;
        auto box_centers = centers.index_select(0, idx).contiguous();
        auto box_sizes_local = box_sizes.index_select(0, idx).contiguous();
        auto box_rgba = torch::zeros(
            {idx.size(0), 4},
            torch::TensorOptions().dtype(torch::kFloat32).device(box_centers.device()));
        box_rgba.index_put_({torch::indexing::Slice(), 0}, rgba[0]);
        box_rgba.index_put_({torch::indexing::Slice(), 1}, rgba[1]);
        box_rgba.index_put_({torch::indexing::Slice(), 2}, rgba[2]);
        box_rgba.index_put_({torch::indexing::Slice(), 3}, rgba[3]);
        sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
            box_centers,
            box_sizes_local,
            box_rgba,
            iter,
            entity_path);
    };

    const int32_t current_kf_count = static_cast<int32_t>(scene_->keyframes().size());
    auto promote_mask =
        (candidate_mask &
         (support_count >= opt_params_.rendered_depth_candidate_promote_min_support_)).to(torch::kBool);
    const int64_t n_promote = promote_mask.sum().item<int64_t>();
    if (n_promote > 0) {
        auto promote_depth_mask =
            (promote_mask & (source_kind == kRenderedCandidateSourceDepthInsert)).to(torch::kBool);
        auto promote_mono_mask =
            (promote_mask & (source_kind == kRenderedCandidateSourceMonoPrior)).to(torch::kBool);
        const int64_t n_promote_depth = promote_depth_mask.sum().item<int64_t>();
        const int64_t n_promote_mono = promote_mono_mask.sum().item<int64_t>();
        if (enable_rerun_ && !rerun_final_only_) {
            if (rerun_rendered_depth_insert_) {
                log_candidate_boxes(
                    promote_depth_mask,
                    "world/rendered_depth_insert/promoted",
                    {0.0f, 1.0f, 0.0f, 0.95f});
            }
        }
        voxel_model_->promoteRenderedDepthCandidates(promote_mask);
        // std::cout << "[rendered_candidate/promote] iter=" << iter
        //           << " promoted=" << n_promote
        //           << " rendered_depth_insert=" << n_promote_depth
        //           << " mono_prior=" << n_promote_mono
        //           << " min_support=" << opt_params_.rendered_depth_candidate_promote_min_support_
        //           << std::endl;
    }

    candidate_mask = voxel_model_->renderedDepthCandidateMask();
    support_count = voxel_model_->renderedDepthCandidateSupportCount();
    last_seen_kf = voxel_model_->renderedDepthCandidateLastSeenKf();
    source_kind = voxel_model_->renderedDepthCandidateSourceKind();
    if (!candidate_mask.defined() || !support_count.defined() || !last_seen_kf.defined()) {
        return;
    }
    candidate_mask = flatten_mask(candidate_mask.to(centers.device()).to(torch::kBool));
    support_count = flatten_mask(support_count.to(centers.device()).to(torch::kInt32));
    last_seen_kf = flatten_mask(last_seen_kf.to(centers.device()).to(torch::kInt32));
    if (!source_kind.defined()) {
        source_kind = torch::zeros(
            {candidate_mask.numel()},
            torch::TensorOptions().dtype(torch::kInt32).device(centers.device()));
    } else {
        source_kind = flatten_mask(source_kind.to(centers.device()).to(torch::kInt32));
    }
    if (candidate_mask.numel() != centers.size(0) ||
        support_count.numel() != centers.size(0) ||
        last_seen_kf.numel() != centers.size(0) ||
        source_kind.numel() != centers.size(0)) {
        return;
    }

    auto age_kf = (torch::full(
        {candidate_mask.size(0)},
        current_kf_count,
        torch::TensorOptions().dtype(torch::kInt32).device(candidate_mask.device())) -
        last_seen_kf).to(torch::kInt32);
    auto prune_mask =
        (candidate_mask &
         (support_count < opt_params_.rendered_depth_candidate_promote_min_support_) &
         (last_seen_kf >= 0) &
         (age_kf >= opt_params_.rendered_depth_candidate_prune_kf_age_)).to(torch::kBool);
    const int64_t n_prune = prune_mask.sum().item<int64_t>();
    if (n_prune > 0) {
        auto prune_depth_mask =
            (prune_mask & (source_kind == kRenderedCandidateSourceDepthInsert)).to(torch::kBool);
        auto prune_mono_mask =
            (prune_mask & (source_kind == kRenderedCandidateSourceMonoPrior)).to(torch::kBool);
        const int64_t n_prune_depth = prune_depth_mask.sum().item<int64_t>();
        const int64_t n_prune_mono = prune_mono_mask.sum().item<int64_t>();
        if (enable_rerun_ && !rerun_final_only_) {
            if (rerun_rendered_depth_insert_) {
                log_candidate_boxes(
                    prune_depth_mask,
                    "world/rendered_depth_insert/stale",
                    {1.0f, 0.5f, 0.0f, 0.85f});
            }
        }
        // Do not prune here. Topology-removing pruning requires the normal adapt
        // path, which rebuilds the trainer/optimizer after the topology change.
        // std::cout << "[rendered_candidate/prune_deferred] iter=" << iter
        //           << " stale_candidates=" << n_prune
        //           << " rendered_depth_insert=" << n_prune_depth
        //           << " mono_prior=" << n_prune_mono
        //           << " prune_kf_age=" << opt_params_.rendered_depth_candidate_prune_kf_age_
        //           << " action=defer_to_adapt_prune"
        //           << std::endl;
    }
}

bool VoxelMapper::isStopped() const {
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    return this->stopped_;
}

void VoxelMapper::signalStop(const bool going_to_stop)
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    this->stopped_ = going_to_stop;
}

void VoxelMapper::increaseKeyframeTimesOfUse(
        const std::shared_ptr<VoxelKeyframe>& pkf,
        int times)
 {
     pkf->remaining_times_of_use_ += times;
 }

void VoxelMapper::writeKeyframeUsedTimes(std::filesystem::path result_dir, std::string name_suffix)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    std::filesystem::path result_path = result_dir / ("keyframe_used_times" + name_suffix + ".txt");
    std::ofstream out_stream;
    out_stream.open(result_path, std::ios::app);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open json at " + result_path.string());

    out_stream << "##[Voxel Mapper]Iteration " << getIteration() << " keyframe id, used times, remaining times:\n";
    for (const auto& used_times_it : kfs_used_times_)
        out_stream << used_times_it.first << " "
                   << used_times_it.second << " "
                   << scene_->keyframes().at(used_times_it.first)->remaining_times_of_use_
                   << "\n";
    out_stream << "##=========================================" <<std::endl;

    out_stream.close();
}

void VoxelMapper::recordKeyframeRendered(
    torch::Tensor&           rendered,
    torch::Tensor&           ground_truth,
    unsigned long            kfid,
    std::filesystem::path    result_img_dir,
    std::filesystem::path    result_gt_dir,
    std::filesystem::path    result_loss_dir,
    std::string              name_suffix)
{
    if (record_rendered_image_) {
         auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered);
         cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
         image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
         cv::imwrite(result_img_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + ".jpg"), image_cv);
     }
 
     if (record_ground_truth_image_) {
         auto gt_image_cv = tensor_utils::torchTensor2CvMat_Float32(ground_truth);
         cv::cvtColor(gt_image_cv, gt_image_cv, CV_RGB2BGR);
         gt_image_cv.convertTo(gt_image_cv, CV_8UC3, 255.0f);
         cv::imwrite(result_gt_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_gt.jpg"), gt_image_cv);
     }
 
     if (record_loss_image_) {
         torch::Tensor loss_tensor = torch::abs(rendered - ground_truth);
         auto loss_image_cv = tensor_utils::torchTensor2CvMat_Float32(loss_tensor);
         cv::cvtColor(loss_image_cv, loss_image_cv, CV_RGB2BGR);
         loss_image_cv.convertTo(loss_image_cv, CV_8UC3, 255.0f);
         cv::imwrite(result_loss_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_loss.jpg"), loss_image_cv);
     }
}

void VoxelMapper::renderAndRecordKeyframe(
    std::shared_ptr<VoxelKeyframe> pkf,
    float&       dssim,
    float&       psnr,
    float&       depth_l1,
    float&       depth_f1,
    double&      render_ms,
    const std::filesystem::path& result_img_dir,
    const std::filesystem::path& result_gt_dir,
    const std::filesystem::path& result_loss_dir,
    const std::filesystem::path& result_depth_dir,
    const std::filesystem::path& result_normal_dir,
    const std::string&           name_suffix,
    std::optional<float>         global_depth_scale)
{
    // std::cout << "pkf image height and width: " << pkf->image_height_ << " " << pkf->image_width_ << std::endl;
    sv::MiniCam cam = pkf->toMiniCam(pkf->image_height_, pkf->image_width_);

    // Render options: ensure depth is requested
    sv::RenderOpts ropts;
    ropts.output_T     = true;   // if you want T for other losses
    ropts.output_depth = true;   // IMPORTANT for depth saving
    ropts.output_normal = true;  // for normal debug saving

    auto start_timing = std::chrono::steady_clock::now();
    // Render
    auto render_pkg = voxel_model_->render(
        cam,
        pkf->image_height_,
        pkf->image_width_,
        /* gt_image     */ pkf->original_image_,
        /* color_mode   */ nullptr,
        /* track_max_w  */ false,
        /* ss           */ std::nullopt,
        /* output_depth */ ropts.output_depth,
        /* output_normal*/ ropts.output_normal,
        /* output_T     */ ropts.output_T,
        /* rand_bg      */ false,
        /* use_auto_exp */ false,
        ropts
    );
    // auto render_pkg = voxel_model_->render(cam, pkf->image_height_, pkf->image_width_, pkf->original_image_);
    torch::Tensor rendered_image = render_pkg.at("color").to(mDevice);          // (1,3,H,W)
    // Mask and GT on the same device
    torch::Tensor mask = undistort_mask_[pkf->camera_id_]
                            .to(mDevice)
                            .to(torch::kFloat32);                        // (3,H,W) or (1,3,H,W)
    torch::Tensor gt_image = pkf->original_image_.to(mDevice);          // (3,H,W)
    // Broadcast mask over batch if needed
    torch::Tensor masked_image = rendered_image * mask;                 // (1,3,H,W)
    masked_image = masked_image.squeeze(0);                             // (3,H,W)

    torch::cuda::synchronize();
    auto end_timing = std::chrono::steady_clock::now();
    auto render_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_timing - start_timing).count();
    render_ms = 1e-6 * render_time_ns;

    dssim = loss_utils::ssim(masked_image, gt_image, device_type_).item().toFloat();
    psnr = loss_utils::psnr(masked_image, gt_image).item().toFloat();
    depth_l1 = -1.0f;
    depth_f1 = -1.0f;

    recordKeyframeRendered(masked_image, gt_image, pkf->fid_, result_img_dir, result_gt_dir, result_loss_dir, name_suffix);    

    std::ostringstream ss;
    ss << "kf_" << std::setw(5) << std::setfill('0') << pkf->fid_;
    const std::string stem = ss.str();
    cv::Mat gt_depth_meters_eval;
    const bool has_gt_depth_eval =
        getKeyframeDepthMetersForEval(pkf, pkf->image_height_, pkf->image_width_, gt_depth_meters_eval);
    torch::Tensor eval_mask = mask.detach().to(torch::kCPU).to(torch::kFloat32);
    if (eval_mask.dim() == 4 && eval_mask.size(0) == 1) {
        eval_mask = eval_mask.squeeze(0);
    }
    if (eval_mask.dim() == 3) {
        eval_mask = eval_mask.index({0});
    }
    if (eval_mask.dim() == 2 &&
        eval_mask.size(0) == pkf->image_height_ &&
        eval_mask.size(1) == pkf->image_width_) {
        eval_mask = eval_mask > 0.5f;
    } else {
        eval_mask = torch::ones(
            {pkf->image_height_, pkf->image_width_},
            torch::TensorOptions().dtype(torch::kBool));
    }
    bool have_main_depth_viz_range = false;
    float main_viz_min = 0.0f;
    float main_viz_max = 1.0f;

    // ---- Depth saving / GT-vs-render depth debug ----
    torch::Tensor pred_depth;
    if (renderPkgToMetricDepthForEval(render_pkg, pred_depth)) {
        pred_depth = pred_depth.to(torch::kCPU).contiguous();
        const int H = static_cast<int>(pred_depth.size(0));
        const int W = static_cast<int>(pred_depth.size(1));

        cv::Mat gt_depth_meters = has_gt_depth_eval ? gt_depth_meters_eval : cv::Mat();
        torch::Tensor pred_depth_for_main_viz = pred_depth;
        if (sensor_type_ == MONOCULAR &&
            global_depth_scale.has_value() &&
            std::isfinite(*global_depth_scale) &&
            *global_depth_scale > 0.0f) {
            pred_depth_for_main_viz = pred_depth_for_main_viz * (*global_depth_scale);
        }
        have_main_depth_viz_range = computeSharedDepthVizRange(
            pred_depth_for_main_viz,
            has_gt_depth_eval ? gt_depth_meters : cv::Mat(),
            RGBD_min_depth_,
            RGBD_max_depth_,
            main_viz_min,
            main_viz_max);

        std::ostringstream final_depth_tag;
        final_depth_tag << "_iter_" << std::setw(5) << std::setfill('0') << getIteration();
        const std::filesystem::path depth_path =
            result_depth_dir / (stem + final_depth_tag.str() + ".png");
        const std::filesystem::path depth_gt_path = result_depth_dir / (stem + "_gt.png");
        const std::filesystem::path depth_pair_path =
            result_depth_dir / (stem + "_pair" + final_depth_tag.str() + ".png");
        saveDepthComparisonDebugPngs(
            pred_depth,
            has_gt_depth_eval ? gt_depth_meters : cv::Mat(),
            RGBD_min_depth_,
            RGBD_max_depth_,
            depth_path,
            depth_gt_path,
            depth_pair_path,
            sensor_type_ == MONOCULAR ? global_depth_scale : std::nullopt);

        if (record_depth_metrics_ && has_gt_depth_eval) {
            torch::Tensor gt_depth = torch::from_blob(
                gt_depth_meters.data,
                {H, W},
                torch::TensorOptions().dtype(torch::kFloat32)).clone();
            torch::Tensor pred_depth_for_eval = pred_depth;
            if (sensor_type_ == MONOCULAR &&
                global_depth_scale.has_value() &&
                std::isfinite(*global_depth_scale) &&
                *global_depth_scale > 0.0f) {
                pred_depth_for_eval = pred_depth_for_eval * (*global_depth_scale);
            }

            const torch::Tensor valid_pred =
                torch::isfinite(pred_depth_for_eval) &
                (pred_depth_for_eval > RGBD_min_depth_) &
                (pred_depth_for_eval < RGBD_max_depth_) &
                eval_mask;
            const torch::Tensor valid_gt =
                torch::isfinite(gt_depth) &
                (gt_depth > RGBD_min_depth_) &
                (gt_depth < RGBD_max_depth_) &
                eval_mask;
            const torch::Tensor valid_both = valid_pred & valid_gt;

            const int64_t pred_count = valid_pred.sum().item<int64_t>();
            const int64_t gt_count = valid_gt.sum().item<int64_t>();
            const int64_t both_count = valid_both.sum().item<int64_t>();
            if (pred_count > 0 && gt_count > 0 && both_count > 0) {
                const torch::Tensor abs_err = (pred_depth_for_eval - gt_depth).abs();
                depth_l1 = abs_err.masked_select(valid_both).mean().item<float>();

                const int64_t tp =
                    (valid_both & (abs_err < depth_f1_threshold_m_)).sum().item<int64_t>();
                const float precision =
                    pred_count > 0 ? static_cast<float>(tp) / static_cast<float>(pred_count) : 0.0f;
                const float recall =
                    gt_count > 0 ? static_cast<float>(tp) / static_cast<float>(gt_count) : 0.0f;
                depth_f1 = (precision + recall > 0.0f)
                    ? (2.0f * precision * recall) / (precision + recall)
                    : 0.0f;
            }
        }
    }

    // ---- Normal saving / GT-from-depth-vs-render normal debug ----
    torch::Tensor pred_normal;
    if (renderPkgToNormalForEval(render_pkg, pred_normal)) {
        const int Hn = static_cast<int>(pred_normal.size(1));
        const int Wn = static_cast<int>(pred_normal.size(2));
        torch::Tensor eval_mask_normal = eval_mask;
        if (!(eval_mask_normal.dim() == 2 &&
              eval_mask_normal.size(0) == Hn &&
              eval_mask_normal.size(1) == Wn)) {
            eval_mask_normal = torch::ones(
                {Hn, Wn},
                torch::TensorOptions().dtype(torch::kBool));
        }

        const torch::Tensor pred_normal_mag =
            pred_normal.square().sum(0).sqrt();
        const torch::Tensor pred_valid =
            torch::isfinite(pred_normal).all(0) &
            (pred_normal_mag > 1e-6f) &
            eval_mask_normal;
        torch::Tensor pred_normal_unit = torch::nn::functional::normalize(
            pred_normal,
            torch::nn::functional::NormalizeFuncOptions().dim(0).eps(1e-12));
        pred_normal_unit = torch::where(
            pred_valid.unsqueeze(0),
            pred_normal_unit,
            torch::zeros_like(pred_normal_unit));

        const cv::Mat pred_normal_bgr = colorizeNormalMapBgr(pred_normal_unit);
        const std::filesystem::path normal_path = result_normal_dir / (stem + ".png");
        std::filesystem::create_directories(normal_path.parent_path());
        if (!pred_normal_bgr.empty()) {
            cv::imwrite(normal_path.string(), pred_normal_bgr);
        }

        if (has_gt_depth_eval) {
            cv::Mat gt_depth_for_normal = gt_depth_meters_eval;
            if (gt_depth_for_normal.rows != Hn || gt_depth_for_normal.cols != Wn) {
                cv::resize(
                    gt_depth_for_normal,
                    gt_depth_for_normal,
                    cv::Size(Wn, Hn),
                    0.0,
                    0.0,
                    cv::INTER_NEAREST);
            }

            torch::Tensor gt_depth = torch::from_blob(
                gt_depth_for_normal.data,
                {Hn, Wn},
                torch::TensorOptions().dtype(torch::kFloat32)).clone();
            sv::MiniCam cam_cpu = cam;
            cam_cpu.c2w = cam.c2w.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
            torch::Tensor gt_normal = depth2normalSVRaster(
                cam_cpu,
                gt_depth,
                /*ks=*/3,
                /*tol_cos=*/0.0f).to(torch::kCPU).to(torch::kFloat32).contiguous();

            const torch::Tensor gt_depth_valid =
                torch::isfinite(gt_depth) &
                (gt_depth > RGBD_min_depth_) &
                (gt_depth < RGBD_max_depth_) &
                eval_mask_normal;
            const torch::Tensor gt_normal_mag = gt_normal.square().sum(0).sqrt();
            const torch::Tensor gt_valid =
                torch::isfinite(gt_normal).all(0) &
                (gt_normal_mag > 1e-6f) &
                gt_depth_valid;
            torch::Tensor gt_normal_unit = torch::nn::functional::normalize(
                gt_normal,
                torch::nn::functional::NormalizeFuncOptions().dim(0).eps(1e-12));
            gt_normal_unit = torch::where(
                gt_valid.unsqueeze(0),
                gt_normal_unit,
                torch::zeros_like(gt_normal_unit));

            const cv::Mat gt_normal_bgr = colorizeNormalMapBgr(gt_normal_unit);
            const std::filesystem::path normal_gt_path =
                result_normal_dir / (stem + "_gt_from_depth.png");
            if (!gt_normal_bgr.empty()) {
                cv::imwrite(normal_gt_path.string(), gt_normal_bgr);
            }

            if (!pred_normal_bgr.empty() && !gt_normal_bgr.empty()) {
                cv::Mat normal_pair_bgr;
                cv::hconcat(std::vector<cv::Mat>{gt_normal_bgr, pred_normal_bgr}, normal_pair_bgr);
                const std::filesystem::path normal_pair_path =
                    result_normal_dir / (stem + "_pair.png");
                cv::imwrite(normal_pair_path.string(), normal_pair_bgr);
            }

            constexpr float kRadToDeg = 57.29577951308232f;
            const torch::Tensor valid_both = pred_valid & gt_valid;
            const torch::Tensor dot =
                (pred_normal_unit * gt_normal_unit).sum(0).clamp(-1.0f, 1.0f);
            const torch::Tensor err_deg = torch::where(
                valid_both,
                torch::acos(dot) * kRadToDeg,
                torch::full_like(dot, std::numeric_limits<float>::quiet_NaN()));
            const cv::Mat err_bgr = appendJetLegendBar(
                colorizeFiniteScalarMatJet(
                    depthTensorToCvMatFloat(err_deg),
                    0.0f,
                    45.0f),
                0.0f,
                45.0f,
                " deg");
            const std::filesystem::path normal_err_path =
                result_normal_dir / (stem + "_angular_err.png");
            cv::imwrite(normal_err_path.string(), err_bgr);
        }
    }

    constexpr bool save_shutdown_mono_prior_depth_debug = true;
    if (save_shutdown_mono_prior_depth_debug && sensor_type_ == MONOCULAR) {
        const int mono_prior_apply_iter = pkf->mono_prior_first_apply_iter_;
        const bool mono_prior_was_applied = mono_prior_apply_iter >= 0;
        const int mono_prior_depth_loss_iter = pkf->mono_prior_first_depth_loss_iter_;
        const bool mono_prior_was_optimized = mono_prior_depth_loss_iter >= 0;

        if ((mono_prior_was_applied || mono_prior_was_optimized) &&
            ((opt_params_.lambda_depthanythingv2_ > 0.0f) ||
             (pkf->mono_prior_.defined() && pkf->mono_prior_.numel() > 0))) {
            if (ensureMonoPriorForKeyframe(pkf)) {
                std::ostringstream mono_prior_apply_tag;
                mono_prior_apply_tag << "_iter_"
                                     << std::setw(5) << std::setfill('0')
                                     << mono_prior_apply_iter;
                std::ostringstream mono_prior_loss_tag;
                mono_prior_loss_tag << "_iter_"
                                    << std::setw(5) << std::setfill('0')
                                    << mono_prior_depth_loss_iter;
                std::ostringstream shutdown_iter_tag;
                shutdown_iter_tag << "_iter_"
                                  << std::setw(5) << std::setfill('0')
                                  << getIteration();
                const std::string mono_prior_raw_eval_stem =
                    stem + "_mono_prior_raw_densification" + mono_prior_apply_tag.str();
                const std::string mono_prior_aligned_eval_stem =
                    stem + "_mono_prior_aligned_densification" + mono_prior_apply_tag.str();
                const std::string mono_prior_target_eval_stem =
                    stem + "_mono_prior_target" + mono_prior_loss_tag.str();

                torch::Tensor mono_prior_viz =
                    pkf->mono_prior_.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
                if (mono_prior_viz.dim() == 3 && mono_prior_viz.size(0) == 1) {
                    mono_prior_viz = mono_prior_viz.squeeze(0);
                }

                torch::Tensor mono_target_viz;
                torch::Tensor aligned_mono_prior_eval_viz;
                bool have_target_debug_map = false;
                const bool apply_mono_prior_eval_scale =
                    sensor_type_ == MONOCULAR &&
                    global_depth_scale.has_value() &&
                    std::isfinite(*global_depth_scale) &&
                    *global_depth_scale > 0.0f;
                const bool have_aligned_eval_map =
                    buildAlignedMonoPriorDepthForKeyframe(
                        pkf,
                        cam,
                        pkf->image_width_,
                        pkf->image_height_,
                        aligned_mono_prior_eval_viz);
                if (have_aligned_eval_map) {
                    aligned_mono_prior_eval_viz =
                        aligned_mono_prior_eval_viz.to(torch::kCPU).to(torch::kFloat32).contiguous();
                    if (apply_mono_prior_eval_scale) {
                        aligned_mono_prior_eval_viz =
                            aligned_mono_prior_eval_viz * (*global_depth_scale);
                    }
                }
                if (mono_prior_was_optimized) {
                    if (mono_prior_loss_mode_ == "aligned" ||
                        isMetric3DModelId(mono_prior_model_id_)) {
                        if (have_aligned_eval_map) {
                            mono_target_viz = aligned_mono_prior_eval_viz;
                            have_target_debug_map = true;
                        }
                    } else {
                        torch::Tensor unused_mono_prior_resized;
                        torch::Tensor unused_render_mono_prior_viz;
                        have_target_debug_map = renderPkgToMonoPriorDebugMaps(
                            render_pkg,
                            pkf->mono_prior_,
                            cam.near,
                            unused_mono_prior_resized,
                            mono_target_viz,
                            unused_render_mono_prior_viz);
                    }
                }
                constexpr float kMonoPriorValidMin = 1e-6f;
                constexpr float kMonoPriorValidMax = 1e6f;
                if (mono_prior_was_applied) {
                    torch::Tensor mono_prior_raw_depth_viz = mono_prior_viz;
                    if (!isMetric3DModelId(mono_prior_model_id_)) {
                        const torch::Tensor raw_valid =
                            torch::isfinite(mono_prior_viz) & (mono_prior_viz > 1e-6f);
                        mono_prior_raw_depth_viz = torch::where(
                            raw_valid,
                            1.0f / mono_prior_viz.clamp_min(1e-6f),
                            torch::full_like(
                                mono_prior_viz,
                                std::numeric_limits<float>::quiet_NaN()));
                    }

                    float mono_prior_viz_min = 0.0f;
                    float mono_prior_viz_max = 1.0f;
                    if (computeSharedDepthVizRange(
                            mono_prior_raw_depth_viz,
                            cv::Mat(),
                            kMonoPriorValidMin,
                            kMonoPriorValidMax,
                            mono_prior_viz_min,
                            mono_prior_viz_max)) {
                        const cv::Mat mono_prior_bgr = colorizeDepthMatJet(
                            depthTensorToCvMatFloat(mono_prior_raw_depth_viz),
                            kMonoPriorValidMin,
                            kMonoPriorValidMax,
                            mono_prior_viz_min,
                            mono_prior_viz_max);
                        const std::filesystem::path mono_prior_raw_eval_path =
                            result_depth_dir / (mono_prior_raw_eval_stem + ".png");
                        std::filesystem::create_directories(mono_prior_raw_eval_path.parent_path());
                        cv::imwrite(mono_prior_raw_eval_path.string(), mono_prior_bgr);
                    }

                    if (have_aligned_eval_map) {
                        float mono_aligned_viz_min = 0.0f;
                        float mono_aligned_viz_max = 1.0f;
                        if (computeSharedDepthVizRange(
                                aligned_mono_prior_eval_viz,
                                cv::Mat(),
                                RGBD_min_depth_,
                                RGBD_max_depth_,
                                mono_aligned_viz_min,
                                mono_aligned_viz_max)) {
                            const cv::Mat mono_aligned_bgr = colorizeDepthMatJet(
                                depthTensorToCvMatFloat(aligned_mono_prior_eval_viz),
                                RGBD_min_depth_,
                                RGBD_max_depth_,
                                mono_aligned_viz_min,
                                mono_aligned_viz_max);
                            const std::filesystem::path mono_aligned_eval_path =
                                result_depth_dir / (mono_prior_aligned_eval_stem + ".png");
                            std::filesystem::create_directories(mono_aligned_eval_path.parent_path());
                            cv::imwrite(mono_aligned_eval_path.string(), mono_aligned_bgr);
                        }
                    }
                }

                if (mono_prior_was_optimized && have_target_debug_map) {
                    const cv::Mat mono_target_mat = depthTensorToCvMatFloat(mono_target_viz);
                    float mono_target_viz_min = 0.0f;
                    float mono_target_viz_max = 1.0f;
                    if (computeSharedDepthVizRange(
                            mono_target_viz,
                            cv::Mat(),
                            kMonoPriorValidMin,
                            kMonoPriorValidMax,
                            mono_target_viz_min,
                            mono_target_viz_max)) {
                        const cv::Mat mono_target_bgr = colorizeDepthMatJet(
                            mono_target_mat,
                            kMonoPriorValidMin,
                            kMonoPriorValidMax,
                            mono_target_viz_min,
                            mono_target_viz_max);

                        const std::filesystem::path mono_target_path =
                            result_depth_dir / (mono_prior_target_eval_stem + ".png");
                        std::filesystem::create_directories(mono_target_path.parent_path());
                        cv::imwrite(mono_target_path.string(), mono_target_bgr);
                    }
                }
            }
        }

    }

    // Per-pixel photometric error (L2 or L1)
    torch::Tensor diff = masked_image - gt_image;  // (3,H,W)
    // Option 1: L2 error per pixel
    torch::Tensor per_pixel_err = diff.pow(2).mean(0);   // [H,W]
    std::filesystem::path heatmap_dir = result_loss_dir.parent_path() / "photo_loss";
    std::filesystem::create_directories(heatmap_dir);
    std::ostringstream err_name;
    err_name << getIteration() << "_"
            << pkf->fid_
            << name_suffix
            << "_photoloss.png";
    std::filesystem::path err_path = heatmap_dir / err_name.str();
    savePhotometricErrorHeatmapAsPng(per_pixel_err, err_path);
 }

void VoxelMapper::renderAndRecordAllKeyframes(const std::string& name_suffix)
{
    // Create result directory with current iteration number and suffix
    std::filesystem::path result_dir = result_dir_ / (std::to_string(getIteration()) + name_suffix);
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir);

    // Create subdirectories if needed
    std::filesystem::path image_dir = result_dir / "image";
    if (record_rendered_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_dir);

    std::filesystem::path image_gt_dir = result_dir / "image_gt";
    if (record_ground_truth_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_gt_dir);

    std::filesystem::path image_loss_dir = result_dir / "image_loss";
    if (record_loss_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_loss_dir);

    // New: depth directory inside the same x_shutdown folder
    std::filesystem::path depth_dir = result_dir / "depth";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(depth_dir);
    copyPngFilesToDirectory(runtimeOrbDepthDebugDir(result_dir_), depth_dir);

    std::filesystem::path normal_dir = result_dir / "normal";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(normal_dir);

    // Open logging files
    std::filesystem::path render_time_path = result_dir / "render_time.txt";
    std::ofstream out_time(render_time_path);
    out_time << "##[Voxel Mapper]Render time statistics: keyframe id, time(milliseconds)\n";

    std::filesystem::path dssim_path = result_dir / "dssim.txt";
    std::ofstream out_dssim(dssim_path);
    out_dssim << "##[Voxel Mapper]keyframe id, dssim\n";

    std::filesystem::path psnr_path = result_dir / "psnr.txt";
    std::ofstream out_psnr(psnr_path);
    out_psnr << "##[Voxel Mapper]keyframe id, psnr\n";

    std::ofstream out_depth_l1;
    std::ofstream out_depth_f1;
    if (record_depth_metrics_) {
        std::filesystem::path depth_l1_path = result_dir / "depth_l1.txt";
        out_depth_l1.open(depth_l1_path);
        out_depth_l1 << "##[Voxel Mapper]keyframe id, depth_l1_m\n";

        std::filesystem::path depth_f1_path = result_dir / "depth_f1.txt";
        out_depth_f1.open(depth_f1_path);
        out_depth_f1 << "##[Voxel Mapper]keyframe id, depth_f1_at_tau\n";
    }

    const std::size_t nkfs = scene_->keyframes().size();
    std::optional<float> global_depth_scale = std::nullopt;
    if (sensor_type_ == MONOCULAR) {
        std::filesystem::path depth_scale_debug_path = result_dir / "depth_scale_debug.txt";
        std::ofstream out_depth_scale(depth_scale_debug_path);
        out_depth_scale << "##[Voxel Mapper]Monocular depth debug scale diagnostics\n";
        out_depth_scale << "# columns: kfid has_gt selected_ch selected_scale selected_overlap selected_ratio_after_trim "
                           "selected_q25 selected_q50 selected_q75 selected_pred_min selected_pred_max "
                           "ch0_scale ch0_overlap ch0_ratio_after_trim ch2_scale ch2_overlap ch2_ratio_after_trim\n";

        auto fmt_stat = [](float v) -> std::string {
            if (!std::isfinite(v)) {
                return "nan";
            }
            std::ostringstream os;
            os << std::fixed << std::setprecision(6) << v;
            return os.str();
        };

        auto render_pkg_for_depth_debug =
            [this](const std::shared_ptr<VoxelKeyframe>& pkf)
            -> std::unordered_map<std::string, torch::Tensor>
        {
            sv::MiniCam cam = pkf->toMiniCam(pkf->image_height_, pkf->image_width_);
            sv::RenderOpts ropts;
            ropts.output_T = true;
            ropts.output_depth = true;
            return voxel_model_->render(
                cam,
                pkf->image_height_,
                pkf->image_width_,
                /* gt_image     */ pkf->original_image_,
                /* color_mode   */ nullptr,
                /* track_max_w  */ false,
                /* ss           */ std::nullopt,
                /* output_depth */ ropts.output_depth,
                /* output_normal*/ false,
                /* output_T     */ ropts.output_T,
                /* rand_bg      */ false,
                /* use_auto_exp */ false,
                ropts);
        };

        std::vector<std::pair<float, double>> selected_scale_samples;
        std::vector<std::pair<float, double>> ch0_scale_samples;
        std::vector<std::pair<float, double>> ch2_scale_samples;
        std::size_t selected_valid_kfs = 0;
        std::size_t ch0_valid_kfs = 0;
        std::size_t ch2_valid_kfs = 0;

        auto kfit_debug = scene_->keyframes().begin();
        for (std::size_t i = 0; i < nkfs; ++i, ++kfit_debug) {
            const auto& pkf = (*kfit_debug).second;
            const auto render_pkg = render_pkg_for_depth_debug(pkf);

            torch::Tensor depth_tensor;
            auto it_depth = render_pkg.find("depth");
            if (it_depth == render_pkg.end() || !it_depth->second.defined()) {
                it_depth = render_pkg.find("raw_depth");
            }
            if (it_depth != render_pkg.end() && it_depth->second.defined()) {
                depth_tensor = it_depth->second;
            }

            torch::Tensor pred_selected;
            const bool has_pred_selected =
                renderPkgToMetricDepthForEval(render_pkg, pred_selected);

            cv::Mat gt_depth_meters;
            bool has_gt_depth = false;
            int selected_channel = -1;
            DepthScaleFitStats selected_stats;
            DepthScaleFitStats ch0_stats;
            DepthScaleFitStats ch2_stats;

            if (has_pred_selected) {
                pred_selected = pred_selected.to(torch::kCPU).contiguous();
                has_gt_depth = getKeyframeDepthMetersForEval(
                    pkf,
                    static_cast<int>(pred_selected.size(0)),
                    static_cast<int>(pred_selected.size(1)),
                    gt_depth_meters);

                if (depth_tensor.defined()) {
                    torch::Tensor d = depth_tensor.detach();
                    if (d.dim() == 4 && d.size(0) == 1) {
                        d = d.squeeze(0);
                    }
                    if (d.dim() == 3) {
                        selected_channel = (d.size(0) > 2) ? 2 : 0;
                    } else if (d.dim() == 2) {
                        selected_channel = 0;
                    }
                }

                if (has_gt_depth) {
                    computeDepthScaleFitStats(
                        pred_selected,
                        gt_depth_meters,
                        RGBD_min_depth_,
                        RGBD_max_depth_,
                        selected_stats);
                    if (selected_stats.valid) {
                        selected_scale_samples.emplace_back(
                            selected_stats.scale,
                            static_cast<double>(std::max<int64_t>(selected_stats.ratio_count_after_trim, 1)));
                        ++selected_valid_kfs;
                    }
                }
            }

            if (depth_tensor.defined() && has_gt_depth) {
                torch::Tensor pred_ch0 = tensorToEvalMapExactChannel(depth_tensor, 0);
                if (pred_ch0.defined()) {
                    pred_ch0 = pred_ch0.to(torch::kCPU).contiguous();
                    computeDepthScaleFitStats(
                        pred_ch0,
                        gt_depth_meters,
                        RGBD_min_depth_,
                        RGBD_max_depth_,
                        ch0_stats);
                    if (ch0_stats.valid) {
                        ch0_scale_samples.emplace_back(
                            ch0_stats.scale,
                            static_cast<double>(std::max<int64_t>(ch0_stats.ratio_count_after_trim, 1)));
                        ++ch0_valid_kfs;
                    }
                }

                torch::Tensor pred_ch2 = tensorToEvalMapExactChannel(depth_tensor, 2);
                if (pred_ch2.defined()) {
                    pred_ch2 = pred_ch2.to(torch::kCPU).contiguous();
                    computeDepthScaleFitStats(
                        pred_ch2,
                        gt_depth_meters,
                        RGBD_min_depth_,
                        RGBD_max_depth_,
                        ch2_stats);
                    if (ch2_stats.valid) {
                        ch2_scale_samples.emplace_back(
                            ch2_stats.scale,
                            static_cast<double>(std::max<int64_t>(ch2_stats.ratio_count_after_trim, 1)));
                        ++ch2_valid_kfs;
                    }
                }
            }

            out_depth_scale
                << (*kfit_debug).first << " "
                << (has_gt_depth ? 1 : 0) << " "
                << selected_channel << " "
                << fmt_stat(selected_stats.scale) << " "
                << selected_stats.overlap_count << " "
                << selected_stats.ratio_count_after_trim << " "
                << fmt_stat(selected_stats.ratio_q25) << " "
                << fmt_stat(selected_stats.ratio_q50) << " "
                << fmt_stat(selected_stats.ratio_q75) << " "
                << fmt_stat(selected_stats.pred_min) << " "
                << fmt_stat(selected_stats.pred_max) << " "
                << fmt_stat(ch0_stats.scale) << " "
                << ch0_stats.overlap_count << " "
                << ch0_stats.ratio_count_after_trim << " "
                << fmt_stat(ch2_stats.scale) << " "
                << ch2_stats.overlap_count << " "
                << ch2_stats.ratio_count_after_trim
                << "\n";
        }

        float selected_scale_value = 1.0f;
        float ch0_scale_value = 1.0f;
        float ch2_scale_value = 1.0f;
        const bool have_selected_scale =
            computeWeightedMedianScale(selected_scale_samples, selected_scale_value);
        const bool have_ch0_scale =
            computeWeightedMedianScale(ch0_scale_samples, ch0_scale_value);
        const bool have_ch2_scale =
            computeWeightedMedianScale(ch2_scale_samples, ch2_scale_value);

        if (have_selected_scale) {
            global_depth_scale = selected_scale_value;
        }

        out_depth_scale << "# summary selected_valid_kfs " << selected_valid_kfs << "\n";
        out_depth_scale << "# summary selected_global_scale "
                        << (have_selected_scale ? fmt_stat(selected_scale_value) : std::string("nan")) << "\n";
        out_depth_scale << "# summary ch0_valid_kfs " << ch0_valid_kfs << "\n";
        out_depth_scale << "# summary ch0_global_scale "
                        << (have_ch0_scale ? fmt_stat(ch0_scale_value) : std::string("nan")) << "\n";
        out_depth_scale << "# summary ch2_valid_kfs " << ch2_valid_kfs << "\n";
        out_depth_scale << "# summary ch2_global_scale "
                        << (have_ch2_scale ? fmt_stat(ch2_scale_value) : std::string("nan")) << "\n";

        if (have_selected_scale) {
            std::cout << "[DepthDebug] monocular global depth scale="
                      << std::fixed << std::setprecision(6) << selected_scale_value
                      << " valid_kfs=" << selected_valid_kfs
                      << " ch0_global=" << (have_ch0_scale ? fmt_stat(ch0_scale_value) : std::string("nan"))
                      << " ch2_global=" << (have_ch2_scale ? fmt_stat(ch2_scale_value) : std::string("nan"))
                      << "\n";
        } else {
            std::cout << "[DepthDebug] could not estimate a robust monocular global depth scale.\n";
        }
    }

    // Loop through all keyframes deterministically
    auto kfit = scene_->keyframes().begin();
    float dssim, psnr, depth_l1, depth_f1;
    double render_time;
    double depth_l1_sum = 0.0;
    double depth_f1_sum = 0.0;
    std::size_t depth_eval_count = 0;
    for (std::size_t i = 0; i < nkfs; ++i) {
        renderAndRecordKeyframe(
            (*kfit).second,
            dssim,
            psnr,
            depth_l1,
            depth_f1,
            render_time,
            image_dir,
            image_gt_dir,
            image_loss_dir,
            depth_dir,
            normal_dir,
            name_suffix,
            global_depth_scale);
        out_time << (*kfit).first << " " << std::fixed << std::setprecision(8) << render_time << std::endl;

        out_dssim   << (*kfit).first << " " << std::fixed << std::setprecision(10) << dssim   << std::endl;
        out_psnr    << (*kfit).first << " " << std::fixed << std::setprecision(10) << psnr    << std::endl;
        if (record_depth_metrics_) {
            out_depth_l1 << (*kfit).first << " " << std::fixed << std::setprecision(10) << depth_l1 << std::endl;
            out_depth_f1 << (*kfit).first << " " << std::fixed << std::setprecision(10) << depth_f1 << std::endl;

            if (depth_l1 >= 0.0f && depth_f1 >= 0.0f) {
                depth_l1_sum += static_cast<double>(depth_l1);
                depth_f1_sum += static_cast<double>(depth_f1);
                ++depth_eval_count;
            }
        }

        ++kfit;
    }

    if (record_depth_metrics_) {
        std::filesystem::path depth_metrics_summary_path = result_dir / "depth_metrics_summary.txt";
        std::ofstream out_depth_summary(depth_metrics_summary_path);
        out_depth_summary << "##[Voxel Mapper]Depth evaluation summary\n";
        out_depth_summary << "num_kfs_total " << nkfs << "\n";
        out_depth_summary << "num_kfs_with_depth_eval " << depth_eval_count << "\n";
        out_depth_summary << "f1_threshold_m " << depth_f1_threshold_m_ << "\n";
        if (depth_eval_count > 0) {
            const double mean_depth_l1 = depth_l1_sum / static_cast<double>(depth_eval_count);
            const double mean_depth_f1 = depth_f1_sum / static_cast<double>(depth_eval_count);
            out_depth_summary << std::fixed << std::setprecision(10);
            out_depth_summary << "mean_depth_l1_m " << mean_depth_l1 << "\n";
            out_depth_summary << "mean_depth_f1_at_tau " << mean_depth_f1 << "\n";
            std::cout << "[DepthEval] valid_kfs=" << depth_eval_count
                      << " mean_L1(m)=" << mean_depth_l1
                      << " mean_F1@tau=" << mean_depth_f1
                      << " tau(m)=" << depth_f1_threshold_m_ << "\n";
        } else {
            out_depth_summary << "mean_depth_l1_m -1\n";
            out_depth_summary << "mean_depth_f1_at_tau -1\n";
            std::cout << "[DepthEval] no keyframes with valid GT depth.\n";
        }
    }
}

void VoxelMapper::saveRenderedTsdfMeshPly(const std::filesystem::path& result_path)
{
    switch (rendered_mesh_backend_)
    {
    case 1:
        saveRenderedTsdfMeshPlySparseCpp(result_path);
        return;
    case 0:
    default:
        saveRenderedTsdfMeshPlySvrasterPython(result_path);
        return;
    }
}

void VoxelMapper::saveRenderedTsdfMeshPlySvrasterPython(const std::filesystem::path& result_path)
{
    namespace fs = std::filesystem;
    namespace py = pybind11;
    torch::NoGradGuard no_grad;

    if (!scene_ || scene_->keyframes().empty()) {
        std::cout << "[saveRenderedTsdfMeshPly] No keyframes, skipping.\n";
        return;
    }
    if (!result_path.parent_path().empty()) {
        fs::create_directories(result_path.parent_path());
    }

    py::gil_scoped_acquire gil;
    static py::module_ torch_mod = py::module_::import("torch");
    static py::module_ oct_utils = py::module_::import("src.utils.octree_utils");
    static py::module_ fuser_utils = py::module_::import("src.utils.fuser_utils");
    static py::module_ mc_utils = py::module_::import("src.utils.marching_cubes_utils");
    static py::module_ svm_mod = py::module_::import("src.sparse_voxel_model");
    static py::object nnf_grid_sample =
        torch_mod.attr("nn").attr("functional").attr("grid_sample");

    py::object py_svm = voxel_model_->svm();
    if (py_svm.is_none()) {
        std::cout << "[saveRenderedTsdfMeshPly] Python voxel model unavailable, skipping.\n";
        return;
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
    for (const auto& kv : scene_->keyframes()) {
        const std::shared_ptr<VoxelKeyframe>& pkf = kv.second;
        if (!pkf) {
            continue;
        }
        const int image_width = pkf->image_width_;
        const int image_height = pkf->image_height_;
        if (image_width <= 0 || image_height <= 0) {
            continue;
        }
        py::object py_cam = MiniCam_to_py(pkf->toMiniCam(image_height, image_width));
        move_attr_to_cuda_if_tensor(py_cam, "w2c");
        move_attr_to_cuda_if_tensor(py_cam, "c2w");
        move_attr_to_cuda_if_tensor(py_cam, "position");
        move_attr_to_cuda_if_tensor(py_cam, "lookat");
        py_cams.append(py_cam);
    }
    if (py::len(py_cams) == 0) {
        std::cout << "[saveRenderedTsdfMeshPly] No valid train cameras, skipping.\n";
        return;
    }

    const float target_vox_size =
        (opt_params_.subdivide_target_vox_size_ > 0.0f)
            ? opt_params_.subdivide_target_vox_size_
            : std::max(0.01f, voxel_model_->fixedVoxSize());
    const int outside_level = py::hasattr(py_svm, "outside_level")
        ? py::cast<int>(py_svm.attr("outside_level"))
        : 0;
    torch::Tensor scene_extent = py_svm.attr("scene_extent").cast<torch::Tensor>().contiguous();
    torch::Tensor scene_center = py_svm.attr("scene_center").cast<torch::Tensor>().contiguous();
    torch::Tensor inside_extent = py_svm.attr("inside_extent").cast<torch::Tensor>().contiguous();
    torch::Tensor vox_size_t = torch::full(
        {1},
        target_vox_size,
        scene_extent.options().dtype(torch::kFloat32));
    torch::Tensor final_lv_t = oct_utils.attr("vox_size_2_level")(
        scene_extent,
        py::cast(vox_size_t)).cast<torch::Tensor>().round().to(torch::kInt32);
    int final_lv = final_lv_t.clamp_min(1).clamp_max(voxel_model_->maxNumLevels()).item<int>() - outside_level;
    final_lv = std::max(1, final_lv);
    const float bbox_scale = 1.0f;
    const float bandwidth_vox = 5.0f;
    const float crop_border = 0.01f;
    const float alpha_thres = 0.5f;

    std::cout << "[saveRenderedTsdfMeshPly] begin kfs=" << py::len(py_cams)
              << " target_vox_size=" << target_vox_size
              << " final_lv=" << final_lv
              << " alpha_thres=" << alpha_thres
              << " crop_border=" << crop_border
              << " bandwidth_vox=" << bandwidth_vox
              << " use_mean=0\n";

    bool froze_geo = false;
    py::object inference_ctx = torch_mod.attr("inference_mode")();
    try {
        torch_mod.attr("cuda").attr("empty_cache")();
        inference_ctx.attr("__enter__")();

        if (py::hasattr(py_svm, "freeze_vox_geo")) {
            py_svm.attr("freeze_vox_geo")();
            froze_geo = true;
        }

        const int target_lv = outside_level + final_lv;
        py::tuple clamped = oct_utils.attr("clamp_level")(
            py_svm.attr("octpath"),
            py_svm.attr("octlevel"),
            py::int_(target_lv));
        py::object SparseVoxelModel = svm_mod.attr("SparseVoxelModel");
        py::object vol = SparseVoxelModel(py::arg("sh_degree") = 0);
        vol.attr("octpath_init")(
            scene_center,
            scene_extent,
            clamped[0],
            clamped[1]);

        torch::Tensor inside_min = (scene_center - 0.5f * inside_extent * bbox_scale).contiguous();
        torch::Tensor inside_max = (scene_center + 0.5f * inside_extent * bbox_scale).contiguous();
        torch::Tensor grid_pts_xyz = vol.attr("grid_pts_xyz").cast<torch::Tensor>().contiguous();
        torch::Tensor vox_key = vol.attr("vox_key").cast<torch::Tensor>().contiguous();

        torch::Tensor gridpts_outside =
            ((grid_pts_xyz < inside_min) | (grid_pts_xyz > inside_max)).any(-1);
        torch::Tensor corners_outside = gridpts_outside.index({vox_key});
        torch::Tensor prune_mask = corners_outside.all(-1);
        vol.attr("pruning")(prune_mask);

        grid_pts_xyz = vol.attr("grid_pts_xyz").cast<torch::Tensor>().contiguous();
        vox_key = vol.attr("vox_key").cast<torch::Tensor>().contiguous();
        torch::Tensor vox_size = vol.attr("vox_size").cast<torch::Tensor>().contiguous();
        const float bandwidth = bandwidth_vox * vox_size.min().item<float>();

        py::object Fuser = fuser_utils.attr("Fuser");
        py::object fuser = Fuser(
            py::arg("xyz") = grid_pts_xyz,
            py::arg("bandwidth") = bandwidth,
            py::arg("use_trunc") = true,
            py::arg("fuse_tsdf") = true,
            py::arg("feat_dim") = 0,
            py::arg("alpha_thres") = alpha_thres,
            py::arg("crop_border") = crop_border,
            py::arg("normal_weight") = false,
            py::arg("depth_weight") = false,
            py::arg("border_weight") = false,
            py::arg("use_half") = false);

        const ssize_t num_cams = py::len(py_cams);
        for (ssize_t i = 0; i < num_cams; ++i) {
            py::object cam = py_cams[i];
            py::dict render_pkg = py_svm.attr("render")(
                cam,
                py::arg("output_depth") = true,
                py::arg("output_T") = true).cast<py::dict>();
            torch::Tensor raw_depth = render_pkg["raw_depth"].cast<torch::Tensor>().contiguous();
            torch::Tensor frame_depth = raw_depth.index({2}).unsqueeze(0).contiguous();
            torch::Tensor raw_T = render_pkg["raw_T"].cast<torch::Tensor>().contiguous();
            torch::Tensor frame_alpha = (1.0f - raw_T).contiguous();
            fuser.attr("integrate")(cam, frame_depth, py::arg("alpha") = frame_alpha);

            render_pkg = py::dict();
            raw_depth = torch::Tensor();
            frame_depth = torch::Tensor();
            raw_T = torch::Tensor();
            frame_alpha = torch::Tensor();
            if ((i % 8) == 7) {
                torch_mod.attr("cuda").attr("empty_cache")();
            }
        }

        torch_mod.attr("cuda").attr("empty_cache")();

        torch::Tensor grid_tsdf = fuser.attr("tsdf").cast<torch::Tensor>().squeeze(1).contiguous();
        py::tuple mc = mc_utils.attr("torch_marching_cubes_grid")(
            py::arg("grid_pts_val") = grid_tsdf,
            py::arg("grid_pts_xyz") = grid_pts_xyz,
            py::arg("vox_key") = vox_key,
            py::arg("iso") = 0.0f).cast<py::tuple>();
        torch::Tensor verts = mc[0].cast<torch::Tensor>().to(torch::kFloat32).contiguous();
        torch::Tensor faces = mc[1].cast<torch::Tensor>().to(torch::kInt64).to(torch::kCPU).contiguous();

        const int64_t num_vertices = verts.size(0);
        const int64_t num_faces = faces.size(0);
        if (num_vertices == 0 || num_faces == 0) {
            throw std::runtime_error("SVRaster-style fusion produced an empty mesh");
        }

        torch::Tensor closest_color = torch::full(
            {num_vertices, 3},
            0.5f,
            verts.options().dtype(torch::kFloat32));
        torch::Tensor closest_dist = torch::full(
            {num_vertices},
            std::numeric_limits<float>::infinity(),
            verts.options().dtype(torch::kFloat32));

        for (ssize_t i = 0; i < num_cams; ++i) {
            py::object cam = py_cams[i];
            py::dict render_pkg = py_svm.attr("render")(
                cam,
                py::arg("color_mode") = "sh0",
                py::arg("output_depth") = true,
                py::arg("output_T") = true).cast<py::dict>();

            torch::Tensor frame_color = render_pkg["color"].cast<torch::Tensor>().contiguous();
            torch::Tensor raw_depth = render_pkg["raw_depth"].cast<torch::Tensor>().contiguous();
            torch::Tensor frame_depth = raw_depth.index({2}).unsqueeze(0).contiguous();
            torch::Tensor raw_T = render_pkg["raw_T"].cast<torch::Tensor>().contiguous();
            torch::Tensor frame_alpha = (1.0f - raw_T).contiguous();

            torch::Tensor pts_uv =
                cam.attr("project")(py::cast(verts)).cast<torch::Tensor>().contiguous();
            torch::Tensor inside_mask = (pts_uv.abs() <= 1.0f).all(-1);
            torch::Tensor valid_pts_idx = torch::nonzero(inside_mask).view({-1});
            if (valid_pts_idx.numel() == 0) {
                render_pkg = py::dict();
                frame_color = torch::Tensor();
                raw_depth = torch::Tensor();
                frame_depth = torch::Tensor();
                raw_T = torch::Tensor();
                frame_alpha = torch::Tensor();
                pts_uv = torch::Tensor();
                inside_mask = torch::Tensor();
                valid_pts_idx = torch::Tensor();
                if ((i % 8) == 7) {
                    torch_mod.attr("cuda").attr("empty_cache")();
                }
                continue;
            }

            torch::Tensor valid_pts = verts.index_select(0, valid_pts_idx);
            pts_uv = pts_uv.index_select(0, valid_pts_idx).contiguous();

            torch::Tensor pts_frame_alpha = nnf_grid_sample(
                frame_alpha.view({1, 1, frame_alpha.size(-2), frame_alpha.size(-1)}),
                pts_uv.view({1, 1, -1, 2}),
                py::arg("mode") = "bilinear",
                py::arg("align_corners") = false).cast<torch::Tensor>().flatten();
            torch::Tensor alpha_mask = pts_frame_alpha > alpha_thres;
            torch::Tensor alpha_keep_idx = torch::nonzero(alpha_mask).view({-1});
            if (alpha_keep_idx.numel() == 0) {
                render_pkg = py::dict();
                frame_color = torch::Tensor();
                raw_depth = torch::Tensor();
                frame_depth = torch::Tensor();
                raw_T = torch::Tensor();
                frame_alpha = torch::Tensor();
                pts_uv = torch::Tensor();
                inside_mask = torch::Tensor();
                valid_pts_idx = torch::Tensor();
                valid_pts = torch::Tensor();
                pts_frame_alpha = torch::Tensor();
                alpha_mask = torch::Tensor();
                alpha_keep_idx = torch::Tensor();
                if ((i % 8) == 7) {
                    torch_mod.attr("cuda").attr("empty_cache")();
                }
                continue;
            }

            valid_pts_idx = valid_pts_idx.index_select(0, alpha_keep_idx);
            valid_pts = valid_pts.index_select(0, alpha_keep_idx);
            pts_uv = pts_uv.index_select(0, alpha_keep_idx).contiguous();

            torch::Tensor pts_frame_depth = nnf_grid_sample(
                frame_depth.view({1, 1, frame_depth.size(-2), frame_depth.size(-1)}),
                pts_uv.view({1, 1, -1, 2}),
                py::arg("mode") = "bilinear",
                py::arg("align_corners") = false).cast<torch::Tensor>().flatten();
            torch::Tensor cam_position = cam.attr("position").cast<torch::Tensor>().contiguous();
            torch::Tensor cam_lookat = cam.attr("lookat").cast<torch::Tensor>().contiguous();
            torch::Tensor pts_depth =
                ((valid_pts - cam_position) * cam_lookat).sum(-1);
            torch::Tensor pts_dist = (pts_frame_depth - pts_depth).abs();

            torch::Tensor prev_dist = closest_dist.index_select(0, valid_pts_idx);
            torch::Tensor better_mask = pts_dist < prev_dist;
            torch::Tensor better_idx = torch::nonzero(better_mask).view({-1});
            if (better_idx.numel() > 0) {
                torch::Tensor better_pts_idx = valid_pts_idx.index_select(0, better_idx);
                torch::Tensor better_uv = pts_uv.index_select(0, better_idx).contiguous();
                torch::Tensor pts_color = nnf_grid_sample(
                    frame_color.unsqueeze(0),
                    better_uv.view({1, 1, -1, 2}),
                    py::arg("mode") = "bilinear",
                    py::arg("align_corners") = false).cast<torch::Tensor>();
                pts_color = pts_color.squeeze(0).squeeze(1).transpose(0, 1).contiguous();

                closest_dist.index_put_(
                    {better_pts_idx},
                    pts_dist.index_select(0, better_idx));
                closest_color.index_put_(
                    {better_pts_idx},
                    pts_color);
            }

            render_pkg = py::dict();
            frame_color = torch::Tensor();
            raw_depth = torch::Tensor();
            frame_depth = torch::Tensor();
            raw_T = torch::Tensor();
            frame_alpha = torch::Tensor();
            pts_uv = torch::Tensor();
            inside_mask = torch::Tensor();
            valid_pts_idx = torch::Tensor();
            valid_pts = torch::Tensor();
            pts_frame_alpha = torch::Tensor();
            alpha_mask = torch::Tensor();
            alpha_keep_idx = torch::Tensor();
            pts_frame_depth = torch::Tensor();
            cam_position = torch::Tensor();
            cam_lookat = torch::Tensor();
            pts_depth = torch::Tensor();
            pts_dist = torch::Tensor();
            prev_dist = torch::Tensor();
            better_mask = torch::Tensor();
            better_idx = torch::Tensor();
            if ((i % 8) == 7) {
                torch_mod.attr("cuda").attr("empty_cache")();
            }
        }

        torch::Tensor verts_cpu = verts.to(torch::kCPU).contiguous();
        std::vector<float> vertices_xyz(static_cast<size_t>(verts.numel()));
        std::copy(
            verts_cpu.data_ptr<float>(),
            verts_cpu.data_ptr<float>() + verts_cpu.numel(),
            vertices_xyz.begin());

        torch::Tensor vertices_rgb_cpu =
            (closest_color.clamp(0.0f, 1.0f) * 255.0f)
                .round()
                .to(torch::kUInt8)
                .to(torch::kCPU)
                .contiguous();
        std::vector<uint8_t> vertices_rgb(static_cast<size_t>(vertices_rgb_cpu.numel()));
        std::copy(
            vertices_rgb_cpu.data_ptr<uint8_t>(),
            vertices_rgb_cpu.data_ptr<uint8_t>() + vertices_rgb_cpu.numel(),
            vertices_rgb.begin());

        const int64_t* face_ptr = faces.data_ptr<int64_t>();
        std::vector<uint32_t> tri_idx(static_cast<size_t>(faces.numel()));
        for (int64_t i = 0; i < faces.numel(); ++i) {
            tri_idx[static_cast<size_t>(i)] = static_cast<uint32_t>(face_ptr[i]);
        }

        std::filebuf fb;
        fb.open(result_path, std::ios::out | std::ios::binary);
        std::ostream out(&fb);
        if (out.fail()) {
            throw std::runtime_error("saveRenderedTsdfMeshPly: open failed: " + result_path.string());
        }

        tinyply::PlyFile ply;
        ply.add_properties_to_element(
            "vertex", {"x", "y", "z"},
            tinyply::Type::FLOAT32, static_cast<uint64_t>(num_vertices),
            reinterpret_cast<uint8_t*>(vertices_xyz.data()),
            tinyply::Type::INVALID, 0);
        ply.add_properties_to_element(
            "vertex", {"red", "green", "blue"},
            tinyply::Type::UINT8, static_cast<uint64_t>(num_vertices),
            reinterpret_cast<uint8_t*>(vertices_rgb.data()),
            tinyply::Type::INVALID, 0);
        ply.add_properties_to_element(
            "face", {"vertex_indices"},
            tinyply::Type::UINT32, static_cast<uint64_t>(num_faces),
            reinterpret_cast<uint8_t*>(tri_idx.data()),
            tinyply::Type::UINT8, 3);
        ply.get_comments().push_back("generated_from_svraster_style_tsdf_fusion");
        ply.get_comments().push_back("vertex_colors projected_sh0");
        ply.get_comments().push_back("alpha_thres " + std::to_string(alpha_thres));
        ply.get_comments().push_back("crop_border " + std::to_string(crop_border));
        ply.get_comments().push_back("bandwidth_vox " + std::to_string(bandwidth_vox));
        ply.write(out, /*binary=*/true);
        fb.close();
        inference_ctx.attr("__exit__")(py::none(), py::none(), py::none());
    } catch (const py::error_already_set& e) {
        try {
            inference_ctx.attr("__exit__")(py::none(), py::none(), py::none());
            torch_mod.attr("cuda").attr("empty_cache")();
        } catch (...) {}
        if (froze_geo && py::hasattr(py_svm, "unfreeze_vox_geo")) {
            py_svm.attr("unfreeze_vox_geo")();
        }
        throw std::runtime_error(
            std::string("saveRenderedTsdfMeshPly SVRaster-style extraction failed: ") + e.what());
    } catch (...) {
        try {
            inference_ctx.attr("__exit__")(py::none(), py::none(), py::none());
            torch_mod.attr("cuda").attr("empty_cache")();
        } catch (...) {}
        if (froze_geo && py::hasattr(py_svm, "unfreeze_vox_geo")) {
            py_svm.attr("unfreeze_vox_geo")();
        }
        throw;
    }

    if (froze_geo && py::hasattr(py_svm, "unfreeze_vox_geo")) {
        py_svm.attr("unfreeze_vox_geo")();
    }

    std::cout << "[saveRenderedTsdfMeshPly] Wrote SVRaster-style fused mesh to "
              << result_path << "\n";
}

void VoxelMapper::saveRenderedTsdfMeshPlySparseCpp(const std::filesystem::path& result_path)
{
    namespace fs = std::filesystem;
    namespace py = pybind11;
    torch::NoGradGuard no_grad;

    const auto& keyframes = scene_->keyframes();
    if (keyframes.empty())
    {
        std::cout << "[saveRenderedTsdfMeshPly] skipped: no keyframes available.\n";
        return;
    }
    if (!result_path.parent_path().empty())
    {
        fs::create_directories(result_path.parent_path());
    }

    auto centers_cpu = voxel_model_->voxCenter().detach().to(torch::kCPU).contiguous();
    if (!centers_cpu.defined() || centers_cpu.numel() == 0)
    {
        std::cout << "[saveRenderedTsdfMeshPly] skipped: no voxels available.\n";
        return;
    }

    auto min_res = centers_cpu.min(0, false);
    auto max_res = centers_cpu.max(0, false);
    auto centers_min = std::get<0>(min_res);
    auto centers_max = std::get<0>(max_res);
    Eigen::Vector3f bb_min(
        centers_min.index({0}).item<float>(),
        centers_min.index({1}).item<float>(),
        centers_min.index({2}).item<float>());
    Eigen::Vector3f bb_max(
        centers_max.index({0}).item<float>(),
        centers_max.index({1}).item<float>(),
        centers_max.index({2}).item<float>());

    for (const auto& [_, pkf] : keyframes)
    {
        if (!pkf) continue;
        const Sophus::SE3f Twc = pkf->getPosef().inverse();
        bb_min = bb_min.cwiseMin(Twc.translation());
        bb_max = bb_max.cwiseMax(Twc.translation());
    }

    const Eigen::Vector3f margin = Eigen::Vector3f::Constant(3.0f * kCommonEvalSdfTrunc);
    bb_min -= margin;
    bb_max += margin;
    const Eigen::Vector3f extent = bb_max - bb_min;
    SparseTsdfVolume volume(kCommonEvalVoxelLength, kCommonEvalSdfTrunc);

    py::gil_scoped_acquire gil;
    py::object py_svm = voxel_model_->svm();
    if (py_svm.is_none())
    {
        std::cout << "[saveRenderedTsdfMeshPly] Python voxel model unavailable, skipping.\n";
        return;
    }

    static py::module_ torch_mod = py::module_::import("torch");
    py::object py_cuda = torch_mod.attr("device")("cuda");
    auto move_attr_to_cuda_if_tensor = [&](py::object& obj, const char* name) {
        if (py::hasattr(obj, name)) {
            py::object t = obj.attr(name);
            if (py::hasattr(t, "is_cuda") && !py::bool_(t.attr("is_cuda"))) {
                obj.attr(name) = t.attr("to")(py_cuda);
            }
        }
    };

    std::cout << "[saveRenderedTsdfMeshPly] begin kfs=" << keyframes.size()
              << " backend=cpp_sparse_tsdf"
              << " voxel_length=" << std::fixed << std::setprecision(8) << kCommonEvalVoxelLength
              << " sdf_trunc=" << kCommonEvalSdfTrunc
              << " depth_trunc=" << kCommonEvalDepthTrunc
              << " bounds_min=(" << bb_min.x() << "," << bb_min.y() << "," << bb_min.z() << ")"
              << " bounds_max=(" << bb_max.x() << "," << bb_max.y() << "," << bb_max.z() << ")"
              << " extent=(" << extent.x() << "," << extent.y() << "," << extent.z() << ")"
              << std::endl;

    bool froze_geo = false;
    py::object inference_ctx = torch_mod.attr("inference_mode")();
    try
    {
        torch_mod.attr("cuda").attr("empty_cache")();
        inference_ctx.attr("__enter__")();

        if (py::hasattr(py_svm, "freeze_vox_geo"))
        {
            py_svm.attr("freeze_vox_geo")();
            froze_geo = true;
        }

        std::size_t frame_idx = 0;
        for (const auto& [kfid, pkf] : keyframes)
        {
            if (!pkf) continue;
            if (pkf->image_width_ <= 0 || pkf->image_height_ <= 0 || pkf->intr_.size() < 4) continue;
            const int image_height = pkf->image_height_;
            const int image_width = pkf->image_width_;

            const sv::MiniCam cam = pkf->toMiniCam(pkf->image_height_, pkf->image_width_);
            std::unordered_map<std::string, torch::Tensor> render_pkg =
                voxel_model_->render(
                    cam,
                    image_height,
                    image_width,
                    torch::Tensor(),
                    nullptr,
                    false,
                    1.0f,
                    true,
                    false,
                    true,
                    false,
                    false,
                    sv::RenderOpts{});
            if (render_pkg.empty()) {
                std::cout << "[saveRenderedTsdfMeshPly] render failed for kf=" << kfid
                          << ", skipping frame.\n";
                continue;
            }

            auto normalize_color_chw = [&](torch::Tensor t) {
                t = t.detach().contiguous();
                if (t.dim() == 4 && t.size(0) == 1) t = t.squeeze(0);
                if (t.dim() != 3) {
                    throw std::runtime_error("saveRenderedTsdfMeshPlySparseCpp: unexpected color tensor rank");
                }
                if (t.size(0) == 3 && t.size(1) == image_height && t.size(2) == image_width) {
                    return t.to(torch::kFloat32).contiguous();
                }
                if (t.size(0) == 3 && t.size(1) == image_width && t.size(2) == image_height) {
                    return t.transpose(1, 2).to(torch::kFloat32).contiguous();
                }
                if (t.size(0) == image_height && t.size(1) == image_width && t.size(2) == 3) {
                    return t.permute({2, 0, 1}).to(torch::kFloat32).contiguous();
                }
                if (t.size(0) == image_width && t.size(1) == image_height && t.size(2) == 3) {
                    return t.permute({2, 1, 0}).to(torch::kFloat32).contiguous();
                }
                std::ostringstream oss;
                oss << "saveRenderedTsdfMeshPlySparseCpp: unsupported color shape " << t.sizes();
                throw std::runtime_error(oss.str());
            };
            auto normalize_hw = [&](torch::Tensor t, const char* name) {
                t = t.detach().to(torch::kFloat32).contiguous();
                if (t.dim() == 3 && t.size(0) == 1) t = t.squeeze(0);
                if (t.dim() != 2) {
                    std::ostringstream oss;
                    oss << "saveRenderedTsdfMeshPlySparseCpp: unexpected " << name << " tensor rank";
                    throw std::runtime_error(oss.str());
                }
                if (t.size(0) == image_height && t.size(1) == image_width) {
                    return t.contiguous();
                }
                if (t.size(0) == image_width && t.size(1) == image_height) {
                    return t.transpose(0, 1).contiguous();
                }
                std::ostringstream oss;
                oss << "saveRenderedTsdfMeshPlySparseCpp: unsupported " << name << " shape " << t.sizes();
                throw std::runtime_error(oss.str());
            };

            auto it_color = render_pkg.find("color");
            auto it_depth = render_pkg.find("depth");
            auto it_T = render_pkg.find("T");
            if (it_color == render_pkg.end() || it_depth == render_pkg.end() ||
                it_T == render_pkg.end() ||
                !it_color->second.defined() || !it_depth->second.defined() ||
                !it_T->second.defined()) {
                std::cout << "[saveRenderedTsdfMeshPly] incomplete render package for kf="
                          << kfid << ", skipping frame.\n";
                continue;
            }

            torch::Tensor rendered_color = normalize_color_chw(it_color->second);
            torch::Tensor depth_pkg = it_depth->second.detach().contiguous();
            torch::Tensor rendered_depth;
            if (depth_pkg.dim() == 3 && depth_pkg.size(0) >= 3)
            {
                rendered_depth = normalize_hw(depth_pkg.index({2}), "depth[2]");
            }
            else
            {
                rendered_depth = normalize_hw(depth_pkg, "depth");
            }
            torch::Tensor T_pkg = it_T->second.detach().contiguous();
            torch::Tensor rendered_alpha =
                normalize_hw(1.0f - T_pkg, "T");

            const auto mask_it = undistort_mask_.find(pkf->camera_id_);
            if (mask_it != undistort_mask_.end())
            {
                auto mask = mask_it->second;
                if (mask.dim() == 3) mask = mask.index({0});
                mask = normalize_hw(mask.to(rendered_depth.device()), "undistort_mask");
                rendered_color = rendered_color * mask.unsqueeze(0);
                rendered_depth = rendered_depth * mask;
                rendered_alpha = rendered_alpha * mask;
            }

            rendered_depth = torch::where(
                rendered_alpha > 1e-4f,
                rendered_depth,
                torch::zeros_like(rendered_depth));

            auto color_cpu = rendered_color.clamp(0, 1)
                .mul(255.0f)
                .to(torch::kUInt8)
                .permute({1, 2, 0})
                .contiguous()
                .to(torch::kCPU);
            auto depth_cpu = rendered_depth
                .to(torch::kFloat32)
                .contiguous()
                .to(torch::kCPU);

            cv::Mat color_mat(
                image_height, image_width, CV_8UC3, color_cpu.data_ptr<uint8_t>());
            cv::Mat depth_mat(
                image_height, image_width, CV_32FC1, depth_cpu.data_ptr<float>());

            cv::Mat filtered_depth = filterDepthOutliersLikeGaussianSlam(depth_mat.clone());
            volume.integrate(
                color_mat.clone(),
                filtered_depth,
                pkf->intr_,
                pkf->getPosef(),
                kCommonEvalDepthTrunc);

            ++frame_idx;
            if (frame_idx % 10 == 0 || frame_idx == keyframes.size())
            {
                std::cout << "[saveRenderedTsdfMeshPly] fused " << frame_idx << "/" << keyframes.size()
                          << " keyframes (last_kfid=" << kfid << ")"
                          << " active_voxels=" << volume.voxels.size() << std::endl;
            }
            torch_mod.attr("cuda").attr("empty_cache")();
        }

        auto mesh = volume.extractMesh();
        if (mesh.vertices.empty() || mesh.faces.empty())
        {
            std::cout << "[saveRenderedTsdfMeshPly] extraction produced an empty mesh.\n";
            inference_ctx.attr("__exit__")(py::none(), py::none(), py::none());
            if (froze_geo && py::hasattr(py_svm, "unfreeze_vox_geo"))
            {
                py_svm.attr("unfreeze_vox_geo")();
            }
            return;
        }

        for (auto& v : mesh.vertices) v += kCommonEvalCompensation;

        if (!saveTriangleMeshPly(result_path, mesh))
        {
            throw std::runtime_error("saveRenderedTsdfMeshPly: failed to write mesh PLY");
        }

        inference_ctx.attr("__exit__")(py::none(), py::none(), py::none());
    }
    catch (const py::error_already_set& e)
    {
        try
        {
            inference_ctx.attr("__exit__")(py::none(), py::none(), py::none());
            torch_mod.attr("cuda").attr("empty_cache")();
        }
        catch (...) {}
        if (froze_geo && py::hasattr(py_svm, "unfreeze_vox_geo"))
        {
            py_svm.attr("unfreeze_vox_geo")();
        }
        throw std::runtime_error(
            std::string("saveRenderedTsdfMeshPly C++ sparse TSDF extraction failed: ") + e.what());
    }
    catch (...)
    {
        try
        {
            inference_ctx.attr("__exit__")(py::none(), py::none(), py::none());
            torch_mod.attr("cuda").attr("empty_cache")();
        }
        catch (...) {}
        if (froze_geo && py::hasattr(py_svm, "unfreeze_vox_geo"))
        {
            py_svm.attr("unfreeze_vox_geo")();
        }
        throw;
    }

    if (froze_geo && py::hasattr(py_svm, "unfreeze_vox_geo"))
    {
        py_svm.attr("unfreeze_vox_geo")();
    }

    std::cout << "[saveRenderedTsdfMeshPly] Wrote C++ sparse-TSDF fused mesh to "
              << result_path << "\n";
}

void VoxelMapper::savePly(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    // keyframesToJson(result_dir);
    // saveModelParams(result_dir);

    std::filesystem::path ply_dir = result_dir / "voxel_model";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

    ply_dir = ply_dir / ("iteration_" + std::to_string(getIteration()));
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

    // Reconstructed voxel scene
    voxel_model_->savePly(ply_dir / "voxel_model.ply");
    // Input sparse points (from ORB-SLAM map) for reference
    // voxel_model_->saveSparsePointsPly(result_dir / "input.ply");
}

void VoxelMapper::savePlannerNPZ(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    std::filesystem::path ply_dir = result_dir / "voxel_model";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)
    ply_dir = ply_dir / ("iteration_" + std::to_string(getIteration()));
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

    // Reconstructed voxel scene
    voxel_model_->savePlannerNPZ(ply_dir / "voxel_model");
}

void VoxelMapper::keyframesToJson(const std::filesystem::path&){ }

/* ---------------- runtime getter / setter ---------------- */
// VariableParameters VoxelMapper::getVariableParameters() const
// {
//     std::lock_guard<std::mutex> lk(mutex_status_);
//     VariableParameters p;
//     p.position_lr_init          = position_lr_init_;
//     p.new_keyframe_times_of_use_ = var_params_.new_keyframe_times_of_use_;
//     p.do_inactive_geo_densify   = do_inactive_geo_densify_;
//     p.keep_training = keep_training_;
//     return p;
// }

// void VoxelMapper::setVariableParameters(const VariableParameters& p)
// {
//     std::lock_guard<std::mutex> lk(mutex_status_);
//     /* apply only what VoxelMapper still honours */
//     position_lr_init_                               = p.position_lr_init;
//     new_keyframe_times_of_use_ = p.new_keyframe_times_of_use_;
//     do_inactive_geo_densify_               = p.do_inactive_geo_densify;
//     keep_training_                         = p.keep_training;
// }

cv::Mat VoxelMapper::renderFromPose(
    const Sophus::SE3f &Tcw,
    const int width,
    const int height,
    const bool main_vision)
{
    // Same guard as Photo-SLAM: no rendering before we have something
    if (!initial_mapped_ || getIteration() <= 0) {
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    }

    // Build a temporary keyframe for the viewer pose
    std::shared_ptr<VoxelKeyframe> pkf = std::make_shared<VoxelKeyframe>();
    // pkf->zfar_ = z_far_;   // only if you actually use z_far_ anywhere
    pkf->znear_ = z_near_;

    // Pose
    pkf->setPose(
        Tcw.unit_quaternion().cast<double>(),
        Tcw.translation().cast<double>());

    try {
        // Camera
        sv::Camera& camera = scene_->cameras_.at(viewer_camera_id_);
        pkf->setCameraParams(camera);
        // If your VoxelKeyframe has this (like GaussianKeyframe), call it:
        // pkf->computeTransformTensors();
    }
    catch (const std::out_of_range&) {
        throw std::runtime_error("[VoxelMapper::renderFromPose] KeyFrame Camera not found!");
    }

    // Build MiniCam for the viewer resolution
    sv::MiniCam cam = pkf->toMiniCam(height, width);

    // We don't want gradients in the viewer
    torch::NoGradGuard no_grad;

    // Call voxel_model_->render under the same render mutex
    std::unordered_map<std::string, torch::Tensor> pkg;
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);

        pkg = voxel_model_->render(
            cam,
            height,
            width,
            /* gt_image      */ torch::Tensor(),  // none
            /* color_mode    */ nullptr,
            /* track_max_w   */ false,
            /* ss            */ std::nullopt,
            /* output_depth  */ false,
            /* output_normal */ false,
            /* output_T      */ false,
            /* rand_bg       */ false,
            /* use_auto_exp  */ false,
            sv::RenderOpts{}   // default options
        );
    }

    // Check we actually got a color image
    auto it = pkg.find("color");
    if (it == pkg.end() || !it->second.defined()) {
        // Fallback: black image
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    }

    torch::Tensor color = it->second;  // expected shape [1,3,H,W] or [3,H,W]

    // Masking exactly like GaussianMapper
    torch::Tensor mask;
    if (main_vision) {
        mask = viewer_main_undistort_mask_[pkf->camera_id_];
    } else {
        mask = viewer_sub_undistort_mask_[pkf->camera_id_];
    }

    // Make sure mask is on the same device as color
    if (mask.device() != color.device()) {
        mask = mask.to(color.device());
    }

    // Both should be broadcastable: mask is usually [1,3,H,W] or [3,H,W]
    torch::Tensor masked_image = color * mask;

    // Reuse Photo-SLAM utility to convert to cv::Mat (float32 RGB)
    return tensor_utils::torchTensor2CvMat_Float32(masked_image);
}

// VoxelMapper::~VoxelMapper() {
//     // Explicitly reset any Python or Torch objects that may call Python at destruction
//     voxel_model_.reset();  // Deallocates all tensors and Python wrappers
//     mpSLAM.reset();
// }

int VoxelMapper::getIteration()
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    return iteration_;
}
void VoxelMapper::increaseIteration(const int inc)
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    iteration_ += inc;
}

float VoxelMapper::geoLearningRateInit()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.geo_lr_;
}

float VoxelMapper::sh0LearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.sh0_lr_;
}

float VoxelMapper::shsLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.shs_lr_;
}

float VoxelMapper::lambdaDssim()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.lambda_dssim_;
}

int VoxelMapper::densifyInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.adapt_every_;
}

int VoxelMapper::newKeyframeTimesOfUse()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return new_keyframe_times_of_use_;
}

int VoxelMapper::stableNumIterExistence()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return stable_num_iter_existence_;
}

bool VoxelMapper::isKeepingTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return keep_training_;
}
bool VoxelMapper::isdoingGausPyramidTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return do_gaus_pyramid_training_;
}

bool VoxelMapper::isdoingInactiveGeoDensify()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return inactive_geo_densify_;
}

 void VoxelMapper::setgeoLearningRateInit(const float lr)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     opt_params_.geo_lr_ = lr;
 }
 void VoxelMapper::setsh0LearningRate(const float lr)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     opt_params_.sh0_lr_ = lr;
 }
 void VoxelMapper::setshsLearningRate(const float lr)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     opt_params_.shs_lr_ = lr;
 }
 void VoxelMapper::setLambdaDssim(const float lambda_dssim)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     opt_params_.lambda_dssim_ = lambda_dssim;
 }

 void VoxelMapper::setDensifyInterval(const int interval)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     opt_params_.adapt_every_ = interval;
 }
 void VoxelMapper::setNewKeyframeTimesOfUse(const int times)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     new_keyframe_times_of_use_ = times;
 }
 void VoxelMapper::setStableNumIterExistence(const int niter)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     stable_num_iter_existence_ = niter;
 }
 void VoxelMapper::setKeepTraining(const bool keep)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     keep_training_ = keep;
 }
 void VoxelMapper::setDoGausPyramidTraining(const bool gaus_pyramid)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     do_gaus_pyramid_training_ = gaus_pyramid;
 }
 
 VariableParameters VoxelMapper::getVaribleParameters()
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     VariableParameters params;
     params.geo_lr = opt_params_.geo_lr_;
     params.sh0_lr = opt_params_.sh0_lr_;
     params.shs_lr = opt_params_.shs_lr_;
     params.lambda_dssim = opt_params_.lambda_dssim_;
     params.densify_interval = opt_params_.adapt_every_;
     params.new_kf_times_of_use = new_keyframe_times_of_use_;
     params.stable_num_iter_existence = stable_num_iter_existence_;
     params.keep_training = keep_training_;
     params.do_gaus_pyramid_training = do_gaus_pyramid_training_;
     return params;
 }
 
 void VoxelMapper::setVaribleParameters(const VariableParameters &params)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     opt_params_.geo_lr_ = params.geo_lr;
     opt_params_.sh0_lr_ = params.sh0_lr;
     opt_params_.shs_lr_ = params.shs_lr;
     opt_params_.lambda_dssim_ = params.lambda_dssim;
     opt_params_.adapt_every_ = params.densify_interval;
     new_keyframe_times_of_use_ = params.new_kf_times_of_use;
     stable_num_iter_existence_ = params.stable_num_iter_existence;
     keep_training_ = params.keep_training;
     do_gaus_pyramid_training_ = params.do_gaus_pyramid_training;
 }

// void VoxelMapper::saveVoxelErrorHeatmap(const sv::MiniCam& cam,
//                                         const torch::Tensor& rendered_img, // (1,3,H,W) float[0..1], device=mDevice
//                                         const torch::Tensor& gt_img,       // (1,3,H,W) float[0..1], device=mDevice
//                                         int fid,
//                                         const std::string& base_dir)
// {
//     namespace fs = std::filesystem;
//     const fs::path dir_kf = fs::path(base_dir) / ("kf" + std::to_string(fid));
//     fs::create_directories(dir_kf);
//     torch::NoGradGuard no_grad;

//     const int H = gt_img.size(2);
//     const int W = gt_img.size(3);

//     // 1) per-pixel MSE (current frame)
//     torch::Tensor mse_map = (rendered_img - gt_img).pow(2).mean(1); // (1,H,W) on device
//     mse_map = mse_map.squeeze(0).detach().to(torch::kCPU).contiguous(); // (H,W) CPU

//     // 2) Build approximate per-pixel -> voxel index map
//     auto geom_cpu = approxGeomFromCentersAndSize(
//     cam,
//     voxel_model_->voxCenter(),         // [N,3]
//     voxel_model_->voxSize(),           // [N,1]
//     H, W
// );

//     // 3) Reduce per-pixel error -> per-voxel error
//     const int64_t N = voxel_model_->numVoxels();
//     auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

//     torch::Tensor idx_flat  = geom_cpu.view({-1});                       // (H*W) int64
//     torch::Tensor valid     = idx_flat >= 0;
//     torch::Tensor sel_idx   = idx_flat.index({valid});                   // (K)
//     torch::Tensor err_flat  = mse_map.view({-1}).index({valid});         // (K)

//     torch::Tensor vox_sum   = torch::zeros({N}, opts_f32);               // per-voxel sum
//     torch::Tensor vox_count = torch::zeros({N}, opts_f32);               // per-voxel hit count
//     if (sel_idx.numel() > 0) {
//         vox_sum.index_add_(0, sel_idx, err_flat);
//         vox_count.index_add_(0, sel_idx, torch::ones_like(err_flat));
//     }
//     torch::Tensor vox_err = vox_sum / vox_count.clamp_min(1.0f);         // (N)

//     // 4) Map per-voxel error back to pixels for visualization
//     torch::Tensor safe_idx = idx_flat.clone();
//     safe_idx.masked_fill_(~valid, 0);
//     torch::Tensor pix_err  = vox_err.index_select(0, safe_idx).view({H, W});
//     pix_err.masked_fill_(~valid.view({H, W}), 0);

//     float vmin = pix_err.min().item<float>();
//     float vmax = pix_err.max().item<float>();
//     float range = std::max(1e-8f, vmax - vmin);
//     torch::Tensor H01 = (pix_err - vmin) / range;

//     // 5) Jet map (B,G,R) like before
//     auto R = (1.5f * H01 - 0.5f).clamp(0, 1);
//     auto G = (1.5f - (2 * H01 - 1).abs()).clamp(0, 1);
//     auto B = (0.5f - 1.5f * H01).clamp(0, 1);
//     torch::Tensor rgb = torch::stack({B, G, R}, -1) * 255.0f; // (H,W,3) BGR for OpenCV
//     rgb = rgb.to(torch::kUInt8).cpu().contiguous();

//     cv::Mat img(H, W, CV_8UC3, rgb.data_ptr<uint8_t>());

//     // 6) Legend bar
//     const int LWIDTH = 32;
//     cv::Mat legend(H, LWIDTH, CV_8UC3);
//     for (int y = 0; y < H; ++y) {
//         float val = 1.f - float(y) / float(H - 1);
//         float r = std::clamp( 1.5f*val - 0.5f , 0.f, 1.f),
//               g = std::clamp( 1.5f - std::abs(2*val -1) , 0.f, 1.f),
//               b = std::clamp( 0.5f - 1.5f*val , 0.f, 1.f);
//         legend.row(y).setTo(cv::Vec3b{ uint8_t(255*b), uint8_t(255*g), uint8_t(255*r) });
//     }
//     cv::putText(legend, "high loss (red)", {2, 14},        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255,255,255), 1, cv::LINE_AA);
//     cv::putText(legend, "low loss (blue)", {2, H - 6},     cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255,255,255), 1, cv::LINE_AA);

//     cv::Mat out; cv::hconcat(img, legend, out);

//     const int x0 = W + LWIDTH + 4;
//     auto put = [&](float frac, const std::string& txt)
//     {
//         int y = int((1.f - frac) * (H - 1));
//         cv::putText(out, txt, {x0, y}, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255,255,255), 1, cv::LINE_AA);
//         cv::line(out, {W, y}, {W + LWIDTH - 1, y}, cv::Scalar(255,255,255), 1, cv::LINE_AA);
//     };

//     std::ostringstream ss; ss.setf(std::ios::fixed); ss.precision(3);
//     ss << vmax;                put(1.f,   ss.str()); ss.str(""); ss.clear();
//     ss << vmax - 0.25f*range;  put(0.75f, ss.str()); ss.str(""); ss.clear();
//     ss << vmin + 0.50f*range;  put(0.50f, ss.str()); ss.str(""); ss.clear();
//     ss << vmin + 0.25f*range;  put(0.25f, ss.str()); ss.str(""); ss.clear();
//     ss << vmin;                put(0.f,   ss.str());

//     // 7) Save PNGs
//     std::ostringstream fn;
//     fn << "kf"   << std::setw(4) << std::setfill('0') << fid
//        << "_iter"<< std::setw(6) << std::setfill('0') << iteration_
//        << ".png";
//     cv::imwrite((dir_kf / fn.str()).string(), out);

//     // Save GT image for this KF
//     auto gt = (gt_img.squeeze(0).mul(255.0f).clamp(0.0f,255.0f).to(torch::kUInt8).permute({1,2,0}).cpu().contiguous());
//     cv::Mat gt_mat(H, W, CV_8UC3, gt.data_ptr<uint8_t>());
//     cv::cvtColor(gt_mat, gt_mat, cv::COLOR_RGB2BGR);
//     std::ostringstream fn2;
//     fn2<< "kf"<<std::setw(4)<<std::setfill('0')<<fid
//        << "_iter"<<std::setw(6)<<std::setfill('0')<<iteration_
//        << "_gt.png";
//     cv::imwrite((dir_kf / fn2.str()).string(), gt_mat);
// }

// void VoxelMapper::saveVoxelErrorHeatmap(const sv::MiniCam&  /*cam*/,
//                                         const torch::Tensor& geom,
//                                         const torch::Tensor&  gt_img,
//                                         int                 fid,          // NEW
//                                         const std::string&  base_dir)     // ← e.g. result_dir_/heatmaps
// {
//     namespace fs = std::filesystem;
//     const fs::path dir_kf = fs::path(base_dir) / ("kf" + std::to_string(fid));
//     fs::create_directories(dir_kf);       // <-- makes .../heatmaps/kf<i>
//     torch::NoGradGuard no_grad;

//     /* ------------------------------------------------------------------ *
//      * ❶  error per voxel  →  per-pixel array  (H,W) in [vmin,vmax]        *
//      * ------------------------------------------------------------------ */
//     torch::Tensor vox_err = (voxel_model_->voxel_error_sum_
//                             / voxel_model_->voxel_hit_count_.clamp_min(1))
//                                 .squeeze(1);                                 // (N)

//     const int H = geom.size(0), W = geom.size(1);

//     torch::Tensor idx_flat = geom.view(-1).to(torch::kLong);
//     torch::Tensor valid    = idx_flat >= 0;
//     torch::Tensor safe_idx = idx_flat.clone().masked_fill(~valid, 0);

//     torch::Tensor pix_err  = vox_err.index_select(0, safe_idx)
//                                    .view({H, W});
//     pix_err.masked_fill_(~valid.view({H, W}), 0);

//     float vmin = pix_err.min().item<float>(),
//           vmax = pix_err.max().item<float>(),
//           range= std::max(1e-6f, vmax - vmin);

//     torch::Tensor H01 = (pix_err - vmin) / range;            // → [0,1]

//     /* ------------------------------------------------------------------ *
//      * ❷  Jet colour-map  (same formula as before)                         *
//      * ------------------------------------------------------------------ */
//     auto R = (1.5f * H01 - 0.5f).clamp(0, 1);
//     auto G = (1.5f - (2 * H01 - 1).abs()).clamp(0, 1);
//     auto B = (0.5f - 1.5f * H01).clamp(0, 1);
//     torch::Tensor rgb = torch::stack({B, G, R}, -1) * 255.0f;  // BGR for OpenCV
//     rgb = rgb.to(torch::kUInt8).cpu().contiguous();            // (H,W,3)

//     /* ------------------------------------------------------------------ *
//      * ❸  convert to cv::Mat                                              *
//      * ------------------------------------------------------------------ */
//     cv::Mat img(H, W, CV_8UC3, rgb.data_ptr<uint8_t>());

//     /* ------------------------------------------------------------------ *
//      * ❹  legend bar  (32 px wide)                                         *
//      * ------------------------------------------------------------------ */
//     const int LWIDTH = 32;
//     cv::Mat legend(H, LWIDTH, CV_8UC3);

//     for (int y = 0; y < H; ++y)
//     {
//         float val = 1.f - float(y) / float(H - 1);   // top=max (red), bottom=min (blue)
//         float r = std::clamp( 1.5f*val - 0.5f , 0.f, 1.f),
//               g = std::clamp( 1.5f - std::abs(2*val -1) , 0.f, 1.f),
//               b = std::clamp( 0.5f - 1.5f*val , 0.f, 1.f);
//         cv::Vec3b col{ uint8_t(255*b), uint8_t(255*g), uint8_t(255*r) };
//         legend.row(y).setTo(col);
//     }

//     const std::string lbl_hi = "high loss (red)";
//     const std::string lbl_lo = "low loss (blue)";
//     // near the top of the legend bar:
//     cv::putText(legend, lbl_hi, {2, 14},
//                 cv::FONT_HERSHEY_SIMPLEX, 0.45,
//                 cv::Scalar(255,255,255), 1, cv::LINE_AA);
//     // near the bottom of the legend bar:
//     cv::putText(legend, lbl_lo, {2, H - 6},
//                 cv::FONT_HERSHEY_SIMPLEX, 0.45,
//                 cv::Scalar(255,255,255), 1, cv::LINE_AA);

//     /* ------------------------------------------------------------------ *
//      * ❺  stack data + legend & annotate tick labels                       *
//      * ------------------------------------------------------------------ */
//     cv::Mat out;
//     cv::hconcat(img, legend, out);

//     const int x0 = W + LWIDTH + 4;        // text anchor (pixels from left)
//     auto put = [&](float frac, const std::string& txt)
//     {
//         int y = int((1.f - frac) * (H - 1));
//         cv::putText(out, txt, {x0, y},
//                     cv::FONT_HERSHEY_SIMPLEX, 0.45,
//                     cv::Scalar(255,255,255), 1, cv::LINE_AA);
//         cv::line(out,
//                  {W, y}, {W + LWIDTH - 1, y},
//                  cv::Scalar(255,255,255), 1, cv::LINE_AA);
//     };

//     std::ostringstream ss; ss.setf(std::ios::fixed); ss.precision(3);
//     ss << vmax; put(1.f, ss.str());  ss.str(""); ss.clear();
//     ss << vmax - 0.25f*range; put(0.75f, ss.str()); ss.str(""); ss.clear();
//     ss << vmin + 0.50f*range; put(0.50f, ss.str()); ss.str(""); ss.clear();
//     ss << vmin + 0.25f*range; put(0.25f, ss.str()); ss.str(""); ss.clear();
//     ss << vmin;  put(0.f, ss.str());

//     /* ------------------------------------------------------------------ *
//      * ❻  write png                                                       *
//      * ------------------------------------------------------------------ */
//     std::ostringstream fn;
//     fn << "kf"   << std::setw(4) << std::setfill('0') << fid
//        << "_iter"<< std::setw(6) << std::setfill('0') << iteration_
//        << ".png";

//     cv::imwrite((dir_kf / fn.str()).string(), out);

//         // ————————————— ❼ convert & write GT image —————————————
//     // assume gt_img is (1,3,H,W) float in [0,1]
//     auto gt = (gt_img.squeeze(0).mul(255.0f)
//                    .clamp(0.0f,255.0f)
//                    .to(torch::kUInt8)
//                    .permute({1,2,0})            // H,W,3 RGB
//                    .cpu()
//                    .contiguous());
//     // convert to BGR for OpenCV:
//     cv::Mat gt_mat(H, W, CV_8UC3, gt.data_ptr<uint8_t>());
//     cv::cvtColor(gt_mat, gt_mat, cv::COLOR_RGB2BGR);

//     std::ostringstream fn2;
//     fn2<< "kf"<<std::setw(4)<<std::setfill('0')<<fid
//        << "_iter"<<std::setw(6)<<std::setfill('0')<<iteration_
//        << "_gt.png";
//     cv::imwrite((dir_kf / fn2.str()).string(), gt_mat);
// }
