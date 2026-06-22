#include "include_voxel/voxel_mapper.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "third_party/tinyply/tinyply.h"

namespace py = pybind11;

namespace {

constexpr float kDefaultEvalVoxelLength = 5.0f / 512.0f;
constexpr float kDefaultEvalSdfTrunc = 0.04f;
constexpr float kEvalDepthTrunc = 30.0f;
constexpr int kEvalMedianKernel = 21; // Gaussian-SLAM evaluator.py uses 20; OpenCV requires an odd kernel.
constexpr float kEvalDepthOutlierThreshold = 0.1f;
const Eigen::Vector3f kEvalCompensation(
    0.0f / 512.0f,
    2.5f / 512.0f,
    -2.5f / 512.0f);

struct TriangleMeshRgb
{
    std::vector<Eigen::Vector3f> vertices;
    std::vector<std::array<uint32_t, 3>> faces;
    std::vector<std::array<uint8_t, 3>> colors;
};

cv::Mat filterDepthOutliersLikeGaussianSlam(const cv::Mat& depth_map)
{
    CV_Assert(depth_map.type() == CV_32FC1);
    if (depth_map.empty()) return depth_map.clone();

    cv::Mat median_filtered = depth_map.clone();
    const int num_passes = std::max(1, kEvalMedianKernel / 5);
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
            if (std::abs(d - m) > kEvalDepthOutlierThreshold) out_ptr[x] = m;
        }
    }
    return filtered;
}

bool saveTriangleMeshPly(
    const std::filesystem::path& ply_path,
    const TriangleMeshRgb& mesh,
    bool write_vertex_colors = true)
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
    const bool has_colors =
        write_vertex_colors && mesh.colors.size() == mesh.vertices.size();
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

bool depthMatToMetersForMeshExtraction(const cv::Mat& depth_in, cv::Mat& depth_meters)
{
    if (depth_in.empty()) return false;

    cv::Mat d = depth_in;
    if (d.channels() > 1) cv::extractChannel(d, d, 0);

    if (d.type() == CV_32FC1)
    {
        depth_meters = d;
        return true;
    }
    if (d.type() == CV_16UC1)
    {
        double max_val = 0.0;
        cv::minMaxLoc(d, nullptr, &max_val);
        const double scale = (max_val > 20000.0) ? (1.0 / 6553.5) : (1.0 / 1000.0);
        d.convertTo(depth_meters, CV_32FC1, scale);
        return true;
    }

    d.convertTo(depth_meters, CV_32FC1);
    return true;
}

bool loadReplicaDepthFromRgbPathForMeshExtraction(
    const std::string& rgb_filename,
    cv::Mat& depth_meters)
{
    if (rgb_filename.empty()) return false;

    const std::filesystem::path rgb_path(rgb_filename);
    if (!std::filesystem::exists(rgb_path)) return false;

    const std::string name = rgb_path.filename().string();
    if (name.rfind("frame", 0) != 0) return false;

    const std::filesystem::path parent = rgb_path.parent_path();
    const std::string stem = rgb_path.stem().string();
    const std::string suffix_stem = (stem.size() > 5 ? stem.substr(5) : std::string());
    const std::string suffix_name = name.substr(5);

    std::vector<std::filesystem::path> candidates;
    candidates.push_back(parent / ("depth" + suffix_name));
    if (!suffix_stem.empty())
    {
        candidates.push_back(parent / ("depth" + suffix_stem + ".png"));
        candidates.push_back(parent / ("depth" + suffix_stem + ".exr"));
        candidates.push_back(parent / ("depth" + suffix_stem + ".tiff"));
        candidates.push_back(parent / ("depth" + suffix_stem + ".tif"));
    }

    for (const auto& path : candidates)
    {
        if (!std::filesystem::exists(path)) continue;
        const cv::Mat depth_raw = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
        if (depth_raw.empty()) continue;
        return depthMatToMetersForMeshExtraction(depth_raw, depth_meters);
    }

    return false;
}

bool loadTumDepthFromRgbPathForMeshExtraction(
    const std::string& rgb_filename,
    cv::Mat& depth_meters)
{
    if (rgb_filename.empty()) return false;

    const std::filesystem::path rgb_path(rgb_filename);
    if (!std::filesystem::exists(rgb_path)) return false;

    const std::filesystem::path rgb_dir = rgb_path.parent_path();
    if (rgb_dir.filename() != "rgb") return false;

    const std::filesystem::path dataset_root = rgb_dir.parent_path();
    const std::filesystem::path depth_dir = dataset_root / "depth";
    const std::filesystem::path depth_txt = dataset_root / "depth.txt";
    if (!std::filesystem::exists(depth_dir)) return false;

    const std::string stem = rgb_path.stem().string();
    const std::filesystem::path exact_depth_path = depth_dir / (stem + rgb_path.extension().string());
    if (std::filesystem::exists(exact_depth_path))
    {
        const cv::Mat depth_raw = cv::imread(exact_depth_path.string(), cv::IMREAD_UNCHANGED);
        if (!depth_raw.empty()) return depthMatToMetersForMeshExtraction(depth_raw, depth_meters);
    }

    double rgb_ts = 0.0;
    try
    {
        rgb_ts = std::stod(stem);
    }
    catch (...)
    {
        return false;
    }

    struct TumDepthIndexEntry
    {
        double timestamp = 0.0;
        std::filesystem::path path;
    };

    static std::mutex s_tum_depth_cache_mutex;
    static std::unordered_map<std::string, std::vector<TumDepthIndexEntry>> s_tum_depth_cache;

    std::vector<TumDepthIndexEntry> depth_index;
    {
        std::lock_guard<std::mutex> lock(s_tum_depth_cache_mutex);
        auto it = s_tum_depth_cache.find(dataset_root.string());
        if (it == s_tum_depth_cache.end())
        {
            std::vector<TumDepthIndexEntry> parsed;
            if (std::filesystem::exists(depth_txt))
            {
                std::ifstream in(depth_txt);
                std::string line;
                while (std::getline(in, line))
                {
                    if (line.empty() || line[0] == '#') continue;
                    std::istringstream iss(line);
                    double ts = 0.0;
                    std::string rel_path;
                    if (!(iss >> ts >> rel_path)) continue;
                    std::filesystem::path path = dataset_root / rel_path;
                    if (!std::filesystem::exists(path)) continue;
                    parsed.push_back({ts, path});
                }
            }
            std::sort(
                parsed.begin(),
                parsed.end(),
                [](const TumDepthIndexEntry& a, const TumDepthIndexEntry& b)
                {
                    return a.timestamp < b.timestamp;
                });
            it = s_tum_depth_cache.emplace(dataset_root.string(), std::move(parsed)).first;
        }
        depth_index = it->second;
    }

    if (depth_index.empty()) return false;

    auto lb = std::lower_bound(
        depth_index.begin(),
        depth_index.end(),
        rgb_ts,
        [](const TumDepthIndexEntry& e, double t)
        {
            return e.timestamp < t;
        });

    auto best_it = depth_index.end();
    double best_dt = std::numeric_limits<double>::infinity();
    if (lb != depth_index.end())
    {
        best_it = lb;
        best_dt = std::abs(lb->timestamp - rgb_ts);
    }
    if (lb != depth_index.begin())
    {
        auto prev = std::prev(lb);
        const double prev_dt = std::abs(prev->timestamp - rgb_ts);
        if (prev_dt < best_dt)
        {
            best_it = prev;
            best_dt = prev_dt;
        }
    }

    constexpr double kTumMaxDepthAssocDeltaSec = 0.05;
    if (best_it == depth_index.end() || !(best_dt <= kTumMaxDepthAssocDeltaSec)) return false;

    const cv::Mat depth_raw = cv::imread(best_it->path.string(), cv::IMREAD_UNCHANGED);
    if (depth_raw.empty()) return false;
    return depthMatToMetersForMeshExtraction(depth_raw, depth_meters);
}

bool getKeyframeDepthMetersForMeshExtraction(
    const std::shared_ptr<VoxelKeyframe>& pkf,
    int expected_h,
    int expected_w,
    cv::Mat& depth_meters)
{
    if (!pkf) return false;

    if (!pkf->img_auxiliary_undist_.empty())
    {
        if (!depthMatToMetersForMeshExtraction(pkf->img_auxiliary_undist_, depth_meters)) return false;
    }
    else
    {
        if (!loadReplicaDepthFromRgbPathForMeshExtraction(pkf->img_filename_, depth_meters) &&
            !loadTumDepthFromRgbPathForMeshExtraction(pkf->img_filename_, depth_meters))
        {
            return false;
        }
    }

    if (depth_meters.empty()) return false;

    if (depth_meters.rows != expected_h || depth_meters.cols != expected_w)
    {
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
                Key last_key{
                    std::numeric_limits<int>::min(),
                    std::numeric_limits<int>::min(),
                    std::numeric_limits<int>::min()};

                for (float zv = z_min; zv <= z_max + 1e-6f; zv += step)
                {
                    const Eigen::Vector3f p_cam(rx * zv, ry * zv, zv);
                    const Eigen::Vector3f p_world = Rwc * p_cam + twc;
                    const Eigen::Vector3i idx =
                        (p_world / voxel_length).array().floor().cast<int>().matrix();
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
                        const float fused =
                            (static_cast<float>(voxel.color[c]) * weight_old +
                             static_cast<float>(rgb[c])) / weight_new;
                        voxel.color[c] = static_cast<uint8_t>(
                            std::lround(std::max(0.0f, std::min(255.0f, fused))));
                    }
                    voxel.weight = static_cast<uint16_t>(weight_new);
                }
            }
        }
    }

    TriangleMeshRgb extractMesh(float min_weight) const
    {
        TriangleMeshRgb mesh;
        if (voxels.empty()) return mesh;
        min_weight = std::max(0.0f, min_weight);

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
                any_weight = any_weight || (static_cast<float>(corner_weight[i]) >= min_weight);
            }
            if (!any_weight) continue;

            for (const auto& tet : tetrahedra)
            {
                bool tet_valid = true;
                for (int local_idx = 0; local_idx < 4; ++local_idx)
                {
                    if (static_cast<float>(corner_weight[tet[local_idx]]) < min_weight)
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
                std::sort(
                    order.begin(),
                    order.end(),
                    [&](int lhs, int rhs)
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

} // namespace

void VoxelMapper::saveRenderedTsdfMeshPly(const std::filesystem::path& result_path)
{
    switch (rerun_params_.rendered_mesh_backend_)
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

    const float target_vox_size = std::max(0.01f, voxel_model_->fixedVoxSize());
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

    const float voxel_length =
        std::max(1.0e-6f, std::isfinite(rerun_params_.rendered_mesh_eval_voxel_size_m_)
                              ? rerun_params_.rendered_mesh_eval_voxel_size_m_
                              : kDefaultEvalVoxelLength);
    const float min_weight =
        std::max(0.0f, std::isfinite(rerun_params_.rendered_mesh_eval_min_weight_)
                          ? rerun_params_.rendered_mesh_eval_min_weight_
                          : 1.0e-4f);
    const Eigen::Vector3f margin = Eigen::Vector3f::Constant(3.0f * kDefaultEvalSdfTrunc);
    bb_min -= margin;
    bb_max += margin;
    const Eigen::Vector3f extent = bb_max - bb_min;
    SparseTsdfVolume volume(voxel_length, kDefaultEvalSdfTrunc);

    py::gil_scoped_acquire gil;
    py::object py_svm = voxel_model_->svm();
    if (py_svm.is_none())
    {
        std::cout << "[saveRenderedTsdfMeshPly] Python voxel model unavailable, skipping.\n";
        return;
    }

    static py::module_ torch_mod = py::module_::import("torch");
    py::object py_cuda = torch_mod.attr("device")("cuda");

    std::cout << "[saveRenderedTsdfMeshPly] begin kfs=" << keyframes.size()
              << " backend=cpp_sparse_tsdf"
              << " voxel_length=" << std::fixed << std::setprecision(8) << voxel_length
              << " min_weight=" << min_weight
              << " sdf_trunc=" << kDefaultEvalSdfTrunc
              << " depth_trunc=" << kEvalDepthTrunc
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
                    false,
                    false,
                    false,
                    sv::RenderOpts{});
            if (render_pkg.empty())
            {
                std::cout << "[saveRenderedTsdfMeshPly] render failed for kf=" << kfid
                          << ", skipping frame.\n";
                continue;
            }

            auto normalize_color_chw = [&](torch::Tensor t)
            {
                t = t.detach().contiguous();
                if (t.dim() == 4 && t.size(0) == 1) t = t.squeeze(0);
                if (t.dim() != 3)
                {
                    throw std::runtime_error("saveRenderedTsdfMeshPlySparseCpp: unexpected color tensor rank");
                }
                if (t.size(0) == 3 && t.size(1) == image_height && t.size(2) == image_width)
                {
                    return t.to(torch::kFloat32).contiguous();
                }
                if (t.size(0) == 3 && t.size(1) == image_width && t.size(2) == image_height)
                {
                    return t.transpose(1, 2).to(torch::kFloat32).contiguous();
                }
                if (t.size(0) == image_height && t.size(1) == image_width && t.size(2) == 3)
                {
                    return t.permute({2, 0, 1}).to(torch::kFloat32).contiguous();
                }
                if (t.size(0) == image_width && t.size(1) == image_height && t.size(2) == 3)
                {
                    return t.permute({2, 1, 0}).to(torch::kFloat32).contiguous();
                }
                std::ostringstream oss;
                oss << "saveRenderedTsdfMeshPlySparseCpp: unsupported color shape " << t.sizes();
                throw std::runtime_error(oss.str());
            };
            auto normalize_hw = [&](torch::Tensor t, const char* name)
            {
                t = t.detach().to(torch::kFloat32).contiguous();
                if (t.dim() == 3 && t.size(0) == 1) t = t.squeeze(0);
                if (t.dim() != 2)
                {
                    std::ostringstream oss;
                    oss << "saveRenderedTsdfMeshPlySparseCpp: unexpected " << name << " tensor rank";
                    throw std::runtime_error(oss.str());
                }
                if (t.size(0) == image_height && t.size(1) == image_width)
                {
                    return t.contiguous();
                }
                if (t.size(0) == image_width && t.size(1) == image_height)
                {
                    return t.transpose(0, 1).contiguous();
                }
                std::ostringstream oss;
                oss << "saveRenderedTsdfMeshPlySparseCpp: unsupported " << name << " shape " << t.sizes();
                throw std::runtime_error(oss.str());
            };

            auto it_color = render_pkg.find("color");
            auto it_depth = render_pkg.find("depth");
            if (it_color == render_pkg.end() || it_depth == render_pkg.end() ||
                !it_color->second.defined() || !it_depth->second.defined())
            {
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

            const auto mask_it = undistort_mask_.find(pkf->camera_id_);
            if (mask_it != undistort_mask_.end())
            {
                auto mask = mask_it->second;
                if (mask.dim() == 3) mask = mask.index({0});
                mask = normalize_hw(mask.to(rendered_depth.device()), "undistort_mask");
                rendered_color = rendered_color * mask.unsqueeze(0);
                rendered_depth = rendered_depth * mask;
            }

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

            cv::Mat depth_for_fusion = depth_mat.clone();
            cv::Mat gt_depth_meters;
            if (getKeyframeDepthMetersForMeshExtraction(pkf, image_height, image_width, gt_depth_meters))
            {
                for (int y = 0; y < image_height; ++y)
                {
                    const float* gt_ptr = gt_depth_meters.ptr<float>(y);
                    float* depth_ptr = depth_for_fusion.ptr<float>(y);
                    for (int x = 0; x < image_width; ++x)
                    {
                        const float gt_depth = gt_ptr[x];
                        if (!std::isfinite(gt_depth) || gt_depth <= 0.0f)
                        {
                            depth_ptr[x] = 0.0f;
                        }
                    }
                }
            }

            cv::Mat filtered_depth = filterDepthOutliersLikeGaussianSlam(depth_for_fusion);
            volume.integrate(
                color_mat.clone(),
                filtered_depth,
                pkf->intr_,
                pkf->getPosef(),
                kEvalDepthTrunc);

            ++frame_idx;
            if (frame_idx % 10 == 0 || frame_idx == keyframes.size())
            {
                std::cout << "[saveRenderedTsdfMeshPly] fused " << frame_idx << "/" << keyframes.size()
                          << " keyframes (last_kfid=" << kfid << ")"
                          << " active_voxels=" << volume.voxels.size() << std::endl;
            }
            torch_mod.attr("cuda").attr("empty_cache")();
        }

        auto mesh = volume.extractMesh(min_weight);
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

        for (auto& v : mesh.vertices) v += kEvalCompensation;

        if (!saveTriangleMeshPly(result_path, mesh))
        {
            throw std::runtime_error("saveRenderedTsdfMeshPly: failed to write mesh PLY");
        }

        const std::filesystem::path nocolor_path =
            result_path.parent_path() /
            (result_path.stem().string() + "_nocolor" + result_path.extension().string());
        if (!saveTriangleMeshPly(nocolor_path, mesh, false))
        {
            throw std::runtime_error("saveRenderedTsdfMeshPly: failed to write no-color mesh PLY");
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
              << result_path << " and "
              << (result_path.parent_path() /
                  (result_path.stem().string() + "_nocolor" + result_path.extension().string()))
              << "\n";
}
