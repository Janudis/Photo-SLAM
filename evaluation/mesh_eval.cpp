#include <opencv2/core.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <jsoncpp/json/json.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <cstring>
#include <type_traits>
#include <vector>

#include "third_party/tinyply/tinyply.h"

namespace
{
struct MeshData
{
    std::vector<cv::Point3f> vertices;
    std::vector<cv::Vec3i> faces;
    std::vector<cv::Vec3b> colors;
    // Optional original 3DGS parameters. These remain in their saved raw form:
    // opacity logits, log-scales, and wxyz quaternions.
    std::vector<float> gaussian_opacity;
    std::vector<cv::Vec3f> gaussian_log_scales;
    std::vector<cv::Vec4f> gaussian_rotations;
    // Optional SVRecon voxel-model parameters. Vertex positions are voxel
    // centers; octlevel and scene_extent determine each cube's edge length.
    std::vector<uint8_t> voxel_octlevels;
    std::vector<std::array<float, 8>> voxel_sdf_corners;
    float voxel_scene_extent = std::numeric_limits<float>::quiet_NaN();

    bool hasGaussianAttributes() const
    {
        return !vertices.empty() &&
               gaussian_opacity.size() == vertices.size() &&
               gaussian_log_scales.size() == vertices.size() &&
               gaussian_rotations.size() == vertices.size();
    }

    bool hasVoxelAttributes() const
    {
        return !vertices.empty() &&
               voxel_octlevels.size() == vertices.size() &&
               voxel_sdf_corners.size() == vertices.size() &&
               std::isfinite(voxel_scene_extent) && voxel_scene_extent > 0.0f;
    }
};

struct FloaterThresholdStats
{
    float threshold_m = 0.0f;
    uint64_t farther_count = 0;
    double farther_ratio = 0.0;
};

struct FloaterSummary
{
    uint64_t count = 0;
    double mean_m = 0.0;
    double p95_m = 0.0;
    double p99_m = 0.0;
    std::vector<FloaterThresholdStats> thresholds;
};

struct CameraIntrinsics
{
    int w = 0;
    int h = 0;
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;

    bool valid() const
    {
        return w > 0 && h > 0 && fx > 0.0f && fy > 0.0f;
    }
};

struct DepthEvalStats
{
    uint64_t frames_used = 0;
    uint64_t pred_pixels = 0;
    uint64_t gt_pixels = 0;
    uint64_t both_pixels = 0;
    uint64_t tp_pixels = 0;
    double abs_l1_sum = 0.0;

    double depth_l1_m = -1.0;
    double depth_precision = -1.0;
    double depth_recall = -1.0;
    double depth_f1 = -1.0;
};

struct DepthHeatmapSettings
{
    bool enabled = false;
    std::filesystem::path out_dir;
    int max_saved_frames = 0;     // 0 = save all evaluated frames
    float clip_max_m = 0.0f;      // optional upper clamp before per-image min/max normalization
};

struct DepthHeatmapVizRange
{
    bool valid = false;
    float min_m = 0.0f;
    float max_m = 0.0f;
};

enum class EvalMode
{
    Current = 0,
    GaussianSlam = 1,
    GaussianSlamSim3 = 2
};

struct GaussianSlamDepthSettings
{
    int width = 500;
    int height = 500;
    float focal = 300.0f;
    int n_views = 1000;
    float near_z = 0.05f;
    float far_z = 20.0f;
    float icp_threshold_m = 0.1f;
    int icp_max_iters = 20;
    int icp_max_points = 100000;
};

enum class PlyScalarType
{
    INVALID = 0,
    INT8,
    UINT8,
    INT16,
    UINT16,
    INT32,
    UINT32,
    FLOAT32,
    FLOAT64
};

struct PlyVertexProp
{
    PlyScalarType type = PlyScalarType::INVALID;
    std::string name;
    size_t offset = 0;
};

PlyScalarType parsePlyScalarType(const std::string& t)
{
    if (t == "char" || t == "int8") return PlyScalarType::INT8;
    if (t == "uchar" || t == "uint8") return PlyScalarType::UINT8;
    if (t == "short" || t == "int16") return PlyScalarType::INT16;
    if (t == "ushort" || t == "uint16") return PlyScalarType::UINT16;
    if (t == "int" || t == "int32") return PlyScalarType::INT32;
    if (t == "uint" || t == "uint32") return PlyScalarType::UINT32;
    if (t == "float" || t == "float32") return PlyScalarType::FLOAT32;
    if (t == "double" || t == "float64") return PlyScalarType::FLOAT64;
    return PlyScalarType::INVALID;
}

size_t plyScalarTypeSize(PlyScalarType t)
{
    switch (t)
    {
        case PlyScalarType::INT8:
        case PlyScalarType::UINT8: return 1;
        case PlyScalarType::INT16:
        case PlyScalarType::UINT16: return 2;
        case PlyScalarType::INT32:
        case PlyScalarType::UINT32:
        case PlyScalarType::FLOAT32: return 4;
        case PlyScalarType::FLOAT64: return 8;
        default: return 0;
    }
}

double readPlyScalarAsDouble(const char* p, PlyScalarType t)
{
    switch (t)
    {
        case PlyScalarType::INT8:   return static_cast<double>(*reinterpret_cast<const int8_t*>(p));
        case PlyScalarType::UINT8:  return static_cast<double>(*reinterpret_cast<const uint8_t*>(p));
        case PlyScalarType::INT16:  { int16_t v; std::memcpy(&v, p, sizeof(v)); return static_cast<double>(v); }
        case PlyScalarType::UINT16: { uint16_t v; std::memcpy(&v, p, sizeof(v)); return static_cast<double>(v); }
        case PlyScalarType::INT32:  { int32_t v; std::memcpy(&v, p, sizeof(v)); return static_cast<double>(v); }
        case PlyScalarType::UINT32: { uint32_t v; std::memcpy(&v, p, sizeof(v)); return static_cast<double>(v); }
        case PlyScalarType::FLOAT32:{ float v; std::memcpy(&v, p, sizeof(v)); return static_cast<double>(v); }
        case PlyScalarType::FLOAT64:{ double v; std::memcpy(&v, p, sizeof(v)); return v; }
        default: return 0.0;
    }
}

template <typename T>
uint8_t colorComponentToU8(T value)
{
    double v = static_cast<double>(value);
    if constexpr (std::is_floating_point_v<T>)
    {
        if (!std::isfinite(v)) v = 0.0;
        else if (v >= 0.0 && v <= 1.0) v *= 255.0;
    }
    const long rounded = std::lround(v);
    return static_cast<uint8_t>(std::clamp<long>(rounded, 0L, 255L));
}

uint8_t readPlyScalarAsU8Color(const char* p, PlyScalarType t)
{
    switch (t)
    {
        case PlyScalarType::INT8:   return colorComponentToU8(*reinterpret_cast<const int8_t*>(p));
        case PlyScalarType::UINT8:  return colorComponentToU8(*reinterpret_cast<const uint8_t*>(p));
        case PlyScalarType::INT16:  { int16_t v; std::memcpy(&v, p, sizeof(v)); return colorComponentToU8(v); }
        case PlyScalarType::UINT16: { uint16_t v; std::memcpy(&v, p, sizeof(v)); return colorComponentToU8(v); }
        case PlyScalarType::INT32:  { int32_t v; std::memcpy(&v, p, sizeof(v)); return colorComponentToU8(v); }
        case PlyScalarType::UINT32: { uint32_t v; std::memcpy(&v, p, sizeof(v)); return colorComponentToU8(v); }
        case PlyScalarType::FLOAT32:{ float v; std::memcpy(&v, p, sizeof(v)); return colorComponentToU8(v); }
        case PlyScalarType::FLOAT64:{ double v; std::memcpy(&v, p, sizeof(v)); return colorComponentToU8(v); }
        default: return 0;
    }
}

bool parseSceneExtentComment(const std::string& comment, float& scene_extent)
{
    std::istringstream iss(comment);
    std::string key;
    float value = 0.0f;
    if (!(iss >> key >> value) || key != "scene_extent" ||
        !std::isfinite(value) || !(value > 0.0f))
    {
        return false;
    }
    scene_extent = value;
    return true;
}

bool loadPlyMeshBinaryFallback(const std::string& ply_path, MeshData& mesh)
{
    std::ifstream in(ply_path, std::ios::binary);
    if (!in.is_open()) return false;

    std::string line;
    std::string format;
    size_t vertex_count = 0;
    size_t face_count = 0;
    std::string current_element;
    std::vector<PlyVertexProp> vertex_props;
    PlyScalarType face_count_type = PlyScalarType::INVALID;
    PlyScalarType face_index_type = PlyScalarType::INVALID;
    bool have_face_indices = false;
    float voxel_scene_extent = std::numeric_limits<float>::quiet_NaN();

    while (std::getline(in, line))
    {
        if (line == "end_header") break;
        if (line.rfind("comment ", 0) == 0)
        {
            parseSceneExtentComment(line.substr(8), voxel_scene_extent);
        }
        else if (line.rfind("format ", 0) == 0)
        {
            std::istringstream iss(line);
            std::string tok;
            iss >> tok >> format;
        }
        else if (line.rfind("element ", 0) == 0)
        {
            std::istringstream iss(line);
            std::string tok;
            iss >> tok >> current_element;
            size_t count = 0;
            iss >> count;
            if (current_element == "vertex") vertex_count = count;
            else if (current_element == "face") face_count = count;
        }
        else if (line.rfind("property ", 0) == 0)
        {
            std::istringstream iss(line);
            std::string tok, kind;
            iss >> tok >> kind;
            if (kind == "list")
            {
                std::string cnt_t, idx_t, name;
                iss >> cnt_t >> idx_t >> name;
                if (current_element == "face" && name == "vertex_indices")
                {
                    face_count_type = parsePlyScalarType(cnt_t);
                    face_index_type = parsePlyScalarType(idx_t);
                    have_face_indices = true;
                }
            }
            else
            {
                std::string name;
                iss >> name;
                if (current_element == "vertex")
                {
                    PlyVertexProp p;
                    p.type = parsePlyScalarType(kind);
                    p.name = name;
                    vertex_props.push_back(p);
                }
            }
        }
    }

    if (format != "binary_little_endian")
    {
        return false;
    }
    if (vertex_count == 0 || vertex_props.empty()) return false;

    size_t vertex_stride = 0;
    int x_prop = -1;
    int y_prop = -1;
    int z_prop = -1;
    int r_prop = -1;
    int g_prop = -1;
    int b_prop = -1;
    int opacity_prop = -1;
    int octlevel_prop = -1;
    std::array<int, 3> scale_props{{-1, -1, -1}};
    std::array<int, 4> rotation_props{{-1, -1, -1, -1}};
    std::array<int, 8> sdf_props{{-1, -1, -1, -1, -1, -1, -1, -1}};
    for (int i = 0; i < static_cast<int>(vertex_props.size()); ++i)
    {
        vertex_props[i].offset = vertex_stride;
        const size_t s = plyScalarTypeSize(vertex_props[i].type);
        if (s == 0) return false;
        vertex_stride += s;
        if (vertex_props[i].name == "x") x_prop = i;
        else if (vertex_props[i].name == "y") y_prop = i;
        else if (vertex_props[i].name == "z") z_prop = i;
        else if (vertex_props[i].name == "red" || vertex_props[i].name == "r") r_prop = i;
        else if (vertex_props[i].name == "green" || vertex_props[i].name == "g") g_prop = i;
        else if (vertex_props[i].name == "blue" || vertex_props[i].name == "b") b_prop = i;
        else if (vertex_props[i].name == "opacity") opacity_prop = i;
        else if (vertex_props[i].name == "octlevel") octlevel_prop = i;
        else if (vertex_props[i].name == "scale_0") scale_props[0] = i;
        else if (vertex_props[i].name == "scale_1") scale_props[1] = i;
        else if (vertex_props[i].name == "scale_2") scale_props[2] = i;
        else if (vertex_props[i].name == "rot_0") rotation_props[0] = i;
        else if (vertex_props[i].name == "rot_1") rotation_props[1] = i;
        else if (vertex_props[i].name == "rot_2") rotation_props[2] = i;
        else if (vertex_props[i].name == "rot_3") rotation_props[3] = i;
        else
        {
            for (int corner = 0; corner < 8; ++corner)
            {
                if (vertex_props[i].name ==
                    "grid" + std::to_string(corner) + "_value")
                {
                    sdf_props[corner] = i;
                    break;
                }
            }
        }
    }
    if (x_prop < 0 || y_prop < 0 || z_prop < 0) return false;
    const bool has_colors = r_prop >= 0 && g_prop >= 0 && b_prop >= 0;
    const bool has_gaussian_attributes =
        opacity_prop >= 0 &&
        std::all_of(scale_props.begin(), scale_props.end(), [](int p) { return p >= 0; }) &&
        std::all_of(rotation_props.begin(), rotation_props.end(), [](int p) { return p >= 0; });
    const bool has_voxel_attributes =
        octlevel_prop >= 0 && std::isfinite(voxel_scene_extent) && voxel_scene_extent > 0.0f &&
        std::all_of(sdf_props.begin(), sdf_props.end(), [](int p) { return p >= 0; });

    mesh.vertices.clear();
    mesh.faces.clear();
    mesh.colors.clear();
    mesh.gaussian_opacity.clear();
    mesh.gaussian_log_scales.clear();
    mesh.gaussian_rotations.clear();
    mesh.voxel_octlevels.clear();
    mesh.voxel_sdf_corners.clear();
    mesh.voxel_scene_extent = std::numeric_limits<float>::quiet_NaN();
    mesh.vertices.reserve(vertex_count);
    if (has_colors) mesh.colors.reserve(vertex_count);
    if (has_gaussian_attributes)
    {
        mesh.gaussian_opacity.reserve(vertex_count);
        mesh.gaussian_log_scales.reserve(vertex_count);
        mesh.gaussian_rotations.reserve(vertex_count);
    }
    if (has_voxel_attributes)
    {
        mesh.voxel_octlevels.reserve(vertex_count);
        mesh.voxel_sdf_corners.reserve(vertex_count);
        mesh.voxel_scene_extent = voxel_scene_extent;
    }
    std::vector<char> vb(vertex_stride);
    for (size_t i = 0; i < vertex_count; ++i)
    {
        in.read(vb.data(), static_cast<std::streamsize>(vertex_stride));
        if (!in) return false;

        const auto read_prop = [&](int pi) -> float
        {
            const PlyVertexProp& p = vertex_props[pi];
            return static_cast<float>(readPlyScalarAsDouble(vb.data() + p.offset, p.type));
        };

        mesh.vertices.emplace_back(read_prop(x_prop), read_prop(y_prop), read_prop(z_prop));
        if (has_gaussian_attributes)
        {
            mesh.gaussian_opacity.push_back(read_prop(opacity_prop));
            mesh.gaussian_log_scales.emplace_back(
                read_prop(scale_props[0]),
                read_prop(scale_props[1]),
                read_prop(scale_props[2]));
            mesh.gaussian_rotations.emplace_back(
                read_prop(rotation_props[0]),
                read_prop(rotation_props[1]),
                read_prop(rotation_props[2]),
                read_prop(rotation_props[3]));
        }
        if (has_voxel_attributes)
        {
            const int level = static_cast<int>(std::lround(read_prop(octlevel_prop)));
            mesh.voxel_octlevels.push_back(
                static_cast<uint8_t>(std::clamp(level, 0, 255)));
            std::array<float, 8> sdf{};
            for (int corner = 0; corner < 8; ++corner)
            {
                sdf[corner] = read_prop(sdf_props[corner]);
            }
            mesh.voxel_sdf_corners.push_back(sdf);
        }
        if (has_colors)
        {
            const auto read_color = [&](int pi) -> uint8_t
            {
                const PlyVertexProp& p = vertex_props[pi];
                return readPlyScalarAsU8Color(vb.data() + p.offset, p.type);
            };
            mesh.colors.emplace_back(read_color(r_prop), read_color(g_prop), read_color(b_prop));
        }
    }

    if (!have_face_indices || face_count == 0) return true;
    const size_t count_size = plyScalarTypeSize(face_count_type);
    const size_t index_size = plyScalarTypeSize(face_index_type);
    if (count_size == 0 || index_size == 0) return false;

    for (size_t i = 0; i < face_count; ++i)
    {
        std::vector<char> cb(count_size);
        in.read(cb.data(), static_cast<std::streamsize>(count_size));
        if (!in) return false;
        const int n = static_cast<int>(readPlyScalarAsDouble(cb.data(), face_count_type));
        if (n <= 0) continue;

        std::vector<int> ids;
        ids.reserve(static_cast<size_t>(n));
        std::vector<char> ib(index_size);
        for (int k = 0; k < n; ++k)
        {
            in.read(ib.data(), static_cast<std::streamsize>(index_size));
            if (!in) return false;
            const int idx = static_cast<int>(readPlyScalarAsDouble(ib.data(), face_index_type));
            ids.push_back(idx);
        }

        if (ids.size() < 3) continue;
        for (size_t k = 1; k + 1 < ids.size(); ++k)
        {
            mesh.faces.emplace_back(ids[0], ids[k], ids[k + 1]);
        }
    }

    return true;
}

template <typename T>
void copyVerticesXYZ(const std::shared_ptr<tinyply::PlyData>& xyz, std::vector<cv::Point3f>& out)
{
    const T* ptr = reinterpret_cast<const T*>(xyz->buffer.get());
    out.resize(xyz->count);
    for (size_t i = 0; i < xyz->count; ++i)
    {
        out[i].x = static_cast<float>(ptr[3 * i + 0]);
        out[i].y = static_cast<float>(ptr[3 * i + 1]);
        out[i].z = static_cast<float>(ptr[3 * i + 2]);
    }
}

template <typename T>
void copyVertexColorsRGB(const std::shared_ptr<tinyply::PlyData>& rgb, std::vector<cv::Vec3b>& out)
{
    const T* ptr = reinterpret_cast<const T*>(rgb->buffer.get());
    out.resize(rgb->count);
    for (size_t i = 0; i < rgb->count; ++i)
    {
        out[i][0] = colorComponentToU8(ptr[3 * i + 0]);
        out[i][1] = colorComponentToU8(ptr[3 * i + 1]);
        out[i][2] = colorComponentToU8(ptr[3 * i + 2]);
    }
}

template <typename T>
void copyVertexScalars(const std::shared_ptr<tinyply::PlyData>& data, std::vector<float>& out)
{
    const T* ptr = reinterpret_cast<const T*>(data->buffer.get());
    out.resize(data->count);
    for (size_t i = 0; i < data->count; ++i)
    {
        out[i] = static_cast<float>(ptr[i]);
    }
}

template <typename T>
void copyVertexVec3(const std::shared_ptr<tinyply::PlyData>& data, std::vector<cv::Vec3f>& out)
{
    const T* ptr = reinterpret_cast<const T*>(data->buffer.get());
    out.resize(data->count);
    for (size_t i = 0; i < data->count; ++i)
    {
        out[i] = cv::Vec3f(
            static_cast<float>(ptr[3 * i + 0]),
            static_cast<float>(ptr[3 * i + 1]),
            static_cast<float>(ptr[3 * i + 2]));
    }
}

template <typename T>
void copyVertexVec4(const std::shared_ptr<tinyply::PlyData>& data, std::vector<cv::Vec4f>& out)
{
    const T* ptr = reinterpret_cast<const T*>(data->buffer.get());
    out.resize(data->count);
    for (size_t i = 0; i < data->count; ++i)
    {
        out[i] = cv::Vec4f(
            static_cast<float>(ptr[4 * i + 0]),
            static_cast<float>(ptr[4 * i + 1]),
            static_cast<float>(ptr[4 * i + 2]),
            static_cast<float>(ptr[4 * i + 3]));
    }
}

template <typename T>
void copyVertexLevels(const std::shared_ptr<tinyply::PlyData>& data, std::vector<uint8_t>& out)
{
    const T* ptr = reinterpret_cast<const T*>(data->buffer.get());
    out.resize(data->count);
    for (size_t i = 0; i < data->count; ++i)
    {
        const int level = static_cast<int>(std::lround(static_cast<double>(ptr[i])));
        out[i] = static_cast<uint8_t>(std::clamp(level, 0, 255));
    }
}

template <typename T>
void copyVertexSdfCorners(
    const std::shared_ptr<tinyply::PlyData>& data,
    std::vector<std::array<float, 8>>& out)
{
    const T* ptr = reinterpret_cast<const T*>(data->buffer.get());
    out.resize(data->count);
    for (size_t i = 0; i < data->count; ++i)
    {
        for (int corner = 0; corner < 8; ++corner)
        {
            out[i][corner] = static_cast<float>(ptr[8 * i + corner]);
        }
    }
}

template <typename T>
void copyFacesIdx3(const std::shared_ptr<tinyply::PlyData>& face_idx, std::vector<cv::Vec3i>& out)
{
    const T* ptr = reinterpret_cast<const T*>(face_idx->buffer.get());
    out.resize(face_idx->count);
    for (size_t i = 0; i < face_idx->count; ++i)
    {
        out[i][0] = static_cast<int>(ptr[3 * i + 0]);
        out[i][1] = static_cast<int>(ptr[3 * i + 1]);
        out[i][2] = static_cast<int>(ptr[3 * i + 2]);
    }
}

bool loadPlyMesh(const std::string& ply_path, MeshData& mesh)
{
    try
    {
        std::ifstream in(ply_path, std::ios::binary);
        if (!in.is_open())
        {
            std::cerr << "[mesh_eval] failed to open: " << ply_path << "\n";
            return false;
        }

        tinyply::PlyFile ply_file;
        ply_file.parse_header(in);

        std::shared_ptr<tinyply::PlyData> xyz;
        std::shared_ptr<tinyply::PlyData> face_idx;
        std::shared_ptr<tinyply::PlyData> rgb;
        std::shared_ptr<tinyply::PlyData> gaussian_opacity;
        std::shared_ptr<tinyply::PlyData> gaussian_scales;
        std::shared_ptr<tinyply::PlyData> gaussian_rotations;
        std::shared_ptr<tinyply::PlyData> voxel_octlevels;
        std::shared_ptr<tinyply::PlyData> voxel_sdf_corners;

        mesh.voxel_scene_extent = std::numeric_limits<float>::quiet_NaN();
        for (const auto& comment : ply_file.get_comments())
        {
            parseSceneExtentComment(comment, mesh.voxel_scene_extent);
        }

        try { xyz = ply_file.request_properties_from_element("vertex", { "x", "y", "z" }); }
        catch (const std::exception& e)
        {
            std::cerr << "[mesh_eval] tinyply vertex exception: " << e.what() << "\n";
            return false;
        }

        try
        {
            // list_size_hint=3 for triangular meshes
            face_idx = ply_file.request_properties_from_element("face", { "vertex_indices" }, 3);
        }
        catch (...) { face_idx.reset(); }

        try
        {
            rgb = ply_file.request_properties_from_element("vertex", { "red", "green", "blue" });
        }
        catch (...)
        {
            try
            {
                rgb = ply_file.request_properties_from_element("vertex", { "r", "g", "b" });
            }
            catch (...) { rgb.reset(); }
        }

        try { gaussian_opacity = ply_file.request_properties_from_element("vertex", {"opacity"}); }
        catch (...) { gaussian_opacity.reset(); }
        try
        {
            gaussian_scales = ply_file.request_properties_from_element(
                "vertex", {"scale_0", "scale_1", "scale_2"});
        }
        catch (...) { gaussian_scales.reset(); }
        try
        {
            gaussian_rotations = ply_file.request_properties_from_element(
                "vertex", {"rot_0", "rot_1", "rot_2", "rot_3"});
        }
        catch (...) { gaussian_rotations.reset(); }
        try
        {
            voxel_octlevels = ply_file.request_properties_from_element(
                "vertex", {"octlevel"});
        }
        catch (...) { voxel_octlevels.reset(); }
        try
        {
            voxel_sdf_corners = ply_file.request_properties_from_element(
                "vertex",
                {"grid0_value", "grid1_value", "grid2_value", "grid3_value",
                 "grid4_value", "grid5_value", "grid6_value", "grid7_value"});
        }
        catch (...) { voxel_sdf_corners.reset(); }

        ply_file.read(in);
        if (!xyz || xyz->count == 0)
        {
            std::cerr << "[mesh_eval] no vertices in: " << ply_path << "\n";
            return false;
        }

        switch (xyz->t)
        {
            case tinyply::Type::FLOAT32: copyVerticesXYZ<float>(xyz, mesh.vertices); break;
            case tinyply::Type::FLOAT64: copyVerticesXYZ<double>(xyz, mesh.vertices); break;
            case tinyply::Type::INT32:   copyVerticesXYZ<int32_t>(xyz, mesh.vertices); break;
            case tinyply::Type::UINT32:  copyVerticesXYZ<uint32_t>(xyz, mesh.vertices); break;
            case tinyply::Type::INT16:   copyVerticesXYZ<int16_t>(xyz, mesh.vertices); break;
            case tinyply::Type::UINT16:  copyVerticesXYZ<uint16_t>(xyz, mesh.vertices); break;
            case tinyply::Type::INT8:    copyVerticesXYZ<int8_t>(xyz, mesh.vertices); break;
            case tinyply::Type::UINT8:   copyVerticesXYZ<uint8_t>(xyz, mesh.vertices); break;
            default:
                std::cerr << "[mesh_eval] unsupported vertex type in: " << ply_path << "\n";
                return false;
        }

        mesh.colors.clear();
        if (rgb && rgb->count == xyz->count)
        {
            switch (rgb->t)
            {
                case tinyply::Type::FLOAT32: copyVertexColorsRGB<float>(rgb, mesh.colors); break;
                case tinyply::Type::FLOAT64: copyVertexColorsRGB<double>(rgb, mesh.colors); break;
                case tinyply::Type::INT32:   copyVertexColorsRGB<int32_t>(rgb, mesh.colors); break;
                case tinyply::Type::UINT32:  copyVertexColorsRGB<uint32_t>(rgb, mesh.colors); break;
                case tinyply::Type::INT16:   copyVertexColorsRGB<int16_t>(rgb, mesh.colors); break;
                case tinyply::Type::UINT16:  copyVertexColorsRGB<uint16_t>(rgb, mesh.colors); break;
                case tinyply::Type::INT8:    copyVertexColorsRGB<int8_t>(rgb, mesh.colors); break;
                case tinyply::Type::UINT8:   copyVertexColorsRGB<uint8_t>(rgb, mesh.colors); break;
                default:
                    std::cerr << "[mesh_eval] unsupported vertex color type in: " << ply_path << "\n";
                    mesh.colors.clear();
                    break;
            }
        }

        mesh.gaussian_opacity.clear();
        mesh.gaussian_log_scales.clear();
        mesh.gaussian_rotations.clear();
        const bool have_gaussian_attributes =
            gaussian_opacity && gaussian_scales && gaussian_rotations &&
            gaussian_opacity->count == xyz->count &&
            gaussian_scales->count == xyz->count &&
            gaussian_rotations->count == xyz->count &&
            gaussian_opacity->t == tinyply::Type::FLOAT32 &&
            gaussian_scales->t == tinyply::Type::FLOAT32 &&
            gaussian_rotations->t == tinyply::Type::FLOAT32;
        if (have_gaussian_attributes)
        {
            copyVertexScalars<float>(gaussian_opacity, mesh.gaussian_opacity);
            copyVertexVec3<float>(gaussian_scales, mesh.gaussian_log_scales);
            copyVertexVec4<float>(gaussian_rotations, mesh.gaussian_rotations);
        }

        mesh.voxel_octlevels.clear();
        mesh.voxel_sdf_corners.clear();
        const bool have_voxel_attributes =
            voxel_octlevels && voxel_sdf_corners &&
            voxel_octlevels->count == xyz->count &&
            voxel_sdf_corners->count == xyz->count &&
            voxel_sdf_corners->t == tinyply::Type::FLOAT32 &&
            std::isfinite(mesh.voxel_scene_extent) && mesh.voxel_scene_extent > 0.0f;
        if (have_voxel_attributes)
        {
            switch (voxel_octlevels->t)
            {
                case tinyply::Type::INT32:   copyVertexLevels<int32_t>(voxel_octlevels, mesh.voxel_octlevels); break;
                case tinyply::Type::UINT32:  copyVertexLevels<uint32_t>(voxel_octlevels, mesh.voxel_octlevels); break;
                case tinyply::Type::INT16:   copyVertexLevels<int16_t>(voxel_octlevels, mesh.voxel_octlevels); break;
                case tinyply::Type::UINT16:  copyVertexLevels<uint16_t>(voxel_octlevels, mesh.voxel_octlevels); break;
                case tinyply::Type::INT8:    copyVertexLevels<int8_t>(voxel_octlevels, mesh.voxel_octlevels); break;
                case tinyply::Type::UINT8:   copyVertexLevels<uint8_t>(voxel_octlevels, mesh.voxel_octlevels); break;
                case tinyply::Type::FLOAT32: copyVertexLevels<float>(voxel_octlevels, mesh.voxel_octlevels); break;
                case tinyply::Type::FLOAT64: copyVertexLevels<double>(voxel_octlevels, mesh.voxel_octlevels); break;
                default: mesh.voxel_octlevels.clear(); break;
            }
            if (!mesh.voxel_octlevels.empty())
            {
                copyVertexSdfCorners<float>(voxel_sdf_corners, mesh.voxel_sdf_corners);
            }
        }

        if (face_idx && face_idx->count > 0)
        {
            switch (face_idx->t)
            {
                case tinyply::Type::INT32:   copyFacesIdx3<int32_t>(face_idx, mesh.faces); break;
                case tinyply::Type::UINT32:  copyFacesIdx3<uint32_t>(face_idx, mesh.faces); break;
                case tinyply::Type::INT16:   copyFacesIdx3<int16_t>(face_idx, mesh.faces); break;
                case tinyply::Type::UINT16:  copyFacesIdx3<uint16_t>(face_idx, mesh.faces); break;
                case tinyply::Type::INT8:    copyFacesIdx3<int8_t>(face_idx, mesh.faces); break;
                case tinyply::Type::UINT8:   copyFacesIdx3<uint8_t>(face_idx, mesh.faces); break;
                default:
                    std::cerr << "[mesh_eval] unsupported face index type in: " << ply_path << "\n";
                    mesh.faces.clear();
                    break;
            }
        }
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[mesh_eval] tinyply failed for " << ply_path
                  << " (" << e.what() << "), trying binary fallback.\n";
        mesh = MeshData{};
        const bool ok = loadPlyMeshBinaryFallback(ply_path, mesh);
        if (!ok)
        {
            std::cerr << "[mesh_eval] binary fallback failed for: " << ply_path << "\n";
        }
        return ok;
    }
}

bool loadTrajectory4x4RowMajor(const std::string& traj_path, std::vector<cv::Matx44f>& traj)
{
    std::ifstream in(traj_path);
    if (!in.is_open())
    {
        std::cerr << "[mesh_eval] failed to open trajectory: " << traj_path << "\n";
        return false;
    }

    traj.clear();
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        float m[16];
        bool ok = true;
        for (int i = 0; i < 16; ++i)
        {
            if (!(iss >> m[i]))
            {
                ok = false;
                break;
            }
        }
        if (!ok) continue;

        traj.emplace_back(
            m[0], m[1], m[2], m[3],
            m[4], m[5], m[6], m[7],
            m[8], m[9], m[10], m[11],
            m[12], m[13], m[14], m[15]);
    }

    if (traj.empty())
    {
        std::cerr << "[mesh_eval] no valid 4x4 poses in trajectory: " << traj_path << "\n";
        return false;
    }
    return true;
}

bool loadTumTrajectoryCenters(const std::string& tum_path, std::vector<cv::Point3d>& centers)
{
    std::ifstream in(tum_path);
    if (!in.is_open())
    {
        std::cerr << "[mesh_eval] failed to open recon TUM trajectory: " << tum_path << "\n";
        return false;
    }

    centers.clear();
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        double t = 0.0, tx = 0.0, ty = 0.0, tz = 0.0;
        double qx = 0.0, qy = 0.0, qz = 0.0, qw = 1.0;
        if (!(iss >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw))
        {
            continue;
        }
        centers.emplace_back(tx, ty, tz);
    }

    if (centers.empty())
    {
        std::cerr << "[mesh_eval] no valid poses in recon TUM trajectory: " << tum_path << "\n";
        return false;
    }
    return true;
}

cv::Point3d cameraCenterFromPose(const cv::Matx44f& T_wc_or_w2c, bool traj_is_c2w)
{
    if (traj_is_c2w)
    {
        return cv::Point3d(
            static_cast<double>(T_wc_or_w2c(0, 3)),
            static_cast<double>(T_wc_or_w2c(1, 3)),
            static_cast<double>(T_wc_or_w2c(2, 3)));
    }

    cv::Matx33d R(
        static_cast<double>(T_wc_or_w2c(0, 0)), static_cast<double>(T_wc_or_w2c(0, 1)), static_cast<double>(T_wc_or_w2c(0, 2)),
        static_cast<double>(T_wc_or_w2c(1, 0)), static_cast<double>(T_wc_or_w2c(1, 1)), static_cast<double>(T_wc_or_w2c(1, 2)),
        static_cast<double>(T_wc_or_w2c(2, 0)), static_cast<double>(T_wc_or_w2c(2, 1)), static_cast<double>(T_wc_or_w2c(2, 2)));
    cv::Vec3d t(
        static_cast<double>(T_wc_or_w2c(0, 3)),
        static_cast<double>(T_wc_or_w2c(1, 3)),
        static_cast<double>(T_wc_or_w2c(2, 3)));
    const cv::Vec3d c = -(R.t() * t);
    return cv::Point3d(c[0], c[1], c[2]);
}

void extractCentersFromTrajectory(
    const std::vector<cv::Matx44f>& traj,
    bool traj_is_c2w,
    std::vector<cv::Point3d>& centers)
{
    centers.clear();
    centers.reserve(traj.size());
    for (const auto& T : traj)
    {
        centers.push_back(cameraCenterFromPose(T, traj_is_c2w));
    }
}

bool estimateSimilarityUmeyama(
    const std::vector<cv::Point3d>& src_pts,
    const std::vector<cv::Point3d>& dst_pts,
    int stride,
    int max_pairs,
    double& scale,
    cv::Matx33d& R,
    cv::Vec3d& t)
{
    const int s = std::max(1, stride);
    const size_t n_all = std::min(src_pts.size(), dst_pts.size());
    if (n_all < 4) return false;

    std::vector<cv::Point3d> src;
    std::vector<cv::Point3d> dst;
    src.reserve(n_all / s + 1);
    dst.reserve(n_all / s + 1);
    for (size_t i = 0; i < n_all; i += static_cast<size_t>(s))
    {
        src.push_back(src_pts[i]);
        dst.push_back(dst_pts[i]);
        if (max_pairs > 0 && static_cast<int>(src.size()) >= max_pairs) break;
    }
    if (src.size() < 4) return false;

    cv::Vec3d mu_src(0, 0, 0), mu_dst(0, 0, 0);
    for (size_t i = 0; i < src.size(); ++i)
    {
        mu_src += cv::Vec3d(src[i].x, src[i].y, src[i].z);
        mu_dst += cv::Vec3d(dst[i].x, dst[i].y, dst[i].z);
    }
    const double inv_n = 1.0 / static_cast<double>(src.size());
    mu_src *= inv_n;
    mu_dst *= inv_n;

    cv::Matx33d cov = cv::Matx33d::zeros();
    double var_src = 0.0;
    for (size_t i = 0; i < src.size(); ++i)
    {
        const cv::Vec3d xs = cv::Vec3d(src[i].x, src[i].y, src[i].z) - mu_src;
        const cv::Vec3d yd = cv::Vec3d(dst[i].x, dst[i].y, dst[i].z) - mu_dst;

        cov(0, 0) += yd[0] * xs[0]; cov(0, 1) += yd[0] * xs[1]; cov(0, 2) += yd[0] * xs[2];
        cov(1, 0) += yd[1] * xs[0]; cov(1, 1) += yd[1] * xs[1]; cov(1, 2) += yd[1] * xs[2];
        cov(2, 0) += yd[2] * xs[0]; cov(2, 1) += yd[2] * xs[1]; cov(2, 2) += yd[2] * xs[2];
        var_src += xs.dot(xs);
    }
    cov *= inv_n;
    var_src *= inv_n;
    if (var_src <= 1e-15) return false;

    cv::Mat cov_m(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            cov_m.at<double>(r, c) = cov(r, c);
        }
    }
    cv::SVD svd(cov_m, cv::SVD::FULL_UV);
    cv::Mat U = svd.u;
    cv::Mat Vt = svd.vt;

    cv::Matx33d Ux, Vtx;
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            Ux(r, c) = U.at<double>(r, c);
            Vtx(r, c) = Vt.at<double>(r, c);
        }
    }

    cv::Matx33d S = cv::Matx33d::eye();
    const cv::Matx33d UV = Ux * Vtx;
    if (cv::determinant(UV) < 0.0) S(2, 2) = -1.0;

    R = Ux * S * Vtx;

    const double d0 = svd.w.at<double>(0, 0);
    const double d1 = svd.w.at<double>(1, 0);
    const double d2 = svd.w.at<double>(2, 0);
    const double trace_ds = d0 * S(0, 0) + d1 * S(1, 1) + d2 * S(2, 2);
    scale = trace_ds / var_src;
    t = mu_dst - scale * (R * mu_src);
    return std::isfinite(scale);
}

void applySimilarityToMesh(MeshData& mesh, double scale, const cv::Matx33d& R, const cv::Vec3d& t)
{
    for (auto& v : mesh.vertices)
    {
        const cv::Vec3d p(v.x, v.y, v.z);
        const cv::Vec3d q = scale * (R * p) + t;
        v.x = static_cast<float>(q[0]);
        v.y = static_cast<float>(q[1]);
        v.z = static_cast<float>(q[2]);
    }
}

bool savePlyMesh(const std::string& ply_path, const MeshData& mesh)
{
    if (mesh.vertices.empty())
    {
        std::cerr << "[mesh_eval] savePlyMesh: empty mesh, skipping " << ply_path << "\n";
        return false;
    }

    try
    {
        std::filesystem::path out_path(ply_path);
        if (!out_path.parent_path().empty())
        {
            std::filesystem::create_directories(out_path.parent_path());
        }

        std::vector<float> vertices_xyz;
        vertices_xyz.reserve(mesh.vertices.size() * 3);
        for (const auto& v : mesh.vertices)
        {
            vertices_xyz.push_back(v.x);
            vertices_xyz.push_back(v.y);
            vertices_xyz.push_back(v.z);
        }

        std::vector<uint32_t> tri_idx;
        tri_idx.reserve(mesh.faces.size() * 3);
        for (const auto& f : mesh.faces)
        {
            tri_idx.push_back(static_cast<uint32_t>(f[0]));
            tri_idx.push_back(static_cast<uint32_t>(f[1]));
            tri_idx.push_back(static_cast<uint32_t>(f[2]));
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
        fb.open(out_path, std::ios::out | std::ios::binary);
        std::ostream out(&fb);
        if (out.fail())
        {
            std::cerr << "[mesh_eval] savePlyMesh: open failed: " << ply_path << "\n";
            return false;
        }

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

        if (!mesh.faces.empty())
        {
            ply.add_properties_to_element(
                "face", {"vertex_indices"},
                tinyply::Type::UINT32, static_cast<uint64_t>(mesh.faces.size()),
                reinterpret_cast<uint8_t*>(tri_idx.data()),
                tinyply::Type::UINT8, 3);
        }

        ply.write(out, /*binary=*/true);
        fb.close();
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[mesh_eval] savePlyMesh failed: " << e.what() << "\n";
        return false;
    }
}

void cleanMeshLikeGaussianSlam(MeshData& mesh, size_t min_component_vertices = 200)
{
    if (mesh.vertices.empty() || mesh.faces.empty())
    {
        return;
    }

    const size_t n = mesh.vertices.size();
    std::vector<int> parent(n, -1);
    std::vector<int> rank(n, 0);
    for (size_t i = 0; i < n; ++i) parent[i] = static_cast<int>(i);

    std::function<int(int)> find_root = [&](int x) -> int
    {
        if (parent[static_cast<size_t>(x)] != x)
        {
            parent[static_cast<size_t>(x)] = find_root(parent[static_cast<size_t>(x)]);
        }
        return parent[static_cast<size_t>(x)];
    };

    auto unite = [&](int a, int b)
    {
        a = find_root(a);
        b = find_root(b);
        if (a == b) return;
        if (rank[static_cast<size_t>(a)] < rank[static_cast<size_t>(b)]) std::swap(a, b);
        parent[static_cast<size_t>(b)] = a;
        if (rank[static_cast<size_t>(a)] == rank[static_cast<size_t>(b)])
        {
            ++rank[static_cast<size_t>(a)];
        }
    };

    for (const auto& f : mesh.faces)
    {
        if (f[0] < 0 || f[1] < 0 || f[2] < 0 ||
            f[0] >= static_cast<int>(n) ||
            f[1] >= static_cast<int>(n) ||
            f[2] >= static_cast<int>(n))
        {
            continue;
        }
        unite(f[0], f[1]);
        unite(f[1], f[2]);
        unite(f[2], f[0]);
    }

    std::vector<size_t> comp_sizes(n, 0);
    for (size_t i = 0; i < n; ++i)
    {
        ++comp_sizes[static_cast<size_t>(find_root(static_cast<int>(i)))];
    }

    std::vector<uint8_t> keep_vertex(n, 0);
    size_t kept_vertices = 0;
    for (size_t i = 0; i < n; ++i)
    {
        const int root = find_root(static_cast<int>(i));
        if (comp_sizes[static_cast<size_t>(root)] >= min_component_vertices)
        {
            keep_vertex[i] = 1;
            ++kept_vertices;
        }
    }

    if (kept_vertices == 0 || kept_vertices == n)
    {
        return;
    }

    std::vector<int> old_to_new(n, -1);
    MeshData cleaned;
    cleaned.vertices.reserve(kept_vertices);
    if (!mesh.colors.empty()) cleaned.colors.reserve(kept_vertices);
    for (size_t i = 0; i < n; ++i)
    {
        if (!keep_vertex[i]) continue;
        old_to_new[i] = static_cast<int>(cleaned.vertices.size());
        cleaned.vertices.push_back(mesh.vertices[i]);
        if (!mesh.colors.empty())
        {
            cleaned.colors.push_back(i < mesh.colors.size() ? mesh.colors[i] : cv::Vec3b(255, 255, 255));
        }
    }

    std::vector<cv::Vec3i> faces_tmp;
    faces_tmp.reserve(mesh.faces.size());
    for (const auto& f : mesh.faces)
    {
        if (f[0] < 0 || f[1] < 0 || f[2] < 0 ||
            f[0] >= static_cast<int>(n) ||
            f[1] >= static_cast<int>(n) ||
            f[2] >= static_cast<int>(n))
        {
            continue;
        }
        if (!keep_vertex[static_cast<size_t>(f[0])] ||
            !keep_vertex[static_cast<size_t>(f[1])] ||
            !keep_vertex[static_cast<size_t>(f[2])])
        {
            continue;
        }
        cv::Vec3i nf(
            old_to_new[static_cast<size_t>(f[0])],
            old_to_new[static_cast<size_t>(f[1])],
            old_to_new[static_cast<size_t>(f[2])]);
        if (nf[0] == nf[1] || nf[1] == nf[2] || nf[0] == nf[2]) continue;
        faces_tmp.push_back(nf);
    }

    auto canonical_face = [](const cv::Vec3i& f) -> cv::Vec3i
    {
        std::array<int, 3> a{f[0], f[1], f[2]};
        std::sort(a.begin(), a.end());
        return cv::Vec3i(a[0], a[1], a[2]);
    };

    std::sort(faces_tmp.begin(), faces_tmp.end(), [&](const cv::Vec3i& a, const cv::Vec3i& b)
    {
        const cv::Vec3i ca = canonical_face(a);
        const cv::Vec3i cb = canonical_face(b);
        if (ca[0] != cb[0]) return ca[0] < cb[0];
        if (ca[1] != cb[1]) return ca[1] < cb[1];
        return ca[2] < cb[2];
    });

    cleaned.faces.reserve(faces_tmp.size());
    cv::Vec3i last(-1, -1, -1);
    bool have_last = false;
    for (const auto& f : faces_tmp)
    {
        const cv::Vec3i cf = canonical_face(f);
        if (!have_last || cf != last)
        {
            cleaned.faces.push_back(f);
            last = cf;
            have_last = true;
        }
    }

    std::cout << "[mesh_eval] gaussian_slam clean_mesh: vertices "
              << mesh.vertices.size() << " -> " << cleaned.vertices.size()
              << " faces " << mesh.faces.size() << " -> " << cleaned.faces.size() << "\n";
    mesh = std::move(cleaned);
}

bool loadCameraIntrinsicsFromJson(const std::string& cam_json_path, CameraIntrinsics& K)
{
    std::ifstream in(cam_json_path);
    if (!in.is_open())
    {
        std::cerr << "[mesh_eval] failed to open camera json: " << cam_json_path << "\n";
        return false;
    }

    Json::CharReaderBuilder builder;
    Json::Value root;
    std::string errs;
    if (!Json::parseFromStream(builder, in, &root, &errs))
    {
        std::cerr << "[mesh_eval] failed to parse camera json: " << errs << "\n";
        return false;
    }

    const Json::Value cam = root["camera"];
    if (cam.isNull())
    {
        std::cerr << "[mesh_eval] camera json missing `camera` object\n";
        return false;
    }

    K.w = cam.get("w", 0).asInt();
    K.h = cam.get("h", 0).asInt();
    K.fx = cam.get("fx", 0.0f).asFloat();
    K.fy = cam.get("fy", 0.0f).asFloat();
    K.cx = cam.get("cx", 0.0f).asFloat();
    K.cy = cam.get("cy", 0.0f).asFloat();
    return K.valid();
}

bool loadNpyPoints3(const std::string& npy_path, std::vector<cv::Point3f>& pts)
{
    std::ifstream in(npy_path, std::ios::binary);
    if (!in.is_open())
    {
        std::cerr << "[mesh_eval] failed to open npy: " << npy_path << "\n";
        return false;
    }

    char magic[6] = {};
    in.read(magic, 6);
    if (!in || std::string(magic, 6) != "\x93NUMPY")
    {
        std::cerr << "[mesh_eval] invalid npy magic: " << npy_path << "\n";
        return false;
    }

    uint8_t major = 0, minor = 0;
    in.read(reinterpret_cast<char*>(&major), 1);
    in.read(reinterpret_cast<char*>(&minor), 1);
    if (!in)
    {
        std::cerr << "[mesh_eval] failed to read npy version: " << npy_path << "\n";
        return false;
    }

    uint32_t header_len = 0;
    if (major == 1)
    {
        uint16_t header_len_u16 = 0;
        in.read(reinterpret_cast<char*>(&header_len_u16), sizeof(header_len_u16));
        header_len = header_len_u16;
    }
    else
    {
        in.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    }
    if (!in || header_len == 0)
    {
        std::cerr << "[mesh_eval] failed to read npy header length: " << npy_path << "\n";
        return false;
    }

    std::string header(header_len, '\0');
    in.read(header.data(), static_cast<std::streamsize>(header_len));
    if (!in)
    {
        std::cerr << "[mesh_eval] failed to read npy header: " << npy_path << "\n";
        return false;
    }

    std::smatch m;
    std::regex descr_re("'descr'\\s*:\\s*'([^']+)'");
    std::regex fortran_re("'?fortran_order'?\\s*:\\s*(True|False)");
    std::regex shape_re("'?shape'?\\s*:\\s*\\(([^\\)]*)\\)");

    std::string descr;
    bool fortran = false;
    std::vector<int64_t> shape;

    if (std::regex_search(header, m, descr_re) && m.size() >= 2)
    {
        descr = m[1].str();
    }
    if (std::regex_search(header, m, fortran_re) && m.size() >= 2)
    {
        fortran = (m[1].str() == "True");
    }
    if (std::regex_search(header, m, shape_re) && m.size() >= 2)
    {
        std::stringstream ss(m[1].str());
        std::string tok;
        while (std::getline(ss, tok, ','))
        {
            std::stringstream ts(tok);
            int64_t v = 0;
            if (ts >> v) shape.push_back(v);
        }
    }

    if (fortran)
    {
        std::cerr << "[mesh_eval] unsupported Fortran-order npy: " << npy_path << "\n";
        return false;
    }
    if (shape.size() != 2 || shape[1] != 3 || shape[0] <= 0)
    {
        std::cerr << "[mesh_eval] expected npy shape (N,3): " << npy_path << "\n";
        return false;
    }

    const size_t n = static_cast<size_t>(shape[0]);
    pts.clear();
    pts.resize(n);

    if (descr == "<f4" || descr == "|f4")
    {
        std::vector<float> data(n * 3);
        in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
        if (!in) return false;
        for (size_t i = 0; i < n; ++i)
        {
            pts[i] = cv::Point3f(data[3 * i + 0], data[3 * i + 1], data[3 * i + 2]);
        }
        return true;
    }
    if (descr == "<f8" || descr == "|f8")
    {
        std::vector<double> data(n * 3);
        in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(double)));
        if (!in) return false;
        for (size_t i = 0; i < n; ++i)
        {
            pts[i] = cv::Point3f(
                static_cast<float>(data[3 * i + 0]),
                static_cast<float>(data[3 * i + 1]),
                static_cast<float>(data[3 * i + 2]));
        }
        return true;
    }

    std::cerr << "[mesh_eval] unsupported npy dtype `" << descr << "` in " << npy_path << "\n";
    return false;
}

const char* evalModeName(EvalMode mode)
{
    switch (mode)
    {
        case EvalMode::Current: return "current";
        case EvalMode::GaussianSlam: return "gaussian_slam";
        case EvalMode::GaussianSlamSim3: return "gaussian_slam_sim3";
        default: return "unknown";
    }
}

bool parseEvalMode(const std::string& s, EvalMode& mode)
{
    if (s == "current")
    {
        mode = EvalMode::Current;
        return true;
    }
    if (s == "gaussian_slam")
    {
        mode = EvalMode::GaussianSlam;
        return true;
    }
    if (s == "gaussian_slam_sim3")
    {
        mode = EvalMode::GaussianSlamSim3;
        return true;
    }
    return false;
}

std::vector<cv::Point3f> sampleVertices(
    const std::vector<cv::Point3f>& vertices,
    size_t max_points,
    uint32_t seed)
{
    if (vertices.empty() || max_points == 0) return {};
    if (vertices.size() <= max_points) return vertices;

    std::mt19937 rng(seed);
    std::vector<size_t> ids(vertices.size());
    std::iota(ids.begin(), ids.end(), size_t(0));
    std::shuffle(ids.begin(), ids.end(), rng);

    std::vector<cv::Point3f> out;
    out.reserve(max_points);
    for (size_t i = 0; i < max_points; ++i) out.push_back(vertices[ids[i]]);
    return out;
}

cv::Mat pointsToMat(const std::vector<cv::Point3f>& pts);

bool estimateRigidKabsch(
    const std::vector<cv::Point3f>& src_pts,
    const std::vector<cv::Point3f>& dst_pts,
    cv::Matx33d& R,
    cv::Vec3d& t)
{
    if (src_pts.size() != dst_pts.size() || src_pts.size() < 3) return false;

    cv::Vec3d mu_src(0, 0, 0), mu_dst(0, 0, 0);
    for (size_t i = 0; i < src_pts.size(); ++i)
    {
        mu_src += cv::Vec3d(src_pts[i].x, src_pts[i].y, src_pts[i].z);
        mu_dst += cv::Vec3d(dst_pts[i].x, dst_pts[i].y, dst_pts[i].z);
    }
    const double inv_n = 1.0 / static_cast<double>(src_pts.size());
    mu_src *= inv_n;
    mu_dst *= inv_n;

    cv::Matx33d H = cv::Matx33d::zeros();
    for (size_t i = 0; i < src_pts.size(); ++i)
    {
        const cv::Vec3d xs = cv::Vec3d(src_pts[i].x, src_pts[i].y, src_pts[i].z) - mu_src;
        const cv::Vec3d yd = cv::Vec3d(dst_pts[i].x, dst_pts[i].y, dst_pts[i].z) - mu_dst;
        H(0, 0) += xs[0] * yd[0]; H(0, 1) += xs[0] * yd[1]; H(0, 2) += xs[0] * yd[2];
        H(1, 0) += xs[1] * yd[0]; H(1, 1) += xs[1] * yd[1]; H(1, 2) += xs[1] * yd[2];
        H(2, 0) += xs[2] * yd[0]; H(2, 1) += xs[2] * yd[1]; H(2, 2) += xs[2] * yd[2];
    }

    cv::Mat Hm(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            Hm.at<double>(r, c) = H(r, c);
        }
    }

    cv::SVD svd(Hm, cv::SVD::FULL_UV);
    cv::Mat U = svd.u;
    cv::Mat Vt = svd.vt;
    cv::Mat Rm = Vt.t() * U.t();
    if (cv::determinant(Rm) < 0.0)
    {
        cv::Mat V = Vt.t();
        V.col(2) *= -1.0;
        Rm = V * U.t();
    }

    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            R(r, c) = Rm.at<double>(r, c);
        }
    }
    t = mu_dst - R * mu_src;
    return true;
}

void applyRigidToMesh(MeshData& mesh, const cv::Matx33d& R, const cv::Vec3d& t)
{
    for (auto& v : mesh.vertices)
    {
        const cv::Vec3d p(v.x, v.y, v.z);
        const cv::Vec3d q = R * p + t;
        v.x = static_cast<float>(q[0]);
        v.y = static_cast<float>(q[1]);
        v.z = static_cast<float>(q[2]);
    }
}

bool estimateRigidIcpPointToPoint(
    const MeshData& src_mesh,
    const MeshData& dst_mesh,
    float threshold_m,
    int max_iters,
    int max_points,
    uint32_t seed,
    cv::Matx33d& R_out,
    cv::Vec3d& t_out,
    int& inliers_out)
{
    if (src_mesh.vertices.size() < 3 || dst_mesh.vertices.size() < 3) return false;

    const size_t sample_n = static_cast<size_t>(std::max(3, max_points));
    const auto src = sampleVertices(src_mesh.vertices, sample_n, seed);
    const auto dst = sampleVertices(dst_mesh.vertices, sample_n, seed + 1u);
    if (src.size() < 3 || dst.size() < 3) return false;

    cv::Mat dst_m = pointsToMat(dst);
    cv::flann::Index kdtree(dst_m, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_L2);

    R_out = cv::Matx33d::eye();
    t_out = cv::Vec3d(0.0, 0.0, 0.0);
    inliers_out = 0;
    const float threshold2 = threshold_m * threshold_m;

    for (int iter = 0; iter < std::max(1, max_iters); ++iter)
    {
        std::vector<cv::Point3f> transformed;
        transformed.reserve(src.size());
        for (const auto& p : src)
        {
            const cv::Vec3d q = R_out * cv::Vec3d(p.x, p.y, p.z) + t_out;
            transformed.emplace_back(
                static_cast<float>(q[0]),
                static_cast<float>(q[1]),
                static_cast<float>(q[2]));
        }

        cv::Mat query = pointsToMat(transformed);
        cv::Mat idx(query.rows, 1, CV_32S);
        cv::Mat d2(query.rows, 1, CV_32F);
        kdtree.knnSearch(query, idx, d2, 1, cv::flann::SearchParams(64));

        std::vector<cv::Point3f> src_inliers;
        std::vector<cv::Point3f> dst_inliers;
        src_inliers.reserve(transformed.size());
        dst_inliers.reserve(transformed.size());

        for (int i = 0; i < query.rows; ++i)
        {
            const float dist2 = d2.at<float>(i, 0);
            if (!(dist2 <= threshold2)) continue;
            const int j = idx.at<int>(i, 0);
            if (j < 0 || j >= static_cast<int>(dst.size())) continue;
            src_inliers.push_back(transformed[static_cast<size_t>(i)]);
            dst_inliers.push_back(dst[static_cast<size_t>(j)]);
        }

        inliers_out = static_cast<int>(src_inliers.size());
        if (inliers_out < 3) return false;

        cv::Matx33d dR = cv::Matx33d::eye();
        cv::Vec3d dt(0.0, 0.0, 0.0);
        if (!estimateRigidKabsch(src_inliers, dst_inliers, dR, dt))
        {
            return false;
        }

        const double rot_delta =
            std::abs(dR(0, 0) - 1.0) + std::abs(dR(1, 1) - 1.0) + std::abs(dR(2, 2) - 1.0);
        const double trans_delta = cv::norm(dt);
        R_out = dR * R_out;
        t_out = dR * t_out + dt;
        if (rot_delta < 1e-8 && trans_delta < 1e-7) break;
    }

    return true;
}

static inline cv::Point3f barycentricSample(
    const cv::Point3f& v0,
    const cv::Point3f& v1,
    const cv::Point3f& v2,
    std::mt19937& rng)
{
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    const float u = uni(rng);
    const float v = uni(rng);
    const float su = std::sqrt(u);
    const float b0 = 1.0f - su;
    const float b1 = su * (1.0f - v);
    const float b2 = su * v;
    return b0 * v0 + b1 * v1 + b2 * v2;
}

std::vector<cv::Point3f> sampleMeshPoints(
    const MeshData& mesh,
    size_t n_samples,
    uint32_t seed)
{
    if (mesh.vertices.empty()) return {};
    std::mt19937 rng(seed);

    // Point-cloud fallback (no faces)
    if (mesh.faces.empty())
    {
        if (mesh.vertices.size() <= n_samples) return mesh.vertices;
        std::vector<size_t> ids(mesh.vertices.size());
        std::iota(ids.begin(), ids.end(), size_t(0));
        std::shuffle(ids.begin(), ids.end(), rng);
        std::vector<cv::Point3f> out;
        out.reserve(n_samples);
        for (size_t i = 0; i < n_samples; ++i) out.push_back(mesh.vertices[ids[i]]);
        return out;
    }

    // Area-weighted face sampling for remaining points
    std::vector<double> areas;
    areas.reserve(mesh.faces.size());
    for (const auto& f : mesh.faces)
    {
        if (f[0] < 0 || f[1] < 0 || f[2] < 0 ||
            f[0] >= static_cast<int>(mesh.vertices.size()) ||
            f[1] >= static_cast<int>(mesh.vertices.size()) ||
            f[2] >= static_cast<int>(mesh.vertices.size()))
        {
            areas.push_back(0.0);
            continue;
        }
        const cv::Point3f& v0 = mesh.vertices[f[0]];
        const cv::Point3f& v1 = mesh.vertices[f[1]];
        const cv::Point3f& v2 = mesh.vertices[f[2]];
        const cv::Point3f e1 = v1 - v0;
        const cv::Point3f e2 = v2 - v0;
        const cv::Point3f c = e1.cross(e2);
        const double a = 0.5 * std::sqrt(c.dot(c));
        areas.push_back(a);
    }

    const double area_sum = std::accumulate(areas.begin(), areas.end(), 0.0);
    if (!(area_sum > 0.0))
    {
        // Degenerate faces: fallback to vertex random resampling.
        std::uniform_int_distribution<int> vdist(0, static_cast<int>(mesh.vertices.size() - 1));
        std::vector<cv::Point3f> out;
        out.reserve(n_samples);
        while (out.size() < n_samples) out.push_back(mesh.vertices[vdist(rng)]);
        return out;
    }

    std::discrete_distribution<size_t> face_dist(areas.begin(), areas.end());
    std::vector<cv::Point3f> out;
    out.reserve(n_samples);
    while (out.size() < n_samples)
    {
        const size_t fi = face_dist(rng);
        const auto& f = mesh.faces[fi];
        if (f[0] < 0 || f[1] < 0 || f[2] < 0 ||
            f[0] >= static_cast<int>(mesh.vertices.size()) ||
            f[1] >= static_cast<int>(mesh.vertices.size()) ||
            f[2] >= static_cast<int>(mesh.vertices.size()))
        {
            continue;
        }
        const cv::Point3f p = barycentricSample(
            mesh.vertices[f[0]], mesh.vertices[f[1]], mesh.vertices[f[2]], rng);
        out.push_back(p);
    }
    return out;
}

struct GaussianSupportSampleResult
{
    std::vector<cv::Point3f> points;
    size_t eligible_gaussians = 0;
    size_t samples_per_primitive = 0;
    double opacity_weight = 0.0;
};

float sigmoidStable(float value)
{
    if (value >= 0.0f)
    {
        const float e = std::exp(-value);
        return 1.0f / (1.0f + e);
    }
    const float e = std::exp(value);
    return e / (1.0f + e);
}

cv::Matx33d quaternionWxyzToRotation(const cv::Vec4f& raw_q)
{
    const double norm = std::sqrt(
        static_cast<double>(raw_q[0]) * raw_q[0] +
        static_cast<double>(raw_q[1]) * raw_q[1] +
        static_cast<double>(raw_q[2]) * raw_q[2] +
        static_cast<double>(raw_q[3]) * raw_q[3]);
    if (!(norm > 1.0e-12) || !std::isfinite(norm)) return cv::Matx33d::eye();
    const double w = raw_q[0] / norm;
    const double x = raw_q[1] / norm;
    const double y = raw_q[2] / norm;
    const double z = raw_q[3] / norm;
    return cv::Matx33d(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
        2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
        2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y));
}

GaussianSupportSampleResult sampleGaussianSupport(
    const MeshData& mesh,
    size_t samples_per_primitive,
    float sigma,
    float min_opacity,
    double alignment_scale,
    const cv::Matx33d& alignment_rotation,
    uint32_t seed)
{
    GaussianSupportSampleResult result;
    if (!mesh.hasGaussianAttributes() || samples_per_primitive == 0 || !(sigma > 0.0f))
        return result;

    struct Primitive
    {
        cv::Vec3d center;
        cv::Vec3d axes;
        cv::Matx33d rotation;
    };
    std::vector<Primitive> primitives;
    primitives.reserve(mesh.vertices.size());
    const double world_scale = std::abs(alignment_scale);
    for (size_t i = 0; i < mesh.vertices.size(); ++i)
    {
        const float opacity = sigmoidStable(mesh.gaussian_opacity[i]);
        if (!std::isfinite(opacity) || opacity < min_opacity) continue;

        cv::Vec3d axes;
        bool valid = true;
        for (int axis = 0; axis < 3; ++axis)
        {
            const double log_scale = std::clamp(
                static_cast<double>(mesh.gaussian_log_scales[i][axis]), -20.0, 10.0);
            axes[axis] = static_cast<double>(sigma) * world_scale * std::exp(log_scale);
            valid = valid && std::isfinite(axes[axis]) && axes[axis] > 0.0;
        }
        if (!valid) continue;
        const double weight = static_cast<double>(opacity);
        if (!(weight > 0.0) || !std::isfinite(weight)) continue;

        const cv::Point3f& center = mesh.vertices[i];
        Primitive primitive;
        primitive.center = cv::Vec3d(center.x, center.y, center.z);
        primitive.axes = axes;
        primitive.rotation =
            alignment_rotation * quaternionWxyzToRotation(mesh.gaussian_rotations[i]);
        primitives.push_back(primitive);
        result.opacity_weight += weight;
    }
    result.eligible_gaussians = primitives.size();
    result.samples_per_primitive = samples_per_primitive;
    if (primitives.empty()) return result;

    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    result.points.reserve(primitives.size() * samples_per_primitive);
    for (const Primitive& primitive : primitives)
    {
        for (size_t sample = 0; sample < samples_per_primitive; ++sample)
        {
            cv::Vec3d local(0.0, 0.0, 0.0);
            if (sample < 6)
            {
                const int axis = static_cast<int>(sample / 2);
                local[axis] = (sample % 2 == 0 ? -1.0 : 1.0) * primitive.axes[axis];
            }
            else
            {
                cv::Vec3d direction(1.0, 0.0, 0.0);
                bool accepted = false;
                for (int attempt = 0; attempt < 64; ++attempt)
                {
                    direction = cv::Vec3d(normal(rng), normal(rng), normal(rng));
                    const double norm = cv::norm(direction);
                    if (!(norm > 1.0e-12)) continue;
                    direction *= 1.0 / norm;

                    // Correct transformed-sphere sampling to be uniform over ellipsoid area.
                    const cv::Vec3d inverse_scaled(
                        direction[0] / primitive.axes[0],
                        direction[1] / primitive.axes[1],
                        direction[2] / primitive.axes[2]);
                    const double min_axis = std::min({
                        primitive.axes[0], primitive.axes[1], primitive.axes[2]});
                    const double accept_probability = std::clamp(
                        min_axis * cv::norm(inverse_scaled), 0.0, 1.0);
                    if (uniform(rng) <= accept_probability)
                    {
                        accepted = true;
                        break;
                    }
                }
                if (!accepted)
                {
                    const double norm = cv::norm(direction);
                    if (norm > 1.0e-12) direction *= 1.0 / norm;
                }
                local = cv::Vec3d(
                    primitive.axes[0] * direction[0],
                    primitive.axes[1] * direction[1],
                    primitive.axes[2] * direction[2]);
            }
            const cv::Vec3d world = primitive.center + primitive.rotation * local;
            result.points.emplace_back(
                static_cast<float>(world[0]),
                static_cast<float>(world[1]),
                static_cast<float>(world[2]));
        }
    }
    return result;
}

struct VoxelSupportSampleResult
{
    std::vector<cv::Point3f> points;
    size_t total_voxels = 0;
    size_t eligible_zero_crossing_voxels = 0;
    size_t samples_per_primitive = 0;
    double min_edge_m = 0.0;
    double max_edge_m = 0.0;
};

bool hasSdfSignChange(const std::array<float, 8>& sdf)
{
    int min_sign = 1;
    int max_sign = -1;
    for (float value : sdf)
    {
        if (!std::isfinite(value)) return false;
        const int sign = value < 0.0f ? -1 : (value > 0.0f ? 1 : 0);
        min_sign = std::min(min_sign, sign);
        max_sign = std::max(max_sign, sign);
    }
    // Matches SVRecon's `signs.min() != signs.max()` surface-cell test.
    return min_sign != max_sign;
}

VoxelSupportSampleResult sampleVoxelSupport(
    const MeshData& mesh,
    size_t samples_per_primitive,
    double alignment_scale,
    const cv::Matx33d& alignment_rotation,
    uint32_t seed)
{
    VoxelSupportSampleResult result;
    if (!mesh.hasVoxelAttributes() || samples_per_primitive == 0) return result;
    result.total_voxels = mesh.vertices.size();

    struct Primitive
    {
        cv::Vec3d center;
        double half_edge = 0.0;
    };
    std::vector<Primitive> primitives;
    primitives.reserve(mesh.vertices.size());
    const double world_scale = std::abs(alignment_scale);
    result.min_edge_m = std::numeric_limits<double>::infinity();
    result.max_edge_m = 0.0;
    for (size_t i = 0; i < mesh.vertices.size(); ++i)
    {
        if (!hasSdfSignChange(mesh.voxel_sdf_corners[i])) continue;
        const int level = static_cast<int>(mesh.voxel_octlevels[i]);
        const double edge = world_scale * std::ldexp(
            static_cast<double>(mesh.voxel_scene_extent), -level);
        if (!(edge > 0.0) || !std::isfinite(edge)) continue;

        const cv::Point3f& center = mesh.vertices[i];
        primitives.push_back({cv::Vec3d(center.x, center.y, center.z), 0.5 * edge});
        result.min_edge_m = std::min(result.min_edge_m, edge);
        result.max_edge_m = std::max(result.max_edge_m, edge);
    }
    result.eligible_zero_crossing_voxels = primitives.size();
    result.samples_per_primitive = samples_per_primitive;
    if (primitives.empty())
    {
        result.min_edge_m = 0.0;
        return result;
    }

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> choose_face(0, 5);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    result.points.reserve(primitives.size() * samples_per_primitive);
    for (const Primitive& primitive : primitives)
    {
        for (size_t sample = 0; sample < samples_per_primitive; ++sample)
        {
            const double h = primitive.half_edge;
            cv::Vec3d local;
            if (sample < 8)
            {
                local = cv::Vec3d(
                    (sample & 1u) ? h : -h,
                    (sample & 2u) ? h : -h,
                    (sample & 4u) ? h : -h);
            }
            else
            {
                const double u = h * uniform(rng);
                const double v = h * uniform(rng);
                switch (choose_face(rng))
                {
                    case 0: local = cv::Vec3d(-h, u, v); break;
                    case 1: local = cv::Vec3d( h, u, v); break;
                    case 2: local = cv::Vec3d(u, -h, v); break;
                    case 3: local = cv::Vec3d(u,  h, v); break;
                    case 4: local = cv::Vec3d(u, v, -h); break;
                    default: local = cv::Vec3d(u, v, h); break;
                }
            }
            const cv::Vec3d world = primitive.center + alignment_rotation * local;
            result.points.emplace_back(
                static_cast<float>(world[0]),
                static_cast<float>(world[1]),
                static_cast<float>(world[2]));
        }
    }
    return result;
}

std::vector<float> maxSupportDistancePerPrimitive(
    const std::vector<float>& support_distances,
    size_t primitive_count,
    size_t samples_per_primitive)
{
    if (primitive_count == 0 || samples_per_primitive == 0 ||
        support_distances.size() != primitive_count * samples_per_primitive)
    {
        return {};
    }
    std::vector<float> primitive_distances(
        primitive_count, -std::numeric_limits<float>::infinity());
    for (size_t primitive = 0; primitive < primitive_count; ++primitive)
    {
        const size_t begin = primitive * samples_per_primitive;
        for (size_t sample = 0; sample < samples_per_primitive; ++sample)
        {
            primitive_distances[primitive] = std::max(
                primitive_distances[primitive], support_distances[begin + sample]);
        }
    }
    return primitive_distances;
}

bool computeOrientedBounds(
    const MeshData& mesh,
    cv::Matx33d& basis_rows,
    cv::Vec3d& center,
    cv::Vec3d& local_min,
    cv::Vec3d& local_max)
{
    if (mesh.vertices.size() < 3) return false;

    center = cv::Vec3d(0.0, 0.0, 0.0);
    for (const auto& v : mesh.vertices)
    {
        center += cv::Vec3d(v.x, v.y, v.z);
    }
    center *= (1.0 / static_cast<double>(mesh.vertices.size()));

    cv::Mat pts(static_cast<int>(mesh.vertices.size()), 3, CV_64F);
    for (int i = 0; i < pts.rows; ++i)
    {
        pts.at<double>(i, 0) = static_cast<double>(mesh.vertices[static_cast<size_t>(i)].x) - center[0];
        pts.at<double>(i, 1) = static_cast<double>(mesh.vertices[static_cast<size_t>(i)].y) - center[1];
        pts.at<double>(i, 2) = static_cast<double>(mesh.vertices[static_cast<size_t>(i)].z) - center[2];
    }

    cv::Mat cov, mean_dummy;
    cv::calcCovarMatrix(pts, cov, mean_dummy, cv::COVAR_NORMAL | cv::COVAR_ROWS, CV_64F);
    cov /= static_cast<double>(std::max(1, pts.rows - 1));

    cv::Mat eigenvalues, eigenvectors;
    if (!cv::eigen(cov, eigenvalues, eigenvectors)) return false;
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            basis_rows(r, c) = eigenvectors.at<double>(r, c);
        }
    }

    local_min = cv::Vec3d(
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max());
    local_max = cv::Vec3d(
        -std::numeric_limits<double>::max(),
        -std::numeric_limits<double>::max(),
        -std::numeric_limits<double>::max());

    for (const auto& v : mesh.vertices)
    {
        const cv::Vec3d p(v.x, v.y, v.z);
        const cv::Vec3d q = basis_rows * (p - center);
        for (int k = 0; k < 3; ++k)
        {
            local_min[k] = std::min(local_min[k], q[k]);
            local_max[k] = std::max(local_max[k], q[k]);
        }
    }
    return true;
}

cv::Matx44f makeLookAtC2W(const cv::Vec3f& dir, const cv::Vec3f& up, const cv::Vec3f& pos)
{
    auto normalize3 = [](const cv::Vec3f& v) -> cv::Vec3f
    {
        const float n = std::sqrt(std::max(1e-20f, v.dot(v)));
        return v * (1.0f / n);
    };

    const cv::Vec3f vec2 = normalize3(dir);
    const cv::Vec3f vec0 = normalize3(up.cross(vec2));
    const cv::Vec3f vec1 = normalize3(vec2.cross(vec0));
    return cv::Matx44f(
        vec0[0], vec1[0], vec2[0], pos[0],
        vec0[1], vec1[1], vec2[1], pos[1],
        vec0[2], vec1[2], vec2[2], pos[2],
        0.0f,    0.0f,    0.0f,    1.0f);
}

bool projectsAnyPoints(
    const std::vector<cv::Point3f>& pts,
    const cv::Matx44f& T_c2w,
    const CameraIntrinsics& K)
{
    if (pts.empty()) return false;
    const cv::Matx44f T_w2c = T_c2w.inv();
    for (const auto& pw : pts)
    {
        const cv::Vec4f p4(pw.x, pw.y, pw.z, 1.0f);
        const cv::Vec4f pc = T_w2c * p4;
        const float z = pc[2];
        if (!(z > 1e-4f)) continue;
        const float u = K.fx * (pc[0] / z) + K.cx;
        const float v = K.fy * (pc[1] / z) + K.cy;
        if (u >= 0.0f && u < static_cast<float>(K.w) &&
            v >= 0.0f && v < static_cast<float>(K.h))
        {
            return true;
        }
    }
    return false;
}

void generateGaussianSlamViews(
    const MeshData& gt_mesh,
    const std::vector<cv::Point3f>& unseen_pts,
    const GaussianSlamDepthSettings& gs,
    uint32_t seed,
    std::vector<cv::Matx44f>& traj)
{
    traj.clear();
    cv::Matx33d basis_rows = cv::Matx33d::eye();
    cv::Vec3d center(0.0, 0.0, 0.0), local_min, local_max;
    if (!computeOrientedBounds(gt_mesh, basis_rows, center, local_min, local_max))
    {
        return;
    }

    cv::Vec3d local_extent = local_max - local_min;
    local_extent[0] *= 0.3;
    local_extent[1] *= 0.7;
    local_extent[2] *= 0.7;
    const cv::Vec3d local_center = 0.5 * (local_min + local_max);
    const cv::Matx33d basis_cols = basis_rows.t();

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uni01(0.0f, 1.0f);
    std::uniform_real_distribution<float> dir_uni(-1.0f, 1.0f);

    CameraIntrinsics K;
    K.w = gs.width;
    K.h = gs.height;
    K.fx = gs.focal;
    K.fy = gs.focal;
    K.cx = static_cast<float>(gs.width) * 0.5f - 0.5f;
    K.cy = static_cast<float>(gs.height) * 0.5f - 0.5f;

    const int max_trials = 50;
    for (int i = 0; i < gs.n_views; ++i)
    {
        cv::Matx44f chosen = cv::Matx44f::eye();
        bool accepted = false;
        for (int trial = 0; trial < max_trials; ++trial)
        {
            cv::Vec3d pos_local(
                local_center[0] + (uni01(rng) - 0.5f) * local_extent[0],
                local_center[1] + (uni01(rng) - 0.5f) * local_extent[1],
                local_center[2] + (uni01(rng) - 0.5f) * local_extent[2] + 0.4);

            cv::Vec3d pos_world_d = center + basis_cols * pos_local;
            cv::Vec3f pos_world(
                static_cast<float>(pos_world_d[0]),
                static_cast<float>(pos_world_d[1]),
                static_cast<float>(pos_world_d[2]));

            cv::Vec3f dir(dir_uni(rng), dir_uni(rng), dir_uni(rng));
            if (dir.dot(dir) < 1e-8f)
            {
                dir = cv::Vec3f(0.0f, 0.0f, 1.0f);
            }
            chosen = makeLookAtC2W(dir, cv::Vec3f(0.0f, 0.0f, -1.0f), pos_world);

            if (!unseen_pts.empty() && projectsAnyPoints(unseen_pts, chosen, K))
            {
                continue;
            }
            accepted = true;
            break;
        }
        if (accepted)
        {
            traj.push_back(chosen);
        }
    }
}

cv::Mat pointsToMat(const std::vector<cv::Point3f>& pts)
{
    cv::Mat m(static_cast<int>(pts.size()), 3, CV_32F);
    for (int i = 0; i < m.rows; ++i)
    {
        m.at<float>(i, 0) = pts[i].x;
        m.at<float>(i, 1) = pts[i].y;
        m.at<float>(i, 2) = pts[i].z;
    }
    return m;
}

std::vector<float> nearestDistancesL2(
    const std::vector<cv::Point3f>& src,
    const std::vector<cv::Point3f>& tgt)
{
    if (src.empty() || tgt.empty()) return {};

    cv::Mat tgt_m = pointsToMat(tgt);
    cv::flann::Index kdtree(tgt_m, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_L2);

    cv::Mat src_m = pointsToMat(src);
    cv::Mat idx(src_m.rows, 1, CV_32S);
    cv::Mat d2(src_m.rows, 1, CV_32F);
    kdtree.knnSearch(src_m, idx, d2, 1, cv::flann::SearchParams(64));

    std::vector<float> dist(src_m.rows, 0.0f);
    for (int i = 0; i < src_m.rows; ++i)
    {
        dist[i] = std::sqrt(std::max(0.0f, d2.at<float>(i, 0)));
    }
    return dist;
}

std::vector<cv::Point3f> voxelDownsamplePoints(
    const std::vector<cv::Point3f>& pts,
    float voxel_size)
{
    if (pts.empty() || !(voxel_size > 0.0f)) return pts;

    struct Key
    {
        int x = 0;
        int y = 0;
        int z = 0;
        bool operator<(const Key& other) const
        {
            if (x != other.x) return x < other.x;
            if (y != other.y) return y < other.y;
            return z < other.z;
        }
    };

    std::map<Key, std::pair<cv::Vec3d, int>> acc;
    const double inv = 1.0 / static_cast<double>(voxel_size);
    for (const auto& p : pts)
    {
        Key k;
        k.x = static_cast<int>(std::floor(static_cast<double>(p.x) * inv));
        k.y = static_cast<int>(std::floor(static_cast<double>(p.y) * inv));
        k.z = static_cast<int>(std::floor(static_cast<double>(p.z) * inv));
        auto& slot = acc[k];
        slot.first += cv::Vec3d(p.x, p.y, p.z);
        slot.second += 1;
    }

    std::vector<cv::Point3f> out;
    out.reserve(acc.size());
    for (const auto& kv : acc)
    {
        const cv::Vec3d mean = kv.second.first * (1.0 / std::max(1, kv.second.second));
        out.emplace_back(
            static_cast<float>(mean[0]),
            static_cast<float>(mean[1]),
            static_cast<float>(mean[2]));
    }
    return out;
}

double meanOf(const std::vector<float>& v)
{
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (float x : v) s += static_cast<double>(x);
    return s / static_cast<double>(v.size());
}

double ratioBelow(const std::vector<float>& v, float th)
{
    if (v.empty()) return 0.0;
    size_t c = 0;
    for (float x : v) if (x < th) ++c;
    return static_cast<double>(c) / static_cast<double>(v.size());
}

double percentileOf(const std::vector<float>& values, double quantile)
{
    if (values.empty()) return 0.0;
    std::vector<float> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const double q = std::clamp(quantile, 0.0, 1.0);
    const double position = q * static_cast<double>(sorted.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(position));
    const size_t upper = static_cast<size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return (1.0 - fraction) * static_cast<double>(sorted[lower]) +
           fraction * static_cast<double>(sorted[upper]);
}

FloaterSummary summarizeFloaters(
    const std::vector<float>& distances,
    const std::vector<float>& thresholds_m)
{
    FloaterSummary summary;
    summary.count = static_cast<uint64_t>(distances.size());
    summary.mean_m = meanOf(distances);
    summary.p95_m = percentileOf(distances, 0.95);
    summary.p99_m = percentileOf(distances, 0.99);
    if (distances.empty()) return summary;

    summary.thresholds.reserve(thresholds_m.size());
    for (float threshold_m : thresholds_m)
    {
        FloaterThresholdStats stats;
        stats.threshold_m = threshold_m;
        for (size_t i = 0; i < distances.size(); ++i)
        {
            if (distances[i] < threshold_m) continue;
            ++stats.farther_count;
        }
        stats.farther_ratio =
            static_cast<double>(stats.farther_count) / static_cast<double>(distances.size());
        summary.thresholds.push_back(stats);
    }
    return summary;
}

void drawChartFrame(
    cv::Mat& image,
    const std::string& title,
    const std::string& x_label,
    const std::string& y_label,
    int left,
    int top,
    int right,
    int bottom)
{
    const cv::Scalar text_color(45, 45, 45);
    const cv::Scalar axis_color(90, 90, 90);
    cv::line(image, cv::Point(left, bottom), cv::Point(right, bottom), axis_color, 1, cv::LINE_AA);
    cv::line(image, cv::Point(left, bottom), cv::Point(left, top), axis_color, 1, cv::LINE_AA);
    cv::putText(image, title, cv::Point(left, 38), cv::FONT_HERSHEY_SIMPLEX,
                0.75, text_color, 2, cv::LINE_AA);
    cv::putText(image, x_label, cv::Point((left + right) / 2 - 75, image.rows - 18),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv::LINE_AA);
    cv::putText(image, y_label, cv::Point(10, top - 12),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv::LINE_AA);
}

bool writeFloaterCountCurve(
    const std::filesystem::path& out_dir,
    const std::vector<float>& distances,
    float threshold_step_m,
    float max_distance_m,
    const std::string& artifact_stem,
    const std::string& chart_title,
    const std::string& sample_column,
    const std::string& y_label,
    const cv::Scalar& bar_color)
{
    if (distances.empty() || !(threshold_step_m > 0.0f) || !(max_distance_m > 0.0f)) return false;

    static constexpr std::array<const char*, 8> kLegacyArtifacts = {{
        "surface_floater_histogram.csv", "surface_floater_histogram.png",
        "surface_floater_tail_curve.csv", "surface_floater_tail_curve.png",
        "voxel_floater_histogram.csv", "voxel_floater_histogram.png",
        "voxel_floater_tail_curve.csv", "voxel_floater_tail_curve.png"
    }};
    for (const char* filename : kLegacyArtifacts)
    {
        std::error_code ignored;
        std::filesystem::remove(out_dir / filename, ignored);
    }

    const int threshold_count =
        std::max(1, static_cast<int>(std::floor(max_distance_m / threshold_step_m)));
    std::vector<uint64_t> farther_counts(static_cast<size_t>(threshold_count), 0);
    for (int step = 1; step <= threshold_count; ++step)
    {
        const float threshold = static_cast<float>(step) * threshold_step_m;
        uint64_t count = 0;
        for (float distance : distances)
        {
            if (distance >= threshold) ++count;
        }
        farther_counts[static_cast<size_t>(step - 1)] = count;
    }

    const auto csv_path = out_dir / (artifact_stem + ".csv");
    const auto png_path = out_dir / (artifact_stem + ".png");
    {
        std::ofstream out(csv_path);
        out << "distance_threshold_cm," << sample_column << ",fraction_farther\n";
        out << std::fixed << std::setprecision(6);
        for (int step = 1; step <= threshold_count; ++step)
        {
            const uint64_t count = farther_counts[static_cast<size_t>(step - 1)];
            out << 100.0 * static_cast<double>(step) * threshold_step_m << ','
                << count << ','
                << static_cast<double>(count) / static_cast<double>(distances.size()) << '\n';
        }
    }

    constexpr int width = 1100;
    constexpr int height = 650;
    constexpr int left = 95;
    constexpr int top = 70;
    constexpr int right = width - 35;
    constexpr int bottom = height - 75;
    const cv::Scalar grid_color(225, 225, 225);
    const cv::Scalar text_color(70, 70, 70);

    cv::Mat image(height, width, CV_8UC3, cv::Scalar(250, 250, 250));
    drawChartFrame(image, chart_title,
                   "Distance threshold (cm)", y_label, left, top, right, bottom);
    const uint64_t max_count = *std::max_element(farther_counts.begin(), farther_counts.end());
    const double y_max = static_cast<double>(std::max<uint64_t>(1, max_count));
    for (int grid = 0; grid <= 5; ++grid)
    {
        const int y = bottom - (bottom - top) * grid / 5;
        cv::line(image, cv::Point(left, y), cv::Point(right, y), grid_color, 1, cv::LINE_AA);
        const uint64_t label = static_cast<uint64_t>(std::llround(y_max * grid / 5.0));
        cv::putText(image, std::to_string(label), cv::Point(8, y + 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1, cv::LINE_AA);
    }
    const double slot_width = static_cast<double>(right - left) / threshold_count;
    for (int step = 0; step < threshold_count; ++step)
    {
        const int x0 = left + static_cast<int>(std::floor(step * slot_width));
        const int x1 = left + static_cast<int>(std::floor((step + 1) * slot_width)) - 1;
        const int y = bottom - static_cast<int>(std::round(
            (bottom - top) * farther_counts[static_cast<size_t>(step)] / y_max));
        cv::rectangle(image, cv::Point(x0 + 1, y), cv::Point(std::max(x0 + 1, x1), bottom - 1),
                      bar_color, cv::FILLED);
    }
    const int tick_every = std::max(1, static_cast<int>(std::lround(0.10f / threshold_step_m)));
    for (int step = tick_every; step <= threshold_count; step += tick_every)
    {
        const int x = left + static_cast<int>(std::round((right - left) *
            static_cast<double>(step) / threshold_count));
        cv::line(image, cv::Point(x, bottom), cv::Point(x, bottom + 5), text_color, 1);
        const int cm = static_cast<int>(std::lround(100.0f * step * threshold_step_m));
        cv::putText(image, std::to_string(cm), cv::Point(x - 10, bottom + 22),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv::LINE_AA);
    }
    cv::imwrite(png_path.string(), image);

    return true;
}

static inline float edgeFunction(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& p)
{
    return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
}

void renderDepthFromMesh(
    const MeshData& mesh,
    const cv::Matx44f& T_wc_or_w2c,
    bool traj_is_c2w,
    const CameraIntrinsics& K,
    float near_z,
    float far_z,
    cv::Mat1f& depth)
{
    const cv::Matx44f T_w2c = traj_is_c2w ? T_wc_or_w2c.inv() : T_wc_or_w2c;

    depth = cv::Mat1f(K.h, K.w, std::numeric_limits<float>::infinity());
    if (mesh.vertices.empty() || mesh.faces.empty())
    {
        depth.setTo(0.0f);
        return;
    }

    struct ProjVertex
    {
        cv::Point2f uv;
        float z = 0.0f;
        float invz = 0.0f;
        bool valid = false;
    };

    std::vector<ProjVertex> proj(mesh.vertices.size());
    for (size_t i = 0; i < mesh.vertices.size(); ++i)
    {
        const cv::Point3f& vw = mesh.vertices[i];
        const cv::Vec4f pw(vw.x, vw.y, vw.z, 1.0f);
        const cv::Vec4f pc4 = T_w2c * pw;
        const float z = pc4[2];
        if (!(z > near_z && z < far_z))
        {
            proj[i].valid = false;
            continue;
        }

        const float u = K.fx * (pc4[0] / z) + K.cx;
        const float v = K.fy * (pc4[1] / z) + K.cy;
        proj[i].uv = cv::Point2f(u, v);
        proj[i].z = z;
        proj[i].invz = 1.0f / z;
        proj[i].valid = true;
    }

    constexpr float eps = 1e-7f;
    for (const cv::Vec3i& f : mesh.faces)
    {
        const int i0 = f[0], i1 = f[1], i2 = f[2];
        if (i0 < 0 || i1 < 0 || i2 < 0 ||
            i0 >= static_cast<int>(proj.size()) ||
            i1 >= static_cast<int>(proj.size()) ||
            i2 >= static_cast<int>(proj.size()))
        {
            continue;
        }

        const ProjVertex& v0 = proj[i0];
        const ProjVertex& v1 = proj[i1];
        const ProjVertex& v2 = proj[i2];
        if (!v0.valid || !v1.valid || !v2.valid) continue;

        const float min_xf = std::min({v0.uv.x, v1.uv.x, v2.uv.x});
        const float max_xf = std::max({v0.uv.x, v1.uv.x, v2.uv.x});
        const float min_yf = std::min({v0.uv.y, v1.uv.y, v2.uv.y});
        const float max_yf = std::max({v0.uv.y, v1.uv.y, v2.uv.y});

        if (max_xf < 0.0f || max_yf < 0.0f || min_xf > static_cast<float>(K.w - 1) || min_yf > static_cast<float>(K.h - 1))
        {
            continue;
        }

        const int min_x = std::max(0, static_cast<int>(std::floor(min_xf)));
        const int max_x = std::min(K.w - 1, static_cast<int>(std::ceil(max_xf)));
        const int min_y = std::max(0, static_cast<int>(std::floor(min_yf)));
        const int max_y = std::min(K.h - 1, static_cast<int>(std::ceil(max_yf)));
        if (min_x > max_x || min_y > max_y) continue;

        const float denom = edgeFunction(v1.uv, v2.uv, v0.uv);
        if (std::abs(denom) <= eps) continue;

        for (int y = min_y; y <= max_y; ++y)
        {
            const float py = static_cast<float>(y) + 0.5f;
            for (int x = min_x; x <= max_x; ++x)
            {
                const float px = static_cast<float>(x) + 0.5f;
                const cv::Point2f p(px, py);

                const float w0 = edgeFunction(v1.uv, v2.uv, p) / denom;
                const float w1 = edgeFunction(v2.uv, v0.uv, p) / denom;
                const float w2 = edgeFunction(v0.uv, v1.uv, p) / denom;
                if (w0 < -eps || w1 < -eps || w2 < -eps) continue;

                const float invz = w0 * v0.invz + w1 * v1.invz + w2 * v2.invz;
                if (invz <= eps) continue;
                const float z = 1.0f / invz;
                if (!(z > near_z && z < far_z)) continue;

                float& zbuf = depth(y, x);
                if (z < zbuf) zbuf = z;
            }
        }
    }

    for (int y = 0; y < depth.rows; ++y)
    {
        float* row = depth.ptr<float>(y);
        for (int x = 0; x < depth.cols; ++x)
        {
            if (!std::isfinite(row[x])) row[x] = 0.0f;
        }
    }
}

float clampDepthHeatmapError(float err_m, float clip_max_m)
{
    if (clip_max_m > 0.0f)
    {
        return std::min(err_m, clip_max_m);
    }
    return err_m;
}

DepthHeatmapVizRange computeDepthHeatmapVizRange(
    const cv::Mat1f& abs_err,
    const cv::Mat1b& valid_mask,
    float clip_max_m)
{
    CV_Assert(abs_err.size() == valid_mask.size());

    DepthHeatmapVizRange range;
    float min_err = std::numeric_limits<float>::max();
    float max_err = 0.0f;
    for (int y = 0; y < abs_err.rows; ++y)
    {
        const float* err_row = abs_err.ptr<float>(y);
        const uchar* mask_row = valid_mask.ptr<uchar>(y);
        for (int x = 0; x < abs_err.cols; ++x)
        {
            if (!mask_row[x]) continue;
            const float err = clampDepthHeatmapError(err_row[x], clip_max_m);
            min_err = std::min(min_err, err);
            max_err = std::max(max_err, err);
            range.valid = true;
        }
    }

    if (!range.valid)
    {
        return range;
    }

    range.min_m = min_err;
    range.max_m = max_err;
    if (range.max_m - range.min_m < 1e-6f)
    {
        range.max_m = range.min_m + 1e-6f;
    }
    return range;
}

cv::Mat renderDepthL1HeatmapBgr(
    const cv::Mat1f& abs_err,
    const cv::Mat1b& valid_mask,
    const DepthHeatmapVizRange& viz_range,
    float clip_max_m)
{
    CV_Assert(abs_err.size() == valid_mask.size());

    cv::Mat1b gray(abs_err.rows, abs_err.cols, uchar(0));
    for (int y = 0; y < abs_err.rows; ++y)
    {
        const float* err_row = abs_err.ptr<float>(y);
        const uchar* mask_row = valid_mask.ptr<uchar>(y);
        uchar* gray_row = gray.ptr<uchar>(y);
        for (int x = 0; x < abs_err.cols; ++x)
        {
            if (!mask_row[x]) continue;
            const float err = clampDepthHeatmapError(err_row[x], clip_max_m);
            const float norm =
                std::clamp((err - viz_range.min_m) / (viz_range.max_m - viz_range.min_m), 0.0f, 1.0f);
            gray_row[x] = static_cast<uchar>(std::lround(norm * 255.0f));
        }
    }

    cv::Mat heat_bgr;
    cv::applyColorMap(gray, heat_bgr, cv::COLORMAP_JET);
    heat_bgr.setTo(cv::Scalar(0, 0, 0), valid_mask == 0);
    return heat_bgr;
}

cv::Mat appendDepthHeatmapLegend(
    const cv::Mat& heat_bgr,
    const DepthHeatmapVizRange& viz_range,
    float clip_max_m)
{
    if (heat_bgr.empty()) return heat_bgr;

    const int legend_pad = 16;
    const int legend_bar_w = 28;
    const int legend_w = 190;
    const int canvas_w = heat_bgr.cols + legend_w;
    cv::Mat canvas(heat_bgr.rows, canvas_w, CV_8UC3, cv::Scalar(20, 20, 20));
    heat_bgr.copyTo(canvas(cv::Rect(0, 0, heat_bgr.cols, heat_bgr.rows)));

    const int bar_h = std::max(80, heat_bgr.rows - 2 * legend_pad);
    const int bar_x = heat_bgr.cols + 20;
    const int bar_y = (heat_bgr.rows - bar_h) / 2;

    cv::Mat1b legend_gray(bar_h, 1);
    for (int y = 0; y < bar_h; ++y)
    {
        const float t = bar_h > 1 ? (1.0f - static_cast<float>(y) / static_cast<float>(bar_h - 1)) : 1.0f;
        legend_gray(y, 0) = static_cast<uchar>(std::lround(t * 255.0f));
    }
    cv::Mat legend_gray_resized;
    cv::resize(legend_gray, legend_gray_resized, cv::Size(legend_bar_w, bar_h), 0.0, 0.0, cv::INTER_LINEAR);
    cv::Mat legend_bgr;
    cv::applyColorMap(legend_gray_resized, legend_bgr, cv::COLORMAP_JET);
    legend_bgr.copyTo(canvas(cv::Rect(bar_x, bar_y, legend_bar_w, bar_h)));
    cv::rectangle(canvas, cv::Rect(bar_x, bar_y, legend_bar_w, bar_h), cv::Scalar(255, 255, 255), 1);

    const int text_x = bar_x + legend_bar_w + 12;
    const auto put = [&](const std::string& s, int x, int y, double scale = 0.5)
    {
        cv::putText(
            canvas,
            s,
            cv::Point(x, y),
            cv::FONT_HERSHEY_SIMPLEX,
            scale,
            cv::Scalar(255, 255, 255),
            1,
            cv::LINE_AA);
    };

    std::ostringstream ss_max;
    ss_max << std::fixed << std::setprecision(4) << viz_range.max_m << " m";
    std::ostringstream ss_min;
    ss_min << std::fixed << std::setprecision(4) << viz_range.min_m << " m";

    put("High", text_x, bar_y + 14);
    put(ss_max.str(), text_x, bar_y + 34);
    put("Low", text_x, bar_y + bar_h - 18);
    put(ss_min.str(), text_x, bar_y + bar_h);
    put("red = high", bar_x - 4, std::max(18, bar_y - 26));
    put("blue = low", bar_x - 4, std::min(canvas.rows - 8, bar_y + bar_h + 24));
    if (clip_max_m > 0.0f)
    {
        std::ostringstream ss_cap;
        ss_cap << "cap " << std::fixed << std::setprecision(3) << clip_max_m << " m";
        put(ss_cap.str(), bar_x - 4, std::min(canvas.rows - 28, bar_y + bar_h + 46));
    }

    return canvas;
}

void annotateDepthHeatmap(
    cv::Mat& heat_bgr,
    const std::string& label,
    double mean_l1_m,
    const DepthHeatmapVizRange& viz_range,
    float clip_max_m)
{
    if (heat_bgr.empty()) return;
    std::ostringstream ss0;
    ss0 << label << "  mean_l1=" << std::fixed << std::setprecision(4) << mean_l1_m << "m";
    std::ostringstream ss1;
    ss1 << "viz min=" << std::fixed << std::setprecision(4) << viz_range.min_m
        << "m  max=" << std::fixed << std::setprecision(4) << viz_range.max_m
        << "m  blue=low  red=high";
    if (clip_max_m > 0.0f)
    {
        ss1 << "  cap=" << std::fixed << std::setprecision(3) << clip_max_m << "m";
    }
    cv::putText(
        heat_bgr,
        ss0.str(),
        cv::Point(12, 24),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(255, 255, 255),
        2,
        cv::LINE_AA);
    cv::putText(
        heat_bgr,
        ss1.str(),
        cv::Point(12, 48),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(255, 255, 255),
        2,
        cv::LINE_AA);
}

void saveDepthL1Heatmap(
    const cv::Mat1f& abs_err,
    const cv::Mat1b& valid_mask,
    float clip_max_m,
    const std::filesystem::path& out_path,
    const std::string& label)
{
    double sum_err = 0.0;
    uint64_t count = 0;
    for (int y = 0; y < abs_err.rows; ++y)
    {
        const float* err_row = abs_err.ptr<float>(y);
        const uchar* mask_row = valid_mask.ptr<uchar>(y);
        for (int x = 0; x < abs_err.cols; ++x)
        {
            if (!mask_row[x]) continue;
            sum_err += static_cast<double>(err_row[x]);
            ++count;
        }
    }

    const double mean_l1_m = count > 0 ? (sum_err / static_cast<double>(count)) : 0.0;
    const DepthHeatmapVizRange viz_range = computeDepthHeatmapVizRange(abs_err, valid_mask, clip_max_m);
    cv::Mat heat_bgr = renderDepthL1HeatmapBgr(abs_err, valid_mask, viz_range, clip_max_m);
    cv::Mat heat_annotated = appendDepthHeatmapLegend(heat_bgr, viz_range, clip_max_m);
    annotateDepthHeatmap(heat_annotated, label, mean_l1_m, viz_range, clip_max_m);
    std::filesystem::create_directories(out_path.parent_path());
    cv::imwrite(out_path.string(), heat_annotated);
}

bool evaluateDepthMetricsFromMeshes(
    const MeshData& recon_mesh,
    const MeshData& gt_mesh,
    const std::vector<cv::Matx44f>& traj,
    bool traj_is_c2w,
    const CameraIntrinsics& K,
    float tau_m,
    int frame_stride,
    int max_frames,
    uint32_t seed,
    float near_z,
    float far_z,
    const DepthHeatmapSettings& heatmap_settings,
    DepthEvalStats& stats)
{
    if (recon_mesh.faces.empty() || gt_mesh.faces.empty())
    {
        std::cerr << "[mesh_eval] depth-from-mesh requires triangle meshes with faces. "
                  << "recon_faces=" << recon_mesh.faces.size()
                  << " gt_faces=" << gt_mesh.faces.size() << "\n";
        return false;
    }

    if (!K.valid())
    {
        std::cerr << "[mesh_eval] invalid camera intrinsics for depth evaluation\n";
        return false;
    }
    if (traj.empty())
    {
        std::cerr << "[mesh_eval] empty trajectory for depth evaluation\n";
        return false;
    }

    stats = DepthEvalStats{};
    int heat_saved = 0;
    const int stride = std::max(1, frame_stride);
    std::vector<int> eval_indices;
    eval_indices.reserve((traj.size() + static_cast<size_t>(stride) - 1) / static_cast<size_t>(stride));
    for (int i = 0; i < static_cast<int>(traj.size()); i += stride)
    {
        eval_indices.push_back(i);
    }
    if (eval_indices.empty())
    {
        return false;
    }

    if (max_frames > 0 && static_cast<int>(eval_indices.size()) > max_frames)
    {
        std::mt19937 rng(seed);
        std::shuffle(eval_indices.begin(), eval_indices.end(), rng);
        eval_indices.resize(static_cast<size_t>(max_frames));
        std::sort(eval_indices.begin(), eval_indices.end());
    }

    int used = 0;
    for (const int i : eval_indices)
    {
        cv::Mat1f d_recon, d_gt;
        renderDepthFromMesh(recon_mesh, traj[i], traj_is_c2w, K, near_z, far_z, d_recon);
        renderDepthFromMesh(gt_mesh, traj[i], traj_is_c2w, K, near_z, far_z, d_gt);

        uint64_t pred_count = 0;
        uint64_t gt_count = 0;
        uint64_t both_count = 0;
        uint64_t tp = 0;
        double abs_sum = 0.0;
        cv::Mat1f abs_err_frame;
        cv::Mat1b valid_mask;
        if (heatmap_settings.enabled)
        {
            abs_err_frame = cv::Mat1f(K.h, K.w, 0.0f);
            valid_mask = cv::Mat1b(K.h, K.w, uchar(0));
        }

        for (int y = 0; y < K.h; ++y)
        {
            const float* pr = d_recon.ptr<float>(y);
            const float* pg = d_gt.ptr<float>(y);
            for (int x = 0; x < K.w; ++x)
            {
                const float zr = pr[x];
                const float zg = pg[x];
                const bool vr = zr > 0.0f;
                const bool vg = zg > 0.0f;
                if (vr) ++pred_count;
                if (vg) ++gt_count;
                if (vr && vg)
                {
                    ++both_count;
                    const float err = std::abs(zr - zg);
                    abs_sum += static_cast<double>(err);
                    if (err < tau_m) ++tp;
                    if (heatmap_settings.enabled)
                    {
                        abs_err_frame(y, x) = err;
                        valid_mask(y, x) = 255;
                    }
                }
            }
        }

        if (heatmap_settings.enabled &&
            both_count > 0 &&
            (heatmap_settings.max_saved_frames <= 0 || heat_saved < heatmap_settings.max_saved_frames))
        {
            std::ostringstream name;
            name << "frame_" << std::setw(6) << std::setfill('0') << i << "_l1.png";
            saveDepthL1Heatmap(
                abs_err_frame,
                valid_mask,
                heatmap_settings.clip_max_m,
                heatmap_settings.out_dir / "depth_l1_heatmaps" / name.str(),
                "frame " + std::to_string(i));
            ++heat_saved;
        }

        stats.pred_pixels += pred_count;
        stats.gt_pixels += gt_count;
        stats.both_pixels += both_count;
        stats.tp_pixels += tp;
        stats.abs_l1_sum += abs_sum;
        ++stats.frames_used;
        ++used;
        if ((used % 50) == 0)
        {
            std::cout << "[mesh_eval][depth] processed_frames=" << used
                      << " both_pixels=" << stats.both_pixels << "\n";
        }
    }

    if (stats.frames_used == 0 || stats.both_pixels == 0 || stats.pred_pixels == 0 || stats.gt_pixels == 0)
    {
        return false;
    }

    stats.depth_l1_m = stats.abs_l1_sum / static_cast<double>(stats.both_pixels);
    stats.depth_precision = static_cast<double>(stats.tp_pixels) / static_cast<double>(stats.pred_pixels);
    stats.depth_recall = static_cast<double>(stats.tp_pixels) / static_cast<double>(stats.gt_pixels);
    const double d = stats.depth_precision + stats.depth_recall;
    stats.depth_f1 = d > 0.0 ? (2.0 * stats.depth_precision * stats.depth_recall / d) : 0.0;

    return true;
}

bool evaluateDepthMetricsGaussianSlamStyle(
    const MeshData& recon_mesh,
    const MeshData& gt_mesh,
    const GaussianSlamDepthSettings& gs,
    const std::vector<cv::Point3f>& unseen_pts,
    uint32_t seed,
    const DepthHeatmapSettings& heatmap_settings,
    DepthEvalStats& stats)
{
    if (recon_mesh.faces.empty() || gt_mesh.faces.empty())
    {
        std::cerr << "[mesh_eval] gaussian_slam depth evaluation requires triangle meshes with faces.\n";
        return false;
    }

    CameraIntrinsics K;
    K.w = gs.width;
    K.h = gs.height;
    K.fx = gs.focal;
    K.fy = gs.focal;
    K.cx = static_cast<float>(gs.width) * 0.5f - 0.5f;
    K.cy = static_cast<float>(gs.height) * 0.5f - 0.5f;

    std::vector<cv::Matx44f> views;
    generateGaussianSlamViews(gt_mesh, unseen_pts, gs, seed, views);
    if (views.empty())
    {
        std::cerr << "[mesh_eval] gaussian_slam depth evaluation could not sample any valid views\n";
        return false;
    }

    stats = DepthEvalStats{};
    int heat_saved = 0;
    for (size_t i = 0; i < views.size(); ++i)
    {
        cv::Mat1f d_recon, d_gt;
        renderDepthFromMesh(recon_mesh, views[i], /*traj_is_c2w=*/true, K, gs.near_z, gs.far_z, d_recon);
        renderDepthFromMesh(gt_mesh, views[i], /*traj_is_c2w=*/true, K, gs.near_z, gs.far_z, d_gt);

        double abs_sum = 0.0;
        uint64_t both_count = 0;
        uint64_t pred_count = 0;
        cv::Mat1f abs_err_frame;
        cv::Mat1b valid_mask;
        if (heatmap_settings.enabled)
        {
            abs_err_frame = cv::Mat1f(K.h, K.w, 0.0f);
            valid_mask = cv::Mat1b(K.h, K.w, uchar(0));
        }
        for (int y = 0; y < K.h; ++y)
        {
            const float* pr = d_recon.ptr<float>(y);
            const float* pg = d_gt.ptr<float>(y);
            for (int x = 0; x < K.w; ++x)
            {
                const bool vr = pr[x] > 0.0f;
                if (!vr) continue;
                ++pred_count;
                if (pg[x] > 0.0f)
                {
                    const float err = std::abs(static_cast<float>(pg[x]) - static_cast<float>(pr[x]));
                    abs_sum += static_cast<double>(err);
                    ++both_count;
                    if (heatmap_settings.enabled)
                    {
                        abs_err_frame(y, x) = err;
                        valid_mask(y, x) = 255;
                    }
                }
            }
        }

        if (pred_count == 0 || both_count == 0) continue;
        if (heatmap_settings.enabled &&
            (heatmap_settings.max_saved_frames <= 0 || heat_saved < heatmap_settings.max_saved_frames))
        {
            std::ostringstream name;
            name << "view_" << std::setw(6) << std::setfill('0') << i << "_l1.png";
            saveDepthL1Heatmap(
                abs_err_frame,
                valid_mask,
                heatmap_settings.clip_max_m,
                heatmap_settings.out_dir / "depth_l1_heatmaps" / name.str(),
                "view " + std::to_string(i));
            ++heat_saved;
        }
        stats.abs_l1_sum += abs_sum / static_cast<double>(both_count);
        ++stats.frames_used;
        stats.pred_pixels += pred_count;
        stats.both_pixels += both_count;
        if (((i + 1) % 50) == 0)
        {
            std::cout << "[mesh_eval][depth_gs] processed_views=" << (i + 1)
                      << " accepted=" << stats.frames_used << "\n";
        }
    }

    if (stats.frames_used == 0) return false;
    stats.depth_l1_m = stats.abs_l1_sum / static_cast<double>(stats.frames_used);
    stats.depth_precision = -1.0;
    stats.depth_recall = -1.0;
    stats.depth_f1 = -1.0;

    return true;
}
} // namespace

int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ? | | show help }"
        "{recon         | | reconstructed mesh/pcd ply }"
        "{gt            | | ground-truth mesh ply }"
        "{out           | | output directory }"
        "{tau_cm        |1.0| F-score threshold in cm (for both geometry and depth F1) }"
        "{eval_floaters |1| write surface/voxel floater metrics, CSV distributions, and plots }"
        "{floater_samples|100000| fixed number of reconstructed-surface samples for floater counts }"
        "{floater_bin_cm|1.0| distance-threshold step in cm for cumulative floater counts }"
        "{floater_max_cm|50.0| maximum plotted floater threshold in cm }"
        "{eval_gaussian_support|1| evaluate covariance support when Gaussian PLY attributes are present }"
        "{gaussian_support_sigma|3.0| finite Gaussian support contour in standard deviations }"
        "{gaussian_min_opacity|0.0| minimum activated opacity included in support sampling }"
        "{eval_voxel_support|1| evaluate cube support of zero-crossing SVRecon voxels when attributes are present }"
        "{support_samples_per_primitive|32| boundary probes used to assign one max-support distance per voxel/Gaussian }"
        "{recon_samples |500000| number of sampled points from reconstruction }"
        "{gt_samples    |500000| number of sampled points from GT }"
        "{seed          |0| random seed }"
        "{eval_mode     |current| evaluation mode: current, gaussian_slam, or gaussian_slam_sim3 }"
        "{eval_depth_mesh|0| evaluate depth L1/F1 by rendering both meshes }"
        "{save_depth_heatmaps|0| save per-view depth L1 heatmaps for mesh-depth evaluation }"
        "{depth_heatmap_max_frames|50| max number of per-view heatmaps to save (0=all evaluated views) }"
        "{depth_heatmap_clip_m|0.0| optional upper clamp in meters before per-image min/max normalization (<=0 disables) }"
        "{traj          | | trajectory txt (Nx16 row-major matrices) }"
        "{traj_mode     |c2w| trajectory mode: c2w or w2c }"
        "{cam_json      | | camera json with keys camera.w camera.h camera.fx camera.fy camera.cx camera.cy }"
        "{img_w         |0| image width (used if cam_json not given) }"
        "{img_h         |0| image height (used if cam_json not given) }"
        "{fx            |0| fx (used if cam_json not given) }"
        "{fy            |0| fy (used if cam_json not given) }"
        "{cx            |0| cx (used if cam_json not given) }"
        "{cy            |0| cy (used if cam_json not given) }"
        "{frame_stride  |1| prefilter trajectory by taking every N-th pose before random sampling }"
        "{max_frames    |1000| cap number of random poses for depth evaluation (0=all sampled) }"
        "{near          |0.05| near plane in meters }"
        "{far           |20.0| far plane in meters }"
        "{align_recon_to_gt|0| align reconstruction to GT: Sim(3) for current, rigid ICP for gaussian_slam, Sim(3)+rigid ICP for gaussian_slam_sim3 }"
        "{recon_traj_tum| | reconstructed camera trajectory in TUM format }"
        "{align_stride  |1| stride for trajectory pair sampling in Sim(3) fit }"
        "{align_max_pairs|0| max pose pairs for Sim(3) fit (0=all sampled pairs) }"
        "{save_aligned_mesh|0| save the aligned reconstruction mesh to the output directory }"
        "{gs_unseen_npy | | unseen GT point cloud .npy for gaussian_slam random-view rejection; if empty and --gt ends with _culled.ply, infer _pc_unseen.npy }"
        "{gs_depth_views|1000| number of random views for gaussian_slam depth evaluation }"
        "{gs_depth_w    |500| gaussian_slam depth image width }"
        "{gs_depth_h    |500| gaussian_slam depth image height }"
        "{gs_depth_focal|300.0| gaussian_slam focal length }"
        "{gs_icp_threshold|0.1| gaussian_slam rigid ICP threshold in meters }"
        "{gs_icp_max_iters|20| gaussian_slam rigid ICP max iterations }"
        "{gs_icp_max_points|100000| gaussian_slam rigid ICP max sampled vertices per mesh }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Mesh evaluation: geometry, floater distributions, and optional mesh-depth metrics");
    if (parser.has("help") || !parser.has("recon") || !parser.has("gt") || !parser.has("out"))
    {
        parser.printMessage();
        return 0;
    }

    const std::string recon_path = parser.get<std::string>("recon");
    const std::string gt_path = parser.get<std::string>("gt");
    const std::string out_dir = parser.get<std::string>("out");
    const float tau_cm = parser.get<float>("tau_cm");
    const float tau_m = tau_cm / 100.0f;
    const bool eval_floaters = parser.get<int>("eval_floaters") != 0;
    const int floater_samples = parser.get<int>("floater_samples");
    const float floater_bin_cm = parser.get<float>("floater_bin_cm");
    const float floater_max_cm = parser.get<float>("floater_max_cm");
    const bool eval_gaussian_support = parser.get<int>("eval_gaussian_support") != 0;
    const float gaussian_support_sigma = parser.get<float>("gaussian_support_sigma");
    const float gaussian_min_opacity = parser.get<float>("gaussian_min_opacity");
    const bool eval_voxel_support = parser.get<int>("eval_voxel_support") != 0;
    const int support_samples_per_primitive =
        parser.get<int>("support_samples_per_primitive");
    const int recon_samples = parser.get<int>("recon_samples");
    const int gt_samples = parser.get<int>("gt_samples");
    const int seed = parser.get<int>("seed");
    const std::string eval_mode_str = parser.get<std::string>("eval_mode");

    const bool eval_depth_mesh = parser.get<int>("eval_depth_mesh") != 0;
    const bool save_depth_heatmaps = parser.get<int>("save_depth_heatmaps") != 0;
    const int depth_heatmap_max_frames = parser.get<int>("depth_heatmap_max_frames");
    const float depth_heatmap_clip_m = parser.get<float>("depth_heatmap_clip_m");
    const std::string traj_path = parser.get<std::string>("traj");
    const std::string traj_mode = parser.get<std::string>("traj_mode");
    const std::string cam_json = parser.get<std::string>("cam_json");
    const int frame_stride = parser.get<int>("frame_stride");
    const int max_frames = parser.get<int>("max_frames");
    const float near_z = parser.get<float>("near");
    const float far_z = parser.get<float>("far");
    const bool align_recon_to_gt = parser.get<int>("align_recon_to_gt") != 0;
    const std::string recon_traj_tum_path = parser.get<std::string>("recon_traj_tum");
    const int align_stride = parser.get<int>("align_stride");
    const int align_max_pairs = parser.get<int>("align_max_pairs");
    const bool save_aligned_mesh = parser.get<int>("save_aligned_mesh") != 0;
    std::string gs_unseen_npy = parser.get<std::string>("gs_unseen_npy");

    std::filesystem::create_directories(out_dir);
    DepthHeatmapSettings heatmap_settings;
    heatmap_settings.enabled = eval_depth_mesh && save_depth_heatmaps;
    heatmap_settings.out_dir = out_dir;
    heatmap_settings.max_saved_frames = depth_heatmap_max_frames;
    heatmap_settings.clip_max_m = depth_heatmap_clip_m;
    GaussianSlamDepthSettings gs_settings;
    gs_settings.n_views = parser.get<int>("gs_depth_views");
    gs_settings.width = parser.get<int>("gs_depth_w");
    gs_settings.height = parser.get<int>("gs_depth_h");
    gs_settings.focal = parser.get<float>("gs_depth_focal");
    gs_settings.icp_threshold_m = parser.get<float>("gs_icp_threshold");
    gs_settings.icp_max_iters = parser.get<int>("gs_icp_max_iters");
    gs_settings.icp_max_points = parser.get<int>("gs_icp_max_points");

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }
    if (!std::filesystem::exists(recon_path))
    {
        std::cerr << "[mesh_eval] recon not found: " << recon_path << "\n";
        return 1;
    }
    if (!std::filesystem::exists(gt_path))
    {
        std::cerr << "[mesh_eval] gt not found: " << gt_path << "\n";
        return 1;
    }
    if (eval_floaters &&
        (floater_samples <= 0 || !(floater_bin_cm > 0.0f) || !(floater_max_cm > 0.0f)))
    {
        std::cerr << "[mesh_eval] floater_samples, floater_bin_cm, and floater_max_cm must be positive\n";
        return 1;
    }
    if (eval_gaussian_support &&
        (support_samples_per_primitive <= 0 || !(gaussian_support_sigma > 0.0f) ||
         gaussian_min_opacity < 0.0f || gaussian_min_opacity > 1.0f))
    {
        std::cerr << "[mesh_eval] invalid Gaussian support sampling parameters\n";
        return 1;
    }
    if (eval_voxel_support && support_samples_per_primitive <= 0)
    {
        std::cerr << "[mesh_eval] support_samples_per_primitive must be positive\n";
        return 1;
    }
    if (gs_unseen_npy.empty())
    {
        const std::string suffix = "_culled.ply";
        if (gt_path.size() >= suffix.size() &&
            gt_path.compare(gt_path.size() - suffix.size(), suffix.size(), suffix) == 0)
        {
            gs_unseen_npy = gt_path.substr(0, gt_path.size() - suffix.size()) + "_pc_unseen.npy";
            if (!std::filesystem::exists(gs_unseen_npy))
            {
                std::cout << "[mesh_eval] inferred gaussian_slam unseen file not found: "
                          << gs_unseen_npy << "\n";
                gs_unseen_npy.clear();
            }
            else
            {
                std::cout << "[mesh_eval] inferred gaussian_slam unseen file: "
                          << gs_unseen_npy << "\n";
            }
        }
    }

    EvalMode eval_mode = EvalMode::Current;
    if (!parseEvalMode(eval_mode_str, eval_mode))
    {
        std::cerr << "[mesh_eval] unsupported --eval_mode: " << eval_mode_str
                  << " (expected current, gaussian_slam, or gaussian_slam_sim3)\n";
        return 1;
    }

    MeshData recon_mesh, gt_mesh;
    if (!loadPlyMesh(recon_path, recon_mesh)) return 1;
    if (!loadPlyMesh(gt_path, gt_mesh)) return 1;
    const bool recon_has_faces = !recon_mesh.faces.empty();
    const bool gt_has_faces = !gt_mesh.faces.empty();
    const char* recon_geom_type = recon_has_faces ? "mesh" : "point_cloud";
    const char* gt_geom_type = gt_has_faces ? "mesh" : "point_cloud";

    std::cout << "[mesh_eval] recon geometry: " << recon_geom_type
              << " verts=" << recon_mesh.vertices.size()
              << " faces=" << recon_mesh.faces.size() << "\n";
    std::cout << "[mesh_eval] gt geometry: " << gt_geom_type
              << " verts=" << gt_mesh.vertices.size()
              << " faces=" << gt_mesh.faces.size() << "\n";

    if (!recon_has_faces)
    {
        std::cout << "[mesh_eval] recon has no faces; geometry metrics will use the vertex set as a point cloud.\n";
    }
    if (!gt_has_faces)
    {
        std::cout << "[mesh_eval] gt has no faces; geometry metrics will use the vertex set as a point cloud.\n";
    }

    if ((eval_mode == EvalMode::GaussianSlam || eval_mode == EvalMode::GaussianSlamSim3) && recon_has_faces)
    {
        cleanMeshLikeGaussianSlam(recon_mesh, 100);
    }

    bool traj_is_c2w = true;
    std::vector<cv::Matx44f> traj;
    const bool need_current_traj =
        ((eval_mode == EvalMode::Current) && (eval_depth_mesh || align_recon_to_gt)) ||
        ((eval_mode == EvalMode::GaussianSlamSim3) && align_recon_to_gt);
    if (need_current_traj)
    {
        if (traj_path.empty())
        {
            std::cerr << "[mesh_eval] --traj is required for the selected eval mode and alignment settings\n";
            return 1;
        }

        if (traj_mode == "c2w") traj_is_c2w = true;
        else if (traj_mode == "w2c") traj_is_c2w = false;
        else
        {
            std::cerr << "[mesh_eval] unsupported --traj_mode: " << traj_mode << " (expected c2w or w2c)\n";
            return 1;
        }

        if (!loadTrajectory4x4RowMajor(traj_path, traj)) return 1;
    }

    bool alignment_ok = false;
    double align_scale = 1.0;
    cv::Matx33d align_R = cv::Matx33d::eye();
    cv::Vec3d align_t(0.0, 0.0, 0.0);
    size_t align_pairs = 0;
    if (align_recon_to_gt)
    {
        if (eval_mode == EvalMode::Current || eval_mode == EvalMode::GaussianSlamSim3)
        {
            if (recon_traj_tum_path.empty())
            {
                std::cerr << "[mesh_eval] align_recon_to_gt=1 requires --recon_traj_tum\n";
                return 1;
            }

            std::vector<cv::Point3d> recon_centers;
            std::vector<cv::Point3d> gt_centers;
            if (!loadTumTrajectoryCenters(recon_traj_tum_path, recon_centers)) return 1;
            extractCentersFromTrajectory(traj, traj_is_c2w, gt_centers);

            if (!estimateSimilarityUmeyama(
                    recon_centers,
                    gt_centers,
                    align_stride,
                    align_max_pairs,
                    align_scale,
                    align_R,
                    align_t))
            {
                std::cerr << "[mesh_eval] failed to estimate Sim(3) from trajectories\n";
                return 1;
            }

            const size_t n_all = std::min(recon_centers.size(), gt_centers.size());
            align_pairs = 0;
            for (size_t i = 0; i < n_all; i += static_cast<size_t>(std::max(1, align_stride)))
            {
                ++align_pairs;
                if (align_max_pairs > 0 && static_cast<int>(align_pairs) >= align_max_pairs) break;
            }
            applySimilarityToMesh(recon_mesh, align_scale, align_R, align_t);
            alignment_ok = true;
            if (eval_mode == EvalMode::Current)
            {
                std::cout << "[mesh_eval] align_recon_to_gt enabled: pairs=" << align_pairs
                          << " scale=" << align_scale
                          << " t=(" << align_t[0] << "," << align_t[1] << "," << align_t[2] << ")\n";
            }
            else
            {
                std::cout << "[mesh_eval] gaussian_slam_sim3 pre-scale enabled: pairs=" << align_pairs
                          << " scale=" << align_scale
                          << " t=(" << align_t[0] << "," << align_t[1] << "," << align_t[2] << ")\n";

                int icp_inliers = 0;
                cv::Matx33d icp_R = cv::Matx33d::eye();
                cv::Vec3d icp_t(0.0, 0.0, 0.0);
                if (!estimateRigidIcpPointToPoint(
                        recon_mesh,
                        gt_mesh,
                        gs_settings.icp_threshold_m,
                        gs_settings.icp_max_iters,
                        gs_settings.icp_max_points,
                        static_cast<uint32_t>(seed),
                        icp_R,
                        icp_t,
                        icp_inliers))
                {
                    std::cerr << "[mesh_eval] failed to estimate rigid ICP refinement after Sim(3) pre-scale\n";
                    return 1;
                }
                applyRigidToMesh(recon_mesh, icp_R, icp_t);
                align_R = icp_R * align_R;
                align_t = icp_R * align_t + icp_t;
                align_pairs = static_cast<size_t>(icp_inliers);
                std::cout << "[mesh_eval] gaussian_slam_sim3 rigid ICP refinement: inliers="
                          << icp_inliers
                          << " threshold=" << gs_settings.icp_threshold_m
                          << " t=(" << icp_t[0] << "," << icp_t[1] << "," << icp_t[2] << ")\n";
            }
        }
        else
        {
            int icp_inliers = 0;
            if (!estimateRigidIcpPointToPoint(
                    recon_mesh,
                    gt_mesh,
                    gs_settings.icp_threshold_m,
                    gs_settings.icp_max_iters,
                    gs_settings.icp_max_points,
                    static_cast<uint32_t>(seed),
                    align_R,
                    align_t,
                    icp_inliers))
            {
                std::cerr << "[mesh_eval] failed to estimate rigid ICP alignment\n";
                return 1;
            }
            align_scale = 1.0;
            align_pairs = static_cast<size_t>(icp_inliers);
            applyRigidToMesh(recon_mesh, align_R, align_t);
            alignment_ok = true;
            std::cout << "[mesh_eval] gaussian_slam rigid ICP alignment enabled: inliers="
                      << icp_inliers
                      << " threshold=" << gs_settings.icp_threshold_m
                      << " t=(" << align_t[0] << "," << align_t[1] << "," << align_t[2] << ")\n";
        }
    }

    std::vector<cv::Point3f> recon_pts;
    std::vector<cv::Point3f> gt_pts;
    if (eval_mode == EvalMode::Current)
    {
        recon_pts = sampleMeshPoints(
            recon_mesh, static_cast<size_t>(std::max(1, recon_samples)), static_cast<uint32_t>(seed));
        gt_pts = sampleMeshPoints(
            gt_mesh, static_cast<size_t>(std::max(1, gt_samples)), static_cast<uint32_t>(seed + 1));
    }
    else
    {
        const size_t target_points =
            std::max(recon_mesh.vertices.size(), gt_mesh.vertices.size());
        if (gt_mesh.vertices.size() < recon_mesh.vertices.size())
        {
            recon_pts = sampleVertices(recon_mesh.vertices, target_points, static_cast<uint32_t>(seed));
            gt_pts = sampleMeshPoints(gt_mesh, target_points, static_cast<uint32_t>(seed + 1));
        }
        else
        {
            recon_pts = sampleMeshPoints(recon_mesh, target_points, static_cast<uint32_t>(seed));
            gt_pts = sampleVertices(gt_mesh.vertices, target_points, static_cast<uint32_t>(seed + 1));
        }
        recon_pts = voxelDownsamplePoints(recon_pts, tau_m);
        gt_pts = voxelDownsamplePoints(gt_pts, tau_m);
    }

    auto d_r2g = nearestDistancesL2(recon_pts, gt_pts); // Accuracy direction
    auto d_g2r = nearestDistancesL2(gt_pts, recon_pts); // Completeness direction

    const double accuracy = meanOf(d_r2g);
    const double completeness = meanOf(d_g2r);
    const double chamfer = accuracy + completeness;
    const double precision = ratioBelow(d_r2g, tau_m);
    const double recall = ratioBelow(d_g2r, tau_m);
    const double completion = recall;
    const double fscore = (precision + recall > 0.0)
                              ? (2.0 * precision * recall / (precision + recall))
                              : 0.0;

    std::vector<float> floater_thresholds_m = {0.01f, 0.02f, 0.05f, 0.10f};
    floater_thresholds_m.push_back(tau_m);
    std::sort(floater_thresholds_m.begin(), floater_thresholds_m.end());
    floater_thresholds_m.erase(
        std::unique(
            floater_thresholds_m.begin(),
            floater_thresholds_m.end(),
            [](float a, float b) { return std::abs(a - b) < 1.0e-6f; }),
        floater_thresholds_m.end());

    FloaterSummary surface_floater_summary;
    bool surface_floater_artifacts_saved = false;
    uint64_t surface_floater_count_at_tau = 0;
    double surface_floater_ratio_at_tau = 0.0;
    if (eval_floaters)
    {
        const auto floater_recon_pts = sampleMeshPoints(
            recon_mesh,
            static_cast<size_t>(floater_samples),
            static_cast<uint32_t>(seed + 101));
        const auto floater_gt_pts = sampleMeshPoints(
            gt_mesh,
            static_cast<size_t>(std::max(1, gt_samples)),
            static_cast<uint32_t>(seed + 102));
        const auto floater_distances = nearestDistancesL2(floater_recon_pts, floater_gt_pts);
        surface_floater_summary = summarizeFloaters(floater_distances, floater_thresholds_m);
        for (const auto& stats : surface_floater_summary.thresholds)
        {
            if (std::abs(stats.threshold_m - tau_m) >= 1.0e-6f) continue;
            surface_floater_count_at_tau = stats.farther_count;
            surface_floater_ratio_at_tau = stats.farther_ratio;
            break;
        }
        surface_floater_artifacts_saved = writeFloaterCountCurve(
            out_dir,
            floater_distances,
            floater_bin_cm / 100.0f,
            floater_max_cm / 100.0f,
            "surface_floater_count",
            recon_has_faces ? "Reconstructed surface floaters" : "Primitive-center floaters",
            recon_has_faces ? "surface_samples_farther" : "primitive_centers_farther",
            recon_has_faces ? "Surface samples farther" : "Primitive centers farther",
            cv::Scalar(190, 116, 45));
    }

    FloaterSummary gaussian_support_summary;
    bool gaussian_support_evaluated = false;
    bool gaussian_support_artifacts_saved = false;
    size_t gaussian_support_eligible = 0;
    double gaussian_support_opacity_weight = 0.0;
    uint64_t gaussian_support_count_at_tau = 0;
    double gaussian_support_ratio_at_tau = 0.0;
    if (eval_gaussian_support && recon_mesh.hasGaussianAttributes())
    {
        auto support_samples = sampleGaussianSupport(
            recon_mesh,
            static_cast<size_t>(support_samples_per_primitive),
            gaussian_support_sigma,
            gaussian_min_opacity,
            alignment_ok ? align_scale : 1.0,
            alignment_ok ? align_R : cv::Matx33d::eye(),
            static_cast<uint32_t>(seed + 201));
        gaussian_support_eligible = support_samples.eligible_gaussians;
        gaussian_support_opacity_weight = support_samples.opacity_weight;
        if (!support_samples.points.empty())
        {
            const auto support_gt_pts = sampleMeshPoints(
                gt_mesh,
                static_cast<size_t>(std::max(1, gt_samples)),
                static_cast<uint32_t>(seed + 202));
            const auto support_distances = nearestDistancesL2(
                support_samples.points, support_gt_pts);
            const auto primitive_distances = maxSupportDistancePerPrimitive(
                support_distances,
                support_samples.eligible_gaussians,
                support_samples.samples_per_primitive);
            gaussian_support_summary = summarizeFloaters(
                primitive_distances, floater_thresholds_m);
            for (const auto& stats : gaussian_support_summary.thresholds)
            {
                if (std::abs(stats.threshold_m - tau_m) >= 1.0e-6f) continue;
                gaussian_support_count_at_tau = stats.farther_count;
                gaussian_support_ratio_at_tau = stats.farther_ratio;
                break;
            }
            gaussian_support_artifacts_saved = writeFloaterCountCurve(
                out_dir,
                primitive_distances,
                floater_bin_cm / 100.0f,
                floater_max_cm / 100.0f,
                "gaussian_support_floater_count",
                "Gaussian 3-sigma support floaters",
                "gaussians_farther",
                "Gaussians farther",
                cv::Scalar(72, 72, 210));
            gaussian_support_evaluated = true;
            std::cout << "[mesh_eval] Gaussian support: sigma="
                      << gaussian_support_sigma
                      << " eligible=" << gaussian_support_eligible
                      << " probes=" << support_samples.points.size()
                      << " probes_per_gaussian=" << support_samples.samples_per_primitive
                      << " opacity_weight=" << gaussian_support_opacity_weight
                      << "\n";
        }
    }
    else if (eval_gaussian_support)
    {
        std::cout << "[mesh_eval] Gaussian support skipped: PLY has no complete "
                     "opacity/scale/rotation attributes.\n";
    }

    FloaterSummary voxel_support_summary;
    bool voxel_support_evaluated = false;
    bool voxel_support_artifacts_saved = false;
    size_t voxel_support_total = 0;
    size_t voxel_support_eligible = 0;
    double voxel_support_min_edge_m = 0.0;
    double voxel_support_max_edge_m = 0.0;
    uint64_t voxel_support_count_at_tau = 0;
    double voxel_support_ratio_at_tau = 0.0;
    if (eval_voxel_support && recon_mesh.hasVoxelAttributes())
    {
        auto support_samples = sampleVoxelSupport(
            recon_mesh,
            static_cast<size_t>(support_samples_per_primitive),
            alignment_ok ? align_scale : 1.0,
            alignment_ok ? align_R : cv::Matx33d::eye(),
            static_cast<uint32_t>(seed + 301));
        voxel_support_total = support_samples.total_voxels;
        voxel_support_eligible = support_samples.eligible_zero_crossing_voxels;
        voxel_support_min_edge_m = support_samples.min_edge_m;
        voxel_support_max_edge_m = support_samples.max_edge_m;
        if (!support_samples.points.empty())
        {
            const auto support_gt_pts = sampleMeshPoints(
                gt_mesh,
                static_cast<size_t>(std::max(1, gt_samples)),
                static_cast<uint32_t>(seed + 302));
            const auto support_distances = nearestDistancesL2(
                support_samples.points, support_gt_pts);
            const auto primitive_distances = maxSupportDistancePerPrimitive(
                support_distances,
                support_samples.eligible_zero_crossing_voxels,
                support_samples.samples_per_primitive);
            voxel_support_summary = summarizeFloaters(
                primitive_distances, floater_thresholds_m);
            for (const auto& stats : voxel_support_summary.thresholds)
            {
                if (std::abs(stats.threshold_m - tau_m) >= 1.0e-6f) continue;
                voxel_support_count_at_tau = stats.farther_count;
                voxel_support_ratio_at_tau = stats.farther_ratio;
                break;
            }
            voxel_support_artifacts_saved = writeFloaterCountCurve(
                out_dir,
                primitive_distances,
                floater_bin_cm / 100.0f,
                floater_max_cm / 100.0f,
                "voxel_support_floater_count",
                "Zero-crossing voxel cube-support floaters",
                "voxels_farther",
                "Voxels farther",
                cv::Scalar(70, 155, 70));
            voxel_support_evaluated = true;
            std::cout << "[mesh_eval] Voxel support: total="
                      << voxel_support_total
                      << " zero_crossing=" << voxel_support_eligible
                      << " probes=" << support_samples.points.size()
                      << " probes_per_voxel=" << support_samples.samples_per_primitive
                      << " edge_m=[" << voxel_support_min_edge_m
                      << "," << voxel_support_max_edge_m << "]\n";
        }
        else
        {
            std::cout << "[mesh_eval] Voxel support skipped: no zero-crossing cells.\n";
        }
    }
    else if (eval_voxel_support)
    {
        std::cout << "[mesh_eval] Voxel support skipped: PLY has no complete "
                     "scene_extent/octlevel/grid[0..7]_value attributes.\n";
    }

    DepthEvalStats depth_stats;
    bool depth_ok = false;
    if (eval_depth_mesh)
    {
        if (eval_mode == EvalMode::Current)
        {
            CameraIntrinsics K;
            if (!cam_json.empty())
            {
                if (!loadCameraIntrinsicsFromJson(cam_json, K))
                {
                    std::cerr << "[mesh_eval] failed to load camera intrinsics from --cam_json\n";
                    return 1;
                }
            }
            else
            {
                K.w = parser.get<int>("img_w");
                K.h = parser.get<int>("img_h");
                K.fx = parser.get<float>("fx");
                K.fy = parser.get<float>("fy");
                K.cx = parser.get<float>("cx");
                K.cy = parser.get<float>("cy");
                if (!K.valid())
                {
                    std::cerr << "[mesh_eval] provide either --cam_json or valid --img_w/--img_h/--fx/--fy/--cx/--cy\n";
                    return 1;
                }
            }
            std::cout << "[mesh_eval] depth-from-mesh enabled: traj=" << traj_path
                      << " mode=" << (traj_is_c2w ? "c2w" : "w2c")
                      << " K=(w=" << K.w << ",h=" << K.h
                      << ",fx=" << K.fx << ",fy=" << K.fy
                      << ",cx=" << K.cx << ",cy=" << K.cy << ")"
                      << " stride=" << std::max(1, frame_stride)
                      << " max_frames=" << max_frames
                      << " seed=" << seed
                      << " near=" << near_z
                      << " far=" << far_z
                      << " save_heatmaps=" << (heatmap_settings.enabled ? 1 : 0)
                      << "\n";

            depth_ok = evaluateDepthMetricsFromMeshes(
                recon_mesh,
                gt_mesh,
                traj,
                traj_is_c2w,
                K,
                tau_m,
                frame_stride,
                max_frames,
                static_cast<uint32_t>(seed),
                near_z,
                far_z,
                heatmap_settings,
                depth_stats);
        }
        else
        {
            std::vector<cv::Point3f> unseen_pts;
            if (!gs_unseen_npy.empty() && !loadNpyPoints3(gs_unseen_npy, unseen_pts))
            {
                std::cerr << "[mesh_eval] failed to load --gs_unseen_npy: " << gs_unseen_npy << "\n";
                return 1;
            }
            std::cout << "[mesh_eval] gaussian_slam depth-from-mesh enabled:"
                      << " views=" << gs_settings.n_views
                      << " K=(w=" << gs_settings.width << ",h=" << gs_settings.height
                      << ",fx=" << gs_settings.focal << ",fy=" << gs_settings.focal
                      << ",cx=" << (static_cast<float>(gs_settings.width) * 0.5f - 0.5f)
                      << ",cy=" << (static_cast<float>(gs_settings.height) * 0.5f - 0.5f) << ")"
                      << " unseen_pts=" << unseen_pts.size()
                      << " near=" << gs_settings.near_z
                      << " far=" << gs_settings.far_z
                      << " save_heatmaps=" << (heatmap_settings.enabled ? 1 : 0)
                      << "\n";

            depth_ok = evaluateDepthMetricsGaussianSlamStyle(
                recon_mesh,
                gt_mesh,
                gs_settings,
                unseen_pts,
                static_cast<uint32_t>(seed),
                heatmap_settings,
                depth_stats);
        }

        if (!depth_ok)
        {
            std::cerr << "[mesh_eval] depth-from-mesh evaluation produced no valid stats\n";
        }
    }

    const std::filesystem::path json_path = std::filesystem::path(out_dir) / "mesh_eval.json";
    const std::filesystem::path txt_path = std::filesystem::path(out_dir) / "mesh_eval.txt";
    const std::filesystem::path aligned_mesh_path =
        std::filesystem::path(out_dir) / "recon_mesh_aligned.ply";

    bool aligned_mesh_saved = false;
    if (save_aligned_mesh)
    {
        if (!alignment_ok)
        {
            std::cerr << "[mesh_eval] save_aligned_mesh=1 requires align_recon_to_gt=1 and successful alignment\n";
        }
        else
        {
            aligned_mesh_saved = savePlyMesh(aligned_mesh_path.string(), recon_mesh);
            if (aligned_mesh_saved)
            {
                std::cout << "[mesh_eval] saved aligned mesh: " << aligned_mesh_path << "\n";
            }
        }
    }

    Json::Value root;
    root["recon_mesh"] = recon_path;
    root["gt_mesh"] = gt_path;
    root["eval_mode"] = evalModeName(eval_mode);
    root["recon_geometry_type"] = recon_geom_type;
    root["gt_geometry_type"] = gt_geom_type;
    root["recon_has_faces"] = recon_has_faces;
    root["gt_has_faces"] = gt_has_faces;
    root["accuracy_mean"] = accuracy;
    root["completeness_mean"] = completeness;
    root["chamfer_l1"] = chamfer;
    root["chamfer_l1_mean"] = 0.5 * chamfer;
    root["precision"] = precision;
    root["recall"] = recall;
    root["completion"] = completion;
    root["completion_percent"] = 100.0 * completion;
    root["fscore"] = fscore;
    root["threshold_m"] = tau_m;
    root["threshold_cm"] = tau_cm;
    root["n_recon_samples"] = static_cast<Json::UInt64>(recon_pts.size());
    root["n_gt_samples"] = static_cast<Json::UInt64>(gt_pts.size());
    root["floater_evaluation_enabled"] = eval_floaters;
    root["floater_samples"] = floater_samples;
    root["floater_bin_cm"] = floater_bin_cm;
    root["floater_max_cm"] = floater_max_cm;
    if (eval_floaters)
    {
        root["surface_floater_count"] = static_cast<Json::UInt64>(surface_floater_count_at_tau);
        root["surface_floater_samples"] = static_cast<Json::UInt64>(surface_floater_summary.count);
        root["surface_floater_ratio"] = surface_floater_ratio_at_tau;
        root["surface_floater_threshold_cm"] = tau_cm;
        root["surface_distance_mean_m"] = surface_floater_summary.mean_m;
        root["surface_distance_p95_m"] = surface_floater_summary.p95_m;
        root["surface_distance_p99_m"] = surface_floater_summary.p99_m;
        root["surface_floater_artifacts_saved"] = surface_floater_artifacts_saved;
        if (surface_floater_artifacts_saved)
        {
            root["surface_floater_count_csv"] =
                (std::filesystem::path(out_dir) / "surface_floater_count.csv").string();
            root["surface_floater_count_png"] =
                (std::filesystem::path(out_dir) / "surface_floater_count.png").string();
        }
        Json::Value thresholds(Json::arrayValue);
        for (const auto& stats : surface_floater_summary.thresholds)
        {
            Json::Value item;
            item["threshold_cm"] = 100.0 * stats.threshold_m;
            item["farther_count"] = static_cast<Json::UInt64>(stats.farther_count);
            item["farther_ratio"] = stats.farther_ratio;
            thresholds.append(item);
        }
        root["surface_floater_thresholds"] = thresholds;
    }
    root["gaussian_support_requested"] = eval_gaussian_support;
    root["gaussian_support_evaluated"] = gaussian_support_evaluated;
    root["gaussian_support_sigma"] = gaussian_support_sigma;
    root["gaussian_support_min_opacity"] = gaussian_min_opacity;
    root["gaussian_support_samples_per_primitive"] = support_samples_per_primitive;
    if (gaussian_support_evaluated)
    {
        root["gaussian_support_eligible_gaussians"] =
            static_cast<Json::UInt64>(gaussian_support_eligible);
        root["gaussian_support_opacity_weight"] = gaussian_support_opacity_weight;
        root["gaussian_support_primitive_count"] =
            static_cast<Json::UInt64>(gaussian_support_summary.count);
        root["gaussian_support_probe_samples"] = static_cast<Json::UInt64>(
            gaussian_support_eligible * static_cast<size_t>(support_samples_per_primitive));
        root["gaussian_support_distance_statistic"] = "max_boundary_distance";
        root["gaussian_support_floater_count"] =
            static_cast<Json::UInt64>(gaussian_support_count_at_tau);
        root["gaussian_support_floater_ratio"] = gaussian_support_ratio_at_tau;
        root["gaussian_support_floater_threshold_cm"] = tau_cm;
        root["gaussian_support_distance_mean_m"] = gaussian_support_summary.mean_m;
        root["gaussian_support_distance_p95_m"] = gaussian_support_summary.p95_m;
        root["gaussian_support_distance_p99_m"] = gaussian_support_summary.p99_m;
        root["gaussian_support_artifacts_saved"] = gaussian_support_artifacts_saved;
        if (gaussian_support_artifacts_saved)
        {
            root["gaussian_support_floater_count_csv"] =
                (std::filesystem::path(out_dir) / "gaussian_support_floater_count.csv").string();
            root["gaussian_support_floater_count_png"] =
                (std::filesystem::path(out_dir) / "gaussian_support_floater_count.png").string();
        }
        Json::Value thresholds(Json::arrayValue);
        for (const auto& stats : gaussian_support_summary.thresholds)
        {
            Json::Value item;
            item["threshold_cm"] = 100.0 * stats.threshold_m;
            item["farther_count"] = static_cast<Json::UInt64>(stats.farther_count);
            item["farther_ratio"] = stats.farther_ratio;
            thresholds.append(item);
        }
        root["gaussian_support_floater_thresholds"] = thresholds;
    }
    root["voxel_support_requested"] = eval_voxel_support;
    root["voxel_support_evaluated"] = voxel_support_evaluated;
    root["voxel_support_samples_per_primitive"] = support_samples_per_primitive;
    if (voxel_support_evaluated)
    {
        root["voxel_support_total_voxels"] =
            static_cast<Json::UInt64>(voxel_support_total);
        root["voxel_support_eligible_zero_crossing_voxels"] =
            static_cast<Json::UInt64>(voxel_support_eligible);
        root["voxel_support_scene_extent"] = recon_mesh.voxel_scene_extent;
        root["voxel_support_min_edge_m"] = voxel_support_min_edge_m;
        root["voxel_support_max_edge_m"] = voxel_support_max_edge_m;
        root["voxel_support_primitive_count"] =
            static_cast<Json::UInt64>(voxel_support_summary.count);
        root["voxel_support_probe_samples"] = static_cast<Json::UInt64>(
            voxel_support_eligible * static_cast<size_t>(support_samples_per_primitive));
        root["voxel_support_distance_statistic"] = "max_boundary_distance";
        root["voxel_support_floater_count"] =
            static_cast<Json::UInt64>(voxel_support_count_at_tau);
        root["voxel_support_floater_ratio"] = voxel_support_ratio_at_tau;
        root["voxel_support_floater_threshold_cm"] = tau_cm;
        root["voxel_support_distance_mean_m"] = voxel_support_summary.mean_m;
        root["voxel_support_distance_p95_m"] = voxel_support_summary.p95_m;
        root["voxel_support_distance_p99_m"] = voxel_support_summary.p99_m;
        root["voxel_support_artifacts_saved"] = voxel_support_artifacts_saved;
        if (voxel_support_artifacts_saved)
        {
            root["voxel_support_floater_count_csv"] =
                (std::filesystem::path(out_dir) / "voxel_support_floater_count.csv").string();
            root["voxel_support_floater_count_png"] =
                (std::filesystem::path(out_dir) / "voxel_support_floater_count.png").string();
        }
        Json::Value thresholds(Json::arrayValue);
        for (const auto& stats : voxel_support_summary.thresholds)
        {
            Json::Value item;
            item["threshold_cm"] = 100.0 * stats.threshold_m;
            item["farther_count"] = static_cast<Json::UInt64>(stats.farther_count);
            item["farther_ratio"] = stats.farther_ratio;
            thresholds.append(item);
        }
        root["voxel_support_floater_thresholds"] = thresholds;
    }
    root["align_recon_to_gt_enabled"] = align_recon_to_gt;
    root["align_recon_to_gt_success"] = alignment_ok;
    root["save_aligned_mesh"] = save_aligned_mesh;
    root["aligned_mesh_saved"] = aligned_mesh_saved;
    if (alignment_ok)
    {
        root["align_scale"] = align_scale;
        root["align_pairs"] = static_cast<Json::UInt64>(align_pairs);
        root["align_t_x"] = align_t[0];
        root["align_t_y"] = align_t[1];
        root["align_t_z"] = align_t[2];
    }
    if (aligned_mesh_saved)
    {
        root["aligned_mesh_path"] = aligned_mesh_path.string();
    }

    root["depth_from_mesh_enabled"] = eval_depth_mesh;
    root["depth_from_mesh_success"] = depth_ok;
    root["save_depth_heatmaps"] = heatmap_settings.enabled;
    root["depth_heatmap_clip_m"] = heatmap_settings.clip_max_m;
    root["depth_heatmap_max_frames"] = heatmap_settings.max_saved_frames;
    if (heatmap_settings.enabled)
    {
        root["depth_heatmap_dir"] = (std::filesystem::path(out_dir) / "depth_l1_heatmaps").string();
    }
    if (eval_depth_mesh)
    {
        root["depth_frames_used"] = static_cast<Json::UInt64>(depth_stats.frames_used);
        root["depth_pred_pixels"] = static_cast<Json::UInt64>(depth_stats.pred_pixels);
        root["depth_gt_pixels"] = static_cast<Json::UInt64>(depth_stats.gt_pixels);
        root["depth_both_pixels"] = static_cast<Json::UInt64>(depth_stats.both_pixels);
        root["depth_tp_pixels"] = static_cast<Json::UInt64>(depth_stats.tp_pixels);
        root["depth_l1_m"] = depth_stats.depth_l1_m;
        root["depth_precision"] = depth_stats.depth_precision;
        root["depth_recall"] = depth_stats.depth_recall;
        root["depth_f1"] = depth_stats.depth_f1;
    }

    {
        std::ofstream f(json_path);
        Json::StreamWriterBuilder b;
        b["indentation"] = "  ";
        f << Json::writeString(b, root);
    }
    {
        std::ofstream f(txt_path);
        f << "## Mesh Evaluation\n";
        f << "eval_mode " << evalModeName(eval_mode) << "\n";
        f << "accuracy_mean " << accuracy << "\n";
        f << "completeness_mean " << completeness << "\n";
        f << "chamfer_l1 " << chamfer << "\n";
        f << "chamfer_l1_mean " << 0.5 * chamfer << "\n";
        f << "precision " << precision << "\n";
        f << "recall " << recall << "\n";
        f << "completion " << completion << "\n";
        f << "completion_percent " << 100.0 * completion << "\n";
        f << "fscore " << fscore << "\n";
        f << "recon_geometry_type " << recon_geom_type << "\n";
        f << "gt_geometry_type " << gt_geom_type << "\n";
        f << "recon_has_faces " << (recon_has_faces ? 1 : 0) << "\n";
        f << "gt_has_faces " << (gt_has_faces ? 1 : 0) << "\n";
        f << "threshold_m " << tau_m << "\n";
        f << "n_recon_samples " << recon_pts.size() << "\n";
        f << "n_gt_samples " << gt_pts.size() << "\n";
        f << "floater_evaluation_enabled " << (eval_floaters ? 1 : 0) << "\n";
        if (eval_floaters)
        {
            f << "floater_samples " << floater_samples << "\n";
            f << "surface_floater_samples " << surface_floater_summary.count << "\n";
            f << "surface_floater_count " << surface_floater_count_at_tau << "\n";
            f << "surface_floater_ratio " << surface_floater_ratio_at_tau << "\n";
            f << "surface_floater_threshold_cm " << tau_cm << "\n";
            f << "surface_distance_mean_m " << surface_floater_summary.mean_m << "\n";
            f << "surface_distance_p95_m " << surface_floater_summary.p95_m << "\n";
            f << "surface_distance_p99_m " << surface_floater_summary.p99_m << "\n";
            for (const auto& stats : surface_floater_summary.thresholds)
            {
                f << "surface_floater_at_cm " << 100.0 * stats.threshold_m
                  << " count " << stats.farther_count
                  << " ratio " << stats.farther_ratio << "\n";
            }
        }
        f << "gaussian_support_requested " << (eval_gaussian_support ? 1 : 0) << "\n";
        f << "gaussian_support_evaluated " << (gaussian_support_evaluated ? 1 : 0) << "\n";
        if (gaussian_support_evaluated)
        {
            f << "gaussian_support_sigma " << gaussian_support_sigma << "\n";
            f << "gaussian_support_min_opacity " << gaussian_min_opacity << "\n";
            f << "gaussian_support_eligible_gaussians " << gaussian_support_eligible << "\n";
            f << "gaussian_support_samples_per_primitive "
              << support_samples_per_primitive << "\n";
            f << "gaussian_support_probe_samples "
              << gaussian_support_eligible * static_cast<size_t>(support_samples_per_primitive)
              << "\n";
            f << "gaussian_support_primitive_count " << gaussian_support_summary.count << "\n";
            f << "gaussian_support_distance_statistic max_boundary_distance\n";
            f << "gaussian_support_floater_count " << gaussian_support_count_at_tau << "\n";
            f << "gaussian_support_floater_ratio " << gaussian_support_ratio_at_tau << "\n";
            f << "gaussian_support_distance_mean_m " << gaussian_support_summary.mean_m << "\n";
            f << "gaussian_support_distance_p95_m " << gaussian_support_summary.p95_m << "\n";
            f << "gaussian_support_distance_p99_m " << gaussian_support_summary.p99_m << "\n";
            for (const auto& stats : gaussian_support_summary.thresholds)
            {
                f << "gaussian_support_floater_at_cm " << 100.0 * stats.threshold_m
                  << " count " << stats.farther_count
                  << " ratio " << stats.farther_ratio << "\n";
            }
        }
        f << "voxel_support_requested " << (eval_voxel_support ? 1 : 0) << "\n";
        f << "voxel_support_evaluated " << (voxel_support_evaluated ? 1 : 0) << "\n";
        if (voxel_support_evaluated)
        {
            f << "voxel_support_total_voxels " << voxel_support_total << "\n";
            f << "voxel_support_eligible_zero_crossing_voxels "
              << voxel_support_eligible << "\n";
            f << "voxel_support_scene_extent " << recon_mesh.voxel_scene_extent << "\n";
            f << "voxel_support_min_edge_m " << voxel_support_min_edge_m << "\n";
            f << "voxel_support_max_edge_m " << voxel_support_max_edge_m << "\n";
            f << "voxel_support_samples_per_primitive "
              << support_samples_per_primitive << "\n";
            f << "voxel_support_probe_samples "
              << voxel_support_eligible * static_cast<size_t>(support_samples_per_primitive)
              << "\n";
            f << "voxel_support_primitive_count " << voxel_support_summary.count << "\n";
            f << "voxel_support_distance_statistic max_boundary_distance\n";
            f << "voxel_support_floater_count " << voxel_support_count_at_tau << "\n";
            f << "voxel_support_floater_ratio " << voxel_support_ratio_at_tau << "\n";
            f << "voxel_support_distance_mean_m " << voxel_support_summary.mean_m << "\n";
            f << "voxel_support_distance_p95_m " << voxel_support_summary.p95_m << "\n";
            f << "voxel_support_distance_p99_m " << voxel_support_summary.p99_m << "\n";
            for (const auto& stats : voxel_support_summary.thresholds)
            {
                f << "voxel_support_floater_at_cm " << 100.0 * stats.threshold_m
                  << " count " << stats.farther_count
                  << " ratio " << stats.farther_ratio << "\n";
            }
        }
        f << "align_recon_to_gt_enabled " << (align_recon_to_gt ? 1 : 0) << "\n";
        f << "align_recon_to_gt_success " << (alignment_ok ? 1 : 0) << "\n";
        f << "save_aligned_mesh " << (save_aligned_mesh ? 1 : 0) << "\n";
        f << "aligned_mesh_saved " << (aligned_mesh_saved ? 1 : 0) << "\n";
        if (alignment_ok)
        {
            f << "align_scale " << align_scale << "\n";
            f << "align_pairs " << align_pairs << "\n";
            f << "align_t_x " << align_t[0] << "\n";
            f << "align_t_y " << align_t[1] << "\n";
            f << "align_t_z " << align_t[2] << "\n";
        }
        if (aligned_mesh_saved)
        {
            f << "aligned_mesh_path " << aligned_mesh_path.string() << "\n";
        }

        f << "depth_from_mesh_enabled " << (eval_depth_mesh ? 1 : 0) << "\n";
        f << "depth_from_mesh_success " << (depth_ok ? 1 : 0) << "\n";
        f << "save_depth_heatmaps " << (heatmap_settings.enabled ? 1 : 0) << "\n";
        f << "depth_heatmap_clip_m " << heatmap_settings.clip_max_m << "\n";
        f << "depth_heatmap_max_frames " << heatmap_settings.max_saved_frames << "\n";
        if (heatmap_settings.enabled)
        {
            f << "depth_heatmap_dir " << (std::filesystem::path(out_dir) / "depth_l1_heatmaps").string() << "\n";
        }
        if (eval_depth_mesh)
        {
            f << "depth_frames_used " << depth_stats.frames_used << "\n";
            f << "depth_pred_pixels " << depth_stats.pred_pixels << "\n";
            f << "depth_gt_pixels " << depth_stats.gt_pixels << "\n";
            f << "depth_both_pixels " << depth_stats.both_pixels << "\n";
            f << "depth_tp_pixels " << depth_stats.tp_pixels << "\n";
            f << "depth_l1_m " << depth_stats.depth_l1_m << "\n";
            f << "depth_precision " << depth_stats.depth_precision << "\n";
            f << "depth_recall " << depth_stats.depth_recall << "\n";
            f << "depth_f1 " << depth_stats.depth_f1 << "\n";
        }
    }

    std::cout << "=== Mesh Eval ===\n";
    std::cout << "eval_mode: " << evalModeName(eval_mode) << "\n";
    std::cout << "recon: " << recon_path << "\n";
    std::cout << "gt:    " << gt_path << "\n";
    std::cout << "recon_geometry_type: " << recon_geom_type << "\n";
    std::cout << "gt_geometry_type:    " << gt_geom_type << "\n";
    std::cout << "Accuracy (recon->gt): " << accuracy << " m\n";
    std::cout << "Completeness (gt->recon): " << completeness << " m\n";
    std::cout << "Chamfer L1: " << chamfer << " m\n";
    std::cout << "Precision@" << tau_cm << "cm: " << precision << "\n";
    std::cout << "Recall@" << tau_cm << "cm: " << recall << "\n";
    std::cout << "Completion@" << tau_cm << "cm: " << completion
              << " (" << 100.0 * completion << "%)\n";
    std::cout << "F-score@" << tau_cm << "cm: " << fscore << "\n";
    if (eval_floaters)
    {
        std::cout << (recon_has_faces ? "Surface samples" : "Primitive centers")
                  << " farther than " << tau_cm << "cm: "
                  << surface_floater_count_at_tau << " / " << surface_floater_summary.count
                  << " (" << surface_floater_ratio_at_tau << ")\n";
        std::cout << (recon_has_faces ? "Surface" : "Primitive-center")
                  << " distance P95/P99: "
                  << surface_floater_summary.p95_m << " / "
                  << surface_floater_summary.p99_m << " m\n";
        std::cout << "Floater count plot: "
                  << (std::filesystem::path(out_dir) / "surface_floater_count.png") << "\n";
    }
    if (gaussian_support_evaluated)
    {
        std::cout << "Gaussians whose " << gaussian_support_sigma
                  << "sigma support extends farther than " << tau_cm << "cm: "
                  << gaussian_support_count_at_tau << " / " << gaussian_support_summary.count
                  << " (" << gaussian_support_ratio_at_tau << ")\n";
        std::cout << "Gaussian support distance P95/P99: "
                  << gaussian_support_summary.p95_m << " / "
                  << gaussian_support_summary.p99_m << " m\n";
        std::cout << "Gaussian support plot: "
                  << (std::filesystem::path(out_dir) /
                      "gaussian_support_floater_count.png") << "\n";
    }
    if (voxel_support_evaluated)
    {
        std::cout << "Zero-crossing voxels whose cube support extends farther than "
                  << tau_cm << "cm: "
                  << voxel_support_count_at_tau << " / " << voxel_support_summary.count
                  << " (" << voxel_support_ratio_at_tau << ")\n";
        std::cout << "Voxel support distance P95/P99: "
                  << voxel_support_summary.p95_m << " / "
                  << voxel_support_summary.p99_m << " m\n";
        std::cout << "Voxel support plot: "
                  << (std::filesystem::path(out_dir) /
                      "voxel_support_floater_count.png") << "\n";
    }
    if (alignment_ok)
    {
        if (eval_mode == EvalMode::Current)
        {
            std::cout << "Alignment Sim3: scale=" << align_scale
                      << " pairs=" << align_pairs
                      << " t=(" << align_t[0] << "," << align_t[1] << "," << align_t[2] << ")\n";
        }
        else if (eval_mode == EvalMode::GaussianSlam)
        {
            std::cout << "Alignment rigid ICP: inliers=" << align_pairs
                      << " t=(" << align_t[0] << "," << align_t[1] << "," << align_t[2] << ")\n";
        }
        else
        {
            std::cout << "Alignment Sim3 + rigid ICP: scale=" << align_scale
                      << " inliers=" << align_pairs
                      << " t=(" << align_t[0] << "," << align_t[1] << "," << align_t[2] << ")\n";
        }
    }
    if (aligned_mesh_saved)
    {
        std::cout << "Aligned mesh: " << aligned_mesh_path << "\n";
    }

    if (eval_depth_mesh)
    {
        if (depth_ok)
        {
            std::cout << "Depth L1 (mesh-rendered): " << depth_stats.depth_l1_m << " m\n";
            if (heatmap_settings.enabled)
            {
                std::cout << "Depth heatmaps: "
                          << (std::filesystem::path(out_dir) / "depth_l1_heatmaps") << "\n";
            }
            if (depth_stats.depth_precision >= 0.0)
            {
                std::cout << "Depth Precision@" << tau_cm << "cm: " << depth_stats.depth_precision << "\n";
                std::cout << "Depth Recall@" << tau_cm << "cm: " << depth_stats.depth_recall << "\n";
                std::cout << "Depth F1@" << tau_cm << "cm: " << depth_stats.depth_f1 << "\n";
            }
            std::cout << "Depth frames used: " << depth_stats.frames_used << "\n";
        }
        else
        {
            std::cout << "Depth-from-mesh evaluation failed (no valid pixels/frames).\n";
        }
    }

    std::cout << "saved: " << json_path << "\n";
    std::cout << "saved: " << txt_path << "\n";

    return 0;
}

// Replica 
// ./bin/mesh_eval \
//   --eval_mode=gaussian_slam_sim3 \
//   --recon=/home/dimitris/Photo-SLAM/results/replica_rgbd_voxel/office0/1/ply/voxel_model/iteration_2241/voxel_surface_mesh.ply \
//   --gt=/home/dimitris/Photo-SLAM/scripts/data/Replica/office0_mesh.ply \
//   --out=/home/dimitris/Photo-SLAM/results/replica_rgbd_voxel/office0/1/mesh_eval_gs_sim3 \
//   --tau_cm=1.0 \
//   --eval_depth_mesh=1 \
//   --align_recon_to_gt=1 \
//   --traj=/home/dimitris/Photo-SLAM/scripts/data/Replica/office0/traj.txt \
//   --traj_mode=c2w \
//   --recon_traj_tum=/home/dimitris/Photo-SLAM/results/replica_rgbd_voxel/office0/CameraTrajectory_TUM.txt \
//   --save_aligned_mesh=1

// TUM
//   ./bin/mesh_eval \
//   --recon=results/tum_rgbd/rgbd_dataset_freiburg1_desk/experiments_SVRECON/2/ply/voxel_model/iteration_2241/voxel_surface_mesh.ply \
//   --gt=results/tum_rgbd/rgbd_dataset_freiburg1_desk/nvblox/nvblox_color_mesh.ply \
//   --out=results/tum_rgbd/rgbd_dataset_freiburg1_desk/experiments_SVRECON/2/mesh_eval_nvblox \
//   --tau_cm=5.0 \
//   --recon_samples=500000 \
//   --gt_samples=500000

// ESLAM culled Replica
// ./bin/mesh_eval \
//   --eval_mode=gaussian_slam_sim3 \
//   --recon=/home/dimitris/Photo-SLAM/results/replica_rgbd_voxel/office0/<experiment>/ply/voxel_model/<iteration>/voxel_surface_mesh.ply \
//   --gt=/home/dimitris/Photo-SLAM/third_party/ESLAM/cull_replica_mesh/office0_culled.ply \
//   --out=/home/dimitris/Photo-SLAM/results/replica_rgbd_voxel/office0/<experiment>/mesh_eval_gs_table3 \
//   --tau_cm=1.0 \
//   --eval_depth_mesh=1 \
//   --align_recon_to_gt=1 \
//   --traj=/home/dimitris/Photo-SLAM/scripts/data/Replica/office0/traj.txt \
//   --traj_mode=c2w \
//   --recon_traj_tum=/home/dimitris/Photo-SLAM/results/replica_rgbd_voxel/office0/CameraTrajectory_TUM.txt \
//   --save_aligned_mesh=1

// Replica ESLAM culled mesh evaluation example with floaters
// ./bin/mesh_eval \
//   --eval_mode=gaussian_slam_sim3 \
//   --recon=results/replica_rgbd_voxel/office0/4141_shutdown/ply/voxel_model/iteration_4141/voxel_model.ply \
//   --gt=third_party/ESLAM/cull_replica_mesh/office0_culled.ply \
//   --out=results/replica_rgbd_voxel/office0/4141_shutdown/voxel_support_eval \
//   --tau_cm=5.0 \
//   --align_recon_to_gt=1 \
//   --traj=scripts/data/Replica/office0/traj.txt \
//   --traj_mode=c2w \
//   --recon_traj_tum=results/replica_rgbd_voxel/office0/CameraTrajectory_TUM.txt \
//   --eval_floaters=0 \
//   --gt_samples=500000 \
//   --floater_bin_cm=1.0 \
//   --floater_max_cm=50.0 \
//   --eval_gaussian_support=0 \
//   --eval_voxel_support=1 \
//   --support_samples_per_primitive=32

// Original Replica RGBD evaluation with floaters and Gaussian support
// ./bin/mesh_eval \
//   --eval_mode=gaussian_slam_sim3 \
//   --recon=results/replica_rgbd_original/office0/3381_shutdown/ply/point_cloud/iteration_3381/point_cloud.ply \
//   --gt=third_party/ESLAM/cull_replica_mesh/office0_culled.ply \
//   --out=results/replica_rgbd_original/office0/3381_shutdown/gaussian_support_eval_equal \
//   --tau_cm=5.0 \
//   --align_recon_to_gt=1 \
//   --traj=scripts/data/Replica/office0/traj.txt \
//   --traj_mode=c2w \
//   --recon_traj_tum=results/replica_rgbd_original/office0/CameraTrajectory_TUM.txt \
//   --eval_floaters=0 \
//   --gt_samples=500000 \
//   --floater_bin_cm=1.0 \
//   --floater_max_cm=50.0 \
//   --eval_gaussian_support=1 \
//   --gaussian_support_sigma=3.0 \
//   --gaussian_min_opacity=0.0 \
//   --eval_voxel_support=0 \
//   --support_samples_per_primitive=32
