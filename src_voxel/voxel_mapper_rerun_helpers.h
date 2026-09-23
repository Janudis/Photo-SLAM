#pragma once

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

static bool saveTriangleMeshPly(
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

static void capTriangleMeshForExport(
    TriangleMeshRgb& mesh,
    std::size_t max_vertices,
    std::size_t max_faces,
    float quantization_eps)
{
    if (mesh.vertices.empty() || mesh.faces.empty()) return;
    if ((max_vertices == 0 || mesh.vertices.size() <= max_vertices) &&
        (max_faces == 0 || mesh.faces.size() <= max_faces)) {
        return;
    }

    struct QuantizedVertexKey
    {
        std::int64_t x = 0;
        std::int64_t y = 0;
        std::int64_t z = 0;

        bool operator==(const QuantizedVertexKey& other) const noexcept
        {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct QuantizedVertexKeyHash
    {
        std::size_t operator()(const QuantizedVertexKey& key) const noexcept
        {
            const std::uint64_t x = static_cast<std::uint64_t>(key.x) * 73856093ull;
            const std::uint64_t y = static_cast<std::uint64_t>(key.y) * 19349663ull;
            const std::uint64_t z = static_cast<std::uint64_t>(key.z) * 83492791ull;
            return static_cast<std::size_t>(x ^ y ^ z);
        }
    };

    const float eps = std::max(quantization_eps, 1.0e-8f);
    auto make_key = [eps](const Eigen::Vector3f& p) -> QuantizedVertexKey
    {
        return QuantizedVertexKey{
            static_cast<std::int64_t>(std::llround(static_cast<double>(p.x()) / eps)),
            static_cast<std::int64_t>(std::llround(static_cast<double>(p.y()) / eps)),
            static_cast<std::int64_t>(std::llround(static_cast<double>(p.z()) / eps))};
    };

    const bool has_colors = mesh.colors.size() == mesh.vertices.size();
    TriangleMeshRgb capped;
    capped.vertices.reserve(max_vertices > 0 ? std::min(max_vertices, mesh.vertices.size()) : mesh.vertices.size());
    capped.faces.reserve(max_faces > 0 ? std::min(max_faces, mesh.faces.size()) : mesh.faces.size());
    if (has_colors) {
        capped.colors.reserve(capped.vertices.capacity());
    }

    std::unordered_map<QuantizedVertexKey, uint32_t, QuantizedVertexKeyHash> vertex_map;
    vertex_map.reserve(capped.vertices.capacity());

    const std::size_t face_limit = max_faces > 0 ? max_faces : mesh.faces.size();
    for (const auto& face : mesh.faces)
    {
        if (capped.faces.size() >= face_limit) break;

        std::array<QuantizedVertexKey, 3> keys;
        std::array<uint32_t, 3> remapped{};
        bool valid_face = true;
        std::size_t new_vertices = 0;

        for (int i = 0; i < 3; ++i)
        {
            if (face[i] >= mesh.vertices.size()) {
                valid_face = false;
                break;
            }
            keys[i] = make_key(mesh.vertices[face[i]]);
            const auto it = vertex_map.find(keys[i]);
            if (it != vertex_map.end()) {
                remapped[i] = it->second;
                continue;
            }

            bool duplicate_new_key = false;
            for (int j = 0; j < i; ++j)
            {
                if (keys[i] == keys[j] && vertex_map.find(keys[i]) == vertex_map.end()) {
                    duplicate_new_key = true;
                    break;
                }
            }
            if (!duplicate_new_key) ++new_vertices;
        }
        if (!valid_face) continue;
        if (max_vertices > 0 && capped.vertices.size() + new_vertices > max_vertices) {
            continue;
        }

        for (int i = 0; i < 3; ++i)
        {
            const auto it = vertex_map.find(keys[i]);
            if (it != vertex_map.end()) {
                remapped[i] = it->second;
                continue;
            }

            const uint32_t new_index = static_cast<uint32_t>(capped.vertices.size());
            vertex_map.emplace(keys[i], new_index);
            remapped[i] = new_index;
            capped.vertices.push_back(mesh.vertices[face[i]]);
            if (has_colors) {
                capped.colors.push_back(mesh.colors[face[i]]);
            }
        }

        if (remapped[0] == remapped[1] ||
            remapped[1] == remapped[2] ||
            remapped[0] == remapped[2]) {
            continue;
        }
        capped.faces.push_back({remapped[0], remapped[1], remapped[2]});
    }

    if (!capped.vertices.empty() && !capped.faces.empty()) {
        mesh = std::move(capped);
    }
}

static void weldTriangleMeshVertices(TriangleMeshRgb& mesh, float quantization_eps = 1.0e-6f)
{
    if (mesh.vertices.empty() || mesh.faces.empty()) return;
    if (quantization_eps <= 0.0f) quantization_eps = 1.0e-6f;

    struct QuantizedVertexKey
    {
        std::int64_t x = 0;
        std::int64_t y = 0;
        std::int64_t z = 0;
        bool operator==(const QuantizedVertexKey& other) const noexcept
        {
            return x == other.x && y == other.y && z == other.z;
        }
    };
    struct QuantizedVertexHash
    {
        std::size_t operator()(const QuantizedVertexKey& key) const noexcept
        {
            std::size_t h = static_cast<std::size_t>(key.x);
            h ^= static_cast<std::size_t>(key.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= static_cast<std::size_t>(key.z) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };
    auto make_key = [&](const Eigen::Vector3f& v)
    {
        return QuantizedVertexKey{
            static_cast<std::int64_t>(std::llround(v.x() / quantization_eps)),
            static_cast<std::int64_t>(std::llround(v.y() / quantization_eps)),
            static_cast<std::int64_t>(std::llround(v.z() / quantization_eps))};
    };

    std::unordered_map<QuantizedVertexKey, uint32_t, QuantizedVertexHash> remap;
    remap.reserve(mesh.vertices.size());
    TriangleMeshRgb welded;
    welded.vertices.reserve(mesh.vertices.size());
    welded.colors.reserve(mesh.colors.size());
    welded.faces.reserve(mesh.faces.size());

    auto get_or_add = [&](uint32_t old_idx) -> uint32_t
    {
        if (old_idx >= mesh.vertices.size()) return 0;
        const QuantizedVertexKey key = make_key(mesh.vertices[old_idx]);
        auto it = remap.find(key);
        if (it != remap.end()) return it->second;
        const uint32_t new_idx = static_cast<uint32_t>(welded.vertices.size());
        remap.emplace(key, new_idx);
        welded.vertices.push_back(mesh.vertices[old_idx]);
        if (mesh.colors.size() == mesh.vertices.size()) {
            welded.colors.push_back(mesh.colors[old_idx]);
        }
        return new_idx;
    };

    for (const auto& face : mesh.faces)
    {
        const uint32_t a = get_or_add(face[0]);
        const uint32_t b = get_or_add(face[1]);
        const uint32_t c = get_or_add(face[2]);
        if (a == b || b == c || a == c) continue;
        welded.faces.push_back({a, b, c});
    }

    mesh = std::move(welded);
}

static void triangleMeshToTensors(
    const TriangleMeshRgb& mesh,
    torch::Tensor& vertices,
    torch::Tensor& colors,
    torch::Tensor& triangles)
{
    vertices = torch::empty(
        {static_cast<int64_t>(mesh.vertices.size()), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto v_acc = vertices.accessor<float, 2>();
    for (int64_t i = 0; i < vertices.size(0); ++i)
    {
        const auto& v = mesh.vertices[static_cast<std::size_t>(i)];
        v_acc[i][0] = v.x();
        v_acc[i][1] = v.y();
        v_acc[i][2] = v.z();
    }

    if (mesh.colors.size() == mesh.vertices.size())
    {
        colors = torch::empty(
            {static_cast<int64_t>(mesh.colors.size()), 3},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
        auto c_acc = colors.accessor<uint8_t, 2>();
        for (int64_t i = 0; i < colors.size(0); ++i)
        {
            const auto& c = mesh.colors[static_cast<std::size_t>(i)];
            c_acc[i][0] = c[0];
            c_acc[i][1] = c[1];
            c_acc[i][2] = c[2];
        }
    }
    else
    {
        colors = torch::Tensor();
    }

    triangles = torch::empty(
        {static_cast<int64_t>(mesh.faces.size()), 3},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto f_acc = triangles.accessor<int64_t, 2>();
    for (int64_t i = 0; i < triangles.size(0); ++i)
    {
        const auto& f = mesh.faces[static_cast<std::size_t>(i)];
        f_acc[i][0] = static_cast<int64_t>(f[0]);
        f_acc[i][1] = static_cast<int64_t>(f[1]);
        f_acc[i][2] = static_cast<int64_t>(f[2]);
    }
}

} // namespace
