#include "src_voxel/voxel_model_internal.h"

namespace sv {

namespace {

constexpr int kMaxNumLevels = 16;

std::array<int64_t, 3> decodeIjk(uint64_t path, int level)
{
    path >>= 3 * (kMaxNumLevels - level);
    int64_t i = 0;
    int64_t j = 0;
    int64_t k = 0;
    for (int l = 0; l < level; ++l) {
        const uint64_t bits = path & 0x7u;
        i |= static_cast<int64_t>((bits >> 2) & 0x1u) << l;
        j |= static_cast<int64_t>((bits >> 1) & 0x1u) << l;
        k |= static_cast<int64_t>(bits & 0x1u) << l;
        path >>= 3;
    }
    return {i, j, k};
}

torch::Tensor decodeCentersFromOctree(
    const torch::Tensor& octpath,
    const torch::Tensor& octlevel,
    const torch::Tensor& scene_center,
    const torch::Tensor& scene_extent)
{
    auto paths = octpath.contiguous().view({-1}).to(torch::kInt64);
    auto levels = octlevel.contiguous().view({-1}).to(torch::kInt32);
    const int64_t count = paths.size(0);
    const float extent = scene_extent.item<float>();
    const float min_x = scene_center[0].item<float>() - 0.5f * extent;
    const float min_y = scene_center[1].item<float>() - 0.5f * extent;
    const float min_z = scene_center[2].item<float>() - 0.5f * extent;

    auto centers = torch::empty(
        {count, 3}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    auto center_access = centers.accessor<float, 2>();
    const auto* path_data = paths.data_ptr<int64_t>();
    const auto* level_data = levels.data_ptr<int32_t>();
    for (int64_t index = 0; index < count; ++index) {
        const int level = level_data[index];
        const auto ijk = decodeIjk(static_cast<uint64_t>(path_data[index]), level);
        const float voxel_size = std::ldexp(extent, -level);
        center_access[index][0] = min_x + (static_cast<float>(ijk[0]) + 0.5f) * voxel_size;
        center_access[index][1] = min_y + (static_cast<float>(ijk[1]) + 0.5f) * voxel_size;
        center_access[index][2] = min_z + (static_cast<float>(ijk[2]) + 0.5f) * voxel_size;
    }
    return centers;
}

} // namespace

void VoxelModel::savePly(const std::filesystem::path& result_path)
{
    // Exports every active leaf and its shared-corner SDF/SH attributes; this is
    // a viewer-friendly format rather than SVRecon's checkpoint IO.
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
    if (is_leaf_.defined() && is_leaf_.size(0) == op_cpu.size(0)) {
        auto leaf_idx = torch::nonzero(
            is_leaf_.detach().to(torch::kCPU).to(torch::kBool).reshape({-1}))
                            .reshape({-1}).to(torch::kLong);
        op_cpu = op_cpu.index_select(0, leaf_idx).contiguous();
        lv_cpu = lv_cpu.index_select(0, leaf_idx).contiguous();
        sh0_cpu = sh0_cpu.index_select(0, leaf_idx).contiguous();
        shs_cpu = shs_cpu.index_select(0, leaf_idx).contiguous();
        voxkey = voxkey.index_select(0, leaf_idx).contiguous();
    }

    // Decode centers (viewer’s computation)
    torch::Tensor xyz = decodeCentersFromOctree(op_cpu, lv_cpu, sc_cpu, se_cpu); // [N,3]
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

    // Preserve every active SH band. Dropping degree 2/3 made the exported
    // viewer model look substantially worse than the model used for rendering.
    int64_t K = (shs_cpu.dim()>=2) ? shs_cpu.size(1) : 0;
    const int export_sh_degree = std::max(0, active_sh_degree_);
    const int64_t export_k =
        static_cast<int64_t>((export_sh_degree + 1) * (export_sh_degree + 1) - 1);
    if (K < export_k) {
        std::cerr << "[savePly] NOTE: shs_ has only " << K
                  << " coeffs per channel; padding active degree "
                  << export_sh_degree << " with zeros.\n";
    }
    torch::Tensor shs_active = torch::zeros(
        {N, export_k, 3}, torch::dtype(torch::kFloat32));
    if (export_k > 0 && K > 0 && shs_cpu.dim()==3 && shs_cpu.size(2)==3) {
        const auto copy_k = std::min<int64_t>(K, export_k);
        shs_active.index_put_({torch::indexing::Slice(),
                               torch::indexing::Slice(0,copy_k),
                               torch::indexing::Slice()},
                              shs_cpu.index({torch::indexing::Slice(),
                                             torch::indexing::Slice(0,copy_k),
                                             torch::indexing::Slice()}).to(torch::kFloat32));
    }
    auto frest = shs_active.view({N, export_k * 3}).contiguous();

    // Vox key sanity
    if (!(voxkey.dim()==2 && voxkey.size(0)==N && voxkey.size(1)==8)) {
        std::cerr << "[savePly] ERROR: vox_key_ must be [N,8]; got " << voxkey.sizes() << "\n";
        return;
    }
    if (geo_cpu.dim()!=1) geo_cpu = geo_cpu.view({-1});
    const int64_t G = geo_cpu.size(0);

    std::vector<int64_t> keep_idx(static_cast<size_t>(N));
    std::iota(keep_idx.begin(), keep_idx.end(), 0);
    const int64_t M = (int64_t)keep_idx.size();
    auto idx_t = torch::from_blob(keep_idx.data(), {M}, torch::TensorOptions().dtype(torch::kLong)).clone();

    // Slice tensors
    xyz   = xyz.index_select(0, idx_t).contiguous();
    fdc   = fdc.index_select(0, idx_t).contiguous();
    frest = frest.index_select(0, idx_t).contiguous();
    voxkey= voxkey.index_select(0, idx_t).contiguous();

    // tinyply has no uint64 scalar. Octree paths use up to 48 bits at level 16,
    // which FLOAT64 preserves exactly; UINT32 truncated every path above level 10.
    std::vector<double>   op_f64; op_f64.reserve(M);
    std::vector<uint8_t>  lv_u8;  lv_u8.reserve(M);
    {
        auto op64 = op_cpu.view({-1}).to(torch::kInt64);
        const int64_t* op_ptr = op64.data_ptr<int64_t>();
        for (auto i : keep_idx) op_f64.push_back(static_cast<double>(op_ptr[i]));
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

    // octpath (exact integer represented as float64)
    ply.add_properties_to_element(
        "vertex", {"octpath"},
        tinyply::Type::FLOAT64, M,
        reinterpret_cast<uint8_t*>(op_f64.data()),
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

    // f_rest_* (all active SH bands)
    {
        if (export_k > 0) {
            std::vector<std::string> names;
            names.reserve(static_cast<std::size_t>(export_k * 3));
            for (int64_t i=0; i<export_k * 3; ++i)
                names.emplace_back("f_rest_" + std::to_string(i));
            ply.add_properties_to_element(
                "vertex", names,
                tinyply::Type::FLOAT32, M,
                reinterpret_cast<uint8_t*>(frest.data_ptr<float>()),
                tinyply::Type::INVALID, 0);
        }
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
        ply.get_comments().push_back("active_sh_degree " + std::to_string(export_sh_degree));
    } catch (...) {}

    ply.write(out, /*binary*/ true);
    fb.close();

}

bool VoxelModel::refreshDenseCoreBBFromCurrentVoxels()
{
    // Re-estimates the robust dense-core box used by near/far pruning.
    // SVRecon reference: bounding_utils.main_scene_bound_pcd_heuristic.
    torch::Tensor pts_cpu;
    if (real_pcd_points_accum_cpu_.defined() &&
        real_pcd_points_accum_cpu_.numel() > 0) {
        pts_cpu = real_pcd_points_accum_cpu_;
    } else if (sparse_points_xyz_.defined() &&
               sparse_points_xyz_.numel() > 0) {
        pts_cpu = sparse_points_xyz_
                      .detach()
                      .to(torch::kCPU)
                      .to(torch::kFloat32)
                      .contiguous();
    }

    if (!pts_cpu.defined() || pts_cpu.numel() == 0 ||
        pts_cpu.dim() != 2 || pts_cpu.size(1) != 3) {
        return false;
    }

    auto bound = mainSceneBoundPcdHeuristicCpp(
        pts_cpu.contiguous(),
        dense_core_pcd_density_rate_);
    if (!bound) {
        return false;
    }

    const torch::Device dev =
        center_.defined() ? center_.device() :
        (scene_center_.defined() ? scene_center_.device() : torch::Device(torch::kCUDA));
    auto core_center =
        bound->first.to(dev).to(torch::kFloat32).contiguous().view({3});
    const float radius = bound->second;
    if (!std::isfinite(radius) || radius <= 0.0f) {
        return false;
    }

    auto core_radius =
        torch::full({3}, radius, torch::dtype(torch::kFloat32).device(dev));
    dense_core_bb_min_ = (core_center - core_radius).contiguous();
    dense_core_bb_max_ = (core_center + core_radius).contiguous();
    has_dense_core_bb_ = true;
    return true;
}

} // namespace sv
