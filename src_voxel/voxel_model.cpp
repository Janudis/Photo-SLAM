#include "src_voxel/voxel_model_internal.h"

namespace sv {

void VoxelModel::setTopologySdfInitializationMode(const std::string& mode)
{
    if (mode == "orb_prior") {
        topology_sdf_init_mode_ = SdfInitMode::OrbPriorOnly;
    } else if (mode == "weak_positive") {
        topology_sdf_init_mode_ = SdfInitMode::WeakPositive;
    } else if (mode == "source_points") {
        topology_sdf_init_mode_ = SdfInitMode::SignedPointPrior;
    } else if (mode == "weak_surface_prior") {
        topology_sdf_init_mode_ = SdfInitMode::WeakSurfacePrior;
    } else {
        throw std::invalid_argument(
            "VoxelModel topology SDF initialization mode must be one of: "
            "orb_prior, weak_positive, source_points, weak_surface_prior");
    }
}
VoxelModel::~VoxelModel() = default;

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
}


const torch::Tensor& sv::VoxelModel::geoGridPts() const { return _geo_grid_pts_; }
const torch::Tensor& sv::VoxelModel::sh0()        const { return sh0_; }
const torch::Tensor& sv::VoxelModel::shs()        const { return shs_; }

int64_t sv::VoxelModel::numGridPts() const {
    return (grid_pts_key_.defined() && grid_pts_key_.dim() > 0)
           ? grid_pts_key_.size(0)
           : 0;
}
// Converts shared integer corner keys to world coordinates.
// SVRecon reference: SVProperties.grid_pts_xyz.
torch::Tensor VoxelModel::gridPointsWorld() const
{
    TORCH_CHECK(grid_pts_key_.defined() &&
                    grid_pts_key_.dim() == 2 &&
                    grid_pts_key_.size(1) == 3,
                "gridPointsWorld: grid_pts_key_ must be [M,3]");
    TORCH_CHECK(scene_center_.defined() && scene_center_.numel() == 3,
                "gridPointsWorld: scene_center_ must be [3]");
    TORCH_CHECK(scene_extent_.defined() && scene_extent_.numel() == 1,
                "gridPointsWorld: scene_extent_ must be [1]");

    auto dev = grid_pts_key_.device();
    torch::Tensor scene_center =
        scene_center_.to(dev).to(torch::kFloat32).contiguous().view({3});
    torch::Tensor scene_extent =
        scene_extent_.to(dev).to(torch::kFloat32).contiguous().view({1});
    torch::Tensor scene_min = scene_center - 0.5f * scene_extent;

    const float finest_scale = std::ldexp(1.0f, -max_num_levels_);
    torch::Tensor finest_vox = scene_extent * finest_scale;
    return (scene_min.view({1, 3}) +
            grid_pts_key_.to(torch::kFloat32) * finest_vox.view({1, 1}))
        .contiguous();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
VoxelModel::buildSvreconDenseExtractionGrid(int inside_level) const
{
    // Builds the initial dense octree grid for progressive rendered-depth TSDF
    // extraction. SVRecon reference: extract_mesh.py::extract_mesh_progressive.
    TORCH_CHECK(scene_center_.defined() && scene_extent_.defined() && inside_extent_.defined(),
                "buildSvreconDenseExtractionGrid: scene bounds are not initialized");
    const int use_inside_level = std::clamp(
        inside_level,
        1,
        std::max(1, max_num_levels_ - outside_level_));
    auto dev = scene_center_.device();
    auto [octpath, octlevel] = genSvreconDenseOctpath(
        outside_level_,
        use_inside_level,
        max_num_levels_,
        dev);
    auto [grid_pts_key, vox_key] =
        buildGridPtsLink(octpath, octlevel, max_num_levels_);

    const auto scene_min = scene_center_.view({1, 3}) - 0.5f * scene_extent_.view({1, 1});
    const float finest_scale = std::ldexp(1.0f, -max_num_levels_);
    const auto finest_vox = scene_extent_.view({1, 1}) * finest_scale;
    auto grid_xyz =
        (scene_min + grid_pts_key.to(torch::kFloat32) * finest_vox).contiguous();

    // Match extract_mesh_progressive(): keep dense cells touching the inside box.
    const auto inside_min =
        scene_center_.view({1, 3}) - 0.5f * inside_extent_.view({1, 1});
    const auto inside_max =
        scene_center_.view({1, 3}) + 0.5f * inside_extent_.view({1, 1});
    const auto grid_inside = ((grid_xyz >= inside_min) & (grid_xyz <= inside_max)).all(1);
    const auto vox_inside = grid_inside.index({vox_key}).any(1);
    const auto inside_idx = torch::nonzero(vox_inside).view({-1}).to(torch::kLong);
    octpath = octpath.index_select(0, inside_idx).contiguous();
    octlevel = octlevel.index_select(0, inside_idx).contiguous();

    std::tie(grid_pts_key, vox_key) =
        buildGridPtsLink(octpath, octlevel, max_num_levels_);
    grid_xyz =
        (scene_min + grid_pts_key.to(torch::kFloat32) * finest_vox).contiguous();
    return {
        octpath,
        octlevel,
        grid_xyz,
        vox_key.contiguous()};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
VoxelModel::subdivideSvreconExtractionGrid(
    const torch::Tensor& octpath,
    const torch::Tensor& octlevel) const
{
    // Replaces each retained extraction cell with its eight children.
    // SVRecon reference: extract_mesh.py::extract_mesh_progressive.
    TORCH_CHECK(scene_center_.defined() && scene_extent_.defined(),
                "subdivideSvreconExtractionGrid: scene bounds are not initialized");
    TORCH_CHECK(octpath.defined() && octlevel.defined() &&
                    octpath.size(0) == octlevel.size(0),
                "subdivideSvreconExtractionGrid: invalid topology");
    auto [child_path, child_level] = genChildrenOctpath(
        octpath.to(scene_center_.device()).contiguous(),
        octlevel.to(scene_center_.device()).contiguous(),
        max_num_levels_);
    auto [grid_pts_key, vox_key] =
        buildGridPtsLink(child_path, child_level, max_num_levels_);
    const auto scene_min = scene_center_.view({1, 3}) - 0.5f * scene_extent_.view({1, 1});
    const float finest_scale = std::ldexp(1.0f, -max_num_levels_);
    const auto finest_vox = scene_extent_.view({1, 1}) * finest_scale;
    auto grid_xyz =
        (scene_min + grid_pts_key.to(torch::kFloat32) * finest_vox).contiguous();
    return {
        child_path.contiguous(),
        child_level.contiguous(),
        grid_xyz,
        vox_key.contiguous()};
}

std::tuple<torch::Tensor, torch::Tensor>
VoxelModel::querySdfTrilinear(const torch::Tensor& points_world) const
{
    // Locates the finest active leaf containing each query and interpolates its
    // eight shared SDF corners. This is an online supervision/evidence helper.
    auto dev = _geo_grid_pts_.defined() ? _geo_grid_pts_.device() : points_world.device();
    auto value_opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
    auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(dev);

    if (!points_world.defined() || points_world.numel() == 0) {
        return {
            torch::empty({0}, value_opts),
            torch::empty({0}, bool_opts)
        };
    }

    torch::Tensor pts = points_world.to(dev).to(torch::kFloat32).reshape({-1, 3}).contiguous();
    const int64_t P = pts.size(0);
    torch::Tensor sdf_out = torch::zeros({P}, value_opts);
    torch::Tensor valid_out = torch::zeros({P}, bool_opts);

    if (!_geo_grid_pts_.defined() || _geo_grid_pts_.numel() == 0 ||
        !vox_key_.defined() || vox_key_.dim() != 2 || vox_key_.size(1) != 8 ||
        !oct_path_.defined() || !oct_level_.defined() ||
        oct_path_.numel() == 0 || oct_level_.numel() == 0 ||
        !scene_center_.defined() || scene_center_.numel() != 3 ||
        !scene_extent_.defined() || scene_extent_.numel() != 1 ||
        max_num_levels_ <= 0 || max_num_levels_ > 20) {
        return {sdf_out, valid_out};
    }

    torch::Tensor scene_center =
        scene_center_.to(dev).to(torch::kFloat32).contiguous().view({3});
    torch::Tensor scene_extent =
        scene_extent_.to(dev).to(torch::kFloat32).contiguous().view({1});
    torch::Tensor scene_min = scene_center - 0.5f * scene_extent;
    const float scene_extent_f =
        scene_extent_.detach().to(torch::kCPU).to(torch::kFloat32).view({1}).item<float>();
    if (!std::isfinite(scene_extent_f) || scene_extent_f <= 0.0f) {
        return {sdf_out, valid_out};
    }

    torch::Tensor octpath = oct_path_.to(dev).to(torch::kLong).view({-1}).contiguous();
    torch::Tensor octlevel = oct_level_.to(dev).to(torch::kLong).view({-1}).contiguous();
    torch::Tensor vox_key = vox_key_.to(dev).to(torch::kLong).contiguous();
    if (is_leaf_.defined() && is_leaf_.size(0) == octpath.size(0)) {
        auto leaf_idx = torch::nonzero(is_leaf_.to(dev).to(torch::kBool).reshape({-1}))
                            .reshape({-1}).to(torch::kLong);
        octpath = octpath.index_select(0, leaf_idx);
        octlevel = octlevel.index_select(0, leaf_idx);
        vox_key = vox_key.index_select(0, leaf_idx);
    }
    if (octpath.size(0) != octlevel.size(0) || octpath.size(0) != vox_key.size(0)) {
        return {sdf_out, valid_out};
    }

    torch::Tensor active_keys = (octpath * 256 + octlevel).to(torch::kLong).contiguous();
    auto sorted = torch::sort(active_keys);
    torch::Tensor sorted_keys = std::get<0>(sorted).contiguous();
    torch::Tensor sorted_voxel_idx = std::get<1>(sorted).to(torch::kLong).contiguous();

    torch::Tensor levels_cpu = octlevel.detach().to(torch::kCPU).contiguous();
    std::vector<int64_t> levels(
        levels_cpu.data_ptr<int64_t>(),
        levels_cpu.data_ptr<int64_t>() + levels_cpu.numel());
    std::sort(levels.begin(), levels.end(), std::greater<int64_t>());
    levels.erase(std::unique(levels.begin(), levels.end()), levels.end());

    torch::Tensor geo = _geo_grid_pts_.to(dev).to(torch::kFloat32).view({-1});

    for (const int64_t level : levels) {
        if (level < 0 || level > max_num_levels_) {
            continue;
        }
        const int64_t grid_dim = 1LL << static_cast<int>(level);
        const float vox_size_f = std::ldexp(scene_extent_f, -static_cast<int>(level));
        if (!std::isfinite(vox_size_f) || vox_size_f <= 0.0f) {
            continue;
        }

        torch::Tensor vox_size =
            torch::full({1}, vox_size_f, value_opts);
        torch::Tensor coord =
            (pts - scene_min.view({1, 3})) / vox_size.view({1, 1});
        torch::Tensor ijk = torch::floor(coord).to(torch::kLong).contiguous();
        torch::Tensor in_bounds =
            (ijk >= 0).all(/*dim=*/1) &
            (ijk < grid_dim).all(/*dim=*/1) &
            torch::logical_not(valid_out);
        if (!in_bounds.any().item<bool>()) {
            continue;
        }

        torch::Tensor candidate_point_idx =
            torch::nonzero(in_bounds).view({-1}).to(torch::kLong).contiguous();
        torch::Tensor ijk_candidates =
            ijk.index_select(0, candidate_point_idx).contiguous();
        torch::Tensor level_candidates =
            torch::full(
                {ijk_candidates.size(0), 1},
                static_cast<int64_t>(level),
                torch::TensorOptions().dtype(torch::kInt8).device(dev));
        torch::Tensor candidate_octpath =
            SVRECON_UTILS::ijk_2_octpath(ijk_candidates, level_candidates)
                .to(torch::kLong)
                .view({-1})
                .contiguous();
        torch::Tensor candidate_keys =
            (candidate_octpath * 256 + level).to(torch::kLong).contiguous();

        torch::Tensor pos =
            at::searchsorted(
                sorted_keys,
                candidate_keys,
                /*out_int32=*/false,
                /*right=*/false)
                .to(torch::kLong)
                .contiguous();
        torch::Tensor in_range = pos < sorted_keys.size(0);
        torch::Tensor pos_clamped =
            pos.clamp(0, std::max<int64_t>(0, sorted_keys.size(0) - 1)).contiguous();
        torch::Tensor key_at_pos = sorted_keys.index_select(0, pos_clamped);
        torch::Tensor matched =
            (in_range & (key_at_pos == candidate_keys)).to(torch::kBool);
        if (!matched.any().item<bool>()) {
            continue;
        }

        torch::Tensor matched_candidate_idx =
            torch::nonzero(matched).view({-1}).to(torch::kLong).contiguous();
        torch::Tensor sample_idx =
            candidate_point_idx.index_select(0, matched_candidate_idx).contiguous();
        torch::Tensor voxel_idx =
            sorted_voxel_idx.index_select(
                0,
                pos.index_select(0, matched_candidate_idx).to(torch::kLong)).contiguous();

        torch::Tensor ijk_matched =
            ijk_candidates.index_select(0, matched_candidate_idx).to(torch::kFloat32);
        torch::Tensor pts_matched = pts.index_select(0, sample_idx);
        torch::Tensor local_min =
            scene_min.view({1, 3}) + ijk_matched * vox_size.view({1, 1});
        torch::Tensor frac =
            ((pts_matched - local_min) / vox_size.view({1, 1})).clamp(0.0f, 1.0f);
        torch::Tensor fx = frac.index({torch::indexing::Slice(), 0});
        torch::Tensor fy = frac.index({torch::indexing::Slice(), 1});
        torch::Tensor fz = frac.index({torch::indexing::Slice(), 2});
        torch::Tensor one_x = 1.0f - fx;
        torch::Tensor one_y = 1.0f - fy;
        torch::Tensor one_z = 1.0f - fz;
        torch::Tensor weights = torch::stack(
            {
                one_x * one_y * one_z,
                one_x * one_y * fz,
                one_x * fy * one_z,
                one_x * fy * fz,
                fx * one_y * one_z,
                fx * one_y * fz,
                fx * fy * one_z,
                fx * fy * fz
            },
            /*dim=*/1);

        torch::Tensor corner_idx =
            vox_key.index_select(0, voxel_idx).reshape({-1}).contiguous();
        torch::Tensor corner_sdf =
            geo.index_select(0, corner_idx)
                .view({voxel_idx.size(0), 8})
                .contiguous();
        torch::Tensor sdf_interp =
            (corner_sdf * weights).sum(/*dim=*/1).contiguous();

        sdf_out.index_put_({sample_idx}, sdf_interp);
        valid_out.index_put_(
            {sample_idx},
            torch::ones({sample_idx.size(0)}, bool_opts));
    }

    return {sdf_out, valid_out};
}

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

torch::Tensor VoxelModel::activeRenderableMask() const
{
    // Every cell stored by VoxelModel is active; evidence-only cells live in
    // VoxelMapper and are never inserted into this topology.
    const int64_t N = center_.defined() ? center_.size(0) : 0;
    auto opts = torch::TensorOptions().dtype(torch::kBool).device(device_type_);
    if (N <= 0) {
        return torch::empty({0}, opts);
    }
    return torch::ones({N}, opts);
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

float VoxelModel::insertionVoxSize() const
{
    if (vox_eff_.defined() && vox_eff_.numel() > 0) {
        return vox_eff_.detach().reshape({-1})[0].item<float>();
    }
    return fixed_vox_size_;
}

torch::Tensor VoxelModel::InsideExtent() const {
    return this->inside_extent_;
}

void VoxelModel::oneUpShDegree()
{
    // '''
    // Number of spherical-harmonic coefficients per color channel.
    // '''
    if (this->active_sh_degree_ < this->max_sh_degree_)
        this->active_sh_degree_ += 1;
}

void VoxelModel::setShDegree(const int sh)
{
    this->active_sh_degree_ = (sh > this->max_sh_degree_ ? this->max_sh_degree_ : sh);
}

const torch::Tensor& sv::VoxelModel::fusedSdfGridPts() const { return fused_sdf_grid_pts_; }
const torch::Tensor& sv::VoxelModel::fusedSdfWeights() const { return fused_sdf_weights_; }
bool VoxelModel::hasFusedSdfField() const
{
    return fused_sdf_grid_pts_.defined() &&
           fused_sdf_weights_.defined() &&
           fused_sdf_grid_pts_.dim() == 2 &&
           fused_sdf_weights_.dim() == 2 &&
           fused_sdf_grid_pts_.size(1) == 1 &&
           fused_sdf_weights_.size(1) == 1 &&
           grid_pts_key_.defined() &&
           fused_sdf_grid_pts_.size(0) == grid_pts_key_.size(0) &&
           fused_sdf_weights_.size(0) == grid_pts_key_.size(0);
}

void VoxelModel::setEmptyFusedSdfField_()
{
    auto dev = _geo_grid_pts_.defined()
        ? _geo_grid_pts_.device()
        : torch::Device(device_type_);
    auto value_opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
    fused_sdf_grid_pts_ = torch::empty({0, 1}, value_opts);
    fused_sdf_weights_ = torch::empty({0, 1}, value_opts);
}

torch::Tensor VoxelModel::voxelCornerScalarFromGrid_(const torch::Tensor& grid_scalar) const
{
    const int64_t N = (vox_key_.defined() && vox_key_.dim() == 2) ? vox_key_.size(0) : 0;
    auto dev = (grid_scalar.defined() ? grid_scalar.device() :
                (_geo_grid_pts_.defined() ? _geo_grid_pts_.device() : torch::Device(device_type_)));
    auto value_opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
    if (N == 0 ||
        !grid_scalar.defined() ||
        grid_scalar.numel() == 0 ||
        !vox_key_.defined() ||
        vox_key_.dim() != 2 ||
        vox_key_.size(1) != 8) {
        return torch::zeros({N, 8, 1}, value_opts);
    }

    auto scalar = grid_scalar.to(dev).to(torch::kFloat32).reshape({-1, 1}).contiguous();
    return scalar.index({vox_key_.to(dev).to(torch::kLong)}).contiguous();
}

torch::Tensor VoxelModel::voxelGeoCorners() const
{
    torch::Tensor corners = voxelCornerScalarFromGrid_(_geo_grid_pts_);
    if (corners.defined() && corners.dim() == 3 && corners.size(2) == 1) {
        corners = corners.squeeze(2);
    }
    return corners.contiguous().detach();
}

torch::Tensor VoxelModel::voxelSdfWeightCorners() const
{
    torch::Tensor corners = voxelCornerScalarFromGrid_(fused_sdf_weights_);
    if (corners.defined() && corners.dim() == 3 && corners.size(2) == 1) {
        corners = corners.squeeze(2);
    }
    return corners.contiguous().detach();
}

void VoxelModel::rebuildFusedSdfFieldFromVoxelCorners_(
    const torch::Tensor& voxel_sdf_values,
    const torch::Tensor& voxel_sdf_weights)
{
    torch::NoGradGuard no_grad;
    if (!grid_pts_key_.defined() ||
        grid_pts_key_.dim() != 2 ||
        grid_pts_key_.size(1) != 3 ||
        !vox_key_.defined() ||
        vox_key_.dim() != 2 ||
        vox_key_.size(1) != 8) {
        setEmptyFusedSdfField_();
        return;
    }

    const int64_t M = grid_pts_key_.size(0);
    const int64_t N = vox_key_.size(0);
    auto dev = _geo_grid_pts_.defined() ? _geo_grid_pts_.device() : grid_pts_key_.device();
    auto value_opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
    if (M == 0) {
        fused_sdf_grid_pts_ = torch::empty({0, 1}, value_opts);
        fused_sdf_weights_ = torch::empty({0, 1}, value_opts);
        return;
    }
    if (N == 0 ||
        !voxel_sdf_values.defined() ||
        !voxel_sdf_weights.defined() ||
        voxel_sdf_values.size(0) != N ||
        voxel_sdf_values.size(1) != 8 ||
        voxel_sdf_weights.size(0) != N ||
        voxel_sdf_weights.size(1) != 8) {
        fused_sdf_grid_pts_ = torch::zeros({M, 1}, value_opts);
        fused_sdf_weights_ = torch::zeros({M, 1}, value_opts);
        return;
    }

    auto sdf_vox = voxel_sdf_values.to(dev).to(torch::kFloat32);
    auto w_vox = voxel_sdf_weights.to(dev).to(torch::kFloat32);
    if (sdf_vox.dim() == 2) {
        sdf_vox = sdf_vox.unsqueeze(-1);
    }
    if (w_vox.dim() == 2) {
        w_vox = w_vox.unsqueeze(-1);
    }

    fused_sdf_grid_pts_ =
        aggregateVoxelCornersIntoGridPts(M, vox_key_.to(dev), sdf_vox).detach().contiguous();
    fused_sdf_weights_ =
        aggregateVoxelCornersIntoGridPts(M, vox_key_.to(dev), w_vox).detach().contiguous();
}

void VoxelModel::ensureFusedSdfField()
{
    if (!grid_pts_key_.defined() || grid_pts_key_.dim() != 2 || grid_pts_key_.size(1) != 3) {
        setEmptyFusedSdfField_();
        return;
    }

    const int64_t M = grid_pts_key_.size(0);
    auto tensor_dev = _geo_grid_pts_.defined() ? _geo_grid_pts_.device() : grid_pts_key_.device();
    auto value_opts = torch::TensorOptions().dtype(torch::kFloat32).device(tensor_dev);

    auto make_empty_values = [&]() {
        fused_sdf_grid_pts_ = torch::zeros({M, 1}, value_opts);
        fused_sdf_weights_ = torch::zeros({M, 1}, value_opts);
    };

    if (M == 0) {
        make_empty_values();
        return;
    }

    if (fused_sdf_grid_pts_.defined() &&
        fused_sdf_weights_.defined() &&
        fused_sdf_grid_pts_.size(0) == M &&
        fused_sdf_weights_.size(0) == M) {
        if (fused_sdf_grid_pts_.device() != tensor_dev) {
            fused_sdf_grid_pts_ = fused_sdf_grid_pts_.to(tensor_dev).contiguous();
            fused_sdf_weights_ = fused_sdf_weights_.to(tensor_dev).contiguous();
        }
        return;
    }

    make_empty_values();
}

void VoxelModel::resetFusedSdfField()
{
    if (!grid_pts_key_.defined() || grid_pts_key_.dim() != 2 || grid_pts_key_.size(1) != 3) {
        setEmptyFusedSdfField_();
        return;
    }

    const int64_t M = grid_pts_key_.size(0);
    auto tensor_dev = _geo_grid_pts_.defined() ? _geo_grid_pts_.device() : grid_pts_key_.device();
    auto value_opts = torch::TensorOptions().dtype(torch::kFloat32).device(tensor_dev);
    fused_sdf_grid_pts_ = torch::zeros({M, 1}, value_opts);
    fused_sdf_weights_ = torch::zeros({M, 1}, value_opts);
}

void VoxelModel::fuseProjectiveSdfGridSamples(
    const torch::Tensor& tsdf_values,
    const torch::Tensor& weights,
    const torch::Tensor& valid_mask,
    float max_weight)
{
    // Maintains a non-learnable weighted projective SDF field aligned with the
    // shared corner table; it never replaces the optimizer's parameter tensor.
    ensureFusedSdfField();
    if (!hasFusedSdfField() || grid_pts_key_.size(0) == 0) {
        return;
    }

    const int64_t M = grid_pts_key_.size(0);
    torch::NoGradGuard no_grad;
    auto dev = fused_sdf_grid_pts_.device();
    torch::Tensor sample_tsdf = tsdf_values.to(dev).to(torch::kFloat32).reshape({M, 1});
    torch::Tensor sample_w = weights.to(dev).to(torch::kFloat32).reshape({M, 1});
    sample_w = torch::where(
        torch::isfinite(sample_w),
        sample_w,
        torch::zeros_like(sample_w));
    torch::Tensor valid = valid_mask.to(dev).to(torch::kBool).reshape({M, 1}) &
                          torch::isfinite(sample_tsdf) &
                          (sample_w > 0.0f);

    torch::Tensor old_w = fused_sdf_weights_;
    torch::Tensor old_sdf = fused_sdf_grid_pts_;
    torch::Tensor fused_w = old_w + sample_w;
    torch::Tensor fused_sdf =
        (old_sdf * old_w + sample_tsdf * sample_w) / fused_w.clamp_min(1.0e-6f);
    if (max_weight > 0.0f) {
        fused_w = fused_w.clamp_max(max_weight);
    }

    fused_sdf_grid_pts_ = torch::where(valid, fused_sdf, old_sdf).contiguous();
    fused_sdf_weights_ = torch::where(valid, fused_w, old_w).contiguous();
}

} // namespace sv
