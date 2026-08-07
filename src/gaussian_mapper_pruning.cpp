#include "include/gaussian_mapper.h"

#include <cmath>

namespace
{
constexpr float kNearDistance = 0.2f;
constexpr float kGaussianSupportSigma = 3.0f;
constexpr float kDenseCoreDensityRate = 0.005f;
}

torch::Tensor GaussianMapper::cameraFacingNearGaussianMask(
    const torch::Tensor& points,
    const torch::Tensor& support_radii) const
{
    TORCH_CHECK(points.defined() && points.dim() == 2 && points.size(1) == 3,
                "cameraFacingNearGaussianMask expects points [N,3]");
    TORCH_CHECK(support_radii.defined() &&
                    support_radii.numel() == points.size(0),
                "cameraFacingNearGaussianMask expects one support radius per point");

    const int64_t point_count = points.size(0);
    auto mask = torch::zeros(
        {point_count},
        torch::TensorOptions().dtype(torch::kBool).device(points.device()));
    if (point_count == 0 || !scene_) {
        return mask;
    }

    const auto points_f32 =
        points.detach().to(torch::kFloat32).contiguous();
    const auto radii =
        support_radii.detach()
            .to(points.device())
            .to(torch::kFloat32)
            .reshape({point_count})
            .clamp_min(0.0f)
            .contiguous();

    using torch::indexing::Slice;
    for (const auto& [kfid, keyframe] : scene_->keyframes()) {
        (void)kfid;
        if (!keyframe || keyframe->intr_.size() < 4 ||
            keyframe->image_width_ <= 0 || keyframe->image_height_ <= 0) {
            continue;
        }

        const auto Tcw = tensor_utils::EigenMatrix2TorchTensor(
            keyframe->getPosef().matrix(),
            points.device().type());
        const auto Rcw = Tcw.index({Slice(0, 3), Slice(0, 3)});
        const auto tcw = Tcw.index({Slice(0, 3), 3}).view({1, 3});
        const auto camera_points =
            points_f32.matmul(Rcw.transpose(0, 1)) + tcw;

        const auto x = camera_points.index({Slice(), 0});
        const auto y = camera_points.index({Slice(), 1});
        const auto z = camera_points.index({Slice(), 2});
        const auto safe_z = z.clamp_min(1.0e-6f);

        const float fx = keyframe->intr_[0];
        const float fy = keyframe->intr_[1];
        const float cx = keyframe->intr_[2];
        const float cy = keyframe->intr_[3];
        const auto u = fx * x / safe_z + cx;
        const auto v = fy * y / safe_z + cy;
        const auto radius_x = fx * radii / safe_z;
        const auto radius_y = fy * radii / safe_z;

        const auto overlaps_image =
            ((z + radii) > 0.0f) &
            ((u + radius_x) >= 0.0f) &
            ((u - radius_x) < static_cast<float>(keyframe->image_width_)) &
            ((v + radius_y) >= 0.0f) &
            ((v - radius_y) < static_cast<float>(keyframe->image_height_));
        mask |= overlaps_image & (z <= (kNearDistance + radii));
    }
    return mask.contiguous();
}

torch::Tensor GaussianMapper::geometricNearGaussianMask(
    const torch::Tensor& points,
    const torch::Tensor& support_radii) const
{
    TORCH_CHECK(points.defined() && points.dim() == 2 && points.size(1) == 3,
                "geometricNearGaussianMask expects points [N,3]");
    TORCH_CHECK(support_radii.defined() &&
                    support_radii.numel() == points.size(0),
                "geometricNearGaussianMask expects one support radius per point");

    const int64_t point_count = points.size(0);
    auto mask = torch::zeros(
        {point_count},
        torch::TensorOptions().dtype(torch::kBool).device(points.device()));
    if (point_count == 0 || !scene_) {
        return mask;
    }

    const auto points_f32 =
        points.detach().to(torch::kFloat32).contiguous();
    const auto near_radius =
        (support_radii.detach()
             .to(points.device())
             .to(torch::kFloat32)
             .reshape({point_count})
             .clamp_min(0.0f) +
         kNearDistance)
            .contiguous();
    const auto near_radius_sq = near_radius.square();

    for (const auto& [kfid, keyframe] : scene_->keyframes()) {
        (void)kfid;
        if (!keyframe) {
            continue;
        }
        const Eigen::Vector3f camera_position =
            keyframe->getPosef().inverse().translation();
        std::vector<float> camera_position_values = {
            camera_position.x(),
            camera_position.y(),
            camera_position.z()};
        const auto camera_position_tensor =
            torch::from_blob(
                camera_position_values.data(),
                {1, 3},
                torch::TensorOptions().dtype(torch::kFloat32))
                .clone()
                .to(points.device());
        const auto distance_sq =
            (points_f32 - camera_position_tensor).square().sum(/*dim=*/1);
        mask |= distance_sq <= near_radius_sq;
    }
    return mask.contiguous();
}

std::map<point3D_id_t, Point3D> GaussianMapper::filterNearInsertionPoints(
    const std::map<point3D_id_t, Point3D>& points) const
{
    if (!filter_near_voxels_ || points.empty() ||
        !scene_ || scene_->keyframes().empty()) {
        return points;
    }

    std::vector<float> xyz;
    std::vector<point3D_id_t> ids;
    xyz.reserve(points.size() * 3);
    ids.reserve(points.size());
    for (const auto& [id, point] : points) {
        ids.push_back(id);
        xyz.push_back(static_cast<float>(point.xyz_(0)));
        xyz.push_back(static_cast<float>(point.xyz_(1)));
        xyz.push_back(static_cast<float>(point.xyz_(2)));
    }

    auto point_tensor =
        torch::from_blob(
            xyz.data(),
            {static_cast<int64_t>(ids.size()), 3},
            torch::TensorOptions().dtype(torch::kFloat32))
            .clone()
            .to(device_type_);
    const auto support_radii =
        torch::zeros({point_tensor.size(0)}, point_tensor.options());
    const auto keep =
        (~cameraFacingNearGaussianMask(point_tensor, support_radii))
            .to(torch::kCPU)
            .contiguous();
    const auto keep_access = keep.accessor<bool, 1>();

    std::map<point3D_id_t, Point3D> filtered;
    for (int64_t index = 0; index < static_cast<int64_t>(ids.size()); ++index) {
        if (keep_access[index]) {
            filtered.emplace(ids[index], points.at(ids[index]));
        }
    }
    return filtered;
}

void GaussianMapper::filterNearInsertionPoints(
    std::vector<float>& points,
    std::vector<float>& colors) const
{
    TORCH_CHECK(points.size() == colors.size() && points.size() % 3 == 0,
                "Gaussian insertion points and colors must be aligned xyz/rgb arrays");
    if (!filter_near_voxels_ || points.empty() ||
        !scene_ || scene_->keyframes().empty()) {
        return;
    }

    const int64_t point_count = static_cast<int64_t>(points.size() / 3);
    auto point_tensor =
        torch::from_blob(
            points.data(),
            {point_count, 3},
            torch::TensorOptions().dtype(torch::kFloat32))
            .clone()
            .to(device_type_);
    auto color_tensor =
        torch::from_blob(
            colors.data(),
            {point_count, 3},
            torch::TensorOptions().dtype(torch::kFloat32))
            .clone()
            .to(device_type_);
    filterNearInsertionPoints(point_tensor, color_tensor);

    const auto points_cpu =
        point_tensor.to(torch::kCPU).to(torch::kFloat32).contiguous();
    const auto colors_cpu =
        color_tensor.to(torch::kCPU).to(torch::kFloat32).contiguous();
    points.assign(
        points_cpu.data_ptr<float>(),
        points_cpu.data_ptr<float>() + points_cpu.numel());
    colors.assign(
        colors_cpu.data_ptr<float>(),
        colors_cpu.data_ptr<float>() + colors_cpu.numel());
}

void GaussianMapper::filterNearInsertionPoints(
    torch::Tensor& points,
    torch::Tensor& colors) const
{
    TORCH_CHECK(points.defined() && points.dim() == 2 && points.size(1) == 3,
                "Gaussian insertion points must be [N,3]");
    TORCH_CHECK(colors.defined() && colors.dim() == 2 &&
                    colors.size(0) == points.size(0) && colors.size(1) == 3,
                "Gaussian insertion colors must be [N,3]");
    if (!filter_near_voxels_ || points.size(0) == 0 ||
        !scene_ || scene_->keyframes().empty()) {
        return;
    }

    const auto support_radii =
        torch::zeros({points.size(0)}, points.options().dtype(torch::kFloat32));
    const auto keep =
        ~cameraFacingNearGaussianMask(points, support_radii);
    const auto keep_indices =
        torch::nonzero(keep).view({-1}).to(torch::kLong);
    points = points.index_select(0, keep_indices).contiguous();
    colors = colors.index_select(
        0,
        keep_indices.to(colors.device())).contiguous();
}

torch::Tensor GaussianMapper::denseCoreFarGaussianMask() const
{
    const auto active_points = gaussians_->getXYZ().detach();
    auto empty_mask = torch::zeros(
        {active_points.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(active_points.device()));
    if (active_points.size(0) == 0 ||
        !gaussians_->sparse_points_xyz_.defined() ||
        gaussians_->sparse_points_xyz_.numel() == 0) {
        return empty_mask;
    }

    auto support =
        gaussians_->sparse_points_xyz_
            .detach()
            .to(active_points.device())
            .to(torch::kFloat32)
            .reshape({-1, 3})
            .contiguous();
    const auto finite_support = torch::isfinite(support).all(/*dim=*/1);
    support = support.index({finite_support}).contiguous();
    if (support.size(0) < 2) {
        return empty_mask;
    }

    const auto center =
        std::get<0>(support.median(/*dim=*/0, /*keepdim=*/false))
            .contiguous();
    auto distance =
        std::get<0>(
            (support - center.view({1, 3}))
                .abs()
                .max(/*dim=*/1, /*keepdim=*/false))
            .to(torch::kFloat32)
            .contiguous();
    distance =
        std::get<0>(distance.sort(/*dim=*/0, /*descending=*/false))
            .contiguous();

    const int64_t support_count = distance.size(0);
    auto rank = torch::arange(
        1,
        support_count + 1,
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(distance.device()));
    const auto density =
        rank * (distance > 0.0f).to(torch::kFloat32) /
        ((2.0f * distance).pow(3) + 1.0e-6f);

    int64_t begin_index = static_cast<int64_t>(
        std::llround(static_cast<double>(support_count) * 0.05));
    begin_index =
        std::max<int64_t>(0, std::min<int64_t>(support_count - 1, begin_index));
    const auto density_tail =
        density.index({
            torch::indexing::Slice(
                begin_index,
                torch::indexing::None)});
    if (density_tail.numel() == 0) {
        return empty_mask;
    }

    const int64_t peak_index =
        begin_index + density_tail.argmax().item<int64_t>();
    const float target_density =
        kDenseCoreDensityRate * density.index({peak_index}).item<float>();
    const auto right_density =
        density.index({
            torch::indexing::Slice(
                peak_index,
                torch::indexing::None)});
    const auto below_target =
        torch::nonzero(right_density < target_density)
            .view({-1})
            .contiguous();
    if (below_target.numel() == 0) {
        return empty_mask;
    }

    const int64_t radius_index =
        peak_index + below_target.index({0}).item<int64_t>();
    const float radius = distance.index({radius_index}).item<float>();
    if (!std::isfinite(radius) || radius <= 0.0f) {
        return empty_mask;
    }

    const auto bb_min = (center - radius).view({1, 3});
    const auto bb_max = (center + radius).view({1, 3});
    const auto finite_active = torch::isfinite(active_points).all(/*dim=*/1);
    const auto in_dense_core =
        finite_active &
        (active_points >= bb_min).all(/*dim=*/1) &
        (active_points <= bb_max).all(/*dim=*/1);
    return (~in_dense_core).to(torch::kBool).contiguous();
}

void GaussianMapper::runGaussianHeuristicPrune(bool final_cleanup)
{
    if ((!final_cleanup &&
         !filter_near_voxels_ &&
         !prune_far_voxels_ &&
         !prune_near_voxels_geometric_) ||
        (final_cleanup && !final_special_prune_enable_)) {
        return;
    }
    if (!gaussians_ || !gaussians_->optimizer_ ||
        gaussians_->getXYZ().size(0) == 0) {
        return;
    }

    const auto points = gaussians_->getXYZ().detach();
    const int64_t point_count = points.size(0);
    const auto support_radii =
        kGaussianSupportSigma *
        std::get<0>(
            gaussians_->getScalingActivation()
                .detach()
                .max(/*dim=*/1, /*keepdim=*/false));
    auto prune_mask = torch::zeros(
        {point_count},
        torch::TensorOptions().dtype(torch::kBool).device(points.device()));
    int64_t camera_near_count = 0;
    int64_t geometric_near_count = 0;
    int64_t far_count = 0;

    if (final_cleanup || filter_near_voxels_) {
        const auto camera_near =
            cameraFacingNearGaussianMask(points, support_radii);
        camera_near_count = camera_near.sum().item<int64_t>();
        prune_mask |= camera_near;
    }
    if (prune_near_voxels_geometric_) {
        const auto geometric_near =
            geometricNearGaussianMask(points, support_radii);
        geometric_near_count = geometric_near.sum().item<int64_t>();
        prune_mask |= geometric_near;
    }
    if (!final_cleanup && prune_far_voxels_) {
        const auto far = denseCoreFarGaussianMask();
        far_count = far.sum().item<int64_t>();
        prune_mask |= far;
    }

    const int64_t selected = prune_mask.sum().item<int64_t>();
    if (selected == 0) {
        return;
    }
    gaussians_->prunePoints(prune_mask);
    std::cout << (final_cleanup ? "[Gaussian FINAL/prune]" : "[Gaussian PRUNE]")
              << " before=" << point_count
              << " selected=" << selected
              << " camera_near=" << camera_near_count
              << " geometric_near=" << geometric_near_count;
    if (!final_cleanup) {
        std::cout << " far=" << far_count;
    }
    std::cout << " remaining=" << gaussians_->getXYZ().size(0) << "\n";
}
