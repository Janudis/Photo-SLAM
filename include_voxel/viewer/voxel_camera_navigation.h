#pragma once

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>

namespace voxel_viewer_navigation
{
inline Eigen::Matrix3f alignedRotation(
    const Eigen::Matrix3f& fallback, const Eigen::Vector3f& forward,
    const Eigen::Vector3f& reference_down)
{
    if (!forward.allFinite() || !reference_down.allFinite() || forward.norm() < 1e-6f)
        return fallback;
    const Eigen::Vector3f z = forward.normalized();
    const Eigen::Vector3f right = reference_down.cross(z);
    if (right.norm() < 1e-6f)
        return fallback;
    Eigen::Matrix3f rotation;
    rotation.col(0) = right.normalized();
    rotation.col(1) = z.cross(rotation.col(0));
    rotation.col(2) = z;
    return rotation;
}

inline float fitDistance(
    const Eigen::Vector3f& minimum, const Eigen::Vector3f& maximum,
    const Eigen::Matrix3f& camera_to_world,
    float fx, float fy, float cx, float cy, float width, float height,
    float padding = 1.08f)
{
    const Eigen::Vector3f center = 0.5f * (minimum + maximum);
    float distance = 0.1f;
    // Fit all eight corners in the actual, possibly off-center camera frustum.
    for (int corner = 0; corner < 8; ++corner)
    {
        Eigen::Vector3f point;
        for (int axis = 0; axis < 3; ++axis)
            point[axis] = (corner & (1 << axis)) ? maximum[axis] : minimum[axis];
        const Eigen::Vector3f p = camera_to_world.transpose() * (point - center);
        const float tan_x = std::max((p.x() < 0 ? cx : width - cx) / fx, 0.01f);
        const float tan_y = std::max((p.y() < 0 ? cy : height - cy) / fy, 0.01f);
        distance = std::max(distance, padding * std::abs(p.x()) / tan_x - p.z());
        distance = std::max(distance, padding * std::abs(p.y()) / tan_y - p.z());
        distance = std::max(distance, 0.1f - p.z());
    }
    return distance;
}

inline Eigen::Vector3f zoomPosition(
    const Eigen::Vector3f& position, const Eigen::Vector3f& center, float wheel)
{
    const Eigen::Vector3f offset = position - center;
    const float distance = offset.norm();
    if (distance < 1.0e-6f)
        return position;
    const float next = std::clamp(
        distance * std::pow(1.2f, -std::clamp(wheel, -20.0f, 20.0f)),
        0.05f, 900.0f);
    return center + offset * (next / distance);
}

inline Eigen::Vector3f panTranslation(
    const Eigen::Matrix3f& camera_to_world, float dx, float dy,
    float distance, float screen_fx, float screen_fy)
{
    return camera_to_world * Eigen::Vector3f(
        -dx * distance / screen_fx, -dy * distance / screen_fy, 0.0f);
}

inline Eigen::Vector3f orbitPosition(
    const Eigen::Vector3f& position, const Eigen::Vector3f& center,
    const Eigen::Matrix3f& rotation, const Eigen::Matrix3f& local_rotation)
{
    return center + rotation * local_rotation * rotation.transpose() * (position - center);
}
} // namespace voxel_viewer_navigation
