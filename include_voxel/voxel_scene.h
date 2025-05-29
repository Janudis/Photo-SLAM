#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <tuple>
#include <filesystem>

#include "voxel_parameters.h"
#include "voxel_keyframe.h"

class voxelScene
{
public:
    VoxelScene(
        VoxelModelParams& args,
        int load_iteration = 0,
        bool shuffle = true,
        std::vector<float> resolution_scales = {1.0f});

public:
    void addCamera(Camera& camera);
    Camera& getCamera(camera_id_t cameraId);

    void addKeyframe(std::shared_ptr<voxelKeyframe> new_kf, bool* shuffled);
    std::shared_ptr<voxelKeyframe> getKeyframe(std::size_t fid);
    std::map<std::size_t, std::shared_ptr<voxelKeyframe>>& keyframes();
    std::map<std::size_t, std::shared_ptr<voxelKeyframe>> getAllKeyframes();

    void cachePoint3D(point3D_id_t point3D_id, Point3D& point3d);
    Point3D& getPoint3D(point3D_id_t point3DId);
    void clearCachedPoint3D();

    void applyScaledTransformation(
        const float s = 1.0,
        const Sophus::SE3f T = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()));

    std::tuple<Eigen::Vector3f, float> getNerfppNorm();

    std::tuple<std::map<std::size_t, std::shared_ptr<voxelKeyframe>>,
               std::map<std::size_t, std::shared_ptr<voxelKeyframe>>>
        splitTrainAndTestKeyframes(const float test_ratio);

public:
    float cameras_extent_; ///< scene_info.nerf_normalization["radius"]

    int loaded_iter_;

    std::map<camera_id_t, Camera> cameras_;
    std::map<std::size_t, std::shared_ptr<voxelKeyframe>> keyframes_;
    std::map<point3D_id_t, Point3D> cached_point_cloud_;

protected:
    std::mutex mutex_kfs_;
};
