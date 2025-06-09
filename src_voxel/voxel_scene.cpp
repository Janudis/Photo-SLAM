#include "include_voxel/voxel_scene.h"
#include <algorithm>
#include <iostream>

namespace sv {

/* ───────────────── constructor ─────────────────────────────────────── */
VoxelScene::VoxelScene(VoxelModelParams& /*args*/,
                       int  load_iteration,
                       bool /*shuffle*/,
                       std::vector<float> /*resolution_scales*/)
{
    loaded_iter_ = load_iteration;
    if (load_iteration)
        std::cout << "[VoxelScene] Loading voxel model at iteration "
                  << load_iteration << std::endl;
}

/* ───────────────── camera pool ─────────────────────────────────────── */
void VoxelScene::addCamera(const Camera& cam)
{
    cameras_.emplace(cam.camera_id_, cam);
}

Camera& VoxelScene::getCamera(camera_id_t id)
{
    return cameras_.at(id);   // will throw if id is unknown
}

/* ───────────────── keyframe pool ───────────────────────────────────── */
void VoxelScene::addKeyframe(const std::shared_ptr<VoxelKeyframe>& kf,
                             bool* shuffled)
{
    std::lock_guard<std::mutex> lock(mutex_kfs_);
    keyframes_.emplace(kf->fid_, kf);
    if (shuffled) *shuffled = false;
}

std::shared_ptr<VoxelKeyframe> VoxelScene::getKeyframe(std::size_t fid)
{
    std::lock_guard<std::mutex> lock(mutex_kfs_);
    auto it = keyframes_.find(fid);
    return it == keyframes_.end() ? nullptr : it->second;
}

std::map<std::size_t,std::shared_ptr<VoxelKeyframe>>& VoxelScene::keyframes()
{
    return keyframes_;
}

std::map<std::size_t,std::shared_ptr<VoxelKeyframe>>
VoxelScene::getAllKeyframes()
{
    std::lock_guard<std::mutex> lock(mutex_kfs_);
    return keyframes_;   // copy
}

/* ───────────────── sparse point cache ──────────────────────────────── */
void VoxelScene::cachePoint3D(point3D_id_t id, const Point3D& pt)
{
    cached_point_cloud_[id] = pt;
}

Point3D& VoxelScene::getPoint3D(point3D_id_t id)
{
    return cached_point_cloud_[id];   // default-constructs if absent
}

void VoxelScene::clearCachedPoint3D()
{
    cached_point_cloud_.clear();
}

/* ───────────────── helper transforms ───────────────────────────────── */
void VoxelScene::applyScaledTransformation(float s, const Sophus::SE3f T)
{
    for (auto& [id, kf] : keyframes_)
    {
        Sophus::SE3f Twc = kf->getPosef().inverse();
        Twc.translation() *= s;
        Sophus::SE3f Tyc = T * Twc;
        Sophus::SE3f Tcy = Tyc.inverse();
        kf->setPose(Tcy.unit_quaternion().cast<double>(),
                    Tcy.translation().cast<double>());
        kf->computeTransformTensors();
    }
}

std::tuple<Eigen::Vector3f,float> VoxelScene::getNerfppNorm()
{
    /* identical math to Photo-SLAM version (trimmed for brevity) */
    std::vector<Eigen::Vector3f> centers;
    centers.reserve(keyframes_.size());
    for (auto& [_, kf] : keyframes_)
    {
        Eigen::Matrix4f W2C = kf->getPosef().matrix();
        Eigen::Vector3f cam = W2C.inverse().block<3,1>(0,3);
        centers.push_back(cam);
    }

    Eigen::Vector3f mean = Eigen::Vector3f::Zero();
    for (auto& c : centers) mean += c;
    mean /= static_cast<float>(centers.size());

    float max_dist = 0.f;
    for (auto& c : centers)
        max_dist = std::max(max_dist, (c - mean).norm());

    return { -mean, max_dist * 1.1f };
}

} // namespace sv
