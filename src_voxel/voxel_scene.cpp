#include "include_voxel/voxel_scene.h"
#include <algorithm>
#include <iostream>

namespace sv {

/* ───────────────── constructor ─────────────────────────────────────── */
VoxelScene::VoxelScene(
    VoxelModelParams& args,
    int load_iteration,
    bool shuffle,
    std::vector<float> resolution_scales)
{
    if (load_iteration)
    {
        this->loaded_iter_ = load_iteration;
        std::cout << "Loading trained model at iteration " << load_iteration << std::endl;
    }
}

/* ───────────────── camera pool ─────────────────────────────────────── */
void VoxelScene::addCamera(const Camera& camera)
{
    this->cameras_.emplace(camera.camera_id_, camera);
}

Camera& VoxelScene::getCamera(camera_id_t cameraId)
{
    return this->cameras_[cameraId];
}

/* ───────────────── keyframe pool ───────────────────────────────────── */
void VoxelScene::addKeyframe(const std::shared_ptr<VoxelKeyframe>& new_kf,
                             bool* shuffled)
{
    std::unique_lock<std::mutex> lock_kfs(this->mutex_kfs_);
    this->keyframes_.emplace(new_kf->fid_, new_kf);
    *shuffled = false;
}

std::shared_ptr<VoxelKeyframe> VoxelScene::getKeyframe(std::size_t fid)
{
    std::unique_lock<std::mutex> lock_kfs(this->mutex_kfs_);
    if (this->keyframes_.find(fid) != this->keyframes_.end())
        return this->keyframes_[fid];
    else
        return nullptr;
}

std::map<std::size_t,std::shared_ptr<VoxelKeyframe>>& VoxelScene::keyframes()
{
    return this->keyframes_;
}

std::map<std::size_t,std::shared_ptr<VoxelKeyframe>>
VoxelScene::getAllKeyframes()
{
    std::lock_guard<std::mutex> lock(mutex_kfs_);
    return this->keyframes_;
}

/* ───────────────── sparse point cache ──────────────────────────────── */
void VoxelScene::cachePoint3D(point3D_id_t point3D_id, const Point3D& point3d)
{
    this->cached_point_cloud_[point3D_id] = point3d;
}

Point3D& VoxelScene::getPoint3D(point3D_id_t point3DId)
{
    if (this->cached_point_cloud_.find(point3DId) == this->cached_point_cloud_.end())
        std::cout << "VoxelScene::getPoint3D(" << point3DId << ") invalid point Id, creating new point." << std::endl;

    return this->cached_point_cloud_[point3DId];
}

void VoxelScene::clearCachedPoint3D()
{
    this->cached_point_cloud_.clear();
}

void VoxelScene::applyScaledTransformation(float s, const Sophus::SE3f T)
{
    // Apply the scaled transformation on gaussian keyframes
    for (auto& kfit : keyframes_) {
        std::shared_ptr<VoxelKeyframe> pkf = kfit.second;
        Sophus::SE3f Twc = pkf->getPosef().inverse();
        Twc.translation() *= s;
        Sophus::SE3f Tyc = T * Twc;
        Sophus::SE3f Tcy = Tyc.inverse();
        pkf->setPose(Tcy.unit_quaternion().cast<double>(), Tcy.translation().cast<double>());
        pkf->computeTransformTensors();
    }
}

std::tuple<Eigen::Vector3f,float> VoxelScene::getNerfppNorm()
{
    std::vector<Eigen::Matrix<float, 3, 1>> cam_centers;
    auto kfs = this->getAllKeyframes();
    std::size_t n_cams = kfs.size();
    cam_centers.reserve(n_cams);
    for (auto& kfit : kfs) {
        auto pkf = kfit.second;
        auto W2C = pkf->getWorld2View2();
        auto C2W = W2C.inverse();
        auto cam_center = C2W.block<3, 1>(0, 3);
        cam_centers.emplace_back(cam_center);
    }

    // get_center_and_diag(cam_centers)
    Eigen::Vector3f avg_cam_center;
    avg_cam_center.setZero();
    for (const auto& cam_center : cam_centers) {
        avg_cam_center.x() += cam_center.x();
        avg_cam_center.y() += cam_center.y();
        avg_cam_center.z() += cam_center.z();
    }
    avg_cam_center.x() /= n_cams;
    avg_cam_center.y() /= n_cams;
    avg_cam_center.z() /= n_cams;

    float max_dist = 0.0f; // diagonal
    for (std::size_t cam_idx = 0; cam_idx < n_cams; ++cam_idx) {
        float dist = (cam_centers[cam_idx] - avg_cam_center).norm();
        if (dist > max_dist)
            max_dist = dist;
    }

    float radius = max_dist * 1.1;

    Eigen::Vector3f translate = -avg_cam_center;

    return std::make_tuple(translate, radius);
}

} // namespace sv
