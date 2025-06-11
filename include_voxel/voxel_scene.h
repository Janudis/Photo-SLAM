#pragma once
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <filesystem>

#include "include/types.h"                 // camera_id_t / point3D_id_t
#include "include_voxel/voxel_parameters.h"
#include "include_voxel/voxel_camera.h"    // sv::Camera
#include "include_voxel/voxel_keyframe.h"  // VoxelKeyframe
#include "include/point3d.h"

namespace sv {

class VoxelScene
{
public:
    VoxelScene( VoxelModelParams& args,
                int  load_iteration = 0,
                bool shuffle        = true,
                std::vector<float> resolution_scales = {1.f} );

    /* ───── camera pool ─────────────────────────────────────────────── */
    void   addCamera   (const Camera& cam);
    Camera& getCamera  (camera_id_t id);

    /* ───── keyframe pool ───────────────────────────────────────────── */
    void addKeyframe(const std::shared_ptr<VoxelKeyframe>& kf, bool* shuffled);
    std::shared_ptr<VoxelKeyframe>             getKeyframe (std::size_t fid);
    std::map<std::size_t,std::shared_ptr<VoxelKeyframe>>&  keyframes();
    std::map<std::size_t,std::shared_ptr<VoxelKeyframe>>   getAllKeyframes();

    /* ───── sparse point cache (for colours etc.) ───────────────────── */
    void     cachePoint3D(point3D_id_t id, const Point3D& pt);
    Point3D& getPoint3D (point3D_id_t id);
    void     clearCachedPoint3D();

    /* ───── helpers ─────────────────────────────────────────────────── */
    void applyScaledTransformation(float s = 1.f,
                                   const Sophus::SE3f T = Sophus::SE3f(Eigen::Matrix3f::Identity(),
                                                                       Eigen::Vector3f::Zero()));
    std::tuple<Eigen::Vector3f,float> getNerfppNorm();

    std::tuple< std::map<std::size_t,std::shared_ptr<VoxelKeyframe>>,
                std::map<std::size_t,std::shared_ptr<VoxelKeyframe>> >
        splitTrainAndTestKeyframes(float test_ratio);

public:
    float cameras_extent_ = 1.f;   ///< nerf-pp normalisation radius
    int   loaded_iter_    = 0;

    std::map<camera_id_t, Camera>                       cameras_;
    std::map<std::size_t,std::shared_ptr<VoxelKeyframe>> keyframes_;
    std::map<point3D_id_t, Point3D>                      cached_point_cloud_;

protected:
    std::mutex mutex_kfs_;
};

} // namespace sv
