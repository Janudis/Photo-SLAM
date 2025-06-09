#pragma once
/* -------------------------------------------------------------------------- */
/*  Photo-SLAM-style voxel key-frame                                           */
/* -------------------------------------------------------------------------- */
#include "include_voxel/voxel_camera.h"
#include "include_voxel/mini_cam.h"

#include <torch/torch.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <vector>
#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

/**
 * A light wrapper that stores exactly the fields the mapper uses, mirroring
 * Photo-SLAM’s GaussianKeyframe interface so the training / rendering code
 * can stay unchanged.
 */
struct VoxelKeyframe
{
    /* ------------------------------------------------------------------ ctor */
    VoxelKeyframe(std::size_t fid = 0, int create_iter = 0)
        : fid_(fid), create_iter_(create_iter) {}

    /* ---------------------------------------------------- pose setters (two) */
    /** Raw 7-number form (kept for completeness) */
    void setPose(double qw,double qx,double qy,double qz,
                 double tx,double ty,double tz);

    /** Convenient Eigen overload (what mapper calls) */
    void setPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& t);

    /* --------------------------------------------------------- pose getters */
    Sophus::SE3f getPosef() const;
    Sophus::SE3d getPose()  const;

    /* --------------------------------------------------------- camera stuff */
    void               setCameraParams(const sv::Camera& cam);
    const sv::Camera&  getCamera() const;

    /* -------------------------------------------- build c2w / w2c 4×4 mats */
    void computeTransformTensors();

    /* --------------------------------------------- helper for sverte raster */
    sv::MiniCam toMiniCam() const;

    /* ---------------------------------------------------------------- data */
    std::size_t   fid_         = 0;
    int           create_iter_ = 0;

    Sophus::SE3f  Tcw;                    // world→camera pose (float)

    /* near / far copied from Mapper globals for each new KF */
    float         zfar_  = 4.f;
    float         znear_ = 0.02f;

    sv::Camera    cam_;                   // intrinsics + pre-built undistort maps

    torch::Tensor c2w_, w2c_;             // 4×4 float32 matrices
    torch::Tensor original_image_;        // (C,H,W)   – raw RGB after undistort
    torch::Tensor img_tensor;             // copy on device for training
    torch::Tensor mask_tensor;            // (1,H,W) float32 mask

    /* bookkeeping */
    int           remaining_times_of_use_ = 0;
    int           camera_id_ = 0;         // <== NEW: used to index into undistort_mask_

    /* --------------- filenames (logger / viewer expect these fields) ------ */
    std::string   img_filename_;          // just the basename
    std::string   img_path_;              // absolute or relative path
};
