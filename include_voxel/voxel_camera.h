// -----------------------------------------------------------------------------
// include_voxel/camera.h   • minimal Photo-SLAM-compatible camera for voxels
// -----------------------------------------------------------------------------
#pragma once
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <array>
#include <vector>
#include <cassert>

namespace sv {

class Camera
{
public:
    enum CameraModelType { INVALID=0, PINHOLE=1 };

    Camera() = default;

    Camera(std::size_t width, std::size_t height,
           float fx, float fy, float cx, float cy,
           int cam_id = 0,
           const std::array<float,4>& dist = {0,0,0,0})
        : camera_id_(cam_id), width_(width), height_(height), model_id_(PINHOLE)
    {
        params_ = {fx, fy, cx, cy};
        dist_coeff_ = (cv::Mat_<float>(1,4) <<
                       dist[0], dist[1], dist[2], dist[3]);
    }

    /* ---------- intrinsic getters -------------------------------------- */
    float fx() const { return params_[0]; }
    float fy() const { return params_[1]; }
    float cx() const { return params_[2]; }
    float cy() const { return params_[3]; }
    int   width()  const { return static_cast<int>(width_); }
    int   height() const { return static_cast<int>(height_); }

    /* ---------- set model id (required by old code paths) --------------- */
    void setModelId(CameraModelType id) { model_id_ = id; }

    /* ---------- once-per-camera pre-compute ----------------------------- */
    void initUndistortRectifyMapAndMask()
    {
        cv::Mat K = (cv::Mat_<float>(3,3) <<
                     fx(), 0,   cx(),
                     0,    fy(), cy(),
                     0,    0,    1);
        cv::initUndistortRectifyMap(
            K, dist_coeff_,
            cv::Mat::eye(3,3,CV_32F), K,
            cv::Size(width(), height()),
            CV_32F,
            undistort_map1_, undistort_map2_);

        cv::Mat white(height(), width(), CV_32FC3, cv::Scalar(1,1,1));
        undistortImage(white, undistort_mask_);
        cv::cvtColor(undistort_mask_, undistort_mask_, cv::COLOR_BGR2GRAY);
    }

    template<typename Src, typename Dst>
    void undistortImage(const Src& src, Dst& dst) const
    {
        cv::remap(src, dst, undistort_map1_, undistort_map2_, cv::INTER_LINEAR);
    }

    /* ---------- public data -------------------------------------------- */
    int camera_id_ = 0;
    int model_id_  = PINHOLE;

    std::size_t width_ = 0, height_ = 0;
    std::vector<float> params_{0,0,0,0};             // fx,fy,cx,cy

    cv::Mat dist_coeff_;
    cv::Mat undistort_map1_, undistort_map2_;
    cv::Mat undistort_mask_;
};

} // namespace sv
