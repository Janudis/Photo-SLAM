// -----------------------------------------------------------------------------
// include_voxel/camera.h   • minimal Photo-SLAM-compatible camera for voxels
// -----------------------------------------------------------------------------
#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/cudawarping.hpp>
#include <torch/torch.h>
#include <array>
#include <vector>
#include <cassert>

#include "include/types.h"
#include "include/tensor_utils.h"

namespace sv {

class Camera
{
public:
    Camera() {}

    Camera (
        camera_id_t camera_id,
        std::size_t width,
        std::size_t height,
        std::vector<double> params,
        int model_id = 0,
        bool prior_focal_length = true)
        : camera_id_(camera_id),
          width_(width),
          height_(height),
          params_(params),
          model_id_(model_id),
          prior_focal_length_(prior_focal_length)
    {}

    enum CameraModelType{
        INVALID = 0,
        PINHOLE = 1,
        FISHEYE = 2};

public:
    /* ---------- small convenience getters (needed by mini_cam.h etc.) ----- */
    inline float fx()  const { return static_cast<float>(params_.at(0)); }
    inline float fy()  const { return static_cast<float>(params_.at(1)); }
    inline float cx()  const { return static_cast<float>(params_.at(2)); }
    inline float cy()  const { return static_cast<float>(params_.at(3)); }
    inline int   width()  const { return static_cast<int>(width_);  }
    inline int   height() const { return static_cast<int>(height_); }

    inline void setModelId(const CameraModelType model_id)
    {
        model_id_ = model_id;
        switch (model_id_)
        {
        case PINHOLE: // Pinhole
            params_.resize(4);
            break;

        default:
            break;
        }
    }

    inline void initUndistortRectifyMapAndMask(
        cv::InputArray old_camera_matrix,
        const cv::Size old_size,
        cv::InputArray new_camera_matrix)
    {
        cv::initUndistortRectifyMap(
            old_camera_matrix,
            dist_coeff_,
            cv::Mat::eye(3, 3, CV_32F),
            new_camera_matrix,
            cv::Size(this->width_, this->height_),
            CV_32F,
            undistort_map1,
            undistort_map2
        );

        cv::Mat white(old_size, CV_32FC3, cv::Vec3f(1.0f, 1.0f, 1.0f));
        undistortImage(white, undistort_mask);
    }

    inline void undistortImage(cv::InputArray src, cv::OutputArray dst)
    {
        cv::remap(
            src,
            dst,
            undistort_map1,
            undistort_map2,
            cv::InterpolationFlags::INTER_LINEAR
        );
    }

public:
    camera_id_t camera_id_ = 0U;

    int model_id_ = 0;

    std::size_t width_ = 0UL;
    std::size_t height_ = 0UL;

    std::vector<double> params_;

    bool prior_focal_length_ = false;

    float stereo_bf_ = 0.0f;

    cv::Mat dist_coeff_ = (cv::Mat_<float>(1, 4) << 0.0f, 0.0f, 0.0f, 0.0f);
    cv::Mat undistort_map1, undistort_map2;
    cv::Mat undistort_mask;

    // float tanfovx;
    // float tanfovy;
    // std::string cam_mode = "persp";
};

}