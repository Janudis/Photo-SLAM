#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace sv {

struct TandemMvsResult
{
    // Depth uses the same coordinate units as the supplied camera poses.
    cv::Mat depth;
    cv::Mat confidence;
};

// Ownership and input-validation layer around the project-local copy of
// TANDEM's published asynchronous DrMvsnet runtime.
class TandemMvsBackend
{
public:
    explicit TandemMvsBackend(const std::filesystem::path& model_file);
    ~TandemMvsBackend();

    TandemMvsBackend(const TandemMvsBackend&) = delete;
    TandemMvsBackend& operator=(const TandemMvsBackend&) = delete;

    bool hasPending() const;
    bool resultReady() const;

    void launch(
        const std::vector<cv::Mat>& bgr_images,
        const Eigen::Matrix3f& intrinsics,
        const std::vector<Eigen::Matrix4f>& camera_to_world,
        float depth_min,
        float depth_max,
        float discard_percentage);

    std::optional<TandemMvsResult> collect(bool wait_for_result);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sv
