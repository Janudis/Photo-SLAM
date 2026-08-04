#pragma once

#include <opencv2/core.hpp>

#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace sv {

struct OmnidataDepthInput
{
    unsigned long keyframe_id = 0;
    cv::Mat rgb;
};

struct OmnidataDepthFrame
{
    unsigned long keyframe_id = 0;
    cv::Mat relative_depth;
};

struct OmnidataDepthResult
{
    std::vector<OmnidataDepthFrame> frames;
};

// Asynchronous depth-only wrapper around a fixed-resolution TorchScript export
// of HI-SLAM2's Omnidata DPT model.
class OmnidataDepthBackend
{
public:
    OmnidataDepthBackend(
        const std::filesystem::path& model_file,
        int input_size,
        bool use_amp);
    ~OmnidataDepthBackend();

    OmnidataDepthBackend(const OmnidataDepthBackend&) = delete;
    OmnidataDepthBackend& operator=(const OmnidataDepthBackend&) = delete;

    bool hasPending() const;
    bool resultReady() const;

    void launch(
        const std::vector<OmnidataDepthInput>& inputs,
        int output_width,
        int output_height,
        float depth_multiplier);

    std::optional<OmnidataDepthResult> collect(bool wait_for_result);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sv
