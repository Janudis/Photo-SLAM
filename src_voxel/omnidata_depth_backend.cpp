#include "include_voxel/omnidata_depth_backend.h"

#include <ATen/autocast_mode.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <future>
#include <stdexcept>
#include <utility>

namespace sv {
namespace {

torch::Tensor rgbMatToTensor(const cv::Mat& rgb)
{
    if (rgb.empty() || rgb.type() != CV_8UC3 || !rgb.isContinuous()) {
        throw std::runtime_error(
            "Omnidata input must be a continuous CV_8UC3 RGB image");
    }
    return torch::from_blob(
               const_cast<uint8_t*>(rgb.ptr<uint8_t>()),
               {rgb.rows, rgb.cols, 3},
               torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
        .permute({2, 0, 1})
        .unsqueeze(0)
        .to(torch::kFloat32)
        .div_(255.0f)
        .clone();
}

class AutocastGuard
{
public:
    explicit AutocastGuard(const bool enabled)
        : previous_(at::autocast::is_enabled())
    {
        at::autocast::set_autocast_gpu_dtype(torch::kFloat16);
        at::autocast::set_enabled(enabled);
    }

    ~AutocastGuard()
    {
        at::autocast::set_enabled(previous_);
        at::autocast::clear_cache();
    }

private:
    bool previous_;
};

} // namespace

class OmnidataDepthBackend::Impl
{
public:
    Impl(
        const std::filesystem::path& model_file,
        const int input_size_value,
        const bool use_amp_value)
        : module(torch::jit::load(model_file.string(), torch::kCUDA)),
          input_size(input_size_value),
          use_amp(use_amp_value)
    {
        module.eval();
    }

    torch::jit::Module module;
    int input_size = 512;
    bool use_amp = true;
    bool pending = false;
    std::future<OmnidataDepthResult> future;
};

OmnidataDepthBackend::OmnidataDepthBackend(
    const std::filesystem::path& model_file,
    const int input_size,
    const bool use_amp)
{
    if (!std::filesystem::exists(model_file)) {
        throw std::runtime_error(
            "Omnidata TorchScript model does not exist: " +
            model_file.string());
    }
    if (input_size <= 0 || input_size % 32 != 0) {
        throw std::runtime_error(
            "Omnidata input size must be a positive multiple of 32");
    }
    impl_ = std::make_unique<Impl>(model_file, input_size, use_amp);
}

OmnidataDepthBackend::~OmnidataDepthBackend()
{
    if (!impl_ || !impl_->pending) {
        return;
    }
    try {
        impl_->future.wait();
        impl_->future.get();
        impl_->pending = false;
    } catch (...) {
        // Destructors must not propagate inference failures.
    }
}

bool OmnidataDepthBackend::hasPending() const
{
    return impl_ && impl_->pending;
}

bool OmnidataDepthBackend::resultReady() const
{
    return impl_ && impl_->pending &&
        impl_->future.wait_for(std::chrono::seconds(0)) ==
            std::future_status::ready;
}

void OmnidataDepthBackend::launch(
    const std::vector<OmnidataDepthInput>& inputs,
    const int output_width,
    const int output_height,
    const float depth_multiplier)
{
    if (!impl_) {
        throw std::runtime_error("Omnidata backend is not initialized");
    }
    if (impl_->pending) {
        throw std::runtime_error(
            "Omnidata launch requested while inference is pending");
    }
    if (inputs.empty() || output_width <= 0 || output_height <= 0 ||
        !(depth_multiplier > 0.0f)) {
        throw std::runtime_error("Invalid Omnidata inference request");
    }

    std::vector<OmnidataDepthInput> owned_inputs;
    owned_inputs.reserve(inputs.size());
    for (const OmnidataDepthInput& input : inputs) {
        if (input.rgb.empty() || input.rgb.type() != CV_8UC3) {
            throw std::runtime_error(
                "Omnidata inputs must be CV_8UC3 RGB images");
        }
        owned_inputs.push_back({input.keyframe_id, input.rgb.clone()});
    }

    Impl* const impl = impl_.get();
    impl_->pending = true;
    impl_->future = std::async(
        std::launch::async,
        [impl,
         inputs = std::move(owned_inputs),
         output_width,
         output_height,
         depth_multiplier]() mutable {
            torch::InferenceMode inference_guard;
            OmnidataDepthResult result;
            result.frames.reserve(inputs.size());

            const torch::Tensor mean = torch::tensor(
                {0.485f, 0.456f, 0.406f},
                torch::TensorOptions().dtype(torch::kFloat32))
                .view({1, 3, 1, 1})
                .to(torch::kCUDA);
            const torch::Tensor std_tensor = torch::tensor(
                {0.229f, 0.224f, 0.225f},
                torch::TensorOptions().dtype(torch::kFloat32))
                .view({1, 3, 1, 1})
                .to(torch::kCUDA);

            for (const OmnidataDepthInput& input : inputs) {
                torch::Tensor image = rgbMatToTensor(input.rgb).to(torch::kCUDA);
                image = image.sub(mean).div(std_tensor);
                image = torch::nn::functional::interpolate(
                    image,
                    torch::nn::functional::InterpolateFuncOptions()
                        .size(std::vector<int64_t>{
                            impl->input_size, impl->input_size})
                        .mode(torch::kBilinear)
                        .align_corners(false)
                        .antialias(true));

                torch::Tensor depth;
                {
                    AutocastGuard autocast(impl->use_amp);
                    depth = impl->module.forward({image}).toTensor();
                }
                if (depth.dim() == 3) {
                    depth = depth.unsqueeze(1);
                } else if (depth.dim() == 2) {
                    depth = depth.unsqueeze(0).unsqueeze(0);
                }
                if (depth.dim() != 4 || depth.size(0) != 1 ||
                    depth.size(1) != 1) {
                    throw std::runtime_error(
                        "Omnidata TorchScript output must be [1,H,W] or [1,1,H,W]");
                }
                depth = torch::nn::functional::interpolate(
                            depth.to(torch::kFloat32) * depth_multiplier,
                            torch::nn::functional::InterpolateFuncOptions()
                                .size(std::vector<int64_t>{
                                    output_height, output_width})
                                .mode(torch::kBicubic)
                                .align_corners(false))
                            .squeeze(0)
                            .squeeze(0)
                            .to(torch::kCPU)
                            .contiguous();

                cv::Mat depth_mat(
                    output_height,
                    output_width,
                    CV_32FC1,
                    depth.data_ptr<float>());
                result.frames.push_back(
                    {input.keyframe_id, depth_mat.clone()});
            }
            return result;
        });
}

std::optional<OmnidataDepthResult> OmnidataDepthBackend::collect(
    const bool wait_for_result)
{
    if (!impl_ || !impl_->pending) {
        return std::nullopt;
    }
    if (wait_for_result) {
        impl_->future.wait();
    } else if (!resultReady()) {
        return std::nullopt;
    }

    try {
        OmnidataDepthResult result = impl_->future.get();
        impl_->pending = false;
        return result;
    } catch (...) {
        impl_->pending = false;
        throw;
    }
}

} // namespace sv
