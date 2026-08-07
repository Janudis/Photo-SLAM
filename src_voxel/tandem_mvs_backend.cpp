#include "include_voxel/tandem_mvs_backend.h"
#include "include_voxel/tandem_mvs_runtime.h"

#include <torch/csrc/jit/codegen/cuda/interface.h>

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

namespace sv {

class TandemMvsBackend::Impl
{
public:
    explicit Impl(const std::filesystem::path& model_file)
        : model(model_file.string().c_str())
    {
    }

    DrMvsnet model;
    bool pending = false;
    int width = 0;
    int height = 0;
};

TandemMvsBackend::TandemMvsBackend(
    const std::filesystem::path& model_file)
{
    if (!std::filesystem::exists(model_file)) {
        throw std::runtime_error(
            "TANDEM MVS model does not exist: " + model_file.string());
    }

    // TANDEM was exported and validated with LibTorch 1.9. LibTorch 2.0's
    // nvFuser profiling executor is unstable when this TorchScript model runs
    // concurrently with SVRecon's eager CUDA workload: repeated fusion
    // fallbacks eventually corrupt the JIT interpreter stack. The model uses
    // fixed input shapes, so disabling this optional fuser preserves its
    // numerics while retaining the regular TorchScript executor and the
    // official asynchronous TANDEM worker.
    if (torch::jit::fuser::cuda::isEnabled()) {
        torch::jit::fuser::cuda::setEnabled(false);
        std::cout
            << "[VoxelMapper] Disabled TorchScript nvFuser for TANDEM/LibTorch "
               "2.0 compatibility.\n";
    }
    impl_ = std::make_unique<Impl>(model_file);
}

TandemMvsBackend::~TandemMvsBackend()
{
    if (!impl_ || !impl_->pending) {
        return;
    }
    try {
        impl_->model.Wait();
        std::unique_ptr<DrMvsnetOutput> output(impl_->model.GetResult());
        impl_->pending = false;
    } catch (...) {
        // Destructors must not propagate failures during shutdown.
    }
}

bool TandemMvsBackend::hasPending() const
{
    return impl_ && impl_->pending;
}

bool TandemMvsBackend::resultReady() const
{
    return impl_ && impl_->pending && impl_->model.Ready();
}

void TandemMvsBackend::launch(
    const std::vector<cv::Mat>& bgr_images,
    const Eigen::Matrix3f& intrinsics,
    const std::vector<Eigen::Matrix4f>& camera_to_world,
    const float depth_min,
    const float depth_max,
    const float discard_percentage)
{
    if (!impl_) {
        throw std::runtime_error("TANDEM MVS backend is not initialized");
    }
    if (impl_->pending) {
        throw std::runtime_error(
            "TANDEM MVS launch requested while an inference is pending");
    }
    if (bgr_images.size() < 2 ||
        bgr_images.size() != camera_to_world.size()) {
        throw std::runtime_error(
            "TANDEM MVS requires matching image/pose arrays with at least two views");
    }
    if (!(depth_min > 0.0f && depth_max > depth_min)) {
        throw std::runtime_error("Invalid TANDEM MVS depth interval");
    }

    const int height = bgr_images.front().rows;
    const int width = bgr_images.front().cols;
    std::vector<unsigned char*> image_ptrs;
    image_ptrs.reserve(bgr_images.size());
    for (const cv::Mat& image : bgr_images) {
        if (image.empty() || image.rows != height || image.cols != width ||
            image.type() != CV_8UC3 || !image.isContinuous()) {
            throw std::runtime_error(
                "TANDEM MVS images must be continuous CV_8UC3 with equal dimensions");
        }
        image_ptrs.push_back(const_cast<unsigned char*>(image.ptr<unsigned char>()));
    }

    std::array<float, 9> intrinsics_row_major{};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            intrinsics_row_major[3 * row + col] = intrinsics(row, col);
        }
    }

    std::vector<std::array<float, 16>> poses_row_major(
        camera_to_world.size());
    std::vector<float*> pose_ptrs;
    pose_ptrs.reserve(camera_to_world.size());
    for (std::size_t view = 0; view < camera_to_world.size(); ++view) {
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                poses_row_major[view][4 * row + col] =
                    camera_to_world[view](row, col);
            }
        }
        pose_ptrs.push_back(poses_row_major[view].data());
    }

    // DrMvsnet copies all input arrays before CallAsync returns.
    impl_->model.CallAsync(
        height,
        width,
        static_cast<int>(bgr_images.size()),
        /*ref_index=*/0,
        image_ptrs.data(),
        intrinsics_row_major.data(),
        pose_ptrs.data(),
        depth_min,
        depth_max,
        discard_percentage,
        /*debug_print=*/false);
    impl_->width = width;
    impl_->height = height;
    impl_->pending = true;
}

std::optional<TandemMvsResult> TandemMvsBackend::collect(
    const bool wait_for_result)
{
    if (!impl_ || !impl_->pending) {
        return std::nullopt;
    }
    if (wait_for_result) {
        impl_->model.Wait();
    } else if (!impl_->model.Ready()) {
        return std::nullopt;
    }

    std::unique_ptr<DrMvsnetOutput> output(impl_->model.GetResult());
    TandemMvsResult result;
    result.depth = cv::Mat(
        impl_->height,
        impl_->width,
        CV_32FC1,
        output->depth).clone();
    result.confidence = cv::Mat(
        impl_->height,
        impl_->width,
        CV_32FC1,
        output->confidence).clone();
    impl_->pending = false;
    return result;
}

} // namespace sv
