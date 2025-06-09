#pragma once

#include <torch/torch.h>
#include <memory>
#include <vector>
#include <functional>

#include "include_voxel/voxel_model.h"
#include "include_voxel/voxel_scene.h"
#include "include/loss_utils.h"

namespace sv {

class VoxelTrainer {
public:
    VoxelTrainer() = delete;

    static void trainingOnce(
        std::shared_ptr<VoxelScene> scene,
        std::shared_ptr<VoxelModel> voxels,
        VoxelModelParams& dataset,
        VoxelOptimizationParams& optParams,
        VoxelPipelineParams& pipeParams,
        torch::DeviceType device_type = torch::kCUDA,
        std::vector<int> testing_iterations = {},
        std::vector<int> saving_iterations = {},
        std::vector<int> checkpoint_iterations = {}
    );

    /// After each optimization step (or at logging intervals), print a concise report:
    ///   iteration, num_iterations, L1‐loss, total loss, EMA, elapsed_time, 
    ///   VoxelModel state, VoxelScene state, pipeline flags, background image, etc.
    static void trainingReport(
        int iteration,
        int num_iterations,
        torch::Tensor& Ll1,
        torch::Tensor& loss,
        float ema_loss_for_log,
        std::function<torch::Tensor(torch::Tensor&, torch::Tensor&)> l1_loss,
        int64_t elapsed_time,
        VoxelModel& voxels,
        VoxelScene& scene,
        VoxelPipelineParams& pipeParams,
        torch::Tensor& background
    );
};
} // namespace sv
