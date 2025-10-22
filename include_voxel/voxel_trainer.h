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
    VoxelTrainer();

    static void trainingOnce(
        std::shared_ptr<sv::VoxelScene> scene,
        std::shared_ptr<sv::VoxelModel> voxels,
        VoxelModelParams& dataset,
        VoxelOptimizationParams& optParams,
        VoxelPipelineParams& pipeParams,
        torch::DeviceType device_type = torch::kCUDA,
        std::vector<int> testing_iterations = {},
        std::vector<int> saving_iterations = {},
        std::vector<int> checkpoint_iterations = {}
    );

    static void trainingReport(
        int iteration,
        int num_iterations,
        torch::Tensor& Ll1,
        torch::Tensor& loss,
        float ema_loss_for_log,
        torch::Tensor& mse,
        int64_t elapsed_time,
        sv::VoxelModel& voxels,
        sv::VoxelScene& scene,
        VoxelPipelineParams& pipeParams,
        torch::Tensor& background
    );
};

}