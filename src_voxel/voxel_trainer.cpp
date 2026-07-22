#include "include_voxel/voxel_trainer.h"

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <random>
#include <iostream>

sv::VoxelTrainer::VoxelTrainer()
{}

// void sv::VoxelTrainer::trainingOnce(
//     std::shared_ptr<VoxelScene>    scene,
//     std::shared_ptr<VoxelModel>    voxels,
//     VoxelModelParams&              dataset,
//     VoxelOptimizationParams&       optParams,
//     VoxelPipelineParams&           pipeParams,     // now really used
//     c10::DeviceType                device_type,
//     std::vector<int> /*testing_iterations*/,
//     std::vector<int> /*saving_iterations*/,
//     std::vector<int> /*checkpoint_iterations*/
// ) {
//     // 1) Set up optimizer & background
//     voxels->trainingSetup(optParams);
//     torch::Tensor background = torch::tensor(
//         dataset.white_background_ ? std::vector<float>{1,1,1}
//                                   : std::vector<float>{0,0,0},
//         torch::TensorOptions().dtype(torch::kFloat32).device(device_type)
//     );

//     float ema_loss_for_log = 0.0f;

//     // 2) Main training loop
//     for (int iteration = 1; iteration <= optParams.iterations_; ++iteration) {
//         auto iter_start = std::chrono::steady_clock::now();

//         // 2.1) Update LR & maybe bump SH‐degree
//         voxels->updateLearningRate(iteration);
//         if (iteration % 1000 == 0) voxels->oneUpShDegree();

//         // 2.2) Pick a random keyframe
//         auto &kfs = scene->keyframes();
//         if (kfs.empty()) {
//             std::cerr << "[VoxelTrainer] No keyframes available\n";
//             return;
//         }
//         int idx = std::rand() % kfs.size();
//         auto it = kfs.begin();
//         std::advance(it, idx);
//         auto vf = it->second;

//         // 2.3) Make a H×W uint8 NumPy image from original_image_
//         // original_image_ is (3,H,W) float in [0..1]
//         int H = vf->original_image_.size(1),
//             W = vf->original_image_.size(2);
//         auto cpu_img = (vf->original_image_ * 255.0f)
//                          .clamp(0,255)
//                          .to(torch::kU8)
//                          .permute({1,2,0})
//                          .contiguous();                // (H,W,3)
//         py::array_t<uint8_t> rgb_np = tensorToNumpyRGB(cpu_img);

//         sv::MiniCam cam = vf->toMiniCam();
//         cam.c2w = cam.c2w.contiguous().to(device_type);   // make sure contiguous + on CUDA
//         cam.w2c = cam.w2c.contiguous().to(device_type);        

//         // 2.5) Render (only returns “rgb”)
//         auto render_map = voxels->render(cam, rgb_np, "");
//         auto image      = render_map.at("rgb").to(device_type);  // (1,3,H,W)

//         // 2.6) Photometric + SSIM loss
//         auto gt = vf->original_image_.unsqueeze(0).to(device_type); // (1,3,H,W)
//         auto Ll1 = loss_utils::l1_loss(image, gt);
//         auto loss = Ll1;
//         loss.backward();

//         // 2.7) Sync + timing
//         if (device_type == torch::kCUDA) torch::cuda::synchronize();
//         auto iter_end = std::chrono::steady_clock::now();
//         int64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
//                               iter_end - iter_start
//                              ).count();

//         // 2.8) Logging, densify/prune, optimizer step
//         {
//             torch::NoGradGuard ng;
//             ema_loss_for_log = 0.4f * loss.template item<float>()
//                              + 0.6f * ema_loss_for_log;

//             // trainingReport needs a real pipeParams&
//             trainingReport(
//                 iteration,
//                 optParams.iterations_,
//                 Ll1, loss, ema_loss_for_log,
//                 loss_utils::l1_loss,
//                 elapsed_ms,
//                 *voxels, *scene,
//                 pipeParams,    // <— pass the real pipeline params
//                 background
//             );

//             // // densify & prune
//             // if (iteration < optParams.densify_until_iter_) {
//             //     if (iteration > optParams.densify_from_iter_
//             //         && iteration % optParams.densification_interval_ == 0)
//             //     {
//             //         int sz = (iteration > optParams.opacity_reset_interval_) ? 20 : 0;
//             //         voxels->densifyAndPrune(
//             //             optParams.densify_grad_threshold_,
//             //             0.005f,
//             //             scene->cameras_extent_,
//             //             sz
//             //         );
//             //     }
//             //     if (iteration % optParams.opacity_reset_interval_ == 0
//             //         || (dataset.white_background_ && iteration == optParams.densify_from_iter_))
//             //     {
//             //         voxels->resetOpacity();
//             //     }
//             // }

//             // optimizer
//             if (iteration < optParams.iterations_) {
//                 voxels->optimizer_->step();
//                 voxels->optimizer_->zero_grad(true);
//             }
//         }
//     }
// }

void sv::VoxelTrainer::trainingReport(
    int iteration,
    int num_iterations,
    const torch::Tensor& photo_loss,
    const char* photo_loss_name,
    const torch::Tensor& ssim_loss,
    float ema_total_loss,
    int64_t elapsed_time,
    sv::VoxelModel& voxels,
    sv::VoxelScene& scene,
    VoxelPipelineParams& pipeParams,
    torch::Tensor& background
) {
    std::cout << std::fixed << std::setprecision(8)
              << "Training iteration " << iteration << "/" << num_iterations
              << ", time elapsed: " << (elapsed_time / 1000.0f) << "s"
              << ", ema_total_loss: " << ema_total_loss
              << ", photo_" << photo_loss_name << ": " << photo_loss.item<float>();
    if (ssim_loss.defined()) {
        std::cout << ", one_minus_SSIM: " << ssim_loss.item<float>();
    }
    std::cout << ", num_voxels: " << voxels.numVoxels() << std::endl;
}
