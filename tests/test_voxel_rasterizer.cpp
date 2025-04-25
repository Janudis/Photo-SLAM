#include <torch/extension.h>
#include <iostream>
#include "cuda_voxel_rasterizer/voxel_rasterizer_api.h"

int main() {
    torch::Device dev(torch::kCUDA);
    // One voxel, 16-float layout as in wrapper
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
    torch::Tensor vox = torch::zeros({1,16}, opts);
    vox[0][0] = 0.f; vox[0][1] = 0.f; vox[0][2] = 3.f;   // xyz
    vox[0][3] = 2.f;                                     // half size
    vox.slice(1, 4, 12).fill_(4.f);                      // densities
    vox.slice(1, 12, 15).copy_(torch::tensor({1.f,0.f,0.f}, opts)); // red
    vox[0][15] = 1.f;                                    // level

    torch::Tensor K   = torch::eye(3, opts);
    torch::Tensor Twc = torch::eye(4, opts);

    auto out = voxel_rasterizer_forward(vox, K, Twc, 64, 64);
    std::cout << "Output size: " << out.sizes() << std::endl;
    return 0;
}
