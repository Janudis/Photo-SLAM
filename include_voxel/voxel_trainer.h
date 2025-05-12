#pragma once

#include <torch/torch.h>
#include <filesystem>
#include <string>
#include <vector>
#include <unordered_map>
#include <pybind11/numpy.h>

namespace sv {
    class MiniCam;
}

namespace sv {

class VoxelTrainer {
public:
    explicit VoxelTrainer(int grid_res);

    // render into `output_dir`; returns a map of output tensors (e.g. "rgb")
    std::unordered_map<std::string, torch::Tensor>
    render(const MiniCam& cam,
           const pybind11::array_t<uint8_t>& rgb_image,
           const std::string& output_dir);

    // supply the voxel data each frame
    void set_voxels(torch::Tensor center,
                    torch::Tensor size,
                    torch::Tensor geo,
                    torch::Tensor sh0,
                    torch::Tensor shs,
                    torch::Tensor opacity,
                    torch::Tensor octpath);

    // save final model
    void save_torch(const std::filesystem::path& p) const;

    // for optimizer if you want to train
    std::vector<torch::Tensor> parameters();

private:
    int G_;
    torch::Tensor center_, size_, geo_, sh0_, shs_, opacity_, oct_path_;
};

} // namespace sv
