#pragma once

#include <torch/torch.h>
#include <filesystem>
#include <string>
#include <vector>
#include <unordered_map>
#include <pybind11/numpy.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "ORB-SLAM3/include/System.h"

#include "include_voxel/voxel_parameters.h"

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
                    torch::Tensor octpath,
                    torch::Tensor octlevel,
                    torch::Tensor subdiv_meta,
                    torch::Tensor subdiv_p);

    void increasePcd(const std::vector<Eigen::Vector3f>& points,
                 const std::vector<Eigen::Vector3f>& colors,
                 int iteration);

    // save final model
    void save_torch(const std::filesystem::path& p) const;

    // for optimizer if you want to train
    std::vector<torch::Tensor> parameters();
    
    torch::Tensor get_tensor(const std::string& name) const;
    void subdivide(const torch::Tensor& mask);   // mask.shape == (N,) • bool
    void prune(const torch::Tensor& mask_keep);
    inline int num_voxels() const { return center_.size(0); }
    void set_subdiv_meta(const torch::Tensor& updated);
    torch::Tensor get_subdiv_priority_grad() const;
    void accumulate_subdiv_gradients(const torch::Tensor& parent_idx, const torch::Tensor& parent_grads);

    void trainingSetup(const VoxelOptimizationParams& opt_params);
    float updateLearningRate(int step);
    void setGeoLearningRate(float lr);        // mirrors setPositionLearningRate
    void setSh0LearningRate(float lr);        // mirrors setFeatureLearningRate(group-1)
    void setShsLearningRate(float lr);        // mirrors setFeatureLearningRate(group-2)

    void setShDegree(const int sh);

    std::shared_ptr<torch::optim::Adam> optimizer_;

protected:
    float exponLrFunc(int step);

private:
    int G_;
    torch::Tensor center_, size_, geo_, sh0_, shs_, opacity_, oct_path_;
    torch::Tensor oct_level_, subdiv_meta_;
    torch::Tensor subdiv_p_;
    torch::Tensor subdiv_p_grad_buffer_;
    int active_sh_degree_ = 0;
    int max_sh_degree_    = 3;

protected:
    float geo_lr;           
    float sh0_lr;           
    float shs_lr;          
    float meta_accum_lr;    
    float lr_init_;
    float lr_final_;
    int lr_delay_steps_;
    float lr_delay_mult_;
    int max_steps_;

    std::mutex mutex_settings_;
};

} // namespace sv
