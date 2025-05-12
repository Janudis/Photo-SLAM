#pragma once

#include <torch/torch.h>
#include <System.h>               // ORB_SLAM3::System
#include <filesystem>
#include <vector>
#include <string>
#include <memory>
#include <pybind11/numpy.h>       // for pybind11::array_t

namespace sv {
    class VoxelTrainer;
}

class VoxelMapper {
public:
    VoxelMapper(std::shared_ptr<ORB_SLAM3::System> slam,
        const std::filesystem::path& voxel_cfg,
        const std::filesystem::path& seq_dir,
        const std::filesystem::path& out_dir,
        torch::Device device);
    ~VoxelMapper();  
    PyThreadState* m_savedState = nullptr;   // remembers the thread‑state
    void run();

private:
    void LoadImages(const std::string& rgb_txt_path);
    bool initializeVoxelsFromMap(torch::Tensor& out_centers, float voxel_size);
    void finalize();
    void renderAndDumpAllKeyframes(const std::string &tag = "0000");
    void trainLoopWithOptimization(int num_iters);

    std::shared_ptr<ORB_SLAM3::System> mpSLAM;
    std::shared_ptr<sv::VoxelTrainer> mpTrainer;

    std::filesystem::path mSeqDir;
    std::filesystem::path mVoxelCfg;
    std::filesystem::path mOutDir;
    torch::Device mDevice;

    std::vector<std::string> mImagePaths;
    std::vector<double> mTimestamps;
    std::vector<Sophus::SE3f> mTcwList;      // <- keep the poses we get
    std::vector<int>          mTrackState;

    std::vector<Sophus::SE3f> mKeyframePoses;
    std::vector<int>          mKeyframeIds;
    std::vector<std::string>  mKeyframeImages;
    int mIteration = 0;
    int getIteration() const { return mIteration; }
    std::vector<int> mKeyFrameIds;
};