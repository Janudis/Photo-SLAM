#include <torch/torch.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <pybind11/embed.h>   //  <-- add this line
#include "ORB-SLAM3/include/System.h"
#include "include_voxel/voxel_mapper.h"

int main(int argc, char** argv)
{
    pybind11::initialize_interpreter();   // acquires the GIL
    pybind11::gil_scoped_release release; // <-- immediately give it up

    if (argc != 7)
    {
        std::cerr << "Usage: " << argv[0]
                  << " path_to_vocabulary"
                  << " path_to_orbslam3_settings"
                  << " path_to_voxel_mapper_settings"
                  << " path_to_sequence_folder"
                  << " path_to_output_directory"
                  << " [viewer|no_viewer]" << std::endl;
        return 1;
    }

    std::string voc_path    = argv[1];
    std::string orb_cfg     = argv[2];
    std::string voxel_cfg   = argv[3];
    std::string sequence    = argv[4];
    std::string output_dir  = argv[5];
    std::string viewer_flag = argv[6];

    bool use_viewer = (viewer_flag != "no_viewer");

    // Choose device
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "[INFO] Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

    // Initialize ORB-SLAM3
    auto slam = std::make_shared<ORB_SLAM3::System>(
        voc_path, orb_cfg, ORB_SLAM3::System::MONOCULAR, use_viewer);

    VoxelMapper mapper(slam, voxel_cfg, sequence, output_dir, device);
    // VoxelMapper mapper(slam_system, voxel_cfg, voxel_yaml, sequence, output_dir, device);
    // Start the voxel training loop in the background:
    std::thread training_thd(&VoxelMapper::run, &mapper);
    // Wait until the mapper finishes its whole SLAM/mapping routine
    training_thd.join();
    slam->Shutdown();
    return 0;
}
