# SVR-SLAM Demo

This branch contains the runtime code for monocular, monocular with MVS, and RGB-D voxel-splat SLAM demos. It targets CUDA desktops and Jetson Orin systems with an optional Intel RealSense input.

## Dependencies

- CUDA and cuDNN
- LibTorch with CUDA
- OpenCV with CUDA modules
- Eigen3, Boost, jsoncpp, OpenGL, GLFW, GLM, glog, pybind11
- Intel RealSense SDK for the live RGB-D runner
- Python Rerun SDK when `PHOTOSLAM_ENABLE_RERUN=ON`

Initialize the retained submodules before building:

```bash
git submodule update --init --recursive third_party/SVRecon
./build.sh
```

Set `LIBTORCH_ROOT` if LibTorch is not installed in a standard CMake search path. The build script accepts `PHOTOSLAM_BUILD_JOBS`, `PHOTOSLAM_CUDA_ARCHITECTURES`, `PHOTOSLAM_ENABLE_RERUN`, and `PHOTOSLAM_BUILD_REALSENSE`.

## Demos

Replica supports RGB-D by default and monocular or monocular+MVS through the mapper configuration:

```bash
REPLICA_ROOT=/path/to/Replica ./run_replica_voxel.sh office0
REPLICA_SENSOR_MODE=monocular REPLICA_ROOT=/path/to/Replica ./run_replica_voxel.sh office0
```

TUM supports both sensor modes:

```bash
TUM_DATA_ROOT=/path/to/TUM ./run_tum_voxel.sh rgbd_dataset_freiburg1_desk
TUM_SENSOR_MODE=rgbd TUM_DATA_ROOT=/path/to/TUM ./run_tum_voxel.sh rgbd_dataset_freiburg1_desk
```

ScanNet inputs are prepared from a `.sens` stream on first use:

```bash
SCANNET_ROOT=/path/to/ScanNet ./run_scannet_voxel.sh scene0000_00
SCANNET_SENSOR_MODE=rgbd SCANNET_ROOT=/path/to/ScanNet ./run_scannet_voxel.sh scene0000_00
```

Run the live RealSense demo with:

```bash
./run_realsense_rgbd_voxel.sh
```

Set `VOXEL_VIEWER=0` for a headless dataset run, or pass `no_viewer` to the RealSense launcher. Results, trajectories, voxel models, and extracted meshes are written below `results/` unless an output environment variable overrides the location.

## Jetson

The Jetson deployment is under `docker/ubuntu-jammy-jetson`. Build the image, create the container with external dataset and result locations, then enter it:

```bash
docker/ubuntu-jammy-jetson/build_image.sh
PHOTOSLAM_DATA_ROOT=/path/to/datasets \
PHOTOSLAM_RESULTS_ROOT=/path/to/results \
docker/ubuntu-jammy-jetson/create_container.sh
docker/ubuntu-jammy-jetson/run_container.sh
```

The implementation retains Photo-SLAM and ORB-SLAM3 licensing terms. See `LICENSE`, `LICENSE.md`, and the dependency repositories for details.
