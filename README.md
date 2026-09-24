# SVR-SLAM Demo

## Dependencies

- CUDA and cuDNN
- LibTorch with CUDA
- OpenCV with CUDA modules
- Eigen3, Boost, jsoncpp, OpenGL, GLFW, GLM, glog, pybind11
- Intel RealSense SDK for the live RGB-D runner
- Python Rerun SDK when `PHOTOSLAM_ENABLE_RERUN=ON`

## Build

```bash
./build.sh
```

Set `LIBTORCH_ROOT` if LibTorch is not installed in a standard CMake search path. The build script accepts `PHOTOSLAM_BUILD_JOBS`, `PHOTOSLAM_CUDA_ARCHITECTURES`, `PHOTOSLAM_ENABLE_RERUN`, and `PHOTOSLAM_BUILD_REALSENSE`.

## Demos

For monocular runs, set `Mapper.monocular_mvs_tsdf_evidence` to `0` for RGB or `1` for RGB+MVS in the corresponding mapper configuration.

### Replica

```bash
# RGB
REPLICA_SENSOR_MODE=monocular REPLICA_ROOT=/path/to/Replica ./scripts/run_replica_voxel.sh office0

# RGB + MVS
REPLICA_SENSOR_MODE=monocular REPLICA_ROOT=/path/to/Replica ./scripts/run_replica_voxel.sh office0

# RGB-D
REPLICA_SENSOR_MODE=rgbd REPLICA_ROOT=/path/to/Replica ./scripts/run_replica_voxel.sh office0
```

### TUM RGB-D

```bash
# RGB
TUM_SENSOR_MODE=monocular TUM_DATA_ROOT=/path/to/TUM ./scripts/run_tum_voxel.sh rgbd_dataset_freiburg1_desk

# RGB + MVS
TUM_SENSOR_MODE=monocular TUM_DATA_ROOT=/path/to/TUM ./scripts/run_tum_voxel.sh rgbd_dataset_freiburg1_desk

# RGB-D
TUM_SENSOR_MODE=rgbd TUM_DATA_ROOT=/path/to/TUM ./scripts/run_tum_voxel.sh rgbd_dataset_freiburg1_desk
```

### ScanNet

ScanNet inputs are prepared from a `.sens` stream on first use.

```bash
# RGB
SCANNET_SENSOR_MODE=monocular SCANNET_ROOT=/path/to/ScanNet ./scripts/run_scannet_voxel.sh scene0000_00

# RGB + MVS
SCANNET_SENSOR_MODE=monocular SCANNET_ROOT=/path/to/ScanNet ./scripts/run_scannet_voxel.sh scene0000_00

# RGB-D
SCANNET_SENSOR_MODE=rgbd SCANNET_ROOT=/path/to/ScanNet ./scripts/run_scannet_voxel.sh scene0000_00
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

- RGB: run a monocular dataset command above with MVS disabled.
- RGB + MVS: run a monocular dataset command above with MVS enabled.
- RGB-D with Intel RealSense:

```bash
./scripts/run_realsense_rgbd_voxel.sh
```

The implementation retains Photo-SLAM and ORB-SLAM3 licensing terms. See `LICENSE`, `LICENSE.md`, and the dependency repositories for details.
