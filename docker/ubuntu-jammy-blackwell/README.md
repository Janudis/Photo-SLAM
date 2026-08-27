# Photo-SLAM x86/Blackwell container

This container builds the SVRecon and original Photo-SLAM targets on an
x86_64 NVIDIA Blackwell host. It is separate from the Jetson deployment and
uses uniquely prefixed Docker resources.

## Build the dependency image

```bash
cd docker/ubuntu-jammy-blackwell
./build_image.sh
```

The default image is `dimitris-photoslam-server:cu128`. The build uses CUDA
12.8, PyTorch 2.7.1, and OpenCV CUDA compiled for compute capability 12.0.

## Create and enter the development container

No dataset or results directory is required for compilation:

```bash
./create_container.sh
./run_container.sh
```

Inside the container, build the project with:

```bash
PHOTOSLAM_BUILD_JOBS=4 ./build.sh
```

The source tree is bind-mounted at its host path, so generated binaries remain
in the server checkout.

## Optional dataset and results mounts

Set these only after the binaries pass their build and CUDA smoke tests:

```bash
export PHOTOSLAM_DATA_ROOT=/path/to/datasets
export PHOTOSLAM_RESULTS_ROOT=/path/to/results
./create_container.sh
```

The dataset is mounted read-only at `/datasets`; results are mounted read-write
at `/results`.

## Resource isolation

This deployment creates only:

- image `dimitris-photoslam-server:cu128`;
- container `dimitris-photoslam-server`;
- resources labeled `photoslam.owner=dimitris` and
  `photoslam.project=server-evaluation`.

Do not use global Docker cleanup commands on the shared server. Remove only
the exact resources above when they are no longer needed.
