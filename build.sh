#!/usr/bin/env bash

set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
is_jetson=0
if [[ "$(uname -m)" == "aarch64" && -f /etc/nv_tegra_release ]]; then
    is_jetson=1
fi

if (( is_jetson )); then
    jobs="${PHOTOSLAM_BUILD_JOBS:-4}"
    cuda_architectures="${PHOTOSLAM_CUDA_ARCHITECTURES:-87}"
    torch_cuda_arch_list="${PHOTOSLAM_TORCH_CUDA_ARCH_LIST:-8.7}"
    enable_rerun="${PHOTOSLAM_ENABLE_RERUN:-OFF}"
    build_original="${PHOTOSLAM_BUILD_ORIGINAL:-OFF}"
    build_waymo="${PHOTOSLAM_BUILD_WAYMO:-OFF}"
else
    jobs="${PHOTOSLAM_BUILD_JOBS:-8}"
    cuda_architectures="${PHOTOSLAM_CUDA_ARCHITECTURES:-75;86}"
    torch_cuda_arch_list="${PHOTOSLAM_TORCH_CUDA_ARCH_LIST:-8.9}"
    enable_rerun="${PHOTOSLAM_ENABLE_RERUN:-ON}"
    build_original="${PHOTOSLAM_BUILD_ORIGINAL:-ON}"
    build_waymo="${PHOTOSLAM_BUILD_WAYMO:-ON}"
fi

# Preserve the existing desktop installation while allowing containers to use
# PyTorch's own CMake package without a machine-specific path.
if [[ -z "${LIBTORCH_ROOT:-}" && \
      -d "$HOME/opt/libtorch_2.0.1_cu118/libtorch" ]]; then
    export LIBTORCH_ROOT="$HOME/opt/libtorch_2.0.1_cu118/libtorch"
fi

build_cmake_project() {
    local source_dir="$1"
    local build_dir="$2"
    shift 2
    cmake -S "$source_dir" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE=Release "$@"
    cmake --build "$build_dir" --parallel "$jobs"
}

echo "Building ORB-SLAM3 dependencies ..."
build_cmake_project \
    "$root_dir/ORB-SLAM3/Thirdparty/DBoW2" \
    "$root_dir/ORB-SLAM3/Thirdparty/DBoW2/build"
build_cmake_project \
    "$root_dir/ORB-SLAM3/Thirdparty/g2o" \
    "$root_dir/ORB-SLAM3/Thirdparty/g2o/build"
build_cmake_project \
    "$root_dir/ORB-SLAM3/Thirdparty/Sophus" \
    "$root_dir/ORB-SLAM3/Thirdparty/Sophus/build"

if [[ ! -f "$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt" ]]; then
    echo "Uncompressing ORB vocabulary ..."
    tar -xf "$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt.tar.gz" \
        -C "$root_dir/ORB-SLAM3/Vocabulary"
fi

echo "Building ORB-SLAM3 ..."
build_cmake_project "$root_dir/ORB-SLAM3" "$root_dir/ORB-SLAM3/build"

echo "Building Photo-SLAM SVRecon targets ..."
photoslam_cmake_args=(
    "-DPHOTOSLAM_CUDA_ARCHITECTURES=$cuda_architectures"
    "-DPHOTOSLAM_ENABLE_RERUN=$enable_rerun"
    "-DPHOTOSLAM_BUILD_ORIGINAL_PHOTOSLAM=$build_original"
    "-DPHOTOSLAM_BUILD_WAYMO=$build_waymo"
)
if (( ! is_jetson )) && [[ -x /usr/bin/python3.10 ]]; then
    photoslam_cmake_args+=("-DPython3_EXECUTABLE=/usr/bin/python3.10")
fi
if [[ -n "$torch_cuda_arch_list" ]]; then
    photoslam_cmake_args+=("-DTORCH_CUDA_ARCH_LIST=$torch_cuda_arch_list")
fi
if [[ -n "${LIBTORCH_ROOT:-}" ]]; then
    photoslam_cmake_args+=("-DLIBTORCH_ROOT=$LIBTORCH_ROOT")
fi

build_cmake_project \
    "$root_dir" \
    "$root_dir/build" \
    "${photoslam_cmake_args[@]}"
