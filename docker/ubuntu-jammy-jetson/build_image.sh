#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

if [[ "$(uname -m)" != "aarch64" || ! -f /etc/nv_tegra_release ]]; then
    echo "This image must be built natively on the Jetson (aarch64/L4T)." >&2
    exit 1
fi
if ! grep -q "R36" /etc/nv_tegra_release; then
    echo "Expected JetPack 6 / L4T R36; found:" >&2
    head -n 1 /etc/nv_tegra_release >&2
    exit 1
fi

jetson_containers_dir="${JETSON_CONTAINERS_DIR:-}"
if [[ -z "$jetson_containers_dir" ]]; then
    if [[ -x "$repo_root/third_party/jetson-containers/jetson-containers" ]]; then
        jetson_containers_dir="$repo_root/third_party/jetson-containers"
    elif [[ -x "$HOME/jetson-containers/jetson-containers" ]]; then
        jetson_containers_dir="$HOME/jetson-containers"
    else
        echo "jetson-containers was not found." >&2
        echo "Set JETSON_CONTAINERS_DIR or clone it to $HOME/jetson-containers." >&2
        exit 1
    fi
fi

jetson_containers="$jetson_containers_dir/jetson-containers"
if [[ ! -x "$jetson_containers" ]]; then
    echo "Missing executable: $jetson_containers" >&2
    exit 1
fi

cuda_version="${CUDA_VERSION:-12.6}"
pytorch_version="${PYTORCH_VERSION:-2.3.1}"
pytorch_package="${PYTORCH_PACKAGE:-pytorch:2.3.1}"
opencv_package="${OPENCV_PACKAGE:-opencv:4.8.1}"
base_image="${PHOTOSLAM_BASE_IMAGE:-photoslam-svrecon-base:r36.4.7-cu126-torch231}"
final_image="${PHOTOSLAM_IMAGE:-photoslam-svrecon:r36.4.7}"

echo "Building Jetson base image: $base_image"
CUDA_VERSION="$cuda_version" \
PYTORCH_VERSION="$pytorch_version" \
PYTHON_VERSION=3.10 \
    "$jetson_containers" build \
        --name="$base_image" \
        "$pytorch_package" \
        "$opencv_package"

docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then
    docker_cmd=(sudo docker)
fi

echo "Building Photo-SLAM development image: $final_image"
DOCKER_BUILDKIT=1 "${docker_cmd[@]}" build \
    --network=host \
    --build-arg "BASE_IMAGE=$base_image" \
    --build-arg "USER_NAME=$USER" \
    --build-arg "USER_UID=$(id -u)" \
    --build-arg "USER_GID=$(id -g)" \
    --tag "$final_image" \
    "$script_dir"

echo "Built $final_image"
