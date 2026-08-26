#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$(uname -m)" != "aarch64" || ! -f /etc/nv_tegra_release ]]; then
    echo "This image must be built natively on the Jetson (aarch64/L4T)." >&2
    exit 1
fi
if ! grep -q "R36" /etc/nv_tegra_release; then
    echo "Expected JetPack 6 / L4T R36; found:" >&2
    head -n 1 /etc/nv_tegra_release >&2
    exit 1
fi

base_image="${PHOTOSLAM_BASE_IMAGE:-ubuntu-jammy-humble-jetson:latest}"
final_image="${PHOTOSLAM_IMAGE:-photoslam-svrecon:r36.4.7}"

docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then
    docker_cmd=(sudo docker)
fi

if ! "${docker_cmd[@]}" image inspect "$base_image" >/dev/null 2>&1; then
    echo "Required local base image was not found: $base_image" >&2
    echo "This deployment derives from the existing Jetson image without modifying it." >&2
    echo "Set PHOTOSLAM_BASE_IMAGE to another local CUDA/PyTorch image if needed." >&2
    exit 1
fi

base_arch="$("${docker_cmd[@]}" image inspect \
    --format '{{.Architecture}}' "$base_image")"
if [[ "$base_arch" != "arm64" ]]; then
    echo "Expected an arm64 base image; $base_image is $base_arch." >&2
    exit 1
fi

echo "Base image (read-only): $base_image"
echo "Building new Photo-SLAM image: $final_image"
DOCKER_BUILDKIT=1 "${docker_cmd[@]}" build \
    --network=host \
    --build-arg "BASE_IMAGE=$base_image" \
    --build-arg "OPENCV_BUILD_JOBS=${OPENCV_BUILD_JOBS:-4}" \
    --build-arg "LIBREALSENSE_VERSION=${LIBREALSENSE_VERSION:-2.57.6}" \
    --build-arg "LIBREALSENSE_BUILD_JOBS=${LIBREALSENSE_BUILD_JOBS:-4}" \
    --tag "$final_image" \
    "$script_dir"

echo "Built $final_image"
