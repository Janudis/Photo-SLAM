#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "This image must be built on an x86_64 host." >&2
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    echo "Docker is unavailable to the current user." >&2
    exit 1
fi

base_image="${PHOTOSLAM_BASE_IMAGE:-nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04}"
final_image="${PHOTOSLAM_IMAGE:-dimitris-photoslam-server:cu128}"

echo "Base image: $base_image"
echo "Building isolated image: $final_image"
echo "No existing image or container will be removed."

DOCKER_BUILDKIT=1 docker build \
    --build-arg "BASE_IMAGE=$base_image" \
    --build-arg "OPENCV_BUILD_JOBS=${OPENCV_BUILD_JOBS:-4}" \
    --build-arg "TORCH_VERSION=${TORCH_VERSION:-2.7.1}" \
    --build-arg "TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.22.1}" \
    --label "photoslam.owner=dimitris" \
    --label "photoslam.project=server-evaluation" \
    --tag "$final_image" \
    "$script_dir"

echo "Built image: $final_image"
