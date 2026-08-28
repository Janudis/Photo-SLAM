#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"

usage() {
    echo "Usage: $0 {monogs|hislam2|tandem}" >&2
}

if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi

method="$1"
case "$method" in
    monogs)
        default_image="dimitris-monogs-blackwell:cu128"
        commit="6c9254c319d8bff5caeef65259e6bb0941a9b9f6"
        ;;
    hislam2)
        default_image="dimitris-hislam2-blackwell:cu128"
        commit="76c833c7d8ed474f0f3ba18056c1803e032a537f"
        ;;
    tandem)
        default_image="dimitris-tandem-blackwell:cu128"
        commit="f8816c7d9a92b29e84e3d9055c2d3e28056e4a37"
        ;;
    *)
        usage
        exit 2
        ;;
esac

if docker info >/dev/null 2>&1; then
    docker_cmd=(docker)
elif command -v sudo >/dev/null 2>&1; then
    docker_cmd=(sudo docker)
else
    echo "Docker is unavailable and sudo is not installed." >&2
    exit 1
fi

image="${BASELINE_IMAGE:-$default_image}"
base_image="${BASELINE_BASE_IMAGE:-nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04}"

echo "Method: $method"
echo "Source commit: $commit"
echo "Base image: $base_image"
echo "Output image: $image"

if "${docker_cmd[@]}" image inspect "$image" >/dev/null 2>&1; then
    echo "Image already exists: $image" >&2
    echo "Choose another BASELINE_IMAGE; this script will not replace it." >&2
    exit 1
fi

DOCKER_BUILDKIT=1 "${docker_cmd[@]}" build \
    --build-arg "BASE_IMAGE=$base_image" \
    --build-arg "SOURCE_COMMIT=$commit" \
    --build-arg "MAX_JOBS=${MAX_JOBS:-4}" \
    --label "photoslam.owner=dimitris" \
    --label "photoslam.project=paper-baselines" \
    --label "photoslam.baseline=$method" \
    --tag "$image" \
    --file "$script_dir/${method}.Dockerfile" \
    "$repo_root"

echo "Built image: $image"
