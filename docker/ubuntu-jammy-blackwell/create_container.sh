#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_project_root="$(cd "$script_dir/../.." && pwd)"

image="${PHOTOSLAM_IMAGE:-dimitris-photoslam-server:cu128}"
container="${PHOTOSLAM_CONTAINER:-dimitris-photoslam-server}"
project_root="${PHOTOSLAM_PROJECT_ROOT:-$default_project_root}"
data_root="${PHOTOSLAM_DATA_ROOT:-}"
results_root="${PHOTOSLAM_RESULTS_ROOT:-}"

if ! docker info >/dev/null 2>&1; then
    echo "Docker is unavailable to the current user." >&2
    exit 1
fi
if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "Image does not exist: $image" >&2
    echo "Run build_image.sh first." >&2
    exit 1
fi
if [[ ! -f "$project_root/CMakeLists.txt" ]]; then
    echo "Photo-SLAM source directory is invalid: $project_root" >&2
    exit 1
fi
if [[ -n "$data_root" && ! -d "$data_root" ]]; then
    echo "Dataset directory does not exist: $data_root" >&2
    exit 1
fi

if docker container inspect "$container" >/dev/null 2>&1; then
    echo "Container already exists: $container" >&2
    echo "Use run_container.sh, or remove this exact container explicitly." >&2
    exit 1
fi

create_args=(
    create
    --name "$container"
    --label "photoslam.owner=dimitris"
    --label "photoslam.project=server-evaluation"
    --gpus all
    --shm-size 16g
    --ulimit memlock=-1
    --ulimit stack=67108864
    --user "$(id -u):$(id -g)"
    --env "HOME=/tmp"
    --env "USER=${USER:-v4rl}"
    --env NVIDIA_VISIBLE_DEVICES=all
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
    --env PHOTOSLAM_ENABLE_RERUN=OFF
    --env PHOTOSLAM_BUILD_ORIGINAL=ON
    --env PHOTOSLAM_BUILD_WAYMO=OFF
    --env PHOTOSLAM_BUILD_REALSENSE=OFF
    --env PHOTOSLAM_CUDA_ARCHITECTURES=120
    --env PHOTOSLAM_TORCH_CUDA_ARCH_LIST=12.0
    --env TORCH_CUDA_ARCH_LIST=12.0
    --env PYTHONNOUSERSITE=1
    --volume "$project_root:$project_root:rw"
    --workdir "$project_root"
)

if [[ -n "$data_root" ]]; then
    create_args+=(
        --env PHOTOSLAM_DATA_ROOT=/datasets
        --volume "$data_root:/datasets:ro"
    )
fi
if [[ -n "$results_root" ]]; then
    mkdir -p "$results_root"
    create_args+=(
        --env PHOTOSLAM_RESULTS_ROOT=/results
        --volume "$results_root:/results:rw"
    )
fi

docker "${create_args[@]}" "$image" sleep infinity

echo "Created container: $container"
echo "Project: $project_root"
if [[ -n "$data_root" ]]; then
    echo "Datasets: $data_root -> /datasets (read-only)"
fi
if [[ -n "$results_root" ]]; then
    echo "Results: $results_root -> /results"
fi
echo "Enter it with: $script_dir/run_container.sh"
