#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_project_root="$(cd "$script_dir/../.." && pwd)"

image="${PHOTOSLAM_IMAGE:-photoslam-svrecon:r36.4.7}"
container="${PHOTOSLAM_CONTAINER:-photoslam-svrecon}"
project_root="${PHOTOSLAM_PROJECT_ROOT:-$default_project_root}"
data_root="${PHOTOSLAM_DATA_ROOT:-}"
results_root="${PHOTOSLAM_RESULTS_ROOT:-$HOME/Photo-SLAM-results}"

if [[ -z "$data_root" ]]; then
    echo "PHOTOSLAM_DATA_ROOT is required." >&2
    echo "Point it to the external-SSD or NFS directory whose contents match scripts/data/." >&2
    exit 1
fi
if [[ ! -d "$data_root" ]]; then
    echo "Dataset directory does not exist: $data_root" >&2
    exit 1
fi
if [[ ! -d "$project_root" || ! -f "$project_root/CMakeLists.txt" ]]; then
    echo "Photo-SLAM source directory is invalid: $project_root" >&2
    exit 1
fi

mkdir -p "$results_root"

docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then
    docker_cmd=(sudo docker)
fi

if "${docker_cmd[@]}" container inspect "$container" >/dev/null 2>&1; then
    echo "Container already exists: $container"
    echo "Use run_container.sh to enter it. Remove it explicitly before recreating it."
    exit 0
fi

create_args=(
    create
    --name "$container"
    --runtime nvidia
    --network host
    --ipc host
    --ulimit memlock=-1
    --ulimit stack=67108864
    --env "DISPLAY=${DISPLAY:-:0}"
    --env NVIDIA_VISIBLE_DEVICES=all
    --env NVIDIA_DRIVER_CAPABILITIES=all
    --env PHOTOSLAM_JETSON=1
    --env PHOTOSLAM_ENABLE_RERUN=OFF
    --env PHOTOSLAM_BUILD_ORIGINAL=OFF
    --env PHOTOSLAM_BUILD_WAYMO=OFF
    --env PHOTOSLAM_CUDA_ARCHITECTURES=87
    --env PHOTOSLAM_TORCH_CUDA_ARCH_LIST=8.7
    --env TORCH_CUDA_ARCH_LIST=8.7
    --env PHOTOSLAM_DATA_ROOT=/datasets
    --env PHOTOSLAM_RESULTS_ROOT=/results
    --env "PHOTOSLAM_PROJECT_ROOT=$project_root"
    --env PYTHONNOUSERSITE=1
    --volume "$HOME:$HOME:rw"
    --volume "$data_root:/datasets:ro"
    --volume "$data_root:$project_root/scripts/data:ro"
    --volume "$results_root:/results:rw"
    --volume "$results_root:$project_root/results:rw"
    --workdir "$project_root"
)

if [[ "$project_root" != "$HOME" && "$project_root" != "$HOME/"* ]]; then
    create_args+=(--volume "$project_root:$project_root:rw")
fi
if [[ -d /tmp/.X11-unix ]]; then
    create_args+=(--volume /tmp/.X11-unix:/tmp/.X11-unix:rw)
fi

xauthority="${XAUTHORITY:-$HOME/.Xauthority}"
if [[ -f "$xauthority" ]]; then
    create_args+=(--env "XAUTHORITY=$xauthority")
else
    echo "Viewer note: no Xauthority file was found at $xauthority."
    echo "Headless runs are unaffected; the desktop session may need to grant X access."
fi

for device_group in video render; do
    if getent group "$device_group" >/dev/null; then
        create_args+=(--group-add "$(getent group "$device_group" | cut -d: -f3)")
    fi
done

"${docker_cmd[@]}" "${create_args[@]}" "$image" sleep infinity

echo "Created container: $container"
echo "Datasets: $data_root -> /datasets (read-only)"
echo "Results:  $results_root -> /results"
echo "Enter it with: $script_dir/run_container.sh"
