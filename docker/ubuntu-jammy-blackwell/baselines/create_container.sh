#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
        default_name="dimitris-monogs-benchmark"
        ;;
    hislam2)
        default_image="dimitris-hislam2-blackwell:cu128"
        default_name="dimitris-hislam2-benchmark"
        ;;
    tandem)
        default_image="dimitris-tandem-blackwell:cu128"
        default_name="dimitris-tandem-benchmark"
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
container="${BASELINE_CONTAINER:-$default_name}"
data_root="${PHOTOSLAM_DATA_ROOT:?Set PHOTOSLAM_DATA_ROOT to the host dataset directory}"
results_root="${PHOTOSLAM_RESULTS_ROOT:?Set PHOTOSLAM_RESULTS_ROOT to the host result directory}"

if [[ ! -d "$data_root" ]]; then
    echo "Dataset directory does not exist: $data_root" >&2
    exit 1
fi
mkdir -p "$results_root"

if "${docker_cmd[@]}" container inspect "$container" >/dev/null 2>&1; then
    echo "Container already exists: $container" >&2
    echo "Use docker start/exec, or choose another BASELINE_CONTAINER." >&2
    exit 1
fi

"${docker_cmd[@]}" create \
    --name "$container" \
    --gpus all \
    --ipc host \
    --network host \
    --shm-size 16g \
    --user "$(id -u):$(id -g)" \
    --env HOME=/tmp/baseline-home \
    --env PHOTOSLAM_DATA_ROOT=/datasets \
    --env PHOTOSLAM_RESULTS_ROOT=/results \
    --env WANDB_MODE=disabled \
    --env CUDA_MODULE_LOADING=LAZY \
    --volume "$data_root:/datasets:ro" \
    --volume "$results_root:/results" \
    --label "photoslam.owner=dimitris" \
    --label "photoslam.project=paper-baselines" \
    --label "photoslam.baseline=$method" \
    "$image" \
    sleep infinity

echo "Created container: $container"
echo "Datasets: $data_root -> /datasets (read-only)"
echo "Results:  $results_root -> /results"
echo "Enter with: $script_dir/run_container.sh $method"
