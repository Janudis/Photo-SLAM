#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_project_root="$(cd "$script_dir/../.." && pwd)"
container="${PHOTOSLAM_CONTAINER:-dimitris-photoslam-server}"
project_root="${PHOTOSLAM_PROJECT_ROOT:-$default_project_root}"

if ! docker info >/dev/null 2>&1; then
    echo "Docker is unavailable to the current user." >&2
    exit 1
fi
if ! docker container inspect "$container" >/dev/null 2>&1; then
    echo "Container does not exist: $container" >&2
    echo "Run create_container.sh first." >&2
    exit 1
fi

if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" != "true" ]]; then
    docker start "$container" >/dev/null
fi

exec_args=(exec -i)
if [[ -t 0 && -t 1 ]]; then
    exec_args+=(-t)
fi
exec_args+=(--workdir "$project_root")

if (( $# == 0 )); then
    command=(/bin/bash)
else
    command=("$@")
fi

exec docker "${exec_args[@]}" "$container" "${command[@]}"
