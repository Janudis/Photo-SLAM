#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_project_root="$(cd "$script_dir/../.." && pwd)"
container="${PHOTOSLAM_CONTAINER:-photoslam-svrecon}"
project_root="${PHOTOSLAM_PROJECT_ROOT:-$default_project_root}"

docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then
    docker_cmd=(sudo docker)
fi

if ! "${docker_cmd[@]}" container inspect "$container" >/dev/null 2>&1; then
    echo "Container does not exist: $container" >&2
    echo "Run create_container.sh first." >&2
    exit 1
fi

if [[ "$("${docker_cmd[@]}" inspect -f '{{.State.Running}}' "$container")" != "true" ]]; then
    "${docker_cmd[@]}" start "$container" >/dev/null
fi

workdir="$project_root"
if [[ "$PWD" == "$HOME" || "$PWD" == "$HOME/"* ]]; then
    workdir="$PWD"
fi

exec_args=(exec -i)
if [[ -t 0 && -t 1 ]]; then
    exec_args+=(-t)
fi
exec_args+=(
    --env "DISPLAY=${DISPLAY:-:0}"
    --workdir "$workdir"
)
if [[ -n "${XAUTHORITY:-}" ]]; then
    exec_args+=(--env "XAUTHORITY=$XAUTHORITY")
fi

if (( $# == 0 )); then
    command=(/bin/bash)
else
    command=("$@")
fi

exec "${docker_cmd[@]}" "${exec_args[@]}" "$container" "${command[@]}"
