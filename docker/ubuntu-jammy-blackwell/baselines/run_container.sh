#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 {monogs|hislam2|tandem} [command ...]" >&2
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

method="$1"
shift

case "$method" in
    monogs)
        default_name="dimitris-monogs-benchmark-r2"
        workdir="/opt/MonoGS"
        ;;
    hislam2)
        default_name="dimitris-hislam2-benchmark"
        workdir="/opt/HI-SLAM2"
        ;;
    tandem)
        default_name="dimitris-tandem-benchmark"
        workdir="/opt/tandem/tandem"
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

container="${BASELINE_CONTAINER:-$default_name}"
"${docker_cmd[@]}" start "$container" >/dev/null

if [[ $# -eq 0 ]]; then
    exec "${docker_cmd[@]}" exec -it --workdir "$workdir" "$container" /bin/bash
fi

terminal_args=()
if [[ -t 0 && -t 1 ]]; then
    terminal_args=(-it)
fi

exec "${docker_cmd[@]}" exec "${terminal_args[@]}" \
    --workdir "$workdir" "$container" "$@"
