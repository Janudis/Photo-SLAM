#!/usr/bin/env bash

set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
replica_root="${REPLICA_ROOT:-/media/dimitris/v4rl_rog_2t/Dimitris/Datasets_indoor/Replica}"
results_root="${REPLICA_RESULTS_ROOT:-$root_dir/results/replica_rgb_original}"
sequence_list="${REPLICA_SEQUENCES:-office0 office1 office2 office3 office4 room0 room1 room2}"

read -r -a sequences <<< "$sequence_list"

binary="$root_dir/bin/replica_mono"
vocabulary="$root_dir/ORB-SLAM3/Vocabulary/ORBvoc.txt"
mapper_config="$root_dir/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml"

for required in "$binary" "$vocabulary" "$mapper_config"; do
    if [[ ! -e "$required" ]]; then
        printf '[ERROR] Required Photo-SLAM file is missing: %s\n' "$required" >&2
        exit 1
    fi
done

export LD_LIBRARY_PATH="/home/dimitris/opt/libtorch_2.0.1_cu118/libtorch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$root_dir/third_party/simple-knn:${PYTHONPATH:-}"

viewer_args=(no_viewer)
if [[ "${REPLICA_VIEWER:-0}" == "1" ]]; then
    viewer_args=()
fi

for sequence in "${sequences[@]}"; do
        case "$sequence" in
            office0|office1|office2|office3|office4|room0|room1|room2)
                ;;
            *)
                printf '[ERROR] Unsupported Replica sequence: %s\n' "$sequence" >&2
                exit 2
                ;;
        esac

        sequence_dir="$replica_root/$sequence"
        orb_config="$root_dir/cfg/ORB_SLAM3/Monocular/Replica/$sequence.yaml"
        output_dir="$results_root/$sequence"

        if [[ ! -d "$sequence_dir/results" ]]; then
            printf '[ERROR] Replica RGB frames are missing: %s/results\n' \
                "$sequence_dir" >&2
            exit 1
        fi
        if [[ ! -f "$orb_config" ]]; then
            printf '[ERROR] ORB-SLAM3 configuration is missing: %s\n' \
                "$orb_config" >&2
            exit 1
        fi

        printf '\n=== Original Photo-SLAM | Replica/%s ===\n' "$sequence"
        printf 'Mapper config: %s\n' "$mapper_config"
        printf 'Output:        %s\n' "$output_dir"

        completed_mesh="$(find "$output_dir" -type f \
            -name 'gaussian_surface_mesh.ply' -print -quit 2>/dev/null || true)"
        if [[ -n "$completed_mesh" ]]; then
            printf 'Skipping completed result with surface mesh: %s\n' \
                "$completed_mesh"
            continue
        fi

        if [[ -d "$output_dir" ]] &&
            [[ -n "$(find "$output_dir" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
            printf '[ERROR] Existing result is incomplete or has no surface mesh: %s\n' \
                "$output_dir" >&2
            printf 'Move it to a backup location before rerunning this sequence.\n' >&2
            exit 1
        fi

        if [[ "${REPLICA_DRY_RUN:-0}" == "1" ]]; then
            continue
        fi

        mkdir -p "$output_dir"
        "$binary" \
            "$vocabulary" \
            "$orb_config" \
            "$mapper_config" \
            "$sequence_dir" \
            "$output_dir" \
            "${viewer_args[@]}"
done
