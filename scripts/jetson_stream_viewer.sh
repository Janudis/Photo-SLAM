#!/usr/bin/env bash

set -euo pipefail

container="${PHOTOSLAM_CONTAINER:-photoslam-svrecon-realsense}"
display="${PHOTOSLAM_DISPLAY:-:0}"
xauthority="${PHOTOSLAM_XAUTHORITY:-$HOME/.Xauthority}"
vnc_port="${PHOTOSLAM_VNC_PORT:-5900}"
wait_seconds="${PHOTOSLAM_VIEWER_WAIT_SECONDS:-180}"
vnc_log="${PHOTOSLAM_VNC_LOG:-/tmp/photoslam-x11vnc.log}"

if ! docker container inspect "$container" >/dev/null 2>&1; then
    echo "Container does not exist: $container" >&2
    exit 1
fi

if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" != "true" ]]; then
    docker start "$container" >/dev/null
fi

DISPLAY="$display" XAUTHORITY="$xauthority" xrandr --fb 1920x1080

deadline=$((SECONDS + wait_seconds))
window_id=""
while (( SECONDS < deadline )); do
    window_id="$({
        DISPLAY="$display" XAUTHORITY="$xauthority" \
            xwininfo -root -tree 2>/dev/null || true
    } | awk '/"Photo-SLAM SVRecon"/ {print $1; exit}')"
    if [[ -n "$window_id" ]]; then
        break
    fi
    sleep 1
done

if [[ -z "$window_id" ]]; then
    echo "Timed out waiting for the Photo-SLAM viewer on $display." >&2
    exit 1
fi

docker exec "$container" pkill -x x11vnc >/dev/null 2>&1 || true
for _ in {1..20}; do
    if ! docker exec "$container" pgrep -x x11vnc >/dev/null 2>&1; then
        break
    fi
    sleep 0.1
done

if ss -H -ltn | awk -v endpoint=":$vnc_port" \
    '$4 ~ (endpoint "$") { found = 1 } END { exit !found }'; then
    echo "Cannot start the viewer stream: localhost:$vnc_port is already in use on the Jetson." >&2
    echo "Run the SSH -L forwarding command in a local laptop terminal, not in a Jetson SSH shell." >&2
    ss -ltnp | awk -v endpoint=":$vnc_port" '$4 ~ (endpoint "$")' >&2 || true
    exit 1
fi

docker exec "$container" rm -f "$vnc_log"
docker exec -d \
    -e "DISPLAY=$display" \
    -e "XAUTHORITY=$xauthority" \
    "$container" \
    x11vnc \
        -display "$display" \
        -auth "$xauthority" \
        -id "$window_id" \
        -localhost \
        -forever \
        -shared \
        -nopw \
        -repeat \
        -noxdamage \
        -rfbport "$vnc_port" \
        -o "$vnc_log"

stream_ready=false
for _ in {1..50}; do
    if docker exec "$container" pgrep -x x11vnc >/dev/null 2>&1 && \
       ss -H -ltn | awk -v endpoint=":$vnc_port" \
           '$4 ~ (endpoint "$") { found = 1 } END { exit !found }'; then
        stream_ready=true
        break
    fi
    sleep 0.1
done

if [[ "$stream_ready" != true ]]; then
    echo "x11vnc failed to stream the Photo-SLAM viewer." >&2
    docker exec "$container" cat "$vnc_log" >&2 2>/dev/null || true
    exit 1
fi

echo "Streaming Photo-SLAM viewer window $window_id on localhost:$vnc_port"
echo "The viewer stream is running in the background; returning to the shell is expected."
echo "Keep the experiment and the laptop SSH tunnel running."
