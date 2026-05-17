#!/usr/bin/env bash
# Drive the SO-101 follower to a saved pose (or pass --target explicitly).
#
# Default target comes from ./home_pose.txt (created by ./save_pose.sh).
# If that file is missing, defaults to the calibrated centre (all zeros).
#
# Usage:
#   ./run_home.sh                          # go to saved pose / centre
#   ./run_home.sh --keep-torque            # ... and hold there
#   ./run_home.sh --target 0,0,0,0,0,0     # one-off override
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM_PORT="${ARM_PORT:-/dev/ttyACM0}"

if [[ ! -e "${ARM_PORT}" ]]; then
    echo "ERROR: arm port ${ARM_PORT} does not exist." >&2
    exit 1
fi
if [[ ! -f "${HERE}/calibration.json" ]]; then
    echo "ERROR: calibration.json not found - run ./run_calibrate.sh first." >&2
    exit 1
fi

# Inject --target from home_pose.txt unless caller already passed --target.
EXTRA_ARGS=()
if ! printf "%s\n" "$@" | grep -q "^--target$"; then
    if [[ -f "${HERE}/home_pose.txt" ]]; then
        SAVED=$(cat "${HERE}/home_pose.txt")
        EXTRA_ARGS+=( --target "${SAVED}" )
        echo "Using saved home pose: ${SAVED}"
    else
        echo "No home_pose.txt - defaulting to calibrated centre (all zeros)"
    fi
fi

docker run --rm -it \
    --device="${ARM_PORT}" \
    -v "${HERE}/calibration.json:/app/calibration.json:ro" \
    --entrypoint python \
    openpi-client:so101 \
    /app/home.py \
        --port "${ARM_PORT}" \
        --calibration /app/calibration.json \
        "${EXTRA_ARGS[@]}" \
        "$@"
