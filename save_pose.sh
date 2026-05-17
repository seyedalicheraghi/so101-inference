#!/usr/bin/env bash
# Read current SO-101 pose and store it as the default target for run_home.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM_PORT="${ARM_PORT:-/dev/ttyACM0}"

if [[ ! -f "${HERE}/calibration.json" ]]; then
    echo "ERROR: calibration.json not found - run ./run_calibrate.sh first." >&2
    exit 1
fi

POSE=$(docker run --rm \
    --device="${ARM_PORT}" \
    -v "${HERE}/calibration.json:/app/calibration.json:ro" \
    --entrypoint python \
    openpi-client:so101 \
    /app/save_pose.py --port "${ARM_PORT}" --calibration /app/calibration.json)

echo "${POSE}" > "${HERE}/home_pose.txt"
echo
echo "==> Saved pose: ${POSE}"
echo "    Persisted to: ${HERE}/home_pose.txt"
echo "    Use it: ./run_home.sh         (defaults to this pose now)"
