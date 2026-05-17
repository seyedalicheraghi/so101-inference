#!/usr/bin/env bash
# One-command SO-101 follower-arm calibration. Saves calibration.json
# next to this script (on the host); run_client.sh will mount it back in.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM_PORT="${ARM_PORT:-/dev/ttyACM0}"
OUT="${OUT:-${HERE}/calibration.json}"
ID="${ID:-my_follower_arm}"

if [[ ! -e "${ARM_PORT}" ]]; then
    echo "ERROR: arm port ${ARM_PORT} does not exist. Plug in the SO-101." >&2
    echo "Available serial devices:" >&2
    ls /dev/serial/by-id/ 2>/dev/null || ls /dev/ttyACM* 2>/dev/null || true
    exit 1
fi

# Mount the host directory at /app so the JSON lands directly on the host.
docker run --rm -it \
    --device="${ARM_PORT}" \
    -v "${HERE}:/app/out" \
    --entrypoint python \
    openpi-client:so101 \
    /app/calibrate.py \
        --port "${ARM_PORT}" \
        --id   "${ID}" \
        --out  "/app/out/calibration.json"

echo
echo "==> Calibration saved to ${OUT}"
echo "    Use it for inference:"
echo "    ./run_client.sh --calibration /app/calibration.json ..."
