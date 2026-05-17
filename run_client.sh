#!/usr/bin/env bash
# Run the dockerised SO-101 inference client. Forwards every USB device the
# client needs (arm + 3 cameras) and uses host networking so it can reach the
# policy server on ws://localhost:8000.
#
# Usage:
#   ./run_client.sh --list-devices
#   ./run_client.sh --arm-port /dev/ttyACM0 --cam-front 8 --cam-top 10 --cam-wrist 12 \
#                   --calibration /app/calibration.json
#
# Env overrides:
#   ARM_PORT  CAM_FRONT  CAM_TOP  CAM_WRIST   used to build --device flags
#   TAG       docker image tag (default openpi-client:so101)
set -euo pipefail
cd "$(dirname "$0")"

TAG="${TAG:-openpi-client:so101}"
NAME="${NAME:-openpi-client-so101}"

ARM_PORT="${ARM_PORT:-/dev/ttyACM0}"
CAM_FRONT="${CAM_FRONT:-8}"
CAM_TOP="${CAM_TOP:-10}"
CAM_WRIST="${CAM_WRIST:-12}"

# Special "discover devices" mode: pass /dev through and skip per-device flags.
if [[ "${1:-}" == "--list-devices" ]]; then
  exec docker run --rm --network host -v /dev:/dev --privileged "$TAG" --list-devices
fi


# Override defaults from explicit --cam-*/--arm-port flags so device
# passthrough matches whatever we send to the python script.
prev=""
for tok in "$@"; do
  case "$prev" in
    --arm-port)  ARM_PORT="$tok" ;;
    --cam-front) CAM_FRONT="$tok" ;;
    --cam-top)   CAM_TOP="$tok" ;;
    --cam-wrist) CAM_WRIST="$tok" ;;
  esac
  prev="$tok"
done

# Build --device flags for whichever USB nodes actually exist on the host.
DEV_FLAGS=()
for d in "$ARM_PORT" "/dev/video${CAM_FRONT}" "/dev/video${CAM_TOP}" "/dev/video${CAM_WRIST}"; do
  if [[ -e "$d" ]]; then
    DEV_FLAGS+=( "--device=$d" )
  else
    echo "WARN: $d not present on host (skipping)" >&2
  fi
done

# Bind-mount the calibration file if present.
MOUNTS=()
if [[ -f run_so101.py ]]; then
  MOUNTS+=( -v "$(pwd)/run_so101.py:/app/run_so101.py:ro" )
fi
if [[ -f calibration.json ]]; then
  MOUNTS+=( -v "$(pwd)/calibration.json:/app/calibration.json:ro" )
else
  echo "WARN: ./calibration.json not found - the model expects degrees but" >&2
  echo "      without calibration the client will send raw ticks. See README." >&2
fi

exec docker run --rm -i --network host -v /tmp/inference_diag:/tmp/inference_diag \
  --name "$NAME" \
  "${DEV_FLAGS[@]}" \
  "${MOUNTS[@]}" \
  "$TAG" "$@"
