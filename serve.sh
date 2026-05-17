#!/usr/bin/env bash
# Start the openpi policy server for the trained SO-101 LoRA checkpoint.
#
# Listens on ws://0.0.0.0:8000 and sends back action chunks for every observation.
# Re-uses the same docker image that trained the model (openpi-train:so101) so
# all Thor-specific patches (orbax pin, JAX 0.9 sharding fix, PyAV video decode)
# are present at inference time too.
#
# Usage:
#   ./serve.sh                  # default checkpoint, default prompt
#   ./serve.sh 5000             # use step-5000 checkpoint instead
#   PROMPT="..." ./serve.sh     # override prompt
set -euo pipefail

STEP="${1:-7999}"
CKPT_HOST="$HOME/src/openpi/checkpoints/pi05_so101_low_mem_finetune/so101_20260511_1638/${STEP}"
PROMPT="${PROMPT:-Pick up the white box and place it in the white target area.}"
PORT="${PORT:-8000}"
IMAGE="${IMAGE:-openpi-train:so101}"
NAME="${NAME:-openpi-serve-so101}"

if [[ ! -d "$CKPT_HOST" ]]; then
  echo "ERROR: checkpoint not found: $CKPT_HOST" >&2
  ls "$HOME/src/openpi/checkpoints/pi05_so101_low_mem_finetune/so101_20260511_1638/" 2>/dev/null || true
  exit 1
fi

# Stop any previous server.
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "Starting policy server:"
echo "  checkpoint : $CKPT_HOST"
echo "  prompt     : $PROMPT"
echo "  listening  : ws://0.0.0.0:${PORT}"
echo

exec docker run --runtime=nvidia --network host --name "$NAME" --rm \
  -v "$HOME/src/openpi:/opt/openpi-train" \
  -v "$HOME/src/dataset:/data/models/huggingface/lerobot/alicheraghi/robot-arm" \
  -v "$HOME/.cache/openpi:/root/.cache/openpi" \
  -w /opt/openpi-train \
  -e PYTHONUNBUFFERED=1 \
  "$IMAGE" \
  bash -lc "uv pip install --no-deps -e . >/dev/null && \
            python scripts/serve_policy.py \
              --port ${PORT} \
              --default-prompt '${PROMPT//\'/\'\\\'\'}' \
              policy:checkpoint \
              --policy.config=pi05_so101_low_mem_finetune \
              --policy.dir=/opt/openpi-train/checkpoints/pi05_so101_low_mem_finetune/so101_20260511_1638/${STEP}"
