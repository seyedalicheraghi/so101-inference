#!/usr/bin/env bash
# Start a PyTorch policy server backed by the LeRobot loader for the SO-101
# step-015000 checkpoint. No TRT; same websocket protocol as serve.sh.
#
# Usage:
#   ./serve_lerobot.sh                     # uses default checkpoint
#   CKPT_PT=/some/dir ./serve_lerobot.sh   # override
#   LEROBOT_VERSION=0.4.4 ./serve_lerobot.sh
#
# The first run will pip-install a newer lerobot inside the container
# (~30-60s). Subsequent runs reuse the image-installed copy if it stuck.

set -euo pipefail

CKPT_PT="${CKPT_PT:-$HOME/src/openpi/checkpoints/pi05_so101_low_mem_finetune/so101_20260511_1638/015000/pretrained_model}"
LEROBOT_VERSION="${LEROBOT_VERSION:-0.5.1}"
PROMPT="${PROMPT:-Pick up the white box and place it in the white target area.}"
PORT="${PORT:-8000}"
IMAGE="${IMAGE:-openpi-pi0.5:latest}"
NAME="${NAME:-openpi-serve-so101-lerobot}"
SERVE_PY="${SERVE_PY:-$HOME/src/inference/serve_lerobot.py}"

[[ -d "$CKPT_PT"  ]] || { echo "ERROR: PyTorch ckpt not found: $CKPT_PT"  >&2; exit 1; }
[[ -f "$SERVE_PY" ]] || { echo "ERROR: serve_lerobot.py not found: $SERVE_PY" >&2; exit 1; }

CKPT_REAL="$(readlink -f "$CKPT_PT")"
EXTRA_MOUNT=()
if [[ "$CKPT_REAL" == "$HOME/src/openpi/"* ]]; then
  # Already covered by the $HOME/src/openpi -> /opt/openpi-train mount below.
  CKPT_PT_C="${CKPT_REAL/#$HOME\/src\/openpi/\/opt\/openpi-train}"
else
  # Bind-mount the checkpoint at the same path inside the container.
  EXTRA_MOUNT+=(-v "$CKPT_REAL:$CKPT_REAL:ro")
  CKPT_PT_C="$CKPT_REAL"
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

LEROBOT_SPEC="lerobot==${LEROBOT_VERSION}"

echo "Starting LeRobot policy server:"
echo "  checkpoint : $CKPT_PT"
echo "  lerobot    : $LEROBOT_SPEC"
echo "  prompt     : $PROMPT"
echo "  listening  : ws://0.0.0.0:${PORT}"
echo

exec docker run --runtime=nvidia --network host --name "$NAME" --rm \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/src/openpi:/opt/openpi-train" \
  "${EXTRA_MOUNT[@]}" \
  -v "$HOME/.cache/openpi:/root/.cache/openpi" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$SERVE_PY:/app/serve_lerobot.py:ro" \
  -w /opt/openpi-train \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/opt/openpi-train/packages/openpi-client/src:/opt/openpi-train/src:/opt/openpi-train \
  "$IMAGE" \
  bash -c "
    set -e
    echo '>>> Installing lerobot[pi]==${LEROBOT_VERSION} (pulls transformers 5.3.0; may take ~5 min) ...'
    pip install --quiet --no-cache-dir --upgrade 'lerobot[pi]==${LEROBOT_VERSION}'
    python -c 'import lerobot, huggingface_hub, transformers; print(f\"lerobot={lerobot.__version__} hf_hub={huggingface_hub.__version__} transformers={transformers.__version__}\")'
    echo '>>> Starting serve_lerobot.py ...'
    exec python /app/serve_lerobot.py \
      --checkpoint-dir '$CKPT_PT_C' \
      --default-prompt '$PROMPT' \
      --port $PORT
  "
