#!/usr/bin/env bash
# Start the TRT-backed openpi policy server for the SO-101 LoRA checkpoint.
#
# Mirrors serve.sh but uses the converted PyTorch SafeTensors + the FP8/NVFP4
# TensorRT engine instead of the JAX checkpoint. Same websocket protocol, so
# the existing run_so101 client works unchanged.
#
# Usage:
#   ./serve_trt.sh                                          # defaults
#   PROMPT="..." ./serve_trt.sh                             # override prompt
#   CKPT_PT=/some/dir ENGINE=/some/file ./serve_trt.sh      # override paths
set -euo pipefail

CKPT_PT="${CKPT_PT:-$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_so101_finetune_pytorch}"
ENGINE="${ENGINE:-$HOME/.cache/openpi/onnx_so101_fp8/onnx/model_fp8_nvfp4.engine}"
PROMPT="${PROMPT:-Pick up the orange ball and place it in the red bucket.}"
PORT="${PORT:-8000}"
IMAGE="${IMAGE:-openpi-pi0.5:latest}"
NAME="${NAME:-openpi-serve-so101-trt}"
SERVE_PY="${SERVE_PY:-$HOME/src/inference/serve_trt.py}"

[[ -d "$CKPT_PT"  ]] || { echo "ERROR: PyTorch ckpt not found: $CKPT_PT"  >&2; exit 1; }
[[ -f "$ENGINE"   ]] || { echo "ERROR: TRT engine not found: $ENGINE"     >&2; exit 1; }
[[ -f "$SERVE_PY" ]] || { echo "ERROR: serve_trt.py not found: $SERVE_PY" >&2; exit 1; }

# Re-map host paths under $HOME to /root (the docker user) for inside-container paths.
CKPT_PT_C="${CKPT_PT/#$HOME/\/root}"
ENGINE_C="${ENGINE/#$HOME/\/root}"

docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "Starting TRT policy server:"
echo "  pytorch ckpt : $CKPT_PT"
echo "  TRT engine   : $ENGINE"
echo "  prompt       : $PROMPT"
echo "  listening    : ws://0.0.0.0:${PORT}"
echo

exec docker run --runtime=nvidia --network host --name "$NAME" --rm \
  -v "$HOME/src/openpi:/opt/openpi-train" \
  -v "$HOME/.cache/openpi:/root/.cache/openpi" \
  -v "$SERVE_PY:/app/serve_trt.py:ro" \
  -w /opt/openpi-train \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/opt/openpi-train/packages/openpi-client/src:/opt/openpi-train/src:/opt/openpi-train \
  "$IMAGE" \
  python /app/serve_trt.py \
    --config pi05_so101_low_mem_finetune \
    --checkpoint-dir "$CKPT_PT_C" \
    --engine-path "$ENGINE_C" \
    --default-prompt "$PROMPT" \
    --port "$PORT"
