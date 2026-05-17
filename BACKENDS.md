# SO-101 inference backends

Two policy servers, both listening on `ws://0.0.0.0:8000`, both speaking the
same websocket protocol. The client (`run_client.sh`) does not change.

| Backend | Server script | Image | Source artifact | Size |
|---|---|---|---|---|
| **JAX** (reference) | `./serve.sh` | `openpi-train:so101` | JAX checkpoint at `~/src/openpi/checkpoints/.../7999` (symlink to `~/archive/...`) | 15 GB |
| **TRT engine** (FP8 + NVFP4) | `./serve_trt.sh` | `openpi-pi0.5:latest` | `~/.cache/openpi/onnx_so101_fp8/onnx/model_fp8_nvfp4.engine` | 3.0 GB |

## Conversion pipeline that produced the engine

```
JAX so101 LoRA  ──[examples/convert_jax_model_to_pytorch.py]──►  PyTorch SafeTensors  (6.8 GB)
                ──[openpi_on_thor/pytorch_to_onnx.py]─────────►  FP8+NVFP4 ONNX        (5.5 GB)
                ──[openpi_on_thor/build_engine.sh]────────────►  TensorRT engine       (3.0 GB)
```

Intermediate artifacts on Thor:
- PyTorch  : `~/.cache/openpi/openpi-assets/checkpoints/pi05_so101_finetune_pytorch/`
- ONNX     : `~/.cache/openpi/onnx_so101_fp8/onnx/model_fp8_nvfp4.{onnx,data}`
- Engine   : `~/.cache/openpi/onnx_so101_fp8/onnx/model_fp8_nvfp4.engine`

## How to run with each backend

Pick one server in terminal A. Both share the same port; only one at a time.

```bash
# Terminal A: JAX backend
cd ~/src/inference && ./serve.sh

# OR

# Terminal A: TRT engine backend (faster on Thor)
cd ~/src/inference && ./serve_trt.sh
```

Then in terminal B (same client either way):

```bash
cd ~/src/inference
./run_client.sh \
  --cam-front 0 --cam-top 8 --cam-wrist 2 \
  --arm-port /dev/ttyACM0 \
  --max-joint-step-deg 1.5 \
  --max-steps 200 \
  --calibration /app/calibration.json
# add --dry-run to test without moving the arm
```

## Reproducing the conversion (if you ever retrain or fine-tune again)

All three steps run inside `openpi-pi0.5:latest`. The container needs
`~/src/openpi` mounted (for the so101 config) and `~/.cache/openpi` mounted
(for ckpts + outputs).

```bash
# 1. JAX -> PyTorch (~10 min)
cd ~/src/openpi
docker run --rm --runtime=nvidia \
  -v "$PWD":/workspace \
  -v "$HOME/.cache/openpi":/root/.cache/openpi \
  -v "$HOME/archive":/home/smartoaster/archive \
  -w /workspace \
  -e PYTHONPATH=packages/openpi-client/src:src:.:$PYTHONPATH \
  openpi-pi0.5:latest \
  bash -c "
    cp -r ./src/openpi/models_pytorch/transformers_replace/* /usr/local/lib/python3.12/dist-packages/transformers/
    python openpi_on_thor/patches/apply_gemma_fixes.py
    python examples/convert_jax_model_to_pytorch.py \
      --checkpoint_dir /root/.cache/openpi/openpi-assets/checkpoints/pi05_so101_finetune \
      --output_path  /root/.cache/openpi/openpi-assets/checkpoints/pi05_so101_finetune_pytorch \
      --config-name pi05_so101_low_mem_finetune
  "
# Then copy assets:
cp -r ~/.cache/openpi/openpi-assets/checkpoints/pi05_so101_finetune/assets \
      ~/.cache/openpi/openpi-assets/checkpoints/pi05_so101_finetune_pytorch/

# 2. PyTorch -> FP8+NVFP4 ONNX (~15 min)
docker run --rm --runtime=nvidia --network host \
  -v "$HOME/src/openpi":/workspace \
  -v "$HOME/.cache/openpi":/root/.cache/openpi \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -w /workspace \
  -e PYTHONPATH=packages/openpi-client/src:src:.:$PYTHONPATH \
  openpi-pi0.5:latest \
  bash -c "
    cp -r ./src/openpi/models_pytorch/transformers_replace/* /usr/local/lib/python3.12/dist-packages/transformers/
    python openpi_on_thor/patches/apply_gemma_fixes.py
    python openpi_on_thor/pytorch_to_onnx.py \
      --checkpoint_dir /root/.cache/openpi/openpi-assets/checkpoints/pi05_so101_finetune_pytorch \
      --output_path /root/.cache/openpi/onnx_so101_fp8 \
      --config_name pi05_so101_low_mem_finetune \
      --precision fp8 --enable_llm_nvfp4 --quantize_attention_matmul \
      --num_calibration_samples 16
  "

# 3. ONNX -> TRT engine (~20 min)
docker run --rm --runtime=nvidia \
  -v "$HOME/src/openpi":/workspace \
  -v "$HOME/.cache/openpi":/root/.cache/openpi \
  -w /workspace \
  -e ACTION_HORIZON=10 \
  openpi-pi0.5:latest \
  bash -c "
    cp -r ./src/openpi/models_pytorch/transformers_replace/* /usr/local/lib/python3.12/dist-packages/transformers/
    bash openpi_on_thor/build_engine.sh \
      /root/.cache/openpi/onnx_so101_fp8/onnx/model_fp8_nvfp4.onnx \
      /root/.cache/openpi/onnx_so101_fp8/onnx/model_fp8_nvfp4.engine \
      1 1 1 10
  "
# trtexec will print "FAILED" at the verification step because so101 has
# static max_token_len=200 (trtexec tests with 1x128). The engine file is
# still saved successfully.

# Then chown the outputs (docker creates them as root):
echo C@t_R0b0tics | sudo -S chown -R smartoaster:smartoaster \
  ~/.cache/openpi/openpi-assets/checkpoints/pi05_so101_finetune_pytorch \
  ~/.cache/openpi/onnx_so101_fp8
```

## Expected speedup

Per the Jetson AI Lab benchmark on `pi05_libero` (single Thor box, MAXN):

| Backend | Total latency | Speedup |
|---|---|---|
| PyTorch BF16 | ~163 ms | 1.0x |
| **TensorRT FP8 + NVFP4** | **~94 ms** | **~1.7x** |

JAX was not in the published benchmark; in practice it sits near PyTorch BF16
because both share the same compute graph. Real numbers will vary on so101
because we have 3 cameras (vs libero's 2) and a different action horizon.
