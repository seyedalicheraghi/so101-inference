# SO-101 closed-loop inference (pi0.5 LoRA)

Drives the SO-101 arm in real time using the LoRA fine-tuned in
`~/src/openpi/checkpoints/pi05_so101_low_mem_finetune/so101_20260511_1638/`.

```
                 +-------------------+   ws://localhost:8000   +----------------------+
USB cams ------> | openpi-client:    | <---------------------> | openpi-train:so101  |
USB SO-101 arm-> | so101 (docker)    |  msgpack obs/actions    | (docker, GPU)       |
                 |  run_so101.py     |                         |  scripts/serve_policy|
                 +-------------------+                         +----------------------+
                  ./run_client.sh                                ./serve.sh
```

Both the **server** and the **client** run in their own docker containers - no
Python on Thor itself.

| Image | Purpose | Built from |
|---|---|---|
| `openpi-train:so101`  | Policy server (GPU) | already built during training |
| `openpi-client:so101` | Robot client (CPU)  | `./build_client.sh` here |

The client image has **no lerobot dependency** - motors are driven directly
via `feetech-servo-sdk`, cameras via OpenCV/V4L2.

---

## 1. One-time setup

```bash
cd ~/src/inference
./build_client.sh         # builds openpi-client:so101 (~3 min)
```

### Calibration (REQUIRED)

The model was trained on joint positions in **degrees relative to a
per-joint homing offset**. You need the same calibration that was used at
data-collection time. lerobot's calibration files live at:

```
~/.cache/huggingface/lerobot/calibration/robots/so101_follower/<id>.json
```

Copy it next to the run scripts:

```bash
cp ~/.cache/huggingface/lerobot/calibration/robots/so101_follower/*.json \
   ~/src/inference/calibration.json
```

Format (lerobot 0.4): one JSON object keyed by joint name -

```json
{
  "shoulder_pan":  {"id":1,"drive_mode":0,"homing_offset":2048,"range_min":1000,"range_max":3000},
  "shoulder_lift": {"id":2,"drive_mode":0,"homing_offset":2048,"range_min":900, "range_max":3100},
  ...
  "gripper":       {"id":6,"drive_mode":0,"homing_offset":2048,"range_min":1500,"range_max":3500}
}
```

If you don't have the file, run lerobot's calibration wizard once on a host
with lerobot installed, then copy the result here.

### Find your USB devices

Plug in arm + 3 cameras, then:

```bash
./run_client.sh --list-devices
```

Note the `/dev/serial/by-id/...` for the arm and the `/dev/videoN` indices
for the front, top, wrist cameras (they must match the dataset roles).

---

## 2. Start the policy server

In one terminal:

```bash
cd ~/src/inference
./serve.sh                  # final checkpoint (step 7999), training prompt
./serve.sh 5000             # try the step-5000 checkpoint
PROMPT="..." ./serve.sh     # override prompt
```

Wait for `Creating server (host: thor, ip: ...)` (~30 s warmup).

---

## 3. Start the RealSense bridge (HOST, not docker)

The D415 top camera is read via the `realsense2_camera` ROS2 node + a small
host-side bridge that writes the latest RGB frame to a shared file. Opening
the D415 directly with V4L2 inside docker tends to grab the wrong
sub-device (depth/IR).

In a second terminal, on the host:

```bash
cd ~/src/inference
./start_rs_bridge.sh        # launches realsense2_camera + rs_bridge.py
```

You should see `Frame N: (480, 640, 3) RGB mean=...` logs once it's streaming.

## 4. Run the robot

In a third terminal:

```bash
cd ~/src/inference
./run_client.sh \
    --arm-port /dev/ttyACM0 \
    --cam-front 8 --cam-top 10 --cam-wrist 12 \
    --calibration /app/calibration.json \
    --max-steps 300                    # 10 s at 30 fps
```

Useful flags:

| flag | default | meaning |
|---|---|---|
| `--host` / `--port` | `127.0.0.1` / `8000` | server URL |
| `--prompt` | (training prompt) | language goal |
| `--chunk-steps` | `16` | actions per chunk before re-planning |
| `--fps` | `30` | dataset fps - do not change unless retrained |
| `--max-steps` | `300` | total control steps before stopping |
| `--calibration` | `/app/calibration.json` | lerobot-format JSON; empty -> raw ticks (broken) |
| `--top-source` | `bridge` | `bridge` reads RealSense via `rs_bridge.py`; `v4l2` opens `/dev/video<cam_top>` directly (rarely correct for D415) |
| `--dry-run` | off | print planned targets, don't actually move arm |

**First-run safety**: try `--dry-run --max-steps 30` first to see what the
model wants to do without the arm moving. The per-tick joint delta is hard-
clamped to `MAX_STEP_TICKS=100` (~9°) in `run_so101.py`.

`./run_client.sh` automatically passes `--device` for the arm + each
`/dev/videoN` and bind-mounts `./calibration.json` into the container at
`/app/calibration.json`.

---

## 5. Switch checkpoints

| step | train loss | recommended for |
|------|------------|-----------------|
| 2500 | 0.013 | sanity check |
| 5000 | 0.010 | fallback if final overfits |
| 7500 | 0.009 | similar to final |
| **7999** | **0.0087** | start here |

`./serve.sh <step>`.

---

## 6. Common issues

| symptom | likely cause / fix |
|---|---|
| arm moves to the same position regardless of obs | calibration missing; raw ticks ≠ degrees |
| `Could not open /dev/videoN` | wrong index → `--list-devices`; another process owns it → `fuser /dev/videoN` |
| `Could not open /dev/ttyACM0` | wrong port; check `ls /dev/serial/by-id/` for stable name |
| `Connection refused` | server still loading; `docker logs -f openpi-serve-so101` |
| arm jerks | first try with `--dry-run`, then lower `MAX_STEP_TICKS` in `run_so101.py` |
| top camera blank / wrong scene | RealSense bridge isn't running — start `./start_rs_bridge.sh` on the host first, then check `/tmp/inference_diag/rs_top.npy` exists |
| inference > 100 ms first call | normal: JAX compile; warm calls 30-80 ms |
| `read failed on motor N` | bus baud wrong, motor unpowered, or ID mismatch |

---

## 7. Files

```
~/src/inference/
├── README.md           (this file)
├── Dockerfile          openpi-client:so101 image
├── build_client.sh     builds openpi-client:so101 from this dir
├── run_client.sh       docker-runs the client with USB passthrough
├── run_so101.py        client entrypoint (lives at /app/run_so101.py in image)
├── serve.sh            docker-runs serve_policy from openpi-train:so101
├── serve_lerobot.sh    docker-runs the PyTorch/LeRobot policy server
├── start_rs_bridge.sh  HOST-side launcher for realsense2_camera + rs_bridge.py
├── rs_bridge.py        ROS2 node that writes /tmp/inference_diag/rs_top.npy
└── calibration.json    YOU PROVIDE - copied from data-collection host
```
