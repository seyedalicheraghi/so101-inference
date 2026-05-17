"""Closed-loop inference for the SO-101 trained pi0.5 LoRA checkpoint.

Talks to the openpi policy server (websocket) and drives the SO-101 follower
arm + 3 cameras. No lerobot dependency: motors are driven directly via
the Feetech STS3215 SDK with GroupSyncRead/GroupSyncWrite for low latency.

Pipeline per control step (target 30 Hz to match the dataset):
  1. Read all 6 joint positions in one GroupSyncRead, normalize via calibration.
  2. Grab frames:
       - top: RGB frame from RealSense bridge file (./start_rs_bridge.sh)
       - front, wrist: V4L2 UVC ArduCams
  3. Send the normalized state + 3 RGB frames + task to the policy server,
     receive an action chunk [H, 6] in LeRobot-normalized units.
  4. Per-tick: clip target-vs-current delta to MAX_STEP_TICKS, un-normalize,
     write all 6 goals in one GroupSyncWrite.

CALIBRATION
  Trained model expects joint positions in LeRobot-normalized units
  (-100..100 for body joints, 0..100 for gripper), computed from each
  motor's range_min/range_max. Copy from your data-collection host:
      ~/.cache/huggingface/lerobot/calibration/robots/so101_follower/<id>.json
  and pass it via --calibration. Without it the client uses identity
  ranges and the model output will be misinterpreted.

CAMERAS
  Top: start the RealSense bridge FIRST on the host: ./start_rs_bridge.sh
  It launches realsense2_camera and writes the latest RGB frame to
  /tmp/inference_diag/rs_top.npy (atomic). This client reads from that
  file rather than V4L2 directly — opening the D415 via /dev/videoN
  picks the wrong sub-device (depth or IR) and ruins the observation.
  Front, wrist: V4L2 UVC ArduCams via /dev/videoN — set --cam-front, --cam-wrist.

Run policy server first: ./serve_lerobot.sh
Then this client (also in docker): ./run_client.sh ...
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from openpi_client import websocket_client_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("so101")

# SO-101 motor layout (lerobot order). IDs 1..6 on the Feetech bus.
JOINT_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
)
MOTOR_IDS = (1, 2, 3, 4, 5, 6)
TICKS_PER_REV = 4096  # STS3215 12-bit encoder

# STS3215 register addresses — match so101_ros2/safe_teleop.py.
ADDR_MAX_TORQUE_LIMIT     = 16   # 2-byte EEPROM (the inference client was writing
                                 # to addr 48 here, which is NOT Max_Torque_Limit;
                                 # the cap silently never applied. Fixed: 16.)
ADDR_P_COEFFICIENT        = 21   # 1-byte
ADDR_D_COEFFICIENT        = 22   # 1-byte
ADDR_PROTECTION_CURRENT   = 28   # 2-byte (gripper only)
ADDR_OVERLOAD_TORQUE      = 36   # 1-byte (gripper only)
ADDR_TORQUE_ENABLE        = 40   # 1-byte
ADDR_ACCELERATION         = 41   # 1-byte
ADDR_GOAL_POSITION        = 42   # 2-byte
ADDR_PRESENT_POSITION     = 56   # 2-byte
ADDR_MAXIMUM_ACCELERATION = 85   # 1-byte EEPROM

# Per-motor settings (mirror so101_ros2/safe_teleop.py MOTOR_LIMITS).
MOTOR_LIMITS = {
    1: ("shoulder_pan",  1000, 254, 16, 32),
    2: ("shoulder_lift", 1000, 254, 16, 32),
    3: ("elbow_flex",    1000, 254, 16, 32),
    4: ("wrist_flex",    1000, 254, 16, 32),
    5: ("wrist_roll",    1000, 254, 16, 32),
    6: ("gripper",        500, 254, 16, 32),
}
GRIPPER_PROTECTION_CURRENT = 250
GRIPPER_OVERLOAD_TORQUE    = 25
MAX_STEP_TICKS = 100              # per-tick cap (~9° at 4096 ticks/rev)
BAUDRATE       = 1_000_000
READ_RETRIES   = 3

# RealSense bridge: rs_bridge.py writes the latest RGB frame here.
RS_BRIDGE_FRAME_PATH = "/tmp/inference_diag/rs_top.npy"
RS_BRIDGE_STALE_S    = 1.0        # warn if the .npy file is older than this


# --------------------------- arguments ---------------------------
@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    prompt: str = "Pick up the white box and place it in the white target area."
    arm_port: str = "/dev/ttyACM0"
    # Top camera: read from rs_bridge .npy by default. Use --top-source v4l2
    # only if you've explicitly verified that /dev/video{cam_top} is the colour
    # sub-device (rarely the case for D415).
    top_source: str = "bridge"        # "bridge" | "v4l2"
    cam_top: int = 2                  # only used when top_source == "v4l2"
    cam_front: int = 0
    cam_wrist: int = 4
    fps: int = 30
    chunk_steps: int = 16
    max_steps: int = 300
    # Path to lerobot-style calibration JSON. If empty, raw ticks are used.
    calibration: str = "/app/calibration.json"
    # Which server API to talk to:
    #   "lerobot" -> dotted keys, prompt key "task"
    #   "openpi"  -> slashed keys, prompt key "prompt"
    api: str = "lerobot"
    # Print devices and exit.
    list_devices: bool = False
    # Don't actually move the arm - just print the planned action.
    dry_run: bool = False


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    a = Args()
    for f in dataclasses.fields(a):
        kw = {"default": f.default}
        if f.type is bool or f.type == "bool":
            kw["action"] = "store_true"
        else:
            kw["type"] = type(f.default)
        p.add_argument(f"--{f.name.replace('_','-')}", **kw)
    return Args(**vars(p.parse_args()))


# --------------------------- device discovery ---------------------------
def list_devices() -> None:
    import glob
    print("=== USB serial (look for the SO-101 controller) ===")
    for x in sorted(glob.glob("/dev/serial/by-id/*") + glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")):
        print(" ", x)
    print()
    print("=== Video devices ===")
    for x in sorted(glob.glob("/dev/video*")):
        print(" ", x)
    print()
    print("=== RealSense bridge ===")
    if Path(RS_BRIDGE_FRAME_PATH).exists():
        age = time.time() - Path(RS_BRIDGE_FRAME_PATH).stat().st_mtime
        print(f"  {RS_BRIDGE_FRAME_PATH}  ({age:.1f}s old)")
    else:
        print(f"  {RS_BRIDGE_FRAME_PATH}  NOT FOUND — run ./start_rs_bridge.sh on the host")


# --------------------------- calibration ---------------------------
@dataclasses.dataclass
class JointCal:
    """LeRobot-style normalization between raw motor ticks and policy units.

    Mirrors so101_ros2/data_collector.py::_normalize_follower and
    lerobot/motors/motors_bus.py::_normalize:
      body joints (id 1-5, RANGE_M100_100):  out = (raw-min)/(max-min)*200 - 100
      gripper    (id 6, RANGE_0_100):        out =  (raw-min)/(max-min)*100
    """
    homing_offset: int = 0
    drive_mode: int = 0
    range_min: int = 0
    range_max: int = TICKS_PER_REV - 1
    motor_id: int = 0

    def _is_gripper(self) -> bool:
        return self.motor_id == 6

    def ticks_to_norm(self, raw: int) -> float:
        span = max(1, self.range_max - self.range_min)
        bounded = max(self.range_min, min(self.range_max, int(raw)))
        frac = (bounded - self.range_min) / span
        if self._is_gripper():
            return float(frac * 100.0)
        return float(frac * 200.0 - 100.0)

    def norm_to_ticks(self, val: float) -> int:
        span = self.range_max - self.range_min
        if self._is_gripper():
            frac = float(val) / 100.0
        else:
            frac = (float(val) + 100.0) / 200.0
        raw = int(round(self.range_min + frac * span))
        return int(np.clip(raw, self.range_min, self.range_max))


def load_calibration(path: str) -> dict[str, JointCal]:
    """Load lerobot-style calibration JSON. Empty path -> identity map."""
    if not path:
        log.warning("No --calibration provided; using identity (raw ticks). "
                    "Model output will be misinterpreted.")
        return {n: JointCal() for n in JOINT_NAMES}
    obj = json.loads(Path(path).read_text())
    out: dict[str, JointCal] = {}
    for n in JOINT_NAMES:
        if n not in obj:
            raise KeyError(f"calibration file missing joint '{n}'")
        e = obj[n]
        out[n] = JointCal(
            homing_offset=int(e.get("homing_offset", 0)),
            drive_mode=int(e.get("drive_mode", 0)),
            range_min=int(e.get("range_min", 0)),
            range_max=int(e.get("range_max", TICKS_PER_REV - 1)),
            motor_id=int(e.get("id", 0)),
        )
    return out


# --------------------------- motor bus ---------------------------
class Bus:
    """Feetech bus with GroupSyncRead/Write for atomic 6-motor I/O.

    Both reads and writes are batched into single bus round-trips — one
    sync_read per state observation (~5 ms) instead of 6 sequential reads
    (~30 ms, half the 33 ms tick budget at 30 Hz).
    """

    def __init__(self, port: str, calib: dict[str, JointCal]):
        import scservo_sdk as sdk
        self.sdk = sdk
        self.calib = calib
        self.port = sdk.PortHandler(port)
        if not self.port.openPort():
            raise RuntimeError(f"Could not open {port}")
        if not self.port.setBaudRate(BAUDRATE):
            raise RuntimeError(f"Could not set baud rate {BAUDRATE} on {port}")
        self.pkt = sdk.PacketHandler(0)  # protocol 0 for STS3215
        # One sync_read for present position across all 6 motors.
        self._sync_read = sdk.GroupSyncRead(
            self.port, self.pkt, ADDR_PRESENT_POSITION, 2
        )
        for mid in MOTOR_IDS:
            self._sync_read.addParam(mid)
        # One sync_write for goal position across all 6 motors.
        self._sync_write = sdk.GroupSyncWrite(
            self.port, self.pkt, ADDR_GOAL_POSITION, 2
        )

    def enable_torque(self, on: bool = True) -> None:
        for mid in MOTOR_IDS:
            self.pkt.write1ByteTxRx(self.port, mid, ADDR_TORQUE_ENABLE, 1 if on else 0)

    def apply_safe_limits(self) -> None:
        """Per-motor EEPROM tuning (mirrors safe_teleop._apply_safe_limits).

        Writes the torque cap to ADDR_MAX_TORQUE_LIMIT=16 (NOT 48 — the old
        client wrote to the wrong register and the cap silently never applied).
        """
        for mid in MOTOR_IDS:
            joint, torque_cap, accel_cap, p_gain, d_gain = MOTOR_LIMITS[mid]
            self.pkt.write1ByteTxRx(self.port, mid, ADDR_TORQUE_ENABLE, 0)
            try:
                pos = self._read_pos_single(mid)
                self.pkt.write2ByteTxRx(self.port, mid, ADDR_GOAL_POSITION, pos)
            except Exception:
                pass
            self.pkt.write2ByteTxRx(self.port, mid, ADDR_MAX_TORQUE_LIMIT,     torque_cap)
            self.pkt.write1ByteTxRx(self.port, mid, ADDR_MAXIMUM_ACCELERATION, accel_cap)
            self.pkt.write1ByteTxRx(self.port, mid, ADDR_ACCELERATION,         accel_cap)
            self.pkt.write1ByteTxRx(self.port, mid, ADDR_P_COEFFICIENT,        p_gain)
            self.pkt.write1ByteTxRx(self.port, mid, ADDR_D_COEFFICIENT,        d_gain)
            if mid == 6:
                self.pkt.write2ByteTxRx(self.port, mid, ADDR_PROTECTION_CURRENT, GRIPPER_PROTECTION_CURRENT)
                self.pkt.write1ByteTxRx(self.port, mid, ADDR_OVERLOAD_TORQUE,    GRIPPER_OVERLOAD_TORQUE)
            self.pkt.write1ByteTxRx(self.port, mid, ADDR_TORQUE_ENABLE, 1)
            log.info("  id=%d %-14s torque=%d accel=%d P=%d D=%d",
                     mid, joint, torque_cap, accel_cap, p_gain, d_gain)

    def _read_pos_single(self, mid: int, retries: int = 3, sleep_s: float = 0.003) -> int:
        """One-motor read with retries — used only by apply_safe_limits()."""
        last = None
        for _attempt in range(retries + 1):
            raw, comm, _err = self.pkt.read2ByteTxRx(self.port, mid, ADDR_PRESENT_POSITION)
            if comm == self.sdk.COMM_SUCCESS:
                return int(raw)
            last = self.pkt.getTxRxResult(comm)
            time.sleep(sleep_s)
        raise RuntimeError(f"read failed on motor {mid} after {retries+1} attempts: {last}")

    def _sync_read_retry(self) -> dict[int, int]:
        """Atomic 6-motor present-position read with retries on bus glitches."""
        last = "?"
        for _attempt in range(READ_RETRIES + 1):
            comm = self._sync_read.txRxPacket()
            if comm == self.sdk.COMM_SUCCESS:
                return {mid: self._sync_read.getData(mid, ADDR_PRESENT_POSITION, 2)
                        for mid in MOTOR_IDS}
            last = f"comm={comm}"
            time.sleep(0.003)
        raise RuntimeError(f"sync_read after {READ_RETRIES + 1} tries: {last}")

    def read_state(self) -> np.ndarray:
        """Return LeRobot-normalized state [-100..100 body, 0..100 gripper]."""
        pos = self._sync_read_retry()
        out = np.zeros(6, dtype=np.float32)
        for i, (mid, name) in enumerate(zip(MOTOR_IDS, JOINT_NAMES)):
            out[i] = self.calib[name].ticks_to_norm(pos[mid])
        return out

    def write_action(self, action_norm: np.ndarray) -> None:
        """Sync-write all 6 goal positions in one bus round-trip."""
        self._sync_write.clearParam()
        for i, (mid, name) in enumerate(zip(MOTOR_IDS, JOINT_NAMES)):
            ticks = self.calib[name].norm_to_ticks(float(action_norm[i]))
            data = [self.sdk.SCS_LOBYTE(ticks), self.sdk.SCS_HIBYTE(ticks)]
            self._sync_write.addParam(mid, bytes(data))
        comm = self._sync_write.txPacket()
        if comm != self.sdk.COMM_SUCCESS:
            raise RuntimeError(f"sync_write comm={comm}")

    def close(self) -> None:
        try: self.enable_torque(False)
        except Exception: pass
        try: self.port.closePort()
        except Exception: pass


# --------------------------- cameras ---------------------------
def make_v4l2_camera(idx: int, w: int = 640, h: int = 480, fps: int = 30,
                     warmup_frames: int = 60):
    """Open a V4L2 camera and warm it up (auto-exposure needs ~1-2s to settle)."""
    import cv2
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open /dev/video{idx}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    for fourcc in ("MJPG", "YUYV"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        ok, _ = cap.read()
        if ok:
            for _ in range(warmup_frames):
                cap.read()
            log.info("  /dev/video%d: opened %dx%d @ %s (%d warmup frames)",
                     idx, w, h, fourcc, warmup_frames)
            return cap
    cap.release()
    raise RuntimeError(f"/dev/video{idx} opened but no FOURCC produced frames at {w}x{h}")


def grab_v4l2(cap) -> np.ndarray:
    import cv2
    ok, bgr = cap.read()
    if not ok:
        raise RuntimeError("Camera read failed")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class RsBridgeReader:
    """Reads the latest RGB frame written by rs_bridge.py.

    rs_bridge.py runs on the host (outside docker) and atomically replaces
    /tmp/inference_diag/rs_top.npy each time a new RealSense colour frame
    arrives. We just np.load it. Atomic via os.replace, so a partial frame
    is never observed.
    """

    def __init__(self, path: str = RS_BRIDGE_FRAME_PATH, wait_s: float = 10.0):
        self.path = path
        deadline = time.time() + wait_s
        while not os.path.exists(path):
            if time.time() > deadline:
                raise RuntimeError(
                    f"RealSense bridge frame {path!r} not found after {wait_s:.0f}s — "
                    "start the bridge on the host: ./start_rs_bridge.sh"
                )
            time.sleep(0.2)
        # Warm up: wait until the bridge has written at least one new frame
        # so we don't reuse a stale snapshot from a previous run.
        mtime0 = os.path.getmtime(path)
        deadline = time.time() + 5.0
        while os.path.getmtime(path) == mtime0:
            if time.time() > deadline:
                log.warning("RS bridge frame mtime hasn't moved in 5s — bridge may be stalled")
                break
            time.sleep(0.05)
        self._last_mtime = 0.0
        self._stale_logged = False
        log.info("  rs_bridge: reading %s (latest frame %.1fs old)",
                 path, time.time() - os.path.getmtime(path))

    def read(self) -> np.ndarray:
        st = os.stat(self.path)
        age = time.time() - st.st_mtime
        if age > RS_BRIDGE_STALE_S and not self._stale_logged:
            log.warning("rs_bridge frame is %.1fs stale; is ./start_rs_bridge.sh running?", age)
            self._stale_logged = True
        elif age <= RS_BRIDGE_STALE_S:
            self._stale_logged = False
        return np.load(self.path)

    def release(self) -> None:
        pass


# --------------------------- main loop ---------------------------
def main(args: Args) -> int:
    if args.list_devices:
        list_devices()
        return 0

    log.info("Connecting to policy server ws://%s:%d ...", args.host, args.port)
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    log.info("Server metadata keys: %s", list(client.get_server_metadata().keys()))

    log.info("Loading calibration: %s", args.calibration or "(none)")
    calib = load_calibration(args.calibration)

    identity_count = sum(1 for jc in calib.values() if jc.range_min == 0 and jc.range_max == 4095)
    if identity_count == len(calib):
        log.error("ALL joints have identity calibration (0-4095). Model expects proper calibration!")
        log.error("  Pass --calibration /app/calibration.json or ensure calibration.json exists.")
    else:
        log.info("Calibration ranges: %s", {n: f"[{jc.range_min},{jc.range_max}] id={jc.motor_id}"
                                              for n, jc in calib.items()})

    log.info("Opening cameras (top=%s, front=%d, wrist=%d) ...",
             args.top_source if args.top_source == "bridge" else f"v4l2:{args.cam_top}",
             args.cam_front, args.cam_wrist)
    if args.top_source == "bridge":
        cam_top = RsBridgeReader()
        grab_top = cam_top.read
    elif args.top_source == "v4l2":
        cap_top = make_v4l2_camera(args.cam_top)
        cam_top = cap_top
        grab_top = lambda: grab_v4l2(cap_top)
    else:
        raise ValueError(f"--top-source must be 'bridge' or 'v4l2', got {args.top_source!r}")
    cam_front = make_v4l2_camera(args.cam_front)
    cam_wrist = make_v4l2_camera(args.cam_wrist)

    log.info("Connecting to SO-101 on %s ...", args.arm_port)
    bus = Bus(args.arm_port, calib)
    if not args.dry_run:
        log.info("Applying per-motor safety limits (torque, accel, P/D gains) ...")
        bus.apply_safe_limits()
        bus.enable_torque(True)

    period = 1.0 / args.fps
    step = 0
    bus_glitches_total = 0
    try:
        while step < args.max_steps:
            t0 = time.time()
            try:
                state = bus.read_state()
            except Exception as e:
                bus_glitches_total += 1
                if bus_glitches_total <= 3 or bus_glitches_total % 30 == 0:
                    log.warning("bus glitch #%d (obs read): %s", bus_glitches_total, e)
                time.sleep(0.005)
                continue
            img_top   = grab_top()
            img_front = grab_v4l2(cam_front)
            img_wrist = grab_v4l2(cam_wrist)
            if args.api == "lerobot":
                obs = {
                    "observation.state":         state,
                    "observation.images.front":  img_front,
                    "observation.images.top":    img_top,
                    "observation.images.wrist":  img_wrist,
                    "task":                      args.prompt,
                }
            else:  # openpi (legacy)
                obs = {
                    "observation/state":         state,
                    "observation/image_front":   img_front,
                    "observation/image_top":     img_top,
                    "observation/image_wrist":   img_wrist,
                    "prompt":                    args.prompt,
                }
            t1 = time.time()
            actions = np.asarray(client.infer(obs)["actions"])
            t2 = time.time()
            log.info("step=%d obs=%.0fms infer=%.0fms chunk=%s state=%s",
                     step, (t1 - t0) * 1e3, (t2 - t1) * 1e3,
                     actions.shape, np.round(state, 1).tolist())
            if step == 0:
                import cv2 as _cv2
                _diag_dir = Path("/tmp/inference_diag")
                _diag_dir.mkdir(exist_ok=True)
                _cv2.imwrite(str(_diag_dir / "cam_front.jpg"), _cv2.cvtColor(img_front, _cv2.COLOR_RGB2BGR))
                _cv2.imwrite(str(_diag_dir / "cam_top.jpg"),   _cv2.cvtColor(img_top,   _cv2.COLOR_RGB2BGR))
                _cv2.imwrite(str(_diag_dir / "cam_wrist.jpg"), _cv2.cvtColor(img_wrist, _cv2.COLOR_RGB2BGR))
                log.info("Saved camera frames to %s/ — verify front/top/wrist assignment!", _diag_dir)
                log.info("--- FIRST CHUNK DIAGNOSTIC ---")
                log.info("  state:      %s", np.round(state, 2).tolist())
                log.info("  actions[0]: %s", np.round(actions[0], 2).tolist())
                if actions.shape[0] > 1:
                    log.info("  actions[1]: %s", np.round(actions[1], 2).tolist())
                log.info("  actions[-1]: %s", np.round(actions[-1], 2).tolist())
                log.info("  action range: min=%s max=%s",
                         np.round(actions.min(axis=0), 2).tolist(),
                         np.round(actions.max(axis=0), 2).tolist())
                log.info("------------------------------")

            # Per-joint normalized cap derived from MAX_STEP_TICKS (tick-space cap).
            caps = np.empty(6, dtype=np.float32)
            for i, name in enumerate(JOINT_NAMES):
                jc = calib[name]
                span = max(1, jc.range_max - jc.range_min)
                scale = 100.0 if jc.motor_id == 6 else 200.0
                caps[i] = MAX_STEP_TICKS * scale / span
            bus_glitches = 0
            for k in range(min(args.chunk_steps, actions.shape[0])):
                try:
                    cur = bus.read_state()
                except Exception as e:
                    bus_glitches += 1
                    if bus_glitches <= 3 or bus_glitches % 30 == 0:
                        log.warning("  bus glitch #%d: %s", bus_glitches, e)
                    time.sleep(0.005)
                    step += 1
                    continue
                tgt = actions[k].astype(np.float32)
                delta = np.clip(tgt - cur, -caps, caps)
                target = (cur + delta).astype(np.float32)
                if args.dry_run:
                    log.info("  DRY %2d  target=%s", k, np.round(target, 1).tolist())
                else:
                    try:
                        bus.write_action(target)
                    except Exception as e:
                        bus_glitches += 1
                        log.warning("  write glitch #%d: %s", bus_glitches, e)
                step += 1
                sleep = period - (time.time() - t2 - k * period)
                if sleep > 0:
                    time.sleep(sleep)
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        log.info("Shutting down.")
        bus.close()
        for c in (cam_top, cam_front, cam_wrist):
            try: c.release()
            except Exception: pass
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
