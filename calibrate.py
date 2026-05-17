"""Calibrate the SO-101 follower arm.

Captures the same data lerobot-calibrate writes:
  - homing_offset  (encoder reading when joint is at its mechanical centre)
  - range_min / range_max  (full sweep extremes)
  - drive_mode (0 normal; we record raw and let the model handle signs)

Output is a lerobot-compatible JSON consumable by run_so101.py:
  {
    "shoulder_pan":  {"id": 1, "drive_mode": 0,
                      "homing_offset": 2048,
                      "range_min": 800, "range_max": 3300},
    ...
  }

No torque, no motion -- pure read-only. The operator physically moves the arm.
Safe to interrupt with Ctrl-C; partial output is not written.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import scservo_sdk as sdk

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper")
MOTOR_IDS = (1, 2, 3, 4, 5, 6)
TICKS_PER_REV = 4096
ADDR_PRESENT_POSITION = 56
ADDR_TORQUE_ENABLE = 40
BAUDRATE = 1_000_000


def open_bus(port_path: str):
    port = sdk.PortHandler(port_path)
    if not port.openPort():
        raise RuntimeError(f"Could not open {port_path}")
    if not port.setBaudRate(BAUDRATE):
        raise RuntimeError(f"Could not set baud {BAUDRATE} on {port_path}")
    pkt = sdk.PacketHandler(0)
    return port, pkt


def disable_torque(port, pkt) -> None:
    for mid in MOTOR_IDS:
        pkt.write1ByteTxRx(port, mid, ADDR_TORQUE_ENABLE, 0)


def ping_all(port, pkt) -> None:
    missing = []
    for mid in MOTOR_IDS:
        _, comm, _ = pkt.ping(port, mid)
        if comm != sdk.COMM_SUCCESS:
            missing.append(mid)
    if missing:
        raise RuntimeError(f"Could not reach motor IDs {missing} on bus")


def read_positions(port, pkt) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, mid in zip(JOINT_NAMES, MOTOR_IDS):
        raw, comm, err = pkt.read2ByteTxRx(port, mid, ADDR_PRESENT_POSITION)
        if comm != sdk.COMM_SUCCESS:
            raise RuntimeError(f"read failed motor {mid}: {pkt.getTxRxResult(comm)}")
        out[name] = int(raw)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0",
                    help="Serial device for the SO-101 follower arm bus")
    ap.add_argument("--id", default="my_follower_arm",
                    help="Calibration name (for log only)")
    ap.add_argument("--out", default="/app/calibration.json",
                    help="Where to write the calibration JSON")
    args = ap.parse_args()

    print(f"==> Opening {args.port}", flush=True)
    port, pkt = open_bus(args.port)
    try:
        print("==> Pinging motors 1..6 ...", flush=True)
        ping_all(port, pkt)

        print("==> Disabling torque so you can move the arm freely.", flush=True)
        disable_torque(port, pkt)

        print()
        print("=" * 70)
        print(" STEP 1 / 2 -- Centre the arm")
        print("=" * 70)
        print(" Pose every joint roughly half-way between its mechanical limits:")
        print("   shoulder_pan     -- straight forward (link points away from base)")
        print("   shoulder_lift    -- horizontal upper arm")
        print("   elbow_flex       -- forearm horizontal, ~90 deg at elbow")
        print("   wrist_flex       -- gripper level with forearm")
        print("   wrist_roll       -- gripper jaws horizontal")
        print("   gripper          -- half-open")
        print(" When ready, press ENTER. (Ctrl-C to abort.)")
        input()
        homing = read_positions(port, pkt)
        print("    homing offsets recorded:")
        for n in JOINT_NAMES:
            print(f"      {n:<14s} = {homing[n]:4d} ticks")

        print()
        print("=" * 70)
        print(" STEP 2 / 2 -- Sweep every joint through its full range")
        print("=" * 70)
        print(" Slowly walk each joint (except wrist_roll) from one mechanical")
        print(" stop to the other, then back. wrist_roll is auto-set to a full")
        print(" 360 deg range because it has no hard stops.")
        print(" Sampling at 30 Hz; press ENTER when done.", flush=True)
        input("    -> Press ENTER to START recording, then move the arm ...")

        mins = {n: homing[n] for n in JOINT_NAMES}
        maxs = {n: homing[n] for n in JOINT_NAMES}

        import threading
        stop = threading.Event()

        def wait_for_enter():
            try:
                input("    -> Press ENTER again to STOP recording ...")
            except EOFError:
                pass
            stop.set()
        t = threading.Thread(target=wait_for_enter, daemon=True)
        t.start()

        samples = 0
        t0 = time.time()
        while not stop.is_set():
            try:
                pos = read_positions(port, pkt)
            except RuntimeError as e:
                print(f"    [warn] {e}", file=sys.stderr)
                continue
            for n in JOINT_NAMES:
                if pos[n] < mins[n]: mins[n] = pos[n]
                if pos[n] > maxs[n]: maxs[n] = pos[n]
            samples += 1
            time.sleep(1.0 / 30.0)
        dt = time.time() - t0

        # Auto-fill wrist_roll with full 360.
        mins["wrist_roll"] = 0
        maxs["wrist_roll"] = TICKS_PER_REV - 1
        print(f"    captured {samples} samples in {dt:.1f}s")
        print("    range per joint (ticks):")
        for n in JOINT_NAMES:
            span_deg = (maxs[n] - mins[n]) * 360.0 / TICKS_PER_REV
            print(f"      {n:<14s} min={mins[n]:4d} max={maxs[n]:4d} span={span_deg:6.1f} deg")

        out = {}
        for n, mid in zip(JOINT_NAMES, MOTOR_IDS):
            out[n] = {
                "id": mid,
                "drive_mode": 0,
                "homing_offset": homing[n],
                "range_min": mins[n],
                "range_max": maxs[n],
            }

        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2))
        print()
        print(f"==> Wrote calibration to {path}")
        print(f"    (label: {args.id})")
        return 0
    finally:
        try: disable_torque(port, pkt)
        except Exception: pass
        try: port.closePort()
        except Exception: pass


if __name__ == "__main__":
    sys.exit(main())
