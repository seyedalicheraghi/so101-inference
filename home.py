"""Smoothly drive the SO-101 follower back to its calibrated centre pose.

Centre = the pose you held during STEP 1 of calibration (homing offset).
In degrees that is exactly [0, 0, 0, 0, 0, 0].

Usage:
    python /app/home.py --port /dev/ttyACM0 --calibration /app/calibration.json \\
        [--target 0,0,0,0,0,0] [--seconds 3] [--tol-deg 1.0] [--keep-torque]
"""
from __future__ import annotations
import argparse, sys, time
import numpy as np

sys.path.insert(0, "/app")
from run_so101 import Bus, load_calibration, JOINT_NAMES


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--calibration", default="/app/calibration.json")
    p.add_argument("--target", default="0,0,0,0,0,0",
                   help="6 comma-separated target degrees (default centre)")
    p.add_argument("--seconds", type=float, default=3.0,
                   help="time the interpolation ramp takes")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--tol-deg", type=float, default=1.0,
                   help="settle tolerance per joint before exiting")
    p.add_argument("--settle-timeout", type=float, default=4.0,
                   help="max seconds to wait for settle after the ramp")
    p.add_argument("--keep-torque", action="store_true",
                   help="leave torque ON when done (arm holds target)")
    return p.parse_args()


def fmt(arr: np.ndarray) -> str:
    return ", ".join(f"{n}={v:+6.1f}" for n, v in zip(JOINT_NAMES, arr))


def main() -> int:
    a = parse()
    target = np.array([float(x) for x in a.target.split(",")], dtype=np.float32)
    if target.shape != (6,):
        print(f"ERROR: --target must have 6 values, got {target}", file=sys.stderr)
        return 2

    calib = load_calibration(a.calibration)
    bus = Bus(a.port, calib)
    try:
        start = bus.read_state_deg().copy()
        print("current : " + fmt(start))
        print("target  : " + fmt(target))
        print(f"ramping over {a.seconds:.1f}s @ {a.fps} Hz; settle tol {a.tol_deg:.1f} deg")

        bus.enable_torque(True)

        # Phase 1: linear interpolation from current to target.
        n_steps = max(1, int(round(a.seconds * a.fps)))
        dt = 1.0 / a.fps
        for k in range(1, n_steps + 1):
            alpha = k / n_steps
            cmd = (1.0 - alpha) * start + alpha * target
            bus.write_action_deg(cmd)
            time.sleep(dt)

        # Phase 2: hold target while polling actual position; exit when
        # every joint is within tolerance OR settle_timeout elapses.
        deadline = time.time() + a.settle_timeout
        last_err = None
        while time.time() < deadline:
            bus.write_action_deg(target)
            actual = bus.read_state_deg()
            err = np.abs(actual - target)
            last_err = err
            if np.all(err <= a.tol_deg):
                break
            time.sleep(dt)

        final = bus.read_state_deg()
        within = np.all(np.abs(final - target) <= a.tol_deg)
        print("done    : " + fmt(final))
        if not within:
            print(f"WARN: did not settle within {a.tol_deg} deg "
                  f"(max joint error {float(np.max(np.abs(final - target))):.2f} deg)")

        if a.keep_torque:
            print("Torque LEFT ON (--keep-torque). Arm will hold this pose.")
            print("Run again without --keep-torque (or another script) to release.")
            return 0
    finally:
        if not a.keep_torque:
            bus.close()  # disables torque, arm goes limp
    return 0


if __name__ == "__main__":
    sys.exit(main())
