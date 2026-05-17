"""Read the current SO-101 follower pose in degrees and print it as
a comma-separated string suitable for `home.py --target`.

Usage (in container):
    python /app/save_pose.py --port /dev/ttyACM0 --calibration /app/calibration.json
"""
from __future__ import annotations
import argparse, sys
sys.path.insert(0, "/app")
from run_so101 import Bus, load_calibration, JOINT_NAMES


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--calibration", default="/app/calibration.json")
    a = p.parse_args()

    calib = load_calibration(a.calibration)
    bus = Bus(a.port, calib)
    try:
        pose = bus.read_state_deg()
    finally:
        bus.close()

    # Pretty summary to stderr (so stdout stays clean / pipeable)
    print("# current pose (deg):", file=sys.stderr)
    for n, v in zip(JOINT_NAMES, pose):
        print(f"#   {n:>13s} = {v:+7.2f}", file=sys.stderr)

    # Machine-readable on stdout
    print(",".join(f"{v:.2f}" for v in pose))
    return 0


if __name__ == "__main__":
    sys.exit(main())
