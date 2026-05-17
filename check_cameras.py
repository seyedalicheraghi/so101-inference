#!/usr/bin/env python3
"""Camera diagnostic tool for SO-101 inference on Thor."""
import cv2, numpy as np, sys
from pathlib import Path

CAMERAS = {
    "top (video4, RealSense)": 4,
    "front (video6, ArduCam)": 6,
    "wrist (video8, ArduCam)": 8,
}
SEP = "=" * 60

def capture_frame(idx, warmup=60):
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for fourcc in ("MJPG", "YUYV"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        ok, _ = cap.read()
        if ok:
            break
    else:
        cap.release()
        return None
    for _ in range(warmup):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None

def analyze(name, frame, save_dir):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_b = frame.mean()
    white_pct = ((hsv[:,:,1] < 30) & (hsv[:,:,2] > 200)).mean() * 100
    colorful_pct = (hsv[:,:,1] > 100).mean() * 100
    tl = frame[:h//2, :w//2].mean()
    tr = frame[:h//2, w//2:].mean()
    bl = frame[h//2:, :w//2].mean()
    br = frame[h//2:, w//2:].mean()
    asym_lr = abs(tl + bl - tr - br) / 2
    print(f"\n{SEP}")
    print(f"  {name}")
    print(f"{SEP}")
    print(f"  Mean brightness:  {mean_b:.0f}/255")
    print(f"  White objects:    {white_pct:.1f}%")
    print(f"  Colorful pixels:  {colorful_pct:.1f}%")
    print(f"  Quadrants:        TL={tl:.0f} TR={tr:.0f} BL={bl:.0f} BR={br:.0f}")
    print(f"  L-R asymmetry:    {asym_lr:.0f} {!! HIGH if asym_lr > 40 else OK}")
    if white_pct < 1:
        print(f"  WARNING: No white objects visible")
    if save_dir:
        safe = name.split("(")[0].strip().replace(" ", "_")
        path = save_dir / f"{safe}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"  Saved: {path}")

def main():
    save_dir = Path("/tmp/cam_diag")
    save_dir.mkdir(exist_ok=True)
    print("Camera Diagnostic (60-frame warmup per camera)...")
    for name, idx in CAMERAS.items():
        frame = capture_frame(idx)
        if frame is None:
            print(f"\n  FAIL: {name}")
        else:
            analyze(name, frame, save_dir)
    print(f"\n{SEP}")
    print("Check saved frames: scp smartoaster@thor:/tmp/cam_diag/*.jpg .")

if __name__ == "__main__":
    main()
