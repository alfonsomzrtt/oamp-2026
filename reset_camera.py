#!/usr/bin/env python3
"""Reset camera to hardware defaults — run if calibration broke your webcam."""

import cv2
import time

idx = int(input(f"Camera index [0]: ") or "0")
cap = cv2.VideoCapture(idx)

if not cap.isOpened():
    print(f"Cannot open camera {idx}")
    exit(1)

props = {
    "Brightness": cv2.CAP_PROP_BRIGHTNESS,
    "Contrast": cv2.CAP_PROP_CONTRAST,
    "Saturation": cv2.CAP_PROP_SATURATION,
    "Gain": cv2.CAP_PROP_GAIN,
    "Exposure": cv2.CAP_PROP_EXPOSURE,
    "Sharpness": cv2.CAP_PROP_SHARPNESS,
}

print(f"\nCamera {idx} CURRENT values:")
for name, prop in props.items():
    val = cap.get(prop)
    print(f"  {name}: {val}")

print(f"\nResetting to defaults (0 or mid-range)...")
for name, prop in props.items():
    current = cap.get(prop)
    # Most cameras: 128 or 0 = default
    default = 128 if current > 10 else 0
    cap.set(prop, default)
    time.sleep(0.05)

# Read back
print(f"\nCamera {idx} AFTER reset:")
for name, prop in props.items():
    val = cap.get(prop)
    print(f"  {name}: {val}")

cap.release()
print(f"\nDone. Camera {idx} reset. Close any app using the camera and reopen.")
print(f"If still calibrated, try rebooting — the change persists in camera driver.")
