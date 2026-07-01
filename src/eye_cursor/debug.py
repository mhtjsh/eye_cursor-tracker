from __future__ import annotations

import time

import cv2

from .overlay import draw_tracking_overlay
from .tracker import FaceTracker, open_camera


def run_debug(args) -> int:
    cap = open_camera(args.camera, args.width, args.height)
    last = time.perf_counter()
    fps = 0.0
    try:
        with FaceTracker() as tracker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Camera read failed.")
                now = time.perf_counter()
                dt = max(now - last, 1e-6)
                last = now
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

                tracking = tracker.track_bgr(frame)
                draw_tracking_overlay(frame, tracking, fps=fps, text="Press q or Esc to quit")
                cv2.imshow("eye-cursor debug", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()
