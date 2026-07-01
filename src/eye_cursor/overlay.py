from __future__ import annotations

import cv2

from .features import LEFT_EYE, LEFT_IRIS, RIGHT_EYE, RIGHT_IRIS
from .tracker import TrackingResult


def draw_tracking_overlay(frame, tracking: TrackingResult | None, fps: float | None = None, text: str = ""):
    h, w = frame.shape[:2]
    if tracking is not None:
        min_x, min_y, box_w, box_h = tracking.features.face_box
        p1 = (int(min_x * w), int(min_y * h))
        p2 = (int((min_x + box_w) * w), int((min_y + box_h) * h))
        cv2.rectangle(frame, p1, p2, (0, 180, 255), 2)

        for idx in LEFT_EYE + RIGHT_EYE:
            if idx < len(tracking.landmark_xy):
                x, y = tracking.landmark_xy[idx]
                cv2.circle(frame, (int(x * w), int(y * h)), 1, (0, 255, 0), -1)
        for idx in LEFT_IRIS + RIGHT_IRIS:
            if idx < len(tracking.landmark_xy):
                x, y = tracking.landmark_xy[idx]
                cv2.circle(frame, (int(x * w), int(y * h)), 2, (255, 120, 0), -1)

        q = tracking.features.quality
        quality_text = (
            f"face={q['face_width']:.2f}x{q['face_height']:.2f} "
            f"eyes={q['left_eye_open']:.2f}/{q['right_eye_open']:.2f}"
        )
        cv2.putText(frame, quality_text, (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        cv2.putText(frame, "No face", (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if fps is not None:
        cv2.putText(frame, f"{fps:5.1f} FPS", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    if text:
        cv2.putText(frame, text, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame
