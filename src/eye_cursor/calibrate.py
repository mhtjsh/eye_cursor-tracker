from __future__ import annotations

import json
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .features import feature_dim
from .paths import profile_paths
from .screen import get_screen_size
from .tracker import FaceTracker, open_camera


def _target_grid(screen_w: int, screen_h: int, grid_x: int, grid_y: int, margin: float) -> list[tuple[int, int]]:
    x0 = int(screen_w * margin)
    x1 = int(screen_w * (1.0 - margin))
    y0 = int(screen_h * margin)
    y1 = int(screen_h * (1.0 - margin))
    xs = np.linspace(x0, x1, grid_x)
    ys = np.linspace(y0, y1, grid_y)
    return [(int(x), int(y)) for y in ys for x in xs]


def _draw_target(canvas, point: tuple[int, int], progress: float, message: str) -> None:
    h, w = canvas.shape[:2]
    canvas[:] = (12, 12, 12)
    x, y = point
    radius = 18
    cv2.circle(canvas, (x, y), radius, (0, 210, 255), 2)
    cv2.circle(canvas, (x, y), 4, (0, 210, 255), -1)
    cv2.line(canvas, (x - 32, y), (x + 32, y), (0, 210, 255), 1)
    cv2.line(canvas, (x, y - 32), (x, y + 32), (0, 210, 255), 1)
    cv2.rectangle(canvas, (0, h - 8), (int(w * progress), h), (0, 210, 255), -1)
    cv2.putText(canvas, message, (32, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2)


def _instruction_screen(width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (12, 12, 12)
    lines = [
        "Eye Cursor Calibration",
        "",
        "Look at each target until it moves.",
        "Keep your head natural and the laptop still.",
        "Press Space to start. Press Esc to cancel.",
    ]
    y = height // 2 - 110
    for i, line in enumerate(lines):
        scale = 1.0 if i == 0 else 0.72
        color = (0, 210, 255) if i == 0 else (235, 235, 235)
        cv2.putText(canvas, line, (width // 2 - 360, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
        y += 46
    return canvas


def _save_calibration(
    paths,
    calibration_path: Path,
    features: list[np.ndarray],
    targets: list[tuple[int, int]],
    screen_size: tuple[int, int],
    camera_size: tuple[int, int],
    timestamps: list[float],
    qualities: list[dict[str, float]],
) -> None:
    x = np.vstack(features).astype(np.float32)
    y = np.asarray(targets, dtype=np.float32)
    quality_json = np.asarray([json.dumps(q, sort_keys=True) for q in qualities])
    np.savez_compressed(
        calibration_path,
        features=x,
        targets=y,
        screen_size=np.asarray(screen_size, dtype=np.int32),
        camera_size=np.asarray(camera_size, dtype=np.int32),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        quality_json=quality_json,
        feature_dim=np.asarray([feature_dim()], dtype=np.int32),
        created_at=np.asarray([datetime.now().isoformat(timespec="seconds")]),
    )
    paths.latest_calibration_marker.write_text(str(calibration_path.name), encoding="utf-8")


def run_calibration(args) -> int:
    paths = profile_paths(args.workspace, args.profile)
    paths.ensure_dirs(save_frames=args.save_frames)

    screen_w, screen_h = get_screen_size()
    cap = open_camera(args.camera, args.width, args.height)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    calibration_path = paths.profile_dir / f"calibration_{timestamp}.npz"

    targets_base = _target_grid(screen_w, screen_h, args.grid_x, args.grid_y, args.margin)
    if not targets_base:
        raise RuntimeError("Calibration grid produced no targets.")

    target_sequence: list[tuple[int, int]] = []
    rng = random.Random(timestamp)
    for _ in range(args.rounds):
        round_targets = targets_base[:]
        rng.shuffle(round_targets)
        target_sequence.extend(round_targets)

    print(f"Profile: {args.profile}")
    print(f"Screen: {screen_w}x{screen_h}")
    print(f"Targets: {len(target_sequence)}")
    print(f"Output: {calibration_path}")

    features: list[np.ndarray] = []
    targets: list[tuple[int, int]] = []
    timestamps: list[float] = []
    qualities: list[dict[str, float]] = []

    window = "eye-cursor calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window, _instruction_screen(screen_w, screen_h))

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            print("Calibration cancelled before start.")
            return 1
        if key == 32:
            break

    try:
        with FaceTracker() as tracker:
            for idx, target in enumerate(target_sequence, start=1):
                started = time.perf_counter()
                while True:
                    elapsed = time.perf_counter() - started
                    if elapsed >= args.seconds_per_target:
                        break

                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("Camera read failed during calibration.")

                    progress = ((idx - 1) + min(elapsed / args.seconds_per_target, 1.0)) / len(target_sequence)
                    message = f"Target {idx}/{len(target_sequence)}"
                    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                    _draw_target(canvas, target, progress, message)
                    cv2.imshow(window, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        raise KeyboardInterrupt

                    if elapsed < args.settle_seconds:
                        continue

                    tracking = tracker.track_bgr(frame)
                    if tracking is None:
                        continue

                    features.append(tracking.features.vector)
                    targets.append(target)
                    timestamps.append(time.time())
                    qualities.append(tracking.features.quality)

                    if args.save_frames and len(features) % 10 == 0:
                        frame_path = paths.captures_dir / f"{timestamp}_{len(features):06d}.jpg"
                        cv2.imwrite(str(frame_path), frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(features) < 100:
        raise RuntimeError(f"Only collected {len(features)} usable samples. Need at least 100.")

    _save_calibration(
        paths,
        calibration_path,
        features,
        targets,
        (screen_w, screen_h),
        (args.width, args.height),
        timestamps,
        qualities,
    )
    print(f"Saved {len(features)} samples to {calibration_path}")
    print("Next: eye-cursor train --profile", args.profile)
    return 0
