from __future__ import annotations

import threading
import time

import cv2
import joblib
import numpy as np

from .features import feature_dim
from .filters import PointSmoother
from .mouse import MouseController
from .overlay import draw_tracking_overlay
from .paths import profile_paths
from .tracker import FaceTracker, open_camera


class OnnxGazeModel:
    def __init__(self, path):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ONNX Runtime is required for server-trained ONNX models. "
                "Install it locally with: python -m pip install -e .[onnx]"
            ) from exc

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, x):
        x = np.asarray(x, dtype=np.float32)
        return self.session.run([self.output_name], {self.input_name: x})[0]


class RuntimeState:
    def __init__(self) -> None:
        self.paused = False
        self.stop = False
        self.lock = threading.Lock()

    def toggle_pause(self) -> None:
        with self.lock:
            self.paused = not self.paused
            print("Paused" if self.paused else "Resumed")

    def request_stop(self) -> None:
        with self.lock:
            self.stop = True
            print("Stop requested")

    def snapshot(self) -> tuple[bool, bool]:
        with self.lock:
            return self.paused, self.stop


def _start_hotkeys(state: RuntimeState, mouse: MouseController):
    try:
        from pynput import keyboard
    except ImportError as exc:
        raise RuntimeError("pynput is not installed. Run: python -m pip install -e .") from exc

    hotkeys = keyboard.GlobalHotKeys(
        {
            "<ctrl>+<alt>+g": state.toggle_pause,
            "<ctrl>+<alt>+q": state.request_stop,
            "<ctrl>+<alt>+<enter>": mouse.left_click,
            "<ctrl>+<alt>+<backspace>": mouse.right_click,
        }
    )
    hotkeys.start()
    return hotkeys


def _load_model(path):
    onnx_path = path.with_name("model.onnx")
    onnx_metadata_path = path.with_name("model_metadata.json")
    if onnx_path.exists() and onnx_metadata_path.exists():
        import json

        metadata = json.loads(onnx_metadata_path.read_text(encoding="utf-8"))
        model = OnnxGazeModel(onnx_path)
    elif path.exists():
        artifact = joblib.load(path)
        model = artifact.get("model")
        metadata = artifact.get("metadata", {})
        if model is None:
            raise RuntimeError(f"Invalid model artifact: {path}")
    else:
        raise RuntimeError(f"No model found in {path.parent}. Run train or fetch a server-trained model first.")

    if int(metadata.get("feature_dim", -1)) != feature_dim():
        raise RuntimeError("Model feature dimension does not match this code. Re-train the model.")
    return model, metadata


def _valid_prediction(pred: np.ndarray, width: int, height: int) -> bool:
    if pred.shape != (2,):
        return False
    if not np.all(np.isfinite(pred)):
        return False
    margin_x = width * 0.25
    margin_y = height * 0.25
    return -margin_x <= pred[0] <= width + margin_x and -margin_y <= pred[1] <= height + margin_y


def run_cursor(args) -> int:
    paths = profile_paths(args.workspace, args.profile)
    model, metadata = _load_model(paths.model_path)
    screen_w, screen_h = [int(v) for v in metadata["screen_size"]]

    cap = open_camera(args.camera, args.width, args.height)
    mouse = MouseController(failsafe_margin=args.failsafe_margin)
    smoother = PointSmoother(min_cutoff=args.min_cutoff, beta=args.beta)
    state = RuntimeState()
    hotkeys = _start_hotkeys(state, mouse)

    print("Eye cursor running.")
    print("Hotkeys: Ctrl+Alt+G pause/resume, Ctrl+Alt+Q stop, Ctrl+Alt+Enter left click.")
    print(f"Model screen: {screen_w}x{screen_h}; current screen: {mouse.width}x{mouse.height}")
    if (int(mouse.width), int(mouse.height)) != (screen_w, screen_h):
        print("Warning: screen size changed since training. Recalibration is recommended.")

    last_cursor: tuple[int, int] | None = None
    last = time.perf_counter()
    fps = 0.0

    try:
        with FaceTracker() as tracker:
            while True:
                paused, stop = state.snapshot()
                if stop:
                    break

                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Camera read failed.")

                now = time.perf_counter()
                dt = max(now - last, 1e-6)
                last = now
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

                tracking = tracker.track_bgr(frame)
                status = "paused" if paused else "tracking"

                if not paused and tracking is not None:
                    vec = tracking.features.vector.reshape(1, -1)
                    pred = np.asarray(model.predict(vec)[0], dtype=np.float32)
                    if _valid_prediction(pred, screen_w, screen_h):
                        x, y = smoother.filter(float(pred[0]), float(pred[1]), time.perf_counter())
                        px, py = mouse.clamp(x, y)
                        if last_cursor is None or abs(px - last_cursor[0]) > args.dead_zone or abs(py - last_cursor[1]) > args.dead_zone:
                            last_cursor = mouse.move_to(px, py)
                        status = f"{int(px)}, {int(py)}"
                    else:
                        smoother.reset()
                        status = "invalid prediction"
                elif tracking is None:
                    smoother.reset()
                    status = "no face"

                if args.show:
                    draw_tracking_overlay(frame, tracking, fps=fps, text=status)
                    cv2.imshow("eye-cursor runtime", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                else:
                    cv2.waitKey(1)
    finally:
        try:
            hotkeys.stop()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()

    print("Eye cursor stopped.")
    return 0
