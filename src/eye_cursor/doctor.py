from __future__ import annotations

import importlib
import platform
import sys

from .screen import get_screen_size
from .tracker import open_camera


REQUIRED_MODULES = [
    "cv2",
    "joblib",
    "mediapipe",
    "numpy",
    "pyautogui",
    "pynput",
    "sklearn",
]


def _check_import(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        return False, str(exc)
    version = getattr(mod, "__version__", "installed")
    return True, str(version)


def run_doctor(args) -> int:
    print(f"Python: {sys.version.split()[0]} at {sys.executable}")
    print(f"Platform: {platform.platform()}")

    py_ok = (3, 11) <= sys.version_info[:2] < (3, 13)
    if not py_ok:
        print("Python status: unsupported for this project. Use Python 3.11 or 3.12.")
    else:
        print("Python status: ok")

    import_ok = True
    for name in REQUIRED_MODULES:
        ok, detail = _check_import(name)
        import_ok = import_ok and ok
        state = "ok" if ok else "missing"
        print(f"{name}: {state} ({detail})")

    try:
        width, height = get_screen_size()
        print(f"Screen: {width}x{height}")
        screen_ok = True
    except Exception as exc:
        print(f"Screen: failed ({exc})")
        screen_ok = False

    camera_ok = False
    try:
        cap = open_camera(args.camera, 640, 480)
        ok, _frame = cap.read()
        camera_ok = bool(ok)
        print(f"Camera {args.camera}: {'ok' if camera_ok else 'opened but no frame'}")
    except Exception as exc:
        print(f"Camera {args.camera}: failed ({exc})")
    finally:
        try:
            cap.release()
        except Exception:
            pass

    if not camera_ok:
        print("Camera hint: check Windows Settings > Privacy & security > Camera.")
        print("Camera hint: enable camera access for desktop apps and close apps already using the camera.")

    return 0 if py_ok and import_ok and screen_ok and camera_ok else 1
