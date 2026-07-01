from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import feature_dim
from .paths import profile_paths


def _latest_calibration(paths) -> Path:
    marker = paths.latest_calibration_marker
    if marker.exists():
        candidate = paths.profile_dir / marker.read_text(encoding="utf-8").strip()
        if candidate.exists():
            return candidate

    candidates = sorted(paths.profile_dir.glob("calibration_*.npz"))
    if not candidates:
        raise RuntimeError(f"No calibration files found in {paths.profile_dir}. Run calibrate first.")
    return candidates[-1]


def _pixel_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.linalg.norm(y_pred - y_true, axis=1)


def run_training(args) -> int:
    paths = profile_paths(args.workspace, args.profile)
    paths.ensure_dirs()

    calibration_path = Path(args.calibration).expanduser().resolve() if args.calibration else _latest_calibration(paths)
    data = np.load(calibration_path, allow_pickle=False)
    x = data["features"].astype(np.float32)
    y = data["targets"].astype(np.float32)
    screen_size = tuple(int(v) for v in data["screen_size"])

    if len(x) < 100:
        raise RuntimeError(f"Need at least 100 calibration samples, found {len(x)}.")
    if x.shape[1] != feature_dim():
        raise RuntimeError(f"Feature dimension mismatch: data has {x.shape[1]}, code expects {feature_dim()}.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
    )

    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-2, 4, 15)),
    )
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    train_err = _pixel_errors(y_train, pred_train)
    test_err = _pixel_errors(y_test, pred_test)

    metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "profile": args.profile,
        "calibration_file": str(calibration_path),
        "sample_count": int(len(x)),
        "feature_dim": int(x.shape[1]),
        "screen_size": [int(screen_size[0]), int(screen_size[1])],
        "train_median_px": float(np.median(train_err)),
        "train_p90_px": float(np.percentile(train_err, 90)),
        "test_median_px": float(np.median(test_err)),
        "test_p90_px": float(np.percentile(test_err, 90)),
        "test_mae_x_px": float(mean_absolute_error(y_test[:, 0], pred_test[:, 0])),
        "test_mae_y_px": float(mean_absolute_error(y_test[:, 1], pred_test[:, 1])),
    }

    artifact = {
        "model": model,
        "metadata": metrics,
    }
    joblib.dump(artifact, paths.model_path)
    paths.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Calibration: {calibration_path}")
    print(f"Saved model: {paths.model_path}")
    print(f"Samples: {len(x)}")
    print(f"Train median/p90: {metrics['train_median_px']:.1f}px / {metrics['train_p90_px']:.1f}px")
    print(f"Test median/p90: {metrics['test_median_px']:.1f}px / {metrics['test_p90_px']:.1f}px")
    print(f"Test MAE x/y: {metrics['test_mae_x_px']:.1f}px / {metrics['test_mae_y_px']:.1f}px")
    print("Next: eye-cursor run --profile", args.profile)
    return 0
