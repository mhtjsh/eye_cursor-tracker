from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Sequence

import numpy as np


LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
HEAD = [1, 4, 10, 152, 61, 291, 199, 234, 454, 127, 356]
SELECTED_INDICES = list(dict.fromkeys(LEFT_EYE + RIGHT_EYE + LEFT_IRIS + RIGHT_IRIS + HEAD))


@dataclass(frozen=True)
class LandmarkPoint:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class FeatureResult:
    vector: np.ndarray
    face_box: tuple[float, float, float, float]
    quality: dict[str, float]


def _point(landmarks: Sequence[LandmarkPoint], index: int) -> LandmarkPoint | None:
    if 0 <= index < len(landmarks):
        return landmarks[index]
    return None


def _distance(a: LandmarkPoint | None, b: LandmarkPoint | None) -> float:
    if a is None or b is None:
        return 0.0
    return float(hypot(a.x - b.x, a.y - b.y))


def _mean_point(landmarks: Sequence[LandmarkPoint], indices: Sequence[int]) -> LandmarkPoint | None:
    pts = [_point(landmarks, idx) for idx in indices]
    pts = [pt for pt in pts if pt is not None]
    if not pts:
        return None
    return LandmarkPoint(
        float(np.mean([pt.x for pt in pts])),
        float(np.mean([pt.y for pt in pts])),
        float(np.mean([pt.z for pt in pts])),
    )


def _relative_scalar(value: float, center: float, scale: float) -> float:
    return (value - center) / scale if scale > 1e-6 else 0.0


def extract_features(landmarks: Sequence[LandmarkPoint]) -> FeatureResult:
    if not landmarks:
        raise ValueError("No landmarks were provided.")

    xs = np.asarray([lm.x for lm in landmarks], dtype=np.float32)
    ys = np.asarray([lm.y for lm in landmarks], dtype=np.float32)
    zs = np.asarray([lm.z for lm in landmarks], dtype=np.float32)

    min_x = float(xs.min())
    max_x = float(xs.max())
    min_y = float(ys.min())
    max_y = float(ys.max())
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)
    scale = max(width, height)
    z_center = float(zs.mean())

    feats: list[float] = [
        center_x,
        center_y,
        width,
        height,
        width / max(height, 1e-6),
    ]

    for idx in SELECTED_INDICES:
        pt = _point(landmarks, idx)
        if pt is None:
            feats.extend([0.0, 0.0, 0.0, 0.0])
            continue
        feats.extend(
            [
                _relative_scalar(pt.x, center_x, scale),
                _relative_scalar(pt.y, center_y, scale),
                _relative_scalar(pt.z, z_center, scale),
                1.0,
            ]
        )

    left_width = _distance(_point(landmarks, 33), _point(landmarks, 133))
    right_width = _distance(_point(landmarks, 263), _point(landmarks, 362))
    left_open = _distance(_point(landmarks, 159), _point(landmarks, 145)) / max(left_width, 1e-6)
    right_open = _distance(_point(landmarks, 386), _point(landmarks, 374)) / max(right_width, 1e-6)

    left_iris = _mean_point(landmarks, LEFT_IRIS)
    right_iris = _mean_point(landmarks, RIGHT_IRIS)
    left_inner = _point(landmarks, 133)
    left_outer = _point(landmarks, 33)
    right_inner = _point(landmarks, 362)
    right_outer = _point(landmarks, 263)

    def iris_ratio(iris: LandmarkPoint | None, outer: LandmarkPoint | None, inner: LandmarkPoint | None) -> tuple[float, float]:
        if iris is None or outer is None or inner is None:
            return 0.0, 0.0
        eye_width = max(_distance(outer, inner), 1e-6)
        return (iris.x - outer.x) / eye_width, (iris.y - outer.y) / eye_width

    left_iris_x, left_iris_y = iris_ratio(left_iris, left_outer, left_inner)
    right_iris_x, right_iris_y = iris_ratio(right_iris, right_outer, right_inner)

    nose = _point(landmarks, 1)
    left_cheek = _point(landmarks, 234)
    right_cheek = _point(landmarks, 454)
    yaw_proxy = 0.0
    if nose is not None and left_cheek is not None and right_cheek is not None:
        yaw_proxy = (_distance(nose, left_cheek) - _distance(nose, right_cheek)) / max(
            _distance(left_cheek, right_cheek), 1e-6
        )

    feats.extend(
        [
            left_open,
            right_open,
            left_iris_x,
            left_iris_y,
            right_iris_x,
            right_iris_y,
            yaw_proxy,
        ]
    )

    quality = {
        "face_width": width,
        "face_height": height,
        "left_eye_open": float(left_open),
        "right_eye_open": float(right_open),
        "yaw_proxy": float(yaw_proxy),
    }
    return FeatureResult(
        vector=np.asarray(feats, dtype=np.float32),
        face_box=(min_x, min_y, width, height),
        quality=quality,
    )


def feature_dim() -> int:
    return 5 + len(SELECTED_INDICES) * 4 + 7
