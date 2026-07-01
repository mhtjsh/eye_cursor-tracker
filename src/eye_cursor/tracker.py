from __future__ import annotations

from dataclasses import dataclass

import cv2

from .features import FeatureResult, LandmarkPoint, extract_features


@dataclass(frozen=True)
class TrackingResult:
    features: FeatureResult
    landmark_xy: list[tuple[float, float]]


class FaceTracker:
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed. Use Python 3.11/3.12 and run: "
                "python -m pip install -e ."
            ) from exc

        self._mp = mp
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self._mesh.close()

    def __enter__(self) -> "FaceTracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def track_bgr(self, frame_bgr) -> TrackingResult | None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        face = result.multi_face_landmarks[0]
        landmarks = [LandmarkPoint(float(lm.x), float(lm.y), float(lm.z)) for lm in face.landmark]
        features = extract_features(landmarks)
        xy = [(float(lm.x), float(lm.y)) for lm in face.landmark]
        return TrackingResult(features=features, landmark_xy=xy)


def open_camera(index: int, width: int, height: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap
