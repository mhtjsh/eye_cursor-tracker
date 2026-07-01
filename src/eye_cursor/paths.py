from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProfilePaths:
    workspace: Path
    profile: str

    @property
    def profile_dir(self) -> Path:
        return self.workspace / "profiles" / self.profile

    @property
    def captures_dir(self) -> Path:
        return self.profile_dir / "captures"

    @property
    def latest_calibration_marker(self) -> Path:
        return self.profile_dir / "latest_calibration.txt"

    @property
    def model_path(self) -> Path:
        return self.profile_dir / "model.joblib"

    @property
    def onnx_model_path(self) -> Path:
        return self.profile_dir / "model.onnx"

    @property
    def onnx_metadata_path(self) -> Path:
        return self.profile_dir / "model_metadata.json"

    @property
    def metrics_path(self) -> Path:
        return self.profile_dir / "metrics.json"

    def ensure_dirs(self, save_frames: bool = False) -> None:
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        if save_frames:
            self.captures_dir.mkdir(parents=True, exist_ok=True)


def profile_paths(workspace: str | Path, profile: str) -> ProfilePaths:
    return ProfilePaths(Path(workspace).expanduser().resolve(), profile)
