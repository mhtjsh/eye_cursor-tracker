from __future__ import annotations

from dataclasses import dataclass
from math import pi


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


@dataclass
class LowPassFilter:
    value: float | None = None

    def filter(self, value: float, alpha: float) -> float:
        if self.value is None:
            self.value = value
        else:
            self.value = alpha * value + (1.0 - alpha) * self.value
        return self.value


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 0.7, beta: float = 0.05, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = LowPassFilter()
        self._dx = LowPassFilter()
        self._last_time: float | None = None

    def reset(self) -> None:
        self._x = LowPassFilter()
        self._dx = LowPassFilter()
        self._last_time = None

    def filter(self, value: float, timestamp: float) -> float:
        if self._last_time is None:
            self._last_time = timestamp
            return self._x.filter(value, 1.0)

        dt = max(timestamp - self._last_time, 1e-3)
        self._last_time = timestamp

        previous = self._x.value if self._x.value is not None else value
        dx = (value - previous) / dt
        edx = self._dx.filter(dx, _alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x.filter(value, _alpha(cutoff, dt))


class PointSmoother:
    def __init__(self, min_cutoff: float = 0.7, beta: float = 0.05) -> None:
        self.x_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.y_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)

    def reset(self) -> None:
        self.x_filter.reset()
        self.y_filter.reset()

    def filter(self, x: float, y: float, timestamp: float) -> tuple[float, float]:
        return self.x_filter.filter(x, timestamp), self.y_filter.filter(y, timestamp)
