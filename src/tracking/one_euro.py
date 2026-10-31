"""One-Euro filter (Casiez et al., CHI 2012): adaptive low-pass for noisy live input.

The cutoff rises with signal speed, trading jitter at rest for low lag in motion:

    alpha(fc, dt) = 1 / (1 + tau / dt),  tau = 1 / (2 * pi * fc)
    dx  = (x - x_raw_prev) / dt
    edx = alpha(d_cutoff, dt) * dx + (1 - alpha(d_cutoff, dt)) * edx_prev
    fc  = min_cutoff + beta * |edx|
    x^  = alpha(fc, dt) * x + (1 - alpha(fc, dt)) * x^_prev

dt comes from consecutive timestamps (seconds), so no frame rate is assumed.
"""

from __future__ import annotations

import math


def smoothing_factor(cutoff: float, dt: float) -> float:
    """Return the low-pass alpha for a cutoff in Hz over a step of dt seconds."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Filters one scalar signal."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("min_cutoff and d_cutoff must be positive")
        if beta < 0:
            raise ValueError("beta must not be negative")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        """Forget the signal; the next sample passes through unfiltered."""
        self._last_time: float | None = None
        self._last_raw = 0.0
        self._last_filtered = 0.0
        self._last_derivative = 0.0

    def filter(self, value: float, timestamp: float) -> float:
        """Return the filtered value. Timestamps must be strictly increasing."""
        if not math.isfinite(value):
            raise ValueError(f"non-finite sample {value}")
        if self._last_time is None:
            self._last_time = timestamp
            self._last_raw = value
            self._last_filtered = value
            return value
        dt = timestamp - self._last_time
        if dt <= 0:
            raise ValueError(f"timestamps must be strictly increasing, got dt={dt}")

        d_alpha = smoothing_factor(self.d_cutoff, dt)
        derivative = (value - self._last_raw) / dt
        derivative = d_alpha * derivative + (1.0 - d_alpha) * self._last_derivative

        alpha = smoothing_factor(self.min_cutoff + self.beta * abs(derivative), dt)
        filtered = alpha * value + (1.0 - alpha) * self._last_filtered

        self._last_time = timestamp
        self._last_raw = value
        self._last_filtered = filtered
        self._last_derivative = derivative
        return filtered


class OneEuroFilter2D:
    """Filters an (x, y) signal with one independent filter per axis."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self._x = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._y = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def reset(self) -> None:
        self._x.reset()
        self._y.reset()

    def filter(self, x: float, y: float, timestamp: float) -> tuple[float, float]:
        if not math.isfinite(y):  # Checked up front so a bad y cannot desync the axes.
            raise ValueError(f"non-finite sample {y}")
        return self._x.filter(x, timestamp), self._y.filter(y, timestamp)
