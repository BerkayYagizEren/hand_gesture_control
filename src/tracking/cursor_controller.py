"""Maps the tracked hand to a cursor position with relative, accelerated motion.

Each frame's anchor displacement is scaled by a speed-dependent gain and added to
the cursor position, so slow hand motion is precise and fast motion covers the
screen. This class only computes positions: the caller picks the move or drag
event, since that depends on gesture state.

Clutch and pinch freeze drop the hand anchor and the filter state, and the first
frame after them re-anchors. Keeping the filter running instead would replay its
lag as a slide after release, which turns into a drag if a button is held.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from src.controller.base import Point, ScreenBounds
from src.tracking.one_euro import OneEuroFilter2D
from src.vision.detector import HandLandmark, Landmark


@dataclass(frozen=True)
class CursorConfig:
    # Gain is in screen points per image height of hand travel.
    min_gain: float = 800.0
    max_gain: float = 4000.0
    # Hand speed, in image heights per second, over which gain ramps from min to max.
    low_speed: float = 0.1
    high_speed: float = 1.5
    # Filter parameters for anchor coordinates in image heights.
    min_cutoff: float = 1.0
    beta: float = 10.0
    d_cutoff: float = 1.0
    pinch_freeze_seconds: float = 0.12
    # A longer frame gap means tracking was lost, so the hand is re-anchored.
    max_gap_seconds: float = 0.2

    def __post_init__(self) -> None:
        if not 0 < self.min_gain <= self.max_gain:
            raise ValueError("gains must satisfy 0 < min_gain <= max_gain")
        if not 0 <= self.low_speed < self.high_speed:
            raise ValueError("speeds must satisfy 0 <= low_speed < high_speed")
        if self.pinch_freeze_seconds < 0 or self.max_gap_seconds <= 0:
            raise ValueError("pinch_freeze_seconds and max_gap_seconds are out of range")

    def gain(self, speed: float) -> float:
        """Return the gain for a hand speed, easing between the limits with a smoothstep."""
        t = min(max((speed - self.low_speed) / (self.high_speed - self.low_speed), 0.0), 1.0)
        return self.min_gain + (self.max_gain - self.min_gain) * t * t * (3.0 - 2.0 * t)


class CursorController:
    """Turns timestamped hand landmarks into a clamped screen position."""

    def __init__(
        self,
        bounds: ScreenBounds,
        start: Point | None = None,
        config: CursorConfig = CursorConfig(),
    ) -> None:
        self.bounds = bounds
        self.config = config
        self.clutch = False  # While True the cursor stays put and the hand can reposition.
        self._position = bounds.clamp(*(bounds.center if start is None else start))
        self._filter = OneEuroFilter2D(config.min_cutoff, config.beta, config.d_cutoff)
        self._last_anchor: tuple[float, float] | None = None
        self._last_time = 0.0
        self._freeze_until: float | None = None

    @property
    def position(self) -> Point:
        return self._position

    def start_pinch_freeze(self, timestamp: float) -> None:
        """Hold the cursor for the freeze window; call when a pinch starts."""
        self._freeze_until = timestamp + self.config.pinch_freeze_seconds

    def is_frozen(self, timestamp: float) -> bool:
        return self.clutch or (self._freeze_until is not None and timestamp < self._freeze_until)

    def reset(self, position: Point | None = None) -> None:
        """Drop hand tracking and any freeze, on hand loss or when taking over the cursor.

        The clutch is left alone because the gesture layer owns it. Pass the OS
        cursor position if the physical mouse may have moved.
        """
        self._drop_anchor()
        self._freeze_until = None
        if position is not None:
            self._position = self.bounds.clamp(*position)

    def update(
        self, landmarks: Sequence[Landmark], timestamp: float, aspect_ratio: float = 1.0
    ) -> Point:
        """Advance with a new frame and return the cursor position.

        `landmarks` come from a mirrored image, so image +x matches the user's
        rightward motion. `aspect_ratio` (`DetectionResult.aspect_ratio`) makes
        both axes the same scale. `timestamp` is in seconds.
        """
        if self.is_frozen(timestamp):
            self._drop_anchor()
            return self._position

        gap = timestamp - self._last_time
        if self._last_anchor is not None and gap > self.config.max_gap_seconds:
            self._drop_anchor()

        anchor = landmarks[HandLandmark.INDEX_FINGER_MCP]
        x, y = self._filter.filter(anchor.x * aspect_ratio, anchor.y, timestamp)
        last_anchor, last_time = self._last_anchor, self._last_time
        self._last_anchor = (x, y)
        self._last_time = timestamp

        # The clamped position is stored, so pushing past an edge builds no hidden
        # overshoot that the hand would have to undo before the cursor comes back.
        if last_anchor is not None:
            dx, dy = x - last_anchor[0], y - last_anchor[1]
            gain = self.config.gain(math.hypot(dx, dy) / (timestamp - last_time))
            self._position = self.bounds.clamp(
                self._position.x + dx * gain, self._position.y + dy * gain
            )
        return self._position

    def _drop_anchor(self) -> None:
        self._filter.reset()
        self._last_anchor = None
