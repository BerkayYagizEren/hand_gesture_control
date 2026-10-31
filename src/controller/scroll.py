"""Turns vertical hand motion into smooth pixel scrolling.

The filtered anchor drags a deadzone window along: motion inside the window does
nothing, and only travel past its edge scrolls. Jitter narrower than the window
cannot keep scrolling: a hand stopping at the edge after a move may nudge the page
once by the jitter's amplitude, then stays inside. Reversing first crosses the window.
Fractions of a pixel carry over to later frames, so slow motion is not lost.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from src.controller.base import BaseMouseController
from src.tracking.one_euro import OneEuroFilter
from src.vision.detector import HandLandmark, Landmark


@dataclass(frozen=True)
class ScrollConfig:
    speed_multiplier: float = 2000.0  # Pixels per image height of hand travel.
    deadzone: float = 0.008  # Half width of the window, in image heights.
    # True: content follows the hand, as on a trackpad with natural scrolling.
    natural: bool = True
    # Filter parameters for an anchor coordinate in image heights.
    min_cutoff: float = 1.0
    beta: float = 10.0
    d_cutoff: float = 1.0
    # A longer frame gap means tracking was lost, so the hand is re-anchored.
    max_gap_seconds: float = 0.2

    def __post_init__(self) -> None:
        if self.speed_multiplier <= 0 or self.max_gap_seconds <= 0:
            raise ValueError("speed_multiplier and max_gap_seconds must be positive")
        if self.deadzone < 0:
            raise ValueError("deadzone must not be negative")


class ScrollController:
    """Posts scroll events to `mouse` from timestamped hand landmarks."""

    def __init__(self, mouse: BaseMouseController, config: ScrollConfig = ScrollConfig()) -> None:
        self.mouse = mouse
        self.config = config
        self._filter = OneEuroFilter(config.min_cutoff, config.beta, config.d_cutoff)
        self._window_center: float | None = None
        self._remainder = 0.0
        self._last_time = 0.0

    def reset(self) -> None:
        """Drop hand tracking; the next frame only re-anchors. Call when scrolling starts or ends."""
        self._filter.reset()
        self._window_center = None
        self._remainder = 0.0

    def update(self, landmarks: Sequence[Landmark], timestamp: float) -> int:
        """Advance with a new frame and return the pixels scrolled. `timestamp` is in seconds."""
        if self._window_center is not None and timestamp - self._last_time > self.config.max_gap_seconds:
            self.reset()

        y = self._filter.filter(landmarks[HandLandmark.INDEX_FINGER_MCP].y, timestamp)
        self._last_time = timestamp
        if self._window_center is None:
            self._window_center = y
            return 0

        travel = y - self._window_center
        if abs(travel) <= self.config.deadzone:
            return 0
        moved = travel - math.copysign(self.config.deadzone, travel)
        self._window_center += moved

        # Image y grows downward and +dy scrolls up, so hand up gives content up: same sign.
        sign = 1.0 if self.config.natural else -1.0
        self._remainder += sign * moved * self.config.speed_multiplier
        pixels = int(self._remainder)
        if pixels == 0:
            return 0
        self._remainder -= pixels
        self.mouse.scroll(0, pixels)
        return pixels
