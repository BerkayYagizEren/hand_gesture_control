"""Detects hand swipes in four directions from voted poses and hand motion.

DISARMED -> ARMED -> (swipe) -> COOLDOWN -> DISARMED

Arming takes the armed pose held still for a moment. The stillness matters for
the return movement: pulling the hand back keeps the same pose and can outlast the
cooldown, but it is motion, so it cannot arm and cannot fire a reverse swipe.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto

from src.vision.classifier import Pose
from src.vision.detector import HandLandmark, Landmark


class SwipeDirection(Enum):
    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()


class SwipePhase(Enum):
    DISARMED = auto()
    ARMED = auto()
    COOLDOWN = auto()


@dataclass(frozen=True)
class SwipeConfig:
    armed_pose: Pose = Pose.OPEN_PALM
    # Distances are in image heights, speeds in image heights per second.
    min_distance: float = 0.15
    min_speed: float = 0.8
    # The longer axis must exceed this many times the shorter, which rejects diagonal motion.
    dominance_ratio: float = 2.0
    window_seconds: float = 0.35  # A swipe must fit in this long.
    cooldown_seconds: float = 0.6
    # To arm, the hand stays within the radius for the hold time.
    arm_hold_seconds: float = 0.15
    arm_radius: float = 0.03
    # A longer frame gap means tracking was lost, so motion across it is not trusted.
    max_gap_seconds: float = 0.2

    def __post_init__(self) -> None:
        if min(self.min_distance, self.min_speed, self.dominance_ratio) <= 0:
            raise ValueError("min_distance, min_speed and dominance_ratio must be positive")
        if self.window_seconds <= 0 or self.arm_radius <= 0 or self.max_gap_seconds <= 0:
            raise ValueError("window_seconds, arm_radius and max_gap_seconds must be positive")
        if self.cooldown_seconds < 0 or self.arm_hold_seconds < 0:
            raise ValueError("cooldown_seconds and arm_hold_seconds must not be negative")


class SwipeDetector:
    """Reports each swipe once. Call `reset` on hand loss."""

    def __init__(self, config: SwipeConfig = SwipeConfig()) -> None:
        self.config = config
        self._phase = SwipePhase.DISARMED
        self._history: deque[tuple[float, float, float]] = deque()  # (time, x, y)
        self._hold_start: tuple[float, float, float] | None = None
        self._cooldown_until = 0.0
        self._last_time: float | None = None

    @property
    def phase(self) -> SwipePhase:
        return self._phase

    def reset(self) -> None:
        """Forget the hand and require arming again."""
        self._phase = SwipePhase.DISARMED
        self._history.clear()
        self._hold_start = None
        self._last_time = None

    def update(
        self,
        pose: Pose,
        landmarks: Sequence[Landmark],
        timestamp: float,
        aspect_ratio: float = 1.0,
    ) -> SwipeDirection | None:
        """Advance one frame and return a swipe if one just completed.

        `pose` should be the voted pose, since motion blur breaks single-frame poses
        mid-swipe. `landmarks` come from a mirrored image, so image +x is the user's
        right, and image y grows downward. `aspect_ratio`
        (`DetectionResult.aspect_ratio`) makes both axes the same scale. `timestamp` is in seconds and must strictly increase.
        """
        anchor = landmarks[HandLandmark.INDEX_FINGER_MCP]
        x, y = anchor.x * aspect_ratio, anchor.y
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"non-finite anchor ({x}, {y})")
        if self._last_time is not None:
            gap = timestamp - self._last_time
            if gap <= 0:
                raise ValueError(f"timestamps must be strictly increasing, got dt={gap}")
            if gap > self.config.max_gap_seconds:
                self._drop_motion()
        self._last_time = timestamp

        if self._phase is SwipePhase.COOLDOWN:
            if timestamp < self._cooldown_until:
                return None
            self._phase = SwipePhase.DISARMED

        if pose is not self.config.armed_pose:
            self._drop_motion()
            return None
        if self._phase is SwipePhase.DISARMED:
            self._step_disarmed(timestamp, x, y)
            return None
        return self._step_armed(timestamp, x, y)

    def _step_disarmed(self, timestamp: float, x: float, y: float) -> None:
        hold = self._hold_start
        if hold is None or math.dist((x, y), hold[1:]) > self.config.arm_radius:
            self._hold_start = (timestamp, x, y)
        hold = self._hold_start
        if timestamp - hold[0] >= self.config.arm_hold_seconds:
            self._phase = SwipePhase.ARMED
            self._hold_start = None
            self._history.append((timestamp, x, y))

    def _step_armed(self, timestamp: float, x: float, y: float) -> SwipeDirection | None:
        window_start = timestamp - self.config.window_seconds
        while self._history and self._history[0][0] < window_start:
            self._history.popleft()

        # Every start point is tried, so a swipe that follows a pause or a stray
        # motion on the other axis inside the window is still judged on its own stroke.
        for start_time, start_x, start_y in self._history:
            dx, dy = x - start_x, y - start_y
            major, minor = max(abs(dx), abs(dy)), min(abs(dx), abs(dy))
            if (
                major >= self.config.min_distance
                and major > self.config.dominance_ratio * minor
                and major / (timestamp - start_time) >= self.config.min_speed
            ):
                self._history.clear()
                self._phase = SwipePhase.COOLDOWN
                self._cooldown_until = timestamp + self.config.cooldown_seconds
                if abs(dx) > abs(dy):
                    return SwipeDirection.RIGHT if dx > 0 else SwipeDirection.LEFT
                return SwipeDirection.UP if dy < 0 else SwipeDirection.DOWN

        self._history.append((timestamp, x, y))
        return None

    def _drop_motion(self) -> None:
        self._history.clear()
        self._hold_start = None
        if self._phase is SwipePhase.ARMED:
            self._phase = SwipePhase.DISARMED
