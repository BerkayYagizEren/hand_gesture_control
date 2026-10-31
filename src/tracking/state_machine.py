"""Turns voted hand poses into mouse actions, with guaranteed button release.

Layers: landmarks -> `PoseClassifier` -> `TemporalVotingFilter` -> this FSM -> mouse.
Feed it with `process` (one `DetectionResult` per frame), or `update` for a frame
that has already been classified. Call it on frames without a hand too, and when
the camera times out, because the hand-loss timeout only fires inside a call.

A pinch presses the button as soon as it is recognized. The cursor then stays put
until the press becomes a drag, so a click cannot slide; releasing before that
snaps the internal cursor back to the press point.

While scrolling, the voted pose decides when the mode ends, but only frames that
are themselves TWO_FINGER scroll, so changing pose does not nudge the page.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto

from src.controller.base import BaseMouseController
from src.controller.scroll import ScrollController
from src.tracking.cursor_controller import CursorController
from src.vision.classifier import (
    PINCH_START_RATIO,
    PoseClassifier,
    PoseObservation,
    Pose,
    TemporalVotingFilter,
)
from src.vision.detector import DetectionResult, Landmark


class GestureState(Enum):
    IDLE = auto()  # No hand.
    TRACKING = auto()
    PINCH_ACTIVE = auto()
    RIGHT_CLICK_ACTIVE = auto()
    SCROLL_ACTIVE = auto()
    CLUTCH_ACTIVE = auto()
    COOLDOWN = auto()


@dataclass(frozen=True)
class GestureConfig:
    pinch_start_ratio: float = PINCH_START_RATIO
    pinch_release_ratio: float = 0.70
    # Consecutive frames past the release ratio, so one noisy frame cannot drop a drag.
    release_frames: int = 2
    drag_hold_ms: float = 250.0
    drag_move_threshold: float = 8.0  # Screen points from the press point.
    hand_loss_timeout_ms: float = 200.0
    cooldown_ms: float = 150.0

    def __post_init__(self) -> None:
        if not 0 < self.pinch_start_ratio < self.pinch_release_ratio:
            raise ValueError("ratios must satisfy 0 < pinch_start_ratio < pinch_release_ratio")
        if self.release_frames < 1:
            raise ValueError("release_frames must be at least 1")
        if min(self.drag_hold_ms, self.drag_move_threshold, self.cooldown_ms) < 0:
            raise ValueError("drag_hold_ms, drag_move_threshold and cooldown_ms must not be negative")
        if self.hand_loss_timeout_ms <= 0:
            raise ValueError("hand_loss_timeout_ms must be positive")


@dataclass(frozen=True)
class GestureFrame:
    observation: PoseObservation  # Unvoted; the state machine does the voting.
    landmarks: Sequence[Landmark]  # Image landmarks, for the cursor.
    aspect_ratio: float = 1.0


class GestureStateMachine:
    """Drives `mouse` and `cursor`. Use as a context manager to release on shutdown.

    Outside PINCH_ACTIVE and RIGHT_CLICK_ACTIVE no button is ever held. Any exception
    during a frame resets the machine before propagating.
    """

    def __init__(
        self,
        mouse: BaseMouseController,
        cursor: CursorController,
        config: GestureConfig = GestureConfig(),
        classifier: PoseClassifier | None = None,
        voting: TemporalVotingFilter | None = None,
        scroll: ScrollController | None = None,
    ) -> None:
        self.mouse = mouse
        self.cursor = cursor
        self.config = config
        self.classifier = classifier or PoseClassifier(config.pinch_start_ratio)
        self.voting = voting or TemporalVotingFilter()
        self.scroll = scroll or ScrollController(mouse)
        self._state = GestureState.IDLE
        self._last_hand_time = 0.0
        self._cooldown_until = 0.0
        self._press_time = 0.0
        self._release_count = 0
        self._dragging = False

    @property
    def state(self) -> GestureState:
        return self._state

    @property
    def is_dragging(self) -> bool:
        return self._dragging

    def process(self, result: DetectionResult) -> GestureState:
        """Advance with one detector result, using its first hand if there are several."""
        try:
            frame = None
            if result.hands:
                hand = result.hands[0]
                frame = GestureFrame(
                    self.classifier.classify(hand), hand.landmarks, result.aspect_ratio
                )
        except BaseException:
            self.reset()
            raise
        return self.update(frame, result.timestamp_ms / 1000)

    def update(self, frame: GestureFrame | None, timestamp: float) -> GestureState:
        """Advance one frame. `frame` is None if no hand was seen; `timestamp` is in seconds."""
        try:
            self._step(frame, timestamp)
        except BaseException:
            self.reset()
            raise
        return self._state

    def reset(self) -> None:
        """Return to IDLE and release every button. Call on shutdown or any abort."""
        self._state = GestureState.IDLE
        self._dragging = False
        self.voting.reset()
        self.cursor.clutch = False
        self.cursor.reset()
        self.scroll.reset()
        self.mouse.ensure_all_released()

    def __enter__(self) -> GestureStateMachine:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.reset()

    def _step(self, frame: GestureFrame | None, timestamp: float) -> None:
        # A failed release leaves the button held after the reset, so keep retrying it.
        if (
            self._state not in (GestureState.PINCH_ACTIVE, GestureState.RIGHT_CLICK_ACTIVE)
            and self.mouse.held_buttons
        ):
            self.mouse.ensure_all_released()

        # Checked on hand frames too: a stalled camera shows up as a long gap between them.
        timeout = self.config.hand_loss_timeout_ms / 1000
        if self._state is not GestureState.IDLE and timestamp - self._last_hand_time >= timeout:
            self.reset()
        if frame is None:
            return

        self._last_hand_time = timestamp
        pose = self.voting.update(frame.observation.pose)
        if self._state is GestureState.IDLE:
            # The physical mouse may have moved while gesture control was idle.
            self.cursor.reset(self.mouse.sync_position())
            self._state = GestureState.TRACKING

        if self._state is GestureState.COOLDOWN:
            if timestamp < self._cooldown_until:
                self._move_cursor(frame, timestamp)
                return
            self._state = GestureState.TRACKING

        match self._state:
            case GestureState.TRACKING:
                self._step_tracking(frame, pose, timestamp)
            case GestureState.PINCH_ACTIVE:
                self._step_pinch(frame, timestamp)
            case GestureState.RIGHT_CLICK_ACTIVE:
                if self._should_release(frame.observation.middle_pinch_ratio):
                    self.mouse.release_right()
                    self._finish_action(timestamp)
            case GestureState.SCROLL_ACTIVE:
                if pose is not Pose.TWO_FINGER:
                    self.scroll.reset()
                    self._finish_action(timestamp)
                elif frame.observation.pose is Pose.TWO_FINGER:
                    self.scroll.update(frame.landmarks, timestamp)
            case GestureState.CLUTCH_ACTIVE:
                if pose is not Pose.FIST:
                    self.cursor.clutch = False
                    self._state = GestureState.TRACKING
                    self._move_cursor(frame, timestamp)

    def _step_tracking(self, frame: GestureFrame, pose: Pose, timestamp: float) -> None:
        # The live ratio is checked as well, because the vote still says PINCH for a
        # few frames after a release and must not re-press.
        observation = frame.observation
        if pose is Pose.PINCH and observation.pinch_ratio < self.config.pinch_start_ratio:
            self.mouse.press_left()
            self.cursor.start_pinch_freeze(timestamp)
            self._press_time = timestamp
            self._release_count = 0
            self._state = GestureState.PINCH_ACTIVE
        elif (
            pose is Pose.MIDDLE_PINCH
            and observation.middle_pinch_ratio < self.config.pinch_start_ratio
        ):
            self.mouse.press_right()
            self.cursor.reset()
            self._release_count = 0
            self._state = GestureState.RIGHT_CLICK_ACTIVE
        elif pose is Pose.TWO_FINGER:
            self.cursor.reset()
            self.scroll.reset()
            self.scroll.update(frame.landmarks, timestamp)
            self._state = GestureState.SCROLL_ACTIVE
        elif pose is Pose.FIST:
            self.cursor.clutch = True
            self.cursor.reset()
            self._state = GestureState.CLUTCH_ACTIVE
        else:
            self._move_cursor(frame, timestamp)

    def _step_pinch(self, frame: GestureFrame, timestamp: float) -> None:
        if self._should_release(frame.observation.pinch_ratio):
            self.mouse.release_left()
            self._dragging = False
            self._finish_action(timestamp)
            return
        position = self.cursor.update(frame.landmarks, timestamp, frame.aspect_ratio)
        if not self._dragging:
            held_for = (timestamp - self._press_time) * 1000
            moved = math.dist(position, self.mouse.position)
            self._dragging = (
                held_for >= self.config.drag_hold_ms or moved > self.config.drag_move_threshold
            )
        if self._dragging and position != self.mouse.position:
            self.mouse.drag_to(*position)

    def _should_release(self, ratio: float) -> bool:
        # Written so a nan ratio counts toward release: when unsure, let go.
        if ratio <= self.config.pinch_release_ratio:
            self._release_count = 0
            return False
        self._release_count += 1
        return self._release_count >= self.config.release_frames

    def _finish_action(self, timestamp: float) -> None:
        self.cursor.reset(self.mouse.position)
        self._cooldown_until = timestamp + self.config.cooldown_ms / 1000
        self._state = GestureState.COOLDOWN

    def _move_cursor(self, frame: GestureFrame, timestamp: float) -> None:
        position = self.cursor.update(frame.landmarks, timestamp, frame.aspect_ratio)
        if position != self.mouse.position:
            self.mouse.move_to(*position)
