"""Per-frame static pose classification and N-of-M temporal voting over its output."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum, auto

from src.vision.detector import Hand, HandLandmark
from src.vision.geometry import (
    Finger,
    FingerState,
    direction,
    finger_states,
    middle_pinch_ratio,
    pinch_ratio,
)

PINCH_START_RATIO = 0.55
# Thumb direction's upward component (image y is down); 0.7 is about 45 degrees off vertical.
THUMBS_UP_MIN_COMPONENT = 0.7


class Pose(Enum):
    UNKNOWN = auto()
    POINTING = auto()
    PINCH = auto()
    MIDDLE_PINCH = auto()
    FIST = auto()  # Drives the clutch.
    TWO_FINGER = auto()
    OPEN_PALM = auto()
    THUMBS_UP = auto()


@dataclass(frozen=True)
class PoseObservation:
    """One frame's pose, plus the raw ratios the state machine needs for hysteresis."""

    pose: Pose
    pinch_ratio: float
    middle_pinch_ratio: float


class PoseClassifier:
    """Classifies a single frame from world landmarks, with no memory between frames."""

    def __init__(self, pinch_start_ratio: float = PINCH_START_RATIO) -> None:
        self.pinch_start_ratio = pinch_start_ratio

    def classify(self, hand: Hand) -> PoseObservation:
        world = hand.world_landmarks
        index_ratio = pinch_ratio(world)
        middle_ratio = middle_pinch_ratio(world)
        return PoseObservation(
            self._pose(hand, index_ratio, middle_ratio), index_ratio, middle_ratio
        )

    def _pose(self, hand: Hand, index_ratio: float, middle_ratio: float) -> Pose:
        states = finger_states(hand.world_landmarks)
        extended = {finger for finger, state in states.items() if state is FingerState.EXTENDED}
        four_fingers = (Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY)
        all_four_curled = all(states[finger] is FingerState.CURLED for finger in four_fingers)

        # A fist has all four fingers curled into the palm.
        if all_four_curled and states[Finger.THUMB] is not FingerState.EXTENDED:
            return Pose.FIST

        # Pinch detection: when not a full fist, distance between tips defines the pinch.
        is_pinch = index_ratio < self.pinch_start_ratio and (
            states[Finger.INDEX] is not FingerState.CURLED
        )
        is_middle_pinch = middle_ratio < self.pinch_start_ratio and (
            states[Finger.MIDDLE] is not FingerState.CURLED
        )
        if is_pinch and is_middle_pinch:
            return Pose.PINCH if index_ratio <= middle_ratio else Pose.MIDDLE_PINCH
        if is_pinch:
            return Pose.PINCH
        if is_middle_pinch:
            return Pose.MIDDLE_PINCH

        if all_four_curled:
            thumb = direction(hand.world_landmarks, HandLandmark.THUMB_MCP, HandLandmark.THUMB_TIP)
            return Pose.THUMBS_UP if -thumb.y >= THUMBS_UP_MIN_COMPONENT else Pose.UNKNOWN

        if extended == set(Finger):
            return Pose.OPEN_PALM
        # The thumb is free in the next two, as a relaxed pointing hand often leaves it out.
        pointing_fingers = extended - {Finger.THUMB}
        if pointing_fingers == {Finger.INDEX}:
            return Pose.POINTING
        if pointing_fingers == {Finger.INDEX, Finger.MIDDLE}:
            return Pose.TWO_FINGER
        return Pose.UNKNOWN


class TemporalVotingFilter:
    """Reports a new pose only once it holds `min_votes` of the last `window` frames.

    Until then the previous winner stands, so a one-frame glitch changes nothing.
    A strict majority is required, which leaves at most one possible winner.
    """

    def __init__(self, window: int = 5, min_votes: int = 3) -> None:
        if not 0 < min_votes <= window or min_votes * 2 <= window:
            raise ValueError(f"{min_votes} of {window} votes is not a strict majority")
        self.min_votes = min_votes
        self._votes: deque[Pose] = deque(maxlen=window)
        self._pose = Pose.UNKNOWN

    @property
    def pose(self) -> Pose:
        return self._pose

    def update(self, pose: Pose) -> Pose:
        self._votes.append(pose)
        winner, votes = Counter(self._votes).most_common(1)[0]
        if votes >= self.min_votes:
            self._pose = winner
        return self._pose

    def reset(self) -> None:
        self._votes.clear()
        self._pose = Pose.UNKNOWN
