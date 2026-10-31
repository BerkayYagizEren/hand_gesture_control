"""MediaPipe Tasks HandLandmarker wrapper and the landmark types it produces."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import NamedTuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    RunningMode as VisionRunningMode,
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"
NUM_LANDMARKS = 21


class HandLandmark(IntEnum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


class Landmark(NamedTuple):
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Hand:
    landmarks: tuple[Landmark, ...]  # Normalized to image width/height.
    world_landmarks: tuple[Landmark, ...]  # Metres, origin near the hand's center.
    handedness: str  # "Left" or "Right"; assumes a mirrored input image.
    handedness_score: float

    def __post_init__(self) -> None:
        if len(self.landmarks) != NUM_LANDMARKS or len(self.world_landmarks) != NUM_LANDMARKS:
            raise ValueError(
                f"expected {NUM_LANDMARKS} landmarks, got {len(self.landmarks)} "
                f"and {len(self.world_landmarks)} world landmarks"
            )


@dataclass(frozen=True)
class DetectionResult:
    timestamp_ms: int
    image_width: int
    image_height: int
    hands: tuple[Hand, ...]

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError(f"invalid image size {self.image_width}x{self.image_height}")

    @property
    def aspect_ratio(self) -> float:
        return self.image_width / self.image_height


def palm_width(landmarks: Sequence[Landmark], aspect_ratio: float = 1.0) -> float:
    """Index MCP to pinky MCP distance in the image plane, in units of image height.

    Normalized x and y have different scales on non-square images, so pass
    `DetectionResult.aspect_ratio` to get a true distance. Shrinks when the palm
    turns away from the camera.
    """
    index = landmarks[HandLandmark.INDEX_FINGER_MCP]
    pinky = landmarks[HandLandmark.PINKY_MCP]
    return math.hypot((index.x - pinky.x) * aspect_ratio, index.y - pinky.y)


def world_palm_width(world_landmarks: Sequence[Landmark]) -> float:
    """Index MCP to pinky MCP distance in metres; stable under hand rotation."""
    index = world_landmarks[HandLandmark.INDEX_FINGER_MCP]
    pinky = world_landmarks[HandLandmark.PINKY_MCP]
    return math.dist(index, pinky)


class MonotonicTimestamp:
    """Forces timestamps strictly increasing, as `detect_for_video` requires."""

    def __init__(self) -> None:
        self._last_ms: int | None = None

    def next(self, timestamp_ms: int) -> int:
        if self._last_ms is not None and timestamp_ms <= self._last_ms:
            timestamp_ms = self._last_ms + 1
        self._last_ms = timestamp_ms
        return timestamp_ms


class HandDetector:
    """Synchronous hand detector in VIDEO mode on the CPU delegate."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"hand landmarker model not found: {model_path}")
        self._clock = clock
        self._timestamps = MonotonicTimestamp()
        self._landmarker: HandLandmarker | None = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(model_path),
                    delegate=BaseOptions.Delegate.CPU,
                ),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        )

    def detect(self, image_bgr: np.ndarray, timestamp_ms: int | None = None) -> DetectionResult:
        """Detect hands in a BGR uint8 frame.

        `timestamp_ms` defaults to the clock. Non-increasing values are bumped
        to last + 1, and the value actually used is on the returned result.
        """
        if self._landmarker is None:
            raise RuntimeError("detector is closed")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
            raise ValueError(
                f"expected an HxWx3 uint8 BGR image, got {image_bgr.shape} {image_bgr.dtype}"
            )
        if timestamp_ms is None:
            timestamp_ms = int(self._clock() * 1000)
        timestamp_ms = self._timestamps.next(timestamp_ms)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        raw = self._landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
        )
        height, width = image_bgr.shape[:2]
        return DetectionResult(timestamp_ms, width, height, _convert_hands(raw))

    def close(self) -> None:
        """Release the landmarker. Safe to call repeatedly."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> HandDetector:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _convert_hands(raw: HandLandmarkerResult) -> tuple[Hand, ...]:
    return tuple(
        Hand(
            landmarks=tuple(Landmark(p.x, p.y, p.z) for p in landmarks),
            world_landmarks=tuple(Landmark(p.x, p.y, p.z) for p in world_landmarks),
            handedness=categories[0].category_name,
            handedness_score=categories[0].score,
        )
        for landmarks, world_landmarks, categories in zip(
            raw.hand_landmarks, raw.hand_world_landmarks, raw.handedness, strict=True
        )
    )
