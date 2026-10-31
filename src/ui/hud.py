"""Live OpenCV overlay: hand skeleton, pinch feedback ring and a translucent status panel."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np

from src.vision.detector import DetectionResult, Hand, HandLandmark, Landmark

HAND_CONNECTIONS: tuple[tuple[HandLandmark, HandLandmark], ...] = (
    (HandLandmark.WRIST, HandLandmark.THUMB_CMC),
    (HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP),
    (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP),
    (HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP),
    (HandLandmark.WRIST, HandLandmark.INDEX_FINGER_MCP),
    (HandLandmark.INDEX_FINGER_MCP, HandLandmark.INDEX_FINGER_PIP),
    (HandLandmark.INDEX_FINGER_PIP, HandLandmark.INDEX_FINGER_DIP),
    (HandLandmark.INDEX_FINGER_DIP, HandLandmark.INDEX_FINGER_TIP),
    (HandLandmark.INDEX_FINGER_MCP, HandLandmark.MIDDLE_FINGER_MCP),
    (HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.MIDDLE_FINGER_PIP),
    (HandLandmark.MIDDLE_FINGER_PIP, HandLandmark.MIDDLE_FINGER_DIP),
    (HandLandmark.MIDDLE_FINGER_DIP, HandLandmark.MIDDLE_FINGER_TIP),
    (HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.RING_FINGER_MCP),
    (HandLandmark.RING_FINGER_MCP, HandLandmark.RING_FINGER_PIP),
    (HandLandmark.RING_FINGER_PIP, HandLandmark.RING_FINGER_DIP),
    (HandLandmark.RING_FINGER_DIP, HandLandmark.RING_FINGER_TIP),
    (HandLandmark.RING_FINGER_MCP, HandLandmark.PINKY_MCP),
    (HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP),
    (HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP),
    (HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP),
    (HandLandmark.WRIST, HandLandmark.PINKY_MCP),
)

# Colors are BGR, as OpenCV draws them.
COLOR_BONE = (225, 225, 225)
COLOR_JOINT = (255, 190, 90)
COLOR_ANCHOR = (60, 220, 255)
COLOR_PANEL = (32, 26, 22)
COLOR_TEXT = (240, 240, 240)
COLOR_LABEL = (165, 165, 165)
COLOR_PILL_OFF = (75, 70, 65)
COLOR_CLUTCH = (70, 110, 255)
COLOR_FROZEN = (235, 190, 70)

MODE_COLORS = {
    "MOUSE": (120, 220, 110),
    "SCROLL": (235, 190, 70),
    "PRESENTATION": (70, 170, 255),
    "MEDIA": (225, 120, 215),
    "FANTASY": (255, 229, 0),  # Chakra blue, #00E5FF.
}

PANEL_ORIGIN = (12, 12)
PANEL_SIZE = (320, 158)  # (width, height)
PANEL_RADIUS = 12
PANEL_OPACITY = 0.68

_FONT = cv2.FONT_HERSHEY_SIMPLEX


class PinchFeedback(Enum):
    NONE = auto()
    LEFT_CLICK = auto()
    DRAG = auto()
    RIGHT_CLICK = auto()


RING_COLORS = {
    PinchFeedback.LEFT_CLICK: (110, 230, 90),
    PinchFeedback.DRAG: (50, 165, 255),
    PinchFeedback.RIGHT_CLICK: (220, 90, 225),
}


@dataclass(frozen=True)
class HudStatus:
    mode: str  # A `MODE_COLORS` key.
    state: str
    pose: str
    fps: float = 0.0
    p50_ms: float | None = None  # None until the profiler has samples.
    p95_ms: float | None = None
    clutch: bool = False
    frozen: bool = False
    pinch: PinchFeedback = PinchFeedback.NONE
    last_action: str | None = None
    mode_progress: float = 0.0  # How far a mode switch hold has come, from 0 to 1.
    next_mode: str | None = None  # A `MODE_COLORS` key.


class HUDVisualizer:
    """Draws onto the frame it is given. Bad landmarks are skipped, so it cannot stop the loop."""

    def __init__(self) -> None:
        self._panel_mask = _rounded_mask(PANEL_SIZE, PANEL_RADIUS)

    def draw(
        self,
        image: np.ndarray,
        result: DetectionResult | None,
        status: HudStatus,
        effects: Callable[[np.ndarray], None] | None = None,
    ) -> np.ndarray:
        """Render one frame in place and return it. `result` is None if nothing was detected.

        `effects` draws between the hand and the panel, so it covers one but not the other.
        """
        hands = result.hands if result is not None else ()
        for hand in hands:
            self._draw_hand(image, hand)
        # The state machine only follows the first hand, so that one gets the ring.
        if hands and status.pinch is not PinchFeedback.NONE:
            self._draw_pinch_ring(image, hands[0], status.pinch)
        if effects is not None:
            effects(image)
        self._draw_panel(image, status)
        return image

    def _draw_hand(self, image: np.ndarray, hand: Hand) -> None:
        points = _to_pixels(hand.landmarks, image)
        for start, end in HAND_CONNECTIONS:
            if points[start] is not None and points[end] is not None:
                cv2.line(image, points[start], points[end], COLOR_BONE, 2, cv2.LINE_AA)
        for point in points:
            if point is not None:
                cv2.circle(image, point, 3, COLOR_JOINT, -1, cv2.LINE_AA)
        anchor = points[HandLandmark.INDEX_FINGER_MCP]
        if anchor is not None:
            cv2.circle(image, anchor, 7, COLOR_ANCHOR, -1, cv2.LINE_AA)
            cv2.circle(image, anchor, 10, COLOR_ANCHOR, 1, cv2.LINE_AA)

    def _draw_pinch_ring(self, image: np.ndarray, hand: Hand, pinch: PinchFeedback) -> None:
        points = _to_pixels(hand.landmarks, image)
        finger = (
            HandLandmark.MIDDLE_FINGER_TIP
            if pinch is PinchFeedback.RIGHT_CLICK
            else HandLandmark.INDEX_FINGER_TIP
        )
        thumb, tip = points[HandLandmark.THUMB_TIP], points[finger]
        if thumb is None or tip is None:
            return
        center = ((thumb[0] + tip[0]) // 2, (thumb[1] + tip[1]) // 2)
        radius = max(round(math.dist(thumb, tip) / 2) + 10, 14)
        cv2.circle(image, center, radius, RING_COLORS[pinch], 3, cv2.LINE_AA)

    def _draw_panel(self, image: np.ndarray, status: HudStatus) -> None:
        self._blend_panel_background(image)
        x, y = PANEL_ORIGIN
        left = x + 14
        mode_color = MODE_COLORS.get(status.mode, COLOR_TEXT)
        cv2.circle(image, (left + 5, y + 21), 5, mode_color, -1, cv2.LINE_AA)
        _text(image, status.mode, (left + 18, y + 28), mode_color, 0.62, 2)
        _text(image, f"{status.fps:.1f} FPS", (x + PANEL_SIZE[0] - 90, y + 27), COLOR_TEXT, 0.45)

        self._draw_row(image, "STATE", status.state, y + 54)
        self._draw_row(image, "POSE", status.pose, y + 76)
        latency = f"p50 {_format_ms(status.p50_ms)}  p95 {_format_ms(status.p95_ms)} ms"
        self._draw_row(image, "LATENCY", latency, y + 98)
        self._draw_row(image, "ACTION", status.last_action or "-", y + 120)

        pill_y = y + 132
        next_x = _pill(image, "CLUTCH", (left, pill_y), COLOR_CLUTCH if status.clutch else None)
        next_x = _pill(image, "FREEZE", (next_x + 8, pill_y), COLOR_FROZEN if status.frozen else None)
        if status.mode_progress > 0:
            self._draw_mode_progress(image, status, next_x + 12, pill_y)

    def _draw_mode_progress(self, image: np.ndarray, status: HudStatus, left: int, y: int) -> None:
        color = MODE_COLORS.get(status.next_mode, COLOR_TEXT)
        _text(image, f"> {status.next_mode}", (left, y + 8), color, 0.38)
        width = PANEL_ORIGIN[0] + PANEL_SIZE[0] - 14 - left
        height = 6
        _capsule(image, (left, y + 12), width, height, COLOR_PILL_OFF)
        filled = round(width * min(status.mode_progress, 1.0))
        _capsule(image, (left, y + 12), max(filled, height), height, color)

    def _draw_row(self, image: np.ndarray, label: str, value: str, y: int) -> None:
        left = PANEL_ORIGIN[0] + 14
        _text(image, label, (left, y), COLOR_LABEL, 0.4)
        _text(image, value, (left + 74, y), COLOR_TEXT, 0.45)

    def _blend_panel_background(self, image: np.ndarray) -> None:
        # Only the panel's pixels are touched, which is what keeps this cheap per frame.
        x, y = PANEL_ORIGIN
        roi = image[y : y + PANEL_SIZE[1], x : x + PANEL_SIZE[0]]
        if roi.size == 0:
            return
        alpha = self._panel_mask[: roi.shape[0], : roi.shape[1], np.newaxis] * PANEL_OPACITY
        roi[:] = (roi * (1.0 - alpha) + np.array(COLOR_PANEL) * alpha).astype(np.uint8)


def _to_pixels(
    landmarks: Sequence[Landmark], image: np.ndarray
) -> list[tuple[int, int] | None]:
    height, width = image.shape[:2]
    return [
        (round(p.x * width), round(p.y * height))
        if math.isfinite(p.x) and math.isfinite(p.y)
        else None
        for p in landmarks
    ]


def _rounded_mask(size: tuple[int, int], radius: int) -> np.ndarray:
    width, height = size
    mask = np.zeros((height, width), np.uint8)
    cv2.rectangle(mask, (radius, 0), (width - radius - 1, height - 1), 255, -1)
    cv2.rectangle(mask, (0, radius), (width - 1, height - radius - 1), 255, -1)
    for corner in (
        (radius, radius),
        (width - radius - 1, radius),
        (radius, height - radius - 1),
        (width - radius - 1, height - radius - 1),
    ):
        cv2.circle(mask, corner, radius, 255, -1, cv2.LINE_AA)
    return mask.astype(np.float32) / 255.0


def _format_ms(value: float | None) -> str:
    return "--" if value is None else f"{value:.1f}"


def _text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, origin, _FONT, scale, color, thickness, cv2.LINE_AA)


def _capsule(
    image: np.ndarray,
    origin: tuple[int, int],
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    x, y = origin
    radius = height // 2
    cv2.rectangle(image, (x + radius, y), (x + width - radius, y + height), color, -1, cv2.LINE_AA)
    cv2.circle(image, (x + radius, y + radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(image, (x + width - radius, y + radius), radius, color, -1, cv2.LINE_AA)


def _pill(
    image: np.ndarray,
    label: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] | None,
) -> int:
    """Draw an indicator, lit if `color` is given, and return the x of its right edge."""
    x, y = origin
    (text_width, _), _ = cv2.getTextSize(label, _FONT, 0.38, 1)
    width = text_width + 20
    _capsule(image, origin, width, 18, COLOR_PILL_OFF if color is None else color)
    text_color = COLOR_LABEL if color is None else (20, 20, 20)
    _text(image, label, (x + 10, y + 13), text_color, 0.38)
    return x + width
