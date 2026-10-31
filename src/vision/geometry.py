"""Scale-invariant hand measurements: distance ratios, joint angles, finger states, orientation.

Unless a function says otherwise it takes world landmarks, which are metric and
hold up under hand rotation. MediaPipe's axes apply: +x image right, +y image
down, +z away from the camera.

Degenerate input never raises, because one bad frame must not stop the control
loop: distance ratios become `inf` and angles `nan`, and both classify as
"not pinching" and "not clearly extended or curled".
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum, auto
from typing import NamedTuple

from src.vision.detector import HandLandmark, Landmark, palm_width, world_palm_width

# Total flexion over a finger's joints, in degrees. Between the limits it is PARTIAL.
# A visibly open hand measured 60-100 on real detector output, so "extended" is generous.
FINGER_EXTENDED_BELOW = 105.0
FINGER_CURLED_ABOVE = 150.0
# The thumb bends less, so it also uses its reach: tip to index MCP over palm width.
THUMB_EXTENDED_BELOW = 80.0
THUMB_CURLED_ABOVE = 110.0
THUMB_REACH_EXTENDED = 0.7
THUMB_REACH_CURLED = 0.5


class Vector(NamedTuple):
    x: float
    y: float
    z: float


class Finger(Enum):
    THUMB = auto()
    INDEX = auto()
    MIDDLE = auto()
    RING = auto()
    PINKY = auto()


class FingerState(Enum):
    EXTENDED = auto()
    PARTIAL = auto()
    CURLED = auto()


class PalmFacing(Enum):
    TOWARD_CAMERA = auto()
    AWAY_FROM_CAMERA = auto()
    UP = auto()
    DOWN = auto()
    SIDEWAYS = auto()


# Landmark chains; each interior point is a flexion joint. The thumb chain starts at
# the CMC because the wrist-CMC-MCP angle reflects palm shape rather than bending.
FINGER_CHAINS: dict[Finger, tuple[HandLandmark, ...]] = {
    Finger.THUMB: (
        HandLandmark.THUMB_CMC,
        HandLandmark.THUMB_MCP,
        HandLandmark.THUMB_IP,
        HandLandmark.THUMB_TIP,
    ),
    Finger.INDEX: (
        HandLandmark.WRIST,
        HandLandmark.INDEX_FINGER_MCP,
        HandLandmark.INDEX_FINGER_PIP,
        HandLandmark.INDEX_FINGER_DIP,
        HandLandmark.INDEX_FINGER_TIP,
    ),
    Finger.MIDDLE: (
        HandLandmark.WRIST,
        HandLandmark.MIDDLE_FINGER_MCP,
        HandLandmark.MIDDLE_FINGER_PIP,
        HandLandmark.MIDDLE_FINGER_DIP,
        HandLandmark.MIDDLE_FINGER_TIP,
    ),
    Finger.RING: (
        HandLandmark.WRIST,
        HandLandmark.RING_FINGER_MCP,
        HandLandmark.RING_FINGER_PIP,
        HandLandmark.RING_FINGER_DIP,
        HandLandmark.RING_FINGER_TIP,
    ),
    Finger.PINKY: (
        HandLandmark.WRIST,
        HandLandmark.PINKY_MCP,
        HandLandmark.PINKY_PIP,
        HandLandmark.PINKY_DIP,
        HandLandmark.PINKY_TIP,
    ),
}


def distance_ratio(
    landmarks: Sequence[Landmark],
    a: HandLandmark,
    b: HandLandmark,
    aspect_ratio: float | None = None,
) -> float:
    """Distance between two landmarks in palm widths; `inf` if the palm width is zero.

    By default `landmarks` are world landmarks and the distance is 3D. Given an
    `aspect_ratio`, they are image landmarks and it is measured in the image plane
    against `palm_width`, so it shrinks consistently when the palm turns away.
    """
    p, q = landmarks[a], landmarks[b]
    if aspect_ratio is None:
        distance, width = math.dist(p, q), world_palm_width(landmarks)
    else:
        distance = math.hypot((p.x - q.x) * aspect_ratio, p.y - q.y)
        width = palm_width(landmarks, aspect_ratio)
    return distance / width if width > 0 else math.inf


def pinch_ratio(landmarks: Sequence[Landmark], aspect_ratio: float | None = None) -> float:
    """Index tip to thumb tip distance in palm widths. See `distance_ratio`."""
    return distance_ratio(
        landmarks, HandLandmark.THUMB_TIP, HandLandmark.INDEX_FINGER_TIP, aspect_ratio
    )


def middle_pinch_ratio(landmarks: Sequence[Landmark], aspect_ratio: float | None = None) -> float:
    """Middle tip to thumb tip distance in palm widths. See `distance_ratio`."""
    return distance_ratio(
        landmarks, HandLandmark.THUMB_TIP, HandLandmark.MIDDLE_FINGER_TIP, aspect_ratio
    )


def joint_angle(
    landmarks: Sequence[Landmark], a: HandLandmark, b: HandLandmark, c: HandLandmark
) -> float:
    """Angle a-b-c at `b` in degrees: 180 is straight. `nan` if two points coincide."""
    u = _subtract(landmarks[a], landmarks[b])
    v = _subtract(landmarks[c], landmarks[b])
    lengths = math.hypot(*u) * math.hypot(*v)
    if lengths == 0:
        return math.nan
    cosine = sum(i * j for i, j in zip(u, v, strict=True)) / lengths
    return math.degrees(math.acos(min(max(cosine, -1.0), 1.0)))


def finger_joint_angles(landmarks: Sequence[Landmark], finger: Finger) -> tuple[float, ...]:
    """Joint angles from the base of the finger to its tip."""
    chain = FINGER_CHAINS[finger]
    return tuple(
        joint_angle(landmarks, *chain[i : i + 3]) for i in range(len(chain) - 2)
    )


def finger_flexion(landmarks: Sequence[Landmark], finger: Finger) -> float:
    """Total bend over the finger's joints in degrees: 0 is perfectly straight."""
    return sum(180.0 - angle for angle in finger_joint_angles(landmarks, finger))


def finger_state(landmarks: Sequence[Landmark], finger: Finger) -> FingerState:
    """Classify one finger. A `nan` measurement fails every comparison and gives PARTIAL."""
    flexion = finger_flexion(landmarks, finger)
    if finger is Finger.THUMB:
        reach = distance_ratio(landmarks, HandLandmark.THUMB_TIP, HandLandmark.INDEX_FINGER_MCP)
        if flexion < THUMB_EXTENDED_BELOW and reach > THUMB_REACH_EXTENDED:
            return FingerState.EXTENDED
        if flexion > THUMB_CURLED_ABOVE or reach < THUMB_REACH_CURLED:
            return FingerState.CURLED
        return FingerState.PARTIAL
    if flexion < FINGER_EXTENDED_BELOW:
        return FingerState.EXTENDED
    if flexion > FINGER_CURLED_ABOVE:
        return FingerState.CURLED
    return FingerState.PARTIAL


def finger_states(landmarks: Sequence[Landmark]) -> dict[Finger, FingerState]:
    return {finger: finger_state(landmarks, finger) for finger in Finger}


def direction(landmarks: Sequence[Landmark], start: HandLandmark, end: HandLandmark) -> Vector:
    """Unit vector from `start` to `end`; the zero vector if they coincide."""
    return _normalized(_subtract(landmarks[end], landmarks[start]))


def hand_direction(landmarks: Sequence[Landmark]) -> Vector:
    """Unit vector from the wrist toward the fingers."""
    return direction(landmarks, HandLandmark.WRIST, HandLandmark.MIDDLE_FINGER_MCP)


def palm_normal(landmarks: Sequence[Landmark], handedness: str) -> Vector:
    """Unit vector out of the palm side of the hand; the zero vector if degenerate.

    `handedness` is `Hand.handedness` and decides the sign, since a left and a right
    hand wind opposite ways. On real detector output the label follows how the hand
    looks in the image, so a mirrored stream reports the user's right hand as "Left".
    """
    wrist = landmarks[HandLandmark.WRIST]
    to_index = _subtract(landmarks[HandLandmark.INDEX_FINGER_MCP], wrist)
    to_pinky = _subtract(landmarks[HandLandmark.PINKY_MCP], wrist)
    normal = _cross(to_index, to_pinky)
    if handedness == "Left":
        normal = Vector(-normal.x, -normal.y, -normal.z)
    return _normalized(normal)


def palm_facing(landmarks: Sequence[Landmark], handedness: str) -> PalmFacing:
    """Name the axis the palm normal is closest to."""
    x, y, z = palm_normal(landmarks, handedness)
    if abs(z) >= abs(x) and abs(z) >= abs(y):
        return PalmFacing.TOWARD_CAMERA if z < 0 else PalmFacing.AWAY_FROM_CAMERA
    if abs(y) >= abs(x):
        return PalmFacing.UP if y < 0 else PalmFacing.DOWN
    return PalmFacing.SIDEWAYS


def _subtract(p: Landmark, q: Landmark) -> Vector:
    return Vector(p.x - q.x, p.y - q.y, p.z - q.z)


def _cross(u: Vector, v: Vector) -> Vector:
    return Vector(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x)


def _normalized(v: Vector) -> Vector:
    length = math.hypot(*v)
    return Vector(v.x / length, v.y / length, v.z / length) if length > 0 else Vector(0.0, 0.0, 0.0)
