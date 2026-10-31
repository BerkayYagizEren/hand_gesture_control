"""Builds anatomically plausible synthetic hands for geometry, classifier and FSM tests.

The default hand looks like a left hand in the image (the user's right hand in a
mirrored stream): fingers up, palm toward the camera, index on the image's left.
Fingers bend toward the camera.
"""

import math

from src.vision.detector import NUM_LANDMARKS, Hand, HandLandmark, Landmark

STRAIGHT = (0.0, 0.0, 0.0)
PARTIAL = (30.0, 50.0, 30.0)
CURLED = (80.0, 100.0, 50.0)

PALM_WIDTH = 0.06
_MCP_X = {5: -0.03, 9: -0.01, 13: 0.01, 17: 0.03}  # First landmark of each finger.
_SEGMENTS = (0.04, 0.025, 0.02)

THUMB_OUT = ((-0.03, -0.02, 0.0), (-0.055, -0.04, 0.0), (-0.075, -0.06, 0.0), (-0.09, -0.08, 0.0))
THUMB_TUCKED = (
    (-0.03, -0.02, 0.0),
    (-0.035, -0.05, -0.01),
    (-0.025, -0.075, -0.02),
    (-0.015, -0.085, -0.02),
)
THUMB_LATERAL = ((-0.03, -0.02, 0.0), (-0.06, -0.03, 0.0), (-0.085, -0.03, 0.0), (-0.11, -0.03, 0.0))


def _finger(first, flexions):
    x, y, z = _MCP_X[first], -0.09, 0.0
    points = [(x, y, z)]
    angle = 0.0
    for flexion, length in zip(flexions, _SEGMENTS, strict=True):
        angle += math.radians(flexion)
        y -= length * math.cos(angle)
        z -= length * math.sin(angle)
        points.append((x, y, z))
    return points


def world_points(
    index=STRAIGHT,
    middle=STRAIGHT,
    ring=STRAIGHT,
    pinky=STRAIGHT,
    thumb=THUMB_OUT,
    pinch=None,
    middle_pinch=None,
    rotate_90=False,
):
    """Return world landmarks. `pinch`/`middle_pinch` place the thumb tip at that ratio."""
    points = [(0.0, 0.0, 0.0)] * NUM_LANDMARKS
    points[HandLandmark.THUMB_CMC : HandLandmark.THUMB_TIP + 1] = thumb
    for first, flexions in ((5, index), (9, middle), (13, ring), (17, pinky)):
        points[first : first + 4] = _finger(first, flexions)
    for ratio, tip in ((pinch, HandLandmark.INDEX_FINGER_TIP), (middle_pinch, HandLandmark.MIDDLE_FINGER_TIP)):
        if ratio is not None:
            x, y, z = points[tip]
            points[HandLandmark.THUMB_TIP] = (x - ratio * PALM_WIDTH, y, z)
    if rotate_90:  # Quarter turn in the image plane: fingers end up pointing image right.
        points = [(-y, x, z) for x, y, z in points]
    return tuple(Landmark(*point) for point in points)


def make_hand(world=None, x=0.5, y=0.5, handedness="Left"):
    """Wrap world landmarks in a `Hand`, with the wrist of the image landmarks at (x, y)."""
    world = world_points() if world is None else world
    image = tuple(Landmark(p.x + x, p.y + y, p.z) for p in world)
    return Hand(image, world, handedness, 0.99)


POSES = {
    "POINTING": world_points(middle=CURLED, ring=CURLED, pinky=CURLED, thumb=THUMB_TUCKED),
    "POINTING_THUMB_OUT": world_points(middle=CURLED, ring=CURLED, pinky=CURLED),
    "PINCH": world_points(index=PARTIAL, pinch=0.1),
    "MIDDLE_PINCH": world_points(middle=PARTIAL, middle_pinch=0.1),
    "FIST": world_points(CURLED, CURLED, CURLED, CURLED, THUMB_TUCKED),
    "TWO_FINGER": world_points(ring=CURLED, pinky=CURLED, thumb=THUMB_TUCKED),
    "OPEN_PALM": world_points(),
    "THUMBS_UP": world_points(CURLED, CURLED, CURLED, CURLED, THUMB_LATERAL, rotate_90=True),
    "THREE_FINGER": world_points(pinky=CURLED, thumb=THUMB_TUCKED),
}
