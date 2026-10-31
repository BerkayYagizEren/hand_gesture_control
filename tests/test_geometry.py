import math

import pytest

from hand_factory import CURLED, PALM_WIDTH, PARTIAL, THUMB_TUCKED, make_hand, world_points
from src.vision.detector import NUM_LANDMARKS, HandLandmark, Landmark
from src.vision.geometry import (
    Finger,
    FingerState,
    PalmFacing,
    direction,
    distance_ratio,
    finger_flexion,
    finger_joint_angles,
    finger_state,
    finger_states,
    hand_direction,
    joint_angle,
    middle_pinch_ratio,
    palm_facing,
    palm_normal,
    pinch_ratio,
)

FLAT = tuple(Landmark(0.0, 0.0, 0.0) for _ in range(NUM_LANDMARKS))


def scaled(points, factor):
    return tuple(Landmark(p.x * factor, p.y * factor, p.z * factor) for p in points)


def rotated_about_y(points, degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return tuple(Landmark(c * p.x + s * p.z, p.y, -s * p.x + c * p.z) for p in points)


def with_point(points, index, x, y, z=0.0):
    points = list(points)
    points[index] = Landmark(x, y, z)
    return tuple(points)


def test_distance_ratio_is_in_palm_widths():
    world = world_points()
    assert distance_ratio(
        world, HandLandmark.INDEX_FINGER_MCP, HandLandmark.PINKY_MCP
    ) == pytest.approx(1.0)
    assert distance_ratio(
        world, HandLandmark.INDEX_FINGER_MCP, HandLandmark.MIDDLE_FINGER_MCP
    ) == pytest.approx(0.02 / PALM_WIDTH)


def test_ratios_do_not_depend_on_hand_size():
    world = world_points(index=PARTIAL, pinch=0.2)
    assert pinch_ratio(scaled(world, 3.0)) == pytest.approx(pinch_ratio(world))


def test_world_ratio_survives_rotation():
    world = world_points(index=PARTIAL, pinch=0.2)
    assert pinch_ratio(rotated_about_y(world, 70)) == pytest.approx(0.2)


def test_image_ratio_corrects_for_aspect_ratio():
    # 0.3 of the width on a 2:1 image equals 0.6 of the height.
    points = with_point(FLAT, HandLandmark.INDEX_FINGER_MCP, 0.0, 0.0)
    points = with_point(points, HandLandmark.PINKY_MCP, 0.0, 0.6)
    points = with_point(points, HandLandmark.THUMB_TIP, 0.0, 0.0)
    points = with_point(points, HandLandmark.INDEX_FINGER_TIP, 0.3, 0.0)
    assert pinch_ratio(points, aspect_ratio=2.0) == pytest.approx(1.0)
    assert pinch_ratio(points, aspect_ratio=1.0) == pytest.approx(0.5)


def test_image_ratio_ignores_depth():
    points = with_point(FLAT, HandLandmark.PINKY_MCP, 0.0, 0.5)
    points = with_point(points, HandLandmark.INDEX_FINGER_TIP, 0.0, 0.0, z=0.9)
    assert pinch_ratio(points, aspect_ratio=1.0) == 0.0


def test_pinch_ratios_measure_their_own_finger():
    index_pinch = world_points(index=PARTIAL, pinch=0.1)
    middle_pinch = world_points(middle=PARTIAL, middle_pinch=0.1)
    assert pinch_ratio(index_pinch) == pytest.approx(0.1)
    assert middle_pinch_ratio(index_pinch) > 0.25
    assert middle_pinch_ratio(middle_pinch) == pytest.approx(0.1)
    assert pinch_ratio(middle_pinch) > 0.25


def test_zero_palm_width_gives_infinite_ratio():
    assert pinch_ratio(FLAT) == math.inf
    assert middle_pinch_ratio(FLAT, aspect_ratio=1.5) == math.inf


@pytest.mark.parametrize(
    ("c", "expected"),
    [((2.0, 0.0), 180.0), ((1.0, 1.0), 90.0), ((0.0, 0.0), 0.0), ((2.0, 1.0), 135.0)],
)
def test_joint_angle(c, expected):
    points = with_point(FLAT, 0, 0.0, 0.0)
    points = with_point(points, 1, 1.0, 0.0)
    points = with_point(points, 2, *c)
    assert joint_angle(points, 0, 1, 2) == pytest.approx(expected)


def test_joint_angle_of_coincident_points_is_nan():
    assert math.isnan(joint_angle(FLAT, 0, 1, 2))


def test_joint_angle_tolerates_rounding_past_straight():
    points = with_point(FLAT, 0, 0.1, 0.1)
    points = with_point(points, 1, 0.2, 0.2)
    points = with_point(points, 2, 0.3, 0.3)
    assert joint_angle(points, 0, 1, 2) == pytest.approx(180.0)


def test_finger_has_three_joint_angles_and_thumb_two():
    world = world_points()
    assert len(finger_joint_angles(world, Finger.INDEX)) == 3
    assert len(finger_joint_angles(world, Finger.THUMB)) == 2


def test_flexion_adds_up_the_bend_of_each_joint():
    world = world_points(middle=(10.0, 20.0, 30.0))
    assert finger_flexion(world, Finger.MIDDLE) == pytest.approx(60.0, abs=8.0)
    assert finger_flexion(world_points(), Finger.MIDDLE) < 10.0


@pytest.mark.parametrize("finger", [Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY])
@pytest.mark.parametrize(
    ("flexions", "state"),
    [
        ((0.0, 0.0, 0.0), FingerState.EXTENDED),
        (PARTIAL, FingerState.PARTIAL),
        (CURLED, FingerState.CURLED),
    ],
)
def test_finger_state(finger, flexions, state):
    world = world_points(**{finger.name.lower(): flexions})
    assert finger_state(world, finger) is state


def test_thumb_state():
    assert finger_state(world_points(), Finger.THUMB) is FingerState.EXTENDED
    assert finger_state(world_points(thumb=THUMB_TUCKED), Finger.THUMB) is FingerState.CURLED


def test_straight_thumb_lying_along_the_palm_is_curled():
    alongside = ((-0.05, -0.02, 0.0), (-0.05, -0.05, 0.0), (-0.05, -0.07, 0.0), (-0.05, -0.09, 0.0))
    assert finger_state(world_points(thumb=alongside), Finger.THUMB) is FingerState.CURLED


def test_finger_states_are_independent():
    states = finger_states(world_points(index=CURLED))
    assert states[Finger.INDEX] is FingerState.CURLED
    assert states[Finger.PINKY] is FingerState.EXTENDED
    assert set(states) == set(Finger)


def test_degenerate_finger_is_partial():
    assert finger_state(FLAT, Finger.INDEX) is FingerState.PARTIAL


def test_finger_state_does_not_depend_on_hand_rotation():
    world = rotated_about_y(world_points(index=CURLED, thumb=THUMB_TUCKED), 60)
    assert finger_state(world, Finger.INDEX) is FingerState.CURLED
    assert finger_state(world, Finger.MIDDLE) is FingerState.EXTENDED


def test_direction_is_a_unit_vector():
    points = with_point(FLAT, 1, 3.0, 4.0)
    assert direction(points, 0, 1) == pytest.approx((0.6, 0.8, 0.0))
    assert direction(FLAT, 0, 1) == (0.0, 0.0, 0.0)


def test_hand_direction_points_along_the_fingers():
    assert hand_direction(world_points()).y == pytest.approx(-1.0, abs=0.05)
    assert hand_direction(world_points(rotate_90=True)).x == pytest.approx(1.0, abs=0.05)


def test_palm_normal_faces_camera_for_left_label_palm_forward():
    x, y, z = palm_normal(world_points(), "Left")
    assert z == pytest.approx(-1.0, abs=0.01)
    assert math.hypot(x, y, z) == pytest.approx(1.0)


def test_right_label_palm_forward_also_faces_camera():
    # The mirror image of a left hand is a right hand with the same palm direction.
    mirrored = tuple(Landmark(-p.x, p.y, p.z) for p in world_points())
    assert palm_facing(mirrored, "Right") is PalmFacing.TOWARD_CAMERA


@pytest.mark.parametrize(
    ("handedness", "wrist", "index", "pinky"),
    [
        ("Right", (-0.077, 0.012, -0.033), (0.01, 0.024, 0.006), (-0.017, -0.033, -0.017)),
        ("Left", (0.093, 0.007, -0.04), (0.0, 0.026, 0.014), (0.01, -0.027, -0.027)),
    ],
)
def test_palm_facing_matches_real_detector_output(handedness, wrist, index, pinky):
    # Values from HandDetector on a photo of two palms shown to the camera.
    points = with_point(FLAT, HandLandmark.WRIST, *wrist)
    points = with_point(points, HandLandmark.INDEX_FINGER_MCP, *index)
    points = with_point(points, HandLandmark.PINKY_MCP, *pinky)
    assert palm_facing(points, handedness) is PalmFacing.TOWARD_CAMERA


def test_back_of_hand_faces_away():
    assert palm_facing(rotated_about_y(world_points(), 180), "Left") is PalmFacing.AWAY_FROM_CAMERA


def test_palm_turned_edge_on_is_sideways():
    assert palm_facing(rotated_about_y(world_points(), 90), "Left") is PalmFacing.SIDEWAYS


@pytest.mark.parametrize(("degrees", "facing"), [(90, PalmFacing.DOWN), (-90, PalmFacing.UP)])
def test_palm_tilted_flat_faces_up_or_down(degrees, facing):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    # Fingers tipped toward the camera turn the palm down; tipped away, up.
    world = tuple(Landmark(p.x, c * p.y - s * p.z, s * p.y + c * p.z) for p in world_points())
    assert palm_facing(world, "Left") is facing


def test_degenerate_palm_normal_is_zero():
    assert palm_normal(FLAT, "Right") == (0.0, 0.0, 0.0)


def test_hand_factory_wraps_world_points():
    assert make_hand(x=0.2).landmarks[HandLandmark.WRIST].x == pytest.approx(0.2)
