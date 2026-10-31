import random
import statistics

import pytest

from src.controller.base import MockMouseController, MouseEventType, Point, ScreenBounds
from src.tracking.cursor_controller import CursorConfig, CursorController
from src.vision.detector import NUM_LANDMARKS, HandLandmark, Landmark

BOUNDS = ScreenBounds(0, 0, 1512, 982)
DT = 1 / 30


def hand(x, y, tip=None):
    points = [Landmark(0.9, 0.9, 0.0)] * NUM_LANDMARKS
    points[HandLandmark.INDEX_FINGER_MCP] = Landmark(x, y, 0.0)
    if tip is not None:
        points[HandLandmark.INDEX_FINGER_TIP] = Landmark(*tip, 0.0)
    return tuple(points)


def sweep(cursor, x0, x1, duration, t0=0.0, y=0.5, settle=0.5):
    """Move the anchor linearly in x, hold still to drain filter lag, return the end time."""
    frames = max(round(duration / DT), 1)
    t = t0
    for i in range(frames + 1):
        t = t0 + i * DT
        cursor.update(hand(x0 + (x1 - x0) * i / frames, y), t)
    for _ in range(round(settle / DT)):
        t += DT
        cursor.update(hand(x1, y), t)
    return t


def test_starts_centered_and_first_frame_does_not_move():
    cursor = CursorController(BOUNDS)
    assert cursor.update(hand(0.9, 0.1), 0.0) == BOUNDS.center


def test_start_position_is_clamped():
    assert CursorController(BOUNDS, start=Point(9999, -5)).position == (1511, 0)


def test_mapping_is_relative_to_previous_frame_not_absolute():
    # The same hand location maps to different cursor locations depending on the start.
    a = CursorController(BOUNDS, start=Point(100, 100))
    b = CursorController(BOUNDS, start=Point(900, 700))
    for cursor in (a, b):
        cursor.update(hand(0.5, 0.5), 0.0)
        cursor.update(hand(0.5, 0.5), DT)
    assert a.position == (100, 100)
    assert b.position == (900, 700)


def test_motion_direction_matches_hand():
    cursor = CursorController(BOUNDS)
    cursor.update(hand(0.5, 0.5), 0.0)
    x, y = cursor.update(hand(0.52, 0.47), DT)
    assert x > BOUNDS.center.x
    assert y < BOUNDS.center.y


def test_gain_curve_limits_and_monotonic():
    config = CursorConfig()
    assert config.gain(0.0) == config.min_gain
    assert config.gain(config.low_speed) == config.min_gain
    assert config.gain(config.high_speed) == config.max_gain
    assert config.gain(100.0) == config.max_gain
    gains = [config.gain(s / 10) for s in range(25)]
    assert gains == sorted(gains)


def test_slow_motion_uses_min_gain():
    config = CursorConfig()
    cursor = CursorController(BOUNDS)
    sweep(cursor, 0.4, 0.5, duration=2.0)  # 0.05 heights/s, under low_speed.
    travelled = cursor.position.x - BOUNDS.center.x
    assert travelled == pytest.approx(0.1 * config.min_gain, rel=0.05)


def test_fast_motion_accelerates_over_same_hand_distance():
    slow = CursorController(BOUNDS, start=Point(0, 400))
    fast = CursorController(BOUNDS, start=Point(0, 400))
    sweep(slow, 0.4, 0.6, duration=2.0)
    sweep(fast, 0.4, 0.6, duration=0.1)
    assert fast.position.x > 2 * slow.position.x


def test_travel_is_consistent_across_frame_rates():
    def travel(fps):
        cursor = CursorController(BOUNDS, start=Point(0, 400))
        n = round(1.0 * fps)
        for i in range(n + 1):
            cursor.update(hand(0.3 + 0.2 * i / n, 0.5), i / fps)
        return cursor.position.x

    assert travel(60) == pytest.approx(travel(20), rel=0.1)


def test_aspect_ratio_equalizes_axes():
    cursor = CursorController(BOUNDS, start=Point(500, 500))
    # Equal physical distances on a 2:1 image: dx in normalized units is half of dy.
    cursor.update(hand(0.5, 0.5), 0.0, aspect_ratio=2.0)
    x, y = cursor.update(hand(0.51, 0.52), DT, aspect_ratio=2.0)
    assert x - 500 == pytest.approx(y - 500)


def test_anchor_is_index_mcp_not_fingertip():
    cursor = CursorController(BOUNDS)
    cursor.update(hand(0.5, 0.5, tip=(0.5, 0.3)), 0.0)
    for i in range(1, 10):
        position = cursor.update(hand(0.5, 0.5, tip=(0.5 + i * 0.03, 0.3)), i * DT)
    assert position == BOUNDS.center


def test_jitter_at_rest_stays_small():
    rng = random.Random(11)
    cursor = CursorController(BOUNDS)
    xs = []
    for i in range(150):
        noisy = hand(0.5 + rng.gauss(0, 0.001), 0.5 + rng.gauss(0, 0.001))
        xs.append(cursor.update(noisy, i * DT).x)
    assert statistics.pstdev(xs) < 2.0


def test_clutch_freezes_cursor_and_release_does_not_jump():
    cursor = CursorController(BOUNDS)
    t = sweep(cursor, 0.4, 0.45, duration=0.5)
    held = cursor.position

    cursor.clutch = True
    for i in range(1, 20):
        t += DT
        assert cursor.update(hand(0.45 - i * 0.02, 0.5), t) == held
    hand_x = 0.45 - 19 * 0.02

    cursor.clutch = False
    t += DT
    assert cursor.update(hand(hand_x, 0.5), t) == held
    for _ in range(30):  # A still hand must not replay filter lag as a slide.
        t += DT
        assert cursor.update(hand(hand_x, 0.5), t) == held


def test_motion_resumes_after_clutch_release():
    cursor = CursorController(BOUNDS, start=Point(100, 400))
    cursor.update(hand(0.5, 0.5), 0.0)
    cursor.clutch = True
    cursor.update(hand(0.5, 0.5), DT)
    cursor.clutch = False
    cursor.update(hand(0.5, 0.5), 2 * DT)
    x, _ = cursor.update(hand(0.55, 0.5), 3 * DT)
    assert x > 100


def test_pinch_freeze_holds_only_for_window():
    config = CursorConfig(pinch_freeze_seconds=0.1)
    cursor = CursorController(BOUNDS, config=config)
    cursor.update(hand(0.5, 0.5), 0.0)
    held = cursor.update(hand(0.5, 0.5), 0.03)

    cursor.start_pinch_freeze(0.03)
    assert cursor.update(hand(0.52, 0.5), 0.06) == held
    assert cursor.update(hand(0.54, 0.5), 0.1) == held
    assert cursor.is_frozen(0.129)
    assert not cursor.is_frozen(0.13)
    assert cursor.update(hand(0.56, 0.5), 0.14) == held  # First frame re-anchors.
    assert cursor.update(hand(0.58, 0.5), 0.17).x > held.x


def test_pinch_freeze_discards_frozen_motion():
    # A hand that drifted during the freeze and then stopped must not slide the cursor.
    config = CursorConfig(pinch_freeze_seconds=0.1)
    cursor = CursorController(BOUNDS, config=config)
    cursor.update(hand(0.5, 0.5), 0.0)
    held = cursor.update(hand(0.5, 0.5), 0.03)
    cursor.start_pinch_freeze(0.03)
    cursor.update(hand(0.6, 0.5), 0.06)
    cursor.update(hand(0.6, 0.5), 0.1)
    for i in range(30):
        assert cursor.update(hand(0.6, 0.5), 0.14 + i * DT) == held


def test_zero_length_freeze_never_freezes():
    cursor = CursorController(BOUNDS, config=CursorConfig(pinch_freeze_seconds=0.0))
    cursor.start_pinch_freeze(1.0)
    assert not cursor.is_frozen(1.0)


@pytest.mark.parametrize(
    "x0, x1, y0, y1, expected_x, expected_y",
    [
        (0.5, 3.0, 0.5, 0.5, 1511, None),
        (0.5, -3.0, 0.5, 0.5, 0, None),
        (0.5, 0.5, 0.5, 3.0, None, 981),
        (0.5, 0.5, 0.5, -3.0, None, 0),
    ],
)
def test_position_is_clamped_to_screen(x0, x1, y0, y1, expected_x, expected_y):
    cursor = CursorController(BOUNDS)
    for i in range(16):
        t = i * DT
        position = cursor.update(hand(x0 + (x1 - x0) * i / 15, y0 + (y1 - y0) * i / 15), t)
    if expected_x is not None:
        assert position.x == expected_x
    if expected_y is not None:
        assert position.y == expected_y


def test_no_hidden_overshoot_at_screen_edge():
    cursor = CursorController(BOUNDS, start=Point(1500, 400))
    t = sweep(cursor, 0.2, 0.9, duration=0.5, settle=0.3)
    assert cursor.position.x == 1511
    x_before_return = cursor.position.x
    for i in range(1, 6):
        x, _ = cursor.update(hand(0.9 - i * 0.01, 0.5), t + i * DT)
    assert x < x_before_return  # Moves away from the edge immediately.


def test_clamps_to_display_with_non_zero_origin():
    bounds = ScreenBounds(-1440, 0, 1440, 900)
    cursor = CursorController(bounds)
    sweep(cursor, 0.5, 3.0, duration=0.5)
    assert cursor.position.x == -1


def test_frame_gap_reanchors_instead_of_jumping():
    cursor = CursorController(BOUNDS)
    cursor.update(hand(0.2, 0.5), 0.0)
    held = cursor.update(hand(0.2, 0.5), DT)
    assert cursor.update(hand(0.8, 0.8), DT + 0.5) == held


def test_reset_reanchors_and_adopts_position_and_clears_freeze():
    cursor = CursorController(BOUNDS)
    cursor.update(hand(0.2, 0.5), 0.0)
    cursor.start_pinch_freeze(0.0)
    cursor.reset(Point(10, 20))
    assert cursor.position == (10, 20)
    assert not cursor.is_frozen(0.01)
    assert cursor.update(hand(0.8, 0.8), 0.01) == (10, 20)


def test_reset_keeps_clutch():
    cursor = CursorController(BOUNDS)
    cursor.clutch = True
    cursor.reset()
    assert cursor.is_frozen(0.0)


def test_positions_drive_mock_controller_within_bounds():
    mouse = MockMouseController(bounds=BOUNDS)
    cursor = CursorController(mouse.screen_bounds(), start=mouse.position)
    for i in range(60):
        mouse.move_to(*cursor.update(hand(0.1 + i * 0.05, 0.5), i * DT))
    assert {event.type for event in mouse.events} == {MouseEventType.MOVED}
    assert all(0 <= event.x <= 1511 for event in mouse.events)
    assert mouse.position.x == 1511


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_gain": 0},
        {"min_gain": 5000},
        {"low_speed": 2.0},
        {"pinch_freeze_seconds": -1},
        {"max_gap_seconds": 0},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        CursorConfig(**kwargs)
