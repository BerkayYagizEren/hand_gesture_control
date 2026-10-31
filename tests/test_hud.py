import numpy as np
import pytest

from hand_factory import POSES, make_hand
from src.ui.hud import (
    COLOR_ANCHOR,
    HAND_CONNECTIONS,
    MODE_COLORS,
    PANEL_ORIGIN,
    PANEL_SIZE,
    RING_COLORS,
    HudStatus,
    HUDVisualizer,
    PinchFeedback,
)
from src.vision.detector import NUM_LANDMARKS, DetectionResult, Hand, HandLandmark, Landmark

WIDTH, HEIGHT = 640, 480
STATUS = HudStatus(mode="MOUSE", state="TRACKING", pose="POINTING", fps=30.0, p50_ms=20.0, p95_ms=30.0)


def blank(value=0):
    return np.full((HEIGHT, WIDTH, 3), value, np.uint8)


def result_with(*hands):
    return DetectionResult(0, WIDTH, HEIGHT, tuple(hands))


def pixel(landmark):
    return round(landmark.y * HEIGHT), round(landmark.x * WIDTH)


def outside_panel(image):
    x, y = PANEL_ORIGIN
    masked = image.copy()
    masked[y : y + PANEL_SIZE[1], x : x + PANEL_SIZE[0]] = 0
    return masked


def has_color(image, color):
    return bool(np.any(np.all(image == np.array(color, np.uint8), axis=2)))


def test_connections_cover_all_21_landmarks():
    joined = {landmark for connection in HAND_CONNECTIONS for landmark in connection}
    assert joined == set(HandLandmark)
    assert len(joined) == NUM_LANDMARKS


def test_draw_works_in_place_and_returns_the_same_image():
    image = blank()
    assert HUDVisualizer().draw(image, None, STATUS) is image


def test_no_hand_only_touches_the_panel():
    image = HUDVisualizer().draw(blank(), result_with(), STATUS)
    assert image.any()
    assert not outside_panel(image).any()


def test_panel_is_translucent_with_rounded_corners():
    background = blank(200)
    empty = HudStatus(mode="MOUSE", state="", pose="")
    image = HUDVisualizer().draw(background.copy(), None, empty)
    x, y = PANEL_ORIGIN
    inside = image[y + 2, x + PANEL_SIZE[0] // 2]
    # Darkened by the panel, but the bright background still shows through.
    assert 0 < inside[0] < 200 and inside[0] > 32
    # The very corner of the panel's box is cut off by the radius.
    assert tuple(image[y, x]) == (200, 200, 200)


def test_panel_leaves_pixels_outside_its_box_alone():
    image = HUDVisualizer().draw(blank(123), None, STATUS)
    assert (image[HEIGHT - 1, WIDTH - 1] == 123).all()
    x, y = PANEL_ORIGIN
    assert (image[y + PANEL_SIZE[1] + 1, x + 5] == 123).all()


def test_skeleton_joints_and_anchor_are_drawn_at_landmark_pixels():
    hand = make_hand(POSES["OPEN_PALM"], x=0.6, y=0.8)
    image = HUDVisualizer().draw(blank(), result_with(hand), STATUS)
    for landmark in hand.landmarks:
        assert image[pixel(landmark)].any()
    anchor = hand.landmarks[HandLandmark.INDEX_FINGER_MCP]
    assert tuple(image[pixel(anchor)]) == COLOR_ANCHOR


def test_every_hand_is_drawn():
    first, second = make_hand(x=0.3, y=0.8), make_hand(x=0.75, y=0.8)
    image = HUDVisualizer().draw(blank(), result_with(first, second), STATUS)
    for hand in (first, second):
        assert image[pixel(hand.landmarks[HandLandmark.WRIST])].any()


@pytest.mark.parametrize("pinch", list(RING_COLORS))
def test_pinch_ring_uses_the_color_of_its_action(pinch):
    hand = make_hand(POSES["PINCH"], x=0.6, y=0.8)
    status = HudStatus(mode="MOUSE", state="PINCH_ACTIVE", pose="PINCH", pinch=pinch)
    image = HUDVisualizer().draw(blank(), result_with(hand), status)
    assert has_color(outside_panel(image), RING_COLORS[pinch])


def test_no_ring_without_pinch():
    hand = make_hand(POSES["PINCH"], x=0.6, y=0.8)
    image = HUDVisualizer().draw(blank(), result_with(hand), STATUS)
    for color in RING_COLORS.values():
        assert not has_color(outside_panel(image), color)


def test_ring_is_centered_between_the_pinching_fingertips():
    hand = make_hand(POSES["PINCH"], x=0.6, y=0.8)
    status = HudStatus(mode="MOUSE", state="", pose="", pinch=PinchFeedback.LEFT_CLICK)
    plain = HUDVisualizer().draw(blank(), result_with(hand), HudStatus(mode="MOUSE", state="", pose=""))
    ringed = HUDVisualizer().draw(blank(), result_with(hand), status)
    rows, columns = np.nonzero(np.any(ringed != plain, axis=2))
    thumb = pixel(hand.landmarks[HandLandmark.THUMB_TIP])
    tip = pixel(hand.landmarks[HandLandmark.INDEX_FINGER_TIP])
    assert (rows.min() + rows.max()) / 2 == pytest.approx((thumb[0] + tip[0]) / 2, abs=2)
    assert (columns.min() + columns.max()) / 2 == pytest.approx((thumb[1] + tip[1]) / 2, abs=2)


@pytest.mark.parametrize(
    "fields",
    [{"clutch": True}, {"frozen": True}, {"fps": 12.5}, {"p50_ms": None, "p95_ms": None}, {"mode": "MEDIA"}],
)
def test_panel_reflects_status_fields(fields):
    changed = HudStatus(**{**STATUS.__dict__, **fields})
    baseline = HUDVisualizer().draw(blank(), None, STATUS)
    image = HUDVisualizer().draw(blank(), None, changed)
    assert np.any(image != baseline)


def switching(progress, next_mode="MEDIA"):
    return HudStatus(**{**STATUS.__dict__, "mode_progress": progress, "next_mode": next_mode})


def count_color(image, color):
    return int(np.count_nonzero(np.all(image == np.array(color, np.uint8), axis=2)))


def test_no_mode_progress_without_a_hold():
    plain = HUDVisualizer().draw(blank(), None, STATUS)
    idle = HUDVisualizer().draw(blank(), None, switching(0.0))
    assert np.array_equal(plain, idle)
    assert not has_color(idle, MODE_COLORS["MEDIA"])


@pytest.mark.parametrize("next_mode", ["MOUSE", "PRESENTATION", "MEDIA", "FANTASY"])
def test_mode_progress_is_drawn_in_the_next_modes_color_inside_the_panel(next_mode):
    image = HUDVisualizer().draw(blank(), None, switching(0.5, next_mode))
    assert has_color(image, MODE_COLORS[next_mode])
    assert not outside_panel(image).any()


def test_mode_progress_fill_grows_with_the_hold():
    color = MODE_COLORS["MEDIA"]
    filled = [
        count_color(HUDVisualizer().draw(blank(), None, switching(progress)), color)
        for progress in (0.1, 0.5, 1.0)
    ]
    assert filled == sorted(set(filled))


def test_mode_progress_stays_inside_the_panel_when_over_full():
    over = HUDVisualizer().draw(blank(), None, switching(2.0))
    full = HUDVisualizer().draw(blank(), None, switching(1.0))
    assert np.array_equal(over, full)


def test_non_finite_landmarks_are_skipped():
    good = make_hand(x=0.6, y=0.8)
    bad = list(good.landmarks)
    bad[HandLandmark.THUMB_TIP] = Landmark(float("nan"), float("inf"), 0.0)
    hand = Hand(tuple(bad), good.world_landmarks, good.handedness, good.handedness_score)
    status = HudStatus(mode="MOUSE", state="", pose="", pinch=PinchFeedback.LEFT_CLICK)
    image = HUDVisualizer().draw(blank(), result_with(hand), status)
    assert image[pixel(good.landmarks[HandLandmark.WRIST])].any()


def test_landmarks_outside_the_image_do_not_raise():
    hand = make_hand(x=2.5, y=-3.0)
    HUDVisualizer().draw(blank(), result_with(hand), STATUS)


@pytest.mark.parametrize("size", [(4, 4), (100, 100), (PANEL_SIZE[1] + 12, PANEL_SIZE[0] + 12)])
def test_images_smaller_than_the_panel_are_clipped(size):
    image = np.zeros((*size, 3), np.uint8)
    assert HUDVisualizer().draw(image, None, STATUS).shape == (*size, 3)
