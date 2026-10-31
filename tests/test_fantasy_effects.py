import math

import numpy as np
import pytest

from hand_factory import POSES, make_hand
from src.fantasy.effects.chidori import MAX_ARCS, MIN_ARCS, ChidoriEffect
from src.fantasy.effects.common import clipped_region
from src.fantasy.effects.rasengan import SPIN_LAYERS, RasenganEffect
from src.fantasy.engine import EffectKind, FantasyPhase
from src.fantasy.mode import MIN_RADIUS, FantasyMode, _effect_anchor
from src.vision.classifier import Pose, PoseClassifier
from src.vision.detector import DetectionResult, Hand, Landmark

WIDTH, HEIGHT = 640, 480
EFFECTS = [ChidoriEffect, RasenganEffect]
RADIUS = 40.0


def blank(value=0):
    return np.full((HEIGHT, WIDTH, 3), value, np.uint8)


def make_effect(cls, seed=7):
    return cls(np.random.default_rng(seed))


def rendered(cls, center=(320, 240), radius=RADIUS, level=1.0, timestamp=1.0, base=0):
    image = blank(base)
    make_effect(cls).render(image, center, radius, level, timestamp)
    return image


def lit_box(image):
    ys, xs = np.nonzero(image.any(axis=2))
    return xs.min(), xs.max(), ys.min(), ys.max()


# Output shape and bounds


@pytest.mark.parametrize("cls", EFFECTS)
def test_renders_in_place_keeping_shape_and_dtype(cls):
    image = blank()
    assert make_effect(cls).render(image, (320, 240), RADIUS, 1.0, 1.0) is None
    assert image.shape == (HEIGHT, WIDTH, 3) and image.dtype == np.uint8
    assert image.any()


@pytest.mark.parametrize("cls", EFFECTS)
def test_only_pixels_near_the_anchor_change(cls):
    left, right, top, bottom = lit_box(rendered(cls))
    limit = RADIUS * 3.5
    assert left >= 320 - limit and right <= 320 + limit
    assert top >= 240 - limit and bottom <= 240 + limit


@pytest.mark.parametrize("cls", EFFECTS)
def test_zero_level_draws_nothing(cls):
    assert not rendered(cls, level=0.0).any()


@pytest.mark.parametrize("cls", EFFECTS)
def test_stronger_level_is_more_light(cls):
    assert rendered(cls, level=0.2).sum() < rendered(cls, level=1.0).sum()


@pytest.mark.parametrize("cls", EFFECTS)
def test_bright_background_saturates_instead_of_wrapping(cls):
    image = rendered(cls, base=250)
    # A wrapped addition would leave near-black pixels on a near-white frame.
    assert np.count_nonzero(image.sum(axis=2) < 30) == 0


# Anchors at and beyond the edges


@pytest.mark.parametrize("cls", EFFECTS)
@pytest.mark.parametrize(
    "center",
    [(0, 0), (WIDTH - 1, 0), (0, HEIGHT - 1), (WIDTH - 1, HEIGHT - 1), (-20, 240), (WIDTH + 20, 240)],
)
def test_anchor_on_the_edge_draws_clipped_without_error(cls, center):
    image = rendered(cls, center=center)
    assert image.shape == (HEIGHT, WIDTH, 3) and image.dtype == np.uint8
    assert image.any()


@pytest.mark.parametrize("cls", EFFECTS)
@pytest.mark.parametrize("center", [(-5000, 240), (5000, 240), (320, -5000), (320, 5000)])
def test_anchor_far_outside_draws_nothing(cls, center):
    assert not rendered(cls, center=center).any()


@pytest.mark.parametrize("cls", EFFECTS)
def test_works_on_an_image_smaller_than_the_effect(cls):
    image = np.zeros((12, 16, 3), np.uint8)
    make_effect(cls).render(image, (8, 6), 100.0, 1.0, 1.0)
    assert image.any()


def test_region_is_a_view_of_the_image():
    image = blank()
    region = clipped_region(image, (5, 5), 10)
    assert (region.x, region.y) == (0, 0) and region.view.shape == (16, 16, 3)
    region.view[:] = 9
    assert image[0, 0, 0] == 9 and image[20, 20, 0] == 0


# Determinism


@pytest.mark.parametrize("cls", EFFECTS)
def test_same_seed_and_calls_give_identical_pixels(cls):
    def run(seed):
        effect, frames = make_effect(cls, seed), []
        for i in range(12):
            image = blank()
            effect.render(image, (320, 240), RADIUS, 1.0, 1.0 + i * 0.033)
            frames.append(image)
        return frames

    first, second = run(3), run(3)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    assert not all(np.array_equal(a, b) for a, b in zip(first, run(4), strict=True))


@pytest.mark.parametrize("cls", EFFECTS)
def test_rendering_does_not_touch_the_global_random_state(cls):
    np.random.seed(1)
    expected = np.random.random()
    np.random.seed(1)
    rendered(cls)
    assert np.random.random() == expected


# Chidori


def test_chidori_draws_four_to_six_arcs():
    effect = make_effect(ChidoriEffect)
    counts = set()
    for _ in range(90):
        effect.render(blank(), (320, 240), RADIUS, 1.0, 1.0)
        counts.add(len(effect._arcs))
    assert counts == set(range(MIN_ARCS, MAX_ARCS + 1))


def test_chidori_arcs_are_renewed_every_two_or_three_frames():
    effect = make_effect(ChidoriEffect)
    runs, current, previous = [], 0, None
    for _ in range(60):
        effect.render(blank(), (320, 240), RADIUS, 1.0, 1.0)
        if effect._arcs is previous:
            current += 1
        else:
            if previous is not None:
                runs.append(current)
            previous, current = effect._arcs, 1
    assert set(runs) == {2, 3}


def test_chidori_is_blue_white():
    image = rendered(ChidoriEffect)
    lit = image[image.any(axis=2)]
    assert lit[:, 0].mean() > lit[:, 2].mean()
    assert np.any(np.all(image >= 200, axis=2))


# Rasengan


def test_rasengan_has_a_white_core_inside_blue():
    image = rendered(RasenganEffect, radius=60.0)
    assert np.all(image[240, 320] >= 240)
    rim = image[240, 320 + 55]
    assert rim[0] > rim[2]


def test_rasengan_grows_with_its_level():
    small = lit_box(rendered(RasenganEffect, level=0.25))
    large = lit_box(rendered(RasenganEffect, level=1.0))
    assert large[1] - large[0] > small[1] - small[0]


def test_rasengan_layers_turn_at_their_own_speeds():
    assert len({speed for _, speed in SPIN_LAYERS}) == len(SPIN_LAYERS) >= 2
    assert not np.array_equal(
        rendered(RasenganEffect, timestamp=1.0), rendered(RasenganEffect, timestamp=1.05)
    )


def test_rasengan_particles_move_inward_and_are_replaced():
    effect = make_effect(RasenganEffect)
    effect.render(blank(), (320, 240), RADIUS, 1.0, 0.0)
    before = effect._distances.copy()
    effect.render(blank(), (320, 240), RADIUS, 1.0, 0.033)
    moved = effect._distances < before
    assert moved.sum() + (effect._distances > before).sum() == before.size
    assert moved.any()
    for i in range(100):
        effect.render(blank(), (320, 240), RADIUS, 1.0, 0.033 * (i + 2))
    assert effect._distances.max() > before.min()


def test_rasengan_camera_stall_does_not_teleport_particles():
    effect = make_effect(RasenganEffect)
    effect.render(blank(), (320, 240), RADIUS, 1.0, 0.0)
    before = effect._distances.copy()
    effect.render(blank(), (320, 240), RADIUS, 1.0, 100.0)
    survivors = effect._distances < before
    assert np.all(before[survivors] - effect._distances[survivors] <= 0.111)


# Anchors from landmarks


def hand_at(x=0.5, y=0.5, pose="OPEN_PALM"):
    return make_hand(POSES[pose], x=x, y=y)


def test_rasengan_anchor_is_midway_between_wrist_and_middle_mcp():
    hand = hand_at(0.5, 0.5)
    center, _ = _effect_anchor(hand, EffectKind.RASENGAN, blank())
    wrist, middle = hand.landmarks[0], hand.landmarks[9]
    assert center == (round((wrist.x + middle.x) / 2 * WIDTH), round((wrist.y + middle.y) / 2 * HEIGHT))


def test_chidori_anchor_is_among_the_knuckles():
    hand = hand_at(0.5, 0.5, "FIST")
    center, _ = _effect_anchor(hand, EffectKind.CHIDORI, blank())
    assert hand.landmarks[9].y * HEIGHT < center[1] < hand.landmarks[0].y * HEIGHT


def test_effect_size_has_a_floor():
    assert _effect_anchor(hand_at(), EffectKind.RASENGAN, blank())[1] >= MIN_RADIUS
    tiny = tuple(Landmark(0.5, 0.5, 0.0) for _ in range(21))
    hand = Hand(tiny, tiny, "Left", 0.9)
    assert _effect_anchor(hand, EffectKind.CHIDORI, blank())[1] == MIN_RADIUS


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_landmarks_have_no_anchor(bad):
    points = tuple(Landmark(bad, 0.5, 0.0) for _ in range(21))
    hand = Hand(points, points, "Left", 0.9)
    assert _effect_anchor(hand, EffectKind.CHIDORI, blank()) is None


# Driving the effects through the mode


def frames(mode, pose, count, start_ms, x=0.4, y=0.6, handedness="Left"):
    hand = make_hand(POSES[pose], x=x, y=y, handedness=handedness)
    t_ms = start_ms
    for _ in range(count):
        mode.update(DetectionResult(t_ms, WIDTH, HEIGHT, (hand,)), t_ms / 1000)
        t_ms += 33
    return t_ms


def new_mode(seed=0):
    return FantasyMode(PoseClassifier(), seed=seed)


def test_fist_lights_chidori():
    mode = new_mode()
    frames(mode, "FIST", 15, 1000)
    assert (mode.engine.phase, mode.engine.effect) == (FantasyPhase.ACTIVE, EffectKind.CHIDORI)


def test_open_palm_facing_the_camera_lights_rasengan():
    mode = new_mode()
    frames(mode, "OPEN_PALM", 15, 1000)
    assert (mode.engine.phase, mode.engine.effect) == (FantasyPhase.ACTIVE, EffectKind.RASENGAN)


def test_back_of_the_hand_does_not_light_rasengan():
    # The same points labeled as the other hand: the palm now faces away from the camera.
    mode = new_mode()
    frames(mode, "OPEN_PALM", 30, 1000, handedness="Right")
    assert mode.engine.phase is FantasyPhase.IDLE


@pytest.mark.parametrize(
    "tilt, lights",
    [(0, True), (48, True), (-48, True), (55, True), (70, False), (-70, False)],
)
def test_rasengan_tolerates_a_tilted_palm_but_not_an_edge_on_one(tilt, lights):
    # 48 degrees is about what a real raised palm measured: normal (-0.16, 0.73, -0.66).
    cos, sin = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
    world = tuple(
        Landmark(p.x, p.y * cos - p.z * sin, p.y * sin + p.z * cos) for p in POSES["OPEN_PALM"]
    )
    mode, t = new_mode(), 1000
    hand = make_hand(world, x=0.4, y=0.6)
    for _ in range(15):
        mode.update(DetectionResult(t, WIDTH, HEIGHT, (hand,)), t / 1000)
        t += 33
    assert (mode.engine.phase is FantasyPhase.ACTIVE) is lights


@pytest.mark.parametrize("pose", ["POINTING", "PINCH", "MIDDLE_PINCH", "TWO_FINGER", "THUMBS_UP", "THREE_FINGER"])
def test_other_poses_light_nothing(pose):
    mode = new_mode()
    frames(mode, pose, 30, 1000)
    assert mode.engine.phase is FantasyPhase.IDLE


def test_update_reports_activation_once():
    mode = new_mode()
    hand = hand_at(pose="FIST")
    fired = [
        mode.update(DetectionResult(1000 + i * 33, WIDTH, HEIGHT, (hand,)), (1000 + i * 33) / 1000)
        for i in range(30)
    ]
    assert [name for name in fired if name] == ["chidori"]


def test_a_one_frame_glitch_does_not_end_an_effect():
    mode = new_mode()
    t = frames(mode, "FIST", 15, 1000)
    t = frames(mode, "POINTING", 1, t)
    frames(mode, "FIST", 3, t)
    assert mode.engine.phase is FantasyPhase.ACTIVE


def test_opening_the_hand_ends_chidori():
    mode = new_mode()
    t = frames(mode, "FIST", 15, 1000)
    frames(mode, "POINTING", 5, t)
    assert mode.engine.phase is FantasyPhase.COOLDOWN


def test_nothing_is_drawn_while_idle():
    mode, image = new_mode(), blank()
    mode.render(image, 1.0)
    assert not image.any()


def test_effect_stays_at_the_last_anchor_through_a_dropout_and_then_dies():
    mode = new_mode()
    t = frames(mode, "FIST", 15, 1000)
    mode.update(DetectionResult(t + 200, WIDTH, HEIGHT, ()), (t + 200) / 1000)
    image = blank()
    mode.render(image, (t + 200) / 1000)
    assert image.any()

    mode.update(None, (t + 400) / 1000)  # Past the hand loss timeout.
    assert mode.engine.phase is FantasyPhase.COOLDOWN
    mode.update(None, (t + 1600) / 1000)  # Past the release and the cooldown.
    image = blank()
    mode.render(image, (t + 1600) / 1000)
    assert not image.any()
    assert mode.engine.phase is FantasyPhase.IDLE


def test_vote_does_not_outlive_a_real_hand_loss():
    mode = new_mode()
    t = frames(mode, "FIST", 15, 1000)
    mode.update(None, (t + 400) / 1000)
    assert mode.pose is Pose.UNKNOWN


def test_reset_clears_the_effect():
    mode = new_mode()
    t = frames(mode, "FIST", 15, 1000)
    mode.reset()
    image = blank()
    mode.render(image, t / 1000)
    assert not image.any() and mode.state_name == "IDLE"


def test_mode_rendering_is_deterministic_for_a_seed():
    def run(seed):
        mode, out = new_mode(seed), []
        t = 1000
        for pose in ("FIST", "POINTING", "OPEN_PALM"):
            hand = hand_at(0.4, 0.6, pose)
            for _ in range(40):
                mode.update(DetectionResult(t, WIDTH, HEIGHT, (hand,)), t / 1000)
                image = blank()
                mode.render(image, t / 1000)
                out.append(image)
                t += 33
        return np.stack(out)

    assert np.array_equal(run(5), run(5))
    assert not np.array_equal(run(5), run(6))
