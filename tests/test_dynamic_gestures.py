import pytest

from hand_factory import make_hand
from src.vision.classifier import Pose
from src.vision.dynamic_gestures import SwipeConfig, SwipeDetector, SwipeDirection, SwipePhase

DT = 1 / 30


class Rig:
    def __init__(self, config=SwipeConfig()):
        self.detector = SwipeDetector(config)
        self.t = 0.0
        self.x = 0.5
        self.y = 0.5

    def frame(self, x=None, y=None, pose=Pose.OPEN_PALM, aspect_ratio=1.0):
        self.t += DT
        self.x = self.x if x is None else x
        self.y = self.y if y is None else y
        hand = make_hand(x=self.x, y=self.y)
        return self.detector.update(pose, hand.landmarks, self.t, aspect_ratio)

    def hold(self, seconds, pose=Pose.OPEN_PALM):
        return [self.frame(pose=pose) for _ in range(round(seconds / DT))]

    def armed(self):
        self.hold(0.3)
        assert self.detector.phase is SwipePhase.ARMED
        return self

    def move(self, dx, dy=0.0, frames=6, pose=Pose.OPEN_PALM, aspect_ratio=1.0):
        """Move evenly over `frames` frames and return every non-None result."""
        start_x, start_y = self.x, self.y
        results = [
            self.frame(start_x + dx * i / frames, start_y + dy * i / frames, pose, aspect_ratio)
            for i in range(1, frames + 1)
        ]
        return [result for result in results if result is not None]


def test_starts_disarmed_and_arms_after_a_still_hold():
    rig = Rig()
    assert rig.detector.phase is SwipePhase.DISARMED
    rig.frame()
    assert rig.detector.phase is SwipePhase.DISARMED
    rig.hold(0.2)
    assert rig.detector.phase is SwipePhase.ARMED


def test_wrong_pose_never_arms():
    rig = Rig()
    rig.hold(1.0, Pose.POINTING)
    assert rig.detector.phase is SwipePhase.DISARMED


def test_moving_hand_never_arms():
    rig = Rig()
    assert rig.move(0.6, frames=20) == []
    assert rig.detector.phase is SwipePhase.DISARMED


@pytest.mark.parametrize("dx, direction", [(0.3, SwipeDirection.RIGHT), (-0.3, SwipeDirection.LEFT)])
def test_horizontal_swipe_is_reported_once(dx, direction):
    rig = Rig().armed()
    assert rig.move(dx) == [direction]
    assert rig.detector.phase is SwipePhase.COOLDOWN


def test_swipe_needs_an_armed_hand():
    rig = Rig()
    assert rig.move(0.3) == []


def test_swipe_needs_the_armed_pose_during_the_stroke():
    rig = Rig().armed()
    assert rig.move(0.3, pose=Pose.POINTING) == []
    assert rig.detector.phase is SwipePhase.DISARMED


@pytest.mark.parametrize("dy", [0.3, -0.3, 0.2, 0.16])
def test_diagonal_motion_is_rejected(dy):
    rig = Rig().armed()
    assert rig.move(0.3, dy) == []


def test_exactly_two_to_one_is_rejected_and_just_above_is_accepted():
    # Each ratio is checked on one straight 2-frame stroke, with a coarse grid for exact floats.
    config = SwipeConfig(min_distance=0.25)
    assert Rig(config).armed().move(0.25, 0.125, frames=2) == []
    assert Rig(config).armed().move(0.25, 0.12, frames=2) == [SwipeDirection.RIGHT]


@pytest.mark.parametrize("dy, direction", [(-0.3, SwipeDirection.UP), (0.3, SwipeDirection.DOWN)])
def test_vertical_swipe_is_reported_once_with_image_y_growing_downward(dy, direction):
    rig = Rig().armed()
    assert rig.move(0.0, dy) == [direction]
    assert rig.detector.phase is SwipePhase.COOLDOWN


@pytest.mark.parametrize("dx", [0.3, -0.3, 0.2, 0.16])
@pytest.mark.parametrize("dy", [0.3, -0.3])
def test_diagonal_motion_is_rejected_on_the_vertical_side_too(dx, dy):
    rig = Rig().armed()
    assert rig.move(dx, dy) == []


def test_vertical_exactly_two_to_one_is_rejected_and_just_above_is_accepted():
    config = SwipeConfig(min_distance=0.25)
    assert Rig(config).armed().move(0.125, -0.25, frames=2) == []
    assert Rig(config).armed().move(0.12, -0.25, frames=2) == [SwipeDirection.UP]


def test_small_sideways_drift_does_not_spoil_a_vertical_swipe():
    rig = Rig().armed()
    assert rig.move(0.05, 0.3) == [SwipeDirection.DOWN]


def test_short_vertical_motion_is_below_the_distance_threshold():
    rig = Rig().armed()
    assert rig.move(0.0, -0.1) == []


def test_slow_vertical_motion_is_below_the_speed_threshold():
    rig = Rig().armed()
    assert rig.move(0.0, 0.4, frames=60) == []
    assert rig.detector.phase is SwipePhase.ARMED


def test_aspect_ratio_scales_sideways_drift_but_not_vertical_distance():
    def armed_at(aspect_ratio):
        rig = Rig()
        for _ in range(9):
            rig.frame(aspect_ratio=aspect_ratio)
        assert rig.detector.phase is SwipePhase.ARMED
        return rig

    wide = 16 / 9
    assert armed_at(1.0).move(0.07, 0.2) == [SwipeDirection.DOWN]
    assert armed_at(wide).move(0.07, 0.2, aspect_ratio=wide) == []
    assert armed_at(wide).move(0.0, 0.1, aspect_ratio=wide) == []
    assert armed_at(wide).move(0.0, 0.2, aspect_ratio=wide) == [SwipeDirection.DOWN]


def test_swipe_after_sideways_drift_inside_the_window_is_judged_on_its_own_stroke():
    rig = Rig().armed()
    rig.move(0.1, frames=3)
    assert rig.move(0.0, -0.3, frames=4) == [SwipeDirection.UP]


@pytest.mark.parametrize(
    "dy, direction", [(-0.3, SwipeDirection.UP), (0.3, SwipeDirection.DOWN)]
)
def test_fast_vertical_return_does_not_fire_a_reverse_swipe(dy, direction):
    rig = Rig().armed()
    assert rig.move(0.0, dy) == [direction]
    assert rig.move(0.0, -dy) == []


def test_vertical_return_outlasting_the_cooldown_does_not_fire():
    rig = Rig().armed()
    assert rig.move(0.0, -0.3) == [SwipeDirection.UP]
    results = (
        rig.move(0.0, 0.3, frames=12) + rig.move(0.0, -0.3, frames=12) + rig.move(0.0, 0.3, frames=12)
    )
    assert results == []
    assert rig.detector.phase is SwipePhase.DISARMED


def test_cooldown_then_still_hold_allows_the_opposite_vertical_swipe():
    rig = Rig().armed()
    assert rig.move(0.0, -0.3) == [SwipeDirection.UP]
    rig.hold(0.6)
    rig.hold(0.2)
    assert rig.move(0.0, 0.3) == [SwipeDirection.DOWN]


def test_short_motion_is_below_the_distance_threshold():
    rig = Rig().armed()
    assert rig.move(0.1) == []


def test_slow_motion_is_below_the_speed_threshold():
    rig = Rig().armed()
    # 0.4 image heights in 2 seconds is 0.2 per second.
    assert rig.move(0.4, frames=60) == []
    assert rig.detector.phase is SwipePhase.ARMED


def test_swipe_after_a_pause_inside_the_window_is_judged_on_its_own_stroke():
    rig = Rig().armed()
    rig.move(0.0, 0.1, frames=3)  # Vertical drift that would spoil the whole-window direction.
    assert rig.move(0.3, frames=4) == [SwipeDirection.RIGHT]


def test_aspect_ratio_scales_horizontal_distance():
    config = SwipeConfig()
    wide, narrow = Rig(config).armed(), Rig(config).armed()
    assert [
        result
        for result in (wide.frame(0.5 + 0.1 * i / 6, aspect_ratio=16 / 9) for i in range(1, 7))
        if result
    ] == [SwipeDirection.RIGHT]
    assert narrow.move(0.1) == []


def test_fast_return_movement_does_not_fire_a_reverse_swipe():
    rig = Rig().armed()
    assert rig.move(0.3) == [SwipeDirection.RIGHT]
    assert rig.move(-0.3) == []


def test_return_movement_outlasting_the_cooldown_does_not_fire():
    rig = Rig().armed()
    assert rig.move(0.3) == [SwipeDirection.RIGHT]
    # Back and forth for 1.2 seconds, well past the cooldown, without ever pausing.
    results = rig.move(-0.3, frames=12) + rig.move(0.3, frames=12) + rig.move(-0.3, frames=12)
    assert results == []
    assert rig.detector.phase is SwipePhase.DISARMED


def test_a_still_hand_during_the_cooldown_does_not_arm():
    rig = Rig().armed()
    rig.move(0.3)
    rig.hold(0.3)
    assert rig.detector.phase is SwipePhase.COOLDOWN


def test_cooldown_then_still_hold_allows_the_next_swipe():
    rig = Rig().armed()
    assert rig.move(0.3) == [SwipeDirection.RIGHT]
    rig.hold(0.6)
    rig.hold(0.2)
    assert rig.detector.phase is SwipePhase.ARMED
    assert rig.move(-0.3) == [SwipeDirection.LEFT]


def test_leaving_the_pose_disarms():
    rig = Rig().armed()
    rig.frame(pose=Pose.FIST)
    assert rig.detector.phase is SwipePhase.DISARMED


def test_frame_gap_disarms_so_a_jump_is_not_a_swipe():
    rig = Rig().armed()
    rig.t += 0.5
    assert rig.frame(x=0.9) is None
    assert rig.detector.phase is SwipePhase.DISARMED


def test_reset_disarms():
    rig = Rig().armed()
    rig.detector.reset()
    assert rig.detector.phase is SwipePhase.DISARMED
    assert rig.move(0.3) == []


def test_non_finite_anchor_raises():
    rig = Rig().armed()
    with pytest.raises(ValueError):
        rig.frame(x=float("nan"))


def test_non_increasing_timestamp_raises():
    rig = Rig().armed()
    rig.t -= DT
    with pytest.raises(ValueError):
        rig.frame()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_distance": 0},
        {"min_speed": 0},
        {"dominance_ratio": 0},
        {"window_seconds": 0},
        {"arm_radius": 0},
        {"max_gap_seconds": 0},
        {"cooldown_seconds": -1},
        {"arm_hold_seconds": -1},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SwipeConfig(**kwargs)
