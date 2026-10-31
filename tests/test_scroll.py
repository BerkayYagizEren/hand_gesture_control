import pytest
import Quartz

from hand_factory import make_hand
from src.controller.base import MockMouseController, Point, ScreenBounds, ScrollEvent
from src.controller.quartz_mouse import QuartzMouseController
from src.controller.scroll import ScrollConfig, ScrollController
from src.tracking.cursor_controller import CursorController
from src.tracking.state_machine import GestureFrame, GestureState, GestureStateMachine
from src.vision.classifier import Pose, PoseObservation

DT = 1 / 30
BOUNDS = ScreenBounds(0, 0, 1512, 982)


def landmarks(y):
    return make_hand(y=y).landmarks


def total_dy(mouse):
    return sum(event.dy for event in mouse.scroll_events)


class Scroller:
    def __init__(self, config=ScrollConfig()):
        self.mouse = MockMouseController()
        self.scroll = ScrollController(self.mouse, config)
        self.t = 0.0

    def run(self, ys):
        for y in ys:
            self.t += DT
            self.scroll.update(landmarks(y), self.t)


def ramp(start, stop, count):
    return [start + (stop - start) * i / (count - 1) for i in range(count)]


# Mouse driver


def test_scroll_records_whole_pixels_without_moving_the_cursor():
    mouse = MockMouseController(start=Point(100, 100))
    mouse.scroll(1.6, -3.2)
    assert mouse.scroll_events == [ScrollEvent(2, -3)]
    assert mouse.position == Point(100, 100)
    assert mouse.events == []


def test_scroll_of_less_than_half_a_pixel_posts_nothing():
    mouse = MockMouseController()
    mouse.scroll(0.2, -0.4)
    assert mouse.scroll_events == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_scroll_rejects_non_finite_values(bad):
    mouse = MockMouseController()
    with pytest.raises(ValueError):
        mouse.scroll(0, bad)
    assert mouse.scroll_events == []


def test_quartz_scroll_posts_a_pixel_wheel_event():
    posted = []
    QuartzMouseController(post_event=posted.append).scroll(3, -7)
    assert len(posted) == 1
    event = posted[0]
    assert Quartz.CGEventGetType(event) == Quartz.kCGEventScrollWheel
    assert Quartz.CGEventGetIntegerValueField(event, Quartz.kCGScrollWheelEventIsContinuous) == 1
    assert Quartz.CGEventGetIntegerValueField(event, Quartz.kCGScrollWheelEventPointDeltaAxis1) == -7
    assert Quartz.CGEventGetIntegerValueField(event, Quartz.kCGScrollWheelEventPointDeltaAxis2) == 3


# ScrollController


def test_first_frame_only_anchors():
    rig = Scroller()
    rig.run([0.9])
    assert rig.mouse.scroll_events == []


def test_hand_up_scrolls_content_up_by_default():
    rig = Scroller()
    rig.run(ramp(0.7, 0.3, 20))
    assert total_dy(rig.mouse) < 0
    assert all(event.dx == 0 for event in rig.mouse.scroll_events)


def test_hand_down_scrolls_the_other_way():
    rig = Scroller()
    rig.run(ramp(0.3, 0.7, 20))
    assert total_dy(rig.mouse) > 0


def test_natural_false_inverts_the_direction():
    rig = Scroller(ScrollConfig(natural=False))
    rig.run(ramp(0.7, 0.3, 20))
    assert total_dy(rig.mouse) > 0


def test_jitter_inside_the_deadzone_does_not_scroll():
    rig = Scroller()
    rig.run([0.5 + (0.004 if i % 2 else -0.004) for i in range(60)])
    assert rig.mouse.scroll_events == []


def test_jitter_after_a_move_nudges_at_most_once_by_its_amplitude():
    rig = Scroller()
    rig.run(ramp(0.3, 0.6, 15) + [0.6] * 60)  # The filter settles at the end of the move.
    jitter = [0.6 + (0.003 if i % 2 else -0.003) for i in range(60)]
    settled = total_dy(rig.mouse)
    rig.run(jitter)
    nudged = total_dy(rig.mouse)
    assert 0 <= nudged - settled <= 0.003 * ScrollConfig().speed_multiplier
    events = len(rig.mouse.scroll_events)
    rig.run(jitter)
    assert len(rig.mouse.scroll_events) == events


def test_scrolled_distance_is_travel_past_the_deadzone_times_the_multiplier():
    config = ScrollConfig(speed_multiplier=1000.0, deadzone=0.01)
    rig = Scroller(config)
    rig.run(ramp(0.3, 0.6, 15) + [0.6] * 120)
    assert total_dy(rig.mouse) == pytest.approx((0.3 - 0.01) * 1000, abs=2)


def test_speed_multiplier_scales_the_output():
    slow, fast = Scroller(ScrollConfig(speed_multiplier=500.0)), Scroller(ScrollConfig(speed_multiplier=1500.0))
    for rig in (slow, fast):
        rig.run(ramp(0.3, 0.6, 15) + [0.6] * 120)
    assert total_dy(fast.mouse) == pytest.approx(3 * total_dy(slow.mouse), abs=3)


def test_slow_motion_accumulates_sub_pixel_steps():
    # One pixel of travel takes several frames, so rounding each frame would give zero.
    config = ScrollConfig(speed_multiplier=100.0, deadzone=0.0, beta=0.0, min_cutoff=30.0)
    rig = Scroller(config)
    rig.run(ramp(0.5, 0.6, 100))
    assert total_dy(rig.mouse) == pytest.approx(10, abs=1)


def test_smoothing_spreads_a_landmark_jump_over_frames():
    rig = Scroller()
    rig.run([0.5] * 10 + [0.6])
    first_step = total_dy(rig.mouse)
    rig.run([0.6] * 60)
    assert 0 < first_step < total_dy(rig.mouse)


def test_large_frame_gap_re_anchors_instead_of_jumping():
    rig = Scroller()
    rig.run([0.5] * 5)
    rig.t += 1.0
    rig.run([0.9, 0.9])
    assert rig.mouse.scroll_events == []


def test_reset_re_anchors_and_drops_the_fraction():
    rig = Scroller()
    rig.run(ramp(0.3, 0.4, 5))
    before = total_dy(rig.mouse)
    rig.scroll.reset()
    rig.run([0.9])
    assert total_dy(rig.mouse) == before


@pytest.mark.parametrize(
    "kwargs", [{"speed_multiplier": 0}, {"deadzone": -0.1}, {"max_gap_seconds": 0}]
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ScrollConfig(**kwargs)


# State machine wiring


class Rig:
    def __init__(self):
        self.mouse = MockMouseController(BOUNDS, Point(700, 500))
        self.fsm = GestureStateMachine(self.mouse, CursorController(BOUNDS))
        self.t = 0.0

    def frame(self, pose=Pose.POINTING, y=0.5):
        self.t += DT
        frame = GestureFrame(PoseObservation(pose, 1.0, 1.0), landmarks(y))
        return self.fsm.update(frame, self.t)

    def scrolling(self, y=0.5):
        for _ in range(5):
            self.frame(y=y)
        for _ in range(3):
            state = self.frame(Pose.TWO_FINGER, y=y)
        assert state is GestureState.SCROLL_ACTIVE
        return self


def test_two_finger_motion_scrolls_and_holds_the_cursor():
    rig = Rig().scrolling()
    position, events = rig.mouse.position, len(rig.mouse.events)
    for y in ramp(0.5, 0.2, 15):
        rig.frame(Pose.TWO_FINGER, y=y)
    assert total_dy(rig.mouse) < 0
    assert rig.mouse.position == position
    assert len(rig.mouse.events) == events
    assert rig.mouse.held_buttons == frozenset()


def test_entering_scroll_does_not_jump_from_the_earlier_hand_position():
    rig = Rig()
    for _ in range(5):
        rig.frame(y=0.2)
    for _ in range(3):
        rig.frame(Pose.TWO_FINGER, y=0.2)
    rig.frame(Pose.TWO_FINGER, y=0.2)
    assert rig.mouse.scroll_events == []


def test_tracking_does_not_scroll():
    rig = Rig()
    for y in ramp(0.5, 0.2, 15):
        rig.frame(y=y)
    assert rig.mouse.scroll_events == []


def test_a_frame_with_another_pose_does_not_scroll_and_keeps_the_mode():
    rig = Rig().scrolling()
    for y in ramp(0.5, 0.48, 3):
        rig.frame(Pose.TWO_FINGER, y=y)
    before = total_dy(rig.mouse)
    assert rig.frame(Pose.UNKNOWN, y=0.3) is GestureState.SCROLL_ACTIVE
    assert total_dy(rig.mouse) == before
    assert rig.frame(Pose.TWO_FINGER, y=0.3) is GestureState.SCROLL_ACTIVE
    assert total_dy(rig.mouse) < before


def test_leaving_the_pose_ends_scrolling_with_a_cooldown():
    rig = Rig().scrolling()
    states = [rig.frame(y=0.2) for _ in range(3)]
    assert states[-1] is GestureState.COOLDOWN
    assert rig.mouse.scroll_events == []


def test_hand_loss_stops_scrolling_and_a_new_scroll_re_anchors():
    rig = Rig().scrolling(y=0.5)
    rig.t += 0.3
    assert rig.fsm.update(None, rig.t) is GestureState.IDLE
    rig.scrolling(y=0.1)
    assert rig.mouse.scroll_events == []


def test_non_finite_landmark_while_scrolling_resets_the_machine():
    rig = Rig().scrolling()
    with pytest.raises(ValueError):
        rig.frame(Pose.TWO_FINGER, y=float("nan"))
    assert rig.fsm.state is GestureState.IDLE
    assert rig.fsm.reset() is None
