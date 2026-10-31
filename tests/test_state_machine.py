import math

import pytest

from hand_factory import PARTIAL, POSES, make_hand, world_points
from src.controller.base import MockMouseController, MouseButton, MouseEventType, Point, ScreenBounds
from src.tracking.cursor_controller import CursorController
from src.tracking.state_machine import GestureConfig, GestureFrame, GestureState, GestureStateMachine
from src.vision.classifier import Pose, PoseObservation
from src.vision.detector import DetectionResult
from src.vision.recorder import LandmarkRecorder
from src.vision.replayer import LandmarkReplayer

BOUNDS = ScreenBounds(0, 0, 1512, 982)
START = Point(700, 500)
DT = 1 / 30


class Rig:
    """A machine on a mock mouse with its own frame clock."""

    def __init__(self, config=GestureConfig()):
        self.mouse = MockMouseController(BOUNDS, START, clock=lambda: self.t)
        self.cursor = CursorController(BOUNDS)
        self.fsm = GestureStateMachine(self.mouse, self.cursor, config)
        self.t = 0.0
        self.x = 0.5

    def frame(self, pose=Pose.POINTING, pinch=1.0, middle=1.0, x=None):
        self.t += DT
        if x is not None:
            self.x = x
        observation = PoseObservation(pose, pinch, middle)
        landmarks = make_hand(x=self.x).landmarks
        return self.fsm.update(GestureFrame(observation, landmarks), self.t)

    def frames(self, count, *args, **kwargs):
        for _ in range(count):
            state = self.frame(*args, **kwargs)
        return state

    def no_hand(self, seconds=DT):
        self.t += seconds
        return self.fsm.update(None, self.t)

    def tracking(self):
        self.frames(5)
        assert self.fsm.state is GestureState.TRACKING
        return self

    def pinching(self):
        self.tracking()
        assert self.frames(3, Pose.PINCH, pinch=0.1) is GestureState.PINCH_ACTIVE
        return self

    def event_types(self):
        return [event.type for event in self.mouse.events]

    def button_events(self):
        # The filter's rounding can post sub-point moves even for a still hand.
        return [t for t in self.event_types() if t is not MouseEventType.MOVED]


def test_starts_idle_and_a_hand_starts_tracking():
    rig = Rig()
    assert rig.fsm.state is GestureState.IDLE
    assert rig.frame() is GestureState.TRACKING


def test_no_hand_stays_idle():
    rig = Rig()
    assert rig.no_hand() is GestureState.IDLE
    assert rig.mouse.events == []


def test_tracking_starts_from_the_os_cursor_position():
    rig = Rig()
    rig.frame()
    assert rig.cursor.position == START


def test_tracking_moves_the_cursor_without_buttons():
    rig = Rig().tracking()
    before = rig.mouse.position
    for i in range(5):
        rig.frame(x=0.5 + 0.02 * i)
    assert rig.mouse.position.x > before.x
    assert set(rig.event_types()) == {MouseEventType.MOVED}


def test_voting_delays_pinch_and_ignores_single_frame():
    rig = Rig().tracking()
    assert rig.frame(Pose.PINCH, pinch=0.1) is GestureState.TRACKING
    assert rig.frame(Pose.POINTING) is GestureState.TRACKING
    assert rig.mouse.held_buttons == frozenset()
    rig.frames(2, Pose.PINCH, pinch=0.1)
    assert rig.fsm.state is GestureState.PINCH_ACTIVE


def test_quick_pinch_is_one_click_without_any_drag():
    rig = Rig().pinching()
    rig.frames(2, Pose.PINCH, pinch=0.9)
    assert rig.fsm.state is GestureState.COOLDOWN
    assert rig.button_events() == [MouseEventType.LEFT_DOWN, MouseEventType.LEFT_UP]
    assert not rig.fsm.is_dragging


def test_click_does_not_slide_under_small_hand_jitter():
    rig = Rig().pinching()
    press = rig.mouse.position
    rig.frame(Pose.PINCH, pinch=0.1, x=rig.x + 0.0005)
    rig.frames(2, Pose.PINCH, pinch=0.9, x=rig.x + 0.0005)
    up = rig.mouse.events[-1]
    assert up.type is MouseEventType.LEFT_UP
    assert (up.x, up.y) == press
    assert rig.cursor.position == press


def test_hysteresis_keeps_pinch_between_the_thresholds():
    rig = Rig().pinching()
    assert (
        rig.frames(5, Pose.UNKNOWN, pinch=rig.fsm.config.pinch_release_ratio - 0.02)
        is GestureState.PINCH_ACTIVE
    )
    assert rig.mouse.held_buttons == {MouseButton.LEFT}
    assert (
        rig.frames(2, Pose.UNKNOWN, pinch=rig.fsm.config.pinch_release_ratio + 0.02)
        is GestureState.COOLDOWN
    )
    assert rig.mouse.held_buttons == frozenset()


def test_single_frame_ratio_spike_does_not_release():
    rig = Rig().pinching()
    for _ in range(4):
        assert rig.frame(Pose.PINCH, pinch=0.9) is GestureState.PINCH_ACTIVE
        assert rig.frame(Pose.PINCH, pinch=0.1) is GestureState.PINCH_ACTIVE


def test_ratio_between_thresholds_does_not_start_a_pinch():
    rig = Rig().tracking()
    assert rig.frames(5, Pose.PINCH, pinch=0.60) is GestureState.TRACKING
    assert rig.mouse.held_buttons == frozenset()


def test_nan_ratio_releases():
    rig = Rig().pinching()
    assert rig.frames(2, Pose.UNKNOWN, pinch=math.nan) is GestureState.COOLDOWN
    assert rig.mouse.held_buttons == frozenset()


def test_holding_past_drag_hold_becomes_a_drag_without_motion():
    rig = Rig(GestureConfig(drag_hold_ms=100)).pinching()
    assert not rig.fsm.is_dragging
    rig.frames(4, Pose.PINCH, pinch=0.1)
    assert rig.fsm.is_dragging


def test_motion_becomes_a_drag_before_the_hold_time():
    rig = Rig(GestureConfig(drag_hold_ms=100_000)).pinching()
    x = rig.x
    for _ in range(15):
        x += 0.02
        rig.frame(Pose.PINCH, pinch=0.1, x=x)
    assert rig.fsm.is_dragging
    after_press = rig.event_types()[rig.event_types().index(MouseEventType.LEFT_DOWN) + 1 :]
    assert set(after_press) == {MouseEventType.LEFT_DRAGGED}


def test_pinch_freeze_holds_the_cursor_at_first():
    rig = Rig().pinching()
    press = rig.mouse.position
    rig.frame(Pose.PINCH, pinch=0.1, x=rig.x + 0.1)
    assert rig.cursor.position == press


def test_drag_release_ends_the_drag():
    rig = Rig(GestureConfig(drag_hold_ms=0)).pinching()
    x = rig.x
    for _ in range(10):
        x += 0.02
        rig.frame(Pose.PINCH, pinch=0.1, x=x)
    rig.frames(2, Pose.PINCH, pinch=0.9, x=x)
    assert not rig.fsm.is_dragging
    drag, up = rig.mouse.events[-2:]
    assert drag.type is MouseEventType.LEFT_DRAGGED
    assert up.type is MouseEventType.LEFT_UP
    assert (up.x, up.y) == (drag.x, drag.y)


def test_stale_vote_does_not_repress_after_cooldown():
    # A very short cooldown: the vote still says PINCH but the fingers are open.
    rig = Rig(GestureConfig(cooldown_ms=0)).pinching()
    rig.frames(2, Pose.UNKNOWN, pinch=0.9)
    assert rig.frame(Pose.UNKNOWN, pinch=0.9) is GestureState.TRACKING
    assert rig.event_types().count(MouseEventType.LEFT_DOWN) == 1


def test_cooldown_blocks_a_new_press_then_expires():
    rig = Rig(GestureConfig(cooldown_ms=100)).pinching()
    rig.frames(2, Pose.PINCH, pinch=0.9)
    assert rig.frame(Pose.PINCH, pinch=0.1) is GestureState.COOLDOWN
    assert rig.event_types().count(MouseEventType.LEFT_DOWN) == 1
    rig.frames(3, Pose.PINCH, pinch=0.1)
    assert rig.fsm.state is GestureState.PINCH_ACTIVE
    assert rig.event_types().count(MouseEventType.LEFT_DOWN) == 2


def test_cursor_moves_during_cooldown():
    rig = Rig(GestureConfig(cooldown_ms=10_000)).pinching()
    rig.frames(2, Pose.PINCH, pinch=0.9)
    before = rig.mouse.position
    x = rig.x
    for _ in range(5):
        x += 0.02
        assert rig.frame(Pose.POINTING, x=x) is GestureState.COOLDOWN
    assert rig.mouse.position != before


def test_two_quick_pinches_make_a_double_click():
    rig = Rig().pinching()
    rig.frames(2, Pose.UNKNOWN, pinch=0.9)
    rig.frames(4, Pose.POINTING)
    rig.frames(6, Pose.PINCH, pinch=0.1)
    assert rig.fsm.state is GestureState.PINCH_ACTIVE
    downs = [e for e in rig.mouse.events if e.type is MouseEventType.LEFT_DOWN]
    assert [down.click_state for down in downs] == [1, 2]


def test_right_click_press_and_release():
    rig = Rig().tracking()
    assert rig.frames(3, Pose.MIDDLE_PINCH, middle=0.1) is GestureState.RIGHT_CLICK_ACTIVE
    assert rig.mouse.held_buttons == {MouseButton.RIGHT}
    assert rig.frames(5, Pose.MIDDLE_PINCH, middle=0.39) is GestureState.RIGHT_CLICK_ACTIVE
    assert rig.frames(2, Pose.UNKNOWN, middle=0.75) is GestureState.COOLDOWN
    assert rig.button_events() == [MouseEventType.RIGHT_DOWN, MouseEventType.RIGHT_UP]


def test_right_click_ignores_hand_motion():
    rig = Rig().tracking()
    rig.frames(3, Pose.MIDDLE_PINCH, middle=0.1)
    x = rig.x
    for _ in range(5):
        x += 0.05
        rig.frame(Pose.MIDDLE_PINCH, middle=0.1, x=x)
    assert MouseEventType.RIGHT_DRAGGED not in rig.event_types()


def test_cursor_does_not_jump_after_right_click():
    rig = Rig().tracking()
    position = rig.mouse.position
    rig.frames(3, Pose.MIDDLE_PINCH, middle=0.1, x=rig.x)
    rig.frames(3, Pose.MIDDLE_PINCH, middle=0.1, x=0.8)
    rig.frames(2, Pose.UNKNOWN, middle=0.9, x=0.8)
    rig.frame(Pose.POINTING, x=0.8)
    assert rig.mouse.position == position


def test_scroll_holds_the_cursor_and_ends_with_cooldown():
    rig = Rig().tracking()
    assert rig.frames(3, Pose.TWO_FINGER) is GestureState.SCROLL_ACTIVE
    position = rig.mouse.position
    rig.frames(10, Pose.TWO_FINGER, x=0.8)
    assert rig.mouse.position == position
    assert rig.frames(3, Pose.POINTING, x=0.8) is GestureState.COOLDOWN
    assert rig.mouse.position == position


def test_clutch_freezes_and_repositions_the_hand():
    rig = Rig().tracking()
    assert rig.frames(3, Pose.FIST) is GestureState.CLUTCH_ACTIVE
    assert rig.cursor.clutch
    position = rig.mouse.position
    rig.frames(10, Pose.FIST, x=0.9)
    assert rig.frames(3, Pose.POINTING, x=0.9) is GestureState.TRACKING
    assert not rig.cursor.clutch
    assert rig.mouse.position == position


@pytest.mark.parametrize("pose", [Pose.OPEN_PALM, Pose.THUMBS_UP, Pose.UNKNOWN])
def test_other_poses_only_track(pose):
    rig = Rig().tracking()
    assert rig.frames(6, pose) is GestureState.TRACKING
    assert rig.mouse.held_buttons == frozenset()


def test_modes_are_exclusive_while_pinching():
    rig = Rig().pinching()
    for pose in (Pose.TWO_FINGER, Pose.FIST, Pose.MIDDLE_PINCH):
        assert rig.frames(5, pose, pinch=0.1, middle=0.1) is GestureState.PINCH_ACTIVE
    assert rig.mouse.held_buttons == {MouseButton.LEFT}


def test_short_dropout_keeps_the_pinch():
    rig = Rig().pinching()
    assert rig.no_hand(0.1) is GestureState.PINCH_ACTIVE
    assert rig.mouse.held_buttons == {MouseButton.LEFT}
    assert rig.frame(Pose.PINCH, pinch=0.1) is GestureState.PINCH_ACTIVE


def test_hand_loss_times_out_at_the_configured_time():
    rig = Rig().pinching()
    assert rig.no_hand(0.15) is GestureState.PINCH_ACTIVE
    assert rig.no_hand(0.06) is GestureState.IDLE
    assert rig.mouse.held_buttons == frozenset()
    assert rig.event_types()[-1] is MouseEventType.LEFT_UP


def test_dropout_timer_is_not_extended_by_empty_frames():
    rig = Rig().pinching()
    for _ in range(7):
        rig.no_hand(DT)
    assert rig.fsm.state is GestureState.IDLE
    assert rig.mouse.held_buttons == frozenset()


def test_hand_loss_during_drag_releases():
    rig = Rig(GestureConfig(drag_hold_ms=0)).pinching()
    rig.frame(Pose.PINCH, pinch=0.1)
    assert rig.fsm.is_dragging
    rig.no_hand(0.3)
    assert rig.mouse.held_buttons == frozenset()
    assert not rig.fsm.is_dragging


def test_hand_loss_during_right_click_releases():
    rig = Rig().tracking()
    rig.frames(3, Pose.MIDDLE_PINCH, middle=0.1)
    rig.no_hand(0.3)
    assert rig.mouse.held_buttons == frozenset()
    assert rig.event_types()[-1] is MouseEventType.RIGHT_UP


def test_camera_stall_releases_when_the_hand_frame_finally_arrives():
    rig = Rig().pinching()
    rig.t += 1.0
    state = rig.frame(Pose.PINCH, pinch=0.1)
    assert state is GestureState.TRACKING
    assert MouseEventType.LEFT_UP in rig.event_types()
    assert rig.mouse.held_buttons == frozenset()


def test_hand_loss_clears_the_vote_so_a_returning_hand_cannot_press_at_once():
    rig = Rig().pinching()
    rig.no_hand(0.3)
    assert rig.frame(Pose.PINCH, pinch=0.1) is GestureState.TRACKING
    assert rig.mouse.held_buttons == frozenset()


def test_hand_loss_clears_the_clutch():
    rig = Rig().tracking()
    rig.frames(3, Pose.FIST)
    rig.no_hand(0.3)
    assert rig.fsm.state is GestureState.IDLE
    assert not rig.cursor.clutch


def test_returning_hand_adopts_the_physical_mouse_position():
    rig = Rig().tracking()
    rig.no_hand(0.3)
    rig.mouse.os_position = Point(50, 60)
    rig.frame()
    assert rig.cursor.position == (50, 60)


def test_reset_releases_every_button():
    rig = Rig().pinching()
    rig.mouse.press_right()
    rig.fsm.reset()
    assert rig.fsm.state is GestureState.IDLE
    assert rig.mouse.held_buttons == frozenset()


def test_context_manager_releases_on_exit_and_on_error():
    rig = Rig().pinching()
    with pytest.raises(KeyboardInterrupt):
        with rig.fsm:
            raise KeyboardInterrupt
    assert rig.mouse.held_buttons == frozenset()
    assert rig.fsm.state is GestureState.IDLE


def test_error_during_a_frame_releases_and_propagates():
    rig = Rig().pinching()
    rig.cursor.update = lambda *args: 1 / 0
    with pytest.raises(ZeroDivisionError):
        rig.frame(Pose.PINCH, pinch=0.1)
    assert rig.mouse.held_buttons == frozenset()
    assert rig.fsm.state is GestureState.IDLE


def test_error_while_classifying_releases_and_propagates():
    rig = Rig().pinching()
    rig.fsm.classifier.classify = lambda hand: 1 / 0
    with pytest.raises(ZeroDivisionError):
        rig.fsm.process(DetectionResult(int(rig.t * 1000) + 33, 640, 480, (make_hand(),)))
    assert rig.mouse.held_buttons == frozenset()


def test_failed_release_is_retried_on_the_next_frame():
    rig = Rig().pinching()
    real_post = rig.mouse._post
    # Fails the release and the reset that update() attempts after it.
    failures = [RuntimeError("post failed"), RuntimeError("post failed")]

    def flaky(event):
        if event.type is MouseEventType.LEFT_UP and failures:
            raise failures.pop()
        real_post(event)

    rig.mouse._post = flaky
    with pytest.raises(RuntimeError):
        rig.no_hand(0.3)
    assert rig.mouse.held_buttons == {MouseButton.LEFT}
    assert rig.fsm.state is GestureState.IDLE
    rig.no_hand()
    assert rig.mouse.held_buttons == frozenset()


def test_mouse_up_always_pairs_with_mouse_down():
    rig = Rig().pinching()
    rig.frames(2, Pose.UNKNOWN, pinch=0.9)
    rig.no_hand(0.3)
    rig.fsm.reset()
    types = rig.event_types()
    assert types.count(MouseEventType.LEFT_DOWN) == types.count(MouseEventType.LEFT_UP) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pinch_start_ratio": 0.5, "pinch_release_ratio": 0.4},
        {"pinch_start_ratio": 0.0},
        {"release_frames": 0},
        {"drag_hold_ms": -1},
        {"hand_loss_timeout_ms": 0},
    ],
)
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        GestureConfig(**kwargs)


def detection(timestamp_ms, world=None, x=0.5):
    hands = () if world is None else (make_hand(world, x=x),)
    return DetectionResult(timestamp_ms, 640, 480, hands)


def scripted_session():
    """Point, pinch and drag, lose the hand mid-drag, return, then make a clean click."""
    pinch = POSES["PINCH"]
    script = [(POSES["POINTING"], 0.5 + i * 0.002) for i in range(10)]
    script += [(pinch, 0.52)] * 4
    script += [(pinch, 0.52 + i * 0.02) for i in range(12)]
    script += [(None, 0.0)] * 8
    script += [(POSES["POINTING"], 0.6)] * 6
    script += [(pinch, 0.6)] * 4
    script += [(POSES["OPEN_PALM"], 0.6)] * 4
    return [detection(33 * i + 33, world, x) for i, (world, x) in enumerate(script)]


def run_session(results):
    mouse = MockMouseController(BOUNDS, START, clock=lambda: 0.0)
    fsm = GestureStateMachine(mouse, CursorController(BOUNDS))
    states = [fsm.process(result) for result in results]
    return mouse, states


def test_process_runs_the_whole_chain_from_detector_output():
    mouse, states = run_session(scripted_session())
    types = [event.type for event in mouse.events]
    assert GestureState.PINCH_ACTIVE in states
    assert GestureState.IDLE in states
    assert MouseEventType.LEFT_DRAGGED in types
    assert types.count(MouseEventType.LEFT_DOWN) == 2
    assert types.count(MouseEventType.LEFT_UP) == 2
    assert mouse.held_buttons == frozenset()


def test_replayed_session_produces_identical_events(tmp_path):
    results = scripted_session()
    path = tmp_path / "session.jsonl"
    with LandmarkRecorder(path) as recorder:
        for result in results:
            recorder.record(result)
    live, _ = run_session(results)
    replayed, _ = run_session(list(LandmarkReplayer(path)))
    assert replayed.events == live.events
    assert live.events


def test_process_with_no_hands_counts_as_hand_loss():
    mouse = MockMouseController(BOUNDS, START, clock=lambda: 0.0)
    fsm = GestureStateMachine(mouse, CursorController(BOUNDS))
    fsm.process(detection(33, POSES["POINTING"]))
    assert fsm.process(detection(400)) is GestureState.IDLE


def test_fist_and_scroll_poses_reach_their_states_through_the_classifier():
    mouse = MockMouseController(BOUNDS, START, clock=lambda: 0.0)
    fsm = GestureStateMachine(mouse, CursorController(BOUNDS))
    time = 0
    for world, expected in (
        (POSES["FIST"], GestureState.CLUTCH_ACTIVE),
        (POSES["OPEN_PALM"], GestureState.TRACKING),
        (POSES["TWO_FINGER"], GestureState.SCROLL_ACTIVE),
        (world_points(index=PARTIAL, pinch=0.1), GestureState.COOLDOWN),
        (POSES["MIDDLE_PINCH"], GestureState.RIGHT_CLICK_ACTIVE),
    ):
        for _ in range(9):
            time += 33
            state = fsm.process(detection(time, world))
            if state is expected:
                break
        assert state is expected
    assert mouse.held_buttons == {MouseButton.RIGHT}
