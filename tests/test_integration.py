"""Runs the whole pipeline from recorded landmarks to mock drivers, without a camera."""

import signal
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from hand_factory import POSES, make_hand
from src.camera.stream import CameraError, Frame
from src.controller.base import MockMouseController, MouseButton, MouseEventType, Point
from src.controller.media import MediaKey, MockMediaController
from src.controller.presentation import PresentationController
from src.main import (
    KEY_ESC,
    KEY_NONE,
    AppMode,
    GestureControlApp,
    LiveSource,
    ReplaySource,
    Tick,
    build_parser,
    main,
)
from src.tracking.state_machine import GestureState
from src.vision.detector import DetectionResult
from src.vision.recorder import LandmarkRecorder

DT_MS = 33
ROOT = Path(__file__).resolve().parents[1]


class Script:
    """Builds a timed sequence of detector results."""

    def __init__(self):
        self.results = []
        self.t_ms = 1000

    def hand(self, pose, x=0.3, y=0.5):
        hand = make_hand(POSES[pose], x=x, y=y)
        self.results.append(DetectionResult(self.t_ms, 640, 480, (hand,)))
        self.t_ms += DT_MS
        return self

    def hold(self, pose, frames, x=0.3, y=0.5):
        for _ in range(frames):
            self.hand(pose, x, y)
        return self

    def move(self, pose, frames, start=(0.3, 0.5), stop=(0.3, 0.5)):
        for i in range(frames):
            t = i / (frames - 1)
            self.hand(pose, start[0] + (stop[0] - start[0]) * t, start[1] + (stop[1] - start[1]) * t)
        return self

    def no_hand(self, frames):
        for _ in range(frames):
            self.results.append(DetectionResult(self.t_ms, 640, 480, ()))
            self.t_ms += DT_MS
        return self

    def save(self, path):
        with LandmarkRecorder(path) as recorder:
            for result in self.results:
                recorder.record(result)
        return path

    def ticks(self):
        return [Tick(result.timestamp_ms / 1000, result) for result in self.results]


class ListSource:
    """Yields prepared ticks, optionally running a hook before one of them."""

    def __init__(self, ticks, before=None):
        self.ticks = ticks
        self.before = before or {}
        self.opened = False
        self.closed = 0

    def open(self):
        self.opened = True

    def frames(self, profiler):
        for i, tick in enumerate(self.ticks):
            if i in self.before:
                self.before[i]()
            yield tick

    def close(self):
        self.closed += 1


class FakeWindow:
    def __init__(self, keys=()):
        self.keys = list(keys)
        self.shown = []
        self.polled = 0
        self.closed = 0

    def _next(self):
        return self.keys.pop(0) if self.keys else KEY_NONE

    def show(self, image):
        self.shown.append(image)
        return self._next()

    def poll(self):
        self.polled += 1
        return self._next()

    def close(self):
        self.closed += 1


class SentEvents:
    def __init__(self):
        self.events = []

    def __call__(self, event):
        self.events.append(event)


def make_app(source, mode=AppMode.MOUSE, window=None, record_path=None):
    return GestureControlApp(
        source,
        MockMouseController(start=Point(960, 540)),
        MockMediaController(),
        PresentationController(post_event=SentEvents()),
        mode=mode,
        window=window,
        record_path=record_path,
    )


def replay_app(path, mode=AppMode.MOUSE, window=None, realtime=False):
    return make_app(ReplaySource(path, realtime=realtime), mode, window)


def event_types(mouse):
    return [event.type for event in mouse.events]


def mouse_session():
    return (
        Script()
        .move("POINTING", 15, (0.3, 0.5), (0.4, 0.55))  # Cursor motion.
        .move("PINCH", 15, (0.4, 0.55), (0.5, 0.6))  # Press, hold, drag.
        .hold("POINTING", 12, 0.5, 0.6)  # Release and cooldown.
        .hand("MIDDLE_PINCH")
        .hold("MIDDLE_PINCH", 5)
        .hold("POINTING", 12)
        .move("TWO_FINGER", 20, (0.3, 0.5), (0.3, 0.35))  # Scroll.
        .hold("POINTING", 12)
        .hold("FIST", 8)  # Clutch.
        .hold("POINTING", 8)
        .no_hand(10)  # Hand loss.
    )


# Mouse mode, end to end


def test_mouse_replay_triggers_every_state_and_shuts_down_clean(tmp_path):
    path = mouse_session().save(tmp_path / "mouse.jsonl")
    app = replay_app(path)
    summary = app.run()

    assert summary.frames == len(mouse_session().results)
    assert summary.hand_frames == summary.frames - 10
    assert {
        state.name
        for state in (
            GestureState.TRACKING,
            GestureState.PINCH_ACTIVE,
            GestureState.RIGHT_CLICK_ACTIVE,
            GestureState.SCROLL_ACTIVE,
            GestureState.CLUTCH_ACTIVE,
            GestureState.COOLDOWN,
            GestureState.IDLE,
        )
    } <= summary.states_seen

    types = event_types(app.mouse)
    assert types.count(MouseEventType.LEFT_DOWN) == types.count(MouseEventType.LEFT_UP) == 1
    assert types.count(MouseEventType.RIGHT_DOWN) == types.count(MouseEventType.RIGHT_UP) == 1
    assert MouseEventType.MOVED in types
    assert MouseEventType.LEFT_DRAGGED in types
    assert app.mouse.scroll_events
    assert app.mouse.held_buttons == frozenset()
    assert app.fsm.state is GestureState.IDLE


def test_replay_is_deterministic(tmp_path):
    path = mouse_session().save(tmp_path / "mouse.jsonl")
    first, second = replay_app(path), replay_app(path)
    first.run()
    second.run()
    assert first.mouse.events == second.mouse.events
    assert first.mouse.scroll_events == second.mouse.scroll_events


def test_replay_ending_mid_drag_releases_the_button(tmp_path):
    path = Script().hold("POINTING", 3).hold("PINCH", 15).save(tmp_path / "cut.jsonl")
    app = replay_app(path)
    app.run()
    assert app.mouse.events[-1].type is MouseEventType.LEFT_UP
    assert app.mouse.held_buttons == frozenset()


def test_hand_loss_during_pinch_releases_within_the_timeout():
    script = Script().hold("POINTING", 3).hold("PINCH", 8).no_hand(8)
    released_at = []
    app = make_app(ListSource(script.ticks()))
    original = app.mouse._post

    def post(event):
        if event.type is MouseEventType.LEFT_UP:
            released_at.append(app.summary.frames)
        original(event)

    app.mouse._post = post
    app.run()
    # Frames arrive 33 ms apart, so a 200 ms timeout fires on the 7th frame after the last hand.
    assert released_at == [3 + 8 + 7]


# Presentation mode


def swipe_session(direction=1):
    start, stop = (0.3, 0.5), (0.7, 0.5)
    if direction < 0:
        start, stop = stop, start
    return Script().hold("OPEN_PALM", 12, *start).move("OPEN_PALM", 8, start, stop)


def test_presentation_replay_swipes_change_slides(tmp_path):
    script = swipe_session(1).hold("OPEN_PALM", 30, 0.7, 0.5).move(
        "OPEN_PALM", 8, (0.7, 0.5), (0.3, 0.5)
    )
    app = replay_app(script.save(tmp_path / "slides.jsonl"), AppMode.PRESENTATION)
    summary = app.run()
    assert summary.actions == {"next_slide": 1, "previous_slide": 1}
    assert {"DISARMED", "ARMED", "COOLDOWN"} <= summary.states_seen
    # A tap is a key down and a key up.
    assert len(app.presentation._post_event.events) == 4
    assert app.mouse.events == []


def test_presentation_mode_ignores_pinches(tmp_path):
    path = Script().hold("POINTING", 3).hold("PINCH", 20).save(tmp_path / "pinch.jsonl")
    app = replay_app(path, AppMode.PRESENTATION)
    app.run()
    assert app.mouse.events == []
    assert app.summary.actions == {}


# Media mode


@pytest.mark.parametrize(
    "start, stop, action, key",
    [
        ((0.3, 0.5), (0.7, 0.5), "next_track", MediaKey.NEXT_TRACK),
        ((0.7, 0.5), (0.3, 0.5), "prev_track", MediaKey.PREV_TRACK),
        ((0.5, 0.7), (0.5, 0.3), "volume_up", MediaKey.VOLUME_UP),
        ((0.5, 0.3), (0.5, 0.7), "volume_down", MediaKey.VOLUME_DOWN),
    ],
)
def test_media_swipe_in_each_direction_fires_its_action(tmp_path, start, stop, action, key):
    script = Script().hold("OPEN_PALM", 12, *start).move("OPEN_PALM", 8, start, stop)
    app = replay_app(script.save(tmp_path / "media.jsonl"), AppMode.MEDIA)
    assert app.run().actions == {action: 1}
    assert [event.key for event in app.media.events] == [key, key]


def test_media_volume_return_movement_does_not_fire_the_opposite_action():
    up, down = (0.5, 0.7), (0.5, 0.3)
    script = (
        Script()
        .hold("OPEN_PALM", 12, *up)
        .move("OPEN_PALM", 8, up, down)
        .move("OPEN_PALM", 8, down, up)
        .move("OPEN_PALM", 8, up, down)
        .move("OPEN_PALM", 8, down, up)
    )
    assert make_app(ListSource(script.ticks()), AppMode.MEDIA).run().actions == {"volume_up": 1}


def test_presentation_ignores_vertical_swipes():
    up, down = (0.5, 0.7), (0.5, 0.3)
    script = Script().hold("OPEN_PALM", 12, *up).move("OPEN_PALM", 8, up, down)
    app = make_app(ListSource(script.ticks()), AppMode.PRESENTATION)
    assert app.run().actions == {}
    assert app.presentation._post_event.events == []


def test_media_thumbs_up_toggles_play_pause_once_per_gesture(tmp_path):
    script = (
        Script().hold("POINTING", 5).hold("THUMBS_UP", 30).hold("POINTING", 10).hold("THUMBS_UP", 10)
    )
    app = replay_app(script.save(tmp_path / "play.jsonl"), AppMode.MEDIA)
    assert app.run().actions == {"play_pause": 2}
    assert {event.key for event in app.media.events} == {MediaKey.PLAY_PAUSE}


def test_media_thumbs_up_regained_after_a_dropout_does_not_fire_again():
    script = Script().hold("THUMBS_UP", 8).no_hand(8).hold("THUMBS_UP", 8)
    app = make_app(ListSource(script.ticks()), AppMode.MEDIA)
    assert app.run().actions == {"play_pause": 1}


# Mode switching


def test_switching_mode_mid_drag_releases_the_button():
    script = Script().hold("POINTING", 3).hold("PINCH", 10).hold("PINCH", 5)
    source = ListSource(script.ticks())
    app = make_app(source)
    held_after_switch = []

    def switch():
        assert MouseButton.LEFT in app.mouse.held_buttons
        app.set_mode(AppMode.MEDIA)
        held_after_switch.append(app.mouse.held_buttons)

    source.before[13] = switch
    app.run()
    assert held_after_switch == [frozenset()]
    assert app.fsm.state is GestureState.IDLE


def test_m_key_cycles_through_the_modes():
    window = FakeWindow([ord("m"), ord("m"), ord("m"), ord("m")])
    source = ListSource([tick_with_image() for _ in range(4)])
    app = make_app(source, window=window)
    seen = []
    original = app.set_mode

    def spy(mode):
        original(mode)
        seen.append(mode)

    app.set_mode = spy
    app.run()
    assert seen == [AppMode.PRESENTATION, AppMode.MEDIA, AppMode.FANTASY, AppMode.MOUSE]


# Three finger mode switching


def run_session(script, mode=AppMode.MOUSE):
    app = make_app(ListSource(script.ticks()), mode)
    app.run()
    return app


@pytest.mark.parametrize(
    "start, expected",
    [
        (AppMode.MOUSE, AppMode.PRESENTATION),
        (AppMode.PRESENTATION, AppMode.MEDIA),
        (AppMode.MEDIA, AppMode.FANTASY),
        (AppMode.FANTASY, AppMode.MOUSE),
    ],
)
def test_three_finger_hold_moves_to_the_next_mode(start, expected):
    assert run_session(Script().hold("THREE_FINGER", 25), start).mode is expected


def test_three_finger_hold_fires_at_half_a_second_not_before():
    # The vote needs 3 frames, then 16 frames of 33 ms are the first to reach 500 ms.
    assert run_session(Script().hold("THREE_FINGER", 18)).mode is AppMode.MOUSE
    assert run_session(Script().hold("THREE_FINGER", 19)).mode is AppMode.PRESENTATION


def test_long_three_finger_hold_is_a_single_switch():
    # Across the switch's reset of the mode's own vote, the latch must not let go.
    assert run_session(Script().hold("THREE_FINGER", 120)).mode is AppMode.PRESENTATION


def test_leaving_the_pose_allows_the_next_switch():
    script = Script()
    for _ in range(4):
        script.hold("THREE_FINGER", 25).hold("POINTING", 5)
    assert run_session(script).mode is AppMode.MOUSE  # The whole cycle, back to the start.


def test_leaving_the_pose_early_cancels_the_hold():
    script = Script().hold("THREE_FINGER", 12).hold("POINTING", 5).hold("THREE_FINGER", 12)
    assert run_session(script).mode is AppMode.MOUSE


def test_a_one_frame_glitch_does_not_cancel_the_hold():
    script = Script().hold("THREE_FINGER", 12).hand("FIST").hold("THREE_FINGER", 10)
    assert run_session(script).mode is AppMode.PRESENTATION


def test_hand_loss_cancels_the_hold():
    script = Script().hold("THREE_FINGER", 12).no_hand(8).hold("THREE_FINGER", 12)
    assert run_session(script).mode is AppMode.MOUSE


def test_tracking_dropout_during_a_latched_hold_does_not_switch_again():
    script = Script().hold("THREE_FINGER", 25).no_hand(8).hold("THREE_FINGER", 25)
    assert run_session(script).mode is AppMode.PRESENTATION


def test_hand_returning_after_a_real_absence_can_switch_again():
    script = Script().hold("THREE_FINGER", 25).no_hand(40).hold("THREE_FINGER", 25)
    assert run_session(script).mode is AppMode.MEDIA


def test_camera_stall_cancels_the_hold():
    script = Script().hold("THREE_FINGER", 12)
    script.t_ms += 300
    script.hold("THREE_FINGER", 12)
    assert run_session(script).mode is AppMode.MOUSE


def test_mode_switch_fires_no_action_and_presses_nothing():
    for mode in AppMode:
        app = run_session(Script().hold("THREE_FINGER", 25), mode)
        assert app.summary.actions == {}
        assert app.media.events == []
        assert app.presentation._post_event.events == []
        assert app.mouse.held_buttons == frozenset()


def test_new_mode_works_right_after_the_switch():
    start, stop = (0.3, 0.5), (0.7, 0.5)
    script = (
        Script()
        .hold("THREE_FINGER", 25, *start)
        .hold("OPEN_PALM", 12, *start)
        .move("OPEN_PALM", 8, start, stop)
    )
    app = run_session(script, AppMode.PRESENTATION)
    assert app.mode is AppMode.MEDIA
    assert app.summary.actions == {"next_track": 1}


def test_m_key_during_a_hold_restarts_it():
    script = Script().hold("THREE_FINGER", 15).hold("THREE_FINGER", 10)
    source = ListSource(script.ticks())
    app = make_app(source)
    source.before[15] = lambda: app.set_mode(AppMode.PRESENTATION)
    app.run()
    assert app.mode is AppMode.PRESENTATION


def test_mode_switch_replays_from_a_recording(tmp_path):
    path = Script().hold("THREE_FINGER", 25).save(tmp_path / "switch.jsonl")
    app = replay_app(path, AppMode.MEDIA)
    app.run()
    assert app.mode is AppMode.FANTASY


def test_hud_status_reports_mode_switch_progress():
    script = Script().hold("THREE_FINGER", 25)
    app = make_app(ListSource(script.ticks()), AppMode.PRESENTATION)
    progress = []
    for tick in script.ticks():
        app._apply_gestures(tick)
        status = app._hud_status(tick.timestamp)
        progress.append(status.mode_progress)
    # Nothing until the vote settles, then rising, then cleared by the switch.
    assert progress[:3] == [0.0, 0.0, 0.0]
    assert progress[3:18] == sorted(progress[3:18])
    assert 0.9 < progress[17] < 1.0
    assert progress[18:] == [0.0] * 7
    assert status.mode == "MEDIA" and status.next_mode == "FANTASY"


# Shutdown guarantees


def tick_with_image(t=1.0, result=None):
    return Tick(t, result or DetectionResult(int(t * 1000), 64, 48, ()), np.zeros((48, 64, 3), np.uint8))


@pytest.mark.parametrize("key", [ord("q"), KEY_ESC])
def test_quit_keys_stop_the_loop_and_release(key):
    script = Script().hold("POINTING", 3).hold("PINCH", 8)
    ticks = [
        Tick(tick.timestamp, tick.result, np.zeros((48, 64, 3), np.uint8)) for tick in script.ticks()
    ]
    window = FakeWindow([KEY_NONE] * 8 + [key])
    source = ListSource(ticks + ticks)  # Would raise on repeated timestamps if it kept going.
    app = make_app(source, window=window)
    app.run()
    assert len(window.shown) == 9
    assert left_click_was_released(app.mouse)
    assert window.closed == 1
    assert source.closed == 1


def left_click_was_released(mouse):
    types = event_types(mouse)
    return types.count(MouseEventType.LEFT_DOWN) == 1 and types[-1] is MouseEventType.LEFT_UP


def test_sigint_stops_the_loop_and_releases():
    script = Script().hold("POINTING", 3).hold("PINCH", 8).hold("PINCH", 20)
    source = ListSource(script.ticks(), before={11: lambda: signal.raise_signal(signal.SIGINT)})
    app = make_app(source)
    previous = signal.getsignal(signal.SIGINT)
    summary = app.run()  # Would raise KeyboardInterrupt if the handler were missing.
    assert summary.frames == 12
    assert left_click_was_released(app.mouse)
    assert source.closed == 1
    assert signal.getsignal(signal.SIGINT) is previous


def test_sigterm_stops_the_loop_too():
    source = ListSource(Script().hold("POINTING", 5).ticks(), before={2: lambda: signal.raise_signal(signal.SIGTERM)})
    assert make_app(source).run().frames == 3


def test_run_outside_the_main_thread_skips_signal_handlers():
    app = make_app(ListSource(Script().hold("POINTING", 3).ticks()))
    errors = []

    def target():
        try:
            app.run()
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(5)
    assert errors == []
    assert app.summary.frames == 3


def test_pipeline_error_propagates_after_releasing_everything():
    script = Script().hold("POINTING", 3).hold("PINCH", 10)

    def explode():
        raise RuntimeError("detector crashed")

    window = FakeWindow()
    source = ListSource(script.ticks(), before={12: explode})
    app = make_app(source, window=window)
    with pytest.raises(RuntimeError, match="detector crashed"):
        app.run()
    assert left_click_was_released(app.mouse)
    assert source.closed == 1
    assert window.closed == 1


def test_a_failing_shutdown_step_does_not_skip_the_others():
    script = Script().hold("POINTING", 3).hold("PINCH", 10)
    source = ListSource(script.ticks())
    app = make_app(source)
    window = FakeWindow()
    app.window = window

    def fail():
        raise OSError("disk gone")

    source.close = fail
    with pytest.raises(OSError, match="disk gone"):
        app.run()
    assert app.mouse.held_buttons == frozenset()
    assert window.closed == 1


def test_shutdown_is_idempotent():
    source = ListSource(Script().hold("POINTING", 3).ticks())
    app = make_app(source)
    app.run()
    app.shutdown()
    assert source.closed == 1


# Live source, with a fake camera and detector


class FakeCamera:
    def __init__(self, reads):
        self.reads = list(reads)
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def read(self, timeout=None):
        item = self.reads.pop(0) if self.reads else CameraError("capture ended")
        if callable(item):
            item = item()
        if isinstance(item, Exception):
            raise item
        return item


class FakeDetector:
    def __init__(self, results):
        self.results = iter(results)
        self.closed = False
        self.timestamps = []

    def detect(self, image, timestamp_ms=None):
        self.timestamps.append(timestamp_ms)
        return next(self.results)

    def close(self):
        self.closed = True


def camera_frames(results):
    image = np.zeros((48, 64, 3), np.uint8)
    return [Frame(image, i + 1, r.timestamp_ms / 1000) for i, r in enumerate(results)]


def test_camera_stall_releases_the_button_through_the_timeout_path():
    script = Script().hold("POINTING", 3).hold("PINCH", 8)
    last = script.results[-1].timestamp_ms / 1000
    times = iter([last + 0.1, last + 0.3])
    held_before_shutdown = []

    def capture_ends():
        held_before_shutdown.append(app.mouse.held_buttons)
        return CameraError("capture ended")

    camera = FakeCamera([*camera_frames(script.results), None, None, capture_ends])
    source = LiveSource(camera, FakeDetector(script.results), clock=lambda: next(times))
    app = make_app(source)
    with pytest.raises(CameraError):
        app.run()
    assert held_before_shutdown == [frozenset()]
    assert left_click_was_released(app.mouse)


def test_live_source_passes_capture_time_and_closes_everything():
    script = Script().hold("POINTING", 2)
    camera = FakeCamera(camera_frames(script.results))
    detector = FakeDetector(script.results)
    app = make_app(LiveSource(camera, detector))
    with pytest.raises(CameraError):
        app.run()
    assert detector.timestamps == [r.timestamp_ms for r in script.results]
    assert camera.started and camera.stopped and detector.closed
    stats = app.profiler.stats()
    assert stats["end_to_end"].count == 2


def test_no_frame_polls_the_window_for_keys():
    script = Script().hold("POINTING", 1)
    times = iter([2.0, 2.1])
    camera = FakeCamera([None, None])
    window = FakeWindow([KEY_NONE, ord("q")])
    source = LiveSource(camera, FakeDetector(script.results), clock=lambda: next(times))
    app = make_app(source, window=window)
    app.run()
    assert window.polled == 2
    assert window.shown == []


# HUD through the app


def test_window_receives_hud_frames_with_panel(tmp_path):
    path = Script().hold("POINTING", 3).save(tmp_path / "hud.jsonl")
    window = FakeWindow()
    app = replay_app(path, window=window, realtime=True)
    app.run()
    assert len(window.shown) == 3
    assert window.shown[0].shape == (480, 640, 3)
    assert window.shown[0].any()


@pytest.mark.parametrize(
    "state_frames, mode, expected",
    [
        (Script().hold("TWO_FINGER", 8), AppMode.MOUSE, ("SCROLL", "SCROLL_ACTIVE", "TWO_FINGER")),
        (Script().hold("POINTING", 8), AppMode.MOUSE, ("MOUSE", "TRACKING", "POINTING")),
        (Script().hold("OPEN_PALM", 8), AppMode.PRESENTATION, ("PRESENTATION", "ARMED", "OPEN_PALM")),
        (Script().hold("OPEN_PALM", 2), AppMode.MEDIA, ("MEDIA", "DISARMED", "UNKNOWN")),
        (Script().hold("FIST", 8), AppMode.FANTASY, ("FANTASY", "CHARGING", "FIST")),
    ],
)
def test_hud_status_names_mode_state_and_pose(state_frames, mode, expected):
    app = make_app(ListSource(state_frames.ticks()), mode)
    for tick in state_frames.ticks():
        app._apply_gestures(tick)
    status = app._hud_status(state_frames.ticks()[-1].timestamp)
    assert (status.mode, status.state, status.pose) == expected


def test_hud_status_reports_clutch():
    script = Script().hold("FIST", 8)
    app = make_app(ListSource(script.ticks()))
    for tick in script.ticks():
        app._apply_gestures(tick)
    status = app._hud_status(script.ticks()[-1].timestamp)
    assert status.clutch and not status.frozen


def test_hud_status_reports_pinch_freeze_right_after_the_press():
    script = Script().hold("POINTING", 3).hold("PINCH", 3)
    app = make_app(ListSource(script.ticks()))
    for tick in script.ticks():
        app._apply_gestures(tick)
        if app.fsm.state is GestureState.PINCH_ACTIVE:
            break
    status = app._hud_status(tick.timestamp)
    assert status.frozen and not status.clutch
    assert status.pinch.name == "LEFT_CLICK"


# Recording and the command line


def test_record_then_replay_reproduces_the_session(tmp_path):
    path = tmp_path / "records" / "session.jsonl"
    original = make_app(ListSource(mouse_session().ticks()), record_path=path)
    original.run()
    replayed = replay_app(path)
    replayed.run()
    assert original.summary.frames == replayed.summary.frames
    assert [e.type for e in original.mouse.events] == [e.type for e in replayed.mouse.events]
    assert original.mouse.scroll_events == replayed.mouse.scroll_events


def test_recording_refuses_to_replace_a_file_and_still_shuts_down(tmp_path):
    path = tmp_path / "taken.jsonl"
    path.write_text("precious")
    source = ListSource([])
    with pytest.raises(FileExistsError):
        make_app(source, record_path=path).run()
    assert source.closed == 1
    assert path.read_text() == "precious"


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert (args.headless, args.record, args.replay, args.mode, args.camera_id) == (
        False,
        None,
        None,
        "mouse",
        0,
    )


def test_parser_reads_every_option():
    args = build_parser().parse_args(
        ["--headless", "--record", "a.jsonl", "--mode", "media", "--camera-id", "2"]
    )
    assert (args.headless, args.record, args.mode, args.camera_id) == (True, "a.jsonl", "media", 2)


@pytest.mark.parametrize(
    "argv", [["--record", "a", "--replay", "b"], ["--mode", "scroll"], ["--camera-id", "x"]]
)
def test_parser_rejects_bad_arguments(argv):
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


def test_main_replays_headless_and_returns_zero(tmp_path):
    path = swipe_session().save(tmp_path / "cli.jsonl")
    assert main(["--replay", str(path), "--headless", "--mode", "presentation"]) == 0


def test_main_reports_a_missing_recording(tmp_path):
    assert main(["--replay", str(tmp_path / "missing.jsonl"), "--headless"]) == 1


def test_main_reports_a_corrupt_recording(tmp_path):
    path = tmp_path / "corrupt.jsonl"
    path.write_text('{"type": "nonsense"}\n')
    assert main(["--replay", str(path), "--headless"]) == 1


def test_module_entry_point_runs_end_to_end(tmp_path):
    path = mouse_session().save(tmp_path / "entry.jsonl")
    done = subprocess.run(
        [sys.executable, "-m", "src.main", "--replay", str(path), "--headless"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert "session finished" in done.stderr
    assert "PINCH_ACTIVE" in done.stderr
