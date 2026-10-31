"""Application entry point: wires capture, detection, gestures, drivers and the HUD together.

Capture runs on `CameraStream`'s thread and hands over only the newest frame. All
other work stays on the main thread, because macOS requires it for OpenCV windows.

    .venv/bin/python -m src.main --mode presentation
    .venv/bin/python -m src.main --replay records/session.jsonl --headless

Replay drives mock controllers, so a recording can never move the real cursor.
Modes are mutually exclusive, and switching one releases everything first.
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Protocol

import cv2
import numpy as np

from src.camera.stream import CameraError, CameraStream
from src.controller.base import BaseMouseController, MockMouseController
from src.controller.media import BaseMediaController, MockMediaController, QuartzMediaController
from src.controller.presentation import PresentationController
from src.controller.quartz_mouse import QuartzMouseController, check_accessibility_permission
from src.controller.scroll import ScrollController
from src.tracking.cursor_controller import CursorController
from src.tracking.state_machine import GestureState, GestureStateMachine
from src.ui.hud import HudStatus, HUDVisualizer, PinchFeedback
from src.utils.profiler import LatencyProfiler, Trace
from src.vision.classifier import Pose, PoseClassifier, TemporalVotingFilter
from src.vision.detector import DetectionResult, HandDetector
from src.vision.dynamic_gestures import SwipeDetector, SwipeDirection
from src.vision.recorder import LandmarkRecorder
from src.vision.replayer import LandmarkReplayer, ReplayFormatError

logger = logging.getLogger(__name__)

WINDOW_TITLE = "Hand Gesture Control"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
# Short enough that hand-loss timeouts and stop requests are served while the camera stalls.
CAMERA_READ_TIMEOUT = 0.05
HAND_LOSS_TIMEOUT = 0.2
POSE_ACTION_COOLDOWN = 1.0  # A held pose that tracking drops and regains must not fire twice.
MODE_SWITCH_HOLD = 0.5
KEY_ESC = 27
KEY_NONE = 255


class AppMode(Enum):
    MOUSE = "mouse"
    PRESENTATION = "presentation"
    MEDIA = "media"


class Action(NamedTuple):
    name: str
    run: Callable[[], None]


@dataclass(frozen=True)
class Tick:
    """One pass of the loop. `result` is None if the camera produced nothing in time."""

    timestamp: float  # Seconds, on the same clock as `result.timestamp_ms`.
    result: DetectionResult | None = None
    image: np.ndarray | None = None  # BGR canvas for the HUD, if the source has one.
    trace: Trace | None = None


class FrameSource(Protocol):
    def open(self) -> None: ...

    def frames(self, profiler: LatencyProfiler) -> Iterator[Tick]: ...

    def close(self) -> None: ...


class Window(Protocol):
    def show(self, image: np.ndarray) -> int:
        """Display a frame and return the pressed key, or `KEY_NONE`."""

    def poll(self) -> int:
        """Return the pressed key, or `KEY_NONE`, while there is no new frame."""

    def close(self) -> None: ...


class OpenCvWindow:
    def __init__(self, title: str = WINDOW_TITLE) -> None:
        self._title = title
        self._shown = False

    def show(self, image: np.ndarray) -> int:
        cv2.imshow(self._title, image)
        self._shown = True
        return self.poll()

    def poll(self) -> int:
        key = cv2.waitKey(1) & 0xFF
        # Closing the window with its button means quit; imshow would just reopen it.
        if self._shown and cv2.getWindowProperty(self._title, cv2.WND_PROP_VISIBLE) < 1:
            return ord("q")
        return key

    def close(self) -> None:
        if self._shown:
            cv2.destroyWindow(self._title)
            cv2.waitKey(1)  # Windows only disappear once the event loop has run.
            self._shown = False


class LiveSource:
    """Runs the detector on the newest camera frame."""

    def __init__(
        self,
        camera: CameraStream,
        detector: HandDetector,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._camera = camera
        self._detector = detector
        self._clock = clock

    def open(self) -> None:
        self._camera.start()

    def frames(self, profiler: LatencyProfiler) -> Iterator[Tick]:
        while True:
            frame = self._camera.read(timeout=CAMERA_READ_TIMEOUT)
            if frame is None:
                yield Tick(self._clock())
                continue
            trace = profiler.begin(frame.timestamp)
            result = self._detector.detect(frame.image, int(frame.timestamp * 1000))
            trace.mark("detection")
            yield Tick(result.timestamp_ms / 1000, result, frame.image, trace)

    def close(self) -> None:
        try:
            self._camera.stop()
        finally:
            self._detector.close()


class ReplaySource:
    """Plays a recording. As fast as possible, or at recorded speed with `realtime`."""

    def __init__(self, path: str | Path, realtime: bool = False) -> None:
        self._replayer = LandmarkReplayer(path)
        self._realtime = realtime
        self.now = 0.0  # Recorded time, so that replay is a pure function of the file.

    def open(self) -> None:
        pass

    def frames(self, profiler: LatencyProfiler) -> Iterator[Tick]:
        previous = None
        for result in self._replayer:
            self.now = result.timestamp_ms / 1000
            image = None
            if self._realtime:
                if previous is not None:
                    time.sleep(self.now - previous)
                image = np.zeros((result.image_height, result.image_width, 3), np.uint8)
            previous = self.now
            trace = profiler.begin()
            trace.mark("detection")
            yield Tick(self.now, result, image, trace)

    def close(self) -> None:
        pass


class SwipeActionMode:
    """Fires actions from swipes and held poses, for the modes that do not use the cursor."""

    def __init__(
        self,
        classifier: PoseClassifier,
        swipes: Mapping[SwipeDirection, Action],
        poses: Mapping[Pose, Action] | None = None,
    ) -> None:
        self._classifier = classifier
        self._swipes = swipes
        self._poses = poses or {}
        self._voting = TemporalVotingFilter()
        self._detector = SwipeDetector()
        self._last_hand_time: float | None = None
        self._last_pose = Pose.UNKNOWN
        self._last_pose_action = -POSE_ACTION_COOLDOWN

    @property
    def pose(self) -> Pose:
        return self._voting.pose

    @property
    def state_name(self) -> str:
        return self._detector.phase.name

    def reset(self) -> None:
        self._voting.reset()
        self._detector.reset()
        self._last_hand_time = None
        self._last_pose = Pose.UNKNOWN

    def update(self, result: DetectionResult | None, timestamp: float) -> str | None:
        """Advance one tick and return the name of the action it fired, if any."""
        # Checked on hand frames too: a stalled camera shows up as a long gap between them.
        if self._last_hand_time is not None and timestamp - self._last_hand_time >= HAND_LOSS_TIMEOUT:
            self.reset()
        if result is None or not result.hands:
            return None

        self._last_hand_time = timestamp
        hand = result.hands[0]
        pose = self._voting.update(self._classifier.classify(hand).pose)
        swipe = self._detector.update(pose, hand.landmarks, timestamp, result.aspect_ratio)
        action = self._swipes.get(swipe) if swipe is not None else None
        # Fires when the pose starts, not for as long as it is held.
        if (
            action is None
            and pose is not self._last_pose
            and pose in self._poses
            and timestamp - self._last_pose_action >= POSE_ACTION_COOLDOWN
        ):
            action = self._poses[pose]
            self._last_pose_action = timestamp
        self._last_pose = pose
        if action is None:
            return None
        action.run()
        return action.name


class ModeSwitchGesture:
    """Reports one mode switch each time THREE_FINGER is held for `MODE_SWITCH_HOLD`.

    Votes on its own, because switching resets the active mode's filter and the
    UNKNOWN that follows would read as the hand having left the pose.
    """

    def __init__(self, classifier: PoseClassifier) -> None:
        self._classifier = classifier
        self._voting = TemporalVotingFilter()
        self._last_hand_time: float | None = None
        self._held: float | None = None  # Seconds in the pose; None while not in it.
        self._latched = False

    @property
    def progress(self) -> float:
        """How far the current hold is toward a switch, from 0 to 1."""
        return 0.0 if self._held is None else min(self._held / MODE_SWITCH_HOLD, 1.0)

    def restart(self) -> None:
        """Drop the hold's progress. The latch stays, so the same hold cannot fire again."""
        self._held = None

    def update(self, result: DetectionResult | None, timestamp: float) -> bool:
        """Advance one tick and return whether the hold just completed."""
        elapsed = 0.0 if self._last_hand_time is None else timestamp - self._last_hand_time
        if elapsed >= HAND_LOSS_TIMEOUT:
            self._held = None
        # The latch outlives a tracking dropout, but a hand gone this long has really left.
        if elapsed >= POSE_ACTION_COOLDOWN:
            self._voting.reset()
            self._last_hand_time = None
            self._latched = False
        if result is None or not result.hands:
            return False

        self._last_hand_time = timestamp
        pose = self._voting.update(self._classifier.classify(result.hands[0]).pose)
        if pose is not Pose.THREE_FINGER:
            self._held = None
            self._latched = False
            return False
        if self._latched:
            return False
        self._held = 0.0 if self._held is None else self._held + elapsed
        if self._held < MODE_SWITCH_HOLD:
            return False
        self._held = None
        self._latched = True
        return True


@dataclass
class SessionSummary:
    frames: int = 0
    hand_frames: int = 0
    states_seen: set[str] = field(default_factory=set)
    actions: Counter[str] = field(default_factory=Counter)


class GestureControlApp:
    """Owns the main loop. `run` always releases everything, however the loop ends."""

    def __init__(
        self,
        source: FrameSource,
        mouse: BaseMouseController,
        media: BaseMediaController,
        presentation: PresentationController,
        mode: AppMode = AppMode.MOUSE,
        window: Window | None = None,
        record_path: str | Path | None = None,
    ) -> None:
        self.source = source
        self.mouse = mouse
        self.media = media
        self.presentation = presentation
        self.mode = mode
        self.window = window
        self.summary = SessionSummary()
        self.profiler = LatencyProfiler()
        self.hud = HUDVisualizer()
        self.fsm = GestureStateMachine(
            mouse,
            CursorController(mouse.screen_bounds(), mouse.position),
            scroll=ScrollController(mouse),
        )
        self._swipe_modes = {
            AppMode.PRESENTATION: SwipeActionMode(
                self.fsm.classifier,
                {
                    SwipeDirection.RIGHT: Action("next_slide", presentation.next_slide),
                    SwipeDirection.LEFT: Action("previous_slide", presentation.previous_slide),
                },
            ),
            AppMode.MEDIA: SwipeActionMode(
                self.fsm.classifier,
                {
                    SwipeDirection.RIGHT: Action("next_track", media.next_track),
                    SwipeDirection.LEFT: Action("prev_track", media.prev_track),
                    SwipeDirection.UP: Action("volume_up", media.volume_up),
                    SwipeDirection.DOWN: Action("volume_down", media.volume_down),
                },
                {Pose.THUMBS_UP: Action("play_pause", media.play_pause)},
            ),
        }
        self._mode_switch = ModeSwitchGesture(self.fsm.classifier)
        self._record_path = record_path
        self._recorder: LandmarkRecorder | None = None
        self._stop = threading.Event()
        self._last_action: str | None = None
        self._is_shut_down = False

    def request_stop(self) -> None:
        """Ask the loop to finish. Safe from signal handlers and other threads."""
        self._stop.set()

    def set_mode(self, mode: AppMode) -> None:
        """Switch modes. Nothing held or half-recognized survives the change."""
        if mode is self.mode:
            return
        self._reset_gestures()
        self._mode_switch.restart()
        self.mode = mode
        logger.info("mode: %s", mode.value)

    def run(self) -> SessionSummary:
        """Run until the source ends or a stop is requested, then shut down."""
        restore_signals = self._install_signal_handlers()
        try:
            self.source.open()
            # Opened after the camera, so a failed start leaves no empty recording behind.
            if self._record_path is not None:
                self._recorder = LandmarkRecorder(
                    self._record_path, metadata={"mode": self.mode.value}
                )
            for tick in self.source.frames(self.profiler):
                self._step(tick)
                if self._stop.is_set():
                    break
        finally:
            # Handlers stay in place until the end, so Ctrl+C cannot interrupt the release.
            try:
                self.shutdown()
            finally:
                restore_signals()
        return self.summary

    def shutdown(self) -> None:
        """Release buttons and close every resource. Each step runs even if another fails."""
        if self._is_shut_down:
            return
        self._is_shut_down = True
        steps: list[Callable[[], None]] = [self._reset_gestures, self.mouse.ensure_all_released]
        if self._recorder is not None:
            steps.append(self._recorder.close)
        steps.append(self.source.close)
        if self.window is not None:
            steps.append(self.window.close)

        first_error: Exception | None = None
        for step in steps:
            try:
                step()
            except Exception as error:
                logger.exception("shutdown step failed")
                first_error = first_error or error
        if first_error is not None:
            raise first_error

    def _install_signal_handlers(self) -> Callable[[], None]:
        # A flag rather than KeyboardInterrupt, so the signal cannot land mid-event
        # and the loop always gets to release in order.
        if threading.current_thread() is not threading.main_thread():
            return lambda: None

        def handle(signum: int, _frame: object) -> None:
            logger.info("received %s, shutting down", signal.Signals(signum).name)
            self.request_stop()

        previous = {sig: signal.signal(sig, handle) for sig in (signal.SIGINT, signal.SIGTERM)}

        def restore() -> None:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

        return restore

    def _step(self, tick: Tick) -> None:
        if tick.result is not None:
            self.summary.frames += 1
            self.summary.hand_frames += bool(tick.result.hands)
            # Written first, so the frame that crashes the pipeline can be replayed.
            if self._recorder is not None:
                self._recorder.record(tick.result)

        self._apply_gestures(tick)
        if tick.trace is not None:
            tick.trace.mark("action")
            tick.trace.finish()

        self.summary.states_seen.add(self._state_name())
        if self.window is not None:
            self._handle_key(self._render(tick))

    def _apply_gestures(self, tick: Tick) -> None:
        if self._mode_switch.update(tick.result, tick.timestamp):
            self.set_mode(self._next_mode())
        swipe_mode = self._swipe_modes.get(self.mode)
        if swipe_mode is not None:
            action = swipe_mode.update(tick.result, tick.timestamp)
            if action is not None:
                self._last_action = action
                self.summary.actions[action] += 1
                logger.info("action: %s", action)
        elif tick.result is not None:
            self.fsm.process(tick.result)
        else:
            self.fsm.update(None, tick.timestamp)

    def _render(self, tick: Tick) -> int:
        if tick.image is None:
            return self.window.poll()
        return self.window.show(
            self.hud.draw(tick.image, tick.result, self._hud_status(tick.timestamp))
        )

    def _handle_key(self, key: int) -> None:
        if key in (ord("q"), KEY_ESC):
            self.request_stop()
        elif key == ord("m"):
            self.set_mode(self._next_mode())

    def _next_mode(self) -> AppMode:
        modes = list(AppMode)
        return modes[(modes.index(self.mode) + 1) % len(modes)]

    def _reset_gestures(self) -> None:
        for swipe_mode in self._swipe_modes.values():
            swipe_mode.reset()
        self.fsm.reset()

    def _state_name(self) -> str:
        swipe_mode = self._swipe_modes.get(self.mode)
        return swipe_mode.state_name if swipe_mode is not None else self.fsm.state.name

    def _hud_status(self, timestamp: float) -> HudStatus:
        latency = self.profiler.stats().get(LatencyProfiler.END_TO_END)
        swipe_mode = self._swipe_modes.get(self.mode)
        if swipe_mode is not None:
            mode_label, pose = self.mode.name, swipe_mode.pose
            clutch = frozen = False
            pinch = PinchFeedback.NONE
        else:
            is_scrolling = self.fsm.state is GestureState.SCROLL_ACTIVE
            mode_label, pose = "SCROLL" if is_scrolling else "MOUSE", self.fsm.voting.pose
            clutch = self.fsm.cursor.clutch
            frozen = not clutch and self.fsm.cursor.is_frozen(timestamp)
            pinch = self._pinch_feedback()
        return HudStatus(
            mode=mode_label,
            state=self._state_name(),
            pose=pose.name,
            fps=self.profiler.fps(),
            p50_ms=latency.p50_ms if latency is not None else None,
            p95_ms=latency.p95_ms if latency is not None else None,
            clutch=clutch,
            frozen=frozen,
            pinch=pinch,
            last_action=self._last_action,
            mode_progress=self._mode_switch.progress,
            next_mode=self._next_mode().name,
        )

    def _pinch_feedback(self) -> PinchFeedback:
        if self.fsm.state is GestureState.RIGHT_CLICK_ACTIVE:
            return PinchFeedback.RIGHT_CLICK
        if self.fsm.state is GestureState.PINCH_ACTIVE:
            return PinchFeedback.DRAG if self.fsm.is_dragging else PinchFeedback.LEFT_CLICK
        return PinchFeedback.NONE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="src.main",
        description="Control the Mac with hand gestures. In the window: q/ESC quits, m switches mode.",
    )
    parser.add_argument("--headless", action="store_true", help="run without an OpenCV window")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--record", metavar="FILE.jsonl", help="save the session's landmarks")
    source.add_argument(
        "--replay",
        metavar="FILE.jsonl",
        help="play a recording without a camera, against mock controllers",
    )
    parser.add_argument(
        "--mode", choices=[mode.value for mode in AppMode], default=AppMode.MOUSE.value
    )
    parser.add_argument("--camera-id", type=int, default=0, metavar="INT", help="camera index")
    return parser


def build_app(args: argparse.Namespace) -> GestureControlApp:
    source: FrameSource
    if args.replay is not None:
        source = replay = ReplaySource(args.replay, realtime=not args.headless)
        mouse: BaseMouseController = MockMouseController(clock=lambda: replay.now)
        media: BaseMediaController = MockMediaController()
        presentation = PresentationController(post_event=lambda _event: None)
    else:
        if not check_accessibility_permission(prompt=True):
            logger.warning("no Accessibility permission: macOS will drop all input events")
        source = LiveSource(
            CameraStream(args.camera_id, width=CAMERA_WIDTH, height=CAMERA_HEIGHT), HandDetector()
        )
        mouse = QuartzMouseController()
        media = QuartzMediaController()
        presentation = PresentationController()
    return GestureControlApp(
        source,
        mouse,
        media,
        presentation,
        mode=AppMode(args.mode),
        window=None if args.headless else OpenCvWindow(),
        record_path=args.record,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        summary = build_app(args).run()
    except (CameraError, ReplayFormatError, OSError) as error:
        logger.error("%s", error)
        return 1
    logger.info(
        "session finished: %d frames, %d with a hand, states %s, actions %s",
        summary.frames,
        summary.hand_frames,
        sorted(summary.states_seen),
        dict(summary.actions),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
