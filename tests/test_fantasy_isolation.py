"""FANTASY mode must never reach the OS: no events out, and no route to the event APIs."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hand_factory import POSES, make_hand
from src.controller.base import MockMouseController, MouseButton, Point
from src.controller.media import MockMediaController, QuartzMediaController
from src.controller.quartz_mouse import QuartzMouseController
from src.controller.swap import SwapController
from src.fantasy.mode import FantasyMode
from src.main import AppMode, GestureControlApp, ReplaySource, Tick, build_parser
from src.tracking.state_machine import GestureState
from src.ui.hud import MODE_COLORS, HudStatus, HUDVisualizer
from src.vision.detector import DetectionResult
from src.vision.recorder import LandmarkRecorder

ROOT = Path(__file__).resolve().parents[1]
FANTASY_DIR = ROOT / "src" / "fantasy"
OS_MODULES = ("AppKit", "Quartz", "objc", "Foundation", "CoreFoundation")
WIDTH, HEIGHT = 640, 480


def session():
    """Every pose, with swipes in all directions, and hand loss: what would fire elsewhere."""
    results, t = [], 1000

    def add(pose, frames, start=(0.3, 0.5), stop=None):
        nonlocal t
        stop = stop or start
        for i in range(frames):
            f = i / max(frames - 1, 1)
            x, y = start[0] + (stop[0] - start[0]) * f, start[1] + (stop[1] - start[1]) * f
            hand = make_hand(POSES[pose], x=x, y=y) if pose else None
            results.append(DetectionResult(t, WIDTH, HEIGHT, (hand,) if hand else ()))
            t += 33

    add("POINTING", 15, (0.3, 0.5), (0.5, 0.6))
    add("PINCH", 20, (0.5, 0.6), (0.3, 0.5))
    add("POINTING", 10)
    add("MIDDLE_PINCH", 15)
    add("TWO_FINGER", 20, (0.3, 0.5), (0.3, 0.3))
    add("FIST", 25)
    add("POINTING", 40)
    add("THUMBS_UP", 20)
    for start, stop in (
        ((0.3, 0.5), (0.7, 0.5)),
        ((0.7, 0.5), (0.3, 0.5)),
        ((0.5, 0.7), (0.5, 0.3)),
        ((0.5, 0.3), (0.5, 0.7)),
    ):
        add("OPEN_PALM", 15, start)
        add("OPEN_PALM", 8, start, stop)
        add("OPEN_PALM", 35, stop)
    add(None, 15)
    add("FIST", 25)
    add(None, 45)
    return results


def save_session(path):
    with LandmarkRecorder(path) as recorder:
        for result in session():
            recorder.record(result)
    return path


class ListSource:
    def __init__(self, ticks):
        self.ticks = ticks

    def open(self):
        pass

    def frames(self, profiler):
        yield from self.ticks

    def close(self):
        pass


class Window:
    def __init__(self):
        self.shown = 0

    def show(self, image):
        self.shown += 1
        return 255

    def poll(self):
        return 255

    def close(self):
        pass


def session_ticks(with_image=False):
    return [
        Tick(
            result.timestamp_ms / 1000,
            result,
            np.zeros((HEIGHT, WIDTH, 3), np.uint8) if with_image else None,
        )
        for result in session()
    ]


def mock_app(source, mode=AppMode.FANTASY, window=None):
    posted = []
    app = GestureControlApp(
        source,
        MockMouseController(start=Point(960, 540)),
        MockMediaController(),
        SwapController(post_event=posted.append),
        mode=mode,
        window=window,
    )
    return app, posted


def assert_silent(app, posted):
    assert app.mouse.events == []
    assert app.mouse.scroll_events == []
    assert app.media.events == []
    assert posted == []


# Nothing leaves the process


def test_replay_in_fantasy_mode_sends_nothing(tmp_path):
    app, posted = mock_app(ReplaySource(save_session(tmp_path / "all.jsonl")))
    summary = app.run()
    assert summary.frames == len(session())
    assert {"CHARGING", "ACTIVE", "COOLDOWN", "IDLE"} <= summary.states_seen
    assert set(summary.actions) == {"chidori", "rasengan"}
    assert_silent(app, posted)


def test_stream_with_rendering_in_fantasy_mode_sends_nothing():
    window = Window()
    app, posted = mock_app(ListSource(session_ticks(with_image=True)), window=window)
    app.run()
    assert window.shown == len(session())
    assert_silent(app, posted)


def test_real_drivers_post_nothing_in_fantasy_mode():
    # The real classes, so that a path around the mocks' `events` would still be caught.
    posted = []
    app = GestureControlApp(
        ListSource(session_ticks()),
        QuartzMouseController(post_event=posted.append),
        QuartzMediaController(post_event=posted.append),
        SwapController(post_event=posted.append),
        mode=AppMode.FANTASY,
    )
    app.run()
    assert posted == []


def test_mouse_state_machine_does_not_run_in_fantasy_mode():
    app, _ = mock_app(ListSource(session_ticks()))

    def fail(*_args):
        raise AssertionError("the mouse state machine ran in FANTASY mode")

    app.fsm.process = fail
    app.fsm.update = fail
    app.run()
    assert app.fsm.state is GestureState.IDLE
    assert GestureState.TRACKING.name not in app.summary.states_seen


def test_ticks_without_a_detection_stay_out_of_the_state_machine():
    app, posted = mock_app(ListSource([Tick(1.0 + i * 0.05) for i in range(20)]))
    app.fsm.update = lambda *_args: pytest.fail("the mouse state machine ran")
    app.run()
    assert_silent(app, posted)


def test_entering_fantasy_mid_drag_releases_and_then_stays_silent():
    results = session()
    hand = make_hand(POSES["PINCH"], x=0.3, y=0.5)
    pointing = make_hand(POSES["POINTING"], x=0.3, y=0.5)
    ticks = []
    t = 1000
    for current in [pointing] * 3 + [hand] * 10:
        ticks.append(Tick(t / 1000, DetectionResult(t, WIDTH, HEIGHT, (current,))))
        t += 33
    app, posted = mock_app(ListSource([]), AppMode.MOUSE)
    for tick in ticks:
        app._apply_gestures(tick)
    assert MouseButton.LEFT in app.mouse.held_buttons

    app.set_mode(AppMode.FANTASY)
    assert app.mouse.held_buttons == frozenset()
    count = len(app.mouse.events)
    for result in results:
        app._apply_gestures(Tick((result.timestamp_ms + t) / 1000, result))
    assert len(app.mouse.events) == count
    assert app.mouse.scroll_events == [] and app.media.events == [] and posted == []


def test_mouse_works_again_after_leaving_fantasy():
    ticks = []
    t = 1000
    for i in range(15):
        hand = make_hand(POSES["POINTING"], x=0.3 + i * 0.01, y=0.5)
        ticks.append(Tick(t / 1000, DetectionResult(t, WIDTH, HEIGHT, (hand,))))
        t += 33
    app, _ = mock_app(ListSource([]))
    app.set_mode(AppMode.MOUSE)
    for tick in ticks:
        app._apply_gestures(tick)
    assert app.mouse.events


# AppKit and Quartz stay out of the process


def run_python(code, *args):
    done = subprocess.run(
        [sys.executable, "-c", code, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


LOADED = f"[name for name in {OS_MODULES!r} if name in sys.modules]"


def test_importing_the_fantasy_package_loads_no_os_module():
    code = (
        "import sys, json\n"
        "import src.fantasy.engine, src.fantasy.mode\n"
        "import src.fantasy.effects.chidori, src.fantasy.effects.rasengan\n"
        f"print(json.dumps({LOADED}))"
    )
    assert run_python(code) == []


def test_fantasy_replay_through_the_cli_never_loads_appkit(tmp_path):
    path = save_session(tmp_path / "all.jsonl")
    code = (
        "import sys, json\n"
        "from src.main import main\n"
        "code = main(['--replay', sys.argv[1], '--mode', 'fantasy', '--headless'])\n"
        "assert code == 0\n"
        f"print(json.dumps({LOADED}))"
    )
    assert run_python(code, str(path)) == []


def test_the_isolation_check_can_detect_appkit():
    # Guards the two tests above against passing because the check itself is blind.
    code = f"import sys, json\nimport src.controller.quartz_mouse\nprint(json.dumps({LOADED}))"
    assert "AppKit" in run_python(code)


@pytest.mark.parametrize(
    "path", sorted(FANTASY_DIR.rglob("*.py")), ids=lambda path: path.name
)
def test_fantasy_sources_import_nothing_that_reaches_the_os(path):
    forbidden = set(OS_MODULES) | {"src.controller", "src.main", "src.tracking", "pynput", "pyautogui"}
    names = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    assert not [
        name for name in names if any(name == bad or name.startswith(bad + ".") for bad in forbidden)
    ]


# Wiring


def test_cli_accepts_the_fantasy_mode():
    assert build_parser().parse_args(["--mode", "fantasy"]).mode == "fantasy"


def test_cycle_order():
    assert list(AppMode) == [AppMode.MOUSE, AppMode.SWAP, AppMode.MEDIA, AppMode.FANTASY]


def test_fantasy_is_routed_like_the_modes_that_bypass_the_mouse():
    app, _ = mock_app(ListSource([]))
    assert isinstance(app._swipe_modes[AppMode.FANTASY], FantasyMode)


def test_switching_away_clears_the_effect_and_its_cooldown():
    app, _ = mock_app(ListSource([]))
    t = 1000
    for _ in range(15):
        hand = make_hand(POSES["FIST"], x=0.3, y=0.5)
        app._apply_gestures(Tick(t / 1000, DetectionResult(t, WIDTH, HEIGHT, (hand,))))
        t += 33
    assert app._state_name() == "ACTIVE"
    app.set_mode(AppMode.MOUSE)
    app.set_mode(AppMode.FANTASY)
    assert app._state_name() == "IDLE"


def test_hud_has_a_chakra_blue_fantasy_label():
    assert MODE_COLORS["FANTASY"] == (0xFF, 0xE5, 0x00)  # BGR for #00E5FF.
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    HUDVisualizer().draw(image, None, HudStatus(mode="FANTASY", state="IDLE", pose="UNKNOWN"))
    assert np.any(np.all(image == np.array(MODE_COLORS["FANTASY"], np.uint8), axis=2))


def test_app_status_reports_fantasy():
    app, _ = mock_app(ListSource([]))
    status = app._hud_status(1.0)
    assert (status.mode, status.state, status.next_mode) == ("FANTASY", "IDLE", "MOUSE")


def test_effects_are_drawn_under_the_panel_and_over_the_hand():
    calls = []
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    status = HudStatus(mode="FANTASY", state="ACTIVE", pose="FIST")
    hud = HUDVisualizer()
    original = hud._draw_panel
    hud._draw_panel = lambda *args: (calls.append("panel"), original(*args))
    hud.draw(image, None, status, lambda _image: calls.append("effects"))
    assert calls == ["effects", "panel"]
