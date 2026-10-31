import pytest
import Quartz

from src.controller.base import (
    MockMouseController,
    MouseButton,
    MouseEvent,
    MouseEventType,
    Point,
    ScreenBounds,
)
from src.controller.quartz_mouse import QuartzMouseController, check_accessibility_permission


class FakeClock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def mouse(clock):
    return MockMouseController(start=Point(100, 100), clock=clock)


def types(mouse):
    return [event.type for event in mouse.events]


def test_move_to_records_position(mouse):
    mouse.move_to(300, 200)
    assert mouse.events == [MouseEvent(MouseEventType.MOVED, 300, 200, 0)]
    assert mouse.position == Point(300, 200)


@pytest.mark.parametrize(
    "target, expected",
    [
        ((-50, 500), (0, 500)),
        ((5000, 500), (1919, 500)),
        ((500, -1), (500, 0)),
        ((500, 9999), (500, 1079)),
    ],
)
def test_move_to_clamps_to_screen(mouse, target, expected):
    mouse.move_to(*target)
    assert mouse.position == expected


def test_bounds_with_non_zero_origin_clamp_to_that_display():
    bounds = ScreenBounds(-1440, -200, 1440, 900)
    assert bounds.clamp(10, -500) == (-1, -200)
    assert bounds.clamp(-5000, 5000) == (-1440, 699)
    assert bounds.center == (-720, 250)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_move_to_rejects_non_finite_without_posting(mouse, bad):
    with pytest.raises(ValueError):
        mouse.move_to(bad, 10)
    assert mouse.events == []
    assert mouse.position == Point(100, 100)


def test_invalid_bounds_rejected():
    with pytest.raises(ValueError):
        ScreenBounds(0, 0, 0, 100)


def test_click_left_and_right_send_down_up_at_cursor(mouse):
    mouse.click_left()
    mouse.click_right()
    assert mouse.events == [
        MouseEvent(MouseEventType.LEFT_DOWN, 100, 100, 1),
        MouseEvent(MouseEventType.LEFT_UP, 100, 100, 1),
        MouseEvent(MouseEventType.RIGHT_DOWN, 100, 100, 1),
        MouseEvent(MouseEventType.RIGHT_UP, 100, 100, 1),
    ]
    assert mouse.held_buttons == frozenset()


def test_press_and_release_are_idempotent(mouse):
    mouse.press_left()
    mouse.press_left()
    assert types(mouse) == [MouseEventType.LEFT_DOWN]
    mouse.release_left()
    mouse.release_left()
    mouse.release_right()
    assert types(mouse) == [MouseEventType.LEFT_DOWN, MouseEventType.LEFT_UP]


def test_drag_sends_dragged_events_and_releases_at_destination(mouse):
    mouse.press_left()
    mouse.drag_to(150, 160)
    mouse.release_left()
    assert mouse.events == [
        MouseEvent(MouseEventType.LEFT_DOWN, 100, 100, 1),
        MouseEvent(MouseEventType.LEFT_DRAGGED, 150, 160, 1),
        MouseEvent(MouseEventType.LEFT_UP, 150, 160, 1),
    ]


def test_drag_to_is_clamped(mouse):
    mouse.press_left()
    mouse.drag_to(99999, -5)
    assert mouse.position == (1919, 0)


def test_drag_to_without_press_raises_and_posts_nothing(mouse):
    with pytest.raises(RuntimeError):
        mouse.drag_to(10, 10)
    assert mouse.events == []


@pytest.mark.parametrize(
    "press, expected",
    [
        ("press_left", MouseEventType.LEFT_DRAGGED),
        ("press_right", MouseEventType.RIGHT_DRAGGED),
    ],
)
def test_move_while_held_becomes_drag(mouse, press, expected):
    getattr(mouse, press)()
    mouse.move_to(120, 120)
    assert mouse.events[-1].type is expected


def test_double_click_increments_click_state(mouse):
    mouse.double_click_left()
    assert [(e.type, e.click_state) for e in mouse.events] == [
        (MouseEventType.LEFT_DOWN, 1),
        (MouseEventType.LEFT_UP, 1),
        (MouseEventType.LEFT_DOWN, 2),
        (MouseEventType.LEFT_UP, 2),
    ]


def test_triple_click_reaches_click_state_three(mouse):
    for _ in range(3):
        mouse.click_left()
    assert mouse.events[-1].click_state == 3


def test_click_state_resets_after_interval(mouse, clock):
    mouse.click_left()
    clock.now += 0.51
    mouse.click_left()
    assert mouse.events[-1].click_state == 1


def test_click_state_counts_at_exact_interval(mouse, clock):
    mouse.click_left()
    clock.now += 0.5
    mouse.click_left()
    assert mouse.events[-1].click_state == 2


def test_click_state_resets_when_cursor_moved(mouse):
    mouse.click_left()
    mouse.move_to(150, 100)
    mouse.click_left()
    assert mouse.events[-1].click_state == 1


def test_click_state_survives_tiny_cursor_motion(mouse):
    mouse.click_left()
    mouse.move_to(102, 101)
    mouse.click_left()
    assert mouse.events[-1].click_state == 2


def test_click_state_resets_on_other_button(mouse):
    mouse.click_left()
    mouse.click_right()
    assert mouse.events[-1].click_state == 1


def test_ensure_all_released_releases_each_held_button(mouse):
    mouse.press_left()
    mouse.press_right()
    mouse.ensure_all_released()
    assert types(mouse)[-2:] == [MouseEventType.LEFT_UP, MouseEventType.RIGHT_UP]
    assert mouse.held_buttons == frozenset()


def test_ensure_all_released_is_noop_when_nothing_held(mouse):
    mouse.ensure_all_released()
    assert mouse.events == []


def test_ensure_all_released_attempts_all_buttons_when_one_fails(clock):
    class FailsOnLeftUp(MockMouseController):
        def _post(self, event):
            if event.type is MouseEventType.LEFT_UP:
                raise OSError("post failed")
            super()._post(event)

    mouse = FailsOnLeftUp(clock=clock)
    mouse.press_left()
    mouse.press_right()
    with pytest.raises(OSError):
        mouse.ensure_all_released()
    assert mouse.events[-1].type is MouseEventType.RIGHT_UP
    assert mouse.held_buttons == {MouseButton.LEFT}  # Still held, so a retry can release it.


def test_context_manager_releases_on_exception(mouse):
    with pytest.raises(KeyError):
        with mouse:
            mouse.press_left()
            raise KeyError("boom")
    assert types(mouse)[-1] is MouseEventType.LEFT_UP


def test_sync_position_adopts_physical_mouse(mouse):
    mouse.os_position = Point(700, 800)
    assert mouse.position == Point(100, 100)
    assert mouse.sync_position() == Point(700, 800)
    mouse.click_left()
    assert (mouse.events[0].x, mouse.events[0].y) == (700, 800)


# QuartzMouseController builds real CGEvents, but the injected sink captures them so
# the tests never move the developer's cursor.


@pytest.fixture
def posted():
    return []


@pytest.fixture
def quartz_mouse(posted, clock):
    return QuartzMouseController(clock=clock, post_event=posted.append)


def decode(event):
    location = Quartz.CGEventGetLocation(event)
    return (
        Quartz.CGEventGetType(event),
        location.x,
        location.y,
        Quartz.CGEventGetIntegerValueField(event, Quartz.kCGMouseEventClickState),
        Quartz.CGEventGetIntegerValueField(event, Quartz.kCGMouseEventButtonNumber),
    )


def test_quartz_bounds_match_main_display(quartz_mouse):
    rect = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    assert quartz_mouse.screen_bounds() == ScreenBounds(
        rect.origin.x, rect.origin.y, rect.size.width, rect.size.height
    )


def test_quartz_move_builds_mouse_moved_event(quartz_mouse, posted):
    quartz_mouse.move_to(10, 20)
    event_type, x, y, _, _ = decode(posted[0])
    assert (event_type, x, y) == (Quartz.kCGEventMouseMoved, 10, 20)


def test_quartz_left_drag_sequence(quartz_mouse, posted):
    quartz_mouse.move_to(10, 20)
    quartz_mouse.press_left()
    quartz_mouse.drag_to(30, 40)
    quartz_mouse.release_left()
    assert [decode(event) for event in posted[1:]] == [
        (Quartz.kCGEventLeftMouseDown, 10, 20, 1, Quartz.kCGMouseButtonLeft),
        (Quartz.kCGEventLeftMouseDragged, 30, 40, 1, Quartz.kCGMouseButtonLeft),
        (Quartz.kCGEventLeftMouseUp, 30, 40, 1, Quartz.kCGMouseButtonLeft),
    ]


def test_quartz_right_click_uses_right_button(quartz_mouse, posted):
    quartz_mouse.move_to(10, 20)
    quartz_mouse.click_right()
    assert [decode(event) for event in posted[1:]] == [
        (Quartz.kCGEventRightMouseDown, 10, 20, 1, Quartz.kCGMouseButtonRight),
        (Quartz.kCGEventRightMouseUp, 10, 20, 1, Quartz.kCGMouseButtonRight),
    ]


def test_quartz_double_click_sets_click_state_field(quartz_mouse, posted):
    quartz_mouse.double_click_left()
    assert [decode(event)[3] for event in posted] == [1, 1, 2, 2]


def test_quartz_ensure_all_released_posts_mouse_up(quartz_mouse, posted):
    quartz_mouse.press_left()
    quartz_mouse.ensure_all_released()
    assert decode(posted[-1])[0] == Quartz.kCGEventLeftMouseUp


def test_quartz_rejects_null_event(quartz_mouse, posted, monkeypatch):
    monkeypatch.setattr(Quartz, "CGEventCreateMouseEvent", lambda *args: None)
    with pytest.raises(RuntimeError):
        quartz_mouse.move_to(1, 1)
    assert posted == []


@pytest.mark.parametrize(
    "preflight, request_result, prompt, expected, requested",
    [
        (True, False, False, True, False),
        (True, False, True, True, False),
        (False, True, False, False, False),
        (False, True, True, True, True),
        (False, False, True, False, True),
    ],
)
def test_check_accessibility_permission(
    monkeypatch, preflight, request_result, prompt, expected, requested
):
    calls = []
    monkeypatch.setattr(Quartz, "CGPreflightPostEventAccess", lambda: preflight)
    monkeypatch.setattr(
        Quartz, "CGRequestPostEventAccess", lambda: calls.append(1) or request_result
    )
    assert check_accessibility_permission(prompt=prompt) is expected
    assert bool(calls) is requested
