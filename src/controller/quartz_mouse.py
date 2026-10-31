"""Native macOS mouse driver built on Quartz event services."""

from __future__ import annotations

import time
from collections.abc import Callable

import Quartz

from src.controller.base import (
    DEFAULT_DOUBLE_CLICK_INTERVAL,
    DEFAULT_DOUBLE_CLICK_SLOP,
    BaseMouseController,
    MouseEvent,
    MouseEventType,
    Point,
    ScreenBounds,
    ScrollEvent,
)

_QUARTZ_EVENTS = {
    MouseEventType.MOVED: (Quartz.kCGEventMouseMoved, Quartz.kCGMouseButtonLeft),
    MouseEventType.LEFT_DOWN: (Quartz.kCGEventLeftMouseDown, Quartz.kCGMouseButtonLeft),
    MouseEventType.LEFT_UP: (Quartz.kCGEventLeftMouseUp, Quartz.kCGMouseButtonLeft),
    MouseEventType.LEFT_DRAGGED: (Quartz.kCGEventLeftMouseDragged, Quartz.kCGMouseButtonLeft),
    MouseEventType.RIGHT_DOWN: (Quartz.kCGEventRightMouseDown, Quartz.kCGMouseButtonRight),
    MouseEventType.RIGHT_UP: (Quartz.kCGEventRightMouseUp, Quartz.kCGMouseButtonRight),
    MouseEventType.RIGHT_DRAGGED: (Quartz.kCGEventRightMouseDragged, Quartz.kCGMouseButtonRight),
}


def check_accessibility_permission(prompt: bool = False) -> bool:
    """Return whether this process may post input events.

    macOS silently drops posted events without it. With `prompt`, the system
    permission dialog is shown if access has not been granted. The grant applies
    to the host app (Terminal, PyCharm, ...), which may need a restart afterwards.
    """
    if Quartz.CGPreflightPostEventAccess():
        return True
    return bool(Quartz.CGRequestPostEventAccess()) if prompt else False


def post_to_hid_tap(event: object) -> None:
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


class QuartzMouseController(BaseMouseController):
    """Posts CGEvents at the HID tap. Controls the main display unless `display_id` is given."""

    def __init__(
        self,
        display_id: int | None = None,
        double_click_interval: float = DEFAULT_DOUBLE_CLICK_INTERVAL,
        double_click_slop: float = DEFAULT_DOUBLE_CLICK_SLOP,
        clock: Callable[[], float] = time.perf_counter,
        post_event: Callable[[object], None] = post_to_hid_tap,
    ) -> None:
        self._display_id = Quartz.CGMainDisplayID() if display_id is None else display_id
        self._post_event = post_event
        super().__init__(double_click_interval, double_click_slop, clock)

    def screen_bounds(self) -> ScreenBounds:
        # Read every time so resolution changes apply without a restart.
        rect = Quartz.CGDisplayBounds(self._display_id)
        return ScreenBounds(rect.origin.x, rect.origin.y, rect.size.width, rect.size.height)

    def _read_position(self) -> Point:
        location = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        return Point(location.x, location.y)

    def _post(self, event: MouseEvent) -> None:
        event_type, button = _QUARTZ_EVENTS[event.type]
        cg_event = Quartz.CGEventCreateMouseEvent(None, event_type, (event.x, event.y), button)
        if cg_event is None:
            raise RuntimeError(f"CGEventCreateMouseEvent failed for {event}")
        if event.click_state:
            Quartz.CGEventSetIntegerValueField(
                cg_event, Quartz.kCGMouseEventClickState, event.click_state
            )
        self._post_event(cg_event)

    def _post_scroll(self, event: ScrollEvent) -> None:
        # The fixed-arity variant, since the plain one is variadic in C. Wheel 1 is vertical.
        cg_event = Quartz.CGEventCreateScrollWheelEvent2(
            None, Quartz.kCGScrollEventUnitPixel, 2, event.dy, event.dx, 0
        )
        if cg_event is None:
            raise RuntimeError(f"CGEventCreateScrollWheelEvent2 failed for {event}")
        self._post_event(cg_event)
