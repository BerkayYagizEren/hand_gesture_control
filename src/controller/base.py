"""Mouse controller interface, its button/click state machine, and a recording mock.

`BaseMouseController` owns all the logic (clamping, held buttons, click counting)
so `MockMouseController` exercises the same code paths as the Quartz driver.
Coordinates are global screen points with the origin at the main display's top left.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple

DEFAULT_DOUBLE_CLICK_INTERVAL = 0.5
DEFAULT_DOUBLE_CLICK_SLOP = 4.0


class Point(NamedTuple):
    x: float
    y: float


@dataclass(frozen=True)
class ScreenBounds:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"invalid screen size {self.width}x{self.height}")

    @property
    def center(self) -> Point:
        return Point(self.x + self.width / 2, self.y + self.height / 2)

    def clamp(self, x: float, y: float) -> Point:
        """Limit a point to the display; the far edge is one point inside the size."""
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"non-finite coordinates ({x}, {y})")
        return Point(
            min(max(x, self.x), self.x + self.width - 1),
            min(max(y, self.y), self.y + self.height - 1),
        )


class MouseButton(Enum):
    LEFT = auto()
    RIGHT = auto()


class MouseEventType(Enum):
    MOVED = auto()
    LEFT_DOWN = auto()
    LEFT_UP = auto()
    LEFT_DRAGGED = auto()
    RIGHT_DOWN = auto()
    RIGHT_UP = auto()
    RIGHT_DRAGGED = auto()


_DOWN = {MouseButton.LEFT: MouseEventType.LEFT_DOWN, MouseButton.RIGHT: MouseEventType.RIGHT_DOWN}
_UP = {MouseButton.LEFT: MouseEventType.LEFT_UP, MouseButton.RIGHT: MouseEventType.RIGHT_UP}


@dataclass(frozen=True)
class MouseEvent:
    type: MouseEventType
    x: float
    y: float
    click_state: int = 0  # kCGMouseEventClickState; 0 for plain moves.


@dataclass(frozen=True)
class ScrollEvent:
    """Whole pixels, signed as Quartz wheels: +dy scrolls up, +dx scrolls left."""

    dx: int
    dy: int


class BaseMouseController(ABC):
    """Mouse driver. Use as a context manager to guarantee release on exit.

    The position is the last commanded one, because the OS may not have applied
    a just-posted move when it is read back. Call `sync_position` when gesture
    control takes over from the physical mouse.
    """

    def __init__(
        self,
        double_click_interval: float = DEFAULT_DOUBLE_CLICK_INTERVAL,
        double_click_slop: float = DEFAULT_DOUBLE_CLICK_SLOP,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._double_click_interval = double_click_interval
        self._double_click_slop = double_click_slop
        self._clock = clock
        self._held: dict[MouseButton, int] = {}  # Button -> click state of its press.
        self._last_press: tuple[MouseButton, float, Point, int] | None = None
        self._position = self._read_position()

    @abstractmethod
    def screen_bounds(self) -> ScreenBounds:
        """Return the bounds of the controlled display, in points."""

    @abstractmethod
    def _read_position(self) -> Point:
        """Return the cursor position as the OS currently reports it."""

    @abstractmethod
    def _post(self, event: MouseEvent) -> None:
        """Deliver one event to the OS."""

    @abstractmethod
    def _post_scroll(self, event: ScrollEvent) -> None:
        """Deliver one scroll event to the OS."""

    @property
    def position(self) -> Point:
        return self._position

    @property
    def held_buttons(self) -> frozenset[MouseButton]:
        return frozenset(self._held)

    def sync_position(self) -> Point:
        """Adopt the OS cursor position, e.g. after the physical mouse moved it."""
        self._position = self._read_position()
        return self._position

    def move_to(self, x: float, y: float) -> None:
        """Move the cursor. Sends a drag event while a button is held, as the OS would."""
        target = self.screen_bounds().clamp(x, y)
        if MouseButton.LEFT in self._held:
            event_type, click_state = MouseEventType.LEFT_DRAGGED, self._held[MouseButton.LEFT]
        elif MouseButton.RIGHT in self._held:
            event_type, click_state = MouseEventType.RIGHT_DRAGGED, self._held[MouseButton.RIGHT]
        else:
            event_type, click_state = MouseEventType.MOVED, 0
        self._post(MouseEvent(event_type, target.x, target.y, click_state))
        self._position = target

    def drag_to(self, x: float, y: float) -> None:
        """Move with the left button held. Raises if it is not, since that is a caller bug."""
        if MouseButton.LEFT not in self._held:
            raise RuntimeError("drag_to requires the left button to be pressed")
        self.move_to(x, y)

    def press_left(self) -> None:
        self._press(MouseButton.LEFT)

    def release_left(self) -> None:
        self._release(MouseButton.LEFT)

    def press_right(self) -> None:
        self._press(MouseButton.RIGHT)

    def release_right(self) -> None:
        self._release(MouseButton.RIGHT)

    def click_left(self) -> None:
        self.press_left()
        self.release_left()

    def click_right(self) -> None:
        self.press_right()
        self.release_right()

    def double_click_left(self) -> None:
        self.click_left()
        self.click_left()

    def scroll(self, dx: float, dy: float) -> None:
        """Scroll by pixels at the cursor, signed as `ScrollEvent`. Rounds to whole pixels."""
        if not (math.isfinite(dx) and math.isfinite(dy)):
            raise ValueError(f"non-finite scroll ({dx}, {dy})")
        event = ScrollEvent(round(dx), round(dy))
        if event.dx or event.dy:
            self._post_scroll(event)

    def ensure_all_released(self) -> None:
        """Release every held button. Tries them all even if one fails, then re-raises."""
        first_error: Exception | None = None
        for button in tuple(self._held):
            try:
                self._release(button)
            except Exception as error:
                first_error = first_error or error
        if first_error is not None:
            raise first_error

    def __enter__(self) -> BaseMouseController:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.ensure_all_released()

    def _press(self, button: MouseButton) -> None:
        if button in self._held:
            return
        now = self._clock()
        click_state = self._next_click_state(button, now)
        self._post(MouseEvent(_DOWN[button], *self._position, click_state))
        self._held[button] = click_state
        self._last_press = (button, now, self._position, click_state)

    def _release(self, button: MouseButton) -> None:
        if button not in self._held:
            return
        self._post(MouseEvent(_UP[button], *self._position, self._held[button]))
        del self._held[button]

    def _next_click_state(self, button: MouseButton, now: float) -> int:
        if self._last_press is None:
            return 1
        last_button, last_time, last_position, last_state = self._last_press
        is_repeat = (
            last_button is button
            and now - last_time <= self._double_click_interval
            and math.dist(last_position, self._position) <= self._double_click_slop
        )
        return last_state + 1 if is_repeat else 1


class MockMouseController(BaseMouseController):
    """Records events in `events` instead of touching the OS."""

    def __init__(
        self,
        bounds: ScreenBounds = ScreenBounds(0, 0, 1920, 1080),
        start: Point = Point(0, 0),
        double_click_interval: float = DEFAULT_DOUBLE_CLICK_INTERVAL,
        double_click_slop: float = DEFAULT_DOUBLE_CLICK_SLOP,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.bounds = bounds
        self.os_position = start  # Set directly to simulate the physical mouse.
        self.events: list[MouseEvent] = []
        self.scroll_events: list[ScrollEvent] = []
        super().__init__(double_click_interval, double_click_slop, clock)

    def screen_bounds(self) -> ScreenBounds:
        return self.bounds

    def _read_position(self) -> Point:
        return self.os_position

    def _post(self, event: MouseEvent) -> None:
        self.events.append(event)
        self.os_position = Point(event.x, event.y)

    def _post_scroll(self, event: ScrollEvent) -> None:
        self.scroll_events.append(event)
