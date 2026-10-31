"""Slide navigation through native arrow key events."""

from __future__ import annotations

from collections.abc import Callable

import Quartz

from src.controller.quartz_mouse import post_to_hid_tap

KEY_CODE_LEFT_ARROW = 123
KEY_CODE_RIGHT_ARROW = 124


class PresentationController:
    """Taps arrow keys for whichever app has keyboard focus."""

    def __init__(self, post_event: Callable[[object], None] = post_to_hid_tap) -> None:
        self._post_event = post_event

    def next_slide(self) -> None:
        self._tap(KEY_CODE_RIGHT_ARROW)

    def previous_slide(self) -> None:
        self._tap(KEY_CODE_LEFT_ARROW)

    def _tap(self, key_code: int) -> None:
        # Both events exist before the first post, so a failed creation cannot strand a key.
        down = self._key_event(key_code, True)
        up = self._key_event(key_code, False)
        try:
            self._post_event(down)
        finally:
            self._post_event(up)

    @staticmethod
    def _key_event(key_code: int, is_down: bool) -> object:
        event = Quartz.CGEventCreateKeyboardEvent(None, key_code, is_down)
        if event is None:
            raise RuntimeError(f"CGEventCreateKeyboardEvent failed for key {key_code}")
        return event
