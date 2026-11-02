"""macOS Spaces and Mission Control navigation through native keyboard events."""

from __future__ import annotations

from collections.abc import Callable

KEY_CODE_LEFT_ARROW = 123
KEY_CODE_RIGHT_ARROW = 124
KEY_CODE_UP_ARROW = 126


class SwapController:
    """Taps Control + arrow, which macOS handles whichever app has keyboard focus."""

    def __init__(self, post_event: Callable[[object], None] | None = None) -> None:
        if post_event is None:
            from src.controller.quartz_mouse import post_to_hid_tap

            post_event = post_to_hid_tap
        self._post_event = post_event

    def next_space(self) -> None:
        self._tap(KEY_CODE_RIGHT_ARROW)

    def previous_space(self) -> None:
        self._tap(KEY_CODE_LEFT_ARROW)

    def mission_control(self) -> None:
        self._tap(KEY_CODE_UP_ARROW)

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
        # Imported on use, because Quartz loads AppKit and replay must not need it.
        import Quartz

        event = Quartz.CGEventCreateKeyboardEvent(None, key_code, is_down)
        if event is None:
            raise RuntimeError(f"CGEventCreateKeyboardEvent failed for key {key_code}")
        # The flag travels on the event, so no real Control key can be left held down.
        Quartz.CGEventSetFlags(event, Quartz.kCGEventFlagMaskControl)
        return event
