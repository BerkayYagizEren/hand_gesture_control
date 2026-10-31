"""macOS media keys, posted as NSSystemDefined events (subtype 8)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

from AppKit import NSEvent, NSEventTypeSystemDefined

from src.controller.quartz_mouse import post_to_hid_tap

_AUX_CONTROL_SUBTYPE = 8
_KEY_DOWN_STATE = 0xA
_KEY_UP_STATE = 0xB


class MediaKey(IntEnum):
    """NX_KEYTYPE_* codes from IOKit's ev_keymap.h."""

    VOLUME_UP = 0
    VOLUME_DOWN = 1
    MUTE = 7
    PLAY_PAUSE = 16
    NEXT_TRACK = 17
    PREV_TRACK = 18


@dataclass(frozen=True)
class MediaKeyEvent:
    key: MediaKey
    is_down: bool


class BaseMediaController(ABC):
    """Media key driver. Every action is a full tap, so no key is left down."""

    @abstractmethod
    def _post(self, event: MediaKeyEvent) -> None:
        """Deliver one key transition to the OS."""

    def volume_up(self) -> None:
        self.tap(MediaKey.VOLUME_UP)

    def volume_down(self) -> None:
        self.tap(MediaKey.VOLUME_DOWN)

    def mute(self) -> None:
        self.tap(MediaKey.MUTE)

    def play_pause(self) -> None:
        self.tap(MediaKey.PLAY_PAUSE)

    def next_track(self) -> None:
        self.tap(MediaKey.NEXT_TRACK)

    def prev_track(self) -> None:
        self.tap(MediaKey.PREV_TRACK)

    def tap(self, key: MediaKey) -> None:
        """Press and release `key`. The release is attempted even if the press fails."""
        try:
            self._post(MediaKeyEvent(key, True))
        finally:
            self._post(MediaKeyEvent(key, False))


class MockMediaController(BaseMediaController):
    """Records events in `events` instead of touching the OS."""

    def __init__(self) -> None:
        self.events: list[MediaKeyEvent] = []

    def _post(self, event: MediaKeyEvent) -> None:
        self.events.append(event)


class QuartzMediaController(BaseMediaController):
    """Posts the media key events at the HID tap, as the keyboard's own keys arrive."""

    def __init__(self, post_event: Callable[[object], None] = post_to_hid_tap) -> None:
        self._post_event = post_event

    def _post(self, event: MediaKeyEvent) -> None:
        state = _KEY_DOWN_STATE if event.is_down else _KEY_UP_STATE
        ns_event = NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
            NSEventTypeSystemDefined,
            (0, 0),
            state << 8,
            0,
            0,
            None,
            _AUX_CONTROL_SUBTYPE,
            (int(event.key) << 16) | (state << 8),
            -1,
        )
        if ns_event is None:
            raise RuntimeError(f"NSEvent creation failed for {event}")
        self._post_event(ns_event.CGEvent())
