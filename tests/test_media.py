import pytest
import Quartz
from AppKit import NSEvent, NSEventTypeSystemDefined

from src.controller.media import (
    MediaKey,
    MediaKeyEvent,
    MockMediaController,
    QuartzMediaController,
)
from src.controller.presentation import PresentationController

ACTIONS = [
    ("volume_up", MediaKey.VOLUME_UP, 0),
    ("volume_down", MediaKey.VOLUME_DOWN, 1),
    ("mute", MediaKey.MUTE, 7),
    ("play_pause", MediaKey.PLAY_PAUSE, 16),
    ("next_track", MediaKey.NEXT_TRACK, 17),
    ("prev_track", MediaKey.PREV_TRACK, 18),
]


class FailingPost:
    """Raises on the first post and records the rest."""

    def __init__(self):
        self.posted = []

    def __call__(self, event):
        self.posted.append(event)
        if len(self.posted) == 1:
            raise OSError("post failed")


@pytest.mark.parametrize("method, key, _code", ACTIONS)
def test_mock_records_a_full_tap(method, key, _code):
    media = MockMediaController()
    getattr(media, method)()
    assert media.events == [MediaKeyEvent(key, True), MediaKeyEvent(key, False)]


@pytest.mark.parametrize("method, key, code", ACTIONS)
def test_quartz_posts_system_defined_subtype_8_events(method, key, code):
    posted = []
    getattr(QuartzMediaController(post_event=posted.append), method)()
    assert len(posted) == 2
    for cg_event, state in zip(posted, (0xA, 0xB), strict=True):
        event = NSEvent.eventWithCGEvent_(cg_event)
        assert event.type() == NSEventTypeSystemDefined
        assert event.subtype() == 8
        assert event.data1() == (code << 16) | (state << 8)


def test_media_key_release_is_attempted_when_the_press_fails():
    post = FailingPost()
    with pytest.raises(OSError):
        QuartzMediaController(post_event=post).mute()
    assert len(post.posted) == 2
    assert NSEvent.eventWithCGEvent_(post.posted[1]).data1() == (7 << 16) | (0xB << 8)


@pytest.mark.parametrize(
    "method, key_code", [("next_slide", 124), ("previous_slide", 123)]
)
def test_presentation_taps_the_arrow_key(method, key_code):
    posted = []
    getattr(PresentationController(post_event=posted.append), method)()
    assert [Quartz.CGEventGetType(event) for event in posted] == [
        Quartz.kCGEventKeyDown,
        Quartz.kCGEventKeyUp,
    ]
    for event in posted:
        assert Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode) == key_code


def test_presentation_key_up_is_attempted_when_key_down_fails():
    post = FailingPost()
    with pytest.raises(OSError):
        PresentationController(post_event=post).next_slide()
    assert Quartz.CGEventGetType(post.posted[1]) == Quartz.kCGEventKeyUp
