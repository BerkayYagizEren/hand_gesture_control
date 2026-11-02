import pytest
import Quartz

from src.controller.swap import SwapController

ACTIONS = [("next_space", 124), ("previous_space", 123), ("mission_control", 126)]


class FailingPost:
    """Raises on the first post and records the rest."""

    def __init__(self):
        self.posted = []

    def __call__(self, event):
        self.posted.append(event)
        if len(self.posted) == 1:
            raise OSError("post failed")


@pytest.mark.parametrize("method, key_code", ACTIONS)
def test_swap_taps_the_arrow_key(method, key_code):
    posted = []
    getattr(SwapController(post_event=posted.append), method)()
    assert [Quartz.CGEventGetType(event) for event in posted] == [
        Quartz.kCGEventKeyDown,
        Quartz.kCGEventKeyUp,
    ]
    for event in posted:
        assert Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode) == key_code


@pytest.mark.parametrize("method, _key_code", ACTIONS)
def test_swap_holds_only_control_on_both_events(method, _key_code):
    posted = []
    getattr(SwapController(post_event=posted.append), method)()
    for event in posted:
        assert Quartz.CGEventGetFlags(event) == Quartz.kCGEventFlagMaskControl


def test_swap_key_up_is_attempted_when_key_down_fails():
    post = FailingPost()
    with pytest.raises(OSError):
        SwapController(post_event=post).next_space()
    assert Quartz.CGEventGetType(post.posted[1]) == Quartz.kCGEventKeyUp


def test_swap_failed_event_creation_posts_nothing(monkeypatch):
    posted = []
    monkeypatch.setattr(Quartz, "CGEventCreateKeyboardEvent", lambda *_args: None)
    with pytest.raises(RuntimeError):
        SwapController(post_event=posted.append).next_space()
    assert posted == []
