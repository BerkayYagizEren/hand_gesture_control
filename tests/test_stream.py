import threading
import time

import cv2
import numpy as np
import pytest

from src.camera.stream import CameraError, CameraStream, Frame, LatestFrameBuffer


class FakeCapture:
    """Stands in for cv2.VideoCapture; pixel (0, 0) holds the frame number."""

    def __init__(self, opened=True, interval=0.001, fail_after=None, raise_after=None):
        self.opened = opened
        self.interval = interval
        self.fail_after = fail_after
        self.raise_after = raise_after
        self.props = {cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480}
        self.set_calls = {}
        self.count = 0
        self.released = threading.Event()

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.set_calls[prop] = value
        self.props[prop] = value
        return True

    def get(self, prop):
        return self.props.get(prop, 0)

    def read(self):
        time.sleep(self.interval)
        if self.raise_after is not None and self.count >= self.raise_after:
            raise OSError("device gone")
        if self.fail_after is not None and self.count >= self.fail_after:
            return False, None
        self.count += 1
        image = np.zeros((2, 4, 3), dtype=np.uint8)
        image[0, 0, 0] = self.count % 256
        return True, image

    def release(self):
        self.released.set()


def make_frame(seq):
    return Frame(np.zeros((1, 1, 3), dtype=np.uint8), seq, float(seq))


def stream_for(capture, **kwargs):
    return CameraStream(capture_factory=lambda source: capture, **kwargs)


def test_buffer_keeps_only_newest_frame_and_counts_drops():
    buffer = LatestFrameBuffer()
    for seq in (1, 2, 3):
        buffer.put(make_frame(seq))
    assert buffer.take(0).seq == 3
    assert buffer.dropped == 2


def test_buffer_take_consumes_the_frame():
    buffer = LatestFrameBuffer()
    buffer.put(make_frame(1))
    buffer.take(0)
    assert buffer.take(0) is None
    assert buffer.dropped == 0


def test_buffer_take_blocks_until_frame_arrives():
    buffer = LatestFrameBuffer()
    timer = threading.Timer(0.02, buffer.put, args=(make_frame(7),))
    timer.start()
    assert buffer.take(5.0).seq == 7
    timer.join()


def test_buffer_take_times_out():
    assert LatestFrameBuffer().take(0.01) is None


def test_buffer_close_wakes_blocked_consumer():
    buffer = LatestFrameBuffer()
    timer = threading.Timer(0.02, buffer.close, args=("boom",))
    timer.start()
    with pytest.raises(CameraError, match="boom"):
        buffer.take(5.0)
    timer.join()


def test_buffer_delivers_last_frame_before_reporting_close():
    buffer = LatestFrameBuffer()
    buffer.put(make_frame(1))
    buffer.close()
    assert buffer.take(0).seq == 1
    with pytest.raises(CameraError):
        buffer.take(0)


def test_read_returns_frames_with_increasing_seq_and_timestamp():
    with stream_for(FakeCapture()) as camera:
        first = camera.read(timeout=5.0)
        second = camera.read(timeout=5.0)
    assert second.seq > first.seq
    assert second.timestamp > first.timestamp


def test_read_never_returns_the_same_frame_twice():
    with stream_for(FakeCapture()) as camera:
        seqs = [camera.read(timeout=5.0).seq for _ in range(20)]
    assert seqs == sorted(set(seqs))


def test_slow_consumer_skips_stale_frames():
    capture = FakeCapture()
    with stream_for(capture) as camera:
        camera.read(timeout=5.0)
        time.sleep(0.1)  # Capture runs ahead while the consumer is busy.
        captured_before_read = capture.count
        frame = camera.read(timeout=5.0)
        assert frame.seq >= captured_before_read - 1
        assert camera.dropped_frames > 0


def test_read_times_out_when_no_new_frame():
    with stream_for(FakeCapture(interval=0.5)) as camera:
        assert camera.read(timeout=0.01) is None


def test_mirror_flips_horizontally():
    with stream_for(FakeCapture(), mirror=True) as camera:
        image = camera.read(timeout=5.0).image
    assert image[0, -1, 0] != 0 and image[0, 0, 0] == 0


def test_mirror_disabled_keeps_orientation():
    with stream_for(FakeCapture(), mirror=False) as camera:
        image = camera.read(timeout=5.0).image
    assert image[0, 0, 0] != 0 and image[0, -1, 0] == 0


def test_requested_resolution_is_applied():
    capture = FakeCapture()
    with stream_for(capture, width=1280, height=720) as camera:
        assert camera.resolution == (1280, 720)
    assert capture.set_calls[cv2.CAP_PROP_FRAME_WIDTH] == 1280
    assert capture.set_calls[cv2.CAP_PROP_FRAME_HEIGHT] == 720


def test_resolution_reports_what_the_driver_granted():
    capture = FakeCapture()
    capture.set = lambda prop, value: False
    with stream_for(capture, width=1280, height=720) as camera:
        assert camera.resolution == (640, 480)


def test_resolution_untouched_by_default():
    capture = FakeCapture()
    with stream_for(capture) as camera:
        assert camera.resolution == (640, 480)
    assert cv2.CAP_PROP_FRAME_WIDTH not in capture.set_calls


def test_internal_capture_buffer_is_minimized():
    capture = FakeCapture()
    with stream_for(capture):
        pass
    assert capture.set_calls[cv2.CAP_PROP_BUFFERSIZE] == 1


def test_fps_is_measured():
    with stream_for(FakeCapture(interval=0.005)) as camera:
        deadline = time.monotonic() + 5.0
        while camera.fps == 0.0 and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.1)
        assert 0 < camera.fps <= 1 / 0.005 * 1.2


def test_context_manager_stops_thread_and_releases_capture():
    capture = FakeCapture()
    with stream_for(capture) as camera:
        assert camera.is_running
    assert not camera.is_running
    assert capture.released.is_set()


def test_context_manager_stops_on_exception():
    capture = FakeCapture()
    with pytest.raises(KeyError):
        with stream_for(capture):
            raise KeyError("consumer bug")
    assert capture.released.is_set()


def test_open_failure_raises_and_releases():
    capture = FakeCapture(opened=False)
    camera = stream_for(capture)
    with pytest.raises(CameraError, match="could not open"):
        camera.start()
    assert capture.released.is_set()
    assert not camera.is_running


def test_read_before_start_raises():
    with pytest.raises(CameraError, match="not started"):
        stream_for(FakeCapture()).read(timeout=0)


def test_read_after_stop_raises_instead_of_hanging():
    camera = stream_for(FakeCapture())
    with camera:
        camera.read(timeout=5.0)
    with pytest.raises(CameraError):
        for _ in range(2):  # One frame may still be pending in the buffer.
            camera.read(timeout=5.0)


def test_persistent_read_failure_surfaces_to_consumer():
    capture = FakeCapture(fail_after=0)
    with stream_for(capture, max_read_failures=3) as camera:
        with pytest.raises(CameraError, match="3 times"):
            camera.read(timeout=5.0)
    assert capture.released.is_set()


def test_transient_read_failure_recovers():
    capture = FakeCapture()
    real_read = capture.read
    results = iter([(False, None), (False, None)])
    capture.read = lambda: next(results, None) or real_read()
    with stream_for(capture, max_read_failures=3) as camera:
        assert camera.read(timeout=5.0) is not None


def test_capture_exception_surfaces_to_consumer():
    capture = FakeCapture(raise_after=0)
    with stream_for(capture) as camera:
        with pytest.raises(CameraError, match="device gone"):
            camera.read(timeout=5.0)
    assert capture.released.is_set()


def test_double_start_rejected():
    with stream_for(FakeCapture()) as camera:
        with pytest.raises(RuntimeError, match="already running"):
            camera.start()


def test_stop_is_idempotent_and_safe_before_start():
    camera = stream_for(FakeCapture())
    camera.stop()
    camera.start()
    camera.stop()
    camera.stop()


def test_stream_can_restart():
    captures = [FakeCapture(), FakeCapture()]
    remaining = iter(captures)
    camera = CameraStream(capture_factory=lambda source: next(remaining))
    with camera:
        camera.read(timeout=5.0)
    with camera:
        assert camera.read(timeout=5.0).seq >= 1
    assert all(capture.released.is_set() for capture in captures)


@pytest.mark.parametrize(
    "kwargs", [{"width": 0}, {"height": -1}, {"max_read_failures": 0}]
)
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        CameraStream(**kwargs)
