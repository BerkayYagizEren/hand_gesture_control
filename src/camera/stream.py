"""Threaded OpenCV camera capture with a single-slot latest-frame buffer."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from src.utils.profiler import FpsCounter

logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """The camera could not be opened or the capture has ended."""


@dataclass(frozen=True)
class Frame:
    image: np.ndarray  # BGR, as delivered by OpenCV.
    seq: int  # 1-based capture counter; gaps mean frames were dropped.
    timestamp: float  # Capture time in seconds on the stream's clock.


class LatestFrameBuffer:
    """Single-slot buffer: a new frame atomically replaces an unread one."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._closed = False
        self._error: str | None = None
        self._dropped = 0

    def put(self, frame: Frame) -> None:
        with self._cond:
            if self._frame is not None:
                self._dropped += 1
            self._frame = frame
            self._cond.notify_all()

    def take(self, timeout: float | None = None) -> Frame | None:
        """Return the newest unread frame, waiting up to `timeout` seconds.

        Returns None on timeout. Once closed, raises `CameraError` as soon as
        the last frame has been consumed, so a consumer can never hang.
        """
        with self._cond:
            self._cond.wait_for(lambda: self._frame is not None or self._closed, timeout)
            frame, self._frame = self._frame, None
            if frame is None and self._closed:
                raise CameraError(self._error or "camera stream is stopped")
            return frame

    def close(self, error: str | None = None) -> None:
        with self._cond:
            self._closed = True
            self._error = error
            self._cond.notify_all()

    @property
    def dropped(self) -> int:
        with self._cond:
            return self._dropped


class CameraStream:
    """Captures on a background thread; consumers always get the newest frame.

    Frames the consumer was too slow to read are discarded, so processing time
    never turns into accumulated lag.

        with CameraStream(width=1280, height=720) as camera:
            frame = camera.read(timeout=1.0)
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int | None = None,
        height: int | None = None,
        mirror: bool = True,
        max_read_failures: int = 30,
        capture_factory: Callable[[int | str], cv2.VideoCapture] = cv2.VideoCapture,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if (width is not None and width <= 0) or (height is not None and height <= 0):
            raise ValueError("width and height must be positive")
        if max_read_failures <= 0:
            raise ValueError("max_read_failures must be positive")
        self._source = source
        self._width = width
        self._height = height
        self._mirror = mirror
        self._max_read_failures = max_read_failures
        self._capture_factory = capture_factory
        self._clock = clock
        self._fps = FpsCounter(clock=clock)
        self._buffer: LatestFrameBuffer | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._resolution = (0, 0)

    def start(self) -> CameraStream:
        if self.is_running:
            raise RuntimeError("camera stream is already running")
        # Opened on the caller's thread: AVFoundation's permission prompt is
        # unreliable when it is first triggered from a background thread.
        capture = self._capture_factory(self._source)
        if not capture.isOpened():
            capture.release()
            raise CameraError(f"could not open camera source {self._source!r}")
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self._width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        if self._height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        # Drivers snap to the nearest supported mode, so report what was granted.
        self._resolution = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self._buffer = LatestFrameBuffer()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._capture_loop,
            args=(capture, self._buffer, self._stop_event),
            name="camera-capture",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self, join_timeout: float = 2.0) -> None:
        """Stop capturing. Safe to call repeatedly."""
        self._stop_event.set()
        if self._thread is None:
            return
        self._thread.join(join_timeout)
        if self._thread.is_alive():
            # The loop releases the capture itself once the blocked read returns.
            logger.warning("camera thread did not stop within %.1fs", join_timeout)
        self._thread = None

    def read(self, timeout: float | None = None) -> Frame | None:
        """Return the newest frame not yet read, or None after `timeout` seconds.

        Raises `CameraError` if the stream was never started, was stopped, or
        the capture failed.
        """
        if self._buffer is None:
            raise CameraError("camera stream is not started")
        return self._buffer.take(timeout)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def fps(self) -> float:
        """Measured capture rate over the last second."""
        return self._fps.fps()

    @property
    def resolution(self) -> tuple[int, int]:
        """(width, height) granted by the driver; (0, 0) before `start`."""
        return self._resolution

    @property
    def dropped_frames(self) -> int:
        """Frames discarded unread since the last `start`."""
        return self._buffer.dropped if self._buffer is not None else 0

    def __enter__(self) -> CameraStream:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def _capture_loop(
        self,
        capture: cv2.VideoCapture,
        buffer: LatestFrameBuffer,
        stop_event: threading.Event,
    ) -> None:
        seq = 0
        failures = 0
        error = None
        try:
            while not stop_event.is_set():
                ok, image = capture.read()
                timestamp = self._clock()
                if not ok or image is None:
                    failures += 1
                    if failures >= self._max_read_failures:
                        error = f"camera read failed {failures} times in a row"
                        break
                    stop_event.wait(0.01)
                    continue
                failures = 0
                if self._mirror:
                    image = cv2.flip(image, 1)
                seq += 1
                buffer.put(Frame(image, seq, timestamp))
                self._fps.tick(timestamp)
        except Exception as exc:
            logger.exception("camera capture thread crashed")
            error = f"camera capture crashed: {exc}"
        finally:
            capture.release()
            buffer.close(error)
