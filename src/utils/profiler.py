"""Lightweight end-to-end latency profiler (capture -> detection -> action).

All timestamps are seconds on one monotonic clock (`time.perf_counter` by
default), the same clock `CameraStream` uses to stamp frames.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass


def percentile(samples: Sequence[float], q: float) -> float:
    """Return the q-th percentile (0-100) using linear interpolation."""
    if not samples:
        raise ValueError("percentile of an empty sequence")
    if not 0 <= q <= 100:
        raise ValueError(f"q must be in [0, 100], got {q}")
    ordered = sorted(samples)
    rank = (len(ordered) - 1) * q / 100
    low = math.floor(rank)
    high = math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


class FpsCounter:
    """Thread-safe FPS meter over a sliding time window.

    Reads evict old ticks too, so the rate decays to 0 when ticks stop instead
    of reporting the last value forever.
    """

    def __init__(
        self,
        window_seconds: float = 1.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._window = window_seconds
        self._clock = clock
        self._ticks: deque[float] = deque()
        self._lock = threading.Lock()

    def tick(self, now: float | None = None) -> None:
        now = self._clock() if now is None else now
        with self._lock:
            self._ticks.append(now)
            self._evict(now)

    def fps(self) -> float:
        with self._lock:
            self._evict(self._clock())
            if len(self._ticks) < 2:
                return 0.0
            elapsed = self._ticks[-1] - self._ticks[0]
            return (len(self._ticks) - 1) / elapsed if elapsed > 0 else 0.0

    def _evict(self, now: float) -> None:
        while self._ticks and now - self._ticks[0] > self._window:
            self._ticks.popleft()


@dataclass(frozen=True)
class LatencyStats:
    count: int
    p50_ms: float
    p95_ms: float


class Trace:
    """Timing of one frame through the pipeline. Create via `LatencyProfiler.begin`."""

    def __init__(self, profiler: LatencyProfiler, start: float) -> None:
        self._profiler = profiler
        self.start = start
        self._marks: list[tuple[str, float]] = []
        self._finished = False

    def mark(self, stage: str, now: float | None = None) -> None:
        """Record that `stage` has just completed."""
        if self._finished:
            raise RuntimeError("trace is already finished")
        self._profiler._check_stage(stage)
        self._marks.append((stage, self._profiler._clock() if now is None else now))

    def finish(self) -> None:
        """Submit the trace. Stages that were not marked get no sample."""
        if self._finished:
            raise RuntimeError("trace is already finished")
        self._finished = True
        self._profiler._submit(self.start, self._marks)


class LatencyProfiler:
    """Collects per-stage and end-to-end latency over the last `window` samples.

    A stage's latency is the time since the previous mark (or since capture for
    the first one). End-to-end only counts frames that reached the final stage,
    so frames without an action cannot skew it.

        trace = profiler.begin(frame.timestamp)
        result = detector.detect(frame.image)
        trace.mark("detection")
        controller.apply(result)
        trace.mark("action")
        trace.finish()
    """

    END_TO_END = "end_to_end"

    def __init__(
        self,
        stages: Iterable[str] = ("detection", "action"),
        window: int = 300,
        fps_window_seconds: float = 1.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._stages = tuple(stages)
        if not self._stages:
            raise ValueError("at least one stage is required")
        if self.END_TO_END in self._stages:
            raise ValueError(f"{self.END_TO_END!r} is a reserved stage name")
        if window <= 0:
            raise ValueError("window must be positive")
        self._clock = clock
        self._samples: dict[str, deque[float]] = {
            name: deque(maxlen=window) for name in (*self._stages, self.END_TO_END)
        }
        self._fps = FpsCounter(fps_window_seconds, clock)
        self._lock = threading.Lock()

    def begin(self, capture_timestamp: float | None = None) -> Trace:
        """Start a trace from the frame's capture time (defaults to now)."""
        return Trace(self, self._clock() if capture_timestamp is None else capture_timestamp)

    def stats(self) -> dict[str, LatencyStats]:
        """Return stats for each stage and `END_TO_END`; unsampled ones are omitted."""
        with self._lock:
            snapshot = {name: list(values) for name, values in self._samples.items() if values}
        return {
            name: LatencyStats(
                count=len(values),
                p50_ms=percentile(values, 50) * 1000,
                p95_ms=percentile(values, 95) * 1000,
            )
            for name, values in snapshot.items()
        }

    def fps(self) -> float:
        """Rate of finished traces, i.e. pipeline throughput."""
        return self._fps.fps()

    def _check_stage(self, stage: str) -> None:
        if stage not in self._stages:
            raise ValueError(f"unknown stage {stage!r}, expected one of {self._stages}")

    def _submit(self, start: float, marks: list[tuple[str, float]]) -> None:
        with self._lock:
            previous = start
            for stage, timestamp in marks:
                self._samples[stage].append(timestamp - previous)
                previous = timestamp
            if marks and marks[-1][0] == self._stages[-1]:
                self._samples[self.END_TO_END].append(previous - start)
        self._fps.tick()
