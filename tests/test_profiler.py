import threading

import numpy as np
import pytest

from src.utils.profiler import FpsCounter, LatencyProfiler, LatencyStats, percentile


class FakeClock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.mark.parametrize("q", [0, 25, 50, 90, 95, 99, 100])
def test_percentile_matches_numpy(q):
    samples = [5.0, 1.0, 9.0, 3.0, 7.0, 2.5, 11.0]
    assert percentile(samples, q) == pytest.approx(np.percentile(samples, q))


def test_percentile_single_sample():
    assert percentile([4.0], 95) == 4.0


def test_percentile_rejects_empty_and_out_of_range():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_fps_is_zero_until_two_ticks():
    clock = FakeClock()
    counter = FpsCounter(clock=clock)
    assert counter.fps() == 0.0
    counter.tick()
    assert counter.fps() == 0.0


def test_fps_measures_tick_rate():
    clock = FakeClock()
    counter = FpsCounter(window_seconds=1.0, clock=clock)
    for _ in range(11):
        counter.tick()
        clock.now += 0.02
    clock.now -= 0.02
    assert counter.fps() == pytest.approx(50.0)


def test_fps_only_uses_the_window():
    clock = FakeClock()
    counter = FpsCounter(window_seconds=1.0, clock=clock)
    for _ in range(50):  # Slow start: 10 fps for 5 seconds.
        counter.tick()
        clock.now += 0.1
    for _ in range(100):  # Then 100 fps for 1 second.
        counter.tick()
        clock.now += 0.01
    clock.now -= 0.01
    assert counter.fps() == pytest.approx(100.0)


def test_fps_decays_to_zero_when_ticks_stop():
    clock = FakeClock()
    counter = FpsCounter(window_seconds=1.0, clock=clock)
    counter.tick()
    clock.now += 0.1
    counter.tick()
    assert counter.fps() > 0
    clock.now += 5.0
    assert counter.fps() == 0.0


def test_fps_rejects_invalid_window():
    with pytest.raises(ValueError):
        FpsCounter(window_seconds=0)


def run_trace(profiler, clock, detection_s, action_s):
    trace = profiler.begin(clock.now)
    clock.now += detection_s
    trace.mark("detection")
    clock.now += action_s
    trace.mark("action")
    trace.finish()


def test_stage_and_end_to_end_latency():
    clock = FakeClock()
    profiler = LatencyProfiler(clock=clock)
    run_trace(profiler, clock, 0.020, 0.005)
    stats = profiler.stats()
    assert stats["detection"].p50_ms == pytest.approx(20.0)
    assert stats["action"].p50_ms == pytest.approx(5.0)
    assert stats[LatencyProfiler.END_TO_END].p50_ms == pytest.approx(25.0)


def test_latency_counts_from_capture_time_not_begin_call():
    clock = FakeClock()
    profiler = LatencyProfiler(stages=("detection",), clock=clock)
    trace = profiler.begin(capture_timestamp=clock.now - 0.010)
    trace.mark("detection")
    trace.finish()
    assert profiler.stats()["detection"].p50_ms == pytest.approx(10.0)


def test_p50_and_p95_over_many_traces():
    clock = FakeClock()
    profiler = LatencyProfiler(stages=("detection",), clock=clock)
    for ms in range(1, 101):
        trace = profiler.begin(clock.now)
        clock.now += ms / 1000
        trace.mark("detection")
        trace.finish()
    stats = profiler.stats()["detection"]
    assert stats.count == 100
    assert stats.p50_ms == pytest.approx(50.5)
    assert stats.p95_ms == pytest.approx(95.05)


def test_window_discards_old_samples():
    clock = FakeClock()
    profiler = LatencyProfiler(stages=("detection",), window=2, clock=clock)
    for seconds in (10.0, 0.001, 0.003):
        trace = profiler.begin(clock.now)
        clock.now += seconds
        trace.mark("detection")
        trace.finish()
    stats = profiler.stats()["detection"]
    assert stats.count == 2
    assert stats.p50_ms == pytest.approx(2.0)


def test_frame_without_action_is_excluded_from_end_to_end():
    clock = FakeClock()
    profiler = LatencyProfiler(clock=clock)
    trace = profiler.begin(clock.now)
    clock.now += 0.020
    trace.mark("detection")
    trace.finish()
    stats = profiler.stats()
    assert stats["detection"].count == 1
    assert "action" not in stats
    assert LatencyProfiler.END_TO_END not in stats


def test_skipped_stage_counts_from_previous_mark():
    clock = FakeClock()
    profiler = LatencyProfiler(stages=("a", "b", "c"), clock=clock)
    trace = profiler.begin(clock.now)
    clock.now += 0.010
    trace.mark("a")
    clock.now += 0.030
    trace.mark("c")
    trace.finish()
    stats = profiler.stats()
    assert stats["c"].p50_ms == pytest.approx(30.0)
    assert stats[LatencyProfiler.END_TO_END].p50_ms == pytest.approx(40.0)


def test_unfinished_trace_is_not_counted():
    clock = FakeClock()
    profiler = LatencyProfiler(clock=clock)
    profiler.begin(clock.now).mark("detection")
    assert profiler.stats() == {}


def test_profiler_fps_counts_finished_traces():
    clock = FakeClock()
    profiler = LatencyProfiler(clock=clock)
    for _ in range(5):
        run_trace(profiler, clock, 0.010, 0.010)
    assert profiler.fps() == pytest.approx(50.0)


def test_unknown_stage_rejected():
    profiler = LatencyProfiler()
    with pytest.raises(ValueError, match="unknown stage"):
        profiler.begin().mark("nope")


def test_trace_cannot_be_reused_after_finish():
    trace = LatencyProfiler().begin()
    trace.finish()
    with pytest.raises(RuntimeError):
        trace.finish()
    with pytest.raises(RuntimeError):
        trace.mark("detection")


@pytest.mark.parametrize(
    "kwargs", [{"stages": ()}, {"stages": ("end_to_end",)}, {"window": 0}]
)
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        LatencyProfiler(**kwargs)


def test_concurrent_use_loses_no_samples():
    profiler = LatencyProfiler(stages=("detection",), window=10_000)

    def worker():
        for _ in range(500):
            trace = profiler.begin()
            trace.mark("detection")
            trace.finish()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert profiler.stats()["detection"].count == 2000


def test_stats_type():
    clock = FakeClock()
    profiler = LatencyProfiler(clock=clock)
    run_trace(profiler, clock, 0.001, 0.001)
    assert isinstance(profiler.stats()["detection"], LatencyStats)
