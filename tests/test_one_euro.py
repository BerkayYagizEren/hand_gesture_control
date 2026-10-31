import math
import random
import statistics

import pytest

from src.tracking.one_euro import OneEuroFilter, OneEuroFilter2D, smoothing_factor


def run(filter_, samples):
    return [filter_.filter(value, timestamp) for timestamp, value in samples]


def test_smoothing_factor_matches_definition():
    dt = 1 / 30
    tau = 1 / (2 * math.pi * 1.0)
    assert smoothing_factor(1.0, dt) == pytest.approx(dt / (dt + tau))


def test_first_sample_passes_through():
    assert OneEuroFilter().filter(0.42, 1.0) == 0.42


def test_second_step_matches_hand_computed_values():
    min_cutoff, beta, d_cutoff, dt = 1.5, 0.7, 2.0, 0.04
    filter_ = OneEuroFilter(min_cutoff, beta, d_cutoff)
    filter_.filter(1.0, 0.0)

    derivative = smoothing_factor(d_cutoff, dt) * (2.0 - 1.0) / dt
    alpha = smoothing_factor(min_cutoff + beta * abs(derivative), dt)
    assert filter_.filter(2.0, dt) == pytest.approx(alpha * 2.0 + (1 - alpha) * 1.0)


def test_derivative_uses_raw_previous_sample_across_steps():
    min_cutoff, beta, d_cutoff, dt = 1.0, 0.5, 1.0, 0.1
    filter_ = OneEuroFilter(min_cutoff, beta, d_cutoff)
    filter_.filter(0.0, 0.0)
    d_alpha = smoothing_factor(d_cutoff, dt)
    d1 = d_alpha * 1.0 / dt
    a1 = smoothing_factor(min_cutoff + beta * d1, dt)
    x1 = a1 * 1.0
    assert filter_.filter(1.0, dt) == pytest.approx(x1)

    d2 = d_alpha * (3.0 - 1.0) / dt + (1 - d_alpha) * d1
    a2 = smoothing_factor(min_cutoff + beta * d2, dt)
    assert filter_.filter(3.0, 2 * dt) == pytest.approx(a2 * 3.0 + (1 - a2) * x1)


def test_constant_signal_is_unchanged():
    outputs = run(OneEuroFilter(1.0, 5.0), [(i / 30, 0.5) for i in range(30)])
    assert outputs == pytest.approx([0.5] * 30)


def test_jitter_is_damped_at_rest():
    rng = random.Random(7)
    noisy = [0.5 + rng.gauss(0, 0.002) for _ in range(300)]
    outputs = run(OneEuroFilter(1.0, 10.0), [(i / 30, v) for i, v in enumerate(noisy)])
    assert statistics.pstdev(outputs[30:]) < statistics.pstdev(noisy[30:]) / 3


def test_fast_motion_tracks_with_little_lag():
    ramp = [(i / 30, i / 30 * 2.0) for i in range(30)]  # 2 units per second.
    adaptive = run(OneEuroFilter(1.0, 10.0), ramp)[-1]
    fixed = run(OneEuroFilter(1.0, 0.0), ramp)[-1]
    true_value = ramp[-1][1]
    assert true_value - adaptive < 0.05
    assert true_value - fixed > 5 * (true_value - adaptive)


def test_larger_dt_moves_further_toward_a_step():
    # Same step, longer gap: a fixed-FPS filter would treat both identically.
    short = OneEuroFilter(1.0, 0.0)
    long = OneEuroFilter(1.0, 0.0)
    short.filter(0.0, 0.0)
    long.filter(0.0, 0.0)
    assert long.filter(1.0, 0.2) > short.filter(1.0, 0.01)


def test_variable_dt_gives_same_lag_as_regular_sampling():
    rng = random.Random(3)
    times = [0.0]
    while times[-1] < 2.0:
        times.append(times[-1] + rng.uniform(0.008, 0.05))
    regular_times = [i / 60 for i in range(121)]

    irregular = run(OneEuroFilter(1.0, 10.0), [(t, t * 0.8) for t in times])
    regular = run(OneEuroFilter(1.0, 10.0), [(t, t * 0.8) for t in regular_times])
    irregular_lag = times[-1] * 0.8 - irregular[-1]
    regular_lag = regular_times[-1] * 0.8 - regular[-1]
    assert irregular_lag == pytest.approx(regular_lag, rel=0.25)


@pytest.mark.parametrize("second_timestamp", [1.0, 0.5])
def test_non_increasing_timestamp_rejected_without_state_change(second_timestamp):
    filter_ = OneEuroFilter()
    filter_.filter(1.0, 1.0)
    with pytest.raises(ValueError):
        filter_.filter(2.0, second_timestamp)
    reference = OneEuroFilter()
    reference.filter(1.0, 1.0)
    assert filter_.filter(2.0, 1.1) == reference.filter(2.0, 1.1)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_sample_rejected(bad):
    filter_ = OneEuroFilter()
    with pytest.raises(ValueError):
        filter_.filter(bad, 0.0)


@pytest.mark.parametrize(
    "kwargs", [{"min_cutoff": 0}, {"d_cutoff": -1}, {"beta": -0.1}]
)
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        OneEuroFilter(**kwargs)


def test_reset_restarts_from_next_sample():
    filter_ = OneEuroFilter()
    filter_.filter(0.0, 0.0)
    filter_.filter(1.0, 0.1)
    filter_.reset()
    assert filter_.filter(5.0, 0.05) == 5.0


def test_2d_matches_two_independent_1d_filters():
    filter_2d = OneEuroFilter2D(1.0, 2.0)
    fx, fy = OneEuroFilter(1.0, 2.0), OneEuroFilter(1.0, 2.0)
    for i in range(20):
        t, x, y = i * 0.03, math.sin(i * 0.3), i * 0.01
        assert filter_2d.filter(x, y, t) == pytest.approx((fx.filter(x, t), fy.filter(y, t)))


def test_2d_bad_y_leaves_x_axis_untouched():
    filter_2d = OneEuroFilter2D()
    filter_2d.filter(0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        filter_2d.filter(1.0, float("nan"), 0.1)
    reference = OneEuroFilter2D()
    reference.filter(0.0, 0.0, 0.0)
    assert filter_2d.filter(1.0, 1.0, 0.1) == reference.filter(1.0, 1.0, 0.1)


def test_2d_reset():
    filter_2d = OneEuroFilter2D()
    filter_2d.filter(0.0, 0.0, 0.0)
    filter_2d.reset()
    assert filter_2d.filter(3.0, 4.0, 0.0) == (3.0, 4.0)
