import pytest

from src.fantasy.engine import EffectKind, FantasyConfig, FantasyEngine, FantasyPhase

CHIDORI, RASENGAN = EffectKind.CHIDORI, EffectKind.RASENGAN
DT = 0.033


def drive(engine, trigger, seconds, start, hand=True):
    """Tick at about 30 FPS for `seconds` after `start`, and return the time reached."""
    steps = round(seconds / DT)
    for step in range(1, steps + 1):
        engine.update(trigger, hand, start + step * DT)
    return start + steps * DT


def active(effect=CHIDORI):
    engine = FantasyEngine()
    engine.update(effect, True, 0.0)
    engine.update(effect, True, 0.25)
    assert engine.phase is FantasyPhase.ACTIVE
    return engine


def in_cooldown(start=0.25):
    engine = active()
    engine.update(None, True, start + 0.1)
    assert engine.phase is FantasyPhase.COOLDOWN
    return engine, start + 0.1


def test_starts_idle_with_nothing_to_draw():
    engine = FantasyEngine()
    assert (engine.phase, engine.effect, engine.level) == (FantasyPhase.IDLE, None, 0.0)


def test_no_trigger_stays_idle():
    engine = FantasyEngine()
    drive(engine, None, 2.0, 0.0)
    assert engine.phase is FantasyPhase.IDLE


@pytest.mark.parametrize("effect", list(EffectKind))
def test_hold_activates_once_at_the_hold_time_and_not_before(effect):
    engine = FantasyEngine()
    assert engine.update(effect, True, 0.0) is None
    assert engine.phase is FantasyPhase.CHARGING and engine.effect is effect
    assert engine.update(effect, True, 0.249) is None
    assert engine.phase is FantasyPhase.CHARGING
    assert engine.update(effect, True, 0.25) is effect
    assert engine.phase is FantasyPhase.ACTIVE
    assert engine.update(effect, True, 0.3) is None


def test_default_hold_is_inside_the_agreed_range():
    assert 0.25 <= FantasyConfig().hold_seconds <= 0.3


def test_level_rises_while_charging_and_is_full_when_active():
    engine = FantasyEngine()
    engine.update(CHIDORI, True, 0.0)
    assert engine.level == 0.0
    engine.update(CHIDORI, True, 0.125)
    assert engine.level == pytest.approx(0.5)
    engine.update(CHIDORI, True, 0.25)
    assert engine.level == 1.0


def test_leaving_the_pose_while_charging_cancels_without_cooldown():
    engine = FantasyEngine()
    engine.update(CHIDORI, True, 0.0)
    engine.update(None, True, 0.1)
    assert (engine.phase, engine.effect) == (FantasyPhase.IDLE, None)
    engine.update(CHIDORI, True, 0.2)
    assert engine.phase is FantasyPhase.CHARGING


def test_changing_pose_while_charging_restarts_the_hold():
    engine = FantasyEngine()
    engine.update(CHIDORI, True, 0.0)
    engine.update(RASENGAN, True, 0.2)
    assert engine.effect is RASENGAN
    assert engine.update(RASENGAN, True, 0.3) is None  # Only 0.1 s into the new hold.
    assert engine.update(RASENGAN, True, 0.45) is RASENGAN


def test_active_effect_lasts_as_long_as_the_pose():
    engine = active()
    drive(engine, CHIDORI, 30.0, 0.25)
    assert engine.phase is FantasyPhase.ACTIVE and engine.level == 1.0


def test_letting_go_starts_the_cooldown_and_the_effect_dies_away():
    engine = active()
    engine.update(None, True, 1.0)
    assert (engine.phase, engine.effect, engine.level) == (FantasyPhase.COOLDOWN, CHIDORI, 1.0)
    engine.update(None, True, 1.125)
    assert engine.level == pytest.approx(0.5)
    engine.update(None, True, 1.25)
    assert engine.level == 0.0 and engine.phase is FantasyPhase.COOLDOWN


def test_cooldown_lasts_a_second_and_blocks_every_trigger():
    engine, start = in_cooldown()
    assert engine.update(RASENGAN, True, start + 0.999) is None
    assert engine.phase is FantasyPhase.COOLDOWN and engine.effect is CHIDORI
    engine.update(RASENGAN, True, start + 1.0)
    assert engine.phase is FantasyPhase.CHARGING and engine.effect is RASENGAN


def test_switching_effects_goes_through_the_cooldown():
    engine = active(CHIDORI)
    engine.update(RASENGAN, True, 1.0)
    assert engine.phase is FantasyPhase.COOLDOWN and engine.effect is CHIDORI
    t = drive(engine, RASENGAN, 0.95, 1.0)
    assert engine.phase is FantasyPhase.COOLDOWN
    activated = [engine.update(RASENGAN, True, t + (i + 1) * DT) for i in range(12)]
    assert activated.count(RASENGAN) == 1
    assert engine.phase is FantasyPhase.ACTIVE and engine.effect is RASENGAN


def test_cooldown_ends_without_a_hand():
    engine, start = in_cooldown()
    engine.update(None, False, start + 1.0)
    assert engine.phase is FantasyPhase.IDLE and engine.effect is None


@pytest.mark.parametrize("gap", [0.1, 0.299, 0.2999])
def test_active_survives_hand_loss_up_to_300_ms(gap):
    engine = active()
    engine.update(None, False, 0.25 + gap)
    assert engine.phase is FantasyPhase.ACTIVE and engine.level == 1.0
    engine.update(CHIDORI, True, 0.25 + gap)
    assert engine.phase is FantasyPhase.ACTIVE


def test_active_goes_out_past_300_ms_of_hand_loss():
    engine = active()
    engine.update(None, False, 0.25 + 0.301)
    assert engine.phase is FantasyPhase.COOLDOWN


def test_dropout_timer_restarts_each_time_the_hand_returns():
    engine = active()
    t = 0.25
    for _ in range(5):
        t += 0.25
        engine.update(None, False, t)
        t += 0.01
        engine.update(CHIDORI, True, t)
    assert engine.phase is FantasyPhase.ACTIVE


def test_camera_stall_counts_as_hand_loss():
    # No empty frames in between: the gap only shows on the next frame with a hand.
    engine = active()
    engine.update(CHIDORI, True, 0.25 + 0.5)
    assert engine.phase is FantasyPhase.COOLDOWN


def test_charging_survives_a_short_dropout():
    engine = FantasyEngine()
    engine.update(CHIDORI, True, 0.0)
    engine.update(None, False, 0.1)
    assert engine.phase is FantasyPhase.CHARGING
    assert engine.update(CHIDORI, True, 0.26) is CHIDORI


def test_no_hand_cannot_complete_a_hold():
    engine = FantasyEngine()
    engine.update(CHIDORI, True, 0.0)
    assert engine.update(None, False, 0.28) is None
    assert engine.phase is FantasyPhase.CHARGING


def test_charging_is_dropped_past_300_ms_of_hand_loss_without_cooldown():
    engine = FantasyEngine()
    engine.update(CHIDORI, True, 0.0)
    engine.update(None, False, 0.301)
    assert (engine.phase, engine.effect) == (FantasyPhase.IDLE, None)


def test_trigger_is_ignored_without_a_hand():
    engine = FantasyEngine()
    engine.update(CHIDORI, False, 0.0)
    assert engine.phase is FantasyPhase.IDLE


def test_expire_hand_reports_a_loss_only_once():
    engine = active()
    assert not engine.expire_hand(0.5)
    assert engine.expire_hand(0.6)
    assert not engine.expire_hand(0.7)


def test_reset_forgets_everything_including_the_cooldown():
    engine, start = in_cooldown()
    engine.reset()
    assert (engine.phase, engine.effect, engine.level) == (FantasyPhase.IDLE, None, 0.0)
    engine.update(CHIDORI, True, start + 0.01)
    assert engine.phase is FantasyPhase.CHARGING


def test_same_inputs_give_the_same_sequence():
    def run():
        engine, seen = FantasyEngine(), []
        for i in range(200):
            trigger = CHIDORI if i % 90 < 40 else (RASENGAN if i % 90 < 70 else None)
            engine.update(trigger, i % 37 != 0, i * DT)
            seen.append((engine.phase, engine.effect, engine.level))
        return seen

    assert run() == run()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hold_seconds": 0},
        {"hand_loss_seconds": 0},
        {"release_seconds": 0},
        {"cooldown_seconds": 0.1, "release_seconds": 0.2},
    ],
)
def test_bad_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        FantasyConfig(**kwargs)
