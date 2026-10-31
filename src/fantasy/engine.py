"""Decides which fantasy effect is showing, from triggers and time alone.

IDLE -> CHARGING -> (held) -> ACTIVE -> (let go or hand lost) -> COOLDOWN -> IDLE

Only one effect exists at a time, because `effect` is a single field. Every exit
from ACTIVE passes through COOLDOWN, which is also what spaces out two effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class FantasyPhase(Enum):
    IDLE = auto()
    CHARGING = auto()
    ACTIVE = auto()
    COOLDOWN = auto()


class EffectKind(Enum):
    CHIDORI = "chidori"
    RASENGAN = "rasengan"


@dataclass(frozen=True)
class FantasyConfig:
    hold_seconds: float = 0.25
    hand_loss_seconds: float = 0.3  # The hand may be gone this long; only a longer gap counts.
    cooldown_seconds: float = 1.0
    release_seconds: float = 0.25  # The start of the cooldown, in which the effect dies away.

    def __post_init__(self) -> None:
        if min(self.hold_seconds, self.hand_loss_seconds, self.release_seconds) <= 0:
            raise ValueError("hold_seconds, hand_loss_seconds and release_seconds must be positive")
        if self.cooldown_seconds < self.release_seconds:
            raise ValueError("cooldown_seconds must not be shorter than release_seconds")


class FantasyEngine:
    """Call `update` every tick, with or without a hand. Pure, so no drawing and no clock."""

    def __init__(self, config: FantasyConfig = FantasyConfig()) -> None:
        self.config = config
        self.reset()

    @property
    def phase(self) -> FantasyPhase:
        return self._phase

    @property
    def effect(self) -> EffectKind | None:
        """The effect being charged, shown or released."""
        return self._effect

    @property
    def level(self) -> float:
        """How strongly to draw `effect`, from 0 to 1, as of the last `update`."""
        return self._level

    def reset(self) -> None:
        self._phase = FantasyPhase.IDLE
        self._effect: EffectKind | None = None
        self._level = 0.0
        self._phase_start = 0.0
        self._last_hand_time: float | None = None

    def expire_hand(self, timestamp: float) -> bool:
        """Apply the hand loss timeout, and return whether the hand just counted as lost."""
        # Checked on hand frames too: a stalled camera shows up as a long gap between them.
        if (
            self._last_hand_time is None
            or timestamp - self._last_hand_time <= self.config.hand_loss_seconds
        ):
            return False
        self._last_hand_time = None
        if self._phase is FantasyPhase.ACTIVE:
            self._enter(FantasyPhase.COOLDOWN, self._effect, timestamp)
        elif self._phase is FantasyPhase.CHARGING:
            self._enter(FantasyPhase.IDLE, None, timestamp)
        return True

    def update(
        self, trigger: EffectKind | None, hand_present: bool, timestamp: float
    ) -> EffectKind | None:
        """Advance one tick and return the effect that just became active, if any.

        `trigger` is the effect the hand is asking for, and is ignored without a hand.
        """
        self.expire_hand(timestamp)
        if (
            self._phase is FantasyPhase.COOLDOWN
            and timestamp - self._phase_start >= self.config.cooldown_seconds
        ):
            self._enter(FantasyPhase.IDLE, None, timestamp)

        activated = None
        if hand_present:
            self._last_hand_time = timestamp
            activated = self._follow_trigger(trigger, timestamp)
        self._level = self._current_level(timestamp)
        return activated

    def _follow_trigger(self, trigger: EffectKind | None, timestamp: float) -> EffectKind | None:
        if self._phase is FantasyPhase.COOLDOWN or (
            self._phase is FantasyPhase.IDLE and trigger is None
        ):
            return None
        if self._phase is FantasyPhase.ACTIVE:
            if trigger is not self._effect:
                self._enter(FantasyPhase.COOLDOWN, self._effect, timestamp)
            return None
        if trigger is None:
            self._enter(FantasyPhase.IDLE, None, timestamp)
        elif self._phase is FantasyPhase.IDLE or trigger is not self._effect:
            self._enter(FantasyPhase.CHARGING, trigger, timestamp)
        elif timestamp - self._phase_start >= self.config.hold_seconds:
            self._enter(FantasyPhase.ACTIVE, trigger, timestamp)
            return trigger
        return None

    def _current_level(self, timestamp: float) -> float:
        elapsed = timestamp - self._phase_start
        if self._phase is FantasyPhase.CHARGING:
            return min(elapsed / self.config.hold_seconds, 1.0)
        if self._phase is FantasyPhase.ACTIVE:
            return 1.0
        if self._phase is FantasyPhase.COOLDOWN:
            return max(1.0 - elapsed / self.config.release_seconds, 0.0)
        return 0.0

    def _enter(self, phase: FantasyPhase, effect: EffectKind | None, timestamp: float) -> None:
        self._phase = phase
        self._effect = effect
        self._phase_start = timestamp
