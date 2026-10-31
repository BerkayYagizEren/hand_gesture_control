"""FANTASY mode: draws effects over the camera image and plays their sounds, nothing else.

This package never imports AppKit, Quartz or the controllers, and has no way to send
a mouse, keyboard or media event. Its one route out is the `afplay` process in `sound`.
"""

from __future__ import annotations

import math

import numpy as np

from src.fantasy.effects.chidori import ChidoriEffect
from src.fantasy.effects.rasengan import RasenganEffect
from src.fantasy.engine import EffectKind, FantasyConfig, FantasyEngine, FantasyPhase
from src.fantasy.sound import BaseSoundPlayer, MockSoundPlayer
from src.vision.classifier import Pose, PoseClassifier, TemporalVotingFilter
from src.vision.detector import DetectionResult, Hand, HandLandmark, palm_width
from src.vision.geometry import palm_normal

DEFAULT_SEED = 0
TRIGGERS = {Pose.FIST: EffectKind.CHIDORI, Pose.OPEN_PALM: EffectKind.RASENGAN}
# Palm normal's component toward the camera; 0.5 is 60 degrees off. Not `palm_facing`, whose
# nearest axis is "DOWN" for a hand raised to a webcam: (-0.16, 0.73, -0.66) on a real palm.
MIN_FACING_COMPONENT = 0.5
# Effect size in palm widths; the floor keeps a distant hand's effect visible.
EFFECT_RADII = {EffectKind.CHIDORI: 0.8, EffectKind.RASENGAN: 0.7}
MIN_RADIUS = 12.0
_FIST_LANDMARKS = (
    HandLandmark.WRIST,
    HandLandmark.INDEX_FINGER_MCP,
    HandLandmark.MIDDLE_FINGER_MCP,
    HandLandmark.RING_FINGER_MCP,
    HandLandmark.PINKY_MCP,
)
_PALM_LANDMARKS = (HandLandmark.WRIST, HandLandmark.MIDDLE_FINGER_MCP)
ANCHORS = {EffectKind.CHIDORI: _FIST_LANDMARKS, EffectKind.RASENGAN: _PALM_LANDMARKS}


class FantasyMode:
    """Has the same `update`, `reset`, `pose` and `state_name` as `SwipeActionMode`."""

    def __init__(
        self,
        classifier: PoseClassifier,
        seed: int = DEFAULT_SEED,
        config: FantasyConfig = FantasyConfig(),
        sound_player: BaseSoundPlayer | None = None,
    ) -> None:
        self._classifier = classifier
        self._engine = FantasyEngine(config)
        self._sound_player = sound_player if sound_player is not None else MockSoundPlayer()
        self._track: str | None = None  # What this tick's phase wants heard.
        self._voting = TemporalVotingFilter()
        rng = np.random.default_rng(seed)
        self._effects = {
            EffectKind.CHIDORI: ChidoriEffect(rng),
            EffectKind.RASENGAN: RasenganEffect(rng),
        }
        self._hand: Hand | None = None  # Last seen, so an effect outlives a short dropout.

    @property
    def pose(self) -> Pose:
        return self._voting.pose

    @property
    def state_name(self) -> str:
        return self._engine.phase.name

    @property
    def engine(self) -> FantasyEngine:
        return self._engine

    @property
    def sound_player(self) -> BaseSoundPlayer:
        return self._sound_player

    def reset(self) -> None:
        self._sound_player.stop()
        self._track = None
        self._engine.reset()
        self._voting.reset()
        self._hand = None
        for effect in self._effects.values():
            effect.reset()

    def update(self, result: DetectionResult | None, timestamp: float) -> str | None:
        """Advance one tick and return the name of the effect it activated, if any."""
        if self._engine.expire_hand(timestamp):
            self._voting.reset()
        trigger = None
        hand_present = result is not None and bool(result.hands)
        if hand_present:
            self._hand = result.hands[0]
            trigger = TRIGGERS.get(self._voting.update(self._observed_pose(self._hand)))
        activated = self._engine.update(trigger, hand_present, timestamp)
        self._sync_sound()
        return activated.value if activated is not None else None

    def render(self, image: np.ndarray, timestamp: float) -> None:
        """Draw the current effect, if there is one, onto `image` in place."""
        effect, level = self._engine.effect, self._engine.level
        if effect is None or level <= 0 or self._hand is None:
            return
        anchor = _effect_anchor(self._hand, effect, image)
        if anchor is not None:
            self._effects[effect].render(image, anchor[0], anchor[1], level, timestamp)

    def _sync_sound(self) -> None:
        phase, effect = self._engine.phase, self._engine.effect
        # Only the Rasengan charges audibly. A clip that ends while the effect holds starts over.
        audible = phase is FantasyPhase.ACTIVE or (
            phase is FantasyPhase.CHARGING and effect is EffectKind.RASENGAN
        )
        track = effect.value if audible else None
        if track != self._track and self._sound_player.is_playing:
            self._sound_player.stop()
        self._track = track
        if track is not None and not self._sound_player.is_playing:
            self._sound_player.play(track)

    def _observed_pose(self, hand: Hand) -> Pose:
        # Goes through the vote with the pose, so one bad normal cannot end a Rasengan.
        pose = self._classifier.classify(hand).pose
        if (
            pose is Pose.OPEN_PALM
            and -palm_normal(hand.world_landmarks, hand.handedness).z < MIN_FACING_COMPONENT
        ):
            return Pose.UNKNOWN
        return pose


def _effect_anchor(
    hand: Hand, effect: EffectKind, image: np.ndarray
) -> tuple[tuple[int, int], float] | None:
    """Where to draw `effect` in pixels, as (center, radius); None on bad landmarks."""
    height, width = image.shape[:2]
    points = [hand.landmarks[landmark] for landmark in ANCHORS[effect]]
    x = sum(point.x for point in points) / len(points) * width
    y = sum(point.y for point in points) / len(points) * height
    palm = palm_width(hand.landmarks, width / height) * height
    if not all(math.isfinite(value) for value in (x, y, palm)):
        return None
    return (round(x), round(y)), max(palm * EFFECT_RADII[effect], MIN_RADIUS)
