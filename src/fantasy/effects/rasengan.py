"""Rasengan: a spinning sphere of chakra that charges up in the palm and collapses into it."""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.fantasy.effects.common import add_light, clipped_region, darken, scaled, soften, tint

# From the rim inward, as (fraction of the radius, BGR color); blurred into a gradient.
BODY_STEPS = (
    (1.0, (170, 60, 0)),
    (0.8, (235, 120, 10)),
    (0.58, (255, 190, 60)),
    (0.36, (255, 235, 180)),
    (0.2, (255, 255, 255)),
)
# Spin layers as (fraction of the radius, degrees per second). They all differ, so the
# sphere shears against itself instead of turning as one disc.
SPIN_LAYERS = ((0.9, 410.0), (0.68, -620.0), (0.46, 880.0))
SPIN_ARCS = 3
SPIN_SWEEP = 75.0
SPIN_FLATTEN = 0.72  # Minor axis over major, which gives the rings their sphere-like tilt.
SPIN_COLOR = (255, 245, 215)
GLOW_COLOR = (255, 170, 40)

PARTICLES = 36
PARTICLE_SPAWN = (1.25, 1.75)  # In sphere radii.
PARTICLE_ABSORBED = 0.45
PARTICLE_INFLOW = 1.1  # Radii per second.
PARTICLE_SWIRL = 3.5  # Radians per second.
PARTICLE_COLOR = (255, 225, 150)
MAX_STEP_SECONDS = 0.1  # So that a stalled camera does not teleport the particles.
REGION_RADII = 2.2


class RasenganEffect:
    """Draws onto the frame it is given. Same generator and calls, same pixels."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self.reset()

    def reset(self) -> None:
        self._distances = self._rng.uniform(PARTICLE_ABSORBED, PARTICLE_SPAWN[1], PARTICLES)
        self._angles = self._rng.uniform(0.0, math.tau, PARTICLES)
        self._last_time: float | None = None

    def render(
        self,
        image: np.ndarray,
        center: tuple[int, int],
        radius: float,
        level: float,
        timestamp: float,
    ) -> None:
        """Draw one frame. The sphere's size follows `level`, from 0, nothing, to 1, full size."""
        if level <= 0:
            return
        self._move_particles(timestamp)
        size = max(radius * level, 1.0)
        region = clipped_region(image, center, math.ceil(size * REGION_RADII))
        if region is None:
            return
        view = region.view
        local = (center[0] - region.x, center[1] - region.y)
        shape = view.shape[:2]

        body = np.zeros((*shape, 3), np.uint8)
        for fraction, color in BODY_STEPS:
            cv2.circle(body, local, max(round(size * fraction), 1), color, -1, cv2.LINE_AA)
        body = soften(body, size * 0.3)

        shadow = np.zeros(shape, np.uint8)
        cv2.circle(shadow, local, round(size * 1.05), 215, -1, cv2.LINE_AA)
        darken(view, soften(shadow, size * 0.3))

        add_light(view, tint(soften(shadow, size * 0.9), scaled(GLOW_COLOR, 1.0)))
        add_light(view, body)
        add_light(view, self._spin_layer(shape, local, size, timestamp))
        add_light(view, self._particle_layer(shape, local, size))

    def _spin_layer(
        self, shape: tuple[int, int], center: tuple[int, int], size: float, timestamp: float
    ) -> np.ndarray:
        mask = np.zeros(shape, np.uint8)
        for fraction, speed in SPIN_LAYERS:
            axes = (max(round(size * fraction), 1), max(round(size * fraction * SPIN_FLATTEN), 1))
            tilt = (speed * timestamp * 0.2) % 360.0
            spin = (speed * timestamp) % 360.0
            for arc in range(SPIN_ARCS):
                start = spin + arc * 360.0 / SPIN_ARCS
                cv2.ellipse(mask, center, axes, tilt, start, start + SPIN_SWEEP, 255, 2, cv2.LINE_AA)
        return tint(soften(mask, 3), scaled(SPIN_COLOR, 0.8))

    def _particle_layer(
        self, shape: tuple[int, int], center: tuple[int, int], size: float
    ) -> np.ndarray:
        mask = np.zeros(shape, np.uint8)
        xs = center[0] + np.cos(self._angles) * self._distances * size
        ys = center[1] + np.sin(self._angles) * self._distances * size
        for x, y in zip(np.rint(xs).astype(int), np.rint(ys).astype(int), strict=True):
            cv2.circle(mask, (int(x), int(y)), 2, 255, -1, cv2.LINE_AA)
        return tint(mask, PARTICLE_COLOR)

    def _move_particles(self, timestamp: float) -> None:
        last, self._last_time = self._last_time, timestamp
        if last is None:
            return
        step = min(max(timestamp - last, 0.0), MAX_STEP_SECONDS)
        self._distances -= PARTICLE_INFLOW * step
        self._angles += PARTICLE_SWIRL * step
        absorbed = self._distances < PARTICLE_ABSORBED
        count = int(np.count_nonzero(absorbed))
        if count:
            self._distances[absorbed] = self._rng.uniform(*PARTICLE_SPAWN, count)
            self._angles[absorbed] = self._rng.uniform(0.0, math.tau, count)
