"""Chidori: jagged lightning arcs crackling around a fist."""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.fantasy.effects.common import add_light, clipped_region, darken, scaled, soften, tint

MIN_ARCS, MAX_ARCS = 4, 6
MIN_SEGMENTS, MAX_SEGMENTS = 6, 9
MIN_HOLD_FRAMES, MAX_HOLD_FRAMES = 2, 3  # How long one set of arcs stays on screen.
# Arc geometry is in fist radii, so that arcs follow the hand between refreshes.
ARC_START = (0.55, 0.95)
ARC_STEP = (0.18, 0.42)
ARC_WANDER = 0.75  # Standard deviation of each segment's turn, in radians.
ARC_REACH = 2.4

CORE_COLOR = (255, 245, 235)  # Colors are BGR.
GLOW_COLOR = (600, 330, 110)
HALO_COLOR = (255, 150, 50)


class ChidoriEffect:
    """Draws onto the frame it is given. Same generator and calls, same pixels."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self.reset()

    def reset(self) -> None:
        self._arcs: list[np.ndarray] = []
        self._hold_frames = 0

    def render(
        self,
        image: np.ndarray,
        center: tuple[int, int],
        radius: float,
        level: float,
        timestamp: float,
    ) -> None:
        """Draw one frame. `level` runs from 0, invisible, to 1, full strength."""
        if level <= 0:
            return
        if self._hold_frames == 0:
            self._arcs = self._new_arcs()
            self._hold_frames = int(self._rng.integers(MIN_HOLD_FRAMES, MAX_HOLD_FRAMES + 1))
        self._hold_frames -= 1

        reach = radius * (0.6 + 0.4 * level)
        region = clipped_region(image, center, math.ceil(reach * (ARC_REACH + 0.5)))
        if region is None:
            return
        view = region.view
        origin = np.array((region.x, region.y))
        local_center = (center[0] - region.x, center[1] - region.y)

        bolts = np.zeros(view.shape[:2], np.uint8)
        for arc in self._arcs:
            points = np.rint(arc * reach + center - origin).astype(np.int32)
            cv2.polylines(bolts, [points], False, 255, 2, cv2.LINE_AA)

        halo = np.zeros(view.shape[:2], np.uint8)
        cv2.circle(halo, local_center, round(reach), 130, -1, cv2.LINE_AA)
        halo = soften(halo, reach)

        flicker = 0.85 + 0.15 * math.sin(timestamp * 47.0)
        strength = level * flicker
        glow = soften(bolts, reach * 0.3)
        darken(view, cv2.convertScaleAbs(cv2.max(glow, halo), alpha=strength * 0.8))
        add_light(view, tint(halo, scaled(HALO_COLOR, strength)))
        add_light(view, tint(glow, scaled(GLOW_COLOR, strength)))
        add_light(view, tint(bolts, scaled(CORE_COLOR, strength)))

    def _new_arcs(self) -> list[np.ndarray]:
        count = int(self._rng.integers(MIN_ARCS, MAX_ARCS + 1))
        return [self._new_arc() for _ in range(count)]

    def _new_arc(self) -> np.ndarray:
        rng = self._rng
        heading = rng.uniform(0.0, math.tau)
        distance = rng.uniform(*ARC_START)
        point = np.array((math.cos(heading), math.sin(heading))) * distance
        points = [point]
        for _ in range(int(rng.integers(MIN_SEGMENTS, MAX_SEGMENTS + 1))):
            heading += rng.normal(0.0, ARC_WANDER)
            step = rng.uniform(*ARC_STEP)
            point = point + np.array((math.cos(heading), math.sin(heading))) * step
            # Kept in reach, because an arc cut off by the region's edge looks wrong.
            point = point * min(ARC_REACH / max(float(np.hypot(*point)), 1e-9), 1.0)
            points.append(point)
        return np.array(points)
