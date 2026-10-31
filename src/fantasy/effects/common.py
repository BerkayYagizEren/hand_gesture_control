"""Compositing shared by the effects: uint8 layers, confined to the effect's region of interest."""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np


WIDE_BLUR = 16


class Region(NamedTuple):
    """The part of an image an effect may touch. `view` writes through to the image."""

    view: np.ndarray
    x: int  # Image position of the view's top left corner.
    y: int


def clipped_region(image: np.ndarray, center: tuple[int, int], half_size: int) -> Region | None:
    """The square around `center`, cut to the image; None if nothing of it is inside."""
    height, width = image.shape[:2]
    left, right = max(center[0] - half_size, 0), min(center[0] + half_size + 1, width)
    top, bottom = max(center[1] - half_size, 0), min(center[1] + half_size + 1, height)
    if left >= right or top >= bottom:
        return None
    return Region(image[top:bottom, left:right], left, top)


def soften(layer: np.ndarray, size: float) -> np.ndarray:
    """Blur over about `size` pixels. Wide blurs run at quarter size, which looks the same."""
    if size <= WIDE_BLUR or min(layer.shape[:2]) < WIDE_BLUR:
        return cv2.GaussianBlur(layer, _kernel(size), 0)
    small = cv2.resize(layer, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, _kernel(size / 4), 0)
    return cv2.resize(small, layer.shape[1::-1], interpolation=cv2.INTER_LINEAR)


def _kernel(size: float) -> tuple[int, int]:
    side = max(round(size) | 1, 3)  # GaussianBlur wants an odd kernel.
    return side, side


def tint(mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    """Turn a one channel mask into BGR. Channels above 255 overdrive and saturate."""
    return cv2.merge([cv2.convertScaleAbs(mask, alpha=channel / 255) for channel in color])


def scaled(color: tuple[float, float, float], factor: float) -> tuple[float, float, float]:
    return (color[0] * factor, color[1] * factor, color[2] * factor)


def darken(view: np.ndarray, mask: np.ndarray) -> None:
    """Darken under an effect in place, so its colors survive on bright skin."""
    cv2.subtract(view, cv2.merge((mask, mask, mask)), dst=view)


def add_light(view: np.ndarray, layer: np.ndarray) -> None:
    """Add a BGR layer in place, saturating rather than wrapping around."""
    cv2.add(view, layer, dst=view)
