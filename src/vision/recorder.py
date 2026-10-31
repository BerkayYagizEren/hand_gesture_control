"""Records detector output to JSONL for deterministic, camera-free replay.

Format: a header line followed by one line per frame.

    {"type": "header", "version": 1, "metadata": {...}}
    {"type": "frame", "timestamp_ms": 12, "image_size": [640, 480], "hands": [
        {"handedness": "Right", "score": 0.98,
         "landmarks": [[x, y, z], ...], "world_landmarks": [[x, y, z], ...]}]}

(Each record is on a single line.) Floats use JSON's shortest round-trip form,
so replayed values are bit-identical to the recorded ones.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.vision.detector import DetectionResult

FORMAT_VERSION = 1


class LandmarkRecorder:
    """Appends `DetectionResult`s to a new JSONL file.

    Line-buffered, so a crash loses at most the line being written. Refuses to
    replace an existing recording unless `overwrite=True`.
    """

    def __init__(
        self,
        path: str | Path,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frame_count = 0
        self._file = open(self.path, "w" if overwrite else "x", encoding="utf-8", buffering=1)
        self._write(
            {"type": "header", "version": FORMAT_VERSION, "metadata": dict(metadata or {})}
        )

    def record(self, result: DetectionResult) -> None:
        if self._file.closed:
            raise RuntimeError("recorder is closed")
        self._write(
            {
                "type": "frame",
                "timestamp_ms": result.timestamp_ms,
                "image_size": [result.image_width, result.image_height],
                "hands": [
                    {
                        "handedness": hand.handedness,
                        "score": hand.handedness_score,
                        "landmarks": hand.landmarks,
                        "world_landmarks": hand.world_landmarks,
                    }
                    for hand in result.hands
                ],
            }
        )
        self.frame_count += 1

    def close(self) -> None:
        """Close the file. Safe to call repeatedly."""
        self._file.close()

    def __enter__(self) -> LandmarkRecorder:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
