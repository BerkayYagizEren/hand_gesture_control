"""Replays a `LandmarkRecorder` JSONL file as if it were live detector output."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from src.vision.detector import DetectionResult, Hand, Landmark
from src.vision.recorder import FORMAT_VERSION

logger = logging.getLogger(__name__)


class ReplayFormatError(ValueError):
    """The recording is corrupt or has an unsupported format."""


class LandmarkReplayer:
    """Iterable of `DetectionResult`s; each iteration re-reads from the start.

    Yields as fast as it is consumed: downstream timing must come from
    `DetectionResult.timestamp_ms`, which keeps replay deterministic.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.metadata: dict[str, Any] = self._read_header()

    def __iter__(self) -> Iterator[DetectionResult]:
        last_timestamp = None
        records = self._records()
        next(records)  # Header, validated in __init__.
        for line_number, record in records:
            if record.get("type") != "frame":
                raise self._error(line_number, f"unexpected record type {record.get('type')!r}")
            result = self._parse_frame(line_number, record)
            if last_timestamp is not None and result.timestamp_ms <= last_timestamp:
                raise self._error(
                    line_number,
                    f"timestamp {result.timestamp_ms} does not increase past {last_timestamp}",
                )
            last_timestamp = result.timestamp_ms
            yield result

    def _read_header(self) -> dict[str, Any]:
        records = self._records()
        try:
            line_number, header = next(records)
        except StopIteration:
            raise ReplayFormatError(f"{self.path}: recording is empty") from None
        finally:
            records.close()
        if header.get("type") != "header":
            raise self._error(line_number, "missing header record")
        if header.get("version") != FORMAT_VERSION:
            raise self._error(line_number, f"unsupported version {header.get('version')!r}")
        return header.get("metadata", {})

    def _records(self) -> Iterator[tuple[int, dict[str, Any]]]:
        with open(self.path, encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    if not line.endswith("\n"):
                        # A recording cut short by a crash is still usable.
                        logger.warning("%s: ignoring truncated last line", self.path)
                        return
                    raise self._error(line_number, f"invalid JSON: {exc}") from exc
                if not isinstance(record, dict):
                    raise self._error(line_number, "record is not an object")
                yield line_number, record

    def _parse_frame(self, line_number: int, record: dict[str, Any]) -> DetectionResult:
        try:
            width, height = record["image_size"]
            return DetectionResult(
                timestamp_ms=record["timestamp_ms"],
                image_width=width,
                image_height=height,
                hands=tuple(
                    Hand(
                        landmarks=tuple(Landmark(*map(float, p)) for p in hand["landmarks"]),
                        world_landmarks=tuple(
                            Landmark(*map(float, p)) for p in hand["world_landmarks"]
                        ),
                        handedness=hand["handedness"],
                        handedness_score=hand["score"],
                    )
                    for hand in record["hands"]
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise self._error(line_number, f"malformed frame: {exc!r}") from exc

    def _error(self, line_number: int, message: str) -> ReplayFormatError:
        return ReplayFormatError(f"{self.path}:{line_number}: {message}")
