"""Sound for FANTASY mode. A failure here costs the sound and never the video loop."""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)

SOUNDS_DIR = Path(__file__).resolve().parents[2] / "assets" / "sounds"
# afplay exits 0 at once on a file it cannot decode, which would otherwise respawn every frame.
MIN_PLAY_SECONDS = 0.2


class BaseSoundPlayer(ABC):
    """Plays one named track at a time. No method may block the caller."""

    @abstractmethod
    def play(self, name: str) -> None:
        """Start `name`, replacing any other track. Does nothing if it is already playing."""

    @abstractmethod
    def stop(self) -> None:
        """Cut off whatever is playing. Safe to call when nothing is."""

    @property
    @abstractmethod
    def is_playing(self) -> bool: ...


class MockSoundPlayer(BaseSoundPlayer):
    """Records calls instead of making a sound. A track plays until `stop`."""

    def __init__(self) -> None:
        self.played_tracks: list[str] = []
        self.stopped_count = 0
        self._current: str | None = None

    def play(self, name: str) -> None:
        if name != self._current:
            self._current = name
            self.played_tracks.append(name)

    def stop(self) -> None:
        self._current = None
        self.stopped_count += 1

    @property
    def is_playing(self) -> bool:
        return self._current is not None


class AfplayAudioPlayer(BaseSoundPlayer):
    """Runs one macOS `afplay` process per track. Safe to call from any thread.

    Popen takes 2-5 ms, so it runs on a thread of its own and outside the lock.
    A track that fails is not tried again this session.
    """

    def __init__(self, sounds_dir: str | Path = SOUNDS_DIR, enabled: bool = True) -> None:
        self._sounds_dir = Path(sounds_dir)
        self._enabled = enabled
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._track: str | None = None
        self._started = 0.0
        self._spawn_id = 0  # Tells a spawn that finishes late it has been overtaken.
        self._spawning = False
        self._stopped: list[subprocess.Popen[bytes]] = []  # Terminated, not yet reaped.
        self._unplayable: set[str] = set()

    def play(self, name: str) -> None:
        with self._lock:
            if not self._enabled or name in self._unplayable:
                return
            if self._track == name and self._is_playing():
                return
            self._halt()
            self._track = name
            self._spawning = True
            spawn_id = self._spawn_id
        # Not a daemon: the interpreter waits for it, so a quit mid-spawn leaves no orphan.
        threading.Thread(target=self._spawn, args=(name, spawn_id)).start()

    def stop(self) -> None:
        with self._lock:
            self._halt()

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._is_playing()

    def _spawn(self, name: str, spawn_id: int) -> None:
        path = self._sounds_dir / f"{name}.mp3"
        process = None
        try:
            if path.is_file():
                process = subprocess.Popen(
                    ["afplay", str(path)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                logger.warning("no sound file: %s", path)
        except (OSError, ValueError) as error:
            logger.warning("cannot run afplay: %s", error)
        with self._lock:
            if process is None:
                self._unplayable.add(name)
            if spawn_id != self._spawn_id:
                if process is not None:
                    self._terminate(process)
                return
            self._spawning = False
            self._process = process
            self._started = time.monotonic()
            if process is None:
                self._track = None

    def _is_playing(self) -> bool:
        self._stopped = [process for process in self._stopped if process.poll() is None]
        if self._spawning:
            return True
        if self._process is None:
            return False
        code = self._process.poll()
        if code is None:
            return True
        if code != 0 or time.monotonic() - self._started < MIN_PLAY_SECONDS:
            logger.warning("afplay failed on %s (exit code %d)", self._track, code)
            self._unplayable.add(self._track)
        self._process = None
        self._track = None
        return False

    def _halt(self) -> None:
        self._spawn_id += 1
        self._spawning = False
        self._track = None
        if self._process is not None:
            self._terminate(self._process)
            self._process = None
        self._stopped = [process for process in self._stopped if process.poll() is None]

    def _terminate(self, process: subprocess.Popen[bytes]) -> None:
        # No wait, which could block; `_stopped` reaps it on a later call.
        try:
            process.terminate()
        except OSError as error:
            logger.warning("cannot stop afplay: %s", error)
        self._stopped.append(process)
