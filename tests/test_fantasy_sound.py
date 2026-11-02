"""FANTASY sound: when it starts and stops, and that failing never touches the video loop.

No test here can make a sound: `real_audio_forbidden` replaces Popen for all of them, and
the tests that need a real process to kill run `sleep` instead of `afplay`.
"""

import subprocess
import threading
import time

import pytest

from hand_factory import POSES, make_hand
from src.controller.base import MockMouseController, Point
from src.controller.media import MockMediaController
from src.controller.swap import SwapController
from src.fantasy import sound
from src.fantasy.engine import FantasyConfig, FantasyPhase
from src.fantasy.mode import FantasyMode
from src.fantasy.sound import AfplayAudioPlayer, BaseSoundPlayer, MockSoundPlayer
from src.main import AppMode, GestureControlApp, Tick, build_parser, build_sound_player, main
from src.vision.classifier import PoseClassifier
from src.vision.detector import DetectionResult
from src.vision.recorder import LandmarkRecorder

WIDTH, HEIGHT = 640, 480
FRAME = 0.033
REAL_POPEN = subprocess.Popen


class FakeProcess:
    def __init__(self, code=None):
        self.code = code
        self.terminated = 0

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated += 1
        self.code = -15


class FakePopen:
    """Stands in for `subprocess.Popen`, handing out `exit_code` processes."""

    def __init__(self, exit_code=None, error=None):
        self.exit_code = exit_code
        self.error = error
        self.calls = []
        self.processes = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if self.error is not None:
            raise self.error
        self.processes.append(FakeProcess(self.exit_code))
        return self.processes[-1]


@pytest.fixture(autouse=True)
def real_audio_forbidden(monkeypatch):
    popen = FakePopen(error=AssertionError("a test tried to start a real process"))
    monkeypatch.setattr(subprocess, "Popen", popen)
    return popen


@pytest.fixture
def popen(monkeypatch):
    fake = FakePopen()
    monkeypatch.setattr(subprocess, "Popen", fake)
    return fake


@pytest.fixture
def sounds_dir(tmp_path):
    (tmp_path / "chidori.mp3").write_bytes(b"audio")
    (tmp_path / "rasengan.mp3").write_bytes(b"audio")
    return tmp_path


def wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached in time"
        time.sleep(0.001)


def join_spawns():
    for thread in threading.enumerate():
        if thread is not threading.current_thread() and not thread.daemon:
            thread.join(5.0)


# FantasyMode drives the player from the FSM


class Session:
    def __init__(self, player=None, config=FantasyConfig()):
        self.player = player if player is not None else MockSoundPlayer()
        self.mode = FantasyMode(PoseClassifier(), config=config, sound_player=self.player)
        self.t = 1000.0

    def tick(self, pose):
        """Advance one frame with `pose`, or without a hand if it is None."""
        hands = () if pose is None else (make_hand(POSES[pose], x=0.5, y=0.5),)
        self.t += FRAME
        self.mode.update(DetectionResult(round(self.t * 1000), WIDTH, HEIGHT, hands), self.t)

    def until(self, pose, phase, limit=200):
        for _ in range(limit):
            self.tick(pose)
            if self.mode.engine.phase is phase:
                return
        pytest.fail(f"never reached {phase}")

    @property
    def phase(self):
        return self.mode.engine.phase


def test_chidori_is_silent_while_charging_and_starts_on_active():
    session = Session()
    session.until("FIST", FantasyPhase.CHARGING)
    assert session.player.played_tracks == []
    session.until("FIST", FantasyPhase.ACTIVE)
    assert session.player.played_tracks == ["chidori"]


def test_rasengan_starts_while_charging_and_carries_over_into_active():
    session = Session()
    session.until("OPEN_PALM", FantasyPhase.CHARGING)
    assert session.player.played_tracks == ["rasengan"]
    session.until("OPEN_PALM", FantasyPhase.ACTIVE)
    assert session.player.played_tracks == ["rasengan"]
    assert session.player.stopped_count == 0


@pytest.mark.parametrize("pose", ["FIST", "OPEN_PALM"])
def test_a_held_effect_never_spams_the_player(pose):
    session = Session()
    session.until(pose, FantasyPhase.ACTIVE)
    for _ in range(300):
        session.tick(pose)
    assert len(session.player.played_tracks) == 1
    assert session.player.stopped_count == 0


class EndsAtOnce(MockSoundPlayer):
    """A clip that has already finished by the next frame."""

    def play(self, name):
        self.played_tracks.append(name)

    @property
    def is_playing(self):
        return False


def test_a_clip_that_ends_while_the_effect_holds_starts_over():
    session = Session(EndsAtOnce())
    session.until("FIST", FantasyPhase.ACTIVE)
    before = len(session.player.played_tracks)
    session.tick("FIST")
    assert len(session.player.played_tracks) == before + 1


@pytest.mark.parametrize("pose", ["FIST", "OPEN_PALM"])
def test_leaving_the_pose_cuts_the_sound_on_the_cooldown_frame(pose):
    session = Session()
    session.until(pose, FantasyPhase.ACTIVE)
    for _ in range(50):
        session.tick("POINTING")
        if session.phase is FantasyPhase.COOLDOWN:
            break
        assert session.player.stopped_count == 0
    assert session.phase is FantasyPhase.COOLDOWN
    assert session.player.stopped_count == 1
    assert not session.player.is_playing


def test_losing_the_hand_cuts_the_sound_after_the_timeout():
    session = Session()
    session.until("FIST", FantasyPhase.ACTIVE)
    session.tick(None)
    assert session.player.is_playing  # A short dropout does not count as losing the hand.
    session.until(None, FantasyPhase.COOLDOWN)
    assert session.player.stopped_count == 1


def test_a_tick_without_a_detection_also_cuts_the_sound():
    session = Session()
    session.until("FIST", FantasyPhase.ACTIVE)
    session.mode.update(None, session.t + 1.0)
    assert session.phase is FantasyPhase.COOLDOWN
    assert not session.player.is_playing


def test_an_abandoned_rasengan_charge_cuts_the_sound():
    session = Session(config=FantasyConfig(hold_seconds=10.0))
    session.until("OPEN_PALM", FantasyPhase.CHARGING)
    assert session.player.is_playing
    session.until("POINTING", FantasyPhase.IDLE)
    assert not session.player.is_playing


def test_changing_a_rasengan_charge_into_a_chidori_charge_cuts_the_sound():
    session = Session(config=FantasyConfig(hold_seconds=10.0))
    session.until("OPEN_PALM", FantasyPhase.CHARGING)
    for _ in range(50):
        session.tick("FIST")
    assert session.phase is FantasyPhase.CHARGING
    assert session.mode.engine.effect.value == "chidori"
    assert not session.player.is_playing


def test_idle_and_cooldown_do_not_stop_over_and_over():
    session = Session()
    session.until("FIST", FantasyPhase.ACTIVE)
    session.until("POINTING", FantasyPhase.COOLDOWN)
    session.until("POINTING", FantasyPhase.IDLE)
    for _ in range(30):
        session.tick("POINTING")
    assert session.player.stopped_count == 1


def test_the_next_effect_plays_after_the_cooldown():
    session = Session()
    session.until("FIST", FantasyPhase.ACTIVE)
    session.until("POINTING", FantasyPhase.IDLE)
    session.until("OPEN_PALM", FantasyPhase.ACTIVE)
    assert session.player.played_tracks == ["chidori", "rasengan"]


def test_reset_stops_the_sound_and_a_held_pose_can_start_it_again():
    session = Session()
    session.until("FIST", FantasyPhase.ACTIVE)
    session.mode.reset()
    assert session.player.stopped_count == 1
    assert not session.player.is_playing
    session.until("FIST", FantasyPhase.ACTIVE)
    assert session.player.played_tracks == ["chidori", "chidori"]


def test_fantasy_mode_is_silent_unless_given_a_player():
    assert isinstance(FantasyMode(PoseClassifier()).sound_player, MockSoundPlayer)


# MockSoundPlayer


def test_mock_is_a_sound_player_that_logs_its_calls():
    player = MockSoundPlayer()
    assert isinstance(player, BaseSoundPlayer)
    assert not player.is_playing
    player.play("chidori")
    player.play("chidori")
    assert player.is_playing
    player.play("rasengan")
    player.stop()
    assert player.played_tracks == ["chidori", "rasengan"]
    assert player.stopped_count == 1
    assert not player.is_playing


def test_the_base_class_cannot_be_used_directly():
    with pytest.raises(TypeError):
        BaseSoundPlayer()


# AfplayAudioPlayer


def test_play_runs_afplay_on_the_file_with_no_terminal_io(popen, sounds_dir):
    player = AfplayAudioPlayer(sounds_dir)
    player.play("chidori")
    join_spawns()
    argv, kwargs = popen.calls[0]
    assert argv == ["afplay", str(sounds_dir / "chidori.mp3")]
    assert kwargs["stdin"] == kwargs["stdout"] == kwargs["stderr"] == subprocess.DEVNULL
    assert player.is_playing


def test_real_assets_exist_where_the_player_looks():
    for name in ("chidori", "rasengan"):
        assert (sound.SOUNDS_DIR / f"{name}.mp3").is_file()


def test_playing_the_same_track_twice_starts_one_process(popen, sounds_dir):
    player = AfplayAudioPlayer(sounds_dir)
    for _ in range(50):
        player.play("chidori")
    join_spawns()
    for _ in range(50):
        player.play("chidori")
    join_spawns()
    assert len(popen.calls) == 1


def test_another_track_replaces_the_current_one(popen, sounds_dir):
    player = AfplayAudioPlayer(sounds_dir)
    player.play("chidori")
    join_spawns()
    player.play("rasengan")
    join_spawns()
    assert popen.processes[0].terminated == 1
    assert popen.calls[1][0][1].endswith("rasengan.mp3")
    assert player.is_playing


def test_stop_terminates_the_process_and_is_safe_to_repeat(popen, sounds_dir):
    player = AfplayAudioPlayer(sounds_dir)
    player.stop()
    player.play("chidori")
    join_spawns()
    player.stop()
    player.stop()
    assert popen.processes[0].terminated == 1
    assert not player.is_playing


def test_stop_kills_a_real_process_at_once(monkeypatch, sounds_dir):
    spawned = []

    def spawn_sleep(_argv, **kwargs):
        spawned.append(REAL_POPEN(["sleep", "30"], **kwargs))
        return spawned[-1]

    monkeypatch.setattr(subprocess, "Popen", spawn_sleep)
    player = AfplayAudioPlayer(sounds_dir)
    player.play("chidori")
    join_spawns()
    assert spawned[0].poll() is None

    player.stop()
    assert spawned[0].wait(timeout=2) == -15


def test_stop_during_a_slow_spawn_kills_the_late_process(monkeypatch, sounds_dir):
    release = threading.Event()
    fake = FakePopen()

    def slow(argv, **kwargs):
        release.wait(5.0)
        return fake(argv, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", slow)
    player = AfplayAudioPlayer(sounds_dir)

    started = time.perf_counter()
    player.play("chidori")
    assert player.is_playing
    player.stop()
    assert time.perf_counter() - started < 0.1  # Neither call waited for the spawn.
    assert not player.is_playing

    release.set()
    join_spawns()
    assert fake.processes[0].terminated == 1
    assert not player.is_playing


def test_the_track_can_play_again_right_after_a_stop(popen, sounds_dir):
    player = AfplayAudioPlayer(sounds_dir)
    player.play("chidori")
    player.stop()
    player.play("chidori")
    join_spawns()
    # The two spawns race, so either process may be the stale one.
    assert sorted(process.terminated for process in popen.processes) == [0, 1]
    assert player.is_playing


def test_a_clip_that_played_to_its_end_can_play_again(popen, sounds_dir, monkeypatch):
    monkeypatch.setattr(sound, "MIN_PLAY_SECONDS", 0.0)
    player = AfplayAudioPlayer(sounds_dir)
    player.play("chidori")
    join_spawns()
    popen.processes[0].code = 0
    assert not player.is_playing
    player.play("chidori")
    join_spawns()
    assert len(popen.calls) == 2


def test_concurrent_calls_leave_at_most_one_live_process(popen, sounds_dir):
    player = AfplayAudioPlayer(sounds_dir)

    def hammer(name):
        for i in range(100):
            player.play(name)
            _ = player.is_playing
            if i % 7 == 0:
                player.stop()

    threads = [threading.Thread(target=hammer, args=(n,)) for n in ("chidori", "rasengan") * 4]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    join_spawns()
    assert sum(process.code is None for process in popen.processes) <= 1


# Failing quietly


def assert_retired(player, popen, name):
    """The track failed, and no later frame may spawn another process for it."""
    wait_for(lambda: not player.is_playing)
    calls = len(popen.calls)
    for _ in range(20):
        player.play(name)
    join_spawns()
    assert len(popen.calls) == calls
    assert not player.is_playing


def test_a_missing_file_is_quiet_and_starts_no_process(popen, tmp_path):
    player = AfplayAudioPlayer(tmp_path)
    player.play("chidori")
    assert_retired(player, popen, "chidori")
    assert popen.calls == []


def test_a_missing_directory_is_quiet(popen, tmp_path):
    player = AfplayAudioPlayer(tmp_path / "nowhere")
    player.play("chidori")
    assert_retired(player, popen, "chidori")
    player.stop()


@pytest.mark.parametrize("error", [FileNotFoundError("afplay"), PermissionError("afplay")])
def test_afplay_that_cannot_start_is_quiet(monkeypatch, sounds_dir, error):
    fake = FakePopen(error=error)
    monkeypatch.setattr(subprocess, "Popen", fake)
    player = AfplayAudioPlayer(sounds_dir)
    player.play("chidori")
    assert_retired(player, fake, "chidori")
    assert len(fake.calls) == 1


@pytest.mark.parametrize("code", [1, 0], ids=["exit 1", "instant exit 0 on a bad file"])
def test_afplay_that_dies_at_once_is_not_respawned_every_frame(monkeypatch, sounds_dir, code):
    fake = FakePopen(exit_code=code)
    monkeypatch.setattr(subprocess, "Popen", fake)
    player = AfplayAudioPlayer(sounds_dir)
    player.play("chidori")
    join_spawns()
    assert_retired(player, fake, "chidori")
    assert len(fake.calls) == 1


def test_one_bad_track_leaves_the_other_playable(popen, tmp_path):
    (tmp_path / "rasengan.mp3").write_bytes(b"audio")
    player = AfplayAudioPlayer(tmp_path)
    player.play("chidori")
    assert_retired(player, popen, "chidori")
    player.play("rasengan")
    join_spawns()
    assert len(popen.calls) == 1
    assert player.is_playing


def test_terminate_failing_is_quiet(popen, sounds_dir):
    player = AfplayAudioPlayer(sounds_dir)
    player.play("chidori")
    join_spawns()

    def refuse():
        raise PermissionError("no")

    popen.processes[0].terminate = refuse
    player.stop()
    assert not player.is_playing


def test_disabled_player_starts_nothing(popen, sounds_dir):
    player = AfplayAudioPlayer(sounds_dir, enabled=False)
    player.play("chidori")
    join_spawns()
    assert popen.calls == []
    assert not player.is_playing
    player.stop()


def test_the_mode_survives_a_track_that_cannot_play(popen, tmp_path):
    session = Session(AfplayAudioPlayer(tmp_path))
    session.until("FIST", FantasyPhase.ACTIVE)
    for _ in range(30):
        session.tick("FIST")
    session.until("POINTING", FantasyPhase.COOLDOWN)
    join_spawns()
    assert popen.calls == []


# The loop is never held up


def test_play_does_not_wait_for_the_process_to_start(monkeypatch, sounds_dir):
    release = threading.Event()

    def slow(*_args, **_kwargs):
        release.wait(5.0)
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", slow)
    player = AfplayAudioPlayer(sounds_dir)
    started = time.perf_counter()
    player.play("chidori")
    _ = player.is_playing
    elapsed = time.perf_counter() - started
    release.set()
    join_spawns()
    assert elapsed < 0.1


# Wiring: only a live, windowed session without --no-sound may make a sound


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], AfplayAudioPlayer),
        (["--mode", "fantasy"], AfplayAudioPlayer),
        (["--no-sound"], MockSoundPlayer),
        (["--headless"], MockSoundPlayer),
        (["--replay", "x.jsonl"], MockSoundPlayer),
        (["--replay", "x.jsonl", "--headless"], MockSoundPlayer),
    ],
)
def test_the_cli_picks_the_player(argv, expected):
    assert type(build_sound_player(build_parser().parse_args(argv))) is expected


def test_no_sound_defaults_to_off():
    assert build_parser().parse_args([]).no_sound is False


def save_fantasy_session(path):
    t = 1000
    with LandmarkRecorder(path) as recorder:
        for pose, frames in (("FIST", 30), ("POINTING", 60), ("OPEN_PALM", 30), (None, 30)):
            for _ in range(frames):
                hands = () if pose is None else (make_hand(POSES[pose], x=0.5, y=0.5),)
                recorder.record(DetectionResult(t, WIDTH, HEIGHT, hands))
                t += 33
    return path


def test_fantasy_replay_through_the_cli_starts_no_process(tmp_path, real_audio_forbidden):
    path = save_fantasy_session(tmp_path / "fantasy.jsonl")
    assert main(["--replay", str(path), "--mode", "fantasy", "--headless"]) == 0
    assert real_audio_forbidden.calls == []


class ListSource:
    def __init__(self, ticks=()):
        self.ticks = ticks

    def open(self):
        pass

    def frames(self, profiler):
        yield from self.ticks

    def close(self):
        pass


def make_app(source=None, mode=AppMode.FANTASY, sound_player=None):
    return GestureControlApp(
        source if source is not None else ListSource(),
        MockMouseController(start=Point(960, 540)),
        MockMediaController(),
        SwapController(post_event=lambda _event: None),
        mode=mode,
        sound_player=sound_player,
    )


def test_app_is_silent_unless_given_a_player():
    assert isinstance(make_app()._fantasy.sound_player, MockSoundPlayer)


def hold_fist(app):
    t = 1000
    for _ in range(15):
        hand = make_hand(POSES["FIST"], x=0.5, y=0.5)
        app._apply_gestures(Tick(t / 1000, DetectionResult(t, WIDTH, HEIGHT, (hand,))))
        t += 33


def active_fist(app):
    hold_fist(app)
    assert app._state_name() == "ACTIVE"


def test_switching_mode_cuts_the_sound():
    player = MockSoundPlayer()
    app = make_app(sound_player=player)
    active_fist(app)
    assert player.is_playing
    app.set_mode(AppMode.MOUSE)
    assert not player.is_playing


def test_quitting_cuts_the_sound():
    player = MockSoundPlayer()
    app = make_app(sound_player=player)
    active_fist(app)
    app.shutdown()
    assert not player.is_playing
    assert player.stopped_count >= 1


def test_shutdown_stops_the_player_even_if_an_earlier_step_fails():
    player = MockSoundPlayer()
    app = make_app(sound_player=player)
    active_fist(app)

    def fail():
        raise RuntimeError("boom")

    app._reset_gestures = fail
    with pytest.raises(RuntimeError):
        app.shutdown()
    assert not player.is_playing


def test_other_modes_never_touch_the_player():
    player = MockSoundPlayer()
    app = make_app(mode=AppMode.MEDIA, sound_player=player)
    hold_fist(app)
    assert player.played_tracks == []
