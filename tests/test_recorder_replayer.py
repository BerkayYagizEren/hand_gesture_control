import json
import logging

import pytest

from src.vision.detector import NUM_LANDMARKS, DetectionResult, Hand, Landmark, world_palm_width
from src.vision.recorder import FORMAT_VERSION, LandmarkRecorder
from src.vision.replayer import LandmarkReplayer, ReplayFormatError


def make_hand(seed=0.0, handedness="Right"):
    # Values with long binary expansions, to prove exact float round-trips.
    landmarks = tuple(
        Landmark(seed + i / 3, seed - i / 7, (seed + i) * 1e-7) for i in range(NUM_LANDMARKS)
    )
    world = tuple(Landmark(seed + i / 11, i / 13, -i / 17) for i in range(NUM_LANDMARKS))
    return Hand(landmarks, world, handedness, 0.987654321)


def make_results():
    return [
        DetectionResult(10, 640, 480, (make_hand(0.1),)),
        DetectionResult(43, 640, 480, ()),  # Hand loss must survive the round trip.
        DetectionResult(77, 640, 480, (make_hand(0.2), make_hand(0.3, "Left"))),
    ]


def record(path, results=None, **kwargs):
    with LandmarkRecorder(path, **kwargs) as recorder:
        for result in results if results is not None else make_results():
            recorder.record(result)
    return path


@pytest.fixture
def path(tmp_path):
    return tmp_path / "session.jsonl"


def valid_frame_line(timestamp_ms=1):
    point = [0.0, 0.0, 0.0]
    return json.dumps(
        {
            "type": "frame",
            "timestamp_ms": timestamp_ms,
            "image_size": [640, 480],
            "hands": [
                {
                    "handedness": "Left",
                    "score": 0.5,
                    "landmarks": [point] * NUM_LANDMARKS,
                    "world_landmarks": [point] * NUM_LANDMARKS,
                }
            ],
        }
    )


def write_lines(path, *lines):
    header = json.dumps({"type": "header", "version": FORMAT_VERSION, "metadata": {}})
    path.write_text("\n".join((header, *lines)) + "\n", encoding="utf-8")
    return path


def test_round_trip_is_exact(path):
    record(path)
    assert list(LandmarkReplayer(path)) == make_results()


def test_replay_is_deterministic_across_iterations_and_instances(path):
    record(path)
    replayer = LandmarkReplayer(path)
    first = list(replayer)
    assert list(replayer) == first
    assert list(LandmarkReplayer(path)) == first


def test_replayed_results_work_with_detector_helpers(path):
    record(path)
    result = next(iter(LandmarkReplayer(path)))
    assert isinstance(result, DetectionResult)
    assert isinstance(result.hands[0].landmarks[0], Landmark)
    assert world_palm_width(result.hands[0].world_landmarks) == world_palm_width(
        make_results()[0].hands[0].world_landmarks
    )
    assert result.aspect_ratio == pytest.approx(4 / 3)


def test_file_is_one_json_object_per_line(path):
    record(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    records = [json.loads(line) for line in lines]
    assert records[0]["type"] == "header"
    assert records[0]["version"] == FORMAT_VERSION
    assert all(r["type"] == "frame" for r in records[1:])
    assert len(records[1]["hands"][0]["landmarks"]) == NUM_LANDMARKS
    assert records[2]["hands"] == []


def test_metadata_round_trip(path):
    record(path, metadata={"gesture": "pinch", "fps": 30})
    assert LandmarkReplayer(path).metadata == {"gesture": "pinch", "fps": 30}


def test_recording_with_no_frames_replays_empty(path):
    record(path, results=[])
    replayer = LandmarkReplayer(path)
    assert replayer.metadata == {}
    assert list(replayer) == []


def test_frame_count(path):
    with LandmarkRecorder(path) as recorder:
        assert recorder.frame_count == 0
        recorder.record(make_results()[0])
        assert recorder.frame_count == 1


def test_recorder_creates_parent_directories(tmp_path):
    nested = tmp_path / "a" / "b" / "session.jsonl"
    record(nested)
    assert nested.exists()


def test_recorder_refuses_to_replace_existing_recording(path):
    record(path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        LandmarkRecorder(path)
    assert path.read_bytes() == before


def test_recorder_overwrite_replaces_recording(path):
    record(path)
    record(path, results=make_results()[:1], overwrite=True)
    assert len(list(LandmarkReplayer(path))) == 1


def test_data_is_on_disk_before_close(path):
    recorder = LandmarkRecorder(path)
    recorder.record(make_results()[0])
    assert list(LandmarkReplayer(path)) == make_results()[:1]
    recorder.close()


def test_record_after_close_raises(path):
    recorder = LandmarkRecorder(path)
    recorder.close()
    recorder.close()
    with pytest.raises(RuntimeError, match="closed"):
        recorder.record(make_results()[0])


def test_recorder_rejects_nan(path):
    hand = make_hand()
    bad = Hand(
        (Landmark(float("nan"), 0.0, 0.0), *hand.landmarks[1:]),
        hand.world_landmarks,
        "Right",
        0.5,
    )
    with LandmarkRecorder(path) as recorder:
        with pytest.raises(ValueError):
            recorder.record(DetectionResult(1, 640, 480, (bad,)))
    assert list(LandmarkReplayer(path)) == []


def test_truncated_last_line_is_ignored_with_warning(path, caplog):
    record(path)
    content = path.read_text(encoding="utf-8")
    path.write_text(content[: len(content) - 40], encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert list(LandmarkReplayer(path)) == make_results()[:2]
    assert "truncated" in caplog.text


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        LandmarkReplayer(tmp_path / "missing.jsonl")


def test_empty_file_raises(path):
    path.write_text("", encoding="utf-8")
    with pytest.raises(ReplayFormatError, match="empty"):
        LandmarkReplayer(path)


def test_missing_header_raises(path):
    path.write_text(valid_frame_line() + "\n", encoding="utf-8")
    with pytest.raises(ReplayFormatError, match="header"):
        LandmarkReplayer(path)


def test_unsupported_version_raises(path):
    path.write_text(json.dumps({"type": "header", "version": 999}) + "\n", encoding="utf-8")
    with pytest.raises(ReplayFormatError, match="version"):
        LandmarkReplayer(path)


def test_valid_fixture_line_parses(path):
    # Guards the corruption tests below: only the injected defect makes them fail.
    write_lines(path, valid_frame_line())
    assert len(list(LandmarkReplayer(path))) == 1


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.pop("hands"), "malformed"),
        (lambda r: r.pop("timestamp_ms"), "malformed"),
        (lambda r: r.update(image_size=[640]), "malformed"),
        (lambda r: r.update(image_size=[0, 480]), "malformed"),
        (lambda r: r["hands"][0].pop("score"), "malformed"),
        (lambda r: r["hands"][0].update(landmarks=[[0.0, 0.0, 0.0]] * 20), "malformed"),
        (lambda r: r["hands"][0].update(landmarks=[[0.0, 0.0]] * NUM_LANDMARKS), "malformed"),
        (lambda r: r["hands"][0].update(landmarks=[["a", 0.0, 0.0]] * 21), "malformed"),
        (lambda r: r.update(type="other"), "record type"),
    ],
)
def test_malformed_frame_raises_with_line_number(path, mutate, message):
    record = json.loads(valid_frame_line())
    mutate(record)
    write_lines(path, json.dumps(record))
    with pytest.raises(ReplayFormatError, match=rf":2: .*{message}"):
        list(LandmarkReplayer(path))


def test_corrupt_middle_line_raises(path):
    write_lines(path, "{not json", valid_frame_line())
    with pytest.raises(ReplayFormatError, match=":2: invalid JSON"):
        list(LandmarkReplayer(path))


def test_non_object_line_raises(path):
    write_lines(path, "[1, 2]")
    with pytest.raises(ReplayFormatError, match="not an object"):
        list(LandmarkReplayer(path))


@pytest.mark.parametrize("second", [5, 4])
def test_non_increasing_timestamps_raise(path, second):
    write_lines(path, valid_frame_line(5), valid_frame_line(second))
    with pytest.raises(ReplayFormatError, match="does not increase"):
        list(LandmarkReplayer(path))
