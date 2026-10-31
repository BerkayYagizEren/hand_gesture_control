import math

import numpy as np
import pytest
from mediapipe.tasks.python.components.containers.category import Category
from mediapipe.tasks.python.components.containers.landmark import (
    Landmark as MpLandmark,
    NormalizedLandmark,
)
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerResult

from src.vision.detector import (
    DEFAULT_MODEL_PATH,
    NUM_LANDMARKS,
    DetectionResult,
    Hand,
    HandDetector,
    HandLandmark,
    Landmark,
    MonotonicTimestamp,
    VisionRunningMode,
    palm_width,
    world_palm_width,
)


def blank(height=48, width=64):
    return np.zeros((height, width, 3), dtype=np.uint8)


def points_with(index_mcp, pinky_mcp):
    points = [Landmark(0.0, 0.0, 0.0)] * NUM_LANDMARKS
    points[HandLandmark.INDEX_FINGER_MCP] = Landmark(*index_mcp)
    points[HandLandmark.PINKY_MCP] = Landmark(*pinky_mcp)
    return points


def raw_result(num_hands=1):
    """A result built from MediaPipe's real container types."""
    return HandLandmarkerResult(
        handedness=[
            [Category(index=0, score=0.9, display_name="", category_name="Right")]
            for _ in range(num_hands)
        ],
        hand_landmarks=[
            [NormalizedLandmark(x=i / 100, y=i / 50, z=-i / 200) for i in range(NUM_LANDMARKS)]
            for _ in range(num_hands)
        ],
        hand_world_landmarks=[
            [MpLandmark(x=i / 1000, y=i / 500, z=i / 250) for i in range(NUM_LANDMARKS)]
            for _ in range(num_hands)
        ],
    )


class FakeLandmarker:
    def __init__(self, result=None):
        self.result = result or raw_result(0)
        self.timestamps = []
        self.images = []
        self.closed = 0

    def detect_for_video(self, image, timestamp_ms):
        self.images.append(image)
        self.timestamps.append(timestamp_ms)
        return self.result

    def close(self):
        self.closed += 1


@pytest.fixture
def fake_landmarker(monkeypatch):
    fake = FakeLandmarker()
    captured = {}

    def create(options):
        captured["options"] = options
        return fake

    monkeypatch.setattr(HandLandmarker, "create_from_options", create)
    fake.captured = captured
    return fake


def test_landmark_indices_cover_all_21_points():
    assert [int(point) for point in HandLandmark] == list(range(NUM_LANDMARKS))
    assert HandLandmark.INDEX_FINGER_MCP == 5
    assert HandLandmark.PINKY_MCP == 17


def test_monotonic_timestamp_passes_increasing_values():
    timestamps = MonotonicTimestamp()
    assert [timestamps.next(t) for t in (1, 5, 9)] == [1, 5, 9]


def test_monotonic_timestamp_bumps_equal_and_backward_values():
    timestamps = MonotonicTimestamp()
    assert [timestamps.next(t) for t in (10, 10, 3, 12, 13)] == [10, 11, 12, 13, 14]


def test_palm_width_square_image():
    points = points_with((0.3, 0.4, 0.5), (0.6, 0.8, -0.5))
    assert palm_width(points) == pytest.approx(0.5)  # z is ignored.


def test_palm_width_corrects_for_aspect_ratio():
    # Physically horizontal: 0.3 of a 640px width is 192px, 0.4 of a 480px height.
    points = points_with((0.2, 0.5, 0.0), (0.5, 0.5, 0.0))
    assert palm_width(points, aspect_ratio=640 / 480) == pytest.approx(0.4)


def test_world_palm_width_is_3d():
    points = points_with((0.01, 0.02, 0.03), (0.04, 0.06, 0.03 + 0.12))
    assert world_palm_width(points) == pytest.approx(math.sqrt(0.03**2 + 0.04**2 + 0.12**2))


def test_hand_rejects_wrong_landmark_count():
    good = tuple(points_with((0, 0, 0), (0, 0, 0)))
    with pytest.raises(ValueError, match="21"):
        Hand(good[:20], good, "Right", 0.9)
    with pytest.raises(ValueError, match="21"):
        Hand(good, good[:20], "Right", 0.9)


def test_result_aspect_ratio_and_size_validation():
    assert DetectionResult(0, 640, 480, ()).aspect_ratio == pytest.approx(4 / 3)
    with pytest.raises(ValueError):
        DetectionResult(0, 0, 480, ())


def test_options_use_video_mode_cpu_and_model(fake_landmarker):
    HandDetector(num_hands=2, min_tracking_confidence=0.7)
    options = fake_landmarker.captured["options"]
    assert options.running_mode == VisionRunningMode.VIDEO
    assert options.base_options.delegate == options.base_options.Delegate.CPU
    assert options.base_options.model_asset_path == str(DEFAULT_MODEL_PATH)
    assert options.num_hands == 2
    assert options.min_tracking_confidence == 0.7


def test_conversion_from_mediapipe_types(fake_landmarker):
    fake_landmarker.result = raw_result(2)
    result = HandDetector().detect(blank(), timestamp_ms=42)
    assert (result.timestamp_ms, result.image_width, result.image_height) == (42, 64, 48)
    assert len(result.hands) == 2
    hand = result.hands[0]
    assert hand.handedness == "Right"
    assert hand.handedness_score == pytest.approx(0.9)
    assert hand.landmarks[HandLandmark.INDEX_FINGER_MCP] == Landmark(0.05, 0.1, -0.025)
    assert hand.world_landmarks[HandLandmark.PINKY_MCP] == Landmark(0.017, 0.034, 0.068)


def test_detect_passes_strictly_increasing_timestamps(fake_landmarker):
    detector = HandDetector()
    results = [detector.detect(blank(), timestamp_ms=t) for t in (100, 100, 50, 200)]
    assert fake_landmarker.timestamps == [100, 101, 102, 200]
    assert [r.timestamp_ms for r in results] == fake_landmarker.timestamps


def test_detect_defaults_to_clock_in_milliseconds(fake_landmarker):
    detector = HandDetector(clock=lambda: 12.3456)
    assert detector.detect(blank()).timestamp_ms == 12345
    assert detector.detect(blank()).timestamp_ms == 12346  # Frozen clock still increases.


def test_detect_converts_bgr_to_rgb(fake_landmarker):
    image = blank()
    image[..., 0] = 200  # Blue channel in BGR.
    HandDetector().detect(image, timestamp_ms=1)
    rgb = fake_landmarker.images[0].numpy_view()
    assert rgb[0, 0, 2] == 200 and rgb[0, 0, 0] == 0


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((4, 4), dtype=np.uint8),
        np.zeros((4, 4, 4), dtype=np.uint8),
        np.zeros((4, 4, 3), dtype=np.float32),
    ],
)
def test_detect_rejects_invalid_images(fake_landmarker, image):
    with pytest.raises(ValueError, match="uint8 BGR"):
        HandDetector().detect(image, timestamp_ms=1)
    assert fake_landmarker.timestamps == []


def test_close_is_idempotent_and_detect_after_close_raises(fake_landmarker):
    detector = HandDetector()
    detector.close()
    detector.close()
    assert fake_landmarker.closed == 1
    with pytest.raises(RuntimeError, match="closed"):
        detector.detect(blank(), timestamp_ms=1)


def test_context_manager_closes(fake_landmarker):
    with HandDetector():
        pass
    assert fake_landmarker.closed == 1


def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        HandDetector(model_path=tmp_path / "missing.task")


def test_real_model_blank_frame_has_no_hands():
    with HandDetector() as detector:
        result = detector.detect(blank(240, 320), timestamp_ms=1)
    assert result == DetectionResult(1, 320, 240, ())


def test_real_model_accepts_repeated_timestamps():
    # MediaPipe itself raises ValueError on these if they reach it unchanged.
    with HandDetector() as detector:
        results = [detector.detect(blank(240, 320), timestamp_ms=5) for _ in range(3)]
    assert [r.timestamp_ms for r in results] == [5, 6, 7]
