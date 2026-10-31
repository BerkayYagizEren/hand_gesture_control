import math

import pytest

from hand_factory import (
    CURLED,
    PARTIAL,
    POSES,
    THUMB_LATERAL,
    THUMB_OUT,
    THUMB_TUCKED,
    make_hand,
    world_points,
)
from src.vision.classifier import PINCH_START_RATIO, Pose, PoseClassifier, TemporalVotingFilter
from src.vision.detector import NUM_LANDMARKS, HandLandmark, Landmark

classifier = PoseClassifier()


def classify(world):
    return classifier.classify(make_hand(world)).pose


@pytest.mark.parametrize(
    ("name", "pose"),
    [
        ("POINTING", Pose.POINTING),
        ("POINTING_THUMB_OUT", Pose.POINTING),
        ("PINCH", Pose.PINCH),
        ("MIDDLE_PINCH", Pose.MIDDLE_PINCH),
        ("FIST", Pose.FIST),
        ("TWO_FINGER", Pose.TWO_FINGER),
        ("OPEN_PALM", Pose.OPEN_PALM),
        ("THUMBS_UP", Pose.THUMBS_UP),
        ("THREE_FINGER", Pose.THREE_FINGER),
    ],
)
def test_each_pose(name, pose):
    assert classify(POSES[name]) is pose


def test_observation_carries_the_raw_ratios():
    observation = classifier.classify(make_hand(world_points(index=PARTIAL, pinch=0.1)))
    assert observation.pinch_ratio == pytest.approx(0.1)
    assert observation.middle_pinch_ratio > 0.25


def test_pinch_threshold():
    assert classify(world_points(index=PARTIAL, pinch=PINCH_START_RATIO - 0.02)) is Pose.PINCH
    assert classify(world_points(index=PARTIAL, pinch=PINCH_START_RATIO + 0.02)) is not Pose.PINCH


def test_pinch_threshold_is_configurable():
    world = world_points(index=PARTIAL, pinch=0.3)
    assert PoseClassifier(pinch_start_ratio=0.35).classify(make_hand(world)).pose is Pose.PINCH


@pytest.mark.parametrize(
    ("offset", "pose"), [(0.006, Pose.PINCH), (0.014, Pose.MIDDLE_PINCH)]
)
def test_closer_finger_wins_when_thumb_is_near_both(offset, pose):
    # The fingertips are 0.02 apart, and the thumb tip sits between them within both ratios.
    world = list(world_points(index=PARTIAL, middle=PARTIAL))
    index_tip = world[HandLandmark.INDEX_FINGER_TIP]
    world[HandLandmark.THUMB_TIP] = Landmark(index_tip.x + offset, index_tip.y, index_tip.z)
    observation = classifier.classify(make_hand(tuple(world)))
    assert max(observation.pinch_ratio, observation.middle_pinch_ratio) < 0.25
    assert observation.pose is pose


def test_thumb_wrapped_over_a_fist_is_not_a_pinch():
    fist = world_points(CURLED, CURLED, CURLED, CURLED, THUMB_TUCKED, pinch=0.1)
    assert classify(fist) is Pose.FIST


def test_fist_with_thumb_on_middle_finger_is_not_a_right_click():
    fist = world_points(CURLED, CURLED, CURLED, CURLED, THUMB_TUCKED, middle_pinch=0.1)
    assert classify(fist) is Pose.FIST


def test_thumbs_up_needs_the_thumb_to_point_up():
    sideways = world_points(CURLED, CURLED, CURLED, CURLED, THUMB_LATERAL)
    assert classify(sideways) is Pose.UNKNOWN


def test_upside_down_thumb_is_not_thumbs_up():
    thumbs_up = POSES["THUMBS_UP"]
    upside_down = tuple(Landmark(p.x, -p.y, p.z) for p in thumbs_up)
    assert classify(upside_down) is Pose.UNKNOWN


def test_open_palm_needs_the_thumb():
    assert classify(world_points(thumb=THUMB_TUCKED)) is Pose.UNKNOWN


def test_half_curled_hand_is_unknown():
    half = world_points(PARTIAL, PARTIAL, PARTIAL, PARTIAL, THUMB_TUCKED)
    assert classify(half) is Pose.UNKNOWN


def test_partially_curled_neighbours_still_count_as_pointing():
    relaxed = world_points(middle=PARTIAL, ring=PARTIAL, pinky=CURLED, thumb=THUMB_TUCKED)
    assert classify(relaxed) is Pose.POINTING


def test_non_adjacent_fingers_are_unknown():
    assert classify(world_points(middle=CURLED, ring=CURLED, thumb=THUMB_TUCKED)) is Pose.UNKNOWN


@pytest.mark.parametrize("pinky", [CURLED, PARTIAL])
@pytest.mark.parametrize("thumb", [THUMB_TUCKED, THUMB_OUT])
def test_three_finger_allows_any_thumb_and_a_pinky_that_is_not_extended(pinky, thumb):
    assert classify(world_points(pinky=pinky, thumb=thumb)) is Pose.THREE_FINGER


def test_extended_pinky_is_not_three_finger():
    assert classify(world_points(thumb=THUMB_TUCKED)) is not Pose.THREE_FINGER


@pytest.mark.parametrize("bent", ["index", "middle", "ring"])
def test_three_finger_needs_index_middle_and_ring(bent):
    fingers = {"pinky": CURLED, "thumb": THUMB_TUCKED, bent: CURLED}
    assert classify(world_points(**fingers)) is not Pose.THREE_FINGER


def test_degenerate_hand_is_unknown():
    flat = tuple(Landmark(0.0, 0.0, 0.0) for _ in range(NUM_LANDMARKS))
    observation = classifier.classify(make_hand(flat))
    assert observation.pose is Pose.UNKNOWN
    assert observation.pinch_ratio == math.inf


def test_nan_landmarks_are_unknown():
    nan_hand = tuple(Landmark(math.nan, math.nan, math.nan) for _ in range(NUM_LANDMARKS))
    assert classify(nan_hand) is Pose.UNKNOWN


def feed(voting, *poses):
    return [voting.update(pose) for pose in poses]


def test_voting_starts_unknown_until_enough_votes():
    voting = TemporalVotingFilter(window=5, min_votes=3)
    assert voting.pose is Pose.UNKNOWN
    assert feed(voting, Pose.PINCH, Pose.PINCH, Pose.PINCH) == [
        Pose.UNKNOWN,
        Pose.UNKNOWN,
        Pose.PINCH,
    ]


def test_voting_ignores_a_single_frame_glitch():
    voting = TemporalVotingFilter()
    feed(voting, *[Pose.POINTING] * 5)
    assert feed(voting, Pose.FIST, Pose.POINTING, Pose.POINTING) == [Pose.POINTING] * 3


def test_voting_counts_non_consecutive_votes_in_window():
    voting = TemporalVotingFilter()
    feed(voting, *[Pose.POINTING] * 5)
    outputs = feed(voting, Pose.PINCH, Pose.POINTING, Pose.PINCH, Pose.UNKNOWN, Pose.PINCH)
    assert outputs[-1] is Pose.PINCH
    assert set(outputs[:-1]) == {Pose.POINTING}


def test_voting_holds_the_last_winner_while_undecided():
    voting = TemporalVotingFilter()
    feed(voting, *[Pose.POINTING] * 5)
    outputs = feed(voting, Pose.FIST, Pose.PINCH, Pose.OPEN_PALM, Pose.TWO_FINGER, Pose.THUMBS_UP)
    assert outputs == [Pose.POINTING] * 5


def test_voting_old_frames_leave_the_window():
    voting = TemporalVotingFilter(window=5, min_votes=3)
    feed(voting, Pose.PINCH, Pose.PINCH, Pose.UNKNOWN, Pose.UNKNOWN, Pose.FIST, Pose.FIST)
    # Window is now UNKNOWN, UNKNOWN, FIST, FIST + next: a third FIST wins despite old pinches.
    assert voting.update(Pose.FIST) is Pose.FIST


def test_voting_reset_clears_history():
    voting = TemporalVotingFilter()
    feed(voting, *[Pose.PINCH] * 5)
    voting.reset()
    assert voting.pose is Pose.UNKNOWN
    assert voting.update(Pose.PINCH) is Pose.UNKNOWN


@pytest.mark.parametrize(("window", "votes"), [(5, 2), (4, 2), (5, 6), (5, 0), (0, 0)])
def test_voting_rejects_non_majority_configs(window, votes):
    with pytest.raises(ValueError):
        TemporalVotingFilter(window, votes)
