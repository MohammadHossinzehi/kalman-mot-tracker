import numpy as np
import pytest

from tracker import MultiObjectTracker


def test_single_object_gets_one_confirmed_track():
    tracker = MultiObjectTracker(max_distance=15.0, min_hits=3)
    confirmed = []
    for t in range(10):
        confirmed = tracker.step([(float(t), 0.0)])
    assert len(confirmed) == 1
    assert confirmed[0].position[0] == pytest.approx(9.0, abs=1.0)


def test_track_id_is_stable_across_frames():
    tracker = MultiObjectTracker(max_distance=15.0, min_hits=2)
    ids_seen = set()
    for t in range(15):
        confirmed = tracker.step([(t * 2.0, 5.0)])
        for track in confirmed:
            ids_seen.add(track.track_id)
    # A single smoothly moving object should only ever get one stable ID.
    assert len(ids_seen) == 1


def test_two_well_separated_objects_get_two_tracks():
    tracker = MultiObjectTracker(max_distance=15.0, min_hits=2)
    for t in range(10):
        dets = [(float(t), 0.0), (float(t), 200.0)]
        confirmed = tracker.step(dets)
    assert len(confirmed) == 2
    ys = sorted(track.position[1] for track in confirmed)
    assert ys[0] < 20
    assert ys[1] > 180


def test_missed_detections_do_not_immediately_kill_a_track():
    tracker = MultiObjectTracker(max_distance=15.0, max_age=5, min_hits=2)
    for t in range(6):
        tracker.step([(float(t), 0.0)])
    # Miss for a couple of frames (fewer than max_age).
    tracker.step([])
    tracker.step([])
    confirmed = tracker.step([(6.0, 0.0)])
    assert len(confirmed) == 1


def test_track_is_dropped_after_max_age_missed_frames():
    tracker = MultiObjectTracker(max_distance=15.0, max_age=3, min_hits=2)
    for t in range(6):
        tracker.step([(float(t), 0.0)])
    assert len(tracker.tracks) == 1
    for _ in range(4):  # exceeds max_age
        tracker.step([])
    assert len(tracker.tracks) == 0


def test_far_apart_detection_spawns_new_track_instead_of_hijacking():
    tracker = MultiObjectTracker(max_distance=10.0, min_hits=2)
    for t in range(5):
        tracker.step([(float(t), 0.0)])
    # A detection far outside max_distance must not be matched to the
    # existing track -- it should start a brand new one.
    tracker.step([(500.0, 500.0)])
    assert len(tracker.tracks) == 2


def test_clutter_does_not_produce_a_confirmed_track_with_low_min_hits_violation():
    tracker = MultiObjectTracker(max_distance=10.0, min_hits=5)
    rng = np.random.default_rng(3)
    for _ in range(4):
        # Random, unrelated single-frame clutter every frame -- nothing
        # should persist long enough to be confirmed.
        tracker.step([(rng.uniform(0, 100), rng.uniform(0, 100))])
    assert len(tracker.confirmed_tracks()) == 0
