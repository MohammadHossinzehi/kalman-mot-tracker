"""
Multi-object tracker: Kalman filters + Hungarian assignment for data
association, following the classic tracking-by-detection design used by
algorithms like SORT (Bewley et al., 2016).

Pipeline, per frame:
  1. Predict every existing track's position one step forward.
  2. Build a cost matrix (Euclidean distance) between predicted track
     positions and the new frame's detections.
  3. Solve it with the Hungarian algorithm to get the least-cost
     one-to-one assignment.
  4. Reject assignments whose distance exceeds `max_distance` (too far
     apart to plausibly be the same object).
  5. Update matched tracks with their assigned detection; age out
     unmatched tracks; spawn new tentative tracks for unmatched
     detections.
  6. Promote tentative tracks to "confirmed" after `min_hits` matches,
     and delete tracks that have gone unmatched for `max_age` frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hungarian import linear_sum_assignment
from kalman import KalmanFilter


@dataclass
class Track:
    track_id: int
    kf: KalmanFilter
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    confirmed: bool = False

    @property
    def position(self) -> tuple[float, float]:
        return self.kf.position


class MultiObjectTracker:
    def __init__(
        self,
        max_distance: float = 30.0,
        max_age: int = 5,
        min_hits: int = 3,
        dt: float = 1.0,
    ) -> None:
        self.max_distance = max_distance
        self.max_age = max_age
        self.min_hits = min_hits
        self.dt = dt
        self.tracks: list[Track] = []
        self._next_id = 1
        self.history: dict[int, list[tuple[float, float]]] = {}

    def _new_track(self, detection: tuple[float, float]) -> Track:
        track = Track(track_id=self._next_id, kf=KalmanFilter(detection, dt=self.dt))
        self._next_id += 1
        return track

    def step(self, detections: list[tuple[float, float]]) -> list[Track]:
        """Advance the tracker by one frame.

        Parameters
        ----------
        detections : list of (x, y) points observed this frame.

        Returns
        -------
        The list of currently confirmed tracks (after this frame's update).
        """
        # 1. Predict.
        for track in self.tracks:
            track.kf.predict()
            track.age += 1
            track.time_since_update += 1

        # 2 & 3. Cost matrix + Hungarian assignment.
        matched_track_idx = set()
        matched_det_idx = set()
        if self.tracks and detections:
            cost = np.zeros((len(self.tracks), len(detections)))
            for i, track in enumerate(self.tracks):
                tx, ty = track.position
                for j, (dx, dy) in enumerate(detections):
                    cost[i, j] = np.hypot(tx - dx, ty - dy)

            row_ind, col_ind = linear_sum_assignment(cost)

            # 4. Reject far-apart matches.
            for i, j in zip(row_ind, col_ind):
                if cost[i, j] <= self.max_distance:
                    matched_track_idx.add(i)
                    matched_det_idx.add(j)
                    track = self.tracks[i]
                    track.kf.update(detections[j])
                    track.hits += 1
                    track.time_since_update = 0
                    if track.hits >= self.min_hits:
                        track.confirmed = True

        # 5. Spawn new tracks for unmatched detections.
        for j, det in enumerate(detections):
            if j not in matched_det_idx:
                self.tracks.append(self._new_track(det))

        # 5 (cont). Drop tracks that have aged out.
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Record history of confirmed tracks for later analysis/plotting.
        for track in self.tracks:
            if track.confirmed:
                self.history.setdefault(track.track_id, []).append(track.position)

        return self.confirmed_tracks()

    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.confirmed]
