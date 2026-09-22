# Kalman + Hungarian Multi-Object Tracker

A from-scratch implementation of the "tracking-by-detection" approach used
by real multi-object trackers such as SORT: each tracked object gets its
own Kalman filter for motion prediction, and a new frame's detections are
assigned to existing tracks by solving a linear assignment problem with
the Hungarian (Kuhn-Munkres) algorithm. No `scipy`, `filterpy`, or
`opencv` — the Kalman filter and the assignment solver are both written
from first principles in `kalman.py` and `hungarian.py`.

## What it does and why it's useful

Given a stream of noisy, unlabeled 2D detections per frame (the kind of
output an object detector produces, minus any identity information), the
tracker answers the question every downstream application actually
needs answered: *which detection belongs to which object, frame over
frame?* It has to do this while objects:

- move with unknown but roughly constant velocity,
- occasionally go undetected for a few frames (occlusion, a missed
  detection),
- cross paths with each other, and
- share the scene with spurious false-positive detections ("clutter").

This is the same core problem behind pedestrian tracking in autonomous
driving, ball/player tracking in sports analytics, and multi-target radar
tracking. The three pieces here are genuinely reusable in isolation:

- `kalman.py` — a linear Kalman filter with a constant-velocity model.
  Point it at any 2D position stream and it will predict + smooth motion.
- `hungarian.py` — a general-purpose O(n²·m) solver for the rectangular
  linear assignment problem (minimum-cost bipartite matching), useful
  anywhere you need to optimally pair two sets of things by cost, not
  just tracking.
- `tracker.py` — glues the two together with track lifecycle management
  (tentative → confirmed → deleted), the same pattern used by SORT and
  its descendants (DeepSORT, ByteTrack, etc., which add appearance
  features and other refinements on top of this same skeleton).

## How to run it

```bash
pip install -r requirements.txt

# Run the full test suite (18 tests: Kalman filter math, assignment
# optimality against brute force, and tracker lifecycle behavior).
pytest -q

# Run the synthetic-scene demo: 3 moving objects, sensor noise, missed
# detections, clutter, and one deliberate occlusion event.
python simulate.py

# Same, but also save a plot of ground truth vs. tracked trajectories.
python simulate.py --plot demo.png
```

Example output:

```
=== Kalman + Hungarian Multi-Object Tracker: simulation report ===
frames simulated:            60
true objects in scene:       3
tracks ever created:         20
tracks that became confirmed: 7
mean position error (px):    3.08
```

`tracks ever created` is higher than `true objects` by design: every
unmatched detection (clutter, or a real object right after it drops out
of occlusion) spawns a new *tentative* track, and only tracks that
accumulate `min_hits` consecutive-ish matches get promoted to
*confirmed* and reported. Short-lived tentative tracks from clutter are
expected and are filtered out of the reported results — this mirrors how
production trackers avoid reporting noise as an object.

## Design decisions and testing

**Constant-velocity motion model.** The Kalman filter's state is
`[x, y, vx, vy]`; the state transition matrix assumes velocity is
constant between frames and lets process noise (`Q`) account for the
error that assumption introduces. This is the standard choice for
short-horizon tracking and is what SORT itself uses; a more complex
model (constant acceleration, or a learned motion model) would trade
implementation simplicity for accuracy on more erratic motion.

**Hungarian algorithm, not greedy nearest-neighbor.** A greedy
"assign each track to its closest detection" approach can produce a
provably suboptimal global assignment when tracks compete for the same
nearby detections (exactly what happens when two objects' paths cross).
The Hungarian algorithm guarantees the assignment that minimizes total
distance cost across *all* tracks simultaneously. `tests/test_hungarian.py`
checks this against a brute-force permutation search on both square and
non-square cost matrices, plus 15 random trials.

**Gating by `max_distance`.** Even the optimal assignment can pair a
track with a detection that's implausibly far away (e.g., when a track
has no real match this frame). Any assignment above `max_distance` is
rejected and treated as "no match," which is what lets the tracker
correctly spawn a new track instead of snapping onto an unrelated
detection.

**Known limitation, demonstrated honestly.** When two objects' paths
actually cross near-simultaneously with detections falling roughly
equidistant from both predicted positions, a pure distance-based
associator can swap identities between them — visible in `demo.png` as
one track jumping from a straight-line path onto the crossing circular
path. This is a well-documented failure mode of SORT-style trackers
(it's the reason DeepSORT added an appearance/re-ID feature on top of
this same skeleton), and it's left in rather than hidden, since it's an
honest demonstration of what motion-only, no-appearance tracking can and
can't do.

**Testing approach.** 18 tests across three files:
- `tests/test_kalman.py` — filter converges to the true position and
  velocity on a noiseless constant-velocity trajectory, uncertainty
  shrinks monotonically with repeated updates, and the covariance matrix
  stays symmetric and positive semi-definite (a correctness invariant
  that's easy to silently break with an unstable covariance update
  formula — this implementation uses the numerically-stable Joseph form
  specifically to guard against that).
- `tests/test_hungarian.py` — optimality verified against brute-force
  search, plus shape handling for non-square cost matrices and the
  empty-input edge case.
- `tests/test_tracker.py` — end-to-end lifecycle behavior: stable IDs
  for a single smoothly-moving object, correct track spawning for
  well-separated objects, tracks surviving brief occlusion but being
  dropped after `max_age` consecutive misses, and clutter never
  accumulating enough hits to be falsely confirmed.

## Files

```
kalman.py           Kalman filter (constant-velocity model)
hungarian.py         Hungarian algorithm (rectangular linear assignment)
tracker.py           MultiObjectTracker: track lifecycle + data association
simulate.py           Synthetic scene generator + demo/report/plot CLI
tests/test_kalman.py
tests/test_hungarian.py
tests/test_tracker.py
requirements.txt
```
