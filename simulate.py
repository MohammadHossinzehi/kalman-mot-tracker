"""
End-to-end demo: generate a synthetic scene of moving 2D objects with
noisy detections, missed detections, false-positive clutter, and one
occlusion event, run them through the MultiObjectTracker, evaluate
tracking quality, and (optionally) render a plot of ground truth vs.
tracked trajectories.

Run:
    python simulate.py                # prints a metrics report
    python simulate.py --plot out.png # also saves a trajectory plot
"""
from __future__ import annotations

import argparse
import numpy as np

from tracker import MultiObjectTracker


def generate_scene(
    n_frames: int = 60,
    seed: int = 7,
    detection_noise: float = 2.0,
    miss_prob: float = 0.05,
    clutter_rate: float = 0.3,
):
    """Simulate a few objects moving on straight/curved paths.

    Returns
    -------
    ground_truth : dict[obj_id] -> list of (frame, x, y)
    frames_detections : list[frame] -> list of (x, y) noisy detections
        (a mix of true detections, misses, and clutter false positives)
    """
    rng = np.random.default_rng(seed)

    # Three objects: one moving in a straight line, one curving, one
    # crossing paths with the first to stress-test data association.
    def straight(t, x0, y0, vx, vy):
        return x0 + vx * t, y0 + vy * t

    def curve(t, cx, cy, r, w, phase):
        return cx + r * np.cos(w * t + phase), cy + r * np.sin(w * t + phase)

    object_paths = {
        1: lambda t: straight(t, 0.0, 50.0, 3.0, 0.4),
        2: lambda t: curve(t, 100.0, 100.0, 40.0, 0.12, 0.0),
        3: lambda t: straight(t, 180.0, 10.0, -2.5, 1.5),
    }

    # Object 2 disappears for a stretch of frames to simulate occlusion.
    occluded_frames = set(range(25, 32))

    ground_truth: dict[int, list[tuple[int, float, float]]] = {k: [] for k in object_paths}
    frames_detections: list[list[tuple[float, float]]] = []

    for t in range(n_frames):
        dets = []
        for obj_id, path in object_paths.items():
            x, y = path(float(t))
            ground_truth[obj_id].append((t, x, y))

            if obj_id == 2 and t in occluded_frames:
                continue  # occluded: no detection this frame
            if rng.random() < miss_prob:
                continue  # random missed detection

            nx = x + rng.normal(0, detection_noise)
            ny = y + rng.normal(0, detection_noise)
            dets.append((nx, ny))

        # Poisson-distributed clutter (false positive detections).
        n_clutter = rng.poisson(clutter_rate)
        for _ in range(n_clutter):
            dets.append((rng.uniform(-20, 220), rng.uniform(-20, 170)))

        rng.shuffle(dets)
        frames_detections.append(dets)

    return ground_truth, frames_detections


def nearest_gt_distance(pos, gt_positions_at_t):
    if not gt_positions_at_t:
        return float("inf")
    dists = [np.hypot(pos[0] - g[0], pos[1] - g[1]) for g in gt_positions_at_t]
    return min(dists)


def run(n_frames=60, plot_path=None, verbose=True):
    ground_truth, frames_detections = generate_scene(n_frames=n_frames)
    tracker = MultiObjectTracker(max_distance=20.0, max_age=6, min_hits=4)

    # Per-frame ground truth positions (for quick nearest-neighbor error).
    gt_by_frame = [
        [ (obj[t][1], obj[t][2]) for obj in ground_truth.values() ]
        for t in range(n_frames)
    ]

    total_confirmed_track_ids = set()
    per_frame_errors = []

    for t in range(n_frames):
        confirmed = tracker.step(frames_detections[t])
        for track in confirmed:
            total_confirmed_track_ids.add(track.track_id)
            err = nearest_gt_distance(track.position, gt_by_frame[t])
            if err != float("inf"):
                per_frame_errors.append(err)

    mean_error = float(np.mean(per_frame_errors)) if per_frame_errors else float("nan")
    n_true_objects = len(ground_truth)
    n_tracks_created = tracker._next_id - 1

    if verbose:
        print("=== Kalman + Hungarian Multi-Object Tracker: simulation report ===")
        print(f"frames simulated:            {n_frames}")
        print(f"true objects in scene:       {n_true_objects}")
        print(f"tracks ever created:         {n_tracks_created}")
        print(f"tracks that became confirmed:{len(total_confirmed_track_ids)}")
        print(f"mean position error (px):    {mean_error:.3f}")
        print(
            "note: tracks created > true objects is expected -- clutter and "
            "the occlusion event both cause extra tentative/short-lived tracks."
        )

    if plot_path:
        render_plot(ground_truth, tracker.history, plot_path)
        if verbose:
            print(f"trajectory plot saved to:    {plot_path}")

    return {
        "n_true_objects": n_true_objects,
        "n_tracks_created": n_tracks_created,
        "n_confirmed_tracks": len(total_confirmed_track_ids),
        "mean_error": mean_error,
    }


def render_plot(ground_truth, history, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    for obj_id, points in ground_truth.items():
        xs = [p[1] for p in points]
        ys = [p[2] for p in points]
        ax.plot(xs, ys, "--", color="gray", linewidth=1.5, alpha=0.7)
        ax.annotate(f"gt {obj_id}", (xs[0], ys[0]), color="gray")

    colors = plt.cm.tab10.colors
    for i, (track_id, points) in enumerate(history.items()):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        color = colors[track_id % len(colors)]
        ax.plot(xs, ys, "-", color=color, linewidth=2, label=f"track {track_id}")

    ax.set_title("Ground truth (dashed) vs. Kalman-filtered tracks (solid)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=60, help="number of frames to simulate")
    parser.add_argument("--plot", type=str, default=None, help="path to save a trajectory PNG")
    args = parser.parse_args()
    run(n_frames=args.frames, plot_path=args.plot)


if __name__ == "__main__":
    main()
