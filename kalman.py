"""
Constant-velocity Kalman filter for 2D point tracking.

State vector: [x, y, vx, vy]  (position and velocity in each axis)
Measurement:  [x, y]          (position only; velocity is inferred)

This is a textbook linear Kalman filter (Kalman, 1960). It is used as the
per-track motion model in the multi-object tracker in tracker.py: each
track owns one KalmanFilter instance, predicts forward every frame, and
gets corrected whenever a detection is associated to it by the Hungarian
algorithm in hungarian.py.
"""
from __future__ import annotations

import numpy as np


class KalmanFilter:
    """Linear Kalman filter with a constant-velocity motion model."""

    def __init__(
        self,
        initial_position: tuple[float, float],
        dt: float = 1.0,
        process_var: float = 1.0,
        measurement_var: float = 10.0,
    ) -> None:
        x0, y0 = initial_position

        # State: [x, y, vx, vy]^T. Velocity starts at zero.
        self.x = np.array([x0, y0, 0.0, 0.0], dtype=float)

        # State transition model: constant velocity.
        self.F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

        # Measurement model: we observe position only.
        self.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=float,
        )

        # Process noise covariance, derived from a discretized white-noise
        # acceleration model (standard formulation, e.g. Bar-Shalom et al.).
        q = process_var
        dt2 = dt * dt
        dt3 = dt2 * dt / 2.0
        dt4 = dt2 * dt2 / 4.0
        self.Q = q * np.array(
            [
                [dt4, 0, dt3, 0],
                [0, dt4, 0, dt3],
                [dt3, 0, dt2, 0],
                [0, dt3, 0, dt2],
            ],
            dtype=float,
        )

        # Measurement noise covariance.
        self.R = np.eye(2, dtype=float) * measurement_var

        # State covariance: high initial uncertainty on velocity, since we
        # cannot observe it directly from a single detection.
        self.P = np.diag([measurement_var, measurement_var, 1000.0, 1000.0]).astype(float)

        self.dt = dt

    def predict(self) -> np.ndarray:
        """Advance the state one time step. Returns the predicted state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, measurement: tuple[float, float]) -> np.ndarray:
        """Correct the state using an observed (x, y) position."""
        z = np.asarray(measurement, dtype=float)
        y = z - self.H @ self.x  # innovation
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        # Joseph form for numerical stability of the covariance update.
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T
        return self.x

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    def position_uncertainty(self) -> float:
        """Trace of the position block of the covariance, a scalar summary
        of how confident the filter is about the current position."""
        return float(self.P[0, 0] + self.P[1, 1])
