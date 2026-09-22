import numpy as np
import pytest

from kalman import KalmanFilter


def test_predict_moves_state_by_zero_when_no_velocity():
    kf = KalmanFilter((10.0, 20.0))
    kf.predict()
    x, y = kf.position
    assert x == pytest.approx(10.0)
    assert y == pytest.approx(20.0)


def test_update_pulls_state_toward_measurement():
    kf = KalmanFilter((0.0, 0.0), measurement_var=1.0)
    kf.predict()
    kf.update((10.0, 0.0))
    x, y = kf.position
    # After one update the filter should have moved substantially toward
    # the measurement, but a single low-variance measurement plus a wide
    # prior won't land exactly on it.
    assert 0.0 < x <= 10.0
    assert y == pytest.approx(0.0, abs=1e-6)


def test_converges_to_constant_velocity_track():
    """Feed noiseless measurements along a straight line at constant
    velocity; after a few steps the filter should track position and
    velocity accurately."""
    kf = KalmanFilter((0.0, 0.0), dt=1.0, process_var=0.01, measurement_var=0.01)
    true_vx, true_vy = 2.0, -1.5
    for t in range(1, 30):
        kf.predict()
        kf.update((true_vx * t, true_vy * t))

    x, y = kf.position
    vx, vy = kf.velocity
    assert x == pytest.approx(true_vx * 29, abs=0.5)
    assert y == pytest.approx(true_vy * 29, abs=0.5)
    assert vx == pytest.approx(true_vx, abs=0.1)
    assert vy == pytest.approx(true_vy, abs=0.1)


def test_position_uncertainty_shrinks_with_repeated_updates():
    kf = KalmanFilter((0.0, 0.0), measurement_var=5.0)
    initial_uncertainty = kf.position_uncertainty()
    for t in range(1, 10):
        kf.predict()
        kf.update((float(t), float(t)))
    final_uncertainty = kf.position_uncertainty()
    assert final_uncertainty < initial_uncertainty


def test_covariance_stays_symmetric_positive_semidefinite():
    kf = KalmanFilter((0.0, 0.0))
    rng = np.random.default_rng(0)
    for _ in range(50):
        kf.predict()
        kf.update((rng.normal(), rng.normal()))
        # Symmetry.
        assert np.allclose(kf.P, kf.P.T, atol=1e-6)
        # PSD: all eigenvalues non-negative (within numerical tolerance).
        eigvals = np.linalg.eigvalsh(kf.P)
        assert np.all(eigvals > -1e-6)
