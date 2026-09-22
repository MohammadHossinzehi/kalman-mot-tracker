import itertools

import numpy as np
import pytest

from hungarian import linear_sum_assignment


def brute_force_assignment(cost):
    """Reference implementation: try every permutation. Only used in tests
    on small matrices to check the fast algorithm's optimality."""
    n, m = cost.shape
    size = min(n, m)
    rows = list(range(n))
    best_cost = None
    best_assignment = None
    for cols in itertools.permutations(range(m), size):
        for row_subset in itertools.combinations(rows, size):
            total = sum(cost[r, c] for r, c in zip(row_subset, cols))
            if best_cost is None or total < best_cost:
                best_cost = total
                best_assignment = list(zip(row_subset, cols))
    return best_cost, best_assignment


def total_cost(cost, row_ind, col_ind):
    return float(cost[row_ind, col_ind].sum())


def test_square_matrix_matches_known_optimum():
    cost = np.array(
        [
            [4, 1, 3],
            [2, 0, 5],
            [3, 2, 2],
        ],
        dtype=float,
    )
    row_ind, col_ind = linear_sum_assignment(cost)
    assert total_cost(cost, row_ind, col_ind) == pytest.approx(5.0)
    assert sorted(col_ind.tolist()) == [0, 1, 2]


def test_more_rows_than_columns():
    cost = np.array(
        [
            [10, 1],
            [2, 8],
            [7, 3],
        ],
        dtype=float,
    )
    row_ind, col_ind = linear_sum_assignment(cost)
    assert len(row_ind) == 2  # min(3, 2)
    got = total_cost(cost, row_ind, col_ind)
    best, _ = brute_force_assignment(cost)
    assert got == pytest.approx(best)


def test_more_columns_than_rows():
    cost = np.array(
        [
            [10, 1, 6, 9],
            [2, 8, 4, 5],
        ],
        dtype=float,
    )
    row_ind, col_ind = linear_sum_assignment(cost)
    assert len(row_ind) == 2  # min(2, 4)
    got = total_cost(cost, row_ind, col_ind)
    best, _ = brute_force_assignment(cost)
    assert got == pytest.approx(best)


def test_matches_brute_force_on_random_matrices():
    rng = np.random.default_rng(42)
    for trial in range(15):
        n = rng.integers(2, 4)
        m = rng.integers(2, 4)
        cost = rng.uniform(0, 20, size=(n, m))
        row_ind, col_ind = linear_sum_assignment(cost)
        got = total_cost(cost, row_ind, col_ind)
        best, _ = brute_force_assignment(cost)
        assert got == pytest.approx(best, abs=1e-6), f"trial {trial} failed: {cost}"


def test_assignment_is_one_to_one():
    rng = np.random.default_rng(1)
    cost = rng.uniform(0, 10, size=(5, 5))
    row_ind, col_ind = linear_sum_assignment(cost)
    assert len(set(row_ind.tolist())) == len(row_ind)
    assert len(set(col_ind.tolist())) == len(col_ind)


def test_empty_input():
    row_ind, col_ind = linear_sum_assignment(np.zeros((0, 3)))
    assert len(row_ind) == 0 and len(col_ind) == 0
