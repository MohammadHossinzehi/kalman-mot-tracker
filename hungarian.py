"""
Hungarian algorithm (Kuhn-Munkres) for the rectangular linear assignment
problem: given an (n x m) cost matrix, find the assignment of rows to
columns that minimizes total cost, assigning min(n, m) pairs.

Implemented from scratch using the Jonker-Volgenant style shortest
augmenting path formulation with dual potentials, which runs in
O(n^2 * m) time and avoids the O(n^4) naive augmenting-path version.
This is the classical algorithm described in:

  Kuhn, H.W. (1955). "The Hungarian Method for the assignment problem."
  Jonker, R. and Volgenant, A. (1987). "A shortest augmenting path
    algorithm for dense and sparse linear assignment problems."

Used by tracker.py to solve the data-association problem between
predicted track positions and new detections each frame.
"""
from __future__ import annotations

import numpy as np

INF = float("inf")


def linear_sum_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve the rectangular assignment problem.

    Parameters
    ----------
    cost_matrix : (n, m) array of non-negative costs. Need not be square;
        internally it is padded with zero-cost dummy rows/columns so that
        exactly min(n, m) real pairs are matched.

    Returns
    -------
    row_ind, col_ind : 1-D arrays of matched indices such that
        cost_matrix[row_ind[k], col_ind[k]] is the k-th assigned pair,
        sorted by row_ind. len(row_ind) == min(n, m).
    """
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    n, m = cost_matrix.shape
    if n == 0 or m == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    transposed = n > m
    if transposed:
        cost_matrix = cost_matrix.T
        n, m = cost_matrix.shape

    # Pad to square with zero-cost dummy columns so every real row gets
    # matched to *something* internally; dummy matches are filtered out
    # at the end. (n <= m after the transpose above.)
    size = m
    cost = np.zeros((size, size), dtype=float)
    cost[:n, :] = cost_matrix

    # Dual potentials for rows (u) and columns (v).
    u = np.zeros(size + 1)
    v = np.zeros(size + 1)
    # col_match[j] = row currently assigned to column j (1-indexed, 0 = free)
    col_match = np.zeros(size + 1, dtype=int)

    for i in range(1, size + 1):
        col_match[0] = i
        j0 = 0
        min_to = np.full(size + 1, INF)
        used = np.zeros(size + 1, dtype=bool)
        way = np.zeros(size + 1, dtype=int)

        while True:
            used[j0] = True
            i0 = col_match[j0]
            delta = INF
            j1 = -1
            for j in range(1, size + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < min_to[j]:
                        min_to[j] = cur
                        way[j] = j0
                    if min_to[j] < delta:
                        delta = min_to[j]
                        j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[col_match[j]] += delta
                    v[j] -= delta
                else:
                    min_to[j] -= delta
            j0 = j1
            if col_match[j0] == 0:
                break

        while j0 != 0:
            j1 = way[j0]
            col_match[j0] = col_match[j1]
            j0 = j1

    # col_match[j] (1-indexed) = row assigned to column j-1 (0-indexed).
    row_ind = []
    col_ind = []
    for j in range(1, size + 1):
        i = col_match[j]
        if i - 1 < n and (j - 1) < m:
            row_ind.append(i - 1)
            col_ind.append(j - 1)

    row_ind = np.array(row_ind, dtype=int)
    col_ind = np.array(col_ind, dtype=int)
    order = np.argsort(row_ind)
    row_ind, col_ind = row_ind[order], col_ind[order]

    if transposed:
        row_ind, col_ind = col_ind, row_ind
        order = np.argsort(row_ind)
        row_ind, col_ind = row_ind[order], col_ind[order]

    return row_ind, col_ind
