r"""The canonical 2D masks used throughout the Mini-1 experiments.

These are exactly the masks from the assignment. We also define a couple of
variants that stress-test the solver (pure random walk on a ring, a periodic
two-state chain, a diagonal corridor, and the three tiles for Task 7).
"""

from __future__ import annotations

import numpy as np


NEG_INF = -np.inf


def mask_assignment_primary() -> np.ndarray:
    r"""The 5x5 mask from Mini-1 Task 4."""
    return np.array(
        [
            [      0, NEG_INF,   -1,      -1,    -1],
            [     -1, NEG_INF,   -1, NEG_INF,    -1],
            [     -1, NEG_INF,   -1, NEG_INF,    -1],
            [     -1,      -1,   -1, NEG_INF,    -1],
            [     -3,      -3,   -3,      -3,    -3],
        ],
        dtype=float,
    )


def mask_cliff_walk() -> np.ndarray:
    r"""A 4x8 cliff-walk style mask. The bottom row is terminal absorbing."""
    m = np.full((4, 8), -1.0)
    m[3, :] = -100.0
    m[3, 0] = 0.0    # start
    m[3, -1] = 0.0   # goal -- terminal
    return m


def mask_task7_tile_A() -> np.ndarray:
    r"""Tile A from the Task 7 illustration (5x5)."""
    return np.array(
        [
            [-6,      -5,       -4, NEG_INF,  -3],
            [-7, NEG_INF,       -3,      -3,  -3],
            [-8, NEG_INF,  NEG_INF, NEG_INF,  -9],
            [NEG_INF, -13, -14,   NEG_INF,  -10],
            [-11, -12, -15, -16,  -16],
        ],
        dtype=float,
    )


def mask_task7_tile_B() -> np.ndarray:
    r"""Tile B from the Task 7 illustration (5x5)."""
    return np.array(
        [
            [ 2,  5,       -5,  NEG_INF,  30],
            [ 2, NEG_INF, -35, NEG_INF,  20],
            [NEG_INF, NEG_INF, -40, NEG_INF, 20],
            [10, 10,      20,      20,  20],
            [10, NEG_INF, NEG_INF,  30, NEG_INF],
        ],
        dtype=float,
    )


def mask_task7_tile_C() -> np.ndarray:
    r"""Tile C from the Task 7 illustration (5x5)."""
    return np.array(
        [
            [-1, -1, -1, NEG_INF, -1],
            [NEG_INF, NEG_INF, -1, -11, -1],
            [10000, NEG_INF, -1, NEG_INF, -1],
            [-50, NEG_INF, -1, NEG_INF, -1],
            [-20, NEG_INF, -1, -1, -1],
        ],
        dtype=float,
    )
