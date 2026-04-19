r"""Grid-world MRPs and MDPs built from 2D reward masks (Mini-1, Task 4 & 6).

A 2D mask is an ``(H, W)`` array of rewards with the convention that entries
equal to ``-numpy.inf`` (the "coding choice" for :math:`-\infty`) denote walls
/ blocked cells that are *not* part of the state space. The assignment asks
us to translate such a mask into :math:`(\mathcal{S}, \mathcal{R}, P)`.

Design choices
--------------

* **Walls are excluded from** :math:`\mathcal{S}`. Otherwise we would need to
  either assign them an arbitrary reward (breaking the "mask -> MRP" contract)
  or keep them with unreachable rows in ``P`` (wasting memory and possibly
  injecting numerical junk into the Bellman solver).
* **Edges of the grid are reflective by default**: if an action would leave
  the grid or hit a wall, the agent stays in place. This is the standard
  convention from Sutton & Barto (2018) §3.5 (the original "Gridworld"
  example) and gives well-defined expected rewards even for masks that
  enclose the agent.
* **The bottom row is treated as terminal absorbing** in the Mini-1 examples
  (the "cliff" / "edge" pattern). We make that a keyword-argument so it can
  be overridden for different masks.
* We support both an MRP flavour (uniformly-random transitions over the 4
  cardinal neighbours, used for tasks 3-5) and an MDP flavour with a
  deterministic-plus-slip policy (task 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .base import MarkovDecisionProcess, MarkovRewardProcess


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    @property
    def delta(self) -> Tuple[int, int]:
        # (drow, dcol). Row 0 is the *top* of the mask.
        return {
            Action.UP: (-1, 0),
            Action.RIGHT: (0, 1),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
        }[self]


@dataclass(frozen=True)
class Cell:
    r: int
    c: int

    def __repr__(self) -> str:  # pragma: no cover
        return f"({self.r},{self.c})"


# ---------------------------------------------------------------------------
def _valid_cells(mask: np.ndarray) -> Tuple[List[Cell], Dict[Cell, int]]:
    cells: List[Cell] = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if np.isfinite(mask[r, c]):
                cells.append(Cell(r, c))
    return cells, {cell: i for i, cell in enumerate(cells)}


def _neighbours(
    cell: Cell, mask: np.ndarray, idx: Dict[Cell, int]
) -> List[Tuple[Action, Cell]]:
    """Return (action, successor) pairs; edges/walls bounce back to ``cell``."""
    out: List[Tuple[Action, Cell]] = []
    H, W = mask.shape
    for a in Action:
        dr, dc = a.delta
        nr, nc = cell.r + dr, cell.c + dc
        neigh = Cell(nr, nc)
        if 0 <= nr < H and 0 <= nc < W and neigh in idx:
            out.append((a, neigh))
        else:
            out.append((a, cell))  # bounce back
    return out


# ---------------------------------------------------------------------------
class GridWorldMRP(MarkovRewardProcess):
    r"""Random-walk MRP induced by a 2D reward mask.

    Transitions are uniform over the four cardinal neighbours (with bounces
    at edges/walls). This is the setting used in the Mini-1 value-convergence
    experiments (tasks 3-5): given the mask, compute
    :math:`V = \mathcal{R} + \gamma P V` and study how :math:`V` depends on
    :math:`\gamma`.
    """

    def __init__(
        self,
        mask: np.ndarray,
        *,
        gamma: float = 0.9,
        terminal_rows: Optional[Iterable[int]] = None,
        terminal_cells: Optional[Iterable[Cell]] = None,
        name: str = "GridWorldMRP",
    ) -> None:
        mask = np.asarray(mask, dtype=float)
        cells, idx = _valid_cells(mask)
        n = len(cells)
        P = np.zeros((n, n))
        R = np.zeros(n)
        terminal: set = set(terminal_cells or ())
        if terminal_rows:
            for r in terminal_rows:
                for c in cells:
                    if c.r == r:
                        terminal.add(c)

        for i, cell in enumerate(cells):
            R[i] = mask[cell.r, cell.c]
            if cell in terminal:
                # absorbing row: self-loop
                P[i, i] = 1.0
                continue
            neigh = _neighbours(cell, mask, idx)
            for _, s_next in neigh:
                P[i, idx[s_next]] += 0.25
        # store grid metadata on the instance
        self.mask = mask
        self.cells = cells
        self._idx = idx
        super().__init__(
            cells, P, R,
            gamma=gamma, terminal_states=terminal, name=name,
        )

    # Convenience ----------------------------------------------------------
    def reshape_values(self, V: np.ndarray) -> np.ndarray:
        """Broadcast a 1-D state-value vector back to an ``(H, W)`` grid.

        Blocked cells are filled with ``numpy.nan`` so callers can mask them
        out of imshow plots.
        """
        H, W = self.mask.shape
        grid = np.full((H, W), np.nan)
        for i, cell in enumerate(self.cells):
            grid[cell.r, cell.c] = V[i]
        return grid


# ---------------------------------------------------------------------------
class GridWorldMDP(MarkovDecisionProcess):
    r"""Controlled grid world (Mini-1 Task 6).

    Actions are the four cardinal moves; with probability :math:`1 - s`
    (``slip_prob`` defaults to 0) the agent moves in the intended direction,
    and with probability :math:`s / 3` in each of the three other directions.
    This is the standard "windy"/"slippery" parameterisation; setting
    ``slip_prob=0`` recovers a deterministic gridworld.
    """

    def __init__(
        self,
        mask: np.ndarray,
        *,
        gamma: float = 0.9,
        slip_prob: float = 0.0,
        terminal_rows: Optional[Iterable[int]] = None,
        terminal_cells: Optional[Iterable[Cell]] = None,
        name: str = "GridWorldMDP",
    ) -> None:
        mask = np.asarray(mask, dtype=float)
        cells, idx = _valid_cells(mask)
        n = len(cells)
        actions = list(Action)
        m = len(actions)
        P = np.zeros((m, n, n))
        R = np.zeros((m, n))
        terminal: set = set(terminal_cells or ())
        if terminal_rows:
            for r in terminal_rows:
                for c in cells:
                    if c.r == r:
                        terminal.add(c)

        for i, cell in enumerate(cells):
            for ai, action in enumerate(actions):
                R[ai, i] = mask[cell.r, cell.c]
                if cell in terminal:
                    P[ai, i, i] = 1.0
                    continue
                neigh = dict(_neighbours(cell, mask, idx))
                # intended direction
                p_intended = 1.0 - slip_prob
                P[ai, i, idx[neigh[action]]] += p_intended
                if slip_prob > 0:
                    for other in actions:
                        if other != action:
                            P[ai, i, idx[neigh[other]]] += slip_prob / 3.0

        self.mask = mask
        self.cells = cells
        self._idx = idx
        super().__init__(
            cells, actions, P, R,
            gamma=gamma, terminal_states=terminal, name=name,
        )

    def reshape_values(self, V: np.ndarray) -> np.ndarray:
        H, W = self.mask.shape
        grid = np.full((H, W), np.nan)
        for i, cell in enumerate(self.cells):
            grid[cell.r, cell.c] = V[i]
        return grid

    # Task-specific utility: uniform random policy ------------------------
    def uniform_policy(self) -> np.ndarray:
        return np.full((self.num_states, self.num_actions), 1.0 / self.num_actions)
