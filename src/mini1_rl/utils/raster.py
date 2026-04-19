r"""Rasterization utilities (``raster_mode=human``, V1 requirement).

These functions convert a ``(H, W)`` grid of values or a ``(|S|, |A|)``
policy into a printable / plottable representation of the grid world.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..mdp.gridworld import Action, GridWorldMDP, GridWorldMRP


def raster_values(env: GridWorldMRP | GridWorldMDP, V: np.ndarray) -> str:
    grid = env.reshape_values(V)
    lines = []
    for row in grid:
        lines.append(
            " ".join(
                "   # " if np.isnan(v) else f"{v:6.2f}" for v in row
            )
        )
    return "\n".join(lines)


def raster_policy(env: GridWorldMDP, policy: np.ndarray) -> str:
    r"""Render a policy as ASCII arrows over the grid."""
    H, W = env.mask.shape
    arrows = {Action.UP: "^", Action.RIGHT: ">", Action.DOWN: "v", Action.LEFT: "<"}
    grid = np.full((H, W), " ", dtype="<U1")
    for i, cell in enumerate(env.cells):
        if env.is_terminal(cell):
            grid[cell.r, cell.c] = "T"
        else:
            a = int(np.argmax(policy[i]))
            grid[cell.r, cell.c] = arrows[Action(a)]
    # draw walls
    for r in range(H):
        for c in range(W):
            if not np.isfinite(env.mask[r, c]):
                grid[r, c] = "#"
    return "\n".join(" ".join(row) for row in grid)
