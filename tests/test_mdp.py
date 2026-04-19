"""Unit tests for the core MDP/MRP abstractions and value solvers."""

from __future__ import annotations

import numpy as np
import pytest

from mini1_rl.mdp import (
    MarkovProcess, MarkovRewardProcess,
    GridWorldMRP, GridWorldMDP,
    solve_value_matrix, solve_value_iterative, solve_value_eigendecomp,
)
from mini1_rl.experiments.masks import mask_assignment_primary


def test_markov_process_row_stochastic() -> None:
    with pytest.raises(ValueError):
        MarkovProcess(["a", "b"], np.array([[0.5, 0.4], [1.0, 0.0]]))


def test_mrp_closed_form_matches_iterative() -> None:
    mrp = GridWorldMRP(mask_assignment_primary(), gamma=0.9, terminal_rows=[4])
    V1 = solve_value_matrix(mrp)
    V2 = solve_value_iterative(mrp, tol=1e-12)
    assert np.max(np.abs(V1 - V2)) < 1e-8


def test_degenerate_swap_matrix() -> None:
    P = np.array([[0.0, 1.0], [1.0, 0.0]])
    R = np.array([1.0, -1.0])
    mrp = MarkovRewardProcess(["A", "B"], P, R, gamma=0.9)
    V1 = solve_value_matrix(mrp)
    V2 = solve_value_iterative(mrp, tol=1e-12)
    V3 = solve_value_eigendecomp(mrp)
    assert np.max(np.abs(V1 - V2)) < 1e-8
    assert np.max(np.abs(V1 - V3)) < 1e-6


def test_mdp_induced_mrp_matches_direct() -> None:
    mdp = GridWorldMDP(mask_assignment_primary(), gamma=0.9, terminal_rows=[4])
    policy = mdp.uniform_policy()
    mrp_from_policy = mdp.induced_mrp(policy)
    V_induced = solve_value_matrix(mrp_from_policy)
    # Compute directly by noting MRP = avg over actions when uniform.
    # Build an equivalent GridWorldMRP by computing its own mask-based
    # transitions and confirming agreement.
    mrp_direct = GridWorldMRP(mdp.mask, gamma=0.9, terminal_rows=[4])
    V_direct = solve_value_matrix(mrp_direct)
    assert np.max(np.abs(V_induced - V_direct)) < 1e-6
