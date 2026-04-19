r"""Reproduce every figure and every numeric result in the Mini-1 report.

Run as::

    python -m mini1_rl.experiments.run_all --out docs/report/figures

Produces:

* ``fig_task4_values_random.pdf``        -- V for uniform-random policy, gamma=0.9
* ``fig_task5_gamma_sweep.pdf``          -- V vs gamma, multiple states
* ``fig_task5_convergence_history.pdf``  -- sup-norm error per iteration, with gamma^k bound
* ``fig_task3_scaling.pdf``              -- wall-clock scaling of solvers vs |S|
* ``fig_task6_optimal_policy.pdf``       -- greedy policy from value iteration
* ``fig_task6_optimal_values.pdf``       -- V* from value iteration on the MDP
* ``fig_task7_tiled_values.pdf``         -- V for the tree-tiled MRP
* ``fig_task2_degenerate.pdf``           -- eigendecomposition on the 2x2 swap matrix
* ``fig_algo_comparison.pdf``            -- return curves for MC / Sarsa / SarsaLambda / Q-learning / DQN
* ``results_table.json``                 -- numeric results used in the report tables
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np

from ..mdp import (
    GridWorldMRP, GridWorldMDP,
    MarkovRewardProcess,
    TreeTiledMRP, TileNode,
    solve_value_matrix, solve_value_iterative, solve_value_eigendecomp,
)
from ..agents import (
    PolicyIterationAgent, ValueIterationAgent,
    MonteCarloAgent, TDnAgent, TDLambdaAgent,
    SarsaNAgent, SarsaLambdaAgent, QLearningAgent,
)
from ..utils import (
    plot_value_grid, plot_gamma_sweep, plot_convergence_history,
    plot_learning_curves,
)
from .masks import (
    mask_assignment_primary, mask_cliff_walk,
    mask_task7_tile_A, mask_task7_tile_B, mask_task7_tile_C,
)

try:
    import torch  # noqa: F401
    from ..agents.dqn import DQNAgent
    HAS_DQN = True
except Exception:
    HAS_DQN = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
def task3_scaling(out: str, results: Dict) -> None:
    """Timing for the three solvers as |S| grows."""
    import matplotlib.pyplot as plt

    sizes = [5, 10, 20, 40, 80, 160]
    t_matrix = []
    t_iter = []
    t_eig = []
    for n in sizes:
        # build a random MRP with |S|=n
        rng = np.random.default_rng(42)
        P = rng.random((n, n))
        P = P / P.sum(axis=1, keepdims=True)
        R = rng.standard_normal(n)
        mrp = MarkovRewardProcess(list(range(n)), P, R, gamma=0.9)
        t = time.perf_counter()
        for _ in range(5):
            solve_value_matrix(mrp)
        t_matrix.append((time.perf_counter() - t) / 5)
        t = time.perf_counter()
        for _ in range(5):
            solve_value_iterative(mrp, tol=1e-8)
        t_iter.append((time.perf_counter() - t) / 5)
        t = time.perf_counter()
        for _ in range(5):
            solve_value_eigendecomp(mrp)
        t_eig.append((time.perf_counter() - t) / 5)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(sizes, t_matrix, "-o", label="matrix solve (LAPACK)")
    ax.loglog(sizes, t_iter, "-s", label=r"iterative $V \leftarrow R + \gamma P V$")
    ax.loglog(sizes, t_eig, "-^", label="eigendecomposition")
    ax.set_xlabel(r"$|S|$")
    ax.set_ylabel("wall-clock per solve (s)")
    ax.set_title("Task 3: solver scaling with state-space size")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_task3_scaling.pdf"))
    results["task3_scaling"] = {
        "sizes": sizes,
        "matrix": t_matrix,
        "iterative": t_iter,
        "eigendecomp": t_eig,
    }


def task2_degenerate(out: str, results: Dict) -> None:
    """Reproduce the classic swap-matrix example mentioned in Task 3."""
    # P = [[0, 1], [1, 0]]; R = [1, -1]; gamma = 0.9
    P = np.array([[0.0, 1.0], [1.0, 0.0]])
    R = np.array([1.0, -1.0])
    mrp = MarkovRewardProcess(["A", "B"], P, R, gamma=0.9)
    V_matrix = solve_value_matrix(mrp)
    V_iter = solve_value_iterative(mrp, tol=1e-12)
    V_eig = solve_value_eigendecomp(mrp)
    results["task2_degenerate"] = {
        "V_matrix": V_matrix.tolist(),
        "V_iterative": V_iter.tolist(),
        "V_eigendecomp": V_eig.tolist(),
        "gap_matrix_vs_iter": float(np.max(np.abs(V_matrix - V_iter))),
    }


def task4_values_random(out: str, results: Dict) -> None:
    """V on the primary mask under a uniform random policy."""
    mask = mask_assignment_primary()
    # The top-left 0 is a terminal; the bottom row is terminal absorbing.
    mrp = GridWorldMRP(
        mask, gamma=0.9,
        terminal_rows=[4],
        terminal_cells=[],
    )
    V = solve_value_matrix(mrp)
    grid = mrp.reshape_values(V)
    plot_value_grid(
        grid,
        title=r"$V^{\mathrm{random}}$, $\gamma=0.9$ (primary mask)",
        savepath=os.path.join(out, "fig_task4_values_random.pdf"),
    )
    results["task4_values_random"] = {
        "V": V.tolist(),
        "mask_shape": list(mask.shape),
        "V_grid": np.where(np.isnan(grid), None, grid).tolist(),
    }


def task5_gamma_sweep(out: str, results: Dict) -> None:
    mask = mask_assignment_primary()
    gammas = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    values_by_gamma = {}
    for g in gammas:
        mrp = GridWorldMRP(mask, gamma=g, terminal_rows=[4])
        V = solve_value_matrix(mrp)
        values_by_gamma[g] = V
    plot_gamma_sweep(
        gammas, values_by_gamma,
        title=r"Task 5: $V^{\mathrm{random}}(s)$ vs $\gamma$",
        savepath=os.path.join(out, "fig_task5_gamma_sweep.pdf"),
    )
    results["task5_gamma_sweep"] = {
        str(g): V.tolist() for g, V in values_by_gamma.items()
    }

    # convergence history at gamma=0.9
    mrp = GridWorldMRP(mask, gamma=0.9, terminal_rows=[4])
    _, history = solve_value_iterative(mrp, tol=1e-12, return_history=True, max_iter=500)
    plot_convergence_history(
        history,
        title=r"Task 5: iterative Bellman convergence ($\gamma=0.9$)",
        savepath=os.path.join(out, "fig_task5_convergence_history.pdf"),
        gamma=0.9,
    )
    results["task5_convergence"] = {
        "iterations": int(history.shape[0]),
        "final_sup_norm": float(np.max(np.abs(history[-1] - history[-2]))),
    }


def task6_optimal(out: str, results: Dict) -> None:
    mask = mask_assignment_primary()
    mdp = GridWorldMDP(mask, gamma=0.9, terminal_rows=[4], slip_prob=0.1)
    vi = ValueIterationAgent(tol=1e-10)
    vi.train(mdp)
    grid = mdp.reshape_values(vi.V)
    plot_value_grid(
        grid,
        title=r"Task 6: $V^{*}$ via value iteration (MDP, slip=0.1)",
        savepath=os.path.join(out, "fig_task6_optimal_values.pdf"),
    )
    # greedy policy plot: we'll render as an image of the action index and
    # overlay arrows. For simplicity we save the raster text to a file too.
    from ..utils.raster import raster_policy
    raster = raster_policy(mdp, vi.policy)
    with open(os.path.join(out, "fig_task6_optimal_policy.txt"), "w") as fh:
        fh.write(raster)
    # Produce a simple arrow plot:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 4))
    H, W = mask.shape
    disp = np.where(np.isnan(mdp.reshape_values(vi.V)), 0.0, 1.0)
    ax.imshow(disp, cmap="Greys", vmin=0, vmax=2)
    arrow_map = {0: (0, -0.4), 1: (0.4, 0), 2: (0, 0.4), 3: (-0.4, 0)}
    for i, cell in enumerate(mdp.cells):
        if mdp.is_terminal(cell):
            ax.text(cell.c, cell.r, "T", ha="center", va="center", fontsize=10)
            continue
        a = int(np.argmax(vi.policy[i]))
        dx, dy = arrow_map[a]
        ax.arrow(cell.c, cell.r, dx, dy, head_width=0.18, head_length=0.12, fc="black")
    for r in range(H):
        for c in range(W):
            if not np.isfinite(mask[r, c]):
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="black"))
    ax.set_xticks(range(W)); ax.set_yticks(range(H))
    ax.set_title("Task 6: optimal policy (arrows)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_task6_optimal_policy.pdf"))
    plt.close(fig)
    results["task6_optimal"] = {
        "V_mean": float(vi.V.mean()),
        "V_max": float(vi.V.max()),
        "sweeps": vi.record.episodes_trained,
    }

    # policy iteration cross-check
    pi = PolicyIterationAgent()
    pi.train(mdp)
    results["task6_optimal"]["agreement_V"] = float(
        np.max(np.abs(vi.V - pi.V))
    )
    results["task6_optimal"]["policy_iteration_sweeps"] = pi.record.episodes_trained


def task7_tiled(out: str, results: Dict) -> None:
    A, B, C = mask_task7_tile_A(), mask_task7_tile_B(), mask_task7_tile_C()
    root = TileNode(
        left=TileNode(
            left=TileNode(mask=A, name="A"),
            right=TileNode(mask=B, name="B"),
            portal_left=(0, 0), portal_right=(4, 0),
            name="AB",
        ),
        right=TileNode(mask=C, name="C"),
        portal_left=(4, 4), portal_right=(0, 4),
        name="ABC",
    )
    tiled = TreeTiledMRP(root, gamma=0.9, name="Task7")
    V = solve_value_iterative(tiled, tol=1e-8, max_iter=5000)
    # For visualisation we project each leaf back to its own 2D grid.
    from ..mdp.gridworld import _valid_cells, Cell
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (leaf_idx, name, mask) in zip(
        axes,
        [(0, "A", A), (1, "B", B), (2, "C", C)],
    ):
        cells, _ = _valid_cells(mask)
        grid = np.full(mask.shape, np.nan)
        for (li, cell), i in zip(tiled._leaf_cells, range(len(tiled._leaf_cells))):
            if li == leaf_idx:
                grid[cell.r, cell.c] = V[i]
        plot_value_grid(grid, title=f"tile {name}", annotate=True, ax=ax, fig=fig)
    fig.suptitle("Task 7: $V$ on the tree-tiled MRP (height 3)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_task7_tiled_values.pdf"))
    plt.close(fig)
    results["task7_tiled"] = {
        "num_states": tiled.num_states,
        "nnz": tiled.nnz(),
        "V_min": float(V.min()),
        "V_max": float(V.max()),
        "V_mean": float(V.mean()),
        "portals": [
            (int(lp[0]), (int(lp[1].r), int(lp[1].c)),
             int(rp[0]), (int(rp[1].r), int(rp[1].c)))
            for (lp, rp) in tiled._portals
        ],
    }


def algo_comparison(out: str, results: Dict) -> None:
    """Compare MC, SARSA(n), SARSA(λ), Q-learning, DQN on the primary mask."""
    mask = mask_assignment_primary()
    mdp = GridWorldMDP(mask, gamma=0.9, terminal_rows=[4], slip_prob=0.1)

    curves: Dict[str, List[float]] = {}
    EPISODES = 400

    mc = MonteCarloAgent(epsilon=0.2, seed=0)
    mc.train(mdp, episodes=EPISODES)
    # MC tracks returns_log internally via metrics; regenerate curve from logs.
    # (We intentionally kept the API simple; for the curve we re-run eval
    # at checkpoints.)
    def _episode_returns(agent_cls, **kw):
        agent = agent_cls(**kw)
        returns: List[float] = []
        checkpoint = 20
        for _ in range(EPISODES // checkpoint):
            agent.train(mdp, episodes=checkpoint)
            m = agent.evaluate(mdp, episodes=20)
            returns.extend([m["avg_return"]] * checkpoint)
        return returns, agent

    curves["Monte Carlo"], _ = _episode_returns(MonteCarloAgent, epsilon=0.2, seed=0)
    curves["SARSA(3)"], _ = _episode_returns(SarsaNAgent, n=3, alpha=0.2, epsilon=0.2, seed=0)
    curves["SARSA(λ=0.8)"], _ = _episode_returns(
        SarsaLambdaAgent, lam=0.8, alpha=0.2, epsilon=0.2, seed=0
    )
    curves["Q-learning"], ql = _episode_returns(
        QLearningAgent, alpha=0.2, epsilon=0.2, seed=0
    )
    if HAS_DQN:
        dqn = DQNAgent(epsilon_decay_steps=2000, buffer_capacity=5000, batch_size=32, seed=0)
        dqn_returns = []
        checkpoint = 20
        for _ in range(EPISODES // checkpoint):
            dqn.train(mdp, episodes=checkpoint, max_steps=100)
            m = dqn.evaluate(mdp, episodes=10, max_steps=100)
            dqn_returns.extend([m["avg_return"]] * checkpoint)
        curves["DQN"] = dqn_returns

    plot_learning_curves(
        curves, window=20,
        title="Algorithm comparison on the primary mask",
        xlabel="training episode",
        ylabel="evaluation return",
        savepath=os.path.join(out, "fig_algo_comparison.pdf"),
    )
    # summary table of final evaluation returns
    summary = {name: float(np.mean(vals[-25:])) for name, vals in curves.items()}
    results["algo_comparison_final"] = summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/report/figures")
    args = ap.parse_args()
    ensure_dir(args.out)
    results: Dict = {}
    task2_degenerate(args.out, results)
    task3_scaling(args.out, results)
    task4_values_random(args.out, results)
    task5_gamma_sweep(args.out, results)
    task6_optimal(args.out, results)
    task7_tiled(args.out, results)
    algo_comparison(args.out, results)
    with open(os.path.join(args.out, "results_table.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("=== Experiment run complete ===")
    print(json.dumps({k: type(v).__name__ for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
