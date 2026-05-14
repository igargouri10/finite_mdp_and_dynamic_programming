r"""Smoke + correctness tests for the V3 Bayesian-HPO subpackage.

Kept deliberately small (no torch, no scipy, one-second runtime) so that
``python tests/run_tests.py`` still completes in under a few seconds.
"""

from __future__ import annotations

import math
import numpy as np

from mini1_rl.hpo import (
    BayesianOptimizer, SearchSpace,
    UniformParam, LogUniformParam, IntParam,
    GaussianProcess, rbf_kernel, expected_improvement,
    QLearningObjective,
)
from mini1_rl.experiments.masks import mask_assignment_primary


# ---------------------------------------------------------------------------
def test_uniform_param_roundtrip() -> None:
    p = UniformParam("x", 0.0, 10.0)
    for u in [0.0, 0.25, 0.5, 1.0]:
        x = p.to_native(u)
        u2 = p.to_unit(x)
        assert abs(u - u2) < 1e-9, (u, x, u2)


def test_log_uniform_param_roundtrip() -> None:
    p = LogUniformParam("alpha", 1e-3, 1.0)
    for u in [0.0, 0.5, 1.0]:
        x = p.to_native(u)
        u2 = p.to_unit(x)
        assert abs(u - u2) < 1e-9, (u, x, u2)


def test_int_param_range() -> None:
    p = IntParam("n", 1, 5)
    seen = sorted({p.to_native(u) for u in np.linspace(0, 1, 200)})
    assert seen == [1, 2, 3, 4, 5], seen


def test_search_space_lhs_is_a_permutation() -> None:
    space = SearchSpace([
        UniformParam("a", 0, 1),
        UniformParam("b", 0, 1),
    ])
    rng = np.random.default_rng(0)
    U = space.sample_lhs(8, rng)
    # Each column must hit every stratum exactly once.
    for j in range(space.dim):
        strata = (U[:, j] * 8).astype(int)
        assert sorted(strata.tolist()) == list(range(8)), strata


# ---------------------------------------------------------------------------
def test_rbf_kernel_symmetry_and_diagonal() -> None:
    X = np.random.default_rng(0).random((4, 3))
    K = rbf_kernel(X, X, length_scale=0.5, signal_var=1.7)
    # Symmetric.
    assert np.allclose(K, K.T)
    # Diagonal = signal_var.
    assert np.allclose(np.diag(K), 1.7)


def test_gp_interpolates_noise_free() -> None:
    """With tiny noise_var the GP must interpolate its training points."""
    rng = np.random.default_rng(1)
    X = rng.random((6, 2))
    y = np.sin(2 * np.pi * X[:, 0]) + 0.3 * X[:, 1]
    gp = GaussianProcess(length_scale=0.4, signal_var=1.0, noise_var=1e-8)
    gp.fit(X, y)
    mu, _ = gp.predict(X)
    assert np.allclose(mu, y, atol=1e-3), (mu, y)


def test_expected_improvement_zero_when_sigma_zero() -> None:
    mu = np.array([0.5, 1.5])
    sigma = np.array([0.0, 0.0])
    ei = expected_improvement(mu, sigma, f_best=1.0)
    assert np.allclose(ei, 0.0)


def test_expected_improvement_positive_above_incumbent() -> None:
    mu = np.array([2.0])
    sigma = np.array([0.5])
    ei = expected_improvement(mu, sigma, f_best=1.0)
    assert ei[0] > 0.0


# ---------------------------------------------------------------------------
def test_bo_finds_optimum_of_quadratic() -> None:
    """Pure unit-cube quadratic: BO should beat random search on a 2-D bowl."""
    space = SearchSpace([UniformParam("x", 0, 1), UniformParam("y", 0, 1)])

    def obj(p):
        x, y = p["x"], p["y"]
        return -((x - 0.7) ** 2 + (y - 0.3) ** 2), {}  # max at (0.7, 0.3).

    bo = BayesianOptimizer(
        space=space, objective=obj,
        n_init=4, n_iter=12, seed=0, n_candidates_per_iter=2048,
    ).run()

    assert bo.best_trial is not None
    bx = bo.best_trial.params_native["x"]
    by = bo.best_trial.params_native["y"]
    dist = math.hypot(bx - 0.7, by - 0.3)
    assert dist < 0.12, (bx, by, dist)


def test_qlearning_objective_smoke() -> None:
    """The RL objective wrapper must return a finite (scalar, dict)."""
    obj = QLearningObjective(
        mask=mask_assignment_primary(),
        terminal_rows=(4,),
        slip_prob=0.1,
        episodes=30, eval_episodes=10,
    )
    scalar, metrics = obj({"alpha": 0.1, "epsilon": 0.1, "gamma": 0.9})
    assert math.isfinite(scalar)
    assert "eval_mean_return" in metrics


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"-- {name} ...", end=" ")
            fn()
            print("OK")
