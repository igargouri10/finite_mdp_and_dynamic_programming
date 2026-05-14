r"""RL-flavoured objective wrappers used by the Bayesian-optimization loop.

Each objective is a callable ``f(params_dict) -> (scalar, metrics_dict)``,
which is the contract expected by :class:`mini1_rl.hpo.BayesianOptimizer`.

The default objective is :class:`QLearningObjective`: it trains a fresh
tabular Q-learning agent on the supplied grid-world MDP, then evaluates
the greedy policy via on-policy Monte-Carlo rollouts, and returns the
mean evaluation return as the BO objective.

Q-learning was chosen as the default tunable because (a) Q-learning is
the cheapest control algorithm in the repository, so we can afford
~25 trials in a few minutes on CPU; (b) the same hyperparameters
(``alpha``, ``epsilon``, ``gamma``) generalize to SARSA(λ) and DQN, so
the result transfers; and (c) the canonical grid-world is fully
observed, which means DQN gives the same asymptotic policy at much
higher compute cost --- using DQN as the BO inner loop would have made
the demonstrative run prohibitively long while teaching no extra
methodology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from ..mdp.gridworld import GridWorldMDP
from ..agents.td import QLearningAgent
from ..agents.mc import rollout


@dataclass
class QLearningObjective:
    """Train a Q-learning agent under ``params`` and return its mean return.

    Parameters
    ----------
    mask : 2-D reward mask used to build a fresh :class:`GridWorldMDP` each
           trial. We rebuild every trial so ``gamma`` (which lives on the
           environment, not the agent) can be tuned by BO.
    terminal_rows : passed through to :class:`GridWorldMDP`.
    slip_prob     : passed through to :class:`GridWorldMDP`.
    episodes      : training episodes per BO trial.
    eval_episodes : Monte-Carlo evaluation rollouts per trial.
    eval_seed_offset : Adds determinism to evaluation; the BO seed is
                       combined with this so that evaluation and training
                       use distinct RNG streams.
    """

    mask: np.ndarray
    terminal_rows: Tuple[int, ...] = (4,)
    slip_prob: float = 0.1
    episodes: int = 300
    eval_episodes: int = 50
    eval_seed_offset: int = 10_000
    trial_seed_fn: Optional[Callable[[Dict[str, Any]], int]] = None

    # Filled with the per-trial seed counter for reproducibility.
    _trial_count: int = field(default=0, init=False, repr=False)

    def _seed_for(self, params: Dict[str, Any]) -> int:
        if self.trial_seed_fn is not None:
            return int(self.trial_seed_fn(params))
        self._trial_count += 1
        return 1_000 + self._trial_count

    # -----------------------------------------------------------------
    def __call__(self, params: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        gamma = float(params.get("gamma", 0.9))
        alpha = float(params["alpha"])
        epsilon = float(params["epsilon"])

        env = GridWorldMDP(
            self.mask,
            gamma=gamma,
            terminal_rows=list(self.terminal_rows),
            slip_prob=self.slip_prob,
        )

        seed = self._seed_for(params)
        agent = QLearningAgent(
            alpha=alpha, epsilon=epsilon, max_steps=200, seed=seed,
        )
        train_metrics = agent.train(env, episodes=self.episodes)

        # Evaluate the greedy-induced policy via MC rollouts.
        eval_rng = np.random.default_rng(seed + self.eval_seed_offset)
        returns = [
            sum(
                r for _, _, r in rollout(
                    env, agent.policy, max_steps=200, rng=eval_rng,
                )
            )
            for _ in range(self.eval_episodes)
        ]
        mean_ret = float(np.mean(returns)) if returns else 0.0
        std_ret = float(np.std(returns)) if returns else 0.0

        metrics = {
            "train_avg_return_last_100": float(
                train_metrics.get("avg_return_last_100", 0.0)
            ),
            "eval_mean_return": mean_ret,
            "eval_std_return": std_ret,
            "eval_episodes": self.eval_episodes,
            "train_episodes": self.episodes,
        }
        return mean_ret, metrics
