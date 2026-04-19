r"""Every-visit Monte-Carlo control agent.

Follows the generalised policy-iteration pattern from Sutton & Barto (2018)
Ch. 5: generate an episode with an :math:`\varepsilon`-soft policy, compute
returns, update :math:`Q(s, a)` by averaging, improve the policy :math:`\varepsilon`-greedily.

Because this is a finite MDP we know the full state list from ``env.states``;
for more complex environments the tabular Q-table would be replaced by a
function approximator, which is exactly what the V2 deep RL agents do.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..mdp.base import MarkovDecisionProcess
from .base import BaseAgent


def epsilon_greedy_policy(Q: np.ndarray, epsilon: float) -> np.ndarray:
    r""":math:`\varepsilon`-greedy policy as a :math:`(|S|, |A|)` array."""
    m, n = Q.shape  # (|A|, |S|)
    pi = np.full((n, m), epsilon / m)
    best = np.argmax(Q, axis=0)
    pi[np.arange(n), best] += 1.0 - epsilon
    return pi


def sample_action(pi_row: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(len(pi_row), p=pi_row))


def rollout(
    mdp: MarkovDecisionProcess,
    pi: np.ndarray,
    *,
    start: Optional[int] = None,
    max_steps: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[int, int, float]]:
    if rng is None:
        rng = np.random.default_rng()
    s = start if start is not None else int(rng.integers(mdp.num_states))
    traj: List[Tuple[int, int, float]] = []
    for _ in range(max_steps):
        a = sample_action(pi[s], rng)
        s_state = mdp.states[s]
        if mdp.is_terminal(s_state):
            break
        probs = mdp.P[a, s]
        s_next = int(rng.choice(mdp.num_states, p=probs))
        r = float(mdp.R[a, s])
        traj.append((s, a, r))
        s = s_next
    return traj


class MonteCarloAgent(BaseAgent):
    algo = "monte_carlo_control"

    def __init__(
        self,
        *,
        epsilon: float = 0.1,
        max_steps: int = 500,
        seed: int = 0,
    ) -> None:
        super().__init__(epsilon=epsilon, max_steps=max_steps, seed=seed)
        self.Q: Optional[np.ndarray] = None
        self.policy: Optional[np.ndarray] = None
        self._N: Optional[np.ndarray] = None

    def _ensure(self, mdp: MarkovDecisionProcess) -> None:
        if self.Q is None:
            self.Q = np.zeros((mdp.num_actions, mdp.num_states))
            self._N = np.zeros_like(self.Q)
            self.policy = np.full(
                (mdp.num_states, mdp.num_actions), 1.0 / mdp.num_actions
            )

    def train(
        self,
        env: MarkovDecisionProcess,
        *,
        episodes: int = 500,
        **_: Any,
    ) -> Dict[str, Any]:
        self._ensure(env)
        rng = np.random.default_rng(self.hyperparameters["seed"])
        eps = self.hyperparameters["epsilon"]
        returns_log = []
        for ep in range(episodes):
            traj = rollout(
                env, self.policy,
                max_steps=self.hyperparameters["max_steps"], rng=rng,
            )
            G = 0.0
            # every-visit MC: walk backwards
            for (s, a, r) in reversed(traj):
                G = r + env.gamma * G
                self._N[a, s] += 1
                self.Q[a, s] += (G - self.Q[a, s]) / self._N[a, s]
            self.policy = epsilon_greedy_policy(self.Q, eps)
            returns_log.append(sum(r for _, _, r in traj))
        self.record.episodes_trained += episodes
        self.record.metrics = {
            "avg_return_last_100": float(np.mean(returns_log[-100:]))
            if returns_log else 0.0,
        }
        return self.record.metrics

    def evaluate(
        self, env: MarkovDecisionProcess, *, episodes: int = 50, **_: Any
    ) -> Dict[str, Any]:
        if self.policy is None:
            raise RuntimeError("Call train() first")
        rng = np.random.default_rng(self.hyperparameters["seed"] + 1)
        returns = []
        for _ in range(episodes):
            traj = rollout(
                env, self.policy,
                max_steps=self.hyperparameters["max_steps"], rng=rng,
            )
            returns.append(sum(r for _, _, r in traj))
        return {"avg_return": float(np.mean(returns)) if returns else 0.0}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "hyperparameters": self.hyperparameters,
            "Q": self.Q, "policy": self.policy, "N": self._N,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.hyperparameters = state["hyperparameters"]
        self.Q = state["Q"]; self.policy = state["policy"]; self._N = state["N"]
