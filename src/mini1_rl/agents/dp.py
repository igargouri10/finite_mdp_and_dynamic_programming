r"""Dynamic Programming agents (V1, requirement 4).

Implements policy iteration (Howard 1960) and value iteration (Bellman 1957)
on a fully-known MDP. The two agents share a common helper for computing
:math:`Q(s, a)` from :math:`V(s)` via one-step lookahead:

.. math::
   Q(s, a) = \mathcal{R}^a_s + \gamma \sum_{s'} P^a_{ss'} V(s').
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..mdp.base import MarkovDecisionProcess
from .base import BaseAgent


def q_from_v(mdp: MarkovDecisionProcess, V: np.ndarray) -> np.ndarray:
    r"""Return Q[a, s] = R[a, s] + gamma * sum_{s'} P[a, s, s'] V[s']."""
    # P has shape (|A|, |S|, |S|); V has shape (|S|,).
    return mdp.R + mdp.gamma * (mdp.P @ V)


def greedy_policy_from_q(Q: np.ndarray) -> np.ndarray:
    """Deterministic greedy policy as a one-hot :math:`(|S|, |A|)` matrix."""
    m, n = Q.shape  # (|A|, |S|)
    best = np.argmax(Q, axis=0)  # (|S|,)
    pi = np.zeros((n, m))
    pi[np.arange(n), best] = 1.0
    return pi


# ---------------------------------------------------------------------------
class PolicyIterationAgent(BaseAgent):
    algo = "policy_iteration"

    def __init__(self, *, tol: float = 1e-10, max_sweeps: int = 10_000) -> None:
        super().__init__(tol=tol, max_sweeps=max_sweeps)
        self.V: Optional[np.ndarray] = None
        self.policy: Optional[np.ndarray] = None

    def policy_evaluation(
        self, mdp: MarkovDecisionProcess, policy: np.ndarray
    ) -> np.ndarray:
        """Solve V^pi closed-form via the induced MRP."""
        mrp = mdp.induced_mrp(policy)
        n = mdp.num_states
        A = np.eye(n) - mrp.gamma * mrp.P
        return np.linalg.solve(A, mrp.R)

    def policy_improvement(
        self, mdp: MarkovDecisionProcess, V: np.ndarray
    ) -> np.ndarray:
        Q = q_from_v(mdp, V)
        return greedy_policy_from_q(Q)

    # ------------------------------------------------------------------
    def train(self, env: MarkovDecisionProcess, **_: Any) -> Dict[str, Any]:
        mdp = env
        # start with a uniform random policy
        policy = np.full((mdp.num_states, mdp.num_actions), 1.0 / mdp.num_actions)
        V = np.zeros(mdp.num_states)
        sweeps = 0
        for sweeps in range(1, self.hyperparameters["max_sweeps"] + 1):
            V = self.policy_evaluation(mdp, policy)
            new_policy = self.policy_improvement(mdp, V)
            if np.array_equal(policy, new_policy):
                break
            policy = new_policy
        self.V = V
        self.policy = policy
        self.record.episodes_trained = sweeps
        self.record.metrics = {"sweeps_to_converge": sweeps}
        return self.record.metrics

    def evaluate(self, env: MarkovDecisionProcess, **_: Any) -> Dict[str, Any]:
        if self.V is None:
            raise RuntimeError("Call train() first")
        return {"V_mean": float(self.V.mean()), "V_max": float(self.V.max())}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "hyperparameters": self.hyperparameters, "V": self.V, "policy": self.policy,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.hyperparameters = state["hyperparameters"]
        self.V = state["V"]
        self.policy = state["policy"]


# ---------------------------------------------------------------------------
class ValueIterationAgent(BaseAgent):
    algo = "value_iteration"

    def __init__(self, *, tol: float = 1e-10, max_sweeps: int = 10_000) -> None:
        super().__init__(tol=tol, max_sweeps=max_sweeps)
        self.V: Optional[np.ndarray] = None
        self.policy: Optional[np.ndarray] = None

    def train(self, env: MarkovDecisionProcess, **_: Any) -> Dict[str, Any]:
        mdp = env
        V = np.zeros(mdp.num_states)
        for sweep in range(1, self.hyperparameters["max_sweeps"] + 1):
            Q = q_from_v(mdp, V)          # (|A|, |S|)
            V_new = Q.max(axis=0)         # Bellman optimality
            delta = np.max(np.abs(V_new - V))
            V = V_new
            if delta < self.hyperparameters["tol"]:
                break
        self.V = V
        self.policy = greedy_policy_from_q(q_from_v(mdp, V))
        self.record.episodes_trained = sweep
        self.record.metrics = {"sweeps_to_converge": sweep}
        return self.record.metrics

    def evaluate(self, env: MarkovDecisionProcess, **_: Any) -> Dict[str, Any]:
        if self.V is None:
            raise RuntimeError("Call train() first")
        return {"V_mean": float(self.V.mean()), "V_max": float(self.V.max())}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "hyperparameters": self.hyperparameters, "V": self.V, "policy": self.policy,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.hyperparameters = state["hyperparameters"]
        self.V = state["V"]
        self.policy = state["policy"]
