r"""Temporal-difference family: TD(n), TD(λ), SARSA(n), SARSA(λ), Q-learning.

Each class follows the same pattern as :class:`MonteCarloAgent` so that
``train(env, episodes=...)`` and ``evaluate(env)`` are drop-in compatible.

Convention for n-step returns (Sutton & Barto 2018, Ch. 7):

.. math::
   G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n}
              + \gamma^n V(S_{t+n}).

Convention for eligibility traces (Sutton & Barto 2018, Ch. 12):

.. math::
   E_t(s) = \gamma \lambda E_{t-1}(s) + \mathbb{1}[S_t = s]
   \quad(\text{accumulating trace}).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..mdp.base import MarkovDecisionProcess
from .base import BaseAgent
from .mc import epsilon_greedy_policy, sample_action


def _greedy_row(Q: np.ndarray, s: int) -> int:
    return int(np.argmax(Q[:, s]))


# ===========================================================================
#  Expected-value methods on a given policy (no control).
# ===========================================================================
class TDnAgent(BaseAgent):
    """Forward-view n-step TD for state values under a fixed policy."""

    algo = "td_n"

    def __init__(
        self,
        *,
        n: int = 3,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        max_steps: int = 500,
        seed: int = 0,
    ) -> None:
        super().__init__(
            n=n, alpha=alpha, epsilon=epsilon,
            max_steps=max_steps, seed=seed,
        )
        self.V: Optional[np.ndarray] = None
        self.policy: Optional[np.ndarray] = None

    def _ensure(self, mdp: MarkovDecisionProcess) -> None:
        if self.V is None:
            self.V = np.zeros(mdp.num_states)
            self.policy = np.full(
                (mdp.num_states, mdp.num_actions), 1.0 / mdp.num_actions
            )

    def train(
        self, env: MarkovDecisionProcess, *, episodes: int = 500, **_: Any
    ) -> Dict[str, Any]:
        self._ensure(env)
        rng = np.random.default_rng(self.hyperparameters["seed"])
        n = self.hyperparameters["n"]
        alpha = self.hyperparameters["alpha"]
        gamma = env.gamma
        for ep in range(episodes):
            s = int(rng.integers(env.num_states))
            traj_r: List[float] = []
            traj_s: List[int] = [s]
            T = np.inf
            t = 0
            while True:
                if t < T:
                    a = sample_action(self.policy[traj_s[t]], rng)
                    probs = env.P[a, traj_s[t]]
                    s_next = int(rng.choice(env.num_states, p=probs))
                    r = float(env.R[a, traj_s[t]])
                    traj_r.append(r)
                    traj_s.append(s_next)
                    if env.is_terminal(env.states[s_next]) or t + 1 >= self.hyperparameters["max_steps"]:
                        T = t + 1
                tau = t - n + 1
                if tau >= 0:
                    G = 0.0
                    for i in range(tau + 1, min(tau + n, T) + 1):
                        G += gamma ** (i - tau - 1) * traj_r[i - 1]
                    if tau + n < T:
                        G += gamma ** n * self.V[traj_s[tau + n]]
                    self.V[traj_s[tau]] += alpha * (G - self.V[traj_s[tau]])
                t += 1
                if tau == T - 1:
                    break
        self.record.episodes_trained += episodes
        return {"V_mean": float(self.V.mean())}

    def evaluate(self, env: MarkovDecisionProcess, **_: Any) -> Dict[str, Any]:
        return {"V_mean": float(self.V.mean())}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "hyperparameters": self.hyperparameters,
            "V": self.V, "policy": self.policy,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.hyperparameters = state["hyperparameters"]
        self.V = state["V"]; self.policy = state["policy"]


class TDLambdaAgent(BaseAgent):
    """Backward-view TD(λ) with accumulating eligibility traces."""

    algo = "td_lambda"

    def __init__(
        self,
        *,
        lam: float = 0.9,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        max_steps: int = 500,
        seed: int = 0,
    ) -> None:
        super().__init__(
            lam=lam, alpha=alpha, epsilon=epsilon,
            max_steps=max_steps, seed=seed,
        )
        self.V: Optional[np.ndarray] = None
        self.policy: Optional[np.ndarray] = None

    def _ensure(self, mdp: MarkovDecisionProcess) -> None:
        if self.V is None:
            self.V = np.zeros(mdp.num_states)
            self.policy = np.full(
                (mdp.num_states, mdp.num_actions), 1.0 / mdp.num_actions
            )

    def train(
        self, env: MarkovDecisionProcess, *, episodes: int = 500, **_: Any
    ) -> Dict[str, Any]:
        self._ensure(env)
        rng = np.random.default_rng(self.hyperparameters["seed"])
        lam = self.hyperparameters["lam"]
        alpha = self.hyperparameters["alpha"]
        gamma = env.gamma
        for ep in range(episodes):
            E = np.zeros(env.num_states)
            s = int(rng.integers(env.num_states))
            for _ in range(self.hyperparameters["max_steps"]):
                if env.is_terminal(env.states[s]):
                    break
                a = sample_action(self.policy[s], rng)
                probs = env.P[a, s]
                s_next = int(rng.choice(env.num_states, p=probs))
                r = float(env.R[a, s])
                delta = r + gamma * self.V[s_next] - self.V[s]
                E *= gamma * lam
                E[s] += 1.0
                self.V += alpha * delta * E
                s = s_next
        self.record.episodes_trained += episodes
        return {"V_mean": float(self.V.mean())}

    def evaluate(self, env: MarkovDecisionProcess, **_: Any) -> Dict[str, Any]:
        return {"V_mean": float(self.V.mean())}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "hyperparameters": self.hyperparameters,
            "V": self.V, "policy": self.policy,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.hyperparameters = state["hyperparameters"]
        self.V = state["V"]; self.policy = state["policy"]


# ===========================================================================
#  SARSA family (on-policy Q-value control).
# ===========================================================================
class SarsaNAgent(BaseAgent):
    """Forward-view n-step SARSA (Sutton & Barto 2018, Ch. 7)."""

    algo = "sarsa_n"

    def __init__(
        self,
        *,
        n: int = 3,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        max_steps: int = 500,
        seed: int = 0,
    ) -> None:
        super().__init__(
            n=n, alpha=alpha, epsilon=epsilon,
            max_steps=max_steps, seed=seed,
        )
        self.Q: Optional[np.ndarray] = None
        self.policy: Optional[np.ndarray] = None

    def _ensure(self, mdp: MarkovDecisionProcess) -> None:
        if self.Q is None:
            self.Q = np.zeros((mdp.num_actions, mdp.num_states))
            self.policy = epsilon_greedy_policy(
                self.Q, self.hyperparameters["epsilon"]
            )

    def train(
        self, env: MarkovDecisionProcess, *, episodes: int = 500, **_: Any
    ) -> Dict[str, Any]:
        self._ensure(env)
        rng = np.random.default_rng(self.hyperparameters["seed"])
        n = self.hyperparameters["n"]
        alpha = self.hyperparameters["alpha"]
        eps = self.hyperparameters["epsilon"]
        gamma = env.gamma
        returns_log = []
        for ep in range(episodes):
            states = [int(rng.integers(env.num_states))]
            actions = [sample_action(self.policy[states[0]], rng)]
            rewards: List[float] = [0.0]  # placeholder so indices line up
            T = np.inf
            t = 0
            ep_return = 0.0
            while True:
                if t < T:
                    probs = env.P[actions[t], states[t]]
                    s_next = int(rng.choice(env.num_states, p=probs))
                    r = float(env.R[actions[t], states[t]])
                    ep_return += r
                    states.append(s_next); rewards.append(r)
                    if env.is_terminal(env.states[s_next]) or t + 1 >= self.hyperparameters["max_steps"]:
                        T = t + 1
                    else:
                        actions.append(sample_action(self.policy[s_next], rng))
                tau = t - n + 1
                if tau >= 0:
                    G = 0.0
                    for i in range(tau + 1, min(tau + n, T) + 1):
                        G += gamma ** (i - tau - 1) * rewards[i]
                    if tau + n < T:
                        G += gamma ** n * self.Q[actions[tau + n], states[tau + n]]
                    a_tau, s_tau = actions[tau], states[tau]
                    self.Q[a_tau, s_tau] += alpha * (G - self.Q[a_tau, s_tau])
                    self.policy = epsilon_greedy_policy(self.Q, eps)
                t += 1
                if tau == T - 1:
                    break
            returns_log.append(ep_return)
        self.record.episodes_trained += episodes
        self.record.metrics = {
            "avg_return_last_100": float(np.mean(returns_log[-100:]))
            if returns_log else 0.0,
        }
        return self.record.metrics

    def evaluate(
        self, env: MarkovDecisionProcess, *, episodes: int = 50, **_: Any
    ) -> Dict[str, Any]:
        from .mc import rollout
        rng = np.random.default_rng(self.hyperparameters["seed"] + 1)
        returns = [
            sum(r for _, _, r in rollout(env, self.policy, max_steps=self.hyperparameters["max_steps"], rng=rng))
            for _ in range(episodes)
        ]
        return {"avg_return": float(np.mean(returns)) if returns else 0.0}

    def state_dict(self) -> Dict[str, Any]:
        return {"hyperparameters": self.hyperparameters, "Q": self.Q, "policy": self.policy}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.hyperparameters = state["hyperparameters"]
        self.Q = state["Q"]; self.policy = state["policy"]


class SarsaLambdaAgent(BaseAgent):
    """Backward-view SARSA(λ) with accumulating eligibility traces."""

    algo = "sarsa_lambda"

    def __init__(
        self,
        *,
        lam: float = 0.9,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        max_steps: int = 500,
        seed: int = 0,
    ) -> None:
        super().__init__(
            lam=lam, alpha=alpha, epsilon=epsilon,
            max_steps=max_steps, seed=seed,
        )
        self.Q: Optional[np.ndarray] = None
        self.policy: Optional[np.ndarray] = None

    def _ensure(self, mdp: MarkovDecisionProcess) -> None:
        if self.Q is None:
            self.Q = np.zeros((mdp.num_actions, mdp.num_states))
            self.policy = epsilon_greedy_policy(
                self.Q, self.hyperparameters["epsilon"]
            )

    def train(
        self, env: MarkovDecisionProcess, *, episodes: int = 500, **_: Any
    ) -> Dict[str, Any]:
        self._ensure(env)
        rng = np.random.default_rng(self.hyperparameters["seed"])
        lam = self.hyperparameters["lam"]
        alpha = self.hyperparameters["alpha"]
        eps = self.hyperparameters["epsilon"]
        gamma = env.gamma
        returns_log = []
        for ep in range(episodes):
            E = np.zeros_like(self.Q)
            s = int(rng.integers(env.num_states))
            a = sample_action(self.policy[s], rng)
            ep_return = 0.0
            for _ in range(self.hyperparameters["max_steps"]):
                if env.is_terminal(env.states[s]):
                    break
                probs = env.P[a, s]
                s_next = int(rng.choice(env.num_states, p=probs))
                r = float(env.R[a, s])
                ep_return += r
                a_next = sample_action(self.policy[s_next], rng)
                delta = r + gamma * self.Q[a_next, s_next] - self.Q[a, s]
                E *= gamma * lam
                E[a, s] += 1.0
                self.Q += alpha * delta * E
                self.policy = epsilon_greedy_policy(self.Q, eps)
                s, a = s_next, a_next
            returns_log.append(ep_return)
        self.record.episodes_trained += episodes
        self.record.metrics = {
            "avg_return_last_100": float(np.mean(returns_log[-100:]))
            if returns_log else 0.0,
        }
        return self.record.metrics

    def evaluate(
        self, env: MarkovDecisionProcess, *, episodes: int = 50, **_: Any
    ) -> Dict[str, Any]:
        from .mc import rollout
        rng = np.random.default_rng(self.hyperparameters["seed"] + 1)
        returns = [
            sum(r for _, _, r in rollout(env, self.policy, max_steps=self.hyperparameters["max_steps"], rng=rng))
            for _ in range(episodes)
        ]
        return {"avg_return": float(np.mean(returns)) if returns else 0.0}

    def state_dict(self) -> Dict[str, Any]:
        return {"hyperparameters": self.hyperparameters, "Q": self.Q, "policy": self.policy}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.hyperparameters = state["hyperparameters"]
        self.Q = state["Q"]; self.policy = state["policy"]


# ===========================================================================
#  Q-learning (off-policy TD control; Watkins & Dayan 1992).
# ===========================================================================
class QLearningAgent(BaseAgent):
    algo = "q_learning"

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        max_steps: int = 500,
        seed: int = 0,
    ) -> None:
        super().__init__(alpha=alpha, epsilon=epsilon, max_steps=max_steps, seed=seed)
        self.Q: Optional[np.ndarray] = None
        self.policy: Optional[np.ndarray] = None

    def _ensure(self, mdp: MarkovDecisionProcess) -> None:
        if self.Q is None:
            self.Q = np.zeros((mdp.num_actions, mdp.num_states))
            self.policy = epsilon_greedy_policy(
                self.Q, self.hyperparameters["epsilon"]
            )

    def train(
        self, env: MarkovDecisionProcess, *, episodes: int = 500, **_: Any
    ) -> Dict[str, Any]:
        self._ensure(env)
        rng = np.random.default_rng(self.hyperparameters["seed"])
        alpha = self.hyperparameters["alpha"]
        eps = self.hyperparameters["epsilon"]
        gamma = env.gamma
        returns_log = []
        for ep in range(episodes):
            s = int(rng.integers(env.num_states))
            ep_return = 0.0
            for _ in range(self.hyperparameters["max_steps"]):
                if env.is_terminal(env.states[s]):
                    break
                a = sample_action(self.policy[s], rng)
                probs = env.P[a, s]
                s_next = int(rng.choice(env.num_states, p=probs))
                r = float(env.R[a, s])
                ep_return += r
                td_target = r + gamma * self.Q[:, s_next].max()
                self.Q[a, s] += alpha * (td_target - self.Q[a, s])
                self.policy = epsilon_greedy_policy(self.Q, eps)
                s = s_next
            returns_log.append(ep_return)
        self.record.episodes_trained += episodes
        self.record.metrics = {
            "avg_return_last_100": float(np.mean(returns_log[-100:]))
            if returns_log else 0.0,
            "returns_log": returns_log,
        }
        return self.record.metrics

    def evaluate(
        self, env: MarkovDecisionProcess, *, episodes: int = 50, **_: Any
    ) -> Dict[str, Any]:
        from .mc import rollout
        rng = np.random.default_rng(self.hyperparameters["seed"] + 1)
        returns = [
            sum(r for _, _, r in rollout(env, self.policy, max_steps=self.hyperparameters["max_steps"], rng=rng))
            for _ in range(episodes)
        ]
        return {"avg_return": float(np.mean(returns)) if returns else 0.0}

    def state_dict(self) -> Dict[str, Any]:
        return {"hyperparameters": self.hyperparameters, "Q": self.Q, "policy": self.policy}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.hyperparameters = state["hyperparameters"]
        self.Q = state["Q"]; self.policy = state["policy"]
