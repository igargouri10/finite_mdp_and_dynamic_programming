from .base import BaseAgent, AgentRecord
from .dp import PolicyIterationAgent, ValueIterationAgent, q_from_v
from .mc import MonteCarloAgent
from .td import TDnAgent, TDLambdaAgent, SarsaNAgent, SarsaLambdaAgent, QLearningAgent

__all__ = [
    "BaseAgent", "AgentRecord",
    "PolicyIterationAgent", "ValueIterationAgent", "q_from_v",
    "MonteCarloAgent",
    "TDnAgent", "TDLambdaAgent",
    "SarsaNAgent", "SarsaLambdaAgent",
    "QLearningAgent",
]

try:
    from .dqn import DQNAgent  # noqa: F401
    __all__.append("DQNAgent")
except ImportError:  # pragma: no cover -- torch not installed
    pass
