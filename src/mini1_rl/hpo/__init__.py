r"""Bayesian hyperparameter optimization for mini1_rl (V3).

This subpackage adds a Gaussian-process Bayesian-optimization (BO) loop on
top of the V1/V2 agent framework. It was added in V3 because the
foundation environment of this repository is a single-protagonist
grid-world, for which Bayesian HPO is one of the few class topics that
provides a *qualitative* improvement over hand-tuning — see
``DECISIONS.md`` for the full justification.

Submodules
----------
``space``       Search-space primitives (uniform / log-uniform).
``gp``          Pure-NumPy Gaussian-process regression with the RBF kernel.
``optimizer``   The actual BO loop with Expected-Improvement acquisition.
``objectives``  Wrappers that turn an :class:`mini1_rl.agents.BaseAgent`
                training run into a scalar objective for the BO loop.

References
----------
* J. Mockus, "Bayesian Approach to Global Optimization" (1989).
* J. Snoek, H. Larochelle, R. P. Adams, "Practical Bayesian Optimization
  of Machine Learning Algorithms," NeurIPS 2012.
* P. I. Frazier, "A Tutorial on Bayesian Optimization," arXiv:1807.02811.
* C. E. Rasmussen and C. K. I. Williams, "Gaussian Processes for Machine
  Learning," MIT Press 2006.
"""

from .space import SearchSpace, UniformParam, LogUniformParam, IntParam
from .gp import GaussianProcess, rbf_kernel
from .optimizer import BayesianOptimizer, BOTrial, expected_improvement
from .objectives import QLearningObjective

__all__ = [
    "SearchSpace",
    "UniformParam",
    "LogUniformParam",
    "IntParam",
    "GaussianProcess",
    "rbf_kernel",
    "BayesianOptimizer",
    "BOTrial",
    "expected_improvement",
    "QLearningObjective",
]
