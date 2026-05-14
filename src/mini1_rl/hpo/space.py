r"""Search-space primitives for Bayesian hyperparameter optimization.

A :class:`SearchSpace` is a named tuple of one-dimensional parameter
distributions. The BO loop works internally on the *unit cube*
:math:`[0, 1]^d`; each parameter is responsible for translating a unit-cube
coordinate into the native parameter value and back. This mirrors the
design used by every mainstream BO library (Spearmint, scikit-optimize,
Optuna) and decouples the GP surrogate from the parameter semantics.

The three parameter shapes covered here are sufficient for the
hyperparameters of the tabular and DQN agents in this repository:

* :class:`UniformParam`     -- continuous, uniform prior (e.g. ``epsilon``).
* :class:`LogUniformParam`  -- continuous, log-uniform prior (e.g. ``alpha``).
* :class:`IntParam`         -- integer, uniform prior (e.g. ``n``-step).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Single-parameter primitives.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UniformParam:
    """Continuous parameter with a uniform prior on ``[low, high]``."""

    name: str
    low: float
    high: float

    def to_native(self, u: float) -> float:
        u = float(np.clip(u, 0.0, 1.0))
        return self.low + (self.high - self.low) * u

    def to_unit(self, x: float) -> float:
        return (x - self.low) / (self.high - self.low)


@dataclass(frozen=True)
class LogUniformParam:
    """Continuous parameter with a log-uniform prior on ``[low, high]``."""

    name: str
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low <= 0 or self.high <= 0:
            raise ValueError("LogUniformParam requires strictly positive bounds.")

    def to_native(self, u: float) -> float:
        u = float(np.clip(u, 0.0, 1.0))
        log_lo, log_hi = np.log(self.low), np.log(self.high)
        return float(np.exp(log_lo + (log_hi - log_lo) * u))

    def to_unit(self, x: float) -> float:
        log_lo, log_hi = np.log(self.low), np.log(self.high)
        return (np.log(x) - log_lo) / (log_hi - log_lo)


@dataclass(frozen=True)
class IntParam:
    """Discrete integer parameter on ``[low, high]`` inclusive."""

    name: str
    low: int
    high: int

    def to_native(self, u: float) -> int:
        u = float(np.clip(u, 0.0, 1.0))
        # Inclusive integer mapping: split [0, 1] into (high-low+1) buckets.
        n_buckets = self.high - self.low + 1
        idx = int(min(n_buckets - 1, np.floor(u * n_buckets)))
        return self.low + idx

    def to_unit(self, x: int) -> float:
        n_buckets = self.high - self.low + 1
        return (x - self.low + 0.5) / n_buckets


Param = Any  # union over the three above; kept as Any for simplicity.


# ---------------------------------------------------------------------------
# A bag of parameters with batch transforms.
# ---------------------------------------------------------------------------
@dataclass
class SearchSpace:
    """An ordered collection of named hyperparameter distributions."""

    params: List[Param] = field(default_factory=list)

    @property
    def dim(self) -> int:
        return len(self.params)

    @property
    def names(self) -> List[str]:
        return [p.name for p in self.params]

    def to_native_dict(self, u: np.ndarray) -> Dict[str, Any]:
        """Map a unit-cube point to a ``{name: native_value}`` dict."""
        u = np.asarray(u, dtype=float)
        if u.shape != (self.dim,):
            raise ValueError(f"Expected shape ({self.dim},), got {u.shape}.")
        return {p.name: p.to_native(float(u[i])) for i, p in enumerate(self.params)}

    def to_unit_array(self, native: Dict[str, Any]) -> np.ndarray:
        """Inverse of :meth:`to_native_dict`."""
        return np.array(
            [p.to_unit(native[p.name]) for p in self.params], dtype=float
        )

    # -----------------------------------------------------------------
    # Random and Latin-Hypercube samplers (initial design).
    # -----------------------------------------------------------------
    def sample_random(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``n`` uniform samples from the unit cube; shape ``(n, dim)``."""
        return rng.random((n, self.dim))

    def sample_lhs(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Latin-Hypercube sample in the unit cube; shape ``(n, dim)``.

        The classical LHS construction (McKay, Beckman & Conover 1979):
        split each dimension into ``n`` equal-width strata and place
        exactly one sample in each. Within each stratum the sample is
        drawn uniformly; the per-dimension assignments are independently
        permuted.
        """
        if n < 1:
            raise ValueError("LHS requires n >= 1.")
        # (n,) strata centres, jittered uniformly inside each stratum.
        strata = (np.arange(n) + rng.random(n)) / n
        out = np.empty((n, self.dim), dtype=float)
        for j in range(self.dim):
            out[:, j] = rng.permutation(strata)
        return out
