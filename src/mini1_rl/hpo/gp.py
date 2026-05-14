r"""Pure-NumPy Gaussian-process regression with an RBF kernel.

This is intentionally a minimal, textbook implementation of the GP
surrogate used inside :class:`mini1_rl.hpo.BayesianOptimizer`. We do not
depend on ``scipy`` or ``sklearn`` so the V3 BO module slots into the
existing CPU-only pipeline.

The mathematical content is Section 2.2 of Rasmussen & Williams (2006):

.. math::

   \mu(x_*) = k(x_*, X)\,\bigl(K + \sigma_n^2 I\bigr)^{-1} y \\
   \sigma^2(x_*) = k(x_*, x_*)
                   - k(x_*, X)\,\bigl(K + \sigma_n^2 I\bigr)^{-1}\,k(X, x_*)

with RBF kernel

.. math::
   k(x, x') = \sigma_f^2 \exp\!\bigl(-\tfrac{1}{2\,\ell^2}\,\|x - x'\|^2\bigr).

The three kernel hyperparameters :math:`(\sigma_f, \ell, \sigma_n)` are
not optimized via type-II MLE in this implementation; instead they use
fixed-but-reasonable defaults derived from the empirical scale of the
training targets. This matches Snoek et al. (2012) §3 in spirit but
removes the optimization layer that is brittle on small datasets. The
small Mini-1 grid worlds rarely give us more than ~25 BO trials per
sweep, which is exactly the regime where MLE on the kernel
hyperparameters is unreliable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


def rbf_kernel(
    X1: np.ndarray, X2: np.ndarray, length_scale: float, signal_var: float
) -> np.ndarray:
    """RBF/Gaussian kernel matrix; ``X1`` is ``(n1, d)``, ``X2`` is ``(n2, d)``."""
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    # Squared Euclidean distances via the (a-b)^2 = a^2 - 2 ab + b^2 trick.
    sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)            # (n1, 1)
    sq2 = np.sum(X2 ** 2, axis=1, keepdims=True).T          # (1,  n2)
    sqd = sq1 + sq2 - 2.0 * X1 @ X2.T
    sqd = np.maximum(sqd, 0.0)                              # numerical floor
    return signal_var * np.exp(-0.5 * sqd / (length_scale ** 2))


@dataclass
class GaussianProcess:
    """Zero-mean GP regression with a fixed RBF kernel.

    Parameters
    ----------
    length_scale : float
        Kernel length-scale :math:`\\ell`.
    signal_var : float
        Kernel signal variance :math:`\\sigma_f^2`.
    noise_var : float
        Observation noise :math:`\\sigma_n^2`. Doubles as a jitter term to
        keep the linear system well-conditioned on small datasets.
    """

    length_scale: float = 0.2
    signal_var: float = 1.0
    noise_var: float = 1e-4

    # Cached after :meth:`fit`.
    _X: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _y: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _L: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _y_mean: float = field(default=0.0, init=False, repr=False)

    # -----------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        """Compute and cache the Cholesky factor of ``K + sigma_n^2 I``."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Inconsistent shapes: X={X.shape}, y={y.shape}."
            )

        # Center y so the zero-mean GP prior is appropriate.
        self._y_mean = float(y.mean())
        y_centered = y - self._y_mean

        n = X.shape[0]
        K = rbf_kernel(X, X, self.length_scale, self.signal_var)
        K_jitter = K + (self.noise_var + 1e-8) * np.eye(n)

        # Cholesky; if that fails fall back to incremental jitter (R&W ch. 5).
        L = None
        jitter = 1e-8
        for _ in range(6):
            try:
                L = np.linalg.cholesky(K_jitter)
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
                K_jitter = K + (self.noise_var + jitter) * np.eye(n)
        if L is None:
            raise np.linalg.LinAlgError("GP Cholesky failed even with jitter.")

        self._X = X
        self._y = y_centered
        self._L = L
        return self

    # -----------------------------------------------------------------
    def predict(
        self, X_star: np.ndarray, *, return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Posterior mean (and optionally standard deviation) at ``X_star``."""
        if self._X is None or self._L is None or self._y is None:
            raise RuntimeError("Fit the GP before calling predict().")
        X_star = np.atleast_2d(np.asarray(X_star, dtype=float))

        K_s = rbf_kernel(self._X, X_star, self.length_scale, self.signal_var)
        # alpha = (K + sigma_n^2 I)^{-1} y_centered  --- two triangular solves.
        v = np.linalg.solve(self._L, self._y)
        alpha = np.linalg.solve(self._L.T, v)
        mu = K_s.T @ alpha + self._y_mean

        if not return_std:
            return mu, None

        # Var[f*] = k(x*, x*) - K_s^T (K + sigma_n^2 I)^{-1} K_s.
        w = np.linalg.solve(self._L, K_s)
        var = self.signal_var - np.sum(w * w, axis=0)
        var = np.maximum(var, 1e-12)
        return mu, np.sqrt(var)

    # -----------------------------------------------------------------
    @staticmethod
    def heuristic_length_scale(X: np.ndarray) -> float:
        """Median heuristic for the RBF length-scale (Garreau et al. 2018)."""
        X = np.asarray(X, dtype=float)
        if X.shape[0] < 2:
            return 0.2
        # All pairwise distances; cheap for the small N used in BO.
        diffs = X[:, None, :] - X[None, :, :]
        d = np.sqrt(np.sum(diffs ** 2, axis=-1))
        triu = d[np.triu_indices_from(d, k=1)]
        med = float(np.median(triu))
        return max(med, 1e-3)
