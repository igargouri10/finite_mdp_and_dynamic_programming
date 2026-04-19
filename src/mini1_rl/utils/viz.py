r"""Matplotlib plotting helpers used by the Mini-1 experiments."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def plot_value_grid(
    grid: np.ndarray,
    *,
    title: str = "V*",
    cmap: str = "viridis",
    annotate: bool = True,
    savepath: Optional[str] = None,
    fig=None, ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4.0))
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(range(grid.shape[1]))
    ax.set_yticks(range(grid.shape[0]))
    if annotate:
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                v = grid[r, c]
                if np.isfinite(v):
                    ax.text(c, r, f"{v:.1f}", ha="center", va="center",
                            color="white" if v < np.nanmean(grid) else "black",
                            fontsize=8)
                else:
                    ax.text(c, r, "#", ha="center", va="center", color="red")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_gamma_sweep(
    gammas: Iterable[float],
    values_by_gamma: Dict[float, np.ndarray],
    *,
    labels: Optional[List[str]] = None,
    title: str = "V* vs gamma",
    savepath: Optional[str] = None,
):
    gammas = list(gammas)
    fig, ax = plt.subplots(figsize=(6, 4))
    states_to_plot = list(values_by_gamma.values())[0].shape[0]
    if states_to_plot > 10:
        # subsample to keep the plot legible
        show = np.linspace(0, states_to_plot - 1, 10, dtype=int)
    else:
        show = range(states_to_plot)
    for s in show:
        ys = [values_by_gamma[g][s] for g in gammas]
        lab = labels[s] if labels else f"s{s}"
        ax.plot(gammas, ys, marker="o", label=lab)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$V^{\pi}(s)$")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_learning_curves(
    curves: Dict[str, List[float]],
    *,
    window: int = 25,
    title: str = "Learning curves",
    xlabel: str = "episode",
    ylabel: str = "episode return",
    savepath: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, vals in curves.items():
        arr = np.asarray(vals, dtype=float)
        if len(arr) >= window:
            kernel = np.ones(window) / window
            smooth = np.convolve(arr, kernel, mode="valid")
            ax.plot(np.arange(len(smooth)) + window - 1, smooth, label=name)
        else:
            ax.plot(arr, label=name)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_convergence_history(
    history: np.ndarray,
    *,
    title: str = "Iterative value convergence",
    savepath: Optional[str] = None,
    gamma: Optional[float] = None,
):
    fig, ax = plt.subplots(figsize=(6, 4))
    V_final = history[-1]
    diffs = np.max(np.abs(history - V_final), axis=1)
    ax.semilogy(diffs, label=r"$\|V_k - V^*\|_{\infty}$")
    if gamma is not None:
        k = np.arange(len(diffs))
        bound = diffs[0] * gamma ** k
        ax.semilogy(k, bound, "--", label=rf"$\gamma^k$-bound (γ={gamma})")
    ax.set_xlabel("iteration k"); ax.set_ylabel("sup-norm error")
    ax.set_title(title); ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, ax
