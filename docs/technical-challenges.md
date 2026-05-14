# Technical Challenges and Engineering Notes

*Companion to `docs/report/report.pdf`. Part of the V2 deliverable: an
honest account of what broke, what was worked around, and what the
consequences for the final code are.*

This document lists every non-trivial engineering problem that came up
while building `mini1_rl`, the decision that was made, and — where
relevant — a pointer to the source file that implements the fix.

---

## 1. Sandboxed Python environment: no `scipy`, no `torch` by default

**Problem.** The initial repo design assumed `scipy.sparse` for the
tree-tiled MRP (Task 7) and an unconditional `import torch` in the DQN
module. Both installations failed in the evaluation sandbox with a
`ProxyError 403 Forbidden` when pulling from PyPI.

**Decision.**

* **Task 7 / tiled MRP.** Replaced `scipy.sparse.coo_matrix` with a
  hand-rolled `_SparseCOO` class (see `src/mini1_rl/mdp/tiled.py`). The
  class stores three parallel lists `(rows, cols, data)`, supports
  `add(i, j, v)` and `zero_row(i)` (for portal installation), and emits
  a dense NumPy matrix on demand for the value solvers. Because all
  Mini-1 grid sizes are small (≤ 256 states for Task 7 composites),
  densification is cheap.
* **DQN / torch.** The `import torch` in `src/mini1_rl/agents/dqn.py` is
  wrapped in a `try/except`; the module exposes a module-level
  `HAS_TORCH` flag and the DQN class raises a clear `ImportError` if
  instantiated without the dependency. The `agents/__init__.py`
  conditionally re-exports `DQNAgent`. `run_all.py` probes `HAS_TORCH`
  at start-up and silently skips the DQN experiment when torch is
  missing — every other experiment (Tasks 2–7, policy/value iteration,
  tabular Q-learning, SARSA, MC) still runs.

**Consequence.** The full paper pipeline reproduces on any CPU-only
machine with only NumPy and Matplotlib. A machine with PyTorch enables
the Section 8 DQN plot; a machine without it produces the classical-RL
comparison alone, and the report's DQN subsection notes the difference.

## 2. `@dataclass` equality vs. NumPy arrays: `TileNode` bug

**Problem.** In `tiled.py` the first draft declared

```python
@dataclass
class TileNode:
    matrix: np.ndarray
    reward: np.ndarray
    ...
```

and then used `leaves.index(node)` when installing portals. `list.index`
calls `__eq__`, which for a default dataclass does element-wise array
equality and returns an ambiguous boolean array. Python then raised
`ValueError: The truth value of an array with more than one element is
ambiguous`.

**Decision.** Two small changes in `src/mini1_rl/mdp/tiled.py`:

1. Declared `@dataclass(eq=False)` so nodes compare by identity.
2. Built `leaf_id_to_idx = {id(leaf): i for i, leaf in enumerate(leaves)}`
   once at construction time and use `leaf_id_to_idx[id(node)]` in the
   portal-installation loop.

**Why both.** `eq=False` alone would let us fall back to `is`-based
comparison, but `list.index` still does an O(n) scan; the identity dict
is O(1) per lookup and was free to add.

## 3. Degenerate swap matrix in the spectral solver

**Problem.** Task 4 requires solving the MRP with transition

```
P = [[0, 1], [1, 0]]
```

for several γ. The matrix solver `(I - γP)` is well-conditioned, and
fixed-point iteration converges for γ < 1. The spectral (eigendecomposition)
solver, however, is numerically unstable here: `P` has repeated
eigenvalue 1 with a full basis *only in exact arithmetic*; a small
round-off on the order of 1e-16 can make the computed eigenvector matrix
effectively singular in float64.

**Decision.** `solve_value_eigendecomp` in `src/mini1_rl/mdp/value.py`
computes the condition number of the eigenvector matrix and, if it
exceeds `cond_thresh=1e10`, transparently falls back to
`solve_value_matrix`. A logging-style `warnings.warn` message records
the fall-back so the behaviour is visible to the user.

**Consequence.** All three solvers return the same value vector to
within 1e-12 on the swap matrix, which is what the report's Task-4
table reports.

## 4. Matplotlib rendering of `-inf` walls in the grid value heatmap

**Problem.** Walls in the grid-world mask carry reward `-inf`. A
naïve `imshow(values)` call renders walls as white or NaN, depending on
the colormap, because Matplotlib's normaliser maps `-inf` to the minimum
of the numeric data.

**Decision.** `plot_value_grid` in `src/mini1_rl/utils/viz.py` masks wall
cells with `np.ma.masked_invalid` and configures the colormap with
`cmap.set_bad('0.2')`. Walls render as a dark grey, consistent across
all task figures.

## 5. Bouncing vs. absorbing grid boundaries

**Problem.** The assignment PDF is silent on what happens if an action
would move off the grid. Two conventions are common in the RL literature
— bounce-back (agent stays put) and absorbing (episode ends in a
penalty state). They produce quantitatively different value functions
on small grids.

**Decision.** Implemented *bounce-back* in `_neighbours`
(`src/mini1_rl/mdp/gridworld.py`) and noted the choice in §Task 6 of the
report, with a citation to \citet{sutton2018} Chapter 3, which uses the
same convention. The value maps in the figures reflect bounce-back
dynamics.

## 6. Per-action transition stack layout

**Problem.** The `MarkovDecisionProcess.induced_mrp(policy)` method has
to contract a `|A| × |S| × |S|` tensor with a `|S| × |A|` policy. We
considered (a) storing a Python list of `|A|` dense `|S| × |S|` matrices
versus (b) a single `numpy.ndarray` of shape `(A, S, S)`.

**Decision.** Kept the `(A, S, S)` ndarray. The contraction becomes a
clean `np.einsum('sa,ass->ss', policy, P)` (and an analogous one for
the reward), which outperforms a Python loop for the 12×12 grids and
reads closer to the Bellman-operator literature.

## 7. LaTeX tooling: `algorithm`/`algpseudocode` unavailable in the sandbox

**Problem.** The sandbox's TeX Live install lacks `texlive-science`,
so `\usepackage{algorithm}` fails and the whole PDF build stops at page
1.

**Decision.** We did not have any `\begin{algorithm}` blocks — the
packages were imported defensively from an earlier draft — so the two
`\usepackage{…}` lines were simply removed. All remaining packages
(`natbib`, `amsmath`, `graphicx`, `subcaption`, `listings`, …) are in
`texlive-latex-extra`, which is installed. A full rebuild now works
with the standard `pdflatex → bibtex → pdflatex → pdflatex` dance.

## 8. Reproducibility

Every experiment fixes seeds explicitly (`np.random.default_rng(42)`
for NumPy and — when used — `torch.manual_seed(42)`). The
`results_table.json` artefact is regenerated deterministically from
`run_all.py`. The report reads its numeric claims from this file
rather than hard-coding them, so the written report and the code
cannot silently drift.

---

## Quick summary table

| # | Challenge | Fix | File |
|---|-----------|-----|------|
| 1 | no `scipy` / `torch` | hand-rolled `_SparseCOO`; optional torch import | `mdp/tiled.py`, `agents/dqn.py` |
| 2 | dataclass `__eq__` on numpy arrays | `eq=False` + `id()`-keyed dict | `mdp/tiled.py` |
| 3 | spectral solver on swap matrix | cond-number fallback | `mdp/value.py` |
| 4 | `-inf` walls in heatmap | masked array + `set_bad` | `utils/viz.py` |
| 5 | off-grid action convention | bounce-back, documented | `mdp/gridworld.py` |
| 6 | `(A,S,S)` contraction | `np.einsum`-based `induced_mrp` | `mdp/base.py` |
| 7 | missing `algorithm.sty` | drop unused package imports | `docs/report/report.tex` |
| 8 | reproducibility drift | seeded RNG + regenerated JSON | `experiments/run_all.py` |
| 9 | GP Cholesky failure on small N | incremental jitter loop | `hpo/gp.py` |
| 10 | `scipy.stats.norm` not available | `math.erf`-based Phi/phi for EI | `hpo/optimizer.py` |
| 11 | EI saturates on tiny `sigma` | explicit `sigma < 1e-9` zero-out | `hpo/optimizer.py` |
| 12 | LHS hidden mode-collapse | per-dim permutation, asserted in tests | `hpo/space.py`, `tests/test_hpo.py` |
| 13 | BO duplicate-restart on EI argmax | min-distance jitter guard | `hpo/optimizer.py` |
| 14 | BO trial budget vs. CPU time | reduced episodes/trial for the demo run | `scripts/bayesian_hpo.py` |

---

## V3 additions: Bayesian hyperparameter optimization

### 9. GP Cholesky breakdown at small training sets

**Problem.** With ~5 LHS points the kernel matrix
:math:`K + \sigma_n^2 I` is small but its conditioning depends on the
heuristic length-scale. Twice during integration the
`numpy.linalg.cholesky` raised `LinAlgError` after a near-duplicate LHS
sample collapsed the kernel matrix toward rank-deficiency.

**Fix.** `hpo/gp.py` wraps the factorization in an incremental-jitter
loop (Rasmussen & Williams 2006 §5.4.2): start with the configured
`noise_var`, double the jitter each retry, cap at six tries. In
practice the second attempt always succeeds.

### 10. No `scipy.stats.norm` -> hand-rolled standard-normal CDF

**Problem.** The Expected-Improvement acquisition needs
:math:`\Phi(z)` and :math:`\phi(z)`. Pulling `scipy.stats` for one
function would have re-broken the "CPU-only, NumPy-only" story we
locked in for V2 (item 1 above).

**Fix.** `expected_improvement(...)` in `hpo/optimizer.py` computes
:math:`\phi(z) = e^{-z^2/2}/\sqrt{2\pi}` directly and uses
`math.erf` (stdlib) for the CDF:
:math:`\Phi(z) = \tfrac12(1 + \operatorname{erf}(z/\sqrt{2}))`. Tests
in `tests/test_hpo.py` confirm `EI = 0` when `sigma = 0` and
`EI > 0` strictly above the incumbent.

### 11. EI's degeneracy on noiseless replays

**Problem.** When BO re-evaluates a point very close to a prior
sample, the GP predicts almost the training value with
:math:`\sigma \to 0`, so EI is numerically `nan/inf` instead of zero.

**Fix.** Explicit `ei[sigma < 1e-9] = 0.0` in
`expected_improvement(...)`, plus a min-distance check in the
optimizer that re-jitters the chosen unit-cube point if it is within
:math:`10^{-3}` of any prior observation.

### 12. Hidden Latin-Hypercube failure mode

**Problem.** The first LHS implementation forgot to permute the
strata per dimension and produced a *diagonal* design that defeated
the BO. The bug only surfaced because the BO never improved over the
LHS phase.

**Fix.** Per-dimension `rng.permutation(strata)` in
`SearchSpace.sample_lhs` (matches McKay-Beckman-Conover 1979). The
test `test_search_space_lhs_is_a_permutation` asserts each column
visits every stratum exactly once and would have caught this earlier.

### 13. Compute budget for the V3 demonstrative run

**Problem.** The original `300 episodes × 50 eval × 25 trials × 3
seeds` configuration takes ~3 minutes in the sandbox and was timing
out on a 45 s call. V3 still requires that the BO actually run and
not just exist as code.

**Fix.** Two-tier run plan baked into the docs:

* `scripts/bayesian_hpo.py` defaults: 300 train / 50 eval / 25 trials.
  Single seed; reproduces the headline `fig_v3_bo_history.pdf`.
* Comparison run (`v3_bo_vs_random.py` in `outputs/`): 150 train /
  30 eval / 20 trials / 2 seeds. Produces
  `fig_v3_bo_vs_random.pdf` and `bo_vs_random_summary.json`. Cheaper
  but enough to show GP-BO > random search at 2× the gap of either
  one's standard deviation.
