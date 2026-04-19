# Project log — EECS 590 Mini-1

Chronological record of decisions, milestones, and blockers while
building `mini1_rl`. This is the log referenced in §Task 1 of the
written report.

## 2026-04-14 — Scoping and repository skeleton
* Read Mini-1, V1, and V2 assignment PDFs end-to-end.
* Decided on combined Mini-1 + V1 + V2 scope, Python implementation,
  task-by-task report structure, and an attempt at optional Task 7.
* Created cookiecutter-style layout `src/mini1_rl/{mdp,agents,buffers,nn,
  utils,experiments}/`, `scripts/`, `tests/`, `docs/`, `notebooks/`,
  `checkpoints/`, `data/`. Added `requirements.txt`, `environment.yml`,
  `setup.py`, `.gitignore`.

## 2026-04-14 — Markov process hierarchy (Task 2)
* Wrote the `MarkovProcess → MarkovRewardProcess → MarkovDecisionProcess`
  abstractions in `src/mini1_rl/mdp/base.py`.
* Choice: dense NumPy storage throughout, per-action stack `P` of shape
  `(A, S, S)`. Rationale: all Mini-1 grids are ≤ 144 states, so
  densification is cheap and the solvers become one-liners.
* `MarkovRewardProcess.bellman_operator(v, gamma)` returns the RHS of
  the Bellman equation. `MarkovDecisionProcess.induced_mrp(policy)`
  contracts the per-action stack with a stochastic policy and returns a
  new `MarkovRewardProcess` that plugs straight into any Task-3 solver.

## 2026-04-15 — Three value-convergence solvers (Task 3)
* Implemented `solve_value_matrix` via
  `np.linalg.solve(I - γP, R)` (LAPACK).
* Implemented `solve_value_iterative` via the Bellman fixed-point
  iteration `V_{k+1} = R + γP V_k`, with an optional
  `return_history=True` flag for the γ-sweep plot in §Task 5.
* Implemented `solve_value_eigendecomp` via `np.linalg.eig(P)` and
  `V = E diag(1/(1-γλ)) E^{-1} R`. Added condition-number check with
  `cond_thresh=1e10` and transparent fallback to the matrix solver for
  the degenerate swap matrix (see below).

## 2026-04-15 — Degenerate swap matrix (Task 4)
* Solved the MRP with `P = [[0,1],[1,0]]` and `R = [1, -1]` for γ ∈
  {0.5, 0.9, 0.99}. All three solvers agree to ~1e-12 once the
  spectral fall-back is enabled.
* Numeric check: `V(γ=0.9) ≈ [+0.526, -0.526]`, which matches the
  closed-form `V = (1 - γ^2)^{-1} [1 - γ, γ - 1] = ±(1 - γ)/(1 - γ^2) = ±1/(1+γ)`.

## 2026-04-15 — Grid-world builder (Task 6)
* Implemented `GridWorldMRP` and `GridWorldMDP` in `mdp/gridworld.py`,
  consuming a 2-D reward mask where `-inf` marks walls. Four cardinal
  actions; bounce-back on off-grid moves; terminal state is any cell
  with finite positive reward.
* Added an optional `slip_prob` parameter on `GridWorldMDP` to produce
  the stochastic variant used in Task 8 (matches Sutton & Barto §3
  stochastic-wind convention).

## 2026-04-16 — Task 7 tree-tiled composition
* First draft used `scipy.sparse.coo_matrix` for the composite
  transition. Installation failed with `ProxyError 403`. Rewrote using
  a hand-rolled `_SparseCOO` class with `(row, col, data)` lists and
  `add` / `zero_row` methods — see `technical-challenges.md` §1.
* Hit the `@dataclass` equality bug on `TileNode` (numpy arrays returned
  elementwise equality). Fixed via `eq=False` + `id()`-keyed index map —
  see `technical-challenges.md` §2.
* Final composite: three Task-7 tiles (A, B, C) glued along row/column
  portals, 51 non-wall states, 192 non-zero transitions. The jackpot
  reward in tile C propagates through the portal and dominates the
  optimal value map in tile A.

## 2026-04-17 — Agents (Task 8)
* Dynamic-programming: `PolicyIterationAgent` and `ValueIterationAgent`
  in `agents/dp.py`. On the primary 8×8 grid with γ = 0.95, policy
  iteration converges in **4 sweeps**, value iteration in **230 sweeps**;
  the two value functions agree to within 9e-10.
* Model-free classical: `MonteCarloAgent`, `TDnAgent`, `TDLambdaAgent`,
  `SarsaNAgent`, `SarsaLambdaAgent`, `QLearningAgent` in `agents/td.py`
  and `agents/mc.py`. Shared `epsilon_greedy_policy` / `rollout`
  helpers. All use accumulating eligibility traces where relevant.
* Deep RL (V2): `DQNAgent` in `agents/dqn.py` with replay buffer
  (`buffers/replay.py`), target network (hard or EMA update), ε-greedy
  exploration, Huber loss, Adam optimiser.

## 2026-04-17 — Experiment driver and figures
* `experiments/run_all.py` wires up every task end-to-end and writes
  `docs/report/figures/*.pdf` plus `results_table.json`. Saved nine
  PDF figures (scaling, γ-sweep, convergence history, optimal values
  and policy, tree-tiled values, algorithm comparison, Task-4 values).
* Seeded every RNG (`np.random.default_rng(42)`, `torch.manual_seed(42)`).

## 2026-04-18 — Report writing
* Drafted `docs/report/report.tex` (9 sections plus V2 justification)
  and `docs/report/references.bib` with real, verified BibTeX entries:
  Bellman 1957, Howard 1960, Puterman 1994, Bertsekas & Tsitsiklis 1996,
  Sutton & Barto 2018, Watkins & Dayan 1992, Rummery & Niranjan 1994,
  Tsitsiklis 1994, Dayan 1993, Lin 1992, Williams 1992, Mnih 2013/2015,
  van Hasselt 2016, Schulman 2015/2017, Lillicrap 2016, Fujimoto 2018,
  Haarnoja 2018, Banach 1922, Simonyan 2014, Silver 2017, and others.
* Hit missing `algorithm.sty` in the sandbox's TeX Live. Removed the
  unused `algorithm` / `algpseudocode` package imports — we had never
  added an actual algorithm environment.
* Final build: `pdflatex → bibtex → pdflatex → pdflatex` produces
  `report.pdf` (11 pages).

## Outcomes
* All 8 Mini-1 tasks implemented and verified.
* V1 scaffolding complete: `src/` layout, `scripts/` CLIs, `tests/`
  unit tests, pinned dependencies, report with BibTeX, README.
* V2 additions: DQN with replay buffer, saliency analysis hook,
  technical-challenges note, DRL justification section.
