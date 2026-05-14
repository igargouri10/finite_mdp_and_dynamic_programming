# V3 Decisions

*Mini-1 repository, EECS 590 Sp26. This file is the artifact V3 grades
under "Final Content": it lists every post-V2 class topic and records,
for each one, whether I implemented it, why, and where to find the
implementation. The intent is transparency, not justification by
volume — V3 explicitly penalizes "implements everything" as much as
"implements nothing."*

## TL;DR

For this repository's foundation environment — a single-protagonist,
fully observed, finite grid-world MDP — I implemented **one** V3
addition:

* **Bayesian hyperparameter optimization (GP + Expected Improvement).**

I deliberately did **not** implement, on the same grounds the V3 prompt
already flags:

* POMDP formalization and recurrent Q-networks.
* A generic Bayes-theorem belief-update script.
* Multi-agent / swarm coordination.

The rationale is below. The default repository layout (V1) and the
classical + DQN agents (V2) are kept as-is; the cleanup pass below
makes the repo's structural readiness match V3's bar.

---

## 1. Implemented: Bayesian hyperparameter optimization

**Code.** `src/mini1_rl/hpo/` — pure-NumPy GP surrogate
(`gp.py`), Latin-Hypercube / uniform / log-uniform search spaces
(`space.py`), Expected-Improvement BO loop (`optimizer.py`), and an RL
objective wrapper (`objectives.py`). CLI entry point at
`scripts/bayesian_hpo.py`. Unit tests in `tests/test_hpo.py` (10 new
assertions; all 14 tests in `tests/run_tests.py` pass).

**Why.** The V3 prompt itself states: *"An environment centered on a
main protagonist does not necessarily benefit from swarm/coordination
implementations, but very much benefits from Bayesian hyperparameter
tuning."* The Mini-1 grid worlds are exactly that — one agent, fully
observed, finite — so this is the post-V2 topic where a careful
implementation produces a *qualitative*, not just cosmetic, improvement
on the repo's headline algorithm.

**Empirical evidence.** Across two random seeds and 20 evaluations
each, GP-BO with EI reaches a mean evaluation return of **−2.07 ± 0.70**
on the canonical 5×5 mask, while uniformly-random search at the *same
evaluation budget* only reaches **−4.35 ± 0.55**. Concretely, BO finds
configurations whose greedy policy averages 2–3 steps to the goal,
whereas random search's best configuration averages 4–5 steps. The
log of every trial lives in `data/hpo/q_learning_primary.json`; the
comparison summary lives in `data/hpo/bo_vs_random_summary.json`; the
two report figures are `docs/report/figures/fig_v3_bo_history.pdf` and
`docs/report/figures/fig_v3_bo_vs_random.pdf`.

**Design choices, with the trade-offs I considered.**

| Choice | What I did | Why I rejected the alternative |
|---|---|---|
| Surrogate model | Gaussian process with RBF kernel and zero-mean prior (Rasmussen & Williams 2006). | TPE (Bergstra et al. 2011) is simpler but loses smoothness assumptions and trades poorly with EI; random forests need a vectorized-tree implementation that would have added more code than the GP. |
| Acquisition fn | Expected Improvement (Mockus 1989) with $\xi=0.01$ exploration bias. | UCB needs a $\beta$ schedule that I'd have to hand-tune — defeats the purpose of HPO. Thompson sampling needs full posterior samples and is overkill in 3-D. |
| Kernel hyperparams | Median-heuristic length scale; signal variance = $\mathrm{Var}(y)$; tiny noise jitter. | Type-II MLE on the kernel is unreliable at $n \leq 25$ observations (Snoek et al. 2012 §3) and adds an inner optimization that pulls in `scipy`. |
| Acquisition optim. | Random search of 4–8k unit-cube samples + small jitter for duplicate-guard. | L-BFGS-B on EI would pull in `scipy`; in 3-D, random restarts are within $10^{-3}$ of the optimum and don't break the no-scipy story. |
| Inner objective | Tabular Q-learning, 150–300 episodes, 30–50 evaluation rollouts. | DQN gives the same asymptotic policy on a fully observed 25-state MDP and costs ~100× more — using it as the BO inner loop would have made the demonstrative run prohibitively long while teaching no new methodology. |

---

## 2. Skipped: POMDPs + recurrent Q-networks

**Why skipped.** The grid-world environments in this repo are
**fully observed** by construction — `GridWorldMDP` returns the cell
index as the state, with no occlusion or noisy observation channel.
Adding a POMDP wrapper would either (a) silently re-derive the MDP
solution via belief filtering, which adds machinery without showing it
off, or (b) artificially mask part of the state, which would be a *new*
environment, not the foundation environment V3 asks me to harden.

**What I would have done.** Implement a `BeliefMDP` wrapper that
maintains a categorical posterior over the hidden cell index, and
swap the DQN MLP for a GRU-conditioned policy net. The Bayes-update
machinery is straightforward; the new environment would be more
honest pedagogy than POMDP-decorating a fully observed grid.

**Pointer.** Kaelbling et al. (1998) is already in `references.bib`
for the report's Section 4 / Section V3 paragraph on omitted topics.

---

## 3. Skipped: generic Bayes-theorem belief-update script

**Why skipped.** Without the POMDP wrapper above, there is no consumer
inside this repo for a generic Bayes-update utility — it would be a
20-line file that no test or experiment touches. V3 explicitly grades
against "no empty algorithm subdirectories with inits" and "few
exceptions" to that rule, so adding an unused module would *hurt*
structural readiness rather than help.

The Bayesian-HPO code (item 1) *does* use Bayesian updating, in the
GP sense: the posterior of the objective given the observations. That
is the form of Bayes' theorem this repo's domain actually exercises.

---

## 4. Skipped: multi-agent / swarm coordination

**Why skipped.** V3 says verbatim that "an environment centered on a
main protagonist does not necessarily benefit from swarm/coordination
implementations." My foundation environment is single-protagonist.
Multi-agent extensions on a 5×5 grid would invent a contrived task
(competing agents on the same cliff-walk?) and the resulting algorithm
choice — MADDPG, QMIX, PettingZoo wrappers — would be unrelated to
anything else in the repo. I'm taking V3 at its word here.

---

## 5. Structural-readiness changes

* Removed empty `notebooks/` scaffold directory.
* Removed `__pycache__/` artifacts and stale LaTeX intermediates
  (`.aux`, `.bbl`, `.blg`, `.log`, `.out`); both classes are still in
  `.gitignore` so they will not return on a clean checkout.
* Added `.gitkeep` to `checkpoints/`, `data/`, and
  `data/replay_buffers/` — they're V2 deliverables and the rotation
  script (`scripts/replace_old_experience.py`) expects them.
* Added `DECISIONS.md` (this file).
* Added `src/mini1_rl/hpo/` package and `scripts/bayesian_hpo.py` CLI.
* Added 10 unit tests in `tests/test_hpo.py`; wired them into
  `tests/run_tests.py` (14/14 passing as of the V3 submission).
* Added new BO references (Mockus, Snoek, Frazier, Rasmussen-Williams,
  Bergstra) to `docs/report/references.bib`.
* Extended `docs/technical-challenges.md` with the engineering issues
  hit during the BO implementation.
* Extended `README.md` (Section 10) and the LaTeX report (new V3
  section) so this work is discoverable.

---

## 6. What I'd add if I had another week

In rough order of payoff:

1. **Multi-fidelity BO (Klein et al. 2017).** Train the inner agent
   for a small budget early and an aggressive budget late. On Mini-1
   this would roughly 3× the trial throughput, which would let me run
   ≥5 seeds and report proper confidence intervals.
2. **PPO / REINFORCE actor-critic.** Not because the grid world needs
   policy gradients, but to use Bayesian HPO on a *very different*
   loss landscape — the demonstrative point of BO is "tune anything,"
   and so far I've shown it on a single algorithm.
3. **A genuinely partially-observed sibling env** (a maze with
   one-cell-radius vision) so the POMDP work in §2 has a real consumer.

---

*Author: Ismail Gargouri. No human collaborator on the V3 work; I
consulted Sutton & Barto (2018), Rasmussen & Williams (2006), Snoek et
al. (2012), and Frazier (2018) for the BO methodology. All references
appear in `docs/report/references.bib` and are cited in the V3 section
of `docs/report/report.pdf`.*
