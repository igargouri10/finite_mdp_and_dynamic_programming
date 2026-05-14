# mini1_rl — EECS 590 Mini-1 Reinforcement-Learning Repository

**Course.** EECS 590 Advanced Topics (Reinforcement Learning), Spring 2026.
**Author.** Ismail (igargouri1@gmail.com).
**Scope.** Mini-1 + V1 + V2 + **V3** combined deliverable: finite Markov
processes (MP, MRP, MDP), three value-convergence solvers, grid-world
builders from 2-D reward masks, a tree-tiled MRP (Task 7), classical RL
agents (MC, TD, SARSA, Q-learning), a Deep Q-Network (DQN), and a
from-scratch **Gaussian-process Bayesian hyperparameter optimization**
loop (V3) — organised as a professional research repository.

> **What V3 added** is documented in `DECISIONS.md` (the V3
> transparency file: implemented vs. deliberately skipped, with
> justifications). Section 10 of this README is the short version.

---

## Table of Contents

1.  Overview and scope
2.  Quick start
3.  Repository layout
4.  Task map (Mini-1 tasks 1–8)
5.  Running the experiments
6.  Deep RL algorithm justification (V2)
7.  Development notes and citations
8.  Collaborations and external material
9.  Limitations and future work
10. V3: Bayesian hyperparameter optimization

---

## 1. Overview and scope

This repository implements the full Mini-1 assignment together with the
optional V1 (professional repo structure) and V2 (add one Deep-RL
algorithm, document challenges, justify choices) extensions. All code is
in Python; numerics rely on NumPy, plots on Matplotlib, and the DQN
implementation uses PyTorch (optional import: the remainder of the code
runs without PyTorch).

The overall design intentionally mirrors the *abstract hierarchy* suggested
by the assignment: `MarkovProcess` is the plain transition-matrix base
class, `MarkovRewardProcess` adds a reward vector and a Bellman operator,
and `MarkovDecisionProcess` adds a per-action transition stack and the
method `induced_mrp(policy)`. The convergence solvers
(`solve_value_matrix`, `solve_value_iterative`, `solve_value_eigendecomp`)
are pure functions that take any `MarkovRewardProcess` and return a value
vector. Grid worlds, tree-tiled Cartesian compositions, and all the
learning agents plug into this hierarchy.

## 2. Quick start

```bash
# Python 3.10+ recommended.
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pip install -e .                      # install the mini1_rl package

# Reproduce all paper figures and the results table:
python -m mini1_rl.experiments.run_all

# Train a DQN on the primary grid (requires torch):
python scripts/train.py --agent dqn --mask primary --episodes 2000

# Evaluate any saved agent:
python scripts/evaluate.py --checkpoint checkpoints/dqn_primary.pt

# V3: Bayesian hyperparameter optimization on Q-learning:
python scripts/bayesian_hpo.py --task primary --algo q_learning \
    --n-init 5 --n-iter 20 \
    --log data/hpo/q_learning_primary.json \
    --fig docs/report/figures/fig_v3_bo_history.pdf
```

The paper (`docs/report/report.pdf`, compiled from `report.tex`) is
regenerated from the artifacts under `docs/report/figures/` and the BibTeX
database `docs/report/references.bib`.

## 3. Repository layout

```
mini1_rl_repo/
├── README.md                     # this file
├── requirements.txt              # pinned runtime dependencies
├── environment.yml               # optional conda environment
├── setup.py                      # editable install hook
├── src/mini1_rl/
│   ├── mdp/
│   │   ├── base.py               # MarkovProcess / MRP / MDP abstractions
│   │   ├── value.py              # matrix, iterative, and spectral solvers
│   │   ├── gridworld.py          # build MRP/MDP from 2-D reward masks
│   │   └── tiled.py              # Task 7: tree-tiled Cartesian MRP
│   ├── agents/
│   │   ├── base.py               # BaseAgent ABC and AgentRecord metadata
│   │   ├── dp.py                 # Policy Iteration, Value Iteration
│   │   ├── mc.py                 # Monte Carlo control with ε-greedy
│   │   ├── td.py                 # TD(n), TD(λ), SARSA(n), SARSA(λ), Q-learning
│   │   └── dqn.py                # Deep Q-Network (V2)
│   ├── buffers/replay.py         # experience replay + disk persistence (V2)
│   ├── nn/architectures.py       # MLP and ConvQNet
│   ├── utils/                    # viz, raster, saliency helpers
│   ├── hpo/                      # Bayesian hyperparameter optimization (V3)
│   │   ├── space.py              #   search-space primitives (U / log-U / int)
│   │   ├── gp.py                 #   pure-NumPy GP regression, RBF kernel
│   │   ├── optimizer.py          #   GP-BO loop with Expected Improvement
│   │   └── objectives.py         #   QLearning -> scalar objective wrapper
│   └── experiments/
│       ├── masks.py              # assignment masks and Task-7 tiles
│       └── run_all.py            # reproduces every figure in the report
├── scripts/                      # CLI entry points (train / evaluate / …)
├── tests/                        # unit tests for the MDP layer
├── docs/
│   ├── report/report.tex         # LaTeX source of the written report
│   ├── report/references.bib     # real BibTeX citations
│   ├── report/figures/           # generated PDFs + results_table.json
│   ├── technical-challenges.md   # V2 + V3 engineering notes
│   └── notes/project-log.md      # chronological decision log
├── DECISIONS.md                  # V3 transparency: what was/wasn't done
├── checkpoints/                  # trained agent state-dicts
└── data/                         # rollouts, replay buffers, hpo/ logs
```

## 4. Task map

| Task | Deliverable | Code location |
|------|-------------|---------------|
| 1    | Repo scaffold / project log | `src/mini1_rl/`, `docs/notes/project-log.md` |
| 2    | `MarkovProcess` class family | `src/mini1_rl/mdp/base.py` |
| 3    | Value convergence: matrix, iterative, spectral | `src/mini1_rl/mdp/value.py` |
| 4    | Value solve on degenerate swap matrix | `src/mini1_rl/experiments/run_all.py` §Task 4 |
| 5    | Solvers agree on an assignment-sized MRP | `src/mini1_rl/experiments/run_all.py` §Task 5 |
| 6    | Grid-world MRP/MDP from 2-D reward masks | `src/mini1_rl/mdp/gridworld.py` |
| 7    | Tree-tiled Cartesian composition (optional) | `src/mini1_rl/mdp/tiled.py` |
| 8    | Classical + deep RL agents; DQN baseline | `src/mini1_rl/agents/*.py` |

## 5. Running the experiments

`python -m mini1_rl.experiments.run_all` walks through every task end-to-end:
solvers, swap-matrix check, 4×4 / 8×8 / 12×12 scaling, γ-sweep and
convergence history on the canonical mask, Policy vs Value iteration,
classical-RL learning curves, and (if `torch` is installed) a DQN
baseline. It writes PDFs into `docs/report/figures/` and dumps scalar
results into `docs/report/figures/results_table.json`. The LaTeX report
reads both.

## 6. Deep Reinforcement Learning algorithm justification (V2 requirement)

We implemented **Deep Q-Learning (DQN)** as the V2 Deep-RL algorithm. The
choice is motivated by:

1. *Task match.* The primary evaluation surfaces are finite grid-worlds
   with discrete actions, which is exactly the setting for which DQN was
   designed and first evaluated on Atari
   \citep{mnih2013,mnih2015}.
2. *Natural bridge from tabular Q-learning.* Section 8 of the report
   compares DQN directly against tabular Q-learning
   \citep{watkins1992}; using the same target
   (`r + γ max_a' Q(s',a')`) makes the comparison crisp and lets us
   isolate the effect of function approximation.
3. *Pedagogical coverage.* DQN exercises both of the course's hard
   engineering requirements: an **experience replay buffer** (V2) and an
   optional **target network** \citep{lin1992,mnih2015}.
4. *Scope.* Policy-gradient / actor-critic methods such as REINFORCE
   \citep{williams1992}, PPO \citep{schulman2017ppo},
   DDPG \citep{lillicrap2016}, TD3
   \citep{fujimoto2018}, or SAC \citep{haarnoja2018}
   are designed for continuous control and would have required a
   fundamentally different evaluation set-up; they are discussed in
   §V2-Justification of the report but not implemented.

A full theoretical and empirical discussion is given in Section V2 of
`docs/report/report.pdf`.

## 7. Development notes and citations

All non-trivial factual claims in the written report cite their source.
Citations live in `docs/report/references.bib` and include, among others:
\citet{bellman1957} for the Bellman equation, \citet{howard1960} for
policy iteration, \citet{puterman1994} and \citet{bertsekas1996} for the
modern MDP / dynamic-programming treatment, \citet{banach1922} for the
contraction-mapping theorem, \citet{sutton2018} for the canonical RL
textbook, \citet{tsitsiklis1994} for Q-learning convergence,
\citet{dayan1993} for TD(λ), \citet{lin1992} for experience replay,
\citet{mnih2013,mnih2015} for DQN, and \citet{simonyan2014saliency} for
the saliency technique used to probe the DQN Q-network.

A short engineering log (what broke, how we fixed it, what we chose to
keep or drop) lives in `docs/technical-challenges.md`.

## 8. Collaborations and external material

* No human collaborators on any part of the code or the report.
* No pre-existing RL frameworks (Stable-Baselines3, RLlib, CleanRL, tf-agents,
  Dopamine, Acme) were used; all algorithms are written from scratch in this
  repo.
* Third-party libraries: NumPy, Matplotlib, PyTorch (optional). Their
  versions are pinned in `requirements.txt`.
* TeX Live packages: `natbib`, `amsmath`, `graphicx`, `subcaption`,
  `booktabs`, `hyperref`, `listings`.
* I consulted standard textbooks and papers for algorithmic correctness;
  every such source appears as a citation in the report.

## 9. Limitations and future work

* The DQN is trained on tabular grid-world state encodings for
  reproducibility on CPU; a pixel-input convolutional variant
  (`ConvQNet`) is included in `src/mini1_rl/nn/architectures.py` but not
  tuned. Extending to an Atari-scale evaluation
  \citep{mnih2015,van2016double} is the obvious next step.
* The tree-tiled MRP (Task 7) currently hand-rolls a tiny sparse COO
  matrix; a production build would drop in `scipy.sparse` for the
  matrix-free solve.
* Continuous-control algorithms (DDPG/TD3/SAC;
  \citealp{lillicrap2016,fujimoto2018,haarnoja2018}) and policy-gradient
  methods (REINFORCE, PPO; \citealp{williams1992,schulman2017ppo}) are
  discussed in the report but not implemented — future Mini-2 scope.

## 10. V3: Bayesian hyperparameter optimization

The V3 deliverable adds a from-scratch Gaussian-process Bayesian
optimization (GP-BO) module that tunes the canonical Q-learning agent
on the primary 5×5 mask. The rationale, in one sentence, is that V3
explicitly flags Bayesian HPO as the right tool for a
single-protagonist environment and flags multi-agent / swarm methods
as the *wrong* tool — so I implemented one and deliberately did not
implement the other. The full reasoning, including the V2-onwards
topics I considered and rejected (POMDPs, recurrent Q-nets,
multi-agent), is in `DECISIONS.md`.

Empirically, with a budget of 20 trials and two seeds, GP-BO with
Expected Improvement attains a mean evaluation return of −2.07 ± 0.70
on the primary mask versus −4.35 ± 0.55 for uniformly-random search
under the same budget (Section V3 of the report;
`data/hpo/bo_vs_random_summary.json`). The module is pure NumPy: no
scipy, no scikit-learn, no Optuna. References (Mockus 1989; Snoek,
Larochelle & Adams 2012; Frazier 2018; Rasmussen & Williams 2006;
Bergstra et al. 2011) are in `docs/report/references.bib`.
