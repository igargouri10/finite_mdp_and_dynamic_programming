r"""Run Bayesian hyperparameter optimization on a tabular Q-learning agent (V3).

Example
-------
::

    python scripts/bayesian_hpo.py \
        --task primary --algo q_learning \
        --n-init 5 --n-iter 20 \
        --episodes 300 --eval-episodes 50 \
        --seed 0 \
        --log data/hpo/q_learning_primary.json \
        --fig docs/report/figures/fig_task_v3_bo_history.pdf

The script writes:

* a JSON log of every trial (parameters, objective, wall-clock seconds);
* an optional convergence-history figure with the running-best curve.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import numpy as np

from mini1_rl.hpo import (
    BayesianOptimizer, SearchSpace,
    LogUniformParam, UniformParam,
    QLearningObjective,
)
from mini1_rl.experiments.masks import (
    mask_assignment_primary, mask_cliff_walk,
)


TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "primary": {
        "mask_fn": mask_assignment_primary,
        "terminal_rows": (4,),
        "slip_prob": 0.1,
    },
    "cliff": {
        "mask_fn": mask_cliff_walk,
        "terminal_rows": (3,),
        "slip_prob": 0.0,
    },
}


def build_space() -> SearchSpace:
    return SearchSpace(
        params=[
            LogUniformParam("alpha",   low=1e-3, high=5e-1),
            UniformParam   ("epsilon", low=0.01, high=0.30),
            UniformParam   ("gamma",   low=0.70, high=0.99),
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", choices=sorted(TASK_REGISTRY), default="primary")
    ap.add_argument("--algo", choices=["q_learning"], default="q_learning",
                    help="V3 ships with a Q-learning objective; SARSA/DQN "
                         "objectives can be added with the same contract.")
    ap.add_argument("--n-init", type=int, default=5)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=300,
                    help="Training episodes per BO trial.")
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--xi", type=float, default=0.01,
                    help="Expected-Improvement exploration bias.")
    ap.add_argument("--log", default=None,
                    help="Where to write the JSON trial log.")
    ap.add_argument("--fig", default=None,
                    help="Where to write a convergence-history PDF (optional).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    task = TASK_REGISTRY[args.task]
    objective = QLearningObjective(
        mask=task["mask_fn"](),
        terminal_rows=task["terminal_rows"],
        slip_prob=task["slip_prob"],
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
    )

    space = build_space()
    bo = BayesianOptimizer(
        space=space,
        objective=objective,
        n_init=args.n_init,
        n_iter=args.n_iter,
        seed=args.seed,
        xi=args.xi,
        verbose=args.verbose,
    )
    bo.run()

    # ---------- JSON log ----------
    payload = {
        "version": "v3",
        "task": args.task,
        "algo": args.algo,
        "n_init": args.n_init,
        "n_iter": args.n_iter,
        "episodes_per_trial": args.episodes,
        "eval_episodes_per_trial": args.eval_episodes,
        "seed": args.seed,
        "xi": args.xi,
        "search_space": [
            {
                "name": p.name,
                "kind": type(p).__name__,
                "low": p.low, "high": p.high,
            }
            for p in space.params
        ],
        "best_trial": bo.best_trial.to_dict() if bo.best_trial else None,
        "running_best": bo.running_best(),
        "trials": bo.history(),
    }
    if args.log:
        os.makedirs(os.path.dirname(os.path.abspath(args.log)) or ".", exist_ok=True)
        with open(args.log, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(f"[bayesian_hpo] wrote {args.log}")
    else:
        json.dump(payload["best_trial"], sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")

    # ---------- Convergence-history figure ----------
    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        running = bo.running_best()
        objs = [t.objective for t in bo.trials]
        x = np.arange(1, len(running) + 1)

        os.makedirs(os.path.dirname(os.path.abspath(args.fig)) or ".", exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.0, 3.5))
        ax.plot(x, objs, "o", color="tab:gray", alpha=0.6,
                label="trial objective")
        ax.plot(x, running, "-", color="tab:blue", lw=2.0,
                label="running best")
        ax.axvline(args.n_init + 0.5, color="k", ls=":", lw=1.0,
                   label="end of initial LHS design")
        ax.set_xlabel("BO trial")
        ax.set_ylabel("mean evaluation return (Q-learning)")
        ax.set_title(f"Bayesian HPO on '{args.task}' grid")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.fig, bbox_inches="tight")
        plt.close(fig)
        print(f"[bayesian_hpo] wrote {args.fig}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
