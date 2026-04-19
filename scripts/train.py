r"""Train or further-train an agent (V1 requirement 5).

Usage example::

    python scripts/train.py --algo q_learning --task primary --episodes 500 \
        --out checkpoints/q_learning/primary/run1
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from mini1_rl.agents import (
    MonteCarloAgent, SarsaNAgent, SarsaLambdaAgent, QLearningAgent,
    PolicyIterationAgent, ValueIterationAgent,
    TDnAgent, TDLambdaAgent,
)
from mini1_rl.mdp import GridWorldMDP
from mini1_rl.experiments.masks import mask_assignment_primary, mask_cliff_walk


AGENT_REGISTRY = {
    "mc": MonteCarloAgent,
    "sarsa_n": SarsaNAgent,
    "sarsa_lambda": SarsaLambdaAgent,
    "q_learning": QLearningAgent,
    "policy_iteration": PolicyIterationAgent,
    "value_iteration": ValueIterationAgent,
    "td_n": TDnAgent,
    "td_lambda": TDLambdaAgent,
}
TASK_REGISTRY = {
    "primary": lambda: GridWorldMDP(mask_assignment_primary(), gamma=0.9, terminal_rows=[4], slip_prob=0.1),
    "cliff": lambda: GridWorldMDP(mask_cliff_walk(), gamma=0.99, terminal_rows=[3], slip_prob=0.0),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=sorted(AGENT_REGISTRY), required=True)
    ap.add_argument("--task", choices=sorted(TASK_REGISTRY), default="primary")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hyper", type=json.loads, default={})
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    env = TASK_REGISTRY[args.task]()
    agent = AGENT_REGISTRY[args.algo](seed=args.seed, **args.hyper)
    agent.record.task = args.task
    if args.resume:
        agent.load(args.resume)

    metrics = agent.train(env, episodes=args.episodes)
    agent.save(args.out)
    print(json.dumps({"algo": args.algo, "task": args.task, "metrics": metrics}, indent=2, default=str))


if __name__ == "__main__":
    main()
