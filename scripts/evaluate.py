r"""Evaluate an already-trained agent against a foundation-environment task."""

from __future__ import annotations

import argparse
import json

from mini1_rl.agents import (
    MonteCarloAgent, SarsaNAgent, SarsaLambdaAgent, QLearningAgent,
    ValueIterationAgent, PolicyIterationAgent,
)
from mini1_rl.mdp import GridWorldMDP
from mini1_rl.experiments.masks import mask_assignment_primary, mask_cliff_walk


AGENTS = {
    "mc": MonteCarloAgent,
    "sarsa_n": SarsaNAgent,
    "sarsa_lambda": SarsaLambdaAgent,
    "q_learning": QLearningAgent,
    "value_iteration": ValueIterationAgent,
    "policy_iteration": PolicyIterationAgent,
}
TASKS = {
    "primary": lambda: GridWorldMDP(mask_assignment_primary(), gamma=0.9, terminal_rows=[4], slip_prob=0.1),
    "cliff": lambda: GridWorldMDP(mask_cliff_walk(), gamma=0.99, terminal_rows=[3], slip_prob=0.0),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=sorted(AGENTS), required=True)
    ap.add_argument("--task", choices=sorted(TASKS), default="primary")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    env = TASKS[args.task]()
    agent = AGENTS[args.algo]()
    agent.load(args.checkpoint)
    print(json.dumps(agent.evaluate(env, episodes=args.episodes), indent=2))


if __name__ == "__main__":
    main()
