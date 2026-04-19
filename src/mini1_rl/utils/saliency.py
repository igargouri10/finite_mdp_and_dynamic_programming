r"""Saliency analysis for DRL agents (V2 requirement).

For a tabular / small state-space DQN agent, saliency reduces to
:math:`\partial Q_\theta(s, a) / \partial s` where ``s`` is the one-hot
observation. We provide a minimal PyTorch-based implementation that works
for the :class:`~mini1_rl.agents.dqn.DQNAgent` out of the box.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


def input_gradient_saliency(
    model: Any,
    state: np.ndarray,
    action: Optional[int] = None,
) -> np.ndarray:
    r"""Return :math:`|\partial Q_\theta(s, a) / \partial s|` as a NumPy array.

    If ``action`` is None we take the argmax action.
    """
    if not HAS_TORCH:
        raise RuntimeError("Saliency requires PyTorch")
    s = torch.as_tensor(state, dtype=torch.float32, requires_grad=True)
    q = model(s)
    if action is None:
        action = int(q.argmax().item())
    q[action].backward()
    return s.grad.detach().cpu().numpy()
