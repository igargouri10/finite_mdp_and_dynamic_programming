"""Minimal pytest-free test runner.

Executes the assertions in ``test_mdp.py`` without requiring the
``pytest`` dependency — we still treat the file as the canonical
test specification.
"""

from __future__ import annotations

import sys
import traceback

import numpy as np

# Shim pytest.raises for the row-stochastic test.
import types
_pytest = types.ModuleType("pytest")


class _Raises:
    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(f"Expected {self.exc.__name__}, nothing raised")
        return issubclass(exc_type, self.exc)


def _raises(exc):
    return _Raises(exc)


_pytest.raises = _raises
sys.modules["pytest"] = _pytest

from tests import test_mdp  # noqa: E402

TESTS = [
    test_mdp.test_markov_process_row_stochastic,
    test_mdp.test_mrp_closed_form_matches_iterative,
    test_mdp.test_degenerate_swap_matrix,
    test_mdp.test_mdp_induced_mrp_matches_direct,
]


def main() -> int:
    failures = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} tests passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
