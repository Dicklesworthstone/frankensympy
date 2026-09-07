"""Inference and SAT solving for FrankenSymPy logic."""

from __future__ import annotations

from typing import Any

from ..core import Symbol
from .boolalg import _native_bool, false, true


def satisfiable(expr: Any, algorithm: Any = None, all_models: bool = False) -> Any:
    """Check satisfiability of a boolean expression via DPLL.

    Returns:
        A dict mapping Symbol to bool if satisfiable, or False if unsatisfiable.
    """
    if expr is false or expr is False:
        return False
    if expr is true or expr is True:
        return {}

    nb = _native_bool(expr)
    res = nb.satisfiable()
    if res is None:
        return False
    return {Symbol(k): v for k, v in res.items()}


__all__ = [
    "satisfiable",
]
