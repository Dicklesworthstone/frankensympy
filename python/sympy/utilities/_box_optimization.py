"""Exact admission for independently checked global finite-box optimization.

The native call runs bounded candidate search and independent complete-cover
verification under one caller ledger. Python never supplies objective values,
derivatives, a claimed error, or an unchecked accepted receipt.
"""

from ._certified_numeric import (
    _call_budgeted, _charge_exact_work, _native_box, _native_number, _rational,
)
from ._certified_objective import BoxOptimizationReceipt, _prepare_box
from .lambdify import _check_evaluation_budget


def optimize_box_certified(program, box, kind, tolerance, precision, max_leaves,
                           *, budget=None):
    """Find a feasible point with a proved absolute global value gap."""
    _check_evaluation_budget(budget)
    if type(kind) is not str:
        raise TypeError("box-optimization kind must be an exact string")
    if kind not in ("minimum", "maximum"):
        raise ValueError("kind must be 'minimum' or 'maximum'")
    if type(max_leaves) is not int or not 1 <= max_leaves <= 1024:
        raise ValueError("box optimization max_leaves must be an integer from 1 through 1024")
    box = _prepare_box(program, box, precision, stationary=False, budget=budget)
    tolerance = _rational(tolerance, budget)
    _charge_exact_work(budget, tolerance)
    if tolerance <= 0:
        raise ValueError("box optimization tolerance must be a positive exact rational")
    receipt = _call_budgeted(
        program.native_objective.optimize_box,
        _native_box(box, budget), _native_number(tolerance, budget),
        kind, precision, max_leaves, budget=budget,
    )
    result = BoxOptimizationReceipt(receipt)
    _check_evaluation_budget(budget)
    return result
