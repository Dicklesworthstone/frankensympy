"""Inference and SAT solving for FrankenSymPy logic."""

from __future__ import annotations

from typing import Any

from ..core import Symbol
from .boolalg import _native_bool, false, true


def satisfiable(expr: Any, algorithm: Any = None, all_models: bool = False) -> Any:
    """Check satisfiability of a boolean expression via DPLL.

    Parameters:
        expr: boolean expression
        algorithm: optional solver algorithm (ignored, DPLL used)
        all_models: if True, yields all satisfying models; if False, returns the first model or False.

    Returns:
        If all_models is False:
            A dict mapping Symbol to bool if satisfiable, or False if unsatisfiable.
        If all_models is True:
            A generator yielding dicts mapping Symbol to bool.
    """
    from .boolalg import And, Not, Or

    if not all_models:
        if expr is false or expr is False:
            return False
        if expr is true or expr is True:
            return {}

        nb = _native_bool(expr)
        res = nb.satisfiable()
        if res is None:
            return False
        return {Symbol(k): v for k, v in res.items()}

    def _model_generator():
        if expr is false or expr is False:
            return
        if expr is true or expr is True:
            yield {}
            return

        cur_expr = expr
        while True:
            nb = _native_bool(cur_expr)
            res = nb.satisfiable()
            if res is None:
                break
            model = {Symbol(k): v for k, v in res.items()}
            yield model
            blocking_literals = [Not(k) if v else k for k, v in model.items()]
            if not blocking_literals:
                break
            cur_expr = And(cur_expr, Or(*blocking_literals))

    return _model_generator()


def valid(expr: Any) -> bool:
    """Check if a propositional formula is valid (a tautology)."""
    from .boolalg import Not
    return not satisfiable(Not(expr))


def pl_true(expr: Any, model: dict[Any, bool] | None = None) -> bool | None:
    """Return True if propositional expression is true under model, False if false, or None if unknown."""
    if model is None:
        model = {}
    if expr is true or expr is True:
        return True
    if expr is false or expr is False:
        return False
    if isinstance(expr, Symbol):
        return model.get(expr, None)

    from .boolalg import And, Or, Not, Implies, Equivalent, Xor, Nand, Nor, Xnor, ITE

    if isinstance(expr, Not):
        v = pl_true(expr.args[0], model)
        return None if v is None else not v
    if isinstance(expr, And):
        vals = [pl_true(a, model) for a in expr.args]
        if False in vals:
            return False
        if None in vals:
            return None
        return True
    if isinstance(expr, Or):
        vals = [pl_true(a, model) for a in expr.args]
        if True in vals:
            return True
        if None in vals:
            return None
        return False
    if isinstance(expr, Implies):
        a_val = pl_true(expr.args[0], model)
        b_val = pl_true(expr.args[1], model)
        if a_val is False or b_val is True:
            return True
        if a_val is True and b_val is False:
            return False
        return None
    if isinstance(expr, Equivalent):
        vals = [pl_true(a, model) for a in expr.args]
        if None in vals:
            return None
        return all(v == vals[0] for v in vals)
    if isinstance(expr, Xor):
        a_val = pl_true(expr.args[0], model)
        b_val = pl_true(expr.args[1], model)
        if a_val is None or b_val is None:
            return None
        return a_val != b_val
    if isinstance(expr, Nand):
        a = pl_true(And(*expr.args), model)
        return None if a is None else not a
    if isinstance(expr, Nor):
        o = pl_true(Or(*expr.args), model)
        return None if o is None else not o
    if isinstance(expr, Xnor):
        x = pl_true(Xor(*expr.args), model)
        return None if x is None else not x
    if isinstance(expr, ITE):
        c_val = pl_true(expr.args[0], model)
        if c_val is True:
            return pl_true(expr.args[1], model)
        if c_val is False:
            return pl_true(expr.args[2], model)
        return None
    return None


__all__ = [
    "pl_true",
    "satisfiable",
    "valid",
]
