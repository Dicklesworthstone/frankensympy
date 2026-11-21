"""Inference and SAT solving for FrankenSymPy logic."""

from __future__ import annotations

from typing import Any

from ..core import Symbol, _charge_nested_if_active, _native
from .boolalg import _BooleanRequest, _MAX_BOOLEAN_NODES, false, true


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
    if not all_models:
        request = _BooleanRequest(expr)
        res = request.native.satisfiable()
        request.unchanged()
        if res is None:
            return False
        return request.lift_model(res)

    def _model_generator():
        request = _BooleanRequest(expr)
        blocked = []
        blocked_size = 0
        while True:
            # Preserve one source/atom universe for the whole enumeration.
            # Re-lowering a growing surface formula both loses source owners
            # and can observe changes to caller-owned nodes between yields.
            request.unchanged()
            size = 1 + request.native_size + blocked_size
            if size > _MAX_BOOLEAN_NODES:
                raise ValueError("Boolean model enumeration exceeds the formula limit")
            _charge_nested_if_active(size)
            nb = (_native.BoolExpr.bool_and([request.native, *blocked])
                  if blocked else request.native)
            res = nb.satisfiable()
            request.unchanged()
            if res is None:
                break
            model = request.lift_model(res)
            # Save native assignments before yielding. Mutating a returned
            # model must not change which assignment the next solve excludes.
            assignment = tuple((token, res[token]) for token in request.atoms)
            yield model
            request.unchanged()
            if not assignment:
                break
            next_size = 1 + sum(2 if value else 1 for _, value in assignment)
            if 1 + request.native_size + blocked_size + next_size > _MAX_BOOLEAN_NODES:
                raise ValueError("Boolean model enumeration exceeds the formula limit")
            _charge_nested_if_active(next_size)
            literals = []
            for token, value in assignment:
                literal = _native.BoolExpr.bool_var(token)
                literals.append(_native.BoolExpr.bool_not(literal) if value else literal)
            blocked.append(_native.BoolExpr.bool_or(literals))
            blocked_size += next_size

    return _model_generator()


def valid(expr: Any) -> bool:
    """Check if a propositional formula is valid (a tautology)."""
    request = _BooleanRequest(expr)
    # Negate the admitted native source, without consulting a custom surface
    # Not subclass or reconstructing a changed argument owner.
    result = _native.BoolExpr.bool_not(request.native).satisfiable()
    request.unchanged()
    # The satisfiable constant True has the valid empty assignment {}.
    # Its falsey Python truth value is not an UNSAT result.
    return result is None


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
        vals = [pl_true(a, model) for a in expr.args]
        if None in vals:
            return None
        return sum(vals) % 2 == 1
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
    return model.get(expr, None)


__all__ = [
    "pl_true",
    "satisfiable",
    "valid",
]
