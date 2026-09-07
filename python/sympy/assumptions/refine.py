"""Expression refinement under mathematical assumptions for FrankenSymPy."""

from __future__ import annotations

from typing import Any

from ..core import Add, Basic, Expr, Mul, Pow, S, Symbol, sympify
from .ask import Q, ask, global_assumptions


def refine_abs(expr: Any, assumptions: Any = True) -> Any:
    """Refine Abs(arg)."""
    arg = expr.args[0]
    refined_arg = refine(arg, assumptions)
    if ask(Q.positive(refined_arg), assumptions) is True or ask(Q.nonnegative(refined_arg), assumptions) is True:
        return refined_arg
    if ask(Q.negative(refined_arg), assumptions) is True or ask(Q.nonpositive(refined_arg), assumptions) is True:
        return -refined_arg
    if ask(Q.zero(refined_arg), assumptions) is True:
        return S.Zero
    from ..functions import Abs
    return Abs(refined_arg)


def refine_pow(expr: Any, assumptions: Any = True) -> Any:
    """Refine Pow(base, exp)."""
    base, exp = expr.args
    refined_base = refine(base, assumptions)
    refined_exp = refine(exp, assumptions)

    # (-1)**x
    if refined_base == -1 or refined_base == S.NegativeOne:
        if ask(Q.even(refined_exp), assumptions) is True:
            return S.One
        if ask(Q.odd(refined_exp), assumptions) is True:
            return S.NegativeOne

    # sqrt(x**2) or (x**2)**(1/2)
    if refined_exp == S.Half or (isinstance(refined_exp, Expr) and getattr(refined_exp, "is_rational", False) and str(refined_exp) == "1/2"):
        if isinstance(refined_base, Pow) and (refined_base.args[1] == 2 or str(refined_base.args[1]) == "2"):
            inner = refine(refined_base.args[0], assumptions)
            if ask(Q.positive(inner), assumptions) is True or ask(Q.nonnegative(inner), assumptions) is True:
                return inner
            if ask(Q.real(inner), assumptions) is True:
                from ..functions import Abs
                return Abs(inner)

    return Pow(refined_base, refined_exp)


def refine_sign(expr: Any, assumptions: Any = True) -> Any:
    """Refine sign(arg)."""
    arg = expr.args[0]
    refined_arg = refine(arg, assumptions)
    if ask(Q.positive(refined_arg), assumptions) is True:
        return S.One
    if ask(Q.negative(refined_arg), assumptions) is True:
        return S.NegativeOne
    if ask(Q.zero(refined_arg), assumptions) is True:
        return S.Zero
    from ..functions import sign
    return sign(refined_arg)


def refine_re(expr: Any, assumptions: Any = True) -> Any:
    """Refine re(arg)."""
    arg = expr.args[0]
    refined_arg = refine(arg, assumptions)
    if ask(Q.real(refined_arg), assumptions) is True:
        return refined_arg
    if ask(Q.imaginary(refined_arg), assumptions) is True:
        return S.Zero
    from ..functions import re
    return re(refined_arg)


def refine_im(expr: Any, assumptions: Any = True) -> Any:
    """Refine im(arg)."""
    arg = expr.args[0]
    refined_arg = refine(arg, assumptions)
    if ask(Q.real(refined_arg), assumptions) is True:
        return S.Zero
    if ask(Q.imaginary(refined_arg), assumptions) is True:
        from ..core import I
        return -I * refined_arg
    from ..functions import im
    return im(refined_arg)


_REFINE_HANDLERS: dict[str, Any] = {
    "Abs": refine_abs,
    "Pow": refine_pow,
    "sign": refine_sign,
    "re": refine_re,
    "im": refine_im,
}


def register_handler(name: str, handler: Any) -> None:
    """Register a handler for refining a named expression type."""
    _REFINE_HANDLERS[name] = handler


def remove_handler(name: str) -> None:
    """Remove a registered handler."""
    _REFINE_HANDLERS.pop(name, None)


def refine(expr: Any, assumptions: Any = True) -> Any:
    """Simplify an expression under given mathematical assumptions.

    Parameters
    ----------
    expr : Expr or Basic
        The expression to simplify.
    assumptions : Boolean or bool, optional
        The assumptions to use. Defaults to True (using global_assumptions).

    Returns
    -------
    Expr
        The refined expression.
    """
    if not isinstance(expr, Basic):
        expr = sympify(expr)

    if not hasattr(expr, "args") or not expr.args:
        return expr

    cls_name = expr.__class__.__name__
    if cls_name in _REFINE_HANDLERS:
        return _REFINE_HANDLERS[cls_name](expr, assumptions)

    if isinstance(expr, Add):
        new_args = [refine(a, assumptions) for a in expr.args]
        return Add(*new_args)

    if isinstance(expr, Mul):
        new_args = [refine(a, assumptions) for a in expr.args]
        return Mul(*new_args)

    new_args = [refine(a, assumptions) for a in expr.args]
    if tuple(new_args) != expr.args:
        try:
            return expr.__class__(*new_args)
        except Exception:
            return expr
    return expr
