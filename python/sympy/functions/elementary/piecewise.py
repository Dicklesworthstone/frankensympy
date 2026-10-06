"""Piecewise-defined expressions (upstream sympy.functions.elementary.piecewise).

Piecewise and ExprCondPair are structural shell objects: conditions that
evaluate to ``false`` are dropped, a leading ``true`` condition returns its
expression, and substitution re-evaluates. They are not lowered to the
native kernel (arithmetic with a held Piecewise raises).
"""

from __future__ import annotations

from typing import Any

from ...core import Expr


def _as_boolean(c: Any) -> Any:
    from ...core import sympify
    from ...logic.boolalg import false, true

    if c is True:
        return true
    if c is False:
        return false
    return sympify(c)


class ExprCondPair(Expr):
    """One ``(expr, cond)`` branch of a Piecewise."""

    __slots__ = ()

    def __new__(cls, expr: Any, cond: Any):
        from ...core import sympify

        obj = object.__new__(cls)
        obj._struct_args = (sympify(expr), _as_boolean(cond))
        return obj

    @property
    def expr(self):
        return self._struct_args[0]

    @property
    def cond(self):
        return self._struct_args[1]

    def __iter__(self):
        return iter(self._struct_args)

    def __str__(self) -> str:
        return f"({self.expr}, {self.cond})"

    __repr__ = __str__


class Piecewise(Expr):
    """``Piecewise((expr1, cond1), (expr2, cond2), ...)``."""

    __slots__ = ()

    def __new__(cls, *pairs: Any, evaluate: bool = True):
        from ...logic.boolalg import BooleanFalse, BooleanTrue

        branches = []
        for p in pairs:
            expr, cond = (p.expr, p.cond) if isinstance(p, ExprCondPair) else tuple(p)
            pair = ExprCondPair(expr, cond)
            if evaluate and isinstance(pair.cond, BooleanFalse):
                continue
            if evaluate and any(pair.cond == b.cond and pair.expr == b.expr for b in branches):
                continue
            branches.append(pair)
            if evaluate and isinstance(pair.cond, BooleanTrue):
                break
        if not branches:
            from ...core import nan

            return nan
        if evaluate and isinstance(branches[0].cond, BooleanTrue):
            return branches[0].expr
        obj = object.__new__(cls)
        obj._struct_args = tuple(branches)
        return obj

    @property
    def free_symbols(self) -> set:
        out: set = set()
        for b in self._struct_args:
            out |= set(getattr(b.expr, "free_symbols", set()))
            out |= set(getattr(b.cond, "free_symbols", set()))
        return out

    def subs(self, *args: Any, **kwargs: Any) -> Any:
        return Piecewise(
            *[(b.expr.subs(*args, **kwargs), b.cond.subs(*args, **kwargs)
               if hasattr(b.cond, "subs") else b.cond) for b in self._struct_args]
        )

    def diff(self, *symbols: Any) -> Any:
        return Piecewise(*[(b.expr.diff(*symbols), b.cond) for b in self._struct_args])

    def doit(self, **hints: Any) -> Any:
        return Piecewise(*[(b.expr.doit(**hints) if hasattr(b.expr, "doit") else b.expr, b.cond)
                           for b in self._struct_args])


def piecewise_fold(expr: Any) -> Any:
    """Identity for expressions without nested Piecewise arithmetic."""
    return expr


__all__ = ["ExprCondPair", "Piecewise", "piecewise_fold"]
