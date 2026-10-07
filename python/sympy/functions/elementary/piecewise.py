"""Piecewise-defined expressions (upstream sympy.functions.elementary.piecewise).

Piecewise is a native-lowered function application over
``Tuple(expr, cond)`` branches; ExprCondPair is the structural view of one
branch. Conditions that evaluate to ``false`` are dropped and a leading
``true`` condition returns its expression.
"""

from __future__ import annotations

from typing import Any

from ...core import Expr, Function


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


def _cond_native(c: Any) -> Any:
    """Native lowering of a branch condition (relationals, True/False,
    And/Or/Not/Xor trees); TypeError for anything else."""
    from ...core import _native, _native_expr
    from ...logic.boolalg import BooleanFalse, BooleanTrue

    if isinstance(c, BooleanTrue) or c is True:
        return _native.Expr("True")
    if isinstance(c, BooleanFalse) or c is False:
        return _native.Expr("False")
    name = type(c).__name__
    if name in ("And", "Or", "Not", "Xor", "Implies", "Equivalent"):
        return _native.py_function(name, *[_cond_native(a) for a in c.args])
    return _native_expr(c)


def _cond_lift(v: Any) -> Any:
    from ...core import _wrap
    from ...logic import boolalg

    name = v.func_name
    if name == "Symbol" and str(v) in ("True", "False"):
        return boolalg.true if str(v) == "True" else boolalg.false
    if name in ("And", "Or", "Not", "Xor", "Implies", "Equivalent"):
        return getattr(boolalg, name)(*[_cond_lift(a) for a in v.args])
    rel = _RELATIONALS.get(name)
    if rel is not None and len(v.args) == 2:
        # Through the constructor, so substituted numbers decide it.
        return rel(_wrap(v.args[0]), _wrap(v.args[1]))
    return _wrap(v)


def _relationals() -> dict:
    from ...core import Eq, Ge, Gt, Le, Lt, Ne

    return {
        "Equality": Eq, "Unequality": Ne, "StrictLessThan": Lt,
        "LessThan": Le, "StrictGreaterThan": Gt, "GreaterThan": Ge,
    }


class _LazyRelationals(dict):
    def get(self, key, default=None):  # type: ignore[override]
        if not self:
            self.update(_relationals())
        return super().get(key, default)


_RELATIONALS = _LazyRelationals()


def _canonical_branches(pairs: tuple, evaluate: bool) -> list:
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
    return branches


class Piecewise(Function):
    """``Piecewise((expr1, cond1), (expr2, cond2), ...)``.

    Lowered natively as ``Piecewise(Tuple(e1, c1), ...)`` so it takes part
    in ordinary arithmetic, substitution and differentiation; branches
    whose condition is false drop and a leading true condition returns its
    expression, both at construction and whenever a native result is
    lifted back (e.g. after ``subs``).
    """

    __slots__ = ()

    def __new__(cls, *pairs: Any, evaluate: bool = True):
        branches = _canonical_branches(pairs, evaluate)
        if not branches:
            from ...core import nan

            return nan
        from ...logic.boolalg import BooleanTrue

        if evaluate and isinstance(branches[0].cond, BooleanTrue):
            return branches[0].expr
        return cls._build(branches)

    @classmethod
    def _build(cls, branches: list) -> "Piecewise":
        from ...core import _native, _native_expr

        native = []
        for b in branches:
            native.append(_native.py_function("Tuple", _native_expr(b.expr), _cond_native(b.cond)))
        obj = object.__new__(cls)
        obj._value = _native.py_function("Piecewise", *native)
        return obj

    @classmethod
    def _from_native(cls, value: Any) -> Any:
        """Lift a native ``Piecewise`` and re-evaluate its branches."""
        from ...core import _wrap

        pairs = []
        for t in value.args:
            if t.func_name != "Tuple" or len(t.args) != 2:
                obj = object.__new__(cls)
                obj._value = value
                return obj
            pairs.append((_wrap(t.args[0]), _cond_lift(t.args[1])))
        return cls(*pairs)

    @property
    def args(self) -> tuple:
        from ...core import _wrap

        out = []
        for t in self._value.args:
            out.append(ExprCondPair(_wrap(t.args[0]), _cond_lift(t.args[1])))
        return tuple(out)

    @property
    def free_symbols(self) -> set:
        out: set = set()
        for b in self.args:
            out |= set(getattr(b.expr, "free_symbols", set()))
            out |= set(getattr(b.cond, "free_symbols", set()))
        return out

    def subs(self, *args: Any, **kwargs: Any) -> Any:
        return Piecewise(
            *[(b.expr.subs(*args, **kwargs), b.cond.subs(*args, **kwargs)
               if hasattr(b.cond, "subs") else b.cond) for b in self.args]
        )

    def diff(self, *symbols: Any) -> Any:
        return Piecewise(*[(b.expr.diff(*symbols), b.cond) for b in self.args])

    def doit(self, **hints: Any) -> Any:
        return Piecewise(*[(b.expr.doit(**hints) if hasattr(b.expr, "doit") else b.expr, b.cond)
                           for b in self.args])


def piecewise_fold(expr: Any) -> Any:
    """Identity for expressions without nested Piecewise arithmetic."""
    return expr


__all__ = ["ExprCondPair", "Piecewise", "piecewise_fold"]
