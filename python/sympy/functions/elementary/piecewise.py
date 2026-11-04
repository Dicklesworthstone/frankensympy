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

    def __reduce__(self):
        from ...logic.boolalg import BooleanFalse, BooleanTrue

        expression, condition = self._struct_args
        # Exact Boolean atoms reconstruct through their zero-argument
        # singleton constructors. A plain Symbol("True") stays a Symbol.
        if type(condition) is BooleanTrue or type(condition) is BooleanFalse:
            condition = type(condition) is BooleanTrue
        return type(self), (expression, condition)

    def __str__(self) -> str:
        return f"({self.expr}, {self.cond})"

    __repr__ = __str__


def _cond_native(c: Any) -> Any:
    """Lower closed condition syntax, or retain its original Python surface.

    Ordinary Symbol/Dummy conditions are valid symbolic Piecewise inputs, but
    are not the BooleanTrue/BooleanFalse atoms in the native condition protocol.
    A structural guard preserves their typed identity through native arithmetic;
    custom Boolean classes retain their Python surface instead.
    """
    from ...core import Dummy, Symbol, _native, _native_expr
    from ...logic.boolalg import (
        And, BooleanFalse, BooleanFunction, BooleanTrue, Equivalent, Implies, Not, Or, Xor,
    )

    if type(c) is BooleanTrue or c is True:
        return _native.Expr("True")
    if type(c) is BooleanFalse or c is False:
        return _native.Expr("False")
    if type(c) is Symbol or type(c) is Dummy:
        # This is an internal structural marker, never a recovered Python
        # function class. In particular Symbol("True") stays a symbolic guard.
        return _native.py_function("_FsymBooleanSymbol", _native_expr(c))
    logical = ((And, "And"), (Or, "Or"), (Not, "Not"), (Xor, "Xor"),
               (Implies, "Implies"), (Equivalent, "Equivalent"))
    name = next((name for cls, name in logical if type(c) is cls), None)
    if name is not None:
        children = tuple(_cond_native(a) for a in BooleanFunction.__dict__["_args"].__get__(c, type(c)))
        if any(child is None for child in children):
            return None
        return _native.py_function(name, *children)
    if any(type(c) is cls for cls in _relationals().values()):
        try:
            return _native_expr(c)
        except NotImplementedError:
            return None
    return None


def _cond_lift(v: Any) -> Any:
    from ...core import _native, _wrap
    from ...logic import boolalg

    name = v.func_name
    if name == "_FsymBooleanSymbol":
        children = v.args
        if len(children) != 1:
            raise ValueError("invalid symbolic Piecewise condition")
        if children[0].is_symbol:
            return _wrap(children[0])
        # Native substitution may replace a symbolic guard by a relational
        # or Boolean tree. Recover that typed condition, without interpreting
        # an original Symbol("True") as a Boolean literal.
        if (children[0].func_name in _relationals()
                or children[0].func_name in ("And", "Or", "Not", "Xor", "Implies", "Equivalent")):
            return _cond_lift(children[0])
        raise ValueError("invalid substituted Piecewise condition")
    # Typed same-name symbols are not the legacy plain Boolean protocol atoms.
    if v == _native.Expr("True"):
        return boolalg.true
    if v == _native.Expr("False"):
        return boolalg.false
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

    __slots__ = ("_piecewise_branches",)

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

        conditions = tuple(_cond_native(branch.cond) for branch in branches)
        obj = object.__new__(cls)
        # Retain valid symbolic Boolean inputs that have no closed native
        # condition identity. Existing surface arithmetic keeps these exact
        # objects and refuses algorithms that require a closed native source.
        if any(condition is None for condition in conditions):
            obj._value = None
            Function.__dict__["_args"].__set__(obj, tuple(branches))
        else:
            native = [_native.py_function("Tuple", _native_expr(branch.expr), condition)
                      for branch, condition in zip(branches, conditions)]
            obj._value = _native.py_function("Piecewise", *native)
        # Numerical admission reads this slot without reconstructing conditions
        # or calling their public properties. Preserve held relationals and the
        # exact typed Symbol/Dummy objects chosen by the caller.
        Piecewise.__dict__["_piecewise_branches"].__set__(
            obj, tuple((branch.expr, branch.cond) for branch in branches)
        )
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

        try:
            pairs = Piecewise.__dict__["_piecewise_branches"].__get__(self, Piecewise)
        except AttributeError:
            pairs = None
        if pairs is not None:
            return tuple(ExprCondPair(expr, cond) for expr, cond in pairs)
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
