"""Lambda and Subs (upstream sympy.core.function).

Registered into the synthetic ``sympy.core.function`` module by
``sympy/__init__.py`` so upstream import paths resolve.
"""

from __future__ import annotations

from typing import Any

from . import Expr


class Lambda(Expr):
    """``Lambda(x, expr)`` / ``Lambda((x, y), expr)``: an anonymous function."""

    __slots__ = ()

    def __new__(cls, signature: Any, expr: Any):
        from . import sympify

        sig = tuple(signature) if isinstance(signature, (tuple, list)) else (signature,)
        obj = object.__new__(cls)
        obj._struct_args = (sig, sympify(expr))
        return obj

    @property
    def signature(self):
        return self._struct_args[0]

    variables = signature

    @property
    def expr(self):
        return self._struct_args[1]

    @property
    def free_symbols(self) -> set:
        return set(self.expr.free_symbols) - set(self.signature)

    def __call__(self, *args: Any) -> Any:
        if len(args) != len(self.signature):
            raise TypeError(f"{self} takes {len(self.signature)} arguments ({len(args)} given)")
        return self.expr.subs(dict(zip(self.signature, args)))

    def __str__(self) -> str:
        sig = self.signature
        sig_s = str(sig[0]) if len(sig) == 1 else "(" + ", ".join(str(s) for s in sig) + ")"
        return f"Lambda({sig_s}, {self.expr})"

    __repr__ = __str__


class Subs(Expr):
    """Unevaluated substitution ``Subs(expr, x, point)``; ``doit`` applies it."""

    __slots__ = ()

    def __new__(cls, expr: Any, variables: Any, point: Any):
        from . import sympify

        vs = tuple(variables) if isinstance(variables, (tuple, list)) else (variables,)
        ps = tuple(point) if isinstance(point, (tuple, list)) else (point,)
        obj = object.__new__(cls)
        obj._struct_args = (sympify(expr), vs, tuple(sympify(p) for p in ps))
        return obj

    @property
    def expr(self):
        return self._struct_args[0]

    @property
    def variables(self):
        return self._struct_args[1]

    @property
    def point(self):
        return self._struct_args[2]

    def doit(self, **hints: Any) -> Any:
        return self.expr.subs(dict(zip(self.variables, self.point)))

    def __str__(self) -> str:
        def tup(t):
            return str(t[0]) if len(t) == 1 else "(" + ", ".join(str(a) for a in t) + ")"
        return f"Subs({self.expr}, {tup(self.variables)}, {tup(self.point)})"

    __repr__ = __str__
