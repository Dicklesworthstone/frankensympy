"""Lambda and Subs (upstream sympy.core.function).

Registered into the synthetic ``sympy.core.function`` module by
``sympy/__init__.py`` so upstream import paths resolve.
"""

from __future__ import annotations

from typing import Any

from . import Expr, Function


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


class Subs(Function):
    """Unevaluated substitution ``Subs(expr, x, point)``: a native binding
    construct with args ``(expr, Tuple(vars), Tuple(points))``."""

    __slots__ = ()

    def __new__(cls, expr: Any, variables: Any, point: Any, **options: Any):
        from . import Tuple, sympify

        def items(v: Any) -> tuple:
            if type(v).__name__ == "Tuple":
                return tuple(v.args)
            return tuple(v) if isinstance(v, (tuple, list)) else (v,)

        vs = tuple(sympify(v) for v in items(variables))
        ps = tuple(sympify(p) for p in items(point))
        if len(vs) != len(ps):
            raise ValueError("Number of point values must be the same as the number of variables.")
        expr = sympify(expr)
        keep = [(v, p) for v, p in zip(vs, ps) if v in expr.free_symbols and v != p]
        if not keep:
            return expr
        return Function.__new__(
            cls, expr, Tuple(*[v for v, _ in keep]), Tuple(*[p for _, p in keep]), evaluate=False
        )

    @property
    def expr(self):
        return self.args[0]

    @property
    def variables(self):
        return tuple(self.args[1].args)

    @property
    def point(self):
        return tuple(self.args[2].args)

    @property
    def free_symbols(self) -> set:
        syms = set(self.expr.free_symbols) - set(self.variables)
        for p in self.point:
            syms |= set(getattr(p, "free_symbols", set()))
        return syms

    def doit(self, **hints: Any) -> Any:
        e = self.expr
        if hints.get("deep", True) and hasattr(e, "doit"):
            e = e.doit(**hints)
        return e.subs(dict(zip(self.variables, self.point)))

    def __str__(self) -> str:
        def tup(t):
            return str(t[0]) if len(t) == 1 else "(" + ", ".join(str(a) for a in t) + ")"
        return f"Subs({self.expr}, {tup(self.variables)}, {tup(self.point)})"

    __repr__ = __str__
