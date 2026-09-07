"""Integration expressions and classes for FrankenSymPy."""

from typing import Any, Tuple
from ..core import Basic, Expr, Symbol, _native_expr, _wrap


class Integral(Expr):
    """An unevaluated integral."""

    __slots__ = ("_function", "_limits")

    def __new__(cls, function: Any, *limits: Any, evaluate: bool = False):
        if evaluate:
            from .. import integrate
            return integrate(function, *limits)
        obj = object.__new__(cls)
        obj._function = _wrap(_native_expr(function))
        obj._limits = limits
        return obj

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def function(self) -> Expr:
        return self._function

    @property
    def limits(self) -> Tuple[Any, ...]:
        return self._limits

    @property
    def args(self) -> Tuple[Any, ...]:
        return (self._function, *self._limits)

    @property
    def free_symbols(self) -> set:
        syms = set(self._function.free_symbols)
        for lim in self._limits:
            if isinstance(lim, (tuple, list)):
                var = lim[0]
                if var in syms:
                    syms.remove(var)
                for bound in lim[1:]:
                    if hasattr(bound, "free_symbols"):
                        syms.update(bound.free_symbols)
            elif isinstance(lim, Symbol):
                if lim in syms:
                    syms.remove(lim)
        return syms

    def doit(self, **hints: Any) -> Any:
        from .. import integrate
        func = self._function.doit(**hints) if isinstance(self._function, Basic) else self._function
        return integrate(func, *self._limits)

    def __repr__(self) -> str:
        if self._limits:
            lims_str = ", ".join(str(lim) for lim in self._limits)
            return f"Integral({self._function}, {lims_str})"
        return f"Integral({self._function})"

    def __str__(self) -> str:
        return self.__repr__()


__all__ = ["Integral"]
