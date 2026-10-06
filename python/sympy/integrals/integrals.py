"""Unevaluated integrals for FrankenSymPy.

``Integral(f, (x, a, b), ...)`` is a native binding construct: its args are
``(f, Tuple(x, a, b), ...)`` (or ``Tuple(x)`` for an indefinite limit), so it
takes part in native arithmetic, while the kernel treats the limit variables
as bound — substitution never reaches them inside the body and
differentiation follows the Leibniz rule.
"""

from __future__ import annotations

from typing import Any

from ..core import Function, Symbol, Tuple, sympify


def _limit_tuple(lim: Any) -> Any:
    if type(lim).__name__ == "Tuple":
        items = tuple(lim.args)
    elif isinstance(lim, (tuple, list)):
        items = tuple(sympify(v) for v in lim)
    else:
        items = (sympify(lim),)
    if not items or not isinstance(items[0], Symbol):
        raise ValueError("Invalid limits given: %s" % (lim,))
    if len(items) not in (1, 3):
        raise ValueError("integration limits must be (x,) or (x, a, b): %s" % (lim,))
    return Tuple(*items)


class Integral(Function):
    """An unevaluated integral."""

    __slots__ = ()

    def __new__(cls, function: Any, *limits: Any, evaluate: bool = False, **options: Any):
        if evaluate:
            from .. import integrate

            return integrate(function, *limits)
        function = sympify(function)
        lims = [_limit_tuple(lim) for lim in limits]
        if isinstance(function, Integral):
            lims = list(function.args[1:]) + lims
            function = function.function
        if not lims:
            free = sorted(function.free_symbols, key=lambda s: s.name)
            if len(free) != 1:
                raise ValueError(
                    "specify integration variables to integrate %s" % function
                )
            lims = [Tuple(free[0])]
        return Function.__new__(cls, function, *lims, evaluate=False)

    @property
    def function(self) -> Any:
        return self.args[0]

    @property
    def limits(self) -> tuple:
        return tuple(tuple(t.args) for t in self.args[1:])

    @property
    def variables(self) -> list:
        return [lim[0] for lim in self.limits]

    @property
    def free_symbols(self) -> set:
        syms = set(self.function.free_symbols)
        for lim in self.limits:
            syms.discard(lim[0])
            for bound in lim[1:]:
                syms |= set(getattr(bound, "free_symbols", set()))
        return syms

    @property
    def is_number(self) -> bool:
        return not self.free_symbols

    def doit(self, **hints: Any) -> Any:
        from .. import integrate

        func = self.function
        if hints.get("deep", True) and hasattr(func, "doit"):
            func = func.doit(**hints)
        return integrate(func, *self.limits)


__all__ = ["Integral"]
