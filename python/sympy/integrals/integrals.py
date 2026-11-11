"""Unevaluated integrals for FrankenSymPy.

``Integral(f, (x, a, b), ...)`` has args
``(f, Tuple(x, a, b), ...)`` (or ``Tuple(x)`` for an indefinite limit).
An indefinite integration coordinate remains free, even for a constant
integrand. Definite limits and evaluation limits ``Tuple(x, point)`` bind
their coordinate in the integrand and all earlier limits. Their own
endpoints remain outside that scope. Public substitution is owned by the
shell so profile-visible binding behavior survives native arithmetic.
"""

from __future__ import annotations

from typing import Any

from ..core import AppliedUndef, Function, Symbol, Tuple, sympify


def _limit_tuple(lim: Any) -> Any:
    if type(lim).__name__ == "Tuple":
        items = tuple(lim.args)
    elif isinstance(lim, (tuple, list)):
        items = tuple(sympify(v) for v in lim)
    else:
        items = (sympify(lim),)
    if not items or not isinstance(items[0], Symbol):
        raise ValueError("Invalid limits given: %s" % (lim,))
    if len(items) not in (1, 2, 3):
        raise ValueError("integration limits must be (x,), (x, p), or (x, a, b): %s" % (lim,))
    return Tuple(*items)


class Integral(Function):
    """An unevaluated integral."""

    __slots__ = ()

    def __new__(cls, function: Any, *limits: Any, evaluate: bool = False, **options: Any):
        if evaluate:
            return cls(function, *limits, evaluate=False, **options).doit()
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
    def bound_symbols(self) -> list:
        return [lim[0] for lim in self.limits if len(lim) > 1]

    @property
    def free_symbols(self) -> set:
        syms = set(self.function.free_symbols)
        for lim in self.limits:
            if len(lim) == 1:
                # Even Integral(1, x) depends on x. Work from the innermost
                # limit out so a later definite limit can bind this occurrence.
                syms.add(lim[0])
            else:
                syms.discard(lim[0])
                for bound in lim[1:]:
                    syms |= set(bound.free_symbols)
        return syms

    @property
    def is_number(self) -> bool:
        return not self.free_symbols

    def _eval_subs(self, old: Any, new: Any) -> Any:
        """Substitute from the outermost limit inward using public scopes.

        Endpoints are outside their own binding scope. An indefinite
        coordinate may be renamed to a symbol, while replacement by a value
        evaluates the primitive there. Compound substitutions cannot replace
        occurrences involving an integration coordinate in the integrand.
        """
        limits = list(self.limits)
        blocked = False
        old_symbols = old.free_symbols
        for index in range(len(limits) - 1, -1, -1):
            if blocked:
                break
            limit = limits[index]
            variable = limit[0]
            if len(limit) == 1 and old == variable:
                if isinstance(new, Symbol):
                    limits[index] = (new,)
                else:
                    limits[index] = (variable, new)
                    blocked = True
                continue
            bounds = tuple(bound.subs(old, new) for bound in limit[1:])
            limits[index] = (variable, *bounds)
            if variable in old_symbols:
                blocked = True

        if isinstance(old, AppliedUndef):
            # Replacing an unknown function is allowed under an integral,
            # provided its explicit arguments already name any integration
            # coordinates used by the replacement.
            coordinates = set(self.variables)
            introduced = coordinates.intersection(new.atoms(Symbol))
            permitted = coordinates.intersection(old.args)
            if not introduced.issubset(permitted):
                raise ValueError("substitution cannot create dummy dependencies")
            blocked = False

        limits = [
            (limit[0],) if len(limit) == 2 and limit[1] == limit[0] else limit
            for limit in limits
        ]
        function = self.function if blocked else self.function.subs(old, new)
        if function is self.function and tuple(limits) == self.limits:
            return self
        return self.func(function, *limits)

    def doit(self, **hints: Any) -> Any:
        from .. import integrate

        func = self.function
        if hints.get("deep", True) and hasattr(func, "doit"):
            func = func.doit(**hints)
        limits = self.limits
        if all(len(limit) != 2 for limit in limits):
            return integrate(func, *limits)
        for index, limit in enumerate(limits):
            if len(limit) == 2:
                variable, point = limit
                primitive = integrate(func, variable)
                if isinstance(primitive, Integral):
                    # Retain the evaluation point and remaining scopes if
                    # the existing integration engine cannot find a primitive.
                    return self.func(func, *limits[index:])
                func = primitive.subs(variable, point)
            else:
                func = integrate(func, limit)
        return func


__all__ = ["Integral"]
