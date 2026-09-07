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
        if isinstance(function, Integral):
            return Integral(function.function, *function.limits, *limits)
        obj = object.__new__(cls)
        try:
            obj._function = _wrap(_native_expr(function))
        except (NotImplementedError, TypeError):
            from ..core import sympify
            obj._function = sympify(function)
        norm_limits = []
        for lim in limits:
            if isinstance(lim, (tuple, list)):
                norm_limits.append(tuple(lim))
            else:
                norm_limits.append(lim)
        obj._limits = tuple(norm_limits)
        return obj

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def func(self) -> type:
        return Integral

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
    def variables(self) -> list[Any]:
        vars_list: list[Any] = []
        for lim in self._limits:
            if isinstance(lim, (tuple, list)):
                vars_list.append(lim[0])
            else:
                vars_list.append(lim)
        return vars_list

    @property
    def is_number(self) -> bool:
        return len(self.free_symbols) == 0

    @property
    def free_symbols(self) -> set:
        syms = set(self._function.free_symbols) if hasattr(self._function, "free_symbols") else set()
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

    def _eval_derivative(self, symbol: Any) -> Any:
        from ..core import Add, Integer, diff
        if not self._limits:
            return diff(self._function, symbol)

        if len(self._limits) > 1:
            inner = Integral(self._function, *self._limits[:-1])
            outer_lim = self._limits[-1]
            return Integral(inner, outer_lim)._eval_derivative(symbol)

        lim = self._limits[0]
        if isinstance(lim, (tuple, list)):
            if len(lim) == 1:
                var = lim[0]
                if var == symbol:
                    return self._function
                diff_func = diff(self._function, symbol)
                return Integral(diff_func, lim)
            elif len(lim) == 3:
                var, a, b = lim
                if var == symbol:
                    return Integer(0)
                terms = []
                if hasattr(b, "diff"):
                    db = diff(b, symbol)
                    if db != 0:
                        sub_b = self._function.subs(var, b) if hasattr(self._function, "subs") else self._function
                        terms.append(sub_b * db)
                if hasattr(a, "diff"):
                    da = diff(a, symbol)
                    if da != 0:
                        sub_a = self._function.subs(var, a) if hasattr(self._function, "subs") else self._function
                        terms.append(-sub_a * da)
                diff_func = diff(self._function, symbol)
                if diff_func != 0:
                    terms.append(Integral(diff_func, lim))
                if not terms:
                    return Integer(0)
                return Add(*terms) if len(terms) > 1 else terms[0]
        else:
            if lim == symbol:
                return self._function
            diff_func = diff(self._function, symbol)
            return Integral(diff_func, lim)

    def diff(self, *variables: Any) -> Any:
        from ..core import diff as core_diff
        return core_diff(self, *variables)

    def _eval_subs(self, old: Any, new: Any) -> Any:
        if self == old:
            return new
        bound_vars = set(self.variables)
        new_limits = []
        for lim in self._limits:
            if isinstance(lim, (tuple, list)):
                new_bounds = [lim[0]]
                for b in lim[1:]:
                    new_bounds.append(b.subs(old, new) if hasattr(b, "subs") else b)
                new_limits.append(tuple(new_bounds))
            else:
                new_limits.append(lim)
        if old in bound_vars:
            new_func = self._function
        else:
            new_func = self._function.subs(old, new) if hasattr(self._function, "subs") else self._function
        return Integral(new_func, *new_limits)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Integral):
            return False
        return self._function == other._function and self._limits == other._limits

    def __hash__(self) -> int:
        return hash((self.__class__, self._function, self._limits))

    def __repr__(self) -> str:
        if self._limits:
            lims_str = ", ".join(str(lim) for lim in self._limits)
            return f"Integral({self._function}, {lims_str})"
        return f"Integral({self._function})"

    def __str__(self) -> str:
        return self.__repr__()


__all__ = ["Integral"]
