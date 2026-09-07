"""Calculus utilities for FrankenSymPy."""

from typing import Any
from ..core import Basic, Expr, Integer, Symbol, _native_expr, _require_symbol, _wrap
from ..sets import Complement, EmptySet, FiniteSet, Interval, Reals, Set
from .singularities import singularities

pi = Expr("pi")


class AccumBounds(Basic):
    """An interval representation of the accumulation points of an oscillating function."""

    __slots__ = ("_min", "_max")

    def __new__(cls, min: Any, max: Any) -> "AccumBounds":
        min_val = _wrap(_native_expr(min))
        max_val = _wrap(_native_expr(max))
        obj = object.__new__(cls)
        obj._min = min_val
        obj._max = max_val
        return obj

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def min(self) -> Any:
        return self._min

    @property
    def max(self) -> Any:
        return self._max

    @property
    def args(self) -> tuple[Any, Any]:
        return (self._min, self._max)

    def __repr__(self) -> str:
        return f"AccumBounds({self._min}, {self._max})"

    def __str__(self) -> str:
        return self.__repr__()

    def __add__(self, other: Any) -> "AccumBounds":
        if isinstance(other, AccumBounds):
            return AccumBounds(self.min + other.min, self.max + other.max)
        other_expr = _wrap(_native_expr(other))
        return AccumBounds(self.min + other_expr, self.max + other_expr)

    def __radd__(self, other: Any) -> "AccumBounds":
        return self.__add__(other)

    def __neg__(self) -> "AccumBounds":
        return AccumBounds(-self.max, -self.min)

    def __sub__(self, other: Any) -> "AccumBounds":
        if isinstance(other, AccumBounds):
            return AccumBounds(self.min - other.max, self.max - other.min)
        other_expr = _wrap(_native_expr(other))
        return AccumBounds(self.min - other_expr, self.max - other_expr)

    def __rsub__(self, other: Any) -> "AccumBounds":
        return (-self) + other

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AccumBounds):
            return self.min == other.min and self.max == other.max
        return False

    def __mul__(self, other: Any) -> "AccumBounds":
        c = _wrap(_native_expr(other))
        p1 = self.min * c
        p2 = self.max * c
        try:
            if float(c) < 0:
                return AccumBounds(p2, p1)
        except Exception:
            pass
        return AccumBounds(p1, p2)

    def __rmul__(self, other: Any) -> "AccumBounds":
        return self.__mul__(other)


AccumulationBounds = AccumBounds


def continuous_domain(expression: Any, symbol: Any = None, domain: Any = None) -> Set:
    """Return the domain on which expression is continuous."""
    if domain is None:
        domain = Reals
    sings = singularities(expression, symbol)
    if getattr(sings, "is_empty", None) is True or len(getattr(sings, "args", ())) == 0:
        return domain
    return Complement(domain, sings)


def stationary_points(expression: Any, symbol: Any = None, domain: Any = None) -> Set:
    """Return the stationary points of expression where its derivative is zero."""
    from ..core import diff
    from ..solvers import solveset

    expr = _wrap(_native_expr(expression))
    if symbol is None:
        syms = expr.free_symbols
        if len(syms) == 1:
            symbol = next(iter(syms))
        elif len(syms) == 0:
            return EmptySet()
        else:
            raise ValueError("symbol must be specified when multiple free symbols exist")
    else:
        symbol = _require_symbol(symbol)
    d = diff(expr, symbol)
    return solveset(d, symbol, domain=domain)


def periodicity(f: Any, symbol: Any = None, check: bool = False) -> Any:
    """Return the fundamental period of f with respect to symbol, or None if not periodic."""
    expr = _wrap(_native_expr(f))
    if symbol is None:
        syms = expr.free_symbols
        if len(syms) == 1:
            symbol = next(iter(syms))
        elif len(syms) == 0:
            return Integer(0)
        else:
            raise ValueError("symbol must be specified when multiple free symbols exist")
    else:
        symbol = _require_symbol(symbol)

    if symbol not in expr.free_symbols:
        return Integer(0)

    func_name = getattr(getattr(expr, "func", None), "__name__", None) or str(getattr(expr, "func", None))
    if func_name in ("sin", "cos", "sec", "csc", "tan", "cot"):
        arg = expr.args[0]
        from ..polys import Poly
        from ..core import Rational
        try:
            p = Poly(arg, symbol)
            if p.degree() == 1:
                a = p.leading_coeff()
                a_val = abs(a)
                num = 2 if func_name in ("sin", "cos", "sec", "csc") else 1
                coeff = Rational(num, a_val)
                if coeff == 1:
                    return pi
                return coeff * pi
        except Exception:
            pass

    return None


def _critical_and_boundary_values(f: Any, symbol: Any = None, domain: Any = None) -> list[Any]:
    expr = _wrap(_native_expr(f))
    if symbol is None:
        syms = expr.free_symbols
        if len(syms) == 1:
            symbol = next(iter(syms))
        elif len(syms) == 0:
            return [expr]
        else:
            raise ValueError("symbol must be specified when multiple free symbols exist")
    else:
        symbol = _require_symbol(symbol)

    pts = stationary_points(expr, symbol, domain)
    cand_x = list(getattr(pts, "args", ()))
    if domain is not None and hasattr(domain, "start") and hasattr(domain, "end"):
        s_str = str(domain.start)
        e_str = str(domain.end)
        if s_str not in ("-oo", "-Infinity", "oo", "Infinity"):
            cand_x.append(domain.start)
        if e_str not in ("-oo", "-Infinity", "oo", "Infinity"):
            cand_x.append(domain.end)

    vals = []
    for cx in cand_x:
        try:
            val = expr.subs(symbol, cx)
            vals.append(val)
        except Exception:
            pass
    return vals


def function_range(f: Any, symbol: Any, domain: Any) -> Set:
    """Return the range of f over the given domain."""
    vals = _critical_and_boundary_values(f, symbol, domain)
    if vals:
        return Interval(min(vals), max(vals))
    return domain


def maximum(f: Any, symbol: Any = None, domain: Any = None) -> Any:
    """Return the maximum of f over domain."""
    vals = _critical_and_boundary_values(f, symbol, domain)
    if vals:
        return max(vals)
    raise ValueError("maximum could not be determined")


def minimum(f: Any, symbol: Any = None, domain: Any = None) -> Any:
    """Return the minimum of f over domain."""
    vals = _critical_and_boundary_values(f, symbol, domain)
    if vals:
        return min(vals)
    raise ValueError("minimum could not be determined")


__all__ = [
    "AccumBounds",
    "AccumulationBounds",
    "continuous_domain",
    "function_range",
    "maximum",
    "minimum",
    "periodicity",
    "stationary_points",
]
