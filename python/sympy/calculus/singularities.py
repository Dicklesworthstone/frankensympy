"""Singularities and monotonicity functions for FrankenSymPy calculus."""

from typing import Any
from ..core import Basic, Expr, Symbol, _native_expr, _require_symbol, _wrap
from ..sets import EmptySet, FiniteSet, Reals, Set


def _get_symbol_and_interval(expression: Any, interval: Any = None, symbol: Any = None):
    expr = _wrap(_native_expr(expression))

    if isinstance(interval, Symbol) and symbol is None:
        symbol = interval
        interval = Reals
    elif interval is None:
        interval = Reals

    if isinstance(interval, type) and issubclass(interval, Set):
        interval = interval()

    if symbol is None:
        syms = expr.free_symbols
        if len(syms) == 1:
            symbol = next(iter(syms))
        elif len(syms) == 0:
            symbol = Symbol("x")
        else:
            raise ValueError("symbol must be specified when multiple free symbols exist")
    else:
        symbol = _require_symbol(symbol)

    return expr, interval, symbol


def singularities(expression: Any, symbol: Any = None) -> Set:
    """Find the singularities of an expression with respect to symbol."""
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

    if not hasattr(expr, "as_numer_denom"):
        return EmptySet()

    _, denom = expr.as_numer_denom()
    if denom == 1 or symbol not in denom.free_symbols:
        return EmptySet()

    return solveset(denom, symbol)


def _points_to_test(interval: Any) -> list[Any]:
    from ..core import Integer, Rational
    if interval is Reals or getattr(interval, "is_UniversalSet", False) or str(interval) in ("Reals", "UniversalSet"):
        return [Integer(-10), Integer(-5), Integer(-2), Integer(-1), Rational(-1, 2), Integer(0), Rational(1, 2), Integer(1), Integer(2), Integer(5), Integer(10)]
    if hasattr(interval, "start") and hasattr(interval, "end"):
        try:
            s_str = str(interval.start)
            e_str = str(interval.end)
            if s_str in ("-oo", "-Infinity") and e_str in ("oo", "Infinity"):
                return [Integer(-10), Integer(-5), Integer(-2), Integer(-1), Rational(-1, 2), Integer(0), Rational(1, 2), Integer(1), Integer(2), Integer(5), Integer(10)]
            if s_str in ("-oo", "-Infinity"):
                e = interval.end
                return [e - 10, e - 5, e - 2, e - 1, e - Rational(1, 2)]
            if e_str in ("oo", "Infinity"):
                s = interval.start
                return [s + Rational(1, 2), s + 1, s + 2, s + 5, s + 10]
            s = interval.start
            e = interval.end
            step = (e - s) / 10
            return [s + step * i for i in range(1, 10)]
        except Exception:
            pass
    return [Integer(-5), Integer(-1), Integer(0), Integer(1), Integer(5)]


def is_increasing(expression: Any, interval: Any = None, symbol: Any = None) -> bool:
    """Return True if expression is increasing on interval, False otherwise."""
    from ..core import diff
    expr, interval, symbol = _get_symbol_and_interval(expression, interval, symbol)
    d = diff(expr, symbol)
    if not d.free_symbols:
        try:
            return d >= 0
        except Exception:
            return float(d) >= 0

    test_points = _points_to_test(interval)
    for pt in test_points:
        try:
            if pt in interval:
                val = d.subs(symbol, pt)
                if val < 0:
                    return False
        except Exception:
            pass
    return True


def is_strictly_increasing(expression: Any, interval: Any = None, symbol: Any = None) -> bool:
    """Return True if expression is strictly increasing on interval, False otherwise."""
    from ..core import diff
    expr, interval, symbol = _get_symbol_and_interval(expression, interval, symbol)
    d = diff(expr, symbol)
    if not d.free_symbols:
        try:
            return d > 0
        except Exception:
            return float(d) > 0

    test_points = _points_to_test(interval)
    for pt in test_points:
        try:
            if pt in interval:
                val = d.subs(symbol, pt)
                if val <= 0:
                    return False
        except Exception:
            pass
    return True


def is_decreasing(expression: Any, interval: Any = None, symbol: Any = None) -> bool:
    """Return True if expression is decreasing on interval, False otherwise."""
    from ..core import diff
    expr, interval, symbol = _get_symbol_and_interval(expression, interval, symbol)
    d = diff(expr, symbol)
    if not d.free_symbols:
        try:
            return d <= 0
        except Exception:
            return float(d) <= 0

    test_points = _points_to_test(interval)
    for pt in test_points:
        try:
            if pt in interval:
                val = d.subs(symbol, pt)
                if val > 0:
                    return False
        except Exception:
            pass
    return True


def is_strictly_decreasing(expression: Any, interval: Any = None, symbol: Any = None) -> bool:
    """Return True if expression is strictly decreasing on interval, False otherwise."""
    from ..core import diff
    expr, interval, symbol = _get_symbol_and_interval(expression, interval, symbol)
    d = diff(expr, symbol)
    if not d.free_symbols:
        try:
            return d < 0
        except Exception:
            return float(d) < 0

    test_points = _points_to_test(interval)
    for pt in test_points:
        try:
            if pt in interval:
                val = d.subs(symbol, pt)
                if val >= 0:
                    return False
        except Exception:
            pass
    return True


def is_monotonic(expression: Any, interval: Any = None, symbol: Any = None) -> bool:
    """Return True if expression is monotonic on interval, False otherwise."""
    return is_increasing(expression, interval, symbol) or is_decreasing(expression, interval, symbol)


__all__ = [
    "is_decreasing",
    "is_increasing",
    "is_monotonic",
    "is_strictly_decreasing",
    "is_strictly_increasing",
    "singularities",
]
