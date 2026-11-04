"""Singularities and bounded exact monotonicity for FrankenSymPy calculus."""

from __future__ import annotations

from typing import Any
from ..core import Basic, Expr, Symbol, _native_expr, _require_symbol, _wrap
from ..sets import EmptySet, FiniteSet, Reals, Set


def _get_symbol_and_interval(expression: Any, interval: Any = None, symbol: Any = None):
    """Admit the original rational surface before differentiation can erase it."""
    from ..core import Add, Dummy, Half, Integer, Mul, NegativeOne, One, Pow, Rational, Zero

    expr = expression if isinstance(expression, Basic) else _wrap(_native_expr(expression))
    pending = [(expr, 0)]
    symbols = set()
    visited = 0
    while pending:
        node, depth = pending.pop()
        visited += 1
        if visited > 2048 or depth > 64:
            raise NotImplementedError("exact monotonicity source limit exceeded")
        kind = type(node)
        if kind in (Integer, Rational, Zero, One, NegativeOne, Half):
            continue
        if kind in (Symbol, Dummy):
            symbols.add(node)
            continue
        if kind not in (Add, Mul, Pow):
            raise NotImplementedError("monotonicity currently requires an exact rational function")
        children = node.args
        if kind is Pow and type(children[1]) not in (Integer, Zero, One, NegativeOne):
            raise NotImplementedError("monotonicity currently requires integer powers")
        if len(children) > 256 or len(children) > 2048 - visited - len(pending):
            raise NotImplementedError("exact monotonicity source limit exceeded")
        pending.extend((child, depth + 1) for child in children)

    if isinstance(interval, Symbol) and symbol is None:
        symbol = interval
        interval = Reals()
    elif interval is None or interval is Reals:
        interval = Reals()
    elif interval is EmptySet:
        interval = EmptySet()

    if symbol is None:
        if len(symbols) > 1:
            raise NotImplementedError("monotonicity is not implemented for multivariate expressions")
        symbol = next(iter(symbols)) if symbols else Symbol("x")
    else:
        symbol = _require_symbol(symbol)
    if symbols - {symbol}:
        raise NotImplementedError("monotonicity currently requires rational coefficients")

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


def _monotonicity_bound(value):
    """An exact extended-real bound; no approximate ordering is admitted."""
    from ..core import Half, Integer, NegativeOne, One, Rational, Zero, oo
    from ..polys.rootoftools import CRootOf

    if type(value) in (Integer, Rational, Zero, One, NegativeOne, Half):
        if max(abs(value.p).bit_length(), value.q.bit_length()) > 4096:
            raise NotImplementedError("monotonicity bound coefficient limit exceeded")
        return 0, value
    if type(value) is Expr:
        if value == oo:
            return 1, None
        if value == -oo:
            return -1, None
    if type(value) is CRootOf:
        if value.is_real is not True:
            raise NotImplementedError("monotonicity requires real algebraic bounds")
        return 0, value
    raise NotImplementedError(
        "monotonicity bounds require exact rational numbers or real CRootOf values"
    )


def _compare_monotonicity_bounds(left, right):
    if left[0] != right[0]:
        return (left[0] > right[0]) - (left[0] < right[0])
    if left[0]:
        return 0
    from ..core import Half, Integer, NegativeOne, One, Rational, Zero
    from ..polys.rootoftools import _algebraic_comparison

    a, b = left[1], right[1]
    atoms = (Integer, Rational, Zero, One, NegativeOne, Half)
    if type(a) in atoms and type(b) in atoms:
        difference = a.p * b.q - b.p * a.q
        return (difference > 0) - (difference < 0)
    # Every non-rational endpoint is an exact built-in CRootOf. This helper
    # checks the selected real embeddings; it has no binary64 fallback.
    return _algebraic_comparison(a, b)


def _monotonicity_interval(interval):
    from ..sets import Interval

    if type(interval) is EmptySet:
        return None
    if type(interval) not in (Interval, Reals):
        if not isinstance(interval, Set):
            raise TypeError("monotonicity interval must be a real interval")
        raise NotImplementedError("monotonicity currently supports connected real intervals")
    lower = _monotonicity_bound(interval.start)
    upper = _monotonicity_bound(interval.end)
    order = _compare_monotonicity_bounds(lower, upper)
    if order > 0 or (order == 0 and (lower[0] or interval.left_open or interval.right_open)):
        return None
    return lower, upper, interval.left_open, interval.right_open


def _monotonicity_chart(expression, symbol, operation):
    """Lift the complete checked sign chart without replacing roots by radicals."""
    from ..core import _lift_native_result, _native, _symbol_binding
    from ..polys.polytools import Poly
    from ..polys.rootoftools import real_roots
    from ..solvers.inequalities import _rational_surface

    if not _rational_surface(expression, symbol):
        raise NotImplementedError("exact monotonicity has an unsupported derivative")
    checked = _native.solve_real_rational_inequality_native(
        _native_expr(expression), _symbol_binding(symbol), operation,
    )
    if checked is None:
        raise NotImplementedError("exact monotonicity could not admit its rational source")
    polynomial, gaps, points = checked
    count = len(points)
    if count > 64 or len(gaps) != count + 1:
        raise ValueError("incomplete monotonicity sign chart")
    if any(type(value) is not bool for value in (*gaps, *points)):
        raise ValueError("invalid monotonicity sign chart masks")
    if count:
        polynomial = _lift_native_result(polynomial, expression, symbol)
        associated = real_roots(Poly(polynomial, symbol), multiple=False, radicals=False)
        if len(associated) != count or any(multiplicity != 1 for _, multiplicity in associated):
            raise ValueError("incomplete monotonicity critical-root association")
        if len({root for root, _ in associated}) != count:
            raise ValueError("distinct monotonicity critical roots were lost during lifting")
        roots = [(0, root) for root, _ in associated]
    else:
        roots = []
    cells = [False] * (2 * count + 1)
    cells[::2] = gaps
    cells[1::2] = points
    return roots, cells


def _monotonicity_cell(bound, roots):
    """Locate a bound in the alternating open-gap/root cells by exact order."""
    if bound[0] < 0:
        return 0, False
    if bound[0] > 0:
        return 2 * len(roots), False
    lower, upper = 0, len(roots)
    while lower < upper:
        middle = (lower + upper) // 2
        if _compare_monotonicity_bounds(roots[middle], bound) < 0:
            lower = middle + 1
        else:
            upper = middle
    if lower < len(roots) and _compare_monotonicity_bounds(roots[lower], bound) == 0:
        return 2 * lower + 1, True
    return 2 * lower, False


def _monotonicity_covered(chart, interval):
    roots, cells = chart
    lower, upper, left_open, right_open = interval
    first, first_is_root = _monotonicity_cell(lower, roots)
    last, last_is_root = _monotonicity_cell(upper, roots)
    if first_is_root and left_open:
        first += 1
    if last_is_root and right_open:
        last -= 1
    if first > last:
        raise ValueError("nonempty interval has no monotonicity chart cells")
    # Nonempty connected intervals meet every cell between their endpoints.
    # A disallowed isolated root therefore cannot disappear between samples.
    return all(cells[first:last + 1])


def _exact_monotonicity(expression, interval, symbol, operation):
    from ..core import Mul, diff_with_receipt

    expr, interval, symbol = _get_symbol_and_interval(expression, interval, symbol)
    try:
        domain = _monotonicity_interval(interval)
        if domain is None:
            return True
        # The checker visits all original children before multiplying by zero.
        # This retains poles hidden by cancellation, zero products and powers.
        original_domain = _monotonicity_chart(Mul(0, expr, evaluate=False), symbol, ">=")
        if not _monotonicity_covered(original_domain, domain):
            return False
        derivative, _receipt = diff_with_receipt(expr, symbol)
        operations = (">=", "<=") if operation is None else (operation,)
        for relation in operations:
            if _monotonicity_covered(_monotonicity_chart(derivative, symbol, relation), domain):
                return True
        return False
    except (ValueError, OverflowError) as exc:
        # An exhausted native check or undecidable endpoint is not evidence
        # for either sign. Do not restart through a numerical/sample fallback.
        raise NotImplementedError("exact monotonicity could not be determined: %s" % exc) from None


def is_increasing(expression: Any, interval: Any = None, symbol: Any = None) -> bool:
    """Check a nonnegative derivative throughout an exact real interval.

    The bounded lane admits rational functions and rational/real CRootOf
    endpoints. Original domain exclusions are retained. Unsupported syntax,
    endpoints or exhausted native checks raise NotImplementedError.
    """
    return _exact_monotonicity(expression, interval, symbol, ">=")


def is_strictly_increasing(expression: Any, interval: Any = None, symbol: Any = None) -> bool:
    """Check a positive derivative at every included point, as in SymPy 1.14.

    In particular an included isolated derivative zero makes this predicate
    false; excluding an endpoint zero can change the answer.
    """
    return _exact_monotonicity(expression, interval, symbol, ">")


def is_decreasing(expression: Any, interval: Any = None, symbol: Any = None) -> bool:
    """Check a nonpositive derivative throughout an admitted real interval."""
    return _exact_monotonicity(expression, interval, symbol, "<=")


def is_strictly_decreasing(expression: Any, interval: Any = None, symbol: Any = None) -> bool:
    """Check a negative derivative at every included point of the interval."""
    return _exact_monotonicity(expression, interval, symbol, "<")


def is_monotonic(expression: Any, interval: Any = None, symbol: Any = None) -> bool:
    """Check that one weak derivative sign holds throughout the interval.

    This retains the shell's existing increasing-or-decreasing convention.
    It is not the upstream 1.14 turning-point-absence convention: for example,
    this shell regards x**3 as monotonic even though its derivative vanishes.
    """
    return _exact_monotonicity(expression, interval, symbol, None)


__all__ = [
    "is_decreasing",
    "is_increasing",
    "is_monotonic",
    "is_strictly_decreasing",
    "is_strictly_increasing",
    "singularities",
]
