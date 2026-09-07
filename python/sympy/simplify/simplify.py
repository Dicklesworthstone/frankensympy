from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Iterable

from ..core import (
    Add,
    Basic,
    Expr,
    Integer,
    Mul,
    Pow,
    Rational,
    Symbol,
    pi,
    simplify as _core_simplify,
    sqrt,
    sympify,
)
from ..polys import cancel, together


def simplify(expr: Any, **kwargs: Any) -> Any:
    """General expression simplification."""
    return _core_simplify(expr)


def trigsimp(expr: Any, **kwargs: Any) -> Any:
    """Trigonometric and hyperbolic expression simplification."""
    return _core_simplify(expr)


def powsimp(expr: Any, **kwargs: Any) -> Any:
    """Simplify products of powers by combining exponents."""
    return _core_simplify(expr)


def combsimp(expr: Any) -> Any:
    """Combinatorial expression simplification."""
    return _core_simplify(expr)


def ratsimp(expr: Any) -> Any:
    """Put rational expression over a common denominator and cancel factors."""
    return cancel(together(expr))


def radsimp(expr: Any, **kwargs: Any) -> Any:
    """Rationalize the denominator of a radical expression."""
    expr = sympify(expr)
    if isinstance(expr, Pow) and expr.args[1] == -1:
        den = expr.args[0]
        if isinstance(den, Pow) and den.args[1] == Rational(1, 2):
            c = den.args[0]
            return Mul(sqrt(c), Pow(c, -1))
        if isinstance(den, Add) and len(den.args) == 2:
            term1, term2 = den.args

            def _get_sqrt(t: Any):
                if isinstance(t, Pow) and t.args[1] == Rational(1, 2):
                    return Integer(1), t.args[0]
                if isinstance(t, Mul):
                    for f in t.args:
                        if isinstance(f, Pow) and f.args[1] == Rational(1, 2):
                            rest = [x for x in t.args if x != f]
                            coeff = Mul(*rest) if len(rest) > 1 else rest[0]
                            return coeff, f.args[0]
                return None

            s1 = _get_sqrt(term1)
            s2 = _get_sqrt(term2)
            if s2 is not None and s1 is None:
                a = term1
                b, c = s2
                conj = a - b * sqrt(c)
                den_norm = a**2 - b**2 * c
                if isinstance(conj, Add):
                    return Add(*(t / den_norm for t in conj.args))
                return conj / den_norm
            elif s1 is not None and s2 is None:
                a = term2
                b, c = s1
                conj = a - b * sqrt(c)
                den_norm = a**2 - b**2 * c
                if isinstance(conj, Add):
                    return Add(*(t / den_norm for t in conj.args))
                return conj / den_norm
    return expr


def logcombine(expr: Any, **kwargs: Any) -> Any:
    """Combine logarithmic terms: log(x) + log(y) => log(x*y)."""
    expr = sympify(expr)
    if not isinstance(expr, Add):
        return expr
    from ..functions import log

    logs: list[tuple[Any, Any]] = []
    rest: list[Any] = []
    for t in expr.args:
        if getattr(getattr(t, "func", None), "__name__", "") in ("log", "ln"):
            logs.append((t.args[0], Integer(1)))
        elif isinstance(t, Mul):
            log_factors = [f for f in t.args if getattr(getattr(f, "func", None), "__name__", "") in ("log", "ln")]
            other_factors = [f for f in t.args if getattr(getattr(f, "func", None), "__name__", "") not in ("log", "ln")]
            if len(log_factors) == 1:
                coeff = Mul(*other_factors) if len(other_factors) > 1 else other_factors[0] if other_factors else Integer(1)
                logs.append((log_factors[0].args[0], coeff))
            else:
                rest.append(t)
        else:
            rest.append(t)

    if len(logs) > 1:
        inner_factors = [Pow(u, c) if c != 1 else u for u, c in logs]
        combined = log(Mul(*inner_factors))
        if rest:
            return Add(combined, *rest)
        return combined
    return expr


def collect(expr: Any, syms: Any, evaluate: bool = True) -> Any:
    """Collect additive terms with respect to a symbol or expression."""
    expr = sympify(expr)
    if isinstance(syms, (list, tuple, set)):
        sym = next(iter(syms))
    else:
        sym = syms

    if not isinstance(expr, Add):
        return expr

    coeff_map: dict[Any, list[Any]] = {}
    rest: list[Any] = []
    for term in expr.args:
        if term == sym:
            coeff_map.setdefault(1, []).append(Integer(1))
        elif isinstance(term, Pow) and term.args[0] == sym:
            coeff_map.setdefault(term.args[1], []).append(Integer(1))
        elif isinstance(term, Mul) and sym in term.args:
            factors = [f for f in term.args if f != sym]
            c = Mul(*factors) if len(factors) > 1 else factors[0] if factors else Integer(1)
            coeff_map.setdefault(1, []).append(c)
        else:
            found = False
            if isinstance(term, Mul):
                for i, f in enumerate(term.args):
                    if isinstance(f, Pow) and f.args[0] == sym:
                        factors = [term.args[j] for j in range(len(term.args)) if j != i]
                        c = Mul(*factors) if len(factors) > 1 else factors[0] if factors else Integer(1)
                        coeff_map.setdefault(f.args[1], []).append(c)
                        found = True
                        break
            if not found:
                rest.append(term)

    res_terms: list[Any] = []
    for p, coeffs in coeff_map.items():
        c_expr = Add(*coeffs) if len(coeffs) > 1 else coeffs[0]
        if p == 1:
            res_terms.append(c_expr * sym)
        else:
            res_terms.append(c_expr * Pow(sym, p))
    if rest:
        res_terms.extend(rest)
    if not res_terms:
        return Integer(0)
    return Add(*res_terms) if len(res_terms) > 1 else res_terms[0]


def separatevars(expr: Any, symbols: Any = None, dict: bool = False) -> Any:
    """Separate multiplicative factors in an expression."""
    expr = sympify(expr)
    return expr


def nsimplify(expr: Any, constants: Iterable[Any] = (), tolerance: float | None = None, full: bool = False, rational: bool | None = None) -> Any:
    """Find a simple exact formula that approximates a numerical expression."""
    tol = tolerance if tolerance is not None else 1e-10
    try:
        val = float(expr)
    except Exception:
        return expr

    if abs(val - round(val)) < tol:
        return Integer(int(round(val)))
    if abs(val - math.pi) < tol:
        return pi
    if abs(val - math.e) < tol:
        from ..core import E
        return E
    val_sq = val**2
    if val > 0 and abs(val_sq - round(val_sq)) < tol:
        sq_int = int(round(val_sq))
        if sq_int > 0:
            return sqrt(sq_int)
    frac = Fraction(val).limit_denominator(10000)
    if abs(float(frac) - val) < tol:
        return Rational(frac.numerator, frac.denominator)
    return expr


__all__ = [
    "collect",
    "combsimp",
    "logcombine",
    "nsimplify",
    "powsimp",
    "radsimp",
    "ratsimp",
    "separatevars",
    "simplify",
    "trigsimp",
]
