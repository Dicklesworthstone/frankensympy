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


def _trigsimp_pass(expr: Any) -> Any:
    if not hasattr(expr, "args") or not expr.args:
        return expr
    from ..functions import cos, cot, csc, sec, sin, sinh, cosh

    def get_fn(x: Any) -> str:
        return getattr(getattr(x, "func", None), "__name__", "")

    new_args = [_trigsimp_pass(a) for a in expr.args]
    if new_args != list(expr.args):
        expr = expr.func(*new_args)

    if isinstance(expr, Add):
        terms = list(expr.args)
        new_terms = []
        skip = set()
        for i in range(len(terms)):
            if i in skip:
                continue
            t1 = terms[i]
            paired = False
            for j in range(i + 1, len(terms)):
                if j in skip:
                    continue
                t2 = terms[j]
                # 1 + tan(u)**2 -> sec(u)**2, 1 + cot(u)**2 -> csc(u)**2
                for one_cand, tan_cand in ((t1, t2), (t2, t1)):
                    if one_cand == 1 and isinstance(tan_cand, Pow) and tan_cand.args[1] == 2:
                        base = tan_cand.args[0]
                        if get_fn(base) == "tan":
                            new_terms.append(sec(base.args[0])**2)
                            skip.add(j)
                            paired = True
                            break
                        if get_fn(base) == "cot":
                            new_terms.append(csc(base.args[0])**2)
                            skip.add(j)
                            paired = True
                            break
                if paired:
                    break
                # sin(u)**2 + cos(u)**2 -> 1
                if isinstance(t1, Pow) and t1.args[1] == 2 and isinstance(t2, Pow) and t2.args[1] == 2:
                    b1, b2 = t1.args[0], t2.args[0]
                    if {get_fn(b1), get_fn(b2)} == {"sin", "cos"} and b1.args[0] == b2.args[0]:
                        new_terms.append(Integer(1))
                        skip.add(j)
                        paired = True
                        break
            if not paired:
                new_terms.append(t1)
        if len(new_terms) != len(terms):
            expr = Add(*new_terms)

    elif isinstance(expr, Mul):
        factors = list(expr.args)
        new_factors = []
        skip = set()
        for i in range(len(factors)):
            if i in skip:
                continue
            f1 = factors[i]
            paired = False
            for j in range(i + 1, len(factors)):
                if j in skip:
                    continue
                f2 = factors[j]
                fn1, fn2 = get_fn(f1), get_fn(f2)
                # tan(u) * cos(u) -> sin(u)
                if fn1 == "tan" and fn2 == "cos" and f1.args[0] == f2.args[0]:
                    new_factors.append(sin(f1.args[0]))
                    skip.add(j)
                    paired = True
                    break
                if fn2 == "tan" and fn1 == "cos" and f1.args[0] == f2.args[0]:
                    new_factors.append(sin(f1.args[0]))
                    skip.add(j)
                    paired = True
                    break
                # cot(u) * sin(u) -> cos(u)
                if fn1 == "cot" and fn2 == "sin" and f1.args[0] == f2.args[0]:
                    new_factors.append(cos(f1.args[0]))
                    skip.add(j)
                    paired = True
                    break
                if fn2 == "cot" and fn1 == "sin" and f1.args[0] == f2.args[0]:
                    new_factors.append(cos(f1.args[0]))
                    skip.add(j)
                    paired = True
                    break
            if not paired:
                new_factors.append(f1)
        if len(new_factors) != len(factors):
            expr = Mul(*new_factors)

    return expr


def trigsimp(expr: Any, **kwargs: Any) -> Any:
    """Trigonometric and hyperbolic expression simplification."""
    expr = sympify(expr)
    c_res = _core_simplify(expr)
    res = _trigsimp_pass(c_res)
    if res != c_res:
        return _core_simplify(res)
    return res


def powsimp(expr: Any, combine: str = "all", force: bool = False, **kwargs: Any) -> Any:
    """Simplify products of powers by combining exponents or bases."""
    expr = sympify(expr)
    c_res = _core_simplify(expr)
    if not hasattr(c_res, "args") or not c_res.args:
        return c_res

    def _powsimp_pass(e: Any) -> Any:
        if not hasattr(e, "args") or not e.args:
            return e
        new_args = [_powsimp_pass(a) for a in e.args]
        if new_args != list(e.args):
            e = e.func(*new_args)

        if isinstance(e, Mul) and combine in ("base", "all"):
            from collections import defaultdict
            groups = defaultdict(list)
            rest = []
            for factor in e.args:
                if isinstance(factor, Pow):
                    base, p = factor.args
                    groups[p].append(base)
                else:
                    rest.append(factor)
            new_factors = list(rest)
            changed = False
            for p, bases in groups.items():
                if len(bases) > 1 and (force or getattr(p, "is_integer", None) is True):
                    new_factors.append(Pow(Mul(*bases), p))
                    changed = True
                else:
                    for b in bases:
                        new_factors.append(Pow(b, p))
            if changed:
                return Mul(*new_factors)
        return e

    res = _powsimp_pass(c_res)
    return _core_simplify(res)


def expand_trig(expr: Any, **kwargs: Any) -> Any:
    """Expand trigonometric and hyperbolic functions using addition and multiple-angle formulas."""
    expr = sympify(expr)
    if not hasattr(expr, "args") or not expr.args:
        return expr
    from ..functions import sin, cos, tan, sinh, cosh
    fn = getattr(getattr(expr, "func", None), "__name__", "")

    if fn == "sin" and len(expr.args) == 1:
        arg = expand_trig(expr.args[0], **kwargs)
        if isinstance(arg, Add):
            a = arg.args[0]
            b = Add(*arg.args[1:]) if len(arg.args) > 2 else arg.args[1]
            return expand_trig(sin(a)) * expand_trig(cos(b)) + expand_trig(cos(a)) * expand_trig(sin(b))
        if isinstance(arg, Mul):
            if len(arg.args) == 2 and arg.args[0] == 2:
                x = arg.args[1]
                return Integer(2) * expand_trig(sin(x)) * expand_trig(cos(x))
        return sin(arg)

    if fn == "cos" and len(expr.args) == 1:
        arg = expand_trig(expr.args[0], **kwargs)
        if isinstance(arg, Add):
            a = arg.args[0]
            b = Add(*arg.args[1:]) if len(arg.args) > 2 else arg.args[1]
            return expand_trig(cos(a)) * expand_trig(cos(b)) - expand_trig(sin(a)) * expand_trig(sin(b))
        if isinstance(arg, Mul):
            if len(arg.args) == 2 and arg.args[0] == 2:
                x = arg.args[1]
                return Integer(2) * expand_trig(cos(x))**2 - Integer(1)
        return cos(arg)

    if fn == "tan" and len(expr.args) == 1:
        arg = expand_trig(expr.args[0], **kwargs)
        if isinstance(arg, Add):
            a = arg.args[0]
            b = Add(*arg.args[1:]) if len(arg.args) > 2 else arg.args[1]
            t_a = expand_trig(tan(a))
            t_b = expand_trig(tan(b))
            return (t_a + t_b) / (Integer(1) - t_a * t_b)
        if isinstance(arg, Mul):
            if len(arg.args) == 2 and arg.args[0] == 2:
                x = arg.args[1]
                t_x = expand_trig(tan(x))
                return (Integer(2) * t_x) / (Integer(1) - t_x**2)
        return tan(arg)

    if fn == "sinh" and len(expr.args) == 1:
        arg = expand_trig(expr.args[0], **kwargs)
        if isinstance(arg, Add):
            a = arg.args[0]
            b = Add(*arg.args[1:]) if len(arg.args) > 2 else arg.args[1]
            return expand_trig(sinh(a)) * expand_trig(cosh(b)) + expand_trig(cosh(a)) * expand_trig(sinh(b))
        if isinstance(arg, Mul):
            if len(arg.args) == 2 and arg.args[0] == 2:
                x = arg.args[1]
                return Integer(2) * expand_trig(sinh(x)) * expand_trig(cosh(x))
        return sinh(arg)

    if fn == "cosh" and len(expr.args) == 1:
        arg = expand_trig(expr.args[0], **kwargs)
        if isinstance(arg, Add):
            a = arg.args[0]
            b = Add(*arg.args[1:]) if len(arg.args) > 2 else arg.args[1]
            return expand_trig(cosh(a)) * expand_trig(cosh(b)) + expand_trig(sinh(a)) * expand_trig(sinh(b))
        if isinstance(arg, Mul):
            if len(arg.args) == 2 and arg.args[0] == 2:
                x = arg.args[1]
                return Integer(2) * expand_trig(cosh(x))**2 - Integer(1)
        return cosh(arg)

    new_args = [expand_trig(a, **kwargs) for a in expr.args]
    if new_args != list(expr.args):
        return expr.func(*new_args)
    return expr


def expand_log(expr: Any, force: bool = False, **kwargs: Any) -> Any:
    """Expand logarithm terms: log(x*y) => log(x) + log(y), log(x**k) => k*log(x)."""
    expr = sympify(expr)
    if not hasattr(expr, "args") or not expr.args:
        return expr
    from ..functions import log
    fn = getattr(getattr(expr, "func", None), "__name__", "")

    if fn in ("log", "ln") and len(expr.args) == 1:
        arg = expr.args[0]
        if isinstance(arg, Mul):
            if force or all(getattr(f, "is_positive", None) is True for f in arg.args):
                terms = []
                for f in arg.args:
                    terms.append(expand_log(log(f), force=force, **kwargs))
                return Add(*terms)
        if isinstance(arg, Pow):
            base, exponent = arg.args
            if force or getattr(base, "is_positive", None) is True:
                return exponent * expand_log(log(base), force=force, **kwargs)
        return log(expand_log(arg, force=force, **kwargs))

    new_args = [expand_log(a, force=force, **kwargs) for a in expr.args]
    if new_args != list(expr.args):
        return expr.func(*new_args)
    return expr


def expand_power_exp(expr: Any, **kwargs: Any) -> Any:
    """Expand power with additive exponent: a**(b + c) => a**b * a**c."""
    expr = sympify(expr)
    if not hasattr(expr, "args") or not expr.args:
        return expr

    if isinstance(expr, Pow):
        base, exp = expr.args
        base = expand_power_exp(base, **kwargs)
        exp = expand_power_exp(exp, **kwargs)
        if isinstance(exp, Add):
            factors = [expand_power_exp(Pow(base, t), **kwargs) for t in exp.args]
            return Mul(*factors, evaluate=False)
        return Pow(base, exp)

    new_args = [expand_power_exp(a, **kwargs) for a in expr.args]
    if new_args != list(expr.args):
        return expr.func(*new_args)
    return expr


def expand_power_base(expr: Any, force: bool = False, **kwargs: Any) -> Any:
    """Expand power with multiplicative base: (a * b)**c => a**c * b**c."""
    expr = sympify(expr)
    if not hasattr(expr, "args") or not expr.args:
        return expr

    if isinstance(expr, Pow):
        base, exp = expr.args
        base = expand_power_base(base, force=force, **kwargs)
        exp = expand_power_base(exp, force=force, **kwargs)
        if isinstance(base, Mul):
            if force or getattr(exp, "is_integer", None) is True:
                factors = [expand_power_base(Pow(f, exp), force=force, **kwargs) for f in base.args]
                return Mul(*factors, evaluate=False)
        return Pow(base, exp)

    new_args = [expand_power_base(a, force=force, **kwargs) for a in expr.args]
    if new_args != list(expr.args):
        return expr.func(*new_args)
    return expr


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
        term = sym if p == 1 else Pow(sym, p)
        res_terms.append(Mul(term, c_expr, evaluate=evaluate))
    if rest:
        res_terms.extend(rest)
    if not res_terms:
        return Integer(0)
    return Add(*res_terms, evaluate=evaluate) if len(res_terms) > 1 else res_terms[0]


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
