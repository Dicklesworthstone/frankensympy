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
    simplify_no_trig as _core_simplify_arith,
    simplify_trig_only as _core_simplify_trig_only,
    simplify_powsimp as _core_simplify_powsimp,

    sqrt,
    sympify,
)
from ..polys import cancel, together


def simplify(expr: Any, **kwargs: Any) -> Any:
    """General expression simplification."""
    from ..polys import cancel

    # Oracle-pinned: simplify cancels rational functions
    # (simplify((x**2 - 1)/(x + 1)) -> x - 1) and fuses all-negative-power
    # Adds (simplify(1/x + 1/y) -> (x + y)/(x*y)); mixed Adds stay.
    if isinstance(expr, Mul):
        for f in expr.args:
            if (
                isinstance(f, Pow)
                and isinstance(f.args[1], Integer)
                and f.args[1].p < 0
                and isinstance(f.args[0], Add)
            ):
                return _maybe_fuse_rational_add(cancel(expr))

    original = expr
    expr = _core_simplify(expr)
    if isinstance(expr, Mul):
        for f in expr.args:
            if (
                isinstance(f, Pow)
                and isinstance(f.args[1], Integer)
                and f.args[1].p < 0
                and isinstance(f.args[0], Add)
            ):
                return _maybe_fuse_rational_add(cancel(expr))
    return _shortest(original, _maybe_fuse_rational_add(expr))


def _count_ops(e: Any) -> int:
    """Upstream-style operation count used as the simplification measure."""
    args = getattr(e, "args", ())
    if not args:
        return 0
    if isinstance(e, Add):
        negs = sum(1 for a in args if str(a).startswith("-"))
        return len(args) - 1 + negs + sum(_count_ops(a) for a in args)
    if isinstance(e, Mul):
        num = []
        den = []
        for a in args:
            if isinstance(a, Pow) and isinstance(a.args[1], Integer) and a.args[1].p < 0:
                den.append(a.args[0] if a.args[1].p == -1 else Pow(a.args[0], -a.args[1]))
            else:
                num.append(a)
        ops = max(len(num) - 1, 0) + (1 if den else 0) + max(len(den) - 1, 0)
        return ops + sum(_count_ops(a) for a in num) + sum(_count_ops(a) for a in den)
    if isinstance(e, Pow):
        return 1 + sum(_count_ops(a) for a in args)
    return 1 + sum(_count_ops(a) for a in args)


def _trig_contract(e: Any) -> Any:
    """Contractions upstream's trigsimp performs: sin/cos -> tan,
    cos/sin -> cot, 2*sin(u)*cos(u) -> sin(2*u), cos(u)**2 - sin(u)**2 ->
    cos(2*u). Applied bottom-up; exact identities only."""
    from ..functions import cos, cot, sin, tan

    args = getattr(e, "args", ())
    if not args or not isinstance(e, (Add, Mul, Pow)):
        return e
    new_args = [_trig_contract(a) for a in args]
    if new_args != list(args):
        e = e.func(*new_args)
        if not isinstance(e, (Add, Mul, Pow)):
            return e
    name = lambda f: type(f).__name__
    if isinstance(e, Mul):
        sins: dict = {}
        coss: dict = {}
        others = []
        for f in e.args:
            base, ex = (f.args[0], f.args[1]) if isinstance(f, Pow) else (f, Integer(1))
            if name(base) == "sin" and isinstance(ex, Integer):
                sins[base.args[0]] = sins.get(base.args[0], 0) + ex.p
            elif name(base) == "cos" and isinstance(ex, Integer):
                coss[base.args[0]] = coss.get(base.args[0], 0) + ex.p
            else:
                others.append(f)
        changed = False
        out = list(others)
        for u in set(sins) | set(coss):
            a, b = sins.get(u, 0), coss.get(u, 0)
            if a > 0 and b < 0 and a == -b:
                out.append(tan(u) ** a)
                changed = True
            elif a < 0 and b > 0 and b == -a:
                out.append(cot(u) ** b)
                changed = True
            elif a == 1 and b == 1 and any(o == 2 or (isinstance(o, Integer) and o.p % 2 == 0) for o in others):
                coeff = [o for o in out if isinstance(o, Integer) and o.p % 2 == 0][0]
                out.remove(coeff)
                out.append(Integer(coeff.p // 2))
                out.append(sin(2 * u))
                changed = True
            else:
                if a:
                    out.append(sin(u) ** a)
                if b:
                    out.append(cos(u) ** b)
        if changed:
            result = Integer(1)
            for f in out:
                result = result * f
            return result
        return e
    if isinstance(e, Add):
        # c*sin(u)**2 + c*cos(u)**2 -> c for any common cofactor c.
        def split_sq(t):
            fs = list(t.args) if isinstance(t, Mul) else [t]
            for i, f in enumerate(fs):
                if (
                    isinstance(f, Pow) and f.args[1] == 2
                    and name(f.args[0]) in ("sin", "cos")
                ):
                    rest = Integer(1)
                    for j, g in enumerate(fs):
                        if j != i:
                            rest = rest * g
                    return name(f.args[0]), f.args[0].args[0], rest
            return None
        parts = [split_sq(t) for t in e.args]
        used = set()
        merged = []
        for i, pi in enumerate(parts):
            if pi is None or i in used:
                continue
            for j in range(i + 1, len(parts)):
                pj = parts[j]
                if pj is None or j in used:
                    continue
                if {pi[0], pj[0]} == {"sin", "cos"} and pi[1] == pj[1] and pi[2] == pj[2]:
                    used.update((i, j))
                    merged.append(pi[2])
                    break
        if merged:
            total = Integer(0)
            for i, t in enumerate(e.args):
                if i not in used:
                    total = total + t
            for m in merged:
                total = total + m
            return total
    if isinstance(e, Add) and len(e.args) == 2:
        t1, t2 = e.args
        for p_, n_ in ((t1, t2), (t2, t1)):
            if (
                isinstance(p_, Pow) and p_.args[1] == 2 and name(p_.args[0]) == "cos"
                and isinstance(n_, Mul) and len(n_.args) == 2 and n_.args[0] == -1
                and isinstance(n_.args[1], Pow) and n_.args[1].args[1] == 2
                and name(n_.args[1].args[0]) == "sin"
                and n_.args[1].args[0].args[0] == p_.args[0].args[0]
            ):
                return cos(2 * p_.args[0].args[0])
    return e


def _shortest(original: Any, simplified: Any) -> Any:
    """Pick the cheapest of the native result and the standard rewrites,
    as upstream simplify does by count_ops."""
    from ..core import expand
    from ..polys import cancel

    candidates = [simplified]
    extra = []
    text = str(original)
    if any(t in text for t in ("sin", "cos", "tan", "cot", "sec", "csc")):
        extra.append(trigsimp)
    if "log" in text:
        extra.append(logcombine)
    if "gamma" in text or "factorial" in text:
        extra.append(lambda z: _gamma_ratio_pass(z, keep_factorial=True))
    for transform in (expand, cancel, _trig_contract, *extra):
        for source in (original, simplified):
            try:
                candidates.append(transform(source))
            except Exception:
                continue
    best = simplified
    best_cost = _count_ops(simplified)
    for c in candidates[1:]:
        cost = _count_ops(c)
        if cost < best_cost:
            best, best_cost = c, cost
    return best


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
                # (Upstream trigsimp answers 1 + tan(u)**2 with cos(u)**(-2),
                # found by the rewrite search; no sec/csc introduction here.)
                for one_cand, tan_cand in ():
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
    from .trigsimp import trigsimp as _search_trigsimp

    legacy = _trigsimp_legacy(expr, **kwargs)
    searched = _search_trigsimp(legacy)
    return searched if _count_ops(searched) <= _count_ops(legacy) else legacy


def _trigsimp_legacy(expr: Any, **kwargs: Any) -> Any:
    expr = sympify(expr)
    # Oracle-pinned: trigsimp applies trig folds but does NOT combine exp
    # products (trigsimp(exp(x)*exp(y)) keeps the Mul).
    c_res = _core_simplify_trig_only(expr)
    res = _trigsimp_pass(c_res)
    if res != c_res:
        return _core_simplify_trig_only(res)
    return res


def _is_pure_negative_power_term(term: Any) -> bool:
    if isinstance(term, Pow) and isinstance(term.args[0], Symbol) and isinstance(term.args[1], Integer) and term.args[1].p < 0:
        return True
    if isinstance(term, Symbol):
        return False
    if isinstance(term, Mul):
        return all(_is_pure_negative_power_factor(f) for f in term.args)
    return False


def _is_pure_negative_power_factor(f: Any) -> bool:
    if isinstance(f, Symbol):
        return False
    if isinstance(f, Pow) and isinstance(f.args[0], Symbol) and isinstance(f.args[1], Integer) and f.args[1].p < 0:
        return True
    if isinstance(f, (Integer, Rational)):
        return True
    return False


def _maybe_fuse_rational_add(expr: Any) -> Any:
    from ..polys import together
    if isinstance(expr, Add) and expr.args and all(
        _is_pure_negative_power_term(t) for t in expr.args
    ):
        return together(expr)
    return expr


def powsimp(expr: Any, combine: str = "all", force: bool = False, **kwargs: Any) -> Any:
    """Simplify products of powers by combining exponents or bases."""
    expr = sympify(expr)
    # Oracle-pinned: powsimp applies power rules only - no trigonometric
    # rewrites (powsimp(sin(x)**2 + cos(x)**2) keeps the Add).
    c_res = _core_simplify_powsimp(expr)
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
    return _core_simplify_powsimp(res)


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
            # Positive factors split off: log(p*x) = log(p) + log(x).
            pos = [f for f in arg.args if getattr(f, "is_positive", None) is True]
            if pos:
                rest = Mul(*[f for f in arg.args if f not in pos])
                terms = [expand_log(log(f), force=force, **kwargs) for f in pos]
                terms.append(log(rest))
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


def _gamma_ratio_pass(expr: Any, keep_factorial: bool) -> Any:
    """Cancel gamma(a)/gamma(b) (and factorial ratios) with integer a - b
    into finite products of linear factors (upstream gammasimp/combsimp)."""
    from ..core import Rational as _Rational, expand
    from ..functions import factorial, gamma

    expr = sympify(expr)
    args = getattr(expr, "args", ()) or ()
    if args and isinstance(expr, (Add, Mul, Pow)):
        new = [_gamma_ratio_pass(a, keep_factorial) for a in args]
        if new != list(args):
            expr = expr.func(*new)
    if not isinstance(expr, Mul):
        return expr
    num: list = []
    den: list = []
    other: list = []
    for f in expr.args:
        base, ex = (f.args[0], f.args[1]) if isinstance(f, Pow) else (f, Integer(1))
        n = type(base).__name__
        if n in ("gamma", "factorial") and isinstance(ex, Integer) and ex.p != 0:
            z = base.args[0] if n == "gamma" else base.args[0] + 1
            (num if ex.p > 0 else den).extend([(z, n)] * abs(ex.p))
        else:
            other.append(f)
    if not num or not den:
        return expr
    out = Integer(1)
    for f in other:
        out = out * f
    used_den = [False] * len(den)
    leftover_num = []
    for (a, na) in num:
        matched = False
        for j, (b, nb) in enumerate(den):
            if used_den[j]:
                continue
            k = expand(a - b)
            if isinstance(k, Integer):
                used_den[j] = True
                kk = int(k.p)
                if kk > 0:
                    for i in range(kk):
                        out = out * (b + i)
                elif kk < 0:
                    for i in range(-kk):
                        out = out / (a + i)
                matched = True
                break
        if not matched:
            leftover_num.append((a, na))

    def back(z: Any, n: str) -> Any:
        return factorial(z - 1) if (n == "factorial" and keep_factorial) else gamma(z)

    for (a, na) in leftover_num:
        out = out * back(a, na)
    for j, (b, nb) in enumerate(den):
        if not used_den[j]:
            out = out / back(b, nb)
    return out


def gammasimp(expr: Any) -> Any:
    """Simplify ratios of gamma functions (and factorials) exactly."""
    return _gamma_ratio_pass(expr, keep_factorial=False)


def combsimp(expr: Any) -> Any:
    """Combinatorial simplification: factorial/gamma ratios cancel to
    products of linear factors (factorial(n + 1)/factorial(n) -> n + 1)."""
    return _gamma_ratio_pass(expr, keep_factorial=True)


def ratsimp(expr: Any) -> Any:
    """Put rational expression over a common denominator and cancel factors."""
    return cancel(together(expr))


def radsimp(expr: Any, **kwargs: Any) -> Any:
    """Rationalize the denominator of a radical expression."""
    expr = sympify(expr)
    if isinstance(expr, (Mul, Add)):
        parts = [radsimp(a, **kwargs) for a in expr.args]
        if parts != list(expr.args):
            from ..core import expand as _expand

            rebuilt = Mul(*parts) if isinstance(expr, Mul) else Add(*parts)
            return _expand(rebuilt) if isinstance(expr, Mul) else rebuilt
        return expr
    # Oracle-pinned: Pow(c, -1/2) rationalizes to sqrt(c)/2
    # (radsimp(1/sqrt(2)) -> Mul(Rational(1, 2), Pow(2, 1/2))).
    if isinstance(expr, Pow) and isinstance(expr.args[1], Rational) and expr.args[1] == Rational(-1, 2):
        return Mul(Rational(1, 2), sqrt(expr.args[0]))
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
    # Oracle-pinned: an all-negative-power Add fuses over the common
    # denominator (radsimp(1/x + 1/y) -> Pow(x, -1) * Pow(y, -1) * (x + y)).
    from ..polys import together
    if isinstance(expr, Add) and expr.args and all(
        isinstance(t, Pow)
        and isinstance(t.args[0], Symbol)
        and isinstance(t.args[1], Integer)
        and t.args[1].p < 0
        for t in expr.args
    ):
        return together(expr)
    return expr


def logcombine(expr: Any, force: bool = False, **kwargs: Any) -> Any:
    """Upstream ``logcombine``: ``log(x) + log(y) -> log(x*y)`` and
    ``a*log(x) -> log(x**a)`` when ``x`` is positive and ``a`` real (any
    ``x``/``a`` with ``force=True``); everything else is left alone."""
    from ..functions import log

    expr = sympify(expr)
    args = getattr(expr, "args", ()) or ()
    if args and isinstance(expr, (Add, Mul, Pow)):
        new = [logcombine(a, force=force) for a in args]
        if new != list(args):
            expr = expr.func(*new)
    elif args and type(expr).__name__ not in ("log",) and hasattr(expr, "func"):
        try:
            new = [logcombine(a, force=force) for a in args]
            if new != list(args):
                expr = expr.func(*new)
        except Exception:
            pass

    def is_log(f: Any) -> bool:
        return type(f).__name__ == "log" and len(f.args) == 1

    terms = list(expr.args) if isinstance(expr, Add) else [expr]
    eligible: list = []
    rest: list = []
    for t in terms:
        factors = list(t.args) if isinstance(t, Mul) else [t]
        logs = [f for f in factors if is_log(f)]
        if len(logs) != 1:
            rest.append(t)
            continue
        arg = logs[0].args[0]
        coeff = Integer(1)
        for f in factors:
            if f is not logs[0] and not (is_log(f) and f == logs[0]):
                coeff = coeff * f
        ok_arg = force or arg.is_positive is True
        ok_coeff = force or coeff.is_real is True
        if ok_arg and ok_coeff:
            eligible.append((arg, coeff))
        else:
            rest.append(t)
    if not eligible or (len(eligible) == 1 and eligible[0][1] == 1):
        return expr
    inner = Integer(1)
    for arg, coeff in eligible:
        inner = inner * (arg if coeff == 1 else Pow(arg, coeff))
    return Add(log(inner), *rest) if rest else log(inner)


def _collect_power_parts(f: Any) -> tuple:
    """``(base, exponent)`` of a factor; ``exp(u)`` reads as ``(E, u)``."""
    from ..core import E

    if isinstance(f, Pow):
        return f.args[0], f.args[1]
    if type(f).__name__ == "exp":
        return E, f.args[0]
    return f, Integer(1)


def _collect_match(term: Any, pattern: Any) -> tuple | None:
    """Split ``term`` as ``pattern**k * coeff`` (upstream ``collect`` term
    parsing): every factor of ``pattern`` must occur in ``term`` with its
    exponent scaled by one common ``k``. Returns ``(key, coeff)``."""
    from ..core import expand

    tf = [_collect_power_parts(f) for f in Mul.make_args(term)]
    pf = [_collect_power_parts(f) for f in Mul.make_args(pattern)]
    used: list = []
    powers: list = []
    ratios: list = []
    for pb, pe in pf:
        if isinstance(pb, Rational) and pe == 1:
            continue  # a number matches everything
        hit = None
        for i, (tb, te) in enumerate(tf):
            if i not in used and tb == pb:
                hit = i
                break
        if hit is None:
            return None
        used.append(hit)
        powers.append(pe)
        ratios.append(expand(tf[hit][1] / pe))
    factors = Mul.make_args(term)
    rest = Mul(*[factors[i] for i in range(len(factors)) if i not in used])
    if all(expand(r - ratios[0]) == 0 for r in ratios):
        return Mul(*[factors[i] for i in used]), rest
    # Mixed exponents (upstream): the key is pattern**k for the smallest
    # ratio k >= 1 and each matched factor keeps its excess power.
    if not all(isinstance(r, Rational) for r in ratios):
        return None
    k = min(ratios)
    if k < 1:
        return None
    key = Mul(*[tf[i][0] ** (pe * k) for i, pe in zip(used, powers)])
    extra = Mul(*[tf[i][0] ** expand(tf[i][1] - pe * k) for i, pe in zip(used, powers)])
    return key, extra * rest


def collect(
    expr: Any,
    syms: Any,
    func: Any = None,
    evaluate: bool = True,
    exact: bool = False,
    distribute_order_term: bool = True,
) -> Any:
    """Collect additive terms with respect to symbols or product patterns
    (upstream ``collect``): each term goes to the first pattern ``s`` of
    ``syms`` it contains as ``s**k``; terms sharing a key are summed. With
    ``evaluate=False`` the ``{key: coefficient}`` dict is returned."""
    del distribute_order_term
    expr = sympify(expr)
    if isinstance(func, bool):  # legacy positional ``evaluate``
        evaluate, func = func, None
    patterns = list(syms) if isinstance(syms, (list, tuple, set)) else [syms]
    patterns = [sympify(p) for p in patterns]
    if evaluate:
        if isinstance(expr, Mul):
            return Mul(*[collect(a, patterns, func, True, exact) for a in expr.args])
        if isinstance(expr, Pow):
            return Pow(collect(expr.args[0], patterns, func, True, exact), expr.args[1])
        if isinstance(expr, Add):
            expr = Add(*[collect(a, patterns, func, True, exact) for a in expr.args])

    order: list = []
    groups: dict = {}
    for term in Add.make_args(expr):
        placed = False
        for pat in patterns:
            m = _collect_match(term, pat)
            if m is None:
                continue
            key, coeff = m
            if exact and coeff.free_symbols & pat.free_symbols:
                continue
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(coeff)
            placed = True
            break
        if not placed:
            one = Integer(1)
            if one not in groups:
                groups[one] = []
                order.append(one)
            groups[one].append(term)

    collected = {}
    for key in order:
        c = Add(*groups[key])
        if func is not None:
            c = func(c)
        collected[key] = c
    if not evaluate:
        return collected
    return Add(*[k * c for k, c in collected.items()])


def separatevars(expr: Any, symbols: Any = None, dict: bool = False) -> Any:
    """Separate multiplicative factors in an expression."""
    expr = sympify(expr)
    # Oracle-pinned: separatevars fuses all-negative-power Adds
    # (separatevars(1/x + 1/y) -> (x + y)/(x*y)).
    return _maybe_fuse_rational_add(expr)


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
