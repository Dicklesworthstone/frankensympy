"""Trigonometric and hyperbolic simplification.

A cost-driven rewrite search in the spirit of upstream ``futrig``: each
subexpression (bottom-up) is offered a small set of exact rewrites and the
cheapest result by operation count is kept.

Rewrites (all exact identities):

* reciprocal functions to ``sin``/``cos`` (``tan = sin/cos``, ...), with
  rational cancellation;
* Pythagorean collapse ``c*sin(u)**2 + c*cos(u)**2 -> c`` and the
  hyperbolic ``c*cosh(u)**2 - c*sinh(u)**2 -> c``;
* angle-addition contraction of two-term sums
  (``sin(a)cos(b) + cos(a)sin(b) -> sin(a + b)``,
  ``cos(a)cos(b) - sin(a)sin(b) -> cos(a + b)`` and the hyperbolic forms),
  which includes the double-angle cases ``2*sin(u)*cos(u) -> sin(2*u)``,
  ``cos(u)**2 - sin(u)**2 -> cos(2*u)``, ``cosh(u)**2 + sinh(u)**2 ->
  cosh(2*u)``;
* factor-then-contract (``sin**4 - cos**4 -> -cos(2*x)``);
* quotient contraction ``sin/cos -> tan``, ``cos/sin -> cot``.
"""

from __future__ import annotations

from typing import Any

from ..core import Add, Integer, Mul, Pow, Rational, sympify

_TRIG = ("sin", "cos", "tan", "cot", "sec", "csc")
_HYP = ("sinh", "cosh", "tanh", "coth", "sech", "csch")


def _name(e: Any) -> str:
    return type(e).__name__


def _has_trig(e: Any) -> bool:
    if _name(e) in _TRIG + _HYP:
        return True
    return any(_has_trig(a) for a in getattr(e, "args", ()) or ())


def _trig_count(e: Any) -> int:
    """Upstream fu ``L``: number of trigonometric (not hyperbolic) nodes."""
    own = 1 if _name(e) in _TRIG else 0
    return own + sum(_trig_count(a) for a in getattr(e, "args", ()) or ())


def _ops(e: Any) -> tuple:
    from .simplify import _count_ops

    return (_trig_count(e), _count_ops(e), len(str(e)))


def _product_to_sum(e: Any) -> Any:
    """Upstream TR8: products and powers of sin/cos become sums of single
    sin/cos terms (sin(a)*cos(b) = (sin(a+b) + sin(a-b))/2, ...)."""
    from ..core import expand
    from ..functions import cos, sin

    def term_to_sum(t: Any) -> Any:
        rest, trig = _split_term(t)
        units = []
        for (n, u), k in trig.items():
            if n in ("sin", "cos"):
                units.extend([(n, u)] * k)
            else:
                rest = rest * _rebuild(Integer(1), {(n, u): k})
        if len(units) < 2:
            return t
        acc = [(Integer(1), units[0])]
        for (n2, v) in units[1:]:
            new = []
            for (c, (n1, u)) in acc:
                if n1 == "sin" and n2 == "sin":
                    new += [(c / 2, ("cos", u - v)), (-c / 2, ("cos", u + v))]
                elif n1 == "cos" and n2 == "cos":
                    new += [(c / 2, ("cos", u - v)), (c / 2, ("cos", u + v))]
                elif n1 == "sin":
                    new += [(c / 2, ("sin", u + v)), (c / 2, ("sin", u - v))]
                else:
                    new += [(c / 2, ("sin", u + v)), (c / 2, ("sin", v - u))]
            acc = new
        total = Integer(0)
        for c, (n, u) in acc:
            total = total + c * (sin(u) if n == "sin" else cos(u))
        return rest * total

    ex = expand(e)
    terms = list(ex.args) if isinstance(ex, Add) else [ex]
    return expand(Add(*[term_to_sum(t) for t in terms]))


def _to_sin_cos(e: Any) -> Any:
    """Rewrite reciprocal/quotient trig and hyperbolic functions."""
    from ..functions import cos, cosh, sin, sinh

    args = getattr(e, "args", ()) or ()
    if not args or not hasattr(e, "func"):
        return e
    n = _name(e)
    if n in ("tan", "cot", "sec", "csc", "tanh", "coth", "sech", "csch"):
        u = _to_sin_cos(args[0])
        if n in ("tan", "cot", "sec", "csc"):
            s, c = sin(u), cos(u)
        else:
            s, c = sinh(u), cosh(u)
        return {
            "tan": s / c, "cot": c / s, "sec": 1 / c, "csc": 1 / s,
            "tanh": s / c, "coth": c / s, "sech": 1 / c, "csch": 1 / s,
        }[n]
    if isinstance(e, (Add, Mul, Pow)) or n in _TRIG + _HYP:
        new = [_to_sin_cos(a) for a in args]
        if new != list(args):
            return e.func(*new)
    return e


def _split_term(t: Any) -> tuple[Any, dict]:
    """``t = coeff * prod(f(u)**k)`` over sin/cos/sinh/cosh factors:
    returns (rest, {(name, u): k})."""
    factors = list(t.args) if isinstance(t, Mul) else [t]
    rest = Integer(1)
    trig: dict = {}
    for f in factors:
        base, ex = (f.args[0], f.args[1]) if isinstance(f, Pow) else (f, Integer(1))
        bn = _name(base)
        if bn in ("sin", "cos", "sinh", "cosh") and isinstance(ex, Integer) and ex.p > 0:
            key = (bn, base.args[0])
            trig[key] = trig.get(key, 0) + int(ex.p)
        else:
            rest = rest * f
    return rest, trig


def _rebuild(rest: Any, trig: dict) -> Any:
    from ..functions import cos, cosh, sin, sinh

    fn = {"sin": sin, "cos": cos, "sinh": sinh, "cosh": cosh}
    out = rest
    for (n, u), k in trig.items():
        if k:
            out = out * fn[n](u) ** k
    return out


def _pair_contract(t1: Any, t2: Any) -> Any:
    """Contract ``t1 + t2`` by a two-term identity, or return None."""
    from ..functions import cos, cosh, sin, sinh

    r1, f1 = _split_term(t1)
    r2, f2 = _split_term(t2)
    # Pythagorean collapse: c*s(u)^2 + c*c(u)^2 -> c (and hyperbolic
    # c*cosh^2 - c*sinh^2 -> c).
    for (sq_a, sq_b, sign) in (("sin", "cos", 1), ("cosh", "sinh", -1)):
        for (ra, fa, rb, fb) in ((r1, f1, r2, f2), (r2, f2, r1, f1)):
            ka = [k for k in fa if k[0] == sq_a and fa[k] >= 2]
            for key in ka:
                u = key[1]
                other = (sq_b, u)
                if fb.get(other, 0) < 2:
                    continue
                fa2 = dict(fa)
                fa2[key] -= 2
                fb2 = dict(fb)
                fb2[other] -= 2
                ca = _rebuild(ra, fa2)
                cb = _rebuild(rb, fb2)
                if (ca - sign * cb) == 0 or _expand_zero(ca - sign * cb):
                    return ca
    # Angle addition over products of two first powers.
    def linear_pairs(f: dict) -> list:
        """(name1, u1, name2, u2, leftover) choices of two unit factors."""
        units = []
        for key, k in f.items():
            units.extend([key] * k)
        out = []
        for i in range(len(units)):
            for j in range(i + 1, len(units)):
                left = dict(f)
                left[units[i]] -= 1
                left[units[j]] -= 1
                out.append((units[i], units[j], left))
        return out

    for (p, q, left1) in linear_pairs(f1):
        for (r, s, left2) in linear_pairs(f2):
            c1 = _rebuild(r1, left1)
            c2 = _rebuild(r2, left2)
            res = _angle_rule(p, q, c1, r, s, c2)
            if res is not None:
                return res
    return None


def _expand_zero(e: Any) -> bool:
    from ..core import expand

    try:
        return expand(e) == 0
    except Exception:
        return False


def _angle_rule(p: tuple, q: tuple, c1: Any, r: tuple, s: tuple, c2: Any) -> Any:
    """Match ``c1*p*q + c2*r*s`` against the addition formulas
    sin(A)cos(B) +- cos(A)sin(B) = sin(A +- B),
    cos(A)cos(B) -+ sin(A)sin(B) = cos(A +- B) and the hyperbolic forms
    (cosh(A)cosh(B) +- sinh(A)sinh(B) = cosh(A +- B))."""
    from ..functions import cos, cosh, sin, sinh

    def same(a: Any, b: Any) -> bool:
        return a == b or _expand_zero(a - b)

    def opposite(a: Any, b: Any) -> bool:
        return _expand_zero(a + b)

    def mixed(x: tuple, y: tuple, sn: str, cn: str) -> tuple | None:
        if x[0] == sn and y[0] == cn:
            return (x[1], y[1])
        if y[0] == sn and x[0] == cn:
            return (y[1], x[1])
        return None

    for (sn, cn, sfn, cfn, hyp) in (("sin", "cos", sin, cos, False), ("sinh", "cosh", sinh, cosh, True)):
        m1 = mixed(p, q, sn, cn)
        m2 = mixed(r, s, sn, cn)
        if m1 is not None and m2 is not None:
            (x1, y1), (x2, y2) = m1, m2
            if x2 == y1 and y2 == x1:
                if same(c1, c2):
                    return c1 * sfn(x1 + y1)
                if opposite(c1, c2):
                    return c1 * sfn(x1 - y1)
        for (pp, qq, cp, rr, ss, cr) in ((p, q, c1, r, s, c2), (r, s, c2, p, q, c1)):
            if pp[0] == qq[0] == cn and rr[0] == ss[0] == sn:
                if sorted([str(pp[1]), str(qq[1])]) != sorted([str(rr[1]), str(ss[1])]):
                    continue
                a_, b_ = pp[1], qq[1]
                plus = same(cr, cp) if hyp else opposite(cr, cp)
                minus = opposite(cr, cp) if hyp else same(cr, cp)
                if plus:
                    return cp * cfn(a_ + b_)
                if minus:
                    return cp * cfn(a_ - b_)
    return None


def _contract_sum(e: Any) -> Any:
    """Repeatedly contract pairs of terms of an Add."""
    terms = list(e.args)
    changed = True
    while changed and len(terms) > 1:
        changed = False
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                r = _pair_contract(terms[i], terms[j])
                if r is not None:
                    terms = [t for k, t in enumerate(terms) if k not in (i, j)] + [r]
                    changed = True
                    break
            if changed:
                break
    return Add(*terms) if len(terms) != 1 else terms[0]


def _double_angle_product(e: Any) -> Any:
    """``c*sin(u)*cos(u) -> (c/2)*sin(2u)`` (and sinh/cosh) inside a Mul."""
    from ..functions import sin, sinh

    if not isinstance(e, Mul):
        return e
    rest, trig = _split_term(e)
    for (sn, cn, fn) in (("sin", "cos", sin), ("sinh", "cosh", sinh)):
        for key in list(trig):
            if key[0] != sn or trig.get(key, 0) < 1:
                continue
            other = (cn, key[1])
            if trig.get(other, 0) < 1:
                continue
            coeff = rest
            t2 = dict(trig)
            t2[key] -= 1
            t2[other] -= 1
            return _rebuild(coeff * Rational(1, 2) * fn(2 * key[1]), t2)
    return e


def _quotient_contract(e: Any) -> Any:
    """``sin(u)**k/cos(u)**k -> tan(u)**k`` and ``cos/sin -> cot``."""
    from ..functions import cot, tan, tanh, coth

    if not isinstance(e, Mul):
        return e
    counts: dict = {}
    others = []
    for f in e.args:
        base, ex = (f.args[0], f.args[1]) if isinstance(f, Pow) else (f, Integer(1))
        if _name(base) in ("sin", "cos", "sinh", "cosh") and isinstance(ex, Integer):
            key = (_name(base), base.args[0])
            counts[key] = counts.get(key, 0) + int(ex.p)
        else:
            others.append(f)
    out = Integer(1)
    for f in others:
        out = out * f
    done = set()
    changed = False
    for (n, u), k in list(counts.items()):
        if (n, u) in done:
            continue
        pair = {"sin": "cos", "cos": "sin", "sinh": "cosh", "cosh": "sinh"}[n]
        m = counts.get((pair, u), 0)
        if k > 0 and m < 0 and n in ("sin", "sinh"):
            j = min(k, -m)
            out = out * (tan(u) if n == "sin" else tanh(u)) ** j
            counts[(n, u)] -= j
            counts[(pair, u)] += j
            changed = True
        elif k > 0 and m < 0 and n in ("cos", "cosh"):
            j = min(k, -m)
            out = out * (cot(u) if n == "cos" else coth(u)) ** j
            counts[(n, u)] -= j
            counts[(pair, u)] += j
            changed = True
        if k > 0 and m < 0:
            done.update({(n, u), (pair, u)})
    if not changed:
        return e
    return _rebuild(out, {k: v for k, v in counts.items()}) if all(
        v >= 0 for v in counts.values()) else e


def _pythagorean(e: Any, keep: str) -> Any:
    """Replace even powers of the partner function by the Pythagorean
    identity: keep='sin' maps cos(u)**(2k) -> (1 - sin(u)**2)**k."""
    from ..functions import cos, sin

    args = getattr(e, "args", ()) or ()
    if isinstance(e, Pow) and isinstance(e.args[1], Integer) and e.args[1].p % 2 == 0:
        base = e.args[0]
        k = e.args[1].p // 2
        if keep == "sin" and _name(base) == "cos":
            return (1 - sin(base.args[0]) ** 2) ** k
        if keep == "cos" and _name(base) == "sin":
            return (1 - cos(base.args[0]) ** 2) ** k
    if args and isinstance(e, (Add, Mul, Pow)):
        new = [_pythagorean(a, keep) for a in args]
        if new != list(args):
            return e.func(*new)
    return e


def _candidates(e: Any) -> list:
    from ..core import expand
    from ..polys import cancel, factor
    from .simplify import expand_trig

    out = [e]
    sc = _to_sin_cos(e)
    for keep in ("sin", "cos"):
        try:
            out.append(cancel(_pythagorean(sc, keep)))
        except Exception:
            pass
    try:
        et = cancel(expand_trig(sc))
        out.append(et)
        for keep in ("sin", "cos"):
            out.append(cancel(_pythagorean(et, keep)))
    except Exception:
        pass
    try:
        if _trig_count(sc) > 1:
            out.append(_product_to_sum(sc))
    except Exception:
        pass
    for base in {e, sc}:
        try:
            c = cancel(base) if base is sc else base
        except Exception:
            c = base
        out.append(c)
        for f in (lambda z: z, expand):
            try:
                g = f(c)
            except Exception:
                continue
            if isinstance(g, Add):
                out.append(_contract_sum(g))
            out.append(_double_angle_product(g))
            out.append(_quotient_contract(g))
        try:
            fac = factor(c)
        except Exception:
            fac = None
        if fac is not None and fac != c:
            out.append(_contract_factors(fac))
    return out


def _contract_factors(e: Any) -> Any:
    """Contract each factor of a product (after factoring)."""
    if isinstance(e, Mul):
        res = Integer(1)
        for f in e.args:
            res = res * _contract_factors(f)
        if res != e:
            return _contract_factors_once(res)
        return res
    if isinstance(e, Pow) and isinstance(e.args[1], Integer):
        return _contract_factors(e.args[0]) ** e.args[1]
    if isinstance(e, Add):
        return _contract_sum(e)
    return e


def _contract_factors_once(e: Any) -> Any:
    from ..core import expand

    try:
        ex = expand(e)
    except Exception:
        return e
    if isinstance(ex, Add):
        c = _contract_sum(ex)
        if _ops(c) < _ops(e):
            return c
    return e


def _best(e: Any) -> Any:
    best = e
    best_cost = _ops(e)
    for c in _candidates(e):
        try:
            cost = _ops(c)
        except Exception:
            continue
        if cost < best_cost:
            best, best_cost = c, cost
    return best


def _bottom_up(e: Any) -> Any:
    args = getattr(e, "args", ()) or ()
    if args and isinstance(e, (Add, Mul, Pow)):
        new = [_bottom_up(a) for a in args]
        if new != list(args):
            e = e.func(*new)
    if not _has_trig(e):
        return e
    return _best(e)


def trigsimp(expr: Any, **kwargs: Any) -> Any:
    """Trigonometric and hyperbolic simplification by exact rewrites."""
    expr = sympify(expr)
    if not _has_trig(expr):
        return expr
    current = expr
    for _ in range(3):
        nxt = _bottom_up(current)
        if nxt == current:
            break
        current = nxt
    return current


__all__ = ["trigsimp"]
