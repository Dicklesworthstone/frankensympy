"""Periodic (trigonometric / complex exponential) equations for solveset.

Two upstream strategies, both exact:

* inversion (``_invert_real``): ``A*T(a*x + b) + C = 0`` for a single
  ``T`` in sin/cos/tan, or ``exp`` over the complexes, gives the principal
  values ``asin(c), pi - asin(c)`` / ``acos(c), -acos(c)`` / ``atan(c)``
  (``log(c)`` for exp) plus the period;
* ``_solve_trig1``: with every trig argument linear in ``x``, the
  substitution ``x = mu*t`` and ``y = exp(I*t)`` turns the equation into a
  rational equation in ``y``; each root ``r`` gives ``t = arg(r) + 2*pi*n``
  (unit-modulus roots only over the reals).

Representatives are reduced into ``[0, period)`` when they are rational
multiples of pi, equal ones merge, and each becomes
``ImageSet(Lambda(_n, period*_n + rep), Integers)``. Anything outside
these shapes raises ``NotImplementedError`` (no truncated answers).
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from ..core import Add, Dummy, I, Integer, Mul, Rational, Symbol, expand, pi, sympify

_TRIG = ("sin", "cos", "tan", "cot", "sec", "csc")


def _name(e: Any) -> str:
    return type(e).__name__


def _nodes(e: Any):
    yield e
    for a in getattr(e, "args", ()) or ():
        yield from _nodes(a)


def is_periodic_in(e: Any, x: Any) -> bool:
    return any(_name(a) in _TRIG and x in a.free_symbols for a in _nodes(e))


def _pi_fraction(v: Any) -> Fraction | None:
    """``v / pi`` as an exact fraction, if ``v`` is a rational multiple of pi."""
    v = expand(sympify(v))
    if v == 0:
        return Fraction(0)
    q = expand(v / pi)
    if isinstance(q, Rational):
        return Fraction(int(q.p), int(q.q))
    return None


def _normalize(rep: Any, period_over_pi: Fraction) -> Any:
    """Reduce ``rep`` into ``[0, period)`` when it is a rational multiple of pi."""
    f = _pi_fraction(rep)
    if f is None:
        return expand(rep)
    r = f % period_over_pi
    return Rational(r.numerator, r.denominator) * pi


def _image_sets(reps: list, period: Any) -> Any:
    from ..core.lambdify_subs import Lambda
    from ..sets.fancysets import ImageSet, Integers
    from ..sets.sets import _EMPTY_SET, Union

    n = Dummy("n")
    unique: list = []
    for r in reps:
        if not any(expand(r - u) == 0 for u in unique):
            unique.append(r)
    if not unique:
        return _EMPTY_SET

    def order(r: Any) -> tuple:
        f = _pi_fraction(r)
        if f is not None:
            return (0, float(f))
        try:
            return (1, float(r))
        except Exception:
            return (2, str(r))

    unique.sort(key=order)
    sets = [ImageSet(Lambda(n, period * n + r), Integers()) for r in unique]
    return sets[0] if len(sets) == 1 else Union(*sets)


def _linear_parts(u: Any, x: Any) -> tuple | None:
    """``u = a*x + b`` with x-free a != 0, b."""
    from ..core import diff

    a = expand(diff(u, x))
    if a == 0 or x in a.free_symbols:
        return None
    b = expand(u - a * x)
    if x in b.free_symbols:
        return None
    return a, b


def _invert(f: Any, x: Any, real: bool) -> Any:
    """Single-function inversion; None when the shape does not apply."""
    from ..functions import acos, asin, atan, log

    f = expand(f)
    terms = list(f.args) if isinstance(f, Add) else [f]
    dep = [t for t in terms if x in t.free_symbols]
    if len(dep) != 1:
        return None
    rest = Add(*[t for t in terms if x not in t.free_symbols]) if len(dep) != len(terms) else Integer(0)
    term = dep[0]
    factors = list(term.args) if isinstance(term, Mul) else [term]
    inner = [g for g in factors if x in g.free_symbols]
    if len(inner) != 1:
        return None
    coeff = Mul(*[g for g in factors if x not in g.free_symbols])
    g = inner[0]
    name = _name(g)
    if name not in ("sin", "cos", "tan", "exp"):
        return None
    lin = _linear_parts(g.args[0], x)
    if lin is None:
        return None
    a, b = lin
    c = expand(-rest / coeff)
    if real and c.is_real is not True:
        return None
    from ..sets.sets import _EMPTY_SET

    if name == "exp":
        if real:
            return None
        if c == 0:
            return _EMPTY_SET
        reps = [log(c)]
        period = 2 * pi * I
    elif name == "tan":
        reps = [atan(c)]
        period = pi
    else:
        if real and ((c - 1).is_positive is True or (c + 1).is_negative is True):
            return _EMPTY_SET
        if name == "sin":
            s0 = asin(c)
            reps = [s0, pi - s0]
        else:
            s0 = acos(c)
            reps = [s0, -s0]
        period = 2 * pi
    # x = (rep - b)/a with period/a.
    xp = expand(period / a)
    out = []
    if name == "exp":
        return _image_sets([expand((r - b) / a) for r in reps], xp)
    per = _pi_fraction(xp)
    for r in reps:
        val = expand((r - b) / a)
        out.append(_normalize(val, per) if per is not None and per > 0 else val)
    return _image_sets(out, xp if per is None or per > 0 else -xp)


def _solve_trig1(f: Any, x: Any, real: bool) -> Any:
    """Rewrite in y = exp(I*t), x = mu*t, and solve the rational equation."""
    from math import gcd, lcm
    from .. import solve
    from ..core import diff
    from ..functions import im, log, re
    from ..functions.elementary.trigonometric import atan2
    from ..polys import cancel, together

    args = [a.args[0] for a in _nodes(f) if _name(a) in _TRIG and x in a.free_symbols]
    nums, dens = [], []
    for u in args:
        lin = _linear_parts(u, x)
        if lin is None:
            return None
        a, b = lin
        if b != 0 or not isinstance(a, Rational):
            return None
        nums.append(int(a.p))
        dens.append(int(a.q))
    if not nums:
        return None
    mu = Rational(lcm(*dens), gcd(*nums))
    t = Dummy("t")
    y = Dummy("y")
    g = f.subs(x, mu * t)

    def rewrite(e: Any) -> Any:
        n = _name(e)
        if n in _TRIG:
            k = expand(e.args[0] / t)
            if not isinstance(k, Integer):
                raise ValueError("non-integer multiple")
            p = y ** int(k.p)
            s = (p - 1 / p) / (2 * I)
            c = (p + 1 / p) / 2
            return {"sin": s, "cos": c, "tan": s / c, "cot": c / s, "sec": 1 / c, "csc": 1 / s}[n]
        args_ = getattr(e, "args", ()) or ()
        if args_ and isinstance(e, (Add, Mul)) or type(e).__name__ == "Pow":
            return e.func(*[rewrite(a) for a in args_])
        return e

    try:
        h = cancel(together(rewrite(g)))
    except ValueError:
        return None
    if t in h.free_symbols:
        return None
    from .. import fraction

    num, den = fraction(h)
    num, den = expand(num), expand(den)
    # Strip the y**k content (y = exp(I*t) never vanishes).
    from ..polys.polytools import Poly

    try:
        low = min(m[0] for m, _ in Poly(num, y).terms())
    except Exception:
        return None
    if low:
        num = expand(num / y ** low)
    try:
        roots = solve(num, y)
    except Exception:
        return None
    reps = []
    for r in roots:
        if den != 0 and expand(den.subs(y, r)) == 0:
            continue
        ang = _unit_root_angle(num, y, r)
        if ang is None:
            if real:
                # Not provably on the unit circle at a rational multiple
                # of pi: refuse rather than guess membership.
                if _probe_modulus(r) is None or abs(_probe_modulus(r) - 1) > 1e-9:
                    continue
                return None
            ang = -I * log(r)
        val = expand(mu * ang)
        reps.append(val)
    period = expand(2 * pi * mu)
    per = _pi_fraction(period)
    reps = [_normalize(r, per) if per is not None else r for r in reps]
    return _image_sets(reps, period)


def _probe(r: Any) -> complex | None:
    from ..core import _native, _native_expr

    try:
        v = _native.complex_value_expr(str(_native_expr(r)))
    except Exception:
        return None
    return None if v is None else complex(v[0], v[1])


def _probe_modulus(r: Any) -> float | None:
    v = _probe(r)
    return None if v is None else abs(v)


def _unit_root_angle(num: Any, y: Any, r: Any) -> Any:
    """``theta`` with ``exp(I*theta) = r`` as a rational multiple of pi in
    [0, 2*pi), located numerically and verified exactly: ``exp(I*theta)``
    (radical form) must be a root of ``num``."""
    import cmath
    from fractions import Fraction as _F
    from ..functions import cos, sin
    from ..simplify import simplify

    v = _probe(r)
    if v is None or abs(abs(v) - 1) > 1e-9:
        return None
    t = cmath.phase(v) / cmath.pi % 2
    frac = _F(t).limit_denominator(48)
    if abs(float(frac) - t) > 1e-9:
        return None
    theta = Rational(frac.numerator, frac.denominator) * pi
    zeta = cos(theta) + I * sin(theta)
    if expand(num.subs(y, zeta)) == 0 or simplify(expand(num.subs(y, zeta))) == 0:
        return theta
    return None


def solve_periodic(f: Any, x: Any, domain: Any) -> Any:
    """Complete solution set of a periodic equation ``f = 0`` over ``domain``."""
    from ..sets.sets import Reals

    real = domain is not None and (domain == Reals() or isinstance(domain, Reals))
    res = _invert(f, x, real)
    if res is None:
        res = _solve_trig1(f, x, real)
    if res is None:
        raise NotImplementedError(
            "periodic equation %s = 0 is outside the supported trigonometric shapes" % f
        )
    return res


__all__ = ["is_periodic_in", "solve_periodic"]
