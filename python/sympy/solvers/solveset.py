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
``ImageSet(Lambda(_n, period*_n + rep), Integers)``. Unit-circle membership
and denominator exclusions are exact decisions. Unresolved obligations
remain a ``ConditionSet``; unsupported shapes raise ``NotImplementedError``.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from ..core import Add, Dummy, I, Integer, Mul, Pow, Rational, Symbol, expand, pi, sympify

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


def _conditional(f: Any, x: Any, real: bool) -> Any:
    """Retain the original equation when a completeness guard is unknown."""
    from ..core import Eq
    from ..sets import Complexes, ConditionSet, Reals

    return ConditionSet(x, Eq(f, 0, evaluate=False), Reals() if real else Complexes())


def _has_approximate_atoms(e: Any) -> bool:
    # CRootOf's current shell assumptions use binary64 root ordering. They
    # cannot decide exact zero or unit-circle membership for this solver.
    return any(_name(node) in ("Float", "CRootOf") for node in _nodes(e))


def _rectangular_radicals(e: Any) -> Any:
    """Expose principal square roots of Gaussian rationals exactly.

    Quadratic solving can return sqrt(-I), whose real and imaginary parts
    are not handled by the shell's structural decomposition. Choose the
    numerically stable algebraic identity on each sign of the real part;
    every square-root argument used here is provably positive.
    """
    from ..functions.elementary.complexes import _split_real_imag

    if type(e) not in (Add, Mul, Pow):
        return e
    args = [_rectangular_radicals(arg) for arg in e.args]
    e = e.func(*args)
    if type(e) is not Pow or e.args[1] != Rational(1, 2):
        return e
    parts = _split_real_imag(e.args[0])
    if parts is None or not all(isinstance(part, Rational) for part in parts):
        return e
    a, b = parts
    if b == 0:
        return e
    norm = Pow(a * a + b * b, Rational(1, 2))
    if a.p >= 0:
        real = Pow((norm + a) / 2, Rational(1, 2))
        imaginary = b / (2 * real)
    else:
        imaginary = (1 if b.p > 0 else -1) * Pow((norm - a) / 2, Rational(1, 2))
        real = b / (2 * imaginary)
    return real + I * imaginary


def _zero_status(e: Any) -> bool | None:
    """Prove zero/nonzero using exact simplification and structural facts."""
    from ..functions.elementary.complexes import _split_real_imag
    from ..simplify import simplify

    if _has_approximate_atoms(e):
        return None
    e = simplify(expand(_rectangular_radicals(e)))
    if e == 0:
        return True
    status = e.is_zero
    if status is not None:
        return status
    if e.is_irrational is True:
        return False
    parts = _split_real_imag(e)
    if parts is not None:
        parts = [simplify(expand(part)) for part in parts]
        statuses = [False if part.is_irrational is True else part.is_zero for part in parts]
        if any(status is False for status in statuses):
            return False
        if all(status is True for status in statuses):
            return True
    return None


def _unit_circle_parts(r: Any) -> tuple[bool | None, tuple | None]:
    """Exact membership and real/imaginary parts; uncertainty is preserved."""
    from ..functions.elementary.complexes import _split_real_imag
    from ..simplify import simplify

    if _has_approximate_atoms(r):
        return None, None
    parts = _split_real_imag(_rectangular_radicals(r))
    if parts is None:
        return None, None
    a, b = (simplify(expand(part)) for part in parts)
    return _zero_status(a * a + b * b - 1), (a, b)


def _roots_cover_polynomial(polynomial: Any, variable: Any, roots: list) -> bool:
    """Check a complete linear factorization, including multiplicities.

    A solver's list is only a candidate enumeration. Exact synthetic
    division must consume every degree before the list can define a complete
    periodic solution set. Unknown coefficients, unsupported radicals and
    verification growth remain conditional instead of certifying a partial
    list. The bounded check covers degree <= 128 and 16,384 coefficient steps.
    """
    from ..polys.polytools import Poly
    from ..simplify import simplify

    coefficients = Poly(polynomial, variable).all_coeffs()
    if len(coefficients) > 129 or _zero_status(coefficients[0]) is not False:
        return False
    distinct = []
    for root in roots:
        if root not in distinct:
            distinct.append(root)
    steps = 0
    for root in distinct:
        multiplicity = 0
        while len(coefficients) > 1:
            quotient = [coefficients[0]]
            for coefficient in coefficients[1:]:
                steps += 1
                if steps > 16_384:
                    return False
                value = coefficient + root * quotient[-1]
                if any(index >= 2048 for index, _ in enumerate(_nodes(value))):
                    return False
                quotient.append(simplify(expand(value)))
            remainder = quotient.pop()
            if _zero_status(remainder) is not True:
                break
            coefficients = quotient
            multiplicity += 1
        if multiplicity == 0:
            return False
    return len(coefficients) == 1


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
            return (0, f)
        return (1, r.sort_key())

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
    # Dividing by an unknown parameter loses the parameter-zero branch.
    # Real-domain inversion also requires a real affine argument; otherwise
    # the resulting complex ImageSet is not a set of real solutions.
    if _zero_status(coeff) is not False or _zero_status(a) is not False:
        return _conditional(f, x, real)
    if real and (a.is_real is not True or b.is_real is not True):
        return _conditional(f, x, real)
    c = expand(-rest / coeff)
    from ..sets.sets import _EMPTY_SET

    if real and c.is_real is not True:
        return _EMPTY_SET if c.is_real is False else _conditional(f, x, real)
    if name == "exp":
        if real:
            return None
        zero = _zero_status(c)
        if zero is True:
            return _EMPTY_SET
        if zero is None:
            return _conditional(f, x, real)
        reps = [log(c)]
        period = 2 * pi * I
    elif name == "tan":
        # tan(z) omits +I and -I. Its exponential inversion degenerates
        # at these values; atan(+-I) is not a finite representative.
        exceptional = _zero_status(1 + c * c)
        if exceptional is True:
            return _EMPTY_SET
        if exceptional is None:
            return _conditional(f, x, real)
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
        if real and not (
            ((c + 1).is_nonnegative is True and (1 - c).is_nonnegative is True)
            or s0.is_real is True
        ):
            return _conditional(f, x, real)
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
    from ..functions import log
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
    exclusions = []

    def rewrite(e: Any) -> Any:
        n = _name(e)
        if n in _TRIG:
            k = expand(e.args[0] / t)
            if not isinstance(k, Integer):
                raise ValueError("non-integer multiple")
            p = y ** int(k.p)
            s = (p - 1 / p) / (2 * I)
            c = (p + 1 / p) / 2
            # Cancellation in the rational representation must not erase
            # poles of the original trigonometric expression.
            if n in ("tan", "sec"):
                exclusions.append(c)
            elif n in ("cot", "csc"):
                exclusions.append(s)
            return {"sin": s, "cos": c, "tan": s / c, "cot": c / s, "sec": 1 / c, "csc": 1 / s}[n]
        args_ = getattr(e, "args", ()) or ()
        if args_ and isinstance(e, (Add, Mul, Pow)):
            rewritten = [rewrite(a) for a in args_]
            if isinstance(e, Pow) and isinstance(args_[1], Rational) and args_[1].p < 0:
                # Child reciprocals are recorded before their parents.
                exclusions.append(rewritten[0])
            return e.func(*rewritten)
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
    if num == 0:
        return _conditional(f, x, real)
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
    roots = [_rectangular_radicals(root) for root in roots]
    if not _roots_cover_polynomial(num, y, roots):
        return _conditional(f, x, real)
    reps = []
    for r in roots:
        # y = exp(I*t) is never zero. Unknown zero/pole status cannot be
        # interpreted as an excluded root or as an unconditional solution.
        root_zero = _zero_status(r)
        if root_zero is True:
            continue
        poles = []
        for denominator in [*exclusions, den]:
            poles.append(_zero_status(denominator.subs(y, r)))
            if poles[-1] is True:
                break
        if any(pole is True for pole in poles):
            continue
        if root_zero is None or any(pole is None for pole in poles):
            return _conditional(f, x, real)
        parts = None
        if real:
            membership, parts = _unit_circle_parts(r)
            if membership is False:
                continue
            if membership is None:
                return _conditional(f, x, real)
        ang = _unit_root_angle(num, y, r)
        if ang is None:
            if real:
                # The exact norm proof admits non-root-of-unity solutions
                # too. atan2 is an exact angle, not a numerical estimate.
                a, b = parts
                ang = atan2(b, a)
            else:
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
        v = _native.complex_value_native(_native_expr(r))
    except Exception:
        return None
    return None if v is None else complex(v[0], v[1])


def _unit_root_angle(num: Any, y: Any, r: Any) -> Any:
    """``theta`` with ``exp(I*theta) = r`` as a rational multiple of pi in
    [0, 2*pi), located numerically and verified exactly: ``exp(I*theta)``
    (radical form) must equal this root, not merely another root of num.

    The probe only proposes an optional nicer representative. Failure to
    recognize one never decides membership or discards a solution.
    """
    import cmath
    import math
    from fractions import Fraction as _F
    from ..functions import cos, sin

    v = _probe(r)
    if v is None or not (math.isfinite(v.real) and math.isfinite(v.imag)):
        return None
    t = cmath.phase(v) / cmath.pi % 2
    frac = _F(t).limit_denominator(48)
    if abs(float(frac) - t) > 1e-9:
        return None
    theta = Rational(frac.numerator, frac.denominator) * pi
    zeta = cos(theta) + I * sin(theta)
    if _zero_status(r - zeta) is True:
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
