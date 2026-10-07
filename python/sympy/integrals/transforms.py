"""Integral transforms for FrankenSymPy.

Implements Laplace, Fourier, Sine, Cosine, Hankel, and Mellin transforms
and their inverses.
"""

from __future__ import annotations
from typing import Any

from ..core import (
    Basic,
    Expr,
    Function,
    I,
    Integer,
    Rational,
    Symbol,
    _native,
    _native_expr,
    _native_symbol_key,
    _parse_result,
    _require_symbol,
    _wrap,
    Pow,
    diff,
    expand,
    oo,
    pi,
    sympify,
)
from ..functions import cos, exp, sin, sqrt
from ..series import residue


class Transform(Expr):
    """Base class for unevaluated integral transforms."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        obj = object.__new__(cls)
        obj._args = tuple(sympify(a) for a in args)
        return obj

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def args(self) -> tuple[Any, ...]:
        return getattr(self, "_args", ())

    def __repr__(self) -> str:
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.__class__.__name__}({args_str})"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return False
        return self.args == other.args

    def __hash__(self) -> int:
        return hash((self.__class__, self.args))

    def doit(self, **hints: Any) -> Any:
        raise NotImplementedError


class LaplaceTransform(Transform):
    """Unevaluated Laplace transform: L{f(t)}(s)."""

    def __new__(cls, f: Any, t: Any, s: Any, **kwargs: Any) -> Any:
        return super().__new__(cls, f, t, s)

    def doit(self, **hints: Any) -> Any:
        f, t, s = self.args
        hints.setdefault("noconds", True)
        return laplace_transform(f, t, s, **hints)


class InverseLaplaceTransform(Transform):
    """Unevaluated inverse Laplace transform: L^-1{F(s)}(t)."""

    def __new__(cls, F: Any, s: Any, t: Any, plane: Any = None, **kwargs: Any) -> Any:
        return super().__new__(cls, F, s, t)

    def doit(self, **hints: Any) -> Any:
        F, s, t = self.args[:3]
        return inverse_laplace_transform(F, s, t, **hints)


class FourierTransform(Transform):
    """Unevaluated Fourier transform: F{f(x)}(k)."""

    def __new__(cls, f: Any, x: Any, k: Any, **kwargs: Any) -> Any:
        return super().__new__(cls, f, x, k)

    def doit(self, **hints: Any) -> Any:
        f, x, k = self.args
        return fourier_transform(f, x, k, **hints)


class InverseFourierTransform(Transform):
    """Unevaluated inverse Fourier transform: F^-1{F(k)}(x)."""

    def __new__(cls, F: Any, k: Any, x: Any, **kwargs: Any) -> Any:
        return super().__new__(cls, F, k, x)

    def doit(self, **hints: Any) -> Any:
        F, k, x = self.args
        return inverse_fourier_transform(F, k, x, **hints)


class SineTransform(Transform):
    """Unevaluated sine transform."""

    def __new__(cls, f: Any, t: Any, k: Any, **kwargs: Any) -> Any:
        return super().__new__(cls, f, t, k)

    def doit(self, **hints: Any) -> Any:
        f, t, k = self.args
        return sine_transform(f, t, k, **hints)


class InverseSineTransform(Transform):
    """Unevaluated inverse sine transform."""

    def __new__(cls, F: Any, k: Any, t: Any, **kwargs: Any) -> Any:
        return super().__new__(cls, F, k, t)

    def doit(self, **hints: Any) -> Any:
        F, k, t = self.args
        return inverse_sine_transform(F, k, t, **hints)


class CosineTransform(Transform):
    """Unevaluated cosine transform."""

    def __new__(cls, f: Any, t: Any, k: Any, **kwargs: Any) -> Any:
        return super().__new__(cls, f, t, k)

    def doit(self, **hints: Any) -> Any:
        f, t, k = self.args
        return cosine_transform(f, t, k, **hints)


class InverseCosineTransform(Transform):
    """Unevaluated inverse cosine transform."""

    def __new__(cls, F: Any, k: Any, t: Any, **kwargs: Any) -> Any:
        return super().__new__(cls, F, k, t)

    def doit(self, **hints: Any) -> Any:
        F, k, t = self.args
        return inverse_cosine_transform(F, k, t, **hints)


class HankelTransform(Transform):
    """Unevaluated Hankel transform."""

    def __new__(cls, f: Any, r: Any, k: Any, nu: Any, **kwargs: Any) -> Any:
        return super().__new__(cls, f, r, k, nu)

    def doit(self, **hints: Any) -> Any:
        f, r, k, nu = self.args
        return hankel_transform(f, r, k, nu, **hints)


class InverseHankelTransform(Transform):
    """Unevaluated inverse Hankel transform."""

    def __new__(cls, F: Any, k: Any, r: Any, nu: Any, **kwargs: Any) -> Any:
        return super().__new__(cls, F, k, r, nu)

    def doit(self, **hints: Any) -> Any:
        F, k, r, nu = self.args
        return inverse_hankel_transform(F, k, r, nu, **hints)


class MellinTransform(Transform):
    """Unevaluated Mellin transform."""

    def __new__(cls, f: Any, x: Any, s: Any, **kwargs: Any) -> Any:
        return super().__new__(cls, f, x, s)

    def doit(self, **hints: Any) -> Any:
        f, x, s = self.args
        return mellin_transform(f, x, s, **hints)


class InverseMellinTransform(Transform):
    """Unevaluated inverse Mellin transform."""

    def __new__(cls, F: Any, s: Any, x: Any, strip: Any = None, **kwargs: Any) -> Any:
        return super().__new__(cls, F, s, x)

    def doit(self, **hints: Any) -> Any:
        F, s, x = self.args[:3]
        return inverse_mellin_transform(F, s, x, **hints)



def _has_integral(e: Any) -> bool:
    """True when ``e`` still contains an unevaluated ``Integral``."""
    if type(e).__name__ == "Integral":
        return True
    return any(_has_integral(a) for a in getattr(e, "args", ()) or ())

def _laplace_abscissa(F: Any, s: Any) -> Any:
    """Convergence abscissa from the poles of F: the largest real part
    (upstream reports it as ``a`` in ``(F, a, cond)``); -oo when entire."""
    from .. import fraction
    from ..core import S as _S
    from ..functions import re
    from ..functions.elementary.miscellaneous import Max
    from ..polys import roots as _roots

    from ..polys import together

    _, den = fraction(together(F))
    if s not in den.free_symbols:
        return _S.NegativeInfinity
    try:
        rts = list(_roots(den, s))
    except Exception:
        rts = []
    if not rts:
        return Integer(0)
    parts = []
    for r in rts:
        v = re(r)
        if v not in parts:
            parts.append(v)
    try:
        return max(parts, key=float) if all(v.is_number for v in parts) else Max(*parts)
    except Exception:
        return Max(*parts)


def _heaviside_shift(f: Any, t: Any) -> Any:
    """``(c, g)`` with ``f = Heaviside(t - c) * g(t)``, c > 0 constant."""
    from ..core import Mul

    factors = list(f.args) if isinstance(f, Mul) else [f]
    hs = [h for h in factors if type(h).__name__ == "Heaviside"]
    if len(hs) != 1:
        return None
    arg = expand(hs[0].args[0])
    c = expand(t - arg)
    if t in c.free_symbols or c == 0 or expand(arg - (t - c)) != 0:
        return None
    rest = Mul(*[h for h in factors if h is not hs[0]])
    return c, rest


def _laplace_F(expression: Any, t: Any, s: Any) -> Any:
    from ..core import Add
    from ..functions import exp as _exp

    expression = sympify(expression)
    if isinstance(expression, Add):
        return Add(*[_laplace_F(term, t, s) for term in expression.args])
    shifted = _heaviside_shift(expression, t)
    if shifted is not None:
        # L{H(t - c) g(t)} = exp(-c*s) * L{g(t + c)}.
        c, g = shifted
        return _exp(-c * s) * _laplace_F(expand(g.subs(t, t + c)), t, s)
    result = _native.laplace_expr(
        str(_wrap(_native_expr(expression))),
        _native_symbol_key(t),
        _native_symbol_key(s),
    )
    return _parse_result(result)


def laplace_transform(expression: Any, t: Any, s: Any, noconds: bool = False, **kwargs: Any) -> Any:
    """Laplace transform L{f(t)}(s) = int_0^oo f(t) exp(-s*t) dt.

    As upstream, returns ``(F, a, cond)`` with the convergence abscissa
    ``a`` (Re(s) > a) unless ``noconds=True``.
    """
    t_sym = _require_symbol(t)
    s_sym = _require_symbol(s)
    res = _laplace_F(expression, t_sym, s_sym)
    if noconds:
        return res
    return (res, _laplace_abscissa(res, s_sym), True)


def _inverse_rational_term(term: Any, s: Any, t: Any) -> Any:
    """Inverse of one real partial fraction A/(s - r)**m or
    (A*s + B)/(quadratic) via the table; None otherwise."""
    from .. import fraction
    from ..core import sqrt as _sqrt
    from ..functions import exp as _exp, factorial as _fact
    from ..functions import cos as _cos, sin as _sin
    from ..polys.polytools import Poly as _Poly

    num, den = fraction(term)
    if s not in den.free_symbols:
        return None
    try:
        pn = _Poly(num, s)
        base, m = den, 1
        if isinstance(den, Pow) and isinstance(den.args[1], Integer):
            base, m = den.args[0], int(den.args[1])
        # A rational coefficient in the denominator (3*(s + 2)).
        k = Integer(1)
        if type(base).__name__ == "Mul":
            consts = [f for f in base.args if s not in f.free_symbols]
            others = [f for f in base.args if s in f.free_symbols]
            if len(others) == 1:
                k = _mul(consts)
                base = others[0]
                if isinstance(base, Pow) and isinstance(base.args[1], Integer):
                    base, m = base.args[0], m * int(base.args[1])
                    k = k ** 1
        pb = _Poly(base, s)
    except Exception:
        return None
    if pn.degree() < 0:
        return Integer(0)
    db = pb.degree()
    coeffs = pb.all_coeffs()
    lead = coeffs[0]
    if db == 1 and pn.degree() == 0:
        r = -coeffs[1] / lead
        A = num / (k * lead**m)
        return expand(A * t ** (m - 1) / _fact(m - 1) * _exp(r * t))
    if db == 2 and m == 1 and pn.degree() <= 1:
        b2, c2 = coeffs[1] / lead, coeffs[2] / lead
        p = b2 / 2
        w2 = expand(c2 - p**2)
        if not (w2.is_positive is True):
            return None
        w = _sqrt(w2)
        nc = pn.all_coeffs()
        A = nc[0] / (k * lead) if len(nc) == 2 else Integer(0)
        B = (nc[-1]) / (k * lead)
        damp = _exp(-p * t)
        return expand(damp * (A * _cos(w * t) + (B - A * p) / w * _sin(w * t)))
    return None


def _mul(xs: list) -> Any:
    out = Integer(1)
    for x_ in xs:
        out = out * x_
    return out


def _split_exp_shift(F: Any, s: Any) -> tuple:
    """``F = exp(-c*s) * G(s)`` -> (c, G) with c free of s (c may be 0)."""
    from ..core import Mul

    factors = list(F.args) if isinstance(F, Mul) else [F]
    c = Integer(0)
    rest = []
    for f in factors:
        if type(f).__name__ == "exp":
            arg = expand(f.args[0])
            lin = expand(arg / s)
            if s not in lin.free_symbols and expand(arg - lin * s) == 0:
                c = c - lin
                continue
        rest.append(f)
    return c, _mul(rest)


def inverse_laplace_transform(F: Any, s: Any, t: Any, plane: Any = None, noconds: bool = True, **kwargs: Any) -> Any:
    """Inverse Laplace transform (causal): rational F through real partial
    fractions and the transform table, ``exp(-c*s)`` factors as time
    shifts, each term carrying ``Heaviside(t - c)`` as upstream returns."""
    from ..core import Add
    from ..functions.special.delta_functions import Heaviside
    from ..polys import apart

    F = sympify(F)
    s = Symbol(str(s)) if not isinstance(s, Symbol) else s
    t = Symbol(str(t)) if not isinstance(t, Symbol) else t

    total = Integer(0)
    ok = True
    for term in Add.make_args(expand(F)):
        c, G = _split_exp_shift(term, s)
        try:
            pieces = Add.make_args(apart(G, s))
        except Exception:
            pieces = (G,)
        for piece in pieces:
            f = _inverse_rational_term(piece, s, t)
            if f is None:
                ok = False
                break
            if c != 0:
                f = expand(f.subs(t, t - c))
            for part in Add.make_args(f):
                total = total + part * Heaviside(t - c)
        if not ok:
            break
    if ok:
        return total if noconds else (total, Integer(0), True)
    return _inverse_laplace_residue(F, s, t, noconds)


def _inverse_laplace_residue(F: Any, s: Any, t: Any, noconds: bool) -> Any:
    from ..solvers import solve
    from ..simplify import simplify

    num, den = F.as_numer_denom()
    try:
        roots = solve(den, s)
        if roots:
            integrand = F * exp(s * t)
            res_sum = Integer(0)
            for r in roots:
                res_sum = res_sum + residue(integrand, s, r)
            simplified = simplify(res_sum)
            return simplified if noconds else (simplified, Integer(0), True)
    except Exception:
        pass
    return InverseLaplaceTransform(F, s, t) if noconds else (InverseLaplaceTransform(F, s, t), Integer(0), True)


def fourier_transform(expression: Any, x: Any, k: Any, noconds: bool = True, **kwargs: Any) -> Any:
    """Compute the Fourier transform of expression: F{f(x)}(k) = int_-oo^oo f(x) e^(-2*pi*I*x*k) dx."""
    x_sym = _require_symbol(x)
    k_sym = _require_symbol(k)
    result = _native.fourier_expr(
        str(_wrap(_native_expr(expression))),
        _native_symbol_key(x_sym),
        _native_symbol_key(k_sym),
    )
    res = _parse_result(result)
    if noconds:
        return res
    return (res, True)


def inverse_fourier_transform(F: Any, k: Any, x: Any, noconds: bool = True, **kwargs: Any) -> Any:
    """Compute the inverse Fourier transform: F^-1{F(k)}(x)."""
    from .. import integrate
    from ..simplify import simplify

    F = sympify(F)
    k = Symbol(str(k)) if not isinstance(k, Symbol) else k
    x = Symbol(str(x)) if not isinstance(x, Symbol) else x

    try:
        res = integrate(F * exp(2 * pi * I * k * x), (k, -oo, oo))
        if res is not None and not _has_integral(res):
            simplified = simplify(res)
            return simplified if noconds else (simplified, True)
    except Exception:
        pass

    return InverseFourierTransform(F, k, x) if noconds else (InverseFourierTransform(F, k, x), True)


def sine_transform(f: Any, t: Any, k: Any, noconds: bool = True, **kwargs: Any) -> Any:
    """Compute the sine transform: int_0^oo f(t) sin(k*t) dt."""
    from .. import integrate
    from ..simplify import simplify

    f = sympify(f)
    t = Symbol(str(t)) if not isinstance(t, Symbol) else t
    k = Symbol(str(k)) if not isinstance(k, Symbol) else k

    try:
        res = integrate(f * sin(k * t), (t, 0, oo))
        if res is not None and not _has_integral(res):
            return simplify(res) if noconds else (simplify(res), True)
    except Exception:
        pass
    return SineTransform(f, t, k) if noconds else (SineTransform(f, t, k), True)


def inverse_sine_transform(F: Any, k: Any, t: Any, noconds: bool = True, **kwargs: Any) -> Any:
    """Compute the inverse sine transform: 2/pi * int_0^oo F(k) sin(k*t) dk."""
    from .. import integrate
    from ..simplify import simplify

    F = sympify(F)
    k = Symbol(str(k)) if not isinstance(k, Symbol) else k
    t = Symbol(str(t)) if not isinstance(t, Symbol) else t

    try:
        res = integrate(Rational(2, 1) / pi * F * sin(k * t), (k, 0, oo))
        if res is not None and not _has_integral(res):
            return simplify(res) if noconds else (simplify(res), True)
    except Exception:
        pass
    return InverseSineTransform(F, k, t) if noconds else (InverseSineTransform(F, k, t), True)


def cosine_transform(f: Any, t: Any, k: Any, noconds: bool = True, **kwargs: Any) -> Any:
    """Compute the cosine transform: int_0^oo f(t) cos(k*t) dt."""
    from .. import integrate
    from ..simplify import simplify

    f = sympify(f)
    t = Symbol(str(t)) if not isinstance(t, Symbol) else t
    k = Symbol(str(k)) if not isinstance(k, Symbol) else k

    try:
        res = integrate(f * cos(k * t), (t, 0, oo))
        if res is not None and not _has_integral(res):
            return simplify(res) if noconds else (simplify(res), True)
    except Exception:
        pass
    return CosineTransform(f, t, k) if noconds else (CosineTransform(f, t, k), True)


def inverse_cosine_transform(F: Any, k: Any, t: Any, noconds: bool = True, **kwargs: Any) -> Any:
    """Compute the inverse cosine transform: 2/pi * int_0^oo F(k) cos(k*t) dk."""
    from .. import integrate
    from ..simplify import simplify

    F = sympify(F)
    k = Symbol(str(k)) if not isinstance(k, Symbol) else k
    t = Symbol(str(t)) if not isinstance(t, Symbol) else t

    try:
        res = integrate(Rational(2, 1) / pi * F * cos(k * t), (k, 0, oo))
        if res is not None and not _has_integral(res):
            return simplify(res) if noconds else (simplify(res), True)
    except Exception:
        pass
    return InverseCosineTransform(F, k, t) if noconds else (InverseCosineTransform(F, k, t), True)


def hankel_transform(f: Any, r: Any, k: Any, nu: Any, noconds: bool = True, **kwargs: Any) -> Any:
    """Compute the Hankel transform: int_0^oo f(r) J_nu(k*r) r dr."""
    from .. import integrate
    from ..functions.special.bessel import besselj
    from ..simplify import simplify

    f = sympify(f)
    r = Symbol(str(r)) if not isinstance(r, Symbol) else r
    k = Symbol(str(k)) if not isinstance(k, Symbol) else k
    nu = sympify(nu)

    try:
        res = integrate(f * besselj(nu, k * r) * r, (r, 0, oo))
        if res is not None and not _has_integral(res):
            return simplify(res) if noconds else (simplify(res), True)
    except Exception:
        pass
    return HankelTransform(f, r, k, nu) if noconds else (HankelTransform(f, r, k, nu), True)


def inverse_hankel_transform(F: Any, k: Any, r: Any, nu: Any, noconds: bool = True, **kwargs: Any) -> Any:
    """Compute the inverse Hankel transform: int_0^oo F(k) J_nu(k*r) k dk."""
    from .. import integrate
    from ..functions.special.bessel import besselj
    from ..simplify import simplify

    F = sympify(F)
    k = Symbol(str(k)) if not isinstance(k, Symbol) else k
    r = Symbol(str(r)) if not isinstance(r, Symbol) else r
    nu = sympify(nu)

    try:
        res = integrate(F * besselj(nu, k * r) * k, (k, 0, oo))
        if res is not None and not _has_integral(res):
            return simplify(res) if noconds else (simplify(res), True)
    except Exception:
        pass
    return InverseHankelTransform(F, k, r, nu) if noconds else (InverseHankelTransform(F, k, r, nu), True)


def mellin_transform(f: Any, x: Any, s: Any, noconds: bool = True, **kwargs: Any) -> Any:
    """Compute the Mellin transform: int_0^oo f(x) x^(s-1) dx."""
    from .. import integrate
    from ..simplify import simplify

    f = sympify(f)
    x = Symbol(str(x)) if not isinstance(x, Symbol) else x
    s = Symbol(str(s)) if not isinstance(s, Symbol) else s

    try:
        res = integrate(f * (x ** (s - 1)), (x, 0, oo))
        if res is not None and not _has_integral(res):
            return simplify(res) if noconds else (simplify(res), True)
    except Exception:
        pass
    return MellinTransform(f, x, s) if noconds else (MellinTransform(f, x, s), True)


def inverse_mellin_transform(F: Any, s: Any, x: Any, strip: Any = None, noconds: bool = True, **kwargs: Any) -> Any:
    """Compute the inverse Mellin transform: 1/(2*pi*I) * int_c-I*oo^c+I*oo F(s) x^(-s) ds."""
    from ..solvers import solve
    from ..simplify import simplify

    F = sympify(F)
    s = Symbol(str(s)) if not isinstance(s, Symbol) else s
    x = Symbol(str(x)) if not isinstance(x, Symbol) else x

    # Residue theorem: sum of residues of F(s) * x**(-s) at poles
    num, den = F.as_numer_denom()
    try:
        roots = solve(den, s)
        if roots:
            integrand = F * (x ** (-s))
            res_sum = Integer(0)
            for r in roots:
                res_sum = res_sum + residue(integrand, s, r)
            simplified = simplify(res_sum)
            return simplified if noconds else (simplified, True)
    except Exception:
        pass

    return InverseMellinTransform(F, s, x) if noconds else (InverseMellinTransform(F, s, x), True)


__all__ = [
    "CosineTransform",
    "FourierTransform",
    "HankelTransform",
    "InverseCosineTransform",
    "InverseFourierTransform",
    "InverseHankelTransform",
    "InverseLaplaceTransform",
    "InverseMellinTransform",
    "InverseSineTransform",
    "LaplaceTransform",
    "MellinTransform",
    "SineTransform",
    "Transform",
    "cosine_transform",
    "fourier_transform",
    "hankel_transform",
    "inverse_cosine_transform",
    "inverse_fourier_transform",
    "inverse_hankel_transform",
    "inverse_laplace_transform",
    "inverse_mellin_transform",
    "inverse_sine_transform",
    "laplace_transform",
    "mellin_transform",
    "sine_transform",
]
