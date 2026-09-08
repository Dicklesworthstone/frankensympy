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
    diff,
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


def laplace_transform(expression: Any, t: Any, s: Any, noconds: bool = True, **kwargs: Any) -> Any:
    """Compute the Laplace transform of expression: L{f(t)}(s) = int_0^oo f(t) e^(-st) dt."""
    t_sym = _require_symbol(t)
    s_sym = _require_symbol(s)
    result = _native.laplace_expr(
        str(_wrap(_native_expr(expression))),
        _native_symbol_key(t_sym),
        _native_symbol_key(s_sym),
    )
    res = _parse_result(result)
    if noconds:
        return res
    return (res, Integer(0), True)


def inverse_laplace_transform(F: Any, s: Any, t: Any, plane: Any = None, noconds: bool = True, **kwargs: Any) -> Any:
    """Compute the inverse Laplace transform: L^-1{F(s)}(t) via Bromwich residue theorem."""
    from ..solvers import solve
    from ..polys import apart
    from ..simplify import simplify

    F = sympify(F)
    s = Symbol(str(s)) if not isinstance(s, Symbol) else s
    t = Symbol(str(t)) if not isinstance(t, Symbol) else t

    # Try partial fraction expansion for rational functions
    try:
        F_apart = apart(F, s)
        if F_apart != F:
            if hasattr(F_apart, "args") and F_apart.func.__name__ == "Add":
                terms = [inverse_laplace_transform(term, s, t, noconds=True) for term in F_apart.args]
                res = sum(terms, Integer(0))
                return res if noconds else (res, Integer(0), True)
    except Exception:
        pass

    # Direct residue theorem for meromorphic / rational functions:
    # L^-1{F(s)}(t) = sum_k Res(F(s) * exp(s * t), s_k)
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

    # Fallback to unevaluated transform
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
        if res is not None and not isinstance(res, integrate.__class__):
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
        if res is not None:
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
        if res is not None:
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
        if res is not None:
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
        if res is not None:
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
        if res is not None:
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
        if res is not None:
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
        if res is not None:
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
