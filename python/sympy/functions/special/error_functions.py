"""Error and related special functions for FrankenSymPy."""

from __future__ import annotations
from typing import Any

from ...core import (
    Basic,
    Expr,
    Function,
    Integer,
    _native,
    _native_expr,
    _wrap,
    oo,
)


class erf(Function):
    """The Gauss Error Function: erf(z) = (2/√π) ∫₀^z e^(-t²) dt."""

    @classmethod
    def eval(cls, z: Any) -> Any:
        if z == 0 or (isinstance(z, (int, Integer)) and int(z) == 0):
            return Integer(0)
        if z == oo:
            return Integer(1)
        if z == -oo:
            return Integer(-1)
        try:
            native_res = _native.py_erf(_native_expr(z))
            wrapped = _wrap(native_res)
            if not (isinstance(wrapped, erf) or getattr(wrapped, "func_name", "") == "erf"):
                return wrapped
        except Exception:
            pass
        return None


class erfc(Function):
    """The Complementary Error Function: erfc(z) = 1 - erf(z)."""

    @classmethod
    def eval(cls, z: Any) -> Any:
        if z == 0 or (isinstance(z, (int, Integer)) and int(z) == 0):
            return Integer(1)
        if z == oo:
            return Integer(0)
        if z == -oo:
            return Integer(2)
        try:
            native_res = _native.py_erfc(_native_expr(z))
            wrapped = _wrap(native_res)
            if not (isinstance(wrapped, erfc) or getattr(wrapped, "func_name", "") == "erfc"):
                return wrapped
        except Exception:
            pass
        return None


class erfi(Function):
    """The Imaginary Error Function: erfi(z) = -i erf(iz)."""

    @classmethod
    def eval(cls, z: Any) -> Any:
        if z == 0 or (isinstance(z, (int, Integer)) and int(z) == 0):
            return Integer(0)
        return None


class erfinv(Function):
    """The Inverse Error Function: erf(erfinv(z)) = z."""

    @classmethod
    def eval(cls, z: Any) -> Any:
        if z == 0 or (isinstance(z, (int, Integer)) and int(z) == 0):
            return Integer(0)
        return None


class erfcinv(Function):
    """The Inverse Complementary Error Function: erfc(erfcinv(z)) = z."""

    @classmethod
    def eval(cls, z: Any) -> Any:
        if z == 1 or (isinstance(z, (int, Integer)) and int(z) == 1):
            return Integer(0)
        return None


class FresnelS(Function):
    """The Fresnel Sine Integral: S(z) = ∫₀^z sin(π t² / 2) dt."""

    @classmethod
    def eval(cls, z: Any) -> Any:
        if z == 0 or (isinstance(z, (int, Integer)) and int(z) == 0):
            return Integer(0)
        return None


class FresnelC(Function):
    """The Fresnel Cosine Integral: C(z) = ∫₀^z cos(π t² / 2) dt."""

    @classmethod
    def eval(cls, z: Any) -> Any:
        if z == 0 or (isinstance(z, (int, Integer)) and int(z) == 0):
            return Integer(0)
        return None


class Ei(Function):
    """The Exponential Integral: Ei(z) = -∫_{-z}^∞ e^(-t) / t dt."""
    pass


class Si(Function):
    """The Sine Integral: Si(z) = ∫₀^z sin(t) / t dt."""

    @classmethod
    def eval(cls, z: Any) -> Any:
        if z == 0 or (isinstance(z, (int, Integer)) and int(z) == 0):
            return Integer(0)
        return None


class Ci(Function):
    """The Cosine Integral: Ci(z) = -∫_z^∞ cos(t) / t dt."""
    pass


class Shi(Function):
    """The Hyperbolic Sine Integral: Shi(z) = ∫₀^z sinh(t) / t dt."""

    @classmethod
    def eval(cls, z: Any) -> Any:
        if z == 0 or (isinstance(z, (int, Integer)) and int(z) == 0):
            return Integer(0)
        return None


class Chi(Function):
    """The Hyperbolic Cosine Integral: Chi(z) = γ + ln(z) + ∫₀^z (cosh(t) - 1) / t dt."""
    pass


__all__ = [
    "Chi",
    "Ci",
    "Ei",
    "FresnelC",
    "FresnelS",
    "Shi",
    "Si",
    "erf",
    "erfc",
    "erfcinv",
    "erfi",
    "erfinv",
]
