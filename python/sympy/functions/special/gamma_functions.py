"""Gamma and related special functions for FrankenSymPy."""

from __future__ import annotations
from typing import Any

from ...core import (
    Basic,
    Expr,
    Function,
    Integer,
    Rational,
    _native,
    _native_expr,
    _wrap,
    pi,
    sqrt,
    zoo,
)
from ..combinatorial.factorials import factorial


class gamma(Function):
    """The Gamma function: Γ(z) = ∫₀^∞ t^(z-1) e^(-t) dt."""

    @classmethod
    def eval(cls, arg: Any) -> Any:
        if isinstance(arg, (int, Integer)):
            n = int(arg)
            if n <= 0:
                return zoo
            return factorial(n - 1)
        if isinstance(arg, Rational):
            # Half-integer values
            p, q = arg.p, arg.q
            if q == 2:
                if p == 1:
                    return sqrt(pi)
                if p == 3:
                    return sqrt(pi) / Integer(2)
                if p == 5:
                    return Integer(3) * sqrt(pi) / Integer(4)
                if p == 7:
                    return Integer(15) * sqrt(pi) / Integer(8)
                if p == -1:
                    return Integer(-2) * sqrt(pi)
                if p == -3:
                    return Integer(4) * sqrt(pi) / Integer(3)
        try:
            native_res = _native.py_gamma(_native_expr(arg))
            wrapped = _wrap(native_res)
            if not (isinstance(wrapped, gamma) or getattr(wrapped, "func_name", "") == "gamma"):
                return wrapped
        except Exception:
            pass
        return None


class lowergamma(Function):
    """The lower incomplete gamma function: γ(s, x) = ∫₀^x t^(s-1) e^(-t) dt."""

    @classmethod
    def eval(cls, s: Any, x: Any) -> Any:
        from ..elementary.exponential import exp
        if x == 0 or (isinstance(x, (int, Integer)) and int(x) == 0):
            return Integer(0)
        if s == 1 or (isinstance(s, (int, Integer)) and int(s) == 1):
            return Integer(1) - exp(-x)
        return None


class uppergamma(Function):
    """The upper incomplete gamma function: Γ(s, x) = ∫_x^∞ t^(s-1) e^(-t) dt."""

    @classmethod
    def eval(cls, s: Any, x: Any) -> Any:
        from ..elementary.exponential import exp
        if x == 0 or (isinstance(x, (int, Integer)) and int(x) == 0):
            return gamma(s)
        if s == 1 or (isinstance(s, (int, Integer)) and int(s) == 1):
            return exp(-x)
        return None


class polygamma(Function):
    """The polygamma function: ψ^(n)(x) = d^(n+1)/dx^(n+1) ln Γ(x)."""

    @classmethod
    def eval(cls, n: Any, x: Any) -> Any:
        from ...core import EulerGamma
        if n == 0 or (isinstance(n, (int, Integer)) and int(n) == 0):
            if x == 1 or (isinstance(x, (int, Integer)) and int(x) == 1):
                return -EulerGamma
        if n == 1 or (isinstance(n, (int, Integer)) and int(n) == 1):
            if x == 1 or (isinstance(x, (int, Integer)) and int(x) == 1):
                return pi**2 / Integer(6)
        return None


def digamma(x: Any) -> Any:
    """The digamma function: ψ(x) = polygamma(0, x)."""
    return polygamma(Integer(0), x)


def trigamma(x: Any) -> Any:
    """The trigamma function: ψ'(x) = polygamma(1, x)."""
    return polygamma(Integer(1), x)


class loggamma(Function):
    """The principal branch of the logarithm of the gamma function: ln Γ(z)."""

    @classmethod
    def eval(cls, z: Any) -> Any:
        if z in (1, 2) or (isinstance(z, (int, Integer)) and int(z) in (1, 2)):
            return Integer(0)
        return None


class beta(Function):
    """The Euler Beta function: B(x, y) = Γ(x)Γ(y) / Γ(x + y)."""

    @classmethod
    def eval(cls, x: Any, y: Any) -> Any:
        if x == 1 or (isinstance(x, (int, Integer)) and int(x) == 1):
            return Integer(1) / y
        if y == 1 or (isinstance(y, (int, Integer)) and int(y) == 1):
            return Integer(1) / x
        if isinstance(x, (int, Integer)) and isinstance(y, (int, Integer)):
            ix, iy = int(x), int(y)
            if ix > 0 and iy > 0:
                num = factorial(ix - 1) * factorial(iy - 1)
                den = factorial(ix + iy - 1)
                return num / den
        return None


__all__ = [
    "beta",
    "digamma",
    "gamma",
    "loggamma",
    "lowergamma",
    "polygamma",
    "trigamma",
    "uppergamma",
]
