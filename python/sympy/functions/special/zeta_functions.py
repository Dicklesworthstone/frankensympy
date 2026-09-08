"""Zeta and related special functions for FrankenSymPy."""

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
    zoo,
)


class zeta(Function):
    """The Riemann and Hurwitz zeta functions: ζ(s, a) = ∑ₖ₌₀^∞ 1/(k + a)^s."""

    @classmethod
    def eval(cls, s: Any, a: Any = None) -> Any:
        if a is None or a == 1 or (isinstance(a, (int, Integer)) and int(a) == 1):
            if s == 0 or (isinstance(s, (int, Integer)) and int(s) == 0):
                return Rational(-1, 2)
            if s == 1 or (isinstance(s, (int, Integer)) and int(s) == 1):
                return zoo
            if s == 2 or (isinstance(s, (int, Integer)) and int(s) == 2):
                return pi**2 / Integer(6)
            if s == 4 or (isinstance(s, (int, Integer)) and int(s) == 4):
                return pi**4 / Integer(90)
            if s == 6 or (isinstance(s, (int, Integer)) and int(s) == 6):
                return pi**6 / Integer(945)
            if isinstance(s, (int, Integer)):
                val = int(s)
                if val < 0 and val % 2 == 0:
                    return Integer(0)
                if val < 0 and val % 2 != 0:
                    from ..combinatorial.numbers import bernoulli
                    m = 1 - val
                    return -bernoulli(m) / Integer(m)
            try:
                native_res = _native.py_zeta(_native_expr(s))
                wrapped = _wrap(native_res)
                if not (isinstance(wrapped, zeta) or getattr(wrapped, "func_name", "") == "zeta"):
                    return wrapped
            except Exception:
                pass
        return None


class dirichlet_eta(Function):
    """The Dirichlet eta function: η(s) = (1 - 2^(1-s)) ζ(s)."""

    @classmethod
    def eval(cls, s: Any) -> Any:
        from ..elementary.exponential import log
        if s == 1 or (isinstance(s, (int, Integer)) and int(s) == 1):
            return log(Integer(2))
        return None


class polylog(Function):
    """The polylogarithm function: Li_s(z) = ∑ₖ₌₁^∞ z^k / k^s."""

    @classmethod
    def eval(cls, s: Any, z: Any) -> Any:
        from ..elementary.exponential import log
        if z == 0 or (isinstance(z, (int, Integer)) and int(z) == 0):
            return Integer(0)
        if z == 1 or (isinstance(z, (int, Integer)) and int(z) == 1):
            return zeta(s)
        if s == 1 or (isinstance(s, (int, Integer)) and int(s) == 1):
            return -log(Integer(1) - z)
        return None


class lerchphi(Function):
    """The Lerch transcendent: Φ(z, s, a) = ∑ₖ₌₀^∞ z^k / (k + a)^s."""
    pass


__all__ = [
    "dirichlet_eta",
    "lerchphi",
    "polylog",
    "zeta",
]
