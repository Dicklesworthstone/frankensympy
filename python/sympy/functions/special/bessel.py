"""Bessel and Airy special functions for FrankenSymPy."""

from __future__ import annotations
from typing import Any

from ...core import (
    Basic,
    Expr,
    Function,
    Integer,
    oo,
)


class besselj(Function):
    """Bessel function of the first kind: J_\u03bd(z)."""

    @classmethod
    def eval(cls, nu: Any, z: Any) -> Any:
        if z == 0 or (isinstance(z, (int, Integer)) and int(z) == 0):
            if nu == 0 or (isinstance(nu, (int, Integer)) and int(nu) == 0):
                return Integer(1)
            if isinstance(nu, (int, Integer)) and int(nu) > 0:
                return Integer(0)
        if isinstance(nu, (int, Integer)):
            n = int(nu)
            if n < 0:
                sign = 1 if (-n) % 2 == 0 else -1
                return Integer(sign) * besselj(-n, z)
        return None


class bessely(Function):
    """Bessel function of the second kind: Y_\u03bd(z)."""

    @classmethod
    def eval(cls, nu: Any, z: Any) -> Any:
        if z == 0 or (isinstance(z, (int, Integer)) and int(z) == 0):
            return -oo
        if isinstance(nu, (int, Integer)):
            n = int(nu)
            if n < 0:
                sign = 1 if (-n) % 2 == 0 else -1
                return Integer(sign) * bessely(-n, z)
        return None


class besseli(Function):
    """Modified Bessel function of the first kind: I_\u03bd(z)."""

    @classmethod
    def eval(cls, nu: Any, z: Any) -> Any:
        if z == 0 or (isinstance(z, (int, Integer)) and int(z) == 0):
            if nu == 0 or (isinstance(nu, (int, Integer)) and int(nu) == 0):
                return Integer(1)
            if isinstance(nu, (int, Integer)) and int(nu) > 0:
                return Integer(0)
        if isinstance(nu, (int, Integer)):
            n = int(nu)
            if n < 0:
                return besseli(-n, z)
        return None


class besselk(Function):
    """Modified Bessel function of the second kind: K_\u03bd(z)."""

    @classmethod
    def eval(cls, nu: Any, z: Any) -> Any:
        if isinstance(nu, (int, Integer)):
            n = int(nu)
            if n < 0:
                return besselk(-n, z)
        return None


class hankel1(Function):
    """Hankel function of the first kind: H_\u03bd^(1)(z)."""
    pass


class hankel2(Function):
    """Hankel function of the second kind: H_\u03bd^(2)(z)."""
    pass


class jn(Function):
    """Spherical Bessel function of the first kind: j_n(z)."""

    @classmethod
    def eval(cls, n: Any, z: Any) -> Any:
        if z == 0 or (isinstance(z, (int, Integer)) and int(z) == 0):
            if n == 0 or (isinstance(n, (int, Integer)) and int(n) == 0):
                return Integer(1)
            if isinstance(n, (int, Integer)) and int(n) > 0:
                return Integer(0)
        return None


class yn(Function):
    """Spherical Bessel function of the second kind: y_n(z)."""

    @classmethod
    def eval(cls, n: Any, z: Any) -> Any:
        if z == 0 or (isinstance(z, (int, Integer)) and int(z) == 0):
            return -oo
        return None


class airyai(Function):
    """Airy Ai function: Ai(z)."""
    pass


class airybi(Function):
    """Airy Bi function: Bi(z)."""
    pass


__all__ = [
    "airyai",
    "airybi",
    "besseli",
    "besselj",
    "besselk",
    "bessely",
    "hankel1",
    "hankel2",
    "jn",
    "yn",
]
