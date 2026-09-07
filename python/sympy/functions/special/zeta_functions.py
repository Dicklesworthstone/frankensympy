"""Zeta functions for FrankenSymPy."""

from ...core import _native, _native_expr, _wrap


def zeta(expression):
    return _wrap(_native.py_zeta(_native_expr(expression)))


__all__ = [
    "zeta",
]
