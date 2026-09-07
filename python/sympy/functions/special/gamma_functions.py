"""Gamma and related functions for FrankenSymPy."""

from ...core import _native, _native_expr, _wrap


def gamma(expression):
    return _wrap(_native.py_gamma(_native_expr(expression)))


__all__ = [
    "gamma",
]
