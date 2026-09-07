"""Error functions for FrankenSymPy."""

from ...core import _native, _native_expr, _wrap


def erf(expression):
    return _wrap(_native.py_erf(_native_expr(expression)))


def erfc(expression):
    return _wrap(_native.py_erfc(_native_expr(expression)))


__all__ = [
    "erf",
    "erfc",
]
