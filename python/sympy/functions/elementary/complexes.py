"""Complex/real elementary functions for FrankenSymPy."""

from ...core import Abs, _native, _native_expr, _wrap


def sign(expression):
    return _wrap(_native.py_sign(_native_expr(expression)))


__all__ = [
    "Abs",
    "sign",
]
