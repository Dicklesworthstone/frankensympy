"""Integer-valued elementary functions for FrankenSymPy."""

from ...core import _native, _native_expr, _wrap


def floor(expression):
    return _wrap(_native.py_floor(_native_expr(expression)))


def ceiling(expression):
    return _wrap(_native.py_ceiling(_native_expr(expression)))


__all__ = [
    "ceiling",
    "floor",
]
