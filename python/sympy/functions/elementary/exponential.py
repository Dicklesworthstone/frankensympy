"""Exponential and logarithmic functions for FrankenSymPy."""

from ...core import _native, _native_expr, _wrap


def exp(expression):
    return _wrap(_native.py_exp(_native_expr(expression)))


def log(expression):
    return _wrap(_native.py_log(_native_expr(expression)))


__all__ = [
    "exp",
    "log",
]
