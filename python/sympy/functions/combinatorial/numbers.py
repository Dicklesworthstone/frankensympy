"""Combinatorial number functions for FrankenSymPy."""

from ...core import _native, _native_expr, _wrap


def fibonacci(expression):
    return _wrap(_native.py_fibonacci(_native_expr(expression)))


def lucas(expression):
    return _wrap(_native.py_lucas(_native_expr(expression)))


def bernoulli(expression):
    return _wrap(_native.py_bernoulli(_native_expr(expression)))


def bell(expression):
    return _wrap(_native.py_bell(_native_expr(expression)))


def harmonic(expression):
    return _wrap(_native.py_harmonic(_native_expr(expression)))


def catalan(expression):
    return _wrap(_native.py_catalan(_native_expr(expression)))


__all__ = [
    "bell",
    "bernoulli",
    "catalan",
    "fibonacci",
    "harmonic",
    "lucas",
]
