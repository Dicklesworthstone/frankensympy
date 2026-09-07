"""Combinatorial factorial functions for FrankenSymPy."""

from ...core import _native, _native_expr, _wrap


def factorial(expression):
    return _wrap(_native.py_factorial(_native_expr(expression)))


def subfactorial(expression):
    return _wrap(_native.py_subfactorial(_native_expr(expression)))


def binomial(n, k):
    return _wrap(_native.py_binomial(_native_expr(n), _native_expr(k)))


__all__ = [
    "binomial",
    "factorial",
    "subfactorial",
]
