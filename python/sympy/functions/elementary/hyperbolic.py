"""Hyperbolic functions for FrankenSymPy."""

from ...core import _native, _native_expr, _wrap


def sinh(expression):
    return _wrap(_native.py_sinh(_native_expr(expression)))


def cosh(expression):
    return _wrap(_native.py_cosh(_native_expr(expression)))


def tanh(expression):
    return _wrap(_native.py_tanh(_native_expr(expression)))


def coth(expression):
    return _wrap(_native.py_coth(_native_expr(expression)))


def sech(expression):
    return _wrap(_native.py_sech(_native_expr(expression)))


def csch(expression):
    return _wrap(_native.py_csch(_native_expr(expression)))


def asinh(expression):
    return _wrap(_native.py_asinh(_native_expr(expression)))


def acosh(expression):
    return _wrap(_native.py_acosh(_native_expr(expression)))


def atanh(expression):
    return _wrap(_native.py_atanh(_native_expr(expression)))


def acoth(expression):
    return _wrap(_native.py_acoth(_native_expr(expression)))


def asech(expression):
    return _wrap(_native.py_asech(_native_expr(expression)))


def acsch(expression):
    return _wrap(_native.py_acsch(_native_expr(expression)))


__all__ = [
    "acosh",
    "acoth",
    "acsch",
    "asech",
    "asinh",
    "atanh",
    "cosh",
    "coth",
    "csch",
    "sech",
    "sinh",
    "tanh",
]
