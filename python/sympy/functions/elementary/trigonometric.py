"""Trigonometric functions for FrankenSymPy."""

from ...core import _native, _native_expr, _wrap


def sin(expression):
    return _wrap(_native.py_sin(_native_expr(expression)))


def cos(expression):
    return _wrap(_native.py_cos(_native_expr(expression)))


def tan(expression):
    return _wrap(_native.py_tan(_native_expr(expression)))


def cot(expression):
    return _wrap(_native.py_cot(_native_expr(expression)))


def sec(expression):
    return _wrap(_native.py_sec(_native_expr(expression)))


def csc(expression):
    return _wrap(_native.py_csc(_native_expr(expression)))


def asin(expression):
    return _wrap(_native.py_asin(_native_expr(expression)))


def acos(expression):
    return _wrap(_native.py_acos(_native_expr(expression)))


def atan(expression):
    return _wrap(_native.py_atan(_native_expr(expression)))


def acot(expression):
    return _wrap(_native.py_acot(_native_expr(expression)))


def asec(expression):
    return _wrap(_native.py_asec(_native_expr(expression)))


def acsc(expression):
    return _wrap(_native.py_acsc(_native_expr(expression)))


def sinc(expression):
    return _wrap(_native.py_sinc(_native_expr(expression)))


__all__ = [
    "acos",
    "acot",
    "acsc",
    "asec",
    "asin",
    "atan",
    "cos",
    "cot",
    "csc",
    "sec",
    "sin",
    "sinc",
    "tan",
]
