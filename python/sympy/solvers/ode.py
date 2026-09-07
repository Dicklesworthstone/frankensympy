"""Exact ordinary differential equation (ODE) solvers for FrankenSymPy (WS19)."""

from typing import Any
from ..core import (
    Symbol,
    _native,
    _native_expr,
    _native_symbol_key,
    _parse_result,
    _require_symbol,
    _wrap,
)


def dsolve_linear_first_order(P: Any, Q: Any, x: Any) -> Any:
    """Solve first-order linear ODE: y'(x) + P(x)*y(x) = Q(x)."""
    x_sym = _require_symbol(x)
    p_str = str(_wrap(_native_expr(P)))
    q_str = str(_wrap(_native_expr(Q)))
    res = _native.dsolve_linear_first_order_expr(p_str, q_str, _native_symbol_key(x_sym))
    return _parse_result(res)


def dsolve_const_coeff_second_order(a: int, b: int, c: int, x: Any) -> Any:
    """Solve homogeneous 2nd-order ODE with constant coefficients: a*y''(x) + b*y'(x) + c*y(x) = 0."""
    x_sym = _require_symbol(x)
    res = _native.dsolve_const_coeff_second_order_expr(int(a), int(b), int(c), _native_symbol_key(x_sym))
    return _parse_result(res)


def dsolve_const_coeff_second_order_nonhomogeneous(
    a: int, b: int, c: int, f: Any, x: Any
) -> Any:
    """Solve non-homogeneous 2nd-order ODE with constant coefficients: a*y''(x) + b*y'(x) + c*y(x) = f(x)."""
    x_sym = _require_symbol(x)
    f_str = str(_wrap(_native_expr(f)))
    res = _native.dsolve_const_coeff_second_order_nonhomogeneous_expr(
        int(a), int(b), int(c), f_str, _native_symbol_key(x_sym)
    )
    return _parse_result(res)


def dsolve_cauchy_euler(a: int, b: int, c: int, x: Any) -> Any:
    """Solve Cauchy-Euler 2nd-order ODE: a*x^2*y''(x) + b*x*y'(x) + c*y(x) = 0."""
    x_sym = _require_symbol(x)
    res = _native.dsolve_cauchy_euler_expr(int(a), int(b), int(c), _native_symbol_key(x_sym))
    return _parse_result(res)


def dsolve_separable_linear(f: Any, x: Any) -> Any:
    """Solve separable ODE: y'(x) = f(x)*y(x)."""
    x_sym = _require_symbol(x)
    f_str = str(_wrap(_native_expr(f)))
    res = _native.dsolve_separable_linear_expr(f_str, _native_symbol_key(x_sym))
    return _parse_result(res)


__all__ = [
    "dsolve_cauchy_euler",
    "dsolve_const_coeff_second_order",
    "dsolve_const_coeff_second_order_nonhomogeneous",
    "dsolve_linear_first_order",
    "dsolve_separable_linear",
]
