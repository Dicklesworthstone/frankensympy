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


def checkodesol(
    ode: Any,
    sol: Any,
    func: Any = None,
    order: Any = "auto",
    solve_for_func: bool = True,
) -> Any:
    """Check whether a candidate solution satisfies a given ODE.

    Returns (True, 0) if the candidate solution is verified to satisfy the ODE,
    or (False, residual) if the residual is nonzero.

    If sol is a list or tuple of solutions, returns a list of (bool, residual) tuples.
    """
    if isinstance(sol, (list, tuple)):
        return [
            checkodesol(ode, s, func=func, order=order, solve_for_func=solve_for_func)
            for s in sol
        ]

    from ..core import Equality, Integer

    if isinstance(ode, Equality):
        ode_expr = ode.lhs - ode.rhs
    else:
        ode_expr = _wrap(_native_expr(ode))

    if isinstance(sol, Equality):
        if func is None:
            func = sol.lhs
            rhs = sol.rhs
        else:
            func = _wrap(_native_expr(func))
            if sol.lhs == func:
                rhs = sol.rhs
            elif sol.rhs == func:
                rhs = sol.lhs
            else:
                rhs = sol.rhs
    else:
        if func is None:
            known_atom_names = {
                "Add", "Mul", "Pow", "Symbol", "Integer", "Rational", "Float",
                "exp", "sin", "cos", "tan", "sinh", "cosh", "tanh",
                "asin", "acos", "atan", "asinh", "acosh", "atanh",
                "log", "ln", "sqrt", "diff", "Derivative", "Abs",
            }
            candidates = []
            for atom in ode_expr.atoms():
                fname = getattr(atom, "func", None)
                fname_str = getattr(fname, "__name__", None) or str(fname)
                if fname_str not in known_atom_names and hasattr(atom, "args") and atom.args:
                    candidates.append(atom)
            if candidates:
                func = candidates[0]
            else:
                raise ValueError("func must be specified if sol is not an Equality")
        else:
            func = _wrap(_native_expr(func))
        rhs = _wrap(_native_expr(sol))

    subbed = ode_expr.subs(func, rhs)
    evaluated = subbed.doit()
    residual = evaluated.simplify()

    if getattr(residual, "is_zero", False) or residual == 0 or residual == Integer(0):
        return (True, 0)
    else:
        return (False, residual)


def verify_linear_first_order_solution(sol: Any, P: Any, Q: Any, x: Any) -> bool:
    """Verify candidate solution of y'(x) + P(x)*y(x) = Q(x) via independent native verifier."""
    x_sym = _require_symbol(x)
    sol_str = str(_wrap(_native_expr(sol)))
    p_str = str(_wrap(_native_expr(P)))
    q_str = str(_wrap(_native_expr(Q)))
    return _native.verify_linear_first_order_solution_expr(
        sol_str, p_str, q_str, _native_symbol_key(x_sym)
    )


def verify_const_coeff_second_order_solution(sol: Any, a: int, b: int, c: int, x: Any) -> bool:
    """Verify candidate solution of a*y''(x) + b*y'(x) + c*y(x) = 0 via independent native verifier."""
    x_sym = _require_symbol(x)
    sol_str = str(_wrap(_native_expr(sol)))
    return _native.verify_const_coeff_second_order_solution_expr(
        sol_str, int(a), int(b), int(c), _native_symbol_key(x_sym)
    )


def verify_cauchy_euler_solution(sol: Any, a: int, b: int, c: int, x: Any) -> bool:
    """Verify candidate solution of a*x^2*y''(x) + b*x*y'(x) + c*y(x) = 0 via independent native verifier."""
    x_sym = _require_symbol(x)
    sol_str = str(_wrap(_native_expr(sol)))
    return _native.verify_cauchy_euler_solution_expr(
        sol_str, int(a), int(b), int(c), _native_symbol_key(x_sym)
    )


__all__ = [
    "checkodesol",
    "dsolve_cauchy_euler",
    "dsolve_const_coeff_second_order",
    "dsolve_const_coeff_second_order_nonhomogeneous",
    "dsolve_linear_first_order",
    "dsolve_separable_linear",
    "verify_cauchy_euler_solution",
    "verify_const_coeff_second_order_solution",
    "verify_linear_first_order_solution",
]
