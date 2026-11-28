"""Algebraic and differential equation solvers for FrankenSymPy (WS19)."""

from .. import checksol, dsolve, solve, solveset
from ._linear_eq_to_matrix import linear_eq_to_matrix
from .certified import ScalarRootBracketReceipt, certified_root_bracket
from .exact_ode import dsolve_exact_first_order
from .inequalities import solve_univariate_inequality
from .numerical import RootSearchError, nsolve_certified
from .ode import (
    checkodesol,
    dsolve_cauchy_euler,
    dsolve_const_coeff_second_order,
    dsolve_const_coeff_second_order_nonhomogeneous,
    dsolve_linear_first_order,
    dsolve_separable_linear,
    verify_cauchy_euler_solution,
    verify_const_coeff_second_order_solution,
    verify_linear_first_order_solution,
)
from .pde import pdsolve
from .recurr import rsolve_poly
from .polysys import nonlinsolve, solve_poly_system
from .solvers import linsolve, nsolve, solve_linear, solve_linear_system, solve_linear_system_LU

__all__ = [
    "checkodesol",
    "checksol",
    "certified_root_bracket",
    "ScalarRootBracketReceipt",
    "dsolve",
    "dsolve_cauchy_euler",
    "dsolve_const_coeff_second_order",
    "dsolve_const_coeff_second_order_nonhomogeneous",
    "dsolve_exact_first_order",
    "dsolve_linear_first_order",
    "dsolve_separable_linear",
    "linear_eq_to_matrix",
    "linsolve",
    "nonlinsolve",
    "nsolve",
    "nsolve_certified",
    "RootSearchError",
    "pdsolve",
    "rsolve_poly",
    "solve",
    "solve_linear",
    "solve_linear_system",
    "solve_linear_system_LU",
    "solve_poly_system",
    "solve_univariate_inequality",
    "solveset",
    "verify_cauchy_euler_solution",
    "verify_const_coeff_second_order_solution",
    "verify_linear_first_order_solution",
]

