"""Bounded scalar Newton iteration over the typed numerical math lane."""

import math

from ..core import (
    Float, _ExactDecimalFloat, _admitted_python_float, _require_symbol, diff,
)
from ..utilities.lambdify import lambdify

__all__ = ["nsolve"]

_MAX_NEWTON_ITERATIONS = 10_000


def _finite_real(value, name):
    if type(value) is str:
        raise TypeError(f"nsolve {name} must be a real number")
    result = (value._as_python_float() if type(value) is _ExactDecimalFloat
              else _admitted_python_float(value))
    if not math.isfinite(result):
        raise ValueError(f"nsolve {name} must be finite and real")
    return result


def nsolve(expr, var, initial, tol=1e-12, maxiter=200):
    """Find a finite real scalar Newton candidate for ``expr = 0``.

    Typed symbolic differentiation supplies the derivative. ``tol`` is an
    absolute tolerance for both the residual and the Newton step; an exactly
    zero binary64 residual also terminates. A small step alone never establishes
    convergence. Nonfinite values, zero derivatives away from a root, and
    stagnation above the residual tolerance raise ``ValueError``.

    The returned ``Float`` is an approximate numerical candidate, not a
    certified root or interval enclosure. Inputs and tolerance must be finite
    real numbers; ``maxiter`` is a built-in integer from 1 through 10,000.
    """
    var = _require_symbol(var)
    x_value = _finite_real(initial, "initial value")
    tolerance = _finite_real(tol, "tolerance")
    if tolerance <= 0:
        raise ValueError("nsolve tolerance must be positive")
    if type(maxiter) is not int:
        raise TypeError("nsolve maxiter must be a built-in integer")
    if not 1 <= maxiter <= _MAX_NEWTON_ITERATIONS:
        raise ValueError("nsolve maxiter must be between 1 and 10,000")

    residual = lambdify(var, expr, modules="math")
    derivative = lambdify(var, diff(expr, var), modules="math")

    def evaluate(function, value, name):
        try:
            return _finite_real(function(value), name)
        except (OverflowError, ZeroDivisionError, TypeError) as error:
            raise ValueError(f"nsolve {name} left the finite real domain") from error

    f_value = evaluate(residual, x_value, "residual")
    # Multiple roots and roots with a singular derivative may already satisfy
    # the equation. Do not evaluate or divide by the derivative in that case.
    if f_value == 0.0:
        return Float(x_value)

    for _ in range(maxiter):
        derivative_value = evaluate(derivative, x_value, "derivative")
        if derivative_value == 0.0:
            raise ValueError("nsolve derivative is zero away from a root")
        step = f_value / derivative_value
        candidate = x_value - step
        if not math.isfinite(step) or not math.isfinite(candidate):
            raise ValueError("nsolve Newton step left the finite real domain")
        candidate_residual = evaluate(residual, candidate, "residual")
        if candidate_residual == 0.0 or (
            abs(candidate_residual) <= tolerance and abs(step) <= tolerance
        ):
            return Float(candidate)
        if candidate == x_value:
            raise ValueError("nsolve stagnated before meeting the residual tolerance")
        x_value, f_value = candidate, candidate_residual

    raise ValueError(
        f"nsolve failed to converge in {maxiter} iterations "
        f"(last x = {x_value}, residual = {f_value})"
    )
