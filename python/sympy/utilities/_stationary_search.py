"""Bounded stationary-point proposals followed by original-source acceptance.

The numerical phase solves gradient = 0 using the already checked objective
program. Its convergence, Hessian and floating point are never proof inputs.
Only the independent second-order interval checker can publish a receipt.
"""

from fractions import Fraction
import math

from ._certified_numeric import (
    _MAX_ROOT_SEARCH_ATTEMPTS, _MAX_VARIABLES, _call_budgeted,
    _charge_exact_work, _sequence,
)
from ._certified_objective import certify_stationary_point
from .lambdify import _call_arguments, _check_evaluation_budget
from .objective import StationarySearchError


def _prepare_search(program, tol, maxiter, precision, max_attempts):
    """Refuse static ineligibility before numerical evaluation or conversion."""
    from ..solvers.numerical import _MAX_NEWTON_ITERATIONS

    if type(precision) is not int or not 8 <= precision <= 128:
        raise ValueError("certified precision must be an integer from 8 through 128")
    if (type(max_attempts) is not int
            or not 1 <= max_attempts <= _MAX_ROOT_SEARCH_ATTEMPTS):
        raise ValueError("stationary search max_attempts must be an integer from 1 through 24")
    if type(maxiter) is not int or not 1 <= maxiter <= _MAX_NEWTON_ITERATIONS:
        raise ValueError("stationary search maxiter must be an integer from 1 through 10,000")
    dimension = program.native_objective.variable_count
    if not 1 <= dimension <= _MAX_VARIABLES:
        raise ValueError("certified stationary search requires 1..16 variables")
    if program._has_inexact_source:
        raise ValueError("certified methods require exact source literals; replace Float with explicit Rational")
    if program._assumptions is None or len(program._assumptions) != dimension:
        raise ValueError("certified boxes require bounded Boolean assumptions matching captured symbol bindings")
    supported = {
        "positive", "negative", "nonnegative", "nonpositive", "nonzero", "zero",
        "extended_positive", "extended_negative", "extended_nonnegative",
        "extended_nonpositive", "extended_nonzero", "real", "extended_real",
        "complex", "finite", "commutative", "hermitian", "imaginary", "infinite",
    }
    for facts in program._assumptions:
        if any(key not in supported for key, _ in facts):
            raise NotImplementedError("certified real boxes do not support one or more declared symbol assumptions")
    tolerance = _call_arguments((None,), (tol,), native=True)[0]
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("stationary search tolerance must be positive, finite and real")
    return dimension, tolerance


def _initial_values(initial, dimension, budget):
    if type(initial) in (list, tuple):
        values = _sequence(initial, dimension, "stationary initial vector")
    elif dimension == 1:
        values = (initial,)
    else:
        raise TypeError("stationary initial vector must be a flat list/tuple in variable order")
    if budget is not None:
        budget._charge(dimension)
    values = _call_arguments((None,) * dimension, values, native=True)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("stationary initial values must be finite and real")
    return values


def _iterate_stationary(program, values, tolerance, maxiter, *, budget=None):
    """Use one immutable objective, including every original domain check."""
    from ..solvers.numerical import _iterate_system_functions

    native = program.native_objective
    dimension = native.variable_count
    if budget is not None:
        budget._charge(dimension * dimension)
    pattern = tuple(native.hessian_pattern)
    cached_point, cached_gradient, cached_hessian = None, None, None

    def evaluate(point, *, budget=None):
        nonlocal cached_point, cached_gradient, cached_hessian
        _check_evaluation_budget(budget)
        if budget is not None:
            budget._charge(dimension)
        key = tuple(point)
        if key != cached_point:
            # This full native evaluation checks the original value and both
            # derivative orders before exposing any result. A rejected trial
            # cannot hide a source hole behind a simplified gradient.
            _, gradient, sparse = _call_budgeted(native.evaluate, point, budget=budget)
            if budget is not None:
                budget._charge(dimension * dimension + len(pattern))
            hessian = [[0.0] * dimension for _ in range(dimension)]
            for (row, column), value in zip(pattern, sparse):
                hessian[row][column] = value
            _check_evaluation_budget(budget)
            cached_point, cached_gradient, cached_hessian = key, gradient, hessian
        return cached_gradient, cached_hessian

    def gradient(point, *, budget=None):
        return evaluate(point, budget=budget)[0]

    def hessian(point, *, budget=None):
        return evaluate(point, budget=budget)[1]

    def trial_gradient(point, *, budget=None):
        try:
            return gradient(point, budget=budget)
        except ValueError:
            # Only a closed native mathematical/domain refusal rejects this
            # trial. Operational errors and cancellation are different types
            # and propagate with all consumed work retained.
            return None

    # The existing bounded kernel applies an equilibrated, pivoted QR solve
    # and backtracking to the gradient residual, so maxima and saddles remain
    # valid stationary candidates. This is not a global minimization method.
    result = _iterate_system_functions(
        values, tolerance, maxiter, rows=dimension, columns=dimension,
        residuals=gradient, jacobian=hessian, trial_residuals=trial_gradient,
        budget=budget,
    )
    _check_evaluation_budget(budget)
    return result


def _search_stationary_box(program, candidate, precision, max_attempts, *, budget=None):
    """Vary exact witness boxes without trusting floating convergence."""
    dimension = program.native_objective.variable_count
    try:
        candidate = _sequence(candidate, dimension, "stationary candidate")
    except (TypeError, ValueError) as error:
        raise StationarySearchError(
            "candidate", "stationary search received an invalid finite-real candidate",
        ) from error
    if any(type(value) is not float or not math.isfinite(value) for value in candidate):
        raise StationarySearchError(
            "candidate", "stationary search received an invalid finite-real candidate",
        )
    _charge_exact_work(budget, bits=1075, operations=2 * dimension)
    center = tuple(Fraction(*value.as_integer_ratio()) for value in candidate)
    radii = tuple((abs(value) or Fraction(1)) / (1 << 20) for value in center)
    failures = []
    for attempt in range(max_attempts):
        # Narrow boxes reduce interval dependency/domain overestimation; wide
        # boxes can contain a stationary point outside an inaccurate proposal.
        # One total attempt count covers both directions and precision changes.
        exponent = (0 if attempt == 0 else
                    4 * ((attempt + 1) // 2) * (-1 if attempt % 2 else 1))
        _charge_exact_work(budget, *center, *radii, operations=4 * dimension + 1)
        scale = Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)
        box = tuple((value - radius * scale, value + radius * scale)
                    for value, radius in zip(center, radii))
        digits = min(128, precision << (attempt // 2))
        try:
            result = _call_budgeted(
                certify_stationary_point, program, box, center, None, digits, budget=budget,
            )
        except ValueError as error:
            # Native mathematical refusals use ValueError; some unsupported
            # native functions use it too and remain bounded failed trials.
            # TypeError, NotImplementedError, typed budget/cancellation errors
            # and memory failures propagate rather than resetting the search.
            failures.append(str(error))
        else:
            _check_evaluation_budget(budget)
            return result
    raise StationarySearchError(
        "certification",
        f"no stationary-point certificate accepted after {max_attempts} bounded trials; "
        f"last rejection: {failures[-1]}",
        failures=failures,
    )


def solve_stationary_certified(program, initial, tol, maxiter, precision, max_attempts,
                               *, budget=None):
    _check_evaluation_budget(budget)
    dimension, tolerance = _prepare_search(program, tol, maxiter, precision, max_attempts)
    values = _initial_values(initial, dimension, budget)
    try:
        candidate = _call_budgeted(
            _iterate_stationary, program, values, tolerance, maxiter, budget=budget,
        )
    except (ValueError, OverflowError, ZeroDivisionError) as error:
        raise StationarySearchError(
            "candidate", f"stationary search could not generate a candidate: {error}",
        ) from error
    _check_evaluation_budget(budget)
    result = _call_budgeted(
        _search_stationary_box, program, candidate, precision, max_attempts, budget=budget,
    )
    _check_evaluation_budget(budget)
    return result
