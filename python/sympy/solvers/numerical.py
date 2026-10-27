"""Bounded finite-real Newton candidates over typed numerical programs."""

import math

from ..core import (
    Add, Eq, Float, Mul, _ExactDecimalFloat, _admitted_python_float,
    _require_symbol, diff,
)
from ..utilities.lambdify import compile_residual_system, lambdify

__all__ = ["nsolve"]

_MAX_NEWTON_ITERATIONS = 10_000
_MAX_SYSTEM_VARIABLES = 64
_MAX_SYSTEM_RESIDUALS = 256
_MAX_JACOBIAN_ENTRIES = 4096
_MAX_LINE_SEARCH_STEPS = 64
_MAX_SYSTEM_EVALUATIONS = 4096
_MAX_LINEAR_WORK = 20_000_000


def _finite_real(value, name):
    if type(value) is str:
        raise TypeError(f"nsolve {name} must be a real number")
    result = (value._as_python_float() if type(value) is _ExactDecimalFloat
              else _admitted_python_float(value))
    if not math.isfinite(result):
        raise ValueError(f"nsolve {name} must be finite and real")
    return result


def _equation_residual(value):
    if type(value) is Eq:
        # Held built-in constructors do not run evaluation hooks on either
        # side. Closed numerical lowering will reject any opaque children.
        return Add(value.lhs, Mul(-1, value.rhs, evaluate=False), evaluate=False)
    return value


def _is_vector(value):
    from ..matrices import Matrix, ImmutableDenseMatrix, SparseMatrix

    return type(value) in (list, tuple, Matrix, ImmutableDenseMatrix, SparseMatrix)


def _vector(value, name, maximum):
    """Snapshot only admitted built-in containers, after checking their size."""
    from ..matrices import Matrix, ImmutableDenseMatrix, SparseMatrix

    if type(value) in (list, tuple):
        size = len(value)
        if not 1 <= size <= maximum:
            raise ValueError(f"nsolve {name} must contain 1 through {maximum} entries")
        return list(value)
    if type(value) in (Matrix, ImmutableDenseMatrix, SparseMatrix):
        rows, columns = value.rows, value.cols
        size = rows * columns
        if not 1 <= size <= maximum:
            raise ValueError(f"nsolve {name} must contain 1 through {maximum} entries")
        if rows != 1 and columns != 1:
            raise ValueError(f"nsolve {name} must be a row or column vector")
        return [value[row, column] for row in range(rows) for column in range(columns)]
    raise TypeError(f"nsolve {name} must be a built-in list, tuple, or matrix vector")


def _least_squares_step(jacobian, residuals):
    """Solve ``J * step = residual`` by equilibrated, pivoted Householder QR.

    Row equilibration prevents independently scaled equations from disappearing;
    column equilibration preserves small but nonzero variable sensitivities.
    Column-pivoted QR avoids forming normal equations and their squared condition
    number. Numerically rank-deficient systems refuse. This is approximate
    linear algebra, with no root, rank, or conditioning certificate implied.
    """
    rows, columns = len(jacobian), len(jacobian[0])
    weights = [max(abs(value), *(abs(entry) for entry in row)) or 1.0
               for row, value in zip(jacobian, residuals)]
    matrix = [[entry / weight for entry in row]
              for row, weight in zip(jacobian, weights)]
    rhs = [value / weight for value, weight in zip(residuals, weights)]
    scales = [max(abs(row[column]) for row in matrix) for column in range(columns)]
    if any(scale == 0.0 for scale in scales):
        raise ValueError("nsolve Jacobian is numerically rank-deficient away from a root")
    for row in matrix:
        for column in range(columns):
            row[column] /= scales[column]

    permutation = list(range(columns))
    largest_norm = max(math.hypot(*(row[column] for row in matrix))
                       for column in range(columns))
    rank_threshold = 8.0 * math.ulp(1.0) * max(rows, columns) * largest_norm
    for column in range(columns):
        norms = [math.hypot(*(matrix[row][other] for row in range(column, rows)))
                 for other in range(column, columns)]
        pivot = column + max(range(len(norms)), key=norms.__getitem__)
        norm = norms[pivot - column]
        if norm <= rank_threshold:
            raise ValueError("nsolve Jacobian is numerically rank-deficient or ill-conditioned")
        if pivot != column:
            for row in matrix:
                row[column], row[pivot] = row[pivot], row[column]
            permutation[column], permutation[pivot] = permutation[pivot], permutation[column]

        diagonal = -math.copysign(norm, matrix[column][column])
        reflector = [matrix[column][column] - diagonal]
        reflector.extend(matrix[row][column] for row in range(column + 1, rows))
        reflector_norm = math.hypot(*reflector)
        reflector = [value / reflector_norm for value in reflector]
        for other in range(column + 1, columns):
            projection = 2.0 * math.fsum(
                value * matrix[column + offset][other]
                for offset, value in enumerate(reflector)
            )
            for offset, value in enumerate(reflector):
                matrix[column + offset][other] -= projection * value
        projection = 2.0 * math.fsum(
            value * rhs[column + offset] for offset, value in enumerate(reflector)
        )
        for offset, value in enumerate(reflector):
            rhs[column + offset] -= projection * value
        matrix[column][column] = diagonal
        for row in range(column + 1, rows):
            matrix[row][column] = 0.0

    permuted = [0.0] * columns
    for row in range(columns - 1, -1, -1):
        remainder = math.fsum(matrix[row][column] * permuted[column]
                              for column in range(row + 1, columns))
        permuted[row] = (rhs[row] - remainder) / matrix[row][row]
        if not math.isfinite(permuted[row]):
            raise ValueError("nsolve linear solve left the finite real domain")
    step = [0.0] * columns
    for column, original in enumerate(permutation):
        step[original] = permuted[column] / scales[original]
    if any(not math.isfinite(value) for value in step):
        raise ValueError("nsolve Newton step left the finite real domain")
    return step, weights


def _solve_system(expr, var, initial, tolerance, maxiter):
    variables = _vector(var, "variables", _MAX_SYSTEM_VARIABLES)
    variables = [_require_symbol(value) for value in variables]
    if len(set(variables)) != len(variables):
        raise ValueError("nsolve has duplicate typed variables")
    residuals = _vector(expr, "residuals", _MAX_SYSTEM_RESIDUALS)
    rows, columns = len(residuals), len(variables)
    if rows < columns:
        raise ValueError("nsolve needs at least as many equations as variables")
    if rows * columns > _MAX_JACOBIAN_ENTRIES:
        raise ValueError("nsolve system exceeds the dense Jacobian entry limit")
    values = (_vector(initial, "initial values", _MAX_SYSTEM_VARIABLES)
              if _is_vector(initial) else [initial])
    if len(values) != columns:
        raise TypeError("nsolve initial vector must match the number of variables")
    values = [_finite_real(value, "initial value") for value in values]
    residuals = [_equation_residual(value) for value in residuals]
    program = compile_residual_system(residuals, variables)

    evaluations = 0

    def evaluate(function, at):
        nonlocal evaluations
        if evaluations >= _MAX_SYSTEM_EVALUATIONS:
            raise ValueError("nsolve system exhausted its numerical evaluation budget")
        evaluations += 1
        return function(*at)

    result = evaluate(program.residuals, values)
    norm = max(abs(value) for value in result)
    if norm == 0.0:
        return variables, values
    linear_work = 0
    for _ in range(maxiter):
        # Charge the dense factorization before allocating its working copy.
        linear_work += rows * columns * (columns + 1)
        if linear_work > _MAX_LINEAR_WORK:
            raise ValueError("nsolve system exhausted its dense linear-algebra work budget")
        jacobian = evaluate(program.jacobian, values)
        try:
            step, weights = _least_squares_step(jacobian, result)
        except (OverflowError, ZeroDivisionError) as error:
            raise ValueError("nsolve linear solve left the finite real domain") from error
        step_norm = max(abs(value) for value in step)
        if norm <= tolerance and step_norm <= tolerance:
            return variables, values
        merit = math.hypot(*(value / weight for value, weight in zip(result, weights)))
        fraction = 1.0
        for _ in range(_MAX_LINE_SEARCH_STEPS):
            candidate = [value - fraction * change for value, change in zip(values, step)]
            if candidate == values:
                raise ValueError("nsolve stagnated before meeting the residual and step tolerances")
            candidate_result = None
            if all(math.isfinite(value) for value in candidate):
                # A trial may cross a pole or leave a function's real domain.
                # Only that rejected trial is discarded; the iteration budget
                # remains consumed, and unsupported compilation never falls back.
                if evaluations >= _MAX_SYSTEM_EVALUATIONS:
                    raise ValueError("nsolve system exhausted its numerical evaluation budget")
                try:
                    candidate_result = evaluate(program.residuals, candidate)
                except ValueError:
                    pass
            if candidate_result is not None:
                candidate_norm = max(abs(value) for value in candidate_result)
                if candidate_norm == 0.0 or (
                    candidate_norm <= tolerance and step_norm <= tolerance
                ):
                    return variables, candidate
                candidate_merit = math.hypot(*(
                    value / weight for value, weight in zip(candidate_result, weights)
                ))
                # Weights stay fixed throughout this line search. An undamped
                # least-squares direction therefore minimizes the same local
                # residual objective that decides whether a trial is useful.
                if candidate_norm <= tolerance or (
                    candidate_merit < merit
                    and candidate_merit <= (1.0 - 0.0001 * fraction) * merit
                ):
                    values, result, norm = candidate, candidate_result, candidate_norm
                    break
            fraction *= 0.5
        else:
            raise ValueError("nsolve line search could not reduce the residual within its budget")

    raise ValueError(
        f"nsolve failed to converge in {maxiter} iterations (last residual norm = {norm})"
    )


def nsolve(expr, var, initial, tol=1e-12, maxiter=200, *, dict=False):
    """Find a finite real Newton candidate for an equation or residual system.

    Scalars use typed symbolic differentiation. Systems supply equations and
    variables as list, tuple, or matrix vectors, using a checked native symbolic
    Jacobian, equilibrated column-pivoted QR, and a bounded backtracking line
    search. Square and overdetermined systems are supported; underdetermined
    and numerically rank-deficient systems refuse. Matrix inputs are snapshots.

    ``tol`` is an absolute tolerance for every residual and the undamped Newton
    step. An exactly zero binary64 residual also terminates. A small or damped
    step alone never establishes convergence. Nonfinite values and stagnation
    above tolerance raise ``ValueError``. System dimensions, evaluations, dense
    linear-algebra work, and line-search trials have independent fixed bounds.

    Return a ``Float`` for a scalar or a column ``Matrix`` for a system.
    ``dict=True`` returns one solution mapping inside a list. Results are
    approximate binary64 candidates, not certified roots or interval enclosures.
    Complex arithmetic, arbitrary precision, custom numerical callbacks, and
    alternative solver/back-end keywords are outside this bounded preview.
    ``maxiter`` is a built-in integer from 1 through 10,000.
    """
    if type(dict) is not bool:
        raise TypeError("nsolve dict must be a built-in boolean")
    tolerance = _finite_real(tol, "tolerance")
    if tolerance <= 0:
        raise ValueError("nsolve tolerance must be positive")
    if type(maxiter) is not int:
        raise TypeError("nsolve maxiter must be a built-in integer")
    if not 1 <= maxiter <= _MAX_NEWTON_ITERATIONS:
        raise ValueError("nsolve maxiter must be between 1 and 10,000")

    if _is_vector(expr) or _is_vector(var):
        from ..matrices import Matrix

        variables, values = _solve_system(expr, var, initial, tolerance, maxiter)
        roots = [Float(value) for value in values]
        if dict:
            return [{variable: root for variable, root in zip(variables, roots)}]
        return Matrix(roots)

    var = _require_symbol(var)
    x_value = _finite_real(initial, "initial value")
    expr = _equation_residual(expr)

    def solution(value):
        root = Float(value)
        return [{var: root}] if dict else root

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
        return solution(x_value)

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
            return solution(candidate)
        if candidate == x_value:
            raise ValueError("nsolve stagnated before meeting the residual tolerance")
        x_value, f_value = candidate, candidate_residual

    raise ValueError(
        f"nsolve failed to converge in {maxiter} iterations "
        f"(last x = {x_value}, residual = {f_value})"
    )
