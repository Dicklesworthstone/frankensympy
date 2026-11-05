"""Bounded real Newton candidates and independently certified local roots."""

import math

from ..core import (
    Add, Basic, Eq, Float, Mul, Relational, _ExactDecimalFloat, _admitted_python_float,
    _require_symbol, diff,
)
from ..utilities.lambdify import (
    _check_evaluation_budget, compile_numeric, compile_residual_system, lambdify,
)

__all__ = ["nsolve", "nsolve_certified", "RootSearchError"]

_MAX_NEWTON_ITERATIONS = 10_000
_MAX_SYSTEM_VARIABLES = 64
_MAX_SYSTEM_RESIDUALS = 256
_MAX_JACOBIAN_ENTRIES = 4096
_MAX_LINE_SEARCH_STEPS = 64
_MAX_SYSTEM_EVALUATIONS = 4096
_MAX_LINEAR_WORK = 20_000_000
_RELATIONAL_SIDES = Relational.__dict__["_relational_sides"]
_TYPE_MRO = type.__dict__["__mro__"]


class RootSearchError(ValueError):
    """A bounded search did not produce an accepted local root certificate.

    ``stage`` distinguishes candidate generation from certification. The
    immutable ``failures`` tuple records rejected certificate trials; neither
    this exception nor a floating candidate carries a root-existence claim.
    """

    def __init__(self, stage, message, *, failures=()):
        self.stage = stage
        self.failures = tuple(failures)
        self.attempts = len(self.failures)
        super().__init__(message)


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
        try:
            sides = _RELATIONAL_SIDES.__get__(value, Relational)
        except AttributeError:
            raise NotImplementedError(
                "numerical equations require retained Python sides; "
                "reconstruct Eq from explicit operands"
            ) from None
        if (type(sides) is not tuple or len(sides) != 2
                or any(not any(base is Basic for base in _TYPE_MRO.__get__(type(side), type))
                       for side in sides)):
            raise ValueError("numerical equation source requires an exact captured pair of Basic sides")
        # Held built-in constructors do not run evaluation hooks on either
        # side. Never use public operand getters or name-based native lifting;
        # closed numerical lowering will reject any opaque retained children.
        return Add(sides[0], Mul(-1, sides[1], evaluate=False), evaluate=False)
    return value


def _is_vector(value):
    from ..matrices import Matrix, ImmutableDenseMatrix, ImmutableSparseMatrix, SparseMatrix

    return any(type(value) is kind for kind in (
        list, tuple, Matrix, ImmutableDenseMatrix, SparseMatrix, ImmutableSparseMatrix,
    ))


def _vector(value, name, maximum):
    """Snapshot only admitted built-in containers, after checking their size."""
    from ..matrices import Matrix, ImmutableDenseMatrix, ImmutableSparseMatrix, SparseMatrix

    if type(value) in (list, tuple):
        size = len(value)
        if not 1 <= size <= maximum:
            raise ValueError(f"nsolve {name} must contain 1 through {maximum} entries")
        try:
            snapshot = [value[index] for index in range(size)]
        except IndexError:
            raise ValueError(f"nsolve {name} length changed during its snapshot") from None
        if len(value) != size:
            raise ValueError(f"nsolve {name} length changed during its snapshot")
        return snapshot
    if any(type(value) is kind for kind in (
        Matrix, ImmutableDenseMatrix, SparseMatrix, ImmutableSparseMatrix,
    )):
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


def _solve_system(expr, var, initial, tolerance, maxiter, budget=None):
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
    return variables, _iterate_system(program, values, tolerance, maxiter, budget=budget)


def _iterate_system(program, values, tolerance, maxiter, budget=None):
    """Generate candidates using one captured native residual/Jacobian program.

    Inputs are already admitted finite floats in flat native-variable order.
    Neither convergence nor this function's return value grants evidence.
    An optional caller ledger is shared by all evaluations and logical dense
    solve work. Rejected backtracking trials retain their consumed resources.
    """
    _check_evaluation_budget(budget)
    native = program.native_system

    def trial_residuals(point, *, budget=None):
        try:
            return (native.residuals(point) if budget is None else
                    native.residuals(point, budget=budget))
        except ValueError:
            # This closed native lane has no Python effects. Domain refusals
            # can reject a line-search trial without replaying unknown work.
            return None

    return _iterate_system_functions(
        values, tolerance, maxiter,
        rows=native.residual_count, columns=native.variable_count,
        residuals=native.residuals, jacobian=native.jacobian,
        trial_residuals=trial_residuals, budget=budget,
    )


def _iterate_system_functions(values, tolerance, maxiter, *, rows, columns,
                              residuals, jacobian, trial_residuals,
                              budget=None, charge_work=None):
    """Bounded QR iteration over already admitted numerical evaluators.

    ``trial_residuals`` alone decides whether a known domain refusal may
    become ``None``. Callback exceptions and operational failures are never
    classified here or formatted into new exceptions. A supervising caller
    can additionally charge the same dense work to its owning callback session.
    This kernel produces approximate candidates without derivative evidence.
    """
    _check_evaluation_budget(budget)

    evaluations = 0

    def evaluate(function, at):
        nonlocal evaluations
        if evaluations >= _MAX_SYSTEM_EVALUATIONS:
            raise ValueError("nsolve system exhausted its numerical evaluation budget")
        evaluations += 1
        return function(at) if budget is None else function(at, budget=budget)

    def charge(units):
        if budget is not None:
            budget._charge(units)
        if charge_work is not None:
            charge_work(units)

    result = evaluate(residuals, values)
    norm = max(abs(value) for value in result)
    if norm == 0.0:
        return values
    linear_work = 0
    for _ in range(maxiter):
        # Charge the dense factorization before allocating its working copy.
        linear_units = rows * columns * (columns + 1)
        linear_work += linear_units
        if linear_work > _MAX_LINEAR_WORK:
            raise ValueError("nsolve system exhausted its dense linear-algebra work budget")
        charge(linear_units)
        jacobian_values = evaluate(jacobian, values)
        try:
            step, weights = _least_squares_step(jacobian_values, result)
        except (OverflowError, ZeroDivisionError) as error:
            raise ValueError("nsolve linear solve left the finite real domain") from error
        step_norm = max(abs(value) for value in step)
        if norm <= tolerance and step_norm <= tolerance:
            return values
        merit = math.hypot(*(value / weight for value, weight in zip(result, weights)))
        fraction = 1.0
        for _ in range(_MAX_LINE_SEARCH_STEPS):
            charge(3 * columns)
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
                candidate_result = evaluate(trial_residuals, candidate)
            if candidate_result is not None:
                candidate_norm = max(abs(value) for value in candidate_result)
                if candidate_norm == 0.0 or (
                    candidate_norm <= tolerance and step_norm <= tolerance
                ):
                    return candidate
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


def _newton_options(tol, maxiter):
    tolerance = _finite_real(tol, "tolerance")
    if tolerance <= 0:
        raise ValueError("nsolve tolerance must be positive")
    if type(maxiter) is not int:
        raise TypeError("nsolve maxiter must be a built-in integer")
    if not 1 <= maxiter <= _MAX_NEWTON_ITERATIONS:
        raise ValueError("nsolve maxiter must be between 1 and 10,000")
    return tolerance


def _candidate_options(backend, tol, maxiter, max_step_norm, *, allow_python):
    """Admit backend-specific controls before any donor or conversion work."""
    if type(backend) is not str:
        raise TypeError("numerical backend must be an exact built-in string")
    allowed = ("python", "frankennumpy", "frankenscipy") if allow_python else (
        "frankennumpy", "frankenscipy",
    )
    if backend not in allowed:
        raise ValueError("unsupported numerical backend")
    if backend == "frankenscipy":
        if tol is not None or maxiter is not None or max_step_norm is not None:
            raise ValueError(
                "FrankenSciPy has fixed solver controls; omit tol, maxiter and max_step_norm"
            )
        return None, None, None
    if backend == "python":
        if max_step_norm is not None:
            raise ValueError("the python backend does not accept max_step_norm")
        tol = 1e-12 if tol is None else tol
        maxiter = 200 if maxiter is None else maxiter
        return _newton_options(tol, maxiter), maxiter, None
    tol = _finite_real(1e-12 if tol is None else tol, "tolerance")
    if tol < 0:
        raise ValueError("FrankenNumPy tolerance must be nonnegative")
    maxiter = 64 if maxiter is None else maxiter
    if type(maxiter) is not int:
        raise TypeError("FrankenNumPy maxiter must be a built-in integer")
    if not 0 <= maxiter <= 1024:
        raise ValueError("FrankenNumPy maxiter must be between 0 and 1024")
    max_step_norm = _finite_real(
        1e6 if max_step_norm is None else max_step_norm, "maximum step norm",
    )
    if max_step_norm <= 0:
        raise ValueError("FrankenNumPy max_step_norm must be positive")
    return tol, maxiter, max_step_norm


def _candidate_initial(program, initial):
    columns = program.native_system.variable_count
    if not 1 <= columns <= 16 or program.native_system.residual_count != columns:
        raise ValueError("numerical consumers require a square system with 1..16 variables")
    values = (_vector(initial, "initial values", columns)
              if _is_vector(initial) else [initial])
    if len(values) != columns:
        raise TypeError("initial vector must match the number of variables")
    return [_finite_real(value, "initial value") for value in values]


def _solve_compiled_certified(program, initial, *, tol, maxiter, precision, max_attempts,
                              backend="python", max_step_norm=None, budget=None):
    from ..utilities._certified_numeric import prepare_root_search, search_root_box

    _check_evaluation_budget(budget)
    tolerance, iterations, step_limit = _candidate_options(
        backend, tol, maxiter, max_step_norm, allow_python=True,
    )
    # Check the original immutable program, shape, assumptions and policy
    # before running any floating candidate work or user conversion hooks.
    prepare_root_search(program, precision, max_attempts)
    values = _candidate_initial(program, initial)
    try:
        if backend == "python":
            candidate = (_iterate_system(program, values, tolerance, iterations)
                         if budget is None else
                         _iterate_system(program, values, tolerance, iterations, budget=budget))
        else:
            # The real donor supplies only a floating proposal. It cannot
            # bypass source/assumption admission or the independent checker.
            candidate = program.solve_candidate(
                values, backend=backend, tol=tolerance, maxiter=iterations,
                max_step_norm=step_limit, budget=budget,
            ).root
    except (ValueError, OverflowError, ZeroDivisionError) as error:
        raise RootSearchError("candidate", f"certified root search could not generate a candidate: {error}") from error
    _check_evaluation_budget(budget)
    result = (search_root_box(program, candidate, precision, max_attempts)
              if budget is None else
              search_root_box(program, candidate, precision, max_attempts, budget=budget))
    _check_evaluation_budget(budget)
    return result


def nsolve_certified(expr, var, initial, *, tol=None, maxiter=None,
                     backend="python", max_step_norm=None,
                     precision=32, max_attempts=12, budget=None):
    """Locate and independently certify one nonsingular real root locally.

    Accept a scalar equation and Symbol or equal-length equation/variable
    vectors. One immutable exact-source residual program supplies both the
    approximate Newton search and the independent original-source checker.
    Systems must be square with 1..16 variables. Floating initial guesses are
    untrusted proposals; source literals must be exact and custom callbacks
    are not admitted. Only supported Boolean real/sign assumptions are allowed.

    After bounded numerical iteration, exact rational boxes are proposed around
    its binary64 candidate. At most ``max_attempts`` trials (1..24) adjust the
    widths and increase interval precision from ``precision`` (8..128 digits)
    up to 128. Every accepted box satisfies the independent native C1-domain,
    strict Krawczyk inclusion and contraction checks. Return a ``RootBoxReceipt``
    proving one unique nonsingular root in its declared box, with exact root
    enclosures. This proves nothing about other boxes or global completeness.

    ``backend="python"`` preserves the scaled QR/Newton candidate generator
    with default ``tol=1e-12`` and ``maxiter=200``. ``frankennumpy`` selects
    the pinned dense linear solver and checked symbolic Jacobian, defaulting
    to ``tol=1e-12``, ``maxiter=64`` (0..1024), and ``max_step_norm=1e6``.
    ``frankenscipy`` selects the pinned fsolve with its finite-difference
    Jacobian, at most 200 iterations and a residual tolerance of 1e-10. None
    selects backend defaults. SciPy refuses every non-None solver-control
    override, and Python refuses ``max_step_norm``. No backend silently falls
    back to another. Static certification eligibility is checked before any
    donor executes; all three proposals go through the same independent
    original-source and captured-assumption box checks.

    ``tol`` controls only candidate generation, never mathematical acceptance.
    ``RootSearchError`` distinguishes exhausted candidate and certificate
    searches; numerical convergence alone is never returned as certification.
    The bounded search may refuse even when a root exists. For a known exact
    box or roots outside binary64 range, use ``certify_root_box`` directly.

    ``budget`` optionally shares one EvaluationBudget across Newton work,
    exact witness proposals, interval evaluation and every certificate retry.
    Consumed work stays spent on rejection. Cancellation and resource errors
    propagate immediately without becoming failed-domain retries. Interval
    work includes conservative 64 MiB reservations per native checker call;
    automatic preconditioning uses a separate point-enclosure call, so callers
    planning several trials should supply sufficient cumulative memory fuel.
    Compilation and symbolic proof admission remain outside this ledger.
    The pinned donor kernels are synchronous: cancellation is cooperative
    between kernels and callbacks, with no forced wall-time guarantee.
    """
    from ..utilities._certified_numeric import root_search_options

    _check_evaluation_budget(budget)
    _candidate_options(backend, tol, maxiter, max_step_norm, allow_python=True)
    root_search_options(precision, max_attempts)
    if _is_vector(expr) or _is_vector(var):
        variables = _vector(var, "variables", 16)
        residuals = _vector(expr, "residuals", 16)
        if len(variables) != len(residuals):
            raise ValueError("nsolve_certified requires a square system")
        variables = [_require_symbol(value) for value in variables]
    else:
        variables = [_require_symbol(var)]
        residuals = [expr]
    if len(set(variables)) != len(variables):
        raise ValueError("nsolve_certified has duplicate typed variables")
    program = compile_residual_system([_equation_residual(value) for value in residuals], variables)
    return _solve_compiled_certified(
        program, initial, tol=tol, maxiter=maxiter, precision=precision, max_attempts=max_attempts,
        backend=backend, max_step_norm=max_step_norm, budget=budget,
    )


def nsolve(expr, var, initial, tol=1e-12, maxiter=200, *, dict=False, budget=None):
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

    An optional reusable ``EvaluationBudget`` limits cumulative native
    evaluations and logical dense-solve work across all iterations and rejected
    line-search trials. Exhaustion and cooperative cancellation raise distinct
    ``RuntimeError`` subclasses and are never retried as domain failures.
    With a budget, the scalar path uses the finite-real native compiler;
    unsupported native functions refuse explicitly. Compilation, certification
    and CPython allocation overhead are outside this evaluation ledger.
    """
    if type(dict) is not bool:
        raise TypeError("nsolve dict must be a built-in boolean")
    _check_evaluation_budget(budget)
    tolerance = _newton_options(tol, maxiter)

    if _is_vector(expr) or _is_vector(var):
        from ..matrices import Matrix

        variables, values = _solve_system(expr, var, initial, tolerance, maxiter, budget=budget)
        roots = [Float(value) for value in values]
        result = ([{variable: root for variable, root in zip(variables, roots)}]
                  if dict else Matrix(roots))
        if budget is not None:
            budget._checkpoint()
        return result

    var = _require_symbol(var)
    x_value = _finite_real(initial, "initial value")
    expr = _equation_residual(expr)

    def solution(value):
        root = Float(value)
        result = [{var: root}] if dict else root
        if budget is not None:
            budget._checkpoint()
        return result

    if budget is None:
        residual = lambdify(var, expr, modules="math")
        derivative = lambdify(var, diff(expr, var), modules="math")
    else:
        residual = compile_numeric(var, expr)
        derivative = compile_numeric(var, diff(expr, var))

    def evaluate(function, value, name):
        try:
            result = function(value) if budget is None else function(value, budget=budget)
            return _finite_real(result, name)
        except (OverflowError, ZeroDivisionError, TypeError) as error:
            raise ValueError(f"nsolve {name} left the finite real domain") from error

    f_value = evaluate(residual, x_value, "residual")
    # Multiple roots and roots with a singular derivative may already satisfy
    # the equation. Do not evaluate or divide by the derivative in that case.
    if f_value == 0.0:
        return solution(x_value)

    for _ in range(maxiter):
        if budget is not None:
            budget._charge(4)
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
