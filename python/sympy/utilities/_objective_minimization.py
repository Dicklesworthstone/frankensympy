"""Bounded descent proposals followed by independent strict-minimum acceptance.

Only the immutable native stationary receipt can establish a local minimum.
Approximate value, gradient and convergence metadata are not proof inputs.
"""

from .lambdify import EvaluationBudget, _check_evaluation_budget
from .objective import MinimizationSearchError, StationarySearchError
from ._stationary_search import (
    _prepare_stationary_certification, _search_stationary_box,
)


def minimize_certified(program, initial, gtol, maxiter, max_step, precision, max_attempts,
                       *, budget=None):
    from ..solvers.numerical import _candidate_options

    _check_evaluation_budget(budget)
    gtol, maxiter, max_step = _candidate_options(
        "frankennumpy", 1e-10 if gtol is None else gtol, maxiter,
        max_step, allow_python=False,
    )
    # Reject static ineligibility before converting initial coordinates or
    # spending numerical work. This uses the captured source and assumptions.
    _prepare_stationary_certification(program, precision, max_attempts)
    if budget is None:
        # One default ledger covers both phases and every rejected trial.
        budget = EvaluationBudget()
    try:
        candidate = program.minimize(
            initial, gtol=gtol, maxiter=maxiter, max_step=max_step, budget=budget,
        )
    except (ValueError, OverflowError, ZeroDivisionError) as error:
        raise MinimizationSearchError(
            "candidate", f"minimization could not generate a candidate: {error}",
        ) from error
    _check_evaluation_budget(budget)
    try:
        result = _search_stationary_box(
            program, candidate.point, precision, max_attempts, budget=budget,
            required_kind="strict_minimum",
        )
    except StationarySearchError as error:
        raise MinimizationSearchError(
            error.stage, str(error), failures=error.failures,
        ) from error
    _check_evaluation_budget(budget)
    return result
