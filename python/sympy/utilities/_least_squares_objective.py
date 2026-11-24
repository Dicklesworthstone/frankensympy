"""Preserve a compiled residual system's source in its exact squared objective."""

from .lambdify import _check_evaluation_budget
from .source_certification import _source_budget


def least_squares_objective(program, *, budget=None):
    """Use captured native rows and declarations without rebuilding Python input."""
    from .objective import _ObjectiveCallable

    budget = _source_budget(budget)
    native = program.native_system.least_squares_objective(budget=budget)
    result = _ObjectiveCallable(
        program._layout, native, program.variables,
        program._has_inexact_source, program._assumptions,
    )
    _check_evaluation_budget(budget)
    return result
