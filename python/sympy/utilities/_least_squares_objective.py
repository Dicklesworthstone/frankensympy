"""Preserve a compiled residual system's source in its exact squared objective."""

from ._certified_numeric import _native_number, _rational, _sequence
from .lambdify import _check_evaluation_budget
from .source_certification import _source_budget


def least_squares_objective(program, *, weights=None, budget=None):
    """Use captured native rows and declarations without rebuilding Python input."""
    from .objective import _ObjectiveCallable

    budget = _source_budget(budget)
    if weights is None:
        native = program.native_system.least_squares_objective(budget=budget)
    else:
        size = program.native_system.residual_count
        if size > 256:
            raise ValueError("weighted loss exceeds the 256-node objective limit")
        rows = _sequence(weights, size, "least-squares weights")
        values = []
        for item in rows:
            weight = _rational(item, budget)
            if weight < 0:
                raise ValueError("least-squares weights must be nonnegative")
            values.append(_native_number(weight, budget))
        native = program.native_system.least_squares_objective(weights=values, budget=budget)
    result = _ObjectiveCallable(
        program._layout, native, program.variables,
        program._has_inexact_source, program._assumptions,
    )
    _check_evaluation_budget(budget)
    return result
