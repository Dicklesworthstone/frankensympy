"""Uniform original-source root claims over exact parameter boxes.

A midpoint Jacobian inverse only proposes a witness. The independent native
checker retains the full parameter intervals in both the centered residuals
and the unknown Jacobian; only its accepted receipt authorizes publication.
"""

from ..core import _native, _wrap
from ._certified_numeric import (
    _Receipt, _bounds, _bounded_inverse, _call_budgeted, _charge_exact_work,
    _native_box, _native_number, _prepare_box, _rational, _sequence,
)
from .lambdify import _check_evaluation_budget

__all__ = ["ParametricRootBoxReceipt"]


class ParametricRootBoxReceipt(_Receipt):
    """Exactly one nonsingular unknown root for every declared parameter value.

    root_enclosure contains the union of these parameter-dependent roots.
    This is a claim within unknown_domain, with no exact symbolic solution
    function, continuation-path claim, or assertion outside that domain.
    """
    __slots__ = ()

    def __init__(self, receipt):
        if hasattr(self, "_receipt"):
            raise AttributeError("accepted numerical receipts are immutable")
        if type(receipt) is not _native.ParametricRootBoxReceipt:
            raise TypeError("an accepted native parametric root-box receipt is required")
        object.__setattr__(self, "_receipt", receipt)

    @property
    def unknown_variables(self):
        return tuple(self._receipt.unknown_variables)

    @property
    def parameters(self):
        return tuple(self._receipt.parameters)

    @property
    def unknown_domain(self):
        return _bounds(self._receipt.unknown_domain)

    @property
    def parameter_domain(self):
        return _bounds(self._receipt.parameter_domain)

    @property
    def center(self):
        return tuple(_wrap(value) for value in self._receipt.center)

    @property
    def preconditioner(self):
        return tuple(tuple(_wrap(value) for value in row) for row in self._receipt.preconditioner)

    @property
    def root_enclosure(self):
        return _bounds(self._receipt.root_enclosure)

    @property
    def contraction_bound(self):
        return _wrap(self._receipt.contraction_bound)

    @property
    def claim(self):
        return ("for every parameter value in the declared parameter box, "
                "one unique nonsingular real root in the declared unknown box")

    def __repr__(self):
        return (f"ParametricRootBoxReceipt(root_enclosure={self.root_enclosure!r}, "
                f"unknown_domain={self.unknown_domain!r}, "
                f"parameter_domain={self.parameter_domain!r}, claim={self.claim!r})")


def _fallback_budget():
    """One private ledger, using the native certification fallback envelope."""
    limit = 512 * 1024 * 1024
    return _native.EvaluationBudget(
        max_steps=limit, max_memory_bytes=limit, max_allocations=limit,
    )


def certify_parametric_root_box(
        program, unknown_box, parameter_box, center, preconditioner, precision, *, budget=None):
    """Prepare a bounded proposal, then independently check the entire family."""
    _check_evaluation_budget(budget)
    if budget is None:
        # Proposal and verification must not receive separate default budgets.
        # This ledger also covers exact preparation and final publication.
        budget = _fallback_budget()
    _check_evaluation_budget(budget)
    dimension = program.native_system.residual_count
    total = program.native_system.variable_count
    if not 1 <= dimension < total <= 16:
        raise ValueError(
            "parametric roots require one unknown per residual, at least one parameter, "
            "and at most 16 total variables"
        )
    unknown_box = _sequence(unknown_box, dimension, "unknown box")
    parameter_box = _sequence(parameter_box, total - dimension, "parameter box")
    product = _prepare_box(
        program, unknown_box + parameter_box, precision, square=False, budget=budget,
    )
    unknown_box, parameter_box = product[:dimension], product[dimension:]
    if any(lo == hi for lo, hi in unknown_box):
        raise ValueError("parametric unknown coordinates require positive width")

    if center is None:
        _charge_exact_work(
            budget, *(value for pair in unknown_box for value in pair), operations=2 * dimension,
        )
        center = tuple(_rational((lo + hi) / 2, budget) for lo, hi in unknown_box)
    else:
        center = tuple(_rational(value, budget) for value in _sequence(center, dimension, "center"))
    _charge_exact_work(
        budget, *center, *(value for pair in unknown_box for value in pair), operations=2 * dimension,
    )
    if any(not lo <= value <= hi for (lo, hi), value in zip(unknown_box, center)):
        raise ValueError("parametric root center must lie inside the declared unknown box")

    if preconditioner is None:
        _charge_exact_work(
            budget, *(value for pair in parameter_box for value in pair),
            operations=2 * len(parameter_box),
        )
        parameter_midpoints = tuple(_rational((lo + hi) / 2, budget) for lo, hi in parameter_box)
        # A parameter midpoint is used only to propose C. The uniform check
        # below receives the full original parameter box, never this point.
        point_box = tuple((value, value) for value in center + parameter_midpoints)
        point = _call_budgeted(
            program.native_system.enclose, _native_box(point_box, budget), precision, budget=budget,
        )
        if type(point) is not _native.ResidualJacobianEnclosure:
            raise TypeError("an accepted native residual enclosure is required for the proposal")
        matrix = []
        for row in point.jacobian:
            values = []
            # Parameter derivative columns do not belong to the square map on X.
            for lo, hi in row[:dimension]:
                lo, hi = _rational(_wrap(lo), budget), _rational(_wrap(hi), budget)
                _charge_exact_work(budget, lo, hi, operations=2)
                values.append((lo + hi) / 2)
            matrix.append(tuple(values))
        preconditioner = _bounded_inverse(tuple(matrix), budget)
    else:
        preconditioner = tuple(
            tuple(_rational(value, budget) for value in _sequence(row, dimension, "preconditioner row"))
            for row in _sequence(preconditioner, dimension, "preconditioner")
        )
    receipt = _call_budgeted(
        program.native_system.certify_parametric_root_box,
        _native_box(unknown_box, budget), _native_box(parameter_box, budget),
        [_native_number(value, budget) for value in center],
        [[_native_number(value, budget) for value in row] for row in preconditioner],
        precision, budget=budget,
    )
    result = ParametricRootBoxReceipt(receipt)
    _check_evaluation_budget(budget)
    return result
